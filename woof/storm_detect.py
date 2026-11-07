"""Table-driven GPU medicane and polar-low tracks from completed model fields."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from woof.tc_tracks import binary, dump, normalize_manifest, run, sha, stamp, utc, write_csv

TABLE = Path(__file__).with_name('storm_detect_rows.json')


def load_rows(path=TABLE):
    table=json.loads(Path(path).read_text())
    if table.get('schema')!='gpuwm.storm-detect-rows.v1':raise ValueError('invalid storm detection table schema')
    ids=[r['id'] for r in table['basins']]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('basin ids must be nonempty and unique')
    for row in table['basins']:
        if row['kind'] not in ('medicane','polar-low'):raise ValueError('unknown detection kind')
        if row['min_duration_hours']<0 or row['max_speed_ms']<=0 or row['max_gap_hours']<=0:raise ValueError('invalid time or motion thresholds')
        s,n,w,e=row['bounds']
        if not (-90<s<n<90 and -180<=w<e<=180):raise ValueError('invalid basin bounds')
        for vocabulary in table['vocabularies'].values():
            if not set(row['fields'])<=vocabulary['fields'].keys():raise ValueError('vocabulary misses required fields')
    return table


def eligible(track,row):
    """Require a continuous classified episode, rather than distant snapshots."""
    start=previous=None
    longest=0.
    for fix in track['fixes']:
        now=utc(fix['valid'])
        if not fix['classified']:
            start=previous=None
            continue
        if start is None or previous is None or (now-previous).total_seconds()>row['max_gap_hours']*3600:start=now
        previous=now
        longest=max(longest,(now-start).total_seconds()/3600)
    return longest>=row['min_duration_hours'] and any(f['classified'] for f in track['fixes']),longest


def verify_reference(tracks,path,licence,max_distance_km=150):
    """Match one whole reference trajectory, never switch identities per time."""
    if not licence:raise ValueError('--reference requires --reference-license naming the source licence')
    ref=json.loads(Path(path).read_text())
    if not ref.get('source_url') or ref.get('license')!=licence:raise ValueError('reference needs source_url and matching license metadata')
    from woof.tc_tracks import distance
    if not math.isfinite(max_distance_km) or max_distance_km<=0:raise ValueError('reference distance gate must be positive')
    fixes=ref['fixes']
    if len({f['valid'] for f in fixes})!=len(fixes):raise ValueError('duplicate reference valid time')
    comparisons=[]
    for track in tracks:
        bytime={f['valid']:f for f in track['fixes']}
        pairs=[dict(valid=f['valid'],distance_km=distance(bytime[f['valid']],f),classified=bytime[f['valid']]['classified']) for f in fixes if f['valid'] in bytime]
        if pairs:
            matched=[p for p in pairs if p['distance_km']<=max_distance_km]
            comparisons.append(dict(track_id=track['track_id'],qualified=track.get('qualified',False),matched=len(matched),classified_matches=sum(p['classified'] for p in matched),time_overlap=len(pairs),reference_count=len(fixes),mean_distance_km=sum(f['distance_km'] for f in pairs)/len(pairs),max_distance_km=max(f['distance_km'] for f in pairs),pairs=pairs))
    comparisons.sort(key=lambda x:(-x['matched'],x['mean_distance_km']))
    return dict(source_url=ref['source_url'],license=licence,max_distance_km=max_distance_km,reference_sha256=sha(path),reference=ref,comparisons=comparisons)


def render_tracks(document,out):
    from woof.rustwx import find_renderer,renderer_env
    renderer=find_renderer()
    if renderer is None:raise ValueError('rw_wrfbatch is required to render storm tracks')
    target=out/'d01'/'storm-tracks'/document['members'][0]['init'][:10]/'tracks.png'
    target.parent.mkdir(parents=True,exist_ok=True)
    run([renderer,'--tc-track-plume',out/'qualified-tracks.json',target],env=renderer_env())
    return str(target)


def execute(args):
    begin=time.monotonic();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    table=load_rows(args.rows);row=next((r for r in table['basins'] if r['id']==args.basin),None)
    if row is None:raise ValueError('unknown basin; available: '+', '.join(r['id'] for r in table['basins']))
    manifest=normalize_manifest(json.loads(Path(args.manifest).read_text()),Path(args.manifest).resolve().parent)
    if args.workers<1:raise ValueError('--workers must be positive')
    receipt=dict(schema='gpuwm.storm-detect-receipt.v1',command=[sys.executable,'-m','woof',*sys.argv[1:]],basin=row,table_sha256=sha(args.rows),stages=[],peak_vram_mib=0)
    work=Path(tempfile.mkdtemp(prefix='.storm-work-',dir=out));deleted=[];stop=threading.Event();peak=[0]
    def monitor():
        while not stop.is_set():
            result=subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            for line in result.stdout.splitlines():
                parts=line.split(',')
                if len(parts)==2 and parts[0].strip()==str(os.getpid()):
                    try:peak[0]=max(peak[0],int(parts[1]))
                    except ValueError:pass
            stop.wait(.2)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        requests=[];members=[]
        for mi,member in enumerate(manifest['members']):
            frames=[]
            for fi,frame in enumerate(member['frames']):
                directory=work/f'm{mi:04d}-f{fi:05d}'
                vocabulary=table['vocabularies'][frame.get('vocabulary',member.get('vocabulary','earth2studio'))]
                req=work/f'm{mi:04d}-f{fi:05d}.json'
                dump(req,dict(path=frame['path'],out=str(directory),select=frame.get('select',{}),vocabulary=vocabulary,fields=row['fields']))
                requests.append(req);frames.append(dict(frame,directory=str(directory)))
            members.append((member,frames))
        start=time.monotonic();slots=min(args.workers,len(requests));threads=min(24,max(1,(os.cpu_count() or 1)//slots))
        def extract(req):
            with req.with_suffix('.log').open('w') as log:run([binary('rw_tc_fields'),'extract',req],env=dict(os.environ,RAYON_NUM_THREADS=str(threads)),stdout=log,stderr=subprocess.STDOUT)
        with ThreadPoolExecutor(max_workers=slots) as pool:list(pool.map(extract,requests))
        receipt['stages'].append(dict(stage='native_extract',seconds=time.monotonic()-start,parallelism=f'{slots} Rust processes x {threads} Rayon threads'))
        from woof.storm_detect_gpu import Detector,associate
        import cupy as cp
        tracks=[]
        for member,frames in members:
            local=[];grid=None;detector=None
            for frame in frames:
                start=time.monotonic();current=json.loads((Path(frame['directory'])/'grid.json').read_text())
                if grid!=current:grid=current;detector=Detector(grid,row)
                data=detector.read(frame['directory']);indices=detector.candidates(data)
                positions=[dict(lat=grid['lat'][y],lon=(grid['lon'][x]+180)%360-180) for y,x in indices]
                links=associate(detector,local,positions,frame['valid'],row['max_speed_ms'],row['max_gap_hours'])
                targets=[];motions=[]
                for j,(y,x) in enumerate(indices):
                    track=links.get(j)
                    motion=None
                    if track is None:
                        track=dict(track_id=f'{member["member_id"]}-{len(local):04d}',member_id=member['member_id'],weight=member['weight'],fixes=[]);local.append(track)
                    elif track['fixes']:
                        previous=track['fixes'][-1];position=positions[j]
                        motion=(((position['lon']-previous['lon']+180)%360-180)*math.cos(math.radians(position['lat'])),position['lat']-previous['lat'])
                    targets.append(track);motions.append(motion)
                diagnostics=detector.diagnostics(data,indices,motions)
                for track,fix in zip(targets,diagnostics):
                    fix.update(valid=frame['valid'],lead_hours=(utc(frame['valid'])-utc(member['init'])).total_seconds()/3600,flags=[] if fix['classified'] else ['unclassified'])
                    track['fixes'].append(fix)
                cp.cuda.get_current_stream().synchronize()
                peak[0]=max(peak[0],round(cp.get_default_memory_pool().total_bytes()/1048576))
                receipt['stages'].append(dict(stage='gpu_detect_diagnose_associate',valid=frame['valid'],seconds=time.monotonic()-start,parallelism='CUDA field kernels and GPU pair distances; bounded metadata orchestration',grid_shape=grid['shape'],candidates=len(indices)))
                print(frame['valid'], 'candidates',len(indices),'classified',sum(t['fixes'][-1]['classified'] for t in local if t['fixes'][-1]['valid']==frame['valid']),flush=True)
                del data
            for track in local:track['qualified'],track['classified_duration_hours']=eligible(track,row)
            tracks.extend(local)
        receipt['peak_vram_mib']=peak[0]
        start=time.monotonic();paths=sorted({f['path'] for m in manifest['members'] for f in m['frames']})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:hashes=dict(zip(paths,pool.map(sha,paths)))
        receipt['stages'].append(dict(stage='input_hash',seconds=time.monotonic()-start,parallelism=f'{min(args.workers,len(paths))} workers'))
        revision=run(['git','rev-parse','HEAD'],capture_output=True).stdout.strip()
        document=dict(schema='gpuwm.tc-tracks.v1',tracker=dict(backend='gpu-basin-physical',kind=row['kind'],config=row,code_revision=revision,table_sha256=sha(args.rows)),members=[{**{k:m[k] for k in ('member_id','family','source','init','weight','ancestry')},'inputs':[dict(path=p,sha256=hashes[p]) for p in sorted({f['path'] for f in m['frames']})]} for m in manifest['members']],tracks=tracks)
        document['plot']=dict(title=f"{row['kind'].capitalize()} candidate tracks",subtitle_left=f"Candidate fixes | tracks with >= {row['min_duration_hours']:g} h qualifying episode",subtitle_right='Classification and physical metrics in tracks.json',reference_label='Published reference')
        if args.reference:
            document['verification']=verify_reference(tracks,args.reference,args.reference_license,row['reference_match_radius_km'])
            document['best_track']=dict(source_url=document['verification']['source_url'],license=args.reference_license,fixes=[dict(f,mslp_pa=f.get('mslp_pa'),max_wind10_ms=f.get('max_wind10_ms'),flags=f.get('flags',[])) for f in document['verification']['reference']['fixes']])
        dump(out/'tracks.json',document);write_csv(out/'tracks.csv',tracks)
        dump(out/'qualified-tracks.json',dict(document,tracks=[t for t in tracks if t['qualified']]))
        if args.render:
            start=time.monotonic();receipt['pictures']=[render_tracks(document,out)]
            receipt['stages'].append(dict(stage='rust_render',seconds=time.monotonic()-start,parallelism='Rayon renderer through woof.rustwx'))
        receipt.update(code_revision=revision,code_sha256={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('storm_detect_gpu.py'))},input_sha256=hashes,track_count=len(tracks),qualified_track_count=sum(t['qualified'] for t in tracks),classified_fixes=sum(f['classified'] for t in tracks for f in t['fixes']),total_seconds=time.monotonic()-begin)
        return 0
    except Exception as error:
        receipt.update(error=str(error),total_seconds=time.monotonic()-begin)
        receipt['native_error_logs']={str(p):p.read_text()[-6000:] for p in work.glob('*.log') if p.stat().st_size}
        raise
    finally:
        stop.set();watcher.join(timeout=5)
        # Only this invocation's staging, never the supplied model output.
        for p in sorted(work.rglob('*')):
            if p.is_file() and p.suffix in ('.f32','.u8'):
                deleted.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
        dump(out/'deleted-files.json',deleted)
        dump(out/'receipt.json',receipt)


def register_cli(sub):
    p=sub.add_parser('storm-detect',help='GPU medicane and polar-low detection from completed model fields')
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--basin',required=True,help='basin id in the threshold table')
    p.add_argument('--rows',type=Path,default=TABLE)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--render',action='store_true')
    p.add_argument('--reference',type=Path)
    p.add_argument('--reference-license',help='explicit research opt-in for the reference data licence')
    p.set_defaults(func=main)


def main(args):
    try:return execute(args)
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print(f'woof storm-detect: {error}',file=sys.stderr);return 2


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(required=True);register_cli(sub)
    args=parser.parse_args();raise SystemExit(args.func(args))
