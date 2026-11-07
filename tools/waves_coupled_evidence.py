"""Orchestrate native history reductions, shared-scale maps and track scores."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def score_tracks(output, members, tracks, best, binary=None):
    """Retain the exact native score request and executable identity for paper evidence."""
    from woof import tc_tracks
    from woof.cyclone_tracks import native_track_binary
    binary=Path(binary or native_track_binary('cyclone-score')).resolve()
    limit=json.loads(tc_tracks.TABLE.read_text())['verification']['association_max_distance_km']
    request=output/'track-score-request.json'
    write(request,dict(members=members,tracks=tracks,best=best,association_max_distance_km=limit))
    command=[str(binary),'cyclone-score','--request',str(request)]
    with binary.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    write(output/'track-score-command.json',dict(command=command,binary_sha256=digest))
    result=subprocess.run(command,text=True,capture_output=True,check=True,timeout=120)
    (output/'track-score-stderr.log').write_text(result.stderr)
    scores=json.loads(result.stdout);write(output/'track-scores.json',scores)
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    args=parser.parse_args();request=json.loads(args.request.read_text())
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
    pictures=Path(request['pictures']);pictures.mkdir(parents=True,exist_ok=True)
    members=[];tracks=[];reductions=[]
    for arm in request['arms']:
        frame_paths={}
        for root in arm['roots']:
            for path in sorted(Path(root).rglob('wrfout*')):
                match=re.fullmatch(r'wrfout_d01_(\d{4}-\d\d-\d\d_\d\d[:_]\d\d[:_]\d\d)(?:\.nc)?',path.name)
                if path.is_file() and match:
                    valid=match[1][:10]+'T'+match[1][11:].replace('_',':')+'Z'
                    frame_paths.setdefault(valid,path.resolve())
        if len(frame_paths)!=request['expected_frames']:
            raise ValueError(f"{arm['label']}: expected {request['expected_frames']} frames, got {len(frame_paths)}")
        times=sorted(frame_paths);start=datetime.fromisoformat(times[0].replace('Z','+00:00'))
        frames=[dict(path=str(frame_paths[t]),valid=t,reduce=(int((datetime.fromisoformat(t.replace('Z','+00:00'))-start).total_seconds())%21600==0 or t==times[-1])) for t in times]
        folder=output/arm['label'];receipt=folder/'receipt.json'
        native=dict(output=str(folder),frames=frames,threads=24,initial_center=request['initial_center'],search_radius_km=500,wind_radius_km=500,edge_cells=5)
        native_path=output/(arm['label']+'-request.json');write(native_path,native)
        if not receipt.exists():
            with (output/(arm['label']+'-reduce.log')).open('w') as log:
                subprocess.run([request['binary'],str(native_path)],check=True,timeout=3600,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(receipt.read_text());reductions.append((arm,result))
        fixes=[dict(row['fix'],valid=row['valid'],flags=[]) for row in result['frames']]
        members.append(dict(member_id=arm['label'],family='arwen',source='arwen',init=times[0],weight=1.,ancestry={},frames=[dict(valid=t) for t in times]))
        tracks.append(dict(member_id=arm['label'],track_id='pressure-minimum',weight=1.,fixes=fixes))
    # Shared fixed ranges make every arm and time directly comparable.
    rows={'products':{'wind10':{'title':'10 m wind','units':'m/s','range':[0,75]},'mslp':{'title':'Mean sea level pressure','units':'hPa','range':[900,1030]}}}
    rows_path=output/'atmosphere-rows.json';write(rows_path,rows)
    from woof import rustwx
    for arm,result in reductions:
        frames=[];first=None
        for row in result['frames']:
            if 'reduced' not in row:continue
            reduced=row['reduced'];first=first or reduced;directory=Path(reduced['directory'])
            frames.append(dict(valid_time=row['valid'],fields={k:str(directory/(k+'.f32')) for k in rows['products']}))
        directory=Path(first['directory'])
        render=dict(schema='gpuwm.ocean.products/v1',out_dir=str(pictures/arm['label']),domain='d01',family='atmosphere',source_label=arm['label'],nx=first['shape'][1],ny=first['shape'][0],latitude=str(directory/'latitude.f32'),longitude=str(directory/'longitude.f32'),rows=str(rows_path),frames=frames)
        render_path=output/(arm['label']+'-render.json');write(render_path,render)
        if not Path(render['out_dir']).exists():
            with (output/(arm['label']+'-render.log')).open('w') as log:
                subprocess.run([str(rustwx.find_renderer()),'--ocean-products',str(render_path)],check=True,timeout=1800,env=rustwx.renderer_env(),stdout=log,stderr=subprocess.STDOUT)
    document=dict(schema='gpuwm.tc-tracks.v1',tracker=dict(backend='declared-radius-grid-pressure-minimum'),members=members,tracks=tracks)
    from woof import tc_tracks
    if request.get('best_track'):
        document['best_track']=tc_tracks.best_track(request['best_track'],request.get('storm_id'))
        document['scores']=score_tracks(output,members,tracks,document['best_track'],request.get('score_binary'))
    if document.get('best_track'):
        valid_times=[row['valid'] for track in tracks for row in track['fixes']]
        lo,hi=min(valid_times),max(valid_times)
        document['best_track']['fixes']=[row for row in document['best_track']['fixes'] if lo<=row['valid']<=hi]
        document['best_track']['display_window']={'start':lo,'end':hi}
    write(output/'tracks.json',document)
    subprocess.run([str(rustwx.find_renderer()),'--tc-track-plume',str(output/'tracks.json'),str(pictures/'track.png')],check=True,timeout=300,env=rustwx.renderer_env())
    tc_tracks.intensity_chart(document,pictures/'intensity.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for arm,result in reductions:
        for axis,lo in zip(axes.flat,range(30,60,5)):
            entries=[(row['valid'],b) for row in result['frames'][1:] for b in row['drag_bins'] if b['wind_low']==lo]
            times=[datetime.fromisoformat(t.replace('Z','+00:00')) for t,b in entries]
            axis.plot(times,[b['mean_cd'] if b['count'] else float('nan') for t,b in entries],'.-',label=arm['label'])
            axis.set_title(f'{lo}-{lo+5} m/s water cells');axis.set_ylabel('Effective drag coefficient');axis.set_xlabel('UTC');axis.grid(alpha=.25);axis.legend(fontsize=8);axis.tick_params(axis='x',rotation=25)
    fig.savefig(pictures/'actual-drag.png',dpi=140);plt.close(fig)
    write(output/'postprocess.json',dict(status='complete',request=request,tracks=str(output/'tracks.json'),pictures=str(pictures)))

if __name__=='__main__':main()
