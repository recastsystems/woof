"""Exercise CLI contracts that unit-level geometry does not cover. Synthetic, not skill evidence."""
import argparse,hashlib,json,pathlib,subprocess,tempfile
p=argparse.ArgumentParser(__doc__);p.add_argument('--binary',required=True);p.add_argument('--out',required=True);a=p.parse_args();binary=str(pathlib.Path(a.binary).resolve())
with tempfile.TemporaryDirectory(prefix='tc-verify-contract-',dir=pathlib.Path(a.out).resolve().parent)as tmp:
 d=pathlib.Path(tmp)
 def put(name,obj):
  (d/name).write_text(json.dumps(obj));return str(d/name)
 fixes=[dict(valid='2024-06-01T00:00:00Z',lat=20,lon=-60,vmax_kt=50,mslp_hpa=990,radii_nmi={}),dict(valid='2024-06-01T06:00:00Z',lat=21,lon=-60,vmax_kt=60,mslp_hpa=980)]
 put('fixes.json',{'fixes':fixes})
 subprocess.run([binary,'export','--input',str(d/'fixes.json'),'--storm','AL012024','--model','TEST','--init','2024060100','--out',str(d/'ours.dat')],check=True)
 lines=(d/'ours.dat').read_text().splitlines();assert len(lines)==2,'empty radii object must not drop a fix'
 (d/'best.dat').write_text('AL, 01, 2024060100, 00, BEST, 0, 200N, 0600W, 50, 990, TS\nAL, 01, 2024060106, 00, BEST, 0, 210N, 0600W, 60, 980, TS\n')
 (d/'ofcl.dat').write_text('AL, 01, 2024060100, 03, OFCL, 0, 200N, 0600W, 50, 0, TS\nAL, 01, 2024060100, 03, OFCL, 6, 210N, 0600W, 60, 0, TS\n')
 # An unrelated cycle can contain an archive conflict. It must not break this requested cycle.
 with (d/'ofcl.dat').open('a')as f:f.write('AL, 01, 2024053100, 03, OFCL, 0, 200N, 0600W, 50, 0, TS\nAL, 01, 2024053100, 03, OFCL, 0, 210N, 0600W, 50, 0, TS\n')
 put('rain.json',dict(nx=2,ny=2,dx_km=4,cell_area_km2=16,grid_id='synthetic',start_utc='2024060100',end_utc='2024060106',observation_source='synthetic contract test',observed=[100,0,0,None],forecasts={'TEST':[100,0,0,None]},thresholds_mm=[50],neighborhoods_km=[4]))
 put('station.json',[dict(station='synthetic',variable='surge',units='m',datum='synthetic',observation_source='synthetic contract test',times=['2024060100','2024060106'],observed=[0,1],forecasts={'TEST':[0,1]})])
 case=dict(storm='AL012024',best='best.dat',adecks=['ours.dat','ofcl.dat'],models=['TEST','OFCL'],cycles=['2024060100'],plots=False,metric_models={'mslp_hpa':['TEST'],'rain':['TEST'],'stations':['TEST']},rain='rain.json',stations='station.json')
 put('manifest.json',{'cases':[case]});before={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in d.iterdir()if p.is_file()}
 subprocess.run([binary,'score','--manifest',str(d/'manifest.json'),'--out',str(d/'output')],check=True)
 r=json.load(open(d/'output/AL012024.json'));prs=[s for s in r['homogeneous']if s['metric']=='mslp_hpa'];assert len(prs)==2 and all(s['cohort']==['TEST']and s['n']==1 and s['mae']==0 for s in prs)
 assert r['rain']['rows'][0]['fss']==1 and r['rain']['common_valid_cells']==3
 assert r['stations']['rows'][0]['rmse']==0
 assert all(hashlib.sha256((d/k).read_bytes()).hexdigest()==v for k,v in before.items())
 case['metric_models']={};case.pop('rain');case.pop('stations');put('manifest.json',{'cases':[case]})
 subprocess.run([binary,'score','--manifest',str(d/'manifest.json'),'--out',str(d/'without-override')],check=True)
 r=json.load(open(d/'without-override/AL012024.json'));assert not any(s['metric']=='mslp_hpa'for s in r['homogeneous']),'missing pressure must not silently shrink cohort'
 result={'passed':True,'synthetic_only':True,'checks':['empty-radius ATCF export retains fix','explicit pressure cohort has exact scores','absent pressure excludes full default cohort','rain and station independent cohorts','common rain missing mask','input hashes unchanged','unrequested conflicting cycle excluded']}
 pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
