import json,subprocess,pathlib,argparse
parser=argparse.ArgumentParser(__doc__)
parser.add_argument("--manifest",required=True)
parser.add_argument("--binary",required=True)
parser.add_argument("--out",required=True)
args=parser.parse_args()
base=pathlib.Path(args.manifest).resolve().parent
out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
source={c['storm']:c for c in json.load(open(args.manifest))['cases']}
for c in source.values():
 c['best']=str(base/pathlib.Path(c['best']));c['adecks']=[str(base/pathlib.Path(p))for p in c['adecks']]
track_models='OFCL OCD5 AVNI HWFI HMNI HFAI HFBI EMXI CMCI CTCI TVCA TVCX GFEX TVDG HCCA AEMI TABS TABM TABD'.split()
int_models='OFCL OCD5 HWFI HMNI HFAI HFBI DSHP LGEM ICON IVCN IVDR CTCI AVNI EMXI HCCA'.split()
expected=json.load(open(pathlib.Path(__file__).with_name('nhc_2024_tables.json')))['tables']
rows=[]
for storm,e in expected.items():
 for category,models,metric in [('track',track_models,'met_track_nmi'),('wind',int_models,'vmax_kt')]:
  models=models+(['UEMI'] if storm=='AL142024' and category=='track' else [])+(['EGRI','FSSE'] if storm=='AL092024' and category=='track' else ['FSSE'] if storm=='AL092024' else [])
  priority={'HFAI':['HFAI','HFA2'],'HWFI':['HWFI','HWF2'],'EMXI':['EMXI','EMX2'],'CMCI':['CMCI','CMC2'],'UEMI':['UEMI','UEM2'],'EGRI':['EGRI','EGR2']};c=dict(source[storm],cycles=[],plots=False,models=models,metric_models={},aid_priority={k:v for k,v in priority.items() if k in models});p=str(out/f'{storm}-{category}');json.dump({'cases':[c]},open(p+'.json','w'))
  subprocess.run([args.binary,'score','--manifest',p+'.json','--out',p],check=True)
  report=json.load(open(p+'/'+storm+'.json'));actual={(s['model'],s['lead']):s for s in report['homogeneous'] if s['metric']==metric}
  for model in ['OFCL','HFAI','HFBI']:
   for i,lead in enumerate(e['leads']):
    a=actual.get((model,lead),{});v={'storm':storm,'category':category,'model':model,'lead':lead,'expected_n':e[category]['n'][i],'n':a.get('n'),'expected_mae':e[category][model][i],'mae':a.get('mae')};v['pass']=v['n']==v['expected_n'] and v['mae'] is not None and abs(v['mae']-v['expected_mae'])<=0.051;rows.append(v)
print('total',len(rows),'passing',sum(r['pass']for r in rows));print(json.dumps([r for r in rows if not r['pass']][:10],indent=2))
json.dump({'tables':expected,'rows':rows,'passed':all(r['pass']for r in rows)},open(out/'nhc-comparison.json','w'),indent=2)

raise SystemExit(0 if all(r["pass"] for r in rows) else 1)
