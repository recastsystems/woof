"""Compare every formal output word of real native NOAHMP_SFLX captures."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import argparse, os
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--metadata', required=True, type=Path)
parser.add_argument('--captures', required=True, type=Path)
parser.add_argument('--namelist', required=True, type=Path)
parser.add_argument('--receipt', required=True, type=Path)
parser.add_argument('--tanh-profile', choices=('glibc_2_39','glibc_2_43'), default='glibc_2_39')
parser.add_argument('--atan-profile', choices=('glibc_2_39','glibc_2_43'), default='glibc_2_39')
parser.add_argument("--log10-profile",choices=("glibc_2_39","glibc_2_43"),default="glibc_2_39")
cli = parser.parse_args()
if os.environ.get('GPUWM_WRF_EXACT') != '1':
    parser.error('GPUWM_WRF_EXACT=1 is required for the native replay')
import cupy as cp
from woof.core.noahmp_column_slab import evaluate_sflx_slab,parameter_fields,slab_parameter_widths,PARAMETER_INTS
meta=json.loads(cli.metadata.read_text())
pin=json.loads(Path(__file__).with_name('source_pin.json').read_text())
lsm=next(item for item in pin['files'] if item['upstream_path']=='src/module_sf_noahmplsm.F')
assert meta['original_sha256']==lsm['sha256'], 'capture source differs from native pin'
assert meta['native_pin']==pin['wrf_noahmp_pin'], 'capture commit differs from native pin'
args={x['name']:x for x in meta['args']}
pars={x['name']:x for x in meta['parameters']}
paths=sorted(cli.captures.glob('noahmp-real-*.csv'))
assert paths,'No real native captures'
columns=[]
for path in paths:
    stages={'input':{},'output':{}}
    for row in csv.DictReader(path.open()):
        name=row['field']; stage=row['stage']; key=(name,int(row['index']))
        if name=='llanduse':continue
        spec=pars.get(name[4:]) if name.startswith('par_') else args.get(name)
        kind=spec['kind'] if spec else 'integer'
        dtype=np.int32 if kind in ('integer','logical') else np.float32
        value=np.array([int(row['hex'],16)],np.uint32).view(dtype)[0]
        stages[stage][key]=value
    columns.append(stages)
options={name:int(columns[0]['input'][(name,0)]) for name in ('opt_btr','opt_rsf','opt_alb','opt_stc')}
assert all(all(int(col['input'][(k,0)])==v for k,v in options.items()) for col in columns)
dveg=int(columns[0]['input'][('dveg',0)])
assert dveg in (1,2,3,4,5,6,7,8,9)
if dveg!=4: options['dveg']=dveg
# Every captured branch selector must agree with the imported stock namelist.
# Static irrigation geography remains valid when irrigation is disabled.
for col in columns:
    values=col['input']
    for name,expected in dict(dveg=dveg,soil_update_steps=int(columns[0]['input'][('soil_update_steps',0)])).items():
        assert int(values[(name,0)])==expected, (name,values[(name,0)])

from woof.namelist_import import read_namelist_role
namelist=read_namelist_role(cli.namelist,'native replay namelist')
settings=namelist['noah_mp']
soil_interval=float(settings.get('soiltstep',[0.0])[0])
cadence_kwargs={}
if soil_interval>0:
    from woof.core.noahmp_soil_cadence import cadence
    steps=int(columns[0]['input'][('soil_update_steps',0)])
    calc=bool(columns[0]['input'][('calculate_soil',0)])
    assert all(int(c['input'][('soil_update_steps',0)])==cadence(soil_interval,float(c['input'][('dt',0)]),0)[0] and bool(c['input'][('calculate_soil',0)])==calc for c in columns)
    cadence_kwargs=dict(soil_update_steps=steps,calculate_soil=calc,carry_soil_accumulators=True)
else:
    assert all(int(c['input'][('soil_update_steps',0)])==1 for c in columns)

for name,default in (('opt_snf',1),('opt_inf',1),('opt_run',3),('opt_rad',3),('opt_frz',1),('opt_tbot',2),('opt_crs',1),('opt_sfc',1),('opt_soil',1),('opt_infdv',0),('opt_tdrn',0),('opt_irr',0),('opt_irrm',0),('opt_crop',0)):
    selected=settings.get(name,[default])
    assert len(set(selected))==1
    if int(selected[0])!=default:options[name]=int(selected[0])
    for col in columns:
        if (name,0) in col['input']:
            assert int(col['input'][(name,0)])==int(selected[0]),name
from woof.config import NOAHMP_OPTION_IDENTITY
for name,default in NOAHMP_OPTION_IDENTITY.items():
    expected=soil_interval if name=='soiltstep' else options.get(name,default)
    actual=settings.get(name,[default])
    assert all(value==expected for value in actual),(name,actual,expected)

fields={}
for name,spec in args.items():
    if spec['intent'] not in ('IN','INOUT') or name=='llanduse':continue
    keys=sorted(k for k in columns[0]['input'] if k[0]==name)
    vals=[[col['input'][k] for k in keys] for col in columns]
    a=np.array(vals)
    if spec['bounds'] is None:a=a[:,0]
    fields[name]=cp.asarray(a)
rows=[]
widths=slab_parameter_widths(opt_alb=options['opt_alb'],opt_rsf=options['opt_rsf'],opt_run=options.get('opt_run',3),opt_rad=options.get('opt_rad',3),opt_crs=options.get('opt_crs',1),opt_sfc=options.get('opt_sfc',1),dveg=dveg,opt_tdrn=options.get("opt_tdrn",0),opt_irr=options.get("opt_irr",0),opt_crop=options.get("opt_crop",0))
for col in columns:
    row={}
    for name,width in {**widths,**{k:1 for k in PARAMETER_INTS}}.items():
        n='par_'+name.lower()
        keys=sorted(k for k in col['input'] if k[0]==n)
        vals=[col['input'][k] for k in keys]
        if width==1:row[name]=vals[0]
        else:row[name]=np.asarray(vals,np.float32)
    rows.append(row)
fields.update(parameter_fields(rows,np.arange(len(rows)),opt_alb=options['opt_alb'],opt_rsf=options['opt_rsf'],opt_run=options.get('opt_run',3),opt_rad=options.get('opt_rad',3),opt_crs=options.get('opt_crs',1),opt_sfc=options.get('opt_sfc',1),dveg=dveg,opt_tdrn=options.get("opt_tdrn",0),opt_irr=options.get("opt_irr",0),opt_crop=options.get("opt_crop",0)))
if options.get('opt_irr',0):
    fields['croplu']=cp.asarray([int(str(next(r['hex'] for r in csv.DictReader(p.open()) if r['field']=='llanduse')).strip()=='MODIFIED_IGBP_MODIS_NOAH' and int(col['input'][('vegtyp',0)]) in (12,14) or str(next(r['hex'] for r in csv.DictReader(p.open()) if r['field']=='llanduse')).strip()=='USGS' and 3<=int(col['input'][('vegtyp',0)])<=6) for p,col in zip(paths,columns)],dtype=cp.int32)
    for name in ('pltday','hsday'):
        fields[name]=cp.asarray([col['input'][('par_'+name,0)]for col in columns],dtype=cp.float32)
if options.get('opt_crop',0):
    assert options['opt_crop']==1, 'GECROS full-module binding not yet implemented'
    from woof.core.noahmp_crop_slab import SCALARS,STAGES
    fields['crop_pars']=cp.asarray(np.array([[col['input'][('par_'+name,0)]for name in SCALARS]+[col['input'][('par_'+name,k)]for name in STAGES for k in range(1,9)]for col in columns],np.float32))
else:
    assert all(col['input'][('croptype',0)]==0 for col in columns)
trace={}
result=evaluate_sflx_slab(fields,len(columns),noahmp_snow_age='wrf_461',module_trace=trace,tanh_profile=cli.tanh_profile,log10_profile=cli.log10_profile,atan_profile=cli.atan_profile,**options,**cadence_kwargs)
# Bind the native formal ABI to results already produced on the device.
# These INOUT fields are untouched for RUN3/4/6/7, prescribed DVEG, CROP0/IRR0; native source
# anchors below witness the disabled branches rather than supplying outputs.
pass_through={
    'zlvl':'ENERGY receives INTENT(IN) ZREF, leaving formal ZLVL unchanged: 952, 1743, 1840, 2108',
    'zwt':'SOILWATER run3/4/6/7 and WATER run1/run5 guards: 6217-6262, 7364-7412',
    'wa':'WATER groundwater guards: 6225-6230, 6244-6253',
    'wt':'WATER groundwater guards: 6225-6230',
    'smcwtd':'SOILWATER run5 guard: 7428-7433',
    'deeprech':'SOILWATER run5 guard: 7428-7433',
    'rech':'WATER run5 guard: 6244-6253',
    'gecros1d':'VEGE_FLUX opt_crop==2 guard: 3957-4005',
    **{name:'NOAHMP_SFLX prescribed DVEG/CROP0 disables carbon: 1016-1038'
       for name in ('lfmass','rtmass','stmass','wood','stblcp','fastcp','grain','gdd','pgs')},
    **{name:'NOAHMP_SFLX inactive irrigation: 878-936; WATER: 6187-6201'
       for name in ('ircntsi','ircntmi','ircntfi','irsirate','irmirate','irfirate','firr','eirr')},
}
if options.get("opt_run",3) in (1,2) or options.get("opt_tdrn",0):
    del pass_through["zwt"]
    if options.get("opt_run",3)==1:
        del pass_through["wa"];del pass_through["wt"]
if options.get("opt_run",3)==5:
    for name in ("zwt","wa","smcwtd","deeprech","rech"):del pass_through[name]
if dveg in (2,5,6) or options.get('opt_crop',0):
    from woof.core.noahmp_carbon_slab import CARBON_POOLS
    for name in CARBON_POOLS:del pass_through[name]
    if options.get("opt_crop",0):
        for name in ("grain","gdd","pgs"):del pass_through[name]
if options.get('opt_irr',0):
    for name in ('ircntsi','ircntmi','ircntfi','irsirate','irmirate','irfirate','firr','eirr'):
        del pass_through[name]
module={name:fields[name] for name in pass_through}
# The three amounts are reset to zero under IRRFRA < IRR_FRAC (919-923).
# The captured inputs satisfy the disabled irrigation contract; the explicit
# assertion prevents a future active capture being accepted as a pass-through.
for name in (() if options.get('opt_irr',0) else ('iramtsi','iramtmi','iramtfi')):
    assert not cp.asnumpy(fields[name]).view(np.uint32).any(), name
    module[name]=fields[name]
    pass_through[name]='native IRRFRA=0 resets amounts to +0: 919-923'
for stage in ('pre','geometry','radiation','vegetation','bare','energy3','energy4','phasechange','post1','water','post2','carbon'):
    for name,value in trace[stage].items():
        if name in args and name not in pass_through: module[name]=value
for name,key in (('chleaf','CHLEAF'),('chuc','CHUC'),('chv2','CAH2')):
    module[name]=trace['vegetation_raw'][key]
module['chb2']=trace['bare']['ehb2']
module.update({name:value for name,value in result.items() if name in args})
required={name for name,spec in args.items() if spec['intent'] in ('OUT','INOUT')}
assert required==set(module),(sorted(required-set(module)),sorted(set(module)-required))
differences=[]; witnesses=[]; words=0; field_counts={}
for name,device in module.items():
    actual=cp.asnumpy(device)
    for col,stages in enumerate(columns):
        keys=sorted(k for k in stages['output'] if k[0]==name)
        assert keys,name
        values=np.asarray(actual[col]).reshape(-1)
        assert len(keys)==len(values),(name,keys,values)
        for key,value in zip(keys,values):
            native=stages['output'][key]
            got=np.asarray(value).view(np.uint32).item();want=np.asarray(native).view(np.uint32).item()
            words+=1
            field_counts[name]=field_counts.get(name,0)+1
            if got!=want:
                differences.append(dict(column=col,field=name,index=key[1],got=f'{got:08x}',native=f'{want:08x}',value=float(value),reference=float(native)))
report={'origin':'real native WRF v4.6.1 NOAHMP_SFLX capture; case identity and inputs are separately hashed',
    'active_crop_columns':sum(int(c['input'][('croptype',0)])>0 for c in columns),'strict':True,'libm_profiles':dict(tanh=cli.tanh_profile,atan=cli.atan_profile,log10=cli.log10_profile),'options':options,'columns':len(columns),
    'soil_cadence':dict(soiltstep=soil_interval,**cadence_kwargs),'formal_output_fields':len(required),'compared_fields':len(field_counts),
    'compared_words':words,'differing_words':len(differences),'differences':differences,
    'native_undefined':witnesses,'pass_through_source_witnesses':pass_through,
    'namelist_sha256':hashlib.sha256(cli.namelist.read_bytes()).hexdigest(),
    'field_word_counts':field_counts,
    'captures':[dict(name=p.name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]}
cli.receipt.parent.mkdir(parents=True,exist_ok=True)
cli.receipt.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('differences','captures','native_undefined','field_word_counts','pass_through_source_witnesses')},indent=2))
print('First differences',json.dumps(differences[:20],indent=2))
raise SystemExit(bool(differences))
