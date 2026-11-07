"""Replay observation-only camrad captures from a complete native WRF forecast."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys
import numpy as np
from make_driver import arguments
from check_camrad_cpu import PARAMS

def read_capture(root):
    payload=(root/'columns.bin').read_bytes()
    result={}
    for row in csv.DictReader((root/'columns.tsv').open()):
        key=(int(row['column']),row['field'])
        if key in result:raise ValueError('duplicate capture '+str(key))
        result[key]=np.frombuffer(payload,'<f8',int(row['count']),int(row['offset_bytes'])).copy()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('capture',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--workers',type=int,default=96);p.add_argument('--staged',action='store_true',help='replay through the production ordered GPU schedule');a=p.parse_args()
    import cupy as cp
    from woof.core.kernels import get_kernel_int_defines,module_options
    data=read_capture(a.capture);columns=sorted({c for c,k in data});nz=int(data[(1,'input:kte')][0]-data[(1,'input:kts')][0]+1)
    _,decl=arguments(a.source.read_text(),'camrad')
    meta=json.loads((Path(__file__).resolve().parents[2]/'woof/core/kernels/cam3_emissivity.json').read_text())['camrad']
    normalize={key:value for key,value in PARAMS.items() if key in ('ids','ide','jds','jde','ims','ime','jms','jme','its','ite','jts','jte')}
    vals=[]
    for field in meta['parameters']:
        name=field['name'];dtype={'int':np.int32,'float':np.float32,'double':np.float64}[field['dtype']]
        value=np.stack([data[(c,'input:'+name)] for c in columns]).astype(dtype)
        if name in normalize:value.fill(normalize[name])
        count=eval(field.get('count_expression',str(field['count'])),{'__builtins__':{}},{'CAM_LEVELS':nz})
        if value.shape!=(len(columns),count):raise ValueError((name,value.shape,count))
        vals.append(cp.asarray(value))
    state=cp.asarray(np.frombuffer((a.capture/'state.bin').read_bytes(),np.uint8))
    work_bytes=eval(meta['workspace_expression'],{'__builtins__':{}},{'CAM_LEVELS':nz})
    from woof.core.cam3_memory import BLOCK_THREADS
    block=BLOCK_THREADS;workers=((a.workers+block-1)//block)*block;workspace=cp.empty((workers,work_bytes),cp.uint8)
    kernel=None if a.staged else get_kernel_int_defines('cam3_emissivity','cam_driver',(('CAM_LEVELS',nz),))
    start=cp.cuda.Event();end=cp.cuda.Event();start.record()
    if a.staged:
        from woof.core.cam3_launch import launch_columns
        refresh=np.array([bool(data[(c,'input:dolw')][0] and data[(c,'input:doabsems')][0]) for c in columns])
        solar=np.array([bool(data[(c,'input:dosw')][0]) for c in columns])
        from itertools import product
        thermal=np.array([bool(data[(c,'input:dolw')][0]) for c in columns])
        for enabled,shortwave,longwave in product((False,True),repeat=3):
            indices=cp.asarray(np.flatnonzero((refresh==enabled)&(solar==shortwave)&(thermal==longwave)))
            if not indices.size:continue
            group=[cp.ascontiguousarray(v[indices]) for v in vals]
            launch_columns(nz,state,group,workers,refresh=enabled,shortwave=shortwave,longwave=longwave)
            for original,returned in zip(vals,group):original[indices]=returned
    else:
        kernel((workers//block,),(block,),(np.int32(len(columns)),state,workspace,*vals))
    end.record();end.synchronize()
    rows=[];witnesses=[]
    for field,actual in zip(meta['parameters'],vals):
        name=field['name']
        if name=='gases' or not any(role in decl[name] for role in ('intent(out)','intent(inout)')):continue
        dtype=np.int32 if field['dtype']=='int' else np.float32
        expected=np.stack([data[(c,'output:'+name)] for c in columns]).astype(dtype)
        before=np.stack([data[(c,'input:'+name)] for c in columns]).astype(dtype)
        actual=cp.asnumpy(actual)
        # Every returned native slot is compared, including native-untouched slots.
        delta=expected.view(np.uint32)!=actual.view(np.uint32)
        undefined=np.zeros(delta.shape,bool)
        if name in ('taucldc','taucldi'):
            for i,c in enumerate(columns):
                if data[(c,'input:dosw')][0]==0: undefined[i,:nz]=True
        residues=int((delta&undefined).sum());delta &= ~undefined
        rows.append({'native_undefined_words':int(undefined.sum()),'native_undefined_differing_words':residues,'field':name,'words':expected.size,'differing_words':int(delta.sum()),'nonfinite_native':int((~np.isfinite(expected)).sum()),'max_abs':float(np.nanmax(np.abs(actual.astype(float)-expected.astype(float)))),'first_mismatches':np.argwhere(delta)[:8].tolist()})
        unchanged=expected.view(np.uint32)==before.view(np.uint32)
        witnesses.append({'field':name,'unchanged_from_entry_words':int(unchanged.sum())})
    result={'pass':not any(r['differing_words'] for r in rows),'words':sum(r['words'] for r in rows),'differing_words':sum(r['differing_words'] for r in rows),'columns':len(columns),'levels':nz,'radiation_call_times_minutes':sorted({float(data[(c,'input:xtime')][0]) for c in columns}),'workers':workers,'kernel_ms':cp.cuda.get_elapsed_time(start,end),'fields':rows,'native_untouched_witness':witnesses,'strict':os.environ.get('GPUWM_WRF_EXACT')=='1','options':__import__('woof.wrf_exact',fromlist=['effective_options']).effective_options(module_options('cam3_emissivity')),'compile_receipt':__import__('woof.wrf_exact',fromlist=['compile_receipt']).compile_receipt(),'native_undefined_witness':'WRF module_ra_cam.F camrad locals tauxcl/tauxci have no initialization; radctl assigns them only inside if(dosw), camrad copies them unconditionally. Mask only kts:kte in taucldc/taucldi when captured dosw is false; untouched upper slots remain compared.','capture_sha256':{n:hashlib.sha256((a.capture/n).read_bytes()).hexdigest() for n in ('columns.tsv','columns.bin','state.bin')},'kernel_attributes':kernel.attributes if kernel is not None else {'production_spectra_specialized':True}}
    result['production_schedule']=a.staged
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('fields','native_untouched_witness')}));print([r for r in rows if r['differing_words']][:8]);return not result['pass']
if __name__=='__main__':sys.exit(main())
