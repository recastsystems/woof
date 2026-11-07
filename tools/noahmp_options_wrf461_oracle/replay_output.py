"""Replay all 58 native Noah-MP budget outputs from real WRF captures."""
from pathlib import Path
import argparse,csv,hashlib,json,os
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','captures','namelist','receipt'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if os.environ.get('GPUWM_WRF_EXACT')!='1':p.error('GPUWM_WRF_EXACT=1 required')
    import cupy as cp
    from woof.core.noahmp_output import ACCUMULATORS,INPUTS,CARRIERS,advance
    pin=json.loads(Path(__file__).with_name('source_pin.json').read_text())
    meta=json.loads(a.metadata.read_text())
    assert meta['wrf_pin']==pin['wrf_pin']
    assert meta['original_sha256']==next(f['sha256'] for f in pin['files'] if f['upstream_path']=='phys/module_diag_misc.F')
    assert tuple(meta['acc'])==ACCUMULATORS and tuple(meta['inputs'])==INPUTS
    paths=sorted(a.captures.glob('noahmp-output-real-*.csv'));assert paths
    groups={}
    for path in paths:
        col={'input':{},'output':{}}
        for row in csv.DictReader(path.open()):col[row['stage']][row['field']]=int(row['hex'],16)
        key=tuple(col['input'][n] for n in ('dt','noahmp_acc_dt','curr_secs2','opt_run','itimestep','stepwtd'))
        groups.setdefault(key,[]).append((path,col))
    diff=[];words=0;periods=[];field_counts=dict.fromkeys(ACCUMULATORS,0)
    for key,cols in groups.items():
        dt,minutes,elapsed=np.array(key[:3],np.uint32).view(np.float32)
        opt_run,itimestep,stepwtd=np.array(key[3:],np.uint32).view(np.int32)
        fields={CARRIERS[n]:cp.asarray(np.array([col['input'][n] for _,col in cols],np.uint32).view(np.int32 if n=='isnow' else np.float32)) for n in (*ACCUMULATORS,*INPUTS)}
        advance(fields,dt=float(dt),minutes=float(minutes),elapsed_seconds=float(elapsed),opt_run=int(opt_run),itimestep=int(itimestep),stepwtd=int(stepwtd))
        periods.append(dict(elapsed_seconds=float(elapsed),itimestep=int(itimestep),columns=len(cols)))
        for n in ACCUMULATORS:
            actual=cp.asnumpy(fields[CARRIERS[n]]).view(np.uint32)
            for i,(path,col) in enumerate(cols):
                got=int(actual[i]);want=col['output'][n];words+=1;field_counts[n]+=1
                if got!=want:diff.append(dict(capture=path.name,field=n,got=f'{got:08x}',native=f'{want:08x}'))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(origin='real native WRF v4.6.1 noahmp_output_calc, official UCAR 2000-01-24 input',strict=True,columns=len(paths),formal_output_fields=58,compared_fields=58,compared_words=words,differing_words=len(diff),differences=diff,native_undefined=[],excluded_fields=[],periods=sorted(periods,key=lambda x:x['elapsed_seconds']),field_word_counts=field_counts,namelist_sha256=sha(a.namelist),captures=[dict(name=f.name,size=f.stat().st_size,sha256=sha(f)) for f in paths])
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('captures','field_word_counts')},indent=2))
    return bool(diff)
if __name__=='__main__':raise SystemExit(main())
