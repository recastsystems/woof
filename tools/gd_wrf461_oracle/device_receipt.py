"""Write field hashes and ULP grades from the actual GD CUDA launch route."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import gd_cases
from validate_kernel import compare

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args=parser.parse_args()
    import cupy as cp
    from woof.core.gf import gd_launch, GD_MODULE_OPTIONS, gd_module_source
    root=Path(__file__).resolve().parents[2]
    fixture=root/'tests/data/oracles/gd'
    args.out.mkdir(parents=True,exist_ok=True)
    report={'mode':'strict' if os.environ.get('GPUWM_WRF_EXACT')=='1' else 'default',
            'source_sha256':hashlib.sha256(gd_module_source().encode()).hexdigest(),
            'options':list(GD_MODULE_OPTIONS),'cases':[], 'deleted_temp_files':[]}
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import kernel_manifest
    original_compile = compiler.compile_using_nvrtc
    captures = []
    def recorded_compile(*positional, **keywords):
        result = original_compile(*positional, **keywords)
        ptx = result[0].encode() if isinstance(result[0], str) else bytes(result[0])
        captures.append({"ptx_sha256":hashlib.sha256(ptx).hexdigest(),
                         "ptx_bytes":len(ptx),
                         "ftz_modifier_count":ptx.count(b".ftz")})
        return result
    compiler.compile_using_nvrtc = recorded_compile
    for name in ('columns','edge','cadence'):
        for fqi in (0,1):
            raw=bytearray(gzip.decompress((fixture/f'gd-{name}.in.gz').read_bytes()))
            raw[28:32]=int(fqi).to_bytes(4,'little')
            inp_path=args.out/f'input-{name}-{fqi}.in'
            inp_path.write_bytes(raw)
            inp=gd_cases.read_input(inp_path)
            for arm,count in (('stock',1),('fixed',0)):
                nz=inp['nz']
                def run(lvin,scin,iin):
                    ens=cp.empty((lvin.shape[0],288),cp.float32)
                    lev,sca,isc=gd_launch(cp.asarray(lvin),cp.asarray(scin),cp.asarray(iin),nz=nz,wrf_closure_count=count,ens=ens)
                    return tuple(cp.asnumpy(a) for a in (lev,sca,isc,ens))
                got=gd_cases.emulate_driver(inp,run)
                oracle_path=args.out/f'oracle-{name}-{fqi}-{arm}.out'
                oracle_path.write_bytes(gzip.decompress((fixture/f'gd-{name}-fqi{fqi}.{arm}.out.gz').read_bytes()))
                want=gd_cases.read_output(oracle_path,inp['nx'],inp['ny'],nz,inp['nsteps'])
                grade=compare(want,got)
                fields=[]
                for step,(a,b) in enumerate(zip(want,got)):
                    for field in grade:
                        if field=='due':
                            continue
                        av,bv=np.ascontiguousarray(a[field]),np.ascontiguousarray(b[field])
                        raw_misses=int(np.count_nonzero(av.view(np.uint32)!=bv.view(np.uint32)))
                        fields.append({'step':step,'field':field,'oracle_sha256':digest(av),'cuda_sha256':digest(bv),'raw_word_misses':raw_misses})
                report['cases'].append({'fixture':name,'f_qi':fqi,'arm':arm,'columns':inp['nx']*inp['ny'],'steps':inp['nsteps'],'grade':{k:list(v) for k,v in grade.items()},'fields':fields})
                report['deleted_temp_files'].append({'path':str(oracle_path),'bytes':oracle_path.stat().st_size})
                oracle_path.unlink()
            report['deleted_temp_files'].append({'path':str(inp_path),'bytes':inp_path.stat().st_size})
            inp_path.unlink()
    compiler.compile_using_nvrtc = original_compile
    report["kernel_manifest"] = kernel_manifest()
    report["actual_ptx"] = captures
    target=args.out/'device-receipt.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    bad=sum(v[0] for case in report['cases'] for v in case['grade'].values())
    bits=sum(f['raw_word_misses'] for case in report['cases'] for f in case['fields'])
    print(f'{report["mode"]}: {bad} nonzero-ULP words; {bits} raw-bit misses; {target}')
    return int(bad!=0 or bits!=0 or len(captures)!=1 or any(c["ftz_modifier_count"] for c in captures))

if __name__=='__main__':
    raise SystemExit(main())
