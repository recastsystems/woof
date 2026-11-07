#!/usr/bin/env python3
"""Check fused pressure-bound optimization on real and inverted columns.

This is a fast-path equivalence check, not an independent native replay.
The separate strict replay is the native arithmetic authority.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cupy as cp
    from woof.core.level_diagnostics import LevelInputs, pressure_levels
    from woof.core.kernels import get_kernel
    from woof.wrf_exact import ENABLED
    if ENABLED:
        parser.error('this check requires default fast numerics; strict ignores private bounds')
    with np.load(args.capture, allow_pickle=False) as data:
        arrays = {k:np.ascontiguousarray(data[k], dtype='f4') for k in
                  ('u','v','t','qv','ph','phb','pp','pb','p','pw')}
        provenance=json.loads(str(data['provenance']))
    if not provenance.get('real_native_capture'):
        parser.error('capture must retain real native-run provenance')
    rows=[]
    for case in ('real_native','derived_inverted_pressure'):
        a={k:v.copy() for k,v in arrays.items()}
        if case != 'real_native':
            # The maxima lie inside the column, so checking only its surface
            # would wrongly reject a valid bracket. Targets also reverse order.
            for name in ('p','pw'):
                a[name][a[name].shape[0]//3]=np.float32(110000.)
        d={k:cp.asarray(v) for k,v in a.items()}
        nz,ny,nx=a['p'].shape;n=ny*nx
        held_p=cp.empty_like(d['p']);held_pw=cp.empty_like(d['pw'])
        bounds=cp.empty((2,ny,nx),dtype='f4')
        get_kernel('level_diagnostics','level_capture_pressure')(((n+127)//128,),(128,),
            (d['p'],d['pw'],held_p,held_pw,bounds,np.int32(nz),np.int32(n)))
        inputs=LevelInputs(*(d[k] for k in ('u','v','t','qv','ph','phb','pp','pb')))
        targets=cp.asarray([115000.,105000.,92500.,110000.,70000.,85000.,30000.],dtype='f4')
        for kind in (1,2):
            for extrap in (1,2):
                plain=cp.empty((8,targets.size,ny,nx),dtype='f4');cached=cp.empty_like(plain)
                pressure_levels(inputs,held_p,held_pw,targets,plain,pressure_kind=kind,extrapolate=extrap)
                pressure_levels(inputs,held_p,held_pw,targets,cached,pressure_kind=kind,extrapolate=extrap,bounds=bounds)
                x,y=map(cp.asnumpy,(plain,cached))
                different=int(np.count_nonzero(x.view('u4')!=y.view('u4')))
                rows.append(dict(case=case,pressure_kind=kind,extrapolate=extrap,words=x.size,
                    differing_words=different,sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    result=dict(status='PASS' if all(x['differing_words']==0 for x in rows) else 'FAIL',
        method='default fast fused/inline bound equivalence, not native arithmetic replay',
        capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        provenance=provenance,command=sys.argv,rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],sum(x['words'] for x in rows),'compared words')
    return result['status']!='PASS'

if __name__=='__main__':
    raise SystemExit(main())
