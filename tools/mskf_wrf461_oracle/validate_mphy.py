"""Replay compact WRF aerosol call recordings through the column microphysics."""
import json
from pathlib import Path
import argparse
import numpy as np


def main(output):
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.certify.kernel_manifest import kernel_manifest
    root=Path('tests/data/oracles/mskf')
    schema=json.loads((root/'mphy-schema.json').read_text())
    with np.load(root/'mphy-records.npz') as data:
        inputs=data['inputs'];ref=data['outputs']
    nz=40
    kernel=get_kernel('mskf_aerosol_replay','mskf_mphy_replay')
    workspace=cp.empty(512*(nz+2)*32,dtype=cp.float64)
    constants=np.fromfile(root/'mphy-constants.bin','<f8')
    device_constants=cp.empty(constants.size,dtype=cp.float64)
    get_kernel('mskf_aerosol_replay','mskf_aero_constants')((1,),(32,),(device_constants,workspace))
    actual_constants=device_constants.get()
    differences=np.flatnonzero(constants.view(np.uint64)!=actual_constants.view(np.uint64))
    print('constants',[(int(i),float(constants[i]),float(actual_constants[i])) for i in differences])
    rows=[]
    for i, record in enumerate(inputs):
        inp=cp.asarray(record)
        out=cp.empty_like(inp)
        status=cp.zeros(1,dtype=cp.int32)
        kernel((1,),(32,),(inp,out,workspace,status,np.int32(nz),np.int32(record.size)))
        got=cp.asnumpy(out)
        fields={}
        for field in schema:
            sl=slice(field['offset'],field['offset']+field['size'])
            a,b=ref[i,sl],got[sl]
            d=a.view(np.uint64)!=b.view(np.uint64)
            if d.any():
                at=int(np.flatnonzero(d)[0])
                fields[field['name']]={'different':int(d.sum()),'first_index':at,
                                        'first_ref':float(a[at]),'first_port':float(b[at])}
        rows.append({'case':i+1,'status':int(status.get()[0]),'fields':fields})
        print(i+1,rows[-1])
    output.write_text(json.dumps({'rows':rows,'kernel_manifest':kernel_manifest()},indent=2)+'\n')
    return int(any(r['fields'] or r['status'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    raise SystemExit(main(p.parse_args().output))
