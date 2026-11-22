#!/usr/bin/env python3
"""Generate the CUDA transcription from the licence-checked native pin."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from build import HERE, MANIFESTS, verify

#: The canonical transcription of each pinned release.  v4.8.0's differs
#: from v4.7.1's by the one moved RATIO statement; the shipping kernel
#: composes it from ntu.cu (woof.mp_generations), and
#: tests/test_mp_generations.py holds that composition to this digest.
EXPECTED_BY_WRF = {
    '4.7.1': 'cff4807966d0ee63217616278536bedb58a0fa60446238f0327af95cd29c1e0b',
    '4.8.0': 'c3f150a6ef85dd8b742a7c846939295b080cfc1d92524eeb1c0a2db151771f2d',
}
EXPECTED = EXPECTED_BY_WRF['4.7.1']


def generate(source, licence, out, compiler, wrf='4.7.1'):
    source, licence, out = source.resolve(), licence.resolve(), out.resolve()
    verify(source, licence, wrf)
    out.mkdir(parents=True,exist_ok=True)
    translator=HERE.parent/'ntu_wrf461_oracle'/'f2cu.py'
    params=out/'params.f90'
    values=out/'params.txt'
    generated=out/'ntu_gen.cuh'
    commands=[
        [sys.executable,str(translator),str(source),'--params-out',str(params)],
        [compiler,'-O2','-fno-fast-math','-ffp-contract=off','-ffree-form',
         '-ffree-line-length-none',str(params),'-o',str(out/'params')],
        [str(out/'params')],
        [sys.executable,str(translator),str(source),'--params-in',str(values),'--out',str(generated)],
    ]
    for i,command in enumerate(commands):
        if i==2:
            with values.open('w') as f:
                subprocess.run(command,cwd=out,stdout=f,check=True)
        else:
            subprocess.run(command,cwd=out,check=True)
    digest=hashlib.sha256(generated.read_bytes()).hexdigest()
    expected=EXPECTED_BY_WRF[wrf]
    if digest!=expected:
        raise ValueError(f'transcription changed: expected {expected}, got {digest}')
    receipt=dict(source=MANIFESTS[wrf],commands=commands,generated_sha256=digest,
                 compiler=subprocess.check_output([compiler,'--version'],text=True))
    (out/'generate-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','licence','out'):
        ap.add_argument(name,type=Path)
    ap.add_argument('--compiler',default='gfortran')
    ap.add_argument('--wrf',choices=tuple(MANIFESTS),default='4.7.1')
    args=ap.parse_args()
    generate(args.source,args.licence,args.out,args.compiler,args.wrf)
