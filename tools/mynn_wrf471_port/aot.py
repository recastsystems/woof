#!/usr/bin/env python3
"""Build a source-bound MYNN image with parallel offline optimization."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from source import cuda_source

p = argparse.ArgumentParser()
p.add_argument('work', type=Path)
p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
p.add_argument('--arch', default='sm_120')
p.add_argument('--threads', type=int, default=4)
p.add_argument('--mode', choices=('strict', 'fast'), default='strict')
a = p.parse_args()
source = cuda_source(a.work)
unit = a.work / ('mynn471-' + a.mode + '.cu')
unit.write_text(source)
image = unit.with_suffix('.cubin')
options = ['--cubin', '-std=c++17', '-arch=' + a.arch, '--ftz=false',
           '--fmad=' + ('false' if a.mode == 'strict' else 'true'),
           '--prec-div=true', '--prec-sqrt=true', '--split-compile=' + str(a.threads)]
if a.mode == 'strict':
    options += ['-DGPUWM_WRF_EXACT=1']
command = [a.nvcc, *options, str(unit), '-o', str(image)]
print(json.dumps(dict(command=command)), flush=True)
subprocess.run(command, check=True)
manifest = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                schema_sha256=hashlib.sha256((a.work/'schema.json').read_bytes()).hexdigest(),
                arch=a.arch, mode=a.mode, command=command,
                compiler=subprocess.check_output([a.nvcc, '--version'], text=True))
image.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2), flush=True)
