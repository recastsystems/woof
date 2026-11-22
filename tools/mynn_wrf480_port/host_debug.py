#!/usr/bin/env python3
"""Host-only translation diagnostic; never a GPU qualification."""
import argparse
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('work', type=Path)
a = p.parse_args()
s = (a.work/'mynn471.cu').read_text()
s = s.replace('__device__', '').replace('__noinline__', '').replace('__global__', '')
s = s.replace('int c=blockIdx.x*blockDim.x+threadIdx.x;', 'int c=0;')
prefix = '#include <cmath>\ntypedef float real;\n'
for name, host in [('powf', 'powf'), ('expf', 'expf'), ('logf', 'logf'),
                   ('log10f', 'log10f'), ('atanf', 'atanf')]:
    prefix += '#define mynn_glibc_' + name + ' ' + host + '\n'
prefix += '#define mynn_tanhf tanhf\n'
(a.work/'host.cpp').write_text(prefix+s)
subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2',
                '-fno-fast-math', '-ffp-contract=off', '-fno-tree-vectorize',
                str(a.work/'host.cpp'), '-o', str(a.work/'host.so')], check=True)
