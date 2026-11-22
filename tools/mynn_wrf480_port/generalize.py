#!/usr/bin/env python3
"""Recover integer dimension expressions from two adjacent lowerings.

F2C needs fixed private array sizes. Its two outputs must differ only in
integer tokens affine in NZ; every other byte must agree. The resulting
CUDA source specializes at build time, without an f2c runtime dependency.
"""
import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('first', type=Path)
p.add_argument('second', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
first_meta = json.loads((a.first/'schema.json').read_text())
second_meta = json.loads((a.second/'schema.json').read_text())
nz = first_meta['nz']
assert second_meta['nz'] == nz + 1
left = re.split(r'\b(\d+)\b', (a.first/'mynn471.cu').read_text())
right = re.split(r'\b(\d+)\b', (a.second/'mynn471.cu').read_text())
assert len(left) == len(right), 'Lowered token streams have different lengths'
changed = 0
for i, (x, y) in enumerate(zip(left, right)):
    if x == y:
        continue
    if i % 2 != 1:
        raise ValueError(('Non-integer code changed', i, x, y))
    dx = int(y) - int(x)
    left[i] = '(' + str(dx) + '*(MYNN471_NZ-' + str(nz) + ')+' + x + ')'
    changed += 1
assert changed, 'No array dimensions changed'
a.output.write_text(''.join(left))
print(json.dumps(dict(base_nz=nz, checked_nz=nz+1, dimension_expressions=changed)))
