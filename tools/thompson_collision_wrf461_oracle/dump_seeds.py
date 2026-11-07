#!/usr/bin/env python3
"""Append a native-word dumper; do not change the public-domain WRF physics."""
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'thompson_classic_constants'))
import dump_constants as native

SOURCE_SHA256 = 'fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'

def main(source, destination):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected pinned public WRF v4.6.1 module_mp_thompson.F')
    native.REAL_ARRAYS = ['r_r', 'r_g', 'N0r_exp', 'N0g_exp', 'am_g']
    native.REAL_SCALARS = []
    native.DOUBLE_ARRAYS = ['dtr', 'dtg']
    native.INTS = []
    text = data.decode()
    anchor = 'END MODULE module_mp_thompson'
    if text.count(anchor) != 1:
        raise ValueError('native module end moved')
    Path(destination).write_text(text.replace(anchor, native.proc() + anchor))

if __name__ == '__main__':
    main(*sys.argv[1:])
