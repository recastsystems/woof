"""Emit exact predecessor roots for independent GPU table-build segments."""
from pathlib import Path
import hashlib
import sys
import numpy as np
words = Path(sys.argv[1]).read_bytes()
expected = '46b826ff2e7519cd89149ca9c351bc8aa5819f5e1a1812dea1f306f5e6735a61'
if hashlib.sha256(words).hexdigest() != expected:
    raise SystemExit('table dump differs from pinned native WRF v4.6.1')
tdp = np.frombuffer(words, dtype='<f8')[16 + 7501:16 + 7501 + 5001]
seeds = [208.0] + [float(tdp[i-1]) for i in range(32, 5001, 32)]
text = ('// Generated from public-domain WRF v4.6.1 module_gfs_funcphys.F.\n'
        '// Table SHA-256: ' + expected + '\n'
        '// Regenerate: python3 tools/osas_wrf461_oracle/table_seeds.py TABLE_DUMP OUTPUT\n'
        '__device__ __constant__ double osas_tdp_seed[] = {\n')
text += ''.join('    ' + v.hex() + ',\n' for v in seeds) + '};\n'
Path(sys.argv[2]).write_text(text)
