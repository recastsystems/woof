"""Write small receipts from the real column and restart checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import cupy as cp
from test_keps_wrf461_parity import GRADED, run_set
from woof.verify.keps_oracle import SETS, load_set
from woof.core.keps import keps_source

parser = argparse.ArgumentParser()
parser.add_argument('out', type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
report = {'mode': 'strict' if os.getenv('GPUWM_WRF_EXACT') == '1' else 'default',
          'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'kernel_sha256': hashlib.sha256(keps_source().encode()).hexdigest(),
          'sets': {}, 'tolerance_ulp': 0}
for name in SETS:
    fx = load_set(name)
    differences = run_set(name)
    fields = {}
    for field in GRADED:
        compared = sum(fx.step(field, step).size for step in range(1, fx.nstep + 1))
        finite = all(np.isfinite(fx.step(field, step)).all() for step in range(1, fx.nstep + 1))
        fields[field] = {'words': compared, 'differing_words': differences[field][0],
                         'max_ulp': differences[field][1], 'oracle_finite': bool(finite)}
    report['sets'][name] = {'columns': fx.ncol, 'steps': fx.nstep, 'levels': fx.nz, 'fields': fields}
    assert all(row['differing_words'] == 0 and row['oracle_finite'] for row in fields.values()), name
source = keps_source()
planted = np.nextafter(np.float32(.09), np.float32(1))
source = source.replace('#define KE_CMU     0.09f', f'#define KE_CMU     {float(planted).hex()}f')
report['planted_constant'] = {'before_word': int(np.float32(.09).view(np.uint32)),
                            'after_word': int(planted.view(np.uint32)),
                            'fields': run_set('Adev', source=source)}
assert sum(row[0] for row in report['planted_constant']['fields'].values()) > 0
report['verdict'] = 'PASS'
(args.out / 'columns.json').write_text(json.dumps(report, indent=2) + '\n')

from woof.core.dycore import run_steps
from woof.io import restart
from test_keps_driver_seam import _keps_restart_state
from test_restart import _assert_restart_equal
state, cfg, _ = _keps_restart_state(cp)
run_steps(state, cfg, 40)
reference = restart.write_restart(args.out / 'reference.npz', state, cfg)
state, cfg, _ = _keps_restart_state(cp)
run_steps(state, cfg, 20)
mid = restart.write_restart(args.out / 'mid.npz', state, cfg)
state, cfg, _ = _keps_restart_state(cp)
restart.restore_restart(mid, state, cfg)
roundtrip = restart.write_restart(args.out / 'roundtrip.npz', state, cfg)
_assert_restart_equal(mid, roundtrip)
run_steps(state, cfg, 20)
resumed = restart.write_restart(args.out / 'resumed.npz', state, cfg)
_assert_restart_equal(reference, resumed)
file_hashes = {label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
               for label, path in [('mid', mid), ('roundtrip', roundtrip),
                                   ('reference', reference), ('resumed', resumed)]}
receipts = {}
for label, path in [('mid', mid), ('roundtrip', roundtrip), ('reference', reference), ('resumed', resumed)]:
    with np.load(path, allow_pickle=False) as data:
        receipts[label] = {key: hashlib.sha256(data[key].tobytes()).hexdigest()
                           for key in data.files if key != '__gpuwm_restart_header__'}
assert receipts['mid'] == receipts['roundtrip']
assert receipts['reference'] == receipts['resumed']
(args.out / 'restart.json').write_text(json.dumps({'verdict': 'PASS', 'mode': report['mode'],
    'commit': report['commit'], 'file_hashes': file_hashes, 'arrays': receipts, 'header_check': 'test_restart._assert_restart_equal'}, indent=2) + '\n')
for path in [mid, roundtrip, reference, resumed]:
    size = Path(path).stat().st_size
    with (args.out / 'deleted-files.log').open('a') as log:
        log.write(f'{path} {size} bytes\n')
    Path(path).unlink()
print('PASS: columns, planted control, restart payload hashes and relevant headers')
