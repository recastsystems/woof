#!/usr/bin/env python3
"""Compare physical AMD strict outputs with retained unchanged CUDA authorities.

python engine/tools/amd_strict_rerun.py --authority /path/authorities --out /path/run
Authorities: authority-arwen-whole-strict.npz, authority-arwen-real-strict.npz,
math-audit-full-cuda.json. The math diagnostic enumerates all binary32 words;
double functions get their exact conversions, and powf uses EOS exponent 1.4.
It is not exhaustive binary64 or two-operand pow coverage.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    authority, out = args.authority.resolve(), args.out.resolve()
    for name in ('authority-arwen-whole-strict.npz', 'authority-arwen-real-strict.npz',
                 'math-audit-full-cuda.json'):
        if not (authority / name).is_file():
            parser.error(f'missing CUDA authority: {authority / name}')
    audit = json.loads((authority / 'math-audit-full-cuda.json').read_text())
    try:
        from tools.hip_math_audit import FUNCTIONS
    except ImportError:  # run as a path: the sibling is beside this file
        from hip_math_audit import FUNCTIONS
    if not audit['complete'] or not audit['exhaustive_binary32_inputs']:
        parser.error('math authority is incomplete')
    if set(audit['functions']) != set(FUNCTIONS) or any(
            row['enumerated_inputs'] != 2**32 for row in audit['functions'].values()):
        parser.error('math authority must enumerate all requested functions and inputs')
    if audit.get('compilation_route') != 'direct; no CuPy RawModule implicit FTZ append':
        parser.error('use the direct-compiled CUDA math authority')
    out.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    env = dict(os.environ, WOOF_BACKEND='amd', GPUWM_NUMERICS='strict',
               WOOF_RUST_BACKEND='hip', NUMPY_MADVISE_HUGEPAGE='0')
    result = dict(complete=False, passed=False, stages=[])

    def run(name, command):
        with (out / (name + '.log')).open('w') as handle:
            code = subprocess.run(command, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        result['stages'].append(dict(name=name, argv=list(map(str, command)), exit_code=code))
        (out / 'receipt.json').write_text(json.dumps(result, indent=2))
        return code == 0

    # Refuse CUDA CuPy before producing a result labelled physical AMD.
    check = ('import cupy as c; assert c.cuda.runtime.is_hip, "physical AMD/ROCm CuPy required"; '
             'free,total=c.cuda.runtime.memGetInfo(); '
             'print({"free_bytes":free,"total_bytes":total}); '
             'assert free >= 4*1024**3, "at least 4 GiB free is required"')
    if not run('physical-amd-precheck', [sys.executable, '-c', check]):
        return 1
    for kind, script in (('whole', 'hip_whole_step.py'), ('real', 'hip_real_replay.py')):
        actual = out / f'amd-arwen-{kind}-strict.npz'
        if run(kind, [sys.executable, str(tools / script), '--output', str(actual)]):
            run(kind + '-compare', [sys.executable, str(tools / 'hip_real_replay.py'),
                '--compare', str(authority / f'authority-arwen-{kind}-strict.npz'), str(actual)])
    run('exhaustive-math', [sys.executable, str(tools / 'hip_math_audit.py'),
        '--authority', str(authority / 'math-audit-full-cuda.json'),
        '--output', str(out / 'math-audit-amd.json')])
    result['complete'] = True
    result['passed'] = all(row['exit_code'] == 0 for row in result['stages'])
    (out / 'receipt.json').write_text(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
