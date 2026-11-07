#!/usr/bin/env python3
"""Install and inspect the prepared Linux wheels, without a source install.

The hosted test job and route P use this same entry point. Source contracts
run separately: -I and an outside cwd keep the checkout off the runtime path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv


EXPECTED_GAPS = (r'^cupy \(GPU runtime\)$', r'^WPS_GEOG$')


def doctor_problems(returncode: int, output: str) -> list[str]:
    """Honor doctor's blocking verdict, with the CPU-only exceptions of R12."""
    try:
        rows = json.loads(output)
        if not isinstance(rows, list) or not rows:
            raise ValueError('expected a nonempty check list')
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('blocking'), bool):
                raise ValueError('doctor check lacks its blocking verdict')
            if not isinstance(row.get('name'), str) or row.get('status') not in {
                    'verified', 'present', 'untested', 'missing', 'info'}:
                raise ValueError('invalid doctor check')
    except (ValueError, TypeError):
        return ['doctor did not produce a structured check list']
    gaps = [row for row in rows if row['status'] == 'missing' and row['blocking']]
    problems = [row['name'] + ': ' + row.get('detail', '') for row in gaps
                if not any(re.search(pattern, row['name']) for pattern in EXPECTED_GAPS)]
    expected_code = 1 if gaps else 0
    if returncode != expected_code:
        problems.append(f'doctor exited {returncode}, diagnostics require {expected_code}')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('operation', choices=('install', 'check'))
    parser.add_argument('--dists', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    # A release version is metadata, never a path fragment or pip option.
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?', args.version):
        parser.error('invalid release version')
    dists, out = args.dists.resolve(), args.out.resolve()
    wheels = [dists / f'gpuwm-{args.version}-py3-none-manylinux_2_28_x86_64.whl',
              dists / f'woof_data-{args.version}-py3-none-any.whl',
              dists / f'gpuwm_kernels-{args.version}-py3-none-any.whl',
              # 141dbf24d07: woof requires the startup companion; without it
              # pip refuses the install in the pinned image.
              dists / f'gpuwm_kernels_startup-{args.version}-py3-none-any.whl']
    identities = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}
    python = out / 'venv/bin/python'
    if args.operation == 'install':
        if out.exists():
            parser.error('smoke output exists: reusing an installed environment can hide missing wheel files')
        out.mkdir(parents=True)
        venv.EnvBuilder(with_pip=True).create(out / 'venv')
        subprocess.run([str(python), '-m', 'pip', 'install', str(wheels[0]) + '[render]', *map(str, wheels[1:])], check=True)
        (out / 'WHEELS.json').write_text(json.dumps(identities, indent=2) + '\n')
        return 0
    if json.loads((out / 'WHEELS.json').read_text()) != identities:
        parser.error('packet wheels changed after installation')
    outside, home = out / 'outside', out / 'home'
    outside.mkdir(); home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'GPUWM_', 'ARWEN_'))}
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    env.update(HOME=str(home), GPUWM_NO_LOCAL_GPU='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    command = [str(python), '-I', '-B']
    subprocess.run(command + ['-c', 'import woof, sys; print(woof.__file__, woof.__version__); '
        'assert woof.__version__ == sys.argv[1], "installed version differs from release"', args.version], cwd=outside, env=env, check=True)
    subprocess.run(command + ['-m', 'pip', 'check'], cwd=outside, env=env, check=True)
    doctor = subprocess.run(command + ['-m', 'woof.cli', 'doctor', '--json'], cwd=outside, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(doctor.stderr, file=sys.stderr, flush=True)
    print(doctor.stdout, flush=True)
    (out / 'doctor.json').write_text(doctor.stdout)
    (out / 'doctor.log').write_text(doctor.stderr)
    problems = doctor_problems(doctor.returncode, doctor.stdout)
    if problems:
        print('\n'.join(problems), file=sys.stderr)
        return 1
    (out / 'SMOKE.json').write_text(json.dumps({'status': 'PASS', 'version': args.version,
        'wheels': identities, 'doctor_exit': doctor.returncode}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
