"""Witness CU7's unassigned inactive diagnostics using one real input capture.

Two private driver copies seed only IDEEP and JT before the native call.
The original capture remains the referee; these copies identify undefined
storage, and never replace or normalize that referee's words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .build import FLAGS, ROOT, SOURCES, verify
from .capture import layouts, read


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('capture', type=Path)
    ap.add_argument('scratch', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    source, work = args.source.resolve(), args.scratch.resolve()
    pin = verify(source)
    work.mkdir(parents=True, exist_ok=False)
    harness = ROOT / 'tools/camzm_wrf461_oracle'
    meta, data = read(args.capture, layouts()['cu7']['in'])
    ncol, nz = meta['nx'] * meta['ny'], meta['nz']
    case = work / 'real-input.cases'
    with case.open('wb') as stream:
        np.array([1129991479, 1, ncol, nz, 1, meta['pbl'], meta['sfc'],
                  meta['step'], 1], '<i4').tofile(stream)
        np.array([meta['dt'], meta['dx'], meta['cudt']], '<f4').tofile(stream)
        for name, array in data.items():
            array.tofile(stream)
    runner = (harness / 'run_camzm.F90').read_text()
    start = runner.index('     do i = 1, ncol\n        if (lengath2d(i,1) == 0) then')
    end = runner.index('     zmio_prefix', start)
    runner = runner[:start] + runner[end:]
    (work / 'run_camzm.F90').write_text(runner)
    commands = []

    def run(command):
        commands.append(command)
        # One compiler process, with headroom on this shared 48-load node.
        import os
        while os.getloadavg()[0] > 46.5:
            time.sleep(3)
        subprocess.run(command, cwd=work, check=True)

    compiler = shutil.which('gfortran')
    objects = []
    paths = [harness / 'stub_wrf.F90'] + [source / f for f in SOURCES['cu7']]
    paths += [harness / 'zmio.F90', work / 'run_camzm.F90']
    for path in paths:
        obj = path.stem + '.o'
        run([compiler, '-c', *FLAGS, '-o', obj, str(path)])
        # Compile the original driver's module interface for the harness;
        # each witness executable links its own seeded driver object.
        if path.name != 'module_cu_camzm_driver.F':
            objects.append(obj)
    native = (source / SOURCES['cu7'][-1]).read_text()
    anchor = '!\n! Check to see if this is a convection timestep...'
    assert native.count(anchor) == 1
    results = []
    from woof.verify.camzm_oracle import load, step_arrays
    for seed in (111111111, 222222222):
        driver = work / f'driver-{seed}.F90'
        driver.write_text(native.replace(anchor,
            f'  ideep = {seed}\n  jt = {seed}\n' + anchor))
        obj = driver.stem + '.o'
        run([compiler, '-c', *FLAGS, '-o', obj, str(driver)])
        exe = work / f'run-{seed}'
        run([compiler, '-o', str(exe), *objects, obj])
        stem = work / f'out-{seed}'
        run([str(exe), str(case), str(stem)])
        outputs = step_arrays(load(stem), 1)['out']
        results.append((seed, outputs))
    mask = results[0][1]['lengath2d'] == 0
    changed = {}
    for name, a in results[0][1].items():
        b = results[1][1][name]
        count = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
        if count:
            changed[name] = count
    passed = (mask.all() and set(changed) == {'ideep2d', 'jt2d'}
              and all(np.all(out[name] == seed) for seed, out in results
                      for name in ('ideep2d', 'jt2d'))
              and all(np.all(out['maxg2d'] == 1) for _, out in results))
    receipt = dict(kind='native-undefined-storage-witness', passed=bool(passed),
                   source_commit=pin['commit'], commands=commands,
                   input_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                   columns=ncol, meta=meta, seeds=[s for s, _ in results],
                   changed_words=changed, undefined_fields=['ideep2d', 'jt2d'],
                   defined_maxg=1, all_other_output_words_identical=passed,
                   original_capture_unmodified=True,
                   witness_only=True)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'commands'}))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
