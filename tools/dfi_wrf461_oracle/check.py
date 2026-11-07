"""Compare Rust coefficients to original WRF routines and witness RHOFIL.

This is a coefficient/operator check, NOT a real native forecast capture.
No product selector or whole-model qualification follows from this receipt.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def run(argv, *, cwd, log):
    completed = subprocess.run([str(x) for x in argv], cwd=cwd,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write('$ ' + ' '.join(str(x) for x in argv) + '\n')
    log.write(completed.stdout.decode(errors='replace'))
    log.flush()
    return completed


def words(data):
    return [int.from_bytes(data[i:i+4], 'little') for i in range(0, len(data), 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='the pinned original share/dfi.F')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.read_bytes()
    pin = json.loads((HERE / 'SOURCE.json').read_text())['files']['share/dfi.F']
    assert hashlib.sha256(source).hexdigest() == pin, 'WRF source hash mismatch'
    text = source.decode()
    start = text.index('   SUBROUTINE dfcoef (')
    end = text.index('    SUBROUTINE optfil (')
    # Extract complete unmodified routines: dfcoef, windows, RHO and CNR.
    extracted = out / 'wrf_dfi_weights.F90'
    extracted.write_text(text[start:end])
    receipt = {'scope': 'coefficient routines, synthetic time-window inputs',
               'source_sha256': pin, 'real_forecast_capture': False,
               'cases': [], 'words': 0, 'differing_words': 0}
    with (out / 'commands.log').open('w') as log:
        for cmd in (['gfortran', '--version'], ['rustc', '--version']):
            assert run(cmd, cwd=out, log=log).returncode == 0
        flags = ['-cpp', '-DWRFPLUS=0', '-ffree-form', '-ffree-line-length-none',
                 '-fno-fast-math', '-ffp-contract=off', '-fno-tree-vectorize']
        for arm, optimization in [('o0', '-O0'), ('o2', '-O2')]:
            compiled = run(['gfortran', *flags, optimization,
                            extracted, HERE / 'driver.F90', '-o', out / arm],
                           cwd=out, log=log)
            assert compiled.returncode == 0, compiled.stdout.decode()
        compiled = run(['rustc', '--crate-type', 'cdylib', '--edition', '2021',
                        '-C', 'opt-level=3', ROOT / 'tools/dfi-runtime/src/lib.rs',
                        '-o', out / 'libarwen_dfi.so'], cwd=out, log=log)
        assert compiled.returncode == 0, compiled.stdout.decode()
        lib = ctypes.CDLL(str(out / 'libarwen_dfi.so'))
        weight_fn = lib.aw_dfi_weights
        weight_fn.argtypes = [ctypes.c_int32, ctypes.c_float, ctypes.c_float,
                              ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
                              ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        weight_fn.restype = ctypes.c_int32
        # One bounded operator replay: all defined native filters and both
        # odd/even chronological mirroring. Not a forecast/scheme sweep.
        for nf in range(8):
            for samples in (60, 61):
                steps = (samples + 1) // 2 - 1
                outputs = []
                for arm in ('o0', 'o2'):
                    path = out / f'{arm}-{nf}-{samples}.bin'
                    p = run([out / arm, steps, nf, 30, 3600, path], cwd=out, log=log)
                    assert p.returncode == 0, p.stdout.decode()
                    outputs.append(path.read_bytes())
                assert outputs[0] == outputs[1], f'native O0/O2 differ: {nf}, {samples}'
                half = words(outputs[0])
                mirror = half[:-1] if samples % 2 else half
                native = half + list(reversed(mirror))
                result = (ctypes.c_float * samples)()
                written = ctypes.c_size_t()
                status = weight_fn((samples - 1) * 30, 30, 3600, nf,
                                   result, samples, ctypes.byref(written))
                assert status == 0 and written.value == samples
                rust = words(bytes(result))
                diffs = sum(a != b for a, b in zip(native, rust))
                receipt['cases'].append({'filter': nf, 'samples': samples,
                    'words': samples, 'differing_words': diffs,
                    'native_sha256': hashlib.sha256(outputs[0]).hexdigest()})
                receipt['words'] += samples
                receipt['differing_words'] += diffs
        # Bounds-check the actual original routine in its own subprocess.
        # nsteps=10 allocates FROW(0:10). WRF clears FROW(0:100).
        p = run(['gfortran', *flags, '-O0', '-fcheck=bounds', '-g', extracted,
                 HERE / 'driver.F90', '-o', out / 'bounds'], cwd=out, log=log)
        assert p.returncode == 0, p.stdout.decode()
        p = run([out / 'bounds', 10, 8, 30, 3600, out / 'undefined.bin'], cwd=out, log=log)
        diagnostic = p.stdout.decode(errors='replace')
        receipt['filter8'] = {'exit_code': p.returncode,
            'native_undefined_witness': p.returncode != 0 and
                'Index' in diagnostic and 'above upper bound' in diagnostic,
            'diagnostic': diagnostic}
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['differing_words'] == 0 and receipt['filter8']['native_undefined_witness'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
