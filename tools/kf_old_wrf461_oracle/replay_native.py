"""Grade opt-in native KFCPS captures, including all returned diagnostic words.

Capture layout is versioned and big endian, matching the strict native build.
The capture instrumentation belongs to the proof build, never the step loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUTS = ('u', 'v', 'w', 't', 'th', 'qv', 'dz8w', 'p', 'rho', 'pi', 'w0avg')
VECTORS = ('rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten',
           'rqscuten', 'w0avg_out', 'udr_kf', 'ddr_kf', 'uer_kf', 'der_kf')
SCALARS = ('raincv', 'pratec', 'nca_out', 'timec_kf')


def read_capture(path: Path):
    raw = path.read_bytes()
    header = np.frombuffer(raw, dtype='>i4', count=13).astype(np.int32)
    magic, version, nz, ktau, i, j, stepcu, adapt, warm, qi, qs, qr, edrates = header
    if magic != 0x4b463939 or version != 1 or not 8 <= nz <= 128:
        raise ValueError(f'{path}: invalid native KFCPS capture header {header}')
    expected = 13*4 + 4*4 + (11*nz+1)*4 + 4 + 4*4 + 11*nz*4
    if len(raw) != expected:
        raise ValueError(f'{path}: expected {expected} bytes, found {len(raw)}')
    offset = 13*4
    def floats(count):
        nonlocal offset
        result = np.frombuffer(raw, dtype='>f4', count=count,
                               offset=offset).astype(np.float32)
        offset += count*4
        return result
    dt, dx, cudt, nca = floats(4)
    inputs = {name: floats(nz+1 if name == 'w' else nz) for name in INPUTS}
    active = np.frombuffer(raw, dtype='>i4', count=1, offset=offset).astype(np.int32)
    offset += 4
    outputs = dict(zip(SCALARS, floats(4).reshape(4, 1)))
    outputs.update({name: floats(nz) for name in VECTORS})
    outputs['cu_act_flag'] = active
    outputs['fatal'] = np.zeros(1, np.int32)
    assert offset == len(raw)
    return dict(name=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw), nz=int(nz), ktau=int(ktau), i=int(i), j=int(j),
                stepcu=int(stepcu), adaptive=bool(adapt), warm_rain=bool(warm),
                f_qi=bool(qi), f_qs=bool(qs), f_qr=bool(qr), kf_edrates=bool(edrates),
                dt=float(dt), dx=float(dx), cudt_minutes=float(cudt), nca=nca,
                inputs=inputs, outputs=outputs)


def replay(directory: Path):
    import cupy as cp
    from woof.core.kf import KFOldSpecies, launch_kf_old
    from woof.wrf_exact import ENABLED
    from woof.numerics import mode
    fields = {name: dict(words=0, differing=0) for name in VECTORS+SCALARS+('cu_act_flag','fatal')}
    records = []
    for path in sorted(directory.glob('kf-*.bin')):
        capture = read_capture(path)
        def col(name):
            return cp.asarray(capture['inputs'][name].reshape(-1, 1, 1))
        mean = col('w0avg')
        out = launch_kf_old(col('u'), col('v'), col('w'), col('t'), col('qv'),
                            col('p'), col('rho'), col('dz8w'), col('pi'), mean,
                            cp.full((1,1), capture['nca'], dtype=cp.float32),
                            dx=capture['dx'], dt=capture['dt'],
                            cudt_minutes=capture['cudt_minutes'],
                            stepcu=capture['stepcu'], adaptive=capture['adaptive'],
                            species=KFOldSpecies(capture['warm_rain'], capture['f_qi'],
                                                 capture['f_qs'], capture['f_qr']),
                            kf_edrates=capture['kf_edrates'])
        out['w0avg_out'] = mean
        mismatches = []
        for name, want in capture['outputs'].items():
            if name not in out:
                # Disabled optional diagnostics have no native output contract.
                if not capture['kf_edrates'] and name in ('udr_kf','ddr_kf','uer_kf','der_kf','timec_kf'):
                    continue
                raise KeyError(name)
            have = cp.asnumpy(out[name]).ravel().astype(want.dtype)
            differing = np.flatnonzero(have.view(np.uint32) != want.view(np.uint32))
            fields[name]['words'] += want.size
            fields[name]['differing'] += differing.size
            if differing.size:
                first = int(differing[0])
                mismatches.append(dict(field=name, differing=int(differing.size), first=first,
                                       have_bits=int(have.view(np.uint32)[first]),
                                       want_bits=int(want.view(np.uint32)[first])))
        records.append({**{k:v for k,v in capture.items() if k not in ('inputs','outputs','nca')},
                        'mismatches':mismatches,
                        'native_pratec':float(capture['outputs']['pratec'][0]),
                        'native_raincv':float(capture['outputs']['raincv'][0]),
                        'native_active':int(capture['outputs']['cu_act_flag'][0])})
    if not records:
        raise ValueError(f'{directory}: no native captures')
    return dict(schema='gpuwm-cu99-native-replay-v1', strict=mode() == 'strict',
                numerics=mode(), wrf_exact=ENABLED,
                fields=fields, words=sum(v['words'] for v in fields.values()),
                captured_output_words=sum(v['words'] for name, v in fields.items()
                                          if name != 'fatal'),
                derived_status_words=fields['fatal']['words'],
                status_scope='fatal=0 is inferred from the completed native call; '
                             'all other fields are captured native output words',
                differing=sum(v['differing'] for v in fields.values()), captures=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    result = replay(args.directory)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'captures'}))
    return bool(result['differing'])

if __name__ == '__main__':
    raise SystemExit(main())
