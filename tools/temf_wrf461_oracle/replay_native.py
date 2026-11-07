"""Replay the two untouched-physics captures and count every output word."""
from pathlib import Path
import hashlib
import json
import struct
import sys
import numpy as np


def read_capture(root, routine):
    root = Path(root)
    fixture_path = root/'native-capture.json'
    if fixture_path.is_file():
        fixture = json.loads(fixture_path.read_text())
        layout = fixture['layout']
        data = bytes.fromhex(fixture['captures'][routine])
    else:
        layout = json.loads((root/'capture-layout.json').read_text())
        data = (root/f'{routine}-capture.bin').read_bytes()
    ncol, nz = struct.unpack_from('<ii', data)
    offset = 8
    dt = None
    if routine == 'temfpbl':
        dt, = struct.unpack_from('<f', data, offset)
        offset += 4
    result = {}
    for side in ('inputs', 'outputs'):
        result[side] = {}
        for name, expression in layout[routine][side]:
            levels = nz+1 if 'kte+1' in expression else nz
            shape = (levels, ncol) if 'kts:kte' in expression else (ncol,)
            dtype = '<i4' if name == 'kpbl2d' else '<f4'
            value = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape)),
                                  offset=offset).copy().reshape(shape)
            offset += value.nbytes
            result[side][name] = value
    assert offset == len(data), (routine, offset, len(data))
    result.update(columns=ncol, levels=nz, dt=dt,
                  capture_sha256=hashlib.sha256(data).hexdigest())
    return result

def compare(expected, got):
    fields = {}
    for name, value in expected.items():
        actual = got[name]
        different = value.view(np.uint32) != actual.view(np.uint32)
        indices = np.flatnonzero(different)
        fields[name] = {
            'words': value.size, 'differing_words': int(different.sum()),
            'native_nonfinite': int((~np.isfinite(value)).sum()),
            'gpu_nonfinite': int((~np.isfinite(actual)).sum()),
            'native_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            'gpu_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
            'first_differences': [
                {'flat_index': int(i),
                 'native': f'{value.view(np.uint32).flat[i]:08x}',
                 'gpu': f'{actual.view(np.uint32).flat[i]:08x}'}
                for i in indices[:4]]}
    return {'words': sum(f['words'] for f in fields.values()),
            'differing_words': sum(f['differing_words'] for f in fields.values()),
            'fields': fields}

def replay(capture, routine, stock):
    import cupy as cp
    from woof.core.temf import launch_temf_pbl, launch_temf_sfclay
    from woof.core.physics_inventory import (
        TEMF_PBL_STATE_3D, TEMF_PBL_STATE_2D,
        TEMF_SFCLAY_INOUT, TEMF_SFCLAY_OUTPUTS)
    inputs = {n: cp.asarray(v) for n, v in capture['inputs'].items()}
    output = {n: cp.zeros(v.shape, v.dtype)
              for n, v in capture['outputs'].items()}
    if routine == 'temfpbl':
        aliases = {'u': 'u3d', 'v': 'v3d', 'th': 'th3d', 't': 't3d',
                   'qv': 'qv3d', 'qc': 'qc3d', 'p': 'p3d', 'pi': 'pi3d',
                   'rho': 'rho', 'z': 'z'}
        columns = {n: inputs[src] for n, src in aliases.items()}
        surface = {n: inputs[n if n != 'fcor' else 'fCor']
                   for n in ('znt', 'ht', 'hfx', 'qfx', 'tsk', 'fcor')}
        inout = {n: inputs[src] for n, src in {
            'ust': 'ust', 'pblh': 'hpbl', 'kpbl': 'kpbl2d',
            'u10': 'u10', 'v10': 'v10', 't2': 't2'}.items()}
        carried = {n: inputs[n] for n in TEMF_PBL_STATE_3D}
        carried2 = {n: inputs[n] for n in TEMF_PBL_STATE_2D}
        rates = {n: output[n] for n in (
            'rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten')}
        launch_temf_pbl(columns, inputs['p3di'], surface, inout, carried,
                        carried2, rates, dt=capture['dt'],
                        verify_stock_wrf=stock)
        output.update(carried)
        output.update(carried2)
        for n, src in [('ust','ust'), ('hpbl','pblh'), ('kpbl2d','kpbl'),
                       ('u10','u10'), ('v10','v10'), ('t2','t2')]:
            output[n] = inout[src]
    else:
        aliases = {'u1':'u3d', 'v1':'v3d', 'th1':'th3d', 'qv1':'qv3d',
                   'p1':'p3d', 'pi1':'pi3d', 'rho1':'rho', 'z1':'z',
                   **{n:n for n in ('ht','psfc','znt','mavail','tsk','hd_temf')},
                   'fcor':'fCor'}
        surface = {n: inputs[src] for n, src in aliases.items()}
        state = {n: output[n] for n in TEMF_SFCLAY_INOUT if n != 'te1'}
        state.update(hfx=inputs['hfx'], wm_temf=inputs['wm_temf'],
                     te1=inputs['te_temf'])
        launch_temf_sfclay(surface, state, {n: output[n] for n in TEMF_SFCLAY_OUTPUTS},
                          verify_stock_wrf=stock)
        output.update({n:v for n,v in state.items() if n != 'te1'})
        output['te_temf'] = state['te1']
    cp.cuda.get_current_stream().synchronize()
    return compare(capture['outputs'], {n:cp.asnumpy(v) for n,v in output.items()})


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    root = args.capture
    layout = ('temfpbl', 'temfsfclay')
    import cupy as cp
    from woof.core.kernels import module_options
    from woof.wrf_exact import ENABLED
    assert ENABLED, 'the native proof requires GPUWM_WRF_EXACT=1'
    receipt = {'schema':'temf-real-native-replay-v1', 'strict':ENABLED,
               'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
               'options':{n:module_options(n) for n in ('temf','temf_sfclay')},
               'reference':'Native WRF v4.6.1 run, unmodified physics, no math shim',
               'routines':{}}
    for routine in layout:
        capture = read_capture(root, routine)
        receipt['routines'][routine] = {
            k:capture[k] for k in ('columns','levels','dt','capture_sha256')}
        for stock in (True, False):
            receipt['routines'][routine]['stock-transcription' if stock else 'production'] = replay(capture, routine, stock)
    path = args.out or root/'replay-receipt.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    for routine, row in receipt['routines'].items():
        for arm in ('stock-transcription','production'):
            print(routine, arm, row[arm]['differing_words'], '/', row[arm]['words'])
    return int(any(row["production"]["differing_words"] for row in receipt["routines"].values()))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
