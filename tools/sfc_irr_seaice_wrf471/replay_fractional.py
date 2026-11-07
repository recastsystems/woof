"""Replay native real-input MM5 fractional surface-driver captures."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('case', type=Path)
p.add_argument('--layout', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--dt', type=float, default=20.)
p.add_argument('--undefined-witness', type=Path,
               help='second native run with MALLOC_PERTURB_=64; only witnessed USTM ice words are excluded')
a = p.parse_args()
import cupy as cp
from woof.core.noah_fractional_seaice import mm5
from woof.core.physics_inventory import SFCLAY_OUTPUTS
from woof.core.sfclay import SFClayResult

layout = json.loads((a.layout/'fractional-layout.json').read_text())

def read(path):
    raw = path.read_bytes()
    header = np.frombuffer(raw, '<i4', 9).copy()
    threshold = np.frombuffer(raw, '<f4', 1, 36)[0]
    nx, ny = map(int, header[:2])
    values = np.frombuffer(raw, '<f4', offset=40).reshape(len(layout['f2']), ny, nx)
    return header, threshold, dict(zip(layout['f2'], values))

rows = []
for path in sorted((a.case/'captures').glob('fractional-*-in.bin')):
    h, threshold, host = read(path)
    _, _, ref = read(path.with_name(path.name.replace('-in.bin', '-out.bin')))
    undefined = np.zeros_like(host['xice'], dtype=bool)
    if a.undefined_witness is not None:
        witness_path = a.undefined_witness/'captures'/path.name.replace('-in.bin','-out.bin')
        wh, wt, witness = read(witness_path)
        assert np.array_equal(h, wh) and threshold == wt
        # USTM_SEA is a local automatic array never assigned before SFCLAYREV
        # reads it (WRF 4.7.1 module_surface_driver-capture.F:6341,6595). Do not excuse any
        # other output, any land word, or a word that the two native runs agree on.
        for name in ref:
            if name != 'ustm':
                assert np.array_equal(ref[name].view('u4'),witness[name].view('u4')), (witness_path,name)
        undefined = ((ref['ustm'].view('u4') != witness['ustm'].view('u4'))
                     & (host['xice'] >= np.float32(.02)) & (host['xice'] <= np.float32(1)))
    assert threshold == np.float32(.02)
    assert h[6] == 0, 'capture selects nondefault tice2tsk_if2cold'
    f = {n: cp.asarray(v) for n, v in host.items()}
    for n in SFCLAY_OUTPUTS:
        f.setdefault(n+'_sea', f[n].copy())
    atmosphere = {n: f[n][None] for n in
                  ('u', 'v', 'temperature', 'qv', 'pressure', 'dz')}
    atmosphere['p_interface'] = cp.stack((f['psfc'], f['psfc']))
    f['dx2d'] = f['dx']
    cfg = SimpleNamespace(dt=a.dt, dx=float(host['dx'][0,0]), sf_sfclay_physics=int(h[8]),
                          isfflx=int(h[3]), isftcflx=int(h[4]), iz0tlnd=int(h[5]))
    driver = SimpleNamespace(fields=f,
        sfclay_result=SFClayResult(*(f[n] for n in SFCLAY_OUTPUTS)),
        state=SimpleNamespace(elapsed_seconds=(int(h[2])-1)*a.dt))
    mm5(driver, atmosphere, cfg)
    cp.cuda.get_current_stream().synchronize()
    words = differing = 0
    fields = {}
    for name, expected in ref.items():
        actual = cp.asnumpy(f[name])
        left, right = actual.view(np.uint32), expected.view(np.uint32)
        compare = ~undefined if name == 'ustm' else np.ones_like(undefined)
        count = int(np.count_nonzero((left != right) & compare))
        words += int(compare.sum())
        differing += count
        if count:
            index = tuple(np.argwhere((left != right) & compare)[0])
            fields[name] = dict(differing=count, first_index=list(map(int,index)),
                gpu=float(actual[index]), native=float(expected[index]),
                gpu_word=hex(int(left[index])), native_word=hex(int(right[index])))
    row = dict(capture=path.name, words=words, differing=differing, fields=fields,
               excluded_native_undefined_ustm_words=int(undefined.sum()))
    rows.append(row)
    print(json.dumps(row), flush=True)
a.out.write_text(json.dumps(dict(words=sum(r['words'] for r in rows),
    differing=sum(r['differing'] for r in rows), captures=rows,
    undefined_witness=str(a.undefined_witness) if a.undefined_witness else None,
    excluded_native_undefined_ustm_words=sum(r['excluded_native_undefined_ustm_words'] for r in rows)), indent=2)+'\n')
