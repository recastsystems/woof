#!/usr/bin/env python3
"""Strict module replay of captured native mynnsfc_driver calls (RRFS referee of record, x1.2562 hrrrv5).

For every captured call (main and sea-ice) of each case:
  1. harness witness: the standalone native v6.0.0 build (sfc_referee.py v600) on the captured PRE must give the
     captured whole-model native POST in every argument word;
  2. pinned port: sf_mynnsfc_v600 on the GPU against the same captured POST;
  3. candidate: sf_mynnsfc_v610 on the GPU against the standalone native v6.1.0 build (sfc_referee.py v610)
     run on the same PRE.
Case c3 (DA-cycled start) re-runs c2's captured PRE with restart=false, cycling=true: native 44c0849 itself
segfaults in RRTMG SW (taumol_sw) at the first step of a do_DAcycling start, before the surface layer is called,
so no whole-model cycled capture exists.  Native PRE is the only numerical input; POST is opened only to compare.
Usage: WOOF_HEX_NUMERICS=strict sfc_replay.py LANE OUT.json [--timing N]  (PYTHONPATH=src:tests)
       WOOF_HEX_NUMERICS=fast sfc_replay.py LANE OUT.json --timing N --fast-timing   (speed only)
"""
import ctypes, hashlib, json, os, sys
from pathlib import Path
import numpy as np

lane = Path(sys.argv[1]); out = Path(sys.argv[2])
timing = int(sys.argv[sys.argv.index('--timing')+1]) if '--timing' in sys.argv else 0
fast = '--fast-timing' in sys.argv   # speed only: fast numerics is not bitwise to native and is not judged here
assert os.environ.get('WOOF_HEX_NUMERICS') == ('fast' if fast else 'strict')
from rrfs_driver_dump_replay import Records
import cupy as cp
from woof.hex.mpasphys.rrfs import sf_mynnsfc_v600 as v600, sf_mynnsfc_v610 as v610

codec = Records(Path(os.environ['RRFS_CAPTURE_CODEC']))   # the A3 Rust record codec (bridge.so)
F = v600.FIELDS
PROFILE = ('u3d', 'v3d', 't3d', 'qv3d', 'p3d', 'dz8w', 'th3d', 'rho3d')
LSM = {'sf_noaa': 1, 'sf_noahmp': 2, 'sf_ruc': 3}
DOMAIN = ('ids', 'ide', 'jds', 'jde', 'kds', 'kde', 'ims', 'ime', 'jms', 'jme', 'kms', 'kme',
          'its', 'ite', 'jts', 'jte', 'kts', 'kte')


def library(version):
    path = lane/'referee'/version/'libmynnsfc.so'
    build = json.loads((path.parent/'build.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == build['library_sha256']
    lib = ctypes.CDLL(str(path))
    lib.native_mynnsfc_init()
    fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
    lib.native_mynnsfc_observed.argtypes = [fp, ip, ip, fp, ip, ctypes.c_float, ctypes.c_int64, ctypes.c_int,
                                            ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ip]
    return lib, build


native = {v: library(v) for v in ('v600', 'v610')}


def load_call(capture, prefix):
    pre, post, present = {}, {}, {}
    for path in capture.glob(prefix+'.pre.*.bin'):
        name = path.name[len(prefix+'.pre.'):-4]
        if name.endswith('.present'):
            present[name[:-8]] = bool(np.asarray(codec.load(path)[0]).item())
        else:
            pre[name] = codec.load(path)[0]
    for path in capture.glob(prefix+'.post.*.bin'):
        name = path.name[len(prefix+'.post.'):-4]
        if not name.endswith('.present'):
            post[name] = codec.load(path)[0]
    return pre, post, present


def inputs(pre, present, overrides):
    scalar = lambda k: int(np.asarray(pre[k]).item())
    n = scalar('ime') - scalar('ims') + 1
    levels = scalar('kme') - scalar('kms') + 1
    assert (scalar('its'), scalar('ite'), scalar('jts'), scalar('jte')) == (scalar('ims'), scalar('ime'), 1, 1)
    f = np.zeros((n, len(F)), np.float32, order='F')
    prof = np.zeros((n, levels, 1, 9), np.float32, order='F')
    for k, name in enumerate(PROFILE):
        prof[:, :, 0, k] = np.asarray(pre[name], np.float32).reshape((n, levels), order='F')
    for i, name in enumerate(F):
        if name in ('u3d2', 'v3d2', 'dz8w2'):
            f[:, i] = prof[:, 1, 0, PROFILE.index(name[:-1])]
        elif name in PROFILE:
            f[:, i] = prof[:, 0, 0, PROFILE.index(name)]
        elif name in pre:
            f[:, i] = np.asarray(pre[name], np.float32).reshape(n, order='F')
    values = {}
    for name in v600.OPTIONS:
        if name == 'flagc_lsm':
            values[name] = LSM.get(str(pre[name]).strip(), 4) if present.get(name) else 4
        elif name in pre and present.get(name, True):
            values[name] = int(np.asarray(pre[name]).item())
        else:
            values[name] = 1 if name == 'flag_iter' else 0
    values.update(overrides)
    options = np.asarray([values[k] for k in v600.OPTIONS], np.int32)
    mask = sum(1 << i for i, name in enumerate(v600.OPTIONAL) if present.get(name, False))
    domains = np.asarray([scalar(k) for k in DOMAIN], np.int32)
    threshold = np.float32(np.asarray(pre['xice_threshold']).item())
    return n, levels, f, prof, options, mask, domains, threshold


def run_native(version, n, levels, f, prof, options, mask, domains, threshold):
    lib, _ = native[version]
    f, prof = f.copy(order='F'), prof.copy(order='F')
    veg = np.ones(n, np.int32)
    text = np.zeros(512, np.uint8)
    flag = ctypes.c_int()
    fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
    lib.native_mynnsfc_observed(f.ctypes.data_as(fp), options.ctypes.data_as(ip), veg.ctypes.data_as(ip),
        prof.ctypes.data_as(fp), domains.ctypes.data_as(ip), threshold, mask, n, levels,
        text.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.byref(flag))
    return f, prof, flag.value


def run_cuda(module, f, options, mask, threshold, repeat=0, **steps):
    fields = cp.asarray(np.ascontiguousarray(f.T))
    opts = cp.asarray(options)
    errors = module.run_packed(fields, opts, xice_threshold=float(threshold), present=mask, **steps)
    return cp.asnumpy(fields).T.copy(order='F'), cp.asnumpy(errors), []


def time_pair(f, options, mask, threshold, repeat):
    """Interleaved CUDA-event times of one column launch, v6.0.0 then v6.1.0, on fresh copies of the same PRE."""
    opts = cp.asarray(options)
    seconds = dict(v600=[], v610=[])
    for _ in range(repeat):
        for name, module in (('v600', v600), ('v610', v610)):
            work = cp.asarray(np.ascontiguousarray(f.T))
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            module.run_packed(work, opts, xice_threshold=float(threshold), present=mask)
            end.record()
            end.synchronize()
            seconds[name].append(cp.cuda.get_elapsed_time(start, end) / 1e3)
    return seconds


def words(a, b):
    a = np.ascontiguousarray(np.asarray(a))
    b = np.ascontiguousarray(np.asarray(b))
    assert a.size == b.size and a.dtype.itemsize == b.dtype.itemsize, (a.shape, b.shape)
    va = a.view('V'+str(a.dtype.itemsize)).reshape(-1)
    vb = b.view('V'+str(b.dtype.itemsize)).reshape(-1)
    return int(va.size), int(np.count_nonzero(va != vb))


def against_post(post, n, levels, f, prof, flag):
    rows = []
    for name, expected in sorted(post.items()):
        if name == 'errmsg':
            continue
        expected = np.asarray(expected)
        if name == 'errflg':
            actual = np.asarray(flag, np.int32)
        elif name in PROFILE:
            actual = prof[:, :, 0, PROFILE.index(name)].reshape(expected.shape, order='F')
        elif name in F:
            actual = f[:, F.index(name)].reshape(expected.shape, order='F')
        else:
            continue   # scalars and bounds are intent(in)
        w, d = words(np.asarray(actual, expected.dtype), expected)
        rows.append(dict(name=name, words=w, different=d))
    return rows


def fields_rows(names, a, b):
    rows = []
    for name in names:
        w, d = words(a[:, F.index(name)], b[:, F.index(name)])
        rows.append(dict(name=name, words=w, different=d))
    return rows


summary = dict(schema='rrfs-mynnsfc-v61-module-replay-v1', numerics='fast' if fast else 'strict',
    device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
    referees={v: native[v][1]['label'] for v in native},
    referee_library_sha256={v: native[v][1]['library_sha256'] for v in native},
    module_sha256={m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (v600, v610)},
    cases=[])
total = dict(witness=[0, 0], pinned=[0, 0], candidate=[0, 0], candidate_resident_door=[0, 0], pinned_resident_door=[0, 0])
cases = [('c1', 'c1', {}), ('c2', 'c2', {}), ('c3', 'c2', dict(restart=0, cycling=1))]
for label, source, overrides in cases:
    capture = lane/'captures'/source/'captures'
    receipt = json.loads((lane/'captures'/source/'capture-receipt.json').read_text())
    assert receipt['different'] == 0
    for name, digest in receipt['captures'].items():
        assert hashlib.sha256((capture/name).read_bytes()).hexdigest() == digest, name
    for call in (c for c in receipt['calls'] if c['phase'] == 'pre'):
        pre, post, present = load_call(capture, call['prefix'])
        n, levels, f, prof, options, mask, domains, threshold = inputs(pre, present, overrides)
        assert mask == v600.MPAS_PRESENT, (hex(mask), hex(v600.MPAS_PRESENT))
        row = dict(case=label, prefix=call['prefix'], step=call['step'], call=call['call'], cells=n,
                   options=dict(zip(v600.OPTIONS, map(int, options))))
        nf, nprof, nflag = run_native('v600', n, levels, f, prof, options, mask, domains, threshold)
        if not overrides:
            row['witness'] = against_post(post, n, levels, nf, nprof, nflag)
        gf, gerr, _ = run_cuda(v600, f, options, mask, threshold)
        row['pinned'] = (against_post(post, n, levels, gf, prof, int(gerr.max(initial=0))) if not overrides
                         else fields_rows(v600.OUTPUTS, gf, nf))
        cf, cprof, cflag = run_native('v610', n, levels, f, prof, options, mask, domains, threshold)
        hf, herr, _ = run_cuda(v610, f, options, mask, threshold)
        row['candidate'] = fields_rows(v610.OUTPUTS, hf, cf) + [dict(name='errflg', words=n,
            different=int(np.count_nonzero(herr != cflag)))]
        # Resident door (resident_sfclay_rrfs.ResidentSfclaySlot): options carry itimestep=0, initflag=0 and the
        # time-varying integers travel as the column launch's step/init arguments.
        resident = options.copy()
        resident[v600.OPTIONS.index('itimestep')] = 0
        resident[v600.OPTIONS.index('initflag')] = 0
        steps = dict(itimestep=int(options[v600.OPTIONS.index('itimestep')]),
                     initflag=int(options[v600.OPTIONS.index('initflag')]))
        rf, rerr, _ = run_cuda(v610, f, resident, mask, threshold, **steps)
        row['candidate_resident_door'] = fields_rows(v610.OUTPUTS, rf, cf)
        pf, perr, _ = run_cuda(v600, f, resident, mask, threshold, **steps)
        row['pinned_resident_door'] = fields_rows(v600.OUTPUTS, pf, gf)
        row['v610_vs_v600_native_outputs'] = fields_rows(v600.OUTPUTS, cf, nf)
        if timing:
            row['seconds'] = time_pair(f, options, mask, threshold, timing)
        for key in total:
            if key in row:
                total[key][0] += sum(r['words'] for r in row[key])
                total[key][1] += sum(r['different'] for r in row[key])
        summary['cases'].append(row)
        brief = {k: sum(r['different'] for r in row[k]) for k in total if k in row}
        changed = {r['name']: r['different'] for r in row['v610_vs_v600_native_outputs'] if r['different']}
        print(label, call['prefix'][-34:], 'differ', brief, 'v6.1-v6.0 changed', changed, flush=True)
summary['total'] = {k: dict(words=w, different=d) for k, (w, d) in total.items()}
out.write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary['total']))
raise SystemExit(0 if fast else int(any(d for _, d in total.values())))
