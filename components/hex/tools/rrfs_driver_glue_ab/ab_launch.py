"""Real-door A/B for resident RRFS driver copies, tendency sources and reset.

Run the shared hrrrv5 recipe with this LAUNCHER, DRIVER_BASE pointing to the
base tree, GLUE_AB_OUT to a JSONL receipt, and DRIVER_AB=1 (default). At steps
1/5/6/12 compare every resident word, allocation tail, returned tendency and
transport-state array. Cached array identities survive restoration. Step12
also records nine warm host/CUDA-event pairs, followed by separate untimed
CUPTI arms. DRIVER_AB=0 installs only the owner's three dispatch hooks for
the whole-run smoke. No forecast arithmetic is mocked.
"""
import json
import os
import sys
import time
import statistics
from dataclasses import replace, fields

STEPS = {int(x) for x in os.environ.get("GLUE_AB_STEPS", "1,5,6,12").split(",") if x}
OUT = os.environ["GLUE_AB_OUT"]


def _state(provider):
    import cupy as cp
    facade = provider.native
    arrays = {}
    for key, (host, packed, selection) in provider.resident_buffers.items():
        if isinstance(packed, cp.ndarray):
            arrays[("resident",) + tuple(str(k) for k in key)] = packed
    for attr in ("_field_rest", "_variable_whole", "device_owned"):
        for key, value in getattr(facade, attr).items():
            if isinstance(value, cp.ndarray):
                arrays[(attr,) + (tuple(str(k) for k in key) if isinstance(key, tuple) else (str(key),))] = value
    glue = provider.session.resident_glue
    coupling = glue.coupling
    transport = getattr(coupling, 'transport', None)
    if transport is not None:
        state, saved, outer, rate = transport
        for prefix, obj in (('state', state), ('saved', saved), ('outer', outer)):
            for row in fields(obj):
                key, value = row.name, getattr(obj, row.name)
                if isinstance(value, cp.ndarray):
                    arrays[('transport', prefix, key)] = value
        if isinstance(rate, cp.ndarray):
            arrays[('transport', 'rate')] = rate
    return arrays


def _snapshot(provider):
    return {key: value.copy() for key, value in _state(provider).items()}


def _restore(provider, snap, members):
    import cupy as cp
    facade = provider.native
    for key, value in _state(provider).items():
        if key in snap and snap[key].shape == value.shape:
            value[...] = snap[key]
    # Buffers the first arm made resident (first use of a field in the run) start from the host words they were
    # uploaded from, as they would have for the second arm; the host storage is not written by either arm.
    for key, (host, packed, selection) in provider.resident_buffers.items():
        name = ("resident",) + tuple(str(k) for k in key)
        if name in snap or not isinstance(packed, cp.ndarray):
            continue
        if key[0] == "prepared":
            count = provider.session.layout.dimensions["nCellsSolve"]
            view = (host[:count, :, 0, :].transpose(2, 1, 0) if host.ndim == 4 else
                    host[:count, :, 0].T if host.ndim == 3 else host[:count, 0])
            packed[...] = cp.asarray(view)
        else:
            selected = host[selection]
            packed[...] = cp.asarray(selected).reshape(packed.shape)
    for attr in ("_field_rest", "_variable_whole", "device_owned"):
        table = getattr(facade, attr)
        for key in list(table):
            if key not in members[attr]:
                del table[key]
        for key, value in members[attr].items():
            original, words = value
            original[...] = words
            table[key] = original


def _words(a):
    import cupy as cp
    a = cp.ascontiguousarray(a)
    return a.view(cp.uint32) if a.dtype.itemsize == 4 and a.size else a.view(cp.uint8)


def _compare(kind, step, a, b):
    import cupy as cp
    only_a = sorted(map(list, set(a) - set(b)))
    only_b = sorted(map(list, set(b) - set(a)))
    words = different = 0
    worst = []
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            worst.append({"array": list(key), "shape": [list(x.shape), list(y.shape)]})
            continue
        wx, wy = _words(x), _words(y)
        n = int((wx != wy).sum())
        words += int(wx.size)
        different += n
        if n:
            worst.append({"array": list(key), "different": n, "words": int(wx.size)})
    record = {"slot": kind, "itimestep": step, "arrays": len(set(a) & set(b)), "words": words,
              "different": different, "differing_arrays": worst[:40], "only_previous": only_a,
              "only_fused": only_b}
    with open(OUT, "a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"GLUE A/B {kind} step {step}: {different} of {words} words different, "
          f"{len(worst)} arrays, only-previous {len(only_a)}, only-fused {len(only_b)}", flush=True)


def _check_plans(kind, step, plans):
    checked = []
    for plan in plans:
        try:
            plan.check_disjoint()
            checked.append({"plan": plan.name, "ops": len(plan), "race_free": True})
        except RuntimeError as error:
            checked.append({"plan": plan.name, "ops": len(plan), "race_free": False, "error": str(error)})
    with open(OUT, "a") as stream:
        stream.write(json.dumps({"slot": kind, "itimestep": step, "plans": checked}) + "\n")
    print(f"GLUE PLANS {kind} step {step}: " + ", ".join(
        f"{c['plan']} {c['ops']} ops {'race-free' if c['race_free'] else 'RACE ' + c['error']}" for c in checked),
        flush=True)


def _ab(kind, provider, call, run_previous, run_fused, plans):
    import cupy as cp
    if call.itimestep not in STEPS:
        return run_fused()
    facade = provider.native
    cp.cuda.get_current_stream().synchronize()
    snap = _snapshot(provider)
    members = {attr: {k: (v, v.copy()) for k, v in getattr(facade, attr).items() if isinstance(v, cp.ndarray)}
               for attr in ("_field_rest", "_variable_whole", "device_owned")}
    old_result = run_previous()
    cp.cuda.get_current_stream().synchronize()
    previous = _snapshot(provider)
    if old_result is not None and hasattr(old_result, 'rho_theta'):
        previous.update({('returned', k): v.copy() for k, v in vars(old_result).items() if isinstance(v, cp.ndarray)})
    _restore(provider, snap, members)
    cp.cuda.get_current_stream().synchronize()
    result = run_fused()
    cp.cuda.get_current_stream().synchronize()
    fused = _snapshot(provider)
    if result is not None and hasattr(result, 'rho_theta'):
        fused.update({('returned', k): v.copy() for k, v in vars(result).items() if isinstance(v, cp.ndarray)})
    _compare(kind, call.itimestep, previous, fused)
    _check_plans(kind, call.itimestep, plans())
    if call.itimestep == 12 and os.environ.get('DRIVER_TIME', '1') == '1':
        samples = {'previous': [], 'fused': []}
        for repeat in range(9):
            for label, run in (('previous', run_previous), ('fused', run_fused)):
                _restore(provider, snap, members)
                cp.cuda.get_current_stream().synchronize()
                start, stop = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                begun = time.perf_counter()
                run()
                host_ms = (time.perf_counter()-begun)*1000
                stop.record()
                stop.synchronize()
                samples[label].append({'host_ms': host_ms, 'event_ms': cp.cuda.get_elapsed_time(start, stop)})
        record = {'slot': kind, 'itimestep': call.itimestep, 'timing_samples': samples,
                  'timing_median': {label: {metric: statistics.median(row[metric] for row in rows)
                      for metric in ('host_ms', 'event_ms')} for label, rows in samples.items()}}
        with open(OUT, 'a') as stream:
            stream.write(json.dumps(record)+'\n')
        _restore(provider, snap, members)
        result = run_fused()
    if os.environ.get('DRIVER_AUDIT', '1') == '1' and (call.itimestep == 12 or (kind == 'atm_reset_diagnostics' and call.itimestep == 5)):
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent / 'audit'))
        from audit import Audit
        for label, run in (('previous', run_previous), ('fused', run_fused)):
            _restore(provider, snap, members)
            cp.cuda.get_current_stream().synchronize()
            audit = Audit(os.environ['DRIVER_AUDIT_CUPTI']).start()
            try:
                run()
            finally:
                audit.stop(OUT + '.' + kind + '.' + label + '.audit.json')
        _restore(provider, snap, members)
        result = run_fused()
    del snap, previous, fused
    return result


def install():
    from woof.hex.mpasphys.rrfs import resident_rrfs as rr, driver_rrfs as dr
    from woof.hex.mpasphys.runtime_v842 import ResidentNativeCoupling
    import importlib.util
    spec = importlib.util.spec_from_file_location('hexcore.mpasphys.rrfs._driver_reference',
        os.environ['DRIVER_BASE'] + '/src/hexcore/mpasphys/rrfs/driver_rrfs.py')
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    old_prepare, old_tend = rr.ResidentPhysicsGlue.prepare, rr.ResidentPhysicsGlue.get_tend
    old_sync = ResidentNativeCoupling.synchronize
    old_transport = ResidentNativeCoupling.prepare_transport
    ab = os.environ.get('DRIVER_AB', '1') == '1'

    def prepare(self, call):
        if not hasattr(self, '_driver_preparation'):
            self._driver_preparation = dr.ResidentDriverPreparation(self, call)
        if not ab:
            return dr.resident_prepare(self, call)
        return _ab('MPAS_to_physics', self.provider, call, lambda: old_prepare(self, call),
                   lambda: dr.resident_prepare(self, call), lambda: (self._driver_preparation.plan,))

    def tend(self, call):
        if not hasattr(self, '_driver_tendencies'):
            self._driver_tendencies = dr.ResidentDriverTendencies(self)
        if not ab:
            return dr.resident_get_tend(self, call)
        return _ab('physics_get_tend', self.provider, call, lambda: old_tend(self, call),
                   lambda: dr.resident_get_tend(self, call), lambda: (self._driver_tendencies.plan,))

    def sync(self, atmosphere, *, level=1, prepared_columns=False):
        if getattr(self, '_driver_fused_transport', False) and level == 2 and not prepared_columns:
            return dr.resident_prepare_transport(self, atmosphere)
        return old_sync(self, atmosphere, level=level, prepared_columns=prepared_columns)

    def transport(self, call):
        def run(fused):
            self._driver_fused_transport = fused
            try:
                return old_transport(self, call)
            finally:
                self._driver_fused_transport = False
        if not ab:
            before = getattr(self, '_rrfs_transport_copy', None)
            result = run(True)
            after = getattr(self, '_rrfs_transport_copy', None)
            print('DRIVER PLAN ' + json.dumps({'step': call.itimestep,
                  'layout_rebuilt': after is not before, 'device_table': False}), flush=True)
            return result
        return _ab('post_transport_prepare', self.session.device_pools, call, lambda: run(False),
                   lambda: run(True), lambda: (dr.resident_transport_copy_plan(self),))

    old_call = dr.PhysicsDriver._call
    def call(self, operation, step, time, flags, pools, *, slot=None):
        if operation != 'atm_reset_diagnostics' or not ab:
            return old_call(self, operation, step, time, flags, pools, slot=slot)
        from types import SimpleNamespace
        provider = pools.device_pools
        for name in ('refl10cm_1km_max', 'max_hail_diameter_sfc_acc', 'max_hail_diameter_column_acc'):
            if ('diag_physics', name) in pools.native._fields:
                provider.native.get_field('diag_physics', name)
        return _ab(operation, provider, SimpleNamespace(itimestep=step),
            lambda: ref.reset_native_diagnostics(pools.native, pools),
            lambda: old_call(self, operation, step, time, flags, pools, slot=slot),
            lambda: (pools._rrfs_reset_plan,))

    rr.ResidentPhysicsGlue.prepare = prepare
    rr.ResidentPhysicsGlue.get_tend = tend
    ResidentNativeCoupling.synchronize = sync
    ResidentNativeCoupling.prepare_transport = transport
    dr.PhysicsDriver._call = call


if __name__ == '__main__':
    import subprocess
    with open(OUT + '.neighbours.csv', 'w') as stream:
        subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], stdout=stream, timeout=10)
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
