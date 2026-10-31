"""Measurement launcher (not product code): the hex CLI with an in-process A/B of the RRFS resident slot glue.

At the chosen steps (GLUE_AB_STEPS, itimestep values), every resident MYNN-SFC call runs twice from the same device state taken from the live forecast (real inputs): first
the previous glue (resident_sfclay_rrfs.FUSED off, previous glue at 7876465), then the
fused glue. Every resident device array (the provider's resident pool and module buffers, the facade's allocation
tails and device-owned arrays) is snapshotted before, restored before the second run, and compared word for word
(32-bit words, NaN payloads included) after each run. The forecast then continues from the fused run's state.
After each compared fused call, every fused copy plan of that slot is checked race-free (glue_plan.CopyPlan.
check_disjoint: no word written twice in one launch, no word read by one op and written by another).
Receipt: one JSON line per compared call in GLUE_AB_OUT.
"""
import json
import os
import sys

STEPS = {int(x) for x in os.environ.get("GLUE_AB_STEPS", "1,12,24").split(",") if x}
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
    return arrays


def _snapshot(provider):
    return {key: value.copy() for key, value in _state(provider).items()}


def _restore(provider, snap, members):
    import cupy as cp
    facade = provider.native
    for key, value in _state(provider).items():
        if key[0] == "resident" and key in snap and snap[key].shape == value.shape:
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
            table[key] = value.copy()


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
    if different or worst or only_a or only_b:
        raise AssertionError("MYNN-SFC resident state differs")


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
    if not all(c["race_free"] for c in checked):
        raise AssertionError("MYNN-SFC copy plan races")


def _ab(kind, provider, call, run_previous, run_fused, plans):
    import cupy as cp
    if call.itimestep not in STEPS:
        return run_fused()
    facade = provider.native
    cp.cuda.get_current_stream().synchronize()
    snap = _snapshot(provider)
    members = {attr: {k: v.copy() for k, v in getattr(facade, attr).items() if isinstance(v, cp.ndarray)}
               for attr in ("_field_rest", "_variable_whole", "device_owned")}
    if os.environ.get('SFCLAY_NVTX'):
        cp.cuda.nvtx.RangePush('sfclay:previous')
    try:
        run_previous()
    finally:
        if os.environ.get('SFCLAY_NVTX'):
            cp.cuda.nvtx.RangePop()
    cp.cuda.get_current_stream().synchronize()
    previous = _snapshot(provider)
    _restore(provider, snap, members)
    cp.cuda.get_current_stream().synchronize()
    if os.environ.get('SFCLAY_NVTX'):
        cp.cuda.nvtx.RangePush('sfclay:fused')
    try:
        result = run_fused()
    finally:
        if os.environ.get('SFCLAY_NVTX'):
            cp.cuda.nvtx.RangePop()
    cp.cuda.get_current_stream().synchronize()
    fused = _snapshot(provider)
    _compare(kind, call.itimestep, previous, fused)
    _check_plans(kind, call.itimestep, plans())
    del snap, previous, fused
    return result


def install():
    from woof.hex.mpasphys.rrfs import resident_sfclay_rrfs as rs
    from woof.hex.mpasphys.rrfs import sf_mynnsfc_v600 as sfc
    run = rs.ResidentSfclaySlot.run

    def surface(self, call):
        def previous():
            rs.FUSED = False
            try:
                return sfc.launch_slot(call)
            finally:
                rs.FUSED = True
        result = _ab("sfclay", self.provider, call, previous, lambda: run(self, call), lambda: self.plans)
        if call.itimestep == 24 and os.environ.get('SFCLAY_TIMING_OUT'):
            from time_slot import benchmark
            benchmark(self.provider, call, run)
        return result

    rs.ResidentSfclaySlot.run = surface


if __name__ == "__main__":
    install()
    from woof.hex import cli
    a = cli.build_parser().parse_args(sys.argv[1:])
    cli._select_numerics(a)
    sys.exit(int(a.handler(a)))
