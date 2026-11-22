#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1 (ad/phys) recorder front end: the recorder's observe_slotted_census.py, unchanged, plus one read-only capture at
every outer boundary (the same source lines binder_mark_step marks): the TEMPO scheduling words.

TEMPO orders its columns heaviest first from the previous call's per-column cycle counts (tempo_v303 _ORDER_WORK:
cost = clock64 cycles / 16, written by tempo_warp_steps; order = an atomic counting-sort scatter of cost; hist =
the bucket starts). They change which warp runs which column, never a model word ("scheduling only, no word
changes"), but they are not reproducible run to run, so a replay cannot rebuild the forward's values and the
carried allocations holding them (P1 pkg6: ids 414 and 417) can never hash equal. THE BREAKAGE THIS PREVENTS:
those two allocations failed the step hash in every replay. With the forward's own pointers and values recorded
here, the sealer (p1fix_package.py) declares exactly these arrays excluded residue (zeroed on both sides, every
other byte of the allocation still compared) and residue_witness.py compares two forwards to witness that
they differ run to run in the ordinary forward itself.

Writes OUT/tempo-sched/step-N.json: {key, device pointers, bytes, hex} for cost, order and hist of every entry.
Reads only (device-to-host copies after a full synchronize, the same synchronize binder_mark_step performs).
Usage: record_observer.py --observer-dir DIR --out OUT -- forecast ...
"""
import json
import linecache
import os
import sys


def start_profile(path):
    """AD_RECORD_PYPROF=FILE: an in-process sampling profile of the main thread (every 20 ms), for a separate profiling
    recording only (never a timed one): the box container has no ptrace, so py-spy cannot attach."""
    import atexit, collections, threading, time
    main_id = threading.main_thread().ident
    inclusive, exclusive, stacks = collections.Counter(), collections.Counter(), collections.Counter()
    samples = [0]

    def run():
        while True:
            time.sleep(0.02)
            frame = sys._current_frames().get(main_id)
            if frame is None:
                continue
            chain = []
            while frame is not None and len(chain) < 60:
                chain.append(f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_code.co_name}")
                frame = frame.f_back
            samples[0] += 1
            exclusive[chain[0]] += 1
            for name in set(chain):
                inclusive[name] += 1
            stacks[" <- ".join(chain[:8])] += 1
            if samples[0] % 1500 == 0:
                dump()

    def dump():
        with open(path, "w") as f:
            json.dump({"samples": samples[0], "interval_s": 0.02, "inclusive": inclusive.most_common(80),
                       "exclusive": exclusive.most_common(60), "stacks": stacks.most_common(40)}, f, indent=1)
    threading.Thread(target=run, daemon=True, name="ad-record-pyprof").start()
    atexit.register(dump)


def main():
    args = sys.argv[1:]
    if os.environ.get("AD_RECORD_PYPROF"):
        start_profile(os.environ["AD_RECORD_PYPROF"])
    obs_dir = args[args.index('--observer-dir') + 1]
    out = args[args.index('--out') + 1]
    sys.path.insert(0, obs_dir)
    import observe_slotted_census as sc
    owner = sc.owner
    # The same module wiring observe_slotted_census.py does under __main__.
    owner.deep.deep_discover = sc.slotted_discover
    owner.launches.sites = owner.sites_with_owners
    owner.callbacks.discover = owner.discover_with_owners
    owner.host.source_stack = owner.ruc.source_stack_with_radiation_counts
    owner.base.Observer = owner.launches.LaunchSiteObserver
    original = owner.ruc.boundary_ruc_and_existing_trace
    # P1_LW_CHUNK=N fixes the RRTMG-LW wrapper's column chunk (rrtmg_lw_v842._lw_wrapper_chunk, sized from memGetInfo).
    # P1_FIXED_FREE_MIB=N makes every forward reading of free device memory (cupy.cuda.runtime.memGetInfo: the LW
    # wrapper chunk, the MYNN-EDMF arena budget and hexcore's other readers) return N MiB free, so the two recordings of
    # one plan take the same memory-sized decisions on a shared card. THE BREAKAGE THIS PREVENTS: recordings of one
    # limited-area forecast on a shared card split the LW solver (step 1) and the MYNN-EDMF launch (steps 18-19)
    # differently, the launch streams no longer aligned and the sealer could not pair them (a 3 km storm
    # recording pair, 2026-11-07). Only batching changes (columns are independent); total is reported as the card's own. Unset: the
    # forward's own rule, unchanged.
    if os.environ.get('P1_FIXED_FREE_MIB'):
        import cupy as _cp
        _real = _cp.cuda.runtime.memGetInfo
        _free = int(os.environ['P1_FIXED_FREE_MIB']) << 20
        _cp.cuda.runtime.memGetInfo = lambda: (_free, _real()[1])
    if os.environ.get('P1_LW_CHUNK'):
        from woof.hex.mpasphys import rrtmg_lw_v842 as _lw
        _fixed = int(os.environ['P1_LW_CHUNK'])
        _lw._lw_wrapper_chunk = lambda ncol, nlayers: int(min(max(1, ncol), _fixed))
    target = os.path.join(out, 'tempo-sched')
    os.makedirs(target, exist_ok=True)

    def plain(key):   # dict keys of the TEMPO caches nest tuples (r18a attempt 3 died on int(tuple))
        if isinstance(key, (tuple, list)):
            return [plain(k) for k in key]
        return key if isinstance(key, (bool, int, float, str)) or key is None else (int(key) if hasattr(key, "__index__") else repr(key))

    def tempo_rows():
        """(key, name, array) of every TEMPO scheduling word array and per-warp scratch array: the rows capture writes
        and the rows register hands the recorder, from one list so the two are the same rows."""
        from woof.hex.mpasphys.rrfs import tempo_v303 as tempo
        rows = []
        for key, arrays in list(tempo._ORDER_WORK.items()):
            rows += [(key, name, array) for name, array in zip(('cost', 'order', 'hist'), arrays)]
        for key, arrays in list(tempo._WARP_WORK.items()):
            rows += [(key, name, array) for name, array in zip(('warp_states', 'warp_work', 'warp_blwork'), arrays)]
        return rows

    def capture(step):
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
        rows = []
        for key, name, array in tempo_rows():
            if name in ('cost', 'order', 'hist'):
                data = cp.asnumpy(array).tobytes()
                rows.append({'key': plain(key), 'name': name, 'pointer': int(array.data.ptr),
                             'bytes': len(data), 'hex': data.hex()})
            else:
                # Per-warp scratch (states, work, PBL work): pointers and sizes only, no bytes. The kernel zeroes work
                # itself and starts each column's state afresh; which warp ran which column follows the order above.
                rows.append({'key': plain(key), 'name': name, 'pointer': int(array.data.ptr),
                             'bytes': int(array.nbytes)})
        with open(os.path.join(target, f'step-{step}.json'), 'w') as f:
            json.dump({'schema': 'ad.binder.tempo-scheduling-words/v1', 'step': step, 'rows': rows}, f)

    # lean store: the same rows go to the recorder BEFORE it marks the boundary (binder_residue_rows), so it writes the
    # masked digest of each allocation holding them (BoundaryMaskedDigest) and keeps none of their bytes. Pointers and
    # sizes only: no device read, nothing the forward sees. A library without the symbol (an older recorder) is skipped.
    import ctypes
    try:
        _register = ctypes.CDLL(None).binder_residue_rows
        _register.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_uint]
        _register.restype = ctypes.c_int
    except AttributeError:
        _register = None

    def register():
        if _register is None:
            return
        rows = tempo_rows()
        pointers = (ctypes.c_ulonglong * max(1, len(rows)))(*[int(a.data.ptr) for _, _, a in rows])
        sizes = (ctypes.c_ulonglong * max(1, len(rows)))(*[int(a.nbytes) for _, _, a in rows])
        if _register(pointers, sizes, len(rows)):
            raise RuntimeError('recorder: binder_residue_rows failed')

    def boundary_trace(observer):
        inner = original(observer)

        def trace(frame, event, arg):
            local = inner(frame, event, arg)
            if (local is None or frame.f_code.co_name != '_run_native_dynamics' or
                    not frame.f_code.co_filename.endswith('/hexcore/mpasphys/runtime_v842.py')):
                return local   # the observer's own tracers for other functions, untouched
            lines = linecache.getlines(frame.f_code.co_filename)
            loop = next(i for i, line in enumerate(lines) if 'for step in range(steps+1):' in line)
            before = next(i + 1 for i in range(loop + 1, len(lines))
                          if 'if step and nest_look is not None:' in lines[i])
            finish = next(i + 1 for i in range(loop + 1, len(lines))
                          if lines[i].strip() == 'deferred_checks.finish()')
            state = {'local': local}

            def mine(current, kind, value):
                if kind == 'line' and current.f_lineno in (before, finish) and current.f_locals.get('step', 0):
                    register()   # before the inner trace marks the boundary
                result = state['local'](current, kind, value)
                if result is not None:
                    state['local'] = result
                if kind == 'line' and current.f_lineno == before:
                    step = current.f_locals.get('step', 0)
                    if step:
                        capture(step - 1)
                elif kind == 'line' and current.f_lineno == finish:
                    step = current.f_locals.get('step', 0)
                    if step:
                        capture(step)
                return mine
            return mine
        return trace

    owner.base.boundary_trace = boundary_trace
    forecast = args[args.index('--') + 1:]
    sys.argv = [os.path.join(obs_dir, 'observe_slotted_census.py'), '--out', out, '--'] + forecast
    return owner.base.main()


if __name__ == '__main__':
    sys.exit(main())
