"""Full resident-state and argument A/B against the base RUC slot."""
import importlib.util
import os


def install():
    from woof.hex.mpasphys.rrfs import resident_ruc_rrfs as rr
    from ab_launch import _ab, _compare
    spec = importlib.util.spec_from_file_location(
        'hexcore.mpasphys.rrfs._ruc_glue_base', os.environ['GLUE_BASE_RUC'])
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    original = rr.ResidentRucSlot.run
    slots = {}

    def run(slot, call):
        if call.itimestep not in (1, 12, 24):
            return original(slot, call)
        old = slots.get(id(slot))
        if old is None:
            old = slots[id(slot)] = base.ResidentRucSlot(slot.provider, call, slot.entry)
        # Materialize authorities before the snapshot, without executing a
        # scheme. Each run rebuilds its pointer table if restoration moves it.
        old._resolve(); slot._resolve()
        expected = {}
        def previous():
            result = old.run(call)
            expected.update({('argument', name): value.copy()
                             for name, value in result['arguments'].items()})
            expected[('errors',)] = result['errors'].copy()
        def current():
            result = original(slot, call)
            actual = {('argument', name): value for name, value in result['arguments'].items()}
            actual[('errors',)] = result['errors']
            _compare(slot.entry + ' arguments', call.itimestep, expected, actual)
            return result
        result = _ab(slot.entry + ' resident state', slot.provider, call, previous, current,
                     lambda: (slot.gather, slot.scatter))
        if call.itimestep == 24:
            import cupy as cp
            import json
            import statistics
            from woof.hex.mpasphys.rrfs import glue_plan as gp
            from tiled_plans import run as run_tiled
            rows = []
            for arm, plan in (('base', old.gather), ('tip', slot.gather)):
                kernel = gp._kernel(cp, False)
                values = []
                for _ in range(9):
                    a, b = cp.cuda.Event(), cp.cuda.Event()
                    a.record(); kernel(plan._grid, (gp._THREADS,), (plan._device,)); b.record(); b.synchronize()
                    values.append(cp.cuda.get_elapsed_time(a, b))
                rows.append(dict(arm=arm, entry=slot.entry, ms=statistics.median(values),
                    bytes=sum(int(r['count'])*(4 if r['mode'] else 8) for r in plan._rows)))
                run_tiled(plan, kernel, gp, cp, call.itimestep, arm)
            with open(os.environ['GLUE_AB_OUT'] + '.ruc-timings.jsonl', 'a') as stream:
                stream.write(json.dumps(rows) + '\n')
        return result
    rr.ResidentRucSlot.run = run
