"""Real-call slot replay: tiled independent columns, every pool word, nine timings."""
import json
import math
import os
import statistics
import time
from dataclasses import replace
from types import SimpleNamespace


def benchmark(provider, call, run):
    import cupy as cp
    from woof.hex.mpasphys.rrfs import sf_mynnsfc_v600 as sfc
    from woof.hex.mpasphys.rrfs.resident_sfclay_rrfs import ResidentSfclaySlot, _PROFILES

    class Pools(dict):
        pass

    source = {
        'diag_physics': {n: provider['diag_physics'][n] for n in dict.fromkeys(
            (*sfc._DIAG_INPUT, 'hpbl', *(sfc._DIAG_NAMES.get(n, n) for n in sfc._WRITE)))},
        'sfc_input': {n: provider['sfc_input'][n] for n in dict.fromkeys((*sfc._SURFACE.values(), 'sst'))},
        'atm_input': {n: provider['atm_input'][n][:2] for n in _PROFILES},
        'prepared': {n + '_sea': provider['prepared'][n + '_sea'] for n in sfc._SEA_HANDOFF},
        'mesh': {'meshDensity': provider['mesh']['meshDensity']},
    }
    source['atm_input']['psfcpa'] = provider['atm_input']['psfcpa']
    for factor in (1, math.ceil(100000 / provider.session.layout.dimensions['nCellsSolve'])):
        n = factor * provider.session.layout.dimensions['nCellsSolve']
        p = Pools({pool: {name: cp.tile(value, (1,) * (value.ndim - 1) + (factor,))
                         for name, value in fields.items()} for pool, fields in source.items()})
        p.cp = cp
        p.session = SimpleNamespace(layout=SimpleNamespace(dimensions={'nCellsSolve': n}))
        p['configs'] = provider['configs']
        p['scalar_state'] = {'xice_threshold': provider['scalar_state']['xice_threshold']}
        arrays = {(pool, name): value for pool, fields in p.items() if pool in source
                  for name, value in fields.items()}
        initial = {key: a.copy() for key, a in arrays.items()}
        tiled_bad = sum(int(cp.count_nonzero(
            value.view(cp.uint32).reshape(-1, factor, source[pool][name].shape[-1]) !=
            source[pool][name].view(cp.uint32).reshape(-1, 1, source[pool][name].shape[-1])))
            for (pool, name), value in initial.items())
        c = replace(call, pools=p)
        slot = ResidentSfclaySlot(p, c)

        def previous():
            return sfc.driver(p['mesh'], p['diag_physics'], p['sfc_input'], p['atm_input'],
                itimestep=c.itimestep, len_disp=p['configs']['config_len_disp'],
                xice_threshold=p['scalar_state']['xice_threshold'],
                config_frac_seaice=c.options['config_frac_seaice'],
                config_do_restart=p['configs']['config_do_restart'],
                config_do_DAcycling=p['configs']['config_do_DAcycling'],
                config_lsm_scheme=c.options['config_lsm_scheme'],
                config_mynn_sfcflux_land=c.options['config_mynn_sfcflux_land'],
                config_mynn_sfcflux_water=c.options['config_mynn_sfcflux_water'], sea_state=p['prepared'])

        def restore():
            for key, array in arrays.items():
                array[...] = initial[key]
            cp.cuda.get_current_stream().synchronize()

        restore()
        previous()
        old = {key: a.copy() for key, a in arrays.items()}
        restore()
        run(slot, c)
        different = sum(int(cp.count_nonzero(old[key].view(cp.uint32) != a.view(cp.uint32)))
                        for key, a in arrays.items())
        for plan in slot.plans:
            plan.check_disjoint()
        result = dict(columns=n, itimestep=call.itimestep, tiled_different=tiled_bad,
                      compared_words=sum(a.size for a in arrays.values()), different=different,
                      race_free=True, samples=9)
        for name, arm in (('previous', previous), ('fused', lambda: run(slot, c))):
            gpu, host = [], []
            for _ in range(9):
                restore()
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record()
                t = time.perf_counter()
                arm()
                host.append(1000 * (time.perf_counter() - t))
                end.record()
                end.synchronize()
                gpu.append(cp.cuda.get_elapsed_time(start, end))
            result[name] = dict(event_median_ms=statistics.median(gpu), host_median_ms=statistics.median(host),
                                event_ms_per_100000=statistics.median(gpu) * 100000 / n)
        with open(os.environ['SFCLAY_TIMING_OUT'], 'a') as f:
            f.write(json.dumps(result) + '\n')
        print('SFCLAY TIMING ' + json.dumps(result), flush=True)
        assert different == tiled_bad == 0
