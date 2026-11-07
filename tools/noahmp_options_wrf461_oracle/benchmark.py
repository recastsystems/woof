#!/usr/bin/env python3
"""Bounded GPU stage timing with constructed inputs, no forecast speed claim."""
from __future__ import annotations

import json
import statistics

import cupy as cp
import numpy as np

from woof import wrf_exact
from woof.core.noahmp_slab_libm import root_fraction_slabs
from woof.core.noahmp_energy_slab import energy_seg2


def main():
    n = 65536

    def rows(values):
        return cp.broadcast_to(cp.asarray(values, dtype=cp.float32),
                               (n, len(values))).copy()

    state = dict(nroot=cp.arange(n, dtype=cp.int32) % 4 + 1,
                 sh2o=rows([.08, .15, .25, .35]),
                 smcwlt=rows([.04, .06, .1, .15]),
                 smcref=rows([.15, .2, .3, .4]),
                 psisat=rows([.035, .1, .3, .6]),
                 smcmax=rows([.35, .4, .45, .5]),
                 bexp=rows([4.05, 4.9, 7.12, 11.55]),
                 zsoil=rows([-.1, -.4, -1., -2.]),
                 dzsnso=rows([0., 0., 0., .1, .3, .6, 1.]))
    state.update({name: cp.full(n, value, dtype=cp.float32) for name, value in
                  dict(elai=2., esai=.1, eg=.95, snow_emis=.98, rsurf_exp=5.,
                       rsurf_snow=5000., tg=278., tv=279., sfcprs=90000.,
                       fveg=.7, snowh=.1).items()})
    state['urban_flag'] = cp.zeros(n, dtype=cp.bool_)
    geometry = dict(fsno=cp.full(n, .5, dtype=cp.float32),
                    veg=cp.ones(n, dtype=cp.bool_))

    def measure(call):
        # Compile and warm once. Record launches and allocations without a
        # host synchronization between repetitions; read event times at end.
        last = call()
        cp.cuda.get_current_stream().synchronize()
        pairs = []
        for _ in range(20):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            last = call()
            end.record()
            pairs.append((start, end))
        pairs[-1][1].synchronize()
        samples = [float(cp.cuda.get_elapsed_time(a, b)) for a, b in pairs]
        arrays = list(last.values()) if isinstance(last, dict) else list(last)
        finite = all(bool(cp.isfinite(a).all()) for a in arrays)
        return dict(median_ms=statistics.median(samples), min_ms=min(samples),
                    max_ms=max(samples), samples_ms=samples, finite=finite)

    btran = {str(option): measure(lambda option=option: root_fraction_slabs(
        state, 3, 4, 1.e-6, opt_btr=option)) for option in (1, 2, 3)}
    resistance = {str(option): measure(lambda option=option: energy_seg2(
        state, geometry, opt_rsf=option)) for option in (1, 2, 3, 4)}
    for group in (btran, resistance):
        for row in group.values():
            row['ratio_to_option_1'] = row['median_ms'] / group['1']['median_ms']
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = device['name']
    print(json.dumps(dict(scope='constructed stage timing, shared GPU; not forecast steps',
                          columns=n, repetitions=20, wrf_exact=wrf_exact.ENABLED,
                          device=name.decode() if isinstance(name, bytes) else name,
                          btran=btran, energy_seg2_rsf=resistance), indent=2))


if __name__ == '__main__':
    main()
