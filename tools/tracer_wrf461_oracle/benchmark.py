"""Fast-mode stage timing on the real forecast's parent-grid dimensions.

Timing is a module check, not another forecast. CUDA events bracket the
production calls; synchronization happens after the timed batch only.
"""
from pathlib import Path
import json
import sys
import cupy as cp
import numpy as np

from woof.config import RunConfig
from woof.core.grid import make_base_state, make_vertical_coord
from woof.core.moist import init_moist_balanced, moist_species, advance_scalars_stage
from woof.core.dycore import _save_time_t
from woof.core.mynn_scalar_mix_gpu import scalar_pblmix_columns_cuda
from woof.wrf_exact import ENABLED


def timed(call, repeats=10):
    call()
    call()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / repeats


def transport(*, tracer, mp):
    cfg = RunConfig(nx=73, ny=60, nz=44, dx=30000., dy=30000., ztop=20000.,
                    dt=180., run_seconds=180., moist=True, mp_physics=mp,
                    tracer_opt=2 if tracer else 0)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298. + .003*np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                               lambda z: .005*np.exp(-np.asarray(z)/2400.))
    for name in moist_species(state):
        getattr(state, name)[0, 20:30, 25:35] = cp.float32(.1)
    _save_time_t(state)
    mass = state.c1h[:, None, None]*state.total_mu()[None] + state.c2h[:, None, None]
    ru = cp.empty_like(state.u); ru[...] = mass[:, :, :1]*cp.float32(5.)
    rv = cp.empty_like(state.v); rv[...] = mass[:, :1, :]*cp.float32(.5)
    ww = cp.zeros_like(state.w)
    result = {}
    for final in (False, True):
        result['pd' if final else 'ordinary'] = timed(lambda: advance_scalars_stage(
            state, cfg, ru, rv, ww, 180. if final else 60., final=final))
    result['species'] = len(moist_species(state))
    return result


def main(output):
    if ENABLED:
        raise ValueError('benchmark requires the default fast mode')
    baseline = transport(tracer=False, mp=9)
    passive = transport(tracer=True, mp=0)
    ncol, nz = 73*60, 44
    q = cp.ones((ncol, nz), dtype=cp.float32, order='F')
    dz, rho, kh = (q*cp.float32(x) for x in (100., 1., 10.))
    def diffuse(count):
        for _ in range(count):
            scalar_pblmix_columns_cuda(q, dz, rho, kh, 180.)
    diffusion = {str(n): timed(lambda: diffuse(n)) for n in (4, 8)}
    from types import SimpleNamespace
    from woof.core.passive_tracer_contract import TRACER_NAMES
    from woof.core.tracer_pblmix import tracer_pblmix_rates
    cache = {}
    def scratch(shape, name):
        if name not in cache:
            cache[name] = cp.empty(shape, cp.float32)
        return cache[name]
    shape = (nz, 60, 73)
    state = SimpleNamespace(scratch=scratch, **{name: cp.ones(shape, cp.float32) for name in TRACER_NAMES})
    fields = [cp.full(shape, x, cp.float32) for x in (100., 1., 10.)]
    diffusion['8_shared'] = timed(lambda: tracer_pblmix_rates(state, *fields, 180.))
    receipt = dict(mode='fast', gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        dimensions=[44, 60, 73], units='milliseconds per stage',
        nearest_transport='MY2 mp9 scalar inventory, same existing CUDA operators',
        nearest=baseline, passive=passive,
        transport_ratios={k: passive[k]/baseline[k] for k in ('ordinary', 'pd')},
        diffusion=diffusion, diffusion_ratio=diffusion['8_shared']/diffusion['4'],
        diffusion_cost='Eight tracer right-hand sides share coefficients and factorization; nearest is four mp28 independent solves')
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
