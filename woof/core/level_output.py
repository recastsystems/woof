"""Resident native PLD/ZLD driver; no device reads in the step seam."""
from __future__ import annotations
import math
import numpy as np
from woof.level_output_config import active


def scratch_shapes(cfg):
    if not active(cfg):
        return {}
    m = (cfg.nz, cfg.ny, cfg.nx)
    w = (cfg.nz+1, cfg.ny, cfg.nx)
    shapes = dict(level_theta=m, level_pressure=m, level_pw=w, level_phb=w,
                  level_zero=m, level_pressure_bounds=(2, cfg.ny, cfg.nx))
    if cfg.p_lev_diags:
        shapes.update(level_press_targets=(len(cfg.press_levels),),
                      level_pl=(8, len(cfg.press_levels), cfg.ny, cfg.nx))
    if cfg.z_lev_diags:
        shapes.update(level_z_targets=(len(cfg.z_levels),),
                      level_zl=(9, len(cfg.z_levels), cfg.ny, cfg.nx))
    return shapes


def initialize(state, cfg):
    if not active(cfg):
        return
    from woof.core.state import cp
    for name, shape in scratch_shapes(cfg).items():
        state.scratch(shape, name)
    if cfg.p_lev_diags:
        state.existing_scratch('level_press_targets')[...] = cp.asarray(cfg.press_levels, dtype=cp.float32)
    if cfg.z_lev_diags:
        state.existing_scratch('level_z_targets')[...] = cp.asarray(cfg.z_levels, dtype=cp.float32)


def capture_atmosphere(state, cfg, atmosphere):
    """Hydrostatic pressures held from phy_prep, as native diagnostics read."""
    if 'p' not in _due(cfg, state.elapsed_seconds + cfg.dt):
        return
    s = state.existing_scratch
    from woof.core.kernels import get_kernel
    n = cfg.nx*cfg.ny
    get_kernel('level_diagnostics', 'level_capture_pressure')(((n+127)//128,), (128,),
        (atmosphere['pressure'], atmosphere['p_interface'], s('level_pressure'),
         s('level_pw'), s('level_pressure_bounds'), np.int32(cfg.nz), np.int32(n)))


def _due(cfg, seconds, *, initial=False):
    due = []
    for prefix, stream in (('p', 23), ('z', 22)):
        if not getattr(cfg, prefix+'_lev_diags'):
            continue
        interval = (getattr(cfg, prefix+'_lev_interval_s')
                    or getattr(cfg, f'auxhist{stream}_interval_s'))
        if initial or cfg.use_adaptive_time_step or (
                int(math.floor(seconds+.5)) % int(math.floor(interval+.5)) == 0):
            due.append(prefix)
    return due


def compute(state, cfg, seconds, *, initial=False):
    from woof.core.level_diagnostics import LevelInputs, pressure_levels, height_levels
    due = _due(cfg, seconds, initial=initial)
    if not due:
        return
    s = state.existing_scratch
    s('level_phb')[...] = (state.phb[:, None, None] if state.phb.ndim == 1 else state.phb)
    if not initial:
        # diagnostics_driver calls update_phys_fields first. Its dry
        # theta is the final state, while P_HYD still comes from phy_prep.
        s('level_theta')[...] = state.total_theta() - np.float32(300.)
    inputs = LevelInputs(state.u, state.v, s('level_theta'),
        state.qv if state.qv is not None else s('level_zero'),
        state.php, s('level_phb'), state.p, s('level_zero'))
    for prefix in due:
        if prefix == 'p':
            pressure_levels(inputs, s('level_pressure'), s('level_pw'),
                s('level_press_targets'), s('level_pl'),
                pressure_kind=cfg.use_tot_or_hyd_p, extrapolate=cfg.extrap_below_grnd,
                missing=cfg.p_lev_missing, bounds=s('level_pressure_bounds'))
        else:
            height_levels(inputs, state.ht, s('level_z_targets'), s('level_zl'),
                          missing=cfg.z_lev_missing)


def initial_diagnostics(state, cfg, metadata):
    """start_em initializes p_hyd from total P, then recurs upward for PW."""
    from woof.core.kernels import get_kernel
    from woof.core.ieva import theta_minus_t0
    s = state.existing_scratch
    s('level_theta')[...] = theta_minus_t0(state)
    s('level_pressure')[...] = state.p
    s('level_phb')[...] = (state.phb[:, None, None] if state.phb.ndim == 1 else state.phb)
    # These are the actual file's reference-profile constants, not a new
    # case-specific reference profile. Prepared non-file inputs supply their
    # own constants through the same metadata frame.
    constants = []
    for name in ('P00', 'T00', 'TLP'):
        if name not in metadata:
            raise ValueError('native level initialization requires input reference constant '+name)
        constants.append(np.float32(np.asarray(metadata[name]).reshape(-1)[0]))
    n = cfg.nx*cfg.ny
    get_kernel('level_diagnostics', 'level_initial_pressure')(((n+127)//128,), (128,),
        (state.p, state.pb, state.ht, state.c3f, state.c4f, s('level_pw'), s('level_pressure_bounds'),
         *constants, np.float32(state.p_top), np.int32(cfg.nz), np.int32(n),
         np.int32(state.pb.ndim == 3)))
    compute(state, cfg, 0., initial=True)


def device_fields(state, cfg, prefix):
    from woof.core.level_diagnostics import FIELDS
    stream = getattr(state, '_streamed_domain', None)
    s = state.existing_scratch if stream is None else lambda name: stream.store['scratch/'+name]
    suffix = 'PL' if prefix == 'p' else 'ZL'
    names = FIELDS if prefix == 'p' else (*FIELDS, 'P')
    out = s('level_pl' if prefix == 'p' else 'level_zl')
    result = {name+'_'+suffix: out[i] for i, name in enumerate(names)}
    result['P_PL' if prefix == 'p' else 'Z_ZL'] = (
        s('level_press_targets' if prefix == 'p' else 'level_z_targets')
        if stream is None else np.asarray(
            cfg.press_levels if prefix == 'p' else cfg.z_levels,
            dtype=np.float32))
    return result
