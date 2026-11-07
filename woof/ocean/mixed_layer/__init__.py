"""GPU prognostic ocean, WRF 3DPWP circulation and vertical mixing.

The source notice is adjacent to pwp.cu. Explicit repairs include finite
absorption at zero radiation, real-valued unequal-layer stirring, masked
land/lake stencils, symmetric corners, and zero-gradient inflow through the
column floor (WRF's floor difference is downwind for upwelling and runs away).
This is not a bitwise WRF claim.
"""
from pathlib import Path
import warnings
import numpy as np

PROFILE_FIELDS = ('om_tmp', 'om_s', 'om_u', 'om_v', 'om_depth',
                  'om_density', 'om_tini', 'om_sini')
STATE_FIELDS = (*PROFILE_FIELDS, 'om_ml', 'om_active')
PROGNOSTIC = ('om_tmp', 'om_s', 'om_u', 'om_v')
_MODULES = {}


def kernels(levels):
    import cupy as cp
    key = (cp.cuda.runtime.getDevice(), levels)
    if key not in _MODULES:
        from woof.wrf_exact import ENABLED
        options = ('--std=c++17', f'-DOCEAN_LEVELS={levels}')
        if ENABLED:
            options += ('--fmad=false', '--prec-div=true', '--prec-sqrt=true')
        from woof.certify.kernel_manifest import record_module
        source = Path(__file__).with_name('pwp.cu').read_text('utf-8')
        module = cp.RawModule(code=source, options=options)
        record_module(f'woof.ocean.mixed_layer:pwp:levels={levels}',
                      source=source, options=options, module=module)
        _MODULES[key] = module
    return _MODULES[key]


def _launch(module, name, n, args, block=128):
    module.get_function(name)(((n+block-1)//block,), (block,), args)


def ocean_land(fields):
    """Keep inland lakes with their lake surface model, outside ocean transport."""
    import cupy as cp
    lake = fields.get('lakemask', 0)
    active = fields.get('om_active', 1)
    return cp.ascontiguousarray(cp.where((lake >= 0.5) | (active < 1 - 1.e-6),
                                         cp.float32(1), fields['xland']))


def initialize(fields, cfg, initial=None, *, start=True):
    import cupy as cp
    initial = initial or {}
    levels = cfg.ocean_levels
    shape = (levels, *fields['tsk'].shape)
    for name in PROFILE_FIELDS:
        value = initial.get(name)
        if value is None:
            fields[name] = cp.zeros(shape, dtype=cp.float32)
        else:
            a = cp.asarray(value, dtype=cp.float32)
            if a.shape != shape:
                raise ValueError(f'{name.upper()} must have shape {shape}, got {a.shape}')
            fields[name] = cp.ascontiguousarray(a).copy()
    active = cp.asarray(initial.get('om_active', cp.ones(shape[1:], dtype=cp.float32)), dtype=cp.float32)
    if active.shape != shape[1:] or bool(cp.any(~cp.isfinite(active) | (active < 0) | (active > 1))):
        raise ValueError('OM_ACTIVE must be a finite 2D ocean coverage fraction in [0,1]')
    fields['om_active'] = cp.ascontiguousarray(active).copy()
    fields['om_ml'] = cp.zeros(shape[1:], dtype=cp.float32)
    fields['_pwp_base'] = cp.zeros((4*levels, *shape[1:]), dtype=cp.float32)
    fields['_pwp_tend'] = cp.zeros_like(fields['_pwp_base'])
    fields['_pwp_w'] = cp.zeros(shape, dtype=cp.float32)
    fields['_pwp_p'] = cp.zeros(shape, dtype=cp.float32)
    fields['_pwp_status'] = cp.zeros(1, dtype=cp.int32)
    fields['_pwp_land'] = ocean_land(fields)
    for name in ('_pwp_mx', '_pwp_my'):
        fields[name] = cp.ones(shape[1:], dtype=cp.float32)
    if not start:
        return
    if cfg.ocean_z:
        from woof.surface_ocean import validate_pwp_profile
        validate_pwp_profile(cfg)
        if initial:
            raise ValueError('ocean_z/ocean_t/ocean_s and explicit ocean_initial both supply profiles')
        for name, values in (('om_depth', cfg.ocean_z), ('om_tmp', cfg.ocean_t), ('om_s', cfg.ocean_s)):
            fields[name][...] = cp.asarray(values, dtype=cp.float32)[:, None, None]
        initial = {name:fields[name] for name in ('om_depth','om_tmp','om_s')}
    required = ('om_tmp', 'om_s', 'om_depth')
    supplied = [name in initial for name in required]
    if any(supplied) and not all(supplied):
        raise ValueError('3DPWP analysis requires OM_TMP, OM_S and OM_DEPTH together; partial profiles corrupt density and mixing')
    if not any(supplied):
        if cfg.ocean_mld_climatology != 'configured_uniform_mld':
            raise ValueError('WOA18 climatology initialization needs forecast coordinates and UTC start through initialize_ocean; configured_uniform_mld is the explicit spatially uniform alternative')
        if cfg.oml_hml0 <= 0:
            raise ValueError("3DPWP configured_uniform_mld fallback requires positive oml_hml0; no analysis mixed-layer depth was supplied")
        warnings.warn('OCEAN INITIALIZATION: configured_uniform_mld fallback; '
                      f'MLD {cfg.oml_hml0:g} m, 35 PSU, resting currents, configured OML lapse rate, 5 m layers. '
                      'No public ocean analysis was supplied.', RuntimeWarning, stacklevel=2)
        _launch(kernels(levels), 'pwp_fallback', fields['om_tmp'].size,
                tuple(fields[n] for n in ('om_tmp','om_s','om_u','om_v','om_depth','tsk'))+
                (np.float32(cfg.oml_hml0), np.float32(cfg.oml_gamma), np.int32(fields['tsk'].size)))
    wet = fields['_pwp_land'] > 1.5
    z = fields['om_depth']
    bad = (~cp.isfinite(z) | (z <= 0))
    bad[1:] |= z[1:] <= z[:-1]
    for name in ('om_tmp', 'om_s', 'om_u', 'om_v'):
        bad |= ~cp.isfinite(fields[name])
    bad |= (fields['om_tmp'] < 260) | (fields['om_tmp'] > 330)
    bad |= (fields['om_s'] < 0) | (fields['om_s'] > 50)
    if bool(cp.any(bad & wet[None])):
        raise ValueError('3DPWP analysis has missing, nonfinite or nonphysical ocean columns, or non-increasing positive layer depths; repair the ocean analysis before coupling')
    fields['om_tini'][...] = initial.get('om_tini', fields['om_tmp'])
    fields['om_sini'][...] = initial.get('om_sini', fields['om_s'])
    bad_ref = (~cp.isfinite(fields['om_tini']) | ~cp.isfinite(fields['om_sini']) |
               (fields['om_tini'] < 260) | (fields['om_tini'] > 330) |
               (fields['om_sini'] < 0) | (fields['om_sini'] > 50))
    if bool(cp.any(bad_ref & wet[None])):
        raise ValueError('3DPWP reference temperature and salinity must be finite physical profiles')
    _launch(kernels(levels), 'pwp_diagnose', fields['tsk'].size,
            tuple(fields[name] for name in ('om_tmp','om_s','om_depth','om_tini','om_sini','om_density','om_ml','_pwp_land'))+
            (np.int32(fields['tsk'].size),))
    cp.copyto(fields['tsk'], fields['om_tmp'][0], where=wet)


def step(fields, *, dt, dx, dy, stbolt=5.67051e-8):
    """Two-stage WRF predictor/corrector; no device-to-host transfers."""
    import cupy as cp
    if dt <= 0 or dx <= 0 or dy <= 0:
        raise ValueError('3DPWP dt, dx and dy must be positive')
    levels,ny,nx = fields['om_tmp'].shape
    n=nx*ny; module=kernels(levels)
    base=fields['_pwp_base'].reshape((4,levels,ny,nx))
    for j,name in enumerate(PROGNOSTIC):
        cp.copyto(base[j], fields[name])
    common=tuple(fields[name] for name in (*PROGNOSTIC,'om_depth'))
    # With the sea-ice model active, the columns whose surface fluxes its step
    # set take their heat from ICE_OCEAN_HEAT in seaice.reconcile_ocean.
    coupled_ice=int('ice_ocean_coupled' in fields)
    ice_mask=fields['ice_ocean_coupled'] if coupled_ice else fields['_pwp_land']
    for first,stage_dt in ((1,2*dt),(0,0.5*dt)):
        _launch(module,'pwp_geometry',n, common +
                tuple(fields[name] for name in ('om_tini','om_sini','_pwp_land','_pwp_mx','_pwp_my','_pwp_w','_pwp_p'))+
                (np.float32(1/dx),np.float32(1/dy),np.int32(nx),np.int32(ny)))
        _launch(module,'pwp_transport',n*levels,common+
                tuple(fields[name] for name in ('_pwp_w','_pwp_p','_pwp_land','_pwp_mx','_pwp_my','_pwp_tend'))+
                (np.float32(1/dx),np.float32(1/dy),np.int32(nx),np.int32(ny),np.int32(first)))
        _launch(module,'pwp_column',n,
                tuple(fields[name] for name in (*PROGNOSTIC,'om_density','om_ml','tsk','_pwp_base','om_depth','om_tini','om_sini',
                                               'hfx','lh','gsw','glw','qfx','ust','uair','vair','rhoair','f','_pwp_land'))+
                (ice_mask,fields['_pwp_tend'],fields['_pwp_status'])+
                (np.float32(dt),np.float32(stage_dt),np.float32(stbolt),np.int32(coupled_ice),np.int32(n)),block=32)


def check(fields):
    """Synchronize only at history/checkpoint boundaries to surface kernel errors."""
    status=int(fields['_pwp_status'].item())
    if status:
        raise RuntimeError(f'3DPWP CUDA failure status={status}: bit 1 Richardson mixing did not converge; bit 2 nonfinite prognostic state')


def memory_planes(levels):
    # Eight profiles, four base, four tendencies, two circulation scratch,
    # plus mixed-layer depth, map factors, coast mask, SST diagnostics
    # and a conservatively rounded scalar error flag.
    return 18*levels+8
