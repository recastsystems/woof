"""Setup/seam adapter for the published regional dycore.

Only the captured CUDA graph is called from dycore.step. Analysis endpoints
are physical fields on this domain's eta surfaces and staggered wind grids.
The caller owns analysis interpolation, time seams and restart reattachment.
"""
from pathlib import Path
import numpy as np
from .config import NudgingConfig


def attach(state, run_config, config: NudgingConfig, analyses, *,
           interval_start, interval_end, pbl_height=None):
    """Attach at initialization or a synchronized forcing/restart seam.

    Off returns before importing CUDA or accessing state/analysis. Enabled
    initially supports a fixed time step, resident domain, fast numerics.
    A restart must reconstruct the graph from the restored state clock and
    the same analysis endpoints; this object is not serialized.
    """
    if not config.enabled:
        return None
    import cupy as cp
    from woof.wrf_exact import DIAGNOSTICS_ENABLED
    from .gpu import FieldBinding, SpectralNudger
    if DIAGNOSTICS_ENABLED or run_config.use_adaptive_time_step:
        raise ValueError('spectral nudging requires fast numerics and a fixed time step')
    if getattr(state, '_spectral_nudging_graph', None) is not None:
        raise ValueError('nudging already attached; replace its forcing interval at a seam')
    if not (run_config.specified or run_config.nested):
        raise ValueError('NUDGING_REGIONAL_GEOMETRY: nudging needs a specified root or a fixed regional nest')
    if config.convention == "regional" and config.boundary_cells < run_config.spec_zone + run_config.relax_zone:
        raise ValueError('nudging boundary_cells must protect the full lateral relaxation zone')
    nz, ny, nx = state.thp.shape
    if pbl_height is None:
        fields = getattr(getattr(state, 'physics', None), 'fields', {})
        pbl_height = fields.get('pblh')
    if (pbl_height is None and config.convention == 'wrf'
            and not any(getattr(config, 'if_no_pbl_nudging_'+suffix)
                        for suffix in ('uv','t','ph','q'))):
        pbl_height = cp.zeros((ny,nx),dtype=cp.float32)
    if pbl_height is None:
        raise ValueError('bind live PBL height, or an explicit analyzed PBL height buffer')
    if (not isinstance(pbl_height, cp.ndarray) or pbl_height.dtype != cp.float32
            or pbl_height.shape != (ny, nx) or not pbl_height.flags.c_contiguous):
        raise ValueError('PBL height must be a contiguous float32 mass-grid device buffer')
    physical = {'u': state.u, 'v': state.v,
                'temperature': cp.empty_like(state.thp),
                'geopotential': cp.empty_like(state.php)}
    previous = {name: cp.empty_like(physical[name]) for name in ('temperature','geopotential')}
    heights = {name: cp.empty_like(a) for name,a in physical.items()}
    pbl = {name: cp.empty(a.shape[-2:], dtype=cp.float32) for name,a in physical.items()}
    source = Path(__file__).with_name('forecast.cu').read_text()
    module = cp.RawModule(code=source, options=('--std=c++11',))
    from woof.certify.kernel_manifest import record_module
    record_module('woof.spectral_nudging.forecast:forecast', source=source,
                  options=('--std=c++11',), module=module)
    pack, unpack, weight_kernel = (module.get_function(n) for n in ('pack_fields','unpack_fields','wrf_weights'))
    ints = tuple(np.int32(v) for v in (nz, ny, nx, state.thb.ndim == 3))
    potential = np.int32(config.convention == "wrf")
    weights = {name: cp.empty_like(a) for name, a in physical.items()} if potential else {}
    if config.coefficient('qv') > 0:
        physical['qv'] = state.qv
        heights['qv'] = heights['temperature']
        pbl['qv'] = pbl['temperature']
        if potential:
            weights['qv'] = cp.empty_like(state.qv)
    n = max(a.size for a in physical.values())
    grid, block = ((n+255)//256,), (256,)
    def before():
        pack(grid, block, (state.thp, state.thb, state.p, state.php, state.phb,
             state.ht, pbl_height, physical['temperature'], previous['temperature'],
             physical['geopotential'], previous['geopotential'],
             *(heights[k] for k in ('u','v','temperature','geopotential')),
             *(pbl[k] for k in ('u','v','temperature','geopotential')), *ints, potential))
        for name, w in weights.items():
            field = ('u', 'v', 'temperature', 'geopotential', 'qv').index(name)
            suffix = {'u':'uv', 'v':'uv', 'temperature':'t', 'geopotential':'ph', 'qv':'q'}[name]
            controls = tuple(np.int32(v) for v in (field, config.mode == 'grid',
                getattr(config, 'if_no_pbl_nudging_'+suffix),
                getattr(config, 'if_zfac_'+suffix), getattr(config, 'k_zfac_'+suffix),
                getattr(config, 'dk_zfac_'+suffix)))
            weight_kernel(((w.shape[-2]*w.shape[-1]+255)//256,), block,
                (state.php, state.phb, state.ht, pbl_height, w, *ints, *controls))
    def after():
        unpack(grid, block, (state.thp, state.php, state.p,
               physical['temperature'], previous['temperature'],
               physical['geopotential'], previous['geopotential'], *ints[:3], potential))
    bindings = {}
    for name, value in physical.items():
        if config.coefficient(name) == 0:
            continue
        a,b = analyses[name]
        if a.shape != value.shape or b.shape != value.shape:
            raise ValueError(f'{name} analysis must match its model grid')
        bindings[name] = FieldBinding(value, cp.array(a,dtype=cp.float32,order='C'),
             cp.array(b,dtype=cp.float32,order='C'), heights[name], pbl[name],
             weights.get(name), getattr(state, 'nudging_tendency_'+name, None))
    plan = SpectralNudger(config, bindings, elapsed_seconds=state.elapsed_seconds,
             dt_seconds=run_config.dt, interval_start=interval_start,
             interval_end=interval_end, record_before=before, record_after=after)
    plan.forecast_storage = (module, physical, previous, heights, pbl, pbl_height, weights)
    # Store the native graph directly. The published dispatcher performs one
    # Cython graph.launch; there is no Python scheme callback in the step path.
    state._spectral_nudging_graph = plan.graph
    state._spectral_nudging_owner = plan
    return plan
