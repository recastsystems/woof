"""WRF 4.7.1's optional resolved subfilter stress history fields.

This module does not add a stress force: the existing diffusion operators
already consume the same stresses. Nothing is allocated with m_opt=0.
"""
import numpy as np
from woof.core.kernels import get_kernel

NAMES = ('m11', 'm22', 'm33', 'm12', 'm13', 'm23')


def capture(state, cfg, kmh, kmv, common, *, boundary_x, boundary_y):
    if not getattr(cfg, 'm_opt', 0):
        return
    nz, ny, nx = kmh.shape
    if cfg.diff_opt != 2:
        return
    arrays = [getattr(state, name) for name in NAMES]
    fields = getattr(getattr(state, 'physics', None), 'fields', {})
    ustm = fields.get('ustm', state.mup0)
    active = int(cfg.sf_sfclay_physics != 0 and 'ustm' in fields)
    get_kernel('sfs_output', 'wrf_stress_output')(
        ((nx+127)//128, ny, nz), (128,1,1),
        tuple(common + [kmh, kmh if kmv is None else kmv, ustm,
                       np.int32(active), np.int32(cfg.isfflx),
                       np.float32(cfg.tke_drag_coefficient)] + arrays +
              [np.int32(cfg.bl_pbl_physics == 0), np.int32(nz), np.int32(ny),
               np.int32(nx), np.int32(state.phb.ndim == 3),
               np.int32(boundary_x), np.int32(boundary_y)]))
