"""WRF chem/module_chem_utilities.F:8-181 column preparation.

Mass fields use (nz, ny, nx), w fields use (nz+1, ny, nx). WRF's extra
mass padding copies at kde are omitted; dz8w's padding zero is omitted too.
No consumer receives uninitialized z/rh padding. state.p is already full
pressure (state.py:452); adding state.pb would count base pressure twice.
"""
from __future__ import annotations

import numpy as np

from woof.core.chem_context import (CHEM_PREP_FIELDS as NAMES,
                                     CHEM_PREP_W_FIELDS as W_LEVEL_NAMES)




class ChemPrep:
    """Reusable device arrays, refreshed once per chem call.

    Base profiles may be 1-D or per-column (state.py:811-820). qv/qc/qr
    remain state views. Numerical preparation uses the CUDA kernel.
    """
    def __init__(self, state):
        self._fields = {}
        self.refresh(state)

    def get(self, name):
        return self._fields.get(name)

    def refresh(self, state):
        p = state.p
        if isinstance(p, np.ndarray):
            raise TypeError(
                "ChemPrep requires CUDA arrays: NumPy input would supply "
                "host pointers to the chemistry preparation kernel")
        nz,ny,nx = p.shape
        qv = state.qv
        import cupy as cp
        from woof.core.kernels import get_kernel
        if not isinstance(p, cp.ndarray):
            raise TypeError("ChemPrep requires CUDA arrays for its kernel pointers")
        if nz < 2:
            raise ValueError('chem_prep needs two mass levels for boundary extrapolation')
        for name in NAMES:
            shape=(nz+(name in W_LEVEL_NAMES),ny,nx)
            if name not in self._fields or self._fields[name].shape != shape:
                self._fields[name]=cp.empty(shape,dtype=cp.float32)
        u, v = state.u, state.v
        if getattr(state, "_moving_frame_velocity", None):
            from woof.moving_frame import ground_relative_winds
            u, v = ground_relative_winds(state)
        args=(p,state.thp,state.thb,state.alt,state.php,state.phb,u,v,
              p if qv is None else qv,state.fnm,state.fnp,
              np.int32(state.thb.ndim==3),np.int32(state.phb.ndim==3),
              np.int32(qv is not None),np.int32(nz),np.int32(ny),np.int32(nx))
        get_kernel('chem_prep','chem_prep')(((ny*nx+127)//128,),(128,),
                   args+tuple(self._fields[name] for name in NAMES))
        for name in ('qv','qc','qr'):
            value=getattr(state,name,None)
            if value is not None: self._fields[name]=value
            else: self._fields.pop(name,None)
        # Views the process lanes read by their WRF names (no copies):
        # chem_driver's vvel is grid%w_2 at the bottom face of each mass level
        # (chem_driver.F:735); msftx/msfty are WRF's two mass-point map
        # factors, which woof carries as one (state.msft); oro is the
        # terrain height GSL's smoke prep reads; rainncv is the grid-scale
        # rain of the step just closed (mm, WRF grid%rainncv, read by
        # wetdep_ls at chem_driver.F:1689), the microphysics accumulator's
        # per-step view, absent under mp_physics = 0.
        micro = getattr(getattr(state, 'physics', None), 'microphysics', None)
        for name, value in (('w', getattr(state, 'w', None)),
                            ('msftx', getattr(state, 'msft', None)),
                            ('msfty', getattr(state, 'msft', None)),
                            ('oro', getattr(state, 'ht', None)),
                            ('rainncv', getattr(micro, 'rainncv', None))):
            if value is not None: self._fields[name]=value
            else: self._fields.pop(name,None)
