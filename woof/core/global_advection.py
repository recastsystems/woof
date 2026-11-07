"""Native global fifth-order transport with caller-owned longitude halos."""
import numpy as np


class GlobalAdvection:
    def __init__(self, nx, ny, nz, *, profiles):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if (any(type(v) is not int or v < 1 for v in (nx, ny, nz))
                or nx < 7 or ny < 7 or nz < 4):
            raise ValueError("native global transport requires nx/ny>=7 and nz>=4")
        if (not isinstance(profiles, cp.ndarray) or profiles.dtype != cp.float32
                or profiles.shape != (3, nz+1) or not profiles.flags.c_contiguous):
            raise ValueError("transport profiles must be resident float32 (3,nz+1)")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.profiles = profiles
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz))
        module = load_module_no_ftz('global_advection', no_fmad=True)
        self.kernel = module.get_function('global_advect')
        self.pack_kernel = module.get_function('global_zonal_pack')

    def pack(self, packed, halo, *, stagger_x=False):
        """Fill existing device storage; no allocation or host data read."""
        import cupy as cp
        nx = self.nx
        if type(stagger_x) is not bool:
            raise ValueError("stagger_x must be logical")
        if (not isinstance(packed, cp.ndarray) or packed.ndim != 3
                or packed.shape[2] != nx+int(stagger_x)):
            raise ValueError("packed transport field has the wrong zonal staggering")
        nz, ny, _ = packed.shape
        for value, shape in ((packed, (nz, ny, nx+int(stagger_x))),
                              (halo, (nz, ny, nx+7))):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"transport field must be resident float32 {shape}")
        self.pack_kernel(((halo.size+127)//128,), (128,),
                         (np.int32(nx), np.int32(ny), np.int32(nz),
                          np.int32(stagger_x), packed, halo))

    def apply(self, *, kind, q, ru, rv, ww, metric, vx, tendency,
              rdx, rdy, vorder=3):
        import cupy as cp
        if kind not in ('scalar', 'u', 'v') or vorder not in (3, 5):
            raise ValueError("native global scalar/u/v and vertical order 3/5 required")
        nx, ny, nz = self.nx, self.ny, self.nz
        nxs, nys = nx+int(kind == 'u'), ny+int(kind == 'v')
        arrays = (q, ru, rv, ww, metric, vx, tendency)
        shapes = ((nz, nys, nx+7), (nz, ny, nx+7), (nz, ny+1, nx+7),
                  (nz+1, ny, nx+7), (nys, nxs), (ny+1, nx), (nz, nys, nxs))
        for value, shape in zip(arrays, shapes):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"transport carrier must be resident float32 {shape}")
        self.kernel(((tendency.size+127)//128,), (128,),
                    (*self.dims, np.int32(('scalar', 'u', 'v').index(kind)),
                     np.int32(vorder), np.float32(rdx), np.float32(rdy),
                     q, ru, rv, ww, metric, vx, self.profiles[0],
                     self.profiles[1], self.profiles[2], tendency))
