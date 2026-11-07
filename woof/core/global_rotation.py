"""Resident native Coriolis and spherical curvature with directional metrics."""
import numpy as np


class GlobalRotation:
    def __init__(self, nx, ny, nz, *, metrics, geography, profiles, zonal_halo,
                 reradius=1.0/6370000.0, degrad=np.pi/180.0):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)):
            raise ValueError("rotation dimensions must be positive integers")
        shapes = ((ny, nx), (ny, nx), (ny, nx+1), (ny, nx+1),
                  (ny+1, nx), (ny+1, nx))
        if len(metrics) != 6 or len(geography) != 5:
            raise ValueError("six directional metrics and f/e/sina/cosa/latitude required")
        values = (*metrics, *geography, profiles, zonal_halo)
        shapes = (*shapes, *((ny, nx),)*5, (2, nz+1),
                  (4*ny+nz*(ny+1)+(nz+1)*ny,))
        for value, shape in zip(values, shapes):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"rotation input must be packed resident float32 {shape}")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.metrics, self.geography, self.profiles = tuple(metrics), tuple(geography), profiles
        self.zonal_halo = zonal_halo
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz))
        self.constants = (np.float32(reradius), np.float32(degrad))
        self.grid, self.block = ((nz*(ny+1)*(nx+1)+127)//128,), (128,)
        module = load_module_no_ftz("global_rotation", no_fmad=True)
        self.coriolis = module.get_function("global_coriolis")
        self.curvature = module.get_function("global_curvature")
        self.halo = module.get_function("global_rotation_halo")

    def apply(self, *, ru, rv, rw, u, v, ru_tend, rv_tend, rw_tend,
              refresh_zonal_halo=True):
        import cupy as cp
        nx, ny, nz = self.nx, self.ny, self.nz
        arrays = (ru, rv, rw, u, v, ru_tend, rv_tend, rw_tend)
        ushape, vshape, wshape = (nz, ny, nx+1), (nz, ny+1, nx), (nz+1, ny, nx)
        for value, shape in zip(arrays, (ushape, vshape, wshape, ushape, vshape,
                                         ushape, vshape, wshape)):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"rotation carrier must be packed resident float32 {shape}")
        if type(refresh_zonal_halo) is not bool:
            raise ValueError("refresh_zonal_halo must be logical")
        if refresh_zonal_halo:
            count = max(nz*(ny+1), (nz+1)*ny)
            self.halo(((count+127)//128,), self.block,
                      (*self.dims, self.geography[0], self.geography[1],
                       self.geography[3], self.geography[4], rv, rw, self.zonal_halo))
        self.coriolis(self.grid, self.block,
                      (*self.dims, ru, rv, rw, *self.metrics, *self.geography[:4],
                       self.profiles[0], self.profiles[1], self.zonal_halo, ru_tend, rv_tend, rw_tend))
        self.curvature(self.grid, self.block,
                       (*self.dims, *self.constants, ru, rv, rw, u, v,
                        *self.metrics, self.geography[4], self.profiles[0],
                        self.profiles[1], self.zonal_halo, ru_tend, rv_tend, rw_tend))
