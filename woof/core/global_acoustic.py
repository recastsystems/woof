"""Directional hydrostatic acoustic transport from WRF's global solver."""
import numpy as np


class GlobalAcousticTransport:
    def __init__(self, nx, ny, nz, *, profiles, metrics, rdx, rdy,
                 emdiv=.01, epssm=.1):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)):
            raise ValueError("acoustic dimensions must be positive integers")
        if len(metrics) != 7:
            raise ValueError("tx/ty/ux/uy/vx/vy/vxinv metrics required")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.shapes = ((ny, nx), (ny, nx), (ny, nx+1), (ny, nx+1),
                       (ny+1, nx), (ny+1, nx), (ny+1, nx))
        self._validate((profiles, *metrics), ((6, nz+1), *self.shapes))
        if not np.isfinite((rdx, rdy, emdiv, epssm)).all() or min(rdx, rdy) <= 0:
            raise ValueError("finite positive inverse spacing and finite damping required")
        self.profiles, self.metrics = profiles, tuple(metrics)
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz))
        self.spacing = np.float32(rdx), np.float32(rdy)
        self.emdiv, self.epssm = np.float32(emdiv), np.float32(epssm)
        module = load_module_no_ftz('global_acoustic', no_fmad=True)
        self.uv = module.get_function('global_advance_uv')
        self.mu_theta = module.get_function('global_advance_mass_theta')
        self.pack_kernel = module.get_function('global_pack_one')

    @staticmethod
    def _validate(arrays, shapes):
        import cupy as cp
        for value, shape in zip(arrays, shapes):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"acoustic carrier must be resident float32 {shape}")

    def pack(self, value, halo):
        nx, ny = self.nx, self.ny
        if value.ndim not in (2, 3):
            raise ValueError("one-cell halo source must have two or three dimensions")
        nz = 1 if value.ndim == 2 else value.shape[0]
        self._validate((value, halo),
                       ((ny, nx) if value.ndim == 2 else (nz, ny, nx),
                        (ny+2, nx+2) if value.ndim == 2 else (nz, ny+2, nx+2)))
        self.pack_kernel(((halo.size+127)//128,), (128,),
                         (np.int32(nx), np.int32(ny), np.int32(nz), value, halo))

    def advance_uv(self, *, dt, muu, muv, mudf, ph, alt, p, pb, al,
                   ru_tend, rv_tend, u, v):
        nx, ny, nz = self.nx, self.ny, self.nz
        ushape, vshape = (nz, ny, nx+1), (nz, ny+1, nx)
        halo = (nz, ny+2, nx+2)
        values = (muu, muv, mudf, ph, alt, p, pb, al, ru_tend, rv_tend, u, v)
        self._validate(values, ((ny, nx+1), (ny+1, nx), (ny+2, nx+2),
                               (nz+1, ny+2, nx+2), halo, halo, halo, halo,
                               ushape, vshape, ushape, vshape))
        self.uv(((max(u.size, v.size)+127)//128,), (128,),
                (*self.dims, *self.spacing, np.float32(dt), self.emdiv,
                 self.profiles[0], self.profiles[1], *self.metrics[2:4],
                 self.metrics[4], self.metrics[5], self.metrics[6], *values))

    def advance_mass_theta(self, *, dt, muu, muv, mut, u, v, u_ref, v_ref,
                           theta_ref, ww_ref, theta_tend, mu_tend, mu, muts,
                           muave, mudf, ww, theta, theta_ave, divv):
        nx, ny, nz = self.nx, self.ny, self.nz
        mass, scalar = (ny, nx), (nz, ny, nx)
        ushape, vshape, wshape = (nz, ny, nx+1), (nz, ny+1, nx), (nz+1, ny, nx)
        values = (muu, muv, mut, u, v, u_ref, v_ref, theta_ref, ww_ref,
                  theta_tend, mu_tend, mu, muts, muave, mudf, ww, theta, theta_ave, divv)
        self._validate(values, ((ny, nx+1), (ny+1, nx), mass,
                               ushape, vshape, ushape, vshape, (nz, ny+2, nx+2),
                               wshape, scalar, mass, mass, mass, mass, mass,
                               wshape, scalar, scalar, scalar))
        self.mu_theta(((nx*ny+127)//128,), (128,),
                      (*self.dims, *self.spacing, np.float32(dt), self.epssm,
                       *(self.profiles[i] for i in range(6)),
                       *self.metrics[:2], self.metrics[3], self.metrics[6], *values))
