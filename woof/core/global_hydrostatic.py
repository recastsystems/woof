"""Resident hydrostatic diagnostics from the public WRF global case.

The caller owns every carrier, including packed profiles and constants.
Construction and validation happen before capture; each diagnostic call
enqueues one column kernel with no allocation or host read.
"""
import numpy as np


class HydrostaticDiagnostics:
    def __init__(self, nx, ny, nz, *, constants, profiles, nmoist=0,
                 first_scalar=1, qv_index=1, use_theta_m=1, hypsometric_opt=1):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)):
            raise ValueError("hydrostatic dimensions must be positive integers")
        if hypsometric_opt not in (1, 2) or use_theta_m not in (0, 1):
            raise ValueError("native hypsometric_opt=1/2 and use_theta_m=0/1 required")
        if (type(nmoist) is not int or nmoist < 0 or type(first_scalar) is not int
                or first_scalar < 0 or type(qv_index) is not int
                or (nmoist > first_scalar and not 0 <= qv_index < nmoist)):
            raise ValueError("invalid native moisture species indices")
        for name, value, shape in (("constants", constants, (6,)),
                                   ("profiles", profiles, (9, nz+1))):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"{name} must be packed resident float32 {shape}")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nmoist = nmoist
        self.constants, self.profiles = constants, profiles
        self.args = tuple(np.int32(v) for v in
                          (nx, ny, nz, nmoist, first_scalar, qv_index,
                           use_theta_m, hypsometric_opt))
        self.kernel = load_module_no_ftz("global_hydrostatic", no_fmad=True).get_function("hydrostatic_diagnostics")

    def apply(self, *, mu, muts, alb, pb, t, phb, moist, p, al, ph,
              full_pressure, alt, window=None):
        import cupy as cp
        from woof.core.diagnostics import _validated_window
        nx, ny, nz = self.nx, self.ny, self.nz
        arrays = (mu, muts, alb, pb, t, phb, moist, p, al, ph, full_pressure, alt)
        shapes = ((ny, nx), (ny, nx), (nz, ny, nx), (nz, ny, nx),
                  (nz, ny, nx), (nz+1, ny, nx), (self.nmoist, nz, ny, nx),
                  (nz, ny, nx), (nz, ny, nx), (nz+1, ny, nx),
                  (nz, ny, nx), (nz, ny, nx))
        for value, shape in zip(arrays, shapes):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"hydrostatic carrier must be packed resident float32 {shape}")
        j0, i0, nyw, nxw = _validated_window(window, ny, nx)
        self.kernel(((nyw*nxw+127)//128,), (128,),
                    (*self.args, *(np.int32(v) for v in (j0, i0, nyw, nxw)),
                     self.constants, self.profiles, *arrays[:7],
                     p, al, ph, full_pressure, alt))


class HydrostaticAcoustic:
    """Native linearized hydrostatic pressure and geopotential recurrence.

    Pressure history is updated after diagnosing geopotential, as in WRF.
    Every mutable array is supplied by the caller and remains resident.
    """
    def __init__(self, nx, ny, nz, *, profiles, t0=300.0, smdiv=0.1):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)):
            raise ValueError("hydrostatic dimensions must be positive integers")
        if (not isinstance(profiles, cp.ndarray) or profiles.dtype != cp.float32
                or profiles.shape != (4, nz) or not profiles.flags.c_contiguous):
            raise ValueError("acoustic profiles must be packed resident float32 (4,nz)")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.profiles = profiles
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz))
        self.t0, self.smdiv = np.float32(t0), np.float32(smdiv)
        self.kernel = load_module_no_ftz("global_hydrostatic", no_fmad=True).get_function("hydrostatic_acoustic")

    def apply(self, *, mu, mut, alt, theta_coupled, theta_ref, c2a,
              p, al, ph, pm1, first):
        import cupy as cp
        if type(first) is not bool:
            raise ValueError("first must identify acoustic initialization as a logical")
        nx, ny, nz = self.nx, self.ny, self.nz
        arrays = (mu, mut, alt, theta_coupled, theta_ref, c2a, p, al, ph, pm1)
        shapes = ((ny, nx), (ny, nx), *((nz, ny, nx),)*6,
                  (nz+1, ny, nx), (nz, ny, nx))
        for value, shape in zip(arrays, shapes):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"hydrostatic acoustic carrier must be resident float32 {shape}")
        self.kernel(((nx*ny+127)//128,), (128,),
                    (*self.dims, np.int32(0 if first else 1), self.t0, self.smdiv,
                     self.profiles, *arrays))


class HydrostaticVerticalVelocity:
    """Native diagnose_w with distinct directional metrics and terrain BC.

    Height includes one ghost cell per edge; the caller fills these using
    the domain topology before stepping. WRF's first three wind levels are
    required by the lower boundary. This operator does not fill ghosts.
    """
    def __init__(self, nx, ny, nz, *, constants, profiles):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if any(type(v) is not int or v < 1 for v in (nx, ny, nz)) or nz < 3:
            raise ValueError("hydrostatic W requires positive nx/ny and nz>=3")
        for name, value, shape in (("constants", constants, (7,)),
                                   ("profiles", profiles, (2, nz+1))):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"{name} must be packed resident float32 {shape}")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.constants, self.profiles = constants, profiles
        self.dims = tuple(np.int32(v) for v in (nx, ny, nz))
        self.kernel = load_module_no_ftz("global_hydrostatic", no_fmad=True).get_function("hydrostatic_vertical_velocity")

    def apply(self, *, mut, msftx, msfty, height_halo, u, v,
              ph_tend, ph_new, ph_old, w):
        import cupy as cp
        nx, ny, nz = self.nx, self.ny, self.nz
        arrays = (mut, msftx, msfty, height_halo, u, v, ph_tend, ph_new, ph_old, w)
        shapes = ((ny, nx), (ny, nx), (ny, nx), (ny+2, nx+2),
                  (nz, ny, nx+1), (nz, ny+1, nx), *((nz+1, ny, nx),)*4)
        for value, shape in zip(arrays, shapes):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"diagnosed W carrier must be resident float32 {shape}")
        self.kernel(((nx*ny+127)//128,), (128,),
                    (*self.dims, self.constants, self.profiles, *arrays))
