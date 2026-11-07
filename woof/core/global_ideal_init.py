"""GPU analytic initialization from WRF's public global ideal initializer.

This module constructs the native state; admission to the regional solver
separately requires its directional metrics, hydrostatic stepping and polar
operators. It does not substitute a regional approximation for that solver.
"""
import numpy as np

PROFILE_NAMES = ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
                 "c1f", "c2f", "c3f", "c4f", "c1h", "c2h", "c3h", "c4h")


def initialize(nx, ny, nz, *, ztop=120000.0, hybrid_opt=0, etac=0.2,
               random_samples, constants=None):
    """Return resident native fields, with caller-owned RNG samples.

    Making the RNG input explicit permits native replays without inventing
    a match to Fortran's unspecified random_seed() sequence. The ordinary
    initializer can supply GPU-generated samples at cold start.
    """
    import cupy as cp
    from woof.core.kernels import load_module_no_ftz
    if any(type(v) is not int or v < minimum
           for v, minimum in ((nx, 2), (ny, 2), (nz, 3))):
        raise ValueError("global ideal dimensions require nx/ny>=2 and nz>=3")
    if hybrid_opt not in (0, 1, 2, 3):
        raise ValueError("native global hybrid_opt is 0, 1, 2 or 3")
    if (not isinstance(random_samples, cp.ndarray)
            or random_samples.dtype != cp.float32
            or random_samples.shape != (nz, ny, nx)
            or not random_samples.flags.c_contiguous):
        raise ValueError("random_samples must be resident packed float32 (nz,ny,nx)")
    if constants is None:
        rd, cp_air = np.float32(287), np.float32(1004.5)
        constants = np.asarray((ztop, etac, np.float32(np.pi/180), rd, cp_air,
                                -(cp_air-rd)/cp_air, rd/cp_air, 300, 100000,
                                9.81, np.float32(1)/np.float32(6370000),
                                7.2921e-5), dtype=np.float32)
    if np.shape(constants) != (12,):
        raise ValueError("native constants must contain twelve float32 words")
    c = cp.asarray(constants, dtype=cp.float32)
    fields = {name: cp.empty((ny, nx), dtype=cp.float32)
              for name in ("xlat", "xlong", "msftx", "msfty", "f", "e", "mub", "mu_2", "tsk", "tmn")}
    fields.update({name: cp.empty((ny+1, nx), dtype=cp.float32)
                   for name in ("msfvx", "msfvy", "msfvx_inv")})
    fields.update({name: cp.empty((nz, ny, nx), dtype=cp.float32)
                   for name in ("pb", "t_init", "alb", "t_2", "p", "al")})
    fields.update({name: cp.empty((nz+1, ny, nx), dtype=cp.float32)
                   for name in ("phb", "ph_2")})
    profile = cp.zeros((16, nz+1), dtype=cp.float32)
    scalars = cp.empty(8, dtype=cp.float32)
    module = load_module_no_ftz("global_ideal_init", no_fmad=True)
    dims = tuple(np.int32(v) for v in (nx, ny, nz))
    module.get_function("hs_init_coord")(((nz+256)//256,), (256,),
        (*dims, np.int32(hybrid_opt), c, scalars, profile))
    module.get_function("hs_init_geography")(((nx*(ny+1)+255)//256,), (256,),
        (*dims[:2], c, *(fields[n] for n in ("xlat", "xlong", "msftx", "msfty", "f", "e", "msfvx", "msfvy", "msfvx_inv"))))
    module.get_function("hs_init_columns")(((nx*ny+127)//128,), (128,),
        (*dims, c, scalars, profile, random_samples,
         *(fields[n] for n in ("mub", "mu_2", "pb", "t_init", "alb", "t_2", "p", "al", "phb", "ph_2", "tsk", "tmn"))))
    fields.update({name: profile[i] for i, name in enumerate(PROFILE_NAMES)})
    fields["scalars"] = scalars
    return fields
