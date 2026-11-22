"""Native SFIRE ideal sounding and atmospheric initialization on the device."""
from functools import lru_cache
import numpy as np

MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")


def module_source(kernel_dir=None):
    from woof.core.kernels import module_source as source
    return source("sfire_ideal_atmos") if kernel_dir is None else source("sfire_ideal_atmos", kernel_dir=kernel_dir)


@lru_cache(maxsize=None)
def _module(device):
    from woof.core.sfire_core import _cupy
    cp = _cupy()
    from cupy.cuda import compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile
    key = "woof.core.sfire_ideal_atmos:sfire_ideal_atmos"
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(key):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_ideal_atmos.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(key, source=source, options=MODULE_OPTIONS, module=None)
    return module


def _launch(name, size, args):
    from woof.core.sfire_core import _cupy
    cp = _cupy()
    _module(cp.cuda.Device().id).get_function(name)(((size + 127)//128,), (128,), args)


def _array(value, name, shape=None):
    from woof.core.sfire_core import _cupy
    cp = _cupy()
    a = cp.asarray(value, dtype=cp.float32)
    a = cp.ascontiguousarray(a).reshape(a.shape)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError(f"SFIRE ideal {name} shape {a.shape} differs from {tuple(shape)}")
    if not bool(cp.isfinite(a).all()):
        raise ValueError(f"SFIRE ideal {name} must be finite")
    return a


def prepare_sounding(raw, *, dry=False):
    """WRF get_sounding; raw qv is g/kg, returned qv is kg/kg.

    Surface scalars are psurf in mb and theta_surface in K. The native
    routine reads qv_surface but uses the first profile qv for surface density.
    """
    from woof.core.sfire_core import _cupy
    cp = _cupy()
    height = _array(raw["height"], "height")
    if height.ndim != 1 or not 2 <= height.size <= 1000:
        raise ValueError("SFIRE ideal sounding needs 2..1000 one-dimensional input levels")
    n = int(height.size)
    data = {key: _array(raw[key], key, (n,)) for key in ("theta", "qv", "u", "v")}
    if not bool(cp.all(height[1:] > height[:-1])):
        raise ValueError("SFIRE ideal sounding heights must increase strictly")
    if bool(cp.any(data["theta"] <= 0)) or bool(cp.any(data["qv"] < 0)):
        raise ValueError("SFIRE ideal sounding needs positive theta and nonnegative water vapor")
    surface = cp.concatenate((_array(raw["psurf"], "psurf").reshape(1),
                              _array(raw["theta_surface"], "theta_surface").reshape(1)))
    if bool(cp.any(surface <= 0)):
        raise ValueError("SFIRE ideal surface pressure and potential temperature must be positive")
    pm, pd, rho, qv = [cp.empty(n, dtype=cp.float32) for _ in range(4)]
    _launch("sfire_ideal_prepare_sounding", 1,
            (height, data["theta"], data["qv"], surface, pm, pd, rho, qv,
             np.int32(n), np.int32(bool(dry))))
    if any(not bool(cp.isfinite(v).all()) or bool(cp.any(v <= 0)) for v in (pm,pd,rho)):
        raise ValueError("SFIRE ideal sounding hydrostatic integration produced invalid pressure or density")
    return dict(height=height, p_moist=pm, p_dry=pd, theta=data["theta"], rho=rho,
                u=data["u"], v=data["v"], qv=qv)


def interpolate_sounding(values, coordinates, target):
    """WRF interp_0, including both endpoint extrapolation branches."""
    from woof.core.sfire_core import _cupy
    cp = _cupy()
    z = _array(coordinates, "interpolation coordinates")
    if z.ndim != 1 or z.size < 2:
        raise ValueError("SFIRE ideal interpolation needs at least two coordinates")
    if not (bool(cp.all(z[1:] > z[:-1])) or bool(cp.all(z[1:] < z[:-1]))):
        raise ValueError("SFIRE ideal interpolation coordinates must be strictly monotonic")
    v = _array(values, "interpolation values", z.shape)
    t = _array(target, "interpolation target")
    result = cp.empty_like(t)
    _launch("sfire_ideal_interp", int(t.size), (v,z,t,result,np.int32(z.size),np.int32(t.size)))
    return result


def initialize_atmosphere(cfg, terrain, vertical, sounding, *, native_use_theta_m=1):
    """Native dry base, moist hydrostatics, wind and thermal perturbation.

    T_DRY is the physical dry-theta perturbation used by WOOF-ArWen. T retains
    the native final theta_m conversion for explicit initialization grading.
    No DomainState is mutated until the complete mapping has been validated.
    """
    from woof.core.sfire_core import _cupy
    cp = _cupy()
    nx,ny,nz = int(cfg.nx),int(cfg.ny),int(cfg.nz)
    if nz < 3 or nx < 2 or ny < 2:
        raise ValueError("SFIRE ideal atmosphere needs three mass levels and two cells per horizontal axis")
    if isinstance(native_use_theta_m, bool) or native_use_theta_m not in (0,1):
        raise ValueError("native ideal theta_m choice must be integer 0 or 1")
    ht = _array(terrain, "terrain", (ny,nx))
    coeff = {name:_array(vertical[name], name, (nz+1,) if name in ('c1f','c2f') else (nz,))
             for name in ('dnw','rdnw','rdn','c1f','c2f','c1h','c2h','c3h','c4h')}
    ptop = _array(vertical['p_top'], 'p_top').reshape(1)
    weights = cp.concatenate([_array(vertical[name],name).reshape(1) for name in ('cf1','cf2','cf3')])
    dry,moist = sounding['dry'],sounding['moist']
    nd,nm = len(dry['height']),len(moist['height'])
    if nd != nm or not bool(cp.array_equal(dry['height'],moist['height'])):
        raise ValueError("native dry and moist initialization must use the same sounding levels")
    d = {key:_array(dry[key], 'dry '+key, (nd,)) for key in ('height','p_moist','theta')}
    m = {key:_array(moist[key], 'moist '+key, (nm,)) for key in ('height','p_moist','p_dry','theta','qv','u','v')}
    mass=(nz,ny,nx);interfaces=(nz+1,ny,nx)
    out={name:cp.empty(mass,cp.float32) for name in ('PB','P','T_INIT','T_DRY','T','QVAPOR','ALB','ALT','AL')}
    out.update({name:cp.empty(interfaces,cp.float32) for name in ('PHB','PH')})
    out.update({name:cp.empty((ny,nx),cp.float32) for name in ('MUB','MU','TSK','TMN')})
    status=cp.empty((ny,nx),cp.int32)
    scalar=[np.float32(getattr(cfg,key)) for key in ('dx','dy','delt_perturbation','xrad_perturbation',
              'yrad_perturbation','zrad_perturbation','hght_perturbation')]
    _launch('sfire_ideal_columns',nx*ny,
        (ht,d['height'],d['p_moist'],d['theta'],m['height'],m['p_moist'],m['p_dry'],m['theta'],m['qv'],
         *[coeff[k] for k in ('dnw','rdnw','rdn','c1f','c2f','c1h','c2h','c3h','c4h')],weights,ptop,
         *[out[k] for k in ('PHB','PH','MUB','MU','PB','P','T_INIT','T_DRY','T','QVAPOR','ALB','ALT','AL','TSK','TMN')],
         status,*map(np.int32,(nx,ny,nz,nd,nm)),*scalar,np.int32(native_use_theta_m)))
    if bool(cp.any(status)):
        raise ValueError("SFIRE ideal hydrostatic column has invalid pressure, dry mass or density")
    out['U']=cp.empty((nz,ny,nx+1),cp.float32);out['V']=cp.empty((nz,ny+1,nx),cp.float32)
    for axis,name,wind in ((0,'U','u'),(1,'V','v')):
        _launch('sfire_ideal_wind',nx*ny+(ny if axis==0 else nx),
                (out['PHB'],m['height'],m['p_moist'],m[wind],coeff['c3h'],coeff['c4h'],ptop,out[name],
                 *map(np.int32,(nx,ny,nz,nm,axis))))
    out['W']=cp.zeros(interfaces,cp.float32);out['H_DIABATIC']=cp.zeros(mass,cp.float32)
    if bool(cfg.sfc_full_init):
        out.pop('TSK');out.pop('TMN')
    return out
