"""Sub-grid terrain drag from WRF v4.7.1, on the device.

Three WRF options, all off by default:

* ``topo_wind`` (``&physics``, max_domains, default 0) corrects the surface
  wind over complex terrain, under YSU only (it is the YSU scheme that reads
  the coefficients):

  - 1 -- Jimenez and Dudhia (2012, JAMC): the first-level momentum drag is
    multiplied by ``ln(sqrt(VAR_SSO))`` where the sub-grid terrain deviation
    exceeds e metres, weighted toward stable columns by the convective
    velocity, and tapered to zero on hill tops, where the 10 m wind is also
    blended toward the first-level wind;
  - 2 -- the drag multiplied by ``min(1.575, VAR*0.4/200 + 1.175)**2`` over
    land (the D. Ovens and C. Mass form);

  the coefficients are static, formed once per domain from its terrain
  (``dyn_em/start_em.F:1539-1626``): :func:`topo_wind_coefficients`.
* ``gwd_opt`` (``&dynamics``, max_domains, default 0; WRF requires one value
  on every domain or 0) adds orographic drag to the PBL momentum tendencies
  after the PBL scheme, at the PBL cadence (``module_pbl_driver.F:2129-2185``):

  - 1 -- the KIM orographic gravity-wave drag with flow blocking
    (``module_bl_gwdo.F`` -> ``bl_gwdo_run``), in two generations chosen by
    ``gwdo_version``: ``"wrf_480"`` (the default) is WRF v4.8.0's revision
    (Hong et al. 2025, Koo and Hong 2025, the nonhydrostatic factor of Xu
    et al. 2024), which also reads ELVMAX, the maximum sub-grid orographic
    height, and the ``gwd_dx_factor``/``gwd_if_nonhyd`` keys;
    ``"wrf_471"`` is the WRF v4.6.1/v4.7.1 scheme (Choi and Hong 2015),
    from VAR, CON, OA1-4 and OL1-4 alone: :func:`launch_gwdo`;
  - 3 -- the GSL drag suite (``module_bl_gwdo_gsl.F``): large-scale
    gravity-wave drag and flow blocking from the large-scale statistics
    (VARLS, CONLS, OA1LS-OA4LS, OL1LS-OL4LS), small-scale gravity-wave drag
    and turbulent orographic form drag from the small-scale ones (VARSS,
    CONSS, OA1SS-OA4SS, OL1SS-OL4SS); each component tapers off with grid
    length (large-scale below 13 km, gone at 3 km; small-scale below 12 km,
    gone at 1 km): :func:`launch_gwdo_gsl`.

The static fields come from the static builder, from WPS_GEOG's own
orographic datasets interpolated as GEOGRID.TBL.ARW's ``default`` rows say:
VAR_SSO from ``varsso_10m`` (average_gcell(4.0)+four_pt+average_4pt),
VAR/CON/OA/OL from ``orogwd_10m`` and the GSL statistics from ``orogwd3_10m``
(average_4pt, water masked to 0).  The kernels
(``kernels/terrain_drag.cu`` and the ``ysu_column_topo`` arm of
``kernels/ysu.cu``) are held to WRF's own Fortran by the oracle in
``tools/terrain_drag_wrf471_oracle`` (``tests/test_terrain_drag_wrf471_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from woof.static.orographic import (GSL_LS_FIELDS, GSL_SS_FIELDS,
                                    GWD_FIELDS, GWDO_FIELDS,
                                    TOPO_WIND_FIELDS, drag_static_fields,
                                    gwd_fields, required_static_fields)

#: WRF v4.7.1 topo_wind values and gwd_opt values ported here.
TOPO_WIND_OPTIONS = (0, 1, 2)
GWD_OPTIONS = (0, 1, 3)
#: gwd_opt = 1 generations: WRF v4.8.0's revised KIM drag (the default) and
#: the WRF v4.6.1/v4.7.1 scheme (bl_gwdo.F90 is byte-identical in the two).
GWDO_VERSIONS = ("wrf_480", "wrf_471")
DEFAULT_GWDO_VERSION = "wrf_480"
#: WRF v4.8.0 Registry.EM_COMMON:2902-2903 defaults.
DEFAULT_GWD_DX_FACTOR = 2.0
DEFAULT_GWD_IF_NONHYD = True

#: No FMA contraction (gfortran contracts nothing) and no flush to zero.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.terrain_drag:terrain_drag"
_SOURCES = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "terrain_drag.cu")
#: WRF v4.8.0 gwd_opt = 1: its own module, the frozen unit's text followed
#: by terrain_drag_gwdo480.cu (which reuses td_atan2f and the clamps), so the
#: frozen module's source, options and image stay exactly as they were.
GWDO480_MODULE_KEY = "woof.core.terrain_drag:gwdo480"
_GWDO480_SOURCES = _SOURCES + ("terrain_drag_gwdo480.cu",)
_BLOCK = 128
_KMAX = 128
#: A null device pointer for the kernels' optional diagnostic outputs.
_NULL = np.uintp(0)

#: WRF's module_model_constants values the PBL driver hands both schemes:
#: CP = 7*R_D/2, G, R_D, R_V, EP_1 = R_V/R_D - 1 (all float32 in WRF) and
#: the literal PI=3.141592653 of module_pbl_driver.F:2141 and :2177.
_G = np.float32(9.81)
_R_D = np.float32(287.0)
_R_V = np.float32(461.6)
_CP = np.float32(np.float32(7.0) * _R_D / np.float32(2.0))
_EP_1 = np.float32(_R_V / _R_D - np.float32(1.0))
_PI = np.float32(3.141592653)


def _compose(names) -> str:
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in names)


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    return _compose(_SOURCES)


def gwdo480_module_source() -> str:
    """The exact source string NVRTC compiles for WRF v4.8.0 gwd_opt = 1."""
    return _compose(_GWDO480_SOURCES)


@lru_cache(maxsize=None)
def _module(device: int):
    """NVRTC directly, -fmad=false and --ftz=false, one module per device.

    Not ``cupy.RawModule``: CuPy appends ``-ftz=true`` after the caller's
    options and NVRTC honours the last one, so ``--ftz=false`` would be
    dropped.  MEASURED on the RTX 5090 (sm_120) and RTX 4090 (sm_89),
    cupy 14.2: through RawModule a product of two normal floats with a
    subnormal result came back 0 on both cards, and on sm_120 ``x / 10.0f``
    came back as the reciprocal product (0.6936707 for (-23.063293 + 30) /
    10, where IEEE division and gfortran give 0.69367063).  Through
    ``compile_using_nvrtc`` both are IEEE on both cards.  The form-drag
    tendencies (``gwdo_gsl_column``) are subnormal well inside the column
    and topo_wind's taper divides by 10, so the loader is what holds the
    kernels to WRF.  ``-arch`` is left to CuPy (NVRTC 13 refuses it twice).
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "terrain_drag.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


@lru_cache(maxsize=None)
def _gwdo480_module(device: int):
    """WRF v4.8.0 gwd_opt = 1, loaded exactly as :func:`_module` loads its
    unit (NVRTC directly, -fmad=false and --ftz=false, one per device)."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler

    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = gwdo480_module_source()
    with cp.cuda.Device(device), observe_module_compile(GWDO480_MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, MODULE_OPTIONS, None, "terrain_drag_gwdo480.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(GWDO480_MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.Device().id).get_function(name)


def _gwdo480_kernel():
    import cupy as cp

    return _gwdo480_module(cp.cuda.Device().id).get_function(
        "gwdo480_column")


def gwdo_options(*, gwdo_version=DEFAULT_GWDO_VERSION,
                 dx_factor=DEFAULT_GWD_DX_FACTOR,
                 if_nonhyd=DEFAULT_GWD_IF_NONHYD):
    """``(gwdo_version, dx_factor, if_nonhyd)`` checked, or ValueError.

    ``dx_factor`` must be finite and positive: the base stress divides by
    dx_eff = dx_factor*dx, so 0 makes it infinite and a negative factor
    reverses the drag.
    """
    if gwdo_version not in GWDO_VERSIONS:
        raise ValueError(f"gwdo_version must be one of {GWDO_VERSIONS}, got "
                         f"{gwdo_version!r}")
    if isinstance(dx_factor, bool):
        raise ValueError("gwd_dx_factor must be a number, got a bool")
    factor = float(dx_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            f"gwd_dx_factor = {dx_factor!r}: the KIM base stress divides by "
            "gwd_dx_factor*dx, so it must be finite and positive")
    if not isinstance(if_nonhyd, (bool, np.bool_)):
        raise ValueError(f"gwd_if_nonhyd must be a bool, got {if_nonhyd!r}")
    return gwdo_version, factor, bool(if_nonhyd)


def _f32(xp, array, shape, name):
    out = xp.ascontiguousarray(xp.asarray(array, dtype=xp.float32))
    if out.shape != tuple(shape):
        raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return out


def topo_wind_coefficients(ht, xland, *, topo_wind: int, var_sso=None,
                           var2d=None):
    """``(ctopo, ctopo2, lap_hgt)``: start_em.F:1539-1626 on one domain.

    ``ht``, ``xland`` (1 land, 2 water) and the option's field (``var_sso``
    for 1, ``var2d`` for 2) are (ny, nx).  ``topo_wind = 0`` returns the
    Registry fill, ones.  The neighbours of the terrain laplacian clamp at
    the domain edges, WRF's non-periodic arm.
    """
    import cupy as cp

    topo_wind = int(topo_wind)
    if topo_wind not in TOPO_WIND_OPTIONS:
        raise ValueError(f"topo_wind must be one of {TOPO_WIND_OPTIONS}, "
                         f"got {topo_wind}")
    ht = _f32(cp, ht, np.shape(ht), "ht")
    if ht.ndim != 2:
        raise ValueError(f"ht must be (ny, nx), got {ht.shape}")
    ny, nx = ht.shape
    xland = _f32(cp, xland, (ny, nx), "xland")
    if topo_wind == 1 and var_sso is None:
        # WRF's own fatal error (start_em.F:222-223).
        raise ValueError("topo_wind = 1 requires VAR_SSO data")
    if topo_wind == 2 and var2d is None:
        raise ValueError("topo_wind = 2 requires VAR (the orographic "
                         "standard deviation) data")
    zeros = cp.zeros((ny, nx), dtype=cp.float32)
    var_sso = zeros if var_sso is None else _f32(cp, var_sso, (ny, nx),
                                                 "var_sso")
    var2d = zeros if var2d is None else _f32(cp, var2d, (ny, nx), "var2d")
    ctopo = cp.empty((ny, nx), dtype=cp.float32)
    ctopo2 = cp.empty((ny, nx), dtype=cp.float32)
    lap = cp.empty((ny, nx), dtype=cp.float32)
    n = ny * nx
    _kernel("topo_wind_static")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (ht, var_sso, var2d, xland, lap, ctopo, ctopo2, np.int32(nx),
         np.int32(ny), np.int32(topo_wind)))
    return ctopo, ctopo2, lap


def gsl_kpblmax(znu) -> int:
    """``kpblmax`` as module_bl_gwdo_gsl.F:162-164 forms it from ZNU."""
    kpblmax = None
    for k, value in enumerate(np.asarray(znu, dtype=np.float32), start=1):
        if value > np.float32(0.6):
            kpblmax = k + 1
    if kpblmax is None:
        raise ValueError("no eta level above 0.6: module_bl_gwdo_gsl.F "
                         "would leave kpblmax undefined")
    return kpblmax


def pbl_top_from_height(dz, pblh):
    """One-based first mass level at or above a diagnosed BL height.

    SASE supplies its own bulk-Richardson height instead of WRF's PBL
    scheme outputs.  The layer centers use the same FP64 cumulative depth
    as its height diagnostic; the floor is the first interior center and
    no crossing uses the top center.  The result supplies GSL's KPBL
    without writing SASE's surface-layer PBLH or KPBL carriers.
    """
    import cupy as cp

    dz = _f32(cp, dz, np.shape(dz), "dz")
    if dz.ndim != 3:
        raise ValueError(f"dz must be (nz, ny, nx), got {dz.shape}")
    nz, ny, nx = dz.shape
    pblh = _f32(cp, pblh, (ny, nx), "pblh")
    kpbl = cp.empty((ny, nx), dtype=cp.int32)
    n = ny * nx
    _kernel("terrain_pbl_top")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (dz, pblh, kpbl, np.int32(nz), np.int32(ny), np.int32(nx)))
    return kpbl


def _stack4(cp, planes, shape, name):
    arrays = [_f32(cp, plane, shape, f"{name}{m + 1}")
              for m, plane in enumerate(planes)]
    return cp.ascontiguousarray(cp.stack(arrays))


def _column_inputs(cp, atmosphere, nz, ny, nx):
    names = ("u", "v", "temperature", "qv", "pressure", "exner")
    out = {name: _f32(cp, atmosphere[name], (nz, ny, nx), name)
           for name in names}
    out["p_interface"] = _f32(cp, atmosphere["p_interface"], (nz + 1, ny, nx),
                              "p_interface")
    if "z" in atmosphere:
        out["z"] = _f32(cp, atmosphere["z"], (nz, ny, nx), "z")
    else:
        # phy_prep (module_big_step_utilities_em.F:4893):
        # z = 0.5*(z_at_w(k) + z_at_w(k+1)), heights above sea level.
        zi = _f32(cp, atmosphere["z_interface"], (nz + 1, ny, nx),
                  "z_interface")
        out["z"] = cp.ascontiguousarray(
            cp.float32(0.5) * (zi[:-1] + zi[1:]))
    out["dz"] = _f32(cp, atmosphere["dz"], (nz, ny, nx), "dz")
    return out


def launch_gwdo(atmosphere: Mapping[str, object], du, dv, *, var, con, oa,
                ol, sina, cosa, dx: float, dt: float, diagnostics=None,
                gwdo_version: str = "wrf_471", omax=None,
                dx_factor: float = DEFAULT_GWD_DX_FACTOR,
                if_nonhyd: bool = DEFAULT_GWD_IF_NONHYD):
    """gwd_opt = 1: add WRF's KIM orographic drag to ``du``/``dv`` in place.

    ``atmosphere`` is the physics driver's phy_prep dict (mass-point winds,
    temperature, vapour, hydrostatic pressures, Exner, interface heights).
    ``oa``/``ol`` are the four directional planes each.  ``diagnostics``,
    when a dict, receives DTAUX3D, DTAUY3D, DUSFCG, DVSFCG.

    ``gwdo_version = "wrf_471"`` (this function's default, so existing
    callers keep the scheme they were proven on) runs WRF v4.6.1/v4.7.1's
    scheme; ``"wrf_480"`` runs WRF v4.8.0's, which also reads ``omax``
    (ELVMAX, (ny, nx), metres above sea level), ``dx_factor``
    (gwd_dx_factor) and ``if_nonhyd`` (gwd_if_nonhyd).
    """
    gwdo_version, dx_factor, if_nonhyd = gwdo_options(
        gwdo_version=gwdo_version, dx_factor=dx_factor, if_nonhyd=if_nonhyd)
    if gwdo_version == "wrf_480":
        if omax is None:
            raise ValueError(
                "gwdo_version = 'wrf_480' reads ELVMAX (omax): without it "
                "WRF v4.8.0 turns the drag off on every column (omax <= "
                "50 m), so the field is required")
        _launch_gwdo480(atmosphere, du, dv, var=var, con=con, oa=oa, ol=ol,
                        omax=omax, sina=sina, cosa=cosa, dx=dx, dt=dt,
                        dx_factor=dx_factor, if_nonhyd=if_nonhyd,
                        diagnostics=diagnostics)
        return
    import cupy as cp

    nz, ny, nx = du.shape
    if nz > _KMAX:
        raise ValueError(f"gwd_opt = 1 holds a column of at most {_KMAX} "
                         f"levels per thread, got {nz}")
    col = _column_inputs(cp, atmosphere, nz, ny, nx)
    shape2 = (ny, nx)
    oa4 = _stack4(cp, oa, shape2, "oa")
    ol4 = _stack4(cp, ol, shape2, "ol")
    dtx = dty = dsx = dsy = None
    if diagnostics is not None:
        dtx = cp.zeros((nz, ny, nx), dtype=cp.float32)
        dty = cp.zeros((nz, ny, nx), dtype=cp.float32)
        dsx = cp.zeros(shape2, dtype=cp.float32)
        dsy = cp.zeros(shape2, dtype=cp.float32)

    n = ny * nx
    _kernel("gwdo_column")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (col["u"], col["v"], col["temperature"], col["qv"], col["pressure"],
         col["p_interface"], col["exner"], col["z"],
         _f32(cp, var, shape2, "var"), _f32(cp, con, shape2, "con"),
         oa4, ol4, _f32(cp, sina, shape2, "sina"),
         _f32(cp, cosa, shape2, "cosa"), du, dv,
         dtx if dtx is not None else _NULL, dty if dty is not None else _NULL,
         dsx if dsx is not None else _NULL, dsy if dsy is not None else _NULL,
         np.float32(dx), np.float32(dt), _G, _CP, _R_D, _EP_1, _PI,
         np.int32(nz), np.int32(ny), np.int32(nx)))
    if diagnostics is not None:
        diagnostics.update(DTAUX3D=dtx, DTAUY3D=dty, DUSFCG=dsx, DVSFCG=dsy)


def _launch_gwdo480(atmosphere, du, dv, *, var, con, oa, ol, omax, sina,
                    cosa, dx, dt, dx_factor, if_nonhyd, diagnostics):
    """WRF v4.8.0 gwd_opt = 1 (kernels/terrain_drag_gwdo480.cu)."""
    import cupy as cp

    nz, ny, nx = du.shape
    if nz > _KMAX:
        raise ValueError(f"gwd_opt = 1 holds a column of at most {_KMAX} "
                         f"levels per thread, got {nz}")
    if nz < 4:
        # kgwdmax = nz/2 must reach kgwdmin = 2 for the reference level.
        raise ValueError("gwd_opt = 1 (WRF v4.8.0) needs at least 4 levels, "
                         f"got {nz}")
    col = _column_inputs(cp, atmosphere, nz, ny, nx)
    shape2 = (ny, nx)
    oa4 = _stack4(cp, oa, shape2, "oa")
    ol4 = _stack4(cp, ol, shape2, "ol")
    dtx = dty = dsx = dsy = None
    if diagnostics is not None:
        dtx = cp.zeros((nz, ny, nx), dtype=cp.float32)
        dty = cp.zeros((nz, ny, nx), dtype=cp.float32)
        dsx = cp.zeros(shape2, dtype=cp.float32)
        dsy = cp.zeros(shape2, dtype=cp.float32)
    n = ny * nx
    _gwdo480_kernel()(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (col["u"], col["v"], col["temperature"], col["qv"], col["pressure"],
         col["p_interface"], col["exner"], col["z"],
         _f32(cp, var, shape2, "var"), _f32(cp, con, shape2, "con"),
         oa4, ol4, _f32(cp, omax, shape2, "omax"),
         _f32(cp, sina, shape2, "sina"), _f32(cp, cosa, shape2, "cosa"),
         du, dv,
         dtx if dtx is not None else _NULL, dty if dty is not None else _NULL,
         dsx if dsx is not None else _NULL, dsy if dsy is not None else _NULL,
         np.float32(dx), np.float32(dt), np.float32(dx_factor),
         np.int32(1 if if_nonhyd else 0), _G, _CP, _R_D, _EP_1, _PI,
         np.int32(nz), np.int32(ny), np.int32(nx)))
    if diagnostics is not None:
        diagnostics.update(DTAUX3D=dtx, DTAUY3D=dty, DUSFCG=dsx, DVSFCG=dsy)


#: GSL component diagnostic names, in the kernel's (8, ...) order.
GSL_DIAGNOSTIC_NAMES = ("ls", "bl", "ss", "fd")


def launch_gwdo_gsl(atmosphere: Mapping[str, object], du, dv, *, ls, ss,
                    sina, cosa, xland, br, pblh, kpbl, kpblmax: int,
                    dx: float, dt: float, diagnostics=None, rstoch=None):
    """gwd_opt = 3: add the GSL drag suite to ``du``/``dv`` in place.

    ``ls`` and ``ss`` are mappings with ``var``, ``con``, ``oa`` (4 planes)
    and ``ol`` (4 planes): the large-scale (2.5') and small-scale (30")
    statistics.  ``kpbl`` is the PBL scheme's one-based top level, ``pblh``
    its height, ``br`` the surface-layer bulk Richardson number.
    ``rstoch`` is the (ny, nx) lowest level of the SPP PBL pattern under
    spp_pbl = 1 (WRF's ``rstoch1D = pattern_spp_pbl(i,kts,j)``), which
    perturbs VAR, VARSS and the two variance caps; ``None`` is spp_pbl = 0.
    ``diagnostics``, when a dict, receives the eight DTAUX3D_*/DTAUY3D_*
    components and eight column stresses DUSFCG_*/DVSFCG_* (gwd_diags = 1).
    """
    import cupy as cp

    nz, ny, nx = du.shape
    if nz > _KMAX:
        raise ValueError(f"gwd_opt = 3 holds a column of at most {_KMAX} "
                         f"levels per thread, got {nz}")
    col = _column_inputs(cp, atmosphere, nz, ny, nx)
    shape2 = (ny, nx)
    kpbl = cp.ascontiguousarray(cp.asarray(kpbl, dtype=cp.int32))
    if kpbl.shape != shape2:
        raise ValueError(f"kpbl has shape {kpbl.shape}, expected {shape2}")
    if rstoch is not None:
        rstoch = _f32(cp, rstoch, shape2, "rstoch")
    dtau = dsfc = None
    if diagnostics is not None:
        dtau = cp.zeros((8, nz, ny, nx), dtype=cp.float32)
        dsfc = cp.zeros((8, ny, nx), dtype=cp.float32)
    n = ny * nx
    _kernel("gwdo_gsl_column")(
        ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
        (col["u"], col["v"], col["temperature"], col["qv"], col["pressure"],
         col["p_interface"], col["exner"], col["z"], col["dz"],
         _f32(cp, ls["var"], shape2, "VARLS"),
         _f32(cp, ls["con"], shape2, "CONLS"),
         _stack4(cp, ls["oa"], shape2, "OALS"),
         _stack4(cp, ls["ol"], shape2, "OLLS"),
         _f32(cp, ss["var"], shape2, "VARSS"),
         _f32(cp, ss["con"], shape2, "CONSS"),
         _stack4(cp, ss["oa"], shape2, "OASS"),
         _stack4(cp, ss["ol"], shape2, "OLSS"),
         _f32(cp, sina, shape2, "sina"), _f32(cp, cosa, shape2, "cosa"),
         _f32(cp, xland, shape2, "xland"), _f32(cp, br, shape2, "br"),
         _f32(cp, pblh, shape2, "pblh"), kpbl,
         rstoch if rstoch is not None else _NULL, du, dv,
         dtau if dtau is not None else _NULL,
         dsfc if dsfc is not None else _NULL,
         np.float32(dx), np.float32(dt), _G, _CP, _R_D, _EP_1, _PI,
         np.int32(kpblmax), np.int32(nz), np.int32(ny), np.int32(nx)))
    if diagnostics is not None:
        for m, comp in enumerate(GSL_DIAGNOSTIC_NAMES):
            diagnostics[f"DTAUX3D_{comp}"] = dtau[2 * m]
            diagnostics[f"DTAUY3D_{comp}"] = dtau[2 * m + 1]
            diagnostics[f"DUSFCG_{comp}"] = dsfc[2 * m]
            diagnostics[f"DVSFCG_{comp}"] = dsfc[2 * m + 1]


@dataclass
class TerrainDrag:
    """One domain's topo_wind coefficients and gwd_opt statistics, on device.

    Built once by :func:`build_terrain_drag` from the domain's static fields;
    the physics driver hands ``ctopo``/``ctopo2`` to YSU and calls
    :meth:`apply_gwd` after its PBL scheme.
    """

    topo_wind: int
    gwd_opt: int
    ctopo: object = None
    ctopo2: object = None
    gwd: dict | None = None
    kpblmax: int | None = None
    #: gwd_opt = 1 only: the generation and WRF v4.8.0's two keys.
    gwdo_version: str = DEFAULT_GWDO_VERSION
    gwd_dx_factor: float = DEFAULT_GWD_DX_FACTOR
    gwd_if_nonhyd: bool = DEFAULT_GWD_IF_NONHYD

    def geography(self) -> dict:
        """Static device arrays that a tiled rank gathers column for column.

        Carry the finished topo_wind coefficients, not the raw terrain:
        recomputing its Laplacian at a tile edge would clamp an interior
        neighbour and change the drag.  ``kpblmax`` depends only on the
        domain's shared eta coordinate and has no horizontal extent.
        """
        arrays = {}
        if self.topo_wind:
            arrays.update({"terrain_drag/ctopo": self.ctopo,
                           "terrain_drag/ctopo2": self.ctopo2})
        if self.gwd_opt:
            arrays.update({f"terrain_drag/gwd/{name}": self.gwd[name]
                           for name in gwd_fields(self.gwd_opt,
                                                  self.gwdo_version)})
        return arrays

    def apply_gwd(self, atmosphere, du, dv, *, sina, cosa, xland, br, pblh,
                  kpbl, dx: float, dt: float, rstoch=None) -> None:
        """Add gwd_opt's drag to the PBL momentum tendencies, in place.

        ``rstoch`` is the SPP PBL pattern's lowest level under spp_pbl = 1;
        WRF hands it to gwd_opt = 3 only (``gwdo`` takes no pattern).
        """
        if self.gwd_opt == 1:
            g = self.gwd
            launch_gwdo(atmosphere, du, dv, var=g["VAR"], con=g["CON"],
                        oa=[g[f"OA{m}"] for m in range(1, 5)],
                        ol=[g[f"OL{m}"] for m in range(1, 5)],
                        sina=sina, cosa=cosa, dx=dx, dt=dt,
                        gwdo_version=self.gwdo_version,
                        omax=g.get("ELVMAX"),
                        dx_factor=self.gwd_dx_factor,
                        if_nonhyd=self.gwd_if_nonhyd)
        elif self.gwd_opt == 3:
            g = self.gwd
            ls = {"var": g["VARLS"], "con": g["CONLS"],
                  "oa": [g[f"OA{m}LS"] for m in range(1, 5)],
                  "ol": [g[f"OL{m}LS"] for m in range(1, 5)]}
            ss = {"var": g["VARSS"], "con": g["CONSS"],
                  "oa": [g[f"OA{m}SS"] for m in range(1, 5)],
                  "ol": [g[f"OL{m}SS"] for m in range(1, 5)]}
            launch_gwdo_gsl(atmosphere, du, dv, ls=ls, ss=ss, sina=sina,
                            cosa=cosa, xland=xland, br=br, pblh=pblh,
                            kpbl=kpbl, kpblmax=int(self.kpblmax), dx=dx,
                            dt=dt, rstoch=rstoch)


def build_terrain_drag(*, topo_wind: int, gwd_opt: int, static, ht, xland,
                       znu, gwdo_version: str = DEFAULT_GWDO_VERSION,
                       gwd_dx_factor: float = DEFAULT_GWD_DX_FACTOR,
                       gwd_if_nonhyd: bool = DEFAULT_GWD_IF_NONHYD
                       ) -> TerrainDrag | None:
    """The domain's :class:`TerrainDrag`, or ``None`` with both options off.

    ``static`` maps the field names of :data:`TOPO_WIND_FIELDS` and
    :data:`GWD_FIELDS` to (ny, nx) arrays.  A field the options need and the
    mapping lacks is refused by name: WRF itself stops on topo_wind = 1
    without VAR_SSO (start_em.F:222-223), and without the statistics the
    drag schemes would run on zeros, which turns them off without saying so.
    """
    import cupy as cp

    topo_wind, gwd_opt = int(topo_wind), int(gwd_opt)
    if topo_wind not in TOPO_WIND_OPTIONS:
        raise ValueError(f"topo_wind must be one of {TOPO_WIND_OPTIONS}")
    if gwd_opt not in GWD_OPTIONS:
        raise ValueError(f"gwd_opt must be one of {GWD_OPTIONS}")
    gwdo_version, gwd_dx_factor, gwd_if_nonhyd = gwdo_options(
        gwdo_version=gwdo_version, dx_factor=gwd_dx_factor,
        if_nonhyd=gwd_if_nonhyd)
    if topo_wind == 0 and gwd_opt == 0:
        return None
    need = drag_static_fields(topo_wind, gwd_opt, gwdo_version)
    have = static if static is not None else {}
    missing = [name for name in need if have.get(name) is None]
    if missing:
        from woof.static.orographic import elvmax_missing_note
        raise ValueError(
            f"topo_wind = {topo_wind} / gwd_opt = {gwd_opt} need the static "
            f"field(s) {', '.join(missing)}, which this domain's prepared "
            "static data does not carry; prepare it again with the option "
            "set, so the static builder adds them from WPS_GEOG's "
            "orographic datasets" + elvmax_missing_note(missing))
    ht = cp.asarray(ht, dtype=cp.float32)
    ny, nx = ht.shape
    drag = TerrainDrag(topo_wind=topo_wind, gwd_opt=gwd_opt,
                       gwdo_version=gwdo_version, gwd_dx_factor=gwd_dx_factor,
                       gwd_if_nonhyd=gwd_if_nonhyd)
    if topo_wind:
        drag.ctopo, drag.ctopo2, _ = topo_wind_coefficients(
            ht, xland, topo_wind=topo_wind,
            var_sso=have.get("VAR_SSO") if topo_wind == 1 else None,
            var2d=have.get("VAR") if topo_wind == 2 else None)
    if gwd_opt:
        drag.gwd = {name: _f32(cp, have[name], (ny, nx), name)
                    for name in gwd_fields(gwd_opt, gwdo_version)}
        if gwd_opt == 3:
            drag.kpblmax = gsl_kpblmax(cp.asnumpy(cp.asarray(znu)))
    return drag
