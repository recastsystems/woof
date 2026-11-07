"""Device-resident, statement-rounded legacy RRTMG wrapper preparation.

Only configuration scalars (trace gases and flags) are evaluated on the
host. Column arithmetic runs in the CUDA unit. The direct NVRTC route
preserves subnormal operands, results and comparisons; a live probe guards
each device before first use. McICA retains its existing synchronizations.
"""

from functools import lru_cache
from woof.core.device_cache import cuda_cache
from inspect import signature
from pathlib import Path

import numpy as np

from woof.core import rrtmg_legacy_prep as ref
from woof.core import rrtmg_mcica

__all__ = ["lwrad_prep_batch_device", "swrad_prep_batch_device"]

_SIGNATURES = {False: signature(ref.lwrad_prep_batch),
               True: signature(ref.swrad_prep_batch)}
_PREFLIGHTED = set()


@lru_cache(maxsize=None)
def _gpu_module(device, cloud_form="wrf_461", radii_units="meters"):
    """Direct NVRTC PTX load, no RawModule and no explicit -arch option.

    CuPy supplies the architecture itself. Duplicating it fails on NVRTC
    13. The explicit unflushed route is also needed for comparisons.
    """
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    source = (Path(__file__).parent / "kernels" /
              "rrtmg_legacy_prep.cu").read_text(encoding="utf-8")
    from woof.physics_params import edit_kernel_source, note_compiled
    source = edit_kernel_source("rrtmg_legacy_prep", source)
    fork_cloud = ref.cloud_form_is_fork(cloud_form)
    options = ("-std=c++17", "--ftz=false")
    if fork_cloud:
        options += ("-DRP_CLOUD_FORM=1",)
    if radii_units == "microns":
        options += ("-DRP_RADIUS_MICRONS=1",)
    with cp.cuda.Device(device):
        ptx, _ = compiler.compile_using_nvrtc(
            source, options, None,
            "rrtmg_legacy_prep.cu")
        mod = cp.cuda.function.Module()
        mod.load(ptx.encode() if isinstance(ptx, str) else ptx)
    note_compiled("rrtmg_legacy_prep", mod)
    return mod


def gpu_preflight(force=False):
    """Prove subnormal multiplication, division and comparison on-card.

    This first-use host read is intentional, like the legacy engines'
    preflight. Subsequent prep calls do not read device data except the
    SW day refusal and the existing McICA twin's own checks.
    """
    import cupy as cp
    device = cp.cuda.runtime.getDevice()
    if device in _PREFLIGHTED and not force:
        return
    x = np.array([1.e-30, 1.e-10, 1.e-39, 2.0, 0.0, -0.0], np.float32)
    y = cp.empty(5, np.float32)
    _gpu_module(device).get_function("rp_probe")(
        (1,), (32,), (cp.asarray(x), y))
    expected = np.array([x[0]*x[1], 1.0, x[2]/x[3],
                         np.maximum(x[4], x[5]),
                         np.minimum(x[4], x[5])], np.float32)
    actual = cp.asnumpy(y).view(np.uint32)
    if not np.array_equal(actual, expected.view(np.uint32)):
        raise RuntimeError("legacy prep preflight failed: uint32 got %r want %r"
                           % (actual.tolist(), expected.view(np.uint32).tolist()))
    _PREFLIGHTED.add(device)


def gpu_local_frame_bytes():
    """Return the CUDA local-frame size for every kernel in the unit."""
    import cupy as cp
    from cupy.cuda import driver
    mod = _gpu_module(cp.cuda.runtime.getDevice())
    return {name: int(driver.funcGetAttribute(driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                              mod.get_function(name).ptr))
            for name in ("rp_prep", "rp_scon", "rp_probe", "rp_day")}


@cuda_cache(maxsize=16, ready=True)
def _tables(device):
    """The legacy RRTMG reference tables on ``device`` (the current card).

    Published with their upload event: every slab on the card reads the
    same copy from its own stream, and an upload still in flight on the
    first slab's stream is not a table the others may read.
    """
    import cupy as cp
    with cp.cuda.Device(device):
        return tuple(cp.asarray(a) for a in (
            ref._RETAB, ref._PPROF, ref._TPROF, ref._O3WRK, ref._PPWRKH))


def _bands(sw, ncol, nlay):
    """The constant McICA band inputs (LW taucld 0; SW taucld 0, ssacld 1,
    asmcld 0, fsfcld 0), built on the device for this call.

    Per call, not cached: the day-column count and so the last chunk's
    width change every radiation call, and a shape-keyed cache would keep
    one device slab per width ever seen.  A fill is a memset, and these are
    the slabs legacy_radiation_vram_bytes prices in the generate phase."""
    import cupy as cp
    nb = rrtmg_mcica.NBNDSW if sw else rrtmg_mcica.NBNDLW
    zero = cp.zeros((nb, ncol, nlay), np.float32)
    if sw:
        return (zero, cp.ones(zero.shape, np.float32), zero, zero)
    return (zero,)


def _profile(cp, value, shape, name):
    if not isinstance(value, cp.ndarray) or value.dtype != np.float32:
        raise TypeError(f"{name} must be a CuPy float32 array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    return cp.ascontiguousarray(value)


def _surface(cp, value, ncol, name):
    if np.isscalar(value):
        return cp.full(ncol, np.float32(value), np.float32)
    if isinstance(value, cp.ndarray) and value.ndim == 0:
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be CuPy float32")
        return cp.broadcast_to(value, (ncol,)).copy()
    return _profile(cp, value, (ncol,), name)


def _prep(sw, kw):
    import cupy as cp
    # Deferral is an adapter-side option, not an argument of the NumPy
    # oracle whose signature the rest of ``kw`` binds against.
    kw = dict(kw)
    defer = bool(kw.pop("defer_subcolumns", False))
    cam_lradius = kw.pop("cam_lradius", None)
    cam_iradius = kw.pop("cam_iradius", None)
    cam = int(kw.get("mp_physics", 0)) == 11
    if cam and (cam_lradius is None or cam_iradius is None):
        raise ValueError("CAMMGMP radiation requires LRADIUS and IRADIUS")
    radii_units = kw.pop("radii_units", "meters")
    if radii_units not in ("meters", "microns"):
        raise ValueError("radii_units must be meters or microns")
    bound = _SIGNATURES[sw].bind(**kw)
    bound.apply_defaults()
    k = bound.arguments
    name = "swrad_prep_batch" if sw else "lwrad_prep_batch"
    hc, hi, hs = (int(k[n]) for n in ("has_reqc", "has_reqi", "has_reqs"))
    ref._require_radii(name, hc, hi, hs, k["re_cloud"], k["re_ice"], k["re_snow"])
    if sw:
        ref._mp_guard(k["mp_physics"], allowed_extra=(85,))
        ref._require_spectral_albedo(k["sf_surface_physics"], k["spectral_albedo"])
    else:
        ref._mp_guard(k["mp_physics"])
    p = k["p3d"]
    if not isinstance(p, cp.ndarray) or p.ndim != 2:
        raise ValueError(f"{name} expects (ncol, kte) device profiles")
    ncol, nk = p.shape
    if not ncol:
        raise ValueError("empty batch (ncol == 0)")
    if not nk:
        raise ValueError("empty vertical profile")
    nl = nk+1 if sw else int(k["nlayers"])
    if nl < nk+1:
        raise ValueError("nlayers must be at least kte+1")
    prof = {n: _profile(cp, k[n], (ncol, nk+1 if n in ("p8w", "t8w") else nk), n)
            for n in ("p3d", "p8w", "t3d", "t8w", "dz8w", "qv3d")}
    zero = cp.zeros((ncol, nk), np.float32)
    for n in ("qc3d", "qr3d", "qi3d", "qs3d", "qg3d", "cldfra3d",
              "re_cloud", "re_ice", "re_snow"):
        prof[n] = zero if k[n] is None else _profile(cp, k[n], zero.shape, n)
    prof["o33d"] = (_profile(cp, k["o33d"], zero.shape, "o33d")
                    if k["o3input"] == 2 else zero)
    surf = {n: _surface(cp, k[n], ncol, n)
            for n in (("tsk", "albedo", "xland", "xice", "snow", "xlat",
                       "xcoszen", "solcon", "obscur") if sw else
                      ("tsk", "emiss", "xland", "xice", "snow", "xlat"))}
    device = cp.cuda.runtime.getDevice()
    gpu_preflight()
    mod = _gpu_module(device, k["rrtmg_cloud_optics_form"], radii_units)
    if sw:
        night = cp.empty(ncol, cp.uint8)
        mod.get_function("rp_day")(((ncol+127)//128,), (128,),
                                    (ncol, surf["xcoszen"], night))
        def _night_verdict(any_night):
            if bool(any_night):
                raise ValueError("night column (coszen <= 0) in batch: swrad_prep_batch "
                                 "takes pre-gathered day columns only; route night "
                                 "columns through swrad_night_outputs")
        # Read now, or once per radiation call (deferred_device_checks).
        from woof.core.deferred_device_checks import check
        check(night.any(), _night_verdict)
    inflg, iceflg = 2, 3
    if k["icloud"] != 0:
        if hc: inflg = 3
        if hi: inflg, iceflg = 4, 4
        if hs or (not hs and hi and hc): inflg, iceflg = 5, 5
    icld = int(k["cldovrlp"])
    juldat = int(np.float32(k["julian"]))
    # Keep the oracle's scalar helper call, including override validation.
    if sw:
        co2, ch4, n2o, o2 = ref.option4_trace_gases(k["yr"], k["trace_gas_overrides"])
        gases = dict(co2=co2, ch4=ch4, n2o=n2o, o2=o2)
    else:
        gases = ref.lw_trace_gases(k["yr"], k["trace_gas_overrides"])
    out = {"ncol": ncol, "nlay": nl, "icld": icld, "juldat": juldat}
    for n in ("plev", "tlev"):
        out[n] = cp.empty((ncol, nl+1), np.float32)
    for n in ("play", "tlay", "hgt", "h2ovmr", "o3vmr"):
        out[n] = cp.empty((ncol, nl), np.float32)
    for n in ("o31d", "pdel"):
        out[n] = cp.empty((ncol, nk), np.float32)
    cloud = {n: cp.empty((ncol, nl), np.float32)
             for n in ("cldfrac", "clwpth", "ciwpth", "cswpth", "rel", "rei", "res")}
    args = (ncol, nk, nl, int(sw), int(k["icloud"]), int(k["warm_rain"]),
            *(int(k[n]) for n in ("f_qc", "f_qr", "f_qi", "f_qs")),
            hc, hi, hs, inflg, iceflg, int(k["o3input"]),
            (2 if int(k["mp_physics"]) == 95 else int(ref._is_ferrier(k["mp_physics"]))), int(cam), np.float32(k["g"]),
            *(prof[n] for n in ("p3d", "p8w", "t3d", "t8w", "dz8w", "qv3d",
                               "qc3d", "qr3d", "qi3d", "qs3d", "cldfra3d", "o33d",
                               "re_cloud", "re_ice", "re_snow")),
            zero if not cam else _profile(cp, cam_lradius, zero.shape, "cam_lradius"),
            zero if not cam else _profile(cp, cam_iradius, zero.shape, "cam_iradius"),
            *(surf[n] for n in ("xland", "xice", "snow")), *_tables(device),
            *(out[n] for n in ("plev", "tlev", "play", "tlay", "hgt", "h2ovmr",
                              "o3vmr", "o31d", "pdel")),
            *(cloud[n] for n in ("cldfrac", "clwpth", "ciwpth", "cswpth", "rel", "rei", "res")))
    mod.get_function("rp_prep")(((ncol+127)//128,), (128,), args)
    out["tsfc"] = surf["tsk"].copy()
    for n, value in gases.items():
        out[n+"vmr"] = cp.full((ncol, nl), value, np.float32)
    # Resolve the module attribute on every call, including forecast traps.
    generate = k["subcolumn_generator"]
    if generate is None:
        generate = (rrtmg_mcica.gpu_generate_sw_subcolumns if sw else
                    rrtmg_mcica.gpu_generate_lw_subcolumns)
    mc_keys = ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl",
               "reicmcl", "relqmcl", "resnmcl")
    if sw:
        mc_keys += ("ssacmcl", "asmcmcl", "fsfcmcl")
    hgt = out["hgt"]
    idcor = int(k["idcor"])
    seed = rrtmg_mcica.SW_PERMUTESEED if sw else rrtmg_mcica.LW_PERMUTESEED
    ngpt = rrtmg_mcica.NGPTSW if sw else rrtmg_mcica.NGPTLW
    # Deferred windows of the real device generator share one states pass
    # over the whole batch (each column's stream depends on its own play
    # only), instead of a one-thread-per-column launch per window.
    base = getattr(generate, "func", generate)
    span_states = (rrtmg_mcica.gpu_subcolumn_states(
        ncol, nl, ngpt, seed, out["play"])
        if defer and icld == rrtmg_mcica.MCICA_DEVICE_ICLD
        and base in rrtmg_mcica.DEVICE_GENERATORS else None)

    def subcolumns(w0, w1):
        """The McICA slabs of prep columns [w0, w1).  Each column's
        kissvec stream depends only on that column's own play profile and
        the permute seed (rrtmg_mcica's chunking contract), so a window
        holds the same bits as the same columns of a whole-batch call."""
        rows = slice(int(w0), int(w1))
        nw = int(w1) - int(w0)
        extra = {} if span_states is None else {"states": span_states[rows]}
        mc = generate(1, nw, nl, icld, seed, 0, out["play"][rows],
                      cloud["cldfrac"][rows], cloud["ciwpth"][rows],
                      cloud["clwpth"][rows], cloud["cswpth"][rows],
                      cloud["rei"][rows], cloud["rel"][rows],
                      cloud["res"][rows], *_bands(sw, nw, nl), hgt[rows],
                      idcor, juldat, surf["xlat"][rows], layout="column",
                      **extra)
        return {name: mc[name] for name in mc_keys}

    if defer:
        # The adapter generates the slabs one engine window at a time, so
        # this prep can span many engine chunks without holding
        # (column, layer, g-point) slabs for all of them at once.
        out["mcica_window"] = subcolumns
    else:
        out.update(subcolumns(0, ncol))
    side = "sw" if sw else "lw"
    out.update({"inflg"+side: inflg, "iceflg"+side: iceflg, "liqflg"+side: 1})
    if sw:
        spectral_albedo = k.get("spectral_albedo")
        for n in ("asdir", "asdif", "aldir", "aldif"):
            out[n] = (surf["albedo"].copy() if spectral_albedo is None
                      else _surface(cp, spectral_albedo[n], ncol, n).copy())
        out.update(coszen=surf["xcoszen"].copy(), coszr=surf["xcoszen"].copy(),
                   adjes=np.float32(1), dyofyr=0)
        out["scon"] = cp.empty(ncol, np.float32)
        mod.get_function("rp_scon")(((ncol+127)//128,), (128,),
                                    (ncol, surf["solcon"], surf["obscur"], out["scon"]))
        out["mcica_inputs"] = {"play": out["play"].copy(), **cloud,
                               "hgt": out["hgt"].copy(), "icld": icld,
                               "idcor": int(k["idcor"]), "juldat": juldat,
                               "lat": surf["xlat"].copy(),
                               "permuteseed": rrtmg_mcica.SW_PERMUTESEED, "irng": 0}
        del out["hgt"]
    else:
        out["emis"] = cp.broadcast_to(surf["emiss"][:, None],
                                      (ncol, rrtmg_mcica.NBNDLW)).copy()
        out["tauaer"] = cp.zeros((ncol, nl, rrtmg_mcica.NBNDLW), np.float32)
        out["cldfrac"] = cloud["cldfrac"]
    return out


def lwrad_prep_batch_device(**kw):
    """LW prep with unchanged keys and five (column, layer, g-point) slabs.

    ``defer_subcolumns=True`` returns, in place of the McICA slabs, a
    callable ``mcica_window(w0, w1)`` that builds the slabs of prep
    columns [w0, w1) on demand (same bits as the eager slabs' rows)."""
    return _prep(False, kw)


def swrad_prep_batch_device(**kw):
    """SW day batch prep, same keywords and return keys as the NumPy oracle.

    ``defer_subcolumns=True``: see :func:`lwrad_prep_batch_device`."""
    return _prep(True, kw)
