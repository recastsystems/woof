"""ENERGY's own composition, evaluated for every land column at once.

``module_sf_noahmplsm.F:1741-2396``.  :mod:`woof.core.noahmp_energy` holds the
transcription the unmodified-WRF fixtures pin: one CPython frame per column,
Python floats, an explicit ``f32`` rounding call after every operation, and
ordinary ``if`` statements.  Measured with ``cProfile`` at 352 land columns it
is **48% of a Noah-MP land-surface call**, warm and snow alike, and it does not
amortise across columns because there is nothing in it that is shared.

This module is the same arithmetic over the column axis.  It is not a second
port of the physics -- it is the same statements with the array spelling of
each operation, in the same order, and the scalar module remains the authority
that ``tests/test_noahmp_energy_slab.py`` compares every field against.

Four segments, because ENERGY is a composition and the leaves interrupt it
====================================================================
The scalar generator suspends at six physical leaf calls.  Vectorised, the
composition becomes four straight-line segments with the batched leaves
between them -- the same order, the same values, no per-column control flow
left anywhere:

======== ==========================================================
seg1     entry to THERMOPROP: UR, VAI, FSNO, the roughness and
         displacement geometry, the urban override, ZLVL, CWP
seg2     RADIATION's return to VEGE_FLUX: the emissivities, BTRAN,
         the ground surface resistance, the psychrometric constants
seg3     BARE_FLUX's return: the tile average, FIRE, EMISSI, TRAD,
         APAR, PSN, ACC_SSOIL and the soil-column averaging
seg4     PHASECHANGE's return: EFLXB's scaling and the slots WRF
         leaves undefined, which woof defines as zero
======== ==========================================================

Where a branch became a mask, and where it could not
----------------------------------------------------
Every ``if`` on a per-column value is a :func:`cupy.where`.  Two of them need
their guarded expression to stay finite on the columns the mask discards, and
both are written with a substituted denominator rather than left to produce a
NaN that the mask happens to throw away:

* :2068 ``BDSNO = SNEQV/SNOWH`` divides by a snow depth that is exactly zero on
  every snow-free column.
* :2201's ``PSI`` and :2186's ``L_RSURF`` divide by ``SMCMAX``, which is
  non-zero for every admitted soil category, so those are left alone.

The BTRAN loop (:2154-2161) keeps its layer loop.  ``NROOT`` varies per column
and FP32 addition is not associative, so the accumulation runs in layer order
with the out-of-root layers contributing an exact ``+0.0``; only the column
axis inside it is vectorised.  ``-ZSOIL(NROOT)`` is a per-column gather.

``min`` and ``max`` are :func:`woof.core.noahmp_slab_libm.fmn`/``fmx``, not
``cupy.minimum``/``maximum``: the scalar spelling returns the second argument
on a tie and the CuPy builtins do not, which is observable on a signed zero.

What is deliberately refused rather than approximated
------------------------------------------------------
``ICE != 0`` (the glacier ground-emissivity leg at :2145-2147) and ``IST != 1``
(the lake legs at :2076-2082 and :2178-2181) are refused here exactly as the
scalar module refuses them, and for the same reason: no fixture in this tree
admits either, so there is nothing that could hold a transcription of them at
``max_ulp 0``.  A slab is refused if *any* column carries them, with the
column index in the message.

The emitted-longwave check at :2323-2329 is where WRF calls
``wrf_error_fatal``.  A slab cannot raise for one column and continue for the
rest, so it raises for the slab and names the first offending column, which is
what the scalar path does with the same message.
"""

from __future__ import annotations

import numpy as np

from woof.core.noahmp_energy import (CPAIR, GRAV, HSUB, HVAP, MPE, NSNOW,
                                      NSOIL, RW, SB, TFRZ, Z0)
from woof.core.noahmp_slab_libm import (fmn, fmx, slab_expf, slab_powf,
                                         slab_sqrtf, slab_tanhf)


def _f32(value):
    return np.float32(value)


_ZERO = _f32(0.0)
_ONE = _f32(1.0)


def _layer(sequence, k, nsnow: int = NSNOW):
    """Fortran ``seq(k)`` for a slab array declared ``(-nsnow+1:nsoil)``.

    The column axis is first and the layer axis last, so this is a column of
    the slab, not a column of the model.
    """
    return sequence[:, k + nsnow - 1]


def _refuse_unadmitted(ice, ist):
    """``ICE != 0`` and ``IST != 1`` are refused, with the column named."""
    import cupy as cp
    from woof.core import health_ledger

    ledger = health_ledger.active()
    if ledger is not None:
        def refuse(flags):
            branch = "glacier (ICE != 0)" if flags & 1 else "lake (IST != 1)"
            raise NotImplementedError(
                f"Noah-MP land ENERGY received a {branch} column; "
                "its dedicated solver must own that column"
                + health_ledger.deferred_note())
        flags = ((cp.asarray(ice) != 0).any().astype(cp.uint32)
                 | ((cp.asarray(ist) != 1).any().astype(cp.uint32) << 1))
        ledger.record("noahmp.energy.land_identity", flags.reshape(1), refuse)
        return

    bad_ice = cp.asnumpy(cp.asarray(ice) != 0)
    if bad_ice.any():
        raise NotImplementedError(
            f"land column {int(np.argmax(bad_ice))} has ICE != 0, the glacier "
            "column: it selects the ground-emissivity leg at "
            "module_sf_noahmplsm.F:2145-2147 and, in NOAHMP_SFLX, "
            "NOAHMP_GLACIER.  No fixture in this tree admits it")
    bad_ist = cp.asnumpy(cp.asarray(ist) != 1)
    if bad_ist.any():
        raise NotImplementedError(
            f"land column {int(np.argmax(bad_ist))} has IST != 1, a lake "
            "column: it selects the Z0MG leg at :2076-2082 and the "
            "RSURF/RHSUR leg at :2178-2181.  No fixture in this tree admits it")


# ---------------------------------------------------------------------------
# seg1: entry (:2033) to the THERMOPROP call (:2119)
# ---------------------------------------------------------------------------

def energy_seg1(s, *, tanh_profile="glibc_2_39"):
    """UR, VAI, FSNO, the roughness geometry, the urban override, ZLVL, CWP.

    ``s`` maps the scalar generator's argument names to CuPy arrays over the
    column axis.  The returned dict is what seg2 and the THERMOPROP batch read.
    """
    import cupy as cp

    _refuse_unadmitted(s["ice"], s["ist"])

    uu, vv = s["uu"], s["vv"]
    snowh, sneqv = s["snowh"], s["sneqv"]

    # -- :2057  UR.  The exponent is a REAL literal, so this is powf. -------
    ur = fmx(slab_sqrtf(slab_powf(uu, _f32(2.0)) + slab_powf(vv, _f32(2.0))),
             _ONE)

    # -- :2061-2063  vegetated or not ---------------------------------------
    vai = s["elai"] + s["esai"]
    veg = vai > _ZERO

    # -- :2067-2073  ground snow-cover fraction [Niu and Yang, 2007] ---------
    # The scalar form guards this whole block on SNOWH > 0.  Substituting the
    # denominator keeps the discarded columns finite rather than relying on a
    # NaN being masked away, which is the same value and a smaller surface.
    has_snow = snowh > _ZERO
    safe_snowh = cp.where(has_snow, snowh, _ONE)
    bdsno = sneqv / safe_snowh
    fmelt = slab_powf(bdsno / _f32(100.0), s["mfsno"])
    fsno = cp.where(has_snow,
                    slab_tanhf(safe_snowh / (s["scffac"] * fmelt), profile=tanh_profile),
                    _ZERO)

    # -- :2077-2085  ground roughness length (IST == 1 leg) ------------------
    z0mg = (_f32(Z0) * (_ONE - fsno)) + (fsno * s["z0sno"])

    # -- :2089-2097  roughness length and displacement height ----------------
    zpdg = snowh
    zpd_veg = _f32(0.65) * s["hvt"]
    zpd_veg = cp.where(snowh > zpd_veg, snowh, zpd_veg)
    z0m = cp.where(veg, s["z0mvt"], z0mg)
    zpd = cp.where(veg, zpd_veg, zpdg)

    # -- :2101-2106  urban override -----------------------------------------
    urban = s["urban_flag"]
    urban_zpd = _f32(0.65) * s["hvt"]
    z0mg = cp.where(urban, s["z0mvt"], z0mg)
    zpdg = cp.where(urban, urban_zpd, zpdg)
    z0m = cp.where(urban, z0mg, z0m)
    zpd = cp.where(urban, zpdg, zpd)

    # -- :2109-2110 ----------------------------------------------------------
    zlvl = fmx(zpd, s["hvt"]) + s["zref"]
    # Unreachable for ZREF > 0; transcribed because the source has it.
    zlvl = cp.where(zpdg >= zlvl, zpdg + s["zref"], zlvl)

    return {
        "ur": ur, "vai": vai, "veg": veg, "fsno": fsno,
        "z0mg": z0mg, "zpdg": zpdg, "z0m": z0m, "zpd": zpd,
        "zlvl": zlvl, "cwp": s["cwpvt"],                     # :2115
    }


# ---------------------------------------------------------------------------
# seg2: RADIATION's return (:2139) to the VEGE_FLUX call (:2238)
# ---------------------------------------------------------------------------

def energy_seg2(s, g, *, opt_rsf=1, opt_btr=1):
    """The emissivities, BTRAN, RSURF/RHSUR and the psychrometric constants.

    ``g`` is seg1's output; ``s`` is the entry state plus RADIATION's.
    RSF=4 requires actual transferred ``s['rsurf_snow']``; it blends the
    SZ09 resistance with snow before the native dry and urban caps.
    """
    import cupy as cp

    if type(opt_rsf) is not int or opt_rsf not in (1, 2, 3, 4):
        raise ValueError("ENERGY opt_rsf must be scalar int in (1, 2, 3, 4)")
    if type(opt_btr) is not int or opt_btr not in (1, 2, 3):
        raise ValueError("ENERGY opt_btr must be scalar int in (1, 2, 3)")

    fsno = g["fsno"]

    # -- :2139-2147  emissivities (ICE == 0 leg) -----------------------------
    emv = _ONE - slab_expf(-(s["elai"] + s["esai"]) / _ONE)
    emg = (s["eg"] * (_ONE - fsno)) + (s["snow_emis"] * fsno)

    # -- :2151-2173  BTRAN (Noah, CLM or SSiB) ------------------------------
    # The layer loop stays a loop: FP32 addition is not associative and NROOT
    # varies per column, so the accumulation runs in layer order with the
    # out-of-root layers contributing an exact +0.0.
    nroot = s["nroot"]
    sh2o = s["sh2o"]
    smcwlt, smcref = s["smcwlt"], s["smcref"]
    from woof.core.noahmp_slab_libm import root_fraction_slabs
    btran, btrani = root_fraction_slabs(s, NSNOW, NSOIL, MPE, opt_btr=opt_btr)

    # -- :2177-2206  ground surface resistance (IST == 1, RSF == 1..4) ------
    smcmax0 = s["smcmax"][:, 0]
    sh2o0 = sh2o[:, 0]
    if opt_rsf in (2, 3):
        from woof.core.noahmp_slab_libm import sellers_surface_resistance
        rsurf = sellers_surface_resistance(sh2o0, smcmax0, fsno, opt_rsf=opt_rsf)
    else:
        sat = fmn(_ONE, sh2o0 / smcmax0)
        l_rsurf = ((-s["zsoil"][:, 0])
                   * (slab_expf(slab_powf(_ONE - sat, s["rsurf_exp"])) - _ONE)) \
            / (_f32(2.71828) - _ONE)
        d_rsurf = ((_f32(2.2e-5) * smcmax0) * smcmax0) \
            * slab_powf(_ONE - (s["smcwlt"][:, 0] / smcmax0),
                        _f32(2.0) + (_f32(3.0) / s["bexp"][:, 0]))
        rsurf = l_rsurf / d_rsurf
    if opt_rsf == 4:
        from woof.core.noahmp_slab_libm import snow_surface_resistance
        rsurf = snow_surface_resistance(rsurf, fsno, s["rsurf_snow"])
    # :2199
    rsurf = cp.where((sh2o0 < _f32(0.01)) & (s["snowh"] == _ZERO),
                     _f32(1.0e6), rsurf)
    psi = (-s["psisat"][:, 0]) \
        * slab_powf(fmx(_f32(0.01), sh2o0) / smcmax0, -s["bexp"][:, 0])
    from woof.core.noahmp_slab_libm import slab_add_product
    # The exponential may be subnormal. CuPy's multiply would flush it
    # before BARE_FLUX scales RHSUR into QSFC; retain both WRF roundings.
    rhsur = slab_add_product(fsno, _ONE - fsno,
                            slab_expf((psi * _f32(GRAV)) / (_f32(RW) * s["tg"])))
    # :2204-2206
    rsurf = cp.where(s["urban_flag"] & (s["snowh"] == _ZERO),
                     _f32(1.0e6), rsurf)

    # -- :2211-2227  psychrometric constants ---------------------------------
    frozen_canopy = ~(s["tv"] > _f32(TFRZ))
    latheav = cp.where(frozen_canopy, _f32(HSUB), _f32(HVAP))
    gammav = (_f32(CPAIR) * s["sfcprs"]) / (_f32(0.622) * latheav)
    frozen_ground = ~(s["tg"] > _f32(TFRZ))
    latheag = cp.where(frozen_ground, _f32(HSUB), _f32(HVAP))
    gammag = (_f32(CPAIR) * s["sfcprs"]) / (_f32(0.622) * latheag)

    # -- :2238  which columns run VEGE_FLUX at all ---------------------------
    tile_veg = g["veg"] & (s["fveg"] > _ZERO)

    return {
        "emv": emv, "emg": emg, "btran": btran, "btrani": btrani,
        "rsurf": rsurf, "rhsur": rhsur,
        "gammav": gammav, "gammag": gammag,
        "latheav": latheav, "latheag": latheag,
        "frozen_canopy": frozen_canopy, "frozen_ground": frozen_ground,
        "tile_veg": tile_veg,
    }


# ---------------------------------------------------------------------------
# seg3: BARE_FLUX's return (:2282) to the TSNOSOI call (:2349)
# ---------------------------------------------------------------------------

def energy_seg3(s, g, e, v, b):
    """The tile average, FIRE, EMISSI, TRAD, APAR, PSN and ACC_SSOIL.

    ``v`` is VEGE_FLUX's answer broadcast over every column -- the vegetated
    columns carry its result and the rest carry the zeros :2033-2053 wrote,
    which is exactly what the scalar path holds at this point.  ``b`` is
    BARE_FLUX's, which every land column has.
    """
    import cupy as cp

    from woof.core.noahmp_slab_libm import energy_average_slabs

    result = energy_average_slabs(s, g, e, v, b, SB)
    fire = result.pop("_fire")
    from woof.core.noahmp_surface_device import active_probe
    from woof.core import health_ledger
    ledger = health_ledger.active()
    if ledger is not None:
        def refuse(_flags):
            raise ValueError(
                "Noah-MP emitted longwave <= 0: SHDFAC is inconsistent "
                "with LAI (module_sf_noahmplsm.F:2323-2329)"
                + health_ledger.deferred_note())
        ledger.record("noahmp.energy.longwave",
                      active_probe(fire <= _ZERO).any().astype(cp.uint32).reshape(1), refuse)
    else:
        bad = cp.asnumpy(active_probe(fire <= _ZERO))
        if bad.any():
            raise ValueError(
                f"land column {int(np.argmax(bad))}: emitted longwave <= 0: "
                "SHDFAC is inconsistent with LAI "
                "(module_sf_noahmplsm.F:2323-2329 calls wrf_error_fatal here)")
    result["trad"] = slab_powf(result.pop("_trad_base"), _f32(0.25))
    return result


# ---------------------------------------------------------------------------
# seg4: PHASECHANGE's return to ENERGY's return
# ---------------------------------------------------------------------------


_ENERGY_MASK_KERNELS = {}


def energy_seg4(s, g, e, out):
    """Batch snow, root and night masks, preserving the existing arithmetic."""
    import cupy as cp
    from woof.core.noahmp_kernel_sources import compile_generated_slab_kernel

    names = ("hcpct", "imelt", "snicev", "snliqv", "epore",
             "fsrv", "fsrg", "bgap", "wgap")
    n = int(s["isnow"].size)
    nlay = NSNOW + NSOIL
    # Preserve the general array spelling for wider or unusual carriers.
    if (any(out[k].dtype != (cp.int32 if k == "imelt" else cp.float32)
            for k in names) or e["btrani"].dtype != cp.float32
            or s["isnow"].dtype != cp.int32 or s["nroot"].dtype != cp.int32
            or s["isnow"].shape != (n,) or s["nroot"].shape != (n,)
            or s["cosz"].shape != (n,)
            or any(out[k].shape != (n, width) for k, width in
                   (("hcpct", nlay), ("imelt", nlay), ("snicev", NSNOW),
                    ("snliqv", NSNOW), ("epore", NSNOW)))
            or e["btrani"].shape != (n, NSOIL)
            or any(out[k].shape != (n,) for k in names[-4:])):
        return _energy_seg4_reference(s, g, e, out)

    # Keep these exact CuPy operations, including their rounding and FTZ
    # behavior. The generated kernel below performs only predicate tests and
    # word copies, so signed zeros and NaN payloads pass through unchanged.
    result = {"eflxb": out["eflxb"] * e["dt_soil"]}
    night = cp.ascontiguousarray(s["cosz"] <= _ZERO)
    values = tuple(cp.ascontiguousarray(out[k]) for k in names)
    roots = cp.ascontiguousarray(e["btrani"])
    isnow = cp.ascontiguousarray(s["isnow"])
    nroot = cp.ascontiguousarray(s["nroot"])
    result.update({k: cp.empty_like(v) for k, v in zip(names, values)})
    result["btrani"] = cp.empty_like(roots)
    if n:
        device = cp.cuda.runtime.getDevice()
        key = (device, NSNOW, NSOIL)
        kernel = _ENERGY_MASK_KERNELS.get(key)
        if kernel is None:
            args = ", ".join(f"const unsigned int* v{k}, unsigned int* o{k}"
                             for k in range(len(names)))
            lines = []
            for k, width in enumerate((nlay, nlay, NSNOW, NSNOW, NSNOW)):
                for j in range(width):
                    lines.append(
                        f"o{k}[{width}LL*c+{j}] = "
                        f"({j-NSNOW+1} > isnow[c]) ? "
                        f"v{k}[{width}LL*c+{j}] : 0u;")
            for k in range(5, len(names)):
                lines.append(f"o{k}[c] = night[c] ? 0u : v{k}[c];")
            for j in range(NSOIL):
                lines.append(
                    f"rout[{NSOIL}LL*c+{j}] = ({j+1} <= nroot[c]) ? "
                    f"roots[{NSOIL}LL*c+{j}] : 0u;")
            source = ('extern "C" __global__ void noahmp_energy_masks('
                      'int n, const int* isnow, const int* nroot, '
                      'const bool* night, const unsigned int* roots, '
                      f'unsigned int* rout, {args}) {{ '
                      'int c=blockDim.x*blockIdx.x+threadIdx.x; '
                      'if(c>=n)return; ' + "\n".join(lines) + '}')
            kernel = compile_generated_slab_kernel(source, "noahmp_energy_masks")
            _ENERGY_MASK_KERNELS[key] = kernel
        arrays = tuple(a for k, v in zip(names, values)
                       for a in (v, result[k]))
        kernel(((n+127)//128,), (128,),
               (np.int32(n), isnow, nroot, night, roots, result["btrani"], *arrays))
    return result


def _energy_seg4_reference(s, g, e, out):
    """EFLXB's scaling and the slots WRF leaves undefined.

    WRF does not initialise the layers above ISNOW; woof produces the defined
    0.0 rather than reproducing an uninitialised read, which is the standing
    rule for this port.  The scalar module does it with per-layer tuple
    comprehensions and this does it with one comparison against the layer
    ladder.
    """
    import cupy as cp

    isnow = s["isnow"]
    nlay = NSNOW + NSOIL
    # Fortran layer index of each slot: slot k is layer k - NSNOW + 1.
    layer = cp.arange(-NSNOW + 1, NSOIL + 1, dtype=cp.int32)[None, :]
    live_full = layer > isnow[:, None]
    live_snow = live_full[:, :NSNOW]

    eflxb = out["eflxb"] * e["dt_soil"]

    hcpct = cp.where(live_full, out["hcpct"], _ZERO)
    imelt = cp.where(live_full, out["imelt"], 0)
    snicev = cp.where(live_snow, out["snicev"], _ZERO)
    snliqv = cp.where(live_snow, out["snliqv"], _ZERO)
    epore = cp.where(live_snow, out["epore"], _ZERO)

    root = cp.arange(1, NSOIL + 1, dtype=cp.int32)[None, :]
    btrani = cp.where(root <= s["nroot"][:, None], e["btrani"], _ZERO)

    night = s["cosz"] <= _ZERO
    fsrv = cp.where(night, _ZERO, out["fsrv"])
    fsrg = cp.where(night, _ZERO, out["fsrg"])
    bgap = cp.where(night, _ZERO, out["bgap"])
    wgap = cp.where(night, _ZERO, out["wgap"])

    assert hcpct.shape[1] == nlay
    return {
        "eflxb": eflxb, "hcpct": hcpct, "imelt": imelt, "snicev": snicev,
        "snliqv": snliqv, "epore": epore, "btrani": btrani,
        "fsrv": fsrv, "fsrg": fsrg, "bgap": bgap, "wgap": wgap,
    }


__all__ = ["energy_seg1", "energy_seg2", "energy_seg3", "energy_seg4"]
