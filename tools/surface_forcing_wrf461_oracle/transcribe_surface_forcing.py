"""Independent WRF v4.6.1 surface-forcing source transcription probes.

This module intentionally imports no ``woof`` code.  It spells the pinned
Fortran statements in NumPy float32 so tests compare the WOOF-ArWen ports with the
authority, not with a second call into the implementation under test.

Authority checkout:
  WRF tag v4.6.1, d66e442fccc04111067e29274c9f9eaccc3cef28
"""

from __future__ import annotations

import numpy as np


F = np.float32


def _array(value):
    return np.asarray(value, dtype=np.float32)


def ruc_arw_precipitation(
        *, rainbl, rainncv, snowncv, graupelncv, frzfrac, tabs, dt):
    """``module_sf_ruclsm.F:611-652`` compiled with ``EM_CORE==1``."""
    rainbl = _array(rainbl)
    rainncv = _array(rainncv)
    snowncv = _array(snowncv)
    graupelncv = _array(graupelncv)
    frzfrac = _array(frzfrac)
    tabs = _array(tabs)
    dt = F(dt)
    zero = F(0.0)
    one = F(1.0)

    prcpncliq = (rainncv * (one - frzfrac)).astype(np.float32)
    prcpncfr = (rainncv * frzfrac).astype(np.float32)
    residual = (rainbl - rainncv).astype(np.float32)
    mixed_cold = (frzfrac > zero) & (tabs < F(273.0))
    cold = tabs < F(273.0)
    prcpculiq = np.where(
        mixed_cold,
        np.maximum(zero, residual * (one - frzfrac)),
        np.where(cold, zero, np.maximum(zero, residual)),
    ).astype(np.float32)
    prcpcufr = np.where(
        mixed_cold,
        np.maximum(zero, residual * frzfrac),
        np.where(cold, np.maximum(zero, residual), zero),
    ).astype(np.float32)
    prcpms = (((prcpncliq + prcpculiq) / dt) * F(1.0e-3)).astype(
        np.float32)
    newsnms = (((prcpncfr + prcpcufr) / dt) * F(1.0e-3)).astype(
        np.float32)

    frozen = (prcpncfr + prcpcufr).astype(np.float32)
    falling = frozen > zero
    denominator = np.where(falling, frozen, one).astype(np.float32)
    snowrat = np.where(
        falling,
        np.minimum(one, np.maximum(zero, snowncv / denominator)),
        zero,
    ).astype(np.float32)
    grauprat = np.where(
        falling,
        np.minimum(one, np.maximum(zero, graupelncv / denominator)),
        zero,
    ).astype(np.float32)
    icerat = np.where(
        falling,
        np.minimum(
            one,
            np.maximum(
                zero,
                (prcpncfr - snowncv - graupelncv) / denominator,
            ),
        ),
        zero,
    ).astype(np.float32)
    curat = np.where(
        falling,
        np.minimum(one, np.maximum(zero, prcpcufr / denominator)),
        zero,
    ).astype(np.float32)
    return {
        "prcpms": prcpms,
        "newsnms": newsnms,
        "snowrat": snowrat,
        "grauprat": grauprat,
        "icerat": icerat,
        "curat": curat,
    }


def noahmp_six_rates(
        *, rainbl, sr, rainc, rainnc, rainshv, snow, graupel, hail, dt):
    """``module_sf_noahmpdrv.F:776-789`` when every MP_* is present."""
    dt = F(dt)
    prcp = (_array(rainbl) / dt).astype(np.float32)
    prcpconv = (_array(rainc) / dt).astype(np.float32)
    prcpnonc = (_array(rainnc) / dt).astype(np.float32)
    prcpshcv = (_array(rainshv) / dt).astype(np.float32)
    prcpsnow = (_array(snow) / dt).astype(np.float32)
    prcpgrpl = (_array(graupel) / dt).astype(np.float32)
    prcphail = (_array(hail) / dt).astype(np.float32)
    prcpothr = np.maximum(
        F(0.0), prcp - prcpconv - prcpnonc - prcpshcv).astype(np.float32)
    prcpnonc = (prcpnonc + prcpothr).astype(np.float32)
    prcpsnow = (prcpsnow + _array(sr) * prcpothr).astype(np.float32)
    return {
        "prcpconv": prcpconv,
        "prcpnonc": prcpnonc,
        "prcpshcv": prcpshcv,
        "prcpsnow": prcpsnow,
        "prcpgrpl": prcpgrpl,
        "prcphail": prcphail,
    }


def ruc_fractional_pre(*, blended_albedo, blended_emiss, xice):
    """``module_surface_driver.F:3461-3473`` optics deblending."""
    xice = _array(xice)
    return {
        "albedo": ((_array(blended_albedo)
                    - (F(1.0) - xice) * F(0.08)) / xice).astype(np.float32),
        "emiss": ((_array(blended_emiss)
                   - (F(1.0) - xice) * F(0.98)) / xice).astype(np.float32),
    }


def ruc_fractional_post(*, ice, sea, xice):
    """``module_surface_driver.F:3530-3572`` common reblend expression."""
    xice = _array(xice)
    return (_array(ice) * xice
            + (F(1.0) - xice) * _array(sea)).astype(np.float32)

