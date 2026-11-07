"""Goddard GCE three-ice microphysics (WRF ``mp_physics=97``): constant table.

WRF v4.6.1 ``phys/module_mp_gsfcgce.F`` recomputes its constants on every
call: ``consat_s`` fills the module's private coefficient block, and
``fall_flux`` forms its fall-speed prefactors and four gamma values.  They
depend only on ``gsfcgce_hail`` (the dense-ice class), so this module
evaluates them once per switch value, in float32 and in WRF's operation
order, and hands the column kernel (``kernels/gsfcgce.cu``) one packed
float32 table.  CuPy-free, so the memory preflight and the tests can price
and inspect the table on a host with no device.

What "WRF's value" means here is the value the compiled ``wrf.exe`` holds,
which is not always what the source text says at run time.  The 0 ULP
referee (gfortran 13.3.0 ``-O2 -fno-fast-math -ffp-contract=off``) folds at
compile time every constant expression it can see, with correct rounding,
and calls the C library only for what it cannot fold.  The disassembly of
the referee object shows exactly which is which (tools/
gsfcgce_wrf461_oracle/README.md lists it), and this module follows it:

* folded powers -- ``(cpi*roqr*tnw)**0.25`` and its snow/dense-ice twins,
  ``zrc**bw`` inside ``vrc`` and its twins, ``0.01**(1-constb)`` inside
  ``consta``/``constc`` -- are evaluated correctly rounded
  (:func:`_folded_pow`, a decimal evaluation at 80 digits), as gcc's MPFR
  folding does;
* run-time powers -- the 31-entry ``rn13``/``rn12b``/``rn25a`` loop over
  the ``aa1``/``aa2`` DATA arrays -- go through WOOF's own float32
  ``powf`` (:mod:`woof.core.noahmp_libm`), the routine the kernel calls;
* ``gammagce`` (consat_s) runs at run time: a binary64 Lanczos sum whose
  logarithm is the host's binary64 ``log`` and whose final exponential is
  WOOF's float32 ``expf`` -- the implicit-REAL ``gammln`` makes it so;
* ``ggamma`` (fall_flux) runs at run time with its integer powers through
  libgcc's ``__powisf2`` repeated squaring, reproduced in
  :func:`_powi_runtime`;
* everything else is IEEE float32 arithmetic, identical folded or not.

The table is checked against the oracle in two ways: the folded immediates
the referee object stores are compared word for word
(``tests/test_gsfcgce_constants.py``), and the column oracle compares every
state array the table feeds.
"""

from __future__ import annotations

import decimal
import math
from functools import lru_cache

import numpy as np

F32 = np.float32

#: The order of the packed table.  ``kernels/gsfcgce.cu``'s ``enum`` names
#: the same slots in the same order (the test reads it back), so the kernel
#: indexes the table without a per-slot argument.
SCALAR_SLOTS = (
    # fall_flux
    "ff_pi", "ff_consta", "ff_constc", "ff_gambp4", "ff_gamdp4",
    "ff_gam4pt5", "ff_gam4bbar",
    # consat_s: thermodynamics and saturation
    "al", "cp", "c358", "c610", "c149", "c879", "c172", "c409", "c76",
    "c218", "c580", "c141", "alv", "alf", "als", "t0", "t00", "avc", "afc",
    "asc",
    # consat_s: size distributions and fall speeds
    "ag", "bg", "as_", "bs", "aw", "bw", "bgh", "bgq", "bsh", "bsq", "bwh",
    "bwq", "tnw", "tns", "tng", "roqs", "roqg", "roqr", "zrc", "zgc", "zsc",
    "vrc", "vgc", "vsc",
    # consat_s: process coefficients
    "rn1", "bnd1", "rn2", "bnd2", "rn3", "rn4", "rn5", "rn6", "rn7", "rn8",
    "rn9", "rn10", "rn101", "rn10a", "rn11", "rn11a", "rn12", "rn14",
    "rn15", "rn15a", "rn16", "rn17", "rn17a", "rn17b", "rn17c", "rn18",
    "rn18a", "rn19", "rn19a", "rn19b", "rn20", "rn20a", "rn20b", "bnd3",
    "rn21", "rn22", "rn23", "rn23a", "rn23b", "rn25", "rn30a", "rn30b",
    "rn30c", "rn31", "beta", "rn32", "rn10b", "rn10c", "rnn191", "rnn192",
    "rn30", "rnn30a", "rn33", "rn331", "rn332",
)
#: The four 31-entry temperature tables (index 1..31 = -1..-31 C), then the
#: two Koenig (1971) DATA arrays they are built from.
ARRAY_SLOTS = ("rn12a", "rn12b", "rn13", "rn25a", "aa1", "aa2")
ARRAY_LEN = 31
TABLE_LEN = len(SCALAR_SLOTS) + ARRAY_LEN * len(ARRAY_SLOTS)

#: module_mp_gsfcgce.F:53-66, the Koenig (1971) Bergeron coefficients.
AA1 = (.7939e-7, .7841e-6, .3369e-5, .4336e-5, .5285e-5,
       .3728e-5, .1852e-5, .2991e-6, .4248e-6, .7434e-6,
       .1812e-5, .4394e-5, .9145e-5, .1725e-4, .3348e-4,
       .1725e-4, .9175e-5, .4412e-5, .2252e-5, .9115e-6,
       .4876e-6, .3473e-6, .4758e-6, .6306e-6, .8573e-6,
       .7868e-6, .7192e-6, .6513e-6, .5956e-6, .5333e-6,
       .4834e-6)
AA2 = (.4006, .4831, .5320, .5307, .5319,
       .5249, .4888, .3894, .4047, .4318,
       .4771, .5183, .5463, .5651, .5813,
       .5655, .5478, .5203, .4906, .4447,
       .4126, .3960, .4149, .4320, .4506,
       .4483, .4460, .4433, .4413, .4382,
       .4361)

#: module_mp_gsfcgce.F:73-78: the PSD intercepts and dense-ice densities the
#: sedimentation and the reflectivity share.
XNOR, XNOS, XNOH, XNOG = F32(8.0e6), F32(1.6e7), F32(2.0e5), F32(4.0e6)
RHOHAIL, RHOGRAUL = F32(917.0), F32(400.0)

#: The two native undefined reads this port repairs, each a
#: configuration WRF itself offers (``gsfcgce_2ice``).  The oracle's FIX arm
#: applies exactly these to WRF's source (tools/gsfcgce_wrf461_oracle/
#: make_fixed_source.py); every other configuration and field is graded
#: against the unmodified wrf.exe objects.
DECLARED_DIVERGENCES = (
    ("gsfcgce_2ice=3 rain evaporation",
     "the warm-rain branch of saticel_s evaporates rain with scv, tca and "
     "dwv (module_mp_gsfcgce.F:1859-1860), which only the ice branch "
     "computes (:1912-1915); in wrf.exe they are uninitialised automatic "
     "arrays and signaling-NaN initialization traps; WOOF computes them in the "
     "warm branch with the ice branch's four statements"),
    ("gsfcgce_2ice=1 reflectivity",
     "refl10cm_gsfc is handed a graupel column the ice+snow branch never "
     "fills (module_mp_gsfcgce.F:2881-2884), so REFL_10CM reads "
     "uninitialised memory; WOOF zeroes it, as the warm branch does"),
)


def _f(x) -> np.float32:
    return F32(x)


def _round_decimal_to_f32(value: decimal.Decimal) -> np.float32:
    """Round an exact decimal to the nearest binary32 (ties to even)."""
    guess = F32(float(value))
    best, best_err = None, None
    for cand in (np.nextafter(guess, F32(-np.inf)), guess,
                 np.nextafter(guess, F32(np.inf))):
        err = abs(decimal.Decimal(float(cand)) - value)
        if best is None or err < best_err or (
                err == best_err and int(cand.view(np.uint32)) % 2 == 0):
            best, best_err = cand, err
    return F32(best)


def _folded_pow(x: np.float32, y: np.float32) -> np.float32:
    """``x**y`` for constant binary32 operands, correctly rounded.

    The value gcc's compile-time folding (MPFR, round to nearest) gives a
    REAL power whose operands it can prove constant.
    """
    ctx = decimal.Context(prec=80)
    exact = ctx.power(decimal.Decimal(float(x)), decimal.Decimal(float(y)))
    return _round_decimal_to_f32(exact)


def _folded_sqrt(x: np.float32) -> np.float32:
    return F32(np.sqrt(F32(x)))


def _powi_runtime(x: np.float32, m: int) -> np.float32:
    """libgcc's ``__powisf2``: float32 repeated squaring, as ggamma calls it."""
    n = -m if m < 0 else m
    y = x if n % 2 else F32(1.0)
    while True:
        n >>= 1
        if not n:
            break
        x = F32(x * x)
        if n % 2:
            y = F32(y * x)
    return F32(F32(1.0) / y) if m < 0 else y


def ggamma(x: np.float32) -> np.float32:
    """module_mp_gsfcgce.F:848-875 ``ggamma`` (fall_flux's gamma)."""
    b = [F32(v) for v in (-.577191652, .988205891, -.897056937, .918206857,
                          -.756704078, .482199394, -.193527818, .035868343)]
    pf = F32(1.0)
    temp = F32(x)
    for _ in range(200):
        if temp <= F32(2.0):
            break
        temp = F32(temp - F32(1.0))
        pf = F32(pf * temp)
    else:
        raise ValueError(f"ggamma argument {x} too large")
    g1to2 = F32(1.0)
    temp = F32(temp - F32(1.0))
    for k1 in range(1, 9):
        g1to2 = F32(g1to2 + F32(b[k1 - 1] * _powi_runtime(temp, k1)))
    return F32(pf * g1to2)


def gammagce(xx: np.float32) -> np.float32:
    """module_mp_gsfcgce.F:2918-2938 ``gammagce`` (consat_s's gamma).

    ``cof``/``stp`` are REAL*8 initialised from default-REAL literals, so
    they hold the binary32 values widened; ``gammln`` is undeclared, hence
    an implicit REAL, so the binary64 sum is rounded to binary32 before the
    single-precision EXP.
    """
    from woof.core.noahmp_libm import expf

    cof = [float(F32(v)) for v in (76.18009173, -86.50532033, 24.01409822,
                                   -1.231739516, .120858003e-2, -.536382e-5)]
    stp = float(F32(2.50662827465))
    x = float(F32(xx)) - 1.0
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.0
    for c in cof:
        x = x + 1.0
        ser = ser + c / x
    gammln = F32(tmp + math.log(stp * ser))
    return F32(expf(gammln))


@lru_cache(maxsize=2)
def constants(ihail: int) -> dict[str, object]:
    """The coefficient block for ``gsfcgce_hail = ihail`` (0 graupel, 1 hail)."""
    from woof.core.noahmp_libm import powf

    if int(ihail) not in (0, 1):
        raise ValueError(f"gsfcgce_hail must be 0 or 1, got {ihail}")
    f = _f
    k = {}
    # ---- fall_flux (module_mp_gsfcgce.F:449-459)
    constb, constd, bbar = f(0.8), f(0.11), f(0.37)
    k["ff_pi"] = F32(math.pi)                       # acos(-1.), folded
    k["ff_consta"] = F32(f(2115.0) * _folded_pow(f(0.01), F32(f(1) - constb)))
    k["ff_constc"] = F32(f(78.63) * _folded_pow(f(0.01), F32(f(1) - constd)))
    k["ff_gambp4"] = ggamma(F32(constb + f(4.0)))
    k["ff_gamdp4"] = ggamma(F32(constd + f(4.0)))
    k["ff_gam4pt5"] = ggamma(f(4.5))
    k["ff_gam4bbar"] = ggamma(F32(f(4.0) + bbar))
    # ---- consat_s (module_mp_gsfcgce.F:1013-1233)
    k["al"] = f(2.5e10)
    k["cp"] = cp = f(1.004e7)
    rd1, rd2 = f(1.e-3), f(2.2)
    cpi = F32(f(4.0) * F32(math.atan(1.0)))
    cpi2 = F32(cpi * cpi)
    grvt = f(980.)
    cd1 = f(6.e-1)
    cd2 = F32(F32(f(4.) * grvt) / F32(f(3.) * cd1))
    tca, dwv, dva, amw, ars = f(2.43e3), f(.226), f(1.718e-4), f(18.016), f(8.314e7)
    k["t0"], k["t00"] = f(273.16), f(238.16)
    alv = k["alv"] = f(2.5e10)
    alf = k["alf"] = f(3.336e9)
    als = k["als"] = f(2.8336e10)
    k["avc"] = F32(alv / cp)
    k["afc"] = F32(alf / cp)
    k["asc"] = F32(als / cp)
    rw, cw, ci = f(4.615e6), f(4.187e7), f(2.093e7)
    k["c76"], k["c358"], k["c172"] = f(7.66), f(35.86), f(17.26939)
    k["c409"], k["c218"], k["c580"] = f(4098.026), f(21.87456), f(5807.695)
    k["c610"], k["c149"], k["c879"] = f(6.1078e3), f(1.496286e-5), f(8.794142)
    k["c141"] = f(1.4144354e7)
    if int(ihail) == 1:
        roqg = f(.9)
        ag = _folded_sqrt(F32(cd2 * roqg))
        bg = f(.5)
        tng = f(.002)
    else:
        roqg = f(.4)
        ag = f(351.2)
        bg = f(.37)
        tng = f(.04)
    tns, roqs, as_, bs = f(.16), f(.1), f(78.63), f(.11)
    aw, bw, roqr, tnw = f(2115.), f(.8), f(1.), f(.08)
    k.update(ag=ag, bg=bg, as_=as_, bs=bs, aw=aw, bw=bw, tnw=tnw, tns=tns,
             tng=tng, roqs=roqs, roqg=roqg, roqr=roqr)
    k["bgh"], k["bsh"], k["bwh"] = F32(f(.5) * bg), F32(f(.5) * bs), F32(f(.5) * bw)
    k["bgq"], k["bsq"], k["bwq"] = F32(f(.25) * bg), F32(f(.25) * bs), F32(f(.25) * bw)
    ga3b = gammagce(F32(f(3.) + bw))
    ga4b = gammagce(F32(f(4.) + bw))
    ga6b = gammagce(F32(f(6.) + bw))
    ga5bh = gammagce(F32(F32(f(5.) + bw) / f(2.)))
    ga3g = gammagce(F32(f(3.) + bg))
    ga4g = gammagce(F32(f(4.) + bg))
    ga5gh = gammagce(F32(F32(f(5.) + bg) / f(2.)))
    ga3d = gammagce(F32(f(3.) + bs))
    ga4d = gammagce(F32(f(4.) + bs))
    ga5dh = gammagce(F32(F32(f(5.) + bs) / f(2.)))
    ac1, ac2, ac3, cc1 = aw, ag, as_, as_
    zrc = _folded_pow(F32(F32(cpi * roqr) * tnw), f(0.25))
    zsc = _folded_pow(F32(F32(cpi * roqs) * tns), f(0.25))
    zgc = _folded_pow(F32(F32(cpi * roqg) * tng), f(0.25))
    k.update(zrc=zrc, zsc=zsc, zgc=zgc)
    k["vrc"] = F32(F32(aw * ga4b) / F32(f(6.) * _folded_pow(zrc, bw)))
    k["vsc"] = F32(F32(as_ * ga4d) / F32(f(6.) * _folded_pow(zsc, bs)))
    k["vgc"] = F32(F32(ag * ga4g) / F32(f(6.) * _folded_pow(zgc, bg)))

    def chain(*xs):
        v = F32(xs[0])
        for x in xs[1:]:
            v = F32(v * F32(x))
        return v

    k["rn1"], k["bnd1"], k["rn2"], k["bnd2"] = f(9.4e-15), f(6.e-4), f(1.e-3), f(2.0e-3)
    k["rn3"] = chain(.25, cpi, tns, cc1, ga3d)
    esw = f(1.)
    k["rn4"] = chain(.25, cpi, esw, tns, cc1, ga3d)
    eri = f(.1)
    k["rn5"] = chain(.25, cpi, eri, tnw, ac1, ga3b)
    ami = F32(f(1.) / F32(f(24.) * f(6.e-9)))
    k["rn6"] = chain(cpi2, eri, tnw, ac1, roqr, ga6b, ami)
    k["rn7"] = chain(cpi2, f(.5), tnw, tns, roqs)
    k["rn8"] = chain(cpi2, f(1.), tnw, tns, roqr)
    k["rn9"] = chain(cpi2, tns, tng, roqs)
    k["rn10"] = chain(2., cpi, tns)
    k["rn101"] = F32(F32(f(.31) * ga5dh) * _folded_sqrt(cc1))
    k["rn10a"] = F32(F32(als * als) / rw)
    k["rn10b"] = F32(alv / tca)
    k["rn10c"] = F32(ars / F32(dwv * amw))
    k["rn11"] = F32(chain(2., cpi, tns) / alf)
    k["rn11a"] = F32(cw / alf)
    ami50, ami40 = f(3.84e-6), f(3.08e-8)
    eiw, ui50 = f(1.), f(100.)
    ri50 = F32(f(2.) * f(5.e-3))
    cmn = f(1.05e-15)
    k["rn12"] = chain(cpi, eiw, ui50, F32(ri50 * ri50))
    rn13, rn12a, rn12b, rn25a = [], [], [], []
    for a1, a2 in zip(AA1, AA2):
        a1, a2 = f(a1), f(a2)
        y1 = F32(f(1.) - a2)
        v = F32(F32(a1 * y1) / F32(powf(ami50, y1) - powf(ami40, y1)))
        rn13.append(v)
        rn12a.append(F32(v / ami50))
        rn12b.append(F32(a1 * powf(ami50, a2)))
        rn25a.append(F32(a1 * powf(cmn, a2)))
    egw = f(1.)
    k["rn14"] = chain(.25, cpi, egw, tng, ga3g, ag)
    k["rn15"] = chain(.25, cpi, f(.1), tng, ga3g, ag)
    k["rn15a"] = chain(.25, cpi, f(1.), tng, ga3g, ag)
    k["rn16"] = chain(cpi2, f(1.), tng, tnw, roqr)
    k["rn17"] = chain(2., cpi, tng)
    k["rn17a"] = F32(F32(f(.31) * ga5gh) * _folded_sqrt(ag))
    k["rn17b"] = F32(cw - ci)
    k["rn17c"] = cw
    apri = f(.66)
    bpri = F32(f(0.5) * f(1.e-4))
    k["rn18"] = chain(20., cpi2, bpri, tnw, roqr)
    k["rn18a"] = apri
    k["rn19"] = F32(chain(2., cpi, tng) / alf)
    k["rn19a"] = F32(F32(f(.31) * ga5gh) * _folded_sqrt(ag))
    k["rn19b"] = F32(cw / alf)
    k["rnn191"] = f(.78)
    k["rnn192"] = F32(F32(f(.31) * ga5gh) * _folded_sqrt(F32(ac2 / dva)))
    k["rn20"] = chain(2., cpi, tng)
    k["rn20a"] = F32(F32(als * als) / rw)
    k["rn20b"] = F32(F32(f(.31) * ga5gh) * _folded_sqrt(ag))
    k["bnd3"] = f(2.e-3)
    k["rn21"] = F32(F32(f(1.e3) * f(1.569e-12)) / f(0.15))
    k["rn22"] = chain(.25, cpi, f(1.), ac1, tnw, ga3b)
    k["rn23"] = chain(2., cpi, tnw)
    k["rn23a"] = F32(F32(f(.31) * ga5bh) * _folded_sqrt(ac1))
    k["rn23b"] = F32(F32(alv * alv) / rw)
    # itaobraun = 1 (module_mp_gsfcgce.F:230, :1209-1211).
    cn0, beta = f(1.e-6), f(-.46)
    k["beta"] = beta
    k["rn25"] = cn0
    k["rn30a"] = F32(F32(F32(alv * als) * amw) / F32(tca * ars))
    k["rn30b"] = F32(alv / tca)
    k["rn30c"] = F32(ars / F32(dwv * amw))
    k["rn31"] = f(1.e-17)
    k["rn32"] = F32(f(4.) * f(51.545e-4))
    k["rn30"] = chain(2., cpi, tng)
    k["rnn30a"] = F32(F32(F32(alv * alv) * amw) / F32(tca * ars))
    k["rn33"] = F32(f(4.) * tns)
    k["rn331"] = f(.65)
    k["rn332"] = F32(F32(f(.44) * _folded_sqrt(F32(ac3 / dva))) * ga5dh)
    del rd1, rd2
    arrays = {"rn12a": rn12a, "rn12b": rn12b, "rn13": rn13, "rn25a": rn25a,
              "aa1": [f(v) for v in AA1], "aa2": [f(v) for v in AA2]}
    missing = [s for s in SCALAR_SLOTS if s not in k]
    if missing:
        raise AssertionError(f"gsfcgce table slots unset: {missing}")
    return {"scalars": {s: F32(k[s]) for s in SCALAR_SLOTS},
            "arrays": {a: np.asarray(arrays[a], dtype=F32)
                       for a in ARRAY_SLOTS}}


@lru_cache(maxsize=2)
def table(gsfcgce_hail: int) -> np.ndarray:
    """The packed float32 table ``kernels/gsfcgce.cu`` reads."""
    c = constants(gsfcgce_hail)
    out = np.concatenate(
        [np.array([c["scalars"][s] for s in SCALAR_SLOTS], dtype=F32)]
        + [c["arrays"][a] for a in ARRAY_SLOTS]).astype(F32)
    if out.shape != (TABLE_LEN,):
        raise AssertionError(out.shape)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=2)
def radar_parameters(gsfcgce_hail: int) -> dict[str, float]:
    """The m(D) inputs gsfcgce hands ``radar_init`` (module_mp_gsfcgce.F:
    204-216) and the gamma moments radar_init derives from them
    (module_mp_radar.F:152-189), as the float32 values WRF holds."""
    from woof.core.noahmp_libm import expf

    def gammln(xx: np.float32) -> np.float32:     # module_mp_radar.F:593-613
        cof = (76.18009172947146, -86.50532032941677, 24.01409824083091,
               -1.231739572450155, .1208650973866179e-2, -.5395239384953e-5)
        x = float(F32(xx))
        y = x
        tmp = x + 5.5
        tmp = (x + 0.5) * math.log(tmp) - tmp
        ser = 1.000000000190015
        for c in cof:
            y = y + 1.0
            ser = ser + c / y
        return F32(tmp + math.log(2.5066282746310005 * ser / x))

    def wgamma(y):
        return F32(expf(gammln(y)))

    pi_trunc = F32(3.14159)
    xam_r = F32(F32(pi_trunc * F32(1000.)) / F32(6.))
    xam_s = F32(F32(pi_trunc * F32(100.)) / F32(6.))
    ihail = int(gsfcgce_hail)
    rho_g = RHOHAIL if ihail == 1 else RHOGRAUL
    xam_g = F32(F32(pi_trunc * rho_g) / F32(6.))
    xbm = F32(3.0)
    xce = (F32(1. + xbm), F32(1.), F32(F32(1. + xbm) + F32(0.)),
           F32(F32(1. + F32(2. * xbm)) + F32(0.)))
    xcg = tuple(wgamma(v) for v in xce)
    xoams = F32(F32(1.) / xam_s)
    xoamg = F32(F32(1.) / xam_g)
    xobm = F32(F32(1.) / xbm)
    return {
        "xam_r": float(xam_r), "xam_s": float(xam_s), "xam_g": float(xam_g),
        "xcre1": float(xce[0]), "xcre4": float(xce[3]),
        "xcg3": float(xcg[2]), "xcg4": float(xcg[3]),
        "xocms": float(powf_f32(xoams, xobm)),
        "xocmg": float(powf_f32(xoamg, xobm)),
        "n0g": float(XNOH if ihail == 1 else XNOG),
        # module_mp_radar.F:76: PI5 is a product of default-REAL literals.
        "pi5": float(F32(F32(F32(F32(pi_trunc * pi_trunc) * pi_trunc)
                             * pi_trunc) * pi_trunc)),
    }


def refl_species(gsfcgce_2ice: int) -> tuple[int, int]:
    """Which of snow and graupel the reflectivity column carries
    (module_mp_gsfcgce.F:2876-2895): 3-ice both, ice+snow snow only (WRF
    leaves graupel uninitialised there; WOOF passes zero, a declared
    divergence), ice+graupel graupel only, warm rain neither."""
    return {0: (1, 1), 1: (1, 0), 2: (0, 1), 3: (0, 0)}[int(gsfcgce_2ice)]


@lru_cache(maxsize=1)
def native_radar_materials():
    """WRF radar_init materials, preserving default-REAL and CMPLX kinds.

    These initializers run once. The shared Morrison initializer uses wider
    constants and cannot supply a bitwise native GCE diagnostic.
    Authority: module_mp_radar.F:236-306 (WRF v4.7.1).
    """
    import cmath
    from dataclasses import replace
    from woof.core.refl import radar_init
    f = lambda x: float(F32(x))
    lam, t = f(.10), 0.0
    tk = t + f(273.16)
    epsinf = 5.27137 + .02164740*t - .00131198*t*t
    epss = 78.54*(1.0 - .004579*(t-25.0)
                   + .00001190*(t-25.0)*(t-25.0)
                   - .00000002800*(t-25.0)*(t-25.0)*(t-25.0))
    alpha = -16.8129/tk + .0609265
    lambdas = .00033836*math.exp(2513.98/tk)*f(.01)
    a = (lambdas/lam)**(1.0-alpha)
    angle = alpha*math.pi*.5
    den = 1.0 + 2.0*a*math.sin(angle) + (lambdas/lam)**(2.0-2.0*alpha)
    epsr = epsinf + ((epss-epsinf)*(a*math.sin(angle)+1.0))/den
    epsi = ((epss-epsinf)*(a*math.cos(angle)+0.0))/den + lam*f(1.25664)/f(1.88496)
    # SQRT(CMPLX(...)) is complex32, widened only on return.
    water32 = complex(f(epsr), f(-epsi))
    root = cmath.sqrt(water32)
    mw = complex(f(root.real), f(root.imag))
    freq = 2.99e8/lam*1.e-9
    db = math.exp(f(-10.02)+f(.0364)*(tk-f(273.16)))
    bm = (f(.0207)/tk)*(math.exp(335.0/tk)/((math.exp(335.0/tk)-1)**2)) + 1.16e-11*freq*freq
    theta = f(300.0)/tk-f(1.0)
    alfa = (.00504+.0062*theta)*math.exp(-22.1*theta)
    # The ice function assigns to complex64 before its final SQRT.
    ice = complex(f(3.1884)+f(.00091)*(tk-f(273.16)),
                  f(alfa/freq+(bm+db)*freq))
    mi = complex(np.sqrt(np.conj(ice)))
    kw = abs((mw*mw-1.0)/(mw*mw+2.0))**2
    return replace(radar_init(),m_w_0=mw,m_i_0=mi,k_w=kw,
                   lamda4=lam*lam*lam*lam)


def powf_f32(x, y) -> np.float32:
    from woof.core.noahmp_libm import powf
    return F32(powf(F32(x), F32(y)))


__all__ = ["ARRAY_LEN", "ARRAY_SLOTS", "DECLARED_DIVERGENCES",
           "SCALAR_SLOTS", "TABLE_LEN", "constants", "gammagce", "ggamma",
           "radar_parameters", "refl_species", "table"]
