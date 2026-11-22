"""Float32 CPU reference for WRF v4.6.1 ``CUP_gf`` -- the Grell-Freitas deep
cloud model, ``module_cu_gf_deep.F:39-1868`` plus the 15 procedures it calls.

Stage 1 of the port (``gf_ref.gf_driver_prep``) is GFDRV's column preparation.
This module is stage 2 onward: everything from ``cup_env`` to
``cup_output_ens_3d``, graded stage by stage against
``woof/data/gf/oracle/gf-deep-levels.csv`` and ``gf-deep-surface.csv``, which
``tools/gf_wrf461_oracle/run_gf_stages.F90`` writes out of a replication of
``CUP_gf`` that reproduces the module's own ``cup_gf`` bitwise on all 216
columns.

Only the arm WRF can reach is implemented.  ``module_cu_gf_wrfdrv.F:68-74``
fixes ``imid_gf = 0``, ``ichoice_s = 0``, ``dicycle = 1`` and ``ideep = 1`` as
``parameter``s, and ``module_cu_gf_deep.F`` fixes ``irainevap = 0``,
``autoconv = 1``, ``aeroevap = 1``, ``use_excess = 1`` and ``iversion = 1``
the same way.  The mid-level arm, the Berry autoconversion arm, the aerosol
precipitation-efficiency arm, the SAS rain-evaporation block and the
"real cloud-work function" version of the diurnal closure are therefore dead
code in WRF's GF as shipped, and are named here rather than implemented.
``gfinit`` and ``DERIV3`` are dead too -- ``DERIV3`` completely: nothing in
any of the three GF modules calls it.

Precision
---------
Everything is float32, one rounding per operation, in WRF's association
order.  The cloud model does NOT share the driver's GFS constants: it
declares its own ``g = 9.81``, ``cp = 1004.``, ``xlv = 2.5e6``,
``r_v = 461.`` as plain real(4) parameters, and ``cup_env``/``cup_env_clev``
additionally spell 9.81/1004./2.5e6 as bare literals rather than using the
parameters.  Both constant sets are live in one call; nothing here
harmonises them, and ``tests/test_gf_wrf461_parity.py`` fails if a tidy-up
ever does.

libm, per ``gf-pow-probe.txt`` on the oracle's own toolchain (gfortran
13.3.0, glibc 2.39, -O0):

* ``x**2`` and ``x**3`` with *integer* literal exponents fold bitwise to
  multiply chains on all 24 probes, so ``VSHEAR**2``, ``VSHEAR**3``,
  ``US**2`` and ``(1.-frh)**2`` are spelled as multiplies.  ``kratio**(alpha
  -1.0)`` and ``(1.-kratio)**(beta-1.0)`` are *not*: those exponents are
  runtime values and stay ``powf``.
* ``x**.3333`` is a literal power, not a cube root: ``.3333`` is
  ``0x3EAAA64C`` against ``1./3.``'s ``0x3EAAAAAB``, and they disagree on 10
  of 12 probed x.
* ``expf``, ``logf`` and ``powf`` are modelled as **correctly rounded** --
  float64 with a :mod:`decimal` refinement near the float32 rounding
  boundary, not the float64-round-once surrogate the Shin-Hong port used.
  The surrogate double-rounds, and an 8640-lane fixture finds it: it misses
  ``powf(0x3F0D923B, 0x3E999998)`` by 1 ULP where the 26-form Shin-Hong probe
  never did.

Two things this reference does NOT reproduce bitwise.  Both are measured,
both are glibc rather than WRF, and neither is absorbed into a tolerance.

**``powf`` is not correctly rounded either.**  glibc computes it as a
double-precision ``exp2(y*log2(x))`` with about 0.82 ULP of worst-case error,
so on an argument whose true value sits a whisker from a rounding boundary it
can land on the far side.  The fixture contains exactly one such argument:
``powf(0x3F0D923B, 0x3E999998)``, whose true value to 80 digits is
0.83718320727036911868... against a midpoint of 0.83718320727348327636... --
correctly rounded ``0x3F5651A3``, and it wins by 5.2e-5 of a ULP.  glibc
returns ``0x3F5651A4`` (``ppowhard`` in ``gf-pow-probe.txt``, evaluated
through variables so gfortran cannot fold it).  This port computes the
correctly rounded value, which is the *defined* answer, and the consequence
is bounded: 10 lanes of ``zu`` out of 8640, all on ``ierr == 6`` columns WRF
rejects before they produce a tendency.  Every output field is bitwise.

**``tgammaf`` is not modelled at all, and that one is expensive.**
``get_zu_zd_pdf_fim`` normalises its beta-function mass-flux shape with
``fzu = gamma(alpha+beta)/(gamma(alpha)*gamma(beta))``.  glibc's float32
``tgammaf`` is 1-2 ULP off correctly rounded on 31 of the 51 arguments the
scheme reaches -- ``tgammaf(4.0)`` is ``0x40C00001``, a full ULP above 6 --
and three candidate models were measured against it and all three failed:
``float32(tgamma_float64(x))`` misses 31, ``expf(lgammaf(x))`` misses 39, and
the exp-lgamma-times-product-recurrence shape glibc's own ``e_gammaf_r.c``
uses misses 32 (``plgamma`` in ``gf-pow-probe.txt`` prints the decomposition).
Nothing short of glibc's own algorithm closes it, and WOOF-ArWen does not
reproduce it.

That gap is NOT a rounding footnote, because **GF amplifies it by five orders
of magnitude**.  Perturb ``fzu`` by one ULP and re-run a converged column and
``xmb`` moves by up to **7.3 per cent** (median 1.9).  ``cup_forcing_ens_3d``
builds every stability closure as ``-xff/xk`` with ``xk = (xaa0-aa1)/mbdt``,
a difference of two cloud work functions that agree to several digits; a
last-bit change in the mass-flux shape walks into the vertical integral
``aa1`` (450 ULP) and the cancellation does the rest.

So :func:`get_zu_zd_pdf_fim` takes an ``fzu`` override.  The parity gate runs
the whole chain twice -- once with :func:`tgammaf` and once with the oracle's
captured ``fzu`` -- and the second run is bitwise on every field, which is
what attributes the residual to ``tgammaf`` alone.  A caller who needs WRF's
answer must supply ``fzu``; a caller who does not gets a per-cent-level
answer and this docstring.

Indexing
--------
Arrays are 1-based to match the Fortran: a per-level array has length
``nz + 1`` and slot 0 is unused.  This is not decoration -- ``cup_kbcon``,
``cup_minimi``, ``get_cloud_bc`` and ``get_zu_zd_pdf_fim`` all do index
arithmetic that is a rich source of off-by-one in a 0-based transcription.

One deliberate divergence, per the project's standing rule that where WRF is
undefined the port implements the defined behaviour and records the gap:
:func:`get_inversion_layers` reads ``t_cup(kend+8)``, and both live call
sites pass ``kend = kstabi``, which ``cup_minimi`` bounds only by ``ktf-1``.
At ``kte = 40`` any column with ``kstabi > 32`` reads off the end of the
array.  This port clamps the loop instead.  The oracle capture clamps the
same way and counts the clamps; the count is 0 on the present fixture, so
the divergence is documented and not exercised.
"""

from __future__ import annotations

import math

import numpy as np

from woof.verify.gf_ref import DEEP_CP, DEEP_G, DEEP_RV, DEEP_XLV

__all__ = [
    "F",
    "_powf",
    "_expf",
    "_logf",
    "_tgammaf",
    "satvap",
    "cup_env",
    "cup_env_clev",
    "get_cloud_bc",
    "cup_kbcon",
    "cup_maximi",
    "cup_minimi",
    "rates_up_pdf_deep",
    "rates_up_pdf_shallow",
    "get_zu_zd_pdf_fim",
    "get_lateral_massflux",
    "cup_up_moisture",
    "cup_dd_moisture",
    "cup_up_aa0",
    "cup_up_aa1bl",
    "cup_dd_edt",
    "cup_forcing_ens_3d",
    "cup_output_ens_3d",
    "neg_check",
    "get_inversion_layers",
    "MAXENS3",
]

F = np.float32

# --- module_cu_gf_deep.F:6-33, the cloud model's own parameters -------------
G = DEEP_G
CP = DEEP_CP
XLV = DEEP_XLV
R_V = DEEP_RV
TCRIT = F(258.0)
C1 = F(0.001)
FRH_THRESH = F(0.9)
RH_THRESH = F(0.97)
BETAJB = F(1.5)
FLUXTUNE = F(1.5)
PGCD = F(1.0)
MAXENS3 = 16

_TINY32 = F(np.finfo(np.float32).tiny)

_Z0 = F(0.0)
_Z1 = F(1.0)
_HALF = F(0.5)


# --------------------------------------------------------------------------
# libm surrogates
# --------------------------------------------------------------------------
def _round_correctly(r64, exact):
    """Round a float64 estimate to float32 the way a correctly-rounded libm
    would, refining with :mod:`decimal` only when the estimate sits near a
    float32 rounding boundary.

    The float64-then-float32 double rounding this replaces is not a
    theoretical worry.  It was measured: ``powf(0x3F0D923B, 0x3E999998)``
    -- ``(1-kratio)**(beta-1)`` at level 17 of one column of the GF fixture --
    is ``0x3F5651A4`` out of glibc and ``0x3F5651A3`` out of the double-
    rounding surrogate, and that 1 ULP walks all the way through
    ``get_zu_zd_pdf_fim``'s normalisation into ``zu``.  The Shin-Hong port's
    26-form probe never hit the case; an 8640-lane fixture does.

    The refinement window is 1e-4 float32 ULP.  glibc's float64 ``pow``/
    ``exp``/``log`` are within a few ULP of double, i.e. order 1e-8 float32
    ULP, so the window is four orders of magnitude of margin and fires on
    roughly two lanes in ten thousand.
    """
    r32 = F(r64)
    if not np.isfinite(r32) or r32 == _Z0:
        return r32
    up = np.float64(np.nextafter(r32, F(np.inf)))
    dn = np.float64(np.nextafter(r32, F(-np.inf)))
    c = np.float64(r32)
    ulp = up - c
    if ulp == 0.0:
        return r32
    if min(abs(r64 - (c + up) * 0.5), abs(r64 - (c + dn) * 0.5)) > ulp * 1e-4:
        return r32
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 60
        e = exact()
        best = min(
            (F(dn), r32, F(up)), key=lambda v: abs(e - Decimal(float(v)))
        )
    return F(best)


def _powf(x, y):
    """glibc's ``powf``, correctly rounded.  See :func:`_round_correctly`."""
    from decimal import Decimal

    xf = F(x)
    yf = F(y)
    r64 = np.float64(np.float64(xf) ** np.float64(yf))
    if xf <= _Z0:
        return F(r64)
    return _round_correctly(
        r64, lambda: (Decimal(float(yf)) * Decimal(float(xf)).ln()).exp()
    )


def _expf(x):
    from decimal import Decimal

    xf = F(x)
    r64 = np.float64(np.exp(np.float64(xf)))
    return _round_correctly(r64, lambda: Decimal(float(xf)).exp())


def _logf(x):
    from decimal import Decimal

    xf = F(x)
    r64 = np.float64(np.log(np.float64(xf)))
    if xf <= _Z0:
        return F(r64)
    return _round_correctly(r64, lambda: Decimal(float(xf)).ln())


def _tgammaf(x):
    """The CORRECTLY ROUNDED float32 gamma -- and, since 2.7.0, the shipped
    kernel's answer too.

    It is NOT glibc's ``tgammaf``, and this docstring used to describe it as
    a model of one.  It is not: ``float32(math.gamma(float64(x)))`` is
    correctly rounded on every float32 of [0.25, 36] and glibc 2.39 is not
    on 39.44 per cent of them, worst 6 ULP.  Up to 2 ULP separate the two;
    31 of the 51 arguments in ``gf-pow-probe.txt``'s ``pgamma`` table
    disagree.  This is the only non-bitwise-against-WRF call in the
    reference and it is confined to ``fzu``.

    Through 2.6.5 the CUDA kernel's earlier gamma returned glibc's words and
    so disagreed with THIS function.  That gamma is replaced and
    ``gfk_tgamma`` is now WOOF-ArWen's own correctly rounded gamma: MEASURED, it
    returns the same word as the line below on all 59,768,833 float32 of
    [0.25, 36], so the CPU and CUDA paths agree bitwise where they used to
    differ on 23,575,230 of them.  The divergence from WRF that both now
    carry is written up in ``docs/gf_gamma_known_delta.md``.
    """
    return F(math.gamma(float(F(x))))


# gfortran folds log(10.), log(6.1071) and log(1013.246) in the front end, at
# the type's own precision, so each is a real(4) rounding and the quotients
# are a second one.
_LOG10 = _logf(F(10.0))
_LOG6_OVER_LOG10 = F(_logf(F(6.1071)) / _LOG10)
_LOG1013_OVER_LOG10 = F(_logf(F(1013.246)) / _LOG10)


def satvap(temp2):
    """module_cu_gf_deep.F:3646-3668.  Goff-Gratch, in base-10 powers.

    ``10 ** eilog`` is an *integer* base raised to a real exponent, which
    Fortran converts to ``powf(10.0f, eilog)``; the ``log(x)/log(10.)`` forms
    are a runtime ``logf`` over a folded constant, not ``log10f``.
    """
    temp2 = F(temp2)
    temp = F(temp2 - F(273.155))
    if temp < F(-20.0):
        toot = F(F(273.16) / temp2)
        toto = F(_Z1 / toot)
        e = F(F(-9.09718) * F(toot - _Z1))
        e = F(e - F(F(3.56654) * F(_logf(toot) / _LOG10)))
        e = F(e + F(F(0.876793) * F(_Z1 - toto)))
        e = F(e + _LOG6_OVER_LOG10)
        return _powf(F(10.0), e)
    tsot = F(F(373.16) / temp2)
    ewlog = F(F(-7.90298) * F(tsot - _Z1))
    ewlog = F(ewlog + F(F(5.02808) * F(_logf(tsot) / _LOG10)))
    ewlog2 = F(
        ewlog
        - F(
            F(1.3816e-07)
            * F(_powf(F(10.0), F(F(11.344) * F(_Z1 - F(_Z1 / tsot)))) - _Z1)
        )
    )
    ewlog3 = F(
        ewlog2
        + F(F(0.0081328) * F(_powf(F(10.0), F(F(-3.49149) * F(tsot - _Z1))) - _Z1))
    )
    ewlog4 = F(ewlog3 + _LOG1013_OVER_LOG10)
    return _powf(F(10.0), ewlog4)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _a(nz):
    """A 1-based float32 level array: index 0 exists and is never read."""
    return np.zeros(nz + 1, dtype=F)


def _maxloc(arr, lo, hi):
    """Fortran ``MAXLOC(arr(lo:hi), 1)`` -- the FIRST maximum, 1-based within
    the section.  Returns the absolute index."""
    best = lo
    bv = arr[lo]
    for k in range(lo + 1, hi + 1):
        if arr[k] > bv:
            bv = arr[k]
            best = k
    return best


def _minloc(arr, lo, hi):
    best = lo
    bv = arr[lo]
    for k in range(lo + 1, hi + 1):
        if arr[k] < bv:
            bv = arr[k]
            best = k
    return best


def get_cloud_bc(array, k22, add=None):
    """module_cu_gf_deep.F:3670-3693.  A 3-point mean below ``k22``."""
    local_order = min(k22, 3)
    x = _Z0
    for i in range(1, local_order + 1):
        x = F(x + array[k22 - i + 1])
    x = F(x / F(local_order))
    if add is not None:
        x = F(x + add)
    return x


# --------------------------------------------------------------------------
# cup_env / cup_env_clev
# --------------------------------------------------------------------------
def cup_env(z, t, q, p, nz):
    """module_cu_gf_deep.F:2141-2269 with ``itest = -1``.

    ``itest = -1`` is the only value ``CUP_gf`` ever passes, and it means the
    height stack is NOT recomputed -- ``z`` passes through untouched and the
    ``ALOG(P)`` hydrostatic integration at :2229-2243 is dead.  ``he`` is
    still assigned, because the guard is ``itest .le. 0``.
    """
    qes = _a(nz)
    he = _a(nz)
    hes = _a(nz)
    for k in range(1, nz + 1):
        e = satvap(t[k])
        qes[k] = F(F(F(0.622) * e) / max(F(1.0e-8), F(p[k] - e)))
        if qes[k] <= F(1.0e-16):
            qes[k] = F(1.0e-16)
        if qes[k] < q[k]:
            qes[k] = q[k]
    for k in range(1, nz + 1):
        he[k] = F(
            F(F(F(9.81) * z[k]) + F(F(1004.0) * t[k])) + F(F(2.5e06) * q[k])
        )
        hes[k] = F(
            F(F(F(9.81) * z[k]) + F(F(1004.0) * t[k])) + F(F(2.5e06) * qes[k])
        )
        if he[k] >= hes[k]:
            he[k] = hes[k]
    return qes, he, hes


def cup_env_clev(t, qes, q, he, hes, z, p, psur, z1, nz):
    """module_cu_gf_deep.F:2272-2371.  Half-level (``_cup``) environment.

    Note :2361-2364: ``z_cup(1)`` and ``p_cup(1)`` are each assigned twice,
    and the second assignment -- ``z1`` and ``psur`` -- wins.  The first is
    dead but is a trap for a port that reads only the first.
    """
    qes_cup = _a(nz)
    q_cup = _a(nz)
    he_cup = _a(nz)
    hes_cup = _a(nz)
    z_cup = _a(nz)
    p_cup = _a(nz)
    t_cup = _a(nz)
    gamma_cup = _a(nz)
    xlv_over_cp = F(XLV / CP)
    for k in range(2, nz + 1):
        qes_cup[k] = F(_HALF * F(qes[k - 1] + qes[k]))
        q_cup[k] = F(_HALF * F(q[k - 1] + q[k]))
        hes_cup[k] = F(_HALF * F(hes[k - 1] + hes[k]))
        he_cup[k] = F(_HALF * F(he[k - 1] + he[k]))
        if he_cup[k] > hes_cup[k]:
            he_cup[k] = hes_cup[k]
        z_cup[k] = F(_HALF * F(z[k - 1] + z[k]))
        p_cup[k] = F(_HALF * F(p[k - 1] + p[k]))
        t_cup[k] = F(_HALF * F(t[k - 1] + t[k]))
        gamma_cup[k] = F(
            F(xlv_over_cp * F(XLV / F(F(R_V * t_cup[k]) * t_cup[k]))) * qes_cup[k]
        )
    qes_cup[1] = qes[1]
    q_cup[1] = q[1]
    hes_cup[1] = F(
        F(F(F(9.81) * z1) + F(F(1004.0) * t[1])) + F(F(2.5e6) * qes[1])
    )
    he_cup[1] = F(F(F(F(9.81) * z1) + F(F(1004.0) * t[1])) + F(F(2.5e6) * q[1]))
    z_cup[1] = z1
    p_cup[1] = psur
    t_cup[1] = t[1]
    gamma_cup[1] = F(
        F(xlv_over_cp * F(XLV / F(F(R_V * t_cup[1]) * t_cup[1]))) * qes_cup[1]
    )
    return qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup


# --------------------------------------------------------------------------
# cup_kbcon
# --------------------------------------------------------------------------
def cup_kbcon(
    *, cap_inc, iloop, k22, he_cup, hes_cup, hkb, ierr, kbmax, p_cup, cap_max,
    ztexec, zqexec, z_cup, entr_rate, heo, nz,
):
    """module_cu_gf_deep.F:2722-2858, ``imid = 0``.

    Transcribed from the ``GO TO`` graph rather than from its intent: label 32
    is the loop head, 31 is the "raise kbcon and retry" edge, 27 is every exit.
    ``k22`` and ``hkb`` are both INOUT and both move when the cap test fails.
    """
    kbcon = 1
    if ierr != 0:
        return kbcon, k22, hkb, ierr, None
    hcot = _a(nz)
    start_level = k22
    kbcon = k22 + 1

    def fill_hcot(start, hk):
        hcot[1 : start + 1] = hk
        for k in range(start + 1, min(kbmax + 3, nz) + 1):
            dz = F(z_cup[k] - z_cup[k - 1])
            hcot[k] = F(
                F(
                    F(F(_Z1 - F(F(_HALF * entr_rate) * dz)) * hcot[k - 1])
                    + F(F(entr_rate * dz) * heo[k - 1])
                )
                / F(_Z1 + F(F(_HALF * entr_rate) * dz))
            )

    if iloop == 5:
        # The shallow entry point (module_cu_gf_sh.F:402) is the ONLY caller
        # that passes iloop = 5, and it changes four things: kbcon starts at
        # k22 rather than k22+1, the negative-buoyancy depth is allowed two
        # levels rather than one, `plus` is a flat 150 mb, and a cap_max above
        # 200 mb -- which is what `cap_max = po_cup(kpbl)` gives whenever the
        # boundary layer is deep -- re-references pbcdif to cap_max itself
        # instead of to p_cup(k22).
        kbcon = k22
    fill_hcot(start_level, hkb)
    ierrc = None
    while True:
        hetest = hcot[kbcon]
        if hetest < hes_cup[kbcon]:
            kbcon += 1
            if kbcon > kbmax + 2:
                ierr = 3
                ierrc = "could not find reasonable kbcon in cup_kbcon"
                break
            continue
        if kbcon - k22 == 1:
            break
        if iloop == 5 and (kbcon - k22) <= 2:
            break
        pbcdif = F(-p_cup[kbcon] + p_cup[k22])
        plus = max(F(25.0), F(cap_max - F(F(iloop - 1) * cap_inc)))
        if iloop == 5:
            plus = F(150.0)
            if cap_max > F(200.0):
                pbcdif = F(-p_cup[kbcon] + cap_max)
        if pbcdif <= plus:
            break
        k22 += 1
        kbcon = k22 + 1
        x_add = F(F(XLV * zqexec) + F(CP * ztexec))
        hkb = get_cloud_bc(he_cup, k22, x_add)
        start_level = k22
        fill_hcot(start_level, hkb)
        if iloop == 5:
            kbcon = k22
        if kbcon > kbmax + 2:
            ierr = 3
            ierrc = "could not find reasonable kbcon in cup_kbcon"
            break
    return kbcon, k22, hkb, ierr, ierrc


def cup_maximi(array, ks, ke, ierr):
    """module_cu_gf_deep.F:2861-2914.  ``.GE.`` -- so this is the LAST
    maximum, unlike Fortran's ``MAXLOC``, which is the first."""
    maxx = ks
    if ierr != 0:
        return maxx
    x = array[ks]
    for k in range(ks, ke + 1):
        if array[k] >= x:
            x = array[k]
            maxx = k
    return maxx


def cup_minimi(array, ks, kend, ierr):
    """module_cu_gf_deep.F:2917-2965."""
    kt = ks
    if ierr != 0:
        return kt
    x = array[ks]
    kstop = max(ks + 1, kend)
    for k in range(ks + 1, kstop + 1):
        if array[k] < x:
            x = array[k]
            kt = k
    return kt


# --------------------------------------------------------------------------
# the beta-function mass-flux profiles
# --------------------------------------------------------------------------
def get_zu_zd_pdf_fim(
    *, draft, p, kb, kt, kpbli, csum, zubeg, nz, ktf, fzu_override=None
):
    """module_cu_gf_deep.F:3825-3987, drafts ``UP``, ``SH2`` and ``DOWN``.

    ``SH3`` is dead (nothing passes it) and ``MID`` belongs to the dead
    ``imid = 1`` arm.  ``max_mass``, ``pmin_lev``, ``rand_vmas`` and ``kklev``
    are dead on every branch reached here, and so is ``kb_adj`` on ``DOWN`` --
    it is computed and then the branch indexes ``p(kb)`` directly.

    ``SH2`` is the shallow scheme's shape and is NOT a re-parameterised
    ``UP``.  Three differences, all live: the tunning clamp is 0.8 rather than
    0.9 and takes ``p(kpbli)`` directly instead of blending it toward
    ``p(kt)`` through ``lev_start``; ``beta`` is 2.5 rather than 1.3; and the
    trailing ``kb_adj`` scan is dead, because unlike ``UP`` and ``MID`` the
    branch neither raises ``kb_adj`` to 2 nor zeroes ``zu`` below it.  On
    ``SH2`` the base of the profile is therefore left wherever the beta
    function put it, and ``CUP_gf_sh:462-468`` does the zeroing itself.

    ``fzu_override`` exists because ``fzu`` is the reference's one
    non-bitwise value; see the module docstring.
    """
    zu = _a(nz)
    kb_adj = max(kb, 2)
    if draft == "UP":
        lev_start = min(F(0.9), F(F(0.4) + F(F(csum) * F(0.013))))
        kb_adj = max(kb, 2)
        tunning = F(p[kt] + F(F(p[kpbli] - p[kt]) * lev_start))
        tunning = min(
            F(0.9), F(F(tunning - p[kb_adj]) / F(p[kt] - p[kb_adj]))
        )
        tunning = max(F(0.2), tunning)
        beta = F(1.3)
    elif draft == "SH2":
        tunning = min(
            F(0.8), F(F(p[kpbli] - p[kb_adj]) / F(p[kt] - p[kb_adj]))
        )
        tunning = max(F(0.2), tunning)
        beta = F(2.5)
    else:
        tunning = p[kb]
        tunning = min(F(0.9), F(F(tunning - p[1]) / F(p[kt] - p[1])))
        tunning = max(F(0.2), tunning)
        beta = F(4.0)
    alpha = F(F(F(F(tunning * F(beta - F(2.0))) + _Z1)) / F(_Z1 - tunning))
    if fzu_override is None:
        fzu = F(
            _tgammaf(F(alpha + beta)) / F(_tgammaf(alpha) * _tgammaf(beta))
        )
    else:
        fzu = F(fzu_override)
    ea = F(alpha - _Z1)
    eb = F(beta - _Z1)
    if draft in ("UP", "SH2"):
        klo, khi = kb_adj, min(nz, kt)
        pbase = p[kb_adj]
    else:
        klo, khi = 2, min(kt, ktf)
        pbase = p[1]
    for k in range(klo, khi + 1):
        kratio = F(F(p[k] - pbase) / F(p[kt] - pbase))
        zu[k] = F(
            zubeg + F(F(fzu * _powf(kratio, ea)) * _powf(F(_Z1 - kratio), eb))
        )
    hi = min(ktf, kt + 1)
    peak = zu[1 : hi + 1].max()
    if peak > _Z0:
        zu[1 : hi + 1] = (zu[1 : hi + 1] / peak).astype(F)
    if draft in ("UP", "SH2"):
        for k in range(_maxloc(zu, 1, nz), 0, -1):
            if zu[k] < F(1.0e-6):
                kb_adj = k + 1
                break
        if draft == "UP":
            kb_adj = max(2, kb_adj)
            zu[1:kb_adj] = _Z0
    else:
        zu[1] = _Z0
    return zu, dict(tunning=tunning, alpha=alpha, beta=beta, fzu=fzu, kb_adj=kb_adj)


def rates_up_pdf_deep(
    *, ktop, ierr, p_cup, entr_rate_2d, hkbo, heo, heso_cup, z_cup, kstabi,
    k22, kbcon, csum, nz, ktf, fzu_override=None,
):
    """module_cu_gf_deep.F:3697-3823, ``name == 'deep'``.

    Three things a reader of the intent would get wrong.  ``kbcon`` is raised
    to at least 2 for EVERY column, ierr or not (:3726).  The ramp built into
    ``zuo`` between ``k22`` and ``kbcon`` (:3731-3737) is then thrown away --
    ``get_zu_zd_pdf_fim`` zeroes ``zu`` on entry.  And the ``kpbl`` argument
    the deep call passes is ``kbcon``, not the boundary layer at all; it is
    read only on the ``shallow`` branch.
    """
    zuo = _a(nz)
    kbcon = max(kbcon, 2)
    info = dict(kklev=0, kfinalzu=0, ktopdby=0)
    if ierr != 0:
        return zuo, ktop, 0, kbcon, ierr, info
    dby = _a(nz)
    dbm = _a(nz)
    hcot = _a(nz)
    start_level = k22
    zuo[start_level] = F(0.1)
    for k in range(start_level + 1, kbcon + 1):
        dz = F(z_cup[k] - z_cup[k - 1])
        massent = F(F(dz * entr_rate_2d[k - 1]) * zuo[k - 1])
        massdetr = F(F(dz * F(1.0e-9)) * zuo[k - 1])
        zuo[k] = F(F(zuo[k - 1] + massent) - massdetr)
    ktop = 0
    hcot[start_level] = hkbo
    for k in range(start_level + 1, ktf - 1):
        dz = F(z_cup[k] - z_cup[k - 1])
        hcot[k] = F(
            F(
                F(F(_Z1 - F(F(_HALF * entr_rate_2d[k - 1]) * dz)) * hcot[k - 1])
                + F(F(entr_rate_2d[k - 1] * dz) * heo[k - 1])
            )
            / F(_Z1 + F(F(_HALF * entr_rate_2d[k - 1]) * dz))
        )
        if k >= kbcon:
            dby[k] = F(dby[k - 1] + F(F(hcot[k] - heso_cup[k]) * dz))
            dbm[k] = F(hcot[k] - heso_cup[k])
    ktopdby = _maxloc(dby, 1, nz)
    kklev = _maxloc(dbm, 1, nz)
    dbymax = dby[1 : nz + 1].max()
    kfinalzu = ktf - 2
    ktop = kfinalzu
    for k in range(ktopdby + 1, ktf - 1):
        if dby[k] < F(_Z1 * dbymax):
            kfinalzu = k - 1
            ktop = kfinalzu
            break
    info = dict(kklev=kklev, kfinalzu=kfinalzu, ktopdby=ktopdby)
    if kfinalzu <= kbcon + 2:
        ierr = 41
        ktop = 0
        return zuo, ktop, ktopdby, kbcon, ierr, info
    zuo, pdf = get_zu_zd_pdf_fim(
        draft="UP", p=p_cup, kb=k22, kt=kfinalzu, kpbli=kstabi, csum=csum,
        zubeg=F(0.1), nz=nz, ktf=ktf, fzu_override=fzu_override,
    )
    info.update(pdf)
    return zuo, ktop, ktopdby, kbcon, ierr, info


def rates_up_pdf_shallow(
    *, ktop, ierr, p_cup, entr_rate_2d, z_cup, kpbl, k22, kbcon, nz, ktf,
    fzu_override=None,
):
    """module_cu_gf_deep.F:3697-3823, ``name == 'shallow'``.

    Much shorter than the deep arm because ``ktop`` arrives already decided --
    ``CUP_gf_sh:434-447`` takes it from ``get_inversion_layers``' 800 hPa slot
    or from 200 mb above ``kbcon`` -- so there is no ``hcot`` integration and
    no ``dby`` search here.  What remains is the same three things the deep
    arm does before its own search: ``kbcon`` is raised to at least 2 for
    EVERY column, ierr or not; a linear ramp is built into ``zuo`` between
    ``k22`` and ``kbcon``; and that ramp is then thrown away, because
    ``get_zu_zd_pdf_fim`` zeroes ``zu`` on entry.

    The ramp survives on exactly one path: ``ktop <= kbcon + 2`` sets
    ``ierr = 41`` and returns WITHOUT calling the pdf, so ``zuo`` comes back
    carrying it.  ``CUP_gf_sh`` never reads ``zuo`` on a rejected column, so
    that is invisible to WRF and visible to a bitwise capture.

    ``hkbo``, ``heo`` and ``heso_cup`` are arguments of the Fortran routine
    and are read only on the deep branch, so they are not parameters here.
    """
    zuo = _a(nz)
    kbcon = max(kbcon, 2)
    info = dict(tunning=_Z0, alpha=_Z0, beta=_Z0, fzu=_Z0, kb_adj=0, kfinalzu=0)
    if ierr != 0:
        return zuo, ktop, kbcon, ierr, info
    start_level = k22
    zuo[start_level] = F(0.1)
    for k in range(start_level + 1, kbcon + 1):
        dz = F(z_cup[k] - z_cup[k - 1])
        massent = F(F(dz * entr_rate_2d[k - 1]) * zuo[k - 1])
        massdetr = F(F(dz * F(1.0e-9)) * zuo[k - 1])
        zuo[k] = F(F(zuo[k - 1] + massent) - massdetr)
    if ktop <= kbcon + 2:
        ierr = 41
        ktop = 0
        return zuo, ktop, kbcon, ierr, info
    kfinalzu = ktop
    zuo, pdf = get_zu_zd_pdf_fim(
        draft="SH2", p=p_cup, kb=k22, kt=kfinalzu, kpbli=kpbl, csum=0,
        zubeg=F(0.1), nz=nz, ktf=ktf, fzu_override=fzu_override,
    )
    info.update(pdf)
    info["kfinalzu"] = kfinalzu
    return zuo, ktop, kbcon, ierr, info


# --------------------------------------------------------------------------
def get_lateral_massflux(*, ierr, ktop, zo_cup, zuo, cd, entr_rate_2d, kbcon,
                         k22, nz, ktf, lambau=None):
    """module_cu_gf_deep.F:4239-4334.  Writes ``cd`` and ``entr_rate_2d`` back.

    ``lambau`` is ``None`` on the shallow call site (module_cu_gf_sh.F:490),
    which omits the OPTIONAL ``up_massentru``/``up_massdetru``/``lambau``
    triple entirely -- so the momentum-transport limb at :4315-4320 does not
    run and the returned ``upmeu``/``upmdu`` stay zero.

    ``up_massentr``/``up_massdetr`` -- the non-``o`` pair -- are a straight
    copy of ``up_massentro``/``up_massdetro`` over ``k = 1 .. ktf-2``
    (:4312-4315), taken AFTER the ``ktop`` assignment and the above-``ktop``
    zeroing.  Since ``ktop <= ktf-2`` on every column that survives
    ``CUP_gf_sh``'s own guard, the copy is total and the two pairs are equal
    everywhere; this function returns one pair and the callers use it for
    both.
    """
    upme = _a(nz)
    upmd = _a(nz)
    upmeu = _a(nz)
    upmdu = _a(nz)
    if ierr != 0:
        return upme, upmd, upmeu, upmdu, cd, entr_rate_2d
    kpeak = _maxloc(zuo, 1, nz)
    for k in range(max(2, k22 + 1), kpeak + 1):
        dz = F(zo_cup[k] - zo_cup[k - 1])
        upmd[k - 1] = F(F(cd[k - 1] * dz) * zuo[k - 1])
        upme[k - 1] = F(F(zuo[k] - zuo[k - 1]) + upmd[k - 1])
        if upme[k - 1] < _Z0:
            upme[k - 1] = _Z0
            upmd[k - 1] = F(zuo[k - 1] - zuo[k])
            if zuo[k - 1] > _Z0:
                cd[k - 1] = F(upmd[k - 1] / F(dz * zuo[k - 1]))
        if zuo[k - 1] > _Z0:
            entr_rate_2d[k - 1] = F(upme[k - 1] / F(dz * zuo[k - 1]))
    for k in range(kpeak + 1, ktop + 1):
        dz = F(zo_cup[k] - zo_cup[k - 1])
        upme[k - 1] = F(F(entr_rate_2d[k - 1] * dz) * zuo[k - 1])
        upmd[k - 1] = F(F(zuo[k - 1] + upme[k - 1]) - zuo[k])
        if upmd[k - 1] < _Z0:
            upmd[k - 1] = _Z0
            upme[k - 1] = F(zuo[k] - zuo[k - 1])
            if zuo[k - 1] > _Z0:
                entr_rate_2d[k - 1] = F(upme[k - 1] / F(dz * zuo[k - 1]))
        if zuo[k - 1] > _Z0:
            cd[k - 1] = F(upmd[k - 1] / F(dz * zuo[k - 1]))
    upmd[ktop] = zuo[ktop]
    upme[ktop] = _Z0
    for k in range(ktop + 1, ktf + 1):
        cd[k] = _Z0
        entr_rate_2d[k] = _Z0
        upme[k] = _Z0
        upmd[k] = _Z0
    if lambau is not None:
        for k in range(2, ktf):
            upmeu[k - 1] = F(upme[k - 1] + F(lambau * upmd[k - 1]))
            upmdu[k - 1] = F(upmd[k - 1] + F(lambau * upmd[k - 1]))
    return upme, upmd, upmeu, upmdu, cd, entr_rate_2d


# --------------------------------------------------------------------------
def cup_up_moisture(*, ierr, z_cup, p_cup, kbcon, ktop, dby, xland1, q,
                    gamma_cup, zu, qes_cup, k22, qe_cup, zqexec, ccn, rho,
                    c1d, t, up_massentr, up_massdetr, nz):
    """module_cu_gf_deep.F:3355-3642, ``autoconv = 1`` (the non-Berry arm).

    ``c0`` is the trap.  It is set to .004 once per column, and the
    below-LFC loop (:3496-3512) then multiplies it by ``exp(.07*(T-273.15))``
    CUMULATIVELY across levels without resetting -- so a column with several
    sub-freezing levels below ``kbcon`` carries a compounded ``c0`` into every
    later one.  The above-LFC loop (:3517) does reset it, every level.  A port
    that hoists the reset out of one loop or into the other is wrong in
    opposite directions.
    """
    qc = _a(nz)
    qrc = _a(nz)
    pw = _a(nz)
    clw_all = _a(nz)
    pwav = _Z0
    psum = _Z0
    psumh = _Z0
    if ierr != 0:
        return qc, qrc, pw, clw_all, pwav, psum, psumh, ierr
    for k in range(1, nz + 1):
        qc[k] = qe_cup[k]
    qch = qc.copy()
    start_level = k22
    qaver = get_cloud_bc(qe_cup, k22)
    qc[start_level] = qaver
    qch[start_level] = qaver
    for k in range(1, start_level):
        qc[k] = qe_cup[k]
        qch[k] = qe_cup[k]

    c0 = F(0.004)
    for k in range(k22 + 1, kbcon + 1):
        if t[k] < F(273.15):
            c0 = F(c0 * _expf(F(F(0.07) * F(t[k] - F(273.15)))))
        qc[k] = F(
            F(
                F(F(qc[k - 1] * zu[k - 1]) - F(F(_HALF * up_massdetr[k - 1]) * qc[k - 1]))
                + F(up_massentr[k - 1] * q[k - 1])
            )
            / F(F(zu[k - 1] - F(_HALF * up_massdetr[k - 1])) + up_massentr[k - 1])
        )
        qrch = F(
            qes_cup[k]
            + F(
                F(F(_Z1 / XLV) * F(gamma_cup[k] / F(_Z1 + gamma_cup[k])))
                * dby[k]
            )
        )
        if k < kbcon:
            qrch = qc[k]
        if qc[k] > qrch:
            dz = F(z_cup[k] - z_cup[k - 1])
            qrc[k] = F(F(qc[k] - qrch) / F(_Z1 + F(c0 * dz)))
            pw[k] = F(F(F(c0 * dz) * qrc[k]) * zu[k])
            qc[k] = F(qrch + qrc[k])
            clw_all[k] = qrc[k]

    for k in range(kbcon + 1, ktop + 1):
        c0 = F(0.004)
        if t[k] < F(273.15):
            c0 = F(c0 * _expf(F(F(0.07) * F(t[k] - F(273.15)))))
        denom = F(F(zu[k - 1] - F(_HALF * up_massdetr[k - 1])) + up_massentr[k - 1])
        if denom < F(1.0e-8):
            ierr = 51
            break
        dz = F(z_cup[k] - z_cup[k - 1])
        qrch = F(
            qes_cup[k]
            + F(
                F(F(_Z1 / XLV) * F(gamma_cup[k] / F(_Z1 + gamma_cup[k])))
                * dby[k]
            )
        )
        qc[k] = F(
            F(
                F(F(qc[k - 1] * zu[k - 1]) - F(F(_HALF * up_massdetr[k - 1]) * qc[k - 1]))
                + F(up_massentr[k - 1] * q[k - 1])
            )
            / denom
        )
        qch[k] = F(
            F(
                F(F(qch[k - 1] * zu[k - 1]) - F(F(_HALF * up_massdetr[k - 1]) * qch[k - 1]))
                + F(up_massentr[k - 1] * q[k - 1])
            )
            / denom
        )
        if qc[k] <= qrch:
            qc[k] = qrch
        if qch[k] <= qrch:
            qch[k] = qrch
        clw_all[k] = max(_Z0, F(qc[k] - qrch))
        qrc[k] = max(_Z0, F(qc[k] - qrch))
        qrc[k] = F(F(qc[k] - qrch) / F(_Z1 + F(F(c1d[k] + c0) * dz)))
        pw[k] = F(F(F(c0 * dz) * qrc[k]) * zu[k])
        if qrc[k] < _Z0:
            qrc[k] = _Z0
            pw[k] = _Z0
        qc[k] = F(qrc[k] + qrch)
        pwav = F(pwav + pw[k])
        psum = F(psum + F(F(clw_all[k] * zu[k]) * dz))
    # The ierr = 51 ``exit`` leaves the k loop but NOT the ``if(ierr.eq.0)``
    # block, so this sweep runs on both paths.
    for k in range(k22 + 1, ktop + 1):
        qc[k] = F(qc[k] - qrc[k])
    return qc, qrc, pw, clw_all, pwav, psum, psumh, ierr


def cup_dd_moisture(*, ierr, zd, hcd, hes_cup, qes_cup, q_cup, z_cup,
                    dd_massentr, dd_massdetr, jmin, gamma_cup, q, nz):
    """module_cu_gf_deep.F:1996-2139, ``iloop = 1``."""
    qcd = _a(nz)
    qrcd = _a(nz)
    pwd = _a(nz)
    pwev = _Z0
    bu = _Z0
    if ierr != 0:
        return qcd, qrcd, pwd, pwev, bu, ierr, None
    k = jmin
    dz = F(z_cup[k + 1] - z_cup[k])
    qcd[k] = q_cup[k]
    dh = F(hcd[k] - hes_cup[k])
    if dh < _Z0:
        qrcd[k] = F(
            qes_cup[k]
            + F(F(F(_Z1 / XLV) * F(gamma_cup[k] / F(_Z1 + gamma_cup[k]))) * dh)
        )
    else:
        qrcd[k] = qes_cup[k]
    pwd[jmin] = F(zd[jmin] * min(_Z0, F(qcd[k] - qrcd[k])))
    qcd[k] = qrcd[k]
    pwev = F(pwev + pwd[jmin])
    bu = F(dz * dh)
    ierrc = None
    for ki in range(jmin - 1, 0, -1):
        dz = F(z_cup[ki + 1] - z_cup[ki])
        denom = F(F(zd[ki + 1] - F(_HALF * dd_massdetr[ki])) + dd_massentr[ki])
        if denom < F(1.0e-8):
            ierr = 51
            break
        qcd[ki] = F(
            F(
                F(F(qcd[ki + 1] * zd[ki + 1]) - F(F(_HALF * dd_massdetr[ki]) * qcd[ki + 1]))
                + F(dd_massentr[ki] * q[ki])
            )
            / denom
        )
        dh = F(hcd[ki] - hes_cup[ki])
        bu = F(bu + F(dz * dh))
        qrcd[ki] = F(
            qes_cup[ki]
            + F(F(F(_Z1 / XLV) * F(gamma_cup[ki] / F(_Z1 + gamma_cup[ki]))) * dh)
        )
        dqeva = F(qcd[ki] - qrcd[ki])
        if dqeva > _Z0:
            dqeva = _Z0
            qrcd[ki] = qcd[ki]
        pwd[ki] = F(zd[ki] * dqeva)
        qcd[ki] = qrcd[ki]
        pwev = F(pwev + pwd[ki])
    if pwev == _Z0:
        ierr = 7
        ierrc = "problem with buoy in cup_dd_moisture"
    if bu >= _Z0:
        ierr = 7
        ierrc = "problem2 with buoy in cup_dd_moisture"
    return qcd, qrcd, pwd, pwev, bu, ierr, ierrc


# --------------------------------------------------------------------------
def cup_up_aa0(*, z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, ktf):
    """module_cu_gf_deep.F:2968-3035.  ``dby`` is read one level BELOW ``k``."""
    aa0 = _Z0
    if ierr != 0:
        return aa0
    for k in range(2, ktf + 1):
        if k < kbcon or k > ktop:
            continue
        dz = F(z[k] - z[k - 1])
        da = F(
            F(F(F(zu[k] * dz) * F(F(9.81) / F(F(1004.0) * t_cup[k]))) * dby[k - 1])
            / F(_Z1 + gamma_cup[k])
        )
        aa0 = F(aa0 + max(_Z0, da))
        if aa0 < _Z0:
            aa0 = _Z0
    return aa0


def cup_up_aa1bl(*, t, tn, q, qo, dtime, z, kbcon, ierr, ktf):
    """module_cu_gf_deep.F:3990-4061.

    ``zu``, ``dby``, ``gamma_cup`` and ``t_cup`` are declared ``intent(in)``
    and never read, which is essential: ``CUP_gf:1186`` passes ``dbyo_bl``,
    ``GAMMAo_CUP_bl`` and ``tn_cup_bl``, three arrays that are only ever
    filled on the dead ``iversion = 0`` branch.  WRF passes uninitialised
    memory here on every call and gets away with it.
    """
    aa0 = _Z0
    if ierr != 0:
        return aa0
    for k in range(2, ktf + 1):
        if k > kbcon:
            continue
        dz = F(z[k] - z[k - 1])
        da = F(
            F(
                F(dz * F(9.81))
                * F(F(tn[k] - t[k]) + F(F(0.608) * F(qo[k] - q[k])))
            )
            / dtime
        )
        aa0 = F(aa0 + da)
    return aa0


def cup_dd_edt(*, ierr, us, vs, z, ktop, kbcon, p, pwav, pwev, edtmax, edtmin,
               ktf):
    """module_cu_gf_deep.F:1871-1993, ``aeroevap = 1`` (the aerosol arm off).

    ``VSHEAR**2`` and ``**3`` are integer literal exponents and fold to
    multiply chains, bitwise, per the probe.
    """
    if ierr != 0:
        return _Z0, _Z0
    vws = _Z0
    sdp = _Z0
    vshear = _Z0
    for kk in range(1, ktf):
        if min(ktop, ktf) >= kk >= kbcon:
            vws = F(
                vws
                + F(
                    F(
                        abs(F(F(us[kk + 1] - us[kk]) / F(z[kk + 1] - z[kk])))
                        + abs(F(F(vs[kk + 1] - vs[kk]) / F(z[kk + 1] - z[kk])))
                    )
                    * F(p[kk] - p[kk + 1])
                )
            )
            # (sdp + p(kk)) - p(kk+1), NOT sdp + (p(kk) - p(kk+1)).  The
            # difference is 1 ULP in vshear and 4 in edt, and the fixture
            # sees it on 50 of the 60 converged columns.
            sdp = F(F(sdp + p[kk]) - p[kk + 1])
        if kk == ktf - 1:
            vshear = F(F(F(1.0e3) * vws) / sdp)
    pef = F(
        F(
            F(F(1.591) - F(F(0.639) * vshear))
            + F(F(0.0953) * F(vshear * vshear))
        )
        - F(F(0.00496) * F(F(vshear * vshear) * vshear))
    )
    if pef > F(0.9):
        pef = F(0.9)
    if pef < F(0.1):
        pef = F(0.1)
    zkbc = F(z[kbcon] * F(3.281e-3))
    prezk = F(0.02)
    if zkbc > F(3.0):
        prezk = F(
            F(0.96729352)
            + F(
                zkbc
                * F(
                    F(-0.70034167)
                    + F(
                        zkbc
                        * F(
                            F(0.162179896)
                            + F(
                                zkbc
                                * F(
                                    F(-1.2569798e-2)
                                    + F(
                                        zkbc
                                        * F(F(4.2772e-4) - F(zkbc * F(5.44e-6)))
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    if zkbc > F(25.0):
        prezk = F(2.4)
    pefb = F(_Z1 / F(_Z1 + prezk))
    if pefb > F(0.9):
        pefb = F(0.9)
    if pefb < F(0.1):
        pefb = F(0.1)
    edt = F(_Z1 - F(_HALF * F(pefb + pef)))
    einc = F(F(0.2) * edt)
    edtc = F(edt - einc)
    edtc = F(F(-edtc * pwav) / pwev)
    if edtc > edtmax:
        edtc = edtmax
    if edtc < edtmin:
        edtc = edtmin
    return edt, edtc


# --------------------------------------------------------------------------
def cup_forcing_ens_3d(*, closure_n, xland1, aa0, aa1, xaa0, mbdt, dtime,
                       ierr, ierr2, ierr3, axx, mconv, p_cup, ktop, omeg, zd,
                       k22, zu, pr_ens, edt, kbcon, ichoice, dicycle,
                       tau_ecmwf, aa1_bl, nz):
    """module_cu_gf_deep.F:2373-2720.  16 members in four families.

    ``ens_adj`` is 1 on every path: the two lines that would set it to
    .666/.333 over water are commented out at :2494-2495, so the whole
    ``xland < 0.1`` block below is an identity multiply that WRF still
    executes.  It is kept because removing it would be a different rounding
    on a denormal, not because it does anything.
    """
    xf_ens = np.zeros(MAXENS3 + 1, dtype=F)
    forcing = np.zeros(11, dtype=F)
    xf_dicycle = _Z0
    if ierr != 0:
        return xf_ens, forcing, xf_dicycle, closure_n
    xff = np.zeros(MAXENS3 + 1, dtype=F)
    ens_adj = _Z1
    kloc = _maxloc(zu, 1, nz)
    a_ave = axx
    a_ave = max(_Z0, a_ave)
    a_ave = min(a_ave, aa1)
    a_ave = max(_Z0, a_ave)
    xff0 = F(F(aa1 - aa0) / dtime)
    for n in (1, 2, 3, 16):
        xff[n] = max(_Z0, F(F(aa1 - aa0) / dtime))
    forcing[1] = xff[2]

    xomg = _Z0
    kk = 0
    for k in range(kbcon - 1, kbcon + 2):
        if zu[k] > _Z0:
            xomg = F(
                xomg
                - F(
                    F(omeg[k] / F(9.81))
                    / max(_HALF, F(_Z1 - F(F(edt * zd[k]) / zu[k])))
                )
            )
            kk += 1
    if kk > 0:
        xff[4] = F(xomg / F(kk))
    xff[4] = F(BETAJB * xff[4])
    xff[5] = xff[4]
    xff[6] = xff[4]
    for n in (4, 5, 6):
        if xff[n] < _Z0:
            xff[n] = _Z0
    xff[14] = F(BETAJB * xff[4])
    forcing[2] = xff[4]

    den = max(_HALF, F(_Z1 - F(F(edt * zd[kbcon]) / zu[kloc])))
    for n in (7, 8, 9, 15):
        xff[n] = F(mconv / den)
    forcing[3] = xff[8]

    for n in (10, 11, 12, 13):
        xff[n] = F(aa1 / tau_ecmwf)
    if dicycle == 1:
        xff_dicycle = max(_Z0, F(aa1_bl / tau_ecmwf))
    else:
        xff_dicycle = _Z0

    if ichoice == 0 and xff0 < _Z0:
        for n in (1, 2, 3, 10, 11, 12, 13, 16):
            xff[n] = _Z0
        closure_n = F(12.0)

    xk = F(F(xaa0 - aa1) / mbdt)
    forcing[4] = aa0
    forcing[5] = aa1
    forcing[6] = xaa0
    forcing[7] = xk
    if xk <= _Z0 and xk > F(F(-0.01) * mbdt):
        xk = F(F(-0.01) * mbdt)
    if xk > _Z0 and xk < F(1.0e-2):
        xk = F(1.0e-2)

    if xland1 < 1:  # xland(i) is the INTEGER xland1 here; .lt.0.1 means == 0
        if ierr2 > 0 or ierr3 > 0:
            for n in range(1, MAXENS3 + 1):
                xff[n] = F(ens_adj * xff[n])
            xff_dicycle = F(ens_adj * xff_dicycle)

    if xk < _Z0:
        for n in (1, 2, 3, 16):
            if xff[n] > _Z0:
                xf_ens[n] = max(_Z0, F(F(-xff[n]) / xk))
    else:
        for n in (1, 2, 3, 16):
            xff[n] = _Z0

    for n in (4, 5, 6, 14):
        xf_ens[n] = max(_Z0, xff[n])
    for n, floor in ((7, F(1.0e-5)), (8, F(1.0e-5)), (9, F(1.0e-5)), (15, F(1.0e-3))):
        a1 = max(floor, pr_ens[n])
        xf_ens[n] = max(_Z0, F(xff[n] / a1))

    if xk < _Z0:
        for n in (10, 11, 12, 13):
            xf_ens[n] = max(_Z0, F(F(-xff[n]) / xk))
        forcing[8] = xf_ens[11]
    else:
        for n in (10, 11, 12, 13):
            xf_ens[n] = _Z0
        forcing[8] = _Z0

    if xk < _Z0:
        xf_dicycle = max(_Z0, F(F(-xff_dicycle) / xk))
    else:
        xf_dicycle = _Z0

    if ichoice >= 1:
        for n in range(1, MAXENS3 + 1):
            xf_ens[n] = xf_ens[ichoice]
    return xf_ens, forcing, xf_dicycle, closure_n


def cup_output_ens_3d(*, xf_ens, ierr, dellat, dellaq, dellaqc, zu, pw, ktop,
                      edt, pwd, p_cup, pr_ens, sig, closure_n, xmbs_in,
                      dicycle, xf_dicycle, nz):
    """module_cu_gf_deep.F:3142-3353, ``imid = 0``.

    Two quirks a tidy port loses.  ``xf_ens(i,:) = sig(i)*xf_ens(i,:)`` sits
    INSIDE the k loop (:3344), so the ensemble is scaled by ``sig`` once per
    level from ``kts`` to ``ktop`` -- ``sig**ktop``, not ``sig``.  And
    ``xmb_ave = min(xmb_ave, xmb_ave - xf_dicycle)`` (:3268) is the diurnal
    subtraction written as a min, which is the same thing only because
    ``xf_dicycle >= 0``.
    """
    outt = _a(nz)
    outq = _a(nz)
    outqc = _a(nz)
    pre = _Z0
    xmb = _Z0
    xf_ens = xf_ens.copy()
    if ierr != 0:
        return outt, outq, outqc, pre, xmb, xf_ens, ierr
    for n in range(1, MAXENS3 + 1):
        if pr_ens[n] <= _Z0:
            xf_ens[n] = _Z0
    xmb_ave = _Z0
    for n in range(1, MAXENS3 + 1):
        xmb_ave = F(xmb_ave + xf_ens[n])
    xmb_ave = F(xmb_ave / F(MAXENS3))
    if dicycle == 2:
        xmb_ave = F(xmb_ave - max(_Z0, xmbs_in))
        xmb_ave = max(_Z0, xmb_ave)
    elif dicycle == 1:
        xmb_ave = min(xmb_ave, F(xmb_ave - xf_dicycle))
        xmb_ave = max(_Z0, xmb_ave)
    clos_wei = F(F(16.0) / max(_Z1, closure_n))
    xmb_ave = min(xmb_ave, F(100.0))
    xmb = F(F(clos_wei * sig) * xmb_ave)
    if xmb < F(1.0e-16):
        ierr = 19
    pwtot = _Z0
    if ierr != 0:
        return outt, outq, outqc, pre, xmb, xf_ens, ierr
    for k in range(1, ktop + 1):
        pwtot = F(pwtot + pw[k])
    for k in range(1, ktop + 1):
        dp = F(F(F(100.0) * F(p_cup[k] - p_cup[k + 1])) / G)
        dtt = dellat[k]
        dtq = dellaq[k]
        dtpwd = F(-F(pwd[k] * edt))
        dtqc = F(F(dellaqc[k] * dp) - dtpwd)
        if dtqc < _Z0:
            dtpwd = F(dtpwd - F(dellaqc[k] * dp))
            dtqc = _Z0
        else:
            dtpwd = _Z0
            dtqc = F(dtqc / dp)
        outt[k] = F(xmb * dtt)
        outq[k] = F(xmb * dtq)
        outqc[k] = F(xmb * dtqc)
        xf_ens[1:] = (F(sig) * xf_ens[1:]).astype(F)
        pre = F(pre - F(xmb * dtpwd))
    pre = F(F(-pre) + F(xmb * pwtot))
    return outt, outq, outqc, pre, xmb, xf_ens, ierr


def neg_check(name, dt, outq, outt, outu, outv, outqc, pret, ktf):
    """module_cu_gf_deep.F:3038-3139.

    The routine ``return``s at :3102, so everything below it -- the negative-q
    rescale the comment describes at length -- is unreachable.  Only the
    heating-rate cap runs.  ``q`` and ``dt`` are therefore never read.
    """
    thresh = F(300.01)
    names = F(1.0)
    if name == "shallow":
        thresh = F(148.01)
        names = F(2.0)
    qmemf = _Z1
    for k in range(1, ktf + 1):
        qmem = F(outt[k] * F(86400.0))
        if qmem > thresh:
            qmemf = min(qmemf, F(thresh / qmem))
        if qmem < F(F(F(-_HALF) * thresh) * names):
            qmemf = min(qmemf, F(F(F(F(-_HALF) * names) * thresh) / qmem))
    for k in range(1, ktf + 1):
        outq[k] = F(outq[k] * qmemf)
        outt[k] = F(outt[k] * qmemf)
        outu[k] = F(outu[k] * qmemf)
        outv[k] = F(outv[k] * qmemf)
        outqc[k] = F(outqc[k] * qmemf)
    return F(pret * qmemf), qmemf


def get_inversion_layers(*, ierr, p_cup, t_cup, z_cup, kstart, kend, nz, ktf):
    """module_cu_gf_deep.F:4063-4159.

    Reached only from ``CUP_gf_sh`` (module_cu_gf_sh.F:413) and the dead
    ``imid = 1`` arm.  ``qo_cup`` and ``qeso_cup`` are formal arguments and
    are never read.

    DIVERGENCE, deliberate: the first-derivative loop runs to ``kend+7`` and
    reads ``t_cup(k+1)``, i.e. index ``kend+8``, against an array declared
    ``kts:kte``.  Both call sites pass ``kend = kstabi``, bounded by
    ``cup_minimi`` only at ``ktf-1``, so at ``kte = 40`` any ``kstabi > 32``
    is an out-of-bounds read.  This port clamps ``kend`` to ``ktf-8`` and
    reports the clamp; the oracle capture clamps identically and counts 0
    clamps on the present fixture.
    """
    dtempdz = _a(nz)
    k_inv = np.ones(nz + 1, dtype=np.int32)
    clamped = False
    if ierr != 0:
        return dtempdz, k_inv, clamped
    if kend > ktf - 8:
        kend = ktf - 8
        clamped = True
    first = _a(nz)
    sec = _a(nz)
    kend_p3 = kend + 3
    for k in range(2, kend_p3 + 5):
        first[k] = F(F(t_cup[k + 1] - t_cup[k - 1]) / F(z_cup[k + 1] - z_cup[k - 1]))
        dtempdz[k] = first[k]
    for k in range(3, kend_p3 + 4):
        sec[k] = F(F(first[k + 1] - first[k - 1]) / F(z_cup[k + 1] - z_cup[k - 1]))
        sec[k] = abs(sec[k])
    ilev = max(3, kstart + 1)
    ix = 1
    k = ilev
    while ilev < kend_p3:
        for kk in range(k, kend_p3 + 3):
            if sec[kk] < sec[kk + 1] and sec[kk] < sec[kk - 1]:
                k_inv[ix] = kk
                ix = min(5, ix + 1)
                ilev = kk + 1
                break
            ilev = kk + 1
        k = ilev
    kadd = 0
    ken = _maxloc(k_inv, 1, nz)
    for k in range(1, ken + 1):
        kk = k_inv[k + kadd]
        if kk == 1:
            break
        if dtempdz[kk] < dtempdz[kk - 1] and dtempdz[kk] < dtempdz[kk + 1]:
            kadd += 1
            for kj in range(k, ken + 1):
                if k_inv[kj + kadd] > 1:
                    k_inv[kj] = k_inv[kj + kadd]
                if k_inv[kj + kadd] == 1:
                    k_inv[kj] = 1
    # the 800 / 550 hPa slots
    big = F(1.0e9)
    sd = np.full(nz + 1, big, dtype=F)
    top = _maxloc(k_inv, 1, nz)
    for k in range(1, top + 1):
        dp = F(p_cup[k_inv[k]] - p_cup[kstart])
        sd[k] = F(abs(dp) - F(100.0))
    k800 = _minloc(np.abs(sd), 1, nz)
    sd = np.full(nz + 1, big, dtype=F)
    for k in range(1, top + 1):
        dp = F(p_cup[k_inv[k]] - p_cup[kstart])
        sd[k] = F(abs(dp) - F(300.0))
    k550 = _minloc(np.abs(sd), 1, nz)
    shal = k_inv[k800]
    mid = k_inv[k550]
    k_inv[1] = shal
    k_inv[2] = mid
    k_inv[3:] = -1
    return dtempdz, k_inv, clamped

