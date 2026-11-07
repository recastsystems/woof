"""P3 microphysics: CUDA WRF ``mp_physics=50/52`` and mp=50 CPU reference.

Transcription authority: byte-frozen WRF v4.6.1 ``phys/module_mp_p3.F``
(P3 v4.5.2, Morrison & Milbrandt 2015): ``p3_init`` constants :177-316,
``mp_p3_wrapper_wrf`` :690-932, ``p3_main`` :1905-5201, and the helper
subprograms ``access_lookup_table`` :5205-5243, ``access_lookup_table_coll``
:5246-5313, ``polysvp1`` :6017-6086, ``find_lookupTable_indices_1a/1b/3``
:6316-6410/:6589-6632, ``get_cloud_dsd2`` :6635-6701, ``get_rain_dsd2``
:6705-6780, ``calc_bulkRhoRime`` :6784-6830, ``impose_max_total_Ni``
:6833-6855, ``qv_sat`` :6859-6889.  Every citation ``:NNNN`` below is a
line in that file.

SCOPE: the ``mp_physics = 50`` (``p3_1category``) and ``mp_physics = 51``
(``p3_1category_nc``) configurations, which are how WRF's driver calls the
scheme (module_microphysics_driver.F:1557-1602 passes no ``nc_3d`` for 50;
:1607-1647 passes ``nc_3d = qnc_curr`` for 51; neither passes ``qzi1_3d``;
module_physics_init.F:4568-4571 calls ``p3_init('.', 1, .false., 'WRF',
...)`` for both):

* ``nCat = 1``: the multi-category blocks (``iceice_interaction1``
  :2754-2829, ``icecat_destination``, the category-merge section
  :4615-4697, lookup table 2) are structurally unreachable and NOT ported;
  the CUDA adapter below carries those blocks for ``mp_physics = 52``.
* 2-moment ice: the ``log_3momentIce`` branches (``zitot``, the 3-moment
  lookup tables, ``compute_mu_3moment``/``G_of_mu``) are absent from this
  CPU reference; the CUDA kernels carry them for ``mp_physics=53``.
  A reference-backend request is refused by name.
* ``log_predictNc`` follows the wrapper's ``present(nc_3d)`` (:819-820):
  ``.false.`` (mp = 50) makes droplet number the diagnostic ``nccnst/rho``
  (:2350, :3875); ``.true.`` (mp = 51) carries it as a prognostic with the
  CCN activation of :3313-3339 (and P3's own ``DERF``, :6090-6204), the
  droplet-number sinks of :3770 and :3872-3873 and the two-moment cloud
  sedimentation of :4009-4066.
* ``scpf_on = .false.`` (the wrapper hardcodes it, :815): ``compute_SCPF``'s
  off-branch is transcribed (SCF = iSCF = SPF = iSPF = 1, SPF_clr = 0,
  Qv_cld = qv, :1889-1899); the Sundqvist cloud-fraction branch is NOT
  ported.  ``SCF_out`` is not computed: WRF's wrapper receives it into a
  local ``cldfrac`` it never reads (:811, :868).
* ``typeDiags_ON = .false.``, ``debug_on = .false.`` (wrapper parameters,
  :802-803): the precip-type partition and ``check_values`` are NOT ported.
* The visibility diagnostics (``diag_vis*``) and user diagnostic arrays
  ``diag_2d``/``diag_3d`` are absent/zeroed in the WRF call and are NOT
  ported; ``mflux_r``/``mflux_i`` exist only to feed ``diag_vis2/3``
  (:4989-4997) and are not stored.

TRANSCRIBED QUIRKS (faithful, deliberately not "fixed"):

* The nucleation-possible test keeps Fortran operator precedence
  (``.and.`` binds tighter than ``.or.``), so ``.not. SCPF_on`` guards only
  the warm branch (:2358-2360).
* ``xxlv`` uses the CONSTANT-Lv expression ``3.1484e6-2370.*273.15``
  (:2325, the ``t(i,k)`` factor is commented out in the authority).
* ``t`` is diagnosed once from theta at entry and deliberately NOT
  refreshed after latent heating (:3856-3858 comment); homogeneous
  freezing thresholds use the stale value (:4540, :4578).
* The first-step guard ``max(t_old, 1.)`` (:2329) covers WRF's
  zero-initialized ``th_old`` on the first call.

PRECISION MODEL, float32 throughout (NumPy scalars under NEP 50 weak
promotion), with the authority's explicit double-precision excursions kept
in float64: the ``dexp(dble(...))`` saturation-relaxation factor (:3202,
:3204, :3231) and the rain-freezing log-space products (:3025-3029).
The predicted-Nc CPU reference uses the shared host twins of the device
transcendentals and the native multiplication order for integer powers.
The specified-Nc reference retains the candidate's NumPy arithmetic so
adding selector 51 does not change selector 50. A context-local mode keeps
concurrent reference calls independent. These routines are proof/debug
code; production column physics runs in the CUDA translation unit.

EXECUTION MODEL: this is the CPU float32 authority (scalar per-cell
loops, matching the Fortran's statement order exactly), reachable as
``run.p3_backend = "reference"`` (:func:`apply_reference`).  The
production path is the device port (woof/core/p3_device.py,
woof/core/kernels/p3.cu), which :func:`apply` dispatches to.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps

import numpy as np

from woof.core.noahmp_libm import expf as _native_expf
from woof.core.noahmp_libm import logf as _native_logf
from woof.core.noahmp_libm import powf as _native_powf
from woof.core.noahmp_libm import powf_array as _native_powf_array
from woof.core.p3_tables import (
    DENSIZE,
    ISIZE,
    RCOLLSIZE,
    RIMSIZE,
    generate_rain_tables,
    load_lookup_table_1,
    load_lookup_table_2,
    p3_table_root,
)

f32 = np.float32
f64 = np.float64


def _rescue_overflowed_product(*factors):
    """The same product, evaluated so a PARTIAL product cannot overflow.

    THE DEFECT, measured on a real 6 h forecast, not postulated.  A GFS run
    on the shipped ``p3-mp50-...`` suite went non-finite at step 284 in two
    cells near 198 K, and BOTH arms -- the CUDA kernel and this
    transcription -- produced it identically, out of
    ``module_mp_p3.F:2994-2996``::

        tmp1  = cdist1(i,k)*exp(aimm*(273.15-t(i,k)))
        Q_nuc = cons6*gamma(7.+mu_c(i,k))*tmp1*dum**2

    At the measured cell -- t = 198.99 K, cdist1 = 3.394e5, mu_c = 12.576,
    lamc = 1.6846e4 -- the LEFT-TO-RIGHT single-precision partial product
    ``cons6*gamma*tmp1`` reaches 4.8e45 and overflows to +Inf, while the
    mathematical result, 4.4e19 once the ``dum**2`` = 9.26e-27 factor is
    applied, is an ordinary float32.  The Inf then meets the conservation
    limiter at :3571-3583, where ``ratio = sources/sinks`` is 0 and
    ``qcheti = Inf * 0`` is NaN; that NaN is what reached theta and vapour.

    WRF IS UNDEFINED HERE rather than wrong-but-authoritative.  Fortran does
    not fix the association of ``a*b*c*d``, so whether the expression
    overflows is the compiler's choice, and P3's own authors left the
    double-precision form commented out three lines below the live one
    (:2999-3002) -- they met this.  woof's standing rule for that case is
    to implement defined behaviour and document the divergence, never to be
    bit-exact to a bug.

    THE DIVERGENCE IS EXACTLY THE OVERFLOWING CASE AND NOTHING ELSE.  Every
    caller evaluates WRF's single-precision chain first and reaches for this
    only when the result is not finite, so every value WRF's arithmetic can
    represent is still produced by WRF's arithmetic, bit for bit, and every
    Fortran-oracle receipt already issued still stands.
    """

    product = f64(1.0)
    for factor in factors:
        product = product * f64(factor)
    return f32(product)

# ---------------------------------------------------------------------------
# p3_init physical constants (:177-316), computed in float32 in source order.
# ---------------------------------------------------------------------------

PI = f32(3.14159265)                      # :178
THRD = f32(1.0) / f32(3.0)                # :180
SXTH = f32(1.0) / f32(6.0)                # :181
PIOV3 = PI * THRD                         # :182
PIOV6 = PI * SXTH                         # :183
MAX_TOTAL_NI = f32(2000.0e3)              # :186
NCCNST = f32(200.0e6)                     # :196
CP = f32(1005.0)                          # :203
INV_CP = f32(1.0) / CP                    # :204
G = f32(9.816)                            # :205
RD = f32(287.15)                          # :206
RV = f32(461.51)                          # :207
EP_2 = f32(0.622)                         # :208
RHOSUR = f32(100000.0) / (RD * f32(273.15))   # :209
RHOSUI = f32(60000.0) / (RD * f32(253.15))    # :210
F1R = f32(0.78)                           # :213
F2R = f32(0.32)                           # :214
RHOW = f32(1000.0)                        # :216
CPW = f32(4218.0)                         # :217
INV_RHOW = f32(1.0) / RHOW                # :218
MU_R_CONSTANT = f32(0.0)                  # :219
INV_DRMAX = f32(1.0) / f32(0.002)         # :222
RHO_RIMEMIN = f32(50.0)                   # :225
RHO_RIMEMAX = f32(900.0)                  # :226
INV_RHO_RIMEMAX = f32(1.0) / RHO_RIMEMAX  # :227
QSMALL = f32(1.0e-14)                     # :230
NSMALL = f32(1.0e-16)                     # :231
BSMALL = QSMALL * INV_RHO_RIMEMAX         # :232
BIMM = f32(2.0)                           # :239 Barklie and Gokhale (1959)
AIMM = f32(0.65)                          # :240
MI0 = f32(4.0) * PIOV3 * f32(900.0) * f32(1.0e-18)   # :242
ECI = f32(0.5)                            # :244
ERI = f32(1.0)                            # :245
BCN = f32(2.0)                            # :246
NMLTRATIO = f32(1.0)                      # :251
CONS1 = PIOV6 * RHOW                                       # :259
CONS2 = f32(4.0) * PIOV3 * RHOW                            # :260
CONS3 = f32(1.0) / (CONS2 * f32(25.0e-6) ** f32(3.0))      # :261
CONS5 = PIOV6 * BIMM                                       # :263
CONS6 = PIOV6 ** f32(2.0) * RHOW * BIMM                    # :264
CONS7 = f32(4.0) * PIOV3 * RHOW * f32(1.0e-6) ** f32(3.0)  # :265
# aerosol/droplet activation parameters (:268-294), mp = 51.  bact is
# evaluated as p3_init evaluates it, left to right in float32.
MW = f32(0.018)                           # :269
OSM = f32(1.0)                            # :270
VI = f32(3.0)                             # :271
EPSM = f32(0.9)                           # :272
RHOA = f32(1777.0)                        # :273
MAP = f32(0.132)                          # :274
RR = f32(8.3187)                          # :276
BACT = VI * OSM * EPSM * MW * RHOA / (MAP * RHOW)   # :277
INV_RM1 = f32(2.0e7)                      # :281 mode 1
SIG1 = f32(2.0)                           # :282
NANEW1 = f32(300.0e6)                     # :283
INV_RM2 = f32(7.6923076e5)                # :290 mode 2
SIG2 = f32(2.5)                           # :291
NANEW2 = f32(0.0)                         # :292

# CPU reference arithmetic is scheme-scoped. The MP51 restoration must not
# change the established MP50 reference trajectory when MP51 is off.
_PREDICTED_NC_MATH = ContextVar("p3_predicted_nc_math", default=False)


def _nc_reference_math(function):
    @wraps(function)
    def call(*args, **kwargs):
        token = _PREDICTED_NC_MATH.set(bool(kwargs.get("log_predictNc", False)))
        try:
            return function(*args, **kwargs)
        finally:
            _PREDICTED_NC_MATH.reset(token)
    return call


def _expf(x):
    return _native_expf(x) if _PREDICTED_NC_MATH.get() else np.exp(x)


def _logf(x):
    return _native_logf(x) if _PREDICTED_NC_MATH.get() else np.log(x)


def _powf(x, y):
    return _native_powf(x, y) if _PREDICTED_NC_MATH.get() else x ** y


def _powf_array(x, y):
    return _native_powf_array(x, y) if _PREDICTED_NC_MATH.get() else x ** y


def _square(x):
    return x * x if _PREDICTED_NC_MATH.get() else x ** f32(2.0)


def _sixth(x):
    if not _PREDICTED_NC_MATH.get():
        return x ** f32(6.0)
    x2 = x * x
    return (x2 * x2) * x2


def _log10f(x):
    if not _PREDICTED_NC_MATH.get():
        return np.log10(x)
    return f32(math.log10(float(x)))


def _dexp(x):
    return math.exp(float(x)) if _PREDICTED_NC_MATH.get() else np.exp(f64(x))


def polysvp1(t: np.float32, i_type: int) -> np.float32:
    """Saturation vapor pressure [Pa] (:6017-6086), float32.

    Flatau et al. (1992) polynomials above 195.8 K (ice) / 202.0 K
    (liquid), Goff-Gratch below.
    """
    t = f32(t)
    if i_type == 1 and t < f32(273.15):
        if t >= f32(195.8):
            dt = t - f32(273.15)
            a0i, a1i, a2i = f32(6.11147274), f32(0.503160820), f32(0.188439774e-1)
            a3i, a4i, a5i = f32(0.420895665e-3), f32(0.615021634e-5), f32(0.602588177e-7)
            a6i, a7i, a8i = f32(0.385852041e-9), f32(0.146898966e-11), f32(0.252751365e-14)
            out = a0i + dt * (a1i + dt * (a2i + dt * (a3i + dt * (a4i + dt * (
                a5i + dt * (a6i + dt * (a7i + a8i * dt)))))))
            return out * f32(100.0)
        return _powf(f32(10.0), f32(-9.09718) * (f32(273.16) / t - f32(1.0))
                     - f32(3.56654) * _log10f(f32(273.16) / t)
                     + f32(0.876793) * (f32(1.0) - t / f32(273.16))
                     + _log10f(f32(6.1071))) * f32(100.0)
    # liquid branch (i_type == 0 .or. t >= 273.15)
    if t >= f32(202.0):
        dt = t - f32(273.15)
        a0, a1, a2 = f32(6.11239921), f32(0.443987641), f32(0.142986287e-1)
        a3, a4, a5 = f32(0.264847430e-3), f32(0.302950461e-5), f32(0.206739458e-7)
        a6, a7, a8 = f32(0.640689451e-10), f32(-0.952447341e-13), f32(-0.976195544e-15)
        out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt * (a4 + dt * (
            a5 + dt * (a6 + dt * (a7 + a8 * dt)))))))
        return out * f32(100.0)
    return _powf(f32(10.0), f32(-7.90298) * (f32(373.16) / t - f32(1.0))
                 + f32(5.02808) * _log10f(f32(373.16) / t)
                 - f32(1.3816e-7) * (_powf(f32(10.0), f32(11.344) * (
                     f32(1.0) - t / f32(373.16))) - f32(1.0))
                 + f32(8.1328e-3) * (_powf(f32(10.0), f32(-3.49149) * (
                     f32(373.16) / t - f32(1.0))) - f32(1.0))
                 + _log10f(f32(1013.246))) * f32(100.0)


#: Saturation pressure at T = 0 C (:257) -- depends on polysvp1, so defined
#: after it, exactly as p3_init orders the assignments.
E0 = polysvp1(f32(273.15), 0)


def qv_sat(t_atm: np.float32, p_atm: np.float32, i_wrt: int) -> np.float32:
    """Saturation mixing ratio w.r.t. liquid (0) or ice (1) (:6859-6889)."""
    e_pres = polysvp1(t_atm, i_wrt)
    return EP_2 * e_pres / max(f32(1.0e-3), p_atm - e_pres)


def _gammaf(x) -> np.float32:
    """Fortran REAL ``gamma``: gamma in double, rounded once to float32
    (p3.cu ``p3_gam``; the oracle's correctly rounded ``tgammaf``)."""
    return f32(math.gamma(float(x)))


#: DERF's two coefficient tables (:6098-6173), REAL(4) as gfortran makes
#: them from the DATA statements' decimal text -- checked bit for bit
#: against gfortran 13.3.0's own conversion of those statements, and held
#: to p3.cu's P3_DERF_A/P3_DERF_B by tests/test_p3_cuda.py.
DERF_A = np.asarray([
    0.00000000005958930743, -0.00000000113739022964,
    0.00000001466005199839, -0.00000016350354461960,
    0.00000164610044809620, -0.00001492559551950604,
    0.00012055331122299265, -0.00085483269811296660,
    0.00522397762482322257, -0.02686617064507733420,
    0.11283791670954881569, -0.37612638903183748117,
    1.12837916709551257377,
    0.00000000002372510631, -0.00000000045493253732,
    0.00000000590362766598, -0.00000006642090827576,
    0.00000067595634268133, -0.00000621188515924000,
    0.00005103883009709690, -0.00037015410692956173,
    0.00233307631218880978, -0.01254988477182192210,
    0.05657061146827041994, -0.21379664776456006580,
    0.84270079294971486929,
    0.00000000000949905026, -0.00000000018310229805,
    0.00000000239463074000, -0.00000002721444369609,
    0.00000028045522331686, -0.00000261830022482897,
    0.00002195455056768781, -0.00016358986921372656,
    0.00107052153564110318, -0.00608284718113590151,
    0.02986978465246258244, -0.13055593046562267625,
    0.67493323603965504676,
    0.00000000000382722073, -0.00000000007421598602,
    0.00000000097930574080, -0.00000001126008898854,
    0.00000011775134830784, -0.00000111992758382650,
    0.00000962023443095201, -0.00007404402135070773,
    0.00050689993654144881, -0.00307553051439272889,
    0.01668977892553165586, -0.08548534594781312114,
    0.56909076642393639985,
    0.00000000000155296588, -0.00000000003032205868,
    0.00000000040424830707, -0.00000000471135111493,
    0.00000005011915876293, -0.00000048722516178974,
    0.00000430683284629395, -0.00003445026145385764,
    0.00024879276133931664, -0.00162940941748079288,
    0.00988786373932350462, -0.05962426839442303805,
    0.49766113250947636708], dtype=np.float32)
DERF_B = np.asarray([
    -0.00000000029734388465, 0.00000000269776334046,
    -0.00000000640788827665, -0.00000001667820132100,
    -0.00000021854388148686, 0.00000266246030457984,
    0.00001612722157047886, -0.00025616361025506629,
    0.00015380842432375365, 0.00815533022524927908,
    -0.01402283663896319337, -0.19746892495383021487,
    0.71511720328842845913,
    -0.00000000001951073787, -0.00000000032302692214,
    0.00000000522461866919, 0.00000000342940918551,
    -0.00000035772874310272, 0.00000019999935792654,
    0.00002687044575042908, -0.00011843240273775776,
    -0.00080991728956032271, 0.00661062970502241174,
    0.00909530922354827295, -0.20160072778491013140,
    0.51169696718727644908,
    0.00000000003147682272, -0.00000000048465972408,
    0.00000000063675740242, 0.00000003377623323271,
    -0.00000015451139637086, -0.00000203340624738438,
    0.00001947204525295057, 0.00002854147231653228,
    -0.00101565063152200272, 0.00271187003520095655,
    0.02328095035422810727, -0.16725021123116877197,
    0.32490054966649436974,
    0.00000000002319363370, -0.00000000006303206648,
    -0.00000000264888267434, 0.00000002050708040581,
    0.00000011371857327578, -0.00000211211337219663,
    0.00000368797328322935, 0.00009823686253424796,
    -0.00065860243990455368, -0.00075285814895230877,
    0.02585434424202960464, -0.11637092784486193258,
    0.18267336775296612024,
    -0.00000000000367789363, 0.00000000020876046746,
    -0.00000000193319027226, -0.00000000435953392472,
    0.00000018006992266137, -0.00000078441223763969,
    -0.00000675407647949153, 0.00008428418334440096,
    -0.00017604388937031815, -0.00239729611435071610,
    0.02064129023876022970, -0.06905562880005864105,
    0.09084526782065478489], dtype=np.float32)


def derf(x) -> np.float32:
    """P3's own REAL error function (:6090-6204), which the activation's
    ``derf`` calls (the module procedure shadows the intrinsic).  Statement
    order and mixed-kind details as written: ``W`` is compared with the
    DOUBLE literals 2.2D0/6.9D0, ``INT`` truncates, ``T - K`` converts
    ``K`` to REAL first."""
    x = f32(x)
    w = abs(x)
    if float(w) < 2.2:                                   # :6175, 2.2D0
        t = w * w
        k = int(t)
        t = t - f32(k)
        k = k * 13
        A = DERF_A
        y = ((((((((((((A[k] * t + A[k + 1]) * t + A[k + 2]) * t
                       + A[k + 3]) * t + A[k + 4]) * t + A[k + 5]) * t
                    + A[k + 6]) * t + A[k + 7]) * t + A[k + 8]) * t
                 + A[k + 9]) * t + A[k + 10]) * t + A[k + 11]) * t
             + A[k + 12]) * w
    elif float(w) < 6.9:                                 # :6185, 6.9D0
        k = int(w)
        t = w - f32(k)
        k = 13 * (k - 2)
        B = DERF_B
        y = (((((((((((B[k] * t + B[k + 1]) * t + B[k + 2]) * t
                      + B[k + 3]) * t + B[k + 4]) * t + B[k + 5]) * t
                   + B[k + 6]) * t + B[k + 7]) * t + B[k + 8]) * t
                + B[k + 9]) * t + B[k + 10]) * t + B[k + 11]) * t + B[k + 12]
        y = y * y
        y = y * y
        y = y * y
        y = f32(1.0) - y * y
    else:
        y = f32(1.0)
    if x < f32(0.0):                                     # :6201
        y = -y
    return f32(y)


@dataclass(frozen=True)
class P3Runtime:
    """The p3_init products the solver reads (tables; constants are module
    level).  ``itab``/``itabcoll`` come from the SHA-256-validated packaged
    table; ``vn/vm/revap`` are the generated rain tables.  A three-moment
    p3_init (``trplMomI=.true.``, mp_physics=53) reads the 3momI table
    INSTEAD of the 2momI one (module_mp_p3.F:343-472), so exactly one of
    the two table pairs is filled."""
    itab: np.ndarray | None        # (densize, rimsize, isize, tabsize)
    itabcoll: np.ndarray | None    # (densize, rimsize, isize, rcollsize, 2)
    vn_table: np.ndarray    # (300, 10)
    vm_table: np.ndarray    # (300, 10)
    revap_table: np.ndarray  # (300, 10)
    itab_3mom: np.ndarray | None = None      # (zsize, densize, rimsize, isize, 15)
    itabcoll_3mom: np.ndarray | None = None  # (zsize, densize, rimsize, isize, rcollsize, 2)

    itabcolli1: np.ndarray | None = None
    itabcolli2: np.ndarray | None = None
    n_cat: int = 1

    @property
    def three_moment(self) -> bool:
        return self.itab_3mom is not None


_RUNTIME_CACHE: dict[tuple[str, int, bool], P3Runtime] = {}


def p3_init(n_cat: int = 1, *, three_moment: bool = False) -> P3Runtime:
    """The WRF ``CASE (P3_1CATEGORY): CALL p3_init('.',1,.false.,'WRF',...)``
    equivalent (module_physics_init.F:4568-4569): read table 1 (2momI) and
    generate the rain tables, once per process per table root.
    ``three_moment=True`` is ``CASE (P3_1CAT_3MOM): CALL
    p3_init('.',1,.true.,'WRF',...)`` (:4574-4575), which reads the 3momI
    table instead."""
    n_cat = int(n_cat)
    if n_cat not in (1, 2) or (n_cat > 1 and three_moment):
        raise ValueError("P3 supports one-category two/three-moment ice or two-category two-moment ice")
    root = p3_table_root()
    key = (root, n_cat, bool(three_moment))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is None:
        if three_moment:
            from woof.core.p3_rust import init_three_moment
            itab, itabcoll, vn, vm, revap = init_three_moment(root)
            runtime = P3Runtime(itab=None, itabcoll=None, vn_table=vn,
                                vm_table=vm, revap_table=revap,
                                itab_3mom=itab, itabcoll_3mom=itabcoll)
        else:
            itab, itabcoll = load_lookup_table_1(root)
            vn, vm, revap = generate_rain_tables()
            runtime = P3Runtime(itab=itab, itabcoll=itabcoll, vn_table=vn,
                                vm_table=vm, revap_table=revap)
        extra = {"n_cat": n_cat}
        if n_cat > 1:
            from woof.core.p3_rust import init_table2
            extra["itabcolli1"], extra["itabcolli2"] = init_table2(root)
        runtime = replace(runtime, **extra)
        _RUNTIME_CACHE[key] = runtime
    return runtime


# ---------------------------------------------------------------------------
# Lookup-table access and index helpers (1-based Fortran index convention is
# kept in the integer values; the -1 shift happens only at array access).
# ---------------------------------------------------------------------------

def access_lookup_table(itab, dumjj, dumii, dumi, index, dum1, dum4, dum5):
    """Trilinear interpolation in the main ice table (:5205-5243)."""
    t = itab
    iproc1 = t[dumjj-1, dumii-1, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj-1, dumii-1, dumi, index-1] - t[dumjj-1, dumii-1, dumi-1, index-1])
    gproc1 = t[dumjj-1, dumii, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj-1, dumii, dumi, index-1] - t[dumjj-1, dumii, dumi-1, index-1])
    tmp1 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    iproc1 = t[dumjj, dumii-1, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj, dumii-1, dumi, index-1] - t[dumjj, dumii-1, dumi-1, index-1])
    gproc1 = t[dumjj, dumii, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj, dumii, dumi, index-1] - t[dumjj, dumii, dumi-1, index-1])
    tmp2 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    return tmp1 + (dum5 - f32(dumjj)) * (tmp2 - tmp1)


def access_lookup_table_coll(itabcoll, dumjj, dumii, dumj, dumi, index,
                             dum1, dum3, dum4, dum5):
    """Quadrilinear interpolation for ice-rain collection (:5246-5313)."""
    t = itabcoll

    def pair(jj, ii):
        dproc1 = t[jj, ii, dumi-1, dumj-1, index-1] + (dum1 - f32(dumi)) * (
            t[jj, ii, dumi, dumj-1, index-1] - t[jj, ii, dumi-1, dumj-1, index-1])
        dproc2 = t[jj, ii, dumi-1, dumj, index-1] + (dum1 - f32(dumi)) * (
            t[jj, ii, dumi, dumj, index-1] - t[jj, ii, dumi-1, dumj, index-1])
        return dproc1 + (dum3 - f32(dumj)) * (dproc2 - dproc1)

    iproc1 = pair(dumjj-1, dumii-1)
    gproc1 = pair(dumjj-1, dumii)
    tmp1 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    iproc1 = pair(dumjj, dumii-1)
    gproc1 = pair(dumjj, dumii)
    tmp2 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    return tmp1 + (dum5 - f32(dumjj)) * (tmp2 - tmp1)


def find_lookupTable_indices_1a(qitot, nitot, qirim, rhop):
    """Main ice-table indices (:6316-6370): returns
    (dumi, dumjj, dumii, dum1, dum4, dum5)."""
    dum1 = (_log10f(qitot / nitot) + f32(18.0)) * f32(3.444606) - f32(10.0)
    dumi = int(dum1)
    dum1 = min(dum1, f32(ISIZE))
    dum1 = max(dum1, f32(1.0))
    dumi = max(1, dumi)
    dumi = min(ISIZE - 1, dumi)

    dum4 = (qirim / qitot) * f32(3.0) + f32(1.0)
    dumii = int(dum4)
    dum4 = min(dum4, f32(RIMSIZE))
    dum4 = max(dum4, f32(1.0))
    dumii = max(1, dumii)
    dumii = min(RIMSIZE - 1, dumii)

    if rhop <= f32(650.0):
        dum5 = (rhop - f32(50.0)) * f32(0.005) + f32(1.0)
    else:
        dum5 = (rhop - f32(650.0)) * f32(0.004) + f32(4.0)
    dumjj = int(dum5)
    dum5 = min(dum5, f32(DENSIZE))
    dum5 = max(dum5, f32(1.0))
    dumjj = max(1, dumjj)
    dumjj = min(DENSIZE - 1, dumjj)
    return dumi, dumjj, dumii, dum1, dum4, dum5


def find_lookupTable_indices_1b(qr, nr):
    """Rain index for ice-rain collection (:6374-6410): (dumj, dum3)."""
    if qr >= QSMALL and nr > f32(0.0):
        dumlr = _powf(qr / (PI * RHOW * nr), THRD)
        dum3 = (_log10f(f32(1.0) * dumlr) + f32(5.0)) * f32(10.70415)
        dumj = int(dum3)
        dum3 = min(dum3, f32(RCOLLSIZE))
        dum3 = max(dum3, f32(1.0))
        dumj = max(1, dumj)
        dumj = min(RCOLLSIZE - 1, dumj)
    else:
        dumj = 1
        dum3 = f32(1.0)
    return dumj, dum3


def find_lookupTable_indices_3(mu_r, lamr):
    """Rain fallspeed/ventilation table indices (:6589-6632):
    (dumii, dumjj, dum1, rdumii, rdumjj, inv_dum3)."""
    dum1 = (mu_r + f32(1.0)) / lamr
    if dum1 <= f32(195.0e-6):
        inv_dum3 = f32(0.1)
        rdumii = (dum1 * f32(1.0e6) + f32(5.0)) * inv_dum3
        rdumii = max(rdumii, f32(1.0))
        rdumii = min(rdumii, f32(20.0))
        dumii = int(rdumii)
        dumii = max(dumii, 1)
        dumii = min(dumii, 20)
    else:
        inv_dum3 = THRD * f32(0.1)
        rdumii = (dum1 * f32(1.0e6) - f32(195.0)) * inv_dum3 + f32(20.0)
        rdumii = max(rdumii, f32(20.0))
        rdumii = min(rdumii, f32(300.0))
        dumii = int(rdumii)
        dumii = max(dumii, 20)
        dumii = min(dumii, 299)
    rdumjj = mu_r + f32(1.0)
    rdumjj = max(rdumjj, f32(1.0))
    rdumjj = min(rdumjj, f32(10.0))
    dumjj = int(rdumjj)
    dumjj = max(dumjj, 1)
    dumjj = min(dumjj, 9)
    return dumii, dumjj, dum1, rdumii, rdumjj, inv_dum3


def get_cloud_dsd2(qc_grd, nc_grd, rho, iscf):
    """Cloud DSD parameters (:6635-6701).

    Returns ``(nc_grd, mu_c, nu, lamc, cdist, cdist1)`` -- ``nc_grd`` is the
    (possibly lambda-limited) modified grid-mean droplet number the Fortran
    passes back through its INOUT argument.
    """
    qc = qc_grd * iscf
    if qc >= QSMALL:
        nc = nc_grd * iscf
        nc = max(nc, NSMALL)
        mu_c = f32(0.0005714) * (nc * f32(1.0e-6) * rho) + f32(0.2714)
        mu_c = f32(1.0) / (mu_c * mu_c) - f32(1.0)  # 1./(mu_c**2)-1.
        mu_c = max(mu_c, f32(2.0))
        mu_c = min(mu_c, f32(15.0))
        nu = f32(0.0)  # WRF p3_init pins iparam=3 (:193).
        lamc = _powf(CONS1 * nc * (mu_c + f32(3.0)) * (mu_c + f32(2.0))
                     * (mu_c + f32(1.0)) / qc, THRD)
        lammin = (mu_c + f32(1.0)) * f32(2.5e4)
        lammax = (mu_c + f32(1.0)) * f32(1.0e6)
        if lamc < lammin:
            lamc = lammin
            # :6681 -- lamc**3 is an INTEGER exponent, which gfortran expands
            # to multiplications at every -O level; it does not call powf.
            nc = f32(6.0) * ((lamc * lamc) * lamc) * qc / (
                PI * RHOW * (mu_c + f32(3.0)) * (mu_c + f32(2.0))
                * (mu_c + f32(1.0)))
        elif lamc > lammax:
            lamc = lammax
            nc = f32(6.0) * ((lamc * lamc) * lamc) * qc / (   # :6684, int **3
                PI * RHOW * (mu_c + f32(3.0)) * (mu_c + f32(2.0))
                * (mu_c + f32(1.0)))
        cdist = nc * (mu_c + f32(1.0)) / lamc
        cdist1 = nc / _gammaf(mu_c + f32(1.0))
        nc_grd = nc / iscf
        return nc_grd, mu_c, nu, lamc, cdist, cdist1
    return nc_grd, f32(0.0), f32(0.0), f32(0.0), f32(0.0), f32(0.0)


def get_rain_dsd2(qr_grd, nr_grd, ispf):
    """Rain DSD parameters (:6705-6780).

    Returns ``(nr_grd, mu_r, lamr, cdistr, logn0r)`` -- ``nr_grd`` is the
    modified grid-mean rain number.
    """
    qr = qr_grd * ispf
    if qr >= QSMALL:
        nr = nr_grd * ispf
        nr = max(nr, NSMALL)
        # inv_dum feeds only the commented-out variable-mu_r path; computed
        # for parity with the authority's statement order (:6734).
        _ = _powf(qr / (CONS1 * nr * f32(6.0)), THRD)
        mu_r = MU_R_CONSTANT
        lamr = _powf(CONS1 * nr * (mu_r + f32(3.0)) * (mu_r + f32(2.0))
                     * (mu_r + f32(1.0)) / qr, THRD)
        lammax = (mu_r + f32(1.0)) * f32(1.0e5)
        lammin = (mu_r + f32(1.0)) * INV_DRMAX
        if lamr < lammin:
            lamr = lammin
            nr = _expf(f32(3.0) * _logf(lamr) + _logf(qr)
                       + _logf(_gammaf(mu_r + f32(1.0)))
                       - _logf(_gammaf(mu_r + f32(4.0)))) / CONS1
        elif lamr > lammax:
            lamr = lammax
            nr = _expf(f32(3.0) * _logf(lamr) + _logf(qr)
                       + _logf(_gammaf(mu_r + f32(1.0)))
                       - _logf(_gammaf(mu_r + f32(4.0)))) / CONS1
        logn0r = _log10f(nr) + (mu_r + f32(1.0)) * _log10f(lamr) \
            - _log10f(_gammaf(mu_r + f32(1.0)))
        cdistr = nr / _gammaf(mu_r + f32(1.0))
        nr_grd = nr / ispf
        return nr_grd, mu_r, lamr, cdistr, logn0r
    return nr_grd, f32(0.0), f32(0.0), f32(0.0), f32(0.0)


def calc_bulkRhoRime(qi_tot, qi_rim, bi_rim):
    """Bulk rime density with qirim/birim consistency (:6784-6830).

    Returns ``(qi_rim, bi_rim, rho_rime)``.
    """
    if bi_rim >= f32(1.0e-15):
        rho_rime = qi_rim / bi_rim
        if rho_rime < RHO_RIMEMIN:
            rho_rime = RHO_RIMEMIN
            bi_rim = qi_rim / rho_rime
        elif rho_rime > RHO_RIMEMAX:
            rho_rime = RHO_RIMEMAX
            bi_rim = qi_rim / rho_rime
    else:
        qi_rim = f32(0.0)
        bi_rim = f32(0.0)
        rho_rime = f32(0.0)
    if qi_rim > qi_tot and rho_rime > f32(0.0):
        qi_rim = qi_tot
        bi_rim = qi_rim / rho_rime
    if qi_rim < QSMALL:
        qi_rim = f32(0.0)
        bi_rim = f32(0.0)
    return qi_rim, bi_rim, rho_rime


def impose_max_total_Ni(nitot, inv_rho_local):
    """nCat = 1 form of the total ice number cap (:6833-6855)."""
    if nitot >= f32(1.0e-20):
        dum = MAX_TOTAL_NI * inv_rho_local / nitot
        nitot = nitot * min(dum, f32(1.0))
    return nitot


# ---------------------------------------------------------------------------
# p3_main (:1905-5201), nCat = 1 / 2-moment ice / WRF orientation.
# ---------------------------------------------------------------------------

def _rain_table_value(table, ii, jj, rdumii, rdumjj):
    dum1 = table[ii-1, jj-1] + (rdumii - f32(ii)) * (
        table[ii, jj-1] - table[ii-1, jj-1])
    dum2 = table[ii-1, jj] + (rdumii - f32(ii)) * (
        table[ii, jj] - table[ii-1, jj])
    return dum1 + (rdumjj - f32(jj)) * (dum2 - dum1)


def _sediment_column(moments, fraction, dt, rho, inv_rho, inv_dz, speeds_at):
    """The shared adaptive flux transport, preserving scalar float32 order."""
    mass = moments[0]
    nk = mass.size
    top = next((k for k in range(nk - 1, -1, -1)
                if mass[k] * fraction >= QSMALL), None)
    if top is None:
        return None
    bottom = next(k for k in range(top + 1) if mass[k] * fraction >= QSMALL)
    velocities = [np.zeros(nk, dtype=np.float32) for _ in moments]
    fluxes = [np.zeros(nk, dtype=np.float32) for _ in moments]
    dt_left = dt
    prt_accum = f32(0.0)
    while dt_left > f32(1.0e-4):
        Co_max = f32(0.0)
        for velocity in velocities:
            velocity[:] = f32(0.0)
        for k in range(top, bottom - 1, -1):
            for velocity, value in zip(velocities, speeds_at(k)):
                velocity[k] = value
            Co_max = max(Co_max, velocities[0][k] * dt_left * inv_dz[k])
        tmpint1 = int(Co_max + f32(1.0))
        dt_sub = min(dt_left, dt_left / f32(tmpint1))
        k_temp = bottom if bottom == 0 else bottom - 1
        for k in range(k_temp, top + 1):
            for flux, velocity, q in zip(fluxes, velocities, moments):
                flux[k] = velocity[k] * q[k] * rho[k]
        if bottom == 0:
            prt_accum = prt_accum + fluxes[0][0] * dt_sub
        k = top
        divs = [-flux[k] * inv_dz[k] for flux in fluxes]
        for q, div in zip(moments, divs):
            q[k] = q[k] + div * dt_sub * inv_rho[k]
        for k in range(top - 1, k_temp - 1, -1):
            divs = [(flux[k + 1] - flux[k]) * inv_dz[k] for flux in fluxes]
            for q, div in zip(moments, divs):
                q[k] = q[k] + div * dt_sub * inv_rho[k]
        dt_left = dt_left - dt_sub
        if bottom != 0:
            bottom = bottom - 1
    return prt_accum


@_nc_reference_math
def p3_main(qc, nc, qr, nr, th_old, th, qv_old, qv, dt, qitot, qirim, nitot,
            birim, ssat, pres, dzq, it, prt_liq, prt_sol,
            diag_ze, diag_effc, diag_effi, diag_vmi, diag_di, diag_rhoi,
            *, n_cat: int = 1, log_predictNc: bool = False,
            model: str = "WRF", clbfact_dep: float = 1.0,
            clbfact_sub: float = 1.0, runtime: P3Runtime | None = None
            ) -> None:
    """The P3 main solver on an (ni, nk) slab, k index 0 = surface.

    All (ni, nk) float32 arrays are updated in place exactly as the
    Fortran INOUT arguments are; ``prt_liq``/``prt_sol`` (ni,) receive the
    liquid/solid surface precipitation rates [m s-1].  ``uzpl`` (vertical
    velocity) is omitted: the authority reads one element solely to
    silence an unused-argument warning (:2204).
    """
    if n_cat != 1:
        raise NotImplementedError(
            "woof's P3 CPU reference carries one ice category (mp_physics="
            "50 or 51); mp_physics=52 requires the two-category CUDA kernels.")
    if model != "WRF":
        raise NotImplementedError("only the WRF orientation is ported")
    rt = runtime if runtime is not None else p3_init()
    itab = rt.itab
    itabcoll = rt.itabcoll
    vn_table = rt.vn_table
    vm_table = rt.vm_table
    revap_table = rt.revap_table

    ni_pts, nk = qc.shape

    dt = f32(dt)
    odt = f32(1.0) / dt                 # :2261
    # timeScaleFactor (:2265) feeds only the commented-out soft rain-lambda
    # limiter (:4741-4750); not carried.

    inv_dzq = (f32(1.0) / dzq).astype(np.float32)   # :2260

    prt_liq[:] = f32(0.0)
    prt_sol[:] = f32(0.0)
    diag_ze[:] = f32(-99.0)
    diag_effc[:] = f32(10.0e-6)
    diag_effi[:] = f32(25.0e-6)
    diag_vmi[:] = f32(0.0)
    diag_di[:] = f32(0.0)
    diag_rhoi[:] = f32(0.0)
    ze_ice = np.full((ni_pts, nk), f32(1.0e-22), dtype=np.float32)
    ze_rain = np.full((ni_pts, nk), f32(1.0e-22), dtype=np.float32)
    rhorime_c = f32(400.0)              # :2290 (assigned once, outside loops)

    # :2293-2297 -- slab-wide diagnostics at entry.
    tmparr1 = _powf_array((pres * f32(1.0e-5)).astype(np.float32),
                          RD * INV_CP).astype(np.float32)
    invexn = (f32(1.0) / tmparr1).astype(np.float32)
    t = (th * tmparr1).astype(np.float32)
    t_old = (th_old * tmparr1).astype(np.float32)
    np.maximum(qv, f32(0.0), out=qv)

    # constant-Lv thermodynamic coefficients (:2325-2327) -- these carry no
    # t dependence in the authority, so they are plain scalars here.
    xxlv = f32(3.1484e6) - f32(2370.0) * f32(273.15)
    xxls = xxlv + f32(0.3337e6)
    xlf = xxls - xxlv

    for i in range(ni_pts):
        log_hydrometeorsPresent = False
        log_nucleationPossible = False

        rho = np.empty(nk, dtype=np.float32)
        inv_rho = np.empty(nk, dtype=np.float32)
        qvs = np.empty(nk, dtype=np.float32)
        qvi = np.empty(nk, dtype=np.float32)
        sup = np.empty(nk, dtype=np.float32)
        supi = np.empty(nk, dtype=np.float32)
        rhofacr = np.empty(nk, dtype=np.float32)
        rhofaci = np.empty(nk, dtype=np.float32)
        acn = np.empty(nk, dtype=np.float32)

        # ------------------------------------------------------------------
        # k_loop_1 (:2320-2411): atmospheric variables + dry-air clipping.
        # ------------------------------------------------------------------
        for k in range(nk):
            rho[k] = pres[i, k] / (RD * t[i, k])
            inv_rho[k] = f32(1.0) / rho[k]
            # first-call guard: t_old is 0 before p3_main ever ran (:2328-2330)
            qvs[k] = qv_sat(max(t_old[i, k], f32(1.0)), pres[i, k], 0)
            qvi[k] = qv_sat(max(t_old[i, k], f32(1.0)), pres[i, k], 1)
            # THE COLD-START qvs/qvi FLOOR, DEFAULT-ON IN ALL THREE ARMS
            # (2026-08-31; ``max(qvs, 1.e-20)`` after both qv_sat calls,
            # the remedy P3 releases newer than this port's v4.5.2
            # transcription target carry themselves).  On
            # the very first p3_main call both th_old and qv_old are the
            # zero the allocation left (WRF never initialises them either
            # -- Registry.EM_COMMON:1598-1599 declare them plain state
            # and no real.exe or start_em path writes them); WRF's
            # `max(t_old,1.)` guard at :2329 keeps polysvp1's argument
            # finite, but polysvp1(1 K) underflows to 0, so stock qvs and
            # qvi are BOTH exactly 0 and the sup/supi diagnoses below
            # evaluate 0.0/0.0 -- a quiet domain-wide NaN this port used
            # to TRANSCRIBE deliberately.  The floor pins step-1 sup/supi
            # at exactly -1 (fully subsaturated, the intended meaning)
            # and is inert from step 2 on, once t_old is real.  Never
            # bit-exact to a bug: where the reference computes an
            # undefined 0/0, this port implements the defined behaviour
            # and documents the divergence, here.
            #
            # This is a REAL behavior change beyond the NaN cosmetics
            # and it is DECLARED, not hidden: with sup = -1 finite, the
            # step-1 dry-air clip below (`qc < 1e-8 .and. sup < -0.1`)
            # can fire in cells whose wrfinput carries trace condensate,
            # where NaN made every comparison false.  Runs are therefore
            # bit-identical to shipped 2.6.0 from step 2 onward; step 1
            # carries exactly this documented floor delta.  The old
            # NaN-containment pin flipped to the floor pin:
            # tests/test_p3_port.py::
            # test_the_first_step_qvs_floor_pins_sup_at_minus_one.
            # The rejected alternative -- an opt-in flag, or a floor on
            # one arm only -- would either ship the defect in a bare
            # default run ("fixed means default") or fork the core into
            # two step-1 behaviors under one module, the exact property
            # the three-arm byte gate exists to protect.
            qvs[k] = max(qvs[k], f32(1.0e-20))
            qvi[k] = max(qvi[k], f32(1.0e-20))
            # log_predictSsat = .false. (:2252) -> always the diagnosed branch
            ssat[i, k] = qv_old[i, k] - qvs[k]
            sup[k] = qv_old[i, k] / qvs[k] - f32(1.0)
            supi[k] = qv_old[i, k] / qvi[k] - f32(1.0)
            rhofacr[k] = _powf(RHOSUR * inv_rho[k], f32(0.54))
            rhofaci[k] = _powf(RHOSUI * inv_rho[k], f32(0.54))
            dum = f32(1.496e-6) * _powf(t[i, k], f32(1.5)) / (t[i, k]
                                                              + f32(120.0))
            acn[k] = G * RHOW / (f32(18.0) * dum)
            # specified droplet number (1-moment cloud, :2349-2351); the
            # prognostic one (mp = 51) is the wrapper's nc_3d, untouched
            if not log_predictNc:
                nc[i, k] = NCCNST * inv_rho[k]

            # Fortran precedence quirk preserved: A .or. (B .and. .not.SCPF)
            # (:2358-2360; scpf_on is hardwired false in the WRF wrapper)
            if ((t[i, k] < f32(273.15) and supi[k] >= f32(-0.05))
                    or (t[i, k] >= f32(273.15) and sup[k] >= f32(-0.05))):
                log_nucleationPossible = True

            # mass clipping in dry air (:2365-2407)
            if qc[i, k] < QSMALL or (qc[i, k] < f32(1.0e-8)
                                     and sup[k] < f32(-0.1)):
                qv[i, k] = qv[i, k] + qc[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qc[i, k] * xxlv * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qr[i, k] < QSMALL or (qr[i, k] < f32(1.0e-8)
                                     and sup[k] < f32(-0.1)):
                qv[i, k] = qv[i, k] + qr[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qr[i, k] * xxlv * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qitot[i, k] < QSMALL or (qitot[i, k] < f32(1.0e-8)
                                        and supi[k] < f32(-0.1)):
                qv[i, k] = qv[i, k] + qitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xxls * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if (qitot[i, k] >= QSMALL and qitot[i, k] < f32(1.0e-8)
                    and t[i, k] >= f32(273.15)):
                qr[i, k] = qr[i, k] + qitot[i, k]
                nr[i, k] = nr[i, k] + nitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xlf * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)

        # first compute_SCPF (:2432-2434), scpf off-branch (:1889-1899):
        # SCF = iSCF = SPF = iSPF = 1, SPF_clr = 0, Qv_cld = qv snapshot.
        iscf = f32(1.0)
        scf = f32(1.0)
        spf = f32(1.0)
        ispf = f32(1.0)
        spf_clr = f32(0.0)
        qv_cld = qv[i, :].copy()

        if not (log_nucleationPossible or log_hydrometeorsPresent):
            continue    # goto 333 (:2439); SCF_out is discarded by WRF
        log_hydrometeorsPresent = False   # :2441

        # ------------------------------------------------------------------
        # k_loop_main (:2449-3927): process rates + prognostic update.
        # ------------------------------------------------------------------
        for k in range(nk):
            # 555-skip: dry + subsaturated level with no hydrometeors
            log_exitlevel = not (qc[i, k] >= QSMALL or qr[i, k] >= QSMALL
                                 or qitot[i, k] >= QSMALL)
            if log_exitlevel and (
                    (t[i, k] < f32(273.15) and supi[k] < f32(-0.05))
                    or (t[i, k] >= f32(273.15) and sup[k] < f32(-0.05))):
                continue    # goto 555 (:2462-2464)

            # zero process rates (:2467-2485); scalars, nCat = 1
            qcacc = qrevp = qccon = f32(0.0)
            qcaut = qcevp = qrcon = f32(0.0)
            ncacc = ncnuc = ncslf = f32(0.0)
            ncautc = qcnuc = nrslf = f32(0.0)
            nrevp = ncautr = f32(0.0)
            qchetc = qisub = nrshdr = f32(0.0)
            qcheti = qrcol = qcshd = f32(0.0)
            qrhetc = qimlt = qccol = f32(0.0)
            qrheti = qinuc = nimlt = f32(0.0)
            nchetc = nccol = ncshdc = f32(0.0)
            ncheti = nrcol = nislf = f32(0.0)
            nrhetc = ninuc = qidep = f32(0.0)
            nrheti = nisub = qwgrth = f32(0.0)
            qrmul = nimul = f32(0.0)
            log_wetgrowth = False
            # predict_supersaturation (:2488-2519): log_predictSsat = false.

            # 444-skip: no hydrometeors -> jump to nucleation (:2523-2528)
            log_exitlevel = not (qc[i, k] >= QSMALL or qr[i, k] >= QSMALL
                                 or qitot[i, k] >= QSMALL)

            # values shared between the process region and the 444 region
            dqsdT = f32(0.0)
            epsi = f32(0.0)
            epsi_tot = f32(0.0)
            f1pr04 = f1pr05 = f1pr09 = f1pr10 = f1pr14 = f32(0.0)
            f1pr02 = f1pr03 = f32(0.0)
            f1pr07 = f1pr08 = f32(-99.0)
            mu_c = lamc = cdist = cdist1 = f32(0.0)
            mu_r = lamr = cdistr = logn0r = f32(0.0)
            abi = f32(1.0)

            if not log_exitlevel:
                # time/space varying physical variables (:2531-2554)
                mu = f32(1.496e-6) * _powf(t[i, k], f32(1.5)) / (
                    t[i, k] + f32(120.0))
                dv = f32(8.794e-5) * _powf(t[i, k], f32(1.81)) / pres[i, k]
                sc = mu / (rho[k] * dv)
                dum = f32(1.0) / (RV * _square(t[i, k]))   # int **2
                dqsdT = xxlv * qvs[k] * dum
                dqsidT = xxls * qvi[k] * dum
                ab = f32(1.0) + dqsdT * xxlv * INV_CP
                abi = f32(1.0) + dqsidT * xxls * INV_CP
                kap = f32(1.414e3) * mu
                if t[i, k] < f32(253.15):
                    eii = f32(0.001)
                elif t[i, k] < f32(273.15):
                    eii = f32(0.001) + (t[i, k] - f32(253.15)) * (
                        f32(0.3) - f32(0.001)) / f32(20.0)
                else:
                    eii = f32(0.3)

                nc[i, k], mu_c, nu, lamc, cdist, cdist1 = get_cloud_dsd2(
                    qc[i, k], nc[i, k], rho[k], iscf)
                nr[i, k], mu_r, lamr, cdistr, logn0r = get_rain_dsd2(
                    qr[i, k], nr[i, k], ispf)

                epsi_tot = f32(0.0)   # :2564
                nitot[i, k] = impose_max_total_Ni(nitot[i, k], inv_rho[k])

                eii_fact = f32(1.0)
                if qitot[i, k] >= QSMALL:   # qitot_notsmall_1 (:2570-2679)
                    nitot[i, k] = max(nitot[i, k], NSMALL)
                    nr[i, k] = max(nr[i, k], NSMALL)
                    # mean-mass ice diameter (:2577-2578) feeds only the
                    # rime-splintering/multicat paths (off at nCat=1).
                    qirim[i, k], birim[i, k], rhop = calc_bulkRhoRime(
                        qitot[i, k], qirim[i, k], birim[i, k])
                    dumi, dumjj, dumii, dum1, dum4, dum5 = \
                        find_lookupTable_indices_1a(
                            qitot[i, k], nitot[i, k], qirim[i, k], rhop)
                    dumj, dum3 = find_lookupTable_indices_1b(qr[i, k], nr[i, k])

                    f1pr02 = access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                                 dum1, dum4, dum5)
                    f1pr03 = access_lookup_table(itab, dumjj, dumii, dumi, 3,
                                                 dum1, dum4, dum5)
                    f1pr04 = access_lookup_table(itab, dumjj, dumii, dumi, 4,
                                                 dum1, dum4, dum5)
                    f1pr05 = access_lookup_table(itab, dumjj, dumii, dumi, 5,
                                                 dum1, dum4, dum5)
                    f1pr09 = access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                                 dum1, dum4, dum5)
                    f1pr10 = access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                                 dum1, dum4, dum5)
                    f1pr14 = access_lookup_table(itab, dumjj, dumii, dumi, 10,
                                                 dum1, dum4, dum5)
                    if qr[i, k] >= QSMALL:
                        f1pr07 = access_lookup_table_coll(
                            itabcoll, dumjj, dumii, dumj, dumi, 1,
                            dum1, dum3, dum4, dum5)
                        f1pr08 = access_lookup_table_coll(
                            itabcoll, dumjj, dumii, dumj, dumi, 2,
                            dum1, dum3, dum4, dum5)
                    else:
                        f1pr07 = f32(-99.0)
                        f1pr08 = f32(-99.0)

                    # lambda limiters on nitot (:2649-2650)
                    nitot[i, k] = min(nitot[i, k], f1pr09 * qitot[i, k])
                    nitot[i, k] = max(nitot[i, k], f1pr10 * qitot[i, k])

                    # ice-ice collection shutoff factor (:2665-2677)
                    if qirim[i, k] > f32(0.0):
                        tmp1 = qirim[i, k] / qitot[i, k]
                        if tmp1 < f32(0.6):
                            eii_fact = f32(1.0)
                        elif tmp1 < f32(0.9):
                            eii_fact = f32(1.0) - (tmp1 - f32(0.6)) / f32(0.3)
                        else:
                            eii_fact = f32(0.0)
                    else:
                        eii_fact = f32(1.0)

                # -- collection of droplets (:2697-2710)
                if (qitot[i, k] >= QSMALL and qc[i, k] >= QSMALL
                        and t[i, k] <= f32(273.15)):
                    qccol = rhofaci[k] * f1pr04 * qc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                    nccol = rhofaci[k] * f1pr04 * nc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                if (qitot[i, k] >= QSMALL and qc[i, k] >= QSMALL
                        and t[i, k] > f32(273.15)):
                    qcshd = rhofaci[k] * f1pr04 * qc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                    nccol = rhofaci[k] * f1pr04 * nc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                    ncshdc = qcshd * f32(1.923e6)

                # -- collection of rain (:2725-2748); SPF-SPF_clr = 1 here
                if (qitot[i, k] >= QSMALL and qr[i, k] >= QSMALL
                        and t[i, k] <= f32(273.15)):
                    qrcol = _powf(f32(10.0), f1pr08 + logn0r) * rho[k] \
                        * rhofaci[k] * ERI * nitot[i, k] * iscf \
                        * (spf - spf_clr)
                    nrcol = _powf(f32(10.0), f1pr07 + logn0r) * rho[k] \
                        * rhofaci[k] * ERI * nitot[i, k] * iscf \
                        * (spf - spf_clr)
                if (qitot[i, k] >= QSMALL and qr[i, k] >= QSMALL
                        and t[i, k] > f32(273.15)):
                    nrcol = _powf(f32(10.0), f1pr07 + logn0r) * rho[k] \
                        * rhofaci[k] * ERI * nitot[i, k] * iscf \
                        * (spf - spf_clr)
                    # collected rain mass (:2743) feeds only the
                    # commented-out shedding source (:2747).

                # iceice_interaction1 (:2754-2829): iice >= 2 only -- never
                # reached at nCat = 1.

                # -- self-collection of ice (:2840-2842)
                if qitot[i, k] >= QSMALL:
                    nislf = f1pr03 * rho[k] * eii * eii_fact * rhofaci[k] \
                        * nitot[i, k] * nitot[i, k] * iscf

                # -- melting (:2851-2865)
                if qitot[i, k] >= QSMALL and t[i, k] > f32(273.15):
                    qsat0 = f32(0.622) * E0 / (pres[i, k] - E0)
                    dum = f32(0.0)
                    qimlt = ((f1pr05 + f1pr14 * _powf(sc, THRD)
                              * _powf(rhofaci[k] * rho[k] / mu, f32(0.5)))
                             * ((t[i, k] - f32(273.15)) * kap
                                - rho[k] * xxlv * dv * (qsat0 - qv_cld[k]))
                             * f32(2.0) * PI / xlf + dum) * nitot[i, k]
                    qimlt = max(qimlt, f32(0.0))
                    nimlt = qimlt * (nitot[i, k] / qitot[i, k])

                # -- wet growth (:2873-2894)
                if (qitot[i, k] >= QSMALL
                        and (qc[i, k] + qr[i, k]) >= f32(1.0e-6)
                        and t[i, k] < f32(273.15)):
                    qsat0 = f32(0.622) * E0 / (pres[i, k] - E0)
                    qwgrth = ((f1pr05 + f1pr14 * _powf(sc, THRD)
                               * _powf(rhofaci[k] * rho[k] / mu, f32(0.5)))
                              * f32(2.0) * PI
                              * (rho[k] * xxlv * dv * (qsat0 - qv_cld[k])
                                 - (t[i, k] - f32(273.15)) * kap)
                              / (xlf + CPW * (t[i, k] - f32(273.15)))) \
                        * nitot[i, k]
                    qwgrth = max(qwgrth, f32(0.0))
                    dum = max(f32(0.0), (qccol + qrcol) - qwgrth)
                    if dum >= f32(1.0e-10):
                        nrshdr = nrshdr + dum * f32(1.923e6)
                        if (qccol + qrcol) >= f32(1.0e-10):
                            dum1 = f32(1.0) / (qccol + qrcol)
                            qcshd = qcshd + dum * qccol * dum1
                            qccol = qccol - dum * qccol * dum1
                            qrcol = qrcol - dum * qrcol * dum1
                        log_wetgrowth = True

                # -- inverse supersaturation relaxation timescale (:2900-2906)
                if qitot[i, k] >= QSMALL and t[i, k] < f32(273.15):
                    epsi = ((f1pr05 + f1pr14 * _powf(sc, THRD)
                             * _powf(rhofaci[k] * rho[k] / mu, f32(0.5)))
                            * f32(2.0) * PI * rho[k] * dv) * nitot[i, k]
                    epsi_tot = epsi_tot + epsi
                else:
                    epsi = f32(0.0)

                # -- rime density, Cober and List 1993 (:2925-2969)
                if qccol >= QSMALL and t[i, k] < f32(273.15):
                    vtrmi1 = f1pr02 * rhofaci[k]
                    iTc = f32(1.0) / min(f32(-0.001), t[i, k] - f32(273.15))
                    if qc[i, k] >= QSMALL:
                        Vt_qc = acn[k] * _gammaf(f32(4.0) + BCN + mu_c) / (
                            _powf(lamc, BCN) * _gammaf(mu_c + f32(4.0)))
                        D_c = (mu_c + f32(4.0)) / lamc
                        V_impact = abs(vtrmi1 - Vt_qc)
                        Ri = -(f32(0.5e6) * D_c) * V_impact * iTc
                        Ri = max(f32(1.0), min(Ri, f32(12.0)))
                        if Ri <= f32(8.0):
                            rhorime_c = (f32(0.051) + f32(0.114) * Ri
                                         - f32(0.0055) * _square(Ri)) \
                                * f32(1000.0)
                        else:
                            rhorime_c = f32(611.0) + f32(72.25) * (Ri - f32(8.0))
                else:
                    rhorime_c = f32(400.0)

                # -- immersion freezing of droplets (:2984-3015)
                if qc[i, k] >= QSMALL and t[i, k] <= f32(269.15):
                    inv_lamc = f32(1.0) / lamc
                    dum = (inv_lamc * inv_lamc) * inv_lamc  # :2988
                    # WRF's own chain first and unchanged, so a value its
                    # arithmetic can represent is still its value, bit for
                    # bit.  The rescue below is reached only when it cannot
                    # -- see _rescue_overflowed_product.
                    with np.errstate(over="ignore", invalid="ignore"):
                        tmp1 = cdist1 * _expf(AIMM * (f32(273.15) - t[i, k]))
                        gam_q = _gammaf(f32(7.0) + mu_c)
                        gam_n = _gammaf(mu_c + f32(4.0))
                        Q_nuc = CONS6 * gam_q * tmp1 * _square(dum)  # int **2
                        N_nuc = CONS5 * gam_n * tmp1 * dum
                    if not np.isfinite(Q_nuc) or not np.isfinite(N_nuc):
                        # Argument in float32, exp in double: the authority's
                        # own commented-out form, dexp(dble(aimm*(273.15-t)))
                        # at :2999, and what the rain branch below does.
                        exp_aimm = _dexp(AIMM * (f32(273.15) - t[i, k]))
                        if not np.isfinite(Q_nuc):
                            Q_nuc = _rescue_overflowed_product(
                                CONS6, gam_q, cdist1, exp_aimm, dum, dum)
                        if not np.isfinite(N_nuc):
                            N_nuc = _rescue_overflowed_product(
                                CONS5, gam_n, cdist1, exp_aimm, dum)
                    qcheti = Q_nuc
                    ncheti = N_nuc

                # -- immersion freezing of rain (:3021-3043)
                if qr[i, k] * ispf >= QSMALL and t[i, k] <= f32(269.15):
                    tmpdbl1 = _dexp(_logf(cdistr)
                                    + _logf(_gammaf(f32(7.0) + mu_r))
                                    - f32(6.0) * _logf(lamr))
                    tmpdbl2 = _dexp(_logf(cdistr)
                                    + _logf(_gammaf(mu_r + f32(4.0)))
                                    - f32(3.0) * _logf(lamr))
                    tmpdbl3 = _dexp(AIMM * (f32(273.15) - t[i, k]))
                    # Same class, same rescue.  WRF's ``sngl(tmpdbl1*tmpdbl3)``
                    # (:3037-3038) rounds a double to float32 and can land on
                    # +Inf for the same cold-cloud reason, reaching the same
                    # limiter and the same NaN.  The double product is already
                    # in hand, so the rescue is one more multiply.
                    with np.errstate(over="ignore", invalid="ignore"):
                        qrheti = CONS6 * f32(tmpdbl1 * tmpdbl3) * spf
                        nrheti = CONS5 * f32(tmpdbl2 * tmpdbl3) * spf
                    if not np.isfinite(qrheti):
                        qrheti = _rescue_overflowed_product(
                            CONS6, tmpdbl1 * tmpdbl3, spf)
                    if not np.isfinite(nrheti):
                        nrheti = _rescue_overflowed_product(
                            CONS5, tmpdbl2 * tmpdbl3, spf)

                # rime splintering (:3049-3106): log_hmossopOn = (nCat>1)
                # (:2256) is FALSE for the ported 1-category configuration.

                # -- condensation/evaporation/deposition/sublimation
                if qr[i, k] * ispf >= QSMALL:   # :3114-3129
                    dumii_r, dumjj_r, dum1, rdumii, rdumjj, inv_dum3 = \
                        find_lookupTable_indices_3(mu_r, lamr)
                    dum = _rain_table_value(revap_table, dumii_r, dumjj_r, rdumii, rdumjj)
                    epsr = f32(2.0) * PI * cdistr * rho[k] * dv * (
                        F1R * _gammaf(mu_r + f32(2.0)) / lamr
                        + F2R * _powf(rho[k] / mu, f32(0.5))
                        * _powf(sc, THRD) * dum)
                else:
                    epsr = f32(0.0)

                if qc[i, k] >= QSMALL:
                    epsc = f32(2.0) * PI * rho[k] * dv * cdist
                else:
                    epsc = f32(0.0)

                if t[i, k] < f32(273.15):
                    oabi = f32(1.0) / abi
                    xx = epsc + epsr + epsi_tot * (
                        f32(1.0) + xxls * INV_CP * dqsdT) * oabi
                else:
                    oabi = f32(1.0) / abi   # value irrelevant on this branch
                    xx = epsc + epsr

                dumqvi = qvi[k]    # :3144

                # 'A' term (:3171-3180)
                dum = -CP / G * (t[i, k] - t_old[i, k]) * odt
                if t[i, k] < f32(273.15):
                    aaa = (qv[i, k] - qv_old[i, k]) * odt \
                        - dqsdT * (-dum * G * INV_CP) \
                        - (qvs[k] - dumqvi) * (f32(1.0) + xxls * INV_CP
                                               * dqsdT) * oabi * epsi_tot
                else:
                    aaa = (qv[i, k] - qv_old[i, k]) * odt \
                        - dqsdT * (-dum * G * INV_CP)

                xx = max(f32(1.0e-20), xx)   # :3182
                oxx = f32(1.0) / xx

                # scpf off (:3185-3190)
                ssat_cld = ssat[i, k]
                ssat_r = ssat[i, k]
                sup_cld = sup[k]
                sup_r = sup[k]
                supi_cld = supi[k]

                if qc[i, k] >= QSMALL:   # :3201-3202
                    qccon = (aaa * epsc * oxx
                             + (ssat_cld * scf - aaa * oxx) * odt * epsc
                             * oxx * (f32(1.0) - f32(_dexp(-f64(xx * dt))))
                             ) / ab
                if qr[i, k] >= QSMALL:   # :3203-3204
                    qrcon = (aaa * epsr * oxx
                             + (ssat_r * spf - aaa * oxx) * odt * epsr
                             * oxx * (f32(1.0) - f32(_dexp(-f64(xx * dt))))
                             ) / ab

                if sup_cld < f32(-0.001) and qc[i, k] < f32(1.0e-12):
                    qccon = -qc[i, k] * odt
                if sup_r < f32(-0.001) and qr[i, k] < f32(1.0e-12):
                    qrcon = -qr[i, k] * odt

                if qccon < f32(0.0):
                    qcevp = -qccon
                    qccon = f32(0.0)
                else:
                    qccon = min(qccon, qv[i, k] * odt)

                if qrcon < f32(0.0):
                    qrevp = -qrcon
                    nrevp = qrevp * (nr[i, k] / qr[i, k])
                    qrcon = f32(0.0)
                else:
                    qrcon = min(qrcon, qv[i, k] * odt)

                # iice_loop_depsub (:3226-3255)
                if qitot[i, k] >= QSMALL and t[i, k] < f32(273.15):
                    qidep = (aaa * epsi * oxx
                             + (ssat_cld * scf - aaa * oxx) * odt * epsi
                             * oxx * (f32(1.0) - f32(_dexp(-f64(xx * dt))))
                             ) * oabi + (qvs[k] - dumqvi) * epsi * oabi
                if supi_cld < f32(-0.001) and qitot[i, k] < f32(1.0e-12):
                    qidep = -qitot[i, k] * odt
                if qidep < f32(0.0):
                    qisub = -qidep
                    qisub = qisub * f32(clbfact_sub)
                    qisub = min(qisub, qitot[i, k] * odt)
                    nisub = qisub * (nitot[i, k] / qitot[i, k])
                    qidep = f32(0.0)
                else:
                    qidep = qidep * f32(clbfact_dep)
                    qidep = min(qidep, qv[i, k] * odt)

            # 444 continue (:3257) -------------------------------------------

            # -- deposition/condensation-freezing nucleation (:3264-3293)
            sup_cld = sup[k]
            supi_cld = supi[k]
            if t[i, k] < f32(258.15) and supi_cld >= f32(0.05):
                dum = f32(0.005) * _expf(f32(0.304) * (f32(273.15) - t[i, k])) \
                    * f32(1000.0) * inv_rho[k]                # Cooper (1986)
                dum = min(dum, f32(100.0e3) * inv_rho[k] * scf)
                N_nuc = max(f32(0.0), (dum - nitot[i, k]) * odt)
                if N_nuc >= f32(1.0e-20):
                    Q_nuc = max(f32(0.0), (dum - nitot[i, k]) * MI0 * odt)
                    qinuc = Q_nuc
                    ninuc = N_nuc

            # -- droplet activation, specified-Nc branch (:3303-3311)
            if not log_predictNc and sup_cld > f32(1.0e-6) and it > 1:
                dum = NCCNST * inv_rho[k] * CONS7 - qc[i, k]
                dum = max(f32(0.0), dum * iscf)
                dumqvs = qv_sat(t[i, k], pres[i, k], 0)
                dqsdT_l = xxlv * dumqvs / (RV * t[i, k] * t[i, k])
                ab_l = f32(1.0) + dqsdT_l * xxlv * INV_CP
                dum = max(f32(0.0), min(dum, (qv_cld[k] - dumqvs) / ab_l))
                qcnuc = dum * odt * scf
            # -- droplet activation, predicted-Nc branch (:3313-3339): two
            # prescribed lognormal aerosol modes; also applied at it = 1,
            # where only the number is activated (the mass comes from the
            # first-step saturation adjustment below).
            if log_predictNc and sup_cld > f32(1.0e-6):
                dum1 = f32(1.0) / _powf(BACT, f32(0.5))
                sigvl = f32(0.0761) - f32(1.55e-4) * (t[i, k] - f32(273.15))
                aact = f32(2.0) * MW / (RHOW * RR * t[i, k]) * sigvl
                sm1 = f32(2.0) * dum1 * _powf(aact * THRD * INV_RM1, f32(1.5))
                sm2 = f32(2.0) * dum1 * _powf(aact * THRD * INV_RM2, f32(1.5))
                uu1 = f32(2.0) * _logf(sm1 / sup_cld) / (f32(4.242)
                                                         * _logf(SIG1))
                uu2 = f32(2.0) * _logf(sm2 / sup_cld) / (f32(4.242)
                                                         * _logf(SIG2))
                dum1 = NANEW1 * f32(0.5) * (f32(1.0) - derf(uu1))
                dum2 = NANEW2 * f32(0.5) * (f32(1.0) - derf(uu2))
                dum2 = min(NANEW1 + NANEW2, dum1 + dum2)
                dum2 = (dum2 - nc[i, k] * iscf) * odt * scf
                dum2 = max(f32(0.0), dum2)
                ncnuc = dum2
                if it <= 1:
                    qcnuc = f32(0.0)
                else:
                    qcnuc = ncnuc * CONS7

            # -- first-step saturation adjustment (:3348-3356)
            if it <= 1:
                dumt = th[i, k] * _powf(pres[i, k] * f32(1.0e-5), RD * INV_CP)
                dumqv = qv_cld[k]
                dumqvs = qv_sat(dumt, pres[i, k], 0)
                dums = dumqv - dumqvs
                qccon = dums / (f32(1.0) + _square(xxlv) * dumqvs
                                / (CP * RV * _square(dumt))) * odt * scf
                qccon = max(f32(0.0), qccon)
                if qccon <= f32(1.0e-7):
                    qccon = f32(0.0)

            # -- autoconversion, iparam = 3 (KK2000) (:3362-3416)
            if qc[i, k] * iscf >= f32(1.0e-8):
                dum = qc[i, k] * iscf
                qcaut = f32(1350.0) * _powf(dum, f32(2.47)) \
                    * _powf(nc[i, k] * iscf * f32(1.0e-6) * rho[k],
                            f32(-1.79)) \
                    * scf
                ncautr = qcaut * CONS3
                ncautc = qcaut * nc[i, k] / qc[i, k]
                if qcaut == f32(0.0):
                    ncautc = f32(0.0)
                if ncautc == f32(0.0):
                    qcaut = f32(0.0)

            # -- self-collection of droplets, iparam = 3 (:3421-3435)
            if qc[i, k] >= QSMALL:
                ncslf = f32(0.0)

            # -- accretion, iparam = 3 (:3440-3472)
            if qr[i, k] >= QSMALL and qc[i, k] >= QSMALL:
                dum2 = spf - spf_clr
                qcacc = f32(67.0) * _powf(qc[i, k] * iscf * qr[i, k] * ispf,
                                          f32(1.15)) * dum2
                ncacc = qcacc * nc[i, k] / qc[i, k]
                if qcacc == f32(0.0):
                    ncacc = f32(0.0)
                if ncacc == f32(0.0):
                    qcacc = f32(0.0)

            # -- rain self-collection/breakup, iparam = 3 (:3478-3504)
            if qr[i, k] >= QSMALL:
                dum1 = f32(280.0e-6)
                nr[i, k] = max(nr[i, k], NSMALL)
                dum2 = _powf(qr[i, k] / (PI * RHOW * nr[i, k]), THRD)
                if dum2 < dum1:
                    dum = f32(1.0)
                else:
                    dum = f32(2.0) - _expf(f32(2300.0) * (dum2 - dum1))
                nrslf = dum * f32(5.78) * nr[i, k] * ispf * qr[i, k] * ispf \
                    * rho[k] * spf

            # -- conservation of mass (:3515-3644)
            dumqvs = qv_sat(t[i, k], pres[i, k], 0)
            qcon_satadj = (qv_cld[k] - dumqvs) / (
                f32(1.0) + _square(xxlv) * dumqvs
                / (CP * RV * _square(t[i, k]))) * odt * scf
            tmp1 = qccon + qrcon + qcnuc
            if tmp1 > f32(0.0) and qcon_satadj < f32(0.0):
                qccon = f32(0.0)
                qrcon = f32(0.0)
                qcnuc = f32(0.0)
                ncnuc = f32(0.0)
            else:
                if tmp1 > f32(0.0) and tmp1 > qcon_satadj:
                    ratio = max(f32(0.0), qcon_satadj) / tmp1
                    ratio = min(f32(1.0), ratio)
                    qccon = qccon * ratio
                    qrcon = qrcon * ratio
                    qcnuc = qcnuc * ratio
                    ncnuc = ncnuc * ratio
                elif qcevp + qrevp > f32(0.0):
                    ratio = max(f32(0.0), -qcon_satadj) / (qcevp + qrevp)
                    ratio = min(f32(1.0), ratio)
                    qcevp = qcevp * ratio
                    qrevp = qrevp * ratio
                    nrevp = nrevp * ratio

            qv_tmp = qv_cld[k] + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt
            t_tmp = t[i, k] + (qcnuc + qccon + qrcon - qcevp - qrevp) \
                * xxlv * INV_CP * dt
            dumqvi = qv_sat(t_tmp, pres[i, k], 1)
            qdep_satadj = (qv_tmp - dumqvi) / (
                f32(1.0) + _square(xxls) * dumqvi
                / (CP * RV * _square(t_tmp))) * odt * scf
            tmp1 = qidep + qinuc
            if tmp1 > f32(0.0) and qdep_satadj < f32(0.0):
                qidep = f32(0.0)
                qinuc = f32(0.0)
                ninuc = f32(0.0)
            else:
                if tmp1 > f32(0.0) and tmp1 > qdep_satadj:
                    ratio = max(f32(0.0), qdep_satadj) / tmp1
                    ratio = min(f32(1.0), ratio)
                    qidep = qidep * ratio
                    qinuc = qinuc * ratio
                    ninuc = ninuc * ratio
                dum = max(qisub, f32(1.0e-20))
                qisub = qisub * min(f32(1.0), max(f32(0.0), -qdep_satadj)
                                    / max(qisub, f32(1.0e-20)))
                nisub = nisub * min(f32(1.0), qisub / dum)

            # cloud (:3571-3590)
            sinks = (qcaut + qcacc + qccol + qcevp + qchetc + qcheti
                     + qcshd) * dt
            sources = qc[i, k] + (qccon + qcnuc) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qcaut = qcaut * ratio
                qcacc = qcacc * ratio
                qcevp = qcevp * ratio
                qccol = qccol * ratio
                qcheti = qcheti * ratio
                qcshd = qcshd * ratio
                ncautc = ncautc * ratio
                ncacc = ncacc * ratio
                nccol = nccol * ratio
                ncheti = ncheti * ratio

            # rain (:3593-3606)
            sinks = (qrevp + qrcol + qrhetc + qrheti + qrmul) * dt
            sources = qr[i, k] + (qrcon + qcaut + qcacc + qimlt + qcshd) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qrevp = qrevp * ratio
                qrcol = qrcol * ratio
                qrheti = qrheti * ratio
                qrmul = qrmul * ratio
                nrevp = nrevp * ratio
                nrcol = nrcol * ratio
                nrheti = nrheti * ratio

            # ice (:3609-3630)
            sinks = (qisub + qimlt) * dt
            sources = qitot[i, k] + (qidep + qinuc + qrcol + qccol + qrhetc
                                     + qrheti + qchetc + qcheti + qrmul) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qisub = qisub * ratio
                qimlt = qimlt * ratio
                nisub = nisub * ratio
                nimlt = nimlt * ratio

            # vapor (:3633-3644)
            sinks = (qccon + qrcon + qcnuc + qidep + qinuc) * dt
            sources = qv[i, k] + (qcevp + qrevp + qisub) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qccon = qccon * ratio
                qrcon = qrcon * ratio
                qcnuc = qcnuc * ratio
                qidep = qidep * ratio
                qinuc = qinuc * ratio
                ninuc = ninuc * ratio
                ncnuc = ncnuc * ratio

            # update_refl_processes (:3652-3743): 3-moment only, not ported.

            # -- update prognostic variables (:3756-3921)
            rimevolume = f32(0.0)
            rimefraction = f32(0.0)
            if qitot[i, k] >= QSMALL:   # iice_loop2 (:3756-3765)
                tmp1 = f32(1.0) / qitot[i, k]
                rimevolume = birim[i, k] * tmp1
                rimefraction = qirim[i, k] * tmp1

            # iice_loop3 (:3767-3865)
            qc[i, k] = qc[i, k] + (-qchetc - qcheti - qccol - qcshd) * dt
            if log_predictNc:                                       # :3770
                nc[i, k] = nc[i, k] + (-nccol - nchetc - ncheti) * dt
            qr[i, k] = qr[i, k] + (-qrcol + qimlt - qrhetc - qrheti + qcshd
                                   - qrmul) * dt
            nr[i, k] = nr[i, k] + (-nrcol - nrhetc - nrheti
                                   + NMLTRATIO * nimlt + nrshdr + ncshdc) * dt

            birim[i, k] = birim[i, k] - (qisub + qimlt) * dt * rimevolume
            qirim[i, k] = qirim[i, k] - (qisub + qimlt) * dt * rimefraction
            qitot[i, k] = qitot[i, k] - (qisub + qimlt) * dt

            dum = (qrcol + qccol + qrhetc + qrheti + qchetc + qcheti
                   + qrmul) * dt
            qitot[i, k] = qitot[i, k] + (qidep + qinuc) * dt + dum
            qirim[i, k] = qirim[i, k] + dum
            birim[i, k] = birim[i, k] + (qrcol * INV_RHO_RIMEMAX
                                         + qccol / rhorime_c
                                         + (qrhetc + qrheti + qchetc + qcheti
                                            + qrmul) * INV_RHO_RIMEMAX) * dt
            nitot[i, k] = nitot[i, k] + (ninuc - nimlt - nisub - nislf
                                         + nrhetc + nrheti + nchetc + ncheti
                                         + nimul) * dt

            if qirim[i, k] < f32(0.0):
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)

            # wet-growth densification (:3835-3838)
            if log_wetgrowth:
                qirim[i, k] = qitot[i, k]
                birim[i, k] = qirim[i, k] * INV_RHO_RIMEMAX

            # melting densification toward solid ice (:3841-3845)
            if (qitot[i, k] >= QSMALL and birim[i, k] >= BSMALL
                    and qimlt > f32(0.0)):
                tmp1 = qirim[i, k] / birim[i, k]
                tmp2 = qitot[i, k] + qimlt * dt
                birim[i, k] = qirim[i, k] / (tmp1 + (f32(917.0) - tmp1)
                                             * qimlt * dt / tmp2)

            qv[i, k] = qv[i, k] + (-qidep + qisub - qinuc) * dt
            th[i, k] = th[i, k] + invexn[i, k] * (
                (qidep - qisub + qinuc) * xxls * INV_CP
                + (qrcol + qccol + qchetc + qcheti + qrhetc + qrheti
                   + qrmul - qimlt) * xlf * INV_CP) * dt

            # warm-phase updates (:3869-3885)
            qc[i, k] = qc[i, k] + (-qcacc - qcaut + qcnuc + qccon - qcevp) * dt
            qr[i, k] = qr[i, k] + (qcacc + qcaut + qrcon - qrevp) * dt
            if log_predictNc:                                       # :3872
                nc[i, k] = nc[i, k] + (-ncacc - ncautc + ncslf + ncnuc) * dt
            else:
                nc[i, k] = NCCNST * inv_rho[k]                       # :3875
            nr[i, k] = nr[i, k] + (ncautr - nrslf - nrevp) * dt  # iparam=3
            qv[i, k] = qv[i, k] + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt
            th[i, k] = th[i, k] + invexn[i, k] * (
                (qcnuc + qccon + qrcon - qcevp - qrevp) * xxlv * INV_CP) * dt

            # clipping (:3889-3918)
            if qc[i, k] < QSMALL:
                qv[i, k] = qv[i, k] + qc[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qc[i, k] * xxlv * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qr[i, k] < QSMALL:
                qv[i, k] = qv[i, k] + qr[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qr[i, k] * xxlv * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qitot[i, k] < QSMALL:
                qv[i, k] = qv[i, k] + qitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xxls * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            qv[i, k] = max(f32(0.0), qv[i, k])
            nitot[i, k] = impose_max_total_Ni(nitot[i, k], inv_rho[k])
            # 555 continue

        # second compute_SCPF (:3968-3970): refresh the qv snapshot.
        qv_cld = qv[i, :].copy()

        if not log_hydrometeorsPresent:
            continue    # goto 333 (:3972)

        # ==================================================================
        # Sedimentation (:3984-4495), adaptive substepping.
        # ==================================================================

        # Cloud mass and, under MP51, droplet number use the same adaptive
        # column flux operator (WRF :4009-4066 / :4068-4119).
        def cloud_speeds(k):
            V_qc = f32(0.0)
            V_nc = f32(0.0)
            if qc[i, k] * iscf > QSMALL:
                nc[i, k], mu_c, nu, lamc, _t1, _t2 = get_cloud_dsd2(
                    qc[i, k], nc[i, k], rho[k], iscf)
                dum = f32(1.0) / _powf(lamc, BCN)
                V_qc = acn[k] * _gammaf(f32(4.0) + BCN + mu_c) \
                    * dum / _gammaf(mu_c + f32(4.0))
                if log_predictNc:
                    V_nc = acn[k] * _gammaf(f32(1.0) + BCN + mu_c) \
                        * dum / _gammaf(mu_c + f32(1.0))
            return (V_qc, V_nc) if log_predictNc else (V_qc,)

        prt_accum = _sediment_column(
            (qc[i], nc[i]) if log_predictNc else (qc[i],), iscf, dt,
            rho, inv_rho, inv_dzq[i], cloud_speeds)
        if prt_accum is not None:
            prt_liq[i] = prt_accum * INV_RHOW * odt

        def rain_speeds(k):
            V_qr = f32(0.0)
            V_nr = f32(0.0)
            if qr[i, k] * ispf > QSMALL:
                nr[i, k] = max(nr[i, k], NSMALL)
                nr[i, k], mu_r, lamr, cdistr, logn0r = get_rain_dsd2(
                    qr[i, k], nr[i, k], ispf)
                dumii_r, dumjj_r, dum1, rdumii, rdumjj, inv_dum3 = \
                    find_lookupTable_indices_3(mu_r, lamr)
                V_qr = _rain_table_value(vm_table, dumii_r, dumjj_r, rdumii, rdumjj)
                V_qr = V_qr * rhofacr[k]
                V_nr = _rain_table_value(vn_table, dumii_r, dumjj_r, rdumii, rdumjj)
                V_nr = V_nr * rhofacr[k]
            return (V_qr, V_nr,)

        prt_accum = _sediment_column(
            (qr[i], nr[i],), ispf, dt,
            rho, inv_rho, inv_dzq[i], rain_speeds)
        if prt_accum is not None:
            prt_liq[i] = prt_liq[i] + prt_accum * INV_RHOW * odt

        def ice_speeds(k):
            V_qit = f32(0.0)
            V_nit = f32(0.0)
            if qitot[i, k] >= QSMALL:
                nitot[i, k] = max(nitot[i, k], NSMALL)
                qirim[i, k], birim[i, k], rhop = calc_bulkRhoRime(
                    qitot[i, k], qirim[i, k], birim[i, k])
                dumi, dumjj, dumii, dum1, dum4, dum5 = \
                    find_lookupTable_indices_1a(
                        qitot[i, k], nitot[i, k], qirim[i, k], rhop)
                f1pr01 = access_lookup_table(
                    itab, dumjj, dumii, dumi, 1, dum1, dum4, dum5)
                f1pr02 = access_lookup_table(
                    itab, dumjj, dumii, dumi, 2, dum1, dum4, dum5)
                f1pr09 = access_lookup_table(
                    itab, dumjj, dumii, dumi, 7, dum1, dum4, dum5)
                f1pr10 = access_lookup_table(
                    itab, dumjj, dumii, dumi, 8, dum1, dum4, dum5)
                nitot[i, k] = min(nitot[i, k], f1pr09 * qitot[i, k])
                nitot[i, k] = max(nitot[i, k], f1pr10 * qitot[i, k])
                V_qit = f1pr02 * rhofaci[k]
                V_nit = f1pr01 * rhofaci[k]
            return (V_qit, V_qit, V_qit, V_nit,)

        prt_accum = _sediment_column(
            (qitot[i], qirim[i], birim[i], nitot[i],), f32(1.0), dt,
            rho, inv_rho, inv_dzq[i], ice_speeds)
        if prt_accum is not None:
            prt_sol[i] = prt_sol[i] + prt_accum * INV_RHOW * odt

        # third compute_SCPF (:4521-4523), quick: off-branch again.
        qv_cld = qv[i, :].copy()

        # ---- homogeneous freezing of cloud and rain (:4528-4610) ----
        for k in range(nk):
            for q, number in ((qc, nc), (qr, nr)):
                if q[i, k] >= QSMALL and t[i, k] < f32(233.15):
                    Q_nuc = q[i, k]
                    number[i, k] = max(number[i, k], NSMALL)
                    N_nuc = number[i, k]
                    qirim[i, k] = qirim[i, k] + Q_nuc
                    qitot[i, k] = qitot[i, k] + Q_nuc
                    birim[i, k] = birim[i, k] + Q_nuc * INV_RHO_RIMEMAX
                    nitot[i, k] = nitot[i, k] + N_nuc
                    th[i, k] = th[i, k] + invexn[i, k] * Q_nuc * xlf * INV_CP
                    q[i, k] = f32(0.0)
                    number[i, k] = f32(0.0)

        # category merge (:4615-4697): nCat > 1 only, not ported.

        # ---- final checks + diagnostics (:4722-4895) ----
        for k in range(nk):
            if qc[i, k] * iscf >= QSMALL:
                nc[i, k], mu_c, nu, lamc, _t1, _t2 = get_cloud_dsd2(
                    qc[i, k], nc[i, k], rho[k], iscf)
                diag_effc[i, k] = f32(0.5) * (mu_c + f32(3.0)) / lamc
            else:
                qv[i, k] = qv[i, k] + qc[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qc[i, k] * xxlv * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)

            if qr[i, k] >= QSMALL:
                nr[i, k], mu_r, lamr, _t1, _t2 = get_rain_dsd2(
                    qr[i, k], nr[i, k], f32(1.0))   # iSPF literal 1. (:4739)
                lam6 = _sixth(lamr)                  # :4756, int **6
                ze_rain[i, k] = rho[k] * nr[i, k] * (mu_r + f32(6.0)) \
                    * (mu_r + f32(5.0)) * (mu_r + f32(4.0)) \
                    * (mu_r + f32(3.0)) * (mu_r + f32(2.0)) \
                    * (mu_r + f32(1.0)) / lam6
                ze_rain[i, k] = max(ze_rain[i, k], f32(1.0e-22))
            else:
                qv[i, k] = qv[i, k] + qr[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qr[i, k] * xxlv * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)

            nitot[i, k] = impose_max_total_Ni(nitot[i, k], inv_rho[k])

            if qitot[i, k] >= QSMALL:
                nitot[i, k] = max(nitot[i, k], NSMALL)
                nr[i, k] = max(nr[i, k], NSMALL)
                qirim[i, k], birim[i, k], rhop = calc_bulkRhoRime(
                    qitot[i, k], qirim[i, k], birim[i, k])
                dumi, dumjj, dumii, dum1, dum4, dum5 = \
                    find_lookupTable_indices_1a(
                        qitot[i, k], nitot[i, k], qirim[i, k], rhop)
                f1pr02 = access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                             dum1, dum4, dum5)
                f1pr06 = access_lookup_table(itab, dumjj, dumii, dumi, 6,
                                             dum1, dum4, dum5)
                f1pr09 = access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                             dum1, dum4, dum5)
                f1pr10 = access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                             dum1, dum4, dum5)
                f1pr13 = access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                             dum1, dum4, dum5)
                f1pr15 = access_lookup_table(itab, dumjj, dumii, dumi, 11,
                                             dum1, dum4, dum5)
                f1pr16 = access_lookup_table(itab, dumjj, dumii, dumi, 12,
                                             dum1, dum4, dum5)
                # f1pr22/f1pr23 (lambda_i/mu_i, table cols 13/14) feed only
                # the absent diag_lami/diag_mui/diag_dhmax outputs (:4860-4865)
                nitot[i, k] = min(nitot[i, k], f1pr09 * qitot[i, k])
                nitot[i, k] = max(nitot[i, k], f1pr10 * qitot[i, k])
                if qirim[i, k] < QSMALL:
                    qirim[i, k] = f32(0.0)
                    birim[i, k] = f32(0.0)
                diag_vmi[i, k] = f1pr02 * rhofaci[k]
                diag_effi[i, k] = f1pr06
                diag_di[i, k] = f1pr15
                diag_rhoi[i, k] = f1pr16
                ze_ice[i, k] = ze_ice[i, k] + f32(0.1892) * f1pr13 \
                    * nitot[i, k] * rho[k]
                ze_ice[i, k] = max(ze_ice[i, k], f32(1.0e-22))
            else:
                qv[i, k] = qv[i, k] + qitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xxls * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)
                diag_di[i, k] = f32(0.0)

            diag_ze[i, k] = f32(10.0) * _log10f(
                (ze_rain[i, k] + ze_ice[i, k]) * f32(1.0e18))

            if qr[i, k] < QSMALL:
                nr[i, k] = f32(0.0)

        # 333 continue: SCF_out (:4941-4952) discarded by the WRF wrapper.

    # save end-of-microphysics theta/qv for the next step (:5018-5021)
    th_old[...] = th
    qv_old[...] = qv


# ---------------------------------------------------------------------------
# WRF wrapper (:690-932) on a flattened column slab.
# ---------------------------------------------------------------------------

def mp_p3_wrapper_wrf(th, qv, qc, qr, nr, qi, qir, ni, qib, th_old, qv_old,
                      pres, dz, dt, itimestep, rainnc, rainncv, sr, snownc,
                      snowncv, refl_10cm, re_cloud, re_ice, vmi3d, di3d,
                      rhopo3d, *, nc_3d: np.ndarray | None = None,
                      runtime: P3Runtime | None = None) -> None:
    """``mp_p3_wrapper_wrf`` for the mp=50 and mp=51 call shapes
    (module_microphysics_driver.F:1569-1602 passes no ``nc_3d``; :1619-1647
    passes ``nc_3d = qnc_curr``; neither passes ``qzi1_3d``; ``n_iceCat =
    1``).  ``log_predictNc`` is ``nc_3d is not None``, the wrapper's own
    ``present(nc_3d)`` (:819-820); a prognostic ``nc_3d`` is updated in
    place (:844, :901-903).

    All arrays are float32; 3-D fields are (ni, nk) column slabs with k=0
    at the surface (the WRF j-loop over independent (i, k) slabs and a
    single flattened slab are the same computation -- columns never
    interact inside p3_main).  Precipitation fields are (ni,) and are
    updated with WRF's mm conversions (:892-898).  ``pii`` (Exner) is a
    wrapper argument WRF marks "currently not used" (:725) and is omitted;
    ``w`` likewise feeds only the unused ``uzpl`` (:2204).
    """
    ni_pts, nk = th.shape
    log_predictNc = nc_3d is not None                  # :819-820
    if log_predictNc:
        nc = nc_3d                                     # :844, updated in place
    else:
        nc = np.zeros((ni_pts, nk), dtype=np.float32)  # :847 (specified Nc)
    ssat = np.zeros((ni_pts, nk), dtype=np.float32)   # :851
    prt_liq = np.zeros(ni_pts, dtype=np.float32)
    prt_sol = np.zeros(ni_pts, dtype=np.float32)

    p3_main(qc, nc, qr, nr, th_old, th, qv_old, qv, dt, qi, qir, ni, qib,
            ssat, pres, dz, itimestep, prt_liq, prt_sol,
            refl_10cm, re_cloud, re_ice, vmi3d, di3d, rhopo3d,
            n_cat=1, log_predictNc=log_predictNc, model="WRF",
            clbfact_dep=1.0, clbfact_sub=1.0, runtime=runtime)

    dum1 = f32(1000.0) * f32(dt)
    total = prt_liq + prt_sol
    rainnc += total * dum1
    rainncv[:] = total * dum1
    snownc += prt_sol * dum1
    snowncv[:] = prt_sol * dum1
    sr[:] = prt_sol / (prt_liq + prt_sol + f32(1.0e-12))


# ---------------------------------------------------------------------------
# DomainState adapter (woof dispatch surface).
# ---------------------------------------------------------------------------

def _host(arr) -> np.ndarray:
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _to_slab(field3d) -> np.ndarray:
    """(nz, ny, nx) -> contiguous (ny*nx, nz) float32 host slab."""
    host = _host(field3d)
    nz = host.shape[0]
    return np.ascontiguousarray(host.reshape(nz, -1).T)


def _as_backend(dev, host):
    if hasattr(dev, "get"):
        import cupy as cp
        return cp.asarray(host)
    return host


def _from_slab(dev, slab) -> None:
    """Write an (ncol, nz) slab back into the (nz, ny, nx) device field."""
    host = np.ascontiguousarray(slab.T).reshape(dev.shape)
    dev[...] = _as_backend(dev, host)


def _p3_required_fields(state, cfg) -> bool:
    """Refuse a state that lacks the variant's fields; return predict_nc.

    The breakage this refusal prevents is concrete: a mp=51 state without
    its prognostic ``nc`` would otherwise run the specified-Nc branch
    (droplet number reset to nccnst/rho every step) under an mp=51 label,
    and every droplet-number output would be the wrong scheme's.
    """
    from woof.config import p3_predicts_nc

    predict_nc = p3_predicts_nc(cfg.mp_physics)
    required = ("qi", "nr", "ni", "qir", "qib", "th_old", "qv_old") + (
        ("nc",) if predict_nc else ())
    from woof.config import p3_variant
    variant = p3_variant(cfg.mp_physics)
    if variant is not None:
        required = ("qi", "nr", "ni", "qir", "qib", "th_old", "qv_old") + variant.extra_species
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(f"mp_physics={cfg.mp_physics} state lacks P3 "
                         "fields: " + ", ".join(missing))
    return predict_nc


def apply_reference(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """THE REFERENCE/DEBUG PATH: prepare WRF's inputs and run the CPU
    float32 transcription on ``state`` in place.

    This is not the production path -- :func:`apply` dispatches to the
    device kernels (``woof/core/p3_device.py``).  It is kept, and kept
    reachable through ``run.p3_backend = "reference"``, because it is
    what every later device optimisation is checkable against and it is
    what runs where there is no CuPy at all.

    The CPU float32 authority runs on host copies of the device state and
    the results are written back; see the module docstring for the
    execution-model caveat.  The WRF prep contract matches the other
    adapters (full theta, layer depths from the full geopotential) and the
    theta update goes through the ``moist_physics_prep/finish`` bracket.
    """
    if cfg.mp_physics == 52:
        raise ValueError("P3 two-category integration needs the CUDA backend; "
                         "the CPU reference carries one category and would "
                         "omit ice-ice collection, category merge and category-2 sedimentation")
    from woof.core import constants as c
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta,
                                         prepare_microphysics_exner)
    from woof.core.state import DTYPE

    predict_nc = _p3_required_fields(state, cfg)
    runtime = p3_init()   # loud table refusal happens here, before any prep

    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]

    th_dev = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    dz_dev = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th_dev[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
               else thb + state.thp)
    prepare_microphysics_exner(state.p, pii)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz_dev[...] = z8w[1:] - z8w[:-1]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    sr = state.scratch(surface, "mp_sr")
    vmi3d = state.scratch((nz, ny, nx), "p3_vmi")
    di3d = state.scratch((nz, ny, nx), "p3_di")
    rhopo3d = state.scratch((nz, ny, nx), "p3_rhopo")

    # WRF's itimestep starts at 1 (:1980 note); the counter gates only the
    # first-step saturation adjustment (:3348) and the it>1 activation
    # guard (:3303) -- nothing reads its magnitude, only ``it <= 1``.
    #
    # WRF restart-carries itimestep, and woof honours that WITHOUT putting
    # a scalar in the restart stream: the only thing a resume needs to know
    # is whether this is the model's first step, and the model clock
    # already says so and is already serialized.  Seeding from
    # ``elapsed_seconds`` is therefore equivalent to carrying the counter,
    # and it is what stops a resumed trajectory from replaying the it=1
    # saturation adjustment in the middle of itself.  The attribute is
    # classified ``infra`` in woof/io/restart.py for exactly this reason:
    # it is reconstructed from the clock, not dropped.
    prior = getattr(state, "p3_itimestep", None)
    if prior is None:
        prior = 1 if float(getattr(state, "elapsed_seconds", 0.0)) > 0.0 else 0
    it = int(prior) + 1
    state.p3_itimestep = it

    slabs = {}
    fields = {"th": th_dev, "qv": state.qv, "qc": state.qc, "qr": state.qr,
              "nr": state.nr, "qi": state.qi, "qir": state.qir,
              "ni": state.ni, "qib": state.qib, "th_old": state.th_old,
              "qv_old": state.qv_old, "pres": state.p, "dz": dz_dev}
    if predict_nc:
        fields["nc"] = state.nc          # nc_3d = qnc_curr (mp = 51)
    for name, dev in fields.items():
        slabs[name] = _to_slab(dev)

    ncol = ny * nx
    host_surface = {name: _host(arr).reshape(ncol).astype(np.float32,
                                                          copy=True)
                    for name, arr in (("rainnc", rainnc),
                                      ("rainncv", rainncv),
                                      ("sr", sr), ("snownc", snownc),
                                      ("snowncv", snowncv))}
    refl = np.empty((ncol, nz), dtype=np.float32)
    re_cloud = np.empty((ncol, nz), dtype=np.float32)
    re_ice = np.empty((ncol, nz), dtype=np.float32)
    vmi_slab = np.empty((ncol, nz), dtype=np.float32)
    di_slab = np.empty((ncol, nz), dtype=np.float32)
    rhopo_slab = np.empty((ncol, nz), dtype=np.float32)

    save_pre_mp_theta(state)   # WRF moist_physics_prep_em
    mp_p3_wrapper_wrf(
        slabs["th"], slabs["qv"], slabs["qc"], slabs["qr"], slabs["nr"],
        slabs["qi"], slabs["qir"], slabs["ni"], slabs["qib"],
        slabs["th_old"], slabs["qv_old"], slabs["pres"], slabs["dz"],
        float(dt), it,
        host_surface["rainnc"], host_surface["rainncv"], host_surface["sr"],
        host_surface["snownc"], host_surface["snowncv"],
        refl, re_cloud, re_ice, vmi_slab, di_slab, rhopo_slab,
        nc_3d=slabs["nc"] if predict_nc else None, runtime=runtime)

    for name in ("qv", "qc", "qr", "nr", "qi", "qir", "ni", "qib",
                 "th_old", "qv_old") + (("nc",) if predict_nc else ()):
        _from_slab(fields[name], slabs[name])
    _from_slab(th_dev, slabs["th"])
    for name, dev in (("rainnc", rainnc), ("rainncv", rainncv), ("sr", sr),
                      ("snownc", snownc), ("snowncv", snowncv)):
        dev[...] = _as_backend(dev, host_surface[name].reshape(ny, nx))
    # effective radii: P3 returns metres; the state carries the
    # radiation-facing MICRON convention (state.py mp arms).
    state.effc[...] = _as_backend(
        state.effc, np.ascontiguousarray(
            (re_cloud * np.float32(1.0e6)).T).reshape(nz, ny, nx))
    state.effi[...] = _as_backend(
        state.effi, np.ascontiguousarray(
            (re_ice * np.float32(1.0e6)).T).reshape(nz, ny, nx))
    _from_slab(vmi3d, vmi_slab)
    _from_slab(di3d, di_slab)
    _from_slab(rhopo3d, rhopo_slab)

    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl_dev = state.scratch((nz, ny, nx), "refl_10cm")
        _from_slab(refl_dev, refl)
        stash_refl_10cm(state, refl_dev)

    moist_physics_finish(state, cfg, th_dev, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv)


# ---------------------------------------------------------------------------
# The production adapter: device-resident P3.
# ---------------------------------------------------------------------------

def _p3_backend(state, cfg) -> str:
    """Which arm this call takes, and why.

    ``run.p3_backend`` selects; ``cuda`` is the default and is what "fixed"
    means here.  The reference path is taken WITHOUT being asked when the
    state is not on a CuPy backend at all -- a host-array state has no
    device to launch on, and refusing there would turn every CPU-side unit
    test of the scheme into a GPU test.
    """
    want = str(getattr(cfg, "p3_backend", "cuda"))
    if want == "reference":
        return "reference"
    if not hasattr(state.qv, "get"):        # host arrays: no device to use
        return "reference"
    return want


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Run P3 on ``state`` in place, on the card.

    Prognostic state never leaves the device.  The lookup tables are
    uploaded once per process and stay resident.  The per-step host work is
    the kernel launches and nothing else: no slab transpose, no host copy
    of any prognostic field, no Python loop over columns.

    ``run.p3_backend = "reference"`` selects the CPU float32 transcription
    (:func:`apply_reference`) instead, and a state whose arrays are not on
    a CuPy backend takes that path automatically.
    """
    from woof.config import p3_variant

    variant = p3_variant(cfg.mp_physics)
    if variant is None:
        raise ValueError(
            f"woof.core.p3.apply was called for mp_physics={cfg.mp_physics}, "
            "which is not a P3 package (woof.config.P3_VARIANTS)")
    from woof.core.p3_tables import P3_CATEGORY2_FIELDS
    n_cat = variant.n_cat
    backend = _p3_backend(state, cfg)
    if backend == "reference":
        if variant.three_moment_ice or variant.n_cat > 1:
            # The CPU transcription carries mp=50's call shape only; a
            # host-array state takes this branch without asking, so the
            # refusal has to live here as well as in validate_run_config.
            raise ValueError(
                f"mp_physics={cfg.mp_physics} ({variant.label}) runs on the "
                "card only: the CPU reference transcription "
                "(woof/core/p3.py p3_main) carries two-moment ice with "
                "specified droplet number, so it would integrate a different "
                "scheme and leave qzi/nc untouched.  This state's arrays are "
                "not on a CuPy backend.")
        return apply_reference(state, cfg, dt, refl_10cm_due=refl_10cm_due)

    import cupy as cp

    if cfg.mp_physics == 52 and not hasattr(state.qv, "get"):
        raise ValueError("P3 two-category integration needs the CUDA backend; "
                         "the CPU reference carries one category and would "
                         "omit ice-ice collection, category merge and category-2 sedimentation")
    from woof.core import constants as c
    from woof.core import p3_device as PD
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    from woof.core.state import DTYPE

    log_predict_nc = _p3_required_fields(state, cfg)
    tables = PD.device_tables(n_cat=n_cat, three_moment=variant.three_moment_ice)

    nz, ny, nx = state.p.shape
    ncol = ny * nx
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]

    th_dev = state.scratch((nz, ny, nx), "mp_th")
    dz_dev = state.scratch((nz, ny, nx), "mp_dz8w")
    fused_adapter = variant.three_moment_ice or n_cat > 1
    if not fused_adapter:
        z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
        th_dev[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
                   else thb + state.thp)
        z8w[...] = (phb + state.php) / DTYPE(c.G)
        dz_dev[...] = z8w[1:] - z8w[:-1]
    # ``mp_pii`` is NOT allocated on this path.  The device kernels build
    # the Exner factor themselves from pres, exactly where the authority
    # builds it (:2293), so a separate full-domain array would be a second
    # copy of a value the kernel already holds in a register.

    surface = (ny, nx)
    diag_slots = {"zdbz": "refl_10cm", "effc": "p3_effc", "effi": "p3_effi",
                  "vmi": "p3_vmi", "di": "p3_di", "rhopo": "p3_rhopo"}
    if n_cat > 1:
        diag_slots.update(vmi2="p3_vmi2", di2="p3_di2", rhopo2="p3_rhopo2")
    diag = {name: state.scratch((nz, ny, nx), slot)
            for name, slot in diag_slots.items()}
    surf_slots = {"prt_liq": "p3_prt_liq", "prt_sol": "p3_prt_sol",
                  "rainnc": "mp_rainnc", "rainncv": "mp_rainncv",
                  "sr": "mp_sr", "snownc": "mp_snownc",
                  "snowncv": "mp_snowncv"}
    surf = {name: state.scratch(surface, slot)
            for name, slot in surf_slots.items()}

    # SEAM 1: ssat is a real (nz, ny, nx) device array threaded through
    # every kernel that the authority threads it through.  It is zero on
    # entry (the wrapper's :851) and p3_main DIAGNOSES it; the seam is that
    # it is storage and an argument, not a constant folded away because
    # log_predictSsat is false today.
    ssat = state.scratch((nz, ny, nx), "p3_ssat")
    if fused_adapter:
        from woof.core.p3_adapter import prepare
        prepare(state, th_dev, dz_dev, ssat)
    else:
        ssat[...] = 0.0
    # SEAM 3: log_predictNc is the authority's `present(nc_3d)` (:819-820)
    # and log_3momentIce its `present(qzi1_3d)` (:822-823); WRF's driver
    # decides both by package, and so does the P3_VARIANTS row here.  A
    # specified-Nc package gets the wrapper's zeroed nc array (:847) as
    # scratch; a predicted-Nc package passes its own prognostic field.
    log_predict_nc = bool(variant.predict_nc)
    log_3moment_ice = bool(variant.three_moment_ice)
    nc_dev = state.nc if log_predict_nc else state.scratch(
        (nz, ny, nx), "p3_nc")

    fields = {"qc": state.qc, "nc": nc_dev, "qr": state.qr, "nr": state.nr,
              "qi": state.qi, "qir": state.qir, "ni": state.ni,
              "qib": state.qib, "th": th_dev, "qv": state.qv,
              "th_old": state.th_old, "qv_old": state.qv_old,
              "ssat": ssat, "pres": state.p, "dz": dz_dev}
    if log_3moment_ice:
        # The ADVECTED (N*Z)**0.5 (Registry QZI); the kernels convert it to
        # Z and back where the wrapper does (:826-838, :912-924).
        fields["qzi"] = state.qzi

    if n_cat > 1:
        fields.update({name: getattr(state, name) for name in P3_CATEGORY2_FIELDS})

    # See woof/core/p3.py:apply_reference for why the clock, not a
    # serialized counter, is what makes ``it`` survive a restart.
    prior = getattr(state, "p3_itimestep", None)
    if prior is None:
        prior = 1 if float(getattr(state, "elapsed_seconds", 0.0)) > 0.0 else 0
    it = int(prior) + 1
    state.p3_itimestep = it

    ws = getattr(state, "_p3_workspace", None)
    if (ws is None or ws.ncol != ncol or ws.nk != nz
            or ws.three_moment != log_3moment_ice):
        # Every P3 companion comes from DomainState.scratch, so the
        # allocation gate sees all of it and a normal timestep allocates
        # nothing.  woof/core/preflight.py prices these slots for mp=50.
        ws = PD.make_workspace(
            ncol, nz,
            allocate=lambda slot, nlev: state.scratch((nlev, ny, nx), slot),
            three_moment=log_3moment_ice)
        state._p3_workspace = ws

    if not fused_adapter:
        save_pre_mp_theta(state)      # WRF moist_physics_prep_em
    PD.run_p3_device(
        {k: v.reshape(nz, ncol) for k, v in fields.items()},
        {k: v.reshape(nz, ncol) for k, v in diag.items()},
        {k: v.reshape(ncol) for k, v in surf.items()},
        workspace=ws, tables=tables, dt=float(dt), it=it,
        log_predictNc=log_predict_nc, log_3momentIce=log_3moment_ice, n_cat=n_cat,
        arm=PD.CONFIG_ARM[str(getattr(cfg, "p3_backend", "cuda"))])

    # P3 returns effective radii in METRES; the state carries the
    # radiation-facing MICRON convention (woof/core/state.py mp arms).
    if fused_adapter:
        from woof.core.p3_adapter import radii
        radii(diag, state)
    else:
        state.effc[...] = diag["effc"] * cp.float32(1.0e6)
        state.effi[...] = diag["effi"] * cp.float32(1.0e6)

    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        stash_refl_10cm(state, diag["zdbz"])

    moist_physics_finish(state, cfg, th_dev, dt)
    return MicrophysicsDiagnostics(
        rainnc=surf["rainnc"], rainncv=surf["rainncv"], sr=surf["sr"],
        snownc=surf["snownc"], snowncv=surf["snowncv"])
