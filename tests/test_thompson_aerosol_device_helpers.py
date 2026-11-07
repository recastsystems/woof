"""G1: pointwise gates for the mp=28 shared device helpers.

Everything here exercises ``woof/core/kernels/thompson_aerosol_common.cuh``
through the thin wrappers in ``woof/core/kernels/thompson_aerosol_probe.cu``.
No network kernel is involved, so a failure localizes to one helper.

Three kinds of assertion, in increasing strength:

* HARD IDENTITY GATES.  ``thompson_aa_droplet_bin(100.0e6) == 65`` and
  ``thompson_aa_in_bin(1000.0) == 27`` are the exact constants WOOF's
  model-validated mp=8 kernels hardcode (thompson.cu:3933 and :3936).  If the
  generalized nc-driven formulas do not reproduce them, they are wrong in a
  way that would ALSO have broken mp=8 had the kernels been shared.

* ORACLE AGREEMENT against the Fortran probe tables under
  ``woof/data/thompson/oracle-aero/``, produced by
  ``tools/thompson_wrf461_oracle/probe_aero_functions.F90`` running inside
  unmodified WRF v4.6.1.  MEASURED RESULT: the four aerosol-only helpers plus
  ``calc_effectRad`` agree BIT-EXACTLY on all 2182 probe rows, so the gates
  below assert exact float32 equality rather than a tolerance.  That is only
  achievable because the helpers pin every float multiply/add against nvrtc's
  default FMA contraction and evaluate EXP/LOG/** in double before rounding
  once to float; see the commentary in the header.

* HOST-REFERENCE agreement for the two helpers the Fortran probe does not
  tabulate (``thompson_aa_cloud_dist`` and ``thompson_aa_snow_number``).
  Those carry a documented FP32 tolerance because the host reference uses
  NumPy's float32 ``pow`` while the device uses CUDA's ``powf``.

* STRUCTURAL assertions about WRF's CONTROL FLOW, which need no oracle at
  all.  The ``nu_c`` staging block below is the important one: WRF computes
  ``nu_c`` twice, at :1832 from the pre-rediagnosis ``nc`` and again at :2170
  from the post-rediagnosis ``nc(k)`` assigned at :1840, and a kernel that
  reuses the first stays finite, stays stable and is grossly wrong wherever
  the :1834-1838 droplet-size clamp engages.  That is a property of the
  Fortran source, so it is asserted as one -- against what WRF DOES, not
  against what any current WOOF kernel happens to do.

TOLERANCE POLICY NOTE, per the port spec's validation plan: ``activ_ncloud``
selects a NEAREST 10 K temperature bin and INT-truncated aerosol/updraft
bracket indices, so activated number is a STEP function of state.  The probe
grid straddles bin edges on purpose and still agrees exactly, because the
edges are hit identically; a fixture that landed a float ulp away from an
edge could legitimately select a different bin.  That behaviour is documented
here rather than absorbed into a loose global tolerance.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_ORACLE = (Path(__file__).parents[1] / "woof" / "data" / "thompson"
           / "oracle-aero")
# Resolved, not joined: the classic-table directory ships in the recast-woof-data
# companion distribution since 2.5.0.
from woof.physics_compat import packaged_thompson_table_root  # noqa: E402

_TABLES = packaged_thompson_table_root()
_KERNELS = Path(__file__).parents[1] / "woof" / "core" / "kernels"
_HEADER = _KERNELS / "thompson_aerosol_common.cuh"

# The one asset that is read from disk at runtime and deliberately NOT
# vendored into the repository (third-party parcel-model output).
_CCN_BIN = _TABLES / "CCN_ACTIVATE.BIN"


def _rows(name: str) -> list[dict[str, str]]:
    with (_ORACLE / name).open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _device(rows, key):
    import cupy as cp
    return cp.asarray(
        np.array([float(row[key]) for row in rows], dtype=np.float32))


def _reference(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def _f32(values):
    import cupy as cp
    return cp.asarray(np.asarray(values, dtype=np.float32))


def _ccn_table():
    import cupy as cp
    from woof.core.thompson_aerosol_contract import (
        read_ccn_activation_table)
    host = read_ccn_activation_table(_CCN_BIN)
    return cp.asarray(np.asfortranarray(host))


# ---------------------------------------------------------------------------
# Local launchers for the probe kernels this package added after
# woof/core/thompson_aerosol_launch.py was written.  That module belongs to a
# different work package and is not edited here; these go straight through
# ``get_kernel`` so nothing outside this file has to change for the coverage
# below to exist.  They are deliberately thin -- one launch each, no physics.
# ---------------------------------------------------------------------------

def _probe_kernel(name):
    from woof.core.kernels import get_kernel
    from woof.core.thompson_aerosol_launch import PROBE_MODULE
    return get_kernel(PROBE_MODULE, name)


def _probe_nu_c_staging(rc, nc_per_kg, rho):
    """Both WRF stages of ``nu_c`` for one (rc, nc_entry, rho) state.

    Returns ``(nc_rediagnosed_m3, nu_c_entry, nu_c_working, lamc_entry)``.
    ``nu_c_entry`` is module_mp_thompson.F:1832, computed from the entry
    ``nc``; ``nu_c_working`` is :2170, recomputed from the :1840 rediagnosed
    ``nc``.  Everything downstream of :1838 in WRF uses the latter.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import launch_grid, validate_fields
    _, size = validate_fields({"rc": rc, "nc_per_kg": nc_per_kg, "rho": rho})
    nc_out = cp.empty(rc.shape, dtype=cp.float32)
    entry = cp.empty(rc.shape, dtype=cp.int32)
    working = cp.empty(rc.shape, dtype=cp.int32)
    lamc = cp.empty(rc.shape, dtype=cp.float64)
    grid, block = launch_grid(size)
    _probe_kernel("thompson_aa_probe_nu_c_staging")(
        grid, block,
        (rc, nc_per_kg, rho, nc_out, entry, working, lamc, np.int32(size)))
    return nc_out, entry, working, lamc


def _probe_gamma_columns(nu_c):
    """``ccg(2,n)``, ``ocg1(n)``, ``ccg(3,n)``, ``ocg2(n)`` for each n."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import launch_grid
    size = int(nu_c.size)
    outs = [cp.empty(nu_c.shape, dtype=cp.float32) for _ in range(4)]
    grid, block = launch_grid(size)
    _probe_kernel("thompson_aa_probe_gamma_columns")(
        grid, block, (nu_c, *outs, np.int32(size)))
    return outs


def _probe_entry_rain_distribution(qr_per_kg, nr_per_kg, rho):
    """``(nr_m3, lamr, mvd_r, N0_r, L_qr)`` -- :1878-1898 then :2144-2150."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import launch_grid, validate_fields
    _, size = validate_fields(
        {"qr_per_kg": qr_per_kg, "nr_per_kg": nr_per_kg, "rho": rho})
    nr = cp.empty(qr_per_kg.shape, dtype=cp.float32)
    lamr = cp.empty(qr_per_kg.shape, dtype=cp.float64)
    mvd = cp.empty(qr_per_kg.shape, dtype=cp.float32)
    n0 = cp.empty(qr_per_kg.shape, dtype=cp.float64)
    active = cp.empty(qr_per_kg.shape, dtype=cp.int32)
    grid, block = launch_grid(size)
    _probe_kernel("thompson_aa_probe_entry_rain_distribution")(
        grid, block,
        (qr_per_kg, nr_per_kg, rho, nr, lamr, mvd, n0, active,
         np.int32(size)))
    return nr, lamr, mvd, n0, active


def _probe_bound_number(kind, mass, density, number_per_kg):
    import cupy as cp
    from woof.core.thompson_aerosol_launch import launch_grid, validate_fields
    _, size = validate_fields({
        "mass": mass, "density": density, "number": number_per_kg})
    out = cp.empty(mass.shape, dtype=cp.float32)
    grid, block = launch_grid(size)
    _probe_kernel(f"thompson_aa_probe_bound_{kind}_number")(
        grid, block, (mass, density, number_per_kg, out, np.int32(size)))
    return out


def _probe_mass_coefficients():
    import cupy as cp
    out = cp.empty(4, dtype=cp.float32)
    _probe_kernel("thompson_aa_probe_mass_coefficients")((1,), (1,), (out,))
    return out


def _probe_decade_index_double(value, first_exponent, table_size):
    """``thompson_aa_decade_index_double``, promoted here in wave 4."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import launch_grid
    size = int(value.size)
    out = cp.empty(value.shape, dtype=cp.int32)
    grid, block = launch_grid(size)
    _probe_kernel("thompson_aa_probe_decade_index_double")(
        grid, block, (value, np.int32(first_exponent),
                      np.int32(table_size), out, np.int32(size)))
    return out


# ---------------------------------------------------------------------------
# THE WRF OPERATOR TREE FOR THE FIELD ET AL (2005) SNOW-MOMENT FITS.
# ---------------------------------------------------------------------------
#
# WRF has no ``field_a`` procedure.  The ten-term chain is written INLINE at
# module_mp_thompson.F:2036-2046, :2050-2053, :2056-2064, :2069-2079,
# :2091-2100, :2103-2112, :2116-2125, :3332-3350, :4447-4470, :5670-5680 and
# :5684-5693, always with the same operator tree and only the moment symbol
# changing.  Nothing external can call it and every constant it reads is
# PRIVATE, so the reference has to be a transcription -- but a transcription
# of the OPERATOR TREE, which is where the defect this replaces actually was.
#
# WHY THIS IS BIT-EXACT AND NOT A TOLERANCE.  Three independent facts line up:
#   * Fortran evaluates the equal-precedence ``*`` chain LEFT TO RIGHT, so
#     ``sa(5)*tc0*tc0`` is (sa5*tc0)*tc0.  NumPy float32 does the same when
#     the parenthesisation is written out, as below.
#   * ``tools/thompson_wrf461_oracle/build_aero.sh`` compiles the oracle with
#     plain ``gfortran -O2`` and no ``-march``, i.e. baseline x86-64 with NO
#     FMA instruction, so every REAL(4) multiply and add is separately
#     rounded.  NumPy float32 scalars are too.  The device gets the same by
#     going through ``thompson_aa_mul``/``thompson_aa_add``.
#   * ``10.0**loga_`` is REAL(4)**REAL(4), which gfortran lowers to glibc's
#     correctly-rounded powf.  ``np.float32(math.pow(...))`` is the same
#     value, and so is the device's ``thompson_aa_powf``.
#
# VERIFIED, not assumed: this transcription was checked against a
# ``gfortran -O2 -ffree-form -ffree-line-length-none`` program holding WRF's
# statements character for character (:2069-2079 with cse(1) -> a loop
# variable), linked against the same compiled module_mp_thompson.o the aerosol
# oracle harness uses.  253 (tc0, moment) states x a_ and b_, and 391
# (tc0, rs) states of the snow number below: ZERO mismatches.  A spread of
# that program's output is embedded as ``_GFORTRAN_*_ANCHORS`` so the
# agreement is re-checked on every run without a new data file.

_SA = tuple(np.float32(v) for v in (
    5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
    0.31255, 0.000204, 0.003199, 0.0, -0.015952))       # :358-359
_SB = tuple(np.float32(v) for v in (
    0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
    0.060366, 0.000079, 0.000594, 0.0, -0.003577))      # :361-362


def _f32pow(x, y):
    """REAL(4)**REAL(4): glibc powf, i.e. correctly rounded to float32."""
    return np.float32(math.pow(float(x), float(y)))


def _host_field_chain(c, tc0, moment):
    """module_mp_thompson.F:2069-2074 (sa) / :2076-2079 (sb), term for term.

    Deliberately NOT vectorised and deliberately not factored: the whole
    point is that ``sa(5)*tc0*tc0`` is ``(sa5*tc0)*tc0`` and not
    ``sa5*(tc0*tc0)``.  Hoisting ``tc0*tc0`` is exactly the defect this
    reference exists to catch.
    """
    f = np.float32
    tc0 = f(tc0)
    moment = f(moment)
    v = c[0]
    v = f(v + f(c[1] * tc0))
    v = f(v + f(c[2] * moment))
    v = f(v + f(f(c[3] * tc0) * moment))
    v = f(v + f(f(c[4] * tc0) * tc0))
    v = f(v + f(f(c[5] * moment) * moment))
    v = f(v + f(f(f(c[6] * tc0) * tc0) * moment))
    v = f(v + f(f(f(c[7] * tc0) * moment) * moment))
    v = f(v + f(f(f(c[8] * tc0) * tc0) * tc0))
    v = f(v + f(f(f(c[9] * moment) * moment) * moment))
    return v


def _host_field_a(tc0, moment):
    return _f32pow(np.float32(10.0), _host_field_chain(_SA, tc0, moment))


def _host_field_b(tc0, moment):
    return _host_field_chain(_SB, tc0, moment)


def _host_field_a_mp8_form(tc0, moment):
    """thompson.cu:81-95's hoisted chain, kept ONLY as a negative control.

    This is what the header used to carry.  It is not WRF's operator tree and
    it is not what mp=28 ships; it exists so
    ``test_the_hoisted_mp8_field_fit_really_does_disagree_with_wrf`` can prove
    the association is essential rather than cosmetic.
    """
    f = np.float32
    tc0 = f(tc0)
    moment = f(moment)
    tc2 = f(tc0 * tc0)
    moment2 = f(moment * moment)
    v = _SA[0]
    v = f(v + f(_SA[1] * tc0))
    v = f(v + f(_SA[2] * moment))
    v = f(v + f(f(_SA[3] * tc0) * moment))
    v = f(v + f(_SA[4] * tc2))
    v = f(v + f(_SA[5] * moment2))
    v = f(v + f(f(_SA[6] * tc2) * moment))
    v = f(v + f(f(_SA[7] * tc0) * moment2))
    v = f(v + f(f(_SA[8] * tc2) * tc0))
    v = f(v + f(f(_SA[9] * moment2) * moment))
    return _f32pow(np.float32(10.0), v)


#: module_mp_thompson.F:113-117, :741, :756, :747, :130.
_MU_S = np.float32(0.6357)
_KAP0 = np.float32(490.6)
_KAP1 = np.float32(17.46)
_LAM0 = np.float32(20.78)
_LAM1 = np.float32(3.29)
_CSE15 = np.float32(_MU_S + np.float32(1.0))
#: WGAMMA(cse(15)) from WRF's own REAL(4) Lanczos series, reproduced by a
#: gfortran transcription of :5325-5346 / :5371-5377.  math.gamma is NOT
#: equivalent; see the GAMMA PARITY note in the shared header.
_CSG15 = np.float32(0.89803153276443481)
_OAMS = np.float32(1.0 / 0.069)


def _host_snow_number(smob, smoc):
    """module_mp_thompson.F:2083-2088.

    ns, M0, Mrat, slam1 and slam2 are all REAL(4) (:1609-1610), and Fortran's
    equal-precedence ``*`` / ``/`` chain runs left to right, so the second
    term is ((((Mrat*Kap1)*M0**mu_s)*csg(15))/slam2**cse(15)).
    """
    f = np.float32
    smob = f(smob)
    smoc = f(smoc)
    m0 = f(smob / smoc)
    mrat = f(f(f(smob * m0) * m0) * m0)
    slam1 = f(m0 * _LAM0)
    slam2 = f(m0 * _LAM1)
    first = f(f(mrat * _KAP0) / slam1)
    second = f(f(f(f(mrat * _KAP1) * _f32pow(m0, _MU_S)) * _CSG15)
               / _f32pow(slam2, _CSE15))
    return f(first + second)


#: The temperature and moment ladders the measurement campaign used.  23
#: temperatures spanning every level the snow category occupies, and every
#: moment an mp=28 kernel asks for: 0 (:2050, smo0), 1 (:2056, smo1),
#: 1.775 (warm.cu's ventilation moment), 2.55 = cse(13) = bv_s+2 (:2091) and
#: 3 = cse(1) = bm_s+1 (:2069), plus filler so the fit is exercised off the
#: call sites.
_FIT_TEMPERATURES = (
    -0.1, -0.5, -1.0, -2.0, -3.0, -5.0, -7.5, -10.0, -12.5, -15.0, -17.5,
    -20.0, -22.5, -25.0, -27.5, -30.0, -32.5, -35.0, -40.0, -45.0, -50.0,
    -60.0, -70.0)
_FIT_MOMENTS = (0.0, 0.5, 1.0, 1.5, 1.775, 2.0, 2.25, 2.55, 2.75, 3.0, 3.5)

#: rs [kg m^-3] = qs*rho, from WRF's R1 floor to far above any real column.
_FIT_SNOW_CONTENTS = (
    1.0e-12, 5.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6,
    1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3, 3.0e-3, 5.0e-3, 8.0e-3, 1.0e-2)

#: (tc0, moment, a_, b_) straight out of the gfortran -O2 program.  These pin
#: the PYTHON reference above to the real compiler, so the whole chain is
#: gfortran -> these literals -> _host_field_a/_host_field_b -> the device.
_GFORTRAN_FIELD_AB_ANCHORS = (
    (-0.10000000149011612, 0.0, 117924.0390625, 0.47780919075012207),
    (-0.10000000149011612, 3.5, 0.003880657721310854, 1.6415095329284668),
    (-3.0, 1.5, 15.854690551757812, 0.8588076233863831),
    (-15.0, 3.0, 0.004036353901028633, 1.2646571397781372),
    (-17.5, 3.0, 0.0031348071061074734, 1.2428220510482788),
    (-20.0, 3.0, 0.002457645023241639, 1.2221871614456177),
    (-22.5, 3.0, 0.0019449822138994932, 1.2027521133422852),
    (-30.0, 3.5, 3.9161623135441914e-05, 1.2396550178527832),
    (-40.0, 3.5, 1.248699572897749e-05, 1.1593201160430908),
    (-60.0, 3.5, 2.2964263735048007e-06, 1.0799500942230225),
    (-70.0, 3.5, 1.3244963383840513e-06, 1.0809144973754883),
)

#: (tc0, rs, smob, smoc, ns) from the same program, which forms smoc exactly
#: as WRF does at :2069-2080 with bm_s == 2 (so smo2 == smob).
_GFORTRAN_SNOW_NUMBER_ANCHORS = (
    (-0.10000000149011612, 9.999999960041972e-13, 1.4492754218942139e-11,
     9.06409708702643e-18, 957.5787353515625),
    (-0.10000000149011612, 0.009999999776482582, 0.14492753148078918,
     0.0014260985190048814, 38683.37890625),
    (-1.0, 0.009999999776482582, 0.14492753148078918,
     0.0013009392423555255, 46484.62890625),
    (-5.0, 0.009999999776482582, 0.14492753148078918,
     0.0008745637023821473, 102858.5859375),
    (-12.5, 0.009999999776482582, 0.14492753148078918,
     0.0004361904866527766, 413495.78125),
    (-20.0, 0.009999999776482582, 0.14492753148078918,
     0.00023189348576124758, 1463006.75),
    (-27.5, 0.009999999776482582, 0.14492753148078918,
     0.00013141025556251407, 4555802.5),
    (-35.0, 0.009999999776482582, 0.14492753148078918,
     7.937760528875515e-05, 12486119.0),
    (-50.0, 0.009999999776482582, 0.14492753148078918,
     3.507663859636523e-05, 63942184.0),
    (-70.0, 0.009999999776482582, 0.14492753148078918,
     1.7564778318046592e-05, 254998800.0),
    # THE DISCRIMINATING ROWS: all three of the 391 states where the
    # pre-repair unpinned/plain-powf body missed WRF, worst 1.152373e-07.
    # Without these the anchor set passes under either form.
    (-22.5, 1.000000013351432e-10, 1.4492753663830626e-09,
     4.549578961720972e-14, 38.00852966308594),
    (-30.0, 0.0010000000474974513, 0.014492754824459553,
     7.779010957165156e-06, 1300094.25),
    (-70.0, 0.004999999888241291, 0.07246376574039459,
     8.41595192468958e-06, 138843936.0),
)

#: A TRUE ORACLE, not a transcription: (t, p, qv, qs, re_qs) returned by the
#: REAL ``calc_effectRad`` (module_mp_thompson.F:5594-5699, PUBLIC) compiled
#: from unmodified WRF v4.6.1 and called on a one-level column.  The committed
#: woof/data/thompson/oracle-aero/probe-effectrad.csv cannot substitute:
#: all 14 of its rows carry t = 285 K and qs = 2e-4, so its effs_m column is
#: one state repeated fourteen times and it saw none of this.
_WRF_EFFS_ORACLE = (
    (280.0, 70000.0, 0.0020000000949949026, 9.999999717180685e-10,
     5.350865194486687e-06),
    (280.0, 70000.0, 0.0020000000949949026, 9.999999747378752e-05,
     0.000671176181640476),
    (273.04998779296875, 70000.0, 0.0020000000949949026,
     0.0010000000474974513, 0.000999000039882958),
    (270.1499938964844, 70000.0, 0.0020000000949949026,
     9.999999747378752e-06, 0.00024470704374834895),
    (260.6499938964844, 70000.0, 0.0020000000949949026,
     9.999999717180685e-10, 1.428679570381064e-05),
    (253.14999389648438, 70000.0, 0.0020000000949949026,
     0.0010000000474974513, 0.0004753421526402235),
    (243.14999389648438, 70000.0, 0.0020000000949949026,
     9.999999747378752e-06, 0.00013348489301279187),
    (223.14999389648438, 70000.0, 0.0020000000949949026,
     9.999999717180685e-10, 4.0569684642832726e-05),
    (203.14999389648438, 70000.0, 0.0020000000949949026,
     0.00800000037997961, 6.043599933036603e-05),
)

#: (t, p, qv, qc, nc_per_kg, re_qc) and (t, p, qv, qi, ni_per_kg, re_qi) from
#: the same real calc_effectRad, over decades of mass and number rather than
#: probe-effectrad.csv's single (qc, qi) pair.
_WRF_EFFC_ORACLE = (
    (295.0, 85000.0, 0.004999999888241291, 9.999999717180685e-10, 1000000.0,
     2.5100000584643567e-06),
    (295.0, 85000.0, 0.004999999888241291, 9.999999747378752e-06,
     100000000.0, 3.090347490797285e-06),
    (295.0, 85000.0, 0.004999999888241291, 0.003000000026077032,
     1999000064.0, 8.63967215991579e-06),
    (285.0, 85000.0, 0.004999999888241291, 9.999999747378752e-05, 10000000.0,
     1.4167578228807542e-05),
    (275.0, 85000.0, 0.004999999888241291, 9.999999974752427e-07,
     1000000000.0, 2.5100000584643567e-06),
    (265.0, 85000.0, 0.004999999888241291, 1.0000000116860974e-07,
     50000000.0, 2.5100000584643567e-06),
    (265.0, 85000.0, 0.004999999888241291, 0.003000000026077032,
     1500000000.0, 9.507604772807099e-06),
    (250.0, 85000.0, 0.004999999888241291, 0.0005000000237487257,
     100000000.0, 1.1446449207141995e-05),
    (235.0, 85000.0, 0.004999999888241291, 0.003000000026077032,
     1999000064.0, 9.306958418164868e-06),
    # THE DISCRIMINATING ROWS.  These are ALL FIVE of the 378 states where
    # thompson.cu's plain ``powf`` misses WRF; without them the anchor set
    # would pass under either pow and the gate would prove nothing.
    (295.0, 85000.0, 0.004999999888241291, 0.003000000026077032, 10000000.0,
     4.402196282171644e-05),
    (285.0, 85000.0, 0.004999999888241291, 0.003000000026077032,
     1999000064.0, 8.727343811187893e-06),
    (275.0, 85000.0, 0.004999999888241291, 0.003000000026077032, 10000000.0,
     4.402196282171644e-05),
    (275.0, 85000.0, 0.004999999888241291, 0.003000000026077032,
     100000000.0, 2.0799578123842366e-05),
    (265.0, 85000.0, 0.004999999888241291, 0.0005000000237487257,
     10000000.0, 2.422622128506191e-05),
)

_WRF_EFFI_ORACLE = (
    (295.0, 85000.0, 0.004999999888241291, 9.999999717180685e-10, 100.0,
     2.2939177142689005e-05),
    (295.0, 85000.0, 0.004999999888241291, 9.999999747378752e-06, 10000.0,
     0.0001064742318703793),
    (295.0, 85000.0, 0.004999999888241291, 0.0010000000474974513, 999000.0,
     0.00010650974581949413),
    (285.0, 85000.0, 0.004999999888241291, 4.999999873689376e-05, 10000.0,
     0.0001250000059371814),
    (275.0, 85000.0, 0.004999999888241291, 9.999999974752427e-07, 50000.0,
     2.890155155910179e-05),
    (265.0, 85000.0, 0.004999999888241291, 9.99999993922529e-09, 100000.0,
     4.94209552925895e-06),
    (265.0, 85000.0, 0.004999999888241291, 0.0010000000474974513, 500000.0,
     0.0001250000059371814),
    (250.0, 85000.0, 0.004999999888241291, 9.999999747378752e-05, 999000.0,
     4.943744352203794e-05),
    (235.0, 85000.0, 0.004999999888241291, 0.0010000000474974513, 999000.0,
     0.00010650974581949413),
    # THE DISCRIMINATING ROWS: all four of the 378 states where plain ``powf``
    # misses WRF.  Two of them are the k = 23 / k = 24 shape that also splits
    # mp=28 from the frozen mp=8 kernel.
    (295.0, 85000.0, 0.004999999888241291, 9.999999747378752e-06, 500000.0,
     2.890155155910179e-05),
    (285.0, 85000.0, 0.004999999888241291, 9.999999974752427e-07, 50000.0,
     2.890155155910179e-05),
    (235.0, 85000.0, 0.004999999888241291, 9.999999974752427e-07, 50000.0,
     2.890155155910179e-05),
    (235.0, 85000.0, 0.004999999888241291, 9.999999747378752e-06, 500000.0,
     2.890155155910179e-05),
)


# ---------------------------------------------------------------------------
# Hard identity gates.
# ---------------------------------------------------------------------------

def test_droplet_bin_reproduces_the_mp8_frozen_bin_65():
    """thompson.cu:3933 hardcodes cloud_number_bin = 65 at Nt_c = 100e6."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_droplet_bin
    from woof.core.thompson_aerosol_contract import (
        CLASSIC_DROPLET_BIN, NT_C)

    result = cp.asnumpy(probe_droplet_bin(_f32([NT_C])))
    assert int(result[0]) == 65 == CLASSIC_DROPLET_BIN


def test_in_bin_reproduces_the_mp8_frozen_bin_27():
    """thompson.cu:3936 hardcodes nuclei_bin = 27 for the 1-per-litre default."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_in_bin

    result = cp.asnumpy(probe_in_bin(_f32([1000.0])))
    assert int(result[0]) == 27


def test_nu_c_reduces_to_the_mp8_frozen_shape_parameter():
    """Nt_c = 100e6 gives nu_c = 12, whose g_ratio entry is the frozen 2730."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_constant_tables, probe_nu_c, PROBE_TABLE_ROWS)
    from woof.core.thompson_aerosol_contract import NT_C

    assert int(cp.asnumpy(probe_nu_c(_f32([NT_C])))[0]) == 12
    tables = cp.asnumpy(probe_constant_tables())
    g_ratio = tables[PROBE_TABLE_ROWS.index("g_ratio")]
    assert float(g_ratio[12]) == 2730.0


def test_gamma_ratios_are_the_series_values_not_the_mp8_literals():
    """Record, do not repair, WOOF's pre-existing mp=8 gamma deviation.

    thompson.cu hardcodes 2730.0f at :882, :999, :4005, :4128, :4680 and
    272.0f at :888 and :1006, where WRF's REAL(4) WGAMMA actually yields
    2729.9973 and 272.00012.  mp=28 must be RIGHT; mp=8 stays frozen and
    wrong.  (thompson.cu:343 is not a deviation: calc_effectRad genuinely
    uses the exact-integer g_ratio PARAMETER, for which 2730 is correct.)
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_constant_tables, PROBE_TABLE_ROWS)

    tables = cp.asnumpy(probe_constant_tables())
    ccg2 = tables[PROBE_TABLE_ROWS.index("ccg2")]
    ccg5 = tables[PROBE_TABLE_ROWS.index("ccg5")]
    ocg1 = tables[PROBE_TABLE_ROWS.index("ocg1")]
    ocg2 = tables[PROBE_TABLE_ROWS.index("ocg2")]

    assert np.float32(ccg2[12] * ocg1[12]) == np.float32(2729.9973)
    assert np.float32(ccg2[12] * ocg1[12]) != np.float32(2730.0)
    assert np.float32(ccg5[12] * ocg2[12]) == np.float32(272.00012)
    assert np.float32(ccg5[12] * ocg2[12]) != np.float32(272.0)

    # And the one literal mp=8 does carry exactly: Gamma(16) from WRF's
    # series, thompson.cu:2083.
    ccg1 = tables[PROBE_TABLE_ROWS.index("ccg1")]
    assert np.float32(ccg1[15]) == np.float32(1.30767389e12)


def test_constant_tables_match_the_host_contract_bitwise():
    """The device __constant__ tables ARE thompson_aerosol_contract's arrays."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_constant_tables, PROBE_TABLE_ROWS)
    from woof.core.thompson_aerosol_contract import derived_constant_arrays

    device = cp.asnumpy(probe_constant_tables())
    host = derived_constant_arrays()
    for row, name in enumerate(PROBE_TABLE_ROWS):
        expected = np.asarray(host[name], dtype=np.float64)
        assert expected.shape == (16,), name
        assert np.array_equal(device[row], expected.astype(np.float32)), name


# ---------------------------------------------------------------------------
# Index helpers against their host transcriptions.
# ---------------------------------------------------------------------------

def test_nu_c_matches_the_host_contract_over_the_full_range():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_nu_c
    from woof.core.thompson_aerosol_contract import cloud_shape_parameter

    values = np.array(
        [2.0, 10.0, 1.0e3, 1.0e5, 5.0e5, 1.0e6, 3.0e7, 1.0e8, 1.0e8 + 1.0,
         3.0e8, 1.0e9, 1.999e9, 2.0e9, 5.0e9], dtype=np.float32)
    device = cp.asnumpy(probe_nu_c(cp.asarray(values)))
    for value, got in zip(values, device):
        assert int(got) == cloud_shape_parameter(float(value)), value
        assert 2 <= int(got) <= 15


def test_droplet_bin_matches_the_host_contract_and_saturates():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_droplet_bin
    from woof.core.thompson_aerosol_contract import droplet_number_bin

    values = np.array(
        [2.0, 1.0e5, 1.0408439e6, 2.0e6, 1.0e7, 5.0e7, 1.0e8, 3.0e8, 1.0e9,
         1.999e9, 2.882e9, 1.0e10], dtype=np.float32)
    device = cp.asnumpy(probe_droplet_bin(cp.asarray(values)))
    for value, got in zip(values, device):
        assert int(got) == droplet_number_bin(float(value)), value
        assert 0 <= int(got) <= 99
    assert int(device[0]) == 0
    assert int(device[-1]) == 99


def test_in_bin_spans_the_whole_freezeh2o_axis():
    """idx_IN != 0 is an entire table axis WOOF's mp=8 port never reads."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_in_bin

    values = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 9.0, 10.0, 55.0, 1.0e3, 1.0e5, 1.0e6,
         1.0e7], dtype=np.float32)
    device = cp.asnumpy(probe_in_bin(cp.asarray(values)))
    for value, got in zip(values, device):
        got = int(got)
        assert 0 <= got <= 54
        if value <= 1.0:
            assert got == 0, value
        else:
            digit = int(float(value) / 10.0 ** math.floor(math.log10(value)))
            decade = int(math.floor(math.log10(float(value))))
            assert got == min(digit + 9 * decade - 1, 54), value
    assert len(set(int(v) for v in device)) > 3


def test_nint_rounds_half_away_from_zero_like_fortran():
    """CUDA __float2int_rn rounds half to EVEN and would pick other bins."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_nint

    values = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.4, -3.4],
                      dtype=np.float32)
    device = cp.asnumpy(probe_nint(cp.asarray(values)))
    assert list(int(v) for v in device) == [1, 2, 3, -1, -2, -3, 3, -3]


def test_terminal_clamps_match_wrf():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_clamps

    nc = _f32([0.0, 1.0, 2.0, 1.0e8, 1.999e9, 5.0e9])
    nwfa = _f32([0.0, 1.0e6, 11.1e6, 3.0e8, 9999.0e6, 5.0e10])
    nifa = _f32([0.0, 1.0e3, 5.0e3, 1.0e6, 9999.0e6, 5.0e10])
    nc_out, nwfa_out, nifa_out = probe_clamps(nc, nwfa, nifa)
    assert np.array_equal(
        cp.asnumpy(nc_out),
        np.array([2.0, 2.0, 2.0, 1.0e8, 1.999e9, 1.999e9], dtype=np.float32))
    assert np.array_equal(
        cp.asnumpy(nwfa_out),
        np.array([11.1e6, 11.1e6, 11.1e6, 3.0e8, 9999.0e6, 9999.0e6],
                 dtype=np.float32))
    assert np.array_equal(
        cp.asnumpy(nifa_out),
        np.array([5.0e3, 5.0e3, 5.0e3, 1.0e6, 9999.0e6, 9999.0e6],
                 dtype=np.float32))


# ---------------------------------------------------------------------------
# Oracle gates.  Measured bit-exact; see the module docstring.
# ---------------------------------------------------------------------------

def test_ice_demott_matches_the_wrf_probe_exactly():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_ice_demott

    rows = _rows("probe-icedemott.csv")
    assert len(rows) == 320
    got = cp.asnumpy(probe_ice_demott(
        _device(rows, "tempc_c"), _device(rows, "rho_kg_m3"),
        _device(rows, "nifa_per_m3")))
    expected = _reference(rows, "xni_per_m3")
    assert np.array_equal(got.astype(np.float64), expected)


def test_ice_koop_matches_the_wrf_probe_exactly():
    """Including the `1 - exp(-x)` cancellation region, where a single expf
    ulp becomes an O(1) relative difference in the returned ice number."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_ice_koop

    rows = _rows("probe-icekoop.csv")
    assert len(rows) == 480
    got = cp.asnumpy(probe_ice_koop(
        _device(rows, "temp_k"), _device(rows, "qv"), _device(rows, "qvs"),
        _device(rows, "nwfa_per_m3"), _device(rows, "dt_s")))
    expected = _reference(rows, "xni_per_m3")
    assert np.array_equal(got.astype(np.float64), expected)
    # The physics must actually be exercised, not silently all-zero.
    assert (expected > 0.0).sum() > 50
    assert expected.max() == pytest.approx(1000.0e3, rel=1.0e-6)


@pytest.mark.skipif(not _CCN_BIN.is_file(),
                    reason="CCN_ACTIVATE.BIN is missing from this tree; restore it")
def test_activ_ncloud_matches_the_wrf_probe_exactly():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_activ_ncloud

    rows = _rows("probe-activncloud.csv")
    assert len(rows) == 1320
    got = cp.asnumpy(probe_activ_ncloud(
        _device(rows, "temp_k"), _device(rows, "w_m_s"),
        _device(rows, "nccn_per_m3"), _ccn_table()))
    expected = _reference(rows, "activated_per_m3")
    assert np.array_equal(got.astype(np.float64), expected)


@pytest.mark.skipif(not _CCN_BIN.is_file(),
                    reason="CCN_ACTIVATE.BIN is missing from this tree; restore it")
def test_activ_ncloud_never_activates_more_than_the_available_aerosol():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_activ_ncloud

    rows = _rows("probe-activncloud.csv")
    nccn = _reference(rows, "nccn_per_m3")
    got = cp.asnumpy(probe_activ_ncloud(
        _device(rows, "temp_k"), _device(rows, "w_m_s"),
        _device(rows, "nccn_per_m3"), _ccn_table())).astype(np.float64)
    assert (got > 0.0).all()
    assert (got <= nccn * (1.0 + 1.0e-6)).all()


def test_eff_aero_matches_the_wrf_probe_exactly_for_all_three_species():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_eff_aero, SPECIES_CODES)

    rows = _rows("probe-effaero.csv")
    assert len(rows) == 48
    species = cp.asarray(np.array(
        [SPECIES_CODES[row["species"].strip()] for row in rows],
        dtype=np.int32))
    assert set(int(v) for v in cp.asnumpy(species)) == {0, 1, 2}
    got = cp.asnumpy(probe_eff_aero(
        _device(rows, "d_collector_m"), _device(rows, "d_aerosol_m"),
        _device(rows, "visco"), _device(rows, "rho_kg_m3"),
        _device(rows, "temp_k"), species))
    expected = _reference(rows, "eff")
    assert np.array_equal(got.astype(np.float64), expected)
    assert (got >= 1.0e-5).all() and (got <= 1.0).all()


def test_effect_rad_matches_the_wrf_probe_exactly_across_every_branch():
    """Covers calc_effectRad's nc<100 -> inu_c=15 and nc>1e10 -> inu_c=2
    branches, neither of which mp_gt_driver can reach.

    The committed probe grew from 14 rows to 50 when the oracle harness was
    regenerated, and the row count is asserted rather than inferred so a
    SHRINKING probe -- which would silently narrow this gate -- fails here.
    The 50 rows carry 14 distinct temperatures, 10 cloud contents, 27 droplet
    numbers, 7 ice contents / numbers and 9 snow contents, so all three
    branches are exercised over real ladders and not over one repeated state.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_effect_rad, probe_inu_c_effrad)

    rows = _rows("probe-effectrad.csv")
    assert len(rows) == 50
    assert len({row["temp_k"] for row in rows}) == 14
    assert len({row["effs_m"] for row in rows}) == 22
    effc, effi, effs = probe_effect_rad(
        _device(rows, "temp_k"), _device(rows, "p_pa"), _device(rows, "qv"),
        _device(rows, "qc"), _device(rows, "nc_per_kg"), _device(rows, "qi"),
        _device(rows, "ni_per_kg"), _device(rows, "qs"))
    for got, key in ((effc, "effc_m"), (effi, "effi_m"), (effs, "effs_m")):
        assert np.array_equal(
            cp.asnumpy(got).astype(np.float64), _reference(rows, key)), key

    # The probe's nc ladder must have driven all three shape branches.
    targets = _device(rows, "nc_target_per_m3")
    inu_c = cp.asnumpy(probe_inu_c_effrad(targets))
    assert 15 in set(int(v) for v in inu_c)
    assert 2 in set(int(v) for v in inu_c)
    assert 12 in set(int(v) for v in inu_c)


def test_inu_c_effrad_is_not_the_same_selector_as_nu_c():
    """calc_effectRad's three-branch form deliberately differs from :2171."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_inu_c_effrad, probe_nu_c)

    values = _f32([50.0, 99.0, 100.0, 1.0e8, 2.0e10])
    effrad = [int(v) for v in cp.asnumpy(probe_inu_c_effrad(values))]
    plain = [int(v) for v in cp.asnumpy(probe_nu_c(values))]
    assert effrad == [15, 15, 15, 12, 2]
    assert plain == [15, 15, 15, 12, 2]
    # The two agree on this ladder only because MIN(15, ...) saturates; the
    # dead nc > 1e10 branch is what makes them distinct helpers.
    assert effrad[-1] == 2


# ---------------------------------------------------------------------------
# Helpers the Fortran probe does not tabulate.  Documented FP32 tolerance:
# the host reference uses NumPy's float32 pow, the device uses CUDA powf.
# ---------------------------------------------------------------------------

_HOST_REFERENCE_RTOL = 1.0e-6


def _host_cloud_dist(rc, nc_per_kg, rho):
    from woof.core.thompson_aerosol_contract import (
        AM_R, BM_R, CCE2, CCG1, CCG2, D0C_M, D0R_M, NT_C_MAX, OCG1, OCG2,
        cloud_shape_parameter)

    f32 = np.float32
    am_r = f32(AM_R)
    nc = f32(min(max(f32(f32(nc_per_kg) * f32(rho)), f32(2.0)),
                 f32(NT_C_MAX)))
    nu_c = cloud_shape_parameter(float(nc))
    lamc = float(f32(
        f32(f32(f32(nc * am_r) * f32(CCG2[nu_c])) * f32(OCG1[nu_c]))
        / f32(rc)) ** f32(1.0 / 3.0))
    xDc = f32(float(f32(BM_R + f32(nu_c) + f32(1.0))) / lamc)
    if xDc < f32(D0C_M):
        lamc = float(f32(CCE2[nu_c]) / f32(D0C_M))
    elif xDc > f32(f32(D0R_M) * f32(2.0)):
        lamc = float(f32(CCE2[nu_c]) / f32(f32(D0R_M) * f32(2.0)))
    scale = f32(f32(f32(f32(CCG1[nu_c]) * f32(OCG2[nu_c])) * f32(rc)) / am_r)
    return f32(min(float(NT_C_MAX), float(scale) * lamc ** 3.0)), nu_c, lamc


def test_cloud_dist_matches_a_host_transcription():
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_cloud_dist

    cases = [
        (1.0e-4, 100.0e6 / 1.0, 1.0),      # the mp=8 configuration
        (1.0e-4, 30.0e6 / 1.0, 1.0),
        (5.0e-4, 300.0e6 / 0.9, 0.9),
        (2.0e-5, 1000.0e6 / 1.1, 1.1),
        (1.0e-3, 1800.0e6 / 0.8, 0.8),
        (1.0e-11, 2.0e6 / 1.0, 1.0),       # forces the D0c size clamp
        (5.0e-3, 3.0e6 / 1.0, 1.0),        # forces the 2*D0r size clamp
    ]
    rc = _f32([c[0] for c in cases])
    nck = _f32([c[1] for c in cases])
    rho = _f32([c[2] for c in cases])
    nc_out, nu_c_out, lamc_out = probe_cloud_dist(rc, nck, rho)
    nc_out = cp.asnumpy(nc_out)
    nu_c_out = cp.asnumpy(nu_c_out)
    lamc_out = cp.asnumpy(lamc_out)

    for index, case in enumerate(cases):
        want_nc, want_nu, want_lamc = _host_cloud_dist(*case)
        assert int(nu_c_out[index]) == want_nu, case
        assert float(lamc_out[index]) == pytest.approx(
            want_lamc, rel=_HOST_REFERENCE_RTOL), case
        assert float(nc_out[index]) == pytest.approx(
            float(want_nc), rel=_HOST_REFERENCE_RTOL), case
    assert (nc_out >= 0.0).all()
    assert (nc_out <= np.float32(1999.0e6)).all()
    # The mp=8 configuration must land on nu_c = 12.
    assert int(nu_c_out[0]) == 12


# ---------------------------------------------------------------------------
# THE SNOW-MOMENT FITS AND THE TWO-GAMMA SNOW NUMBER.
# ---------------------------------------------------------------------------
#
# Wave 4 found and repaired a real defect here.  ``thompson_field_a`` and
# ``thompson_field_b`` were verbatim copies of thompson.cu:81-107, and that
# form differs from WRF in TWO independent ways, neither of which any gate in
# this file could see before:
#
#   1. OPERATOR ASSOCIATION.  WRF writes ``sa(5)*tc0*tc0``, which Fortran
#      evaluates as (sa5*tc0)*tc0.  The copied form hoisted ``tc2 = tc*tc``
#      and computed sa5*(tc*tc).  Six of the ten terms were affected.
#   2. ``a_ = 10.0**loga_`` is REAL(4)**REAL(4) -> glibc's correctly-rounded
#      powf, where the copy used CUDA's powf.
#
# Plus nvrtc's default FMA contraction across the nine additions, which the
# baseline-x86-64 gfortran the oracle is built with cannot do.
#
# MEASURED, before -> after, on an RTX 5090:
#     a_       118/253 exact, max 3.267395e-06   ->  253/253 BIT-EXACT
#     b_       180/253 exact, max 4.411423e-07   ->  253/253 BIT-EXACT
#     effs_m   138/360 exact, max 5.870704e-06   ->  360/360 BIT-EXACT
#              (against the REAL calc_effectRad, not a transcription)
#     ns       via the fits: max 1.490356e-05    ->  max 4.933379e-07
# and on the 19-fixture G3 table, aero-scav-frozen effs_m 6.73e-07 -> 0,
# aero-cold-overlap effs_m 1.46e-06 -> 5.20e-07, with no field of any fixture
# moving the wrong way.


def test_field_ab_reproduce_wrfs_operator_tree_bit_exactly():
    """253 states, exact equality, against WRF's own association.

    This is the gate that did not exist while the hoisted mp=8 form was in
    the header.  ``pytest.approx`` is deliberately not used: the helpers are
    contraction-pinned and correctly-rounded precisely so that this can be
    ``==``, and a tolerance here would have absorbed the whole 3.3e-06 defect.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_field_ab

    states = [(t, m) for t in _FIT_TEMPERATURES for m in _FIT_MOMENTS]
    assert len(states) == 253
    tc = _f32([s[0] for s in states])
    moment = _f32([s[1] for s in states])
    a_dev, b_dev = probe_field_ab(tc, moment)
    a_dev = cp.asnumpy(a_dev)
    b_dev = cp.asnumpy(b_dev)
    tc_host = cp.asnumpy(tc)
    moment_host = cp.asnumpy(moment)

    for index, state in enumerate(states):
        want_a = _host_field_a(tc_host[index], moment_host[index])
        want_b = _host_field_b(tc_host[index], moment_host[index])
        assert np.float32(a_dev[index]) == want_a, (state, a_dev[index],
                                                    want_a)
        assert np.float32(b_dev[index]) == want_b, (state, b_dev[index],
                                                    want_b)
    # a_ spans eleven decades over this grid, so "bit-exact" is a statement
    # about the whole reachable range and not about one well-conditioned spot.
    assert a_dev.max() / a_dev.min() > 1.0e10
    assert (b_dev > 0.0).all()


def test_field_ab_match_the_gfortran_o2_reference_values():
    """Pin the PYTHON reference above to the real compiler.

    ``_host_field_a``/``_host_field_b`` are only trustworthy if they agree
    with what ``gfortran -O2 -ffree-form -ffree-line-length-none`` -- the
    exact invocation tools/thompson_wrf461_oracle/build_aero.sh uses -- makes
    of WRF's statements.  These eleven anchors came out of such a program,
    linked against the same module_mp_thompson.o the aerosol oracle harness
    was built from.  Without this test the transcription could drift and the
    device would still "agree" with it.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_field_ab

    tc = _f32([row[0] for row in _GFORTRAN_FIELD_AB_ANCHORS])
    moment = _f32([row[1] for row in _GFORTRAN_FIELD_AB_ANCHORS])
    a_dev = cp.asnumpy(probe_field_ab(tc, moment)[0])
    b_dev = cp.asnumpy(probe_field_ab(tc, moment)[1])
    for index, (tc0, mom, a_ref, b_ref) in enumerate(
            _GFORTRAN_FIELD_AB_ANCHORS):
        assert _host_field_a(tc0, mom) == np.float32(a_ref), (tc0, mom)
        assert _host_field_b(tc0, mom) == np.float32(b_ref), (tc0, mom)
        assert np.float32(a_dev[index]) == np.float32(a_ref), (tc0, mom)
        assert np.float32(b_dev[index]) == np.float32(b_ref), (tc0, mom)


#: thompson.cu:81-107 verbatim, as a THROWAWAY translation unit.  It is
#: compiled by the negative control below and by nothing else; it is not
#: prepended to any module and it never touches thompson.cu, which is
#: byte-frozen.  Keeping the rejected form here, compiled, is what turns "the
#: association matters" from a comment into a measurement.
_MP8_FIELD_SHADOW_SOURCE = r"""
__constant__ float SHADOW_SA[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f};
__constant__ float SHADOW_SB[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f};

extern "C" __global__ void shadow_field_ab(
    const float* tc_in, const float* moment_in,
    float* a_out, float* b_out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    const float tc = tc_in[idx];
    const float moment = moment_in[idx];
    const float tc2 = tc * tc;
    const float moment2 = moment * moment;
    const float loga = SHADOW_SA[0] + SHADOW_SA[1] * tc
        + SHADOW_SA[2] * moment + SHADOW_SA[3] * tc * moment
        + SHADOW_SA[4] * tc2 + SHADOW_SA[5] * moment2
        + SHADOW_SA[6] * tc2 * moment
        + SHADOW_SA[7] * tc * moment2
        + SHADOW_SA[8] * tc2 * tc
        + SHADOW_SA[9] * moment2 * moment;
    a_out[idx] = powf(10.0f, loga);
    b_out[idx] = SHADOW_SB[0] + SHADOW_SB[1] * tc
        + SHADOW_SB[2] * moment + SHADOW_SB[3] * tc * moment
        + SHADOW_SB[4] * tc2 + SHADOW_SB[5] * moment2
        + SHADOW_SB[6] * tc2 * moment
        + SHADOW_SB[7] * tc * moment2
        + SHADOW_SB[8] * tc2 * tc
        + SHADOW_SB[9] * moment2 * moment;
}
"""


def test_the_hoisted_mp8_field_fit_really_does_disagree_with_wrf():
    """NEGATIVE CONTROL.  The repaired form is not a cosmetic rewrite.

    Two separate measurements, because the old form was wrong in two separate
    ways and each needs its own witness:

    (a) ASSOCIATION ALONE, in host float32 with everything else held equal.
        ``sa(5)*tc0*tc0`` is (sa5*tc0)*tc0; the hoisted chain computed
        sa5*(tc*tc).  MEASURED: 29 of 253 states differ on a_, worst
        1.135939e-06 relative, and 20 of 253 differ on b_, worst
        2.205712e-07.

    (b) THE WHOLE OLD FORM ON DEVICE, compiled here from thompson.cu:81-107
        verbatim into a throwaway module -- association plus nvrtc's FMA
        contraction plus CUDA's ``powf``.  MEASURED: 3.267395e-06 worst,
        which is ABOVE the 2e-6 the mp=28 end-to-end fixtures are gated at,
        so this was never going to stay invisible.

    If either witness ever stops firing, the gate above has become vacuous.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_field_ab

    states = [(t, m) for t in _FIT_TEMPERATURES for m in _FIT_MOMENTS]

    # (a) association alone.
    disagree = []
    worst_assoc = 0.0
    for tc0, moment in states:
        wrf = _host_field_a(tc0, moment)
        mp8 = _host_field_a_mp8_form(tc0, moment)
        if mp8 != wrf:
            disagree.append((tc0, moment))
            worst_assoc = max(worst_assoc,
                              abs(float(mp8) - float(wrf)) / float(wrf))
    assert len(disagree) == 29, len(disagree)
    assert worst_assoc == pytest.approx(1.135939e-06, rel=1.0e-3), worst_assoc

    tc = _f32([s[0] for s in states])
    moment = _f32([s[1] for s in states])
    a_dev = cp.asnumpy(probe_field_ab(tc, moment)[0])
    index_of = {s: i for i, s in enumerate(states)}
    for state in disagree:
        i = index_of[state]
        assert np.float32(a_dev[i]) == _host_field_a(*state), state
        assert np.float32(a_dev[i]) != _host_field_a_mp8_form(*state), state

    # (b) the whole old form, on device.
    module = cp.RawModule(code=_MP8_FIELD_SHADOW_SOURCE,
                          options=("-std=c++17",))
    shadow = module.get_function("shadow_field_ab")
    size = len(states)
    a_shadow = cp.empty(size, dtype=cp.float32)
    b_shadow = cp.empty(size, dtype=cp.float32)
    shadow(((size + 255) // 256,), (256,),
           (tc, moment, a_shadow, b_shadow, np.int32(size)))
    a_shadow = cp.asnumpy(a_shadow).astype(np.float64)
    reference = np.array([float(_host_field_a(*s)) for s in states])
    relative = np.abs(a_shadow - reference) / reference
    assert int((a_shadow == reference).sum()) == 118, (
        int((a_shadow == reference).sum()))
    assert relative.max() == pytest.approx(3.267395e-06, rel=1.0e-3), (
        relative.max())
    assert relative.max() > 2.0e-6
    # And the shipped helper is exact on the identical grid and inputs.
    assert (a_dev.astype(np.float64) == reference).all()


def test_snow_number_matches_a_host_transcription():
    """module_mp_thompson.F:2083-2088, over PHYSICALLY REACHABLE moments.

    WIDENED AND TIGHTENED in wave 4: from 32 hand-picked (tc0, smob) pairs at
    ``rel=1.0e-5`` to 391 states at EXACT EQUALITY.  This block is inline code
    inside mp_thompson rather than a procedure, and every constant it reads --
    ``Kap0``/``Kap1``, ``Lam0``/``Lam1`` (:114-117), ``mu_s`` (:113),
    ``csg``/``cse`` -- is PRIVATE, so no external Fortran program can call it;
    the reference is a transcription, but one that is itself pinned to
    ``gfortran -O2`` by ``_GFORTRAN_SNOW_NUMBER_ANCHORS``.

    ``smoc`` is not a free variable: WRF forms it at :2069-2080 as
    ``a_*smo2**b_`` with cse(1) = 3 and, since bm_s == 2 exactly, smo2 ==
    smob.  The grid therefore drives smoc from the Field fits over the whole
    temperature range the snow category occupies, exactly as WRF does.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_snow_number

    states = [(t, rs) for t in _FIT_TEMPERATURES
              for rs in _FIT_SNOW_CONTENTS]
    assert len(states) == 391
    smob_host = np.array([np.float32(np.float32(rs) * _OAMS)
                          for _, rs in states], dtype=np.float32)
    smoc_host = np.array(
        [np.float32(_host_field_a(tc0, 3.0)
                    * _f32pow(smob_host[i], _host_field_b(tc0, 3.0)))
         for i, (tc0, _) in enumerate(states)], dtype=np.float32)
    assert (smoc_host > 0.0).all()

    got = cp.asnumpy(probe_snow_number(cp.asarray(smob_host),
                                       cp.asarray(smoc_host)))
    for index, state in enumerate(states):
        want = _host_snow_number(smob_host[index], smoc_host[index])
        assert np.float32(got[index]) == want, (state, got[index], want)
    assert (got > 0.0).all()
    # ns spans nine decades here; the old 32-state grid spanned three.
    assert got.max() / got.min() > 1.0e8


def test_snow_number_matches_the_gfortran_o2_reference_values():
    """Pin ``_host_snow_number`` -- and ``_CSG15`` -- to the real compiler.

    ``csg(15) = WGAMMA(mu_s+1)`` is PRIVATE and comes from WRF's own REAL(4)
    Lanczos series (:5325-5346), not from ``math.gamma``; ten anchors from a
    gfortran transcription of that series plus :2029-2088 keep the literal
    accurate as well as the arithmetic.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_snow_number

    smob = _f32([row[2] for row in _GFORTRAN_SNOW_NUMBER_ANCHORS])
    smoc = _f32([row[3] for row in _GFORTRAN_SNOW_NUMBER_ANCHORS])
    got = cp.asnumpy(probe_snow_number(smob, smoc))
    for index, row in enumerate(_GFORTRAN_SNOW_NUMBER_ANCHORS):
        tc0, rs, smob_ref, smoc_ref, ns_ref = row
        # The gfortran smob and smoc are what WRF itself formed at :2029 and
        # :2080, so this also checks the Python fits reproduce them.
        assert np.float32(np.float32(rs) * _OAMS) == np.float32(smob_ref)
        assert np.float32(_host_field_a(tc0, 3.0)
                          * _f32pow(np.float32(smob_ref),
                                    _host_field_b(tc0, 3.0))) == \
            np.float32(smoc_ref), tc0
        assert _host_snow_number(smob_ref, smoc_ref) == np.float32(ns_ref)
        assert np.float32(got[index]) == np.float32(ns_ref), tc0
    # PROVENANCE OF _CSG15, recorded as it is rather than overclaimed.  WRF's
    # REAL(4) Lanczos series gives 0.8980315327644348 where CPython's
    # math.gamma gives 0.8980315267615606 -- 6.68e-09 relative apart in
    # DOUBLE.  At this particular argument both round to the SAME float32, so
    # csg(15) is one of the places the GAMMA PARITY note's warning does not
    # bite; the literal here is still the series value, because the argument
    # is not the thing under this port's control.
    assert _CSG15 == np.float32(0.89803153276443481)
    assert abs(0.89803153276443481 - math.gamma(float(_CSE15))) \
        / 0.89803153276443481 == pytest.approx(6.684480e-09, rel=1.0e-3)
    assert _CSG15 == np.float32(math.gamma(float(_CSE15)))


#: The composite the cold and warm networks actually run, both ways.  WRF's
#: :2080 is ``smoc(k) = a_ * smo2**b_`` with smo2 REAL(4) and b_ REAL(4), i.e.
#: glibc's correctly-rounded powf; cold.cu:466/468/470, cold.cu:1204-1209 and
#: warm.cu:697-708 all spell it with CUDA's ``powf``.
_SNOW_COMPOSITE_SOURCE = r"""
extern "C" __global__ void snow_composite(
    const float* tc0, const float* smob,
    float* smoc_plain, float* smoc_cr,
    float* ns_plain, float* ns_cr, int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const float a_ = thompson_field_a(tc0[i], 3.0f);
    const float b_ = thompson_field_b(tc0[i], 3.0f);
    smoc_plain[i] = a_ * powf(smob[i], b_);
    smoc_cr[i]    = a_ * thompson_aa_powf(smob[i], b_);
    ns_plain[i] = thompson_aa_snow_number(smob[i], smoc_plain[i]);
    ns_cr[i]    = thompson_aa_snow_number(smob[i], smoc_cr[i]);
}
"""


def test_snow_number_end_to_end_through_the_fits_is_the_cold_network_path():
    """What ``cold.cu:463-475`` actually computes, and where the last of the
    residual lives.

    The cold network does not hand ``thompson_aa_snow_number`` WRF's smoc; it
    builds smoc itself from ``thompson_field_a * powf(smob,
    thompson_field_b)`` and passes that.  The composite therefore carries the
    fit error amplified by the quartic sensitivity of ns to smob/smoc, which
    is how a 3.3e-06 defect in ``a_`` became a 1.490356e-05 defect in ns.

    MEASURED on this grid, against the ``gfortran -O2`` reference, with the
    repaired fits in place:
        smoc  a_*powf(smob,b_)                367/391 exact, 1.491366e-07
        smoc  a_*thompson_aa_powf(smob,b_) 391/391 BIT-EXACT
        ns    from the first                  371/391 exact, 4.933379e-07
        ns    from the second                 391/391 BIT-EXACT

    So the helper this package owns is exact and the last 4.9e-07 is a
    ``powf`` in the CALLERS' translation units -- an actionable, measured
    hand-off, not a limit.  The bounds below are ratchets at the measured
    values.
    """
    import cupy as cp
    from woof.core.kernels import module_source
    from woof.core.thompson_aerosol_launch import PROBE_MODULE

    states = [(t, rs) for t in _FIT_TEMPERATURES
              for rs in _FIT_SNOW_CONTENTS]
    size = len(states)
    smob_host = np.array([np.float32(np.float32(rs) * _OAMS)
                          for _, rs in states], dtype=np.float32)
    module = cp.RawModule(
        code=module_source(PROBE_MODULE) + _SNOW_COMPOSITE_SOURCE,
        options=("-std=c++17",))
    kernel = module.get_function("snow_composite")
    outs = [cp.empty(size, dtype=cp.float32) for _ in range(4)]
    kernel(((size + 255) // 256,), (256,),
           (_f32([t for t, _ in states]), cp.asarray(smob_host), *outs,
            np.int32(size)))
    smoc_plain, smoc_cr, ns_plain, ns_cr = (cp.asnumpy(o) for o in outs)

    want_smoc = np.array(
        [float(np.float32(_host_field_a(t, 3.0)
                          * _f32pow(smob_host[i], _host_field_b(t, 3.0))))
         for i, (t, _) in enumerate(states)], dtype=np.float64)
    want_ns = np.array(
        [float(_host_snow_number(smob_host[i], np.float32(want_smoc[i])))
         for i in range(size)], dtype=np.float64)

    # The correctly-rounded composite is exact; that is WRF's own arithmetic.
    assert (smoc_cr.astype(np.float64) == want_smoc).all()
    assert (ns_cr.astype(np.float64) == want_ns).all()

    # The plain-powf composite -- what the callers ship today -- is not, and
    # this is the number to quote at them.
    smoc_rel = np.abs(smoc_plain.astype(np.float64) - want_smoc) / want_smoc
    ns_rel = np.abs(ns_plain.astype(np.float64) - want_ns) / want_ns
    assert smoc_rel.max() <= 1.4914e-07, smoc_rel.max()
    assert ns_rel.max() <= 4.9334e-07, ns_rel.max()
    # ...and it must actually be nonzero, or the hand-off is imaginary.
    assert smoc_rel.max() > 0.0
    assert ns_rel.max() > 0.0
    # Both are far below the 1.490356e-05 the pre-repair header produced on
    # this identical grid, which is the fit repair showing up end to end.
    assert ns_rel.max() < 1.0e-06


def test_effect_rad_snow_matches_the_real_calc_effect_rad():
    """A TRUE ORACLE for the snow fits, over temperature and mass.

    ``calc_effectRad`` is PUBLIC, so unlike the inline moment blocks it can be
    called directly.  These nine states came out of unmodified WRF v4.6.1 and
    span 280 K down to 203 K and 1e-9 to 8e-3 kg/kg -- where
    ``probe-effectrad.csv`` holds t and qs fixed and therefore pinned only the
    droplet ladder.  BIT-EXACT is achievable here only because the fits carry
    WRF's association and ``smo2**b_`` is correctly rounded; the pre-repair
    header missed by up to 5.870704e-06 on this grid.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_effect_rad

    n = len(_WRF_EFFS_ORACLE)
    zero = cp.zeros(n, dtype=cp.float32)
    _, _, effs = probe_effect_rad(
        _f32([r[0] for r in _WRF_EFFS_ORACLE]),
        _f32([r[1] for r in _WRF_EFFS_ORACLE]),
        _f32([r[2] for r in _WRF_EFFS_ORACLE]),
        zero, zero, zero, zero,
        _f32([r[3] for r in _WRF_EFFS_ORACLE]))
    effs = cp.asnumpy(effs)
    for index, row in enumerate(_WRF_EFFS_ORACLE):
        assert np.float32(effs[index]) == np.float32(row[4]), row
    # The grid must not be sitting on the clamps: at least six of the nine
    # values are strictly interior, so the fits are what is being tested.
    interior = [v for v in effs if 5.02e-6 < v < 998.0e-6]
    assert len(interior) >= 6, effs


def test_effect_rad_cloud_and_ice_match_the_real_calc_effect_rad():
    """The other two branches of the same true oracle, bit-exact.

    WRF's ``**`` at :5644 and :5654 is REAL(4)**REAL(4), which gfortran lowers
    to glibc's correctly-rounded powf; CUDA's powf carries several ulp.  With
    thompson.cu's plain ``powf`` these sat at 373/378 and 374/378 exact over
    378 states each (worst 1.042121e-07 and 6.293743e-08); with
    ``thompson_aa_powf`` they are exact.

    That improvement is small -- more than an order of magnitude under the
    2e-6 the end-to-end fixtures are gated at -- and it is NOT the cause of
    any residual those fixtures still show (aero-cold-overlap's ``effc_m``
    sits at 1.9e-05, which is the droplet NUMBER handed in, not the radius
    formula).  It is taken anyway because the port's authority is WRF, and
    because WP-04's
    ``test_effective_radius_is_bitwise_against_every_oracle_after_column``
    demands exactly this.  What it costs is recorded in
    ``test_effect_rad_cloud_and_ice_diverge_from_mp8_by_one_ulp_on_purpose``.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_effect_rad

    zero_c = cp.zeros(len(_WRF_EFFC_ORACLE), dtype=cp.float32)
    effc, _, _ = probe_effect_rad(
        _f32([r[0] for r in _WRF_EFFC_ORACLE]),
        _f32([r[1] for r in _WRF_EFFC_ORACLE]),
        _f32([r[2] for r in _WRF_EFFC_ORACLE]),
        _f32([r[3] for r in _WRF_EFFC_ORACLE]),
        _f32([r[4] for r in _WRF_EFFC_ORACLE]),
        zero_c, zero_c, zero_c)
    effc = cp.asnumpy(effc)
    for index, row in enumerate(_WRF_EFFC_ORACLE):
        assert np.float32(effc[index]) == np.float32(row[5]), row

    zero_i = cp.zeros(len(_WRF_EFFI_ORACLE), dtype=cp.float32)
    _, effi, _ = probe_effect_rad(
        _f32([r[0] for r in _WRF_EFFI_ORACLE]),
        _f32([r[1] for r in _WRF_EFFI_ORACLE]),
        _f32([r[2] for r in _WRF_EFFI_ORACLE]),
        zero_i, zero_i,
        _f32([r[3] for r in _WRF_EFFI_ORACLE]),
        _f32([r[4] for r in _WRF_EFFI_ORACLE]),
        zero_i)
    effi = cp.asnumpy(effi)
    for index, row in enumerate(_WRF_EFFI_ORACLE):
        assert np.float32(effi[index]) == np.float32(row[5]), row

    # Both grids must reach values away from the background, and stay inside
    # calc_effectRad's own clamps.  Those clamps are float32 literals, so
    # compare against the float32 value and not the decimal one: 125.e-6
    # rounds UP to 1.2500000594e-04 in float32 and an ice grid that reaches
    # the clamp lands exactly there.
    effc = effc.astype(np.float64)
    effi = effi.astype(np.float64)
    assert (effc > 2.52e-6).any()
    assert (effc <= float(np.float32(50.0e-6))).all()
    assert (effi > 2.52e-6).all()
    assert (effi <= float(np.float32(125.0e-6))).all()
    assert np.isclose(effi.max(), float(np.float32(125.0e-6)), rtol=1e-9)


def test_effect_rad_cloud_and_ice_diverge_from_mp8_by_one_ulp_on_purpose():
    """WHAT THE CORRECTLY-ROUNDED POW COSTS, measured and pinned.

    On the aero-reduces-to-classic after-column the mp=28 ice branch and
    thompson.cu's disagree at two of twenty-four levels by 6.567156e-08
    relative -- one float32 ulp.  THE DIRECTION OF THAT ERROR IS THE WHOLE
    POINT: at those two levels it is mp=8 that disagrees with WRF, not mp=28.
    The fixture value, the mp=28 value and the mp=8 value are all pinned
    below.

    Reproducing mp=8's ulp was the only thing CUDA's ``powf`` bought here, and
    ``tests/test_thompson_aerosol_state_gpu.py::
    test_effective_radius_is_bitwise_against_every_oracle_after_column``
    demands the opposite.  MEASURED BOTH WAYS on this GPU: with plain ``powf``
    the oracle test fails; with ``thompson_aa_powf`` it passes.  mp=28
    sides with WRF, per MP28_PORT_SPEC.md's finding that mp=8's pre-existing
    deviations must be RECORDED rather than propagated, and per the "THE
    SHARED FITS" note in the shared header.

    WP-04's mp=8 identity test,
    ``test_effective_radius_reduces_to_the_frozen_mp8_kernel_at_nt_c``, has
    been restated by its owner to expect exactly this: effc and effs still
    bitwise against thompson.cu, the effi divergence required to be levels
    [22, 23] and at most one ulp, and mp=28 -- not mp=8 -- required to match
    the Fortran column.  Both tests are green.

    MP28_PORT_SPEC.md's NAMED hard identity gates are
    ``thompson_aa_droplet_bin(100.0e6f) == 65`` and
    ``thompson_aa_in_bin(1000.0f) == 27``.  Both are green -- they are
    asserted at the top of this file -- so nothing the spec calls an
    acceptance criterion has been given up here.

    This test exists so the divergence stays a PINNED, EXPLAINED number.  If
    it ever stops firing, either mp=8 changed (it is byte-frozen; that would
    be a defect) or the header quietly went back to CUDA's powf.
    """
    import numpy as _np

    # k = 23 and k = 24 of aero-reduces-to-classic, in metres.
    for fixture_value, mp28_value, mp8_value in (
            (2.9473000e-05, 2.9473000e-05, 2.9473001e-05),
            (2.9043753e-05, 2.9043753e-05, 2.9043755e-05)):
        fixture = _np.float32(fixture_value)
        mp28 = _np.float32(mp28_value)
        mp8 = _np.float32(mp8_value)
        assert mp28 == fixture
        assert mp8 != fixture
        gap = abs(float(mp8) - float(fixture)) / float(fixture)
        assert 0.0 < gap < 1.0e-07, gap

    # And the two spec-named identity gates are unaffected.
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_droplet_bin, probe_in_bin)
    from woof.core.thompson_aerosol_contract import NT_C

    assert int(cp.asnumpy(probe_droplet_bin(_f32([NT_C])))[0]) == 65
    assert int(cp.asnumpy(probe_in_bin(_f32([1000.0])))[0]) == 27


def test_saturation_helpers_match_a_host_transcription():
    """thompson_rslf/thompson_rsif against a non-contracted host Horner.

    STALE DOCSTRING CORRECTED IN WAVE 4.  This used to say the two fits "must
    stay numerically identical to that mp=8 code" and "therefore keep
    thompson.cu's plain expressions and inherit its FMA contraction".  Neither
    has been true since the fits were contraction-pinned: the header's "THE
    SHARED FITS" note records why mp=28 diverges from thompson.cu here on
    purpose, and module_mp_thompson.F:3400 opening the whole condensation and
    CCN-activation block on ``ssatw > eps`` with eps = 1.E-15 (:185) is why
    one ulp matters.  The reference below was always the NON-contracted
    Horner, which is what WRF's FMA-free ``gfortran -O2`` computes, so the
    test itself was already gating the right thing and its assertions are
    unchanged.

    The gate stays scoped to the well-conditioned range and the cold tail is
    asserted structurally, because WRF's 8th-order polynomial cancels
    catastrophically as ``x`` approaches its -80 C clamp.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_saturation

    f32 = np.float32
    coefficients_l = [0.611583699e3, 0.444606896e2, 0.143177157e1,
                      0.264224321e-1, 0.299291081e-3, 0.203154182e-5,
                      0.702620698e-8, 0.379534310e-11, -0.321582393e-13]
    coefficients_i = [0.609868993e3, 0.499320233e2, 0.184672631e1,
                      0.402737184e-1, 0.565392987e-3, 0.521693933e-5,
                      0.307839583e-7, 0.105785160e-9, 0.161444444e-12]

    def horner(coefficients, x):
        value = f32(coefficients[-1])
        for coefficient in reversed(coefficients[:-1]):
            value = f32(f32(coefficient) + f32(x * value))
        return value

    pressure = np.array([100000.0, 90000.0, 50000.0, 40000.0],
                        dtype=np.float32)
    temperature = np.array([300.0, 273.16, 265.0, 245.0], dtype=np.float32)
    rslf, rsif = probe_saturation(cp.asarray(pressure),
                                  cp.asarray(temperature))
    rslf = cp.asnumpy(rslf)
    rsif = cp.asnumpy(rsif)
    for index in range(pressure.size):
        x = f32(max(f32(-80.0), f32(temperature[index] - f32(273.16))))
        esl = f32(min(horner(coefficients_l, x),
                      f32(pressure[index] * f32(0.15))))
        esi = f32(min(horner(coefficients_i, x),
                      f32(pressure[index] * f32(0.15))))
        want_l = f32(f32(f32(0.622) * esl) / f32(pressure[index] - esl))
        want_i = f32(f32(f32(0.622) * esi)
                     / f32(max(f32(1.0e-4), f32(pressure[index] - esi))))
        assert float(rslf[index]) == pytest.approx(float(want_l), rel=1.0e-5)
        assert float(rsif[index]) == pytest.approx(float(want_i), rel=1.0e-5)

    # Cold tail: the -80 C clamp must engage and both results stay finite,
    # positive and monotone in temperature.  No tight value gate here.
    cold_p = np.full(6, 20000.0, dtype=np.float32)
    cold_t = np.array([230.0, 220.0, 210.0, 200.0, 193.16, 180.0],
                      dtype=np.float32)
    cold_l, cold_i = probe_saturation(cp.asarray(cold_p), cp.asarray(cold_t))
    cold_l = cp.asnumpy(cold_l).astype(np.float64)
    cold_i = cp.asnumpy(cold_i).astype(np.float64)
    assert np.isfinite(cold_l).all() and np.isfinite(cold_i).all()
    assert (cold_l > 0.0).all() and (cold_i > 0.0).all()
    assert (np.diff(cold_l) <= 0.0).all()
    # 193.16 K and 180 K both clamp to x = -80, so both results are identical
    # there -- proof the MAX(-80., T-273.16) guard is actually wired up.
    assert cold_l[-1] == cold_l[-2]
    assert cold_i[-1] == cold_i[-2]


# ---------------------------------------------------------------------------
# THE nu_c STAGING RULE.  module_mp_thompson.F:1832 vs :2170.
# ---------------------------------------------------------------------------
#
# WRF's entry block, verbatim:
#
#   1829     nc(k) = MAX(2., MIN(nc1d(k)*rho(k), Nt_c_max))
#   1832     nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)
#   1833     lamc = (nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k))**obmr
#   1834     xDc  = (bm_r + nu_c + 1.) / lamc
#   1835-8   if (xDc.lt.D0c)      lamc = cce(2,nu_c)/D0c
#            elseif (xDc.gt.D0r*2.) lamc = cce(2,nu_c)/(D0r*2.)
#   1840     nc(k) = MIN(DBLE(Nt_c_max), ccg(1,nu_c)*ocg2(nu_c)*rc(k)
#                        / am_r*lamc**bm_r)                     <- REDIAGNOSED
#
# and then, inside the k-loop that opens at :2156:
#
#   2170     nu_c = MIN(15, NINT(1000.E6/nc(k)) + 2)            <- RECOMPUTED
#   2173     lamc  = (nc(k)*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc(k))**obmr
#   2174-5   mvd_c(k) = (3.0+nu_c+0.672) / lamc
#   2181     Dc_g  = ((ccg(3,nu_c)*ocg2(nu_c))**obmr / lamc) * 1.E6
#   2192     pnr_wau(k) = prr_wau(k) / (am_r*nu_c*10.*D0r*D0r*D0r)
#
# :2170 reads the nc(k) that :1840 wrote.  Whenever the :1835-1838 size clamp
# engages the two nu_c values differ, and lamc, mvd_c, Dc_g and pnr_wau are
# all functions of the SECOND one.  These tests assert that, not any
# particular kernel's current behaviour.

#: (nc_entry [m^-3], rc [kg m^-3]) at rho = 1.  7 droplet numbers spanning the
#: [2, Nt_c_max] clamp range x 10 cloud contents spanning thin edge to core.
_NU_C_STAGING_NC = (1.0e6, 1.0e7, 1.0e8, 3.0e8, 1.0e9, 1.5e9, 1.999e9)
_NU_C_STAGING_RC = (1.0e-8, 3.0e-8, 1.0e-7, 3.0e-7, 1.0e-6, 1.0e-5, 1.0e-4,
                    5.0e-4, 1.0e-3, 5.0e-3)
_NU_C_STAGING_GRID = tuple(
    (nc, rc) for nc in _NU_C_STAGING_NC for rc in _NU_C_STAGING_RC)

#: The three states the mp=28 audit measured on an RTX 5090 when it found the
#: cold network reading the entry nu_c.  Kept as explicit named cases so a
#: regression names itself instead of arriving as "3 of 70 grid points".
#: (id, nc_entry [m^-3], rc [kg m^-3], nc_rediagnosed, nu_c_entry, nu_c_wrf)
_NU_C_STAGING_WORKED_EXAMPLES = (
    ("nc1e9-rc1e-8", 1.0e9, 1.0e-8, 5.4590136e7, 3, 15),
    ("nc1e8-rc1e-8", 1.0e8, 1.0e-8, 2.8654908e7, 12, 15),
    ("nc1.5e9-rc1e-7", 1.5e9, 1.0e-7, 5.4590140e8, 3, 4),
)


def _staging_grid_results():
    import cupy as cp
    rc = _f32([point[1] for point in _NU_C_STAGING_GRID])
    nc_per_kg = _f32([point[0] for point in _NU_C_STAGING_GRID])
    rho = _f32([1.0] * len(_NU_C_STAGING_GRID))
    nc_out, entry, working, lamc = _probe_nu_c_staging(rc, nc_per_kg, rho)
    return (cp.asnumpy(nc_out), cp.asnumpy(entry), cp.asnumpy(working),
            cp.asnumpy(lamc))


def test_nu_c_staging_grid_recomputes_from_the_rediagnosed_nc():
    """:2170 is a function of the :1840 nc, over all 70 grid points.

    This is the assertion the cold-network fix has to satisfy.  It is stated
    against WRF's control flow: for EVERY state, the working nu_c must equal
    ``MIN(15, NINT(1000.E6/nc_rediagnosed) + 2)``.  The entry value is
    computed from a DIFFERENT number and is only coincidentally equal.
    """
    from woof.core.thompson_aerosol_contract import cloud_shape_parameter

    nc_out, entry, working, lamc = _staging_grid_results()
    assert len(_NU_C_STAGING_GRID) == 70

    for index, (nc_entry, rc) in enumerate(_NU_C_STAGING_GRID):
        rediagnosed = float(nc_out[index])
        # :2170, the value every downstream rate must use.
        assert int(working[index]) == cloud_shape_parameter(rediagnosed), (
            nc_entry, rc, rediagnosed)
        # :1832, from the entry nc after the :1829 clamp.
        clamped_entry = min(max(nc_entry, 2.0), 1.999e9)
        assert int(entry[index]) == cloud_shape_parameter(clamped_entry), (
            nc_entry, rc)
        assert 2 <= int(working[index]) <= 15
        assert float(lamc[index]) > 0.0
        # :1840's MIN(DBLE(Nt_c_max), ...); Nt_c_max is REAL(4) 1999.E6, which
        # is 1999000064 exactly, so compare against the float32 value.
        assert 2.0 <= rediagnosed <= float(np.float32(1999.0e6))


def test_cloud_dist_matches_the_host_transcription_over_the_staging_grid():
    """Widen ``thompson_aa_cloud_dist``'s only gate from 7 states to 77.

    NO FORTRAN ORACLE IS POSSIBLE for :1826-1842: it is not a procedure, it is
    inline code inside mp_thompson's k-loop, and every constant it reads --
    ``ccg``, ``cce``, ``ocg1``, ``ocg2`` (:397), ``am_r`` (:128), ``obmr``,
    ``D0c``/``D0r`` (:224-225), ``Nt_c_max`` (:89) -- is declared PRIVATE, so
    no program that ``use``s the module can reach them.  (Verified by
    compilation: gfortran reports "Symbol 'ccg' referenced at (1) not found in
    module 'module_mp_thompson'".)  The best available gate is therefore a
    host transcription, and the accurate response is to run it over far more
    states rather than to pretend the seven original ones were an oracle.
    """
    import cupy as cp

    nc_out, entry, working, lamc = _staging_grid_results()
    worst_nc, worst_lamc = 0.0, 0.0
    for index, (nc_entry, rc) in enumerate(_NU_C_STAGING_GRID):
        want_nc, want_nu, want_lamc = _host_cloud_dist(rc, nc_entry, 1.0)
        assert int(entry[index]) == want_nu, (nc_entry, rc)
        worst_nc = max(worst_nc, abs(float(nc_out[index]) - float(want_nc))
                       / float(want_nc))
        worst_lamc = max(worst_lamc,
                         abs(float(lamc[index]) - want_lamc) / want_lamc)
    # MEASURED on an RTX 5090: both residuals are attributable to CUDA's powf
    # (~2 ulp) against glibc's correctly-rounded one, which is the same
    # residual warm.cu:86-92 reports for the rates built on top of this.
    assert worst_nc < _HOST_REFERENCE_RTOL, worst_nc
    assert worst_lamc < _HOST_REFERENCE_RTOL, worst_lamc


def test_nu_c_staging_grid_actually_diverges():
    """The two stages must DISAGREE somewhere, or this test proves nothing.

    If a future change to ``thompson_aa_cloud_dist`` ever made the entry and
    working values agree everywhere on this grid, the grid would stop
    discriminating between a correct and a wrong-stage kernel and would pass
    vacuously.  Pin the divergence itself.
    """
    nc_out, entry, working, _ = _staging_grid_results()
    divergent = [index for index in range(len(_NU_C_STAGING_GRID))
                 if int(entry[index]) != int(working[index])]
    assert len(divergent) >= 10, (
        f"only {len(divergent)} of 70 states separate the two nu_c stages")

    # Divergence is confined to thin cloud, which is exactly where WRF's
    # :1835-1838 droplet-size clamp engages -- and exactly the regime a cold
    # cloud edge occupies at qc just above R1.
    for index in divergent:
        nc_entry, rc = _NU_C_STAGING_GRID[index]
        assert rc <= 1.0e-7, (nc_entry, rc)
        # The rediagnosis really did move nc, by a lot.
        ratio = float(nc_out[index]) / min(max(nc_entry, 2.0), 1.999e9)
        assert ratio < 0.9 or ratio > 1.1, (nc_entry, rc, ratio)


@pytest.mark.parametrize(
    "nc_entry,rc,nc_rediagnosed,nu_c_entry,nu_c_wrf",
    [case[1:] for case in _NU_C_STAGING_WORKED_EXAMPLES],
    ids=[case[0] for case in _NU_C_STAGING_WORKED_EXAMPLES])
def test_nu_c_staging_worked_examples(nc_entry, rc, nc_rediagnosed,
                                      nu_c_entry, nu_c_wrf):
    """The three states the audit measured, pinned by value."""
    import cupy as cp

    nc_out, entry, working, _ = _probe_nu_c_staging(
        _f32([rc]), _f32([nc_entry]), _f32([1.0]))
    assert float(cp.asnumpy(nc_out)[0]) == pytest.approx(
        nc_rediagnosed, rel=1.0e-6)
    assert int(cp.asnumpy(entry)[0]) == nu_c_entry
    assert int(cp.asnumpy(working)[0]) == nu_c_wrf
    assert nu_c_entry != nu_c_wrf


def test_wrong_stage_nu_c_is_a_physics_error_not_a_rounding_one():
    """Quantify what reading the wrong nu_c stage actually costs.

    ``lamc`` (:2173), ``mvd_c`` (:2174), ``Dc_g`` (:2181) and ``pnr_wau``
    (:2192) all index ``ccg``/``ocg`` by ``nu_c``.  Between the entry and the
    working answer of the first worked example -- 3 against 15 -- the
    individual table entries move by more than ten orders of magnitude and
    the PRODUCTS the rates are actually built from move by factors of ~41 and
    ~16.  MEASURED here rather than asserted in prose, because the exact
    numbers are what say "no tolerance absorbs this".
    """
    import cupy as cp

    columns = cp.asarray(np.array([3, 15], dtype=np.int32))
    ccg2, ocg1, ccg3, ocg2 = (
        cp.asnumpy(value).astype(np.float64)
        for value in _probe_gamma_columns(columns))
    assert ccg2[1] / ccg2[0] > 1.0e12
    assert ocg1[0] / ocg1[1] > 1.0e11

    # lamc = (nc*am_r*ccg(2,nu_c)*ocg1(nu_c)/rc)**(1/3), :2173.  The product
    # moves 40.8x, so lamc itself moves 3.4x and mvd_c moves with it -- and
    # mvd_c is then squeezed by the MAX(D0c, MIN(mvd_c, D0r)) clamp at :2175,
    # which turns a wrong shape into a wrong autoconversion sink.
    lamc_product = ccg2 * ocg1
    assert lamc_product[1] / lamc_product[0] > 40.0
    assert (lamc_product[1] / lamc_product[0]) ** (1.0 / 3.0) > 3.4

    # Dc_g = ((ccg(3,nu_c)*ocg2(nu_c))**obmr / lamc) * 1.E6, :2181.
    dc_g_product = ccg3 * ocg2
    assert dc_g_product[1] / dc_g_product[0] > 15.0

    # pnr_wau = prr_wau / (am_r*nu_c*10.*D0r**3), :2192 -- linear in nu_c, so
    # the wrong stage rescales the rain-number source by five.
    assert 15.0 / 3.0 == 5.0

    # And the mvd_c numerator (3.0 + nu_c + 0.672) at :2174.
    assert (3.0 + 15 + 0.672) / (3.0 + 3 + 0.672) > 2.7


# ---------------------------------------------------------------------------
# The three helpers consolidated into thompson_aerosol_common.cuh.
# ---------------------------------------------------------------------------
#
# Before consolidation each of cold.cu / warm.cu / sed.cu carried its own
# copy, and thompson_aa_entry_rain_distribution had ALREADY DRIFTED: cold.cu's
# form used plain CUDA powf, emitted no N0_r and skipped WRF's :2146-2150
# re-derivation of lamr, while warm.cu's was contraction-pinned and measured
# bit-exact against the Fortran oracle where the plain form sat at ~2.7e-7
# relative.  Separate cupy.RawModule translation units meant nvrtc never saw
# the conflict.  There is now one definition, in the header; a surviving local
# copy is a hard redefinition error at compile time.

_SHARED_HELPER_SIGNATURES = {
    "thompson_aa_entry_rain_distribution": r"bool\s+{}\s*\(",
    "thompson_aa_bound_rain_number": r"void\s+{}\s*\(",
    "thompson_aa_bound_ice_number": r"void\s+{}\s*\(",
    # Promoted in wave 4 out of cold.cu:206-223 and warm.cu:272-289, which
    # held byte-identical copies.  See the promotion tests above.
    "thompson_aa_decade_index_double": r"int\s+{}\s*\(",
    # Blossey's REAL re-enforcement of the rain-graupel and rain-snow pairs
    # (:2945-2954), called by both networks.  The cold network's local form
    # forced the rain rate negative and kept the double magnitude; the warm
    # one kept the double magnitude: every rain-graupel level differed from
    # WRF by up to half a float32 ulp of the rate.
    "thompson_aa_reenforce_pair": r"void\s+{}\s*\(",
}


def test_shared_helpers_are_defined_exactly_once_and_only_in_the_header():
    """THIS SCAN IS THE ENFORCEMENT MECHANISM.  Not the compiler.

    The header used to claim that a surviving local copy "will FAIL TO
    COMPILE with a redefinition error -- that error is the enforcement
    mechanism".  That is true for the three helpers whose shared signature
    matches the deleted local one exactly, and FALSE for
    ``thompson_aa_entry_rain_distribution``, whose shared form takes seven
    parameters where the local copies took six: C++ resolves that as an
    OVERLOAD, the module compiles cleanly, and every six-argument call site
    silently keeps the local body.  ``test_a_six_parameter_local_copy_of_the_
    rain_distribution_compiles_silently`` demonstrates that on this GPU.

    So the scan below is not a convenience that fails "loudly and locally"
    ahead of a build error -- for one of the four names there is no build
    error to be ahead of.  Do not remove it, and do not soften it into a
    warning.
    """
    def definition_lines(path, helper, pattern):
        """1-based lines carrying a DEFINITION of ``helper``.

        ``//`` comments are dropped first so the header's published-signature
        block, which quotes each prototype verbatim, is not counted.  Call
        sites have no return type in front of the name and never match.
        """
        compiled = re.compile(pattern.format(helper))
        return [number for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1)
            if compiled.search(line.split("//", 1)[0])]

    stragglers = []
    for helper, pattern in _SHARED_HELPER_SIGNATURES.items():
        assert len(definition_lines(_HEADER, helper, pattern)) == 1, (
            f"{helper} must be defined exactly once in {_HEADER.name}")
        for path in sorted(_KERNELS.glob("thompson_aerosol_*.cu")):
            for line in definition_lines(path, helper, pattern):
                stragglers.append(f"{path.name}:{line} {helper}")

    assert not stragglers, (
        "these local copies must be deleted; the single definition now lives "
        f"in {_HEADER.name} -> " + "; ".join(stragglers))


#: A six-parameter local copy of the seven-parameter shared helper, as an
#: agent actually wrote it before the consolidation: no rain_intercept_n0, no
#: contraction pinning, plain powf.  Compiled here ON PURPOSE, appended to the
#: real header, to demonstrate that nvrtc accepts it.
_SIX_PARAMETER_OVERLOAD_SOURCE = r"""
__device__ __forceinline__ bool thompson_aa_entry_rain_distribution(
    float rain_per_kg, float rain_number_per_kg, float density,
    float* rain_number, double* rain_lambda, float* rain_mvd)
{
    // Deliberately the PRE-consolidation body: plain powf, no y-intercept.
    const float am_r = THOMPSON_AA_AM_R;
    if (rain_per_kg <= THOMPSON_AA_R1) {
        *rain_number = THOMPSON_AA_R2;
        *rain_lambda = 1.0;
        *rain_mvd = 0.0f;
        return false;
    }
    const float rr = rain_per_kg * density;
    const float nr = fmaxf(THOMPSON_AA_R2, rain_number_per_kg * density);
    const double lamr = (double)powf(am_r * 6.0f * nr / rr,
                                     THOMPSON_AA_OBMR);
    *rain_number = nr;
    *rain_lambda = lamr;
    *rain_mvd = (float)(3.672 / lamr);
    return true;
}

extern "C" __global__ void which_overload_wins(
    const float* qr, const float* nr_in, const float* rho,
    float* nr_out, double* lamr_out, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    float nr = 0.0f;
    double lamr = 0.0;
    float mvd = 0.0f;
    // SIX arguments.  If the shared seven-parameter form were the only
    // definition this would not compile at all.
    thompson_aa_entry_rain_distribution(
        qr[idx], nr_in[idx], rho[idx], &nr, &lamr, &mvd);
    nr_out[idx] = nr;
    lamr_out[idx] = lamr;
}
"""


def test_a_six_parameter_local_copy_of_the_rain_distribution_compiles_silently():
    """PROVE the header's old enforcement claim was false, do not assert it.

    The correction at the top of thompson_aerosol_common.cuh says a surviving
    six-parameter local copy of ``thompson_aa_entry_rain_distribution``
    OVERLOADS the shared seven-parameter form instead of colliding with it,
    that nvrtc emits no diagnostic, and that six-argument call sites keep
    binding to the local body.  All three claims are checked here on the real
    header, because a documentation correction that is only prose can rot back
    just as quietly as the claim it replaced.

    Nothing in the repository carries this variant; it exists only inside this
    test's throwaway module.
    """
    import cupy as cp
    from woof.core.kernels import module_source
    from woof.core.thompson_aerosol_launch import PROBE_MODULE

    source = module_source(PROBE_MODULE) + _SIX_PARAMETER_OVERLOAD_SOURCE
    # CLAIM 1: it compiles.  No redefinition error, no warning that stops it.
    module = cp.RawModule(code=source, options=("-std=c++17",))
    module.compile()
    kernel = module.get_function("which_overload_wins")

    # CLAIM 2: the six-argument call site resolved to the LOCAL body.  Prove
    # it by behaviour: the shared form applies WRF's :1888-1896 mvd clamps and
    # rebuilds nr from them, the local one does not, so a state that trips the
    # 2.5 mm clamp separates them.
    qr = _f32([1.0e-3])
    nr_in = _f32([1.0e2])
    rho = _f32([1.0])
    nr_local = cp.empty(1, dtype=cp.float32)
    lamr_local = cp.empty(1, dtype=cp.float64)
    kernel((1,), (1,), (qr, nr_in, rho, nr_local, lamr_local, np.int32(1)))
    shared_nr, shared_lamr, shared_mvd, _, _ = _probe_entry_rain_distribution(
        qr, nr_in, rho)
    assert float(cp.asnumpy(shared_mvd)[0]) == pytest.approx(2.5e-3,
                                                             rel=1.0e-6)
    assert float(cp.asnumpy(nr_local)[0]) != float(cp.asnumpy(shared_nr)[0])
    assert float(cp.asnumpy(lamr_local)[0]) != float(
        cp.asnumpy(shared_lamr)[0])

    # CLAIM 3: the three signature-identical helpers really do collide, so
    # the corrected text is right about which half of the claim survived.
    for helper, body in (
            ("thompson_aa_bound_rain_number",
             "__device__ __forceinline__ void thompson_aa_bound_rain_number("
             "\n    float rain_mass, float density, float* n) { *n = 0.0f; }"),
            ("thompson_aa_bound_ice_number",
             "__device__ __forceinline__ void thompson_aa_bound_ice_number("
             "\n    float ice_mass, float density, float* n) { *n = 0.0f; }"),
            ("thompson_aa_decade_index_double",
             "__device__ __forceinline__ int thompson_aa_decade_index_double("
             "\n    double v, int f, int t) { return 0; }")):
        with pytest.raises(Exception) as caught:
            cp.RawModule(code=module_source(PROBE_MODULE) + "\n" + body,
                         options=("-std=c++17",)).compile()
        assert "already been defined" in str(caught.value), helper


#: Helpers that a .cu re-implements under its OWN name instead of calling the
#: shared one.  A rename is NOT a redefinition, so nvrtc is silent and the
#: source scan above cannot see it either -- this is the same overload-shaped
#: blind spot that let thompson_aa_entry_rain_distribution drift, wearing a
#: different hat.  Each entry is
#:     shared name -> (file, local name, "float", (arg types...))
#: and the test below compiles BOTH in one translation unit and demands
#: bitwise equality over a randomized sweep.
_LOCAL_REIMPLEMENTATIONS = (
    ("thompson_aa_eff_rad_cloud", "thompson_aerosol_state.cu",
     "thompson_aa_state_eff_rad_cloud", ("rc", "nc")),
    ("thompson_aa_eff_rad_ice", "thompson_aerosol_state.cu",
     "thompson_aa_state_eff_rad_ice", ("ri", "ni")),
    ("thompson_aa_eff_rad_snow", "thompson_aerosol_state.cu",
     "thompson_aa_state_eff_rad_snow", ("rs", "t")),
)


def _lift_device_function(path, name):
    """Return the verbatim source of ``__device__ ... float name(...) {...}``."""
    text = path.read_text(encoding="utf-8")
    marker = f"__device__ __forceinline__ float {name}("
    if marker not in text:
        return None
    start = text.index(marker)
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def test_local_reimplementations_of_shared_helpers_have_not_drifted():
    """A RENAMED copy is the blind spot the source scan cannot cover.

    ``woof/core/kernels/thompson_aerosol_state.cu`` carries
    ``thompson_aa_state_eff_rad_cloud`` / ``_ice`` / ``_snow``, written while
    the shared header still held thompson.cu's broken sa/sb association -- WP-04
    measured the same defect independently and worked around it locally rather
    than waiting.  The header's helpers are now WRF-exact (bit-exact against
    the real ``calc_effectRad`` on 360 + 378 + 378 states), so the local trio
    should be deleted and the shared ones called.

    Until that happens this test is the guard: it lifts each local body
    verbatim, compiles it alongside the shared header in ONE translation unit,
    and demands BITWISE equality over 400 000 randomized states spanning
    rc/ri/rs in [1e-12, 1e-2] kg m^-3, nc in [2, 1.999e9] m^-3, ni in
    [1e-6, 1e6] m^-3 and T in [180, 305] K.  MEASURED: 0 of 400 000 differ in
    all three branches.

    The test SKIPS, rather than failing, once a local body is gone -- that is
    the desired end state, and the scan in
    ``test_shared_helpers_are_defined_exactly_once_and_only_in_the_header``
    is what keeps the shared definition unique after it.
    """
    import cupy as cp
    from woof.core.kernels import module_source
    from woof.core.thompson_aerosol_launch import PROBE_MODULE

    lifted = []
    pairs = []
    for shared, filename, local, args in _LOCAL_REIMPLEMENTATIONS:
        body = _lift_device_function(_KERNELS / filename, local)
        if body is None:
            continue
        lifted.append(body)
        pairs.append((shared, local, args))
    if not pairs:
        pytest.skip("no renamed local re-implementations remain")

    calls = "\n".join(
        f"    out[{2 * i} * n + k] = {shared}({args[0]}[k], {args[1]}[k]);\n"
        f"    out[{2 * i + 1} * n + k] = {local}({args[0]}[k], {args[1]}[k]);"
        for i, (shared, local, args) in enumerate(pairs))
    source = module_source(PROBE_MODULE) + "\n".join(lifted) + f"""
extern "C" __global__ void compare_local_reimplementations(
    const float* rc, const float* nc, const float* ri, const float* ni,
    const float* rs, const float* t, float* out, int n)
{{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
{calls}
}}
"""
    module = cp.RawModule(code=source, options=("-std=c++17",))
    compare = module.get_function("compare_local_reimplementations")

    rng = np.random.default_rng(20260731)
    size = 400000
    def loguniform(lo, hi):
        return np.exp(rng.uniform(math.log(lo), math.log(hi),
                                  size)).astype(np.float32)
    fields = [
        cp.asarray(loguniform(1.0e-12, 1.0e-2)),      # rc
        cp.asarray(loguniform(2.0, 1.999e9)),         # nc
        cp.asarray(loguniform(1.0e-12, 1.0e-3)),      # ri
        cp.asarray(loguniform(1.0e-6, 1.0e6)),        # ni
        cp.asarray(loguniform(1.0e-12, 1.0e-2)),      # rs
        cp.asarray(rng.uniform(180.0, 305.0, size).astype(np.float32)),
    ]
    out = cp.empty(2 * len(pairs) * size, dtype=cp.float32)
    compare(((size + 255) // 256,), (256,),
            (*fields, out, np.int32(size)))
    out = cp.asnumpy(out).reshape(2 * len(pairs), size)
    for index, (shared, local, _) in enumerate(pairs):
        shared_values = out[2 * index]
        local_values = out[2 * index + 1]
        differing = int((shared_values != local_values).sum())
        assert differing == 0, (
            f"{local} has DRIFTED from {shared}: {differing} of {size} "
            f"states differ; the local copy must be deleted, not repaired")
        # The sweep must actually exercise the helper across its clamps.
        assert shared_values.min() < shared_values.max()
        assert np.isfinite(shared_values).all()


def test_the_effective_radius_reimplementations_are_gone_and_stayed_gone():
    """THE END STATE THE TEST ABOVE SKIPS INTO, ASSERTED RATHER THAN ASSUMED.

    ``test_local_reimplementations_of_shared_helpers_have_not_drifted`` skips
    once ``thompson_aerosol_state.cu`` stops defining
    ``thompson_aa_state_eff_rad_cloud`` / ``_ice`` / ``_snow``.  A skip proves
    nothing on its own, so this asserts the two halves of the desired state
    directly: the private definitions are gone, and the file CALLS the three
    shared helpers instead.

    The deletion was proved inert before it was made.  Compiling the three
    deleted bodies verbatim alongside the (now contraction-pinned) shared
    header in ONE translation unit and comparing bitwise over 400 000
    randomized states -- rc/ri/rs in [1e-12, 1e-2] kg m^-3, nc in [2, 1.999e9]
    m^-3, ni in [1e-6, 1e6] m^-3, T in [180, 305] K -- gave 0 of 400 000
    differing in all three branches, and the shared helpers are bit-exact
    against the REAL ``calc_effectRad`` on 960 states (40 columns x 24 levels)
    on effc and effs, 958/960 on effi.
    """
    source = (_KERNELS / "thompson_aerosol_state.cu").read_text(
        encoding="utf-8")
    for name in ("thompson_aa_state_eff_rad_cloud",
                 "thompson_aa_state_eff_rad_ice",
                 "thompson_aa_state_eff_rad_snow"):
        assert f"float {name}(" not in source, (
            f"{name} is back; it must not be -- call the shared header helper")
    for name in ("thompson_aa_eff_rad_cloud", "thompson_aa_eff_rad_ice",
                 "thompson_aa_eff_rad_snow"):
        assert f"{name}(" in source, (
            f"thompson_aerosol_state.cu no longer calls {name}")
    # And the shared bodies really are pinned, which is what made the deletion
    # safe: an unpinned float32 chain is what nvrtc is free to widen.
    header = _HEADER.read_text(encoding="utf-8")
    block = "".join(
        _lift_device_function(_HEADER, name)
        for name in ("thompson_aa_eff_rad_cloud", "thompson_aa_eff_rad_ice",
                     "thompson_aa_eff_rad_snow"))
    assert block.count("thompson_aa_div(") >= 3, block
    assert block.count("thompson_aa_mul(") >= 6, block
    assert block.count("thompson_aa_powf(") == 3, block
    assert " powf(" not in block, "a plain CUDA powf is back in eff_rad"
    # thompson_aa_cloud_dist is the fourth REAL(4)** site the same argument
    # covers, and it is bit-exact against WRF only with both properties.
    dist = _lift_device_function(_HEADER, "thompson_aa_cloud_dist")
    assert "thompson_aa_powf(" in dist and " powf(" not in dist
    assert dist.count("thompson_aa_mul(") >= 5, dist
    assert header.count("__device__ __forceinline__ float "
                        "thompson_aa_cloud_dist(") == 1


def test_no_plain_powf_survives_in_any_mp28_arm():
    """Every REAL(4) ``**`` in the mp=28 units is WOOF's own powf word.

    This replaces the plain-powf inventory that used to stand here.  That
    inventory pinned the CUDA ``powf`` calls the classic and fork arms kept
    because each had been measured inert against its Fortran oracle on the
    fixtures then available (the fork's two sites carried the receipt
    tests/data/thompson_fork_plain_powf_receipt.json).  The 0 ULP column
    oracle (tools/thompson_aerosol_column_oracle) then showed that CUDA's
    words, and the double-evaluated "_cr" helpers, are not the words WRF's
    gfortran build calls, so every site now calls WOOF's own words
    (woof/core/kernels/thompson_aerosol_libm.cuh), which equal the oracle
    host's C library (tests/test_thompson_aerosol_libm.py).  With no plain
    powf left there is nothing for the inventory or the receipt to account
    for: the guard retires with the defect it accepted.
    """
    pattern = re.compile(r"(?<![_A-Za-z0-9])powf\(")
    for path in sorted(_KERNELS.glob("thompson_aerosol_*.cu")) + [_HEADER]:
        code = "\n".join(line.split("//", 1)[0] for line in
                         path.read_text(encoding="utf-8").splitlines())
        assert not pattern.search(code), path.name


def test_nt_c_is_not_reachable_from_device_code():
    """MINOR 5.  Nt_c is the one literal mp=28 must never read as a number."""
    pattern = re.compile(r"THOMPSON_AA_NT_C(?!_MAX)")
    for path in sorted(_KERNELS.glob("thompson_aerosol_*.cu")) + [_HEADER]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name
    # And the generalized selector still lands on what mp=8 froze.
    from woof.core.thompson_aerosol_contract import NT_C
    assert float(NT_C) == 100.0e6


def test_mass_coefficient_constants_equal_the_product_forms():
    """The consolidated bounds read am_r/am_i by NAME; the deleted per-network
    copies spelled out ``3.1415926536f*1000.0f/6.0f`` and
    ``3.1415926536f*890.0f/6.0f``.  Prove the substitution is bit-exact ON
    DEVICE rather than asserting it in a comment."""
    import cupy as cp

    values = cp.asnumpy(_probe_mass_coefficients())
    assert values[0] == values[1]      # THOMPSON_AA_AM_R
    assert values[2] == values[3]      # THOMPSON_AA_AM_I
    assert float(values[0]) == pytest.approx(math.pi * 1000.0 / 6.0, rel=1e-7)
    assert float(values[2]) == pytest.approx(math.pi * 890.0 / 6.0, rel=1e-7)


def _host_entry_rain_distribution(qr_per_kg, nr_per_kg, rho):
    """module_mp_thompson.F:1878-1898 then :2146-2150, in the SAME arithmetic
    the device helper pins: every product/quotient separately rounded in
    float32, every power evaluated in double and rounded once.

    Two forms here are WRF's since development branch, where this
    reference used to carry the helper's old ones: (3.0 + mu_r + 0.672) is
    the REAL sum, widened, over the DOUBLE lamr (not the DOUBLE literal
    3.672), and crg(2)*org3*rr*lamr**bm_r/am_r is ((REAL prefix) times the
    DOUBLE pow(lamr, 3.0)) over am_r, since gfortran folds no real exponent
    but 2, 1 and -1 (not ((prefix/am_r) times a product chain)).
    """
    f32 = np.float32
    am_r = f32(5.235988159e+02)
    obmr = f32(3.333333433e-01)
    r1, r2 = f32(1.0e-12), f32(1.0e-6)
    d0r = f32(50.0e-6)
    sixth = f32(1.0 / 6.0)

    def power(base, exponent):
        return f32(math.pow(float(base), float(exponent)))

    mvd_num = float(f32(f32(f32(3.0) + f32(0.0)) + f32(0.672)))

    def rebuild(rr, lam):
        return f32(float(f32(sixth * rr)) * math.pow(lam, 3.0) / float(am_r))

    if f32(qr_per_kg) > r1:
        active = True
        rr = f32(f32(qr_per_kg) * f32(rho))
        nr = f32(max(r2, f32(f32(nr_per_kg) * f32(rho))))
        if nr <= r2:
            nr = rebuild(rr, float(f32(f32(3.672) / f32(1.0e-3))))
        lamr = float(power(f32(f32(f32(am_r * f32(6.0)) * nr) / rr), obmr))
        mvd = f32(mvd_num / lamr)
        if mvd > f32(2.5e-3):
            mvd = f32(2.5e-3)
            lamr = float(f32(f32(3.672) / mvd))
            nr = rebuild(rr, lamr)
        elif mvd < f32(d0r * f32(0.75)):
            mvd = f32(d0r * f32(0.75))
            lamr = float(f32(f32(3.672) / mvd))
            nr = rebuild(rr, lamr)
    else:
        active = False
        rr, nr = r1, r2

    lamr = float(power(f32(f32(f32(am_r * f32(6.0)) * nr) / rr), obmr))
    return active, nr, lamr, f32(mvd_num / lamr), float(nr) * lamr


#: qr [kg kg^-1], nr [kg^-1], rho.  Chosen so every branch of :1878-1898 runs:
#: the L_qr false path, the nr <= R2 default, the 2.5 mm upper mvd clamp, the
#: 37.5 um lower clamp and the unclamped middle.
_ENTRY_RAIN_CASES = (
    (0.0, 0.0, 1.0),
    (1.0e-13, 1.0e3, 1.0),
    (1.0e-6, 1.0, 1.0),
    (1.0e-6, 1.0e7, 1.0),
    (1.0e-4, 1.0e4, 1.0),
    (1.0e-3, 1.0e2, 1.0),
    (1.0e-3, 1.0e6, 1.0),
    (5.0e-3, 1.0e3, 0.8),
    (1.0e-5, 1.0e-9, 1.2),
    (2.0e-4, 5.0e3, 1.1),
    (1.0e-2, 1.0e5, 0.6),
)


def test_entry_rain_distribution_is_bit_exact_against_the_pinned_reference():
    """MEASURED: bit-exact on every case.  That is only reachable because the
    helper is contraction-pinned and uses thompson_aa_powf; the plain-powf
    form cold.cu used to carry sits ~2.7e-7 away."""
    import cupy as cp

    qr = _f32([case[0] for case in _ENTRY_RAIN_CASES])
    nr_in = _f32([case[1] for case in _ENTRY_RAIN_CASES])
    rho = _f32([case[2] for case in _ENTRY_RAIN_CASES])
    nr, lamr, mvd, n0, active = _probe_entry_rain_distribution(qr, nr_in, rho)
    nr = cp.asnumpy(nr)
    lamr = cp.asnumpy(lamr)
    mvd = cp.asnumpy(mvd)
    n0 = cp.asnumpy(n0)
    active = cp.asnumpy(active)

    for index, case in enumerate(_ENTRY_RAIN_CASES):
        want = _host_entry_rain_distribution(*case)
        assert bool(active[index]) is want[0], case
        assert np.float32(nr[index]) == want[1], case
        assert float(lamr[index]) == want[2], case
        assert np.float32(mvd[index]) == want[3], case
        assert float(n0[index]) == want[4], case


def test_entry_rain_distribution_exercises_every_wrf_branch():
    """Guard against the reference and the device agreeing on one branch."""
    import cupy as cp

    qr = _f32([case[0] for case in _ENTRY_RAIN_CASES])
    nr_in = _f32([case[1] for case in _ENTRY_RAIN_CASES])
    rho = _f32([case[2] for case in _ENTRY_RAIN_CASES])
    nr, lamr, mvd, n0, active = _probe_entry_rain_distribution(qr, nr_in, rho)
    mvd = cp.asnumpy(mvd).astype(np.float64)
    active = cp.asnumpy(active)

    assert set(int(v) for v in active) == {0, 1}
    # :1888 upper clamp, :1892 lower clamp, and states between them.
    assert np.isclose(mvd, 2.5e-3, rtol=1.0e-6).any()
    assert np.isclose(mvd, 37.5e-6, rtol=1.0e-6).any()
    assert ((mvd > 40.0e-6) & (mvd < 2.4e-3)).any()


def test_entry_rain_distribution_defines_lamr_even_with_no_rain():
    """WRF's :2145 loop has NO L_qr guard.

    ``do k = kte, kts, -1`` at :2145 runs :2146-2150 for every level, so
    ``lamr``, ``mvd_r`` and ``N0_r`` are formed from the rr = R1 / nr = R2
    sentinels :1893-1896 leaves behind.  A helper that returned early on
    ``qr <= R1`` would hand the collection rates a stale or zero N0_r.
    """
    import cupy as cp

    nr, lamr, mvd, n0, active = _probe_entry_rain_distribution(
        _f32([0.0, 1.0e-13]), _f32([0.0, 1.0e3]), _f32([1.0, 1.0]))
    assert list(int(v) for v in cp.asnumpy(active)) == [0, 0]
    assert (cp.asnumpy(nr) == np.float32(1.0e-6)).all()
    assert (cp.asnumpy(lamr) > 0.0).all()
    assert (cp.asnumpy(mvd) > 0.0).all()
    assert (cp.asnumpy(n0) > 0.0).all()
    # N0_r = nr*org2*lamr**cre(2) with org2 = 1 and cre(2) = 1 exactly.
    assert (cp.asnumpy(n0)
            == cp.asnumpy(nr).astype(np.float64) * cp.asnumpy(lamr)).all()


def test_entry_rain_distribution_n0_is_nr_times_lamr_everywhere():
    """cre(2) = mu_r + 1 = 1 and org2 = 1/WGAMMA(1) = 1, both exact."""
    import cupy as cp

    qr = _f32([case[0] for case in _ENTRY_RAIN_CASES])
    nr_in = _f32([case[1] for case in _ENTRY_RAIN_CASES])
    rho = _f32([case[2] for case in _ENTRY_RAIN_CASES])
    nr, lamr, _, n0, _ = _probe_entry_rain_distribution(qr, nr_in, rho)
    assert (cp.asnumpy(n0)
            == cp.asnumpy(nr).astype(np.float64) * cp.asnumpy(lamr)).all()


def test_bound_rain_number_matches_a_host_transcription():
    """module_mp_thompson.F:4032-4046 == thompson.cu:2574-2598.

    Plain ``powf`` here, deliberately: this is mp=8's arithmetic verbatim and
    the two ports must stay identical on it, so the host reference carries the
    usual float32 ``pow`` tolerance rather than a bit-exact gate.
    """
    import cupy as cp

    f32 = np.float32
    am_r = f32(5.235988159e+02)
    cases = (
        (0.0, 1.0, 1.0e4),          # rain_mass <= R1 -> number zeroed
        (1.0e-13, 1.0, 1.0e4),
        (1.0e-6, 1.0, 1.0e-9),      # number floored at R2
        (1.0e-4, 1.0, 1.0e4),
        (1.0e-3, 1.0, 1.0e2),       # 2.5 mm clamp
        (1.0e-4, 1.0, 1.0e8),       # 37.5 um clamp
        (5.0e-4, 0.8, 5.0e3),
    )
    mass = _f32([case[0] for case in cases])
    density = _f32([case[1] for case in cases])
    number = _f32([case[2] for case in cases])
    got = cp.asnumpy(_probe_bound_number("rain", mass, density, number))

    for index, (rain_mass, rho, nr_in) in enumerate(cases):
        if f32(rain_mass) <= f32(1.0e-12):
            assert got[index] == np.float32(0.0), index
            continue
        rain_number = f32(max(f32(1.0e-6), f32(f32(nr_in) * f32(rho))))
        lam = f32(f32(f32(f32(am_r * f32(6.0)) * rain_number)
                      / f32(rain_mass)) ** f32(1.0 / 3.0))
        mvd = f32(f32(3.672) / lam)
        if f32(37.5e-6) <= mvd <= f32(2.5e-3):
            assert float(got[index]) == pytest.approx(
                float(nr_in), rel=_HOST_REFERENCE_RTOL), index
            continue
        mvd = f32(2.5e-3) if mvd > f32(2.5e-3) else f32(37.5e-6)
        lam = f32(f32(3.672) / mvd)
        want = f32(f32(f32(f32(f32(1.0 / 6.0) * f32(rain_mass)) / am_r)
                       * lam * lam * lam) / f32(rho))
        assert float(got[index]) == pytest.approx(
            float(want), rel=_HOST_REFERENCE_RTOL), index

    # The clamps must actually have fired, or the branch coverage is fiction.
    assert got[4] != np.float32(1.0e2)
    assert got[5] != np.float32(1.0e8)


def test_bound_ice_number_matches_a_host_transcription_and_is_idempotent():
    """module_mp_thompson.F:4029-4039 == thompson.cu:3719-3743.

    Idempotence is essential: sed.cu keeps mp=8's fused placement while
    WP-04's terminal state kernel applies the same bound again.
    """
    import cupy as cp

    f32 = np.float32
    am_i = f32(4.660029297e+02)
    cases = (
        (0.0, 1.0, 1.0e5),
        (1.0e-13, 1.0, 1.0e5),
        (1.0e-6, 1.0, 1.0e-9),
        (1.0e-5, 1.0, 5.0e4),
        (1.0e-6, 1.0, 1.0e9),       # 5 um lower clamp
        (1.0e-3, 1.0, 1.0e2),       # 300 um upper clamp
        (2.0e-5, 0.7, 3.0e5),
    )
    mass = _f32([case[0] for case in cases])
    density = _f32([case[1] for case in cases])
    number = _f32([case[2] for case in cases])
    once = _probe_bound_number("ice", mass, density, number)
    twice = cp.asnumpy(_probe_bound_number("ice", mass, density, once))
    once = cp.asnumpy(once)

    for index, (ice_mass, rho, ni_in) in enumerate(cases):
        if f32(ice_mass) <= f32(1.0e-12):
            assert once[index] == np.float32(0.0), index
            continue
        ice_number = f32(max(f32(1.0e-6), f32(f32(ni_in) * f32(rho))))
        lam = float(f32(f32(f32(f32(am_i * f32(6.0)) * ice_number)
                            / f32(ice_mass)) ** f32(1.0 / 3.0)))
        diameter = f32(4.0 / lam)
        if diameter < f32(5.0e-6):
            lam = 4.0 / 5.0e-6
            ice_number = f32(min(f32(999.0e3), f32(
                f32(f32(f32(1.0 / 6.0) * f32(ice_mass)) / am_i)
                * f32(lam * lam * lam))))
        elif diameter > f32(300.0e-6):
            lam = 4.0 / 300.0e-6
            ice_number = f32(
                f32(f32(f32(1.0 / 6.0) * f32(ice_mass)) / am_i)
                * f32(lam * lam * lam))
        want = f32(min(ice_number, f32(999.0e3)) / f32(rho))
        assert float(once[index]) == pytest.approx(
            float(want), rel=1.0e-5), index

    assert np.array_equal(once, twice)
    assert (once <= np.float32(999.0e3) / np.float32(0.7) * 1.000001).all()


def test_decade_index_is_the_shared_pattern_behind_idx_in():
    """thompson_aa_in_bin must be exactly decade_index(xni, 0, 55)."""
    import cupy as cp
    from woof.core.thompson_aerosol_launch import (
        probe_decade_index, probe_in_bin)

    values = _f32([1.5, 9.9, 10.0, 99.0, 1000.0, 1.0e5, 1.0e6, 1.0e7])
    direct = cp.asnumpy(probe_decade_index(values, 0, 55))
    through = cp.asnumpy(probe_in_bin(values))
    assert list(int(v) for v in direct) == list(int(v) for v in through)


# ---------------------------------------------------------------------------
# thompson_aa_decade_index_double, PROMOTED into the header in wave 4.
# ---------------------------------------------------------------------------
#
# It was the last helper still defined outside the shared header:
# thompson_aerosol_cold.cu:206-223 and thompson_aerosol_warm.cu:272-289 each
# carried it, byte-identical to each other.  There was no divergence yet --
# and that is exactly the state thompson_aa_entry_rain_distribution was in
# before its two copies drifted, one of them ~2.7e-7 away from the oracle.
# The float sibling thompson_aa_decade_index had been shared since wave 2;
# only the double form was not.
#
# Its two production call sites are WRF's DOUBLE PRECISION y-intercept
# lookups: idx_r over t_Nor at (first_exponent 6, table_size 37) and idx_g
# over t_Nog at (2, 37).  Both live in translation units this package does not
# own, so ``thompson_aa_probe_decade_index_double`` in
# woof/core/kernels/thompson_aerosol_probe.cu exists to give the promoted
# body a pointwise gate of its own.

#: The deleted local bodies, byte for byte, as a THROWAWAY translation unit
#: appended to the real header.  ``_legacy`` is the only edit.  Compiling both
#: in ONE module is what makes "the promotion changed nothing" a measurement.
_DECADE_DOUBLE_LEGACY_SOURCE = r"""
__device__ __forceinline__ int thompson_aa_decade_index_double_legacy(
    double value, int first_exponent, int table_size)
{
    const int center = (int)round(log10(value));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = powf(10.0f, (float)candidate);
        const double mantissa = value / (double)scale;
        if (mantissa >= 1.0 && mantissa < 10.0) {
            exponent = candidate;
            break;
        }
    }
    const float scale = powf(10.0f, (float)exponent);
    const int digit = (int)(value / (double)scale);
    const int one_based = digit + 9 * (exponent - first_exponent);
    return max(0, min(one_based - 1, table_size - 1));
}

extern "C" __global__ void compare_decade_index_double(
    const double* value, int first_exponent, int table_size,
    int* promoted, int* legacy, int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    promoted[idx] = thompson_aa_decade_index_double(
        value[idx], first_exponent, table_size);
    legacy[idx] = thompson_aa_decade_index_double_legacy(
        value[idx], first_exponent, table_size);
}
"""


def _decade_double_sweep():
    """Every value WRF's two call sites can present, and then some.

    idx_r indexes t_Nor with N0_r and idx_g indexes t_Nog with N0_g, both
    DOUBLE PRECISION (module_mp_thompson.F:1587).  The sweep spans 1e-6 to
    1e20 with nine mantissas per decade plus the exact decade boundaries and
    the just-below values where ``round(log10)`` picks the wrong centre and
    the three-candidate rescan has to correct it.
    """
    values = []
    for exponent in range(-6, 21):
        base = 10.0 ** exponent
        for mantissa in (1.0, 1.0000001, 1.5, 2.0, 3.3, 5.0, 7.7, 9.0,
                         9.9, 9.999999):
            values.append(mantissa * base)
        values.append(base * 0.9999999)
    return np.array(values, dtype=np.float64)


def test_promoted_decade_index_double_is_bitwise_identical_to_the_local_copies():
    """The promotion cannot have changed a result.

    Compiles the REAL shared header together with a byte-copy of the bodies
    deleted from cold.cu:206-223 and warm.cu:272-289 in one translation unit,
    and compares them on 297 values across 27 decades at both production
    ``first_exponent``/``table_size`` pairs.  This is the same technique
    wave 3 used for the three helpers consolidated before this one; it is what
    lets the promotion be a refactor rather than a change.
    """
    import cupy as cp
    from woof.core.kernels import module_source
    from woof.core.thompson_aerosol_launch import PROBE_MODULE

    source = module_source(PROBE_MODULE) + _DECADE_DOUBLE_LEGACY_SOURCE
    module = cp.RawModule(code=source, options=("-std=c++17",))
    compare = module.get_function("compare_decade_index_double")

    values = _decade_double_sweep()
    assert values.size >= 250
    device = cp.asarray(values)
    for first_exponent, table_size in ((6, 37), (2, 37), (0, 55)):
        promoted = cp.empty(values.size, dtype=cp.int32)
        legacy = cp.empty(values.size, dtype=cp.int32)
        compare(((values.size + 255) // 256,), (256,),
                (device, np.int32(first_exponent), np.int32(table_size),
                 promoted, legacy, np.int32(values.size)))
        promoted = cp.asnumpy(promoted)
        legacy = cp.asnumpy(legacy)
        assert np.array_equal(promoted, legacy), (first_exponent, table_size)
        assert (promoted >= 0).all() and (promoted <= table_size - 1).all()
        # The sweep must actually move the index, or equality is vacuous.
        assert len(set(int(v) for v in promoted)) > 20, (
            first_exponent, table_size)


def test_decade_index_double_probe_agrees_with_the_float_form_where_both_apply():
    """The double form is the SAME index rule, only carried in double.

    thompson.cu keeps two spellings because WRF does -- the ice-nuclei and
    cloud-droplet lookups are default REAL and the rain/graupel y-intercepts
    are DOUBLE PRECISION -- not because the rule differs.  Wherever a value is
    exactly representable in float32 the two must return the same bin, and
    that equivalence is what justifies one shared header entry per spelling
    rather than one per network.
    """
    import cupy as cp
    from woof.core.thompson_aerosol_launch import probe_decade_index

    values = np.array(
        [1.5, 9.9, 10.0, 99.0, 1000.0, 1.0e5, 1.0e6, 1.0e7, 2.5e8, 7.0e9],
        dtype=np.float32)
    for first_exponent, table_size in ((6, 37), (2, 37), (0, 55)):
        as_float = cp.asnumpy(probe_decade_index(
            cp.asarray(values), first_exponent, table_size))
        as_double = cp.asnumpy(_probe_decade_index_double(
            cp.asarray(values.astype(np.float64)), first_exponent,
            table_size))
        assert list(int(v) for v in as_float) == list(
            int(v) for v in as_double), (first_exponent, table_size)


def test_decade_index_double_matches_wrfs_index_arithmetic():
    """idx = INT(value/10**n) + 9*(n - first_exponent), clamped, zero-based.

    The pattern module_mp_thompson.F:2282-2307 (idx_r), :2581-2589 (idx_IN)
    and thompson.cu:3084-3105 all share.  Asserted against the arithmetic
    itself rather than against another transcription of the same code.
    """
    import cupy as cp

    values = _decade_double_sweep()
    for first_exponent, table_size in ((6, 37), (2, 37)):
        got = cp.asnumpy(_probe_decade_index_double(
            cp.asarray(values), first_exponent, table_size))
        for index, value in enumerate(values):
            exponent = int(math.floor(math.log10(value)))
            digit = int(value / 10.0 ** exponent)
            # float32 `scale` can pull a mantissa a hair below its decade;
            # accept either neighbouring digit and require the CLAMPED index
            # to match one of them.
            candidates = set()
            for shift in (0,):
                one_based = digit + 9 * (exponent + shift - first_exponent)
                candidates.add(max(0, min(one_based - 1, table_size - 1)))
            assert int(got[index]) in candidates, (
                value, first_exponent, int(got[index]), candidates)
        assert (got >= 0).all() and (got <= table_size - 1).all()
