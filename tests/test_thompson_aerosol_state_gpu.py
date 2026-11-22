"""GPU gates for WP-04, the mp=28 aerosol state kernels.

Authority for every expectation is either an oracle fixture under
``woof/data/thompson/oracle-aero/`` (generated from
``wrf461-pristine/phys/module_mp_thompson.F``, WRF v4.6.1 commit
``d66e442``) or a float32 host transcription of a named WRF line range.  Each
test says which, because "gated against a host transcription" is a materially
weaker claim than "gated against Fortran".

Fixture coverage actually exercised here
----------------------------------------
=========================== ==========================================
fixture                     what it pins in this file
=========================== ==========================================
``aero-init-profile``       thompson_init's CCN/IN profile fill and the
                            ``nwfa2d`` derivation (kernel 5).
``aero-sfc-emit``           the unclamped lowest-level emission (kernel 4)
                            AND -- because that scenario's tendencies are
                            all zero -- the terminal apply (kernel 3) in
                            its identity configuration.
``aero-nc-effrad``          effective radius over the ``nc`` ladder
``aero-nc-cap``             {2, 50, 5e6, 1e8, 5e9, 2e10} m^-3 and above
                            ``Nt_c_max`` (kernel 6).
``probe-effectrad``         ``calc_effectRad`` called directly, 14 rows,
                            all three ``inu_c`` branches plus the ice and
                            snow branches the column fixtures leave at
                            background (kernel 6).  All 14 rows carry
                            t = 285 K and qs = 2e-4, so its ``effs_m``
                            column is ONE state repeated -- the snow
                            branch's temperature dependence is gated on the
                            column fixtures and the wide-range sweep, not
                            here.
all 19 ``aero-*`` columns   effective radius, BITWISE, driven by the
                            oracle's own post-step state (kernel 6).
=========================== ==========================================

Two float32 facts this file measures rather than assumes
--------------------------------------------------------
1. **nvrtc widens a float32 chain whose result feeds a double expression**,
   and a named ``float`` local does not stop it -- only ``__fmul_rn`` /
   ``__fdiv_rn`` do.  WRF's :4012, :4015/:4017 and :4019 are REAL(4)
   sub-expressions meeting a DOUBLE ``lamc``, so those roundings are part of
   the answer.  ``test_state_finalize_rounds_every_real4_subexpression_...``
   compiles the three candidate spellings side by side.
2. **The oracle's ``**`` is glibc ``powf``**, not numpy's float32 ``power``
   (which disagrees with libm on about a fifth of arguments) and not CUDA's
   ``powf``.  Every host transcription here calls :func:`_glibc_powf`, which
   is the same symbol ``gfortran -O2`` emits.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

# glibc's powf, as a transcription rather than as a ``ctypes`` load of
# ``libm.so.6``.  The symbol is the right one to compare against -- the oracle
# is gfortran -O2 against glibc -- but dlopening it makes the test unrunnable
# anywhere glibc is not the C library, and this suite has to run on the
# Windows box the kernels are developed on.  ``noahmp_libm`` is a bit-exact
# transcription of the same glibc 2.39 ``sysdeps/ieee754/flt-32/e_powf.c``
# (its module docstring invites exactly this reuse), so the comparison is
# unchanged in substance and now happens on every host instead of one.
from woof.core.noahmp_libm import powf as _libm_powf

pytestmark = pytest.mark.gpu

_ORACLE = (Path(__file__).parents[1] / "woof" / "data" / "thompson"
           / "oracle-aero")

F = np.float32

# WRF REAL(4) constants, module_mp_thompson.F line numbers in comments.
_R_DRY = F(287.04)          # :217
_R1 = F(1.0e-12)            # :183
_R2 = F(1.0e-6)             # :184
_NT_C_MAX = F(1999.0e6)     # :89
_NWFA_FLOOR = F(11.1e6)     # :1805
_NIFA_FLOOR = F(5.0e3)      # :1806, naIN1*0.01
_AERO_CEIL = F(9999.0e6)    # :1805-1806, :3979-3981
_D0C = F(1.0e-6)            # :224
_D0R = F(50.0e-6)           # :225
_OBMR = F(1.0) / F(3.0)     # :721
_BM_R = F(3.0)              # :129

#: mp_gt_driver:1475-1477, the second clamp WRF applies to calc_effectRad's
#: output before radiation sees it.  (lower, upper) in METRES, per field.
_DRIVER_CLAMPS = ((F(2.49e-6), F(50.0e-6)),      # re_cloud, RE_QC_BG
                  (F(4.99e-6), F(125.0e-6)),     # re_ice,   RE_QI_BG
                  (F(9.99e-6), F(999.0e-6)))     # re_snow,  RE_QS_BG

#: The ONLY levels in the 19 committed after-columns where the effective
#: radius is not bitwise against the oracle, and the reason.
#:
#: ``tools/thompson_wrf461_oracle/run_column_aero.F90:600`` records the column
#: temperature as ``temp = theta * exner`` while ``mp_gt_driver:1357`` stored
#: ``th(i,k,j) = t1d(k)/pii(i,k,j)``.  That divide-then-multiply round trip is
#: not the identity in float32, so the fixture's ``temp_k`` can sit one ulp
#: away from the ``t1d`` ``calc_effectRad`` was actually handed.  The snow
#: branch is exponentially sensitive to it (:5689's ``10.0**loga_`` multiplies
#: a ``d(loga)/dtc0 ~ 0.034`` slope by ln(10)), and the ice branch inherits it
#: through ``rho``.
#:
#: ``test_effective_radius_is_bitwise_against_every_oracle_after_column``
#: does not merely tolerate these: it PROVES each one by exhibiting a ``t1d``
#: within a few ulps that reproduces the fixture EXACTLY *and* round-trips to
#: the recorded ``temp_k`` through WRF's own ``t1d/pii``, ``theta*exner``.
#:
#: SHRANK from eight entries to five when the oracle harness repaired its own
#: ``pii``: the exponent is now formed as ``(p/p0)**rd_over_cp`` and the
#: fixtures' ``theta_k``/``pii``/``temp_k`` moved by ulps, which removed three
#: of the round-trip levels outright.  Every surviving entry is proved at
#: EXACTLY ONE ulp (deltas +1, +1, +1, -1, +1 in the order listed), where the
#: old list needed the search to run to +-4.  Adding an entry here without the
#: proof succeeding is not possible: the test fails on any level it cannot
#: solve, and fails again if a listed level stops needing the exception.
_FIXTURE_TEMPERATURE_ROUND_TRIP_LEVELS = {
    ("aero-cold-overlap", "effs_m"): (6, 7),
    ("aero-ice-demott-dep", "effi_m"): (6,),
    ("aero-ice-demott-idxin", "effi_m"): (6, 14),
}


# ---------------------------------------------------------------------------
# Fixture readers.
# ---------------------------------------------------------------------------

def _column(scenario: str, phase: str) -> list[dict[str, str]]:
    path = _ORACLE / f"{scenario}-column.csv"
    with path.open(newline="", encoding="ascii") as stream:
        rows = [row for row in csv.DictReader(stream) if row["phase"] == phase]
    assert len(rows) == 24, f"{path} {phase} rows"
    return rows


def _surface(scenario: str) -> dict[str, str]:
    path = _ORACLE / f"{scenario}-surface.csv"
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    return rows[0]


def _probe(name: str) -> list[dict[str, str]]:
    path = _ORACLE / f"probe-{name}.csv"
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _f32(rows, key) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float32)


def _dev(array):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(array, dtype=np.float32))


# ---------------------------------------------------------------------------
# Host transcriptions.  Each names the WRF line range it reproduces.
# ---------------------------------------------------------------------------

def _glibc_powf(base, exponent):
    """``powf`` from the C library gfortran links the oracle against.

    ``tools/thompson_wrf461_oracle/build_aero.sh`` compiles every oracle
    binary with plain ``gfortran -O2``, and gfortran lowers a ``REAL(4) **
    REAL(4)`` to ``powf``.  Calling the same symbol here -- rather than
    ``numpy``'s float32 ``power``, whose SIMD kernel disagrees with libm on
    about a fifth of random arguments -- is what makes "bitwise against a host
    transcription" mean something.

    MEASURED on this machine: over 20000 random cube-root arguments,
    ``libm.powf`` and a round-once-from-double evaluation agree on 19983;
    ``numpy``'s float32 ``power`` agrees with ``libm.powf`` on 15811.

    Evaluated by :mod:`woof.core.noahmp_libm`'s bit-exact transcription of
    glibc 2.39's ``e_powf.c`` rather than by dlopening ``libm.so.6``, so the
    host this runs on does not have to be a glibc host.  Nothing about what is
    compared moves; what moves is that the comparison no longer dies with
    ``AttributeError: 'NoneType' object has no attribute 'powf'`` off Linux.
    """
    return F(_libm_powf(F(base), F(exponent)))


def _host_density(pressure, temperature, qv):
    """module_mp_thompson.F:1802 / :3193 / :5624 in float32."""
    return F(0.622) * pressure / (_R_DRY * temperature * (qv + F(0.622)))


def _host_entry_snapshot(temperature, pressure, qv, nwfa, nifa):
    """:1801-1806, aer_init_opt < 2."""
    qv_local = np.maximum(F(1.0e-10), qv)
    rho = _host_density(pressure, temperature, qv_local)
    nwfa_m3 = np.maximum(_NWFA_FLOOR, np.minimum(_AERO_CEIL, nwfa * rho))
    nifa_m3 = np.maximum(_NIFA_FLOOR, np.minimum(_AERO_CEIL, nifa * rho))
    return rho, nwfa_m3, nifa_m3


def _host_state_finalize(qc, nc, nwfa, nifa, ncten, nwfaten, nifaten,
                         rho, dt):
    """:3972-4021, aer_init_opt < 2, wif_input_opt = 0."""
    from woof.core.thompson_aerosol_contract import (
        AM_R, CCE2, CCG1, CCG2, OCG1, OCG2)

    am_r = F(AM_R)
    dt = F(dt)
    qc_out = np.array(qc, dtype=np.float32, copy=True)

    nc_new = np.maximum(F(2.0) / rho,
                        np.minimum(nc + ncten * dt, _NT_C_MAX))
    nwfa_out = np.maximum(_NWFA_FLOOR,
                          np.minimum(_AERO_CEIL, nwfa + nwfaten * dt))
    nifa_out = np.maximum(_NIFA_FLOOR,
                          np.minimum(_AERO_CEIL, nifa + nifaten * dt))
    nc_out = np.empty_like(nc_new)

    for i in range(nc_new.size):
        if qc_out.flat[i] <= _R1:
            qc_out.flat[i] = F(0.0)
            nc_out.flat[i] = F(0.0)
            continue
        nc_i = nc_new.flat[i]
        rho_i = rho.flat[i]
        qc_i = qc_out.flat[i]
        # :4013  nu_c = MIN(15, NINT(1000.E6/(nc1d*rho)) + 2)
        nu_c = min(15, int(math.floor(F(F(1000.0e6) / (nc_i * rho_i)) + 0.5))
                   + 2)
        # :4012  a REAL(4) base to a REAL(4) power; only the RESULT is widened
        # to the DOUBLE lamc, so the base is rounded to float32 first and the
        # power itself is glibc's powf.
        lamc = float(_glibc_powf(
            F(F(F(F(am_r * F(CCG2[nu_c])) * F(OCG1[nu_c])) * nc_i) / qc_i),
            _OBMR))
        # :4013
        x_dc = F((_BM_R + F(nu_c) + F(1.0)) / lamc)
        # :4015 / :4017  a REAL(4) QUOTIENT assigned to the DOUBLE lamc.  D0c
        # and D0r*2. are not exact in binary, so the float32 rounding here is
        # observable: 10.0f/1.0e-6f is 10000000.0, in double it is
        # 10000000.025.
        if x_dc < _D0C:
            lamc = float(F(F(CCE2[nu_c]) / _D0C))
        elif x_dc > _D0R * F(2.0):
            lamc = float(F(F(CCE2[nu_c]) / F(_D0R * F(2.0))))
        # :4019-4020  MIN(<double>, DBLE(Nt_c_max)/rho).  ccg(1,nu_c),
        # ocg2(nu_c), qc1d(k) and am_r are all REAL(4), so the prefactor is
        # rounded to float32 BEFORE it meets the DOUBLE lamc**bm_r.
        value = (float(F(F(F(F(CCG1[nu_c]) * F(OCG2[nu_c])) * qc_i) / am_r))
                 * lamc ** 3.0)
        nc_out.flat[i] = F(min(value, float(_NT_C_MAX) / float(rho_i)))
    return qc_out, nc_out, nwfa_out, nifa_out


def _host_init_profile(hgt):
    """thompson_init:493-551 for one column; ``hgt`` is 1-D over k."""
    hgt = np.asarray(hgt, dtype=np.float32)
    h1 = hgt[0]
    if h1 <= F(1000.0):
        h_01 = F(0.8)
    elif h1 >= F(2500.0):
        h_01 = F(0.01)
    else:
        h_01 = F(0.8) * F(math.cos(float(h1 * F(0.001) - F(1.0))))

    def profile(a0, a1):
        ni3 = F(-1.0) * F(math.log(float(F(a1) / F(a0)))) / h_01
        out = np.empty_like(hgt)
        z1 = hgt[1] - hgt[0]
        out[0] = F(a1) + F(a0) * F(math.exp(float(-(z1 / F(1000.0)) * ni3)))
        for k in range(1, hgt.size):
            out[k] = F(a1) + F(a0) * F(math.exp(
                float(-((hgt[k] - h1) / F(1000.0)) * ni3)))
        return out

    nwfa = profile(300.0e6, 50.0e6)
    nifa = profile(1.5e6, 0.5e6)
    z1 = hgt[1] - hgt[0]
    nwfa2d = F(F(nwfa[0] * F(0.000196)) * F(F(50.0) / z1))
    return nwfa, nifa, nwfa2d


# ===========================================================================
# Kernel 5 -- thompson_init's synthetic CCN / IN profile.
# ===========================================================================

def test_init_profile_matches_wrf_fixture_101():
    """ORACLE gate.  aero-init-profile's 'before' rows are the state
    thompson_init itself wrote (the harness snapshots AFTER init precisely so
    the fill is observable), and the surface CSV carries the derived nwfa2d.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        aerosol_profile_needs_fill, launch_aerosol_init_profile)

    rows = _column("aero-init-profile", "before")
    hgt_1d = _f32(rows, "z_m")
    expected_nwfa = _f32(rows, "nwfa_per_kg")
    expected_nifa = _f32(rows, "nifa_per_kg")
    expected_nwfa2d = F(float(_surface("aero-init-profile")["nwfa2d_kg_s"]))
    expected_nifa2d = F(float(_surface("aero-init-profile")["nifa2d_kg_s"]))
    assert expected_nifa2d == F(0.0), "WRF never derives a nifa2d"

    nz, ny, nx = hgt_1d.size, 3, 5
    hgt = cp.asarray(np.broadcast_to(hgt_1d[:, None, None],
                                     (nz, ny, nx)).copy())
    nwfa = cp.zeros((nz, ny, nx), dtype=cp.float32)
    nifa = cp.zeros((nz, ny, nx), dtype=cp.float32)
    nwfa2d = cp.zeros((ny, nx), dtype=cp.float32)
    nifa2d = cp.zeros((ny, nx), dtype=cp.float32)

    assert aerosol_profile_needs_fill(nwfa)
    assert aerosol_profile_needs_fill(nifa)
    launch_aerosol_init_profile(hgt, nwfa, nifa, nwfa2d,
                                fill_ccn=True, fill_in=True)
    cp.cuda.Stream.null.synchronize()

    # MEASURED: BITWISE EQUAL on all 24 levels of nwfa and nifa and on
    # nwfa2d.  The kernel evaluates exp/log/cos in double and rounds once
    # (glibc is correctly rounded, CUDA's expf/logf/cosf are not) and pins
    # float32 contraction, so this is an exact-equality claim, not a
    # tolerance.  A regression here means an FMA crept back in or a
    # transcendental lost its correct rounding -- both worth failing on.
    for name, got, want in (("nwfa", nwfa, expected_nwfa),
                            ("nifa", nifa, expected_nifa)):
        actual = cp.asnumpy(got)
        for j in range(ny):
            for i in range(nx):
                np.testing.assert_array_equal(
                    actual[:, j, i], want,
                    err_msg=f"{name} column ({j},{i})")
    np.testing.assert_array_equal(
        cp.asnumpy(nwfa2d), np.full((ny, nx), expected_nwfa2d))
    # WRF derives no nifa2d at all; the kernel must leave it untouched.
    assert np.array_equal(cp.asnumpy(nifa2d), np.zeros((ny, nx), np.float32))


def test_init_profile_level_one_uses_the_level_two_height_difference():
    """HOST transcription.  WRF evaluates level 1 at hgt(2)-hgt(1), not at
    zero (thompson_init:508 and :540).  If it used zero the k=0 value would be
    naCCN1+naCCN0 = 350e6 exactly; fixture 101 shows 147.9e6.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import launch_aerosol_init_profile

    hgt_1d = _f32(_column("aero-init-profile", "before"), "z_m")
    nz = hgt_1d.size
    hgt = cp.asarray(np.broadcast_to(hgt_1d[:, None, None],
                                     (nz, 1, 1)).copy())
    nwfa = cp.zeros((nz, 1, 1), dtype=cp.float32)
    nifa = cp.zeros((nz, 1, 1), dtype=cp.float32)
    nwfa2d = cp.zeros((1, 1), dtype=cp.float32)
    launch_aerosol_init_profile(hgt, nwfa, nifa, nwfa2d,
                                fill_ccn=True, fill_in=True)
    cp.cuda.Stream.null.synchronize()
    assert float(nwfa[0, 0, 0]) != pytest.approx(350.0e6, rel=1e-6)
    assert float(nwfa[0, 0, 0]) == pytest.approx(1.47898656e8, rel=1e-6)


@pytest.mark.parametrize("terrain", (0.0, 500.0, 999.0, 1000.0, 1000.5,
                                     1500.0, 2000.0, 2499.5, 2500.0, 3200.0))
def test_init_profile_h01_branches_are_absolute_terrain_height(terrain):
    """HOST transcription of thompson_init:499-506.

    Blocking unknown #4.  ``hgt(i,1,j)`` is the ABSOLUTE (above sea level)
    height of the lowest w level, i.e. terrain elevation, and it is compared
    against absolute 1000 m / 2500 m thresholds -- while the profile body uses
    the AGL difference ``hgt(k)-hgt(1)``.  This test drives a column whose
    terrain sweeps across both thresholds and requires the kernel to switch
    branches at exactly those absolute values.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import launch_aerosol_init_profile

    nz = 12
    hgt_1d = (np.float32(terrain)
              + np.arange(nz, dtype=np.float32) * F(400.0))
    hgt = cp.asarray(np.broadcast_to(hgt_1d[:, None, None],
                                     (nz, 1, 1)).copy())
    nwfa = cp.zeros((nz, 1, 1), dtype=cp.float32)
    nifa = cp.zeros((nz, 1, 1), dtype=cp.float32)
    nwfa2d = cp.zeros((1, 1), dtype=cp.float32)
    launch_aerosol_init_profile(hgt, nwfa, nifa, nwfa2d,
                                fill_ccn=True, fill_in=True)
    cp.cuda.Stream.null.synchronize()

    want_nwfa, want_nifa, want_nwfa2d = _host_init_profile(hgt_1d)
    np.testing.assert_allclose(cp.asnumpy(nwfa)[:, 0, 0], want_nwfa,
                               rtol=1.2e-7, atol=0.0)
    np.testing.assert_allclose(cp.asnumpy(nifa)[:, 0, 0], want_nifa,
                               rtol=1.2e-7, atol=0.0)
    np.testing.assert_allclose(float(nwfa2d[0, 0]), float(want_nwfa2d),
                               rtol=1.2e-7, atol=0.0)


def test_init_profile_fills_ccn_and_in_independently():
    """HOST.  WRF tests MAXVAL(nwfa) and MAXVAL(nifa) against eps SEPARATELY
    (thompson_init:493 and :530), so a domain can get one fill and not the
    other, and nwfa2d is derived only inside the CCN branch.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        aerosol_profile_needs_fill, launch_aerosol_init_profile)

    nz = 8
    hgt_1d = np.arange(nz, dtype=np.float32) * F(500.0) + F(120.0)
    hgt = cp.asarray(np.broadcast_to(hgt_1d[:, None, None],
                                     (nz, 2, 2)).copy())
    seeded = cp.full((nz, 2, 2), 4.0e8, dtype=cp.float32)
    nwfa = seeded.copy()
    nifa = cp.zeros((nz, 2, 2), dtype=cp.float32)
    nwfa2d = cp.full((2, 2), -1.0, dtype=cp.float32)

    assert not aerosol_profile_needs_fill(nwfa)
    assert aerosol_profile_needs_fill(nifa)
    launch_aerosol_init_profile(hgt, nwfa, nifa, nwfa2d,
                                fill_ccn=False, fill_in=True)
    cp.cuda.Stream.null.synchronize()

    assert cp.array_equal(nwfa, seeded), "CCN branch must not have run"
    assert float(nwfa2d[0, 0]) == -1.0, "nwfa2d is CCN-branch only"
    _, want_nifa, _ = _host_init_profile(hgt_1d)
    np.testing.assert_allclose(cp.asnumpy(nifa)[:, 0, 0], want_nifa,
                               rtol=1.2e-7, atol=0.0)


def test_init_profile_never_touches_nc():
    """thompson_init writes nwfa/nifa/nwfa2d and nothing else; nc stays 0 and
    is bootstrapped by the first call's terminal rediagnosis.  Guarded here by
    the launcher signature: there is no nc argument to pass.
    """
    import inspect

    from woof.core.thompson_aerosol_state import launch_aerosol_init_profile

    names = set(inspect.signature(launch_aerosol_init_profile)
                .parameters)
    assert "nc" not in names
    assert names == {"hgt", "nwfa", "nifa", "nwfa2d", "fill_ccn", "fill_in"}


# ===========================================================================
# Kernel 4 -- surface emission, and kernel 3 in its identity configuration.
# ===========================================================================

def test_surface_emission_and_finalize_match_wrf_fixture_102():
    """ORACLE gate.  aero-sfc-emit is deeply subsaturated and hydrometeor
    free, so every tendency is zero and the whole call reduces to
    (terminal apply) then (surface emission).  Both are reproduced end to end
    against the fixture's before -> after transition.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_entry_snapshot, launch_aerosol_state_finalize,
        launch_aerosol_surface_emission, zero_aerosol_accumulators)

    before = _column("aero-sfc-emit", "before")
    after = _column("aero-sfc-emit", "after")
    surface = _surface("aero-sfc-emit")
    dt = float(surface["dt_s"])
    nwfa2d_value = F(float(surface["nwfa2d_kg_s"]))
    nifa2d_value = F(float(surface["nifa2d_kg_s"]))

    nz = len(before)
    temperature = _dev(_f32(before, "temp_k").reshape(nz, 1, 1))
    pressure = _dev(_f32(before, "p_pa").reshape(nz, 1, 1))
    qv = _dev(_f32(before, "qv").reshape(nz, 1, 1))
    qc = _dev(_f32(before, "qc").reshape(nz, 1, 1))
    nc = _dev(_f32(before, "nc_per_kg").reshape(nz, 1, 1))
    nwfa = _dev(_f32(before, "nwfa_per_kg").reshape(nz, 1, 1))
    nifa = _dev(_f32(before, "nifa_per_kg").reshape(nz, 1, 1))

    rho = cp.empty_like(temperature)
    nwfa_m3 = cp.empty_like(temperature)
    nifa_m3 = cp.empty_like(temperature)
    launch_aerosol_entry_snapshot(temperature, pressure, qv, nwfa, nifa,
                                  rho, nwfa_m3, nifa_m3)

    ncten = cp.empty_like(temperature)
    nwfaten = cp.empty_like(temperature)
    nifaten = cp.empty_like(temperature)
    zero_aerosol_accumulators(ncten, nwfaten, nifaten)

    nc_out = cp.empty_like(temperature)
    nwfa_out = cp.empty_like(temperature)
    nifa_out = cp.empty_like(temperature)
    launch_aerosol_state_finalize(qc, nc, nwfa, nifa, ncten, nwfaten,
                                  nifaten, rho, dt,
                                  nc_out, nwfa_out, nifa_out)

    nwfa2d = cp.full((1, 1), nwfa2d_value, dtype=cp.float32)
    nifa2d = cp.full((1, 1), nifa2d_value, dtype=cp.float32)
    launch_aerosol_surface_emission(nwfa_out, nifa_out, nwfa2d, nifa2d, dt)
    cp.cuda.Stream.null.synchronize()

    # MEASURED: BITWISE EQUAL to WRF on all 24 levels of nc, nwfa and nifa.
    np.testing.assert_array_equal(cp.asnumpy(nc_out).ravel(),
                                  _f32(after, "nc_per_kg"))
    np.testing.assert_array_equal(cp.asnumpy(nwfa_out).ravel(),
                                  _f32(after, "nwfa_per_kg"))
    np.testing.assert_array_equal(cp.asnumpy(nifa_out).ravel(),
                                  _f32(after, "nifa_per_kg"))
    # The emission must land on k=0 and nowhere else.
    assert cp.asnumpy(nwfa_out)[0, 0, 0] != cp.asnumpy(nwfa)[0, 0, 0]
    np.testing.assert_array_equal(cp.asnumpy(nwfa_out)[1:],
                                  cp.asnumpy(nwfa)[1:])


def test_surface_emission_is_lowest_level_only_and_unclamped():
    """HOST.  mp_gt_driver:1316-1327 has NO clamp: it runs after
    mp_thompson's terminal ceiling, so the aerosol fields may legitimately sit
    above 9999.E6 until the next call's entry pack re-applies it.  A
    "defensive" MIN here would change the boundary-layer budget every step.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_surface_emission)

    nz, ny, nx = 6, 3, 4
    nwfa = cp.full((nz, ny, nx), 9.0e9, dtype=cp.float32)
    nifa = cp.full((nz, ny, nx), 9.0e9, dtype=cp.float32)
    nwfa2d = cp.full((ny, nx), 5.0e8, dtype=cp.float32)
    nifa2d = cp.full((ny, nx), 2.0e8, dtype=cp.float32)
    dt = 60.0

    launch_aerosol_surface_emission(nwfa, nifa, nwfa2d, nifa2d, dt)
    cp.cuda.Stream.null.synchronize()

    got_w = cp.asnumpy(nwfa)
    got_i = cp.asnumpy(nifa)
    assert np.allclose(got_w[0], F(9.0e9) + F(5.0e8) * F(dt))
    assert got_w[0].min() > float(_AERO_CEIL), "the emission must not clamp"
    assert np.allclose(got_i[0], F(9.0e9) + F(2.0e8) * F(dt))
    assert np.array_equal(got_w[1:], np.full((nz - 1, ny, nx), F(9.0e9)))
    assert np.array_equal(got_i[1:], np.full((nz - 1, ny, nx), F(9.0e9)))


# ===========================================================================
# Kernel 1 -- entry snapshot.
# ===========================================================================

def test_entry_snapshot_matches_host_transcription_on_every_fixture():
    """HOST transcription of :1801-1806, evaluated on the 'before' state of
    all 19 aerosol column fixtures so the input distribution is WRF's, not
    invented.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_entry_snapshot)

    scenarios = sorted(p.name[: -len("-column.csv")]
                       for p in _ORACLE.glob("aero-*-column.csv"))
    assert len(scenarios) == 19

    for scenario in scenarios:
        rows = _column(scenario, "before")
        temperature = _f32(rows, "temp_k")
        pressure = _f32(rows, "p_pa")
        qv = _f32(rows, "qv")
        nwfa = _f32(rows, "nwfa_per_kg")
        nifa = _f32(rows, "nifa_per_kg")

        d_t, d_p, d_qv = _dev(temperature), _dev(pressure), _dev(qv)
        d_nwfa, d_nifa = _dev(nwfa), _dev(nifa)
        rho = cp.empty_like(d_t)
        nwfa_m3 = cp.empty_like(d_t)
        nifa_m3 = cp.empty_like(d_t)
        launch_aerosol_entry_snapshot(d_t, d_p, d_qv, d_nwfa, d_nifa,
                                      rho, nwfa_m3, nifa_m3)
        cp.cuda.Stream.null.synchronize()

        want_rho, want_w, want_i = _host_entry_snapshot(
            temperature, pressure, qv, nwfa, nifa)
        # MEASURED: bitwise equal on all three outputs across all 19 fixtures.
        np.testing.assert_array_equal(cp.asnumpy(rho), want_rho,
                                      err_msg=scenario)
        np.testing.assert_array_equal(cp.asnumpy(nwfa_m3), want_w,
                                      err_msg=scenario)
        np.testing.assert_array_equal(cp.asnumpy(nifa_m3), want_i,
                                      err_msg=scenario)
        # state must be untouched
        np.testing.assert_array_equal(cp.asnumpy(d_nwfa), nwfa)
        np.testing.assert_array_equal(cp.asnumpy(d_nifa), nifa)


def test_entry_snapshot_applies_both_bounds():
    """HOST.  :1805-1806 clamp on BOTH sides, unlike the working refresh."""
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_entry_snapshot)

    temperature = _dev(np.full(4, 285.0))
    pressure = _dev(np.full(4, 90000.0))
    qv = _dev(np.full(4, 0.008))
    rho_scalar = float(_host_density(F(90000.0), F(285.0), F(0.008)))
    nwfa = _dev(np.asarray([0.0, 1.0e6, 1.0e10, 3.0e8]) / rho_scalar)
    nifa = _dev(np.asarray([0.0, 1.0e2, 1.0e10, 1.0e6]) / rho_scalar)

    rho = cp.empty_like(temperature)
    nwfa_m3 = cp.empty_like(temperature)
    nifa_m3 = cp.empty_like(temperature)
    launch_aerosol_entry_snapshot(temperature, pressure, qv, nwfa, nifa,
                                  rho, nwfa_m3, nifa_m3)
    cp.cuda.Stream.null.synchronize()

    got_w = cp.asnumpy(nwfa_m3)
    got_i = cp.asnumpy(nifa_m3)
    assert got_w[0] == _NWFA_FLOOR and got_w[1] == _NWFA_FLOOR
    assert got_w[2] == _AERO_CEIL
    assert got_w[3] == pytest.approx(3.0e8, rel=1e-6)
    assert got_i[0] == _NIFA_FLOOR and got_i[1] == _NIFA_FLOOR
    assert got_i[2] == _AERO_CEIL
    assert got_i[3] == pytest.approx(1.0e6, rel=1e-6)


def test_entry_cloud_number_agrees_with_the_shared_device_helper():
    """Cross-check kernel 1b against WP-02's independent pointwise probe of
    ``thompson_aa_cloud_dist`` -- two call paths into the same header helper,
    which is what makes an entry-state disagreement impossible to hide.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_launch import probe_cloud_dist
    from woof.core.thompson_aerosol_state import (
        launch_aerosol_entry_cloud_number)

    rows = _column("aero-nc-cap", "before") + _column("aero-nc-auto", "before")
    temperature = _f32(rows, "temp_k")
    pressure = _f32(rows, "p_pa")
    qv = _f32(rows, "qv")
    qc = _f32(rows, "qc")
    nc = _f32(rows, "nc_per_kg")
    rho = _host_density(pressure, temperature, np.maximum(F(1.0e-10), qv))
    active = qc > _R1
    assert active.any() and not active.all()

    d_qc, d_nc, d_rho = _dev(qc), _dev(nc), _dev(rho)
    rc = cp.empty_like(d_qc)
    nc_m3 = cp.empty_like(d_qc)
    nu_c = cp.empty(d_qc.shape, dtype=cp.int32)
    l_qc = cp.empty(d_qc.shape, dtype=cp.int32)
    launch_aerosol_entry_cloud_number(d_qc, d_nc, d_rho, rc, nc_m3,
                                      nu_c, l_qc)
    cp.cuda.Stream.null.synchronize()

    assert np.array_equal(cp.asnumpy(l_qc).astype(bool), active)
    # WRF zeroes qc and nc on the inactive branch (:1844-1845).
    assert np.all(cp.asnumpy(d_qc)[~active] == 0.0)
    assert np.all(cp.asnumpy(d_nc)[~active] == 0.0)

    probe_nc, probe_nu, _ = probe_cloud_dist(
        _dev(np.maximum(_R1, qc * rho)), _dev(nc), d_rho)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(nc_m3)[active],
                                  cp.asnumpy(probe_nc)[active])
    np.testing.assert_array_equal(cp.asnumpy(nu_c)[active],
                                  cp.asnumpy(probe_nu)[active])


# ===========================================================================
# Kernel 2 -- working refresh.
# ===========================================================================

def test_working_number_has_no_ceiling_and_no_nifa_counterpart():
    """HOST transcription of :3211.

    The asymmetry against the entry snapshot is the whole point: no 9999.E6
    ceiling, no nifa counterpart, and the TAU+1 density.  If a future edit
    "harmonized" the two snapshots this test fails.
    """
    import cupy as cp
    import inspect

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_working_number)

    names = list(inspect.signature(launch_aerosol_working_number).parameters)
    assert "nifa" not in names and "nifaten" not in names

    nwfa = _dev(np.asarray([1.0e6, 3.0e8, 8.0e9, 0.0]))
    nwfaten = _dev(np.asarray([0.0, 1.0e6, 2.0e8, 0.0]))
    rho = _dev(np.asarray([1.1, 1.0, 1.2, 0.9]))
    dt = 30.0
    out = cp.empty_like(nwfa)
    launch_aerosol_working_number(nwfa, nwfaten, rho, dt, out)
    cp.cuda.Stream.null.synchronize()

    want = np.maximum(
        _NWFA_FLOOR,
        (cp.asnumpy(nwfa) + cp.asnumpy(nwfaten) * F(dt)) * cp.asnumpy(rho))
    np.testing.assert_allclose(cp.asnumpy(out), want, rtol=1.2e-7, atol=0.0)
    # 8e9*1.2 + 2e8*30*1.2 = 1.68e10, far above the entry snapshot's ceiling.
    assert float(out[2]) > float(_AERO_CEIL)


def test_tau1_density_matches_entry_density_definition():
    """:3193 and :1802 are the same formula on different inputs; both come
    from this file so they cannot drift.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_entry_snapshot, launch_tau1_density)

    rows = _column("aero-warm-overlap", "after")
    temperature = _dev(_f32(rows, "temp_k"))
    pressure = _dev(_f32(rows, "p_pa"))
    qv = _dev(_f32(rows, "qv"))
    zeros = cp.zeros_like(temperature)

    rho_tau1 = cp.empty_like(temperature)
    launch_tau1_density(temperature, pressure, qv, rho_tau1)
    rho_entry = cp.empty_like(temperature)
    launch_aerosol_entry_snapshot(temperature, pressure, qv, zeros, zeros,
                                  rho_entry, cp.empty_like(temperature),
                                  cp.empty_like(temperature))
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(rho_tau1), cp.asnumpy(rho_entry))


def test_working_cloud_is_a_plain_clamp_without_the_lamc_rediagnosis():
    """HOST transcription of :3213-3221.  Unlike the entry diagnosis there is
    no lamc / D0c / 2*D0r step here; conflating them would rewrite nc at every
    condensation step.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_working_cloud)

    qc = _dev(np.asarray([0.0, 1.0e-13, 3.0e-4, 3.0e-4, 3.0e-4]))
    qcten = _dev(np.asarray([0.0, 0.0, 0.0, -1.0e-5, 1.0e-6]))
    nc = _dev(np.asarray([1.0e8, 1.0e8, 1.0e8, 1.0, 3.0e9]))
    ncten = _dev(np.asarray([0.0, 0.0, 1.0e6, 0.0, 0.0]))
    rho = _dev(np.asarray([1.0, 1.0, 1.05, 1.05, 1.05]))
    dt = 20.0

    rc = cp.empty_like(qc)
    nc_m3 = cp.empty_like(qc)
    l_qc = cp.empty(qc.shape, dtype=cp.int32)
    launch_aerosol_working_cloud(qc, qcten, nc, ncten, rho, dt,
                                 rc, nc_m3, l_qc)
    cp.cuda.Stream.null.synchronize()

    qc_h, qcten_h = cp.asnumpy(qc), cp.asnumpy(qcten)
    nc_h, ncten_h, rho_h = cp.asnumpy(nc), cp.asnumpy(ncten), cp.asnumpy(rho)
    updated = qc_h + qcten_h * F(dt)
    active = updated > _R1
    want_rc = np.where(active, updated * rho_h, _R1).astype(np.float32)
    want_nc = np.where(
        active,
        np.maximum(F(2.0), np.minimum((nc_h + ncten_h * F(dt)) * rho_h,
                                      _NT_C_MAX)),
        F(2.0)).astype(np.float32)
    np.testing.assert_allclose(cp.asnumpy(rc), want_rc, rtol=1.2e-7, atol=0.0)
    np.testing.assert_allclose(cp.asnumpy(nc_m3), want_nc,
                               rtol=1.2e-7, atol=0.0)
    assert np.array_equal(cp.asnumpy(l_qc).astype(bool), active)


# ===========================================================================
# Kernel 3 -- terminal apply and clamp.
# ===========================================================================

def test_state_finalize_matches_host_transcription_over_the_fixture_states():
    """HOST transcription of :3972-4021, driven by every 'after' state in the
    19 aerosol fixtures plus a synthetic accumulator field, so the branch
    coverage is WRF's own distribution of qc/nc rather than a guess.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_state_finalize)

    scenarios = sorted(p.name[: -len("-column.csv")]
                       for p in _ORACLE.glob("aero-*-column.csv"))
    rng = np.random.default_rng(2804)
    dt = 20.0

    for scenario in scenarios:
        rows = _column(scenario, "after")
        temperature = _f32(rows, "temp_k")
        pressure = _f32(rows, "p_pa")
        qv = _f32(rows, "qv")
        qc = _f32(rows, "qc")
        nc = _f32(rows, "nc_per_kg")
        nwfa = _f32(rows, "nwfa_per_kg")
        nifa = _f32(rows, "nifa_per_kg")
        rho = _host_density(pressure, temperature,
                            np.maximum(F(1.0e-10), qv))
        ncten = (rng.standard_normal(qc.size).astype(np.float32)
                 * F(2.0e6))
        nwfaten = (rng.standard_normal(qc.size).astype(np.float32)
                   * F(5.0e6))
        nifaten = (rng.standard_normal(qc.size).astype(np.float32)
                   * F(1.0e4))

        d_qc = _dev(qc)
        args = [_dev(nc), _dev(nwfa), _dev(nifa), _dev(ncten),
                _dev(nwfaten), _dev(nifaten), _dev(rho)]
        nc_out = cp.empty_like(d_qc)
        nwfa_out = cp.empty_like(d_qc)
        nifa_out = cp.empty_like(d_qc)
        launch_aerosol_state_finalize(d_qc, *args, dt,
                                      nc_out, nwfa_out, nifa_out)
        cp.cuda.Stream.null.synchronize()

        want_qc, want_nc, want_nwfa, want_nifa = _host_state_finalize(
            qc, nc, nwfa, nifa, ncten, nwfaten, nifaten, rho, dt)
        np.testing.assert_array_equal(cp.asnumpy(d_qc), want_qc)
        np.testing.assert_allclose(cp.asnumpy(nwfa_out), want_nwfa,
                                   rtol=1.2e-7, atol=0.0, err_msg=scenario)
        np.testing.assert_allclose(cp.asnumpy(nifa_out), want_nifa,
                                   rtol=1.2e-7, atol=0.0, err_msg=scenario)
        # TIGHTENED from rtol=1.0e-6 to BITWISE.  The old bound existed
        # because the kernel skipped two float32 roundings WRF performs (see
        # test_state_finalize_rounds_every_real4_subexpression_that_
        # feeds_a_double) and because the host side used numpy's float32
        # pow instead of the libm powf gfortran emits.  With both
        # repaired the comparison is exact on all 456 fixture states.
        np.testing.assert_array_equal(cp.asnumpy(nc_out), want_nc,
                                      err_msg=scenario)


def test_state_finalize_reproduces_wrfs_per_kg_vs_per_m3_clamp():
    """HOST.  :3976 compares the PER-KILOGRAM nc1d against the volumetric
    Nt_c_max while converting only the lower bound, and :3979-3982 clamp
    per-kilogram nwfa/nifa against per-m3 constants with no density at all.
    Both are WRF unit inconsistencies and both must be reproduced literally.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_state_finalize)

    rho_value = 0.5   # far from 1 so a stray /rho or *rho is visible
    n = 3
    qc = _dev(np.zeros(n))                      # inactive branch: nc -> 0
    nc = _dev(np.asarray([5.0e9, 0.0, 1.0e8]))
    nwfa = _dev(np.asarray([2.0e10, 0.0, 3.0e8]))
    nifa = _dev(np.asarray([2.0e10, 0.0, 1.0e6]))
    zeros = _dev(np.zeros(n))
    rho = _dev(np.full(n, rho_value))
    nc_out = cp.empty_like(nc)
    nwfa_out = cp.empty_like(nc)
    nifa_out = cp.empty_like(nc)

    launch_aerosol_state_finalize(qc, nc, nwfa, nifa, zeros, zeros, zeros,
                                  rho, 10.0, nc_out, nwfa_out, nifa_out)
    cp.cuda.Stream.null.synchronize()

    # qc == 0 <= R1 everywhere, so nc is zeroed after the (a) clamp.
    assert np.all(cp.asnumpy(nc_out) == 0.0)
    # nwfa/nifa are clamped in PER-KILOGRAM space against the per-m3 numbers.
    np.testing.assert_array_equal(
        cp.asnumpy(nwfa_out),
        np.asarray([9999.0e6, 11.1e6, 3.0e8], dtype=np.float32))
    np.testing.assert_array_equal(
        cp.asnumpy(nifa_out),
        np.asarray([9999.0e6, 5.0e3, 1.0e6], dtype=np.float32))

    # Now the active branch, to see the (a) clamp itself.
    qc_active = _dev(np.full(n, 3.0e-4))
    launch_aerosol_state_finalize(qc_active, nc, nwfa, nifa, zeros, zeros,
                                  zeros, rho, 10.0,
                                  nc_out, nwfa_out, nifa_out)
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(nc_out)
    # entry 5e9 per kg is clamped to Nt_c_max PER KG (not Nt_c_max/rho),
    # then rediagnosed; entry 0 is floored at 2/rho = 4 per kg.
    assert np.all(np.isfinite(got))
    assert got.max() <= float(_NT_C_MAX) / rho_value * (1.0 + 1e-6)


#: What each NVRTC that has run the demonstration below does with the two
#: unpinned spellings of the REAL(4) prefactor: "widens" keeps the float32
#: chain in double (the pins in thompson_aerosol_state.cu are essential),
#: "real4" rounds it as Fortran does (the pins are belt and suspenders).
#: One row per (major, minor) that was run, because the behaviour is not
#: monotonic in the version: 13.1 rounds, 13.0 and 13.4 widen.
NVRTC_UNPINNED_CLASS = {
    # RTX 5090 sm_120 hosts; 13.0 on driver 610.74, 2026-08-03.
    (12, 8): "widens",
    (12, 9): "widens",
    (13, 0): "widens",
    # RTX 4090 sm_89, driver 590.48.01, 2026-08-03 (arwen-stress-4090).
    (13, 1): "real4",
    # RTX 4090 sm_89, NVRTC 13.3, 2026-10-03: both unpinned spellings
    # retain the REAL(4) prefactor; the pinned Fortran answer is unchanged.
    (13, 3): "real4",
    # An RTX PRO 4500 (sm_120, 2026-09-28) and the RTX 5070 Ti (sm_120,
    # NVRTC 13.4.92, driver 595.91.07, 2026-09-29): both spellings give
    # 3.5430368e+08, the widened answer, against WRF's 3.543037e+08.
    (13, 4): "widens",
}


def test_state_finalize_rounds_every_real4_subexpression_that_feeds_a_double():
    """REGRESSION.  nvrtc widens a float32 chain whose result is consumed by a
    double expression, and WRF's :4012-4020 does not.

    ``module_mp_thompson.F:4019`` reads

        nc1d(k) = MIN(ccg(1,nu_c)*ocg2(nu_c)*qc1d(k)/am_r*lamc**bm_r, ...)

    where ``ccg``, ``ocg2``, ``qc1d`` and ``am_r`` are all REAL(4) and only
    ``lamc`` is DOUBLE PRECISION.  Fortran therefore rounds the prefactor to
    float32 BEFORE the double multiply.  So does :4012's lambda base and
    :4015/:4017's ``cce(2,nu_c)/D0c`` quotient.

    This test compiles the three candidate spellings side by side so the
    mechanism is a measurement rather than a claim, then asserts the kernel is
    BITWISE equal to a Fortran-faithful host transcription over every fixture
    state.  BEFORE the pins were added the same comparison differed at 38 of
    456 states by up to 1.164e-07 relative, and the end-to-end ``nc_per_kg``
    residual sat at 2.4e-07 to 4.2e-07 on nearly every fixture.

    THE CONTROL IS TOOLCHAIN-VERSIONED, AND THE BOUNDARY IS MEASURED, NOT
    ASSUMED.  Widening is measured on NVRTC 12.8, 12.9 AND 13.0 (all sm_120
    RTX 5090 hosts; the 13.0 point is driver 610.74, 2026-08-03, on the
    reference 5090 box), which is what proved the ``.cu`` pins
    essential.  Exact agreement is measured only at NVRTC 13.1 (RTX 4090
    sm_89, driver 590.48.01, 2026-08-03, arwen-stress-4090): there all
    three spellings agree bitwise with the Fortran-faithful answer, 1 FP32
    ULP away from the widened one.  NVRTC 13.4 widens again (an RTX PRO
    4500 and the RTX 5070 Ti, both sm_120, 13.4.92 on the 5070 Ti,
    2026-09-29), which falsified the ``>= (13, 1)`` boundary this test used
    to draw, as 13.0 had falsified the ``>= (13, 0)`` guess before it.  So
    the classes are a table of the versions that were run
    (``NVRTC_UNPINNED_CLASS``), and a version missing from it fails until
    it is measured and recorded.

    This test fired its own "re-measure before relaxing the pins"
    instruction on 13.1, and the re-measurement's verdict is recorded per
    toolchain below: on a widening NVRTC the unpinned spellings must
    produce the WIDENED answer (the pins are essential); on a
    non-widening NVRTC they must produce the WRF answer EXACTLY (the pins
    are belt and suspenders -- they pin behaviour the toolchain now
    guarantees only by default, and they STAY, because the same wheel
    still runs on the toolchains of the first class).  Either way the
    toolchain gets exactly one asserted expectation, so a future NVRTC
    that behaves like neither class fails loudly instead of being
    absorbed.  Part 2 is toolchain-independent: the pinned kernel is
    bitwise against WRF's arithmetic under both classes, measured on both.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_contract import AM_R, CCG1, OCG2
    from woof.core.thompson_aerosol_state import (
        launch_aerosol_state_finalize)

    # ---- 1.  The mechanism, measured. ------------------------------------
    # Same operands, three spellings, one nvrtc invocation with the options
    # woof/core/kernels/__init__.py::load_module actually uses.
    # The operands reach the arithmetic exactly the way the real kernel's do:
    # AM_R as a float literal macro and the gamma columns as __constant__
    # tables.  That matters -- with all four passed as global-memory floats
    # nvrtc keeps the chain in float32 and the effect disappears, which is
    # why this demonstration is a transcription and not a sketch.
    source = r"""
    #define AA_AM_R 5.235988159e+02f
    __constant__ float AA_CCG1[16] = {
        0.000000000e+00f, 1.000000000e+00f, 2.000000000e+00f,
        6.000000000e+00f, 2.400000000e+01f, 1.200000076e+02f,
        7.200000610e+02f, 5.040001953e+03f, 4.031999609e+04f,
        3.628799688e+05f, 3.628801750e+06f, 3.991680000e+07f,
        4.790018560e+08f, 6.227022336e+09f, 8.717829734e+10f,
        1.307673887e+12f};
    __constant__ float AA_OCG2[16] = {
        0.000000000e+00f, 4.166666791e-02f, 8.333332837e-03f,
        1.388888806e-03f, 1.984126284e-04f, 2.480158946e-05f,
        2.755732112e-06f, 2.755730577e-07f, 2.505210794e-08f,
        2.087674478e-09f, 1.605904021e-10f, 1.147074449e-11f,
        7.647166320e-13f, 4.779478950e-14f, 2.811457147e-15f,
        1.561918299e-16f};
    extern "C" __global__ void spellings(
        const float* qc, const double* lamc, const int* nu,
        float* inlined, float* via_named_float, float* pinned, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int u = nu[i];
        inlined[i] = (float)(
            (double)(AA_CCG1[u] * AA_OCG2[u] * qc[i] / AA_AM_R)
            * pow(lamc[i], (double)3.0f));
        float pref = AA_CCG1[u] * AA_OCG2[u] * qc[i] / AA_AM_R;
        via_named_float[i] = (float)((double)pref
                                     * pow(lamc[i], (double)3.0f));
        float pinned_pref = __fdiv_rn(
            __fmul_rn(__fmul_rn(AA_CCG1[u], AA_OCG2[u]), qc[i]), AA_AM_R);
        pinned[i] = (float)((double)pinned_pref
                            * pow(lamc[i], (double)3.0f));
    }
    """
    nu_c = 6
    qc_value = F(8.9721725e-06)
    lamc_value = 2184250.25
    kernel = cp.RawKernel(source, "spellings", options=("-std=c++17",))
    d_qc = _dev(np.asarray([qc_value]))
    d_lamc = cp.asarray(np.asarray([lamc_value], dtype=np.float64))
    d_nu = cp.asarray(np.asarray([nu_c], dtype=np.int32))
    outs = [cp.zeros(1, cp.float32) for _ in range(3)]
    kernel((1,), (1,), (d_qc, d_lamc, d_nu, *outs, np.int32(1)))
    cp.cuda.Stream.null.synchronize()
    inlined, via_named, pinned = (float(o.get()[0]) for o in outs)

    prefactor_f32 = float(F(F(F(F(CCG1[nu_c]) * F(OCG2[nu_c])) * qc_value)
                            / F(AM_R)))
    prefactor_f64 = ((float(F(CCG1[nu_c])) * float(F(OCG2[nu_c])))
                     * float(qc_value)) / float(F(AM_R))
    wrf_answer = F(prefactor_f32 * lamc_value ** 3.0)
    widened_answer = F(prefactor_f64 * lamc_value ** 3.0)

    assert wrf_answer != widened_answer, (
        "the demonstration operands no longer separate the two spellings; "
        "pick operands that do rather than deleting the test")
    assert F(pinned) == wrf_answer, (
        "the rounding intrinsics must reproduce Fortran's REAL(4) prefactor")
    from cupy.cuda import nvrtc

    nvrtc_version = tuple(nvrtc.getVersion())
    measured = NVRTC_UNPINNED_CLASS.get(nvrtc_version)
    assert measured is not None, (
        f"NVRTC {nvrtc_version} has no recorded class for the unpinned "
        f"spellings (widened: {F(inlined) == widened_answer}, REAL(4): "
        f"{F(inlined) == wrf_answer}); record what it does in "
        f"NVRTC_UNPINNED_CLASS before touching the pins in "
        f"woof/core/kernels/thompson_aerosol_state.cu")
    if measured == "real4":
        # Measured on NVRTC 13.1: the toolchain rounds every float32
        # subexpression to REAL(4) exactly as the pinned spelling does, so
        # all three spellings agree bitwise and the pins are belt and
        # suspenders here.  13.1 is the only toolchain measured to do so:
        # 13.0 widens (reference RTX 5090, driver 610.74, 2026-08-03) and
        # so does 13.4.  Asserted exactly -- an unpinned spelling landing
        # anywhere else is a new toolchain behaviour to measure, not one to
        # absorb.
        assert F(inlined) == wrf_answer and F(via_named) == wrf_answer, (
            f"NVRTC {nvrtc_version} was recorded rounding the unpinned "
            f"spellings to the REAL(4) answer and no longer does; "
            f"re-measure and record this toolchain's class before touching "
            f"the pins in woof/core/kernels/thompson_aerosol_state.cu")
        print("NVRTC %s does not widen the unpinned spellings: all three "
              "agree with the REAL(4) answer bitwise; the .cu pins are "
              "belt and suspenders on this toolchain and essential on "
              "the toolchains measured to widen (12.8, 12.9, 13.0, 13.4)"
              % (".".join(str(p) for p in nvrtc_version),))
    else:
        # Widening measured at 12.8, 12.9, 13.0 and 13.4.  The pins are
        # essential here.
        assert (F(inlined) == widened_answer
                and F(via_named) == widened_answer), (
            f"NVRTC {nvrtc_version} no longer widens the unpinned "
            f"spellings, but widening is what was recorded for it in "
            f"NVRTC_UNPINNED_CLASS; re-measure and record its class there "
            f"rather than relaxing the pins in "
            f"woof/core/kernels/thompson_aerosol_state.cu")

    # ---- 2.  The kernel itself, bitwise, over every fixture state. --------
    scenarios = sorted(p.name[: -len("-column.csv")]
                       for p in _ORACLE.glob("aero-*-column.csv"))
    rng = np.random.default_rng(90210)
    dt = 20.0
    checked = 0
    for scenario in scenarios:
        rows = _column(scenario, "after")
        qc = _f32(rows, "qc")
        nc = _f32(rows, "nc_per_kg")
        nwfa = _f32(rows, "nwfa_per_kg")
        nifa = _f32(rows, "nifa_per_kg")
        rho = _host_density(_f32(rows, "p_pa"), _f32(rows, "temp_k"),
                            np.maximum(F(1.0e-10), _f32(rows, "qv")))
        ncten = rng.standard_normal(qc.size).astype(np.float32) * F(3.0e6)
        zeros = np.zeros_like(qc)

        d_qc = _dev(qc)
        nc_out = cp.empty_like(d_qc)
        nwfa_out = cp.empty_like(d_qc)
        nifa_out = cp.empty_like(d_qc)
        launch_aerosol_state_finalize(
            d_qc, _dev(nc), _dev(nwfa), _dev(nifa), _dev(ncten),
            _dev(zeros), _dev(zeros), _dev(rho), dt,
            nc_out, nwfa_out, nifa_out)
        cp.cuda.Stream.null.synchronize()

        _, want_nc, _, _ = _host_state_finalize(
            qc, nc, nwfa, nifa, ncten, zeros, zeros, rho, dt)
        np.testing.assert_array_equal(cp.asnumpy(nc_out), want_nc,
                                      err_msg=scenario)
        checked += qc.size
    assert checked == 24 * len(scenarios)


def test_terminal_clamp_block_member_by_member_against_wrf():
    """:3972-4021 read one member at a time, each with a state that isolates
    it.  This is the block every other aerosol package depends on -- it runs
    exactly once per call and it is the only place ``nc``/``nwfa``/``nifa``
    reach state -- so each clause gets its own arranged input rather than
    being covered incidentally by a fixture sweep.

    Members, in WRF's order:

    ===== ================================================================
    :3976 ``nc1d = MAX(2./rho, MIN(nc1d + ncten*DT, Nt_c_max))`` -- lower
          bound density-converted, upper bound NOT (per-kg vs per-m3).
    :3979 ``nwfa1d = MAX(11.1E6, MIN(9999.E6, nwfa1d + nwfaten*DT))``
          -- per-kg value, per-m3 constants, no rho anywhere.
    :3981 ``nifa1d = MAX(naIN1*0.01, MIN(9999.E6, ...))`` -- the floor is
          the folded constant 5000.0 exactly.
    :3987 ``nbca1d = 0.0`` -- wif_input_opt = 0, not carried at all.
    :4007 ``qc1d .le. R1`` zeroes BOTH qc1d and nc1d, and the zeroing beats
          the :3976 floor.
    :4011 ``nu_c`` from the POST-clamp ``nc1d*rho``, MIN'd at 15, no floor.
    :4013 ``xDc`` is REAL(4) even though ``lamc`` is DOUBLE.
    :4015 ``xDc < D0c`` and :4017 ``xDc > D0r*2.`` replace lamc outright.
    :4020 the rediagnosis cap IS ``DBLE(Nt_c_max)/rho`` -- the same constant
          as :3976's, but converted.  Three conventions in nine lines.
    ===== ================================================================
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_state_finalize)

    def run(qc, nc, nwfa, nifa, ncten, nwfaten, nifaten, rho, dt):
        d_qc = _dev(np.asarray(qc, np.float32))
        args = [_dev(np.asarray(a, np.float32)) for a in
                (nc, nwfa, nifa, ncten, nwfaten, nifaten, rho)]
        outs = [cp.empty_like(d_qc) for _ in range(3)]
        launch_aerosol_state_finalize(d_qc, *args, dt, *outs)
        cp.cuda.Stream.null.synchronize()
        return (cp.asnumpy(d_qc), *(cp.asnumpy(o) for o in outs))

    zero = [0.0] * 4

    # -- :3976 UPPER bound, the per-kg vs per-m3 asymmetry. -----------------
    # nc1d is per kilogram and Nt_c_max is volumetric, and WRF compares them
    # directly.  At rho = 0.25 a kernel that "fixed" the units would clamp at
    # Nt_c_max/rho = 7.996e9 instead of 1.999e9, which lands nu_c on 3 rather
    # than 4 and so changes the rediagnosed droplet number by a factor of
    # ccg(1,3)*ocg2(3) / ccg(1,4)*ocg2(4).  The output is therefore a real
    # discriminator, not a formality.
    rho_value = 0.25
    _, nc_out, _, _ = run([1.0e-4], [5.0e9], zero[:1], zero[:1], zero[:1],
                          zero[:1], zero[:1], [rho_value], 10.0)
    _, want_capped, _, _ = _host_state_finalize(
        np.asarray([1.0e-4], np.float32), np.asarray([5.0e9], np.float32),
        np.zeros(1, np.float32), np.zeros(1, np.float32),
        np.zeros(1, np.float32), np.zeros(1, np.float32),
        np.zeros(1, np.float32), np.asarray([rho_value], np.float32), 10.0)
    assert nc_out[0] == want_capped[0]
    nu_c_wrf = min(15, int(math.floor(
        F(F(1000.0e6) / (_NT_C_MAX * F(rho_value))) + 0.5)) + 2)
    nu_c_if_converted = min(15, int(math.floor(
        F(F(1000.0e6) / (F(5.0e9) * F(rho_value))) + 0.5)) + 2)
    assert nu_c_wrf != nu_c_if_converted, (
        "this state no longer separates the two spellings of :3976")

    # -- :3976 ACCUMULATION, and :3976 LOWER bound (accurately scoped). -------
    # ncten reaches the clamp multiplied by DT: 1e6 + 2e5*10 = 3e6.
    _, accumulated, _, _ = run([1.0e-4], [1.0e6], zero[:1], zero[:1],
                               [2.0e5], zero[:1], zero[:1], [1.0], 10.0)
    _, direct, _, _ = run([1.0e-4], [3.0e6], zero[:1], zero[:1], zero[:1],
                          zero[:1], zero[:1], [1.0], 10.0)
    assert accumulated[0] == direct[0]
    #
    # The LOWER bound 2./rho is NOT observable in this kernel's output and
    # this test does not pretend otherwise.  :3976's result is overwritten at
    # :4009 (zeroed) or :4020 (rediagnosed), and the only path from it to the
    # rediagnosis is nu_c = MIN(15, NINT(1000.E6/(nc1d*rho)) + 2).  When the
    # floor binds, nc1d*rho is exactly 2.0, so nu_c saturates at 15 -- and it
    # would also saturate for the unconverted spelling 2.0, which gives
    # nc1d*rho = 2*rho, for every physical density (nu_c only leaves 15 above
    # nc1d*rho ~ 7.7e7).  Asserted, so the claim is measured:
    for floor_rho in (0.05, 0.25, 1.0, 4.0):
        floored = F(F(2.0) / F(floor_rho))
        assert min(15, int(math.floor(
            F(F(1000.0e6) / (floored * F(floor_rho))) + 0.5)) + 2) == 15
        assert min(15, int(math.floor(
            F(F(1000.0e6) / (F(2.0) * F(floor_rho))) + 0.5)) + 2) == 15
    # So :3976's lower bound is transcribed from the source and gated by
    # reading, not by this test; the kernel comment at the top of
    # thompson_aerosol_state.cu records the same thing.

    # -- :4007-4009 vs :3976: the zeroing wins. ----------------------------
    qc_out, nc_out, _, _ = run([_R1, F(_R1) * F(2.0), 0.0, 1.0e-3],
                               [1.0e8] * 4, zero, zero, zero, zero, zero,
                               [1.0] * 4, 10.0)
    assert qc_out[0] == F(0.0) and nc_out[0] == F(0.0), "qc == R1 is NOT > R1"
    assert qc_out[2] == F(0.0) and nc_out[2] == F(0.0)
    assert qc_out[1] != F(0.0) and nc_out[1] != F(0.0), "qc > R1 survives"
    assert qc_out[3] == F(1.0e-3)

    # -- :3979-3982, per-kg values against per-m3 constants, no rho. --------
    for rho_value in (0.25, 1.0, 4.0):
        _, _, nwfa_out, nifa_out = run(
            [0.0] * 4,
            [0.0] * 4,
            [0.0, 1.0e4, 3.0e8, 2.0e10],
            [0.0, 1.0, 1.0e6, 2.0e10],
            zero, zero, zero, [rho_value] * 4, 10.0)
        np.testing.assert_array_equal(
            nwfa_out, np.asarray([_NWFA_FLOOR, _NWFA_FLOOR, 3.0e8,
                                  _AERO_CEIL], np.float32),
            err_msg=f"nwfa at rho={rho_value}")
        np.testing.assert_array_equal(
            nifa_out, np.asarray([_NIFA_FLOOR, _NIFA_FLOOR, 1.0e6,
                                  _AERO_CEIL], np.float32),
            err_msg=f"nifa at rho={rho_value}")
    # The nifa floor is naIN1*0.01 folded in REAL(4): 0.5E6 * 0.01f rounds to
    # exactly 5000.0, so 5.0e3f is the right literal and not an approximation.
    assert F(F(0.5e6) * F(0.01)) == _NIFA_FLOOR

    # -- :3979-3982 use the ACCUMULATED value, i.e. dt is applied. ----------
    _, _, nwfa_out, nifa_out = run([0.0], [0.0], [3.0e8], [1.0e6],
                                   [0.0], [1.0e7], [-2.0e4], [1.0], 15.0)
    assert nwfa_out[0] == F(F(3.0e8) + F(F(1.0e7) * F(15.0)))
    assert nifa_out[0] == F(F(1.0e6) + F(F(-2.0e4) * F(15.0)))

    # -- :4015 / :4017, the two size clamps, each reached deliberately. -----
    # Tiny qc with a huge droplet number drives xDc below D0c; a large qc
    # with the floor droplet number drives it above 2*D0r.
    _, nc_small, _, _ = run([2.0e-12], [1.999e9], zero[:1], zero[:1],
                            zero[:1], zero[:1], zero[:1], [1.0], 10.0)
    _, nc_big, _, _ = run([5.0e-3], [4.0], zero[:1], zero[:1],
                          zero[:1], zero[:1], zero[:1], [1.0], 10.0)
    _, want_nc, _, _ = _host_state_finalize(
        np.asarray([2.0e-12, 5.0e-3], np.float32),
        np.asarray([1.999e9, 4.0], np.float32),
        np.zeros(2, np.float32), np.zeros(2, np.float32),
        np.zeros(2, np.float32), np.zeros(2, np.float32),
        np.zeros(2, np.float32), np.ones(2, np.float32), 10.0)
    assert nc_small[0] == want_nc[0] and nc_big[0] == want_nc[1]
    # And they really are the clamped branches, not the general one.
    from woof.core.thompson_aerosol_contract import AM_R, CCG2, OCG1
    for qc_value, nc_value, expect in ((2.0e-12, 1.999e9, "D0c"),
                                       (5.0e-3, 4.0, "2*D0r")):
        nc_i = F(max(F(2.0), min(F(nc_value), _NT_C_MAX)))
        nu_c = min(15, int(math.floor(F(F(1000.0e6) / nc_i) + 0.5)) + 2)
        lamc = float(_glibc_powf(
            F(F(F(F(F(AM_R) * F(CCG2[nu_c])) * F(OCG1[nu_c])) * nc_i)
              / F(qc_value)), _OBMR))
        x_dc = F((_BM_R + F(nu_c) + F(1.0)) / lamc)
        assert (x_dc < _D0C) == (expect == "D0c"), (qc_value, x_dc)
        assert (x_dc > _D0R * F(2.0)) == (expect == "2*D0r"), (qc_value, x_dc)

    # -- :4020, the cap that IS density-converted. -------------------------
    # The rediagnosis returns approximately the droplet number it was given,
    # and :3976 has already capped that at Nt_c_max PER KILOGRAM, so :4020's
    # DBLE(Nt_c_max)/rho ceiling can only ever bind for rho >= 1.  Both sides
    # of that are asserted: a kernel that used the unconverted Nt_c_max would
    # return 1.999e9 at every density below.
    for rho_value in (1.0, 2.5, 4.0):
        _, capped, _, _ = run([1.0e-2], [1.999e9], zero[:1], zero[:1],
                              zero[:1], zero[:1], zero[:1], [rho_value], 10.0)
        assert capped[0] == F(float(_NT_C_MAX) / rho_value), (
            f":4020 must cap at DBLE(Nt_c_max)/rho, not Nt_c_max, at "
            f"rho={rho_value}")
    _, uncapped, _, _ = run([1.0e-2], [1.999e9], zero[:1], zero[:1],
                            zero[:1], zero[:1], zero[:1], [0.3], 10.0)
    assert uncapped[0] < F(float(_NT_C_MAX) / 0.3), (
        "at rho < 1 the :4020 ceiling is unreachable, so this state must "
        "come out of the rediagnosis itself")




# ---------------------------------------------------------------------------
# WRF's OWN TERMINAL LOOP, :3972-4021, INPUTS AND OUTPUTS.
# ---------------------------------------------------------------------------
#
# Produced by a build of PRISTINE ``module_mp_thompson.F`` carrying nothing but
# added ``write`` statements and four diagnostic-only locals (no physics line
# changed), linked against the committed ``run_column_aero.F90`` with
# ``build_aero.sh``'s exact flags and the same four assets.  FIDELITY PROOF:
# that build reproduces all 22 committed ``oracle-aero`` column CSVs BYTE FOR
# BYTE, so the numbers below are WRF's, not a transcription's.
#
# One row per level.  ``qc_post`` is ``qc1d(k)`` after :3975 and before
# :4008's zeroing -- i.e. exactly the kernel's ``qc`` input -- and
# ``nc_pre``/``nwfa_pre``/``nifa_pre`` are the entry per-kilogram values as of
# :3972.  ``rho_term`` is ``rho(k)`` as the loop finds it (:3193, then per
# level :3490 and :3572); ``rho_entry`` is the :1802 entry density the adapter
# passes today.  The last four are WRF's answers.
#
#   (qc_post, nc_pre, nwfa_pre, nifa_pre, ncten, nwfaten, nifaten,
#    rho_term, rho_entry, qc_final, nc_final, nwfa_final, nifa_final)
#
# THREE FIXTURES, chosen as the three whose two densities differ MOST:
# aero-cold-overlap (7.4672e-03), aero-reduces-to-classic (4.1039e-03) and
# aero-cloud-freeze-nc (7.2995e-04).  The value in each tuple is
# ``(dt_seconds, rows)``.
_WRF_TERMINAL_LOOP = {
    "aero-cold-overlap": (50, (
        (0.0, 0.0, 2132891136.0, 35548184.0, 0.0, 0.0, 0.0, 1.4065415859222412, 1.4065415859222412, 0.0, 0.0, 2132891136.0, 35548184.0),
        (0.0, 0.0, 2270491648.0, 37841524.0, -0.0, 0.0, 0.0, 1.3212997913360596, 1.3212997913360596, 0.0, 0.0, 2270491648.0, 37841524.0),
        (0.0, 80573448.0, 2417203456.0, 40286724.0, -1611468.875, -362833.3125, -1053.8345947265625, 1.2318360805511475, 1.2411036491394043, 0.0, 0.0, 2399061760.0, 40234032.0),
        (-2.9103830456733704e-11, 85772280.0, 2573168384.0, 42886140.0, -1715445.5, -305933.875, -890.6412963867188, 1.1584088802337646, 1.1658778190612793, 0.0, 0.0, 2557871616.0, 42841608.0),
        (1.4551915228366852e-11, 91306704.0, 2739201024.0, 45653352.0, -1826134.0, -117665.7109375, -541.6853637695312, 1.0909086465835571, 1.0952098369598389, 1.4551915228366852e-11, 1.8333361148834229, 2733317632.0, 45626268.0),
        (9.334253263659775e-07, 97198424.0, 2915952640.0, 48599212.0, -1929414.75, 1360967.625, -237.1900634765625, 1.0280100107192993, 1.0288232564926147, 9.334253263659775e-07, 727689.0, 2984001024.0, 48587352.0),
        (1.8189894035458565e-12, 103470536.0, 3104115968.0, 51735268.0, -2069410.5, 2000621.875, -74.83736419677734, 0.9665140509605408, 0.9664587378501892, 1.8189894035458565e-12, 8.000011444091797, 3204146944.0, 51731528.0),
        (9.094947017729282e-13, 110147584.0, 3304427520.0, 55073792.0, -2202951.75, 2196630.75, -4226.90966796875, 0.9078854918479919, 0.9078728556632996, 0.0, 0.0, 3414258944.0, 54862448.0),
        (0.0, 117255752.0, 3517672704.0, 58627876.0, -2345115.0, 2344245.75, -4763.82080078125, 0.8528359532356262, 0.8528366088867188, 0.0, 0.0, 3634885120.0, 58389684.0),
        (0.0, 0.0, 3744134656.0, 62402244.0, -0.0, 0.0, 0.0, 0.8012532591819763, 0.8012532591819763, 0.0, 0.0, 3744134656.0, 62402244.0),
        (0.0, 0.0, 3985737216.0, 66428956.0, -0.0, 0.0, 0.0, 0.7526838183403015, 0.7526838183403015, 0.0, 0.0, 3985737216.0, 66428956.0),
        (0.0, 0.0, 4242939392.0, 70715656.0, -0.0, 0.0, 0.0, 0.707056999206543, 0.707056999206543, 0.0, 0.0, 4242939392.0, 70715656.0),
        (0.0, 0.0, 4516747776.0, 75279128.0, -0.0, 0.0, 0.0, 0.6641947031021118, 0.6641947031021118, 0.0, 0.0, 4516747776.0, 75279128.0),
        (0.0, 0.0, 4808238080.0, 80137296.0, -0.0, 0.0, 0.0, 0.6239292025566101, 0.6239292025566101, 0.0, 0.0, 4808238080.0, 80137296.0),
        (0.0, 0.0, 5118552064.0, 85309200.0, -0.0, 0.0, 0.0, 0.5861032605171204, 0.5861032605171204, 0.0, 0.0, 5118552064.0, 85309200.0),
        (0.0, 0.0, 5448906752.0, 90815112.0, -0.0, 0.0, 0.0, 0.5505691766738892, 0.5505691766738892, 0.0, 0.0, 5448906752.0, 90815112.0),
        (0.0, 0.0, 5800600576.0, 96676672.0, -0.0, 0.0, 0.0, 0.5171878337860107, 0.5171878337860107, 0.0, 0.0, 5800600576.0, 96676672.0),
        (0.0, 0.0, 6175010816.0, 102916848.0, -0.0, 0.0, 0.0, 0.48582911491394043, 0.48582911491394043, 0.0, 0.0, 6175010816.0, 102916848.0),
        (0.0, 0.0, 6573609984.0, 109560168.0, -0.0, 0.0, 0.0, 0.4563702344894409, 0.4563702344894409, 0.0, 0.0, 6573609984.0, 109560168.0),
        (0.0, 0.0, 6997962240.0, 116632704.0, -0.0, 0.0, 0.0, 0.4286961853504181, 0.4286961853504181, 0.0, 0.0, 6997962240.0, 116632704.0),
        (0.0, 0.0, 7449735168.0, 124162256.0, -0.0, 0.0, 0.0, 0.4026988744735718, 0.4026988744735718, 0.0, 0.0, 7449735168.0, 124162256.0),
        (0.0, 0.0, 7930703872.0, 132178392.0, -0.0, 0.0, 0.0, 0.37827664613723755, 0.37827664613723755, 0.0, 0.0, 7930703872.0, 132178392.0),
        (0.0, 0.0, 8442759680.0, 140712672.0, -0.0, 0.0, 0.0, 0.3553340435028076, 0.3553340435028076, 0.0, 0.0, 8442759680.0, 140712672.0),
        (0.0, 0.0, 8987915264.0, 149798592.0, -0.0, 0.0, 0.0, 0.33378151059150696, 0.33378151059150696, 0.0, 0.0, 8987915264.0, 149798592.0),
    )),
    "aero-reduces-to-classic": (10, (
        (0.0, 87746416.0, 263239232.0, 877464.125, -8774642.0, 8703404.0, -1.3279492855072021, 1.1400806903839111, 1.1396477222442627, 0.0, 0.0, 350273280.0, 877450.875),
        (0.0, 92025960.0, 276077888.0, 920259.625, -9202596.0, 9003012.0, -2.094386339187622, 1.0885814428329468, 1.0866498947143555, 0.0, 0.0, 366108000.0, 920238.6875),
        (0.00023584612063132226, 96564376.0, 289693120.0, 965643.75, -9562048.0, 9342127.0, -2.1324191093444824, 1.0385808944702148, 1.0355786085128784, 0.00023584612063132226, 943897.1875, 383114400.0, 965622.4375),
        (0.0003842678270302713, 101367312.0, 304101952.0, 1013673.125, -10036535.0, 9917921.0, -1.4019719362258911, 0.9901026487350464, 0.9865112900733948, 0.0003842678270302713, 1028144.0, 403281152.0, 1013659.125),
        (0.00010038772597908974, 106441560.0, 319324704.0, 1064415.625, -10538010.0, 10507849.0, -0.598703920841217, 0.9433382153511047, 0.9394826292991638, 0.00010038772597908974, 1061465.125, 424403200.0, 1064409.625),
        (0.0, 111794784.0, 335384352.0, 1117947.75, -11179479.0, 11192972.0, -0.17210711538791656, 0.8962605595588684, 0.8944961428642273, 0.0, 0.0, 447314080.0, 1117946.0),
        (0.0, 117435512.0, 352306528.0, 1174355.125, -11743551.0, 11752925.0, 0.0, 0.851894736289978, 0.8515312075614929, 0.0, 0.0, 469835776.0, 1174355.125),
        (0.0, 123373064.0, 370119200.0, 1233730.625, -12337307.0, 12337602.0, 0.0, 0.8105901479721069, 0.8105496764183044, 0.0, 0.0, 493495232.0, 1233730.625),
        (-5.684341886080802e-14, 129617568.0, 388852704.0, 1296175.75, -12961757.0, 12961762.0, 0.0, 0.7715027928352356, 0.7715003490447998, 0.0, 0.0, 518470336.0, 1296175.75),
        (0.0, 136179840.0, 408539520.0, 1361798.375, -13617984.0, 3854316.0, 0.0, 0.7343231439590454, 0.7343230843544006, 0.0, 0.0, 447082688.0, 1361798.375),
        (0.0, 143071408.0, 429214240.0, 1430714.125, -14307141.0, 69728.0, 0.0, 0.6989516615867615, 0.6989516615867615, 0.0, 0.0, 429911520.0, 1430714.125),
        (2.168404344971009e-19, 150304352.0, 450913088.0, 1503043.625, -15030436.0, 681.6000366210938, 0.0, 0.6653167009353638, 0.6653167009353638, 0.0, 0.0, 450919904.0, 1503043.625),
        (0.0, 0.0, 473674464.0, 1578914.875, -0.0, 0.0, 0.0, 0.6333463191986084, 0.6333463191986084, 0.0, 0.0, 473674464.0, 1578914.875),
        (0.0, 0.0, 497538432.0, 1658461.375, -0.0, 0.0, 0.0, 0.6029685139656067, 0.6029685139656067, 0.0, 0.0, 497538432.0, 1658461.375),
        (0.0, 0.0, 522546656.0, 1741822.25, -0.0, 0.0, 0.0, 0.5741115212440491, 0.5741115212440491, 0.0, 0.0, 522546656.0, 1741822.25),
        (0.0, 0.0, 548742272.0, 1829141.0, -0.0, 0.0, 0.0, 0.5467047095298767, 0.5467047095298767, 0.0, 0.0, 548742272.0, 1829141.0),
        (0.0, 0.0, 576170496.0, 1920568.25, -0.0, 0.0, -1440.7490234375, 0.5206791162490845, 0.520679235458374, 0.0, 0.0, 576170496.0, 1906160.75),
        (0.0, 0.0, 604877120.0, 2016257.125, -0.0, 0.0, -2202.623779296875, 0.49596843123435974, 0.4959684908390045, 0.0, 0.0, 604877120.0, 1994230.875),
        (0.0, 0.0, 634910080.0, 2116367.0, -0.5, -7229573.0, -3313.09375, 0.47244057059288025, 0.47250786423683167, 0.0, 0.0, 562614336.0, 2083236.0),
        (0.0, 0.0, 666318144.0, 2221060.5, 0.0, -7587210.0, -4921.01806640625, 0.45009368658065796, 0.45023536682128906, 0.0, 0.0, 590446080.0, 2171850.25),
        (0.0, 0.0, 699151424.0, 2330504.75, 0.0, -7961078.5, -7238.48583984375, 0.4289039969444275, 0.42909160256385803, 0.0, 0.0, 619540608.0, 2258120.0),
        (0.0, 0.0, 733460800.0, 2444869.5, 0.0, -8351759.0, -10568.267578125, 0.40880849957466125, 0.40901979804039, 0.0, 0.0, 649943232.0, 2339186.75),
        (0.0, 0.0, 769298816.0, 2564329.5, 0.0, -8759844.0, -15343.52734375, 0.38974738121032715, 0.3899655044078827, 0.0, 0.0, 681700352.0, 2410894.25),
        (0.0, 0.0, 806717952.0, 2689059.75, 0.0, -9185939.0, -22185.423828125, 0.3716640770435333, 0.37187719345092773, 0.0, 0.0, 714858560.0, 2467205.5),
    )),
    "aero-cloud-freeze-nc": (10, (
        (0.00010036135790869594, 21328910.0, 213289104.0, 710963.6875, -440368.25, 62381.05078125, 0.0, 1.4073185920715332, 1.4065415859222412, 0.00010036135790869594, 16925250.0, 213912912.0, 710963.6875),
        (0.00019456296286080033, 75683048.0, 227049152.0, 756830.5, -5498039.5, 4869815.0, 0.0, 1.3220783472061157, 1.3212997913360596, 0.00019456296286080033, 20702680.0, 275747296.0, 756830.5),
        (0.0002009438758250326, 241697264.0, 241697264.0, 805657.5625, -23934122.0, 23325046.0, 0.0, 1.2420129776000977, 1.2412221431732178, 0.0002009438758250326, 2356051.0, 474947712.0, 805657.5625),
        (0.00011036646901629865, 857635648.0, 257290688.0, 857635.625, -84906968.0, 84803528.0, 0.0, 1.1667972803115845, 1.1659963130950928, 0.00011036646901629865, 8565962.0, 1105325952.0, 857635.625),
        (1.4771649148315191e-06, 1643342848.0, 273890464.0, 912968.25, -162691152.0, 162670352.0, 0.0, 1.0961278676986694, 1.0953283309936523, 1.4771649148315191e-06, 16431382.0, 1900593920.0, 912968.25),
        (0.0, 29156170.0, 291561696.0, 971872.3125, -2915617.0, 2896874.75, 0.0, 1.029160499572754, 1.0289417505264282, 0.0, 0.0, 320530432.0, 971872.3125),
        (-9.094947017729282e-13, 103457840.0, 310373536.0, 1034578.4375, -10345785.0, 10345482.0, 0.0, 0.9666172862052917, 0.9665772914886475, 0.0, 0.0, 413828352.0, 1034578.4375),
        (5.684341886080802e-14, 330399616.0, 330399616.0, 1101332.0, -33039958.0, 33039958.0, 0.0, 0.9079961180686951, 0.9079914093017578, 0.0, 0.0, 660799232.0, 1101332.0),
        (3.552713678800501e-15, 1172394624.0, 351718400.0, 1172394.625, -117239464.0, 26493038.0, 0.0, 0.8529554009437561, 0.8529551029205322, 0.0, 0.0, 616648768.0, 1172394.625),
        (0.0, 0.0, 374413440.0, 1248044.875, -0.0, 0.0, 0.0, 0.8012532591819763, 0.8012532591819763, 0.0, 0.0, 374413440.0, 1248044.875),
        (0.0, 0.0, 398573728.0, 1328579.125, -0.0, 0.0, 0.0, 0.7526838183403015, 0.7526838183403015, 0.0, 0.0, 398573728.0, 1328579.125),
        (0.0, 0.0, 424293952.0, 1414313.125, -0.0, 0.0, 0.0, 0.707056999206543, 0.707056999206543, 0.0, 0.0, 424293952.0, 1414313.125),
        (0.0, 0.0, 451674784.0, 1505582.625, -0.0, 0.0, 0.0, 0.6641947031021118, 0.6641947031021118, 0.0, 0.0, 451674784.0, 1505582.625),
        (0.0, 0.0, 480823776.0, 1602746.0, -0.0, 0.0, 0.0, 0.6239292025566101, 0.6239292025566101, 0.0, 0.0, 480823776.0, 1602746.0),
        (0.0, 0.0, 511855200.0, 1706184.0, -0.0, 0.0, 0.0, 0.5861032605171204, 0.5861032605171204, 0.0, 0.0, 511855200.0, 1706184.0),
        (0.0, 0.0, 544890688.0, 1816302.25, -0.0, 0.0, 0.0, 0.5505691766738892, 0.5505691766738892, 0.0, 0.0, 544890688.0, 1816302.25),
        (0.0, 0.0, 580060032.0, 1933533.5, -0.0, 0.0, 0.0, 0.5171878337860107, 0.5171878337860107, 0.0, 0.0, 580060032.0, 1933533.5),
        (0.0, 0.0, 617501056.0, 2058336.875, -0.0, 0.0, 0.0, 0.48582911491394043, 0.48582911491394043, 0.0, 0.0, 617501056.0, 2058336.875),
        (0.0, 0.0, 657361024.0, 2191203.5, -0.0, 0.0, 0.0, 0.4563702344894409, 0.4563702344894409, 0.0, 0.0, 657361024.0, 2191203.5),
        (0.0, 0.0, 699796224.0, 2332654.25, -0.0, 0.0, 0.0, 0.4286961853504181, 0.4286961853504181, 0.0, 0.0, 699796224.0, 2332654.25),
        (0.0, 0.0, 744973504.0, 2483245.0, -0.0, 0.0, 0.0, 0.4026988744735718, 0.4026988744735718, 0.0, 0.0, 744973504.0, 2483245.0),
        (0.0, 0.0, 793070400.0, 2643568.0, -0.0, 0.0, 0.0, 0.37827664613723755, 0.37827664613723755, 0.0, 0.0, 793070400.0, 2643568.0),
        (0.0, 0.0, 844275968.0, 2814253.25, -0.0, 0.0, 0.0, 0.3553340435028076, 0.3553340435028076, 0.0, 0.0, 844275968.0, 2814253.25),
        (0.0, 0.0, 898791552.0, 2995971.75, -0.0, 0.0, 0.0, 0.33378151059150696, 0.33378151059150696, 0.0, 0.0, 898791552.0, 2995971.75),
    )),
}



def test_terminal_apply_matches_wrfs_own_terminal_loop_on_every_fixture():
    """:3972-4021 against WRF's OWN inputs and answers -- and the density.

    THE DEFECT THIS FIXES IS NOT IN THE KERNEL.  ``thompson_aa_state_finalize``
    reproduces WRF's terminal loop BITWISE on all 72 states x 4 output fields
    below when it is handed the density WRF actually has there.  What was
    wrong is WHICH density the port hands it: ``rho`` was documented as "ENTRY
    density, :1802" and ``woof/core/microphysics_aerosol.py`` passes exactly
    that.

    ``rho(k)`` is written in four places in ``mp_thompson`` -- :1802 (entry),
    :3193 (the unconditional TAU+1 refresh), :3490 (per level, inside the
    condensation block) and :3572 (per level, inside rain evaporation) -- and
    nothing after :3574 touches it.  So :3976, :4011, :4019 and :4020 read
    whichever of the last three ran.  MEASURED over all 21 fixtures x 24
    levels: ``max |rho_terminal - rho_entry| / rho_terminal = 7.4672e-03``.

    THE "21 FIXTURES" AND "504 LEVELS" IN THIS DOCSTRING ARE THE DECK AS IT
    STOOD WHEN THE INSTRUMENTED WRF RUN WAS MADE, AND THE DECK IS NOW 22
    COLUMNS (528 levels).  Re-deriving the three numbers below needs WRF's own
    terminal ``rho(k)``, which no committed fixture records -- the CSVs carry
    the before/after state of the whole call, not a mid-call density -- so
    they are left as the measurement they were rather than silently re-scaled.
    What IS re-measured here, on the device, every run, is the consequence:
    the ``run(...)`` calls below drive the finalize kernel with both densities
    and require the difference in both directions.  WP-14 re-ran this file:
    all of it passes on the current tree.

    THE COST, MEASURED INSIDE WRF ITSELF and reproduced here.  Recomputing
    :3976-4021 with ``rho_entry`` substituted, in an instrumented pristine
    build, moves ``nc1d`` at exactly ONE of 504 fixture levels --
    aero-cold-overlap k=5, ``1.833336115 -> 1.826136470`` kg^-1, i.e.
    ``3.9271e-03`` relative, 1963x the 2.0e-06 end-to-end gate -- through
    :3976's ``2./rho(k)`` floor.  This test reproduces that number on the
    device, in both directions, so the size of the defect is a receipt and not
    an adjective.  :4011's ``nu_c`` and :4020's ``Nt_c_max/rho(k)`` read the
    same density; they are an INTEGER selector and a ceiling, so on these
    fixtures they are latent rather than quiet.

    THE FIX IS IN THE CALLER and is filed as an integration request:
    ``launch_tau1_density(temperature, state.p, state.qv, tau1_density)``
    immediately after ``launch_aerosol_rain_evaporation`` and before
    sedimentation, then pass ``tau1_density`` here instead of
    ``entry_density``.  MEASURED: taken at that point it reproduces WRF's
    terminal ``rho(k)`` bitwise at 501 of the 504 levels (worst 1.2334e-07);
    taken at the finalize call instead it is 5.39e-05 off, because WOOF-ArWen's
    ``temperature`` keeps absorbing the melt/freeze cleanup's ``tten`` while
    WRF's ``temp(k)`` snapshot does not.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_state_finalize)

    def run(table, density_column):
        dt, rows = table
        values = np.asarray(rows, dtype=np.float64)
        qc = _dev(values[:, 0].astype(F))
        args = [qc] + [_dev(values[:, i].astype(F)) for i in range(1, 7)]
        args.append(_dev(values[:, density_column].astype(F)))
        outs = [cp.empty_like(qc) for _ in range(3)]
        launch_aerosol_state_finalize(*args, dt, *outs)
        cp.cuda.Stream.null.synchronize()
        got = [cp.asnumpy(qc)] + [cp.asnumpy(o) for o in outs]
        want = [values[:, i].astype(F) for i in (9, 10, 11, 12)]
        return got, want

    fields = ("qc", "nc", "nwfa", "nifa")

    # (1) WITH WRF'S OWN TERMINAL DENSITY: bitwise, every field, every level.
    compared = 0
    for scenario, table in _WRF_TERMINAL_LOOP.items():
        got, want = run(table, 7)
        for name, mine, theirs in zip(fields, got, want):
            np.testing.assert_array_equal(
                mine, theirs, err_msg=f"{scenario} {name}")
            compared += mine.size
    assert compared == 3 * 24 * 4

    # (2) WITH THE ENTRY DENSITY the adapter passes today: exactly one state
    #     moves, and it moves by the number WRF itself produced.
    moved = {}
    for scenario, table in _WRF_TERMINAL_LOOP.items():
        got, want = run(table, 8)
        for name, mine, theirs in zip(fields, got, want):
            bad = np.nonzero(mine != theirs)[0]
            for k in bad:
                moved[(scenario, name, int(k))] = (
                    abs(float(mine[k]) - float(theirs[k]))
                    / abs(float(theirs[k])))
    assert set(moved) == {("aero-cold-overlap", "nc", 4)}, sorted(moved)
    assert moved[("aero-cold-overlap", "nc", 4)] == pytest.approx(
        3.9271e-03, rel=1.0e-3), moved


def test_state_finalize_output_may_alias_its_input():
    """The adapter passes state.nc for both nc and nc_out; each thread reads
    its own element before writing it, so that is legal.  Pinned here because
    the contract depends on it.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_state_finalize)

    rows = _column("aero-warm-overlap", "after")
    qc = _f32(rows, "qc")
    nc = _f32(rows, "nc_per_kg")
    nwfa = _f32(rows, "nwfa_per_kg")
    nifa = _f32(rows, "nifa_per_kg")
    rho = _host_density(_f32(rows, "p_pa"), _f32(rows, "temp_k"),
                        np.maximum(F(1.0e-10), _f32(rows, "qv")))
    ncten = np.full(qc.size, 1.0e6, dtype=np.float32)
    nwfaten = np.full(qc.size, -1.0e6, dtype=np.float32)
    nifaten = np.full(qc.size, 5.0e2, dtype=np.float32)
    dt = 15.0

    def run(alias):
        d_qc = _dev(qc)
        d_nc, d_nwfa, d_nifa = _dev(nc), _dev(nwfa), _dev(nifa)
        if alias:
            outs = (d_nc, d_nwfa, d_nifa)
        else:
            outs = (cp.empty_like(d_nc), cp.empty_like(d_nc),
                    cp.empty_like(d_nc))
        launch_aerosol_state_finalize(
            d_qc, d_nc, d_nwfa, d_nifa, _dev(ncten), _dev(nwfaten),
            _dev(nifaten), _dev(rho), dt, *outs)
        cp.cuda.Stream.null.synchronize()
        return tuple(cp.asnumpy(o) for o in outs)

    for aliased, separate in zip(run(True), run(False)):
        np.testing.assert_array_equal(aliased, separate)


def test_zero_aerosol_accumulators_is_the_single_reset_point():
    import cupy as cp

    from woof.core.thompson_aerosol_state import zero_aerosol_accumulators

    shape = (4, 3, 2)
    tens = [cp.full(shape, 7.0, dtype=cp.float32) for _ in range(3)]
    zero_aerosol_accumulators(*tens)
    cp.cuda.Stream.null.synchronize()
    for t in tens:
        assert not bool(cp.any(t))


# ===========================================================================
# Kernel 6 -- effective radius.
# ===========================================================================

def test_effective_radius_matches_wrf_calc_effectrad_probe():
    """ORACLE gate.  probe-effectrad.csv is a DIRECT call to calc_effectRad
    with a droplet ladder chosen to straddle every branch, including the two
    (nc < 100 and nc > 1e10) that mp_gt_driver cannot reach.  Metres in,
    metres out -- no driver clamp and no micron conversion.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius)

    rows = _probe("effectrad")
    # The harness regeneration WIDENED this probe from 14 rows to 50.  The
    # count is asserted, not inferred, so a probe that shrinks back -- which
    # would silently narrow this gate -- fails here rather than passing
    # quietly on fewer states.
    assert len(rows) == 50
    args = [_dev(_f32(rows, key)) for key in
            ("temp_k", "p_pa", "qv", "qc", "nc_per_kg", "qi", "ni_per_kg",
             "qs")]
    outs = [cp.empty_like(args[0]) for _ in range(3)]
    launch_aerosol_effective_radius(*args, *outs, metres=True)
    cp.cuda.Stream.null.synchronize()

    # TIGHTENED from rtol (1.2e-7, 1.2e-7, 5.0e-6) to BITWISE, and the
    # tightening now means something on all three branches.  The 14-row probe
    # carried ONE temperature and ONE snow content, so its effs_m column was a
    # single state repeated fourteen times; the 50-row probe carries 14
    # temperatures, 10 cloud contents, 27 droplet numbers and 9 snow contents,
    # and 22 DISTINCT effs_m values.  Those counts are asserted below so a
    # future regeneration cannot quietly collapse the ladders again.
    for out, key in ((outs[0], "effc_m"), (outs[1], "effi_m"),
                     (outs[2], "effs_m")):
        np.testing.assert_array_equal(cp.asnumpy(out), _f32(rows, key),
                                      err_msg=key)
    assert len(set(_f32(rows, "temp_k").tolist())) == 14
    assert len(set(_f32(rows, "effs_m").tolist())) == 22
    assert len(set(_f32(rows, "effc_m").tolist())) == 25


def test_effective_radius_probe_exercises_all_three_inu_c_branches():
    """The nc<100, general and nc>1e10 selectors must all be hit by the probe
    row set, otherwise the gate above proves less than it appears to.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_launch import probe_inu_c_effrad

    rows = _probe("effectrad")
    nc_m3 = _f32(rows, "nc_target_per_m3")
    inu_c = cp.asnumpy(probe_inu_c_effrad(_dev(nc_m3)))
    cp.cuda.Stream.null.synchronize()
    assert (inu_c[nc_m3 < 100.0] == 15).all()
    assert (inu_c[nc_m3 > 1.0e10] == 2).all()
    middle = (nc_m3 >= 100.0) & (nc_m3 <= 1.0e10)
    assert middle.any() and set(inu_c[middle].tolist()) - {15, 2}
    # And the mp=8 identity: Nt_c = 100e6 selects inu_c = 12, whose exact
    # g_ratio entry is the 2730 thompson.cu:343 hardcodes.
    assert int(cp.asnumpy(probe_inu_c_effrad(_dev(np.asarray([1.0e8]))))[0]) \
        == 12


@pytest.mark.parametrize("scenario", ("aero-nc-effrad", "aero-nc-cap"))
def test_effective_radius_matches_wrf_column_fixture(scenario):
    """ORACLE gate.  mp_gt_driver calls calc_effectRad with the FINAL column
    state, which is exactly the fixtures' 'after' rows, then applies its own
    clamps at :1476-1478.  woof's contract is microns, so the expectation
    applies the identical float32 conversion re*1.E6.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius)

    rows = _column(scenario, "after")
    args = [_dev(_f32(rows, key)) for key in
            ("temp_k", "p_pa", "qv", "qc", "nc_per_kg", "qi", "ni_per_kg",
             "qs")]
    outs = [cp.empty_like(args[0]) for _ in range(3)]
    launch_aerosol_effective_radius(*args, *outs)
    cp.cuda.Stream.null.synchronize()

    # TIGHTENED from rtol=5.0e-6 / atol=2.0e-5 to BITWISE on both scenarios.
    for out, key, (lo, hi) in zip(outs, ("effc_m", "effi_m", "effs_m"),
                                  _DRIVER_CLAMPS):
        expected = np.maximum(lo, np.minimum(_f32(rows, key), hi)) * F(1.0e6)
        np.testing.assert_array_equal(cp.asnumpy(out), expected, err_msg=key)


def test_effective_radius_is_bitwise_against_every_oracle_after_column():
    """ORACLE gate, the strong one.  ``mp_gt_driver:1471-1473`` calls
    ``calc_effectRad`` with the FINAL column state, which is exactly the
    fixtures' 'after' rows, so driving this kernel with those rows isolates
    ``calc_effectRad`` from every upstream residual in the port.

    All 19 fixtures, all three fields, 456 levels each: BITWISE, except the
    eight levels in :data:`_FIXTURE_TEMPERATURE_ROUND_TRIP_LEVELS`, each of
    which is proved here to be the harness's own float32 temperature
    round trip rather than a kernel defect.

    This is what makes the end-to-end effc/effi residuals attributable: if
    ``calc_effectRad`` is exact on WRF's own state, anything G3 sees in
    ``effc_m``/``effi_m`` is inherited from ``qc``/``nc``/``qi``/``ni``.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius)

    scenarios = sorted(p.name[: -len("-column.csv")]
                       for p in _ORACLE.glob("aero-*-column.csv"))
    assert len(scenarios) == 19

    unexplained = []
    proved = 0
    for scenario in scenarios:
        rows = _column(scenario, "after")
        args = [_dev(_f32(rows, key)) for key in
                ("temp_k", "p_pa", "qv", "qc", "nc_per_kg", "qi", "ni_per_kg",
                 "qs")]
        outs = [cp.empty_like(args[0]) for _ in range(3)]
        launch_aerosol_effective_radius(*args, *outs, metres=True)
        cp.cuda.Stream.null.synchronize()

        temperature = _f32(rows, "temp_k")
        exner = _f32(rows, "pii")
        for out, field in zip(outs, ("effc_m", "effi_m", "effs_m")):
            got = cp.asnumpy(out)
            want = _f32(rows, field)
            allowed = _FIXTURE_TEMPERATURE_ROUND_TRIP_LEVELS.get(
                (scenario, field), ())
            for k in np.nonzero(got != want)[0]:
                if int(k) not in allowed:
                    unexplained.append(
                        f"{scenario} {field} k={k + 1} got={got[k]!r} "
                        f"want={want[k]!r}")
                    continue
                # PROVE it: some t1d within +-4 ulp of the recorded temp_k
                # must reproduce the fixture EXACTLY and must round-trip
                # through WRF's own th = t1d/pii, temp = th*exner.
                solved = False
                for delta in range(-4, 5):
                    if delta == 0:
                        continue
                    candidate = temperature[k]
                    for _ in range(abs(delta)):
                        candidate = np.nextafter(
                            candidate,
                            F(np.inf if delta > 0 else -np.inf),
                            dtype=np.float32)
                    if F(F(candidate / exner[k]) * exner[k]) != temperature[k]:
                        continue
                    probe = [_dev(np.asarray([candidate]))]
                    probe += [_dev(np.asarray([_f32(rows, key)[k]]))
                              for key in ("p_pa", "qv", "qc", "nc_per_kg",
                                          "qi", "ni_per_kg", "qs")]
                    trial = [cp.empty_like(probe[0]) for _ in range(3)]
                    launch_aerosol_effective_radius(*probe, *trial,
                                                    metres=True)
                    cp.cuda.Stream.null.synchronize()
                    index = ("effc_m", "effi_m", "effs_m").index(field)
                    if cp.asnumpy(trial[index])[0] == want[k]:
                        solved = True
                        break
                if solved:
                    proved += 1
                else:
                    unexplained.append(
                        f"{scenario} {field} k={k + 1} is NOT a temperature "
                        f"round trip: got={got[k]!r} want={want[k]!r}")

    assert not unexplained, "\n".join(unexplained)
    assert proved == sum(len(v) for v
                         in _FIXTURE_TEMPERATURE_ROUND_TRIP_LEVELS.values()), (
        "the round-trip level list is stale; every entry must still be a "
        "real, proved exception")


def _host_calc_effect_rad(t_k, p_pa, qv, qc, nc, qi, ni, qs,
                          pow_fn=None):
    """calc_effectRad:5624-5695 for one level, in float32, term for term.

    Every REAL(4) operation is rounded to float32 where WRF rounds it, every
    ``**`` goes through the libm ``powf`` gfortran emits, and only ``lamc`` /
    ``lami`` are DOUBLE.  Returns metres, i.e. before mp_gt_driver's second
    clamp.

    ``pow_fn`` selects the float32 power: the default is the libm ``powf``
    gfortran emits, and the alternative is the device's round-once-from-double
    ``thompson_aa_powf``.  Having both lets a mismatch be ATTRIBUTED rather
    than tolerated.
    """
    if pow_fn is None:
        pow_fn = _glibc_powf
    # G_RATIO is the index-0-unused form, so G_RATIO[n] is Fortran's
    # g_ratio(n) directly -- calc_effectRad:5611-5613's exact integers.
    from woof.core.thompson_aerosol_contract import AM_R, G_RATIO

    rho = _host_density(F(p_pa), F(t_k), F(qv))
    rc = max(_R1, F(F(qc) * rho))
    nc_m3 = max(F(2.0), min(F(F(nc) * rho), _NT_C_MAX))
    ri = max(_R1, F(F(qi) * rho))
    ni_m3 = max(_R2, F(F(ni) * rho))
    rs = max(_R1, F(F(qs) * rho))

    reqc, reqi, reqs = F(2.49e-6), F(4.99e-6), F(9.99e-6)

    if rc > _R1 and nc_m3 > _R2:
        # :5639-5645
        if nc_m3 < F(100.0):
            inu_c = 15
        elif nc_m3 > F(1.0e10):
            inu_c = 2
        else:
            inu_c = min(15, int(math.floor(F(F(1000.0e6) / nc_m3) + 0.5)) + 2)
        # :5646
        lamc = float(pow_fn(
            F(F(F(nc_m3 * F(AM_R)) * F(G_RATIO[inu_c])) / rc), _OBMR))
        # :5647
        reqc = max(F(2.51e-6), min(F(0.5 * float(F(F(3.0) + F(inu_c)))
                                     / lamc), F(50.0e-6)))
    if ri > _R1 and ni_m3 > _R2:
        # :137 am_i = PI*rho_i/6 with rho_i = 890, folded in REAL(4).
        am_i = F(F(F(3.1415926536) * F(890.0)) / F(6.0))
        # :5654  am_i*cig(2)*oig1 with cig(2) = WGAMMA(4) = 6 and oig1 = 1.
        lami = float(pow_fn(
            F(F(F(F(am_i * F(6.0)) * F(1.0)) * ni_m3) / ri), _OBMR))
        # :5655  mu_i = 0.
        reqi = max(F(2.51e-6), min(F(0.5 * float(F(F(3.0) + F(0.0))) / lami),
                                   F(125.0e-6)))
    if rs > _R1:
        tc0 = min(F(-0.1), F(F(t_k) - F(273.15)))            # :5662
        smob = F(rs * F(F(1.0) / F(0.069)))                  # :5663
        moment = F(3.0)                                      # cse(1)

        def fit(coefficients):
            value = F(coefficients[0])
            value = F(value + F(F(coefficients[1]) * tc0))
            value = F(value + F(F(coefficients[2]) * moment))
            value = F(value + F(F(F(coefficients[3]) * tc0) * moment))
            value = F(value + F(F(F(coefficients[4]) * tc0) * tc0))
            value = F(value + F(F(F(coefficients[5]) * moment) * moment))
            value = F(value + F(F(F(F(coefficients[6]) * tc0) * tc0)
                                * moment))
            value = F(value + F(F(F(F(coefficients[7]) * tc0) * moment)
                                * moment))
            value = F(value + F(F(F(F(coefficients[8]) * tc0) * tc0) * tc0))
            value = F(value + F(F(F(F(coefficients[9]) * moment) * moment)
                                * moment))
            return value

        sa = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
              0.31255, 0.000204, 0.003199, 0.0, -0.015952)      # :167-172
        sb = (0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
              0.060366, 0.000079, 0.000594, 0.0, -0.003577)     # :176-181
        a_ = pow_fn(F(10.0), fit(sa))                      # :5689
        smoc = F(a_ * pow_fn(smob, fit(sb)))               # :5694
        reqs = max(F(5.01e-6), min(F(F(0.5) * F(smoc / smob)),   # :5695
                                   F(999.0e-6)))
    return reqc, reqi, reqs


def test_effective_radius_is_bitwise_against_a_fortran_faithful_host_sweep():
    """WIDE-RANGE gate.  The 19 fixtures reach 456 states; this reaches 2400,
    chosen to cover ground they do not: the ``nc < 100`` and ``nc > 1e10``
    ``inu_c`` branches, the 2.51/50 um and 5.01/999 um clamp ends, snow from
    -0.1 to -73 C, and densities from 0.24 to 1.3 kg m^-3.

    The reference is a float32 host transcription of :5624-5695 that rounds
    where WRF rounds and calls the same ``libm.powf`` gfortran emits, so
    "bitwise" here means the kernel makes the same arithmetic choices as the
    Fortran, not merely that it lands close.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius)

    rng = np.random.default_rng(4661)
    n = 2400
    t_k = rng.uniform(200.0, 300.0, n).astype(np.float32)
    p_pa = rng.uniform(2.0e4, 1.02e5, n).astype(np.float32)
    qv = (10.0 ** rng.uniform(-6.0, -1.7, n)).astype(np.float32)
    qc = (10.0 ** rng.uniform(-13.0, -2.0, n)).astype(np.float32)
    nc = (10.0 ** rng.uniform(-1.0, 10.0, n)).astype(np.float32)
    qi = (10.0 ** rng.uniform(-13.0, -3.0, n)).astype(np.float32)
    ni = (10.0 ** rng.uniform(0.0, 7.0, n)).astype(np.float32)
    qs = (10.0 ** rng.uniform(-13.0, -2.0, n)).astype(np.float32)
    # Force a slab of exactly-zero hydrometeors so the background branches
    # are exercised too.
    qc[:100] = F(0.0)
    qi[100:200] = F(0.0)
    qs[200:300] = F(0.0)

    fields = (t_k, p_pa, qv, qc, nc, qi, ni, qs)
    outs = [cp.empty(n, cp.float32) for _ in range(3)]
    launch_aerosol_effective_radius(*[_dev(f) for f in fields], *outs,
                                    metres=True)
    cp.cuda.Stream.null.synchronize()
    got = [cp.asnumpy(o) for o in outs]

    want = np.empty((3, n), np.float32)
    for i in range(n):
        want[:, i] = _host_calc_effect_rad(*(f[i] for f in fields))

    # Branch coverage, asserted rather than assumed.
    rho = _host_density(p_pa, t_k, qv)
    nc_m3 = np.maximum(F(2.0), np.minimum(nc * rho, _NT_C_MAX))
    assert (nc_m3 < F(100.0)).sum() >= 50, "nc < 100 branch under-sampled"
    assert (nc_m3 > F(1.0e10)).sum() == 0, (
        ":5626 caps nc at Nt_c_max, so :5641's nc > 1.E10 branch is "
        "unreachable from a kernel launch -- probe-effectrad.csv is the only "
        "place it can be exercised")
    assert (got[0] == F(2.49e-6)).sum() >= 90, "background effc under-sampled"
    assert (got[0] >= F(49.9e-6)).sum() >= 5, "upper effc clamp unexercised"
    assert (got[2] >= F(998.0e-6)).sum() >= 5, "upper effs clamp unexercised"

    # effc and effi: BITWISE against the Fortran on all 2400 states.
    np.testing.assert_array_equal(got[0], want[0], err_msg="effc")
    np.testing.assert_array_equal(got[1], want[1], err_msg="effi")

    # effs: bitwise on all but a handful, and every exception ATTRIBUTED.
    # :5689 and :5694 make two non-integer ``**`` calls per level, and the
    # device's stand-in for glibc powf (evaluate in double, round once) is
    # not glibc powf -- MEASURED, they differ on about 0.09% of arguments.
    # For every mismatch, rerunning the host with the device's own power must
    # reproduce the kernel EXACTLY.  That turns a tolerance into a proof, and
    # it is the tightest statement available without shipping a bit-exact
    # reimplementation of glibc's powf on the GPU.
    def _round_once_from_double(base, exponent):
        return F(np.float64(F(base)) ** np.float64(F(exponent)))

    mismatched = np.nonzero(got[2] != want[2])[0]
    assert mismatched.size <= 6, (
        f"{mismatched.size} of {n} snow levels disagree; the libm-vs-"
        f"round-once difference measures ~0.09% per power call")
    for i in mismatched:
        assert abs(float(got[2][i]) - float(want[2][i])) <= float(
            np.spacing(np.float32(want[2][i]))), (
            f"snow mismatch at {i} exceeds one float32 ulp")
        alternative = _host_calc_effect_rad(
            *(f[i] for f in fields), pow_fn=_round_once_from_double)[2]
        assert got[2][i] == alternative, (
            f"snow mismatch at {i} is NOT the powf stand-in: kernel "
            f"{got[2][i]!r}, libm {want[2][i]!r}, stand-in {alternative!r}")


def test_effective_radius_residual_is_inherited_from_qc_and_nc():
    """ATTRIBUTION.  ``re_qc = 0.5*(3+inu_c)/lamc`` with
    ``lamc = (nc*am_r*g_ratio/rc)**(1/3)``, so away from the ``inu_c`` steps a
    relative error eps in qc moves effc by eps/3 and a relative error in nc
    moves it by -eps/3.

    That is the whole explanation for G3's ``effc_m`` column.  Measured on the
    tree at the time of writing, with the sibling packages' residuals as they
    stood: aero-cold-overlap ``nc_per_kg`` 5.715e-05 -> ``effc_m`` 1.900e-05
    (ratio 0.3325), aero-cloud-freeze-nc ``qc`` 1.478e-05 -> ``effc_m``
    5.018e-06 (0.3395), aero-ice-koop ``|qi - ni|`` 1.52e-04 -> ``effi_m``
    5.093e-05 (0.335).  This test pins the mechanism so the attribution
    survives whatever the sibling residuals do next.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius)

    rows = _column("aero-nc-effrad", "after")
    base = [_f32(rows, key) for key in
            ("temp_k", "p_pa", "qv", "qc", "nc_per_kg", "qi", "ni_per_kg",
             "qs")]
    active = ((_f32(rows, "qc") > F(1.0e-9))
              & (_f32(rows, "nc_per_kg") > F(1.0)))
    assert active.sum() >= 8, "need a well-populated cloudy column"

    def effc(fields):
        outs = [cp.empty_like(_dev(fields[0])) for _ in range(3)]
        launch_aerosol_effective_radius(*[_dev(f) for f in fields], *outs,
                                        metres=True)
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(outs[0])

    reference = effc(base)
    for index, name, sign in ((3, "qc", +1.0), (4, "nc_per_kg", -1.0)):
        for eps in (1.0e-4, 1.0e-3):
            perturbed = list(base)
            perturbed[index] = (base[index] * F(1.0 + eps)).astype(np.float32)
            moved = effc(perturbed)
            ratio = ((moved[active] - reference[active])
                     / reference[active] / (sign * eps))
            # One third, to within the float32 noise of a 1e-4 perturbation.
            assert np.allclose(ratio, 1.0 / 3.0, atol=2.0e-3), (
                f"{name} eps={eps}: d(effc)/effc / (d{name}/{name}) = "
                f"{ratio.min():.6f}..{ratio.max():.6f}, expected 1/3")


def test_effective_radius_reduces_to_the_frozen_mp8_kernel_at_nt_c():
    """IDENTITY gate, AND the receipt for the one place mp=28 leaves mp=8.

    At nc = Nt_c the generalized prognostic form must reproduce woof's
    model-validated mp=8 kernel, which hardcodes 100.0e6f and the
    exact-integer g_ratio 2730.0f.  If it does not, the mp=28 shape selector
    or the g_ratio table is wrong in a way that would ALSO have broken mp=8
    had the kernels been shared.  That part is still asserted BITWISE, on the
    cloud branch where the selector and the table live, and on snow.

    The ice branch is the exception, and it is measured rather than
    tolerated.  ``calc_effectRad:5654`` is ``REAL(4)**REAL(4)``, which
    gfortran lowers to glibc powf; ``thompson_aerosol_state.cu`` therefore
    evaluates it correctly rounded while byte-frozen ``thompson.cu`` uses
    CUDA's powf.  On this column they differ at 2 of 24 levels by exactly one
    float32 ulp, and it is the FROZEN kernel that misses the Fortran oracle
    there.  MP28_PORT_SPEC.md's tie-break -- already applied to
    thompson_rslf/thompson_rsif -- is that the authority is WRF, not WOOF-ArWen's
    mp=8.  Both directions are asserted below so the divergence cannot grow
    or migrate unnoticed.

    ``thompson.cu`` is byte-frozen; this only CALLS it.
    """
    import cupy as cp

    from woof.core.thompson import launch_effective_radius
    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius)

    rows = _column("aero-reduces-to-classic", "after")
    temperature = _f32(rows, "temp_k")
    pressure = _f32(rows, "p_pa")
    qv = _f32(rows, "qv")
    qc = _f32(rows, "qc")
    qi = _f32(rows, "qi")
    ni = _f32(rows, "ni_per_kg")
    qs = _f32(rows, "qs")
    rho = _host_density(pressure, temperature, qv)
    nc = (F(100.0e6) / rho).astype(np.float32)

    shared = [_dev(temperature), _dev(pressure), _dev(qv), _dev(qc)]
    classic = [cp.empty_like(shared[0]) for _ in range(3)]
    launch_effective_radius(*shared, _dev(qi), _dev(ni), _dev(qs), *classic)
    aerosol = [cp.empty_like(shared[0]) for _ in range(3)]
    launch_aerosol_effective_radius(*shared, _dev(nc), _dev(qi), _dev(ni),
                                    _dev(qs), *aerosol)
    cp.cuda.Stream.null.synchronize()

    # (1) THE IDENTITY ITSELF, still BITWISE.  effc is the branch the shape
    #     selector and the g_ratio table live in, and it is bit-for-bit equal
    #     to the frozen mp=8 kernel on all 24 levels even though the two are
    #     separate translation units with independent nvrtc inlining and
    #     contraction decisions, and even though nc*rho only recovers 100e6
    #     to float32 round-trip accuracy.  effs is likewise bitwise: this
    #     column never leaves tc0 = -0.1, where the repaired sa/sb
    #     association and thompson.cu's plain one happen to agree.
    got_c, got_i, got_s = (cp.asnumpy(a) for a in aerosol)
    want_c, want_i, want_s = (cp.asnumpy(a) for a in classic)
    np.testing.assert_array_equal(got_c, want_c, err_msg="effc")
    np.testing.assert_array_equal(got_s, want_s, err_msg="effs")

    # (2) THE ONE DIVERGENCE, MEASURED IN BOTH DIRECTIONS.  mp=28's ice branch
    #     uses thompson_aa_powf for :5654 because gfortran lowers
    #     REAL(4)**REAL(4) to glibc powf; thompson.cu uses CUDA's powf and is
    #     byte-frozen.  They differ at exactly ONE of these 24 levels, by
    #     exactly one float32 ulp -- and mp=28 is the one that matches WRF.
    #
    #     WAS [22, 23].  The oracle harness's own ``pii`` repair moved this
    #     column's after-state by ulps and level 22 now agrees; the assertion
    #     is kept as an EQUALITY on the exact level list, not a bound on its
    #     length, so the divergence cannot migrate or grow unnoticed.
    differing = np.nonzero(got_i != want_i)[0]
    assert differing.tolist() == [23], differing.tolist()
    for k in differing:
        assert abs(float(got_i[k]) - float(want_i[k])) <= float(
            np.spacing(np.float32(want_i[k]))), k

    # (3) STRICTLY STRONGER THAN THE OLD ASSERTION: against the Fortran
    #     oracle, not against a sibling port.  The ice and snow branches take
    #     no droplet number at all, so both kernels' effi/effs above are
    #     directly comparable with WRF's own column.  mp=28 is BITWISE;
    #     frozen mp=8 misses effi at exactly the two levels named in (2).
    #     The tie-break MP28_PORT_SPEC.md records for thompson_rslf/rsif --
    #     the authority is WRF, not ArWen's mp=8 -- is what this measures.
    oracle = [np.maximum(lo, np.minimum(_f32(rows, key), hi)) * F(1.0e6)
              for key, (lo, hi) in zip(("effc_m", "effi_m", "effs_m"),
                                       _DRIVER_CLAMPS)]
    np.testing.assert_array_equal(got_i, oracle[1], err_msg="mp28 effi")
    np.testing.assert_array_equal(got_s, oracle[2], err_msg="mp28 effs")
    classic_misses = np.nonzero(want_i != oracle[1])[0]
    assert classic_misses.tolist() == [23], (
        "the frozen mp=8 ice branch no longer misses the oracle where mp=28 "
        "matches it; re-measure before changing the ice powf choice")

    #     And with the fixture's OWN prognostic droplet number -- the state a
    #     forecast actually carries -- the cloud branch is bitwise too.  (The
    #     comparison above had to substitute nc = Nt_c/rho to make the frozen
    #     mp=8 kernel comparable at all; mp=8 has no prognostic nc.)
    prognostic = [cp.empty_like(shared[0]) for _ in range(3)]
    launch_aerosol_effective_radius(
        *shared, _dev(_f32(rows, "nc_per_kg")), _dev(qi), _dev(ni), _dev(qs),
        *prognostic)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(prognostic[0]), oracle[0],
                                  err_msg="mp28 effc, prognostic nc")


# ===========================================================================
# Launcher hygiene.
# ===========================================================================

def test_launchers_reject_shape_and_dtype_mistakes():
    import cupy as cp

    from woof.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius, launch_aerosol_init_profile,
        launch_aerosol_state_finalize, launch_aerosol_surface_emission)

    good = [cp.zeros((3, 2), dtype=cp.float32) for _ in range(11)]
    bad = list(good)
    bad[4] = cp.zeros((6,), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_aerosol_effective_radius(*bad)
    bad = list(good)
    bad[2] = cp.zeros((3, 2), dtype=cp.float64)
    with pytest.raises(TypeError, match="float32"):
        launch_aerosol_effective_radius(*bad)

    fin = [cp.zeros((3, 2), dtype=cp.float32) for _ in range(8)]
    outs = [cp.zeros((3, 2), dtype=cp.float32) for _ in range(3)]
    bad = list(fin)
    bad[3] = cp.zeros((3, 2), dtype=cp.float64)
    with pytest.raises(TypeError, match="float32"):
        launch_aerosol_state_finalize(*bad, 1.0, *outs)

    nwfa = cp.zeros((4, 2, 3), dtype=cp.float32)
    with pytest.raises(ValueError, match="shape"):
        launch_aerosol_surface_emission(
            nwfa, nwfa.copy(), cp.zeros((3, 2), dtype=cp.float32),
            cp.zeros((3, 2), dtype=cp.float32), 1.0)
    with pytest.raises(ValueError, match="nz"):
        launch_aerosol_init_profile(
            cp.zeros((1, 2, 3), dtype=cp.float32),
            cp.zeros((1, 2, 3), dtype=cp.float32),
            cp.zeros((1, 2, 3), dtype=cp.float32),
            cp.zeros((2, 3), dtype=cp.float32),
            fill_ccn=True, fill_in=True)


def test_state_module_is_in_the_shared_header_allow_list():
    """The kernels here call thompson_aerosol_common.cuh helpers, so the
    module must be named in woof/core/kernels/__init__.py's _EXTRA_HEADERS.
    """
    from woof.core.kernels import EXTRA_HEADERS, module_source
    from woof.core.thompson_aerosol_launch import (
        AEROSOL_COMMON_HEADER, STATE_MODULE)

    # WOOF's own libm words ride ahead of the shared header
    # (woof/core/kernels/thompson_aerosol_libm.cuh).
    assert EXTRA_HEADERS[STATE_MODULE] == (
        "glibc_flt32.cuh", "glibc_flt64.cuh",
        "thompson_aerosol_libm.cuh", AEROSOL_COMMON_HEADER)
    source = module_source(STATE_MODULE)
    assert "thompson_aa_cloud_dist" in source
    assert "thompson_aa_inu_c_effrad" in source
    # No mp=8 literal may survive in this translation unit's CODE.  The
    # frozen constants 100.0e6f (Nt_c), 65 (cloud_number_bin) and
    # 2730.0f / 272.0f (the gamma ratios) are exactly what mp=28 must derive
    # from nc; a stray one runs, stays stable, and is silently wrong.
    # Comments are stripped first because the header block deliberately
    # quotes those literals while explaining why they are forbidden here.
    text = (Path(__file__).parents[1] / "woof" / "core" / "kernels"
            / "thompson_aerosol_state.cu").read_text(encoding="utf-8")
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    for literal in ("100.0e6f", "2730.0f", "272.0f", "100.E6", "1000.E6"):
        assert literal not in code, literal
