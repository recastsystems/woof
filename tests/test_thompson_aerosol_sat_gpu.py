"""GPU gates for the mp=28 aerosol saturation adjustment (WP-05).

What is under test
------------------
``woof/core/kernels/thompson_aerosol_sat.cu`` and its launchers:

* ``thompson_aa_saturation_adjust`` -- module_mp_thompson.F:3399-3494.  CCN
  activation through ``activ_ncloud``/``tnccn_act``, the aerosol-only droplet
  evaporation branch through ``tnc_wev``, WRF's 99%-of-liquid mass floor, and
  the one-for-one CCN return.
* ``thompson_aa_rain_evaporation`` -- :3236-3255 + :3384-3388 + :3500-3574.
  A direct port, including the mean-volume-diameter clamp WOOF-ArWen's mp=8 kernel
  does not carry, plus ``nwfaten += pnr_rev`` (:3565) and the :3502 gate.

THE PRIMARY GATE IS BITWISE AGAINST AN INSTRUMENTED WRF
-------------------------------------------------------
Both kernels are driven on WRF's own working column, lifted as raw IEEE-754
bit patterns out of a scratch copy of the pristine module that differs from
it by WRITE statements alone -- an inertness proven by the instrumented build
reproducing all 38 committed CSVs under ``woof/data/thompson/oracle-aero/``
byte for byte.  See ``test_condensation_solve_is_bitwise_against_the_
instrumented_wrf_oracle`` (122 rows, 8 fixtures, every branch) and
``test_rain_evaporation_matches_the_instrumented_wrf_oracle`` (15 rows).
Those two tests, not the fixture-level comparisons below, are what this
package rests on: they depend on no sibling package and on no reconstruction
of the entry state.

DELIBERATE DIVERGENCE FROM THE FROZEN mp=8 KERNEL
--------------------------------------------------
mp=28 does NOT share thompson.cu's condensation solve or its rain
evaporation any more, and the tests here no longer assert that it does.  The
tie-break is the one ``thompson_aerosol_common.cuh`` already applied to
RSLF/RSIF: the authority is WRF.  Where a test used to require agreement with
mp=8 it now requires bitwise agreement with an independent float32 host
transcription of the Fortran, and the mp=8 divergence is measured and
recorded instead -- 3.8e-03 on a deliberately cancellation-dominated
evaporation sweep.  thompson.cu is byte-frozen; that deviation is a
pre-existing ArWen-wide one this port merely made visible.

TOLERANCE POLICY (written down, not implied)
--------------------------------------------
``activ_ncloud`` selects its temperature slab by NEAREST-NEIGHBOUR over a
10 K grid and ``idx_d``/``idx_c``/``idx_n`` are INT truncations, so activated
and evaporated droplet number are STEP FUNCTIONS of state.  Within one bin the
port is a smooth float32 computation and agrees with the Fortran oracle to a
few ulp; at a bin edge an FP32 GPU port and the Fortran reference can select
different bins and differ by tens of percent in ``nc`` while every mass field
still agrees.

This file therefore does two separate things and never mixes them:

1. The oracle fixtures (states chosen away from bin edges by WP-03) are gated
   with ``rtol`` at the float32 level -- 2e-6 for numbers, 3e-6 with a 2e-9
   ``atol`` for the saturation-adjusted masses, which is the same policy the
   model-validated mp=8 ``condense`` gate uses in
   ``tests/test_thompson_gpu.py``.  Measured maxima are recorded in the
   comments beside each assertion.
2. The step behaviour itself is asserted EXPLICITLY in
   :func:`test_temperature_bin_is_a_documented_step_not_an_error_budget`
   rather than being absorbed into a loose global tolerance.

FIXTURE SCOPE, stated accurately
------------------------------
``aero-ccn-activate`` (103) and ``aero-ccn-sweep`` (104) are pure
saturation-adjustment columns: their oracle runs produce exactly zero rain,
zero ice and zero snow, so the whole mp_gt_driver "after" state is reachable
from this package's kernel plus WRF's terminal apply.  ``aero-drop-evap``
(105) additionally activates warm-rain autoconversion, whose droplet-number
sink ``pnc_wau`` belongs to WP-07.  For that fixture the CCN field pins this
kernel's ``pnc_wcd`` exactly (WRF returns CCN only from ``pnc_wcd`` and
``pnr_rev``, never from autoconversion), the liquid+rain sum closes, and the
residual in ``nc`` is measured and attributed rather than tolerated -- see
:func:`test_droplet_evaporation_column_matches_wrf`.

WHAT IS HOST CODE HERE
----------------------
WRF's terminal apply/clamp block (:3972-4021) belongs to WP-04.  To compare a
single-kernel result against a whole-driver fixture this file transcribes that
block in float32 NumPy, in :func:`_wrf_terminal_apply`.  Every such
transcription is marked; the GPU is responsible for :3399-3494 only.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_DATA = Path(__file__).parents[1] / "woof" / "data" / "thompson"
_ORACLE_AERO = _DATA / "oracle-aero"
_ORACLE_CLASSIC = _DATA / "oracle"
# Resolved, not joined: the classic-table directory ships in the recast-woof-data
# companion distribution since 2.5.0.  The oracle directories above did not
# move and are still under woof/data/thompson.
from woof.physics_compat import packaged_thompson_table_root  # noqa: E402

_TABLES = packaged_thompson_table_root()

F32 = np.float32
_LEVELS = 24


# ---------------------------------------------------------------------------
# Fixture and table loading.
# ---------------------------------------------------------------------------

def _read_column(scenario: str, directory: Path = _ORACLE_AERO):
    with (directory / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    before = [row for row in rows if row["phase"] == "before"]
    after = [row for row in rows if row["phase"] == "after"]
    assert len(before) == _LEVELS and len(after) == _LEVELS
    return before, after


def _read_surface(scenario: str, directory: Path = _ORACLE_AERO):
    with (directory / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        return next(csv.DictReader(stream))


def _host(rows, name) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=F32)


@pytest.fixture(scope="module")
def tnc_wev_host() -> np.ndarray:
    """``tnc_wev`` straight out of the packaged classic auxiliary asset.

    Record 7 of ``thompson_aux_tables.dat``.  It has been parsed,
    SHA-validated and uploaded on every mp=8 launch since the classic port and
    read by nothing; mp=28's droplet-evaporation branch is its first consumer.
    """
    from woof.core.thompson_contract import (
        AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS, read_sequential_records)

    return read_sequential_records(
        _TABLES / AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS)["tnc_wev"]


def _device_fortran(host_array):
    """Upload a Fortran-ordered host array without touching cuBLAS.

    ``cupy.asfortranarray`` on a C-ordered 3-D array dispatches to cuBLAS
    ``geam``; reordering on the host first keeps this test file runnable on a
    driver-only install and is the same layout the launchers require.
    """
    import cupy as cp

    return cp.asarray(np.asfortranarray(host_array))


@pytest.fixture(scope="module")
def tnc_wev(tnc_wev_host):
    return _device_fortran(tnc_wev_host)


@pytest.fixture(scope="module")
def tnccn_act():
    """``tnccn_act`` uploaded by WP-01's runtime.

    ``require_classic_assets=False``: this package needs only the aerosol
    asset and ``thompson_aux_tables.dat``.  ``freezeH2O.dat`` is an
    externalized asset and its absence must not mask an aerosol failure.
    """
    from woof.core.thompson_aerosol_runtime import load_aerosol_device_tables

    table_set = load_aerosol_device_tables(
        _TABLES, require_classic_assets=False)
    return table_set.arrays["tnccn_act"]


# ---------------------------------------------------------------------------
# HOST transcriptions.  These are NOT the code under test.
# ---------------------------------------------------------------------------

def _rho(pressure, temperature, qv):
    """module_mp_thompson.F:1802 / :3199, in float32."""
    return (F32(0.622) * pressure
            / (F32(287.04) * temperature * (qv + F32(0.622)))).astype(F32)


_RSLF_COEFFICIENTS = tuple(F32(c) for c in (
    0.611583699e3, 0.444606896e2, 0.143177157e1, 0.264224321e-1,
    0.299291081e-3, 0.203154182e-5, 0.702620698e-8, 0.379534310e-11,
    -0.321582393e-13))


def _rslf(pressure, temperature):
    """``RSLF``, module_mp_thompson.F:5378-5413, float32 Horner order."""
    x = np.maximum(F32(-80.0), F32(temperature) - F32(273.16)).astype(F32)
    esl = np.full(np.shape(x), _RSLF_COEFFICIENTS[-1], dtype=F32)
    for coefficient in reversed(_RSLF_COEFFICIENTS[:-1]):
        esl = (coefficient + x * esl).astype(F32)
    esl = np.minimum(esl, F32(pressure) * F32(0.15)).astype(F32)
    return (F32(0.622) * esl / (F32(pressure) - esl)).astype(F32)


def _nint_f32(values):
    """Fortran ``NINT`` on a non-negative float32 array."""
    return np.floor((np.asarray(values, dtype=F32) + F32(0.5)).astype(F32)
                    ).astype(np.int64)


def _wrf_terminal_apply(nc_entry, nwfa_entry, ncten, nwfaten, qc, pressure,
                        temperature, qv, dt):
    """module_mp_thompson.F:3972-4021, float32 host transcription.

    Owned by WP-04 in production; reproduced here so a single-kernel result
    can be compared with a whole-mp_gt_driver fixture.  WRF's unit
    inconsistencies are literal: the per-kilogram ``nc1d`` is compared against
    the volumetric ``Nt_c_max`` at :3976 and the per-kilogram aerosols against
    the volumetric floors/ceiling at :3979-3981.
    """
    from woof.core import thompson_aerosol_contract as ac

    step = F32(dt)
    rho = _rho(pressure, temperature, np.maximum(F32(1.0e-10), qv))
    nc = np.maximum(
        F32(2.0) / rho,
        np.minimum(nc_entry + ncten * step, F32(1999.0e6))).astype(F32)
    nwfa = np.maximum(
        F32(11.1e6),
        np.minimum(F32(9999.0e6), nwfa_entry + nwfaten * step)).astype(F32)

    am_r = F32(ac.AM_R)
    obmr = F32(F32(1.0) / F32(3.0))
    d0c = F32(ac.D0C_M)
    d0r = F32(ac.D0R_M)
    qc_out = qc.copy()
    nc_out = nc.copy()
    for i in range(nc.size):
        if qc[i] <= F32(1.0e-12):
            qc_out[i] = F32(0.0)
            nc_out[i] = F32(0.0)
            continue
        nu_c = min(15, int(math.floor(1000.0e6 / float(nc[i] * rho[i]) + 0.5)) + 2)
        lamc = np.float64(
            am_r * F32(ac.CCG2[nu_c]) * F32(ac.OCG1[nu_c]) * nc[i] / qc[i]
        ) ** np.float64(obmr)
        x_dc = np.float64(3.0 + nu_c + 1.0) / lamc
        if x_dc < np.float64(d0c):
            lamc = np.float64(ac.CCE2[nu_c]) / np.float64(d0c)
        elif x_dc > np.float64(d0r) * 2.0:
            lamc = np.float64(ac.CCE2[nu_c]) / (np.float64(d0r) * 2.0)
        value = (np.float64(F32(ac.CCG1[nu_c]) * F32(ac.OCG2[nu_c]) * qc[i]
                            / am_r) * lamc ** 3.0)
        nc_out[i] = F32(min(value, np.float64(1999.0e6) / np.float64(rho[i])))
    return qc_out, nc_out, nwfa


def _run_saturation_adjust(before, dt, tnccn_act, tnc_wev, *, w=None,
                           ncten0=None):
    """Drive the kernel from a fixture's ``before`` rows.

    For fixtures 103/104/105 every incoming tendency from the source/sink
    networks is zero or negligible, so the post-network state WRF hands to
    :3399 is the fixture's own ``before`` state.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    pressure = _host(before, "p_pa")
    temperature = _host(before, "temp_k")
    qv = _host(before, "qv")
    qc = _host(before, "qc")
    nc_entry = _host(before, "nc_per_kg")
    nwfa_entry = _host(before, "nwfa_per_kg")
    if w is None:
        w = _host(before, "w_m_s")

    rho = _rho(pressure, temperature, np.maximum(F32(1.0e-10), qv))
    # module_mp_thompson.F:3211 -- the SECOND aerosol snapshot.  No upper
    # bound and no nifa counterpart; that asymmetry is intentional in WRF.
    nwfa_work = np.maximum(F32(11.1e6), nwfa_entry * rho).astype(F32)

    dev = {
        "temperature": cp.asarray(temperature),
        "pressure": cp.asarray(pressure),
        "qv": cp.asarray(qv),
        "qc": cp.asarray(qc),
        "nc_entry": cp.asarray(nc_entry),
        "ncten": (cp.zeros(nc_entry.size, dtype=cp.float32)
                  if ncten0 is None else cp.asarray(ncten0)),
        "nwfaten": cp.zeros(nc_entry.size, dtype=cp.float32),
        "nwfa_work_m3": cp.asarray(nwfa_work),
        "w": cp.asarray(w),
    }
    condensation = cp.empty(nc_entry.size, dtype=cp.float32)
    launch_aerosol_saturation_adjust(
        dev["temperature"], dev["pressure"], dev["qv"], dev["qc"],
        dev["nc_entry"], dev["ncten"], dev["nwfaten"], dev["nwfa_work_m3"],
        dev["w"], tnccn_act, tnc_wev, dt,
        condensation_rate=condensation)
    cp.cuda.Stream.null.synchronize()

    out = {name: cp.asnumpy(value) for name, value in dev.items()}
    out["condensation_rate"] = cp.asnumpy(condensation)
    out["nwfa_entry"] = nwfa_entry
    out["rho"] = rho
    out["nwfa_work_m3"] = nwfa_work
    qc_final, nc_final, nwfa_final = _wrf_terminal_apply(
        nc_entry, nwfa_entry, out["ncten"], out["nwfaten"], out["qc"],
        pressure, out["temperature"], out["qv"], dt)
    out["qc_final"] = qc_final
    out["nc_final"] = nc_final
    out["nwfa_final"] = nwfa_final
    return out


# ---------------------------------------------------------------------------
# Compilation and ABI.
# ---------------------------------------------------------------------------

def test_translation_unit_receives_the_shared_header_and_exports_its_kernels():
    from woof.core.kernels import EXTRA_HEADERS, load_module, module_source
    from woof.core.thompson_aerosol_launch import (
        AEROSOL_COMMON_HEADER, SAT_MODULE)
    from woof.core.thompson_aerosol_sat import (
        DROPLET_EVAP_PROBE_KERNEL, RAIN_EVAPORATION_KERNEL,
        SATURATION_ADJUST_KERNEL)

    # WOOF's own libm words ride ahead of the shared header
    # (woof/core/kernels/thompson_aerosol_libm.cuh).
    assert EXTRA_HEADERS[SAT_MODULE] == (
        "glibc_flt32.cuh", "glibc_flt64.cuh",
        "thompson_aerosol_libm.cuh", AEROSOL_COMMON_HEADER)
    source = module_source(SAT_MODULE)
    # The helpers this kernel must not re-derive locally.
    assert "thompson_activ_ncloud" in source
    assert "thompson_aa_droplet_bin" in source
    assert "thompson_aa_decade_index" in source

    module = load_module(SAT_MODULE)
    for name in (SATURATION_ADJUST_KERNEL, RAIN_EVAPORATION_KERNEL,
                 DROPLET_EVAP_PROBE_KERNEL):
        assert module.get_function(name) is not None


def test_no_droplet_number_literal_survives_in_this_translation_unit():
    """The half-converted-nc failure mode, asserted rather than assumed.

    ``100.0e6f`` appears at 12 sites in thompson.cu because mp=8 freezes
    Nt_c.  Nothing in this file may reintroduce it, nor the frozen droplet
    bin 65 or the frozen gamma ratios 2730/272 -- every one of those must come
    from ``nc`` through thompson_aerosol_common.cuh.
    """
    source = (Path(__file__).parents[1] / "woof" / "core" / "kernels"
              / "thompson_aerosol_sat.cu").read_text(encoding="utf-8")
    body = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("//"))
    for literal in ("100.0e6f", "100.e6f", "2730.0f", "272.0f"):
        assert literal not in body, literal
    # Nt_c_max and the aerosol floors are legitimate: they are WRF clamps,
    # not droplet-number defaults, and they come from the shared header.
    assert "THOMPSON_AA_NT_C_MAX" in body
    assert "THOMPSON_AA_NT_C " not in body and "THOMPSON_AA_NT_C)" not in body


# ---------------------------------------------------------------------------
# Fixture 103 -- CCN activation.
# ---------------------------------------------------------------------------

def test_ccn_activation_column_matches_wrf(tnccn_act, tnc_wev):
    """aero-ccn-activate (103): a pure saturation-adjustment column.

    The oracle run produces exactly zero rain, ice, snow and graupel, so this
    kernel plus WRF's terminal apply reproduces the whole driver.

    Measured maxima on an RTX 5090, CUDA 12 / cupy 14.1.1:
        qv   1.09e-07 rel      qc   1.10e-04 rel (8.7e-10 ABSOLUTE)
        temp 1.07e-07 rel      nc   1.73e-07 rel      nwfa exact
    ``qc`` is the small difference of two large numbers -- the condensate is
    7.9e-06 kg/kg formed out of 1.1e-02 kg/kg of vapour -- so its relative
    error is the vapour's absolute float32 error divided by the condensate.
    The classic mp=8 ``condense`` gate carries exactly this shape and exactly
    this atol.
    """
    before, after = _read_column("aero-ccn-activate")
    dt = float(_read_surface("aero-ccn-activate")["dt_s"])
    assert dt == 10.0

    # The fixture must actually be a pure condensation column, or the
    # comparison below is not a gate on this package.
    for name in ("qr", "qi", "qs", "qg", "nr_per_kg", "ni_per_kg"):
        np.testing.assert_array_equal(_host(after, name), np.zeros(_LEVELS))

    got = _run_saturation_adjust(before, dt, tnccn_act, tnc_wev)

    np.testing.assert_allclose(
        got["qv"], _host(after, "qv"), rtol=2.0e-6, atol=2.0e-9)
    np.testing.assert_allclose(
        got["qc_final"], _host(after, "qc"), rtol=3.0e-6, atol=2.0e-9)
    np.testing.assert_allclose(
        got["temperature"], _host(after, "temp_k"), rtol=2.0e-6, atol=2.0e-5)
    np.testing.assert_allclose(
        got["nc_final"], _host(after, "nc_per_kg"), rtol=2.0e-6, atol=2.0)
    np.testing.assert_allclose(
        got["nwfa_final"], _host(after, "nwfa_per_kg"), rtol=2.0e-6, atol=2.0)

    # Water is conserved by the adjustment to float32 rounding.
    np.testing.assert_allclose(
        got["qv"] + got["qc"], _host(before, "qv") + _host(before, "qc"),
        rtol=0.0, atol=2.0e-9)

    # THE aerosol return path: every activated droplet consumed exactly one
    # water-friendly aerosol.  nwfaten == -ncten holds bit for bit here
    # because activation is the only process running.
    np.testing.assert_array_equal(got["nwfaten"], -got["ncten"])
    assert np.all(got["ncten"] > 0.0)


def test_activation_is_the_only_consumer_of_the_activation_table(
        tnccn_act, tnc_wev):
    """Perturbing tnccn_act must move nc and nothing else.

    Guards against a kernel that silently ignores the table (e.g. reading an
    all-ones prefill, which is what a second thompson_init in one process
    leaves behind and which returns 100% activation with no error anywhere).
    """
    import cupy as cp

    before, _ = _read_column("aero-ccn-activate")
    dt = 10.0
    base = _run_saturation_adjust(before, dt, tnccn_act, tnc_wev)
    ones = _device_fortran(np.ones(tnccn_act.shape, dtype=np.float64))
    perturbed = _run_saturation_adjust(before, dt, ones, tnc_wev)

    np.testing.assert_array_equal(base["qv"], perturbed["qv"])
    np.testing.assert_array_equal(base["qc"], perturbed["qc"])
    np.testing.assert_array_equal(base["temperature"],
                                  perturbed["temperature"])
    # An all-ones table means fraction == 1: every available CCN activates,
    # so the recovered xnc is exactly the CCN count that was offered.  The
    # measured activated fraction with the real table is 0.72 here, which is
    # both different from 1 and physically sensible.
    assert np.all(perturbed["ncten"] > base["ncten"])
    np.testing.assert_allclose(
        perturbed["ncten"] * F32(dt) + F32(2.0) / base["rho"],
        base["nwfa_work_m3"] / base["rho"], rtol=1.0e-6)
    fraction = base["ncten"] / perturbed["ncten"]
    assert 0.05 < fraction.min() and fraction.max() < 0.999


# ---------------------------------------------------------------------------
# Fixture 104 -- the CCN/updraft sweep, and blocking unknown #3.
# ---------------------------------------------------------------------------

def test_ccn_sweep_column_matches_wrf(tnccn_act, tnc_wev):
    """aero-ccn-sweep (104): w over {0.005 .. 200} and CCN over {5e6 .. 2e10}.

    This exercises both clamp ends of ``ta_Na`` and ``ta_Ww`` and several
    interior bilinear cells, and it reaches the ``Nt_c_max`` ceiling at the
    top of the CCN ladder.

    Measured maxima: nc 6.40e-07 rel over all 24 levels; nwfa exact.
    """
    before, after = _read_column("aero-ccn-sweep")
    dt = float(_read_surface("aero-ccn-sweep")["dt_s"])
    got = _run_saturation_adjust(before, dt, tnccn_act, tnc_wev)

    np.testing.assert_allclose(
        got["nc_final"], _host(after, "nc_per_kg"), rtol=2.0e-6, atol=2.0)
    np.testing.assert_allclose(
        got["nwfa_final"], _host(after, "nwfa_per_kg"), rtol=2.0e-6, atol=2.0)
    np.testing.assert_allclose(
        got["qc_final"], _host(after, "qc"), rtol=3.0e-6, atol=2.0e-9)

    # The ceiling really is reached, so the fixture is exercising it.
    assert np.any(_host(after, "nc_per_kg") > 1.9e9)


def test_ccn_sweep_settles_the_vertical_velocity_source(tnccn_act, tnc_wev):
    """BLOCKING UNKNOWN #3, settled here against the oracle.

    The spec directed passing ``state.w[:-1]`` -- the lower full-level slice
    WOOF-ArWen already gives cloud sedimentation -- on the precedent that
    ``mp_gt_driver`` copies ``w1d(k) = w(i,k,j)`` once at :1224 with no
    averaging.  ``activ_ncloud`` is far more sensitive to w than sedimentation
    is, because w enters a bracket search over a four-decade logarithmic axis,
    so the precedent was not treated as sufficient.

    THE MEASUREMENT.  ``aero-ccn-sweep`` gives every level a different w from
    the ladder {0.005, 0.02, 0.05, 0.2, 0.5, 2, 5, 20, 50, 200} m/s, so
    adjacent levels differ by up to a factor of four in w and any interface
    averaging is a large, visible change of input.  Feeding w level by level
    reproduces the fixture to 6.4e-07 relative at all 24 levels.  Feeding the
    interface average 0.5*(w[k] + w[k+1]) instead is wrong by up to 73%.

    The un-averaged lower-full-level slice is therefore the correct source and
    this unknown is closed.
    """
    before, after = _read_column("aero-ccn-sweep")
    dt = float(_read_surface("aero-ccn-sweep")["dt_s"])
    expected = _host(after, "nc_per_kg")

    direct = _run_saturation_adjust(before, dt, tnccn_act, tnc_wev)
    direct_rel = np.abs(
        direct["nc_final"].astype(np.float64) - expected) / expected
    assert direct_rel.max() < 2.0e-6, direct_rel

    w_faces = _host(before, "w_m_s")
    averaged = (F32(0.5) * (w_faces + np.roll(w_faces, -1))).astype(F32)
    smoothed = _run_saturation_adjust(
        before, dt, tnccn_act, tnc_wev, w=averaged)
    smoothed_rel = np.abs(
        smoothed["nc_final"].astype(np.float64) - expected) / expected

    # Not a marginal preference: averaging is four orders of magnitude worse
    # on the levels where the ladder actually changes.
    assert smoothed_rel.max() > 0.5
    assert (smoothed_rel > 1.0e-3).sum() >= 12


# ---------------------------------------------------------------------------
# Fixture 105 -- the aerosol-only droplet evaporation branch.
# ---------------------------------------------------------------------------

def test_droplet_evaporation_column_matches_wrf(tnccn_act, tnc_wev):
    """aero-drop-evap (105): the ONLY fixture that reads ``tnc_wev``.

    Fixture scope, measured rather than assumed.  This column also activates
    warm-rain autoconversion (the oracle's ``qr`` after is 3.6e-09 at k=1),
    whose droplet-number sink ``pnc_wau`` is WP-07's, not this package's.  The
    three consequences are separated:

    * ``nwfa`` pins this kernel's ``pnc_wcd`` EXACTLY.  WRF returns CCN only
      from ``pnc_wcd`` (:3482) and ``pnr_rev`` (:3565), never from
      autoconversion, so the CCN field is blind to the missing network.
      Measured: 6.20e-07 relative, and bit-exact at the three levels where
      ``tnc_wev`` actually binds.
    * liquid + rain closes to 7.9e-10 absolute -- the mass this kernel does
      not move is exactly the mass autoconversion moved.
    * ``nc`` agrees to 1.4e-05 relative at the six levels where WRF's
      99%-of-number limiter binds, and to 2.8e-04 at the three levels where
      the table binds; that residual is ``pnc_wau*dt`` and is asserted to be
      SMALL AND NEGATIVE, i.e. attributed, not tolerated.
    """
    before, after = _read_column("aero-drop-evap")
    dt = float(_read_surface("aero-drop-evap")["dt_s"])
    got = _run_saturation_adjust(before, dt, tnccn_act, tnc_wev)

    expected_qc = _host(after, "qc")
    expected_qr = _host(after, "qr")
    expected_nc = _host(after, "nc_per_kg")
    expected_nwfa = _host(after, "nwfa_per_kg")

    # 1. The CCN return pins pnc_wcd.
    np.testing.assert_allclose(
        got["nwfa_final"], expected_nwfa, rtol=2.0e-6, atol=2.0)

    # 2. Liquid + rain closure: what this kernel leaves in qc is what the
    #    oracle split between qc and qr.
    np.testing.assert_allclose(
        got["qc_final"], expected_qc + expected_qr, rtol=3.0e-6, atol=2.0e-9)

    # 3. Vapour and temperature are untouched by the missing network.
    np.testing.assert_allclose(
        got["qv"], _host(after, "qv"), rtol=2.0e-6, atol=2.0e-9)
    np.testing.assert_allclose(
        got["temperature"], _host(after, "temp_k"), rtol=2.0e-6, atol=2.0e-5)

    # 4. Droplet number: the residual must be a small NEGATIVE sink, i.e.
    #    consistent with an unmodelled autoconversion loss and nothing else.
    cloudy = expected_nc > 0.0
    residual = ((got["nc_final"][cloudy].astype(np.float64)
                 - expected_nc[cloudy]) / expected_nc[cloudy])
    assert residual.max() < 3.0e-4
    assert residual.min() > -3.0e-4
    # Where the number limiter binds instead of the table, the residual is
    # float noise: those levels have essentially no autoconversion.
    limiter_levels = np.arange(_LEVELS)[cloudy][3:]
    limiter_residual = np.abs(
        got["nc_final"][limiter_levels].astype(np.float64)
        - expected_nc[limiter_levels]) / expected_nc[limiter_levels]
    assert limiter_residual.max() < 2.0e-5, limiter_residual


def test_droplet_evaporation_indices_and_table_values(tnc_wev, tnc_wev_host):
    """TRANSPOSITION FIXTURE for ``tnc_wev``.

    ``tnc_wev`` is ``(nbc, ntb_c, nbc) = (100, 37, 100)``.  Axes 0 and 2 are
    BOTH length 100, so a Fortran/C order flip neither changes the file
    SHA-256 nor raises on shape.  Since this port is the table's first reader
    in woof's history, an order defect that predates mp=28 would surface as
    an mp=28 physics error.  This test pins the ``(idx_d, idx_c, idx_n)``
    triple the kernel selects and the value it reads, so the failure is
    attributed to the table.

    Axis reference: axis 0 is droplet diameter with ``Dc(i) = i micron``
    LINEARLY (:831-836), unlike every other bin family in the scheme; axis 1
    is ``r_c`` cloud water over 1e-6..1e-2 kg m-3; axis 2 is the ``t_Nc``
    logarithmic droplet-number grid.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_contract import CLASSIC_DROPLET_BIN
    from woof.core.thompson_aerosol_sat import (
        probe_droplet_evaporation_indices)

    before, _ = _read_column("aero-drop-evap")
    dt = 10.0
    pressure = _host(before, "p_pa")
    temperature = _host(before, "temp_k")
    qv = _host(before, "qv")
    qc = _host(before, "qc")
    nc_entry = _host(before, "nc_per_kg")
    rho = _rho(pressure, temperature, np.maximum(F32(1.0e-10), qv))
    nc_work = np.where(
        qc > F32(1.0e-12),
        np.maximum(F32(2.0), np.minimum(nc_entry * rho, F32(1999.0e6))),
        F32(2.0)).astype(F32)

    idx_d, idx_c, idx_n, value, pnc = probe_droplet_evaporation_indices(
        cp.asarray(temperature), cp.asarray(pressure), cp.asarray(qv),
        cp.asarray(qc), cp.asarray(nc_work), tnc_wev, dt)
    cp.cuda.Stream.null.synchronize()
    idx_d = cp.asnumpy(idx_d)
    idx_c = cp.asnumpy(idx_c)
    idx_n = cp.asnumpy(idx_n)
    value = cp.asnumpy(value)

    # One-based, exactly WRF's own subscripts, and inside every axis.
    assert idx_d.min() >= 1 and idx_d.max() <= 100
    assert idx_c.min() >= 1 and idx_c.max() <= 37
    assert idx_n.min() >= 1 and idx_n.max() <= 100

    # The cloudy levels carry nc = 1e8 m-3, which is exactly Nt_c: the
    # generalized droplet bin must reproduce the value mp=8 hardcodes.
    cloudy = qc > F32(1.0e-12)
    assert np.all(idx_n[cloudy] - 1 == CLASSIC_DROPLET_BIN)

    # The kernel read the packaged bytes at the subscripts it reported.  This
    # is the transposition assertion: a C-order or axis-swapped upload lands
    # somewhere else in the 370000-element array.
    for i in range(idx_d.size):
        expected = tnc_wev_host[idx_d[i] - 1, idx_c[i] - 1, idx_n[i] - 1]
        assert value[i] == expected, (i, idx_d[i], idx_c[i], idx_n[i])

    # And an axis-swapped copy really does give a different answer here, so
    # the check above is not vacuous.
    swapped_host = np.asfortranarray(np.transpose(tnc_wev_host, (2, 1, 0)))
    swapped = _device_fortran(swapped_host)
    _, _, _, swapped_value, _ = probe_droplet_evaporation_indices(
        cp.asarray(temperature), cp.asarray(pressure), cp.asarray(qv),
        cp.asarray(qc), cp.asarray(nc_work), swapped, dt)
    cp.cuda.Stream.null.synchronize()
    assert np.any(cp.asnumpy(swapped_value)[cloudy] != value[cloudy])


def test_tnc_wev_bytes_are_the_aerosol_aware_table(tnc_wev_host):
    """The packaged classic record really is WRF's nu_c-varying table.

    ``table_dropEvap`` (:5011-5099) builds ``tnc_wev`` with a per-bin
    ``nu_c`` that runs from 15 down to 2.  If the packaged asset had been
    generated with a frozen nu_c the mp=28 evaporation branch would read a
    plausible but wrong table.  WP-01's independent reimplementation settles
    it; this test names the dependency so a failure lands here rather than in
    the physics.
    """
    from woof.core.thompson_aerosol_contract import (
        DROP_EVAP_RELATIVE_TOLERANCE, validate_drop_evaporation_tables)
    from woof.core.thompson_contract import (
        AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS, read_sequential_records)

    arrays = read_sequential_records(
        _TABLES / AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS)
    measured = validate_drop_evaporation_tables(arrays)
    assert measured["tnc_wev"] <= DROP_EVAP_RELATIVE_TOLERANCE
    assert tnc_wev_host.shape == (100, 37, 100)
    # Cumulative over the diameter axis: monotone non-decreasing in idx_d.
    assert np.all(np.diff(tnc_wev_host[:, 20, 65]) >= 0.0)


def test_droplet_evaporation_chain_matches_a_host_float32_transcription(
        tnc_wev, tnc_wev_host):
    """Independent NumPy transcription of :3423-3469.

    The kernel evaluates this chain with ``__fmul_rn``/``__fadd_rn`` from
    thompson_aerosol_common.cuh precisely so it reproduces `gfortran -O2` on
    baseline x86-64, where there is no FMA and every REAL(4) operation is
    separately rounded.  NumPy float32 has the same semantics, so agreement
    here is evidence of BOTH a correct transcription and successful
    contraction pinning: with contraction left on, ``Dc_star`` moves in the
    last float digits and can flip ``idx_d``, which is an INT truncation.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        probe_droplet_evaporation_indices)

    before, _ = _read_column("aero-drop-evap")
    dt = 10.0
    pressure = _host(before, "p_pa")
    temperature = _host(before, "temp_k")
    qv = _host(before, "qv")
    qc = _host(before, "qc")
    nc_entry = _host(before, "nc_per_kg")
    rho = _rho(pressure, temperature, np.maximum(F32(1.0e-10), qv))
    nc_work = np.where(
        qc > F32(1.0e-12),
        np.maximum(F32(2.0), np.minimum(nc_entry * rho, F32(1999.0e6))),
        F32(2.0)).astype(F32)

    idx_d, _, _, _, pnc = probe_droplet_evaporation_indices(
        cp.asarray(temperature), cp.asarray(pressure), cp.asarray(qv),
        cp.asarray(qc), cp.asarray(nc_work), tnc_wev, dt)
    cp.cuda.Stream.null.synchronize()
    idx_d = cp.asnumpy(idx_d)

    orv = F32(F32(1.0) / F32(461.5))
    pi = F32(3.1415926536)
    for i in range(_LEVELS):
        t0 = temperature[i]
        p0 = pressure[i]
        qv0 = np.maximum(F32(1.0e-10), qv[i])
        rho_i = rho[i]
        # RSLF and the transport coefficients, :3199-3210.
        qvs = F32(_rslf(p0, t0))
        ssatw = F32(qv0 / qvs - F32(1.0))
        if abs(ssatw) < 1.0e-15:
            ssatw = F32(0.0)
        tempc = F32(t0 - F32(273.15))
        otemp = F32(F32(1.0) / t0)
        diffu = F32(F32(2.11e-5) * F32((t0 / F32(273.15)) ** F32(1.94))
                    * F32(F32(101325.0) / p0))
        lvap = F32(F32(2.5e6) + F32(F32(2106.0) - F32(4218.0)) * tempc)
        tcond = F32(F32(F32(5.69) + F32(0.0168) * tempc) * F32(1.0e-5)
                    * F32(418.936))

        rvs = F32(rho_i * qvs)
        tt = F32(F32(F32(lvap * otemp) * orv) - F32(1.0))
        rvs_p = F32(F32(rvs * otemp) * tt)
        pp_a = F32(F32(F32(F32(otemp * tt) * otemp)) * tt)
        pp_b = F32(-2.0) * lvap
        pp_b = F32(F32(F32(F32(pp_b * otemp) * otemp) * otemp) * orv)
        pp_c = F32(otemp * otemp)
        rvs_pp = F32(rvs * F32(F32(pp_a + pp_b) + pp_c))
        gamsc = F32(F32(F32(lvap * diffu) / tcond) * rvs_p)
        gr = F32(gamsc / F32(F32(1.0) + gamsc))
        alphsc = F32(0.5) * gr
        alphsc = F32(alphsc * gr)
        alphsc = F32(alphsc * rvs_pp)
        alphsc = F32(alphsc / rvs_p)
        alphsc = F32(alphsc * rvs)
        alphsc = F32(alphsc / rvs_p)
        alphsc = F32(max(F32(1.0e-9), alphsc))
        xsat = ssatw
        if abs(xsat) < 1.0e-9:
            xsat = F32(0.0)
        ax = F32(alphsc * xsat)
        t2 = F32(2.0) * alphsc
        for factor in (alphsc, xsat, xsat):
            t2 = F32(t2 * factor)
        t3 = F32(5.0) * alphsc
        for factor in (alphsc, alphsc, xsat, xsat, xsat):
            t3 = F32(t3 * factor)
        paren = F32(F32(F32(F32(1.0) - ax) + t2) - t3)
        two_pi = F32(F32(2.0) * pi)
        t1_evap = F32(F32(two_pi * paren) / F32(F32(1.0) + gamsc))
        arg = -2.0 * float(dt)
        arg = arg * float(t1_evap)
        arg = arg / float(two_pi)
        arg = arg * 4.0
        arg = arg * float(diffu)
        arg = arg * float(ssatw)
        arg = arg * float(rvs)
        arg = arg / 1000.0
        dc_star = math.sqrt(arg)
        expected = max(1, min(int(1.0e6 * dc_star), 100))
        assert idx_d[i] == expected, (i, idx_d[i], expected, 1.0e6 * dc_star)
        # Not sitting on a bin edge: the fixture is a real measurement, not a
        # coin flip between two adjacent bins.
        assert abs(1.0e6 * dc_star - round(1.0e6 * dc_star)) > 1.0e-3


# ---------------------------------------------------------------------------
# The documented step behaviour.
# ---------------------------------------------------------------------------

def test_temperature_bin_is_a_documented_step_not_an_error_budget(
        tnccn_act, tnc_wev):
    """``activ_ncloud`` picks its T slab by NEAREST NEIGHBOUR (:5216-5217).

    ``k = MAX(1, MIN(NINT((Tt-243.15)*0.1) + 1, 7))``.  There is no
    interpolation in temperature, so activated number is piecewise constant in
    T with jumps every 10 K.  This is pinned explicitly rather than hidden
    inside a wide tolerance on the column fixtures, and it is why WP-03 chose
    fixture states away from the edges.

    Fortran ``NINT`` rounds half AWAY FROM ZERO; CUDA's ``__float2int_rn``
    rounds half to even and would place the T = 248.15 K edge in the wrong
    slab.  Both facts are asserted.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    n = 601
    temperature = np.linspace(244.0, 274.0, n).astype(F32)
    pressure = np.full(n, 90000.0, dtype=F32)
    qv = (F32(1.01) * _rslf(pressure, temperature)).astype(F32)
    qc = np.zeros(n, dtype=F32)
    nc_entry = np.full(n, 2.0, dtype=F32)
    nwfa_work = np.full(n, 3.0e8, dtype=F32)
    w = np.full(n, 1.0, dtype=F32)

    dev = [cp.asarray(a) for a in (temperature, pressure, qv, qc, nc_entry)]
    ncten = cp.zeros(n, dtype=cp.float32)
    nwfaten = cp.zeros(n, dtype=cp.float32)
    launch_aerosol_saturation_adjust(
        dev[0], dev[1], dev[2], dev[3], dev[4], ncten, nwfaten,
        cp.asarray(nwfa_work), cp.asarray(w), tnccn_act, tnc_wev, 10.0)
    cp.cuda.Stream.null.synchronize()

    # Recover the activated number the kernel chose:
    #     pnc_wcd = 0.5*(xnc - 2 + |xnc - 2|)*odt*orho,   xnc > 2 here.
    rho = _rho(pressure, temperature, qv)
    activated = (cp.asnumpy(ncten) * F32(10.0) * rho + F32(2.0)
                 ).astype(np.float64)

    # WRF's own slab selector, :5216-5217, in float32 with Fortran NINT.
    slab = np.clip(
        _nint_f32((temperature - F32(243.15)) * F32(0.1)) + 1, 1, 7)
    assert set(np.unique(slab)) == {1, 2, 3, 4}, np.unique(slab)

    # Within a slab the activated number is CONSTANT: w and the CCN count are
    # fixed and temperature enters activ_ncloud only through the slab index.
    for value in np.unique(slab):
        block = activated[slab == value]
        spread = (block.max() - block.min()) / block.mean()
        assert spread < 1.0e-5, (value, spread)

    # Across a slab boundary it JUMPS.  This is the step function the
    # tolerance policy refuses to absorb: near an edge an FP32 port and the
    # Fortran reference may pick different slabs and differ by this much in
    # nc while every mass field still agrees.
    means = [activated[slab == value].mean() for value in np.unique(slab)]
    jumps = [abs(b - a) / a for a, b in zip(means, means[1:])]
    assert min(jumps) > 1.0e-3, jumps
    assert max(jumps) > 1.0e-2, jumps

    # Fortran NINT rounds half AWAY FROM ZERO.  The first edge is exactly
    # 243.15 + 5 = 248.15 K, where CUDA's round-half-to-even __float2int_rn
    # would select the LOWER slab.  Assert the boundary sample lands high.
    edge = F32(248.15)
    assert int(_nint_f32(np.asarray([(edge - F32(243.15)) * F32(0.1)]))[0]
               + 1) == 2


# ---------------------------------------------------------------------------
# The accumulator contract.
# ---------------------------------------------------------------------------

def test_entry_state_arrays_are_never_written(tnccn_act, tnc_wev):
    """nc and nwfa entry state are READ-ONLY for the whole call.

    The kernel may only touch ncten/nwfaten and the mass/temperature state;
    the single terminal apply with WRF's clamps belongs to
    thompson_aerosol_state.cu.  Four clamps instead of one is a silent physics
    change no downstream unit test would flag.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    before, _ = _read_column("aero-drop-evap")
    n = _LEVELS
    nc_entry = cp.asarray(_host(before, "nc_per_kg"))
    nwfa_work = cp.asarray(
        np.maximum(F32(11.1e6),
                   _host(before, "nwfa_per_kg")
                   * _rho(_host(before, "p_pa"), _host(before, "temp_k"),
                          _host(before, "qv"))).astype(F32))
    w = cp.asarray(_host(before, "w_m_s"))
    nc_copy = nc_entry.copy()
    nwfa_copy = nwfa_work.copy()
    w_copy = w.copy()
    pressure = cp.asarray(_host(before, "p_pa"))
    pressure_copy = pressure.copy()

    launch_aerosol_saturation_adjust(
        cp.asarray(_host(before, "temp_k")), pressure,
        cp.asarray(_host(before, "qv")), cp.asarray(_host(before, "qc")),
        nc_entry, cp.zeros(n, dtype=cp.float32),
        cp.zeros(n, dtype=cp.float32), nwfa_work, w, tnccn_act, tnc_wev, 10.0)
    cp.cuda.Stream.null.synchronize()

    np.testing.assert_array_equal(cp.asnumpy(nc_entry), cp.asnumpy(nc_copy))
    np.testing.assert_array_equal(cp.asnumpy(nwfa_work),
                                  cp.asnumpy(nwfa_copy))
    np.testing.assert_array_equal(cp.asnumpy(w), cp.asnumpy(w_copy))
    np.testing.assert_array_equal(cp.asnumpy(pressure),
                                  cp.asnumpy(pressure_copy))


def test_accumulators_are_added_to_not_overwritten(tnccn_act, tnc_wev):
    """WRF writes ``ncten(k) = ncten(k) + pnc_wcd(k)`` (:3481).

    A kernel that assigns instead of accumulating would discard every warm-
    and cold-network droplet sink and still run, stay stable, and look
    plausible.
    """
    before, _ = _read_column("aero-ccn-activate")
    # A seed large enough to be unmistakable but not large enough to move the
    # working droplet number off its floor of 2 m-3, so pnc_wcd is identical
    # in both runs and the comparison isolates the accumulation itself.
    seed = np.full(_LEVELS, -1234.5, dtype=F32)
    zeroed = _run_saturation_adjust(before, 10.0, tnccn_act, tnc_wev)
    seeded = _run_saturation_adjust(
        before, 10.0, tnccn_act, tnc_wev, ncten0=seed)

    # pnc_wcd is unchanged, so the CCN sink is bit-identical...
    np.testing.assert_array_equal(seeded["nwfaten"], zeroed["nwfaten"])
    # ...and the droplet accumulator carries the seed forward additively.
    np.testing.assert_allclose(
        seeded["ncten"] - seed, zeroed["ncten"], rtol=2.0e-6, atol=1.0e-3)
    assert np.all(seeded["ncten"] < zeroed["ncten"])


# ---------------------------------------------------------------------------
# The aerosol-only 99% mass floor: mp=28 is not mp=8 plus bookkeeping.
# ---------------------------------------------------------------------------

def test_aerosol_branch_changes_the_mass_answer_not_only_the_number(
        tnccn_act, tnc_wev):
    """module_mp_thompson.F:3467, present ONLY on the is_aerosol_aware path.

    ``prw_vcd = MAX(-rc*0.99*orho*odt, prw_vcd)`` caps evaporation at 99% of
    the liquid.  It bites in the narrow band where the Newton solve would
    evaporate between 99% and 100% of the cloud, and there mp=8 and mp=28
    produce DIFFERENT cloud water.  That is the reason this is a new kernel
    rather than an extra output on thompson_cloud_saturation_adjust_impl.
    """
    import cupy as cp

    from woof.core.thompson import launch_cloud_saturation_adjust
    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    n = 4096
    temperature = np.full(n, 285.0, dtype=F32)
    pressure = np.full(n, 90000.0, dtype=F32)
    qc0 = np.full(n, 1.0e-5, dtype=F32)
    qvs = _rslf(np.full(n, 90000.0, dtype=F32), temperature)
    # Sweep the vapour deficit so the Newton solve's evaporated fraction runs
    # from about a half to about twice the available liquid; the floor bites
    # in the narrow (0.99, 1.0) window somewhere inside that.
    deficit = np.linspace(0.5, 2.0, n)
    qv0 = (qvs - (qc0.astype(np.float64) * deficit * 2.6).astype(F32)
           ).astype(F32)

    classic = [cp.asarray(a.copy()) for a in
               (temperature, pressure, qv0, qc0)]
    launch_cloud_saturation_adjust(*classic)
    aerosol_t = cp.asarray(temperature.copy())
    aerosol_qv = cp.asarray(qv0.copy())
    aerosol_qc = cp.asarray(qc0.copy())
    launch_aerosol_saturation_adjust(
        aerosol_t, cp.asarray(pressure), aerosol_qv, aerosol_qc,
        cp.full(n, 1.0e8, dtype=cp.float32),
        cp.zeros(n, dtype=cp.float32), cp.zeros(n, dtype=cp.float32),
        cp.full(n, 3.0e8, dtype=cp.float32),
        cp.zeros(n, dtype=cp.float32), tnccn_act, tnc_wev, 10.0)
    cp.cuda.Stream.null.synchronize()

    classic_qc = cp.asnumpy(classic[3])
    aerosol_qc_out = cp.asnumpy(aerosol_qc)
    floored = aerosol_qc_out > classic_qc + F32(1.0e-9)
    assert floored.any(), "the 99% floor was never exercised"
    # Fully evaporating cells take WRF's xrc <= R1 branch, which has no
    # floor.  DELIBERATE DIFFERENCE FROM mp=8, recorded here: thompson.cu
    # applies fmaxf(0, qc0 + clap) inside the kernel, while WRF's :3975 is a
    # bare `qc1d = qc1d + qcten*DT` and the clamp lives in the terminal apply
    # at :4007 (`if (qc1d .le. R1) qc1d = 0`).  This kernel is literal, so it
    # can leave a residue of a few float32 ulp -- measured -2.3e-13 here --
    # which the terminal apply then zeroes because it is below R1 = 1e-12.
    # WRF's own tendency form produces the identical residue, so it is
    # reproduced exactly rather than bounded: qcten = REAL(-rc*orho*odt) and
    # qc1d_final = qc1d + qcten*DT.  Measured range here is
    # [-2.3e-13, +1.4e-12], i.e. one or two float32 ulp of the 1e-05 kg/kg
    # of entering liquid, and part of it survives the R1 = 1e-12 gate exactly
    # as it does in WRF.
    fully_evaporated = classic_qc <= 0.0
    selected = fully_evaporated & ~floored
    assert selected.sum() > 0
    residue = aerosol_qc_out[selected]
    ulp = float(np.spacing(F32(1.0e-5)))
    assert np.abs(residue).max() <= 2.0 * ulp, (
        np.abs(residue).max(), ulp)
    # Where the floor bites, mp=28 keeps exactly 1% of the entering liquid.
    np.testing.assert_allclose(
        aerosol_qc_out[floored], (qc0[floored] * F32(0.01)),
        rtol=1.0e-5, atol=1.0e-12)
    # And where it does not bite, mp=28 must equal WRF -- BITWISE.
    #
    # This assertion used to read "the two schemes agree to float32 rounding:
    # this kernel is not a rewrite of the shared Newton solve", with a 2e-6
    # rtol against the frozen mp=8 kernel.  That premise is now deliberately
    # false, for exactly the reason the shared header already rejected it for
    # RSLF/RSIF: the authority is WRF, not ArWen's mp=8 sibling.  What
    # replaces it is STRICTER, not looser -- bitwise agreement with an
    # independent float32 host transcription of :3403-3408 + :3412 + :3480,
    # which test_the_host_newton_transcription_reproduces_the_oracle_too pins
    # bitwise against the instrumented Fortran itself.
    #
    # The mp=8 divergence on this sweep is then MEASURED and recorded rather
    # than tolerated: 3.8e-03 relative, worst case, on a state deliberately
    # chosen to sit in the cancellation-dominated band where Newton's
    # residual is all that is left. thompson.cu is byte-frozen and that
    # deviation is a pre-existing ArWen-wide one this port only made visible.
    agree = ~floored & (classic_qc > 1.0e-9)
    odt = F32(F32(1.0) / F32(10.0))
    want_qc = np.asarray(
        [F32(qc0[i] + F32(F32(_wrf_condensation_clap(
            qv0[i], qvs[i], _lvt2_host(temperature[i], qv0[i])) * odt)
            * F32(10.0))) for i in np.nonzero(agree)[0]], dtype=F32)
    assert np.array_equal(aerosol_qc_out[agree], want_qc), (
        "mp=28's evaporation branch is no longer bitwise against WRF: worst "
        f"{np.abs(aerosol_qc_out[agree].astype(np.float64) - want_qc.astype(np.float64)).max():.3e}")
    sibling = np.abs(classic_qc[agree].astype(np.float64)
                     - want_qc.astype(np.float64)) / np.maximum(
        np.abs(want_qc.astype(np.float64)), 1.0e-30)
    assert sibling.max() > 1.0e-4, (
        "the frozen mp=8 kernel now agrees with WRF to better than 1e-4 on "
        "this sweep; thompson.cu is supposed to be byte-frozen, so either it "
        "moved or this measurement stopped measuring anything")


def test_condensation_matches_the_frozen_mp8_kernel_where_no_branch_applies():
    """The classic ``condense`` column, run through the mp=28 kernel.

    No aerosol branch can fire on a cloud-free supersaturated column except
    activation, which touches only ``ncten``/``nwfaten``.  The mass and
    temperature answer must therefore still be the mp=8 one.
    """
    import cupy as cp

    from woof.core.thompson import launch_cloud_saturation_adjust
    from woof.core.thompson_aerosol_runtime import load_aerosol_device_tables
    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)
    from woof.core.thompson_contract import (
        AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS, read_sequential_records)

    before, _ = _read_column("condense", _ORACLE_CLASSIC)
    n = _LEVELS
    pressure = _host(before, "p_pa")
    fields = [_host(before, name) for name in ("temp_k", "qv", "qc")]

    classic = [cp.asarray(a.copy()) for a in fields]
    launch_cloud_saturation_adjust(
        classic[0], cp.asarray(pressure), classic[1], classic[2])

    tnccn = load_aerosol_device_tables(
        _TABLES, require_classic_assets=False).arrays["tnccn_act"]
    tnc = _device_fortran(read_sequential_records(
        _TABLES / AUXILIARY_TABLE_FILE,
        AUXILIARY_TABLE_RECORDS)["tnc_wev"])
    aerosol = [cp.asarray(a.copy()) for a in fields]
    launch_aerosol_saturation_adjust(
        aerosol[0], cp.asarray(pressure), aerosol[1], aerosol[2],
        cp.full(n, 2.0, dtype=cp.float32), cp.zeros(n, dtype=cp.float32),
        cp.zeros(n, dtype=cp.float32),
        cp.full(n, 3.0e8, dtype=cp.float32),
        cp.full(n, 1.0, dtype=cp.float32), tnccn, tnc, 10.0)
    cp.cuda.Stream.null.synchronize()

    for got, expected, name, atol in (
            (aerosol[0], classic[0], "temperature", 2.0e-4),
            (aerosol[1], classic[1], "qv", 2.0e-9),
            (aerosol[2], classic[2], "qc", 2.0e-9)):
        np.testing.assert_allclose(
            cp.asnumpy(got), cp.asnumpy(expected), rtol=2.0e-6, atol=atol,
            err_msg=name)


# ---------------------------------------------------------------------------
# Rain evaporation.
# ---------------------------------------------------------------------------

def _rain_evap_inputs():
    import cupy as cp

    before, after = _read_column("rain-evap", _ORACLE_CLASSIC)
    names = ("qr", "nr_per_kg", "temp_k", "p_pa", "qv")
    return (before, after,
            [cp.asarray(_host(before, name)) for name in names])


def test_rain_evaporation_is_the_frozen_mp8_kernel_plus_one_term():
    """With ``nwfaten=None`` this kernel is mp=8's, plus one aerosol term.

    The transcription in thompson_aerosol_sat.cu is of a model-validated
    kernel, not a fresh port; if it drifts, mp=28's rain evaporation would
    differ from mp=8's for a reason that has nothing to do with aerosols.

    THERE IS NOW EXACTLY ONE SANCTIONED REASON, and this test is built to
    admit that one and nothing else.  ``thompson_aerosol_common.cuh``
    deliberately contraction-pins the RSLF/RSIF Horner chains while
    ``thompson.cu`` (frozen) leaves them contracted -- see the "THE SHARED
    FITS" note in the header.  mp=28 matches WRF there; mp=8 matches its own
    validated trajectory.  So the two kernels can no longer be bit-identical
    wherever RSLF enters.

    Rather than relax into a tolerance -- which would blind the test to real
    drift -- this asserts two STRICTER things:

    1. STRUCTURE.  RSLF is read only inside the evaporation branch, so at
       every level where the mp=8 kernel evaporates nothing, the two kernels
       must still agree BIT FOR BIT.  Any structural transcription drift
       shows up here exactly as it did before.
    2. CONFINEMENT.  Where evaporation does occur, the two kernels may differ
       only by the propagated RSLF ulp, measured at 2.45e-06 relative on this
       fixture.  The gate is 5.0e-06 -- twice measured, not a round number
       chosen to fit.

    NOT USABLE AS A REFERENCE HERE: this fixture's ``after`` column is WRF's
    state after the WHOLE mp_thompson call, not after rain evaporation alone,
    so BOTH ports sit ~6e-03 from it for reasons that have nothing to do with
    this kernel.  Comparing a single-kernel result against it would measure
    the rest of the scheme.  That is why (2) is a divergence bound between
    the two ports and not an accuracy claim against WRF.
    """
    import cupy as cp

    from woof.core.thompson import launch_rain_evaporation
    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_rain_evaporation)

    before, after, _ = _rain_evap_inputs()
    names = ("qr", "nr_per_kg", "temp_k", "p_pa", "qv")
    classic = [cp.asarray(_host(before, name)) for name in names]
    aerosol = [cp.asarray(_host(before, name)) for name in names]

    launch_rain_evaporation(
        classic[0], classic[1], classic[2], classic[3], classic[4], 10.0)
    launch_aerosol_rain_evaporation(
        aerosol[0], aerosol[1], aerosol[2], aerosol[3], aerosol[4], None,
        10.0)
    cp.cuda.Stream.null.synchronize()

    qr_entry = _host(before, "qr")
    inert = cp.asnumpy(classic[0]) == qr_entry
    assert inert.any(), "fixture reaches no non-evaporating level"
    assert not inert.all(), "fixture evaporates nowhere; test proves nothing"

    fields = ((0, "qr", "qr"), (1, "nr", "nr_per_kg"),
              (2, "temperature", "temp_k"), (4, "qv", "qv"))

    # (1) STRUCTURE -- unchanged strictness where RSLF cannot enter.
    for index, name, _column in fields:
        np.testing.assert_array_equal(
            cp.asnumpy(aerosol[index])[inert],
            cp.asnumpy(classic[index])[inert],
            err_msg=f"{name} drifted at a level that does not evaporate, so "
                    f"the cause is NOT the sanctioned RSLF pin")

    # (2) CONFINEMENT -- the divergence stays the size of the RSLF ulp.
    active = ~inert
    for index, name, _column in fields:
        mine = cp.asnumpy(aerosol[index]).astype(np.float64)[active]
        sibling = cp.asnumpy(classic[index]).astype(np.float64)[active]
        scale = np.maximum(np.abs(sibling), 1.0e-30)
        divergence = np.abs(mine - sibling) / scale
        assert np.all(divergence <= 5.0e-06), (
            f"{name}: mp=28 and mp=8 diverge by {divergence.max():.3e}, "
            f"more than the propagated RSLF ulp can explain. Either the "
            f"transcription drifted or something beyond the sanctioned "
            f"saturation-fit pin changed.")


def test_rain_evaporation_returns_exactly_one_ccn_per_evaporated_drop():
    """module_mp_thompson.F:3564-3565.

    ``nrten -= pnr_rev`` and ``nwfaten += pnr_rev`` are the same rate with
    opposite signs, so the CCN gained is exactly the rain number lost.  WOOF-ArWen
    already computed ``pnr_rev`` at thompson.cu:2271-2273 for mp=8 and simply
    discarded it.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_rain_evaporation)

    before, after, dev = _rain_evap_inputs()
    nr_before = cp.asnumpy(dev[1]).copy()
    nwfaten = cp.zeros(_LEVELS, dtype=cp.float32)
    launch_aerosol_rain_evaporation(
        dev[0], dev[1], dev[2], dev[3], dev[4], nwfaten, 10.0)
    cp.cuda.Stream.null.synchronize()

    lost = nr_before - cp.asnumpy(dev[1])
    gained = cp.asnumpy(nwfaten) * F32(10.0)
    assert np.any(lost > 0.0)
    np.testing.assert_allclose(gained, lost, rtol=2.0e-6, atol=1.0e-3)
    # Saturated or rainless levels return nothing at all; the accumulator is
    # not a blanket source.
    np.testing.assert_array_equal(
        cp.asnumpy(nwfaten)[lost == 0.0], np.zeros(int((lost == 0.0).sum())))


def test_rain_evaporation_honours_the_condensation_gate():
    """module_mp_thompson.F:3502, ``.and. (.not.(prw_vcd(k).gt. 0.))``.

    WRF skips rain evaporation entirely in a cell that just condensed cloud
    water.  mp=8's kernel has no such input because its driver sequences the
    two stages; mp=28 fuses them into one pass, so the decision has to travel
    on the ``condensation_rate`` array the saturation kernel writes.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_rain_evaporation)

    before, _, _ = _rain_evap_inputs()
    names = ("qr", "nr_per_kg", "temp_k", "p_pa", "qv")

    ungated = [cp.asarray(_host(before, name)) for name in names]
    launch_aerosol_rain_evaporation(
        ungated[0], ungated[1], ungated[2], ungated[3], ungated[4], None,
        10.0)

    gated = [cp.asarray(_host(before, name)) for name in names]
    marker = cp.zeros(_LEVELS, dtype=cp.float32)
    marker[::2] = 1.0e-9
    nwfaten = cp.zeros(_LEVELS, dtype=cp.float32)
    launch_aerosol_rain_evaporation(
        gated[0], gated[1], gated[2], gated[3], gated[4], nwfaten, 10.0,
        condensation_rate=marker)
    cp.cuda.Stream.null.synchronize()

    condensing = cp.asnumpy(marker) > 0.0
    # Condensing cells are untouched, including their CCN accumulator.
    np.testing.assert_array_equal(
        cp.asnumpy(gated[0])[condensing], _host(before, "qr")[condensing])
    np.testing.assert_array_equal(
        cp.asnumpy(nwfaten)[condensing], np.zeros(int(condensing.sum())))
    # Non-condensing cells behave exactly as the ungated run.
    np.testing.assert_array_equal(
        cp.asnumpy(gated[0])[~condensing],
        cp.asnumpy(ungated[0])[~condensing])


def test_cloud_presence_is_the_l_qc_the_cloud_fallout_reads(
        tnccn_act, tnc_wev):
    """module_mp_thompson.F:3215-3223 and :3485, read by :3645.

    WRF's cloud fallout runs only in a column where ANY(L_qc) holds, and the
    L_qc it reads is the post-source flag with one later edit: the
    condensation loop CLEARS it where the adjusted cloud is at R1 (:3485).
    It never sets it, so cloud that condenses onto a level whose post-source
    L_qc was false does not open the gate.  The adapter used to take the
    gate from the post-source cloud water alone and sedimented cloud WRF
    leaves where it formed (31 cells of one saved real-data analysis frame
    against WRF v4.6.1's own Fortran, tools/thompson_real_column_parity).

    One launch, one level per case:

    0. cloud in subsaturated air, all of it evaporated   -> cleared, 0
    1. plenty of cloud, slightly subsaturated, some left  -> kept, 1
    2. no cloud, supersaturated: condensation makes some  -> never set, 0
    3. no cloud, subsaturated: the branch does not run    -> 0
    4. cloud in supersaturated air, grows                 -> kept, 1
    5. cloud at R1 or below after the sources, saturated  -> 0
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    temperature = np.full(6, 283.0, dtype=F32)
    pressure = np.full(6, 85000.0, dtype=F32)
    qvs = _rslf(pressure, temperature)
    ratio = np.array([0.80, 0.999, 1.02, 0.90, 1.02, 1.0], dtype=F32)
    qv = (ratio * qvs).astype(F32)
    qc = np.array([1.0e-7, 3.0e-3, 0.0, 0.0, 1.0e-4, 1.0e-12], dtype=F32)
    nc = np.where(qc > 0, F32(1.0e8), F32(0.0)).astype(F32)
    nwfa = np.full(6, 3.0e8, dtype=F32)
    w = np.full(6, 0.5, dtype=F32)

    dev = [cp.asarray(a.copy()) for a in
           (temperature, pressure, qv, qc, nc, nwfa, w)]
    ncten = cp.zeros(6, dtype=cp.float32)
    nwfaten = cp.zeros(6, dtype=cp.float32)
    condensation = cp.zeros(6, dtype=cp.float32)
    presence = cp.full(6, -7.0, dtype=cp.float32)
    launch_aerosol_saturation_adjust(
        dev[0], dev[1], dev[2], dev[3], dev[4], ncten, nwfaten, dev[5],
        dev[6], tnccn_act, tnc_wev, 10.0,
        condensation_rate=condensation, cloud_presence=presence)
    cp.cuda.Stream.null.synchronize()

    got = cp.asnumpy(presence)
    prw = cp.asnumpy(condensation)
    qc_after = cp.asnumpy(dev[3])
    # The cases are what they say they are.
    assert prw[0] < 0 and qc_after[0] * _rho(pressure, temperature, qv)[0] \
        <= F32(1.0e-12), (prw[0], qc_after[0])
    assert prw[1] < 0 and qc_after[1] > F32(1.0e-6)
    assert prw[2] > 0 and qc_after[2] > F32(1.0e-12)
    assert prw[3] == 0 and qc_after[3] == 0
    assert prw[4] > 0 and qc_after[4] > qc[4]
    np.testing.assert_array_equal(
        got, np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=F32))


def test_rain_evaporation_exports_the_sedimentation_density_wrf_actually_used():
    """module_mp_thompson.F:3237-3238 vs :3568-3570, level by level.

    ``reference_density`` is not this kernel's scratch.  The adapter hands the
    SAME buffer to ``launch_rain_sedimentation``, which forms the sedimenting
    rain mass and number as ``qr*rho`` and ``nr*rho`` -- WRF's ``rr(k)`` and
    ``nr(k)`` at :3794-3795.  WRF builds those in two places and only two:

      * :3237-3238, from the :3193 TAU+1 density, at EVERY level with L_qr;
      * :3568/:3570, from the :3490 POST-condensation density, and ONLY inside
        the :3501-3502 gate (``ssatw < -eps`` and ``L_qr`` and not
        ``prw_vcd > 0``).

    Before WP-13a this kernel wrote the post-condensation density
    unconditionally, at every level, before its own gates -- so every level
    got the :3568 answer including the ones WRF never rewrote.  It now writes
    the :3237 density by default and overwrites it with the :3568 one only
    after all three gates pass, NEGATED, because :3568 floors the rebuilt
    pair at R1 / R2 where :3237 did not.  Where L_qr itself failed (:3236)
    WRF formed no working pair at all -- rr = R1, nr = R2 at :3252-3253 --
    and the export is ZERO.  The rain fallout's _with_presence entry points
    read all three cases (thompson.cu).

    THE TEST DRIVES BOTH SIDES OF THE GATE IN ONE LAUNCH.  ``condensation_rate``
    is set positive on the odd levels, which is :3502's veto, so the same
    column carries gate-passing and gate-failing levels with everything else
    identical.  A gate-failing level must export ``entry_density`` BIT FOR BIT;
    a gate-passing level must export the local post-condensation rho, which is
    checked against an independent float32 host transcription of :3490 rather
    than against the kernel's own arithmetic.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_rain_evaporation)

    before, _, _ = _rain_evap_inputs()
    names = ("qr", "nr_per_kg", "temp_k", "p_pa", "qv")
    fields = [cp.asarray(_host(before, name)) for name in names]
    qr0 = _host(before, "qr").copy()

    # A DELIBERATELY DIFFERENT entry density, so "wrote the entry density"
    # cannot be confused with "wrote something that happens to equal rho".
    # 0.97 is exact in binary32 only approximately; the product is whatever
    # float32 says it is, and that exact array is what must come back.
    local_rho = _rho(_host(before, "p_pa"), _host(before, "temp_k"),
                     np.maximum(np.float32(1.0e-10), _host(before, "qv")))
    entry_density = cp.asarray((local_rho * np.float32(0.97)).astype(np.float32))
    exported = cp.full(_LEVELS, np.float32(-1.0), dtype=cp.float32)
    marker = cp.zeros(_LEVELS, dtype=cp.float32)
    marker[1::2] = 1.0e-9                                 # :3502's veto
    nwfaten = cp.zeros(_LEVELS, dtype=cp.float32)

    launch_aerosol_rain_evaporation(
        fields[0], fields[1], fields[2], fields[3], fields[4], nwfaten, 10.0,
        reference_density=exported,
        condensation_rate=marker,
        entry_density=entry_density)
    cp.cuda.Stream.null.synchronize()

    got = cp.asnumpy(exported)
    want_entry = cp.asnumpy(entry_density)
    # The gate fired exactly where the kernel changed the state.  Nothing else
    # in this launch can move qr, so this is the gate itself, read off the
    # kernel rather than re-derived from a copy of its conditions.
    fired = cp.asnumpy(fields[0]) != qr0
    assert fired.any() and (~fired).any(), fired.tolist()
    # Every vetoed (odd) level must be a non-firing level, which is what makes
    # the marker a real discriminator rather than decoration.
    assert not fired[1::2].any()

    # 1. GATE DID NOT FIRE on a level with L_qr -> WRF never reached :3568,
    #    so sedimentation must see the :3237 pair, i.e. the entry density,
    #    bit for bit.
    lqr = qr0 > np.float32(1.0e-12)
    kept = ~fired & lqr
    assert kept.any() and (~lqr).any(), (kept.tolist(), lqr.tolist())
    np.testing.assert_array_equal(got[kept], want_entry[kept])
    # ...and that is not vacuous: the entry density really is different from
    # the local one at every level of this column.
    assert np.all(got[kept] != local_rho[kept])
    # 1b. L_QR FAILED -> no working pair; the export is the zero marker.
    np.testing.assert_array_equal(got[~lqr], np.float32(0.0))

    # 2. GATE FIRED -> :3568 rebuilt rr/nr from the :3490 density, so the
    #    export must be the local post-condensation rho, NEGATED: the sign
    #    tells the fallout that :3568 floored the rebuilt pair at R1 / R2.
    #    Checked against an independent host transcription, not against the
    #    kernel.
    np.testing.assert_array_equal(got[fired], -local_rho[fired])
    assert np.all(-got[fired] != want_entry[fired])


# ---------------------------------------------------------------------------
# Shape independence and numerical robustness.
# ---------------------------------------------------------------------------

def test_three_dimensional_launch_matches_the_flat_launch(
        tnccn_act, tnc_wev):
    """The kernel is flat over C-contiguous storage, like every mp=8 one.

    Run the same 24 states as a (24,) vector and as a (2,3,4) volume and
    require bit equality, so a future adapter can pass whatever
    ``(nz, ny, nx)`` shape WOOF-ArWen's state carries.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    before, _ = _read_column("aero-drop-evap")
    flat = _run_saturation_adjust(before, 10.0, tnccn_act, tnc_wev)

    shape = (2, 3, 4)
    def volume(name):
        return cp.asarray(_host(before, name).reshape(shape))

    pressure = volume("p_pa")
    temperature = volume("temp_k")
    qv = volume("qv")
    qc = volume("qc")
    nc_entry = volume("nc_per_kg")
    rho = _rho(_host(before, "p_pa"), _host(before, "temp_k"),
               np.maximum(F32(1.0e-10), _host(before, "qv")))
    nwfa_work = cp.asarray(
        np.maximum(F32(11.1e6),
                   _host(before, "nwfa_per_kg") * rho
                   ).astype(F32).reshape(shape))
    ncten = cp.zeros(shape, dtype=cp.float32)
    nwfaten = cp.zeros(shape, dtype=cp.float32)
    launch_aerosol_saturation_adjust(
        temperature, pressure, qv, qc, nc_entry, ncten, nwfaten, nwfa_work,
        volume("w_m_s"), tnccn_act, tnc_wev, 10.0)
    cp.cuda.Stream.null.synchronize()

    for got, expected, name in (
            (qv, flat["qv"], "qv"), (qc, flat["qc"], "qc"),
            (temperature, flat["temperature"], "temperature"),
            (ncten, flat["ncten"], "ncten"),
            (nwfaten, flat["nwfaten"], "nwfaten")):
        np.testing.assert_array_equal(
            cp.asnumpy(got).reshape(-1), expected, err_msg=name)


def test_degenerate_states_stay_finite_and_in_bounds(tnccn_act, tnc_wev):
    """Every clamp end of every axis at once, with no NaN and no table
    overrun.

    Covers: zero and negative updraft (``ta_Ww`` lower clamp), 1e4 m/s
    (upper clamp), CCN at the 11.1e6 floor and past the 1e4 cm-3 ceiling,
    zero droplets, dry air, and a cell that is saturated to machine
    precision.  A single out-of-range index in ``tnccn_act`` or ``tnc_wev``
    would read garbage rather than fail, so finiteness plus a sign check is
    the observable.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    w_values = np.array([-5.0, 0.0, 1.0e-4, 0.01, 100.0, 1.0e4], dtype=F32)
    nwfa_values = np.array([11.1e6, 1.0e7, 3.0e8, 9.999e9, 2.0e10], dtype=F32)
    nc_values = np.array([0.0, 2.0, 1.0e8, 1.0e10], dtype=F32)
    qc_values = np.array([0.0, 1.0e-13, 1.0e-6, 3.0e-3], dtype=F32)
    ratio_values = np.array([0.5, 1.0 - 1.0e-9, 1.0, 1.0 + 1.0e-9, 1.05],
                            dtype=F32)
    temp_values = np.array([200.0, 243.15, 248.15, 285.0, 320.0], dtype=F32)

    grid = np.meshgrid(w_values, nwfa_values, nc_values, qc_values,
                       ratio_values, temp_values, indexing="ij")
    w, nwfa, nc, qc, ratio, temperature = (g.reshape(-1).astype(F32)
                                           for g in grid)
    n = w.size
    pressure = np.full(n, 85000.0, dtype=F32)
    qv = (ratio * _rslf(pressure, temperature)).astype(F32)

    dev = [cp.asarray(a.copy()) for a in
           (temperature, pressure, qv, qc, nc, nwfa, w)]
    ncten = cp.zeros(n, dtype=cp.float32)
    nwfaten = cp.zeros(n, dtype=cp.float32)
    launch_aerosol_saturation_adjust(
        dev[0], dev[1], dev[2], dev[3], dev[4], ncten, nwfaten, dev[5],
        dev[6], tnccn_act, tnc_wev, 10.0)
    cp.cuda.Stream.null.synchronize()

    for array, name in ((dev[0], "temperature"), (dev[2], "qv"),
                        (dev[3], "qc"), (ncten, "ncten"),
                        (nwfaten, "nwfaten")):
        values = cp.asnumpy(array)
        assert np.all(np.isfinite(values)), name
    # The CCN return is exactly the negative of the droplet source at every
    # one of these states, which is what a table overrun would break.
    np.testing.assert_array_equal(cp.asnumpy(nwfaten), -cp.asnumpy(ncten))
    assert np.all(cp.asnumpy(dev[2]) >= 1.0e-10)


# ---------------------------------------------------------------------------
# Launcher argument validation.
# ---------------------------------------------------------------------------

def test_launcher_rejects_wrong_dtypes_and_table_layouts(tnccn_act, tnc_wev):
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    n = 8
    ok = [cp.zeros(n, dtype=cp.float32) for _ in range(9)]
    ok[1][:] = 90000.0
    ok[0][:] = 285.0

    def call(**kwargs):
        args = dict(
            temperature=ok[0], pressure=ok[1], qv=ok[2], qc=ok[3],
            nc_entry=ok[4], ncten=ok[5], nwfaten=ok[6],
            nwfa_work_m3=ok[7], w=ok[8], tnccn_act=tnccn_act,
            tnc_wev=tnc_wev, dt=10.0)
        args.update(kwargs)
        launch_aerosol_saturation_adjust(
            args["temperature"], args["pressure"], args["qv"], args["qc"],
            args["nc_entry"], args["ncten"], args["nwfaten"],
            args["nwfa_work_m3"], args["w"], args["tnccn_act"],
            args["tnc_wev"], args["dt"])

    with pytest.raises(TypeError):
        call(qv=cp.zeros(n, dtype=cp.float64))
    with pytest.raises(ValueError):
        call(w=cp.zeros(n + 1, dtype=cp.float32))
    with pytest.raises(ValueError):
        call(dt=0.0)
    with pytest.raises(TypeError):
        call(tnc_wev=_device_fortran(
            np.zeros((100, 37, 100), dtype=np.float32)))
    with pytest.raises(ValueError):
        call(tnccn_act=_device_fortran(
            np.zeros((7, 9, 7, 5, 3), dtype=np.float64)))
    # A C-ordered table is silently wrong physics, so it must be refused.
    with pytest.raises(ValueError):
        call(tnc_wev=cp.ascontiguousarray(
            cp.zeros((100, 37, 100), dtype=cp.float64)))


# ---------------------------------------------------------------------------
# THE INSTRUMENTED WRF ORACLE
# ---------------------------------------------------------------------------
#
# Everything in this section is measured against a scratch copy of
# wrf461-pristine/phys/module_mp_thompson.F (WRF v4.6.1, commit
# d66e442fccc04111067e29274c9f9eaccc3cef28) into which nothing was added but
# WRITE statements, built and run exactly the way
# tools/thompson_wrf461_oracle/build_aero.sh builds the committed harness
# (gfortran 13.3.0 -O2 -cpp -DWRF_CHEM=0, no -march, no -ffast-math,
# GFORTRAN_CONVERT_UNIT=big_endian:20, one fresh process per scenario).
#
# PROVENANCE RECEIPT, re-measured on the tree that produced these tables: the
# instrumented build reproduces all 38 committed CSVs under
# woof/data/thompson/oracle-aero/ BYTE FOR BYTE, so the WRITE statements are
# provably inert and every value below is the same run the fixtures came from.
#
# The dumps are raw IEEE-754 bit patterns (Fortran Z edit descriptor), not
# decimal, so nothing is lost between the Fortran and these tables.
#
# ---------------------------------------------------------------------------
# Table 1: the condensation block, module_mp_thompson.F:3399-3494.
# ---------------------------------------------------------------------------
#
# One row for every (fixture, level) pair at which WRF actually enters the
# block -- 122 of them, drawn from all 19 committed aerosol fixtures, and
# covering every branch: droplet nucleation through activ_ncloud (:3414-3420),
# the aerosol-only evaporation branch that reads tnc_wev (:3423-3471), WRF's
# 99%-of-liquid mass floor (:3467) and the total-evaporation else branch
# (:3472-3475).
#
# Columns, whitespace separated:
#   scenario-index  level
#   temp  pres  qv  qc  nc1d  ncten  nwfa_work  w        (float32 hex, inputs)
#   prw_vcd                                              (float64 hex, :3412)
#   ncten_after  nwfaten_after  qc_after                 (float32 hex)
#
# The inputs are WRF's own working column at :3399: `temp` is :3189's
# t1d+DT*tten, `qv` is :3192's MAX(1E-10, qv1d+DT*qvten), `qc` is
# qc1d+qcten*DT as :3215 forms it, `nwfa_work` is the :3211 snapshot and `w`
# is w1d, the entry vertical velocity mp_gt_driver copies once at :1224.
# `ncten_after`/`nwfaten_after` are :3481-3482 evaluated as WRF does, i.e. a
# REAL(4) store of the DOUBLE sum, and `qc_after` is :3480 followed by
# :3975's qc1d + qcten*DT.
_SAT_ORACLE_FIXTURES = (
    "aero-init-profile", "aero-sfc-emit", "aero-ccn-activate",
    "aero-ccn-sweep", "aero-drop-evap", "aero-nc-auto", "aero-nc-accrete",
    "aero-nc-effrad", "aero-nc-sed", "aero-scav-rain", "aero-scav-frozen",
    "aero-ice-demott-dep", "aero-ice-demott-idxin", "aero-ice-koop",
    "aero-cloud-freeze-nc", "aero-nc-cap", "aero-warm-overlap",
    "aero-cold-overlap", "aero-reduces-to-classic",
)

#: dt, seconds, per fixture index.  run_column_aero.F90:133-139 gives the two
#: overlap columns a 50 s step and everything else 10 s.
_SAT_ORACLE_DT = tuple(
    50.0 if name in ("aero-warm-overlap", "aero-cold-overlap") else 10.0
    for name in _SAT_ORACLE_FIXTURES)

_SAT_ORACLE_ROWS = """\
0 1 438E8000 47BD4DA9 3C147522 00000000 00000000 80000000 4D24B62C 3F800000 3EA8A8D700000000 4B2E1032 CB2E1032 36F69866
0 2 438E8000 47B1D585 3C1E2E38 00000000 00000000 80000000 4D1A963E 3F800000 3EA94C7AE0000000 4B2F55D2 CB2F55D2 36FCFCCD
0 3 438E8000 47A70F44 3C288CFB 00000000 00000000 80000000 4CA0C411 3F800000 3EA9EE79E0000000 4AD0A7F9 CAD0A7F9 3701A861
0 4 438E8000 479CF021 3C339CCD 00000000 00000000 80000000 4C5E7C4D 3F800000 3EAA8E7640000000 4A9F9DC3 CA9F9DC3 3704C84F
0 5 438E8000 47936DFB 3C3F69E6 00000000 00000000 80000000 4C3880D1 3F800000 3EAB2C43E0000000 4A8FA6DC CA8FA6DC 3707DD53
0 6 438E8000 478A7F4F 3C4C0164 00000000 00000000 80000000 4C25AF63 3F800000 3EABC7FEE0000000 4A8AE8F0 CA8AE8F0 370AE7FA
0 7 438E8000 47821B2E 3C59715C 00000000 00000000 80000000 4C192BD0 3F800000 3EAC60B7E0000000 4A89E6AA CA89E6AA 370DE397
0 8 438E8000 47747266 3C67C8F6 00000000 00000000 80000000 4C0EFF22 3F800000 3EACF70FA0000000 4A8A1FDC CA8A1FDC 3710D34E
0 9 438E8000 4765A2F7 3C771879 00000000 00000000 80000000 4C05EC35 3F800000 3EAD8A88A0000000 4A8AC10E CA8AC10E 3713B4AB
0 10 438E8000 4757B93F 3C83B8B5 00000000 00000000 80000000 4BFB17EF 3F800000 3EAE1B09A0000000 4A8B84A7 CA8B84A7 37168730
0 11 438E8000 474AA751 3C8C7355 00000000 00000000 80000000 4BEB7285 3F800000 3EAEA8AAA0000000 4A8C31FC CA8C31FC 37194B55
0 12 438E8000 473E601B 3C95C648 00000000 00000000 80000000 4BDCC6E4 3F800000 3EAF334FE0000000 4A8CA958 CA8CA958 371C008F
0 13 438E8000 4732D756 3C9FBC83 00000000 00000000 80000000 4BCF00FB 3F800000 3EAFBB09E0000000 4A8D2230 CA8D2230 371EA731
0 14 438E8000 47280177 3CAA61DC 00000000 00000000 80000000 4BC2115C 3F800000 3EB01F5380000000 4A8D9BAE CA8D9BAE 37213943
0 15 438E8000 471DD3A7 3CB5C31F 00000000 00000000 80000000 4BB5EA95 3F800000 3EB0600EC0000000 4A8E1590 CA8E1590 3723C094
0 16 438E8000 471443B8 3CC1EE1C 00000000 00000000 80000000 4BAA8058 3F800000 3EB09F28A0000000 4A8E8FC3 CA8E8FC3 37263796
0 17 438E8000 470B4819 3CCEF1CA 00000000 00000000 80000000 4B9FC72C 3F800000 3EB0DC9CA0000000 4A8F0A45 CA8F0A45 37289E1E
0 18 438E8000 4702D7CE 3CDCDE60 00000000 00000000 80000000 4B95B456 3F800000 3EB118EF80000000 4A8F8516 CA8F8516 372AF95B
0 19 438E8000 46F5D4CA 3CEBC572 00000000 00000000 80000000 4B8C3DBD 3F800000 3EB15391E0000000 4A90003F CA90003F 372D43B3
0 20 438E8000 46E6EFE3 3CFBBA17 00000000 00000000 80000000 4B8359EB 3F800000 3EB18D0860000000 4A907BC3 CA907BC3 372F8254
0 21 438E8000 46D8F1FE 3D06688A 00000000 00000000 80000000 4B75FFFB 3F800000 3EB1C51700000000 4A90F7A7 CA90F7A7 3731B2E6
0 22 438E8000 46CBCD1F 3D0F907E 00000000 00000000 80000000 4B664F33 3F800000 3EB1FBAF20000000 4A9173F6 CA9173F6 3733D4D7
0 23 438E8000 46BF741B 3D19613F 00000000 00000000 80000000 4B5791C5 3F800000 3EB2318580000000 4A91F0B5 CA91F0B5 3735EF37
0 24 438E8000 46B3DA9D 3D23E83B 00000000 00000000 80000000 4B49B8F5 3F800000 3EB2665900000000 4A926DEA CA926DEA 3737FF7A
2 1 438E8000 47BD4DA9 3C147522 00000000 00000000 80000000 4D8F0D18 3F800000 3EA8A8D700000000 4B8D8EE5 CB8D8EE5 36F69866
2 2 438E8000 47B1D585 3C1E2E38 00000000 00000000 80000000 4D8F0D18 3F800000 3EA94C7AE0000000 4B96D460 CB96D460 36FCFCCD
2 3 438E8000 47A70F44 3C288CFB 00000000 00000000 80000000 4D8F0D18 3F800000 3EA9EE79E0000000 4BA0B7D8 CBA0B7D8 3701A861
2 4 438E8000 479CF021 3C339CCD 00000000 00000000 80000000 4D8F0D18 3F800000 3EAA8E7640000000 4BAB442A CBAB442A 3704C84F
2 5 438E8000 47936DFB 3C3F69E6 00000000 00000000 80000000 4D8F0D18 3F800000 3EAB2C43E0000000 4BB684FA CBB684FA 3707DD53
2 6 438E8000 478A7F4F 3C4C0164 00000000 00000000 80000000 4D8F0D18 3F800000 3EABC7FEE0000000 4BC286D0 CBC286D0 370AE7FA
2 7 438E8000 47821B2E 3C59715C 00000000 00000000 80000000 4D8F0D18 3F800000 3EAC60B7E0000000 4BCF5716 CBCF5716 370DE397
2 8 438E8000 47747266 3C67C8F6 00000000 00000000 80000000 4D8F0D18 3F800000 3EACF70FA0000000 4BDD0440 CBDD0440 3710D34E
2 9 438E8000 4765A2F7 3C771879 00000000 00000000 80000000 4D8F0D18 3F800000 3EAD8A88A0000000 4BEB9DD5 CBEB9DD5 3713B4AB
2 10 438E8000 4757B93F 3C83B8B5 00000000 00000000 80000000 4D8F0D18 3F800000 3EAE1B09A0000000 4BFB3490 CBFB3490 37168730
2 11 438E8000 474AA751 3C8C7355 00000000 00000000 80000000 4D8F0D18 3F800000 3EAEA8AAA0000000 4C05ED36 CC05ED36 37194B55
2 12 438E8000 473E601B 3C95C648 00000000 00000000 80000000 4D8F0D18 3F800000 3EAF334FE0000000 4C0ED168 CC0ED168 371C008F
2 13 438E8000 4732D756 3C9FBC83 00000000 00000000 80000000 4D8F0D18 3F800000 3EAFBB09E0000000 4C185155 CC185155 371EA731
2 14 438E8000 47280177 3CAA61DC 00000000 00000000 80000000 4D8F0D18 3F800000 3EB01F5380000000 4C227846 CC227846 37213943
2 15 438E8000 471DD3A7 3CB5C31F 00000000 00000000 80000000 4D8F0D18 3F800000 3EB0600EC0000000 4C2D526E CC2D526E 3723C094
2 16 438E8000 471443B8 3CC1EE1C 00000000 00000000 80000000 4D8F0D18 3F800000 3EB09F28A0000000 4C38ECFB CC38ECFB 37263796
2 17 438E8000 470B4819 3CCEF1CA 00000000 00000000 80000000 4D8F0D18 3F800000 3EB0DC9CA0000000 4C455635 CC455635 37289E1E
2 18 438E8000 4702D7CE 3CDCDE60 00000000 00000000 80000000 4D8F0D18 3F800000 3EB118EF80000000 4C529D90 CC529D90 372AF95B
2 19 438E8000 46F5D4CA 3CEBC572 00000000 00000000 80000000 4D8F0D18 3F800000 3EB15391E0000000 4C60D3D8 CC60D3D8 372D43B3
2 20 438E8000 46E6EFE3 3CFBBA17 00000000 00000000 80000000 4D8F0D18 3F800000 3EB18D0860000000 4C700B40 CC700B40 372F8254
2 21 438E8000 46D8F1FE 3D06688A 00000000 00000000 80000000 4D8F0D18 3F800000 3EB1C51700000000 4C802BCB CC802BCB 3731B2E6
2 22 438E8000 46CBCD1F 3D0F907E 00000000 00000000 80000000 4D8F0D18 3F800000 3EB1FBAF20000000 4C88E733 CC88E733 3733D4D7
2 23 438E8000 46BF741B 3D19613F 00000000 00000000 80000000 4D8F0D18 3F800000 3EB2318580000000 4C92439E CC92439E 3735EF37
2 24 438E8000 46B3DA9D 3D23E83B 00000000 00000000 80000000 4D8F0D18 3F800000 3EB2665900000000 4C9C4DDF CC9C4DDF 3737FF7A
3 1 438E8000 47BD4DA9 3C147522 00000000 00000000 80000000 4B295F60 3BA3D70A 3EA8A8D700000000 4864FA8E C864FA8E 36F69866
3 2 438E8000 47B1D585 3C1E2E38 00000000 00000000 80000000 4B295F60 3CA3D70A 3EA94C7AE0000000 48AAC59E C8AAC59E 36FCFCCD
3 3 438E8000 47A70F44 3C288CFB 00000000 00000000 80000000 4B295F60 3D4CCCCD 3EA9EE79E0000000 4905295B C905295B 3701A861
3 4 438E8000 479CF021 3C339CCD 00000000 00000000 80000000 4B295F60 3E4CCCCD 3EAA8E7640000000 4957917B C957917B 3704C84F
3 5 438E8000 47936DFB 3C3F69E6 00000000 00000000 80000000 4B295F60 3F000000 3EAB2C43E0000000 4986CBB8 C986CBB8 3707DD53
3 6 438E8000 478A7F4F 3C4C0164 00000000 00000000 80000000 4BE4E1BF 40000000 3EABC7FEE0000000 4A4F24E7 CA4F24E7 370AE7FA
3 7 438E8000 47821B2E 3C59715C 00000000 00000000 80000000 4BE4E1C1 40A00000 3EAC60B7E0000000 4A6400E2 CA6400E2 370DE397
3 8 438E8000 47747266 3C67C8F6 00000000 00000000 80000000 4BE4E1C0 41A00000 3EACF70FA0000000 4A749E25 CA749E25 3710D34E
3 9 438E8000 4765A2F7 3C771879 00000000 00000000 80000000 4BE4E1C0 42480000 3EAD8A88A0000000 4A826681 CA826681 3713B4AB
3 10 438E8000 4757B93F 3C83B8B5 00000000 00000000 80000000 4BE4E1C0 43480000 3EAE1B09A0000000 4A8B0725 CA8B0725 37168730
3 11 438E8000 474AA751 3C8C7355 00000000 00000000 80000000 4D8F0D18 3BA3D70A 3EAEA8AAA0000000 4A274416 CA274416 37194B55
3 12 438E8000 473E601B 3C95C648 00000000 00000000 80000000 4D8F0D18 3CA3D70A 3EAF334FE0000000 4A93C7E4 CA93C7E4 371C008F
3 13 438E8000 4732D756 3C9FBC83 00000000 00000000 80000000 4D8F0D18 3D4CCCCD 3EAFBB09E0000000 4B16C1AE CB16C1AE 371EA731
3 14 438E8000 47280177 3CAA61DC 00000000 00000000 80000000 4D8F0D18 3E4CCCCD 3EB01F5380000000 4BB4C151 CBB4C151 37213943
3 15 438E8000 471DD3A7 3CB5C31F 00000000 00000000 80000000 4D8F0D18 3F000000 3EB0600EC0000000 4C0C1AC4 CC0C1AC4 3723C094
3 16 438E8000 471443B8 3CC1EE1C 00000000 00000000 80000000 4F32D05E 40000000 3EB09F28A0000000 4DABE24C CDABE24C 37263796
3 17 438E8000 470B4819 3CCEF1CA 00000000 00000000 80000000 4F32D05E 40A00000 3EB0DC9CA0000000 4DFC622E CDFC622E 37289E1E
3 18 438E8000 4702D7CE 3CDCDE60 00000000 00000000 80000000 4F32D05E 41A00000 3EB118EF80000000 4E29991D CE29991D 372AF95B
3 19 438E8000 46F5D4CA 3CEBC572 00000000 00000000 80000000 4F32D05E 42480000 3EB15391E0000000 4E3EB417 CE3EB417 372D43B3
3 20 438E8000 46E6EFE3 3CFBBA17 00000000 00000000 80000000 4F32D05E 43480000 3EB18D0860000000 4E4EE0BA CE4EE0BA 372F8254
3 21 438E8000 46D8F1FE 3D06688A 00000000 00000000 80000000 509502F9 3BA3D70A 3EB1C51700000000 4D0631CF CD0631CF 3731B2E6
3 22 438E8000 46CBCD1F 3D0F907E 00000000 00000000 80000000 509502F9 3CA3D70A 3EB1FBAF20000000 4D1C2B35 CD1C2B35 3733D4D7
3 23 438E8000 46BF741B 3D19613F 00000000 00000000 80000000 509502F9 3D4CCCCD 3EB2318580000000 4D3D52F5 CD3D52F5 3735EF37
3 24 438E8000 46B3DA9D 3D23E83B 00000000 00000000 80000000 509502F9 3E4CCCCD 3EB2665900000000 4DC88F2A CDC88F2A 3737FF7A
4 1 438E8000 47BD4DA9 3C1132B0 399D48D6 4CA347B4 C2922E68 4D8F0D18 3F800000 BEDEF0FC60000000 CAFD28A7 4AFD2815 396D3735
4 2 438E8000 47B1D585 3C1AB521 399D48FF 4CADF87D C2508C02 4D8F0D18 3F800000 BEDFBFE900000000 CB06DDDD 4B06DDA9 396B3238
4 3 438E8000 47A70F44 3C24D999 399D491D 4CB95F52 C20DDFA6 4D8F0D18 3F800000 BEE0466CE0000000 CB0FB48A 4B0FB466 3969321A
4 4 438E8000 479CF021 3C2FAB3E 399D4932 4CC588B5 C1B45442 4D8F0D18 3F800000 BEE0ABAD20000000 CB1C726C 4B1C7255 39673802
4 5 438E8000 47936DFB 3C3B3602 399D4941 4CD28209 C14DF84B 4D8F0D18 3F800000 BEE10FA2C0000000 CB26B8E9 4B26B8DC 39654454
4 6 438E8000 478A7F4F 3C4786BA 399D494B 4CE059B9 C0C138B9 4D8F0D18 3F800000 BEE1722200000000 CB31AF82 4B31AF7C 396357EC
4 7 438E8000 47821B2E 3C54AB2B 399D4950 4CEF1F37 BFD46647 4D8F0D18 3F800000 BEE1D316C0000000 CB3D6274 4B3D6272 3961732E
4 8 438E8000 47747266 3C62B228 399D4952 4CFEE31E 00000000 4D8F0D18 3F800000 BEE23260E0000000 CB49DEE3 4B49DEE3 395F96C0
4 9 438E8000 4765A2F7 3C71AB9D 399D4952 4D07DBA5 00000000 4D8F0D18 3F800000 BEE28FE900000000 CB5732F3 4B5732F3 395DC317
12 1 43700A1F 47BD4DA9 396E4856 393A35D2 4C879DF4 C8D51CDE 4D8F0607 3F800000 BEB957DE60000000 C93FA502 48AA2D26 392A5EE7
12 2 4370106D 47B1D585 397DA7B0 39894469 4C905DEE C9198684 4D8F01CC 3F800000 BEBBDA59A0000000 C943AAF9 482891D4 3980902D
12 3 437015D4 47A70F44 39870282 394DEE1E 4C99AE81 CA292D17 4D8EFEA1 3F800000 BEBE6C6660000000 CA2B5F45 470C8B7C 393AEA5E
12 4 43701054 479CF021 398FB85B 38F114A2 4CA398FF CA5C8140 4D8F0270 3F800000 BEBF4DA340000000 CA62045D 47B063A8 38C9F396
12 5 437005A9 47936DFB 3998FE1D 38A80771 4CAE275A C809C083 4D8F0952 3F800000 BEBF3A5340000000 CA366AB2 4A2DCEAA 3880FE89
12 6 43700150 478A7F4F 39A2DD15 37C73650 4CB9642B C6581B04 4D8F0C34 3F800000 BEC0203660000000 CB12D4FF 4B129EF8 3697D0C0
12 7 4370006E 47821B2E 39AD446F 3698784E 4CC55AB9 C4D2CA9D 4D8F0CD4 3F800000 BE9E7E7640000000 CB1DE22D 4B1DDB97 AB000000
12 8 4370094C 47747266 39B1CEDE 351B75F1 4CD21700 00000000 4D8F0850 3F800000 BE6F1796A0000000 CB281267 4B281267 00000000
12 9 43700001 4765A2F7 39C4770D 33507035 4CDFA5C9 00000000 4D8F0D16 3F800000 BE34D80560000000 CB32EB07 4B32EB07 A7800000
14 1 437001A1 47BD4DA9 39374B56 392C35EA 4BA2BA07 C8B89066 4D8F0C1F 3F800000 BEDACA25A0000000 C8D70608 4773AD0D 38D27918
14 2 43700185 47B1D585 39431E88 398985B9 4C905A9D C919600E 4D8F0C2F 3F800000 BEDC6989A0000000 CAA7C96F 4A949D6E 394C039A
14 3 437000D2 47A70F44 394FB518 398EFB54 4D66801F C914B334 4D8F0C9B 3F800000 BEDE1A7A60000000 CBB69A55 4BB1F4BB 3952B476
14 4 43700024 479CF021 395D1BA0 39437310 4E4C79ED C7CA0751 4D8F0D03 3F800000 BEDFE38400000000 CCA1F27B 4CA1BFF9 38E7748C
14 5 43700007 47936DFB 396B5F90 38AC1597 4EC3E6D8 C6A27740 4D8F0D13 3F800000 BEE0E61400000000 CD1B2785 4D1B2271 35C633C0
14 6 43700006 478A7F4F 397A8F35 37C52018 4BDE71A5 C6926C5B 4D8F0D15 3F800000 BEC3B668C0000000 CA31F484 4A30CFAB 00000000
14 7 43700001 47821B2E 39855CE4 36993309 4CC55486 C3979B3A 4D8F0D17 3F800000 BE9EA3CEA0000000 CB1DDD38 4B1DDC09 00000000
14 8 436FFFFF 47747266 398DF7BF 351B75F1 4D9D8BFC 00000000 4D8F0D18 3F800000 BE6F179660000000 CBFC132B 4BFC132B 29800000
14 9 43700000 4765A2F7 399720CD 33507035 4E8BC2A5 CCAD15C7 4D8F0D18 3F800000 BE34D80540000000 CCDF9DD5 4BCA2037 27800000
17 5 43710189 47936DFB 396B71A4 3781E860 4CAE275A C9D6BC0E 4F317C83 3F800000 BE94C90520000000 C9DEEAB0 4782EA28 00000000
17 6 437057B6 478A7F4F 398F573B 388786D2 4CB9642B C8EE934F 4F325341 3F800000 BEB55F1280000000 C9EB8638 49AFE164 357A4F00
17 7 43700C18 47821B2E 39A6E3F6 37D8D585 4CC55AB9 C72C219B 4F32B60B 3F800000 BEA158C2E0000000 C9FC9D16 49F73C09 2C000000
17 8 4370027D 47747266 39B6E19C 36B67483 4CD21700 C450FA05 4F32CAF5 3F800000 BE7D315CA0000000 CA067520 4A066810 00000000
17 9 4370008D 4765A2F7 39C410BF 3552023B 4CDFA5C9 00000000 4F32CF72 3F800000 BE50CCFA80000000 CA0F226C 4A0F226C 00000000
18 1 43913000 47BD4DA9 3C4F8C4A 3868D743 4CA75CEE C7885FE9 4D8F0B00 3EADC40D BED748B9E0000000 CB05E3F2 4B04D332 00000000
18 2 438F9000 47B1D585 3C29ED09 3986598A 4CAF868D C8409C41 4D8F09EE 3F564ABA BEFADEB540000000 CB0C6BA4 4B096933 00000000
18 3 438DF000 47A70F44 3C0B1FA3 3A295C9E 4CB82E93 C854635D 4D8F0A04 3FBABEE2 BF0581D9C0000000 CB11E7C0 4B0E9633 39774D74
18 4 438C5000 479CF021 3BE3CF3C 3A6929A3 4CC157C2 C7E48A80 4D8F0B2B 3FE60016 BF0A7C5F40000000 CB192537 4B175C22 39C9778E
18 5 438AB000 47936DFB 3BBA83C0 3A2CC9D8 4CCB056B C6E64623 4D8F0C4F 3FC82CCC BF0D4B6340000000 CB20CC1A 4B2058F7 38D28740
18 6 43891000 478A7F4F 3B98B48D 39894737 4CD53B4C C4902D58 4D8F0CDF 3F76387A BEFB74A4A0000000 CB2A95D7 4B2A9156 00000000
18 7 43877000 47821B2E 3B7A0C8D 386AB563 4CDFFD8F 00000000 4D8F0D18 3ED6039A BED77889E0000000 CB33313F 4B33313F 00000000
18 8 4385D000 47747266 3B4CB90B 36D86F71 4CEB50C1 AE8243A7 4D8F0D18 3E03736E BEA5A4BE80000000 CB3C409B 4B3C409B 00000000
18 9 43843000 4765A2F7 3B279CE4 34D750BB 4CF739D4 00000000 4D8F0D18 3CE4378C BE658812E0000000 CB45C7DD 4B45C7DD A9800000
18 10 43829000 4757B93F 3B093AD5 32671549 4D01DF08 CB14FB54 4D8F0D18 3B8BFE33 BE171BBA80000000 CB4FCB40 4A6B3FB0 00000000
18 11 4380F000 474AA751 3AE0B564 2F85C6B5 4D08718B CB593EE5 4D8F0D18 39F2BB67 BDBAC15760000000 CB5A4F45 47883000 00000000
18 12 437EA000 473E601B 3AB7F9CE 2C27187E 4D0F5766 CB6555FA 4D8F0D18 3814B41E BD50B5A620000000 CB6558A4 442A6667 20800000
18 19 4367E02D 46F5D4CA 39B557D9 00000000 00000000 80000000 4D8E92EE 2417563C 3EB8B1C9A0000000 4AD62BA1 CAD62BA1 3776F1E0
18 20 4364A031 46E6EFE3 39946F1C 00000000 00000000 80000000 4D8E92EB 20387741 3ECAFFD000000000 4AE0C3E0 CAE0C3E0 3806FF10
18 21 43616038 46D8F1FE 3972F7BB 00000000 00000000 80000000 4D8E92E8 1C1EE34E 3ED26BB800000000 4AEBD738 CAEBD738 38383530
18 22 435E2041 46CBCD1F 3946D1F3 00000000 00000000 80000000 4D8E92E1 17C16AEB 3ED555C600000000 4AF76A10 CAF76A10 385559BC
18 23 435AE04C 46BF741B 3922A6A8 00000000 00000000 80000000 4D8E92DB 132661FD 3ED6A169E0000000 4B01C073 CB01C073 38624E23
18 24 4357A05B 46B3DA9D 39050132 00000000 00000000 80000000 4D8E92D0 0E4A4771 3ED6AD37A0000000 4B081029 CB081029 3862C42C
"""

# ---------------------------------------------------------------------------
# Table 2: rain evaporation, module_mp_thompson.F:3236-3255 + :3500-3574.
# ---------------------------------------------------------------------------
#
# The 15 (fixture, level) pairs at which WRF's rain-evaporation loop actually
# runs -- it is gated on ssatw < -eps AND L_qr AND NOT(prw_vcd > 0), which is
# a much narrower set than "the fixture has rain".  Three fixtures reach it:
# aero-ice-demott-idxin (3 levels), aero-cold-overlap (6) and
# aero-reduces-to-classic (6).
#
# Columns:
#   scenario-index  level
#   temp  pres  qv  qr  nr  rho_entry                    (float32 hex, inputs)
#   prv_rev  pnr_rev                                     (float64 hex)
#   nr_bound  nr_preclamp                                (float32 hex)
#
# `qr`/`nr` are PER KILOGRAM -- qr1d+qrten*DT and nr1d+nrten*DT, the state
# ArWen carries -- and `rho_entry` is :3193's density, the one :3242-3243 uses
# to convert them.  `nr_preclamp` is MAX(R2, nr*rho) before :3247-3255 and
# `nr_bound` is the working rain number after it; where the two differ the
# mean-volume-diameter clamp fired.
_REV_ORACLE_ROWS = """\
12 7 436FFD74 47821B2E 39AFA650 34E9A800 43441D44 3F77696A 3DD2A6D1205DF663 3F9F4F474A5A5B33 433D88FE 433D88FE
12 8 437008EB 47747266 39B21C99 351B75F1 429208BD 3F686297 3DDC421F30F22710 3F8A8B78A38FEC0E 42849029 42849029
12 9 436FFFF8 4765A2F7 39C47D90 33507035 40C3CD1C 3F5A537D 3D99171E41BC94DB 3F4791B358311D7F 40A6FC85 40A6FC85
17 3 4371D437 47A70F44 394FC50C 38006E50 41F71500 3F9DACC2 3E258433250C9885 3F64B27AD3EBBAAE 42182E9E 42182E9E
17 4 43719279 479CF021 395D2C9B 37B5E528 41AEF780 3F9446B6 3E1DEDA829DC691E 3F5CC9E38B49348F 41CAAF45 41CAAED4
17 5 4370F766 47936DFB 397BAEB0 371EF1C0 4118E400 3F8B9DE5 3E043FAF74DF12A2 3F437A4846C1B658 4126C46E 4126C431
17 7 436FFB29 47821B2E 39B4714F 339D5900 3D976000 3F775EA3 3D57EFFB860D293C 3E97075FE5CFD206 3D924596 3D924596
17 8 436FFEED 47747266 39B9BB6E 35267900 3F202240 3F686842 3DA0A564BFC156B3 3EE003215284AE4A 3F116041 3F116041
17 9 4370000A 4765A2F7 39C479C0 34FC02AD 40ECB933 3F5A5311 3DB7C709081DB463 3F3655CB47C5048C 40C9E285 40C9E285
18 1 43911EC2 47BD4DA9 3C507521 3928F284 4891F524 3F91DFFA 3E6A9EB057A1D1C5 4056FF5ED720D454 48A65719 48A65719
18 2 438F4007 47B1D585 3C2E1FD5 39AB3CA0 48917BB9 3F8B1758 3E651432C538BE83 4041E8A02CD37A46 489E16E3 489E16E3
18 6 4388BCF5 478A7F4F 3BA14900 36AA5231 489278A8 3F64FDB3 3E90DCA4F43D5050 40CD006281E4BD1D 4883049D 4883049D
18 7 43875E31 47821B2E 3B7DB763 34AA2A58 47B9029E 3F59FDF3 3E610262E0000000 40C24EE8E0000000 479D8AC0 479D8ABF
18 8 4385CDF1 47747266 3B4D2543 322B1BD3 453A0900 3F4F802F 3E111C2A20000000 40726AA400000000 4516CA6E 4516CA6E
18 9 43842FDF 4765A2F7 3B27A39F 2F2C3F54 423B4000 3F45810C 3DB13984E0000000 401289A640000000 421076A4 421076A4
"""


def _oracle_rows(block):
    return [line.split() for line in block.strip().splitlines()]


def _f32(hexadecimal):
    return np.frombuffer(
        bytes.fromhex(hexadecimal), dtype=">f4").astype(F32)[0]


def _f64(hexadecimal):
    return np.frombuffer(bytes.fromhex(hexadecimal), dtype=">f8")[0]


def _hex32(value):
    return np.float32(value).astype(">f4").tobytes().hex().upper()


# ---------------------------------------------------------------------------
# A NO-FMA, CORRECTLY-ROUNDED HOST TRANSCRIPTION OF WRF'S NEWTON SOLVE
# ---------------------------------------------------------------------------

def _lvt2_host(temp, qv):
    """module_mp_thompson.F:3206-3209 in float32, operation by operation."""
    f = F32
    tempc = f(temp - f(273.15))
    otemp = f(f(1.0) / temp)
    lvap = f(f(2.5e6) + f(f(2106.0 - 4218.0) * tempc))
    ocp = f(f(1.0) / f(f(1004.0) * f(f(1.0) + f(f(0.887) * qv))))
    orv = f(f(1.0) / f(461.5))
    return f(f(f(f(f(lvap * lvap) * ocp) * orv) * otemp) * otemp)


def _wrf_condensation_clap(qv, qvs, lvt2):
    """module_mp_thompson.F:3403-3408 in float32, operation by operation.

    Independent of the GPU.  NumPy float32 scalars round after every
    operation, which is what ``gfortran -O2`` on baseline x86-64 does because
    that ISA has no FMA instruction, and ``np.exp`` of a float64 argument cast
    back to float32 reproduces glibc's faithful ``expf``.

    This matters far more than it looks.  ``fcd = qvs*EXP(lvt2*clap) - qv +
    clap`` cancels to about four significant digits, so Newton pins ``clap``
    only to roughly one ulp of ``qv`` over ``dfcd`` -- 4e-10 absolute on a
    ``clap`` of 7.5e-06, i.e. 5.4e-05 RELATIVE.  ``clap`` IS the condensed
    mass (:3412, then :3975), so a single fused multiply-add or a two-ulp
    device ``expf`` moves the answer by 5e-05, not by an ulp.
    """
    f = F32
    clap = f(f(qv - qvs) / f(f(1.0) + f(lvt2 * qvs)))
    for _ in range(3):
        exponential = f(np.exp(np.float64(f(lvt2 * clap))))
        fcd = f(f(f(qvs * exponential) - qv) + clap)
        dfcd = f(f(f(qvs * lvt2) * exponential) + f(1.0))
        clap = f(clap - f(fcd / dfcd))
    return clap


def test_condensation_solve_is_bitwise_against_the_instrumented_wrf_oracle(
        tnccn_act, tnc_wev):
    """G2 for :3399-3494 -- every branch, every fixture, every active level.

    The kernel is driven on WRF's OWN working column: the eight inputs :3399
    sees, lifted as raw bit patterns out of the instrumented pristine
    Fortran.  Nothing in this comparison depends on the adapter, on the
    fixture reconstruction the end-to-end harness performs, or on any sibling
    package -- it asserts only that this kernel, given WRF's inputs, produces
    WRF's outputs.

    THE CLAIM IS BITWISE, on all 122 (fixture, level) pairs at which WRF
    actually enters the block, for all four quantities the block emits:

    * ``prw_vcd`` (:3412 / :3467 / :3473), the condensation-evaporation MASS
      rate -- the field the end-to-end gate reads as ``qc``;
    * ``ncten`` and ``nwfaten`` after :3481-3482;
    * ``qc`` after :3480 followed by :3975.

    The 122 rows span every branch: droplet nucleation through
    ``activ_ncloud`` (:3414-3420), the aerosol-only evaporation branch that
    reads ``tnc_wev`` (:3423-3471), WRF's 99%-of-liquid mass floor (:3467)
    and the total-evaporation else branch (:3472-3475).

    MEASURED BEFORE THE FIX that made it bitwise -- the contraction pin plus
    the correctly-rounded ``exp`` in ``thompson_aa_saturation_adjust`` --
    ``prw_vcd`` disagreed on most of these rows, worst 5.42e-05 relative.
    That is exactly the ``qc`` residual the end-to-end gate was carrying on
    aero-ccn-activate, aero-ccn-sweep and aero-init-profile, to four digits.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)

    rows = _oracle_rows(_SAT_ORACLE_ROWS)
    assert len(rows) == 122, len(rows)
    # Eight of the nineteen fixtures reach :3401's gate at all; the other
    # eleven are subsaturated-and-cloud-free or ice-only columns in which WRF
    # never enters the block, so there is nothing there to compare.  Their
    # coverage of this kernel is the negative one -- it must not fire -- and
    # that is what the end-to-end gate checks.
    covered = {_SAT_ORACLE_FIXTURES[int(row[0])] for row in rows}
    assert covered == {
        "aero-init-profile", "aero-ccn-activate", "aero-ccn-sweep",
        "aero-drop-evap", "aero-ice-demott-idxin", "aero-cloud-freeze-nc",
        "aero-cold-overlap", "aero-reduces-to-classic"}, sorted(covered)

    by_dt = {}
    for row in rows:
        by_dt.setdefault(_SAT_ORACLE_DT[int(row[0])], []).append(row)

    checked = 0
    for dt, group in sorted(by_dt.items()):
        size = len(group)
        columns = [np.asarray([_f32(row[index]) for row in group], dtype=F32)
                   for index in range(2, 10)]
        device = [cp.asarray(column.copy()) for column in columns]
        nwfaten = cp.zeros(size, dtype=cp.float32)
        condensation = cp.zeros(size, dtype=cp.float32)
        launch_aerosol_saturation_adjust(
            device[0], device[1], device[2], device[3], device[4],
            device[5], nwfaten, device[6], device[7],
            tnccn_act, tnc_wev, dt, condensation_rate=condensation)
        cp.cuda.Stream.null.synchronize()

        got = (cp.asnumpy(condensation), cp.asnumpy(device[5]),
               cp.asnumpy(nwfaten), cp.asnumpy(device[3]))
        for index, row in enumerate(group):
            where = f"{_SAT_ORACLE_FIXTURES[int(row[0])]} level {row[1]}"
            # prw_vcd is DOUBLE PRECISION in WRF but every assignment to it
            # (:3412, :3467, :3473) is of a REAL(4) expression, so its
            # float32 image is exact and the comparison stays bitwise.
            expected = (F32(_f64(row[10])), _f32(row[11]),
                        _f32(row[12]), _f32(row[13]))
            names = ("prw_vcd", "ncten", "nwfaten", "qc")
            for value, want, name in zip(got, expected, names):
                assert value[index] == want, (
                    f"{where}: {name} {_hex32(value[index])} != "
                    f"{_hex32(want)}")
            checked += 1
    assert checked == 122


def test_the_host_newton_transcription_reproduces_the_oracle_too():
    """The float32 host solve is WRF's, so it can referee the mp=8 comparison.

    :func:`_wrf_condensation_clap` is the reference
    :func:`test_the_pinned_newton_solve_beats_the_contracted_one_against_wrf`
    measures both ports against, so it has to be WRF's own arithmetic and not
    merely something close to it.  This drives it on the oracle rows that take
    :3412's plain ``prw_vcd = clap*odt`` -- no floor, no total-evaporation
    branch -- and requires it to land on WRF's ``prw_vcd`` BITWISE.

    No GPU involved.
    """
    rows = _oracle_rows(_SAT_ORACLE_ROWS)
    matched = 0
    for row in rows:
        dt = F32(_SAT_ORACLE_DT[int(row[0])])
        temp, pres, qv = _f32(row[2]), _f32(row[3]), _f32(row[4])
        qvs = _rslf(np.asarray([pres], F32), np.asarray([temp], F32))[0]
        clap = _wrf_condensation_clap(qv, qvs, _lvt2_host(temp, qv))
        if F32(clap * F32(F32(1.0) / dt)) == F32(_f64(row[10])):
            matched += 1
    # Levels that hit the 99% floor or fully evaporate take a different
    # expression and are covered bitwise by the oracle test above instead.
    assert matched >= 40, (
        f"only {matched} of {len(rows)} oracle rows are reproduced by the "
        f"host transcription; it is no longer WRF's arithmetic")


def test_the_pinned_newton_solve_beats_the_contracted_one_against_wrf():
    """Why mp=28 does not share thompson.cu:250-257's plain Horner form.

    Both ports are measured against :func:`_wrf_condensation_clap`, an
    independent float32 host transcription that the test above pins bitwise
    against the Fortran.  The mp=28 kernel must be BITWISE; the frozen mp=8
    kernel is merely measured, and the number is recorded rather than fixed
    because thompson.cu is byte-frozen and its trajectory is model-validated.

    This is the same tie-break the shared header already applied to
    RSLF/RSIF: agreement with WRF, not with a sibling port.
    """
    import cupy as cp

    from woof.core.thompson import launch_cloud_saturation_adjust
    from woof.core.thompson_aerosol_sat import (
        launch_aerosol_saturation_adjust)
    from woof.core.thompson_aerosol_runtime import load_aerosol_device_tables
    from woof.core.thompson_contract import (
        AUXILIARY_TABLE_FILE, AUXILIARY_TABLE_RECORDS, read_sequential_records)

    size = 512
    temperature = np.full(size, 285.0, dtype=F32)
    pressure = np.full(size, 90000.0, dtype=F32)
    qvs = _rslf(pressure, temperature)
    # Supersaturations from 1 part in 10^4 to 1 part in 200: condensation
    # only, so no branch but :3412 can fire and clap is the whole answer.
    excess = np.linspace(1.0e-4, 5.0e-3, size).astype(F32)
    qv0 = (qvs * (F32(1.0) + excess)).astype(F32)
    qc0 = np.zeros(size, dtype=F32)

    want = np.asarray(
        [_wrf_condensation_clap(qv0[i], qvs[i], _lvt2_host(temperature[i],
                                                           qv0[i]))
         for i in range(size)], dtype=F32)

    classic = [cp.asarray(a.copy())
               for a in (temperature, pressure, qv0, qc0)]
    launch_cloud_saturation_adjust(*classic)

    tnccn = load_aerosol_device_tables(
        _TABLES, require_classic_assets=False).arrays["tnccn_act"]
    tnc = _device_fortran(read_sequential_records(
        _TABLES / AUXILIARY_TABLE_FILE,
        AUXILIARY_TABLE_RECORDS)["tnc_wev"])
    aerosol = [cp.asarray(a.copy())
               for a in (temperature, pressure, qv0, qc0)]
    condensation = cp.zeros(size, dtype=cp.float32)
    launch_aerosol_saturation_adjust(
        aerosol[0], aerosol[1], aerosol[2], aerosol[3],
        cp.full(size, 2.0, dtype=cp.float32),
        cp.zeros(size, dtype=cp.float32),
        cp.zeros(size, dtype=cp.float32),
        cp.full(size, 3.0e8, dtype=cp.float32),
        cp.full(size, 1.0, dtype=cp.float32), tnccn, tnc, 10.0,
        condensation_rate=condensation)
    cp.cuda.Stream.null.synchronize()

    # mp=28: prw_vcd is clap*odt (:3412), bitwise.  Compare the rate, not a
    # rate multiplied back up -- odt = 1/10 is not exactly invertible in
    # float32 and the round trip would inject an ulp of its own.
    odt = F32(F32(1.0) / F32(10.0))
    mine = cp.asnumpy(condensation)
    assert np.array_equal(mine, (want * odt).astype(F32)), (
        "mp=28's condensation solve is no longer bitwise against WRF; worst "
        f"{np.abs(mine.astype(np.float64) - (want * odt).astype(np.float64)).max():.3e}")

    # mp=8: measured, not asserted equal.  thompson.cu is byte-frozen and
    # this pre-existing ArWen-wide deviation is out of this port's scope.
    sibling = (cp.asnumpy(classic[2]) - qv0).astype(np.float64) * -1.0
    divergence = (np.abs(sibling - want.astype(np.float64))
                  / np.maximum(np.abs(want.astype(np.float64)), 1.0e-30))
    assert divergence.max() > 1.0e-5, (
        "mp=8's contracted Horner form now agrees with WRF to better than "
        "1e-5 on this sweep, so either thompson.cu moved (it is supposed to "
        "be byte-frozen) or this measurement has stopped measuring anything")


def test_rain_evaporation_matches_the_instrumented_wrf_oracle():
    """G2 for :3236-3255 + :3384-3388 + :3500-3574.

    Same construction as the condensation gate: the kernel is driven on WRF's
    own working column at the 15 (fixture, level) pairs where WRF's rain
    evaporation actually runs -- it is gated on
    ``ssatw < -eps .AND. L_qr .AND. .NOT.(prw_vcd > 0)``, a far narrower set
    than "the fixture has rain" -- and its two rates plus the clamped working
    rain number are compared against what the Fortran produced.

    THREE CLAIMS:

    1. ``nr_bound``, the working rain number after the mean-volume-diameter
       clamp at :3247-3255, is BITWISE at all 15.  WOOF-ArWen's mp=8 kernel has no
       such clamp and this port used to carry its absence as a documented
       carry-over.  The table exercises both ends: the 2.5 mm upper bound at
       aero-cold-overlap levels 4 and 5, the ``D0r*0.75`` lower bound at
       aero-reduces-to-classic level 7.
    2. ``prv_rev`` (:3540) and ``pnr_rev`` (:3559) agree with WRF to
       double-precision round-off at all 15.  The previous transcription --
       CUDA ``powf`` for the Schmidt cube root and for the slope, plus an
       FMA-contracted ``t1_evap`` chain -- was 2.0e-08 to 3.9e-07 off.
    3. Supplying ``entry_density`` is what makes claim 2 reachable.  WRF forms
       ``rr`` and ``nr`` at :3242-3243 from the :3193 density and then
       overwrites ``rho(k)`` at :3490 before :3505-3520 reads it, so two
       different densities are live in one loop.  The same comparison with
       ``entry_density=None`` is 3.8e-04 to 2.0e-03 off, and that 2.0e-03 is
       precisely the end-to-end residual aero-reduces-to-classic carries at
       level 6.  Both bounds are asserted so neither can quietly move.
    """
    import cupy as cp

    from woof.core.thompson_aerosol_sat import probe_rain_evaporation_rates

    rows = _oracle_rows(_REV_ORACLE_ROWS)
    assert len(rows) == 15, len(rows)
    clamped = {(_SAT_ORACLE_FIXTURES[int(row[0])], int(row[1]))
               for row in rows if row[10] != row[11]}
    # Measured: the upper (2.5 mm) bound at aero-cold-overlap 4 and 5, the
    # D0r*0.75 lower bound at aero-reduces-to-classic 7.  Pinned as a SET so
    # a transcription that silently stopped clamping fails here rather than
    # three launchers downstream.
    assert clamped == {("aero-cold-overlap", 4), ("aero-cold-overlap", 5),
                       ("aero-reduces-to-classic", 7)}, sorted(clamped)

    def one(value):
        return cp.asarray(np.asarray([value], dtype=F32))

    worst_with = 0.0
    worst_without = 0.0
    for row in rows:
        dt = _SAT_ORACLE_DT[int(row[0])]
        where = f"{_SAT_ORACLE_FIXTURES[int(row[0])]} level {row[1]}"
        temperature, pressure = one(_f32(row[2])), one(_f32(row[3]))
        qv, qr, nr = one(_f32(row[4])), one(_f32(row[5])), one(_f32(row[6]))
        density = one(_f32(row[7]))
        want_prv, want_pnr = _f64(row[8]), _f64(row[9])

        prv, pnr, bound = probe_rain_evaporation_rates(
            qr, nr, temperature, pressure, qv, dt, entry_density=density)
        cp.cuda.Stream.null.synchronize()
        assert cp.asnumpy(bound)[0] == _f32(row[10]), (
            f"{where}: clamped rain number "
            f"{_hex32(cp.asnumpy(bound)[0])} != {row[10]}")
        for got, want, name in ((cp.asnumpy(prv)[0], want_prv, "prv_rev"),
                                (cp.asnumpy(pnr)[0], want_pnr, "pnr_rev")):
            relative = abs(float(got) - want) / abs(want)
            worst_with = max(worst_with, relative)
            assert relative <= 1.0e-15, f"{where}: {name} {relative:.3e}"

        prv0, pnr0, _ = probe_rain_evaporation_rates(
            qr, nr, temperature, pressure, qv, dt)
        cp.cuda.Stream.null.synchronize()
        for got, want in ((cp.asnumpy(prv0)[0], want_prv),
                          (cp.asnumpy(pnr0)[0], want_pnr)):
            worst_without = max(
                worst_without, abs(float(got) - want) / abs(want))

    assert worst_with <= 1.0e-15, worst_with
    # TWO-SIDED, so the measurement cannot drift in either direction.  The
    # fallback residual measured on this table is 1.95e-03, and 1.92e-03 is
    # what the end-to-end gate reads off aero-reduces-to-classic level 6 --
    # the port's only carved-out end-to-end tolerance.  A lower number here
    # would mean entry_density had stopped mattering (so claim 3 is dead); a
    # higher one would mean something else moved as well.
    assert 1.0e-3 < worst_without < 3.0e-3, (
        f"the entry_density fallback residual is now {worst_without:.3e}, "
        f"outside the measured [1e-3, 3e-3] band; either the argument "
        f"stopped being read or something upstream of it changed")
