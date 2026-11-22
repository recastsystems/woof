"""Grades the differential oracle and the defect it found.

THE DEFECT.  :func:`woof.ingest.real.initialize_real` used to raise the
interpolated total pressure to the target dry pressure
(``np.maximum(total_pressure_h, dry_pressure)``) before forming theta.
``real.exe`` has no such step: it hands the vert_interp result straight to
``t_to_theta`` (dyn_em/module_initialize_real.F:1848-1852).  The clamp only
ever fired in the below-surface extrapolation branch -- where WRF's own CRC
formula deliberately returns a pressure BELOW the target dry pressure -- and
because theta is formed once and never recomputed, a pressure raised there
was baked into the initial state permanently.

Every test below runs on CPU.  The synthetic ones run anywhere; the ones
that need the matched forcing/wrfinput pairs skip when those files are not
on the box, and say which path was missing.
"""

from pathlib import Path
import shutil

import numpy as np
import pytest

from woof.config import RunConfig
from woof.case_data import case_data_root
from woof.core import constants as c
from woof.core.grid import make_vertical_coord, finalize_vertical_coord
from woof.ingest import real as R
from woof.ingest.horiz import HorizontalSnapshot
from woof.ingest.preprocess_backend import resolve_preprocess_backend
from woof.verify import metem_differential as MD

from datetime import datetime


CASE_2021 = case_data_root() / "wrf_gpu_run_3km_gnu_batch1"
FORCING_2021 = CASE_2021 / "met_em.d01.2021-12-30_17:00:00.nc"
FORCING_2021_NEXT = CASE_2021 / "met_em.d01.2021-12-30_18:00:00.nc"
WRFINPUT_2021 = CASE_2021 / "wrfinput_d01"

_pairs = pytest.mark.skipif(
    not (FORCING_2021.is_file() and WRFINPUT_2021.is_file()),
    reason=f"matched forcing/wrfinput pair absent: {FORCING_2021}")


# --------------------------------------------------------------------------
# A synthetic case whose target terrain sits far BELOW the forcing's own
# terrain, so the lowest eta levels fall below the source column's surface
# pseudo-level and the extrapolation branch -- the only branch the clamp
# ever reached -- is genuinely exercised.
# --------------------------------------------------------------------------

def _synthetic(terrain_m, source_orography_m, *, nz=8, ny=3, nx=3):
    # The ladder must reach ABOVE p_top or the target column has no source
    # at its own top, which the kernel refuses (and rightly).
    levels_hpa = np.array([700.0, 500.0, 300.0, 150.0, 80.0, 40.0],
                          dtype=np.float64)
    nsrc = levels_hpa.size
    mass = (nsrc, ny, nx)
    pressure = np.broadcast_to(levels_hpa[:, None, None], mass) * 100.0
    # A standard-atmosphere column: hydrostatically consistent enough for
    # the setup arithmetic.  The test grades ArWen against ArWen's own
    # kernel, not against nature.
    profile = 44330.0 * (1.0 - (levels_hpa * 100.0 / 101325.0) ** 0.1903)
    height = np.broadcast_to(profile[:, None, None], mass).copy()
    tt = np.maximum(216.65, 288.15 - 6.5e-3 * height)
    fields = {
        "TT": tt.astype(np.float32),
        "RH": np.full(mass, 55.0, dtype=np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((nsrc, ny, nx + 1), 7.0, dtype=np.float32),
        "VV": np.full((nsrc, ny + 1, nx), -3.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 70000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 283.0, dtype=np.float32),
        "RH2": np.full((ny, nx), 60.0, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 4.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -2.0, dtype=np.float32),
    }
    snapshot = HorizontalSnapshot(valid_time=datetime(2021, 12, 30, 17),
                                 levels_hpa=levels_hpa, fields=fields)
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=0.2)
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=20000.0,
                    dt=12.0, run_seconds=60.0, moist=True, mp_physics=6,
                    base_temp=290.0, hybrid_opt=2, etac=0.2,
                    hypsometric_opt=2, terrain_opt=1)
    terrain = np.full((ny, nx), float(terrain_m), dtype=np.float64)
    orography = np.full((ny, nx), float(source_orography_m), dtype=np.float64)
    return snapshot, cfg, coord, terrain, orography


def _kernel_replay(snapshot, cfg, coord, terrain, orography, p_top=5000.0):
    """What the SAME interpolation kernel says T and p are on the eta grid.

    A transcription of initialize_real's own float64 setup, stopping at the
    two vert_interp results.  theta must be exactly the pairing of these
    two, because that is what ``t_to_theta`` is.
    """
    finalize_vertical_coord(coord, p_top)
    f = {k: np.asarray(v, dtype=np.float64) for k, v in snapshot.fields.items()}
    pressure = np.broadcast_to(
        snapshot.levels_hpa[:, None, None] * 100.0, f["TT"].shape).copy()
    surface_qv = R._saturation_mixing_ratio(f["T2"], f["PSFC"], f["RH2"])
    source_qv = R._cap_stratospheric_qv(
        R._saturation_mixing_ratio(f["TT"], pressure, f["RH"]), pressure)
    psfc_adj = R.surface_pressure_from_surface(
        f["PSFC"], orography, terrain, f["T2"], surface_qv)
    source_pd, intq, order = R._integrate_moisture(
        source_qv, pressure, f["TT"], f["GHT"], f["PSFC"], f["T2"],
        surface_qv, orography)
    pressure_o = R._ordered_levels(pressure, order)
    temperature_o = R._ordered_levels(f["TT"], order)
    surface_pd = f["PSFC"] - intq
    dry_mass = psfc_adj - intq - p_top
    dry_pressure = (coord.c3h[:, None, None] * dry_mass[None]
                    + coord.c4h[:, None, None] + p_top)
    backend = resolve_preprocess_backend("cpu")
    plan = backend.prepare_wrf_vertical(backend.float32(source_pd),
                                        backend.float32(surface_pd),
                                        backend.float32(dry_pressure))
    temperature = np.asarray(plan.apply(
        backend.float32(temperature_o), backend.float32(f["T2"]),
        interp_in_logp=True, extrap="temperature"), dtype=np.float64)
    total = np.asarray(plan.apply(
        backend.float32(pressure_o), backend.float32(f["PSFC"]),
        interp_in_logp=False, extrap="temperature"), dtype=np.float64)
    return temperature, total, dry_pressure


def test_the_synthetic_case_really_reaches_the_extrapolation_branch():
    """POSITIVE CONTROL for the gate below: without this the gate is vacuous."""
    args = _synthetic(terrain_m=400.0, source_orography_m=2600.0)
    _, total, dry_pressure = _kernel_replay(*args)
    below = total < dry_pressure
    assert below.any(), (
        "the fixture no longer exercises the below-surface extrapolation "
        "branch, so the clamp gate would pass for the wrong reason")
    assert float((dry_pressure - total)[below].max()) > 1000.0


def test_theta_is_the_interpolated_temperature_at_the_interpolated_pressure():
    """THE GATE.  Fails with the clamp in place, passes without it.

    real.exe forms theta as ``t_to_theta(grid%t_2, grid%p)`` -- the
    vert_interp temperature at the vert_interp pressure, with nothing in
    between.  Any floor, clamp or substitution applied to that pressure
    shows up here and nowhere else, because theta is never recomputed.
    """
    snapshot, cfg, coord, terrain, orography = _synthetic(
        terrain_m=400.0, source_orography_m=2600.0)
    result = R.initialize_real(
        snapshot, cfg, coord, terrain, source_orography=orography,
        p_top=5000.0, preprocess_backend="cpu", state_backend="preprocess")
    temperature, total, _ = _kernel_replay(
        snapshot, cfg, coord, terrain, orography)
    expected = R._potential_temperature_from_temperature(temperature, total)
    actual = (np.asarray(result.state.thb, dtype=np.float64)
              + np.asarray(result.state.thp, dtype=np.float64))
    assert np.max(np.abs(actual - expected)) < 1.0e-3, (
        "theta was not formed from the interpolated pressure: max "
        f"|d| = {np.max(np.abs(actual - expected)):.6g} K")


def test_the_gate_is_inert_where_the_branch_is_not_reached():
    """The same identity holds on a case with no below-surface extrapolation.

    The clamp could only fire where the interpolated pressure fell below the
    target dry pressure, so a case that never gets there must be unchanged
    by its removal -- and this is how that is stated as a measurement rather
    than an assertion about history.
    """
    args = _synthetic(terrain_m=2500.0, source_orography_m=2500.0)
    temperature, total, dry_pressure = _kernel_replay(*args)
    assert not (total < dry_pressure).any()
    snapshot, cfg, coord, terrain, orography = args
    result = R.initialize_real(
        snapshot, cfg, coord, terrain, source_orography=orography,
        p_top=5000.0, preprocess_backend="cpu", state_backend="preprocess")
    expected = R._potential_temperature_from_temperature(temperature, total)
    actual = (np.asarray(result.state.thb, dtype=np.float64)
              + np.asarray(result.state.thp, dtype=np.float64))
    assert np.max(np.abs(actual - expected)) < 1.0e-3


@pytest.mark.parametrize(
    "bad_pressure", [-1.0, 0.0, np.nan, np.inf, -np.inf],
    ids=["negative", "zero", "nan", "positive-infinity", "negative-infinity"],
)
def test_an_invalid_interpolated_pressure_is_refused_not_clamped(
        monkeypatch, bad_pressure):
    """NEGATIVE CONTROL for the guard that replaced the clamp.

    The clamp used to silently absorb a pathological interpolated pressure.
    What replaced it must REFUSE one, by name.
    """
    snapshot, cfg, coord, terrain, orography = _synthetic(
        terrain_m=400.0, source_orography_m=2600.0)
    original = R.resolve_preprocess_backend

    class Poisoned:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def prepare_wrf_vertical(self, *args, **kwargs):
            plan = self._inner.prepare_wrf_vertical(*args, **kwargs)
            outer = self

            class Wrapped:
                def apply(self, field, surface, **kw):
                    value = plan.apply(field, surface, **kw)
                    if kw.get("interp_in_logp") is False:
                        value = np.array(value, copy=True)
                        value[0, 0, 0] = np.float32(bad_pressure)
                    return value
            return Wrapped()

    monkeypatch.setattr(
        R, "resolve_preprocess_backend",
        lambda *a, **k: Poisoned(original(*a, **k)))
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        R.initialize_real(
            snapshot, cfg, coord, terrain, source_orography=orography,
            p_top=5000.0, preprocess_backend="cpu",
            state_backend="preprocess")


def test_the_clamp_is_gone_from_the_source_and_says_why():
    """The removal carries its citation, so nobody restores it by accident."""
    text = (Path(R.__file__)).read_text(encoding="utf-8")
    assert "np.maximum(\n        _host(total_pressure)" not in text
    assert "NOT clamped to the target dry pressure" in text
    assert "module_initialize_real.F:1795-1807" in text


# --------------------------------------------------------------------------
# The oracle's own refusals.
# --------------------------------------------------------------------------

def test_read_metem_refuses_a_file_without_the_surface_in_level_one(tmp_path):
    """NEGATIVE CONTROL: the ladder/surface split is checked, not assumed."""
    netCDF4 = pytest.importorskip("netCDF4")
    if not FORCING_2021.is_file():
        pytest.skip(f"forcing absent: {FORCING_2021}")
    broken = tmp_path / "broken.nc"
    shutil.copy(FORCING_2021, broken)
    ds = netCDF4.Dataset(str(broken), "a")
    pressure = np.asarray(ds.variables["PRES"][0])
    pressure[0] = pressure[0] + 1.0
    ds.variables["PRES"][0] = pressure
    ds.close()
    with pytest.raises(ValueError, match="surface-in-level-one convention"):
        MD.read_metem(broken)


def test_read_metem_keeps_the_3d_pressure_of_a_model_level_rh_file(tmp_path):
    """RH on levels that are not isobaric is not read on a flat ladder.

    Before, the oracle took levels_hpa from column (0, 0) and put the file
    on the RH lane, which flattens PRES: a model-level RH file (an Arwen
    Global set) was scored against a state built on the wrong pressure.
    """
    pytest.importorskip("netCDF4")
    from test_metem_ingest import write_case

    def tilt(ds):
        pres = np.asarray(ds["PRES"][0], np.float64)
        pres[1:] += np.linspace(-1000.0, 1000.0, pres.shape[2])[None, None, :]
        ds["PRES"][0] = pres
    path = write_case(tmp_path / "met_em.d01.2021-06-01_00_00_00.nc",
                      rh=63.0, mutate=tilt)
    case = MD.read_metem(path)
    assert case.lane == "relative-humidity-3d-pressure"
    assert "RH" not in case.fields and {"PRES", "SPFH", "Q2"} <= set(case.fields)
    assert float(np.ptp(case.fields["PRES"][0])) > 1000.0
    # The oracle's own transcription agrees with the reader's derivation.
    from woof.ingest.metem import read_met_em
    door = read_met_em(path).snapshot.fields
    np.testing.assert_allclose(case.fields["SPFH"], door["SPFH"], rtol=2e-7, atol=0)
    np.testing.assert_allclose(case.fields["Q2"], door["Q2"], rtol=2e-7, atol=0)
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    flat = MD.read_metem(write_case(flat_dir / "met_em.d01.2021-06-01_00_00_00.nc"))
    assert flat.lane == "relative-humidity" and "RH" in flat.fields


@_pairs
def test_a_mismatched_pair_is_refused_on_shape():
    """NEGATIVE CONTROL: a forcing file from another case cannot be scored."""
    other = case_data_root() / "ruc-real-wrfinput-20260726-noahcontrol"
    forcing = sorted(other.glob("met_em.d01.*"))
    if not forcing:
        pytest.skip(f"second case absent: {other}")
    with pytest.raises(ValueError, match="shapes differ"):
        MD.run_differential(forcing[0], WRFINPUT_2021, mp_physics=6)


# --------------------------------------------------------------------------
# The measurement itself.
# --------------------------------------------------------------------------

@_pairs
def test_arwen_reproduces_real_exe_on_the_2021_case():
    """The oracle's headline, as thresholds rather than prose.

    Every bound here is loose by at least 3x against the measured value, so
    it fails on a regression rather than on noise.  The theta bound is the
    one the clamp broke: with the clamp it was 3.56 K.
    """
    report = MD.run_differential(FORCING_2021, WRFINPUT_2021)
    rows = {row["field"]: row for row in report["rows"]}
    bounds = {
        "HGT (terrain)": (0.0, 0.0),
        "PSFC (surface pressure)": (0.05, 0.02),
        "MUB (base dry mass)": (0.2, 0.05),
        "MU+MUB (dry column mass)": (0.05, 0.02),
        "PB (base pressure)": (0.2, 0.03),
        "T_INIT (base theta-300)": (3.0e-4, 1.0e-4),
        "PHB (base geopotential)": (0.5, 0.1),
        "T (theta-300, dry)": (0.01, 1.0e-4),
        "THM (moist theta-300)": (0.01, 2.0e-4),
        "QVAPOR": (1.0e-5, 2.0e-7),
        "U": (0.01, 1.0e-4),
        "V": (0.01, 1.0e-4),
        "P+PB (total pressure)": (200.0, 2.0),
        "PH+PHB (total geopotential)": (6.0, 0.5),
    }
    for field, (max_abs, rms) in bounds.items():
        row = rows[field]
        assert row["max_abs"] <= max_abs, (
            f"{field}: max_abs {row['max_abs']:.6g} exceeds {max_abs}")
        assert row["rms"] <= rms, (
            f"{field}: rms {row['rms']:.6g} exceeds {rms}")


@_pairs
def test_the_wrong_valid_time_blows_past_every_bound():
    """NEGATIVE CONTROL for the whole oracle.

    Score the 18:00 forcing against the 17:00 wrfinput.  Same grid, same
    physics, same file layout -- only the weather is an hour wrong.  An
    oracle that still passed would be measuring nothing.
    """
    if not FORCING_2021_NEXT.is_file():
        pytest.skip(f"second frame absent: {FORCING_2021_NEXT}")
    report = MD.run_differential(FORCING_2021_NEXT, WRFINPUT_2021)
    rows = {row["field"]: row for row in report["rows"]}
    assert rows["PSFC (surface pressure)"]["rms"] > 10.0
    assert rows["T (theta-300, dry)"]["rms"] > 0.5
    assert rows["U"]["rms"] > 0.5
    assert rows["QVAPOR"]["rms"] > 1.0e-5


@_pairs
def test_displacing_the_terrain_moves_the_surface_pressure_hydrostatically(
        tmp_path):
    """NEGATIVE CONTROL with a PREDICTED magnitude, not merely a change.

    Raise the target terrain by 10 m and the sfcprs2 adjustment must lower
    the surface pressure by rho*g*dz ~ 100 Pa.  A detector that fired on any
    perturbation would not distinguish a defect from a rounding change.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    moved = tmp_path / "moved.nc"
    shutil.copy(FORCING_2021, moved)
    ds = netCDF4.Dataset(str(moved), "a")
    ds.variables["HGT_M"][0] = np.asarray(ds.variables["HGT_M"][0]) + 10.0
    ds.close()
    rows = {row["field"]: row
            for row in MD.run_differential(moved, WRFINPUT_2021)["rows"]}
    assert 9.9 < rows["HGT (terrain)"]["rms"] < 10.1
    assert 70.0 < rows["PSFC (surface pressure)"]["rms"] < 140.0


@_pairs
def test_uniform_tanh_alternatives_differ_from_real_exes_eta_ladder():
    """The generic coordinate constructor does not select a WRF generator."""
    grid = MD.read_wrfinput_grid(WRFINPUT_2021)
    eta = MD.eta_ladder_comparison(grid["znw"])
    assert eta["uniform_max_abs_deta"] > 0.25
    assert eta["best_tanh_max_abs_deta"] > 0.25
    assert eta["real_thickness_ratio"] > 10.0
    with pytest.raises(TypeError):
        make_vertical_coord(eta["nz"], auto_levels_opt=2)
