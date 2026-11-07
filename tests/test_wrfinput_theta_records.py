"""File-backed dry and moist temperature boundary contracts."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from woof.ingest import wrfinput as wi
from woof.ingest.lateral_bc import evaluate_boundary_side
from test_analyzed_scalar_boundaries import _cfg, _input, _read, _boundary


@pytest.mark.parametrize("moist,selected", [(False, 0), (True, 0), (True, 1)])
def test_file_temperature_forcing_matches_its_declared_representation(tmp_path, monkeypatch, moist, selected):
    if selected:
        monkeypatch.setattr(wi, "WRF_EXACT", True)
    cfg = _cfg()
    path = _input(tmp_path / "input", cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.USE_THETA_M = int(moist)
    initial = _read(path, cfg)
    cfg = SimpleNamespace(**{**vars(cfg), "use_theta_m": selected})
    boundary = _boundary(tmp_path / "boundary", initial, cfg)
    ratio = np.float32(461.6) / np.float32(287.0)
    with netCDF4.Dataset(boundary, "a") as dataset:
        for suffix in ("XS", "XE", "YS", "YE"):
            theta = dataset[f"T_B{suffix}"]
            vapour = dataset[f"QVAPOR_B{suffix}"]
            start = np.asarray(theta[0], np.float32)
            if moist:
                # The fixture has total dry mass 13 and unit map factors.
                dry = start / np.float32(13.) + np.float32(300.)
                factor = np.float32(1.) + ratio * (
                    np.asarray(vapour[0], np.float32) / np.float32(13.))
                start = (factor * dry - np.float32(300.)) * np.float32(13.)
            theta[0] = start
            theta[1] = start + np.float32(60.) * np.float32(.125)
            dataset[f"T_BT{suffix}"][:] = np.float32(.125)
            first_q = np.asarray(vapour[0], np.float32)
            vapour[1] = first_q + np.float32(60.) * np.float32(.00002)
            dataset[f"QVAPOR_BT{suffix}"][:] = np.float32(.00002)
    result = wi.read_wrfbdy(boundary, restored=initial, run_seconds=60,
                            forcing_interval_seconds=60, cfg=cfg)
    with netCDF4.Dataset(boundary) as dataset:
        for side, suffix in (("west", "XS"), ("east", "XE"),
                             ("south", "YS"), ("north", "YE")):
            order = (1, 2, 0) if side in ("west", "east") else (1, 0, 2)
            read = lambda name: np.asarray(dataset[name + suffix][0], float)
            A, Ad, Q, Qd = read("T_B"), read("T_BT"), read("QVAPOR_B"), read("QVAPOR_BT")
            for seconds in (0., 12., 30., 60.):
                encoded = A + seconds * Ad
                expected = (13. * ((encoded / 13. + 300.) /
                            (1. + float(ratio) * (Q + seconds * Qd) / 13.) - 300.)
                            if moist and not selected else encoded)
                actual, _ = evaluate_boundary_side(
                    getattr(result.intervals[0].fields["theta"], side), seconds)
                np.testing.assert_allclose(actual, expected.transpose(order),
                                           rtol=2e-13, atol=2e-12)



def test_selected_moist_input_retains_native_prognostic_and_dry_history(tmp_path):
    cfg = _cfg()
    path = _input(tmp_path / "input", cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.USE_THETA_M = 1
        dry = np.asarray(dataset["T"][:], dtype=np.float32)
        moist = (dry + np.float32(300.)) * np.float32(1.02) - np.float32(300.)
        dataset.createVariable("THM", "f4", dataset["T"].dimensions)[:] = moist
    cfg = SimpleNamespace(**{**vars(cfg), "use_theta_m": 1})
    restored = _read(path, cfg)
    np.testing.assert_array_equal(restored.raw["THM"], moist[0])
    np.testing.assert_array_equal(restored.raw["T"], dry[0])

def test_moist_flag_with_dry_temperature_records_is_not_silently_accepted(tmp_path):
    cfg = _cfg()
    path = _input(tmp_path / "input", cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.USE_THETA_M = 1
    initial = _read(path, cfg)
    boundary = _boundary(tmp_path / "boundary", initial, cfg)
    with pytest.raises(ValueError, match="does not match initial"):
        wi.read_wrfbdy(boundary, restored=initial, run_seconds=60,
                       forcing_interval_seconds=60, cfg=cfg)


def test_single_interval_character_end_time_reaches_the_boundary_reader(tmp_path):
    cfg = _cfg()
    initial = _read(_input(tmp_path / "input", cfg), cfg)
    original = _boundary(tmp_path / "two-records", initial, cfg)
    single = tmp_path / "single-record"
    with netCDF4.Dataset(original) as source, netCDF4.Dataset(single, "w") as target:
        for name, dimension in source.dimensions.items():
            target.createDimension(name, 1 if name == "Time" else len(dimension))
        target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        for name, variable in source.variables.items():
            target.createVariable(name, variable.datatype, variable.dimensions)[:] = variable[:1]
        name = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
        target.createVariable(name, "S1", ("Time", "DateStrLen"))[:] = np.frombuffer(
            b"2026-08-25_18:01:00", dtype="S1").reshape(1, 19)
    result = wi.read_wrfbdy(single, restored=initial, run_seconds=60,
                            forcing_interval_seconds=60, cfg=cfg)
    assert len(result.intervals) == 1
    assert result.intervals[0].end_seconds == 60


# ---------------------------------------------------------------------------
# Stock real.exe pairs.  tests/data/real_em_461_theta_seam_west_strip.npz is
# a west strip cut from two WRF 4.6.1 real.exe runs of one met_em set that
# differ only in use_theta_m; the README beside it states the provenance.
# WRF's own files: T is dry theta-300 under both settings, THM is the
# prognostic (moist under use_theta_m=1), and T_BXS couples THM.
# ---------------------------------------------------------------------------
_STRIP = Path(__file__).parent / "data" / "real_em_461_theta_seam_west_strip.npz"
_WIDTH = 5
_INPUT_NAMES = ("T", "QVAPOR", "MU", "MUB", "C1H", "C2H", "C1F", "C2F",
                "MAPFAC_M", "MAPFAC_U", "MAPFAC_V")


def _stock_strip(moist):
    """The reader's seam inputs for one arm of the stock pair, west side only."""
    data = np.load(_STRIP)
    arm = "moist" if moist else "dry"
    # WRF writes side tables as (width, z, y); the reader's tables are (z, y, width).
    side = lambda name: np.ascontiguousarray(np.transpose(data[name], (1, 2, 0)))
    mu_side = lambda name: np.ascontiguousarray(np.transpose(data[name], (1, 0))[None])
    tables = {"theta": {"west": (side(f"T_BXS_{arm}"), side(f"T_BTXS_{arm}"))},
              "qv": {"west": (side("QVAPOR_BXS"), side("QVAPOR_BTXS"))},
              "mu": {"west": (mu_side("MU_BXS"), mu_side("MU_BTXS"))}}
    restored = SimpleNamespace(
        path=Path("wrfinput_d01"), raw={name: data[name] for name in _INPUT_NAMES},
        global_attributes={"USE_THETA_M": int(moist)})
    layouts = {key: wi._WRFBDY_FIELDS[key] for key in ("theta", "mu", "qv")}
    return restored, tables, layouts, data


def _stock_coupling(data, field):
    """real_em.F:872 couple(): field * (C1H*MU + (C1H*MUB + C2H)), FP32."""
    c1h = data["C1H"][:, None, None]
    c2h = data["C2H"][:, None, None]
    weight = np.asarray(c1h * data["MU"][None] + (c1h * data["MUB"][None] + c2h), np.float32)
    return np.asarray(field * weight, np.float32)


def test_stock_real_exe_writes_dry_T_and_moist_THM_under_use_theta_m_1():
    data = np.load(_STRIP)
    ratio = np.float32(461.6) / np.float32(287.0)
    moist = np.asarray((np.float32(1.) + ratio * data["QVAPOR"])
                       * (data["T"] + np.float32(300.)) - np.float32(300.), np.float32)
    assert np.array_equal(data["THM_moist"], moist)
    assert np.abs(data["THM_moist"] - data["T"]).max() > 1.0
    # The boundary the same run wrote couples THM, bit for bit.
    assert np.array_equal(data["T_BXS_moist"],
                          np.transpose(_stock_coupling(data, data["THM_moist"]), (2, 0, 1)))
    assert np.array_equal(data["T_BXS_dry"],
                          np.transpose(_stock_coupling(data, data["T"]), (2, 0, 1)))


@pytest.mark.parametrize("moist", [False, True])
def test_stock_real_exe_pair_passes_the_seam_at_both_settings(moist):
    restored, tables, layouts, _ = _stock_strip(moist)
    wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)


def test_comparing_the_stock_moist_boundary_with_dry_T_fails_and_says_so():
    # The suggested change ("compare the boundary directly against T") is
    # what USE_THETA_M=0 does; on a stock use_theta_m=1 pair it refuses.
    restored, tables, layouts, _ = _stock_strip(True)
    restored.global_attributes["USE_THETA_M"] = 0
    with pytest.raises(ValueError, match="does not match initial") as refusal:
        wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)
    text = str(refusal.value)
    assert "USE_THETA_M=0 on both files" in text
    assert "3920 of 3920 points differ" in text
    assert "equals the MOIST coupling of this wrfinput T at every point" in text


def test_a_dry_boundary_under_the_moist_flag_names_the_writers_that_produce_it():
    restored, tables, layouts, _ = _stock_strip(False)
    restored.global_attributes["USE_THETA_M"] = 1
    with pytest.raises(ValueError, match="does not match initial") as refusal:
        wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)
    text = str(refusal.value)
    assert "USE_THETA_M=1 on both files" in text
    assert "converted to moist theta with wrfinput QVAPOR" in text
    assert "equals the DRY coupling of this wrfinput T at every point" in text
    assert "WRF 3.7 to 3.9.1.1 real.exe always does" in text


def test_a_boundary_from_another_real_exe_run_reports_the_size_of_the_mismatch():
    restored, tables, layouts, _ = _stock_strip(True)
    value, tendency = tables["theta"]["west"]
    tables["theta"]["west"] = (np.asarray(value * np.float32(1.002), np.float32), tendency)
    with pytest.raises(ValueError, match="does not match initial") as refusal:
        wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)
    text = str(refusal.value)
    assert "3920 of 3920 points differ" in text
    assert "0.002 of the largest value" in text
    assert "not the DRY coupling of this T either" in text
    assert "use the pair produced by the same real.exe run" in text


# ---------------------------------------------------------------------------
# A real.exe whose compiler contracts multiply-adds (GCC with FMA in its
# -march) rounds the moist-theta conversion and couple() once per fused
# pair.  Float64 holds every float32 product exactly, so a float64 product
# rounded once to float32 is that build's single final rounding.
# ---------------------------------------------------------------------------
def _fma_moist_theta(data):
    """(t_2 + T0) * (1. + (R_v/R_d) * qv) - T0 with 1. + a*qv and s*f - T0 fused."""
    ratio = np.float64(np.float32(461.6) / np.float32(287.0))
    s = np.asarray(data["T"] + np.float32(300.), np.float32).astype(np.float64)
    f = np.asarray(1.0 + ratio * data["QVAPOR"].astype(np.float64), np.float32)
    return np.asarray(s * f.astype(np.float64) - 300.0, np.float32)


def _fma_coupling(data, field):
    """couple(): field * (C1H*MU + (C1H*MUB + C2H)), both multiply-adds fused."""
    c1h = data["C1H"][:, None, None].astype(np.float64)
    c2h = data["C2H"][:, None, None].astype(np.float64)
    inner = np.asarray(c1h * data["MUB"][None] + c2h, np.float32).astype(np.float64)
    weight = np.asarray(c1h * data["MU"][None] + inner, np.float32).astype(np.float64)
    return np.asarray(field.astype(np.float64) * weight, np.float32)


def _fma_pair(moist_boundary=True, moist_flag=True):
    """The stock strip with its T_BXS and QVAPOR_BXS as a contracting build writes them."""
    restored, tables, layouts, data = _stock_strip(moist_flag)
    theta = _fma_moist_theta(data) if moist_boundary else data["T"]
    tables["theta"]["west"] = (_fma_coupling(data, theta), tables["theta"]["west"][1])
    tables["qv"]["west"] = (_fma_coupling(data, data["QVAPOR"]), tables["qv"]["west"][1])
    return restored, tables, layouts, data


def test_an_fma_contracting_real_exe_pair_passes_the_seam():
    # A user's WRF 4.6.1 real.exe pair (3 km, 74 levels) was refused this
    # way: "24752 of 57860 points differ, max |difference| 4.6875".
    restored, tables, layouts, data = _fma_pair()
    boundary = tables["theta"]["west"][0]
    ours = _stock_coupling(data, wi._moist_theta_from_dry(data["T"], data["QVAPOR"]))
    over = np.abs(boundary.astype(np.float64) - ours) > 4.0 * np.abs(np.spacing(ours)) + 1e-6
    # The fixture is the defect: a per-element budget of the coupled theta-300
    # refuses it, because the reconstruction's rounding sits on the full theta.
    assert np.count_nonzero(over) > 100
    wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)


def test_the_moist_theta_bound_is_a_few_spacings_of_the_full_theta():
    data = np.load(_STRIP)
    bound = wi._moist_theta_build_bound(data["T"], data["QVAPOR"])
    ours = wi._moist_theta_from_dry(data["T"], data["QVAPOR"]).astype(np.float64)
    assert np.all(np.abs(_fma_moist_theta(data) - ours) <= bound)
    full = np.abs(np.spacing((ours + 300.0).astype(np.float32))).astype(np.float64)
    # Scale-aware, not a floor: two to six spacings of theta_m, wherever
    # theta_m - 300 sits.
    assert np.all(bound >= 2.0 * full) and np.all(bound <= 6.0 * full)
    near_300 = np.abs(ours) < 4.0
    assert near_300.any()
    assert np.all(bound[near_300] > 16.0 * np.abs(np.spacing(ours[near_300].astype(np.float32))))


def test_a_dry_fma_boundary_under_the_moist_flag_still_refuses_and_says_so():
    restored, tables, layouts, _ = _fma_pair(moist_boundary=False)
    with pytest.raises(ValueError, match="does not match initial") as refusal:
        wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)
    text = str(refusal.value)
    assert "USE_THETA_M=1 on both files" in text
    assert "equals the DRY coupling of this wrfinput T at every point" in text


def test_a_moist_fma_boundary_under_the_dry_flag_is_named_moist():
    restored, tables, layouts, _ = _fma_pair(moist_flag=False)
    with pytest.raises(ValueError, match="does not match initial") as refusal:
        wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)
    text = str(refusal.value)
    assert "USE_THETA_M=0 on both files" in text
    assert "equals the MOIST coupling of this wrfinput T at every point" in text


def test_the_next_forcing_time_boundary_still_refuses():
    # The strip's forcing interval is six hours; its tendency carries the
    # first record to the second, which is not this initial state.
    restored, tables, layouts, _ = _fma_pair()
    value, tendency = tables["theta"]["west"]
    later = np.asarray(value + np.float32(21600.) * tendency, np.float32)
    tables["theta"]["west"] = (later, tendency)
    with pytest.raises(ValueError, match="does not match initial") as refusal:
        wi._check_initial_boundary_pair(restored, tables, _WIDTH, layouts=layouts)
    text = str(refusal.value)
    assert "not the DRY coupling of this T either" in text
    assert "use the pair produced by the same real.exe run" in text


def test_the_wrf_input_doors_carry_every_field_the_tree_runner_reads():
    """run_prepared_tree serves the prepared, wrfinput and met_em doors, so
    every ``inputs.<name>`` it reads must be a field of the WRF-input inputs
    class too: a stock real.exe pair completed its simulation and then died
    on ``physics_profile_assertion`` before the report was written."""
    import dataclasses
    from woof import prepared_domain_tree_forecast as tree
    from woof.wrfinput_forecast import WrfTreeInputs
    function = next(node for node in ast.walk(ast.parse(inspect.getsource(tree)))
                    if isinstance(node, ast.FunctionDef) and node.name == "run_prepared_tree")
    read = {node.attr for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "inputs"}
    fields = {field.name for field in dataclasses.fields(WrfTreeInputs)}
    assert read <= fields, sorted(read - fields)
