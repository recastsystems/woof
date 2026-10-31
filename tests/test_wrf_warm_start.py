"""Warm start from a WRF-written restart file, and the doors it opens.

CPU only.  The restart reader is exercised through a stand-in for the Rust
NetCDF bridge (the dataset protocol it returns), so these run on any
install; the GPU half is proved by the matched restart runs.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof import netcdf_bridge
from woof.ingest import wrfrst
from woof.namelist_import import import_namelists


# ---------------------------------------------------------------------------
# The namelist half: what NCAR's benchmark namelists need from the importer.
# ---------------------------------------------------------------------------

WPS = """\
&share
 wrf_core = 'ARW', max_dom = 1,
 start_date = '2019-11-26_12:00:00',
/
&geogrid
 parent_id = 1, parent_grid_ratio = 1, i_parent_start = 1, j_parent_start = 1,
 e_we = 61, e_sn = 51, dx = 12000, dy = 12000, map_proj = 'lambert',
 ref_lat = 40.0, ref_lon = -98.0, truelat1 = 30.0, truelat2 = 60.0,
 stand_lon = -98.0,
/
"""

INPUT = """\
&time_control
 run_hours = 1,
 start_year = 2019, start_month = 11, start_day = 26, start_hour = 12,
 history_interval_m = 60,
/
&domains
 time_step = {dt},
 max_dom = 1, e_we = 61, e_sn = 51, e_vert = 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000, dx = 12000, dy = 12000,
 grid_id = 1, parent_id = 0, i_parent_start = 0, j_parent_start = 0,
 parent_grid_ratio = 1, parent_time_step_ratio = 1,
{domains}
/
&physics
{physics}
/
&dynamics
 use_theta_m = 0, diff_opt = 1, km_opt = 4,
{dynamics}
/
&bdy_control
 spec_bdy_width = 5, spec_zone = 1, relax_zone = 4, specified = .true.,
/
"""

CONUS_BUT_CU16 = """\
 physics_suite = 'conus',
 mp_physics = -1, ra_lw_physics = -1, ra_sw_physics = -1,
 sf_sfclay_physics = -1, sf_surface_physics = -1, bl_pbl_physics = -1,
 cu_physics = 16, radt = {radt}, bldt = 0, cudt = 0,
"""


def _import(tmp_path, *, physics, dt=60, radt=12, domains="", dynamics=""):
    (tmp_path / "namelist.wps").write_text(WPS)
    (tmp_path / "namelist.input").write_text(INPUT.format(
        dt=dt, domains=domains, dynamics=dynamics,
        physics=physics.format(radt=radt)))
    return import_namelists(tmp_path / "namelist.wps",
                            tmp_path / "namelist.input",
                            rrtmg_variant="rrtmg_legacy")


def _defaults(report):
    return {row.key: row for row in report.namelist_defaults}


def test_conus_suite_resolves_minus_one_the_way_wrf_does(tmp_path):
    text, report = _import(tmp_path, physics=CONUS_BUT_CU16)
    rows = _defaults(report)
    for key, value in (("mp_physics", 8), ("ra_lw_physics", 4),
                       ("ra_sw_physics", 4), ("bl_pbl_physics", 2),
                       ("sf_sfclay_physics", 2), ("sf_surface_physics", 2)):
        assert rows[key].value == value
        assert "module_check_a_mundo.F:3116" in rows[key].reason
    # A selector written explicitly is the user's, exactly as in WRF.
    assert "cu_physics" not in rows
    assert "cu_physics = 16" in text
    assert "mp_physics = 8" in text
    assert any(item.key == "physics_suite" for item in report.dropped)


def test_conus_suite_cumulus_is_wrfs_tiedtke(tmp_path):
    physics = CONUS_BUT_CU16.replace("cu_physics = 16", "cu_physics = -1")
    try:
        text, _ = _import(tmp_path, physics=physics)
    except ValueError as error:
        # Until the original Tiedtke port is admitted the importer refuses
        # it by number; either way the suite resolved -1 to WRF's 6.
        assert "cu_physics = 6" in str(error)
    else:
        assert "cu_physics = 6" in text


def test_tropical_suite_resolves_its_own_row(tmp_path):
    physics = CONUS_BUT_CU16.replace("'conus'", "'tropical'").replace(
        "cu_physics = 16", "cu_physics = -1")
    text, report = _import(tmp_path, physics=physics)
    rows = _defaults(report)
    assert {key: rows[key].value for key in (
        "mp_physics", "cu_physics", "bl_pbl_physics",
        "sf_sfclay_physics")} == {"mp_physics": 6, "cu_physics": 16,
                                  "bl_pbl_physics": 1,
                                  "sf_sfclay_physics": 91}


def test_unknown_suite_is_refused_like_wrf(tmp_path):
    physics = CONUS_BUT_CU16.replace("'conus'", "'midlatitude'")
    with pytest.raises(ValueError, match="not a WRF physics suite"):
        _import(tmp_path, physics=physics)


def test_suite_none_leaves_the_sentinel_refusal(tmp_path):
    physics = CONUS_BUT_CU16.replace("'conus'", "'none'")
    with pytest.raises(ValueError, match="-1"):
        _import(tmp_path, physics=physics)


def test_radiation_cadence_takes_wrfs_whole_steps(tmp_path):
    text, report = _import(tmp_path, physics=CONUS_BUT_CU16, dt=72, radt=15)
    assert "radt = 15.6" in text
    (row,) = [item for item in report.fixed if item.key == "radt[1]"]
    assert row.fixed_value == 15.6
    assert "NINT = 13" in row.reason


def test_radiation_cadence_with_no_decimal_spelling_keeps_wrfs_value(tmp_path):
    """radt = 12.5 at 40 s is 18.75 steps; wrf.exe runs 19 (12.666... min,
    no exact decimal).  The importer refused it; it now keeps radt = 12.5,
    which the loader and clock run on the same 19 steps."""
    from fractions import Fraction

    from woof.core.clock import resolve_clock
    from woof.experiment import load_experiment

    text, report = _import(tmp_path, physics=CONUS_BUT_CU16, dt=40, radt=12.5)
    assert "radt = 12.5" in text
    assert not [item for item in report.fixed if item.key == "radt[1]"]
    config = tmp_path / "imported.toml"
    config.write_text(text, encoding="utf-8")
    exp = load_experiment(config)
    assert exp.dt_exact(1) == Fraction(40)
    assert resolve_clock(exp).spec(1).stepra == 19


def test_commensurate_cadence_is_unchanged(tmp_path):
    text, report = _import(tmp_path, physics=CONUS_BUT_CU16, dt=72, radt=12)
    assert "radt = 12" in text
    assert not [item for item in report.fixed if item.key == "radt[1]"]


def test_inert_benchmark_keys_are_consumed_with_reasons(tmp_path):
    _, report = _import(
        tmp_path, physics=CONUS_BUT_CU16 + " num_soil_layers = 1,\n",
        domains=" perturb_input = .false.,",
        dynamics=" use_baseparam_fr_nml = .t.,")
    dropped = {item.key: item.reason for item in report.dropped}
    assert "frame/module_io.F" in dropped["perturb_input"]
    assert "base-state" in dropped["use_baseparam_fr_nml"]
    (soil,) = [item for item in report.fixed if item.key == "num_soil_layers"]
    assert soil.fixed_value == 4
    assert "module_check_a_mundo.F:3552-3555" in soil.reason


def test_perturbed_input_is_refused(tmp_path):
    with pytest.raises(ValueError, match="perturb_real"):
        _import(tmp_path, physics=CONUS_BUT_CU16,
                domains=" perturb_input = .true.,")


# ---------------------------------------------------------------------------
# The restart window and file discovery.
# ---------------------------------------------------------------------------

def test_restart_window_rewrites_the_time_control():
    from woof.fortran_namelist import parse_namelist_text
    from woof.wrfinput_door import restart_window_namelist

    text = ("&time_control\n run_hours = 1,\n start_hour = 23,\n"
            " restart = .true.,\n restart_interval = 66000,\n"
            " override_restart_timers = .true.,\n/\n&domains\n max_dom = 1,\n/\n")
    out = restart_window_namelist(text, start=datetime(2019, 11, 26, 12),
                                  end=datetime(2019, 11, 27, 0), max_dom=1)
    control = parse_namelist_text(out)["time_control"]
    assert control["start_hour"] == [12] and control["end_day"] == [27]
    assert control["restart_interval"] == [66000]
    for key in ("restart", "run_hours", "override_restart_timers"):
        assert key not in control


def test_namelist_restart_request_reads_wrfs_end(tmp_path):
    from woof.wrfinput_door import namelist_restart_request

    path = tmp_path / "namelist.input"
    path.write_text("&time_control\n run_hours = 1,\n start_year = 2019,\n"
                    " start_month = 11,\n start_day = 26,\n start_hour = 23,\n"
                    " end_year = 2030,\n restart = .true.,\n/\n")
    restart, start, end = namelist_restart_request(path)
    assert restart and start == datetime(2019, 11, 26, 23)
    assert end == datetime(2019, 11, 27, 0)  # run_* wins over end_*


def test_wrfs_restart_name():
    assert wrfrst.restart_file_name(1, datetime(2019, 11, 26, 23)) == (
        "wrfrst_d01_2019-11-26_23:00:00")


def test_wps_synthesis_spells_the_root_parent_start_as_wps_does():
    from woof.wrfinput_door import synthesize_wps_namelist

    attrs = {"MAP_PROJ": 1, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
             "STAND_LON": -98.0, "CEN_LAT": 40.0, "CEN_LON": -98.0,
             "DX": 12000.0, "DY": 12000.0, "GRID_ID": 1, "PARENT_ID": 0,
             "I_PARENT_START": 0, "J_PARENT_START": 0, "PARENT_GRID_RATIO": 1}
    item = SimpleNamespace(path=Path("wrfrst_d01"), global_attributes=attrs,
                           nx=10, ny=8, start_date="2019-11-26_12:00:00")
    text = synthesize_wps_namelist({1: item})
    assert " i_parent_start    = 1," in text
    assert " j_parent_start    = 1," in text


# ---------------------------------------------------------------------------
# The restart reader, through a stand-in for the bridge's dataset protocol.
# ---------------------------------------------------------------------------

DIMS = {"Time": 1, "west_east": 4, "west_east_stag": 5, "south_north": 3,
        "south_north_stag": 4, "bottom_top": 2, "bottom_top_stag": 3,
        "soil_layers_stag": 4, "DateStrLen": 19}


class _Variable:
    def __init__(self, name, dimensions, data):
        self.name, self.dimensions = name, ("Time",) + tuple(dimensions)
        self._data = np.asarray(data)[None]

    def __getitem__(self, key):
        return self._data[key]

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size


class _Dataset:
    def __init__(self, variables, attributes):
        self.variables = variables
        self.dimensions = {name: range(size) for name, size in DIMS.items()}
        self.attributes = attributes
        self.decoder_runs = []

    def prefetch(self, names):
        """The bridge's batch decode: one decoder run for ``names``."""
        self.decoder_runs.append(list(names))
        return 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ncattrs(self):
        return tuple(self.attributes)

    def getncattr(self, name):
        return self.attributes[name]


def _restart_dataset(**changes):
    """A complete little Thompson/Noah/MYJ restart, by the wrfinput rules."""
    from woof.ingest.wrfinput import (REQUIRED_WRFINPUT, WRFINPUT_DIMENSIONS,
                                       THOMPSON_NUMBER_WRFINPUT,
                                       BASE_MOISTURE_WRFINPUT, ICE_MASS_WRFINPUT)

    sizes = {name: size for name, size in DIMS.items()}
    variables = {}

    def add(name, dims, value=1.0):
        shape = tuple(sizes[dim] for dim in dims)
        variables[name] = _Variable(name, dims, np.full(shape, value, np.float32))

    renames = {v: k for k, v in wrfrst.PROGNOSTIC_RENAMES.items()}
    for name in (set(REQUIRED_WRFINPUT) | {"SEAICE", "ALBBCK", "LAI", "P_TOP"}
                 | set(BASE_MOISTURE_WRFINPUT) | set(ICE_MASS_WRFINPUT)
                 | set(THOMPSON_NUMBER_WRFINPUT)) - {"T", "QNSNOW", "QNGRAUPEL"}:
        dims = WRFINPUT_DIMENSIONS[name]
        add(renames.get(name, name), dims, 0.01 if name.startswith("Q") else 1.0)
    add("THM_2", WRFINPUT_DIMENSIONS["T"], 10.0)
    add("T", WRFINPUT_DIMENSIONS["T"], 9.0)
    add("U_1", WRFINPUT_DIMENSIONS["U"], 5.0)   # a time level: never read
    for name, value in (("ITIMESTEP", 550), ("XTIME", 660.0), ("DTBC", 7200.0)):
        variables[name] = _Variable(name, (), np.asarray(value))
    times = np.frombuffer(b"2019-11-26_23:00:00", dtype="S1")
    variables["Times"] = _Variable("Times", ("DateStrLen",), times)
    attributes = {"FLAG_RESTART": 1, "SIMULATION_START_DATE": "2019-11-26_12:00:00",
                  "START_DATE": "2019-11-26_12:00:00", "DT": 72.0, "GRID_ID": 1,
                  "MP_PHYSICS": 8, "SF_SURFACE_PHYSICS": 2, "BL_PBL_PHYSICS": 2,
                  "SF_SFCLAY_PHYSICS": 2, "CU_PHYSICS": 6, "USE_THETA_M": 1}
    for name, value in changes.items():
        if isinstance(value, _Variable):
            variables[name] = value
        else:
            attributes[name] = value
    return _Dataset(variables, attributes)


def _cfg(**overrides):
    base = dict(nx=4, ny=3, nz=2, moist=True, mp_physics=8, cu_physics=16,
                bl_pbl_physics=2, sf_sfclay_physics=2, sf_surface_physics=2,
                sf_urban_physics=0, num_soil_layers=4, ifire=0, cycling=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _expected():
    return {"west_east": 4, "west_east_stag": 5, "south_north": 3,
            "south_north_stag": 4, "bottom_top": 2, "bottom_top_stag": 3,
            "soil_layers_stag": 4}


@pytest.fixture
def fake_bridge(monkeypatch):
    holder = {}
    monkeypatch.setattr(netcdf_bridge, "open_dataset", lambda path: holder["ds"])
    return holder


def test_reader_restores_dry_theta_from_moist_prognostic(fake_bridge):
    fake_bridge["ds"] = _restart_dataset()
    restored = wrfrst.read_wrfrst("wrfrst_d01", expected_dimensions=_expected(),
                                  cfg=_cfg())
    rv = np.float32(461.6) / np.float32(287.0)
    want = (np.float32(310.0) / (np.float32(1.0) + rv * np.float32(0.01))
            - np.float32(300.0))
    assert np.allclose(restored.raw["T"], want, rtol=0, atol=1e-5)
    assert "U" in restored.raw and "U_2" not in restored.raw
    assert "U_1" not in restored.raw
    assert restored.warm_start.clock.itimestep == 550
    assert restored.warm_start.clock.elapsed_seconds == 39600.0
    assert restored.warm_start.dispositions["U_1"].startswith("time_level")


def test_restart_variables_decode_in_batches_not_one_run_per_variable(fake_bridge):
    # NCAR's 12 km restart has 575 variables, and the importer started one
    # rw_netcdf process per variable it read (345 of them).  Every variable
    # the importer reads is decoded before its first read, in read order,
    # one decoder run per batch.
    fake_bridge["ds"] = dataset = _restart_dataset()
    wrfrst.read_wrfrst("wrfrst_d01", expected_dimensions=_expected(), cfg=_cfg())
    read = [name for name in dataset.variables if name != "Times"
            and (wrfrst._disposition(name) is None
                 or wrfrst._disposition(name).kind == "read")]
    assert "U_2" in read and "U_1" not in read
    assert dataset.decoder_runs == [read]


def test_restart_decode_batches_stop_at_the_byte_cap(fake_bridge):
    dataset = _restart_dataset()
    names = ["U_2", "V_2", "MU_2"]
    decode = netcdf_bridge.BatchedDecode(
        dataset, names, batch_bytes=8 * (dataset.variables["U_2"].size
                                         + dataset.variables["V_2"].size))
    for name in names:
        decode.ready(name)
        decode.ready(name)
    assert dataset.decoder_runs == [["U_2", "V_2"], ["MU_2"]]


def test_reader_refuses_a_file_wrf_did_not_stamp_as_a_restart(fake_bridge):
    fake_bridge["ds"] = _restart_dataset(FLAG_RESTART=0)
    with pytest.raises(ValueError, match="not a WRF restart"):
        wrfrst.read_wrfrst("wrfout", expected_dimensions=_expected(), cfg=_cfg())


def test_reader_refuses_an_adaptive_step_clock(fake_bridge):
    fake_bridge["ds"] = _restart_dataset(
        ITIMESTEP=_Variable("ITIMESTEP", (), np.asarray(549)))
    with pytest.raises(ValueError, match="adaptive time step"):
        wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(), cfg=_cfg())


def test_reader_refuses_nonzero_unclassified_state(fake_bridge):
    fake_bridge["ds"] = _restart_dataset(
        QKE_NEW=_Variable("QKE_NEW", ("south_north", "west_east"),
                          np.ones((3, 4), np.float32)))
    with pytest.raises(ValueError, match="QKE_NEW"):
        wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(), cfg=_cfg())


def test_reader_accepts_and_names_zero_unclassified_variables(fake_bridge):
    fake_bridge["ds"] = _restart_dataset(
        NEW_FLAG=_Variable("NEW_FLAG", ("south_north", "west_east"),
                           np.zeros((3, 4), np.float32)))
    restored = wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(),
                                  cfg=_cfg())
    assert "identically zero" in restored.warm_start.dispositions["NEW_FLAG"]


def test_reader_refuses_state_of_a_selected_unmapped_scheme(fake_bridge):
    fake_bridge["ds"] = _restart_dataset(
        TR_URB=_Variable("TR_URB", ("south_north", "west_east"),
                         np.ones((3, 4), np.float32)))
    with pytest.raises(ValueError, match="urban canopy"):
        wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(),
                           cfg=_cfg(sf_urban_physics=1))


def test_reader_refuses_a_cut_down_restart(fake_bridge):
    dataset = _restart_dataset()
    del dataset.variables["MU_2"]
    fake_bridge["ds"] = dataset
    with pytest.raises(ValueError, match="MU"):
        wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(), cfg=_cfg())


def test_every_variable_of_ncars_benchmark_restart_has_a_disposition():
    path = Path(__file__).parent / "data" / "wrfrst" / "ncar-v44-conus12km-variables.txt"
    names = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert len(names) == 575
    missing = [name for name in names
               if name != "Times" and name not in wrfrst.WRFRST_DISPOSITIONS]
    assert missing == []
    assert {rule.kind for rule in wrfrst.WRFRST_DISPOSITIONS.values()} <= (
        wrfrst.DISPOSITION_KINDS)


@pytest.mark.parametrize("written,run,admitted", [
    (6, 16, True), (16, 6, True), (6, 3, True), (1, 16, False), (16, 1, False)])
def test_cumulus_may_change_only_between_stateless_schemes(written, run, admitted):
    files = {"MP_PHYSICS": 8, "BL_PBL_PHYSICS": 2, "SF_SFCLAY_PHYSICS": 2,
             "SF_SURFACE_PHYSICS": 2, "CU_PHYSICS": written}
    cfg = _cfg(cu_physics=run)
    if admitted:
        notes = wrfrst.check_restart_physics(files, cfg, path="wrfrst")
        assert notes and f"{written} -> {run}" in notes[0]
    else:
        with pytest.raises(ValueError, match="CU_PHYSICS"):
            wrfrst.check_restart_physics(files, cfg, path="wrfrst")


@pytest.mark.parametrize("cu_physics", [3, 6, 16])
def test_held_rain_starts_cold_for_schemes_that_hold_none(cu_physics):
    """WRF's PRATEC/RAINCV/NCA are last-call diagnostics outside Kain-Fritsch.

    Restoring PRATEC armed woof's cumulus clock, which adds PRATEC * DT to
    RAINC every step: NCAR's CONUS 12 km restart hour then rained twice
    WRF's convective rain under cu6 (and cu16).
    """
    for name, cold in wrfrst.HELD_RAIN_COLD_VALUES.items():
        plan = wrfrst.scratch_restore_plan(name, cu_physics)
        assert plan[0] == "cold" and plan[1] == cold, (name, plan)
    assert wrfrst.scratch_restore_plan("cu_rainc", cu_physics) == ("fill", "RAINC")


def test_kain_fritsch_keeps_its_held_rain():
    for name, source in (("cu_nca", "NCA"), ("cu_pratec", "PRATEC"),
                         ("cu_raincv", "RAINCV")):
        assert wrfrst.scratch_restore_plan(name, 1) == ("fill", source)


def test_correct_mode_settled_cloud_total_starts_at_the_restart():
    """2.9.0 refused every default (correct-mode) Thompson warm start from a
    wrf.exe restart, NCAR's benchmark restart leg among them, because WRF
    keeps no settled-cloud total to fill CLOUDNC from."""
    for name in ("mp_cloudnc", "mp_cloudncv"):
        plan = wrfrst.scratch_restore_plan(name, 6)
        assert plan[0] == "cold" and plan[1] == 0.0, plan


#: The carriers NCAR's documented benchmark warm start named when it exited 2
#: in apply_warm_start_physics (RULING-8, 2026-10-17).
_EXACT_CHAIN_ROWS = (
    "acgrdflx", "achfx", "aclhf", "coszen", "p_hyd", "refd_max", "sstsk",
    "swnorm", "xicem",
    *(f"{side}{flux}{level}{clear}" for side in ("sw", "lw")
      for flux in ("up", "dn") for level in ("t", "b") for clear in ("", "c")),
    *(f"ac{side}{flux}{level}{clear}" for side in ("sw", "lw")
      for flux in ("up", "dn") for level in ("t", "b") for clear in ("", "c")),
)


def test_exact_chain_history_rows_are_read_from_ncars_restart():
    """Every such row NCAR's restart carries is read, as wrf.exe reads it."""
    path = Path(__file__).parent / "data" / "wrfrst" / "ncar-v44-conus12km-variables.txt"
    names = {line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")}
    for name in _EXACT_CHAIN_ROWS:
        wrf_name = name.upper()
        if wrf_name == "REFD_MAX":
            assert wrf_name not in names  # WRF never restart-writes it
            continue
        assert wrf_name in names, wrf_name
        assert wrfrst.WRFRST_DISPOSITIONS[wrf_name].kind == "read", wrf_name
        assert wrfrst.fields_restore_plan(name, names) == ("fill", wrf_name)


def test_exact_chain_history_rows_take_wrfs_rule_when_the_restart_lacks_them():
    for name in _EXACT_CHAIN_ROWS:
        plan = wrfrst.fields_restore_plan(name, ())
        assert plan is not None, name
        if name == "p_hyd":
            assert plan[:2] == ("copy", "state/p")
        elif name == "xicem":
            assert plan[:2] == ("copy", "fields/xice")
        else:
            assert plan[:2] == ("cold", 0.0), (name, plan)
    assert wrfrst.fields_restore_plan("not_a_carrier", ()) is None


def test_stateful_physics_must_match_the_restart():
    files = {"MP_PHYSICS": 10, "BL_PBL_PHYSICS": 2, "SF_SFCLAY_PHYSICS": 2,
             "SF_SURFACE_PHYSICS": 2, "CU_PHYSICS": 16}
    with pytest.raises(ValueError, match="MP_PHYSICS"):
        wrfrst.check_restart_physics(files, _cfg(), path="wrfrst")


@pytest.mark.parametrize("itimestep,stepra,last", [
    (550, 13, 547), (546, 13, 534), (547, 13, 547), (1, 13, 1), (5, 1, 5)])
def test_last_radiation_step_follows_wrfs_phase(itimestep, stepra, last):
    assert wrfrst._last_radiation_step(itimestep, stepra) == last


def test_pbl_carriers_take_the_mass_levels_of_wrfs_staggered_allocation():
    raw = {"TKE_PBL": np.arange(3 * 2, dtype=np.float32).reshape(3, 1, 2)}
    out = wrfrst._host_field(raw, "TKE_PBL", (2, 1, 2), label="fields/tke_myj")
    assert out.shape == (2, 1, 2) and out[0, 0, 0] == 0.0
    with pytest.raises(ValueError, match="cannot fill"):
        wrfrst._host_field(raw, "TKE_PBL", (1, 1, 2), label="fields/tke_myj")


def test_roundoff_negative_species_are_zeroed_and_counted(fake_bridge):
    dataset = _restart_dataset()
    qc = np.full((2, 3, 4), 1e-4, np.float32)
    qc[0, 2, 3] = -4.6e-23
    dataset.variables["QCLOUD"] = _Variable(
        "QCLOUD", ("bottom_top", "south_north", "west_east"), qc)
    fake_bridge["ds"] = dataset
    restored = wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(),
                                  cfg=_cfg())
    assert restored.raw["QCLOUD"].min() == 0.0
    row = restored.warm_start.checks["roundoff_negative_moisture_set_to_zero"]
    assert row["QCLOUD"]["cells"] == 1


def test_negative_species_beyond_roundoff_are_refused(fake_bridge):
    dataset = _restart_dataset()
    qc = np.full((2, 3, 4), 1e-4, np.float32)
    qc[0, 2, 3] = -1e-6
    dataset.variables["QCLOUD"] = _Variable(
        "QCLOUD", ("bottom_top", "south_north", "west_east"), qc)
    fake_bridge["ds"] = dataset
    with pytest.raises(ValueError, match="beyond roundoff"):
        wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(), cfg=_cfg())


def _counters(cu_physics, cudt_minutes, written_stepcu, *, bl_pbl_physics=2,
              radiation_active=True):
    driver = SimpleNamespace(radt_minutes=10.0, radiation_active=radiation_active,
                             fields={}, carriers=None, microphysics_updates=0)
    cfg = SimpleNamespace(dt=15.0, cu_physics=cu_physics, cudt_minutes=cudt_minutes,
                          bldt=0.0, bl_pbl_physics=bl_pbl_physics,
                          sf_surface_physics=2)
    raw = {"STEPRA": np.array([40]), "STEPCU": np.array([written_stepcu]),
           "STEPBL": np.array([1])}
    restored = SimpleNamespace(
        raw=raw, warm_start=SimpleNamespace(clock=SimpleNamespace(itimestep=1200)))
    report = SimpleNamespace(checks={})
    wrfrst._warm_start_driver_counters(driver, restored, cfg, report)
    return report.checks


def test_a_cumulus_cadence_is_not_compared_when_no_cumulus_runs(monkeypatch):
    """Named breakage: NCAR's CONUS 2.5 km restart (cu_physics = 0, cudt = 0,
    WRF's STEPCU = 1) was refused after its whole read, because the omitted
    cudt_minutes took RunConfig's 5.0 (20 steps of 15 s)."""
    from woof.core import radiation_carriers
    monkeypatch.setattr(radiation_carriers, "consumer_carriers", lambda _sf: ())
    checks = _counters(0, 5.0, 1)
    assert checks["stepcu_wrf_vs_gpuwm"][:2] == [1, 20]
    assert "not compared" in checks["stepcu_wrf_vs_gpuwm"][2]
    assert checks["stepra_wrf_vs_gpuwm"] == [40, 40]


def test_a_running_cumulus_on_another_cadence_is_still_refused(monkeypatch):
    from woof.core import radiation_carriers
    monkeypatch.setattr(radiation_carriers, "consumer_carriers", lambda _sf: ())
    with pytest.raises(ValueError, match="STEPCU = 1 steps but this run's cadence is 20"):
        _counters(1, 5.0, 1)


class _CountingVariable(_Variable):
    """A field that records every read, so a test can see what was read."""

    reads: list = []

    def __getitem__(self, key):
        _CountingVariable.reads.append(self.name)
        return super().__getitem__(key)


def _cadence_restart(stepcu):
    dataset = _restart_dataset()
    for name, variable in list(dataset.variables.items()):
        if len(variable.dimensions) > 2:
            dataset.variables[name] = _CountingVariable(
                name, variable.dimensions[1:], variable._data[0])
    for name, value in (("STEPRA", 40), ("STEPCU", stepcu), ("STEPBL", 1)):
        dataset.variables[name] = _Variable(name, (), np.asarray(value, np.int32))
    _CountingVariable.reads = []
    return dataset


def _cadence_cfg(cu_physics):
    return _cfg(dt=15.0, cu_physics=cu_physics, cudt_minutes=5.0, bldt=0.0,
                radt=0.0, radt_minutes=10.0, ra_physics=4)


def test_a_wrong_cadence_is_refused_before_any_field_is_read(fake_bridge):
    """Named breakage: NCAR's CONUS 2.5 km restart was refused on STEPCU only
    after 44 minutes of reading its 4D fields.  The cadences are scalars, so
    a refusal now comes before the first field is read."""
    fake_bridge["ds"] = _cadence_restart(stepcu=1)
    with pytest.raises(ValueError, match="STEPCU = 1 steps but this run's cadence is 20"):
        wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(),
                           cfg=_cadence_cfg(cu_physics=1))
    assert _CountingVariable.reads == []


def test_the_early_cadence_check_skips_a_scheme_that_does_not_run(fake_bridge):
    fake_bridge["ds"] = _cadence_restart(stepcu=1)
    restored = wrfrst.read_wrfrst("wrfrst", expected_dimensions=_expected(),
                                  cfg=_cadence_cfg(cu_physics=0))
    checks = restored.warm_start.checks
    assert checks["stepcu_wrf_vs_gpuwm"][:2] == [1, 20]
    assert "not compared" in checks["stepcu_wrf_vs_gpuwm"][2]
    assert checks["stepra_wrf_vs_gpuwm"] == [40, 40]
    assert _CountingVariable.reads


@pytest.mark.parametrize("minutes,dt", [(0.0, 15.0), (5.0, 15.0), (10.0, 72.0),
                                        (12.0, 72.0), (30.0, 72.0), (0.5, 7.0),
                                        (7.0, 18.0), (1.0, 40.0)])
def test_the_readers_cadence_steps_match_the_physics_driver(minutes, dt):
    physics = pytest.importorskip("woof.core.physics")
    assert wrfrst.wrf_interval_steps(minutes, dt) == \
        physics._physics_interval_steps(minutes, dt)
