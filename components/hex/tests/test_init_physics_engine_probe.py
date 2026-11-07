"""The forecast-time physics init names an engine that predates suite-aware init.

THE BREAKAGE (golden 0.4.0 round 2, install gate on the gpuwm 2.9.0
candidate b32d6fa48): `--physics config_convection_scheme=off` (Noah-MP land)
stopped at the door with an unnamed "Command [... rw_mpas_init
--aerosol-status ...] returned non-zero exit status 1".  An rw_mpas_init
without the suite-aware entries answers the flag with its usage.  Now the
door probes the usage: aerosols and a soil-level change are refused by name,
and the Noah-MP shoreline soil-category check alone is skipped and said so.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

OLD_USAGE = "rw_mpas_init: --met was not given, and it has no default\n\nusage: rw_mpas_init --met MET ...\n"
NEW_USAGE = OLD_USAGE + "  --aerosol-status INIT.nc\n  --augment-init INIT.nc --scalar-names A,B\n"


def _init(path, soil_levels=4, scheme="mp_wsm6"):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", 2)
        ds.createDimension("nSoilLevels", soil_levels)
        ds.setncattr("config_start_time", "2026-10-11_12:00:00")
        from woof.hex.species_row import species_row_for_scheme
        for name in species_row_for_scheme(scheme).names():
            ds.createVariable(name, "f4", ("Time", "nCells"))[0] = np.zeros(2, np.float32)
    return path


def _engine(monkeypatch, tmp_path, usage):
    from woof.hex import init_door, init_physics

    engine = tmp_path / "rw_mpas_init"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append([str(a) for a in argv])
        if argv[1] == "--help":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=usage)
        if argv[1] == "--aerosol-status" and "--aerosol-status" in usage:
            return subprocess.CompletedProcess(argv, 0, stdout='{"analysis_pair": false, "soil_category_mismatches": 0}', stderr="")
        # an older binary answers any unknown entry with its usage, exit 1
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=OLD_USAGE)

    monkeypatch.setattr(init_door, "resolve_engine", lambda explicit=None: engine)
    monkeypatch.setattr(init_physics.subprocess, "run", fake_run)
    return calls


def _physics(lsm, mp="mp_wsm6"):
    return {"selectors": {"config_lsm_scheme": lsm, "config_microp_scheme": mp}}


def test_engine_only_thompson_needs_no_legacy_carried_metadata(monkeypatch, tmp_path):
    from woof.hex import init_physics

    # A complete Thompson init already carries every scalar; incomplete inits now augment them.
    init = _init(tmp_path / "init.nc", scheme="mp_thompson")
    calls = _engine(monkeypatch, tmp_path, NEW_USAGE)
    monkeypatch.setattr(init_physics, "_synthetic_source",
                        lambda path: pytest.fail("non-aerosol Thompson asked for climatology"))
    assert init_physics.prepare_forecast_init(
        init, tmp_path / "out.nc", _physics("sf_noahmp", "mp_thompson"), None) == (init, None)
    assert [c[1] for c in calls] == ["--aerosol-status"]


def test_noahmp_on_an_older_engine_runs_without_the_category_check_and_says_so(monkeypatch, tmp_path, capsys):
    from woof.hex.init_physics import prepare_forecast_init

    init = _init(tmp_path / "init.nc")
    calls = _engine(monkeypatch, tmp_path, OLD_USAGE)
    assert prepare_forecast_init(init, tmp_path / "out.nc", _physics("sf_noahmp"), None) == (init, None)
    assert [c[1] for c in calls] == ["--aerosol-status", "--help"]
    assert "shoreline soil-category check not run" in capsys.readouterr().out
    assert not (tmp_path / "out.nc").exists()


def test_aerosol_microphysics_on_an_older_engine_is_refused_by_name(monkeypatch, tmp_path):
    from woof.hex.init_physics import prepare_forecast_init

    init = _init(tmp_path / "init.nc")
    _engine(monkeypatch, tmp_path, OLD_USAGE)
    with pytest.raises(ValueError, match="predates it .*--aerosol-status"):
        prepare_forecast_init(init, tmp_path / "out.nc", _physics("sf_noahmp", "mp_tempo"), None)


def test_a_soil_level_change_on_an_older_engine_is_refused_by_name(monkeypatch, tmp_path):
    from woof.hex.init_physics import prepare_forecast_init

    init = _init(tmp_path / "init.nc")
    _engine(monkeypatch, tmp_path, OLD_USAGE)
    with pytest.raises(ValueError, match="ruc with 9 soil levels needs gpuwm 2.9.0"):
        prepare_forecast_init(init, tmp_path / "out.nc", _physics("sf_ruc"), None)


def test_a_suite_aware_engine_is_asked_for_the_status(monkeypatch, tmp_path):
    from woof.hex.init_physics import prepare_forecast_init

    # A complete WSM6 init isolates the status probe from missing-scalar augmentation.
    init = _init(tmp_path / "init.nc")
    calls = _engine(monkeypatch, tmp_path, NEW_USAGE)
    assert prepare_forecast_init(init, tmp_path / "out.nc", _physics("sf_noahmp"), None) == (init, None)
    assert [c[1] for c in calls] == ["--aerosol-status"]


def test_a_suite_aware_engine_that_fails_reports_its_own_error(monkeypatch, tmp_path):
    from woof.hex import init_physics

    calls = _engine(monkeypatch, tmp_path, NEW_USAGE)
    real = init_physics.subprocess.run

    def failing(argv, **kwargs):
        if argv[1] == "--aerosol-status":
            calls.append([str(a) for a in argv])
            return subprocess.CompletedProcess(argv, 3, stdout="", stderr="cannot open init")
        return real(argv, **kwargs)

    monkeypatch.setattr(init_physics.subprocess, "run", failing)
    with pytest.raises(subprocess.CalledProcessError):
        init_physics.aerosol_status(tmp_path / "rw_mpas_init", _init(tmp_path / "init.nc"))
