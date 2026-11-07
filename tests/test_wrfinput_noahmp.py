"""Real-input surface records retain the selected cold-start contract."""

from dataclasses import replace
from types import SimpleNamespace
import sys

import netCDF4
import numpy as np
import pytest

from woof.ingest import wrfinput as wi
from test_wrfinput_physics_fields import _cfg, _dimensions, _input, _surface_values


def _noahmp_input(tmp_path, *, extra=True):
    cfg = _cfg(sf_surface_physics=4, num_soil_layers=4)
    values = _surface_values(cfg)
    path = _input(tmp_path / "input", cfg, values)
    # Registry/registry.noahmp names: prognostic placeholders, flux output
    # and an inactive crop field coexist in an ordinary real.exe input.
    if extra:
        with netCDF4.Dataset(path, "a") as dataset:
            for name in ("ALBOLD", "CANICE", "TV", "ISNOW", "APAR", "RUNSB", "PONDING", "GRAIN"):
                dataset.createVariable(name, "i4" if name == "ISNOW" else "f4",
                                       ("Time", "south_north", "west_east"))[:] = 0.
    return path, cfg, values


def test_standard_noahmp_records_do_not_refuse_the_selected_scheme(tmp_path):
    path, cfg, values = _noahmp_input(tmp_path)
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                require_complete=False)
    for name in ("SMOIS", "TSLB", "SH2O", "SNOW", "TSK"):
        np.testing.assert_array_equal(restored.raw[name], values[name])
    for name in ("ALBOLD", "CANICE", "TV", "ISNOW", "APAR", "RUNSB", "PONDING", "GRAIN"):
        assert name not in restored.raw
        assert name in restored.surface_input_dispositions


@pytest.mark.parametrize("soil", [2, 3, 4])
@pytest.mark.parametrize("bad_geometry", [False, True])
def test_selected_soil_records_have_native_geometry_and_reach_restore(tmp_path, soil, bad_geometry):
    path, cfg, _ = _noahmp_input(tmp_path)
    cfg = replace(cfg, opt_soil=soil)
    with netCDF4.Dataset(path, "a") as dataset:
        if soil == 2:
            dims = ("west_east", "south_north") if bad_geometry else ("south_north", "west_east")
            for k in range(1, 5):
                dataset.createVariable(f"SOILCL{k}", "f4", ("Time", *dims))[:] = k + 2.
        elif soil == 3:
            dataset.createDimension("soil_composition_layers_stag", 7 if bad_geometry else 8)
            dataset.createVariable("SOILCOMP", "f4", (
                "Time", "soil_composition_layers_stag", "south_north", "west_east"))[:] = 41.
        else:
            from woof.core.noahmp_soil_options import DIRECT_SOIL_INPUTS
            dims = (("soil_layers_stag", "west_east", "south_north") if bad_geometry
                    else ("soil_layers_stag", "south_north", "west_east"))
            for index, name in enumerate(DIRECT_SOIL_INPUTS):
                dataset.createVariable(name, "f4", ("Time", *dims))[:] = index + .125
    if bad_geometry:
        with pytest.raises(ValueError, match="(SOIL(CL|COMP)|.*_3D).*shape mismatch"):
            wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg), require_complete=False)
    else:
        result = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg), require_complete=False)
        names = ([f"SOILCL{k}" for k in range(1, 5)] if soil == 2 else
                 ["SOILCOMP"] if soil == 3 else DIRECT_SOIL_INPUTS)
        for name in names:
            assert name in result.raw
            assert result.surface_input_dispositions[name] == "consumed by native soil parameter selection"
        if soil == 3:
            assert result.raw["SOILCOMP"].shape == (8, cfg.ny, cfg.nx)
        if soil == 4:
            for index, name in enumerate(names):
                np.testing.assert_array_equal(result.raw[name],
                    np.full((4,cfg.ny,cfg.nx),index+.125,np.float32))


def test_noahmp_names_do_not_hide_a_different_surface_selection(tmp_path):
    path, cfg, _ = _noahmp_input(tmp_path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.SF_SURFACE_PHYSICS = 2
    with pytest.raises(ValueError, match="unmapped WRF variable"):
        wi.read_wrfinput(path, cfg=replace(cfg, sf_surface_physics=2),
                         expected_dimensions=_dimensions(cfg), require_complete=False)


def test_unknown_prognostic_still_requires_a_consumer(tmp_path):
    path, cfg, _ = _noahmp_input(tmp_path, extra=False)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.createVariable("NEW_SURFACE_STATE", "f4",
                               ("Time", "south_north", "west_east"))[:] = 7.
    with pytest.raises(ValueError, match="NEW_SURFACE_STATE"):
        wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                         require_complete=False)


def test_surface_cold_start_corrections_survive_the_generic_restore(tmp_path, monkeypatch):
    path, cfg, values = _noahmp_input(tmp_path, extra=False)
    restored = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                require_complete=False)
    # NOAHMP_INIT partitions frozen water, initializes canopy state and
    # computes leaf area. These outputs differ from its original inputs.
    fields = {"sh2o": np.full(values["SH2O"].shape, .1, np.float32),
              "lai": np.full(values["LAI"].shape, 3., np.float32),
              "albbck": np.full(values["ALBBCK"].shape, .2, np.float32),
              "tvxy": np.full(values["TSK"].shape, 290., np.float32)}
    expected = {name: value.copy() for name, value in fields.items()}
    driver = SimpleNamespace(fields=fields, rainc=None)
    monkeypatch.setitem(sys.modules, "cupy", np)
    # Since 01402fcb1 the door first asks woof.core.physics whether any
    # scheme needs a driver (the re-export of the inventory's predicate),
    # so the stand-in module carries the real predicate beside the fake
    # initializer.
    from woof.core.physics_inventory import physics_driver_required
    monkeypatch.setitem(sys.modules, "woof.core.physics", SimpleNamespace(
        initialize_physics=lambda *args, **kwargs: driver,
        physics_driver_required=physics_driver_required))
    wi.initialize_wrfinput_physics(object(), restored, cfg)
    for name in expected:
        np.testing.assert_array_equal(fields[name], expected[name])


def test_active_crop_option_requires_its_native_geography(tmp_path):
    path, cfg, _ = _noahmp_input(tmp_path)
    with pytest.raises(ValueError, match="CROPTYPE.*HARVEST.*PLANTING.*SEASON_GDD"):
        wi.read_wrfinput(path, cfg=replace(cfg, opt_crop=1),
                         expected_dimensions=_dimensions(cfg), require_complete=False)


@pytest.mark.parametrize("marker", ["restart", "step", "elapsed", "model", "untagged"])
def test_carried_state_cannot_be_discarded_as_cold_placeholders(tmp_path, marker):
    path, cfg, _ = _noahmp_input(tmp_path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["TV"][:] = 291.
        dataset["CANICE"][:] = .4
        if marker == "restart":
            dataset.RESTART = 1
        elif marker == "step":
            dataset.createVariable("ITIMESTEP", "i4", ("Time",))[:] = 10
        elif marker == "elapsed":
            dataset.createVariable("XTIME", "f4", ("Time",))[:] = 10.
        elif marker == "model":
            dataset.TITLE = "OUTPUT FROM WRF V4.6.1 MODEL"
    declared = path.read_bytes()
    with pytest.raises(ValueError, match="original real.exe inputs.*--restart"):
        wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                         require_complete=False)
    assert path.read_bytes() == declared


def test_real_preprocessor_output_retains_its_cold_initialization_contract(tmp_path):
    path, cfg, _ = _noahmp_input(tmp_path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.TITLE = " OUTPUT FROM REAL_EM V4.6.1 PREPROCESSOR"
        dataset["TV"][:] = 291.
        dataset["CANICE"][:] = .4
    result = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                              require_complete=False)
    assert "TV" not in result.raw and "CANICE" not in result.raw
    assert "TV" in result.surface_input_dispositions


@pytest.mark.parametrize("advanced", [False, True])
def test_native_initial_child_uses_the_native_cold_branch(tmp_path, advanced):
    path, cfg, _ = _noahmp_input(tmp_path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.GRID_ID = 2
        dataset.TITLE = "OUTPUT FROM WRF V4.7.1 MODEL"
        dataset.START_DATE = "2024-05-25_18:00:00"
        if "DateStrLen" not in dataset.dimensions:
            dataset.createDimension("DateStrLen", 19)
        if "Times" not in dataset.variables:
            dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamp = "2024-05-25_18:01:00" if advanced else dataset.START_DATE
        dataset["Times"][:] = np.asarray(list(stamp), dtype="S1")[None, :]
        dataset["TV"][:] = 291.
        dataset["CANICE"][:] = .4
    if advanced:
        with pytest.raises(ValueError, match="cannot be discarded"):
            wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                             require_complete=False)
    else:
        result = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                                  require_complete=False)
        assert "TV" not in result.raw and "TV" in result.surface_input_dispositions


def test_untagged_missing_placeholders_can_initialize_from_common_fields(tmp_path):
    path, cfg, _ = _noahmp_input(tmp_path)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["TV"].missing_value = np.float32(9.e20)
        dataset["TV"][:] = np.float32(9.e20)
    result = wi.read_wrfinput(path, cfg=cfg, expected_dimensions=_dimensions(cfg),
                              require_complete=False)
    assert "TV" not in result.raw
    assert "SMOIS" in result.raw and "TSK" in result.raw


def test_warm_identity_is_refused_before_the_supervised_gpu_selection(tmp_path, monkeypatch):
    from test_wrfinput_identity import _file
    from woof import supervisor, wrfinput_door, wrfinput_forecast
    from woof.ingest.wrfinput_identity import read_wrfinput_identity

    path = _file(tmp_path, SF_SURFACE_PHYSICS=4, RESTART=1)
    monkeypatch.setattr(wrfinput_door, "resolve_wrfinput_run",
                        lambda *args, **kwargs: read_wrfinput_identity(path))
    monkeypatch.setattr(supervisor, "select_gpu",
                        lambda *args: pytest.fail("GPU selected before input identity review"))
    with pytest.raises(ValueError, match="WRF restart"):
        wrfinput_forecast.run_wrf_forecast(tmp_path / "source", tmp_path / "out")


@pytest.mark.parametrize("elapsed", [False, True])
def test_native_initial_child_uses_wrf_cold_branch_and_preserves_stepped_refusal(tmp_path,elapsed):
    path,cfg,_=_noahmp_input(tmp_path)
    with netCDF4.Dataset(path,"a") as dataset:
        dataset.TITLE="OUTPUT FROM WRF V4.6.1 MODEL"
        dataset.GRID_ID=2
        dataset.START_DATE="2000-01-24_12:00:00"
        dataset.createDimension("DateStrLen",19)
        dataset.createVariable("Times","S1",("Time","DateStrLen"))[:]=np.frombuffer(b"2000-01-24_12:00:00",dtype="S1").reshape(1,19)
        dataset["TV"][:]=291.
        dataset["CANICE"][:]=.4
        if elapsed:dataset.createVariable("XTIME","f4",("Time",))[:]=10.
    if elapsed:
        with pytest.raises(ValueError,match="original real.exe inputs.*--restart"):
            wi.read_wrfinput(path,cfg=cfg,expected_dimensions=_dimensions(cfg),require_complete=False)
    else:
        result=wi.read_wrfinput(path,cfg=cfg,expected_dimensions=_dimensions(cfg),require_complete=False)
        assert "TV" not in result.raw and "CANICE" not in result.raw
