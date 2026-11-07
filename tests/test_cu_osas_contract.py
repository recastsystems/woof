"""CPU checks for the OSAS front door and the committed oracle streams."""
from pathlib import Path
import numpy as np
from woof.core import column_cumulus_r8 as cc

FIXTURES = Path(__file__).parent / "data" / "oracles" / "osas"

def test_fixture_contract_covers_every_column_output():
    for grid in ("k44", "k29", "k44-default", "k29-default"):
        with np.load(FIXTURES / f"osas-{grid}.npz") as fx:
            assert int(fx["im"]) * int(fx["jm"]) >= 64
            assert int(fx["ncall"]) >= 6
            for n in range(int(fx["ncall"])):
                for name in cc.LEVEL_OUTPUTS + cc.SURFACE_OUTPUTS:
                    assert f"c{n}_out_{name}" in fx

def test_osasscheme_is_admitted_with_both_arithmetic_arms():
    from woof.config import CU_SCHEMES, RunConfig, validate_run_config
    from woof.physics_registry import physics_registry
    assert 95 in CU_SCHEMES
    cfg = RunConfig(nx=24, ny=24, nz=44, dx=12000., dy=12000., ztop=20000., dt=60., run_seconds=120., cu_physics=95, moist=True)
    validate_run_config(cfg)
    row = physics_registry()["components"]["cumulus"]["options"]["old-gfs-sas"]
    assert row["implemented"]
    assert cc.COLUMN_CUMULUS_SCHEMES[95].strict_flags == 1

def test_scale_aware_sas_door_names_the_osasscheme_replacement():
    import pytest
    from woof.config import RunConfig, validate_run_config
    with pytest.raises(ValueError, match="New SAS"):
        validate_run_config(RunConfig(nx=24, ny=24, nz=44, dx=12000., dy=12000., ztop=20000., dt=60., run_seconds=120., cu_physics=4, moist=True))

def test_default_counter_draw_is_reproducible_without_restart_state():
    a = cc.random_input_host(96, step=3)
    b = cc.random_input_host(96, step=3)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != cc.random_input_host(96, step=4).tobytes()

def test_osasscheme_public_namelist_preserves_positive_cudt(tmp_path):
    from test_namelist_import import _import_with, _load
    text, report = _import_with(
        tmp_path, inp_filter=lambda text: text.replace(
            " cu_physics = 1, 0,", " cu_physics = 95, 0,"))
    exp = _load(tmp_path, text)
    assert exp.root.run.cu_physics == 95
    assert exp.root.run.cudt_minutes == 5.0
    assert "cu_physics = 95" in text

def test_osasscheme_short_columns_are_refused_before_a_card_is_opened():
    import pytest
    from woof.config import RunConfig, validate_run_config
    cfg = RunConfig(nx=24, ny=24, nz=2, dx=12000., dy=12000.,
                    ztop=20000., dt=60., run_seconds=120.,
                    cu_physics=95, moist=True)
    with pytest.raises(ValueError, match="nz must be >= 4") :
        validate_run_config(cfg)

def test_native_osas_restart_restores_cloud_bounds_and_held_rain(monkeypatch):
    from woof import netcdf_bridge
    from woof.ingest import wrfrst
    from test_wrf_warm_start import _restart_dataset, _Variable, _expected, _cfg
    dataset = _restart_dataset(CU_PHYSICS=95)
    for name, value in (("SHDAVG", 30.), ("CUTOP", 20.), ("CUBOT", 3.),
                        ("HTOP", 0.), ("HBOT", 0.),
                        ("PRATEC", .001), ("RAINCV", .06)):
        dataset.variables[name] = _Variable(
            name, ("south_north", "west_east"), np.full((3,4), value, np.float32))
    monkeypatch.setattr(netcdf_bridge, "open_dataset", lambda path: dataset)
    restored = wrfrst.read_wrfrst("wrfrst_d01", expected_dimensions=_expected(),
                                cfg=_cfg(cu_physics=95))
    assert restored.warm_start.dispositions["SHDAVG"].startswith("setup:")
    assert "SHDAVG" not in restored.raw
    assert "HTOP" not in restored.raw and "HBOT" not in restored.raw
    for name in ("CUTOP", "CUBOT", "PRATEC", "RAINCV"):
        np.testing.assert_array_equal(restored.raw[name], dataset.variables[name]._data[0])
    assert wrfrst.scratch_restore_plan("cu_pratec", 95) == ("fill", "PRATEC")
    assert wrfrst.scratch_restore_plan("cu_raincv", 95) == ("fill", "RAINCV")

def test_osas_warm_overlay_uses_cumulus_bounds_instead_of_radiation_bounds(monkeypatch):
    from types import SimpleNamespace
    from woof.ingest import wrfrst
    from woof.io import restart
    from test_wrf_warm_start import _cfg
    targets = {"cumulus/last_htop": np.zeros((3, 4), np.float32),
               "cumulus/last_hbot": np.zeros((3, 4), np.float32)}
    raw = {name: np.full((3, 4), value, np.float32)
           for name, value in (("CUTOP", 20.), ("CUBOT", 3.),
                               ("HTOP", 0.), ("HBOT", 0.))}
    monkeypatch.setattr(restart, "state_manifest", lambda state: {})
    monkeypatch.setattr(restart, "_scratch_manifest", lambda state: {})
    monkeypatch.setattr(restart, "_driver_manifest", lambda driver: targets)
    monkeypatch.setattr(wrfrst, "_copy_into", np.copyto)
    monkeypatch.setattr(wrfrst, "_recompose_held_tendencies", lambda *args: None)
    monkeypatch.setattr(wrfrst, "_warm_start_driver_counters", lambda *args: None)
    monkeypatch.setattr(wrfrst, "_record_eos_checks", lambda *args: None)
    report = wrfrst.apply_warm_start_physics(
        SimpleNamespace(), SimpleNamespace(cumulus_callable=None),
        SimpleNamespace(raw=raw), _cfg(cu_physics=95))
    np.testing.assert_array_equal(targets["cumulus/last_htop"], raw["CUTOP"])
    np.testing.assert_array_equal(targets["cumulus/last_hbot"], raw["CUBOT"])
    assert report.filled == {"cumulus/last_htop": "CUTOP",
                             "cumulus/last_hbot": "CUBOT"}
