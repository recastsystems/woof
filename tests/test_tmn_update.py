"""The surface-state contract and the public-source oracle for TMN update."""
import dataclasses
from pathlib import Path
import numpy as np
import pytest


def test_tmn_kernel_uses_shared_loader():
    from woof.core.kernels import module_source
    source = module_source("tmn_update")
    assert "tmn_columns" in source
    assert "tmn_counters" in source
    assert "__fdiv_rn" in source


def test_tmn_configuration():
    from woof.config import RunConfig, validate_run_config
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=8000.,
                    dt=30., run_seconds=120., tmn_update=1, lagday=150)
    assert cfg.tmn_update == 1
    with pytest.raises(ValueError, match="lagday=150"):
        validate_run_config(dataclasses.replace(cfg, lagday=149))


def test_tmn_pin_retired():
    from woof.namelist_import import PHYSICS_PINS
    assert "tmn_update" not in {key for key, _, _ in PHYSICS_PINS}


def test_tmn_restart_metadata():
    from woof.wrf_namelist_registry import TMN_UPDATE_STATE
    assert set(TMN_UPDATE_STATE) == {"tmn", "tlag", "tyr", "tyra", "tdly", "nyear", "nday"}
    assert TMN_UPDATE_STATE["tlag"]["dimensions"] == ("lagday", "y", "x")
    assert all(row["restart"] for row in TMN_UPDATE_STATE.values())


def test_tmn_namelist_admitted_and_validated(tmp_path):
    from test_namelist_import import _import_with
    text, _ = _import_with(tmp_path, extra_physics=" tmn_update=1, lagday=150,\n")
    assert "tmn_update = 1" in text and "lagday = 150" in text
    for value in ("149", "0", "1.5", ".true."):
        with pytest.raises(ValueError, match="lagday"):
            _import_with(tmp_path, extra_physics=f" tmn_update=1, lagday={value},\n")
    for value in ("2", "1.5", ".true."):
        with pytest.raises(ValueError, match="tmn_update"):
            _import_with(tmp_path, extra_physics=f" tmn_update={value},\n")


def test_tmn_preflight_inventory():
    from woof.config import RunConfig
    from woof.core.preflight import physics_array_shapes
    from woof.wrf_namelist_registry import TMN_UPDATE_STATE
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=8000.,
                    dt=30., run_seconds=120., tmn_update=1, lagday=150)
    shapes = physics_array_shapes(cfg)
    assert shapes["fields/tlag"] == (150, 8, 8)
    assert shapes["fields/nyear"] == shapes["fields/nday"] == (1,)
    assert all("fields/"+name in shapes for name in TMN_UPDATE_STATE)


def test_tmn_disabled_restart_identity_stays_absent():
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo
    from dataclasses import replace
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=8000.,
                    dt=30., run_seconds=120.)
    assert "tmn_update" not in configuration_echo(cfg)
    assert "lagday" not in configuration_echo(cfg)
    enabled = configuration_echo(replace(cfg, tmn_update=1))
    assert enabled["tmn_update"] == 1 and enabled["lagday"] == 150


def test_tmn_only_domain_dispatches_physics():
    from woof.config import RunConfig
    from woof.core.physics_inventory import physics_enabled, physics_driver_required
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=8000.,
                    dt=30., run_seconds=120., tmn_update=1)
    assert physics_enabled(cfg) and physics_driver_required(cfg)


def test_tmn_year_counter_is_checked_by_the_real_health_policy():
    from woof.core.health import gpu_integer_policy, field_from_array, validate_fields_cpu, GPU_INTEGER_EXCLUSIONS
    assert gpu_integer_policy("surface.nyear", np.int32) is not None
    assert "surface.nyear" not in GPU_INTEGER_EXCLUSIONS
    legal = field_from_array("surface.nyear", np.array([0, 1, 366, 1000], np.int32))
    assert validate_fields_cpu([legal]).ok
    invalid = field_from_array("surface.nyear", np.array([-1], np.int32))
    report = validate_fields_cpu([invalid])
    assert not report.ok and report.first_bad_field == "surface.nyear"
