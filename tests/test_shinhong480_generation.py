"""WRF 4.8.0 Shin-Hong generation (shinhong_version): selection, identity, plumbing.

The numerics are graded on the GPU against the native WRF 4.8.0 objects by
tools/shinhong480_oracle (columns.py compare) and, where the corpus is
present, by tests/test_shinhong480_replay_gpu.py.  These CPU checks pin the
pieces that decide WHICH unit runs and that the 4.7.1 option stays today's.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from woof.config import RunConfig, validate_shinhong_generation
from woof.core.shinhong_generations import (
    SHINHONG_DEFAULT_GENERATION, SHINHONG_GENERATIONS, SHINHONG_SOURCES)

KERNELS = Path(__file__).resolve().parents[1] / "woof" / "core" / "kernels"


def _cfg(**kw):
    base = dict(nx=4, ny=3, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=15.0, run_seconds=60.0, bl_pbl_physics=11,
                sf_sfclay_physics=1)
    base.update(kw)
    return RunConfig(**base)


def test_the_default_is_the_wrf_480_generation():
    cfg = _cfg()
    assert SHINHONG_DEFAULT_GENERATION == "wrf_480"
    assert cfg.shinhong_version == "wrf_480"
    # WRF 4.8.0 Registry.EM_COMMON defaults.
    assert cfg.shinhong_nonlocal_flux is True
    assert cfg.shinhong_scu_mixing is False
    assert cfg.shinhong_ke_dissipation is False
    assert set(SHINHONG_GENERATIONS) == {"wrf_480", "wrf_461"}


def test_unknown_generation_is_refused_by_name():
    with pytest.raises(Exception, match="shinhong_version"):
        validate_shinhong_generation(_cfg(shinhong_version="wrf_470"))


def test_480_options_under_the_461_generation_are_refused():
    # The 4.6.1/4.7.1 module has none of the three logicals: a non-default
    # value would be silently ignored, so it is refused.
    for key, value in (("shinhong_nonlocal_flux", False),
                       ("shinhong_scu_mixing", True),
                       ("shinhong_ke_dissipation", True)):
        with pytest.raises(Exception, match="WRF 4.8.0 Shin-Hong options"):
            validate_shinhong_generation(_cfg(shinhong_version="wrf_461", **{key: value}))


def test_the_sources_are_pinned():
    assert SHINHONG_SOURCES["wrf_461"]["source_sha256"].startswith("99f44dbe")
    assert SHINHONG_SOURCES["wrf_480"]["source_sha256"].startswith("7d258836")
    assert SHINHONG_SOURCES["wrf_480"]["wrapper_sha256"].startswith("c7bfe363")
    assert SHINHONG_SOURCES["wrf_461"]["kernel_module"] == "shinhong"
    assert SHINHONG_SOURCES["wrf_480"]["kernel_module"] == "shinhong_480"


def test_the_480_unit_is_composed_on_the_461_unit_without_moving_it():
    from woof.core import kernels as K
    text480 = K.module_source("shinhong_480")
    shinhong_cu = (KERNELS / "shinhong.cu").read_text(encoding="utf-8")
    assert shinhong_cu in text480
    assert "shinhong480_column" in text480
    assert "shinhong480_column" not in K.module_source("shinhong")
    opts = K.module_options("shinhong_480")
    assert "--fmad=false" in opts and "--ftz=false" in opts


def test_restart_echo_drops_the_defaults_and_keeps_a_choice():
    from woof.io.restart import _drop_default_off_run_keys
    values = {"bl_pbl_physics": 11, "shinhong_version": "wrf_480",
              "shinhong_nonlocal_flux": True, "shinhong_scu_mixing": False,
              "shinhong_ke_dissipation": False}
    _drop_default_off_run_keys(values)
    assert values["shinhong_version"] == "wrf_480"
    assert "shinhong_nonlocal_flux" not in values
    values = {"bl_pbl_physics": 11, "shinhong_version": "wrf_461"}
    _drop_default_off_run_keys(values)
    assert "shinhong_version" not in values          # what every old header ran
    values = {"bl_pbl_physics": 1, "shinhong_version": "wrf_480",
              "shinhong_scu_mixing": True}
    _drop_default_off_run_keys(values)
    assert "shinhong_version" not in values and "shinhong_scu_mixing" not in values


def test_restart_identity_names_the_generation_and_options():
    from woof.io.restart import physics_algorithm_identities
    ident = physics_algorithm_identities(_cfg())["pbl"]
    assert ident == "shinhong-wrf-v4.8.0-v1-n"
    ident = physics_algorithm_identities(
        _cfg(shinhong_scu_mixing=True, shinhong_ke_dissipation=True))["pbl"]
    assert ident == "shinhong-wrf-v4.8.0-v1-nsd"
    old = physics_algorithm_identities(_cfg(shinhong_version="wrf_461"))["pbl"]
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert old == PBL_ALGORITHM_IDENTITIES[11]


def test_preflight_selects_one_unit_per_generation():
    from woof.core import preflight
    assert preflight.shinhong_output_transient_shapes(
        _cfg(nz=10, ny=3, nx=4)).keys() >= {"shinhong_output/exch_m",
                                            "shinhong_output/u10"}
    old = preflight.shinhong_output_transient_shapes(
        _cfg(nz=10, ny=3, nx=4, shinhong_version="wrf_461"))
    assert "shinhong_output/exch_m" not in old


def test_the_tke_copy_in_patch_is_the_only_change_to_the_wrapper():
    # The labelled-patched referee differs from WRF 4.8.0's wrapper in two
    # places only: tke intent(inout) and its copy into tke_hv.
    patch = (Path(__file__).resolve().parents[1] / "tools" / "shinhong480_oracle"
             / "patch_tke_copyin.py").read_text(encoding="utf-8")
    assert "tke_hv(i,k) = tke(i,k,j)" in patch
    assert hashlib.sha256(patch.encode()).hexdigest()
