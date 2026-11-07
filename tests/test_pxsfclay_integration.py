"""The Pleim-Xiu surface layer (sf_sfclay_physics=7): its doors and seams.

No device is opened here.  What is pinned:

* the doors a user reaches the scheme through: the RunConfig key, the WRF
  namelist importer and the registry option;
* WRF's pairing law (PXSFCSCHEME sets isfc=7, phys/module_physics_init.F:
  3153-3155): PBL-off, the UW PBL and SASE admitted; YSU, MYNN and
  Shin-Hong refused by the WRF matrix rows, MYJ by its own pairing law;
* the dispatch row, the checkpoint identity, the published field set and
  the constants the launcher hands the kernel, each held to its authority;
* the kernel keeps strict scalar math behind the proof opt-in and CUDA
  math for default execution, and the oracle fixtures are pinned words.

The card-side grade is tests/test_pxsfclay_wrf461_parity.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from woof.config import PX_SFCLAY_SCHEME, RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=16, ny=16, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=9, sf_sfclay_physics=PX_SFCLAY_SCHEME,
                sf_surface_physics=2)
    base.update(overrides)
    return RunConfig(**base)


def test_the_selector_is_wrfs_pxsfcscheme_number():
    assert PX_SFCLAY_SCHEME == 7


@pytest.mark.parametrize("pbl", [0, 9])
@pytest.mark.parametrize("lsm,soil", [(0, 4), (2, 4), (3, 9), (4, 4)])
def test_admitted_with_every_pbl_and_land_model_that_reads_its_fields(
        pbl, lsm, soil):
    validate_run_config(_cfg(bl_pbl_physics=pbl, sf_surface_physics=lsm,
                             num_soil_layers=soil))


@pytest.mark.parametrize("pbl", [1, 5, 11])
def test_wrf_fatal_pbls_are_refused_with_wrfs_citation(pbl):
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        validate_run_config(_cfg(bl_pbl_physics=pbl))
    message = str(caught.value)
    assert "phys/module_physics_init.F:3153-3155" in message
    assert "isfc=7" in message


def test_myj_is_refused_by_its_own_pairing_law():
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(bl_pbl_physics=2))
    assert "requires sf_sfclay_physics=2" in str(caught.value)


def test_the_wrf_matrix_rows_are_wrfs_isfc_law():
    from woof.wrf461_compatibility import (WRFVerdict,
                                            pbl_surface_layer_verdict)
    assert pbl_surface_layer_verdict(0, 7)[0] is WRFVerdict.LEGAL
    for pbl in (1, 5, 11):
        assert pbl_surface_layer_verdict(pbl, 7)[0] is WRFVerdict.FATAL


def test_dispatch_routes_seven_to_its_own_runner():
    from woof.core.physics import PHYSICS_SLOT_DISPATCH, PhysicsDriver
    assert PHYSICS_SLOT_DISPATCH["sf_sfclay_physics"][7] == "_run_px_sfclay"
    assert callable(getattr(PhysicsDriver, "_run_px_sfclay"))


def test_the_checkpoint_identity_names_the_wrf_version_ported():
    from woof.checkpoint_identity import SURFACE_LAYER_ALGORITHM_IDENTITIES
    assert SURFACE_LAYER_ALGORITHM_IDENTITIES[7] == \
        "pleim-xiu-surface-layer-wrf-v4.6.1-v1"


def test_the_published_set_lives_in_the_shared_surface_fields():
    """Nothing new is allocated: every word is an SFCLAY_OUTPUTS plane,
    and every open-water word has a RUC carrier."""
    from woof.core.physics_inventory import (PX_SFCLAY_SEA,
                                              PX_SFCLAY_STATE, SFCLAY_OUTPUTS)
    from woof.core.ruc_runtime import RUC_FRACTIONAL_SEAICE_FIELDS
    assert set(PX_SFCLAY_STATE) <= set(SFCLAY_OUTPUTS)
    assert {f"{name}_sea" for name in PX_SFCLAY_SEA} <= (
        set(RUC_FRACTIONAL_SEAICE_FIELDS))
    # WRF's PXSFCLAY writes none of these; their consumers are refused.
    assert not {"fm", "fh", "ustm", "t2", "th2", "q2"} & set(PX_SFCLAY_STATE)


def test_the_launcher_constants_are_module_model_constants_words():
    from woof.core.pxsfclay import wrf_constants
    c = wrf_constants()
    assert all(isinstance(v, np.float32) for v in c.values())
    f = np.float32
    assert c["cp"] == f(1004.5)
    assert c["rovcp"] == f(287.0) / f(1004.5)
    assert c["ep1"] == f(f(461.6) / f(287.0)) - f(1.0)
    assert c["ep2"] == f(287.0) / f(461.6)
    assert (c["g"], c["karman"], c["xlv"]) == (f(9.81), f(0.4), f(2.5e6))
    assert (c["svp1"], c["svp2"], c["svp3"], c["svpt0"]) == (
        f(0.6112), f(17.67), f(29.65), f(273.15))


def test_the_kernel_retains_strict_scalar_math_and_fast_cuda_math():
    source = (ROOT / "woof/core/kernels/pxsfclay.cu").read_text(
        encoding="utf-8")
    code = re.sub(r"//[^\n]*", "", source)
    bare = re.sub(r"\b(?:gfk|glibc)_\w+", "", code)
    found = re.findall(
        r"\b(?:__)?(?:expf|logf|powf|atanf|exp|log|pow|atan)\s*\(", bare)
    assert found == []
    for own in ("gfk_exp(", "gfk_log(", "gfk_pow(", "woof_atan_scalar("):
        assert own in code


def test_the_loader_is_the_exact_route():
    from woof.core import pxsfclay
    from woof.core.kernels import module_options
    from woof.wrf_exact import ENABLED
    assert pxsfclay.MODULE_OPTIONS == ("-std=c++17",)
    assert ("--fmad=false" in module_options("pxsfclay")) is ENABLED
    source = pxsfclay.module_source()
    assert "pxsfclay_column" in source and "pxsfclay_seaice_column" in source


def test_the_registry_option_is_the_selector():
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    components = registry["components"]
    option = components["surface_layer"]["options"]["pleim-xiu"]
    assert option["selectors"] == {"sf_sfclay_physics": 7}
    assert option["implemented"] is True
    assert option["constraints"]["requires_components"] == {
        "pbl": ["off", "uw", "sase", "acm2"]}
    for lsm in ("noah", "ruc-lsm", "noah-mp"):
        assert "pleim-xiu" in components["land_surface"]["options"][lsm][
            "constraints"]["requires_components"]["surface_layer"]
    for pbl in ("uw", "sase", "acm2"):
        assert "pleim-xiu" in components["pbl"]["options"][pbl][
            "constraints"]["requires_components"]["surface_layer"]
    for pbl in ("ysu", "mynn", "shinhong", "myj"):
        assert "pleim-xiu" not in components["pbl"]["options"][pbl][
            "constraints"].get("requires_components", {}).get(
                "surface_layer", [])


def test_the_namelist_importer_admits_seven(tmp_path):
    from woof.namelist_import import _SFCLAY_ALLOWED
    assert 7 in _SFCLAY_ALLOWED


def test_the_oracle_fixtures_are_pinned():
    from woof.verify.pxsfclay_oracle import fixture_manifest
    manifest = fixture_manifest()
    assert len(manifest["configurations"]) == 5
    assert "-O2 -fno-fast-math -ffp-contract=off" in manifest["referee"]


def test_a_changed_constant_asset_is_refused_before_a_device_opens(tmp_path):
    from woof.core.pxsfclay import constant_document
    from woof.data_assets import data_path
    raw = bytearray(data_path("pxsfclay/constants.json").read_bytes())
    raw[raw.index(b"15.7") if b"15.7" in raw else 40] ^= 1
    (tmp_path / "constants.json").write_bytes(raw)
    with pytest.raises(ValueError, match="changed coefficient words"):
        constant_document(tmp_path)


def test_px_uses_the_shared_source_and_scalar_options():
    from woof.core.kernels import module_options, module_source
    from woof.core.pxsfclay import MODULE_OPTIONS, module_source as px_source
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        assert module_options("pxsfclay") == MODULE_OPTIONS
    assert module_source("pxsfclay") == px_source()


def test_px_history_inventory_contains_all_surface_words_only_for_seven():
    from woof.core.physics_inventory import PX_SFCLAY_STATE
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import PX_SFCLAY_OUTPUT_FIELDS
    on = produced_history_shapes(_cfg())
    off = produced_history_shapes(_cfg(sf_sfclay_physics=1))
    assert {key.upper() for key in PX_SFCLAY_STATE} <= set(on)
    assert set(PX_SFCLAY_OUTPUT_FIELDS) - {"rmol"}
    assert not ({field.netcdf_name for field in PX_SFCLAY_OUTPUT_FIELDS.values()}
                & set(off))
    for key, field in PX_SFCLAY_OUTPUT_FIELDS.items():
        assert field.dtype == "f4" and field.field_type == 104
        assert not field.wrf_history
        assert on[field.netcdf_name] == (16, 16)
