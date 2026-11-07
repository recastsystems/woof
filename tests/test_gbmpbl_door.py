"""The GBM moist TKE PBL (bl_pbl_physics=12): its doors and its seams.

No device is opened here.  What is pinned:

* the doors a user reaches the scheme through: the RunConfig key, the WRF
  namelist importer, and the registry option plan review reads;
* WRF's one law for the scheme (isfc = 1, module_physics_init.F:
  3892-3893) and the WOOF refusal (vapour without a cloud-water species),
  each naming its breakage;
* the dispatch row, checkpoint identity, vertical bound, kernel-module
  row, VRAM terms and history names, each held to the one table it reads;
* default-off: a configuration that does not select the scheme allocates
  none of its fields and prices none of its memory.

The driver on a card is tests/test_gbmpbl_driver_seam.py.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from woof.config import (GBM_PBL_SCHEME, RunConfig, validate_run_config)

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=16, ny=16, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=GBM_PBL_SCHEME,
                sf_sfclay_physics=1, sf_surface_physics=2)
    base.update(overrides)
    return RunConfig(**base)


def test_gbm_is_admitted_with_both_mm5_surface_layers():
    for sfclay in (1, 91):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


@pytest.mark.parametrize("sfclay", [0, 2, 5])
def test_other_surface_layers_are_wrfs_own_fatal(sfclay):
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(sf_sfclay_physics=sfclay,
                                 sf_surface_physics=0))
    message = str(caught.value)
    assert "bl_pbl_physics=12 (GBM)" in message
    assert "module_physics_init.F:3892-3893" in message
    assert "GZ1OZ0 and PSIM" in message


@pytest.mark.parametrize("moist,mp", [(False, 0), (True, 0), (True, 1)])
def test_missing_condensate_state_is_refused(moist, mp):
    with pytest.raises(ValueError, match="requires F_QC and F_QI"):
        validate_run_config(_cfg(moist=moist, mp_physics=mp))


def test_bep_urban_keeps_refusing_gbm():
    # WRF admits only YSU, MYJ and BouLac under BEP/BEM
    # (module_physics_init.F:3705-3790); GBM is not among them.
    from woof.config import URBAN_BEP_PBL_SCHEMES
    assert GBM_PBL_SCHEME not in URBAN_BEP_PBL_SCHEMES


def test_dispatch_routes_twelve_to_its_own_runner():
    from woof.core.physics import PHYSICS_SLOT_DISPATCH, PhysicsDriver
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][GBM_PBL_SCHEME] == \
        "_run_gbmpbl"
    assert callable(getattr(PhysicsDriver, "_run_gbmpbl"))


def test_the_checkpoint_identity_names_the_wrf_version_ported():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert PBL_ALGORITHM_IDENTITIES[GBM_PBL_SCHEME] == \
        "gbm-moist-tke-pbl-wrf-v4.6.1-v4-live-shared-fast"


def test_the_registry_option_is_the_selector_and_names_the_law():
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["pbl"]["options"]["gbm"]
    assert option["selectors"] == {"bl_pbl_physics": GBM_PBL_SCHEME}
    assert option["implemented"] is True
    assert option["constraints"]["requires_components"]["surface_layer"] \
        == ["revised-mm5", "classic-mm5"]
    assert option["consumers"]["restart_algorithm_identity"] == \
        "gbm-moist-tke-pbl-wrf-v4.6.1-v4-live-shared-fast"
    bounds = option["consumers"]["vertical_level_bounds"]
    assert (bounds["minimum"], bounds["maximum"]) == (2, None)
    for route in ("tools.prepared_domain_tree_forecast",
                  "tools.prepared_single_domain_forecast"):
        allowed = registry["runner_routes"][route]["allowed_component_options"]
        assert "gbm" in allowed["pbl"], route


def test_the_namelist_importer_carries_twelve_natively():
    from woof.namelist_import import _BL_MAP
    assert _BL_MAP[12][0] == GBM_PBL_SCHEME


def test_the_compatibility_matrix_declares_the_exclusion():
    from woof.wrf461_compatibility import AXIS_EXCLUSIONS, PBL_OPTIONS
    assert GBM_PBL_SCHEME not in PBL_OPTIONS
    assert "validate_gbmpbl_config" in AXIS_EXCLUSIONS[
        ("bl_pbl_physics", GBM_PBL_SCHEME)]


def test_the_launcher_bound_is_the_contract_bound():
    from woof.core import gbmpbl
    from woof.physics_vertical_contract import GBMPBL_VERTICAL_LEVEL_BOUNDS
    assert gbmpbl.VERTICAL_LEVEL_BOUNDS is GBMPBL_VERTICAL_LEVEL_BOUNDS
    assert GBMPBL_VERTICAL_LEVEL_BOUNDS == (2, None)


def test_the_kernel_geometry_mirror_matches_the_kernel():
    from woof.core.physics_inventory import GBMPBL_BLOCK, GBMPBL_WS_SLOTS
    source = (ROOT / "woof/core/kernels/gbmpbl.cu").read_text(
        encoding="utf-8")
    assert f"#define GBWS_LANES {GBMPBL_BLOCK}" in source
    assert f"#define GBWS_SLOTS {GBMPBL_WS_SLOTS}" in source


def test_the_loader_compiles_without_fma_or_ftz():
    from woof.core import gbmpbl
    assert gbmpbl.MODULE_OPTIONS == ("-std=c++17", "-fmad=false",
                                     "--ftz=false")
    source = gbmpbl.module_source()
    assert "gfk_pow" in source and "gbmpbl_column" in source


def test_preflight_prices_the_scheme_and_only_the_scheme():
    from woof.core import preflight
    from woof.core.physics_inventory import GBMPBL_STATE_3D

    assert preflight._PBL_KERNEL_MODULES[GBM_PBL_SCHEME] == (
        "gbmpbl_composed",)
    frame = preflight.CHAINED_TRANSLATION_UNIT_FRAMES["gbmpbl_composed"]
    assert frame.covers == frozenset({"gbmpbl"})
    on = preflight.gbmpbl_output_transient_shapes(_cfg())
    off = preflight.gbmpbl_output_transient_shapes(_cfg(bl_pbl_physics=1))
    assert on and not off
    assert set(GBMPBL_STATE_3D) == {"tke_pbl", "el_pbl", "exch_tke"}


def test_history_names_are_wrfs():
    from woof.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
    assert SCHEME_OUTPUT_FIELDS["tke_pbl"].netcdf_name == "TKE_PBL"
    assert SCHEME_OUTPUT_FIELDS["tke_pbl"].stagger == "Z"
    assert SCHEME_OUTPUT_FIELDS["exch_tke"].netcdf_name == "EXCH_TKE"
    assert SCHEME_OUTPUT_FIELDS["el_pbl"].netcdf_name == "EL_PBL"


def test_default_off_allocates_nothing():
    """Every GBM allocation is behind its selector (read off the source)."""
    source = (ROOT / "woof/core/physics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            if ast.unparse(node.test) == \
                    "int(cfg.bl_pbl_physics) == GBM_PBL_SCHEME":
                guarded |= "GBMPBL_STATE_3D" in ast.unparse(node)
    assert guarded
    assert source.count("GBMPBL_STATE_3D") == 2  # import + guarded loop


def test_pinned_constant_asset_rejects_a_changed_word(tmp_path):
    from woof.core.thompson_contract import TableAsset
    from woof.table_assets import read_pinned_text_asset
    import hashlib
    source = ROOT / "woof/data/gbmpbl/constants.cuh"
    original = source.read_bytes()
    asset = TableAsset("constants.cuh", len(original), hashlib.sha256(original).hexdigest())
    (tmp_path / asset.filename).write_bytes(original)
    assert read_pinned_text_asset(tmp_path, asset) == source.read_text()
    (tmp_path / asset.filename).write_bytes(original.replace(b"5.869f", b"5.868f"))
    with pytest.raises(ValueError):
        read_pinned_text_asset(tmp_path, asset)


def test_gbm_owned_loader_retains_the_same_source_and_options():
    from woof.core import gbmpbl
    import hashlib
    assert hashlib.sha256(gbmpbl.module_source().encode()).hexdigest() == (
        "b9cd4cfaf4d7f29e7531f05c5aa48b440163ec6698e3877619006b3b35ae9ecc")
    assert gbmpbl.MODULE_OPTIONS == ("-std=c++17", "-fmad=false", "--ftz=false")


def test_metadata_names_every_wrf_binding_and_carried_state():
    document = json.loads((ROOT / "woof/physics_registry_v2.json").read_text())
    contract = document["components"]["pbl"]["options"]["gbm"]["extensions"]["wrf_contract"]
    assert contract["prognostic_restart_state"] == ["tke_pbl"]
    assert contract["radiation_input"] == "RTHRATEN"
    assert contract["cadence"] == "bldt"
    assert contract["required_species_flags"] == ["F_QC", "F_QI"]
    assert contract["preserved_surface_output"] == ["REGIME"]
    assert set(contract["package_state"]) == {"exch_tke", "el_pbl", "tke_pbl"}


def test_gbm_float_constants_are_in_the_pinned_table():
    import re
    source = (ROOT / "woof/core/kernels/gbmpbl.cu").read_text()
    assert not re.search(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f\b", source)


def test_regime_is_published_only_for_gbm_as_a_preserved_diagnostic():
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
    field = SCHEME_OUTPUT_FIELDS["regime"]
    assert field.netcdf_name == "REGIME" and not field.wrf_history
    assert "REGIME" in produced_history_shapes(_cfg())
    assert "REGIME" not in produced_history_shapes(_cfg(bl_pbl_physics=1))


@pytest.mark.parametrize("family,name", [("column", k) for k in
    ("u", "v", "t", "qv", "qc", "qi", "p", "exner", "dz8w", "rthraten")]
    + [("surface", k) for k in ("psfc", "gz1oz0", "psim", "xland", "hfx", "qfx", "tsk")]
    + [("state", "tke_pbl")])
def test_missing_required_input_refuses_before_geometry_or_device(family, name):
    from woof.core import gbmpbl
    columns = dict.fromkeys(gbmpbl.GBMPBL_COLUMN_INPUTS, object())
    surface = dict.fromkeys(gbmpbl.GBMPBL_SURFACE_INPUTS, object())
    tke = object()
    if family == "column": columns.pop(name)
    elif family == "surface": surface.pop(name)
    else: tke = None
    with pytest.raises(KeyError) as caught:
        gbmpbl.launch_gbmpbl(columns, surface, tke, {}, dt=15.0)
    assert (name if family != "state" else "TKE_PBL") in str(caught.value)
    assert "budget" in str(caught.value)


def test_native_restart_keeps_gbm_tke_and_rebuilds_history_only_exchange():
    from woof.ingest.wrfrst import fields_restore_plan

    assert fields_restore_plan("tke_pbl", {"TKE_PBL"}) == ("fill", "TKE_PBL")
    assert fields_restore_plan("el_pbl", {"EL_PBL"}) == ("fill", "EL_PBL")
    assert fields_restore_plan("tke_pbl", set()) is None
    assert fields_restore_plan("exch_tke", {"EXCH_TKE"}) == ("fill", "EXCH_TKE")
    plan = fields_restore_plan("exch_tke", set())
    assert plan[:2] == ("cold", 0.0)
    assert "without r" in plan[2]
