"""The UW moist-turbulence PBL (bl_pbl_physics=9): its doors and its seams.

No device is opened here.  What is pinned:

* the three doors a user reaches the scheme through: the RunConfig key
  (TOML), the WRF namelist importer, and the registry option the plan
  review and per-domain overrides read;
* WRF's one pairing law for the scheme (the BEP/BEM urban fatal,
  module_physics_init.F:3825-3827) and the WOOF-ArWen refusals, each naming its
  breakage;
* the dispatch row, the checkpoint identity and the vertical bound, each
  held equal to the one authority it restates;
* default-off: a configuration that does not select the scheme allocates
  none of its fields and never reaches its seams.

The driver seam on a card is tests/test_uwpbl_driver_seam.py: a helper
there imports cupy, and tests/conftest.py marks a whole module ``gpu`` for
that, which would deselect every check here from the CPU battery.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from woof.config import (UW_PBL_ICE_NUMBER_SPECIES, UW_PBL_SCHEME,
                          RunConfig, validate_run_config)

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**overrides) -> RunConfig:
    base = dict(nx=16, ny=16, nz=30, dx=3000.0, dy=3000.0, dt=15.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                bl_pbl_physics=UW_PBL_SCHEME,
                sf_sfclay_physics=1, sf_surface_physics=2)
    base.update(overrides)
    return RunConfig(**base)


def test_the_uw_pbl_is_admitted_with_every_surface_layer_it_can_read():
    for sfclay in (1, 5, 91):
        validate_run_config(_cfg(sf_sfclay_physics=sfclay))


def test_no_surface_layer_is_refused_by_name_and_a_dry_state_admitted():
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(sf_sfclay_physics=0, sf_surface_physics=0))
    assert "UST, HFX, QFX" in str(caught.value)
    # Dry is admitted for every scheme in the PBL slot
    # (tests/test_myj_port.py::test_a_dry_pbl_config_is_admitted_by_both_
    # config_doors); the scheme then reads the driver's zero planes.
    validate_run_config(_cfg(moist=False, mp_physics=0))


def test_the_eta_layer_refusal_names_the_tke_myj_scan():
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(sf_sfclay_physics=2))
    message = str(caught.value)
    assert "TKE_MYJ" in message and "bl_pbl_physics=9 (UW)" in message


def test_bep_and_bem_urban_are_wrfs_own_fatal():
    """The schema has no sf_urban_physics key at this base; the law is
    read off whatever object carries one, so it fires the day it lands."""
    from types import SimpleNamespace

    from woof.config import validate_uwpbl_config
    for urban in (2, 3):
        cfg = SimpleNamespace(bl_pbl_physics=UW_PBL_SCHEME,
                              sf_sfclay_physics=1, moist=True,
                              sf_urban_physics=urban, bldt=0.0, mp_physics=8)
        with pytest.raises(ValueError) as caught:
            validate_uwpbl_config(cfg)
        assert "module_physics_init.F:3826-3827" in str(caught.value)
    validate_uwpbl_config(SimpleNamespace(
        bl_pbl_physics=UW_PBL_SCHEME, sf_sfclay_physics=1, moist=True,
        sf_urban_physics=1, bldt=0.0, mp_physics=8))


def test_a_positive_pbl_cadence_is_refused_only_where_ice_number_is_held():
    with pytest.raises(NotImplementedError) as caught:
        validate_run_config(_cfg(bldt=5.0, mp_physics=8))
    assert "RQNIBLTEN" in str(caught.value)
    # WSM6 carries no P_QNI: nothing is held outside the manifest.
    validate_run_config(_cfg(bldt=5.0, mp_physics=6))


def test_ice_number_species_are_state_names_the_allocator_makes():
    source = (ROOT / "woof/core/state.py").read_text(encoding="utf-8")
    for mp, name in UW_PBL_ICE_NUMBER_SPECIES.items():
        assert f'"{name}"' in source, (mp, name)


def test_dispatch_routes_nine_to_its_own_runner():
    from woof.core.physics import PHYSICS_SLOT_DISPATCH, PhysicsDriver
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][UW_PBL_SCHEME] == \
        "_run_uwpbl"
    assert callable(getattr(PhysicsDriver, "_run_uwpbl"))


def test_the_checkpoint_identity_names_the_wrf_version_ported():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    assert PBL_ALGORITHM_IDENTITIES[UW_PBL_SCHEME] == \
        "uw-moist-turbulence-pbl-wrf-v4.7.1-v1"


def test_the_registry_option_is_the_selector():
    import json
    registry = json.loads((ROOT / "woof/physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    option = registry["components"]["pbl"]["options"]["uw"]
    assert option["selectors"] == {"bl_pbl_physics": UW_PBL_SCHEME}
    assert option["implemented"] is True
    assert "eta-similarity" not in option["constraints"][
        "requires_components"]["surface_layer"]
    assert "uw" in registry["components"]["surface_layer"]["options"][
        "mynn"]["constraints"]["requires_components"]["pbl"]


def test_the_namelist_importer_carries_nine_natively():
    from woof.namelist_import import _BL_MAP
    assert _BL_MAP[9][0] == UW_PBL_SCHEME


def test_the_launcher_bound_is_the_contract_bound():
    source = (ROOT / "woof/core/uwpbl.py").read_text(encoding="utf-8")
    assert "VERTICAL_LEVEL_BOUNDS = UWPBL_VERTICAL_LEVEL_BOUNDS" in source
    from woof.physics_vertical_contract import UWPBL_VERTICAL_LEVEL_BOUNDS
    assert UWPBL_VERTICAL_LEVEL_BOUNDS == (2, None)


def test_default_off_allocates_nothing_and_reaches_no_seam():
    """Every UW allocation and seam is behind its selector.

    Read off the source: the allocation block is guarded by
    ``bl_pbl_physics == UW_PBL_SCHEME`` and the radiation seam by the
    presence of the scheme's own ``uw_cldfra`` field, which only that block
    allocates.  So a configuration that does not select the scheme builds
    and steps exactly what it built and stepped before the port.
    """
    source = (ROOT / "woof/core/physics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            text = ast.unparse(node.test)
            if text == "int(cfg.bl_pbl_physics) == UW_PBL_SCHEME":
                body = ast.unparse(node)
                guarded |= "UWPBL_STATE_FULL" in body and \
                    "UWPBL_HELD_3D" in body
    assert guarded
    assert source.count('"uw_cldfra" in self.fields') == 1
    assert source.count("UWPBL_HELD_3D") == 2  # import + guarded allocation
