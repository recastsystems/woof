"""The ACM2 PBL (bl_pbl_physics = 7): its fixtures and its doors, no device.

The card half (every output word against WRF v4.6.1's) is
tests/test_acm2_wrf461_parity.py.  Here: the oracle fixtures' pins, the
float32 constants the engine hands the kernel against the ones WRF's
driver handed the oracle, the fixtures' reach across both arms of the
free-convection switch, and the scheme's admission through every door.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from woof.verify.acm2_oracle import load_fixture, load_inputs

ROOT = Path(__file__).resolve().parents[1]
ORACLE = Path(os.environ.get("ACM2_ORACLE_DIR")
              or ROOT / "tests/data/oracles/acm2")
FIXTURES = ("acm2-l40", "acm2-l60")

#: sha256 of the fixtures of record (tools/acm2_wrf461_oracle/build.sh on
#: the sprint CPU box, 2026-10-08).  A regenerated fixture changes these on
#: purpose.
FIXTURE_SHA256 = {
    "acm2-l40.in.bin":
        "585af9158c1cc4bd8bb7fe4de72c1edbba4f05bff8248ef0d4611bf461c6792b",
    "acm2-l40.out.bin":
        "8e9d55ae9c04facf1417257252f3ed7c5829b1631a89e582a35ce8f39cbdd105",
    "acm2-l60.in.bin":
        "d103d1fc424bcf2149ae13ef57dc9da8b72ab90aa4c1b79e5fc9179cb7aca7c3",
    "acm2-l60.out.bin":
        "1bdbcaacc6338e3a518e3e95db1ce0a4a1250e712f4899b6734ecb6a7233161d",
    "fatal.in.bin":
        "e8d76018e6a451e147bbabdead391eed6ef9582516fc421958f76201c9f9e4b9",
}


def _fixture(name):
    stem = ORACLE / name
    if not Path(f"{stem}.out.bin").exists():
        pytest.skip(f"ACM2 oracle fixture {stem} is absent")
    return load_fixture(stem)


def test_the_fixtures_are_the_recorded_words():
    if ORACLE != ROOT / "tests/data/oracles/acm2":
        pytest.skip("pins apply to the packaged fixtures")
    for name, digest in FIXTURE_SHA256.items():
        data = (ORACLE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_the_forecast_constants_are_the_oracles():
    """The engine hands the kernel WRF's float32 module_model_constants."""
    from woof.core.acm2pbl import CP, EP_1, G, R_D
    for name in FIXTURES:
        c = load_inputs(ORACLE / name).constants
        for ours, theirs in ((EP_1, c["ep1"]), (G, c["g"]), (R_D, c["rd"]),
                             (CP, c["cpd"])):
            assert np.float32(ours).view(np.uint32) == \
                np.float32(theirs).view(np.uint32)


def test_the_fixtures_reach_both_sides_of_noconv():
    """The free-convection arm (MATRIX) and the local arm (TRI) both run,
    on step 1 (XTIME = 1 never convects) and on later steps."""
    for name in FIXTURES:
        fx = _fixture(name)
        ep1 = fx.constants["ep1"]
        seen = set()
        for st in fx.steps:
            th, qv = st["state3d"]["th"], st["state3d"]["qv"]
            tv = th * (np.float32(1.0) + ep1 * qv)
            pblh, rmol, kpbl = (st["out2d"]["pblh"], st["out2d"]["rmol"],
                                st["kpbl"])
            noconv = ((pblh * rmol < -0.02) & (kpbl > 3) & (tv[0] > tv[1])
                      & (st["xtime"] > 1))
            seen.update(bool(x) for x in noconv)
        assert seen == {True, False}, name


# ------------------------------------------------------------------- doors

def _cfg(**overrides):
    from woof.config import RunConfig
    values = dict(nx=4, ny=3, nz=40, dx=3000.0, dy=3000.0, ztop=15000.0,
                  dt=15.0, run_seconds=60.0, moist=True, mp_physics=1,
                  bl_pbl_physics=7, sf_sfclay_physics=1,
                  sf_surface_physics=2, num_soil_layers=4)
    values.update(overrides)
    return RunConfig(**values)


@pytest.mark.parametrize("sfclay", (1, 91, 7))
def test_bl_pbl_physics_7_runs_beside_the_mm5_surface_layers(sfclay):
    """A bare config naming bl_pbl_physics = 7 validates: no flag, no
    acknowledgement, no substitution (WRF's isfc = 1 cells)."""
    from woof.config import validate_run_config
    cfg = _cfg(sf_sfclay_physics=sfclay)
    assert validate_run_config(cfg) is cfg


@pytest.mark.parametrize("sfclay", (0, 5))
def test_wrfs_acm2_pairing_fatal_is_the_refusal(sfclay):
    """WRF's ACMPBLSCHEME arm (module_physics_init.F:3728-3730) fatals
    without isfc 1 or 7; the refusal names that law and the breakage."""
    from woof.config import validate_run_config
    from woof.physics_compat import UnsupportedPhysicsSuiteError
    cfg = _cfg(sf_sfclay_physics=sfclay,
               sf_surface_physics=0 if sfclay == 0 else 2)
    with pytest.raises(UnsupportedPhysicsSuiteError) as caught:
        validate_run_config(cfg)
    assert "phys/module_physics_init.F:" in str(caught.value)
    assert "3728-3730" in str(caught.value)
    assert "ACM2 fatals unless" in str(caught.value)
    assert "stops the model before any integration" in str(caught.value)
    assert "fm/fh" not in str(caught.value)


def test_the_eta_surface_layer_is_refused_beside_acm2():
    from woof.config import validate_run_config
    with pytest.raises(ValueError, match="sf_sfclay_physics=2"):
        validate_run_config(_cfg(sf_sfclay_physics=2))


def test_bep_urban_is_refused_beside_acm2():
    """WRF fatals ACM2 with BEP/BEM (module_physics_init.F:3731-3732)."""
    from woof.config import validate_run_config
    with pytest.raises(ValueError, match="bl_pbl_physics=7"):
        validate_run_config(_cfg(sf_urban_physics=2))


def test_every_table_routes_seven():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    from woof.config import ACM2_PBL_SCHEME, PBL_SCHEMES
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    from woof.namelist_import import _BL_MAP
    from woof.physics_vertical_contract import ACM2_VERTICAL_LEVEL_BOUNDS
    from woof.wrf461_compatibility import PBL_OPTIONS
    assert ACM2_PBL_SCHEME == 7 and 7 in PBL_SCHEMES and 7 in PBL_OPTIONS
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][7] == "_run_acm2"
    assert _BL_MAP[7] == (7, "ACM2 (Pleim)", "ACM2")
    assert PBL_ALGORITHM_IDENTITIES[7] == "acm2-pbl-wrf-v4.6.1-v1"
    assert ACM2_VERTICAL_LEVEL_BOUNDS == (4, None)


def test_acm2_history_has_all_carriers_and_retains_native_types():
    from woof.io.history_layout import produced_history_shapes
    from woof.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
    shapes = produced_history_shapes(_cfg())
    for name in ("PBLH", "REGIME", "RMOL", "KPBL"):
        assert shapes[name] == (3, 4), name
    for name in ("EXCH_H", "EXCH_M"):
        assert shapes[name] == (41, 3, 4), name
    assert SCHEME_OUTPUT_FIELDS["kpbl"].dtype == "i4"
    assert SCHEME_OUTPUT_FIELDS["regime"].dtype == "f4"
    # The shared arrays exist under other PBLs too. Only ACM2's history
    # mapping changes here, so its new fields must stay absent for YSU.
    assert "REGIME" not in produced_history_shapes(_cfg(bl_pbl_physics=1))


def test_the_registry_option_is_implemented_and_reachable():
    from woof.physics_registry import physics_registry
    registry = physics_registry()
    option = registry["components"]["pbl"]["options"]["acm2"]
    assert option["implemented"] is True
    assert option["selectors"] == {"bl_pbl_physics": 7}
    assert option["constraints"]["requires_components"] == {
        "surface_layer": ["revised-mm5", "classic-mm5", "pleim-xiu"]}
    # Both sides of a legal pair must admit each other. A one-sided row
    # parses a stock namelist but still refuses a prepared physics plan.
    for layer in ("revised-mm5", "classic-mm5", "pleim-xiu"):
        partners = registry["components"]["surface_layer"]["options"][layer].get(
            "constraints", {}).get("requires_components", {}).get("pbl")
        assert partners is None or "acm2" in partners, layer
    for route in ("tools.prepared_domain_tree_forecast",
                  "tools.prepared_single_domain_forecast"):
        allowed = registry["runner_routes"][route].get(
            "allowed_component_options", {}).get("pbl")
        if allowed is not None:
            assert "acm2" in allowed, route


def test_the_namelist_door_imports_seven_natively(tmp_path):
    """A WRF namelist pair naming bl_pbl_physics = 7 imports as 7: no
    substitution row, every domain keeps the selector."""
    from woof.experiment import load_experiment
    from woof.namelist_import import import_namelists
    from test_namelist_import import INPUT_TEXT, _pair
    inp = INPUT_TEXT.replace(" bl_pbl_physics = 11, 11,",
                             " bl_pbl_physics = 7, 7,")
    assert inp != INPUT_TEXT
    toml_text, report = import_namelists(*_pair(tmp_path, inp=inp))
    assert not any(s.key == "bl_pbl_physics" for s in report.substitutions)
    out = tmp_path / "acm2.toml"
    out.write_text(toml_text)
    exp = load_experiment(out)
    assert all(dc.run.bl_pbl_physics == 7 for dc in exp.domains)


def test_the_memory_preflight_prices_the_workspace():
    from woof.core import preflight as pf
    from woof.core.physics_inventory import (ACM2_BLOCK, ACM2WS_SLOTS,
                                              acm2_workspace_floats)
    assert acm2_workspace_floats(40, 1) == ACM2WS_SLOTS * 42 * ACM2_BLOCK
    assert pf._PBL_KERNEL_MODULES[7] == ("acm2pbl_composed",)
    assert "acm2pbl" in pf.UNMEASURED_KERNEL_MODULES
    assert pf.CHAINED_TRANSLATION_UNIT_FRAMES[
        "acm2pbl_composed"].covers == frozenset({"acm2pbl"})


def test_the_constant_asset_refuses_a_one_bit_substitution(tmp_path):
    from woof.core.acm2pbl import CONSTANT_ASSET, constants_table
    original = (ROOT / "woof" / CONSTANT_ASSET.filename).read_bytes()
    constants_table()
    damaged = bytearray(original)
    damaged[10] ^= 1
    (tmp_path / CONSTANT_ASSET.filename).write_bytes(damaged)
    with pytest.raises(ValueError, match="ACM2 constant asset.*SHA-256") as caught:
        constants_table(tmp_path)
    assert "restore its packaged ACM2 constants" in str(caught.value)
    assert "fetch-tables --schemes thompson" not in str(caught.value)


def test_acm2_workspace_is_priced_only_for_the_selected_domains():
    from datetime import datetime
    from dataclasses import replace
    from woof.core import preflight as pf
    from woof.experiment import experiment_from_run_config
    from woof.core.physics_inventory import acm2_workspace_floats
    profile = pf.DeviceLocalMemoryProfile(name="acm2-workspace",
        multiprocessor_count=2, max_threads_per_multiprocessor=1536)
    cfg = _cfg(nx=257, ny=10)
    exp = experiment_from_run_config(cfg, datetime(2000, 1, 1))
    assert pf.acm2pbl_column_workspace_bytes(exp, profile=profile) == (
        acm2_workspace_floats(40, 2 * 16 * 32) * 4)
    exp_off = experiment_from_run_config(replace(cfg, bl_pbl_physics=1),
                                         datetime(2000, 1, 1))
    assert pf.acm2pbl_column_workspace_bytes(exp_off, profile=profile) == 0
