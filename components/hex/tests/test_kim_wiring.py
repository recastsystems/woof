"""The KIM drag (MPAS PR #1409, bl_kim_gwdo) is reachable from a forecast: knob, tables, Registry, statistics.

CPU-only: what the route resolves and what it builds, not the kernel (tests/test_kim_gwdo_2026.py replays that).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from woof.hex.mpasphys import gwdo_version
from woof.hex.mpasphys.forecast_v842 import native_route


def _route(physics=(), mode=None, suite="convection_permitting"):
    arguments = SimpleNamespace(physics_namelist=None, physics_suite=suite, physics_profile=None,
                                physics=list(physics), physics_backend="mpasphys", physics_pin_set=None,
                                physics_default_backend=None, physics_mode=mode, physics_correct_skip=None,
                                init=None)
    config, _ = native_route(arguments, default_mode="correct")
    return config


def test_default_is_the_v842_pin_in_every_mode():
    # kim2026 failed the obs-skill check beside v842 (hex 60 km 120 h): the pin stays default.
    for config in (_route(), _route(mode="correct")):
        assert config.selectors["gwdo"] == "bl_ysu_gwdo"
        assert config.gwdo is None
        assert "gwdo" not in config.receipt()


def test_kim2026_by_name_runs_kim():
    config = _route(["gwdo_version=kim2026"])
    assert config.selectors["gwdo"] == "bl_kim_gwdo"
    assert config.options["config_gwdo_scheme"] == "bl_kim_gwdo"
    assert dict(config.gwdo) == {"gwdo_version": "kim2026", "config_gwdo_factor": 2.0, "config_gwdo_nonhyd": True}
    assert config.receipt()["gwdo"]["gwdo_version"] == "kim2026"


def test_exact_mode_and_pin_keep_v842_word_for_word():
    for config in (_route(mode="exact"), _route(["gwdo_version=v842"]), _route(["gwdo_version=pinned"])):
        assert config.selectors["gwdo"] == "bl_ysu_gwdo"
        assert config.gwdo is None
        assert "gwdo" not in config.receipt()
        assert "config_gwdo_factor" not in config.options


def test_explicit_selectors_are_never_rewritten():
    assert _route(["config_gwdo_scheme=bl_ysu_gwdo"]).selectors["gwdo"] == "bl_ysu_gwdo"
    assert _route(["config_gwdo_scheme=bl_kim_gwdo"], mode="exact").selectors["gwdo"] == "bl_kim_gwdo"
    # The GSL drag is a different scheme: the version does not touch it.
    assert _route(suite="hrrrv5").selectors["gwdo"] == "bl_ugwp_gwdo"


def test_knobs_and_their_refusals():
    config = _route(["gwdo_version=kim2026", "config_gwdo_factor=1.5", "config_gwdo_nonhyd=false"])
    with pytest.raises(ValueError, match="select or tune the KIM drag"):
        _route(["config_gwdo_factor=1.5"])  # the default is the v842 pin, which never reads the knob
    assert config.gwdo["config_gwdo_factor"] == 1.5 and config.gwdo["config_gwdo_nonhyd"] is False
    with pytest.raises(ValueError, match="never read|select or tune the KIM drag"):
        _route(["gwdo_version=v842", "config_gwdo_factor=1.5"])
    with pytest.raises(ValueError, match="positive"):
        _route(["gwdo_version=kim2026", "config_gwdo_factor=0"])
    with pytest.raises(ValueError, match="gwdo_version must be one of"):
        _route(["gwdo_version=v9"])
    with pytest.raises(ValueError, match="GPU port only"):
        _route(["gwdo_version=kim2026", "config_gwdo_backend=native-cpu"])


def test_three_drag_tables_and_resident_admission():
    from woof.hex.physics_suites import load_table
    from woof.hex.mpasphys.resident_v842 import RESIDENT_SCHEMES
    table = load_table()
    for key in ("mpasphys_v842", "mpasphys_rrfs", "mpasphys_rrfs_44c0849"):
        slot = table[key]["slots"]["gwdo"]
        assert "bl_kim_gwdo" in slot["schemes"]
        entry = slot["cuda_entries"]["bl_kim_gwdo"]
        assert (entry["module"], entry["bind"], entry["call"]) == ("kim_gwdo_2026", "bind_forecast", "forecast_callback")
        assert "bl_kim_gwdo" not in slot["native_entries"]  # native MPAS v8.4.2 has no KIM drag
    assert "bl_kim_gwdo" in RESIDENT_SCHEMES["gwdo"]


def _bundled_registry():
    from woof.hex.mpasphys.pool_v842 import registry_text
    # (WOOF 1.1.7) __import__("woof.hex") returns the top package woof; fromlist returns woof.hex itself
    root = Path(__import__("woof.hex", fromlist=["__file__"]).__file__).parent / "data" / "mpasphys-framework"
    return ET.fromstring(registry_text(root / "src/core_atmosphere/Registry.xml"))


def test_registry_hunks_apply_only_with_kim():
    from woof.hex.mpasphys.kim_gwdo_2026 import prepare_registry
    pinned, kim = _bundled_registry(), _bundled_registry()
    assert prepare_registry(pinned, _route(mode="exact")) is False
    assert ET.tostring(pinned) == ET.tostring(_bundled_registry())
    assert prepare_registry(kim, _route(["gwdo_version=kim2026"])) is True
    assert prepare_registry(_bundled_registry(), _route()) is False  # the default runs the v842 pin
    sfc = next(s for s in kim.iter("var_struct") if s.get("name") == "sfc_input")
    assert any(v.get("name") == "elvmax" and v.get("units") == "m" for v in sfc)
    invariant = [s for s in kim.iter("stream") if any(v.get("name") == "elvmax" for v in s)]
    assert invariant and all(any(v.get("name") == "ol4" for v in s) for s in invariant)
    options = {o.get("name"): o for o in kim.iter("nml_option")}
    assert options["config_gwdo_factor"].get("default_value") == "2.0"
    assert options["config_gwdo_nonhyd"].get("type") == "logical"
    assert prepare_registry(kim, _route(["gwdo_version=kim2026"])) is True  # idempotent
    assert sum(v.get("name") == "elvmax" for v in sfc) == 1


def test_streams_know_elvmax():
    from woof.hex.streams_v842 import field_definition
    assert field_definition("elvmax")["attrs"]["units"] == "m"


def test_door_prepare_predicate_and_default_suite_exemption():
    from woof.hex.kim_gwd_static import forecast_args_select_kim as kim
    from woof.hex.physics_default import names_physics
    assert not kim(["forecast", "--mesh", "x"])  # the hex default suite runs the v842 pin
    assert kim(["forecast", "--mesh", "x", "--physics=gwdo_version=kim2026"])
    assert kim(["--physics=gwdo_version=kim2026"])
    assert not kim(["--physics=gwdo_version=v842"])
    assert not kim(["--physics-mode", "exact"])
    assert not kim(["--physics-suite", "hrrrv5"])
    assert not kim(["--physics-suite", "arwen_mpas_column"])
    # The version knob alone keeps the default MPAS suite; any other --physics names the run's physics.
    assert not names_physics(["--physics=gwdo_version=v842"])
    assert not names_physics(["--physics", "gwdo_version=v842"])
    assert names_physics(["--physics=config_gwdo_factor=1.0"])


def test_aot_inventory_carries_the_kim_kernel():
    from woof.hex.cuda_backend import aot_build
    source = Path(aot_build.__file__).read_text(encoding="utf-8")
    assert '"physics_family": "kim_gwdo"' in source and "kim_gwdo_2026.cu" in source
