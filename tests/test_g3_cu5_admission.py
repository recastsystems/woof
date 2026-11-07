"""Grell 3D (cu_physics = 5) at the config doors (no card needed).

The scheme is admitted as table rows: ``woof.config.CU_SCHEMES`` and
``CU_STOCK_CALLABLES``, the Grell-family key table, the model-step cadence
law, the registry option ``grell-3d`` and the importer's ``_CU_ALLOWED``.
These tests hold every row and every refusal to its named breakage.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_G3_BASE = dict(nx=16, ny=16, nz=30, dx=12000.0, dy=12000.0, ztop=16000.0,
                dt=60.0, run_seconds=0.0, moist=True, mp_physics=8,
                bl_pbl_physics=1, sf_sfclay_physics=91, cu_physics=5,
                cudt_minutes=0.0)


def test_g3_is_an_admitted_scheme_with_a_stock_callable():
    import importlib

    from woof.config import (CU_MODEL_STEP_SCHEMES, CU_SCHEMES,
                              CU_STOCK_CALLABLES,
                              CUMULUS_ADVECTIVE_FORCING_SCHEMES,
                              GRELL_FAMILY_CU_PHYSICS)

    assert 5 in CU_SCHEMES and 5 in GRELL_FAMILY_CU_PHYSICS
    assert 5 in CU_MODEL_STEP_SCHEMES
    assert 5 in CUMULUS_ADVECTIVE_FORCING_SCHEMES
    module, _, name = CU_STOCK_CALLABLES[5].rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    assert cls.__name__ == "Grell3D"
    # every admitted nonzero scheme has exactly one dispatch row: a stock
    # callable, or a row of the column-cumulus table (New SAS, 96), never both
    # BMJ (2) is the third table: a column-kernel scheme built through
    # ColumnCumulus.for_scheme (woof.config.COLUMN_KERNEL_CU_SCHEMES).
    from woof.config import COLUMN_KERNEL_CU_SCHEMES
    from woof.core.column_cumulus import COLUMN_CUMULUS_SCHEMES
    assert not set(CU_STOCK_CALLABLES) & set(COLUMN_CUMULUS_SCHEMES)
    assert not set(COLUMN_KERNEL_CU_SCHEMES) & (
        set(CU_STOCK_CALLABLES) | set(COLUMN_CUMULUS_SCHEMES))
    assert (set(CU_STOCK_CALLABLES) | set(COLUMN_CUMULUS_SCHEMES)
            | set(COLUMN_KERNEL_CU_SCHEMES) == set(CU_SCHEMES) - {0})


@pytest.mark.parametrize("avedx", [1, 3])
@pytest.mark.parametrize("ishallow", [0, 1])
def test_a_g3_config_validates(avedx, ishallow):
    from woof.config import RunConfig, validate_run_config

    cfg = validate_run_config(RunConfig(**_G3_BASE, cugd_avedx=avedx,
                                        ishallow=ishallow, clos_choice=0))
    assert (cfg.cu_physics, cfg.cugd_avedx, cfg.ishallow) == (5, avedx, ishallow)


def test_g3_runs_without_a_pbl_scheme():
    """The deep arm reads no KPBL and the shallow arm guards it with
    kpbl > 5 (module_cu_g3.F:1238-1241): no PBL requirement."""
    from woof.config import RunConfig, validate_run_config

    base = dict(_G3_BASE, bl_pbl_physics=0)
    validate_run_config(RunConfig(**base, ishallow=1))


def test_a_cumulus_cadence_is_refused_by_its_breakage():
    from woof.config import RunConfig, validate_run_config

    with pytest.raises(ValueError, match=r"cu_physics=5 \(Grell 3D\) "
                                         r"requires cudt_minutes=0") as e:
        validate_run_config(RunConfig(**dict(_G3_BASE, cudt_minutes=5.0)))
    assert "PRATEC" in str(e.value) and "drop" in str(e.value)


def test_cugd_avedx_is_refused_where_no_scheme_reads_it():
    from woof.config import RunConfig, validate_run_config

    gf = dict(_G3_BASE, cu_physics=3)
    with pytest.raises(ValueError, match="cugd_avedx=3 is refused with "
                                         "cu_physics=3") as e:
        validate_run_config(RunConfig(**gf, cugd_avedx=3))
    assert "cu_physics=5" in str(e.value)
    kf = dict(_G3_BASE, cu_physics=1, cudt_minutes=5.0)
    with pytest.raises(ValueError, match="Grell-family keys"):
        validate_run_config(RunConfig(**kf, cugd_avedx=3))


def test_closure_choice_range_is_the_grell_family_rule():
    from woof.config import RunConfig, validate_run_config

    validate_run_config(RunConfig(**_G3_BASE, clos_choice=16))
    with pytest.raises(ValueError, match="reads past the end"):
        validate_run_config(RunConfig(**_G3_BASE, clos_choice=17))


def test_workspace_slot_count_is_the_kernel_source_count():
    """g3.G3_SLOT_COUNT (what preflight prices) is the .cu's own count."""
    from woof.core import g3

    src = (Path(g3.__file__).parent / "kernels" / "g3.cu").read_text(
        encoding="utf-8")
    block = re.search(r"#define G3_SLOTS\(X\)(.*?)\n\nenum", src, re.S)
    names = re.findall(r"X\((\w+)\)", block.group(1))
    assert len(set(names)) == len(names)
    ens4 = int(re.search(r"#define G3_ENS4\s+(\d+)", src).group(1))
    maxens2 = int(re.search(r"#define G3_MAXENS2\s+(\d+)", src).group(1))
    assert len(names) + 3 * ens4 + 6 * maxens2 == g3.G3_SLOT_COUNT


def test_the_written_band_is_wrfs():
    from woof.core.g3 import g3_band, spread_band

    assert g3_band(16, 16) == (4, 11, 4, 11)
    assert g3_band(16, 16, periodic_x=True, periodic_y=True) == (0, 15, 0, 15)
    assert spread_band(16, 16) == (4, 11, 4, 11)


def test_nondefault_ensemble_dimensions_are_configurable():
    from woof.config import RunConfig, validate_run_config
    cfg = validate_run_config(RunConfig(**_G3_BASE, maxiens=2, maxens=4,
        maxens2=2, maxens3=17, ensdim=272, cu_diag=1, convtrans_avglen_m=4.))
    assert cfg.ensdim == 272 and cfg.cu_diag == 1


def test_short_ensemble_arrays_name_the_out_of_bounds_read():
    from woof.config import RunConfig, validate_run_config
    with pytest.raises(ValueError, match="member 16"):
        validate_run_config(RunConfig(**_G3_BASE, maxens3=15))
    with pytest.raises(ValueError, match="edtc.*3"):
        validate_run_config(RunConfig(**_G3_BASE, cugd_avedx=3, maxens2=2))


def test_g3_specific_controls_name_the_only_runtime_consumer():
    from woof.physics_registry import physics_registry, registry_knob_is_read
    registry = physics_registry()
    for name in ("maxiens", "maxens", "maxens2", "maxens3", "ensdim"):
        row = registry["parameters"][name]
        assert row["read_when"] == {"cu_physics": 5}
        assert registry_knob_is_read(name, {"cu_physics": 5}, registry)
        for selector in (0, 1, 3, 16):
            assert not registry_knob_is_read(name, {"cu_physics": selector}, registry)
    # GD always carries convective transport diagnostics. GF reads the
    # averaging clock only when its radiation feedback is requested.
    for name in ("cu_diag", "convtrans_avglen_m"):
        for selector in (5, 93):
            assert registry_knob_is_read(name, {"cu_physics": selector}, registry)
        for selector in (0, 1, 3, 16):
            assert not registry_knob_is_read(name, {"cu_physics": selector}, registry)
    assert registry_knob_is_read("convtrans_avglen_m",
                                 {"cu_physics": 3, "cu_rad_feedback": True}, registry)


def test_inactive_g3_controls_do_not_change_existing_cumulus_requirements():
    from woof.config import G3_RUN_DEFAULTS
    from woof.physics_registry import physics_registry
    options = physics_registry()["components"]["cumulus"]["options"]
    for option in options.values():
        if option.get("selectors", {}).get("cu_physics") == 5:
            continue
        required = option.get("constraints", {}).get("required_settings", {})
        assert not (set(G3_RUN_DEFAULTS) - {"cugd_avedx"}) & set(required)


@pytest.mark.parametrize("name,value", [
    ("maxiens", 2), ("maxens", 4), ("maxens2", 4),
    ("maxens3", 17), ("ensdim", 272), ("cu_diag", 1),
    ("convtrans_avglen_m", 4.0),
])
def test_nondefault_g3_controls_remain_refused_without_the_consumer(name, value):
    from woof.config import RunConfig, validate_run_config
    with pytest.raises(ValueError, match="cu_physics=5"):
        validate_run_config(RunConfig(**dict(_G3_BASE, cu_physics=3), **{name: value}))


def test_registry_declares_inactive_control_policy_without_changing_other_options():
    from woof.physics_registry import physics_registry
    registry = physics_registry()
    for name in ("cugd_avedx", "maxiens", "maxens", "maxens2", "maxens3",
                 "ensdim", "cu_diag", "convtrans_avglen_m"):
        assert registry["parameters"][name]["inactive_value_policy"] == "require-default"
    assert registry["components"]["cumulus"]["options"]["off"].get("constraints", {}) == {}


@pytest.mark.parametrize("value", [0, 1])
def test_registry_and_runtime_agree_on_inactive_diagnostics(value):
    from woof.config import RunConfig, validate_run_config
    from woof.physics_registry import physics_registry, validate_physics_plan
    from test_authority_agreement import _tree_plan
    registry = physics_registry()
    profile = "thompson-mp8-ysu-mm5-noah-validation-v1"
    plan = _tree_plan(registry, "tools.prepared_domain_tree_forecast",
                               "gfs", profile, parameters={"cu_diag": value})
    result = validate_physics_plan(plan, registry=registry)
    assert bool(result["launchable"]) is (value == 0)
    if value == 1:
        assert any("discard" in row["message"] for row in result["errors"])
        with pytest.raises(ValueError, match="cu_physics=5"):
            validate_run_config(RunConfig(**dict(_G3_BASE, cu_physics=0), cu_diag=value))


def test_a_streamed_periodic_g3_domain_is_refused_by_name():
    """[tiles] on a periodic axis would move G3's band off the domain edges.

    Measured before the refusal (RTX 5090, tilestream gate, 96x80 periodic,
    2x2 and 3x3 tiles): rthcuten and the G3 carriers differed from the
    resident run.  A specified-boundary domain is clamped and is not refused.
    """
    from types import SimpleNamespace

    from woof.core.streaming import StreamingRefused, attach

    def cfg(**over):
        base = dict(grid_id=2, slope_rad=0, sf_surface_mosaic=0, cu_physics=5,
                    open_x=False, open_y=False, specified=False, nested=False)
        base.update(over)
        return SimpleNamespace(**base)

    with pytest.raises(StreamingRefused,
                       match=r"grid_id = 2 sets cu_physics = 5.*periodic in "
                             r"x and y.*cannot run streamed.*mode = 'off'"):
        attach(None, cfg(), SimpleNamespace(stream=True), tile_state_factory=None)
    # The same scheme on a specified-boundary domain, and another scheme on
    # the periodic one, reach attach's own checks past this refusal.
    for passing in (cfg(specified=True),
                    cfg(cu_physics=1)):
        with pytest.raises(StreamingRefused, match="does not stream"):
            attach(None, passing, SimpleNamespace(stream=False),
                   tile_state_factory=None)
