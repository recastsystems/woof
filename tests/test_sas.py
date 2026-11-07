"""GFS SAS (cu_physics = 94) against WRF v4.6.1, stage by stage, at 0 ULP.

The oracle (tools/sas_wrf461_oracle) links the BUILT strict and scalar WRF
trees' module_cu_sas.o and module_gfs_*.o and records three stages per
column; the fixture is the scalar tree's (byte-identical to strict).  The
GPU tests replay every stage and count words that differ.  A green suite on
a fixture where nothing convects would prove nothing, so the fixture tests
count what the columns actually do first.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import sas_oracle as so


# ---- the fixture exercises what it claims to -------------------------------------
def test_fixture_has_every_family_and_at_least_64_columns():
    cases = so.load_cases()
    assert len(cases) >= 64
    cats = {c["category"] for c in cases}
    assert cats == set(range(len(so.CATEGORIES)))
    assert any(c["nz"] == 2 for c in cases), "no two-level column"
    assert any(c["dt"] * c["stepcu"] >= 900.0 for c in cases), "no long DELT"
    assert {c["xland"] for c in cases} == {1.0, 2.0}, "land and ocean"


def test_stage_records_line_up_with_the_cases():
    cases, s1, s2, s3 = so.load_cases(), so.load_stage1(), so.load_stage2(), so.load_stage3()
    assert len(cases) == len(s1) == len(s2) == len(s3)
    for c, a, b, d in zip(cases, s1, s2, s3):
        assert c["nz"] == a["nz"] == b["nz"] == d["nz"]
        assert a["kcnv"] == b["kcnv"]


def test_the_fixture_convects_deep_and_shallow():
    s1, s2 = so.load_stage1(), so.load_stage2()
    deep = sum(r["kcnv"] for r in s1)
    shallow = sum(int(np.any(r["q_out"] != r["q_in"]) or np.any(r["t_out"] != r["t_in"]))
                  for r in s2)
    raining = sum(r["rn"] > 0.0 for r in s1)
    assert deep >= 30 and raining == deep
    assert shallow >= 30
    # deep and shallow are exclusive by construction (KUO = KCNV gate)
    for a, b in zip(s1, s2):
        if a["kcnv"] == 1:
            assert np.array_equal(b["q_out"], b["q_in"])


def test_dry_and_two_level_columns_are_inert():
    cases, s3 = so.load_cases(), so.load_stage3()
    for c, r in zip(cases, s3):
        if so.CATEGORIES[c["category"]] in ("dry", "two-level"):
            assert r["raincv"] == 0.0
            assert not np.any(r["rthcuten"])


def test_table_digests_are_recorded_for_every_section():
    from woof.core import sas
    assert set(so.table_digests()) == set(sas.TABLE_SECTIONS)
    assert sum(n for _, n in sas.TABLE_SECTIONS.values()) == sas.TB_TOTAL


# ---- the GPU port, word for word ---------------------------------------------------
@requires_gpu
def test_funcphys_tables_are_the_oracle_tables():
    import cupy  # noqa: F401
    rows = so.grade_tables()
    assert all(r["match"] for r in rows.values()), rows


@requires_gpu
def test_stage1_sascnvn_zero_ulp():
    import cupy  # noqa: F401
    rows = so.grade_stage1()
    assert all(r["mismatch"] == 0 for r in rows.values()), rows


@requires_gpu
def test_stage2_old_arw_shalcv_zero_ulp():
    import cupy  # noqa: F401
    rows = so.grade_stage2()
    assert all(r["mismatch"] == 0 for r in rows.values()), rows


@requires_gpu
def test_stage3_cu_sas_zero_ulp():
    import cupy  # noqa: F401
    rows = so.grade_stage3()
    assert all(r["mismatch"] == 0 for r in rows.values()), rows


@requires_gpu
def test_a_planted_one_bit_change_is_caught():
    import cupy  # noqa: F401
    mod = so.planted_module()
    assert sum(r["mismatch"] for r in so.grade_stage1(mod=mod).values()) > 0
    assert sum(r["mismatch"] for r in so.grade_stage3(mod=so.planted_module("driver")).values()) > 0


@requires_gpu
def test_tiling_does_not_move_a_word():
    """The grid is launched in tiles through one workspace; a tile edge
    must not change any column's result."""
    import cupy as cp
    from woof.core import sas
    cases = so.load_cases()
    cases = [c for c in cases if c["nz"] == 45] or cases[:1]
    nz = cases[0]["nz"]
    lev = {k: so._pad([c[k] for c in cases], nz + 1 if k in ("w", "p8w") else nz, np.float32)
           for k in so.CASE_ORDER}
    common = dict(xland=np.array([c["xland"] for c in cases], dtype=np.float32),
                  dt=60.0, stepcu=3, pgcon=0.55, **lev)
    whole, _ = sas.run_columns(**common)
    old = sas.SAS_TILE_COLUMNS
    try:
        sas.SAS_TILE_COLUMNS = 3
        tiled, _ = sas.run_columns(**common)
    finally:
        sas.SAS_TILE_COLUMNS = old
    for k in whole:
        assert cp.array_equal(whole[k].view(cp.uint32), tiled[k].view(cp.uint32)), k

@requires_gpu
def test_stage3_shallow_switch_matches_wrf():
    rows = so.grade_stage3(shalconv=0)
    assert all(r["mismatch"] == 0 for r in rows.values()), rows


@pytest.mark.parametrize("shallow", [0, 1])
def test_bare_namelist_selects_sas_and_its_controls(tmp_path, shallow):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    from woof.physics_registry import stock_callable_class
    text = INPUT_TEXT.replace("cu_physics = 1, 0,", "cu_physics = 94, 94,")
    text = text.replace("cudt = 5, 0,", f"cudt = 5, 0,\n sas_pgcon = 0.3, 0.55,\n shalconv = {shallow}, 1,")
    translated, _ = import_namelists(*_pair(tmp_path, inp=text))
    path = tmp_path / "sas.toml"
    path.write_text(translated)
    exp = load_experiment(path)
    root = exp.domains[0].run
    assert root.cu_physics == 94
    assert root.sas_pgcon == 0.3 and root.shalconv == shallow
    assert root.cudt_minutes == 5
    assert stock_callable_class("cumulus", {"cu_physics": 94}) == "woof.core.sas.GfsSas"


def test_sas_adaptive_step_refused_and_hwrf_only_control_ignored(tmp_path):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    text = INPUT_TEXT.replace("cu_physics = 1, 0,", "cu_physics = 94, 94,")
    imported, receipt = import_namelists(*_pair(tmp_path, inp=text.replace(
        "cudt = 5, 0,", "cudt = 5, 0,\n pert_sas = .true., .false.,")))
    assert "cu_physics = 94" in imported
    assert "HWRF" in str(receipt)
    with pytest.raises(ValueError, match="DT.*STEPCU|adaptive"):
        import_namelists(*_pair(tmp_path, inp=text.replace("time_step = 60,", "time_step = 60,\n use_adaptive_time_step = .true.,")))


@requires_gpu
def test_runtime_module_is_in_the_kernel_manifest():
    from woof.core import sas
    from woof.certify.kernel_manifest import kernel_manifest
    from woof.core.kernels import module_source
    import hashlib
    sas.module()
    row = kernel_manifest()["woof.core.kernels:gfs_sas"]
    assert row["source_sha256"] == hashlib.sha256(module_source("gfs_sas").encode()).hexdigest()
    from woof.core.kernels import module_options
    from woof.wrf_exact import effective_options
    assert row["options"] == list(effective_options(module_options("gfs_sas")))

@pytest.mark.parametrize("value", ["2", "0.5", ".true."])
def test_invalid_shallow_switch_is_refused(tmp_path, value):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    text = INPUT_TEXT.replace("cu_physics = 1, 0,", "cu_physics = 94, 94,")
    text = text.replace("cudt = 5, 0,", f"cudt = 5, 0,\n shalconv = {value}, 1,")
    with pytest.raises(ValueError, match="shalconv"):
        import_namelists(*_pair(tmp_path, inp=text))

def test_bare_sas_namelist_keeps_the_arw_defaults(tmp_path):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    from woof.experiment import load_experiment
    text, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT.replace(
        "cu_physics = 1, 0,", "cu_physics = 94, 94,")))
    path = tmp_path / "bare.toml"; path.write_text(text)
    root = load_experiment(path).domains[0].run
    assert (root.cu_physics, root.sas_pgcon, root.shalconv) == (94, 0.55, 1)


def test_namelist_contract_lists_the_shallow_and_random_trigger_rules():
    from woof.namelist_contract import _value_rules
    rules = _value_rules()
    assert rules[("physics", "shalconv")]["values"] == [0, 1]
    assert rules[("physics", "pert_sas")]["values"] == [False, True]


@requires_gpu
def test_stock_sas_reaches_the_physics_driver_with_the_cudt_clock():
    import cupy as cp
    from test_physics_driver import _full_state
    from woof.core.sas import GfsSas
    state, cfg, driver = _full_state(cu_physics=94, cudt_minutes=1.0)
    assert isinstance(driver.cumulus_callable, GfsSas)
    assert driver.stepcu == 6
    driver.compute(state, cfg)
    held = {key: value.copy() for key, value in driver.cu_rates.items()}
    rate = driver.cu_pratec.copy()
    state.elapsed_seconds = 10.0
    driver.compute(state, cfg)
    for key, value in held.items():
        assert cp.array_equal(driver.cu_rates[key].view(cp.uint32), value.view(cp.uint32)), key
    assert cp.array_equal(driver.cu_pratec.view(cp.uint32), rate.view(cp.uint32))
    assert cp.all(driver.cu_nca == 0)
    # This is the same inventory walk used by forecast finalization.
    from woof.io.restart import _driver_manifest
    manifest = _driver_manifest(driver)
    assert manifest
    for key in ("sas_hbot", "sas_htop"):
        assert key in driver.fields
        assert driver.fields[key].shape == (cfg.ny, cfg.nx)
        assert any(key in str(entry) for entry in manifest)


def test_sas_scratch_is_classified_without_allowing_generic_workspace_state():
    from types import SimpleNamespace
    from woof.io.restart import (_callable_state_check, CUMULUS_CALLABLE_ARRAYS,
                                  CUMULUS_CALLABLE_CONTAINERS, RestartManifestError)
    scratch = SimpleNamespace(array=np.zeros(12, dtype=np.float64))
    _callable_state_check(SimpleNamespace(_sas_workspace=scratch),
                          CUMULUS_CALLABLE_ARRAYS, CUMULUS_CALLABLE_CONTAINERS,
                          "cumulus")
    with pytest.raises(RestartManifestError, match="unclassified"):
        _callable_state_check(SimpleNamespace(_workspace=scratch),
                              CUMULUS_CALLABLE_ARRAYS, CUMULUS_CALLABLE_CONTAINERS,
                              "cumulus")


def test_sas_modes_bind_source_and_checkpoint_identity(monkeypatch):
    import os
    import subprocess
    import sys
    import json
    script = """import json
from woof.core.kernels import module_source
from woof.checkpoint_identity import CUMULUS_ALGORITHM_IDENTITIES
print(json.dumps({'strict': '#define SAS_STRICT 1\\n' in module_source('gfs_sas'),
                  'identity': CUMULUS_ALGORITHM_IDENTITIES[94]}))
"""
    rows = []
    for mode in ("fast", "strict"):
        env = {**os.environ, "GPUWM_NUMERICS": mode, "GPUWM_WRF_EXACT": "0"}
        rows.append(json.loads(subprocess.check_output([sys.executable, "-c", script], env=env)))
    assert [row["strict"] for row in rows] == [False, True]
    assert rows[0]["identity"] != rows[1]["identity"]


def test_sas_controls_are_absent_from_off_scheme_identities():
    import dataclasses
    from woof.config import RunConfig
    from woof.checkpoint_identity import drop_default_cumulus_selectors
    from woof.core.model import _cumulus_scoped_run_fields
    for cu in (0, 1, 3, 5, 6, 16, 96):
        values = dataclasses.asdict(RunConfig(nx=16, ny=16, nz=40, dx=12000., dy=12000.,
                                                    ztop=20000., dt=30., run_seconds=600.,
                                                    moist=True, cu_physics=cu))
        drop_default_cumulus_selectors(values)
        assert "shalconv" not in values
        assert cu not in _cumulus_scoped_run_fields()["shalconv"]


@requires_gpu
def test_sas_workspace_can_be_rebuilt_or_poisoned_without_moving_a_word():
    import cupy as cp
    from woof.core import sas
    cases = so.load_cases()
    cases = [c for c in cases if c["nz"] == 45]
    nz = cases[0]["nz"]
    lev = {k: so._pad([c[k] for c in cases],
                     nz + 1 if k in ("w", "p8w") else nz, np.float32)
           for k in so.CASE_ORDER}
    common = dict(xland=np.array([c["xland"] for c in cases], dtype=np.float32),
                  dt=60.0, stepcu=3, pgcon=0.55, **lev)
    before, workspace = sas.run_columns(**common, diagnostics=True)
    workspace.array.fill(cp.nan)
    after, _ = sas.run_columns(**common, workspace=workspace, diagnostics=True)
    for k in before:
        bits = cp.uint64 if before[k].dtype.itemsize == 8 else cp.uint32
        assert cp.array_equal(before[k].view(bits), after[k].view(bits)), k


def test_sas_persistent_bounds_share_allocation_history_and_memory_inventory():
    from types import SimpleNamespace
    from woof.core.physics_inventory import cumulus_state_layout, CUMULUS_HISTORY_LAYOUT
    from woof.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
    cfg = SimpleNamespace(cu_physics=94, nx=20, ny=10, nz=45)
    layout = cumulus_state_layout(cfg)
    assert layout == {"sas_hbot": ((10,20), "f4"), "sas_htop": ((10,20), "f4")}
    for key, native in (("sas_hbot", "HBOT"), ("sas_htop", "HTOP")):
        assert CUMULUS_HISTORY_LAYOUT[key] == (key, None)
        assert SCHEME_OUTPUT_FIELDS[key].netcdf_name == native
    cfg.cu_physics = 0
    assert not cumulus_state_layout(cfg)


def test_sas_workspace_budget_counts_each_retained_domain_once():
    from types import SimpleNamespace
    from woof.core.preflight import sas_column_workspace_bytes
    from woof.core.sas import workspace_bytes, TB_TOTAL
    runs = [SimpleNamespace(cu_physics=94, nx=20, ny=10, nz=45),
            SimpleNamespace(cu_physics=94, nx=10, ny=10, nz=30),
            SimpleNamespace(cu_physics=0, nx=100, ny=100, nz=50)]
    exp = SimpleNamespace(domains=[SimpleNamespace(run=run) for run in runs])
    assert sas_column_workspace_bytes(exp) == workspace_bytes(45,200)+workspace_bytes(30,100)+8*TB_TOTAL


def test_import_records_a_shallow_control_on_a_domain_that_does_not_read_it(tmp_path):
    from test_namelist_import import _pair, INPUT_TEXT
    from woof.namelist_import import import_namelists
    text = INPUT_TEXT.replace("cudt = 5, 0,", "cudt = 5, 0,\n shalconv = 0, 1,")
    translated, receipt = import_namelists(*_pair(tmp_path, inp=text))
    assert "shalconv = 0" not in translated
    assert "read only by cu_physics = 94" in str(receipt)
