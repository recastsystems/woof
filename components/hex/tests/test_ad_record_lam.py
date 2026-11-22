# SPDX-License-Identifier: Apache-2.0
"""A limited-area mesh records through the same recorder as a global one, with the mesh kind as data (no card).

THE BREAKAGE (2026-11-13): ``woof.hex.ad.record`` ran the forecast with ``--mesh --grid --init`` only, so a limited-area
mesh got no boundaries, static file or contract, and the door's ``sensitivity --mesh <limited-area grid> --init
<start>`` could not record a plan from an install; a 3 km storm plan could only be made by a separate chain on rented
cards. The case (the regional door's work directory, or a LAM_* case file) now gives the recording what it needs.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad.record import chain, env, lam

netCDF4 = pytest.importorskip("netCDF4")


def _grid(path: Path, limited: bool) -> Path:
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("nCells", 4)
        v = ds.createVariable("bdyMaskCell", "i4", ("nCells",))
        v[:] = [0, 0, 1, 7] if limited else [0, 0, 0, 0]
    return path


def _case_files(root: Path) -> dict:
    mesh = root / "mesh"
    mesh.mkdir(parents=True)
    grid = _grid(mesh / "box-cull.grid.nc", True)
    for name in ("box-cull.static.nc", "mesh-rows.json", "q6.96.4.n40w91.ugwp_oro_data.nc"):
        (mesh / name).write_bytes(b"x")
    (root / "init").mkdir()
    (root / "init" / "init.nc").write_bytes(b"x")
    (root / "lbc").mkdir()
    (root / "lbc" / "lbc.2024-05-21_15.00.00.nc").write_bytes(b"x")
    (root / "contract").mkdir()
    return {"grid": grid, "mesh": mesh}


def test_a_case_file_gives_the_limited_area_words(tmp_path):
    f = _case_files(tmp_path)
    case = tmp_path / "box.env"
    case.write_text("\n".join([
        "# a 6 km cut", "LAM_MESH_ROW=q6.96.4.n40w91", f"LAM_GRID={f['grid']}", f"LAM_STATIC={f['mesh']}/box-cull.static.nc",
        f"LAM_INIT={tmp_path}/init/init.nc", f"LAM_LBC_DIR={tmp_path}/lbc", "LAM_START=2024-05-21_15:00:00", "LAM_DT=30",
        f"LAM_ROWS={f['mesh']}/mesh-rows.json", f"LAM_CONTRACT={tmp_path}/contract", "LAM_HISTORY_MINUTES=15"]) + "\n")
    got = lam.read_case(case)
    assert got["row"] == "q6.96.4.n40w91" and got["dt"] == 30.0 and got["form"] == "case-file"
    assert lam.forecast_words(got) == ["--static", f"{f['mesh']}/box-cull.static.nc", "--lbc-dir", f"{tmp_path}/lbc",
                                       "--start-time", "2024-05-21_15:00:00", "--history-every-minutes", "15"]
    assert lam.forecast_env(got) == {"WOOF_HEX_MESH_ROWS": f"{f['mesh']}/mesh-rows.json",
                                     "WOOF_HEX_REGIONAL_CONTRACT_DIR": f"{tmp_path}/contract"}
    assert lam.is_limited_area(f["grid"]) and not lam.is_limited_area(_grid(tmp_path / "globe.grid.nc", False))


def test_a_case_that_travelled_names_its_files_relative_to_itself(tmp_path):
    f = _case_files(tmp_path)
    (f["mesh"] / "mesh-rows.json").write_text(json.dumps({"rows": [
        {"name": "p6", "kind": "generated-parent-grid", "grid": "parent.grid.nc"},
        {"name": "q6.96.4.n40w91", "kind": "generated-cull", "grid": "box-cull.grid.nc", "static": "box-cull.static.nc",
         "cull_receipt": None}]}))
    (tmp_path / "case.env").write_text("\n".join([
        "LAM_MESH_ROW=q6.96.4.n40w91", "LAM_GRID=mesh/box-cull.grid.nc", "LAM_STATIC=mesh/box-cull.static.nc",
        "LAM_INIT=init/init.nc", "LAM_LBC_DIR=lbc", "LAM_START=2024-05-21_15:00:00", "LAM_DT=15",
        "LAM_ROWS=mesh/mesh-rows.json", "LAM_CONTRACT=contract"]) + "\n")
    case = lam.read_case(tmp_path / "case.env")
    assert case["grid"] == str(f["grid"].resolve()) and case["lbc_dir"] == str((tmp_path / "lbc").resolve())
    local = lam.localize_rows(case, tmp_path / "work")
    rows = json.loads(Path(local["rows"]).read_text())["rows"]
    assert rows[0]["grid"] == str((f["mesh"] / "parent.grid.nc").resolve())
    assert rows[1]["static"] == str((f["mesh"] / "box-cull.static.nc").resolve()) and rows[1]["cull_receipt"] is None
    assert lam.forecast_env(local)["WOOF_HEX_MESH_ROWS"] == str(tmp_path / "work" / "mesh-rows.json")
    # a rows file whose names are absolute is used as it is
    assert lam.localize_rows(local | {"rows": local["rows"]}, tmp_path / "work2") == local | {"rows": local["rows"]}


def test_a_regional_work_dir_is_found_from_the_init_inside_it(tmp_path):
    f = _case_files(tmp_path)
    (tmp_path / "regional-plan.json").write_text(json.dumps({
        "start_time": "2024-05-21_15:00:00", "history_every_minutes": 15,
        "paths": {"mesh": str(f["mesh"]), "init": str(tmp_path / "init" / "init.nc"),
                  "contract": str(tmp_path / "contract")}}))
    (f["mesh"] / "box.point-generate.json").write_text(json.dumps({"cull": {
        "row": "q6.96.4.n40w91", "grid": str(f["grid"]), "static": str(f["mesh"] / "box-cull.static.nc"),
        "lbc_dir": str(f["mesh"] / "box-cull.lbc")}}))   # the boundary stage wrote beside the mesh: lbc/ is taken
    got = lam.read_case(tmp_path / "init" / "init.nc")
    assert got["form"] == "regional-work-dir" and got["row"] == "q6.96.4.n40w91"
    assert got["lbc_dir"] == str(tmp_path / "lbc") and got["oro"] == str(f["mesh"] / "q6.96.4.n40w91.ugwp_oro_data.nc")


def test_a_case_without_its_contract_or_boundaries_is_refused_by_name(tmp_path):
    f = _case_files(tmp_path)
    case = tmp_path / "box.env"
    words = ["LAM_MESH_ROW=r", f"LAM_GRID={f['grid']}", f"LAM_STATIC={f['mesh']}/box-cull.static.nc",
             f"LAM_INIT={tmp_path}/init/init.nc", f"LAM_LBC_DIR={tmp_path}/lbc", "LAM_START=2024-05-21_15:00:00",
             f"LAM_ROWS={f['mesh']}/mesh-rows.json", f"LAM_CONTRACT={tmp_path}/no-contract"]
    case.write_text("\n".join(words) + "\n")
    with pytest.raises(lam.CaseError, match="contract deck"):
        lam.read_case(case)
    case.write_text("\n".join(words[:-1]) + "\n")
    with pytest.raises(lam.CaseError, match="LAM_CONTRACT"):
        lam.read_case(case)
    with pytest.raises(lam.CaseError, match="regional-plan.json"):
        lam.read_case(tmp_path / "init" / "init.nc")


def test_the_recorder_adds_the_case_to_the_forecast_and_keeps_the_scrub_tables(tmp_path):
    case = {"static": "s.nc", "lbc_dir": "lbc", "start_time": "2024-05-21_15:00:00", "history_minutes": 15.0,
            "rows": "rows.json", "contract": "contract"}
    o = types.SimpleNamespace(mesh="q6", suite="hrrrv5", init="i.nc", grid="g.nc", hours=0.25, dt=30, numerics="fast",
                              oro=None, forecast_extra=[], lam=case)
    words = chain.forecast_args(o)
    assert words[words.index("--lbc-dir") + 1] == "lbc" and "--static" in words and "--start-time" in words
    o.lam = None
    assert "--lbc-dir" not in chain.forecast_args(o)
    assert lam.RETAIN_SMALL_UPLOADS["limited-area"] >= 800 > lam.RETAIN_SMALL_UPLOADS["global"]


def test_record_refuses_a_limited_area_grid_without_its_case_before_any_card(tmp_path):
    grid = _grid(tmp_path / "box.grid.nc", True)
    done = subprocess.run([sys.executable, "-m", "woof.hex.ad.record", "record", "--mesh", "box", "--init", str(grid),
                           "--grid", str(grid), "--dt", "30", "--hours", "0.25", "--out", str(tmp_path / "pkg"),
                           "--runner", sys.executable], capture_output=True, text=True)
    assert done.returncode == 3 and "limited-area mesh" in done.stderr and "--lam" in done.stderr


def _tool(name: str):
    if str(env.TOOLS) not in sys.path:   # the sealer scripts import their siblings, as when they run as scripts
        sys.path.insert(0, str(env.TOOLS))
    spec = importlib.util.spec_from_file_location(name, env.TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sealer_visits_steps_in_step_order_past_99():
    names = [f"step_{s:02d}" for s in (2, 11, 100, 99, 1)]
    want = ["step_01", "step_02", "step_11", "step_99", "step_100"]
    assert sorted(names, key=_tool("seal_phys_stage3").block_key) == want
    assert sorted(names, key=_tool("seal_phys_stage2").block_key) == want


def test_stage3_refuses_steps_without_boundary_state_and_names_a_prefix_run():
    s3 = _tool("seal_phys_stage3")
    with pytest.raises(SystemExit, match="no boundary state"):
        s3.require_boundary_state([1, 2, 3], {1: {5: 1}, 2: {}, 3: {5: 1}})
    events = [{"event": "calendar", "step": s} for s in range(1, 61)]
    assert s3.name_run_steps({}, events, list(range(1, 30))) == {"run_steps": 60}
    assert s3.name_run_steps({}, events, list(range(1, 61))) == {}


def test_stage2_has_the_limited_area_scrub_rule():
    text = (env.TOOLS / "seal_phys_stage2.py").read_text()
    assert "SCRUB_SITE = 'cuda_solve_region_v841.py:443'" in text and "solve_region_scrub_descriptors" in text


def test_the_door_hands_a_limited_area_row_its_case(tmp_path, monkeypatch):
    f = _case_files(tmp_path)
    (tmp_path / "regional-plan.json").write_text(json.dumps({
        "start_time": "2024-05-21_15:00:00", "paths": {"mesh": str(f["mesh"]), "init": str(tmp_path / "init" / "init.nc"),
                                                       "contract": str(tmp_path / "contract")}}))
    (f["mesh"] / "box.point-generate.json").write_text(json.dumps({"cull": {
        "row": "q6.96.4.n40w91", "grid": str(f["grid"]), "static": str(f["mesh"] / "box-cull.static.nc"),
        "lbc_dir": str(tmp_path / "lbc")}}))
    row = dict(d.load_table()["rows"]["hrrrv5-any"], mesh="box-cull", dt_s=30, mesh_kind="limited-area")
    seen = {}

    def run(argv, stdout=None, stderr=None, env=None, **kw):
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.delenv(d.RECORDER_ENV, raising=False)
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    args = types.SimpleNamespace(grid=str(f["grid"]), oro=None, steps=2, out=str(tmp_path / "out"),
                                 init=str(tmp_path / "init" / "init.nc"), card="0", cards=None, runner=None, case=None)
    with pytest.raises(d.AdDoorError, match="recording from"):
        d.record_plan(args, tmp_path, row)
    argv = seen["argv"]
    assert argv[argv.index("--lam") + 1] == str(tmp_path)
    assert argv[argv.index("--mesh") + 1] == "q6.96.4.n40w91"          # the cull's registered row
    assert argv[argv.index("--oro") + 1].endswith("q6.96.4.n40w91.ugwp_oro_data.nc")
    # a global row records as before, with no case
    assert d.limited_area_case(types.SimpleNamespace(case=None, init="x"), {"mesh_kind": "global"}) is None
    # a limited-area row whose init lies in no regional work dir is refused, naming --case
    lone = tmp_path / "lone.nc"
    lone.write_bytes(b"x")
    with pytest.raises(d.AdDoorError, match="--case"):
        d.limited_area_case(types.SimpleNamespace(case=None, init=str(lone.parent.parent / "x" / "init.nc")),
                            {"mesh_kind": "limited-area"})


def test_the_storm_example_is_a_shipped_row_whose_files_are_pinned():
    from woof.hex.ad import assets
    table, pins = d.load_table(), assets.load_pins()
    row = table["rows"]["hrrrv5-q3.96.38278.n40.60w91.30"]
    assert row["mesh_kind"] == "limited-area" and row["dt_s"] == 15.0 and row["need_free_mib"] == 93000
    holds = assets.pinned_holds(pins)
    for rel in (row["example_plan"], row["assets"]["grid"], row["assets"]["oro"], row["example_case"]):
        assert any((rel + "/").startswith(h) or rel.startswith(h) for h in holds), rel
    assert table["example_plan_libraries"][row["example_plan"]] == \
        table["example_plan_libraries"]["plans/hrrrv5-x1.40962-20230331T12-e4"]


def test_a_door_fetches_only_the_example_it_reads():
    """THE BREAKAGE: every pinned file was fetched on first use, so a 120 km user downloaded the storm box too."""
    from woof.hex.ad import assets
    pins = assets.load_pins()
    kinds = {n: r["kind"] for n, r in pins["assets"].items()}
    table = d.load_table()
    plain = types.SimpleNamespace(plan=None, init=None)
    g120 = assets.wanted(pins, d.example_reads(table["rows"]["hrrrv5-x1.40962"], plain))
    s3 = assets.wanted(pins, d.example_reads(table["rows"]["hrrrv5-q3.96.38278.n40.60w91.30"], plain))
    assert all("q3.96.38278" not in n for n in g120) and any("x1.40962" in n for n in g120)
    assert all("x1.40962" not in n for n in s3) and sum("q3.96.38278" in n for n in s3) == 3   # mesh, plan 1 and 2
    row_files = sorted(n for n, k in kinds.items() if k == "ad-row")
    assert set(row_files) <= set(g120) and set(row_files) <= set(s3)
    own = assets.wanted(pins, d.example_reads(table["rows"]["hrrrv5-q3.96.38278.n40.60w91.30"],
                                              types.SimpleNamespace(plan=None, init=Path(d.EXAMPLE_INIT))))
    assert any(n.startswith("ad-case-") for n in own) and not any(n.startswith("ad-plan-") for n in own)
    assert assets.wanted(pins, None) == sorted(pins["assets"])
