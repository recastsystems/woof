"""The differentiable-engine doors' table, inputs and verdicts (CPU; the GPU windows are proved by the door runs).

THE BREAKAGES THESE HOLD SHUT: a door whose row names an asset key the table does not define (the runner gets a
path that is not there and dies after admission); an objective whose weights do not sum to one (the printed metric
is not a mean); a species read from the wrong scalar slot (the alloc map gives slot numbers, the words are
slot * slot_words further on); an adjoint verdict that passes with no resolved Taylor interval.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import cli


def test_table_rows_doors_and_names_agree():
    table = d.load_table()
    assert set(table["doors"]) == set(d.DOORS)
    assert table["default_row"] in table["rows"]
    for name, row in table["rows"].items():
        keys = set(row["assets"])
        for key in list(row["asset_args"].values()) + list(row["asset_env"].values()):
            assert key in keys, (name, key)
        for door_name in row["doors"]:
            for key in table["doors"][door_name].get("asset_env", {}).values():
                assert key in keys, (name, door_name, key)
        assert row["default_steps"] >= 1 and row["need_free_mib"] > 0
    for metric in table["metrics"].values():
        assert metric["field"] in table["fields"]
    for field in table["tangent_outputs"] + table["sensitivity_inputs"]:
        assert field in table["fields"]
    for key in ("dot_rel", "fd_tangent_err", "taylor_best_err", "zero_nonzero_words", "param_dot_rel"):
        assert key in table["thresholds"]


def test_cli_registers_the_four_doors_in_one_row():
    rows = [names for names, _add in cli._DOORS if "tangent" in names]
    assert rows == [("tangent", "adjoint-test", "sensitivity", "param-grad")]
    for door in d.DOORS:
        args = cli.build_parser(door).parse_args([door, "--out", "x"])
        assert args.handler.__module__ == "woof.hex.ad_doors"


def _runner_lib_map(plan_sha: str) -> dict:
    return {"plan_sha256": plan_sha, "fields": {
        "theta_m": {"canonical": {"alloc": 368, "offset": 0, "shape": [2, 3]},
                    "copies": [{"alloc": 67, "offset": 0}, {"alloc": 368, "offset": 0}]},
        "scalars": {"canonical": {"alloc": 378, "offset": 8, "shape": [3, 2, 3]},
                    "copies": [{"alloc": 62, "offset": 0}, {"alloc": 378, "offset": 8}],
                    "slot_words": 6, "species_slot": {"qv": 0, "qc": 1}},
        "u": {"canonical": {"alloc": 377, "offset": 40, "shape": [2, 4]}, "copies": [{"alloc": 377, "offset": 40}]}}}


def test_field_map_reads_both_schemas_and_species_slots(tmp_path):
    table = d.load_table()
    fmap = d.FieldMap(_runner_lib_map("ab"), "ab", tmp_path / "m.json")
    qc = fmap.resolve(table, "qc")
    assert qc["alloc"] == 378 and qc["offset"] == 8 + 4 * 6 and qc["shape"] == [2, 3]
    assert qc["copies"] == [(62, 24), (378, 32)]
    theta_m = fmap.resolve(table, "theta_m")
    assert theta_m["copies"] == [(67, 0), (368, 0)]
    recorder = {"schema": "ad-record.alloc-map/v1", "plan_sha256": "cd", "species_order": ["qv", "qc"], "fields": {
        "theta_m": {"allocation": 65, "copies": [351], "offset": 0, "shape": [2, 3]},
        "rho": {"allocation": 13, "copies": [367], "offset": 0, "shape": [2, 3]},
        "scalars": {"allocation": 62, "copies": [], "offset": 0, "shape": [2, 2, 3]}}}
    aliases = table["alloc_map_aliases"]["ad-record.alloc-map/v1"]
    other = d.FieldMap(recorder, "cd", tmp_path / "r.json", aliases)
    assert other.resolve(table, "rho_zz")["copies"] == [(13, 0), (367, 0)]   # fields.rho is rho_zz in hex
    assert other.resolve(table, "rho_zz")["alloc"] == 367                    # the pool's copy, not the dycore's
    with pytest.raises(d.AdDoorError, match="not in this plan"):
        other.resolve(table, "rho")
    assert other.resolve(table, "theta_m")["copies"] == [(65, 0), (351, 0)]
    assert other.resolve(table, "qc")["offset"] == 24
    with pytest.raises(d.AdDoorError, match="belongs to plan"):
        d.FieldMap(recorder, "ef", tmp_path / "r.json", aliases)
    with pytest.raises(d.AdDoorError, match="not in this plan"):
        other.resolve(table, "u")


def _geometry():
    lat = np.array([10.0, 20.0, 50.0])
    lon = np.array([-30.0, -25.0, 100.0])
    return {"latCell": lat, "lonCell": lon, "areaCell": np.array([1.0, 3.0, 2.0])}


def test_metric_objective_is_an_area_weighted_mean():
    table = d.load_table()
    fmap = d.FieldMap(_runner_lib_map("ab"), "ab", Path("m.json"))
    table = {**table, "metrics": {**table["metrics"], "tm": {"field": "theta_m", "what": "mean theta_m"}}}
    obj = d.metric_objective(table, fmap, _geometry(), "tm", "1-2", "0,30,-40,0", None)
    assert obj["cells"] == [0, 1] and obj["levels_1based"] == [1, 2]
    assert abs(sum(t[2] for t in obj["terms"]) - 1.0) < 1e-12
    assert [t[1] for t in obj["terms"]] == [0, 1, 3, 4]       # word = k * ncol + cell
    assert obj["terms"][1][2] == pytest.approx(3 * obj["terms"][0][2])
    with pytest.raises(d.AdDoorError, match="no cell"):
        d.metric_objective(table, fmap, _geometry(), "tm", None, "80,85,0,10", None)
    with pytest.raises(d.AdDoorError, match="levels run"):
        d.metric_objective(table, fmap, _geometry(), "tm", "3", None, None)


def test_plan_state_reads_gzip_payloads(tmp_path):
    words = np.arange(8, dtype="<f4")
    payload = tmp_path / "recording" / "p.gz"
    payload.parent.mkdir()
    with gzip.open(payload, "wb") as stream:
        stream.write(words.tobytes())
    (tmp_path / "plan.json").write_text(json.dumps({"allocations": [
        {"id": 5, "initial": {"path": "recording/p.gz", "encoding": "gzip"}}]}))
    state = d.PlanState(tmp_path)
    got = state.field({"alloc": 5, "offset": 8, "words": 6, "shape": [2, 3]})
    assert got.tolist() == [[2, 3, 4], [5, 6, 7]]


def _receipt(errs, w_mx=2.0, rel=1e-8, zero=0):
    rows = [{"h": h, "err": e, "ratio": 1 + e, "fd_quotient": w_mx * (1 + e)} for h, e in zip((1, .3, .1, .03), errs)]
    return {"tangent_dir": {"w_Mx": w_mx, "x_MTw": w_mx, "rel": rel}, "taylor": {"rows": rows},
            "zero_direction": {"nonzero_words": zero}}


def _receipt5(errs, w_mx=2.0):
    rows = [{"h": h, "err": e, "ratio": 1 + e, "fd_quotient": w_mx * (1 + e)}
            for h, e in zip((1, .3, .1, .03, .01), errs)]
    return {"tangent_dir": {"w_Mx": w_mx, "x_MTw": w_mx, "rel": 1e-8}, "taylor": {"rows": rows},
            "zero_direction": {"nonzero_words": 0}}


def test_adjoint_verdict_needs_every_check_and_a_resolved_interval():
    bars = d.load_table()["thresholds"]
    good = d.adjoint_verdict(_receipt([3e-2, 4e-3, 1e-4, 2e-3]), bars)
    # 1e-4 sits 40 and 20 times under both neighbours (a noise crossing) and 2e-3 has only that neighbour: both set
    # aside; the value is the confirmed 4e-3
    assert good["passed"] and good["checks"]["taylor"]["value"] == 4e-3
    assert good["checks"]["taylor"]["set_aside_h"] == [.1, .03]
    assert d.adjoint_verdict(_receipt([1e-6, 5e-6, 9e-6, 2e-6]), bars)["passed"]         # a linear window: flat
    assert not d.adjoint_verdict(_receipt([3e-2, 4e-2, 1e-4, 2e-2]), bars)["passed"]     # one lucky h
    assert not d.adjoint_verdict(_receipt([3e-2, 4e-3, 1e-4, 2e-3], rel=1e-3), bars)["passed"]
    assert not d.adjoint_verdict(_receipt([3e-2, 4e-3, 1e-4, 2e-3], zero=7), bars)["passed"]
    # the zoom globe's d.zg-b rows (window.json taylor.rows): 3.4e-5 between 6.4e-2 and 0.65 was printed as the pass
    zoom = d.adjoint_verdict(_receipt5([1.0226e-2, 2.635e-3, 6.395e-2, 3.366e-5, 0.6475]), bars)
    assert zoom["checks"]["taylor"]["value"] == 2.635e-3 and zoom["checks"]["taylor"]["set_aside_h"] == [.1, .03, .01]
    assert zoom["checks"]["fd"]["set_aside_h"] == [.1, .03, .01]
    # 120 km d120-a: 5.3e-5 at h = 0.01 is 42 times under its only neighbour
    w120 = d.adjoint_verdict(_receipt5([3.261e-3, 1.168e-3, 1.379e-3, 2.220e-3, 5.286e-5]), bars)
    assert w120["checks"]["taylor"]["value"] == 1.168e-3 and w120["checks"]["taylor"]["set_aside_h"] == [.01]
    # every row a noise crossing of the next: nothing confirmed, no pass
    assert not d.adjoint_verdict(_receipt5([1e-1, 1e-3, 1e-1, 1e-5, 1e-1]), bars)["passed"]
    empty = d.adjoint_verdict({}, bars)
    assert not empty["passed"] and empty["checks"]["dot"]["value"] is None


def test_window_command_carries_row_assets_door_args_and_refusals(tmp_path):
    table = d.load_table()
    row = table["rows"][table["default_row"]]
    assets = {key: tmp_path / key for key in row["assets"]}
    for path in assets.values():
        path.write_text("x")
    assets["refuse"].write_text("solar,rlw_taut\n")
    argv, env = d.window_command(table, row, assets, "sensitivity", tmp_path / "hexad", 3, tmp_path / "o",
                                 ["--objective", "j.json"])
    assert argv[1:5] == ["window", "--config", "plan.json", "--steps"] and argv[5] == "3"
    assert argv[argv.index("--kernels") + 1] == str(assets["kernels"])
    assert "--no-tangent" in argv and argv[argv.index("--refuse") + 1] == "solar,rlw_taut"
    assert env["AD_SCHEMES"] == str(assets["bundle"])
    _argv, env = d.window_command(table, row, assets, "param-grad", tmp_path / "hexad", 3, tmp_path / "o", [])
    assert env["AD_SCHEMES"] == str(assets["bundle_params"])
    assets["kernels"].unlink()
    with pytest.raises(d.AdDoorError, match="missing"):
        d.window_command(table, row, assets, "tangent", tmp_path / "hexad", 3, tmp_path / "o", [])


def test_runner_and_assets_refuse_by_name(tmp_path, monkeypatch):
    monkeypatch.delenv("WOOF_HEX_AD_RUNNER", raising=False)
    monkeypatch.delenv("WOOF_HEX_AD_ASSETS", raising=False)
    monkeypatch.setattr(d, "_runner_api", lambda: None)
    monkeypatch.setattr(d, "_hexad", lambda: None)
    monkeypatch.setattr(d.shutil, "which", lambda name: None)
    with pytest.raises(d.AdDoorError, match="no window runner"):
        d.resolve_runner(None)
    with pytest.raises(d.AdDoorError, match="no differentiable-engine asset root"):
        d.resolve_assets(None)
    with pytest.raises(d.AdDoorError, match="not an executable"):
        d.resolve_runner(tmp_path / "missing")


def test_rows_name_the_doors_they_run_and_the_copy_they_read():
    table = d.load_table()
    name, row = d._row(table, None, "param-grad")
    assert name == table["default_row"] and "param-grad" in row["doors"]
    one = {**row, "doors": ["tangent"]}
    with pytest.raises(d.AdDoorError, match="does not run the param-grad door"):
        d._row({**table, "rows": {"r": one}, "default_row": "r"}, "r", "param-grad")
    fmap = d.FieldMap(_runner_lib_map("ab"), "ab", Path("m.json"))
    assert fmap.resolve(table, "theta_m", row)["alloc"] == 368       # the physics pool's copy: the evolving state
    dycore = {**row, "state_copy": "dycore", "holds_physics": True}
    assert fmap.resolve(table, "theta_m", dycore)["alloc"] == 67
    doc = _runner_lib_map("ab")
    doc["fields"]["theta"] = {"canonical": {"alloc": 353, "offset": 0, "shape": [2, 3]}, "copies": []}
    with pytest.raises(d.AdDoorError, match="written by the physics"):
        d.FieldMap(doc, "ab", Path("m.json")).resolve(table, "theta", dycore)


def test_whole_state_dot_is_read_at_the_program_bar_and_only_from_a_runner_that_ran_it():
    """THE BREAKAGE: adjoint-test passing on the objective-direction dot (bar 2e-5) while one field's transpose is
    wrong (seen at 60 km: 2.3e-5 standard, 1.1e-6 whole-state), or a runner without the whole-state dot being
    read as if it had run it."""
    table = d.load_table()
    t = d.dot_bar_in_force(table)
    assert t["rel"] == 4 * 2.0 ** -23 < table["thresholds"]["dot_rel"]
    ran = {"dots": [{"rel_aligned": 7.0e-8, "y_Mx": 113.0, "x_MTy": 112.99999}],
           "self_dot": {"allocations": 113, "words": 168334665}}
    assert d.self_dot_check(ran, t)["pass"] and d.self_dot_check(ran, t)["allocations"] == 113
    assert not d.self_dot_check({**ran, "dots": [{"rel_aligned": 1.1263e-6}]}, t)["pass"]     # the 60 km reading
    old_runner = {"dots": [{"rel_aligned": 3.1e-8}]}                                           # no self_dot block
    check = d.self_dot_check(old_runner, t)
    assert check["value"] is None and not check["pass"]
    spec = d.load_table()["doors"]["adjoint-test"]
    assert spec["self_dot"]["window_env"] == {"AD_DOT_SELF": "1"} and "--seeds" in spec["self_dot_args"]


def test_the_whole_state_line_says_what_was_scored(capsys):
    """THE BREAKAGE: a whole-state dot scored on memory allocations (scratch words weigh as much as a field) printed
    as if it had been scored on the model's fields, or the runner's reason for not scoring the fields dropped on the
    way to the user. The runner says which units it used and why (self_dot.units, units_why); the door repeats it."""
    t = d.dot_bar_in_force(d.load_table())
    dots = [{"rel_aligned": 2.0e-8, "y_Mx": 12.0, "x_MTy": 12.0}]
    fields = d.self_dot_check({"dots": dots, "self_dot": {"allocations": 12, "words": 1, "units": "fields"}}, t)
    assert fields["pass"] and fields["units"] == "fields" and "every model field at the window end" in fields["what"]
    d.print_strict(fields)
    out = capsys.readouterr().out
    assert "(12 model fields; own windows)" in out and "memory allocations" not in out
    why = "plan/alloc-map.json has no boundary 16 (the window end of 15 steps; it holds 15 boundaries)"
    pool = d.self_dot_check({"dots": dots, "self_dot": {"allocations": 105, "words": 1, "units": "pool",
                                                        "units_why": why}}, t)
    assert pool["pass"] and pool["units"] == "pool" and "every state allocation" in pool["what"]
    d.print_strict(pool)
    out = capsys.readouterr().out
    assert "(105 state allocations; own windows)" in out
    assert f"scored on memory allocations, not the model's fields: {why}" in out
    # a receipt of a runner from before the units were named reads as it always did
    old = d.self_dot_check({"dots": dots, "self_dot": {"allocations": 105, "words": 1}}, t)
    d.print_strict(old)
    out = capsys.readouterr().out
    assert old["units"] is None and "(105 state allocations; own windows)" in out and "scored on" not in out


def test_a_field_mapped_by_its_initial_copy_alone_is_not_read_as_the_window_end_state():
    """THE BREAKAGE: on a recorder-made plan the tangent door wrote d_u = 0 everywhere and u = the initial wind as
    'the window end' (zoom globe, 2026-11-09): the recorder maps the edge wind by its start copy only, an allocation
    the forecast never writes again."""
    table = d.load_table()
    doc = {"schema": "ad-record.alloc-map/v1", "plan_sha256": "p", "species_order": ["qv", "qc"], "fields": {
        "theta_m": {"allocation": 67, "copies": [369], "shape": [2, 3], "offset": 0},
        "exner": {"allocation": 68, "copies": [349], "shape": [2, 3], "offset": 0},
        "rho_theta": {"allocation": 56, "copies": [], "shape": [2, 3], "offset": 0},
        "scalars": {"allocation": 62, "copies": [378], "shape": [2, 2, 3], "offset": 0},
        "normal_velocity": {"allocation": 72, "copies": [], "shape": [2, 4], "offset": 0}}}
    fmap = d.FieldMap(doc, "p", Path("alloc-map.json"), table["alloc_map_aliases"]["ad-record.alloc-map/v1"])
    assert fmap.at_window_end(table, "theta_m") and fmap.at_window_end(table, "qv")
    assert not fmap.at_window_end(table, "u") and not fmap.at_window_end(table, "theta")   # theta: not mapped at all
    assert fmap.resolve(table, "theta_m", window_end=True)["alloc"] == 369                 # the model-state copy
    start = fmap.resolve(table, "u")                         # the start copy: a perturbation or an initial gradient
    assert start["alloc"] == 72 and start["copies"] == [(72, 0)]
    with pytest.raises(d.AdDoorError, match="initial"):
        fmap.resolve(table, "u", window_end=True)


def test_a_window_shorter_than_its_recording_reads_the_window_end_boundary_not_the_pool_copy():
    """THE BREAKAGE: the physics pool's copy is filled from the state at the start of each step and made current only
    by the last recorded step, so a window that ends inside its recording read the state one step early: the 60 km
    map's J was the 28-step value for a 29-step window (f52/explain/m60c/RESULT.md (d), accuracy defect 5), and the
    120 km door's 3-step window on its 4-step plan read +24 min, not +36."""
    table = d.load_table()
    doc = {"schema": "ad-record.alloc-map/v1", "plan_sha256": "p", "species_order": ["qv", "qc"], "fields": {
        "theta_m": {"allocation": 67, "copies": [371], "shape": [2, 3], "offset": 0},
        "scalars": {"allocation": 62, "copies": [378], "shape": [2, 2, 3], "offset": 0},
        "normal_velocity": {"allocation": 72, "copies": [], "shape": [2, 4], "offset": 0}},
        "boundaries": {
            "1": {"theta_m": {"allocation": 67, "offset": 0}, "scalars": {"allocation": 62, "offset": 0},
                  "normal_velocity": {"allocation": 72, "offset": 0}},
            "3": {"theta_m": {"allocation": 417, "offset": 4096}, "scalars": {"allocation": 393, "offset": 0}},
            "4": {"theta_m": {"allocation": 402, "offset": 8192}, "scalars": {"allocation": 393, "offset": 512},
                  "normal_velocity": {"allocation": 391, "offset": 64}}}}
    fmap = d.FieldMap(doc, "p", Path("alloc-map.json"), table["alloc_map_aliases"]["ad-record.alloc-map/v1"])
    assert fmap.resolve(table, "theta_m", window_end=True)["alloc"] == 371      # no window set: the old reading
    fmap.window_steps = 3                                                       # 3 steps of a 4-step recording
    assert fmap.end_boundary() == 4
    thm = fmap.resolve(table, "theta_m", window_end=True)
    assert (thm["alloc"], thm["offset"], thm["read_shape"]) == (402, 8192, [2, 3])
    assert "boundary 4" in thm["place"]
    assert fmap.resolve(table, "theta_m")["alloc"] == 371                       # not a window-end read: unchanged
    assert fmap.resolve(table, "theta_m")["copies"] == [(67, 0), (371, 0)]      # the perturbation's copies
    qc = fmap.resolve(table, "qc", window_end=True)                             # slot 1 within boundary 4's scalars
    assert (qc["alloc"], qc["offset"]) == (393, 512 + 4 * 2 * 3)
    assert fmap.at_window_end(table, "u")                                       # the map says where u is at 4
    assert fmap.resolve(table, "u", window_end=True)["alloc"] == 391
    obj = d.metric_objective({**table, "metrics": {**table["metrics"], "tm": {"field": "theta_m", "what": "tm"}}},
                             fmap, _geometry(), "tm", "2", None, None)
    assert {t[0] for t in obj["terms"]} == {402}
    assert [t[1] for t in obj["terms"]] == [8192 // 4 + 3 + c for c in range(3)]
    assert obj["window_steps"] == 3 and "boundary 4" in obj["read_at"]
    fmap.window_steps = 2                                                       # boundary 3 does not name u
    assert not fmap.at_window_end(table, "u")
    with pytest.raises(d.AdDoorError, match="boundary 3"):
        fmap.resolve(table, "u", window_end=True)
    fmap.window_steps = 4                                                       # the recording's end: pool is current
    assert fmap.end_boundary() is None
    assert fmap.resolve(table, "theta_m", window_end=True)["alloc"] == 371
    with pytest.raises(d.AdDoorError, match="initial"):
        fmap.resolve(table, "u", window_end=True)


def test_a_limited_area_plan_reads_its_padded_fields_on_the_mesh_columns(tmp_path, monkeypatch):
    """THE BREAKAGE: on the 3 km storm box (2026-11-10) every door died: the dycore arrays carry MPAS's padding
    column (59 x 38,279 for 38,278 cells) and the physics pool's copies do not, so the recorder found no copy (no
    field readable at the window end) and the bump broadcast (59, 38278) against (59, 38279)."""
    table = d.load_table()
    monkeypatch.setenv(d.OBJECTIVES_ENV, str(tmp_path / "cache" / "objectives"))
    cells, edges = 3, 7
    rng = np.random.default_rng(2)
    root = rng.normal(300.0, 1.0, (2, cells + 1)).astype("<f4")
    qv = rng.normal(0.01, 0.001, (2, 2, cells + 1)).astype("<f4")
    payloads = {67: root.tobytes(), 369: np.ascontiguousarray(root[:, :cells]).tobytes() + b"\0" * 24,
                62: qv.tobytes(), 378: np.ascontiguousarray(qv[..., :cells]).tobytes(),
                56: np.zeros((2, cells + 1), "<f4").tobytes(), 90: np.zeros((2, cells), "<f4").tobytes()}
    (tmp_path / "p").mkdir()
    allocs = []
    for i, data in payloads.items():
        (tmp_path / "p" / f"{i}.bin").write_bytes(data)
        allocs.append({"id": i, "bytes": len(data), "initial": {"path": f"p/{i}.bin", "bytes": len(data)}})
    (tmp_path / "plan.json").write_text(json.dumps({"allocations": allocs}))
    doc = {"schema": "ad-record.alloc-map/v1", "plan_sha256": "p", "species_order": ["qv", "qc"], "fields": {
        "theta_m": {"allocation": 67, "copies": [], "shape": [2, cells + 1], "offset": 0},
        "rho_theta": {"allocation": 56, "copies": [], "shape": [2, cells + 1], "offset": 0},
        "scalars": {"allocation": 62, "copies": [], "shape": [2, 2, cells + 1], "offset": 0}}}
    fmap = d.FieldMap(doc, "p", Path("alloc-map.json"), table["alloc_map_aliases"]["ad-record.alloc-map/v1"],
                      mesh={"cells": cells, "edges": edges}, plan=tmp_path)
    assert fmap.padded_copies_how == "made" and len(list((tmp_path / "cache" / "fieldmaps").glob("*.json"))) == 1
    # THE COST: the search reads every recorded payload (17.6 s on the 3 km storm example) on every command. A later
    # command on the same plan takes what was found from the door's cache and reads no payload; another plan, map or
    # mesh searches again.
    read = []
    monkeypatch.setattr(d.PlanState, "raw", lambda self, alloc: read.append(alloc) or payloads[alloc])
    again = d.FieldMap(doc, "p", Path("alloc-map.json"), table["alloc_map_aliases"]["ad-record.alloc-map/v1"],
                       mesh={"cells": cells, "edges": edges}, plan=tmp_path)
    assert again.padded_copies_how == "kept" and read == [] and again.found == fmap.found
    assert {k: (v["alloc"], v["copies"], v["copy_shapes"]) for k, v in again.fields.items()} == \
        {k: (v["alloc"], v["copies"], v["copy_shapes"]) for k, v in fmap.fields.items()}
    other = d.FieldMap({**doc, "note": "another map"}, "p", Path("alloc-map.json"),
                       table["alloc_map_aliases"]["ad-record.alloc-map/v1"], mesh={"cells": cells, "edges": edges},
                       plan=tmp_path)
    assert other.padded_copies_how == "made" and read
    monkeypatch.undo()
    fmap = again
    thm = fmap.resolve(table, "theta_m", window_end=True)
    assert thm["alloc"] == 369 and thm["shape"] == [2, cells] and thm["read_shape"] == [2, cells]
    assert thm["copies"] == [(67, 0), (369, 0)] and thm["copy_shapes"] == [[2, cells + 1], [2, cells]]
    qc = fmap.resolve(table, "qc", window_end=True)          # slot 1: words per slot follow each copy's own shape
    assert qc["copies"] == [(62, 4 * 2 * (cells + 1)), (378, 4 * 2 * cells)]
    state = d.PlanState(tmp_path)
    assert np.array_equal(state.field(thm), root[:, :cells])
    assert np.array_equal(state.field(fmap.resolve(table, "qc")), qv[1, :, :cells])
    assert not fmap.at_window_end(table, "rho_theta")        # a constant root proves no copy (90 is all zeros)
    assert fmap.resolve(table, "rho_theta")["shape"] == [2, cells]
    obj = d.metric_objective({**table, "metrics": {**table["metrics"], "tm": {"field": "theta_m", "what": "tm"}}},
                             fmap, {"latCell": np.zeros(cells), "lonCell": np.zeros(cells), "areaCell": np.ones(cells)},
                             "tm", "2", None, None)
    assert [t[:2] for t in obj["terms"]] == [[369, cells + c] for c in range(cells)]   # stride of the copy read


def test_map_scale_and_level_are_not_set_by_one_extreme_cell():
    """THE BREAKAGE: one cell 4 to 50 times the field set the colour scale and the level, so the sensitivity map was
    blank but for that cell (240 km and 120 km globes, 2026-11-09)."""
    import numpy as np
    rng = np.random.default_rng(1)
    field = np.abs(rng.normal(1.0, 0.1, 5000))
    spiked = field.copy()
    spiked[7] = 40.0
    plain = d.map_scale(field)
    assert plain["rule"] == "the peak" and plain["cells_above_scale"] == 0 and plain["top"] == field.max()
    chosen = d.map_scale(spiked)
    assert chosen["peak"] == 40.0 and 1.0 < chosen["top"] < 2.0 and chosen["cells_above_scale"] >= 1
    few = np.zeros(5000)
    few[:50] = 3.0                                    # a small region: fewer than 200 cells carry the field
    assert d.map_scale(few)["top"] == 3.0
    assert d.map_scale(np.zeros(10))["top"] == 1.0
    levels = np.tile(field[:1000], (4, 1)) * np.array([[0.1], [1.0], [0.5], [0.2]])
    levels[3, 5] = 500.0                              # the extreme cell sits on level 4; the field is largest on level 2
    assert int(np.argmax(np.abs(levels).max(axis=1))) + 1 == 4
    levels[3, 5] = 20.0
    assert d.map_level(levels) == 2


def test_the_bump_and_metric_default_to_a_limited_area_mesh_s_own_centre():
    """THE BREAKAGE: on the 3 km storm box (40.6N 91.3W) a plain adjoint-test was refused 'the metric region (cells
    within 1500 km of 20.4N -26.2E) holds no cell of this mesh' and a plain tangent ran on a bump of about exp(-34):
    the global default centre lies outside every limited-area mesh (doors6003 round 5, 2026-11-11)."""
    import numpy as np
    from woof.hex import ad_doors as dd
    g = np.linspace(-2.5, 2.5, 21)
    box = {"latCell": (40.6 + g[:, None] + 0 * g[None, :]).ravel(), "lonCell": (-91.3 + 0 * g[:, None] + g[None, :]).ravel()}
    at = dd.resolve_at(None, box)
    la, lo = (float(x) for x in at.split(","))
    assert abs(la - 40.6) < 0.3 and abs(lo + 91.3) < 0.3, at
    cells, _ = dd.select_cells(box, None, f"{at},1500")
    assert cells.size == box["latCell"].size
    # a globe keeps the published default, and a given --at is kept as typed
    lat = np.degrees(np.arcsin(np.linspace(-0.99, 0.99, 41)))[:, None] + 0 * np.arange(72)[None, :]
    lon = 0 * lat + np.arange(-180, 180, 5)[None, :]
    globe = {"latCell": lat.ravel(), "lonCell": lon.ravel()}
    assert dd.resolve_at(None, globe) == dd.DEFAULT_AT
    assert dd.resolve_at("10,20", box) == "10,20"


def test_the_map_date_comes_from_the_plan_when_its_init_file_is_not_on_this_machine(tmp_path):
    """THE BREAKAGE: a plain 3 km sensitivity on box-e5 refused 'the plan names no start time' before the card ran
    (doors6003 round 5, job r5-d3-a3, 2026-11-11): from-state.json and record-receipt.json name the init file by a
    a test host path the box does not have, and plan.json's physics.state_xtime (the recorded clock) was never read."""
    from types import SimpleNamespace
    from woof.hex import ad_doors as dd
    plan = tmp_path / "pkg-lam-r300-e4"
    plan.mkdir()
    gone = "/not-on-this-box/init.nc"
    (plan / "alloc-map.json").write_text("{}")
    (plan / "from-state.json").write_text(json.dumps({"lam": "/x/r300.env", "init": gone}))
    (plan / "record-receipt.json").write_text(json.dumps({"from_state": {"init": gone}}))
    (plan / "plan.json").write_text(json.dumps({"physics": {"dt_seconds": 15, "state_xtime": "2023-03-31_18:00:00"}}))
    fmap = SimpleNamespace(doc={"plan_sha256": "x"}, source=plan / "alloc-map.json")
    assert dd._valid_time(fmap, 0, 15.0) == "2023-03-31T18:00:00"
    assert dd._valid_time(fmap, 4, 15.0) == "2023-03-31T18:01:00"
    # a plan that names no time anywhere is still refused, not dated wrongly
    (plan / "plan.json").write_text(json.dumps({"physics": {"dt_seconds": 15}}))
    with pytest.raises(dd.AdDoorError, match="names no start time"):
        dd._valid_time(fmap, 0, 15.0)


def _render_with_fakes(tmp_path, monkeypatch, geometry, values):
    """render_cell_map with numpy standing in for cupy and the Rust writer and renderer replaced by recorders."""
    import sys
    import types
    shim = types.ModuleType("cupy")
    shim.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    shim.asnumpy = np.asarray
    rustwx = types.ModuleType("woof.rustwx")
    seen = {}

    def renderer(_exe, data, *, out_dir, bounds, **kw):
        seen["bounds"] = bounds
        (Path(out_dir) / "x").mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "x" / "map.png").write_bytes(b"png")
        return 0

    rustwx.run_cf_renderer = renderer
    ocean = types.ModuleType("woof.hex.ocean_products")
    ocean.physical_renderer = lambda: Path("rw_wrfbatch")
    monkeypatch.setitem(sys.modules, "cupy", shim)
    monkeypatch.setitem(sys.modules, "woof.rustwx", rustwx)
    monkeypatch.setitem(sys.modules, "woof.hex.ocean_products", ocean)

    def plane(path, cp, lat, lon, valid, variable, units, data, source):
        seen.update(lat=np.asarray(lat), lon=np.asarray(lon), plane=np.asarray(data, np.float64))

    monkeypatch.setattr(d, "write_cf_plane", plane)
    d.render_cell_map(tmp_path, geometry, values, variable="v", title="t", units="K/K", valid="2023-03-31T18:00:00")
    return seen


def test_a_limited_area_map_is_drawn_at_the_mesh_spacing_over_the_mesh_and_masked_outside(tmp_path, monkeypatch):
    # a 0.1-degree patch of 41 x 41 cells centred on 40.6N 91.3W (the shape of the 3 km storm box, coarser)
    la, lo = np.meshgrid(40.6 + 0.1 * np.arange(-20, 21), -91.3 + 0.13 * np.arange(-20, 21), indexing="ij")
    geometry = {"latCell": la.ravel(), "lonCell": lo.ravel(), "areaCell": np.ones(la.size)}
    values = np.exp(-((la - 40.6) ** 2 + (lo + 91.3) ** 2)).ravel() * 1e-6
    seen = _render_with_fakes(tmp_path, monkeypatch, geometry, values)
    lat, lon, plane = seen["lat"], seen["lon"], seen["plane"]
    step = float(np.diff(lat[:, 0]).mean())
    assert 0.08 <= step <= 0.12                                   # the mesh's own spacing, not 1 degree
    assert lat.min() > 38.0 and lat.max() < 43.0                   # the grid is the mesh, not the globe
    assert lon.min() > -95.0 and lon.max() < -87.0
    w, e, s, n = map(float, seen["bounds"].split(","))
    assert abs(s - 38.6) < 0.01 and abs(n - 42.6) < 0.01 and abs(w + 93.9) < 0.01 and abs(e + 88.7) < 0.01
    assert np.isnan(plane[0, :]).all() and np.isnan(plane[:, 0]).all()   # the padding row and column lie outside
    inside = np.isfinite(plane)
    assert inside.sum() > 0.9 * la.size
    assert np.nanmax(plane) == pytest.approx(values.max() / 1e-6, rel=1e-3)   # drawn in units of 1e-6


def test_a_global_map_keeps_its_one_degree_globe(tmp_path, monkeypatch):
    k = np.arange(5000) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * k / 5000))
    lon = (np.degrees(np.pi * (1 + 5 ** 0.5) * k) + 180.0) % 360.0 - 180.0
    geometry = {"latCell": lat, "lonCell": lon, "areaCell": np.ones(lat.size)}
    values = np.exp(-((lat - 50.0) ** 2 + (lon - 10.0) ** 2) / 200.0)
    seen = _render_with_fakes(tmp_path, monkeypatch, geometry, values)
    assert seen["plane"].shape == (181, 360) and np.isfinite(seen["plane"]).all()
    assert seen["lat"][0, 0] == -90.0 and seen["lat"][-1, 0] == 90.0
    grid = d.map_grid(np, geometry)
    assert grid["kind"] == "global" and grid["outside"] is None


def test_a_compact_field_on_a_global_mesh_is_drawn_on_the_whole_globe(tmp_path, monkeypatch):
    """A 120 km field that is zero over most of the globe with a bump over Iowa (3e-6 K/K) was drawn as one flat peach
    colour with no bump: render_cell_map cropped the view to the bump padded 15 degrees (-119.4,-61.7,15.2,66.6), and
    the Rust renderer draws such a view in its geographic projection as one flat fill, whatever the field (the same
    view of a 120 km orography field is flat too; the whole globe and a Lambert view of the same plane draw the bump).
    The run that showed it is not part of the release."""
    k = np.arange(5000) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * k / 5000))
    lon = (np.degrees(np.pi * (1 + 5 ** 0.5) * k) + 180.0) % 360.0 - 180.0
    geometry = {"latCell": lat, "lonCell": lon, "areaCell": np.ones(lat.size)}
    values = 3e-6 * np.exp(-(d.great_circle_km(41.0, -90.5, lat, lon) / 700.0) ** 2)
    seen = _render_with_fakes(tmp_path, monkeypatch, geometry, values)
    assert seen["bounds"] == "-180,180,-89,89"
    assert seen["plane"].shape == (181, 360) and np.nanmax(seen["plane"]) > 2.0   # the bump is in the plane, in 1e-6
