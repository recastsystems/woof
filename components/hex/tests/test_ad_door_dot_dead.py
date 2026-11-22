# SPDX-License-Identifier: Apache-2.0
"""The strict self dot's left-out memory (fold 1's AD_DOT_DEAD opt-in) is on by default in the adjoint-test door.

THE BREAKAGE THESE HOLD SHUT: the published 120 km recipe leaves 9,337,888 bytes of limiter scratch out of the strict
self dot (AD_DOT_DEAD) with a proof run (AD_DOT_DEAD_PROOF). The door table kept both as runner environment, so every
120 km door window was handed the literal "{L}/lists/dead120-tail-s3.json" and the door could neither use the list
on its self-dot window nor prove it; a masked dot with no proof must never read as a pass (box-d, 2026-11-11).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from woof.hex import ad_doors as d
from woof.hex.ad import rows


def _generator():
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def _recipe(bundle, env=None, args=()):
    return {"env": {"AD_SCHEMES": "{L}/bundle/" + bundle, "AD_TAPE_PACK": "1", **(env or {})},
            "args": [["kernels", "{L}/kset/k/kernels.json"], ["cubins", "{L}/kset/k/cubins"],
                     ["schemes", "{L}/aux/s.json"], ["tempo-cubin", "{L}/aux/t.cubin"], ["fct-frozen", "1"],
                     ["refuse", "a,b"], *args]}


def _table():
    manifest = {"id": "T", "recipes": {
        "480": _recipe("gen-A"), "480-best": _recipe("gen-B9", args=[["tape-gib", "2"]]),
        "120": _recipe("gen-B9", {"AD_TEMPO_ARENA_GIB": "8", "AD_DOT_SELF": "1",
                                  "AD_DOT_DEAD": "{L}/lists/dead120-tail-s3.json",
                                  "AD_DOT_DEAD_PROOF": "{L}/proof/g1-120D"}, [["tape-gib", "6"]])}}
    return _generator().build(manifest, d.load_table())["table"]


def test_the_left_out_memory_is_the_classes_self_dot_entry_never_runner_environment():
    table = _table()
    for suite in ("hrrrv5", "hrrrv5-exact"):
        cls = rows.window_class(table["suites"][suite], 40962)
        assert "AD_DOT_DEAD" not in cls["window_env"] and "AD_DOT_DEAD_PROOF" not in cls["window_env"]
        assert cls["self_dot_dead"]["list"] == "latest/lists/dead120-tail-s3.json"
        assert cls["self_dot_dead"]["recipe_proof"] == "latest/proof/g1-120D"
        assert not any("{L}" in str(v) for v in cls["window_env"].values())
    # a class whose recipe leaves nothing out carries no entry
    assert "self_dot_dead" not in rows.window_class(table["suites"]["hrrrv5"], 2562)


def test_a_made_120_km_row_carries_the_left_out_memory(tmp_path, monkeypatch):
    table = _table()
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(rows.MESH_PATH_ENV, str(tmp_path / "meshes"))
    (tmp_path / "meshes").mkdir()
    for suffix in ("grid.nc", "ugwp_oro_data.nc"):
        (tmp_path / "meshes" / f"x1.40962.{suffix}").write_bytes(b"x")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {
        "cells": 40962, "edges": 122880, "levels": 55, "nominal_dc_m": 120000.0})
    _, row = rows.make_row(table, "hrrrv5", rows.find_mesh("x1.40962"))
    assert row["self_dot_dead"]["list"] == "latest/lists/dead120-tail-s3.json"
    assert "AD_DOT_DEAD" not in row["window_env"]


def _window(words, equal=3, checked=3):
    return {"dots": [{"rel_aligned": 8.1e-09, "reverse": [{"result_sha256": words + "0" * 48}]}],
            "self_dot": {"allocations": 30}, "door_validity": {"boundaries_checked": checked,
                                                               "boundaries_equal": equal}}


def _poison_out(tmp_path, nbytes=9337888):
    out = tmp_path / "selfdot-proof"
    out.mkdir()
    (out / "stderr").write_text("noise\n" + json.dumps({"event": "window_dead_poison_rev", "step": 3, "at": "walk",
                                                        "ranges": 1, "bytes": nbytes}) + "\n")
    return out


def test_the_masked_dot_passes_only_with_a_passing_poison_proof(tmp_path):
    lst = tmp_path / "dead.json"
    lst.write_text(json.dumps({"ranges": [{"allocation": 417, "offset": 425669600, "bytes": 9337888}]}))
    left = d.dead_ranges(lst)
    assert left["bytes_total"] == 9337888 and left["ranges_n"] == 1
    thresholds = d.dot_bar_in_force(d.load_table())
    control = _window("f34021512af41b2e")
    proof = d.dead_proof_check(control, _window("f34021512af41b2e"), _poison_out(tmp_path), left)
    assert proof["ok"], proof["reasons"]
    assert d.self_dot_check(control, thresholds, left, proof)["pass"]
    # under the bar but no proof, or a proof whose gradient moved: not a pass
    assert not d.self_dot_check(control, thresholds, left, None)["pass"]
    moved = d.dead_proof_check(control, _window("0000000000000000"), tmp_path / "selfdot-proof", left)
    assert not moved["ok"] and "differ" in moved["reasons"][0]
    assert not d.self_dot_check(control, thresholds, left, moved)["pass"]
    # no list: the plain self dot as before
    assert d.self_dot_check(control, thresholds)["pass"]


def test_a_proof_that_poisoned_other_bytes_or_none_does_not_pass(tmp_path):
    lst = tmp_path / "dead.json"
    lst.write_text(json.dumps({"ranges": [{"allocation": 417, "offset": 0, "bytes": 9337888}]}))
    left = d.dead_ranges(lst)
    control = _window("ab")
    short = d.dead_proof_check(control, _window("ab"), _poison_out(tmp_path, nbytes=4096), left)
    assert not short["ok"] and "poisoned" in short["reasons"][0]
    empty = tmp_path / "none"
    empty.mkdir()
    assert not d.dead_proof_check(control, _window("ab"), empty, left)["ok"]
