# SPDX-License-Identifier: Apache-2.0
"""What a user's sensitivity command costs: the gradient line (the measured fast route's words as table data), several
objectives on ONE loaded program, the objective kept in the door's cache, and the first-gradient routes.

THE GAP THESE HOLD SHUT: the fast route's seconds lived only in scripts beside the release (59-step 3 km gradient: a
later one 298.0 s from a program kept loaded, objective translated once and kept), while the release's own
sensitivity started a fresh program for every objective, made its objective again every command and set none of the
words that held their answer; and a receipt of a window that compared no step boundary must not say it compared them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d


@pytest.fixture(autouse=True)
def _stand_in_runner_on_hidden_cards(monkeypatch):
    # The battery hides the cards (conftest CUDA_VISIBLE_DEVICES=-1) and these tests start a stand-in runner that
    # opens no device, so the door's hidden-card refusal (refuse_hidden_card) is let through here only.
    from woof.hex import ad_doors as _d
    monkeypatch.setattr(_d, "refuse_hidden_card", lambda: None)

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"

# A stand-in for the runner's window mode with the loaded-program route (resident.rs): it writes window.json in --out,
# and with --resident DIR answers DIR/NAME.done, then serves DIR/*.req in name order until DIR/stop or idle.
FAKE_RUNNER = r'''#!PYTHON
import json, os, sys, time
from pathlib import Path
argv = sys.argv[1:]
assert argv[0] == "window", argv
words = {argv[i][2:]: argv[i + 1] for i in range(1, len(argv) - 1, 2)}
log = os.environ.get("FAKE_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"pid": os.getpid(), "words": sorted(words)}) + "\n")
MAY = {"out", "x0-planted", "objective", "dump-grad", "dump-state", "bstore", "spill-dir", "x0-dir", "x0-h"}
def gradient(w):
    out = Path(w["out"]); out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FAKE_FAIL_ON") and os.environ["FAKE_FAIL_ON"] in w["out"]:
        return 1, "cuMemAlloc scheme scratch failed with CUDA status 2"
    obj = json.loads(Path(w["objective"]).read_text())
    (out / "window.json").write_text(json.dumps({"objective": {"reverse_wall_s": 0.5, "name": obj.get("name")},
                                                 "words": w, "pid": os.getpid()}))
    return 0, None
rdir = words.get("resident")
if os.environ.get("FAKE_NO_RESIDENT"):
    rdir = None
rc, err = gradient(words)
if rdir is None:
    sys.exit(rc)
rdir = Path(rdir)
name = words.get("resident-name", "r001")
idle = float(words.get("resident-idle-s", "900"))
def answer(name, rc, err, index):
    tmp = rdir / (name + ".done.tmp")
    tmp.write_text(json.dumps({"rc": rc, "error": err, "name": name, "index": index, "wall_s": 0.5, "pid": os.getpid()}))
    os.replace(tmp, rdir / (name + ".done"))
index = 1
answer(name, rc, err, index)
if rc:
    sys.exit(1)
t0 = time.time()
while True:
    if (rdir / "stop").exists():
        sys.exit(0)
    reqs = sorted(rdir.glob("*.req"))
    if reqs:
        p = reqs[0]; name = p.stem; taken = p.with_suffix(".taken"); os.rename(p, taken)
        req = json.loads(taken.read_text())
        bad = [k for k in req if k not in MAY]
        index += 1
        if bad:
            answer(name, 1, "request words outside the request list: " + ",".join(bad), index); sys.exit(1)
        merged = dict(words); merged.update({k: v for k, v in req.items() if v != ""})
        rc, err = gradient(merged)
        answer(name, rc, err, index)
        if rc:
            sys.exit(1)
        t0 = time.time()
        continue
    if time.time() - t0 > idle:
        sys.exit(0)
    time.sleep(0.02)
'''


@pytest.fixture
def fake_runner(tmp_path):
    path = tmp_path / "ad-binder"
    path.write_text(FAKE_RUNNER.replace("#!PYTHON", "#!" + sys.executable))
    path.chmod(0o755)
    return path


@pytest.fixture
def table():
    return d.load_table()


def _objective(path: Path, name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "terms": [[1, 0, 1.0]]}))
    return path


def _plant_route_images(tmp_path, row):
    """The row's boundary-layer route images as stand-in files in the test asset root, the table's pins moved to
    them (the table of a test is its own copy)."""
    out = {}
    routes = ((row.get("gradient_line") or {}).get("boundary_layer") or {}).get("routes") or {}
    for route in routes.values():
        for var, item in (route.get("images") or {}).items():
            path = tmp_path / "assets" / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(var.encode())
            item["sha256"] = hashlib.sha256(var.encode()).hexdigest()
            out[var] = path
    return out


def _ctx(tmp_path, table, row_name=STORM, **args):
    row = table["rows"][row_name]
    _plant_route_images(tmp_path, row)
    ns = argparse.Namespace(checked_forward=False, boundary_layer=None, first_gradient=None, wait_card=0.0,
                            no_self_dot=False, strict_from=None, no_map=True, map_level=None, objectives=None,
                            metric="theta_m", levels=None, box=None, circle=None)
    for k, v in args.items():
        setattr(ns, k, v)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return SimpleNamespace(table=table, row=row, row_name=row_name, door="sensitivity", root=tmp_path / "assets",
                           card="0", cards=["0"], arch="sm_120", split=None, out=out, args=ns, steps=3)


# ------------------------------------------------------------------------------------------ the gradient line words
def test_the_storm_rows_gradient_line_is_table_data_and_gives_the_measured_routes_words(tmp_path, table):
    ctx = _ctx(tmp_path, table)
    line = d.gradient_line(ctx, ctx.args)
    assert line is not None and line["loaded_program"]["idle_s"] > 0 and line["empty_move"] == {"name": "empty move",
                                                                                                "terms": []}
    words = d.gradient_line_words(ctx, line, ctx.args, ctx.out)
    extra, env = words["extra"], words["env"]
    assert extra[extra.index("--checkpoints") + 1] == "sqrt"
    move = Path(extra[extra.index("--x0-dir") + 1])
    assert json.loads(move.read_text()) == {"name": "empty move", "terms": []}
    for var in ("AD_FREE_BESIDE", "AD_NATIVE_NODRAIN", "AD_SNAP_REUSE", "AD_RECOMPUTE_UNCHECKED", "AD_MYNN_FWD_LANES",
                "AD_P3_FUSE"):
        assert env[var] == "1", var
    assert env["AD_BS_PREFILL"] == "8"
    # the row's default boundary layer is the tape-free route: its word, its two pinned images, a 33 GiB tape
    assert env["AD_P3_DIRECT"] == "1"
    root = tmp_path / "assets" / "latest" / "aux" / "bl-closure"
    assert env["AD_P3_DIRECT_IMAGE"] == str(root / "direct120.cubin")
    assert env["AD_P3_DIRECT_FWD"] == str(root / "cap120.cubin")
    assert words["replace"] == {"--tape-gib": "33", "--scratch-gib": "4"}
    # the readback words ride on that route's gradient windows only: the answer check's windows leave them out
    assert env["AD_TAPE_PREDICT"] == "1" and env["AD_TAPE_BLOCK_KIB"] == "0"
    assert words["gradient_only"] == ["AD_TAPE_BLOCK_KIB", "AD_TAPE_PREDICT"]   # no book: this ctx names no plan
    check = d.check_line_words(words)
    assert "AD_TAPE_PREDICT" not in check["env"] and "AD_TAPE_BLOCK_KIB" not in check["env"]
    assert check["env"]["AD_P3_DIRECT"] == "1" and check["env"]["AD_BS_PREFILL"] == "8"
    assert "AD_TAPE_PREDICT" in check["not_taken"] and "--x0-dir" in check["not_taken"]
    # the drag image is not in this asset root: left out by name, never passed as a path that is not there
    assert "AD_P4_UGWP_DIRECT" not in env and "AD_P4_UGWP_DIRECT" in words["block"]["optional_assets_left_out"]
    assert words["block"]["boundary_layer"]["route"] == "tape-free"
    assert words["block"]["boundary_layer"]["exact_word"] == "--boundary-layer taped"
    assert "not compared" in words["block"]["boundaries"]


def test_the_storm_sensitivity_is_admitted_on_the_card_memory_its_route_held(tmp_path, table):
    """THE DEFECT: the row stated 93,000 MiB and a 59-step sensitivity on the tape-free route held 95,662 MiB; the
    route then stated 96,000 MiB and its next 96 GB run held 96,214 MiB (96,571 MiB sampled on the card)."""
    ctx = _ctx(tmp_path, table)
    need = types.MethodType(d.Context.need_mib, ctx)
    assert need() == 96600 and ctx.row["need_free_mib"] == 93000
    ctx.args.boundary_layer = "taped"
    assert need() == 93000
    ctx.args.boundary_layer, ctx.args.checked_forward = None, True
    assert need() == 93000, "--checked-forward runs the row's own window"
    ctx.args.checked_forward, ctx.door = False, "tangent"
    assert need() == 93000, "the other commands run the row's own window"
    ctx.door, ctx.args.need_mib = "sensitivity", 5
    assert need() == 5


def test_the_taped_boundary_layer_is_the_named_exact_route(tmp_path, table, capsys):
    ctx = _ctx(tmp_path, table, boundary_layer="taped")
    words = d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args, ctx.out)
    env = words["env"]
    assert not [k for k in env if k.startswith(("AD_P3_DIRECT", "AD_TAPE_PREDICT", "AD_TAPE_BLOCK"))]
    assert words["replace"] == {"--tape-gib": "34", "--scratch-gib": "4"} and words["gradient_only"] == []
    assert words["block"]["boundary_layer"]["route"] == "taped"
    assert "none: the exact route" in capsys.readouterr().out


def test_the_tape_free_route_keeps_its_decision_book_by_plan_kernel_set_runner_and_steps(tmp_path, table, monkeypatch):
    from woof.hex import assets as _assets
    monkeypatch.setattr(_assets, "cache_root", lambda *a, **k: tmp_path / "cache")
    ctx = _ctx(tmp_path, table)
    ctx.fmap = SimpleNamespace(doc={"plan_sha256": "ab" * 32})
    ctx.runner = tmp_path / "ad-binder"
    ctx.runner.write_bytes(b"runner")
    line = d.gradient_line(ctx, ctx.args)
    words = d.gradient_line_words(ctx, line, ctx.args, ctx.out)
    book = Path(words["env"]["AD_TAPE_PREDICT_BOOK"])
    assert book.parent == tmp_path / "cache" / "ad" / "books" and book.parent.is_dir() and book.suffix == ".book"
    assert "AD_TAPE_PREDICT_BOOK" in words["gradient_only"]
    assert "AD_TAPE_PREDICT_BOOK" not in d.check_line_words(words)["env"]
    assert d.gradient_line_words(ctx, line, ctx.args, ctx.out)["env"]["AD_TAPE_PREDICT_BOOK"] == str(book)
    ctx.steps = 59                                      # another window length: another book
    assert d.gradient_line_words(ctx, line, ctx.args, ctx.out)["env"]["AD_TAPE_PREDICT_BOOK"] != str(book)
    ctx.steps = 3
    ctx.fmap.doc = {"plan_sha256": "cd" * 32}           # another plan: another book
    assert d.gradient_line_words(ctx, line, ctx.args, ctx.out)["env"]["AD_TAPE_PREDICT_BOOK"] != str(book)


def test_an_optional_image_is_used_only_when_the_asset_root_holds_the_pinned_build(tmp_path, table):
    image = tmp_path / "assets" / "latest" / "aux" / "ugwp.dad.sm_120.cubin"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"drag image")
    ctx = _ctx(tmp_path, table, boundary_layer="taped")
    line = d.gradient_line(ctx, ctx.args)
    assert "AD_P4_UGWP_DIRECT" not in d.gradient_line_words(ctx, line, ctx.args, ctx.out)["env"]   # another build
    line["row"] = json.loads(json.dumps(line["row"]))
    line["row"]["optional_assets"]["AD_P4_UGWP_DIRECT"]["sha256"] = hashlib.sha256(b"drag image").hexdigest()
    assert d.gradient_line_words(ctx, line, ctx.args, ctx.out)["env"]["AD_P4_UGWP_DIRECT"] == str(image)
    ctx.arch = "sm_89"
    assert "AD_P4_UGWP_DIRECT" not in d.gradient_line_words(ctx, line, ctx.args, ctx.out)["env"]


def test_replace_args_set_the_rows_tape_and_scratch_in_place_once(tmp_path, table):
    row = table["rows"][STORM]
    assets = {}
    for key in set(row["asset_args"].values()) | set(row["asset_env"].values()):
        p = tmp_path / key
        p.write_text("x")
        assets[key] = p
    argv, env = d.window_command(table, row, assets, "sensitivity", Path("ad-binder"), 59, tmp_path / "w",
                                 ["--checkpoints", "sqrt"], extra_env={"AD_SNAP_REUSE": "1"},
                                 replace_args={"--tape-gib": "34", "--scratch-gib": "4", "--new-word": "7"})
    assert argv.count("--tape-gib") == 1 and argv[argv.index("--tape-gib") + 1] == "34"
    assert argv.count("--scratch-gib") == 1 and argv[argv.index("--scratch-gib") + 1] == "4"
    assert argv[argv.index("--new-word") + 1] == "7" and env["AD_SNAP_REUSE"] == "1"
    plain, _ = d.window_command(table, row, assets, "sensitivity", Path("ad-binder"), 59, tmp_path / "w", [])
    assert plain[plain.index("--tape-gib") + 1] == "6", "a window without the gradient line keeps the row's words"


@pytest.mark.parametrize("how", ["checked", "split", "row"])
def test_checked_forward_a_split_window_and_a_row_without_a_line_run_as_before(tmp_path, table, how):
    if how == "row":
        ctx = _ctx(tmp_path, table, row_name=table["default_row"])
    else:
        ctx = _ctx(tmp_path, table, checked_forward=(how == "checked"))
        ctx.split = {"ranks": 2} if how == "split" else None
    assert d.gradient_line(ctx, ctx.args) is None
    assert {"checked": "--checked-forward", "split": "split over cards", "row": "names no gradient line"}[how] \
        in ctx.gradient_line_off


def test_the_tape_free_boundary_layer_is_the_rows_default_and_is_refused_by_name_where_it_cannot_run(
        tmp_path, table, capsys):
    args = argparse.Namespace(boundary_layer="tape-free", checked_forward=False, row=None, mesh=None)
    with pytest.raises(d.AdDoorError, match="has no gradient line"):
        d.refuse_speed_words(args, table)            # the default row (120 km) has none
    args.row = STORM
    d.refuse_speed_words(args, table)
    row = table["rows"][STORM]
    assert row["gradient_line"]["boundary_layer"]["default"] == "tape-free"
    # the pinned download holds both images, and a door on the row asks for them
    from woof.hex.ad import assets
    pins = assets.load_pins()
    route = row["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    held = pins["assets"][route["pinned_in"]]["holds"]
    assert all(item["path"] in held and item["arch"] == "sm_120" for item in route["images"].values())
    plain = SimpleNamespace(plan=None, init=None, boundary_layer=None)
    assert route["pinned_in"] in assets.wanted(pins, d.example_reads(row, plain))
    plain.boundary_layer = "taped"
    assert route["pinned_in"] not in assets.wanted(pins, d.example_reads(row, plain))
    # a card family the images do not cover: refused by name, before any download, with the exact route's word
    with pytest.raises(d.AdDoorError, match=r"tape-free route \(this row's default\) runs images built for sm_120 "
                                            r"cards only \(cap120.cubin, direct120.cubin\) and card 0 is sm_89.*"
                                            r"--boundary-layer taped runs the exact route.*Nothing was downloaded"):
        d.refuse_route_card_family(STORM, row, None, "sm_89", "0")
    d.refuse_route_card_family(STORM, row, "taped", "sm_89", "0")
    d.refuse_route_card_family(STORM, row, None, "sm_120", "0")
    ctx = _ctx(tmp_path, table)
    words = d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args, ctx.out)
    assert words["env"]["AD_P3_DIRECT"] == "1" and words["replace"]["--tape-gib"] == "33"
    said = capsys.readouterr().out
    # no fast-against-exact figure is printed: the default form differs from the exact one by design (dist48)
    assert words["block"]["boundary_layer"]["error"] == row["gradient_line"]["boundary_layer"]["routes"]["tape-free"]["error"]
    assert words["block"]["boundary_layer"]["error_figures_of_this_plan"] is False and "1.2e-6" not in said
    assert "(this row's default)" in said and "not byte-equal" in said and "--boundary-layer taped runs the exact" in said
    ctx.arch = "sm_89"
    with pytest.raises(d.AdDoorError, match="card 0 is sm_89"):
        d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args, ctx.out)
    ctx.arch = "sm_120"
    # an image that is not the pinned build, or not there: refused by name, never left out
    image = tmp_path / "assets" / route["images"]["AD_P3_DIRECT_FWD"]["path"]
    image.write_bytes(b"another build")
    with pytest.raises(d.AdDoorError, match="is not the build the table pins.*--boundary-layer taped"):
        d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args, ctx.out)
    image.unlink()
    with pytest.raises(d.AdDoorError, match="cap120.cubin.*does not hold.*--boundary-layer taped"):
        d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args, ctx.out)
    args.checked_forward = True
    with pytest.raises(d.AdDoorError, match="give one"):
        d.refuse_speed_words(args, table)


# --------------------------------------------------------------------------------------------------- validity
@pytest.mark.parametrize("logged", ["equal", "not compared"])
def test_a_gradient_line_window_never_reads_as_having_compared_its_boundaries(logged):
    runner_says = {"boundaries_checked": 3, "boundaries_equal": 3 if logged == "equal" else 0, "valid": logged == "equal",
                   "refused_beyond_row_list": []}
    v = d.gradient_line_validity(runner_says, {})
    assert v["boundaries_checked"] == 0 and v["boundaries_compared"] is False and v["valid"] is True
    assert v["boundaries_runner_logged_checked"] == 3 and "answer check" in v["what"]
    assert d.gradient_line_validity({**runner_says, "refused_beyond_row_list": ["k"]}, {})["valid"] is False


# --------------------------------------------------------------------------------------------- the kept objective
@pytest.fixture
def objective_ctx(tmp_path, table, monkeypatch):
    monkeypatch.setenv(d.OBJECTIVES_ENV, str(tmp_path / "objectives"))
    grid = tmp_path / "grid.nc"
    grid.write_bytes(b"grid")
    field = {"name": "theta_m", "alloc": 193, "offset": 0, "shape": [59, 4], "grid": "cell", "units": "K"}
    ctx = _ctx(tmp_path, table)
    ctx.fmap = SimpleNamespace(doc={"plan_sha256": "ab" * 32}, resolve=lambda *a, **k: field)
    ctx.geometry, ctx.grid = {}, grid
    made = []

    def make(table_, fmap, geometry, name, levels, box, circle, row=None):
        made.append((name, levels))
        return {"name": f"{name} {levels}", "terms": [[193, k, 0.25] for k in range(4)], "metric": name,
                "field": "theta_m", "units": "K", "cells": [0, 1, 2, 3], "levels_1based": [1]}

    monkeypatch.setattr(d, "metric_objective", make)
    return ctx, made


def test_the_same_objective_asked_again_takes_the_kept_file(objective_ctx, tmp_path):
    ctx, made = objective_ctx
    spec = {"metric": "theta_m", "levels": "1-10", "box": None, "circle": None}
    obj, path, block = d.objective_file(ctx, spec, tmp_path / "a" / "objective.json")
    assert block["how"] == "made" and Path(block["file"]).is_file() and made == [("theta_m", "1-10")]
    obj2, path2, block2 = d.objective_file(ctx, spec, tmp_path / "b" / "objective.json")
    assert block2["how"] == "kept" and made == [("theta_m", "1-10")], "the objective was made again"
    assert path2.read_bytes() == path.read_bytes() and obj2["units"] == "K" and obj2["terms_n"] == 4
    assert json.loads(path2.read_text())["terms"] == [[193, k, 0.25] for k in range(4)]
    # another level range is another objective
    assert d.objective_file(ctx, {**spec, "levels": "1-20"}, tmp_path / "c" / "o.json")[2]["how"] == "made"
    # another plan is another objective
    ctx.fmap.doc = {"plan_sha256": "cd" * 32}
    assert d.objective_file(ctx, spec, tmp_path / "d" / "o.json")[2]["how"] == "made"


def test_a_changed_kept_file_is_made_again_and_says_why(objective_ctx, tmp_path):
    ctx, made = objective_ctx
    spec = {"metric": "theta_m", "levels": None, "box": None, "circle": None}
    first = d.objective_file(ctx, spec, tmp_path / "a" / "objective.json")[2]
    # an edit through the user's folder (a hard link) or of the cache itself changes the kept bytes
    Path(first["file"]).write_text(json.dumps({"name": "edited", "terms": [[193, 0, 9.0]]}))
    again = d.objective_file(ctx, spec, tmp_path / "b" / "objective.json")[2]
    assert again["how"] == "made" and "changed" in again["why_made"] and len(made) == 2
    assert json.loads((tmp_path / "b" / "objective.json").read_text())["terms"][0] == [193, 0, 0.25]


def test_an_unwritable_cache_is_said_and_the_command_still_runs(objective_ctx, tmp_path, monkeypatch, capsys):
    ctx, _ = objective_ctx
    blocker = tmp_path / "objectives-file"
    blocker.write_text("not a folder")
    monkeypatch.setenv(d.OBJECTIVES_ENV, str(blocker / "sub"))
    obj, path, block = d.objective_file(ctx, {"metric": "theta_m", "levels": None, "box": None, "circle": None},
                                        tmp_path / "a" / "objective.json")
    assert path.is_file() and block["how"] == "made" and block["file"] is None
    assert "was not kept" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------- the loaded program
@pytest.fixture
def loaded(tmp_path, fake_runner, monkeypatch):
    monkeypatch.setattr(d, "refuse_no_local_gpu", lambda: None)
    monkeypatch.setattr(d, "process_card_mib", lambda pid: {"0000:01:00.0": 1000})
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "starts.log"))
    plan = tmp_path / "plan"
    plan.mkdir()
    return SimpleNamespace(tmp=tmp_path, runner=fake_runner, plan=plan, log=tmp_path / "starts.log")


def _argv(lo, out, objective):
    return [str(lo.runner), "window", "--config", "plan.json", "--steps", "3", "--out", str(out), "--bstore",
            str(out / "bstore"), "--objective", str(objective), "--dump-grad", "3,46", "--checkpoints", "sqrt",
            "--x0-dir", str(lo.tmp / "empty.json"), "--tangent-only", "1", "--no-tangent", "1"]


def test_two_objectives_run_on_one_program_and_each_gets_its_own_answer(loaded):
    lo = loaded
    program = d.LoadedProgram(lo.tmp / "program", idle_s=30, stop_wait_s=10)
    o1, o2, o3 = (lo.tmp / f"o{k}" / "window" for k in (1, 2, 3))
    doc1 = program.start(_argv(lo, o1, _objective(lo.tmp / "o1" / "objective.json", "first")), {}, lo.plan, "0",
                         "o01", o1)
    doc2 = program.request("o02", {"out": str(o2), "objective": str(_objective(lo.tmp / "o2" / "objective.json",
                                                                                "second")),
                                   "dump-grad": "3,46", "bstore": str(o2 / "bstore")}, o2)
    doc3 = program.request("o03", {"out": str(o3), "objective": str(_objective(lo.tmp / "o3" / "objective.json",
                                                                                "third")),
                                   "dump-grad": "3,46", "bstore": str(o3 / "bstore")}, o3)
    block = program.stop()
    starts = lo.log.read_text().splitlines()
    assert len(starts) == 1, "a later objective started a program of its own"
    assert [doc1["objective"]["name"], doc2["objective"]["name"], doc3["objective"]["name"]] == ["first", "second",
                                                                                                  "third"]
    assert doc1["pid"] == doc2["pid"] == doc3["pid"] == block["pid"]
    assert [x["name"] for x in block["served"]] == ["o01", "o02", "o03"] and block["rc"] == 0
    assert not block["ended_by_door"] and (lo.tmp / "program" / "stop").is_file()
    assert doc2["loaded_program"]["first"] is False and doc1["loaded_program"]["first"] is True
    assert doc2["door_card_peak_mib"] == 1000 or doc2["door_card_peak_mib"] is None
    argv = (lo.tmp / "program" / "cmd").read_text()
    assert "--resident " in argv and "--resident-name o01" in argv and "--resident-idle-s 30" in argv


def test_a_request_changes_only_the_request_words(loaded):
    lo = loaded
    program = d.LoadedProgram(lo.tmp / "program", idle_s=30, stop_wait_s=10)
    o1 = lo.tmp / "o1" / "window"
    program.start(_argv(lo, o1, _objective(lo.tmp / "o1" / "objective.json", "first")), {}, lo.plan, "0", "o01", o1)
    with pytest.raises(d.AdDoorError, match="changes only"):
        program.request("o02", {"out": "x", "steps": "59"}, lo.tmp / "o2")
    program.stop()


def test_a_failed_objective_names_itself_and_a_card_memory_failure_is_called_one(loaded, monkeypatch):
    lo = loaded
    monkeypatch.setenv("FAKE_FAIL_ON", "fail-here")
    program = d.LoadedProgram(lo.tmp / "program", idle_s=30, stop_wait_s=10)
    o1, o2 = lo.tmp / "o1" / "window", lo.tmp / "fail-here" / "window"
    program.start(_argv(lo, o1, _objective(lo.tmp / "o1" / "objective.json", "first")), {}, lo.plan, "0", "o01", o1)
    with pytest.raises(d.CardMemoryError, match="objective o02"):
        program.request("o02", {"out": str(o2), "objective": str(lo.tmp / "o1" / "objective.json"),
                                "dump-grad": "3", "bstore": str(o2 / "bstore")}, o2)
    program.stop()


def test_a_runner_that_serves_no_requests_is_said_and_its_first_answer_kept(loaded, monkeypatch):
    lo = loaded
    monkeypatch.setenv("FAKE_NO_RESIDENT", "1")
    program = d.LoadedProgram(lo.tmp / "program", idle_s=30, stop_wait_s=10)
    o1 = lo.tmp / "o1" / "window"
    with pytest.raises(d.NotLoaded) as caught:
        program.start(_argv(lo, o1, _objective(lo.tmp / "o1" / "objective.json", "first")), {}, lo.plan, "0", "o01",
                      o1)
    assert caught.value.doc["objective"]["name"] == "first" and "without serving requests" in str(caught.value)
    program.stop()


def test_a_program_that_does_not_end_after_stop_is_ended_by_its_own_pid(loaded, tmp_path):
    lo = loaded
    program = d.LoadedProgram(lo.tmp / "program", idle_s=30, stop_wait_s=0.5)
    program.rdir.mkdir()
    sleeper = subprocess_sleeper(tmp_path)
    program.proc = sleeper
    program._streams, program.t0 = (), time.monotonic()
    block = program.stop()
    assert block["ended_by_door"] is True and sleeper.returncode is not None


def subprocess_sleeper(tmp_path):
    import subprocess
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


# ------------------------------------------------------------------------- the call: many objectives, one program
def test_a_call_of_three_objectives_runs_one_program_one_answer_check_and_three_receipts(tmp_path, table, loaded,
                                                                                         monkeypatch):
    lo = loaded
    ctx = _ctx(tmp_path, table)
    ctx.runner, ctx.plan, ctx.arch_assets, ctx.kset_assets = lo.runner, lo.plan, {"x": 1}, {}
    ctx.assets = {}
    for key in set(ctx.row["asset_args"].values()) | set(ctx.row["asset_env"].values()):
        p = tmp_path / "assets-files" / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        ctx.assets[key] = p
    ctx.stages, ctx.windows, ctx.started = [], [], time.time()
    ctx.need_mib = lambda: 1
    ctx.class_label = lambda: None
    for name in ("window_line", "window_loaded", "window_loaded_fallback", "_loaded_doc", "stage"):
        setattr(ctx, name, types.MethodType(getattr(d.Context, name), ctx))
    ctx.fmap = SimpleNamespace(resolve=lambda *a, **k: {"name": "rho", "copies": [(3, 0)], "units": "kg"}, fields={},
                               doc={"plan_sha256": "ab" * 32})
    monkeypatch.setattr(d, "_sensitivity_inputs", lambda c: ([{"name": "rho", "copies": [(3, 0)]}], [3, 46]))
    monkeypatch.setattr(d, "admit_card", lambda *a, **k: {"free_mib": 96000})
    monkeypatch.setattr(d, "host_available_mib", lambda: 200 * 1024)

    def objective_file(c, spec, dest):
        _objective(dest, spec["metric"] + str(spec.get("levels")))
        return ({"name": spec["metric"], "metric": spec["metric"], "units": "K"}, dest,
                {"how": "made" if spec.get("levels") != "1-10" else "kept"})

    monkeypatch.setattr(d, "objective_file", objective_file)
    checks = []

    def strict_check(c, no_self_dot=False, strict_from=None):
        checks.append(str(c.out))
        return {"how": "own", "pass": True, "value": 8e-9, "bar": 1e-7, "windows": ["selfdot"]}

    monkeypatch.setattr(d, "strict_check", strict_check)
    answered = []

    def answer(c, args, objective, inputs, doc, strict):
        answered.append({"out": c.out, "how": strict["how"], "doc": doc, "extra": c.receipt_extra,
                         "windows": list(c.windows), "stages": list(c.stages)})
        return 0, {"rec": {"passed": True, "valid": True}}

    monkeypatch.setattr(d, "_sensitivity_answer", answer)
    specs = [{"metric": "theta_m", "levels": "1-10", "box": None, "circle": None, "name": "low"},
             {"metric": "theta_m", "levels": "20-40", "box": None, "circle": None, "name": None},
             {"metric": "qv", "levels": None, "box": None, "circle": None, "name": "moist"}]
    assert d._sensitivity_many(ctx, ctx.args, specs) == 0
    assert len(lo.log.read_text().splitlines()) == 1, "each objective started its own program"
    assert checks == [str(ctx.out / "o01-low")], "the answer check ran more than once, or not in the first folder"
    assert [a["out"].name for a in answered] == ["o01-low", "o02", "o03-moist"]
    assert [a["how"] for a in answered] == ["own", "same call", "same call"]
    assert answered[1]["extra"]["call"] == {"objective": 2, "of": 3, "receipt": str(ctx.out / "receipt.json")}
    for a in answered:
        w = a["windows"][0]
        assert w["boundaries_compared"] is False and w["valid"] is True
        assert any(s["stage"].startswith("window (window, loaded program") for s in a["stages"])
        assert a["extra"]["gradient_line"]["on"] is True and a["extra"]["first_gradient"]["route"] == "recorded"
    call = json.loads((ctx.out / "receipt.json").read_text())
    assert call["schema"] == d.CALL_SCHEMA and call["loaded_program"]["on"] and call["passed"] is True
    assert [x["name"] for x in call["loaded_program"]["served"]] == ["o01", "o02", "o03"]
    assert [r["objective_how"] for r in call["objectives"]] == ["kept", "made", "made"]
    assert call["strict_check"]["receipt"] == str(ctx.out / "o01-low" / "receipt.json")
    cmd = (ctx.out / "program" / "cmd").read_text()
    assert "--tape-gib 33" in cmd and "--x0-dir" in cmd and "AD_SNAP_REUSE=1" in cmd and "--bstore-ram-gib 120" in cmd
    assert "AD_P3_DIRECT=1" in cmd and "AD_TAPE_PREDICT=1" in cmd


# --------------------------------------------------------------------------- the words, refused before any card
def test_objectives_take_the_commands_words_and_their_own_region(tmp_path, table):
    f = tmp_path / "obj.json"
    f.write_text(json.dumps([{"levels": "1-10"}, {"metric": "qv", "box": "40,41,-92,-91", "name": "moist"}]))
    args = argparse.Namespace(metric="theta_m", levels="5-6", box=None, circle="40.6,-91.3,50", objectives=f)
    a, b = d.objective_specs(args, table)
    assert a == {"metric": "theta_m", "levels": "1-10", "box": None, "circle": "40.6,-91.3,50", "name": None}
    assert b == {"metric": "qv", "levels": "5-6", "box": "40,41,-92,-91", "circle": None, "name": "moist"}
    args.objectives = None
    assert d.objective_specs(args, table) == [{"metric": "theta_m", "levels": "5-6", "box": None,
                                               "circle": "40.6,-91.3,50", "name": None}]


@pytest.mark.parametrize("doc,words", [([{"metrik": "qv"}], "names metrik"), ([{"metric": "nope"}], "not in the table"),
                                        ([], "at least one"), ({"objectives": "x"}, "at least one"),
                                        ([{"box": "1,2,3,4", "circle": "1,2,3"}], "alternatives")])
def test_a_bad_objectives_file_is_refused_before_any_card(tmp_path, table, monkeypatch, doc, words):
    f = tmp_path / "obj.json"
    f.write_text(json.dumps(doc))
    monkeypatch.setattr(d, "Context", lambda *a, **k: pytest.fail("a card or download was reached"))
    args = argparse.Namespace(metric="theta_m", levels=None, box=None, circle=None, objectives=f, row=STORM, mesh=None,
                              boundary_layer=None, checked_forward=False, first_gradient=None, gradient_mesh="native")
    with pytest.raises(d.AdDoorError, match=words):
        d.run_sensitivity(args)


def test_the_first_gradient_routes_and_the_direct_route_refused_with_its_cover_and_break_even(table, monkeypatch):
    args = argparse.Namespace(first_gradient=None)
    route = d.first_gradient_route(args, table)
    assert route["route"] == "recorded" and route["covers"] == "every step of the window"
    assert route["other_routes"]["direct"]["in_this_build"] is False
    args.first_gradient = "direct"
    with pytest.raises(d.AdDoorError) as caught:
        d.first_gradient_route(args, table)
    said = str(caught.value)
    assert "not in this build" in said and "loop steps 2 to N" in said and "12 per cent" in said
    assert "from the third gradient" in said and "978.8 s" in said and "Nothing was downloaded" in said
    monkeypatch.setattr(d, "Context", lambda *a, **k: pytest.fail("a card or download was reached"))
    full = argparse.Namespace(first_gradient="direct", metric="theta_m", levels=None, box=None, circle=None,
                              objectives=None, row=STORM, mesh=None, boundary_layer=None, checked_forward=False,
                              gradient_mesh="native")
    with pytest.raises(d.AdDoorError, match="not in this build"):
        d.run_sensitivity(full)


def test_the_sensitivity_parser_has_the_speed_words():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sensitivity", "--out", "x", "--objectives", "o.json", "--boundary-layer", "tape-free",
                              "--first-gradient", "recorded", "--checked-forward"])
    assert args.objectives == Path("o.json") and args.boundary_layer == "tape-free" and args.checked_forward
    with pytest.raises(SystemExit):
        parser.parse_args(["sensitivity", "--out", "x", "--first-gradient", "sideways"])


def test_the_break_even_in_the_table_is_the_arithmetic_of_its_measured_seconds(table):
    routes = table["doors"]["sensitivity"]["first_gradient"]["routes"]["direct"]
    rec, first, later, direct = 1684.0, 322.5, 298.0, 978.8
    cost = lambda n: (rec + first + later * (n - 1), direct * n)
    assert cost(2)[0] > cost(2)[1] and cost(3)[0] < cost(3)[1]
    assert "2,304.5 against 1,957.6" in routes["break_even"]["storm 3 km"]
    assert "2,602.5 against 2,936.4" in routes["break_even"]["storm 3 km"]
    rec6, first6, later6, direct6 = 724.0, 98.0, 95.0, 142.4
    cost6 = lambda n: (rec6 + first6 + later6 * (n - 1), direct6 * n)
    assert cost6(15)[0] > cost6(15)[1] and cost6(16)[0] < cost6(16)[1]
    assert f"{cost6(15)[0]:,.0f} against {cost6(15)[1]:,.0f}" in routes["break_even"]["storm 6 km cut"]


@pytest.mark.parametrize("checked", [False, True])
def test_one_objective_on_the_storm_row_runs_the_gradient_line_and_no_loaded_program(tmp_path, table, monkeypatch,
                                                                                     checked):
    ctx = _ctx(tmp_path, table, checked_forward=checked)
    ctx.stages = []
    ctx.stage = lambda name, t0, par: ctx.stages.append(name)
    monkeypatch.setattr(d, "objective_file", lambda c, spec, dest: (
        {"name": "J", "metric": "theta_m", "units": "K"}, _objective(dest, "J"), {"how": "kept"}))
    monkeypatch.setattr(d, "_sensitivity_inputs", lambda c: ([], [3, 46]))
    monkeypatch.setattr(d, "strict_check", lambda c, *a: {"how": "stored", "pass": True})
    seen = {}

    def window(extra, **kw):
        seen.update(extra=extra, **kw)
        return {}

    ctx.window = window
    monkeypatch.setattr(d, "_sensitivity_answer", lambda c, args, objective, inputs, doc, strict: (0, {}))
    assert d._sensitivity(ctx, ctx.args)[0] == 0
    assert seen["extra"][:4] == ["--objective", str(ctx.out / "objective.json"), "--dump-grad", "3,46"]
    assert "--resident" not in seen["extra"], "one objective needs no loaded program"
    if checked:
        assert seen["unchecked_forward"] is None and seen["replace_args"] is None and "--x0-dir" not in seen["extra"]
        assert ctx.receipt_extra["gradient_line"] == {"on": False, "why": ctx.gradient_line_off}
    else:
        assert seen["unchecked_forward"] is not None and "--x0-dir" in seen["extra"]
        assert seen["replace_args"] == {"--tape-gib": "33", "--scratch-gib": "4"}
        assert seen["extra_env"]["AD_SNAP_REUSE"] == "1" and ctx.receipt_extra["gradient_line"]["on"] is True
    assert ctx.receipt_extra["objective_kept"] == {"how": "kept"}
    assert ctx.stages == ["inputs (objective kept from an earlier command, field map)"]
