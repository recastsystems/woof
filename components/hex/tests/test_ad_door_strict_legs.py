# SPDX-License-Identifier: Apache-2.0
"""tangent and sensitivity carry the strict whole-state check of their own operator, and a window with no tangent
forward does not report its refused kinds as 0.

THE BREAKAGES THESE HOLD SHUT (box-d d120-g, 2026-11-11, 120 km on pk120e4): the tangent and sensitivity receipts read
valid with no strict check of their own (their windows read valid_derivative false, their receipts carried no
self_dot); and the sensitivity window read refused {}, refuse_list_kinds 0 and complete_operator true on the same
--refuse list on which the tangent window counted 64 kinds, because the runner counts launch kinds only while a
tangent forward runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d


# ------------------------------------------------------------------------------------------- refused-kind counts
def _window_out(tmp_path, refused):
    out = tmp_path / "w"
    out.mkdir()
    step = {"event": "window_tangent_step", "reference_check": True, "boundary_hashes": "equal"}
    (out / "stderr").write_text("\n".join(json.dumps({**step, "step": s}) for s in (1, 2, 3)) + "\n")
    return out, {"refused": {f"REFUSED:{k}: refused for this run (--refuse)": 1 for k in refused}}


def test_a_window_without_a_tangent_forward_reports_its_refusals_as_not_counted(tmp_path):
    out, doc = _window_out(tmp_path, [])
    argv = ["ad-binder", "window", "--seeds", "1", "--tangent-only", "1", "--no-tangent", "1", "--refuse", "a,b,c"]
    v = d.window_validity(out, doc, argv)
    assert v["refused_kinds"] is None and v["refused_by_row_list"] is None     # not 0: nothing was counted
    assert v["refused_counted"] is False and v["refused_listed"] == 3
    assert v["runner_complete_operator_counted"] is False and "tangent forward" in v["refused_why"]
    assert v["valid"] and v["boundaries_equal"] == 3


def test_a_tangent_window_still_counts_its_refusals(tmp_path):
    out, doc = _window_out(tmp_path, ["a", "b", "c"])
    argv = ["ad-binder", "window", "--tangent-only", "1", "--refuse", "a,b,c"]
    v = d.window_validity(out, doc, argv)
    assert v["refused_kinds"] == 3 and v["refused_by_row_list"] == 3 and "refused_counted" not in v
    # a kind the list does not name is still caught when counted
    (tmp_path / "x").mkdir()
    out2, doc2 = _window_out(tmp_path / "x", ["a", "z"])
    v2 = d.window_validity(out2, doc2, argv)
    assert v2["refused_beyond_row_list"] == ["z"] and not v2["valid"]


# ------------------------------------------------------------------------------------- the self-dot window's spec
def _row():
    return {"asset_args": {}, "asset_env": {}, "window_args": ["--fct-frozen", "1"], "window_env": {}}


def test_the_strict_windows_of_tangent_and_sensitivity_use_the_adjoint_test_self_dot_args(tmp_path):
    table = d.load_table()
    for door in ("tangent", "sensitivity"):
        argv, env = d.window_command(table, _row(), {}, door, Path("/r"), 3, tmp_path, [],
                                     door_args_key="self_dot_args", extra_env={"AD_DOT_SELF": "1"},
                                     spec_door="adjoint-test")
        assert argv[argv.index("--direction") + 1] == "field" and "--no-tangent" not in argv
        assert env["AD_DOT_SELF"] == "1"
        plain, _ = d.window_command(table, _row(), {}, door, Path("/r"), 3, tmp_path, [])
        assert plain[-len(table["doors"][door]["window_args"]):] == table["doors"][door]["window_args"]


# ------------------------------------------------------------------------------------------ own strict windows
def _ctx(tmp_path, dead=True, words=("f34021512af41b2e", "f34021512af41b2e"), nbytes=9337888):
    root = tmp_path / "assets"
    (root / "lists").mkdir(parents=True)
    (root / "lists" / "dead.json").write_text(json.dumps(
        {"ranges": [{"allocation": 417, "offset": 0, "bytes": 9337888}]}))
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def window(extra, sub="window", env_overrides=None, door_args_key="window_args", extra_env=None, spec_door=None):
        calls.append({"sub": sub, "key": door_args_key, "env": dict(extra_env or {}), "spec_door": spec_door})
        w = words[0] if sub == "selfdot" else words[1]
        if sub == "selfdot-proof":
            (out / sub).mkdir(exist_ok=True)
            (out / sub / "stderr").write_text(json.dumps({"event": "window_dead_poison_rev", "step": 3, "at": "walk",
                                                          "ranges": 1, "bytes": nbytes}) + "\n")
        return {"dots": [{"rel_aligned": 8.1e-09, "reverse": [{"result_sha256": w + "0" * 48}]}],
                "self_dot": {"allocations": 107}, "door_validity": {"boundaries_checked": 3, "boundaries_equal": 3}}

    row = {"self_dot_dead": {"list": "lists/dead.json", "recipe_proof": "latest/proof/g1-120D"}} if dead else {}
    return SimpleNamespace(table=d.load_table(), row=row, root=root, out=out, window=window), calls


def test_tangent_and_sensitivity_run_the_self_dot_and_its_poison_proof_on_their_own_plan(tmp_path):
    ctx, calls = _ctx(tmp_path)
    check = d.strict_check(ctx)
    assert [c["sub"] for c in calls] == ["selfdot", "selfdot-proof"]
    assert all(c["spec_door"] == "adjoint-test" and c["key"] == "self_dot_args" for c in calls)
    assert calls[0]["env"]["AD_DOT_DEAD"].endswith("lists/dead.json") and "AD_DOT_DEAD_POISON" not in calls[0]["env"]
    assert calls[1]["env"]["AD_DOT_DEAD_POISON"] == "rev"
    assert check["pass"] and check["how"] == "own" and check["allocations"] == 107
    assert check["left_out"]["bytes_total"] == 9337888 and check["proof"]["ok"]


def test_a_proof_whose_gradient_moved_fails_the_strict_check(tmp_path):
    ctx, _ = _ctx(tmp_path, words=("f34021512af41b2e", "0000000000000000"))
    check = d.strict_check(ctx)
    assert not check["pass"] and not check["proof"]["ok"]


def test_no_self_dot_leaves_the_check_not_run(tmp_path):
    ctx, calls = _ctx(tmp_path)
    check = d.strict_check(ctx, no_self_dot=True)
    assert calls == [] and check["how"] == "not run" and not check["pass"]


# --------------------------------------------------------------------------------------------------- the receipt
def _receipt_ctx(tmp_path, windows):
    runner = tmp_path / "ad-binder"
    runner.write_bytes(b"runner")
    root = tmp_path / "assets"
    root.mkdir(exist_ok=True)
    out = tmp_path / "rc"
    out.mkdir(exist_ok=True)
    fmap = SimpleNamespace(doc={"plan_sha256": "4ba2" * 16}, source=tmp_path / "map.json", found=1,
                           end_boundary=lambda: 4)
    ctx = SimpleNamespace(door="tangent", row_name="hrrrv5-x1.40962",
                          row={"what": "w", "mesh": "x1.40962", "suite": "hrrrv5", "dt_s": 720.0, "need_free_mib": 1},
                          steps=3, plan=tmp_path, plan_how="given (--plan)", fmap=fmap, runner=runner, root=root,
                          card="0", cards=["0"], stages=[], started=0.0, arch="sm_120", arch_assets={},
                          kset_assets={"kernels": root / "latest/kset/kset-d12/kernels.json"}, out=out,
                          windows=windows, args=SimpleNamespace(need_mib=None), assets={})
    ctx.need_mib = lambda: 1
    ctx._refused_names = lambda: ["a", "b"]
    ctx.runner_assets = lambda: {"kernels": "latest/kset/kset-d12/kernels.json", "refuse": "rows/hrrrv5-any/refuse.txt"}
    return ctx


def test_the_receipt_is_valid_only_with_a_passing_strict_check(tmp_path):
    ok_window = [{"window": "window", "card_peak_mib": 1, "valid": True}]
    ctx = _receipt_ctx(tmp_path, ok_window)
    passing = {"how": "own", "pass": True}
    assert d.Context.receipt(ctx, {}, True, passing)["valid"] is True
    failing = {"how": "own", "pass": False}
    rec = d.Context.receipt(ctx, {}, True, failing)
    assert rec["valid"] is False and rec["passed"] is False and rec["strict_check"] == failing
    rec = d.Context.receipt(ctx, {}, True, {"how": "not run", "pass": False})
    assert rec["valid"] is None and rec["passed"] is True          # not checked: never true, never a failure
    bad_window = [{"window": "window", "card_peak_mib": 1, "valid": False, "refused_beyond_row_list": ["z"],
                   "boundaries_equal": 3, "boundaries_checked": 3}]
    assert d.Context.receipt(_receipt_ctx(tmp_path, bad_window), {}, True, passing)["valid"] is False


# ------------------------------------------------------------------------------------- a bound adjoint-test receipt
def _adjoint_receipt(ctx, **over):
    doc = {"schema": d.RECEIPT_SCHEMA, "door": "adjoint-test", "passed": True, "valid": True, "card": "0",
           "plan_sha256": ctx.fmap.doc["plan_sha256"], "row": ctx.row_name, "mesh": ctx.row["mesh"],
           "suite": ctx.row["suite"], "steps": ctx.steps, "runner_sha256": d._sha256_file(ctx.runner),
           "assets": "/elsewhere/assets", "kset_assets": {"kernels": "/elsewhere/assets/latest/kset/kset-d12/kernels.json"},
           "arch_assets": {}, "refused_by_name": ["a", "b"], "wall_s": 239.8, "card_peak_mib": 71156,
           "runner_assets": {"kernels": "latest/kset/kset-d12/kernels.json", "refuse": "rows/hrrrv5-any/refuse.txt"},
           "result": {"checks": {"self_dot": {"value": 8.1e-09, "bar": 1e-07, "pass": True, "allocations": 107,
                                               "left_out": {"bytes_total": 9337888, "ranges_n": 1, "allocations": 1,
                                                            "list": "latest/lists/dead120-tail-s3.json"},
                                               "proof": {"ok": True, "words": "f340", "control_words": "f340"}}}}}
    doc.update(over)
    return doc


def test_a_passing_adjoint_test_receipt_of_the_same_operator_binds(tmp_path):
    ctx = _receipt_ctx(tmp_path, [])
    path = tmp_path / "at.json"
    path.write_text(json.dumps(_adjoint_receipt(ctx)))
    check = d.bind_strict(ctx, path)
    assert check["pass"] and check["how"] == "bound" and check["value"] == 8.1e-09 and check["allocations"] == 107
    assert check["bound"]["receipt_sha256"] == d._sha256_file(path)
    assert check["bound"]["matched"]["kset_assets"] == {"kernels": "latest/kset/kset-d12/kernels.json"}
    assert check["left_out"]["bytes_total"] == 9337888 and check["proof"]["ok"]
    assert d.strict_check(SimpleNamespace(**{**vars(ctx), "table": d.load_table()}), strict_from=path)["how"] == "bound"


@pytest.mark.parametrize("over, words", [
    ({"plan_sha256": "0" * 64}, "plan_sha256"),
    ({"runner_sha256": "0" * 64}, "runner_sha256"),
    ({"row": "hrrrv5-exact-x1.40962"}, "row"),
    ({"steps": 30}, "steps"),
    ({"refused_by_name": ["a"]}, "refused_by_name"),
    ({"kset_assets": {"kernels": "/elsewhere/assets/latest/kset/kset-d7/kernels.json"}}, "kset_assets"),
    ({"runner_assets": {"kernels": "latest/kset/kset-d7/kernels.json", "refuse": "rows/hrrrv5-any/refuse.txt"}},
     "runner_assets"),
    ({"runner_assets": None}, "does not name the files"),
    ({"passed": False}, "did not pass"),
    ({"door": "tangent"}, "not an adjoint-test"),
])
def test_a_receipt_of_another_operator_or_a_failing_one_is_refused_by_name(tmp_path, over, words):
    ctx = _receipt_ctx(tmp_path, [])
    path = tmp_path / "at.json"
    path.write_text(json.dumps(_adjoint_receipt(ctx, **over)))
    with pytest.raises(d.AdDoorError, match=words):
        d.bind_strict(ctx, path)


def test_a_bound_receipt_whose_left_out_memory_has_no_proof_is_refused(tmp_path):
    ctx = _receipt_ctx(tmp_path, [])
    doc = _adjoint_receipt(ctx)
    doc["result"]["checks"]["self_dot"]["proof"] = {"ok": False}
    path = tmp_path / "at.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(d.AdDoorError, match="no passing proof"):
        d.bind_strict(ctx, path)


def test_the_receipt_names_the_files_its_row_hands_the_runner(tmp_path):
    root = tmp_path / "assets"
    ctx = SimpleNamespace(root=root, row={"asset_args": {"--kernels": "kernels"}, "asset_env": {"AD_SCHEMES": "bundle"}},
                          assets={"kernels": root / "latest/kset/kset-d12/kernels.json", "bundle": Path("/b/gen-B9"),
                                  "refuse": root / "rows/hrrrv5-any/refuse.txt", "grid": root / "g.nc"})
    assert d.Context.runner_assets(ctx) == {"bundle": "/b/gen-B9", "kernels": "latest/kset/kset-d12/kernels.json",
                                            "refuse": "rows/hrrrv5-any/refuse.txt"}


def test_tangent_and_sensitivity_take_the_strict_flags():
    import argparse
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="cmd"))
    for door in ("tangent", "sensitivity"):
        a = parser.parse_args([door, "--out", "o", "--strict-from", "r.json"])
        assert a.strict_from == Path("r.json") and a.no_self_dot is False
        assert parser.parse_args([door, "--out", "o", "--no-self-dot"]).no_self_dot
