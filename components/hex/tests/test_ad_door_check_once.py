# SPDX-License-Identifier: Apache-2.0
"""The strict whole-state check runs once per plan: the first command on an operator's key runs its two windows and
keeps the verdict; a later tangent, sensitivity or param-grad on the same key takes it, says so in one line and names
it in its receipt; adjoint-test always runs it; any change of key, a failed or unreadable verdict and --self-dot run
it again.

THE BREAKAGE THESE HOLD SHUT: at 120 km (fresh install, one 96 GB card, receipts f41) a sensitivity took 175.2 s, of
which the gradient window was 41.6 s and the strict check's two windows 63.7 s and 63.2 s, run again on every command
although the plan, the runner and the kernel set had not changed. And the reverse: a verdict taken for an operator it
was not made on (another kernel set, runner, step count, card family or a touched plan file) would make a receipt
read valid on a check nobody ran.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d


@pytest.fixture
def op(tmp_path, monkeypatch):
    """One operator on disk: a plan package, a runner build, an asset root with a kernel set and a left-out list."""
    monkeypatch.setenv(d.CHECKS_ENV, str(tmp_path / "checks"))
    for k in list(os.environ):
        if k.startswith(d._KEY_ENV_PREFIXES):
            monkeypatch.delenv(k)
    plan = tmp_path / "plan"
    (plan / "recording" / "payloads").mkdir(parents=True)
    (plan / "plan.json").write_text('{"launches": []}')
    (plan / "recording" / "payloads" / "p-0001.bin").write_bytes(b"\0" * 64)
    os.symlink(tmp_path, plan / "<site-packages>")
    root = tmp_path / "assets"
    kset = root / "latest" / "kset" / "kset-d12"
    (kset / "cubins").mkdir(parents=True)
    (kset / "kernels.json").write_text('{"set": "d12"}')
    (kset / "cubins" / "k1.cubin").write_bytes(b"cubin-1")
    (root / "rows" / "hrrrv5-any").mkdir(parents=True)
    (root / "rows" / "hrrrv5-any" / "refuse.txt").write_text("a,b")
    (root / "latest" / "lists").mkdir()
    (root / "latest" / "lists" / "dead.json").write_text(json.dumps(
        {"ranges": [{"allocation": 417, "offset": 0, "bytes": 9337888}]}))
    runner = tmp_path / "ad-binder"
    runner.write_bytes(b"runner build 1")
    return SimpleNamespace(tmp=tmp_path, plan=plan, root=root, kset=kset, runner=runner, n=0)


def _door(op, door="sensitivity", steps=3, arch="sm_120", value=8.1e-09, proof_words=None, window_valid=True,
          row_extra=None, **args):
    """A door's context on the operator ``op`` (its own output folder), with windows that write what the runner would."""
    op.n += 1
    out = op.tmp / f"out{op.n}-{door}"
    out.mkdir()
    calls = []
    assets = {"kernels": op.kset / "kernels.json", "cubins": op.kset / "cubins",
              "refuse": op.root / "rows" / "hrrrv5-any" / "refuse.txt"}
    row = {"what": "w", "mesh": "x1.40962", "suite": "hrrrv5", "dt_s": 720.0, "need_free_mib": 1,
           "window_args": ["--fct-frozen", "1"],
           "self_dot_dead": {"list": "latest/lists/dead.json", "recipe_proof": "latest/proof/g1-120D"},
           **(row_extra or {})}
    ctx = SimpleNamespace(
        door=door, table=d.load_table(), row_name="hrrrv5-x1.40962", row=row, steps=steps, plan=op.plan,
        plan_how="the row's example plan", fmap=SimpleNamespace(doc={"plan_sha256": "975d" * 16},
                                                                source=op.plan / "alloc-map.json", found=[],
                                                                end_boundary=lambda: None),
        runner=op.runner, root=op.root, assets=assets, kset_assets={}, arch_assets={}, arch=arch, card="0", cards=["0"],
        out=out, stages=[], started=0.0, windows=[],
        args=SimpleNamespace(seed=args.get("seed"), self_dot=args.get("self_dot", False), need_mib=None))
    ctx.need_mib = lambda: 1
    ctx._refused_names = lambda: ["a", "b"]
    ctx.runner_assets = lambda: {k: str(Path(v).relative_to(op.root)) for k, v in assets.items()}

    def window(extra, sub="window", env_overrides=None, door_args_key="window_args", extra_env=None, spec_door=None):
        calls.append(sub)
        words = "f34021512af41b2e"
        if sub == "selfdot-proof":
            (out / sub).mkdir(exist_ok=True)
            (out / sub / "stderr").write_text(json.dumps({"event": "window_dead_poison_rev", "step": 3, "at": "walk",
                                                          "ranges": 1, "bytes": 9337888}) + "\n")
            words = proof_words or words
        ctx.windows.append({"window": sub, "card_peak_mib": 1, "wall_s": 63.7, "valid": window_valid,
                            "boundaries_checked": 3, "boundaries_equal": 3, "refused_beyond_row_list": []})
        return {"dots": [{"rel_aligned": value, "reverse": [{"result_sha256": words + "0" * 48}]}],
                "self_dot": {"allocations": 105}, "door_validity": {"boundaries_checked": 3, "boundaries_equal": 3}}

    ctx.window = window
    return ctx, calls


def _run(ctx):
    """strict_check then the receipt, as every door calls them."""
    check = d.strict_check(ctx)
    return check, d.Context.receipt(ctx, {}, True, check)


# ------------------------------------------------------------------------------------------------ the main path
def test_the_first_command_runs_and_keeps_the_check_and_a_later_one_takes_it_and_says_so(op, capsys):
    first, calls = _door(op, "sensitivity")
    check, rec = _run(first)
    assert calls == ["selfdot", "selfdot-proof"] and check["how"] == "own" and check["pass"]
    assert "not yet run on this plan" in capsys.readouterr().out
    kept = Path(rec["strict_check"]["kept_as"])
    assert kept.is_file() and kept.parent == op.tmp / "checks" and rec["valid"] is True
    verdict = json.loads(kept.read_text())
    assert verdict["pass"] is True and verdict["receipt"] == str(first.out / "receipt.json")
    assert [w["window"] for w in verdict["windows"]] == ["selfdot", "selfdot-proof"]

    later, calls = _door(op, "sensitivity")          # another objective and output folder: same operator, same key
    check, rec = _run(later)
    printed = capsys.readouterr().out
    assert calls == [], "the two check windows ran again on an unchanged operator"
    assert check["how"] == "stored" and check["pass"] and check["value"] == 8.1e-09 and check["bar"] == 4 * 2.0 ** -23
    assert "answer check passed on this plan on" in printed and "not run again" in printed
    assert "--strict runs it" in printed and "8.100e-09 against 4.77e-07 (0.07 of 4 single-precision roundings)" in printed
    assert len([x for x in printed.splitlines() if "answer check" in x]) == 1
    s = rec["strict_check"]["stored"]
    assert s["receipt"] == str(first.out / "receipt.json") and s["verdict"] == str(kept) and s["receipt_present"]
    assert s["by_door"] == "sensitivity" and [w["valid"] for w in s["windows"]] == [True, True]
    assert rec["valid"] is True and rec["passed"] is True and "kept_as" not in rec["strict_check"]
    assert rec["strict_check"]["proof"]["ok"] and rec["strict_check"]["left_out"]["bytes_total"] == 9337888
    assert "not run again" in rec["strict_check"]["what"]


def test_tangent_takes_the_verdict_a_sensitivity_kept_and_param_grad_checks_its_own_operator(op):
    _run(_door(op, "sensitivity")[0])
    tangent, calls = _door(op, "tangent")
    assert _run(tangent)[0]["how"] == "stored" and calls == []
    # param-grad swaps the scheme bundle for the parameter carrier: another operator, so another key
    pg, calls = _door(op, "param-grad")
    assert _run(pg)[0]["how"] == "own" and calls == ["selfdot", "selfdot-proof"]
    pg2, calls = _door(op, "param-grad")
    assert _run(pg2)[0]["how"] == "stored" and calls == []


def test_adjoint_test_always_runs_the_check_and_a_later_door_takes_its_verdict(op):
    _run(_door(op, "sensitivity")[0])
    at, calls = _door(op, "adjoint-test")
    check, rec = _run(at)
    assert calls == ["selfdot", "selfdot-proof"] and check["how"] == "own"
    se, calls = _door(op, "sensitivity")
    check, _ = _run(se)
    assert calls == [] and check["stored"]["by_door"] == "adjoint-test"
    assert check["stored"]["receipt"] == str(at.out / "receipt.json")


# ------------------------------------------------------------------------------- a changed key runs it again
def _plant(op, what):
    if what == "kernel set":
        (op.kset / "kernels.json").write_text('{"set": "d13"}')
    elif what == "a cubin of the kernel set":
        (op.kset / "cubins" / "k1.cubin").write_bytes(b"cubin-2")
    elif what == "runner":
        op.runner.write_bytes(b"runner build 2")
    elif what == "refuse list":
        (op.root / "rows" / "hrrrv5-any" / "refuse.txt").write_text("a")
    elif what == "touched plan file":
        payload = op.plan / "recording" / "payloads" / "p-0001.bin"
        st = payload.stat()
        os.utime(payload, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    elif what == "plan file rewritten with the same size and time":
        payload = op.plan / "recording" / "payloads" / "p-0001.bin"
        st = payload.stat()
        payload.write_bytes(b"\1" * 64)
        os.utime(payload, ns=(st.st_atime_ns, st.st_mtime_ns))
    elif what == "file added to the plan":
        (op.plan / "recording" / "payloads" / "p-0002.bin").write_bytes(b"\0")
    elif what == "left-out list":
        (op.root / "latest" / "lists" / "dead.json").write_text(json.dumps(
            {"ranges": [{"allocation": 418, "offset": 0, "bytes": 9337888}]}))
    else:
        raise AssertionError(what)


@pytest.mark.parametrize("what, words", [
    ("kernel set", "kernel set"),
    ("a cubin of the kernel set", "kernel set"),
    ("runner", "runner"),
    ("refuse list", "kernel set or runner files"),
    ("touched plan file", "plan files"),
    ("plan file rewritten with the same size and time", "plan files"),
    ("file added to the plan", "plan files"),
    ("left-out list", "left-out list"),
])
def test_a_planted_change_of_the_operator_runs_the_check_again(op, capsys, what, words):
    _run(_door(op)[0])
    _plant(op, what)
    capsys.readouterr()
    ctx, calls = _door(op)
    check, rec = _run(ctx)
    printed = capsys.readouterr().out
    assert calls == ["selfdot", "selfdot-proof"] and check["how"] == "own", what
    assert "was made with another" in printed and words in printed, printed
    # and the new verdict is the one the next command takes
    again, calls = _door(op)
    assert _run(again)[0]["how"] == "stored" and calls == []


@pytest.mark.parametrize("change, words", [
    ({"steps": 6}, "steps"),
    ({"arch": "sm_90"}, "card family"),
    ({"seed": 2}, "test direction"),
    ({"row_extra": {"window_args": ["--fct-frozen", "0"]}}, "row words"),
])
def test_another_step_count_card_family_direction_or_row_runs_the_check_again(op, capsys, change, words):
    _run(_door(op)[0])
    capsys.readouterr()
    ctx, calls = _door(op, **change)
    assert _run(ctx)[0]["how"] == "own" and calls == ["selfdot", "selfdot-proof"]
    assert words in capsys.readouterr().out


def test_a_runner_environment_word_of_the_user_is_part_of_the_key(op, monkeypatch):
    _run(_door(op)[0])
    monkeypatch.setenv("AD_TAPE_BLOCK", "65536")
    ctx, calls = _door(op)
    assert _run(ctx)[0]["how"] == "own" and calls


# ------------------------------------------------------------------------------- what is never taken
def test_a_failed_verdict_is_kept_and_never_taken(op, capsys):
    ctx, _ = _door(op, value=5e-07)                       # over the bar: 4.19 roundings of the dot
    check, rec = _run(ctx)
    assert not check["pass"] and rec["valid"] is False
    assert json.loads(Path(rec["strict_check"]["kept_as"]).read_text())["pass"] is False
    capsys.readouterr()
    again, calls = _door(op)
    check, _ = _run(again)
    assert calls == ["selfdot", "selfdot-proof"] and check["how"] == "own" and check["pass"]
    assert "is not taken: it did not pass" in capsys.readouterr().out


def test_a_verdict_whose_proof_failed_is_not_taken(op, capsys):
    ctx, _ = _door(op, proof_words="0000000000000000")   # the poisoned walk-back moved the gradient
    assert not _run(ctx)[0]["pass"]
    capsys.readouterr()
    again, calls = _door(op)
    assert _run(again)[0]["how"] == "own" and calls
    assert "did not pass" in capsys.readouterr().out


def test_a_passing_dot_on_windows_that_were_not_valid_is_kept_as_not_passed(op, capsys):
    ctx, _ = _door(op, window_valid=False)
    _, rec = _run(ctx)
    assert rec["valid"] is False
    assert json.loads(Path(rec["strict_check"]["kept_as"]).read_text())["pass"] is False
    capsys.readouterr()
    again, calls = _door(op)
    assert _run(again)[0]["how"] == "own" and calls
    assert "did not pass" in capsys.readouterr().out


def test_an_edited_or_unreadable_verdict_is_not_taken(op, capsys):
    _, rec = _run(_door(op)[0])
    kept = Path(rec["strict_check"]["kept_as"])
    doc = json.loads(kept.read_text())
    doc["key"]["runner_sha256"] = "0" * 64              # a verdict file copied from another operator
    kept.write_text(json.dumps(doc))
    ctx, calls = _door(op)
    assert _run(ctx)[0]["how"] == "own" and calls
    assert "its key is not this run's" in capsys.readouterr().out
    kept.write_text("{not json")
    ctx, calls = _door(op)
    assert _run(ctx)[0]["how"] == "own" and calls
    assert "cannot be read" in capsys.readouterr().out


def test_self_dot_runs_the_check_with_a_verdict_kept(op, capsys):
    _run(_door(op)[0])
    ctx, calls = _door(op, self_dot=True)
    check, rec = _run(ctx)
    assert calls == ["selfdot", "selfdot-proof"] and check["how"] == "own"
    assert "--self-dot: running it" in capsys.readouterr().out


def test_no_self_dot_claims_no_verdict_and_keeps_none(op):
    ctx, calls = _door(op)
    check = d.strict_check(ctx, no_self_dot=True)
    rec = d.Context.receipt(ctx, {}, True, check)
    assert calls == [] and rec["valid"] is None and "kept_as" not in rec["strict_check"]
    assert not (op.tmp / "checks").exists()
    ctx, calls = _door(op)                                 # nothing kept: the next command runs it
    assert _run(ctx)[0]["how"] == "own" and calls


def test_a_cache_that_cannot_be_written_is_said_and_the_receipt_keeps_nothing(op, monkeypatch, capsys):
    blocker = op.tmp / "a-file"
    blocker.write_text("")
    monkeypatch.setenv(d.CHECKS_ENV, str(blocker / "checks"))
    ctx, _ = _door(op)
    check, rec = _run(ctx)
    assert check["pass"] and rec["valid"] is True and rec["strict_check"]["kept_as"] is None
    assert "its verdict was not kept" in capsys.readouterr().err
    assert json.loads((ctx.out / "receipt.json").read_text())["strict_check"]["kept_as"] is None


def test_a_context_that_cannot_name_its_operator_always_runs_the_check(op):
    ctx, calls = _door(op)
    del ctx.fmap
    assert d.check_key(ctx) is None
    check = d.strict_check(ctx)
    assert check["how"] == "own" and calls and not hasattr(ctx, "check_key")
    ctx2, _ = _door(op, arch=None)
    assert d.check_key(ctx2) is None


def test_the_key_leaves_out_what_the_check_does_not_read(op):
    a, _ = _door(op, "sensitivity")
    b, _ = _door(op, "tangent")
    b.card, b.cards = "3", ["3"]
    assert d.check_key(a) == d.check_key(b)             # another door, output folder and card index: one key
    assert d.check_verdict_path(d.check_key(a)).name.startswith("975d975d975d975d-")


# ------------------------------------------------------------------------------- words and table
def test_the_table_says_which_doors_take_a_kept_verdict_and_which_always_check():
    once = d.load_table()["doors"]["adjoint-test"]["self_dot"]["once_per_plan"]
    assert set(once["reuse"]) == {"tangent", "sensitivity", "param-grad"} and once["always"] == ["adjoint-test"]
    assert not set(once["reuse"]) & set(once["always"]) and set(once["store_from"]) >= set(d.DOORS)


def test_self_dot_is_a_word_of_the_doors_that_take_a_verdict_and_opposite_words_are_refused():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="cmd"))
    for door in ("tangent", "sensitivity", "param-grad"):
        assert parser.parse_args([door, "--out", "o", "--self-dot"]).self_dot is True
        assert parser.parse_args([door, "--out", "o"]).self_dot is False
    with pytest.raises(SystemExit):
        parser.parse_args(["adjoint-test", "--out", "o", "--self-dot"])   # it always checks
    with pytest.raises(d.AdDoorError, match="opposites"):
        d.refuse_strict_words(SimpleNamespace(self_dot=True, no_self_dot=True, strict_from=None))
    with pytest.raises(d.AdDoorError, match="alternatives"):
        d.refuse_strict_words(SimpleNamespace(self_dot=True, no_self_dot=False, strict_from=Path("r.json")))
    d.refuse_strict_words(SimpleNamespace(self_dot=False, no_self_dot=True, strict_from=None))
