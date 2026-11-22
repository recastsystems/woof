# SPDX-License-Identifier: Apache-2.0
"""What the 3 km storm sensitivity really costs a user, said before it runs: the host memory and disk its step store
uses (and an early refusal when the machine lacks them); the answer check's windows on the gradient line's words, so a
first call does not pay a second, slower route; a verdict shipped with a packed example, taken only on its exact key
and said in one line (--strict reruns it); and the direction of the check named beside every PASS.

THE BREAKAGES THESE HOLD SHUT (a 59-step storm sensitivity through the door, one 96 GB card, 2026-11-14): the door
said 140,000 MiB of host memory and named no disk, and the window held 278 GiB of host memory and wrote 445 GB to its
run folder; the check that followed kept every step boundary too; a whole-state PASS was printed with nothing beside it
although the storm objective's direction has read over the bar on this box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"


@pytest.fixture
def table():
    return d.load_table()


def _assets(tmp_path, row):
    out = {}
    for key in set(row["asset_args"].values()) | set(row["asset_env"].values()):
        p = tmp_path / "a" / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        out[key] = p
    return out


def _ctx(tmp_path, table, steps=59, door="sensitivity"):
    ns = argparse.Namespace(checked_forward=False, boundary_layer=None, first_gradient=None, wait_card=0.0,
                            no_self_dot=False, strict_from=None, self_dot=False, seed=None)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    # the images of the row's boundary-layer routes as stand-in files, the table's pins moved to them (its own copy)
    for route in table["rows"][STORM]["gradient_line"]["boundary_layer"]["routes"].values():
        for var, item in (route.get("images") or {}).items():
            path = tmp_path / "assets" / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(var.encode())
            item["sha256"] = hashlib.sha256(var.encode()).hexdigest()
    return SimpleNamespace(table=table, row=table["rows"][STORM], row_name=STORM, door=door, root=tmp_path / "assets",
                           card="0", cards=["0"], arch="sm_120", split=None, out=out, args=ns, steps=steps)


# --------------------------------------------------------------------------------- host and disk, said and refused
def test_the_gradient_line_states_the_host_and_disk_its_checkpointed_store_uses(table):
    row = table["rows"][STORM]
    need = d.store_need(row, 59, 400_000, line_on=True)
    assert need["boundaries"] == 10                                  # boundary 0, the end and ceil(sqrt(59)) = 8
    assert need["store_mib"] == 10 * row["boundary_mib"]
    assert need["ram_mib"] == need["store_mib"] and need["files_mib"] == 0   # 150 GiB of host store holds them all
    assert need["host_mib"] == row["gradient_line"]["host"]["base_mib"] + need["store_mib"]
    assert need["disk_mib"] == row["disk_margin_mib"]
    # a smaller host: the store budget is 0.6 of it, the rest goes to files, and the need says so
    small = d.store_need(row, 59, 150_000, line_on=True)
    cap = int(150_000 * 0.6 / 1024) * 1024
    assert small["ram_mib"] == cap and small["files_mib"] == small["store_mib"] - cap
    assert small["disk_mib"] == small["files_mib"] + row["disk_margin_mib"]
    with pytest.raises(d.HostMemoryError):
        d.refuse_host_too_small(STORM, small["host_mib"], 150_000)


def test_a_window_that_keeps_every_boundary_states_the_disk_it_writes(table):
    row = table["rows"][STORM]
    need = d.store_need(row, 59, 400_000, line_on=False)
    assert need["boundaries"] == 60 and need["ram_mib"] == 160_000
    # its host need is the runner's own memory and what it keeps of the store there (test_ad_door_window_host.py)
    assert need["host_mib"] == 260_196 and need["line"] is False
    # about the 445 GB the 59-step door run wrote with every boundary kept
    assert 400_000 < need["files_mib"] < 460_000


def test_the_door_refuses_before_any_download_when_the_run_folders_disk_is_short(table, tmp_path):
    need = d.store_need(table["rows"][STORM], 59, 400_000, line_on=False)
    with pytest.raises(d.DiskSpaceError) as err:
        d.refuse_disk_too_small(STORM, need, tmp_path / "run", free=100_000)
    words = str(err.value)
    assert err.value.exit_code == d.HOST_TOO_SMALL_EXIT
    assert "100,000 MiB free" in words and f"{need['disk_mib']:,} MiB" in words and "Nothing but the row's mesh file was downloaded" in words
    assert d.refuse_disk_too_small(STORM, need, tmp_path / "run", free=need["disk_mib"]) == need["disk_mib"]
    assert d.refuse_disk_too_small("x", None, tmp_path, free=1) == 1   # a row with no measured boundary: not judged


def test_a_row_without_a_measured_boundary_is_not_judged(table):
    assert d.store_need(table["rows"]["hrrrv5-x1.40962"], 3, 400_000, line_on=False) is None


# ------------------------------------------------------------------------- the built commands, gradient and check
def test_the_storm_gradient_window_carries_the_fast_routes_words(tmp_path, table, monkeypatch):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 400_000)
    ctx = _ctx(tmp_path, table)
    line = d.gradient_line(ctx, ctx.args)
    words = d.gradient_line_words(ctx, line, ctx.args, ctx.out)
    argv, env = d.window_command(table, ctx.row, _assets(tmp_path, ctx.row), "sensitivity", Path("ad-binder"), 59,
                                 tmp_path / "w", words["extra"], extra_env=words["env"], replace_args=words["replace"])
    assert argv[argv.index("--checkpoints") + 1] == "sqrt"
    assert argv[argv.index("--bstore-ram-gib") + 1] == "150"
    assert argv[argv.index("--tape-gib") + 1] == "33" and argv[argv.index("--scratch-gib") + 1] == "4"
    assert "--x0-dir" in argv
    assert env["AD_RECOMPUTE_UNCHECKED"] == "1" and env["AD_SNAP_REUSE"] == "1" and env["AD_BS_PREFILL"] == "8"
    # the default route: the boundary layer's backward without its tape, its two images, the readback words
    assert env["AD_P3_DIRECT"] == "1" and env["AD_P3_DIRECT_IMAGE"].endswith("direct120.cubin")
    assert env["AD_P3_DIRECT_FWD"].endswith("cap120.cubin")
    assert env["AD_TAPE_PREDICT"] == "1" and env["AD_TAPE_BLOCK_KIB"] == "0"


class _Recorder:
    def __init__(self, ctx):
        self.calls, self.ctx = [], ctx

    def __call__(self, extra, sub="window", env_overrides=None, door_args_key="window_args", extra_env=None,
                 spec_door=None, replace_args=None, unchecked_forward=None):
        self.calls.append({"extra": list(extra), "sub": sub, "door_args_key": door_args_key,
                           "extra_env": dict(extra_env or {}), "spec_door": spec_door, "replace": replace_args})
        return {"self_dot": {"rel": 5e-08, "allocations": 107}, "door_validity": {"valid": True}}


def test_the_answer_checks_window_takes_the_lines_words_but_not_the_moved_start(tmp_path, table, monkeypatch):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 400_000)
    ctx = _ctx(tmp_path, table)
    line = d.gradient_line(ctx, ctx.args)
    words = d.gradient_line_words(ctx, line, ctx.args, ctx.out)
    ctx.check_words = d.check_line_words(words)
    ctx.window = _Recorder(ctx)
    check = d.strict_check(ctx)
    [call] = ctx.window.calls
    assert call["sub"] == "selfdot" and call["spec_door"] == "adjoint-test"
    argv, env = d.window_command(table, ctx.row, _assets(tmp_path, ctx.row), "sensitivity", Path("ad-binder"), 59,
                                 tmp_path / "w", call["extra"], door_args_key=call["door_args_key"],
                                 extra_env=call["extra_env"], spec_door=call["spec_door"],
                                 replace_args=call["replace"])
    assert argv[argv.index("--checkpoints") + 1] == "sqrt", "the check keeps checkpoints, not every boundary"
    assert argv[argv.index("--bstore-ram-gib") + 1] == "150", "its checkpoints are kept in host memory"
    assert "--x0-dir" not in argv, "the check compares every forward step boundary with the recording"
    assert env["AD_P3_DIRECT"] == "1" and "AD_TAPE_PREDICT" not in env and "AD_TAPE_BLOCK_KIB" not in env, \
        "the check runs the default route as the stack's check ran it: tape-free, without the readback words"
    assert argv[argv.index("--tape-gib") + 1] == "33" and argv[argv.index("--scratch-gib") + 1] == "4", \
        "the check runs the gradient's own tape and scratch sizes, as the stack's derivative check did"
    assert env["AD_DOT_SELF"] == "1" and env["AD_RECOMPUTE_UNCHECKED"] == "1" and env["AD_SNAP_REUSE"] == "1"
    assert check["words"]["route"].startswith("the gradient line's") and "--x0-dir" in check["words"]["not_taken"]


def test_a_door_without_the_line_checks_on_the_rows_own_words(tmp_path, table):
    ctx = _ctx(tmp_path, table, door="tangent")
    ctx.window = _Recorder(ctx)
    d.strict_check(ctx)
    [call] = ctx.window.calls
    assert "--checkpoints" not in call["extra"] and "AD_SNAP_REUSE" not in call["extra_env"]


def test_the_check_key_tells_the_lines_words_from_the_rows(tmp_path, table, monkeypatch):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 400_000)
    ctx = _ctx(tmp_path, table)
    words = d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args, ctx.out)
    a = d._check_line_key(d.check_line_words(words))
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 200_000)   # another store budget, same key
    b = d._check_line_key(d.check_line_words(d.gradient_line_words(ctx, d.gradient_line(ctx, ctx.args), ctx.args,
                                                                    ctx.out)))
    assert a == b and a is not None and d._check_line_key(None) is None


# --------------------------------------------------------------------------------------- the shipped verdict
def _key(**over):
    key = {"plan_sha256": "p" * 64, "runner_sha256": "r" * 64, "runner_assets": {"kernels": "k"}, "kset_assets": {},
           "arch_assets": {}, "card_family": "sm_120", "steps": 59, "seed": 1, "check_sha256": "c",
           "check_line": "l", "left_out_list": None, "door_swap": None, "cards": 1, "split_ranks": None, "row": STORM,
           "row_sha256": "w", "plan_files": "stamps of this user's files", "door_code": {"version": "x"}}
    key.update(over)
    return key


def test_a_shipped_verdict_is_taken_only_on_the_packed_example_and_its_exact_key(table):
    row = dict(table["rows"][STORM])
    entry = {"key_sha256": d.shipped_key_sha256(_key()), "value": 5e-08, "bar": 1e-07, "pass": True,
             "card_family": "sm_120", "measured_utc": "2026-11-14 03:00Z"}
    row["shipped_verdicts"] = [entry]
    example = f"the row's example plan ({row['example_plan']})"
    assert d.shipped_verdict(row, _key(), example) == (entry, None)
    assert d.shipped_verdict(row, _key(plan_files="another user's stamps", door_code={"v": 2}), example)[0] is entry
    for part, value in (("plan_sha256", "q" * 64), ("runner_sha256", "s" * 64), ("runner_assets", {"kernels": "j"}),
                        ("card_family", "sm_90"), ("steps", 12), ("check_line", None)):
        taken, why = d.shipped_verdict(row, _key(**{part: value}), example)
        assert taken is None and "another plan, runner program, kernel set or card family" in why, part
    assert d.shipped_verdict(row, _key(), "given (--plan)") == (None, None), "never for a plan of the user's own"
    assert d.shipped_verdict(table["rows"][STORM], _key(), example) == (None, None), "none shipped in this build"


def test_a_shipped_verdict_says_who_made_it_and_a_miss_is_a_miss(tmp_path, table, capsys):
    ctx = _ctx(tmp_path, table)
    miss = d._shipped_check(ctx, "sensitivity", {"value": 2.04e-07, "bar": 1e-07, "pass": False,
                                                 "card_family": "sm_120", "measured_utc": "2026-11-14 03:00Z"})
    printed = capsys.readouterr().out
    assert miss["how"] == "shipped" and miss["pass"] is False
    assert "made by the publisher on an sm_120 card" in printed and "NOT within the bar" in printed and "1.71 roundings" in printed
    assert "--strict reruns it on your card" in printed
    lying = d._shipped_check(ctx, "sensitivity", {"value": 2.04e-07, "bar": 1e-07, "pass": True,
                                                  "card_family": "sm_120", "measured_utc": "x"})
    assert lying["pass"] is False, "a reading over the bar is never taken as a pass"


def test_strict_is_the_plain_word_for_running_the_check_here():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    for door in ("sensitivity", "tangent", "param-grad"):
        assert parser.parse_args([door, "--out", "x", "--strict"]).self_dot is True
        assert parser.parse_args([door, "--out", "x"]).self_dot is False


# ------------------------------------------------------------------------------------- the direction named
def test_every_verdict_names_its_direction_and_the_storm_directions_reading(tmp_path, table, capsys):
    ctx = _ctx(tmp_path, table, door="tangent")
    ctx.window = _Recorder(ctx)
    check = d.strict_check(ctx)
    assert "whole-state direction" in check["direction"] and "over 59 steps" in check["direction"]
    assert "14.75 minutes" in check["direction"]
    assert "1.67e-06 on the tape-free route" in check["other_directions"]
    assert "2.18e-06 on the exact (taped) route against 1e-07 at 59 steps" in check["other_directions"]
    assert "over the bar on both" in check["other_directions"] and "6.3e-08" in check["other_directions"]
    assert "pass" not in check["other_directions"].replace("whole-state PASS does not", "").lower()
    d.print_strict(check)
    printed = capsys.readouterr().out
    assert "direction: the runner's own seeded whole-state direction" in printed
    assert "other directions: the storm-sensitivity direction" in printed and "1e-07" in printed


def test_a_window_under_an_hour_is_written_in_minutes():
    assert d.window_length_words(59 * 15.0) == "14.75 min" and d.window_length_words(3 * 720.0) == "36 min"
    assert d.window_length_words(3600.0) == "1 h" and d.window_length_words(29 * 360.0) == "2.9 h"
