# SPDX-License-Identifier: Apache-2.0
"""param-grad on the rows made for any mesh (the hrrrv5-any template): the door finds the parameter-carrier bundle,
checks it, runs its strict whole-state check on that bundle, and labels a moved point.

THE BREAKAGES THESE HOLD SHUT (2026-11-11): on the three meshes that count (120 km, 60 km, the 3 km storm box) the
door refused param-grad by name, because the template named no carrier bundle (box-d d120-g param-grad/door.err); a
param-grad receipt carried no strict check at all, so "valid" said nothing about the derivative it ran; and a table
that moves a setting had no way through the door, so the gradient could not be compared with a real rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import params

FIVE = ["coare30_charnock", "mynnsfc_zil_kczil", "tofd_beljaars_a1", "mesoscale_gwd_xlinv", "flow_blocking_cd"]


def test_a_row_made_from_the_any_mesh_template_runs_param_grad():
    table = d.load_table()
    template = table["rows"]["hrrrv5-any"]
    assert "param-grad" in template["doors"] and "doors_why" not in template
    assert template["assets"]["bundle_params"] == "rows/hrrrv5-any/bundle-params"
    assert template["param_names"] == FIVE
    assert {params.BY_NAME[n].family for n in FIVE} == {"mynnsfc", "ugwp"}   # the four carrier images' two families
    assert len(template["param_bundle"]["sha256sums_sha256"]) == 64
    row = dict(template, mesh="x1.40962", dt_s=720)
    row.pop("template_only")
    name, got = d._row({**table, "rows": {**table["rows"], "hrrrv5-x1.40962": row}}, "hrrrv5-x1.40962", "param-grad")
    assert got is row
    exact = dict(table["rows"]["hrrrv5-any-exact"], mesh="x1.40962", dt_s=720)
    exact.pop("template_only")
    with pytest.raises(d.AdDoorError, match="parameter-carrier"):   # no carrier bundle of the exact composition
        d._row({**table, "rows": {**table["rows"], "e": exact}}, "e", "param-grad")


def test_the_door_refuses_a_missing_carrier_bundle_by_name(tmp_path):
    table = d.load_table()
    row = table["rows"][table["default_row"]]
    assets = {key: tmp_path / key for key in row["assets"]}
    for key, path in assets.items():
        if key != "bundle_params":
            path.write_text("x")
    with pytest.raises(d.AdDoorError, match="bundle_params.*missing"):
        d.window_command(table, row, assets, "param-grad", tmp_path / "hexad", 3, tmp_path / "o", [])
    d.window_command(table, row, assets, "sensitivity", tmp_path / "hexad", 3, tmp_path / "o", [])   # not its file
    del assets["bundle_params"]
    with pytest.raises(d.AdDoorError, match="bundle_params is not named by this row"):
        d.window_command(table, row, assets, "param-grad", tmp_path / "hexad", 3, tmp_path / "o", [])


def _strict_ctx(tmp_path, door):
    calls = []

    def window(extra, sub="window", env_overrides=None, door_args_key="window_args", extra_env=None, spec_door=None):
        calls.append({"sub": sub, "overrides": env_overrides, "spec_door": spec_door})
        return {"dots": [{"rel_aligned": 8.1e-09, "reverse": [{"result_sha256": "ab" * 32}]}],
                "self_dot": {"allocations": 107}, "door_validity": {"boundaries_checked": 3, "boundaries_equal": 3}}

    return SimpleNamespace(table=d.load_table(), row={}, root=tmp_path, out=tmp_path, window=window, door=door), calls


def test_the_strict_windows_of_param_grad_run_on_the_carrier_bundle(tmp_path):
    ctx, calls = _strict_ctx(tmp_path, "param-grad")
    check = d.strict_check(ctx)
    assert [c["sub"] for c in calls] == ["selfdot"] and check["how"] == "own"
    assert calls[0]["overrides"] == {"AD_SCHEMES": "bundle_params"} and calls[0]["spec_door"] == "adjoint-test"
    ctx, calls = _strict_ctx(tmp_path, "sensitivity")   # the other doors keep the row's own bundle
    d.strict_check(ctx)
    assert calls[0]["overrides"] is None
    # and the environment the runner gets: the carrier bundle, with the self-dot window's arguments
    table = d.load_table()
    row = {"asset_args": {}, "asset_env": {"AD_SCHEMES": "bundle"}, "window_args": [], "window_env": {}}
    (tmp_path / "b").mkdir()
    (tmp_path / "bp").mkdir()
    argv, env = d.window_command(table, row, {"bundle": tmp_path / "b", "bundle_params": tmp_path / "bp"}, "param-grad",
                                 Path("/r"), 3, tmp_path, [], {"AD_SCHEMES": "bundle_params"}, "self_dot_args",
                                 {"AD_DOT_SELF": "1"}, "adjoint-test")
    assert env["AD_SCHEMES"] == str(tmp_path / "bp") and env["AD_DOT_SELF"] == "1" and "--direction" in argv


def test_the_receipt_pins_the_carrier_bundle_param_grad_ran(tmp_path):
    root = tmp_path / "assets"
    row = {"asset_args": {"--kernels": "kernels"}, "asset_env": {"AD_SCHEMES": "bundle"}}
    assets = {"kernels": root / "k.json", "bundle": root / "latest/bundle/gen-B9",
              "bundle_params": root / "rows/hrrrv5-any/bundle-params", "refuse": root / "rows/hrrrv5-any/refuse.txt"}
    table = d.load_table()
    ctx = SimpleNamespace(root=root, row=row, assets=assets, table=table, door="param-grad")
    assert d.Context.runner_assets(ctx)["bundle_params"] == "rows/hrrrv5-any/bundle-params"
    ctx.door = "tangent"
    assert "bundle_params" not in d.Context.runner_assets(ctx)


def _bundle(tmp_path):
    b = tmp_path / "bundle-params"
    (b / "p4").mkdir(parents=True)
    (b / "p4" / "mynnsfc.tl.sm_120.cubin").write_bytes(b"carrier")
    (b / "schemes.json").write_text("{}")
    lines = [f"{hashlib.sha256((b / rel).read_bytes()).hexdigest()}  {rel}"
             for rel in ("p4/mynnsfc.tl.sm_120.cubin", "schemes.json")]
    (b / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return b, hashlib.sha256((b / "SHA256SUMS").read_bytes()).hexdigest()


def test_the_carrier_bundle_is_checked_against_its_own_sums_and_the_rows_pin(tmp_path):
    b, sha = _bundle(tmp_path)
    got = d.param_bundle_check(b, {"name": "ctl5", "sha256sums_sha256": sha})
    assert got == {"path": str(b), "name": "ctl5", "sha256sums_sha256": sha, "files": 2, "pinned": True}
    with pytest.raises(d.AdDoorError, match="pins bundle ctl5"):
        d.param_bundle_check(b, {"name": "ctl5", "sha256sums_sha256": "0" * 64})
    (b / "p4" / "mynnsfc.tl.sm_120.cubin").write_bytes(b"plain")   # one plain image left in place
    with pytest.raises(d.AdDoorError, match="1 of 2 files are missing or differ"):
        d.param_bundle_check(b, {"name": "ctl5", "sha256sums_sha256": sha})
    (b / "SHA256SUMS").unlink()
    assert d.param_bundle_check(b)["sha256sums_sha256"] is None      # the pinned x1.2562 row's bundle has no sums
    with pytest.raises(d.AdDoorError, match="no SHA256SUMS"):
        d.param_bundle_check(b, {"name": "ctl5", "sha256sums_sha256": sha})


class _FakeContext:
    last = None

    def __init__(self, args, door):
        self.table, self.door, self.out, self.steps = d.load_table(), door, Path(args.out), 3
        self.row = {"param_names": FIVE, "dt_s": 720.0}
        self.row_name, self.assets, self.fmap, self.geometry = "hrrrv5-x1.40962", {}, None, None
        self.windows, self.strict, self.result = [], "unset", None
        _FakeContext.last = self

    def stage(self, *a):
        pass

    def window(self, extra, **kw):
        self.windows.append(list(extra))
        return {"param_gradients": [{"name": n, "family": params.BY_NAME[n].family, "dJdlnp": 1.5, "dJdp": 2.5,
                                     "nonzero_columns": 7} for n in FIVE],
                "objective_values": [{"J": 301.25}], "param_dot": None}

    def receipt(self, result, passed=None, strict=None):
        self.result, self.strict = result, strict
        return {"passed": passed}

    def _refused_names(self):
        return ["solar", "rlw_taut"]


def _args(tmp_path, **over):
    return SimpleNamespace(**{"metric": "theta_m", "levels": None, "box": None, "circle": None, "params": None,
                              "check": False, "no_self_dot": False, "out": tmp_path, **over})


def test_param_grad_carries_the_strict_check_the_value_of_j_and_the_refused_kinds(tmp_path, monkeypatch, capsys):
    strict_calls = []
    monkeypatch.setattr(d, "Context", _FakeContext)
    monkeypatch.setattr(d, "metric_objective", lambda *a: {"name": "mean theta_m", "units": "K", "terms": []})
    monkeypatch.setattr(d, "strict_check", lambda ctx, no_self_dot=False, strict_from=None: strict_calls.append(
        no_self_dot) or {"how": "own", "pass": True, "value": 8.1e-09, "bar": 1e-07, "allocations": 107,
                         "left_out": None})
    assert d.run_param_grad(_args(tmp_path)) == 0
    ctx = _FakeContext.last
    assert strict_calls == [False] and ctx.strict["how"] == "own"            # the receipt carries the strict check
    table = json.loads((tmp_path / "params.json").read_text())               # the row's settings, shipped values
    assert [p["name"] for p in table["params"]] == FIVE and not params.moved(table)
    assert "--params-unchecked" not in ctx.windows[0]
    assert ctx.result["J"] == 301.25 and ctx.result["moved"] == []
    out = capsys.readouterr().out
    assert "whole-state dot  8.100e-09" in out and "refused by name (identity both ways): 2 launch kinds" in out
    assert "J = 301.25 K" in out


def test_a_moved_setting_runs_unchecked_and_is_never_called_valid(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(d, "Context", _FakeContext)
    monkeypatch.setattr(d, "metric_objective", lambda *a: {"name": "mean theta_m", "units": "K", "terms": []})
    monkeypatch.setattr(d, "strict_check", lambda *a, **k: pytest.fail("no recording to check a moved point against"))
    zil = params.BY_NAME["mynnsfc_zil_kczil"].default
    moved = tmp_path / "moved.json"
    params.write_carrier_document(moved, FIVE, {"mynnsfc_zil_kczil": zil * 1.03})
    assert d.run_param_grad(_args(tmp_path, params=moved)) == 0
    ctx = _FakeContext.last
    assert ctx.windows[0][-2:] == ["--params-unchecked", "1"]
    assert ctx.strict["how"] == "not run" and ctx.strict["pass"] is False
    assert ctx.result["moved"] == ["mynnsfc_zil_kczil"]
    assert "moved point" in capsys.readouterr().out


def test_param_grad_takes_the_no_self_dot_flag():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="cmd"))
    assert parser.parse_args(["param-grad", "--out", "o"]).no_self_dot is False
    assert parser.parse_args(["param-grad", "--out", "o", "--no-self-dot"]).no_self_dot is True
