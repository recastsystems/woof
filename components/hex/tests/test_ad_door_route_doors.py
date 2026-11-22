# SPDX-License-Identifier: Apache-2.0
"""adjoint-test (and tangent) run the boundary-layer route a plain sensitivity runs, on every row that names routes.

THE BREAKAGE (dist48, 2026-11-15): the README's first command, adjoint-test on the 12 km example, checked the row's own
held linear model and ended FAIL (the gradient's descent direction 4.42e-02 and near-ground moisture 1.18e-01 off the
real model), while the default route a sensitivity runs read PASS on the same cut only through a harness: the release's
own test checked a linear model no default command computes, and said nothing of the one users get.
"""
import argparse
import hashlib
import json
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d

Q12 = "hrrrv5-q12.96.2890.n40.60w91.30"
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
# the boundary layer species outputs the tape-free route holds (runner AD_P3_HOLD_OUT): their cotangents are opposite-signed
# neighbours single precision cannot carry through the dynamics (precfloor, 15 Nov 2026)
HOLD = "rncblten,rniblten,rnifablten,rnwfablten,rqcblten,rqiblten,rqsblten"


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args([*argv, "--out", "o"])


def _table():
    return d.load_table()


def test_the_12km_row_runs_the_3km_rows_default_route_by_default():
    row, storm = _table()["rows"][Q12], _table()["rows"][STORM]
    assert not row.get("gradient_line")                      # its doors run plain windows
    bl, sbl = d.row_boundary_layer(row), d.row_boundary_layer(storm)
    assert bl["default"] == sbl["default"] == "tape-free"
    mine, theirs = bl["routes"]["tape-free"], sbl["routes"]["tape-free"]
    # the species hold is a 12 km row word only (15 Nov 2026): at 3 km it adds scatter (NOAA deck,
    # hold-only seed 1 +3.39 roundings against -1.07 released), so the storm row's route runs without it
    assert mine["window_env"] == {"AD_P3_DIRECT": "1", "AD_P3_HOLD_OUT": HOLD}
    assert theirs["window_env"] == {"AD_P3_DIRECT": "1"}
    assert mine["images"] == theirs["images"]                # the same joined backward and capture forward
    p3 = {r: i for r, i in theirs["scheme_bundle"]["images"].items() if r.startswith("p3/")}
    assert mine["scheme_bundle"]["images"] == p3             # the same form's tangent, taped reverse and count
    assert mine["scheme_bundle"]["over"] == "bundle" and not mine.get("assets")   # the row's own kernel set
    assert not bl["routes"]["taped"]["window_env"] and not bl["routes"]["taped"].get("scheme_bundle")


@pytest.mark.parametrize("door", ["tangent", "adjoint-test", "sensitivity"])
def test_every_route_door_runs_the_route_and_param_grad_and_checked_forward_do_not(door):
    table = _table()
    for name in (Q12, STORM):
        row = table["rows"][name]
        args = _parse([door, "--row", name])
        assert d.routes_on(row, args, door)
        assert not d.routes_on(row, args, door, split={"ranks": 2})
    assert not d.routes_on(table["rows"][STORM], SimpleNamespace(checked_forward=True), "sensitivity")
    assert not d.routes_on(table["rows"][STORM], SimpleNamespace(), "param-grad")
    assert not d.routes_on(table["rows"]["hrrrv5-x1.40962"], SimpleNamespace(), door)   # a row without routes


def test_a_route_the_row_does_not_offer_is_refused_before_any_download():
    table = _table()
    for door in ("adjoint-test", "tangent"):
        with pytest.raises(d.AdDoorError, match=r"offers guarded, tape-free, taped\. Nothing was downloaded"):
            d.refuse_speed_words(_parse([door, "--row", Q12, "--boundary-layer", "floored"]), table)
        d.refuse_speed_words(_parse([door, "--row", Q12, "--boundary-layer", "guarded"]), table)
        with pytest.raises(d.AdDoorError, match="has no gradient line and names no boundary-layer routes"):
            d.refuse_speed_words(_parse([door, "--row", "hrrrv5-x1.40962", "--boundary-layer", "tape-free"]), table)
        d.refuse_speed_words(_parse([door, "--row", Q12, "--boundary-layer", "taped"]), table)
        d.refuse_speed_words(_parse([door, "--row", STORM, "--boundary-layer", "guarded"]), table)


def test_the_12km_row_offers_the_storm_rows_guarded_boundary_layer_as_a_named_option():
    """0.6.1: a measure near the ground under rain reads about 2 per cent short on the 12 km row's default route (the
    plume's own speed, area and condensation and the mixing length are held); the guarded route, the 3 km storm row's
    named option, differentiates them. It is the same three images from the same pack, laid over this row's bundle,
    with this row's species hold, and it is not the default."""
    table = _table()
    bl12 = d.row_boundary_layer(table["rows"][Q12]); bl3 = d.row_boundary_layer(table["rows"][STORM])
    assert bl12["default"] == "tape-free" and sorted(bl12["routes"]) == ["guarded", "tape-free", "taped"]
    g12 = bl12["routes"]["guarded"]; g3 = bl3["routes"]["guarded"]
    images = g12["scheme_bundle"]["images"]
    assert sorted(images) == ["p3/rel.0.cubin", "p3/rel.1.cubin", "p3/rel.2.cubin"]
    for name, item in images.items():
        assert item == g3["scheme_bundle"]["images"][name] and item["path"].startswith("latest/aux/bl-guarded/")
    assert g12["scheme_bundle"]["over"] == "bundle" and g12["scheme_bundle"]["pinned_in"] == "ad-aux-bl-guarded-sm120-v1.tar.gz"
    assert g12["window_env"] == {"AD_P3_HOLD_OUT": bl12["routes"]["tape-free"]["window_env"]["AD_P3_HOLD_OUT"]}
    assert "images" not in g12                      # taped: no tape-free backward image
    name, route, is_default = d.boundary_layer_route(table["rows"][Q12], "guarded")
    assert (name, is_default) == ("guarded", False) and route is g12 or route == g12


def _ctx(tmp_path, asked):
    row = _table()["rows"][Q12]
    root = tmp_path / "assets"
    route = d.row_boundary_layer(row)["routes"]["tape-free"]
    row = json.loads(json.dumps(row))
    for var, item in route["images"].items():
        p = root / item["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(var.encode())
        row["boundary_layer"]["routes"]["tape-free"]["images"][var]["sha256"] = hashlib.sha256(var.encode()).hexdigest()
    return SimpleNamespace(row=row, row_name=Q12, root=root, args=SimpleNamespace(boundary_layer=asked), arch="sm_120",
                           card="0", route_assets=[], plan_how="the row's example plan", steps=15)


def test_the_route_words_every_window_takes_and_the_line_that_says_which_linear_model(tmp_path, capsys):
    ctx = _ctx(tmp_path, None)
    env, block = d.route_window_words(ctx, "adjoint-test")
    images = d.row_boundary_layer(ctx.row)["routes"]["tape-free"]["images"]
    assert env == {"AD_P3_DIRECT": "1", "AD_P3_HOLD_OUT": HOLD, **{v: str(ctx.root / i["path"]) for v, i in images.items()}}
    assert block["route"] == "tape-free" and block["is_default"] and block["window_env"] == env
    out = capsys.readouterr().out
    assert "woof hex adjoint-test: boundary layer tape-free (this row's default):" in out
    assert "--boundary-layer taped runs the exact route" in out
    # the exact route: no route words, and the line names it
    env, block = d.route_window_words(_ctx(tmp_path, "taped"), "adjoint-test")
    assert env == {} and block["route"] == "taped" and not block["is_default"]
    assert "boundary layer taped (--boundary-layer taped): none: the exact route" in capsys.readouterr().out
    # an image that is not the pinned build is refused by name, never left out
    bad = _ctx(tmp_path, None)
    var, item = sorted(d.row_boundary_layer(bad.row)["routes"]["tape-free"]["images"].items())[0]
    (bad.root / item["path"]).write_bytes(b"other")
    with pytest.raises(d.AdDoorError, match="is not the build the table pins"):
        d.route_window_words(bad, "adjoint-test")


def test_the_route_words_reach_every_window_and_name_the_answer_checks_key(tmp_path):
    """Context.window_line lays the route's words under the window's own (the answer check's AD_DOT_SELF stays)."""
    seen = {}

    def window_command(table, row, assets, door, runner, steps, out, extra, env_overrides, key, extra_env, spec, rep):
        seen.update(extra_env)
        return ["runner"], dict(extra_env)

    ctx = SimpleNamespace(table={}, row={}, assets={}, door="adjoint-test", runner="r", steps=15, out=tmp_path,
                          plan=tmp_path, route_env={"AD_P3_DIRECT": "1", "AD_DOT_SELF": "0"})
    real = d.window_command
    try:
        d.window_command = window_command
        _, env = d.Context.window_line(ctx, [], sub="selfdot", extra_env={"AD_DOT_SELF": "1"})
    finally:
        d.window_command = real
    assert env["AD_P3_DIRECT"] == "1" and env["AD_DOT_SELF"] == "1"


def test_the_readme_says_what_the_first_command_checks_and_what_passes():
    """The 12 km section names the default route, its needs from the table, the PASS and what the PASS does not cover."""
    from pathlib import Path
    readme = ((Path(d.__file__).resolve().parents[2] / "components" / "hex") / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("**The 12 km example: first"):readme.index("**The 3 km storm example.**")]
    row = _table()["rows"][Q12]
    need = d.row_boundary_layer(row)["routes"]["tape-free"]["need_free_mib"]
    assert f"{need:,} MiB free" in section and f"{row['need_host_mib']:,} MiB of host memory" in section
    for words in ("default boundary-layer route", "**PASS** (exit status 0)", "UNRESOLVED, which a PASS does not count",
                  "Near-ground moisture is not inside the\nbar", "`--boundary-layer taped` tests the exact route"):
        assert words in section, words
    assert "does not yet run that route" not in readme and "runs the exact route, above)" not in readme
