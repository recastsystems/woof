# SPDX-License-Identifier: Apache-2.0
"""The full-physics rows, and the 60 km class of the fast suite, run the mountain-drag closure held at the trajectory, as the
default row does (bundles phys-B9-gfrz and genA-gfrz). What it prevents: with that closure differentiated (bundles phys-B9
and gen-A) the 120 km tangent grows without bound over high ground after about an hour, so every gradient those rows gave
over a longer window was wrong while their checks passed. No card."""
import json
from pathlib import Path

from woof.hex import ad_doors as d
from woof.hex.ad import assets

LIVE_DRAG = {"latest/bundle/phys-B9", "latest/bundle/gen-B9", "latest/bundle/gen-A"}


def _bundles(table, suite_name):
    suite = table["suites"][suite_name]
    out = {suite["template_row"]: table["rows"][suite["template_row"]]["assets"].get("bundle")}
    for cls in suite.get("window_classes", []):
        b = (cls.get("assets") or {}).get("bundle")
        if b is not None:
            out[f"{suite_name} class {cls.get('from_recipe')}"] = b
    return out


def test_every_full_physics_row_holds_the_drag_closure():
    table = d.load_table()
    for name in ("hrrrv5-any-full", "hrrrv5-full-x1.40962"):
        row = table["rows"][name]
        assert row["assets"]["bundle"] == "latest/bundle/phys-B9-gfrz", name
        assert "mountain-drag closure (GSL drag) is held" in row["what"] and "phys-B9-gfrz" in row["what"], name
    for where, bundle in _bundles(table, "hrrrv5-full").items():
        assert bundle == "latest/bundle/phys-B9-gfrz", (where, bundle)


def test_the_fast_suite_never_swaps_in_a_bundle_that_differentiates_the_drag_closure_on_a_global_mesh():
    table = d.load_table()
    for cls in table["suites"]["hrrrv5"]["window_classes"]:
        b = (cls.get("assets") or {}).get("bundle")
        if b is None or cls.get("mesh_kind") == "limited-area":   # at 3 km the closure's mesoscale part is tapered off
            continue
        assert b not in LIVE_DRAG, (cls.get("from_recipe"), b)
    sixty = [c for c in table["suites"]["hrrrv5"]["window_classes"] if c.get("max_cells") == 200000][0]
    assert sixty["assets"]["bundle"] == "latest/bundle/genA-gfrz" and "mountain-drag closure held" in sixty["what"]


def test_the_row_asset_carries_the_held_bundles_and_not_the_retired_one():
    pins = json.loads(assets.PINS.read_text())
    rows = [k for k, v in pins["assets"].items() if v.get("kind") == "ad-row" and "hrrrv5-any" in k]
    assert rows == ["ad-row-hrrrv5-any-v6.tar.gz"]
    what = pins["assets"][rows[0]]["what"]
    assert "phys-B9-gfrz" in what and "genA-gfrz" in what and "no longer in the asset" in what


def test_the_exact_suite_says_it_differentiates_the_drag_closure_and_states_the_hour_limit():
    """What it prevents: rows made with --suite hrrrv5-exact (120 and 60 km classes, bundle gen-A) differentiate the closure,
    and their words and the docs said nothing of it, so a user read their gradients over windows where they are wrong."""
    table = d.load_table()
    tmpl = table["rows"]["hrrrv5-any-exact"]
    assert tmpl["assets"]["bundle"] in LIVE_DRAG
    assert "mountain-drag closure (GSL drag) is differentiated" in tmpl["what"] and "about an hour" in tmpl["what"]
    assert "about an hour" in table["suites"]["hrrrv5-exact"]["what"]
    for cls in table["suites"]["hrrrv5-exact"]["window_classes"]:
        if cls.get("what"):   # a class with its own words replaces the template's in the made row
            assert "mountain-drag closure is differentiated" in cls["what"], cls.get("from_recipe")
    docs = (Path(d.__file__).resolve().parents[2] / "components" / "hex") / "docs" / "differentiable" / "README.md"
    if docs.exists():   # a source tree; an installed wheel carries no docs
        assert "--suite hrrrv5-exact" in docs.read_text(encoding="utf-8")


def test_every_limited_area_class_says_why_its_drag_closure_is_live_or_held():
    """What it prevents: the fast 3 km class runs the closure live (gen-B9) and the full 3 km class holds it
    (phys-B9-gfrz) with no word on why, which reads as one of them being wrong."""
    table = d.load_table()
    for suite in ("hrrrv5", "hrrrv5-full", "hrrrv5-exact"):
        for cls in table["suites"][suite]["window_classes"]:
            if cls.get("mesh_kind") != "limited-area":
                continue
            assert "ls_taper under 0.01" in cls["what"] and "3.6 km" in cls["what"], (suite, cls.get("from_recipe"))
