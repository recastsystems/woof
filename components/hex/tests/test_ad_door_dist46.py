"""dist46 door facts (processors only): an answer's valid time from the plan's kept facts (no whole parse of plan.json
a target), the host memory a short window states with its forward kept, the word a stopped program differed by, the
exact option's cost as measured through the command, and the refined cloud derivative as the 3 km storm row's
default (the global rows keep their own cloud images)."""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_plan_facts as pf
from woof.hex import ad_session
from woof.hex.ad import assets

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
R120 = "hrrrv5-x1.40962"


def _row(name=STORM):
    return d.load_table()["rows"][name]


def _ns(**kw):
    base = dict(plan=None, init=None, boundary_layer=None, cloud_derivative=None, checked_forward=False, row=STORM,
                mesh=None, keep_forward=None, objectives=None)
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------- item 8: a short window with its forward kept
def test_a_short_window_with_its_forward_kept_states_what_its_later_gradient_holds():
    """THE BREAKAGE: a later 12-step command that attached and took the kept forward held 182,879 MiB of host memory
    against 168,733 MiB stated and admitted (the build before's card job, leg att2b); the 59-step legs were inside."""
    row = _row()
    avail = 498_611   # what that box read
    n12 = d.store_need(row, 12, avail, True, check_runs=False, keep_forward=True)
    assert n12["host_mib"] >= 182_879 and "counted twice" in n12["how"]
    n59 = d.store_need(row, 59, avail, True, check_runs=False, keep_forward=True)
    assert n59["host_mib"] == 249_661 and "counted twice" not in n59["how"]   # 233,026 MiB held: inside
    off = d.store_need(row, 12, avail, True, check_runs=False, keep_forward=False)
    assert off["keep_forward_mib"] == 0 and off["host_mib"] < n12["host_mib"]


# ------------------------------------------------------------------------- item 9: the word that differed
def test_a_stopped_program_is_named_by_the_word_that_differed():
    """THE BREAKAGE: a command with another cloud word printed 'was started with other words (words)' (legs cloud
    and cw of the build before's card job)."""
    then = {"version": "0.6.0", "row": STORM, "steps": 59, "cards": ["0"],
            "words": {"cloud_derivative": "shipped", "mesh": None}, "environ": [["AD_A", "1"]]}
    now = {**then, "words": {"cloud_derivative": "gated", "mesh": None}, "environ": [["AD_A", "2"]]}
    assert ad_session.words_differ(then, now) == ["environment AD_A", "cloud_derivative"]
    assert ad_session.words_differ(then, {**then, "steps": 12}) == ["steps"]
    assert ad_session.words_differ(then, then) == []


def test_the_rows_default_route_typed_out_is_the_same_program_as_no_word():
    base = dict(mesh=None, suite=None, oro=None, dt=None, plan=None, init=None, case=None, runner=None, assets=None,
                gradient_mesh=None, gradient_plan=None, checked_forward=False, boundary_layer=None, first_gradient=None,
                rank_plans=None, keep_forward=None)
    words = lambda **kw: ad_session.command_words(SimpleNamespace(
        args=argparse.Namespace(**{**base, "cloud_derivative": None, **kw}), row_name=STORM, row=_row(), steps=59,
        cards=["0"]))
    assert words() == words(cloud_derivative="number") == words(boundary_layer="tape-free")
    assert words(cloud_derivative="gated") != words()
    assert words(cloud_derivative="shipped") != words()
    assert ad_session.words_differ(words(cloud_derivative="shipped"), words()) == ["cloud_derivative"]


# ------------------------------------------------------------------------ item 10: the exact option's cost
def test_the_exact_options_cost_is_the_one_measured_through_the_command():
    row = _row()
    bl = row["gradient_line"]["boundary_layer"]["routes"]
    for text in (bl["taped"]["what"], bl["tape-free"]["saves"]):
        assert "289.6" in text and "260.8" in text and "239.5" in text and "180.5" in text   # the newest card run
        assert "53 s" not in text and "67 s" not in text
    readme = ((Path(d.__file__).resolve().parents[2] / "components" / "hex") / "README.md").read_text()
    assert ("Against the default route of the build before this one it cost 50 s more on a first gradient and 80 s\n"
            "more on a further target") in readme   # the build before's card job
    assert "53 s more for a\nlater gradient" not in readme and "the script reads 172.6 s" not in readme


# ------------------------------------------------------------- item 3: the refined cloud derivative, 3 km only
def test_the_refined_cloud_derivative_is_the_storm_rows_default_with_the_repaired_images():
    row = _row()
    cd = row["gradient_line"]["cloud_derivative"]
    assert cd["default"] == "number"            # from 0.6.0 the number-following form; gated is a named word
    name, route, default = d.cloud_derivative_route(row, "gated")
    assert (name, default) == ("gated", False)
    fast, taken = d.route_row(row, _ns(cloud_derivative="gated"))
    assert fast["assets"]["tempo_block_cubin"].endswith("cloud-gated/c2chain-0xc6-rd3600n100.cubin")
    assert fast["assets"]["tempo_tlf_cubin"].endswith("cloud-gated/tlfd-0xc6-rd3600n100.cubin")
    assert fast["assets"]["tempo_cubin"].endswith("cloud-gated/split2-0xc6-rd3600n100.cubin")
    shas = {k: v["sha256"][:16] for k, v in route["assets"].items()}
    assert shas == {"tempo_block_cubin": "c96d5e60cee0a0ba", "tempo_tlf_cubin": "44227e04d040dcae",
                    "tempo_cubin": "6d22a804ad93c38a"}   # the repaired tangent and taped images
    shipped, taken = d.route_row(row, _ns(cloud_derivative="shipped"))
    assert "cloud-gated" not in json.dumps(shipped["assets"])
    exact, taken = d.route_row(row, _ns(boundary_layer="taped", cloud_derivative="gated"))   # the exact route too
    assert exact["assets"]["tempo_cubin"].endswith("split2-0xc6-rd3600n100.cubin")


def test_the_global_rows_keep_their_own_cloud_images():
    table = d.load_table()
    for name, row in table["rows"].items():
        if name == STORM:
            continue
        assert not (row.get("gradient_line") or {}).get("cloud_derivative"), name
        assert "rd3600n100" not in json.dumps(row), name


def test_the_new_cloud_file_is_pinned_and_fetched_by_default_and_the_old_one_is_gone():
    pins = json.loads(assets.PINS.read_text())
    assert "ad-aux-cloud-gated-sm120-v1.tar.gz" not in pins["assets"]
    v2 = pins["assets"]["ad-aux-cloud-gated-sm120-v2.tar.gz"]
    assert all("rd3600n100" in h or h.endswith(".txt") for h in v2["holds"])
    row = _row()
    assert "ad-aux-cloud-gated-sm120-v2.tar.gz" in assets.wanted(
        pins, d.example_reads(row, _ns(cloud_derivative="gated")))
    assert "ad-aux-cloud-gated-sm120-v2.tar.gz" not in assets.wanted(
        pins, d.example_reads(row, _ns(cloud_derivative="shipped")))
    assert pins["release_assets"]["tag"] == "ad-assets-v0.6.0"


def test_the_near_ground_wind_caution_names_the_new_default():
    note = _row()["accuracy_notes"][0]
    assert ("With the gated cloud derivative (the number-following default has not been run along this nudge) it "
            "read 1.37") in note
    assert "with --cloud-derivative shipped its terms cancelled to about zero" in note


# ------------------------------------------------------- item 11: the first gradient does not pay for the keeping
def test_the_kept_forward_words_page_lock_its_checkpoints_on_the_helper():
    """THE BREAKAGE: the build before's first gradient of a fresh program read 247.3 s with the forward kept against
    236.3 s with --keep-forward off on the same card: the sweep page-locked the kept checkpoints' buffers inline."""
    row = _row()
    plan = d.keep_forward_plan(row, _ns(objectives="two.json"), True)
    assert plan["on"] and plan["env"] == {"AD_KEEP_FORWARD": "2", "AD_KF_PREFILL": "1"}
    assert d.keep_forward_plan(row, _ns(objectives="two.json", keep_forward="off"), True).get("env") is None
    window_rs = (Path(d.__file__).resolve().parents[2] / "components" / "hex") / "rust/hexad/src/runner/window.rs"
    if not window_rs.is_file():
        pytest.skip("hex's Rust runner source (rust/hexad) is not folded into WOOF: ad-binder ships prebuilt in "
                    "the hex helpers archive, and hex's own battery checks this source on its tree")
    src = window_rs.read_text()
    assert 'std::env::var("AD_KF_PREFILL").as_deref() == Ok("1")' in src and "bs_prefill(&ex, plan_live_bound, w.bs_ram_budget, kf_kept)" in src
