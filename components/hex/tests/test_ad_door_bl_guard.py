"""The weak-shear guard as a named word of --boundary-layer on the 3 km storm row (processors only): the word is a
word of the command, it lays the three guarded boundary-layer images over the row's scheme bundle, its file is pinned
and fetched only when asked for, it is off by default, and its text says what was shown and what was not."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
WORD = "guarded"
AUX = "ad-aux-bl-guarded-sm120-v1.tar.gz"
AUX_SHA256 = "e26ddf62b4c867501ae93ec4325500595c1669a07df783c90de0d13620c3e667"
AUX_BYTES = 8259882
# the kit bl-weakshear-849608c8: tangent, taped reverse, tape count (ad/phys/mynn/weakshear/)
IMAGES = {"p3/rel.0.cubin": "a90522cea9fbf27f68019246eb0ebbf68a8a22a90f5c78297a7160d0700206bc",
          "p3/rel.1.cubin": "849608c867e0ba9fd71ecc4bd1534f0901522ffa3adac80c8441ea8724dfb709",
          "p3/rel.2.cubin": "4c0bd4d62adb66c76757da8fd2ebc609e1a5e7be8d35eefafc80684dd8dbc742"}


def _row(name=STORM):
    return d.load_table()["rows"][name]


def _routes():
    return _row()["gradient_line"]["boundary_layer"]["routes"]


def _ns(**kw):
    base = dict(plan=None, init=None, boundary_layer=None, cloud_derivative=None, checked_forward=False, row=STORM,
                mesh=None, keep_forward=None, objectives=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _parse(*words):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(["sensitivity", "--out", "x", *words])


def test_the_word_is_a_word_of_the_command():
    """THE BREAKAGE: before this word the only near-ground wind answer a user could ask for was the held one (1.37
    times the real change on the measured test); `--boundary-layer guarded` was refused as an unknown word."""
    assert _parse("--boundary-layer", WORD).boundary_layer == WORD
    with pytest.raises(SystemExit):
        _parse("--boundary-layer", "weak-shear")          # an unknown word is still refused
    table = d.load_table()
    d.refuse_speed_words(_ns(boundary_layer=WORD), table)   # offered on the storm row
    for name, row in table["rows"].items():                # and on no other row
        if name != STORM and row.get("gradient_line"):
            assert WORD not in row["gradient_line"]["boundary_layer"]["routes"], name


def test_it_is_off_by_default_and_the_other_routes_keep_their_boundary_layer():
    row = _row()
    assert row["gradient_line"]["boundary_layer"]["default"] == "tape-free"
    assert d.boundary_layer_route(row, None)[0] == "tape-free"
    for name in ("taped", "tape-free"):                      # neither lays the guarded images over the bundle
        images = (_routes()[name].get("scheme_bundle") or {}).get("images") or {}
        assert not {item["sha256"] for rel, item in images.items() if rel.startswith("p3/")} & set(IMAGES.values()), name
    name, route, default = d.boundary_layer_route(row, WORD)
    assert (name, default) == (WORD, False) and route["exact_word"] == "--boundary-layer taped"


def test_the_route_lays_the_three_images_of_the_kit_over_the_rows_bundle(tmp_path, monkeypatch):
    """What the runner is started with: a copy of the row's bundle whose p3/rel.{0,1,2}.cubin are the pinned guarded
    images (and whose radiation images are the exact route's); the row's own bundle is not touched."""
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "cache"))
    route = json.loads(json.dumps(_routes()[WORD]))
    spec = route["scheme_bundle"]
    assert {rel: item["sha256"] for rel, item in spec["images"].items() if rel.startswith("p3/")} == IMAGES
    taped = _routes()["taped"]["scheme_bundle"]["images"]
    assert {rel: item for rel, item in spec["images"].items() if not rel.startswith("p3/")} == taped
    assert route["assets"] == _routes()["taped"]["assets"]          # the exact route's kernel set
    root = tmp_path / "assets"
    base = root / "latest/bundle/gen-B9"
    files = {"schemes.json": b"{}", "p8-sm_120.cubin": b"p8", "p3/gen-info.json": b"{}"}
    files.update({rel: b"held-" + rel.encode() for rel in spec["images"]})
    for rel, b in files.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(b)
    (base / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {rel}\n" for rel, b in sorted(files.items())))
    for rel, item in spec["images"].items():                  # stand-in files, pinned by their own sums
        p = root / item["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"pinned-" + rel.encode())
        item["sha256"] = hashlib.sha256(b"pinned-" + rel.encode()).hexdigest()
    ctx = SimpleNamespace(root=root, assets={"bundle": base}, row_name=STORM, row=_row())
    out = d.route_scheme_bundle(ctx, "--boundary-layer", WORD, route)
    assert out != base and out.name.startswith("gen-B9-rad-step-bl-guarded-")
    for rel in IMAGES:
        assert (out / rel).read_bytes() == b"pinned-" + rel.encode()
        assert (base / rel).read_bytes() == b"held-" + rel.encode()
    assert (out / "p3/gen-info.json").read_bytes() == b"{}" and (out / "p8-sm_120.cubin").read_bytes() == b"p8"
    for line in (out / "SHA256SUMS").read_text().splitlines():        # every line of the copy's listing holds
        want, _, rel = line.partition("  ")
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == want
    (root / spec["images"]["p3/rel.1.cubin"]["path"]).write_bytes(b"other")
    with pytest.raises(d.AdDoorError, match="is not the build the table pins for --boundary-layer guarded"):
        d.route_scheme_bundle(ctx, "--boundary-layer", WORD, route)


def test_the_file_is_pinned_and_fetched_only_when_the_word_is_given():
    pins = json.loads(assets.PINS.read_text())
    pin = pins["assets"][AUX]
    assert (pin["sha256"], pin["bytes"], pin["kind"]) == (AUX_SHA256, AUX_BYTES, "ad-aux")
    held = pin["holds"]
    for rel, item in _routes()[WORD]["scheme_bundle"]["images"].items():
        if rel.startswith("p3/"):
            assert item["path"] in held and item["arch"] == "sm_120"
    assert any(h.endswith("NOTICE.txt") for h in held) and any("LICENSE" in h for h in held)
    row = _row()
    assert AUX in assets.wanted(pins, d.example_reads(row, _ns(boundary_layer=WORD)))
    for other in (None, "taped", "tape-free"):
        assert AUX not in assets.wanted(pins, d.example_reads(row, _ns(boundary_layer=other))), other


def test_the_forward_is_not_kept_and_the_checked_forward_is_refused_with_it():
    table = d.load_table()
    with pytest.raises(d.AdDoorError, match="guarded.*has not been run with"):
        d.refuse_speed_words(_ns(boundary_layer=WORD, keep_forward="on"), table)
    with pytest.raises(d.AdDoorError, match="--checked-forward turns the gradient line off"):
        d.refuse_speed_words(_ns(boundary_layer=WORD, checked_forward=True), table)


def test_the_caution_says_what_was_shown_and_what_was_not():
    """THE BREAKAGE: the pages said the rule holds less than the table's measured figures show. The command's
    line carries the corrections, and the figures it prints are the ones the table keeps."""
    route = _routes()[WORD]
    text = route["error"]
    for words in ("plume and cloud floors", "turbulence energy",            # what else it holds where it binds
                  "43 per cent of the ten lowest interfaces", "about 0.13 under",
                  "1.0031", "1.3674", "three cloud image sets and three seeds",      # shown
                  "through this command on one 96 GB card, 59 steps", "5.34e-08 against the 1e-07 bar",   # the card run
                  "NOT SHOWN: the gradient against the real model at 59 steps",
                  "the storm-direction consistency check", "0.67 to 0.71", "0.35 to 0.44"):
        assert words in text, words
    shown = route["shown"]
    assert shown["real_model_3km_12_steps"]["guarded"] == 1.0031 and shown["real_model_3km_12_steps"]["held"] == 1.3674
    assert max(shown["consistency_12km_15_steps"]["guarded"]) < 1e-07 < min(shown["consistency_12km_15_steps"]["nothing_held_no_rule"])
    assert round(shown["held_share_12km_per_cent"]["ten_lowest_interfaces"]) == 43
    gap = shown["weak_shear_place_cross_wind"]
    assert round(gap["nothing_held_no_rule"] - gap["guarded"], 2) == 0.13
    note = _row()["accuracy_notes"][0]
    assert "--boundary-layer guarded" in note and "The fix is not in this build" not in note
    assert "It has run through this command on a 96 GB card" in note and "5.34e-08" in note
    card = shown["card_3km_59_steps_through_the_command"]
    assert card["answer_check"] < card["answer_check_bar"] and card["largest_word_share_of_squared_dtheta"] < 0.5
    assert route["not_shown"][0].startswith("the gradient against the real model at 59 steps")
    assert "Not read: the storm-direction check" in route["default_flip"]


def test_a_window_past_the_exact_routes_limit_runs_with_a_warning():
    route = _routes()[WORD]
    assert d.route_window_warning(route, 59, 15) is None
    assert "still runs" in d.route_window_warning(route, 240, 15)
