"""The 3 km storm row's default (tape-free) boundary-layer route runs the closure-live form (processors only): its five
images are one pinned file's, the route lays the form's three generated images over the row's scheme bundle (the
tangent windows and the columns the tape-free backward hands to the tape run them), the exact route keeps the held
form, and the texts the command prints say what the default form differentiates and what was not measured.

THE BREAKAGE: the release's own adjoint-test, on four directions, failed on the 12 km cut of the storm box with the
shipped linear model (the gradient's own descent direction 4.42e-02 off the real model, near-ground moisture 1.18e-01
off), because that linear model held the boundary layer's turbulence closure and the plume's heat and moisture."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
AUX = "ad-aux-bl-closure-sm120-v1.tar.gz"
GONE = "ad-aux-bl-tape-free-sm120-v1.tar.gz"
# built on processors from this tree by ad/phys/mynn/fast-closure/build-kit.sh (composed source 33ba6f36)
DIRECT = {"AD_P3_DIRECT_IMAGE": ("latest/aux/bl-closure/direct120.cubin", "d7b12c3802a32446f307a07a16c58d3c36bcd52ee72f198269016974201d4574"),
          "AD_P3_DIRECT_FWD": ("latest/aux/bl-closure/cap120.cubin", "34bc34bf6e05aca1797efefb95539dac5e1fa7b86f4d0ed718219198a5f72042")}
P3 = {"p3/rel.0.cubin": "40c3988a9ebe518feece79cb6b1a0f7d77380ade7262b04dbfd64211ea00c3e4", "p3/rel.1.cubin": "058f8a52e8efd3263fcc743e1e6ff0677651a2e4780715bd9bbef2b30f2d161d", "p3/rel.2.cubin": "9f7f018c3901d3033647183442ab6d98ca7002431ffe665957d4a0fe927afa88"}
AUX_SHA256, AUX_BYTES = "8c316312f0649dd57b3168912ea3f4e0e8f9deab0c812fce2beb368e1ed70716", 10003374


def _row():
    return d.load_table()["rows"][STORM]


def _routes():
    return _row()["gradient_line"]["boundary_layer"]["routes"]


def _ns(**kw):
    base = dict(plan=None, init=None, boundary_layer=None, cloud_derivative=None, checked_forward=False, row=STORM,
                mesh=None, keep_forward=None, objectives=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_default_route_runs_the_five_images_of_one_pinned_file():
    row = _row()
    assert row["gradient_line"]["boundary_layer"]["default"] == "tape-free"
    route = _routes()["tape-free"]
    assert {k: (v["path"], v["sha256"]) for k, v in route["images"].items()} == DIRECT
    spec = route["scheme_bundle"]
    assert {rel: item["sha256"] for rel, item in spec["images"].items() if rel.startswith("p3/")} == P3
    assert all(item["path"].startswith("latest/aux/bl-closure/") for rel, item in spec["images"].items() if rel.startswith("p3/"))
    # the radiation step's images are the exact route's, as before
    assert ({rel: item for rel, item in spec["images"].items() if rel.startswith("p7/")}
            == _routes()["taped"]["scheme_bundle"]["images"])
    pins = json.loads(assets.PINS.read_text())
    assert GONE not in pins["assets"]
    pin = pins["assets"][AUX]
    assert (pin["sha256"], pin["bytes"], pin["kind"]) == (AUX_SHA256, AUX_BYTES, "ad-aux")
    for item in list(route["images"].values()) + [i for r, i in spec["images"].items() if r.startswith("p3/")]:
        assert item["path"] in pin["holds"] and item["arch"] == "sm_120"
    assert any(h.endswith("NOTICE.txt") for h in pin["holds"]) and any("LICENSE" in h for h in pin["holds"])
    assert route["pinned_in"] == AUX and AUX in spec["pinned_in"]
    assert AUX in assets.wanted(pins, d.example_reads(row, _ns()))
    for other in ("taped", "guarded"):                       # the other routes do not fetch it
        assert AUX not in assets.wanted(pins, d.example_reads(row, _ns(boundary_layer=other))), other


def test_the_exact_route_keeps_the_held_form():
    taped = _routes()["taped"]
    images = (taped.get("scheme_bundle") or {}).get("images") or {}
    assert not [rel for rel in images if rel.startswith("p3/")]       # the row's own bundle: the held form
    assert not taped.get("images")


def test_the_route_lays_the_forms_images_over_the_rows_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "cache"))
    route = json.loads(json.dumps(_routes()["tape-free"]))
    spec = route["scheme_bundle"]
    root = tmp_path / "assets"
    base = root / "latest/bundle/gen-B9"
    files = {"schemes.json": b"{}", "p8-sm_120.cubin": b"p8", "p3/gen-info.json": b"{}"}
    files.update({rel: b"held-" + rel.encode() for rel in spec["images"]})
    for rel, b in files.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(b)
    (base / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(b).hexdigest()}  {rel}\n" for rel, b in sorted(files.items())))
    for rel, item in spec["images"].items():
        p = root / item["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"pinned-" + rel.encode())
        item["sha256"] = hashlib.sha256(b"pinned-" + rel.encode()).hexdigest()
    ctx = SimpleNamespace(root=root, assets={"bundle": base}, row_name=STORM, row=_row())
    out = d.route_scheme_bundle(ctx, "--boundary-layer", "tape-free", route)
    assert out != base and out.name.startswith("gen-B9-rad-step-bl-closure-")
    for rel in P3:
        assert (out / rel).read_bytes() == b"pinned-" + rel.encode()
        assert (base / rel).read_bytes() == b"held-" + rel.encode()
    (root / spec["images"]["p3/rel.1.cubin"]["path"]).write_bytes(b"other")
    with pytest.raises(d.AdDoorError, match="is not the build the table pins for --boundary-layer tape-free"):
        d.route_scheme_bundle(ctx, "--boundary-layer", "tape-free", route)


def test_the_printed_words_say_what_the_form_differentiates_and_print_no_figure_of_another_form():
    """The fast-against-exact figures were measured with the held form's images; they are not printed for this one."""
    route = _routes()["tape-free"]
    assert "error_measured" not in route
    plan = f"the row's example plan ({_row()['example_plan']})"
    for ctx in (SimpleNamespace(plan_how=plan, steps=59), SimpleNamespace(plan_how="given (--plan)", steps=12)):
        words, measured = d.route_error_words(ctx, route)
        assert not measured and words == route["error"]
    for w in ("turbulence closure", "heat", "moisture", "height the column returns", "1e-5 s-2", "the exact route holds",
              "Through adjoint-test on this example (59 steps, one 96 GB card) this route read PASS",
              "Its difference from the exact route was not measured on this build"):
        assert w in route["error"], w
    # confirmation 4 (16 Nov 2026): the sentence said this route "was not measured through this command on this
    # example" while the row's known limits print its adjoint-test PASS three lines below it
    assert "not measured through this command on this example" not in route["error"]
    form = route["form"]
    assert form["weak_shear_s2"] == 1e-05 and "c6d08709" in form["derivative_of"] and "v6.1.1" in form["derivative_of"]
    shown = form["shown"]["adjoint_test_12km_15_steps"]
    assert shown["this_form"]["verdict"] == "pass" and shown["held_form_taped"]["verdict"] == "fail"
    assert max(form["shown"]["consistency_12km_15_steps"]["this_form"].values()) < 1e-07
    note = _row()["accuracy_notes"][0]
    assert "1.37" in note and "--boundary-layer guarded" in note and "It has run through this command on a 96 GB card" in note
