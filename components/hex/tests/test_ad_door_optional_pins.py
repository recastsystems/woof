"""The storm row's optional download files: the drag image is fetched by a door on that row (its gradient line names
it); the next kernel set is pinned but fetched only when a row's kernels name its folder; a 120 km door fetches neither.
THE BREAKAGE: an optional image pinned in its own file was never fetched (the door asked only for mesh and plan
paths), so every gradient-line window ran without it."""
import copy
import types

from woof.hex import ad_doors as d
from woof.hex.ad import assets

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
PLAIN = types.SimpleNamespace(plan=None, init=None)


def _names():
    pins = assets.load_pins()
    aux = [n for n, r in pins["assets"].items() if r["kind"] == "ad-aux" and "ugwp" in n]
    kset = [n for n, r in pins["assets"].items() if r["kind"] == "ad-kset" and "dyn6" in n]
    assert len(aux) == 1 and len(kset) == 1
    return pins, aux[0], kset[0]


def test_the_storm_door_fetches_the_drag_image_and_its_kernel_set():
    pins, aux, kset = _names()
    row = d.load_table()["rows"][STORM]
    want = assets.wanted(pins, d.example_reads(row, PLAIN))
    fast = row["gradient_line"]["boundary_layer"]["routes"]["tape-free"]["assets_pinned_in"]
    assert aux in want and fast in want and kset not in want   # the default route runs its own kernel set
    exact = assets.wanted(pins, d.example_reads(row, types.SimpleNamespace(plan=None, init=None, boundary_layer="taped")))
    # 0.6.0: the exact route keeps the single-arithmetic 3 km set its reference bytes were read on; the default
    # route runs the same set with three double-arithmetic dynamics derivative images
    taped = row["gradient_line"]["boundary_layer"]["routes"]["taped"]["assets_pinned_in"]
    assert taped != fast and fast.replace("-ddr-", "-") == taped
    assert aux in exact and taped in exact and fast not in exact and kset not in exact
    other = assets.wanted(pins, d.example_reads(row, PLAIN, False))   # the row's other windows keep the six-kernel set
    assert kset in other and fast not in other
    path = row["gradient_line"]["optional_assets"]["AD_P4_UGWP_DIRECT"]["path"]
    assert path in pins["assets"][aux]["holds"]


def test_a_120_km_door_fetches_neither():
    pins, aux, kset = _names()
    for name in ("hrrrv5-x1.40962", "hrrrv5-full-x1.40962"):
        want = assets.wanted(pins, d.example_reads(d.load_table()["rows"][name], PLAIN))
        assert aux not in want and kset not in want and not any("q3.96.38278" in n for n in want)
        # the full-physics row fetches exactly one image file of its own: its floored boundary layer (F10-full)
        own = ["ad-aux-bl-q1floor-sm120-v1.tar.gz"] if name == "hrrrv5-full-x1.40962" else []
        assert sorted(n for n in want if pins["assets"][n]["kind"] == "ad-aux") == own


def test_the_storm_row_runs_the_stacks_kernel_set_from_its_own_download():
    pins, aux, kset = _names()
    row = d.load_table()["rows"][STORM]
    ks = row["gradient_line"]["kernel_set"]
    assert ks["pinned_in"] == kset and ks["runs_in_this_build"] is True and ks["set"] == "storm-bs5-lean2-astra2-dyn6"
    assert (row["assets"]["kernels"], row["assets"]["cubins"]) == (ks["kernels"], ks["cubins"])
    assert all(p.startswith(pins["assets"][kset]["holds"][0]) for p in (ks["kernels"], ks["cubins"]))
