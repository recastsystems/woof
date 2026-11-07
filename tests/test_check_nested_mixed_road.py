"""``woof check`` prices the road the run door actually takes on a tree.

THE DEFECT, reproduced on the published 2.5.8 and 2.6.0 wheels and again
on this line's pre-merge tip.  On a NESTED tree with ``[tiles]``, ``check``
priced only the fully-resident tree and refused it, while the run door's
own per-domain walk (:func:`woof.core.streaming.steppers_for_tree`) took a
MIXED road -- child streamed, parent resident -- that fits the same card
and completes.  Measured on the tree below at ``--budget-gib 13``: the
report said "the forecast is the memory-binding phase at 17.76 GiB peak
envelope; that EXCEEDS the 15.87 GiB budget by 1.89 GiB", failed
``alloc_estimate_le_wddm_budget`` and exited 1 -- beside an advisory that
admitted in the same breath that "a refusal or a fit below describes the
resident tree, not the mixed-road one the run will take".

A report whose own advisory says it priced a run nobody asked for, and
whose exit code refuses that run anyway, teaches the reader that streaming
has no point.  The same walk now prices the report: 15.04 GiB against the
15.87 GiB budget, exit 0, with the per-domain plan printed.

THE CONTROL IS THE OTHER HALF.  A tree that fits NO road still refuses --
the fix is a different question being asked, not a gate being loosened.
"""
from __future__ import annotations

import json
import textwrap

import pytest

from woof import cli


GIB = 1024 ** 3

#: The declared card where the resident tree does not fit and the mixed
#: road does.  ``--budget-gib`` names an allocation budget; the envelope
#: budget the verdict compares is that plus the reserve less the
#: other-process margin, which is 15.87 GiB here.  Resident 17.76 GiB
#: refuses against it; the mixed road's 15.04 GiB radiation peak fits.
_FITS_MIXED_ROAD_GIB = "13"

#: One GiB down, where automatic selection must seek a smaller fitting road.
_SMALLER_AUTO_BUDGET_GIB = "12"


#: The tree-wide table the default fixture carries.
_TREE_AUTO = """[tiles]
mode = "auto"
"""

#: The PINNED shape, reached the way a user reaches it: no tree-wide
#: table at all and a per-domain ``tiles`` on the CHILD alone.  A
#: tree-wide ``mode = "on"`` is refused by the walk -- it would stream
#: both ends of the d01->d02 edge -- so this is the only road to a
#: pinned streamed domain inside a nest.
_CHILD_PINNED = (
    'tiles = { mode = "on", tile_nx = 256, tile_ny = 256, nbuffers = 2 }')


def _nested_auto_tiles(tmp_path, name="tree", *, tree_tiles=_TREE_AUTO,
                       child_tiles=""):
    """A two-domain tree whose child cannot be held resident on that card.

    ``mode = "auto"`` over the tree by default: the shape a user writes,
    and the one the walk prices.

    A pinned tiling consults no planner and no card.  That is a statement
    about the PLANNER, not about the PRICE: the once-per-process floor
    and the radiation reservation are both reads off the rung's footprint
    (``autoplan.footprint_for``), which takes a config and no machine, so
    a pinned tree is priced by the same walk and its claims stand in the
    forecast term exactly as ``auto``'s do.  ``child_tiles`` reaches that
    shape.

    Written as a TOML and loaded through the product's own loader, for the
    reason ``tests/test_streamed_admission.py`` states: the finding is
    about the config a user actually types.
    """
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent("""\
        [experiment]
        name = "synth"
        start_time = 2024-05-03T12:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [fetch]
        source = "gfs"
        cycle = "2024-05-03T12"
        hours = 1
        cadence = 1

        [shared]
        nz = 49
        ztop = 20000.0
        moist = true
        moist_cq = true
        mp_physics = 10
        ra_lw_physics = 4
        ra_sw_physics = 4
        sf_sfclay_physics = 91
        sf_surface_physics = 2
        bl_pbl_physics = 1
        cu_physics = 1
        nwp_diagnostics = 1

        %(TREE_TILES)s
        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 300
        ny = 300
        time_step = 20
        dx = 9000.0
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 12
        j_parent_start = 12
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 600
        ny = 600
        history_interval_s = 3600.0
        %(CHILD_TILES)s
        """) % {"TREE_TILES": tree_tiles, "CHILD_TILES": child_tiles},
        encoding="utf-8")
    return path


def _check(capsys, config, *flags):
    code = cli.main(["check", str(config), "--explain", *flags])
    captured = capsys.readouterr()
    return code, captured.out


def _plan_rows(out):
    """The per-domain rows of the printed plan, and nothing else.

    Taken as the BLOCK under the heading rather than by matching ``d0``
    anywhere: the resident itemization further down the report opens its
    lines the same way, and a looser filter reads those as plan rows.
    """
    rows = []
    inside = False
    for line in out.splitlines():
        if "MIXED-ROAD PLAN" in line:
            inside = True
            continue
        if inside:
            stripped = line.strip()
            if not stripped.startswith(("d0", "tree budget", "REFUSED:")):
                break
            if stripped.startswith("d0"):
                rows.append(stripped)
    return rows


def test_a_tree_that_only_fits_the_mixed_road_is_admitted(tmp_path, capsys):
    """THE REGRESSION.  Exit 0, on the road the run door takes.

    Asserted as a RELATION, not a pinned figure: the mixed road has to be
    genuinely cheaper than the resident tree it replaced, and the verdict
    has to be about the mixed road.  A gate that had merely stopped
    refusing everything would pass an exit-code-only assertion and still
    price the wrong run.
    """
    config = _nested_auto_tiles(tmp_path)
    code, out = _check(capsys, config, "--budget-gib", _FITS_MIXED_ROAD_GIB)

    assert code == 0, out
    # The verdict names the road it priced, and the resident tree it did
    # not: the two figures beside each other are what say streaming
    # bought something.
    assert "mixed-road forecast" in out
    assert "with the whole tree resident" in out
    # The advisory no longer disowns the numbers under it.
    assert "not the mixed-road one the run will take" not in out


def test_the_plan_names_each_domains_road_and_claim(tmp_path, capsys):
    """The report SHOWS the walk, per domain, default-on and unflagged.

    The exit code alone is not the fix.  A user whose tree was refused for
    a reason the report would not print had no way to see which domain
    claimed what, which is where the lever actually is.
    """
    config = _nested_auto_tiles(tmp_path)
    _code, out = _check(capsys, config, "--budget-gib", _FITS_MIXED_ROAD_GIB)

    assert "MIXED-ROAD PLAN" in out
    plan = _plan_rows(out)
    # Parent resident, child streamed -- the shape the coupler supports
    # and the one the resident-only pricing could not describe.
    assert any(line.startswith("d01 resident:") for line in plan), plan
    assert any(line.startswith("d02 streams (") for line in plan), plan
    # Every row carries its claim, and the streamed row its tiling.
    assert all("claim" in line for line in plan), plan
    assert any("buffer(s) of tile" in line for line in plan), plan
    # ...and the budget the claims were priced against, so a reader can do
    # the arithmetic the walk did.
    assert "tree budget" in out
    # The gate leg explains itself in terms that are TRUE of a mixed road.
    # "the resident domain is never allocated" is the single-domain
    # reason, and it is false about a tree whose parent the plan above
    # just showed the reader sitting resident on the card.
    assert "ALLOC GATE, MIXED ROAD" in out
    assert "the resident domain is never allocated" not in out


def test_a_tree_below_its_shared_process_floor_still_refuses(tmp_path, capsys):
    """A genuine immutable shortage still refuses before any allocation."""
    config = _nested_auto_tiles(tmp_path)
    code, out = _check(capsys, config, "--json", "--free-gib", "2")
    payload = json.loads(out)
    assert code == 1
    assert payload["observed_peak_envelope_exceeds_budget"] is True
    assert "shared process/radiation floor" in payload["tree_road"]["refusal"]
    assert payload["tree_road"]["replaces_forecast_term"] is False


def test_smaller_auto_allowance_can_choose_smaller_tiles_and_still_fit(tmp_path, capsys):
    """Auto reprices its selected road; a fitting alternative must not refuse."""
    config = _nested_auto_tiles(tmp_path)
    reports = []
    for budget in (_FITS_MIXED_ROAD_GIB, _SMALLER_AUTO_BUDGET_GIB):
        code, out = _check(capsys, config, "--json", "--budget-gib", budget)
        value = json.loads(out)
        assert code == 0
        assert value["observed_peak_envelope_exceeds_budget"] is False
        road = value["tree_road"]
        assert road["refusal"] is None and road["replaces_forecast_term"]
        assert value["peak_envelope_bytes"] == road["peak_vram_bytes"]
        assert road["peak_vram_bytes"] == max(
            road["vram_hold_bytes"] + road["radiation_transient_bytes"],
            road["configured_mixed_envelope_bytes"])
        reports.append(value)
    assert reports[1]["peak_envelope_bytes"] <= reports[0]["peak_envelope_bytes"]


def test_the_json_report_publishes_the_walk_as_data(tmp_path, capsys):
    """A machine reader gets the roads as fields, not as prose.

    ``woof check --json`` is how another door asks this question, and
    parsing the plan back out of the advisory sentence is not an
    interface.
    """
    config = _nested_auto_tiles(tmp_path)
    _code, out = _check(capsys, config, "--json",
                        "--budget-gib", _FITS_MIXED_ROAD_GIB)
    payload = json.loads(out)

    road = payload["tree_road"]
    assert road is not None
    assert road["replaces_forecast_term"] is True
    assert road["priced"] is True and road["streams_any"] is True
    assert road["refusal"] is None
    roads = {int(row["grid_id"]): row["road"] for row in road["rows"]}
    assert roads == {1: "resident", 2: "streamed"}
    # The tile belongs to the DOMAIN, never to the tree: a mixed road has
    # no single tiling, and publishing one at the top level is how a
    # reader takes the child's tile for the whole run's.
    assert payload["streamed"]["road"] == "mixed (nested tree)"
    assert "tile_nx" not in payload["streamed"]
    assert payload["tree_road"]["rows"][1]["tile"]["tile_nx"] > 0
    # The figure every gate weighed is the walk's own peak.
    assert payload["peak_envelope_bytes"] == road["peak_vram_bytes"]
    # ...and it is genuinely cheaper than the tree it replaced.
    assert (road["peak_vram_bytes"]
            < payload["streamed"]["resident_forecast_envelope_bytes"])


def test_a_single_domain_config_is_not_given_a_tree_road(tmp_path, capsys):
    """THE NEGATIVE CONTROL.  The question does not arise, so it is not asked.

    A one-domain config has no mixed road, and the single-domain streamed
    envelope -- with its one real tiling -- is what its report must keep
    carrying.  This pins that the tree walk did not become a second answer
    for configs that already had one.
    """
    path = tmp_path / "single.toml"
    text = _nested_auto_tiles(tmp_path, name="src").read_text(encoding="utf-8")
    path.write_text(text.split("[[domain]]")[0]
                    + "[[domain]]" + text.split("[[domain]]")[1],
                    encoding="utf-8")

    _code, out = _check(capsys, path, "--json", "--budget-gib", "13")
    payload = json.loads(out)

    assert payload["tree_road"] is None
    assert "MIXED-ROAD PLAN" not in out
    if payload["streamed_forecast"]:
        # It streams as a single domain, so it has exactly one tiling and
        # says so under the name every existing reader uses.
        assert payload["streamed"]["road"] == "streamed (single domain)"
        assert payload["streamed"]["tile_nx"] > 0


# --- the PINNED child: the same walk, the same standing -------------------
#
# THE SECOND DEFECT, reproduced on this line's tip against the real artifact
# (test host 2, gpuwm 2.6.5, a 4-domain ERA5 tree whose 1 km d04 cannot sit
# resident on a 32 GiB 5090).  Varying only d04's ``tiles`` line:
#
#   { mode = "auto" }                      -> d04 streams, claim 16.24 GiB,
#                                             pinned host store 15.94 GiB,
#                                             BINDING PHASE 31.13 GiB, exit 0
#   { mode = "on", tile 256x256, nb = 2 }  -> d04 streams, claim 6.95 GiB,
#                                             no host-store clause at all,
#                                             BINDING PHASE 62.36 GiB = the
#                                             RESIDENT envelope, refused
#
# The walk planned a BETTER road under the pin -- 6.95 GiB against 16.24 --
# and the report then priced the forecast as if the domain were resident and
# refused it, three lines below printing the road that fits.


def _pinned_child_tree(tmp_path, name="pinned"):
    """The same tree, with the CHILD's tiling pinned instead of planned."""
    return _nested_auto_tiles(tmp_path, name=name, tree_tiles="",
                              child_tiles=_CHILD_PINNED)


def test_a_pinned_child_is_priced_by_the_same_walk(tmp_path, capsys):
    """THE REGRESSION.  A pinned road stands in the forecast term.

    ``priced`` gated the walk's whole claim ledger on whether some domain
    happened to ask the PLANNER a question, and a pinned tiling asks none.
    So a tree whose only streamed domain pinned its tiling was priced at
    the resident envelope and refused -- while the road the run door takes
    was printed, fitting, in the same report.

    Every term of the forecast the walk quotes is a read off the rung's
    footprint and the tiling: the once-per-process floor, the marginal
    claims, the corridors and the radiation reservation all come from
    ``autoplan.footprint_for``, which takes a config and no machine.  The
    card was only ever needed to answer "does it FIT", which preflight
    asks separately against its own budget.
    """
    config = _pinned_child_tree(tmp_path)
    code, out = _check(capsys, config, "--budget-gib", _FITS_MIXED_ROAD_GIB)

    assert code == 0, out
    # The verdict is about the road the run door takes, not the resident
    # tree it would never build.
    assert "mixed-road forecast" in out
    assert "RESIDENT FORECAST PEAK ENVELOPE (replaced by the streamed term "
    assert "every enabled domain PINNED its tiling" not in out
    plan = _plan_rows(out)
    assert any(line.startswith("d01 resident:") for line in plan), plan
    assert any(line.startswith("d02 streams (") for line in plan), plan


def test_a_pinned_streamed_domain_prices_its_pinned_host_store(tmp_path,
                                                               capsys):
    """The row says where the forecast actually lives, on BOTH roads.

    A streamed domain's whole store and its arena are page-locked on the
    box, and that is the binding constraint at every capacity limit
    measured.  ``decide``'s pinned short-circuit returned no ``detail`` at
    all, so ``host_claim_bytes`` was absent, the row dropped its "pinned
    host store" clause, and the walk's host ledger counted the domain as
    free -- which is how two pinned streamed domains would each be priced
    against the whole box, the exact ``cudaHostAlloc`` failure the host
    ledger exists to prevent.

    INDEPENDENT of the claim-ledger defect above, and the pair below is
    what proves it: this is asserted on a tree that streams and is priced.
    """
    config = _pinned_child_tree(tmp_path)
    _code, out = _check(capsys, config, "--json",
                        "--budget-gib", _FITS_MIXED_ROAD_GIB)
    payload = json.loads(out)
    rows = {int(row["grid_id"]): row for row in payload["tree_road"]["rows"]}

    # The streamed domain pins a store; the resident one pins none.
    assert rows[2]["host_claim_bytes"] > 0, rows[2]
    assert rows[1]["host_claim_bytes"] == 0, rows[1]
    # ...and the tree's ledger is the sum of them, not zero.
    assert payload["tree_road"]["host_bytes"] == rows[2]["host_claim_bytes"]

    _code, prose = _check(capsys, config, "--budget-gib",
                          _FITS_MIXED_ROAD_GIB)
    child = [line for line in _plan_rows(prose)
             if line.startswith("d02 streams (")]
    assert child and "pinned host store" in child[0], child


def test_a_resident_sibling_is_priced_resident_under_either_road(tmp_path,
                                                                 capsys):
    """d01 is resident on both roads, so its price cannot move between them.

    The third cause, and the reason the flag alone was not the fix.  With
    no planner consulted the tree budget was zero, and
    ``_decision_claim_bytes`` handed a DECIDED ``mode = "off"`` domain to
    ``_minimum_claim_bytes``, whose "does the resident price fit what is
    left" test every domain fails against a zero budget.  Each resident
    sibling then priced at the one-buffer STREAMED floor -- 0.05 GiB
    against a true 2.37 GiB on this tree, and 0.07 against 0.86/1.30/4.85
    on the four-domain tree this was measured on.  Flipping the flag
    without this would have traded a false refusal for a false admission,
    which under the gate law is the worse of the two.

    An OFF domain cannot stream.  Its floor IS its resident price, which
    is what ``_decision_claim_bytes`` already said it was.
    """
    auto = _nested_auto_tiles(tmp_path, name="auto")
    pinned = _pinned_child_tree(tmp_path)

    _c, auto_out = _check(capsys, auto, "--json",
                          "--budget-gib", _FITS_MIXED_ROAD_GIB)
    _c, pinned_out = _check(capsys, pinned, "--json",
                            "--budget-gib", _FITS_MIXED_ROAD_GIB)
    auto_rows = {int(r["grid_id"]): r
                 for r in json.loads(auto_out)["tree_road"]["rows"]}
    pinned_rows = {int(r["grid_id"]): r
                   for r in json.loads(pinned_out)["tree_road"]["rows"]}

    assert auto_rows[1]["road"] == pinned_rows[1]["road"] == "resident"
    # One byte of slack and no more: the two branches reach the same
    # number by ``resident - overhead`` and ``marginal_resident``, which
    # are the same arithmetic rounded to int in a different order.
    assert abs(auto_rows[1]["claim_bytes"]
               - pinned_rows[1]["claim_bytes"]) <= 1, (
        auto_rows[1], pinned_rows[1])
