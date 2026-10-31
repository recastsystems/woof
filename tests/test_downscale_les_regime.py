"""A sub-250 m child is told which regime it is in, before it runs.

WHAT BREAKAGE THIS PINS (gate law).  ``_derive_child_run_config`` copies
the parent's physics VERBATIM, so a child derived at any spacing at all
carries the parent's ``bl_pbl_physics`` and ``km_opt`` down with it, and
a child that names no ladder of its own runs on the parent's.  At 83 m
that is a different regime from the mesoscale run the parent's
configuration was written for, and the only place the tree said so was
inside ``--child-levels``' own help -- read by people who already knew to
look for the flag.  The walked outcome: a 798 x 798 child at 83.33 m on
its parent's 49 levels grew w_max from 1.1 m/s to 23.0 m/s over 45
minutes of forecast and went non-finite, 49 minutes of a reader's time
after the plan resolved, with nothing said beforehand.

This is a STATEMENT and not a refusal: the shipped nested LES child is
itself a 250 m child on its grandparent's ladder, and nothing here
changes what any run does.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json

import pytest

from woof.cli import main as cli_main
from woof.config import RunConfig
from woof.downscale import _derive_child_run_config, _render_child_toml
from woof.offline_child import LES_CHILD_SPACING_M, les_child_regime

from test_downscale_cli import _add_parent_surface
from test_offline_child import _history


def _cfg(*, dx=83.33333333333333, nz=49, bl_pbl_physics=1, km_opt=4):
    """``diff_opt`` is deliberately absent: it is not a key this tree
    takes.  ``woof.namelist_import`` maps WRF's ``diff_opt = 2`` onto the
    native mixing form on the way in, so ``km_opt`` is the whole of the
    closure selection and a way out that named ``diff_opt`` would be a way
    out that does not exist."""

    return RunConfig(
        nx=96, ny=96, nz=nz, dx=dx, dy=dx, ztop=20000.0,
        dt=0.4166666666666667, run_seconds=600.0, moist=True,
        mp_physics=10, bl_pbl_physics=bl_pbl_physics, km_opt=km_opt)


# --- the rule ----------------------------------------------------------


def test_the_reported_shape_earns_the_statement():
    """83 m columns, the parent's ladder, a 1-D scheme and no 3-D closure."""

    regime = les_child_regime(
        _cfg(), inherits_parent_levels=True, parent_levels=49)
    assert regime is not None
    statement = regime["statement"]
    assert "83.3333 m spacing" in statement
    assert "250 m this tree calls coarse LES" in statement
    assert "docs/public/LES.md" in statement
    assert "inherits the parent's 49-level vertical ladder" in statement
    assert ("runs a 1-D boundary-layer scheme (bl_pbl_physics = 1) with no "
            "3-D closure (km_opt = 4)") in statement
    # THIS failure shape, named.
    assert ("tends to grow vertical velocity check after check until the "
            "field goes non-finite") in statement


def test_the_statement_names_every_way_out_it_has():
    statement = les_child_regime(
        _cfg(), inherits_parent_levels=True, parent_levels=49)["statement"]
    assert "--child-levels N,STRETCH" in statement
    assert "km_opt = 3 (3-D Smagorinsky)" in statement
    assert "km_opt = 2 (prognostic TKE)" in statement
    assert "bl_pbl_physics = 0 in the --child-config TOML" in statement
    assert "--child-surface-from" in statement
    # A way out this tree does not have is not a way out.
    assert "diff_opt" not in statement


def test_a_one_kilometre_child_is_told_nothing():
    """NEGATIVE CONTROL: the statement must stay out of a mesoscale run."""

    assert les_child_regime(
        _cfg(dx=1000.0), inherits_parent_levels=True,
        parent_levels=49) is None


def test_the_threshold_is_inclusive_and_is_the_trees_own_number():
    assert LES_CHILD_SPACING_M == 250.0
    assert les_child_regime(
        _cfg(dx=250.0), inherits_parent_levels=True) is not None
    assert les_child_regime(
        _cfg(dx=250.01), inherits_parent_levels=True) is None


@pytest.mark.parametrize("km_opt", [2, 3])
def test_a_child_configured_for_the_regime_is_told_nothing(km_opt):
    """Its own ladder AND a 3-D closure: nothing left to say.

    BOTH closures this tree admits with the scheme off, because the rule
    that decides silence reads
    ``LES_CHILD_THREE_DIMENSIONAL_CLOSURES`` and a suite that probes one
    of its two members proves nothing about the other.
    """

    assert les_child_regime(
        _cfg(nz=96, km_opt=km_opt, bl_pbl_physics=0),
        inherits_parent_levels=False) is None


@pytest.mark.parametrize("km_opt", [1, 4])
def test_a_child_with_nothing_mixing_it_vertically_earns_the_statement(
        km_opt):
    """THE ARM THE RULE LEFT OUT.

    An 83 m child on its OWN ladder with the boundary-layer scheme off
    and a two-dimensional closure heard nothing, although this same
    function's ``why`` text names exactly that pair as the case with no
    vertical mixing of heat or moisture by any route: ``km_opt`` 1 and 4
    compute no vertical exchange pair, and ``bl_pbl_physics = 0`` is the
    scheme that would otherwise do it.  The rule of record and the rule
    disagreed, and the rule was the narrower one.
    """

    regime = les_child_regime(
        _cfg(nz=96, km_opt=km_opt, bl_pbl_physics=0),
        inherits_parent_levels=False)
    assert regime is not None
    statement = regime["statement"]
    assert ("mixes heat and moisture vertically by no route at all "
            f"(km_opt = {km_opt} computes no vertical exchange pair and "
            "bl_pbl_physics = 0)") in statement
    # The way out is the closure, and the ladder half stays out of it.
    assert "km_opt = 3 (3-D Smagorinsky)" in statement
    assert "--child-levels" not in statement
    assert regime["no_vertical_mixing_of_heat_or_moisture"] is True
    assert regime["pbl_without_three_dimensional_closure"] is False
    # And the shape named is the one that is true of THIS child: it runs
    # no boundary-layer scheme, so it is not warned about one.
    assert "boundary-layer-scheme child" not in statement
    assert ("Nothing carries heat or moisture between this child's levels "
            "except the motion it resolves") in statement


def test_the_ladder_alone_earns_the_statement():
    """A 3-D closure on the parent's ladder is still worth a sentence:
    ``docs/public/LES.md`` says the vertical grid, not the 250 m spacing,
    is the binding constraint on the shipped nested child."""

    regime = les_child_regime(
        _cfg(km_opt=3, bl_pbl_physics=0),
        inherits_parent_levels=True, parent_levels=49)
    assert regime is not None
    assert "--child-levels N,STRETCH" in regime["statement"]
    assert "km_opt = 3 (3-D Smagorinsky)" not in regime["statement"]
    # The shape named is the ladder's own, measured: this child mixes
    # vertically and runs no 1-D scheme, so neither of the other two
    # sentences is true of it.
    assert ("leaves more of its turbulence to the subgrid model than a "
            "resolved column does, 12.7 percent against 7.9"
            ) in regime["statement"]
    assert "boundary-layer-scheme child" not in regime["statement"]


def test_the_closure_alone_earns_the_statement():
    """Its own ladder, but the parent's boundary-layer scheme carried down."""

    regime = les_child_regime(_cfg(nz=96), inherits_parent_levels=False)
    assert regime is not None
    assert "bl_pbl_physics = 1" in regime["statement"]
    assert "--child-levels" not in regime["statement"]


def test_the_statement_carries_the_numbers_as_fields_too():
    """A controller shows this without parsing prose."""

    regime = les_child_regime(
        _cfg(), inherits_parent_levels=True, parent_levels=49)
    assert regime["spacing_m"] == pytest.approx(83.33333333333333)
    assert regime["threshold_m"] == 250.0
    assert regime["nz"] == 49 and regime["parent_nz"] == 49
    assert regime["km_opt"] == 4 and regime["bl_pbl_physics"] == 1
    assert regime["inherits_parent_levels"] is True
    assert regime["pbl_without_three_dimensional_closure"] is True
    assert regime["no_vertical_mixing_of_heat_or_moisture"] is False


# --- the door ----------------------------------------------------------


#: A parent that declares no eta ladder, so the child derived from it
#: declares none either and is built on the parent's, which is the first
#: of the three readings ``child_inherits_parent_levels`` takes.
_PARENT = {
    "nx": 20, "ny": 18, "nz": 4, "dx": 1000.0, "dy": 1000.0,
    "ztop": 9000.0, "dt": 5.0, "run_seconds": 21600.0,
    "output_interval_s": 3600.0, "hybrid_opt": 2, "etac": 0.2,
    "hypsometric_opt": 2, "moist": True, "mp_physics": 8,
    "specified": False, "nested": True, "terrain_opt": 1, "map_proj": 1,
    "grid_id": 3, "time_step_sound": 4, "spec_bdy_width": 5,
    "spec_zone": 1, "relax_zone": 4,
    "sf_surface_physics": 2, "sf_sfclay_physics": 91,
    "bl_pbl_physics": 1, "num_soil_layers": 4,
}


def _door(tmp_path, *, ratio, child_nx=24, extra=()):
    start = datetime(1970, 1, 2, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1970-01-02_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    merged = _derive_child_run_config(
        _PARENT, parent={"dx": 1000.0, "dy": 1000.0}, ratio=ratio,
        child_nx=child_nx, child_ny=child_nx, run_seconds=600.0,
        output_interval_s=300.0)
    child_toml = tmp_path / "child.toml"
    child_toml.write_text(_render_child_toml(merged), encoding="utf-8")
    return [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_toml), "--ratio", str(ratio),
        "--i-parent-start", "4", "--j-parent-start", "4",
        # The fixture archive's coordinates are no real projection and no
        # WPS_GEOG tree is staged: a child of it runs on its parent's
        # terrain, the route this door test is about.
        "--parent-terrain",
        "--accept-parent-cadence",
        "--out", str(tmp_path / "child-run"), "--dry-run", *extra]


def _plan(captured_out):
    return json.loads(captured_out[captured_out.index("{"):])


def test_the_door_states_the_regime_for_the_reported_shape(tmp_path, capsys):
    """THE REPRODUCTION: a child of the report's shape, at the door."""

    assert cli_main(_door(tmp_path, ratio=12)) == 0
    captured = capsys.readouterr()
    assert "83.3333 m spacing" in captured.err
    assert "docs/public/LES.md" in captured.err
    assert "until the field goes non-finite" in captured.err
    assert "--child-levels N,STRETCH" in captured.err
    # A statement, not a refusal: the plan is still printed and the
    # command still exits 0.
    plan = _plan(captured.out)
    assert plan["les_regime"]["spacing_m"] == pytest.approx(83.33333333333333)
    assert plan["les_regime"]["inherits_parent_levels"] is True
    assert plan["les_regime"]["bl_pbl_physics"] == 1
    assert any("83.3333 m spacing" in record.get("action", "")
               for record in plan["warnings"])


def test_the_door_says_nothing_to_a_one_kilometre_child(tmp_path, capsys):
    """NEGATIVE CONTROL, at the door this time."""

    assert cli_main(_door(tmp_path, ratio=1, child_nx=12)) == 0
    captured = capsys.readouterr()
    assert "coarse LES" not in captured.err
    assert _plan(captured.out)["les_regime"] is None


def test_child_levels_takes_the_ladder_half_of_the_statement_away(
        tmp_path, capsys):
    """The way out, taken: the reader who passes --child-levels is no
    longer told to."""

    assert cli_main(_door(tmp_path, ratio=12,
                          extra=["--child-levels", "32,1.8"])) == 0
    captured = capsys.readouterr()
    assert "inherits the parent's" not in captured.err
    # The closure half is still true, so it is still said.
    assert "bl_pbl_physics = 1" in captured.err


def test_a_blown_up_child_reaches_the_reader_as_a_sentence(
        tmp_path, capsys, monkeypatch):
    """THE REAL DOOR, all the way to the CLI's own refusal boundary.

    Only the integration is substituted: the archive is validated, the
    child config is resolved and priced, and the run is dispatched
    exactly as a real one is.  What is pinned here is the SHAPE of the
    exit, which is the half of the 2.7.5 failure a reader met first: a
    ``TypeError`` traceback at exit 1 over a run that had already
    finished 45 minutes of forecast.
    """

    from woof import offline_child_run

    capsule = offline_child_run.describe_nonfinite_child(
        step=48, total_steps=144, model_seconds=120.0, run_seconds=600.0,
        cadence_seconds=60.0,
        trend=[{"step": 24, "model_seconds": 60.0, "w_max": 11.5,
                "cfl": 0.19},
               {"step": 48, "model_seconds": 120.0, "w_max": float("nan"),
                "cfl": None}],
        survey={"surveyed": ["W"], "fields": [{
            "field": "W", "carrier": "w", "shape": [5, 24, 24],
            "size": 2880, "count": 1,
            "bounding_box": {"k": [2, 2], "j": [11, 11], "i": [9, 9]},
            "edges": [], "cell": {"k": 2, "j": 11, "i": 9}}]})

    def blow_up(_namespace):
        raise offline_child_run.OfflineChildNonFinite(capsule)

    monkeypatch.setattr(offline_child_run, "run", blow_up)
    args = [flag for flag in _door(tmp_path, ratio=12) if flag != "--dry-run"]
    # The substituted integration raises before producing frames. Ask for
    # no pictures so the host's renderer cannot mask the child-failure
    # sentence this test exercises. Renderer admission has its own tests.
    args += ["--render-products", "none"]
    # A real run, so the child's surface physics needs a real source: the
    # parent frames carry the inventory the derivation reads.
    for index in range(3):
        _add_parent_surface(
            tmp_path / f"wrfout_d03_1970-01-02_{12 + index:02d}_00_00",
            ny=18, nx=20, lu_water_column=8)
    assert cli_main(args) == 2
    captured = capsys.readouterr()
    assert "woof downscale: The child blew up:" in captured.err
    assert "found W non-finite at one cell, (k=2, j=11, i=9)" in captured.err
    assert "Traceback" not in captured.err
