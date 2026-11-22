"""The any-source cyclone door, and the four defects landing it uncovered.

Every test here is red without the change it guards: the follow-track
validator crashed on a follower with no track table, its cadence refusal
named the configuration file instead of the follower, the run-plan suite
still asserted a refusal that had been retired, the boundary termination
only ever looked at a pressure minimum, and one native registry row did
not declare the sea-level field its own product publishes.
"""
from dataclasses import replace
from types import SimpleNamespace
import argparse
import json
import tomllib

import numpy as np
import pytest

from woof import cyclone_setup as tc
from woof import runplan
from woof import cyclone_sources as cs
from woof import domain_wizard as dw
from woof.core.nest_lifecycle import follower_label, validate_follow_tracks
from woof.core.storm_tracking import (FollowConfig, NestFootprint, StormTracker,
                                       extremum_kind)
from woof.core.track_boundary import boundary_reason
from woof.cyclone_seed import seed_cyclone, source_inventory
from woof.starter_template import render_tables

from cyclone_preset_fit import source_licence
from cyclone_preset_fit import center, cycle_for, holds_the_preset_root

POINT = (18., -65.)
CYCLE = "2026090900"


def _raw(**kwargs):
    text, _ = tc.configuration_text(cycle=CYCLE, point=POINT, **kwargs)
    return tomllib.loads(text)


def _load(raw, source="fixture.toml"):
    """Through the shared loader, which is the door that validates."""
    return dw.experiment_from_text(render_tables(raw), source=source)


# -- the follower that declares no track ------------------------------------

def test_a_following_nest_without_a_track_table_still_loads():
    """RED before the guard came back: AttributeError inside the loader.

    Every storm-following configuration that predates per-follower track
    files -- which is all of them -- has a ``[[domain]].follow`` and no
    ``follow.track``, and the new validator asked the track refusals about
    it anyway, reading ``output_level`` off ``None``.
    """

    raw = _raw()
    del raw["domain"][1]["follow"]["track"]
    experiment = _load(raw)
    assert experiment.domains[1].follow is not None
    assert experiment.domains[1].follow.track is None


def test_a_declared_track_is_still_admitted_and_kept():
    assert _load(_raw()).domains[1].follow.track.path == "storm-track.d02.csv"


# -- whose refusal it is ----------------------------------------------------

def _stash_backed_domains():
    """A follower tracking a stash-backed field on an unservable cadence."""
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT)
    child = exp.domains[1]
    tracker = FollowConfig(field="uh", threshold=100., fallback_threshold=35.,
                           search_margin_cells=5, min_shift_cells=1,
                           max_shift_cells=6, cooldown_seconds=0.)
    follow = replace(child.follow, tracker=tracker, track=None,
                     cadence_seconds=float(exp.root.history_interval_s) / 2. + 1.)
    return exp, [exp.root, replace(child, follow=follow)]


def test_the_load_time_cadence_refusal_names_the_follower_not_the_file():
    exp, domains = _stash_backed_domains()
    with pytest.raises(ValueError) as refusal:
        validate_follow_tracks(domains, exp.relocation, "fixture.toml")
    text = str(refusal.value)
    assert "d02 follow of fixture.toml" in text
    assert "whole multiple" in text


def test_the_runner_door_names_the_follower_the_same_way(monkeypatch, tmp_path):
    """One label helper, so neither door invents its own subject."""

    exp, domains = _stash_backed_domains()
    view = replace(exp, domains=tuple(domains))
    root = SimpleNamespace(cfg=view.domains[0])
    child = SimpleNamespace(cfg=view.domains[1], parent=root)
    model = SimpleNamespace(nodes_by_grid_id={1: root, 2: child},
                            node=lambda gid: {1: root, 2: child}[gid])
    monkeypatch.setattr("woof.core.uh_diag.allocate_declared_follower_windows",
                        lambda *a: None)
    from woof import runtime
    with pytest.raises(ValueError) as refusal:
        runtime.build_real_relocation_runners(view, None, model, tmp_path)
    assert follower_label(2) in str(refusal.value)
    assert "whole multiple" in str(refusal.value)


def test_both_doors_name_the_same_table_for_a_per_domain_follower(
        monkeypatch, tmp_path):
    """RED while the runner build kept the whole-run default.

    One follower, one knob, and two refusals that named different
    tables: the load door said the table the knob is in and the runner
    build said ``[relocation]``, which a per-domain configuration does
    not have at all.  Same operands through both doors, so the two
    sentences are compared rather than described.
    """

    from woof import runtime

    exp, domains = _stash_backed_domains()
    with pytest.raises(ValueError) as load:
        validate_follow_tracks(domains, exp.relocation, "fixture.toml")
    view = replace(exp, domains=tuple(domains))
    root = SimpleNamespace(cfg=view.domains[0])
    child = SimpleNamespace(cfg=view.domains[1], parent=root)
    model = SimpleNamespace(nodes_by_grid_id={1: root, 2: child},
                            node=lambda gid: {1: root, 2: child}[gid])
    monkeypatch.setattr("woof.core.uh_diag.allocate_declared_follower_windows",
                        lambda *a: None)
    with pytest.raises(ValueError) as runner:
        runtime.build_real_relocation_runners(view, None, model, tmp_path)
    # The file is the only thing one door knows and the other does not.
    assert str(load.value).replace(" of fixture.toml", "") == str(runner.value)
    assert "[relocation]" not in str(runner.value)
    from woof.core.nest_lifecycle import FOLLOWER_TABLE
    assert FOLLOWER_TABLE in str(load.value)
    assert FOLLOWER_TABLE in str(runner.value)


def test_the_label_helper_appends_the_file_only_where_there_is_one():
    assert follower_label(2) == "d02 follow"
    assert follower_label(2, "c.toml") == "d02 follow of c.toml"


# -- the boundary test is about the tracked field, not about pressure -------

def _peak(ci, cj=20, n=41, amplitude=200.):
    y, x = np.mgrid[:n, :n]
    return amplitude * np.exp(-((x - ci) ** 2 + (y - cj) ** 2) / 40.)


def _uh_tracker():
    cfg = FollowConfig(field="uh", threshold=20., fallback_threshold=5.,
                       search_margin_cells=50, radius_km=180.,
                       cooldown_seconds=0., min_shift_cells=1, max_shift_cells=6)
    fp = NestFootprint(grid_id=2, i_parent_start=12, j_parent_start=12,
                       child_nx=25, child_ny=25, parent_grid_ratio=3,
                       parent_dx_m=12000.)
    return StormTracker(cfg), fp


def test_a_rotation_tracker_terminates_at_the_parent_edge(monkeypatch):
    """RED while the boundary test only looked for a MINIMUM.

    A rotation maximum sitting on the parent edge is the same loss of an
    enclosed centre a pressure minimum there is, and the row written from
    it reports a storm that has stopped moving.
    """

    import woof.core.storm_tracking as st
    tracker, fp = _uh_tracker()
    monkeypatch.setattr(st, "planes_for", lambda state, *a, **kw: [(None, state.uh)])
    interior = tracker.locate(SimpleNamespace(uh=_peak(20)), fp, 0.)
    assert "track_end_reason" not in interior.evidence
    departed = tracker.locate(SimpleNamespace(uh=_peak(44)), fp, 60.)
    assert "parent-domain boundary" in departed.evidence["track_end_reason"]
    assert departed.evidence["extremum_kind"] == "maximum"


def test_the_boundary_test_reads_the_end_the_receipt_reports(monkeypatch):
    """A maximum plane is not judged by where its minimum sits."""

    import woof.core.storm_tracking as st
    tracker, fp = _uh_tracker()
    # The rotation maximum is interior; the plane's minimum is on the edge,
    # as it is for every localized positive signal.  Looking for a minimum
    # here would end a track that is centred and alive.
    monkeypatch.setattr(st, "planes_for", lambda state, *a, **kw: [(None, state.uh)])
    fix = tracker.locate(SimpleNamespace(uh=_peak(20)), fp, 0.)
    assert "track_end_reason" not in fix.evidence
    plane = _peak(20)
    assert boundary_reason(plane, (slice(0, 41), slice(0, 41)), (20, 20),
                           extremum="minimum") is not None
    assert boundary_reason(plane, (slice(0, 41), slice(0, 41)), (20, 20),
                           extremum="maximum") is None


def test_boundary_reason_refuses_to_assume_which_end_is_the_centre():
    with pytest.raises(TypeError):
        boundary_reason(_peak(20), (slice(0, 41), slice(0, 41)), (20, 20))
    with pytest.raises(ValueError, match="which end of the tracked field"):
        boundary_reason(_peak(20), (slice(0, 41), slice(0, 41)), (20, 20),
                        extremum="lowest")


def test_extremum_kind_is_read_from_the_configuration_for_an_attribute():
    pressure = FollowConfig(field="pressure", threshold=1010., level_hpa=0,
                            search_margin_cells=5, min_shift_cells=1,
                            max_shift_cells=6, cooldown_seconds=0.)
    assert extremum_kind(pressure, "pressure") == "minimum"
    rotation = FollowConfig(field="uh", threshold=100., fallback_threshold=35.,
                            search_margin_cells=5, min_shift_cells=1,
                            max_shift_cells=6, cooldown_seconds=0.)
    assert extremum_kind(rotation, "uh") == "maximum"
    # The echo handoff changes the field under the tracker and the echo is
    # a maximum too, so the used field is what is asked.
    assert extremum_kind(rotation, "reflectivity") == "maximum"


# -- the seeding inventory --------------------------------------------------

def test_the_native_row_declares_the_sea_level_field_its_product_publishes():
    """RED while one native row omitted it: the first seeding rung was
    unreachable on a source whose own product carries the field."""

    for source in ("gfs", "era5", "hrrr"):
        assert "mean_sea_level_pressure" in source_inventory(source).fields


def test_a_supplied_sea_level_analysis_seeds_through_the_first_rung():
    lat, lon = np.mgrid[10.:50.:41j, -130.:-70.:41j]
    pressure = np.full(lat.shape, 101000.)
    pressure[20, 20] = 96000.
    seed = seed_cyclone(source="hrrr",
                        fields={"latitude": lat, "longitude": lon,
                                "mean_sea_level_pressure": pressure})
    assert seed.method == "mslp"
    assert seed.point == pytest.approx((lat[20, 20], lon[20, 20]))


@pytest.mark.parametrize("source", cs.source_ids())
def test_every_planable_source_reaches_a_field_seeding_rung(source):
    """No source is left with a seeding chain that cannot start.

    The vorticity and warm-core rungs need the upper-air quartet, which
    every planable row declares -- through its packaged mapping where it
    has one, through this column where it does not.  A new row that
    declares neither would be listed by --list-sources and then seed from
    nothing but an advisory, which is the silent gap this fails on.
    """

    declared = set(source_inventory(source).fields)
    assert {"air_pressure", "air_temperature", "eastward_wind",
            "northward_wind"} <= declared


# -- the door ---------------------------------------------------------------

def test_two_sources_differ_only_in_source_derived_fields():
    """The arbitrary-acceptance claim, stated as a document diff.

    Two sources with nothing in common but a planable registry row author
    the SAME 12/3 km moving tree at the same point: the grid, the
    projection, the vertical ladder and model top, the nest, the follower
    and the output cadences are byte-equal, and what differs is the
    acquisition block, the preparation recipe, the recommended physics and
    the configuration name.  A per-source code path would show up here as
    a difference in something else.
    """

    left = _raw(forcing_source="gfs")
    right = _raw(forcing_source="ecmwf-open-data")
    # Whole tables that are source-derived by construction ([ingest] holds
    # the GFS route's lake-ice choice, door item 7).
    for table in ("fetch", "case_data", "physics", "ingest"):
        left.pop(table, None)
        right.pop(table, None)
    # One source-derived key inside a shared table: the configuration name
    # carries the source title.  The model top is not one: GFS's certified
    # ladder stops at 100 hPa, but its fetch is asked for the config's own
    # top, so both sources carry the default.
    for row in (left, right):
        row["experiment"].pop("name")
    assert left["shared"]["p_top"] == right["shared"]["p_top"] == 5000.0
    assert left == right
    assert left["domain"][1]["follow"]["track"]["path"] == "storm-track.d02.csv"


def test_the_tree_admission_is_priced_with_the_selected_sources_interval(
        monkeypatch):
    """The interval is an ingest operand, not a label on the document.

    It reaches the tree road through the same ``operands`` the flat budget
    is built from, so a six-hourly source's moving tree is admitted on
    six-hourly boundary storage rather than on the door's old constant.
    """

    seen = []
    original = dw.estimate_phases
    monkeypatch.setattr(dw, "estimate_phases", lambda exp, **kw:
                        (seen.append(kw) or original(exp, **kw)))
    sizing = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture")
    tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=sizing,
                    forcing_source="aifs", tiles="off")
    assert seen
    assert all(kw["ingest_forcing_interval_seconds"] == 21600. for kw in seen)


def test_the_source_menu_is_a_kind_and_carries_the_client_contract(capsys):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--list-sources"])
    assert tc.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == tc.SCHEMA
    assert result["kind"] == "sources"
    assert result["forecast_started"] is False
    rows = {row["source"]: row for row in result["sources"]}
    assert set(rows) == set(cs.source_ids())
    for source, row in rows.items():
        assert set(row) == {"source", "label", "members", "default_member",
                            "forcing_interval_seconds", "cycle_hours",
                            "coverage_envelope", "follow_statics",
                            "max_forecast_hour"}
        assert row["default_member"] in (None, *row["members"])
        assert row["coverage_envelope"] is None or len(row["coverage_envelope"]) == 4


def test_the_document_carries_the_member_id_at_the_top_level():
    sizing = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture")
    result = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=sizing,
                             forcing_source="gefs", member="p03", tiles="off")
    assert result["member"] == "p03"
    assert result["source"] == "gefs"
    assert result["forcing_interval_seconds"] == 10800.


def test_the_default_source_still_authors_the_document_it_always_did():
    """A caller that never learned the flag reads the same keys."""

    result = _raw()
    assert result["fetch"]["source"] == tc.DEFAULT_SOURCE


# -- the authoring door and the run door, about one moving nest -------------


def _global_source(*, integrates: bool) -> str:
    """A global planable source whose chain does or does not feed a nest.

    Chosen from the tables at request time rather than named here: which
    chain a row sits on is registry work, and a test that wrote a model
    name down would be the per-model path the door refuses to have.
    """

    rows = [row for row in cs.source_options()
            if row["coverage_envelope"] is None
            and (row["follow_statics"] is not None) is integrates
            and cs.follow_statics(row["source"])["launch_refusal"] is None]
    assert rows, "no global planable source on that side of the table"
    return rows[0]["source"]


def test_every_listed_source_says_how_its_chain_feeds_a_moving_nest():
    """RED while the menu was silent about it.

    The menu is what a client builds a source picker from, and this door
    authors a MOVING nest on every row of it.  Whether the chosen row's
    chain can deliver the statics that nest travels over is a lookup in
    the run door's own table, so the row carries the answer instead of
    leaving the reader to meet it at the launch.
    """

    from woof import runplan
    from woof.source_cli import preparation_statics

    rows = cs.source_options()
    assert rows
    for row in rows:
        decision = runplan.source_follow_statics(row["source"])
        assert row["follow_statics"] == decision["delivery"]
        if row["follow_statics"] is not None:
            # Not merely non-null: the word is the run door's own, and
            # that chain really does deliver.
            assert (preparation_statics(decision["chain"])["delivery"]
                    == row["follow_statics"])
    assert cs.moving_nest_sources() == tuple(
        row["source"] for row in rows if row["follow_statics"] is not None)


def _planned(source):
    with source_licence(source):
        sizing = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture")
        return tc.plan_cyclone(cycle=cycle_for(source, CYCLE), point=POINT,
                               sizing=sizing, forcing_source=source, tiles="off")


def _missing_corridor_source(monkeypatch):
    """Exercise a future missing capability without mislabeling a live route."""
    from woof import runplan
    from woof.source_cli import preparation_statics

    source = _global_source(integrates=True)
    chain = runplan.source_follow_statics(source)["chain"]
    from dataclasses import replace
    from woof import source_cli

    rows = source_cli.preparation_runners()
    monkeypatch.setattr(source_cli, "preparation_runners", lambda: {
        key: replace(row, corridor_stage=None) if row.chain == chain else row
        for key, row in rows.items()})
    return source


def test_the_document_states_what_the_run_door_will_do_with_the_nest(monkeypatch):
    """RED while the document said nothing: the configuration was emitted
    as admitted and the reader met the launch refusal instead."""

    source = _missing_corridor_source(monkeypatch)
    result = _planned(source)
    block = result["follow_statics"]
    # `refusal` joins the block with the fix that made the note quote the
    # run door instead of paraphrasing it: the sentence a machine reader
    # gets is the one the launch raises, unwrapped.
    assert set(block) == {"source", "chain", "delivery", "launch_refusal",
                          "refusal", "integrates_moving_nest", "reason",
                          "note"}
    assert block["integrates_moving_nest"] is False
    assert block["delivery"] is None
    assert block["reason"]
    # The run door is named, with what it does and when, and the way on
    # is the set of sources that do carry a moving nest.
    assert "woof go" in block["note"]
    assert "before any fetch" in block["note"]
    # The sources offered are the ones that carry a moving nest AND reach
    # this centre; one that does not reach it is not a way on.
    for covering in cs.moving_nest_sources(POINT):
        assert covering in block["note"]
    # And the file that will be read carries the same sentence.
    assert "Moving nest on this source's chain" in result["config_text"]


def test_a_source_whose_chain_feeds_the_nest_says_so_and_adds_no_notice():
    from woof import runplan
    from woof.source_cli import preparation_statics

    source = _global_source(integrates=True)
    result = _planned(source)
    block = result["follow_statics"]
    assert block["integrates_moving_nest"] is True
    assert block["delivery"] == preparation_statics(block["chain"])["delivery"]
    assert block["reason"] is None
    assert "Moving nest on this source's chain" not in result["config_text"]


_CANDIDATE_POINTS: list[tuple[float, float]] = []


def _candidate_points() -> list[tuple[float, float]]:
    """Centres to try, derived from the rows rather than written down.

    Each row publishes a coverage envelope, and the middle of one is
    inside the grid it bounds often enough to author there; a rotated or
    masked grid whose bounding box is wider than the grid itself is why
    this is a ladder and not a formula.  The module's own global point
    leads, so a global row is authored at the same centre every other
    test here uses.
    """

    if not _CANDIDATE_POINTS:
        _CANDIDATE_POINTS.append(POINT)
        for row in cs.source_options():
            envelope = row["coverage_envelope"]
            if envelope is not None:
                south, west, north, east = envelope
                _CANDIDATE_POINTS.append(((south + north) / 2.,
                                          (west + east) / 2.))
    return _CANDIDATE_POINTS


def _planned_at(source, point):
    with source_licence(source):
        sizing = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture")
        return tc.plan_cyclone(cycle=cycle_for(source, CYCLE), point=point,
                               sizing=sizing, forcing_source=source, tiles="off")


def _planned_anywhere(source):
    """This source's setup, authored at the first centre its grid admits.

    ``None`` for a source whose declared window is smaller than the preset
    root: the door refuses it by name at its own centre, listing the
    sources that can carry the preset, and no file is emitted.
    """

    if not holds_the_preset_root(source):
        with pytest.raises(ValueError, match="covering sources"):
            _planned_at(source, center(source))
        return None
    refusals = []
    for point in _candidate_points():
        try:
            return _planned_at(source, point)
        except ValueError as refusal:
            refusals.append(f"{point}: {refusal}")
    raise AssertionError(f"no candidate centre authored on {source}: "
                         + "; ".join(refusals))


def test_every_source_agrees_with_the_run_door_about_what_it_emitted(tmp_path):
    """The seam itself, derived the way the RUN door derives it.

    RED while the comparison handed the document's OWN chain back to
    ``follow_statics_decision``: both sides then indexed one table with
    one value, so the assertion held for any chain at all, including a
    chain the run door would never pick.  What the run door actually
    does with an emitted file is read it: the route comes from whether
    the file carries a ``[case_data]`` table, the chain from
    ``runplan._chain_key`` -- and a source with no launch route reaches
    neither, because ``woof go`` refuses it at
    ``prepared_chain_for_source`` before anything in the file is
    considered.  Every planable row, not one row per side.
    """

    from woof import go_cli, runplan
    from woof.cli import build_parser
    from woof.explain import split

    sources = cs.source_ids()
    assert sources
    for source in sources:
        result = _planned_anywhere(source)
        if result is None:
            # Refused before any file was emitted, so there is nothing
            # for the run door to read.
            continue
        block = result["follow_statics"]
        payload = tomllib.loads(result["config_text"])
        if block.get("launch_refusal") is not None:
            # The run door refuses the SOURCE.  Asserted against the run
            # door's own gate and then against `woof go` itself, so the
            # sentence the reader was shown while choosing is the
            # sentence the launch raises.
            assert block["chain"] is None and block["delivery"] is None
            assert "case_data" not in payload
            with pytest.raises(runplan.PlanError) as gate:
                runplan.prepared_chain_for_source(source)
            assert split(str(gate.value))[0] == block["launch_refusal"]
            config = tmp_path / f"{source}.toml"
            config.write_text(result["config_text"], encoding="utf-8")
            args = build_parser().parse_args(["go", str(config)])
            with pytest.raises(runplan.PlanError) as launch:
                go_cli.go_main(args)
            assert split(str(launch.value))[0] == block["launch_refusal"]
            continue
        route = "experiment" if "case_data" in payload else "prepared"
        chain = runplan._chain_key(route, source)
        assert chain == block["chain"]
        exp = dw.experiment_from_text(result["config_text"],
                                      source="fixture.toml")
        decision = runplan.follow_statics_decision(exp, chain=chain)
        assert decision is not None
        assert decision["delivery"] == block["delivery"]
        assert (decision["refusal"] is None) is block["integrates_moving_nest"]
        # Every currently reachable preparation owns the required geography
        # representation. A compatible future registry row inherits that fact.
        assert decision["refusal"] is None, (source, chain, decision)
        assert block["integrates_moving_nest"] is True


def test_a_source_with_no_launch_route_says_that_and_not_a_statics_answer():
    """RED while one sentence served two different facts.

    A row that reaches no chain cannot launch at all, so a note about
    the statics a moving nest travels over names the wrong obstacle, and
    the way out it offered -- drop the follow source and keep a
    bounds-only ``[relocation]`` -- leaves the configuration exactly as
    unlaunchable as it was.
    """

    from woof import runplan
    from woof.source_cli import preparation_statics
    from woof.explain import split

    rows = [source for source in cs.source_ids()
            if runplan.source_follow_statics(source)["chain"] is None]
    if not rows:
        pytest.skip("every planable row reaches a launch chain in this registry")
    for source in rows:
        block = cs.moving_nest_note(source,
                                    follower_grid_ids=(tc.FOLLOWER_GRID_ID,))
        with pytest.raises(runplan.PlanError) as gate:
            runplan.prepared_chain_for_source(source)
        assert split(str(gate.value))[0] in block["note"]
        assert "bounds-only" not in block["note"]
        assert "no launch route" in block["note"]
        # The way on is still named, and every source it names launches.
        assert cs.moving_nest_sources()
        for covering in cs.moving_nest_sources():
            assert covering in block["note"]
            assert runplan.source_follow_statics(
                covering)["chain"] is not None
        # And the file that will be read says which question it answers.
        assert "Launch route for this source:" in _planned_anywhere(source)["config_text"]


def test_the_way_out_the_note_names_is_the_one_its_config_can_take(monkeypatch):
    """RED while the note sent the reader to a table their file has not.

    The unsupported-chain note used to end "or run this configuration
    with a bounds-only ``[relocation]``".  The configuration this door
    emits declares its follower in ``[[domain]].follow`` and carries no
    ``[relocation]`` table at all, and the run door's predicate stays True
    while any ``[[domain]].follow`` remains -- so a reader who followed
    that sentence to the letter met the same refusal, word for word.
    (Carried from lane land/2.7.4-cyclone-any-model 84cc46fb21.)
    """

    from woof.core.nest_lifecycle import FOLLOWER_TABLE
    from woof.static.corridor import config_declares_follow_source

    result = _planned(_missing_corridor_source(monkeypatch))
    block = result["follow_statics"]
    assert block["launch_refusal"] is None
    # The subject is the follower the file declares, not a table it has not.
    assert FOLLOWER_TABLE in block["refusal"]
    assert f"d{tc.FOLLOWER_GRID_ID:02d}" in block["refusal"]
    assert "[relocation] follow source" not in block["refusal"]
    # The document's note quotes the run door's sentence, reflowed onto
    # the one line the note is, rather than paraphrasing it.
    assert block["refusal"] == runplan.follow_unsupported_refusal(
        block["chain"], (tc.FOLLOWER_GRID_ID,), source=block["source"])
    assert " ".join(block["refusal"].split()) in block["note"]
    assert "delete the follow table from d%02d" % tc.FOLLOWER_GRID_ID in block["refusal"]

    raw = tomllib.loads(result["config_text"])
    assert "relocation" not in raw, "the emitted file carries no [relocation]"
    follower = next(row for row in raw["domain"]
                    if row["grid_id"] == tc.FOLLOWER_GRID_ID)
    # The predicate the run door asks: the way out the refusal names
    # clears it, and the one the note used to offer does not.
    domains = [SimpleNamespace(grid_id=row["grid_id"], follow=row.get("follow"))
               for row in raw["domain"]]
    bounds_only = SimpleNamespace(enabled=True, grid_id=tc.FOLLOWER_GRID_ID,
                                  follow=None, moves=())
    assert config_declares_follow_source(
        SimpleNamespace(domains=domains, relocation=bounds_only))
    del follower["follow"]
    domains = [SimpleNamespace(grid_id=row["grid_id"], follow=row.get("follow"))
               for row in raw["domain"]]
    assert not config_declares_follow_source(
        SimpleNamespace(domains=domains,
                        relocation=SimpleNamespace(enabled=False, grid_id=None,
                                                   follow=None, moves=())))


def test_the_refusal_names_only_the_follow_tables_the_config_carries(monkeypatch):
    """RED while the subject came from the MOVER set.

    ``moving_grid_ids`` counts the ``[relocation]`` mover AND the
    ``[relocation.containment]`` ancestor it slides inside, and that
    ancestor declares nothing at all.  Read as the subject, it made the
    refusal state as a fact that a domain carried a ``[[domain]].follow``
    it has never had.  The declarations are read instead, and both live
    tables are named where both are live.
    (Carried from lane land/2.7.4-cyclone-any-model ca13dcdbd2.)
    """

    from woof import source_cli
    from woof.core.nest_lifecycle import FOLLOWER_TABLE

    monkeypatch.setattr(source_cli, "preparation_statics",
                        lambda chain, source=None: {
                            "delivery": None,
                            "reason": "its preparation seals no corridor"})
    tracker = object()
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1, parent_id=0, follow=None),
                 SimpleNamespace(grid_id=2, parent_id=1, follow=tracker),
                 SimpleNamespace(grid_id=3, parent_id=2, follow=None)),
        relocation=SimpleNamespace(
            enabled=True, grid_id=3, follow=tracker, moves=(),
            containment=SimpleNamespace(grid_id=2)))
    decision = runplan.follow_statics_decision(exp, chain="prepared:staged")
    assert decision["follower_grid_ids"] == [2]
    message = decision["refusal"]
    # d03 carries no follow table, and is never said to.
    assert f"{FOLLOWER_TABLE} on d02 and d03" not in message
    assert "delete the follow table from d02 and d03" not in message
    assert (f"{FOLLOWER_TABLE} on d02 AND a [relocation] follow source on d03"
            in message)
    # The withdrawal names both, because either one left standing keeps
    # `config_declares_follow_source` True.
    assert "withdraw BOTH declarations" in message
    assert "delete the follow table from d02" in message
    assert "drop the follow source for a bounds-only [relocation]" in message


def test_the_limit_is_stated_once_on_the_human_channel(monkeypatch, capsys):
    """The warning the document's note is printed as, counted.

    One line, on the channel a person reads, for a source whose chain
    cannot feed the nest; none at all for one that can.  Red while
    nothing in the suite held the count.
    """

    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    budget = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture", measured=False)
    monkeypatch.setattr(dw, "_domain_target_hardware",
                        lambda args: (budget, None, False))

    def _run(source):
        args = parser.parse_args(["cyclone-setup", "--cycle",
                                  cycle_for(source, CYCLE),
                                  "--point=18,-65", "--vram-gib", "32",
                                  "--tiles", "off", "--source", source])
        assert tc.main(args) == 0
        captured = capsys.readouterr()
        return json.loads(captured.out), captured.err

    with monkeypatch.context() as missing:
        result, errors = _run(_missing_corridor_source(missing))
        note = result["follow_statics"]["note"]
        assert [line for line in errors.splitlines()
                if line.startswith("warning: ")] == ["warning: " + note]

    result, errors = _run(_global_source(integrates=True))
    assert result["follow_statics"]["integrates_moving_nest"] is True
    assert "moving nest" not in errors


def test_every_source_states_its_moving_nest_answer_in_words():
    """Including the rows that reach no chain at all.

    Those are not "on the None chain", and a note that said so would read
    as a defect in the door rather than as a fact about the source.
    """

    for source in cs.source_ids():
        note = cs.moving_nest_note(
            source, follower_grid_ids=(tc.FOLLOWER_GRID_ID,))["note"]
        assert note and "None" not in note
        assert cs.source_adapter(source).display_title in note


def test_the_menu_row_carries_the_moving_nest_answer(capsys):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--list-sources"])
    assert tc.main(args) == 0
    rows = json.loads(capsys.readouterr().out)["sources"]
    assert all("follow_statics" in row for row in rows)
    assert any(row["follow_statics"] is not None for row in rows)
    assert all(row["follow_statics"] is not None
               or cs.follow_statics(row["source"])["launch_refusal"] is not None
               for row in rows)
