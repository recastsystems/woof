"""``output_level`` end to end: config to columns to a parsed file.

The contract itself lives in three places and is pinned there --
``tests/test_relocation_track_config.py`` (which surfaces become
report-only), ``tests/test_storm_track_writer.py`` (which columns, in
which order) and ``tests/test_storm_tracking.py`` (which surfaces vote).
This file is the join: a writer built the way the runtime builds one,
carrying the UNION of steering and report-only surfaces, and the two
properties a reader depends on.

* An ABSENT key is byte-identical to the file this module has always
  written, so no existing config's deck moves.
* Every row is complete and ``csv.DictReader``-parseable at any width,
  including a twenty-surface profile -- the shape the key exists for.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import woof.core.storm_track_writer as tw
from woof.core.storm_track_writer import TrackConfig, TrackWriter
from woof.core.storm_tracking import FollowConfig, all_levels_of

from test_storm_track_writer import _Grid, _native_surface_contract

INIT = tw.dt.datetime(2025, 10, 24, 12, 0, 0)
STEERING = (850.0, 700.0, 500.0)
PROFILE = (925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
           550.0, 500.0, 450.0, 400.0, 350.0, 300.0)


def _follow(level_hpa=STEERING, report=()):
    return FollowConfig(field="pressure", threshold=30.0,
                        level_hpa=level_hpa, report_level_hpa=tuple(report),
                        search_margin_cells=12, min_shift_cells=1,
                        max_shift_cells=6, cooldown_seconds=600.0)


def _fix(levels):
    """A consultation that answered on every surface it was asked for."""
    fix = SimpleNamespace(
        found={"ci": 100.0, "cj": 50.0, "cells": 12, "max_value": -964.4},
        evidence={"t": 3600.0, "extremum_units": "m"},
        field_used="pressure",
        search_box=(slice(0, 60), slice(0, 120)),
        center_parent_ij=(100.0, 50.0), extremum=964.4,
        refined_on=None, refined_cell_ij=None,
        footprint=SimpleNamespace(grid_id=3),
        plane_shape=(60, 120), threshold_used=30.0, raw_shift=(0.0, 0.0))
    fix.levels = [SimpleNamespace(level_hpa=v, fix_ij=(20.0 + i, 30.0 + i),
                                  height_dam=100.0 + i)
                  for i, v in enumerate(levels)]
    return fix


def _state():
    return SimpleNamespace(physics=SimpleNamespace(fields={
        "u10": np.full((8, 8), 30.0), "v10": np.full((8, 8), 40.0)}))


@pytest.fixture(autouse=True)
def _mslp(monkeypatch):
    """The surface block's own reduction, stubbed.

    An ISOBARIC tracker's extremum is metres of height, so the central
    pressure is reduced rather than read off the fix -- and these states
    carry no prognostic column.  What this file is about is which LEVEL
    blocks a row carries, so the surface block is held constant.
    """
    monkeypatch.setattr(tw, "mslp_hpa_from_state",
                        lambda state, window=None: np.full((60, 120), 964.4))


def _writer(tmp_path, follow, output_level=None):
    """Built the way woof.runtime.build_track_writer builds one: the
    writer is handed EVERY surface a consultation computes."""
    return TrackWriter(TrackConfig("track.csv", output_level=output_level),
                       initial_time=INIT, outdir=tmp_path,
                       levels=all_levels_of(follow),
                       tracked_field=follow.field)


def _lines(tmp_path):
    return (tmp_path / "track.csv").read_text(
        encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# The absent key: nothing moves
# ---------------------------------------------------------------------------

def test_an_absent_output_level_still_follows_the_tracking_levels(tmp_path):
    follow = _follow()
    writer = _writer(tmp_path, follow)
    writer.emit(_fix(STEERING), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    writer.close()
    header, row = _lines(tmp_path)
    assert header.split(",") == [
        "valid_time", "lat_deg", "lon_deg", "mslp_mb", "vmax_m_s",
        "850_lat_deg", "850_lon_deg", "850_hgt_dam",
        "700_lat_deg", "700_lon_deg", "700_hgt_dam",
        "500_lat_deg", "500_lon_deg", "500_hgt_dam"]
    cells = row.split(",")
    assert [cells[7], cells[10], cells[13]] == ["100.00", "101.00", "102.00"]


def test_a_run_with_no_report_levels_writes_the_bytes_it_always_wrote(
        tmp_path):
    """THE GOLDEN.  A follow config that names no report surfaces
    produces exactly the file a tree without the key produced: same
    levels, same order, same bytes."""
    plain = TrackWriter(TrackConfig("track.csv"), initial_time=INIT,
                        outdir=tmp_path / "a", levels=STEERING)
    plain.emit(_fix(STEERING), t=3600.0, parent_state=_state(),
               parent_grid=_Grid())
    plain.close()
    wired = _writer(tmp_path / "b", _follow())
    wired.emit(_fix(STEERING), t=3600.0, parent_state=_state(),
               parent_grid=_Grid())
    wired.close()
    assert (tmp_path / "a" / "track.csv").read_bytes() == (
        tmp_path / "b" / "track.csv").read_bytes()


# ---------------------------------------------------------------------------
# The two lists, decoupled, through the writer
# ---------------------------------------------------------------------------

def test_a_profile_reports_surfaces_the_nest_is_not_steered_by(tmp_path):
    """The configuration this exists for: steer on three, print fourteen.

    The writer never searches for a centre; the TRACKER computed all
    fourteen (all_levels_of) and the extras simply did not vote.
    """
    report = tuple(v for v in PROFILE if v not in STEERING)
    follow = _follow(report=report)
    assert all_levels_of(follow) == STEERING + report
    writer = _writer(tmp_path, follow, output_level=(0.0,) + PROFILE)
    row = writer.emit(_fix(all_levels_of(follow)), t=3600.0,
                      parent_state=_state(), parent_grid=_Grid())
    receipt = writer.close()
    assert row["emitted"] and receipt["skipped"] == 0
    want = ["valid_time", "lat_deg", "lon_deg", "mslp_mb", "vmax_m_s"]
    for level in PROFILE:
        want += [f"{level:g}_lat_deg", f"{level:g}_lon_deg",
                 f"{level:g}_hgt_dam"]
    assert _lines(tmp_path)[0].split(",") == want


def test_a_surface_the_file_skips_is_still_steering(tmp_path):
    """output_level chooses what the FILE carries, never what the tracker
    watches: 700 keeps steering the nest while the file prints 925."""
    follow = _follow(report=(925.0,))
    writer = _writer(tmp_path, follow, output_level=(925.0,))
    writer.close()
    assert writer.emitted_levels == (925.0,)
    assert follow.level_hpa == STEERING


def test_every_row_stays_dictreader_parseable_and_complete(tmp_path):
    report = tuple(v for v in PROFILE if v not in STEERING)
    follow = _follow(report=report)
    writer = _writer(tmp_path, follow, output_level=(0.0,) + PROFILE)
    for t in (3600.0, 7200.0):
        writer.emit(_fix(all_levels_of(follow)), t=t,
                    parent_state=_state(), parent_grid=_Grid())
    # ...and a consultation that found nothing, which builds its cells
    # on a separate path and must still match the header's width.
    blank = _fix(all_levels_of(follow))
    blank.found = None
    writer.emit(blank, t=10800.0, parent_state=_state(),
                parent_grid=_Grid())
    writer.close()
    with (tmp_path / "track.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    width = 5 + 3 * len(PROFILE)
    for row in rows:
        assert len(row) == width
        assert None not in row and None not in row.values()
