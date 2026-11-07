"""The track writer: one plain columnar file, rendered from the fix.

There is no Fortran oracle behind this and there was never going to be
one -- WRF-ARW writes no track file; its moving nest prints a centre to
``rsl.out`` and the decks in that ecosystem come from the GFDL vortex
tracker, a separate post-processing program.  So what these tests pin is
not a transcription but a CONTRACT: the columns are where the header
says, the units are what the header says, the clock is the valid time,
and a row is either accurate or absent.

The properties that actually matter, each a section below:

* the time column is the VALID TIME, spelled as wrfout spells it, so a
  row joins onto a history frame by string equality;
* every quantity in a row comes from ONE grid -- the one that located
  the centre;
* a no-signal consultation writes a row with NaN data and a complete
  clock, so a gap is visible rather than inferred;
* a rotation or echo tracker writes POSITION ONLY, and the position is
  the moving domain's own centre rather than the signal's -- with the
  centre arithmetic checked against the real projection, not against
  this file's copy of it;
* nothing the writer can encounter is allowed to raise, because a
  diagnostic that can end a twelve-hour forecast is a defect.
"""

from __future__ import annotations

import datetime as dt
import math
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import storm_track_writer as tw
from woof.core.storm_track_writer import (TrackConfig, TrackRefusal,
                                           TrackWriter, build_track_config,
                                           format_row)

INIT = dt.datetime(2025, 10, 24, 12, 0, 0)


# ---------------------------------------------------------------------------
# The row
# ---------------------------------------------------------------------------

def _row(**over):
    base = dict(valid_time=dt.datetime(2025, 10, 24, 18, 30, 0),
                lat=14.5321, lon=-73.1044, mslp_mb=964.42, vmax_m_s=48.03)
    base.update(over)
    return format_row(**base)


def test_the_row_is_pinned_column_for_column():
    """The whole format in one assertion, so a change has to be
    deliberate.  A reader splits on commas -- or hands the file to
    ``read_csv`` and never thinks about the format again."""
    assert _row() == "2025-10-24_18:30:00,14.5321,-73.1044,964.42,48.03"


def test_the_time_column_is_the_valid_time_not_elapsed_seconds():
    """The trap this format used to have: model seconds made a reader
    need the run's initial time to place a row."""
    stamp = _row().split(",")[0]
    assert stamp == "2025-10-24_18:30:00"
    # Round-trips through the format wrfout uses for its own Times
    # variable and its filenames, so a join is string equality.
    assert dt.datetime.strptime(stamp, tw.TIME_FORMAT) == \
        dt.datetime(2025, 10, 24, 18, 30, 0)


def test_positions_are_signed_decimal_degrees():
    """No hemisphere letters, no tenths: this file is read and plotted."""
    fields = _row().split(",")
    assert float(fields[1]) == pytest.approx(14.5321)
    assert float(fields[2]) == pytest.approx(-73.1044)
    assert float(fields[3]) == pytest.approx(964.42)
    assert float(fields[4]) == pytest.approx(48.03)


@pytest.mark.parametrize("lat,lon", [
    (14.5, 73.1), (14.5, -73.1), (-14.5, 73.1), (-14.5, -73.1),
    (0.0, 0.0), (89.9, 179.9), (-89.9, -179.9),
])
def test_all_four_quadrants_and_both_seams_round_trip(lat, lon):
    fields = _row(lat=lat, lon=lon).split(",")
    assert float(fields[1]) == pytest.approx(lat, abs=5e-5)
    assert float(fields[2]) == pytest.approx(lon, abs=5e-5)


def test_the_header_names_every_column_and_its_unit():
    # LINE ONE IS THE COLUMN NAMES, with no banner above them.  That is
    # the whole reason `read_csv(path)` needs no arguments, so it is
    # worth an assertion of its own rather than an implication of the
    # one below.
    assert not tw.CSV_HEADER.startswith("#")
    for token in ("valid_time", "lat_deg", "lon_deg", "mslp_mb",
                  "vmax_m_s", ):
        assert token in tw.CSV_HEADER
    # One name per column, in order.
    names = tw.CSV_HEADER.split(",")
    assert names == ["valid_time", "lat_deg", "lon_deg", "mslp_mb",
                     "vmax_m_s"]
    assert len(names) == len(_row().split(","))


def test_a_no_signal_row_is_NaN_and_keeps_its_clock():
    line = format_row(valid_time=dt.datetime(2025, 10, 24, 13, 0, 0),
                      lat=None, lon=None, mslp_mb=None, vmax_m_s=None)
    fields = line.split(",")
    assert fields[0] == "2025-10-24_13:00:00"
    assert fields[1:5] == ["NaN"] * 4
    assert all(math.isnan(float(f)) for f in fields[1:5])
    # No marker column: a row of all-NaN data with its clock intact IS
    # the no-signal statement, and the time axis stays complete.
    assert fields[1:5] == ["NaN"] * 4


def test_a_no_signal_row_keeps_the_column_count():
    """CSV owes its reader a fixed column COUNT, not a fixed width.  A
    NaN row carrying fewer commas would slide every column after it into
    the wrong name -- which is the one way a missing value can corrupt a
    value that is present."""
    good, bad = _row(), format_row(
        valid_time=dt.datetime(2025, 10, 24, 18, 30, 0), lat=None, lon=None,
        mslp_mb=None, vmax_m_s=None)
    assert good.count(",") == bad.count(",") == 4
    assert len(good.split(",")) == len(bad.split(",")) == 5
    # And no padding crept back in: a bare split gives usable floats
    # without skipinitialspace.
    assert bad.split(",")[1] == "NaN"
    assert _row().split(",")[1] == "14.5321"


def test_a_non_finite_value_writes_NaN_rather_than_inf():
    fields = _row(mslp_mb=float("inf"), vmax_m_s=float("nan")).split(",")
    assert fields[3] == "NaN" and fields[4] == "NaN"


# ---------------------------------------------------------------------------
# Config: two keys, one required
# ---------------------------------------------------------------------------

def test_a_minimal_table_validates():
    cfg = build_track_config({"path": "track.csv"}, "x")
    assert cfg.path == "track.csv" and cfg.interval_seconds is None


def test_an_interval_is_accepted_and_echoed():
    cfg = build_track_config({"path": "t.csv", "interval_seconds": 60.0}, "x")
    assert cfg.interval_seconds == 60.0
    assert cfg.to_json()["interval_seconds"] == 60.0


def test_no_interval_is_absent_from_the_echo_rather_than_null():
    assert "interval_seconds" not in build_track_config(
        {"path": "t.csv"}, "x").to_json()


@pytest.mark.parametrize("table,match", [
    ({}, "missing required key"),
    ({"path": "t.csv", "pathh": 1}, "does not have key"),
    ({"path": ""}, "must name a file"),
    ({"path": "   "}, "must name a file"),
    ({"path": 7}, "must be a string"),
    ({"path": "t.csv", "interval_seconds": -5.0}, "finite, positive"),
    ({"path": "t.csv", "interval_seconds": 0.0}, "finite, positive"),
    ({"path": "t.csv", "interval_seconds": "60"}, "must be a number"),
    ({"path": "t.csv", "interval_seconds": True}, "must be a number"),
])
def test_a_bad_table_refuses_by_name(table, match):
    with pytest.raises(ValueError, match=match):
        build_track_config(table, "x")


def test_a_near_miss_key_is_offered_a_suggestion():
    with pytest.raises(ValueError, match="interval_second"):
        build_track_config({"path": "t.csv", "interval_second": 60}, "x")


def test_an_array_of_tables_refuses_and_says_why():
    """The block used to be [[relocation.track]]; a config carried over
    from that shape must be told, not silently mis-parsed."""
    with pytest.raises(ValueError, match="single table"):
        build_track_config([{"path": "t.csv"}], "x")


# ---------------------------------------------------------------------------
# The writer, against a stub tree
# ---------------------------------------------------------------------------

class _Grid:
    """A trivially invertible stand-in for a ProjectedGrid.

    ``ij_to_latlon`` takes 1-BASED mass points, which is the convention
    ``latlon_from_grid`` exists to bridge; making that explicit here is
    the point of the fixture.
    """

    def __init__(self, lat0=10.0, lon0=-80.0, step=0.01):
        self.lat0, self.lon0, self.step = lat0, lon0, step
        self.e_we, self.e_sn = 121, 61

    def latlon_to_ij(self, lat, lon):
        return (1. + (lon-self.lon0)/self.step,
                1. + (lat-self.lat0)/self.step)

    def ij_to_latlon(self, x, y):
        return (self.lat0 + (y - 1.0) * self.step,
                self.lon0 + (x - 1.0) * self.step)


def _fix(*, found=True, refined_on=None, extremum_units="hPa",
         extremum=964.4, ci=100.0, cj=50.0, refine_cell=(30.0, 20.0),
         t=3600.0):
    return SimpleNamespace(
        found=({"ci": ci, "cj": cj, "cells": 12,
                "max_value": -extremum} if found else None),
        evidence={"t": t, "extremum_units": extremum_units},
        field_used="pressure",
        search_box=(slice(0, 60), slice(0, 120)),
        center_parent_ij=(ci, cj) if found else None,
        extremum=extremum if found else None,
        refined_on=refined_on,
        refined_cell_ij=refine_cell if refined_on else None,
        footprint=SimpleNamespace(grid_id=3),
        plane_shape=(60, 120), threshold_used=30.0, raw_shift=(0.0, 0.0))


def _state(u10=30.0, v10=40.0):
    return SimpleNamespace(physics=SimpleNamespace(fields={
        "u10": np.full((8, 8), u10), "v10": np.full((8, 8), v10)}))


@pytest.fixture(autouse=True)
def _native_surface_contract(monkeypatch):
    """These are writer/clock tests. Native reductions have separate ABI tests."""
    from woof.tc import field_products
    def sample(state, grid, centre, **kwargs):
        return dict(latitude_deg=centre[0], longitude_deg=centre[1],
                    mslp_hpa=964.4, vmax10_m_s=tw.peak_wind_m_s(state),
                    radii_nm={k:[0.]*4 for k in ('34','50','64')},
                    radii_reach_grid_edge={k:[False]*4 for k in ('34','50','64')})
    monkeypatch.setattr(field_products, 'diagnose', sample)


def _writer(tmp_path, cfg=None, **over):
    return TrackWriter(cfg or TrackConfig("track.csv"),
                       initial_time=INIT, outdir=tmp_path, **over)


def _lines(tmp_path, name="track.csv"):
    return (tmp_path / name).read_text(encoding="utf-8").splitlines()


def test_a_full_emission_writes_a_header_and_one_row(tmp_path):
    writer = _writer(tmp_path)
    writer.emit(_fix(), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    writer.close()
    lines = _lines(tmp_path)
    assert lines[0] == tw.CSV_HEADER
    assert len(lines) == 2
    fields = lines[1].split(",")
    assert fields[0] == "2025-10-24_13:00:00"       # INIT + 3600 s
    assert float(fields[3]) == pytest.approx(964.4)
    # 3-4-5 triangle: |wind| = 50 m/s exactly
    assert float(fields[4]) == pytest.approx(50.0)


def test_the_file_parses_with_the_stdlib_reader_and_no_arguments(tmp_path):
    """The point of the format.  ``DictReader`` with no ``comment=``, no
    ``skiprows=`` and no width table -- and every value reachable by the
    name the header gave it."""
    import csv as _csv
    writer = _writer(tmp_path)
    writer.emit(_fix(), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    writer.close()
    with (tmp_path / "track.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["valid_time"] == "2025-10-24_13:00:00"
    assert float(rows[0]["vmax_m_s"]) == pytest.approx(50.0)
    assert float(rows[0]["mslp_mb"]) == pytest.approx(964.4)
    # No leading blanks to strip: the raw field IS the number.
    assert rows[0]["mslp_mb"] == rows[0]["mslp_mb"].strip()


def test_no_field_ever_needs_quoting(tmp_path):
    """Every column is a number or a timestamp, so no row can contain a
    comma, a quote or a newline -- which is what makes a bare
    ``line.split(",")`` a correct parser and not merely a lucky one."""
    writer = _writer(tmp_path)
    for step in (1, 2):
        writer.emit(_fix(t=3600.0 * step), t=3600.0 * step,
                    parent_state=_state(), parent_grid=_Grid())
    writer.close()
    text = (tmp_path / "track.csv").read_text(encoding="utf-8")
    assert '"' not in text and "'" not in text
    for line in text.splitlines():
        assert len(line.split(",")) == len(tw.CSV_HEADER.split(","))


def test_the_clock_advances_with_model_time(tmp_path):
    writer = _writer(tmp_path)
    for t in (0.0, 1800.0, 3600.0, 90000.0):
        writer.emit(_fix(t=t), t=t, parent_state=_state(),
                    parent_grid=_Grid())
    writer.close()
    stamps = [ln.split(",")[0] for ln in _lines(tmp_path)[1:]]
    assert stamps == ["2025-10-24_12:00:00", "2025-10-24_12:30:00",
                      "2025-10-24_13:00:00", "2025-10-25_13:00:00"]


def test_quantities_come_from_the_refine_grid_when_stage_two_applied(
        tmp_path):
    """One rule: everything in a row is read from the grid that located
    the centre.  A 4.5 km grid cannot resolve an eye, so a position from
    d03 with a pressure from d02 would be a row that is wrong about
    intensity while looking right about place."""
    writer = _writer(tmp_path)
    writer.emit(_fix(refined_on=3), t=3600.0, parent_state=_state(1.0, 0.0),
                parent_grid=_Grid(), refine_state=_state(30.0, 40.0),
                refine_grid=_Grid(lat0=20.0, lon0=-60.0))
    writer.close()
    row = _lines(tmp_path)[1].split(",")
    # The refine grid's origin and the refine grid's wind, not the parent's.
    assert float(row[1]) == pytest.approx(20.0 + 20.0 * 0.01)
    assert float(row[2]) == pytest.approx(-60.0 + 30.0 * 0.01)
    assert float(row[4]) == pytest.approx(50.0)
    assert len(row) == 5


def test_without_a_refinement_the_parent_grid_is_named(tmp_path):
    writer = _writer(tmp_path)
    writer.emit(_fix(), t=3600.0, parent_state=_state(), parent_grid=_Grid())
    writer.close()
    assert len(_lines(tmp_path)[1].split(",")) == 5


def test_final_endpoint_can_be_written_between_requested_cadences(tmp_path):
    writer = _writer(tmp_path, TrackConfig("track.csv", interval_seconds=3600.))
    writer.emit(_fix(t=0.), t=0., parent_state=_state(), parent_grid=_Grid())
    assert writer.emit(_fix(t=1800.), t=1800., parent_state=_state(), parent_grid=_Grid()) is None
    row = writer.emit(_fix(t=1800.), t=1800., parent_state=_state(), parent_grid=_Grid(),force=True)
    assert row['emitted']
    writer.close()
    assert len(_lines(tmp_path)) == 3
    assert _lines(tmp_path)[-1].startswith('2025-10-24_12:30:00,')


def test_no_signal_writes_a_NaN_row(tmp_path):
    writer = _writer(tmp_path)
    writer.emit(_fix(found=False), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    receipt = writer.close()
    lines = _lines(tmp_path)
    assert len(lines) == 2 and lines[1].split(",")[1:5] == ["NaN"] * 4
    assert receipt["records"] == 1 and receipt["skipped"] == 0


def test_the_interval_is_honoured(tmp_path):
    writer = _writer(tmp_path, TrackConfig("track.csv",
                                           interval_seconds=1800.0))
    for step in range(1, 31):                 # 360 s cadence, 3 hours
        t = 360.0 * step
        writer.emit(_fix(t=t), t=t, parent_state=_state(),
                    parent_grid=_Grid())
    writer.close()
    stamps = [ln.split(",")[0] for ln in _lines(tmp_path)[1:]]
    assert stamps == ["2025-10-24_12:06:00", "2025-10-24_12:36:00",
                      "2025-10-24_13:06:00", "2025-10-24_13:36:00",
                      "2025-10-24_14:06:00", "2025-10-24_14:36:00"]


def test_no_interval_emits_every_consultation(tmp_path):
    writer = _writer(tmp_path)
    for step in range(1, 6):
        t = 360.0 * step
        writer.emit(_fix(t=t), t=t, parent_state=_state(),
                    parent_grid=_Grid())
    writer.close()
    assert len(_lines(tmp_path)) == 6


# ---------------------------------------------------------------------------
# It must never be able to fail a forecast
# ---------------------------------------------------------------------------

def test_a_missing_wind_is_a_receipt_fault_not_an_exception(tmp_path):
    writer = _writer(tmp_path)
    naked = SimpleNamespace(physics=SimpleNamespace(fields={}))
    row = writer.emit(_fix(), t=3600.0, parent_state=naked,
                      parent_grid=_Grid())
    receipt = writer.close()
    assert row["emitted"] is False and "u10" in row["reason"]
    assert receipt["skipped"] == 1 and receipt["records"] == 0
    assert receipt["fault_count"] == 1
    assert len(_lines(tmp_path)) == 1              # header only


def test_a_missing_grid_is_a_receipt_fault_not_an_exception(tmp_path):
    writer = _writer(tmp_path)
    row = writer.emit(_fix(), t=3600.0, parent_state=_state(),
                      parent_grid=None)
    writer.close()
    assert row["emitted"] is False
    assert "projected grid" in row["reason"]


def test_an_impossible_position_is_a_fault_not_a_bad_row(tmp_path):
    """A grid that hands back a latitude of 400 must not reach the file."""
    class Broken(_Grid):
        def ij_to_latlon(self, x, y):
            return (400.0, 0.0)

    writer = _writer(tmp_path)
    row = writer.emit(_fix(), t=3600.0, parent_state=_state(),
                      parent_grid=Broken())
    writer.close()
    assert row["emitted"] is False and "impossible position" in row["reason"]
    assert len(_lines(tmp_path)) == 1


def test_a_fault_records_the_clock_not_a_bare_number(tmp_path):
    writer = _writer(tmp_path)
    writer.emit(_fix(), t=3600.0,
                parent_state=SimpleNamespace(physics=None),
                parent_grid=_Grid())
    receipt = writer.close()
    assert receipt["faults"][0].startswith("2025-10-24_13:00:00")


# ---------------------------------------------------------------------------
# Central pressure: the right quantity or none
# ---------------------------------------------------------------------------

def test_an_mslp_tracker_reuses_the_extremum_without_reducing_again(
        monkeypatch):
    """field = "pressure" with no level_hpa: the tracker's own signal IS
    sea-level pressure, so a track row costs nothing extra."""
    monkeypatch.setattr(
        tw, "mslp_hpa_from_state",
        lambda *a, **k: pytest.fail("should not reduce again"))
    assert tw.central_pressure_mb(_fix(extremum_units="hPa", extremum=964.4),
                                  _state(), on_refine_grid=False) == 964.4


@pytest.mark.parametrize("units", ["m", "field"])
def test_a_non_pressure_extremum_is_never_printed_as_a_pressure(units,
                                                                monkeypatch):
    """Under level_hpa the extremum is METRES of geopotential height, and
    under uh/reflectivity it is m2 s-2 or dBZ.  Emitting any of them in
    the mslp_mb column would be a plausible number of the wrong
    quantity."""
    seen = {}

    def fake(state, window=None):
        seen["window"] = window
        return np.full((60, 120), 971.25)

    monkeypatch.setattr(tw, "mslp_hpa_from_state", fake)
    got = tw.central_pressure_mb(_fix(extremum_units=units, extremum=1502.0),
                                 _state(), on_refine_grid=False)
    assert got == pytest.approx(971.25)
    assert seen["window"] is not None          # cropped to the searched box


def test_the_refine_grid_reduction_is_not_cropped_to_a_parent_box(
        monkeypatch):
    """The search box is in the PARENT's index space; applying it to the
    refine grid would crop the wrong rectangle of a different domain."""
    seen = {}

    def fake(state, window=None):
        seen["window"] = window
        return np.full((60, 120), 971.25)

    monkeypatch.setattr(tw, "mslp_hpa_from_state", fake)
    tw.central_pressure_mb(_fix(extremum_units="m", refined_on=3),
                           _state(), on_refine_grid=True)
    assert seen["window"] is None


def test_an_all_nan_reduction_refuses_rather_than_reporting_nan(monkeypatch):
    monkeypatch.setattr(tw, "mslp_hpa_from_state",
                        lambda *a, **k: np.full((4, 4), np.nan))
    with pytest.raises(TrackRefusal, match="no finite value"):
        tw.central_pressure_mb(_fix(extremum_units="m"), _state(),
                               on_refine_grid=False)


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

def test_peak_wind_is_the_maximum_speed_not_a_component():
    state = SimpleNamespace(physics=SimpleNamespace(fields={
        "u10": np.array([[3.0, 0.0], [0.0, 6.0]]),
        "v10": np.array([[4.0, 5.0], [0.0, 8.0]])}))
    assert tw.peak_wind_m_s(state) == pytest.approx(10.0)


def test_a_driver_without_u10_refuses_by_name():
    with pytest.raises(TrackRefusal, match="u10/v10"):
        tw.peak_wind_m_s(SimpleNamespace(physics=None))


# ---------------------------------------------------------------------------
# Crash discipline
# ---------------------------------------------------------------------------

def test_every_row_is_flushed_so_a_killed_process_loses_nothing(tmp_path):
    """The file is readable, complete and correct WHILE the run is
    alive -- that is the whole crash story, and it is also why the file
    is appended to rather than renamed into place (a rename fails on
    Windows while a reader holds it open, which once killed a 6 h run)."""
    writer = _writer(tmp_path)
    for step in range(1, 4):
        t = 3600.0 * step
        writer.emit(_fix(t=t), t=t, parent_state=_state(),
                    parent_grid=_Grid())
        lines = _lines(tmp_path)               # read it back, still open
        assert len(lines) == step + 1
        assert lines[-1].split(",")[0].startswith("2025-10-24_")
    writer.close()


# ---------------------------------------------------------------------------
# The live tail: a reader watching the file WHILE the run writes it
# ---------------------------------------------------------------------------

def test_the_header_is_on_disk_before_the_first_fix_arrives(tmp_path):
    """A tail attaches when the run starts, which is BEFORE the tracker
    has been consulted once.  If the header were still in the buffer the
    reader would have no column names, and a track file whose names
    arrive late is a file every reader has to guess the shape of."""
    import csv as _csv
    writer = _writer(tmp_path)                   # constructed, never emitted
    with (tmp_path / "track.csv").open(encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        assert reader.fieldnames == list(tw.csv_columns())
        assert list(reader) == []                # names, and no rows yet
    assert (tmp_path / "track.csv").stat().st_size == len(tw.CSV_HEADER) + 1
    writer.close()


def test_a_second_handle_mid_stream_only_ever_sees_complete_rows(tmp_path):
    """The tail contract: a reader that opens the file between two
    emissions parses every visible line with ``csv.DictReader`` and gets
    whole rows -- never a half-written one, never a row missing its
    tail columns.  A UI drawing a track polyline from a torn row plots
    the storm somewhere it never was."""
    import csv as _csv
    writer = _writer(tmp_path)
    names = list(tw.csv_columns())
    for step in range(1, 6):
        t = 1800.0 * step
        writer.emit(_fix(t=t), t=t, parent_state=_state(),
                    parent_grid=_Grid())
        # A SECOND HANDLE, opened fresh while the writer still holds its
        # own -- the tail's own posture, not a peek at our buffer.
        with (tmp_path / "track.csv").open(encoding="utf-8",
                                           newline="") as fh:
            text = fh.read()
            rows = list(_csv.DictReader(text.splitlines(),
                                        restkey="_extra", restval="_short"))
        # Every byte on disk ends a line: no partial row is ever visible.
        assert text.endswith("\n")
        assert len(rows) == step
        for row in rows:
            assert list(row) == names            # no _extra, no missing name
            assert "_extra" not in row and "_short" not in row.values()
            assert row["valid_time"] and float(row["lat_deg"]) == \
                pytest.approx(10.5)
    writer.close()


def test_every_row_write_is_followed_by_a_flush(tmp_path):
    """The policy stated as a count.  Buffering a track file to a 8 KiB
    boundary would hold ~150 rows -- hours of a storm -- so the flush is
    per row and this pins it rather than trusting the docstring."""
    writer = _writer(tmp_path)

    class _Counting:
        def __init__(self, inner):
            self.inner, self.writes, self.flushes = inner, 0, 0

        def write(self, text):
            self.writes += 1
            return self.inner.write(text)

        def flush(self):
            self.flushes += 1
            return self.inner.flush()

        def close(self):
            return self.inner.close()

    writer.stream.handle = counting = _Counting(writer.stream.handle)
    for step in range(1, 4):
        t = 1800.0 * step
        writer.emit(_fix(t=t), t=t, parent_state=_state(),
                    parent_grid=_Grid())
        assert counting.flushes == counting.writes == step
    writer.close()


def test_the_track_handle_is_line_buffered_so_the_flush_is_not_a_habit(
        tmp_path):
    """Tail-safety is a property of the FILE, not of a call site
    remembering to flush.  The handle is line buffered, so a row reaches
    the OS when its newline is written even through a write path that
    forgets -- which is the way this guarantee would otherwise rot the
    next time somebody adds one."""
    import csv as _csv
    writer = _writer(tmp_path)
    assert writer.stream.handle.line_buffering is True
    # A row written straight at the handle, with NO flush after it.
    writer.stream.handle.write(format_row(
        valid_time=dt.datetime(2025, 10, 24, 14, 0, 0), lat=1.0, lon=2.0,
        mslp_mb=3.0, vmax_m_s=4.0) + "\n")
    with (tmp_path / "track.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["valid_time"] == "2025-10-24_14:00:00"
    writer.close()


def test_the_writer_creates_missing_parent_directories(tmp_path):
    TrackWriter(TrackConfig("evidence/deep/track.csv"),
                initial_time=INIT, outdir=tmp_path).close()
    assert (tmp_path / "evidence" / "deep" / "track.csv").is_file()


# ---------------------------------------------------------------------------
# Position-only: a rotation or echo tracker writes the MOVER'S centre
# ---------------------------------------------------------------------------

class _MoverGrid(_Grid):
    """A _Grid that also knows how big it is.

    ``e_we``/``e_sn`` are WRF's STAGGERED counts, so a 101x81 grid has
    100x80 mass cells and its 0-based centre index is (49.5, 39.5).
    Against this fixture's linear ij_to_latlon that puts the centre at
    lat0 + 39.5*step and lon0 + 49.5*step -- numbers that can be written
    down by hand, which is the point of the fixture.
    """

    def __init__(self, e_we=101, e_sn=81, **kw):
        super().__init__(**kw)
        self.e_we, self.e_sn = e_we, e_sn


def _uh_fix(*, found=True, t=3600.0):
    """A fix whose centre is deliberately NOWHERE NEAR the mover's, so a
    row that reported the signal instead of the nest is visible."""
    fix = _fix(found=found, ci=999.0, cj=999.0, t=t)
    fix.field_used = "uh"
    fix.evidence = {"t": t, "extremum_units": "m2 s-2"}
    return fix


@pytest.mark.parametrize("tracked", ["uh", "reflectivity"])
def test_the_header_of_a_rotation_or_echo_tracker_is_position_only(tracked):
    """Three columns, and the two that would be a central pressure and a
    peak wind are ABSENT rather than NaN: they are not missing for this
    tracker, they do not exist for it."""
    header = tw.csv_header(tracked_field=tracked)
    assert header.split(",") == ["valid_time", "lat_deg", "lon_deg"]
    assert "mslp_mb" not in header and "vmax_m_s" not in header
    assert tracked in tw.POSITION_ONLY_FIELDS


def test_a_pressure_tracker_keeps_every_column_it_had():
    """The control for the test above: nothing changed for the shape
    every existing config writes."""
    assert tw.csv_header() == tw.CSV_HEADER
    assert tw.CSV_HEADER.split(",") == list(tw.CSV_BASE_COLUMNS)
    assert tw.csv_header((850.0,), tracked_field="pressure").split(",") == [
        "valid_time", "lat_deg", "lon_deg", "mslp_mb", "vmax_m_s",
        "850_lat_deg", "850_lon_deg", "850_hgt_dam"]


def test_the_grid_centre_is_the_projections_own_centre():
    """THE ESSENTIAL CLAIM, checked against the real projection rather
    than against this file's own arithmetic.

    A ProjectedGrid built without known_x/known_y registers itself
    CENTRED -- known_x = e_we/2 -- so ij_to_latlon at the centre returns
    the reference point exactly.  grid_center_latlon has to land on that
    same point, or the moving-nest centre it reports is offset by a
    fraction of a cell from where the projection thinks the domain is.
    """
    from woof.static.lambert import LambertGrid

    for e_we, e_sn in ((373, 285), (241, 241), (300, 200), (150, 150)):
        grid = LambertGrid(ref_lat=18.0, ref_lon=-76.0, truelat1=18.0,
                           truelat2=18.0, stand_lon=-76.0, dx=4500.0,
                           dy=4500.0, e_we=e_we, e_sn=e_sn)
        lat, lon = tw.grid_center_latlon(grid)
        assert lat == pytest.approx(grid.ref_lat, abs=1e-9)
        assert lon == pytest.approx(grid.ref_lon, abs=1e-9)


def test_the_grid_centre_uses_mass_points_not_staggered_ones():
    """Off by one cell is 643 m on the nest this was built for, so the
    e_we-1 mass count is asserted directly rather than left implied."""
    lat, lon = tw.grid_center_latlon(_MoverGrid(e_we=101, e_sn=81))
    assert lon == pytest.approx(-80.0 + 49.5 * 0.01)
    assert lat == pytest.approx(10.0 + 39.5 * 0.01)
    # An even mass count puts the centre on a cell EDGE, which is a
    # half-index and must not be rounded away.
    lat, lon = tw.grid_center_latlon(_MoverGrid(e_we=100, e_sn=100))
    assert lon == pytest.approx(-80.0 + 49.0 * 0.01)


def test_a_position_only_row_is_the_mover_not_the_signal(tmp_path):
    """The fix points at (999, 999); the row must not."""
    writer = _writer(tmp_path, tracked_field="uh")
    assert writer.position_only
    writer.emit(_uh_fix(), t=3600.0, parent_state=_state(),
                parent_grid=_Grid(), mover_grid=_MoverGrid())
    writer.close()
    lines = _lines(tmp_path)
    assert lines[0] == "valid_time,lat_deg,lon_deg"
    fields = lines[1].split(",")
    assert len(fields) == 3
    assert fields[0] == "2025-10-24_13:00:00"
    assert float(fields[1]) == pytest.approx(10.395, abs=1e-4)
    assert float(fields[2]) == pytest.approx(-79.505, abs=1e-4)


def test_a_position_only_row_follows_the_mover_when_it_moves(tmp_path):
    """relocate_child reassigns child_node.grid, so the row is where the
    nest is NOW.  Both directions, so a frozen position cannot pass."""
    writer = _writer(tmp_path, tracked_field="reflectivity")
    for lon0 in (-80.0, -79.0, -81.0):
        writer.emit(_uh_fix(), t=3600.0, parent_state=_state(),
                    parent_grid=_Grid(),
                    mover_grid=_MoverGrid(lon0=lon0))
    writer.close()
    lons = [line.split(",")[2] for line in _lines(tmp_path)[1:]]
    lons = [float(v) for v in lons]
    assert lons[1] == pytest.approx(lons[0] + 1.0)
    assert lons[2] == pytest.approx(lons[0] - 1.0)


def test_a_position_only_row_needs_no_surface_layer_scheme(tmp_path):
    """The failure this removes: peak_wind_m_s refuses without u10/v10,
    so an idealised convective run would have faulted EVERY row."""
    writer = _writer(tmp_path, tracked_field="uh")
    row = writer.emit(_uh_fix(), t=3600.0,
                      parent_state=SimpleNamespace(physics=None),
                      parent_grid=None, mover_grid=_MoverGrid())
    receipt = writer.close()
    assert row["emitted"] is True
    assert receipt["records"] == 1 and receipt["skipped"] == 0
    assert "faults" not in receipt


def test_a_held_tracker_still_writes_the_nest_position(tmp_path):
    """The nest IS somewhere even when the tracker found nothing, so a
    NaN here would be a wrong answer rather than an absent one.  The hold
    goes on the receipt, where the rest of the provenance lives."""
    writer = _writer(tmp_path, tracked_field="uh")
    row = writer.emit(_uh_fix(found=False), t=3600.0, parent_state=_state(),
                      parent_grid=_Grid(), mover_grid=_MoverGrid())
    writer.close()
    assert row["emitted"] is True and row["no_signal"] is True
    fields = _lines(tmp_path)[1].split(",")
    assert "NaN" not in fields
    assert float(fields[1]) == pytest.approx(10.395, abs=1e-4)


def test_a_position_only_writer_accepts_no_fix_at_all(tmp_path):
    """The row does not read the fix, so the caller is free to skip
    locating on a track-only boundary; the receipt still says so."""
    writer = _writer(tmp_path, tracked_field="uh")
    row = writer.emit(None, t=3600.0, parent_state=None,
                      mover_grid=_MoverGrid())
    writer.close()
    assert row["emitted"] is True and row["no_signal"] is True
    assert len(_lines(tmp_path)[1].split(",")) == 3


def test_a_pressure_writer_still_NaNs_a_no_signal_consultation(tmp_path):
    """The control: the visible-gap rule is unchanged where the column
    genuinely is a storm quantity that can be missing."""
    writer = _writer(tmp_path)
    writer.emit(_fix(found=False), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    writer.close()
    assert _lines(tmp_path)[1].split(",")[1:] == ["NaN"] * 4


def test_a_position_only_row_without_a_mover_grid_is_a_receipt_fault(
        tmp_path):
    """Never raises, like every other fault in this module."""
    writer = _writer(tmp_path, tracked_field="uh")
    row = writer.emit(_uh_fix(), t=3600.0, parent_state=_state(),
                      parent_grid=_Grid(), mover_grid=None)
    receipt = writer.close()
    assert row["emitted"] is False
    assert "moving domain" in row["reason"]
    assert receipt["skipped"] == 1 and receipt["records"] == 0


def test_the_position_only_row_never_needs_quoting_either(tmp_path):
    """RFC 4180 by construction, so splitting on commas stays a correct
    parser for every shape this module writes."""
    writer = _writer(tmp_path, tracked_field="uh")
    writer.emit(_uh_fix(), t=3600.0, parent_state=_state(),
                parent_grid=_Grid(), mover_grid=_MoverGrid())
    writer.close()
    text = (tmp_path / "track.csv").read_text(encoding="utf-8")
    assert '"' not in text
    for line in text.splitlines():
        assert len(line.split(",")) == 3


def test_format_row_position_only_has_no_padding_and_three_fields():
    line = format_row(valid_time=dt.datetime(2025, 10, 24, 18, 30, 0),
                      lat=14.5321, lon=-73.1044, position_only=True)
    assert line == "2025-10-24_18:30:00,14.5321,-73.1044"


def test_the_route_takes_the_column_set_from_the_follow_block(tmp_path):
    """[relocation.track] configures WHERE, HOW OFTEN and -- with
    output_level -- WHICH OF the blocks the tracker computes.  What it
    cannot do is ADD one: the available set comes from
    [relocation.follow], so the file and the nest can never be
    describing different trackers.  The refusal for a block the run does
    not produce is in ``test_relocation_track_config.py``."""
    from types import SimpleNamespace as NS

    from woof.core.storm_tracking import build_follow_config, levels_of
    from woof.runtime import build_track_writer

    assert set(tw.TRACK_KEYS) == {"path", "interval_seconds", "output_level", "storm_id"}
    cases = {
        "uh": {"field": "uh", "threshold": 25.0, "fallback_threshold": 40.0},
        "reflectivity": {"field": "reflectivity", "threshold": 40.0},
        "pressure": {"field": "pressure", "threshold": 30.0},
    }
    for name, over in cases.items():
        follow = build_follow_config(
            {"search_margin_cells": 15, "min_shift_cells": 2,
             "max_shift_cells": 8, "cooldown_seconds": 900.0, **over}, "x")
        exp = NS(start_time=INIT,
                 relocation=NS(track=TrackConfig(f"{name}.csv"),
                               follow=follow))
        writer = build_track_writer(exp, tmp_path)
        assert writer.tracked_field == name
        assert writer.position_only is (name in tw.POSITION_ONLY_FIELDS)
        writer.close()
        header = _lines(tmp_path, f"{name}.csv")[0]
        assert header == tw.csv_header(levels_of(follow),
                                       tracked_field=name)


# ---------------------------------------------------------------------------
# output_level: choosing which blocks the file carries
# ---------------------------------------------------------------------------

_LEVELS = (925.0, 850.0, 700.0, 500.0)


def _leveled_fix(levels=_LEVELS, **over):
    """``_fix`` with a per-level answer for each surface, which the base
    fixture has no reason to carry -- and without which every level
    column faults instead of being written."""
    fix = _fix(**over)
    fix.levels = [SimpleNamespace(level_hpa=v, fix_ij=(20.0 + i, 30.0 + i),
                                  height_dam=100.0 + 10 * i)
                  for i, v in enumerate(levels)]
    return fix


def _ol_writer(tmp_path, output_level, levels=_LEVELS):
    return TrackWriter(TrackConfig("track.csv", output_level=output_level),
                       initial_time=INIT, outdir=tmp_path, levels=levels)


@pytest.mark.parametrize("output_level,want", [
    (None, ["valid_time", "lat_deg", "lon_deg", "mslp_mb", "vmax_m_s",
            "925_lat_deg", "925_lon_deg", "925_hgt_dam",
            "850_lat_deg", "850_lon_deg", "850_hgt_dam",
            "700_lat_deg", "700_lon_deg", "700_hgt_dam",
            "500_lat_deg", "500_lon_deg", "500_hgt_dam"]),
    ((0.0,), ["valid_time", "lat_deg", "lon_deg", "mslp_mb", "vmax_m_s"]),
    ((850.0,), ["valid_time", "850_lat_deg", "850_lon_deg", "850_hgt_dam"]),
    ((0.0, 850.0), ["valid_time", "lat_deg", "lon_deg", "mslp_mb",
                    "vmax_m_s", "850_lat_deg", "850_lon_deg", "850_hgt_dam"]),
])
def test_output_level_picks_the_header(tmp_path, output_level, want):
    writer = _ol_writer(tmp_path, output_level)
    writer.close()
    assert _lines(tmp_path)[0].split(",") == want


def test_output_level_names_the_order_too(tmp_path):
    """It selects the blocks AND their sequence, which is the rule with
    no surprises in it: a profile written 925, 900, ... 300 reads down
    the page in pressure order, where the tracker's own order would put
    the steering surfaces first and the rest behind them."""
    full = tw.csv_columns(_LEVELS)
    for picked in ((0.0,), (850.0,), (500.0, 850.0), (0.0, 500.0, 925.0)):
        got = tw.csv_columns(_LEVELS, output_level=picked)
        assert set(got) <= set(full)          # still only ever removes
    # 500 named before 850 now COMES OUT that way.
    assert tw.csv_columns(_LEVELS, output_level=(500.0, 850.0)) == (
        "valid_time", "500_lat_deg", "500_lon_deg", "500_hgt_dam",
        "850_lat_deg", "850_lon_deg", "850_hgt_dam")
    # ...and the surface block LEADS wherever it was named, because a row
    # is "the storm, then the profile".
    assert tw.csv_columns(_LEVELS, output_level=(700.0, 0.0, 925.0)) == (
        "valid_time", "lat_deg", "lon_deg", "mslp_mb", "vmax_m_s",
        "700_lat_deg", "700_lon_deg", "700_hgt_dam",
        "925_lat_deg", "925_lon_deg", "925_hgt_dam")


def test_without_output_level_the_order_is_unchanged():
    """A config that does not use the key gets exactly the file it always
    got: the surface block, then the tracked surfaces in level_hpa's own
    order."""
    assert tw.csv_columns(_LEVELS) == tw.csv_columns(
        _LEVELS, output_level=None)
    assert tw.csv_columns(_LEVELS)[:5] == tw.CSV_BASE_COLUMNS
    assert tw.csv_columns(_LEVELS)[5] == "925_lat_deg"


def test_the_header_and_the_rows_come_from_one_function(tmp_path):
    """The failure this forbids is a file whose header says one order and
    whose rows use another -- worse than no file, because every column
    reads plausibly."""
    picked = (700.0, 0.0, 925.0)
    writer = _ol_writer(tmp_path, picked)
    writer.emit(_leveled_fix(), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    receipt = writer.close()
    header, row = _lines(tmp_path)
    assert receipt["skipped"] == 0
    assert header.split(",") == list(tw.csv_columns(_LEVELS,
                                                    output_level=picked))
    assert len(row.split(",")) == len(header.split(","))
    # 700 leads the profile, and its height is the fixture's own 700
    # answer (110 dam is 850's; 700 is the third level, 120 dam).
    cells = dict(zip(header.split(","), row.split(",")))
    assert cells["700_hgt_dam"] == "120.00"
    assert cells["925_hgt_dam"] == "100.00"


def test_a_row_carries_exactly_the_columns_the_header_named(tmp_path):
    """The one way this could corrupt a file: a header and a row that
    disagree about which blocks are present."""
    for output_level in (None, (0.0,), (850.0,), (0.0, 700.0)):
        writer = _ol_writer(tmp_path, output_level)
        row = writer.emit(_leveled_fix(), t=3600.0, parent_state=_state(),
                          parent_grid=_Grid())
        # ...and a no-signal row, which builds its cells separately
        writer.emit(_leveled_fix(found=False), t=7200.0,
                    parent_state=_state(), parent_grid=_Grid())
        receipt = writer.close()
        lines = _lines(tmp_path)
        # Both rows were WRITTEN, or the width check below is vacuous:
        # a faulted row writes nothing, and one line is trivially one
        # width.
        assert row["emitted"] is True, (output_level, row)
        assert receipt["skipped"] == 0, (output_level, receipt)
        assert len(lines) == 3, (output_level, lines)
        widths = {len(line.split(",")) for line in lines}
        assert len(widths) == 1, (output_level, lines)


def test_dropping_the_surface_block_drops_the_wind_not_just_its_column(
        tmp_path):
    """output_level = 850 asks for a surface, and a surface has no peak
    10-m wind in it -- so the file must not carry one under another
    name."""
    writer = _ol_writer(tmp_path, (850.0,))
    # No surface-layer scheme at all: peak_wind_m_s would refuse, and a
    # column the file does not carry must not be able to fault a row.
    writer.emit(_leveled_fix(), t=3600.0,
                parent_state=SimpleNamespace(physics=None),
                parent_grid=_Grid())
    receipt = writer.close()
    header, row = _lines(tmp_path)
    assert receipt["skipped"] == 0 and "faults" not in receipt
    assert "vmax_m_s" not in header and "mslp_mb" not in header
    cells = row.split(",")
    assert len(cells) == 4
    # ...and what IS there is 850's own answer -- height 110 dam from the
    # fixture -- not the deep-layer mean shifted into its place.
    assert cells[3] == "110.00"


def test_a_level_the_file_skips_is_still_tracked(tmp_path):
    """output_level chooses what the FILE carries, never what the tracker
    watches: the nest is still steered by every configured surface, and
    every one still lands on the receipt."""
    writer = _ol_writer(tmp_path, (850.0,))
    assert writer.levels == _LEVELS               # all four, unchanged
    assert writer.emitted_levels == (850.0,)      # one printed
    writer.close()


def test_the_receipt_names_the_columns_it_wrote(tmp_path):
    """A file whose shape is a config decision should be readable from
    the receipt without opening it."""
    writer = _ol_writer(tmp_path, (0.0, 850.0))
    writer.emit(_leveled_fix(), t=3600.0, parent_state=_state(),
                parent_grid=_Grid())
    receipt = writer.close()
    assert receipt["columns"] == _lines(tmp_path)[0].split(",")


def test_output_level_is_inert_on_a_position_only_writer(tmp_path):
    """Admission refuses the combination (test_relocation_track_config),
    so this is only the belt: a writer handed one anyway still writes the
    three columns that shape has, rather than a fourth shape."""
    writer = TrackWriter(TrackConfig("track.csv", output_level=(0.0,)),
                         initial_time=INIT, outdir=tmp_path,
                         tracked_field="uh")
    writer.close()
    assert _lines(tmp_path)[0] == "valid_time,lat_deg,lon_deg"


# ---------------------------------------------------------------------------
# It may never destroy rows it did not write
# ---------------------------------------------------------------------------

def test_a_stale_file_in_the_runs_own_directory_is_still_truncated(tmp_path):
    """Unchanged, and deliberately so: a fresh output directory is a
    fresh file (the runner refuses an --outdir that already holds a run,
    before any config is read, so it binds every run there is), and a run
    starts its own file rather than appending to a stale one."""
    (tmp_path / "track.csv").write_text("stale garbage\n", encoding="utf-8")
    _writer(tmp_path).close()
    assert _lines(tmp_path) == [tw.CSV_HEADER]


def test_an_absolute_path_holding_another_runs_rows_refuses(tmp_path):
    """THE DATA-LOSS PATH: two legs of one forecast pointed at one
    absolute deck, and the second leg opens it for write.  What proves it
    is that the first leg's bytes are STILL THERE afterwards, not merely
    that something raised."""
    deck = tmp_path / "decks" / "melissa.csv"
    deck.parent.mkdir(parents=True)
    first_leg = (tw.CSV_HEADER
                 + "\n2025-10-24_12:00:00,14.5000,-73.1000,964.40,48.00\n")
    deck.write_text(first_leg, encoding="utf-8")
    with pytest.raises(TrackRefusal) as err:
        TrackWriter(TrackConfig(str(deck)), initial_time=INIT,
                    outdir=tmp_path / "leg2")
    assert "erase rows this run did not write" in str(err.value)
    assert deck.read_text(encoding="utf-8") == first_leg


def test_an_absolute_path_that_does_not_exist_yet_is_fine(tmp_path):
    """The control: absolute paths are honoured as given, and only a
    NON-EMPTY one outside the run's directory is refused."""
    target = tmp_path / "elsewhere" / "t.csv"
    TrackWriter(TrackConfig(str(target)), initial_time=INIT,
                outdir=tmp_path / "outdir").close()
    assert target.is_file()


def test_an_empty_absolute_file_is_not_somebody_elses_rows(tmp_path):
    """Zero bytes is nothing to lose, and pre-creating a path is a normal
    way to reserve one."""
    target = tmp_path / "elsewhere" / "t.csv"
    target.parent.mkdir(parents=True)
    target.touch()
    TrackWriter(TrackConfig(str(target)), initial_time=INIT,
                outdir=tmp_path / "outdir").close()
    assert target.read_text(encoding="utf-8").strip() == tw.CSV_HEADER


def test_an_absolute_path_inside_the_runs_own_directory_is_ours(tmp_path):
    """Spelled absolutely but pointing where the run already writes: the
    guard is about WHOSE rows they are, not how the path was typed."""
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    target = outdir / "track.csv"
    target.write_text("stale\n", encoding="utf-8")
    TrackWriter(TrackConfig(str(target)), initial_time=INIT,
                outdir=outdir).close()
    assert target.read_text(encoding="utf-8").strip() == tw.CSV_HEADER
