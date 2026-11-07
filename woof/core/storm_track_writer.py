"""Write the vortex track out: one CSV per run.

:mod:`woof.core.storm_tracking` already knows where the storm is -- it
has to, to steer the nest -- but that answer only ever reached the
relocation receipts, as parent grid cells, at the relocation cadence.
This turns the SAME fix into a file a human can plot and a script can
parse, without a second centre-finder that could drift from the first.

ONE SOURCE OF TRUTH.  Every row comes from
:meth:`woof.core.storm_tracking.StormTracker.locate`, which is the
first half of ``desired_shift`` with the hysteresis removed.  A run's
track and its nest therefore cannot disagree about where the vortex was:
they are the same computation.  ``locate`` is stateless and silent (no
receipt, no cooldown, no proposal), which is what lets this file be
written oftener than the nest moves -- proven byte-for-byte in
``tests/test_storm_tracking_locate.py`` and end to end against a control
forecast in ``MOVING-NESTS-FINALIZE.md``.

THE FORMAT IS CSV: the column names on line one and nothing above them,
then one row per fix.  ``pandas.read_csv(path)`` and
``csv.DictReader(open(path))`` therefore both work with no arguments --
no ``comment=``, no ``skiprows=``, no width table.  Decimal degrees,
millibars, metres per second, and the valid time spelled the way wrfout
spells it, so a row joins onto a history frame by string equality.
Anything richer -- a deck format, a database -- is post-processing, and
post-processing is better done by the tool that needs it than guessed at
here.

TWO SHAPES, DECIDED BY THE TRACKER'S FIELD, and the header says which
one a reader is holding:

* ``field = "pressure"`` -- a vortex deck.  Where the storm is, how deep
  it is and how hard it is blowing (``valid_time``, ``lat_deg``,
  ``lon_deg``, ``mslp_mb``, ``vmax_m_s``), plus a ``(lat, lon, height)``
  triple per tracked isobaric surface.

* ``field = "uh"`` / ``"reflectivity"`` -- POSITION ONLY: ``valid_time``,
  ``lat_deg``, ``lon_deg``, and those are the MOVING DOMAIN'S OWN CENTRE
  rather than the signal's.  A rotation or echo tracker follows
  convection, which has no central pressure and no storm-scale peak
  wind, and the useful thing to plot for it is the footprint that
  actually integrated.  See POSITION_ONLY_FIELDS.

NO FIELD EVER NEEDS QUOTING.  Every column is a number or the timestamp
``2025-10-24_12:00:00``; no comma, no quote and no newline can appear in
any column of any row.  So the file is valid RFC 4180 *and*
``line.split(",")`` is a correct parser for it -- the property a one-line
awk or shell reader depends on, and the reason nothing here imports
:mod:`csv` to write it.

TAIL-SAFE WHILE IT IS BEING WRITTEN.  The file is LINE BUFFERED and
flushed after every row, so a reader that opens it mid-run -- ``tail
-f``, a live plot, a UI drawing the track as the storm moves -- always
sees complete rows: every consultation that has happened is on disk,
and no half-written row is ever visible, because a row reaches the
operating system the instant its newline is written.  That is what lets
``csv.DictReader`` on a second handle be correct AT ANY MOMENT rather
than only once the run has closed the file.  The buffering is the
guarantee and the explicit flush is the statement of it: neither is
allowed to be the only one, because a write path added later that
forgets to flush would otherwise hold rows back to an 8 KiB boundary --
roughly 150 rows, hours of a storm -- and the live tail would stall
without anything looking broken.

WHAT IS NOT PROMISED IS ``fsync``.  Rows are handed to the OS, never
forced onto the platter.  A KILLED PROCESS therefore loses nothing --
the kernel already holds every row -- while a POWER CUT may lose the
last rows the disk had not yet written.  That is the right trade for a
diagnostic that runs beside a forecast: an fsync per row would put a
device flush on the model's own thread at the relocation cadence, to
protect a file whose next row is due in minutes and whose contents are
recomputable from the run.  Durability across a power loss is the
restart file's job, and the forecast already pays for that one.

THE PROVENANCE BANNER IS GONE, deliberately.  A ``# woof storm track``
line above the names is exactly what forces every reader to pass
``comment="#"``, and the run's receipt already carries the path, the
record count and the contract version.  The column names are
identification enough.

WHAT IT IS NOT.  It is not the GFDL vortex tracker and does not
reproduce its centre-finding; it is WOOF's own tracker, reported.  WRF
writes no track file at all -- its moving nest prints a centre to
``rsl.out``, and the decks in that ecosystem come from the GFDL tracker
as post-processing -- so there is no Fortran oracle behind this output
and it must not borrow credibility from the bitwise claims elsewhere in
this tree.  Its evidence is structural: the columns and units are what
the header says, the clock is the valid time, and a row is either accurate
or explicitly absent.

CONFIG.  ``[relocation.track]`` under ``[relocation]``: ``path``, and
optional ``interval_seconds`` and ``output_level``.  Absent, no file is
written and the run behaves exactly as it did before this module
existed.

``output_level`` names the BLOCKS the file carries -- ``SURFACE_LEVEL``
(0) for the centre with its central pressure and peak wind, and any
isobaric surface in hPa for a ``(lat, lon, height)`` triple -- and the
ORDER it carries them in.  It is INDEPENDENT of
``[relocation.follow]``'s ``level_hpa``: what steers a nest and what a
deck reports are different questions, so a run tracking 850/700/500 may
report the vortex every 50 hPa without adding a single surface to the
steering vote.

A surface named here that ``level_hpa`` does not track becomes a
REPORT-ONLY surface on the follow config
(:attr:`woof.core.storm_tracking.FollowConfig.report_level_hpa`,
derived in :mod:`woof.experiment`, the one place that sees both
blocks).  The TRACKER then computes it -- same plane, same centre
search, its own ``LevelFix`` -- and leaves it out of the steering mean,
so this writer still reads every column off one fix and never runs a
second search of its own.

Absent, the file carries the surface block plus every TRACKED surface in
the tracker's own order, which is what this module has always written.
The tracked ``field`` still decides the column SHAPE: a rotation or echo
tracker is position-only (``POSITION_ONLY_FIELDS``) and refuses
``output_level`` at admission, because there is nothing to choose
between.

NEVER FAILS A FORECAST.  Everything checkable is checked at config load
(:func:`woof.experiment._build_relocation`), before a single step runs.
Anything that still goes wrong per-record is caught, counted, and
reported on the run's receipt -- the same posture the tracker takes
toward a signal it cannot find.  A diagnostic that can kill a 12-hour
forecast is a defect.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from woof.core.storm_tracking import mslp_hpa_from_state

#: Versioned label carried by every receipt row this module emits.
TRACK_CONTRACT = "gpuwm-storm-track.v1"

#: Keys of ``[relocation.track]``.  ``path`` is required; unknown keys
#: refuse by name, as everywhere else in this config surface.
TRACK_KEYS = frozenset({"path", "interval_seconds", "output_level", "storm_id"})

#: ``output_level`` for the SURFACE block -- the deep-layer-mean centre,
#: the central pressure and the peak wind.  Zero is the same spelling
#: :data:`woof.core.storm_tracking.SEA_LEVEL_HPA` uses for the tracker
#: that works there, so one number means one surface across the whole
#: ``[relocation]`` surface, and it can never collide with a real
#: isobaric choice (the admissible band starts at 200 hPa).
SURFACE_LEVEL = 0.0

#: The header, written once at the top of every file.  Named units, so
#: the file is self-describing and a reader never has to guess whether a
#: longitude is signed or a pressure is Pa.
CSV_BASE_COLUMNS = ("valid_time", "lat_deg", "lon_deg", "mslp_mb",
                    "vmax_m_s")

#: Tracker fields whose track file is POSITION ONLY: a clock and the
#: moving nest's own centre, and nothing else.
#:
#: ``mslp_mb`` and ``vmax_m_s`` are tropical-cyclone quantities and they
#: do not survive the move to a rotation or echo tracker.  A supercell
#: has no central pressure -- the sea-level minimum over its
#: neighbourhood is the ambient synoptic low, which is somewhere else
#: entirely and moves on its own -- and ``vmax_m_s`` is the strongest
#: 10-m wind ANYWHERE on the grid, which for a convective case is
#: whatever the domain happens to contain rather than a property of the
#: storm being followed.  Two columns of plausible-looking numbers of
#: the wrong quantity is worse than two columns that are not there.
#:
#: Dropping them also removes a way the file could fail: ``peak_wind_m_s``
#: refuses when no surface-layer scheme published u10/v10, and an
#: idealised convective run without one would have faulted EVERY row.
POSITION_ONLY_FIELDS = ("uh", "reflectivity", "attribute")

#: The whole header for such a run.  ``lat_deg``/``lon_deg`` keep their
#: names so one plotter reads every track file this module writes; what
#: they hold is stated in POSITION_ONLY_FIELDS and on the run's receipt,
#: where the rest of the provenance already lives.
CSV_POSITION_COLUMNS = ("valid_time", "lat_deg", "lon_deg")

#: What each tracked isobaric surface adds.  Named BY THE SURFACE, so
#: `850_lat_deg` reads as what it is and can never be confused with the
#: deep-layer mean in the base columns -- which is the whole risk of
#: putting several centres in one row.
CSV_LEVEL_SUFFIXES = ("lat_deg", "lon_deg", "hgt_dam")


def wanted_levels(output_level=None):
    """``output_level`` as a lookup, or ``None`` for "the tracked set".

    Rounded, because a level is a config number that has been through
    ``float`` twice and 850.0 must match 850 however it was spelled.
    """
    if output_level is None:
        return None
    return {round(float(v), 6) for v in output_level}


def block_order(levels=(), output_level=None) -> tuple:
    """The blocks a row carries, in the order it carries them.

    ``SURFACE_LEVEL`` (0) stands for the surface block -- the
    deep-layer-mean centre with the central pressure and peak wind -- and
    every other entry is an isobaric surface with its own (lat, lon,
    height) triple.  One function, so the header and the row cannot
    disagree about either membership or order; a file whose header and
    rows disagree is worse than no file.
    """
    wanted = wanted_levels(output_level)
    if wanted is None:
        return (SURFACE_LEVEL,) + tuple(float(v) for v in levels)
    available = {round(float(SURFACE_LEVEL), 6): float(SURFACE_LEVEL)}
    for level in levels:
        available.setdefault(round(float(level), 6), float(level))
    out, seen = [], set()
    for value in output_level:
        key = round(float(value), 6)
        if key in seen or key not in available:
            continue
        seen.add(key)
        out.append(available[key])
    # THE SURFACE BLOCK LEADS, wherever output_level happened to name it.
    # It is the headline of a storm deck -- the centre, the central
    # pressure and the peak wind -- and a row is laid out as "the storm,
    # then the profile".  The isobaric surfaces keep output_level's own
    # order, which is the part a reader actually chose.
    surface = round(float(SURFACE_LEVEL), 6)
    if surface in seen:
        out = ([float(SURFACE_LEVEL)]
               + [v for v in out if round(v, 6) != surface])
    return tuple(out)


def csv_columns(levels=(), *, tracked_field: str = "pressure",
                output_level=None) -> tuple:
    """Every column name, in order, for a tracker watching ``levels``.

    THE COLUMN SET IS A FUNCTION OF THE CONFIG, never of what a
    consultation found -- the same rule ``_level_columns`` follows for a
    surface that declined.  A rotation or echo tracker
    (``POSITION_ONLY_FIELDS``) has three columns and a pressure tracker
    has five plus three per surface, and which one a reader is holding
    is answered by the header alone.

    ``output_level`` selects which BLOCKS survive: ``SURFACE_LEVEL`` (0)
    is the deep-layer-mean centre with the central pressure and peak
    wind, and any surface the run computes -- steering or report-only --
    is its own ``(lat, lon, height)`` triple.  ``None`` is every block,
    which is the file as it was before the key existed.

    ``output_level`` NAMES THE ORDER TOO.  The blocks appear in the
    order it lists them, which is the rule with no surprises in it: a
    twenty-surface profile written 925, 900, ... 300 reads down the page
    in pressure order, where the tracker's own order would have put the
    three steering surfaces first and the rest behind them.

    Without ``output_level`` the order is the surface block then the
    tracked surfaces in ``level_hpa``'s own order, unchanged -- so a
    config that does not use the key gets exactly the file it always
    got.
    """
    if tracked_field in POSITION_ONLY_FIELDS:
        return tuple(CSV_POSITION_COLUMNS)
    out = [CSV_BASE_COLUMNS[0]]                     # the clock is not a block
    for level in block_order(levels, output_level):
        if level == SURFACE_LEVEL:
            out += list(CSV_BASE_COLUMNS[1:])
        else:
            out += [f"{level:g}_{suffix}"
                    for suffix in CSV_LEVEL_SUFFIXES]
    return tuple(out)


def csv_header(levels=(), *, tracked_field: str = "pressure",
               output_level=None) -> str:
    """The CSV header line, naming every column in order -- so a reader
    parses the file without knowing the config that produced it, and
    ``read_csv`` finds the names where it already looks for them."""
    return ",".join(csv_columns(levels, tracked_field=tracked_field,
                                output_level=output_level))


#: The header for a tracker with no isobaric surfaces: the shape of
#: every file written before multi-level tracking existed.
CSV_HEADER = csv_header()

#: What a data column holds when the tracker found nothing.  ``NaN``
#: rather than an empty field or a sentinel number: it is in pandas'
#: default NA set, every plotting library already drops it, and no real
#: value can be confused with it.  An EMPTY FIELD is the other CSV
#: convention and is worse here -- it reads as the string "" in the
#: stdlib reader and as NA in pandas, so the two disagree about a row
#: this file has to be unambiguous about.
CSV_MISSING = "NaN"

#: ``strftime`` for the valid-time column.  This is WRF's own ``Times``
#: spelling and the one in a wrfout filename, so a track row joins onto
#: a history frame by string equality rather than by parsing.
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


class TrackRefusal(ValueError):
    """A track request this module will not serve quietly."""


# ---------------------------------------------------------------------------
# Config: [relocation.track]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackConfig:
    """The validated ``[relocation.track]`` block.

    Two keys, one of them optional, because that is all the format
    needs.  ``path`` is relative to the run's output directory unless it
    is absolute.  ``interval_seconds`` is model seconds between rows;
    absent means every tracker consultation, which is the relocation
    cadence.
    """

    path: str
    interval_seconds: float | None = None
    #: Which BLOCKS the file carries: ``SURFACE_LEVEL`` (0) and/or any
    #: isobaric surface in hPa, TRACKED OR NOT.  ``None`` -- the default
    #: -- is the surface block plus every tracked surface, which is the
    #: file this module wrote before the key existed.  Normalised to a
    #: tuple by __post_init__ so every consumer sees one shape, and
    #: reported in the order the config names it.
    output_level: "float | tuple[float, ...] | None" = None
    storm_id: str | None = None

    def __post_init__(self) -> None:
        if self.storm_id is not None:
            import re
            if not isinstance(self.storm_id, str) or not re.fullmatch(r"[A-Z]{2}[0-9]{6}", self.storm_id):
                raise ValueError("storm_id must be an ATCF basin, number and year (BBNNYYYY)")
        if not str(self.path).strip():
            raise ValueError(
                "[relocation.track] path must name a file to write the "
                "track to; an empty path writes nowhere")
        if self.output_level is not None:
            raw = (self.output_level
                   if isinstance(self.output_level, (list, tuple))
                   else [self.output_level])
            if not raw:
                raise ValueError(
                    "[relocation.track] output_level is an empty list, "
                    "which asks for a file with a clock and no data. Name "
                    f"at least one block ({SURFACE_LEVEL:g} for the "
                    "surface, or a tracked surface in hPa), or delete the "
                    "key for all of them")
            picked = tuple(float(v) for v in raw)
            if len(set(picked)) != len(picked):
                raise ValueError(
                    f"[relocation.track] output_level = {list(picked)} "
                    "repeats a block; each one is written once, so a "
                    "duplicate would ask for the same columns twice")
            for level in picked:
                if not math.isfinite(level) or level < 0.0:
                    raise ValueError(
                        f"[relocation.track] output_level = {level!r} is "
                        f"not a block; it is {SURFACE_LEVEL:g} for the "
                        "surface or an isobaric surface in hPa")
            object.__setattr__(self, "output_level", picked)
        if self.interval_seconds is not None:
            if (not math.isfinite(float(self.interval_seconds))
                    or float(self.interval_seconds) <= 0.0):
                raise ValueError(
                    "[relocation.track] interval_seconds = "
                    f"{self.interval_seconds!r} must be a finite, positive "
                    "number of MODEL SECONDS between rows")

    def to_json(self) -> dict[str, object]:
        out: dict[str, object] = {"contract": TRACK_CONTRACT,
                                  "path": str(self.path)}
        if self.interval_seconds is not None:
            out["interval_seconds"] = float(self.interval_seconds)
        if self.output_level is not None:
            out["output_level"] = [float(v) for v in self.output_level]
        if self.storm_id is not None:
            out["storm_id"] = self.storm_id
        return out


def build_track_config(table, source: str) -> TrackConfig:
    """Validate a parsed ``[relocation.track]`` table.

    Honored or refused, never ignored: ``path`` is required by name and
    every unknown key is refused by name, so a misspelt key cannot leave
    a run quietly writing a track nobody configured.
    """
    from woof.experiment import did_you_mean

    if not isinstance(table, dict):
        raise ValueError(
            f"[relocation.track] of {source} must be a table with a path "
            f"key, got {type(table).__name__}. It is a single table, not "
            "an array-of-tables: a run writes one track file, because "
            "there is one vortex.")
    unknown = sorted(set(table) - TRACK_KEYS)
    if unknown:
        named = ", ".join(
            f"{key!r}{did_you_mean(key, TRACK_KEYS)}" for key in unknown)
        raise ValueError(
            f"[relocation.track] of {source} does not have key(s) {named}; "
            f"it has {sorted(TRACK_KEYS)}. No key is ignored, because a "
            "dropped key writes a file nobody asked for.")
    if "path" not in table:
        raise ValueError(
            f"[relocation.track] of {source} is missing required key "
            "'path'; a track has to be written somewhere")
    path = table["path"]
    if not isinstance(path, str):
        raise ValueError(
            f"path in [relocation.track] of {source} must be a string, "
            f"got {path!r}")
    interval = None
    if "interval_seconds" in table:
        interval = table["interval_seconds"]
        if isinstance(interval, bool) or not isinstance(
                interval, (int, float)):
            raise ValueError(
                f"interval_seconds in [relocation.track] of {source} must "
                f"be a number of model seconds, got {interval!r}")
        interval = float(interval)
    output_level = None
    if "output_level" in table:
        raw = table["output_level"]
        seq = raw if isinstance(raw, (list, tuple)) else [raw]
        for index, item in enumerate(seq):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                where = (f"output_level[{index}]"
                         if isinstance(raw, (list, tuple)) else "output_level")
                raise ValueError(
                    f"{where} in [relocation.track] of {source} must be a "
                    f"block -- {SURFACE_LEVEL:g} for the surface, or a "
                    f"tracked surface in hPa -- got {item!r}")
        output_level = tuple(float(v) for v in seq)
    try:
        return TrackConfig(path=path, interval_seconds=interval,
                           output_level=output_level, storm_id=table.get("storm_id"))
    except ValueError as err:
        raise ValueError(f"[relocation.track] of {source}: {err}") from None


# ---------------------------------------------------------------------------
# The quantities a row carries
# ---------------------------------------------------------------------------

def latlon_from_grid(grid, ci: float, cj: float) -> tuple[float, float]:
    """0-based cell index on ``grid`` -> (lat, lon) in decimal degrees.

    ``ProjectedGrid.ij_to_latlon`` takes 1-BASED mass-point coordinates
    (a nest is constructed with ``known_x = known_y = 1.0``, its own
    first mass point), while every index in
    :mod:`woof.core.storm_tracking` is a 0-based array index.  The
    ``+1`` is that conversion and it is the only place it happens.

    Deliberately NOT composed up a placement chain.  ``NestFootprint``
    and :meth:`ProjectedGrid.nest` use different half-cell conventions
    for where a child's first mass point sits in its parent -- they agree
    on whole-cell SHIFTS, which is all the tracker needs, but they differ
    by a fraction of a cell in absolute position.  Asking the grid that
    found the centre, in that grid's own index space, avoids the question
    entirely.  (Measured: a hand-composed chain disagreed with the
    wrfout's own XLAT/XLONG by 0.08 degrees; this route agrees to
    0.18-0.38 km, which is sub-cell at 643 m.)
    """
    lat, lon = grid.ij_to_latlon(float(ci) + 1.0, float(cj) + 1.0)
    return float(lat), float(lon)


def grid_center_latlon(grid) -> tuple[float, float]:
    """The geometric centre of ``grid``, in decimal degrees.

    ``e_we``/``e_sn`` are WRF's STAGGERED dimensions, so the mass grid is
    ``e_we - 1`` by ``e_sn - 1`` and its 0-based centre index is
    ``(e_we - 2) / 2``.  :func:`latlon_from_grid` then adds the 1 that
    takes it to the 1-based mass coordinate ``ij_to_latlon`` wants, which
    lands on ``e_we / 2`` -- exactly the ``known_x`` a
    :class:`~woof.static.projection.ProjectedGrid` gives itself when it
    is built centred.  So this agrees with the projection's own idea of
    where the middle of a domain is by construction rather than by
    convention, and it is checked against it in
    ``tests/test_storm_track_writer.py``.

    Asked of the MOVER'S OWN grid, in the mover's own index space, for
    the reason :func:`latlon_from_grid` gives: a placement chain composed
    by hand disagrees with the wrfout's own XLAT/XLONG by a fraction of a
    cell, and there is no need to compose one.  The grid is rebuilt on
    every relocation (``nest_relocation.relocate_child`` reassigns
    ``child_node.grid``), so this is where the nest is NOW, not where it
    started.
    """
    e_we, e_sn = int(grid.e_we), int(grid.e_sn)
    return latlon_from_grid(grid, (e_we - 2) / 2.0, (e_sn - 2) / 2.0)


def peak_wind_m_s(state) -> float:
    """The strongest 10-m wind on this domain, m/s.

    ``u10``/``v10`` live on the physics driver's field table, not on the
    state: ``woof.core.physics`` allocates every ``SFCLAY_OUTPUTS`` name
    unconditionally and fills them only when a surface-layer scheme runs.
    So the fields always EXIST and are identically zero under
    ``sf_sfclay_physics = 0`` -- which is why the refusal for that case
    is at config load, keyed on the selector, and not here on a missing
    key.  A wind of 0 m/s is a plausible-looking number of the wrong
    quantity, and this module does not emit those.
    """
    physics = getattr(state, "physics", None)
    fields = getattr(physics, "fields", None)
    if not fields or "u10" not in fields or "v10" not in fields:
        raise TrackRefusal(
            "this domain's physics driver publishes no u10/v10, so the "
            "peak 10-m wind cannot be read; a surface-layer scheme "
            "(sf_sfclay_physics) is what produces them")
    u10, v10 = fields["u10"], fields["v10"]
    xp = np if isinstance(u10, np.ndarray) else __import__("cupy")
    speed = xp.sqrt(xp.asarray(u10, dtype=xp.float64) ** 2
                    + xp.asarray(v10, dtype=xp.float64) ** 2)
    return float(speed.max())


def central_pressure_mb(fix, state, *, on_refine_grid: bool) -> float:
    """The vortex's central pressure in mb, from the right grid.

    Free in exactly one configuration and reduced in every other, and
    the difference matters enough to be explicit:

    * ``field = "pressure"`` with no ``level_hpa`` -- the tracker's own
      signal IS sea-level pressure, and ``fix.extremum`` is already its
      minimum in hPa.  Nothing is recomputed.
    * ``level_hpa`` set -- the extremum is METRES OF GEOPOTENTIAL HEIGHT
      on an isobaric surface.  It is not a pressure and must not be
      printed as one.
    * ``uh`` / ``reflectivity`` -- the extremum is m2 s-2 or dBZ.  Same.

    In the latter two the reduction runs here, on the grid the centre
    came from.  When that is the mover's parent the tracker's own search
    box crops it (measured 240 ms -> 26 ms, bitwise); when the centre
    came from a refine grid the tracker searched that grid whole, so
    there is no box to inherit and the reduction is whole-grid too.
    """
    if fix.field_used == "pressure" and fix.evidence.get(
            "extremum_units") == "hPa":
        # Already sea-level pressure, in hPa, already the minimum over
        # the region searched.  hPa and mb are the same unit.
        return float(fix.extremum)
    window = None if on_refine_grid else fix.search_box
    plane = mslp_hpa_from_state(state, window=window)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        # Tested rather than caught from nanmin's warning: an all-NaN
        # window is a real state (every column below the surface, or a
        # window that missed the field) and it has to refuse, not warn.
        raise TrackRefusal(
            "the sea-level pressure reduction produced no finite value "
            "over the searched region; there is no central pressure to "
            "report for this consultation")
    return float(finite.min())


# ---------------------------------------------------------------------------
# The row
# ---------------------------------------------------------------------------

def format_row(*, valid_time, lat, lon, mslp_mb=None, vmax_m_s=None,
               levels=(), position_only: bool = False,
               surface: bool = True) -> str:
    """One track row: valid time, decimal degrees, mb, m/s, then one
    (lat, lon, height) triple per tracked isobaric surface.

    THE TIME COLUMN IS THE VALID TIME, spelled exactly as wrfout spells
    it (``2025-10-24_12:36:00``), so a row joins onto a history frame by
    string equality and a reader never has to know the run's initial time
    to place it.  Model seconds were the first design and were wrong for
    the file's actual audience: a person plotting a storm wants a clock,
    and the elapsed time is recoverable from any two rows anyway.

    DECIMAL DEGREES, signed -- no hemisphere letters.  This file exists
    to be read and plotted, and a plotter that has to un-encode a
    hemisphere is a plotter nobody writes.

    ``levels`` is ``[(lat, lon, height_dam), ...]`` in the order the
    config named the surfaces.  HEIGHT IS IN DECAMETRES, which is the
    unit every upper-air chart uses -- 850 hPa reads 143 dam in the deep
    tropics, not 1434 m -- so a number in this file compares directly
    against the analysis a forecaster is already looking at.

    A surface that produced no centre this consultation contributes
    ``NaN`` in all three of its columns and does NOT shift the ones
    after it: the column count is fixed by the config, not by what the
    atmosphere happened to offer.

    ``None`` in any data column writes ``NaN``.  A consultation that
    found nothing is therefore a row of all-NaN data with its clock
    intact: the time axis stays complete, so a gap is VISIBLE in the
    file rather than inferred from a jump in the clock, and no separate
    marker column is needed to say so.

    NOT PADDED.  The old format was fixed-width as well as
    whitespace-separated, and padding a CSV field is legal but hostile:
    it forces ``skipinitialspace`` on readers that do not strip, and it
    puts trailing blanks inside the one column that is a string.  The
    commas carry the structure now.

    PRECISION IS FIXED PER COLUMN rather than left to ``repr``: four
    decimal places of latitude is 11 m, finer than any grid this tracker
    runs on, and a column that prints 15.7072 on one row and
    15.707199999999998 on the next diffs badly and plots identically.

    WHICH GRID a row came from is deliberately NOT here.  It is on the
    relocation receipt for every consultation, where the rest of the
    provenance already lives, and a column that is the same string on
    almost every row of a plotting file earns nothing.
    """
    def _num(value, spec):
        return (CSV_MISSING
                if value is None or not math.isfinite(value)
                else format(value, spec))
    stamp = (valid_time.strftime(TIME_FORMAT)
             if hasattr(valid_time, "strftime") else str(valid_time))
    if position_only:
        # POSITION_ONLY_FIELDS: a clock and where the nest is, and the
        # columns that would have held a central pressure and a peak wind
        # are absent rather than NaN -- they are not missing here, they
        # do not exist for this tracker.
        return ",".join([stamp, _num(lat, ".4f"), _num(lon, ".4f")])
    # ``surface`` and the CONTENT of ``levels`` are decided by the
    # caller's output_level; this function only lays out what it is
    # given, so the header and the row cannot disagree about which
    # blocks are present.
    cells = [stamp]
    if surface:
        cells += [_num(lat, ".4f"), _num(lon, ".4f"),
                  _num(mslp_mb, ".2f"), _num(vmax_m_s, ".2f")]
    for lat_l, lon_l, hgt in levels:
        cells += [_num(lat_l, ".4f"), _num(lon_l, ".4f"), _num(hgt, ".2f")]
    return ",".join(cells)


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------

@dataclass
class _Stream:
    """The open file, its cadence, and its tally."""

    config: TrackConfig
    path: Path
    #: The isobaric surfaces the tracker watches, in config order.  The
    #: header is a function of these, and it is written once at open, so
    #: the column count is fixed by the CONFIG and never by what the
    #: atmosphere happened to offer on a given consultation.
    levels: tuple = ()
    #: The tracker's own field.  Decides the COLUMN SET (see
    #: :func:`csv_columns`), and like ``levels`` it is fixed at open,
    #: so a file's shape is a statement about the config that
    #: produced it and never about what a consultation found.
    tracked_field: str = "pressure"
    #: Which blocks the file carries; None is all of them.  Fixed at
    #: open like ``levels``, for the same reason.
    output_level: tuple | None = None
    #: The run's own output directory, or None for a stream constructed
    #: without one.  Only used to decide whether this file is OURS to
    #: truncate -- see :meth:`open`.
    root: Path | None = None
    handle: object = None
    emitted: int = 0
    skipped: int = 0
    faults: list[str] = field(default_factory=list)
    _last_emit_t: float | None = None

    def due(self, t: float) -> bool:
        interval = self.config.interval_seconds
        if interval is None or self._last_emit_t is None:
            return True                    # every consultation
        return (t - self._last_emit_t) >= interval - 1.0e-6

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._refuse_to_clobber()
        # Truncate at construction, append thereafter: one file per run,
        # never renamed, so tailing it cannot break the forecast the way
        # a rewrite-and-rename can on Windows (a user tailing the
        # receipts once killed a 6 h run with WinError 5).
        #
        # LINE BUFFERED (buffering=1) is the tail-safety guarantee: the
        # runtime itself flushes when a newline is written, so a row
        # reaches the OS as a whole row through EVERY write path, not
        # only the one below that remembers to flush.  It changes no
        # byte of the file -- only when those bytes leave this process.
        self.handle = self.path.open("w", buffering=1, encoding="utf-8",
                                     newline="\n")
        self.handle.write(
            csv_header(self.levels, tracked_field=self.tracked_field,
                       output_level=self.output_level)
            + "\n")
        # THE HEADER IS ON DISK BEFORE THE FIRST CONSULTATION.  A tail
        # attaches when the run starts, which is before the tracker has
        # been asked anything; a reader that finds an empty file has no
        # column names and has to guess the shape of what follows.
        self.handle.flush()

    def _refuse_to_clobber(self) -> None:
        """Never destroy rows this run did not write.

        A RELATIVE ``path`` cannot reach this: the runner refuses an
        ``--outdir`` that already holds a run (before any config is
        read, so it binds every run there is), which means a fresh
        output directory is a fresh file, and truncating it is correct --
        a run starts its own file rather than appending to a stale one.

        AN ABSOLUTE ``path`` escapes that guard entirely, and the
        combination that loses data is the ordinary one: point two legs
        of the same forecast at one deck, resume the second, and opening
        for write erases the first.  A restart already needs its own
        output directory for everything else it writes; the one file that
        could be aimed outside it is this one, so the refusal is here.

        Refused at CONSTRUCTION -- before a single step -- and not caught
        as a per-record fault, because "the file already has somebody
        else's rows in it" is not something a later row can recover from,
        and a diagnostic that silently deletes a deck is worse than one
        that refuses to start.
        """
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size == 0:
                return                      # nothing of anyone's in it
            root = None if self.root is None else Path(self.root).resolve()
            here = self.path.resolve()
            if root is not None and root in here.parents:
                return                      # this run's own directory
        except OSError:
            # Unreadable is not empty: the one direction that loses data
            # is accepting on a failed probe.  Same ruling the runner's
            # own --outdir guard makes.
            pass
        raise TrackRefusal(
            f"[relocation.track] path {self.path} already exists and is "
            "not empty, and it is outside this run's output directory -- "
            "so opening it would erase rows this run did not write. That "
            "is what happens when two legs of one forecast (a run and its "
            "restart) are pointed at one absolute path. Give this leg its "
            "own path, or delete that file deliberately. The legs' rows "
            "are byte-identical where they overlap, so concatenating them "
            "and de-duplicating on valid_time reconstructs one deck.")

    def write(self, line: str, t: float) -> None:
        """One row, on disk before this returns.

        FLUSH PER ROW, stated rather than inherited: the newline the
        line-buffered handle acts on and this flush say the same thing
        twice on purpose, so the guarantee survives a handle somebody
        later opens differently.  A killed process loses nothing (the
        bytes are already with the OS) and a reader tailing the file
        sees whole rows only, because a row is written and flushed as
        one line.  Not fsync -- see the module docstring for why a
        device flush does not belong on the model's thread.
        """
        self.handle.write(line + "\n")
        self.handle.flush()
        self.emitted += 1
        self._last_emit_t = float(t)

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def receipt(self) -> dict:
        out = {"path": str(self.path), "records": int(self.emitted),
               "skipped": int(self.skipped),
               "columns": list(csv_columns(
                   self.levels, tracked_field=self.tracked_field,
                   output_level=self.output_level))}
        if self.config.interval_seconds is not None:
            out["interval_seconds"] = float(self.config.interval_seconds)
        if self.faults:
            out["faults"] = self.faults[:8]
            out["fault_count"] = len(self.faults)
        return out


class TrackWriter:
    """Renders every consultation's fix into the track file.

    Constructed by the route (it owns the initial time and the output
    directory); driven by
    :class:`woof.core.relocation_runner.RelocationRunner`, which is the
    only thing that knows when a consultation happened.

    The file is TAIL-SAFE for the whole life of the run: the header is
    on disk before the first consultation, and every row is line
    buffered and flushed as it lands, so a reader watching the file
    sees complete rows and never a torn one.  ``fsync`` is deliberately
    not part of that -- both halves are stated in the module docstring.
    """

    def __init__(self, config: TrackConfig, *, initial_time, outdir=None,
                 levels=(), tracked_field: str = "pressure"):
        if not isinstance(config, TrackConfig):
            raise TypeError("config must be a TrackConfig")
        self.initial_time = initial_time
        root = Path(outdir) if outdir is not None else Path(".")
        path = Path(config.path)
        self.levels = tuple(float(v) for v in (levels or ()))
        self.tracked_field = str(tracked_field)
        #: A rotation or echo tracker writes the MOVER'S centre and
        #: nothing else -- see POSITION_ONLY_FIELDS for why the other
        #: columns are absent rather than NaN.
        self.position_only = self.tracked_field in POSITION_ONLY_FIELDS
        #: The blocks this file carries, resolved once from the config.
        #: ``surface`` is whether the deep-layer-mean centre, the central
        #: pressure and the peak wind are in it; ``emitted_levels`` is the
        #: computed surfaces that survived, IN THE ORDER block_order puts
        #: them -- the same function the header is built from, so the row
        #: builder walks exactly what the header named, in the same
        #: sequence.  Deriving these two independently is how a file ends
        #: up with a header and rows that disagree.
        self.output_level = config.output_level
        blocks = block_order(self.levels, self.output_level)
        self.surface = SURFACE_LEVEL in blocks
        self.emitted_levels = tuple(v for v in blocks if v != SURFACE_LEVEL)
        self.stream = _Stream(config=config, levels=self.levels,
                              tracked_field=self.tracked_field,
                              output_level=self.output_level, root=root,
                              path=path if path.is_absolute() else root / path)
        self.stream.open()
        self.termination = None

    def valid_time(self, t: float):
        """Model seconds since the run's start -> the row's clock."""
        return self.initial_time + dt.timedelta(seconds=float(t))

    # -- the seam ----------------------------------------------------------

    def emit(self, fix, *, t: float, parent_state, refine_state=None,
             parent_grid=None, refine_grid=None, mover_grid=None,
             diagnostic_nodes=None, swath_grid=None, force=False
             ) -> dict | None:
        """Render one consultation, if this one is due.

        ``fix`` may be ``None`` under POSITION_ONLY_FIELDS, where the row
        does not read it: the caller is then free to skip locating
        entirely on a track-only boundary rather than pay for a reduction
        whose answer never reaches the file.  A ``None`` fix is reported
        as ``no_signal`` on the returned receipt, same as a fix that
        found nothing.

        NEVER RAISES.  A track writer is a diagnostic and a diagnostic
        that can kill a 12-hour forecast is a defect, so every fault is
        caught, counted and reported on the stream's receipt.
        """
        stream = self.stream
        if self.termination is not None:
            return None
        reason = (getattr(fix, "evidence", {}) or {}).get("track_end_reason")
        if reason:
            self.termination = {"status": "ended", "reason": str(reason), "t": float(t)}
            return {"contract": TRACK_CONTRACT, "t": float(t), "emitted": False,
                    **self.termination}
        if not force and not stream.due(float(t)):
            return None
        stamp = self.valid_time(t)
        no_signal = fix is None or fix.found is None
        if no_signal and not self.position_only:
            # NO SIGNAL: a row whose data columns are all NaN, so the
            # file's time axis stays complete and the gap is visible
            # rather than inferred from a jump in the clock.
            stream.write(format_row(
                valid_time=stamp, lat=None, lon=None, mslp_mb=None,
                vmax_m_s=None, surface=self.surface,
                levels=[(None, None, None)] * len(self.emitted_levels)), t)
            return {"contract": TRACK_CONTRACT, "t": float(t),
                    "emitted": True, "no_signal": True}
        try:
            sample = self._sample(fix, parent_state=parent_state,
                                  refine_state=refine_state,
                                  parent_grid=parent_grid,
                                  refine_grid=refine_grid,
                                  mover_grid=mover_grid,
                                  diagnostic_nodes=diagnostic_nodes,
                                  swath_grid=swath_grid)
            field_fix = sample.pop("field_fix", None)
            stream.write(format_row(valid_time=stamp,
                                    surface=self.surface, **sample), t)
            if field_fix is not None:
                from woof.tc.field_products import publish
                publish(stream.path, field_fix, storm_id=stream.config.storm_id,
                        initial_time=self.initial_time, valid_time=stamp)
        except Exception as error:                       # noqa: BLE001
            stream.skipped += 1
            stream.faults.append(f"{stamp:%Y-%m-%d_%H:%M:%S}: {error}")
            return {"contract": TRACK_CONTRACT, "t": float(t),
                    "emitted": False, "reason": str(error)}
        # The position that reached the FILE, carried back on the receipt.
        # The caller renders the same fix into the per-step event stream,
        # and deriving it a second time there would be a second answer to
        # a question this method has already answered -- the two could
        # then disagree about where the storm was, which is exactly the
        # class of error `_sample`'s one-grid rule exists to prevent.
        row = {"contract": TRACK_CONTRACT, "t": float(t), "emitted": True,
               "lat": sample.get("lat"), "lon": sample.get("lon")}
        if no_signal:
            # POSITION-ONLY reached here with nothing found: the row is
            # still true (the nest IS somewhere), so the fact that the
            # tracker held goes on the RECEIPT rather than into the file
            # as a NaN that would misreport a known position.
            row["no_signal"] = True
        return row

    def _sample(self, fix, *, parent_state, refine_state,
                parent_grid, refine_grid, mover_grid=None,
                diagnostic_nodes=None, swath_grid=None) -> dict:
        """Every quantity in a row, all from ONE grid.

        The grid that LOCATED the centre is the grid every quantity is
        read from.  A position from a 643 m nest paired with a central
        pressure from its 4.5 km parent would be a row that is right
        about where the storm is and wrong about how strong it is, which
        is worse than either error alone: a 4.5 km grid cannot resolve an
        eye.

        UNDER POSITION_ONLY_FIELDS none of that applies, because there is
        only one quantity and it does not come from the fix at all: the
        row is the MOVER'S own centre, read from the mover's own grid.
        That is deliberately a different question from "where is the
        signal" -- the tracker's answer is quantised to whole parent
        cells and gated by the dead-band and the cooldown, so the nest
        sits where it sits, and for a rotation or echo tracker the useful
        thing to plot is the footprint that actually integrated.
        """
        if self.position_only:
            if mover_grid is None:
                raise TrackRefusal(
                    f"a {self.tracked_field!r} tracker writes the moving "
                    "domain's own centre, and no grid was supplied for it; "
                    "the mover's grid is what turns its footprint into a "
                    "latitude and longitude")
            lat, lon = grid_center_latlon(mover_grid)
            if not (math.isfinite(lat) and math.isfinite(lon)
                    and -90.0 <= lat <= 90.0):
                raise TrackRefusal(
                    f"the moving domain's centre converted to an impossible "
                    f"position (lat={lat!r}, lon={lon!r}); its grid and its "
                    "dimensions disagree")
            return {"lat": lat, "lon": lon, "position_only": True}
        on_refine = (fix.refined_on is not None and refine_state is not None
                     and refine_grid is not None)
        if on_refine:
            grid, state = refine_grid, refine_state
            ci, cj = fix.refined_cell_ij
        else:
            if parent_grid is None:
                raise TrackRefusal(
                    "no projected grid was supplied for the domain the "
                    "centre was found on, so the centre cannot be turned "
                    "into a latitude and longitude")
            grid, state = parent_grid, parent_state
            ci, cj = fix.center_parent_ij
        lat, lon = latlon_from_grid(grid, ci, cj)
        if not (math.isfinite(lat) and math.isfinite(lon)
                and -90.0 <= lat <= 90.0):
            raise TrackRefusal(
                f"the centre converted to an impossible position "
                f"(lat={lat!r}, lon={lon!r}); the grid and the centre "
                "disagree about the domain")
        # The surface block is a reduction and a whole-grid maximum, and
        # under `output_level` without 0 the file does not carry either.
        # Skipping them is not only cheaper: peak_wind_m_s REFUSES when
        # no surface-layer scheme published u10/v10, so computing a
        # column nobody asked for is a way to fault a row that is
        # otherwise perfectly writable.  Same rule as
        # POSITION_ONLY_FIELDS, one block down.
        out = {"lat": lat, "lon": lon,
               "levels": self._level_columns(fix, grid)}
        if self.surface:
            from woof.tc.field_products import diagnose
            candidates = ([(n.state, n.grid, n.cfg.grid_id) for n in diagnostic_nodes]
                          if diagnostic_nodes else [(state, grid, fix.refined_on)])
            selected = None
            for candidate, candidate_grid, gid in candidates:
                x, y = candidate_grid.latlon_to_ij(lat, lon)
                if not (1 <= float(x) <= candidate_grid.e_we-1
                        and 1 <= float(y) <= candidate_grid.e_sn-1):
                    continue
                field_fix = diagnose(candidate, candidate_grid, (lat, lon), swath=swath_grid)
                field_fix['grid_id'] = gid
                selected = (candidate, candidate_grid)
                break
            if selected is None:
                raise TrackRefusal("no active grid contains the tracked centre")
            field_centre = [field_fix['latitude_deg'], field_fix['longitude_deg']]
            field_fix['radii_grid_id'] = {k: [field_fix['grid_id']]*4 for k in ('34','50','64')}
            # A fine core may not contain R34. Each censored quadrant takes
            # the next coarser complete radius, about the SAME field centre.
            for candidate, candidate_grid, gid in candidates:
                if candidate is selected[0]:
                    continue
                if not any(any(v) for v in field_fix['radii_reach_grid_edge'].values()):
                    break
                outer = diagnose(candidate, candidate_grid, field_centre, radii_centre=field_centre)
                for threshold in ('34','50','64'):
                    for quadrant in range(4):
                        if (field_fix['radii_reach_grid_edge'][threshold][quadrant]
                                and not outer['radii_reach_grid_edge'][threshold][quadrant]):
                            field_fix['radii_nm'][threshold][quadrant] = outer['radii_nm'][threshold][quadrant]
                            field_fix['radii_reach_grid_edge'][threshold][quadrant] = False
                            field_fix['radii_grid_id'][threshold][quadrant] = gid
            out.update(lat=field_fix['latitude_deg'], lon=field_fix['longitude_deg'],
                       mslp_mb=field_fix['mslp_hpa'], vmax_m_s=field_fix['vmax10_m_s'],
                       field_fix=field_fix)
        return out

    def _level_columns(self, fix, grid) -> list:
        """One (lat, lon, height_dam) triple per CONFIGURED surface.

        Indexed by the configured level, not by what the fix returned:
        a surface that declined this consultation contributes NaN in its
        own three columns and does not shift the ones after it.  The
        column count belongs to the config.

        Every surface here was computed by the TRACKER -- steering
        surfaces and report-only ones alike come back on ``fix.levels``
        (:func:`woof.core.storm_tracking.all_levels_of`), so there is
        exactly one centre per surface per consultation and this writer
        never runs a second search of its own.

        ``LevelFix.fix_ij`` is the centre in the index space of the grid
        that FOUND it -- which under the two-stage tracker is the refine
        grid, not the parent -- because a latitude must be taken from a
        grid in its own index space (see :func:`latlon_from_grid`).
        """
        if not self.emitted_levels:
            return []
        by_level = {round(float(f.level_hpa), 6): f
                    for f in getattr(fix, "levels", ())}
        out = []
        # EMITTED, not computed: a surface the run computes but does not
        # print still steers the nest (or still lands on the receipt) --
        # output_level chooses what the FILE carries, never what the
        # tracker watches.
        for level in self.emitted_levels:
            found = by_level.get(round(float(level), 6))
            if found is None:
                out.append((None, None, None))
                continue
            ci, cj = found.fix_ij
            lat, lon = latlon_from_grid(grid, ci, cj)
            if not (math.isfinite(lat) and math.isfinite(lon)
                    and -90.0 <= lat <= 90.0):
                out.append((None, None, None))
                continue
            out.append((lat, lon, found.height_dam))
        return out

    def close(self) -> dict:
        receipt = self.stream.receipt()
        if self.termination is not None:
            receipt["termination"] = dict(self.termination)
        self.stream.close()
        return receipt


__all__ = [
    "CSV_HEADER", "CSV_POSITION_COLUMNS", "POSITION_ONLY_FIELDS",
    "SURFACE_LEVEL", "wanted_levels", "block_order",
    "TIME_FORMAT", "TRACK_CONTRACT", "TRACK_KEYS",
    "TrackConfig", "TrackRefusal", "TrackWriter", "build_track_config",
    "central_pressure_mb", "csv_columns", "csv_header", "format_row",
    "grid_center_latlon", "latlon_from_grid", "peak_wind_m_s",
]
