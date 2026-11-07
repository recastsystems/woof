"""The long time step each domain's ground and crest-level wind allow.

THE DEFECT THIS EXISTS FOR.  The acoustic substep rule
(:mod:`woof.acoustic_adaptation`) was measured under cross-ridge winds of
10 to 25 m/s.  Under a jet at crest height the same ground stops runs that
rule leaves alone: at 1 km, 5.5 and 6.5 km crests of slope 0.6 to 0.65
stopped on four substeps under 50 to 60 m/s, and at 3 km, 4.5 and 5 km
crests of slope 0.4 under 60 to 70 m/s stopped within two minutes on four
AND six substeps.  There no substep count is the remedy; the long step
is.

MEASURED, NOT ASSUMED.  :data:`MAP_PATH` is the engine's own map: a bell
ridge across the grid in a uniform cross-ridge wind, integrated for half
an hour through the production ``step()`` with the generated dynamics
(the 49-level ladder, the etac the vertical survey derives for the crest,
``epssm`` 0.5, ``smdiv``/``emdiv``, ``w_damping``, the slope-tapered
sixth-order filter, the Rayleigh lid), at 500 m to 12 km, crests of 1.5
to 8.85 km and winds of 20 to 100 m/s.  Each row is one spacing, crest
height, grid-read steepest slope and substep count; for each crest-level
wind it holds the longest step, in seconds per kilometre of grid spacing,
that ran without stopping and without its vertical velocity passing four
times the wind times the slope plus 20 m/s.  ``None`` means no measured
step held.  The probe domain is periodic and 96 to 256 columns wide, so
from 50 m/s up the flow can come back round to the ridge inside the half
hour; every step held there was run again on a domain the flow cannot
wrap, and lowered to what held on it (at 1 km, a 4.5 km crest of slope
0.4 held 5 s/km at 90 and 100 m/s on 96 columns and stopped within
fifteen minutes on 1024).  Then every pair the rule picks at 1 and 3 km
from the generated step under 40 to 80 m/s ran three hours on a ridge
whose flow cannot wrap the domain in that time: 135 of 142 held, and
where one stopped the row now holds the step that held three hours
there, or nothing.

LONGER STEPS AT 3 KM.  Adaptive 3 km runs take steps past the ladder's
15 s, so the 3 km rows of crests 1.5 to 4.5 km and slopes up to 0.4 were
also tried from 40 s down to 20 s at 20 to 90 m/s on four and six
substeps (``tools/terrain_clock_probe.py extend``, which rebuilds these
ridges row key for row key), each step held that way run three hours.
238 of the 336 cells held a step of 20 to 40 s; four substeps never held
40 s, and under a 4.5 km crest of slope 0.36 only 20 m/s held more than
15 s on four.  A row records the longest step tried at each wind
(``top_s_per_km``, the ladder's top where it records none), and a cell
whose entry is shorter than that saw a longer step stop.  That is read
cell by cell: a reading takes every row as gentle as the domain's ground
or gentler and every wind up to its own, and where any of those cells
saw a longer step stop, the shortest such entry caps the step.  So 3 km
ground steeper than the extended rows (0.41 and up, tried to 15 s) is
capped by the stops its gentler rows saw at 20 s and longer, the 100 m/s
column by those seen at 90 m/s, and a spacing between 2 and 3 km by its
3 km rows.  A stop seen under a lower crest counts too, past the longest
step the domain's own crest was tried at: 3 km crests above 4.5 km were
tried only to 15 s, and a longer step seen to stop under the 4.5 km and
lower crests at the same slope and wind caps them there, since the
taller the crest the thinner the layer over it.  Only where no cell read
saw a step stop is a longer step left alone, and a cap never lengthens
the adaptive clock's own longest step.  Over the map:

* at 3 km the generated 15 s step holds every wind to 90 m/s on slopes
  to about 0.25 under crests to 4.5 km.  Steeper ground needs a shorter
  step from 40 to 80 m/s: under a 4.5 km crest a slope of 0.41 holds
  4.0 s/km at 50 m/s on four substeps and 4.5 on six, and 3.0 and 3.5
  at 100;
* at 1 km six substeps hold the generated 5 s step to 90 m/s on slopes
  to about 0.55 under crests to 4.5 km, to 70 m/s on slopes to about
  0.65 under a 5.5 km crest and to 60 m/s on slopes to about 0.75 under
  a 6.5 km crest.  Steeper ground needs a shorter step from 50 to 80 m/s,
  and slopes near 1 from 30;
* the taller the crest, the thinner the layer over it and the lower the
  wind past which no measured step holds at all: at 3 km, 80 m/s under a
  5.5 km crest, 70 under 6.5 km, 60 under 8 km and 40 under 8.85 km.
  There the layer, not the step, is the limit.

WHAT THIS DOES.  Each domain is read for its steepest slope (the reading
the substep rule takes), its crest height (its highest ground) and its
crest-level wind: the strongest wind its start state and boundary data
carry over the forecast window, from the ground up to the first model
level at or above the crest height.  The speed is taken whole, not its
component across the ground, because the wind turns over a forecast and
real ground faces every way; that is the conservative reading.  The map
is read on its conservative side (the next steeper slope, taller crest,
stronger wind and both neighbouring spacings) and the domain runs the
cheapest pair it holds: the smallest whole division of its configured step,
then the fewest substeps that hold at it.  A domain the map holds as
configured runs exactly as configured.  The step is only ever divided by a
whole number, so every cadence that was a whole number of steps stays
one; a nest's step ratio grows by the division its parent does not
already give it.  Under the adaptive clock the held step becomes a
ceiling on ``max_time_step``, unless no cell read saw a step stop: past
the longest step measured on held ground, the adaptive clock's own limits
govern.  Past the strongest wind the map holds a step at, the most stable
pair measured is the ceiling.  A ceiling only ever shortens the step: one
at or above the clock's own longest step (``max_time_step``, or WRF's
8 x dx fill-in where it is -1) is not written, so ground the map measured
no further than it held never runs a longer step than it would without
the map.  The run's ``terrain_clock`` record states the longest step
tried, per spacing and for each domain's reading, and the shortest entry
under a stop it read and the crest that entry was measured under.

THE ADAPTIVE CLOCK'S OWN ENTRIES.  Every entry above is a FIXED step on
a fixed substep count, while the adaptive clock shortens its step on the
CFL it measures and derives its substep count from the live step (eight
at 22.6 s and up at 2.25 km).  Capping it from a fixed four-substep stop
capped a 2.25 km domain under a 3.6 km crest of slope 0.30 at 24 m/s at
15 s, where the adaptive clock held 30 s on that ridge for three hours.
So the four-substep rows at 2 and 3 km of crests 1.5 to 4.5 km and ridge
slopes 0.1 to 0.4, and under the 4.5 km crest ridge slopes 0.45 and 0.5,
were also run on the production adaptive clock
(``tools/terrain_clock_probe.py adaptive-extend``): max_time_step from
15 s/km down to 5 s/km under 20 to 60 m/s, three hours each at the CFL
target pairs 1.2 / 0.84 and 1.4 / 0.98 with 5 percent growth.  Every row
held 15 s/km, the longest tried, at 20 m/s, and every one but the 3 km
row of ridge slope 0.5 (grid 0.47), which held 5 s/km, at 30 m/s; from
40 m/s some stopped, the gentlest slopes under the taller crests first,
and some held none.  An adaptive domain whose reading takes only rows
run that way, within the map's edges, and whose every cell held the
longest step tried there, is capped at that step instead of the
fixed-step stop (never below it), for a clock no faster than the one
measured.  Wherever a cell run that way held a
longest step under a longer one that stopped, that entry caps the clock
whatever the fixed reading says.  A cell that held none gives no cap and
says the domain may still stop.  Fixed steps read the fixed entries
alone, as before.

A FIXED STEP reads the same map.  At 3 km a fixed step of 15 s or less
(the domain wizard writes 15 s) reads exactly as it did before the 3 km
rows were extended, since no entry under 15 s moved.  A fixed 3 km step
above 15 s over ground where 20 s was seen to stop is moved to six
substeps or divided: under a 3.9 km crest of slope 0.36, a fixed 18 s
runs on six substeps instead of four at 30 m/s and is halved to 9 s on
four from 40 m/s, and a fixed 20 s likewise runs on six at 30 m/s and
is halved to 10 s from 40 m/s.

THE LOCAL-FACE CLOCK.  ``terrain_clock = "local_face"`` (the default from
2.8.8; :mod:`woof.terrain_clock_local`) reads every face with the crest and
crest-level wind in a measured neighbourhood around it, times a measured
wind margin and never past the domain-wide wind read here, on these rows
plus the candidate's own, and holds the domain to the least of them.  The
shipped reading below is untouched by it, and it is the floor under the
candidate: where the face-by-face reading would be BEYOND_MEASURED, or
would run a grid on a shorter step than the shipped reading gives, the
grid takes the shipped reading's decision and its receipt and run line say
so (:func:`woof.terrain_clock_local.never_worse`).

UNDER zadvect_implicit.  Every entry was measured with explicit vertical
advection, and a domain that runs WRF's implicit-explicit vertical
advection (:mod:`woof.core.ieva`) reads the map unchanged: the step and
substeps it is held to are the ones that held without the option, never
longer.  That is what the probe measures with the option on, too: on the
map's own ridges at 3 km (crests 3.9 and 4.5 km, slopes 0.36 and 0.4,
20 to 40 m/s, 15 to 20 s on four and six substeps, one hour each, 96 and
384 columns), each of 24 cells run once with it and once without held
both times or stopped both times, with peak vertical velocities within
1.5 m/s of each other wherever they held.  The ridge's limit is the
acoustic and terrain-following one the substeps answer, not the vertical
Courant number the option relaxes, so the map gives the option nothing
to lengthen.  Its run line and its receipt say so (``zadvect_implicit``,
``map_vertical_advection``).
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Mapping, Sequence

import numpy as np

from woof.core import constants as _constants

#: Receipt schema for the derivation this module performs.
TERRAIN_CLOCK_SCHEMA = "gpuwm-terrain-clock-adaptation-v1"

#: The measured map this rule reads.
MAP_PATH = Path(__file__).with_name("terrain_clock_map.json")

#: How far a domain's crest may sit above a mapped crest and still read
#: that row, as a share of the crest.  The mapped heights are the ridges'
#: nominal heights, their own highest cells sit 1 to 13 percent under them,
#: and the height acts through the layer over the crest, which the 15
#: percent margin of :mod:`woof.vertical_adaptation` sets alike on both:
#: a 6456.3 m crest is the 6456 m row's ground, not the 8000 m row's.
CREST_MATCH = 0.01

#: Gravitational acceleration that turns geopotential into height: the
#: dynamics' own.
_G = float(_constants.G)


@dataclass(frozen=True)
class MapRow:
    dx_m: float
    crest_m: float
    slope: float
    sound_steps: int
    #: Longest held step in s/km per mapped wind, ``None`` where none held.
    stable: tuple[float | None, ...]
    #: Longest step TRIED in s/km per mapped wind: an entry this long held
    #: every step tried there, and says nothing about a longer one.  The
    #: map's own ladder top where the row carries none.
    tried: tuple[float, ...] | None = None
    #: On the ADAPTIVE clock, per mapped wind: the longest max_time_step,
    #: s/km, that held three hours (:class:`AdaptiveMeasurement`), ``None``
    #: where none tried held.  ``None`` for a row never run that way.
    adaptive: tuple[float | None, ...] | None = None
    #: The longest max_time_step tried at each wind, s/km, ``None`` at a
    #: wind not run on the adaptive clock.
    adaptive_tried: tuple[float | None, ...] | None = None

    def tried_at(self, index: int, top: float) -> float:
        return float(top) if self.tried is None else float(self.tried[index])

    def measured(self, index: int) -> bool:
        """The row was run at this wind.  Every shipped row was run at
        every mapped wind; a row the local-face candidate adds
        (:mod:`woof.terrain_clock_local`) carries ``None`` in ``tried``
        at a wind it was not run at, and that cell is not read: it is no
        evidence of a step holding or stopping."""
        return self.tried is None or self.tried[index] is not None

    def adaptive_measured_at(self, index: int) -> bool:
        return (self.adaptive_tried is not None
                and self.adaptive_tried[index] is not None)


@dataclass(frozen=True)
class AdaptiveMeasurement:
    """How the map's adaptive entries were measured: the same ridge on the
    production adaptive clock (the controller fed the dycore's own CFL,
    WRF's substep count from the live step, landing on every hour), from a
    first step of the ladder's top, each entry held three hours at the CFL
    target pairs ``targets`` alike, with growth bounded at
    ``max_step_increase_pct``."""

    #: The CFL target pairs every entry held at.
    targets: tuple[tuple[float, float], ...]
    max_step_increase_pct: int
    #: The max_time_step values tried, s/km, longest first.
    ladder: tuple[float, ...]
    seconds: float

    def covers(self, run) -> bool:
        """A clock no faster than every one measured: targets no higher
        than the highest pair and growth no faster.  A faster clock was
        never run on this ground, so its entries are not read as held."""
        cfl = max(pair[0] for pair in self.targets)
        hcfl = max(pair[1] for pair in self.targets)
        return (float(getattr(run, "target_cfl", 1.2)) <= cfl + 1e-12
                and float(getattr(run, "target_hcfl", 0.84)) <= hcfl + 1e-12
                and int(getattr(run, "max_step_increase_pct", 5))
                <= int(self.max_step_increase_pct))


@dataclass(frozen=True)
class StableStepMap:
    winds: tuple[float, ...]
    ladder: tuple[float, ...]
    rows: tuple[MapRow, ...]
    seconds: float
    adaptive: AdaptiveMeasurement | None = None

    @property
    def top(self) -> float:
        """The ladder's longest step, in s/km, tried on every row.  A row
        carrying ``top_s_per_km`` was also tried at longer steps."""
        return float(self.ladder[0])

    def measured_range(self) -> dict[float, float]:
        """The longest step tried at each spacing, in seconds."""
        longest: dict[float, float] = {}
        for row in self.rows:
            tried = max(row.tried_at(i, self.top)
                        for i in range(len(self.winds)) if row.measured(i))
            longest[row.dx_m] = max(longest.get(row.dx_m, 0.0),
                                    tried * row.dx_m / 1000.0)
        return dict(sorted(longest.items()))


@functools.lru_cache(maxsize=1)
def measured_map() -> StableStepMap:
    document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    winds = tuple(float(w) for w in document["winds_m_s"])
    rows = []
    for row in document["rows"]:
        stable = tuple(None if value is None else float(value)
                       for value in row["stable_s_per_km"])
        if len(stable) != len(winds):
            raise ValueError(
                f"{MAP_PATH.name}: a row carries {len(stable)} winds, the "
                f"map declares {len(winds)}")
        tried = row.get("top_s_per_km")
        if tried is not None:
            tried = tuple(float(value) for value in tried)
            if len(tried) != len(winds):
                raise ValueError(
                    f"{MAP_PATH.name}: a row's top_s_per_km carries "
                    f"{len(tried)} winds, the map declares {len(winds)}")
        adaptive = adaptive_tried = None
        if "adaptive_s_per_km" in row:
            adaptive = tuple(None if value is None else float(value)
                             for value in row["adaptive_s_per_km"])
            adaptive_tried = tuple(
                None if value is None else float(value)
                for value in row["adaptive_top_s_per_km"])
            if len(adaptive) != len(winds) or len(adaptive_tried) != len(
                    winds):
                raise ValueError(
                    f"{MAP_PATH.name}: a row's adaptive entries carry "
                    f"{len(adaptive)} and {len(adaptive_tried)} winds, the "
                    f"map declares {len(winds)}")
            if int(row["sound_steps"]) != 4:
                # The adaptive clock is read at four substeps, the count
                # it takes at its shortest steps; an entry on another row
                # would never be read.
                raise ValueError(
                    f"{MAP_PATH.name}: adaptive entries on a "
                    f"{row['sound_steps']}-substep row")
            for entry, top in zip(adaptive, adaptive_tried):
                if entry is not None and (top is None
                                          or entry > top * (1.0 + 1e-9)):
                    # Read as held past the range tried, such an entry
                    # would lift a cap over steps never run.
                    raise ValueError(
                        f"{MAP_PATH.name}: an adaptive entry of {entry} "
                        f"s/km past the {top} s/km tried there")
        rows.append(MapRow(float(row["dx_m"]), float(row["crest_m"]),
                           float(row["slope"]), int(row["sound_steps"]),
                           stable, tried, adaptive, adaptive_tried))
    measurement = None
    if "adaptive" in document:
        block = document["adaptive"]
        measurement = AdaptiveMeasurement(
            targets=tuple((float(a), float(b)) for a, b in block["targets"]),
            max_step_increase_pct=int(block["max_step_increase_pct"]),
            ladder=tuple(float(v) for v in block["ladder_s_per_km"]),
            seconds=float(block["seconds"]))
    elif any(row.adaptive is not None for row in rows):
        raise ValueError(
            f"{MAP_PATH.name}: rows carry adaptive entries but the map "
            "does not say how they were measured")
    return StableStepMap(winds, tuple(float(v) for v in
                                      document["ladder_s_per_km"]),
                         tuple(rows), float(document["seconds"]),
                         measurement)


@dataclass(frozen=True)
class MapReading:
    """What the map says for one domain at one substep count."""

    #: Longest held step in s/km, or ``None`` when the map measured no
    #: step holding there.
    per_km: float | None
    #: The mapped value the reading was taken at, each on its
    #: conservative side.
    dx_rows: tuple[float, ...]
    crest_row: float
    slope_row: float | None
    wind_row: float
    #: Which of the domain's readings lie past the map's edge.
    beyond: tuple[str, ...] = ()
    #: ``per_km`` where the map holds a step; past the strongest wind it
    #: holds one at, the step it held at the strongest wind it did, s/km:
    #: the most stable pair measured, which such a domain runs.
    most_stable_per_km: float | None = None
    #: The longest step tried on every row the reading was taken from, at
    #: every wind it read, s/km: the range measured on all of that ground.
    top_per_km: float = 5.0
    #: The shortest entry, over every cell read (each row the reading was
    #: taken from, at every wind up to its own), of a cell whose entry is
    #: shorter than the longest step tried there, s/km: that cell saw a
    #: longer step stop.  No cell read saw a step this long or shorter
    #: stop.  ``None`` where no cell with a held step saw one stop.  A
    #: lower mapped crest's cell tried past every step this ground was
    #: tried at counts too, at no less than that step (:func:`read_map`).
    stopped_per_km: float | None = None
    #: The mapped crest of the row whose cell gave ``stopped_per_km``, and
    #: whether it is a lower ridge's than the one the reading was taken at.
    stopped_crest_m: float | None = None
    stopped_under_a_lower_crest: bool = False
    #: THE ADAPTIVE CLOCK, read on the same cells (the rows read at this
    #: crest, at every wind up to the domain's; the four-substep rows
    #: only).  Whether every one of them was run on the adaptive clock.
    adaptive_measured: bool = False
    #: Where they all were: the longest max_time_step that held on every
    #: one, s/km, ``None`` where one held none.
    adaptive_per_km: float | None = None
    #: The longest max_time_step tried on every one, s/km.
    adaptive_top_per_km: float | None = None
    #: Over the cells run on the adaptive clock, the shortest entry of one
    #: that held a max_time_step and saw a longer one stop, s/km.  ``None``
    #: where no such cell was read.
    adaptive_stopped_per_km: float | None = None
    #: A cell read held no max_time_step tried: the clock stopped there at
    #: every one, down to the ladder's shortest.
    adaptive_none_held: bool = False
    #: That shortest max_time_step, s/km, where a cell held none.
    adaptive_none_held_below_per_km: float | None = None

    @property
    def adaptive_held_everything_tried(self) -> bool:
        """Every cell read was run on the adaptive clock and held every
        max_time_step tried there."""
        return (self.adaptive_measured and self.adaptive_per_km is not None
                and self.adaptive_stopped_per_km is None)

    @property
    def held_everything_tried(self) -> bool:
        """Every cell read held every step tried there, so a longer step is
        past what was measured, not a step seen to stop.  Decided cell by
        cell: a steeper row measured only to the ladder's top does not
        hide the stop a gentler row read beside it saw at a longer step."""
        return self.per_km is not None and self.stopped_per_km is None


def _sound_row(sound_steps: int) -> int:
    return 6 if int(sound_steps) >= 6 else 4


def read_map(dx: float, crest_m: float, slope: float, wind: float,
             sound_steps: int, table: StableStepMap | None = None
             ) -> MapReading:
    """The longest held step for a domain, read on the map's safe side.

    Spacing: both mapped spacings around ``dx`` (the nearest one past
    either end).  Crest: the lowest mapped crest at or above the domain's
    highest ground, the ridge as tall as its mountains.  A lower ridge of
    the same slope is a narrower one, a few cells wide, which is a
    different ground and not a safer reading of this one.  Slope: at that
    crest, the gentlest mapped slope at or above the domain's, and every
    gentler row, so a noisy row can only lower the answer.  Wind: the
    lowest mapped wind at or above the domain's, and every weaker one.
    The answer is the least of all of them.

    A stop is read the same way, cell by cell, and also on the lower
    mapped crests at the same spacing (each at its own gentlest slope at
    or above the domain's, and every gentler row, at every wind up to the
    domain's), but only at steps the domain's own rows were never tried
    at: a lower ridge's cell tried past the longest step tried on every
    row read at this crest (``top_per_km``), whose entry is shorter than
    the step tried there, gives a stop at its entry or at that longest
    step, whichever is longer.  The taller the crest the thinner the
    layer over it, so a longer step seen to stop under a lower crest is
    not left to run under a taller one never tried that long; below that
    longest step the domain's own rows decide, as they always did.
    """

    table = measured_map() if table is None else table
    count = _sound_row(sound_steps)
    rows = [row for row in table.rows if row.sound_steps == count]
    beyond = []
    spacings = sorted({row.dx_m for row in rows})
    dx = float(dx)
    if dx <= spacings[0]:
        dx_rows = (spacings[0],)
        if dx < spacings[0] * (1.0 - 1e-9):
            beyond.append("spacing")
    elif dx >= spacings[-1]:
        dx_rows = (spacings[-1],)
        if dx > spacings[-1] * (1.0 + 1e-9):
            beyond.append("spacing")
    else:
        exact = [s for s in spacings if abs(s - dx) <= 1e-6 * s]
        if exact:
            dx_rows = (exact[0],)
        else:
            lower = max(s for s in spacings if s < dx)
            upper = min(s for s in spacings if s > dx)
            dx_rows = (lower, upper)
    winds = table.winds
    above_w = [i for i, w in enumerate(winds) if w >= float(wind) - 1e-9]
    if above_w:
        wind_index = above_w[0]
    else:
        wind_index = len(winds) - 1
        beyond.append("wind")

    def gentle_rows(at_crest):
        # The gentlest row at or above the domain's slope, and every
        # gentler one; every row where the domain is steeper than all.
        steeper = sorted(row.slope for row in at_crest
                         if row.slope >= float(slope) - 1e-9)
        cap = steeper[0] if steeper else max(row.slope
                                             for row in at_crest)
        return cap, not steeper, [row for row in at_crest
                                  if row.slope <= cap + 1e-12]

    selected = []
    lower = {}
    slope_rows = []
    crest_rows = []
    slope_beyond = crest_beyond = False
    for spacing in dx_rows:
        crests = sorted({row.crest_m for row in rows
                         if row.dx_m == spacing})
        above = [h for h in crests
                 if h >= float(crest_m) * (1.0 - CREST_MATCH) - 1e-6]
        if above:
            crest_row = above[0]
        else:
            crest_row = crests[-1]
            crest_beyond = True
        crest_rows.append(crest_row)
        governing = [row for row in rows if row.dx_m == spacing
                     and row.crest_m == crest_row]
        slope_cap, past, read = gentle_rows(governing)
        slope_beyond = slope_beyond or past
        slope_rows.append(slope_cap)
        selected.extend(read)
        lower[spacing] = [row for height in crests if height < crest_row
                          for row in gentle_rows(
                              [row for row in rows if row.dx_m == spacing
                               and row.crest_m == height])[2]]
    if crest_beyond:
        beyond.append("crest")
    if slope_beyond:
        beyond.append("slope")
    # A table column is not evidence that every selected ridge was run
    # there. Missing cells must not borrow held steps from weaker winds.
    if any(not row.measured(i) for row in selected
           for i in range(wind_index + 1)) and "wind" not in beyond:
        beyond.append("wind")

    def held_through(index):
        values = []
        for row in selected:
            if any(not row.measured(i) for i in range(index + 1)):
                return None
            window = [row.stable[i] for i in range(index + 1)]
            if any(v is None for v in window):
                return None
            if window:
                values.append(min(window))
        return min(values) if values else None

    def tried_through(index, spacing=None):
        return min((row.tried_at(i, table.top) for row in selected
                    if spacing is None or row.dx_m == spacing
                    for i in range(index + 1) if row.measured(i)),
                   default=table.top)

    def stops(candidates, index):
        # Cell by cell: an entry shorter than the step tried there is a
        # stop seen, whatever the other rows were tried to.
        return [(row, i) for row in candidates for i in range(index + 1)
                if row.measured(i) and row.stable[i] is not None
                and row.stable[i] < row.tried_at(i, table.top)
                * (1.0 - 1e-9)]

    def stopped_through(index):
        found = [(row.stable[i], False, row.crest_m)
                 for row, i in stops(selected, index)]
        for spacing, candidates in lower.items():
            # A lower ridge's stop, only past every step this ground was
            # tried at, and never shorter than that step.
            reach = tried_through(index, spacing)
            found.extend((max(row.stable[i], reach), True, row.crest_m)
                         for row, i in stops(candidates, index)
                         if row.tried_at(i, table.top)
                         > reach * (1.0 + 1e-9))
        if not found:
            return None, False, None
        # The shortest entry; between equal ones, this crest's own.
        return min(found, key=lambda item: (item[0], item[1], -item[2]))

    per_km = held_through(wind_index)
    most_stable = per_km
    if per_km is None:
        # Past the strongest wind every governing row held: what they held
        # at the strongest wind they all did, the most stable pair measured.
        for index in range(wind_index - 1, -1, -1):
            most_stable = held_through(index)
            if most_stable is not None:
                break
    stopped, under_lower, stopped_crest = stopped_through(wind_index)
    adaptive = _adaptive_cells(selected, wind_index, table)
    return MapReading(
        per_km=per_km, dx_rows=tuple(dx_rows), crest_row=max(crest_rows),
        slope_row=max(slope_rows) if slope_rows else None,
        wind_row=winds[wind_index], beyond=tuple(beyond),
        most_stable_per_km=most_stable,
        top_per_km=tried_through(wind_index),
        stopped_per_km=stopped, stopped_crest_m=stopped_crest,
        stopped_under_a_lower_crest=under_lower, **adaptive)


def _adaptive_cells(selected, wind_index: int, table: StableStepMap
                    ) -> dict:
    """The adaptive clock's reading of the cells a reading takes.

    Cell by cell, as a stop is read on the fixed rows: a steeper domain
    reads every row a gentler one does and a stronger wind every wind a
    weaker one does, so adding a row or a wind can only lower the
    shortest entry under a stop and can only leave the cells unmeasured.
    Only the rows at the domain's own crest are read: every crest run on
    the adaptive clock was run over the same ladder, so a lower crest's
    cell was never tried past this crest's, and where this crest was not
    run the cap is the fixed map's reading alone.
    """
    if table.adaptive is None:
        return {}
    cells = [(row, index) for row in selected
             for index in range(wind_index + 1)]
    measured = [(row, index) for row, index in cells
                if row.adaptive_measured_at(index)]
    stops = []
    none_held = False
    for row, index in measured:
        entry, top = row.adaptive[index], row.adaptive_tried[index]
        if entry is None:
            # Every longest step tried stopped, down to the ladder's
            # shortest, which is then no step seen to hold: capping the
            # clock there would buy nothing measured.  The reading says so
            # (the domain may still stop) and is never read as held.
            none_held = True
        elif entry < top * (1.0 - 1e-9):
            stops.append(entry)
    every = bool(cells) and len(measured) == len(cells)
    entries = [row.adaptive[index] for row, index in measured]
    return {
        "adaptive_measured": every,
        "adaptive_per_km": (min(entries) if every and all(
            entry is not None for entry in entries) else None),
        "adaptive_top_per_km": (min(row.adaptive_tried[index]
                                    for row, index in measured)
                                if every else None),
        "adaptive_stopped_per_km": min(stops) if stops else None,
        "adaptive_none_held": none_held,
        "adaptive_none_held_below_per_km": (min(table.adaptive.ladder)
                                            if none_held else None)}


# ---------------------------------------------------------------------------
# Crest-level wind, read from the inputs a run integrates.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrestWind:
    """The strongest wind a domain's inputs carry at or below its crest.

    ``when`` names the input it came from ("start" or "boundary at +3 h"),
    ``source`` which domain's inputs, and ``height_m`` the height of the
    model level it sits on.
    """

    label: str
    crest_height_m: float
    wind_m_s: float
    when: str
    source: str
    height_m: float

    def receipt(self) -> dict:
        return {"crest_height_m": float(self.crest_height_m),
                "crest_level_wind_m_s": float(self.wind_m_s),
                "when": self.when, "source": self.source,
                "height_m": float(self.height_m)}


def crest_band(heights: np.ndarray, crest_height: float) -> np.ndarray:
    """Levels from the ground up to the first at or above the crest.

    ``heights`` is ``(nlevel, ...)`` ordered upward.  A level belongs when
    it is the lowest one or the level below it is under the crest height,
    so the column over the crest itself contributes its lowest level.
    """

    heights = np.asarray(heights, dtype=np.float64)
    under = np.asarray(heights[:-1] < float(crest_height))
    first = np.ones((1,) + heights.shape[1:], dtype=bool)
    return np.concatenate([first, under], axis=0)


def _strongest(speed, heights, crest_height):
    """``(speed, height)`` of the strongest wind in the crest band."""

    speed = np.asarray(speed, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    band = crest_band(heights, crest_height)
    masked = np.where(band & np.isfinite(speed), speed, -np.inf)
    index = int(np.argmax(masked))
    value = float(masked.flat[index])
    if not math.isfinite(value):
        return None
    return value, float(heights.flat[index])


def mass_heights(geopotential_full) -> np.ndarray:
    """Heights of the mass levels from full-level geopotential."""

    z = np.asarray(geopotential_full, dtype=np.float64) / _G
    return 0.5 * (z[:-1] + z[1:])


def _band_heights(php, phb, crest_height):
    """Each band level's mass-point heights, NaN where a column is past it.

    Walks up one level at a time, so no 3-D height array is held, and
    stops at the first level every column has already climbed past.
    """

    levels = []
    previous = None
    for k in range(int(phb.shape[0]) - 1):
        z = 0.5 * ((np.asarray(phb[k], dtype=np.float64)
                    + np.asarray(php[k], dtype=np.float64))
                   + (np.asarray(phb[k + 1], dtype=np.float64)
                      + np.asarray(php[k + 1], dtype=np.float64))) / _G
        if previous is None:
            inside = np.ones(z.shape, dtype=bool)
        else:
            inside = previous < float(crest_height)
            if not inside.any():
                break
        levels.append(np.where(inside, z, np.nan))
        previous = z
    return levels


def _strongest_over_levels(u, v, levels):
    """``(speed, height)`` of the strongest wind on the band levels."""

    best = None
    for k, heights in enumerate(levels):
        uk = np.asarray(u[k], dtype=np.float64)
        vk = np.asarray(v[k], dtype=np.float64)
        speed = np.hypot(0.5 * (uk[:, :-1] + uk[:, 1:]),
                         0.5 * (vk[:-1, :] + vk[1:, :]))
        masked = np.where(np.isfinite(heights) & np.isfinite(speed), speed,
                          -np.inf)
        index = int(np.argmax(masked))
        value = float(masked.flat[index])
        if math.isfinite(value) and (best is None or value > best[0]):
            best = (value, float(heights.flat[index]))
    return best


_START_KEYS = {"u": "state/u", "v": "state/v", "php": "state/php",
               "phb": "base/phb"}


@dataclass(frozen=True)
class StartWinds:
    """One domain's start state, read level by level when asked.

    ``fields`` returns ``u``, ``v``, ``php`` or ``phb`` by name; the arrays
    are loaded two at a time and dropped, so a large domain costs two of
    its 3-D fields at once, not a copy of the state.
    """

    source: str
    fields: Callable[[str], np.ndarray]
    #: How the local-face reading names this input in its receipt.
    kind: ClassVar[str] = "start state"

    @classmethod
    def from_fields(cls, source, u, v, php, phb):
        arrays = {"u": u, "v": v, "php": php, "phb": phb}
        return cls(source, arrays.__getitem__)

    def column_band(self, tops):
        """Per mass column, the strongest wind from the ground up to the
        first level at or above ``tops`` there; NaN where none
        (:func:`woof.terrain_clock_local.start_column_band`)."""
        from woof.terrain_clock_local import start_column_band

        return start_column_band(self, tops)

    def strongest(self, crest_height):
        levels = _band_heights(self.fields("php"), self.fields("phb"),
                               crest_height)
        found = _strongest_over_levels(self.fields("u"), self.fields("v"),
                                       levels)
        return None if found is None else (found[0], found[1], "start")


def _face_mass(mu, low):
    """Column mass on the faces of an x-side slab.

    ``mu`` is ``(ny, w)`` over the slab's mass columns in grid order.  On
    the low side the slab's faces are 0..w-1 and face 0 is the domain's
    outer face; on the high side they are the last w faces and the last is
    the outer one.  An outer face takes its boundary cell, as the coupled
    boundary values were built.
    """

    inner = 0.5 * (mu[:, :-1] + mu[:, 1:])
    if low:
        return np.concatenate([mu[:, :1], inner], axis=1)
    return np.concatenate([inner, mu[:, -1:]], axis=1)


def _row_mass(mu):
    """Column mass on the staggered rows across an x-side slab."""

    inner = 0.5 * (mu[:-1] + mu[1:])
    return np.concatenate([mu[:1], inner, mu[-1:]], axis=0)


@dataclass(frozen=True)
class BoundaryGeometry:
    """What uncoupling a root's boundary winds needs."""

    mub: np.ndarray
    phb: np.ndarray
    c1h: np.ndarray
    c2h: np.ndarray
    c1f: np.ndarray
    c2f: np.ndarray
    msfu: np.ndarray | None
    msfv: np.ndarray | None


@dataclass(frozen=True)
class BoundaryWinds:
    """A root's lateral boundary data over the forecast window.

    Read side by side on the boundary slabs themselves, so the cost is the
    frame and not the domain.
    """

    source: str
    boundaries: object
    geometry: BoundaryGeometry
    run_seconds: float
    kind: ClassVar[str] = "boundary data over the window"

    def instants(self):
        seen = []
        end = float(self.run_seconds)
        for interval in self.boundaries.intervals:
            start = float(interval.start_seconds)
            stop = float(interval.end_seconds)
            for t in (start, min(stop, end)):
                if start <= t <= stop and t <= end + 1e-6:
                    seen.append((interval, t - start, t))
            if stop >= end:
                break
        return seen

    def strongest(self, crest_height):
        best = None
        for interval, offset, at in self.instants():
            fields = interval.fields
            if not {"u", "v", "mu"} <= set(fields):
                continue
            self._refuse_another_grid(fields)
            for side in ("west", "east", "south", "north"):
                found = self._side(fields, side, offset, crest_height)
                if found is not None and (best is None or found[0] > best[0]):
                    best = (found[0], found[1], _boundary_when(at))
        return best

    def _refuse_another_grid(self, fields):
        """Refuse boundary tables or map factors off the base state's grid.

        Breakage it prevents: the reading pairs every boundary column with
        the base state's column mass, ground and map factors by position,
        so tables built on another grid stopped a prepared tree's preflight
        with a bare NumPy broadcast error that named neither the domain nor
        the mismatch.  Such boundaries cannot drive this domain either: the
        restore refuses them against its state
        (``woof.ingest.lateral_bc._validate_lateral_interval``), so the
        reading refuses them by name instead of guessing.
        """
        from woof.ingest.lateral_bc import _boundary_field_shape

        geo = self.geometry
        ny, nx = np.shape(geo.mub)
        nz = int(np.size(geo.c1h))
        cells = f"{nz} levels over {ny} x {nx} columns"

        def dims(shape):
            return " x ".join(str(int(n)) for n in shape)

        wanted = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
                  "mu": (1, ny, nx), "phi": (nz + 1, ny, nx)}
        for name, expected in wanted.items():
            if name not in fields:
                continue
            found = _boundary_field_shape(fields[name])[:3]
            if tuple(found) != expected:
                raise ValueError(
                    f"{self.source}'s lateral boundary data is for another "
                    f"grid than its base state: its {name} tables belong to "
                    f"a {dims(found)} field, where the base state's {cells} "
                    f"make it {dims(expected)}.  The time-step reading "
                    "pairs each boundary column with the base state's "
                    "column mass and ground, and the forecast's restore "
                    "refuses boundary tables of another shape, so neither "
                    "can use them; prepare the boundaries and the start "
                    "state together, in one preparation of this domain")
        for name, value, expected in (
                ("MAPFAC_U", geo.msfu, (ny, nx + 1)),
                ("MAPFAC_V", geo.msfv, (ny + 1, nx))):
            if value is not None and tuple(np.shape(value)) != expected:
                raise ValueError(
                    f"{self.source}'s {name} is {dims(np.shape(value))}, "
                    f"where its base state's {ny} x {nx} columns make it "
                    f"{dims(expected)}.  The time-step reading scales each "
                    "boundary wind by the map factor at its column, so it "
                    "cannot read map factors from another grid; prepare "
                    "the domain's static fields and its start state on one "
                    "grid")

    def column_band(self, tops):
        """Per mass column of the slabs, the strongest crest-band wind over
        the window, the band up to ``tops`` there; NaN off the slabs
        (:func:`woof.terrain_clock_local.boundary_column_band`)."""
        from woof.terrain_clock_local import boundary_column_band

        return boundary_column_band(self, tops)

    def _side(self, fields, side, offset, crest_height):
        speed, heights, _jj, _ii = self._side_cells(fields, side, offset)
        return _strongest(speed, heights, crest_height)

    def _side_cells(self, fields, side, offset):
        """One side's slab at ``offset``: ``(speed, heights, jj, ii)``, the
        uncoupled wind speed and mass-level heights on the slab's mass
        cells (``(nz, rows, cells)``) and each cell's domain row and
        column (``(rows, cells)``)."""
        from woof.ingest.lateral_bc import evaluate_boundary_side

        geo = self.geometry

        def value(name):
            part = getattr(fields[name], side)
            array = np.asarray(evaluate_boundary_side(part, offset)[0],
                               dtype=np.float64).reshape(part.value.shape)
            if side == "east":
                array = array[..., ::-1]
            elif side == "north":
                array = array[..., ::-1, :]
            return array

        across_x = side in ("west", "east")
        low = side in ("west", "south")
        # A south or north slab is an x-side slab with the two horizontal
        # axes exchanged, v and u trading places, and msfv and msfu.
        normal = value("u" if across_x else "v")
        along = value("v" if across_x else "u")
        mu = value("mu")[0]
        phi = value("phi") if "phi" in fields else None
        mub = np.asarray(geo.mub, dtype=np.float64)
        phb = np.asarray(geo.phb, dtype=np.float64)
        msf_normal = geo.msfu if across_x else geo.msfv
        msf_along = geo.msfv if across_x else geo.msfu
        if not across_x:
            normal = np.swapaxes(normal, -1, -2)
            along = np.swapaxes(along, -1, -2)
            mu = mu.T
            phi = None if phi is None else np.swapaxes(phi, -1, -2)
            mub = mub.T
            phb = np.swapaxes(phb, -1, -2)
            msf_normal = None if msf_normal is None else np.asarray(
                msf_normal).T
            msf_along = None if msf_along is None else np.asarray(
                msf_along).T
        w = mu.shape[-1]
        columns = slice(0, w) if low else slice(mub.shape[-1] - w, None)
        faces = (slice(0, w) if low
                 else slice(mub.shape[-1] + 1 - w, None))
        total = mub[:, columns] + mu
        c1h = np.asarray(geo.c1h, dtype=np.float64)[:, None, None]
        c2h = np.asarray(geo.c2h, dtype=np.float64)[:, None, None]
        normal = normal / (c1h * _face_mass(total, low)[None] + c2h)
        along = along / (c1h * _row_mass(total)[None] + c2h)
        if msf_normal is not None:
            normal = normal * np.asarray(msf_normal,
                                         dtype=np.float64)[:, faces][None]
        if msf_along is not None:
            along = along * np.asarray(msf_along,
                                       dtype=np.float64)[:, columns][None]
        if phi is not None:
            c1f = np.asarray(geo.c1f, dtype=np.float64)[:, None, None]
            c2f = np.asarray(geo.c2f, dtype=np.float64)[:, None, None]
            weight = c1f * total[None] + c2f
            # The top full level carries no column mass (c1f = c2f = 0), so
            # its coupled perturbation is zero and says nothing: the base
            # geopotential stands there.
            php = np.divide(phi, weight, out=np.zeros_like(phi),
                            where=weight != 0.0)
        else:
            php = 0.0
        heights = mass_heights(phb[:, :, columns] + php)
        # Mass cells with both of their faces in the slab: all but the one
        # next to the interior.
        cells = slice(0, w - 1) if low else slice(1, w)
        speed = np.hypot(0.5 * (normal[..., :-1] + normal[..., 1:]),
                         0.5 * (along[..., :-1, cells]
                                + along[..., 1:, cells]))
        # Domain indices of the cells: the slab's rows run along the side
        # (rows of the domain on west and east, its columns on south and
        # north), its cells inward from the domain's edge.
        rows = np.arange(speed.shape[-2])
        span = int(mub.shape[-1])
        inward = (np.arange(0, w - 1) if low
                  else np.arange(span - w + 1, span))
        along_idx, in_idx = np.meshgrid(rows, inward, indexing="ij")
        if across_x:
            jj, ii = along_idx, in_idx
        else:
            jj, ii = in_idx, along_idx
        return speed, heights[..., cells], jj, ii


@dataclass(frozen=True)
class MemoStartWinds(StartWinds):
    """A :class:`StartWinds` that reads its crest band once per crest height.

    For a reader asked the same question again and again: the streamed
    clock guard re-derives the clock at every boundary interval, and a
    start state does not change while the forecast runs.  Breakage this
    prevents: every hourly interval of a head-bound HRRR run re-read the
    four whole 3-D start fields (u, v, php, phb, each hashed on read) on the
    stepping thread, part of a 6.5-15.4 s stall per forecast hour on
    8 x RTX 5090 (test host, 2026-10-21).  The answer is the same call's answer.
    """

    memo: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def wrap(cls, start):
        return None if start is None else cls(start.source, start.fields)

    def strongest(self, crest_height):
        key = float(crest_height)
        if key not in self.memo:
            self.memo[key] = StartWinds.strongest(self, crest_height)
        return self.memo[key]


_UNREAD = object()


@dataclass(frozen=True)
class MemoBoundaryWinds(BoundaryWinds):
    """A :class:`BoundaryWinds` that reads each boundary instant once.

    ``memo`` is shared across the readings of one run, so a reading over
    intervals 0..k reads only interval k's slabs when 0..k-1 were read
    before.  Same instants, same sides, same order and the same
    strictly-greater rule as :meth:`BoundaryWinds.strongest`, so the
    strongest wind, its height and its "when" are the ones the full
    reading gives.  Breakage this prevents: the streamed clock guard read
    every interval so far at every new interval, so its cost grew with the
    forecast hour on the stepping thread (the stall grew from 7.3 s to
    15.4 s over 14 h on test host, 2026-10-21).
    """

    memo: dict = field(default_factory=dict, compare=False, repr=False)

    def strongest(self, crest_height):
        best = None
        for interval, offset, at in self.instants():
            fields = interval.fields
            if not {"u", "v", "mu"} <= set(fields):
                continue
            key = (float(interval.start_seconds), float(interval.end_seconds),
                   float(offset), float(crest_height))
            found = self.memo.get(key, _UNREAD)
            if found is _UNREAD:
                self._refuse_another_grid(fields)
                found = None
                for side in ("west", "east", "south", "north"):
                    side_found = self._side(fields, side, offset, crest_height)
                    if side_found is not None and (
                            found is None or side_found[0] > found[0]):
                        found = side_found
                self.memo[key] = found
            if found is not None and (best is None or found[0] > best[0]):
                best = (found[0], found[1], _boundary_when(at))
        return best


def _boundary_when(seconds: float) -> str:
    if float(seconds) <= 0.0:
        return "boundary at the start"
    hours = float(seconds) / 3600.0
    text = f"{hours:.2f}".rstrip("0").rstrip(".")
    return f"boundary at +{text} h"


def strongest_crest_wind(label: str, crest_height: float,
                         sources: Iterable) -> CrestWind | None:
    """The strongest crest-band wind across a domain's input sources."""

    best = None
    for source in sources:
        found = source.strongest(crest_height)
        if found is None:
            continue
        if best is None or found[0] > best[0]:
            best = (found[0], found[1], found[2], source.source)
    if best is None:
        return None
    return CrestWind(label=label, crest_height_m=float(crest_height),
                     wind_m_s=best[0], when=best[2], source=best[3],
                     height_m=best[1])


def _cache_array(reader, key):
    if key not in reader.arrays:
        return None
    return np.asarray(reader.read_array(key))


def start_winds_from_cache(reader, source: str) -> StartWinds | None:
    """Start winds from a prepared cache (every prepared door writes one)."""

    if not all(key in reader.arrays for key in _START_KEYS.values()):
        return None
    return StartWinds(source,
                      lambda name: reader.read_array(_START_KEYS[name]))


def boundary_geometry_from_cache(reader, static) -> BoundaryGeometry | None:
    arrays = {name: _cache_array(reader, key) for name, key in (
        ("mub", "base/mub"), ("phb", "base/phb"), ("c1h", "coord/c1h"),
        ("c2h", "coord/c2h"), ("c1f", "coord/c1f"), ("c2f", "coord/c2f"))}
    if any(value is None for value in arrays.values()):
        return None
    return BoundaryGeometry(
        msfu=_static(static, "MAPFAC_U"), msfv=_static(static, "MAPFAC_V"),
        **arrays)


def _static(static, name):
    if static is None:
        return None
    getter = getattr(static, "get", None)
    value = getter(name) if getter is not None else None
    return None if value is None else np.asarray(value, dtype=np.float64)


#: The boundary fields the wind reading uncouples, of the ones a prepared
#: cache carries.
_BOUNDARY_FIELDS = ("u", "v", "mu", "phi")


def cache_boundaries(reader):
    """The root's boundary winds from its prepared cache, or ``None``.

    Only the fields the reading needs are read.  A cache whose inventory
    holds no complete u, v and mu tables for every interval (a nested
    child's, which carries none) gives ``None``: the reading then rests on
    the start state, and the restore that follows is what judges the
    cache.
    """

    from woof.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries)
    from woof.ingest.prepared_cache import _reader_boundary_side

    lbc = reader.header.get("metadata", {}).get("lbc")
    if not isinstance(lbc, dict) or not isinstance(lbc.get("intervals"),
                                                   list):
        return None
    sides = ("west", "east", "south", "north")
    intervals = []
    for index, row in enumerate(lbc["intervals"]):
        names = [name for name in _BOUNDARY_FIELDS
                 if name in row.get("fields", ())]
        present = [name for name in names if all(
            f"lbc/{index}/{name}/{side}/{part}" in reader.arrays
            for side in sides for part in ("value", "tendency"))]
        if not {"u", "v", "mu"} <= set(present):
            return None
        fields = {name: FieldBoundary(**{
            side: _reader_boundary_side(reader, f"lbc/{index}/{name}/{side}")
            for side in sides}) for name in present}
        intervals.append(BoundaryInterval(
            float(row["start_seconds"]), float(row["end_seconds"]), fields))
    if not intervals:
        return None
    return LateralBoundaries(
        tuple(intervals), int(lbc["spec_bdy_width"]),
        int(lbc.get("spec_zone", 1)), int(lbc.get("relax_zone", 4)))


def start_winds_from_wrfinput(restored, source: str) -> StartWinds | None:
    """Start winds from a wrfinput file's own arrays (U, V, PH, PHB)."""

    raw = getattr(restored, "raw", None)
    names = {"u": "U", "v": "V", "php": "PH", "phb": "PHB"}
    if raw is None or not set(names.values()) <= set(raw):
        return None
    return StartWinds(source,
                      lambda name: np.squeeze(np.asarray(raw[names[name]])))


def boundary_geometry_from_wrfinput(restored, static
                                    ) -> BoundaryGeometry | None:
    raw = getattr(restored, "raw", None)
    names = ("MUB", "PHB", "C1H", "C2H", "C1F", "C2F")
    if raw is None or not set(names) <= set(raw):
        return None

    def field(name):
        return np.squeeze(np.asarray(raw[name], dtype=np.float64))
    return BoundaryGeometry(
        mub=field("MUB"), phb=field("PHB"), c1h=field("C1H"),
        c2h=field("C2H"), c1f=field("C1F"), c2f=field("C2F"),
        msfu=_static(static, "MAPFAC_U"), msfv=_static(static, "MAPFAC_V"))


# ---------------------------------------------------------------------------
# The decision per domain, and the experiment it runs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockAdaptation:
    """One domain's reading and the step and substeps it runs."""

    grid_id: int
    label: str
    slope: float
    crest: CrestWind | None
    dx: float
    configured_dt: Fraction
    configured_sound: int
    division: int
    time_step_sound: int
    #: What the map measured holding on the ground read, s/km: the
    #: reading's own ``per_km``, or the most stable pair measured where it
    #: holds no step at this wind.
    held_per_km: float | None
    reading: MapReading | None
    adaptive: bool = False
    #: On the adaptive clock, the interval it runs as configured:
    #: ``(start, min, max)`` in seconds, with WRF's ``-1`` fill-ins for
    #: min and max and the configured step for an unset start
    #: (:mod:`woof.core.adaptive_clock`).  ``None`` on a fixed clock.
    #: A pinned domain names it, since that clock (not a fixed
    #: configured step) is what it runs.
    adaptive_span: tuple | None = None
    ceiling: Fraction | None = None
    #: The longest step the reading lets the domain run, s/km, where a
    #: cell read saw a longer step stop: ``held_per_km`` or, where the
    #: ground read held its entry without being tried longer, the longer
    #: entry of the cell that saw the stop.  ``None`` where no cell did.
    limit_per_km: float | None = None
    #: On the adaptive clock, the limit its longest step was capped from,
    #: s/km, and where it comes from: ``"map"`` (``limit_per_km``),
    #: ``"adaptive held"`` (the longest step every cell read held on the
    #: adaptive clock) or ``"adaptive stop"`` (the entry under a longest
    #: step seen to stop on it).  :func:`_adaptive_cap`.
    cap_per_km: float | None = None
    cap_source: str = "map"
    adaptive_nest_lattice: bool = False
    #: The domain runs WRF's implicit-explicit vertical advection
    #: (``zadvect_implicit = 1``, :mod:`woof.core.ieva`).  Every entry of
    #: the map was measured with explicit vertical advection, so the
    #: reading is unchanged by it: the step and substeps read are the ones
    #: that held WITHOUT the option.  The option relaxes the vertical
    #: Courant limit, which is not what stops the map's ridges (24 probe
    #: cells held or stopped alike with it and without; module docstring),
    #: so the map never lengthens a step for it.
    implicit_vertical: bool = False
    #: ``RunConfig.terrain_clock = "pinned"``: the configured clock runs
    #: and the measured derivation is carried as :attr:`advice` only.
    pinned: bool = False
    #: Under ``pinned``, what the measured rule would have done:
    #: ``(division, time_step_sound, ceiling)``, the three fields the
    #: measured derivation writes.  ``None`` where the clock is measured.
    advice: tuple | None = None
    #: ``RunConfig.terrain_clock = "local_face"`` (the candidate,
    #: :mod:`woof.terrain_clock_local`): the face-by-face reading at the
    #: substep count the derivation chose
    #: (:class:`woof.terrain_clock_local.LocalFaces`), or ``None``.
    local: object | None = None
    #: Under ``local_face``, why the domain kept the domain-wide reading
    #: (a following nest's corridor, faces not readable with the substep
    #: rule's map factors, no wind over any face); ``None`` otherwise.
    local_note: str | None = None
    #: Under ``local_face``, the never-worse comparison with the shipped
    #: measured clock (:class:`woof.terrain_clock_local.NeverWorse`):
    #: where it is applied this adaptation IS the shipped clock's decision
    #: for the grid, and the face-by-face one it replaced is kept in it for
    #: the receipt and the run line.  ``None`` under every other clock.
    never_worse: object | None = None

    @property
    def dt(self) -> Fraction:
        return self.configured_dt / self.division

    @property
    def faces_read(self):
        """Under ``local_face``, the face-by-face reading made for this
        grid, whether it decided the grid or the never-worse rule put the
        shipped decision in its place; ``None`` where none was made."""
        never_worse = self.never_worse
        if never_worse is not None and never_worse.applied:
            return never_worse.local.local
        return self.local

    @property
    def adapted(self) -> bool:
        return (self.division != 1
                or self.time_step_sound != self.configured_sound
                or self.ceiling is not None)

    @property
    def advice_differs(self) -> bool:
        """Pinned, and the measured rule would have changed the clock."""
        if not self.pinned or self.advice is None:
            return False
        division, count, ceiling = self.advice
        return (int(division) != 1 or int(count) != self.configured_sound
                or ceiling is not None)

    @property
    def adaptive_unheld(self) -> bool:
        """On the adaptive clock, a cell read held no longest step tried
        there: its cap is not a step seen to hold on that clock."""
        return (self.adaptive and self.reading is not None
                and self.reading.adaptive_none_held)

    @property
    def beyond_measured(self) -> bool:
        return self.reading is not None and (
            self.reading.per_km is None or bool(self.reading.beyond)
            or self.adaptive_unheld)

    @property
    def unheld(self) -> bool:
        """The map holds no step at this wind over this ground."""
        return self.reading is not None and self.reading.per_km is None

    @property
    def status(self) -> str:
        if self.pinned:
            return "PINNED"
        if self.crest is None:
            return "NO_WIND_READING"
        # Ground where the adaptive clock held no longest step tried is
        # past what was measured to hold on it, whether or not the fixed
        # reading changed the clock.
        if (self.unheld or self.adaptive_unheld
                or (self.adapted and self.beyond_measured)):
            return "BEYOND_MEASURED"
        return "ADAPTED" if self.adapted else "AS_CONFIGURED"

    def _advice_runs(self) -> str:
        """What the measured rule would have run, in the words
        :meth:`_runs` uses for a changed domain."""
        division, count, ceiling = self.advice
        parts = []
        if int(division) != 1:
            parts.append(f"{_seconds(self.configured_dt / int(division))} "
                         f"steps instead of {_seconds(self.configured_dt)}")
        if int(count) != self.configured_sound:
            least = "at least " if self.adaptive else ""
            parts.append(f"{least}{int(count)} acoustic substeps per step "
                         f"instead of {self.configured_sound}")
        if ceiling is not None:
            parts.append(f"an adaptive step capped at "
                         f"{_seconds(Fraction(ceiling))}")
        return " and ".join(parts)

    def _pinned_runs(self) -> str:
        """What a pinned domain runs as configured.  On the adaptive clock
        that is the clock itself, not a fixed step: it starts at its
        configured step and follows its Courant number between its min
        and max, with the substep count from its live step.  Breakage it
        prevents: the HRRR route's pinned line said "runs 15 s steps on 4
        acoustic substeps" while the run took 1028 adaptive steps from
        11.85 s to 24 s (2.9.3 doors, okc3-hrrr on G12)."""
        if self.adaptive and self.adaptive_span is not None:
            start, low, high = self.adaptive_span
            return (f"its adaptive clock as configured, starting at "
                    f"{_seconds(start)} and following its Courant number "
                    f"between {_seconds(low)} and {_seconds(high)} on at "
                    f"least {self.configured_sound} acoustic substeps")
        return (f"{_seconds(self.configured_dt)} steps on "
                f"{self.configured_sound} acoustic substeps as configured")

    def pinned_sentence(self) -> str:
        """The line a pinned domain prints: what runs, and what the
        measured rule would have done instead (not applied)."""

        what = ("no wind reading" if self.crest is None else self._what())
        if self.advice_differs:
            return (f"time step: {self.label}'s clock is pinned, so it runs "
                    f"{self._pinned_runs()}; {what}, and the measured "
                    f"terrain clock would have run {self._advice_runs()} "
                    "(advice only, recorded in the receipt and not applied)")
        return (f"time step: {self.label}'s clock is pinned and runs "
                f"{self._pinned_runs()}; {what}, and the measured terrain "
                "clock holds that clock")

    def _what(self) -> str:
        crest = self.crest
        local = self.local
        if local is not None and local.governing is not None:
            g = local.governing
            if not g.wind_capped:
                capped = f" with the {local.margin:g} x wind margin"
            elif g.wind_read_m_s > local.domain_wind_m_s * (1.0 + 1e-9):
                # Read past the domain-wide reading: the input-wind floor
                # (terrain_clock_local.INPUT_WIND_FLOOR), not that reading.
                capped = (f", {g.wind_read_m_s / g.input_wind_m_s:.2g} x "
                          f"its own input wind (the floor; the domain-wide "
                          f"reading is {local.domain_wind_m_s:.0f} m/s)")
            else:
                capped = ", the domain-wide reading"
            return (f"{self.label}'s {local.faces} faces were read one by "
                    f"one (steepest slope {self.slope:.2f}); the face that "
                    f"sets its limit has slope {g.slope:.2f} at "
                    f"{g.axis}-face ({g.j}, {g.i}) under a "
                    f"{g.crest_m:.0f} m local crest, and its inputs carry "
                    f"{g.input_wind_m_s:.0f} m/s within "
                    f"{local.radius_m / 1000.0:g} km of it ({g.source}), "
                    f"read as {g.wind_read_m_s:.0f} m/s{capped}")
        return (f"{self.label}'s steepest terrain slope is {self.slope:.2f} "
                f"and the strongest wind its start and boundary data carry "
                f"up to its {crest.crest_height_m:.0f} m crest is "
                f"{crest.wind_m_s:.0f} m/s ({crest.when})")

    def _held(self) -> str:
        if self.held_per_km is None:
            return "no measured step"
        return "steps up to " + _seconds_float(
            self.held_per_km * self.dx / 1000.0)

    def _map_part(self) -> str:
        """What the measured map lets the domain run, and with how many
        substeps.  Where that step is not a step the ground read held
        (``limit_per_km`` past ``held_per_km``: some of that ground was
        never tried longer), the line names the step the clock applies
        first and the shorter range tried after it, so it never says the
        map holds a step shorter than the one the domain then runs.
        Breakage it prevents: NCAR's 2.5 km line said the map "holds steps
        up to 12.5 s there with 6 substeps" and then ran 15 s
        (CLOCK-CHECK-NCAR-2026-10-06/fix/step4/REVIEW.md, gap 3)."""
        limit, held = self.limit_per_km, self.held_per_km
        count = self.time_step_sound
        if limit is None or held is None or limit <= held * (1.0 + 1e-9):
            return (f"the measured map holds {self._held()} there with "
                    f"{count} substeps")
        reading = self.reading
        where = ("under a lower "
                 f"{reading.stopped_crest_m:.0f} m crest at this spacing"
                 if reading.stopped_under_a_lower_crest else
                 "on the gentler rows or weaker winds read with it")
        step = _seconds_float(limit * self.dx / 1000.0)
        return (f"the measured map lets {self.label} run steps up to {step} "
                f"there with {count} substeps (a step longer than {step} "
                f"stopped {where}; no cell read saw {step} or a shorter "
                "step stop, though some of that ground was tried no longer "
                f"than {_seconds_float(held * self.dx / 1000.0)})")

    def _adaptive_part(self) -> str:
        """What the adaptive clock's own entries say, where they set or
        bound the cap."""
        reading = self.reading
        if not self.adaptive or reading is None:
            return ""
        if reading.adaptive_none_held:
            return (", and on the adaptive clock every longest step tried "
                    "stopped on some of the ground read, down to "
                    + _seconds_float(reading.adaptive_none_held_below_per_km
                                     * self.dx / 1000.0))
        if self.ceiling is None or self.cap_per_km is None:
            return ""
        seconds = _seconds_float(self.cap_per_km * self.dx / 1000.0)
        if self.cap_source == "adaptive held":
            return (", and on the adaptive clock every cell read held a "
                    f"longest step of {seconds}, the longest tried there")
        if self.cap_source == "adaptive stop":
            return (", and on the adaptive clock a longest step longer "
                    f"than {seconds} stopped on the ground read")
        return ""

    def _runs(self) -> str:
        parts = []
        if self.division != 1:
            parts.append(f"{_seconds(self.dt)} steps instead of "
                         f"{_seconds(self.configured_dt)}")
        if self.time_step_sound != self.configured_sound:
            # On the adaptive clock the count is a floor under the one
            # the clock derives from its live step.
            least = "at least " if self.adaptive else ""
            parts.append(f"{least}{self.time_step_sound} acoustic substeps "
                         f"per step instead of {self.configured_sound}")
        if self.ceiling is not None:
            parts.append(f"an adaptive step capped at "
                         f"{_seconds(self.ceiling)}")
        return f"{self.label} runs " + " and ".join(parts)

    def _implicit_part(self) -> str:
        if not self.implicit_vertical:
            return ""
        return ("; the map was measured with explicit vertical advection, "
                "so under zadvect_implicit it holds what held without it")

    def sentence(self) -> str:
        """The plain line a run prints when this domain's clock changed."""

        return (f"time step: {self._what()}; "
                f"{self._map_part()}{self._adaptive_part()}"
                f"{self._implicit_part()}, so {self._runs()}")

    def beyond_sentence(self) -> str:
        """The line for a domain past the map's measured edge."""

        reading = self.reading
        if not self.adapted:
            if reading.per_km is not None and self.adaptive_unheld:
                # Held on the fixed rows, over ground where the adaptive
                # clock held no longest step tried.
                return self.adaptive_unheld_sentence()
            return (f"time step: {self._what()}; the measured map holds no "
                    f"step there at any substep count, so {self.label} runs "
                    "as configured and may still stop")
        if reading.per_km is None:
            # Past the strongest wind the map holds a step at: the most
            # stable pair it measured, which is not a held step here.
            return (f"time step: {self._what()}; the measured map holds no "
                    f"step at this wind, and the most stable pair it "
                    f"measured holds {self._held()} with "
                    f"{self.time_step_sound} substeps at a weaker wind, so "
                    f"{self._runs()}, and may still stop")
        if not reading.beyond:
            # Within the map's edges, on ground where the adaptive clock
            # held no longest step tried (:attr:`adaptive_unheld`).
            return (f"time step: {self._what()}; "
                    f"{self._map_part()}{self._adaptive_part()}, so "
                    f"{self._runs()}, and may still stop")
        return (f"time step: {self._what()}; "
                f"{self._map_part()}{self._adaptive_part()} but was "
                f"not measured this far out in "
                f"{', '.join(reading.beyond)}, so {self._runs()}, and may "
                "still stop")

    def adaptive_unheld_sentence(self) -> str:
        """The line for an adaptive domain whose own reading leaves its
        clock as configured, over ground where that clock held no longest
        step tried (:attr:`adaptive_unheld`).  Said even unchanged: its
        longest step is not one seen to hold on that clock there."""

        reading = self.reading
        out = ("" if not reading.beyond else
               f" but was not measured this far out in "
               f"{', '.join(reading.beyond)}")
        return (f"time step: {self._what()}; "
                f"{self._map_part()}{self._adaptive_part()}{out}, so "
                f"{self.label}'s adaptive clock keeps its own longest step "
                "and may still stop")

    def receipt(self) -> dict:
        row = {
            "grid_id": int(self.grid_id),
            "status": self.status,
            "steepest_slope": float(self.slope),
            "configured_dt_s": _rational(self.configured_dt),
            "dt_s": _rational(self.dt),
            "step_division": int(self.division),
            "configured_time_step_sound": int(self.configured_sound),
            "time_step_sound": int(self.time_step_sound),
            "held_s_per_km": self.held_per_km,
            "adaptive": bool(self.adaptive),
        }
        if self.adaptive_nest_lattice:
            row["adaptive_step_mode"] = "nest-lattice-cell-steps-v1"
        if self.implicit_vertical:
            # Written only when on, so every other receipt is unchanged.
            row["zadvect_implicit"] = 1
            row["map_vertical_advection"] = "explicit"
        if self.limit_per_km is not None:
            row["limit_s_per_km"] = self.limit_per_km
        if self.ceiling is not None:
            row["max_time_step_s"] = _rational(self.ceiling)
            row["max_time_step_from"] = self.cap_source
        if self.adaptive and self.time_step_sound != self.configured_sound:
            row["min_time_step_sound"] = int(self.time_step_sound)
        if self.pinned:
            # Written only under the pinned clock, so every measured
            # receipt reads as it did.  The advice is the measured
            # derivation's three fields, in the receipt's own spellings.
            division, count, ceiling = self.advice or (1, self.configured_sound, None)
            row["clock"] = "pinned"
            row["advice"] = {
                "applied": False,
                "dt_s": _rational(self.configured_dt / int(division)),
                "step_division": int(division),
                "time_step_sound": int(count),
                "max_time_step_s": (None if ceiling is None
                                    else _rational(Fraction(ceiling))),
                "differs": bool(self.advice_differs),
            }
        if self.crest is not None:
            row.update(self.crest.receipt())
        never_worse = self.never_worse
        faces_from = (never_worse.local if never_worse is not None
                      and never_worse.applied else self)
        if (faces_from.local is not None
                or faces_from.local_note is not None):
            # Written only under the local-face clock, so every measured
            # or pinned receipt reads as it did.  Where the never-worse
            # rule put the shipped decision in place, the faces are the
            # reading it replaced.
            row["clock"] = "local_face"
            row["local_faces"] = (
                {"kept_domain_wide_reading": faces_from.local_note}
                if faces_from.local is None
                else faces_from.local.receipt(self.dx))
        if never_worse is not None:
            row["clock"] = "local_face"
            row["never_worse"] = never_worse.receipt()
        if self.reading is not None:
            row["map_rows"] = {
                "dx_m": list(self.reading.dx_rows),
                "crest_m": self.reading.crest_row,
                "slope": self.reading.slope_row,
                "wind_m_s": self.reading.wind_row,
                "beyond": list(self.reading.beyond),
                "longest_step_tried_s": (self.reading.top_per_km
                                         * self.dx / 1000.0),
                "shortest_entry_under_a_stop_s": (
                    None if self.reading.stopped_per_km is None
                    else self.reading.stopped_per_km * self.dx / 1000.0),
                "stop_crest_m": self.reading.stopped_crest_m,
                "stop_under_a_lower_crest":
                    self.reading.stopped_under_a_lower_crest}
            if self.adaptive:
                km = self.dx / 1000.0
                reading = self.reading

                def seconds(value):
                    return None if value is None else value * km

                row["map_rows"]["adaptive_clock"] = {
                    "every_cell_measured": bool(reading.adaptive_measured),
                    "longest_step_held_s": seconds(reading.adaptive_per_km),
                    "longest_step_tried_s": seconds(
                        reading.adaptive_top_per_km),
                    "shortest_entry_under_a_stop_s": seconds(
                        reading.adaptive_stopped_per_km),
                    "a_cell_held_none_down_to_s": seconds(
                        reading.adaptive_none_held_below_per_km)}
        return row


def _seconds(value: Fraction) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return f"{value.numerator} s"
    return _seconds_float(float(value))


def _seconds_float(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"{text} s"


def _rational(value: Fraction) -> dict:
    value = Fraction(value)
    return {"seconds": float(value), "numerator": value.numerator,
            "denominator": value.denominator}


def _held_step(reading: MapReading) -> float | None:
    """The longest step the reading lets a domain run, s/km: ``None`` for
    no limit, where every cell read held every step tried there (past the
    longest step measured on that ground the adaptive clock's own limits
    govern, docs/public/CONFIGURATION.md); else the shortest entry of a
    cell that saw a longer step stop.  Only for a reading that holds a
    step (``per_km`` set)."""
    return None if reading.held_everything_tried else reading.stopped_per_km


def _holds(reading: MapReading, configured_per_km) -> bool:
    """No cell read saw the configured step, or a shorter one, stop."""
    if reading.per_km is None:
        return False
    limit = _held_step(reading)
    return limit is None or configured_per_km <= limit * (1.0 + 1e-9)


def derive_clock(grid_id: int, run, dt: Fraction, slope: float,
                 crest: CrestWind | None, *, label: str | None = None,
                 table: StableStepMap | None = None,
                 local=None) -> ClockAdaptation:
    """The step and substep count one domain runs.

    ``dt`` is the domain's configured step as an exact rational.  Without a
    wind reading the domain runs as configured.  Otherwise the configured
    pair stands when the map holds it; else the smallest whole division of
    the step the map holds, at the fewest substeps (the configured count or
    six) that hold it.  Under the adaptive clock the held step is a ceiling
    and the configured step, the clock's first, is divided only when it is
    above that ceiling; the count read is the one that clock takes at its
    shortest steps (:func:`woof.core.adaptive_clock.least_sound_steps`),
    since it derives its count from the live step and gives four at every
    short one, and a larger count it needs becomes the floor under it.

    Under ``run.terrain_clock = "pinned"`` the same reading is taken and
    the same derivation made, but it is carried as the adaptation's
    ``advice`` and the domain runs its configured step and count: division
    1, no ceiling, ``time_step_sound`` as configured.  The receipt row
    then says ``clock = "pinned"`` and what the measured rule would have
    done, so a pinned run never hides the map's verdict; it declines to
    apply it.
    """

    mode = str(getattr(run, "terrain_clock", "measured"))
    if mode == "local_face" and crest is not None and hasattr(
            local, "reading"):
        from woof.terrain_clock_local import derive_local

        return derive_local(grid_id, run, dt, slope, crest, local,
                            label=label)
    measured = _derive_measured(grid_id, run, dt, slope, crest,
                                label=label, table=table)
    if mode == "local_face":
        # The candidate could not read this domain face by face: it runs
        # the shipped domain-wide reading, and its receipt says why.
        return replace(measured, local_note=(
            str(local) if isinstance(local, str) else
            "no wind reading" if crest is None else
            "no face reading was made for this domain"))
    if mode != "pinned":
        return measured
    return replace(
        measured, division=1, time_step_sound=measured.configured_sound,
        ceiling=None, pinned=True,
        advice=(int(measured.division), int(measured.time_step_sound),
                measured.ceiling))


def _derive_measured(grid_id: int, run, dt: Fraction, slope: float,
                     crest: CrestWind | None, *, label: str | None = None,
                     table: StableStepMap | None = None,
                     read: Callable[[int], MapReading] | None = None
                     ) -> ClockAdaptation:
    """:func:`derive_clock`'s measured derivation, whatever the mode.

    ``read(count)``, where given, is the domain's reading at a substep
    count in place of the one triple (:mod:`woof.terrain_clock_local`);
    ``table`` is then the map it was read on."""

    from woof.core.adaptive_clock import least_sound_steps

    table = measured_map() if table is None else table
    label = label or f"d{int(grid_id):02d}"
    dt = Fraction(dt)
    dx = float(run.dx)
    configured_sound = least_sound_steps(run)
    adaptive = bool(getattr(run, "use_adaptive_time_step", False))
    span = None
    if adaptive:
        from woof.core.adaptive_clock import wrf_default_clamps

        # The interval the live clock runs (the adaptive clock's own
        # resolution): an explicit starting_time_step, else the configured
        # step; min and max as set, else WRF's 3 and 8 s per km.
        _start, wrf_max, wrf_min = wrf_default_clamps(run.dx, run.dy)
        whole = int(getattr(run, "starting_time_step", -1))
        den = int(getattr(run, "starting_time_step_den", 0))
        start = (None if whole == -1 and den == 0 else
                 Fraction(whole) if den == 0 else Fraction(whole, den))
        low, high = _adaptive_lower(run), _adaptive_upper(run)
        span = (dt if start is None else start,
                Fraction(wrf_min) if low is None else low,
                Fraction(wrf_max) if high is None else high)
    base = dict(grid_id=int(grid_id), label=label, slope=float(slope),
                crest=crest, dx=dx, configured_dt=dt,
                configured_sound=configured_sound, adaptive=adaptive,
                adaptive_span=span,
                adaptive_nest_lattice=bool(getattr(
                    run, "adaptive_nest_lattice", False)),
                implicit_vertical=int(getattr(
                    run, "zadvect_implicit", 0) or 0) > 0)
    if crest is None:
        return ClockAdaptation(division=1, time_step_sound=configured_sound,
                               held_per_km=None, reading=None, **base)
    km = dx / 1000.0
    configured_per_km = float(dt) / km
    upper = None
    if adaptive:
        from woof.core.adaptive_clock import wrf_default_clamps
        # The longest step the adaptive clock takes: its max_time_step, or
        # WRF's 8 * dx fill-in where that is -1.
        upper = _adaptive_upper(run)
        if upper is None:
            upper = Fraction(wrf_default_clamps(run.dx, run.dy)[1])
        configured_per_km = max(configured_per_km, float(upper) / km)
    def reading_for(count):
        return (read(count) if read is not None else
                read_map(dx, crest.crest_height_m, slope,
                         crest.wind_m_s, count, table))

    own = reading_for(configured_sound)
    if _holds(own, configured_per_km) and not own.beyond:
        ceiling, cap, source = (_adaptive_ceiling(own, _held_step(own), run,
                                                  table, dx, upper)
                                if adaptive else (None, None, "map"))
        return ClockAdaptation(division=1, time_step_sound=configured_sound,
                               held_per_km=own.per_km, reading=own,
                               ceiling=ceiling, cap_per_km=cap,
                               cap_source=source, **base)
    # The configured clock already returned if it holds. Reading an unused
    # alternative here would sort every terrain face a second time at startup.
    # Keep every candidate when choosing a replacement clock.
    readings = {configured_sound: own}
    if configured_sound < 6:
        readings[6] = reading_for(6)
    choices = []
    for count, reading in readings.items():
        if reading.per_km is None:
            continue
        limit = _held_step(reading)
        if limit is None:
            choices.append((1, count, reading, None))
            continue
        need = float(dt) / km / limit
        division = max(1, math.ceil(need - 1e-9))
        choices.append((division, count, reading, limit))
    if choices:
        division, count, reading, limit = min(
            choices, key=lambda item: (item[0], item[1]))
        held = reading.per_km
    else:
        # Past every stable measurement: the most stable pair measured.
        count = max(readings)
        reading = readings[count]
        held = limit = reading.most_stable_per_km
        division = (1 if held is None else
                    max(1, math.ceil(float(dt) / km / held - 1e-9)))
    ceiling, cap, source = (_adaptive_ceiling(reading, limit, run, table,
                                              dx, upper)
                            if adaptive else (None, None, "map"))
    return ClockAdaptation(division=int(division), time_step_sound=int(count),
                           held_per_km=held, reading=reading,
                           ceiling=ceiling, limit_per_km=limit,
                           cap_per_km=cap, cap_source=source, **base)


def _adaptive_cap(reading: MapReading, limit: float | None, run,
                  table: StableStepMap) -> tuple[float | None, str]:
    """The limit, s/km, on the adaptive clock's longest step, and what
    gives it: ``"map"`` (the fixed-step reading's ``limit``, ``None`` for
    none), ``"adaptive held"`` or ``"adaptive stop"``.

    The fixed-step limit is a step seen to stop as a FIXED step.  The
    adaptive clock shortens its step on the CFL it measures and derives
    its substep count from the live step, so where every cell read was
    run on that clock and held every longest step tried there, it is
    capped at that longest step instead (never below the fixed limit),
    for a clock no faster than the one measured
    (:meth:`AdaptiveMeasurement.covers`) on ground within the map's edges
    (a reading past one keeps the fixed limit).  Where a cell run on it saw a
    longest step stop, the entry under that stop caps it whatever the
    fixed reading says; a clock faster than the one measured is no more
    stable, so it reads that stop too.  Adding a row or a wind to a
    reading can only lower either answer (:func:`_adaptive_cells`).
    """
    measured = table.adaptive
    if measured is None:
        return limit, "map"
    cap, source = limit, "map"
    # Only ground within the map's edges is lifted.  A domain steeper than
    # the steepest row at its crest (or past any other edge) reads that
    # row's cells, but no ridge that steep was run on the adaptive clock:
    # lifting it would run a 3 km domain of slope 0.48 under a 4.5 km
    # crest at 45 s, uncapped and reported as configured, from a 0.47 row.
    if (limit is not None and not reading.beyond
            and reading.adaptive_held_everything_tried
            and measured.covers(run)
            and reading.adaptive_per_km > limit * (1.0 + 1e-9)):
        cap, source = reading.adaptive_per_km, "adaptive held"
    stopped = reading.adaptive_stopped_per_km
    if stopped is not None and (cap is None
                                or stopped < cap * (1.0 - 1e-9)):
        cap, source = stopped, "adaptive stop"
    return cap, source


def _adaptive_ceiling(reading: MapReading, limit: float | None, run,
                      table: StableStepMap, dx: float, upper: Fraction
                      ) -> tuple[Fraction | None, float | None, str]:
    """The ``max_time_step`` written for an adaptive domain, the limit it
    comes from (s/km) and that limit's source (:func:`_adaptive_cap`)."""
    cap, source = _adaptive_cap(reading, limit, run, table)
    if cap is None:
        return None, None, source
    # A cell read saw a longer step stop (or, past the strongest wind the
    # map holds a step at, every step): that limit caps the adaptive
    # clock.  On its hundredth-of-a-second lattice, rounded down so the
    # cap never sits above it.  A cap only ever shortens the step: one at
    # or above the clock's own longest step (its max_time_step, or the
    # 8 * dx fill-in) is not written, because a limit read from a stop
    # seen above ground tried no further than it held would otherwise
    # raise max_time_step past anything measured there.
    ceiling = Fraction(math.floor(cap * dx / 1000.0 * 100.0 + 1e-9), 100)
    if upper <= ceiling:
        ceiling = None
    return ceiling, cap, source


def _adaptive_upper(run):
    from woof.config import _adaptive_interval

    return _adaptive_interval(run.max_time_step, run.max_time_step_den,
                              "max_time_step")


def _adaptive_lower(run):
    from woof.config import _adaptive_interval

    return _adaptive_interval(run.min_time_step, run.min_time_step_den,
                              "min_time_step")


def _divisions(exp, wanted: Mapping[int, int]) -> dict[int, tuple[int, int]]:
    """Per domain ``(division, parent_time_step_ratio)``.

    A root's step is divided by what it needs.  A nest's division is the
    smallest one at least its own need whose ratio to its parent's
    division keeps its step ratio whole, so a parent cut deeply enough
    can leave its nest's step where it was.
    """

    result: dict[int, tuple[int, int]] = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        need = max(1, int(wanted.get(gid, 1)))
        if int(dc.parent_id) == 0:
            result[gid] = (need, int(dc.parent_time_step_ratio))
            continue
        parent_division = result[int(dc.parent_id)][0]
        ratio = int(dc.parent_time_step_ratio)
        division = need
        while (ratio * division) % parent_division:
            division += 1
        result[gid] = (division, ratio * division // parent_division)
    return result


def retime_experiment(exp, divisions: Mapping[int, int],
                      sound: Mapping[int, int],
                      ceilings: Mapping[int, Fraction] | None = None):
    """The experiment with each domain's step divided and count raised.

    Returns ``(experiment, {grid_id: (division, ratio)})``.  The root's
    step is written as WRF writes it (whole seconds plus a fraction) and
    every ``run.dt`` is rebuilt as the chained single-precision value the
    clock validates.  On the adaptive clock a raised count is also written
    as ``min_time_step_sound``, or the clock, which derives its count from
    the live step, would put it back to four.  Domains with no change keep
    every field.
    """

    from woof.acoustic_adaptation import adapted_run

    plan = _divisions(exp, divisions)
    ceilings = dict(ceilings or {})
    by_id = {}
    old_by_id = {int(dc.grid_id): np.float32(dc.run.dt)
                 for dc in exp.domains}
    domains = []
    for dc in exp.domains:
        gid = int(dc.grid_id)
        division, ratio = plan[gid]
        run = dc.run
        if int(dc.parent_id) == 0:
            dt = exp.dt_exact(gid) / division
            whole = dt.numerator // dt.denominator
            fraction = dt - whole
            fields = dict(time_step=int(whole),
                          time_step_fract_num=int(fraction.numerator),
                          time_step_fract_den=int(fraction.denominator))
            if division == 1:
                fields = {}
                dt32 = np.float32(run.dt)
            else:
                dt32 = (np.float32(whole) + np.float32(fraction.numerator)
                        / np.float32(fraction.denominator))
        else:
            fields = {}
            parent_id = int(dc.parent_id)
            if ratio != int(dc.parent_time_step_ratio):
                fields["parent_time_step_ratio"] = int(ratio)
            if (not fields and by_id[parent_id] == old_by_id[parent_id]):
                dt32 = np.float32(run.dt)
            else:
                dt32 = by_id[parent_id] / np.float32(ratio)
        run_fields = {}
        if float(dt32) != float(run.dt):
            run_fields["dt"] = float(dt32)
        counted = adapted_run(run, int(sound.get(gid, 0)))
        for name in ("time_step_sound", "min_time_step_sound"):
            if getattr(counted, name, None) != getattr(run, name, None):
                run_fields[name] = getattr(counted, name)
        ceiling = ceilings.get(gid)
        if ceiling is not None:
            ceiling = Fraction(ceiling)
            run_fields.update(_interval_fields("max_time_step", ceiling))
            lower = _adaptive_lower(run)
            if lower is None:
                from woof.core.adaptive_clock import wrf_default_clamps
                lower = Fraction(wrf_default_clamps(run.dx, run.dy)[2])
            if lower > ceiling:
                run_fields.update(_interval_fields("min_time_step", ceiling))
            start = _interval_start(run)
            if start is not None and start > ceiling:
                run_fields.update(_interval_fields("starting_time_step",
                                                   ceiling))
        if run_fields:
            fields["run"] = replace(run, **run_fields)
        by_id[gid] = dt32
        domains.append(replace(dc, **fields) if fields else dc)
    if all(new is old for new, old in zip(domains, exp.domains)):
        return exp, plan
    return replace(exp, domains=tuple(domains)), plan


def _interval_start(run):
    from woof.config import _adaptive_interval

    return _adaptive_interval(run.starting_time_step,
                              run.starting_time_step_den,
                              "starting_time_step")


def _interval_fields(name: str, value: Fraction) -> dict:
    value = Fraction(value)
    if value.denominator == 1:
        return {name: int(value.numerator), f"{name}_den": 0}
    return {name: int(value.numerator), f"{name}_den": int(value.denominator)}


def adapt_experiment_clock(
        exp, slopes: Mapping[int, float], winds: Mapping[int, CrestWind],
        *, announce: Callable[[str], None] | None = None,
        caution: Callable[[str], None] | None = None,
        table: StableStepMap | None = None,
        local: Mapping[int, object] | None = None):
    """Return ``(experiment, adaptations)`` with each read domain's clock.

    ``slopes`` maps ``grid_id`` to the steepest slope the substep rule read
    and ``winds`` to the domain's crest-level wind.  A domain missing
    either runs as configured.  ``announce`` receives one line per changed
    domain, ``caution`` the line for a changed domain past the map's edge
    and, changed or not, for one over ground the map holds no step for or
    where the adaptive clock held no longest step tried.
    """

    adaptations = []
    for dc in exp.domains:
        gid = int(dc.grid_id)
        if gid not in slopes:
            continue
        wind = winds.get(gid)
        # Without a wind reading nothing changes, and the step is only
        # recorded, so an experiment without the rational clock still reads.
        dt = (exp.dt_exact(gid) if wind is not None else
              Fraction(float(getattr(dc.run, "dt", 0.0))))
        adaptations.append(derive_clock(
            gid, dc.run, dt, float(slopes[gid]), wind, table=table,
            local=(local or {}).get(gid)))
    changed = [a for a in adaptations if a.adapted]
    adapted, final = exp, list(adaptations)
    if changed:
        adapted, plan = retime_experiment(
            exp, {a.grid_id: a.division for a in changed},
            {a.grid_id: a.time_step_sound for a in changed},
            {a.grid_id: a.ceiling for a in changed if a.ceiling is not None})
        final = []
        for adaptation in adaptations:
            division, _ratio = plan[adaptation.grid_id]
            if division != adaptation.division:
                adaptation = replace(adaptation, division=int(division))
            final.append(adaptation)
    # One line per changed domain.  A domain the map holds can still take a
    # finer step from its parent's division, and says so in its own words;
    # one over ground and wind the map holds no step for, or where the
    # adaptive clock held no longest step tried, says so even when its
    # clock stays as configured.
    changed_ids = {a.grid_id for a in changed}
    for adaptation in final:
        if adaptation.local_note is not None and announce is not None:
            announce(f"time step: {adaptation.label} keeps the shipped "
                     "domain-wide terrain clock: " + adaptation.local_note)
        if adaptation.pinned:
            # A pinned domain never changes, so it is never in ``changed``;
            # it says what it runs and what the map would have done, and
            # it says so as a caution where that differs, because the
            # map's verdict is a stop seen on measured ground.
            say = caution if adaptation.advice_differs else announce
            if say is not None:
                say(adaptation.pinned_sentence())
            continue
        never_worse = adaptation.never_worse
        if (never_worse is not None and never_worse.applied
                and announce is not None):
            # Said before the shipped decision's own line, changed or not:
            # the face-by-face reading did not decide this grid.
            announce(never_worse.sentence(adaptation.label))
        if adaptation.grid_id in changed_ids or adaptation.unheld:
            if adaptation.beyond_measured:
                if caution is not None:
                    caution(adaptation.beyond_sentence())
            elif announce is not None:
                announce(adaptation.sentence())
            continue
        if adaptation.division != 1 and announce is not None:
            announce(
                f"time step: {adaptation.label} runs "
                f"{_seconds(adaptation.dt)} steps instead of "
                f"{_seconds(adaptation.configured_dt)} because its parent's "
                "step was divided and a nest's step divides its parent's")
        if adaptation.adaptive_unheld and caution is not None:
            caution(adaptation.adaptive_unheld_sentence())
    return adapted, tuple(final)


#: The mechanism, printed under ``--explain``.
_WHY = (
    "Under a strong wind at crest height, flow forced over steep ground "
    "moves farther per long step than the split-explicit step can hold, "
    "and more acoustic substeps do not help once the long step itself is "
    "too long.  The engine's measured map (woof/terrain_clock.py and "
    "woof/terrain_clock_map.json) gives, for each grid spacing, crest "
    "height, slope, crest-level wind and substep count, the longest step "
    "that held; the run takes the smallest whole division of its step "
    "that the map holds.")


def adapt_experiment_clock_to_terrain(exp, slopes, winds, *, local=None):
    """:func:`adapt_experiment_clock` in the run's own voice."""

    from woof.explain import warn

    return adapt_experiment_clock(
        exp, slopes, winds, local=local,
        announce=lambda sentence: warn(sentence, _WHY),
        caution=lambda sentence: warn(sentence, _WHY))


def clock_receipt(adaptations) -> dict:
    """The derivation for the run document: every domain read, changed or
    not."""

    table = measured_map()
    local = {}
    if any(getattr(a, "local", None) is not None
           or getattr(a, "local_note", None) is not None
           for a in adaptations):
        # Written only under the local-face clock (the default from 2.8.8).
        from woof.terrain_clock_local import (ADAPTIVE_ROWS_PATH,
                                               CANDIDATE_MAP_PATH,
                                               INPUT_WIND_FLOOR, SCHEMA,
                                               WIND_MARGIN,
                                               adaptive_document,
                                               candidate_document)
        document = candidate_document()
        local = {"local_face": {
            "schema": SCHEMA, "candidate_rows": CANDIDATE_MAP_PATH.name,
            "criterion": document["criterion"],
            "seconds": document["seconds"],
            "rows": len(document["rows"]),
            "cells_shortened_by_12_h_runs": len(document.get(
                "long_probes", {}).get("patched", ())),
            "adaptive_rows": ADAPTIVE_ROWS_PATH.name,
            "adaptive_rows_seconds": adaptive_document()["adaptive"][
                "seconds"],
            "wind_margin": WIND_MARGIN,
            "input_wind_floor": INPUT_WIND_FLOOR}}
    return {
        "schema": TERRAIN_CLOCK_SCHEMA,
        **local,
        "domains": [adaptation.receipt() for adaptation in adaptations],
        "map": {"path": MAP_PATH.name, "winds_m_s": list(table.winds),
                "ladder_s_per_km": list(table.ladder),
                "longest_step_tried_s": {
                    f"{dx:g}": seconds
                    for dx, seconds in table.measured_range().items()},
                "seconds": table.seconds, "rows": len(table.rows)},
    }


def crest_heights(terrain_by_grid_id: Mapping[int, object]
                  ) -> dict[int, float]:
    """Each domain's crest height: the highest ground it can touch."""

    return {int(gid): float(np.nanmax(np.asarray(terrain, dtype=np.float64)))
            for gid, terrain in terrain_by_grid_id.items()}


def tree_crest_winds(exp, crests: Mapping[int, float],
                     sources_by_grid_id: Mapping[int, Sequence],
                     root_boundary_sources: Sequence = ()
                     ) -> dict[int, CrestWind]:
    """Each domain's crest-level wind across its own and its ancestors'
    inputs.

    A nest's boundary data is its parent's forecast, which no input holds
    before the run, so a nest is read over its own start state, every
    ancestor's start state and the root's boundary data, each at the
    nest's own crest height.
    """

    by_id = {int(dc.grid_id): dc for dc in exp.domains}
    winds = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        if gid not in crests:
            continue
        sources = list(sources_by_grid_id.get(gid, ()))
        parent = int(getattr(dc, "parent_id", 0))
        while parent != 0:
            sources.extend(sources_by_grid_id.get(parent, ()))
            parent = int(getattr(by_id[parent], "parent_id", 0))
        sources.extend(root_boundary_sources)
        wind = strongest_crest_wind(f"d{gid:02d}", crests[gid], sources)
        if wind is not None:
            winds[gid] = wind
    return winds


# ---------------------------------------------------------------------------
# The doors.
# ---------------------------------------------------------------------------


def slopes_from_acoustics(acoustic) -> dict[int, float]:
    """The steepest slope each domain's substep derivation read."""

    return {int(a.grid_id): float(a.reading.slope) for a in acoustic}


def clock_for_domains(exp, acoustic, *, statics: Mapping[int, object],
                      starts: Mapping[int, object],
                      boundary: BoundaryWinds | None = None,
                      corridors: Mapping[int, object] | None = None,
                      announce: bool = True, local_cache=None):
    """Every prepared door's derivation: ``(experiment, adaptations)``.

    ``statics`` maps ``grid_id`` to the static fields each domain runs on
    (``HGT_M`` gives its crest), ``starts`` to its :class:`StartWinds`,
    ``boundary`` is the root's :class:`BoundaryWinds` and ``corridors``
    maps a following nest to the terrain of the corridor it can reach.
    """

    slopes = slopes_from_acoustics(acoustic)
    terrain = {}
    for gid, static in statics.items():
        field = _static(static, "HGT_M")
        if field is not None:
            terrain[int(gid)] = field
    crests = crest_heights(terrain)
    for gid, reach in (corridors or {}).items():
        if reach is None:
            continue
        top = float(np.nanmax(np.asarray(reach, dtype=np.float64)))
        crests[int(gid)] = max(crests.get(int(gid), top), top)
    sources = {int(gid): [start] for gid, start in starts.items()
               if start is not None}
    winds = tree_crest_winds(exp, crests, sources,
                             [] if boundary is None else [boundary])
    local = None
    if any(str(getattr(dc.run, "terrain_clock", "measured")) == "local_face"
           for dc in exp.domains):
        from woof.terrain_clock_local import local_readings

        reader = local_readings if local_cache is None else local_cache.readings
        local = reader(exp, slopes=slopes, winds=winds,
                               statics=statics, sources=sources,
                               boundary=boundary, corridors=corridors)
    if announce:
        return adapt_experiment_clock_to_terrain(exp, slopes, winds,
                                                 local=local)
    return adapt_experiment_clock(exp, slopes, winds, local=local)


def clock_for_prepared_cache(exp, acoustic, *, readers, statics,
                             boundaries=None, corridors=None):
    """The derivation on doors whose domains carry prepared caches.

    ``readers`` maps ``grid_id`` to each domain's
    :class:`woof.ingest.prepared_cache.PreparedCacheReader`; the root's
    boundary data is ``boundaries`` when the door holds it apart (met_em),
    else the one its own cache carries.
    """

    root = int(exp.domains[0].grid_id)
    starts = {int(gid): start_winds_from_cache(reader, f"d{int(gid):02d}")
              for gid, reader in readers.items()}
    reader = readers.get(root)
    if boundaries is None and reader is not None:
        boundaries = cache_boundaries(reader)
    boundary = None
    if boundaries is not None and reader is not None:
        geometry = boundary_geometry_from_cache(reader, statics.get(root))
        if geometry is not None:
            boundary = BoundaryWinds(f"d{root:02d}", boundaries, geometry,
                                     float(exp.run_seconds))
    return clock_for_domains(exp, acoustic, statics=statics, starts=starts,
                             boundary=boundary, corridors=corridors)


def clock_for_wrfinput(exp, acoustic, *, restored, statics, boundaries,
                       corridors=None):
    """The derivation on the wrfinput door, from the files' own arrays."""

    root = int(exp.domains[0].grid_id)
    starts = {int(gid): start_winds_from_wrfinput(item, f"d{int(gid):02d}")
              for gid, item in restored.items()}
    boundary = None
    if boundaries is not None and root in restored:
        geometry = boundary_geometry_from_wrfinput(restored[root],
                                                   statics.get(root))
        if geometry is not None:
            boundary = BoundaryWinds(f"d{root:02d}", boundaries, geometry,
                                     float(exp.run_seconds))
    return clock_for_domains(exp, acoustic, statics=statics, starts=starts,
                             boundary=boundary, corridors=corridors)


@dataclass(frozen=True)
class SnapshotWinds:
    """Decoded forcing on its own grid, over one domain's footprint.

    The ``woof run`` door reads the forcing before it prepares anything,
    because the prepared state already carries the step its physics was
    built for.  The footprint is every source column inside the domain's
    extent, one source spacing wider.
    """

    source: str
    snapshots: tuple
    latitude: np.ndarray
    longitude: np.ndarray
    start_time: object
    kind: ClassVar[str] = "forcing over the window"

    def column_band(self, tops):
        """Per domain column, the strongest crest-band wind the forcing
        carries over the window at the source columns around it.

        Each domain column reads the (up to) four source columns that
        bracket it on the source grid, and each source column's band runs
        up to the highest ``tops`` of the domain columns that read it, so
        no domain column reads a shallower band than its own top.  NaN
        where no snapshot covers a column."""

        from woof.ingest.horiz import source_coordinate_transform

        tops = np.asarray(tops, dtype=np.float64)
        best = np.full(tops.shape, -np.inf)
        for snapshot in self.snapshots:
            fields = snapshot.fields
            if not {"UU", "VV", "GHT"} <= set(fields):
                continue
            transform, projected = source_coordinate_transform(snapshot)
            y, x = transform(np.asarray(self.latitude, dtype=np.float64),
                             np.asarray(self.longitude, dtype=np.float64))
            y = np.asarray(y, dtype=np.float64)
            x = np.asarray(x, dtype=np.float64)
            axis_y = np.asarray(snapshot.latitude, dtype=np.float64)
            axis_x = np.asarray(snapshot.longitude, dtype=np.float64)
            rows = _axis_window(axis_y, y)
            cols = (_axis_window(axis_x, x) if projected
                    else _longitude_window(axis_x, x))
            if rows.size == 0 or cols.size == 0:
                continue
            jy = _brackets(axis_y[rows], y, wrap=False)
            ix = _brackets(axis_x[cols], x, wrap=not projected)
            order = np.argsort(-np.asarray(snapshot.levels_hpa,
                                           dtype=np.float64))

            def cut(name):
                field = np.asarray(fields[name], dtype=np.float64)
                return field[order][:, rows][:, :, cols]
            speed = np.hypot(cut("UU"), cut("VV"))
            heights = cut("GHT")
            ground = fields.get("SOILHGT")
            if ground is not None:
                ground = np.asarray(ground, dtype=np.float64)[rows][:, cols]
                speed = np.where(heights >= ground[None], speed, np.nan)
            source_top = np.full(speed.shape[1:], -np.inf)
            for a in jy:
                for b in ix:
                    np.maximum.at(source_top, (a.ravel(), b.ravel()),
                                  tops.ravel())
            under = heights[:-1] < source_top[None]
            band = np.concatenate(
                [np.ones((1,) + source_top.shape, dtype=bool), under],
                axis=0)
            column = np.where(band & np.isfinite(speed), speed,
                              -np.inf).max(axis=0)
            for a in jy:
                for b in ix:
                    best = np.maximum(best, column[a, b])
        return np.where(np.isfinite(best), best, np.nan)

    def strongest(self, crest_height):
        from woof.ingest.horiz import source_coordinate_transform

        best = None
        for snapshot in self.snapshots:
            fields = snapshot.fields
            if not {"UU", "VV", "GHT"} <= set(fields):
                continue
            transform, projected = source_coordinate_transform(snapshot)
            y, x = transform(np.asarray(self.latitude, dtype=np.float64),
                             np.asarray(self.longitude, dtype=np.float64))
            y = np.asarray(y, dtype=np.float64)
            x = np.asarray(x, dtype=np.float64)
            axis_y = np.asarray(snapshot.latitude, dtype=np.float64)
            axis_x = np.asarray(snapshot.longitude, dtype=np.float64)
            rows = _axis_window(axis_y, y)
            cols = (_axis_window(axis_x, x) if projected
                    else _longitude_window(axis_x, x))
            if rows.size == 0 or cols.size == 0:
                continue
            order = np.argsort(-np.asarray(snapshot.levels_hpa,
                                           dtype=np.float64))

            def cut(name):
                field = np.asarray(fields[name], dtype=np.float64)
                return field[order][:, rows][:, :, cols]
            speed = np.hypot(cut("UU"), cut("VV"))
            heights = cut("GHT")
            ground = fields.get("SOILHGT")
            if ground is not None:
                ground = np.asarray(ground, dtype=np.float64)[rows][:, cols]
                speed = np.where(heights >= ground[None], speed, np.nan)
            found = _strongest(speed, heights, crest_height)
            if found is None:
                continue
            seconds = (snapshot.valid_time - self.start_time).total_seconds()
            when = "start" if seconds <= 0 else _boundary_when(seconds)
            if best is None or found[0] > best[0]:
                best = (found[0], found[1], when)
        return best


def _brackets(axis, values, *, wrap: bool):
    """The two indices of ``axis`` around each of ``values``: ``(lo, hi)``
    arrays of ``values``' shape, clipped at the ends.  ``axis`` is
    monotonic either way; a longitude ``axis`` (``wrap``) may cross the
    seam, and is read as an arc."""

    axis = np.asarray(axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if wrap:
        finite = values[np.isfinite(values)]
        ref = (0.0 if finite.size == 0 else float(np.degrees(np.arctan2(
            np.sin(np.radians(finite)).mean(),
            np.cos(np.radians(finite)).mean())))) - 180.0
        axis = np.mod(axis - ref, 360.0)
        values = np.mod(values - ref, 360.0)
    perm = np.argsort(axis, kind="stable")
    ordered = axis[perm]
    hi = np.clip(np.searchsorted(ordered, values, side="left"), 0,
                 ordered.size - 1)
    lo = np.clip(hi - 1, 0, ordered.size - 1)
    return perm[lo], perm[hi]


def _axis_window(axis, values):
    """Indices of a 1-D source axis inside the values' extent, widened by
    one source spacing on each side."""

    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 2:
        return np.arange(axis.size)
    step = float(np.nanmax(np.abs(np.diff(axis))))
    low = float(np.nanmin(values)) - step
    high = float(np.nanmax(values)) + step
    return np.flatnonzero((axis >= low) & (axis <= high))


def _longitude_window(axis, values):
    """Indices of a longitude axis inside the values' extent, widened by one
    source spacing on each side, across the seam wherever it falls.

    Longitude wraps, so the extent is the shortest arc holding every value:
    the circle minus its widest gap between neighbouring values.  A window
    from the smallest value to the largest instead reads every column of a
    domain that straddles the source's seam (0 degrees on a 0 to 360
    source, the dateline on a -180 to 180 one), and picks up winds half a
    world away.  A domain clear of the seam gets the window it always had.
    """

    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 2:
        return np.arange(axis.size)
    values = np.asarray(values, dtype=np.float64)
    values = np.unique(np.mod(values[np.isfinite(values)], 360.0))
    if values.size == 0:
        return np.arange(0)
    gaps = np.diff(np.append(values, values[0] + 360.0))
    widest = int(np.argmax(gaps))
    start = float(values[(widest + 1) % values.size])
    span = 360.0 - float(gaps[widest])
    spacing = np.abs(np.diff(axis))
    step = float(np.nanmax(np.minimum(spacing, 360.0 - spacing)))
    offset = np.mod(np.mod(axis, 360.0) - (start - step), 360.0)
    # A column exactly on the window's low edge can come back from the two
    # wraps a rounding short of 360 rather than at 0; it is on the edge.
    offset = np.where(offset > 360.0 - _WRAP_ROUNDING, offset - 360.0, offset)
    return np.flatnonzero(offset <= span + 2.0 * step + _WRAP_ROUNDING)


#: Degrees of rounding the wraps above may leave on a window edge.
_WRAP_ROUNDING = 1.0e-9


def forcing_window(snapshots: Mapping, start_time, run_seconds: float):
    """The snapshots a run's window reads: every one from the start to the
    end, and the nearest outside either end when that end falls between
    two."""

    from datetime import timedelta

    end = start_time + timedelta(seconds=float(run_seconds))
    times = sorted(snapshots)
    chosen = [t for t in times if start_time <= t <= end]
    later = [t for t in times if t > end]
    if later and (not chosen or chosen[-1] < end):
        chosen.append(later[0])
    earlier = [t for t in times if t < start_time]
    if earlier and (not chosen or chosen[0] > start_time):
        chosen.insert(0, earlier[-1])
    return tuple(snapshots[t] for t in chosen)
