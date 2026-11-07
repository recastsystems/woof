"""The terrain clock read face by face: ``terrain_clock = "local_face"``.

THE DEFAULT CLOCK from 2.8.8 (it was a default-off candidate before).
Every domain whose run leaves ``terrain_clock`` unset, or names
``"local_face"`` (the key in an experiment, or a
``! gpuwm-physics-selectors-v1`` comment line in a WRF namelist), reads
it; ``"measured"`` (the shipped domain-wide clock) and ``"pinned"`` stay
selectable.  It is never worse than ``"measured"`` (:func:`never_worse`).
The flip rests on the six long cases (NCAR CONUS 12 km and 2.5 km, LA
Santa Ana, SF Diablo, Toronto, Boston): each runs a decision a 48 h or
full-window run already held (CLOCK-CHECK-NCAR-2026-10-06, fix/step3 to
fix/step7).

THE DEFECT IT ANSWERS.  The shipped clock (:mod:`woof.terrain_clock`)
reads one triple per domain: the steepest slope anywhere, the highest
ground anywhere, and the strongest crest-band wind anywhere.  On NCAR's
v4.4 CONUS benchmarks that put a Plains jet (48 m/s at 12 km, 52 m/s at
2.5 km) and a San Juan or Front Range crest on a Sierra face hundreds of
km away, and halved both published steps (72 s to 36 s, 15 s to 7.5 s),
where both grids ran clean at NCAR's own steps (peak WRF vertical Courant
0.52 and 0.57, w_damping never acted; CLOCK-CHECK-NCAR-2026-10-06).

WHAT IT READS.  Every cell face of the domain, not only the steepest:

* its slope, the substep rule's own per-face reading
  (:func:`woof.acoustic_adaptation.face_slopes`);
* its crest: the highest ground in its NEIGHBOURHOOD, the square window of
  :func:`neighbourhood_cells` cells around each of the face's two mass
  columns;
* its crest-level wind: the strongest wind over the same window, from the
  ground up to the first model level at or above the crest band's top,
  over every input the shipped clock reads (start state, the root's
  boundary data over the window, the forcing snapshots over the window);
  each column's band runs up to the highest ground within ``2R + 1``
  cells of it, which is at least the crest of every face whose window
  holds the column, so a column never reads a shallower band than a face
  over it asks for;
* that wind times :data:`WIND_MARGIN`, and never more than the shipped
  domain-wide reading (the strongest wind the inputs carry anywhere, which
  is what the shipped clock reads for every face), but never less than
  :data:`INPUT_WIND_FLOOR` times the face's own input wind
  (:func:`face_wind_read`).  Breakage the cap prevents: a refusal resting
  on a wind no input carries anywhere in the window (the margin times a
  face's own wind past the domain's strongest); on both NCAR benchmark
  grids no face's wind in NCAR's own forecast passed the domain-wide
  reading at any output hour.  Breakage the floor prevents: on the HRRR
  long-run cases the interior jet passed that reading over the faces
  under its core (LA Santa Ana, 1.14 x a step-setting face's own input),
  which moved LA's cap (:data:`INPUT_WIND_FLOOR`).

Each face is read on the map exactly as the shipped clock reads its one
triple (:func:`woof.terrain_clock.read_map`: next steeper slope row,
taller crest row, stronger wind column, every gentler row and weaker
wind), and the domain is held to the least of all of them: a step stands
only where every face's reading holds it.  Faces that read the same map
cells read the same answer, so the faces are grouped by the cells they
read and each group is read once; the receipt records, per group, the
face that leads it (its steepest), its own crest, wind and position, and
what the map said there.

THE NEIGHBOURHOOD, MEASURED ON THE MAP'S OWN RIDGES.  Each map row is a
bell ridge of a stated crest and ridge slope; the window must reach from
the ridge's steepest grid face to its crest, or the local reading would
read that ridge under a lower crest than the row was measured under.  On
every row the map holds at a spacing, that distance is at most 22 cells
at 500 m, 16 at 1 km, 16 at 2 km (the candidate's 4500 m crest, slope
0.05 ridge), 10 at 3 km, 7 at 4 km, 10 at 6 km, 6 at 9 km and 5 at 12 km
(the widest shipped ridges, crest 8.85 km at slope 0.05 to 0.2).  A
domain takes the longest such distance, in metres, over the mapped
spacings it reads (both neighbours of a spacing between two), in whole
cells of its own spacing: 60 km (5 cells) at 12 km, 32 km (13 cells) at
2.5 km.  A shorter window would read a face of a mapped ridge under a
lower crest than the ridge it stands on, so a narrower one is not
measured evidence for it.  :func:`neighbourhood_cells` derives it from the map's
rows at load, so the window follows the map.

THE WIND MARGIN.  The start state and boundary data say nothing about the
wind a face will see later in the window: over NCAR's 12 km Sierra face
the start state carried 11 m/s and NCAR's own forecast 25 m/s across the
face by f012, 2.3 x (REVIEW.md section 4 item 6).  A face's wind is read
at :data:`WIND_MARGIN` times what its window's inputs carry; its
docstring holds the measurement it was checked against.

THE ROWS.  The rows measured for this candidate
(:data:`CANDIDATE_MAP_PATH`): 2, 3 and 12 km; crests 3000, 3500 and 4500
m; ridge slopes 0.05 to 0.40 (grid slopes from 0.049, including 0.089 to
0.10 and NCAR's 12 km 0.0915); 20 to 60 m/s; 6.5 down to 3 s/km; three
hours each; 4 and 6 substeps; at the generated dynamics and at NCAR's.
Every row was swept directly under the held test decided on 2026-10-07
(lead decision (i)): a step HOLDS where its run stays finite for the full
three hours with peak w under ten times the map's old bound
``4 x wind x slope + 20`` m/s, and (lead decision 3 after step 5) with
peak |w| under 100 m/s (``tools/terrain_clock_probe.py``
``BLOWUP_PEAK_W``): every run behind the rows was re-read under it, and
the two fixed entries and fifteen adaptive rows that rested on a run past
100 m/s were walked on down (the map's ``peak_rule``, fix/step6).  1,345 of those cells (every cell either
NCAR grid reads at or above its step, the cells deciding the four HRRR
long-run cases, every held run still growing in its last hour and the
deciding cells' row neighbours) were run again for 12 hours on the wider
domain the probe sizes for 12 hours; the 40 that came out one rung shorter
carry their 12-hour entry (the map's ``long_probes``; every flip stopped
inside 2.6 hours, so the three-hour domain was too narrow, not too short).
The adaptive clock's entries on the 2 and 3 km rows of crests 3000 to
4500 m (and, at NCAR's dynamics, of the 1500 m crest at ridges 0.1 to
0.25) are the 12-hour blow-up-test ones (:data:`ADAPTIVE_ROWS_PATH`).
At 70 m/s the candidate ran the 192 NCAR-dynamics cells of the 2 and 3
km rows that LA Santa Ana's parent grid reads at the input-wind floor,
12 hours each under the same test (the map's ``wind_70_probes``, fix/
step5): read there before, those faces took the shipped map's 70 m/s
cells, measured under the retired old bound.  At the 1.5 floor (fix/step6)
the same faces read 80 m/s, and the 192 cells there were run the same way
under the 100 m/s peak rule (the map's ``wind_80_probes``); three of them
(2 km, 4500 m crest, ridges 0.15 and 0.2) hold no step at all, which
takes LA's parent grid's face-by-face reading past the map (BEYOND_MEASURED),
so that grid keeps the shipped clock's decision (:func:`never_worse`).  Breakage the change prevents: the old
bound alone stopped finite, steady mountain waves on gentle 2 and 3 km
ridges at every step down to 3 s/km (MEASURE.md finding 1), a refusal
naming no breakage; every 12 km cell it stopped also ran away under the
new test, so the 12 km stops stand.  The 3500 m crest rows are measured
ridges, not a rounding: the faces that set both NCAR grids' limits stand
under 3398 m (12 km) and 3420 m (2.5 km) local crests, which the 3000 and
4500 m rows alone read as 4500 m ridges, a third taller than the ground.

Each arm reads every shipped row, merged with its own three-hour row on
the same ridge where there is one (:func:`merge_cell`: the three-hour row
decides every step it tried, both ways, lead decision (iii); the shipped
row only steps longer than any it tried), plus its rows on ridges the
shipped map never ran (:func:`candidate_map`; :func:`moved_rows` lists
every shipped row that moved).  A domain reads the arm measured at its own
dynamics (:func:`dynamics_arms`), both where its dynamics match neither.
No slope is rounded to a gentler row: a face steeper than a row reads the
next steeper one, as the shipped clock does.

WHAT IT DOES NOT READ LOCALLY.  A following (moving) nest's corridor: it
can move over ground its start window never saw, so it keeps the shipped
domain-wide reading.  A domain whose faces cannot be read with the map
factors the substep rule read (:func:`woof.acoustic_adaptation
.steepest_slope`), so its steepest face would not be the substep rule's,
also keeps the domain-wide reading.  The receipt says which and why.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

#: The ``RunConfig.terrain_clock`` value that selects this reading.
LOCAL_FACE = "local_face"

#: The candidate rows this reading adds to the shipped map.
CANDIDATE_MAP_PATH = Path(__file__).with_name(
    "terrain_clock_map_local_face.json")

#: The adaptive-clock entries this reading takes in place of the shipped
#: map's on the 2 and 3 km four-substep rows of crests 3000, 3500 and
#: 4500 m: 224 rows re-measured for 12 hours under the same blow-up held
#: test as the candidate's fixed rows, at both arms' dynamics (test host,
#: 2026-10-07, CLOCK-CHECK-NCAR-2026-10-06/fix/step4/ADAPTIVE-ROWS.md).
#: Breakage they answer: the shipped adaptive entries (three hours, the
#: old ``4 x wind x slope + 20`` bound, generated dynamics only) said
#: "none held" on SF Diablo's parent-grid ground where every one of those
#: runs was a finite mountain wave the old bound stopped, so the two
#: halves of the candidate's reading rested on different tests and SF's
#: cap was decided by the retired one.  Step 5 added the NCAR-dynamics
#: 1500 m crest rows at 2 and 3 km, ridges 0.1 to 0.25 (fix/step5): LA
#: Santa Ana's parent grid reads them at the input-wind floor, where the
#: shipped 2 km ridge 0.1 cell said "none held" at 60 m/s on the retired
#: bound; re-measured it holds 15 s/km for 12 h.
ADAPTIVE_ROWS_PATH = Path(__file__).with_name(
    "terrain_clock_adaptive_local_face.json")

#: The factor on the crest-level wind a face's window carries in its
#: inputs.  Breakage it prevents: a face whose inputs carry a calm start
#: and whose forecast brings a jet over it later in the window, read at
#: the calm wind and admitted at a step that stops.  The value is the
#: growth measured over NCAR's 12 km Sierra face (11 m/s in the start
#: state, 25 m/s by f012 in NCAR's own forecast; REVIEW.md section 4 item
#: 6), kept by lead decision (ii), 2026-10-07.
#:
#: Checked over every face of both NCAR benchmark grids against NCAR's
#: own reference forecasts at the end hour (12 km f012, 2.5 km f006;
#: ``tools/terrain_clock_local_check.py margin``,
#: CLOCK-CHECK-NCAR-2026-10-06/fix/step2/w1/step2-tree7/margin-*.json):
#:
#: * what decides: read with every face at the larger of its margin
#:   reading and the wind NCAR's forecast carried over it, each grid's
#:   limit is the margin reading's own (6.0 s/km at 4 substeps on both),
#:   so no growth NCAR forecast moved either grid's step past the margin;
#: * per face, at 12 km no face needed more than its windowed inputs (the
#:   60 km window already carries 23.9 m/s near the Sierra face; the 58
#:   faces whose limit depends on the wind grew at most 1.35 x);
#: * per face, at 2.5 km 11 of 3.6 million faces needed more than 2.3 x
#:   (up to 2.75 x: calm 7-8 m/s inputs under a 3930 m crest reaching
#:   20-23 m/s by f006), none of them on ground that sets the step.
#:
#: Read past the domain-wide reading it is capped there (module
#: docstring), but never below :data:`INPUT_WIND_FLOOR` times the face's
#: own input wind.  On both NCAR grids no face's wind at any output hour
#: passed the domain-wide reading (worst 0.97 x at 12 km, 0.81 x at 2.5
#: km); on the four HRRR long-run cases interior winds passed it on every
#: case (CLOCK-CHECK-NCAR-2026-10-06/fix/step4/WIND-CHECK.md).  The margin
#: itself was passed on faces whose inputs carried calm winds (up to 6.3 x
#: at NCAR 12 km, 4.6 x on SF Diablo); read on the 12-hour rows with the
#: floor below, at every output hour of all six cases, those passings
#: shortened some faces' limits and refused no step that ran
#: (fix/step4/DECIDE.md), so the margin keeps its measured value.
WIND_MARGIN = 2.3

#: The least a face's crest-level wind is read at, as a factor on the
#: wind its own window's inputs carry, even past the domain-wide reading.
#: Breakage it prevents: LA Santa Ana 2025-01-07 12Z, parent grid (2.25
#: km): a step-setting face (slope 0.339, local crest 3841 m) whose inputs
#: carried 47.5 m/s was read at the 49.3 m/s domain-wide reading while
#: the run's own interior jet carried 51.1 to 54.2 m/s over it at f15 to
#: f19 (1.14 x its input); read at that wind the clock caps the grid at
#: 10.12 s, where the candidate ran 11.25 s (WIND-CHECK.md).  The
#: domain-wide reading is the strongest wind the start state and boundary
#: data carry anywhere, and a face under the jet core can see that jet
#: strengthen inside the domain.  The value: the worst measured need,
#: 1.14 x, from hourly snapshots (a lower bound), rounded up to 1.25.
#: Checked at every output hour of the six long-run cases (NCAR 12 and
#: 2.5 km, LA, SF, Toronto, Boston): with it no face's wind at any hour
#: moves any domain's clock decision (fix/step4/DECIDE.md).  A factor on
#: the domain-wide reading instead (1.2 x and up, the HRRR cases'
#: measured excess) halves NCAR's 2.5 km step on winds NCAR's own
#: forecast never carried (0.81 x of that reading), so the floor is on
#: the face's own wind.  RAISED to 1.5 (lead decision 2 after step 5,
#: 2026-10-07): at 1.25 the floor had no margin on the case it was built
#: from, since on the step-5 tree two of LA's step-setting faces at f17
#: (input 43.21 m/s, actual 54.17 m/s) needed 1.254 x their own input.
#: At 1.5 LA's faces under the jet core read up to 73.9 m/s, the 80 m/s
#: column; every cell read there was run 12 h under the blow-up test with
#: the 100 m/s peak rule (fix/step6, the map's ``wind_80_probes``), and
#: three of them hold no step, so LA's parent grid reads BEYOND_MEASURED
#: (5 s steps capped at 7.87 s on six substeps, the most stable pair at a
#: weaker wind); under :func:`never_worse` (lead decision after step 6)
#: the grid then keeps the shipped clock's 10 s steps capped at 10.12 s
#: on six substeps.  Checked at every output hour of the six long-run cases:
#: the decision re-derived at the actual winds equals the one read and no
#: face would refuse a step that ran; on LA's step-setting faces the
#: strongest actual wind is 0.76 x the read wind, where at 1.25 it was
#: 1.003 x (fix/step6/APPLY.md).
INPUT_WIND_FLOOR = 1.5


def face_wind_read(local, domain_wind: float, margin: float = WIND_MARGIN):
    """The crest-level wind each face is read at, m/s: ``margin`` times
    its window's input wind ``local``, capped at the domain-wide reading
    but never below :data:`INPUT_WIND_FLOOR` times ``local``."""

    local = np.asarray(local, dtype=np.float64)
    cap = np.maximum(float(domain_wind), INPUT_WIND_FLOOR * local)
    return np.minimum(float(margin) * local, cap)


#: How the receipts and run lines name this reading.
SCHEMA = "gpuwm-terrain-clock-local-face-v1"


# ---------------------------------------------------------------------------
# The rows: shipped map plus the candidate's own, by dynamics arm.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def candidate_document() -> dict:
    return json.loads(CANDIDATE_MAP_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def adaptive_document() -> dict:
    return json.loads(ADAPTIVE_ROWS_PATH.read_text(encoding="utf-8"))


def _adaptive_rows(arms: tuple[str, ...], row, winds) -> tuple | None:
    """``(adaptive, adaptive_tried)`` the arms measured on a map row's
    ridge under the blow-up test for 12 hours (:data:`ADAPTIVE_ROWS_PATH`),
    on the map's winds, or ``None`` where an arm did not run that ridge.

    Read by several arms, each cell is the least of theirs, and a cell any
    of them did not run is not read."""

    if int(row.sound_steps) != 4:
        return None
    document = adaptive_document()
    if [float(w) for w in document["winds_m_s"]] != [float(w)
                                                      for w in winds]:
        raise ValueError(f"{ADAPTIVE_ROWS_PATH.name}: its winds are not "
                         "the shipped map's")
    found = []
    for arm in arms:
        match = [r for r in document["rows"] if r["arm"] == arm
                 and float(r["dx_m"]) == float(row.dx_m)
                 and float(r["crest_m"]) == float(row.crest_m)
                 and abs(float(r["slope"]) - float(row.slope)) <= 5e-5]
        if not match:
            return None
        found.append(match[0])
    entries, tops = [], []
    for index in range(len(winds)):
        tried = [r["adaptive_top_s_per_km"][index] for r in found]
        held = [r["adaptive_s_per_km"][index] for r in found]
        if any(t is None for t in tried):
            entries.append(None)
            tops.append(None)
            continue
        entries.append(None if any(h is None for h in held)
                       else float(min(held)))
        tops.append(float(min(tried)))
    return tuple(entries), tuple(tops)


def _arm_matches(run, dynamics: Mapping[str, object]) -> bool:
    """The run integrates the dynamics an arm was measured at.

    The sixth-order filter's factor and slope option say nothing where the
    filter is off on both, and the Rayleigh layer's depth and coefficient
    nothing where it is off on both."""

    def value(name):
        raw = getattr(run, name, None)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    skip = set()
    if int(dynamics.get("diff_6th_opt", 0)) == 0 and value("diff_6th_opt") == 0:
        skip |= {"diff_6th_factor", "diff_6th_slopeopt"}
    if int(dynamics.get("damp_opt", 0)) == 0 and value("damp_opt") == 0:
        skip |= {"zdamp", "dampcoef"}
    for name, want in dynamics.items():
        if name in skip:
            continue
        got = value(name)
        if got is None or abs(got - float(want)) > 1e-9 * max(1.0, abs(
                float(want))):
            return False
    return True


def dynamics_arms(run) -> tuple[str, ...]:
    """The candidate arms a run reads: the one measured at its own
    dynamics, or every arm where it matches none (each arm's rows are a
    measurement of other dynamics, so a run between them reads the least
    of both)."""

    arms = candidate_document()["arms"]
    matched = tuple(sorted(name for name, dynamics in arms.items()
                           if _arm_matches(run, dynamics)))
    return matched or tuple(sorted(arms))


def _coordinate(row: Mapping) -> tuple:
    """The probe ridge a map row was measured on."""
    return (float(row["dx_m"]), float(row["crest_m"]),
            round(float(row["ridge_slope"]), 6), int(row["sound_steps"]))


def merge_cell(shipped_entry, shipped_top: float, entry, top):
    """One wind of a shipped row the candidate re-measured on the same
    ridge: ``(entry, top)`` as the merged row reads it.

    The candidate's three-hour run decides every step it tried (longer
    evidence wins over the shipped half-hour row, both ways: lead decision
    (iii), 2026-10-07).  The shipped row still decides steps longer than
    any the candidate tried.  Breakage that keeps the shipped part: the
    candidate walks 6.5 s/km down, the shipped 3 km rows were tried to
    13.33 s/km; dropping them would read a 3 km domain at 10 s/km as never
    stopped where the shipped map saw it stop.

    * candidate never ran this wind (``top`` ``None``): the shipped cell;
    * the shipped row tried nothing past the candidate's top: the
      candidate's cell;
    * the candidate saw a longer rung run away (``entry`` under ``top``,
      or none held): its entry, under the shipped top, so a stop;
    * the candidate held its top: past it the shipped row decides, its
      entry if longer, else a stop at the candidate's top."""

    if top is None:
        return shipped_entry, shipped_top
    if shipped_top <= top * (1.0 + 1e-9):
        return entry, top
    if entry is None or entry < top * (1.0 - 1e-9):
        return entry, shipped_top
    if shipped_entry is not None and shipped_entry > top:
        return shipped_entry, shipped_top
    return top, shipped_top


def _arm_rows(arm: str):
    """``(merged, added)`` for one candidate arm, as plain dicts on the
    shipped winds: every shipped row (merged with the arm's row on the
    same ridge where there is one, :func:`merge_cell`) and the arm's rows
    on ridges the shipped map never ran."""

    from woof.terrain_clock import MAP_PATH

    shipped = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    winds = [float(w) for w in shipped["winds_m_s"]]
    top = float(max(shipped["ladder_s_per_km"]))
    document = candidate_document()
    cand_winds = [float(w) for w in document["winds_m_s"]]
    index = {w: winds.index(w) for w in cand_winds}
    mine = {_coordinate(row): row for row in document["rows"]
            if row["settings"] == arm}
    used = set()
    merged = []
    for row in shipped["rows"]:
        stable = [None if v is None else float(v)
                  for v in row["stable_s_per_km"]]
        tried = ([float(v) for v in row["top_s_per_km"]]
                 if row.get("top_s_per_km") is not None
                 else [top] * len(winds))
        key = _coordinate(row)
        cand = mine.get(key)
        # The same ridge reads the same grid slope (the probe's own
        # geometry); a row whose slope differs is another ridge.
        if cand is not None and abs(float(cand["slope"])
                                    - float(row["slope"])) <= 5e-5:
            used.add(key)
            new_stable, new_tried = list(stable), list(tried)
            for wind, entry, ctop in zip(cand_winds, cand["stable_s_per_km"],
                                         cand["top_s_per_km"]):
                i = index[wind]
                new_stable[i], new_tried[i] = merge_cell(
                    stable[i], tried[i],
                    None if entry is None else float(entry),
                    None if ctop is None else float(ctop))
            merged.append({"row": row, "arm": arm, "stable": new_stable,
                           "tried": new_tried, "shipped_stable": stable,
                           "shipped_tried": tried})
        else:
            merged.append({"row": row, "arm": None, "stable": stable,
                           "tried": tried, "shipped_stable": stable,
                           "shipped_tried": tried})
    added = []
    for key, row in mine.items():
        if key in used:
            continue
        stable = [None] * len(winds)
        tried = [None] * len(winds)
        for wind, entry, ctop in zip(cand_winds, row["stable_s_per_km"],
                                     row["top_s_per_km"]):
            stable[index[wind]] = None if entry is None else float(entry)
            tried[index[wind]] = None if ctop is None else float(ctop)
        added.append({"row": row, "arm": arm, "stable": stable,
                      "tried": tried})
    return merged, added


def moved_rows(arm: str) -> list[dict]:
    """Every shipped row the candidate arm's three-hour row changes, with
    both readings: the rows that moved (lead decision (iii))."""

    merged, _added = _arm_rows(arm)
    out = []
    for item in merged:
        if item["arm"] is None:
            continue
        if (item["stable"] == item["shipped_stable"]
                and item["tried"] == item["shipped_tried"]):
            continue
        row = item["row"]
        out.append({"dx_m": row["dx_m"], "crest_m": row["crest_m"],
                    "ridge_slope": row["ridge_slope"], "slope": row["slope"],
                    "sound_steps": row["sound_steps"], "arm": arm,
                    "shipped_stable_s_per_km": item["shipped_stable"],
                    "shipped_top_s_per_km": item["shipped_tried"],
                    "merged_stable_s_per_km": item["stable"],
                    "merged_top_s_per_km": item["tried"]})
    return out


@functools.lru_cache(maxsize=8)
def candidate_map(arms: tuple[str, ...]):
    """The map the candidate reads for a run of ``arms``, as one
    :class:`woof.terrain_clock.StableStepMap` on the shipped winds: per
    arm, every shipped row merged with that arm's three-hour row on the
    same ridge (:func:`merge_cell`), plus the arm's rows on ridges the
    shipped map never ran.  A run between arms reads both arms' rows (the
    least of both).

    A candidate row carries ``None`` in ``top_s_per_km`` at a wind it was
    not run at (80 m/s and up, 70 m/s on every row but the step-5 cells
    of ``wind_70_probes``, and past a wind none of its steps held):
    that cell supplies no held or stopped step. A reading selecting it
    is beyond measured wind coverage and keeps the shipped clock.
    Shipped rows keep their adaptive entries, which the candidate never
    measured."""

    from woof.terrain_clock import MAP_PATH, MapRow, measured_map

    base = measured_map()
    # measured_map() keeps the file's rows in order; the merge pairs them.
    if len(base.rows) != len(json.loads(MAP_PATH.read_text(
            encoding="utf-8"))["rows"]):
        raise ValueError("the shipped map's rows and its file disagree")
    rows = []
    seen_shipped = set()

    def adaptive(row, who):
        # The 12-hour blow-up-test adaptive entries where the arms ran the
        # ridge (ADAPTIVE_ROWS_PATH); elsewhere the shipped ones, or none.
        got = _adaptive_rows(who, row, base.winds)
        if got is None:
            return row
        return replace(row, adaptive=got[0], adaptive_tried=got[1])

    for arm in arms:
        merged, added = _arm_rows(arm)
        for position, (item, shipped_row) in enumerate(zip(merged,
                                                           base.rows)):
            if item["arm"] is None:
                if position in seen_shipped:
                    continue
                seen_shipped.add(position)
                rows.append(adaptive(shipped_row, tuple(arms)))
                continue
            rows.append(adaptive(replace(shipped_row,
                                         stable=tuple(item["stable"]),
                                         tried=tuple(item["tried"])),
                                 (arm,)))
        for item in added:
            row = item["row"]
            rows.append(adaptive(MapRow(
                float(row["dx_m"]), float(row["crest_m"]),
                float(row["slope"]), int(row["sound_steps"]),
                tuple(item["stable"]), tuple(item["tried"])), (arm,)))
    return replace(base, rows=tuple(rows))


# ---------------------------------------------------------------------------
# The neighbourhood.
# ---------------------------------------------------------------------------


def ridge_reach_cells(dx: float, crest: float, ridge_slope: float) -> int:
    """Cells from a probe ridge's steepest grid face to its crest.

    The probe's bell ridge (:func:`woof.core.terrain.bell_hill`) of half-
    width ``3 sqrt(3) / 8 * crest / ridge_slope``, on an even number of
    columns so its crest sits on a face as on the map's domains: the
    fewest cells from either mass column of its steepest face to a column
    at its crest height."""

    a = (3.0 * math.sqrt(3.0) / 8.0) * float(crest) / float(ridge_slope)
    nx = max(96, int(math.ceil(10.0 * a / float(dx))))
    nx += nx % 2
    x = (np.arange(nx) + 0.5) * float(dx) - 0.5 * nx * float(dx)
    h = float(crest) / (1.0 + (x / a) ** 2)
    face = int(np.argmax(np.abs(np.diff(h))))
    crest_columns = np.flatnonzero(h >= h.max() - 1e-9)
    return int(min(abs(p - c) for p in (face, face + 1)
                   for c in crest_columns))


@functools.lru_cache(maxsize=8)
def reach_by_spacing(arms: tuple[str, ...]) -> dict[float, float]:
    """Per mapped spacing, the longest face-to-crest distance, metres, over
    every row the reading takes there (shipped and candidate)."""

    from woof.terrain_clock import MAP_PATH

    shipped = json.loads(MAP_PATH.read_text(encoding="utf-8"))["rows"]
    rows = list(shipped) + [row for row in candidate_document()["rows"]
                            if row["settings"] in arms]
    reach: dict[float, float] = {}
    for row in rows:
        dx = float(row["dx_m"])
        metres = ridge_reach_cells(dx, row["crest_m"], row["ridge_slope"]) * dx
        reach[dx] = max(reach.get(dx, 0.0), metres)
    return dict(sorted(reach.items()))


def neighbourhood_cells(dx: float, arms: tuple[str, ...]) -> tuple[int, float]:
    """``(cells, metres)``: the window's half-width for a domain at ``dx``.

    The longest face-to-crest distance over the mapped spacings the domain
    reads (the nearest past either end, both neighbours between two), in
    whole cells of its own spacing, never fewer than one."""

    reach = reach_by_spacing(arms)
    spacings = sorted(reach)
    dx = float(dx)
    if dx <= spacings[0]:
        used = [spacings[0]]
    elif dx >= spacings[-1]:
        used = [spacings[-1]]
    else:
        exact = [s for s in spacings if abs(s - dx) <= 1e-6 * s]
        used = exact or [max(s for s in spacings if s < dx),
                         min(s for s in spacings if s > dx)]
    metres = max(reach[s] for s in used)
    return max(1, int(math.ceil(metres / dx - 1e-9))), metres


# ---------------------------------------------------------------------------
# Per-column crest-band winds.
# ---------------------------------------------------------------------------


def _window_max(field: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter

    if radius <= 0:
        return np.array(field, dtype=np.float64, copy=True)
    return maximum_filter(np.asarray(field, dtype=np.float64),
                          size=2 * int(radius) + 1, mode="nearest")


def band_tops(terrain: np.ndarray, radius: int) -> np.ndarray:
    """Each column's band top: the highest ground within ``2 R + 1``
    cells, at least the crest of every face whose window holds it."""

    return _window_max(terrain, 2 * int(radius) + 1)


def start_column_band(start, tops: np.ndarray) -> np.ndarray:
    """A :class:`woof.terrain_clock.StartWinds`' strongest crest-band wind
    per mass column, the band running up to ``tops`` there; NaN where a
    column carries none.  Level by level, as the shipped reading walks."""

    from woof.terrain_clock import _G

    php = start.fields("php")
    phb = start.fields("phb")
    tops = np.asarray(tops, dtype=np.float64)
    best = np.full(tops.shape, -np.inf)
    previous = None
    u_all = start.fields("u")
    v_all = start.fields("v")
    for k in range(int(phb.shape[0]) - 1):
        z = 0.5 * ((np.asarray(phb[k], dtype=np.float64)
                    + np.asarray(php[k], dtype=np.float64))
                   + (np.asarray(phb[k + 1], dtype=np.float64)
                      + np.asarray(php[k + 1], dtype=np.float64))) / _G
        if previous is None:
            inside = np.ones(z.shape, dtype=bool)
        else:
            inside = previous < tops
            if not inside.any():
                break
        uk = np.asarray(u_all[k], dtype=np.float64)
        vk = np.asarray(v_all[k], dtype=np.float64)
        speed = np.hypot(0.5 * (uk[:, :-1] + uk[:, 1:]),
                         0.5 * (vk[:-1, :] + vk[1:, :]))
        take = inside & np.isfinite(speed)
        best = np.where(take, np.maximum(best, speed), best)
        previous = z
    return np.where(np.isfinite(best), best, np.nan)


def boundary_column_band(boundary, tops: np.ndarray) -> np.ndarray:
    """A :class:`woof.terrain_clock.BoundaryWinds`' strongest crest-band
    wind per mass column of its slabs over the forecast window (every
    instant the shipped reading takes), NaN off the slabs."""

    tops = np.asarray(tops, dtype=np.float64)
    ny, nx = tops.shape
    best = np.full(tops.shape, -np.inf)
    for interval, offset, _at in boundary.instants():
        fields = interval.fields
        if not {"u", "v", "mu"} <= set(fields):
            continue
        boundary._refuse_another_grid(fields)
        for side in ("west", "east", "south", "north"):
            speed, heights, jj, ii = boundary._side_cells(fields, side,
                                                          offset)
            top = tops[jj, ii]
            under = heights[:-1] < top[None]
            band = np.concatenate([np.ones((1,) + top.shape, dtype=bool),
                                   under], axis=0)
            masked = np.where(band & np.isfinite(speed), speed, -np.inf)
            column = masked.max(axis=0)
            np.maximum.at(best, (jj.ravel(), ii.ravel()), column.ravel())
    del ny, nx
    return np.where(np.isfinite(best), best, np.nan)


def column_band(sources: Sequence, tops: np.ndarray) -> tuple[np.ndarray,
                                                               np.ndarray]:
    """``(wind, which)``: per mass column the strongest crest-band wind over
    ``sources`` (each with a ``column_band(tops)``), and the index of the
    source it came from (-1 where none carries one)."""

    tops = np.asarray(tops, dtype=np.float64)
    best = np.full(tops.shape, np.nan)
    which = np.full(tops.shape, -1, dtype=np.int16)
    for index, source in enumerate(sources):
        reader = getattr(source, "column_band", None)
        if reader is None:
            continue
        # Kept at the inputs' own single precision (every wind a door
        # reads is float32 on disk), so a saved fixture holds it exactly.
        field = np.asarray(reader(tops), dtype=np.float32).astype(np.float64)
        stronger = np.isfinite(field) & ~(field <= best)
        best = np.where(stronger, field, best)
        which = np.where(stronger, np.int16(index), which)
    return best, which


# ---------------------------------------------------------------------------
# The faces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceFields:
    """What every face of one domain reads: the inputs of the reading.

    ``terrain`` ``(ny, nx)``; ``msfu`` ``(ny, nx + 1)`` and ``msfv``
    ``(ny + 1, nx)`` or ``None``; ``wind`` the per-column crest-band wind
    over the inputs (:func:`column_band`), NaN where none; ``which`` the
    input each column's wind came from, ``sources`` their names and
    instants.  These are what a fixture saves."""

    terrain: np.ndarray
    msfu: np.ndarray | None
    msfv: np.ndarray | None
    wind: np.ndarray
    which: np.ndarray
    sources: tuple[str, ...]
    radius: int
    dx: float
    dy: float


@dataclass(frozen=True)
class FaceGroup:
    """Faces that read the same map cells, and the face that leads them."""

    count: int
    axis: str
    j: int
    i: int
    slope: float
    crest_m: float
    input_wind_m_s: float
    wind_read_m_s: float
    wind_capped: bool
    source: str
    reading: object


@dataclass(frozen=True)
class LocalFaces:
    """One domain's face-by-face reading at one substep count."""

    sound_steps: int
    radius: int
    radius_m: float
    margin: float
    domain_wind_m_s: float
    arms: tuple[str, ...]
    faces: int
    groups: tuple[FaceGroup, ...]
    #: The group whose reading sets the domain's limit.
    governing: FaceGroup | None
    combined: object

    def receipt(self, dx: float) -> dict:
        km = float(dx) / 1000.0

        def seconds(value):
            return None if value is None else float(value) * km

        def group(g: FaceGroup) -> dict:
            r = g.reading
            return {"faces": int(g.count), "face": {"axis": g.axis,
                                                    "j": int(g.j),
                                                    "i": int(g.i)},
                    "slope": float(g.slope), "crest_m": float(g.crest_m),
                    "input_wind_m_s": float(g.input_wind_m_s),
                    "wind_read_m_s": float(g.wind_read_m_s),
                    "wind_capped_at_domain": bool(g.wind_capped),
                    "wind_from": g.source,
                    "map_rows": {"crest_m": r.crest_row,
                                 "slope": r.slope_row,
                                 "wind_m_s": r.wind_row,
                                 "beyond": list(r.beyond)},
                    "held_s": seconds(r.per_km),
                    "longest_step_tried_s": seconds(r.top_per_km),
                    "shortest_entry_under_a_stop_s": seconds(
                        r.stopped_per_km)}

        return {"sound_steps": int(self.sound_steps),
                "neighbourhood_cells": int(self.radius),
                "neighbourhood_m": float(self.radius_m),
                "wind_margin": float(self.margin),
                "domain_wind_m_s": float(self.domain_wind_m_s),
                "arms": list(self.arms), "faces_read": int(self.faces),
                "governing": (None if self.governing is None
                              else group(self.governing)),
                "groups": [group(g) for g in self.groups]}


def face_fields(terrain, dx: float, dy: float, *, msfu, msfv,
                sources: Sequence, radius: int) -> FaceFields:
    """Every column's crest-band wind over ``sources`` (each a reading
    with ``column_band``), the band up to :func:`band_tops`."""

    terrain = np.asarray(terrain, dtype=np.float64)
    tops = band_tops(terrain, radius)
    wind, which = column_band(sources, tops)
    names = tuple(getattr(source, "source", f"input {index}")
                  + (f" ({getattr(source, 'kind', '')})"
                     if getattr(source, "kind", "") else "")
                  for index, source in enumerate(sources))
    return FaceFields(terrain, None if msfu is None else np.asarray(
        msfu, dtype=np.float64), None if msfv is None else np.asarray(
        msfv, dtype=np.float64), wind, which, names, int(radius),
        float(dx), float(dy))


def _faces(fields: FaceFields):
    """Per-face arrays: ``(axis, j, i, slope, crest, wind, which)`` over
    every x-face then every y-face, flattened."""

    from woof.acoustic_adaptation import face_slopes

    sx, sy = face_slopes(fields.terrain, fields.dx, fields.dy,
                         msfu=fields.msfu, msfv=fields.msfv)
    crest = _window_max(fields.terrain, fields.radius)
    wind = np.where(np.isfinite(fields.wind), fields.wind, -np.inf)
    wmax = _window_max(wind, fields.radius)
    # The input each window's strongest wind came from, at that column.
    out = []
    if sx is not None:
        ny, n = sx.shape
        jj, ii = np.meshgrid(np.arange(ny), np.arange(1, n + 1),
                             indexing="ij")
        c = np.maximum(crest[:, :-1], crest[:, 1:])
        w = np.maximum(wmax[:, :-1], wmax[:, 1:])
        out.append((0, jj.ravel(), ii.ravel(), sx.ravel(), c.ravel(),
                    w.ravel()))
    if sy is not None:
        n, nx = sy.shape
        jj, ii = np.meshgrid(np.arange(1, n + 1), np.arange(nx),
                             indexing="ij")
        c = np.maximum(crest[:-1, :], crest[1:, :])
        w = np.maximum(wmax[:-1, :], wmax[1:, :])
        out.append((1, jj.ravel(), ii.ravel(), sy.ravel(), c.ravel(),
                    w.ravel()))
    axis = np.concatenate([np.full(part[1].size, part[0], dtype=np.int8)
                           for part in out])
    j = np.concatenate([part[1] for part in out])
    i = np.concatenate([part[2] for part in out])
    slope = np.concatenate([part[3] for part in out])
    crest_f = np.concatenate([part[4] for part in out])
    wind_f = np.concatenate([part[5] for part in out])
    return axis, j, i, slope, crest_f, wind_f


def _wind_source(fields: FaceFields, axis: int, j: int, i: int) -> str:
    """Which input carried the strongest wind in a face's window."""

    r = fields.radius
    ny, nx = fields.wind.shape
    js = [j - 1, j] if axis == 1 else [j]
    is_ = [i - 1, i] if axis == 0 else [i]
    j0, j1 = max(0, min(js) - r), min(ny, max(js) + r + 1)
    i0, i1 = max(0, min(is_) - r), min(nx, max(is_) + r + 1)
    window = fields.wind[j0:j1, i0:i1]
    if not np.isfinite(window).any():
        return "none"
    k = int(np.nanargmax(window))
    which = int(fields.which[j0:j1, i0:i1].flat[k])
    return fields.sources[which] if 0 <= which < len(fields.sources) \
        else "none"


def _keys(table, dx: float, slope, crest, wind, sound_steps: int):
    """The map cells each face reads, as one integer per face: what
    :func:`woof.terrain_clock.read_map` selects from (spacing rows,
    crest row, slope row and wind column), vectorised."""

    from woof.terrain_clock import CREST_MATCH, _sound_row

    count = _sound_row(sound_steps)
    rows = [row for row in table.rows if row.sound_steps == count]
    spacings = sorted({row.dx_m for row in rows})
    if dx <= spacings[0]:
        dx_rows = (spacings[0],)
    elif dx >= spacings[-1]:
        dx_rows = (spacings[-1],)
    else:
        exact = [s for s in spacings if abs(s - dx) <= 1e-6 * s]
        dx_rows = (exact[0],) if exact else (
            max(s for s in spacings if s < dx),
            min(s for s in spacings if s > dx))
    winds = np.asarray(table.winds)
    w_index = np.minimum(np.searchsorted(winds, wind - 1e-9, side="left"),
                         winds.size - 1)
    key = w_index.astype(np.int64)
    base = winds.size
    # Past the map's edges read_map says so (``beyond``), and the
    # derivation treats such a reading apart; a face past an edge and one
    # on it read the same cells but not the same reading.
    past = (wind - 1e-9 > winds[-1]).astype(np.int64)
    for spacing in dx_rows:
        crests = np.asarray(sorted({row.crest_m for row in rows
                                    if row.dx_m == spacing}))
        c_index = np.minimum(np.searchsorted(
            crests, crest * (1.0 - CREST_MATCH) - 1e-6, side="left"),
            crests.size - 1)
        s_index = np.zeros(slope.shape, dtype=np.int64)
        widest = 0
        for n, height in enumerate(crests):
            slopes = np.asarray(sorted({row.slope for row in rows
                                        if row.dx_m == spacing
                                        and row.crest_m == height}))
            widest = max(widest, slopes.size)
            here = c_index == n
            s_index[here] = np.minimum(np.searchsorted(
                slopes, slope[here] - 1e-9, side="left"), slopes.size - 1)
            past |= ((slope - 1e-9 > slopes[-1]) & here).astype(
                np.int64) << 1
        past |= (crest * (1.0 - CREST_MATCH) - 1e-6 > crests[-1]).astype(
            np.int64) << 2
        key = key + base * (c_index + crests.size * s_index)
        base *= crests.size * max(1, widest)
    return key * 8 + past


def _grouped_face_leaders(keys, slope, wind, crest):
    """The original lexicographic leader and count for every sorted key."""
    if keys.size < 4096 or not all(
            np.isfinite(values).all() for values in (slope, wind, crest)):
        order = np.lexsort((crest, wind, slope, keys))
        sorted_keys = keys[order]
        last = np.r_[sorted_keys[1:] != sorted_keys[:-1], True]
        return order[last], np.diff(np.r_[-1, np.flatnonzero(last)])
    # Only the last face of each lexicographic group is needed. Sort the
    # integer groups once, then select maxima in the original priority order.
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])]
    counts = np.diff(np.r_[starts, keys.size])
    selected = np.ones(keys.size, dtype=bool)
    for values in (slope, wind, crest):
        values = values[order]
        maxima = np.maximum.reduceat(np.where(selected, values, -np.inf), starts)
        selected &= values == np.repeat(maxima, counts)
    # lexsort is stable: equal slope/wind/crest picks the final input face.
    leaders = np.maximum.reduceat(np.where(selected, order, -1), starts)
    return leaders, counts


def read_faces(fields: FaceFields, *, table, dx: float, sound_steps: int,
               domain_wind: float, margin: float = WIND_MARGIN,
               arms: tuple[str, ...] = (), radius_m: float = 0.0
               ) -> LocalFaces | None:
    """Read every face on ``table`` at ``sound_steps``; ``None`` where no
    face carries a wind reading."""

    from woof.terrain_clock import read_map

    axis, j, i, slope, crest, wind = _faces(fields)
    have = np.isfinite(wind)
    if not have.any():
        return None
    local = np.where(have, wind, float(domain_wind))
    read = face_wind_read(local, float(domain_wind), float(margin))
    keys = _keys(table, float(dx), slope, crest, read, int(sound_steps))
    # Each group's leader: its steepest face, then strongest wind read,
    # then tallest crest (all read the same cells).
    leaders, counts = _grouped_face_leaders(keys, slope, read, crest)
    groups = []
    for leader, n in zip(leaders, counts):
        k = int(leader)
        reading = read_map(float(dx), float(crest[k]), float(slope[k]),
                           float(read[k]), int(sound_steps), table)
        groups.append(FaceGroup(
            count=int(n), axis="x" if int(axis[k]) == 0 else "y",
            j=int(j[k]), i=int(i[k]), slope=float(slope[k]),
            crest_m=float(crest[k]), input_wind_m_s=float(local[k]),
            wind_read_m_s=float(read[k]),
            wind_capped=bool(float(margin) * float(local[k])
                             > float(read[k]) * (1.0 + 1e-12)),
            source=_wind_source(fields, int(axis[k]), int(j[k]), int(i[k])),
            reading=reading))
    combined, governing = combine([g.reading for g in groups])
    groups.sort(key=lambda g: (-g.slope, -g.wind_read_m_s))
    return LocalFaces(
        sound_steps=int(sound_steps), radius=int(fields.radius),
        radius_m=float(radius_m), margin=float(margin),
        domain_wind_m_s=float(domain_wind), arms=tuple(arms),
        faces=int(axis.size), groups=tuple(groups),
        governing=None if governing is None else next(
            g for g in groups if g.reading is governing),
        combined=combined)


def _limit(reading) -> float:
    """What a reading lets a domain run, s/km, for ordering: no held step
    reads lowest, then the shortest entry under a stop, then none."""

    from woof.terrain_clock import _held_step

    if reading.per_km is None:
        most = reading.most_stable_per_km
        return -1.0 if most is None else -1.0 + 1e-6 * most
    limit = _held_step(reading)
    return math.inf if limit is None else float(limit)


def combine(readings: Sequence):
    """One reading for the domain from every face group's: a step holds
    only where it holds on every face.  ``(combined, governing)``, the
    governing reading being the one with the least limit (its rows label
    the combined reading)."""

    from woof.terrain_clock import MapReading

    if not readings:
        return None, None
    governing = min(readings, key=lambda r: (_limit(r), bool(not r.beyond)))

    def least(values):
        values = list(values)
        if any(v is None for v in values):
            return None
        return min(values) if values else None

    def least_present(values):
        values = [v for v in values if v is not None]
        return min(values) if values else None

    stopped = [(r.stopped_per_km, r) for r in readings
               if r.stopped_per_km is not None]
    stop = min(stopped, key=lambda item: item[0]) if stopped else None
    beyond = tuple(sorted({name for r in readings for name in r.beyond}))
    adaptive = {}
    if any(r.adaptive_measured or r.adaptive_none_held
           or r.adaptive_stopped_per_km is not None for r in readings):
        every = all(r.adaptive_measured for r in readings)
        adaptive = {
            "adaptive_measured": every,
            "adaptive_per_km": (least(r.adaptive_per_km for r in readings)
                                if every else None),
            "adaptive_top_per_km": (least_present(
                r.adaptive_top_per_km for r in readings) if every else None),
            "adaptive_stopped_per_km": least_present(
                r.adaptive_stopped_per_km for r in readings),
            "adaptive_none_held": any(r.adaptive_none_held
                                      for r in readings),
            "adaptive_none_held_below_per_km": least_present(
                r.adaptive_none_held_below_per_km for r in readings)}
    combined = MapReading(
        per_km=least(r.per_km for r in readings),
        dx_rows=governing.dx_rows, crest_row=governing.crest_row,
        slope_row=governing.slope_row, wind_row=governing.wind_row,
        beyond=beyond,
        most_stable_per_km=least(r.most_stable_per_km for r in readings),
        top_per_km=min(r.top_per_km for r in readings),
        stopped_per_km=None if stop is None else stop[0],
        stopped_crest_m=None if stop is None else stop[1].stopped_crest_m,
        stopped_under_a_lower_crest=(False if stop is None else
                                     stop[1].stopped_under_a_lower_crest),
        **adaptive)
    return combined, governing


# ---------------------------------------------------------------------------
# The domain, and its derivation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainFaces:
    """What one domain's face-by-face derivation reads, read per substep
    count on demand."""

    fields: FaceFields
    table: object
    arms: tuple[str, ...]
    radius_m: float
    domain_wind: float
    dx: float
    margin: float = WIND_MARGIN
    _readings: dict = field(default_factory=dict, compare=False,
                            repr=False)

    def reading(self, sound_steps: int) -> LocalFaces | None:
        count = int(sound_steps)
        if count not in self._readings:
            self._readings[count] = read_faces(
                self.fields, table=self.table, dx=self.dx,
                sound_steps=count, domain_wind=self.domain_wind,
                margin=self.margin, arms=self.arms, radius_m=self.radius_m)
        return self._readings[count]


# ---------------------------------------------------------------------------
# Never worse than the shipped clock.
# ---------------------------------------------------------------------------

#: Why the shipped decision was put in place of the face-by-face one.
NEVER_WORSE_BEYOND = "beyond_measured"
NEVER_WORSE_SHORTER = "shorter_than_shipped"


def longest_step(adaptation, run) -> Fraction:
    """The longest step a decision lets the grid run, s: the adaptive cap
    written, else the adaptive clock's own longest step (its
    ``max_time_step``, or WRF's ``8 x dx`` fill-in), else the fixed
    step."""

    from woof.terrain_clock import _adaptive_upper

    if adaptation.ceiling is not None:
        return Fraction(adaptation.ceiling)
    if adaptation.adaptive:
        upper = _adaptive_upper(run)
        if upper is None:
            from woof.core.adaptive_clock import wrf_default_clamps
            upper = Fraction(wrf_default_clamps(run.dx, run.dy)[1])
        return Fraction(upper)
    return Fraction(adaptation.dt)


def _decision(adaptation, run) -> dict:
    from woof.terrain_clock import _rational

    return {"status": adaptation.status,
            "dt_s": _rational(adaptation.dt),
            "step_division": int(adaptation.division),
            "time_step_sound": int(adaptation.time_step_sound),
            "max_time_step_s": (None if adaptation.ceiling is None
                                else _rational(adaptation.ceiling)),
            "longest_step_s": _rational(longest_step(adaptation, run))}


def _words(adaptation, run) -> str:
    from woof.terrain_clock import _seconds

    out = (f"{_seconds(adaptation.dt)} steps on "
           f"{int(adaptation.time_step_sound)} acoustic substeps")
    if adaptation.ceiling is not None:
        out += (" with an adaptive step capped at "
                f"{_seconds(Fraction(adaptation.ceiling))}")
    elif adaptation.adaptive:
        out += (" with the adaptive clock's own longest step, "
                f"{_seconds(longest_step(adaptation, run))}")
    return out


@dataclass(frozen=True)
class NeverWorse:
    """The face-by-face decision set against the shipped measured clock's
    for the same grid (:func:`never_worse`)."""

    #: The shipped decision was put in place of the face-by-face one.
    applied: bool
    #: :data:`NEVER_WORSE_BEYOND` or :data:`NEVER_WORSE_SHORTER` where
    #: applied, else ``None``.
    reason: str | None
    #: The face-by-face derivation as read (a ``ClockAdaptation``).
    local: object
    #: The shipped measured derivation on the domain-wide reading.
    shipped: object
    run: object = field(default=None, compare=False, repr=False)

    def receipt(self) -> dict:
        return {"applied": bool(self.applied), "reason": self.reason,
                "rule": ("the grid takes the shipped measured clock's "
                         "decision where the local-face one is "
                         "BEYOND_MEASURED or runs a shorter first or "
                         "longest step"),
                "local_face": _decision(self.local, self.run),
                "shipped_measured": _decision(self.shipped, self.run)}

    def sentence(self, label: str) -> str:
        if self.reason == NEVER_WORSE_BEYOND:
            why = ("which is past what the measured map holds "
                   "(BEYOND_MEASURED)")
        else:
            why = ("which is a shorter step than the shipped measured "
                   "clock gives")
        return (f"time step: {self.local._what()}; read face by face, "
                f"{label} would run {_words(self.local, self.run)}, {why}, "
                f"so {label} keeps the shipped measured clock's decision "
                f"for this grid, {_words(self.shipped, self.run)} "
                f"({self.shipped.status}), and its receipt says so")


def never_worse(local, shipped, run):
    """The decision a ``local_face`` grid runs: the face-by-face one, or
    the shipped measured clock's where the face-by-face one is
    BEYOND_MEASURED or runs a shorter first or longest step
    (:func:`longest_step`) than the shipped one.  The result carries the
    comparison as ``never_worse`` either way.

    Breakage it prevents: LA Santa Ana 2025-01-07 12Z, parent grid (2.25
    km).  At the 1.5 input-wind floor its faces under the jet core read
    the 80 m/s column, where three 2 km, 4500 m crest cells hold no step,
    so the face-by-face reading went BEYOND_MEASURED and would run 5 s
    steps capped at 7.87 s and warn that the grid may still stop
    (CLOCK-CHECK-NCAR-2026-10-06/fix/step6/APPLY.md).  The shipped clock
    runs that grid at 10 s steps capped at 10.12 s on six substeps, and
    the 48 h LA run at it was finite at every hour with no recovery
    (fix/step3/PROVE.md).  Lead decision after step 6: the new clock is
    never worse than the shipped measured clock."""

    reason = None
    if local.status == "BEYOND_MEASURED":
        reason = NEVER_WORSE_BEYOND
    elif (Fraction(local.dt) < Fraction(shipped.dt)
          or longest_step(local, run) < longest_step(shipped, run)):
        reason = NEVER_WORSE_SHORTER
    record = NeverWorse(applied=reason is not None, reason=reason,
                        local=local, shipped=shipped, run=run)
    return replace(shipped if record.applied else local, never_worse=record)


def derive_local(grid_id: int, run, dt, slope: float, crest, faces:
                 DomainFaces, *, label: str | None = None):
    """:func:`woof.terrain_clock.derive_clock` under ``local_face``: the
    shipped derivation (division, substeps, adaptive ceiling) on the
    face-by-face reading of each substep count it weighs, never worse
    than the shipped measured clock's own decision (:func:`never_worse`)."""

    from woof.terrain_clock import _derive_measured, read_map

    seen: dict[int, LocalFaces] = {}

    def read(count):
        local = faces.reading(count)
        if local is None:
            return read_map(float(run.dx), crest.crest_height_m, slope,
                            crest.wind_m_s, count, faces.table)
        seen[count] = local
        return local.combined

    measured = _derive_measured(grid_id, run, dt, slope, crest, label=label,
                                table=faces.table, read=read)
    shipped = _derive_measured(grid_id, run, dt, slope, crest, label=label)
    if not seen:
        return replace(shipped, local_note=(
            "no face carried a wind reading, so it keeps the shipped "
            "domain-wide reading"))
    chosen = next((local for local in seen.values()
                   if local.combined is measured.reading), None)
    local = replace(measured, local=chosen, local_note=(
        None if chosen is not None else
        "no face carried a wind reading, so the domain-wide reading was "
        "read on the candidate rows"))
    return never_worse(local, shipped, run)


def local_readings(exp, *, slopes: Mapping[int, float], winds: Mapping,
                   statics: Mapping[int, object], sources: Mapping[int,
                                                                   Sequence],
                   boundary=None, corridors=None) -> dict[int, object]:
    """Per ``local_face`` domain, its :class:`DomainFaces`, or a sentence
    saying why it keeps the domain-wide reading.

    A domain reads its own start state (the inputs carry its own winds on
    its own grid; a nest's ancestors' start states are on other grids and
    stay in the domain-wide reading that caps every face) and, for the
    root, the boundary data over the window."""

    from woof.acoustic_adaptation import face_slopes
    from woof.terrain_clock import _static

    root = int(exp.domains[0].grid_id)
    out: dict[int, object] = {}
    for dc in exp.domains:
        gid = int(dc.grid_id)
        if str(getattr(dc.run, "terrain_clock", "measured")) != LOCAL_FACE:
            continue
        if gid not in slopes:
            continue
        if corridors and corridors.get(gid) is not None:
            out[gid] = ("a following nest can move over ground its start "
                        "window never saw, so it keeps the domain-wide "
                        "reading over its corridor")
            continue
        static = statics.get(gid)
        terrain = _static(static, "HGT_M")
        if terrain is None:
            out[gid] = "its static fields carry no terrain"
            continue
        if gid not in winds:
            out[gid] = "no wind reading"
            continue
        msfu, msfv = _static(static, "MAPFAC_U"), _static(static, "MAPFAC_V")
        run = dc.run
        sx, sy = face_slopes(terrain, float(run.dx), float(run.dy),
                             msfu=msfu, msfv=msfv)
        steepest = max(float(np.max(part)) for part in (sx, sy)
                       if part is not None)
        want = float(slopes[gid])
        if abs(steepest - want) > 1e-9 * max(1.0, want):
            # Breakage: faces read without the map factors the substep
            # rule read would under-read every slope by up to the factor.
            out[gid] = (f"its faces read {steepest:.4f} at their steepest "
                        f"where the substep rule read {want:.4f} (other "
                        "map factors), so it keeps the domain-wide reading")
            continue
        arms = dynamics_arms(run)
        table = candidate_map(arms)
        radius, metres = neighbourhood_cells(float(run.dx), arms)
        inputs = list(sources.get(gid, ()))
        if gid == root and boundary is not None:
            inputs.append(boundary)
        fields = face_fields(terrain, float(run.dx), float(run.dy),
                             msfu=msfu, msfv=msfv, sources=inputs,
                             radius=radius)
        out[gid] = DomainFaces(fields=fields, table=table, arms=arms,
                               radius_m=metres,
                               domain_wind=float(winds[gid].wind_m_s),
                               dx=float(run.dx))
    return out


class StreamedFaceCache:
    """Reuse fixed head geometry and fold each boundary interval in once.

    Owned by one StreamedClockGuard, whose start states, statics and
    corridors cannot change. A stronger column wind or domain wind
    invalidates that domain's face readings. Unchanged nests and root
    faces keep their already derived readings at each substep count.
    """

    def __init__(self):
        self.domains = None
        self.tops = None
        self.intervals = {}

    def readings(self, exp, *, slopes, winds, statics, sources,
                 boundary=None, corridors=None):
        from woof.terrain_clock import BoundaryWinds
        from woof.ingest.lateral_bc import LateralBoundaries

        root = int(exp.domains[0].grid_id)
        if self.domains is None:
            self.domains = local_readings(
                exp, slopes=slopes, winds=winds, statics=statics,
                sources=sources, corridors=corridors)
        else:
            # A head with no initial wind can acquire its first reading
            # from a later boundary interval.
            missing = [dc for dc in exp.domains
                       if dc.grid_id in winds and (
                           dc.grid_id not in self.domains
                           or self.domains[dc.grid_id] == "no wind reading")]
            if missing:
                recovered = local_readings(
                    exp, slopes=slopes,
                    winds=winds, statics=statics, sources=sources,
                    corridors=corridors)
                for dc in missing:
                    if dc.grid_id in recovered:
                        self.domains[dc.grid_id] = recovered[dc.grid_id]
        faces = self.domains.get(root)
        if isinstance(faces, DomainFaces) and boundary is not None:
            fields = faces.fields
            if self.tops is None:
                self.tops = band_tops(fields.terrain, fields.radius)
            best, which = fields.wind, fields.which
            names = fields.sources
            boundary_name = boundary.source + f" ({boundary.kind})"
            if not names or names[-1] != boundary_name:
                names = (*names, boundary_name)
            for interval in boundary.boundaries.intervals:
                key = float(interval.start_seconds)
                if self.intervals.get(key) is interval:
                    continue
                # Replacement intervals are not a valid append-only
                # stream. Rebuild instead of retaining an obsolete max.
                if key in self.intervals:
                    self.domains = None
                    self.intervals.clear()
                    self.tops = None
                    return self.readings(
                        exp, slopes=slopes, winds=winds, statics=statics,
                        sources=sources, boundary=boundary, corridors=corridors)
                single = BoundaryWinds(
                    boundary.source, LateralBoundaries((interval,), 1, 1, 1),
                    boundary.geometry, boundary.run_seconds)
                incoming = np.asarray(single.column_band(self.tops),
                                      dtype=np.float32).astype(np.float64)
                stronger = np.isfinite(incoming) & ~(incoming <= best)
                if stronger.any():
                    best = np.where(stronger, incoming, best)
                    which = np.where(stronger, np.int16(len(names) - 1), which)
                self.intervals[key] = interval
            if best is not fields.wind or names != fields.sources:
                fields = replace(fields, wind=best, which=which, sources=names)
                faces = replace(faces, fields=fields, _readings={})
                self.domains[root] = faces
        for gid, faces in tuple(self.domains.items()):
            if isinstance(faces, DomainFaces) and gid in winds:
                wind = float(winds[gid].wind_m_s)
                if wind != faces.domain_wind:
                    self.domains[gid] = replace(faces, domain_wind=wind,
                                                 _readings={})
        return self.domains
