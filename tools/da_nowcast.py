"""One front door: any WSR-88D site + a time window -> a cycled nowcast.

``python -m tools.da_nowcast run --site XXXX --window-end <ISO|latest>``
runs the whole live-fire pipeline that round 3 proved, site-parameterized
end to end:

    survey   S3 listing, freshness/lag measurement, echo census, storm
             motion from volume-to-volume centroid displacement
    domain   a range-authority-sized box sited on the echo with
             downstream room, emitted by the ``woof domain`` wizard
             (never hand-typed namelists)
    prepare  authority -> fetch -> manifest -> prepared cache, on THIS
             box (prepared cases are host-bound; a receipted finding)
    obs      one ``gpuwm-obs.radar-grid.v1`` file per cycle via the
             rw_nexrad seam (``tools/obs_radar_grid_build.py``), built
             from one radar or several (``--sites``/``--discover-sites``)
    cycle    N-member cycled GPU LETKF with free forecast legs
             (``tools/da_cycle_prepared.py``)
    render   the map-styled gallery (``tools/da_nowcast_render.py``)

``--site`` is the ANCHOR radar: it sites the domain and names the case.
``--sites XXXX`` (repeatable) adds radars to every observation the run
builds, and ``--discover-sites`` computes them from the domain's own
georeference instead.  One radar measures the wind along its own beam
and nothing else; two whose coverage overlaps constrain what neither
can alone.  The selection is recorded in the receipt under ``radars``,
and the rolling verifier reads it back -- so the observed composites a
case is GRADED against are built from the same radars it ASSIMILATED,
which is the difference between verification and flattery.

Verification is AUTOMATIC: when a run completes, the front door spawns
a detached rolling verifier (``watch`` mode) that polls the archive,
builds each free-forecast frame's observed composite as its valid time
gets covered, re-renders the gallery in place (counts + FSS on every
pair), and exits once the last frame verifies -- the nowcast grades
itself as reality arrives, with no further command.  ``--no-verify``
opts out.  ``python -m tools.da_nowcast verify --case-dir DIR`` remains
as a one-shot re-grade of an existing case directory.

The receipt carries the verification state machine for tooling to
poll: ``verification.state`` is ``pending`` (not started yet),
``rolling`` (a detached verifier is grading), ``complete`` (every
free-forecast frame carries numbers), ``incomplete`` (the verifier
stopped with frames the archive never covered), or ``disabled``
(``--no-verify``).  Beside it sit a verdict line, the watcher's pid
and log, and one entry per frame with its status and its numbers.

Every stage is an existing shipped tool driven over its CLI; this file
adds planning, siting, and receipts.  Each stage writes a versioned JSON
receipt under ``<out>/receipts/`` and the run ends with a
``nowcast-receipt.json`` (schema ``gpuwm-da.nowcast.v1``) -- the seam a
GUI drives later.

ACCURACY: this is a demo-grade nowcast.  UNSCORED, outside any registered
campaign, EXPERIMENTAL like every tool it drives.  No skill claim is
made or implied; the gallery says so on every figure.

No radar-site names belong in this file, its defaults, or its
identifiers: sites, times and buckets are arguments (standing owner
rule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = "gpuwm-da.nowcast.v1"
STAGE_SCHEMA = "gpuwm-da.nowcast-stage.v1"
#: Everything a later process needs to drive the prepared case this run
#: built: the three digests, the authority, and the window plan.
BINDINGS_SCHEMA = "gpuwm-da.nowcast-bindings.v1"
SURVEY_SCHEMA = "gpuwm-da.nowcast-survey.v1"
VERIFY_SCHEMA = "gpuwm-da.nowcast-verification.v1"

#: The verification state machine a GUI polls.  ``disabled`` only when
#: the caller opted out; ``pending`` before the verifier owns the case;
#: ``rolling`` while it grades frames as the archive covers them;
#: ``complete`` when every free-forecast frame carries numbers;
#: ``incomplete`` when the verifier stopped with frames still uncovered.
VERIFY_STATES = ("disabled", "pending", "rolling", "complete",
                 "incomplete")

#: Watcher defaults: how often to re-ask the archive, how long to keep
#: asking, and how far past a valid time to wait before the first ask
#: (a volume is archived some minutes after the time it is stamped).
WATCH_POLL_SECONDS = 240
WATCH_MAX_MINUTES = 180
WATCH_MIN_AGE_SECONDS = 120.0

# ---------------------------------------------------------------------------
# Defaults set from measurement, 2026-08-05
#
# Every number below cites the run that produced it.  Where the run that
# would settle a number has not finished, the number is LEFT ALONE and
# said so -- see "Pending measurement" in docs/da-nowcast-demo.md.
# The archived receipts explicitly use WSM6/Dudhia with longwave off.
# They do not qualify the current MP28/GSD4.1 suite's skill or card fit.
# ---------------------------------------------------------------------------

#: Ensemble size.  MEASURED, not assumed: three N-ladders on the same
#: case scored with the same scorer, and 10 is the knee of all three.
#:
#:   32 GB card, evidence/da-demo/ensemble-size-sweep/n{10,20,36}/
#:     N=10  mean-field FSS 0.7408   wall  465 s
#:     N=20  mean-field FSS 0.7423   wall  847 s   (+82% wall)
#:     N=36  mean-field FSS 0.7396   wall 1826 s   (BELOW N=10, 3.9x wall)
#:   16 GB card, evidence/16gb-frontier/runs/f198n{04,10,20}/
#:     N=4   mean-field FSS 0.7331   wall  431 s
#:     N=10  mean-field FSS 0.7397   wall  727 s
#:     N=20  mean-field FSS 0.7435   wall 1163 s
#:
#: The N=10 -> N=20 gain does not survive its own noise.  With the
#: averaging depth held at 10 for both (so only analysis quality
#: varies) it is +0.0018; per member, the statistic averaging cannot
#: flatter, it is +0.0038 -- against an across-member FSS scatter of
#: 0.0062 at N=10 and 0.0074 at N=20, i.e. larger than the effect
#: (evidence/da-demo/ensemble-size-sweep/skill-decomposition-partial.json).
#: And bigger is not monotonically better here: the metric is computed
#: on the ensemble MEAN composite, and holding the analysis fixed while
#: deepening the average LOWERS FSS monotonically, 0.7470 at depth 1 to
#: 0.7423 at depth 20, because these forecasts already under-produce
#: echo (~2000 columns >=35 dBZ against ~2800 observed) and smoothing
#: moves them further from truth.
#:
#: Downward is measurably worse and buys nothing: N=4 costs 0.007 FSS at
#: +15 min and 0.008 at +90 against N=10 on a 16 GB card, and leaves
#: peak VRAM exactly where it was (below).
#:
#: And upward is where it starts to fall over: the N=36 arm of the
#: 2026-08-05 local sweep ran 1527 s and died with
#: CUDA_ERROR_OUT_OF_MEMORY on a 32 GB card at the default budget,
#: with other work sharing the card.
DEFAULT_MEMBERS = 10

#: The physics profile the nowcast binds when the caller names none.
#:
#: It was ``wsm6-ysu-mm5-noah-no-radiation-v1`` until 2026-08-09: the
#: lw-off / sw-on pairing, shipped as the DEFAULT of a storm-scale
#: nowcast product whose cases are overwhelmingly nocturnal, while
#: docs/public/PHYSICS.md stated in as many words that "no door emits an
#: asymmetric pairing as a default".  This door is what made that
#: sentence false.  Nothing computed the downward longwave, so every
#: member of every cycle ran its land surface against a sky no scheme
#: produced -- and the nocturnal guard stayed silent on the emitted case
#: because ``woof domain`` had written the declaration into it on the
#: user's behalf.
#:
#: The nowcast background is HRRR by default (project ruling, 2026-08-06),
#: so this named product follows the HRRR route's authoritative suite.
#: 4193eb0da, development branch, selects GSD4.1/MP28/RUC with both legacy
#: RRTMG streams and no cumulus. This is source-selected configuration,
#: not new nowcast forecast qualification. Old member/card measurements
#: must be remeasured for this composition; explicit --physics-profile
#: retains the earlier MP8/RTE suite for archival reproduction.
from woof.hrrr_route_inputs import (
    ROUTE_DEFAULT_PHYSICS_PROFILE as NOWCAST_DEFAULT_PHYSICS_PROFILE)

#: The LETKF chunk workspace, in MiB.  This is the ONE term in the
#: memory model an operator controls, which is why it is reachable from
#: the front door at all.
#:
#: Measured on a 16,376 MiB RTX 4080 at this default
#: (evidence/16gb-frontier/runs/f198n10/vram.json): whole-card peak
#: 15,888 MiB -- 97.0% of the card, 488 MiB of headroom -- against a
#: MEDIAN of 2,168 MiB.  The spike is the solve, not the forecast legs.
#:
#: Peak does not scale with ensemble size, because the driver holds one
#: trajectory on the GPU at a time: 15,946 MiB at N=4, 15,888 at N=10,
#: 15,796 at N=20.  The mechanism is in the filter block of those same
#: receipts -- letkf sizes its chunk as budget // per_point and
#: per_point scales with members, so chunk_points halves (6,912 at N=10
#: to 3,444 at N=20) while the chunk's BYTES stay pinned here.
#:
#: 6144 is therefore a MEASURED-TO-FIT value with a thin margin, not a
#: measured-optimal one.  The ladder that would settle a lower default
#: (and the CuPy pool-cap probe beside it, tools/ens_sweep/
#: pool_limit_probe.sh) had not finished when these defaults were set,
#: so the shipped number is unchanged and the knob is exposed instead.
DEFAULT_MEMORY_BUDGET_MIB = 6144.0

#: Named card profiles.  A profile is a configuration that was RUN on
#: the card it names, end to end, and completed -- never an estimate.
#:
#: ``card-16gib`` is the 16 GB frontier's headline result: the shipped
#: demo shape (N=10, 3 km, 49 levels, 198 km box half-width, six
#: applied cycles, six free legs) runs to completion on a 16,376 MiB
#: RTX 4080 and returns the 32 GB card's answer -- FSS 0.7397 mean over
#: six leads against 0.7403 on the 5090, innovation RMS tracking the
#: 32 GB run within 0.35 m/s cycle by cycle
#: (evidence/16gb-frontier/README.md).  So the profile does not shrink
#: anything: it points the memory preflight at the card actually in the
#: box, which is the part the wizard would otherwise guess.
#:
#: Keys are argument names on ``run``, applied only where the caller
#: left the argument at its default.
CARD_PROFILES = {
    "auto": {},
    "card-16gib": {"vram_gib": 16.0},
}

#: What each profile-controlled argument holds when the caller said
#: nothing.  A profile fills in ONLY arguments still sitting on these
#: values, so an explicit flag always beats a profile.
PROFILE_UNSET = {"vram_gib": None}


def apply_card_profile(args) -> dict:
    """Fill a named profile's values into the arguments nobody set.

    Returns what was actually applied, for the receipt -- a run has to
    be able to say which profile shaped it and which of its values
    survived the caller's own flags.
    """

    profile = CARD_PROFILES[getattr(args, "profile", "auto") or "auto"]
    applied: dict = {}
    for key, value in profile.items():
        if getattr(args, key) == PROFILE_UNSET[key]:
            setattr(args, key, value)
            applied[key] = value
    return applied

#: Synoptic GFS cycles and how old a cycle must be before its hourly
#: forecast files are reliably on the archive.
SYNOPTIC_STEP_HOURS = 6
GFS_AVAILABILITY_LAG_S = 4 * 3600.0

KM_PER_DEG_LAT = 111.2

#: WSR-88D style station identifier: four characters, letter first.
_SITE_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{3}$")

STAGES = ("survey", "domain", "fetch", "prepare", "forecast", "obs",
          "cycle", "render")

# The streams --without can subtract from --da full.  Radial velocity is
# not on this list: it is the base configuration every preset stands on,
# and a run without it is a research question for --da custom, not a
# product option.
DA_SUBTRACTABLE = ("reflectivity", "clear-air", "dealias", "surface", "cwp")

#: The shipped dealiasing engine, spelled here so this module keeps its
#: stdlib-only import surface (it builds argv for a subprocess; importing
#: numpy through woof.obs.dealias to read one string would make every
#: `da_nowcast --help` pay for it).  ``tests/test_da_nowcast.py`` binds it
#: to ``woof.obs.dealias.DealiasParams().engine``: a spelling only one
#: side agrees with is a run that dies at its first observation stage,
#: hours in.
DEFAULT_DEALIAS_ENGINE = "region-global"

# The certified full-stack run's CWP vertical localisation
# (evidence/da-demo/full-stack/run.sh).  --da full fills it only when the
# caller did not state one; --da custom still requires it explicitly.
DA_FULL_CWP_VLOC_M = 20000.0


class FrontDoorError(SystemExit):
    """A refusal with its reason; exit code 2 like argparse refusals."""

    def __init__(self, message: str) -> None:
        super().__init__(f"da_nowcast: {message}")


# ---------------------------------------------------------------------------
# small pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def validate_site(text: str) -> str:
    """Uppercase and validate a radar station id; sites are arguments."""

    candidate = str(text).strip().upper()
    if not _SITE_PATTERN.match(candidate):
        raise FrontDoorError(
            f"--site {text!r} is not a station id (four characters, "
            "letter first, e.g. a WSR-88D id)")
    return candidate


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def iso(stamp: datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class SurveyClock:
    """Which time the survey asks the archive about, and why.

    The survey lists volumes, measures the feed's lag, censuses the
    echo and derives the storm motion that sites the domain.  All four
    are statements about a MOMENT, and the moment is the one the caller
    named: a replay of an archived window is about that window's
    volumes.  Reading the wall clock instead answered a question nobody
    asked -- it listed the volumes of the day the command was typed on,
    sited the domain on that day's echo, and every later stage took the
    box as given, so nothing downstream could recover from it.

    ``at`` is what the archive is asked about; ``wall`` is when the
    command ran.  With no window they are the same time, which is the
    live case and the reason the defect could survive: every live run
    was right.
    """

    at: datetime
    wall: datetime
    mode: str                     # "window" | "wall"
    window_end: datetime | None

    @property
    def subject(self) -> str:
        return ("the window end the run was given" if self.mode == "window"
                else "the time the command ran")

    @property
    def why(self) -> str:
        if self.mode == "window":
            return ("a window end was given, so the survey asked the "
                    "archive about that window: a replay is sited, timed "
                    "and graded on the volumes of the window it replays")
        return ("no window end was given (--window-end latest), so the "
                "survey asked the archive about now, which is the live "
                "case")

    def to_payload(self) -> dict:
        return {"mode": self.mode, "at": iso(self.at),
                "wall": iso(self.wall),
                "window_end": (None if self.window_end is None
                               else iso(self.window_end)),
                "why": self.why}


def survey_clock(window_end: str, *, now: datetime) -> SurveyClock:
    """The survey's clock, from the window the caller asked for.

    ``latest`` has no window yet -- it is DERIVED from the newest volume
    the feed carries, which is a question about now -- so it, and only
    it, reads the wall clock.  Anything else is a window, including one
    that ends a minute ago.
    """

    text = window_end.strip()
    if text.lower() == "latest":
        return SurveyClock(at=now, wall=now, mode="wall", window_end=None)
    try:
        end = parse_iso(text)
    except ValueError:
        raise FrontDoorError(
            f"--window-end {window_end!r} is neither an ISO-8601 UTC "
            "time nor 'latest'") from None
    return SurveyClock(at=end, wall=now, mode="window", window_end=end)


#: The perturbation length scale a caller gets when they say nothing.
#: Right for the ~400 km box the front door sites by default; a smaller
#: box cannot carry it, which is what
#: :func:`resolvable_length_scale_km` is for.
DEFAULT_LENGTH_SCALE_KM = 50.0


def perturbation_scale_bounds(*, nx: int, ny: int, dx_m: float,
                              dy_m: float) -> tuple[float, float]:
    """The (floor, ceiling) km a grid can carry, from the lane itself.

    The two constants are imported rather than copied: the perturbation
    lane owns what a grid can represent, and a second copy of its
    numbers here would drift silently and be discovered as a crash three
    minutes into somebody's run.  ``tests/test_da_nowcast.py`` pins the
    import, so a rename there is a failing test rather than a surprise.
    """

    from woof.da.perturb import (_MAX_HORIZONTAL_SPAN_FRACTION,
                                  _MIN_SCALE_CELLS)

    dx_km, dy_km = float(dx_m) / 1000.0, float(dy_m) / 1000.0
    floor = _MIN_SCALE_CELLS * max(dx_km, dy_km)
    span = min(int(nx) * dx_km, int(ny) * dy_km)
    return floor, _MAX_HORIZONTAL_SPAN_FRACTION * span


def resolvable_length_scale_km(*, nx: int, ny: int, dx_m: float,
                               dy_m: float,
                               requested: float | None = None,
                               safety: float = 0.9
                               ) -> tuple[float, str | None]:
    """The perturbation length scale this domain can actually carry.

    A caller who draws a smaller box has not asked for different
    science; they have asked for a smaller box.  So the scale is capped
    to what the box carries and the capping is SAID -- returned as a
    note the receipt and the page both show -- rather than left to be
    discovered when the ensemble refuses to perturb.

    ``safety`` keeps the value off the exact bound, which is a refusal
    rather than a value.
    """

    floor, ceiling = perturbation_scale_bounds(nx=nx, ny=ny, dx_m=dx_m,
                                               dy_m=dy_m)
    usable = safety * ceiling
    if usable < floor:
        raise FrontDoorError(
            f"a {nx}x{ny} grid at {dx_m / 1000:g} km carries no "
            f"perturbation length scale at all: its floor ({floor:g} "
            f"km, two grid spacings) is above its ceiling ({ceiling:g} "
            "km, span/(2*pi)). The domain is too small to perturb")
    if requested is None:
        chosen = min(DEFAULT_LENGTH_SCALE_KM, usable)
        note = None if chosen >= DEFAULT_LENGTH_SCALE_KM else (
            f"perturbation length scale set to {chosen:.1f} km: this "
            f"domain's span carries at most {ceiling:.1f} km "
            "(span/(2*pi)), below the "
            f"{DEFAULT_LENGTH_SCALE_KM:g} km default")
        return round(max(chosen, floor), 1), note
    requested = float(requested)
    if floor <= requested <= ceiling:
        return requested, None
    chosen = min(max(requested, floor), usable)
    return round(chosen, 1), (
        f"--length-scale-km {requested:g} is outside what a {nx}x{ny} "
        f"grid at {dx_m / 1000:g} km carries ({floor:.1f}-{ceiling:.1f} "
        f"km); using {chosen:.1f} km")


def retime_history(toml_text: str, seconds: float) -> tuple[str, float]:
    """Set the root domain's history cadence; return the text and the old.

    The wizard owns the geometry, the ladder and the memory verdict, and
    none of that is touched here.  What it cannot know is how often the
    caller needs a georeference written: its root default is hourly,
    which is right for a forecast and wrong for a nowcast, whose
    observations are gridded onto the wrfout nearest their valid time.
    An hourly lattice makes 'nearest' mean 'up to half an hour stale',
    and layer heights move in half an hour.

    Textual, and deliberately so: rewriting the emitted TOML through a
    parser would round-trip every value the wizard wrote and make this
    edit indistinguishable from a re-emission.  One key changes, in
    place, and the receipt says which.
    """

    pattern = re.compile(r"^(\s*history_interval_s\s*=\s*)([0-9.]+)",
                         re.MULTILINE)
    match = pattern.search(toml_text)
    if match is None:
        raise FrontDoorError(
            "the emitted case TOML carries no history_interval_s; the "
            "wizard's output has changed shape and the georeference "
            "cadence can no longer be set from here")
    before = float(match.group(2))
    return (pattern.sub(rf"\g<1>{float(seconds):g}", toml_text, count=1),
            before)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WindowPlan:
    """The whole run's timing, derived once and receipted."""

    window_end: datetime          # last assimilated cycle's valid time
    cycles: int
    cycle_seconds: int
    free_legs: int
    init: datetime                # model start (whole hour, hard rule)
    cycle_times: tuple[datetime, ...]
    horizon_seconds: int          # applied + free legs
    run_hours: int                # georeference run length (>= horizon)
    background_cycle: datetime    # the background source's own cycle
    forecast_start_hour: int
    background_source: str = "gfs"

    @property
    def gfs_cycle(self) -> datetime:
        """Compatibility shim for the field's pre-HRRR name.

        The cycle stopped being GFS-shaped when the background became a
        registry choice (woof/da/background.py).  Receipts written
        before the rename and callers that still say ``gfs_cycle`` keep
        working; new code says ``background_cycle``.
        """

        return self.background_cycle

    @property
    def run_seconds(self) -> int:
        return self.run_hours * 3600

    def free_leg_times(self) -> tuple[datetime, ...]:
        return tuple(
            self.window_end + timedelta(seconds=self.cycle_seconds * k)
            for k in range(1, self.free_legs + 1))

    def to_payload(self) -> dict:
        return {
            "window_end": iso(self.window_end),
            "cycles": self.cycles,
            "cycle_seconds": self.cycle_seconds,
            "free_legs": self.free_legs,
            "init": iso(self.init),
            "cycle_times": [iso(t) for t in self.cycle_times],
            "free_leg_times": [iso(t) for t in self.free_leg_times()],
            "horizon_seconds": self.horizon_seconds,
            "run_hours": self.run_hours,
            "background_source": self.background_source,
            "background_cycle": self.background_cycle.strftime(
                "%Y-%m-%dT%H"),
            # Compatibility duplicate of background_cycle under the
            # pre-HRRR key, so a reader of an older receipt and a reader
            # of this one see the same shape.  Remove with the shim.
            "gfs_cycle": self.background_cycle.strftime("%Y-%m-%dT%H"),
            "forecast_start_hour": self.forecast_start_hour,
        }


def latest_gfs_cycle(init: datetime, now: datetime) -> datetime:
    """Newest synoptic cycle at/before ``init`` whose files exist by now.

    Kept as the archival GFS reference: tests/test_da_background.py
    proves ``woof.da.background.plan_background_cycle`` returns the
    SAME cycle for GFS, so replayed receipts stay explainable by the
    arithmetic that produced them.  New planning goes through the
    registry.
    """

    candidate = init.replace(minute=0, second=0, microsecond=0)
    candidate -= timedelta(hours=candidate.hour % SYNOPTIC_STEP_HOURS)
    for _ in range(20):
        if ((now - candidate).total_seconds() >= GFS_AVAILABILITY_LAG_S
                and candidate <= init):
            return candidate
        candidate -= timedelta(hours=SYNOPTIC_STEP_HOURS)
    raise FrontDoorError(
        f"no plausibly-available GFS cycle at or before {iso(init)}")


def plan_window(window_end: datetime, *, cycles: int, cycle_seconds: int,
                free_legs: int, now: datetime,
                run_hours: int | None = None,
                source: str = "gfs") -> WindowPlan:
    """Derive the whole run's timing, or refuse and say why.

    ``run_hours`` defaults to the shortest whole-hour run that covers
    the applied cycles, the free legs, and an hour of slack.  A caller
    that will keep cycling on this prepared case for longer than one
    window -- a continuous nowcast -- states a longer one, and it is
    checked against the derived floor rather than trusted.

    ``source`` names a ``woof.da.background`` registry entry; its own
    cadence, publication lag and horizon pick the cycle.  The parameter
    default stays ``gfs`` so a receipt replay and this function's own
    tests keep meaning what they said; the FRONT DOOR's default is
    ``hrrr`` and is named once, at the CLI (project ruling, 2026-08-06).
    """

    if cycles < 1:
        raise FrontDoorError("--cycles must be >= 1")
    if free_legs < 0:
        raise FrontDoorError("--free-legs must be >= 0")
    if cycle_seconds <= 0 or cycle_seconds % 60:
        raise FrontDoorError("--cycle-seconds must be a positive whole "
                             "number of minutes")
    if window_end.second or window_end.microsecond:
        raise FrontDoorError("--window-end must land on a whole minute")
    init = window_end - timedelta(seconds=cycles * cycle_seconds)
    if init.minute or init.second:
        raise FrontDoorError(
            f"model init {iso(init)} (= window end - cycles x cycle "
            "seconds) must land on a whole hour; move --window-end or "
            "--cycles so it does")
    cycle_times = tuple(
        init + timedelta(seconds=cycle_seconds * (k + 1))
        for k in range(cycles))
    horizon = (cycles + free_legs) * cycle_seconds
    floor_hours = math.ceil(horizon / 3600.0) + 1
    if run_hours is None:
        run_hours = floor_hours
    elif run_hours < floor_hours:
        raise FrontDoorError(
            f"--run-hours {run_hours} is shorter than the {floor_hours} "
            "hours this window needs (applied cycles + free legs + an "
            "hour of slack)")
    from woof.da.background import BackgroundError, plan_background_cycle

    try:
        background = plan_background_cycle(
            source, init=init, now=now,
            run_seconds=float(run_hours) * 3600.0)
    except BackgroundError as error:
        raise FrontDoorError(str(error)) from None
    return WindowPlan(
        window_end=window_end, cycles=cycles,
        cycle_seconds=cycle_seconds, free_legs=free_legs, init=init,
        cycle_times=cycle_times, horizon_seconds=horizon,
        run_hours=run_hours, background_cycle=background.cycle,
        forecast_start_hour=background.forecast_start_hour,
        background_source=background.source)


def resolve_latest_window_end(newest_volume: datetime, *, cycles: int,
                              cycle_seconds: int) -> datetime:
    """Largest cycle-aligned time <= newest volume with a whole-hour init."""

    stamp = newest_volume.replace(second=0, microsecond=0)
    stamp -= timedelta(minutes=stamp.minute % max(1, cycle_seconds // 60))
    for _ in range(240):
        init = stamp - timedelta(seconds=cycles * cycle_seconds)
        if init.minute == 0 and init.second == 0:
            return stamp
        stamp -= timedelta(seconds=cycle_seconds)
    raise FrontDoorError(
        f"--cycle-seconds {cycle_seconds} cannot align a whole-hour "
        "init within a day; choose a divisor of 3600 or set "
        "--window-end explicitly")


#: The leg-number offset this door gives the cycle driver.
#:
#: The driver numbers a leg ``--leg-number-offset + index``
#: (``tools/da_cycle_prepared.py``'s ``leg_number``), and this door
#: starts a case at leg zero, so it leaves the driver's own default
#: alone.  Stated here rather than assumed at the point of use: the
#: other caller of that driver, ``tools/da_nowcast_auto.py``, DOES pass
#: an offset when it continues a case that already has legs, and a
#: receipt that names the free legs has to read the same number the
#: frames were written under.  A door that gains an offset changes this
#: constant, and the receipt follows it.
CYCLE_LEG_NUMBER_OFFSET = 0


def forecast_frames_block(cycle_out: Path, *, cycles: int,
                          free_legs: int,
                          leg_number_offset: int = CYCLE_LEG_NUMBER_OFFSET
                          ) -> dict:
    """Where this run's forecast frames are, named in the receipt.

    The cycle writes one frame per leg per trajectory, the FREE legs
    included -- those are the forecast running past the last observation
    -- as a column-max composite ``.npz`` and, beside it, the same field
    as a real wrfout that ``woof render --engine rust`` draws.  A
    receipt that stopped at ``cycle_out`` left a reader to discover that
    layout by listing directories, and the pages describing this door
    named only the gallery.  Nothing here is new output: it is the
    output this door has always produced, said out loud.

    What the frame is, is part of naming it.  The wrfout is a ONE-LEVEL
    surface snapshot of that composite, not the leg's three-dimensional
    state, and the remaining wrfout variables in it are the structural
    stand-ins :func:`woof.io.surface_wrfout.write_surface_wrfout`
    supplies so the file is a readable wrfout at all.  A reader of the
    receipt alone would otherwise infer full fields, which is the
    inference the pages already correct.

    ``leg_number_offset`` is the driver's own numbering, not a second
    convention invented here; see :data:`CYCLE_LEG_NUMBER_OFFSET`.
    """

    first_free = int(leg_number_offset) + int(cycles)
    return {
        "dir": str(cycle_out / "composites"),
        "npz": "legNN_<trajectory>.npz (refl_colmax, elapsed_seconds)",
        "wrfout": ("wrfout_legNN_<trajectory>.nc, drawable by "
                   "`woof render --engine rust`"),
        "vertical": ("one-level surface snapshot of the column-max "
                     "composite (REFL_10CM with its geolocation); the "
                     "three-dimensional leg state is not kept, and a "
                     "run that needs full fields drives the prepared "
                     "forecast (tools/da_cycle_prepared.py) with its "
                     "own history settings"),
        "measured_field": ("composite reflectivity; the other wrfout "
                           "variables in the file are the writer's "
                           "structural stand-ins (woof/io/"
                           "surface_wrfout.py), so composite "
                           "reflectivity is what a picture of one of "
                           "these frames shows"),
        "nested": ("legNN_<trajectory>_dNN.npz and wrfout_legNN_"
                   "<trajectory>_dNN.nc, one pair per nested leg per "
                   "nesting trajectory, when the run asked for a nest; "
                   "the child's file is the same one-level snapshot as "
                   "the parent's"),
        "trajectories": "control, then one per member by index",
        "leg_number_offset": int(leg_number_offset),
        "free_legs": [f"leg{first_free + k:02d}"
                      for k in range(int(free_legs))],
    }


def wrfout_name(init: datetime, seconds: float) -> str:
    stamp = init + timedelta(seconds=seconds)
    return stamp.strftime("wrfout_d01_%Y-%m-%d_%H_%M_%S")


def geojson_box(lat: float, lon: float, half_km: float) -> dict:
    """An axis-aligned lat/lon box for the domain wizard's --polygon."""

    dlat = half_km / KM_PER_DEG_LAT
    dlon = half_km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return {"type": "Polygon", "coordinates": [[
        [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat]]]}


def offset_latlon(lat: float, lon: float, east_km: float,
                  north_km: float) -> tuple[float, float]:
    out_lat = lat + north_km / KM_PER_DEG_LAT
    out_lon = lon + east_km / (KM_PER_DEG_LAT
                               * math.cos(math.radians(lat)))
    return out_lat, out_lon


def motion_from_centroids(older: dict, newer: dict, *,
                          min_gates: int) -> dict | None:
    """Storm motion (m/s east/north) from two volumes' echo centroids.

    ``older``/``newer``: {"valid_time", "gates", "centroid_east_km",
    "centroid_north_km"}.  None when either volume's echo is too weak
    for a displacement to mean anything, or the pair is too close in
    time -- the caller records WHY in the receipt.
    """

    if older["gates"] < min_gates or newer["gates"] < min_gates:
        return None
    # The displacement is between two scans, so the baseline is between
    # the instants the radials were collected, not between two archive
    # key names; the key time stands in only for a pack that did not say.
    dt = (parse_iso(newer.get("start_time") or newer["valid_time"])
          - parse_iso(older.get("start_time") or older["valid_time"])
          ).total_seconds()
    if dt < 300.0:
        return None
    u = (newer["centroid_east_km"] - older["centroid_east_km"]) \
        * 1000.0 / dt
    v = (newer["centroid_north_km"] - older["centroid_north_km"]) \
        * 1000.0 / dt
    speed = math.hypot(u, v)
    toward_deg = math.degrees(math.atan2(u, v)) % 360.0
    return {"u_ms": round(u, 2), "v_ms": round(v, 2),
            "speed_ms": round(speed, 2),
            "toward_deg": round(toward_deg, 1),
            "baseline_seconds": dt,
            "caveat": ("centroid displacement mixes advection with "
                       "growth/decay; good enough to bias the domain "
                       "downstream, nothing more")}


def site_domain_center(antenna_lat: float, antenna_lon: float,
                       newest: dict, motion: dict | None, *,
                       horizon_seconds: float,
                       downstream_fraction: float,
                       max_offset_km: float) -> dict:
    """Center the domain on the echo, biased downstream, clamped so the
    radar keeps observing most of the grid."""

    east = newest["centroid_east_km"]
    north = newest["centroid_north_km"]
    basis = "echo centroid"
    if newest["gates"] <= 0:
        east = north = 0.0
        basis = "antenna (no qualifying echo)"
    if motion is not None and motion["speed_ms"] > 0.5:
        lead_km = (motion["speed_ms"] * horizon_seconds / 1000.0
                   * downstream_fraction)
        east += motion["u_ms"] / motion["speed_ms"] * lead_km
        north += motion["v_ms"] / motion["speed_ms"] * lead_km
        basis += " + downstream lead"
    total = math.hypot(east, north)
    clamped = False
    if total > max_offset_km:
        east *= max_offset_km / total
        north *= max_offset_km / total
        clamped = True
    lat, lon = offset_latlon(antenna_lat, antenna_lon, east, north)
    return {"lat": round(lat, 4), "lon": round(lon, 4),
            "offset_east_km": round(east, 1),
            "offset_north_km": round(north, 1),
            "basis": basis, "clamped_to_km": max_offset_km if clamped
            else None}


def echo_stats(volume, *, threshold_dbz: float,
               max_sweep_elevation_deg: float = 2.0) -> dict:
    """Gate census + centroid of the >= threshold echo on the low sweeps."""

    import numpy as np

    gates = 0
    max_dbz = float("-inf")
    sum_x = sum_y = 0.0
    for sweep in volume.sweeps:
        if sweep.elevation_angle_deg > max_sweep_elevation_deg:
            continue
        moment = sweep.moments.get("REF")
        if moment is None:
            continue
        data = np.asarray(moment.data, float)
        mask = np.isfinite(data) & (data >= threshold_dbz)
        count = int(mask.sum())
        if count == 0:
            continue
        slant = moment.slant_range_m()
        horiz = slant * math.cos(math.radians(
            sweep.elevation_angle_deg))
        az = np.radians(np.asarray(sweep.azimuth_deg, float))
        x = np.sin(az)[:, None] * horiz[None, :]
        y = np.cos(az)[:, None] * horiz[None, :]
        gates += count
        sum_x += float(x[mask].sum())
        sum_y += float(y[mask].sum())
        max_dbz = max(max_dbz, float(data[mask].max()))
    if gates == 0:
        return {"gates": 0, "max_dbz": None,
                "centroid_east_km": 0.0, "centroid_north_km": 0.0}
    return {"gates": gates, "max_dbz": round(max_dbz, 1),
            "centroid_east_km": round(sum_x / gates / 1000.0, 1),
            "centroid_north_km": round(sum_y / gates / 1000.0, 1)}


# ---------------------------------------------------------------------------
def offered_physics_profiles() -> tuple[str, ...]:
    """The suites this door offers, which is the set its route can run.

    ONE TABLE FOR BOTH DOORS.  This door's first stage is ``woof
    domain``, so a suite it offers and that door refuses is not a suite
    this door can run at all -- which is what a profile present in the
    runner's own tables and absent from the configuration door's choice
    list produced: two rows that died in argument parsing with no
    physics reached.  Both doors read
    :func:`woof.physics_menu.shipped_profiles`, which is derived from
    the registry rather than transcribed, so a suite registered tomorrow
    reaches both with no edit at either.
    """

    from woof.physics_menu import shipped_profiles

    return tuple(shipped_profiles())


def current_physics_profile(value: str) -> str:
    """An old profile ID as its current ID, before the offered-list check.

    The offered list holds current IDs only, so without this an old ID that
    every stage of this run accepts was refused in argument parsing.
    """

    from woof.physics_registry import canonical_template_id

    return str(canonical_template_id(value))


# stage command builders (unit-tested; every stage is a shipped CLI)
# ---------------------------------------------------------------------------
def _py() -> str:
    return sys.executable or "python"


def wizard_cmd(*, polygon: Path, out_toml: Path, plan: WindowPlan,
               profile: str, name: str, dx_km: float, source: str,
               vram_gib: float | None = None) -> list[str]:
    """The domain wizard's command line for this run's case.

    ``vram_gib`` names the card the memory preflight sizes against.
    Left unset, the wizard uses its own default tier -- which is the
    right answer for a caller who did not say, and the wrong one for a
    caller who was SHOWN a verdict for a particular card and is now
    asking for the run that verdict was about.
    """

    argv = [
        _py(), "-m", "woof.cli", "domain",
        "--polygon", str(polygon),
        "--root-dx", f"{dx_km:g}",
        "--physics-profile", profile,
        "--source", source,
        "--cycle", plan.background_cycle.strftime("%Y-%m-%dT%H"),
        "--forecast-start-hour", str(plan.forecast_start_hour),
        "--hours", str(plan.run_hours),
        "--name", name,
        "--out", str(out_toml),
    ]
    if vram_gib is not None:
        argv.extend(("--vram-gib", f"{float(vram_gib):g}"))
    return argv


def authority_cmd(*, case_toml: Path, wps_namelist: Path, profile: str,
                  authority_dir: Path, source: str) -> list[str]:
    return [
        _py(), "-m", "woof.prepared_single_domain_forecast",
        "--materialize-authorities", "--source", source,
        "--base-experiment-config", str(case_toml),
        "--base-wps-namelist", str(wps_namelist),
        "--physics-profile", profile,
        "--output-directory", str(authority_dir),
    ]


def fetch_cmd(*, hints: dict, data_dir: Path, source: str,
              now: datetime | None = None) -> list[str]:
    argv = [
        _py(), "-m", "woof.cli", "fetch", "--source", source,
        "--cycle", str(hints["cycle"]),
        "--hours", str(hints["hours"]),
        f"--area={hints['area']}",
        "--out", str(data_dir),
    ]
    if "cadence" in hints:
        # The wizard emits a cadence only for sources that HAVE one to
        # choose (GFS 1/3 h); `woof fetch --source hrrr` refuses the
        # flag outright because HRRR is hourly and nothing else.
        argv.extend(("--cadence", str(hints["cadence"])))
    # `.get(..., 0)`, not `[...]`: the wizard writes forecast_start_hour
    # into its [fetch] hints only when it is nonzero, and an init that
    # lands exactly on a background cycle -- the COMMON case on hourly
    # HRRR cycles, and any synoptic-hour window end on GFS -- is lead 0.
    # Indexing here raised KeyError on exactly those runs (issue #74).
    start = int(hints.get("forecast_start_hour", 0))
    argv.extend(("--forecast-start-hour", str(start)))
    from woof.da.background import late_background_wait_minutes

    wait = late_background_wait_minutes(
        source, datetime.strptime(str(hints["cycle"])[:13], "%Y-%m-%dT%H"),
        start + int(hints["hours"]),
        now=now if now is not None else datetime.now(timezone.utc))
    if wait is not None:
        # The plan took the newest cycle whose window is due on its
        # posting row's expected line, so a cycle running late can still
        # be posting its last leads: the fetch waits for each one, saying
        # which, until the window's last lead is late on its row, instead
        # of refusing the cycle as unpublished.
        argv.extend(("--wait-for", "--wait-timeout-minutes", str(wait)))
    return argv


def manifest_cmd(*, data_dir: Path, authority_dir: Path,
                 source: str) -> list[str]:
    return [
        _py(), "-m", "woof.cli", "fetch", "--source", source,
        "--author-front-door-manifest", "--out", str(data_dir),
        "--wps-namelist", str(authority_dir / "namelist.wps"),
        "--experiment-config", str(authority_dir / "experiment.toml"),
    ]


def prepare_cmd(*, data_dir: Path, authority_dir: Path,
                bridge: Path | None,
                profile: str, geog_root: Path, prepared_root: Path,
                plan: WindowPlan, manifest_sha: str, source: str,
                namelist_input: Path | None = None,
                domain_spec: Path | None = None,
                history_interval_seconds: float | None = None
                ) -> list[str]:
    """The preparation command, in the SOURCE's own argument grammar.

    The two grammars are different on purpose and neither is bent to
    the other: the GFS route reads a decoded series plus the manifest
    the fetch authored, while the HRRR route reads the raw GRIB2
    directory bound by its fetch-written ``SHA256SUMS`` and derives
    every stage clock from cycle + lead (``woof.source_cli``'s own
    ``--source hrrr`` contract; ``--experiment-config`` and ``--bridge``
    are refused there).  ``manifest_sha`` is therefore the digest of a
    DIFFERENT file per source, and the three HRRR-only keyword
    arguments are exactly the wizard-emitted route inputs the GFS
    grammar has no use for.
    """

    if source == "hrrr":
        return [
            _py(), "-m", "woof.source_cli", "--source", source,
            "--source-root", str(data_dir),
            "--source-manifest", str(data_dir / "SHA256SUMS"),
            "--source-manifest-sha256", manifest_sha,
            "--namelist-input", str(namelist_input),
            "--domain-spec", str(domain_spec),
            "--wps-namelist", str(authority_dir / "namelist.wps"),
            "--valid-time",
            plan.background_cycle.strftime("%Y-%m-%d_%H:00:00"),
            "--forecast-start-hour", str(plan.forecast_start_hour),
            "--run-seconds", str(plan.run_seconds),
            "--history-interval-seconds",
            f"{float(history_interval_seconds):g}",
            "--physics-profile", profile,
            "--geog-root", str(geog_root),
            "--output-root", str(prepared_root),
        ]
    return [
        _py(), "-m", "woof.source_cli", "--source", source,
        "--gfs-series", str(data_dir / "gfs-series.tsv"),
        "--cycle", plan.background_cycle.strftime("%Y-%m-%d_%H:00:00"),
        "--bridge", str(bridge),
        "--wps-namelist", str(authority_dir / "namelist.wps"),
        "--experiment-config", str(authority_dir / "experiment.toml"),
        "--source-manifest", str(data_dir / "gfs-input-manifest.json"),
        "--source-manifest-sha256", manifest_sha,
        "--physics-profile", profile,
        "--geog-root", str(geog_root),
        "--output-root", str(prepared_root),
    ]


def forecast_cmd(*, prepared_root: Path, authority_dir: Path,
                 profile: str, run_dir: Path, proof_sha: str,
                 manifest_sha: str, content_sha: str,
                 source: str) -> list[str]:
    return [
        _py(), "-m", "woof.prepared_single_domain_forecast",
        "--source", source,
        "--prepared-root", str(prepared_root),
        "--proof-sha256", proof_sha,
        "--source-manifest-sha256", manifest_sha,
        "--prepared-content-sha256", content_sha,
        "--experiment-config", str(authority_dir / "experiment.toml"),
        "--wps-namelist", str(authority_dir / "namelist.wps"),
        "--physics-profile", profile,
        "--io-mode", "history",
        "--outdir", str(run_dir),
    ]


@dataclass(frozen=True)
class RadarSelection:
    """Which radars every observation of this case is built from.

    One radar measures the wind along its own beam and nothing else, so
    a single-radar analysis is one projection of a three-component
    field.  ``tools/obs_radar_grid_build.py`` has accepted several
    radars per analysis time since the Level-II work; this type is how
    the front door asks for them, and it is the ONE place the request
    is turned into that tool's argv -- the cycle's observations and the
    rolling verifier's observed composites are built from the same
    object, so a case cannot grade itself against a thinner (or
    thicker) analysis than it assimilated.

    ``anchor`` is always the surveyed site: it sites the domain, it
    names the case's files, and it is what a receipt means by
    ``site``.  It is distinct from the radars that CONTRIBUTE, which
    is why both are carried.

    Three shapes, and the default is the old behaviour exactly:

    * neither ``sites`` nor ``discover``  -> ``--site <anchor>``, the
      single-radar argv this function emitted before multi-radar
      existed, byte for byte;
    * ``sites``    -> ``--site`` repeated, anchor first, deduplicated;
    * ``discover`` -> ``--discover-sites``, and the radars are computed
      from the georeference by :mod:`woof.obs.coverage`.  The anchor
      is NOT named in this route -- the obs builder refuses to mix
      naming and discovery, on the grounds that the receipt could not
      then say which route chose which radar, and the anchor sits at
      the domain centre so discovery finds it with the best coverage
      of any site anyway.
    """

    anchor: str
    sites: tuple[str, ...] = ()
    discover: bool = False
    min_coverage_fraction: float | None = None
    max_radars: int | None = None
    min_radars: int | None = None
    max_time_spread_seconds: float | None = None

    @property
    def multi(self) -> bool:
        """Does this ask for more than the anchor radar?"""

        return bool(self.discover or self.sites)

    def argv_tail(self) -> list[str]:
        """The radar-choosing half of the obs builder's command line."""

        if self.discover:
            argv = ["--discover-sites"]
            if self.min_coverage_fraction is not None:
                argv += ["--min-coverage-fraction",
                         str(self.min_coverage_fraction)]
            if self.max_radars is not None:
                argv += ["--max-radars", str(self.max_radars)]
        else:
            named = [self.anchor, *self.sites]
            seen: set[str] = set()
            argv = []
            for site in named:
                if site not in seen:
                    seen.add(site)
                    argv += ["--site", site]
        # These two constrain any route: they are about what the
        # radars DELIVERED, not about how they were chosen.
        if self.min_radars is not None:
            argv += ["--min-radars", str(self.min_radars)]
        if self.max_time_spread_seconds is not None:
            argv += ["--max-radar-time-spread-seconds",
                     str(self.max_time_spread_seconds)]
        return argv

    def to_payload(self) -> dict:
        """The receipt's record of the request (not of the outcome).

        What each build actually got -- which radars answered, which
        refused and why, the measured time spread -- is the obs
        builder's own receipt, per file.  This says what was ASKED.
        """

        return {
            "anchor": self.anchor,
            "route": "discover" if self.discover else "named",
            "sites": list(self.sites),
            "multi_radar": self.multi,
            "min_coverage_fraction": self.min_coverage_fraction,
            "max_radars": self.max_radars,
            "min_radars": self.min_radars,
            "max_radar_time_spread_seconds": self.max_time_spread_seconds,
        }

    @classmethod
    def from_payload(cls, payload: dict | None, *,
                     anchor: str) -> "RadarSelection":
        """Recover a selection from a receipt, anchor-only if absent.

        A receipt written before this existed carries no ``radars``
        block, and the right reading of that is the single-radar run it
        was: the verifier then builds exactly what the cycle did.
        """

        if not payload:
            return cls(anchor=anchor)
        return cls(
            anchor=validate_site(payload.get("anchor") or anchor),
            sites=tuple(validate_site(s)
                        for s in payload.get("sites") or ()),
            discover=bool(payload.get("route") == "discover"),
            min_coverage_fraction=payload.get("min_coverage_fraction"),
            max_radars=payload.get("max_radars"),
            min_radars=payload.get("min_radars"),
            max_time_spread_seconds=payload.get(
                "max_radar_time_spread_seconds"),
        )


def radar_selection(args) -> RadarSelection:
    """The caller's radar request, refused early if contradictory."""

    named = tuple(validate_site(s) for s in (getattr(args, "sites", None)
                                             or ()))
    discover = bool(getattr(args, "discover_sites", False))
    if named and discover:
        raise FrontDoorError(
            "--sites and --discover-sites are mutually exclusive: "
            "naming some radars and discovering others leaves a receipt "
            "that cannot say which route chose which")
    return RadarSelection(
        anchor=args.site,
        sites=named,
        discover=discover,
        min_coverage_fraction=getattr(args, "min_coverage_fraction", None),
        max_radars=getattr(args, "max_radars", None),
        min_radars=getattr(args, "min_radars", None),
        max_time_spread_seconds=getattr(
            args, "max_radar_time_spread_seconds", None),
    )


@dataclass(frozen=True)
class DealiasChoice:
    """How this run unfolds velocity: whether, with which engine, refined.

    One object rather than three parameters, and a required one rather
    than a defaulted one, because of the invariant
    :func:`build_verify_frame` already states: whatever built the
    assimilated observations must also build the verification composites,
    or the run is graded against a differently-built truth and the
    difference shows up as skill.  A defaulted engine keyword would let a
    caller that was not updated build truth with the other solver and say
    nothing; a required object makes that a TypeError at import time.

    It rides the receipt for the same reason the ``dealias`` flag always
    has -- ``da_nowcast verify`` and ``watch`` run days later, from the
    receipt alone.
    """

    #: The engine every run before 2026-08-12 used, and the only one there
    #: was before the selector existed.  Named here rather than spelled
    #: inline twice: it is the value a receipt that predates the selector
    #: reads back as, which is a statement about history and must not
    #: track the shipped default when that moves.
    LEGACY_ENGINE = "vad-region"

    #: ON, like every radar door.  A receipt from a run that did NOT
    #: dealias has to say so, or its verification composites re-run WITH
    #: dealiasing and the A/B is graded against a differently-built truth.
    on: bool = True
    engine: str = DEFAULT_DEALIAS_ENGINE
    refinement: bool = True

    @classmethod
    def from_args(cls, args) -> "DealiasChoice":
        # ``dealias_refinement`` is tri-state on the parser -- None means
        # "the engine's default" -- and this object records what RAN, so
        # None is resolved here, by the same rule the parameter object
        # uses.  A receipt that said "refinement: null" would leave the
        # verification composites unable to state what they were built
        # with.
        on = bool(getattr(args, "dealias", True))
        named = getattr(args, "dealias_engine", None)
        engine = None if named is None else str(named)
        if on and engine is None:
            # The receipt records what will RUN, not what the default was
            # called: the argv this object builds names the engine
            # explicitly, and a named engine is refused rather than
            # resolved, so the resolution has to happen here instead of
            # one process further down.
            from woof.obs.dealias import first_available_engine

            engine = first_available_engine()
            if engine is None:
                # Neither engine can run on this install.  Masking only,
                # and the receipt says so rather than promising an unfold
                # its re-run cannot reproduce.
                on = False
        engine = str(engine or DEFAULT_DEALIAS_ENGINE)
        refinement = getattr(args, "dealias_refinement", None)
        if refinement is None:
            refinement = engine == DEFAULT_DEALIAS_ENGINE
        return cls(on=on, engine=engine, refinement=bool(refinement))

    @classmethod
    def from_payload(cls, payload) -> "DealiasChoice":
        """Read one back out of a receipt.

        A receipt with no ``obs`` block predates the flag entirely and
        reads back off; one with ``dealias`` and no ``dealias_engine``
        predates the selector and reads back as the engine that was the
        only one at the time, which is what those runs did.  That fallback
        stays pinned to :data:`LEGACY_ENGINE` now that the default has
        moved: an old receipt describes an old run, and re-running its
        verification with today's engine would grade a run against a
        differently-built truth.
        """

        payload = payload or {}
        engine = str(payload.get("dealias_engine", cls.LEGACY_ENGINE))
        return cls(on=bool(payload.get("dealias", False)), engine=engine,
                   refinement=bool(payload.get(
                       "dealias_refinement",
                       engine == DEFAULT_DEALIAS_ENGINE)))

    def to_payload(self) -> dict:
        return {"dealias": bool(self.on), "dealias_engine": str(self.engine),
                "dealias_refinement": bool(self.refinement)}

    def argv_tail(self) -> list[str]:
        # "Off" is spelled out too, now that on is the default.  An empty
        # tail would re-run a receipt's verification composites WITH
        # dealiasing for a run that did not dealias, which is exactly the
        # A/B break this object exists to prevent.
        if not self.on:
            return ["--no-dealias"]
        # Both switches are spelled out, always.  This argv builds the
        # verification composites as well as the assimilated observations,
        # and "whatever the default was on the day it re-ran" is not a
        # contract those two can share across a release.
        return ["--dealias-engine", self.engine,
                "--dealias-refinement" if self.refinement
                else "--no-dealias-refinement"]


def default_nest_ratio() -> int:
    """The refinement ratio a caller gets when they say nothing.

    Read off :class:`woof.da.nested_forecast.NestGeometry` rather than
    typed, so this door and the driver it forwards to cannot come to
    disagree about what 'the default nest' is.
    """

    from woof.da.nested_forecast import NestGeometry

    return int(NestGeometry.ratio)


@dataclass(frozen=True)
class NestChoice:
    """The fine nest over the free forecast, as the caller asked for it.

    This front door owns NO nest geometry.  Every field here is the
    prepared cycle driver's own flag with the driver's own default, and
    :meth:`argv_tail` forwards them unchanged, because the nest itself
    -- derivation, admissibility, the child's clock, its leg-end product
    -- is ``woof.da.nested_forecast`` driven from
    ``tools/da_cycle_prepared.py`` and a second implementation of any of
    it here would be a second thing to keep true.

    Asked for the same way the driver is asked: an EXTENT turns it on
    (``--nest-half-width-km``, or ``--nest-nx``/``--nest-ny``), and
    nothing else does.  A run that names no extent builds the same argv
    it built before this existed.
    """

    half_width_km: float | None
    nx: int | None
    ny: int | None
    ratio: int | None
    i_parent_start: int | None
    j_parent_start: int | None
    members: int | None
    history_interval_s: float | None
    acknowledge: tuple[str, ...]

    @property
    def requested(self) -> bool:
        return (self.half_width_km is not None or self.nx is not None
                or self.ny is not None)

    @property
    def resolved_ratio(self) -> int:
        # Resolved here and not in ``from_args`` so a run that asks for
        # no nest never imports the nest machinery: this module builds
        # argv for subprocesses and keeps a stdlib-only import surface
        # on the paths that do not need more (same rule as
        # :data:`DEFAULT_DEALIAS_ENGINE`).
        return (int(self.ratio) if self.ratio is not None
                else default_nest_ratio())

    @classmethod
    def from_args(cls, args) -> "NestChoice":
        return cls(
            half_width_km=getattr(args, "nest_half_width_km", None),
            nx=getattr(args, "nest_nx", None),
            ny=getattr(args, "nest_ny", None),
            ratio=getattr(args, "nest_ratio", None),
            i_parent_start=getattr(args, "nest_i_parent_start", None),
            j_parent_start=getattr(args, "nest_j_parent_start", None),
            members=getattr(args, "nest_members", None),
            history_interval_s=getattr(args, "nest_history_interval_s",
                                       None),
            acknowledge=tuple(getattr(args, "nest_acknowledge", None)
                              or ()))

    def argv_tail(self) -> list[str]:
        if not self.requested:
            return []
        argv = ["--nest-ratio", str(self.resolved_ratio)]
        for flag, value in (("--nest-half-width-km", self.half_width_km),
                            ("--nest-nx", self.nx),
                            ("--nest-ny", self.ny),
                            ("--nest-i-parent-start", self.i_parent_start),
                            ("--nest-j-parent-start", self.j_parent_start),
                            ("--nest-members", self.members),
                            ("--nest-history-interval-s",
                             self.history_interval_s)):
            if value is not None:
                argv.extend((flag, str(value)))
        for acknowledgement in self.acknowledge:
            argv.extend(("--nest-acknowledge", str(acknowledgement)))
        return argv

    def to_payload(self) -> dict:
        if not self.requested:
            return {"requested": False}
        return {"requested": True, "ratio": self.resolved_ratio,
                "half_width_km": self.half_width_km,
                "nx": self.nx, "ny": self.ny,
                "i_parent_start": self.i_parent_start,
                "j_parent_start": self.j_parent_start,
                "members": self.members,
                "history_interval_s": self.history_interval_s,
                "acknowledge": list(self.acknowledge),
                "receipt": "cycle/cycle-report.json -> nest"}


def obs_cmd(*, selection: RadarSelection, valid: datetime,
            grid_wrfout: Path, out_nc: Path, work_dir: Path,
            bucket: str | None, dealias: DealiasChoice) -> list[str]:
    """Build the obs-stage argv.

    ``dealias`` has to reach *both* callers of this function or the run
    scores itself against a differently-built truth.  See
    :func:`build_verify_frame`.
    """

    argv = [
        _py(), "-m", "tools.obs_radar_grid_build",
        *selection.argv_tail(),
        "--valid-time", iso(valid),
        "--grid-wrfout", str(grid_wrfout),
        "--out", str(out_nc),
        "--work-dir", str(work_dir),
        "--overwrite",
    ]
    if bucket:
        argv.extend(("--bucket", bucket))
    argv.extend(dealias.argv_tail())
    return argv


def cycle_cmd(*, prepared_root: Path, authority_dir: Path, profile: str,
              plan: WindowPlan, members: int, obs_files: list[Path],
              grid_wrfouts: list[Path], cycle_out: Path, proof_sha: str,
              manifest_sha: str, content_sha: str, seed: int,
              solve_device: str, horizontal_loc_m: float,
              vertical_loc_m: float, length_scale_km: float,
              source: str, leg_seconds: float | None = None,
              free_legs: int | None = None,
              free_leg_seconds: float | None = None,
              resume_ensemble: Path | None = None,
              save_ensemble: Path | None = None,
              leg_number_offset: int = 0,
              memory_budget_mib: float = DEFAULT_MEMORY_BUDGET_MIB,
              hydrometeors: bool = False,
              positivity_policy: str | None = None,
              reflectivity_analysis: bool = False,
              clear_air_analysis: bool = False,
              surface_obs: Path | None = None,
              sfc_t2_sigma_k: float | None = None,
              sfc_wspd_sigma_ms: float | None = None,
              sfc_max_age_s: float | None = None,
              goes_cwp: list[Path] | None = None,
              cwp_vertical_loc_m: float | None = None,
              nest: "NestChoice | None" = None) -> list[str]:
    """The cycle driver's command line for one run.

    ``plan`` supplies the run length and the default cadence; the
    optional overrides exist for a caller that cycles on the radar's own
    volume times rather than on a fixed cadence, and for one that carries
    the ensemble across processes.  Both callers build the SAME argv
    here, so the flag surface has one author.

    ``nest`` is the fine free-forecast nest, forwarded flag for flag to
    the driver that owns it (:class:`NestChoice`).  It reaches this argv
    only when the caller named an extent, so a run that asks for no nest
    builds exactly the argv it built before the nest was reachable from
    here.  Its cost model is COMPUTED rather than measured
    (``evidence/da-nested-forecast/cost-model.json`` says so in its own
    ``basis`` field); what a nested leg actually costs is in the cycle
    report the run writes, per leg, per trajectory.

    One capability is still deliberately absent from this argv:
    concurrent member advance (``--member-workers``).  It is not in this
    tree at all -- the lane carrying it was held out of the integration
    with its byte-identity proof still open.  There is nothing to turn
    on and nothing to turn off.
    """

    cadence = (float(plan.cycle_seconds) if leg_seconds is None
               else float(leg_seconds))
    legs_free = plan.free_legs if free_legs is None else int(free_legs)
    argv = [
        _py(), "-m", "tools.da_cycle_prepared",
        "--prepared-root", str(prepared_root),
        "--authority-dir", str(authority_dir),
        "--source", source,
        "--proof-sha256", proof_sha,
        "--source-manifest-sha256", manifest_sha,
        "--prepared-content-sha256", content_sha,
        "--physics-profile", profile,
        "--run-seconds", str(plan.run_seconds),
        "--history-interval-seconds", str(plan.cycle_seconds),
        "--leg-seconds", str(cadence),
        "--members", str(members),
        "--free-legs", str(legs_free),
        "--solve-device", solve_device,
        "--save-composites",
        "--horizontal-loc-m", str(horizontal_loc_m),
        "--vertical-loc-m", str(vertical_loc_m),
        "--length-scale-km", str(length_scale_km),
        "--memory-budget-mib", f"{float(memory_budget_mib):g}",
        "--seed", str(seed),
    ]
    for obs in obs_files:
        argv.extend(("--obs", str(obs)))
    for grid in grid_wrfouts:
        argv.extend(("--grid-wrfout", str(grid)))
    if hydrometeors:
        # The science switch, explicit here exactly as it is on the
        # driver: moisture/hydrometeor state is perturbed AND analysed,
        # and a positivity policy must be stated because clip / reject /
        # none are not equivalent (woof.da.positivity).
        argv.append("--hydrometeors")
        argv.extend(("--positivity-policy", str(positivity_policy)))
    if reflectivity_analysis:
        argv.append("--reflectivity-analysis")
    if clear_air_analysis:
        argv.append("--clear-air-analysis")
    # The non-radar streams. They reach the driver from here or not at
    # all: before this, --goes-cwp and --surface-obs existed only on the
    # driver's own command line, so no run launched from this front door
    # -- which is every WaH run -- could have assimilated either of them.
    if surface_obs is not None:
        argv.extend(("--surface-obs", str(surface_obs)))
        # A surface QUANTITY is enabled by stating its error stddev, so
        # these are not decoration: --surface-obs with neither sigma
        # assimilates nothing and the driver refuses it.
        if sfc_t2_sigma_k is not None:
            argv.extend(("--sfc-t2-sigma-k", str(float(sfc_t2_sigma_k))))
        if sfc_wspd_sigma_ms is not None:
            argv.extend(("--sfc-wspd-sigma-ms",
                         str(float(sfc_wspd_sigma_ms))))
        if sfc_max_age_s is not None:
            argv.extend(("--sfc-max-age-s", str(float(sfc_max_age_s))))
    for path in (goes_cwp or ()):
        # One file per leg, in leg order; the driver matches them to legs
        # by position and refuses more files than legs.
        argv.extend(("--goes-cwp", str(path)))
    if cwp_vertical_loc_m is not None:
        argv.extend(("--cwp-vertical-loc-m", str(float(cwp_vertical_loc_m))))
    if nest is not None:
        argv.extend(nest.argv_tail())
    if free_leg_seconds is not None:
        argv.extend(("--free-leg-seconds", str(float(free_leg_seconds))))
    if resume_ensemble is not None:
        argv.extend(("--resume-ensemble", str(resume_ensemble)))
    if save_ensemble is not None:
        argv.extend(("--save-ensemble", str(save_ensemble)))
    if leg_number_offset:
        argv.extend(("--leg-number-offset", str(leg_number_offset)))
    argv.extend(("--out", str(cycle_out)))
    return argv


def render_cmd(*, case_dir: Path, gallery: Path | None = None,
               authority_dir: Path | None = None) -> list[str]:
    """The gallery renderer's command line.

    ``gallery`` and ``authority_dir`` are for a caller whose case
    directory is assembled rather than produced by one run -- a
    continuous nowcast renders many cycles into ONE stable gallery path,
    which is what makes refreshing the page the whole user experience.
    """

    argv = [_py(), "-m", "tools.da_nowcast_render",
            "--case-dir", str(case_dir)]
    if gallery is not None:
        argv.extend(("--gallery", str(gallery)))
    if authority_dir is not None:
        argv.extend(("--authority-dir", str(authority_dir)))
    return argv


def watch_cmd(*, case_dir: Path, poll_seconds: int, max_minutes: int,
              max_offset_seconds: float,
              bucket: str | None) -> list[str]:
    """The detached rolling verifier's own command line.

    It is this same module: the auto-handoff spawns a documented CLI
    mode, so anything the front door does to itself can be done by
    hand, and the receipt records exactly what was started.
    """

    argv = [_py(), "-m", "tools.da_nowcast", "watch",
            "--case-dir", str(case_dir),
            "--poll-seconds", str(poll_seconds),
            "--max-minutes", str(max_minutes),
            "--max-offset-seconds", str(max_offset_seconds)]
    if bucket:
        argv.extend(("--bucket", bucket))
    return argv


# ---------------------------------------------------------------------------
# the verification state machine (pure; the receipt is the GUI's seam)
# ---------------------------------------------------------------------------
def verify_obs_name(site: str, valid: datetime) -> str:
    """The observed-composite file backing one free-forecast frame."""

    return f"verify-{site.lower()}-{valid:%Y%m%d%H%M}.nc"


def initial_frames(free_leg_times) -> list[dict]:
    """One pending entry per free-forecast leg, in valid-time order."""

    return [{"valid": str(stamp), "status": "pending"}
            for stamp in free_leg_times]


def merge_gallery_rows(frames, rows) -> list[dict]:
    """Fold the renderer's published per-frame numbers into the frames.

    The renderer is the ONE place counts and FSS are computed; the
    verifier copies its rows rather than recomputing them, so the
    receipt and the figures can never disagree.  A frame only reaches
    ``verified`` when its numbers exist.
    """

    by_valid = {str(row.get("valid")): row for row in rows or []}
    merged = []
    for frame in frames:
        row = by_valid.get(frame["valid"])
        if row is None:
            merged.append(dict(frame))
            continue
        entry = dict(frame)
        entry.update(row)
        entry["valid"] = frame["valid"]
        entry["status"] = "verified"
        entry.pop("note", None)
        merged.append(entry)
    return merged


def advance_state(frames, *, exhausted: bool) -> str:
    """Where the machine sits, given what carries numbers so far."""

    if all(frame.get("status") == "verified" for frame in frames):
        return "complete"      # vacuously true when there are no legs
    return "incomplete" if exhausted else "rolling"


def verdict_line(frames, state: str) -> str:
    total = len(frames)
    graded = sum(1 for f in frames if f.get("status") == "verified")
    if state == "disabled":
        return "verification not started (--no-verify)"
    if state == "pending":
        return (f"0/{total} free-forecast frames graded; the verifier "
                "has not started")
    if state == "rolling":
        return (f"{graded}/{total} free-forecast frames graded; "
                "waiting for the archive to cover the rest")
    if state == "complete":
        return (f"{graded}/{total} free-forecast frames graded against "
                "observed composites at the same valid times "
                "(demo-grade numbers, unscored)")
    return (f"{graded}/{total} free-forecast frames graded; the rest "
            "were still not covered by the archive when the verifier "
            "stopped")


def verification_block(*, frames, state: str, started: str | None = None,
                       watcher: dict | None = None) -> dict:
    """The receipt's ``verification`` key: one pollable object."""

    if state not in VERIFY_STATES:
        raise FrontDoorError(f"unknown verification state {state!r}")
    return {
        "schema": VERIFY_SCHEMA,
        "state": state,
        "verdict": verdict_line(frames, state),
        "graded": sum(1 for f in frames
                      if f.get("status") == "verified"),
        "total": len(frames),
        "started": started,
        "updated": iso(datetime.now(timezone.utc)),
        "watcher": watcher,
        "honesty": ("demo-grade verification numbers, unscored and "
                    "outside any registered campaign"),
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# verification execution
# ---------------------------------------------------------------------------
def _log(text: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {text}", flush=True)


def read_receipt(case_dir: Path) -> dict:
    path = case_dir / "nowcast-receipt.json"
    if not path.is_file():
        raise FrontDoorError(f"{case_dir} carries no "
                             "nowcast-receipt.json")
    return json.loads(path.read_text(encoding="utf-8"))


#: The five things `_case_context` reads out of a receipt, which is the
#: whole of what `da_nowcast verify` and `watch` need.  Named here so the
#: partial receipt below is written against the requirement rather than
#: against a guess at it.
VERIFY_REQUIRED_FIELDS = ("site", "radars", "plan.init",
                          "plan.free_leg_times", "obs.dealias",
                          "obs.dealias_engine")


def install_provenance() -> dict:
    """This install's identity, or a recorded reason there is none.

    Same resolver every other woof receipt uses (manifest, then a real
    checkout of this tree, then the installed wheel record), so a
    partial receipt can be told apart from a hand-made one by a reader
    who was not there.  It never raises: a provenance nicety may not
    take down a receipt the run is writing on its way out.
    """

    try:
        from woof import runtime_manifest
        from woof import __version__

        root = Path(runtime_manifest.__file__).resolve().parent.parent
        block = {"gpuwm_version": __version__, "package_root": str(root)}
        block.update(runtime_manifest.provenance(root))
        return block
    except Exception as error:                          # noqa: BLE001
        return {"identity_source": None,
                "identity_error": f"{type(error).__name__}: {error}"}


def write_partial_receipt(out: Path, *, receipts: Path,
                          dealias: "DealiasChoice",
                          stopped_after: str) -> Path | None:
    """The receipt a run that stops early still owes its verifier.

    Every `--stop-after` door returns before the full receipt is
    written, so a two-phase case -- the supported route when a later
    stage has to be driven directly, as `tools/da_cycle_prepared` is --
    used to leave a case directory that `da_nowcast verify` refused with
    "carries no nowcast-receipt.json".  The fix people found was to
    hand-synthesize the file; this emits it instead.

    NOTHING HERE IS INVENTED.  `site`, `radars`, `plan`, `domain_center`,
    `seed` and `case_name` are copied verbatim from the run's own
    `receipts/01-plan.json`.  `obs.dealias` is the one field the verifier
    reads that the plan receipt has never carried, and it comes from the
    launch command.  Fields a completed run would fill -- sizing,
    survey, outputs, length_scale_km, the rest of the obs block -- are
    OMITTED rather than guessed, and `receipt_provenance` says so, so a
    reader can tell this apart from a front-door receipt.

    `partial: true` is already the marker `01-plan.json` uses.  Returns
    the path written, or None when there is nothing accurate to write.
    """

    plan_path = receipts / "01-plan.json"
    if not plan_path.is_file():
        # Every door this runs at is after the plan receipt, so this is
        # a run that failed before planning.  Nothing to copy.
        return None
    try:
        plan_receipt = json.loads(plan_path.read_text(encoding="utf-8"))
    except ValueError:
        return None

    target = out / "nowcast-receipt.json"
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
        if not existing.get("partial", False):
            # A complete receipt from an earlier full run of this case
            # outranks a partial one; downgrading it would lose the
            # sizing, outputs and verification a reader depends on.
            return None

    verbatim = [key for key in ("site", "radars", "plan", "domain_center",
                                "seed", "case_name")
                if key in plan_receipt]
    receipt = {"schema": SCHEMA, "partial": True,
               "generated": iso(datetime.now(timezone.utc)),
               "honesty": ("demo-grade nowcast; UNSCORED, outside any "
                           "registered campaign; no skill claim is made "
                           "or implied")}
    receipt.update({key: plan_receipt[key] for key in verbatim})
    receipt["obs"] = dealias.to_payload()
    receipt["receipt_provenance"] = {
        "written_by": f"da_nowcast run --stop-after {stopped_after}",
        "why": ("the run stopped at a --stop-after door, before the "
                "stage that writes the complete receipt; this carries "
                "the fields the verify door reads so a two-phase case "
                "does not have to hand-synthesize them"),
        "verbatim_from_receipts_01_plan_json": verbatim,
        "stated_from_the_launch_command": {
            "obs.dealias": "whether da_nowcast run dealiased "
                           "(--no-dealias turns it off)",
            "obs.dealias_engine": "--dealias-engine passed to da_nowcast run",
            "obs.dealias_refinement":
                "--dealias-refinement passed to da_nowcast run"},
        "not_reconstructed": ["sizing", "survey", "outputs",
                              "length_scale_km", "members",
                              "obs.streams_requested"],
        "note": ("fields a completed front-door run would fill were left "
                 "out rather than guessed; this is not a complete "
                 "receipt and says so with partial = true"),
        "install": install_provenance(),
    }
    target.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return target


def write_verification(case_dir: Path, block: dict) -> None:
    """Rewrite ONLY the receipt's verification key, atomically.

    Atomically because this file is the polling seam: a GUI reading it
    mid-tick must never catch a half-written receipt.
    """

    path = case_dir / "nowcast-receipt.json"
    receipt = read_receipt(case_dir)
    receipt["verification"] = block
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def gallery_rows(case_dir: Path) -> list[dict]:
    """Per-frame numbers the renderer published, if any."""

    path = case_dir / "gallery" / "_verification.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("frames", [])


def spawn_detached(argv: list[str], *, cwd: Path, log_path: Path) -> int:
    """Start a process that outlives this one; return its pid."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP",
                          0x00000200))
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            argv, cwd=str(cwd), stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            **kwargs)
    finally:
        handle.close()
    return proc.pid


def handoff_state(*, no_verify: bool, frames) -> str:
    """What a finished run's verification starts as.

    Verification is ON unless the caller opted out; a run with no free
    forecast has nothing to grade and is complete on arrival.
    """

    if no_verify:
        return "disabled"
    return "pending" if frames else "complete"


def start_verification(case_dir: Path, *, frames, poll_seconds: int,
                       max_minutes: int, max_offset_seconds: float,
                       bucket: str | None, repo_root: Path) -> dict:
    """Hand the finished case to a detached rolling verifier.

    Returns the ``rolling`` block it wrote into the receipt, which
    names the pid and the exact command line, so the handoff is
    auditable and reproducible by hand.
    """

    started = iso(datetime.now(timezone.utc))
    log_path = case_dir / "verify-watch.log"
    argv = watch_cmd(case_dir=case_dir, poll_seconds=poll_seconds,
                     max_minutes=max_minutes,
                     max_offset_seconds=max_offset_seconds,
                     bucket=bucket)
    pid = spawn_detached(argv, cwd=repo_root, log_path=log_path)
    block = verification_block(
        frames=frames, state="rolling", started=started,
        watcher={"pid": pid, "argv": argv, "log": str(log_path),
                 "poll_seconds": poll_seconds,
                 "max_minutes": max_minutes})
    write_verification(case_dir, block)
    return block


def build_verify_frame(*, case_dir: Path, selection: RadarSelection,
                       init: datetime,
                       valid: datetime, bucket: str | None,
                       max_offset_seconds: float,
                       repo_root: Path,
                       dealias: DealiasChoice) -> tuple[bool, str]:
    """Build one frame's observed composite from the live archive.

    Built from the SAME :class:`RadarSelection` the cycle assimilated,
    recovered from the receipt.  A multi-radar nowcast graded against a
    single-radar truth field would be scored on a different observation
    than it was given, and the difference would look like skill.

    ``dealias`` is recovered from the receipt for exactly the same reason
    and carries exactly the same hazard.  Dealiasing changes which velocity
    gates exist and what they are worth; a run that assimilated unfolded
    velocities and was then graded against a masked-only truth field would
    be scored on an observation nobody gave it.
    """

    obsverify = case_dir / "obsverify"
    obsverify.mkdir(parents=True, exist_ok=True)
    out_nc = obsverify / verify_obs_name(selection.anchor, valid)
    grid = (case_dir / "run" / "wrfout"
            / wrfout_name(init, (valid - init).total_seconds()))
    argv = obs_cmd(selection=selection, valid=valid, grid_wrfout=grid,
                   out_nc=out_nc, work_dir=case_dir / "vols",
                   bucket=bucket, dealias=dealias)
    argv.extend(("--max-offset-seconds", str(max_offset_seconds)))
    proc = subprocess.run(argv, cwd=str(repo_root), capture_output=True,
                          text=True, errors="replace")
    (obsverify / f"build-{valid:%Y%m%d%H%M}.json").write_text(
        json.dumps({"argv": argv, "returncode": proc.returncode,
                    "attempted": iso(datetime.now(timezone.utc)),
                    "stdout_tail": proc.stdout.splitlines()[-20:],
                    "stderr_tail": proc.stderr.splitlines()[-20:]},
                   indent=1), encoding="utf-8")
    if proc.returncode == 0:
        return True, "built"
    if out_nc.is_file():
        out_nc.unlink()   # never leave a half-written observation behind
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
    return False, " ".join(tail) or f"obs build exit {proc.returncode}"


def verify_pass(*, case_dir: Path, selection: RadarSelection,
                init: datetime,
                frames: list[dict], bucket: str | None,
                max_offset_seconds: float,
                repo_root: Path,
                dealias: DealiasChoice) -> tuple[list[dict], int]:
    """One sweep: build what the archive now covers, re-render if new.

    The observation file on disk is the record of what was fetched;
    the renderer's published rows are the record of what is graded.
    """

    now = datetime.now(timezone.utc)
    built = 0
    for frame in frames:
        valid = parse_iso(frame["valid"])
        out_nc = (case_dir / "obsverify"
                  / verify_obs_name(selection.anchor, valid))
        if out_nc.is_file():
            frame["obs_file"] = out_nc.name
            continue
        if (now - valid).total_seconds() < WATCH_MIN_AGE_SECONDS:
            frame["note"] = "valid time not reached yet"
            continue
        ok, why = build_verify_frame(
            case_dir=case_dir, selection=selection, init=init, valid=valid,
            bucket=bucket, max_offset_seconds=max_offset_seconds,
            repo_root=repo_root, dealias=dealias)
        if ok:
            built += 1
            frame["obs_file"] = out_nc.name
            frame.pop("note", None)
            _log(f"built observed composite for {frame['valid']}")
        else:
            frame["note"] = why
    if built:
        proc = subprocess.run(render_cmd(case_dir=case_dir),
                              cwd=str(repo_root), capture_output=True,
                              text=True, errors="replace")
        if proc.returncode == 0:
            _log("gallery re-rendered in place")
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            _log("RENDER FAILED: " + (tail[-1] if tail else "no output"))
    return merge_gallery_rows(frames, gallery_rows(case_dir)), built


def resolve_bridge(explicit: Path | None) -> Path:
    """The gfs GRIB2 bridge, via the standard artifact ladder."""

    from woof.bridges import artifact_candidates, executable_name
    if explicit is not None:
        if not explicit.is_file():
            raise FrontDoorError(f"--bridge names a missing file: "
                                 f"{explicit}")
        return explicit
    for candidate in artifact_candidates(
            "WOOF_GFS_GRIB2_BRIDGE",
            executable_name("gfs_grib2_bridge")):
        if candidate.is_file():
            return candidate
    raise FrontDoorError(
        "no gfs_grib2_bridge found; build tools/grib1_bridge "
        "(cargo build --release --locked --offline) or pass --bridge")


def resolve_geog_root(explicit: Path | None) -> Path:
    env = os.environ.get("WOOF_GEOG_ROOT")
    for candidate in (explicit, Path(env) if env else None,
                      Path.home() / "WPS_GEOG"):
        if candidate is not None and candidate.is_dir():
            return candidate
    raise FrontDoorError(
        "no static-geography root: pass --geog-root, set "
        "WOOF_GEOG_ROOT, or place WPS_GEOG in the home directory")


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def run_stage(name: str, argv: list[str], *, cwd: Path,
              receipts_dir: Path, index: int) -> dict:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    print(f"=== {name} {started:%H:%M:%S}Z ===", flush=True)
    proc = subprocess.run(argv, cwd=str(cwd), capture_output=True,
                          text=True, errors="replace")
    wall = time.monotonic() - t0
    receipt = {
        "schema": STAGE_SCHEMA,
        "stage": name,
        "argv": argv,
        "returncode": proc.returncode,
        "wall_seconds": round(wall, 1),
        "started": iso(started),
        "stdout_tail": proc.stdout.splitlines()[-30:],
        "stderr_tail": proc.stderr.splitlines()[-30:],
    }
    receipts_dir.mkdir(parents=True, exist_ok=True)
    (receipts_dir / f"{index:02d}-{name}.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(receipt["stderr_tail"][-8:]
                         or receipt["stdout_tail"][-8:])
        raise FrontDoorError(
            f"stage {name} failed (exit {proc.returncode}):\n{tail}")
    print(f"=== {name} done in {wall:.0f}s ===", flush=True)
    return receipt


def gpu_snapshot() -> str:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20)
        return probe.stdout.strip() or probe.stderr.strip()
    except OSError:
        return "nvidia-smi unavailable"


def survey_site(site: str, *, work_dir: Path, clock: SurveyClock,
                span_seconds: float, motion_baseline_s: float,
                echo_threshold_dbz: float, min_echo_gates: int,
                max_lag_seconds: float, allow_stale: bool,
                bucket: str | None, range_km: float | None) -> dict:
    """S3 survey: freshness, echo census, motion, and the two decodes.

    ``clock`` decides which window is listed and what the lag is
    measured against: the requested window's end for a replay, the wall
    clock for a live run (:class:`SurveyClock`).  Which one was used is
    written into the receipt, because a survey is only readable beside
    the moment it asked about.
    """

    at = clock.at

    from woof.obs.nexrad import (find_nexrad_bin, nexrad_remedy,
                                  run_decode, run_fetch, run_list,
                                  run_verify)
    from woof.obs.superob import SuperobParams
    from woof.obs.sweeps import read_sweep_pack

    binary = find_nexrad_bin()
    if binary is None:
        raise FrontDoorError(f"no rw_nexrad front door: {nexrad_remedy()}")
    authority_km = (range_km if range_km is not None
                    else SuperobParams().max_range_km)

    listing_start = at - timedelta(seconds=span_seconds)
    listing = run_list(
        binary, site=site,
        start=iso(listing_start), end=iso(at),
        bucket=bucket)
    volumes = sorted(
        (v for v in listing.get("volumes", [])
         if not v["filename"].endswith("MDM")),
        key=lambda v: v["valid_time"])
    if not volumes:
        raise FrontDoorError(
            f"{site}: no volumes in the {span_seconds / 60:.0f} minutes "
            f"before {iso(at)} ({clock.subject}) -- "
            + ("site down, id wrong, or archive far behind"
               if clock.mode == "wall" else
               "id wrong, or the archive does not reach that window"))
    work_dir.mkdir(parents=True, exist_ok=True)
    antenna = None

    def surveyed_volume(chosen: dict) -> dict:
        """Fetch, decode, verify and census one listed volume."""

        nonlocal antenna
        stamp = parse_iso(chosen["valid_time"])
        run_fetch(binary, site=site,
                  start=iso(stamp - timedelta(seconds=30)),
                  end=iso(stamp + timedelta(seconds=30)),
                  out=work_dir, bucket=bucket)
        volume_path = work_dir / chosen["filename"]
        if not volume_path.is_file():
            raise FrontDoorError(
                f"survey fetch did not materialise {volume_path}")
        pack_path = work_dir / (chosen["filename"] + ".survey.pack")
        run_decode(binary, volume=volume_path, out=pack_path,
                   moments=("REF",), max_range_km=authority_km,
                   max_elevation_deg=20.0)
        verify = run_verify(binary, pack=pack_path)
        if verify.get("status") != "PASS":
            raise FrontDoorError(f"survey pack verify did not PASS: "
                                 f"{verify}")
        volume = read_sweep_pack(pack_path)
        antenna = {"lat_deg": float(volume.site.lat_deg),
                   "lon_deg": float(volume.site.lon_deg),
                   "alt_m": float(volume.site.alt_m)}
        stats = echo_stats(volume, threshold_dbz=echo_threshold_dbz)
        end_time = getattr(volume, "end_time", None)
        if end_time is None:
            raise FrontDoorError(
                f"{site}: the decoded pack for {chosen['filename']} "
                "carries no end_time, so the survey cannot tell whether "
                f"the volume was complete by {iso(at)} and would admit it "
                "on its start alone, which is up to ten minutes before it "
                "existed. The rw_nexrad this run found predates the "
                "collection instants; rebuild it from this tree "
                "(cargo build --release -p rw-nexrad under tools/rustwx)")
        return {
            "volume": chosen["filename"],
            "key": chosen["key"],
            # The archive key's own stamp: the volume's start as the
            # archive named it.
            "valid_time": chosen["valid_time"],
            # The radials' own clocks, and when the archive published it.
            "start_time": getattr(volume, "start_time", None),
            "end_time": end_time,
            "complete": getattr(volume, "complete", None),
            "availability_time": chosen.get("last_modified") or None,
            "volume_sha256": sha256_file(volume_path),
            **stats,
        }

    # Admission: a volume is the newest usable one when its LAST radial was
    # collected at or before the moment the survey is about.  The listing
    # names volumes by their start, and a volume takes four to ten minutes,
    # so the newest key in a window can name a volume that was still being
    # scanned when the window ended.  Walking back from the newest key costs
    # one extra decode in that case and admits nothing before it existed.
    refused_incomplete = []
    newest_entry = None
    for candidate in reversed(volumes):
        entry = surveyed_volume(candidate)
        if parse_iso(entry["end_time"]) <= at:
            newest_entry = entry
            break
        refused_incomplete.append({
            "volume": entry["volume"],
            "start_time": entry["start_time"],
            "end_time": entry["end_time"],
            "reason": (f"its last radial was collected at "
                       f"{entry['end_time']}, after {iso(at)} "
                       f"({clock.subject})"),
        })
        # At most one listed volume can straddle the clock (the next one
        # starts after it), so three refusals in a row is a listing whose
        # keys do not describe its volumes, and is said rather than walked.
        if len(refused_incomplete) >= 3:
            break
    if newest_entry is None:
        names = ", ".join(r["volume"] for r in refused_incomplete)
        raise FrontDoorError(
            f"{site}: no listed volume was complete by {iso(at)} "
            f"({clock.subject}); refused {names}. The listing names "
            "volumes by their start and every one examined ends after "
            "that moment, which a real archive cannot do three times in a "
            "row -- check the bucket and the site")
    newest = newest_entry
    lag = (at - parse_iso(newest["end_time"])).total_seconds()
    if lag > max_lag_seconds and not allow_stale:
        behind = ("old" if clock.mode == "wall"
                  else f"older than the window end {iso(at)}")
        remedy = ("nowcast from a stale feed" if clock.mode == "wall"
                  else "replay a window the archive covers this thinly")
        raise FrontDoorError(
            f"{site}: newest complete archived volume {newest['volume']} "
            f"was complete at {newest['end_time']}, {lag / 60:.1f} min "
            f"{behind} (ceiling {max_lag_seconds / 60:.0f} min); pass "
            f"--allow-stale to {remedy} anyway")

    # The motion baseline is between scans: aim it at the newest volume's
    # own start and pick the listed volume whose key is nearest, which is
    # the archive's spelling of that same start.
    target = parse_iso(newest["start_time"] or newest["valid_time"]
                       ) - timedelta(seconds=motion_baseline_s)
    older = min(volumes, key=lambda v: abs(
        (parse_iso(v["valid_time"]) - target).total_seconds()))
    surveyed = [newest]
    if older["key"] != newest["key"]:
        surveyed = [surveyed_volume(older), newest]

    motion = None
    if len(surveyed) == 2:
        motion = motion_from_centroids(
            surveyed[0], surveyed[1], min_gates=min_echo_gates)

    return {
        "schema": SURVEY_SCHEMA,
        "site": site,
        # When the survey RAN, unchanged.  Which moment it asked the
        # archive about is the block below, and for a live run they are
        # the same time.
        "surveyed_at": iso(clock.wall),
        "clock": clock.to_payload(),
        "listing_window": {"start": iso(listing_start), "end": iso(at)},
        "volumes_listed": len(volumes),
        "newest_volume": newest["volume"],
        "newest_volume_start_time": newest["start_time"],
        "newest_volume_end_time": newest["end_time"],
        "newest_volume_availability_time": newest["availability_time"],
        "archive_lag_seconds": round(lag, 1),
        "archive_lag_definition": (
            "seconds from the newest admitted volume's last radial "
            "collection instant (its end_time) to the survey clock; the "
            "key name's time is the volume's start and is not the age of "
            "its data"),
        "admission": {
            "rule": ("a volume is admitted when its last radial was "
                     "collected at or before the survey clock; the listing "
                     "names volumes by their start"),
            "clock": iso(at),
            "refused_incomplete_at_clock": refused_incomplete,
        },
        "range_authority_km": authority_km,
        "echo_threshold_dbz": echo_threshold_dbz,
        "antenna": antenna,
        "survey_volumes": surveyed,
        "motion": motion,
        "motion_note": (None if motion is not None else
                        "no displacement motion: echo below "
                        f"{min_echo_gates} gates or baseline too short; "
                        "domain centers on the echo (or antenna) with "
                        "no downstream bias"),
    }


# ---------------------------------------------------------------------------
# the two entry points
# ---------------------------------------------------------------------------
class _StateDealias(argparse.Action):
    """``--dealias`` sets the flag AND records that a person stated it.

    The value alone cannot answer "did this command line ask for
    dealiasing?" now that the answer is true by default, and
    :func:`resolve_da_preset` has to refuse `--dealias --without dealias`
    rather than silently drop the thing the reader asked for.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, "dealias_stated", True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_nowcast",
        description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser(
        "run", help="survey a site and run the whole nowcast pipeline")
    run.add_argument("--site", required=True, type=validate_site,
                     help="four-letter radar station id (argument, "
                          "never a default). The ANCHOR: it sites the "
                          "domain, names the case's files, and always "
                          "contributes observations")
    run.add_argument("--sites", action="append", default=[],
                     type=validate_site, metavar="SITE",
                     help="an ADDITIONAL radar contributing to every "
                          "observation of this run, repeatable. The "
                          "anchor is always included, so --sites names "
                          "what to add to it. Two radars whose coverage "
                          "overlaps measure two projections of the same "
                          "wind and constrain what neither can alone. "
                          "Mutually exclusive with --discover-sites")
    run.add_argument("--discover-sites", action="store_true",
                     help="find the contributing radars from the "
                          "domain's own georeference and the vendored "
                          "NEXRAD site table, instead of naming them. "
                          "The anchor is not named on this route (the "
                          "obs builder refuses to mix naming and "
                          "discovery) but sits at the domain centre, so "
                          "discovery finds it with the best coverage of "
                          "any site")
    run.add_argument("--min-coverage-fraction", type=float, default=None,
                     help="discovery floor: skip a radar reaching less "
                          "than this fraction of the domain (obs "
                          "builder's default when unset). Raising it "
                          "buys density, lowering it buys the edges")
    run.add_argument("--max-radars", type=int, default=None,
                     help="keep at most this many discovered radars, "
                          "best coverage first")
    run.add_argument("--min-radars", type=int, default=None,
                     help="refuse an observation built from fewer than "
                          "this many radars. Applies to both routes. "
                          "The obs builder's default of 1 lets a "
                          "multi-radar request degrade to whatever "
                          "answered; raising it makes the count a hard "
                          "requirement of every cycle AND of every "
                          "verification frame")
    run.add_argument("--max-radar-time-spread-seconds", type=float,
                     default=None,
                     help="refuse if the contributing volumes' own "
                          "valid times span more than this. Radars are "
                          "never synchronous; this is where the caller "
                          "says how much asynchrony is one atmosphere")
    run.add_argument("--window-end", required=True,
                     help="last assimilated cycle valid time "
                          "(ISO-8601 UTC) or 'latest' to derive it "
                          "from the newest archived volume")
    run.add_argument("--cycles", type=int, default=6,
                     help="applied assimilation cycles (default 6)")
    run.add_argument("--cycle-seconds", type=int, default=900,
                     help="cycle cadence in seconds (default 900)")
    run.add_argument("--free-legs", type=int, default=6,
                     help="free forecast legs past the last obs "
                          "(default 6)")
    run.add_argument("--members", type=int, default=DEFAULT_MEMBERS,
                     help=f"ensemble size (default {DEFAULT_MEMBERS}; "
                          "demo-grade). Measured 2026-08-05 on two "
                          "cards with archived WSM6/Dudhia, longwave off; "
                          "current MP28/GSD4.1 requires remeasurement. "
                          "On those archived "
                          "cards: N=20 scores +0.0018 (analysis-only) "
                          "for +82%% wall clock, inside a 0.0062-0.0074 "
                          "across-member scatter, and N=36 scores BELOW "
                          "N=10 because the metric is computed on the "
                          "ensemble mean and averaging penalises an "
                          "under-producing forecast. N=4 costs 0.007 "
                          "FSS and saves no VRAM")
    run.add_argument("--out", type=Path, required=True,
                     help="case directory (created; holds every stage)")
    run.add_argument("--source", default="hrrr",
                     choices=("hrrr", "gfs"),
                     help="background source. Default hrrr -- the "
                          "convection-allowing background is the "
                          "nowcast's background, permanently (project "
                          "ruling, 2026-08-06); gfs is retained for "
                          "archival reproduction of pre-HRRR runs "
                          "only. Both names are woof.da.background "
                          "registry entries, and the NEXT source "
                          "(RRFS, when HRRR retires behind it) must "
                          "be a registry entry there, not another "
                          "branch here")
    run.add_argument("--physics-profile",
                     default=NOWCAST_DEFAULT_PHYSICS_PROFILE,
                     type=current_physics_profile,
                     choices=offered_physics_profiles(),
                     help="shipped physics profile for every stage "
                          f"(default {NOWCAST_DEFAULT_PHYSICS_PROFILE}, "
                          "which runs BOTH radiation streams; see "
                          "NOWCAST_DEFAULT_PHYSICS_PROFILE for why the "
                          "lw-off default was retired).  The list is the "
                          "one the configuration door this run calls "
                          "first accepts, so every name offered here is "
                          "a name every stage takes")
    run.add_argument("--hydrometeors", action="store_true",
                     help="perturb AND analyse the scheme's moisture and "
                          "hydrometeor state instead of u/v alone "
                          "(forwarded to the cycle driver; requires "
                          "--positivity-policy)")
    run.add_argument("--positivity-policy", default=None,
                     choices=("mean-preserving", "clip", "reject", "none"),
                     help="with --hydrometeors; unstated, the cycle "
                          "driver's default (mean-preserving) applies. The "
                          "four are not equivalent and woof.da.positivity "
                          "documents what each costs")
    run.add_argument("--reflectivity-analysis", action="store_true",
                     help="assimilate the merged reflectivity batch beside "
                          "the velocity batches; requires --hydrometeors, "
                          "since reflectivity against a wind-only state "
                          "vector analyses nothing")
    run.add_argument("--clear-air-analysis", action="store_true",
                     help="assimilate clear-air 'zero' observations: cells "
                          "the radar measured and found free of significant "
                          "echo. Suppresses spurious convection. Requires "
                          "--hydrometeors")
    run.add_argument("--surface-obs", type=Path, default=None,
                     help="a gpuwm-obs.asos-surface record (v2, or v1), assimilated as "
                          "2 m temperature and 10 m wind speed at k=0 "
                          "beside the radar batches. Build it with rw_asos "
                          "decode. A quantity is enabled by STATING ITS "
                          "SIGMA below; this flag alone assimilates nothing")
    run.add_argument("--sfc-t2-sigma-k", type=float, default=None,
                     help="assimilate 2 m temperature with this error "
                          "stddev (K). Representativeness, not instrument "
                          "precision -- WoFS-like practice is 1.5-2.5 K at "
                          "storm-scale grids")
    run.add_argument("--sfc-wspd-sigma-ms", type=float, default=None,
                     help="assimilate 10 m wind SPEED (the v1 seam carries "
                          "no direction) with this error stddev (m/s); H is "
                          "hypot(u10, v10) of the member diagnostics")
    run.add_argument("--sfc-max-age-s", type=float, default=None,
                     help="refuse surface reports older or newer than this "
                          "at the analysis (driver default 900). ASOS is "
                          "HOURLY by decoder design and each report enters "
                          "exactly one analysis, so on a sub-hourly cycle "
                          "most cycles legitimately see none; widen this to "
                          "let a cycle reach the nearest hourly report")
    run.add_argument("--goes-cwp", type=Path, action="append", default=[],
                     help="one gpuwm-obs.goes-grid.v1 file per cycle, in "
                          "cycle order, assimilated as a cloud-water-path "
                          "batch. Cycles past the end of this list "
                          "assimilate radar only. Requires --hydrometeors "
                          "and --cwp-vertical-loc-m")
    run.add_argument("--cwp-vertical-loc-m", type=float, default=None,
                     help="vertical localisation for the CWP batch, in "
                          "metres. REQUIRED with --goes-cwp and "
                          "deliberately without a default: CWP is a column "
                          "integral carried at one level, so this radius "
                          "decides whether the observation acts on the "
                          "column it integrated or on a slab")
    run.add_argument("--no-dealias", dest="dealias",
                     action="store_false", default=True,
                     help="mask every gate that might be folded instead of "
                          "unfolding radial velocity per sweep. Unfolding "
                          "is the default because the masks find "
                          "signatures of aliasing rather than aliasing, and "
                          "it recovers the gates above 0.8 * Nyquist that "
                          "carry a mesocyclone's couplet; gates the "
                          "unfolder cannot resolve are still dropped and "
                          "counted. Applied to the assimilated obs AND to "
                          "the verification composites, so the run is "
                          "graded against a truth field built the same "
                          "way it was fed, and the receipt records which "
                          "way that was. The default engine is "
                          "`region-global`, the Rust crate `woof "
                          "fetch-bridges` stages; `--dealias-engine "
                          "vad-region` selects the scipy one instead")
    # The old spelling of the default, still accepted.  It does not enable
    # anything (unfolding is on for every run), it STATES the default, so
    # an existing invocation that asks for it keeps running instead of
    # dying on `unrecognized arguments`.  Registered after --no-dealias so
    # the shared dest keeps that action's default; both write `dealias`,
    # so whichever spelling comes last on the command line wins.
    run.add_argument("--dealias", dest="dealias", action=_StateDealias,
                     nargs=0, default=True,
                     help="accepted and redundant: radial velocity is "
                          "unfolded on every run already, so this names "
                          "the default rather than turning it on. "
                          "--no-dealias is the off switch")
    # The same two options tools.obs_radar_grid_build takes, defined by
    # that tool and added here rather than restated: this front door
    # BUILDS that tool's argv, so a second copy of the flag names is a
    # second thing to keep in step.
    from tools.obs_radar_grid_build import add_dealias_engine_arguments
    add_dealias_engine_arguments(run)
    run.add_argument("--da", choices=("full", "vr", "custom"),
                     default="custom",
                     help="observation-stream preset. 'full' is the "
                          "certified full-stack configuration -- radial "
                          "velocity plus reflectivity, clear air, "
                          "dealiasing, surface and GOES CWP "
                          "(evidence/da-demo/full-stack) -- subtract "
                          "streams with --without. 'vr' is radial "
                          "velocity alone. 'custom' (the default) means "
                          "the individual flags are the whole story, "
                          "exactly as before this flag existed")
    run.add_argument("--without", action="append", default=[],
                     choices=DA_SUBTRACTABLE, metavar="STREAM",
                     help="with --da full: drop one stream from the "
                          "preset (repeatable). One of: "
                          + ", ".join(DA_SUBTRACTABLE))
    run.add_argument("--dx-km", type=float, default=3.0,
                     help="grid spacing in km (default 3)")
    run.add_argument("--vram-gib", type=float, default=None,
                     help="size the memory preflight against this card "
                          "instead of the wizard's default tier. A "
                          "caller who was shown a verdict for one card "
                          "has to be able to ask for the run that "
                          "verdict was about")
    run.add_argument("--profile", choices=tuple(CARD_PROFILES),
                     default="auto",
                     help="a named card profile: a configuration that "
                          "was RUN on the card it names and completed. "
                          "'card-16gib' is the shipped demo shape, "
                          "measured end to end on a 16,376 MiB RTX "
                          "4080 -- it returns the 32 GB card's answer "
                          "(FSS 0.7397 vs 0.7403 over six leads) and "
                          "peaks at 15,888 MiB, 97.0%% of the card. "
                          "These measurements used WSM6/Dudhia with "
                          "longwave off; they do not certify current "
                          "MP28/GSD4.1 card fit. A "
                          "profile only fills in arguments the caller "
                          "left at their defaults")
    run.add_argument("--memory-budget-mib", type=float,
                     default=DEFAULT_MEMORY_BUDGET_MIB,
                     help=f"LETKF chunk workspace in MiB (default "
                          f"{DEFAULT_MEMORY_BUDGET_MIB:g}). The one "
                          "term in the memory model an operator "
                          "controls: peak VRAM does NOT scale with "
                          "--members (15,946/15,888/15,796 MiB at "
                          "N=4/10/20), it tracks this. At the default "
                          "a 16 GB card peaks at 97.0%% with 488 MiB "
                          "spare, so lower it if you want margin -- "
                          "the ladder that would pick a better default "
                          "had not finished when this was set. Those "
                          "measurements used archived WSM6/Dudhia with "
                          "longwave off; current MP28/GSD4.1 card fit "
                          "requires remeasurement")
    run.add_argument("--history-interval-seconds", type=float,
                     default=None,
                     help="how often the georeference forecast writes a "
                          "wrfout (default: --cycle-seconds). This is "
                          "the lattice observations are gridded onto, "
                          "so a coarse one makes every observation's "
                          "georeference stale by up to half of it")
    run.add_argument("--run-hours", type=int, default=None,
                     help="georeference run length in whole hours "
                          "(default: the shortest that covers the "
                          "window). A longer one buys a longer boundary "
                          "window for a caller that keeps cycling on "
                          "this prepared case")
    run.add_argument("--domain-polygon", type=Path, default=None,
                     help="a GeoJSON polygon to use as the domain "
                          "instead of siting one on the echo -- the box "
                          "a caller drew. The survey still runs and is "
                          "still receipted; only the siting is taken "
                          "out of its hands")
    run.add_argument("--box-half-km", type=float, default=198.0,
                     help="half-extent of the domain box in km "
                          "(default 198, sized to the obs range "
                          "authority)")
    run.add_argument("--downstream-fraction", type=float, default=0.35,
                     help="fraction of (storm motion x horizon) added "
                          "downstream of the echo centroid")
    run.add_argument("--max-center-offset-km", type=float, default=60.0,
                     help="clamp on domain-center offset from the "
                          "antenna, keeping the radar in the grid")
    run.add_argument("--echo-threshold-dbz", type=float, default=35.0)
    run.add_argument("--min-echo-gates", type=int, default=500,
                     help="below this the survey refuses to call a "
                          "motion vector")
    run.add_argument("--motion-baseline-seconds", type=float,
                     default=2700.0,
                     help="how far back the second survey volume sits")
    # -- the fine nest over the free forecast ----------------------------
    #
    # The driver's own flags, spelled the driver's way and forwarded
    # unchanged (NestChoice).  An EXTENT turns the nest on and nothing
    # else does, which is how `tools/da_cycle_prepared.py` is asked, so
    # a caller who has read one page has read both.  The parent carries
    # the ensemble and the assimilation; the nest is the detailed view
    # of the forecast that runs past the last observation.
    nest_group = run.add_argument_group(
        "the fine nest over the free forecast",
        "off unless an extent is named; see docs/da-nested-forecast.md")
    nest_group.add_argument(
        "--nest-half-width-km", type=float, default=None,
        help=("half-width of the nest in kilometres, centred in the "
              "parent.  Naming it is what turns the nest on"))
    nest_group.add_argument("--nest-nx", type=int, default=None,
                            help="child extent in CHILD cells "
                                 "(with --nest-ny); also turns it on")
    nest_group.add_argument("--nest-ny", type=int, default=None,
                            help="child extent in CHILD cells "
                                 "(with --nest-nx)")
    nest_group.add_argument(
        "--nest-ratio", type=int, default=None,
        help=("parent-to-child refinement ratio, applied to BOTH space "
              "and time (default: the nest geometry's own, 3 -- which "
              "off a 3 km parent is a 1 km nest)"))
    nest_group.add_argument("--nest-i-parent-start", type=int,
                            default=None,
                            help="1-based parent cell of the child's "
                                 "origin (default: centred)")
    nest_group.add_argument("--nest-j-parent-start", type=int,
                            default=None)
    nest_group.add_argument(
        "--nest-members", type=int, default=None,
        help=("how many ensemble members carry a nest, beside the "
              "control (default: the control only).  Nest cost scales "
              "as ratio^3 per covered parent cell TIMES this number"))
    nest_group.add_argument("--nest-history-interval-s", type=float,
                            default=None,
                            help="child history cadence (default: the "
                                 "parent's)")
    nest_group.add_argument(
        "--nest-acknowledge", action="append", default=[],
        help=("acknowledge a nested-domain admissibility refusal by id, "
              "e.g. nested-forecast:sub-gray-zone-pbl"))
    run.add_argument("--survey-span-seconds", type=float, default=5400.0)
    run.add_argument("--max-lag-seconds", type=float, default=900.0,
                     help="freshness ceiling on the archive feed")
    run.add_argument("--allow-stale", action="store_true")
    run.add_argument("--bucket", default=None,
                     help="S3 bucket override (default: the rw_nexrad "
                          "front door's own)")
    run.add_argument("--range-km", type=float, default=None,
                     help="range authority for survey and obs alike")
    run.add_argument("--bridge", type=Path, default=None,
                     help="gfs_grib2_bridge executable (default: the "
                          "artifact ladder)")
    run.add_argument("--geog-root", type=Path, default=None,
                     help="WPS_GEOG static geography root")
    run.add_argument("--solve-device", default="cuda",
                     choices=("cuda", "host"))
    run.add_argument("--horizontal-loc-m", type=float, default=12000.0)
    run.add_argument("--vertical-loc-m", type=float, default=3000.0)
    run.add_argument("--length-scale-km", type=float, default=None,
                     help="perturbation length scale in km (default: "
                          f"{DEFAULT_LENGTH_SCALE_KM:g}, capped to what "
                          "the fitted domain carries -- a smaller box "
                          "cannot represent a larger scale, and the cap "
                          "is receipted)")
    run.add_argument("--seed", type=int, default=None,
                     help="perturbation seed (default: derived from "
                          "the window end date)")
    run.add_argument("--stop-after", choices=STAGES, default=None,
                     help="end the run after this stage (receipts "
                          "still written)")
    run.add_argument("--no-verify", action="store_true",
                     help="do NOT hand off to the rolling verifier "
                          "when the run completes (verification is on "
                          "by default)")
    run.add_argument("--verify-poll-seconds", type=int,
                     default=WATCH_POLL_SECONDS,
                     help=f"verifier archive poll interval (default "
                          f"{WATCH_POLL_SECONDS})")
    run.add_argument("--verify-max-minutes", type=int,
                     default=WATCH_MAX_MINUTES,
                     help="verifier safety ceiling in minutes "
                          f"(default {WATCH_MAX_MINUTES})")
    run.add_argument("--verify-max-offset-seconds", type=float,
                     default=480.0,
                     help="how far a volume may sit from a frame's "
                          "valid time and still verify it")

    verify = sub.add_parser(
        "verify", help="one pass: build observed composites for the "
                       "free-forecast valid times the archive covers "
                       "now, re-render, and record the verdict")
    verify.add_argument("--case-dir", type=Path, required=True)
    verify.add_argument("--bucket", default=None)
    verify.add_argument("--max-offset-seconds", type=float, default=480.0)

    watch = sub.add_parser(
        "watch", help="roll the verification: keep grading frames as "
                      "the archive covers their valid times, updating "
                      "the gallery and receipt in place, then stop")
    watch.add_argument("--case-dir", type=Path, required=True)
    watch.add_argument("--bucket", default=None)
    watch.add_argument("--max-offset-seconds", type=float, default=480.0)
    watch.add_argument("--poll-seconds", type=int,
                       default=WATCH_POLL_SECONDS)
    watch.add_argument("--max-minutes", type=int,
                       default=WATCH_MAX_MINUTES,
                       help="safety ceiling: stop waiting after this "
                            "long and record what stayed unverified")
    return parser


def _stop(requested: str | None, stage: str) -> bool:
    return (requested is not None
            and STAGES.index(stage) >= STAGES.index(requested))


def run_reaches_render(args) -> bool:
    """Will this invocation execute the render stage?

    Derived from :data:`STAGES` and the same ``_stop`` predicate the run
    itself uses, never from a hand-written list of the ``--stop-after``
    values that happen to stop early today.  A front-door check that
    refused a run which was never going to draw anything would be a
    refusal for a capability the run does not need -- and one spelled a
    second way from the loop it is guarding drifts the first time a
    stage is inserted.

    The predicate is asked of the stages BEFORE render, not of render
    itself, because ``_stop(requested, stage)`` means "this run stops
    having completed ``stage``" and the run checks it *after* running
    each one.  ``--stop-after render`` therefore RUNS the renderer, and
    the first spelling of this function -- ``not _stop(stop_after,
    "render")`` -- read that exactly backwards and would have waved
    through the one ``--stop-after`` value that still needs the map
    reader.  Its own test caught it; the shape below cannot make that
    mistake, because it asks the same question the loop asks.
    """

    requested = getattr(args, "stop_after", None)
    before_render = STAGES[:STAGES.index("render")]
    return not any(_stop(requested, stage) for stage in before_render)


def _stopped_early(out: Path, receipts: Path, args, stage: str) -> int:
    """Leave the partial receipt behind, then report the stop.

    One place, so a door added later cannot forget it -- the whole
    defect was that seven `return 0`s each individually skipped the
    line that writes this file.
    """

    written = write_partial_receipt(
        out, receipts=receipts, dealias=DealiasChoice.from_args(args),
        stopped_after=stage)
    print(f"stopped after {stage} (receipts written)")
    if written is not None:
        print(f"  partial receipt: {written}")
        print("  `da_nowcast verify --case-dir "
              f"{out}` can read this case as it stands")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "run":
        args.profile_applied = apply_card_profile(args)
        return run_pipeline(args)
    if args.mode == "watch":
        return watch_pipeline(args)
    return verify_pipeline(args)


def resolve_da_preset(args) -> None:
    """--da full/vr become concrete stream flags, before validation.

    'full' reproduces the certified full-stack configuration
    (evidence/da-demo/full-stack/run.sh): reflectivity, clear air,
    dealiasing, surface and GOES CWP beside the velocity batches, with
    the pinned positivity policy and CWP localisation.  --without
    subtracts a stream.  Science values the evidence bundle does not pin
    (the surface sigmas) stay the caller's to state.  'vr' is the
    velocity-only base configuration.  'custom' changes nothing.
    """
    without = set(getattr(args, "without", None) or ())
    if args.da == "custom":
        if without:
            raise FrontDoorError(
                "--without subtracts from a preset, and --da custom has "
                "none: every stream is already opt-in, so pass the flags "
                "you want instead")
        return
    if args.da == "vr":
        # Dealiasing is not on this list.  It is not a stream, it is how
        # the velocity stream is prepared, and it is on for every run now:
        # "radial velocity alone" must not quietly mean "radial velocity
        # with a coherent fold left in it".  --no-dealias still turns it
        # off beside any preset.
        stated = [flag for flag, on in (
            ("--reflectivity-analysis", args.reflectivity_analysis),
            ("--clear-air-analysis", args.clear_air_analysis),
            ("--hydrometeors", args.hydrometeors),
            ("--surface-obs", args.surface_obs is not None),
            ("--goes-cwp", bool(args.goes_cwp)),
        ) if on]
        stated += [f"--without {w}" for w in sorted(without)]
        if stated:
            raise FrontDoorError(
                "--da vr is radial velocity alone and contradicts "
                + ", ".join(stated)
                + ". Say --da full --without ... for full-minus-a-stream, "
                  "or --da custom to state flags individually")
        return
    # --da full
    # "dealias" being TRUE is not a statement anybody made -- it is the
    # default -- so only an explicit --dealias contradicts
    # `--without dealias`, which is what dealias_stated records.
    contradictions = sorted(
        w for w in without
        if (w == "reflectivity" and args.reflectivity_analysis)
        or (w == "clear-air" and args.clear_air_analysis)
        or (w == "dealias" and getattr(args, "dealias_stated", False))
        or (w == "surface" and args.surface_obs is not None)
        or (w == "cwp" and bool(args.goes_cwp)))
    if contradictions:
        raise FrontDoorError(
            "--without " + ", ".join(contradictions) + " subtracts a "
            "stream this command line also explicitly enables; state one "
            "intention")
    args.hydrometeors = True
    if args.positivity_policy is None:
        args.positivity_policy = "mean-preserving"
    if "reflectivity" not in without:
        args.reflectivity_analysis = True
    if "clear-air" not in without:
        args.clear_air_analysis = True
    args.dealias = args.dealias and "dealias" not in without
    if "surface" not in without and args.surface_obs is None:
        raise FrontDoorError(
            "--da full includes the surface stream: pass --surface-obs "
            "<gpuwm-obs.asos-surface record, v2 or v1> (build it with rw_asos decode) "
            "plus at least one of --sfc-t2-sigma-k / --sfc-wspd-sigma-ms "
            "(WoFS-like practice is 1.5-2.5 K at storm-scale grids), or "
            "drop the stream with --without surface")
    if "cwp" not in without:
        if not args.goes_cwp:
            raise FrontDoorError(
                "--da full includes the GOES cloud-water-path stream: "
                "pass one --goes-cwp <gpuwm-obs.goes-grid.v1> per cycle "
                "(tools/obs_goes_grid_build.py builds them), or drop the "
                "stream with --without cwp")
        if args.cwp_vertical_loc_m is None:
            args.cwp_vertical_loc_m = DA_FULL_CWP_VLOC_M
    streams = ["vr"] + [s for s in DA_SUBTRACTABLE if s not in without]
    print(f"da preset: full -> streams {', '.join(streams)}"
          + (f" (minus {', '.join(sorted(without))})" if without else ""))


def validate_analysis_flags(args) -> None:
    """The driver's own refusal chain, met at the front door instead.

    tools.da_cycle_prepared refuses these combinations itself, but only
    at the cycle stage -- after the fetch, the preparation and the
    georeference forecast have been paid for.  The same two sentences
    here cost a second.
    """

    if getattr(args, "reflectivity_analysis", False) \
            and not getattr(args, "hydrometeors", False):
        raise FrontDoorError(
            "--reflectivity-analysis needs --hydrometeors: reflectivity "
            "constrains condensate, and against a u/v state vector every "
            "dBZ increment would be sampling noise")
    if getattr(args, "hydrometeors", False) \
            and getattr(args, "positivity_policy", None) is None:
        raise FrontDoorError(
            "--hydrometeors analyses physically non-negative fields and "
            "--positivity-policy is unstated; clip / reject / none are "
            "not equivalent (woof.da.positivity documents the costs)")
    if getattr(args, "dealias", True):
        # The obs stage answers this too, but only once per cycle and
        # only after that cycle's volumes have been fetched.  A run that
        # is going to be unable to dealias should learn it now.  Which
        # prerequisite is missing depends on the engine -- scipy for one,
        # a built shared library for the other -- so the question is asked
        # of the engine rather than of scipy.
        #
        # A NAMED engine that cannot run is a refusal: the operator asked
        # for one solver and would silently get another.  The unnamed
        # default is not, because dealiasing is on for every run now and a
        # missing optional library would otherwise be a dead front door.
        # It resolves down the stated chain instead, warned once.
        from woof.obs.dealias import (DEFAULT_ENGINE_CHAIN,
                                       engine_unavailable_reason)
        named = getattr(args, "dealias_engine", None)
        refinement = getattr(args, "dealias_refinement", None)
        engine = DEFAULT_ENGINE_CHAIN[0] if named is None else str(named)
        if refinement and engine != DEFAULT_DEALIAS_ENGINE:
            raise FrontDoorError(
                f"--dealias-refinement with --dealias-engine {engine}: that "
                f"engine has no refinement pass; only "
                f"{DEFAULT_DEALIAS_ENGINE} does")
        reason = engine_unavailable_reason(engine)
        if reason is not None and (named is not None or refinement):
            raise FrontDoorError(f"--dealias-engine {engine}: {reason}")
    elif getattr(args, "dealias_refinement", None) is not None:
        raise FrontDoorError(
            "--dealias-refinement/--no-dealias-refinement refines a "
            "dealiased field; drop --no-dealias, or drop it")
    if run_reaches_render(args):
        # THE worst failure position in the product, met at the front
        # door.  `render` is the LAST stage -- after the survey, the
        # fetch, the preparation, the free forecast and every DA cycle
        # -- and `tools/da_nowcast_render.py` opened its basemap with a
        # bare `import shapefile`.  A six-hour run therefore ended in
        # `ModuleNotFoundError: No module named 'shapefile'` with no
        # message at all: maximum work destroyed, minimum information
        # given, and nothing in `woof doctor` to warn of it.
        #
        # Same shape as the dealiasing-engine check above and for the
        # same reason: the prerequisite is knowable in the second before
        # the run starts, and a capability check belongs wherever it is
        # cheapest to answer, not wherever it happens to be consumed.
        from woof.rustwx import PYSHP_REMEDY, pyshp_available
        if not pyshp_available():
            raise FrontDoorError(
                "this run ends in the render stage, which draws the map "
                f"frame: {PYSHP_REMEDY}.\n"
                "  # Or stop before it and keep every other receipt:\n"
                "  #   --stop-after cycle")
    # The non-radar streams. Same reasoning as the driver's own refusals,
    # met here so a six-hour run does not pay for the fetch and the
    # georeference forecast before learning it was underspecified.
    if getattr(args, "clear_air_analysis", False) \
            and not getattr(args, "hydrometeors", False):
        raise FrontDoorError(
            "--clear-air-analysis needs --hydrometeors: a clear-air zero "
            "says condensate is absent, and against a u/v state vector "
            "there is no condensate for it to act on")
    if getattr(args, "goes_cwp", None):
        if not getattr(args, "hydrometeors", False):
            raise FrontDoorError(
                "--goes-cwp needs --hydrometeors: cloud water path IS the "
                "column condensate, and against a u/v state vector every "
                "CWP increment would be wind-condensate sampling "
                "covariance, which at storm scale is noise")
        if getattr(args, "cwp_vertical_loc_m", None) is None:
            raise FrontDoorError(
                "--goes-cwp needs an explicit --cwp-vertical-loc-m. CWP is "
                "a column integral carried at one model level, so its "
                "vertical localisation radius is not a tuning detail: it "
                "decides whether the observation acts on the column it "
                "integrated. Inheriting the radar radius would silently "
                "assimilate a whole-column measurement as a 4 km-tall one")
    if getattr(args, "surface_obs", None) is not None \
            and getattr(args, "sfc_t2_sigma_k", None) is None \
            and getattr(args, "sfc_wspd_sigma_ms", None) is None:
        raise FrontDoorError(
            "--surface-obs was given with neither --sfc-t2-sigma-k nor "
            "--sfc-wspd-sigma-ms. A surface quantity is enabled by stating "
            "its error standard deviation, so this run would carry a "
            "surface file and assimilate nothing from it -- which is "
            "exactly the silent-stream shape woof.da.treatment exists to "
            "stop, caught here before the card is taken")
    for path, flag in ((getattr(args, "surface_obs", None), "--surface-obs"),
                       *((p, "--goes-cwp")
                         for p in (getattr(args, "goes_cwp", None) or ()))):
        if path is not None and not Path(path).is_file():
            raise FrontDoorError(
                f"{flag} names a missing file: {path}. An observation "
                "stream that cannot be read assimilates nothing, and a run "
                "that discovers this mid-cycle has already spent the card")
    # The nest's three CLI refusals, met here for the same reason as the
    # rest of this function: the driver raises them at the cycle stage,
    # which is after the fetch, the preparation and the georeference
    # forecast have been paid for.
    nest = NestChoice.from_args(args)
    free_legs = int(getattr(args, "free_legs", 0) or 0)
    if nest.requested and free_legs <= 0:
        raise FrontDoorError(
            "--nest-* needs --free-legs > 0: the nest runs over the FREE "
            "forecast legs, which is the whole point of it -- the parent "
            "carries the assimilation and the nest carries the detail of "
            "the forecast that runs past the observations")
    if nest.members is not None and not nest.requested:
        raise FrontDoorError(
            "--nest-members without a nest extent "
            "(--nest-half-width-km or --nest-nx/--nest-ny): there is no "
            "nest for those members to carry")
    if (nest.members is not None
            and nest.members > int(getattr(args, "members", 0) or 0)):
        raise FrontDoorError(
            f"--nest-members {nest.members} exceeds --members "
            f"{args.members}: the nest is a subset of the parent "
            "ensemble")


def run_pipeline(args) -> int:
    now = datetime.now(timezone.utc)
    resolve_da_preset(args)
    validate_analysis_flags(args)
    # ABSOLUTE, and absolute BEFORE the survey downloads a byte.
    #
    # Every stage below runs as a subprocess with ``cwd=repo_root``,
    # which is the directory holding the ``tools`` package -- in a wheel
    # install that is ``site-packages``, and it is never the directory
    # the user typed the command in.  A relative ``--out`` therefore
    # meant two different directories in one run: this process wrote
    # ``danow/case/domain-box.geojson`` under the CALLER's cwd, and then
    # `woof domain` looked for that same relative path under
    # ``repo_root`` and refused "local GeoJSON file does not exist" for a
    # file that was on disk.  Resolving here makes the child's argv
    # absolute, so the sentence a stage prints is about the path the
    # stage actually looked at.  ``woof go`` fixed the identical defect
    # in its own chain (``woof/go_cli.py:_stage_cwd``); this front door
    # never received it.
    #
    # It is done at INTAKE, ahead of ``survey_site`` below, because that
    # is the last moment the fix is free: one line later the survey has
    # started pulling NEXRAD volumes from S3, and a path contract that
    # only fails after the download has already cost the user the
    # bandwidth this ordering exists to protect.
    out: Path = Path(args.out).resolve()
    args.out = out
    receipts = out / "receipts"
    # Same reason, same moment: a --domain-polygon the caller drew is not
    # read until stage 1, but whether the file is THERE is knowable now,
    # and "you named a missing file" is worth nothing to a user who has
    # already paid for two Level-II volumes to hear it.
    if args.domain_polygon is not None:
        args.domain_polygon = Path(args.domain_polygon).resolve()
        if not args.domain_polygon.is_file():
            raise FrontDoorError(
                f"--domain-polygon names a missing file: "
                f"{args.domain_polygon}")
    # Resolved before the survey moves a byte: a contradictory radar
    # request should cost nothing to refuse.
    selection = radar_selection(args)
    # Same moment, same reason, and this one decides what the survey
    # asks the archive: a window end names the moment the run is about,
    # and only `latest` is a question about now.  Parsed here rather
    # than after the survey (where the window used to be resolved)
    # because the survey needs it, and because "that is not a time" is
    # worth more before two Level-II volumes have been paid for.
    clock = survey_clock(args.window_end, now=now)
    # The nest, resolved once here and forwarded to the cycle stage; its
    # own refusals were met in validate_analysis_flags above.
    nest = NestChoice.from_args(args)
    out.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    # ---- survey ---------------------------------------------------------
    survey = survey_site(
        args.site, work_dir=out / "vols", clock=clock,
        span_seconds=args.survey_span_seconds,
        motion_baseline_s=args.motion_baseline_seconds,
        echo_threshold_dbz=args.echo_threshold_dbz,
        min_echo_gates=args.min_echo_gates,
        max_lag_seconds=args.max_lag_seconds,
        allow_stale=args.allow_stale, bucket=args.bucket,
        range_km=args.range_km)
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "00-survey.json").write_text(
        json.dumps(survey, indent=1), encoding="utf-8")
    print(f"survey: clock {clock.mode} ({iso(clock.at)}), lag "
          f"{survey['archive_lag_seconds'] / 60:.1f} min, "
          f"echo gates {survey['survey_volumes'][-1]['gates']}, "
          f"motion {survey['motion']}")

    # ---- window plan ----------------------------------------------------
    if clock.window_end is None:
        # The window ends no later than the newest volume was COMPLETE:
        # its last analysis has to have a whole volume to assimilate.
        newest = parse_iso(survey["newest_volume_end_time"])
        window_end = resolve_latest_window_end(
            newest, cycles=args.cycles,
            cycle_seconds=args.cycle_seconds)
    else:
        window_end = clock.window_end
    plan = plan_window(window_end, cycles=args.cycles,
                       cycle_seconds=args.cycle_seconds,
                       free_legs=args.free_legs, now=now,
                       run_hours=args.run_hours, source=args.source)
    seed = (args.seed if args.seed is not None
            else int(plan.window_end.strftime("%Y%m%d")))

    # ---- domain ---------------------------------------------------------
    center = site_domain_center(
        survey["antenna"]["lat_deg"], survey["antenna"]["lon_deg"],
        survey["survey_volumes"][-1], survey["motion"],
        horizon_seconds=plan.horizon_seconds,
        downstream_fraction=args.downstream_fraction,
        max_offset_km=args.max_center_offset_km)
    case_name = (f"nowcast_{args.site.lower()}_"
                 f"{plan.window_end:%Y%m%d%H%M}")
    case_dir = out / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    polygon = case_dir / "domain-box.geojson"
    if args.domain_polygon is not None:
        # The caller drew the domain.  Their box is copied in verbatim
        # -- not re-derived, not re-centred -- so what the wizard fits
        # is exactly the geometry they were shown when they committed.
        # Existence was settled at intake, before the survey spent a
        # byte; this read is the only thing left to do here.
        box = json.loads(
            args.domain_polygon.read_text(encoding="utf-8"))
        center = dict(center)
        center["basis"] = ("caller-supplied polygon "
                           f"({args.domain_polygon.name}); the survey's "
                           "own siting was computed and not used")
    else:
        box = geojson_box(center["lat"], center["lon"], args.box_half_km)
    polygon.write_text(json.dumps(box), encoding="utf-8")
    case_toml = case_dir / f"{case_name}.toml"
    plan_receipt = {
        "schema": SCHEMA, "partial": True, "site": args.site,
        "radars": selection.to_payload(),
        "plan": plan.to_payload(), "domain_center": center,
        "seed": seed, "case_name": case_name}
    (receipts / "01-plan.json").write_text(
        json.dumps(plan_receipt, indent=1), encoding="utf-8")
    if _stop(args.stop_after, "survey"):
        return _stopped_early(out, receipts, args, "survey")

    run_stage("domain", wizard_cmd(
        polygon=polygon, out_toml=case_toml, plan=plan,
        profile=args.physics_profile, name=case_name,
        dx_km=args.dx_km, source=args.source,
        vram_gib=args.vram_gib),
        cwd=repo_root, receipts_dir=receipts, index=2)
    wps_namelist = case_toml.with_suffix("").with_name(
        case_toml.stem + ".namelist.wps")
    if not wps_namelist.is_file():
        raise FrontDoorError(
            f"wizard did not emit {wps_namelist.name} beside the TOML")
    namelist_input = domain_spec = None
    if args.source == "hrrr":
        # The HRRR preparation reads namelists and a target-domain
        # document, not the TOML; the wizard emits all of them beside
        # it (woof.hrrr_route_inputs.route_input_paths).
        namelist_input = case_toml.with_name(
            case_toml.stem + ".namelist.input")
        domain_spec = case_toml.with_name(
            case_toml.stem + ".d01-target.json")
        for route_input in (namelist_input, domain_spec):
            if not route_input.is_file():
                raise FrontDoorError(
                    f"wizard did not emit {route_input.name} beside the "
                    "TOML; the HRRR route cannot be prepared without it")

    # The georeference lattice.  Every observation is gridded onto the
    # wrfout nearest its valid time, so how often one is written IS the
    # accuracy of that georeference -- and it has to be set before the
    # authority is materialised, or the authority binds the wrong one.
    history_seconds = (args.history_interval_seconds
                       if args.history_interval_seconds is not None
                       else float(args.cycle_seconds))
    retimed, was = retime_history(
        case_toml.read_text(encoding="utf-8"), history_seconds)
    case_toml.write_text(retimed, encoding="utf-8")
    if namelist_input is not None:
        # The HRRR route reads its experiment from the namelist, not the
        # TOML, so the cadence has to land in BOTH or the prepared
        # bundle's hash-bound experiment carries the wizard's hourly
        # default and every downstream stage that states the run's own
        # cadence is refused against it.  Same key, same pattern, same
        # editor.
        renamelist, _ = retime_history(
            namelist_input.read_text(encoding="utf-8"), history_seconds)
        namelist_input.write_text(renamelist, encoding="utf-8")
    (receipts / "02-history-cadence.json").write_text(json.dumps({
        "schema": STAGE_SCHEMA, "stage": "history-cadence",
        "started": iso(datetime.now(timezone.utc)),
        "file": str(case_toml),
        "also_retimed": (None if namelist_input is None
                         else str(namelist_input)),
        "history_interval_s": {"wizard": was, "run": history_seconds},
        "why": ("observations are gridded onto the wrfout nearest their "
                "valid time; the wizard's root default is hourly, which "
                "would make 'nearest' up to half an hour stale"),
    }, indent=1), encoding="utf-8")
    print(f"georeference cadence: {was:g} s -> {history_seconds:g} s")
    if _stop(args.stop_after, "domain"):
        return _stopped_early(out, receipts, args, "domain")

    # ---- authority + fetch + manifest + prepare -------------------------
    authority_dir = out / "authority"
    run_stage("authority", authority_cmd(
        case_toml=case_toml, wps_namelist=wps_namelist,
        profile=args.physics_profile, authority_dir=authority_dir,
        source=args.source),
        cwd=repo_root, receipts_dir=receipts, index=3)

    import tomllib
    emitted = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    root = emitted["domain"][0]
    length_scale_km, scale_note = resolvable_length_scale_km(
        nx=int(root["nx"]), ny=int(root["ny"]),
        dx_m=float(root["dx"]), dy_m=float(root.get("dy", root["dx"])),
        requested=args.length_scale_km)
    if scale_note:
        print(f"note: {scale_note}")
    hints = emitted.get("fetch")
    if not hints:
        raise FrontDoorError("emitted case TOML carries no [fetch] "
                             "hints; cannot drive the fetch stage")
    data_dir = out / "data"
    run_stage("fetch", fetch_cmd(hints=hints, data_dir=data_dir,
                                 source=args.source),
              cwd=repo_root, receipts_dir=receipts, index=4)
    if args.source == "hrrr":
        # No manifest stage: `woof fetch --source hrrr` already wrote
        # the SHA256SUMS the HRRR front door consumes, and
        # --author-front-door-manifest refuses the source by design.
        # The receipt index 5 is deliberately left unused so every
        # stage keeps the same number on both routes.
        manifest_sha = sha256_file(data_dir / "SHA256SUMS")
    else:
        run_stage("manifest", manifest_cmd(
            data_dir=data_dir, authority_dir=authority_dir,
            source=args.source),
            cwd=repo_root, receipts_dir=receipts, index=5)
        manifest_sha = sha256_file(data_dir / "gfs-input-manifest.json")
    if _stop(args.stop_after, "fetch"):
        return _stopped_early(out, receipts, args, "fetch")

    prepared_root = out / "prepared"
    run_stage("prepare", prepare_cmd(
        data_dir=data_dir, authority_dir=authority_dir,
        # The gfs GRIB2 bridge is a GFS-route artifact; the HRRR route
        # resolves its own native decoder inside the preparation.  Not
        # resolving it here keeps a box that never built the gfs bridge
        # able to run the default route.
        bridge=(None if args.source == "hrrr"
                else resolve_bridge(args.bridge)),
        profile=args.physics_profile,
        geog_root=resolve_geog_root(args.geog_root),
        prepared_root=prepared_root, plan=plan,
        manifest_sha=manifest_sha, source=args.source,
        namelist_input=namelist_input, domain_spec=domain_spec,
        history_interval_seconds=history_seconds),
        cwd=repo_root, receipts_dir=receipts, index=6)
    if args.source == "hrrr":
        # The preparation published the portable bundle INTO the
        # prepared root: proof.json, source-input-manifest.json,
        # experiment.toml and namelist.wps all live there, and the
        # wrapper receipt carries the three digests downstream stages
        # must quote back.  From here on the bundle root IS the
        # authority directory -- its experiment config is the one the
        # proof hash-binds, not the pre-preparation authority's.
        wrapper = json.loads(
            (prepared_root / "public-wrapper-result.json")
            .read_text(encoding="utf-8"))
        pins = wrapper.get("portable_bundle")
        if not pins:
            raise FrontDoorError(
                "the HRRR preparation wrote no portable_bundle block; "
                "that is what a preparation run without --wps-namelist "
                "leaves behind, and this driver always passes it")
        proof_sha = pins["proof_sha256"]
        manifest_sha = pins["source_manifest_sha256"]
        content_sha = pins["prepared_content_sha256"]
        authority_dir = prepared_root
    else:
        proof_sha = sha256_file(prepared_root / "proof.json")
        content_sha = json.loads(
            (prepared_root / "proof.json").read_text(encoding="utf-8")
        )["prepared_cache"]["content_sha256"]
    # The three digests every downstream runner has to quote back.  They
    # are written as their own receipt rather than only into the final
    # one, so a run stopped early (--stop-after prepare/forecast) still
    # hands a later process everything it needs to drive the case.
    (receipts / "06-bindings.json").write_text(json.dumps({
        "schema": BINDINGS_SCHEMA,
        "generated": iso(datetime.now(timezone.utc)),
        "source": args.source,
        "physics_profile": args.physics_profile,
        "prepared_root": str(prepared_root),
        "authority_dir": str(authority_dir),
        "run_dir": str(out / "run"),
        "proof_sha256": proof_sha,
        "source_manifest_sha256": manifest_sha,
        "prepared_content_sha256": content_sha,
        "plan": plan.to_payload(),
        "case_name": case_name,
        "site": args.site,
        "radars": selection.to_payload(),
        "seed": seed,
    }, indent=1), encoding="utf-8")
    if _stop(args.stop_after, "prepare"):
        return _stopped_early(out, receipts, args, "prepare")

    # ---- georeference forecast ------------------------------------------
    print(f"GPU before forecast: {gpu_snapshot()}")
    run_dir = out / "run"
    run_stage("forecast", forecast_cmd(
        prepared_root=prepared_root, authority_dir=authority_dir,
        profile=args.physics_profile, run_dir=run_dir,
        proof_sha=proof_sha, manifest_sha=manifest_sha,
        content_sha=content_sha, source=args.source),
        cwd=repo_root, receipts_dir=receipts, index=7)
    if _stop(args.stop_after, "forecast"):
        return _stopped_early(out, receipts, args, "forecast")

    # ---- obs ladder ------------------------------------------------------
    obs_dir = out / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_files: list[Path] = []
    grid_wrfouts: list[Path] = []
    for k, valid in enumerate(plan.cycle_times):
        grid = run_dir / "wrfout" / wrfout_name(
            plan.init, (k + 1) * plan.cycle_seconds)
        out_nc = obs_dir / (f"obs-{args.site.lower()}-"
                            f"{valid:%Y%m%d%H%M}.nc")
        run_stage(f"obs-{valid:%H%M}", obs_cmd(
            selection=selection, valid=valid, grid_wrfout=grid,
            out_nc=out_nc, work_dir=out / "vols",
            bucket=args.bucket, dealias=DealiasChoice.from_args(args)),
            cwd=repo_root, receipts_dir=receipts, index=8)
        obs_files.append(out_nc)
        grid_wrfouts.append(grid)
    if _stop(args.stop_after, "obs"):
        return _stopped_early(out, receipts, args, "obs")

    # ---- cycled nowcast --------------------------------------------------
    print(f"GPU before cycle: {gpu_snapshot()}")
    cycle_out = out / "cycle"
    run_stage("cycle", cycle_cmd(
        prepared_root=prepared_root, authority_dir=authority_dir,
        profile=args.physics_profile, plan=plan, members=args.members,
        obs_files=obs_files, grid_wrfouts=grid_wrfouts,
        cycle_out=cycle_out, proof_sha=proof_sha,
        manifest_sha=manifest_sha, content_sha=content_sha,
        seed=seed, solve_device=args.solve_device,
        horizontal_loc_m=args.horizontal_loc_m,
        vertical_loc_m=args.vertical_loc_m,
        length_scale_km=length_scale_km, source=args.source,
        memory_budget_mib=args.memory_budget_mib,
        hydrometeors=args.hydrometeors,
        positivity_policy=args.positivity_policy,
        reflectivity_analysis=args.reflectivity_analysis,
        clear_air_analysis=args.clear_air_analysis,
        surface_obs=args.surface_obs,
        sfc_t2_sigma_k=args.sfc_t2_sigma_k,
        sfc_wspd_sigma_ms=args.sfc_wspd_sigma_ms,
        sfc_max_age_s=args.sfc_max_age_s,
        goes_cwp=args.goes_cwp,
        cwp_vertical_loc_m=args.cwp_vertical_loc_m,
        nest=nest),
        cwd=repo_root, receipts_dir=receipts, index=9)
    if _stop(args.stop_after, "cycle"):
        return _stopped_early(out, receipts, args, "cycle")

    # ---- gallery ---------------------------------------------------------
    run_stage("render", render_cmd(case_dir=out),
              cwd=repo_root, receipts_dir=receipts, index=10)

    receipt = {
        "schema": SCHEMA,
        "generated": iso(datetime.now(timezone.utc)),
        "honesty": ("demo-grade nowcast; UNSCORED, outside any "
                    "registered campaign; no skill claim is made or "
                    "implied"),
        "site": args.site,
        # Which radars were ASKED for, and by which route.  The
        # rolling verifier reads this back so its observed composites
        # are built from the same radars the cycle assimilated.
        "radars": selection.to_payload(),
        # How the observations were BUILT, as opposed to which radars
        # they came from.  The rolling verifier reads this back so its
        # truth composites are built the same way the assimilated ones
        # were; a run graded against a differently-built truth field
        # would show the difference as skill.
        "obs": {
            **DealiasChoice.from_args(args).to_payload(),
            # Which streams this run CLAIMS. Whether each one actually
            # assimilated anything is measured per cycle by
            # woof.da.treatment and recorded in cycle-report.json under
            # "treatment" -- these are the claims, those are the counts,
            # and the run stops if they disagree.
            "streams_requested": {
                "radial_velocity": True,
                "reflectivity": bool(args.reflectivity_analysis),
                "clear_air_reflectivity": bool(args.clear_air_analysis),
                "surface": args.surface_obs is not None,
                "cloud_water_path": bool(args.goes_cwp),
            },
            "surface_obs_file": (None if args.surface_obs is None
                                 else str(args.surface_obs)),
            "goes_cwp_files": [str(p) for p in (args.goes_cwp or ())],
        },
        "members": args.members,
        # What shaped this run's size, so a receipt can be read back
        # against the measurements the defaults were set from.
        "sizing": {
            "profile": args.profile,
            "profile_applied": getattr(args, "profile_applied", {}),
            "members": args.members,
            "memory_budget_mib": args.memory_budget_mib,
            "vram_gib": args.vram_gib,
            # What this run ACTUALLY asked the cycle driver for, not a
            # constant: the nest is reachable from this door now, and a
            # receipt that said False either way would be the reason a
            # reader could not tell the two runs apart.
            "nested_free_forecast": nest.requested,
            "nest": nest.to_payload(),
            "concurrent_members": False,
        },
        "plan": plan.to_payload(),
        "seed": seed,
        "survey": {"archive_lag_seconds": survey["archive_lag_seconds"],
                   "motion": survey["motion"],
                   # Which clock listed the volumes this domain was
                   # sited from: a reader of the case a year later
                   # cannot tell a replay from a live run otherwise.
                   "clock": survey["clock"],
                   "receipt": "receipts/00-survey.json"},
        "domain_center": center,
        "case_name": case_name,
        "length_scale_km": length_scale_km,
        "length_scale_note": scale_note,
        "outputs": {
            "case_toml": str(case_toml),
            "authority": str(authority_dir),
            "prepared_root": str(prepared_root),
            "run_dir": str(run_dir),
            "obs": [str(p) for p in obs_files],
            "cycle_out": str(cycle_out),
            "forecast_frames": forecast_frames_block(
                cycle_out, cycles=args.cycles, free_legs=args.free_legs),
            "gallery": str(out / "gallery"),
        },
    }

    # ---- auto-verification handoff --------------------------------------
    # The free forecast runs past the last observation, so its grade
    # does not exist yet.  Rather than leave that to a human, hand the
    # case to a detached verifier that grades each frame as the archive
    # covers its valid time.  DEFAULT ON; --no-verify opts out.
    frames = initial_frames(plan.to_payload()["free_leg_times"])
    state = handoff_state(no_verify=args.no_verify, frames=frames)
    receipt["verification"] = verification_block(frames=frames,
                                                 state=state)
    (out / "nowcast-receipt.json").write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    print("nowcast-receipt.json written; gallery at",
          out / "gallery" / "index.html")

    if state == "disabled":
        print("verification: OFF (--no-verify); start it later with "
              f"`python -m tools.da_nowcast watch --case-dir {out}`")
        return 0
    if state == "complete":
        print("verification: nothing to grade (no free-forecast legs)")
        return 0

    block = start_verification(
        out, frames=frames, poll_seconds=args.verify_poll_seconds,
        max_minutes=args.verify_max_minutes,
        max_offset_seconds=args.verify_max_offset_seconds,
        bucket=args.bucket, repo_root=repo_root)
    print(f"verification: rolling, detached pid "
          f"{block['watcher']['pid']}; it grades each free-forecast "
          "frame as the archive covers it and updates the gallery in "
          "place")
    print(f"  poll {out / 'nowcast-receipt.json'} -> .verification."
          "state (rolling -> complete)")
    print(f"  log  {block['watcher']['log']}")
    return 0


def _case_context(case_dir: Path) -> tuple[RadarSelection, datetime,
                                           list[dict], bool]:
    """Radar selection, model init, free-forecast frames and obs treatment.

    The selection comes from the receipt rather than from the
    verifier's own command line, which is what makes ``watch`` and
    ``verify`` grade against the same radars the cycle assimilated
    without being told twice.  A receipt with no ``radars`` block is a
    single-radar case and reads back as one.

    The dealias flag rides back the same way and for the same reason.  A
    receipt with no ``obs`` block predates the flag and reads back False,
    which is what those runs did.

    Frames resume from whatever the gallery has already graded, so a
    verifier that is re-run (or restarted) never re-grades work nor
    forgets it.
    """

    receipt = read_receipt(case_dir)
    selection = RadarSelection.from_payload(
        receipt.get("radars"), anchor=validate_site(receipt["site"]))
    plan_p = receipt["plan"]
    frames = merge_gallery_rows(
        initial_frames(plan_p["free_leg_times"]),
        gallery_rows(case_dir))
    dealias = DealiasChoice.from_payload(receipt.get("obs"))
    return selection, parse_iso(plan_p["init"]), frames, dealias


def verify_pipeline(args) -> int:
    """One pass, then a verdict: grade what the archive covers now."""

    case_dir: Path = args.case_dir
    selection, init, frames, dealias = _case_context(case_dir)
    repo_root = Path(__file__).resolve().parent.parent
    started = iso(datetime.now(timezone.utc))
    frames, _ = verify_pass(
        case_dir=case_dir, selection=selection, init=init, frames=frames,
        bucket=args.bucket,
        max_offset_seconds=args.max_offset_seconds,
        repo_root=repo_root, dealias=dealias)
    state = advance_state(frames, exhausted=True)
    block = verification_block(frames=frames, state=state,
                               started=started)
    write_verification(case_dir, block)
    print(block["verdict"])
    for frame in frames:
        if frame["status"] != "verified":
            print(f"  unverified {frame['valid']}: "
                  f"{frame.get('note', 'no numbers published')}")
    return 0 if state == "complete" else 1


def watch_pipeline(args) -> int:
    """Roll the verification until every frame is graded, then stop.

    This is what ``run`` hands off to by default.  It owns the case's
    verification block from its first pass onwards: each tick rewrites
    ``verification`` in the receipt, so a GUI can poll one file and
    watch ``rolling`` become ``complete``.
    """

    case_dir: Path = args.case_dir
    selection, init, frames, dealias = _case_context(case_dir)
    repo_root = Path(__file__).resolve().parent.parent
    started = iso(datetime.now(timezone.utc))
    watcher = {"pid": os.getpid(), "log": str(case_dir /
                                              "verify-watch.log"),
               "poll_seconds": args.poll_seconds,
               "max_minutes": args.max_minutes}
    deadline = time.monotonic() + args.max_minutes * 60.0
    _log(f"rolling verification for {case_dir}: {len(frames)} "
         "free-forecast frames to grade")
    while True:
        frames, _ = verify_pass(
            case_dir=case_dir, selection=selection, init=init,
            frames=frames,
            bucket=args.bucket,
            max_offset_seconds=args.max_offset_seconds,
            repo_root=repo_root, dealias=dealias)
        state = advance_state(frames,
                              exhausted=time.monotonic() > deadline)
        block = verification_block(frames=frames, state=state,
                                   started=started, watcher=watcher)
        write_verification(case_dir, block)
        _log(f"pass complete: {block['verdict']}")
        if state != "rolling":
            _log(f"VERIFICATION {state.upper()}")
            return 0 if state == "complete" else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
