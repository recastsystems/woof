"""Delayed-window multi-storm scorecard: many cases, one accurate table.

Every skill number WaH has published so far comes from one storm.  The
caveat "N=1" sits on every claim, and wall clock is why: a live case
cannot be graded until reality arrives, so cases accumulate at one per
night of storms somebody watched.

This harness removes the wall clock.  A front-door run whose
``--window-end`` sits ~90+ minutes in the past is *scientifically
identical* to a live run -- the live chunk feed assembles to the very
bytes the archive stores (``evidence/da-demo/live-feed/paired-run-1.json``:
volume sha256 equal across the live-chunk and archive feeds) -- but every
verification frame already exists, so the full grade is available the
moment the free forecast finishes.  Skill measurement becomes
compute-bound: one card-night is a ten-case scorecard.

Four subcommands, receipts at every stage:

    discover   survey the archive over a date range for qualifying
               convective events -- (site, window) pairs found by the
               front door's own echo census, ranked, and deduplicated so
               one long-lived system does not count five times.
               Selection is BLIND TO SKILL by construction: nothing in
               it looks at a forecast.  Every coverage cap it applies
               -- the census limit, ``--max-cases`` -- is recorded WITH
               the cases it removed, because a cap that keeps the
               largest storms and does not say so is a selection nobody
               can audit.
    plan       freeze a campaign: the discovery's cases, one frozen
               front-door configuration, its hash, and cost estimates.
    run        drive the EXISTING front door end to end per case
               (prepare, obs ladder, cycles, free forecast, one-shot
               verification, scorer of record), sequentially, behind
               the same GPU admission gate every queue on this card
               polls.  Failures are recorded and skipped, never
               silently retried.  Every PLANNED case gets an outcome,
               so a campaign the card never freed for says what it did
               not cover instead of shrinking its own denominator, and
               the delayed-window floor is re-checked HERE, where the
               card is actually taken.  Designed to live inside a
               detached scheduled task (harness tasks die at 60 min).
    scorecard  aggregate across cases: per-lead FSS distributions
               (median/IQR, not just the mean), column-count bias,
               control-relative skill, structure metrics where the
               scorer of record provides them, and a per-case table so
               outliers stay visible.  One receipt JSON plus one summary
               figure, written into the campaign directory.

Discovery is two passes so the archive is not bulk-downloaded:

    pass 1  payload-free: ``rw_nexrad list`` per site per day.  A run of
            volumes at precipitation-VCP cadence and precipitation-sized
            volumes marks a *candidate* window.  This is a PREFILTER and
            the receipt says so; volume size never qualifies a case.
    pass 2  the qualifying test: sample volumes from each candidate are
            fetched and decoded through the same seam the front door
            surveys with, and the case qualifies on the front door's own
            echo census (``tools.da_nowcast.echo_stats``) -- gates at or
            above the reflectivity threshold on the low sweeps.

The site roster comes from ``rw_nexrad sites`` (the vendored table) or
from ``--sites``.  No radar-site name belongs in this file, its
defaults, or its identifiers (standing owner rule).

ACCURACY: the campaign receipt and the summary figure both carry the
configuration hash, the case count, and the full selection criteria, so
nobody can be accused of cherry-picking after the fact -- and both are
labeled demo-grade, because every tool this drives is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:                                    # python -m tools.da_scorecard
    from tools.da_nowcast import (echo_stats, geojson_box, iso,
                                  motion_from_centroids, parse_iso,
                                  site_domain_center, validate_site)
    from tools.da_sweep_run import (Status, wait_for_card,
                                    wait_for_predecessors)
except ImportError:                     # python tools/da_scorecard.py
    from da_nowcast import (echo_stats, geojson_box, iso,
                            motion_from_centroids, parse_iso,
                            site_domain_center, validate_site)
    from da_sweep_run import Status, wait_for_card, wait_for_predecessors

DISCOVERY_SCHEMA = "gpuwm-da.scorecard-discovery.v1"
PLAN_SCHEMA = "gpuwm-da.scorecard-plan.v1"
SCORECARD_SCHEMA = "gpuwm-da.scorecard.v1"

# ---------------------------------------------------------------------------
# Defaults, each with its provenance
# ---------------------------------------------------------------------------

#: The delayed-window floor.  A case's LAST free-forecast valid time must
#: sit at least this far in the past before the campaign will run it, so
#: every verification frame is already archived and the one-shot verifier
#: grades everything on its first pass.  90 minutes = the archive-lag
#: ceiling the front door tolerates (900 s) plus the verifier's own
#: minimum frame age, with an hour of slack for a slow archive night
#: (lag samples: evidence/da-demo/live-feed/lag-samples-20260805.json).
DELAY_FLOOR_SECONDS = 5400.0

#: Pass-1 prefilter: a volume this large at 35 dBZ-era compression is in
#: a precipitation VCP with real echo; clear-air volumes on the same
#: sites run 2-6 MB.  Measured against the proven case's night: its
#: volumes are 14 MB (evidence/da-demo/live-fire-3/build-0415.json names
#: one; the archive listing around it shows 11-16 MB).  PREFILTER ONLY --
#: qualification is pass 2's echo census, never this number.
PREFILTER_MIN_SIZE_BYTES = 8_000_000

#: Pass-1 prefilter: precipitation VCPs (12/212) complete in 4-6 minutes;
#: clear-air VCPs take ~10.  Volumes spaced wider than this break a run.
PREFILTER_MAX_CADENCE_SECONDS = 420.0

#: Pass-1 prefilter: a run shorter than this cannot carry six applied
#: cycles after init snaps to the next whole hour, so listing it as a
#: candidate would only cost a census that must then refuse.
PREFILTER_MIN_DURATION_MINUTES = 90.0

#: Pass 2 qualification: median census gates >= this, at the front
#: door's default 35 dBZ threshold on the low sweeps.  Three times the
#: front door's 500-gate motion floor: motion needs any echo, a
#: scorecard case needs a storm.  Configurable, recorded per case.
MIN_ECHO_GATES = 1500

#: How many candidates (prefilter-ranked) pass 2 will census before
#: stopping.  A cap on downloads, not on accuracy: the receipt records
#: how many candidates existed and how many were censused, so a reader
#: can see exactly where the survey stopped looking.
CENSUS_LIMIT = 48

#: Deduplication: two qualifying windows are the same event when their
#: radars sit within this distance AND their windows are within
#: DEDUP_HOURS of overlapping.  CONUS WSR-88Ds are ~230 km apart, so 300
#: km catches a storm complex seen by neighbors; the time arm keeps
#: yesterday's system at the same site as its own case.
DEDUP_KM = 300.0
DEDUP_HOURS = 3.0

#: The shipped baseline shape this scorecard was built to grade first:
#: the front door's own defaults.  Restated here (not imported) so the
#: frozen-config hash cannot drift when a front-door default moves --
#: a moved default would then show up as a DIFFERENT hash, which is the
#: point of having one.
BASELINE_CONFIG = {
    "what": "shipped 3 km N=10 single-radar baseline "
            "(tools/da_nowcast.py defaults, 2026-08-05)",
    "dx_km": 3.0,
    "members": 10,
    "cycles": 6,
    "cycle_seconds": 900,
    "free_legs": 6,
    "source": "gfs",
    "physics_profile": "wsm6-ysu-mm5-noah-no-radiation-v1",
    "solve_device": "cuda",
    "horizontal_loc_m": 12000.0,
    "vertical_loc_m": 3000.0,
    "echo_threshold_dbz": 35.0,
    "history_interval_seconds": 900.0,
    "seed": "front-door default (derived from the window-end date)",
    "verification": "one-shot (tools.da_nowcast verify) after the run; "
                    "the delayed window guarantees every frame is "
                    "already archived, so one pass grades everything",
}

KM_PER_DEG_LAT = 111.32


class ScorecardError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# small pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float,
                 lon2: float) -> float:
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0 * math.asin(math.sqrt(a))


def config_hash(config: dict) -> str:
    """SHA-256 of the canonical JSON of a frozen configuration."""

    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def ceil_hour(stamp: datetime) -> datetime:
    base = stamp.replace(minute=0, second=0, microsecond=0)
    return base if base == stamp else base + timedelta(hours=1)


def precip_runs(volumes: list[dict], *, min_size_bytes: int,
                max_cadence_seconds: float,
                min_duration_seconds: float) -> list[dict]:
    """Maximal runs of precipitation-mode volumes in one listing.

    ``volumes``: rw_nexrad list entries ({"valid_time", "size_bytes"}),
    any order.  A run extends while consecutive qualifying volumes are
    within the cadence ceiling; it is reported once it spans the
    duration floor.  PREFILTER: this looks at bytes and clocks only.
    """

    qual = sorted((v for v in volumes
                   if int(v.get("size_bytes", 0)) >= min_size_bytes),
                  key=lambda v: v["valid_time"])
    runs: list[dict] = []
    start = prev = None
    count = 0
    sizes: list[int] = []

    def close() -> None:
        if start is None:
            return
        span = (parse_iso(prev["valid_time"])
                - parse_iso(start["valid_time"])).total_seconds()
        if span >= min_duration_seconds:
            runs.append({
                "start": start["valid_time"],
                "end": prev["valid_time"],
                "volumes": count,
                "median_size_bytes": int(statistics.median(sizes)),
                "duration_seconds": span,
            })

    for vol in qual:
        if prev is not None:
            gap = (parse_iso(vol["valid_time"])
                   - parse_iso(prev["valid_time"])).total_seconds()
            if gap > max_cadence_seconds:
                close()
                start, count, sizes = None, 0, []
        if start is None:
            start = vol
        prev = vol
        count += 1
        sizes.append(int(vol.get("size_bytes", 0)))
    close()
    return runs


def merge_runs(runs: list[dict], *, max_gap_seconds: float) -> list[dict]:
    """Merge one site's runs across day-listing boundaries.

    Pass 1 lists per day, so an event crossing midnight arrives as two
    runs neither of which may carry a whole case; adjacent runs within
    the cadence ceiling are one event and are merged back here.
    """

    merged: list[dict] = []
    for run in sorted(runs, key=lambda r: r["start"]):
        if merged:
            prev = merged[-1]
            gap = (parse_iso(run["start"])
                   - parse_iso(prev["end"])).total_seconds()
            if gap <= max_gap_seconds:
                prev["end"] = max(prev["end"], run["end"])
                prev["volumes"] += run["volumes"]
                prev["median_size_bytes"] = max(
                    prev["median_size_bytes"], run["median_size_bytes"])
                prev["duration_seconds"] = (
                    parse_iso(prev["end"])
                    - parse_iso(prev["start"])).total_seconds()
                continue
        merged.append(dict(run))
    return merged


def case_window(run_start: datetime, run_end: datetime, *, cycles: int,
                cycle_seconds: int, free_legs: int, now: datetime,
                delay_floor_seconds: float = DELAY_FLOOR_SECONDS,
                ) -> tuple[datetime, datetime] | tuple[None, str]:
    """(init, window_end) for one candidate run, or (None, why not).

    init is the first whole hour at/after the run's start (the front
    door requires a whole-hour init); the applied cycles must fit inside
    the echo run (window_end <= run end), and the LAST free-forecast
    valid time must clear the delay floor so every verification frame
    is already archived.
    """

    init = ceil_hour(run_start)
    window_end = init + timedelta(seconds=cycles * cycle_seconds)
    if window_end > run_end:
        return None, (f"echo run ends {iso(run_end)}, before the "
                      f"{cycles} applied cycles would end "
                      f"({iso(window_end)}); cycling would outlive "
                      "the storm")
    last_frame = window_end + timedelta(seconds=free_legs * cycle_seconds)
    closes = last_frame + timedelta(seconds=delay_floor_seconds)
    if closes > now:
        return None, (f"case not closed: last free-forecast frame "
                      f"{iso(last_frame)} + {delay_floor_seconds:.0f} s "
                      f"delay floor is still in the future")
    return init, window_end


def place_window(run_start: datetime, run_end: datetime,
                 peak_time: datetime, *, cycles: int, cycle_seconds: int,
                 free_legs: int, now: datetime,
                 delay_floor_seconds: float = DELAY_FLOOR_SECONDS,
                 ) -> tuple[datetime, datetime] | tuple[None, str]:
    """Place a case window inside a long event, centred on its peak.

    ``case_window`` snaps a case to the run's beginning, which for a
    twelve-hour MCS grades the free legs against convective initiation
    twelve hours before the system's maximum.  Given the census's
    peak-gate sample time, this places the window so the graded free
    legs straddle that peak -- init targets ``peak - (cycles +
    free_legs/2) * cadence`` -- clamped to whole-hour inits whose
    cycling stays inside the echo run and whose last frame clears the
    delay floor.  The peak comes from the echo census and only the
    echo census: window placement stays blind to skill.
    """

    init_min = ceil_hour(run_start)
    latest_end = min(
        run_end,
        now - timedelta(seconds=delay_floor_seconds
                        + free_legs * cycle_seconds))
    init_max = (latest_end
                - timedelta(seconds=cycles * cycle_seconds))
    init_max = init_max.replace(minute=0, second=0, microsecond=0)
    if init_max < init_min:
        return case_window(run_start, run_end, cycles=cycles,
                           cycle_seconds=cycle_seconds,
                           free_legs=free_legs, now=now,
                           delay_floor_seconds=delay_floor_seconds)
    target = peak_time - timedelta(
        seconds=(cycles + free_legs / 2.0) * cycle_seconds)
    init = ceil_hour(max(init_min, min(target, init_max)))
    init = max(init_min, min(init, init_max))
    return init, init + timedelta(seconds=cycles * cycle_seconds)


def case_is_closed(window_end: str, *, free_legs: int, cycle_seconds: int,
                   now: datetime,
                   delay_floor_seconds: float = DELAY_FLOOR_SECONDS,
                   ) -> tuple[bool, str]:
    """Does the delayed-window equivalence hold for this case yet?

    The claim this whole harness rests on is that a run whose window sits
    far enough in the past is scientifically identical to a live one
    because every verification frame is already archived.  That is an
    arithmetic statement about one case, and it is made HERE so that
    discovery, preflight and the campaign runner all make the same one
    instead of three copies that can drift apart.

    Returns (closed, why-not).
    """

    last = parse_iso(window_end) + timedelta(
        seconds=free_legs * cycle_seconds)
    margin = (now - last).total_seconds()
    if margin >= delay_floor_seconds:
        return True, ""
    return False, (f"last free-forecast frame {iso(last)} is only "
                   f"{margin / 60:.0f} min old; the delayed-window "
                   f"guarantee needs {delay_floor_seconds / 60:.0f} min")


def apply_max_cases(kept: list[dict], max_cases: int,
                    ) -> tuple[list[dict], list[dict]]:
    """Top-N truncation, with the removed cases handed back.

    ``kept`` arrives rank-ordered by census gates, so this cap does not
    remove an arbitrary tail -- it removes the SMALLEST storms, every
    time.  That is a defensible cap and an indefensible secret, so the
    cases it drops are returned rather than discarded, and the caller
    writes them into the receipt beside the ones it kept.
    """

    if max_cases is None or max_cases < 0 or len(kept) <= max_cases:
        return list(kept), []
    dropped = []
    for index, case in enumerate(kept[max_cases:], start=max_cases + 1):
        entry = dict(case)
        entry["dropped_for"] = (
            f"over --max-cases {max_cases}: ranked {index} of "
            f"{len(kept)} qualifying, deduplicated cases by census "
            f"median gates")
        dropped.append(entry)
    return kept[:max_cases], dropped


def dedup_cases(cases: list[dict], *, dedup_km: float,
                dedup_hours: float) -> tuple[list[dict], list[dict]]:
    """Greedy rank-order dedup: (kept, dropped-with-reasons).

    ``cases`` must be rank-sorted best first and carry ``site``,
    ``lat``, ``lon``, ``init`` and ``window_end`` (ISO).  Two cases are
    one event when the radars sit within ``dedup_km`` AND the windows
    are within ``dedup_hours`` of overlapping.
    """

    kept: list[dict] = []
    dropped: list[dict] = []
    slack = timedelta(hours=dedup_hours)
    for case in cases:
        c0 = parse_iso(case["init"]) - slack
        c1 = parse_iso(case["window_end"]) + slack
        twin = None
        for other in kept:
            near = haversine_km(case["lat"], case["lon"],
                                other["lat"], other["lon"]) <= dedup_km
            o0, o1 = parse_iso(other["init"]), parse_iso(other["window_end"])
            overlaps = c0 <= o1 and o0 <= c1
            if near and overlaps:
                twin = other
                break
        if twin is None:
            kept.append(case)
        else:
            entry = dict(case)
            entry["dropped_for"] = (
                f"same event as {twin['site']} {twin['window_end']}: "
                f"radars {haversine_km(case['lat'], case['lon'], twin['lat'], twin['lon']):.0f} km apart, windows within "
                f"{dedup_hours:g} h")
            dropped.append(entry)
    return kept, dropped


def _quantiles(values: list[float]) -> dict:
    """median/IQR/min/max -- the distribution, not just the mean."""

    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return {"n": 0}
    if n == 1:
        q1 = q3 = ordered[0]
    else:
        q1, _, q3 = statistics.quantiles(ordered, n=4)
    return {"n": n,
            "median": round(statistics.median(ordered), 4),
            "q1": round(q1, 4), "q3": round(q3, 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
            "mean": round(statistics.fmean(ordered), 4)}


def aggregate_frames(case_rows: list[dict]) -> dict:
    """Cross-case aggregation, keyed by lead minutes.

    ``case_rows``: one entry per case, each carrying ``case_id`` and
    ``frames`` -- verified front-door frames with lead_minutes,
    fss30_fcst, fss30_control, obs_cols_gt35, fcst_cols_gt35_in_echo
    (and optionally a ``structure`` block from the scorer of record).
    """

    leads: dict[float, dict[str, list[float]]] = {}
    structure_present = False
    for row in case_rows:
        for frame in row["frames"]:
            lead = frame["lead_minutes"]
            bucket = leads.setdefault(lead, {
                "fss_fcst": [], "fss_control": [], "fss_delta": [],
                "count_bias": []})
            bucket["fss_fcst"].append(frame["fss30_fcst"])
            bucket["fss_control"].append(frame["fss30_control"])
            bucket["fss_delta"].append(
                frame["fss30_fcst"] - frame["fss30_control"])
            if frame.get("obs_cols_gt35"):
                bucket["count_bias"].append(
                    frame["fcst_cols_gt35_in_echo"]
                    / frame["obs_cols_gt35"])
            if frame.get("structure"):
                structure_present = True
    per_lead = []
    for lead in sorted(leads):
        bucket = leads[lead]
        per_lead.append({
            "lead_minutes": lead,
            "fss30_fcst": _quantiles(bucket["fss_fcst"]),
            "fss30_control": _quantiles(bucket["fss_control"]),
            "fss30_delta_vs_control": _quantiles(bucket["fss_delta"]),
            "column_count_bias": _quantiles(bucket["count_bias"]),
            "cases_beating_control": sum(
                1 for d in bucket["fss_delta"] if d > 0.0),
        })
    return {"per_lead": per_lead,
            "structure_metrics_present": structure_present}


def lead_minutes(valid: str, window_end: str) -> float:
    return (parse_iso(valid) - parse_iso(window_end)).total_seconds() / 60.0


def campaign_verdict(*, graded: int, planned: int, failed: list[str],
                     refused: list[str], unreached: list[str],
                     scorecard_note: str = "") -> str:
    """The one line that says what a campaign actually covered.

    Scored against the PLAN, never against how far the loop happened to
    get: a case the campaign never reached is missing coverage, and a
    denominator that shrinks to match the numerator hides exactly that.
    The word is COMPLETE only when every planned case was graded.
    """

    head = ("CAMPAIGN_COMPLETE" if graded == planned and not
            (failed or refused or unreached) else "CAMPAIGN_INCOMPLETE")
    parts = [f"{head} {graded}/{planned} planned case(s) graded"]
    if failed:
        parts.append(f"failed: {', '.join(failed)}")
    if refused:
        parts.append(f"refused: {', '.join(refused)}")
    if unreached:
        parts.append(f"never run: {', '.join(unreached)}")
    return "; ".join(parts) + f".{scorecard_note}"


# ---------------------------------------------------------------------------
# rw_nexrad seams (via woof.obs.nexrad, the front door's own)
# ---------------------------------------------------------------------------
def _nexrad_binary() -> Path:
    from woof.obs.nexrad import find_nexrad_bin, nexrad_remedy
    binary = find_nexrad_bin()
    if binary is None:
        raise ScorecardError(f"no rw_nexrad front door: {nexrad_remedy()}")
    return binary


def site_roster(binary: Path, sites_arg: str | None) -> list[dict]:
    """The site table: id/lat/lon, from the vendored table or --sites."""

    proc = subprocess.run([str(binary), "sites"], capture_output=True,
                          text=True, errors="replace")
    if proc.returncode != 0:
        raise ScorecardError(f"rw_nexrad sites exited {proc.returncode}")
    record = json.loads(proc.stdout)
    if record.get("schema") != "gpuwm-obs.nexrad-sites.v1":
        raise ScorecardError(
            f"rw_nexrad sites printed schema {record.get('schema')!r}")
    roster = [{"site": s["id"], "lat": float(s["lat_deg"]),
               "lon": float(s["lon_deg"])} for s in record["sites"]]
    if sites_arg:
        wanted = {validate_site(s.strip())
                  for s in sites_arg.split(",") if s.strip()}
        roster = [s for s in roster if s["site"] in wanted]
        missing = wanted - {s["site"] for s in roster}
        if missing:
            raise ScorecardError(
                f"--sites names ids the roster does not carry: "
                f"{sorted(missing)}")
    return roster


def list_day(binary: Path, *, site: str, day: datetime,
             bucket: str | None) -> list[dict]:
    from woof.obs.nexrad import run_list
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    listing = run_list(binary, site=site, start=iso(start),
                       end=iso(start + timedelta(days=1)), bucket=bucket)
    return [v for v in listing.get("volumes", [])
            if not v["filename"].endswith("MDM")]


def census_volume(binary: Path, *, site: str, stamp: datetime,
                  work_dir: Path, bucket: str | None,
                  threshold_dbz: float) -> dict | None:
    """Fetch+decode the volume nearest ``stamp``; front-door echo census.

    None when nothing is archived within half a precipitation VCP of the
    asked time -- the caller records the miss.
    """

    from woof.obs.nexrad import run_decode, run_fetch, run_list, run_verify
    from woof.obs.sweeps import read_sweep_pack

    listing = run_list(binary, site=site,
                       start=iso(stamp - timedelta(minutes=10)),
                       end=iso(stamp + timedelta(minutes=10)),
                       bucket=bucket)
    volumes = [v for v in listing.get("volumes", [])
               if not v["filename"].endswith("MDM")]
    if not volumes:
        return None
    chosen = min(volumes, key=lambda v: abs(
        (parse_iso(v["valid_time"]) - stamp).total_seconds()))
    work_dir.mkdir(parents=True, exist_ok=True)
    when = parse_iso(chosen["valid_time"])
    run_fetch(binary, site=site, start=iso(when - timedelta(seconds=30)),
              end=iso(when + timedelta(seconds=30)), out=work_dir,
              bucket=bucket)
    volume_path = work_dir / chosen["filename"]
    if not volume_path.is_file():
        return None
    pack_path = work_dir / (chosen["filename"] + ".census.pack")
    run_decode(binary, volume=volume_path, out=pack_path,
               moments=("REF",), max_range_km=250.0,
               max_elevation_deg=20.0)
    verify = run_verify(binary, pack=pack_path)
    if verify.get("status") != "PASS":
        return None
    volume = read_sweep_pack(pack_path)
    stats = echo_stats(volume, threshold_dbz=threshold_dbz)
    return {"volume": chosen["filename"],
            "valid_time": chosen["valid_time"],
            "antenna": {"lat_deg": float(volume.site.lat_deg),
                        "lon_deg": float(volume.site.lon_deg)},
            **stats}


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------
def discover(args) -> int:
    now = datetime.now(timezone.utc)
    # Raising the floor is a preference.  Lowering it is a change to the
    # claim -- it would select cases whose verification frames are not
    # all archived yet, which is the one thing the delayed window is for.
    # The campaign runner refuses such a case at the card regardless, so
    # this only turns a wasted survey into an immediate refusal.
    if args.delay_floor_seconds < DELAY_FLOOR_SECONDS:
        raise ScorecardError(
            f"--delay-floor-seconds {args.delay_floor_seconds:.0f} is "
            f"below the {DELAY_FLOOR_SECONDS:.0f} s floor the "
            "delayed-window equivalence rests on; it may be raised, "
            "never lowered")
    binary = _nexrad_binary()
    roster = site_roster(binary, args.sites)
    if args.end is not None:
        survey_end = parse_iso(args.end)
    else:
        horizon = (args.cycles + args.free_legs) * args.cycle_seconds
        survey_end = now - timedelta(
            seconds=horizon + args.delay_floor_seconds)
    survey_start = (parse_iso(args.start) if args.start is not None
                    else survey_end - timedelta(days=args.days))
    days = []
    cursor = survey_start.replace(hour=0, minute=0, second=0,
                                  microsecond=0)
    while cursor <= survey_end:
        days.append(cursor)
        cursor += timedelta(days=1)

    print(f"discovery: {len(roster)} site(s), "
          f"{iso(survey_start)} .. {iso(survey_end)} "
          f"({len(days)} day(s)); pass 1 is payload-free", flush=True)

    # ---- pass 1: payload-free prefilter -------------------------------
    def survey_one(pair):
        site_row, day = pair
        try:
            volumes = list_day(binary, site=site_row["site"], day=day,
                               bucket=args.bucket)
        except Exception as error:
            return {"site": site_row["site"], "day": iso(day),
                    "error": f"{error.__class__.__name__}: {error}"}
        # Day listings cover whole days; the survey window does not.
        volumes = [v for v in volumes
                   if survey_start <= parse_iso(v["valid_time"])
                   <= survey_end]
        runs = precip_runs(
            volumes, min_size_bytes=args.prefilter_min_size_bytes,
            max_cadence_seconds=args.prefilter_max_cadence_seconds,
            min_duration_seconds=args.prefilter_min_duration_minutes * 60.0)
        return {"site": site_row["site"], "lat": site_row["lat"],
                "lon": site_row["lon"], "day": iso(day), "runs": runs}

    pairs = [(s, d) for s in roster for d in days]
    with ThreadPoolExecutor(max_workers=args.list_workers) as pool:
        surveys = list(pool.map(survey_one, pairs))
    errors = [s for s in surveys if "error" in s]

    by_site: dict[str, dict] = {}
    for entry in surveys:
        if "runs" not in entry:
            continue
        slot = by_site.setdefault(entry["site"], {
            "lat": entry["lat"], "lon": entry["lon"], "runs": []})
        slot["runs"].extend(entry["runs"])

    candidates = []
    for site, slot in by_site.items():
        for run in merge_runs(
                slot["runs"],
                max_gap_seconds=args.prefilter_max_cadence_seconds):
            t0, t1 = parse_iso(run["start"]), parse_iso(run["end"])
            planned = case_window(
                t0, t1, cycles=args.cycles,
                cycle_seconds=args.cycle_seconds,
                free_legs=args.free_legs, now=now,
                delay_floor_seconds=args.delay_floor_seconds)
            if planned[0] is None:
                continue
            init, window_end = planned
            candidates.append({
                "site": site, "lat": slot["lat"], "lon": slot["lon"],
                "run": run,
                "init": iso(init), "window_end": iso(window_end),
                "prefilter_score": run["median_size_bytes"]
                * run["duration_seconds"],
            })
    candidates.sort(key=lambda c: -c["prefilter_score"])
    print(f"pass 1: {len(candidates)} candidate window(s) "
          f"({len(errors)} listing error(s))", flush=True)

    # ---- pass 2: the qualifying echo census ---------------------------
    work_root = args.out.parent / "census-vols"
    censused = 0
    qualified = []
    census_records = []
    def sample(site: str, stamp: datetime):
        try:
            return census_volume(
                binary, site=site, stamp=stamp,
                work_dir=work_root / site, bucket=args.bucket,
                threshold_dbz=args.echo_threshold_dbz)
        except Exception as error:
            return f"{error.__class__.__name__}: {error}"

    census_stopped_early = None
    for cand in candidates:
        # Both of these are coverage caps, and a coverage cap that does
        # not say what it stopped looking at is indistinguishable from a
        # survey that found nothing more.  The reason and the number of
        # candidates left unexamined go into the receipt.
        if censused >= args.census_limit:
            census_stopped_early = (
                f"census limit {args.census_limit} reached; "
                f"{len(candidates) - censused} prefilter candidate(s) "
                "were never censused")
            break
        if len(qualified) >= args.max_cases * 3:
            census_stopped_early = (
                f"{len(qualified)} qualifying case(s) is three times "
                f"--max-cases {args.max_cases}; "
                f"{len(candidates) - censused} prefilter candidate(s) "
                "were never censused")
            break
        censused += 1
        t0 = parse_iso(cand["run"]["start"])
        t1 = parse_iso(cand["run"]["end"])
        # Three samples across the event: qualification is their
        # median, and the window is then placed around their peak so a
        # twelve-hour system is graded at its sampled maximum rather
        # than at whichever end the listing happened to start.
        latest_end = min(t1, now - timedelta(
            seconds=args.delay_floor_seconds
            + args.free_legs * args.cycle_seconds))
        stamps = {
            "early": ceil_hour(t0) + timedelta(minutes=45),
            "mid": t0 + (min(t1, latest_end) - t0) / 2,
            "late": latest_end,
        }
        samples = {label: sample(cand["site"], stamp)
                   for label, stamp in stamps.items()}
        got = {label: s for label, s in samples.items()
               if isinstance(s, dict) and "gates" in s}
        record = dict(cand)
        record["census"] = samples
        if len(got) < 2:
            record["verdict"] = "census could not sample the window"
            census_records.append(record)
            continue
        median_gates = int(statistics.median(
            s["gates"] for s in got.values()))
        record["census_median_gates"] = median_gates
        record["census_peak_gates"] = max(
            s["gates"] for s in got.values())
        if median_gates < args.min_echo_gates:
            record["verdict"] = (f"median {median_gates} gates < "
                                 f"{args.min_echo_gates} floor")
            census_records.append(record)
            continue
        peak_label = max(got, key=lambda k: got[k]["gates"])
        peak_time = parse_iso(got[peak_label]["valid_time"])
        placed = place_window(
            t0, t1, peak_time, cycles=args.cycles,
            cycle_seconds=args.cycle_seconds,
            free_legs=args.free_legs, now=now,
            delay_floor_seconds=args.delay_floor_seconds)
        if placed[0] is None:
            record["verdict"] = f"window placement refused: {placed[1]}"
            census_records.append(record)
            continue
        init, window_end = placed
        record["init"] = iso(init)
        record["window_end"] = iso(window_end)
        record["window_placed_by"] = (
            f"census peak at {iso(peak_time)} ({peak_label} sample, "
            f"{got[peak_label]['gates']} gates)")
        # Domain siting exactly as a live front-door run at the window
        # would have sited it: echo centroid + downstream lead, via the
        # front door's own functions, from volumes AT the window.
        older = sample(cand["site"],
                       window_end - timedelta(seconds=2700))
        newest = sample(cand["site"], window_end)
        record["census"]["motion-older"] = older
        record["census"]["window-end"] = newest
        if not (isinstance(newest, dict) and "gates" in newest):
            record["verdict"] = ("no volume at the placed window end; "
                                 "cannot site the domain")
            census_records.append(record)
            continue
        motion = None
        if isinstance(older, dict) and older.get("gates", 0) > 0:
            motion = motion_from_centroids(older, newest, min_gates=500)
        horizon = (args.cycles + args.free_legs) * args.cycle_seconds
        center = site_domain_center(
            newest["antenna"]["lat_deg"], newest["antenna"]["lon_deg"],
            newest, motion, horizon_seconds=horizon,
            downstream_fraction=0.35, max_offset_km=60.0)
        record["domain_center"] = center
        record["domain_polygon"] = geojson_box(
            center["lat"], center["lon"], 198.0)
        record["verdict"] = "QUALIFIED"
        record["rank_score"] = median_gates
        census_records.append(record)
        qualified.append(record)
        print(f"  qualified {cand['site']} window-end "
              f"{record['window_end']} (median {median_gates} gates, "
              f"peak {record['census_peak_gates']})", flush=True)

    qualified.sort(key=lambda c: -c["rank_score"])
    kept, dropped = dedup_cases(qualified, dedup_km=args.dedup_km,
                                dedup_hours=args.dedup_hours)
    kept, over_cap = apply_max_cases(kept, args.max_cases)

    receipt = {
        "schema": DISCOVERY_SCHEMA,
        "generated": iso(now),
        "window": {"start": iso(survey_start), "end": iso(survey_end)},
        "sites_surveyed": len(roster),
        # WHICH sites, not just how many.  The count alone cannot
        # distinguish a whole-roster survey from one aimed at a handful
        # of favourable radars, and "nobody can say the cases were
        # cherry-picked" is exactly the property this receipt exists to
        # underwrite.  The restriction is recorded as it was given.
        "site_ids_surveyed": sorted(s["site"] for s in roster),
        "sites_restricted_to": args.sites,
        # Which archive served the bytes.  Not a way to pick favourable
        # storms, but it is the provenance of every number downstream,
        # and "the default" is not recoverable from the receipt later.
        "bucket": args.bucket,
        "selection_criteria": {
            "what": "echo-census selection, blind to skill: nothing in "
                    "this survey reads a forecast",
            "prefilter": {
                "role": "candidate finding ONLY; never qualifies a case",
                "min_size_bytes": args.prefilter_min_size_bytes,
                "max_cadence_seconds": args.prefilter_max_cadence_seconds,
                "min_duration_minutes": args.prefilter_min_duration_minutes,
            },
            "qualification": {
                "echo_threshold_dbz": args.echo_threshold_dbz,
                "min_median_census_gates": args.min_echo_gates,
                "census_samples_per_window": 3,
            },
            "case_geometry": {
                "cycles": args.cycles,
                "cycle_seconds": args.cycle_seconds,
                "free_legs": args.free_legs,
                "delay_floor_seconds": args.delay_floor_seconds,
            },
            "dedup": {"km": args.dedup_km, "hours": args.dedup_hours},
            "census_limit": args.census_limit,
            "max_cases": args.max_cases,
            "candidates_found": len(candidates),
            "candidates_censused": censused,
            "census_stopped_early": census_stopped_early,
            "cases_qualified": len(qualified),
            "cases_dropped_over_max_cases": len(over_cap),
        },
        "listing_errors": errors,
        "census": census_records,
        "dropped_as_duplicates": dropped,
        "dropped_over_max_cases": over_cap,
        "cases": kept,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    if census_stopped_early:
        print(f"pass 2 stopped early: {census_stopped_early}", flush=True)
    if over_cap:
        print(f"--max-cases {args.max_cases} dropped {len(over_cap)} "
              "qualifying case(s), lowest census gates first: "
              + ", ".join(f"{c['site']} {c['window_end']} "
                          f"({c['census_median_gates']} gates)"
                          for c in over_cap))
    print(f"discovery: {len(kept)} case(s) selected "
          f"({len(dropped)} deduplicated) -> {args.out}")
    for case in kept:
        print(f"  {case['site']}  init {case['init']}  window-end "
              f"{case['window_end']}  median gates "
              f"{case['census_median_gates']}")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def front_door_argv(case: dict, *, case_dir: Path, config: dict,
                    polygon: Path) -> list[str]:
    """The frozen front-door command for one case.

    ``--no-verify`` because the campaign grades with the one-shot
    verifier immediately after the run: the delayed window guarantees
    every frame is archived, and a synchronous grade means the case is
    complete before the next one starts.  ``--allow-stale`` because the
    front door's freshness gate measures the CURRENT feed, which is
    irrelevant to an archived window; the domain comes from the census
    polygon either way.
    """

    argv = [
        sys.executable, "-m", "tools.da_nowcast", "run",
        "--site", case["site"],
        "--window-end", case["window_end"],
        "--out", str(case_dir),
        "--domain-polygon", str(polygon),
        "--cycles", str(config["cycles"]),
        "--cycle-seconds", str(config["cycle_seconds"]),
        "--free-legs", str(config["free_legs"]),
        "--members", str(config["members"]),
        "--dx-km", f"{config['dx_km']:g}",
        "--source", config["source"],
        "--physics-profile", config["physics_profile"],
        "--solve-device", config["solve_device"],
        "--horizontal-loc-m", f"{config['horizontal_loc_m']:g}",
        "--vertical-loc-m", f"{config['vertical_loc_m']:g}",
        "--history-interval-seconds",
        f"{config['history_interval_seconds']:g}",
        "--allow-stale",
        "--no-verify",
    ]
    # Variant keys, forwarded only when the frozen config CARRIES them:
    # a plan written before these keys existed produces the byte-same
    # argv it always did, so no stored config_hash is disturbed and an
    # A/B arm differs from its baseline by exactly its stated key(s).
    if config.get("hydrometeors"):
        argv.append("--hydrometeors")
        argv.extend(("--positivity-policy",
                     str(config["positivity_policy"])))
    if config.get("reflectivity_analysis"):
        argv.append("--reflectivity-analysis")
    return argv


def build_plan(args) -> int:
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    if discovery.get("schema") != DISCOVERY_SCHEMA:
        raise ScorecardError(
            f"{args.discovery} is not a discovery receipt "
            f"(schema {discovery.get('schema')!r})")
    cases = discovery["cases"]
    if not cases:
        raise ScorecardError("discovery selected no cases; nothing to plan")
    config = dict(BASELINE_CONFIG)
    frozen_hash = config_hash(config)
    campaign_dir = args.campaign_dir
    plan_cases = []
    for case in cases:
        case_id = (f"{case['site'].lower()}-"
                   f"{parse_iso(case['window_end']):%Y%m%d%H%M}")
        plan_cases.append({
            "case_id": case_id,
            "site": case["site"],
            "init": case["init"],
            "window_end": case["window_end"],
            "census_median_gates": case["census_median_gates"],
            "census_peak_gates": case["census_peak_gates"],
            "domain_center": case["domain_center"],
            "domain_polygon": case["domain_polygon"],
        })
    plan = {
        "schema": PLAN_SCHEMA,
        "generated": iso(datetime.now(timezone.utc)),
        "campaign_dir": str(campaign_dir),
        "discovery_receipt": str(args.discovery),
        "selection_criteria": discovery["selection_criteria"],
        "config": config,
        "config_hash": frozen_hash,
        "n_cases": len(plan_cases),
        "cases": plan_cases,
        "estimates": {
            "disk_gib_per_case": 2.5,
            "disk_gib_total": round(2.5 * len(plan_cases), 1),
            "wall_minutes_per_case": 45,
            "basis": "the proven case's front-door tree: georeference "
                     "run ~0.9 GiB at 900 s cadence, prepared cache "
                     "~0.2 GiB, obs+verify grids ~0.5 GiB, radar "
                     "volumes ~0.3 GiB, cycle output ~0.1 GiB; walls "
                     "from the live-fire and sweep receipts",
        },
        "honesty": {
            "grade": "demo-grade, like every tool this drives",
            "equivalence": "delayed window == live run: live chunk "
                           "assembly is byte-identical to the archive "
                           "(evidence/da-demo/live-feed/"
                           "paired-run-1.json, equal volume sha256 "
                           "across feeds)",
            "selection": "by echo census, blind to skill; criteria and "
                         "census receipts are in the discovery receipt",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"plan: {len(plan_cases)} case(s), config {frozen_hash[:12]}, "
          f"~{plan['estimates']['disk_gib_total']} GiB, -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# preflight (CPU-only; run in-turn at staging and again by the launcher)
# ---------------------------------------------------------------------------
def preflight(args) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ScorecardError(f"{args.plan} is not a campaign plan")
    now = datetime.now(timezone.utc)
    problems: list[str] = []
    notes: list[str] = []

    if config_hash(plan["config"]) != plan["config_hash"]:
        problems.append("config_hash does not match the stored config; "
                        "the plan was edited after it was frozen")

    try:
        binary = _nexrad_binary()
        notes.append(f"rw_nexrad: {binary}")
    except ScorecardError as error:
        problems.append(str(error))

    try:
        from tools.da_nowcast import resolve_bridge, resolve_geog_root
        notes.append(f"bridge: {resolve_bridge(None)}")
        notes.append(f"geog root: {resolve_geog_root(None)}")
    except Exception as error:
        problems.append(f"front-door resource: {error}")

    for case in plan["cases"]:
        validate_site(case["site"])
        closed, why = case_is_closed(
            case["window_end"], free_legs=plan["config"]["free_legs"],
            cycle_seconds=plan["config"]["cycle_seconds"], now=now)
        if not closed:
            problems.append(f"{case['case_id']}: {why}")
        init_age_days = (now - parse_iso(case["init"])).days
        if init_age_days > 9:
            notes.append(
                f"{case['case_id']}: init {init_age_days} days old -- "
                "beyond the ~10-day NOMADS window, the GFS fetch must "
                "come through the S3 archive route")

    verdict = "GO" if not problems else "REFUSED"
    payload = {"schema": "gpuwm-da.scorecard-preflight.v1",
               "checked": iso(now), "verdict": verdict,
               "problems": problems, "notes": notes,
               "plan": str(args.plan),
               "config_hash": plan["config_hash"]}
    out = args.plan.parent / "preflight-verdict.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"preflight {verdict}: {len(problems)} problem(s), "
          f"{len(notes)} note(s) -> {out}")
    for line in problems:
        print(f"  PROBLEM {line}")
    for line in notes:
        print(f"  note {line}")
    return 0 if verdict == "GO" else 1


# ---------------------------------------------------------------------------
# run (the campaign; lives inside a detached scheduled task)
# ---------------------------------------------------------------------------
def run_campaign(args) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ScorecardError(f"{args.plan} is not a campaign plan")
    campaign_dir: Path = args.campaign_dir
    campaign_dir.mkdir(parents=True, exist_ok=True)
    status = Status(campaign_dir / "campaign-status.log")
    verdict_path = campaign_dir / "VERDICT.txt"
    gate_log = campaign_dir / "gate.log"
    repo = Path(__file__).resolve().parent.parent

    status.say(f"CAMPAIGN START pid {os.getpid()} plan {args.plan} "
               f"config {plan['config_hash'][:12]} "
               f"n_cases {plan['n_cases']}")
    deadline = time.monotonic() + args.max_wait_hours * 3600.0
    if args.wait_for and not wait_for_predecessors(
            status, list(args.wait_for), poll_seconds=args.poll_seconds,
            deadline=deadline):
        verdict = ("PREDECESSOR_TIMEOUT a predecessor queue never wrote "
                   "its status file; nothing was run and nothing was "
                   "stopped")
        status.say("VERDICT " + verdict)
        verdict_path.write_text(verdict + "\n", encoding="utf-8")
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WOOF_GEOG_ROOT", str(Path.home() / "WPS_GEOG"))

    def run_step(case_id: str, name: str, argv: list[str]) -> int:
        logs = campaign_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out_path = logs / f"{case_id}.{name}.out"
        err_path = logs / f"{case_id}.{name}.err"
        status.say(f"{case_id}: step {name} start")
        t0 = time.monotonic()
        with out_path.open("w", encoding="utf-8") as out, \
                err_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(argv, cwd=str(repo), env=env,
                                  stdout=out, stderr=err, text=True)
        status.say(f"{case_id}: step {name} exit {proc.returncode} "
                   f"after {time.monotonic() - t0:.1f} s")
        if proc.returncode != 0:
            tail = err_path.read_text(
                encoding="utf-8", errors="replace").strip().splitlines()
            for line in tail[-12:]:
                status.say(f"{case_id}:   stderr| {line}")
        return proc.returncode

    outcomes: dict[str, str] = {}
    for case in plan["cases"]:
        case_id = case["case_id"]
        case_dir = campaign_dir / "cases" / case_id
        done = campaign_dir / "arms" / f"{case_id}.done"
        failed_marker = campaign_dir / "arms" / f"{case_id}.failed"
        done.parent.mkdir(parents=True, exist_ok=True)
        if done.is_file():
            status.say(f"{case_id}: already done - skipping")
            outcomes[case_id] = "skipped (already done)"
            continue
        if failed_marker.is_file() and not args.retry_failed:
            # Never silently retried into difference: a failed case
            # stays failed until a human passes --retry-failed.
            status.say(f"{case_id}: previously failed - skipping "
                       "(pass --retry-failed to rerun)")
            outcomes[case_id] = "skipped (previously failed)"
            continue

        # The equivalence claim, enforced where the card is actually
        # taken.  Discovery places windows behind the floor and preflight
        # re-checks them, but neither of those runs the front door, and
        # the front door is told --allow-stale so its own freshness gate
        # will not catch a case that is not closed.  A hand-edited plan,
        # or a discovery run with --delay-floor-seconds lowered, reaches
        # this loop otherwise.  Checked against the module constant, not
        # against anything the plan carries, so the plan cannot lower it.
        closed, why = case_is_closed(
            case["window_end"], free_legs=plan["config"]["free_legs"],
            cycle_seconds=plan["config"]["cycle_seconds"],
            now=datetime.now(timezone.utc))
        if not closed:
            outcomes[case_id] = f"REFUSED (not a delayed window: {why})"
            status.say(f"{case_id}: REFUSED - {why}. The card was not "
                       "taken and nothing was stopped.")
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        polygon = case_dir / "census-domain.geojson"
        polygon.write_text(json.dumps(case["domain_polygon"]),
                           encoding="utf-8")
        (case_dir / "case-plan.json").write_text(json.dumps({
            "schema": PLAN_SCHEMA + "+case",
            "config_hash": plan["config_hash"],
            **case}, indent=1), encoding="utf-8")

        if not wait_for_card(status, args.gate, gate_log, label=case_id,
                             poll_seconds=args.poll_seconds,
                             deadline=deadline):
            outcomes[case_id] = "not run (card held to the deadline)"
            break

        t0 = time.monotonic()
        steps = [
            ("frontdoor", front_door_argv(
                case, case_dir=case_dir, config=plan["config"],
                polygon=polygon)),
            ("verify", [sys.executable, "-m", "tools.da_nowcast",
                        "verify", "--case-dir", str(case_dir)]),
            ("score", [sys.executable, "-m", "tools.da_sweep_score",
                       "--composites", str(case_dir / "cycle"
                                           / "composites"),
                       "--obs-dir", str(case_dir / "obsverify"),
                       "--obs-glob", "*.nc",
                       "--first-free-leg",
                       str(plan["config"]["cycles"]),
                       "--dx-km", f"{plan['config']['dx_km']:g}",
                       "--label", case_id,
                       "--out", str(case_dir / "sweep-score.json")]),
        ]
        failed = None
        for name, argv in steps:
            code = run_step(case_id, name, argv)
            if code != 0:
                failed = f"{name} exit {code}"
                break
        elapsed = time.monotonic() - t0
        if failed is None:
            done.write_text(f"DONE after {elapsed:.1f} s\n",
                            encoding="utf-8")
            outcomes[case_id] = f"OK in {elapsed:.1f} s"
            status.say(f"{case_id}: COMPLETE in {elapsed:.1f} s")
        else:
            failed_marker.write_text(
                f"FAILED ({failed}) after {elapsed:.1f} s at "
                f"{iso(datetime.now(timezone.utc))}\n", encoding="utf-8")
            outcomes[case_id] = f"FAILED ({failed}) after {elapsed:.1f} s"
            status.say(f"{case_id}: FAILED ({failed}) - recorded and "
                       "skipped, continuing to the next case")

    # ---- aggregate whatever completed ---------------------------------
    # Every PLANNED case gets an outcome, including the ones the loop
    # never reached because it broke out when the card stayed held.
    # Scoring against len(outcomes) made those cases disappear from the
    # numerator AND the denominator at once, so a campaign that got
    # three cases into a twelve-case plan reported "COMPLETE 3/4".
    for case in plan["cases"]:
        outcomes.setdefault(
            case["case_id"],
            "not run (the campaign ended before this case was reached)")

    ok = [n for n, o in outcomes.items() if o.startswith("OK")
          or o.startswith("skipped (already done)")]
    scorecard_note = ""
    if ok:
        code = run_step("campaign", "scorecard", [
            sys.executable, "-m", "tools.da_scorecard", "scorecard",
            "--plan", str(args.plan),
            "--campaign-dir", str(campaign_dir)])
        scorecard_note = (" Scorecard written." if code == 0 else
                          " SCORECARD STEP FAILED - cases intact, "
                          "aggregate by hand.")

    bad = sorted(n for n, o in outcomes.items() if o.startswith("FAILED"))
    refused = sorted(n for n, o in outcomes.items()
                     if o.startswith("REFUSED"))
    unreached = sorted(n for n, o in outcomes.items()
                       if o.startswith("not run"))
    verdict = campaign_verdict(
        # len(cases), not the plan's own n_cases: only one of those two
        # is the list this loop actually walked, and a denominator that
        # can be edited away from the thing it counts is not a count.
        graded=len(ok), planned=len(plan["cases"]), failed=bad,
        refused=refused, unreached=unreached,
        scorecard_note=scorecard_note)
    (campaign_dir / "outcomes.json").write_text(
        json.dumps({"schema": PLAN_SCHEMA + "+outcomes",
                    "planned": plan["n_cases"], "outcomes": outcomes},
                   indent=1), encoding="utf-8")
    status.say("VERDICT " + verdict)
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# scorecard (aggregation + the one figure)
# ---------------------------------------------------------------------------
def collect_case(case_dir: Path, *, window_end: str) -> dict | None:
    """One case's verified frames + optional scorer-of-record blocks."""

    receipt_path = case_dir / "nowcast-receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verification = receipt.get("verification", {})
    frames = []
    structure_by_leg: dict[int, dict] = {}
    score_path = case_dir / "sweep-score.json"
    if score_path.is_file():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        for entry in score.get("frames", []):
            if isinstance(entry.get("structure"), dict):
                structure_by_leg[int(entry["leg"])] = entry["structure"]
    cycles_applied = len(receipt.get("plan", {}).get("cycle_times", []))
    for index, frame in enumerate(verification.get("frames", [])):
        if frame.get("status") != "verified":
            continue
        row = {
            "valid": frame["valid"],
            "lead_minutes": lead_minutes(frame["valid"], window_end),
            "fss30_fcst": float(frame["fss30_fcst"]),
            "fss30_control": float(frame["fss30_control"]),
            "obs_cols_gt35": int(frame.get("obs_cols_gt35", 0)),
            "fcst_cols_gt35_in_echo": int(
                frame.get("fcst_cols_gt35_in_echo", 0)),
            "control_cols_gt35_in_echo": int(
                frame.get("control_cols_gt35_in_echo", 0)),
        }
        structure = structure_by_leg.get(cycles_applied + index)
        if structure is not None:
            row["structure"] = structure
        frames.append(row)
    if not frames:
        return None
    return {"frames": frames,
            "verification_state": verification.get("state"),
            "graded": verification.get("graded"),
            "total": verification.get("total")}


def skip_reason(campaign_dir: Path, case_id: str) -> str:
    """Why a planned case contributed nothing, from the run's own markers.

    "no verified frames (failed, unstarted, or ungraded)" was one string
    for three different things, and the difference is the whole reading
    of a scorecard: a case the front door failed on is a result about the
    model, and a case the campaign never reached is missing coverage.
    The campaign already writes markers that tell them apart.
    """

    failed = campaign_dir / "arms" / f"{case_id}.failed"
    if failed.is_file():
        detail = failed.read_text(encoding="utf-8",
                                  errors="replace").strip()
        return f"the campaign recorded a failure: {detail}"
    if (campaign_dir / "cases" / case_id / "nowcast-receipt.json").is_file():
        return ("the front door ran and left a receipt, but the "
                "verifier graded no frame in it")
    if (campaign_dir / "cases" / case_id).is_dir():
        return ("started: a case directory exists, but the front door "
                "left no receipt")
    return ("never started -- the campaign did not reach this case "
            "(see campaign-status.log and outcomes.json)")


def render_scorecard_figure(payload: dict, out_png: Path) -> None:
    """One figure: FSS distribution, count bias, and the case table.

    Styled beside the gallery rather than inside it: same title
    conventions and quiet axes (da_nowcast_render.py), no map.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leads = [p["lead_minutes"] for p in payload["per_lead"]]
    med_f = [p["fss30_fcst"].get("median") for p in payload["per_lead"]]
    q1_f = [p["fss30_fcst"].get("q1") for p in payload["per_lead"]]
    q3_f = [p["fss30_fcst"].get("q3") for p in payload["per_lead"]]
    med_c = [p["fss30_control"].get("median")
             for p in payload["per_lead"]]
    bias = [p["column_count_bias"].get("median")
            for p in payload["per_lead"]]

    fig = plt.figure(figsize=(11.0, 7.5), dpi=140)
    # top leaves room for the header block below the suptitle, which now
    # carries the graded/planned counts and names the ungraded cases.
    grid = fig.add_gridspec(2, 2, height_ratios=(2.1, 1.4),
                            hspace=0.42, wspace=0.28,
                            left=0.075, right=0.97, top=0.82,
                            bottom=0.05)
    ax_fss = fig.add_subplot(grid[0, 0])
    ax_bias = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    ax_fss.fill_between(leads, q1_f, q3_f, alpha=0.25, color="#2c6fbb",
                        linewidth=0, label="forecast IQR")
    ax_fss.plot(leads, med_f, color="#2c6fbb", marker="o",
                label="forecast median")
    ax_fss.plot(leads, med_c, color="0.45", marker="s", linestyle="--",
                label="control median")
    ax_fss.set_xlabel("lead (min past last observation)", fontsize=8)
    ax_fss.set_ylabel("FSS(30 dBZ, 27 km)", fontsize=8)
    ax_fss.set_ylim(0.0, 1.0)
    ax_fss.legend(fontsize=7, frameon=False)
    ax_fss.set_title("skill by lead, across cases", fontsize=9)

    ax_bias.axhline(1.0, color="0.55", linewidth=0.8)
    ax_bias.plot(leads, bias, color="#b0413e", marker="o")
    ax_bias.set_xlabel("lead (min past last observation)", fontsize=8)
    ax_bias.set_ylabel(">=35 dBZ column count bias (fcst/obs)",
                       fontsize=8)
    ax_bias.set_title("echo production bias by lead", fontsize=9)

    for ax in (ax_fss, ax_bias):
        ax.tick_params(labelsize=7, length=2.5, color="0.55")
        for spine in ax.spines.values():
            spine.set_color("0.55")

    ax_table.axis("off")
    columns = ["case", "window end", "mean FSS", "mean ctrl",
               "delta", "frames"]
    rows = [[r["case_id"], r["window_end"][:16] + "Z",
             f"{r['mean_fss_fcst']:.3f}", f"{r['mean_fss_control']:.3f}",
             f"{r['mean_fss_fcst'] - r['mean_fss_control']:+.3f}",
             f"{r['frames_graded']}"] for r in payload["case_rows"]]
    # The ungraded cases belong in the table too.  The receipt already
    # named them, but the figure is the artifact that travels, and a
    # figure that shows only the cases that produced numbers is a
    # survivor-biased picture of the campaign whatever the JSON says.
    skipped = payload.get("honesty", {}).get("cases_skipped", [])
    for entry in skipped:
        rows.append([entry["case_id"],
                     str(entry.get("window_end", ""))[:16] + "Z",
                     "not graded", "-", "-", "0"])
    table = ax_table.table(cellText=rows, colLabels=columns,
                           loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)

    planned = payload.get("honesty", {}).get("n_cases_planned",
                                             payload["n_cases"])
    fig.suptitle(
        f"WaH delayed-window scorecard - {payload['n_cases']} of "
        f"{planned} planned case(s) graded, baseline configuration",
        fontsize=12, y=0.975)
    fig.text(0.075, 0.912,
             f"config {payload['config_hash'][:12]}  |  selection: echo "
             "census, blind to skill  |  delayed windows are "
             "byte-identical to live runs (paired-run receipt)",
             fontsize=7.5, color="0.30")
    fig.text(0.075, 0.893,
             "DEMO-GRADE. Numbers from the gallery's own scorer; no "
             "operational skill claim is made or implied.",
             fontsize=7.5, color="#8a2d2b")
    if skipped:
        fig.text(0.075, 0.874,
                 f"{len(skipped)} planned case(s) contributed no graded "
                 "frame and are listed in the table as `not graded`: "
                 + ", ".join(e["case_id"] for e in skipped[:6])
                 + (" ..." if len(skipped) > 6 else ""),
                 fontsize=7.5, color="#8a2d2b")
    fig.savefig(out_png)
    plt.close(fig)


def build_scorecard(args) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ScorecardError(f"{args.plan} is not a campaign plan")
    campaign_dir: Path = args.campaign_dir
    case_rows = []
    collected = []
    skipped = []
    for case in plan["cases"]:
        case_dir = campaign_dir / "cases" / case["case_id"]
        got = collect_case(case_dir, window_end=case["window_end"])
        if got is None:
            skipped.append({"case_id": case["case_id"],
                            "site": case["site"],
                            "window_end": case["window_end"],
                            "why": skip_reason(campaign_dir,
                                               case["case_id"])})
            continue
        fss_f = [f["fss30_fcst"] for f in got["frames"]]
        fss_c = [f["fss30_control"] for f in got["frames"]]
        case_rows.append({
            "case_id": case["case_id"],
            "site": case["site"],
            "window_end": case["window_end"],
            "census_median_gates": case["census_median_gates"],
            "frames_graded": len(got["frames"]),
            "verification_state": got["verification_state"],
            "mean_fss_fcst": round(statistics.fmean(fss_f), 4),
            "mean_fss_control": round(statistics.fmean(fss_c), 4),
        })
        collected.append({"case_id": case["case_id"],
                          "frames": got["frames"]})
    if not collected:
        raise ScorecardError("no case produced verified frames; there "
                             "is nothing to aggregate")

    aggregate = aggregate_frames(collected)
    payload = {
        "schema": SCORECARD_SCHEMA,
        "generated": iso(datetime.now(timezone.utc)),
        "config_hash": plan["config_hash"],
        "config": plan["config"],
        "selection_criteria": plan["selection_criteria"],
        "honesty": {
            **plan["honesty"],
            "n_cases_graded": len(collected),
            "n_cases_planned": plan["n_cases"],
            "cases_skipped": skipped,
            "structure_metrics": (
                "present, from the scorer of record"
                if aggregate["structure_metrics_present"] else
                "ABSENT: this tree's tools/da_sweep_score.py does not "
                "emit structure blocks yet (they live on "
                "the corresponding change); the scorecard will carry "
                "them the run after that lane lands"),
        },
        "n_cases": len(collected),
        "per_lead": aggregate["per_lead"],
        "case_rows": case_rows,
        "per_case_frames": collected,
    }
    out_json = campaign_dir / "scorecard.json"
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    out_png = campaign_dir / "scorecard.png"
    try:
        render_scorecard_figure(payload, out_png)
        figure_note = str(out_png)
    except Exception as error:            # aggregation must survive a
        figure_note = f"figure failed: {error}"  # matplotlib-less host
    print(f"scorecard: {len(collected)} case(s) -> {out_json}")
    print(f"  figure: {figure_note}")
    for row in case_rows:
        print(f"  {row['case_id']}  FSS {row['mean_fss_fcst']:.3f} "
              f"(ctrl {row['mean_fss_control']:.3f}) over "
              f"{row['frames_graded']} frame(s)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_scorecard",
        description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    disc = sub.add_parser("discover", help="survey the archive for "
                          "qualifying convective events")
    disc.add_argument("--days", type=float, default=7.0,
                      help="how far back to survey (default 7)")
    disc.add_argument("--start", default=None,
                      help="explicit survey start (ISO-8601 UTC)")
    disc.add_argument("--end", default=None,
                      help="explicit survey end (default: now minus the "
                           "case horizon and the delay floor, so every "
                           "selected case is already closed)")
    disc.add_argument("--sites", default=None,
                      help="comma-separated site ids to restrict the "
                           "survey (default: the whole vendored roster)")
    disc.add_argument("--bucket", default=None)
    disc.add_argument("--out", type=Path, required=True)
    disc.add_argument("--max-cases", type=int, default=10)
    disc.add_argument("--cycles", type=int, default=6)
    disc.add_argument("--cycle-seconds", type=int, default=900)
    disc.add_argument("--free-legs", type=int, default=6)
    disc.add_argument("--delay-floor-seconds", type=float,
                      default=DELAY_FLOOR_SECONDS)
    disc.add_argument("--echo-threshold-dbz", type=float, default=35.0)
    disc.add_argument("--min-echo-gates", type=int,
                      default=MIN_ECHO_GATES)
    disc.add_argument("--prefilter-min-size-bytes", type=int,
                      default=PREFILTER_MIN_SIZE_BYTES)
    disc.add_argument("--prefilter-max-cadence-seconds", type=float,
                      default=PREFILTER_MAX_CADENCE_SECONDS)
    disc.add_argument("--prefilter-min-duration-minutes", type=float,
                      default=PREFILTER_MIN_DURATION_MINUTES)
    disc.add_argument("--census-limit", type=int, default=CENSUS_LIMIT)
    disc.add_argument("--dedup-km", type=float, default=DEDUP_KM)
    disc.add_argument("--dedup-hours", type=float, default=DEDUP_HOURS)
    disc.add_argument("--list-workers", type=int, default=12,
                      help="threads for the payload-free pass-1 listing")

    plan = sub.add_parser("plan", help="freeze a campaign from a "
                          "discovery receipt")
    plan.add_argument("--discovery", type=Path, required=True)
    plan.add_argument("--campaign-dir", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)

    pre = sub.add_parser("preflight", help="CPU-only checks; run at "
                         "staging and again by the launcher")
    pre.add_argument("--plan", type=Path, required=True)

    run = sub.add_parser("run", help="drive the campaign (inside a "
                         "detached scheduled task)")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--campaign-dir", type=Path, required=True)
    run.add_argument("--gate", type=Path, required=True,
                     help="the GPU admission gate script")
    run.add_argument("--wait-for", type=Path, action="append",
                     default=[],
                     help="a predecessor queue's terminal file; the "
                          "campaign does not start until each exists")
    run.add_argument("--poll-seconds", type=int, default=120)
    run.add_argument("--max-wait-hours", type=float, default=12.0)
    run.add_argument("--retry-failed", action="store_true",
                     help="rerun cases that previously FAILED (never "
                          "done silently)")

    score = sub.add_parser("scorecard", help="aggregate a campaign's "
                           "completed cases")
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--campaign-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "discover":
            return discover(args)
        if args.mode == "plan":
            return build_plan(args)
        if args.mode == "preflight":
            return preflight(args)
        if args.mode == "run":
            return run_campaign(args)
        return build_scorecard(args)
    except ScorecardError as error:
        print(f"da_scorecard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
