"""``woof hex add-nest``: a WOOF nest added to a hex run while it runs.

THE BREAKAGE THIS PREVENTS: a regional WOOF nest inside a hex parent could
only be placed at launch.  A user who sees a storm coming at hour 3 of a
48-hour global run had to start a second run, paying a second box start and a
second hex forecast, to get 3 km over it.  This door starts the nest on the
box that is already running the parent, from a frame the parent has already
closed, and from then on is fed by the parent as it writes each frame.

It is the concurrent chain (shape 1) started late, nothing else:

    hex history-ready.jsonl -> woof hex export --follow --follow-from START
      -> python -m woof.local_posting (cycle START) -> woof prep --as-posted
      -> woof sim PREPARED (on the live head)

``--watch`` does the same for every row ``[hex_nests]`` writes to
``<run>/nest-triggers.jsonl`` (shape 3), so a nest a trigger spawns is the same
code as one a person adds.

THE PARENT.  What the add needs (forecast folder, grid, init, start, hours) is
read from the LIVE run: the forecast driver's ``forecast-start.json``, written
when the forecast starts, and the mesh-plan door's receipt, which carries each
stage's argv from the moment the stage starts (``started_stages``).  Until
2026-10-23 it read only the receipt's FINISHED stages, which gain the forecast
row when the forecast ends, so every mid-run add was refused.

THE START FRAME.  ``--start now`` (the default) is the latest frame the parent
has closed; ``--start -2h`` is the latest closed frame at least two hours back
(the nest is spun up by the time it is looked at); an ISO time is that frame.
The nest is cold-started from it through the ``hex-history-netcdf-v1`` profile.

THE CARD.  The fit is a RESERVATION (:mod:`woof.hex.card_ledger`): the engine's
own estimate of the nest (``woof.core.preflight.estimate_experiment``) plus
10 % is claimed atomically against the card's measured free memory less every
other live nest's unallocated claim, so two quick adds cannot both pass and
overfill the card.  A nest that does not fit waits for room, or for the
parent's forecast to end; nothing is refused for size.  With ``--card-queue``
(a box card queue such as a pool runner script) the nest's GPU stage also
takes a pool lease of that size, shared with every other lane on the card.

THE HOLD.  ``--stop-after-cents C --usd-per-hour R`` stops the nest when its
measured cost reaches C: its maps so far are kept.  The measure is the box
time the nest adds, an upper bound: its wall from the click less the time it
sat waiting for the running parent's next frame (the engine's seam-wait log),
so busy time beside hex counts in full and wall after hex's end counts in full.  The site passes C = the hold / the overrun factor, so the
charge (measured cost x 1.25, capped at the hold) is never below the cost.
The receipt's ``measured`` block is what the site prices.

A NEST THAT ENDS BADLY IS REPORTED AS FAILED.  Any non-zero exit of the
nest's forecast that was not the hold's stop marks the receipt's ``measured``
block ``failed`` (``partial`` when some maps were drawn), records the exit
code and the frames written, and makes add-nest itself exit non-zero; the site
charges such a nest its machine time, capped at its quote, never the quote by
default.  (2026-10-24: shapes 1 and 2 died at their last hours on a full
disk, exit 2 and 1, and were reported as finished and charged in full, because
only a failure before the first map counted.)

EVERY STAGE IS TIMED: ``<nest>/add-nest-timeline.jsonl`` carries the click, the
chosen frame, the domain, the card placement, the first exported frame, the
first posted lead, the head, the first nest step and the first nest map, with
wall-clock times, and ``add-nest-receipt.json`` the whole record.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

from .card_ledger import CardLedger
from .errors import MpasPortError
from .export_door import _history_folder, _history_ready_paths, history_valid_time

RECEIPT_NAME = "add-nest-receipt.json"
TIMELINE_NAME = "add-nest-timeline.jsonl"
SCHEMA = "gpuwm-hex.add-nest.v2"
CARD_MARGIN = 0.10
#: The forecast driver's start record (drivers/run_cuda_v841_forecast.py).
START_RECORD_NAME = "forecast-start.json"
#: The real flag for each parent fact (the refusal names these).
PARENT_FLAGS = {"forecast": "--forecast-dir", "mesh": "--mesh", "init": "--init",
                "simulation_start": "--simulation-start", "hours": "--parent-hours"}


class AddNestRefusal(MpasPortError):
    """The add-nest door refuses, and says why."""


def _stamp(moment: _dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d_%H:%M:%S")


def _parse_time(text: str) -> _dt.datetime:
    return _dt.datetime.strptime(text.replace("T", "_").rstrip("Z")[:19], "%Y-%m-%d_%H:%M:%S")


class Timeline:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.marks: dict[str, float] = {}

    def mark(self, event: str, **attrs: Any) -> None:
        now = time.time()
        self.marks.setdefault(event, now)
        row = {"event": event, "t": round(now, 3),
               "utc": _dt.datetime.fromtimestamp(now, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z",
               **attrs}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"ADD-NEST {event} {json.dumps(attrs, sort_keys=True) if attrs else ''}".rstrip(), flush=True)


# --------------------------------------------------------------------------- the parent

def _start_record_folders(root: Path) -> list[Path]:
    folders = [root, root / "forecast"]
    for depth in ("*", "*/*"):
        folders += sorted(p.parent for p in root.glob(f"{depth}/{START_RECORD_NAME}"))
    seen: list[Path] = []
    for folder in folders:
        if folder not in seen:
            seen.append(folder)
    return seen


def _forecast_argv(receipt: dict[str, Any]) -> list[str]:
    stages = receipt.get("stages") or []
    stages = {s.get("stage"): s for s in stages} if isinstance(stages, list) else stages
    argv = (stages.get("forecast") or {}).get("argv")
    if not argv:
        # The live run: mesh-plan records a stage's argv when the stage STARTS.
        argv = ((receipt.get("started_stages") or {}).get("forecast") or {}).get("argv")
    return [str(a) for a in argv or []]


def parent_run(root: Path) -> dict[str, Any]:
    """What the add needs from the running parent: its forecast folder, grid, init, start, hours.

    Read from the live run state (see the module text): the forecast's own
    ``forecast-start.json`` first, then the mesh-plan receipt's forecast argv
    (finished or started).  Flags given on the command line override both.
    """

    root = Path(root)
    found: dict[str, Any] = {"root": str(root), "facts_from": []}
    for folder in _start_record_folders(root):
        record = folder / START_RECORD_NAME
        if not record.is_file():
            continue
        try:
            value = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        found.update(forecast=str(folder), mesh=value.get("grid"), init=value.get("init"),
                     simulation_start=value.get("start_time"), hours=value.get("forecast_hours"),
                     history_minutes=value.get("history_every_minutes"))
        found["facts_from"].append(str(record))
        break
    receipt = root / "global-vr-receipt.json"
    if receipt.is_file():
        try:
            argv = _forecast_argv(json.loads(receipt.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            argv = []

        def flag(name):
            return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None
        if argv:
            facts = {"forecast": flag("--out"), "mesh": flag("--grid"), "init": flag("--init"),
                     "simulation_start": flag("--start-time"),
                     "hours": float(flag("--hours")) if flag("--hours") else None,
                     "history_minutes": float(flag("--history-every-minutes") or 60)}
            for key, value in facts.items():
                if found.get(key) in (None, "") and value not in (None, ""):
                    found[key] = value
            found["facts_from"].append(str(receipt))
    return found


def closed_frames(forecast_dir: Path) -> list[tuple[_dt.datetime, Path]]:
    folder = _history_folder(Path(forecast_dir))
    rows = []
    for path in _history_ready_paths(folder):
        valid = history_valid_time(path)
        if valid is not None:
            rows.append((valid, path))
    return sorted(rows)


def choose_start(frames: Sequence[tuple[_dt.datetime, Path]], start: str,
                 *, wait_seconds: float = 0.0, forecast_dir: Path | None = None,
                 cadence_hours: float = 1.0, on_hour: bool = False) -> _dt.datetime:
    """The frame the nest starts from (see the module text).

    ``on_hour``: only frames on a whole hour may start the nest.  The nest's
    posting numbers its leads in whole hours from an hour cycle
    (``woof.local_posting --cycle YYYY-MM-DDTHH``), so a nest started from a
    15 or 30 min parent's 12:15 frame waits for a 12:00 lead its export skips.
    """

    deadline = time.monotonic() + wait_seconds
    text = start.strip()
    while True:
        if forecast_dir is not None:
            frames = closed_frames(forecast_dir)
        if on_hour:
            frames = [(v, path) for v, path in frames if v.minute == 0 and v.second == 0]
        if frames:
            latest = frames[-1][0]
            if text == "now":
                return latest
            if text.startswith("-") and text.endswith("h"):
                back = latest - _dt.timedelta(hours=float(text[1:-1]))
                earlier = [v for v, _ in frames if v <= back]
                if earlier:
                    return earlier[-1]
                return frames[0][0]
            target = _parse_time(text)
            # The frame at or before the asked time.  A time inside the parent's
            # last closed interval (a trigger's look between two history frames)
            # takes the frame before it now; a later time waits for the parent.
            done = [v for v, _ in frames if v <= target]
            if done and target - latest < _dt.timedelta(hours=cadence_hours):
                return done[-1]
        if time.monotonic() >= deadline:
            raise AddNestRefusal(f"the parent has closed no frame for --start {start!r} yet "
                                 f"({len(frames)} closed)")
        time.sleep(1.0)


# --------------------------------------------------------------------------- the card

def nest_estimate_bytes(config: Path) -> int | None:
    """The engine's own machine-peak envelope for the nest's configuration (pool plus non-pool residency).

    Not ``alloc_estimate_bytes``: that bounds the memory pool only.  In a test run
    (2026-10-23) every 600 km 3 km nest declared its lease from it (1,942 MiB)
    and measured 4,162 MiB on the card, so the pool queue admitted twice what it
    counted.  ``peak_envelope_bytes`` priced the same nest at 4,739 MiB.

    An engine whose estimate has no ``peak_envelope_bytes`` is NOT read for
    its pool-only number (that brought the half-size lease back unseen): the
    estimate is unavailable, said so loudly, and the lease falls back to
    ``--nest-vram-mib``, which is larger than any 3 km nest measured.
    """

    try:
        from woof.core.preflight import estimate_experiment
        from woof.domain_wizard import experiment_from_text
        estimate = estimate_experiment(experiment_from_text(Path(config).read_text(encoding="utf-8"),
                                                            source="add-nest"))
    except Exception as error:  # noqa: BLE001 - recorded; the engine's own admission still runs at sim start
        print(f"ADD-NEST estimate unavailable: {error}", flush=True)
        return None
    peak = getattr(estimate, "peak_envelope_bytes", None)
    if not peak:
        print("ADD-NEST WARNING: this engine's estimate has no peak_envelope_bytes (only the pool-only "
              f"alloc_estimate_bytes={getattr(estimate, 'alloc_estimate_bytes', None)}, about half the nest's "
              "real use); the lease uses --nest-vram-mib instead", flush=True)
        return None
    return int(peak)


def _parent_done(forecast_dir: Path) -> bool:
    from .products_door import FORECAST_RECEIPT

    return (forecast_dir / FORECAST_RECEIPT).is_file() or (forecast_dir / "nest-trigger-receipt.json").is_file()


def place_on_card(ledger: CardLedger, name: str, estimate: int | None, *, forecast_dir: Path,
                  reservation: str | None = None, poll_seconds: float = 5.0,
                  timeline: Timeline | None = None) -> dict[str, Any]:
    """Reserve the nest's memory (atomic); wait for room or for the parent's end.

    A trigger firing arrives with the reservation the loop made at fire time;
    this process adopts it.  A hand add claims its own.
    """

    if reservation:
        if ledger.adopt(reservation, [os.getpid()]):
            return {"placement": "beside", "reservation": reservation, "adopted": True}
    name = reservation or name
    if estimate is None:
        return {"placement": "beside", "reservation": None, "fits": None, "why": "nest estimate unavailable"}
    verdict = ledger.reserve(name, estimate, [os.getpid()])
    if verdict["fits"] is not False:
        return {"placement": "beside", "reservation": name, **verdict}
    if timeline is not None:
        timeline.mark("card_queued", **{k: verdict[k] for k in ("need_bytes", "available_bytes")})
    while True:
        time.sleep(poll_seconds)
        done = _parent_done(forecast_dir)
        others = [n for n in ledger.live_names() if n != name]
        verdict = ledger.reserve(name, estimate, [os.getpid()], force=done and not others)
        if verdict["fits"] is not False or (done and not others):
            return {"placement": "after-parent" if done else "beside-after-wait", "reservation": name, **verdict}


# --------------------------------------------------------------------------- the chain

def _engine(name: str) -> str:
    beside = Path(sys.executable).with_name(name)
    return str(beside) if beside.exists() else (shutil.which(name) or name)


def _spawn(argv: Sequence[str], log: Path, *, env: dict[str, str] | None = None) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = open(log, "ab")  # noqa: SIM115 - owned by the child for its life
    stream.write(("$ " + " ".join(shlex.quote(str(a)) for a in argv) + "\n").encode())
    stream.flush()
    return subprocess.Popen([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def _run(argv: Sequence[str], log: Path) -> None:
    proc = _spawn(argv, log)
    if proc.wait() != 0:
        raise AddNestRefusal(f"{Path(str(argv[0])).name} {argv[1] if len(argv) > 1 else ''} failed "
                             f"(exit {proc.returncode}); see {log}")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()


def _first_png(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    for path in folder.rglob("*.png"):
        return path
    return None


#: Bytes a nest writes per column per forecast hour (history frames, pictures,
#: logs; no restart checkpoints, see nest_config), measured on test run 2
#: (2026-10-24): a 600 km 3 km nest (40,000 columns) wrote 2.3 GB of frames and
#: pictures in 9 h, plus about 0.7 GB of prepared head and exported frames.
OUTPUT_BYTES_PER_COLUMN_HOUR = 8000.0
OUTPUT_FIXED_BYTES = 1.0e9
DISK_MARGIN = 1.5


def disk_check(out: Path, *, size_km: float, dx_km: float, hours: float) -> dict[str, Any]:
    """Refuse a nest whose output does not fit on the disk it writes to.

    THE BREAKAGE THIS PREVENTS: in a test run (2026-10-24) the disk filled while
    shapes 1 and 2 ran; both died at their last hours (ENOSPC on the
    checkpoint, the report and the log) after the user had watched hours of
    maps, and the hex products stage failed with them.  A nest that cannot
    write its run is refused at the click, before any box time is spent.
    """

    columns = (float(size_km) / float(dx_km)) ** 2
    need = int((columns * float(hours) * OUTPUT_BYTES_PER_COLUMN_HOUR + OUTPUT_FIXED_BYTES) * DISK_MARGIN)
    free = shutil.disk_usage(Path(out)).free
    row = {"need_bytes": need, "free_bytes": int(free)}
    if free < need:
        raise AddNestRefusal(f"the disk under {out} has {free / 1e9:.1f} GB free; this nest writes about "
                             f"{need / DISK_MARGIN / 1e9:.1f} GB ({need / 1e9:.1f} GB with margin), so it would "
                             "fail at its last hours on a full disk")
    return row


#: The suite a convection-permitting nest runs when the add names none: aerosol-aware Thompson, MYNN PBL and
#: surface layer, RUC land surface, legacy RRTMG, no cumulus (the engine's HRRR-shape route default,
#: ``woof.hrrr_route_inputs.ROUTE_DEFAULT_PHYSICS_PROFILE``, spelled out because this door only shells out to
#: the engine).
#:
#: THE BREAKAGE THIS PREVENTS (Hex with Regional, 2025-05-19): with no
#: profile, ``woof domain --source hex-global`` hands the nest the gfs/era5-class default, Morrison two-moment
#: with rimed ice as hail.  Its REFL_10CM (WRF's ``refl10cm_hm``) scatters Morrison's large snow mass as wet snow
#: below the melting level, and every 3 km nest showed a 55-65 dBZ shield over its trailing stratiform where MRMS
#: had 25-40 dBZ.  In the Kansas trigger nest at +6 h: 42,822 km2 at or above 50 dBZ (MRMS 1,107) and 2,304 km2
#: at or above 60 (MRMS 9), over rain of 1-8 mm/h.  The same nest from the same hex frames on this suite:
#: 4,302 and 405 km2, and its 6 h rain went from 1.7 to 1.4 times MRMS.
CONVECTION_PERMITTING_PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"
#: Below this spacing a nest resolves deep convection and runs no cumulus scheme (the engine's own
#: convection-permitting line).  A coarser nest keeps the engine's default for its source, which carries one.
CONVECTION_PERMITTING_BELOW_KM = 4.0


def nest_physics_profile(dx_km: float, requested: str | None = None) -> str | None:
    """The ``--physics-profile`` the nest's domain is emitted with (``None``: the engine's source default)."""

    if requested:
        return requested
    if float(dx_km) < CONVECTION_PERMITTING_BELOW_KM:
        return CONVECTION_PERMITTING_PROFILE
    return None


def needs_aerosol_climatology(config: Path) -> bool:
    """Whether the emitted nest runs aerosol-aware Thompson (``mp_physics = 28``).

    A hex frame carries no QNWFA/QNIFA, so mp=28 on a nest reads WRF's monthly WIF aerosol climatology, and the
    engine's prep refuses the nest without it (2026-10-24: "mp_physics=28 on a domain with external
    lateral boundaries (specified) needs WRF's monthly WIF aerosol climatology").
    """

    import re

    return re.search(r"(?m)^mp_physics\s*=\s*28\s*$", Path(config).read_text(encoding="utf-8")) is not None


def nest_config(woof: str, out: Path, name: str, *, lat: float, lon: float, dx_km: float, size_km: float,
                hours: int, cycle: str, cadence_h: float, geog: str | None,
                physics_profile: str | None = None) -> Path:
    config = out / f"{name}.toml"
    profile = nest_physics_profile(dx_km, physics_profile)
    _run([woof, "domain", "--source", "hex-global", "--cadence", f"{cadence_h:g}",
          "--point", f"{lat},{lon}", "--root-dx", f"{dx_km:g}",
          "--point-extent-km", f"{size_km:g}", "--hours", str(hours), "--cycle", cycle,
          *(["--physics-profile", profile] if profile else []),
          "--name", name, "--out", config], out / "logs" / "domain.log")
    no_restarts(config)
    if geog:
        _run([woof, "fetch-geog", "--config", config, "--root", geog], out / "logs" / "geog.log")
        if needs_aerosol_climatology(config):
            # Verified by SHA-256 and kept under ~/.woof/wif; a staged copy is checked, not fetched again.
            _run([woof, "fetch-tables", "--wif-only"], out / "logs" / "tables.log")
    return config


def no_restarts(config: Path) -> None:
    """A nest writes no restart checkpoints: ``restart_interval_s = 0`` (the engine's "0 disables restart writing").

    THE BREAKAGE THIS PREVENTS: the domain wizard's default writes a 323 MB
    checkpoint every hour and keeps them all; nothing restarts a nest (a reclaim
    runs it again in full, which is what its hold prices), so an 18 h 600 km nest
    spent 5.8 GB of disk on checkpoints no one reads.  In a test run (2026-10-24)
    shape 1 died writing its 03Z checkpoint on a full disk.
    """

    import re

    text = Path(config).read_text(encoding="utf-8")
    new, count = re.subn(r"(?m)^restart_interval_s\s*=.*$", "restart_interval_s = 0.0", text)
    if count == 0:
        new, count = re.subn(r"(?m)^\[experiment\]\s*$", "[experiment]\nrestart_interval_s = 0.0", text, count=1)
    if count == 0:
        raise AddNestRefusal(f"{config} has no [experiment] table to switch the nest's restart checkpoints off in")
    Path(config).write_text(new, encoding="utf-8")


def hold_budget_seconds(stop_after_cents: float | None, usd_per_hour: float | None) -> float | None:
    """Wall seconds of box the nest may use before its hold stops it."""

    if stop_after_cents is None:
        return None
    if not usd_per_hour or usd_per_hour <= 0:
        raise AddNestRefusal("--stop-after-cents needs --usd-per-hour, the box's own rate")
    return float(stop_after_cents) / (float(usd_per_hour) * 100.0 / 3600.0)


#: The maps a nest draws by default: the WOOF worker's "met" product set (contracts/product-sets.json), not
#: ``all``.  THE BREAKAGE THIS PREVENTS: on woof 2.9.5 and 2.9.6, ``--render-products all`` includes the
#: hub-height maps, which refuse a dot in the domain folder name, so any nest whose spacing is not a whole
#: number of km (1.5 or 2.5 km) died at its end-of-run render with exit 2 (fixed in 2.9.7 by
#: fix/go-render-exit2).  ``add-nest --render-products all`` still asks for everything.
NEST_PRODUCTS = ",".join((
    "2m_temperature", "2m_temperature_10m_winds", "2m_dewpoint", "2m_dewpoint_10m_winds", "2m_relative_humidity",
    "2m_relative_humidity_10m_winds", "10m_wind_speed_and_direction", "mslp_10m_winds", "terrain_height",
    "850mb_height_winds", "850mb_temperature_height_winds", "850mb_dewpoint_height_winds", "850mb_rh_height_winds",
    "700mb_height_winds", "700mb_temperature_height_winds", "700mb_dewpoint_height_winds", "700mb_rh_height_winds",
    "500mb_height_winds", "500mb_temperature_height_winds", "500mb_rh_height_winds", "300mb_height_winds",
    "300mb_temperature_height_winds", "300mb_rh_height_winds", "250mb_height_winds", "250mb_temperature_height_winds",
    "sbcape", "mlcape", "mucape", "sbcin", "mlcin", "mucin", "srh_0_1km", "srh_0_3km", "bulk_shear_0_1km",
    "bulk_shear_0_6km", "stp_fixed", "ehi_0_1km", "composite_reflectivity", "1km_reflectivity", "qpf_1h",
    "total_qpf", "precipitable_water", "isotherm_height_0c", "low_cloud_cover", "middle_cloud_cover",
    "high_cloud_cover", "cloud_water_column_max"))

#: What the nest's maps call its grid in the headline: ``nest 3 km``, not the ``d01 3 km`` it is in its own
#: one-domain run.  THE BREAKAGE THIS PREVENTS (WOOF Hex with Regional, 2026-10-28, canary run_9a0796b2e9c8e251):
#: every nest map was headlined ``d01 3 km`` while the site files the nest as d05.  The engine (woof >= 2.9.10,
#: ``woof.runtime.GRID_NAME_ENV``) stamps it on d01's frames as ``GPUWM_GRID_NAME`` and ``rw_wrfbatch`` prints it
#: in the headline only, so the domain folder stays ``d01-3km`` (the WOOF worker's ``nest_view`` matches on it).
#: An older engine ignores the variable and draws ``d01`` as before.
NEST_GRID_NAME_ENV = "GPUWM_GRID_NAME"
NEST_GRID_NAME = "nest"


def nest_sim_env() -> dict[str, str]:
    """The environment of the nest's ``woof sim``: this process's, naming the nest's grid for its maps."""

    return {**os.environ, NEST_GRID_NAME_ENV: NEST_GRID_NAME}


#: The engine's seam-wait log in the nest's sim folder (woof.ingest.boundary_stream WAIT_LOG_NAME).
WAIT_LOG_NAME = "waits.jsonl"
PROGRESS_NAME = "add-nest-progress.json"


def current_run_folder(nest_out: Path) -> Path | None:
    """The sim's current run folder: the newest ``run-<stamp>``, or the one ``latest-run.txt`` names if newer.

    ``latest-run.txt`` is written once the run's first picture is drawn, so
    before that the newest stamped folder is the run in progress (the stamps
    sort in time order).
    """

    runs = sorted(p for p in Path(nest_out).glob("run-*") if p.is_dir())
    named = None
    try:
        text = (Path(nest_out) / "latest-run.txt").read_text(encoding="utf-8").strip()
        if text:
            named = Path(text) if Path(text).is_absolute() else Path(nest_out) / text
    except OSError:
        pass
    choices = [p for p in (named, runs[-1] if runs else None) if p is not None and p.is_dir()]
    return max(choices, key=lambda p: p.name) if choices else None


def nest_wait_seconds(nest_out: Path, *, now: float | None = None) -> float:
    """Seconds the nest's forecast has spent WAITING on the running parent (its seam waits).

    While the nest waits for hex to close its next frame, the box is running
    for hex anyway: that time is not box time the nest adds.  Read from the
    engine's own wait log (``<sim outdir>/waits.jsonl``: *_wait_started /
    *_wait_finished); an open wait counts up to ``now``.
    """

    now = time.time() if now is None else now
    # `woof sim --outdir OUT` writes into OUT/run-<stamp>/ (OUT/latest-run.txt names it): the wait log is
    # there, not in OUT.  Reading only OUT/waits.jsonl found nothing on the whole test run (2026-10-23), so
    # every nest was charged its whole wall, seam waits on the parent included.  Only the CURRENT run's log
    # is read: summing every run folder's log subtracted an earlier run's waits too (an earlier run's
    # unfinished wait counted up to now), so a retry undercounted the box time and undercharged.
    folder = current_run_folder(Path(nest_out))
    candidate = (folder if folder is not None else Path(nest_out)) / WAIT_LOG_NAME
    try:
        text = candidate.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if not text.strip():
        return 0.0
    total, opened = 0.0, None
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        event = str(row.get("event", ""))
        stamp = float(row.get("emitted_unix_ms", 0)) / 1000.0
        if event.endswith("_wait_started"):
            opened = stamp
        elif event.endswith("_wait_finished"):
            seconds = row.get("seconds", row.get("waited_seconds"))
            if seconds is None and opened is not None:
                seconds = stamp - opened
            total += max(0.0, float(seconds or 0.0))
            opened = None
    if opened is not None:
        total += max(0.0, now - opened)
    return total


def box_seconds(clicked: float, now: float, nest_out: Path) -> float:
    """The box time the nest adds, an upper bound: its wall from the click less its waits on the parent.

    Busy time while the parent runs is counted in full (no credit for
    sharing the card); wall after the parent's end is counted in full (the
    box stays up for the nest, busy or not: a fast globe ends first).  Only
    the time the nest sat waiting for the parent's next frame is left out.
    """

    return max(0.0, (now - clicked) - nest_wait_seconds(nest_out, now=now))


def watch_nest(procs: dict[str, subprocess.Popen], *, clicked: float, budget_seconds: float | None,
               timeline: Timeline, nest_out: Path | None = None, progress: Path | None = None,
               poll_seconds: float = 1.0, clock=time.time) -> bool:
    """Wait for the nest's sim; stop the whole chain at the hold.  True when stopped at the hold."""

    sim = procs["sim"]
    last_progress = 0.0
    while sim.poll() is None:
        now = clock()
        used = box_seconds(clicked, now, nest_out) if nest_out is not None else now - clicked
        if progress is not None and now - last_progress >= 10.0:
            progress.write_text(json.dumps({"box_seconds": round(used, 3), "t": round(now, 3)}) + "\n",
                                encoding="utf-8")
            last_progress = now
        if budget_seconds is not None and used >= budget_seconds:
            timeline.mark("stopped_at_hold", budget_seconds=round(budget_seconds, 1), box_seconds=round(used, 1))
            for proc in procs.values():
                _stop(proc)
            return True
        time.sleep(poll_seconds)
    return False


def add_nest(arguments: argparse.Namespace, *, lat: float | None = None, lon: float | None = None,
             start: str | None = None, name: str | None = None, firing: dict | None = None) -> dict[str, Any]:
    clicked = time.time()
    lat = arguments.lat if lat is None else lat
    lon = arguments.lon if lon is None else lon
    if lat is None or lon is None:
        raise AddNestRefusal("add-nest needs --lat and --lon (or --watch for trigger firings)")
    budget_seconds = hold_budget_seconds(arguments.stop_after_cents, arguments.usd_per_hour)
    # A nest placed at launch (shape 1) is given with the hex run, before its mesh, static and init
    # stages have built the forecast: wait for the forecast's start record, as --watch does.
    deadline = time.monotonic() + arguments.wait_minutes * 60
    while True:
        parent = parent_run(Path(arguments.run))
        for key, option in (("forecast", "forecast_dir"), ("mesh", "mesh"), ("init", "init"),
                            ("simulation_start", "simulation_start"), ("hours", "parent_hours")):
            given = getattr(arguments, option, None)
            if given is not None:
                parent[key] = str(given) if isinstance(given, Path) else given
        missing = [k for k in PARENT_FLAGS if not parent.get(k)]
        if not missing or time.monotonic() >= deadline:
            break
        time.sleep(2.0)
    if missing:
        raise AddNestRefusal(f"the parent run at {arguments.run} does not say its {missing} yet (no "
                             f"{START_RECORD_NAME} and no forecast argv in its receipt); give "
                             + " ".join(PARENT_FLAGS[m] for m in missing))
    name = name or nest_name(arguments, lat, lon)
    out = Path(arguments.out or (Path(arguments.run) / "nests")) / name
    if out.exists() and any(out.iterdir()):
        raise AddNestRefusal(f"{out} exists and is not empty; a nest folder is never reused")
    out.mkdir(parents=True, exist_ok=True)
    timeline = Timeline(out / TIMELINE_NAME)
    timeline.mark("click", lat=lat, lon=lon, t_click=round(clicked, 3),
                  **({"firing": firing["index"]} if firing else {}),
                  **({"budget_seconds": round(budget_seconds, 1)} if budget_seconds is not None else {}))
    forecast_dir = Path(parent["forecast"])
    sim_start = _parse_time(parent["simulation_start"])
    hex_end = sim_start + _dt.timedelta(hours=float(parent["hours"]))
    cadence_h = float(arguments.cadence_hours)
    # The nest's boundary leads sit on a grid of --cadence-hours from its start
    # frame; the parent writes a frame every history_minutes.  Before 2.9.7 the
    # export took EVERY parent frame and stopped after len(leads), so a parent
    # writing every 15/30 min (WOOF's map shapes) fed a 6 h nest frames covering
    # 0-3 h, and posting waited for its +4 h lead until source_behind (release
    # review 30, finding 2).  Leads are whole hours (local_posting) and must
    # land on parent frames.
    history_min = float(parent.get("history_minutes") or 60.0)
    cadence_min = cadence_h * 60.0
    if cadence_h != int(cadence_h) or cadence_h < 1:
        raise AddNestRefusal(f"--cadence-hours {cadence_h:g}: the nest's boundary leads are whole hours "
                             "(woof.local_posting --leads), so give a whole number of hours")
    if history_min <= 0 or abs(cadence_min / history_min - round(cadence_min / history_min)) > 1e-9:
        raise AddNestRefusal(f"--cadence-hours {cadence_h:g} is not a whole multiple of the parent's "
                             f"{history_min:g} min history spacing: the nest's leads would fall between "
                             "parent frames that are never written")
    start_valid = choose_start([], start or arguments.start, wait_seconds=arguments.wait_minutes * 60,
                               forecast_dir=forecast_dir, cadence_hours=cadence_h,
                               on_hour=history_min % 60 != 0)
    hours = float(arguments.hours) if arguments.hours else (hex_end - start_valid).total_seconds() / 3600
    hours = min(hours, (hex_end - start_valid).total_seconds() / 3600)
    whole = int(hours // cadence_h * cadence_h)
    if whole < cadence_h:
        raise AddNestRefusal(f"the parent ends at {hex_end}; a nest from {start_valid} has under one "
                             f"{cadence_h:g} h boundary interval")
    leads = list(range(0, whole + 1, int(cadence_h)))
    timeline.mark("start_frame", valid=_stamp(start_valid), hours=whole,
                  parent_frames_closed=len(closed_frames(forecast_dir)))
    timeline.mark("disk_checked", **disk_check(out, size_km=arguments.size_km, dx_km=arguments.dx_km, hours=whole))
    # The parent's trigger loop skips storms this nest covers (nest_triggers LIVE_NESTS_NAME).
    from .nest_triggers import note_live_nest
    note_live_nest(forecast_dir, {"event": "started", "name": name, "lat": lat, "lon": lon,
                                  "size_km": arguments.size_km, "start_valid": _stamp(start_valid),
                                  "hours": whole, "firing": (firing or {}).get("index"),
                                  "t": round(time.time(), 3)})
    woof = _engine("woof")
    cycle = start_valid.strftime("%Y-%m-%dT%H")
    config = nest_config(woof, out, name, lat=lat, lon=lon, dx_km=arguments.dx_km, size_km=arguments.size_km,
                         hours=whole, cycle=cycle, cadence_h=cadence_h, geog=arguments.geog,
                         physics_profile=getattr(arguments, "physics_profile", None))
    namelist = config.with_suffix(".namelist.wps")
    timeline.mark("domain_done")

    estimate = nest_estimate_bytes(config)
    ledger = CardLedger(arguments.card_ledger, device=arguments.device)
    reservation = (firing or {}).get("card_reservation") or arguments.card_reservation
    if arguments.card_queue:
        # The box's card queue admits the GPU stage atomically against every
        # lane on the card (pool lease of the nest's own size).
        card = {"placement": "pool-lease", "reservation": None}
        if reservation:
            ledger.release(reservation)  # the pool lease is the claim now
    else:
        card = place_on_card(ledger, f"add-{name}-{os.getpid()}", estimate, forecast_dir=forecast_dir,
                             reservation=reservation, timeline=timeline)
    timeline.mark("card_fit", estimate_bytes=estimate, **{k: v for k, v in card.items() if k != "name"})

    window = f"region-latlon:{lat},{lon},{arguments.size_km * 1.5:g},{arguments.size_km * 1.5:g}"
    hexcli = [sys.executable, "-m", "woof.hex.cli"]
    export = _spawn([*hexcli, "export", forecast_dir, "--follow", "--follow-from", _stamp(start_valid),
                     "--follow-frames", str(len(leads)), "--follow-every-minutes", f"{cadence_min:g}",
                     "--mesh", parent["mesh"], "--init", parent["init"],
                     "--simulation-start", _stamp(sim_start), "--window", window, "--field-set", "full",
                     "--view", "nest", "--out", out / "export"], out / "logs" / "export.log")
    posting_argv = [sys.executable, "-m", "woof.local_posting", "--source", "hex-global", "--cycle", cycle,
                    "--leads", f"0-{leads[-1]}" if cadence_h == 1 else ",".join(map(str, leads)),
                    "--out", out / "fetch", "--ready-list", out / "export" / "export-ready.jsonl",
                    "--supplement-role", "hex_in_band_surface"]
    posting = _spawn([*posting_argv, "--producer-pid", str(export.pid)], out / "logs" / "posting.log")
    procs: dict[str, subprocess.Popen] = {"export": export, "posting": posting}
    prep_flags = subprocess.run([str(a) for a in [*posting_argv, "--print-prep-arguments"]],
                                capture_output=True, text=True, check=True).stdout.split()
    while not (out / "fetch" / "posting" / "schedule.json").is_file():
        time.sleep(0.5)
    prepared = out / "prepared"
    prep = _spawn([woof, "prep", "--source", "hex-global", *prep_flags, "--as-posted", out / "fetch" / "posting",
                   "--author-input-manifest", out / "fetch" / "inputs.json", "--experiment-config", config,
                   "--wps-namelist", namelist, *(["--geog-root", arguments.geog] if arguments.geog else []),
                   "--output-root", prepared], out / "logs" / "prep.log")
    procs["prep"] = prep
    seen_export = seen_post = False
    while not (prepared / "boundary-stream" / "head.json").is_file():
        if not seen_export and (out / "export" / "export-ready.jsonl").is_file():
            timeline.mark("first_frame_exported")
            seen_export = True
        if not seen_post and (out / "fetch" / "posting" / "f000.json").is_file():
            timeline.mark("first_lead_posted")
            seen_post = True
        if prep.poll() is not None:
            raise AddNestRefusal(f"prep ended before its head (exit {prep.returncode}); see {out / 'logs' / 'prep.log'}")
        if budget_seconds is not None and time.time() - clicked >= budget_seconds:
            # The hold is spent before the nest could start (a slow parent or
            # prep): stop here, charge what was measured, start no sim.
            timeline.mark("stopped_at_hold", budget_seconds=round(budget_seconds, 1), before="head")
            for proc in procs.values():
                _stop(proc)
            receipt = {"schema": SCHEMA, "parent": parent, "lat": lat, "lon": lon, "out": str(out),
                       "firing": firing, "card": card, "estimate_bytes": estimate,
                       "returncodes": {k: p.returncode for k, p in procs.items()}}
            _account(receipt, out=out, clicked=clicked, rate=arguments.usd_per_hour, stopped=True,
                     ledger=ledger, card=card, firing=firing, forecast_dir=forecast_dir, timeline=timeline)
            return receipt
        time.sleep(0.5)
    timeline.mark("head_ready")
    # `woof sim PREPARED` (there is no `sim prepared` subcommand: shape 1 and the test run both hit it).
    sim_argv = [woof, "sim", prepared, "--experiment-config", config, "--wps-namelist", namelist,
                "--outdir", out / "nest", "--render-products", arguments.render_products]
    from woof.output_formats import argument_values
    sim_argv += argument_values(arguments, relative=name)
    if arguments.card_queue:
        mib = math.ceil((estimate or arguments.nest_vram_mib * 2**20) * (1 + CARD_MARGIN) / 2**20)
        sim_argv = [arguments.card_queue, f"{arguments.lane_prefix}{name}", "--vram-mib", str(mib),
                    "--wait", str(int(arguments.queue_wait_seconds)), *sim_argv]
    sim = _spawn(sim_argv, out / "logs" / "sim.log", env=nest_sim_env())
    procs["sim"] = sim
    if card.get("reservation"):
        ledger.adopt(card["reservation"], [p.pid for p in procs.values()])
    timeline.mark("sim_started", pid=sim.pid)
    stopped = False
    while _first_png(out / "nest") is None and sim.poll() is None:
        if budget_seconds is not None and time.time() - clicked >= budget_seconds:
            break
        time.sleep(1.0)
    first = _first_png(out / "nest")
    timeline.mark("first_nest_map", png=str(first) if first else None,
                  click_to_map_s=round(time.time() - clicked, 1) if first else None)
    receipt = {
        "schema": SCHEMA, "parent": parent, "lat": lat, "lon": lon, "dx_km": arguments.dx_km,
        "size_km": arguments.size_km, "start_valid": _stamp(start_valid), "hours": whole,
        "card": card, "estimate_bytes": estimate, "firing": firing,
        "pids": {k: p.pid for k, p in procs.items()}, "out": str(out),
        "hold": {"stop_after_cents": arguments.stop_after_cents, "usd_per_hour": arguments.usd_per_hour,
                 "budget_seconds": budget_seconds},
    }
    _write_receipt(out, receipt)
    if arguments.wait_for_nest or budget_seconds is not None:
        # A nest with a hold is watched to its end: the stop is this process's job.
        stopped = watch_nest(procs, clicked=clicked, budget_seconds=budget_seconds, timeline=timeline,
                             nest_out=out / "nest", progress=out / PROGRESS_NAME)
        rc = sim.wait()
        timeline.mark("nest_done", returncode=rc, stopped_at_hold=stopped)
        for what in ("export", "posting", "prep"):
            proc = procs[what]
            if proc.poll() is None and (stopped or rc != 0):
                _stop(proc)
            timeline.mark(f"{what}_done", returncode=proc.wait())
        receipt["returncodes"] = {k: p.returncode for k, p in procs.items()}
        frames = _frames_written(out / "nest")
        receipt["frames_written"] = frames
        receipt["frames_planned"] = len(leads)
        failed = rc != 0 and not stopped
        if failed:
            receipt["failed"] = True
        _account(receipt, out=out, clicked=clicked, rate=arguments.usd_per_hour, stopped=stopped,
                 ledger=ledger, card=card, firing=firing, forecast_dir=forecast_dir, timeline=timeline,
                 failed=failed, partial=failed and _first_png(out / "nest") is not None, returncode=rc)
    return receipt


def _frames_written(nest_out: Path) -> int:
    folder = current_run_folder(nest_out)
    if folder is None:
        return 0
    return sum(1 for p in folder.rglob("wrfout_d01_*") if not p.name.endswith((".json", ".tmp")))


def _account(receipt: dict[str, Any], *, out: Path, clicked: float, rate: float | None, stopped: bool,
             ledger: CardLedger, card: dict[str, Any], firing: dict | None, forecast_dir: Path,
             timeline: Timeline, failed: bool = False, partial: bool = False,
             returncode: int | None = None) -> None:
    """The nest has ended: measured block, card claim released, the firing's spend recorded.

    A nest whose forecast exited non-zero (not at its hold) is marked
    ``failed``, and ``partial`` when it rendered maps first: charged its machine
    time, capped at its quote (the module text).
    """

    from .nest_triggers import note_live_nest
    note_live_nest(forecast_dir, {"event": "ended", "name": Path(out).name, "t": round(time.time(), 3)})
    receipt["timeline"] = timeline.marks
    receipt["measured"] = measured_block(clicked, time.time(), rate, stopped, nest_out=out / "nest")
    if failed:
        receipt["measured"]["failed"] = True
        receipt["measured"]["partial"] = bool(partial)
        if returncode is not None:
            receipt["measured"]["returncode"] = int(returncode)
    if card.get("reservation"):
        ledger.release(card["reservation"])
    if firing is not None:
        _record_spent(forecast_dir, firing, receipt["measured"])
    _write_receipt(out, receipt)


def measured_block(clicked: float, ended: float, usd_per_hour: float | None, stopped: bool,
                   nest_out: Path | None = None) -> dict[str, Any]:
    """What the site prices: the box seconds the nest added (see :func:`box_seconds`), not GPU-busy time.

    GPU-busy time is not an upper bound: a nest that runs past the parent's
    end keeps the box billed for its whole wall there, busy or not.
    """

    wall = max(0.0, ended - clicked)
    waited = nest_wait_seconds(nest_out, now=ended) if nest_out is not None else 0.0
    billable = max(0.0, wall - waited)
    row: dict[str, Any] = {"wall_seconds": round(wall, 3), "parent_wait_seconds": round(waited, 3),
                           "box_seconds": round(billable, 3), "stopped_at_hold": bool(stopped)}
    if usd_per_hour:
        row["usd_per_hour"] = float(usd_per_hour)
        row["cost_cents"] = round(billable * float(usd_per_hour) * 100.0 / 3600.0, 3)
    return row


def _record_spent(forecast_dir: Path, firing: dict[str, Any], measured: dict[str, Any]) -> None:
    from . import nest_budget

    hold = nest_budget.read_hold(forecast_dir)
    if hold is None or "cost_cents" not in measured:
        return
    quote = float(hold.get("per_firing_quote_cents") or hold["per_firing_cents"] / 2.0)
    if measured.get("failed"):
        # A failed or partial nest is charged its machine time, capped at its quote (the site's rule).
        nest_budget.write_spent(forecast_dir, int(firing["index"]), min(math.ceil(measured["cost_cents"]),
                                                                        math.ceil(quote)))
        return
    budget = float(firing.get("budget_cents") or hold["per_firing_cents"])
    committed = nest_budget.committed_cents(
        measured["cost_cents"], budget_cents=budget,
        quote_cents=quote, overrun_factor=float(hold["overrun_factor"]))
    nest_budget.write_spent(forecast_dir, int(firing["index"]), committed)


def _write_receipt(out: Path, receipt: dict[str, Any]) -> None:
    (out / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                                    encoding="utf-8")


# --------------------------------------------------------------------------- watch

def _area_centre(area: dict[str, Any]) -> tuple[float, float]:
    if area.get("kind") == "box":
        return (area["south"] + area["north"]) / 2.0, (area["west"] + area["east"]) / 2.0
    if area.get("kind") == "circle":
        return area["lat"], area["lon"]
    if area.get("kind") == "polygon":
        corners = area["corners"]
        return sum(c[0] for c in corners) / len(corners), sum(c[1] for c in corners) / len(corners)
    return 0.0, 0.0


def price_trigger_nest(arguments: argparse.Namespace, forecast_dir: Path, parent: dict[str, Any]) -> int | None:
    """The table's nest's memory, once, before the first firing (the loop's card-fit claim reads it).

    The size of a nest does not depend on where it lands, so one engine
    estimate at the area's centre serves every firing.
    """

    from .nest_budget import FIRE_BUDGET_NAME
    from .nest_triggers import LOOKS_NAME

    deadline = time.monotonic() + arguments.wait_minutes * 60
    table = None
    while table is None and time.monotonic() < deadline:
        try:
            first = (forecast_dir / LOOKS_NAME).read_text(encoding="utf-8").splitlines()[0]
            table = json.loads(first).get("table")
        except (OSError, ValueError, IndexError):
            time.sleep(2.0)
    if table is None:
        return None
    nest = table["nest"]
    lat, lon = _area_centre(table.get("area") or {})
    out = Path(arguments.out or Path(arguments.run) / "nests") / "_pricing"
    out.mkdir(parents=True, exist_ok=True)
    start = _parse_time(str(parent["simulation_start"]))
    try:
        config = nest_config(_engine("woof"), out, "trigger-nest", lat=lat, lon=lon, dx_km=float(nest["dx_km"]),
                             size_km=float(nest["size_km"]), hours=max(1, int(float(nest["hours"]))),
                             cycle=start.strftime("%Y-%m-%dT%H"), cadence_h=float(arguments.cadence_hours),
                             geog=None, physics_profile=getattr(arguments, "physics_profile", None))
    except AddNestRefusal as error:
        print(f"WATCH nest pricing failed: {error}", flush=True)
        return None
    nbytes = nest_estimate_bytes(config)
    if nbytes is None:
        # No engine envelope: claim the conservative lease size, so firings are
        # not held for ever on "the nest's size is not measured yet".
        nbytes = int(arguments.nest_vram_mib * 2**20)
        print(f"WATCH nest size from --nest-vram-mib ({arguments.nest_vram_mib:g} MiB)", flush=True)
    if nbytes is not None:
        target = forecast_dir / FIRE_BUDGET_NAME
        target.write_text(json.dumps({"nest_bytes": nbytes, "nest": nest}, sort_keys=True) + "\n", encoding="utf-8")
    return nbytes


def watch(arguments: argparse.Namespace) -> int:
    """Add a nest for every firing ``[hex_nests]`` writes, until the parent's forecast ends."""

    from .nest_triggers import read_firings

    deadline = time.monotonic() + arguments.wait_minutes * 60
    while True:
        parent = parent_run(Path(arguments.run))
        forecast_dir = arguments.forecast_dir or parent.get("forecast")
        if forecast_dir or time.monotonic() >= deadline:
            break
        time.sleep(2.0)  # the forecast has not written its start record yet
    forecast_dir = Path(forecast_dir or arguments.run)
    if arguments.simulation_start:
        parent["simulation_start"] = arguments.simulation_start
    if parent.get("simulation_start"):
        nbytes = price_trigger_nest(arguments, forecast_dir, parent)
        print(f"WATCH nest priced at {nbytes} bytes", flush=True)
    done: set[int] = set()
    children: list[subprocess.Popen] = []
    quiet_since = time.monotonic()
    while True:
        firings = [f for f in read_firings(forecast_dir) if f["index"] not in done]
        for firing in firings:
            done.add(firing["index"])
            argv = [sys.executable, "-m", "woof.hex.cli", "add-nest", arguments.run,
                    "--lat", str(firing["lat"]), "--lon", str(firing["lon"]),
                    "--dx-km", str(firing["nest"]["dx_km"]), "--size-km", str(firing["nest"]["size_km"]),
                    "--hours", str(firing["nest"]["hours"]), "--start", firing["nest_start_valid"],
                    "--name", f"trigger-{firing['index']}", "--wait-for-nest",
                    "--wait-minutes", str(arguments.wait_minutes), "--firing-json", json.dumps(firing)]
            from woof.output_formats import argument_values
            argv += argument_values(arguments)
            if firing.get("stop_after_cents") is not None:
                rate = firing.get("usd_per_hour") or arguments.usd_per_hour
                argv += ["--stop-after-cents", str(firing["stop_after_cents"]), "--usd-per-hour", str(rate)]
            elif arguments.usd_per_hour:
                argv += ["--usd-per-hour", str(arguments.usd_per_hour)]
            for option in ("out", "geog", "mesh", "init", "simulation_start", "parent_hours", "card_queue",
                           "card_ledger", "lane_prefix", "forecast_dir", "render_products", "physics_profile"):
                value = getattr(arguments, option, None)
                if value is not None:
                    argv += [f"--{option.replace('_', '-')}", str(value)]
            print(f"WATCH firing {firing['index']} at {firing['look_valid']} {firing['lat']},{firing['lon']}",
                  flush=True)
            children.append(_spawn(argv, Path(arguments.out or Path(arguments.run) / "nests")
                                   / "logs" / f"watch-trigger-{firing['index']}.log"))
            quiet_since = time.monotonic()
        ended = (forecast_dir / "nest-trigger-receipt.json").is_file()
        if ended and not firings:
            break
        if time.monotonic() - quiet_since > arguments.watch_timeout_minutes * 60:
            break
        time.sleep(2.0)
    codes = [child.wait() for child in children]
    print(f"WATCH done firings={len(done)} returncodes={codes}", flush=True)
    return 0 if all(code == 0 for code in codes) else 1


def nest_name(arguments: argparse.Namespace, lat: float | None = None, lon: float | None = None) -> str | None:
    """The nest's folder name: ``--name``, else one made from its point (None without a point)."""

    if arguments.name:
        return arguments.name
    lat = arguments.lat if lat is None else lat
    lon = arguments.lon if lon is None else lon
    if lat is None or lon is None:
        return None
    return f"nest-{lat:+.2f}{lon:+.2f}".replace(".", "p")


def run_add_nest(arguments: argparse.Namespace) -> int:
    if arguments.watch:
        return watch(arguments)
    firing = json.loads(arguments.firing_json) if arguments.firing_json else None
    clicked = time.time()
    # Named before the add, so a failure writes the receipt and the "ended" row
    # even without --name: an add that failed with no "ended" row kept blocking
    # trigger firings over its area until its planned end (f534af2c checker).
    name = nest_name(arguments)
    try:
        receipt = add_nest(arguments, firing=firing, name=name)
    except BaseException:
        # A firing whose nest never ran must not keep its card claim or its whole
        # hold: on the test run (2026-10-23) a failed trigger-0 left its
        # reservation in the ledger and its hold committed for the rest of the run.
        if firing is not None:
            if firing.get("card_reservation"):
                CardLedger(firing.get("card_ledger") or arguments.card_ledger,
                           device=arguments.device).release(firing["card_reservation"])
            measured = {**measured_block(clicked, time.time(), arguments.usd_per_hour, False), "failed": True}
            parent = parent_run(Path(arguments.run))
            if parent.get("forecast"):
                _record_spent(Path(parent["forecast"]), firing, measured)
            out = Path(arguments.out or (Path(arguments.run) / "nests")) / (name or "")
            if name and out.is_dir():
                # What the box reports to the site: the time it took, charged at cost, nothing more.
                _write_receipt(out, {"schema": SCHEMA, "firing": firing, "measured": measured,
                                     "error": repr(sys.exc_info()[1])[:2000]})
        # A nest that never ran covers no storm: the parent's trigger may fire there.
        parent = parent_run(Path(arguments.run))
        if parent.get("forecast") and name:
            from .nest_triggers import note_live_nest
            note_live_nest(Path(parent["forecast"]), {"event": "ended", "name": name,
                                                      "failed": True, "t": round(time.time(), 3)})
        raise
    # A nest that ended badly is not a success to the caller either.
    return 2 if receipt.get("failed") else 0


def add_add_nest_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "add-nest", help="add a WOOF nest to a running hex run, on the same card, from a frame it closed",
        description=__doc__.split("\n\n")[0])
    from woof.output_formats import add_arguments
    add_arguments(parser)
    parser.add_argument("run", help="the hex run's folder (mesh-plan --out-dir, or a forecast --out)")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--dx-km", type=float, default=3.0)
    parser.add_argument("--size-km", type=float, default=400.0)
    parser.add_argument("--hours", type=float, default=None, help="default: to the parent's end")
    parser.add_argument("--start", default="now", help="now | -Nh | YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--cadence-hours", type=float, default=1.0,
                        help="the parent's history spacing, the nest's boundary interval (default 1)")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", type=Path, default=None, help="folder for nests (default RUN/nests)")
    parser.add_argument("--geog", default=None, help="WPS_GEOG root")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--forecast-dir", type=Path, default=None,
                        help="override: the parent forecast's folder (default: read from the live run)")
    parser.add_argument("--mesh", default=None, help="override: the parent's grid file")
    parser.add_argument("--init", default=None, help="override: the parent's init file")
    parser.add_argument("--simulation-start", default=None, help="override: YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--parent-hours", type=float, default=None, help="override: the parent's forecast hours")
    parser.add_argument("--wait-minutes", type=float, default=30.0,
                        help="how long to wait for the parent to close the start frame")
    parser.add_argument("--wait-for-nest", action="store_true", help="stay until the nest's forecast ends")
    parser.add_argument("--stop-after-cents", type=float, default=None,
                        help="stop the nest when its measured cost reaches this (the hold / the overrun factor)")
    parser.add_argument("--usd-per-hour", type=float, default=None, help="this box's own rate, for --stop-after-cents")
    parser.add_argument("--card-ledger", type=Path, default=None,
                        help="the card's reservation ledger (default: one per card in the temp folder)")
    parser.add_argument("--card-reservation", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--card-queue", default=os.environ.get("WOOF_HEX_CARD_QUEUE") or None,
                        help="a box card queue (a pool runner script): the nest's GPU stage takes a pool "
                             "lease of its own size (default $WOOF_HEX_CARD_QUEUE)")
    parser.add_argument("--lane-prefix", default="hexnest-", help="pool lease name prefix with --card-queue")
    parser.add_argument("--queue-wait-seconds", type=float, default=14400.0)
    parser.add_argument("--nest-vram-mib", type=float, default=16000.0,
                        help="pool lease size when the engine estimate is unavailable")
    parser.add_argument("--render-products", default=NEST_PRODUCTS,
                        help="the nest's maps, `woof render --products`' spec (default: WOOF's met set; see "
                             "NEST_PRODUCTS for why not `all`)")
    parser.add_argument("--physics-profile", default=None,
                        help="the nest's physics suite, a `woof domain --physics-profile` id (default: "
                             f"{CONVECTION_PERMITTING_PROFILE} below {CONVECTION_PERMITTING_BELOW_KM:g} km, else the "
                             "engine's default for hex-global)")
    parser.add_argument("--watch", action="store_true",
                        help="add a nest for every [hex_nests] firing in RUN's forecast nest-triggers.jsonl")
    parser.add_argument("--watch-timeout-minutes", type=float, default=600.0)
    parser.add_argument("--firing-json", default=None, help=argparse.SUPPRESS)
    parser.set_defaults(handler=run_add_nest)
