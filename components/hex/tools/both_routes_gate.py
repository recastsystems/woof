#!/usr/bin/env python3
"""The both-routes leg of the hex release gate: every capability family, once, on each physics route.

Usage (from the tree root, or with woof hex installed)::

    PYTHONPATH=src python tools/both_routes_gate.py --work DIR [--routes arwen,mpas] [--legs a,b] \
        [--staged GFS_PRODUCTS_DIR] [--geog WPS_GEOG] [--best-track IBTRACS.csv] [--storm-staged FILE] [--jobs N]
    PYTHONPATH=src python tools/both_routes_gate.py --plan        # print every command, run nothing

THE BREAKAGE THIS PREVENTS: a capability that works on one physics route and is refused, silently dropped or
broken on the other. Before this leg existed the WOOF route carried every capability and the MPAS-physics route
lost them one at a time without a red anywhere: limited-area runs died in the terrain coupling, rosters went to
the wrong route, chemistry and trigger nests were dropped without a word, frames were published only at the end,
restart did not equal the straight run, and multi-card needed a native kit that no longer ships.

The legs live in ``tools/battery/both_routes.json``: one short real run per family (120 km globe, 1-2 h, or a
small variable-resolution or limited-area mesh), each on both routes, with checks that name what they prove. The
CPU tier (``tests/test_both_routes_gate.py``) parses every command with the real ``woof hex`` parser, so a
renamed flag turns that test red instead of this gate.

Cards are never held exclusively (release tests never block a card). Each GPU step goes through, in order:

``HEX_GPU_LEASE=<program>``
    a pool admission program invoked as ``<program> --vram-mib N [--cards K] <lane> -- <command>``, with the
    leg's measured VRAM. K is passed only for multi-card legs; a one-card leg shares its card.
``HEX_GPU_QUEUE`` (``woof.hex.site_runner``)
    the site GPU queue, never with ``--proof``.
otherwise the command runs directly.

Writes ``<work>/both-routes.json`` (machine receipt) and ``<work>/both-routes.md`` (the matrix), and exits 0 only
when every leg passes on every route asked for. Run data stays under ``--work``; delete it after reading the
receipt (``--clean`` removes everything except the receipts and logs).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import glob
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TREE = Path(__file__).resolve().parents[1]
LEGS_FILE = TREE / "tools" / "battery" / "both_routes.json"
LEASE_ENV = "HEX_GPU_LEASE"
STRIP_FROM_BASE = {"--hours": 1, "--out": 1, "--scratch": 1, "--case-label": 1}


def load_legs(path: Path = LEGS_FILE) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "gpuwm-hex.both-routes-gate/v1":
        raise SystemExit(f"{path}: unknown schema {spec.get('schema')!r}")
    return spec


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- argv expansion ------------------------------------------------------------------------------------------------

class Context:
    """Everything a leg's argv may name: run options, the route, and other legs' folders."""

    def __init__(self, spec: dict, options: argparse.Namespace, route: str, work_root: Path):
        self.spec, self.options, self.route, self.work_root = spec, options, route, work_root
        self.cases = spec["cases"]

    def leg_dir(self, leg_id: str) -> Path:
        return self.work_root / self.route / leg_id

    def base_plan(self) -> dict:
        path = self.leg_dir("global") / "global-plan.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def base_forecast(self) -> list[str]:
        """The base leg's own forecast argv (mesh, grid, static, init, route), minus what each leg sets itself."""
        argv = list(self.base_plan()["commands"]["forecast"])
        if argv and argv[0] == "forecast":
            argv = argv[1:]
        out, index = [], 0
        while index < len(argv):
            token = argv[index]
            name = token.split("=", 1)[0]
            if name in STRIP_FROM_BASE:
                index += 1 if "=" in token else 1 + STRIP_FROM_BASE[name]
                continue
            out.append(token)
            index += 1
        return ["forecast", *out]

    def scalar(self, name: str, work: Path) -> str:
        if name == "work":
            return str(work)
        if name == "cycle":
            return self.options.cycle or self.cases["cycle"]
        if name == "storm_sid":
            return self.options.storm or self.cases["storm_best_track_sid"]
        if name == "storm_cycle":
            return self.options.storm_cycle or self.cases["storm_cycle"]
        if name == "best_track":
            return str(self.options.best_track) if self.options.best_track else "{best_track}"
        if name.startswith("leg:"):
            return str(self.leg_dir(name[4:]))
        if name in ("init", "grid", "static"):
            return str(self.base_plan()["paths"][name])
        if name.startswith("checkpoint:"):
            return str(find_checkpoint(work / name.split(":", 1)[1]))
        raise KeyError(name)

    def expand(self, argv: list[str], work: Path) -> list[str]:
        routes = self.spec["routes"][self.route]
        lists = {
            "route_door": routes["door"],
            "route_forecast": routes["forecast"],
            "route_moving": routes["moving"],
            "staged_args": ["--staged", str(self.options.staged)] if self.options.staged else [],
            "storm_staged_args": ["--staged", str(self.options.storm_staged)] if self.options.storm_staged else [],
            "geog_args": ["--geog", str(self.options.geog)] if self.options.geog else [],
        }
        out: list[str] = []
        for token in argv:
            if token == "{base_forecast}":
                out += self.base_forecast()
                continue
            if token.startswith("{") and token.endswith("}") and token[1:-1] in lists:
                out += lists[token[1:-1]]
                continue
            out.append(re.sub(r"\{([a-z_]+(?::[A-Za-z0-9_.-]+)?)\}", lambda m: self.scalar(m.group(1), work), token)
                       if not token.lstrip().startswith("{\"") else token)
        return out


def find_checkpoint(folder: Path) -> Path:
    found = sorted(p for p in folder.rglob("*.pkl") if "checkpoint" in p.name or p.parent.name.startswith("checkpoint"))
    if not found:
        found = sorted(folder.rglob("*.pkl"))
    if not found:
        raise FileNotFoundError(f"no checkpoint (*.pkl) under {folder}")
    return found[0]


# --- builtins: small inputs some legs need -----------------------------------------------------------------------

def builtin_trigger_table(ctx: Context, work: Path) -> None:
    """A table that arms on any run: the check is that the look arms on this route, not that it fires."""
    table = {"hex_nests": {
        "max_firings": 1, "max_live": 1, "cadence_s": 1200, "sustain_looks": 1, "lead_back_s": 0, "cooldown_s": 1800,
        "area": {"box": {"south": 25.0, "north": 50.0, "west": -125.0, "east": -67.0}},
        "nest": {"dx_km": 3, "size_km": 300, "hours": 1},
        "when": [{"all": [{"field": "refl", "op": ">", "value": 40, "cover_km2": 300}]}],
    }}
    (work / "triggers.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def builtin_member_roster(ctx: Context, work: Path) -> None:
    (work / "roster.json").write_text(json.dumps([{"run_id": "m1"}, {"run_id": "m2"}], indent=1), encoding="utf-8")


def builtin_chemistry_inputs(ctx: Context, work: Path) -> None:
    """Two species on the base mesh (contract gpuwm-hex.chem-source-frame/v1): a smoke layer plus a 3 h source, and
    a dust layer with no source. The same inputs feed both routes."""
    import hashlib

    import numpy as np
    from netCDF4 import Dataset

    plan = ctx.base_plan()
    grid, init = Path(plan["paths"]["grid"]), Path(plan["paths"]["init"])
    out = work / "chem"
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(grid.read_bytes()).hexdigest()
    start_dt = datetime.strptime(plan["start_time"], "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    start = int(start_dt.timestamp())
    with Dataset(init) as d:
        lat = np.degrees(d["latCell"][:])
        lon = np.degrees(d["lonCell"][:])
        lon = np.where(lon > 180, lon - 360, lon)
        z = d["zgrid"][:]
    nc, nl = lat.size, z.shape[1] - 1
    zmid = 0.5 * (z[:, 1:] + z[:, :-1]) - z[:, :1]
    box = (lat > 30) & (lat < 45) & (lon > -105) & (lon < -85)
    smoke = np.where(box[:, None] & (zmid < 3000), 2e-8, 0).astype("f4").T
    dust = np.where(zmid < 1000, 5e-9, 0).astype("f4").T
    fire = (lat > 36) & (lat < 40) & (lon > -122) & (lon < -116)

    def frame(path: Path, arrays: dict, dims: tuple, units: str) -> None:
        with Dataset(path, "w") as d:
            d.createDimension("nVertLevels", nl)
            d.createDimension("nCells", nc)
            for key, value in dict(schema="gpuwm-hex.chem-source-frame/v1", basis="dry_air", mesh_sha256=sha).items():
                d.setncattr(key, value)
            d.setncattr("valid_time_unix", np.int64(start))
            for name, array in arrays.items():
                var = d.createVariable(name, "f4", dims)
                var[:] = array
                var.setncattr("units", units)

    frame(out / "initial.nc", {"smoke": smoke, "dust_1": dust}, ("nVertLevels", "nCells"), "kg kg-1")
    frame(out / "emis.nc", {"smoke": np.where(fire, 2e-9, 0).astype("f4")}, ("nCells",), "kg m-2 s-1")
    valid = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    config = {
        "schema": "gpuwm-hex.chemistry/v1",
        "species": [
            {"name": "smoke", "dry_deposition_m_s": 0.001, "settling_m_s": 0.0001, "mass_extinction_m2_kg": 4500.0},
            {"name": "dust_1", "dry_deposition_m_s": 0.002, "settling_m_s": 0.01, "mass_extinction_m2_kg": 800.0},
        ],
        "inputs": {"valid_time": valid, "mesh_sha256": sha, "initial": {"path": "initial.nc"},
                   "emissions": [{"path": "emis.nc", "valid_time": valid, "interval_s": 10800}],
                   "zero_emission_species": ["dust_1"]},
    }
    (out / "chemistry.json").write_text(json.dumps(config, indent=1), encoding="utf-8")


BUILTINS = {"trigger_table": builtin_trigger_table, "member_roster": builtin_member_roster,
            "chemistry_inputs": builtin_chemistry_inputs}


# --- running a step ----------------------------------------------------------------------------------------------

def hex_command(argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "hexcore", *argv]


def lease_prefix(leg: dict, lane: str) -> list[str]:
    program = os.environ.get(LEASE_ENV, "").strip()
    if program:
        prefix = [*shlex.split(program), "--vram-mib", str(int(leg["vram_mib"]))]
        if int(leg["cards"]) > 1:
            prefix += ["--cards", str(int(leg["cards"]))]
        return [*prefix, lane, "--"]
    sys.path.insert(0, str(TREE / "src"))
    from woof.hex import site_runner

    return site_runner.gpu_prefix(int(leg.get("minutes", 30)))


def process_tree(root: int) -> set[int]:
    pids, frontier = {root}, [root]
    while frontier:
        pid = frontier.pop()
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in pids:
                pids.add(child)
                frontier.append(child)
    return pids


def gpu_memory_by_pid() -> dict[int, int]:
    try:
        text = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    usage: dict[int, int] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = usage.get(int(parts[0]), 0) + int(parts[1])
    return usage


def count_rows(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def published_frames(work: Path, patterns: list[str]) -> int:
    """Frames published so far: rows of a history-ready list, or durable sidecars (each route writes one or both)."""
    best = 0
    for pattern in patterns:
        if pattern.endswith(".jsonl"):
            best = max(best, count_rows(work / pattern))
        else:
            best = max(best, len(glob.glob(str(work / pattern))))
    return best


def run_step(ctx: Context, leg: dict, step: dict, work: Path, record: dict) -> int:
    started = time.time()
    entry: dict[str, Any] = {"name": step["name"], "started_utc": utc()}
    try:  # the host's load at the start of each step: a timed leg's red is read against it
        entry["host_load"] = [round(value, 1) for value in os.getloadavg()]
    except (AttributeError, OSError):
        entry["host_load"] = None
    record["steps"].append(entry)
    if "builtin" in step:
        BUILTINS[step["builtin"]](ctx, work)
        entry.update(returncode=0, wall_seconds=round(time.time() - started, 2), builtin=step["builtin"])
        return 0
    argv = ctx.expand(step["argv"], work)
    command = hex_command(argv)
    if step.get("gpu"):
        command = [*lease_prefix(leg, f"both-routes-{ctx.route}-{leg['id']}"), *command]
    entry["argv"] = command
    log = work / f"{step['name']}.log"
    watch = step.get("watch")
    environment = dict(os.environ)
    # Observation scoring is not a family here and downloads whole archives: the doors' --skip score switch.
    environment["WOOF_HEX_OBSERVATION_SCORING"] = "off"
    if "{base_forecast}" in step["argv"]:
        # The base leg's mesh is described by the mesh rows its door wrote; the forecast door reads them from here.
        rows = ctx.base_plan()["paths"].get("mesh_rows")
        if rows:
            environment["WOOF_HEX_MESH_ROWS"] = str(rows)
    peak, live_rows = 0, 0
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=work, env=environment)
        while process.poll() is None:
            if step.get("gpu"):
                usage = gpu_memory_by_pid()
                if usage:
                    peak = max(peak, sum(usage.get(pid, 0) for pid in process_tree(process.pid)))
            if watch is not None:
                live_rows = max(live_rows, published_frames(work, watch))
            time.sleep(2.0)
    entry.update(returncode=process.returncode, wall_seconds=round(time.time() - started, 2), log=str(log))
    if step.get("gpu"):
        entry["peak_vram_mib_sampled"] = peak or None
        record["peak_vram_mib"] = max(record.get("peak_vram_mib") or 0, peak) or None
    if watch is not None:
        # Rows seen while the process was still running: frames published live, not in one batch at the end.
        record["live_rows"] = live_rows
    return process.returncode


# --- checks -------------------------------------------------------------------------------------------------------

def matches(work: Path, patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update(Path(p) for p in glob.glob(str(work / pattern), recursive=True))
    return sorted(found)


def history_frames(folder: Path) -> list[Path]:
    frames = matches(folder, ["**/history.*.nc", "**/cuda-history.*.nc"])
    return [f for f in frames if "scratch" not in f.parts]


def compare_words(a: Path, b: Path) -> tuple[int, int, list[str]]:
    """Differing values over every floating variable the two frames share (the random file_id is not data)."""
    import numpy as np
    from netCDF4 import Dataset

    differing = total = 0
    names: list[str] = []
    with Dataset(a) as da, Dataset(b) as db:
        for name, var in da.variables.items():
            if name not in db.variables or var.dtype.kind not in "fiu":
                continue
            x = np.asarray(var[:].filled(np.nan) if hasattr(var[:], "filled") else var[:])
            y = np.asarray(db.variables[name][:])
            if x.shape != y.shape:
                names.append(f"{name}(shape)")
                differing += x.size
                total += x.size
                continue
            if x.dtype.kind == "f":
                xv, yv = x.astype(x.dtype).view(f"u{x.dtype.itemsize}"), np.asarray(y, dtype=x.dtype).view(f"u{x.dtype.itemsize}")
            else:
                xv, yv = x, y
            count = int(np.count_nonzero(xv != yv))
            total += x.size
            if count:
                differing += count
                names.append(name)
    return differing, total, names


_FRAME_TIME = re.compile(r"(\d{4})-(\d\d)-(\d\d)_(\d\d)[.:](\d\d)[.:](\d\d)")


def warm_seconds_per_forecast_hour(folder: Path) -> dict | None:
    """Wall seconds per forecast hour between the first and last history frame after f000 (file times).

    f000 is written before the first step, so the interval from f001 on carries no start-up; each route writes
    one history file per frame (history.* on the MPAS route, cuda-history.* on the WOOF route). None when fewer
    than three frames exist (no warm interval)."""
    from datetime import datetime

    frames = []
    for path in history_frames(folder):
        found = _FRAME_TIME.search(path.name)
        if found:
            frames.append((datetime(*map(int, found.groups())), path.stat().st_mtime, path.name))
    frames.sort()
    if len(frames) < 3:
        return None
    model_hours = (frames[-1][0] - frames[1][0]).total_seconds() / 3600.0
    wall = frames[-1][1] - frames[1][1]
    if model_hours <= 0 or wall <= 0:
        return None
    hourly = [round(b[1] - a[1], 1) for a, b in zip(frames[1:], frames[2:])]
    return {"seconds_per_forecast_hour": round(wall / model_hours, 2), "model_hours": round(model_hours, 3),
            "frame_to_frame_seconds": hourly, "frames": [name for _, _, name in frames]}


def run_check(ctx: Context, check: dict, work: Path, record: dict) -> dict:
    kind = check["kind"]
    result: dict[str, Any] = {"kind": kind, "why": check.get("why")}
    if kind == "glob":
        found = matches(work, check["patterns"])
        result.update(found=len(found), need=check["min"], ok=len(found) >= check["min"])
    elif kind == "live_rows":
        rows = record.get("live_rows", 0)
        result.update(found=rows, need=check["min"], ok=rows >= check["min"])
    elif kind == "nc_vars":
        from netCDF4 import Dataset

        files = matches(work, check["patterns"])
        missing: dict[str, list[str]] = {}
        for path in files:
            with Dataset(path) as d:
                names = set(d.variables)
            lack = [v for v in check.get("all_of", []) if v not in names]
            if check.get("any_of") and not names.intersection(check["any_of"]):
                lack.append("|".join(check["any_of"]))
            if lack:
                missing[path.name] = lack
        result.update(files=len(files), missing=missing, ok=len(files) >= check.get("min", 1) and not missing)
    elif kind == "text":
        try:
            text = (work / check["file"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        hit = re.search(check["pattern"], text)
        result.update(pattern=check["pattern"], ok=hit is not None)
    elif kind == "jsonl_key":
        rows = []
        try:
            rows = [json.loads(line) for line in (work / check["file"]).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        except (OSError, ValueError):
            pass
        result.update(rows=len(rows), ok=any(check["key"] in row for row in rows))
    elif kind == "json_len":
        try:
            value = json.loads((work / check["file"]).read_text(encoding="utf-8")).get(check["key"]) or []
        except (OSError, ValueError):
            value = []
        result.update(found=len(value), need=check["min"], ok=len(value) >= check["min"])
    elif kind == "lts_speed":
        # Warm wall per forecast hour of the fixed-step arm over the LTS arm (frame to frame, f000 excluded).
        fixed = warm_seconds_per_forecast_hour(work / check["fixed"])
        lts = warm_seconds_per_forecast_hour(work / check["lts"])
        floor = float(check["min_speedup"][ctx.route])
        speedup = None if not fixed or not lts else fixed["seconds_per_forecast_hour"] / lts["seconds_per_forecast_hour"]
        choice = None
        for path in sorted((work / check["lts"]).rglob("fullstep-lts-choice.json")):
            try:
                choice = json.loads(path.read_text(encoding="utf-8")).get("door_choice")
            except (OSError, ValueError):
                choice = None
        result.update(fixed=fixed, lts=lts, speedup=None if speedup is None else round(speedup, 3),
                      min_speedup=floor, recorded=(check.get("recorded") or {}).get(ctx.route),
                      door_choice=None if choice is None else {k: choice.get(k) for k in (
                          "choice", "lts_top_rate", "lts_dt_seconds", "fixed_dt_seconds",
                          "predicted_seconds_per_forecast_hour")},
                      host_load=[step.get("host_load") for step in record.get("steps", [])],
                      ok=speedup is not None and speedup >= floor)
    elif kind == "same_words":
        a = Path(ctx.expand([check["a"]], work)[0])
        b = Path(ctx.expand([check["b"]], work)[0])
        fa, fb = history_frames(a), history_frames(b)
        if not fa or not fb:
            result.update(ok=False, note=f"frames: {len(fa)} in {a}, {len(fb)} in {b}")
        else:
            last_a, last_b = fa[-1], fb[-1]
            if last_a.name != last_b.name:
                result.update(ok=False, note=f"last frames differ in name: {last_a.name} vs {last_b.name}")
            else:
                differing, total, names = compare_words(last_a, last_b)
                result.update(frame=last_a.name, differing=differing, total=total, variables=names[:20],
                              ok=differing == 0 and total > 0)
    else:
        result.update(ok=False, note=f"unknown check kind {kind}")
    return result


# --- step time ----------------------------------------------------------------------------------------------------

def find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value and value[key] is not None:
            return value[key]
        for child in value.values():
            found = find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_key(child, key)
            if found is not None:
                return found
    return None


def step_seconds(work: Path) -> float | None:
    """Median seconds per step after the first: the MPAS route's per-step stages, or the WOOF route's receipt."""
    for path in sorted(work.rglob("native-runtime-timings.json")):
        try:
            stages = json.loads(path.read_text(encoding="utf-8"))["stages"]
        except (OSError, ValueError, KeyError):
            continue
        stepping = next((s for s in stages if s.get("stage") == "lts-stepping"), None)
        if stepping is not None:
            # Active LTS: the per-stage rows are class steps of every rate, so their median is no step of the run.
            # The loop's own measured rate, as wall seconds per base step (woof.hex.mpasphys.lts_v842.stepping_row).
            rate = stepping.get("warm_seconds_per_forecast_hour") or stepping.get("seconds_per_forecast_hour")
            if rate:
                return round(float(rate) * float(stepping["base_dt_seconds"]) / 3600.0, 3)
            continue
        if any("lts_class" in s or "lts_block" in s for s in stages):
            continue   # an LTS timeline without its loop row (a failed run): no step to report
        per_stage: dict[str, list[float]] = {}
        for stage in stages:
            if stage.get("stage") in ("outer-physics", "dycore-and-post-transport", "publish-and-streams"):
                per_stage.setdefault(stage["stage"], []).append(float(stage.get("wall_seconds") or 0.0))
        if per_stage and all(len(v) > 1 for v in per_stage.values()):
            return round(sum(statistics.median(v[1:]) for v in per_stage.values()), 3)
    for path in sorted(work.rglob("forecast-receipt.json")):
        try:
            value = find_key(json.loads(path.read_text(encoding="utf-8")), "seconds_per_step_after_first")
        except (OSError, ValueError):
            continue
        if value is not None:
            return round(float(value), 3)
    return None


# --- the gate -----------------------------------------------------------------------------------------------------

def run_leg(ctx: Context, leg: dict) -> dict:
    work = ctx.leg_dir(leg["id"])
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    record: dict[str, Any] = {"leg": leg["id"], "family": leg["family"], "route": ctx.route, "steps": [],
                              "started_utc": utc()}
    started = time.time()
    try:
        for step in leg["steps"]:
            rc = run_step(ctx, leg, step, work, record)
            if rc != 0:
                record["failed_step"] = step["name"]
                record["error_tail"] = tail(work / f"{step['name']}.log")
                break
        if "failed_step" not in record:
            record["checks"] = [run_check(ctx, check, work, record) for check in leg["checks"]]
    except Exception as error:  # a leg's own breakage is that leg's red, never the whole gate's stop
        record["failed_step"] = record.get("failed_step") or "gate"
        record["error_tail"] = f"{type(error).__name__}: {error}"
    record["wall_seconds"] = round(time.time() - started, 1)
    record["step_seconds_median"] = step_seconds(work)
    record["ok"] = "failed_step" not in record and all(c["ok"] for c in record.get("checks", []))
    (work / "leg-receipt.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def matrix_markdown(spec: dict, routes: list[str], records: list[dict]) -> str:
    by = {(r["leg"], r["route"]): r for r in records}
    lines = ["# Both routes: one run per capability family", "",
             f"Written {utc()}. Each cell: verdict, median step (s), peak VRAM (MiB, sampled).", "",
             "| Family | " + " | ".join(routes) + " |", "|---|" + "---|" * len(routes)]
    for leg in spec["legs"]:
        cells = []
        for route in routes:
            r = by.get((leg["id"], route))
            if r is None:
                cells.append("not run")
                continue
            verdict = "PASS" if r["ok"] else f"FAIL ({r.get('failed_step') or ', '.join(c['kind'] for c in r.get('checks', []) if not c['ok'])})"
            cells.append(f"{verdict}; {r.get('step_seconds_median') or '-'} s; {r.get('peak_vram_mib') or '-'} MiB")
        lines.append(f"| {leg['id']}: {leg['family']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="both_routes_gate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", type=Path, help="run folder (all run data lands here)")
    parser.add_argument("--routes", default="arwen,mpas")
    parser.add_argument("--legs", default=None, help="comma list of leg ids (default: all)")
    parser.add_argument("--cycle", default=None, help="global-case cycle (default from both_routes.json)")
    parser.add_argument("--staged", type=Path, default=None, help="staged GFS for the cycle (doors' --staged)")
    parser.add_argument("--geog", type=Path, default=None, help="WPS_GEOG root for generated meshes")
    parser.add_argument("--best-track", type=Path, default=None, help="IBTrACS CSV or best-track JSON (moving leg)")
    parser.add_argument("--storm", default=None)
    parser.add_argument("--storm-cycle", default=None)
    parser.add_argument("--storm-staged", type=Path, default=None, help="staged source for the storm cycle")
    parser.add_argument("--jobs", type=int, default=3, help="legs in flight at once (the pool admits by VRAM)")
    parser.add_argument("--plan", action="store_true", help="print every leg's commands and run nothing")
    parser.add_argument("--clean", action="store_true", help="after the run keep only receipts and logs")
    options = parser.parse_args(argv)
    spec = load_legs()
    routes = [r for r in options.routes.split(",") if r]
    wanted = set(options.legs.split(",")) if options.legs else None
    legs = [leg for leg in spec["legs"] if wanted is None or leg["id"] in wanted]
    work_root = (options.work or Path("both-routes-plan")).resolve()

    if options.plan:
        for route in routes:
            ctx = Context(spec, options, route, work_root)
            for leg in legs:
                for step in leg["steps"]:
                    if "builtin" in step:
                        print(f"[{route}] {leg['id']}/{step['name']}: builtin {step['builtin']}")
                        continue
                    work = ctx.leg_dir(leg["id"])
                    try:
                        line = shlex.join(ctx.expand([t for t in step["argv"] if t != "{base_forecast}"], work))
                    except (KeyError, OSError, FileNotFoundError):
                        line = " ".join(t for t in step["argv"] if t != "{base_forecast}")
                    prefix = "<global forecast argv> " if "{base_forecast}" in step["argv"] else ""
                    print(f"[{route}] {leg['id']}/{step['name']}: woof hex {prefix}{line}")
        return 0
    if options.work is None:
        parser.error("--work is required unless --plan")

    records: list[dict] = []
    lock = threading.Lock()

    def one(route: str, leg: dict) -> dict:
        record = run_leg(Context(spec, options, route, work_root), leg)
        with lock:
            records.append(record)
            print(f"{utc()} {route:6s} {leg['id']:20s} {'PASS' if record['ok'] else 'FAIL'} "
                  f"{record['wall_seconds']} s", flush=True)
        return record

    # The base legs first: every forecast-level leg reuses the global leg's mesh, init and argv on its route.
    base = [leg for leg in legs if leg["id"] == "global"]
    rest = [leg for leg in legs if leg["id"] != "global"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, options.jobs)) as pool:
        list(pool.map(lambda pair: one(*pair), [(route, leg) for route in routes for leg in base]))
        # A forecast-level leg needs the global leg's plan on its route: made just now, or kept from an earlier run
        # of the same --work (a rerun of single legs with --legs reuses it).
        base_ok = {route: (Context(spec, options, route, work_root).leg_dir("global") / "global-plan.json").is_file()
                   for route in routes}
        jobs = []
        for route in routes:
            for leg in rest:
                if leg.get("after") == "global" and not base_ok.get(route, False):
                    records.append({"leg": leg["id"], "family": leg["family"], "route": route, "ok": False,
                                    "failed_step": "global leg (no base plan)"})
                    continue
                jobs.append((route, leg))
        list(pool.map(lambda pair: one(*pair), jobs))

    order = {leg["id"]: i for i, leg in enumerate(spec["legs"])}
    records.sort(key=lambda r: (order.get(r["leg"], 99), r["route"]))
    receipt = {"schema": "gpuwm-hex.both-routes-receipt/v1", "written_utc": utc(), "routes": routes,
               "legs_file": str(LEGS_FILE), "records": records,
               "ok": all(r["ok"] for r in records) and bool(records)}
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "both-routes.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    (work_root / "both-routes.md").write_text(matrix_markdown(spec, routes, records), encoding="utf-8")
    if options.clean:
        for path in sorted(work_root.rglob("*"), reverse=True):
            if path.is_file() and not (path.suffix in (".log", ".md") or path.name.endswith("receipt.json")
                                       or path.name in ("both-routes.json", "native-runtime-timings.json",
                                                        "global-plan.json", "controller.json")):
                path.unlink()
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    print(matrix_markdown(spec, routes, records))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
