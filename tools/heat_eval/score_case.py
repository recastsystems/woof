"""Score one case's arms and HRRR against MRMS and surface stations.

Radar (registration ``nowcast-heat-v1``,
:func:`woof.verify.obs.registration.nowcast_heat_parameters`):

* truth is the MRMS composite nearest each valid time (two minutes either
  side), fetched and decoded by ``rw_mrms`` through the case's own cache;
* the model is its own REFL_10CM column maximum; HRRR is its own REFC from
  ``wrfsfcfNN`` of the case's cycle, decoded by ``grib2_dump``;
* FSS at 20 and 35 dBZ in 9, 27 and 63 km boxes, the area ratios and the
  contingency tables come from :func:`woof.verify.obs.battery.score_reflectivity`
  (Rust regrid and Rust masked FSS); the RMSE in dBZ (both floored at
  0 dBZ) uses the same remap plan and the same scored cells.

Stations: the frozen scoreboard (scoreboard-v2, ``rw_scoreboard``) against
routine METARs fetched by ``rw_asos``, for T2, Td2 and 10 m wind, on one
station list per case (inside the grid, terrain within 200 m of the
station elevation) that every arm and HRRR is scored on.

Every score is filed by valid time as hours after the case's t0, so arm B
(which starts at t0 - 1 h) lines up with the others.  Output:
``<case>/scores/table.json`` (schema ``gpuwm.heat-eval-scores.v1``, the
decision's input) and the full battery records beside it.

    python -m tools.heat_eval.score_case --case ID [--root /work/nh] [--plan core]
        [--arms A,B,C,E] [--no-stations] [--no-radar]

Complete family frames also use this module:
``python -m tools.heat_eval.score_case --grid-request REQUEST --matcher BINARY --out DIR``.
Their native ensemble registration is separate from the heating decision.

Python here is orchestration: it lists files, starts the Rust decoders and
scorers, and files their numbers.  No weather array is decoded here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tools.heat_eval import decide
from tools.heat_eval import plan as planmod

FRAME_RE = re.compile(r"^wrfout_d01_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})$")
HRRR_LABEL = "HRRR"

STATION_VARIABLES = {"t2": "t2_rmse", "td2": "td2_rmse", "wind10": "wind10_rmse"}
STATION_ALIASES = {"t2": "t2", "temperature_2m": "t2", "td2": "td2", "dewpoint_2m": "td2",
                   "wind10": "wind10", "wind_speed_10m": "wind10"}


def binary(name: str, env: str) -> str:
    path = os.environ.get(env) or shutil.which(name)
    if not path:
        raise SystemExit(f"{name} not found (set {env}); the bootstrap links it under $NH/bin")
    return path


def seam(time: datetime) -> str:
    """The scorer's valid-time spelling (naive UTC)."""
    return time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def find_frames(out_dir: Path) -> dict[str, Path]:
    """Valid time -> history file, for every d01 frame under an arm's outdir."""
    frames: dict[str, Path] = {}
    for path in sorted(Path(out_dir).rglob("wrfout_d01_*")):
        match = FRAME_RE.match(path.name)
        if match:
            day, hh, mm, ss = match.groups()
            frames[f"{day}T{hh}:{mm}:{ss}"] = path
    return frames


class HistoryRun:
    """An arm's history files as a scorer model source (composite only)."""

    def __init__(self, label: str, out_dir: Path, dx_m: float):
        self.label = label
        self.frames = find_frames(out_dir)
        self.dx_m = float(dx_m)
        if not self.frames:
            raise FileNotFoundError(f"{label}: no wrfout_d01 frames under {out_dir}")

    def valid_times(self) -> tuple[str, ...]:
        return tuple(sorted(self.frames))

    def frame(self, valid_time: str) -> Path:
        path = self.frames.get(valid_time)
        if path is None:
            raise ValueError(f"{self.label} has no frame at {valid_time}")
        return path

    def grid(self):
        from woof.verify import field_metrics
        from woof.verify.obs.contracts import ModelGrid, normalize_longitude
        first = self.frame(self.valid_times()[0])
        return ModelGrid(
            latitude=field_metrics.read_frame_field(first, "XLAT"),
            longitude=normalize_longitude(field_metrics.read_frame_field(first, "XLONG")),
            dx_m=self.dx_m,
            terrain_m=field_metrics.read_frame_field(first, "HGT"))

    def composite_reflectivity(self, valid_time: str) -> np.ndarray:
        from woof.verify.obs.model_source import frame_composite_reflectivity
        return frame_composite_reflectivity(self.frame(valid_time))


class HrrrReflectivity:
    """HRRR's own REFC on the run grid, as a centred window of HRRR's grid.

    The door runs on HRRR's own Lambert grid, a whole number of cells
    smaller on every side.  The window offset is accepted only when the run
    grid's first cell lies that many diagonal steps from HRRR's published
    first point; anything else would need a regrid, and is refused.
    """

    REFC = ("0", "16", "196", "10")   # discipline, category, parameter, level type

    def __init__(self, reference_dir: Path, cycle: datetime, grid, work: Path):
        self.reference_dir = Path(reference_dir)
        self.cycle = cycle
        self.grid = grid
        self.work = Path(work)
        self.inventory = binary("grib2_inventory", "WOOF_GRIB2_INVENTORY")
        self.dump = binary("grib2_dump", "WOOF_GRIB2_DUMP")
        self.offset: int | None = None

    def file_for(self, valid_time: str) -> Path:
        valid = datetime.fromisoformat(valid_time).replace(tzinfo=timezone.utc)
        lead = int((valid - self.cycle).total_seconds() // 3600)
        return self.reference_dir / f"hrrr.t{self.cycle.hour:02d}z.wrfsfcf{lead:02d}.grib2"

    def _refc_row(self, path: Path) -> dict[str, str]:
        text = subprocess.run([self.inventory, str(path)], check=True, capture_output=True,
                              text=True).stdout
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        header = lines[0].split("\t")
        for line in lines[1:]:
            row = dict(zip(header, line.split("\t")))
            if (row["discipline"], row["category"], row["parameter"], row["level_type"]) == self.REFC:
                return row
        raise ValueError(f"{path} carries no REFC (entire atmosphere) message")

    def composite_reflectivity(self, valid_time: str) -> np.ndarray:
        path = self.file_for(valid_time)
        row = self._refc_row(path)
        nx, ny = int(row["nx"]), int(row["ny"])
        if int(row["scan_mode"], 16) & 0xE0 != 0x40:
            raise ValueError(f"{path}: scan mode {row['scan_mode']}; rows must run south to north")
        out = self.work / f"refc-{path.stem}"
        if out.exists():
            shutil.rmtree(out)
        subprocess.run([self.dump, str(path), str(out), row["index"]], check=True,
                       capture_output=True)
        values = np.fromfile(out / f"field-{int(row['index']):04d}.f64le", dtype="<f8")
        shutil.rmtree(out, ignore_errors=True)
        field = values.reshape(ny, nx)
        my, mx = self.grid.shape
        offset = self._offset(row, ny, nx, my, mx)
        window = field[offset:offset + my, offset:offset + mx]
        return np.where(np.isfinite(window), window, -99.0)

    def _offset(self, row, ny, nx, my, mx) -> int:
        if self.offset is not None:
            return self.offset
        if (ny - my) % 2 or (nx - mx) % 2 or (ny - my) != (nx - mx) or ny < my:
            raise ValueError(
                f"HRRR {ny}x{nx} is not a centred window around the run's {my}x{mx}; the HRRR "
                f"column would need a regrid, which this scorer does not do")
        offset = (ny - my) // 2
        lat1, lon1, dx = float(row["lat1"]), float(row["lon1"]), float(row["dx"])
        if lon1 > 180.0:
            lon1 -= 360.0
        lat0 = float(self.grid.latitude[0, 0])
        lon0 = float(self.grid.longitude[0, 0])
        r = 6_371_229.0
        p1, p2 = math.radians(lat1), math.radians(lat0)
        h = (math.sin((p2 - p1) / 2) ** 2
             + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon0 - lon1) / 2) ** 2)
        distance = 2 * r * math.asin(min(1.0, math.sqrt(h)))
        expected = offset * dx * math.sqrt(2.0)
        if abs(distance - expected) > 0.15 * dx + 300.0:
            raise ValueError(
                f"the run grid's first cell is {distance:.0f} m from HRRR's first point; a "
                f"centred window of {offset} cells puts it {expected:.0f} m away. Refused rather "
                f"than scored on shifted cells")
        self.offset = offset
        return offset


def _interior(grid, boundary_width_cells: int, rim_m: float) -> np.ndarray:
    from woof.verify.obs import battery
    return battery.interior_mask(grid.shape, boundary_width_cells=boundary_width_cells,
                                 rim_m=rim_m, dx_m=grid.dx_m)


def radar_scores(registration, model, obs, case: planmod.Case, leads: Sequence[int],
                 grid, scored: np.ndarray, plan_cache: dict) -> tuple[list[dict], dict]:
    """Battery FSS/contingency per lead, plus RMSE on the same cells."""
    from woof.verify.obs import battery, fss, regrid
    provenance: list = []
    result = battery.score_reflectivity(
        registration=registration, model=model, obs_source=obs, init_time=seam(case.t0),
        lead_hours=list(leads), grid=grid, scored_region=scored,
        collected_provenance=provenance)
    parameters = registration["parameters"]["reflectivity"]
    rows = []
    for lead in result["leads"]:
        valid = lead["valid_time"]
        records = {(float(r["threshold_obs"]), int(r["half_width_cells"])): r for r in lead["fss"]}
        tables = {float(t["threshold"]): t for t in lead["contingency"]}
        observed = obs.field(valid)
        if "plan" not in plan_cache:
            plan_cache["plan"] = regrid.build_plan(
                source_latitude=observed.latitude, source_longitude=observed.longitude,
                destination_latitude=grid.latitude, destination_longitude=grid.longitude,
                method=str(parameters["regrid_method"]),
                max_distance_m=float(parameters["regrid_max_distance_m"]))
        obs_values, obs_valid = regrid.apply_plan(plan_cache["plan"], observed.values, observed.valid)
        forecast = np.asarray(model.composite_reflectivity(valid), dtype=np.float64)
        cells = fss.shared_validity(obs_valid, np.isfinite(forecast)) & scored
        diff = np.maximum(forecast[cells], 0.0) - np.maximum(obs_values[cells], 0.0)
        rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else None

        def fss_at(threshold, width):
            record = records.get((threshold, width))
            return None if record is None else record["fss"]

        def ratio(threshold):
            table = tables.get(threshold)
            return None if table is None else table.get("frequency_bias")

        rows.append({
            "valid": valid + "Z", "fss35_9km": fss_at(35.0, 1), "fss35_27km": fss_at(35.0, 4),
            "fss35_63km": fss_at(35.0, 10), "fss20_27km": fss_at(20.0, 4),
            "area_ratio35": ratio(35.0), "area_ratio20": ratio(20.0), "rmse_dbz": rmse,
            "scored_cells": int(cells.sum()),
            "observation_sha256": lead["observation_sha256"],
        })
    return rows, result


# -- stations ----------------------------------------------------------------

def _run_request(scoreboard: str, request: Mapping[str, object], path: Path) -> dict:
    path.write_text(json.dumps(request, indent=1), encoding="utf-8")
    subprocess.run([scoreboard, "--request", str(path)], check=True, capture_output=True)
    output = Path(str(request["output"]))
    return json.loads(output.read_text(encoding="utf-8"))


def frozen_stations(work: Path, case: planmod.Case, grid, sample_frame: Path, scoreboard: str,
                    asos: str, screen_m: float) -> list[dict]:
    """The case's one station list: inside the grid, terrain within the screen."""
    work.mkdir(parents=True, exist_ok=True)
    frozen = work / "stations-frozen.json"
    if frozen.is_file():
        return json.loads(frozen.read_text(encoding="utf-8"))
    lat, lon = grid.latitude, grid.longitude
    from woof.obs.surface_networks import networks_for_domain
    west, south, east, north = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    bbox = f"{west:.3f},{south:.3f},{east:.3f},{north:.3f}"
    # Every network whose extent meets the grid, from the frozen table: a
    # grid reaching Canada or Mexico keeps those stations, and one no
    # network reaches is refused by name.
    networks = ",".join(networks_for_domain(west, south, east, north))
    table = work / "stations-archive.json"
    subprocess.run([asos, "stations", "--networks", networks, "--bbox", bbox,
                    "--out", str(table)], check=True)
    doc = json.loads(table.read_text(encoding="utf-8"))
    rows = doc.get("stations", doc) if isinstance(doc, Mapping) else doc
    stations = [{"station_id": str(row.get("station_id", row.get("id"))),
                 "lat": float(row.get("latitude", row.get("lat"))),
                 "lon": float(row.get("longitude", row.get("lon"))),
                 "elevation_m": float(row["elevation_m"])} for row in rows]
    probe = _run_request(scoreboard, {
        "action": "extract", "method_id": "scoreboard-v2", "format": "wrf",
        "input": str(sample_frame), "valid_time": _scoreboard_time(_frame_valid(sample_frame)),
        "stations": stations, "output": str(work / "terrain-probe.json")},
        work / "terrain-probe.request.json")
    terrain = {str(point["station_id"]): point.get("terrain_m") for point in probe.get("points", [])}
    keep = [row for row in stations
            if terrain.get(row["station_id"]) is not None
            and abs(float(terrain[row["station_id"]]) - row["elevation_m"]) <= screen_m]
    frozen.write_text(json.dumps(keep, indent=1), encoding="utf-8")
    return keep


def sheet_stations(work: Path, case: planmod.Case, asos: str) -> Path | None:
    """The decoded station record the surface sheets draw their dots from."""
    out = work / "obs-sheets.json"
    if out.is_file():
        return out
    archive = work / "stations-archive.json"
    csv = work / "iem-sheets.csv"
    start, end = case.t0 + timedelta(minutes=45), case.t0 + timedelta(hours=6, minutes=15)
    for attempt in range(4):   # the IEM archive drops connections a few times an hour
        if subprocess.run([asos, "fetch", "--stations", str(archive), "--start",
                           _scoreboard_time(start), "--end", _scoreboard_time(end),
                           "--out", str(csv)]).returncode == 0:
            break
    else:
        return None
    done = subprocess.run([asos, "decode", "--obs", str(csv), "--stations", str(archive),
                           "--start", _scoreboard_time(case.t0 + timedelta(hours=1)),
                           "--end", _scoreboard_time(case.t0 + timedelta(hours=6)),
                           "--out", str(out)])
    return out if done.returncode == 0 and out.is_file() else None


def _frame_valid(path: Path) -> datetime:
    day, hh, mm, ss = FRAME_RE.match(path.name).groups()
    return datetime.fromisoformat(f"{day}T{hh}:{mm}:{ss}").replace(tzinfo=timezone.utc)


def _scoreboard_time(time: datetime) -> str:
    return time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def station_scores(work: Path, case: planmod.Case, stations: list[dict], valid: datetime,
                   arm_label: str, forecast_request: Mapping[str, object], scoreboard: str,
                   asos: str) -> dict[str, float | None]:
    """RMSE per station variable for one arm (or HRRR) at one valid time."""
    work.mkdir(parents=True, exist_ok=True)
    stamp_ = planmod.stamp(valid)
    observations = work / f"obs-{stamp_}.json"
    if not observations.is_file():
        csv = work / f"iem-{stamp_}.csv"
        # rw_asos fetches by its own frozen table; the screened list above is
        # a subset of it, and the scoreboard scores the screened list only.
        frozen = work / "stations-archive.json"
        start, end = valid - timedelta(minutes=15), valid + timedelta(minutes=15)
        for attempt in range(4):   # the IEM archive drops connections a few times an hour
            done = subprocess.run([asos, "fetch", "--stations", str(frozen), "--start",
                                   _scoreboard_time(start), "--end", _scoreboard_time(end),
                                   "--out", str(csv)])
            if done.returncode == 0:
                break
        else:
            return {key: None for key in STATION_VARIABLES.values()}
        _run_request(scoreboard, {
            "action": "observe", "method_id": "scoreboard-v2", "format": "iem_csv",
            "input": str(csv), "valid_time": _scoreboard_time(valid), "stations": stations,
            "output": str(observations)}, work / f"obs-{stamp_}.request.json")
    forecast = work / f"fc-{arm_label}-{stamp_}.json"
    _run_request(scoreboard, dict(forecast_request, stations=stations, output=str(forecast)),
                 work / f"fc-{arm_label}-{stamp_}.request.json")
    out: dict[str, float | None] = {key: None for key in STATION_VARIABLES.values()}
    for variable, key in STATION_VARIABLES.items():
        scored = _run_request(scoreboard, {
            "action": "score_observations", "method_id": "scoreboard-v2",
            "forecast": str(forecast), "observations": str(observations),
            "variable": variable, "verification_kind": "point", "observation_source": "surface",
            "output": str(work / f"score-{arm_label}-{variable}-{stamp_}.json")},
            work / f"score-{arm_label}-{variable}-{stamp_}.request.json")
        out[key] = scoreboard_rmse(scored, variable)
    return out


def scoreboard_rmse(scored: Mapping[str, object], variable: str) -> float | None:
    """The RMSE of one variable from a scoreboard-v2 ``score_observations`` result.

    The scoreboard states it in ``summary[].woof.rmse``; its ``metrics[]``
    rows carry the raw sums (``woof`` is ``{count, sumSquaredError, ...}``,
    not a number), so the RMSE is read from the summary and, failing that,
    computed from those sums.  Reading ``metrics[].woof`` as the value would
    put a dict where the decision expects kelvin.
    """
    for row in scored.get("summary", []) or []:
        if STATION_ALIASES.get(str(row.get("variable"))) == variable and isinstance(row.get("woof"), Mapping):
            value = row["woof"].get("rmse")
            return None if value is None else float(value)
    for metric in scored.get("metrics", []) or []:
        if metric.get("metric") != "rmse" or STATION_ALIASES.get(str(metric.get("variable"))) != variable:
            continue
        sums = metric.get("woof")
        if isinstance(sums, Mapping) and sums.get("count"):
            return math.sqrt(float(sums["sumSquaredError"]) / float(sums["count"]))
        if isinstance(sums, (int, float)):
            return float(sums)
    return None


# -- the case ---------------------------------------------------------------

def arm_lead_classes(root: Path, case: planmod.Case, arm: planmod.Arm) -> list[str]:
    path = planmod.arm_dir(root, case, arm) / "heating-classes.json"
    if not arm.heated:
        return []
    if not path.is_file():
        raise SystemExit(f"{arm.id}: {path} is missing; run_arm.sh writes it before the run")
    return list(json.loads(path.read_text(encoding="utf-8"))["classes"])


def score_case(root: Path, case: planmod.Case, plan: str, arm_ids: Sequence[str] | None,
               stations_on: bool, radar_on: bool, commit: str) -> dict:
    from woof.verify.obs.nowcast import CachedCompositeSource
    from woof.obs.mrms_fetch import MrmsCompositeCache
    from woof.verify.obs import registration as reg

    table = planmod.load_arms()
    arms = [arm for arm in table.for_case(case, plan) if not arm_ids or arm.id in arm_ids]
    case_root = planmod.case_dir(root, case)
    scores_dir = case_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    a_toml = tomllib.loads((planmod.arm_dir(root, case, table.arm("A")) / "experiment.toml")
                           .read_text(encoding="utf-8"))
    dx = float(a_toml["domain"][0]["dx"])
    width = int(a_toml["shared"]["spec_zone"]) + int(a_toml["shared"]["relax_zone"])
    parameters = reg.nowcast_heat_parameters(boundary_width_cells=width)
    registration = reg.make_nowcast_heat_registration(
        evaluator_commit=commit,
        # Every row, the reserve too: it is registered before any score is
        # seen, so a reserve standing in for a case that failed G3 is scored
        # under the same registration as the rest.
        cases=[{"case_id": c.id, "init_time": seam(c.t0), "role": c.role}
               for c in planmod.load_cases()],
        arms=[{"arm_id": a.id, "column": a.column, "lead_class": a.lead_class}
              for a in table.arms],
        parameters=parameters)
    (scores_dir / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")

    runs: dict[str, object] = {}
    for arm in arms:
        out = planmod.arm_dir(root, case, arm) / "out"
        if (planmod.arm_dir(root, case, arm) / "READY").is_file():
            runs[arm.id] = HistoryRun(arm.id, out, dx)
    if "A" not in runs:
        raise SystemExit(f"{case.id}: arm A has not finished; nothing is scored without the control")
    grid = runs["A"].grid()
    scored = _interior(grid, width, float(parameters["reflectivity"]["interior_rim_m"]))
    leads = [int(h) for h in parameters["scored_lead_hours"]]
    lat, lon = grid.latitude, grid.longitude
    bbox = f"{lon.min() - 0.5:.3f},{lat.min() - 0.5:.3f},{lon.max() + 0.5:.3f},{lat.max() + 0.5:.3f}"
    obs = CachedCompositeSource(
        MrmsCompositeCache(case_root / "mrms-score", bbox=bbox,
                           window_seconds=reg.NOWCAST_HEAT_MATCH_SECONDS),
        match_seconds=reg.NOWCAST_HEAT_MATCH_SECONDS, coverage_floor=0.0)
    hrrr = HrrrReflectivity(case_root / "reference", case.t0, grid, scores_dir / "tmp")
    sources: dict[str, object] = dict(runs)
    sources[HRRR_LABEL] = hrrr
    plan_cache: dict = {}
    scoreboard = binary("rw_scoreboard", "WOOF_RW_SCOREBOARD") if stations_on else ""
    asos = binary("rw_asos", "WOOF_RW_ASOS") if stations_on else ""
    stations: list[dict] = []
    if stations_on:
        first = runs["A"].frame(runs["A"].valid_times()[0])
        stations = frozen_stations(scores_dir / "stations", case, grid, first, scoreboard, asos,
                                   reg.NOWCAST_HEAT_STATION_ELEVATION_SCREEN_M)
        sheet_stations(scores_dir / "stations", case, asos)

    arms_out: dict[str, object] = {}
    full: dict[str, object] = {}
    for label, source in sources.items():
        if label == HRRR_LABEL:
            available = [h for h in leads if hrrr.file_for(seam(case.t0 + timedelta(hours=h))).is_file()]
            column, classes = HRRR_LABEL, []
        else:
            arm = table.arm(label)
            have = set(source.valid_times())
            available = [h for h in leads if seam(case.t0 + timedelta(hours=h)) in have]
            column, classes = arm.column, arm_lead_classes(root, case, arm)
        rows: list[dict] = []
        if radar_on and available:
            rows, full[label] = radar_scores(registration, source, obs, case, available, grid,
                                             scored, plan_cache)
        else:
            rows = [{"valid": seam(case.t0 + timedelta(hours=h)) + "Z"} for h in available]
        if stations_on:
            for row in rows:
                valid = planmod.utc(row["valid"])
                if label == HRRR_LABEL:
                    lead = int((valid - case.t0).total_seconds() // 3600)
                    request = {"action": "extract", "method_id": "scoreboard-v2", "format": "grib2",
                               "input": str(hrrr.file_for(seam(valid))), "model": "hrrr",
                               "comparison_kind": "forecast", "init_time": _scoreboard_time(case.t0),
                               "valid_time": _scoreboard_time(valid), "forecast_hour": lead}
                else:
                    request = {"action": "extract", "method_id": "scoreboard-v2", "format": "wrf",
                               "input": str(source.frame(seam(valid))),
                               "valid_time": _scoreboard_time(valid)}
                row.update(station_scores(scores_dir / "stations", case, stations, valid, label,
                                          request, scoreboard, asos))
        arms_out[label] = {"column": column, "lead_classes": classes,
                           "by_lead": decide.key_by_valid(planmod.iso(case.t0), rows)}
    document = {
        "schema": decide.TABLE_SCHEMA, "case_id": case.id, "role": case.role,
        "issue_time": planmod.iso(case.t0), "plan": plan,
        "registration_sha256": registration["registration_sha256"],
        "rule_status": registration["rule_status"],
        "station_count": len(stations), "arms": arms_out,
    }
    (scores_dir / "table.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    (scores_dir / "battery.json").write_text(json.dumps(full, indent=1, default=str), encoding="utf-8")
    return document


def score_family_frame(request: Mapping, *, matcher: Path, request_path: Path,
                       product_dir: Path) -> dict:
    """Score complete AI/physics families through the native gridded matcher.

    This uses the ensemble registration (20/35/45 dBZ, 33 km neighbourhood,
    causal MRMS), rather than changing the heating decision's registration.
    Python handles metadata only; Rust reads, remaps and scores every field.
    """
    if request.get("quantity") != "composite_reflectivity":
        raise ValueError("unsupported gridded campaign quantity")
    request_path.write_text(json.dumps(dict(request), indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
    result = subprocess.run([str(matcher), "match-grid", str(request_path), str(product_dir)],
                            check=True, text=True, capture_output=True)
    receipt = json.loads(result.stdout)
    if (receipt.get("schema") != "gpuwm-ensemble-grid-match.v1"
            or receipt.get("status") != "complete"
            or receipt.get("quantity") != request["quantity"]
            or receipt.get("valid_time") != request["valid_time"]):
        raise ValueError("native gridded matcher did not complete the requested frame")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.heat_eval.score_case")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case")
    mode.add_argument("--grid-request", type=Path, help="native ensemble family frame request")
    parser.add_argument("--matcher", type=Path)
    parser.add_argument("--out", type=Path, help="new family score output directory")
    parser.add_argument("--root", type=Path, default=planmod.DEFAULT_ROOT)
    parser.add_argument("--plan", default="core", choices=planmod.PLANS)
    parser.add_argument("--arms", default="")
    parser.add_argument("--no-stations", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--commit", default=os.environ.get("NH_ENGINE_COMMIT", ""),
                        help="the evaluating engine commit (40 hex); default $NH_ENGINE_COMMIT")
    args = parser.parse_args(argv)
    if args.grid_request:
        if args.matcher is None or args.out is None:
            parser.error("--grid-request needs --matcher and --out")
        args.out.mkdir(parents=True, exist_ok=False)
        document = score_family_frame(json.loads(args.grid_request.read_text()),
            matcher=args.matcher, request_path=args.out / "request.json",
            product_dir=args.out / "products")
        (args.out / "receipt.json").write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps(document))
        return 0
    case = planmod.case_by_id(args.case)
    document = score_case(args.root, case, args.plan,
                          [a for a in args.arms.split(",") if a] or None,
                          not args.no_stations, not args.no_radar, args.commit)
    print(json.dumps({label: sorted(arm["by_lead"]) for label, arm in document["arms"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
