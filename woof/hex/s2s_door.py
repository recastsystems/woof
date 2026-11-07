"""Persistent multi-week global ensemble orchestration.

The plan is a public interface: it records each member's actual initialization
cycle, common verification window, seed and restart chain before any GPU work.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from .errors import MpasPortError

SCHEMA = "gpuwm-hex.s2s/v1"
STAGES = ("mesh", "static", "init", "surface", "forecast", "export", "products", "diagnostics", "verify")
# Products need a complete week (week-1 is days 0-7); verification covers the
# first ten days, or the whole run when it is shorter. No upper bound.
MIN_DAYS = 7
MAX_MEMBERS = 1000
DURATION_HELP = (f"forecast days ({MIN_DAYS} or more); the minimum is the week-1 product window; verification covers "
                 "the first ten days or the whole shorter run; shorter experiments are not seasonal-skill qualification")


class S2SError(MpasPortError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def cores() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    try:
        budget = int(os.environ.get("RAYON_NUM_THREADS", available))
    except ValueError:
        raise S2SError("RAYON_NUM_THREADS must be a positive integer") from None
    if budget < 1:
        raise S2SError("RAYON_NUM_THREADS must be a positive integer")
    return min(available, budget)


def lag_hours(text: str | None, count: int, source: str) -> list[int]:
    from .init_sources import load_table, resolve_source
    row = resolve_source(load_table(), source)
    published = sorted(row.get("posting", {}).get("cycle_hours", range(24)))
    if text is None:
        gaps = [(b - a) % 24 or 24 for a, b in zip(published, published[1:] + published[:1])]
        stride = 6 if not row.get("posting") else min(gaps)
        return [0] * count if row.get("members") else [stride * n for n in range(count)]
    try:
        values = [int(item) for item in text.split(",")]
    except ValueError:
        raise S2SError("--lagged needs nonnegative initialization offsets in hours, e.g. 0,6,12,18") from None
    offsets = {(a - b) % 24 for a in published for b in published}
    if not values or any(n < 0 or n % 24 not in offsets for n in values):
        raise S2SError(f"--lagged offsets must be nonnegative and connect published source hours {published}")
    if len(values) > count or len(set(values)) != len(values):
        raise S2SError("--lagged must list unique offsets, no more than the total member count")
    return [values[index % len(values)] for index in range(count)]


def plan(args: argparse.Namespace) -> dict[str, Any]:
    from .global_door import plan as global_plan, _cycle
    if args.days < MIN_DAYS:
        # The shortest S2S product window is week-1 (days 0-7); a shorter run
        # has no window to reduce. Shorter forecasts use `woof hex global`.
        raise S2SError(f"--days must be at least {MIN_DAYS}: the shortest S2S product window is week-1 (days 0-7)")
    if not 1 <= args.members <= MAX_MEMBERS:
        raise S2SError(f"--members must be in 1..{MAX_MEMBERS}")
    if args.hours is not None or args.history_every_minutes is not None:
        raise S2SError("s2s uses --days and --history-hours; remove --hours/--history-every-minutes")
    if args.member is not None:
        raise S2SError("s2s uses --members as the total count; remove --member")
    from .s2s_energy import energy_config
    energy = energy_config(args)
    base_args = copy.copy(args)
    base_args.members = None
    base_args.lagged = None
    base_args.hours = args.days * 24
    base_args.history_every_minutes = args.history_hours * 60
    base_args.member = None
    value = global_plan(base_args)
    if args.row not in ("global-60km", "global-120km") and not value["row"].get("s2s"):
        raise S2SError("--row must be global-60km, global-120km or a row declaring s2s=true")
    start = _cycle(args.cycle)
    from .init_sources import Cycle, load_table, resolve_source
    from .lagged_members import expand_lags, forecast_window
    source_row = resolve_source(load_table(), args.source, getattr(args, "accept_licence", ()))
    lags = lag_hours(args.lagged, args.members, args.source)
    work = Path(value["paths"]["work"])
    prepared = {}
    for entry in args.prepared_init:
        if "=" not in entry:
            raise S2SError("--prepared-init needs YYYY-MM-DDTHH[:MEMBER]=INIT.nc")
        key, path = entry.split("=", 1)
        prepared[key] = str(Path(path).expanduser().absolute())
    members = []
    lag_counts = {}
    for index, lag in enumerate(lags, 1):
        cycle = (start - timedelta(hours=lag)).strftime("%Y-%m-%dT%H")
        lag_counts[lag] = lag_counts.get(lag, 0) + 1
        native = source_row.get("members", {}).get("perturbed")
        source_member = native["first"] + lag_counts[lag] - 1 if native else None
        if native and source_member > native["last"]:
            raise S2SError("requested member count exceeds native source members across the selected lags")
        provenance = expand_lags(source_row, Cycle.parse(args.cycle), [source_member], [lag])[0]
        init_key = cycle.replace("-", "").replace("T", "") + (f"-m{source_member:03d}" if source_member else "")
        if value["vertical"]["explicit_binding"]:
            init_key += "-v" + value["vertical"]["sha256"][:12]
        init_dir = work / "inits" / init_key
        argv = list(value["commands"]["init"])
        argv[argv.index("--cycle") + 1] = cycle
        argv[argv.index("--out-dir") + 1] = str(init_dir)
        if source_member is not None:
            argv += ["--member", str(source_member)]
        if provenance["lead_hours"]:
            argv += ["--lead-hours", str(provenance["lead_hours"])]
        prepared_path = prepared.get(cycle + (f":{source_member}" if source_member else ""))
        run_id = f"mem{index:03d}"
        seed = args.seed + index
        members.append({"run_id": run_id, "member": source_member, "cycle": cycle,
                        **provenance, "provenance": provenance,
                        "lag_hours": lag, "seed": seed, "init_dir": str(init_dir),
                        "init": str(init_dir / "init.nc"), "prepared": bool(prepared_path),
                        "prepared_init": prepared_path, "init_argv": argv,
                        **forecast_window(provenance, args.days * 24),
                        "checkpoint_hours": ",".join(str(h + provenance.get("spinup_hours", 0))
                                                     for h in range(120, args.days * 24 + 1, 120)),
                        "manifest": str(work / "members" / run_id / "provenance.json")})
    from .stochastic import member_number, member_options
    try:
        stochastic_options = json.loads(args.stochastic) if args.stochastic is not None else {}
    except (TypeError, ValueError) as error:
        raise S2SError(f"invalid --stochastic JSON: {error}") from error
    if not isinstance(stochastic_options, dict):
        raise S2SError("--stochastic must be a JSON object")
    if any(key in stochastic_options for key in ("member", "seed")):
        raise S2SError("S2S stochastic member identity and seed come from the roster and --seed")
    native_members = len({member["member"] for member in members})
    for member in members:
        member["stochastic"] = member_options(member_number(member["run_id"]),
            seed=member["seed"], source_members=native_members, ensemble_size=len(members),
            **stochastic_options)
    from .global_ensemble import validate_member_bindings
    validate_member_bindings(members, S2SError)
    # Stable identity excludes device placement and invocation stage so the same
    # command can recover an interrupted run on a different card count.
    surface_start = min(m["start_time"] for m in members)
    from .surface_series_reader import SERIES_NAME
    surface_path = work / SERIES_NAME
    legacy_surface = work / "surface-series.nc"
    if not surface_path.exists() and legacy_surface.is_file():
        surface_path = legacy_surface
    surface_cycle = args.surface_cycle or (datetime.strptime(surface_start, "%Y-%m-%d_%H:%M:%S")
                                          - timedelta(hours=48)).strftime("%Y-%m-%dT%H")
    surface = {"owner": "surface-series", "consumer": "surface-update",
               "path": str(args.surface_series or surface_path),
               "start": surface_start, "end": (start + timedelta(days=args.days)).strftime("%Y-%m-%d_%H:%M:%S"),
               "source": args.surface_source, "forecast_cycle": surface_cycle if args.surface_source == "cfs" else None}
    surface["ocean_mode"] = args.ocean_mode
    from .ocean_mixed_layer import configuration
    surface["ocean_options"] = configuration(json.loads(args.ocean_options))
    if args.ocean_mode == "prognostic":
        if args.surface_series:
            raise S2SError("--surface-series requires --ocean-mode prescribed")
        surface.update(owner="prognostic-ocean", source="member initialization SST/ice", path=None, forecast_cycle=None)
    value["commands"]["forecast"] = [*value["commands"]["forecast"],
                                     *mpas_route_surface(value, surface, json.loads(args.ocean_options))]
    tc_option = {}
    if getattr(args, "tc_climate", None) is not None:
        from dataclasses import asdict
        from .tc_climate import Recipe
        recipe = Recipe.load(args.tc_climate)
        if args.history_hours != recipe.cadence_hours:
            raise S2SError("TC climate history must match the recipe cadence")
        # Track origin must include the spinup interval, not begin mid-trajectory.
        for member in members:
            member["history_from_minutes"] = 0
        tc_option = {"tc_climate": {"path": str(args.tc_climate.resolve()),
                                   "recipe": asdict(recipe)}}
    identity = {"row": args.row, "source": args.source, "cycle": args.cycle,
                "days": args.days, "history_hours": args.history_hours, "members": members,
                "surface": surface, "energy": energy,
                "forecast_argv": value["commands"]["forecast"], **tc_option}
    identity["forecast_argv"] = [x for x in identity["forecast_argv"]]
    argv = identity["forecast_argv"]
    if "--samehost-devices" in argv:
        at = argv.index("--samehost-devices")
        del argv[at:at + 2]
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {**value, **tc_option, "schema": SCHEMA, "signature": signature, "days": args.days,
            "members": members, "stages": list(STAGES), "surface_updating": True,
            "stochastic": {"sppt": members[0]["stochastic"]["sppt"],
                           "skeb": members[0]["stochastic"]["skeb"], "seed_base": args.seed,
                           "owner": "stochastic-physics"},
            "surface": surface, "energy": energy,
            "product_options": {"regions": getattr(args, "product_region", []),
                                "climatology": str(args.climatology) if args.climatology else None,
                                "research_data_license": getattr(args, "research_data_license", None)},
            "parallelism": {"mesh": {"backend": "Rust", "threads": cores()},
                            "static": {"backend": "Rust", "threads": cores()},
                            "init": {"backend": "parallel member processes / Rust threads", "cores": cores()},
                            "surface": {"backend": "Rust", "threads": cores()},
                            "forecast": {"backend": "GPU", "members_per_card": args.members_per_card,
                                         "devices": args.devices or "all visible"},
                            "export": {"backend": "Rust", "threads": cores(), "scope": "every bound member"},
                            "products": {"backend": "GPU reductions / Rust renderer", "threads": cores()},
                            "diagnostics": {"backend": "Rust / Rayon / rw_wrfbatch", "threads": cores()},
                            "verify": {"backend": "Rust weekly ERA5 / legacy observation scorecard",
                                       "coverage_days": args.days if args.s2s_observations else min(10, args.days)}}}


def visible_cards(requested: str | None, environment=None) -> list[str]:
    from .global_ensemble import forecast_devices
    environment = os.environ if environment is None else environment
    if not requested and "CUDA_VISIBLE_DEVICES" not in environment:
        result = subprocess.run([*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True)
        requested = ",".join(result.stdout.splitlines())
        if not requested:
            raise S2SError("nvidia-smi reports no GPU cards to launch")
    return forecast_devices(requested, environment, S2SError)


def latest_checkpoint(member: dict, attempts: list[dict]) -> Path | None:
    candidates = []
    end = (datetime.strptime(member["start_time"], "%Y-%m-%d_%H:%M:%S")
           + timedelta(hours=member["hours"])) if "start_time" in member and "hours" in member else None
    for attempt in attempts:
        root = Path(attempt["out"]) / member["run_id"]
        for path in root.glob("checkpoint.*.pkl"):
            try:
                stamp = datetime.strptime(path.name, "checkpoint.%Y-%m-%d_%H.%M.%S.pkl")
            except ValueError:
                # The MPAS suite route names its checkpoint by lead (checkpoint.f024.pkl, or
                # checkpoint.t00005400.pkl off the hour; mpasphys.checkpoint_v842), from the member's start.
                lead = path.name[len("checkpoint."):-len(".pkl")]
                if "start_time" not in member or not lead[1:].isdigit() or lead[:1] not in ("f", "t"):
                    continue
                seconds = int(lead[1:]) * (3600 if lead[0] == "f" else 1)
                stamp = datetime.strptime(member["start_time"], "%Y-%m-%d_%H:%M:%S") + timedelta(seconds=seconds)
            # A killed atomic write leaves .part, never a completed checkpoint.
            # A kill after the final checkpoint but before the completion
            # receipt still needs a restartable, strictly earlier checkpoint.
            if path.stat().st_size and (end is None or stamp < end):
                candidates.append((stamp, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None



def update_manifests(value: dict, state: dict) -> None:
    for member in value["members"]:
        history = [{key: attempt.get(key) for key in ("out", "card", "roster", "log", "state", "resumed", "returncode")}
                   for attempt in state["attempts"] if member["run_id"] in attempt["members"]]
        write_json(Path(member["manifest"]), {"schema": SCHEMA, "source": value["source"], **member,
            "configuration_sha256": value["signature"], "window_start": value["start_time"], "window_days": value["days"],
            "mesh": value["mesh"], "physics": value["physics"],
            "stochastic": member["stochastic"], "stochastic_policy": value["stochastic"],
            "surface": value["surface"], "attempts": history, "complete": member["run_id"] in state["completed"]})


def _inits(value: dict, args, env: dict, status) -> dict:
    from .global_ensemble import _cli, _init_done, init_jobs
    unique = {member["init"]: member for member in value["members"]}
    # Validate every binding before deduplicating builds. A reused path must
    # not hide an earlier member's mismatching source receipt.
    for member in value["members"]:
        if member["prepared"]:
            retained = Path(member["init"])
            source = Path(member.get("prepared_init") or member["init"])
            init = source if source.is_file() else retained
            receipt_path = init.parent / "init-source-receipt.json"
            try:
                receipt = json.loads(receipt_path.read_text())
            except (OSError, ValueError) as error:
                raise S2SError(f"prepared init requires its source receipt {receipt_path}: {error}") from error
            from .init_sources import Cycle
            def same_cycle(actual, expected):
                try:
                    return Cycle.parse(actual).start_time == expected
                except (MpasPortError, TypeError):
                    return False
            if (not (source.is_file() or init.is_file()) or receipt.get("source") != value["source"]
                    or not same_cycle(receipt.get("source_cycle", receipt.get("cycle")), member["source_cycle"])
                    or receipt.get("lead_hours", 0) != member["lead_hours"]
                    or not same_cycle(receipt.get("valid_time", receipt.get("cycle")), member["valid_time"])
                    or receipt.get("member") != member["member"]):
                raise S2SError(f"prepared init source/cycle/member mismatch: {receipt_path}")
            if value.get("vertical", {}).get("explicit_binding"):
                provenance_path = init.with_name(init.name + ".provenance.json")
                try:
                    provenance = json.loads(provenance_path.read_text())
                except (OSError, ValueError) as error:
                    raise S2SError(f"prepared init requires vertical provenance {provenance_path}: {error}") from error
                actual = provenance.get("vertical_source", {}).get("summary", {}).get(
                    "vertical_spec_canonical_sha256")
                if actual != value["vertical"]["sha256"]:
                    raise S2SError(f"prepared init vertical spec mismatch: {provenance_path}")
            # Keep one inode alive through final authority checks and restart.
            # A producer may dispose of its own input after its forecast ends.
            if init != retained:
                retained.parent.mkdir(parents=True, exist_ok=True)
                if retained.exists():
                    if not os.path.samefile(init, retained):
                        raise S2SError(f"retained prepared init differs from selected source: {retained}")
                else:
                    try:
                        os.link(init, retained)
                    except OSError as error:
                        raise S2SError(f"cannot retain prepared init with a hard link: {init} -> {retained}: {error}; "
                                       "stage the shared prepared input on the work directory filesystem") from error
                write_json(retained.parent / "init-source-receipt.json", receipt)
    todo = [member for member in unique.values()
            if not member["prepared"] and not _init_done(member, value)]
    jobs = init_jobs(args.init_jobs, value, max(1, len(todo)))
    threads = max(1, cores() // jobs)
    status.stage("init", step_count=len(todo))
    def build(member):
        log = Path(member["init_dir"]).with_suffix(".log")
        rc = _cli(member["init_argv"], log, {**env, "RAYON_NUM_THREADS": str(threads),
                                           "OMP_NUM_THREADS": str(threads)})
        if rc:
            raise S2SError(f"{member['run_id']} init exited {rc}; log {log}")
        return member["run_id"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for done, run_id in enumerate(pool.map(build, todo), 1):
            status.step(done, note=f"{run_id} init ready")
    return {"built": len(todo), "reused": len(unique) - len(todo), "processes": jobs,
            "threads_per_process": threads, "log_pattern": str(Path(value["paths"]["work"]) / "inits/*.log")}


def mpas_route_surface(value: dict, surface: dict, supplied_ocean: dict) -> list[str]:
    """The forecast flags that give an MPAS-physics-route member its ocean; none on the WOOF route.

    THE BREAKAGE THIS PREVENTS (U-H1, best-180 smoke j1 and sst-boundary hxc tr28, 2026-10-26/27): the door's
    ocean reached the forecast only as WOOF_HEX_SURFACE_SERIES / WOOF_HEX_OCEAN_MODE, which only the WOOF
    route reads (woof.hex.arwen_surface_update).  On the MPAS route, the hex default, config_sst_update stayed
    false and SST and sea ice were bit-identical from day 0 to day 7 with --ocean-mode prescribed
    --surface-series, and with the prognostic default.  So on that route:

    * prescribed: MPAS's own surface update (config_sst_update, on the GPU) reads the series through the
      surface stream (--stream-input surface=SERIES; the default streams take the series' record interval);
    * prognostic: MPAS's own 1-D ocean mixed layer (config_oml1d, mpas_atmphys_driver_oml.F, on the GPU) at
      the door's mixed-layer depth and deep-water lapse rate.  It evolves open-water SST; sea ice keeps its
      initial state, since that scheme carries no ice model.
    """
    from .physics_suites import mpas_suite_names
    physics = value.get("physics") or {}
    if physics.get("suite") not in mpas_suite_names() and not physics.get("namelist"):
        return []
    argv = value["commands"]["forecast"]
    if surface["ocean_mode"] == "prescribed":
        series = str(Path(surface["path"]).absolute())
        surface.update(forecast_route="mpas", consumer="MPAS surface update (config_sst_update) on the GPU")
        flags = ["--physics", "config_sst_update=.true."]
        bound = [item.split("=", 1)[1] for at, item in enumerate(argv) if item.startswith("surface=")
                 and at and argv[at - 1] == "--stream-input"]
        bound += [item.split("=", 2)[2] for item in argv if item.startswith("--stream-input=surface=")]
        if any(Path(path).absolute() != Path(series) for path in bound):
            raise S2SError(f"--forecast-arg binds the surface stream to {bound[0]}, but the ocean series is {series}; "
                           "give the series once, as --surface-series")
        return flags + ([] if bound else ["--stream-input", f"surface={series}"])
    unread = sorted(set(supplied_ocean) - {"depth_m", "stratification_k_m"})
    if unread:
        raise S2SError("--ocean-options " + ", ".join(unread) + " set the WOOF-route ocean's sea ice and radiation; "
                       "the MPAS physics route's ocean (MPAS OML1D) has no such parameter, so they would be accepted "
                       "and never read.  Use --physics-row arwen for them")
    options = surface["ocean_options"]
    surface.update(forecast_route="mpas", ocean_model="MPAS OML1D (config_oml1d) on the GPU",
                   sea_ice="held at its initial state (OML1D carries no ice model)")
    return ["--physics", "config_oml1d=.true.", "--physics", f"config_oml_hml0={float(options['depth_m'])!r}",
            "--physics", f"config_oml_gamma={float(options['stratification_k_m'])!r}"]


def surface_stage(value: dict, args, env: dict) -> dict:
    """Named producer hook, kept separate from the live forecast consumer."""
    if value["surface"].get("ocean_mode") == "prognostic":
        return {"mode": "prognostic", "source": "member initialization SST/ice",
                "future_surface_records": 0, "ocean_options": value["surface"]["ocean_options"]}
    target = Path(value["surface"]["path"])
    if args.surface_series:
        if not target.is_file():
            raise S2SError(f"--surface-series does not exist: {target}")
        return {"path": str(target), "supplied": True}
    from .global_ensemble import _cli
    from .s2s_energy import check_forcing
    surface = value["surface"]
    receipt = target.with_suffix(".receipt.json")
    if target.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("status") == "complete" and saved.get("s2s_surface_plan") == surface:
            if surface.get("forecast_cycle") and not saved.get("object_coverage"):
                saved["object_coverage"] = check_forcing(value)
                write_json(receipt, saved)
            return {"path": str(target), "receipt": str(receipt), "reused": True,
                    "object_coverage": saved.get("object_coverage")}
        raise S2SError(f"surface series belongs to another configuration: {receipt}")
    coverage = check_forcing(value) if value["surface"].get("forecast_cycle") else None
    # The producer refuses an existing output/receipt. A private destination
    # per invocation lets interruption recovery restart acquisition safely.
    build = target.with_name(f"{target.stem}.build-{time.time_ns()}{target.suffix}")
    build_receipt = build.with_suffix(".receipt.json")
    start = datetime.strptime(surface["start"], "%Y-%m-%d_%H:%M:%S")
    end = datetime.strptime(surface["end"], "%Y-%m-%d_%H:%M:%S")
    argv = ["surface-series", "--source", surface["source"], "--cycle", surface["start"],
            "--hours", str(int((end - start).total_seconds() / 3600)), "--static", value["paths"]["static"],
            "--out", str(build), "--receipt", str(build_receipt), "--threads", str(cores()),
            "--jobs", str(min(cores(), 16)), "--interval-hours", "6"]
    if surface["forecast_cycle"]:
        argv += ["--forecast-cycle", surface["forecast_cycle"]]
    log = build.with_suffix(".log")
    rc = _cli(argv, log, env)
    if rc:
        raise S2SError(f"surface-series exited {rc}; {log}")
    saved = json.loads(build_receipt.read_text())
    write_json(receipt, {**saved, "s2s_surface_plan": surface, "object_coverage": coverage})
    build.replace(target)
    return {"path": str(target), "receipt": str(receipt), "argv": argv, "log": str(log), "reused": False, "object_coverage": coverage}


def science_arguments(value: dict) -> list[str]:
    # The surface consumer discovers its fixed native filename beside members.
    from .arwen_surface_update import attach_host  # noqa: F401
    from . import cli
    parser = cli.build_parser()
    commands = next(action.choices for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    flags = commands["forecast"]._option_string_actions
    required = ("--stochastic",)
    missing = [flag for flag in required if flag not in flags]
    if missing:
        raise S2SError("S2S science hooks missing: " + ", ".join(missing)
                       + "; integrate hex/s2s-surface-update and hex/s2s-stochastic")
    extra = ["--stochastic", json.dumps({key: value["stochastic"][key] for key in ("sppt", "skeb")})]
    if value.get("tc_climate") is not None:
        from dataclasses import asdict
        from .tc_climate import Recipe
        if asdict(Recipe.load(value["tc_climate"]["path"])) != value["tc_climate"]["recipe"]:
            raise S2SError("TC recipe changed after planning")
        extra += ["--tc-climate", value["tc_climate"]["path"]]
    return extra


def surface_environment(value: dict) -> dict:
    """Select the series without pre-creating a forecast's fresh output."""
    surface = value["surface"]
    if surface.get("ocean_mode") == "prognostic":
        return {"WOOF_HEX_OCEAN_MODE": "prognostic",
                "WOOF_HEX_OCEAN_OPTIONS": json.dumps({k: v for k, v in surface["ocean_options"].items() if k != "identity"})}
    source = Path(value["surface"]["path"]).resolve(strict=True)
    return {"WOOF_HEX_SURFACE_SERIES": str(source), "WOOF_HEX_OCEAN_MODE": "prescribed"}


def forecasts(value: dict, args, env: dict, state: dict, state_path: Path, status) -> dict:
    from .global_ensemble import forecast_argv, group_limit, _cli
    work = Path(value["paths"]["work"])
    limit = group_limit(value, args)
    cards = visible_cards(args.devices)
    extra = science_arguments(value)
    remaining = [member for member in value["members"] if member["run_id"] not in state["completed"]]
    groups = [remaining[at:at + limit] for at in range(0, len(remaining), limit)]
    # One serial queue per card, different cards run concurrently. State writes
    # stay on the coordinator thread, including before a process is launched.
    status.stage("forecast", step_count=len(remaining))
    pending = iter(groups)
    active = {}
    done = 0
    def launch(pool, card, group):
        number = len(state["attempts"]) + 1
        out = work / "attempts" / f"a{number:04d}"
        scratch = work / "scratch" / f"a{number:04d}"
        out.parent.mkdir(parents=True, exist_ok=True)
        scratch.parent.mkdir(parents=True, exist_ok=True)
        roster = out.with_suffix(".roster.json")
        records = []
        for member in group:
            record = {key: member[key] for key in ("run_id", "init", "start_time", "hours",
                                                   "history_from_minutes", "checkpoint_hours")}
            record["init_source"] = json.dumps(member["provenance"], sort_keys=True)
            checkpoint = latest_checkpoint(member, state["attempts"])
            if checkpoint:
                record["restart_from"] = str(checkpoint)
            # Each private module must receive its own independent seed.
            record["stochastic"] = member["stochastic"]
            records.append(record)
        write_json(roster, records)
        argv = forecast_argv(value, roster, out, scratch, group[0]["init"]) + extra
        attempt = {"out": str(out), "card": card, "roster": str(roster),
                   "members": [m["run_id"] for m in group], "argv": argv,
                   "log": str(out.with_suffix(".log")), "state": "running",
                   "resumed": {r["run_id"]: r.get("restart_from") for r in records},
                   "started": time.time()}
        state["attempts"].append(attempt)
        update_manifests(value, state)
        write_json(state_path, state)
        future = pool.submit(_cli, argv, Path(attempt["log"]),
                             {**env, **surface_environment(value), "CUDA_VISIBLE_DEVICES": card})
        active[future] = attempt
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards)) as pool:
        for card in cards:
            group = next(pending, None)
            if group:
                launch(pool, card, group)
        while active:
            finished, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                attempt = active.pop(future)
                rc = future.result()
                attempt.update(returncode=rc, seconds=round(time.time() - attempt["started"], 3),
                               state="complete" if rc == 0 else "interrupted")
                if rc == 0:
                    state["completed"].extend(attempt["members"])
                    done += len(attempt["members"])
                    status.step(done, note=f"card {attempt['card']}: {len(attempt['members'])} members complete")
                update_manifests(value, state)
                write_json(state_path, state)
                if rc == 0:
                    group = next(pending, None)
                    if group:
                        launch(pool, attempt["card"], group)
    failed = [a for a in state["attempts"] if a["state"] == "interrupted"
              and any(m not in state["completed"] for m in a["members"])]
    if failed or len(state["completed"]) != len(value["members"]):
        raise S2SError("ensemble interrupted; rerun the same command to resume; logs: "
                       + ", ".join(a["log"] for a in failed))
    return {"members": len(state["completed"]), "cards": cards, "members_per_card": limit,
            "attempts": len(state["attempts"]), "restarts": sum(bool(v) for a in state["attempts"] for v in a["resumed"].values())}


def resolve_cycle(args, work: Path) -> None:
    """Resolve latest once per work directory, including after an interruption."""
    from .global_cycle import resolve_cycle_argument
    if str(args.cycle).lower() == "latest":
        saved_path = work / "s2s-plan.json"
        if saved_path.is_file():
            try:
                saved = json.loads(saved_path.read_text())
                if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
                    raise ValueError("not an S2S plan")
                cycle = saved["cycle"]
                from .global_door import _cycle
                _cycle(cycle)
            except (OSError, ValueError, KeyError, TypeError, MpasPortError) as error:
                raise S2SError(f"cannot recover --cycle latest from {saved_path}: {error}") from error
            args.cycle = cycle
        elif (work / "s2s-state.json").is_file():
            raise S2SError(f"cannot recover --cycle latest without its saved plan: {saved_path}")
    resolve_cycle_argument(args, say=lambda text: print(f"[s2s] {text}", flush=True), wait=not args.dry_run)


def run(args) -> int:
    from .file_lock import lock as _flock
    from .global_door import stage_mesh, stage_static
    from .run_status import StatusWriter
    work = Path(args.work_dir).expanduser().absolute()
    if args.dry_run:
        resolve_cycle(args, work)
        value = plan(args)
        print(json.dumps(value, indent=2))
        return 0
    work.mkdir(parents=True, exist_ok=True)
    # Hold the descriptor for the entire run; an interrupted process releases it.
    with (work / "s2s.lock").open("a") as lock:
        try:
            _flock(lock, blocking=False)
        except BlockingIOError:
            raise S2SError(f"another s2s command holds {work / 's2s.lock'}") from None
        resolve_cycle(args, work)
        value = plan(args)
        state_path = work / "s2s-state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {
            "schema": SCHEMA, "signature": value["signature"], "attempts": [], "completed": [], "stages": {}}
        if state["signature"] != value["signature"]:
            raise S2SError("work directory belongs to a different S2S configuration; choose a new --work-dir")
        write_json(work / "s2s-plan.json", value)
        for attempt in state["attempts"]:
            if attempt["state"] == "running":
                attempt["state"] = "interrupted"
        update_manifests(value, state)
        status = StatusWriter(work, "s2s", stages=STAGES, row=args.row)
        env = {"WOOF_HEX_MESH_ROWS": os.pathsep.join(filter(None, [value["paths"]["mesh_rows"],
                       os.environ.get("WOOF_HEX_MESH_ROWS")])), "RAYON_NUM_THREADS": str(cores())}
        score_config = Path(value["paths"]["score_config"])
        write_json(score_config, {**value["score_config"], "verification_owner": "s2s verify stage",
                   "verification_end_time":
                   (datetime.strptime(value["start_time"], "%Y-%m-%d_%H:%M:%S")
                    + timedelta(days=min(10, args.days))).strftime("%Y-%m-%d_%H:%M:%S")})
        env["WOOF_HEX_OBSERVATION_CONFIG"] = str(score_config)
        selected = STAGES if args.stage == "all" else (args.stage,)
        device_environment = dict(os.environ)
        try:
            for stage in selected:
                started = time.monotonic()
                status.stage(stage)
                if stage in ("export", "diagnostics", "verify"):
                    state["stages"][stage] = {"status": "running"}
                    write_json(state_path, state)
                    write_json(work / "s2s-receipt.json", state)
                if stage == "mesh":
                    result = stage_mesh(value, status)
                elif stage == "static":
                    result = stage_static(value, status)
                elif stage == "init":
                    result = _inits(value, args, env, status)
                elif stage == "surface":
                    result = surface_stage(value, args, env)
                elif stage == "forecast":
                    result = forecasts(value, args, env, state, state_path, status)
                elif stage == "export":
                    from .s2s_mjo import export_members
                    result = export_members(value, state, args, env)
                elif stage == "products":
                    # Forecast workers have private CUDA visibility. Select
                    # the product card before this coordinator first imports
                    # CuPy, including when --devices excludes physical card 0.
                    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cards(args.devices, device_environment)[0]
                    from .s2s_products import build_products
                    result = build_products(value, state, args, env)
                    from .s2s_energy import build_energy_products
                    result["energy"] = build_energy_products(value, state, args, env)
                elif stage == "diagnostics":
                    from .s2s_mjo import build_mjo
                    result = build_mjo(value, state, args, env)
                else:
                    if not args.s2s_observations:
                        os.environ["CUDA_VISIBLE_DEVICES"] = visible_cards(args.devices, device_environment)[0]
                    from .s2s_scores import verify
                    result = verify(value, state, args, env)
                state["stages"][stage] = {**result, "seconds": round(time.monotonic() - started, 3),
                    "parallelism": {**value["parallelism"][stage], **result.get("parallelism", {})}}
                write_json(state_path, state)
            state.pop("error", None)
            write_json(state_path, state)
            status.done(f"{len(value['members'])} members x {args.days} days; stages: {', '.join(selected)}")
            write_json(work / "s2s-receipt.json", state)
        except BaseException as error:
            state["error"] = f"{type(error).__name__}: {error}"
            if stage in ("export", "diagnostics", "verify"):
                state["stages"][stage] = {"status": "failed", "error": state["error"],
                                           "seconds": round(time.monotonic() - started, 3)}
            write_json(state_path, state)
            write_json(work / "s2s-receipt.json", state)
            raise
    return 0


def _member_count(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"s2s takes a total member count COUNT (1 to {MAX_MEMBERS}), not a member list: {text!r}") from None


def add_s2s_parser(commands) -> None:
    from .global_door import _add_common
    parser = commands.add_parser("s2s", help="multi-week (7 days and longer) global ensembles with persistent checkpoint recovery")
    _add_common(parser)
    # Reuse the global options and replace only S2S-specific defaults/types.
    members = parser._option_string_actions["--members"]
    members.type = _member_count
    members.default = 4
    members.metavar = "COUNT"
    members.help = f"total output member count (1..{MAX_MEMBERS}, default: 4); balanced across --lagged offsets"
    for flag in ("--row", "--cycle", "--work-dir"):
        parser._option_string_actions[flag].required = True
    parser.add_argument("--prepared-init", action="append", default=[], metavar="CYCLE[:MEMBER]=PATH",
                        help="reuse an existing init with its matching init-source-receipt.json")
    parser.add_argument("--days", type=int, required=True, help=DURATION_HELP)
    parser._option_string_actions["--lagged"].help = "offsets in hours; balance total members across them; deterministic default uses earlier published cycles"
    # S2S energy products are default-on and need 6-hourly samples (daily samples alias the solar cycle).
    parser.add_argument("--history-hours", type=int, choices=(6, 24), default=6)
    parser.add_argument("--tc-climate", type=Path, default=None, metavar="RECIPE",
                        help="opt-in per-member GPU TC trajectories and daily GPI")
    parser.add_argument("--hub-heights", default="80,100", help="comma-separated hub heights in metres AGL, (10,1000]")
    parser.add_argument("--energy-table", type=Path, help="table-driven turbine and horizontal PV proxy rows")
    parser.add_argument("--energy-climatology", type=Path, help="dated energy window climatology for historical energy anomalies")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--ocean-mode", choices=("prognostic", "prescribed"), default="prognostic",
                        help="coupled mixed-layer ocean and thermodynamic ice (default); prescribed uses dated records")
    parser.add_argument("--ocean-options", default="{}", metavar="JSON",
                        help="declared mixed-layer/sea-ice parameter overrides; recorded and checked on restart")
    parser.add_argument("--surface-series", type=Path, help="existing dated native case-8 series for --ocean-mode prescribed; otherwise build it")
    parser.add_argument("--surface-source", default="cfs",
                        help="surface forcing source; default CFS forecasts cover the whole future window")
    parser.add_argument("--surface-cycle", help="CFS forecast cycle; default 48 hours before the earliest member init")
    parser.add_argument("--climatology", type=Path, help="ERA5 window climatology on the product grid")
    parser.add_argument("--product-region", action="append", default=[], metavar="NAME:SOUTH,NORTH,WEST,EAST",
                        help="additional regional S2S maps; repeatable; west>east crosses the date line")
    parser.add_argument("--research-data-license", choices=("copernicus-products",), default="copernicus-products",
                        help="ERA5 data licence recorded in the receipt (Licence to Use Copernicus Products: commercial use permitted with attribution; default)")
    parser.add_argument("--analysis-reference", action="append", default=[], metavar="SOURCE:CYCLE=INIT.nc",
                        help="prepared gfs|era5 analysis and source receipt for direct GPU verification; GFS is fetched otherwise")
    parser.add_argument("--s2s-export-window", default="global", help="existing export door window; full fields for every member")
    parser.add_argument("--mjo-reference", type=Path, help="qualified fixed RMM/BSISO EOFs and dated backgrounds JSON")
    parser.add_argument("--s2s-observations", type=Path, help="qualified weekly ERA5 NetCDF and daily observed RMM manifest")
    parser.add_argument("--s2s-reference-license", help="explicit research opt-in for a non-permissive reference data licence")
    parser.add_argument("--dry-run", action="store_true", help="print the complete plan without downloads or GPU work")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.set_defaults(handler=run, export_field_set="full")
