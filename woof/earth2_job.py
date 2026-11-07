"""Execute a canonical job through the existing acquisition and forecast doors.

This module owns scheduling and receipts, never meteorological arrays. Each
stage is a normal public command, except the small fetch entry which preserves
run_model's richer receipt and initial-condition-source argument.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib

from woof.earth2_door import canonical_json, diagnostic_rows
from woof.earth2_run import model_table


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _gpu_sample(pids):
    """Charge only this command's processes, not another pool tenant."""
    try:
        sample = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        memory = sum(float(cols[1]) for line in sample.stdout.splitlines()
                     if len(cols := line.split(",")) == 2 and int(cols[0]) in pids)
        util = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
            capture_output=True, text=True, timeout=3)
        return memory, max(float(v) for v in util.stdout.splitlines())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None, None


def _command(stage, command, root, receipt):
    command = [str(c) for c in command]
    prefixes = json.loads(os.environ.get("WOOF_EARTH2_STAGE_PREFIX", "{}"))
    if not isinstance(prefixes, dict):
        raise ValueError("WOOF_EARTH2_STAGE_PREFIX needs an object of stage argv prefixes")
    prefix = prefixes.get(stage, [])
    if not isinstance(prefix, list) or any(not isinstance(arg, str) for arg in prefix):
        raise ValueError(f"WOOF_EARTH2_STAGE_PREFIX {stage} needs a string argv list")
    from woof.earth2_licensing import accepted_licences
    public_licence_door = (len(command) > 3 and command[:3] == [sys.executable, "-m", "woof"]
        and command[3] in ("fetch", "prep", "domain", "go", "ensemble", "earth2-diagnose"))
    internal_fetch = command[:4] == [sys.executable, "-m", "woof.earth2_job", "fetch"]
    if public_licence_door or internal_fetch:
        for licence in accepted_licences():
            command.extend(("--accept-licence", licence))
    original = list(command)
    admission = root / (stage + "-admission.json")
    if prefix:
        command = [*prefix, sys.executable, "-m", "woof.earth2_job", "stage", str(admission), *command]
    log = root / (stage + ".log")
    record = dict(stage=stage, command=[str(c) for c in command], log=str(log),
                  started=datetime.now(timezone.utc).isoformat(), peak_vram_mib=None,
                  unwrapped_command=original, scheduler_prefix=prefix,
                  cpu_stalls=[], resource_monitor="psutil process tree and nvidia-smi")
    receipt["stages"].append(record)
    _write(root / "receipt.json", receipt)
    print(f"earth2 {stage}: {shlex.join(record['command'])}", flush=True)
    start = time.monotonic()
    import psutil
    with log.open("w", encoding="utf-8") as stream:
        environment = dict(os.environ)
        environment.setdefault("WOOF_KERNEL_PLAN_DIR", str(root / "kernel-plans"))
        process = subprocess.Popen(record["command"], stdout=stream, stderr=subprocess.STDOUT, env=environment)
        seen, stalls, previous = {}, {}, time.monotonic()
        while True:
            try:
                process.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            pids = {process.pid}
            try:
                children = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
            except psutil.Error:
                children = []
            for child in children:
                pids.add(child.pid)
                try:
                    key = (child.pid, child.create_time())
                    sample = child.cpu_times()
                    cpu = sample.user + sample.system
                    old = seen.get(key)
                    seen[key] = (now, cpu)
                    percent = 100 * (cpu - old[1]) / (now - old[0]) if old else 0
                    stalls[key] = (stalls.get(key, 0) + now - previous) if 85 <= percent <= 115 else 0
                except psutil.Error:
                    continue
            memory, util = _gpu_sample(pids)
            if memory is not None:
                record["peak_vram_mib"] = max(record["peak_vram_mib"] or 0, memory)
            for key, duration in stalls.items():
                if duration >= 30 and util is not None and util <= 5:
                    record["cpu_stalls"].append(dict(pid=key[0], seconds=round(duration, 2), gpu_utilization_percent=util))
                    stalls[key] = 0
            previous = now
        ended = time.monotonic()
        record.update(returncode=process.returncode, wall_seconds=round(ended - start, 3))
        if prefix and admission.is_file():
            admitted = json.loads(admission.read_text())
            record.update(admission=admitted, wall_with_queue_seconds=record["wall_seconds"],
                          queue_wall_seconds=round(admitted["monotonic"] - start, 3),
                          wall_seconds=round(ended - admitted["monotonic"], 3))
        elif prefix:
            record.update(wall_with_queue_seconds=record["wall_seconds"],
                          queue_wall_seconds=record["wall_seconds"], wall_seconds=None)
    _write(root / "receipt.json", receipt)
    if process.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{stage} exited {process.returncode}; log {log}\n{tail}")


def _require_module(name, feature):
    if importlib.util.find_spec("woof." + name) is None:
        raise ValueError(f"{feature} needs woof.{name}; merge the owning track's implementation before running this job")


def _source_token(name):
    # The canonical JobSpec spells models without the acquisition prefix.
    return "earth2-" + name if name in model_table()["models"] else name


def _preflight(spec):
    for name in spec["diagnose"]:
        row = diagnostic_rows()[name]
        from woof.earth2_licensing import require_licence
        require_licence(name, row)
        if row.get("runtime_refusal"):
            raise ValueError(f"diagnostic {name}: {row['runtime_refusal']}")
        weights = row.get("weights", {})
        if weights.get("account_needed"):
            raise ValueError(f"diagnostic {name}: published weights require an account")
    if spec["nest_km"] is not None:
        if spec["storm"] is None:
            raise ValueError("--nest requires --storm auto, an ATCF id or a storm name")
        _require_module("tc_tracks", "storm placement")
        _require_module("cyclone_tracks", "moving nest")
        from woof.tc_tracks import binary
        binary("rw_tc_fields")
        binary("rw_wrfbatch")
    if spec["diagnose"]:
        _require_module("earth2_diagnostics", "diagnostic execution")
    if spec["export"] or spec["diagnose"]:
        _require_module("earth2_export", "Earth2Studio history export")
    if spec["ensemble"]:
        from woof.ensemble.wofs_recipe import parse_source_spec
        parse_source_spec([dict(source=_source_token(r["source"]), members=r["members"])
                           for r in spec["ensemble"]["mix"]], members=spec["ensemble"]["members"])
    if spec["nest_km"] is None and spec["area"] is None:
        raise ValueError("a regional forecast needs --area, or --storm with --nest")


def run_job(spec, out, *, earth2_input=None, geog_root=None):
    root = Path(out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "receipt.json").exists():
        raise ValueError(f"{root} already holds a job receipt; choose a new --out directory")
    spec_path = root / "job-spec.json"
    spec_path.write_text(canonical_json(spec), encoding="utf-8")
    from woof.build_provenance import package_build_identity
    identity = package_build_identity()
    receipt = dict(schema="woof-earth2-receipt-v1", spec=spec,
                   code_revision=identity["commit"], package_provenance=identity,
                   spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(), status="running", stages=[],
                   wall_times_seconds={key: None for key in ("fetch", "model", "prep", "sim", "render")},
                   peak_vram_mib=None, products={})
    from woof.earth2_licensing import accepted_licences
    receipt["accepted_licences"] = list(accepted_licences())
    from woof.cyclone_search import resolve_region
    receipt["storm_search"] = resolve_region(spec["area"], automatic=spec["storm"] == "auto")
    started = time.monotonic()
    cli = [sys.executable, "-m", "woof"]
    try:
        _preflight(spec)
        config = root / "cyclone.toml" if spec["nest_km"] is not None else root / "regional.toml"
        wps = config.with_suffix(".namelist.wps")
        source = _source_token(spec["model"])
        # The job contract uses full UTC ISO; existing public setup flags
        # deliberately accept only YYYY-MM-DDTHH.
        cycle_token = datetime.fromisoformat(spec["cycle"].replace("Z", "+00:00")).strftime("%Y-%m-%dT%H")
        history_interval = str(min(21600, spec["hours"] * 3600))
        hardware_args = ["--card", spec["card"]] if "card" in spec else []
        # --geog-root names the tree every preparation of this job reads (domain, prep, ensemble).
        geog_args = ([] if geog_root is None
                     else ["--geog-root", str(Path(geog_root).expanduser().resolve())])
        if spec["nest_km"] is None:
            south, west, north, east = spec["area"]
            polygon = root / "area.geojson"
            _write(polygon, dict(type="Polygon", coordinates=[[[west, south], [east, south], [east, north], [west, north], [west, south]]]))
            _command("setup", [*cli, "domain", "--source", source, "--cycle", cycle_token,
                "--hours", str(spec["hours"]), "--polygon", polygon, "--root-dx", "3", "--nz", "49",
                "--history-interval", history_interval, *hardware_args, *geog_args, "--out", config], root, receipt)
            planned = tomllib.loads(config.read_text())
            area = planned.get("fetch", {}).get("area")
            if area:
                area = [float(v) for v in area.split(",")] if isinstance(area, str) else area
                acquisition = dict(spec, area=area)
                acquisition_path = root / "acquisition-spec.json"
                acquisition_path.write_text(canonical_json(acquisition))
            else:
                acquisition_path = spec_path
        else:
            acquisition_path = spec_path
        fetch_command = [sys.executable, "-m", "woof.earth2_job", "fetch", acquisition_path, root / "source"]
        if earth2_input is not None:
            receipt["earth2_input"] = str(Path(earth2_input).expanduser().resolve())
            fetch_command.extend(("--earth2-input", receipt["earth2_input"]))
        _command("fetch-model", fetch_command, root, receipt)
        fetched = json.loads((root / "source" / "earth2-run.json").read_text())
        runner = fetched.get("runner", {})
        model_seconds = float(runner.get("load_seconds", 0)) + float(
            runner.get("inference_seconds", runner.get("run_seconds", 0)))
        if not model_seconds:
            model_seconds = sum(float(stage["seconds"]) for stage in runner.get("stages", []))
        receipt["wall_times_seconds"].update(model=model_seconds, fetch=max(0, fetched["wall_seconds"] - model_seconds))
        if fetched.get("cache_reuse"):
            receipt["wall_times_seconds"].update(model=0, fetch=fetched["cache_reuse"]["wall_seconds"])
        receipt["time_basis"] = {"model": "runner load plus synchronized inference when available",
            "fetch": "acquisition wall minus model; includes initial data, donor and staging tail",
            "prep_sim_render": "whole child-command wall time"}
        if fetched.get("cache_reuse"):
            receipt["time_basis"].update(model="zero: completed forecast reused; producer times retained in parent",
                                         fetch="receipt and artifact verification for cache reuse")
        receipt["parent"] = fetched
        coverage = fetched.get("cache_reuse", {}).get("coverage", {})
        if coverage.get("search_region"):
            receipt["storm_search"] = coverage["search_region"]
        receipt["products"]["forecast"] = fetched["forecast"]
        history = None
        if spec["nest_km"] is not None:
            forecasts = fetched["forecast"]
            if isinstance(forecasts, list):
                # The family's parallel writer records one file per member/lead.
                # Pass its clock and ancestry to the existing generic tracker.
                cycle = datetime.fromisoformat(spec["cycle"].replace("Z", "+00:00"))
                members = []
                for member in fetched["members"]:
                    frames = [dict(path=entry["path"], valid=(cycle + timedelta(hours=entry["lead_hours"])).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   select={"time": 0, "lead_time": 0})
                              for entry in runner["files"] if entry["member"] == member]
                    members.append(dict(member_id=str(member), family="ai", source=source, init=spec["cycle"],
                        frames=frames, vocabulary="earth2studio", ancestry=dict(model=spec["model"], parent_receipt=str(root / "source" / "earth2-run.json"))))
                manifest = root / "track-manifest.json"
                _write(manifest, dict(schema="gpuwm.tc-tracks-manifest.v1", members=members))
                track_input = ["--manifest", manifest]
            else:
                track_input = ["--input", forecasts]
            _command("track", [*cli, "tc-tracks", *track_input, "--out", root / "tracks", "--workers", "4"], root, receipt)
            # A reused regional forecast has receipt-checked job bounds even
            # when --area was omitted. Carry those to the selector, while
            # retaining the configured basin polygon for global forecasts.
            search_area = spec["area"]
            if (search_area is None and spec["storm"] == "auto"
                    and coverage.get("search_region", {}).get("id") == "forecast-area"):
                search_area = coverage["search_region"]["area"]
            search_args = (["--search-area=" + ",".join(str(v) for v in search_area)]
                           if search_area is not None else [])
            _command("setup", [*cli, "cyclone-setup", "--source", source, "--cycle", cycle_token,
                       "--hours", str(spec["hours"]), "--tracks", root / "tracks" / "tracks.json",
                       "--storm", spec["storm"], "--nest-km", str(spec["nest_km"]),
                       "--history-interval", history_interval, "--nest-history-interval", history_interval, *search_args, *hardware_args, "--out", config], root, receipt)
        if not config.is_file() or not wps.is_file():
            raise ValueError(f"setup did not publish {config} and {wps}; inspect {root / 'setup.log'} for a fit proposal or admission refusal; no proposed shrink is accepted automatically")
        from woof.earth2_fetch_cache import check_domain_coverage
        check_domain_coverage(fetched, config)
        if spec["ensemble"]:
            # Configuration and all family selection stay in the ensemble door.
            # JSON inline objects are also valid TOML inline tables after using '='.
            rows = ", ".join('{source = ' + json.dumps(_source_token(r["source"])) + ', members = ' + str(r["members"]) + '}'
                             for r in spec["ensemble"]["mix"])
            with config.open("a") as stream:
                stream.write('\n[ensemble]\nrecipe = "wofs"\nmembers = ' + str(spec["ensemble"]["members"]) + '\nsources = [' + rows + ']\n')
            _command("sim", [*cli, "ensemble", config, "--outdir", root / "ensemble", *geog_args], root, receipt)
            history = root / "ensemble"
        else:
            prepared = root / "prepared"
            handoff = root / "source" / "prep-arguments.json"
            if handoff.is_file():
                arguments = json.loads(handoff.read_text())
                if arguments.get("unbound_supplement_roles"):
                    raise ValueError(f"acquisition left unbound donor roles: {arguments['unbound_supplement_roles']}")
                prep = [*cli, "prep", *arguments["argv"]]
            else:
                forecasts = fetched["forecast"]
                if not isinstance(forecasts, str):
                    raise ValueError("multi-file acquisition must publish prep-arguments.json")
                prep = [*cli, "prep", "--source", source, "--input", forecasts,
                        "--supplement", "physical_analysis_surface_data=" + fetched["analysis"],
                        "--author-input-manifest", root / "inputs.json"]
                from woof.source_adapters import get_source_adapter
                if fetched.get("profile") != get_source_adapter(source).packaged_profile:
                    raise ValueError("acquisition donor profile differs from source registry and supplies no bound prep handoff")
            from woof.geog_assets import default_geog_root
            geography = (Path(geog_root).expanduser().resolve() if geog_root is not None
                         else default_geog_root())
            receipt["geog_root"] = str(geography)
            prep.extend(["--experiment-config", config, "--wps-namelist", wps, "--output-root", prepared,
                         "--geog-root", geography,
                         "--preprocess-backend", "cuda", "--preprocess-workers", "4"])
            if spec["nest_km"] is not None:
                prep.append("--statics-corridor")
            _command("prep", prep, root, receipt)
            history = root / "forecast"
            _command("sim", [*cli, "sim", prepared, "--experiment-config", config, "--wps-namelist", wps, "--outdir", history], root, receipt)
        # A nested job renders the child. A regional job renders its root.
        from woof.render_layout import history_frames
        domain = "d02" if spec["nest_km"] is not None else "d01"
        frames = [frame.path for frame in history_frames(history) if frame.grid == domain]
        if not frames:
            raise ValueError(f"simulation produced no {domain} history under {history}")
        _command("render", [*cli, "render", *frames, "--engine", "rust", "--products", "refl,wind10,mslp_10m_winds", "--out", root / "maps"], root, receipt)
        receipt["products"].update(history=str(history), maps=[str(p) for p in sorted((root / "maps").rglob("*.png"))])
        if spec["export"] or spec["diagnose"]:
            if history is None:
                exported = Path(fetched["forecast"])
            else:
                exported = root / "earth2-export"
                command = [*cli, "export", history, "--earth2", "e2s-latlon-0.25", "--domains", domain, "--out", exported, "--threads", "4"]
                if spec["diagnose"]:
                    command.extend(["--for", ",".join(diagnostic_rows()[d].get("consumer", d) for d in spec["diagnose"])])
                _command("export", command, root, receipt)
            receipt["products"]["earth2"] = str(exported)
            for name in spec["diagnose"]:
                _command("diagnose-" + name, [*cli, "earth2-diagnose", "--model", name, "--input", exported, "--out", root / "diagnostics" / name], root, receipt)
        receipt["status"] = "complete"
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        receipt.update(status="failed", error=str(error))
        raise RuntimeError(str(error)) from error
    finally:
        for stage in receipt["stages"]:
            if stage["stage"] in receipt["wall_times_seconds"]:
                receipt["wall_times_seconds"][stage["stage"]] = stage.get("wall_seconds")
        peaks = [s["peak_vram_mib"] for s in receipt["stages"] if s.get("peak_vram_mib") is not None]
        parent = receipt.get("parent", {})
        parent_peak = None if parent.get("cache_reuse") else parent.get("runner", {}).get("peak_gpu_gib")
        if parent_peak is not None:
            peaks.append(float(parent_peak) * 1024)
        receipt.update(wall_seconds=round(time.monotonic() - started, 3), peak_vram_mib=max(peaks) if peaks else None)
        _write(root / "receipt.json", receipt)
    print(root / "receipt.json")
    return 0


def _fetch(spec_path, out, input_dir=None):
    from woof.earth2_run import run_model
    from woof.earth2_fetch_cache import remember, reuse
    spec = json.loads(Path(spec_path).read_text())
    # The simulation may stop between a model's boundary times. Acquire the
    # next boundary too, with cadence defined only by the authority row.
    from woof.earth2_run import resolve_model
    step = int(resolve_model(spec["model"])[1]["step_hours"])
    spec["hours"] = ((spec["hours"] + step - 1) // step) * step
    if (reuse(spec, out, input_dir) if input_dir is not None else reuse(spec, out)) is not None:
        return
    run_model(model=spec["model"], cycle=datetime.fromisoformat(spec["cycle"].replace("Z", "+00:00")),
              hours=spec["hours"], area=spec["area"], out=Path(out), ic_source=spec["ic_source"])
    remember(spec, Path(out) / "earth2-run.json")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "stage":
        _write(sys.argv[2], dict(monotonic=time.monotonic(), pid=os.getpid(),
                                started=datetime.now(timezone.utc).isoformat()))
        os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
    elif len(sys.argv) >= 4 and sys.argv[1] == "fetch":
        from woof.earth2_licensing import licence_scope, _values
        with licence_scope(_values(sys.argv[4:], "--accept-licence")):
            inputs = list(_values(sys.argv[4:], "--earth2-input"))
            if len(inputs) > 1:
                raise ValueError("supply --earth2-input once")
            _fetch(sys.argv[2], sys.argv[3], inputs[0] if inputs else None)
    else:
        raise SystemExit("internal entry: fetch JOB_SPEC OUT, or stage ADMISSION COMMAND ...")
