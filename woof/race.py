"""Local forecast races and object-clock comparisons, with no field data path.

Comparison clocks are S3 LastModified and mirror directory mtime, never the
observer's poll time. Background listings watch every hour; HTTP is never on a
model step.
Source names, aliases and key grammars belong to the recipe registry.
"""
from __future__ import annotations

import argparse
import importlib.util
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, unquote
from urllib.request import urlopen
import xml.etree.ElementTree as ET

UTC = timezone.utc
REGISTRY = Path(__file__).parent / "data/recipes/registry.v1.json"


def instant(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def stamp(value=None):
    return (datetime.now(UTC) if value is None else instant(value)).isoformat().replace("+00:00", "Z")


def cycle_time(value, now=None):
    now = now or datetime.now(UTC)
    if len(value) == 3 and value.endswith("Z") and value[:2].isdigit():
        return now.replace(hour=int(value[:2]), minute=0, second=0, microsecond=0)
    result = instant(value)
    if result.minute or result.second or result.microsecond:
        raise ValueError("A race cycle must be a whole UTC hour (HHZ or YYYY-MM-DDTHH).")
    return result


def recipe_row(name, registry=REGISTRY):
    from woof.region_recipe import recipe_row as registry_row
    try:
        row = registry_row(name, registry)
    except ValueError:
        raise ValueError(f"Unknown recipe or alias: {name}") from None
    if "race" not in row:
        raise ValueError(f"Recipe {row['id']} has no race object-clock row.")
    return row


def object_names(row, cycle, hours):
    fields = {"date": cycle.strftime("%Y%m%d"), "hour": cycle.strftime("%H")}
    source = row["race"]
    return (source["prefix"].format(**fields),
            source["analysis_key"].format(**fields),
            {h: source["hour_key"].format(**fields, lead=h) for h in range(hours + 1)})


def list_objects(endpoint, prefix):
    """One narrow ListObjectsV2 prefix; pagination also works on test servers."""
    objects, token = {}, None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        with urlopen(endpoint.rstrip("/") + "/?" + urlencode(query), timeout=10) as reply:
            root = ET.fromstring(reply.read())
        for item in root.findall("{*}Contents"):
            key = item.findtext("{*}Key")
            modified = item.findtext("{*}LastModified")
            size = int(item.findtext("{*}Size", "0"))
            if key and modified and size > 0:
                objects[key] = {"key": key, "last_modified": stamp(modified), "size": size}
        token = root.findtext("{*}NextContinuationToken")
        if root.findtext("{*}IsTruncated", "false") != "true":
            return objects
        if not token:
            raise ValueError("Truncated object listing omitted its continuation token.")


def comparison_keys(row, cycle, hours, hour_keys):
    result = {h: [key] for h, key in hour_keys.items()}
    if hours == row["race"]["max_hours"]:
        fields = {"date": cycle.strftime("%Y%m%d"), "hour": cycle.strftime("%H"), "lead": hours}
        result[hours] += [key.format(**fields) for key in row["race"].get("final_comparison_keys", [])]
    return result


def mirror_url(row, cycle, override=None):
    # An explicit object-store override is a self-contained test source.
    if override:
        return None
    template = row["race"].get("mirror_directory_url")
    return template.format(date=cycle.strftime("%Y%m%d"), hour=cycle.strftime("%H")) if template else None


def list_directory(url, keys):
    with urlopen(url, timeout=10) as reply:
        body = reply.read().decode("utf-8")
    pattern = r'<a\s+href="([^"]+)"[^>]*>.*?</a>\s+(\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2})'
    entries = {unquote(html.unescape(name)): stamp(datetime.strptime(value, "%d-%b-%Y %H:%M").replace(tzinfo=UTC))
               for name, value in re.findall(pattern, body)}
    return {key: {"key": key, "nomads_mtime": entries[key.rsplit("/", 1)[-1]]}
            for key in keys if key.rsplit("/", 1)[-1] in entries}


def earliest_post(row):
    values = [row[k] for k in ("last_modified", "nomads_mtime") if row.get(k)]
    return min(values, key=instant) if values else None


class ObjectWatch:
    def __init__(self, endpoint, prefix, keys, interval=30, mirror=None):
        self.endpoint, self.prefix, self.keys = endpoint, prefix, set(keys)
        self.interval, self.mirror = interval, mirror
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.objects, self.errors = {}, []
        self.polls = 0
        self.thread = threading.Thread(target=self._run, name="race-object-watch", daemon=True)

    def _run(self):
        while not self.stop.is_set():
            sources = [("nomads", lambda: list_directory(self.mirror, self.keys))] if self.mirror else []
            sources.append(("s3", lambda: list_objects(self.endpoint, self.prefix)))
            for source, listing in sources:
                try:
                    rows = listing()
                    seen = stamp()
                    with self.lock:
                        self.polls += 1
                        for key in self.keys.intersection(rows):
                            previous = self.objects.setdefault(key, {"key": key, "first_seen_at": seen})
                            previous.setdefault(source + "_first_seen_at", seen)
                            for field, value in rows[key].items():
                                if field in previous and previous[field] != value:
                                    previous["replacement_" + field] = value
                                else:
                                    previous[field] = value
                except (OSError, ValueError, ET.ParseError) as error:
                    with self.lock:
                        self.polls += 1
                        self.errors = [*self.errors[-19:], {"at": stamp(), "source": source, "error": str(error)}]
            if self.stop.wait(self.interval):
                break

    def snapshot(self):
        with self.lock:
            return {"objects": {k: dict(v) for k, v in self.objects.items()},
                    "polls": self.polls, "errors": list(self.errors), "poll_seconds": self.interval,
                    "s3_endpoint": self.endpoint, "nomads_directory": self.mirror}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.1)  # A blocked HTTP request never holds the forecast's exit.


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def machine():
    def capture(command):
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"host": platform.node(), "cpu": cpu, "cpu_count": os.cpu_count(),
            "cards_and_driver": capture([*__import__('woof.gpu_smi', fromlist=['command']).command(), "--query-gpu=index,name,memory.total,driver_version",
                                         "--format=csv,noheader"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "engine_dirty": bool(capture(["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"])),
            "engine_sha": capture(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"])}


def dry_inputs(args):
    receipt = read_json(args.receipt) if args.receipt else {}
    values = dict(receipt.get("prediction_inputs", receipt))
    # A completed race is also a measured input, with no log prose parsing.
    if receipt.get("mode") == "live":
        if receipt.get("first_step_at") and receipt.get("launch_at"):
            values.setdefault("launch_to_first_step", (instant(receipt["first_step_at"]) - instant(receipt["launch_at"])).total_seconds())
        if receipt.get("launch_at") and receipt.get("noaa_f00_posted_at"):
            values.setdefault("launch_delay", max(0, (instant(receipt["launch_at"]) - instant(receipt["noaa_f00_posted_at"])).total_seconds()))
        hours = [h for h in receipt.get("publish_receipt", {}).get("hours", [])
                 if h.get("domain") == 1 and h.get("hour_closed_at") and float(h.get("forecast_hour", 0)) > 0]
        hours.sort(key=lambda h: float(h["forecast_hour"]))
        if len(hours) >= 2:
            first, last = hours[0], hours[-1]
            span = float(last["forecast_hour"]) - float(first["forecast_hour"])
            if span > 0:
                values.setdefault("seconds_per_hour", (instant(last["hour_closed_at"]) - instant(first["hour_closed_at"])).total_seconds() / span)
        lags = [(instant(h["online_at"]) - instant(h["hour_closed_at"])).total_seconds() for h in hours if h.get("online_at")]
        if lags:
            values.setdefault("publish_lag", max(lags))
    result = {}
    for key in ("seconds_per_hour", "launch_to_first_step", "publish_lag", "launch_delay"):
        value = getattr(args, key)
        if value is None:
            value = values.get(key)
        if value is None:
            if key == "launch_delay":
                value = 0
            else:
                raise ValueError(f"Dry mode needs --{key.replace('_', '-')} or that key in --receipt.")
        value = float(value)
        if not math.isfinite(value) or value < 0 or (key == "seconds_per_hour" and value == 0):
            raise ValueError(f"{key} must be finite and {'positive' if key == 'seconds_per_hour' else 'nonnegative'}.")
        result[key] = value
    result["receipt"] = str(args.receipt) if args.receipt else None
    return result


def posting_objects(path, cycle, keys):
    document = json.loads(Path(path).read_text())
    rows = document if isinstance(document, list) else document.get("rows", document.get("objects", []))
    if isinstance(rows, dict):
        rows = list(rows.values())
    result = {}
    for row in rows:
        if row.get("cycle") and instant(row["cycle"]) != cycle:
            continue
        key = row.get("key", row.get("file", ""))
        modified = row.get("s3_last_modified", row.get("last_modified"))
        mirror = row.get("nomads_mtime")
        if not modified and not mirror:
            continue
        for wanted in keys:
            if key == wanted or key == wanted.rsplit("/", 1)[-1]:
                result[wanted] = {"key": wanted, "last_modified": stamp(modified) if modified else None,
                                  "nomads_mtime": stamp(mirror) if mirror else None}
    return result


def compare_hours(hour_keys, objects, online, references=None):
    rows = []
    for hour, key in hour_keys.items():
        reference_keys = (references or {}).get(hour, [key])
        candidates = [(candidate, earliest_post(objects.get(candidate, {}))) for candidate in reference_keys]
        available = [(candidate, value) for candidate, value in candidates if value]
        reference, posted = min(available, key=lambda pair: instant(pair[1])) if available else (None, None)
        ours = online.get(hour)
        rows.append({"hour": hour, "noaa_key": key, "noaa_posted_at": posted,
                     "noaa_reference_key": reference,
                     "noaa_s3_posted_at": objects.get(key, {}).get("last_modified"),
                     "noaa_nomads_mtime": objects.get(key, {}).get("nomads_mtime"),
                     "reference_objects": {k: objects.get(k, {}) for k in reference_keys},
                     "our_online_at": ours,
                     "margin_seconds": (instant(posted) - instant(ours)).total_seconds() if posted and ours else None})
    return rows


def write_receipt(root, receipt):
    atomic_json(root / "race.json", receipt)
    def text(value):
        return "unknown" if value is None else str(value)
    lines = ["# Forecast race", "", f"Mode: **{receipt['mode']}**; status: **{receipt['status']}**.", "",
             f"Recipe: `{receipt['recipe']}`; cycle: {receipt['cycle']}; horizon: f{receipt['hours']:02d}.",
             "Positive margin means our products were online first. Reference: earliest observed public posting (NOMADS or S3); final-hour products come from the registry. NOMADS listing precision is one minute. All clocks are UTC.", "",
             f"NOAA analysis f00 posted: {text(receipt.get('noaa_f00_posted_at'))}",
             f"Analysis source clocks: {text(receipt.get('noaa_f00_objects'))}",
             f"Race door started: {text(receipt.get('started_at'))}",
             f"Our launch: {text(receipt.get('launch_at'))}",
             f"Our first completed step: {text(receipt.get('first_step_at'))}",
             f"Measured box label: {text(receipt.get('box_label'))}",
             f"Forecast exit code: {text(receipt.get('forecast_exit_code'))}",
             f"Hydro: {text(receipt.get('hydro'))}",
             f"Products: {text(receipt.get('products'))}", ""]
    lines += [f"- {note}" for note in receipt.get("notes", [])]
    if receipt.get("prediction_inputs"):
        lines += ["", "Prediction inputs (seconds; steady cost includes the measured output workload):",
                  "```json", json.dumps(receipt["prediction_inputs"], indent=2), "```"]
    lines += ["", "| Hour | Our online UTC | NOAA posted UTC | Margin (s) |",
              "|---|---|---|---:|"]
    for row in receipt.get("hourly", []):
        margin = row["margin_seconds"]
        lines.append(f"| f{row['hour']:02d} | {text(row['our_online_at'])} | {text(row['noaa_posted_at'])} | "
                     f"{f'{margin:+.3f}' if margin is not None else 'unknown'} |")
    lines += ["", "Execution box and engine (dry predictions use the measured box label above):", "```json", json.dumps(receipt.get("box", {}), indent=2), "```",
              "", "Stage walls from source receipts (overlapping stages are not summed):", "```json",
              json.dumps(receipt.get("stages", []), indent=2, default=str), "```", "",
              "Object-watch errors:", "```json", json.dumps(receipt.get("noaa_watch", {}).get("errors", []), indent=2), "```", "",
              "Receipt sources:", "```json", json.dumps(receipt.get("receipt_sources", []), indent=2), "```", ""]
    temporary = root / "RACE.md.new"
    temporary.write_text("\n".join(lines))
    temporary.replace(root / "RACE.md")


def dry_race(args, row, cycle, root, receipt):
    from datetime import timedelta
    inputs = dry_inputs(args)
    prefix, analysis, hour_keys = object_names(row, cycle, args.hours)
    references = comparison_keys(row, cycle, args.hours, hour_keys)
    keys = [analysis, *(key for group in references.values() for key in group)]
    if args.posting_times:
        objects = posting_objects(args.posting_times, cycle, keys)
        source = str(args.posting_times.resolve())
    else:
        objects = list_objects(args.object_endpoint or row["race"]["endpoint"], prefix)
        mirror = mirror_url(row, cycle, args.object_endpoint)
        if mirror:
            try:
                for key, item in list_directory(mirror, keys).items():
                    objects.setdefault(key, {"key": key}).update(item)
            except (OSError, ValueError) as error:
                receipt["notes"].append(f"Mirror listing unavailable: {error}; only observed source clocks are compared.")
        source = "live source listings; saved in posting-times.json"
        atomic_json(root / "posting-times.json", {"cycle": stamp(cycle), "rows": list(objects.values())})
    f00 = earliest_post(objects.get(analysis, {}))
    if not f00 or not any(earliest_post(objects.get(k, {})) for k in references[args.hours]):
        raise ValueError(f"Posting evidence lacks the analysis f00 or f{args.hours:02d} for {stamp(cycle)}; cannot price an unobserved window.")
    launched = instant(f00) + timedelta(seconds=inputs["launch_delay"])
    first = launched + timedelta(seconds=inputs["launch_to_first_step"])
    online = {h: stamp(first + timedelta(seconds=h * inputs["seconds_per_hour"] + inputs["publish_lag"]))
              for h in hour_keys if h > 0}
    receipt.update(status="predicted", prediction_inputs=inputs, noaa_f00_posted_at=f00,
                   launch_at=stamp(launched), first_step_at=stamp(first),
                   hourly=compare_hours(hour_keys, objects, online, references), receipt_sources=[source],
                   noaa_f00_objects=objects.get(analysis, {}))
    receipt["margin_seconds"] = receipt["hourly"][-1]["margin_seconds"]
    receipt["window_seconds"] = (instant(receipt["hourly"][-1]["noaa_posted_at"]) - instant(f00)).total_seconds()
    receipt["required_seconds_per_hour"] = (receipt["window_seconds"] - inputs["launch_delay"] -
        inputs["launch_to_first_step"] - inputs["publish_lag"]) / args.hours
    receipt["notes"].append("Prediction only. Launch delay is measured from the earliest observed analysis source clock; Budget = launch delay + launch-to-first-completed-step + N steady forecast hours + publish lag. No forecast ran.")
    write_receipt(root, receipt)


def materialize_config(row, cycle, hours, root):
    import tomllib
    from woof.toml_document import emit_experiment_toml
    config = tomllib.loads((REGISTRY.parent / row["config"]).read_text())
    config["experiment"].update(start_time=cycle.replace(tzinfo=None), run_seconds=float(hours * 3600))
    config["fetch"].update(cycle=cycle.strftime("%Y-%m-%dT%H"), hours=hours, cadence=1)
    # The race publishes hourly. The recipe's operational cadence change
    # (15 min, then hourly after history_interval_change_s) goes with the
    # override: a second interval equal to the first is refused by the fork
    # history alarm ("history_interval2_s must exceed output_interval_s").
    for domain in config["domain"]:
        domain["history_interval_s"] = 3600.0
        domain.pop("history_interval2_s", None)
        domain.pop("history_interval_change_s", None)
    target = root / "experiment.toml"
    target.write_text(emit_experiment_toml(config))
    return target


def command_for(args, row, cycle, root):
    """Use the installed standby door; the explicit base-tree fallback is go."""
    standby = importlib.util.find_spec("woof.standby_cli") is not None
    command = [sys.executable, "-m", "woof", "standby" if standby else "go"]
    if standby:
        command += ["--recipe", row["id"], "--cycle", cycle.strftime("%Y-%m-%dT%H"), "--hours", str(args.hours)]
    else:
        command += [str(materialize_config(row, cycle, args.hours, root)), "--run-stamp", "off"]
    # Several cards hold one domain on the multi-card store road, which
    # refuses enabled [tiles] ("one domain has one road"); "auto" there means
    # the cards hold it, so a plain multi-card race resolves auto to off.
    # An explicit --tiles on still reaches that refusal by name.
    tiles = "off" if args.tiles == "auto" and (args.devices or 0) > 1 else args.tiles
    command += ["--outdir", str(root / "forecast"), "--tiles", tiles,
                "--products", args.products]
    # go treats any nonempty value except "off" as a settings filename.
    # Preserve its absent-option default instead of trying to open "auto".
    # standby has an explicit "auto" spelling for that same default.
    if args.hydro != "auto" or standby:
        command += ["--hydro", "" if args.hydro == "on" else args.hydro]
    if getattr(args, "smoke", False):
        command += ["--smoke"]
    if args.devices:
        command += ["--devices", str(args.devices)]
    if args.initial_inputs and standby:
        raise ValueError("--initial-inputs applies to the go fallback; standby acquires and binds its own cycle analysis, so this manifest would be ignored.")
    if args.initial_inputs:
        command += ["--initial-inputs", str(args.initial_inputs.resolve())]
    if args.data_dir:
        command += ["--data-dir", str(args.data_dir.resolve())]
    # A base-tree fallback still runs available hourly rendering; publication
    # without the publish lane is explicitly missing in the race receipt.
    module = __import__("woof.standby_cli" if standby else "woof.go_cli", fromlist=["register_cli"])
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    module.register_cli(subs)
    options = subs.choices["standby" if standby else "go"]._option_string_actions
    if not standby and "--warm-kernels" in options:
        command += ["--warm-kernels"]
    if "--history-interval" in options:
        command += ["--history-interval", "3600"]
    publish = "--publish-root" in options
    if publish:
        command += ["--publish-root", str(args.publish_root.resolve() if args.publish_root else root / "www")]
    return command, standby, publish


def _json_lines(path):
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except ValueError:
                    pass  # A producer may still be completing its last line.
    except OSError:
        return


def collect_receipts(root, publish_receipt=None, *, publish_root=None, cycle=None):
    """Read only metadata under the run, never scan or open forecast frames."""
    base = root / "forecast"
    roots = [base, base / "run", base / "forecast"]
    pointer = base / "latest-run.txt"
    if pointer.exists():
        child = base / pointer.read_text().strip()
        if child.is_relative_to(base) and child.is_dir():
            roots += [child, child / "run"]
    result = {"stages": [], "receipt_sources": [], "online": {}, "first_step_at": None}
    standby = read_json(base / "standby-receipt.json")
    if standby:
        result["stages"].extend(standby.get("stages", []))
        result["first_step_at"] = stamp(standby["first_step_wall"]) if standby.get("first_step_wall") else None
        if standby.get("launch_wall"):
            result["launch_at"] = stamp(standby["launch_wall"])
        result["receipt_sources"].append(str(base / "standby-receipt.json"))
        result["standby_receipt"] = standby
        if standby.get("hydro") is not None:
            result["hydro"] = standby["hydro"]
    for directory in roots:
        timeline = directory / "timeline.jsonl"
        begins = {}
        for event in _json_lines(timeline):
            stage, phase = event.get("stage"), event.get("phase")
            key = (event.get("pid"), stage)
            if phase == "begin":
                begins[key] = event
            elif phase == "end" and key in begins:
                if stage == "forecast.first_step" and not result["first_step_at"]:
                    result["first_step_at"] = event["t_utc"]
                begin = begins.pop(key)
                result["stages"].append({"name": stage, "start_wall": begin["t_utc"], "end_wall": event["t_utc"],
                    "wall_seconds": event["t_mono"] - begin["t_mono"], "attrs": event.get("attrs"), "source": str(timeline)})
        if timeline.exists():
            result["receipt_sources"].append(str(timeline))
        report = read_json(directory / "report.json")
        if report:
            result["forecast_receipt"] = report
            if "hydro_activation" in report:
                result["hydro"] = report["hydro_activation"]
            result["receipt_sources"].append(str(directory / "report.json"))
            for name, wall in report.get("timing_seconds", {}).items():
                result["stages"].append({"name": name, "wall_seconds": wall, "source": str(directory / "report.json")})
    paths = [Path(publish_receipt)] if publish_receipt else [Path(publish_root or root / "www") / "manifest.json"]
    for path in paths:
        published = read_json(path)
        if not published:
            continue
        result["receipt_sources"].append(str(path))
        selected = [record for record in published.get("hours", [])
                    if (not cycle or record.get("cycle") == cycle.strftime("%Y%m%dT%H%M%SZ")) and record.get("domain") == 1]
        result["publish_receipt"] = {**published, "hours": selected}
        for record in selected:
            # A shared web root may carry older cycles and nested domains.
            # This recipe's race is the root domain, at integer forecast hours.
            if cycle and record.get("cycle") != cycle.strftime("%Y%m%dT%H%M%SZ"):
                continue
            lead = float(record["forecast_hour"])
            if record.get("domain") == 1 and lead.is_integer() and record.get("online_at"):
                result["online"][int(lead)] = stamp(record["online_at"])
        result["products"] = sorted({p for record in selected
                                     for p in record.get("priority_products", [])})
    return result


def live_race(args, row, cycle, root, receipt):
    prefix, analysis, hour_keys = object_names(row, cycle, args.hours)
    references = comparison_keys(row, cycle, args.hours, hour_keys)
    watch = ObjectWatch(args.object_endpoint or row["race"]["endpoint"], prefix,
                        [analysis, *(key for group in references.values() for key in group)], args.poll_seconds,
                        mirror=mirror_url(row, cycle, args.object_endpoint))
    command, standby, publish = command_for(args, row, cycle, root)
    receipt.update(command=command, door="standby" if standby else "go", launch_at=None)
    if not standby:
        receipt["notes"].append("Standby door unavailable: used woof go after observing analysis f00; no warm-launch claim.")
        if not args.initial_inputs:
            receipt["notes"].append("No separate initial-inputs supplied: go uses the recipe's current initial source, not an analysis-initialized race.")
    if not publish:
        receipt["notes"].append("Publish door unavailable: hourly rendering requested, but online times and margins remain unknown.")
    process = None
    deadline = time.monotonic() + args.watch_timeout
    metadata, next_metadata = None, 0.0
    watch.thread.start()
    try:
        while True:
            observation = watch.snapshot()
            objects = observation["objects"]
            receipt["noaa_watch"] = observation
            receipt["noaa_f00_posted_at"] = earliest_post(objects.get(analysis, {}))
            receipt["noaa_f00_objects"] = objects.get(analysis, {})
            if process is None and (standby or analysis in objects):
                receipt["command_started_at"] = stamp()
                if not standby:
                    receipt["launch_at"] = receipt["command_started_at"]
                print("race: " + " ".join(command), flush=True)
                process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
            code = process.poll() if process else None
            if metadata is None or time.monotonic() >= next_metadata or code is not None:
                metadata = collect_receipts(root, args.publish_receipt, publish_root=args.publish_root, cycle=cycle)
                next_metadata = time.monotonic() + 5
            online = metadata["online"]
            receipt.update({k: v for k, v in metadata.items() if k != "online"})
            receipt["hourly"] = compare_hours(hour_keys, objects, online, references)
            receipt["margin_seconds"] = receipt["hourly"][-1]["margin_seconds"]
            all_posted = all(objects.get(key, {}).get("last_modified") and
                             (not watch.mirror or objects.get(key, {}).get("nomads_mtime"))
                             for key in [analysis, *hour_keys.values()])
            all_online = all(h in online for h in range(1, args.hours + 1))
            receipt["status"] = "running" if process and code is None else "watching"
            if code is not None:
                receipt["forecast_exit_code"] = code
                if code:
                    receipt["status"] = "failed"
                elif all_posted:
                    receipt["status"] = "complete" if all_online and receipt.get("first_step_at") and (standby or args.initial_inputs) else "incomplete"
                if code or all_posted:
                    break
            if time.monotonic() >= deadline:
                if process is None:
                    receipt["status"] = "analysis_timeout"
                    break
                if code is not None:
                    receipt["status"] = "observation_timeout"
                    break
                # The observation budget never kills a healthy forecast.
                receipt["notes"] = [n for n in receipt["notes"] if not n.startswith("NOAA observation budget")]
                receipt["notes"].append("NOAA observation budget elapsed; forecast continues until its own completion.")
                watch.stop.set()
            write_receipt(root, receipt)
            time.sleep(1)
    except BaseException:
        receipt["status"] = "interrupted"
        if process and process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        raise
    finally:
        watch.close()
        receipt["finished_at"] = stamp()
        write_receipt(root, receipt)
    return 0 if receipt["status"] == "complete" else (receipt.get("forecast_exit_code") or 2)
