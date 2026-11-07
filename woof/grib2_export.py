"""Native GRIB2 export. Python handles arguments and subprocesses only.

The exporter (``rw_grib2export``, WOOF 2.8.7) computes every product with
the clean-room post-processor ``rw-post`` -- on a GPU when one has room,
else on the CPU (``--post-device``) -- and packs WMO GRIB2 in Rust.  The
catalog is ``woof-post/v1``; ``--definitions upp`` is a one-release alias of
``woof``.  ``--definitions renderer`` (alias ``arwen``) takes the rows the map
renderer draws from the renderer's own diagnostics, ``--extrema-interval-seconds``
adds maximum and minimum rows over that window, and ``--upp-control`` chooses
the composite reflectivity level label.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from woof import ml_export

REQUEST_SCHEMA = "grib2-export.request/v1"
ABI_MARKER = ("rw_grib2export --request REQUEST.json schema=grib2-export.request/v1 "
              "modes=run,append,finalize progress=jsonl default_definitions=woof "
              "grid_geometry=wrf-native-locations/v1 surface_catalog=woof-post/v1 "
              "post_device=auto,gpu,cpu definitions=woof,renderer extrema_window=seconds "
              "composite_level=200,10 gust=auto,similarity,tke")
BINARY_ENV = "WOOF_RW_GRIB2EXPORT"

#: ``--upp-control`` names and the composite reflectivity level type each
#: writes (code table 4.5): 200, NCEP's local entire-atmosphere layer (the
#: SRW control), or 10, WMO's entire atmosphere (the RAP and HRRR control).
#: Nothing else changes with the name.
CONTROL_COMPOSITE_LEVEL = {"srw": 200, "rapr": 10}


@dataclass(frozen=True)
class Binary(ml_export.Binary):
    name: str = "rw_grib2export"
    env_var: str = BINARY_ENV
    subject: str = "the native GRIB2 exporter"

    def find(self) -> Path | None:
        """The ladder, then (on demand since WOOF 1.1.6, PyPI per-file limit) the
        first-use stage from this release's bundle; never inside
        bridges.inspection_only(), so doctor only reports."""

        found = super().find()
        if found is None and not os.environ.get(self.env_var):
            from woof.bridges import fetch_on_demand
            if fetch_on_demand(self.name):
                found = super().find()
        return found

    def probe(self, path: Path) -> tuple[bool, str]:
        try:
            result = subprocess.run([str(path), "--abi"], capture_output=True,
                                    text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            return False, str(error)
        ok = result.returncode == 0 and ABI_MARKER in result.stdout
        return ok, "GRIB2 request contract matches" if ok else "rebuild the GRIB2 exporter for this release"


BINARY = Binary()


def probe_post_device(path: Path, *, timeout: float = 120) -> dict | None:
    """``rw_grib2export --probe-post-device``: does the first card open
    with the GPU post kernels?  ``None`` when this build has no probe (an
    exporter from before it) or the probe could not run.  It opens a CUDA
    context, so callers honour ``GPUWM_NO_LOCAL_GPU`` first."""
    try:
        result = subprocess.run([str(path), "--probe-post-device"], capture_output=True,
                                text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return payload if isinstance(payload, dict) else None


def register_cli(sub) -> None:
    parser = sub.add_parser("export-grib2", help="write native surface and pressure-level GRIB2 files",
                            description="Export every domain and time in history files, a run folder, gzip files or a history ZIP. Array processing and packing run in Rust.")
    parser.add_argument("inputs", nargs="*", metavar="INPUT")
    parser.add_argument("--out", type=Path, help="export directory")
    parser.add_argument("--domains", help="comma-separated d01,d02,...")
    parser.add_argument("--times", help="comma-separated UTC valid times")
    parser.add_argument("--start", help="first UTC valid time to write")
    parser.add_argument("--end", help="last UTC valid time to write")
    parser.add_argument("--fields", choices=("standard", "surface", "pressure", "all"), default="standard")
    parser.add_argument("--packing", choices=("complex", "simple"), default="complex")
    parser.add_argument("--definitions", choices=("woof", "renderer", "arwen", "upp"), default="woof",
                        help="who computes the fields: woof (default, the WOOF post-processor, catalog woof-post/v1); "
                             "renderer takes the fields the product maps draw from the renderer's own diagnostics "
                             "(arwen is the same); upp is a one-release alias of woof")
    parser.add_argument("--extrema-interval-seconds", type=int, metavar="SECONDS",
                        help="also write maximum and minimum updraft helicity and maximum 10 m wind over this "
                             "window ending at each frame (3600 = hourly), from the running extremes the history "
                             "frames of the window carry; pass the whole run folder")
    parser.add_argument("--upp-control", choices=tuple(CONTROL_COMPOSITE_LEVEL), default="srw",
                        help="composite reflectivity level label: srw (default) writes the entire-atmosphere "
                             "layer 200, rapr writes 10 as RAP and HRRR files do")
    parser.add_argument("--post-device", choices=("auto", "gpu", "cpu"), default="auto",
                        help="where products are computed: auto (default) uses a GPU with room, else the CPU")
    parser.add_argument("--gust", choices=("auto", "similarity", "tke"), default="auto",
                        help="surface gust method: auto (default) is the mixed-layer TKE gust "
                             "when the history carries TKE (TKE_PBL or QKE), else the "
                             "surface-layer similarity gust (RULINGS 7, scored against ASOS)")
    parser.add_argument("--winds", choices=("grid", "earth"), default="grid",
                        help="wind component frame, default grid-relative; 100 m components are always earth-relative")
    parser.add_argument("--bits", type=int, default=20, help="quantization bits, default 20")
    parser.add_argument("--levels", help="comma-separated hPa levels, default 100 through 1000 by 25")
    parser.add_argument("--threads", type=int, help="Rust worker threads")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--append", action="store_true", help="append chronologically ordered frames")
    modes.add_argument("--finalize", action="store_true", help="close an incremental export")
    parser.add_argument("--zip", action="store_true", help="write sibling <out>-grib2.zip")
    parser.add_argument("--list", action="store_true", help="show field and packing options")
    parser.add_argument("--json", action="store_true", help="machine-readable options or progress")
    parser.set_defaults(func=main)


def build_request(args) -> dict:
    if args.out is None:
        raise ValueError("--out is required")
    if not args.finalize and not args.inputs:
        raise ValueError("at least one history input is required")
    if not 1 <= args.bits <= 31:
        raise ValueError("--bits must be 1 through 31")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.definitions == "upp":
        print("export-grib2: --definitions upp is now woof (catalog woof-post/v1); "
              "the alias is accepted for this release only", file=sys.stderr)
    seconds = args.extrema_interval_seconds
    if seconds is not None and not 1 <= seconds <= 86400:
        raise ValueError("--extrema-interval-seconds must be 1 through 86400")
    def split(value):
        return [s.strip() for s in (value or "").split(",") if s.strip()]
    domains = split(args.domains)
    if any(len(d) != 3 or d[0] != "d" or not d[1:].isdigit() or d == "d00" for d in domains):
        raise ValueError("--domains uses d01,d02,...")
    levels = [int(p) for p in split(args.levels)] if args.levels else list(range(100, 1001, 25))
    if len(set(levels)) != len(levels) or any(not 1 <= p <= 1100 for p in levels):
        raise ValueError("--levels must contain unique levels from 1 through 1100 hPa")
    request = {"schema": REQUEST_SCHEMA, "mode": "finalize" if args.finalize else "append" if args.append else "run",
            "inputs": [str(Path(p).resolve()) for p in args.inputs], "out": str(args.out.resolve()),
            "fields": args.fields, "packing": args.packing, "bits": args.bits, "levels": levels,
            "definitions": "renderer" if args.definitions in ("renderer", "arwen") else "woof",
            "winds": args.winds,
            "domains": domains, "times": split(args.times), "start": args.start, "end": args.end,
            "zip": bool(args.zip), "threads": args.threads,
            "post_device": args.post_device, "gust": args.gust,
            "extrema_interval_seconds": seconds,
            "composite_level_type": CONTROL_COMPOSITE_LEVEL[args.upp_control]}
    return request


def invoke(exe: Path, request: dict, *, as_json=False, stream=None,
           on_process=None, cancel_event=None) -> int:
    stream = stream or sys.stdout
    if cancel_event is not None and cancel_event.is_set():
        return 130
    with tempfile.TemporaryDirectory(prefix="grib2-export-") as scratch:
        path = Path(scratch) / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        # A file keeps diagnostics from blocking a full stdout pipe.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_log:
            process = subprocess.Popen([str(exe), "--request", str(path)], stdout=subprocess.PIPE,
                                       stderr=error_log, text=True, errors="replace",
                                       env=dict(os.environ))
            try:
                if on_process is not None:
                    on_process(process)
                if cancel_event is not None and cancel_event.is_set() and process.poll() is None:
                    process.terminate()
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.strip()
                    if as_json:
                        print(line, file=stream, flush=True)
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        print(line, file=stream, flush=True)
                        continue
                    if event.get("event") == "file":
                        print(f"{event['path']}: {event['fields']} fields, {event['bytes'] / 1e6:.2f} MB", file=stream, flush=True)
                    elif event.get("event") == "done":
                        print(f"GRIB2: {event['frames']} frames, {event['bytes'] / 1e6:.2f} MB, {event['seconds']:.2f} s", file=stream, flush=True)
                    elif event.get("event") == "frame":
                        devices = ", ".join(f"{k} {v}" for k, v in sorted((event.get("devices") or {}).items()))
                        print(f"{event['domain']} {event['valid']}: {event['seconds']:.2f} s ({devices})", file=stream, flush=True)
                    elif event.get("event") == "warning":
                        print(f"export-grib2: {event.get('message')}", file=sys.stderr, flush=True)
                    elif event.get("event") == "error":
                        print(f"export-grib2 failed: {event['message']}", file=sys.stderr)
                status = process.wait()
                if status and not (cancel_event is not None and cancel_event.is_set()):
                    error_log.seek(0)
                    detail = error_log.read().strip()
                    if detail:
                        print(detail, file=sys.stderr)
                return status
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if process.stdout is not None:
                    process.stdout.close()
                if on_process is not None:
                    on_process(None)


def main(args) -> int:
    try:
        request = None if args.list else build_request(args)
        exe = BINARY.find()
    except (ValueError, FileNotFoundError) as error:
        print(f"export-grib2: {error}", file=sys.stderr)
        return 2
    if exe is None:
        print(BINARY.remedy(), file=sys.stderr)
        return 3
    valid, reason = BINARY.probe(exe)
    if not valid:
        print(f"export-grib2: {reason}", file=sys.stderr)
        return 3
    if args.list:
        result = subprocess.run([str(exe), "--list"], capture_output=True, text=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            return result.returncode
        if args.json:
            print(result.stdout.strip())
        else:
            options = json.loads(result.stdout)
            print("field sets: " + ", ".join(options["fields"]))
            print("packing: " + ", ".join(options["packing"]))
            print("surface fields: " + ", ".join(row["id"] for row in options["surface_fields"]))
            print("pressure fields: " + ", ".join(row["id"] for row in options["pressure_rows"]))
            for row in options["pressure_rows"]:
                print(f"  {row['id']}: {row['grib']} {row['units']} | {row['method']}")
        return 0
    return invoke(exe, request, as_json=args.json)
