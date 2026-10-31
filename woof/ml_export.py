"""``woof ml-export``: a run's history files as a machine-learning dataset.

Any run's history files (a regional run's own, a hex run's converted
frames, a global run's tapes, or the run page's ZIP of them) become one
Zarr dataset per domain: the standard pressure levels, ERA5 and
WeatherBench 2 variable names and units, earth-relative winds, below-ground
points filled by ECMWF extrapolation and flagged with a mask, optionally a
regular latitude-longitude grid.  ``xarray.open_zarr`` reads every grid.
Regular grids with WB2 names and forecast layout carry WeatherBench 2's
evaluation dimensions without variable renaming; projected native grids
retain their own dimensions and two-dimensional geographic coordinates.

Python's part is orchestration only: parse the arguments, resolve the four
tables in ``data/ml_export`` (variables, level sets, lattice spacings,
naming schemes) into the rows this run needs, write the request, run
``rw_mlexport`` and relay its progress.  Every read of a history file and
every array operation (unstaggering, rotation, vertical interpolation,
regridding, compression, the ZIP) is the Rust binary's.

Exit status: 0 done; 2 refused (one sentence naming the breakage the
refusal prevents); 3 the binary is not staged (the message names what
supplies it); 1 failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from woof.bridges import (RUSTWX_CRATE_RELATIVE, accept_resolved,
                           artifact_remedy, executable_name,
                           fetch_on_demand, on_demand_failure,
                           rustwx_build_hint)
from woof.rustwx import _candidates, crate_dir

#: The table folder, relative to the package.
TABLE_DIR = Path(__file__).resolve().parent / "data" / "ml_export"

#: The request schema the binary speaks.
REQUEST_SCHEMA = "ml-export.request/v1"

#: The line ``rw_mlexport --abi`` prints, and the literal
#: :data:`woof.bridges.BRIDGE_ABI_MARKERS` finds in the built binary.  It
#: names the request schema, the modes and the progress grammar, which is
#: what changes when the contract does.  Spelled to match
#: ``rw_mlexport::ABI``; a test binds the three.
ABI_MARKER = ("rw_mlexport --request REQUEST.json schema=ml-export.request/v1 "
              "modes=run,append,finalize progress=jsonl earth2=per-channel-v1")

#: Environment variable naming a prebuilt binary.
BINARY_ENV = "WOOF_RW_MLEXPORT"

#: Exit statuses.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_NOT_STAGED = 3

_PROBE_TIMEOUT_S = 60


class MlExportRefusal(RuntimeError):
    """A request the front door refuses; the message names the breakage."""


# ---------------------------------------------------------------------------
# The binary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Binary:
    """``rw_mlexport``: its resolution ladder and its remedy.

    The ladder every ``tools/rustwx`` artifact uses: the environment
    override, the checkout's release then debug build, ``libexec`` beside
    the package, the wheel-bundled directory, ``~/.woof/bridges``.
    """

    name: str = "rw_mlexport"
    env_var: str = BINARY_ENV
    subject: str = "the ML dataset exporter"

    def candidates(self) -> tuple[Path, ...]:
        return _candidates(self.env_var, self.name)

    def find(self) -> Path | None:
        """The first existing candidate; a named override that is missing
        is an error rather than a fall-through to some other binary."""

        override = os.environ.get(self.env_var)
        for candidate in self.candidates():
            if candidate.is_file():
                return accept_resolved(candidate.resolve())
            if override and candidate == Path(override):
                raise FileNotFoundError(
                    f"{self.env_var} names a missing file: {candidate}")
        return None

    def remedy(self) -> str:
        return artifact_remedy(
            env_var=self.env_var, filename=executable_name(self.name),
            subject=self.subject, crate_relative=RUSTWX_CRATE_RELATIVE,
            one_liner=rustwx_build_hint(), artifact=self.name)

    def probe(self, path: Path) -> tuple[bool, str]:
        """``--abi``: is this the binary this front door was written for?"""

        try:
            result = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"{path} did not run: {error}"
        if result.returncode != 0:
            return False, f"{path} --abi exited {result.returncode}"
        if ABI_MARKER not in (result.stdout or ""):
            return False, ("--abi does not carry the request contract this "
                           f"front door writes; rebuild it: {rustwx_build_hint()}")
        return True, "--abi matches this release's request contract"


BINARY = Binary()


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    return json.loads((TABLE_DIR / f"{name}.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Tables:
    variables: list
    levels: dict
    spacings: list
    names: dict

    @property
    def level_ids(self) -> list[str]:
        return [row["id"] for row in self.levels["sets"]]

    @property
    def scheme_ids(self) -> list[str]:
        return [row["id"] for row in self.names["schemes"]]


def load_tables() -> Tables:
    return Tables(
        variables=_load("variables")["rows"],
        levels=_load("levels"),
        spacings=list(_load("spacings")["degrees"]),
        names=_load("names"),
    )


def options_document(tables: Tables | None = None) -> dict:
    """What ``--list --json`` prints: every choice the tables allow.

    The site's export options are generated from this document, so a new
    table row reaches every front door as data.
    """

    tables = tables or load_tables()
    return {
        "schema": "ml-export.options/v1",
        "levels": [
            {key: row[key] for key in ("id", "kind", "hpa", "note") if key in row}
            for row in tables.levels["sets"]],
        "default_levels": tables.levels["default"],
        "variables": [
            {"id": row["id"], "kind": row["kind"], "default": bool(row.get("default")),
             "level_kinds": row.get("level_kinds", []), "names": row["names"],
             "units": row["units"], "long_name": row.get("long_name", "")}
            for row in tables.variables],
        "names": tables.names["schemes"],
        "default_names": tables.names["default"],
        "grids": ["native", "latlon"],
        "regrid": ["bilinear", "area-mean"],
        "layouts": ["analysis", "forecast"],
        "spacings_deg": tables.spacings,
    }


# ---------------------------------------------------------------------------
# Option grammar
# ---------------------------------------------------------------------------

def parse_levels(text: str, tables: Tables) -> dict:
    """``wb13``, ``era5-37``, ``model``, ``model:1-20``, ``model:1,2,4``
    or a list of hPa: the request's level spec."""

    text = text.strip()
    by_id = {row["id"]: row for row in tables.levels["sets"]}
    head, _, tail = text.partition(":")
    if head in by_id:
        row = by_id[head]
        if row["kind"] == "model":
            return {"set": text, "kind": "model", "hpa": [],
                    "model_levels": _model_selection(tail) if tail else []}
        if tail:
            raise MlExportRefusal(
                f"--levels {text}: only the model set takes a selection after ':'")
        return {"set": row["id"], "kind": "pressure", "hpa": list(row["hpa"]),
                "model_levels": []}
    try:
        hpa = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise MlExportRefusal(
            f"--levels {text} is not a level set ({', '.join(tables.level_ids)}) "
            "or a comma-separated list of whole hPa") from None
    if not hpa:
        raise MlExportRefusal("--levels is empty, so no level variable could be written")
    if len(set(hpa)) != len(hpa):
        raise MlExportRefusal(
            f"--levels {text} lists a level twice, so two slices would hold the same level")
    bad = [p for p in hpa if not 1 <= p <= 1100]
    if bad:
        raise MlExportRefusal(
            f"--levels {text}: {bad[0]} hPa is outside 1 to 1100 hPa")
    return {"set": "custom", "kind": "pressure", "hpa": hpa, "model_levels": []}


def _model_selection(text: str) -> list[int]:
    levels: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            if lo < 1 or hi < lo:
                raise MlExportRefusal(f"model level range {part} is empty or starts below 1")
            levels.extend(range(lo, hi + 1))
        elif part.isdigit() and int(part) >= 1:
            levels.append(int(part))
        else:
            raise MlExportRefusal(
                f"model level '{part}' is not a level number (1 at the bottom) or a range like 1-20")
    if len(set(levels)) != len(levels):
        raise MlExportRefusal("a model level is selected twice")
    return levels


def _applies(row: dict, level_kind: str) -> bool:
    return row["kind"] != "level" or level_kind in row.get("level_kinds", ["pressure", "model"])


def select_variables(text: str | None, tables: Tables, level_kind: str) -> list[dict]:
    """``default``, ``all``, ``+a,b`` (the defaults plus these) or ``a,b``
    (exactly these), by table id or by a name under any scheme."""

    rows = [row for row in tables.variables if _applies(row, level_kind)]
    lookup: dict[str, dict] = {}
    for row in tables.variables:
        lookup[row["id"]] = row
        for name in row["names"].values():
            lookup.setdefault(name, row)
    defaults = [row for row in rows if row.get("default")]
    text = (text or "default").strip()
    if text == "default":
        return defaults
    if text == "all":
        return rows
    extra = text.startswith("+")
    wanted = [part.strip() for part in text.lstrip("+").split(",") if part.strip()]
    chosen = list(defaults) if extra else []
    for name in wanted:
        row = lookup.get(name)
        if row is None:
            raise MlExportRefusal(
                f"--variables: '{name}' is not a row of the variables table; the rows are "
                + ", ".join(r["id"] for r in tables.variables))
        if not _applies(row, level_kind):
            raise MlExportRefusal(
                f"--variables: '{row['id']}' is a {', '.join(row.get('level_kinds', []))}-level "
                f"variable and the level set is {level_kind}")
        if row not in chosen:
            chosen.append(row)
    if not chosen:
        raise MlExportRefusal("--variables selects nothing, so the dataset would be empty")
    return chosen


def parse_grid(text: str, regrid: str) -> dict:
    head, _, tail = text.strip().partition(":")
    if head == "native":
        if tail:
            raise MlExportRefusal("--grid native takes no spacing")
        return {"kind": "native", "deg": None, "method": regrid}
    if head == "latlon":
        deg = None
        if tail:
            try:
                deg = float(tail)
            except ValueError:
                raise MlExportRefusal(
                    f"--grid {text}: '{tail}' is not a spacing in degrees") from None
            if not 0 < deg <= 10:
                raise MlExportRefusal(
                    f"--grid {text}: the spacing must lie in (0, 10] degrees")
        return {"kind": "latlon", "deg": deg, "method": regrid}
    raise MlExportRefusal(f"--grid {text} is not native, latlon or latlon:DEG")


def _request_row(row: dict) -> dict:
    keep = ("id", "kind", "names", "op", "fields", "optional_fields",
            "below_ground", "scale", "units", "long_name", "standard_name",
            "short_name", "param_id", "comment")
    return {key: row[key] for key in keep if key in row}


def engine_name() -> str:
    """The engine's own name, as the installed package spells it."""

    return __name__.split(".")[0]


def engine_version() -> str:
    try:
        from woof import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001 - provenance text, never a refusal
        return "unknown"


def _options_text(args) -> str:
    """The options as given, without any path (paths are machine identity
    and stay out of the dataset)."""

    parts = [f"--levels {args.levels}", f"--variables {args.variables or 'default'}",
             f"--grid {args.grid}", f"--regrid {args.regrid}",
             f"--names {args.names}", f"--layout {args.layout}"]
    if args.domains:
        parts.append(f"--domains {args.domains}")
    if args.every:
        parts.append(f"--every {args.every:g}")
    if args.start:
        parts.append(f"--start {args.start}")
    if args.end:
        parts.append(f"--end {args.end}")
    if args.skip_unavailable:
        parts.append("--skip-unavailable")
    if getattr(args, "chunk_yx", "auto") not in (None, "auto"):
        parts.append(f"--chunk-yx {args.chunk_yx}")
    return " ".join(parts)


def parse_chunk_yx(text: str | None) -> list[int] | None:
    """``--chunk-yx``: ``auto`` (None), ``full`` ([0, 0]) or ``NY,NX``."""

    if text is None or text == "auto":
        return None
    if text == "full":
        return [0, 0]
    parts = text.replace("x", ",").split(",")
    try:
        ny, nx = (int(p) for p in parts)
    except ValueError:
        raise MlExportRefusal(
            f"--chunk-yx {text} is not auto, full or two cell counts like 256,256") from None
    if ny <= 0 or nx <= 0:
        raise MlExportRefusal(
            f"--chunk-yx {text} has a zero or negative edge; give two positive cell counts")
    return [ny, nx]


def build_request(args, tables: Tables | None = None) -> dict:
    """The ``ml-export.request/v1`` document for parsed arguments."""

    tables = tables or load_tables()
    if args.names not in tables.scheme_ids:
        raise MlExportRefusal(
            f"--names {args.names} is not a naming scheme ({', '.join(tables.scheme_ids)})")
    mode = "finalize" if args.finalize else "append" if args.append else "run"
    if args.append and args.finalize:
        raise MlExportRefusal("--append and --finalize are two separate calls")
    if mode != "finalize" and not args.inputs:
        raise MlExportRefusal("no history files were given, so there is nothing to export")
    if mode == "finalize" and args.inputs:
        raise MlExportRefusal(
            "--finalize closes the datasets already appended; give the frames to --append calls")
    levels = parse_levels(args.levels, tables)
    variables = select_variables(args.variables, tables, levels["kind"])
    config_sha = None
    if args.config is not None:
        config = Path(args.config)
        if not config.is_file():
            raise MlExportRefusal(
                f"--config {config} does not exist, so its digest cannot stand for the run's configuration")
        config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        bad = [d for d in domains if not re.fullmatch(r"d\d\d", d)]
        if bad:
            raise MlExportRefusal(f"--domains: '{bad[0]}' is not a domain id like d01")
    return {
        "schema": REQUEST_SCHEMA,
        "mode": mode,
        "inputs": [str(Path(p).resolve()) for p in args.inputs],
        "out": str(Path(args.out).resolve()),
        "overwrite": bool(args.overwrite),
        "zip": bool(args.zip),
        "variables": [_request_row(row) for row in variables],
        "levels": levels,
        "grid": parse_grid(args.grid, args.regrid),
        "names": args.names,
        "layout": args.layout,
        "domains": domains,
        "every_hours": args.every,
        "start": args.start,
        "end": args.end,
        "skip_unavailable": bool(args.skip_unavailable),
        "threads": args.threads,
        "spacings_deg": tables.spacings,
        "chunk_yx": parse_chunk_yx(getattr(args, "chunk_yx", None)),
        "provenance": {
            "engine": engine_name(),
            "exporter_version": engine_version(),
            "config_sha256": config_sha,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "options": _options_text(args),
            "history_attributes": tables.names.get("history_attributes", {}),
            "history_engines": tables.names.get("history_engines", {}),
            "history_engine_titles": tables.names.get("history_engine_titles", {}),
        },
    }


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _mb(n) -> str:
    return f"{(n or 0) / 1e6:,.1f} MB"


def describe_event(event: dict) -> str | None:
    """One human line for a progress event, or None to stay quiet."""

    kind = event.get("event")
    if kind == "frame":
        return (f"frame {event['frame']}/{event['of']}  {event['domain']}  "
                f"{event['valid']}  convert {event['seconds']:.1f} s, read "
                f"{event.get('read_seconds', 0):.1f} s  {_mb(event.get('bytes'))}")
    if kind == "domain":
        dropped = event.get("levels_dropped_above_model_top") or []
        text = (f"{event['domain']}: {event['grid'][0]} x {event['grid'][1]}, "
                f"{event['horizontal_grid']}")
        if event.get("levels"):
            text += f"; levels {event['levels']}"
        if dropped:
            text += (f"; left out above the model top ({event.get('model_top_hpa')} hPa): "
                     f"{dropped}")
        for item in event.get("omitted") or []:
            text += f"\n  not written: {item['variable']} ({item['reason']})"
        return text
    if kind == "zip":
        return f"archive {event.get('name')}: {_mb(event.get('bytes'))}"
    if kind == "finalized":
        return f"finalized {', '.join(event.get('domains') or [])}: {_mb(event.get('bytes'))}"
    if kind == "done":
        return (f"done: {event.get('frames', 0)} frames, {_mb(event.get('bytes'))}, "
                f"{event.get('seconds', 0):.1f} s")
    return None


def run_binary(exe: Path, request: dict, *, as_json: bool, stream=None) -> int:
    """Write the request, run the binary, relay its progress."""

    stream = stream or sys.stdout
    workdir = Path(tempfile.mkdtemp(prefix="ml-export-"))
    try:
        path = workdir / "request.json"
        path.write_text(json.dumps(request, indent=1), encoding="utf-8")
        process = subprocess.Popen(
            [str(exe), "--request", str(path)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, errors="replace")
        assert process.stdout is not None
        last = None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            if as_json:
                print(line, file=stream, flush=True)
            try:
                event = json.loads(line)
            except ValueError:
                if not as_json:
                    print(line, file=stream, flush=True)
                continue
            last = event
            if not as_json:
                text = describe_event(event)
                if text:
                    print(text, file=stream, flush=True)
        stderr = process.stderr.read() if process.stderr else ""
        code = process.wait()
        if code != 0 and not as_json:
            message = (last or {}).get("message") or stderr.strip() or f"exited {code}"
            word = "refused" if code == EXIT_REFUSED else "failed"
            print(f"ml-export {word}: {message}", file=sys.stderr)
        return code if code in (EXIT_OK, EXIT_FAILED, EXIT_REFUSED) else EXIT_FAILED
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _positive_hours(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of hours") from None
    if not value > 0:
        raise argparse.ArgumentTypeError("--every takes a positive number of hours")
    return value


def register_cli(sub) -> None:
    tables_note = ("levels: wb13 (default), era5-37, model, model:1-20, or a list of hPa; "
                   "variables: default, all, +NAME,... (defaults plus these) or NAME,...")
    parser = sub.add_parser(
        "ml-export",
        help="turn a run's history files into a machine-learning dataset (Zarr)",
        description=(
            "Convert history files (wrfout_dNN_*, the run page's ZIP of them, or a folder) "
            "into one Zarr dataset per domain on standard pressure levels with ERA5 and "
            "WeatherBench 2 names and units, earth-relative winds and ERA5's below-ground "
            "fill, ready for xarray.open_zarr.  " + tables_note),
    )
    add_arguments(parser)


def add_arguments(parser, *, earth2=False) -> None:
    parser.add_argument("inputs", nargs="*", metavar="INPUT",
                        help="history files, folders holding them, .gz history files, or ZIPs")
    parser.add_argument("--out", type=Path, metavar="DIR",
                        help="the export folder (one dNN.zarr per domain, README.txt, receipt)")
    if not earth2:
        parser.add_argument("--levels", default=None, metavar="SET",
                            help="wb13 (default), era5-37, model, model:LIST, or hPa list")
        parser.add_argument("--variables", default=None, metavar="LIST",
                            help="default, all, +NAME,... or NAME,... (table ids or names)")
        parser.add_argument("--grid", default="native", metavar="GRID",
                            help="native (default), latlon, or latlon:DEG")
        parser.add_argument("--names", default=None, metavar="SCHEME",
                            help="wb2 (default, WeatherBench 2 long names) or era5 (short names)")
        parser.add_argument("--layout", default="analysis", choices=("analysis", "forecast"),
                            help="analysis: time is valid time; forecast: WeatherBench 2's "
                                 "time + prediction_timedelta")
    parser.add_argument("--regrid", default="bilinear", choices=("bilinear", "area-mean"),
                        help="latlon method (default bilinear)")
    parser.add_argument("--domains", default=None, metavar="LIST", help="e.g. d01,d02")
    parser.add_argument("--every", type=_positive_hours, default=None, metavar="HOURS",
                        help="keep frames on multiples of HOURS from the run's start")
    parser.add_argument("--start", default=None, metavar="TIME", help="first valid time kept")
    parser.add_argument("--end", default=None, metavar="TIME", help="last valid time kept")
    parser.add_argument("--config", default=None, metavar="TOML",
                        help="the run's configuration; its SHA-256 is recorded")
    parser.add_argument("--zip", action="store_true",
                        help="also write <out>-ml.zip (stored entries, opens in place)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an earlier export in --out")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="write the variables the files can make and record the rest")
    parser.add_argument("--chunk-yx", default="auto", metavar="SHAPE",
                        help="horizontal chunk shape: auto (default; 512x512 tiles for planes "
                             "over 1024x1024 cells, one chunk otherwise), full, or NY,NX")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="worker threads (default: every core)")
    parser.add_argument("--append", action="store_true",
                        help="add these frames to the export in --out (one frame at a time)")
    parser.add_argument("--finalize", action="store_true",
                        help="close the export in --out after --append calls")
    parser.add_argument("--list", action="store_true",
                        help="print the level sets, variables and naming schemes")
    parser.add_argument("--json", action="store_true",
                        help="with --list: the options document; otherwise: progress as JSON lines")
    parser.set_defaults(func=main)


def _print_list(tables: Tables) -> None:
    print("level sets:")
    for row in tables.levels["sets"]:
        marker = " (default)" if row["id"] == tables.levels["default"] else ""
        levels = f": {row['hpa']}" if row.get("hpa") else ""
        print(f"  {row['id']}{marker}{levels}")
    print("variables (* = default):")
    for row in tables.variables:
        names = ", ".join(f"{k} {v}" for k, v in row["names"].items())
        star = "*" if row.get("default") else " "
        print(f" {star} {row['id']:<28} {row['kind']:<8} {row['units']:<12} {names}")
    print("naming schemes:")
    for row in tables.names["schemes"]:
        print(f"  {row['id']}: {row['description']}")


def main(args) -> int:
    tables = load_tables()
    if args.list:
        if args.json:
            print(json.dumps(options_document(tables), indent=1))
        else:
            _print_list(tables)
        return EXIT_OK
    args.levels = args.levels or tables.levels["default"]
    args.names = args.names or tables.names["default"]
    scheme = next((r for r in tables.names["schemes"] if r["id"] == args.names), {})
    if "grid_profiles" in scheme:
        from woof import earth2_export
        try:
            args.earth2 = scheme["grid_profiles"][args.grid]
            level_set = parse_levels(args.levels, tables)
            if level_set["kind"] != "pressure":
                raise MlExportRefusal("Earth2Studio channel names require pressure levels")
            selected = select_variables(args.variables, tables, level_set["kind"])
            args.channels = ",".join(
                row["names"][args.names].format(level=level)
                for row in selected
                for level in (level_set["hpa"] if "{level}" in row["names"][args.names] else [None]))
            args.consumers = None
            return earth2_export.main(args)
        except (KeyError, MlExportRefusal) as error:
            print(f"ml-export refused: {error}; use export --earth2 PROFILE to select a fixed Earth2 grid", file=sys.stderr)
            return EXIT_REFUSED
    if args.out is None:
        print("ml-export refused: --out is required (the export folder)", file=sys.stderr)
        return EXIT_REFUSED
    try:
        request = build_request(args, tables)
    except MlExportRefusal as refusal:
        print(f"ml-export refused: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        exe = BINARY.find()
        if (exe is None and not os.environ.get(BINARY.env_var)
                and fetch_on_demand(BINARY.name)):
            # On demand since 2.9.11 (PyPI per-file limit): pinned in this
            # release's bundle, staged by the first export that needs it.
            exe = BINARY.find()
    except FileNotFoundError as error:
        print(f"ml-export: {error}", file=sys.stderr)
        return EXIT_NOT_STAGED
    if exe is None:
        failure = on_demand_failure(BINARY.name)
        print(f"ml-export: {BINARY.subject} ({BINARY.name}) is not built or not staged"
              + (f" ({failure})" if failure else "") + ".\n"
              f"{BINARY.remedy()}", file=sys.stderr)
        return EXIT_NOT_STAGED
    ok, why = BINARY.probe(exe)
    if not ok:
        print(f"ml-export: {exe} is not the exporter this release drives: {why}",
              file=sys.stderr)
        return EXIT_NOT_STAGED
    return run_binary(exe, request, as_json=args.json)
