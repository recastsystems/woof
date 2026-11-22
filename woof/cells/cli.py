"""``woof cells``: the storm-cell doors.

Three verbs, one chain:

* ``woof cells export`` -- wrfout series -> ``input.tfs`` on the height
  ladder, with its receipt.  Needs no titan.
* ``woof cells analyze`` -- export, then ``titan analyze`` into the
  render folder layout (``<out>/<domain>/cells/<first-valid-day>/``),
  then the catalog, overlays and GeoJSON beside the bundle.  Refuses by
  name when no titan binary resolves.
* ``woof cells catalog`` -- re-join an existing bundle to its wrfout
  series (a different threshold profile, a re-run over the same frames).

Every door prints one line per frame as it works and, with ``--json``,
its receipt at the end; refusals go to stderr with exit 2, the code the
MCP layer reads as "the door said no, here is its sentence".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from woof.cells import export as _export
from woof.cells import titan as _titan
from woof.cells.columns import ColumnsError

PRODUCT = "cells"
BUNDLE_DIR = "titan"
ANALYZE_RECEIPT = "analyze-receipt.json"


def _say(text: str) -> None:
    print(text, flush=True)


def _refuse(text: str) -> int:
    print(text, file=sys.stderr, flush=True)
    return 2


def _series_root(out: Path, receipt: dict) -> Path:
    """``<out>/<domain-token>/cells/<first valid day>`` for one series."""

    from woof.render import domain_token
    from woof.render_layout import product_dir, valid_day

    grid = receipt.get("grid") or {}
    token = domain_token(grid.get("domain"), grid.get("dx_m"))
    frames = receipt.get("frames") or []
    day = valid_day(frames[0]["valid"]) if frames else None
    return out / product_dir(domain=token, product=PRODUCT, day=day)


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "wrfout", nargs="+", metavar="WRFOUT",
        help="wrfout NetCDF file(s), a directory of them, or a glob; one "
             "domain per invocation, sorted by name (WRF's stamp order)")


def _add_ladder(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ladder", default=_export.DEFAULT_LADDER, metavar="BOTTOM:TOP:STEP",
        help="the fixed height ladder titan sees, metres above sea level, "
             f"cell centres (default {_export.DEFAULT_LADDER}: 72 levels)")
    parser.add_argument(
        "--no-temperature", dest="temperature", action="store_false",
        help="export reflectivity only; by default the model temperature "
             "rides along on the ladder so titan reports each cell's mean "
             "temperature")


def _export_cmd(args) -> int:
    started = time.perf_counter()
    try:
        receipt = _export.export_series(
            args.wrfout, args.out, ladder=args.ladder,
            temperature=args.temperature, progress=_say)
    except (ColumnsError, _export.ExportError) as error:
        return _refuse(str(error))
    receipt["door_wall_seconds"] = round(time.perf_counter() - started, 3)
    if args.json:
        print(json.dumps(receipt, indent=2))
    return 0


def _analyze_cmd(args) -> int:
    started = time.perf_counter()
    try:
        titan = _titan.resolve_titan(args.titan, what="woof cells analyze")
    except (_titan.TitanMissing, FileNotFoundError) as error:
        return _refuse(str(error))
    from woof.cells import catalog as _catalog

    stage: dict[str, float] = {}
    out = Path(args.out)
    scratch = out / ".cells-export"
    try:
        tick = time.perf_counter()
        export_receipt = _export.export_series(
            args.wrfout, scratch, ladder=args.ladder,
            temperature=args.temperature, progress=_say)
        stage["export_s"] = round(time.perf_counter() - tick, 3)
    except (ColumnsError, _export.ExportError) as error:
        return _refuse(str(error))
    root = _series_root(out, export_receipt)
    root.mkdir(parents=True, exist_ok=True)
    stream = root / _export.STREAM_NAME
    (scratch / _export.STREAM_NAME).replace(stream)
    export_receipt["stream"] = str(stream)
    (root / _export.RECEIPT_NAME).write_text(
        json.dumps(export_receipt, indent=2) + "\n", encoding="utf-8")
    (scratch / _export.RECEIPT_NAME).unlink(missing_ok=True)
    try:
        scratch.rmdir()
    except OSError:
        pass
    bundle_dir = root / BUNDLE_DIR
    try:
        tick = time.perf_counter()
        analyze_receipt = _titan.analyze(
            titan, stream, bundle_dir, profile=args.profile,
            config=args.titan_config,
            timestamps_ms=[f["timestamp_ms"] for f in export_receipt["frames"]])
        stage["titan_analyze_s"] = round(time.perf_counter() - tick, 3)
    except _titan.TitanFailed as error:
        return _refuse(str(error))
    cadence = analyze_receipt["cadence"]
    if cadence["overrides"]:
        _say(f"titan: series cadence {cadence['interval_s']:.0f} s; raised "
             + ", ".join(f"{k}={v}" for k, v in cadence["overrides"].items())
             + f" over the {args.profile} profile")
    _say(f"titan: {analyze_receipt['stdout']} ({stage['titan_analyze_s']} s)")
    catalog_receipt = None
    if args.catalog:
        try:
            tick = time.perf_counter()
            catalog_receipt = _catalog.build_catalog(
                args.wrfout, bundle_dir, root, progress=_say,
                export_receipt=export_receipt)
            stage["catalog_s"] = round(time.perf_counter() - tick, 3)
        except (ColumnsError, _catalog.CatalogError) as error:
            return _refuse(str(error))
    receipt = {
        "schema": "gpuwm-cells-analyze/1",
        "root": str(root),
        "stream": str(stream),
        "bundle": str(bundle_dir),
        "export": {k: v for k, v in export_receipt.items() if k != "frames"},
        "titan": analyze_receipt,
        "catalog": catalog_receipt,
        "stages_seconds": stage,
        "door_wall_seconds": round(time.perf_counter() - started, 3),
    }
    (root / ANALYZE_RECEIPT).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    _say(f"cells: {root} ({receipt['door_wall_seconds']} s)")
    if args.json:
        print(json.dumps(receipt, indent=2))
    return 0


def _catalog_cmd(args) -> int:
    from woof.cells import catalog as _catalog

    started = time.perf_counter()
    export_receipt = None
    for candidate in (Path(args.bundle).parent / _export.RECEIPT_NAME,
                      Path(args.bundle) / _export.RECEIPT_NAME):
        if candidate.is_file():
            export_receipt = json.loads(candidate.read_text("utf-8"))
            break
    try:
        receipt = _catalog.build_catalog(
            args.wrfout, args.bundle, args.out, progress=_say,
            export_receipt=export_receipt)
    except (ColumnsError, _catalog.CatalogError, _titan.TitanFailed) as error:
        return _refuse(str(error))
    receipt["door_wall_seconds"] = round(time.perf_counter() - started, 3)
    if args.json:
        print(json.dumps(receipt, indent=2))
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "cells",
        help="storm cells as objects over WOOF-ArWen history: export volumes "
             "for the titan storm-cell engine, run it, and catalog every "
             "cell with the model's updraft, cloud and freezing-level "
             "attributes")
    commands = parser.add_subparsers(dest="cells_command", required=True)

    export = commands.add_parser(
        "export",
        help="write a titan input stream (input.tfs) from a wrfout series")
    _add_inputs(export)
    export.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="directory for input.tfs and export-receipt.json")
    _add_ladder(export)
    export.add_argument("--json", action="store_true",
                        help="print the export receipt as JSON")
    export.set_defaults(func=_export_cmd)

    analyze = commands.add_parser(
        "analyze",
        help="export, run titan analyze, and write the cell catalog into "
             "<out>/<domain>/cells/<first-valid-day>/")
    _add_inputs(analyze)
    analyze.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="the case folder; the series lands under "
             "<DIR>/<domain>/cells/<first-valid-day>/")
    analyze.add_argument(
        "--profile", choices=_titan.PROFILES, default=_titan.DEFAULT_PROFILE,
        help=f"titan threshold profile (default {_titan.DEFAULT_PROFILE})")
    analyze.add_argument(
        "--titan", type=Path, default=None, metavar="PATH",
        help=f"the titan binary (a separate program WOOF-ArWen does not ship; "
             f"without one, analyze is off); default resolves "
             f"${_titan.TITAN_ENV}, then the bridge directories, then PATH")
    analyze.add_argument(
        "--titan-config", type=Path, default=None, metavar="FILE",
        help="a titan key=value config overriding the profile")
    _add_ladder(analyze)
    analyze.add_argument(
        "--no-catalog", dest="catalog", action="store_false",
        help="stop after titan analyze; write no catalog")
    analyze.add_argument("--json", action="store_true",
                         help="print the analyze receipt as JSON")
    analyze.set_defaults(func=_analyze_cmd)

    catalog = commands.add_parser(
        "catalog",
        help="join an existing titan bundle to its wrfout series: "
             "catalog.csv, catalog.json, cells.geojson, overlays/")
    _add_inputs(catalog)
    catalog.add_argument(
        "--bundle", type=Path, required=True, metavar="DIR",
        help="a titan analyze bundle (frames.jsonl, tracks.json, ...)")
    catalog.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help="where the catalog files go")
    catalog.add_argument("--json", action="store_true",
                         help="print the catalog receipt as JSON")
    catalog.set_defaults(func=_catalog_cmd)


__all__ = ["register_cli"]
