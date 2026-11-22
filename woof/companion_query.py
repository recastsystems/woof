"""Read-only, native geometry for the companion and other CLI clients.

Configuration resolution stays in WOOF-ArWen. Grid coordinates come from the
installed Rust static-fields bridge, including its registered nest placement.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from datetime import date, datetime, time


def inspect_configuration(path: Path) -> dict:
    from woof.experiment import load_experiment, experiment_config_document
    from woof.companion_domains import native_domain_outlines, domain_geojson

    path = path.resolve(strict=True)
    payload = path.read_bytes()
    exp = load_experiment(path)
    raw = tomllib.loads(payload.decode("utf-8-sig"))
    domains = native_domain_outlines(exp)
    if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
        raise ValueError("The configuration changed while its map outline was being read; refresh it.")
    return {
        "schema": "arwen.companion-configuration.v1", "config_path": str(path),
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "geometry_backend": "rust-static-fields", "domains": domains,
        "domains_geojson": domain_geojson(domains),
        "experiment": experiment_config_document(exp),
        "tiles": exp.tiles.to_mapping(),
        "fetch": raw.get("fetch", {}),
        "case_data": {key: value for key, value in raw.get("case_data", {}).items()
                      if key in ("forcing_interval_s", "start_time", "end_time", "wps_namelist")},
    }


def main(args):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = inspect_configuration(args.config)
        encoded = json.dumps(result, allow_nan=False, default=_json_value)
    except Exception as exc:
        print(json.dumps({"schema": "arwen.companion-configuration.v1", "error": str(exc)}))
        return 1
    print(encoded)
    return 0


def _json_value(value):
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported configuration JSON type: {type(value).__name__}")


def register_cli(subparsers):
    parser = subparsers.add_parser("companion-query", help="read saved configuration and native domain outlines as JSON")
    parser.add_argument("config", type=Path)
    parser.set_defaults(func=main)
