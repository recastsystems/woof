"""Wave run options shared by TOML, run and go; no CUDA imports."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os

_ENV = "WOOF_WAVES_LAUNCH"
DEFAULTS = {
    "source_pack": "st4",
    "nf": 36,
    "nd": 36,
    "coarsening": 2,
    "interval_seconds": 300.0,
    "render_interval_seconds": 21600.0,
    "roughness_feedback": False,
    "stations": "auto",
}


def options(value=None):
    if value in (None, ""):
        value = {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"waves-options must be a JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("waves-options must be a JSON object or TOML table")
    unknown = set(value) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown waves-options: {sorted(unknown)}")
    result = dict(DEFAULTS, **value)
    if result["source_pack"] not in ("st4", "janssen"):
        raise ValueError("wave source_pack must be st4 or janssen")
    for key, minimum in (("nf", 4), ("nd", 4), ("coarsening", 1)):
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(f"wave {key} must be an integer >= {minimum}")
    for key in ("interval_seconds","render_interval_seconds"):
        interval = result[key]
        if (type(interval) not in (int, float) or not math.isfinite(interval)
                or interval <= 0):
            raise ValueError(f"wave {key} must be finite and positive")
    if type(result["roughness_feedback"]) is not bool:
        raise ValueError("wave roughness_feedback must be true or false")
    stations = result["stations"]
    if stations != "auto" and not (
            isinstance(stations, list)
            and all(isinstance(item, str) and item for item in stations)
            and len(stations) == len(set(stations))):
        raise ValueError("wave stations must be auto or a unique list of station IDs")
    return result


def resolve(cfg):
    """CLI overrides inherit through the launcher's existing subprocess tree."""
    enabled = getattr(cfg, "waves", "off")
    configured = getattr(cfg, "waves_options", "")
    if enabled not in ("on", "off"):
        raise ValueError('waves must be "on" or "off"')
    result = options(configured)
    inherited = json.loads(os.environ.get(_ENV, "{}"))
    if "enabled" in inherited:
        enabled = inherited["enabled"]
    # Merge only explicit keys; a CLI knob must not reset other TOML knobs.
    if "options" in inherited:
        result = options(dict(result, **inherited["options"]))
    return enabled == "on", result


def add_arguments(parser):
    parser.add_argument("--waves", choices=("on", "off"), default=None,
                        help="couple spectral ocean waves on the parent grid (default off)")
    parser.add_argument("--waves-options", default=None, metavar="JSON",
                        help="wave source_pack, nf, nd, coarsening, interval_seconds, "
                             "render_interval_seconds, roughness_feedback and stations")


@contextmanager
def launch_scope(enabled=None, value=None):
    previous = os.environ.get(_ENV)
    if enabled is None and value is None:
        yield
        return
    override = json.loads(previous or "{}")
    if enabled is not None:
        if enabled not in ("on", "off"):
            raise ValueError('waves must be "on" or "off"')
        override["enabled"] = enabled
    if value is not None:
        options(value)  # validate before any preparation or child launch
        explicit = json.loads(value) if isinstance(value, str) else value
        override["options"] = dict(override.get("options", {}), **explicit)
    os.environ[_ENV] = json.dumps(override, sort_keys=True)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = previous
