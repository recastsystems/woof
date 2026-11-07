"""Common wave switches for every hex door. No GPU runtime at import."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os

MODE_ENV = "WOOF_HEX_WAVES"
OPTIONS_ENV = "WOOF_HEX_WAVES_OPTIONS"


def add_arguments(parser):
    parser.add_argument("--waves", choices=("off", "on"), default="off",
                        help="spectral ocean waves on native wet cells (default: off)")
    parser.add_argument("--waves-options", default="{}", metavar="JSON",
                        help="wave spectral, coupling, preparation and feedback options")


def configuration(mode=None, text=None):
    mode = os.environ.get(MODE_ENV, "off") if mode is None else mode
    text = os.environ.get(OPTIONS_ENV, "{}") if text is None else text
    if mode not in ("off", "on"):
        raise ValueError("waves must be on or off")
    supplied = json.loads(text) if isinstance(text, str) else dict(text)
    if not isinstance(supplied, dict):
        raise ValueError("waves-options must be a JSON object")
    allowed = {"source_pack", "spectral", "nf", "nd", "interval_seconds", "stations",
               "coupling_interval_seconds", "roughness_feedback", "prepared", "preparation",
               "geographic_cfl", "gse_spreading_seconds", "source_prepared"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError("unknown waves-options: " + ", ".join(sorted(unknown)))
    spectral = dict(frequencies=36, directions=36, first_hz=.035, ratio=1.1)
    spec = supplied.get("spectral", {})
    if not isinstance(spec, dict) or set(spec)-set(spectral):
        raise ValueError("spectral must contain declared frequency/direction parameters")
    spectral.update(spec)
    for alias, name in (("nf", "frequencies"), ("nd", "directions")):
        if alias in supplied:
            if name in spec and spec[name] != supplied[alias]:
                raise ValueError(f"wave {alias} and spectral {name} disagree")
            spectral[name] = supplied[alias]
    for key in ("frequencies", "directions"):
        value = spectral[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 4:
            raise ValueError(f"spectral {key} must be an integer >= 4")
    for key in ("first_hz", "ratio"):
        value = spectral[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"spectral {key} must be positive and finite")
    if spectral["ratio"] <= 1:
        raise ValueError("spectral frequency ratio must exceed one")
    result = dict(supplied, enabled=mode == "on", spectral=spectral)
    interval = supplied.get("coupling_interval_seconds", supplied.get("interval_seconds", 300.))
    if ("coupling_interval_seconds" in supplied and "interval_seconds" in supplied and
            supplied["coupling_interval_seconds"] != supplied["interval_seconds"]):
        raise ValueError("wave coupling interval aliases disagree")
    if isinstance(interval, bool) or not isinstance(interval, (float, int)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("wave coupling interval must be positive finite seconds")
    result["coupling_interval_seconds"] = float(interval)
    if "roughness_feedback" in supplied and not isinstance(supplied["roughness_feedback"], bool):
        raise ValueError("wave roughness_feedback must be boolean")
    if supplied.get("source_pack", "st4") not in ("st4", "janssen"):
        raise ValueError("wave source_pack must be st4 or janssen")
    if mode == "on":
        from woof.ocean.settings import DEFAULTS
        from woof.ocean.waves.model import SpectralGrid
        SpectralGrid(**spectral)
        result.setdefault("roughness_feedback", DEFAULTS["roughness_feedback"])
        result.setdefault("source_pack", DEFAULTS["source_pack"])
    return result


def argument_values(arguments):
    mode, text = getattr(arguments, "waves", "off"), getattr(arguments, "waves_options", "{}")
    configuration(mode, text)
    return ["--waves", mode, "--waves-options", text]


@contextmanager
def environment(arguments):
    """Preserve the caller's environment when multiple doors run in-process."""
    mode, text = getattr(arguments, "waves", "off"), getattr(arguments, "waves_options", "{}")
    configuration(mode, text)
    previous = {key: os.environ.get(key) for key in (MODE_ENV, OPTIONS_ENV)}
    os.environ[MODE_ENV], os.environ[OPTIONS_ENV] = mode, text
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
