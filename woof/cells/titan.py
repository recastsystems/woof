"""The ``titan`` binary: where it is, how it is run, what it wrote.

titan is a separate program that WOOF-ArWen does not ship or depend on,
resolved the way every other built artifact in this project is
(:mod:`woof.bridges`): an environment override first, then the bridge
directories an install stages into.  It is not vendored here and nothing
in this package reimplements any of it.  The door is optional: when no
binary resolves, ``woof cells analyze`` is off and says so in one plain
message, because a cell catalog produced by a second segmentation would
not be a titan catalog; ``woof cells export`` and ``woof cells
catalog`` need no titan at all.

The bundle reader is deliberately thin: it loads the JSON the engine
wrote and indexes it by frame and by track, adding nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from woof.bridges import (accept_resolved, default_bridge_dir,
                           legacy_bridge_candidates,
                           executable_name, packaged_bridge_dir)

#: Environment variable naming a built ``titan`` binary.
TITAN_ENV = "WOOF_TITAN"

#: The program's name, and so its filename stem.
TITAN_NAME = "titan"

#: The engine profiles ``titan analyze`` accepts, as its own usage lists
#: them.  ``severe`` (30/45 dBZ envelope/core, 3 km^3 minimum) is the
#: default here as it is titan's own, and by measurement: on a 2 km
#: ArWen run of a real outbreak (48 frames), ``research`` (25/40 dBZ,
#: 0.5 km^3) added ~2,150 objects of median area 12 km^2 (three cells)
#: and median peak updraft 0.7 m/s over ``severe``'s 2,618, and split
#: the merged convective line no better -- the 25 dBZ envelope merges a
#: model MCS as readily as the 30 dBZ one.  ``--profile research`` and
#: ``--titan-config`` remain the way to ask for the smaller objects.
PROFILES = ("legacy", "severe", "research", "operational")
DEFAULT_PROFILE = "severe"

#: The files an analyze bundle holds.
BUNDLE_FILES = ("frames.jsonl", "snapshot.json", "summary.json",
                "tracks.json", "lineage.json", "lineage.dot",
                "objects.geojson", "forecasts.geojson", "resolved.cfg")

_TIMEOUT_S = 3600


class TitanMissing(RuntimeError):
    """No titan binary; the refusal names the breakage and the remedy."""


class TitanFailed(RuntimeError):
    """titan ran and refused or crashed; carries its own stderr."""


def titan_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for ``titan``, best first.

    The bridge ladder's shape, minus the crate rungs (titan-rs is not a
    crate of this tree): the environment override, the install's
    ``libexec/bridges``, the packaged bridge directory, ``~/.woof/bridges``,
    then the PATH.
    """

    filename = executable_name(TITAN_NAME)
    candidates: list[Path] = []
    override = os.environ.get(TITAN_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend((
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
        *legacy_bridge_candidates(filename),
    ))
    on_path = shutil.which(TITAN_NAME)
    if on_path:
        candidates.append(Path(on_path))
    return tuple(candidates)


def find_titan(explicit: Path | str | None = None) -> Path | None:
    """First existing candidate, or None.

    An explicit path (``--titan``) or an environment override naming a
    missing file is a hard error: explicit configuration must fail
    loudly, not fall through to a different binary.
    """

    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(
                f"--titan names a missing file: {path}.  Point it at a "
                f"titan binary, or drop the flag to use the resolution "
                f"ladder.")
        return accept_resolved(path.resolve())
    override = os.environ.get(TITAN_ENV)
    for candidate in titan_candidates():
        if candidate.is_file():
            return accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{TITAN_ENV} names a missing file: {candidate}.  Point it at "
                f"a built titan binary, or unset {TITAN_ENV} to use the "
                f"resolution ladder.")
    return None


def titan_refusal(what: str) -> str:
    """The sentence a door prints when there is no titan to run."""

    ladder = "\n".join(f"  {path}" for path in titan_candidates())
    return (
        f"{what} is off on this install: it runs the titan storm-cell "
        f"engine, a separate program WOOF-ArWen does not ship, and none is "
        f"installed.  Without it there are no cell objects, tracks or "
        f"trends to catalog, and woof does not substitute a second "
        f"segmentation.  woof cells export works without it.  If you have "
        f"a titan binary, set {TITAN_ENV} to it, pass --titan PATH, or "
        f"place {executable_name(TITAN_NAME)} in one of:\n{ladder}")


def resolve_titan(explicit: Path | str | None = None, *,
                  what: str = "woof cells analyze") -> Path:
    found = find_titan(explicit)
    if found is None:
        raise TitanMissing(titan_refusal(what))
    return found


def titan_version(titan: Path) -> str:
    completed = subprocess.run(
        [os.fspath(titan), "version"], capture_output=True, text=True,
        errors="replace", timeout=60)
    if completed.returncode != 0:
        raise TitanFailed(
            f"{titan} version failed (exit {completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}")
    return completed.stdout.strip()


def run_titan(titan: Path, args: list[str], *, timeout_s: float = _TIMEOUT_S
              ) -> subprocess.CompletedProcess:
    command = [os.fspath(titan), *args]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise TitanFailed(
            f"titan {args[0]} exceeded {timeout_s:.0f} s and was stopped: "
            f"{error}") from None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise TitanFailed(
            f"titan {' '.join(args)} failed (exit {completed.returncode}): "
            f"{detail}")
    return completed


def parse_config_text(text: str) -> dict[str, str]:
    """``key=value`` lines, titan's own config grammar."""

    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def profile_config(titan: Path, profile: str) -> dict[str, str]:
    """The engine's own resolved values for ``profile``."""

    return parse_config_text(
        run_titan(titan, ["print-config", "--profile", profile]).stdout)


#: How many frame intervals the trend window must span, and how many a
#: track may coast across.  Three intervals give the trend four points;
#: two give one missed frame before a track ends.
TREND_INTERVALS = 3
GAP_INTERVALS = 2


def cadence_overrides(timestamps_ms, config: dict[str, str]
                      ) -> tuple[dict[str, str], float | None]:
    """titan keys to raise for this series' cadence, and the cadence.

    titan's profiles are tuned for radar scans a few minutes apart:
    the trend that advects every forecast footprint is fitted over
    ``forecast_history_s`` (1,800 s in every profile) and a track ends
    after ``max_gap_seconds`` without an observation.  Model history
    comes at whatever interval the run wrote -- 15 min, or an hour --
    and at an hour the window holds one point, so every cell's trend is
    exactly zero and every footprint stands still.  The measured median
    interval sizes both keys; nothing is lowered, and a caller's own
    config still wins.
    """

    stamps = sorted(int(t) for t in timestamps_ms)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    if not gaps:
        return {}, None
    gaps.sort()
    interval_s = gaps[len(gaps) // 2] / 1000.0
    overrides: dict[str, str] = {}
    history = float(config.get("forecast_history_s", 0) or 0)
    wanted = TREND_INTERVALS * interval_s
    if history < wanted:
        overrides["forecast_history_s"] = f"{int(round(wanted))}"
    gap = float(config.get("max_gap_seconds", 0) or 0)
    wanted_gap = GAP_INTERVALS * interval_s
    if gap < wanted_gap:
        overrides["max_gap_seconds"] = f"{int(round(wanted_gap))}"
    return overrides, interval_s


def analyze(titan: Path, stream: Path, out_dir: Path, *,
            profile: str = DEFAULT_PROFILE, config: Path | None = None,
            timestamps_ms=None) -> dict:
    """``titan analyze`` into ``out_dir``; returns a timing receipt.

    With ``timestamps_ms`` (the frames' instants), the profile is fitted
    to the series' cadence -- see :func:`cadence_overrides` -- through a
    generated config beside the bundle; a caller's ``config`` is merged
    over it, so an explicit key is never overridden.
    """

    if profile not in PROFILES:
        raise TitanFailed(
            f"profile {profile!r} is not one titan defines "
            f"({', '.join(PROFILES)})")
    out_dir.mkdir(parents=True, exist_ok=True)
    cadence: dict = {"interval_s": None, "overrides": {}}
    effective_config = config
    if timestamps_ms is not None:
        base = profile_config(titan, profile)
        user = parse_config_text(config.read_text("utf-8")) if config else {}
        overrides, interval = cadence_overrides(timestamps_ms, {**base, **user})
        overrides = {k: v for k, v in overrides.items() if k not in user}
        cadence = {"interval_s": interval, "overrides": overrides}
        if overrides or user:
            merged = {**overrides, **user}
            effective_config = out_dir.parent / "titan.cfg"
            effective_config.write_text(
                "".join(f"{k}={v}\n" for k, v in merged.items()),
                encoding="utf-8")
    args = ["analyze", "--input", os.fspath(stream), "--out",
            os.fspath(out_dir), "--profile", profile]
    if effective_config is not None:
        args += ["--config", os.fspath(effective_config)]
    started = time.perf_counter()
    completed = run_titan(titan, args)
    seconds = time.perf_counter() - started
    missing = [name for name in BUNDLE_FILES
               if not (out_dir / name).is_file()]
    if missing:
        raise TitanFailed(
            f"titan analyze exited 0 but the bundle at {out_dir} lacks "
            f"{', '.join(missing)}")
    return {
        "titan": os.fspath(titan), "version": titan_version(titan),
        "argv": args, "profile": profile,
        "config": None if effective_config is None else os.fspath(effective_config),
        "user_config": None if config is None else os.fspath(config),
        "cadence": cadence,
        "stdout": completed.stdout.strip(),
        "wall_seconds": round(seconds, 3), "bundle": os.fspath(out_dir),
    }


def inspect(titan: Path, stream: Path) -> str:
    return run_titan(titan, ["inspect", "--input", os.fspath(stream)]).stdout


@dataclass
class Bundle:
    """An analyze bundle, loaded and indexed."""

    path: Path
    frames: list[dict]
    tracks: dict[int, dict]
    lineage: dict
    summary: dict
    resolved_config: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "Bundle":
        root = Path(path)
        missing = [name for name in ("frames.jsonl", "tracks.json",
                                     "lineage.json", "summary.json")
                   if not (root / name).is_file()]
        if missing:
            raise TitanFailed(
                f"{root} is not a titan bundle: missing {', '.join(missing)}")
        frames = []
        with open(root / "frames.jsonl", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        frames.sort(key=lambda frame: frame["timestamp_ms"])
        tracks = {int(track["track_id"]): track
                  for track in json.loads((root / "tracks.json").read_text("utf-8"))}
        lineage = json.loads((root / "lineage.json").read_text("utf-8"))
        summary = json.loads((root / "summary.json").read_text("utf-8"))
        config: dict[str, str] = {}
        resolved = root / "resolved.cfg"
        if resolved.is_file():
            for line in resolved.read_text("utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
        return cls(path=root, frames=frames, tracks=tracks, lineage=lineage,
                   summary=summary, resolved_config=config)

    def track_of(self, frame: dict) -> dict[int, dict]:
        """``object_id -> assignment`` for one frame."""

        return {int(row["object_id"]): row
                for row in frame.get("tracking", {}).get("assignments", [])}

    def active_track_state(self, frame: dict) -> dict[int, dict]:
        return {int(row["track_id"]): row
                for row in frame.get("tracking", {}).get("active_tracks", [])}

    def forecasts_of(self, frame: dict) -> dict[int, dict]:
        return {int(row["track_id"]): row
                for row in frame.get("forecasts", [])}


__all__ = [
    "BUNDLE_FILES", "Bundle", "DEFAULT_PROFILE", "PROFILES", "TITAN_ENV",
    "TITAN_NAME", "TitanFailed", "TitanMissing", "analyze",
    "cadence_overrides", "find_titan", "inspect", "parse_config_text",
    "profile_config", "resolve_titan", "run_titan", "titan_candidates",
    "titan_refusal", "titan_version",
]
