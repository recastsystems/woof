"""One cycle of the coarse-then-corridor cascade, end to end.

Everything this project had built was a SNAPSHOT.  Detection, placement,
culling, a limited-area full-physics forecast and a composite render all
worked once, from one coarse run, at one hour.  Nothing had ever followed
weather from one cycle into the next: the plan layer grew hysteresis, slot
continuity and a move-or-stay decision with nobody to hand them to, and the
regional gate charged a forecast mint for every re-placed cull, which made
following weather cost about three forecasts per swath per cycle.

This module is the loop those parts were built for, and it is ONE function
over a table.  A cycle is:

    parent forecast -> detect + plan (hysteresis decides move or stay)
                    -> cull the parent, at the row's own pad
                    -> a mid-window initial condition, when the threat is late
                    -> boundaries from the coarse parent
                    -> a contract deck on THIS cull's rings
                    -> a registered row written from the cull's own bytes
                    -> full-physics fine forecast
                    -> composite render
                    -> the state the next cycle continues from

THE ARBITRARY ACCEPTANCE TEST.  There is no phenomenon anywhere in this file.
A tropical cyclone, a convective area, a fire-weather region and an
atmospheric river are ROWS in ``threat-metrics.v3``; they reach this loop as
``admitted`` entries carrying a cull region, a mesh spec, an ignition hour and
a pad, and this module cannot tell them apart.  Adding a phenomenon is a row.
If a future edit needs a branch on ``threat_class`` here, the design failed.

WHAT A CYCLE COSTS, and what the 2026-08-27 anchor re-keying changed.  Before
it, a re-placed cull paid a contract deck PLUS two 1,080-step forecast mints
-- 5.5 to 8.7 minutes of card measured on four meshes -- before its own 4-6
minute forecast was allowed to run.  Now the mint is a property of the
CONFIGURATION CLASS and is earned once; the residual per-geometry cost is the
deck alone, and this loop measures it every cycle and puts the number in its
receipt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import cascade_row
from ..cull_door import carry_lineage, cull_one
from ..swath.history import HistoryReader
from ..swath.hysteresis import SwathState
from ..swath.plan import plan_cycle, plan_document
from ..swath import registry as registry_module
from .delayed_start import XTIME_FORMAT, compose_mid_window_init
from .errors import CycleRefusal

#: How this project's own history frames label their valid time.
FRAME_LABEL_FORMAT = "%Y-%m-%d_%H.%M.%S"

CASCADE_SCHEMA = "gpuwm-hex.cascade-cycle/v1"


def _stamp(moment: datetime) -> str:
    return moment.strftime(XTIME_FORMAT)


def _label(moment: datetime) -> str:
    return moment.strftime(FRAME_LABEL_FORMAT)


def _parse_label(text: str) -> datetime:
    return datetime.strptime(str(text).strip()[:19], FRAME_LABEL_FORMAT)


# ---------------------------------------------------------------------------
# the parent's own frames, read off a run receipt
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParentStream:
    """A forecast this project ran, as a sequence of times and files."""

    receipt: Path
    grid: Path
    static: Path
    frames: tuple[tuple[datetime, Path, str], ...]
    #: The rw_mpas_lbc driving-source row that reads these frames: this program's ArWen-route history spells the
    #: edge-normal wind normal_u; the MPAS-physics route writes MPAS-A's own streams (u, xtime).
    lbc_source: str = "unstructured-port-stream"

    @property
    def start(self) -> datetime:
        return self.frames[0][0]

    def at(self, moment: datetime) -> tuple[datetime, Path, str]:
        for row in self.frames:
            if row[0] == moment:
                return row
        available = ", ".join(_label(row[0]) for row in self.frames[:6])
        raise CycleRefusal(
            f"the parent stream {self.receipt} publishes no frame valid at "
            f"{_stamp(moment)}.  A cycle cannot start a fine grid from a "
            f"state its parent never wrote; the frames it has begin "
            f"{available}...  Either move the cycle onto a published hour or "
            f"run the parent with a history cadence that covers it"
        )

    def window(self, first: datetime, last: datetime) -> list[tuple[datetime, Path, str]]:
        chosen = [row for row in self.frames if first <= row[0] <= last]
        if not chosen:
            raise CycleRefusal(
                f"the parent stream {self.receipt} publishes no frame between "
                f"{_stamp(first)} and {_stamp(last)}"
            )
        return chosen


def _mpas_suite_stream(receipt: Path) -> ParentStream | None:
    """The frames of an MPAS-physics forecast (``native-physics-receipt.json``), or None for another receipt.

    THE BREAKAGE THIS PREVENTS (2026-10-25, MPAS physics the hex default): that route writes MPAS's own streams,
    a ``history-ready.jsonl`` row per published frame and ``forecast-start.json`` (grid, static, start), and no
    door receipt with ``snapshot_files``, so ``cycle --coarse-history`` / ``--parent-history`` refused every
    default forecast as "names no history frames".  The frames are named by valid time
    (``history.<FRAME_LABEL_FORMAT>.nc``), so the label is read off the published file name.
    """

    folder = receipt.parent
    ready = folder / "history-ready.jsonl"
    start = folder / "forecast-start.json"
    if not (ready.is_file() and start.is_file()):
        return None
    try:
        begun = json.loads(start.read_text(encoding="utf-8"))
        rows_in = [json.loads(line) for line in ready.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise CycleRefusal(f"{folder} holds an unreadable MPAS-physics frame record ({error})") from error
    rows: list[tuple[datetime, Path, str]] = []
    for row in rows_in:
        path = Path(str(row.get("path", "")))
        if not path.is_file():
            path = folder / path.name
        name = path.name
        if not (name.startswith("history.") and name.endswith(".nc")):
            continue
        try:
            moment = _parse_label(name[len("history."):-len(".nc")])
        except ValueError as error:
            raise CycleRefusal(
                f"{path} is an MPAS-physics frame whose name carries no valid time; a frame index is not a time "
                f"and using one as a time would place every swath from the wrong hour") from error
        rows.append((moment, path, ""))
    if not rows:
        raise CycleRefusal(f"{receipt} names no history frames, so the forecast it describes wrote nothing a cycle "
                           f"could detect in or cull from")
    rows.sort(key=lambda item: item[0])
    return ParentStream(receipt=receipt, grid=Path(str(begun.get("grid", ""))),
                        static=Path(str(begun.get("static", ""))), frames=tuple(rows),
                        lbc_source="unstructured-native-stream")


def read_parent_stream(receipt: Path) -> ParentStream:
    """The frames a ``woof hex forecast`` run receipt names, with their times."""

    receipt = Path(receipt)
    try:
        document = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CycleRefusal(
            f"{receipt} is not a readable forecast run receipt ({error}). "
            f"The cascade reads the receipt this project's own forecast door "
            f"writes beside its frames, because the frames themselves carry "
            f"no valid time -- the history stream publishes no xtime"
        ) from error
    # TWO RECEIPT SHAPES, ONE READER.  ``woof hex forecast`` writes a DOOR
    # receipt that embeds the DRIVER receipt under ``driver_receipt``; the
    # engineering driver writes the driver receipt on its own beside the
    # frames.  Both name the same frames and the same grid, and a user who
    # points at the file their own door wrote should not have to know which
    # one this reader wanted.
    forecast = document.get("forecast") or {}
    if not forecast and isinstance(document.get("driver_receipt"), dict):
        forecast = document["driver_receipt"].get("forecast") or {}
    files = forecast.get("snapshot_files") or {}
    labels = forecast.get("history_labels") or {}
    authority_files = (forecast.get("authority") or {}).get("files") or {}
    grid = authority_files.get("grid") or {}
    # The STATIC, not the grid, is what a reader wants for connectivity: a
    # published grid file can carry sphere_radius=1, in which case its
    # areaCell is a unit-sphere area and every area gate downstream compares a
    # number near zero against its floor and rejects everything with no error.
    # Measured on g96.grid.nc.
    static = authority_files.get("static") or {}
    if not files and not labels:
        mpas = _mpas_suite_stream(receipt)
        if mpas is not None:
            return mpas
    if not files or not labels:
        raise CycleRefusal(
            f"{receipt} names no history frames, so the forecast it describes "
            f"wrote nothing a cycle could detect in or cull from"
        )
    rows: list[tuple[datetime, Path, str]] = []
    for step in sorted(files, key=lambda key: int(key)):
        entry = files[step]
        path = Path(str(entry["path"]))
        if not path.is_file():
            path = receipt.parent / path.name
        label = str(labels.get(step, ""))
        if not label:
            raise CycleRefusal(
                f"{receipt} records a frame at step {step} with no valid "
                f"time.  A frame index is not a time and using one as a time "
                f"would place every swath from the wrong hour"
            )
        rows.append((_parse_label(label), path, str(entry.get("sha256", ""))))
    rows.sort(key=lambda row: row[0])
    return ParentStream(
        receipt=receipt,
        grid=Path(str(grid.get("path", ""))),
        static=Path(str(static.get("path", ""))),
        frames=tuple(rows),
    )


def window_receipt(
    stream: ParentStream, first: datetime, last: datetime, out: Path
) -> Path:
    """A run receipt naming only the frames in one cycle's window.

    WHY THIS EXISTS AND WHAT IT DOES NOT CLAIM.  In an operational cascade the
    parent re-runs every cycle from a fresh analysis, and cycle N's detection
    reads cycle N's parent forecast.  The cascade has ONE parent integration, so
    a cycle's view of it is that integration WINDOWED -- the frames from this
    cycle's valid time forward, re-based so hour zero is this cycle's hour
    zero.  That is a faithful model of what a cycle sees and it is NOT the
    same thing as N independent parent runs: the later cycles read a forecast
    whose initial condition is older than an operational one would be, which
    makes their coarse guidance worse, not better.  The receipt records it.

    The frames' recorded digests are copied unchanged, so the reader that
    consumes this still verifies every frame against the digest the parent's
    own run receipt recorded.
    """

    chosen = stream.window(first, last)
    base = chosen[0][0]
    files: dict[str, Any] = {}
    labels: dict[str, str] = {}
    for index, (moment, path, digest) in enumerate(chosen):
        step = str(int((moment - base).total_seconds()))
        files[step] = {"path": str(path), "sha256": digest,
                       "bytes": path.stat().st_size if path.is_file() else 0}
        labels[step] = _label(moment)
    document = {
        "schema": "gpuwm-hex.cascade-window-receipt/v1",
        "derived_from": str(stream.receipt),
        "window_start": _stamp(first),
        "window_end": _stamp(last),
        "what_this_is": (
            "One cycle's view of a parent forecast: the frames from this "
            "cycle's valid time forward, with their own recorded digests, "
            "re-based so step 0 is this cycle's hour 0.  The cascade runs ONE "
            "parent integration and windows it per cycle rather than "
            "re-running the parent from a fresh analysis each cycle, which is "
            "what an operational cascade would do.  The difference is that "
            "later cycles here read guidance from an older initial condition "
            "than they would operationally -- worse guidance, not better."
        ),
        "forecast": {
            "snapshot_files": files,
            "history_labels": labels,
            "authority": {
                "files": {
                    "grid": {"path": str(stream.grid)},
                    "static": {"path": str(stream.static)},
                }
            },
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return out


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class CascadeConfig:
    """Everything one cascade run needs, and nothing about any phenomenon."""

    out: Path
    parent_row: str
    parent_grid: Path
    parent_static: Path
    parent_init: Path
    parent_history: Path | None
    coarse_history: Path
    coarse_parent_grid: Path
    gpuwm_checkout: Path | None
    repo: Path
    mesh_exe: Path | None = None
    lbc_exe: Path | None = None
    cycles: int = 2
    cycle_hours: float = 6.0
    plan_window_hours: float = 12.0
    fine_hours: float = 6.0
    history_every_minutes: int = 30
    dt_seconds: float = 20.0
    nominal_dx_m: float = 4_000.0
    class_id: str = "graded-4457m-dt20-z7"
    len_disp_m: float = 4_000.0
    max_slots_per_cycle: int = 1
    lbc_interval_seconds: int = 3600
    render: bool = True
    delayed_start: bool = True
    python: str = sys.executable
    state: Path | None = None
    metrics: Path | None = None
    policy: Path | None = None
    #: Flags for every fine forecast, after the cascade's own (e.g. ``--physics-suite=convection_permitting``).
    forecast_args: tuple[str, ...] = ()


@dataclass
class LegTiming:
    name: str
    seconds: float
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"leg": self.name, "seconds": round(self.seconds, 2), **dict(self.detail)}


# ---------------------------------------------------------------------------
# the legs
# ---------------------------------------------------------------------------
def _run(argv: Sequence[str], *, log: Path, env: Mapping[str, str] | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(os.environ)
    if env:
        merged.update(env)
    with open(log, "w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(item) for item in argv) + "\n")
        handle.flush()
        completed = subprocess.run(
            [str(item) for item in argv], stdout=handle, stderr=subprocess.STDOUT,
            env=merged,
        )
    return completed.returncode


def _door(config: CascadeConfig, *arguments: Any) -> list[str]:
    return [
        config.python, "-c",
        "import sys; from woof.hex.cli import main; sys.exit(main())",
        *[str(item) for item in arguments],
    ]


def _tool_env(config: CascadeConfig, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    # The subprocesses import the hexcore THIS process runs -- a checkout's
    # src/ or site-packages -- so every step of a cycle executes one copy.
    # Bytecode is written as usual: PYTHONDONTWRITEBYTECODE=1 here made every
    # leg of a cycle (and every compile worker under it) recompile each module
    # it imports that has no .pyc yet -- a checkout or an overlaid install --
    # on one core before its door opened (member-leg start-up measure,
    # 2026-10-28: about 0.25 s of the door's import and 0.45 s of the
    # forecast's own imports, per leg).
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    if extra:
        env.update(extra)
    return env


#: The legacy spelling of the boundary producer's environment variable.
#: ``woof.hex.engines.LBC`` carries it as a legacy name, so an install line
#: that set it keeps working; new lines set ``$WOOF_HEX_RW_MPAS_LBC``.
LBC_ENGINE_ENVIRONMENT = "RW_MPAS_LBC"


def resolve_lbc_engine(explicit: Path | None) -> Path:
    """The boundary producer, or a refusal naming what cannot be built.

    Delegates to :mod:`woof.hex.engines`, the ONE ladder every door in this
    distribution resolves through.  This function used to read
    ``$RW_MPAS_LBC`` and PATH and nothing else, so a box where
    ``woof fetch-bridges`` had staged ``rw_mpas_lbc`` into
    ``~/.woof/bridges`` -- and where the init door had just resolved
    ``rw_mpas_init`` from that very directory -- refused the boundary leg
    of the same chain.  The shared ladder reads the staging directory and
    the installed package's ``libexec/bridges``, and the refusal names the
    command that fills them.
    """

    from ..engines import LBC, EngineRefusal, resolve

    try:
        return resolve(LBC, explicit)
    except EngineRefusal as error:
        raise CycleRefusal(
            f"{error}  Every fine grid in this cascade is a limited-area cull "
            "whose seven boundary rings hold no atmosphere of their own; with "
            "no boundary producer the cycle can cut a mesh it can never "
            "integrate"
        ) from error


def build_boundaries(
    config: CascadeConfig,
    stream: ParentStream,
    child_init: Path,
    out_dir: Path,
    start: datetime,
    stop: datetime,
    log: Path,
) -> dict[str, Any]:
    """``rw_mpas_lbc`` over the coarse parent's own frames."""

    engine = resolve_lbc_engine(config.lbc_exe)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv: list[Any] = [
        engine, "--source", stream.lbc_source,
        "--grid", child_init,
        "--parent-grid", config.coarse_parent_grid,
        "--out-dir", out_dir,
        "--start-time", _stamp(start),
        "--stop-time", _stamp(stop),
        "--fg-interval-seconds", str(int(config.lbc_interval_seconds)),
        "--receipt", out_dir / "lbc-receipt.json",
    ]
    for moment, path, _ in stream.window(start, stop):
        argv += ["--interval", f"{_stamp(moment)}={path}"]
    started = time.perf_counter()
    code = _run(argv, log=log)
    elapsed = time.perf_counter() - started
    files = sorted(out_dir.glob("lbc.*.nc"))
    if code != 0 or not files:
        raise CycleRefusal(
            f"rw_mpas_lbc refused the boundary series for {out_dir.name} "
            f"(exit {code}, {len(files)} files); the engine's own message is "
            f"in {log}.  A limited-area grid integrated with nothing driving "
            f"its seven rings empties from the edge inward"
        )
    return {"seconds": elapsed, "files": len(files), "dir": str(out_dir)}


def run_contract_deck(
    config: CascadeConfig,
    grid: Path,
    init: Path,
    lbc_dir: Path,
    out: Path,
    mesh_row: str,
    log: Path,
    start: datetime,
) -> dict[str, Any]:
    """The per-geometry half of the anchor: this cull's own rings, measured.

    THIS IS THE RESIDUAL PER-GEOMETRY ADMISSION COST after the 2026-08-27
    re-keying, and the cascade times it every cycle so the number in the
    receipt is measured rather than quoted.
    """

    argv = [
        config.python, "-m", "woof.hex.drivers.run_cuda_regional_contract",
        "--grid", str(grid), "--init", str(init), "--lbc-dir", str(lbc_dir),
        "--class-id", config.class_id, "--mesh-row", mesh_row,
        "--start-time", _stamp(start),
        "--out", str(out),
    ]
    started = time.perf_counter()
    code = _run(argv, log=log, env=_tool_env(config))
    elapsed = time.perf_counter() - started
    if code != 0 or not out.is_file():
        raise CycleRefusal(
            f"the contract deck refused this cull (exit {code}); its output "
            f"is in {log}.  The 22 regional kernels are indexed by ring, so "
            f"without a deck on THESE rings nothing has ever checked this "
            f"cull's specified and relaxation zones against the v8.4.1 CPU "
            f"authority, and two identical runs of a wrong zone would agree "
            f"with each other"
        )
    document = json.loads(out.read_text(encoding="utf-8"))
    return {
        "seconds": elapsed,
        "receipt": str(out),
        "all_decks_bitwise": document.get("all_decks_bitwise"),
        "all_kernels_covered": document.get("all_kernels_covered"),
        "all_controls_have_teeth": document.get("all_controls_have_teeth"),
        "bdy_mask_sha256": document.get("bdy_mask_sha256"),
        "n_cells": document.get("n_cells"),
    }


def run_fine_forecast(
    config: CascadeConfig,
    *,
    mesh_row: str,
    grid: Path,
    static: Path,
    init: Path,
    lbc_dir: Path,
    start: datetime,
    out: Path,
    scratch: Path,
    receipt: Path,
    label: str,
    rows_file: Path,
    ledger_dir: Path,
    log: Path,
) -> dict[str, Any]:
    """The corridor's own full-physics forecast, through the shipped door."""

    argv = _door(
        config, "forecast", "--mesh", mesh_row,
        "--grid", grid, "--static", static, "--init", init,
        "--lbc-dir", lbc_dir,
        "--init-source",
        f"cycling cascade: parent {config.parent_row} culled at the swath's "
        f"own pad, state valid {_stamp(start)}",
        "--start-time", _stamp(start),
        "--hours", repr(float(config.fine_hours)),
        "--history-every-minutes", str(int(config.history_every_minutes)),
        # The cascade registered each cull at its own --dt, so the forecast
        # runs THAT timestep rather than the door's mesh-derived default.
        "--dt", "row",
        "--out", out, "--scratch", scratch, "--receipt", receipt,
        *(() if config.gpuwm_checkout is None
          else ("--gpuwm-checkout", config.gpuwm_checkout)),
        "--case-label", label,
        # The caller's forecast flags come last, so a --physics-suite (or a --dt) given here wins.
        *config.forecast_args,
    )
    started = time.perf_counter()
    code = _run(
        argv, log=log,
        env=_tool_env(config, {
            cascade_row.CASCADE_ROWS_ENVIRONMENT: str(rows_file),
            "WOOF_HEX_REGIONAL_CONTRACT_DIR": str(ledger_dir),
        }),
    )
    elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "returncode": code, "receipt": str(receipt),
            "out": str(out), "log": str(log)}


def render_cycle(
    config: CascadeConfig,
    *,
    history: Sequence[Path],
    mesh: Path,
    out: Path,
    scratch: Path,
    start: datetime,
    log: Path,
) -> dict[str, Any]:
    """Weather-field products, through the port's Rust render door."""

    argv = _door(
        config, "render",
        "--history", *[str(item) for item in history],
        "--mesh", mesh, "--out", out, "--scratch", scratch,
        "--simulation-start", _stamp(start),
    )
    started = time.perf_counter()
    code = _run(argv, log=log, env=_tool_env(config))
    return {
        "seconds": time.perf_counter() - started,
        "returncode": code,
        "out": str(out),
        "pngs": len(list(Path(out).rglob("*.png"))) if Path(out).is_dir() else 0,
        "log": str(log),
    }


# ---------------------------------------------------------------------------
# one slot
# ---------------------------------------------------------------------------
#: How much coarser than its parent's own finest edge a cull may be before
#: the cascade refuses to spend a fine forecast on it.  A cull moves no cell
#: centre, so a cull that CONTAINS the parent's refinement carries the
#: parent's finest edge EXACTLY -- the tolerance is here only so a float
#: comparison never turns an identical mesh into a refusal.
PARENT_RESOLUTION_TOLERANCE = 1.05


def finest_edge(grid: Path) -> float:
    """``min(dcEdge)`` off a grid file, in the FILE'S OWN units.

    Deliberately not converted, and the comparison that uses it is a RATIO for
    exactly that reason.  A published MPAS grid may store ``dcEdge`` on the
    unit sphere -- ``sphere_radius`` is 1 and the Earth-scaled metrics live in
    the static -- and the registry's own Courant admission reads the static
    for that reason.  A cull carries whatever convention its parent used, so
    parent and child are always in the same units as each other and their
    ratio is meaningful whether or not either is in metres.  Printing one of
    them as "m" would be a fabricated unit in a refusal message.
    """

    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        return float(
            np.min(np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64))
        )


def outside_the_parents_refinement(
    cull_edge: float, parent_edge: float, slot_id: str, cells: int
) -> str | None:
    """Refuse a swath the parent has no resolution for, and say why.

    THE BREAKAGE THIS PREVENTS, and it is a real one this cascade hit on its
    first end-to-end run.  A parent mesh is refined over the region SOME
    placement asked for.  A later cycle re-detects and can rank a swath
    somewhere else entirely -- and a cull taken there is a limited-area domain
    made of the parent's BACKGROUND cells.  Measured on the first run: 457
    cells whose finest edge is 70,983 m, against the parent's own 4,457 m.

    Nothing about that fails loudly.  It binds, it admits, and it integrates a
    71 km mesh at a 20 s timestep -- 35x below its own Courant limit -- and
    produces a full set of frames that look like a forecast and resolve
    nothing the coarse parent did not already have.  The cost is a whole fine
    slot, and the output is indistinguishable from a real one without reading
    dcEdge.

    An operational cascade answers this by REGENERATING the parent for the new
    placement (a mesh, a static and an init: about twenty minutes on this
    hardware).  A cascade holding one parent has to skip the slot and say so,
    which is what this does.
    """

    if cull_edge <= parent_edge * PARENT_RESOLUTION_TOLERANCE:
        return None
    ratio = cull_edge / parent_edge if parent_edge > 0.0 else float("inf")
    return (
        f"OUTSIDE-PARENT-REFINEMENT: slot {slot_id}'s cull is {cells} cells "
        f"whose finest edge is {ratio:.1f}x the parent's own "
        f"({cull_edge:.6g} against {parent_edge:.6g}, in the grid files' own "
        f"units -- a published grid may store dcEdge on the unit sphere, so "
        f"the comparison is a ratio and never a fabricated metre).  This "
        f"swath sits outside the region the "
        f"parent mesh is refined over, so a limited-area forecast here would "
        f"integrate the parent's BACKGROUND cells at a fine timestep and "
        f"resolve nothing the coarse run did not already hold -- a whole fine "
        f"slot spent on output nobody could tell from a real one without "
        f"reading dcEdge.  The remedy is to regenerate the parent for this "
        f"placement (mesh + static + init), which a cascade holding one fixed "
        f"parent does not do"
    )


def run_slot(
    config: CascadeConfig,
    row: Mapping[str, Any],
    *,
    cycle_index: int,
    cycle_start: datetime,
    coarse: ParentStream,
    parent: ParentStream | None,
    work: Path,
    ledger: Path,
    parent_edge: float = 0.0,
) -> dict[str, Any]:
    """Everything one admitted swath costs and produces, in order."""

    from ..engines import MESH, EngineRefusal, resolve

    slot_id = str(row["slot_id"])
    tag = f"c{cycle_index:02d}-{slot_id}"
    slot_dir = work / tag
    logs = slot_dir / "logs"
    cull_dir = slot_dir / "cull"
    cull_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    ignite_seconds = float(row.get("ignite_at_seconds") or 0.0)
    start = cycle_start + timedelta(seconds=ignite_seconds)
    stop = start + timedelta(hours=float(config.fine_hours))
    legs: list[dict[str, Any]] = []

    # A DELAYED START IS OWED WHENEVER THE SWATH WANTS TO BEGIN AFTER THE
    # PARENT'S INIT HOUR, and that is two different situations, not one.
    #
    #   * the CYCLE is later than the parent's init -- cycle 2 of a 6-hourly
    #     cascade wants 12Z and the parent was initialised at 06Z; and
    #   * the SWATH is later than its cycle -- a metric row whose
    #     start_policy is time_of_first_exceedance ignites at the hour the
    #     machine derived from the forecast itself.
    #
    # Both are the same defect if unanswered: the only initial condition a
    # cull could have was its parent's init, so covering the window meant
    # integrating the fine grid from the parent's hour zero and discarding
    # everything before the weather.  The saving is exactly the gap.
    #
    # The CONFIGURATION half of that is refused here, before anything is cut:
    # a run that cannot produce a mid-window state should not spend a second
    # cutting files it could never integrate.
    parent_init_time = parent.start if parent is not None else cycle_start
    lead_gap_hours = (start - parent_init_time).total_seconds() / 3600.0
    if lead_gap_hours > 0.0 and config.delayed_start and parent is None:
        raise CycleRefusal(
            f"slot {slot_id} wants to start at {_stamp(start)}, which is "
            f"{lead_gap_hours:.2f} h after the parent's own init hour, so its "
            f"initial condition must be the parent's state at that time -- "
            f"and no --parent-history was given.  Without one the only "
            f"initial condition available is the parent's init, and covering "
            f"this window would mean integrating the fine grid for "
            f"{lead_gap_hours:.2f} h of atmosphere nobody placed a grid for "
            f"before reaching the hours somebody did"
        )

    region_path = cull_dir / f"{tag}.cull-region.json"
    region_path.write_text(
        json.dumps(dict(row["cull_region"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    try:
        engine = resolve(MESH, config.mesh_exe)
    except EngineRefusal as error:
        raise CycleRefusal(str(error)) from error

    started = time.perf_counter()
    cut: dict[str, Any] = {}
    # THE GRID FIRST, AND ALONE, so the cheap question is asked before the
    # expensive files are cut: does the parent actually have resolution here?
    cut["grid"] = cull_one(
        engine, Path(config.parent_grid), region_path,
        cull_dir / f"{tag}.grid.nc",
        graph=cull_dir / f"{tag}.graph.info", clobber=True,
    )
    cut["grid"]["lineage"] = carry_lineage(
        Path(config.parent_grid), cull_dir / f"{tag}.grid.nc",
        drives_boundaries=False,
    )
    cull_edge = finest_edge(cull_dir / f"{tag}.grid.nc")
    cull_cells = int(cut["grid"].get("receipt", {}).get("region_cells") or 0)
    skipped = outside_the_parents_refinement(
        cull_edge, parent_edge, slot_id, cull_cells
    )
    if skipped is not None:
        return {
            "slot_id": slot_id,
            "tag": tag,
            "ran": False,
            "skipped": skipped,
            "metric_id": row.get("metric_id"),
            "threat_class": row.get("threat_class"),
            "cells": cull_cells,
            "cull_finest_edge": cull_edge,
            "parent_finest_edge": parent_edge,
            "finest_edge_ratio": round(cull_edge / parent_edge, 4) if parent_edge else None,
            "wall_seconds": round(time.perf_counter() - started, 2),
        }
    for role, parent_file in (
        ("static", config.parent_static),
        ("init", config.parent_init),
    ):
        target = cull_dir / f"{tag}.{role}.nc"
        cut[role] = cull_one(
            engine, Path(parent_file), region_path, target, clobber=True,
        )
        cut[role]["lineage"] = carry_lineage(
            Path(parent_file), target, drives_boundaries=(role == "init")
        )
    legs.append(LegTiming("cull", time.perf_counter() - started, {
        "cells": cut["grid"].get("receipt", {}).get("region_cells"),
        "parent_cells": cut["grid"].get("receipt", {}).get("parent_cells"),
        "pad": row.get("cull_pad_scale"),
    }).as_dict())

    grid = cull_dir / f"{tag}.grid.nc"
    static = cull_dir / f"{tag}.static.nc"
    init = cull_dir / f"{tag}.init.nc"

    delayed: dict[str, Any] | None = None
    if lead_gap_hours > 0.0 and config.delayed_start:
        started = time.perf_counter()
        _, frame, _ = parent.at(start)
        report = compose_mid_window_init(
            child_init=init,
            child_grid=grid,
            parent_grid=parent.grid if parent.grid.is_file() else Path(config.parent_grid),
            parent_history=frame,
            valid_time=start,
            receipt_path=slot_dir / "delayed-start.json",
        )
        delayed = report.as_dict()
        legs.append(LegTiming("delayed-start", time.perf_counter() - started, {
            "valid_time": report.valid_time,
            "cells_matched": report.cells_matched,
            "edges_matched": report.edges_matched,
            "carried_fields": len(report.carried),
            "not_carried_fields": len(report.not_carried),
            "parent_init_time": _stamp(parent_init_time),
            "hours_not_integrated": round(lead_gap_hours, 3),
        }).as_dict())

    boundaries = build_boundaries(
        config, coarse, init, slot_dir / "lbc", start, stop, logs / "lbc.log"
    )
    legs.append(LegTiming("boundaries", boundaries["seconds"], {
        "files": boundaries["files"]}).as_dict())

    mesh_row = f"cascade-{tag}"
    contract = run_contract_deck(
        config, grid, init, slot_dir / "lbc",
        slot_dir / "contract.json", mesh_row, logs / "contract.log",
        start=start,
    )
    legs.append(LegTiming("contract-deck", contract["seconds"], {
        "all_decks_bitwise": contract["all_decks_bitwise"],
        "n_cells": contract["n_cells"],
    }).as_dict())
    ledger.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(slot_dir / "contract.json", ledger / f"{tag}.contract.json")

    described = cascade_row.describe_cull(
        name=mesh_row,
        parent_row=config.parent_row,
        grid=grid,
        static=static,
        cull_receipt=cull_dir / f"{tag}.grid.nc.cull-receipt.json",
        init=init,
        dt_seconds=config.dt_seconds,
        nominal_dx_m=config.nominal_dx_m,
        lbc_source=f"{config.parent_row}-coarse-parent/{tag}",
        cycle_index=cycle_index,
        slot_id=slot_id,
        cull_pad_scale=float(row.get("cull_pad_scale") or 0.0),
    )
    rows_file = cascade_row.write_rows(slot_dir / "cascade-rows.json", [described])

    forecast = run_fine_forecast(
        config, mesh_row=mesh_row, grid=grid, static=static, init=init,
        lbc_dir=slot_dir / "lbc", start=start,
        out=slot_dir / "forecast", scratch=slot_dir / "scratch",
        receipt=slot_dir / "forecast-receipt.json",
        label=f"cascade-cycle{cycle_index}-{slot_id}",
        rows_file=rows_file, ledger_dir=ledger, log=logs / "forecast.log",
    )
    legs.append(LegTiming("fine-forecast", forecast["seconds"], {
        "returncode": forecast["returncode"]}).as_dict())

    rendered: dict[str, Any] | None = None
    # The state frames only.  The MPAS-physics route also writes MPAS's diag stream (diag.<time>.nc, no vertical
    # levels) beside its history frames; handed to the render door with them it stopped rw_mpas_convert on the
    # first file ("history has no nVertLevels dimension") and the slot rendered no picture (2026-10-25).
    frames = (sorted(path for path in (slot_dir / "forecast").glob("*.nc") if not path.name.startswith("diag."))
              if forecast["returncode"] == 0 else [])
    if config.render and frames:
        rendered = render_cycle(
            config, history=frames, mesh=grid, out=slot_dir / "render",
            scratch=slot_dir / "render-scratch", start=start,
            log=logs / "render.log",
        )
        legs.append(LegTiming("render", rendered["seconds"], {
            "pngs": rendered["pngs"]}).as_dict())

    return {
        "slot_id": slot_id,
        "tag": tag,
        "ran": True,
        "metric_id": row.get("metric_id"),
        "threat_class": row.get("threat_class"),
        "mesh_row": mesh_row,
        "cull_finest_edge": cull_edge,
        "parent_finest_edge": parent_edge,
        "finest_edge_ratio": round(cull_edge / parent_edge, 4) if parent_edge else None,
        "cull_pad_scale": row.get("cull_pad_scale"),
        "ignite_at_seconds": ignite_seconds,
        "start_time": _stamp(start),
        "parent_init_time": _stamp(parent_init_time),
        "lead_gap_hours": round(lead_gap_hours, 3),
        "stop_time": _stamp(stop),
        "hysteresis": row.get("hysteresis"),
        "cells": cut["grid"].get("receipt", {}).get("region_cells"),
        "bdy_mask_sha256": described.bdy_mask_sha256,
        "delayed_start": delayed,
        "contract": contract,
        "forecast": forecast,
        "render": rendered,
        "legs": legs,
        "wall_seconds": round(sum(item["seconds"] for item in legs), 2),
    }


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def run_cascade(config: CascadeConfig) -> dict[str, Any]:
    """Every cycle, in order, carrying its state forward."""

    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "contract-ledger"
    ledger.mkdir(parents=True, exist_ok=True)

    coarse = read_parent_stream(Path(config.coarse_history))
    parent = (
        read_parent_stream(Path(config.parent_history))
        if config.parent_history is not None
        else None
    )
    metrics = registry_module.load_metrics(config.metrics)
    policy = registry_module.load_policy(config.policy)
    state = SwathState.load(config.state) if config.state else SwathState.empty()
    # Measured once: the finest edge the parent actually carries.  Every cull
    # of it that contains the refinement carries this exact value, because a
    # cull moves no cell centre.
    parent_edge = finest_edge(Path(config.parent_grid))

    started_all = time.perf_counter()
    cycles: list[dict[str, Any]] = []
    for index in range(int(config.cycles)):
        cycle_start = coarse.start + timedelta(hours=index * float(config.cycle_hours))
        cycle_dir = out / f"cycle-{index + 1:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        window_end = cycle_start + timedelta(hours=float(config.plan_window_hours))
        receipt = window_receipt(
            coarse, cycle_start, window_end, cycle_dir / "coarse-window.json"
        )
        planned = time.perf_counter()
        with HistoryReader(receipt) as reader:
            result = plan_cycle(reader, metrics, policy, state=state, cycle_index=index + 1)
            document = json.loads(
                json.dumps(plan_document(reader, metrics, policy, result))
            )
        plan_seconds = time.perf_counter() - planned
        (cycle_dir / "swath-plan.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        # The pad each admitted row declares, carried onto the row the cascade
        # acts on: the loop never reads a pad of its own and has no default.
        for row in document["admitted"]:
            row["cull_pad_scale"] = metrics.metric_rows[
                row["metric_id"]
            ].swath.cull_pad_scale

        # WALK THE RANKING UNTIL max_slots HAVE ACTUALLY RUN.  A swath the
        # parent has no resolution for is skipped by name after a 1-3 s grid
        # cull, not run and not silently dropped, and the next candidate gets
        # the slot.  With a single fixed parent that is the whole difference
        # between a cascade that spends a fine forecast on 71 km cells and one
        # that says why it did not.
        slots: list[dict[str, Any]] = []
        ran = 0
        for row in document["admitted"]:
            if ran >= int(config.max_slots_per_cycle):
                break
            record = run_slot(
                config, row,
                cycle_index=index + 1,
                cycle_start=cycle_start,
                coarse=coarse,
                parent=parent,
                work=cycle_dir,
                ledger=ledger,
                parent_edge=parent_edge,
            )
            slots.append(record)
            if record.get("ran"):
                ran += 1
        state = SwathState.from_document(document["state"])
        (cycle_dir / "swath-state.json").write_text(
            json.dumps(document["state"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        cycles.append({
            "cycle_index": index + 1,
            "valid_time": _stamp(cycle_start),
            "plan_seconds": round(plan_seconds, 2),
            "admitted": len(document["admitted"]),
            "declined": len(document["declined"]),
            "churn": document["churn"],
            "ran": [item["slot_id"] for item in slots if item.get("ran")],
            "skipped": [
                {"slot_id": item["slot_id"], "reason": item["skipped"]}
                for item in slots if not item.get("ran")
            ],
            "parent_finest_edge": parent_edge,
            "slots": slots,
        })

    receipt_document = {
        "schema": CASCADE_SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.perf_counter() - started_all, 2),
        "parent_row": config.parent_row,
        "coarse_history": str(config.coarse_history),
        "parent_history": (
            None if config.parent_history is None else str(config.parent_history)
        ),
        "cycles": cycles,
        "metrics_document": {"schema": metrics.schema, "sha256": metrics.sha256},
        "policy_document": {"schema": policy.schema, "sha256": policy.sha256},
    }
    (out / "cascade-receipt.json").write_text(
        json.dumps(receipt_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return receipt_document


__all__ = [
    "CASCADE_SCHEMA",
    "CascadeConfig",
    "ParentStream",
    "read_parent_stream",
    "run_cascade",
    "run_slot",
    "window_receipt",
]
