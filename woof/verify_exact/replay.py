"""The WOOF arm: replay one recorded run on the card this process owns.

Every run goes through WOOF's native WRF-file door (the one door strict mode
has qualified: it keeps WRF's ``T = theta - 300`` words), reads the exact
``wrfinput_d01``/``wrfbdy_d01`` bytes WRF read, applies the combo's
WOOF-only settings the way the combo list's load check did, and integrates
in THIS process through the shared domain-tree runner.  Many runs share one
process, so the kernel cache is warm after the first.

Each history frame is decoded through the Rust NetCDF bridge, hashed in the
recording's form (:func:`woof.verify_exact.compare.field_digest`) and
dropped; only the arrays a later ULP measurement may need are kept.
"""

from __future__ import annotations

import contextlib
import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Mapping

import numpy as np

from woof.verify_exact.compare import FieldMeta, field_digest
from woof.verify_exact.recording import ReplayRun
from woof.config import ConfigurationRefused

_RADIATION_KEYS = ("ra_lw_physics", "ra_sw_physics")
_STORED_DTYPE = {"F32": "<f4", "F64": "<f8", "I32": "<i4", "I16": "<i2", "I8": "|i1",
                 "I64": "<i8", "U8": "|u1", "U16": "<u2", "U32": "<u4"}
#: Digest written for a field whose words cannot be stored in the reference's
#: dtype and shape; it can never equal a SHA-256.
NOT_REPRESENTABLE = "not-representable-in-the-recorded-form"
_FRAME = re.compile(r"wrfout_d01_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})$")


class WoofRefused(RuntimeError):
    """WOOF's door or loader refused the combo before integrating."""


class WoofCrashed(RuntimeError):
    """The integration raised after it started."""


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(value)


def set_toml_key(text: str, key: str, value) -> str:
    """Set ``key`` on every line that already assigns it, else add it to ``[shared]``.

    The same edit the combo list's load check (combo-sweep checks/check_load.py)
    applied when it proved every combo loads; replays apply WOOF-only settings
    exactly as that check did so a replay runs the configuration that was proven.
    """

    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    line = f"{key} = {_toml_value(value)}"
    if pattern.search(text):
        return pattern.sub(lambda _m: line, text)
    if "[shared]\n" not in text:
        raise ValueError(f"cannot place WOOF setting {key}: the imported TOML has no [shared] table")
    return text.replace("[shared]\n", "[shared]\n" + line + "\n", 1)


def _set_namelist_key(text: str, key: str, value: int) -> str:
    pattern = re.compile(rf"(?im)^([ \t]*){re.escape(key)}[ \t]*=.*$")
    if not pattern.search(text):
        raise ValueError(f"namelist has no {key} line to rewrite")
    return pattern.sub(lambda m: f"{m.group(1)}{key} = {value},", text)


def stage_run_directory(run: ReplayRun, directory: Path) -> Path:
    """A real.exe handoff directory for the door: the recorded bytes, linked.

    The namelist is the one WRF ran, with one rewrite the combo list's load
    check also made: a combo that enters through TOML (analytic radiation,
    which the importer cannot map) or that acknowledges longwave off is
    imported with radiation 1 and given its real radiation by its WOOF
    settings afterwards.
    """

    if run.inputs is None or run.namelist is None:
        raise WoofRefused("the recording has no real.exe inputs for this run (real.exe refused it)")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        target = directory / name
        if not target.exists():
            try:
                target.symlink_to(run.inputs / name)
            except OSError:
                shutil.copyfile(run.inputs / name, target)
    text = run.namelist.read_text(encoding="utf-8")
    combo = run.combo_record
    settings = combo.get("woof_settings") or {}
    if str(combo.get("woof_door", "")).startswith("toml"):
        for key in _RADIATION_KEYS:
            text = _set_namelist_key(text, key, 1)
    elif settings.get("acknowledgements"):
        text = _set_namelist_key(text, "ra_lw_physics", 1)
    (directory / "namelist.input").write_text(text, encoding="utf-8")
    return directory


#: The generation selectors a stock WRF v4.6.1 recording ran: each key's
#: oldest generation (woof.core.solar, woof.ysu_generations,
#: woof.noahmp_generation, config.URBAN_WATER_BALANCE_GENERATIONS).
WRF_461_REFERENCE_GENERATIONS = {
    "solar_generation": "wrf_461",
    "ysu_version": "wrf_461",
    "shinhong_version": "wrf_461",
    "noahmp_snow_age": "wrf_461",
    "urban_water_balance": "wrf_471",
    "ntu_version": "wrf_471",
    "udm_version": "wrf_471",
    "cu_gf_version": "wrf_461",
    "gwdo_version": "wrf_471",
    "windfarm_fitch_version": "wrf_471",
}


def resolve_replay(run: ReplayRun, directory: Path):
    """The door's resolved run with the combo's WOOF-only settings applied."""

    import tomllib

    from woof.experiment import build_experiment
    from woof.wrfinput_door import resolve_wrfinput_run

    settings = dict(run.combo_record.get("woof_settings") or {})
    acknowledgements = list(settings.pop("acknowledgements", []) or [])
    variant = settings.get("ra_rrtmg_variant")
    resolved = resolve_wrfinput_run(directory, rrtmg_variant=variant)
    text = resolved.toml_text
    if acknowledgements:
        settings.setdefault("ra_lw_physics", 0)
        settings.setdefault("ra_sw_physics", 1)
        settings.setdefault("ra_physics", 0)
        if "[experiment]\n" not in text:
            raise ValueError("cannot place acknowledgements: the imported TOML has no [experiment] table")
        text = text.replace("[experiment]\n", "[experiment]\nacknowledgements = "
                            + _toml_value(acknowledgements) + "\n", 1)
    # A WRF comparison runs the matched arithmetic: from 2.9.0 the engine
    # default is correct mode, so exact is named here, key by key, unless the
    # combo's own WOOF settings name a correction key itself.
    from woof.correct_mode import CORRECTION_KEYS
    for key in CORRECTION_KEYS:
        settings.setdefault(key, False)
    # The recordings are stock WRF v4.6.1 runs, and from 2.9.0 four
    # selectors default to a later WRF generation (the WRF 4.8 sun and YSU,
    # the WRF 4.7 Noah-MP snow aging and the WRF 4.8 urban water balance).
    # A replay names the generation its reference ran, unless the combo's
    # own WOOF settings name one: otherwise the first radiation call (or
    # the first YSU column) already leaves the reference.
    for key, generation in WRF_461_REFERENCE_GENERATIONS.items():
        settings.setdefault(key, generation)
    for key, value in settings.items():
        if key == "ra_rrtmg_variant" and "ra_rrtmg_variant" not in text:
            continue
        text = set_toml_key(text, key, value)
    if text == resolved.toml_text:
        return resolved
    experiment = build_experiment(tomllib.loads(text), source=f"{directory}:verify-exact")
    return dataclasses.replace(resolved, toml_text=text, experiment=experiment)


def time_step_seconds(namelist: Path) -> float:
    from woof.fortran_namelist import parse_namelist
    domains = parse_namelist(namelist).get("domains", {})
    step = float(domains["time_step"][0])
    num = float((domains.get("time_step_fract_num") or [0])[0])
    den = float((domains.get("time_step_fract_den") or [1])[0]) or 1.0
    return step + num / den


@contextlib.contextmanager
def redirected_output(log_path: Path):
    """Send this process's stdout and stderr (Python and native) to a file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with log_path.open("ab") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


#: What the door and the run preparation raise to refuse a combo by name.
#: Only explicit validator refusals qualify. Generic exceptions can be code
#: bugs and must remain crashes. UnsupportedConfiguration keeps the old
#: NotImplementedError catch compatibility for scalar_pblmix callers.
REFUSAL_ERRORS = (ConfigurationRefused,)


def integrate(resolved, workdir: Path, *, run_seconds: float) -> None:
    """Prepare and integrate in this process; history every frame into ``workdir/wrfout``."""

    from woof.prepared_domain_tree_forecast import run_prepared_tree
    from woof.progress_log import ProgressOptions
    from woof.wrfinput_forecast import WrfInitialization, prepare_wrf_run

    try:
        inputs = prepare_wrf_run(resolved, workdir / "input", run_seconds=run_seconds)
    except REFUSAL_ERRORS as error:
        raise WoofRefused(f"{type(error).__name__}: {error}") from error
    except Exception as error:  # noqa: BLE001 - reported as this run's crash
        raise WoofCrashed(f"{type(error).__name__}: {error}") from error
    try:
        run_prepared_tree(inputs, output_directory=workdir, io_mode="history",
                          initialization=WrfInitialization(inputs),
                          progress_options=ProgressOptions(frame_markers=False))
    except ConfigurationRefused as error:
        raise WoofRefused(f"{type(error).__name__}: {error}") from error
    except Exception as error:  # noqa: BLE001 - reported as this run's crash
        raise WoofCrashed(f"{type(error).__name__}: {error}") from error


def frame_files(wrfout_dir: Path, *, start: datetime, step_seconds: float) -> dict[int, Path]:
    """Model step of every history file, from the valid time in its name."""

    frames: dict[int, Path] = {}
    for path in sorted(Path(wrfout_dir).glob("wrfout_d01_*")):
        match = _FRAME.match(path.name)
        if not match:
            continue
        valid = datetime.strptime(f"{match[1]} {match[2]}:{match[3]}:{match[4]}", "%Y-%m-%d %H:%M:%S")
        steps = (valid - start).total_seconds() / step_seconds
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError(f"{path.name} is not on a step boundary of {step_seconds} s")
        frames[int(round(steps))] = path
    return frames


def read_frame(path: Path) -> dict[str, np.ndarray]:
    """Every numeric field of one single-frame history file, Time removed."""

    from woof import netcdf_bridge

    dataset = netcdf_bridge.open_dataset(path)
    dataset.set_auto_maskandscale(False)  # the stored words, nothing decoded
    names = [n for n, v in dataset.variables.items()
             if not v.is_character and v.dimensions[:1] == ("Time",)]
    dataset.prefetch(names)
    arrays = {}
    for name in names:
        variable = dataset.variables[name]
        values = np.asarray(variable[:])
        dtype = _STORED_DTYPE.get(variable.stored_dtype)
        if dtype is not None:
            with np.errstate(invalid="ignore", over="ignore"):
                values = values.astype(dtype)
        arrays[name] = values[0]
    return arrays


@dataclasses.dataclass
class WoofRun:
    """What one replay produced: digests by step, kept arrays, status."""

    status: str  # OK, REFUSED, CRASH
    error: str
    wall_s: float
    fields: dict[str, FieldMeta]
    digests: dict[int, dict[str, str | None]]
    kept: dict[int, dict[str, np.ndarray]]
    nonfinite_step: int | None
    steps_per_frame: int
    #: Wall seconds of each phase: stage (door and settings), integrate, read (decode and hash).
    phases: dict[str, float] = dataclasses.field(default_factory=dict)


def replay(run: ReplayRun, workdir: Path, *, reference_meta: Mapping[str, FieldMeta],
           keep_steps: set[int], run_seconds: float, step_seconds: float,
           start: datetime, keep_history: bool = False) -> WoofRun:
    """Run WOOF once for ``run`` and hash every frame it wrote.

    ``reference_meta`` gives the dtype and shape each field is hashed in
    (the reference's own); a field the reference lacks is hashed as WOOF
    stored it.  Arrays at ``keep_steps`` are returned for ULP measurement.
    """

    started = time.perf_counter()
    phases: dict[str, float] = {}
    status, error = "OK", ""
    stage = workdir / "handoff"
    log = workdir / "run.log"
    try:
        with redirected_output(log):
            stage_run_directory(run, stage)
            try:
                resolved = resolve_replay(run, stage)
            except REFUSAL_ERRORS as refusal:
                raise WoofRefused(f"{type(refusal).__name__}: {refusal}") from refusal
            except Exception as error:  # noqa: BLE001 - reported as this run's crash
                raise WoofCrashed(f"{type(error).__name__}: {error}") from error
            finally:
                phases["stage"] = round(time.perf_counter() - started, 2)
            integrate(resolved, workdir / "out", run_seconds=run_seconds)
    except WoofRefused as refusal:
        status, error = "REFUSED", str(refusal)
    except WoofCrashed as crash:
        status, error = "CRASH", str(crash)
    except Exception as crash:  # staging bugs are crashes too, never process exits
        status, error = "CRASH", f"{type(crash).__name__}: {crash}"
    fields: dict[str, FieldMeta] = {}
    digests: dict[int, dict[str, str | None]] = {}
    kept: dict[int, dict[str, np.ndarray]] = {}
    nonfinite = None
    try:
        frames = frame_files(workdir / "out" / "wrfout", start=start, step_seconds=step_seconds) \
            if (workdir / "out" / "wrfout").is_dir() else {}
    except Exception as crash:
        status, error = "CRASH", f"{type(crash).__name__}: {crash}"
        frames = {}
    phases["integrate"] = round(time.perf_counter() - started - phases.get("stage", 0.0), 2)
    if status == "REFUSED" and frames:
        status = "CRASH"
    read_started = time.perf_counter()
    per_frame = 1
    if len(frames) > 1:
        steps = sorted(frames)
        per_frame = min(b - a for a, b in zip(steps, steps[1:]))
    for step, arrays in _read_frames(frames):
        if isinstance(arrays, BaseException):  # a half-written last frame
            if status == "OK":
                status, error = "CRASH", f"unreadable history frame at step {step}: {arrays}"
            break
        row: dict[str, str | None] = {}
        for name, values in arrays.items():
            # Hashed in the reference's dtype, in WOOF's own shape: a field
            # WOOF writes in another shape is reported as not comparable,
            # not as a value difference.
            ref = reference_meta.get(name)
            meta = FieldMeta(name, ref.dtype if ref else values.dtype.str, tuple(values.shape))
            fields.setdefault(name, meta)
            # A word the recorded form cannot hold is a difference, never absence.
            row[name] = field_digest(values, meta.dtype, meta.shape) or NOT_REPRESENTABLE
            if (nonfinite is None and values.dtype.kind == "f"
                    and not np.isfinite(values).all()):
                nonfinite = step
        digests[step] = row
        if step in keep_steps:
            kept[step] = arrays
    phases["read"] = round(time.perf_counter() - read_started, 2)
    if not keep_history:
        shutil.rmtree(workdir / "out", ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
    return WoofRun(status, error, round(time.perf_counter() - started, 2), fields,
                   digests, kept, nonfinite, per_frame, phases)


def _read_frames(frames: Mapping[int, Path], workers: int = 8):
    """``(step, arrays)`` in step order; each decode is its own bridge process,
    so several run at once.  A frame that fails to decode yields its exception
    in place of the arrays."""

    from concurrent.futures import ThreadPoolExecutor

    def one(path):
        try:
            return read_frame(path)
        except Exception as failure:  # noqa: BLE001 - reported by the caller
            return failure

    steps = sorted(frames)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(steps)))) as pool:
        yield from zip(steps, pool.map(one, (frames[s] for s in steps)))
