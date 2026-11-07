"""Hash-bound JSON and plain array transport for complete-tree member legs.

Prepared weather inputs remain in their single shared cache. Workers reapply
the ordinary public preflight. Requests never carry a pickle, code import,
fixture callback or a second prepared bundle.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import re
import time
import contextlib
import functools
import socket

import numpy as np

#: Monotonic clock at import, the worker's start-up as near as Python sees it.
_PROCESS_STARTED = time.monotonic()


def _seconds_since_exec() -> float | None:
    """Seconds since this process was created (Linux /proc), or None: the
    interpreter start and imports before this module, which no timer inside
    Python sees."""
    try:
        with open("/proc/self/stat") as stream:
            fields = stream.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/uptime") as stream:
            uptime = float(stream.read().split()[0])
        return round(uptime - start_ticks / os.sysconf("SC_CLK_TCK"), 3)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


_SINCE_EXEC_AT_IMPORT = _seconds_since_exec()

from woof.da.member_wave import atomic_json, sha256, SCHEMA as RESULT_SCHEMA

COMMON_SCHEMA = "gpuwm-da.member-leg-common.v1"
REQUEST_SCHEMA = "gpuwm-da.member-leg-request.v1"


@functools.lru_cache(maxsize=32)
def _timing_card(visible):
    """Resolve telemetry identity only, without opening a CUDA context.

    Workers see one physical UUID. The read-only driver query is cached
    for the life of a card server. Unknown identities stay unknown.
    """
    if visible.isdecimal():
        return int(visible)
    from woof.local_gpu import no_local_gpu
    if no_local_gpu():
        return None
    if not visible.startswith("GPU-") or "," in visible:
        return None
    try:
        import subprocess
        rows = subprocess.run(
            [*__import__('woof.gpu_smi', fromlist=['command']).command(), "--id=" + visible,
             "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2, check=True).stdout
        for row in rows.splitlines():
            index, uuid = (part.strip() for part in row.split(",", 1))
            if uuid == visible and index.isdecimal():
                return int(index)
    except Exception:
        pass
    return None


class DAClock:
    """Always-on metadata timers. No device sync and no field reads.

    Sequential phases partition a caller's wall. Step-observer boundaries
    separately define forecast.integrate. Overlapping members, barriers,
    cycle envelopes and asynchronous writes must never be added together.
    """
    def __init__(self, out, *, cycle=None, member=None, card=None,
                 gpu_uuid=None, resolve_card=True):
        from woof.timeline.ledger import Ledger
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.card = card if card is not None else (
            _timing_card(visible) if resolve_card else None)
        self.attrs = {"cycle": cycle, "member": member,
                      "gpu_uuid": gpu_uuid or (visible if visible.startswith("GPU-") else None),
                      "basis": "monotonic_wall", "card_busy": False}
        self.ledger = Ledger(Path(out)/"timeline.jsonl", product="da", card=self.card)
        self.offset = time.time() - time.monotonic()
        self.active = None
        self.seconds = {}
        self.first_step = self.last_step = self.first_step_end = None

    def emit(self, stage, phase, at=None, **attrs):
        now = time.monotonic() if at is None else at
        return self.ledger.emit(stage, phase, t_mono=now,
                                t_unix=now+self.offset,
                                attrs={**self.attrs, **attrs})

    def interval(self, stage, start, end, **attrs):
        self.emit(stage, "begin", start, **attrs)
        self.emit(stage, "end", end, **attrs)

    def transition(self, stage=None, *, at=None):
        now = time.monotonic() if at is None else at
        if self.active is not None:
            previous, began = self.active
            self.emit(previous, "end", now)
            self.seconds[previous] = self.seconds.get(previous, 0.0) + now-began
        self.active = None if stage is None else (stage, now)
        if stage is not None:
            self.emit(stage, "begin", now)
        return now

    @contextlib.contextmanager
    def span(self, stage, **attrs):
        # Independent spans also cover background writes. The caller supplies
        # a span_id when concurrent operations share a stage and process.
        self.emit(stage, "begin", **attrs)
        try:
            yield
        finally:
            self.emit(stage, "end", **attrs)

    def step(self, *, step_wall_seconds, **metadata):
        end = time.monotonic()
        start = end-float(step_wall_seconds)
        if self.first_step is None:
            self.first_step, self.first_step_end = start, end
        self.last_step = end

    def finish_steps(self, construct_start, *, complete):
        if self.first_step is None:
            self.emit("post.member.step_coverage", "mark", complete=False,
                      reason="no measured model step")
            return
        attrs = {"basis": "execute_experiment.step_observer", "complete": complete}
        self.interval("forecast.construct", construct_start, self.first_step,
                      role="envelope", **attrs)
        self.interval("forecast.first_step", self.first_step, self.first_step_end, **attrs)
        self.interval("forecast.integrate", self.first_step, self.last_step,
                      card_busy=True, **attrs)

    def ready(self, *, at=None):
        now = time.monotonic() if at is None else at
        return {"t_mono": now, "t_unix": now+self.offset,
                "host": socket.gethostname(), "pid": os.getpid(),
                "card": self.card, "attrs": dict(self.attrs)}


def member_barriers(results, out, cycle, closed):
    """Relay each ready-to-cycle-barrier wait on the controller's clock.

    Packed readiness is sampled just before atomic result publication,
    after its array writes and hashes. It excludes controller adoption
    from member work. No scheduler callback or placement is changed.
    """
    waits = {}
    for member, result in results.items():
        ready = (result.record or {}).get("da_timing", {}).get("ready")
        clock = DAClock(out, cycle=cycle, member=member,
                        card=None if ready is None else ready["card"],
                        resolve_card=False)
        if ready is None or ready["host"] != socket.gethostname() or ready["t_mono"] > closed:
            clock.emit("post.member.barrier_coverage", "mark", complete=False,
                       reason="missing or incompatible member readiness clock")
            continue
        clock.attrs.update(ready["attrs"])
        clock.offset = ready["t_unix"]-ready["t_mono"]
        clock.interval("post.member.barrier", ready["t_mono"], closed,
                       span_id=f"{cycle}:{member}", worker_pid=ready["pid"],
                       basis="member_ready_to_complete_roster", card_busy=False)
        waits[str(member)] = closed-ready["t_mono"]
    return waits


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value) if field.init}
    if isinstance(value, argparse.Namespace):
        return plain(vars(value))
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f"member metadata cannot serialize {type(value).__name__}; use a typed array artifact")


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_file(record):
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"member input artifact changed or is missing: {path}")
    return path


def array_file(path, fields):
    path = Path(path)
    if not fields:
        return None
    if any(np.asarray(value).dtype.hasobject for value in fields.values()):
        raise ValueError("member payload refuses object arrays and pickle content")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **fields)
        stream.flush()
        os.fsync(stream.fileno())
    return file_record(path)


def arrays(record):
    if record is None:
        return {}
    path = verify_file(record)
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.array(saved[name], copy=True, order="C") for name in saved.files}


def observation_identity(document):
    """Bind the worker file to the exact document already read by its parent."""
    if document is None:
        return None
    fields = {}
    for name, value in sorted(document["variables"].items()):
        value = np.ascontiguousarray(np.asarray(value))
        if value.dtype.hasobject:
            raise ValueError("observation identity refuses object-array fields")
        fields[name] = {"dtype": value.dtype.str, "shape": list(value.shape),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    metadata = {key: plain(document.get(key)) for key in (
        "schema", "dims", "grid_identity_sha256", "grid_coordinate_sha256")}
    return {**metadata, "fields": fields}


def write_common(path, context):
    """Seal the prepared authority and unchanged perturbation configuration."""
    bindings = {"args": plain(context.args), "identity": plain(context.identity),
                "perturbation": plain(context.cfg_perturb), "hotstart": plain(context.hot_cfg)}
    inputs = context.inputs
    source_files = []
    for name in ("proof_path", "source_manifest_path", "experiment_config", "wps_namelist"):
        candidate = getattr(inputs, name, None)
        if candidate is not None:
            source_files.append(file_record(candidate))
    document = {"schema": COMMON_SCHEMA, **bindings,
                "source_files": source_files, "prepared_cache_identity": plain(dict(inputs.cache_identity))}
    atomic_json(path, document)
    return file_record(path)


#: The observation arrays a member worker reads: the reflectivity and its
#: mask, for the leg-end echo record, the hot start and the radar forcing.
WORKER_OBSERVATION_FIELDS = ("z_obs", "z_mask")


def observation_arrays(directory, document):
    """The worker's observation arrays, sealed once per leg under ``directory``.

    Each worker used to decode the whole radar file again (about 1 GB on
    the 9 km CONUS case, 14.6 of the 18.5 s every worker spent before its
    leg began) to read two arrays the controller already holds.  The
    controller writes those two arrays once; a worker checks the file's
    hash and each array against the controller document's identity, so
    the arrays it uses are still the controller's, byte for byte.  A
    second request of the same leg (the control is sealed before the
    members) finds the file already written and checks it is the same.
    """
    fields = {name: np.ascontiguousarray(np.asarray(document["variables"][name]))
              for name in WORKER_OBSERVATION_FIELDS
              if name in document.get("variables", {})}
    if not fields:
        return None
    path = Path(directory)/"observation-arrays.npz"
    if path.exists():
        record = file_record(path)
        held = arrays(record)
        if sorted(held) != sorted(fields) or any(
                held[k].dtype != v.dtype or held[k].shape != v.shape
                or held[k].tobytes() != v.tobytes() for k, v in fields.items()):
            raise ValueError("this leg's sealed observation arrays differ from its document")
        return record
    return array_file(path, fields)


def observation_seal(context, *, directory=None):
    """The observation half of a request, identical for every member of a leg.

    Hashing the leg's observation file, its grid and every array of the
    already-read document once per MEMBER repeated the same work 32 times
    on the controller between legs; :func:`write_requests` computes it once.
    """
    observation_files = []
    if context.obs_path is not None:
        if context.document is None:
            raise ValueError("controller observation must be read before sealing its worker request")
        observation_files.append(file_record(context.obs_path))
        if context.leg >= len(context.args.grid_wrfout):
            raise ValueError("member observation has no controller grid file")
        observation_files.append(file_record(context.args.grid_wrfout[context.leg]))
    sealed = None
    if context.obs_path is not None and directory is not None:
        sealed = observation_arrays(directory, context.document)
    return {"obs_path": None if context.obs_path is None else str(Path(context.obs_path).resolve()),
            "observation_files": observation_files,
            "observation_identity": observation_identity(context.document),
            "observation_arrays": sealed}


def job_directory(workdir, name):
    """A trajectory's job directory under one leg's ``workdir``."""
    return Path(workdir)/("control" if name == "control" else f"m{name:03d}")


def write_requests(workdir, common, contexts, names, *, threads=None,
                   pending_files=None, pool=None):
    """Seal every member request of one leg; the same files as one by one.

    The observation seal is computed once and shared, and the members'
    pending arrays are written and their restarts hashed concurrently.

    With ``pool`` (an executor the caller owns) the requests are sealed on
    it and a list of futures is returned in ``names`` order, so a member
    can start while later ones are still being sealed
    (:func:`woof.da.member_wave.run_wave` takes futures).
    ``pending_files(name)``, when given, returns the file that already
    holds that trajectory's pending increments (the recovery generation's
    own), waiting for it if it is still being written; the request then
    links that file instead of writing the same arrays a second time.
    """
    from concurrent.futures import ThreadPoolExecutor
    from woof.da.member_wave import HASH_THREADS

    names = list(names)
    if not names:
        return []
    seals = {}
    for name in names:
        key = (contexts[name].leg, None if contexts[name].obs_path is None
               else str(Path(contexts[name].obs_path).resolve()), id(contexts[name].document))
        if key not in seals:
            seals[key] = observation_seal(contexts[name], directory=workdir)
    def one(name):
        context = contexts[name]
        key = (context.leg, None if context.obs_path is None
               else str(Path(context.obs_path).resolve()), id(context.document))
        linked = None
        if pending_files is not None and context.pending:
            linked = pending_files(name)
        return write_request(job_directory(workdir, name), common, context, name,
                             observation=seals[key], pending_file=linked)
    if pool is not None:
        return [pool.submit(one, name) for name in names]
    width = HASH_THREADS if threads is None else int(threads)
    with ThreadPoolExecutor(max_workers=max(1, min(width, len(names)))) as owned:
        return list(owned.map(one, names))


def link_file(source, target):
    """``target`` as a second name of ``source`` (a copy across filesystems),
    and its file record.  The generation's pending file is immutable once
    written, so a link is a copy that cannot diverge."""
    import shutil

    target = Path(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return file_record(target)


def write_request(directory, common, context, name, *, observation=None,
                  pending_file=None):
    """Seal one immutable member input; preserve native array dtype and bits."""
    from woof.io.restart import tree_restart_members
    from tools.da_cycle_prepared import trajectory_fingerprint

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if pending_file is not None:
        pending = link_file(pending_file, directory/"pending.npz")
    else:
        pending = array_file(directory/"pending.npz", dict(context.pending or {}))
    setup_fields = dict(context.setup_arrays or {})
    if context.thb_host is not None:
        setup_fields["__thb_host"] = context.thb_host
    setup = array_file(directory/"setup.npz", setup_fields)
    source = [] if context.restart is None else [file_record(path) for _, path in
                                                 sorted(tree_restart_members(context.restart).items())]
    if observation is None:
        observation = observation_seal(context, directory=directory)
    observation_files = observation["observation_files"]
    document = {
        "schema": REQUEST_SCHEMA, "common": common, "member": name,
        "leg": context.leg, "absolute_leg_number": context.absolute_leg_number,
        "t_start": context.t_start, "t_end": context.t_end, "resumed": context.resumed,
        "restart": None if context.restart is None else str(Path(context.restart).resolve()),
        "restart_files": source, "pending": pending, "setup": setup,
        "observation_files": observation_files,
        "observation_identity": observation["observation_identity"],
        "observation_arrays": observation.get("observation_arrays"),
        "nest_child_dc": plain(context.nest_child_dc), "nested": context.nested,
        "nest_birth": context.nest_birth, "analysis_due": context.analysis_due,
        "obs_path": observation["obs_path"],
        "surface_enabled": context.surface_enabled,
        "output_root": str(directory.resolve()), "forecast_out": str(Path(context.out).resolve()),
    }
    # Radar tendency input is rebuilt from the same bound observation file.
    # Recording its presence prevents a worker from silently skipping forcing.
    document["radar_tten_active"] = context.tten_reflectivity is not None
    if (getattr(context.args, "spread_repair_vtsm", False)
            and context.analysis_due and name != "control"):
        document["spread_vts"] = {"socket": context.args.spread_vts_socket,
            "grid_identity_sha256": context.args.spread_vts_grid_identity}
    atomic_json(directory/"request.json", document)
    return {"member": name, "argv": [sys.executable, "-m", "woof.da.member_transport",
            "--request", str((directory/"request.json").resolve()),
            "--request-sha256", sha256(directory/"request.json"),
            "--result", str((directory/"result.json").resolve())],
            "result_path": str((directory/"result.json").resolve()),
            "request_hash": sha256(directory/"request.json"),
            "output_root": str(directory.resolve()),
            "t_start": context.t_start, "t_end": context.t_end,
            "ensemble_members": int(context.identity.members),
            "expected_domain_ids": [1, int(context.nest_child_dc.grid_id)] if context.nested else [1],
            "expected_trajectory_fingerprint": trajectory_fingerprint(context.identity, name)}


def read_context(request_path, *, expected_hash=None):
    """Rebind only the public prepared door; no request can choose code."""
    from woof.config import RunConfig
    from woof.da import radar_tten
    from woof.da.obs_radar import read_document
    from woof.experiment import DomainConfig
    from woof.obs.target_grid import TargetGrid
    from tools.da_member_leg import MemberLegContext, build_member_context
    from tools.da_ensemble_state import EnsembleIdentity

    request_path = Path(request_path)
    raw = request_path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("immutable member request SHA-256 changed before startup")
    request = json.loads(raw.decode("utf-8"))
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("member request has an incompatible complete-tree schema")
    allowed = {"schema", "common", "member", "leg", "absolute_leg_number", "t_start", "t_end",
               "resumed", "restart", "restart_files", "pending", "setup", "nest_child_dc", "nested",
               "nest_birth", "analysis_due", "obs_path", "surface_enabled", "output_root", "forecast_out", "radar_tten_active", "observation_files", "observation_identity"}
    # Requests sealed before the per-leg observation arrays existed have
    # no such key and read the radar file as before.
    if set(request) - {"observation_arrays", "spread_vts"} != allowed:
        raise ValueError("member request fields changed; callbacks and alternate loaders are not supported")
    common_path = verify_file(request["common"])
    common = json.loads(common_path.read_text(encoding="utf-8"))
    if common.get("schema") != COMMON_SCHEMA:
        raise ValueError("member common input has an incompatible public preflight schema")
    for record in common["source_files"]:
        verify_file(record)
    for record in request["restart_files"]:
        verify_file(record)
    for record in request["observation_files"]:
        verify_file(record)
    if request["restart"] is None:
        if request["restart_files"]:
            raise ValueError("fresh member request cannot substitute restart artifacts")
    else:
        from woof.io.restart import tree_restart_members
        native_paths = {str(path.resolve()) for path in tree_restart_members(request["restart"]).values()}
        sealed_paths = [str(Path(record["path"]).resolve()) for record in request["restart_files"]]
        if set(sealed_paths) != native_paths or len(sealed_paths) != len(native_paths):
            raise ValueError("member restart root differs from its sealed complete tree inventory")
    arguments = dict(common["args"])
    for name in ("prepared_root", "authority_dir"):
        if arguments.get(name) is not None:
            arguments[name] = Path(arguments[name])
    bindings = build_member_context(argparse.Namespace(**arguments))
    if plain(bindings.cfg_perturb) != common["perturbation"] or plain(bindings.hot_cfg) != common["hotstart"]:
        raise ValueError("worker perturbation or hot-start configuration differs from the controller")
    if plain(dict(bindings.inputs.cache_identity)) != common["prepared_cache_identity"]:
        raise ValueError("worker prepared-source identity differs from the controller")
    cfg = bindings.inputs.experiment.root.run
    rebound_identity = EnsembleIdentity(
        members=int(bindings.args.members), nx=int(cfg.nx), ny=int(cfg.ny), nz=int(cfg.nz),
        dt_s=float(cfg.dt), mp_physics=int(cfg.mp_physics), physics_profile=str(bindings.args.physics_profile),
        prepared_content_sha256=str(bindings.args.prepared_content_sha256))
    if plain(rebound_identity) != common["identity"]:
        raise ValueError("worker ensemble identity differs from the public prepared binding")
    child = request["nest_child_dc"]
    if child is not None:
        child = dict(child)
        child["run"] = RunConfig(**child["run"])
        child = DomainConfig(**child)
    identity = EnsembleIdentity(**common["identity"])
    document = None
    if request["obs_path"] is not None:
        if request["leg"] >= len(bindings.args.obs):
            raise ValueError("worker observation file is outside the controller's observed legs")
        expected_observation = Path(bindings.args.obs[request["leg"]]).resolve()
        expected_grid = Path(bindings.args.grid_wrfout[request["leg"]]).resolve()
        if Path(request["obs_path"]).resolve() != expected_observation:
            raise ValueError("member observation differs from the controller's bound observed leg")
        if {str(Path(record["path"]).resolve()) for record in request["observation_files"]} != {str(expected_observation),str(expected_grid)}:
            raise ValueError("member observation and grid hashes are not sealed to this leg")
        sealed = request.get("observation_arrays")
        if sealed is not None:
            # The controller's own arrays (observation_arrays), checked
            # against its document's identity array by array; the radar
            # file itself was verified above and is not decoded again.
            held = arrays(sealed)
            sealed_identity = request["observation_identity"] or {}
            fields = sealed_identity.get("fields", {})
            for name, value in held.items():
                value = np.ascontiguousarray(value)
                entry = fields.get(name)
                if entry is None or entry != {
                        "dtype": value.dtype.str, "shape": list(value.shape),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()}:
                    raise ValueError("worker observation arrays differ from the controller's already-read document")
            document = {key: sealed_identity.get(key) for key in (
                "schema", "dims", "grid_identity_sha256", "grid_coordinate_sha256")}
            document["variables"] = held
        else:
            grid = TargetGrid.from_wrfout(Path(bindings.args.grid_wrfout[request["leg"]]))
            document = read_document(Path(request["obs_path"]), expected_grid=grid)
            if observation_identity(document) != request["observation_identity"]:
                raise ValueError("worker observation arrays differ from the controller's already-read document")
            # A decoder cannot consume a changed file between initial verification
            # and reading. The arrays were built from these same sealed bytes.
            for record in request["observation_files"]:
                verify_file(record)
    elif request["observation_files"] or request["observation_identity"] is not None:
        raise ValueError("a free member leg cannot carry unrelated observed artifacts")
    tendency = None
    if request["radar_tten_active"]:
        if document is None or not request["analysis_due"] or not bindings.args.radar_tten:
            raise ValueError("radar forcing cannot be rebuilt from this member observation contract")
        tendency, _ = radar_tten.reflectivity_from_document(document)
    retained = request.get("spread_vts")
    wants_vts = bool(getattr(bindings.args, "spread_repair_vtsm", False)
                     and request["analysis_due"] and request["member"] != "control")
    if wants_vts != (retained is not None):
        raise ValueError("VTSM packed request must preserve its named retained-slot policy")
    if retained is not None:
        if (set(retained) != {"socket", "grid_identity_sha256"}
                or document is None or tendency is not None):
            raise ValueError("VTSM packed request needs a bound observation grid and no current-volume heating")
        grid = TargetGrid.from_wrfout(Path(bindings.args.grid_wrfout[request["leg"]]))
        if retained["grid_identity_sha256"] != grid.identity_sha256():
            raise ValueError("VTSM packed slot grid differs from this leg's native observation grid")
        if not Path(retained["socket"]).is_absolute():
            raise ValueError("VTSM packed retention socket needs an absolute cycle-owned path")
        bindings.args.spread_vts_socket = retained["socket"]
        bindings.args.spread_vts_grid_identity = retained["grid_identity_sha256"]
    setup = arrays(request["setup"])
    thb = setup.pop("__thb_host", None)
    context = MemberLegContext(
        args=bindings.args, inputs=bindings.inputs, identity=identity,
        cfg_perturb=bindings.cfg_perturb, hot_cfg=bindings.hot_cfg,
        leg=request["leg"], absolute_leg_number=request["absolute_leg_number"],
        t_start=request["t_start"], t_end=request["t_end"],
        stage_root=Path(request["output_root"])/"stage", out=Path(request["forecast_out"]),
        resumed=request["resumed"], restart=None if request["restart"] is None else Path(request["restart"]),
        pending=arrays(request["pending"]) or None, nest_child_dc=child, nested=request["nested"],
        nest_birth=request["nest_birth"], document=document, analysis_due=request["analysis_due"],
        tten_reflectivity=tendency, obs_path=None if request["obs_path"] is None else Path(request["obs_path"]),
        surface_enabled=request["surface_enabled"], setup_arrays=setup or None, thb_host=thb)
    context.validate(request["member"])
    return context, request


class RestartMirrorMismatch(ValueError):
    """The leg-end restart does not hold the bytes of the leg-end mirror."""


def verify_restart_matches_mirror(restart, snapshot) -> int:
    """Refuse a member result whose restart differs from its host mirror.

    Breakage it prevents: the packed controller analyses every member
    against its leg-end RESTART (a lazy view), no longer against the
    mirror the worker took after the leg's diagnostics (reflectivity H(x),
    surface capture, hot-start increments) ran on the live state.  Were any
    of those to write into a serialized state array, the analysis would
    read a background that is not the forecast's leg-end state, silently.
    The synthetic member-leg proof checks this only on a WSM6 fixture
    without hot start; this check runs on every production member leg,
    before the mirror is dropped.  Returns the bytes compared.
    """
    if not snapshot:
        return 0
    from woof.da.radar_assimilation import CheckpointStateView

    compared = 0
    view = CheckpointStateView(restart)
    try:
        for key, value in snapshot.items():
            value = np.ascontiguousarray(value)
            if key not in view:
                raise RestartMirrorMismatch(
                    f"leg-end restart {Path(restart).name} has no state/{key},"
                    " which the leg-end mirror holds")
            stored = np.ascontiguousarray(view[key])
            if (stored.dtype != value.dtype or stored.shape != value.shape
                    or memoryview(stored).cast("B")
                    != memoryview(value).cast("B")):
                raise RestartMirrorMismatch(
                    f"leg-end restart {Path(restart).name} state/{key} differs"
                    " from the leg-end mirror; the analysis would read a"
                    " background that is not the forecast's end state")
            compared += int(value.nbytes)
    finally:
        view.close()
    return compared


def write_result(path, context, name, result, request_hash, *, timing=None):
    """Publish a complete restart/array inventory only after the worker ends."""
    from woof.io.restart import tree_restart_members, read_restart_header

    path = Path(path)
    verify_restart_matches_mirror(result.restart, result.snapshot)
    if (getattr(context.args, "spread_repair_vtsm", False)
            and context.analysis_due and name != "control"):
        receipt = result.record.get("spread_vts", {})
        slots = receipt.get("slots", ())
        if (len(slots) != 3 or {slot["metadata"]["offset_seconds"] for slot in slots} != {-900, 0, 900}
                or any(slot["metadata"]["member"] != int(name) for slot in slots)
                or receipt.get("central_restore", {}).get("central_restored_seconds") != context.t_end):
            raise ValueError("VTSM member must publish all actual shifted slots and restore its central complete tree before result commit")
    # No whole-state host mirror crosses to the controller: the leg-end
    # restart published beside this result carries every state array, and
    # the analysis reads the few it needs from it on demand.  Writing,
    # hashing and reloading a second copy of the member was the largest
    # single host transfer of a packed cycle.  Only observation-space
    # products (H_Z, surface H(x)), the hot-start increments and the small
    # setup arrays travel here.
    data = {}
    if result.H_Z is not None:
        data["H_Z"] = result.H_Z
    data.update({"surface/"+key: value for key, value in (result.H_surface or {}).items()})
    data.update({"hot/"+key: value for key, value in (result.hot_pending or {}).items()})
    data.update({"setup/"+key: value for key, value in result.setup_arrays.items()})
    if result.thb_host is not None:
        data["thb_host"] = result.thb_host
    payload = array_file(path.parent/"arrays.npz", data)
    members = tree_restart_members(result.restart)
    clocks = {}
    for gid, member in members.items():
        header = read_restart_header(member)
        if header["elapsed_seconds"] != context.t_end:
            raise ValueError("member result restart does not reach its declared end clock")
        clocks[str(gid)] = {key: header.get(key) for key in (
            "elapsed_ticks", "tick_den", "step_count", "elapsed_seconds", "domain_start_ticks", "domain_lifecycle", "dtbc_fp32_bits")}
    files = [file_record(member) for _, member in sorted(members.items())]+[payload]
    document = {"schema": RESULT_SCHEMA, "complete": True, "member": name,
                "request_hash": request_hash, "t_start": context.t_start, "t_end": context.t_end,
                "restart": str(Path(result.restart).resolve()), "files": files,
                "arrays": payload, "output_root": str(path.parent.resolve()), "clocks": clocks,
                "nest_birth": result.nest_birth, "record": plain(result.record),
                # The controller owns both the durable generation and stage
                # distinction, so this names the restored source. Its stage
                # owner consumes only its own inputs after the full barrier.
                "consume_restart": None if context.restart is None else str(Path(context.restart).resolve()),
                "pending_consumed": result.pending_consumed}
    if timing is not None:
        # Where the worker's wall went outside the leg itself: start-up,
        # rebinding the request, and this publication.
        timing = dict(timing, publish_seconds=round(
            time.monotonic() - timing.pop("_publish_started"), 3))
        document["worker_seconds"] = timing
        # Also in the leg record the cycle report keeps, where a reader of
        # the run finds it (the result file is cleared with the stage).
        document["record"] = dict(document["record"] or {},
                                  worker_seconds=timing)
    clock = DAClock(context.out, cycle=context.absolute_leg_number, member=name)
    document["record"] = dict(document["record"] or {})
    document["record"]["da_timing"] = dict(
        document["record"].get("da_timing", {}), ready=clock.ready())
    result.record["da_timing"] = document["record"]["da_timing"]
    atomic_json(path, document)
    return document


def load_result(document):
    from tools.da_member_leg import MemberLegResult

    values = arrays(document["arrays"])
    subsets = lambda prefix: {key[len(prefix):]: value for key, value in values.items() if key.startswith(prefix)}
    return MemberLegResult(
        restart=Path(document["restart"]), snapshot=subsets("snapshot/") or None,
        H_Z=values.get("H_Z"), H_surface=subsets("surface/") or None,
        hot_pending=subsets("hot/") or None, setup_arrays=subsets("setup/"),
        thb_host=values.get("thb_host"), nest_birth=document["nest_birth"], record=document["record"],
        consume_restart=None if document["consume_restart"] is None else Path(document["consume_restart"]),
        pending_consumed=document["pending_consumed"])


def run_job(request_path, request_sha256, result_path, *, served_before=0,
            prefetched=None):
    """One member leg from its sealed request to its published result.

    The whole job of a worker process, and of each job a card server
    (:func:`serve`) runs; ``served_before`` counts the jobs the same
    process ran earlier (0 for a fresh worker).  ``prefetched`` is a card
    server's :class:`_Prefetch` of this job: its request already read and
    proved while the previous job stepped.  A prefetch for another request
    or one that failed is ignored and the request is read here, so any
    refusal is this job's own.
    """
    from tools.da_member_leg import run_member_leg

    request_path, result_path = Path(request_path), Path(result_path)
    if not re.fullmatch(r"[0-9a-f]{64}", str(request_sha256)):
        raise ValueError("--request-sha256 must be the immutable controller request digest")
    t_main = time.monotonic()
    bound = None if prefetched is None else prefetched.result_for(
        request_path, request_sha256)
    if bound is None:
        context, request = read_context(request_path, expected_hash=request_sha256)
    else:
        context, request = bound
    if result_path.resolve().parent != Path(request["output_root"]).resolve():
        raise ValueError("worker result must remain inside its immutable job directory")
    t_context = time.monotonic()
    clock = DAClock(context.out, cycle=context.absolute_leg_number,
                    member=request["member"])
    clock.interval("worker.claim", t_main, t_context,
                   prefetched=bound is not None)
    result = run_member_leg(context, request["member"])
    t_leg = time.monotonic()
    clock.transition("forecast.output.publish", at=t_leg)
    try:
        if sha256(request_path) != request_sha256:
            raise ValueError("immutable member request changed during the forecast leg")
        write_result(result_path, context, request["member"], result, request_sha256,
                 timing={"exec_to_module_import_seconds": (
                             _SINCE_EXEC_AT_IMPORT if not served_before else None),
                         "process_start_to_main_seconds": (
                             round(t_main - _PROCESS_STARTED, 3)
                             if not served_before else None),
                         "jobs_served_before": int(served_before),
                         "read_context_seconds": round(t_context - t_main, 3),
                         "read_context_prefetched": bound is not None,
                         "member_leg_seconds": round(t_leg - t_context, 3),
                         "_publish_started": t_leg})
    finally:
        clock.transition()
    return 0


#: Arrays a card server has read and proved against their prepared-cache
#: manifest rows, by (directory, key, digest).  Every member of a run reads
#: the same prepared cache: one read and hash per process, a copy per use.
_MANIFEST_ARRAYS: dict = {}


def _remember_manifest_arrays():
    """Keep each proved prepared-cache array for the rest of the process.

    :func:`woof.ingest.prepared_cache.read_manifest_array` loads one
    ``.npy`` and checks its shape, dtype, size and SHA-256 against the
    manifest; a member re-read and re-hashed the whole 4 GB cache of the
    9 km CONUS case at every leg.  In a card server the first read is the
    proof and later reads return a copy of the proved array (a caller that
    writes into it cannot change the next member's), so the bytes are the
    ones a fresh read gives.
    """
    from woof.ingest import prepared_cache

    original = prepared_cache.read_manifest_array
    if getattr(original, "remembers", False):
        return

    def read_manifest_array(directory, key, spec):
        try:
            digest = spec["sha256"]
        except (KeyError, TypeError):
            return original(directory, key, spec)
        cache_key = (str(Path(directory).resolve()), str(key), str(digest))
        held = _MANIFEST_ARRAYS.get(cache_key)
        if held is None:
            held = original(directory, key, spec)
            _MANIFEST_ARRAYS[cache_key] = held
        return held.copy()

    read_manifest_array.remembers = True
    prepared_cache.read_manifest_array = read_manifest_array


def _release_device_memory():
    import gc

    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


SERVER_JOB_SCHEMA = "gpuwm-da.card-server-job.v1"


class _Prefetch:
    """The next job of a card server, read and proved on a thread.

    Breakage it removes: a member job's host start -- reading and hashing
    its sealed request, the common inputs and its restart set, and the
    prepared-authority preflight -- ran with the card idle between two
    members' stepping (a rented box fullN3, 2026-10-06: 12-17 s of each 40-48 s
    member leg on the host).  The controller queues the next job beside
    the running one (woof.da.member_wave.SERVER_QUEUE_DEPTH); this reads
    it while the running one steps, and also reads its restart files once
    so their restore finds them in the page cache.  Nothing here touches
    the card.
    """

    def __init__(self, queue, index, cancelled):
        import threading

        self.queue, self.index, self.cancelled = Path(queue), int(index), cancelled
        self.done = threading.Event()
        self.value = None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"card-server-prefetch-{index}")
        self.thread.start()

    def _run(self):
        try:
            job_path = self.queue / f"job-{self.index:05d}.json"
            while not job_path.exists():
                if self.cancelled.is_set() or (self.queue / "stop").exists():
                    return
                time.sleep(0.05)
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("schema") != SERVER_JOB_SCHEMA:
                return
            claimed = time.monotonic()
            context, request = read_context(
                Path(job["request"]), expected_hash=job["request_sha256"])
            for record in request.get("restart_files") or ():
                with open(record["path"], "rb") as stream:
                    while stream.read(64 << 20):
                        pass
            completed = time.monotonic()
            clock = DAClock(getattr(context, "out", self.queue),
                            cycle=getattr(context, "absolute_leg_number", None),
                            member=request.get("member"))
            clock.interval("prep.member.prefetch", claimed, completed,
                           span_id=f"prefetch:{self.index}", asynchronous=True)
            self.value = (str(Path(job["request"]).resolve()),
                          job["request_sha256"], context, request)
        except Exception:
            # The job reads its request itself and raises its own refusal.
            self.value = None
        finally:
            self.done.set()

    def result_for(self, request_path, request_sha256):
        """``(context, request)`` when this prefetch read exactly that job."""
        self.done.wait()
        if self.value is None:
            return None
        path, digest, context, request = self.value
        if path != str(Path(request_path).resolve()) or digest != request_sha256:
            return None
        return context, request

    def cancel(self):
        self.cancelled.set()
        self.done.wait()


def serve(queue):
    """A card server: run the member jobs written into ``queue``, in order.

    The controller writes ``job-NNNNN.json`` ({schema, request,
    request_sha256, result}) atomically; this process runs each as
    :func:`run_job` and answers with ``job-NNNNN.done`` ({rc, error}).
    ``stop`` ends it.  Between jobs it returns every device block it held,
    so the card is the next job's (or the analysis's) as after a worker
    exits.  One process per card for the whole run keeps the interpreter,
    the CUDA context, the compiled kernels and the proved prepared-cache
    arrays; a member leg in it is the serial route's complete-tree leg,
    which runs every member in one process.
    """
    import threading

    queue = Path(queue)
    clock = DAClock(queue)
    clock.emit("boot.member_server", "begin", role="envelope")
    _remember_manifest_arrays()
    clock.emit("boot.member_server", "end", role="envelope")
    clock.transition("linger.member_server")
    served = 0
    index = 0
    ahead = None
    while True:
        job_path = queue / f"job-{index:05d}.json"
        if (queue / "stop").exists() and not job_path.exists():
            if ahead is not None:
                ahead.cancel()
            clock.transition()
            clock.emit("post.member_server.stop", "mark", served=served)
            return 0
        if not job_path.exists():
            time.sleep(0.05)
            continue
        # This job's prefetch (started while the previous job ran), and the
        # next job's, started now so it reads while this one steps.
        clock.transition()
        prefetched = ahead if ahead is not None and ahead.index == index else None
        ahead = _Prefetch(queue, index + 1, threading.Event())
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("schema") != SERVER_JOB_SCHEMA:
            raise ValueError("card server job has an unknown schema")
        done = {"schema": SERVER_JOB_SCHEMA, "rc": 0, "error": None}
        # The job's own log, as a worker process writes it: stdout and
        # stderr point at it for the job and back at the server's after.
        saved = None
        if job.get("log"):
            sys.stdout.flush(); sys.stderr.flush()
            saved = (os.dup(1), os.dup(2))
            handle = os.open(job["log"], os.O_WRONLY | os.O_APPEND)
            os.dup2(handle, 1); os.dup2(handle, 2); os.close(handle)
        try:
            run_job(job["request"], job["request_sha256"], job["result"],
                    served_before=served, prefetched=prefetched)
        except BaseException as error:   # reported; the controller refuses
            import traceback
            done = {"schema": SERVER_JOB_SCHEMA, "rc": 1,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-4000:]}
            print(done["traceback"], flush=True)
        finally:
            with clock.span("post.member_server.release", span_id=str(index)):
                _release_device_memory()
            if saved is not None:
                sys.stdout.flush(); sys.stderr.flush()
                os.dup2(saved[0], 1); os.dup2(saved[1], 2)
                os.close(saved[0]); os.close(saved[1])
        atomic_json(queue / f"job-{index:05d}.done", done)
        served += 1
        index += 1
        if done["rc"] != 0:
            # The state of a process whose leg raised is not trusted with
            # another member: the server ends and the controller refuses.
            ahead.cancel()
            return 1
        clock.transition("linger.member_server")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--serve", type=Path,
                        help="run as a card server over this queue directory")
    args = parser.parse_args()
    if args.serve is not None:
        return serve(args.serve)
    if args.request is None or args.request_sha256 is None or args.result is None:
        parser.error("--request, --request-sha256 and --result are required")
    try:
        return run_job(args.request, args.request_sha256, args.result)
    except ValueError as error:
        if "must" in str(error) and ("digest" in str(error) or "inside" in str(error)):
            parser.error(str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
