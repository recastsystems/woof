"""Linux-side, detached remote job service over one bounded stdin/stdout RPC.

No listener, password store, scheduler, or arbitrary-command RPC is installed.
The same interpreter selected by the client owns the durable worker and CLI.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import tomllib
import traceback

from woof.remote_artifacts import _file_sha, _sha

SCHEMA = "gpuwm.remote.result.v1"
TOKEN_ENV = "ARWEN_REMOTE_JOB_TOKEN"
MAX_BYTES = 128 * 1024
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
TERMINAL = {"completed", "failed", "stopped"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path, maximum=MAX_BYTES):
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"{path.name} exceeds its {maximum}-byte limit")
    return payload


def _json(path):
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write(path, value):
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _absolute(value, name):
    if (not isinstance(value, str) or not value or not Path(value).is_absolute()
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"{name} must be an absolute Linux path")
    return Path(value).resolve()


def _workspace(request):
    path = _absolute(request.get("workspace"), "workspace")
    if not path.is_dir():
        raise ValueError("workspace must already exist on the remote Linux node")
    return path


def _store(workspace, *, create=False):
    path = workspace / ".arwen-jobs"
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if path.is_symlink() or not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError(".arwen-jobs must be a private, owned directory (mode 700), without a symlink")
    return path


def _directory(workspace, identifier):
    if not isinstance(identifier, str) or not JOB_ID.fullmatch(identifier):
        raise ValueError("invalid remote job ID")
    directory = _store(workspace) / identifier
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"remote job {identifier} does not exist in this workspace")
    if directory.stat().st_uid != os.getuid():
        raise ValueError("remote job directory is not owned by this account")
    return directory


def runtime():
    import woof
    return {"version": woof.__version__, "python": sys.executable,
            "python_version": sys.version.split()[0], "module_path": str(Path(woof.__file__).resolve()),
            "platform": sys.platform, "prefix": sys.prefix}


def _process(pid):
    """Identity includes boot, start ticks, argv and UID, never just a PID."""
    try:
        root = Path("/proc") / str(int(pid))
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19], "pgid": int(fields[2]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "argv_sha256": _sha((root / "cmdline").read_bytes()),
                "uid": root.stat().st_uid}
    except (OSError, ValueError, IndexError):
        return None


def _has_token(pid, token):
    try:
        return (TOKEN_ENV + "=" + token).encode() in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\x00")
    except OSError:
        return False


def _owned_processes(token, *, owner_pid=None):
    # The worker is a Linux child subreaper: an orphan stays in its tree even
    # if a native child clears its environment or creates another session.
    # RPC callers use the token only unless they supply the verified owner.
    owner_pid = os.getpid() if owner_pid is None else owner_pid
    identities, parents, owned = {}, {}, set()
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and int(path.name) != owner_pid:
            identity = _process(int(path.name))
            if identity and identity["uid"] == os.getuid():
                pid = identity["pid"]
                identities[pid] = identity
                try:
                    stat = (path / "stat").read_text()
                    parents[pid] = int(stat[stat.rfind(")") + 2:].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                if _has_token(pid, token):
                    owned.add(pid)
    pending = {owner_pid, *owned}
    while pending:
        descendants = {pid for pid, parent in parents.items() if parent in pending} - owned
        owned.update(descendants)
        pending = descendants
    return [identities[pid] for pid in sorted(owned)]


def _signal_owned(identity, token, signum):
    # A pidfd pins the process between identity verification and the signal;
    # a PID reused during that interval can never receive this job's signal.
    try:
        handle = os.pidfd_open(identity["pid"], 0)
    except ProcessLookupError:
        return False
    try:
        if _process(identity["pid"]) != identity or identity not in _owned_processes(token):
            return False
        try:
            signal.pidfd_send_signal(handle, signum, None, 0)
            return True
        except ProcessLookupError:
            return False
    finally:
        os.close(handle)


def capabilities():
    """What this node serves, stated once and carried on every reply a client keeps.

    A client that writes this record for itself goes stale silently; a client
    that reads it off the node's own reply cannot.
    """
    return {"durable_jobs": True, "existing_remote_inputs": True,
            "resume": "manifest-valid route checkpoints",
            "host_key_verification": "OpenSSH strict known_hosts",
            "process_handles": _ownership_provider()["signal"],
            "stage_plan_v1": True, "review_plan_v1": True, "start_plan_v1": True,
            "artifact_sync_v1": True, "artifact_index_v1": True,
            "artifact_sequence_v1": True, "input_stream_v1": True,
            "processed_frame_v2": True, "processed_member_stream_v2": True,
            # Protocol facts a 2.7.5 client relies on at every slow door: the
            # node writes keepalives while it works (so the client's deadline
            # measures silence) and a launch names the attempt it belongs to
            # (so a retry reconciles with the job already created).
            "keepalive_v1": True, "launch_attempt_v1": True}


def _reap_owned(token, *, child=None):
    """The token-bounded escalation over exactly this job's owned tree.

    The wrapper runs it as it exits; the stop door runs the same ladder when a
    wrapper exited without proving the tree was gone. One function, both doors.
    """
    for signum, seconds in ((signal.SIGINT, 3), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            owned = _owned_processes(token)
            if not owned:
                return []
            for identity in owned:
                _signal_owned(identity, token, signum)
            if child is not None:
                child.poll()
            time.sleep(.1)
        if not _owned_processes(token):
            return []
    return _owned_processes(token)


def _record(directory):
    record = _json(directory / "job.json")
    if (record.get("schema") != "gpuwm.remote.job.v1" or record.get("id") != directory.name
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("token", ""))):
        raise ValueError("remote job ownership record is invalid")
    return record


#: The entry documents a remote job may launch, and the options each door
#: carries. A door is a row here; it is never a literal argv at a call site,
#: so the review and the launch cannot compose two different commands.
ENTRY_DOORS = {
    "go": {"module": "woof.cli", "command": "go",
           "flags": ("--outdir", "--no-memory-gate", "--geog-root", "--products", "--section",
                     "--prepared-root", "--wps-namelist", "--restart")},
    "run-plan": {"module": "woof.cli", "command": "run-plan", "flags": ()},
}


#: How remote job ownership is established, per platform. The Linux row supplies
#: the four things this ownership model needs: process identity, token
#: membership, descendant discovery and the child-reaper guarantee. Adding a
#: platform is a row plus its provider, never a branch at each call site.
OWNERSHIP_PROVIDERS = {
    "linux": {"identity": "/proc process identity", "membership": "/proc environ token",
              "descendants": "/proc parent walk under PR_SET_CHILD_SUBREAPER",
              "signal": "Linux pidfd"},
}


def _ownership_provider():
    """This node's ownership provider, or a refusal naming what is missing."""
    provider = OWNERSHIP_PROVIDERS.get(sys.platform)
    if provider is None:
        registered = ", ".join(sorted(OWNERSHIP_PROVIDERS)) or "none"
        raise ValueError(
            f"This node reports platform '{sys.platform}'; remote job ownership is verified through "
            "/proc, so a restarted worker could adopt a stranger's process as your job. Run the "
            f"worker on a node whose platform has a registered ownership provider ({registered}), "
            "or register an ownership provider for this platform.")
    return provider


def compose_argv(entry):
    """Build one door's command from its declared row, never from a literal."""
    door = ENTRY_DOORS.get(entry.get("door"))
    if door is None:
        raise ValueError(f"Unknown remote entry door '{entry.get('door')}'. The registered doors are "
                         + ", ".join(sorted(ENTRY_DOORS)) + "; register this door before launching it.")
    flags = []
    for name, value in entry.get("flags", ()):
        if name not in door["flags"]:
            raise ValueError(f"The remote '{entry['door']}' door carries no {name}. It carries "
                             + (", ".join(door["flags"]) or "no options at all")
                             + "; deliver this setting through a door that carries it.")
        # A signed coordinate remains a value, not another argparse option.
        flags += ([name] if value is None else [name + "=" + str(value)]
                  if str(value).startswith("-") else [name, str(value)])
    return [sys.executable, "-I", "-u", "-m", door["module"], door["command"],
            str(entry["document"]), *flags]


ROUTES = {"staged_plan": "a map plan staged from this setup and reviewed before launch",
          "node_config": "a configuration file that already exists on the selected node"}


def route_statement(route, detail):
    """Name the route a review planned and the fact that chose it, once."""
    if route not in ROUTES:
        raise ValueError(f"Unknown remote run route '{route}'. The registered routes are "
                         + ", ".join(sorted(ROUTES)) + "; register this route before reviewing it.")
    return {"route": route, "route_reason": f"{ROUTES[route]}: {detail}"}


#: What a receipt written before the ownership token was recorded on it says
#: about itself: it lives in this job's own folder, which is the only claim a
#: worker of that age could make.
LEGACY_RECEIPT_BASIS = "receipt written before the ownership token was recorded"


def _receipt_token_disagrees(receipt, record):
    """A receipt is another job's only when it names a token and that token differs.

    A worker before 2.7.5 wrote these receipts without a token at all, and a
    node updated under such a job would otherwise refuse that job's status
    forever, until an operator deleted the file by hand. A receipt with no
    token is reported on the basis of the folder it sits in; a token that is
    present and different is still refused.
    """
    return "token" in receipt and receipt["token"] != record["token"]


def _legacy_receipt_basis(receipt):
    return {} if "token" in receipt else {"basis": LEGACY_RECEIPT_BASIS}


def _status(directory):
    record = _record(directory)
    job = {key: record[key] for key in ("id", "created_at", "config", "outdir", "runtime", "action")}
    # Where this job's run actually writes, which is a stamped folder under the
    # directory the reader named unless that directory was itself a run folder.
    job["run_root"] = str(_parent_run_root(record))
    for key in ("route", "route_reason"):
        if isinstance(record.get(key), str):
            job[key] = record[key]
    source = record.get("source")
    # Only a staged review records this desktop source identity, and the shape
    # test below is the whole of it; the action word decided nothing here.
    if (isinstance(source, dict)
            and isinstance(source.get("config_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", source["config_sha256"])
            and isinstance(source.get("config_path"), str) and 0 < len(source["config_path"]) <= 8192
            and (Path(source["config_path"]).is_absolute() or PureWindowsPath(source["config_path"]).is_absolute())
            and not any(ord(character) < 32 for character in source["config_path"])):
        job["source_config_sha256"] = source["config_sha256"]
        job["source_config_path"] = source["config_path"]
    job["parent_job"] = record.get("parent_job")
    job["viewer_capabilities"] = {key: capabilities()[key] for key in (
        "processed_frame_v2", "processed_member_stream_v2",
        "artifact_sync_v1", "artifact_index_v1", "artifact_sequence_v1")}
    job["state"] = "starting"
    if (directory / "started.json").exists():
        started = _json(directory / "started.json")
        if started.get("token") != record["token"]:
            raise ValueError("remote worker ownership token disagrees with job record")
        job["started_at"] = started["started_at"]
        identity = started["identity"]
        actual = _process(identity["pid"])
        job["state"] = ("running" if actual == identity and _has_token(identity["pid"], record["token"])
                        else "ownership_mismatch" if actual is not None else "lost")
    if (directory / "result.json").exists():
        ended = _json(directory / "result.json")
        if ended.get("token") != record["token"]:
            raise ValueError("remote result ownership token disagrees with job record")
        job.update({key: ended[key] for key in ("state", "exit_code", "ended_at")})
        if ended.get("error"):
            job["error"] = ended["error"]
    from woof.remote_artifacts import plan_binding
    if plan_binding(record) is not None:
        try:
            from woof.remote_artifacts import native_progress
            job.update(native_progress(record, job))
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Durable ownership/state remains authoritative during an atomic
            # native update or unavailable progress receipt. Never infer exit.
            # The diagnosis still travels, so a reader is not left to guess why
            # a running job looks like it is still preparing.
            job["native_progress_error"] = {"message": str(error)[:1000],
                                            "receipt": str(_parent_run_root(record) / "run-manifest.json"),
                                            "class": type(error).__name__}
        if directory.parent.name == ".arwen-jobs":
            compact = directory.parent.parent / ".arwen-processed-v2" / record["id"]
            for key, path, schema in (
                ("background_maps", compact / "preparation.json", "arwen.native-store-preparation.v1"),
                ("native_plots", compact / "native-plots" / "status.json", "arwen.native-plot-progress.v1"),
            ):
                try:
                    if path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024:
                        value = _json(path)
                        if value.get("schema") == schema and value.get("job_id") == record["id"]:
                            job[key] = value
                            warning = value.get("render_warning")
                            if (key == "native_plots" and isinstance(warning, str) and warning
                                    and "render_warning" not in job):
                                # The gallery's renderer had no map files. The
                                # job-level field the run's own warning fills,
                                # so the terminal workspace shows it either way.
                                job["render_warning"] = warning[:1600]
                except (OSError, ValueError) as error:
                    job[key + "_error"] = {"message": str(error)[:1000], "receipt": str(path),
                                           "class": type(error).__name__}
    start_error = directory / "preparation-start-error.json"
    if start_error.is_file() and not start_error.is_symlink():
        receipt = _json(start_error)
        if _receipt_token_disagrees(receipt, record):
            raise ValueError(
                "This job's watcher start receipt carries a different job's ownership token, so "
                "reporting it would blame this run for another job's failed watcher. Delete "
                f"{start_error} on the node, then read this job's status again.")
        for key in WATCHERS:
            entry = receipt.get(key)
            # A successful retry always wins: only report the failed start while
            # no live receipt from that watcher exists.
            if isinstance(entry, dict) and key not in job:
                job[key] = {"state": "start_failed", "error": entry.get("error"), "at": entry.get("at"),
                            **_legacy_receipt_basis(receipt)}
    cleanup = directory / "cleanup-error.json"
    if cleanup.is_file() and not cleanup.is_symlink():
        receipt = _json(cleanup)
        if _receipt_token_disagrees(receipt, record):
            raise ValueError(
                "This job's cleanup receipt carries a different job's ownership token, so the "
                "processes it lists cannot be signalled as this job's. Delete "
                f"{cleanup} on the node, then stop this job again.")
        surviving = receipt.get("surviving")
        job["state"] = "cleanup_failed"
        job.update(_legacy_receipt_basis(receipt))
        job["error"] = (str(receipt.get("error", ""))[:2000] + "; this job's cleanup could not prove its "
                        "owned processes were gone, so they may still hold this node's card and output "
                        "tree. Stop this job again to re-signal exactly those processes.")
        job["surviving"] = surviving if isinstance(surviving, list) else []
    return job


def _inputs(source, *, preserve_bound=False, extra_wps=None):
    from woof.hrrr_route_inputs import route_input_paths
    from woof.toml_document import emit_experiment_toml
    payload = _read(source)
    original = tomllib.loads(payload.decode("utf-8"))
    if "experiment" not in original or not isinstance(original.get("domain"), list):
        raise ValueError("remote start requires a WOOF-ArWen experiment TOML with [[domain]] tables")
    from woof.experiment import build_experiment_from_config_tables
    build_experiment_from_config_tables(original, source=str(source), base_dir=source.parent)
    raw = copy.deepcopy(original)
    if "case_data" in raw and not preserve_bound:
        from woof.case_data import resolved_case_data_paths
        raw["case_data"] = resolved_case_data_paths(raw["case_data"], base_dir=source.parent, source=str(source))
    if "static" in raw and not preserve_bound:
        from woof.static.highres_production import parse_static_table
        highres = parse_static_table(raw["static"], source=str(source), base_dir=source.parent)
        if highres is not None and "highres" in raw["static"]:
            raw["static"]["highres"]["cache_root"] = str(highres.cache_root.resolve())
    sources = {str(source): payload}
    snapshots = {}
    wps_hash = None
    for label, companion in route_input_paths(source).items():
        if not companion.exists():
            continue
        content = _read(companion)
        sources[str(companion)] = content
        if label == "wps_namelist":
            from woof.starter_template import _tiles_wps
            wps_hash = _sha(content)
            if not preserve_bound:
                content = _tiles_wps(companion, source.parent / ".remote-snapshot" / companion.name,
                                     text=content.decode("utf-8")).encode("utf-8")
        snapshots[companion.name] = content
    # A declared-input experiment may name a WPS authority or Vtable elsewhere.
    # Those small scientific companions are captured too; only the large data
    # and geography references continue to name the existing remote inputs.
    if "case_data" in raw and not preserve_bound:
        for label, name in (("wps_namelist", "declared-inputs/namelist.wps"),
                            ("vtable", "declared-inputs/Vtable")):
            companion = Path(raw["case_data"][label])
            content = _read(companion)
            sources[str(companion)] = content
            if label == "wps_namelist":
                from woof.starter_template import _tiles_wps
                wps_hash = _sha(content)
                content = _tiles_wps(companion, source.parent / ".remote-snapshot" / name,
                                     text=content.decode("utf-8")).encode("utf-8")
            snapshots[name] = content
            raw["case_data"][label] = name
    if extra_wps is not None:
        content = _read(extra_wps)
        sources[str(extra_wps)] = content
        snapshots["prepared-inputs/namelist.wps"] = content
        wps_hash = _sha(content)
    snapshots[source.name] = payload if preserve_bound else emit_experiment_toml(raw).encode("utf-8")
    hashes = {name: _sha(value) for name, value in sources.items()}
    binding = _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())
    return original, sources, snapshots, {"config_sha256": _sha(payload), "wps_sha256": wps_hash,
                                           "input_sha256": binding, "inputs": hashes}


def _resolve_checkpoint(old, spec):
    """The resolved checkpoint with its disclosures (a ResumeResolution).

    ``_checkpoint`` keeps the path-only shape for callers that want the
    file; the review reads the resolution so the notes the local resume
    command prints reach the operator here too.
    """
    from woof.resume import _check_set, discover_checkpoint_sets, resolve_resume_checkpoint
    from woof.io.restart import read_restart_header
    from woof.supervisor import validate_manifest_checkpoint
    root = _parent_run_root(old).resolve()
    # These are the output routes owned by Go/run-plan, under this job's unique
    # output root. Never glob another job or an arbitrary user's directory.
    directories = [root, root / "run", root / "chain" / "run"]
    if spec != "latest":
        checkpoint = _absolute(spec, "--from")
        if not checkpoint.is_relative_to(root):
            raise ValueError("--from must belong to the source job's recorded output directory")
        validate_manifest_checkpoint(checkpoint)
        candidate = next((item for item in discover_checkpoint_sets(checkpoint.parent)
                          if checkpoint in item.members.values()), None)
        if candidate is not None:
            _check_set(candidate, validate_manifest_checkpoint, read_restart_header)
        elif read_restart_header(checkpoint).get("domain_ids"):
            raise ValueError("tree checkpoint has no discoverable complete checkpoint set")
        # The same door the local resume uses for an explicit path, so the
        # review discloses exactly what that command would print.
        return resolve_resume_checkpoint(checkpoint.parent, checkpoint, config=old["snapshot_config"])
    candidates = []
    failures = []
    for directory in directories:
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("checkpoint directory escapes the source job's output tree")
        try:
            resolved = resolve_resume_checkpoint(directory, "latest", config=old["snapshot_config"])
            if not resolved.checkpoint.resolve().is_relative_to(root.resolve()):
                raise ValueError("checkpoint escapes the source job's output tree")
            candidates.append(resolved)
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    if candidates:
        return max(candidates, key=lambda item: (item.checkpoint_set.valid_time, item.checkpoint.stat().st_mtime_ns))
    raise ValueError("No manifest-valid checkpoint exists in this job's recorded output tree"
                     + "; " + "; ".join(failures)[:4000] + _restart_cadence_note(old["snapshot_config"]))


def _restart_cadence_note(config_path):
    """Say what the source job itself declared, so the cause is in the sentence."""
    import tomllib
    try:
        with Path(config_path).open("rb") as stream:
            declared = tomllib.loads(stream.read(512 * 1024).decode("utf-8"))["experiment"]["restart_interval_s"]
        interval = float(declared)
    except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return ""
    if not interval > 0:
        return f"; the source job declared restart_interval_s = {interval}, so it wrote no checkpoints"
    return f"; the source job declared restart_interval_s = {interval}"


def _parent_run_root(record):
    """The folder a source job's run actually wrote into.

    A job recorded before the run folder was recorded has only the directory
    it was given, and that job's run did write straight into it.
    """
    return Path(record.get("run_root") or record["outdir"])


def _checkpoint(old, spec):
    return _resolve_checkpoint(old, spec).checkpoint


def _checkpoint_binding(checkpoint):
    from woof.resume import discover_checkpoint_sets
    candidate = next((item for item in discover_checkpoint_sets(checkpoint.parent)
                      if checkpoint in item.members.values()), None)
    paths = [checkpoint] if candidate is None else candidate.members.values()
    hashes = {str(path): _file_sha(path) for path in paths}
    return {"checkpoint_sha256": hashes[str(checkpoint)], "checkpoint_inputs": hashes,
            "checkpoint_set_sha256": _sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())}


def _run_root(outdir):
    """Where this run's own tree lands under the output directory it was given.

    `--run-stamp off` is the documented workaround for the stamped layout, and
    this door used to pass it on every remote run, so every remote forecast
    wrote straight into the directory the reader named and a second run of the
    same configuration was refused against the first. The default layout is
    restored, exactly as the local front door lays it: the output directory is
    a case folder that collects runs, each run claims a stamped folder inside
    it, `latest-run.txt` beside them names the newest, and an output directory
    that is itself a run folder is honoured as it is. The folder is named here,
    once, and both the review and the launch carry that one name, so every
    reader of this job's artifacts reads the folder the run actually used
    rather than the directory above it.
    """
    from woof import run_stamp
    outdir = Path(outdir)
    if run_stamp.owning_run(outdir) is not None:
        return outdir
    launch = run_stamp.utcnow()
    for ordinal in range(1, run_stamp.MAX_ORDINAL + 1):
        # The collision spelling is predicted here the way the local door
        # predicts it, so two reviews in one second name two folders.
        candidate = outdir / run_stamp.format_stamp(launch=launch, ordinal=ordinal)
        if not candidate.exists():
            return candidate
    raise ValueError(f"{run_stamp.MAX_ORDINAL} run folders already exist under {outdir} for this launch "
                     "second, so a distinct one cannot be named; give this run its own output "
                     "directory, or start it apart in time.")


def _claim_output(outdir, run_root):
    """Claim this run's folder exclusively, and publish it as the newest run.

    The exclusive claim is the run folder, never the case folder above it: two
    launches into one case folder are two sibling runs, which is the layout
    this exists for. A run folder that already exists is a refusal that names
    it, because two runs in one tree publish receipts describing neither.
    """
    from woof import run_stamp
    outdir, run_root = Path(outdir), Path(run_root)
    if run_root == outdir:
        outdir.mkdir(mode=0o700)
        return
    outdir.mkdir(mode=0o700, exist_ok=True)
    try:
        run_root.mkdir(mode=0o700)
    except FileExistsError:
        raise ValueError(f"The run folder {run_root} was claimed by another launch in the same second, "
                         "so this run cannot own it. Review and start this run again; the next "
                         "second names a different folder.") from None
    run_stamp.record_latest(outdir, run_root)


def _prepared_binding(prepared, config, wps, outdir):
    from woof import bridges, stage_cli
    with bridges.inspection_only():
        bundle = stage_cli.resolve_bundle(prepared)
        # Same read-only receipt and digest relay as woof sim --print-command.
        # The actual forecast still validates every cache and authority byte.
        stage_cli.sim_command(bundle, experiment_config=config, wps_namelist=wps, outdir=outdir)
    return {"prepared_document": str(bundle["document"]), "prepared_sha256": _file_sha(bundle["document"]),
            "prepared_layout": bundle["layout"], "prepared_validation": "receipt and digest relay; full content preflight at launch"}


def _requested(request, old, key, saved_key=None):
    """Explicit restart overrides win; omitted values inherit the saved job."""
    return request.get(key, None if old is None else old.get(saved_key or key))


def _review(request, workspace):
    action = request["action"]
    old, plan_document = None, None
    if action == "resume":
        directory = _directory(workspace, request.get("job"))
        old = _record(directory)
        if _status(directory)["state"] not in TERMINAL:
            raise ValueError("resume requires a completed, failed, or stopped source job")
        source = Path(old["snapshot_config"])
        if old.get("snapshot_plan"):
            # A staged map plan resumes as a staged map plan. Rebuilding it as a
            # configuration run would silently drop every run option the plan
            # carries, the render selection included.
            plan_document = json.loads(_read(Path(old["snapshot_plan"])).decode("utf-8"))
    else:
        source = _absolute(request.get("config"), "config")
    if not source.is_file():
        raise ValueError("config must already exist on the remote node")
    outdir = _absolute(request.get("outdir"), "outdir")
    run_root = _run_root(outdir)
    if outdir.is_symlink() or (outdir.exists() and not outdir.is_dir()):
        raise ValueError(f"outdir {outdir} exists on the node and is not a directory, so no run folder can "
                         "be claimed inside it; name a directory, or one that does not exist yet.")
    if run_root == outdir and outdir.exists():
        # The reader named a run folder itself, and it is already someone's:
        # two runs in one tree publish receipts describing neither.
        raise ValueError(f"outdir names the run folder {outdir}, which already exists, so a second run "
                         "there would publish receipts describing neither run. Name a run folder "
                         "that does not exist yet, or name the directory above it and this run claims "
                         "a stamped folder beside the existing one.")
    if not outdir.exists() and not outdir.parent.is_dir():
        raise ValueError("outdir's parent must already exist on the remote node")
    if old and (run_root.is_relative_to(_parent_run_root(old)) or run_root == _parent_run_root(old)):
        raise ValueError("resume output must be outside the source job's own run folder "
                         f"{_parent_run_root(old)}; a sibling run folder beside it is fine")
    prepared = _requested(request, old, "prepared_root")
    wps = _requested(request, old, "wps_namelist", "snapshot_wps_namelist")
    prepared_basis = None
    if old and plan_document is None and prepared is None and "case_data" not in tomllib.loads(_read(source).decode("utf-8")):
        from woof.resume import select_prepared_root
        # Checkpointless routes retain their specific refusal before bundle discovery.
        resume_checkpoint = _checkpoint(old, request.get("from_checkpoint", "latest"))
        root, prepared_basis = select_prepared_root(_parent_run_root(old), resume_checkpoint)
        prepared = str(root)
        candidate = source.with_suffix(".namelist.wps")
        if wps is None:
            wps = str(candidate) if candidate.is_file() else None
    if prepared is not None:
        prepared = _absolute(prepared, "prepared_root")
    if wps is not None:
        if prepared is None:
            raise ValueError("wps_namelist requires prepared_root")
        wps = _absolute(wps, "wps_namelist")
        if not wps.is_file():
            raise ValueError("wps_namelist must already exist on the remote node")
    raw, sources, snapshots, binding = _inputs(
        source, preserve_bound=prepared is not None or plan_document is not None, extra_wps=wps)
    if prepared is not None:
        binding.update(_prepared_binding(prepared, source, wps, run_root))
    for name in ("config", "wps", "input", "prepared"):
        expected = request.get(f"expected_{name}_sha256")
        if expected is not None and expected != binding.get(f"{name}_sha256"):
            raise ValueError(f"{name} inputs changed since review; review again before starting")
    geog = _requested(request, old, "geog_root")
    if geog is not None:
        geog = str(_absolute(geog, "geog_root"))
        if not Path(geog).is_dir():
            raise ValueError("geog_root must already exist on the remote node")
    products = _requested(request, old, "products")
    section = _requested(request, old, "section")
    device = _requested(request, old, "device")
    if products is not None and (not isinstance(products, str) or len(products) > 16384 or "\x00" in products):
        raise ValueError("products must be a catalog selector string of at most 16384 characters")
    if section is not None and (not isinstance(section, str) or len(section) > 8192
                                or any(ord(char) < 32 for char in section)):
        raise ValueError("section must be a line or node JSON path of at most 8192 characters; "
                         "otherwise the renderer cannot read the cut line")
    cwd = source.parent if old is None else Path(old["cwd"])
    from woof.go_cli import admit_render_products, render_section_value
    section = render_section_value(section, base=cwd)
    checkpoint = None
    resume_notes = []
    if old:
        resolution = _resolve_checkpoint(old, request.get("from_checkpoint", "latest"))
        checkpoint = resolution.checkpoint
        resume_notes = list(resolution.notes)
        binding.update(_checkpoint_binding(checkpoint))
        for name in ("checkpoint", "checkpoint_set"):
            expected = request.get(f"expected_{name}_sha256")
            if expected is not None and expected != binding[f"{name}_sha256"]:
                raise ValueError(f"{name} changed since review; review again before restarting")
    plan_path = None
    if plan_document is not None:
        # A run option this resume named belongs in the plan the run loads, not
        # on a flag the run-plan door does not carry: relocating it here is what
        # keeps a named input from being validated and then silently dropped.
        resumed = _resumed_plan(plan_document, old, run_root, checkpoint, overrides={
            "prepared_root": None if prepared is None else str(prepared),
            "wps_namelist": None if wps is None else str(wps),
            "render_products": request.get("products"),
            "render_section": section if request.get("section") is not None else None,
            "device": device,
            "geog_root": request.get("geog_root")})
        snapshots["plan.json"] = _plan_bytes(resumed)
        plan_path = str(Path(old["snapshot_plan"]).parent / "plan.json")
        _validate_resumed_plan(resumed, snapshots[Path(source).name], Path(source).name)
        # The plan carries its own render selection; it is recorded so this
        # job's own watchers draw what the run draws, and never passed as a flag.
        products = resumed.get("run_options", {}).get("render_products")
        section = resumed.get("run_options", {}).get("render_section")
        entry = {"door": "run-plan", "document": plan_path, "flags": []}
    else:
        admit_render_products(products, section=section)
        # The remote review already carries sizing advice. Do not turn the same
        # estimate into a refusal again inside go; its input and device checks
        # and the runner's real allocation errors still apply.
        flags = [("--outdir", str(run_root)), ("--no-memory-gate", None)]
        if geog:
            flags.append(("--geog-root", geog))
        if products is not None:
            flags.append(("--products", products))
        if section is not None:
            flags.append(("--section", section))
        if prepared is not None:
            flags.append(("--prepared-root", str(prepared)))
        if wps is not None:
            flags.append(("--wps-namelist", str(wps)))
        if checkpoint is not None:
            flags.append(("--restart", str(checkpoint)))
        entry = {"door": "go", "document": str(source), "flags": flags}
    from woof.remote_plan import memory_decision
    # This door prices the card the way the staged door does: memory_decision
    # reads this node's own device probe, so an unmeasured card is priced
    # against the capacity that probe recorded for the card this run will use
    # and that basis is stated here too, rather than answered as nothing
    # recorded. A selector this node does not have is refused here, at review.
    memory = memory_decision(source, device=device)
    value = {"entry": entry, "argv": compose_argv(entry), "config": str(source), "outdir": str(outdir),
             "run_root": str(run_root),
             "cwd": str(cwd), "section": section,
             "geog_root": geog, "products": products, "checkpoint": None if checkpoint is None else str(checkpoint),
             "prepared_root": None if prepared is None else str(prepared), "wps_namelist": None if wps is None else str(wps),
             "prepared_basis": prepared_basis,
             "parent_job": None if old is None else old["id"], "runtime": runtime(), "memory": memory,
             "device": None if device is None else str(device),
             # What woof resume would print about this checkpoint: which memory
             # mode the source job's configuration resolves to and which road
             # wrote the file.  Disclosure only; the review never refuses on it.
             "resume_notes": resume_notes,
             **route_statement("staged_plan" if plan_document is not None else "node_config",
                               plan_path or str(source)), **binding}
    if plan_document is not None:
        value.update({"plan": plan_path, "plan_sha256": _sha(snapshots["plan.json"])})
    return value, sources, snapshots


def _plan_bytes(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _resumed_plan(document, old, outdir, checkpoint, overrides=None):
    """The parent's own plan, pointed at this resume's output and checkpoint."""
    resumed = copy.deepcopy(document)
    resumed["output_root"] = str(outdir)
    options = resumed.setdefault("run_options", {})
    from woof.runplan import ROUTES
    carried = ROUTES[resumed.get("route")].run_options if resumed.get("route") in ROUTES else None
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if carried is not None and key not in carried:
            raise ValueError(f"This resume names '{key}' and the plan it resumes takes the "
                             f"'{resumed.get('route')}' route, which carries run options "
                             f"{', '.join(sorted(carried))}, so the run would refuse the plan this "
                             "resume wrote. Drop that option, or resume a plan whose route carries it.")
        options[key] = value
    if checkpoint is not None:
        options["restart"] = str(checkpoint)
    if resumed.get("route") == "prepared" and options.get("prepared_root") is None and checkpoint is not None:
        from woof.resume import select_prepared_root
        root, _ = select_prepared_root(_parent_run_root(old), checkpoint)
        options["prepared_root"] = str(root)
    return resumed


def _validate_resumed_plan(resumed, config_bytes, config_name):
    """Refuse a bad rewrite at review, never after the resume has launched."""
    import tempfile
    from woof import runplan
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        (directory / "plan.json").write_bytes(_plan_bytes(resumed))
        (directory / config_name).write_bytes(config_bytes)
        declared = resumed.get("config", {})
        if isinstance(declared, dict) and declared.get("path"):
            target = directory / Path(str(declared["path"])).name
            if not target.exists():
                target.write_bytes(config_bytes)
        try:
            runplan.load_plan(directory / "plan.json")
        except Exception as error:  # noqa: BLE001 - one voice for a refused resume.
            raise ValueError("The resumed map plan is not valid: " + str(error)[:2000]) from error


def _launch(request, workspace, *, command_factory=None, worker_command=None):
    """Private test seams are callables; no request key can select a command."""
    existing = reconciled_job(request, workspace)
    if existing is not None:
        # Asked before the review, because the first attempt already claimed
        # this request's output directory and the review would refuse it.
        return existing
    review, sources, snapshots = _review(request, workspace)
    if request.get("dry_run", False):
        return {"dry_run": True, "review": review}
    return _launch_review(request, workspace, review, sources, snapshots,
                          command_factory=command_factory, worker_command=worker_command)


#: What a client calls its own launch attempt. A retry of one attempt is the
#: same attempt, and this is how the node knows that.
REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
#: The schema of a line that says only "this node is still working".
KEEPALIVE_SCHEMA = "gpuwm.remote.keepalive.v1"
KEEPALIVE_SECONDS = 5


def reconciled_job(request, workspace):
    """The job this exact launch request already created, if it already did.

    A launch that times out on the connection has already happened or has not,
    and a client that cannot tell used to be told to list jobs and work it out.
    The client names its attempt; a retry of that attempt is answered with the
    job the first one created rather than a second forecast on the same card.

    The attempt identity lives on the job record itself, inside the job folder
    the node already owns, so nothing new is written into the durable store's
    job-identity grammar and a listing cannot read a receipt as a job.
    """
    request_id = request.get("request_id")
    if request_id is None:
        return None
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise ValueError("A launch request identity must be 32 hexadecimal characters")
    store = _store(workspace)
    if not store.is_dir():
        return None
    for directory in sorted(store.iterdir(), reverse=True):
        if not JOB_ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
            continue
        try:
            record = _record(directory)
        except (OSError, ValueError):
            continue
        if record.get("request_id") == request_id:
            return {"job": _status(directory), "reconciled_request_id": request_id}
    return None


def _launch_review(request, workspace, review, sources, snapshots, *, command_factory=None, worker_command=None):
    """Shared durable owner for the fixed go and run-plan entry documents."""
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(8)
    directory = _store(workspace, create=True) / identifier
    directory.mkdir(mode=0o700)
    inputs = directory / "inputs"
    originals = directory / "original-inputs"
    snapshot_config = inputs / Path(review["config"]).name
    snapshot = inputs / Path(review.get("plan", review["config"])).name
    entry = {**review["entry"], "document": str(snapshot),
             "flags": [(name, str(inputs / "prepared-inputs" / "namelist.wps")
                        if name == "--wps-namelist" else value)
                       for name, value in review["entry"].get("flags", ())]}
    argv = compose_argv(entry)
    if command_factory is not None:
        argv = command_factory(review, snapshot)
    record = {"schema": "gpuwm.remote.job.v1", "id": identifier, "token": secrets.token_hex(32),
              "created_at": _now(), "action": request["action"], **review, "argv": argv,
              "snapshot_config": str(snapshot_config), "snapshot_sha256": _sha(snapshots[snapshot_config.name]),
              "snapshot_plan": str(snapshot) if review.get("plan") else None,
              "snapshot_inputs": {name: _sha(payload) for name, payload in snapshots.items()},
              # The go door captures the WPS authority beside the snapshot; the
              # run-plan door binds the node's own copy through the plan and
              # captures nothing, so its record names the file that exists.
              "snapshot_wps_namelist": (None if review["wps_namelist"] is None
                                        else str(inputs / "prepared-inputs" / "namelist.wps")
                                        if "prepared-inputs/namelist.wps" in snapshots
                                        else str(review["wps_namelist"])),
              "original_inputs": {name: f"{index:03d}-{Path(name).name}" for index, name in enumerate(sources)}}
    if request.get("request_id") is not None:
        # Recorded on the job itself, and written before the worker starts, so
        # a retry that arrives while this launch is still starting is answered
        # with this job rather than a second one.
        record["request_id"] = request["request_id"]
    _write(directory / "job.json", record)
    command = ([sys.executable, "-I", "-u", "-m", "woof.remote_worker"]
               if worker_command is None else worker_command)
    env = dict(os.environ, **{TOKEN_ENV: record["token"]})
    try:
        inputs.mkdir(mode=0o700)
        originals.mkdir(mode=0o700)
        for name, payload in snapshots.items():
            target = inputs / name
            target.parent.mkdir(mode=0o700, exist_ok=True)
            target.write_bytes(payload)
        for name, payload in sources.items():
            (originals / record["original_inputs"][name]).write_bytes(payload)
            if _read(Path(name)) != payload:
                raise ValueError("remote inputs changed while capturing the job; review again before starting")
        # Claim the run folder exclusively before the worker can write a
        # forecast, so the folder this job records is the folder its run
        # writes into, and publish it as the case folder's newest run.
        _claim_output(review["outdir"], record.get("run_root") or review["outdir"])
        with (directory / "job.log").open("ab", buffering=0) as output:
            process = subprocess.Popen([*command, "--run", str(directory), "--token", record["token"]],
                                       cwd=review["cwd"], env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=output, start_new_session=True, close_fds=True)
    except (OSError, ValueError) as exc:
        _write(directory / "result.json", {"token": record["token"], "state": "failed", "exit_code": 2,
                                           "ended_at": _now(), "error": str(exc)})
        raise ValueError(f"job {identifier} failed before launch: {exc}") from exc
    deadline = time.monotonic() + 10
    while not (directory / "started.json").exists():
        if process.poll() is not None:
            _write(directory / "result.json", {"token": record["token"], "state": "failed",
                    "exit_code": process.returncode, "ended_at": _now(), "error": "remote worker failed before startup; read job logs"})
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(.02)
    # The RPC caller launches this detached watcher outside the durable job's
    # descendant tree. It can finish map preparation after integration ends.
    # The watcher only queues compact 2D maps; full science remains on demand.
    _start_watchers(workspace, identifier, directory, record["token"])
    return {"job": _status(directory)}


WATCHERS = ("background_maps", "native_plots")


def _watcher_starters():
    from woof.remote_preparation_v2 import ensure
    from woof.remote_native_plots import ensure as ensure_plots
    return {"background_maps": ensure, "native_plots": ensure_plots}


def _start_watchers(workspace, identifier, directory, token, *, only=None):
    """Start each detached watcher independently and record what failed, by name.

    A watcher that cannot start is a receipt, never a failed launch: the
    forecast is already running and one watcher's import error must not
    cancel the other or make a live job look like it never started.
    """
    failures = {}
    try:
        starters = _watcher_starters()
    except Exception as error:  # noqa: BLE001 - a watcher import must not fail a launch.
        starters = {}
        failures = {name: {"error": str(error)[:2000], "at": _now()} for name in WATCHERS}
    for name, start in starters.items():
        if only is not None and name not in only:
            continue
        try:
            start(workspace, identifier)
        except Exception as error:  # noqa: BLE001 - the launched job stays launched.
            failures[name] = {"error": str(error)[:2000], "at": _now()}
    receipt = directory / "preparation-start-error.json"
    if failures:
        _write(receipt, {"token": token, "at": _now(), **failures})
    elif only is not None:
        # Two pollers may retry the same watcher at once; the one that loses
        # the race must not turn the other's success into a status failure.
        receipt.unlink(missing_ok=True)
    return failures


def _stop(directory):
    job = _status(directory)
    if job["state"] in TERMINAL:
        return {"job": job}
    record = _record(directory)
    if job["state"] == "cleanup_failed":
        # The wrapper is gone, so this door re-runs its ladder on exactly the
        # processes the job still owns rather than waiting for a dead wrapper.
        surviving = _reap_owned(record["token"])
        if surviving:
            raise ValueError("stop could not terminate this job's surviving owned processes "
                             + ", ".join(str(item["pid"]) for item in surviving)
                             + "; inspect those processes on the node, then stop this job again")
        (directory / "cleanup-error.json").unlink(missing_ok=True)
        _write(directory / "result.json", {"token": record["token"], "state": "stopped", "exit_code": 1,
                                           "ended_at": _now(), "error": job.get("error")})
        return {"job": _status(directory)}
    if job["state"] != "running":
        raise ValueError(f"cannot prove ownership of this {job['state']} job; no signal was sent")
    _write(directory / "stop.json", {"token": record["token"], "requested_at": _now()})
    deadline = time.monotonic() + 16
    while time.monotonic() < deadline:
        job = _status(directory)
        if job["state"] in TERMINAL:
            # Wrapper may still be returning after its durable result. Exclude
            # only that verified owner; its child tree must actually be gone.
            owner = _json(directory / "started.json")["identity"]
            if not _owned_processes(record["token"], owner_pid=owner["pid"]):
                return {"job": job}
        time.sleep(.05)
    raise ValueError("stop has not confirmed termination of all owned processes; reconnect and inspect status/logs")


def _logs(directory, request):
    cursor, limit = request.get("cursor", 0), request.get("limit", 16384)
    if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= MAX_BYTES:
        raise ValueError("logs requires cursor >= 0 and limit between 1 and 131072")
    path = directory / "job.log"
    size = path.stat().st_size if path.exists() else 0
    if cursor > size:
        raise ValueError("log cursor is past the end of this job's log; retry with cursor 0")
    payload = b""
    if size:
        with path.open("rb") as stream:
            stream.seek(cursor)
            payload = stream.read(min(limit, 16384))
    # Keep a UTF-8 character split across chunks for the next read. At a tiny
    # caller limit, replacement is permitted and the byte cursor still advances.
    if len(payload) > 3 and cursor + len(payload) < size:
        for count in range(4):
            candidate = payload if count == 0 else payload[:-count]
            try:
                candidate.decode("utf-8")
                payload = candidate
                break
            except UnicodeDecodeError as error:
                if error.reason != "unexpected end of data":
                    break
    return {"job": _status(directory), "text": payload.decode("utf-8", errors="replace"),
            "cursor": cursor + len(payload), "eof": cursor + len(payload) >= size}


def dispatch(request):
    _ownership_provider()
    if not isinstance(request, dict) or request.get("schema") != "gpuwm.remote.request.v1":
        raise ValueError("unsupported remote request schema")
    allowed = {"schema", "action", "workspace", "config", "outdir", "geog_root", "prepared_root", "wps_namelist", "products", "section", "device", "request_id", "job", "cursor", "limit",
               "from_checkpoint", "dry_run", "expected_config_sha256", "expected_wps_sha256", "expected_input_sha256",
               "expected_checkpoint_sha256", "expected_checkpoint_set_sha256", "expected_prepared_sha256",
               "bundle", "bundle_id", "expected_bundle_sha256", "expected_plan_sha256", "domain", "inputs", "expected_source_blobs_sha256",
               "sequence", "after_sequence", "profile", "expected_run_id", "prefetch_sequences"}
    if set(request) - allowed:
        raise ValueError("unsupported remote request fields: " + ", ".join(sorted(set(request) - allowed)))
    workspace = _workspace(request)
    action = request.get("action")
    if action == "input-status":
        from woof.remote_input_transfer import status
        return status(request, workspace)
    if action == "artifacts":
        from woof.remote_artifacts import catalog
        return {"artifacts": catalog(request, workspace)}
    if action == "artifact-index":
        from woof.remote_artifacts import catalog
        value = catalog(request, workspace, metadata_only=True)
        from woof.remote_processed_v2 import index_metadata
        try:
            value["processed"] = index_metadata(workspace, request["job"])
        except (OSError, ValueError, RuntimeError) as error:
            value["processed"] = {"schema": "arwen.native-store-queue.v2", "job_id": request["job"],
                                  "state": "failed", "error": str(error)[:2000]}
        return {"artifact_index": value}
    if action == "processed-frame-v2":
        from woof.remote_processed_v2 import catalog
        return {"processed_frame": catalog(request, workspace)}
    if action == "native-plots":
        from woof.remote_native_plots import catalog
        return {"native_plots": catalog(request, workspace)}
    if action == "list-products":
        if set(request) - {"schema", "action", "workspace"}:
            raise ValueError("unsupported remote product catalog request fields")
        from woof.remote_processed_v2 import node_catalog
        return {"product_catalog": node_catalog(include_sections=True)}
    if action == "processed-frame":
        from woof.remote_processed import catalog
        return {"processed_frame": catalog(request, workspace)}
    if action in ("stage-plan", "review-plan", "start-plan"):
        plan_fields = {"schema", "action", "workspace"} | {
            "stage-plan": {"bundle"},
            "review-plan": {"bundle_id", "expected_bundle_sha256"},
            "start-plan": {"bundle_id", "expected_bundle_sha256", "expected_plan_sha256",
                           "expected_config_sha256", "expected_input_sha256",
                           "expected_source_blobs_sha256", "request_id"},
        }[action]
        if set(request) - plan_fields:
            raise ValueError("unsupported staged-plan request fields: " + ", ".join(sorted(set(request) - plan_fields)))
        from woof import remote_plan
        if action == "stage-plan":
            return remote_plan.stage(request.get("bundle"), workspace)
        if action == "review-plan":
            value, _bundle, _bundle_directory = remote_plan.review(request, workspace)
            return {"dry_run": True, "review": value}
        return remote_plan.launch(request, workspace)
    if action == "probe":
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ValueError("remote Python needs Linux pidfd process handles for safe job ownership")
        handle = os.pidfd_open(os.getpid(), 0)
        os.close(handle)
        from woof.remote_plan import hardware_probe
        probe = hardware_probe()
        return {"runtime": runtime(), "workspace": str(workspace), "probe": probe,
                "capabilities": capabilities()}
    if action in ("start", "resume"):
        if type(request.get("dry_run", False)) is not bool:
            raise ValueError("dry_run must be a boolean")
        return _launch(request, workspace)
    if action == "list":
        limit = request.get("limit", 20)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("list limit must be between 1 and 100")
        store = _store(workspace)
        paths = [] if not store.exists() else sorted((path for path in store.iterdir()
                if JOB_ID.fullmatch(path.name) and path.is_dir() and not path.is_symlink()), reverse=True)
        jobs = []
        for path in paths[:limit]:
            try:
                job_summary = _status(_directory(workspace, path.name))
                # Selected status/logs carry the full native render receipt.
                # A twenty-job list must not multiply that 64 KiB payload.
                job_summary.pop("render_summary", None)
                jobs.append(job_summary)
            except (OSError, ValueError, KeyError) as exc:
                jobs.append({"id": path.name, "state": "unreadable", "error": str(exc)[:1000]})
        return {"jobs": jobs}
    if action not in ("status", "logs", "stop"):
        raise ValueError("unsupported remote action")
    directory = _directory(workspace, request.get("job"))
    if action == "status":
        job = _status(directory)
        retry = [name for name in WATCHERS
                 if isinstance(job.get(name), dict) and job[name].get("state") == "start_failed"]
        if retry and job["state"] not in TERMINAL:
            _start_watchers(workspace, request.get("job"), directory, _record(directory)["token"], only=set(retry))
            job = _status(directory)
        return {"job": job}
    if action == "logs":
        return _logs(directory, request)
    return _stop(directory)


def run_worker(directory, token):
    directory = Path(directory).resolve(strict=True)
    record = _record(directory)
    if token != record["token"] or os.environ.get(TOKEN_ENV) != token:
        raise ValueError("worker token does not match its durable job record")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ValueError("remote Python needs Linux pidfd process handles for safe job ownership")
    handle = os.pidfd_open(os.getpid(), 0)
    os.close(handle)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot own orphaned remote job descendants")
    owner = _process(os.getpid())
    _write(directory / "started.json", {"token": token, "identity": owner, "started_at": _now(),
                                       "run_root": record.get("run_root", record["outdir"])})
    from woof.remote_artifacts import plan_binding
    # Every route whose run publishes a manifest records its runner receipt.
    # That receipt is the only proof of the runner-exit window, so asking the
    # action word here left a configuration job's completion unprovable at the
    # very doors that now serve it, and refused with a race that never happened.
    records_runner = plan_binding(record) is not None
    code, stopped, error = 1, False, None
    child = None
    try:
        job_bytes = _read(directory / "job.json")
        if json.loads(job_bytes) != record:
            raise ValueError("job ownership changed before the worker started")
        for name, digest in record["snapshot_inputs"].items():
            if _file_sha(directory / "inputs" / name) != digest:
                raise ValueError("captured configuration or companion changed before the worker started")
        if record.get("prepared_document") and _file_sha(Path(record["prepared_document"])) != record["prepared_sha256"]:
            raise ValueError("prepared receipt changed before the worker started; review again")
        for path, digest in record.get("checkpoint_inputs", {}).items():
            if _file_sha(Path(path)) != digest:
                raise ValueError("checkpoint set changed before the worker started; review again")
        for name, digest in record.get("external_inputs", {}).items():
            path = Path(name)
            if path.is_symlink() or not path.is_file() or _file_sha(path) != digest:
                raise ValueError("captured raw forcing or its receipt changed before the worker started")
        # The selected card is exported before the child can create a context:
        # the mask is read at context creation, so a selection made afterwards
        # would name one card and run on another. The ownership token stays in
        # this environment, because the child's ownership depends on it.
        child_env = dict(os.environ)
        if record.get("device") is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = str(record["device"])
        child = subprocess.Popen(record["argv"], cwd=record["cwd"], env=child_env,
                                 stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        recorded_runner = False

        def capture_runner():
            # Capture once, from a live and token-owning child. Unverifiable
            # children get no receipt and therefore no completion retry
            # privilege. Attempted before the first poll as well: a runner that
            # exits inside the first loop interval would otherwise leave its
            # own exit window unprovable and permanently refused.
            nonlocal recorded_runner
            if recorded_runner or not records_runner:
                return
            identity = _process(child.pid)
            if (identity is not None and _has_token(child.pid, token)
                    and _process(child.pid) == identity):
                _write(directory / "runner.json", {"token": token, "owner": owner,
                    "identity": identity, "job_sha256": _sha(job_bytes)})
                recorded_runner = True

        capture_runner()
        while child.poll() is None:
            capture_runner()
            marker = directory / "stop.json"
            if marker.exists() and _json(marker).get("token") == token:
                stopped = True
                break
            time.sleep(.1)
        # Also reap surviving descendants after an unsuccessful parent exit.
        # Every signal rechecks the full process identity and inherited token.
        if stopped or _owned_processes(token):
            _reap_owned(token, child=child)
        code = child.wait(timeout=1)
        if _owned_processes(token):
            raise RuntimeError("owned descendants still exist after termination attempts")
    except BaseException as exc:
        error = str(exc)
        traceback.print_exc()
        # A failed cleanup is explicitly nonterminal; never claim stop success.
        if _owned_processes(token):
            _write(directory / "cleanup-error.json", {"token": token, "error": error, "at": _now(),
                                                      "surviving": _owned_processes(token)})
            return 1
    _write(directory / "result.json", {"token": token, "state": "stopped" if stopped else "completed" if code == 0 and error is None else "failed",
           "exit_code": code, "ended_at": _now(), "error": error})
    return 0


def keepalive_writer(action, stream=None, *, interval=KEEPALIVE_SECONDS):
    """Say, every few seconds, that this node is still working on one request.

    A client cannot tell a node that is thinking from a node that is gone, so
    its deadline either covers the slowest review this node can do or it cuts
    one short. The node says so instead, and the client's deadline is reset by
    the saying rather than guessed at.
    """
    stream = sys.stdout.buffer if stream is None else stream
    stop = threading.Event()

    def beat():
        while not stop.wait(interval):
            try:
                stream.write((json.dumps({"schema": KEEPALIVE_SCHEMA, "action": action,
                                          "at": _now()}, separators=(",", ":")) + "\n").encode("utf-8"))
                stream.flush()
            except (OSError, ValueError):
                return

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()

    def finish():
        stop.set()
        thread.join(timeout=2)
    return finish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc", action="store_true")
    parser.add_argument("--artifact-stream", action="store_true")
    parser.add_argument("--processed-stream", action="store_true")
    parser.add_argument("--processed-member-stream-v2", action="store_true")
    parser.add_argument("--input-stream", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--token")
    args = parser.parse_args(argv)
    if args.processed_member_stream_v2:
        from woof.remote_processed_v2 import stream_main
        return stream_main()
    if args.processed_stream:
        from woof.remote_processed import stream_main
        return stream_main()
    if args.input_stream:
        from woof.remote_input_transfer import receive_main
        return receive_main()
    if args.artifact_stream:
        from woof.remote_artifacts import stream_main
        return stream_main()
    if args.run:
        return run_worker(args.run, args.token)
    action = "unknown"
    try:
        if not args.rpc:
            raise ValueError("select --rpc or --run")
        payload = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("remote request exceeds 128 KiB")
        request = json.loads(payload)
        if isinstance(request, dict) and isinstance(request.get("action"), str):
            action = request["action"][:128]
        # A client that asked to be kept awake is told this node is still
        # working, on the same stream, until the one reply line is written.
        finish = (keepalive_writer(action)
                  if isinstance(request, dict) and request.pop("keepalive", None) is True else None)
        # The protocol owns stdout even when a schema validator prints a note.
        import contextlib
        try:
            with contextlib.redirect_stdout(sys.stderr):
                data = dispatch(request)
        finally:
            if finish is not None:
                finish()
        reply = {"schema": SCHEMA, "ok": True, "action": action, **data}
    except Exception as exc:
        reply = {"schema": SCHEMA, "ok": False, "action": action,
                 "error": {"type": type(exc).__name__, "message": str(exc)[:8000]}}
    output = json.dumps(reply, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(output.encode("utf-8")) > MAX_BYTES:
        reply = {"schema": SCHEMA, "ok": False, "action": action,
                 "error": {"type": "ValueError", "message": "remote result exceeds 128 KiB; request fewer jobs"}}
        output = json.dumps(reply, separators=(",", ":"))
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write((output + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    else:
        print(output, flush=True)
    return 0 if reply["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
