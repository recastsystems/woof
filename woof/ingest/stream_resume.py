"""Portable forcing prefixes and a verified continuation of a live producer.

A checkpoint owns an immutable prepared head and every interval its restore
needs. The prefix is copied while later intervals are still being prepared.
A fresh producer may reconstruct its preparation accumulator, but it never
integrates the model from time zero or replaces a byte the checkpoint bound.
Only new, hash-checked interval payloads are exposed to the resumed forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import uuid

from woof.ingest import boundary_stream as stream

PREFIX_SCHEMA = "gpuwm-forcing-continuation-v1"
SOURCE_SCHEMA = "gpuwm-forcing-continuation-source-v1"
DESCRIPTOR_NAME = "continuation.json"
POSTED_PREFIX_DIRNAME = "posted-prefix"
CONTINUATION_ENV = "WOOF_CONTINUATION_PREFIX"
#: A restored prefix is runnable: ``woof go --prepared-root PREFIX
#: --restart`` binds it at its pinned head (stage_cli.resolve_bundle) and its
#: continuation re-prepares an identical head.  Callers that store prefixes
#: for a resume (the WOOF worker's checkpoint inputs) read this marker:
#: before it (gpuwm <= 2.9.8, run_515470d250a1060a) a stored prefix could
#: never be resumed, so they store sealed inputs only.
PREFIX_BINDS_AT_HEAD = True
PRODUCER_LOCK_ENV = "WOOF_CONTINUATION_PRODUCER_LOCK_FILE"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _relative(value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise stream.BoundaryStreamError(f"unsafe continuation path {value!r}")
    return path


def _copy_verified(source, destination, expected=None, *, share_existing=False):
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        raise stream.BoundaryStreamError(f"continuation cannot copy a link: {source}")
    digest = _sha(source)
    if expected is not None and digest != expected:
        raise stream.BoundaryStreamError(f"continuation payload fails its hash: {source}")
    if destination.exists():
        if destination.is_symlink() or _sha(destination) != digest:
            raise stream.BoundaryStreamError(f"continuation changes an immutable prefix file: {destination}")
        if (share_existing and source.suffix in {".npy", ".npz"}
                and source.resolve() != destination.resolve()
                and not os.path.samefile(source, destination)):
            # Only the owned replay producer opts into replacing its source.
            # Place the transient link outside the cache directory so its
            # strict array inventory never observes an extra payload file.
            shared = source.parent.parent / f".{source.name}-{uuid.uuid4().hex}.continuation-share"
            try:
                os.link(destination, shared)
                os.replace(shared, source)
            except OSError:
                # Cross-filesystem or unsupported links retain the verified
                # source copy. No unverified byte is substituted.
                shared.unlink(missing_ok=True)
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".continuation-tmp")
    if source.suffix in {".npy", ".npz"}:
        # Prepared payloads are immutable after publication. Linking their
        # names on one filesystem shares the one physical prepared copy,
        # while deleting either name later leaves the other intact.
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
    else:
        shutil.copyfile(source, temporary)
    if _sha(temporary) != digest:
        raise stream.BoundaryStreamError(f"continuation copy failed verification: {destination}")
    os.replace(temporary, destination)
    return digest


def _copy_array(source_directory, target_directory, key, spec, *, share_existing=False):
    from woof.ingest.prepared_cache import read_manifest_array
    read_manifest_array(source_directory, key, spec)
    return _copy_verified(Path(source_directory) / _relative(spec["file"]),
                          Path(target_directory) / _relative(spec["file"]),
                          share_existing=share_existing)


def source_descriptor(config, *, transport=None, supplements=()):
    """Bind the frozen source cycle and authority bytes, without local paths."""
    config = Path(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    fetch = raw.get("fetch") or {}
    if not fetch.get("source") or not fetch.get("cycle") or fetch["cycle"] == "latest":
        raise stream.BoundaryStreamError("a continuation requires a frozen source and cycle")
    authorities = {config.name: _sha(config)}
    for suffix in (".namelist.wps", ".namelist.input", ".stock.namelist.input", ".d01-target.json"):
        path = config.with_name(config.stem + suffix)
        if path.is_file():
            authorities[path.name] = _sha(path)
    for name in ("namelist.wps", "namelist.input", "stock.namelist.input", "d01-target.json"):
        path = config.parent / name
        if path.is_file():
            authorities[path.name] = _sha(path)
    return {"schema": SOURCE_SCHEMA, "source": str(fetch["source"]),
            "cycle": str(fetch["cycle"]), "fetch": fetch,
            "config_name": config.name, "authorities": authorities,
            "transport": transport, "supplements": list(supplements)}


def seal_prefix(prepared_root, checkpoint_seconds, output_root, *, source=None,
                required_intervals=None):
    """Copy a committed forcing prefix and publish its inventory last.

    The prefix includes the interval containing the first step after the
    checkpoint. At an exact forcing boundary this is the next interval.
    A checkpoint's declared count may ask for a longer prefix than its clock
    alone, so callers pass that count too. Missing forcing defers publication.
    """
    root, output = Path(prepared_root), Path(output_root)
    seconds = float(checkpoint_seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("checkpoint_seconds must be finite and nonnegative")
    if output.exists():
        raise FileExistsError(f"refusing existing forcing prefix {output}")
    head = stream.read_head(root)
    cache = head["basis"]["cache"]
    schedule = cache["lbc"]["schedule"]
    count = next((index + 1 for index, (_, end) in enumerate(schedule)
                  if float(end) > seconds), len(schedule))
    count = max(count, int(required_intervals or 0))
    if seconds > float(schedule[-1][1]) or count > len(schedule):
        raise stream.BoundaryStreamError("forcing prefix ends before the checkpoint clock")
    intervals = stream.StreamedIntervals(root, head=head)
    arrays = dict(cache["arrays"])
    markers = []
    for index in range(count):
        path = stream.segment_marker_path(root, index)
        if not path.is_file():
            raise stream.BoundaryStreamError(f"checkpoint forcing interval {index} is not published")
        marker = intervals.require(index)
        arrays.update(marker["arrays"])
        markers.append(marker)
    output.mkdir(parents=True)
    cache_dir = _relative(cache["directory"])
    # The root cache's array table is authoritative. A writer may have
    # already allocated later array files, which must not enter the prefix.
    # The companion WRF export is written at the seal, from the seal's own
    # cache header.  A prefix cut after it was written but before the seal
    # document carried a manifest bound to the ORIGINAL header, which the
    # prefix never holds; the continuation's seal writes a fresh header and
    # export, so its copy refused as "continuation changes an immutable
    # prefix file: wrf-native-input/manifest.json" (found 2026-10-26 on the
    # run_515470d250a1060a resume repro).  The seal is the export's author.
    from woof.native_domain_artifacts import WRF_EXPORT_DIRNAME
    excluded = {stream.STREAM_DIRNAME, stream.SEALED_HIERARCHY_DIRNAME,
                stream.proof_document_name(head), "public-wrapper-result.json",
                DESCRIPTOR_NAME, WRF_EXPORT_DIRNAME}
    inventory = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in excluded or relative == cache_dir / "header.json":
            continue
        if relative.parent == cache_dir:
            continue
        if (any(part.startswith(".") for part in relative.parts) or path.name.endswith(".tmp")
                or path.name.startswith("progress") or path.name.endswith(".log")):
            continue
        inventory[relative.as_posix()] = _copy_verified(path, output / relative)
    head_relative = Path(stream.STREAM_DIRNAME) / stream.HEAD_NAME
    inventory[head_relative.as_posix()] = _copy_verified(root / head_relative, output / head_relative)
    for key, spec in arrays.items():
        relative = cache_dir / _relative(spec["file"])
        inventory[relative.as_posix()] = _copy_array(root / cache_dir, output / cache_dir, key, spec)
    for index in range(count):
        relative = stream.segment_marker_path(Path("."), index)
        inventory[relative.as_posix()] = _copy_verified(root / relative, output / relative)
    posted = stream.stream_dir(root) / POSTED_PREFIX_DIRNAME
    if posted.is_dir():
        for path in sorted(posted.glob("f*.json")):
            relative = path.relative_to(root)
            inventory[relative.as_posix()] = _copy_verified(path, output / relative)
    payload = {"schema": PREFIX_SCHEMA, "head_sha256": head["head_sha256"],
               "checkpoint_seconds": seconds, "prefix_intervals": count,
               "prefix_end_seconds": float(schedule[count - 1][1]),
               "source": source, "files": inventory,
               "prefix_bytes": sum((output / path).stat().st_size for path in inventory)}
    payload["descriptor_sha256"] = _digest(payload)
    stream._write_json_atomic(output / DESCRIPTOR_NAME, payload)
    return payload


def verify_prefix(root):
    root = Path(root)
    descriptor = _json(root / DESCRIPTOR_NAME)
    unsigned = dict(descriptor)
    claimed = unsigned.pop("descriptor_sha256", None)
    if descriptor.get("schema") != PREFIX_SCHEMA or claimed != _digest(unsigned):
        raise stream.BoundaryStreamError("forcing continuation descriptor fails its digest")
    head = stream.read_head(root, expected_sha256=descriptor["head_sha256"])
    for name, digest in descriptor["files"].items():
        path = root / _relative(name)
        if path.is_symlink() or _sha(path) != digest:
            raise stream.BoundaryStreamError(f"forcing prefix fails verification: {name}")
    return descriptor, head


def _beat(root, *, waiting_for=None):
    record = {"pid": os.getpid(), "host": socket.gethostname(), "state": "continuing",
              "updated_epoch": time.time(), "updated_utc": stream._utc_now(),
              "slowest_build_seconds": 0.0}
    if waiting_for is not None:
        record.update(state="waiting_for_source", waiting_for=waiting_for)
    stream._write_json_atomic(stream.stream_dir(root) / stream.PRODUCER_NAME, record)


def preserve_posted_marker(candidate):
    """Keep a sealed lead's provenance after re-verifying its object bytes."""
    held_root = os.environ.get(CONTINUATION_ENV)
    if not held_root:
        return candidate
    path = stream.stream_dir(held_root) / POSTED_PREFIX_DIRNAME / stream.posted_lead_marker_name(candidate["lead"])
    if not path.is_file():
        return candidate
    held = _json(path)
    # The new fetch may observe a different endpoint or wall time. Object
    # names, lengths and byte hashes, source/member/cycle/valid time must
    # match before the original, already authenticated provenance is used.
    def objects(value):
        return sorted((item.get("name"), item.get("bytes"), item.get("sha256"))
                      for item in value or ())
    keys = ("schema", "source", "member", "cycle", "lead", "valid_time")
    if any(held.get(key) != candidate.get(key) for key in keys) or any(
            objects(held.get(key)) != objects(candidate.get(key)) for key in ("objects", "composed")):
        raise stream.BoundaryStreamError(f"continuation source changed sealed lead {candidate['lead']}")
    return held


def preserve_child_receipts(root, head):
    """Reuse declared telemetry only after the scientific head/cache bytes match."""
    held_root = os.environ.get(CONTINUATION_ENV)
    if not held_root:
        return head
    held = stream.read_head(held_root)
    # A native nested route first prepares its single-domain root before
    # constructing the tree that this prefix actually owns.
    if (head["basis"]["cache"]["directory"] != held["basis"]["cache"]["directory"]
            or (head["basis"].get("tree") is None) != (held["basis"].get("tree") is None)):
        return head
    cache = head["basis"]["cache"]
    if cache != held["basis"]["cache"]:
        raise stream.BoundaryStreamError("continuation changes its immutable prepared head cache")
    from woof.ingest.prepared_cache import read_manifest_array
    for key, spec in cache["arrays"].items():
        read_manifest_array(Path(root) / cache["directory"], key, spec)
        read_manifest_array(Path(held_root) / cache["directory"], key, spec)
    # The cache binds the existing preprocessing identity contract. Its
    # proof also carries machine/worker measurements, which may differ on
    # a replacement box without changing any prepared array.
    from woof.ingest.preprocess_backend import preprocess_identity, preprocess_reports_identity
    proof, old_proof = head["basis"]["proof_head"], held["basis"]["proof_head"]
    if "preprocessing" in proof and "preprocessing" in old_proof:
        if preprocess_identity(proof["preprocessing"]) != preprocess_identity(old_proof["preprocessing"]):
            raise stream.BoundaryStreamError("continuation changes its preprocessing implementation")
        proof["preprocessing"] = old_proof["preprocessing"]
        if "preprocessing_receipt_sha256" in old_proof:
            proof["preprocessing_receipt_sha256"] = old_proof["preprocessing_receipt_sha256"]
    for key in ("workers", "hierarchy_workers"):
        if key in proof and key in old_proof:
            proof[key] = old_proof[key]
    tree = head["basis"].get("tree")
    if tree is None:
        if head["basis"] != held["basis"]:
            raise stream.BoundaryStreamError("continuation changes its immutable prepared head")
        return head
    replacements = []
    for label in tree.get("children_receipts") or {}:
        relative = Path(stream.HIERARCHY_HEAD_DIRNAME) / "domains" / label
        old_dir, new_dir = Path(held_root) / relative, Path(root) / relative
        old_receipt, new_receipt = _json(old_dir / "receipt.json"), _json(new_dir / "receipt.json")
        old_artifacts = json.loads(json.dumps(old_receipt.get("artifacts")))
        new_artifacts = json.loads(json.dumps(new_receipt.get("artifacts")))
        # Some receipts bind header file bytes in addition to the cache's
        # content digest. Remove only that raw-file digest for this comparison;
        # the complete scientific header is checked immediately below.
        for artifacts in (old_artifacts, new_artifacts):
            if isinstance(artifacts, dict):
                prepared_artifact = artifacts.get("prepared_cache")
                if isinstance(prepared_artifact, dict):
                    prepared_artifact.pop("header_sha256", None)
        if old_artifacts != new_artifacts:
            raise stream.BoundaryStreamError(f"continuation changes {label}'s artifacts")
        def receipt_identity(receipt, artifacts):
            identity = dict(receipt)
            identity["artifacts"] = artifacts
            # native_domain_artifacts records this wall time beside the
            # cache, whose metadata excludes it by the existing contract.
            identity.pop("input_preparation_seconds", None)
            if "preprocess_receipt" in identity:
                identity["preprocess_receipt"] = preprocess_reports_identity(
                    preprocess_identity(identity["preprocess_receipt"]))
            return identity
        if receipt_identity(old_receipt, old_artifacts) != receipt_identity(new_receipt, new_artifacts):
            raise stream.BoundaryStreamError(f"continuation changes {label}'s receipt identity")
        old_cache, new_cache = _json(old_dir / "prepared-cache/header.json"), _json(new_dir / "prepared-cache/header.json")
        for receipt, directory in ((old_receipt, old_dir), (new_receipt, new_dir)):
            digest = receipt.get("artifacts", {}).get("prepared_cache", {}).get("header_sha256")
            if digest is not None and digest != _sha(directory / "prepared-cache/header.json"):
                raise stream.BoundaryStreamError(f"continuation {label}'s receipt fails its header digest")
        if ({key: value for key, value in old_cache.items() if key != "created_utc"}
                != {key: value for key, value in new_cache.items() if key != "created_utc"}):
            raise stream.BoundaryStreamError(f"continuation changes {label}'s prepared cache")
        excluded = {Path("receipt.json"), Path("prepared-cache/header.json"),
                    Path("preparation-runtime.json")}
        def payloads(directory):
            return {path.relative_to(directory) for path in directory.rglob("*")
                    if path.is_file() and path.relative_to(directory) not in excluded}
        old_payloads = payloads(old_dir)
        if old_payloads != payloads(new_dir):
            raise stream.BoundaryStreamError(f"continuation changes {label}'s payload inventory")
        for relative_payload in old_payloads:
            if _sha(old_dir / relative_payload) != _sha(new_dir / relative_payload):
                raise stream.BoundaryStreamError(f"continuation changes {label}'s payload {relative_payload}")
        replacements.append((old_dir, new_dir))
        tree["children_receipts"][label] = held["basis"]["tree"]["children_receipts"][label]
    if head["basis"] != held["basis"]:
        raise stream.BoundaryStreamError("continuation changes its immutable prepared head")
    # No original provenance is adopted until every domain's bound bytes
    # and every unknown head/receipt field has passed verification.
    for old_dir, new_dir in replacements:
        header = new_dir / "prepared-cache/header.json"
        temporary = header.with_name("header.json.continuation-tmp")
        shutil.copyfile(old_dir / "prepared-cache/header.json", temporary)
        os.replace(temporary, header)
        shutil.copyfile(old_dir / "receipt.json", new_dir / "receipt.json")
    return head


def preserve_seal_metadata(root, proof):
    """Seal the original head and retain the fresh producer's own telemetry."""
    held_root = os.environ.get(CONTINUATION_ENV)
    if not held_root:
        return proof
    held = stream.read_head(held_root)
    actual = stream.read_head(root)
    if actual["head_sha256"] != held["head_sha256"]:
        return proof
    from woof.ingest.preprocess_backend import preprocess_identity
    old = held["basis"]["proof_head"]
    measurements = {}
    if "preprocessing" in proof and "preprocessing" in old:
        if preprocess_identity(proof["preprocessing"]) != preprocess_identity(old["preprocessing"]):
            raise stream.BoundaryStreamError("continuation seal changes its preprocessing implementation")
        measurements["preprocessing"] = proof["preprocessing"]
        proof = dict(proof)
        proof["preprocessing"] = old["preprocessing"]
        if "preprocessing_receipt_sha256" in old:
            proof["preprocessing_receipt_sha256"] = old["preprocessing_receipt_sha256"]
    for key in ("workers", "hierarchy_workers"):
        if key in proof and key in old:
            measurements[key] = proof[key]
            proof = dict(proof)
            proof[key] = old[key]
    if measurements:
        stream._write_json_atomic(stream.stream_dir(root) / "continuation-preparation.json", {
            "schema": "gpuwm-continuation-preparation-receipt-v1",
            "head_sha256": held["head_sha256"], "fresh_producer": measurements})
    return proof


def _find_producer_root(root, head_digest):
    root = Path(root)
    paths = [root / stream.STREAM_DIRNAME / stream.HEAD_NAME]
    if root.is_dir():
        # Source routes publish their prepared tree by renaming a hidden
        # sibling inside this root. A head written there is not published:
        # pinning its path would race the rename. The root itself may be
        # the worker-owned .continuation directory, so inspect relatives.
        paths.extend(path for path in root.rglob(f"{stream.STREAM_DIRNAME}/{stream.HEAD_NAME}")
                     if not any(part.startswith(".") for part in path.relative_to(root).parts))
    for path in paths:
        if path.is_file():
            candidate_root = path.parent.parent
            candidate = stream.read_head(candidate_root)
            if candidate["head_sha256"] == head_digest:
                return candidate_root, candidate
    return None, None


def continue_prefix(prepared_root, producer_root, *, producer_argv=None,
                    poll_seconds=0.2, timeout_seconds=1800.0, on_segment=None):
    """Relay a fresh producer into a restored prefix, with marker-last writes.

    ``producer_argv`` is a local test seam, never carried in the uploaded
    descriptor. Production builds that command from the frozen authorities.
    This call owns its subprocess and reports a failed stream on every error.
    """
    target = Path(prepared_root)
    descriptor, head = verify_prefix(target)
    for name in (stream.FAILED_NAME, stream.STOP_NAME):
        if (stream.stream_dir(target) / name).exists():
            raise stream.BoundaryStreamError(f"restored prefix has a terminal {name} marker")
    _beat(target)
    process = None
    producer_lease = None
    environment = dict(os.environ)
    environment[CONTINUATION_ENV] = str(target.resolve())
    deadline = time.monotonic() + float(timeout_seconds)
    source = None
    validated = set()
    completed = descriptor["prefix_intervals"]
    total = len(head["basis"]["cache"]["lbc"]["schedule"])
    cache = _relative(head["basis"]["cache"]["directory"])
    signal_handlers = {}
    if os.name != "nt" and threading.current_thread() is threading.main_thread():
        import signal
        def stop_continuation(signum, frame):
            raise stream.BoundaryStreamStopped("the continuation process was stopped")
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal_handlers[signum] = signal.signal(signum, stop_continuation)
    try:
        lock_path = environment.get(PRODUCER_LOCK_ENV)
        if producer_argv and lock_path:
            if os.name == "nt":
                raise stream.BoundaryStreamError("a queued continuation producer requires a Linux worker")
            import fcntl
            producer_lease = Path(lock_path).open("a+b")
            while True:
                try:
                    fcntl.flock(producer_lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    _beat(target)
                    if (stream.stream_dir(target) / stream.STOP_NAME).exists():
                        raise stream.BoundaryStreamStopped("the forecast stopped its queued continuation producer")
                    if time.monotonic() >= deadline:
                        raise stream.BoundaryStreamError("continuation producer exceeded its queue time limit")
                    time.sleep(max(.1, float(poll_seconds)))
            # Time waiting behind another member is not preparation work.
            deadline = time.monotonic() + float(timeout_seconds)
        if producer_argv:
            process = subprocess.Popen(list(producer_argv), env=environment,
                                       start_new_session=os.name != "nt")
        while True:
            if time.monotonic() >= deadline:
                raise stream.BoundaryStreamError("continuation producer exceeded its time limit")
            if (stream.stream_dir(target) / stream.STOP_NAME).exists():
                raise stream.BoundaryStreamStopped("the resumed forecast stopped its continuation producer")
            if source is None:
                source, source_head = _find_producer_root(producer_root, head["head_sha256"])
                if source is not None:
                    # Immutable head arrays and all sealed prefix rows are
                    # checked before even one new row is exposed.
                    for key, spec in head["basis"]["cache"]["arrays"].items():
                        _copy_array(source / cache, target / cache, key, spec,
                                    share_existing=process is not None)
            if source is not None:
                beat = stream._read_json(stream.stream_dir(source) / stream.PRODUCER_NAME) or {}
                _beat(target, waiting_for=beat.get("waiting_for"))
                failure = stream._read_json(stream.stream_dir(source) / stream.FAILED_NAME)
                if failure:
                    if failure.get("code") == stream.SOURCE_BEHIND_CODE:
                        raise stream.SourceBehind(failure.get("details"), message=failure.get("reason"))
                    raise stream.BoundaryProducerFailed(str(failure.get("reason")))
                for index in range(total):
                    marker_path = stream.segment_marker_path(source, index)
                    if not marker_path.is_file():
                        break
                    if index in validated:
                        continue
                    marker = _json(marker_path)
                    if marker.get("head_sha256") != head["head_sha256"] or marker.get("index") != index:
                        raise stream.BoundaryStreamError(f"continuation interval {index} belongs to another head")
                    target_marker = stream.segment_marker_path(target, index)
                    if target_marker.exists() and _json(target_marker) != marker:
                        raise stream.BoundaryStreamError(f"continuation changes sealed interval {index}")
                    for key, spec in marker["arrays"].items():
                        _copy_array(source / cache, target / cache, key, spec,
                                    share_existing=process is not None)
                    if index >= completed:
                        _copy_verified(marker_path, target_marker)
                        completed = index + 1
                        if on_segment is not None:
                            on_segment(index)
                    validated.add(index)
                if stream.prepared_tree_complete(source):
                    stream.verify_seal(source, head=head)
                    if process is not None and process.wait(timeout=30) != 0:
                        raise stream.BoundaryProducerFailed("continuation preparation failed after publishing its seal")
                    # The seal's companions are copied before its completion
                    # document. Root prefix payloads remain immutable.
                    proof_name = stream.proof_document_name(head)
                    for path in sorted(source.rglob("*")):
                        if not path.is_file():
                            continue
                        relative = path.relative_to(source)
                        if relative == Path(proof_name) or relative == Path(stream.STREAM_DIRNAME) / stream.HEAD_NAME:
                            continue
                        if relative.parts[0] == stream.STREAM_DIRNAME and path.name in {
                                stream.PRODUCER_NAME, stream.FAILED_NAME, stream.STOP_NAME}:
                            continue
                        if path.name.endswith(".tmp") or ".tmp-" in str(relative):
                            continue
                        _copy_verified(path, target / relative,
                                       share_existing=process is not None)
                    _copy_verified(source / proof_name, target / proof_name)
                    seal = stream.verify_seal(target, head=head)
                    return {"schema": "gpuwm-forcing-continuation-receipt-v1", "status": "PASS",
                            "checkpoint_seconds": descriptor["checkpoint_seconds"],
                            "prefix_intervals": descriptor["prefix_intervals"],
                            "continued_intervals": total - descriptor["prefix_intervals"], **seal}
            else:
                _beat(target)
            if process is not None and process.poll() is not None:
                raise stream.BoundaryProducerFailed(f"continuation preparation exited {process.returncode} without its pinned head and seal")
            time.sleep(float(poll_seconds))
    except BaseException as error:
        failure = {"reason": str(error), "updated_utc": stream._utc_now()}
        if isinstance(error, stream.SourceBehind):
            failure.update(code=error.code, details=error.details)
        stream._write_json_atomic(stream.stream_dir(target) / stream.FAILED_NAME, failure)
        if process is not None and process.poll() is None:
            if os.name != "nt":
                import signal
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
        raise
    finally:
        if producer_lease is not None:
            producer_lease.close()
        for signum, handler in signal_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    seal = actions.add_parser("seal-prefix")
    seal.add_argument("--prepared-root", type=Path, required=True)
    seal.add_argument("--checkpoint-seconds", type=float)
    seal.add_argument("--checkpoint", type=Path)
    seal.add_argument("--output-root", type=Path, required=True)
    seal.add_argument("--config", type=Path)
    seal.add_argument("--transport")
    continuation = actions.add_parser("continue")
    continuation.add_argument("--prepared-root", type=Path, required=True)
    continuation.add_argument("--producer-root", type=Path, required=True)
    continuation.add_argument("--config", type=Path)
    continuation.add_argument("--data-dir", type=Path)
    continuation.add_argument("--geog-root", type=Path)
    continuation.add_argument("--timeout-seconds", type=float, default=1800)
    continuation.add_argument("--producer", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "seal-prefix":
        count = None
        seconds = args.checkpoint_seconds
        if args.checkpoint is not None:
            from woof.io.restart import read_restart_header
            header = read_restart_header(args.checkpoint)
            seconds = float(header["elapsed_seconds"])
            count = len((header.get("lateral_boundary_prefix") or {}).get("intervals") or ())
        if seconds is None:
            parser.error("seal-prefix requires --checkpoint or --checkpoint-seconds")
        source = None if args.config is None else source_descriptor(args.config, transport=args.transport)
        result = seal_prefix(args.prepared_root, seconds, args.output_root, source=source, required_intervals=count)
    else:
        command = args.producer
        if args.config is not None:
            descriptor, _ = verify_prefix(args.prepared_root)
            source = descriptor.get("source")
            if not isinstance(source, dict) or source.get("schema") != SOURCE_SCHEMA:
                raise stream.BoundaryStreamError("continuation has no frozen source descriptor")
            for name, digest in source["authorities"].items():
                path = args.config.parent / _relative(name)
                if _sha(path) != digest:
                    raise stream.BoundaryStreamError(f"continuation authority changed: {name}")
            command = [sys.executable, "-m", "woof", "go", str(args.config),
                       "--prepare-only", "--run-stamp", "off", "--outdir", str(args.producer_root)]
            for key in ("data_dir", "geog_root"):
                value = getattr(args, key)
                if value is not None:
                    command += ["--" + key.replace("_", "-"), str(value)]
            if source.get("transport") is not None:
                command += ["--transport", source["transport"]]
            for value in source.get("supplements") or ():
                command += ["--supplement", str(value)]
        result = continue_prefix(args.prepared_root, args.producer_root, producer_argv=command,
                                 timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except stream.SourceBehind as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(error.exit_code) from None
