"""Bind durable restart sets to the original ensemble member identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from pathlib import Path
from woof.ensemble.physical_fields import sha256_of_file as digest_file

CONTRACT = "gpuwm-ensemble-restart.v1"
IDENTITIES = "ensemble-identities.json"
ROSTER = "ensemble-restart.json"
INPUTS = "ensemble-inputs.json"
#: Sealed copies of every file a roster names, one directory per seal. The
#: live run directory keeps changing after a seal: the next frame rewrites the
#: domain ensemble-manifest.json and ensemble-history-ledger.json within a
#: second, and members retire superseded checkpoint sets and pending packs.
#: A roster that named only the live files was valid for under a second, so a
#: killed or stopped ensemble could never resume (2.9.1 audit, one card,
#: "ensemble resume file is missing or hash-mismatched: d01/ensemble-manifest.json").
GENERATIONS = ".ensemble-resume/generations"


class CheckpointProgress:
    """Seal member-owned output state after the native checkpoint commits."""
    hosts_forecast = True

    def __init__(self, callback, *, collector, root, manifest_lock):
        self.callback, self.collector = callback, collector
        self.root, self.lock = Path(root), manifest_lock
        self.last_checkpoint = None
        self.last_attempted_checkpoint = None

    def __call__(self, **event):
        checkpoint = event.get("last_checkpoint")
        if checkpoint is not None and str(checkpoint) != self.last_attempted_checkpoint:
            self.last_attempted_checkpoint = str(checkpoint)
            if not callable(getattr(self.collector, "save_resume", None)):
                raise ValueError("checkpointed ensemble requires a durable diagnostic collector")
            # Member files are write-once. Do their first full reads before
            # taking the shared manifest lock; a waiting seal may reuse them.
            bind_checkpoint_member(self.root, checkpoint)
            with self.lock:
                try:
                    # The collector snapshot is taken inside seal, under the
                    # collector checkpoint hold, so members on other cards
                    # cannot retire the packs it names before they are hashed.
                    seal(self.root, collector=self.collector)
                except TimeoutError:
                    # A deferred snapshot leaves this checkpoint unbound, as
                    # before; the previous roster stays the durable one.
                    Path(checkpoint).with_suffix(".member.json").unlink(missing_ok=True)
                    event["ensemble_checkpoint_deferred"] = "product drain exceeded bounded checkpoint hold"
                    return self.callback(**event)
                self.last_checkpoint = str(checkpoint)
            event["ensemble_restart_roster"] = str(self.root / ROSTER)
        return self.callback(**event)

    def __getattr__(self, name):
        return getattr(self.callback, name)


def checkpoint_hold(collector):
    """Hold every member submission and pack retirement while a roster seals.

    Members on other cards keep submitting diagnostics while one member's
    checkpoint seals. Without this hold a submission could complete a frame
    and retire the pending packs that the just-saved collector.json names.
    """
    hold = getattr(collector, "checkpoint_hold", None)
    return hold() if callable(hold) else nullcontext()


def bind_checkpoint_member(root, checkpoint):
    """Stamp the original runner's member owner after its complete set commits."""
    from woof.io.restart import read_restart_header, tree_restart_members
    root, checkpoint = Path(root), Path(checkpoint)
    from woof.supervisor import CheckpointValidationError
    owner = get_checkpoint_member(checkpoint)
    if owner is None:
        raise ValueError("member checkpoint has no bound ensemble identity")
    header = read_restart_header(checkpoint)
    paths = tree_restart_members(checkpoint) if header.get("domain_ids") else {1: checkpoint}
    def inspect(path):
        # Bind ownership as before; seal still applies its original refusal
        # or candidate fallback if validation fails. Only passes are memoized.
        try:
            _validate_checkpoint(path)
        except (ValueError, KeyError, OSError, CheckpointValidationError):
            pass
        return {"path": relative_file(root, path), "sha256": _seal_digest(path),
                "bytes": path.stat().st_size}

    inspected = _parallel_file_map(inspect, paths.values())
    record = {"schema": CONTRACT, "member_id": owner["memberId"], "identity_sha256": owner["memberIdentity"],
        **({"provenance": owner["provenance"]} if owner.get("provenance") else {}),
        "files": inspected}
    atomic_json(checkpoint.with_suffix(".member.json"), record)
    return record


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(record, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    pending.replace(path)


def relative_file(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    return path.relative_to(root).as_posix()


def resolve_file(root, name):
    path = (Path(root) / name).resolve()
    path.relative_to(Path(root).resolve())
    return path


def identity_record(request, member_order, member_inputs, seeds):
    from woof.core.model import restart_identity_payload
    selected = dict(request.receipt())
    # A new box may have a different card count or allocator admission.
    selected.pop("member_device_ids", None)
    selected.pop("max_ordinary_members_per_device", None)
    rows = []
    for member in member_order:
        inputs = member_inputs[member]
        payload = {"member_id": member, "seed": seeds[member],
            "experiment": restart_identity_payload(inputs.experiment),
            "source": getattr(inputs, "source", None),
            "prepared_head_sha256": getattr(inputs, "prepared_head_sha256", None),
            # V1 checkpoint hashes predate per-row provenance. Their request
            # already retains the original labels, so keep those hashes stable.
            **({"provenance": request.provenance[member]} if request.provenance
               and request.provenance[member].get("schema") == "gpuwm-member-provenance.v2" else {})}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        rows.append({**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()})
    return {"schema": CONTRACT, "request": selected, "member_order": list(member_order), "members": rows}


def _matches(path, item):
    return path.is_file() and path.stat().st_size == item["bytes"] and digest_file(path) == item["sha256"]


def _parallel_file_map(function, items):
    from woof.ingest.cpu_backend import available_cpu_count
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(len(items), available_cpu_count()),
                            thread_name_prefix="ensemble-seal") as workers:
        return list(workers.map(function, items))


def verify_roster(path, identities, *, root):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != CONTRACT or record.get("identities") != identities:
        raise ValueError("ensemble restart roster has different member identities, seeds, forcing or physics")
    rows = record.get("members", [])
    if [row.get("member_id") for row in rows] != identities["member_order"]:
        raise ValueError("ensemble restart roster must name every original member exactly once in order")
    files = record.get("files", [])
    matches = _parallel_file_map(lambda item: _matches(
        resolve_file(root, item.get("snapshot", item["path"])), item), files)
    for item, matches_file in zip(files, matches):
        if "snapshot" in item:
            # The sealed copy is the authority; the live file may have moved on.
            if not matches_file:
                raise ValueError(
                    f"ensemble resume file is missing or hash-mismatched: {item['path']} (its sealed copy "
                    f"{item['snapshot']} is missing or changed). Copy the whole run directory, including "
                    ".ensemble-resume, or run the ensemble again into a new --outdir")
        elif not matches_file:
            # A roster sealed before sealed copies existed (gpuwm 2.9.0 and
            # earlier) names only live files that later frames rewrite.
            raise ValueError(
                f"ensemble resume file is missing or hash-mismatched: {item['path']}. This roster keeps no "
                "sealed copy of its files (sealed by gpuwm 2.9.0 or earlier), and the run changed that file "
                "after the seal; run the ensemble again into a new --outdir")
    for row, identity in zip(rows, identities["members"]):
        if row.get("provenance") != identity.get("provenance"):
            raise ValueError("ensemble checkpoint lineage differs from its original member identity")
        if row.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("ensemble checkpoint was assigned to another member")
        if row.get("status") not in ("completed", "checkpoint", "pending"):
            raise ValueError("ensemble restart member has an unknown completion status")
        if row["status"] == "checkpoint":
            prefix = f"members/member-{row['member_id']:04d}/"
            if not row.get("checkpoint", "").startswith(prefix):
                raise ValueError("ensemble checkpoint path belongs to another member")
            resolve_file(root, row["checkpoint"])
        if row["status"] == "completed" and not isinstance(row.get("result"), dict):
            raise ValueError("completed ensemble member has no retained forecast result")
    return record


def _place(source, destination, *, link):
    """Atomically put ``source``'s bytes at ``destination``.

    ``link`` shares the inode when the filesystem allows it (checkpoint sets
    and other write-once files), so a seal or restore costs no copy; JSON
    authorities are always copied, because only a copy is immune to a writer
    that rewrites its file in place.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending")
    pending.unlink(missing_ok=True)
    if link:
        try:
            os.link(source, pending)
        except OSError as error:
            if isinstance(error, FileNotFoundError):
                raise
            shutil.copyfile(source, pending)
    else:
        shutil.copyfile(source, pending)
    pending.replace(destination)
    return destination


def _source_revision(path):
    """A copy may be reused only while its live source has not changed.

    Device, inode, size and mtime name a content change. ctime is left out:
    a reader that hard-links a member's restart files changes their ctime
    without changing a byte, and counting it made every seal repeat its
    copy and then fail the member ("ensemble snapshot source changed during
    copy", WoFS run_8dd42875d520e1d3, 2026-10-26).
    """
    status = path.stat()
    return [status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns]


# Process-local proofs for write-once checkpoints and sealed copies only.
# A Future shares an in-flight read without holding the cache lock over I/O.
# Resume's _matches deliberately bypasses this memo and always re-hashes.
_FILE_MEMO = {}
_FILE_MEMO_LOCK = Lock()


def _memo_file(path, operation, function):
    path = Path(path)
    revision = tuple(_source_revision(path))
    key = (operation, revision)
    with _FILE_MEMO_LOCK:
        future = _FILE_MEMO.get(key)
        owner = future is None
        if owner:
            future = _FILE_MEMO[key] = Future()
    if owner:
        try:
            value = function(path)
            if tuple(_source_revision(path)) != revision:
                raise FileNotFoundError(f"ensemble memo source changed during read: {path}")
        except BaseException as error:
            future.set_exception(error)
            with _FILE_MEMO_LOCK:
                _FILE_MEMO.pop(key, None)
            raise
        else:
            future.set_result(value)
    value = future.result()
    if tuple(_source_revision(path)) != revision:
        raise FileNotFoundError(f"ensemble memo source changed during read: {path}")
    return value


def _seal_digest(path):
    return _memo_file(path, "sha256", digest_file)


def _validate_checkpoint(path):
    from woof.supervisor import validate_manifest_checkpoint
    _memo_file(path, "manifest", validate_manifest_checkpoint)


def _checkpoint_proof(path):
    revision = _source_revision(path)
    _validate_checkpoint(path)
    digest = _seal_digest(path)
    if _source_revision(path) != revision:
        raise FileNotFoundError("ensemble checkpoint changed during validation")
    return path, digest


#: The collector's private trees. The sealed snapshot names every file the
#: collector owns in them, so anything else there was written after the seal.
COLLECTOR_PRIVATE = (".ensemble-diagnostics", "member-diagnostics")


def restore_roster(record, *, root):
    """Put the sealed copies back before the collector and members restart.

    Files the run rewrote or retired after the seal return to their sealed
    bytes. What the run wrote after the seal and the restored state does not
    own is removed, because the resumed members write it again:
    - checkpoint sets an unfinished member committed after the seal (a later
      seal must not name an hour the restored collector has not seen);
    - diagnostic packs, render sources and archive files in the collector's
      private trees (their writers refuse to replace an unowned file:
      "diagnostic scratch path already exists", 2.9.1 audit).
    """
    root = Path(root).resolve()
    named, restored, removed = set(), [], []
    files = record.get("files", ())
    replace_live = _parallel_file_map(lambda item: "snapshot" in item and not _matches(
        resolve_file(root, item["path"]), item), files)
    for item, needs_restore in zip(files, replace_live):
        live = resolve_file(root, item["path"])
        named.add(live)
        if needs_restore:
            _place(resolve_file(root, item["snapshot"]), live, link=not live.name.endswith(".json"))
            restored.append(item["path"])
    if record.get("snapshot_generation"):
        # A roster sealed with copies names every collector-owned file.
        for tree in COLLECTOR_PRIVATE:
            directory = root / tree
            for path in sorted(directory.rglob("*")) if directory.is_dir() else ():
                if (path.is_file() or path.is_symlink()) and path.resolve() not in named:
                    path.unlink(missing_ok=True)
                    removed.append(relative_file(root, path))
    for row in record.get("members", ()):
        if row.get("status") == "completed":
            continue
        directory = root / "members" / f"member-{row['member_id']:04d}"
        for path in sorted(directory.rglob("gpuwmrst_d*.npz")) if directory.is_dir() else ():
            if path.resolve() not in named:
                path.unlink(missing_ok=True)
                path.with_suffix(".member.json").unlink(missing_ok=True)
                removed.append(relative_file(root, path))
    return {"restored": restored, "removed_after_seal": removed}


def get_checkpoint_member(path):
    """Return the original member owner and prepared handoff for transport."""
    path = Path(path).resolve()
    root = next((parent for parent in path.parents if (parent / IDENTITIES).is_file()), None)
    if root is None:
        return None
    identities = json.loads((root / IDENTITIES).read_text(encoding="utf-8"))
    relative = relative_file(root, path)
    identity = next((row for row in identities["members"] if
        relative.startswith(f"members/member-{row['member_id']:04d}/")), None)
    if identity is None:
        raise ValueError("checkpoint is outside every original ensemble member directory")
    descriptors = json.loads((root / INPUTS).read_text(encoding="utf-8"))
    descriptor = next(row for row in descriptors["members"] if row["member_id"] == identity["member_id"])
    collector = root / ".ensemble-resume" / "collector.json"
    files = []
    if collector.is_file():
        document = json.loads(collector.read_text(encoding="utf-8"))
        files = [collector, *(resolve_file(root, name) for name in document.get("files", ()))]
    return {"memberIndex": identities["member_order"].index(identity["member_id"]),
        "memberId": identity["member_id"], "memberIdentity": identity["identity_sha256"],
        **({"provenance": identity["provenance"]} if identity.get("provenance") else {}),
        "config": descriptor.get("experiment_config"), "wps": descriptor.get("wps_namelist"),
        "prepared": descriptor.get("prepared_root"), "root": str(root),
        "roster": str(root / ROSTER), "collectorFiles": [str(item) for item in files],
        "memberInputs": descriptors["members"]}


def resume_recipe_request(request, identities):
    """Restore derived recipe labels after checking the declared request.

    Acquisition creates provenance and resolved family weights. The resume
    door reuses sealed inputs instead of reacquiring them, so it must carry
    that same metadata into production's full member identity validation.
    """
    from dataclasses import replace

    declared, stored = dict(request.receipt()), dict(identities["request"])
    for key in ("member_device_ids", "max_ordinary_members_per_device"):
        declared.pop(key, None)
        stored.pop(key, None)
    provenance = stored.pop("provenance", ())
    weights = stored.pop("family_weights", None)
    supplied_provenance = declared.pop("provenance", ())
    supplied_weights = declared.pop("family_weights", None)
    canonical = lambda value: json.dumps(value, sort_keys=True, default=str)
    if (canonical(declared) != canonical(stored)
            or (supplied_provenance and canonical(supplied_provenance) != canonical(provenance))
            or (supplied_weights is not None and canonical(supplied_weights) != canonical(weights))):
        raise ValueError("ensemble restart roster has a different declared recipe request")
    return replace(request, provenance=tuple(provenance), family_weights=weights)


def resume_prepared_roster(args, request, *, observer=None):
    """Run the fetched member handoffs through the ordinary native front door."""
    from woof import stage_cli
    from woof.ensemble.door import production_run_scope
    from woof.ensemble.recipe_door import member_inputs
    if getattr(args, "restart", None) is not None:
        raise ValueError("--restart-roster cannot be combined with one member's --restart")
    roster = Path(args.restart_roster).resolve()
    root = roster.parent
    if getattr(args, "outdir", None) is not None and Path(args.outdir).resolve() != root:
        raise ValueError("ensemble continuation --outdir must be the fetched restart roster directory")
    document = json.loads((root / INPUTS).read_text(encoding="utf-8"))
    prepared = {row["member_id"]: row for row in document["members"]}
    identities = json.loads((root / IDENTITIES).read_text(encoding="utf-8"))
    verify_roster(roster, identities, root=root)
    request = resume_recipe_request(request, identities)
    first = identities["member_order"][0]
    bundle = stage_cli.resolve_bundle(Path(prepared[first]["prepared_root"]))
    # Resume must retain the recipe's sharing of immutable prepared banks.
    # Reopening one bank per member retains thousands of duplicate mmap
    # regions and can exhaust vm.max_map_count before forecasting starts.
    # Configuration and WPS bindings are part of the key: a shared directory
    # alone does not establish matching physics or preparation arguments.
    def binding(row):
        return tuple(str(Path(row[name]).resolve()) if row.get(name) else None
                     for name in ("prepared_root", "experiment_config", "wps_namelist"))

    banks = {}

    def provider(*, shared_inputs, member_id, **unused):
        banks.setdefault(binding(prepared[first]), shared_inputs)
        key = binding(prepared[member_id])
        if key not in banks:
            banks[key] = member_inputs(shared_inputs, prepared[member_id])
        return banks[key]
    sim = stage_cli.sim_command(bundle, experiment_config=Path(prepared[first]["experiment_config"]),
        wps_namelist=(Path(prepared[first]["wps_namelist"]) if prepared[first]["wps_namelist"] else None),
        outdir=root, physics_profile=None, progress_format="jsonl", render_products=None,
        **({"devices": args.devices} if getattr(args, "devices", None) is not None else {}),
        memory_gate=not bool(getattr(args, "no_memory_gate", False)))
    if bundle["layout"] == "tree":
        from woof import prepared_domain_tree_forecast as runner
    else:
        from woof import prepared_single_domain_forecast as runner
    with production_run_scope(request, output_directory=root, input_provider=provider, restart_roster=roster):
        return runner.main(sim[3:], observer=observer)


#: Whole seal passes tried when a file vanishes mid-pass. Another member's
#: runner, on another card, retires its superseded checkpoint sets
#: (retire_superseded_checkpoints) while this member's roster is being hashed.
SEAL_ATTEMPTS = 8


def seal(root, output=None, *, collector_files=(), collector=None):
    """Publish a roster only after its member checkpoints and output spool exist.

    With ``collector``, its snapshot is saved under :func:`checkpoint_hold`
    and the files it names are hashed before the hold is released. A pass in
    which any listed file disappears is repeated, so a published roster never
    names a file that was already gone when it was hashed.
    """
    for attempt in range(SEAL_ATTEMPTS):
        try:
            return _seal_once(root, output, collector_files=collector_files, collector=collector)
        except FileNotFoundError:
            if attempt == SEAL_ATTEMPTS - 1:
                raise


def _seal_once(root, output, *, collector_files, collector):
    from woof.io.restart import read_restart_header, tree_restart_members
    from woof.resume import discover_checkpoint_sets
    root = Path(root).resolve()
    identities = json.loads((root / IDENTITIES).read_text(encoding="utf-8"))
    manifest_path = root / "ensemble-run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = set(manifest.get("members_completed", ()))
    results = {row["member_id"]: row["result"] for row in manifest.get("completed_member_results", ())}
    files = {root / IDENTITIES, manifest_path}
    if (root / INPUTS).is_file():
        files.add(root / INPUTS)
    files.update(resolve_file(root, name) for name in collector_files)

    def inspect_member(identity):
        member_files, member_hashes = set(), {}
        member = identity["member_id"]
        row = {"member_id": member, **({"provenance": identity["provenance"]} if identity.get("provenance") else {}), "identity_sha256": identity["identity_sha256"], "status": "pending",
               "model_elapsed_seconds": 0.0}
        if member in completed:
            if member not in results:
                raise ValueError(f"completed member {member} has no durable forecast receipt")
            row.update(status="completed", result=results[member],
                       model_elapsed_seconds=float(manifest.get("requested_run_seconds", 0)))
        else:
            candidates = []
            directory = root / "members" / f"member-{member:04d}"
            for parent in {path.parent for path in directory.rglob("gpuwmrst_d*.npz")}:
                candidates.extend(discover_checkpoint_sets(parent))
            candidates.sort(key=lambda item: (item.valid_time, item.handle.stat().st_mtime_ns), reverse=True)
            for candidate in candidates:
                try:
                    header = read_restart_header(candidate.handle)
                    members = (tree_restart_members(candidate.handle) if header.get("domain_ids") else candidate.members)
                    proofs = _parallel_file_map(_checkpoint_proof, members.values())
                    elapsed = float(header["elapsed_seconds"])
                    binding_path = candidate.handle.with_suffix(".member.json")
                    binding = json.loads(binding_path.read_text(encoding="utf-8"))
                    if (binding.get("member_id") != member
                            or binding.get("identity_sha256") != identity["identity_sha256"]):
                        raise ValueError("checkpoint sidecar belongs to another ensemble member")
                    known = {relative_file(root, path): digest for path, digest in proofs}
                    if known != {row["path"]: row["sha256"] for row in binding["files"]}:
                        raise ValueError("member checkpoint files differ from their committed owner binding")
                except (ValueError, KeyError, OSError):
                    continue
                member_files.update(members.values())
                member_files.add(binding_path)
                member_hashes.update({path: known[relative_file(root, path)] for path in members.values()})
                row.update(status="checkpoint", checkpoint=relative_file(root, candidate.handle),
                           model_elapsed_seconds=elapsed)
                break
        return row, member_files, member_hashes

    # Discovery, headers, manifest validation and hashes all run on the
    # allowed CPU pool. No serial full-file reader sits under the lock.
    rows, hashed = [], {}
    for row, member_files, member_hashes in _parallel_file_map(inspect_member, identities["members"]):
        rows.append(row)
        files.update(member_files)
        hashed.update(member_hashes)
    with checkpoint_hold(collector):
        if collector is not None:
            collector.save_resume()
        collector_manifest = root / ".ensemble-resume" / "collector.json"
        if collector_manifest.is_file():
            document = json.loads(collector_manifest.read_text(encoding="utf-8"))
            files.add(collector_manifest)
            files.update(resolve_file(root, name) for name in document.get("files", ()))
        # Every file is copied (or hard-linked) into this seal's generation
        # and the roster names those sealed copies, so the frames, ledger
        # writes and checkpoint retirements that follow the seal cannot
        # invalidate it. A member checkpoint retired during this pass raises
        # FileNotFoundError and the pass repeats.
        created = time.time_ns()
        generation = root / GENERATIONS / f"{created:020d}"
        previous_path = Path(output or root / ROSTER)
        previous = (json.loads(previous_path.read_text(encoding="utf-8"))
                    if previous_path.is_file() else {})
        previous_files = {item["path"]: item for item in previous.get("files", ())}
        try:
            def snapshot_file(path):
                name = relative_file(root, path)
                revision = _source_revision(path)
                old = previous_files.get(name, {})
                retained = (resolve_file(root, old["snapshot"])
                            if old.get("source_revision") == revision and "snapshot" in old else None)
                if retained is not None and retained.is_file():
                    # Link between sealed copies. Linking a live OUTPUT
                    # changes its ctime while another member's writer hashes
                    # it (four-member concurrent WoFS run, 2026-10-17).
                    copy = _place(retained, generation / name, link=True)
                    digest = _seal_digest(copy)
                    if digest != old["sha256"]:
                        raise ValueError(f"ensemble sealed copy changed: {name}")
                elif path in hashed:
                    # A member checkpoint set is write-once (published by
                    # os.replace, retired by unlink, never rewritten in
                    # place) and was just hashed against its committed owner
                    # binding, so the seal links it instead of copying it.
                    # The copy cost 44-59 GB of host disk per sealed set
                    # (U-E10). No checkpoint reader counts ctime (E9).
                    copy = _place(path, generation / name, link=True)
                    digest = hashed[path]
                else:
                    copy = _place(path, generation / name, link=False)
                    digest = _seal_digest(copy)
                if _source_revision(path) != revision:
                    raise FileNotFoundError(f"ensemble snapshot source changed during copy: {name}")
                return {"path": name, "bytes": copy.stat().st_size,
                    "sha256": digest, "source_revision": revision,
                    "snapshot": relative_file(root, copy)}

            # A retained-history roster can name tens of GB. Hash independent
            # files across the allowed host CPUs instead of stalling the GPU
            # on one core at every checkpoint. hashlib releases the GIL for
            # these blocks. Keep every hash, sorted publication, the collector
            # hold and the whole-pass retirement retry unchanged. Joining the
            # pool before cleanup also prevents failed workers recreating it.
            entries = _parallel_file_map(snapshot_file, sorted(files))
            missing = [path for path in hashed if not path.is_file()]
            if missing:
                raise FileNotFoundError(str(missing[0]))
        except BaseException:
            shutil.rmtree(generation, ignore_errors=True)
            raise
        record = {"schema": CONTRACT, "created_time_ns": created, "identities": identities, "members": rows,
            "snapshot_generation": relative_file(root, generation), "files": entries}
        atomic_json(output or root / ROSTER, record)
        if output is None or Path(output).resolve() == root / ROSTER:
            # Older generations are no longer named by the durable roster.
            for older in (root / GENERATIONS).iterdir() if (root / GENERATIONS).is_dir() else ():
                if older.name < generation.name:
                    shutil.rmtree(older, ignore_errors=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    seal(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
