"""``woof branch`` -- seed a NEW run from an existing checkpoint.

``woof resume`` continues the run that wrote the checkpoint: same
config, same output directory, one trajectory.  A what-if is the other
question -- *what would this forecast have done from hour N with the
tracker widened / a trimmed history tape?* -- and answering it means a
SECOND run that starts from the first run's state and writes somewhere
else entirely.

A subcommand rather than ``resume --branch`` because the two doors
disagree about the one word they share: ``resume --outdir`` names the
directory being CONTINUED, and a branch's ``--outdir`` has to name the
directory being CREATED (as it does on ``run``, ``go`` and every other
door in this CLI).  Folding them would have made ``--outdir`` mean the
source in one mode and the destination in the other, which is the kind
of flag nobody can read twice the same way.  ``--from-run`` names the
source instead, and checkpoint LOCATION is shared code
(:func:`woof.resume.resolve_resume_checkpoint`), not a second
implementation.

**What may change, and who decides.**  Not this module.  The restart
contract already publishes the split -- ``woof.core.model``'s
``RESTART_TOLERATED_*`` tables are what ``restart_identity_payload``
strips before the fingerprint is taken, i.e. exactly the settings a
checkpoint does not bind -- so the changeable set here is DERIVED from
those tables.  A branch that let anything else through would not be
refused by this door; it would be refused by the restart guard at
restore time, after the fetch, the static build and the device
allocation.  Refusing here, by name, with the reason, is the same
refusal moved to where it costs nothing.

The three things the door guarantees, in order:

1. every setting is classified before anything is created, so a refused
   branch leaves no directory behind (the poisoned-``--out`` shape);
2. the branched config's restart identity payload is compared against
   the parent's and must be byte-identical -- the name table is the
   fast path, this is the proof;
3. the source run is opened read-only.  Nothing is written outside the
   new run directory, which must be empty and must not be inside the
   run being branched from.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from woof.core.model import (RESTART_TOLERATED_DOMAIN_FIELDS,
                              RESTART_TOLERATED_EXPERIMENT_FIELDS,
                              RESTART_TOLERATED_RUN_FIELDS)

#: The branched configuration, written into the new run directory.  It is
#: that run's authority: `woof branch` dispatches through the ordinary
#: run path with this file as the config, so what the receipt describes
#: and what integrates are the same bytes.
BRANCHED_CONFIG_NAME = "branch-config.toml"

#: The DECISION record: which checkpoint, which settings moved and from
#: what, and the identity comparison that says the branch is legal.
BRANCH_RECEIPT_NAME = "branch_receipt.json"
BRANCH_RECEIPT_SCHEMA = "gpuwm-branch-receipt/v1"

#: The BYTES record: every member of the parent checkpoint set this run
#: was seeded from, sized and hashed.  Separate from the receipt because
#: one answers "what did we decide" and the other "what did we read".
BRANCH_MANIFEST_NAME = "branch_manifest.json"
BRANCH_MANIFEST_SCHEMA = "gpuwm-branch-manifest/v1"


# ---------------------------------------------------------------------------
# Where each restart-tolerated setting is spelled in the TOML
# ---------------------------------------------------------------------------

#: The POLICY -- what a checkpoint does not bind -- is
#: ``woof.core.model``'s tolerance tables.  These two maps say only
#: WHERE each tolerated name lives in an experiment TOML, which the
#: tolerance tables cannot say because they are keyed by the parsed
#: config's field names.  The import-time checks below refuse to load if
#: the two ever disagree: a tolerated setting with no spelling here is a
#: knob the what-if screen would silently drop.
_EXPERIMENT_FIELD_PATH: dict[str, tuple[str, ...]] = {
    "run_seconds": ("experiment", "run_seconds"),
    "restart_interval_s": ("experiment", "restart_interval_s"),
    "acknowledgements": ("experiment", "acknowledgements"),
    # Whole top-level tables.  A branch may edit any leaf inside them.
    "relocation": ("relocation",),
    "tiles": ("tiles",),
    "devices": ("devices",),
    "output": ("output",),
    "simulated_radar": ("simulated_radar",),
}

#: The per-domain half.  ``output_interval_s`` in the tolerance table is
#: the PARSED field; a ``[[domain]]`` table spells it
#: ``history_interval_s`` (see ``_SHARED_FORBIDDEN`` in
#: :mod:`woof.experiment`), and per-domain ``run_seconds`` /
#: ``restart_interval_s`` are not authorable at all -- they are
#: experiment-scope, reachable through the map above.
_DOMAIN_FIELD_SPELLING: dict[str, str] = {
    "history_interval_s": "history_interval_s",
    "history_begin_s": "history_begin_s",
    "history_end_s": "history_end_s",
    "tiles": "tiles",
    "output": "output",
    "surface_energy_diag": "surface_energy_diag",
}
_RUN_FIELD_SPELLING: dict[str, str] = {
    "run_seconds": "run_seconds",
    "restart_interval_s": "restart_interval_s",
    "output_interval_s": "domain.<grid_id>.history_interval_s",
    "history_interval2_s": "domain.<grid_id>.history_interval2_s",
    "history_interval_change_s": "domain.<grid_id>.history_interval_change_s",
    "surface_energy_diag": "domain.<grid_id>.surface_energy_diag",
    # Correct mode WB1: output-only, so a branch may switch it (books CLOUDNC).
    "thompson_cloud_surface_booking": "domain.<grid_id>.thompson_cloud_surface_booking",
}

_unspelled = (set(RESTART_TOLERATED_EXPERIMENT_FIELDS)
              - set(_EXPERIMENT_FIELD_PATH))
if _unspelled:
    raise RuntimeError(
        "woof.branch has no TOML spelling for the restart-tolerated "
        f"experiment field(s) {sorted(_unspelled)}; a branch could not "
        "reach a setting the restart contract says it may change")
_unspelled = set(RESTART_TOLERATED_DOMAIN_FIELDS) - set(
    _DOMAIN_FIELD_SPELLING)
if _unspelled:
    raise RuntimeError(
        "woof.branch has no TOML spelling for the restart-tolerated "
        f"domain field(s) {sorted(_unspelled)}")
_unspelled = set(RESTART_TOLERATED_RUN_FIELDS) - set(_RUN_FIELD_SPELLING)
if _unspelled:
    raise RuntimeError(
        "woof.branch has no TOML spelling for the restart-tolerated "
        f"per-domain run field(s) {sorted(_unspelled)}")

#: The sentence the refusals and ``--help`` both print.  One list, so the
#: door and its documentation cannot drift.
CHANGEABLE_SETTINGS = (
    "run_seconds, restart_interval_s, acknowledgements, relocation.*, "
    "tiles.*, devices.*, output.*, simulated_radar.*, domain.<grid_id>.history_interval_s, "
    "domain.<grid_id>.history_begin_s, domain.<grid_id>.history_end_s, "
    "domain.<grid_id>.tiles.*, domain.<grid_id>.output.*, "
    "domain.<grid_id>.surface_energy_diag")

#: Tables an experiment TOML knows about.  A head that is not one of
#: these and not an ``[experiment]`` key is a TYPO, and a typo must be
#: refused as a typo -- reporting it as "pinned" sends the reader
#: looking for a rule that does not exist.
_KNOWN_TABLES = ("experiment", "shared", "projection", "domain",
                 "relocation", "perturbation", "tiles", "devices", "output", "simulated_radar",
                 "fetch", "case_data", "static", "ingest", "physics_params")

#: Tables whose string values name files on disk, resolved by
#: :func:`woof.experiment.load_experiment` against the CONFIG FILE's own
#: directory.  A branched config lives in the new run folder, so every
#: relative declaration in these tables is rebased to the source config's
#: directory on the way out -- otherwise a branch of a working config
#: points at nothing and the run dies in the fetch stage.
_PATH_TABLES = ("case_data", "static", "ingest")


# ---------------------------------------------------------------------------
# Why each pinned setting is pinned
# ---------------------------------------------------------------------------

_LIFECYCLE_REASON = (
    "[follow], [retire], [rearm] and [spawn] on a [[domain]] decide when "
    "a child integrates at all and where it sits, so they bind value for "
    "value: a branch that changed one would restore state computed under "
    "a different nest history.  The tracker BOUNDS are the branchable "
    "half -- [relocation], [relocation.follow], [relocation.containment] "
    "and [relocation.track] are what a what-if edits")
_GEOMETRY_REASON = (
    "domain geometry fixes the shape of every array the checkpoint "
    "stores, so restoring that state onto a different grid is not a "
    "restart -- the array manifest itself would not match")
_CLOCK_REASON = (
    "the timestep decides which model instant the checkpoint IS, so a "
    "branch that changed it would resume the state at a time it was "
    "never integrated to")
_PHYSICS_REASON = (
    "the checkpoint stores state the configured physics wrote; under a "
    "different scheme the restore would carry fields the new scheme "
    "never produces and miss the ones it needs")
_COLUMN_REASON = (
    "the vertical column and the projection fix the levels and the "
    "ground the checkpoint's arrays sit on")
_INPUT_REASON = (
    "the declared inputs are hashed into the run's identity "
    "(input_catalog_sha256), so a branch pointing at other bytes would "
    "restore a state that was integrated from forcing it no longer has")
_PERTURBATION_REASON = (
    "[perturbation] is applied once, at t=0, and is already integrated "
    "into the checkpoint: changing it cannot retroactively change the "
    "state a branch restores")
_IDENTITY_REASON = (
    "it is part of what the run IS, not of how far it runs or what it "
    "writes down")

_GEOMETRY_FIELDS = frozenset({
    "nx", "ny", "e_we", "e_sn", "dx", "dy", "i_parent_start",
    "j_parent_start", "parent_grid_ratio", "parent_id", "grid_id"})
_CLOCK_FIELDS = frozenset({
    "time_step", "time_step_fract_num", "time_step_fract_den",
    "parent_time_step_ratio", "start_time", "dt"})
_LIFECYCLE_FIELDS = frozenset({"follow", "retire", "rearm", "spawn"})


def _reason_for(path: Sequence[str]) -> str:
    """Name the concrete breakage this particular pin prevents."""

    head = path[0]
    if head == "domain":
        field = path[2] if len(path) > 2 else ""
        if field in _LIFECYCLE_FIELDS:
            return _LIFECYCLE_REASON
        if field in _GEOMETRY_FIELDS:
            return _GEOMETRY_REASON
        if field in _CLOCK_FIELDS:
            return _CLOCK_REASON
        if field == "run":
            leaf = path[3] if len(path) > 3 else ""
            if leaf in _CLOCK_FIELDS:
                return _CLOCK_REASON
            if leaf in _GEOMETRY_FIELDS:
                return _GEOMETRY_REASON
            return _PHYSICS_REASON
        return _IDENTITY_REASON
    if head in ("shared", "projection"):
        return _COLUMN_REASON
    if head == "perturbation":
        return _PERTURBATION_REASON
    if head in ("case_data", "static", "ingest", "fetch"):
        return _INPUT_REASON
    if head in _CLOCK_FIELDS:
        return _CLOCK_REASON
    return _IDENTITY_REASON


def _pinned_refusal(spelling: str, path: Sequence[str]) -> ValueError:
    return ValueError(
        f"`{spelling}` is pinned by the checkpoint's restart identity "
        "and cannot be changed by a branch: "
        f"{_reason_for(path)}.\n"
        "  the guard: woof.core.model.restart_identity_payload binds it "
        "into the fingerprint woof.io.restart compares at restore, so a "
        "branch that changed it would be refused there -- after the "
        "fetch, the static build and the device allocation.\n"
        f"  changeable from a checkpoint: {CHANGEABLE_SETTINGS}\n"
        f"  remedy: to change {spelling}, run the experiment from its "
        "start instead -- `woof run <config> --outdir <new dir>` -- "
        "which integrates the new setting from t=0 rather than "
        "pretending the checkpoint was written under it.")


# ---------------------------------------------------------------------------
# Setting grammar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Override:
    """One accepted ``--set`` edit, resolved to a place in the TOML."""

    spelling: str                 # what the caller typed, left of '='
    path: tuple[str, ...]         # the caller's dotted key path
    target: tuple[str | int, ...]  # where it lands in the raw document
    value: Any


def _parse_value(text: str) -> Any:
    """The right-hand side as a TOML value, or as a plain string.

    TOML first so ``7200.0``, ``true``, ``1974-04-03T18:00:00`` and
    ``["a", "b"]`` arrive as themselves rather than as text.  A bare word
    like ``minimal`` is not a TOML value, and quoting it through a shell
    twice is a papercut nobody should pay for a preset name, so an
    unparseable right-hand side is the string the caller typed.
    """

    try:
        return tomllib.loads(f"value = {text}")["value"]
    except (tomllib.TOMLDecodeError, ValueError):
        return text


def _domain_index(raw: Mapping[str, Any], spelling: str,
                  token: str) -> int:
    domains = raw.get("domain")
    if not isinstance(domains, list):
        raise ValueError(
            f"`{spelling}` addresses a [[domain]] table, but "
            "the configuration declares none")
    try:
        grid_id = int(token)
    except ValueError:
        raise ValueError(
            f"`{spelling}` must address a domain by its grid_id "
            f"(`domain.<grid_id>....`); {token!r} is not a number"
        ) from None
    for index, domain in enumerate(domains):
        if isinstance(domain, Mapping) and domain.get("grid_id") == grid_id:
            return index
    declared = sorted(
        int(d["grid_id"]) for d in domains
        if isinstance(d, Mapping) and isinstance(d.get("grid_id"), int))
    raise ValueError(
        f"`{spelling}` names grid_id = {grid_id}, which this "
        f"configuration does not declare; it has {declared}")


def parse_setting(text: str, raw: Mapping[str, Any]) -> Override:
    """One ``KEY=VALUE`` setting, classified against the restart contract.

    Raises the pinned refusal (by name, with the reason) or the typo
    refusal; returns the resolved edit otherwise.
    """

    if "=" not in text:
        raise ValueError(
            f"`{text}` is not a setting: --set takes KEY=VALUE, for "
            f"example --set run_seconds=7200.  Changeable keys: "
            f"{CHANGEABLE_SETTINGS}")
    spelling, _, literal = text.partition("=")
    spelling = spelling.strip()
    path = tuple(part for part in spelling.split("."))
    if not spelling or any(part == "" for part in path):
        raise ValueError(
            f"`{text}` has an empty key segment; --set takes a dotted "
            "TOML key, for example --set relocation.follow.threshold=40")
    value = _parse_value(literal.strip())
    head = path[0]

    if head == "domain":
        if len(path) < 3:
            raise ValueError(
                f"`{spelling}` is incomplete: a per-domain setting is "
                "spelled domain.<grid_id>.<key>, for example "
                "domain.2.history_interval_s=1800")
        index = _domain_index(raw, spelling, path[1])
        field = path[2]
        if field not in _DOMAIN_FIELD_SPELLING:
            raise _pinned_refusal(spelling, path)
        target: tuple[str | int, ...] = (
            "domain", index, _DOMAIN_FIELD_SPELLING[field], *path[3:])
        return Override(spelling=spelling, path=path, target=target,
                        value=value)

    if head in _EXPERIMENT_FIELD_PATH:
        target = (*_EXPERIMENT_FIELD_PATH[head], *path[1:])
        return Override(spelling=spelling, path=path, target=target,
                        value=value)

    from woof.experiment import _EXPERIMENT_KEYS, did_you_mean

    if head in _KNOWN_TABLES or head in _EXPERIMENT_KEYS:
        raise _pinned_refusal(spelling, path)
    known = sorted(set(_KNOWN_TABLES) | set(_EXPERIMENT_KEYS)
                   | set(_EXPERIMENT_FIELD_PATH) | {"domain"})
    raise ValueError(
        f"`{head}` is not a setting this configuration has"
        f"{did_you_mean(head, known)}.  Changeable from a checkpoint: "
        f"{CHANGEABLE_SETTINGS}")


# ---------------------------------------------------------------------------
# Editing the raw document
# ---------------------------------------------------------------------------


def _read(document: Any, target: Sequence[str | int]) -> Any:
    node = document
    for key in target:
        if isinstance(key, int):
            if not isinstance(node, list) or key >= len(node):
                return None
            node = node[key]
        else:
            if not isinstance(node, Mapping) or key not in node:
                return None
            node = node[key]
    return node


def _write_into(document: Any, target: Sequence[str | int],
                value: Any) -> None:
    node = document
    for key, nxt in zip(target[:-1], target[1:]):
        if isinstance(key, int):
            node = node[key]
            continue
        existing = node.get(key)
        if not isinstance(existing, (dict, list)):
            existing = [] if isinstance(nxt, int) else {}
            node[key] = existing
        node = existing
    last = target[-1]
    if isinstance(last, int):
        node[last] = value
    else:
        node[last] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, (_datetime.datetime, _datetime.date,
                          _datetime.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rebase_declared_paths(raw: dict, base_dir: Path) -> list[dict]:
    """Make every relative declared input absolute against ``base_dir``."""

    rebased: list[dict] = []

    def walk(node: Any, trail: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if isinstance(value, str):
                    moved = _rebase(value, base_dir)
                    if moved is not None:
                        node[key] = moved
                        rebased.append({
                            "setting": ".".join((*trail, key)),
                            "from": value, "to": moved})
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        if isinstance(item, str):
                            moved = _rebase(item, base_dir)
                            if moved is not None:
                                value[index] = moved
                                rebased.append({
                                    "setting":
                                        f"{'.'.join((*trail, key))}[{index}]",
                                    "from": item, "to": moved})
                        else:
                            walk(item, (*trail, key))
                else:
                    walk(value, (*trail, key))

    for table in _PATH_TABLES:
        if isinstance(raw.get(table), dict):
            walk(raw[table], (table,))
    parameters = raw.get("physics_params")
    if isinstance(parameters, dict) and isinstance(parameters.get("set"), str):
        value = parameters["set"]
        moved = _rebase(value, base_dir)
        if moved is not None:
            parameters["set"] = moved
            rebased.append({"setting": "physics_params.set",
                            "from": value, "to": moved})
    return rebased


def _rebase(value: str, base_dir: Path) -> str | None:
    """``value`` as an absolute path, or None when it is not a path here."""

    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return None
    resolved = (base_dir / candidate)
    if not resolved.exists():
        return None
    return str(resolved.resolve())


# ---------------------------------------------------------------------------
# TOML emission
# ---------------------------------------------------------------------------

from woof.toml_document import emit_experiment_toml
from woof.ensemble.physical_fields import sha256_of_file as _sha256


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchPlan:
    """Everything the new run directory was given, and where it came from."""

    outdir: Path
    config_path: Path
    checkpoint: Path
    parent_run: Path
    parent_config: Path
    receipt_path: Path
    manifest_path: Path
    receipt: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _publish(path: Path, encoded: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _encode(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _check_target(outdir: Path, parent_run: Path) -> None:
    if outdir == parent_run or parent_run in outdir.parents:
        raise ValueError(
            f"--outdir {outdir} is inside the run being branched from "
            f"({parent_run}); a branch must not write into its own "
            "source, or the what-if and the run it is asking about "
            "share a directory and neither can be read afterwards")
    if outdir in parent_run.parents:
        raise ValueError(
            f"--outdir {outdir} contains the run being branched from "
            f"({parent_run}); the new run's own outputs would sit "
            "around the source run's, so neither directory names one "
            "forecast any more")
    if outdir.exists():
        if not outdir.is_dir():
            raise ValueError(
                f"--outdir {outdir} exists and is not a directory")
        occupants = sorted(entry.name for entry in outdir.iterdir())
        if occupants:
            raise ValueError(
                f"--outdir {outdir} is not empty ({len(occupants)} "
                f"entries, first {occupants[0]!r}); a branch writes a "
                "fresh run and will not mix its frames with whatever is "
                "already there")


def _checkpoint_members(parent_run: Path, checkpoint: Path,
                        checkpoint_set) -> tuple[dict, str | None]:
    """The set the branch reads, hashed -- and the instant it holds."""

    from woof.resume import discover_checkpoint_sets

    if checkpoint_set is None:
        for candidate in discover_checkpoint_sets(parent_run):
            if checkpoint.resolve() in {
                    member.resolve() for member in candidate.members.values()}:
                checkpoint_set = candidate
                break
    if checkpoint_set is None:
        members = {0: checkpoint}
        valid_time = None
        set_id = None
    else:
        members = dict(checkpoint_set.members)
        valid_time = checkpoint_set.valid_time.isoformat()
        set_id = checkpoint_set.set_id
    rows = [
        {"grid_id": grid_id, "path": str(path.resolve()),
         "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for grid_id, path in sorted(members.items())]
    return ({"schema": BRANCH_MANIFEST_SCHEMA,
             "handle": str(checkpoint.resolve()),
             "set_id": set_id,
             "valid_time": valid_time,
             "members": rows,
             "access": "read-only"}, valid_time)


def prepare_branch(*, config: str | Path, outdir: str | Path,
                   from_run: str | Path | None = None,
                   checkpoint_spec: str | Path = "latest",
                   settings: Iterable[str] = ()) -> BranchPlan:
    """Create the branch run directory, its config and its receipts.

    Nothing is created until every setting has been classified and the
    branched configuration has been loaded and compared against the
    parent's restart identity: a refused branch leaves no directory.
    """

    from woof.core.model import restart_identity_payload
    from woof.experiment import (is_experiment_toml, load_experiment,
                                  readable_config_path)
    from woof.resume import LATEST, resolve_resume_checkpoint

    config = readable_config_path(config).resolve()
    if not is_experiment_toml(config):
        # A legacy [run]-table RunConfig has no [[domain]] tables and no
        # restart identity payload to compare, so the whole
        # changeable-vs-pinned split this door exists to enforce has
        # nothing to read.  Said here rather than left to the experiment
        # loader's schema error, which would name a missing table
        # instead of the reason the door does not serve this shape.
        raise ValueError(
            f"{config} is a legacy [run]-table configuration, and a "
            "branch needs the experiment schema: the restart identity "
            "this door compares before it writes anything "
            "(woof.core.model.restart_identity_payload) is built from "
            "[experiment]/[[domain]] tables.  Run the experiment-shaped "
            "config the checkpointing route uses, or continue the run "
            "in place with `woof resume`.")
    if from_run is None:
        if str(checkpoint_spec) == LATEST:
            raise ValueError(
                "--from-run names the run directory to branch from; pass "
                "it, or pass an explicit --from <gpuwmrst_*.npz> whose "
                "directory is that run")
        parent_run = Path(checkpoint_spec).resolve().parent
    else:
        parent_run = Path(from_run).resolve()
        if not parent_run.is_dir():
            raise ValueError(
                f"--from-run {parent_run} is not a directory; it must be "
                "the --outdir of the run being branched from")
    outdir = Path(outdir).resolve()
    _check_target(outdir, parent_run)

    resolution = resolve_resume_checkpoint(parent_run, checkpoint_spec,
                                           config=config)
    checkpoint = Path(resolution.checkpoint).resolve()

    source_text = config.read_text(encoding="utf-8")
    raw = tomllib.loads(source_text)
    overrides = [parse_setting(text, raw) for text in settings]

    branched_raw = tomllib.loads(source_text)
    stamped: list[dict] = []
    for override in overrides:
        before = _read(branched_raw, override.target)
        _write_into(branched_raw, override.target, override.value)
        stamped.append({
            "setting": override.spelling,
            "from": _jsonable(before),
            "to": _jsonable(override.value),
            "tolerated_by": "woof.core.model.restart_identity_payload"})
    rebased = _rebase_declared_paths(branched_raw, config.parent)
    text = emit_experiment_toml(branched_raw)

    # The branched config is written at its FINAL path before it is
    # loaded, so every refusal and every plan-time warning the loader
    # raises names the file a reader can actually open.  Validating a
    # copy in a scratch directory first would have named a path that
    # stops existing the moment the message is printed.
    #
    # The cost of that choice is cleanup, paid here rather than left
    # behind: if the load or the identity comparison refuses, the only
    # things this call created are removed, so a corrected retry finds
    # the target exactly as it found it the first time (a refused
    # operation that poisons its own --outdir makes the retry fail on
    # something the reader never did).
    created_dir = not outdir.exists()
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / BRANCHED_CONFIG_NAME
    try:
        _publish(config_path, text.encode("utf-8"))
        branched_experiment = load_experiment(config_path)
        parent_payload = _encode(_jsonable(
            restart_identity_payload(load_experiment(config))))
        branched_payload = _encode(_jsonable(
            restart_identity_payload(branched_experiment)))
        if parent_payload != branched_payload:
            moved = _differing(json.loads(parent_payload),
                               json.loads(branched_payload))
            raise ValueError(
                "the branched configuration does not carry the "
                f"checkpoint's restart identity: {', '.join(moved)} "
                "differ(s) from the parent.  woof.io.restart would "
                "refuse the restore, so the branch is refused here "
                "instead; changeable from a checkpoint: "
                f"{CHANGEABLE_SETTINGS}")
    except BaseException:
        config_path.unlink(missing_ok=True)
        if created_dir:
            try:
                outdir.rmdir()
            except OSError:
                pass
        raise

    manifest, valid_time = _checkpoint_members(
        parent_run, checkpoint, resolution.checkpoint_set)
    receipt = {
        "schema": BRANCH_RECEIPT_SCHEMA,
        "created_utc": _datetime.datetime.now(
            _datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "parent": {
            "run_directory": str(parent_run),
            "config": {"path": str(config), "sha256": _sha256(config)},
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "valid_time": valid_time,
                "set_id": manifest["set_id"],
            },
            "skipped_newer_checkpoints": list(resolution.skipped),
        },
        "branch": {
            "run_directory": str(outdir),
            "config": {
                "path": str(outdir / BRANCHED_CONFIG_NAME),
                "sha256": hashlib.sha256(
                    text.encode("utf-8")).hexdigest()},
        },
        "overrides": stamped,
        "rebased_paths": rebased,
        "restart_identity": {
            "definition": "woof.core.model.restart_identity_payload",
            "parent_payload_sha256": hashlib.sha256(
                parent_payload).hexdigest(),
            "payload_sha256": hashlib.sha256(branched_payload).hexdigest(),
            "equal": True,
        },
    }

    receipt_path = outdir / BRANCH_RECEIPT_NAME
    _publish(receipt_path, _encode(receipt))
    manifest_path = outdir / BRANCH_MANIFEST_NAME
    _publish(manifest_path, _encode(manifest))
    return BranchPlan(outdir=outdir, config_path=config_path,
                      checkpoint=checkpoint, parent_run=parent_run,
                      parent_config=config, receipt_path=receipt_path,
                      manifest_path=manifest_path, receipt=receipt,
                      manifest=manifest)


def _differing(parent: Mapping[str, Any],
               branched: Mapping[str, Any]) -> list[str]:
    absent = object()
    return sorted(
        name for name in set(parent) | set(branched)
        if parent.get(name, absent) != branched.get(name, absent))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``woof branch`` beside ``run``/``resume``."""

    branch = subparsers.add_parser(
        "branch",
        help="start a NEW run from an existing run's checkpoint with "
             "changed settings (the what-if door); the source run is "
             "opened read-only")
    from woof.output_formats import add_arguments
    add_arguments(branch)
    branch.add_argument(
        "config", type=Path, metavar="CONFIG",
        help="the config the source run used; the branch edits a copy of "
             "it and the restart identity check refuses any other")
    branch.add_argument(
        "--from-run", dest="from_run", type=Path, default=None,
        metavar="RUNDIR",
        help="the source run's output directory -- where its "
             "gpuwmrst_*.npz checkpoints are.  Optional only when --from "
             "names a checkpoint file explicitly")
    branch.add_argument(
        "--from", dest="from_checkpoint", default="latest",
        metavar="CKPT|latest",
        help="explicit gpuwmrst_*.npz checkpoint to branch from, or "
             "'latest' (default) for the newest valid set in --from-run")
    branch.add_argument(
        "--outdir", type=Path, required=True, metavar="OUT",
        help="the NEW run's output directory; it must be empty and must "
             "not be inside the source run")
    branch.add_argument(
        "--set", dest="settings", action="append", default=[],
        metavar="KEY=VALUE",
        help="a setting to change in the branched run, repeatable.  "
             f"Changeable from a checkpoint: {CHANGEABLE_SETTINGS}.  "
             "Everything else is refused by name, because the restart "
             "identity binds it")
    branch.add_argument(
        "--prepare-only", dest="prepare_only", action="store_true",
        help="write the branch run directory, its config and its "
             "receipts, then stop without integrating -- the price-it-"
             "first step a what-if screen shows before committing a card")


def prepare_branch_from_cli(args: argparse.Namespace) -> BranchPlan:
    """Run the door for one parsed command line and say what it did."""

    plan = prepare_branch(config=args.config, outdir=args.outdir,
                          from_run=args.from_run,
                          checkpoint_spec=args.from_checkpoint,
                          settings=args.settings)
    for note in plan.receipt["parent"]["skipped_newer_checkpoints"]:
        print(f"branch: skipped newer checkpoint {note}")
    changed = plan.receipt["overrides"]
    print(f"branch: new run {plan.outdir} from {plan.checkpoint} "
          f"({len(changed)} setting(s) changed)")
    for row in changed:
        print(f"branch:   {row['setting']}: {row['from']!r} -> "
              f"{row['to']!r}")
    print(f"branch: receipts {plan.receipt_path.name}, "
          f"{plan.manifest_path.name}; config {plan.config_path.name}")
    return plan


__all__ = ["BRANCHED_CONFIG_NAME", "BRANCH_MANIFEST_NAME",
           "BRANCH_MANIFEST_SCHEMA", "BRANCH_RECEIPT_NAME",
           "BRANCH_RECEIPT_SCHEMA", "BranchPlan", "CHANGEABLE_SETTINGS",
           "Override", "emit_experiment_toml", "parse_setting",
           "prepare_branch", "prepare_branch_from_cli", "register_cli"]
