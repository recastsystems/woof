"""One certification capsule, built by one builder, emitted at every exit.

Four routes finish a forecast in this tree -- the front door's single-domain
and domain-tree exits and the two prepared runners -- and the supervisor closes
a supervised run on top of them.  Each of the five writes
``certification-capsule.json`` through :func:`emit_run_capsule`, so there is
exactly one place a field can be added, renamed or lost, and a field that goes
missing goes missing everywhere at once instead of in the route nobody reruns.

The capsule records what was measured and admits what was not.  Sections a
route cannot fill arrive as explicit ``unavailable`` markers rather than
plausible values; :func:`validate_certification_capsule` checks shape, and the
certification path additionally refuses an unresolved pin or an inventory-mode
geography digest -- a directory hashed by its listing rather than its contents
cannot witness that the geography bytes were the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

from woof.certify.kernel_manifest import kernel_manifest, manifest_is_empty
from woof.certify.pins import (PIN_KEYS, STATUS_RESOLVED, STATUS_UNAVAILABLE,
                                resolve_pins, unresolved_pins)
from woof.gpu_stack_identity import gpu_cuda_stack_identity

#: Schema id carried by every capsule this package writes.
CAPSULE_SCHEMA_ID = "gpuwm.certification-capsule/v1"

#: Artifact name, fixed program-wide so every consumer looks in one place.
CAPSULE_FILENAME = "certification-capsule.json"

#: Path of the schema document, beside this module.
SCHEMA_PATH = Path(__file__).resolve().parent / "capsule_v1.schema.json"

#: Top-level sections every capsule carries.  The four-site parity control
#: removes one of these from the builder and requires all four sites to fail.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "schema",
    "emission_site",
    "code",
    "registry",
    "input_bytes",
    "numerical_stack",
    "kernel_manifest",
    "arithmetic_contract",
    "run_shape",
    "output",
    "receipts",
)

#: The five routes permitted to emit.  A capsule names the exit that wrote it.
EMISSION_SITES: tuple[str, ...] = (
    "runtime.run_experiment:single-domain",
    "runtime.run_experiment:domain-tree",
    "prepared_single_domain_forecast",
    "prepared_domain_tree_forecast",
    "supervisor:success",
)

#: Directory-hash algorithm the certification path requires of geography.
GEOGRAPHY_CONTENT_ALGORITHM = "sha256-directory-content"

#: Directory-hash algorithm the certification path refuses.
GEOGRAPHY_INVENTORY_ALGORITHM = "sha256-directory-inventory"


class CapsuleValidationError(ValueError):
    """A capsule does not satisfy the schema or the certification path."""


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": STATUS_UNAVAILABLE, "reason": reason}


def code_identity(code: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Which code produced these numbers.

    Public because a capsule is not the only artifact that has to answer it.
    A spectral receipt binds the arithmetic pins and the input bytes but not
    the evaluator; building a second resolver there would let a capsule and a
    receipt from one run name two different commits, with nothing in either
    saying which is right.  One builder, one answer.
    """

    import woof
    from woof.supervisor import git_commit

    supplied = dict(code or {})
    section = {
        "gpuwm_version": woof.__version__,
        "git_commit": git_commit(),
        "tree_sha256": supplied.get("tree_sha256"),
        "worktree_clean": supplied.get("worktree_clean"),
        "runtime_source_sha256": supplied.get("runtime_source_sha256"),
    }
    return section


def _code_section(code: Mapping[str, Any] | None) -> dict[str, Any]:
    return code_identity(code)


def _schema_document() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def build_capsule(*, emission_site: str,
                  run_context: Mapping[str, Any] | None = None,
                  code: Mapping[str, Any] | None = None,
                  registry: Mapping[str, Any] | None = None,
                  input_bytes: Mapping[str, Any] | None = None,
                  arithmetic_contract: Mapping[str, Any] | None = None,
                  run_shape: Mapping[str, Any] | None = None,
                  output: Mapping[str, Any] | None = None,
                  receipts: Mapping[str, Any] | None = None,
                  require_gpu: bool = True) -> dict[str, Any]:
    """Assemble one capsule.  Unmeasured sections say so; none are dropped."""
    if emission_site not in EMISSION_SITES:
        raise ValueError(f"unknown capsule emission site {emission_site!r}")
    capsule: dict[str, Any] = {
        "schema": CAPSULE_SCHEMA_ID,
        "emission_site": emission_site,
        "code": _code_section(code),
        "registry": dict(registry) if registry else _unavailable(
            "the emitting route resolved no physics registry"),
        "input_bytes": dict(input_bytes) if input_bytes else _unavailable(
            "the emitting route resolved no declared inputs"),
        "numerical_stack": resolve_pins(run_context, require_gpu=require_gpu),
        "kernel_manifest": kernel_manifest(),
        "arithmetic_contract": (
            dict(arithmetic_contract) if arithmetic_contract
            else _unavailable("no FTZ receipt was bound to this run")),
        "run_shape": dict(run_shape) if run_shape else _unavailable(
            "the emitting route recorded no run shape"),
        "output": dict(output) if output else _unavailable(
            "the emitting route recorded no output frames"),
        "receipts": dict(receipts) if receipts else {},
    }
    return capsule


def validate_certification_capsule(
        capsule: Mapping[str, Any], *,
        certification_path: bool = False) -> dict[str, Any]:
    """Check a capsule against the schema, and against the certification path.

    Schema validation is structural.  ``certification_path`` adds the ratified
    refusals: every pin must be ``resolved``, every directory input must be
    hashed by content -- an inventory digest is a listing, not the bytes, and
    cannot carry a byte-identity claim -- and the kernel manifest must record
    at least one compiled module, because a capsule that compiled nothing
    cannot witness which kernels produced the numbers it carries.
    """
    try:
        import jsonschema
    except ModuleNotFoundError as error:
        # Declared in pyproject.toml, so this fires only on an install
        # that predates the declaration; name the one-command remedy
        # instead of surfacing a bare ModuleNotFoundError.
        raise CapsuleValidationError(
            "capsule validation needs the jsonschema package "
            "(pip install jsonschema)") from error

    missing = [name for name in REQUIRED_SECTIONS if name not in capsule]
    if missing:
        raise CapsuleValidationError(
            f"capsule is missing required sections {missing}")
    try:
        jsonschema.validate(instance=dict(capsule), schema=_schema_document())
    except jsonschema.ValidationError as error:
        raise CapsuleValidationError(
            f"capsule does not validate against {CAPSULE_SCHEMA_ID}: "
            f"{error.message}") from error
    if not certification_path:
        return dict(capsule)

    if manifest_is_empty(capsule["kernel_manifest"]):
        raise CapsuleValidationError(
            "the certification path refuses a capsule whose kernel manifest "
            "recorded no compiled module: a run that compiled nothing cannot "
            "witness the kernels its numbers came from")
    unresolved = unresolved_pins(capsule["numerical_stack"])
    if unresolved:
        raise CapsuleValidationError(
            "the certification path refuses a capsule whose pins are not "
            f"resolved: {list(unresolved)}")
    inventory = [
        key for key, record in _input_entries(capsule).items()
        if record.get("algorithm") == GEOGRAPHY_INVENTORY_ALGORITHM
    ]
    if inventory:
        raise CapsuleValidationError(
            "the certification path requires directory inputs hashed by "
            f"content ({GEOGRAPHY_CONTENT_ALGORITHM}); these carry an "
            f"inventory digest: {sorted(inventory)}")
    return dict(capsule)


def _input_entries(capsule: Mapping[str, Any]) -> dict[str, Any]:
    section = capsule.get("input_bytes") or {}
    entries = section.get("entries") if isinstance(section, Mapping) else None
    return dict(entries) if isinstance(entries, Mapping) else {}


def emit_capsule(outdir: str | Path, capsule: Mapping[str, Any], *,
                 certification_path: bool = False) -> Path:
    """Validate, then write the capsule to ``outdir`` under the fixed name."""
    validated = validate_certification_capsule(
        capsule, certification_path=certification_path)
    path = Path(outdir) / CAPSULE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)
    return path


def emit_run_capsule(outdir: str | Path, *, emission_site: str,
                     certification_path: bool = False,
                     **sections: Any) -> Path | None:
    """The one call every emitting route makes.

    Routes differ in what they can supply, never in how the capsule is built
    or checked; keeping build and emit behind this single entry point is what
    makes the four-site parity assertion mean anything.

    NON-FATAL to a completed forecast.  This runs after the last model
    step, when the outputs and receipts already exist on disk; a capsule
    that cannot be built or written (a missing ``jsonschema`` on a
    minimal install, a schema drift, a full disk) must never turn a
    finished run into a crash.  One warning line says what happened and
    ``None`` is returned.  Only ``certification_path=True`` -- the
    caller explicitly asked for a certification verdict -- keeps the
    failure fatal, because there the capsule IS the product.
    """
    try:
        capsule = build_capsule(emission_site=emission_site, **sections)
        path = emit_capsule(outdir, capsule,
                            certification_path=certification_path)
    except Exception as error:  # noqa: BLE001 - the run's outputs stand
        if certification_path:
            raise
        from woof.explain import warn
        warn("certification capsule not written: "
             f"{type(error).__name__}: {error} -- the forecast outputs "
             "and receipts above stand",
             why="The capsule is an audit artifact emitted after the "
                 "run completed; only `woof certify` treats its "
                 "absence as a failure.")
        path = None
    # Output work starts only after durable forecast output and its capsule.
    # Its pending receipt is independent of the physical run's verdict.
    from woof.verification_visuals import finish_line, finish_run
    line = finish_line(finish_run(outdir, sections=sections))
    if line:
        # One line, after the hook returned, so the hook itself still never
        # writes to a stream; a closed stream cannot fail a finished run.
        try:
            print(line, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
    return path


def load_certification_capsule(path: str | Path, *,
                               certification_path: bool = False
                               ) -> dict[str, Any]:
    """Read a capsule from disk and validate it before returning it."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_certification_capsule(
        document, certification_path=certification_path)


__all__ = [
    "CAPSULE_FILENAME",
    "CAPSULE_SCHEMA_ID",
    "EMISSION_SITES",
    "GEOGRAPHY_CONTENT_ALGORITHM",
    "GEOGRAPHY_INVENTORY_ALGORITHM",
    "PIN_KEYS",
    "REQUIRED_SECTIONS",
    "SCHEMA_PATH",
    "CapsuleValidationError",
    "build_capsule",
    "code_identity",
    "emit_capsule",
    "emit_run_capsule",
    "gpu_cuda_stack_identity",
    "load_certification_capsule",
    "validate_certification_capsule",
    "STATUS_RESOLVED",
]
