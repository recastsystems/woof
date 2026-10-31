#!/usr/bin/env python3
"""Run a hash-bound RW-WPS prepared hierarchy through GPUWM.

The public source adapters already publish one verified prepared cache per
domain.  This launcher restores that complete tree and hands it to GPUWM's
existing ``DomainNode``/``NestCoupler``/``execute_experiment`` engine.  It
does not implement another integrator and it does not flatten a nest tree
into the single-domain benchmark runner.

The experiment TOML remains the typed authority for arbitrary supported
static one-way layouts and per-domain physics.  In particular, MP8 outer
domains may feed MP18 inner domains through the existing explicit
``mp8-to-mp18-mass-diagnosed-v1`` policy.  Implemented configurations that do
not yet have retained acceptance evidence are reported as warnings, never as
consent gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from copy import copy
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _names_this_runners_tree(entry: str) -> bool:
    """Whether one ``sys.path`` entry is the tree this module came from."""
    if not entry:
        return False
    try:
        return Path(entry).resolve() == REPOSITORY_ROOT
    except OSError:
        return False


# THE TREE THIS RUNNER WAS LOADED FROM GOES FIRST, under every spelling of
# it.  ``python -m woof.prepared_domain_tree_forecast`` -- the line
# ``woof sim --print-command`` prints for a caller to drive itself -- puts
# the CALLER'S DIRECTORY at ``sys.path[0]``.  Run from a directory holding
# an older checkout's ``tilestream/``, that directory supplied this
# runner's streaming restart types instead of the ones shipped beside this
# file, and a resume died on
#   ImportError: cannot import name 'ValidatedStreamedRestart'
#   from 'tilestream.restart_stream'
# out of woof/io/restart.py, with a forecast's checkpoint on disk and no
# way to continue it.  woof and tilestream are one distribution: a run
# that takes them from two trees is a mismatched pair, not a choice.
#
# The membership test here used to be a STRING compare against sys.path,
# so the insert was skipped exactly when the install path happened to be
# spelled the way ``site`` spelled it -- and a symlink anywhere above the
# environment was the difference between a resume that worked and that
# ImportError.  Resolving both sides removes that coincidence, and taking
# every spelling out before re-inserting puts the pair ahead of the
# caller's directory rather than merely on the path.  An empty entry (the
# caller's own directory) is left where it is, one place further down.
sys.path[:] = [entry for entry in sys.path
               if not _names_this_runners_tree(entry)]
sys.path.insert(0, str(REPOSITORY_ROOT))

from woof.forecast_initialization import TreeInitialization  # noqa: E402
from woof import __version__  # noqa: E402
from woof import prepared_single_domain_forecast as prepared_single  # noqa: E402
from woof.certify.capsule import emit_run_capsule  # noqa: E402
from woof.spectral_seam import (  # noqa: E402
    seam_capsule_receipts as _seam_capsule_receipts,
)
from woof.core.adaptive_clock import (  # noqa: E402
    NestDivideRefusal,
)
from woof.core import streaming  # noqa: E402
from woof.core.microphysics_transition import (  # noqa: E402
    MP8_TO_MP18_POLICY,
    resolve_microphysics_transition,
)
from woof.aerosol_source_receipt import (  # noqa: E402
    AEROSOL_SOURCE_KEY,
    aerosol_source_report_entries,
)
from woof.experiment import load_experiment  # noqa: E402
from woof.acoustic_adaptation import (  # noqa: E402
    acoustic_receipt, adapt_experiment_to_terrain, fold_corridor_reading,
    readings_from_static)
from woof.terrain_clock import (  # noqa: E402
    clock_for_prepared_cache, clock_for_wrfinput, clock_receipt)
from woof.vertical_adaptation import (  # noqa: E402
    adopt_prepared_vertical, prepared_domain_coordinate_refusal)
from woof.kernel_compile_notice import (  # noqa: E402
    COMPILING_STATUS, current_compute_capability, kernel_cache_state,
    scan_kernel_cache,
)
from woof.physics_compat import (  # noqa: E402
    experimental_selection_sentence,
)
from woof.io.restart import RestartMismatchError  # noqa: E402
from woof.io.history_layout import wrf_file_grid_history_fields  # noqa: E402
from woof.ingest.memory_refusal import InitializationMemoryRefused  # noqa: E402
from woof.supervisor import (  # noqa: E402
    HEARTBEAT_NAME,
    quarantine_file as supervisor_quarantine_file,
    replace_file_with_retry as supervisor_replace_file_with_retry,
    restart_attempt,
)
from woof.ingest.prepared_cache import (  # noqa: E402
    PreparedCacheReader,
    compare_prepared_domain_config,
    effective_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_identity_refusal,
    undelayed_identity_defaults,
)
from woof import progress_log  # noqa: E402
from woof.progress_log import (  # noqa: E402
    ProgressOptions, add_progress_arguments)
from woof.receipt_paths import receipt_basename  # noqa: E402
from woof.native_wrf_contract import (  # noqa: E402
    NATIVE_LANDUSE_IDENTITY,
    native_landuse_identity,
    load_native_static_cache,
    verify_native_static_receipt,
)
from woof.static.lambert import grids_from_projection_config  # noqa: E402
from woof.table_assets import MissingTableAssets  # noqa: E402


REPORT_SCHEMA = "gpuwm-prepared-domain-tree-forecast-v1"
PROGRESS_SCHEMA = "gpuwm-prepared-domain-tree-progress-v1"
CAPABILITIES_SCHEMA = "gpuwm-runner-capabilities-v1"
PLAN_SCHEMA = "gpuwm-prepared-domain-tree-plan-v1"
RUNNER = "tools.prepared_domain_tree_forecast"
ARBITRARY_PLAN_ID = "arbitrary-prepared-one-way-domain-tree-v1"
THOMPSON_NSSL_PLAN_ID = "thompson-outer-nssl2-inner-mp8-mp18-v1"
HIERARCHY_SCHEMA = "gpuwm-native-hrrr-hierarchy-direct-v1"
SEALED_EXTENSION_FINGERPRINT_SCHEMA = \
    "gpuwm-prepared-tree-sealed-extension-fingerprint-v1"
# Every source builds its whole domain tree through the same artifact writer,
# so `hierarchy-artifacts/` is identical across all of them. Only the top-level
# document differs: HRRR prepares its tree in a separate pass and writes
# receipt.json, while the namelist-driven sources build theirs inside RW-WPS
# preparation and write proof.json. Reading both is what makes nested execution
# a property of the topology rather than of the source.
_HIERARCHY_DOCUMENTS = (
    {
        "filename": "receipt.json",
        "schema": HIERARCHY_SCHEMA,
        "status": "PASS",
        "source": "hrrr",
    },
    {
        "filename": "proof.json",
        "schema": "gpuwm-era5-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "era5",
    },
    {
        "filename": "proof.json",
        "schema": "gpuwm-gfs-native-hierarchy-proof-v2",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "gfs",
    },
    {
        # v1 predates the front-door physics receipt and therefore cannot
        # be promoted to v2 by inference; it stays independently
        # verifiable on its own terms, exactly as the direct proof's v2
        # does beside v3.
        "filename": "proof.json",
        "schema": "gpuwm-gfs-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "gfs",
    },
    {
        "filename": "proof.json",
        "schema": "gpuwm-mapped-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "mapped",
    },
)
SUPPORTED_SOURCES = tuple(sorted(
    {entry["source"] for entry in _HIERARCHY_DOCUMENTS}
    | prepared_single._MAPPED_SOURCES))
ARTIFACT_RECEIPT_SCHEMA = "gpuwm-native-hierarchy-artifact-build-v1"
ARTIFACT_MANIFEST_SCHEMA = "gpuwm-native-domain-artifacts-v1"
_HEX = frozenset("0123456789abcdef")
_FORECAST_EXECUTOR_MODULES = (
    "woof.core.clock",
    "woof.core.dycore",
    "woof.core.health",
    "woof.core.model",
    "woof.core.nest",
    "woof.io.restart",
    "woof.io.wrfout",
    "woof.state_digest",
)


_sha256 = prepared_single._sha256


_canonical = prepared_single._canonical


def _chem_ledger_receipt(state):
    """One domain's chem mass ledger for the receipt; None when chem is off.

    Read only when the state carries chem, so a chem-off run imports
    nothing more than it did.
    """
    if getattr(state, "chem", None) is None:
        return None
    from woof.core.chem_driver import chem_ledger_receipt

    return _strict_json(chem_ledger_receipt(state))


def _strict_json(value):
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        # v1.1 nests gave DomainConfig an optional per-domain start_time,
        # so a serialized domain config now reaches here carrying a
        # datetime.  ISO 8601 is what every other identity document in
        # the tree already writes (prepared_cache, source_hierarchy,
        # native_domain_artifacts), and identity digests only agree if
        # this agrees with them.
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def _without_forecast_stop(exp):
    """Normalize only the typed experiment's enumerated forecast stops.

    ``ExperimentConfig.run_seconds`` is the one identity field which changes
    between successively longer legs.  The typed loader also copies that
    authority into each ``DomainConfig.run.run_seconds``; remove those exact
    derived paths only after proving they still equal the authority.  Any
    future nested field with the same spelling remains covered.  Keeping the
    exceptions explicit prevents a newly introduced control from silently
    falling outside the restart-extend identity.
    """
    from woof.experiment import experiment_config_document
    value = _strict_json(experiment_config_document(exp))
    if not isinstance(value, Mapping) or "run_seconds" not in value:
        raise ValueError(
            "sealed extension identity requires ExperimentConfig.run_seconds")
    result = dict(value)
    authoritative_stop = result.pop("run_seconds")
    # The mixing-length provenance label is identity-inert here for the
    # same reason woof.core.model.restart_identity_payload drops it:
    # the chosen value binds on run.mix_isotropic, and an auto-selected
    # 1 must extend a written 1's sealed legs (and vice versa).
    result.pop("auto_mix_isotropic", None)
    # Likewise the off-centering provenance label (run.epssm binds).
    result.pop("auto_epssm", None)
    result.pop("simulated_radar", None)
    domains = result.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(
            "sealed extension identity requires typed experiment domains")
    normalized_domains = []
    for index, domain in enumerate(domains):
        if not isinstance(domain, Mapping) or not isinstance(
                domain.get("run"), Mapping):
            raise ValueError(
                f"sealed extension identity domain {index} lacks RunConfig")
        normalized_domain = dict(domain)
        normalized_run = dict(domain["run"])
        if normalized_run.get("run_seconds") != authoritative_stop:
            raise ValueError(
                f"sealed extension identity domain {index} run_seconds "
                "diverges from ExperimentConfig.run_seconds")
        normalized_run.pop("run_seconds")
        normalized_domain["run"] = normalized_run
        normalized_domains.append(normalized_domain)
    result["domains"] = normalized_domains
    return result


def sealed_extension_identity_components(
    exp, runtime_identity
) -> dict[str, object]:
    """The named components whose digest is the sealed-extension identity.

    The sealed-extension counterpart of
    :func:`tree_restart_identity_components`, and named for the same
    reason: these are published beside the digest in the checkpoint
    header, so a refusal can say WHICH component moved instead of only
    that the hash did.  Both checkpoint routes now carry named
    components; before this the sealed route carried none, and a
    horizon extension that failed to line up could only report a bare
    digest difference.
    """

    return {
        "schema": SEALED_EXTENSION_FINGERPRINT_SCHEMA,
        "experiment": _without_forecast_stop(exp),
        "runtime_source_identity": _strict_json(runtime_identity),
    }


def sealed_extension_fingerprint(exp, runtime_identity) -> str:
    """Stable trajectory identity shared by successively longer legs."""
    payload = sealed_extension_identity_components(exp, runtime_identity)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload, *, heartbeat: bool = False) -> None:
    """Publish one JSON document atomically, through the supervisor's replace.

    ``heartbeat=True`` is for the progress publications a watcher is
    TOLD to read (``evidence/progress.json``): on Windows a plain
    ``open()`` for read denies rename over the file, so a poll racing
    the republish raised WinError 5 out of ``progress_callback`` and
    killed the forecast it was reporting on (MEASURED: a 12-3 km tree
    died at outer step 76 of 120 under a reader on the documented file;
    ``woof go``'s own stopwatch heartbeat reads it every 20 s, so the
    product races itself).  The supervisor's doctrine applies verbatim:
    bounded 0.50 s retry for everyone, then durable receipts fail loudly
    while a heartbeat quarantines its temporary and the worker stays up
    -- a stale heartbeat is safer than terminating a healthy CUDA run.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    encoded = (
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        supervisor_replace_file_with_retry(temporary, path)
    except PermissionError:
        if not heartbeat:
            raise
        supervisor_quarantine_file(
            temporary, reason="heartbeat-sharing-violation")


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _digest(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in _HEX for char in normalized):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _bound_authority(authority: Mapping[str, object], head) -> dict:
    """The tree's authority digests, as every domain's cache must carry them.

    An as-posted head's domains bind its input plan where the manifest
    digest goes (the L3 design ruling): exactly this head's placeholder is
    taken, and only in the identity keys the head declares manifest-bound
    or document-bound.  The seal writes the named document's digest, and
    :func:`_seal_tree_inputs` holds every domain to it.  Everything else
    is a SHA-256 digest, as before.
    """

    placeholder = None
    bound: tuple = ()
    if head is not None and head["basis"].get("as_posted") is not None:
        from woof.ingest.boundary_stream import as_posted_placeholder

        posted = head["basis"]["as_posted"]
        placeholder = as_posted_placeholder(posted["input_plan_sha256"])
        bound = (tuple(posted["manifest_bound_identity_keys"])
                 + tuple(posted.get("document_bound_identity_keys") or {}))
    return {label: (value if placeholder is not None and label in bound
                    and value == placeholder else _digest(value, label))
            for label, value in authority.items()}


def _require_file(path: Path, label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _require_directory(path: Path, label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _inside(path: Path, root: Path) -> bool:
    # Compared in the plain spelling: a folder handed over in the extended
    # Windows spelling (a deep runs folder) is the same place as its plain
    # twin, and comparing the two spellings as text let an output folder
    # inside the protected inputs pass as outside them.
    from woof.filesystem_paths import canonical_path

    try:
        canonical_path(path).relative_to(canonical_path(root))
        return True
    except ValueError:
        return False


_sibling_outdir = prepared_single._sibling_outdir


def claim_output_directory(output: Path, *, protected_roots: tuple[Path, ...]) -> Path:
    """Create exactly one output directory without adopting old content.

    The folder comes back in the spelling it was handed: a deep runs folder
    arrives in the extended Windows spelling because the domains and
    pictures below it pass the 260-character limit, and answering in the
    plain spelling would have every write below it fail as a missing file.
    """
    from woof.filesystem_paths import canonical_path, keep_spelling

    result = keep_spelling(output, canonical_path(output))
    for protected in protected_roots:
        protected = canonical_path(protected)
        if _inside(result, protected) or _inside(protected, result):
            raise ValueError(
                f"output directory {result} overlaps protected input "
                f"{protected}; the forecast may not write into its own "
                f"inputs.  Pass an --outdir beside them instead, for "
                f"example {_sibling_outdir(protected)}"
            )
    result.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.mkdir()
    except FileExistsError:
        # The twin of the single-domain runner's claim, and it had the
        # twin defect: EXISTS is not the breakage, HOLDS AN EARLIER RUN
        # is.  `woof sim` allocates this run's stamped folder
        # create-exclusively before dispatching to either runner, so the
        # folder handed here always exists and is always empty; refusing
        # it prevented nothing and killed the route.
        try:
            occupied = any(result.iterdir())
        except OSError:
            # Unreadable is not empty: we cannot prove there is no
            # earlier run in there, and accepting on a failed probe is
            # the one direction that loses data.
            occupied = True
        if occupied:
            from woof.ensemble.runtime_context import current_session
            session = current_session()
            if (session is not None and session.restart_roster is not None
                    and result.resolve() == session.output_directory.resolve()):
                return result
            raise FileExistsError(
                f"refusing output directory that already holds a run: "
                f"{result}") from None
    return result


def _missing_executor_modules() -> list[str]:
    missing = []
    for module in _FORECAST_EXECUTOR_MODULES:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            present = False
        if not present:
            missing.append(module)
    return missing


def runner_capabilities() -> dict[str, object]:
    """Side-effect-free launcher contract for Studio and headless clients."""

    missing = _missing_executor_modules()
    available = not missing
    return {
        "schema": CAPABILITIES_SCHEMA,
        "runner": RUNNER,
        "supported_sources": list(SUPPORTED_SOURCES) if available else [],
        "readiness": (
            "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED_UNVERIFIED"
            if available
            else "FORECAST_EXECUTOR_OMITTED"
        ),
        "modes": {
            "forecast": {
                "available": available,
                "requires_cupy": True,
                "requires_compatible_cuda_gpu": True,
                "missing_executor_modules": missing,
                "included_in_standalone_rw_wps_wheel": False,
            },
        },
        "simulation_plan_ids": [
            ARBITRARY_PLAN_ID,
            THOMPSON_NSSL_PLAN_ID,
        ]
        if available
        else [],
        "simulation_plans": {
            ARBITRARY_PLAN_ID: {
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "topology": "arbitrary-engine-valid-static-one-way-tree",
                "physics": "per-domain-engine-valid-selectors",
                "validity_authority": "woof.experiment.load_experiment",
                "geometry_whitelist": False,
            },
            THOMPSON_NSSL_PLAN_ID: {
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "selectors": {
                    "outer_domains": 8,
                    "inner_domains": 18,
                    "transition": MP8_TO_MP18_POLICY,
                },
                "canonical_four_domain_binding": {
                    "d01": {"mp_physics": 8},
                    "d02": {"mp_physics": 8},
                    "d03": {
                        "mp_physics": 18,
                        "nest_microphysics_transition": MP8_TO_MP18_POLICY,
                    },
                    "d04": {
                        "mp_physics": 18,
                        "nest_microphysics_transition": "same-scheme-only",
                    },
                },
            },
        }
        if available
        else {},
        "warning_policy": {
            "implemented_unverified_is_launchable": True,
            "consent_gate": False,
            "warnings_and_receipts_retained": True,
            "hard_errors": [
                "malformed-or-incompatible-plan",
                "missing-or-mutated-authority",
                "missing-runtime-asset",
                "device-allocation-or-execution-failure",
            ],
        },
        "input": {
            "layout": "rw-wps hierarchy-artifacts/domain-artifacts.json",
            "hash_pins": [
                "preparation-receipt-sha256",
                "experiment-config-sha256",
            ],
            "every_domain_cache_reread_and_sha256_verified": True,
        },
        "relocation": {
            "follow_sources": (
                "accepted only over a sealed, digest-verified statics "
                "corridor prepared with --statics-corridor; corridor-less "
                "bundles refuse with that remedy named"),
            "bounds_only": "accepted (manual/API mechanism, no runner)",
            "corridor_memory": "disk/host only; no GPU residency",
        },
        "output": {
            "directory_policy": "create-only",
            "io_modes": ["history", "none"],
            "history_cadence": "per-domain experiment TOML",
            "restart": "experiment-configured-tree-checkpoints",
        },
        "capability_query": {
            "flag": "--show-capabilities",
            "side_effect_free": True,
            "requires_cupy": False,
            "validates_inputs_or_runtime_assets": False,
        },
    }


@dataclass(frozen=True)
class PreparedDomainBundle:
    grid_id: int
    parent_id: int
    bundle: Path
    cache: Path
    static_path: Path
    geometry_receipt_path: Path
    domain_receipt_path: Path
    cache_reader: PreparedCacheReader
    cache_identity: Mapping[str, object]
    static_fields: Mapping[str, np.ndarray]
    authority_sha256: Mapping[str, str]
    terrain_autosmooth: Mapping[str, object] | None = None
    cache_payload: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PreparedTreeInputs:
    prepared_root: Path
    hierarchy_root: Path
    preparation_receipt_path: Path
    artifact_receipt_path: Path
    artifact_manifest_path: Path
    experiment_config: Path
    experiment: object
    grids: tuple[object, ...]
    domains: tuple[PreparedDomainBundle, ...]
    forcing_hours: tuple[int, ...]
    boundary_interval_seconds: int
    source_identity: Mapping[str, object]
    execution_plan: Mapping[str, object]
    authority_sha256: Mapping[str, str]
    source: str
    #: Per-domain identity fields accepted as schema growth rather
    #: than as a match -- empty on a cache written by this release.
    tolerated_identity_fields: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({}))
    #: The verified statics corridor
    #: (:class:`woof.static.corridor.ChildStaticsCorridor`) when the
    #: experiment configures a [relocation] follow source, else ``None``.
    statics_corridor: object | None = None
    #: The corridor cache file, for the unchanged-during-run re-hash.
    statics_corridor_cache_path: Path | None = None
    mapped_authority_paths: Mapping[str, Path] = field(
        default_factory=lambda: MappingProxyType({}))
    physics_profile_assertion: Mapping[str, object] | None = None
    #: The acoustic substep derivation for every domain
    #: (:func:`woof.acoustic_adaptation.acoustic_receipt`), or ``None``
    #: until :func:`_with_terrain_acoustics` has read the tree's ground.
    acoustic_substeps: Mapping[str, object] | None = None
    #: The long-step derivation for every domain
    #: (:func:`woof.terrain_clock.clock_receipt`), filled in beside the
    #: substep one.
    terrain_clock: Mapping[str, object] | None = None
    #: The chained preparation's ``boundary-stream/head.json`` when this run
    #: is bound to the head (``--prepared-head-sha256``): the root restores
    #: from its streamed cache, the children from ``hierarchy-head/``, and
    #: the seal is bound at the end (:func:`_seal_tree_inputs`).  ``None``
    #: for a sealed binding.
    stream_head: Mapping[str, object] | None = None
    #: The head digest this tree was prepared under, however it is bound:
    #: the head itself, or a sealed proof that names it
    #: (``boundary_stream.head_sha256``).  ``None`` for a tree prepared in
    #: one piece.  It stands in the restart identity for the proof and the
    #: root cache digests, so a checkpoint written before the seal resumes
    #: under either binding.
    prepared_head_sha256: str | None = None
    #: ``{"dNN": content_sha256}`` of each child the head prepared, for a
    #: SEALED binding of an as-posted tree, else ``None``.  Such a tree's
    #: head children carry the input plan where their sealed twins carry
    #: the manifest digest, so the two content digests differ by that
    #: identity alone; the restart identity binds the head's under both
    #: bindings (:func:`tree_restart_identity_components`), each sealed
    #: child held to its head twin (:func:`_as_posted_head_children`).
    head_child_content_sha256: Mapping[str, str] | None = None
    #: What the long-step derivation read before it chose a clock
    #: (:class:`woof.ingest.boundary_stream.ClockBasis`).
    clock_basis: object | None = None
    #: The preflight's own arguments, so the seal runs the same preflight
    #: again on the sealed tree.
    preflight_arguments: Mapping[str, object] | None = None


def _with_terrain_acoustics(inputs: PreparedTreeInputs) -> PreparedTreeInputs:
    """The inputs with each domain's acoustic substeps derived from its ground.

    Every domain is read off the static fields it restores; a relocating
    nest also off its statics corridor, the ground it can move over.  An
    inputs object that already carries the derivation is returned as it
    is, so the runner and a preflight that both ask get one answer.
    """

    if getattr(inputs, "acoustic_substeps", None) is not None:
        return inputs
    exp = inputs.experiment
    grids = {int(dc.grid_id): grid
             for dc, grid in zip(exp.domains, inputs.grids)}
    readings = readings_from_static(
        exp, {bundle.grid_id: bundle.static_fields
              for bundle in inputs.domains},
        grids_by_grid_id=grids)
    runs = {int(dc.grid_id): dc.run for dc in exp.domains}
    corridors = inputs.statics_corridor
    if not isinstance(corridors, Mapping):
        corridors = {}
    reach = {}
    for grid_id, corridor in corridors.items():
        fields = getattr(corridor, "fields", None)
        run = runs.get(int(grid_id))
        if run is None or not fields or "HGT_M" not in fields:
            continue
        fold_corridor_reading(readings, int(grid_id), run, fields["HGT_M"])
        reach[int(grid_id)] = fields["HGT_M"]
    adapted, acoustic = adapt_experiment_to_terrain(exp, readings)
    # THE LONG STEP each domain's ground and crest-level wind allow, read
    # off the start state and boundary data these inputs carry: every
    # prepared and met_em domain holds a prepared cache, and the wrfinput
    # door holds the files' own arrays and its wrfbdy.
    statics = {int(bundle.grid_id): bundle.static_fields
               for bundle in inputs.domains}
    readers = {int(bundle.grid_id): bundle.cache_reader
               for bundle in inputs.domains
               if getattr(bundle, "cache_reader", None) is not None}
    basis = None
    if readers:
        from woof.ingest.boundary_stream import ClockBasis

        # Kept for a head-bound run, which reads the same derivation again
        # as each root boundary interval arrives.
        basis = ClockBasis(experiment=adapted, acoustic=acoustic,
                           readers=MappingProxyType(dict(readers)),
                           statics=MappingProxyType(dict(statics)),
                           reach=MappingProxyType(dict(reach)))
        adapted, clock = clock_for_prepared_cache(
            adapted, acoustic, readers=readers, statics=statics,
            boundaries=getattr(inputs, "boundaries", None),
            corridors=reach)
    else:
        adapted, clock = clock_for_wrfinput(
            adapted, acoustic,
            restored={int(bundle.grid_id): bundle.restored
                      for bundle in inputs.domains
                      if getattr(bundle, "restored", None) is not None},
            statics=statics, boundaries=getattr(inputs, "boundaries", None),
            corridors=reach)
    extra = ({} if basis is None or not hasattr(inputs, "clock_basis")
             else {"clock_basis": basis})
    return replace(inputs, experiment=adapted,
                   acoustic_substeps=MappingProxyType(
                       acoustic_receipt(acoustic, clock)),
                   terrain_clock=MappingProxyType(clock_receipt(clock)),
                   **extra)


def _prepared_planning_nodes(inputs):
    """Price adaptive halos from the statics the cache restore will install.

    Store-first restores allocate from this decision before a full GPU state
    exists. Supply the complete host geometry now so their acoustic reach is
    the same as the live clock's, including maxima outside the first row slab.
    """
    exp = inputs.experiment
    nodes = streaming._config_tree_nodes(exp.domains)
    bundles = {bundle.grid_id: bundle for bundle in inputs.domains}
    for node in nodes:
        if (node.cfg.run.use_adaptive_time_step
                and streaming.options_for_domain(node.cfg, exp.tiles).enabled):
            static = bundles[node.cfg.grid_id].static_fields
            # DomainState.set_map_coriolis installs float32 values; rounding
            # here must agree at acoustic substep thresholds as well.
            node.state = SimpleNamespace(
                msfu=np.asarray(static["MAPFAC_U"], dtype=np.float32),
                msfv=np.asarray(static["MAPFAC_V"], dtype=np.float32))
    return nodes


def _correct_mode_receipt(configs):
    from woof.correct_mode import run_receipt
    return run_receipt(configs)


def _domain_rows(exp) -> list[dict[str, object]]:
    return [_with_physics_cadences(exp, domain, _with_history_window(domain, row))
            for domain, row in zip(exp.domains, _plain_domain_rows(exp))]


def _with_physics_cadences(exp, domain, row: dict[str, object]) -> dict[str, object]:
    """The row, plus each physics cadence wrf.exe rounds to whole steps.

    ``cudt = 5`` at a 45 s step runs every 7 steps (5.25 min), as in
    wrf.exe; the receipt names the minutes the scheme actually ran at.
    Absent when every cadence is a whole number of steps (or the clock is
    adaptive), so every receipt written before this is unchanged.
    """
    from woof.experiment import wrf_physics_cadences

    if getattr(domain.run, "use_adaptive_time_step", False):
        return row
    cadences = wrf_physics_cadences(domain.run, exp.dt_exact(domain.grid_id))
    if cadences:
        row["physics_cadences"] = cadences
    return row


def _with_history_window(domain, row: dict[str, object]) -> dict[str, object]:
    """The row, plus the history window where the domain sets one.

    Absent when unset, so every receipt written before the window existed
    is unchanged.
    """
    begin = float(getattr(domain, "history_begin_s", 0.0) or 0.0)
    end = getattr(domain, "history_end_s", None)
    if begin:
        row["history_begin_s"] = begin
    if end is not None:
        row["history_end_s"] = float(end)
    return row


def _plain_domain_rows(exp) -> list[dict[str, object]]:
    return [
        {
            "grid_id": int(domain.grid_id),
            "parent_id": int(domain.parent_id),
            "i_parent_start": int(domain.i_parent_start),
            "j_parent_start": int(domain.j_parent_start),
            "parent_grid_ratio": int(domain.parent_grid_ratio),
            "parent_time_step_ratio": int(domain.parent_time_step_ratio),
            "nx": int(domain.run.nx),
            "ny": int(domain.run.ny),
            "nz": int(domain.run.nz),
            "dx_m": float(domain.run.dx),
            "dy_m": float(domain.run.dy),
            "dt_s": float(domain.run.dt),
            "history_interval_s": float(domain.history_interval_s),
            "mp_physics": int(domain.run.mp_physics),
            "moist": bool(domain.run.moist),
            "moist_cq": bool(domain.run.moist_cq),
            "nest_microphysics_transition": str(
                domain.run.nest_microphysics_transition
            ),
        }
        for domain in exp.domains
    ]


#: The restart identity this runner binds, component by component.
#:
#: Until 1.4.1 the whole experiment TOML's SHA-256 was one of these
#: components, which made the tree route's restart identity strictly
#: narrower than the contract `woof run --restart` publishes: "only the
#: forecast length / output and restart cadence may differ".  All three
#: of those live in the TOML, so all three moved the digest and all three
#: were refused -- including extending `run_seconds` from a checkpoint,
#: the worked example in FIRST-LIGHT section 7.  The single-domain route
#: never had the problem: `woof.io.restart._require_config_match` diffs
#: the RunConfig field by field and skips exactly those keys
#: (`CONFIG_RUN_LENGTH_FIELDS`).
#:
#: The experiment component is now the same timing-independent identity
#: `woof.core.model.experiment_fingerprint` binds on the native route,
#: so the two restart contracts agree.  Everything else the digest bound
#: -- preparation receipt, per-domain prepared-cache content, execution
#: plan, runtime source identity -- is bound exactly as before.
TREE_RESTART_IDENTITY_COMPONENTS = (
    "schema", "experiment_identity", "preparation_receipt_sha256",
    "domain_cache_content_sha256", "execution_plan",
    "runtime_source_identity",
)
#: The same identity for a tree prepared chained: its head digest stands
#: for the proof, and the children's caches for the domain caches (the
#: root's cache is bound by the head and checked against it at the seal).
CHAINED_TREE_RESTART_IDENTITY_COMPONENTS = (
    "schema", "experiment_identity", "prepared_head_sha256",
    "domain_cache_content_sha256", "execution_plan",
    "runtime_source_identity",
)


def tree_restart_identity_components(
    inputs, runtime_identity, initialization=None
) -> dict[str, object]:
    """The named components whose digest is the tree restart fingerprint.

    Named, and stored beside the fingerprint in the checkpoint header,
    so a mismatch can say WHICH component differs.  A bare hash
    comparison could only ever say that something did -- which is what
    made the refusal a nine-word traceback with nothing actionable in it.
    """

    from woof.core.model import restart_identity_payload
    smoothed = {f"d{bundle.grid_id:02d}": dict(bundle.terrain_autosmooth)
                for bundle in inputs.domains
                if getattr(bundle, "terrain_autosmooth", None) is not None}
    smoothing_identity = {"terrain_autosmooth": smoothed} if smoothed else {}

    # An external-input adapter may separate observations about the
    # preparation's resource budget from its immutable scientific recipe.
    # Raw artifact digests remain verified by that adapter; source, cache,
    # configuration and runtime identity remain bound below.
    receipt_identity = getattr(initialization, 'preparation_receipt_sha256', None)
    head_sha256 = (None if initialization is not None
                   else getattr(inputs, "prepared_head_sha256", None))
    if head_sha256 is not None:
        # A CHAINED TREE is bound by its head whichever way this run binds
        # it (the head at launch, or the sealed proof that names it), so a
        # checkpoint written before the seal resumes under either binding.
        # The head binds the proof's start half, the root's start arrays
        # and every child's receipt; the seal is checked against it, so
        # the proof and root cache digests (which a head-bound run does
        # not have at launch) are not components here.  Each child is the
        # one the head prepared: an as-posted tree's sealed children carry
        # the manifest digest where the head's carry the input plan, so a
        # sealed binding names its head twins' digests.
        root_id = next(int(domain.grid_id)
                       for domain in inputs.experiment.domains
                       if domain.parent_id == 0)
        head_children = getattr(inputs, "head_child_content_sha256", None)
        return _strict_json({
            "schema": REPORT_SCHEMA,
            **smoothing_identity,
            "experiment_identity": restart_identity_payload(
                inputs.experiment),
            "prepared_head_sha256": head_sha256,
            "domain_cache_content_sha256": {
                f"d{bundle.grid_id:02d}": (
                    bundle.cache_reader.content_sha256
                    if head_children is None
                    else head_children[f"d{bundle.grid_id:02d}"])
                for bundle in inputs.domains
                if int(bundle.grid_id) != root_id
            },
            "execution_plan": _plan_restart_identity(inputs.execution_plan),
            "runtime_source_identity": runtime_identity,
        })
    preparation_sha256 = (receipt_identity() if callable(receipt_identity)
                          else inputs.authority_sha256["preparation_receipt"])
    # Strict-JSON at construction, not at hash time: these components are
    # also written into the checkpoint header, and a MappingProxyType or
    # a Path reaching json.dump there fails the checkpoint write itself.
    return _strict_json({
        "schema": REPORT_SCHEMA,
        **smoothing_identity,
        "experiment_identity": restart_identity_payload(inputs.experiment),
        "preparation_receipt_sha256": preparation_sha256,
        "domain_cache_content_sha256": {
            f"d{bundle.grid_id:02d}": (
                bundle.cache_reader.content_sha256 if initialization is None
                else initialization.domain_content_sha256(bundle))
            for bundle in inputs.domains
        },
        "execution_plan": _plan_restart_identity(inputs.execution_plan),
        "runtime_source_identity": runtime_identity,
    })


def _plan_restart_identity(plan) -> dict[str, object]:
    """The execution plan as RESTART identity: what it integrates.

    ``_domain_rows`` describes each domain for the receipt, and one of
    the things it describes is ``history_interval_s`` -- when the run
    writes.  Hashing the receipt as-is re-bound the output cadence the
    ``--restart`` contract publishes as free to change, which is the
    same defect one layer up from the prepared-cache identity.  The plan
    published in the report is unchanged; only this view drops it.
    """

    identity = _strict_json(plan)
    for row in identity.get("domains", ()):
        if isinstance(row, dict):
            for name in ("history_interval_s", "history_begin_s",
                         "history_end_s"):
                row.pop(name, None)
    return identity


#: THE ``[tiles]`` admission this door takes, and the same callable
#: ``woof run``'s tree route takes: it lives beside :func:`decide_tree`
#: in :mod:`woof.core.streaming` so the two run doors cannot grow two
#: implementations of one question.  Re-exported under this module's name
#: because this door was its first caller and its callers name it here.
cold_tree_streaming_decision = streaming.cold_tree_streaming_decision


def _external_door_streams(exp, initialization, refusal):
    """``exp`` with [tiles] mode = 'auto', when only the card refused it resident.

    Breakage this closes (X12): the WRF-input door (``woof run --wrfinput``,
    the deterministic leg of a 5 km CONUS DA cycle: 1080x636x49, HRRR recipe
    + GF) was refused on a 32 GB card at its resident envelope, and the
    refusal's remedy -- "[tiles] mode = 'auto' (or --tiles auto)" -- named
    settings this door does not have: its experiment is imported from
    ``namelist.input``, which carries no [tiles], and it takes no --tiles.
    So the default refused a run the streamed road can hold.  When the
    resident forecast is the only thing that refused and nothing configures
    a mode, the door streams instead and says so; the output is the same
    bytes ([tiles] binds no identity: :func:`woof.core.streaming
    .identity_payload_entry`).

    THE TILES GET WHAT THE RESTORED DOMAIN LEAVES.  An external door restores
    its domain resident and attaches the store from it (``store_ids`` is
    empty for an external initialization), and the attached state stays on
    the card beside the tile buffers.  So the planner is given the card's
    free memory less what that state holds -- the refusal's own model state,
    physics and lateral boundary terms -- as ``[tiles] vram_budget_bytes``;
    the card's whole free figure would plan buffers that only fit an empty
    card.

    ``None`` (keep the refusal) for the prepared door, which runs a config
    that can name its mode (``woof go`` re-states it there), when any
    [tiles] is configured, on the tree or on a domain, and when the refusal
    carries no card reading.
    """
    if initialization is None or refusal.free_bytes is None:
        return None
    options = getattr(exp, "tiles", None) or streaming.OFF
    if options.enabled or any(getattr(dc, "tiles", None) is not None
                              for dc in exp.domains):
        return None
    held = sum(int(refusal.terms.get(name, 0)) for name in (
        "model state", "physics", "lateral boundary tables"))
    # WHAT THE DOOR ITSELF PUTS ON THE CARD beside that state once the tree
    # is restored (the analysis door's IAU forcing and dry-mass increment,
    # woof.da.analysis_door.AnalysisInitialization.device_terms) is held
    # through the steps too, and is allocated after this plan is made.  Left
    # out, the tiles were planned in memory that was already spent:
    # camp-us-20230710-r n5q15 10:00 mem003 at 13.10 GiB free planned
    # 147x146 x12 tiles in 9.30 GiB and ran out of card memory on its first
    # step beside the door's 0.71 GiB.
    door_terms = getattr(initialization, "device_terms", None)
    door_terms = ({str(name): int(value) for name, value in dict(door_terms()).items() if int(value)}
                  if callable(door_terms) else {})
    door_held = sum(door_terms.values())
    budget = int(refusal.free_bytes) - held - door_held
    if budget <= 0:
        return None
    beside = ("" if not door_terms else
              f", and {door_held / 2**30:.2f} GiB the door holds beside it: "
              + ", ".join(f"{name} {value / 2**30:.2f} GiB"
                          for name, value in door_terms.items()))
    print(f"prepared tree: the resident forecast does not fit this card "
          f"({int(refusal.need_bytes) / 2**30:.2f} GiB envelope against "
          f"{int(refusal.free_bytes) / 2**30:.2f} GiB free), so it streams "
          "through host memory: [tiles] mode = 'auto' (identical output), "
          f"tiles within the {budget / 2**30:.2f} GiB the restored domain "
          f"leaves ({held / 2**30:.2f} GiB held{beside})", flush=True)
    return replace(exp, tiles=streaming.StreamingOptions(
        mode="auto", vram_budget_bytes=budget))


def resolve_execution_plan(exp) -> Mapping[str, object]:
    """Resolve every edge through the engine's actual transition authority.

    THE FEEDBACK REFUSAL THAT STOOD HERE IS LIFTED.  It said this product
    "carries a static one-way execution plan", and that was true of the
    ARTIFACTS -- initial states, statics and boundary series are authored
    identically at feedback = 0 and 1, because feedback is a runtime
    coupling behaviour, not an ingest one -- but not of the EXECUTOR,
    which builds the same ConcreteNestCoupler the native route does and
    runs the same clock table whose feedback positions were always
    present (woof/core/clock.py, WRF mediation_integrate.F:443/:523).
    With the coupler's transaction implemented end to end (restriction,
    the interp_fcn.F smoothers, windowed re-diagnosis), the accurate answer
    is the run.  The coupler still refuses the configurations feedback
    cannot serve (mixed microphysics, unequal nz, mismatched inventories)
    by name at construction.
    """
    by_id = {domain.grid_id: domain for domain in exp.domains}
    transitions = []
    for domain in exp.domains:
        if domain.parent_id == 0:
            continue
        parent = by_id[domain.parent_id]
        contract = resolve_microphysics_transition(parent.run, domain.run)
        transitions.append(
            {
                "source_domain": int(parent.grid_id),
                "target_domain": int(domain.grid_id),
                **dict(contract.receipt()),
            }
        )

    canonical_mixed = (
        len(exp.domains) == 4
        and [int(domain.parent_id) for domain in exp.domains] == [0, 1, 2, 3]
        and [int(domain.run.mp_physics) for domain in exp.domains] == [8, 8, 18, 18]
        and [str(domain.run.nest_microphysics_transition) for domain in exp.domains]
        == [
            "same-scheme-only",
            "same-scheme-only",
            MP8_TO_MP18_POLICY,
            "same-scheme-only",
        ]
    )
    mixed = [edge for edge in transitions if edge["mixed"]]
    warnings = [
        "The prepared domain-tree GPU route is implemented but does not yet "
        "have a retained public end-to-end HRRR acceptance run for this "
        "exact topology, source cycle, and physics selection."
    ]
    if mixed:
        warnings.append(
            "Mixed per-domain microphysics is a GPUWM extension and is not "
            "deterministically equivalent to stock WRF, which normalizes "
            "domains to one microphysics selector."
        )
    return MappingProxyType(
        {
            "schema": PLAN_SCHEMA,
            "plan_id": (
                THOMPSON_NSSL_PLAN_ID if canonical_mixed else ARBITRARY_PLAN_ID
            ),
            "status": "IMPLEMENTED_UNVERIFIED",
            "launch_allowed": True,
            "explicit_expert_consent_required": False,
            "domain_count": len(exp.domains),
            "domains": _domain_rows(exp),
            "transitions": transitions,
            "mixed_transition_count": len(mixed),
            # This receipt describes only the edge translation semantics.  It
            # must never be read as a claim that an otherwise arbitrary GPUWM
            # experiment is a certified stock-WRF trajectory.
            "microphysics_edges_stock_wrf_equivalent": not mixed,
            "whole_simulation_stock_wrf_certified": False,
            "warnings": warnings,
        }
    )


def _load_hierarchy_document(prepared_root: Path, expected_sha256: str):
    """Resolve the prepared root's top-level document, whichever source wrote it.

    Returns ``(path, document, source)``.  The digest is still pinned exactly;
    only which filename carries it varies.
    """

    # One note per FILE, not per (file, schema) candidate.  Several
    # sources write `proof.json`, and the GFS entry alone now has two
    # accepted schemas, so appending per candidate printed
    # `proof.json digest differs` once for each -- three or four
    # identical lines saying nothing about which file was read or what
    # its digest actually was.
    seen: dict[str, str] = {}
    for entry in _HIERARCHY_DOCUMENTS:
        candidate = prepared_root / entry["filename"]
        if not candidate.is_file():
            continue
        observed = _sha256(candidate)
        if observed != expected_sha256:
            seen[entry["filename"]] = (
                f"{candidate} has sha256 {observed}")
            continue
        document = _json_object(candidate, "preparation document")
        if document.get("schema") != entry["schema"]:
            seen.setdefault(
                entry["filename"],
                f"{candidate} carries schema "
                f"{document.get('schema')!r}")
            continue
        if document.get("status") != entry["status"]:
            raise ValueError(
                f"prepared root {entry['filename']} is not a "
                f"{entry['status']} {entry['source']} hierarchy"
            )
        source = entry["source"]
        if source == "mapped":
            from woof.stage_cli import packaged_source_of
            source = packaged_source_of(prepared_root) or "mapped"
        return candidate, document, source
    looked_for = ", ".join(
        sorted({e["filename"] for e in _HIERARCHY_DOCUMENTS}))
    if not seen:
        raise ValueError(
            f"prepared root {prepared_root} carries none of the documents a "
            f"hierarchy preparation writes ({looked_for}), so nothing there "
            f"can match --preparation-receipt-sha256 {expected_sha256}")
    raise ValueError(
        f"prepared root {prepared_root} carries no hierarchy document "
        f"matching --preparation-receipt-sha256 {expected_sha256}; "
        + "; ".join(seen[name] for name in sorted(seen))
        + ".  Accepted schemas: "
        + ", ".join(sorted({e["schema"] for e in _HIERARCHY_DOCUMENTS})))


def _load_head_document(prepared_root: Path, head_sha256: str):
    """A chained tree's head and the proof it carries without its seal keys.

    Returns ``(path, document, source, head)``, where ``document`` is the
    head's ``basis.proof_head`` (the tree's proof less
    :data:`woof.ingest.boundary_stream.SEAL_ONLY_PROOF_KEYS`) and
    ``path`` is ``boundary-stream/head.json``.  Refused when the head fails
    its digest or the pin, when its preparation will never seal, or when
    it is a single domain's head.
    """

    from woof.ingest.boundary_stream import (
        HEAD_NAME, LAYOUT_DOMAIN_TREE, BoundaryStreamError, bind_head,
        proof_document_name, stream_dir)
    from woof.stage_cli import packaged_source_of

    try:
        head = bind_head(prepared_root, head_sha256)
    except BoundaryStreamError as error:
        raise ValueError(str(error)) from None
    tree = head["basis"].get("tree")
    if not isinstance(tree, Mapping) or tree.get("layout") \
            != LAYOUT_DOMAIN_TREE:
        raise ValueError(
            f"the prepared head in {prepared_root} is a single domain's "
            "head, which the single-domain runner binds")
    document = dict(head["basis"]["proof_head"])
    # The document the seal writes, which the head names: proof.json, or a
    # native HRRR tree's receipt.json.  Matched with its schema, as the
    # sealed binding matches them.
    name = proof_document_name(head)
    for entry in _HIERARCHY_DOCUMENTS:
        if entry["filename"] != name \
                or document.get("schema") != entry["schema"]:
            continue
        if document.get("status") != entry["status"]:
            raise ValueError(
                f"the prepared head in {prepared_root} is not a "
                f"{entry['status']} {entry['source']} hierarchy")
        source = entry["source"]
        if source == "mapped":
            source = packaged_source_of(prepared_root) or "mapped"
        return stream_dir(prepared_root) / HEAD_NAME, document, source, head
    raise ValueError(
        f"the prepared head in {prepared_root} carries schema "
        f"{document.get('schema')!r}, which is not a hierarchy proof this "
        "runner reads")


def _hierarchy_valid_time(document) -> str:
    """The initialization time, however this source's document records it."""

    value = document.get("valid_time")
    if isinstance(value, str):
        return value
    times = document.get("forcing_times")
    if isinstance(times, list) and times and isinstance(times[0], str):
        return times[0]
    raise ValueError("preparation document declares no initialization time")


def _forcing_hours(receipt, identity) -> tuple[int, ...]:
    raw = receipt.get("forcing_hours")
    if raw is None:
        raw = identity.get("forcing_hours")
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or raw[0] != 0
        or any(later <= earlier for earlier, later in zip(raw, raw[1:]))
    ):
        raise ValueError(
            "prepared hierarchy forcing_hours must be increasing integers "
            "beginning at zero with at least two frames"
        )
    result = tuple(raw)
    deltas = {later - earlier for earlier, later in zip(result, result[1:])}
    if len(deltas) != 1:
        raise ValueError("prepared hierarchy forcing cadence is not uniform")
    return result


def _validate_domain_receipt(
    receipt,
    *,
    domain,
    bundle: Path,
    reader: PreparedCacheReader,
    static_path: Path,
    geometry_path: Path,
) -> None:
    expected_identity = {
        "schema": "gpuwm-native-domain-artifact-build-v1",
        "status": "READY",
        "grid_id": int(domain.grid_id),
        "parent_id": int(domain.parent_id),
        "boundary_mode": (
            "external-specified" if domain.parent_id == 0 else "nested-parent-forced"
        ),
    }
    if any(receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"d{domain.grid_id:02d} artifact receipt identity differs")
    artifacts = receipt.get("artifacts")
    with np.load(static_path, allow_pickle=False) as archive:
        static_fields = sorted(archive.files)
        from woof.static.terrain_autosmooth import verify_receipt
        smoothing = verify_receipt(
            _json_object(geometry_path, "geometry receipt").get("terrain_autosmooth"),
            archive["HGT_M"], domain_id=domain.grid_id)
    expected = {
        "prepared_cache": {
            "path": "prepared-cache",
            "content_sha256": reader.content_sha256,
            "payload_bytes": reader.payload_bytes,
            "array_count": len(reader.arrays),
        },
        "static_cache": {
            "path": "native-static.npz",
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
            "fields": static_fields,
        },
        "geometry_receipt": {
            "path": "geometry-receipt.json",
            "sha256": _sha256(geometry_path),
            "geometry": _json_object(geometry_path, "geometry receipt").get("geometry"),
        },
    }
    if smoothing is not None:
        expected["static_cache"]["terrain_autosmooth"] = smoothing
    if artifacts != expected:
        raise ValueError(f"d{domain.grid_id:02d} artifact hashes differ from its files")
    verification = receipt.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "PASS"
        or verification.get("content_sha256") != reader.content_sha256
        or verification.get("array_count") != len(reader.arrays)
        or verification.get("payload_bytes") != reader.payload_bytes
    ):
        raise ValueError(f"d{domain.grid_id:02d} cache verification receipt differs")
    if receipt_basename(verification.get("path", "")) != "prepared-cache":
        raise ValueError(f"d{domain.grid_id:02d} cache receipt path is not relocatable")
    if bundle.resolve() != geometry_path.parent.resolve():
        raise RuntimeError("domain artifact path escaped its bundle")


_announce_adopted_coordinate = prepared_single._announce_adopted_coordinate


def _validate_vertical(reader: PreparedCacheReader, exp, grid_id: int) -> None:
    metadata = reader.header.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"d{grid_id:02d} cache metadata is missing")
    eta = tuple(float(value) for value in exp.vertical.eta_levels)
    observed = reader.read_array("coord/znw")
    if observed.shape != (len(eta),) or not np.array_equal(
        observed.astype(np.float64), np.asarray(eta, dtype=np.float64)
    ):
        raise ValueError(
            f"d{grid_id:02d} prepared eta coordinate differs from the experiment config"
        )
    base = metadata.get("base_scalars")
    if not isinstance(base, dict) or float(base.get("p_top", -1.0)) != float(
        exp.vertical.p_top
    ):
        raise ValueError(f"d{grid_id:02d} prepared p_top differs from the experiment")
    # The cache restores its OWN c1f..c4h, so the etac beside them is what
    # the model will integrate.  Hold it to the one this run adopted, and
    # hold that one to this domain's own prepared columns: a coordinate
    # that orders the parent and not the nest is exactly the disagreement
    # the adoption exists to make impossible.
    refusal = prepared_domain_coordinate_refusal(
        label=f"d{grid_id:02d}", vertical=exp.vertical,
        coord_scalars=metadata.get("coord_scalars") or {},
        base_arrays={"mub": reader.read_array("base/mub")}
        if "base/mub" in reader.arrays else {})
    if refusal is not None:
        raise ValueError(refusal)


def _validate_delayed_prepared_geometry(exp) -> None:
    """A dated cache needs the same fixed parent ground it was built on."""
    from woof.experiment import delayed_domain_ids
    from woof.static.corridor import moving_grid_ids

    movers = moving_grid_ids(exp)
    for gid in delayed_domain_ids(exp):
        parent_id = exp.domain(gid).parent_id
        while parent_id:
            if parent_id in movers:
                raise ValueError(
                    f"prepared delayed child d{gid:02d} has moving ancestor "
                    f"d{parent_id:02d}; its dated analysis must be prepared "
                    "on the ancestor's live activation footprint, which "
                    "this fixed prepared cache does not supply")
            parent_id = exp.domain(parent_id).parent_id


def _validate_delayed_prepared_time(exp, domain, reader, receipt) -> None:
    """Require an owned child analysis, never infer its date from the root."""
    if exp.domain_start_offset_exact(domain.grid_id) == 0:
        return
    expected = exp.domain_start_time(domain.grid_id).isoformat()
    stamps = {
        "prepared cache initial_valid_time": reader.header.get(
            "metadata", {}).get("user", {}).get("initial_valid_time"),
        "domain receipt valid_time": receipt.get("valid_time"),
    }
    for label, value in stamps.items():
        if value != expected:
            raise ValueError(
                f"d{domain.grid_id:02d} {label} {value!r} differs from "
                f"its delayed start_time {expected}; prepare this child "
                "from the analysis at its activation time")


class StreamedClockGuard:
    """Keeps a head-bound run on the clock a sealed run of its tree chooses.

    The long step each domain runs is derived from the strongest
    crest-level wind in the start states AND in the root's boundary data
    over the whole forecast (:mod:`woof.terrain_clock`).  A run started
    on a prepared head sees only the start states; the boundary intervals
    arrive while it integrates.  The breakage this prevents: a head-bound
    forecast published as the sealed one after stepping with a longer
    step than the terrain map holds for the winds its boundaries bring.
    So each interval, as it loads, is folded into the reading, and a
    changed clock ends the head-bound attempt by name
    (:class:`woof.ingest.boundary_stream.StreamedClockChanged`).  Call it
    with each loaded interval.  It lives here, on the forecast side,
    because the preparation package stages ``boundary_stream`` without
    the forecast's terrain clock.
    """

    def __init__(self, basis, *, run_clock, root_grid_id: int,
                 run_seconds: float):
        from woof.ingest.boundary_stream import derived_clock
        from woof.terrain_clock import (
            MemoStartWinds, boundary_geometry_from_cache,
            start_winds_from_cache)

        self.basis = basis
        self.root = int(root_grid_id)
        self.run_seconds = float(run_seconds)
        self.expected = derived_clock(run_clock)
        # Read once each: the start states do not move while the forecast
        # runs, and each boundary instant is read at the interval that
        # brings it, not again at every later one (MemoStartWinds,
        # MemoBoundaryWinds).
        self.starts = {
            int(gid): MemoStartWinds.wrap(
                start_winds_from_cache(reader, f"d{int(gid):02d}"))
            for gid, reader in basis.readers.items()}
        self._boundary_memo: dict = {}
        reader = basis.readers.get(self.root)
        self.geometry = (None if reader is None else
                         boundary_geometry_from_cache(
                             reader, basis.statics.get(self.root)))
        self.intervals: dict[float, object] = {}
        #: How many intervals were folded into the reading.
        self.checked = 0
        from woof.terrain_clock_local import StreamedFaceCache

        self.local_cache = StreamedFaceCache()

    def __call__(self, interval) -> None:
        from woof.ingest.boundary_stream import (
            StreamedClockChanged, derived_clock)
        from woof.ingest.lateral_bc import LateralBoundaries
        from woof.terrain_clock import (
            MemoBoundaryWinds, clock_for_domains, clock_receipt)

        if self.geometry is None:
            # The sealed derivation reads no boundary winds either
            # (terrain_clock.clock_for_prepared_cache), so nothing a later
            # interval carries can move its clock.
            return
        self.intervals[float(interval.start_seconds)] = interval
        ordered = tuple(self.intervals[key] for key in sorted(self.intervals))
        boundary = MemoBoundaryWinds(
            f"d{self.root:02d}", LateralBoundaries(ordered, 1, 1, 1),
            self.geometry, self.run_seconds,
            memo=getattr(self, "_boundary_memo", {}))
        _, clock = clock_for_domains(
            self.basis.experiment, self.basis.acoustic,
            statics=self.basis.statics, starts=self.starts,
            boundary=boundary, corridors=self.basis.reach, announce=False,
            local_cache=getattr(self, "local_cache", None))
        self.checked += 1
        derived = derived_clock(clock_receipt(clock))
        if derived == self.expected:
            return
        changed = sorted(gid for gid in set(derived) | set(self.expected)
                         if derived.get(gid) != self.expected.get(gid))
        raise StreamedClockChanged(
            f"boundary interval {float(interval.start_seconds):g} s to "
            f"{float(interval.end_seconds):g} s carries a crest-level wind "
            "that moves the terrain-derived clock of "
            + ", ".join(f"d{gid:02d}" for gid in changed)
            + " away from the clock this run started on, so the forecast "
            "so far is not the one the sealed preparation gives")


class TreeHeadNeedsSeal(Exception):
    """A tree bound at its head reads something only its seal writes.

    Not a refusal: the runner says the reason, waits for the preparation
    to seal and runs as a sealed binding (:func:`main`).
    """


def _head_needs_seal(exp, *, relocation_follow: bool,
                     head_corridor: bool = False) -> str | None:
    """Why a head-bound tree starts after the seal, or ``None``.

    Each reason names what the forecast reads that the preparation writes
    only at its seal, which is the breakage waiting prevents: a forecast
    started on files that do not exist yet.

    A tree whose nests follow a source reads the statics corridor they
    re-ground over.  A chained preparation builds it into its head
    (``hierarchy-head/statics-corridor``, bound by the head's proof), so
    ``head_corridor`` is true and such a tree starts at its head; a head
    without one (prepared before heads carried the corridor) waits for
    the seal that writes it.
    """

    if relocation_follow and not head_corridor:
        return ("its nests follow a source, and this head carries no "
                "statics corridor for a moving nest to re-ground over, "
                "which the preparation's seal writes")
    return None


def _head_hierarchy(prepared_root: Path, head, exp):
    """The head's ``hierarchy-head/`` and each child's receipt, checked.

    Returns ``(hierarchy_root, receipts)`` with ``None`` for the root
    (whose static cache the head's root identity binds) and each child's
    ``receipt.json`` held to the digest the head binds
    (``basis.tree.children_receipts``).
    """

    from woof.ingest.boundary_stream import HIERARCHY_HEAD_DIRNAME

    tree = head["basis"]["tree"]
    labels = [f"d{int(domain.grid_id):02d}" for domain in exp.domains]
    if list(tree.get("domains") or ()) != labels:
        raise ValueError(
            f"the prepared head's domains {tree.get('domains')} are not "
            f"this experiment's {labels}")
    if tree.get("children_artifacts") != HIERARCHY_HEAD_DIRNAME:
        raise ValueError(
            "the prepared head keeps its children outside "
            f"{HIERARCHY_HEAD_DIRNAME}/")
    root = f"{HIERARCHY_HEAD_DIRNAME}/domains/{labels[0]}"
    expected_root = {
        "prepared_cache": f"{root}/prepared-cache",
        "static_cache": f"{root}/native-static.npz",
        "geometry_receipt": f"{root}/geometry-receipt.json",
    }
    if (dict(tree.get("root") or {}) != expected_root
            or str(head["basis"]["cache"]["directory"])
            != expected_root["prepared_cache"]):
        raise ValueError(
            "the prepared head's root files are not the canonical "
            f"{root}/ layout")
    receipts = dict(tree.get("children_receipts") or {})
    if sorted(receipts) != sorted(labels[1:]):
        raise ValueError(
            "the prepared head does not bind every child's receipt "
            f"(binds {sorted(receipts)}, the tree has {labels[1:]})")
    hierarchy_root = _require_directory(
        prepared_root / HIERARCHY_HEAD_DIRNAME, "prepared head hierarchy")
    documents: list = [None]
    for label in labels[1:]:
        path = _require_file(
            hierarchy_root / "domains" / label / "receipt.json",
            f"{label} artifact receipt")
        if _sha256(path) != receipts[label]:
            raise ValueError(
                f"{label} receipt differs from the one the prepared head "
                "binds")
        documents.append(_json_object(path, f"{label} artifact receipt"))
    return hierarchy_root, documents


def _as_posted_head_children(prepared_root: Path, proof, bundles):
    """Each child's head-twin content digest, for a sealed as-posted tree.

    ``None`` unless ``proof`` is the sealed proof of an as-posted chained
    tree (it names its head, ``boundary_stream.head_sha256``, and records
    ``posting.as_posted``): any other chained tree's head and sealed
    children are the same caches.  An as-posted head prepares every child
    before the input manifest exists, so each head child carries the input
    plan where its sealed twin carries the manifest digest, and the two
    content digests differ by that identity alone.  A forecast bound to
    the head restores the head's children and its restart identity binds
    their digests; this run, bound to the seal, binds the same ones
    (:func:`tree_restart_identity_components`).

    The head the proof names is read with its digest checked, each sealed
    child is held to its head twin
    (:func:`woof.ingest.boundary_stream.verify_as_posted_tree_children`:
    the receipt the head binds and the cache it records, equal arrays,
    metadata and static files, the identity changed only where the
    manifest goes), and each child this preflight loaded must be the
    sealed twin that check read.  Refused when that cannot be shown,
    which is the breakage this prevents: a sealed run naming child
    digests nothing held to the head, under which a checkpoint written
    before the seal would resume on children that were not the head's.
    """

    posting = proof.get("posting")
    head_sha256 = (proof.get("boundary_stream") or {}).get("head_sha256")
    if head_sha256 is None or not (
            isinstance(posting, Mapping) and posting.get("as_posted")):
        return None
    from woof.ingest.boundary_stream import (
        BoundaryStreamError, read_head, verify_as_posted_tree_children)

    try:
        head = read_head(prepared_root, expected_sha256=str(head_sha256))
        if head["basis"].get("as_posted") is None:
            raise BoundaryStreamError(
                "the head it names was not prepared as posted")
        # The manifest digest is where the head says its proof names it
        # (``proof_manifest_key``, as verify_seal reads it): a mapped tree's
        # proof names it in its composition receipt, not at its top level.
        # Breakage it prevents: every sealed mapped as-posted tree was
        # refused here ("'None' is not one"), so its forecast stopped at
        # the seal (a 500 m nested tree on HRRR 16Z, 2026-10-29).
        named = proof
        for part in str(head["basis"]["as_posted"].get("proof_manifest_key")
                        or "input_manifest_sha256").split("."):
            named = named.get(part) if isinstance(named, Mapping) else None
        checked = verify_as_posted_tree_children(
            prepared_root, head=head, manifest_sha256=str(named))
    except BoundaryStreamError as error:
        raise ValueError(
            f"the sealed as-posted tree in {prepared_root} cannot be held to "
            f"the head its proof names: {error}") from None
    found = {}
    for bundle in bundles:
        if int(bundle.parent_id) == 0:
            continue
        label = f"d{int(bundle.grid_id):02d}"
        record = checked.get(label)
        if record is None or record["sealed_content_sha256"] \
                != bundle.cache_reader.content_sha256:
            raise ValueError(
                f"the sealed {label} this run restores is not the one held "
                "to its as-posted head twin")
        found[label] = str(record["head_content_sha256"])
    return MappingProxyType(found)


def _array_rows(reader) -> dict:
    """A cache reader's array table as plain JSON (for comparison)."""

    return json.loads(_canonical(dict(reader.arrays)))


def _seal_tree_inputs(inputs: PreparedTreeInputs, *,
                      stream=None) -> PreparedTreeInputs:
    """Wait for the tree's seal and bind the forecast to it.

    The root's streamed cache is checked against the head it was
    published under (:func:`woof.ingest.boundary_stream.verify_seal`:
    the sealed header's digest over the head's arrays plus every segment,
    the proof naming the head, every interval this run consumed
    unchanged).  The complete sealed preflight then runs on the sealed
    tree, and each domain the run restored from ``hierarchy-head/`` is
    held to its sealed ``hierarchy-artifacts/`` twin: the same static
    cache, the same geometry receipt, the same array table and cache
    content digest (the root's as sealed), the same identity.  The
    terrain-derived clock the run stepped on must be the one the sealed
    tree derives over its whole boundary series.
    """

    head = inputs.stream_head
    if head is None:
        return inputs
    from woof.ingest.boundary_stream import (
        StreamedIntervals, derived_clock, verify_seal)
    from woof.ingest.prepared_cache import PreparedCacheReader

    if stream is None:
        stream = StreamedIntervals(inputs.prepared_root, head=head)
    if not stream.sealed():
        print("prepared tree: waiting for the preparation to seal "
              f"({inputs.prepared_root})", file=sys.stderr, flush=True)
    stream.wait_sealed()
    sealed = verify_seal(inputs.prepared_root, head=head,
                         consumed=stream.consumed_markers())
    sealed_inputs = preflight_prepared_tree(
        **dict(inputs.preflight_arguments),
        preparation_receipt_sha256=sealed["proof_sha256"])
    if sealed_inputs.prepared_head_sha256 != str(head["head_sha256"]):
        raise RuntimeError(
            "the sealed tree's proof names another head than the one this "
            "forecast started from")
    root_id = int(inputs.experiment.domains[0].grid_id)
    posted = sealed.get("as_posted")
    root_cache = inputs.prepared_root / str(head["basis"]["cache"]["directory"])
    # As posted, the seal wrote the root's one-shot identity, which
    # verify_seal held to the head's (only the manifest digest changes).
    streamed_root = PreparedCacheReader(
        root_cache,
        expected_identity=(
            dict(head["basis"]["cache"]["identity"]) if posted is None
            else _json_object(root_cache / "header.json",
                              "sealed root cache header")["identity"]))
    if streamed_root.content_sha256 != sealed["content_sha256"]:
        raise RuntimeError(
            "the root's streamed cache header is not the one the seal "
            "check read")

    document_digests = None
    if posted is not None and head["basis"]["as_posted"].get(
            "document_bound_identity_keys"):
        from woof.ingest.boundary_stream import document_bound_digests

        manifest_path = inputs.prepared_root / str(
            head["basis"]["as_posted"]["manifest_path"])
        if _sha256(manifest_path) != posted["input_manifest_sha256"]:
            raise RuntimeError(
                "the sealed tree's input manifest changed after the seal "
                "check, so its document digests no longer bind this run")
        document_digests = document_bound_digests(
            head["basis"]["as_posted"],
            _json_object(manifest_path, "sealed tree input manifest"))

    def same_identity(started, bound) -> bool:
        if posted is None:
            return dict(started.cache_identity) == dict(bound.cache_identity)
        from woof.ingest.boundary_stream import (
            BoundaryStreamError, check_as_posted_identity)

        try:
            check_as_posted_identity(
                dict(started.cache_identity), dict(bound.cache_identity),
                plan_sha256=head["basis"]["as_posted"]["input_plan_sha256"],
                manifest_sha256=posted["input_manifest_sha256"],
                manifest_bound=head["basis"]["as_posted"][
                    "manifest_bound_identity_keys"],
                document_bound=document_digests)
        except BoundaryStreamError:
            return False
        return True

    for started, bound in zip(inputs.domains, sealed_inputs.domains):
        label = f"d{int(started.grid_id):02d}"
        root = int(started.grid_id) == root_id
        reader = streamed_root if root else started.cache_reader
        if not root and started.cache_reader.verify_all().get(
                "content_sha256") != started.cache_reader.content_sha256:
            raise RuntimeError(
                f"the head's {label} cache changed during the run")
        # As posted, a child the head prepared carries the plan where its
        # sealed twin carries the manifest digest, so the two content
        # digests differ by that identity alone: verify_seal held the
        # child's metadata, arrays and static files to its twin, and the
        # sealed digest it read must be the twin's.  The sealed binding
        # names this run's head child in its restart identity, so a
        # checkpoint this run writes resumes on the seal.
        content_same = (
            reader.content_sha256 == bound.cache_reader.content_sha256
            if posted is None or root else
            (posted.get("children") or {}).get(label, {}).get(
                "sealed_content_sha256")
            == bound.cache_reader.content_sha256
            and (sealed_inputs.head_child_content_sha256 or {}).get(label)
            == reader.content_sha256)
        differs = [
            name for name, same in (
                ("static cache",
                 _sha256(started.static_path) == _sha256(bound.static_path)),
                ("geometry receipt",
                 _sha256(started.geometry_receipt_path)
                 == _sha256(bound.geometry_receipt_path)),
                ("cache identity", same_identity(started, bound)),
                ("array table",
                 _array_rows(reader) == _array_rows(bound.cache_reader)),
                ("cache content digest", content_same),
            ) if not same]
        if differs:
            raise RuntimeError(
                f"the sealed tree's {label} differs from the head this "
                f"forecast restored it from: {', '.join(differs)}")
    started_corridor = dict(inputs.statics_corridor or {})
    sealed_corridor = dict(sealed_inputs.statics_corridor or {})
    if started_corridor.keys() != sealed_corridor.keys() or any(
            sealed_corridor[grid_id].cache_sha256 != corridor.cache_sha256
            for grid_id, corridor in started_corridor.items()):
        # A moving nest re-grounded over the head's corridor; the sealed
        # tree must carry the same ground.
        raise RuntimeError(
            "the sealed tree's statics corridor differs from the head's, "
            "which this forecast moved its nests over")
    if derived_clock(sealed_inputs.terrain_clock) \
            != derived_clock(inputs.terrain_clock):
        # Every interval passed the clock guard as it loaded, so this is
        # the last word on the same question over the whole series.
        raise RuntimeError(
            "the sealed tree derives a different terrain clock than the one "
            "this head-bound forecast ran on")
    return sealed_inputs


def _restore_streamed_child_at_start(owner, node, restore, build):
    """Restore a streamed delayed nest into a new host store at its start.

    ``owner`` is the stepper the nest was attached with at the forecast's
    start, over the store its startup restore filled, and ``node.state`` is
    that startup restore's :class:`woof.core.streamed_state
    .CanonicalStoreState`.  The owner is closed and the outgoing store let
    go (:func:`woof.core.streaming.release_outgoing_store`) BEFORE
    ``restore()`` allocates the store at the nest's start: while the
    startup view still held it, the nest's start held two copies of it.
    ``build(store_bundle, node)`` attaches the replacement inside the
    owner's reconstruction reservation, from a copy of ``node`` carrying
    the restored state, and the owner is rebound to that state, so the one
    stepper the executor holds for this grid steps the restored nest.
    Returns ``(store_bundle, restored)``.
    """
    owner.tiled_run.close()
    streaming.release_outgoing_store(owner)
    store_bundle, restored = restore()
    state = restored.initial_result.state
    temporary_node = copy(node)
    temporary_node.state = state
    with owner.allocation_scope():
        replacement = build(store_bundle, temporary_node)
    owner.rebind_after_reconstruction(replacement, state=state)
    return store_bundle, restored


def validate_physics_profile(exp, *, source: str, profile: str | None):
    """Assert one named suite on every domain; never rewrite configuration."""
    if profile is None:
        return None
    return prepared_single._validate_profile_switches(
        exp, source=source, profile=profile, all_domains=True)


def _refuse_dormant_without_corridor(exp, preparation) -> None:
    """A dormant nest needs a sealed corridor: its birth ground is a crop of it.

    Breakage it prevents: a spawn firing mid-run at a placement no static
    was prepared for, after hours of integration.  Named here, before
    anything restores, instead.
    """
    from woof.experiment import dormant_domain_ids
    dormant = [int(gid) for gid in dormant_domain_ids(exp)]
    if not dormant:
        return
    corridor_set = preparation.get("statics_corridor")
    covered = (corridor_set.get("domains")
               if isinstance(corridor_set, Mapping) else None)
    missing = [f"d{gid:02d}" for gid in dormant
               if not isinstance(covered, Mapping)
               or f"d{gid:02d}" not in covered]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} declare(s) spawn, but this prepared bundle "
            "seals no statics corridor for them: a nest born at a placement "
            "chosen mid-run takes its ground from its corridor on this route. "
            "Re-prepare with --statics-corridor (woof go passes it for a "
            "config whose nests follow a storm), or give the slot a follow table.")


def preflight_prepared_tree(
    *,
    prepared_root: Path,
    preparation_receipt_sha256: str | None = None,
    experiment_config: Path,
    experiment_config_sha256: str,
    physics_profile: str | None = None,
    prepared_head_sha256: str | None = None,
    devices: int | None = None,
    devices_options=None,
    simulated_radar=None,
) -> PreparedTreeInputs:
    """Verify the complete hierarchy and resolve a runnable CPU-only plan.

    Exactly one binding: the sealed preparation document
    (``preparation_receipt_sha256``) or a chained preparation's head
    (``prepared_head_sha256``).  A head binds the root's start state from
    its streamed cache and every child from ``hierarchy-head/``, each
    checked here as a sealed tree's domains are; the root's boundary
    intervals are hash-checked as they arrive and the seal is bound at
    the end of the run (:func:`_seal_tree_inputs`).
    """

    preflight_arguments = MappingProxyType(dict(
        prepared_root=prepared_root, experiment_config=experiment_config,
        experiment_config_sha256=experiment_config_sha256,
        physics_profile=physics_profile, devices=devices,
        devices_options=devices_options,
        **({} if simulated_radar is None else {"simulated_radar": simulated_radar})))
    if (preparation_receipt_sha256 is None) == (prepared_head_sha256 is None):
        raise ValueError(
            "a prepared tree binds its sealed preparation receipt or its "
            "prepared head, exactly one of the two")
    head = None
    experiment_config_sha256 = _digest(
        experiment_config_sha256, "experiment-config-sha256"
    )
    prepared_root = _require_directory(prepared_root, "prepared root")
    experiment_config = _require_file(experiment_config, "experiment config")
    if prepared_head_sha256 is not None:
        receipt_path, preparation, prepared_source, head = \
            _load_head_document(
                prepared_root,
                _digest(prepared_head_sha256, "prepared-head-sha256"))
    else:
        preparation_receipt_sha256 = _digest(
            preparation_receipt_sha256, "preparation-receipt-sha256"
        )
        receipt_path, preparation, prepared_source = _load_hierarchy_document(
            prepared_root, preparation_receipt_sha256
        )
    if _sha256(experiment_config) != experiment_config_sha256:
        raise ValueError("experiment config differs from --experiment-config-sha256")

    mapped_paths = {}
    mapped_authority = None
    mapped_manifest = None
    mapped_files = {}
    as_posted = None if head is None else head["basis"].get("as_posted")
    if prepared_source in prepared_single._MAPPED_SOURCES \
            and as_posted is not None:
        # An as-posted head: the input manifest does not exist until its
        # seal, so the head binds the manifest its input plan names, with
        # the plan's placeholder where the manifest digest goes, as the
        # single-domain door binds it (prepared_single
        # ._as_posted_head_binding).  The seal re-runs this preflight on
        # the sealed tree, which reads the manifest the seal wrote.
        # Breakage it prevents: every nested run on a mapped source
        # started while its cycle posted failed at its head with "mapped
        # input manifest does not exist" (WOOF run_f530d559497471b8,
        # 2026-10-29).
        manifest_digest = prepared_single._as_posted_head_binding(
            prepared_source, as_posted, None)
        mapped_manifest = dict(as_posted["input_plan"]["manifest"])
        mapped_files, _ = prepared_single._manifest_file_specs(
            prepared_source, mapped_manifest, None, preparation,
            pending=prepared_single._as_posted_pending_roles(
                as_posted, mapped_manifest),
            planned=True)
        paths, mapped_authority, _ = prepared_single._validate_packaged_mapped_evidence(
            prepared_root=prepared_root, proof=preparation,
            manifest=mapped_manifest, manifest_sha256=manifest_digest,
            experiment_config=None, wps_namelist=None, source=prepared_source,
            sealed=False, posted=as_posted)
        mapped_paths = dict(paths)
    elif prepared_source in prepared_single._MAPPED_SOURCES:
        manifest_path = _require_file(
            prepared_root / "source-evidence" / "input-manifest.json",
            "mapped input manifest")
        mapped_manifest = _json_object(manifest_path, "mapped input manifest")
        manifest_digest = _sha256(manifest_path)
        mapped_files, _ = prepared_single._manifest_file_specs(
            prepared_source, mapped_manifest, None, preparation)
        paths, mapped_authority, _ = prepared_single._validate_packaged_mapped_evidence(
            prepared_root=prepared_root, proof=preparation,
            manifest=mapped_manifest, manifest_sha256=manifest_digest,
            experiment_config=None, wps_namelist=None, source=prepared_source,
            sealed=head is None)
        mapped_paths = {"mapped_manifest": manifest_path, **paths}
        if head is not None and head["basis"].get("input_manifest_sha256") \
                not in (None, manifest_digest):
            raise ValueError(
                "the prepared head names a different source manifest than "
                "the one its evidence carries")

    exp = load_experiment(experiment_config)
    # [devices] on a tree: the bound TOML's own table, or the split handed in
    # beside it (woof sim --devices-table / --devices; the bundle binds the
    # TOML's bytes, so a split cannot be asked for by editing it).
    if devices_options is not None:
        exp = replace(exp, devices=devices_options)
    if devices is not None:
        from woof.core.devices import override_device_count
        exp = replace(exp, devices=override_device_count(exp.devices, devices))
    from woof.simulated_radar_config import apply_execution_options
    exp = apply_execution_options(exp, simulated_radar)
    if exp.simulated_radar.enabled:
        # Before the model is built or a card allocated, not at the first
        # history: a missing or stale rw_simradar, or a scan the host
        # memory cannot hold, refuses here.
        from woof.simulated_radar_config import require_admitted
        require_admitted(exp)
    from woof.core.devices import validate_device_road, validate_tree_devices
    validate_device_road(exp.devices, getattr(exp, "tiles", None), exp.domains)
    validate_tree_devices(exp)
    from woof.free_threading import refuse_multi_card_under_gil
    refuse_multi_card_under_gil(exp.devices.count, "prepared tree forecast")
    # THE COORDINATE THE PREPARED INPUTS CARRY, before anything derived
    # from the configuration's own etac exists: the per-domain identity
    # comparison below, a streamed tile buffer's rebuilt coordinate, a
    # nest spawned mid-run, a relocated child's re-initialization.
    exp, _prepared_vertical = adopt_prepared_vertical(
        exp, preparation, announce=_announce_adopted_coordinate)
    if len(exp.domains) < 2:
        raise ValueError("prepared domain-tree runner requires at least two domains")
    profile_assertion = validate_physics_profile(
        exp, source=prepared_source, profile=physics_profile)
    # Same governance one line down from the relocation refusal, and for
    # the same reason: this route restores children from prepared caches,
    # so it neither reserves a dormant nest's VRAM nor evaluates its
    # trigger.  A declared spawn must not integrate as a silently absent
    # nest.
    # Dormant (spawn-declared) nests are reserved and spawned on this route:
    # run_prepared_tree builds the startup tree without them and walks the
    # run in legs (woof.runtime.walk_spawn_legs), cropping each newborn's
    # statics from its sealed corridor (runtime.build_prepared_tree_spawn_runner,
    # which refuses by name a dormant slot the bundle sealed no corridor for).
    _refuse_dormant_without_corridor(exp, preparation)
    _validate_delayed_prepared_geometry(exp)
    # NO streaming refusal for [tiles]: this route wires a streamed-domain
    # builder (streaming.builders_for_tree, below), so mode = 'on' is
    # supported.  The refusal that stood here was written before the wiring
    # and outlived it, rejecting the one mode that asks for streaming
    # unconditionally.  A NEST that fires is still refused, at build time,
    # by prepared_domain_builder -- which is the right place for it, because
    # a tree whose nests fit resident must not be refused for having nests.
    # THE predicate, asked rather than restated.  This runner is the
    # door the whole corridor mechanism exists to satisfy, and it used
    # to hold its own copy of the sentence -- so a reading of
    # "[relocation] with moves but no follow" that the preparation doors
    # agreed on could have differed here, in the one place that decides
    # whether a bundle is accepted.  Supersedes the inline
    # exp.relocation.enabled and (follow is not None or moves) test.
    from woof.static.corridor import config_declares_follow_source
    relocation_follow = config_declares_follow_source(exp)
    if head is not None:
        sealed_need = _head_needs_seal(
            exp, relocation_follow=relocation_follow,
            head_corridor=isinstance(
                preparation.get("statics_corridor"), Mapping))
        if sealed_need is not None:
            raise TreeHeadNeedsSeal(sealed_need)
    if relocation_follow:
        # The [static.highres] idiom: an enabled surface refuses on the
        # lanes that cannot honor it.  A prepared tree deliberately runs
        # WITHOUT its ingest inputs, so a relocated child's statics
        # cannot be rebuilt from a GEOG source at runtime.  What CAN
        # honor a follow source here is the sealed statics corridor:
        # child-resolution statics over the ground the nest can reach, emitted
        # at preparation time (--statics-corridor) with its digest bound
        # into the preparation document, and cropped per footprint at
        # runtime through the same rebuild machinery the case-data route
        # wires.  A bundle prepared without one still refuses -- a
        # config that says "follow the storm" must not integrate as a
        # silently static nest.
        from woof.static.corridor import moving_grid_ids
        unknown_movers = moving_grid_ids(exp) - {
            int(domain.grid_id) for domain in exp.domains}
        if unknown_movers:
            raise ValueError(
                f"follow names grid_id(s) {sorted(unknown_movers)} "
                "which are not domains of this experiment")
        corridor_set = preparation.get("statics_corridor")
        if not isinstance(corridor_set, Mapping):
            raise ValueError(
                "[relocation] configures a follow source, but this "
                "prepared bundle carries no statics corridor: the "
                "prepared domain-tree route runs without the case's "
                "static (GEOG) source and "
                "cannot rebuild a relocated child's statics for a new "
                "footprint at runtime.  Re-prepare the tree with "
                "--statics-corridor (the tree preparation front door "
                "seals child-resolution statics over the ground the nest "
                "can reach, which this runner then crops per move), run "
                "the case-data route (woof run), or remove "
                "[relocation.follow]/[[relocation.move]] from this "
                "config.")
        corridor_domains = corridor_set.get("domains")
        # EVERY relocating grid needs its corridor, not just the tracked
        # mover: a [relocation.containment] parent slides too, and every
        # descendant of a mover re-grounds -- the regrounder indexes its
        # corridor by grid id at move time, so a gap here would surface
        # as a KeyError mid-run instead of a named refusal at the door.
        from woof.static.corridor import relocating_subtree_grid_ids
        needed = [f"d{gid:02d}"
                  for gid in relocating_subtree_grid_ids(exp)]
        missing = [label for label in needed
                   if not isinstance(corridor_domains, Mapping)
                   or label not in corridor_domains]
        if missing:
            covered = (sorted(corridor_domains)
                       if isinstance(corridor_domains, Mapping) else [])
            raise ValueError(
                f"[relocation] needs corridors for {needed}, but this "
                f"bundle's statics corridor covers only {covered} "
                f"(missing {missing}); re-prepare with --statics-corridor "
                "(the bare flag covers every child domain).")
    # The physics this config selects has to be RUNNABLE before the
    # hierarchy is verified, not after: this is the preflight, and a
    # missing lookup table is exactly the class of thing a preflight
    # exists to name before the GPU is touched.
    _verify_thompson_assets(exp)
    if _hierarchy_valid_time(preparation) != exp.start_time.isoformat():
        raise ValueError("preparation valid_time differs from experiment start_time")
    if preparation.get("domain_count") != len(exp.domains):
        raise ValueError("preparation domain count differs from experiment config")
    if head is None:
        hierarchy_root = _require_directory(
            prepared_root / "hierarchy-artifacts", "hierarchy artifact root"
        )
        artifact_receipt_path = _require_file(
            hierarchy_root / "receipt.json", "hierarchy artifact receipt"
        )
        artifact_manifest_path = _require_file(
            hierarchy_root / "domain-artifacts.json", "domain artifact manifest"
        )
        artifact_receipt = _json_object(artifact_receipt_path, "hierarchy artifact receipt")
        artifact_manifest = _json_object(artifact_manifest_path, "domain artifact manifest")
        if artifact_receipt != preparation.get("artifact_receipt"):
            raise ValueError(
                "published hierarchy artifact receipt differs from preparation"
            )
        expected_ids = [int(domain.grid_id) for domain in exp.domains]
        expected_manifest = {
            "schema": ARTIFACT_MANIFEST_SCHEMA,
            "domains": [
                {
                    "grid_id": grid_id,
                    "prepared_cache": f"domains/d{grid_id:02d}/prepared-cache",
                    "static_cache": f"domains/d{grid_id:02d}/native-static.npz",
                    "geometry_receipt": (f"domains/d{grid_id:02d}/geometry-receipt.json"),
                }
                for grid_id in expected_ids
            ],
        }
        if artifact_manifest != expected_manifest:
            raise ValueError("domain artifact manifest is not canonical")
        expected_artifact_identity = {
            "schema": ARTIFACT_RECEIPT_SCHEMA,
            "status": "READY",
            "domain_count": len(exp.domains),
            "grid_ids": expected_ids,
            "manifest": {
                "path": "domain-artifacts.json",
                "sha256": _sha256(artifact_manifest_path),
            },
            "boundary_inventory": {
                "external": [expected_ids[0]],
                "nested_parent_forced": expected_ids[1:],
            },
        }
        if any(
            artifact_receipt.get(key) != value
            for key, value in expected_artifact_identity.items()
        ):
            raise ValueError("hierarchy artifact receipt identity differs")
        domain_receipts = artifact_receipt.get("domains")
        if not isinstance(domain_receipts, list) or len(domain_receipts) != len(
            exp.domains
        ):
            raise ValueError("hierarchy domain receipt inventory is incomplete")
    else:
        # THE HEAD'S TREE: every child complete under hierarchy-head/, the
        # root's static files and its streamed cache beside them; each
        # child's receipt is bound by digest in the head (basis.tree).
        hierarchy_root, domain_receipts = _head_hierarchy(
            prepared_root, head, exp)
        artifact_receipt_path = None
        artifact_manifest_path = None

    grids = tuple(grids_from_projection_config(exp))
    if len(grids) != len(exp.domains):
        raise RuntimeError("experiment grid count differs from domains")
    # The authority triple binds every domain to one preparation. Sources that
    # record it at the top level are pinned against that; the rest are pinned
    # against the first domain's own cache identity, which the per-domain loop
    # below then requires every other domain to equal. Both enforce the same
    # invariant -- one authority across the whole tree -- so neither is weaker.
    provenance = preparation.get("provenance")
    if isinstance(provenance, dict):
        authority = {
            "bridge_manifest_sha256": provenance.get("bridge_manifest_sha256"),
            "source_manifest_sha256": preparation.get(
                "source_manifest_sha256", provenance.get("source_manifest_sha256")
            ),
            "namelist_sha256": provenance.get("native_namelist_input_sha256"),
        }
    else:
        if head is not None:
            # The root's identity is the head's cache identity until the
            # seal writes its header.
            first_identity = head["basis"]["cache"]["identity"]
        else:
            first = exp.domains[0]
            first_header = _json_object(
                hierarchy_root
                / "domains"
                / f"d{int(first.grid_id):02d}"
                / "prepared-cache"
                / "header.json",
                "d01 cache header",
            )
            first_identity = first_header.get("identity")
        if not isinstance(first_identity, dict):
            raise ValueError("d01 cache identity is missing")
        authority = {
            key: first_identity.get(key)
            for key in (
                "bridge_manifest_sha256",
                "source_manifest_sha256",
                "namelist_sha256",
            )
        }
    if mapped_authority is not None:
        # The original configuration's digest is preparation identity. The
        # current forecast may extend its stop time; typed per-domain checks
        # below still compare every setting which changes the prepared state.
        expected_authority = {
            "bridge_manifest_sha256": manifest_digest,
            "source_manifest_sha256": manifest_digest,
            "namelist_sha256": preparation["execution_inputs"][
                "experiment_config"]["sha256"],
        }
        if authority != expected_authority:
            raise ValueError("mapped hierarchy authorities differ from the preparation")
    authority = _bound_authority(authority, head)

    bundles = []
    common_source_identity = None
    forcing_hours = None
    # Which identity fields each domain's cache was accepted WITHOUT,
    # because they postdate the header and hold their not-in-use
    # default.  Normally empty; recorded either way, so a run on an
    # upgraded install can show exactly what it tolerated.
    tolerated_identity: dict[str, list[str]] = {}
    from woof.static.corridor import moving_grid_ids, relocating_subtree_grid_ids
    relocating_ids = relocating_subtree_grid_ids(
        exp, moving_roots=moving_grid_ids(exp))
    for domain, grid, embedded_receipt in zip(exp.domains, grids, domain_receipts):
        label = f"d{int(domain.grid_id):02d}"
        # A head-bound root: its static files and its streamed cache, whose
        # header the seal writes; the head carries its identity.
        head_root = head is not None and int(domain.parent_id) == 0
        bundle = _require_directory(
            hierarchy_root / "domains" / label, f"{label} bundle"
        )
        cache = _require_directory(bundle / "prepared-cache", f"{label} cache")
        static_path = _require_file(
            bundle / "native-static.npz", f"{label} static cache"
        )
        geometry_path = _require_file(
            bundle / "geometry-receipt.json", f"{label} geometry receipt"
        )
        if head_root:
            domain_receipt_path = None
            domain_receipt = {}
            header = {"identity": head["basis"]["cache"]["identity"]}
        else:
            domain_receipt_path = _require_file(
                bundle / "receipt.json", f"{label} artifact receipt"
            )
            domain_receipt = _json_object(
                domain_receipt_path, f"{label} artifact receipt")
            if domain_receipt != embedded_receipt:
                raise ValueError(
                    f"{label} receipt differs from hierarchy receipt")
            header = _json_object(
                cache / "header.json", f"{label} cache header")
        identity = header.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{label} cache identity is missing")
        # Default-tolerant, and only on the document that grows fields
        # as the configuration schema grows.  v1.1.0 added a per-domain
        # `start_time` for staggered nest starts, which made every
        # v1.0.1-era prepared tree unrunnable under a strict-equality
        # check -- refused with a sentence that named the user's
        # experiment file, when the cause was a package upgrade.
        # Both sides through the SAME normalization the hierarchy's root
        # binding uses.  They used to differ: the hierarchy gate pinned an
        # inactive cudt_minutes to 0 and this one compared it raw, so a
        # cumulus-off tree whose cache inherited RunConfig's live 5.0 was
        # refused against a wizard config that wrote the profile's 0.0 --
        # after preparation, on a switch no step of the run reads.  The
        # same table also drops the write cadences and the inert
        # diagnostic toggle, which say when a forecast writes rather than
        # what it integrates.
        tolerated_fields, differing_fields = compare_prepared_domain_config(
            effective_prepared_domain_config(identity.get("domain_config")),
            effective_prepared_domain_config(
                prepared_domain_config_identity(domain)),
            not_in_use=undelayed_identity_defaults(exp))
        if differing_fields:
            raise ValueError(prepared_identity_refusal(
                subject=f"{label} prepared cache", header=header,
                differing=differing_fields,
                re_prepare=(
                    "the front door that wrote this tree, against this "
                    "experiment config")))
        tolerated_identity[label] = list(tolerated_fields)
        for key, expected in authority.items():
            if identity.get(key) != expected:
                raise ValueError(f"{label} cache {key} differs from preparation")
        current_hours = _forcing_hours(preparation, identity)
        if identity.get("forcing_hours") != list(current_hours):
            raise ValueError(f"{label} cache forcing hours differ")
        if forcing_hours is None:
            forcing_hours = current_hours
        elif forcing_hours != current_hours:
            raise ValueError("prepared domain forcing hours differ")
        source_identity = identity.get("source_identity")
        if not isinstance(source_identity, dict) or source_identity.get(
            "grid_id"
        ) != int(domain.grid_id):
            raise ValueError(f"{label} source identity/grid binding differs")
        if mapped_authority is not None:
            prepared_single._validate_source_identity(
                prepared_source, source_identity, manifest_digest, mapped_files,
                preparation, layout="mapped-hierarchy-d01-v1",
                mapped_authority=mapped_authority, grid_id=int(domain.grid_id),
                experiment_config=experiment_config,
                experiment_config_sha256=experiment_config_sha256)
        normalized_source = dict(source_identity)
        normalized_source.pop("grid_id")
        if common_source_identity is None:
            common_source_identity = normalized_source
        elif common_source_identity != normalized_source:
            raise ValueError("prepared domain source identities differ")

        cache_payload = None
        if head_root:
            from woof.ingest.prepared_cache import PreparedHeadReader

            # The start-time half of the root's cache, each array held to
            # the head's manifest row; its content digest exists only at
            # the seal, where _seal_tree_inputs checks it.
            reader = PreparedHeadReader(
                prepared_root, head, expected_identity=identity)
            if reader.path.resolve() != cache.resolve():
                raise ValueError(
                    "the prepared head names another root cache directory "
                    "than the tree's hierarchy-head/")
            reader.verify_all()
        else:
            reader = PreparedCacheReader(cache, expected_identity=identity)
            _validate_delayed_prepared_time(
                exp, domain, reader, domain_receipt)
            # Retain only a wholly sealed tree. Head-bound child activation
            # keeps the existing streaming and seal-verification route.
            if head is None:
                cache_payload = prepared_single._sealed_preflight_payload(reader)
            verified_reader = (reader if cache_payload is None
                               else cache_payload.read_only_reader())
            verified = verified_reader.verify_all()
            if verified.get("content_sha256") != header.get("content_sha256"):
                raise ValueError(f"{label} cache content identity differs")
        _validate_vertical(reader, exp, int(domain.grid_id))
        lbc = reader.header.get("metadata", {}).get("lbc")
        if (domain.parent_id == 0 and not isinstance(lbc, dict)) or (
            domain.parent_id != 0 and lbc is not None
        ):
            raise ValueError(f"{label} external/nested LBC ownership differs")
        if domain.parent_id == 0:
            # The root's specified boundary: refused when it predates the
            # hydrometeors its source publishes, as the single-domain door
            # refuses it (prepared_single._refuse_stale_boundary).
            prepared_single._refuse_stale_boundary(
                domain.run, reader.header.get("metadata"),
                published=prepared_single._boundary_publication(
                    prepared_source, mapped_paths),
                source=prepared_source, prepared_root=prepared_root)
        geometry_receipt = verify_native_static_receipt(
            geometry_path, static_path, grid, domain.run,
            relocating=(domain.grid_id in relocating_ids
                        and not preparation.get("statics_corridor")))
        static = load_native_static_cache(
            static_path, grid, domain.run.ny, domain.run.nx
        )
        from woof.static.terrain_autosmooth import verify_receipt, run_line
        smoothing = verify_receipt(
            geometry_receipt.get("terrain_autosmooth"), static["HGT_M"],
            domain_id=domain.grid_id)
        if smoothing is not None:
            print(run_line(smoothing))
        if head_root:
            # No receipt at the head: the root's static cache is bound by
            # the cache identity the head's digest carries.
            if identity.get("static_cache_sha256") != _sha256(static_path):
                raise ValueError(
                    f"{label} static cache differs from the prepared head's "
                    "root identity")
            hashes = MappingProxyType({
                "static": _sha256(static_path),
                "geometry_receipt": _sha256(geometry_path),
            })
        else:
            _validate_domain_receipt(
                domain_receipt,
                domain=domain,
                bundle=bundle,
                reader=reader,
                static_path=static_path,
                geometry_path=geometry_path,
            )
            hashes = MappingProxyType(
                {
                    "cache_header": _sha256(cache / "header.json"),
                    "cache_content": reader.content_sha256,
                    "static": _sha256(static_path),
                    "geometry_receipt": _sha256(geometry_path),
                    "domain_receipt": _sha256(domain_receipt_path),
                }
            )
        bundles.append(
            PreparedDomainBundle(
                grid_id=int(domain.grid_id),
                parent_id=int(domain.parent_id),
                bundle=bundle,
                cache=cache,
                static_path=static_path,
                geometry_receipt_path=geometry_path,
                domain_receipt_path=domain_receipt_path,
                cache_reader=reader,
                cache_identity=MappingProxyType(identity),
                static_fields=MappingProxyType(static),
                authority_sha256=hashes,
                terrain_autosmooth=smoothing,
                cache_payload=cache_payload,
            )
        )

    if forcing_hours is None:
        raise RuntimeError("prepared hierarchy resolved no forcing schedule")
    if common_source_identity is None:
        raise RuntimeError("prepared hierarchy resolved no source identity")
    statics_corridor = None
    statics_corridor_cache_path = None
    if relocation_follow:
        # Digest-verified against the preparation document (whose own
        # digest the caller pinned), geometry-verified against this
        # experiment, grid-probe-verified against this machine's own
        # arithmetic.  A corridor that fails ANY of these refuses loudly
        # -- it never degrades to a silently static nest.
        from woof.static.corridor import (STATICS_CORRIDOR_DIRNAME,
                                           load_child_statics_corridor,
                                           planned_corridor,
                                           relocating_subtree_grid_ids)
        by_id = {int(domain.grid_id): index
                 for index, domain in enumerate(exp.domains)}
        corridor_directory = hierarchy_root / STATICS_CORRIDOR_DIRNAME
        # The mover AND every descendant of it: a mid-tree move changes
        # the ground under the whole subtree, so each member rebuilds its
        # own statics and each therefore needs its own corridor.  On a
        # leaf mover this is the single grid it always was.
        statics_corridor = {}
        for grid_id in relocating_subtree_grid_ids(exp):
            child = exp.domains[by_id[grid_id]]
            parent_run = exp.domains[by_id[int(child.parent_id)]].run
            # The sealed corridor must cover the ground THIS run's nest
            # can reach, which the loader checks before any move needs it.
            plan = planned_corridor(exp, child)
            statics_corridor[grid_id] = load_child_statics_corridor(
                corridor_directory,
                expected_set_receipt=preparation["statics_corridor"],
                grid_id=grid_id, child_dc=child, parent_run=parent_run,
                reference_grid=grids[by_id[grid_id]],
                sealed_child_statics=bundles[by_id[grid_id]].static_fields,
                frame_kwargs=plan.frame_kwargs,
                required_window=plan.window, reach=plan.reach)
        statics_corridor_cache_path = [
            corridor_directory / (
                preparation["statics_corridor"]["domains"]
                [f"d{grid_id:02d}"]["cache"]["path"])
            for grid_id in sorted(statics_corridor)]
    interval_hours = forcing_hours[1] - forcing_hours[0]
    from woof.experiment import validate_boundary_timing
    validate_boundary_timing(exp, int(interval_hours * 3600),
                             source="prepared tree forcing")
    if exp.run_seconds > forcing_hours[-1] * 3600.0:
        # The gate that keeps a longer run_seconds accurate.  A restart may
        # extend the forecast, but only into boundaries this tree was
        # actually prepared with; naming both numbers is what tells the
        # user whether to shorten the run or re-prepare from more forcing.
        raise ValueError(
            f"experiment run_seconds = {float(exp.run_seconds):g} s exceeds "
            f"the prepared forcing, which reaches "
            f"{forcing_hours[-1] * 3600.0:g} s after start_time (f"
            f"{forcing_hours[-1]:03d}); shorten the run or re-prepare the "
            "tree from a longer fetch")
    execution_plan = resolve_execution_plan(exp)
    authority_hashes = MappingProxyType(
        {
            **({"prepared_head": _sha256(receipt_path)} if head is not None
               else {
                   "preparation_receipt": _sha256(receipt_path),
                   "artifact_receipt": _sha256(artifact_receipt_path),
                   "artifact_manifest": _sha256(artifact_manifest_path),
               }),
            "experiment_config": _sha256(experiment_config),
            **{name: _sha256(path) for name, path in mapped_paths.items()},
        }
    )
    chained_head = (str(head["head_sha256"]) if head is not None
                    else (preparation.get("boundary_stream") or {}).get(
                        "head_sha256"))
    head_children = (None if head is not None else _as_posted_head_children(
        prepared_root, preparation, bundles))
    # THE ACOUSTIC SUBSTEPS EACH DOMAIN'S OWN GROUND NEEDS, on the inputs
    # every later reader takes its experiment from.
    return _with_terrain_acoustics(PreparedTreeInputs(
        prepared_root=prepared_root,
        hierarchy_root=hierarchy_root,
        preparation_receipt_path=receipt_path,
        artifact_receipt_path=artifact_receipt_path,
        artifact_manifest_path=artifact_manifest_path,
        experiment_config=experiment_config,
        experiment=exp,
        grids=grids,
        domains=tuple(bundles),
        forcing_hours=forcing_hours,
        boundary_interval_seconds=interval_hours * 3600,
        source_identity=MappingProxyType(common_source_identity),
        execution_plan=execution_plan,
        authority_sha256=authority_hashes,
        source=prepared_source,
        physics_profile_assertion=profile_assertion,
        mapped_authority_paths=MappingProxyType(mapped_paths),
        tolerated_identity_fields=MappingProxyType({
            label: tuple(names)
            for label, names in tolerated_identity.items()}),
        statics_corridor=statics_corridor,
        statics_corridor_cache_path=statics_corridor_cache_path,
        stream_head=(None if head is None else MappingProxyType(head)),
        prepared_head_sha256=(None if chained_head is None
                              else str(chained_head)),
        head_child_content_sha256=head_children,
        preflight_arguments=preflight_arguments,
    ))


def _resident_domain_reader(bundle, boundary_source=None):
    """Restore writable copies from this sealed domain's verified mappings."""
    if boundary_source is not None:
        return bundle.cache_reader
    payload = getattr(bundle, "cache_payload", None)
    if payload is None:
        return None
    payload.require_binding(bundle.cache, bundle.cache_identity,
                            reader=bundle.cache_reader)
    return payload.copying_reader()


def _prepared_input_bytes(inputs: PreparedTreeInputs) -> int:
    """Price the prepared cache reads without imposing fields on other doors."""
    return (sum(bundle.cache_reader.payload_bytes + bundle.static_path.stat().st_size
                for bundle in inputs.domains)
            + sum(path.stat().st_size
                  for path in inputs.statics_corridor_cache_path or ()))


def _verify_inputs_unchanged(inputs: PreparedTreeInputs) -> None:
    current = {
        "preparation_receipt": _sha256(inputs.preparation_receipt_path),
        "artifact_receipt": _sha256(inputs.artifact_receipt_path),
        "artifact_manifest": _sha256(inputs.artifact_manifest_path),
        "experiment_config": _sha256(inputs.experiment_config),
        **{name: _sha256(path)
           for name, path in inputs.mapped_authority_paths.items()},
    }
    if current != dict(inputs.authority_sha256):
        raise RuntimeError("prepared tree authorities changed during execution")
    if inputs.statics_corridor_cache_path is not None:
        # One per subtree member; each is re-hashed against the corridor
        # object that was actually cropped from during the run.
        for path, grid_id in zip(inputs.statics_corridor_cache_path,
                                 sorted(inputs.statics_corridor)):
            if _sha256(path) != inputs.statics_corridor[grid_id].cache_sha256:
                raise RuntimeError(
                    f"prepared statics corridor d{grid_id:02d} changed "
                    "during execution")
    for bundle in inputs.domains:
        observed = {
            "cache_header": _sha256(bundle.cache / "header.json"),
            "cache_content": bundle.cache_reader.verify_all()["content_sha256"],
            "static": _sha256(bundle.static_path),
            "geometry_receipt": _sha256(bundle.geometry_receipt_path),
            "domain_receipt": _sha256(bundle.domain_receipt_path),
        }
        if observed != dict(bundle.authority_sha256):
            raise RuntimeError(
                f"prepared d{bundle.grid_id:02d} inputs changed during run"
            )


_provenance_receipt = prepared_single._provenance_receipt


#: How long one ``git rev-parse`` may take before it is treated as
#: unavailable.  The identity is resolved once at launch and once at the
#: very END of a run, after every frame is written; a git that hangs
#: there would wedge a finished 72-hour forecast on its receipt.  Matches
#: ``runtime_manifest._GIT_TIMEOUT_S``, which asks git the same question.
_GIT_TIMEOUT_S = 30

#: Attempts per git question.  The failure this exists for is transient
#: and environmental -- a spawn that does not start -- not a repository
#: answering "no".  Two retries cost milliseconds on a healthy box and
#: nothing at all where there is no repository to ask (see
#: :func:`_git_head_query`).
_GIT_ATTEMPTS = 3


def _git_head_query(*arguments: str) -> str | None:
    """One ``git`` question about HEAD, answered or ``None``.

    Never raises.  ``None`` means ONLY "this process could not get an
    answer" -- ``git rev-parse`` prints a 40-character object id on
    success and can never legitimately produce ``None`` -- which is the
    distinction :func:`_runtime_source_identity_change` reads.

    Three hardenings over the bare ``subprocess.check_output(["git",
    ...])`` this replaced, each for a failure this product has already
    paid for once:

    * the executable is resolved through
      :func:`woof.provenance.git_executable`, because ``"git"`` is
      looked up in the CHILD's ``PATH`` and this project composes child
      environments in several places; a composed environment that drops
      the entry carrying git turns every identity question into
      ``FileNotFoundError`` at once;
    * a timeout, so the end-of-run recheck cannot hang a finished run;
    * ``OSError``/``SubprocessError`` rather than the
      ``FileNotFoundError``/``CalledProcessError`` pair, which let a
      ``PermissionError`` or a ``TimeoutExpired`` out of a function whose
      whole contract is to answer or shrug.
    """

    from woof.provenance import git_executable

    executable = git_executable()
    if executable is None:
        return None
    # No repository, no question.  Skipped rather than asked-and-retried
    # so a wheel install -- where ``git rev-parse`` answers 128 every
    # time, deterministically -- pays nothing for the retries below.
    if not (REPOSITORY_ROOT / ".git").exists():
        return None
    for attempt in range(_GIT_ATTEMPTS):
        try:
            return subprocess.check_output(
                [executable, *arguments],
                cwd=REPOSITORY_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_S,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            if attempt + 1 < _GIT_ATTEMPTS:
                time.sleep(0.1 * (attempt + 1))
    return None


def _head_commit_and_tree() -> tuple[str | None, str | None]:
    """``(commit, tree)`` for HEAD, with ``None`` for "could not ask"."""

    commit = _git_head_query("rev-parse", "HEAD")
    tree = _git_head_query("rev-parse", "HEAD^{tree}")
    if commit is None:
        # The commit is written down in ``.git`` in a format git has not
        # changed in its lifetime, so a process that cannot SPAWN git can
        # still say which commit is executing.  Worktree-aware, which
        # matters here because this project's agents work almost
        # exclusively in linked worktrees.  There is no such reader for
        # the tree id -- it lives inside the commit object -- so a git
        # outage still loses that half, and the comparison is built to
        # survive losing it.
        try:
            from woof.provenance import git_dir_identity

            identity = git_dir_identity(REPOSITORY_ROOT)
        except Exception:                               # noqa: BLE001
            identity = None
        if identity:
            commit = str(identity["commit_full"])
    return commit, tree


def _runtime_source_identity_change(
    before: Mapping[str, object], after: Mapping[str, object]
) -> str | None:
    """Name the component that moved, or ``None`` if nothing did.

    A plain ``before != after`` conflated two unrelated events, and the
    cheap one destroyed the expensive one.  Measured 2026-09-02: a
    four-hour two-domain forecast completed all 240 outer steps, wrote 62
    ``wrfout`` frames, and then died on ``forecast implementation changed
    during execution``.  Nothing had changed -- none of the five hashed
    files was modified, HEAD had not moved, no commit was made -- but the
    ``git rev-parse`` at the end of the run failed to run, which flips
    ``git_commit``/``git_tree`` from their real values to ``None``, and
    ``None != "8f3c..."`` is a difference the equality test cannot tell
    from a real one.  The run's receipt was replaced by
    ``evidence/failed-run-receipt.json``.  On a 72-hour forecast that is
    a very expensive way to lose nothing but a git hiccup.

    So the two halves are compared on their own terms:

    * ``gpuwm_version`` and ``source_sha256`` are compared ALWAYS and
      strictly.  They are read from bytes on disk, they are the strongest
      binding available, and they cannot fail to be resolved -- if the
      implementation's bytes moved, this says so.
    * ``git_commit``/``git_tree`` are compared only when BOTH ends
      resolved them.  ``None`` is unambiguous here (see
      :func:`_git_head_query`), so "git could not answer at one end" is
      reported as what it is -- an unanswered question -- rather than as
      a changed answer.

    That is narrower than the old test in exactly one case: a commit that
    lands mid-run, touches none of the five hashed files, AND coincides
    with a git failure at one end.  Every other real change still fails:
    a tracked edit to any hashed file moves ``source_sha256``, and a HEAD
    that moves while git works moves ``git_commit``.
    """

    if before == after:
        return None
    if before["gpuwm_version"] != after["gpuwm_version"]:
        return (f"gpuwm_version {before['gpuwm_version']!r} -> "
                f"{after['gpuwm_version']!r}")
    first = dict(before["source_sha256"])              # type: ignore[arg-type]
    second = dict(after["source_sha256"])              # type: ignore[arg-type]
    for name in sorted(set(first) | set(second)):
        if first.get(name) != second.get(name):
            return (f"source_sha256 {name} {first.get(name)} -> "
                    f"{second.get(name)}")
    for field in ("git_commit", "git_tree"):
        if before[field] is None or after[field] is None:
            continue
        if before[field] != after[field]:
            return f"{field} {before[field]} -> {after[field]}"
    return None


def _runtime_source_identity() -> Mapping[str, object]:
    files = (
        REPOSITORY_ROOT / "woof/core/model.py",
        REPOSITORY_ROOT / "woof/core/nest.py",
        REPOSITORY_ROOT / "woof/core/microphysics_transition.py",
        REPOSITORY_ROOT / "woof/core/kernels/nest_microphysics.cu",
        Path(__file__).resolve(),
    )
    # ``.as_posix()``, not ``str()``: this identity is hashed into
    # ``sealed_extension_fingerprint`` and published in checkpoint
    # headers, so a backslash key makes the SAME code fingerprint
    # differently on Windows than on Linux and a leg cannot extend
    # across machines.  On Linux the two spellings are byte-identical,
    # so no already-sealed fingerprint moves; only Windows converges.
    source_sha256 = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path)
        for path in files
    }
    # The KEYS and their meanings are unchanged: this mapping is hashed
    # into ``sealed_extension_fingerprint`` and into the tree restart
    # identity, both of which are published in checkpoint headers, so a
    # new field here would move every already-sealed fingerprint and
    # strand the legs that carry them.  Only the reliability of the two
    # git values improved.
    commit, tree = _head_commit_and_tree()
    return MappingProxyType(
        {
            "gpuwm_version": __version__,
            "git_commit": commit,
            "git_tree": tree,
            "source_sha256": source_sha256,
        }
    )


def _verify_thompson_assets(exp) -> None:
    """Byte-validate the mp8 tables wherever they resolve from.

    The two process-environment gates this used to demand
    (WOOF_EXPERIMENTAL_THOMPSON_MP8=1 and an explicit
    WOOF_THOMPSON_TABLE_ROOT) predate the packaging promotion: the
    canonical WRF v4.6.1 classic tables now ship as package data,
    `woof fetch-tables` stages the externalized one, and `woof doctor`
    byte-validates all four and reports no gaps.  Demanding the env vars
    anyway meant mp8 -- the wizard's own default at the time -- failed
    twice at runtime, on a machine doctor had just declared clean, with
    neither variable named anywhere in the docs.  What actually
    protected anything was the validation below, and it still runs on
    every launch.
    """

    if not any(domain.run.mp_physics == 8 for domain in exp.domains):
        return
    classic = [domain.run for domain in exp.domains if domain.run.mp_physics == 8]
    from woof.core.thompson_column import table_identity
    for cfg in classic:
        if cfg.thompson_version != "wrf_461":
            table_identity(cfg)
    if not any(cfg.thompson_version == "wrf_461" for cfg in classic):
        return
    from woof.core.thompson_contract import validate_table_assets
    from woof.table_assets import require_thompson_tables

    # Absence first, and in one sentence.  A wheel user reached this
    # line after paying for a fetch and three minutes of preprocessing
    # and got a five-frame FileNotFoundError naming a path inside
    # site-packages -- true, and useless.  require_thompson_tables says
    # which table and which command stages it; the byte validation
    # below is unchanged and still runs on every launch.
    root = require_thompson_tables()
    validate_table_assets(root)


def _rebind_rebuilt_state(state, workspace) -> None:
    if workspace is None:
        return
    for name in workspace.symbols:
        value = getattr(state, name, None)
        if value is not None:
            setattr(state, name, workspace.view(name, value.shape, value.dtype))


def _corridor_host_bytes(corridors):
    """Total host bytes across the corridors in play, or None.

    Same shape rule as :func:`_corridor_echo`: a mapping keyed by
    grid_id is the normal case once a moving subtree needs one corridor
    per member, and a bare corridor is still accepted.
    """
    if corridors is None:
        return None
    entries = (corridors.values() if hasattr(corridors, "values")
               else [corridors])
    return int(sum(c.host_bytes for c in entries))


def _corridor_echo(corridors):
    """Receipt echo for the statics corridors in play, one row each.

    Accepts the mapping the loader builds (grid_id -> corridor) and, for
    any caller still holding one, a bare corridor.  Returns None when
    there are none, so a bounds-only [relocation] echo is unchanged.
    """
    if corridors is None:
        return None
    entries = (corridors.values() if hasattr(corridors, "values")
               else [corridors])

    def row(corridor):
        geometry = corridor.geometry
        return {
            "grid_id": int(geometry["grid_id"]),
            "corridor_nx": int(geometry["corridor_nx"]),
            "corridor_ny": int(geometry["corridor_ny"]),
            "frame_grid_id": int(geometry.get("frame_grid_id", -1)),
            "cache_sha256": corridor.cache_sha256,
            "host_bytes": corridor.host_bytes,
        }

    rows = sorted((row(c) for c in entries), key=lambda r: r["grid_id"])
    return rows or None


def _completed_execution_report(model):
    """The accurate zero-work report for a restore that landed on the stop.

    Counters stay at zero because nothing was integrated, and the clocks
    are the restored clocks, which is what every downstream consumer of
    this report reads.  Fabricating step counts here would put a run's
    work on a process that did none of it.
    """
    from woof.core.clock import ExecutionReport

    clocks = {int(grid_id): node.clock
              for grid_id, node in model.nodes_by_grid_id.items()}
    return ExecutionReport(
        histories={grid_id: 0 for grid_id in clocks}, clocks=clocks)


def _write_failed_run_receipt(outdir, error) -> None:
    """The receipt a run that STARTED and died owes its caller."""
    evidence = outdir / "evidence"
    evidence.mkdir(exist_ok=True)
    from woof.stability_recovery import RECOVERY_RECEIPT
    recovery_path = evidence / RECOVERY_RECEIPT
    _atomic_json(
        evidence / "failed-run-receipt.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "stability_recovery_receipt": (
                str(recovery_path.resolve()) if recovery_path.is_file() else None),
        },
    )


def fingerprint_across_stored_chain(header, fingerprint_as_built, model) -> str:
    """The live restart fingerprint for a checkpoint written after a move.

    Relocation BOUNDS are byte-inert on a fingerprint; an executed MOVE
    binds, by folding each move receipt into the value
    (:func:`woof.core.nest_relocation.mark_fingerprint_across_move`).
    So a fresh build's identity is its own base folded over the history
    the checkpoint records, and that is the ordinary answer.

    THE CASE THIS FUNCTION EXISTS FOR.  The chain is anchored to the base
    the WRITING run computed, so folding it reproduces the stored value
    only when this build computes the identical base.  Anything that
    moves the base -- a widened exemption as much as a changed setting --
    makes EVERY relocation checkpoint ever written unresumable, with a
    message blaming the move history rather than the base.  Measured: a
    checkpoint refused even with its own original configuration restored
    exactly.

    The header carries the components the writing run hashed, so its base
    is recoverable: drop the ``relocation`` block (added as the chain was
    marked) and hash the rest.  Two things must then BOTH hold before
    that base is adopted.

    1. Folding the SAME records from THAT base reproduces the stored
       fingerprint.  This proves the checkpoint is self-consistent -- a
       forged or truncated chain cannot pass, because the records hash
       one-way into the result.

    2. The stored components and the live ones agree once both are
       normalised under this build's rules
       (:func:`woof.io.restart._identity_matches_under_current_rules`).

    (1) ALONE IS NOT IDENTITY, and adopting the replayed value on its
    strength was a hole rather than a widening: the replayed value is by
    construction the stored fingerprint, so assigning it hands the gate
    in :mod:`woof.io.restart` the header to compare against itself --
    and that gate is the only place a tree's preparation receipt, cache
    content, execution plan and runtime source identity are ever
    compared.  A checkpoint from a DIFFERENT prepared tree would have
    resumed silently, because its chain is self-consistent too: the
    chain proves the move history was not tampered with and says nothing
    about which tree made the moves.  (2) is what a foreign tree fails
    and what the widened-exemption case passes.
    """
    from woof.core.nest_relocation import mark_fingerprint_across_move
    from woof.io.restart import _identity_matches_under_current_rules
    from woof.runtime import restore_relocation_fingerprint_components

    chain = (header.get("relocation") or {}).get("record_sha256")
    if not chain:
        return fingerprint_as_built
    marked = fingerprint_as_built
    for record_sha256 in chain:
        marked = mark_fingerprint_across_move(marked, record_sha256)
    stored_fingerprint = header.get("experiment_fingerprint")
    if marked == stored_fingerprint:
        restore_relocation_fingerprint_components(model, chain)
        return marked
    stored_components = header.get("experiment_fingerprint_components")
    if not isinstance(stored_components, dict):
        return marked
    base_components = {key: value
                       for key, value in stored_components.items()
                       if key != "relocation"}
    replayed = hashlib.sha256(
        _canonical(_strict_json(base_components)).encode("utf-8")).hexdigest()
    for record_sha256 in chain:
        replayed = mark_fingerprint_across_move(replayed, record_sha256)
    if replayed != stored_fingerprint:
        return marked
    if not _identity_matches_under_current_rules(
            {"experiment_fingerprint_components": base_components}, model):
        return marked
    restore_relocation_fingerprint_components(model, chain)
    return replayed


def _root_bundle(inputs):
    """The root domain's bundle, found by the experiment's root grid id.

    Every door's bundles carry a ``grid_id``; not every door's carry a
    ``parent_id`` (the wrfinput and met_em bundles do not), so the root
    is named by the experiment, which every door carries.
    """
    root_id = next(int(domain.grid_id)
                   for domain in inputs.experiment.domains
                   if domain.parent_id == 0)
    return next(bundle for bundle in inputs.domains
                if int(bundle.grid_id) == root_id)


def tree_urban_columns(inputs) -> dict[int, int] | None:
    """Each BEP+BEM domain's urban columns, read off the land cover this
    door is about to restore (A176).

    The configuration door prices BEP+BEM's column workspace at every
    column urban because it runs before the land cover exists; this door
    holds it.  Each bundle's static LU_INDEX (a wrfinput's own LU_INDEX
    and FRC_URB2D, which its urban cold start reads), on the land-use
    dataset that domain's physics initializes from: the wrfinput and
    met_em doors' own attributes, the prepared caches' native identity.
    ``None`` when no domain runs BEP+BEM.

    A domain whose ground can change keeps the every-column bound: a mover
    and its subtree, which rebuild their statics for new ground, and a
    spawn nest and its subtree, placed when the trigger fires.  The land
    cover restored at the start is not the one such a domain can hold.
    """
    from woof.core.urban_state import (bem_workspace_counted,
                                        prepared_urban_columns)
    from woof.static.corridor import relocating_subtree_grid_ids

    exp = inputs.experiment
    spawned = frozenset(int(dc.grid_id) for dc in exp.domains
                        if getattr(dc, "spawn", None) is not None)
    moving = set(relocating_subtree_grid_ids(exp))
    if spawned:
        moving.update(relocating_subtree_grid_ids(exp, moving_roots=spawned))
    runs = {int(dc.grid_id): dc.run for dc in exp.domains}
    counts: dict[int, int] = {}
    for bundle in inputs.domains:
        run = runs.get(int(bundle.grid_id))
        if (run is None or int(bundle.grid_id) in moving
                or not bem_workspace_counted(run)):
            continue
        raw = getattr(getattr(bundle, "restored", None), "raw", None) or {}
        static = getattr(bundle, "static_fields", None) or {}
        selection = getattr(bundle, "geog_selection", None)
        attrs = (NATIVE_LANDUSE_IDENTITY if selection is None
                 else selection.landuse_global_attrs())
        if attrs.get("MMINLU") is None:
            continue
        count = prepared_urban_columns(
            run, raw.get("LU_INDEX", static.get("LU_INDEX")),
            landuse_dataset=str(attrs["MMINLU"]),
            frc_urb2d=raw.get("FRC_URB2D", static.get("FRC_URB2D")))
        if count is not None:
            counts[int(bundle.grid_id)] = count
    return counts or None


def _root_setup_fingerprint(node, stream=None):
    """Hash a root's complete setup using its domain geography.

    A store-backed state's slab template can report different map-factor
    or rotation flags from the whole domain. The restart setup rebuilds
    those flags from the full geography before checking the sealed setup.
    """
    from woof.state_serialization_contract import setup_fingerprint

    if stream is None:
        return setup_fingerprint(node.state)
    from tilestream.restart_stream import domain_header_view

    with domain_header_view(
            stream.restart_setup(), stream.template_state, stream.store,
            float(node.clock.elapsed_seconds)) as view:
        return setup_fingerprint(view)


def _priced_external_boundary_source(boundaries, source):
    """What the tree door prices a handed-in root boundary set from.

    The wrfinput and met_em doors hand the runner the root's lateral
    boundary set whole (``initialization.lateral_boundaries``), and its
    intervals' field tables are what the run holds, so the hydrometeor
    masses those tables name are what it prices, read through the same
    inventory reader a sealed cache's header is
    (:func:`woof.boundary_fields.sealed_boundary_species`).  ``source``
    answers only for a set whose intervals name no fields.
    """
    from woof.boundary_fields import sealed_boundary_species

    carried = sealed_boundary_species(
        [{"fields": list(getattr(interval, "fields", None) or ())}
         for interval in boundaries.intervals])
    return source if carried is None else carried


def _admit_devices_tree(exp, split_ids, *, forcing_intervals,
                        forcing_interval_seconds, source, stream_head=None):
    """Per-card memory admission for a split tree, before anything restores.

    Each split grid is priced the way the single-domain door prices one
    (``estimate_devices``: every slab's resident envelope, its packed seam
    bands and the store-building template), each resident grid as a
    resident domain on the first card, and the pinned host stores of the
    split grids together.  Refused (unless ``--no-memory-gate``) when any
    card or the host is over: the breakage it prevents is a CUDA or pinned
    host allocation failure part way through restoring the tree.
    """
    import cupy as cp
    from woof.core.devices import DevicesRefused, validate_device_count
    from woof.core.devices_memory import GIB, estimate_devices_tree
    from woof.core.preflight import host_available_bytes

    validate_device_count(exp.devices, cp.cuda.runtime.getDeviceCount())
    ids = list(dict.fromkeys(exp.devices.device_ids()))
    budgets = {}
    identities = {}
    from woof.core.device_probe import cuda_device_identity
    for dev in ids:
        with cp.cuda.Device(dev):
            budgets[dev] = int(cp.cuda.runtime.memGetInfo()[0])
            identities[dev] = cuda_device_identity(dev)
    budgets = prepared_single._devices_stream_budgets(
        budgets, stream_head, identities=identities)
    # The same pricing `woof check --devices` and the `woof go` gate
    # print before the download (devices_memory.estimate_devices_tree).
    estimate = estimate_devices_tree(
        exp, split_ids=split_ids, forcing_intervals=forcing_intervals,
        forcing_interval_seconds=forcing_interval_seconds, source=source,
        streaming_boundaries=stream_head is not None)
    cards = {row["card"]: int(row["total_bytes"]) for row in estimate["cards"]}
    host = int(estimate["host_bytes"])
    rows = estimate["grids"]
    host_budget = prepared_single._stream_host_budget(host_available_bytes(), stream_head)
    lines = []
    refused = False
    for dev in ids:
        over = cards[dev] > budgets[dev]
        refused |= over
        lines.append(f"card {dev}: {'REFUSED' if over else 'ADMITTED'}: "
                     f"{cards[dev] / GIB:.2f} GiB priced; free "
                     f"{budgets[dev] / GIB:.2f} GiB")
    host_over = host_budget is not None and host > host_budget
    refused |= host_over
    lines.append(f"host: {'REFUSED' if host_over else 'PRICED'}: pinned stores "
                 f"{host / GIB:.2f} GiB")
    verdict = "\n".join(lines)
    print("prepared tree: [devices] grids " + ", ".join(
        f"d{g:02d}" for g in split_ids) + " split on cards "
        f"{list(exp.devices.device_ids())}\n" + verdict, flush=True)
    overridden = False
    if refused:
        from woof.core.resident_admission import memory_gate_overridden
        if not memory_gate_overridden():
            raise DevicesRefused(
                "[devices] tree memory admission refused before anything "
                "restores: prevent a card or pinned host allocation failure "
                "part way through the tree\n" + verdict)
        overridden = True
        print("prepared tree: [devices] memory admission OVERRIDDEN by "
              "--no-memory-gate: the per-card figures are priced upper bounds "
              "and each card's own allocation now decides", flush=True)
    return {"split_grid_ids": list(split_ids), "grids": rows,
            "card_bytes": {str(dev): cards[dev] for dev in ids},
            "card_free_bytes": {str(dev): budgets[dev] for dev in ids},
            "host_bytes": host, "verdict": verdict, "refused": bool(refused),
            "overridden": overridden}


def _devices_tree_receipt(exp, split_ids, steppers, decisions, admission):
    """The split tree's receipt: options, admission, and per split grid the
    slab plan, the transport actually used and how output reached the host."""
    from woof.prepared_single_domain_forecast import _devices_output_road
    grids = {}
    for gid in split_ids:
        stepper = steppers.get(gid)
        run = getattr(stepper, "tiled_run", None)
        decision = decisions.get(gid)
        report = getattr(run, "transport_report", None)
        grids[f"d{gid:02d}"] = {
            "halo": getattr(decision, "halo", None),
            "detail": dict(getattr(decision, "detail", {}) or {}),
            "transport_report": report() if callable(report) else None,
            "output_road": _devices_output_road(run),
        }
    return {"options": exp.devices.to_json(), "admission": admission,
            "grids": grids}


def _both_observers(first, second):
    """One ``step_observer`` calling ``first`` (may be None) then ``second``."""

    def observe(**event):
        if first is not None:
            first(**event)
        second(**event)

    return observe


def _warm_start_resume(initialization, start_time):
    """The step log's restart instant for a WRF warm start, else ``None``.

    Read from the clocks the warm start opens the run at, so the run line
    and the step totals count this run's steps from the restart instant
    (NV recipe dry run, D7).
    """

    clocks = getattr(initialization, "warm_start_clocks", None)
    if not clocks:
        return None
    root = clocks[min(clocks)]
    return progress_log.StepLogResume.of({
        "model_seconds": (root.valid_time - start_time).total_seconds(),
        "steps": {int(grid): int(clock.itimestep)
                  for grid, clock in clocks.items()}})


def _settle_checkpoint(publisher) -> None:
    """Land a checkpoint already snapshotted, without masking a failure.

    Called on a failure path: the run's own error is the news, so a
    checkpoint write that failed too is reported on stderr, not raised.
    """
    try:
        publisher.wait()
    except BaseException as late:  # noqa: BLE001 - keep the first failure
        print("prepared tree: the last checkpoint could not be written "
              f"({type(late).__name__}: {late})", file=sys.stderr, flush=True)


from woof.hydro.runtime import forecast_scope as _hydro_forecast_scope


@_hydro_forecast_scope
def run_prepared_tree(
    inputs: PreparedTreeInputs,
    *,
    output_directory: Path,
    io_mode: str,
    restart: Path | None = None,
    health_debug: bool = False,
    sealed_forcing_extension: bool = False,
    observer=None,
    progress_options=None,
    initialization: TreeInitialization | None = None,
    first_products=None,
    ensemble_bootstrap=None,
    health_retry_products=None,
    schedule_dispatch=None,
    restored_callback=None,
    columns=None,
) -> dict[str, object]:
    """Restore the prepared domains and execute the existing tree engine.

    ``observer`` is an optional second consumer of this run's progress,
    for a caller driving the runner in-process rather than reading its
    ``progress.json`` from outside (:class:`woof.runplan.RunObserver`
    is one).  It receives every per-step progress event this runner
    already builds, and -- if it carries the ``output_committed`` hook
    -- each per-domain wrfout as it becomes durable.  ``None`` leaves
    every byte of this runner's behaviour unchanged.

    ``progress_options`` is :class:`woof.progress_log.ProgressOptions`.
    ``None`` means the DEFAULTS, which are on: one WRF-shaped
    ``Timing for main:`` line per model time step PER DOMAIN -- so a
    nest taking 36 substeps inside one root step prints 36 lines, as
    WRF does -- plus ``progress.jsonl`` and a frame-ready marker per
    durable history file.
    """

    from woof.ensemble.runtime_context import current_session
    ensemble_session = current_session()
    if ensemble_session is not None:
        return ensemble_session.run_prepared(
            run_prepared_tree, inputs, output_directory=output_directory, io_mode=io_mode,
            restart=restart, health_debug=health_debug,
            sealed_forcing_extension=sealed_forcing_extension, observer=observer,
            progress_options=progress_options, initialization=initialization,
            first_products=first_products, ensemble_bootstrap=ensemble_bootstrap,
            health_retry_products=health_retry_products,
            **({} if restored_callback is None else {"restored_callback": restored_callback}),
            **({} if schedule_dispatch is None else {"schedule_dispatch": schedule_dispatch}))

    if io_mode not in {"history", "none"}:
        raise ValueError("io_mode must be 'history' or 'none'")
    if io_mode == "none" and inputs.experiment.simulated_radar.enabled:
        raise ValueError("simulated radar requires io_mode='history': virtual beams read durable atmospheric columns")
    # Every regional route reaches this runner: compile the configuration's
    # whole recorded kernel set on every core now, while the domains are
    # restored, instead of one module at a time inside step 1 with the card
    # idle (woof.kernel_plan).  A door that started it earlier gets the
    # same record back.
    from woof import kernel_prewarm as _kernel_prewarm
    _kernel_prewarm.start_plan(inputs.experiment)
    from woof.output_disk import require_output_space, renderer_products

    require_output_space(
        inputs.experiment, output_directory, restart=restart, io_mode=io_mode,
        render_products=renderer_products(first_products, observer))
    _verify_thompson_assets(inputs.experiment)
    if initialization is not None:
        initialization.verify_inputs(inputs)
        # External adapters have their own time authorities. This prepared
        # cache activation hook cannot establish an external child's date.
        from woof.experiment import refuse_delayed_activation
        refuse_delayed_activation(inputs.experiment, "external initialization")
        if sealed_forcing_extension:
            raise ValueError("this initialization does not supply the sealed forcing-prefix identity required for horizon extension")

    # Doors that assemble their own inputs (the met_em and wrfinput
    # routes) reach the tree here without the preflight, so the
    # derivation is asked again; inputs that carry it pass unchanged.
    inputs = _with_terrain_acoustics(inputs)

    import cupy as cp

    # [devices] ON A TREE: the grids that run split, as resident slabs on
    # the configured cards; every other grid runs resident on the first of
    # them.  Empty -- and every line below the one-card tree it always was --
    # when the experiment does not split.
    from woof.core.devices import DevicesRefused, validate_tree_devices
    split_ids = tuple(int(g) for g in validate_tree_devices(inputs.experiment))
    if split_ids:
        if initialization is not None:
            raise DevicesRefused(
                "[devices] on a tree initialized by an external door (wrfinput, "
                "met_em) is refused: those doors restore every grid resident "
                "before the split could take them, so the grids the split exists "
                "for would be allocated whole on one card")
        cp.cuda.Device(inputs.experiment.devices.device_ids()[0]).use()

    from woof import runtime
    from woof.config import radiation_scheme_ids
    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher,
        default_cupy_probes,
        nvidia_smi_process_probes,
        process_memory_receipt,
    )
    from woof.core.health import StateHealthValidator, health_validator_for_domain
    from woof.core.model import (
        DomainNode,
        ExperimentState,
        ModelMemoryLedger,
        ModelRuntimeStatus,
        SharedRRTMGPChunkWorkspace,
        execute_experiment,
        publish_declared_experiment,
        uses_modern_rrtmgp_workspace,
    )
    from woof.core.nest import NestCoupler
    from woof.core.preflight import estimate_experiment
    from woof.core.state import (
        build_shared_dycore_state_workspace,
        build_shared_scratch_arena,
    )
    from woof.ingest.hrrr_physics import initialize_prepared_physics
    from woof.runtime import declared_constant_glw
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock
    from woof.ingest.prepared_cache import restore_prepared_cache
    from woof.io.restart import (checkpoint_placements,
                                  read_restart_header,
                                  read_tree_lifecycle_header,
                                  restore_tree_restart,
                                  write_tree_restart)
    from woof.io.wrfout import PerDomainWrfoutWriters
    from woof.state_digest import canonical_state_digest
    from woof.supervisor import validate_manifest_checkpoint

    outdir = Path(output_directory).resolve()
    evidence = outdir / "evidence"
    evidence.mkdir()
    progress_path = evidence / "progress.json"
    runtime._preparation_progress(observer, "restore-prepared-domain-tree")
    exp = inputs.experiment
    from woof.case_data import trace_gas_overrides_from_config
    trace_gas_overrides = trace_gas_overrides_from_config(
        inputs.experiment_config, expected_sha256=inputs.authority_sha256["experiment_config"])
    # Opened before the restore, so the first thing a driving script
    # reads is this run announcing itself.  The markers and the JSONL
    # land beside the OUTPUTS (outdir), not under evidence/: they are
    # what a consumer polls, not what a post-mortem reads.
    step_log = (progress_options or ProgressOptions()).open(
        outdir=outdir, start_time=exp.start_time,
        run_seconds=float(exp.run_seconds),
        # Decides whether the step records carry `dt`, and with it which
        # schema the stream declares.  Read from the ROOT because
        # use_adaptive_time_step is Registry scope 1 -- one scalar for
        # the whole run, not a per-domain choice.
        adaptive_dt=bool(exp.root.run.use_adaptive_time_step),
        resume=_warm_start_resume(initialization, exp.start_time))
    # The kernel cache as this run INHERITED it, before a single kernel
    # of this run's own is written into it.  Pure filesystem; asking
    # later reads a cache this run has already been filling, which is
    # how a card swap stayed silent.
    kernel_cache_census = scan_kernel_cache()
    started_total = time.perf_counter()
    timing: dict[str, float] = {}
    runtime_identity = _runtime_source_identity()
    _atomic_json(
        progress_path,
        {
            "schema": PROGRESS_SCHEMA,
            "status": "RESTORING_PREPARED_DOMAIN_TREE",
            "model_elapsed_seconds": 0.0,
            "requested_run_seconds": float(exp.run_seconds),
            "execution_plan": inputs.execution_plan,
        },
        heartbeat=True,
    )

    stream_head = getattr(inputs, "stream_head", None)
    from woof.core.device_probe import cuda_device_identity
    admission_identity = (cuda_device_identity(0) if not split_ids
                          and prepared_single._stream_producer_reserve(stream_head) else None)
    planning_machine = streaming.cold_planning_machine(exp)
    if (planning_machine is None and not split_ids
            and (prepared_single._stream_producer_reserve(stream_head, identity=admission_identity)
                 or prepared_single._stream_host_producer_reserve(stream_head))):
        planning_machine = streaming.cold_admission_machine(
            options=getattr(exp, "tiles", None))
    planning_machine = prepared_single._stream_reserved_machine(
        planning_machine, stream_head, identity=admission_identity)
    planning_exp = prepared_single._stream_reserved_experiment(
        exp, planning_machine, stream_head, identity=admission_identity)
    external_boundaries = getattr(initialization, "lateral_boundaries", None)
    # The root's boundary tables as this run will hold them.  Handed in
    # whole (the wrfinput and met_em doors), they are that set's own
    # tables: those doors' bundles carry no prepared-cache header to read,
    # and reading one here stopped both with AttributeError before step 0.
    # Otherwise they are the tables the root's cache about to be restored
    # carries, as the single-domain door prices its own.
    if external_boundaries is not None:
        retained_intervals = len(external_boundaries.intervals)
        priced_boundary = _priced_external_boundary_source(
            external_boundaries, inputs.source)
    else:
        root_reader = _root_bundle(inputs).cache_reader
        retained_intervals = len(
            root_reader.header["metadata"]["lbc"]["intervals"])
        priced_boundary = prepared_single._priced_boundary_source(
            root_reader, inputs.source)
    # The land cover this run restores prices BEP+BEM's column workspace,
    # in the ledger and at the admission below (A176).
    from woof.core.urban_state import urban_columns_line
    urban_columns = tree_urban_columns(inputs)
    urban_line = urban_columns_line(urban_columns)
    if urban_line is not None:
        print(f"prepared tree: {urban_line}", flush=True)
    # Correct mode: a domain whose schemes carry a finding no correction
    # covers says so here, once, as well as in the receipt.
    from woof.correct_mode import uncorrected_notice
    correct_mode_line = uncorrected_notice(
        {domain.grid_id: domain.run for domain in exp.domains})
    if correct_mode_line is not None:
        print(f"prepared tree: {correct_mode_line}", flush=True)
    estimate = estimate_experiment(
        exp, forcing_interval_seconds=inputs.boundary_interval_seconds,
        forcing_intervals=retained_intervals, lateral_boundaries=external_boundaries,
        urban_columns=urban_columns,
    )
    cold_decisions = {}
    cold_nodes = _prepared_planning_nodes(inputs)
    # THE RUN'S ONE ADMISSION.  Kept, not discarded: the build pass below
    # consumes this decision rather than asking again from the ledger
    # estimate, which is a different question against a different budget.
    cold_tree = cold_tree_streaming_decision(
        planning_exp, cold_nodes, machine=planning_machine, decisions=cold_decisions,
        source=priced_boundary, urban_columns=urban_columns)
    devices_admission = None
    if split_ids:
        # The split grids are priced per card, the resident ones on the
        # first card, all before anything restores; and the split grids are
        # store-direct (never resident on one card), so they join store_ids.
        devices_admission = _admit_devices_tree(
            exp, split_ids, forcing_intervals=retained_intervals,
            forcing_interval_seconds=inputs.boundary_interval_seconds,
            source=priced_boundary, stream_head=stream_head)
        for dc in exp.domains:
            if int(dc.grid_id) in split_ids:
                cold_decisions[dc.grid_id] = streaming.ranked_decision(
                    dc.run, exp.devices)
    if cold_tree is None and not split_ids:
        # NOTHING STREAMS, SO THE WHOLE TREE IS RESIDENT, and it is admitted
        # before the shared workspaces and the first restore allocate: with
        # no [tiles] block the tree walk above consults nothing, and a tree
        # too big for the card stopped in a CUDA out-of-memory part way
        # through its restores.  Priced on the tables this run holds -- the
        # door's own forcing cadence, retained interval count and external
        # boundary set, as `estimate` above, with the analysed hydrometeors
        # the root's cache carries where no external set is handed in -- on
        # the card's own profile, state and physics of every domain together.
        admission_machine = streaming.cold_admission_machine(
            planning_machine, options=getattr(exp, "tiles", None))
        if planning_machine is None:
            admission_machine = prepared_single._stream_reserved_machine(
                admission_machine, stream_head, identity=admission_identity)
        from woof.boundary_fields import source_boundary_species
        from woof.core.resident_admission import ResidentMemoryRefused
        try:
            streaming.admit_resident_road(
                exp, None, machine=admission_machine,
                estimate=(None if admission_machine is None else
                          estimate_experiment(
                              exp,
                              forcing_interval_seconds=(
                                  inputs.boundary_interval_seconds),
                              forcing_intervals=retained_intervals,
                              lateral_boundaries=external_boundaries,
                              profile=getattr(admission_machine,
                                              "device_profile", None),
                              boundary_species=source_boundary_species(
                                  priced_boundary),
                              urban_columns=urban_columns)),
                what="this prepared domain tree, held resident on the card")
        except ResidentMemoryRefused as refusal:
            streamed_exp = _external_door_streams(exp, initialization, refusal)
            if streamed_exp is None:
                raise
            exp = streamed_exp
            inputs = replace(inputs, experiment=exp)
            planning_machine = streaming.cold_planning_machine(exp)
            planning_exp = prepared_single._stream_reserved_experiment(
                exp, planning_machine, stream_head, identity=admission_identity)
            cold_decisions = {}
            cold_nodes = _prepared_planning_nodes(inputs)
            cold_tree = cold_tree_streaming_decision(
                planning_exp, cold_nodes, machine=planning_machine,
                decisions=cold_decisions, source=priced_boundary,
                urban_columns=urban_columns)
    store_ids = ({gid for gid, decision in cold_decisions.items() if decision.stream}
                 if initialization is None else set()) | set(split_ids)
    resident_domains = tuple(dc for dc in exp.domains if dc.grid_id not in store_ids)
    started = time.perf_counter()
    # The estimator and core.model allocate shared arenas only for a tree.
    # A single external domain keeps DomainState's own storage.
    # The whole tree rides along so a resident child under a STREAMED root
    # still sizes its force slots from that root (the mixed road the plan
    # report prices; looking the parent up among the resident domains
    # alone died here with KeyError).
    arena = build_shared_scratch_arena(resident_domains, exp.domains) if len(exp.domains) > 1 and resident_domains else None
    rebuilt = build_shared_dycore_state_workspace(resident_domains) if len(exp.domains) > 1 and resident_domains else None
    # The shared helper, never a local restatement of the predicate: the
    # persistent workspace exists only for the MODERN RTE+RRTMGP
    # adapter, and `any(radiation_scheme_ids == (4, 4))` is also true for
    # the legacy-RRTMG variant, which runs one domain at a time and holds
    # no workspace at all.  estimate_experiment() knows the difference,
    # so a tree run under legacy RRTMG allocated a workspace the
    # preflight had not priced and died on the memory-ledger drift guard
    # ("shared radiation allocation differs from preflight") -- exactly
    # the failure uses_modern_rrtmgp_workspace's docstring predicts.
    radiation_workspace = (
        SharedRRTMGPChunkWorkspace(
            nz=exp.root.run.nz, column_chunk=exp.column_chunk, p_top=exp.vertical.p_top
        )
        if uses_modern_rrtmgp_workspace(exp)
        else None
    )
    from woof.core.preflight import shared_scratch_arena_bytes, shared_dycore_state_workspace_bytes
    expected_scratch = (shared_scratch_arena_bytes(resident_domains, exp.domains) if len(exp.domains) > 1 and resident_domains else 0)
    expected_rebuilt = (shared_dycore_state_workspace_bytes(resident_domains) if len(exp.domains) > 1 and resident_domains else 0)
    if (0 if arena is None else arena.nbytes) != expected_scratch:
        raise RuntimeError("shared scratch allocation differs from preflight")
    if (0 if rebuilt is None else rebuilt.nbytes) != expected_rebuilt:
        raise RuntimeError("shared rebuilt-state allocation differs from preflight")
    if (
        radiation_workspace is not None
        and radiation_workspace.nbytes != estimate.workspace_bytes
    ):
        raise RuntimeError("shared radiation allocation differs from preflight")
    timing["allocate_shared_workspaces"] = time.perf_counter() - started
    ledger = ModelMemoryLedger(
        estimate=estimate,
        shared_scratch_arena_bytes=0 if arena is None else arena.nbytes,
        shared_dycore_state_workspace_bytes=0 if rebuilt is None else rebuilt.nbytes,
        radiation_workspace=radiation_workspace,
    )

    # Dormant (spawn-declared) nests are PRICED above -- the estimate, the
    # shared arena and the dycore workspace all span exp.domains -- and
    # integrate nothing until they fire: clocks, schedule and nodes come
    # from the startup tree (woof.experiment.pre_spawn_experiment), the
    # same split woof.core.model.build_experiment makes on the case route.
    from woof.experiment import dormant_domain_ids, pre_spawn_experiment
    dormant_ids = frozenset(int(gid) for gid in dormant_domain_ids(exp))
    startup_exp = pre_spawn_experiment(exp) if dormant_ids else exp
    clock = resolve_clock(
        startup_exp, lbc_interval_s=float(inputs.boundary_interval_seconds),
        # The declared tree's tick (dormant nests included), so a spawn
        # never re-means a tick the live clocks already count in.
        tick_den_floor=(resolve_clock(
            exp, lbc_interval_s=float(inputs.boundary_interval_seconds)).tick_den
            if dormant_ids else None))
    schedule = build_schedule(startup_exp, clock)
    clocks = clock.clocks()
    nodes = {}
    prepared = {}
    drivers = {}
    early_steppers = {}
    initial_reservations = {}
    initial_perturbation_receipts: list[dict[str, object]] = []
    started = time.perf_counter()
    # Arch-aware, like the single-domain road: the cache is keyed by
    # compute capability, so a box whose card changed has a cache full
    # of entries it cannot load and recompiles everything in silence.
    # The device is asked here rather than inside the notice module,
    # which stays a pure filesystem predicate.
    _compile_state = kernel_cache_state(
        compute_capability=current_compute_capability(),
        census=kernel_cache_census)
    compile_notice = _compile_state.notice
    if compile_notice is not None:
        step_log.announce_kernel_compile(
            reason=_compile_state.reason,
            compute_capability=_compile_state.compute_capability,
            cached_entries=_compile_state.entries,
            cached_entries_for_this_card=_compile_state.entries_for_capability)

    def initialize_cache_physics(restored, domain, grid, bundle):
        from woof.core.cam_ozone import (
            cam_ozone_setup, configure_cam_ozone, ozone_parent_for)
        from woof.core.radiation_composition import attach_modern_workspace

        cam = cam_ozone_setup(exp=exp, dc=domain, grid=grid)
        driver = initialize_prepared_physics(
            restored.initial_result, domain.run, restored.met,
            restored.surface, bundle.static_fields,
            native_landuse_identity(bundle.static_fields),
            grid, exp.domain_start_time(domain.grid_id),
            simulation_start_time=exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp),
            p_top=exp.vertical.p_top, column_chunk=exp.column_chunk,
            trace_gas_overrides=trace_gas_overrides,
            **({"cam_ozone": cam, "ozone_parent": ozone_parent_for(cam)}
               if cam is not None else {}))
        driver = configure_cam_ozone(restored.initial_result.state, domain.run,
                                    exp=exp, dc=domain, grid=grid)
        attach_modern_workspace(getattr(driver, "radiation_callable", None),
                                radiation_workspace)
        return driver

    bundles_by_id = {int(bundle.grid_id): bundle for bundle in inputs.domains}

    # A TREE BOUND AT ITS HEAD.  The root restores its start state from the
    # streamed cache and its boundary intervals arrive through the stream,
    # each hash-checked against its segment marker as it loads and folded
    # into the terrain clock's reading; the children restore from
    # hierarchy-head/, complete at the head.  The seal is bound at the end.
    stream_head = getattr(inputs, "stream_head", None)
    boundary_source = None
    boundary_stream_receipt = None
    seam_waits = None
    clock_guard = None
    #: The root's model time after its last completed step in this process
    #: (``None`` before the first), which dates a seam wait.
    stepped = {"model_elapsed_seconds": None}
    head_started = time.perf_counter()
    if stream_head is not None:
        if initialization is not None or sealed_forcing_extension:
            raise ValueError(
                "a tree bound at its prepared head takes no external "
                "initialization and no sealed forcing extension; bind the "
                "sealed preparation receipt instead")
        from woof.ingest.boundary_stream import (
            PRODUCER_NAME, WAIT_LOG_NAME, SeamWaits, stream_dir,
            streamed_boundaries)

        events = getattr(observer, "events", None)

        def wait_where(index):
            elapsed = stepped["model_elapsed_seconds"]
            if elapsed is None:
                return {"phase": "start", "interval": None,
                        "model_elapsed_seconds": None,
                        "model_valid_time": None}
            valid = exp.start_time + timedelta(seconds=float(elapsed))
            return {"phase": "seam", "interval": index,
                    "model_elapsed_seconds": float(elapsed),
                    "model_valid_time": valid.strftime("%Y-%m-%dT%H:%M:%SZ")}

        def publish_wait(block):
            # The wait while it lasts, dropped the moment it ends, so a
            # reader never sees a stale wait beside a stepping model.
            payload = {
                "schema": PROGRESS_SCHEMA,
                "status": "RUNNING",
                "model_elapsed_seconds": float(
                    stepped["model_elapsed_seconds"] or 0.0),
                "requested_run_seconds": float(exp.run_seconds),
            }
            if block is not None:
                payload["waiting"] = block
            _atomic_json(progress_path, payload, heartbeat=True)

        seam_waits = SeamWaits(
            emit=None if events is None else events.emit,
            observer=observer, publish=publish_wait, model_time=wait_where,
            say=lambda line: print(line, file=sys.stderr, flush=True),
            log_path=outdir / WAIT_LOG_NAME,
            producer_path=stream_dir(inputs.prepared_root) / PRODUCER_NAME)
        clock_basis = getattr(inputs, "clock_basis", None)
        if clock_basis is not None:
            clock_guard = StreamedClockGuard(
                clock_basis, run_clock=getattr(inputs, "terrain_clock", None),
                root_grid_id=int(exp.root.grid_id),
                run_seconds=float(exp.run_seconds))
        boundary_source = streamed_boundaries(
            inputs.prepared_root, head=stream_head, on_wait=seam_waits,
            validate=clock_guard, start_time=exp.start_time)
        boundary_stream_receipt = {
            "chained": True,
            "head_sha256": stream_head["head_sha256"],
            "head_decision": dict(stream_head.get("decision") or {}),
            "ready_at_start": boundary_source.intervals.ready_prefix(),
            "interval_count": len(boundary_source.intervals),
        }

    def restore_store_domain(domain, grid, source):
        from woof.ingest.prepared_store import store_from_prepared_cache
        from woof.ingest.reconstruction_store import ReconstructionReservation
        from woof.core.streamed_relocation import StreamedChildReconstruction
        from woof.core.uh_diag import declared_follower_slots
        from contextlib import nullcontext
        decision = cold_decisions[domain.grid_id]
        cap = decision.detail.get('reconstruction_default_allocator_bytes')
        reservation = initial_reservations.get(domain.grid_id)
        if cap is not None and reservation is None:
            cap = (int(cap)+int(decision.detail.get('corridor_claim_bytes', 0))+511)//512*512
            reservation = ReconstructionReservation(cap)
            initial_reservations[domain.grid_id] = reservation
        cells = (int(decision.tile_nx)+2*int(decision.halo))*(int(decision.tile_ny)+2*int(decision.halo))
        rows = max(1, min(int(domain.run.ny), cells//int(domain.run.nx)))
        slots = declared_follower_slots(exp.domains).get(int(domain.grid_id), ())
        perturbation = None
        perturbation_rows = []
        if (exp.perturbation is not None and restart is None
                and clocks[domain.grid_id].spec.start_ticks == 0):
            from woof.ingest.init_perturbation import build_initial_state_perturbation
            perturbation = build_initial_state_perturbation(exp.perturbation, grid,
                grid_id=int(domain.grid_id), require_containment=domain.parent_id == 0,
                cfg=domain.run)
        def physics(result, cfg, met, surface, static, landuse, slab_grid, valid_time,
                    *, row_start=None, domain_rows=None, **kwargs):
            from woof.core.radiation_composition import attach_modern_workspace
            from woof.core.cam_ozone import (
                cam_ozone_setup, configure_cam_ozone, ozone_parent_for)
            slab_domain = replace(domain, run=cfg)
            cam = cam_ozone_setup(exp=exp, dc=slab_domain, grid=slab_grid)
            if perturbation is not None:
                from woof.core.diagnostics import update_diagnostics
                update_diagnostics(result.state, cfg.hypsometric_opt)
                window_perturbation = copy(perturbation)
                rows = slice(int(row_start), int(row_start)+int(cfg.ny))
                window_perturbation._placed = tuple(replace(p,
                    horizontal_km=(None if p.horizontal_km is None else p.horizontal_km[rows]))
                    for p in perturbation._placed)
                perturbation_rows.append(window_perturbation.apply_to_state(
                    result.state, allow_empty=True))
            driver = initialize_prepared_physics(result, cfg, met, surface, static,
                landuse, slab_grid, valid_time, **kwargs,
                simulation_start_time=exp.start_time, p_top=exp.vertical.p_top,
                column_chunk=exp.column_chunk, trace_gas_overrides=trace_gas_overrides,
                **({'cam_ozone': cam, 'ozone_parent': ozone_parent_for(cam)}
                   if cam is not None else {}))
            driver = configure_cam_ozone(result.state, cfg, exp=exp,
                                         dc=slab_domain, grid=slab_grid)
            attach_modern_workspace(getattr(driver, 'radiation_callable', None), radiation_workspace)
            for slot in slots:
                result.state.scratch((cfg.ny, cfg.nx), slot)
            return driver
        with reservation.activate() if reservation is not None else nullcontext():
            bundle = store_from_prepared_cache(source.cache,
                expected_identity=dict(source.cache_identity), cfg=domain.run,
                static=source.static_fields,
                landuse_attrs=native_landuse_identity(source.static_fields),
                grid=grid, valid_time=exp.domain_start_time(domain.grid_id),
                rows_per_slab=rows, budget_bytes=decision.detail.get('host_claim_bytes'),
                constant_glw_wm2=declared_constant_glw(exp), physics_initializer=physics,
                **({"reader": source.cache_reader,
                    "boundary_source": boundary_source}
                   if boundary_source is not None and domain.parent_id == 0
                   else {}))
        if perturbation_rows:
            receipt = copy(perturbation_rows[0])
            receipt['bubbles'] = [dict(row) for row in receipt['bubbles']]
            for index, row in enumerate(receipt['bubbles']):
                row['cells_touched'] = sum(r['bubbles'][index]['cells_touched']
                                           for r in perturbation_rows)
                if row.get('applied') and row['cells_touched'] == 0:
                    raise ValueError(f"perturbation.bubbles #{index+1} touches zero cells "
                                     f"in domain d{domain.grid_id:02d}")
                for key in ('max_theta_added_k', 'max_qv_delta_kg_kg'):
                    values = [r['bubbles'][index][key] for r in perturbation_rows
                              if key in r['bubbles'][index]]
                    if values:
                        row[key] = max(values)
            initial_perturbation_receipts.append(receipt)
        def scratch(shape, slot, dtype=None):
            with reservation.activate() if reservation is not None else nullcontext():
                return cp.zeros(shape, dtype=np.float32 if dtype is None else dtype)
        state = StreamedChildReconstruction._facade(bundle.template, domain.run,
            bundle.store, bundle.geography, bundle.scalars, scratch_allocator=scratch)
        state.lateral_boundaries = bundle.boundaries
        return bundle, SimpleNamespace(initial_result=SimpleNamespace(
            state=state, base=bundle.base, coord=bundle.coord))

    def initialize_delayed_child(node, clock):
        # Native preparation selected this child's exact analysis and ran
        # the real-nest terrain/base-state SINT. Those parent operands are
        # fixed setup fields; moving ancestors are refused in preflight.
        # Restore again at birth so no pre-activation diagnostic or restart
        # work can become this child's initial condition.
        domain = node.cfg
        bundle = bundles_by_id[int(domain.grid_id)]
        _validate_delayed_prepared_time(
            exp, domain, bundle.cache_reader,
            _json_object(bundle.domain_receipt_path, "delayed child receipt"))
        if clock.ticks != clock.spec.start_ticks:
            raise RuntimeError("prepared child activation is off its start tick")
        if int(domain.grid_id) in early_steppers:
            owner = early_steppers[int(domain.grid_id)]
            # The startup slab template's driver, which this map alone
            # still names once the store is released, goes before the
            # restore allocates, as the resident branch's does below.
            drivers.pop(domain.grid_id, None)
            store_bundle, restored = _restore_streamed_child_at_start(
                owner, node,
                lambda: restore_store_domain(domain, node.grid, bundle),
                lambda store_bundle, temporary: streaming.store_domain_builder(
                    store_bundle, node=temporary, clock=clock)(
                        None, domain.run, cold_decisions[domain.grid_id]))
            state = restored.initial_result.state
            reservation = initial_reservations.get(domain.grid_id)
            owner._reconstruction_reservation = reservation
            owner._reconstruction_host_budget_bytes = cold_decisions[
                domain.grid_id].detail.get('host_claim_bytes')
            drivers[domain.grid_id] = store_bundle.template.physics
            case = SimpleNamespace(static_fields=bundle.static_fields,
                geog_selection=getattr(bundle, 'geog_selection', None),
                initial_result=restored.initial_result, streamed_store=store_bundle)
            return SimpleNamespace(grid=node.grid, state=state), case
        # The startup build's driver goes before the restore allocates its
        # replacement: this map is the one owner of it the executor's
        # release (woof.core.model._release_startup_build) cannot see, and
        # it kept a second child's physics on the card through activation.
        drivers.pop(domain.grid_id, None)
        restored = restore_prepared_cache(
            bundle.cache, expected_identity=dict(bundle.cache_identity),
            cfg=domain.run, static=bundle.static_fields,
            allow_nested_without_lbc=True,
            reader=_resident_domain_reader(bundle))
        _rebind_rebuilt_state(restored.initial_result.state, rebuilt)
        restored.initial_result.state._scratch_arena = arena
        drivers[domain.grid_id] = initialize_cache_physics(
            restored, domain, node.grid, bundle)
        case = SimpleNamespace(
            static_fields=bundle.static_fields,
            geog_selection=getattr(bundle, "geog_selection", None),
            initial_result=restored.initial_result)
        return (SimpleNamespace(grid=node.grid, state=restored.initial_result.state),
                case)

    for domain, grid, bundle in zip(exp.domains, inputs.grids, inputs.domains):
        if int(domain.grid_id) in dormant_ids:
            # Reserved, not built: the spawn walk materializes it from the
            # live parent when its trigger fires.
            continue
        initialized = None
        store_bundle = None
        if domain.grid_id in store_ids and initialization is None:
            store_bundle, restored = restore_store_domain(domain, grid, bundle)
        elif initialization is None:
            streamed = boundary_source is not None and domain.parent_id == 0
            restored = restore_prepared_cache(
                bundle.cache,
                expected_identity=dict(bundle.cache_identity),
                cfg=domain.run,
                static=bundle.static_fields,
                allow_nested_without_lbc=domain.parent_id != 0,
                reader=_resident_domain_reader(
                    bundle, boundary_source if streamed else None),
                **({"boundary_source": boundary_source} if streamed else {}),
            )
            if streamed:
                from woof.ingest.boundary_stream import keep_interval_check

                # A later attachment may replace the series' check; the
                # clock guard stays chained after it.
                keep_interval_check(boundary_source.intervals, clock_guard)
            if restored.surface is None:
                raise ValueError(
                    f"d{domain.grid_id:02d} prepared cache lacks canonical surface")
        else:
            initialized = initialization.restore_domain(
                domain, grid, bundle, start_time=exp.start_time,
                scratch_arena=arena, dycore_state_workspace=rebuilt)
            from woof.forecast_initialization import DomainInitialization
            if not isinstance(initialized, DomainInitialization):
                raise TypeError("input adapter must return DomainInitialization")
            restored = initialized
        if store_bundle is None:
            _rebind_rebuilt_state(restored.initial_result.state, rebuilt)
            restored.initial_result.state._scratch_arena = arena
        if domain.parent_id != 0:
            restored.initial_result.state._nest_restart_classification = "REBUILT"
        starts_at_t0 = clocks[domain.grid_id].spec.start_ticks == 0
        if exp.perturbation is not None and restart is None and not starts_at_t0:
            initial_perturbation_receipts.append({
                "grid_id": int(domain.grid_id), "applied": False,
                "reason": "delayed start: this domain initializes from the "
                          "analysis at its activation time, after the "
                          "perturbation instant"})
        if (exp.perturbation is not None and restart is None and starts_at_t0
                and store_bundle is None):
            # Configured initial-state theta bubbles (PROVENANCE D12).
            # The sealed caches stay the pure analysis; the bubbles are
            # added to the RESTORED state here, per domain, before
            # initialize_prepared_physics rederives p/al/alt from the
            # perturbed prognostics.  A resumed run applies nothing --
            # the checkpoint trajectory already carries the evolved
            # bubble.  Refusals (coarse-domain containment, zero cells)
            # fire here, before any GPU integration.
            from woof.core.diagnostics import update_diagnostics
            from woof.ingest.init_perturbation import (
                build_initial_state_perturbation)
            # The cache serializes p as the preparation left it, and a
            # prepare-only cache leaves the EOS to its consumer -- so
            # diagnose the RESTORED prognostics first; the bubble holds
            # that state.p while it rebalances the geopotential, and its
            # rh_preserve reads it.  initialize_prepared_physics
            # runs the same diagnostics again below, from the perturbed
            # prognostics, which is the order the seam wants anyway.
            update_diagnostics(
                restored.initial_result.state, domain.run.hypsometric_opt)
            applier = build_initial_state_perturbation(
                exp.perturbation, grid, grid_id=int(domain.grid_id),
                require_containment=domain.parent_id == 0, cfg=domain.run)
            initial_perturbation_receipts.append(
                applier.apply_to_state(restored.initial_result.state))
        # The first domain's physics initialization is where a first
        # run pays its one-time NVRTC compile (~100 s on a modern
        # card), and it used to pay it under the stale RESTORING
        # status -- the first field run of the published wheel watched
        # that silence and concluded a hang.  One line and one status
        # flip, only when the kernel cache says the compile is coming.
        if compile_notice is not None:
            print(compile_notice, flush=True)
            _atomic_json(progress_path, {
                "schema": PROGRESS_SCHEMA,
                "status": COMPILING_STATUS,
                "model_elapsed_seconds": 0.0,
                "requested_run_seconds": float(exp.run_seconds),
                "execution_plan": inputs.execution_plan,
            }, heartbeat=True)
            compile_notice = None
        if store_bundle is not None:
            driver = store_bundle.template.physics
        elif initialized is None:
            driver = initialize_cache_physics(restored, domain, grid, bundle)
        else:
            driver = initialized.initialize_physics()
            from woof.core.cam_ozone import configure_cam_ozone
            driver = configure_cam_ozone(restored.initial_result.state, domain.run,
                                        exp=exp, dc=domain, grid=grid)
            from woof.core.radiation_composition import attach_modern_workspace
            attach_modern_workspace(getattr(driver, "radiation_callable", None),
                                    radiation_workspace)
        parent = None if domain.parent_id == 0 else nodes[domain.parent_id]
        node = DomainNode(
            cfg=domain,
            grid=grid,
            state=restored.initial_result.state,
            clock=clocks[domain.grid_id],
            parent=parent,
            children=[],
            coupler=None,
        )
        node._started = starts_at_t0
        if parent is not None:
            node.coupler = NestCoupler(node, feedback=exp.feedback,
                                       smooth_option=exp.smooth_option)
        if store_bundle is not None:
            from contextlib import nullcontext
            reservation = initial_reservations.get(domain.grid_id)
            with reservation.activate() if reservation is not None else nullcontext():
                if int(domain.grid_id) in split_ids:
                    # The split grid's slabs, each built from the pinned
                    # store at its compute window on its own card; a nest's
                    # slabs window the rolling tables its coupler attaches
                    # to node.state (ranked_domain_builder's node=).
                    from woof.core.adaptive_clock import maximum_map_factor
                    cold_decisions[domain.grid_id] = streaming.ranked_decision(
                        domain.run, exp.devices, max_map_factor=maximum_map_factor(
                            geography=store_bundle.geography))
                    stream = streaming.ranked_domain_builder(
                        store_bundle, clock=node.clock, options=exp.devices,
                        node=node)(None, domain.run, cold_decisions[domain.grid_id])
                else:
                    stream = streaming.store_domain_builder(store_bundle, node=node, clock=node.clock)(
                        None, domain.run, cold_decisions[domain.grid_id])
            stream._state = node.state
            node.state._streamed_domain = stream
            from woof.core.streaming import _STORE_ATTR, STREAMED_SCRATCH_ATTR
            # A split grid's store is a mirror of its slabs; publishing it is
            # a reference, not a reason to drain it (the next sweep would
            # then copy the whole store back to every slab for nothing).
            published = (stream._run.raw_store if getattr(stream, "ranked", False)
                         else stream.store)
            setattr(node.state, _STORE_ATTR, published)
            setattr(node.state, STREAMED_SCRATCH_ATTR,
                {key[8:]: value for key, value in published.items() if key.startswith('scratch/')})
            if reservation is not None:
                stream._reconstruction_reservation = reservation
                stream._reconstruction_host_budget_bytes = int(cold_decisions[domain.grid_id].detail['host_claim_bytes'])
            early_steppers[domain.grid_id] = stream
            if boundary_source is not None and domain.parent_id == 0:
                from woof.ingest.boundary_stream import keep_interval_check

                keep_interval_check(boundary_source.intervals, clock_guard)
        if parent is not None:
            parent.children.append(node)
            if exp.feedback == 1 and starts_at_t0:
                # WRF initialization is part of the two-way activation
                # contract, not an optional sweep: med_nest_initial runs
                # med_nest_feedback on each nest as it is built
                # (share/mediation_integrate.F:774-777), input-file
                # nests included, and the parent is re-diagnosed after.
                # The prepared artifacts are authored one-way, so this
                # per-child transaction at restore time is exactly what
                # makes the restored tree equal to a WRF tree whose
                # nests were initialized from input files.  Parent-first
                # restore order makes each parent live before its child
                # feeds back, matching build_experiment's ordering.
                from woof.core.model import FeedbackScratch

                initial = FeedbackScratch()
                node.coupler.feedback_prepare(node, initial)
                node.coupler.feedback_commit(node, start=True)
                node.coupler.feedback_finalize(node, start=True)
            # An input adapter that starts nests the way WRF does finishes
            # each one here, after the feedback that reads it first
            # (woof.wrfinput_forecast.WrfInitialization.finish_nest_start).
            finish_nest_start = getattr(initialization, "finish_nest_start", None)
            if finish_nest_start is not None:
                finish_nest_start(node, bundle)
        nodes[domain.grid_id] = node
        prepared[domain.grid_id] = SimpleNamespace(
            static_fields=bundle.static_fields,
            geog_selection=getattr(bundle, "geog_selection", None),
            initial_result=restored.initial_result,
            streamed_store=store_bundle,
            # The WRF-file door's own grid words, which history writes
            # (woof.io.history_layout.WRF_FILE_GRID_HISTORY_FIELDS).
            wrf_file_grid=MappingProxyType(wrf_file_grid_history_fields(bundle, domain.run)),
        )
        drivers[domain.grid_id] = driver
    # The loop's names for its LAST domain would otherwise live as long as
    # this function, which is the whole run; when that domain is a delayed
    # child they kept its startup restore and driver on the card after its
    # activation had replaced both.  The node and the maps above own
    # everything the run reads.
    del restored, driver, initialized, store_bundle
    # This runner constructs DomainNodes directly rather than going through
    # core.model.build_experiment.  Bind the prepared root's already-attached
    # external mirror before restart validation or the first solve so Davies
    # consumers use WRF's post-increment dtbc semantic (dt..T), not the
    # retired elapsed-based compatibility path (0..T-dt).
    if exp.root.grid_id not in early_steppers:
        bind_lateral_boundary_clock(
            nodes[exp.root.grid_id].state, nodes[exp.root.grid_id].clock)
    timing["restore_tree_and_initialize_physics"] = time.perf_counter() - started
    if exp.perturbation is not None:
        # Treatment proof, before any integration: the accepted config
        # echoed value for value plus what each restored domain actually
        # received.  A resumed run records the resume instead -- the
        # bubbles live in the checkpoint trajectory, not in a fresh
        # application.
        runtime._write_initial_perturbation_receipt(
            evidence, exp,
            initial_perturbation_receipts if restart is None else [{
                "resumed": True,
                "note": "resumed from checkpoint; the initial "
                        "perturbation was applied when the trajectory "
                        "began and is not re-applied",
            }])

    if sealed_forcing_extension:
        # The sealed route keeps its OWN identity and its own digest: it is
        # deliberately stable across successively longer legs, which is the
        # whole point of a horizon extension, so it cannot be replaced by the
        # restart identity below.  What it gains here is named components,
        # published beside the digest exactly as the restart route's are.
        sealed_components = sealed_extension_identity_components(
            exp, runtime_identity)
        fingerprint = hashlib.sha256(
            _canonical(sealed_components).encode("utf-8")).hexdigest()
        fingerprint_components = _strict_json(sealed_components)
    else:
        fingerprint_components = tree_restart_identity_components(
            inputs, runtime_identity, initialization)
        fingerprint = hashlib.sha256(
            _canonical(_strict_json(fingerprint_components)).encode("utf-8")
        ).hexdigest()
    model = ExperimentState(
        root=nodes[exp.root.grid_id],
        nodes_by_grid_id=MappingProxyType(nodes),
        schedule=schedule,
        memory_ledger=ledger,
        experiment_fingerprint=fingerprint,
    )
    # The tree is published with the experiment it was configured from,
    # exactly as every other route publishes it.  A checkpoint that
    # persists a live follower records where each domain was DECLARED
    # beside where it now sits, and the declaration is a fact about this
    # config -- not about which ingest produced the tree.
    publish_declared_experiment(model, exp)
    model._scratch_arena = arena
    model._dycore_state_workspace = rebuilt
    # Published beside the digest so a checkpoint written here can be
    # refused BY NAME rather than as an unexplained hash difference.
    model._experiment_fingerprint_components = fingerprint_components
    model._prepared_by_grid_id = prepared
    model._input_catalog = None
    model._activation_context = {"experiment": exp}
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None
    model._last_checkpoint = None
    # The seam every mid-run lifecycle and relocation emitter reaches
    # for.  Published on the MODEL and not on the runners: this route
    # can rebuild a relocation runner mid-run (a follow target that was
    # dormant acquires one at the leg boundary it is born on), and a log
    # wired into a runner at construction would be a log the run's later
    # runners never got -- the nest that moved would be exactly the one
    # nothing recorded.
    progress_log.publish_step_log(model, step_log)

    # The corridor lift: with a verified statics corridor on hand this
    # route wires the SAME RelocationRunner the case-data route does --
    # corridor crops standing in for per-footprint GEOG rebuilds --
    # which is exactly what makes the preflight's corridor-less refusal
    # accurate rather than permanent.  Bounds-only [relocation] builds no
    # runner, exactly as everywhere else.
    relocation_runner = (
        runtime.build_prepared_tree_relocation_runners(
            exp, statics_corridor=inputs.statics_corridor, model=model,
            outdir=outdir, radiation_workspace=radiation_workspace)
        if inputs.statics_corridor is not None else None)
    # BOUND BEFORE THE FIRST CHECKPOINT, not at the runner's first
    # receipt.  The checkpoint writer is handed a tree and a tick count
    # and asks the tree what followers it drives; a runner that attaches
    # itself only when it first records something leaves every
    # checkpoint taken before the opening cadence boundary silently
    # follower-free, and a resume then rebuilds the follower cold.
    spawn_runner = runtime.build_prepared_tree_spawn_runner(
        exp, statics_corridor=inputs.statics_corridor, model=model,
        outdir=outdir, radiation_workspace=radiation_workspace)
    runtime.publish_lifecycle_runners(
        model, spawn_runner=spawn_runner,
        relocation_runner=relocation_runner)

    from woof.ensemble.runtime_context import bind_current_member_model
    bind_current_member_model(model)
    if getattr(exp, 'nudging', None) is not None:
        from woof.spectral_nudging.prepared import allocate_held_tendencies, selected, validate_fixed_domains
        validate_fixed_domains(exp)
        for grid_id, node in nodes.items():
            allocate_held_tendencies(node.state, node.cfg.run, selected(exp, grid_id))
    restart_info = None
    if restart is not None:
        checkpoint = validate_manifest_checkpoint(Path(restart))
        # The lifecycle peek, before anything is reconstructed: one JSON
        # member off the root says which followers the checkpointed run
        # drove and what each one's history was.  Validated against THIS
        # run the same way the case-data route validates it.
        lifecycle = read_tree_lifecycle_header(checkpoint, model)
        checkpoint = lifecycle.root_path
        # PUT THE NEST BACK WHERE IT WAS, before restoring into it.  The
        # tree above was built from the config, so a nest that moved
        # during the checkpointed run is sitting on its ORIGINAL ground
        # -- different terrain, map factors and base state than the
        # bytes about to be restored.  setup_fingerprint refuses that,
        # correctly, and it is the whole reason a moving-nest run could
        # not resume.  The checkpoint records each placement; adopt it.
        #
        # Needs the relocation runner, because moving a nest to a
        # placement means cropping its statics out of the sealed
        # corridor, and the runner is what holds that wiring.  A
        # checkpoint from a run whose nest never moved needs none of
        # this and takes the fast path in adopt_placement.
        placements = checkpoint_placements(checkpoint, set(nodes))
        moved_grid_ids = {
            int(gid) for gid in
            ((read_restart_header(checkpoint).get("relocation") or {})
             .get("moved_grid_ids") or ())}
        # A grid is put back if its placement differs OR if the
        # checkpointed run relocated it at all.  The second half is not
        # redundant: a nest that wandered away and returned sits at its
        # original i/j carrying setup arrays that came from a relocation
        # rebuild rather than the prepared cache, and those differ --
        # phb by 2**-6, which the setup fingerprint refuses.  Deciding on
        # the placement number alone made a resume work from one
        # checkpoint of a run and fail from an earlier one.
        wanted = {gid: pl for gid, pl in placements.items()
                  if gid in nodes
                  and (gid in moved_grid_ids
                       or (int(pl["i_parent_start"]),
                           int(pl["j_parent_start"]))
                       != (int(nodes[gid].cfg.i_parent_start),
                           int(nodes[gid].cfg.j_parent_start)))}
        if wanted and relocation_runner is None:
            raise RuntimeError(
                "this checkpoint was written by a run whose nest had "
                f"moved (d{sorted(wanted)[0]:02d} is at "
                f"{tuple(sorted(wanted.values())[0].values())[:2]}, the "
                "config starts it elsewhere), and this run has no "
                "relocation runner to put it back -- a [relocation] "
                "block with a verified statics corridor is what supplies "
                "the terrain for a placement the config never names")
        # The as-built fingerprint, captured BEFORE any placement is
        # adopted: adopting one goes through relocate_child, which marks
        # the fingerprint itself, and the chain below has to replay from
        # the value this build started with -- not from a value that has
        # already been marked by the putting-back.
        fingerprint_as_built = model.experiment_fingerprint
        # Parent-first: a containment ancestor must be where it belongs
        # before its descendant's placement is read against it.
        for gid in sorted(wanted):
            relocation_runner.adopt_placement(
                model, nodes[gid],
                i_parent_start=int(wanted[gid]["i_parent_start"]),
                j_parent_start=int(wanted[gid]["j_parent_start"]),
                force=gid in moved_grid_ids)
        # Put the tracker's hysteresis and the audit's counters back, so
        # the resumed run's first consultation is suppressed exactly as
        # long as the unbroken run's would have been, and its ledger
        # continues the count instead of restarting it.
        _rel_header = read_restart_header(checkpoint).get("relocation") or {}
        if lifecycle.block is not None:
            # The follower's OWN entry, taken whole through the seam the
            # writer took it from: the segment chain (so this run's next
            # move chains onto its real predecessor instead of onto the
            # base preparation), the executed-move count, and BOTH
            # cooldown anchors.  The `continuity` fallback below carries
            # two of those four and is what a checkpoint written before
            # the lifecycle block existed has instead.
            seeded = runtime.restore_nest_followers(model, lifecycle)
            if relocation_runner is not None and seeded:
                relocation_runner.receipts.append({
                    "event": "follower-state-restored",
                    "from_checkpoint": Path(checkpoint).name,
                    "grid_ids": [int(gid) for gid in seeded],
                    "prior_moves": _rel_header.get("moves"),
                    "segment_id": _rel_header.get("segment_id"),
                })
        else:
            _continuity = _rel_header.get("continuity")
            if _continuity and relocation_runner is not None:
                applied = relocation_runner.restore_continuity(_continuity)
                relocation_runner.receipts.append({
                    "event": "continuity-restored",
                    "from_checkpoint": Path(checkpoint).name,
                    "prior_moves": _rel_header.get("moves"),
                    "segment_id": _rel_header.get("segment_id"),
                    "applied": applied,
                })
        # RECONSTRUCT THE IDENTITY, do not bypass the gate.  Every
        # executed move chains its record into the live fingerprint
        # (mark_fingerprint_across_move), one-way, so no fresh build ever
        # computes a moved tree's value -- which is exactly what made a
        # moved checkpoint refuse to resume BY CONSTRUCTION rather than by
        # anyone deciding it should.
        #
        # Replaying the SAME record hashes, in the same order, from this
        # build's own fingerprint reproduces that value if and only if
        # this is the run that wrote the checkpoint.  A different config
        # still mismatches, because the base of the chain differs.  So the
        # gate keeps its full meaning and simply stops being unpassable.
        #
        # Unconditional since the resume became exact.  It was behind
        # --allow-restart-across-move while a restart across a move
        # promised nothing; it now reproduces the unbroken run bit for
        # bit, so an opt-in would only be a way to be refused.  The three
        # things that made it inexact are all carried now: the placement
        # and the tracker's hysteresis (above), the acoustic Omega
        # (CHECKPOINT_ONLY_STATE), and the tracker's consultation window
        # (restart.CARRIED_SCRATCH_SLOTS).
        header = read_restart_header(checkpoint)
        model.experiment_fingerprint = fingerprint_across_stored_chain(
            header, fingerprint_as_built, model)
        restart_info = restore_tree_restart(
            checkpoint, model,
            sealed_forcing_extension=sealed_forcing_extension)
        model._resumed = True
    elif getattr(initialization, "warm_start_clocks", None):
        # A WRF warm start (woof.wrfinput_forecast.WrfRestartInitialization):
        # the state and physics were restored from the wrfrst files above,
        # and the clocks open at the restart instant, as a resume opens
        # them at its checkpoint's.  A woof checkpoint taken later in the
        # same run resumes through the branch above instead.
        from woof.ingest.wrfrst import apply_warm_start_clock
        restart_info = apply_warm_start_clock(
            model, nodes, initialization.warm_start_clocks)
    # An initialization that acts once the tree is restored and its clocks
    # are open (the analysis door, woof.da.analysis_door: an increment added
    # at the analysis instant or forced over its window).  Here, after a
    # resume or a warm start has put the clocks where the run continues, and
    # before the initial health gate reads the state it will integrate.
    after_restore = getattr(initialization, "after_restore", None)
    if callable(after_restore):
        after_restore(model=model, nodes=nodes, start_time=exp.start_time,
                      run_seconds=float(exp.run_seconds), restart=restart)
    if restored_callback is not None:
        restored_callback(model=model, nodes=nodes, start_time=exp.start_time,
                          run_seconds=float(exp.run_seconds), restart=restart)
    # Same ruling as the case-data route: a restore point that IS the
    # stop tick is a finished run, and the schedule executor refuses a
    # start period at the end of the schedule, so this route decides it
    # here rather than handing a user a bare ValueError.
    already_complete = runtime._restart_is_complete(restart_info)

    initial_health = {}
    for grid_id, node in nodes.items():
        result = vars(
            health_validator_for_domain(model, node).validate(
                phase=f"initialized.d{grid_id:02d}"
            )
        )
        initial_health[f"d{grid_id:02d}"] = _strict_json(result)
        if not result["ok"]:
            raise FloatingPointError(f"initial d{grid_id:02d} health failed: {result}")

    from woof.ensemble.runtime_context import (initialized_bootstrap_handoff,
        observe_current_counters, observe_initial_diagnostics)
    ensemble_report = initialized_bootstrap_handoff(
        ensemble_bootstrap, inputs=inputs, model=model, node=model.root,
        output_directory=outdir, observer=observer, step_log=step_log)
    if ensemble_report is not None:
        return ensemble_report
    observe_initial_diagnostics(model)
    observe_current_counters(model, start_time=exp.start_time)

    history = []
    # Boundary-only sampling under-reported the peak: the executor trims
    # the CuPy pool per STEP and at period commit BEFORE the progress
    # callback fires, so samples taken only in those callbacks missed
    # the intra-step transient working set (19.41 GiB reported against
    # 22.34 GiB true on the four-domain tree shape).  The watcher polls
    # from a daemon thread as well, and the boundary/end-of-run
    # sample() calls below fold into the same maxima.
    # The per-process NVML views ride beside the runtime and pool views
    # so the receipt can say WHOSE bytes the card carried: a foreign
    # process sharing the card was otherwise read as this run's growth.
    memory_watch = GpuPeakMemoryWatcher(
        default_cupy_probes() + nvidia_smi_process_probes())

    writers = (
        PerDomainWrfoutWriters(
            model,
            outdir / "wrfout",
            start_time=exp.start_time,
            # The title used to say HRRR on every tree, including the GFS
            # trees this runner has executed since the GFS front door
            # opened.  A durable artifact does not get to name a source
            # its run never touched; `inputs.source` is the run's own.
            title=f"woof prepared {inputs.source.upper()} domain tree "
                  f"{exp.name}",
            # Same contract as the single-domain runner: the prepared
            # cache's source identity carries the initial-condition
            # provenance for sources whose front door publishes one.
            initial_condition=inputs.source_identity.get(
                "initial_condition"),
            source=inputs.source,
            # The tree-wide [output] history selection; each domain's own
            # `output = {...}` overrides it inside the writer set.
            history_selection=exp.output,
            simulated_radar=exp.simulated_radar, radar_output_dir=outdir,
        )
        if io_mode == "history"
        else None
    )
    model._io_manager = writers
    forecast_started = time.perf_counter()


    def history_handler(_tree, node, ticks):
        # ASKED OF THE STEPPER, per grid.  ``steppers`` is bound below and
        # read at CALL time.  Under ``[tiles] store = "host"`` the domain
        # is in a pinned host store and ``node.state`` is the copy taken at
        # t = 0 that the sweep never writes, so every history sample would
        # otherwise record the INITIAL nan/w_max/CFL as though it were the
        # frame's.  ``stability_observer`` returns ``dycore.stability_report``
        # itself for a resident grid, which is every grid of a tree that
        # configures no [tiles].
        sample = {
            "grid_id": int(node.cfg.grid_id),
            "ticks": int(ticks),
            "elapsed_seconds": float(node.clock.elapsed_seconds),
            **streaming.stability_observer(
                steppers.get(int(node.cfg.grid_id)))(
                    node.state, node.cfg.run,
                    boundary_width=node.cfg.run.spec_bdy_width),
        }
        history.append(_strict_json(sample))
        if writers is not None:
            runtime._submit_tree_history_frame(writers, node, ticks)
        memory_watch.sample()

    checkpoint_publisher = prepared_single._CheckpointPublisher()
    # ``--checkpoint-at`` (woof.checkpoint_schedule): the instants a later
    # leg restarts from.  Resolved here, before the first step, so an
    # instant this run never reaches is refused now and not discovered by
    # the leg that needed it.  Absent, every scheduled checkpoint is written.
    from woof import checkpoint_schedule
    checkpoint_instants = checkpoint_schedule.current()
    checkpoint_selection = (None if checkpoint_instants is None else
                            checkpoint_schedule.resolve(
                                checkpoint_instants,
                                start_time=model.schedule.clock.start_time,
                                run_ticks=model.schedule.clock.run_ticks,
                                restart_ticks=model.root.clock.spec.restart_ticks,
                                tick_den=model.schedule.clock.tick_den))

    def restart_handler(tree, ticks):
        if (checkpoint_selection is not None
                and not checkpoint_selection.keeps(ticks)
                and not checkpoint_schedule.restart_requested()):
            # Scheduled, and no later leg restarts from it.
            return
        # At most one checkpoint in flight; a failed one fails the run here.
        checkpoint_publisher.wait()
        valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        restart_started = time.perf_counter()
        # Snapshotted between two model steps, at the stop tick too: its own
        # record, sized from the state it writes, or the supervisor times it
        # as a step (see runtime._writing_progress).
        with runtime._writing_progress(
                observer, "checkpoint",
                work_bytes=runtime._checkpoint_work_bytes(tree)):
            pending = write_tree_restart(
                outdir, tree, valid,
                sealed_forcing_extension=sealed_forcing_extension,
                defer_publish=True)
        # Capture this valid time's history targets on the model thread.
        # Later frames must not lengthen this checkpoint's history fence.
        try:
            history_durable = (None if writers is None
                               else writers.durability_barrier())
        except BaseException:
            pending.abandon()
            raise

        def publish(retire=True):
            try:
                if writers is not None:
                    if history_durable is None:
                        writers.drain()
                    else:
                        history_durable()
                path = pending.publish(retire=retire)
                step_log.restart_written(
                    domain=exp.root.grid_id, valid_time=valid, path=path,
                    wall_seconds=time.perf_counter() - restart_started)
            except BaseException:
                pending.abandon()
                raise

        if pending.deferrable and (writers is None or history_durable is not None):
            try:
                # Superseded sets are retired on this thread at the next
                # wait (see prepared_single._CheckpointPublisher).
                checkpoint_publisher.submit(
                    lambda: publish(retire=False),
                    then=pending.retire_superseded)
            except BaseException:
                pending.abandon()
                raise
        else:
            publish()

    restart_handler.orders_history_itself = True
    restart_handler.wait_durable = checkpoint_publisher.wait

    def progress_callback(**event):
        observe_current_counters(model, start_time=exp.start_time)
        if observer is not None:
            observer(**event)
        memory_watch.sample()
        _atomic_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "status": "RUNNING",
                "model_elapsed_seconds": event["model_elapsed_seconds"],
                "outer_step": event["outer_step"],
                "requested_run_seconds": float(exp.run_seconds),
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                    "cuda_device_used"),
                "last_durable_wrfout": event.get("last_durable_wrfout"),
                "last_checkpoint": event.get("last_checkpoint"),
            },
            heartbeat=True,
        )
        if boundary_source is not None:
            # Once per ROOT step, after that instant's frames and
            # checkpoint: the interval the next root step needs is asked
            # for here, so a wait (and a lead that never posts) happens
            # with the tree at a clean seam, not inside a nest's substep.
            elapsed = float(event["model_elapsed_seconds"])
            stepped["model_elapsed_seconds"] = elapsed
            if elapsed < float(exp.run_seconds):
                prepared_single._require_next_interval(
                    boundary_source.intervals, elapsed, writers=writers,
                    exp=exp)

    # After the closures, before any submit (the first happens inside
    # execute_experiment below).  `io_mode="none"` has no writers to
    # attach to and commits no outputs, so there is nothing to announce.
    #
    # ONE attach, with every consumer behind it: attach_progress_callback
    # OVERWRITES the writers' single landing slot, so a second call would
    # silently unhook the first and the run would keep publishing frames
    # with the earlier consumer no longer watching.
    landing = progress_log.LandingFanout(
        getattr(observer, "output_committed", None),
        step_log.output_committed if step_log.enabled else None,
        None if first_products is None else
        first_products.frame_committed,
        # The column store this runner owns when nobody hosts it (the
        # subprocess route of `woof go`):
        # prepared_single._route_owned_columns.
        prepared_single._columns_sink(columns))
    if landing and writers is not None:
        writers.attach_progress_callback(landing)
    if writers is not None:
        # Each history write between two steps beats on the run's
        # heartbeat; the landing fan-out above does not carry it.
        writers.attach_write_progress(observer)
    if relocation_runner is not None and writers is not None:
        # Same seam as the case-data route: a moved domain's later
        # frames must describe the footprint that produced them.
        if getattr(relocation_runner, "is_collection", False):
            relocation_runner.attach_writers(writers)
        else:
            relocation_runner.on_child_built.attach_writers(writers)
        # ...and to the descendant preparers of a MID-TREE move, which
        # are separate instances and would otherwise refresh nothing.
        _fan = getattr(getattr(relocation_runner, "reground_descendant", None),
                       "attach_writers", None)
        if callable(_fan):
            _fan(writers)
        # ...and to the containment leg's preparer, a third separate
        # instance (the sliding parent's own frames must re-georeference
        # after each slide exactly as a moved leaf's do).
        _cont = getattr(relocation_runner, "containment_preparer", None)
        _fan = getattr(_cont, "attach_writers", None)
        if callable(_fan):
            _fan(writers)

    # [tiles].  Absent -- the default -- this is an empty mapping and the
    # executor binds woof.core.dycore.step for every grid, which is the
    # function it always bound: no planner is consulted and no tilestream
    # module is imported, so a resident forecast pays nothing for the mode
    # existing.  Configured, a domain that the planner says will not fit
    # resident gets a streamed stepper here or a loud refusal; what it never
    # gets is a silent resident run that dies at the allocation the mode was
    # turned on to avoid.  The builders are what makes "gets a streamed
    # stepper here" true rather than aspirational: until they were wired,
    # every configuration on this route took the refusal branch, including
    # the ones the mode exists for.  A NEST that fires is still refused, on
    # purpose -- see streaming.prepared_domain_builder.
    #
    # The decisions are recorded per grid because the stepper dict cannot
    # report them: a grid that declined to stream is simply missing from it,
    # which is also what a grid looks like when [tiles] was never
    # configured.  On a TREE that ambiguity is worse than on a single
    # domain -- `auto` can legitimately stream the parent and leave a small
    # nest resident, so "some grids are missing" is the NORMAL case and
    # cannot be read as an anomaly.  Only a per-grid verdict distinguishes
    # that from a run where auto declined on every grid.
    #
    # ONE ADMISSION PER RUN.  The decision handed in here is the one the
    # door took above, from preflight.admission_estimate against the cold
    # planning machine, with the moving subtree marked off the declared
    # experiment.  Deciding again here -- which is what passing the run's
    # own ledger estimate did -- asked a SECOND admission from a richer
    # envelope (retained boundary intervals, real lateral boundaries)
    # against a budget that could carry no withholding where the cold
    # pass withheld a moving nest's rebuild, so a user could be shown one
    # road and given another after the download was already paid for.
    streaming_decisions = cold_decisions
    steppers = early_steppers
    if initialization is not None:
        steppers = streaming.steppers_for_tree(
            model, exp.tiles, builders=streaming.builders_for_tree(model, exp.tiles),
            decisions=streaming_decisions, machine=planning_machine,
            tree_decision=cold_tree)
    # A split grid's decision is the [devices] road's, reported in the
    # devices receipt; the [tiles] summary speaks for the rest.
    tile_decisions = {gid: decision for gid, decision in streaming_decisions.items()
                      if int(gid) not in split_ids}
    streaming_report = streaming.streaming_receipt(
        exp.tiles, tile_decisions)
    if streaming_report:
        print(f"prepared tree: {streaming_report['summary']}", flush=True)

    # ONE line per model time step per DOMAIN, which on a tree is the
    # whole point: `progress_callback` fires once per ROOT step, so a
    # d04 taking 36 substeps inside one of them reported nothing.
    step_observer = step_log.step_observer if step_log.enabled else None
    # An initialization's own per-step observer (the analysis door's balance
    # series: the mean surface-pressure tendency of the analysed start),
    # beside the step log's.  Absent for every other initialization.
    initialization_observer = getattr(initialization, "step_observer", None)
    if callable(initialization_observer):
        step_observer = _both_observers(step_observer, initialization_observer)
    from woof.stability_recovery import NestedHealthRecovery, RecoveryRefused

    def rearm_products(checkpoint):
        nonlocal first_products
        if first_products is not None:
            if health_retry_products is None:
                raise RecoveryRefused(
                    "the caller supplied standalone render consumers without "
                    "a retry rearm hook; history cannot be removed while read")
            first_products = health_retry_products(checkpoint)

    recovery = NestedHealthRecovery(
        model=model, experiment=exp, output_directory=outdir,
        writers=writers, history=history, observer=observer,
        before_rewind=rearm_products,
        sealed_forcing_extension=sealed_forcing_extension)

    def _newborn_stepper(grid_id, _node):
        # A spawned nest was priced by the cold plan before the first step
        # (the reservation contract); it runs on the road that plan chose.
        decision = cold_decisions.get(int(grid_id))
        if decision is not None and getattr(decision, "stream", False):
            raise RuntimeError(
                f"d{int(grid_id):02d} was planned to stream, and a spawned nest "
                "has no streamed-domain builder on the prepared tree route; give "
                "the slot a smaller size_km or the card more room")
        return None

    def execute_leg(active_experiment):
        try:
            if spawn_runner is not None:
                # The leg walk: dormant nests are born mid-run, follow their
                # storm, retire on their own and re-arm (docs/nest-lifecycle.md).
                return runtime.walk_spawn_legs(
                    model, active_experiment, None,
                    spawn_runner=spawn_runner, writers=writers,
                    lbc_interval_s=float(inputs.boundary_interval_seconds),
                    relocation_runner=relocation_runner,
                    relocation_runner_factory=(
                        lambda: runtime.build_prepared_tree_relocation_runners(
                            active_experiment,
                            statics_corridor=inputs.statics_corridor,
                            model=model, outdir=outdir,
                            radiation_workspace=radiation_workspace)),
                    spawned_stepper_factory=_newborn_stepper,
                    history_handler=None if writers is None else history_handler,
                    restart_handler=restart_handler,
                    progress_callback=progress_callback,
                    validate_state=True, health_debug=health_debug,
                    skip_feedback_path=(int(exp.feedback) == 0),
                    steppers=steppers, step_observer=step_observer,
                    delayed_child_initializer=initialize_delayed_child,
                    **({} if schedule_dispatch is None else {"schedule_dispatch": schedule_dispatch}))
            return execute_experiment(
                model, history_handler=None if writers is None else history_handler,
                restart_handler=restart_handler, progress_callback=progress_callback,
                validate_state=True, health_debug=health_debug,
                skip_feedback_path=(int(exp.feedback) == 0),
                relocation_runner=relocation_runner, steppers=steppers,
                step_observer=step_observer, experiment=active_experiment,
                delayed_child_initializer=initialize_delayed_child,
                **({} if schedule_dispatch is None else {"schedule_dispatch": schedule_dispatch}))
        finally:
            # Recovery reads _last_checkpoint as soon as a leg fails.
            # Finish publication while its history writers are still alive.
            checkpoint_publisher.wait()

    # An initialization that forces a step (the WRF-input door's
    # mp_tend_radar = 1) attaches once the steppers are known, so a
    # resident, a ranked and a tile-streamed domain each get the attach
    # that is read on their road, or a refusal before the first step.
    attach_step_forcings = getattr(initialization, "attach_step_forcings", None)
    step_forcings = (None if attach_step_forcings is None
                     else attach_step_forcings(nodes, steppers))
    nudging_forcings = None
    if not already_complete and getattr(exp, 'nudging', None) is not None:
        from woof.spectral_nudging.prepared import PreparedTreeNudging
        nudging_forcings = PreparedTreeNudging(model, nodes, inputs, steppers)
    try:
        memory_watch.start()
        if already_complete:
            print(
                "prepared tree: restart point is this configuration's stop "
                f"tick ({model.schedule.clock.run_ticks} ticks, "
                f"{float(exp.run_seconds):g} s of model time); the run is "
                "already complete, finalizing without integrating",
                flush=True)
        if writers is None:
            execution = (
                _completed_execution_report(model) if already_complete
                else recovery.run(execute_leg))
            checkpoint_publisher.wait()
            wrfout_paths = ()
        else:
            with writers:
                execution = (
                    _completed_execution_report(model) if already_complete
                    else recovery.run(execute_leg))
                # The last checkpoint is durable, or its failure is raised,
                # before anything below reports the run.
                checkpoint_publisher.wait()
                writers.drain(before_domain=runtime._drain_progress(observer))
                wrfout_paths = writers.paths
        exp = recovery.experiment
        inputs = replace(inputs, experiment=exp)
        runtime._finalizing_progress(observer, "close-relocation-receipt")
        if spawn_runner is not None:
            # The walk rebinds the runner as newborn followers come alive;
            # the live binding is the one that holds their moves.
            relocation_runner = getattr(model, "_relocation_runner", relocation_runner)
        if relocation_runner is not None:
            relocation_runner.close_receipt(model)
        if spawn_runner is not None:
            # One row per nest life (spawn, track, stop reason), from the
            # receipts the walk and the relocation runner just closed.  A
            # life still running at the end stops at the run's end, not at
            # the last spawn boundary.
            from woof.nest_lives import write_nest_lives
            try:
                write_nest_lives(outdir, run_end_s=float(exp.run_seconds))
            except Exception as error:  # the forecast is done; never lose it here
                print(f"prepared tree: nest-lives summary not written: {error}", flush=True)
    except BaseException as error:
        from woof.ingest.boundary_stream import SourceBehind

        _settle_checkpoint(checkpoint_publisher)
        stopped = None
        if isinstance(error, SourceBehind):
            # A late source lead: the tree is at a clean seam (asked for
            # between two root steps), so its state is checkpointed there
            # and a relaunch resumes exactly at it.
            checkpoints = ([] if model._last_checkpoint is None
                           else [model._last_checkpoint])

            def seam_checkpoint(tree, ticks):
                restart_handler(tree, ticks)
                checkpoint_publisher.wait()
                checkpoints.append(tree._last_checkpoint)

            stopped = prepared_single._stop_at_seam(
                error, model=model, node=model.root, exp=exp,
                schedule=model.schedule, restart_handler=seam_checkpoint,
                checkpoint_ticks={}, checkpoints=checkpoints,
                seam_waits=seam_waits)
        said = error if stopped is None else stopped
        # The last line a driving script reads has to say what happened.
        step_log.close(status="FAIL",
                       error=f"{type(said).__name__}: {said}")
        if stopped is not None:
            raise stopped from error
        raise
    else:
        # After the drain: every frame this run will ever commit is
        # durable and carries its marker, so run_end is true when read.
        step_log.close(status="SUCCESS")
    finally:
        memory_watch.stop()
        if step_forcings is not None:
            step_forcings.detach()
        if nudging_forcings is not None:
            nudging_forcings.detach()
    if step_forcings is not None and not already_complete:
        step_forcings.require_applied()
    runtime._finalizing_progress(observer, "synchronize-device")
    cp.cuda.Stream.null.synchronize()
    timing["forecast_execution"] = time.perf_counter() - forecast_started
    model._io_manager = None
    memory_watch.sample()

    runtime._finalizing_progress(observer, "microphysics-transition-receipt")
    transition_path, transition_sha, transitions = (
        runtime._write_microphysics_transition_receipt(
            evidence, model, exp, resumed=bool(model._resumed)
        )
    )
    final_health = {}
    final_stability = {}
    final_digests = {}
    # The LIVE tree: a spawned nest still integrating at the end is checked
    # beside the startup grids; a retired one was detached at its boundary.
    for grid_id, node in dict(model.nodes_by_grid_id).items():
        validator = health_validator_for_domain(model, node)
        runtime._finalizing_progress(observer, f"final-health-d{grid_id:02d}",
            work_bytes=getattr(validator, "host_scan_bytes", None))
        result = vars(
            validator.validate(phase=f"final.d{grid_id:02d}")
        )
        final_health[f"d{grid_id:02d}"] = _strict_json(result)
        if not result["ok"]:
            raise FloatingPointError(f"final d{grid_id:02d} health failed: {result}")
        # The health gate above still validates node.state and CANNOT be
        # folded (one block per whole field, no windowing), so under a host
        # store it passes on the t = 0 snapshot; the stability record below
        # is folded per tile and is the domain's.
        final_stability[f"d{grid_id:02d}"] = _strict_json(
            streaming.stability_observer(steppers.get(int(grid_id)))(
                node.state, node.cfg.run,
                boundary_width=node.cfg.run.spec_bdy_width))
        stream = steppers.get(int(grid_id))
        final_digests[f"d{grid_id:02d}"] = (
            stream.canonical_digest(node.clock, scope="trajectory",
                before_hash=runtime._digest_progress(observer, grid_id))
            if stream is not None else
            canonical_state_digest(node.state, node.clock, scope="trajectory",
                before_hash=runtime._digest_progress(observer, grid_id)))

    if boundary_source is not None:
        # THE SEAL.  Every root interval this run integrated was hash-checked
        # as it loaded; here the complete tree is bound: the root's sealed
        # cache against the head and every segment, the full preflight
        # against the sealed tree, and each domain restored from the head
        # against its sealed twin.
        runtime._finalizing_progress(observer, "bind-prepared-seal")
        seal_started = time.perf_counter()
        inputs = _seal_tree_inputs(inputs, stream=boundary_source.intervals)
        # The setup the root's restore could not check at the head (its
        # boundary series did not exist yet), checked as the single
        # domain checks it at its seal.
        recorded = _root_bundle(inputs).cache_reader.metadata.get(
            "setup_fingerprint")
        root_id = int(exp.root.grid_id)
        if _root_setup_fingerprint(nodes[root_id], steppers.get(root_id)) != recorded:
            raise RuntimeError(
                "the sealed root cache records a different setup fingerprint "
                "than the streamed boundaries reproduce")
        waits = boundary_source.intervals.waits
        boundary_stream_receipt.update({
            "seal_wait_seconds": time.perf_counter() - seal_started,
            "waits": [{"interval": index, "seconds": seconds}
                      for index, seconds in waits],
            "wait_seconds_total": sum(seconds for _, seconds in waits),
            "clock_intervals_checked": (
                None if clock_guard is None else clock_guard.checked),
            "proof_sha256": inputs.authority_sha256["preparation_receipt"],
            "restore_to_seal_seconds": time.perf_counter() - head_started,
        })
    runtime._finalizing_progress(observer, "verify-inputs",
        work_bytes=(None if initialization is not None else
                    _prepared_input_bytes(inputs)))
    if initialization is None:
        _verify_inputs_unchanged(inputs)
    else:
        initialization.verify_inputs(inputs)
    final_identity = _runtime_source_identity()
    moved = _runtime_source_identity_change(runtime_identity, final_identity)
    if moved is not None:
        raise RuntimeError(
            f"forecast implementation changed during execution: {moved}")
    outputs = runtime._frame_records(
        wrfout_paths, completed_records=getattr(writers, "completed_records", ()),
        progress_callback=observer)
    runtime._finalizing_progress(observer, "write-receipts")
    timing["total"] = time.perf_counter() - started_total
    # The MYNN column width this process derived, or None when the scheme
    # never ran.  Read here rather than at build time so the register count
    # beside it is the compiled kernel's, not an assumption.
    from woof.core.mynn_pbl_scratch import mynn_column_chunk_receipt

    mynn_column_chunk = mynn_column_chunk_receipt()
    from woof.hydro.runtime import current_settings as _hydro_settings, activation_receipt
    from woof import numerics as _numerics
    from woof.free_threading import host_threads_report as _host_threads_report
    from woof.smoke_receipts import receipt as smoke_execution_receipt
    report = {
        **smoke_execution_receipt(exp.root.run),
        "hydro_activation": activation_receipt(_hydro_settings()),
        # Read by the release gate's defaults-active leg (R15); see the
        # single-domain report.
        "numerics": _numerics.receipt(),
        "host_threads": _host_threads_report(),
        **({} if inputs.physics_profile_assertion is None else
           {"physics_profile_assertion": dict(inputs.physics_profile_assertion)}),
        **({} if inputs.acoustic_substeps is None else
           {"acoustic_substeps": dict(inputs.acoustic_substeps)}),
        **({} if getattr(inputs, "terrain_clock", None) is None else
           {"terrain_clock": dict(inputs.terrain_clock)}),
        **({"terrain_autosmooth": {
            f"d{bundle.grid_id:02d}": dict(bundle.terrain_autosmooth)
            for bundle in inputs.domains
            if getattr(bundle, "terrain_autosmooth", None) is not None}}
           if any(getattr(bundle, "terrain_autosmooth", None) is not None
                  for bundle in inputs.domains) else {}),
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "source": inputs.source,
        "readiness": "IMPLEMENTED_UNVERIFIED",
        # Which install produced this tree of forecasts.  The
        # runtime_source_identity below hashes the implementation's
        # bytes; this names the install those bytes came out of, and
        # whether its two version claims agree with each other.
        "provenance": _provenance_receipt(),
        "execution_plan": inputs.execution_plan,
        "experiment": {
            "name": exp.name,
            "start_time": exp.start_time.isoformat(),
            "run_seconds": float(exp.run_seconds),
            "fingerprint": fingerprint,
            "domains": _domain_rows(exp),
        },
        # The physics-fidelity axis as this run RESOLVED it, not as it was
        # requested: the whole vector, every ledger entry, with the value
        # each one took.  A scoreboard row has to be able to say which arm
        # produced it without reading the configuration back, and an
        # ungoverned run says so in the same shape rather than by omission
        # (the morr_rimed_ice-into-receipt pattern, one axis wider).
        "physics_mode": exp.physics_mode.receipt(),
        # physics.mode (woof/correct_mode.py) as each domain's RunConfig
        # resolved it: correct by default from 2.9.0, exact when named, and
        # exact with forced_by under GPUWM_WRF_EXACT=1.
        "correct_mode": _correct_mode_receipt(
            {domain.grid_id: domain.run for domain in exp.domains}),
        # The initial-perturbation axis, same posture as physics_mode:
        # the arm says what it is in the receipt, not by omission.  None
        # when the config carries no block; configured, the full echo
        # plus per-domain application stats (also standalone in
        # evidence/initial-perturbation.json, written before
        # integration so a run that dies mid-flight still proves its
        # arm).
        "initial_perturbation": (
            None if exp.perturbation is None else {
                "config": exp.perturbation.receipt(),
                "domains": initial_perturbation_receipts,
            }),
        "restart_contract": {
            "mode": ("sealed-forcing-extension"
                     if sealed_forcing_extension else "exact-setup"),
            "restart_input": (
                None if restart is None else str(Path(restart).resolve())),
            # Non-None only when the restored checkpoint was written after
            # a nest relocation: the promises-nothing posture, stated in
            # the receipt of the run that crossed it.
            "relocation_crossed": getattr(
                model, "_restart_crossed_relocation", None),
        },
        "stability_recovery": recovery.receipt,
        # The [relocation] echo (None when the config never opted in).
        # A follow source reaches execution only over a verified statics
        # corridor (the preflight refuses corridor-less bundles), so a
        # non-None block is either bounds-only or carries the corridor
        # binding plus every move receipt the runner recorded.
        "relocation": (
            None if not (exp.relocation.enabled or any(
                getattr(dc, "follow", None) is not None for dc in exp.domains)) else {
                "config": exp.relocation.receipt(),
                **({"followers": {f"d{int(dc.grid_id):02d}": dc.follow.to_json()
                                   for dc in exp.domains if dc.follow is not None}}
                   if any(dc.follow is not None for dc in exp.domains) else {}),
                # A SET, not one corridor.  `statics_corridor` has been a
                # dict keyed by grid_id since the moving subtree needed a
                # corridor per member (the mover's, plus a root-framed one
                # for every descendant carried along); this echo still
                # read it as a single object and died on
                # `'dict' object has no attribute 'geometry'` -- at the
                # very END of the run, while writing the receipt, with
                # every wrfout already on disk.  A 3 h A/B found it; a
                # 114 h forecast would have found it the expensive way.
                "statics_corridor": _corridor_echo(inputs.statics_corridor),
                "receipts": _strict_json(list(
                    getattr(model, "_relocation_receipts", ()) or ())),
            }),
        "wall_seconds": timing["total"],
        "timing_seconds": timing,
        "executor": {
            "pool_trim": getattr(model, "_pool_trim_policy", None),
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "health": {
            "initial": initial_health,
            "final": final_health,
            "final_stability": final_stability,
            "history": history,
        },
        "final_state_digest": final_digests,
        # The chem mass ledger per chem domain (woof/core/chem_driver.py),
        # present only on a chem run so every other receipt keeps its bytes.
        **({} if not (chem_ledgers := {
            f"d{grid_id:02d}": ledger
            for grid_id, node in nodes.items()
            if (ledger := _chem_ledger_receipt(node.state)) is not None})
           else {"chem_ledger": chem_ledgers}),
        "microphysics_transitions": {
            "path": str(transition_path.resolve()),
            "sha256": transition_sha,
            "edges": transitions,
        },
        "memory": {
            "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                "cuda_device_used"),
            "cupy_pool_peak_total_bytes_observed": memory_watch.peak_bytes(
                "cupy_pool_total"),
            "cupy_pool_peak_used_bytes_observed": memory_watch.peak_bytes(
                "cupy_pool_used"),
            # WHOSE bytes: this process, every other process on the card,
            # the card itself, and how long the card was shared.  None
            # where NVML cannot attribute memory per process.
            **process_memory_receipt(memory_watch),
            "preflight_alloc_estimate_bytes": int(estimate.alloc_estimate_bytes),
            # HOW WIDE the MYNN column workspace was made, and from which
            # card terms.  It is the largest single scratch family a
            # bl_pbl_physics=5 run holds and it is derived from the device
            # rather than fixed, so the number that explains this run's
            # floor belongs beside the floor.  Absent -- and the receipt
            # therefore byte-identical to the one written before the width
            # was derived -- whenever MYNN never ran.
            **({"mynn_column_chunk": mynn_column_chunk}
               if mynn_column_chunk else {}),
            # What each number above actually measured, how often it was
            # sampled, and whether observation stayed complete.
            "gpu_peak_sampling": memory_watch.summary(),
            # Host-side, not VRAM: the corridor is cropped on the CPU
            # and the rebuilt child re-occupies the same device
            # footprint, so the GPU preflight estimate is unchanged.
            "statics_corridor_host_bytes": _corridor_host_bytes(
                inputs.statics_corridor),
            # WHICH EXECUTION MODE the two numbers above describe.  The
            # estimate prices a whole resident tree; a streamed domain's
            # observed peak is a few tile buffers.  Absent this field the
            # gap between them reads as slack in the estimator.  Empty --
            # and the receipt therefore byte-identical to the one written
            # before streaming existed -- whenever [tiles] is off.
            "tiles": streaming.receipt_entry(
                exp.tiles, tile_decisions),
        },
        "output": {
            "io_mode": io_mode,
            "frame_count": len(outputs),
            "total_bytes": (sum(item["bytes"] for item in outputs)
                if all(item["bytes"] is not None for item in outputs) else None),
            "files": outputs,
            "last_checkpoint": (
                None
                if model._last_checkpoint is None
                else str(Path(model._last_checkpoint).resolve())
            ),
        },
        "input": {
            "prepared_root": str(inputs.prepared_root),
            "source_identity": dict(inputs.source_identity),
            "forcing_hours": list(inputs.forcing_hours),
            "boundary_interval_seconds": inputs.boundary_interval_seconds,
            "authority_sha256": dict(inputs.authority_sha256),
            "domains": {
                f"d{bundle.grid_id:02d}": dict(bundle.authority_sha256)
                for bundle in inputs.domains
            },
        },
        "runtime_source_identity": runtime_identity,
        # Present only on a run bound at its prepared head: the head, the
        # waits for its boundary intervals and the seal it was bound to.
        **({} if boundary_stream_receipt is None
           else {"boundary_stream": boundary_stream_receipt}),
        # What the end-of-run recheck could actually SEE.  The identity
        # above is the one taken at launch; the gate that compares it
        # skips the git half when either end failed to resolve it, and a
        # skipped comparison that says nothing is how a weakened check
        # goes unnoticed.  ``git_compared: false`` on a receipt means the
        # run was bound by ``source_sha256`` alone.
        "runtime_source_identity_recheck": {
            "git_resolved_at_launch":
                runtime_identity["git_commit"] is not None,
            "git_resolved_at_end":
                final_identity["git_commit"] is not None,
            "git_compared": (runtime_identity["git_commit"] is not None
                             and final_identity["git_commit"] is not None),
        },
    }
    # Join the existing daemon render before a standalone process can exit.
    # Absent/none leaves the previous receipt unchanged, as on the single arm.
    if first_products is not None:
        runtime._finalizing_progress(observer, "finish-first-products")
        receipt = first_products.wait()
        if receipt is not None:
            report["first_products"] = receipt
    # Only when [tiles] was configured, so an unconfigured tree writes
    # the receipt it wrote before the mode existed -- see the same guard in
    # prepared_single_domain_forecast.
    if streaming_report:
        report["tiles"] = streaming_report
    if step_forcings is not None:
        report["radar_tendency"] = step_forcings.receipt()
    if nudging_forcings is not None:
        report['nudging'] = nudging_forcings.receipt()
    if split_ids:
        report["devices"] = _devices_tree_receipt(
            exp, split_ids, early_steppers, cold_decisions, devices_admission)
    # WHICH AEROSOL INITIAL CONDITION EACH DOMAIN STARTED FROM.  PER
    # DOMAIN, because each domain has its own initialization and its own
    # prepared cache: a root fed from WRF's monthly WIF climatology and a
    # child whose cache predates the receipt are two different facts about
    # one run, and one tree-wide verdict would lose both.  Read from the
    # caches the deciding processes wrote it into; never re-resolved here,
    # which would be a second resolution path over the same config field.
    # Absent entirely -- receipt byte-for-byte unchanged -- for a tree
    # whose domains all run schemes with no aerosol number fields.
    aerosol_by_grid_id = {
        int(bundle.grid_id): (
            bundle.cache_reader.metadata if initialization is None
            else initialization.domain_metadata(bundle)).get(AEROSOL_SOURCE_KEY, {})
        for bundle in inputs.domains
    }
    report.update(aerosol_source_report_entries(
        ((f"d{int(domain.grid_id):02d}", domain.run.mp_physics,
          aerosol_by_grid_id.get(int(domain.grid_id), {}))
         for domain in exp.domains),
        when_unrecorded=(
            "this domain's prepared cache carries no "
            "aerosol-initialization receipt, so it was written by a "
            "preparation predating the receipt being stored; re-prepare "
            "the hierarchy to record which source filled its nwfa/nifa")))
    # The initialization's own receipt block (the analysis door's: the
    # bundle, its checks, what went in at once and the IAU forcing's
    # receipt), absent for every initialization that has none.
    initialization_blocks = getattr(initialization, "run_report_blocks", None)
    if callable(initialization_blocks):
        report.update(initialization_blocks(model))
    wind = {str(gid): physics.wind_farm.receipt() for gid, physics in drivers.items()
            if getattr(physics, "wind_farm", None) is not None}
    if wind:
        report["wind_farm"] = wind
        for gid, physics in drivers.items():
            if str(gid) in wind:
                wind[str(gid)]["module_replay"] = physics.wind_farm.write_replay(
                    evidence / f"wind-replay-d{gid:02d}.nc")
    # Default-on hydro lives on the fixed parent (grid 1): cost and products.
    from woof.hydro.cost import receipt as _hydro_cost_receipt
    from woof.hydro.forecast_products import publish as _hydro_products
    _parent_physics = drivers.get(1)
    _parent_run = exp.domains[0].run
    report["hydro_cost"] = _hydro_cost_receipt(
        _parent_physics, forecast_seconds=timing.get("forecast_execution", 0.0),
        atmosphere_steps=int(round(float(exp.run_seconds) / float(_parent_run.dt))))
    report["hydro_products"] = _hydro_products(
        _parent_physics, _hydro_settings(), outdir=outdir, dx_m=float(_parent_run.dx))
    _atomic_json(evidence / "run-receipt.json", report)
    emit_run_capsule(
        outdir, emission_site="prepared_domain_tree_forecast",
        run_context={
            "runner_route_and_io_mode": {
                "route": "prepared_domain_tree_forecast", "io_mode": io_mode},
            "output_and_diagnostic_mode": {"io_mode": io_mode},
            # Both DETERMINISM.md byte pins bind here, git or no git:
            # this route refuses to start unless the experiment TOML
            # matches --experiment-config-sha256, so the pin's value
            # exists on every run (the 4090 stress run's accuracy
            # finding was this route's sibling reporting it
            # "unavailable" on the published wheel).
            "config_bytes": {
                "path": str(inputs.experiment_config),
                "sha256": inputs.authority_sha256["experiment_config"]},
            "input_artifact_bytes": dict(inputs.authority_sha256),
        },
        input_bytes={"entries": {
            **{name: {"algorithm": "sha256", "digest": digest}
               for name, digest in inputs.authority_sha256.items()},
            **{f"d{bundle.grid_id:02d}:{name}": {
                   "algorithm": "sha256", "digest": digest}
               for bundle in inputs.domains
               for name, digest in bundle.authority_sha256.items()},
        }},
        run_shape={
            "route": "prepared_domain_tree_forecast",
            "domain_count": len(exp.domains),
            "run_seconds": float(exp.run_seconds),
            "experiment_fingerprint": fingerprint,
            "domains": _domain_rows(exp),
        },
        output={"frames": outputs, "trajectory_digest": final_digests},
        # The spectral seam's run receipts merge in; an apply run whose
        # step receipts are incomplete refuses a clean capsule here.
        receipts={"run_receipt": {
            "path": str((evidence / "run-receipt.json").resolve())},
            **_seam_capsule_receipts(model),
            "pool_trim": getattr(model, "_pool_trim_policy", None)},
    )
    runtime._finalizing_progress(observer, "publish-completion")
    _atomic_json(
        progress_path,
        {
            "schema": PROGRESS_SCHEMA,
            "status": "PASS",
            "model_elapsed_seconds": float(exp.run_seconds),
            "requested_run_seconds": float(exp.run_seconds),
            "run_receipt": str((evidence / "run-receipt.json").resolve()),
            "frame_count": len(outputs),
        },
        heartbeat=True,
    )
    from woof.hydro.cost import refuse_if_over_budget as _refuse_hydro_cost
    _refuse_hydro_cost(report["hydro_cost"])
    return report


def _sealed_proof_sha256(prepared_root: Path, head_sha256: str, *,
                         on_wait=None) -> str:
    """Wait for a chained tree's seal and return its proof's digest.

    The seal is checked against the head the caller pinned
    (:func:`woof.ingest.boundary_stream.verify_seal`), and a producer that
    fails or falls silent ends the wait by name.  ``on_wait`` hears the
    wait as the end-of-run seal's does (:func:`_start_seal_waits`).  A
    source lead past its late time stays :class:`woof.ingest.
    boundary_stream.SourceBehind`, so the run exits 75 naming the lead
    (:func:`_source_behind_exit`) instead of reading as a refusal.
    """

    from woof.ingest.boundary_stream import (
        BoundaryStreamError, SourceBehind, StreamedIntervals, read_head,
        verify_seal)

    try:
        head = read_head(prepared_root, expected_sha256=head_sha256)
        stream = StreamedIntervals(prepared_root, head=head, on_wait=on_wait)
        stream.wait_sealed()
        return verify_seal(prepared_root, head=head)["proof_sha256"]
    except SourceBehind:
        raise
    except BoundaryStreamError as error:
        raise ValueError(str(error)) from None


def _source_behind_exit(outdir: Path, behind, observer) -> int:
    """End the run on a source lead past its late time, and say which.

    The frames already written and the checkpoint at the seam are kept,
    and the record says which lead, so the door exits 75 naming it (as the
    single domain does).  The same whether the lead fell behind at a seam
    or while the forecast waited for the seal before its first step.
    """

    from woof.ingest.boundary_stream import (
        SOURCE_BEHIND_EXIT_CODE, WAIT_LOG_NAME, SeamWaits)

    wrfout = outdir / "wrfout"
    frames = ([path for path in wrfout.glob("wrfout_*")
               if path.suffix != ".json"] if wrfout.is_dir() else [])
    behind = behind.at(frames_kept=len(frames))
    behind.details.setdefault("checkpoint", None)
    _write_failed_run_receipt(outdir, behind)
    SeamWaits(log_path=outdir / WAIT_LOG_NAME).record_source_behind(behind)
    hook = getattr(observer, "source_behind", None)
    if hook is not None:
        try:
            hook(dict(behind.details))
        except Exception:  # noqa: BLE001 - the refusal stands
            pass
    print(f"prepared_domain_tree_forecast: {behind}", file=sys.stderr)
    return SOURCE_BEHIND_EXIT_CODE


def _start_seal_waits(outdir: Path, prepared_root: Path, observer):
    """The ``on_wait`` of a seal wait before the forecast's first step.

    Said as a seam wait is said (:class:`woof.ingest.boundary_stream.
    SeamWaits`) with the model not stepped yet (``phase: start``): on the
    run's event stream when the observer has one, in the wait log, on
    stderr, and on the supervisor heartbeat as ``waiting:preparation`` or
    ``waiting:source``.  THE BREAKAGE: the seal wait before a rerun on the
    sealed tree said nothing, so ``woof go``'s watchdog saw the attempt's
    last ``integrating`` record stand still and stopped the worker as
    stalled once the wait passed its 120 s step bound.  No ``progress.json``
    block: the run's ``evidence/`` folder does not exist before the
    forecast restores.
    """

    from woof.ingest.boundary_stream import (
        PRODUCER_NAME, WAIT_LOG_NAME, SeamWaits, stream_dir)

    events = getattr(observer, "events", None)
    return SeamWaits(
        emit=None if events is None else events.emit,
        observer=observer,
        model_time=lambda index: {
            "phase": "start", "interval": None,
            "model_elapsed_seconds": None, "model_valid_time": None},
        say=lambda line: print(line, file=sys.stderr, flush=True),
        log_path=Path(outdir) / WAIT_LOG_NAME,
        producer_path=stream_dir(Path(prepared_root)) / PRODUCER_NAME)


#: Where a head-bound attempt's outputs go when the run starts again on
#: the sealed tree (a later interval moved the terrain clock):
#: ``<outdir>/.quarantine/streamed-attempt``.  A later attempt in the same
#: folder takes ``streamed-attempt-2`` and so on.
STREAMED_ATTEMPT_DIRNAME = "streamed-attempt"
#: The folder the attempts are kept under, inside the run folder.  THE
#: BREAKAGE: kept at ``<outdir>/streamed-attempt/``, the attempt's history
#: frames carried the same names as the sealed forecast's and sat inside
#: the run's history tree, so every reader that walks that tree took them
#: as this run's frames.  The WOOF worker keeps the first path per name in
#: sorted order, and ``streamed-attempt/`` sorts before ``wrfout/``, so a
#: published LA run (HRRR 2026-10-21T12, 2.25/0.75 km) carried the
#: discarded attempt's frames up to 15Z and the sealed forecast's after
#: it: d01 RAINNC fell by up to 2.085 mm between 15Z and 16Z at 4,723
#: cells and ``woof ml-export`` refused the run.  Every frame reader
#: already leaves a dot-prefixed folder out (``woof.io.wrfout.
#: is_wrfout_frame``, ``woof.render_layout.history_frames``) and the WOOF
#: worker leaves ``.quarantine`` out by name, which is the folder
#: ``woof.supervisor.quarantine_file`` keeps such files in.
STREAMED_ATTEMPT_PARENT = ".quarantine"
#: What stays in the run folder when an attempt is set aside: the
#: supervisor heartbeat, which describes the worker rather than one
#: attempt's outputs.  THE BREAKAGE: moved aside with the attempt, it left
#: ``woof go``'s watchdog reading nothing new while the rerun waited for
#: the seal, and the watchdog stopped the worker as stalled; on Windows
#: the move also races the watchdog's own reads of the file.  The
#: quarantine folder stays too: it holds the earlier attempts.
KEPT_IN_PLACE = frozenset({HEARTBEAT_NAME, STREAMED_ATTEMPT_PARENT})


def _is_streamed_attempt(name: str) -> bool:
    if name == STREAMED_ATTEMPT_DIRNAME:
        return True
    prefix = f"{STREAMED_ATTEMPT_DIRNAME}-"
    return name.startswith(prefix) and name[len(prefix):].isdigit()


def _set_aside_streamed_attempt(outdir: Path) -> Path:
    """Move every output of a head-bound attempt into one kept folder.

    The folder is ``<outdir>/.quarantine/streamed-attempt[-N]``: out of the
    run's history tree (:data:`STREAMED_ATTEMPT_PARENT`), so no reader
    publishes the attempt's frames as the sealed forecast's.  Each attempt
    gets its own folder and an earlier attempt's folder stays where it is,
    so a folder already holding one is never refused and no attempt is
    nested inside another.  The heartbeat stays in place
    (:data:`KEPT_IN_PLACE`).
    """

    outdir = Path(outdir)
    kept = outdir / STREAMED_ATTEMPT_PARENT
    kept.mkdir(exist_ok=True)
    earlier = {entry.name for entry in kept.iterdir()
               if entry.is_dir() and _is_streamed_attempt(entry.name)}
    name, number = STREAMED_ATTEMPT_DIRNAME, 1
    while name in earlier:
        number += 1
        name = f"{STREAMED_ATTEMPT_DIRNAME}-{number}"
    attempt = kept / name
    attempt.mkdir()
    for entry in sorted(outdir.iterdir()):
        if entry.name not in KEPT_IN_PLACE:
            os.replace(entry, attempt / entry.name)
    return attempt


def _release_attempt_memory() -> None:
    """Hand a set-aside attempt's device memory back before the next restore.

    The driver and state attachments are reference cycles, so a collection
    returns their arrays to the pool, and the pool then returns its blocks
    to the card: the sealed run's restore and every free-memory reading it
    takes see the card as a fresh launch on the sealed proof would.
    """

    import gc

    gc.collect()
    try:
        from woof.core.model import _trim_default_pool

        _trim_default_pool()
    except ImportError:
        # No CuPy on this install: the attempt held nothing on a card.
        return


def build_parser() -> argparse.ArgumentParser:
    """This runner's parser, built without parsing anything.

    Exposed so the docs/CLI parity test can read the option surface of a
    documented door without running it.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    # Declared, not merely intercepted: `main` answers this before
    # argparse sees anything, so without the declaration `--help` never
    # mentioned a flag the runner really has.
    parser.add_argument(
        "--show-capabilities", action="store_true",
        help=("print this runner's capability JSON and exit; it must be "
              "the only argument"))
    parser.add_argument("--prepared-root", type=Path, required=True)
    # Exactly one of these two binds the preparation (checked in main).
    parser.add_argument("--preparation-receipt-sha256", default=None,
                        help="sha256 of the sealed tree's preparation "
                             "document (proof.json or receipt.json)")
    parser.add_argument(
        "--prepared-head-sha256", default=None,
        help=("head_sha256 of boundary-stream/head.json: binds a chained "
              "tree's preparation at its head, so the forecast starts while "
              "the root's later boundary intervals are prepared; the seal "
              "is bound at the end"))
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--experiment-config-sha256", required=True)
    parser.add_argument("--physics-profile", default=None, metavar="ID",
                        help="assert every hash-bound domain uses the named "
                             "suite; omit to preserve mixed per-domain physics")
    parser.add_argument("--io-mode", choices=("history", "none"), default="history")
    parser.add_argument(
        "--restart", type=Path,
        help="resume from any member of a gpuwmrst checkpoint set written "
             "by an earlier run of this prepared tree.  The forecast "
             "length (run_seconds), the output/restart cadence "
             "(history_interval_s, restart_interval_s) and each "
             "domain's history window (history_begin_s, history_end_s) "
             "may differ from "
             "the run that wrote it -- the same contract `woof run "
             "--restart` publishes.  Under an adaptive clock the "
             "controller's targets and clamps (target_cfl, target_hcfl, "
             "the time-step bounds, max_step_increase_pct, the substep "
             "floor min_time_step_sound) may differ too: they govern "
             "future steps rather than model state, and "
             "a resume that retunes them is reported rather than "
             "refused, so a dead run can be recovered with the setting "
             "that would have saved it.  Turning use_adaptive_time_step "
             "itself on or off is still refused, as is anything else")
    parser.add_argument(
        "--sealed-forcing-extension", action="store_true",
        help=("write/restore checkpoints using the explicit append-only "
              "forcing-prefix contract"))
    parser.add_argument("--health-debug", action="store_true")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--render-products", default=None, metavar="SPEC",
                        help="plot selectors for every committed frame of every "
                             "grid, each drawn as it lands, 'all', or 'none'; "
                             "omitted means no rendering")
    from woof.product_publish import add_arguments as add_publish_arguments
    add_publish_arguments(parser)
    parser.add_argument("--render-dir", type=Path, default=None, metavar="DIR",
                        help="picture directory (default OUTDIR/png); "
                             "ignored without --render-products")
    parser.add_argument("--render-section", default=None,
                        metavar="lat,lon,lat,lon|FILE.json",
                        help="the line every xsec: product is cut along, "
                             "`woof render --section`'s own value; "
                             "ignored without --render-products")
    parser.add_argument("--devices", type=int, default=None, metavar="N",
                        help="split every grid the tree's [devices] domains "
                             "names (default every grid) into N resident "
                             "slabs; replaces [devices] count")
    parser.add_argument("--devices-table", default=None, metavar="JSON",
                        dest="devices_table",
                        help="the tree's [devices] table as JSON (count, grid, "
                             "ids, transport, domains); validated here, "
                             "without modifying the prepared configuration "
                             "or its digests")
    parser.add_argument("--no-memory-gate", action="store_true",
                        dest="no_memory_gate",
                        help="restore a tree whose priced peak envelope "
                             "exceeds this card's free memory anyway, as "
                             "`woof go --no-memory-gate` does: the envelope "
                             "is an upper bound and the card's own "
                             "allocation then decides; a model state too big "
                             "to build at all is still refused")
    # The same four flags the single-domain door carries, registered
    # from the same function so the two cannot drift.
    add_progress_arguments(parser)
    from woof.simulated_radar_config import add_execution_argument
    add_execution_argument(parser)
    return parser


def main(argv=None, *, observer=None) -> int:
    """The tree runner's command line.

    ``observer`` is forwarded to :func:`run_prepared_tree` and is
    otherwise inert: it lets a caller hosting this runner in its own
    process receive the progress it publishes plus each per-domain
    wrfout as it lands.  ``None`` -- every command-line invocation --
    changes nothing.
    """

    if argv is None:
        # Before any heavy import, as in the single-domain runner: a
        # free-threaded build keeps the [devices] rank threads of every
        # domain off the interpreter lock, which netCDF4's import would
        # otherwise switch back on (woof.free_threading).
        from woof.free_threading import keep_gil_disabled
        keep_gil_disabled()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--show-capabilities"]:
        print(json.dumps(runner_capabilities(), sort_keys=True))
        return 0
    from woof.ensemble.calibration_admission import refuse_explicit_config_argv
    try:
        refuse_explicit_config_argv(argv)
    except ValueError as error:
        print(f"prepared_domain_tree_forecast: {error}", file=sys.stderr)
        return 2
    # Which tree is about to integrate this tree of domains.  Same
    # contract as the single-domain runner, including leaving
    # ``--show-capabilities`` above untouched.
    from woof.provenance_gate import announce_for_main

    refusal = announce_for_main("woof-prepared-tree-forecast")
    if refusal is not None:
        print(f"prepared_domain_tree_forecast: {refusal}", file=sys.stderr)
        return 2
    if "--show-capabilities" in argv:
        print("prepared_domain_tree_forecast: --show-capabilities must be "
              "the only argument on the command line", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    from woof.first_products import render_without_output_refusal

    render_refusal = render_without_output_refusal(
        args.render_products, args.io_mode)
    if render_refusal is not None:
        print(f"prepared_domain_tree_forecast: refused: {render_refusal}",
              file=sys.stderr)
        return 2
    # THE capability preflight, from the same registry `woof run` and
    # `woof go` refuse with.  Before the output directory is claimed,
    # for the same reason the --outdir guard below is where it is: a gap
    # that was knowable before any work must not be discovered after it.
    #
    # The action half only.  This parser has no `--explain`, so a
    # pointer at that flag would name something this door does not have.
    from woof import capabilities
    from woof.explain import split as split_explanation

    try:
        capabilities.require(
            "python -m woof.prepared_domain_tree_forecast",
            *capabilities.COMMAND_REQUIREMENTS["run"],
            before=("Refusing here, before the output directory is "
                    "claimed and before the tree preflight runs."))
    except capabilities.CapabilityMissing as refused:
        print(split_explanation(str(refused))[0], file=sys.stderr)
        return 2
    # A rejected --outdir is a usage mistake, not a crash: it must read as
    # one sentence naming the problem and a directory that works.  A test host H
    # pilot met this guard as a raw traceback, on a command the front door
    # itself had suggested.
    try:
        outdir = claim_output_directory(
            args.outdir,
            protected_roots=(args.prepared_root, args.experiment_config))
    except (ValueError, FileExistsError) as error:
        print(f"prepared_domain_tree_forecast: --outdir refused: {error}",
              file=sys.stderr)
        return 2
    from woof.runtime import _preparation_progress
    from woof.ingest.boundary_stream import (
        StreamedClockChanged, SourceBehind)

    _preparation_progress(observer, "validate-prepared-inputs")
    started = time.perf_counter()
    binding = {
        "prepared_root": args.prepared_root,
        "experiment_config": args.experiment_config,
        "experiment_config_sha256": args.experiment_config_sha256,
        **({} if args.physics_profile is None else
           {"physics_profile": args.physics_profile}),
    }
    try:
        if args.devices is not None:
            binding["devices"] = args.devices
        if args.devices_table is not None:
            from woof.core.devices import DeviceOptions
            binding["devices_options"] = DeviceOptions.from_mapping(
                json.loads(args.devices_table), source="--devices-table")
        if getattr(args, "simulated_radar_table", None) is not None:
            from woof.simulated_radar_config import execution_argument
            binding["simulated_radar"] = execution_argument(args.simulated_radar_table)
        if (args.preparation_receipt_sha256 is None) \
                == (args.prepared_head_sha256 is None):
            raise ValueError(
                "bind the preparation with exactly one of "
                "--preparation-receipt-sha256 (a sealed tree) and "
                "--prepared-head-sha256 (a chained tree's head)")
        if args.prepared_head_sha256 is not None \
                and args.sealed_forcing_extension:
            raise ValueError(
                "--sealed-forcing-extension binds the sealed forcing "
                "prefix; bind the sealed tree with "
                "--preparation-receipt-sha256")
        try:
            inputs = preflight_prepared_tree(
                **binding,
                **({"prepared_head_sha256": args.prepared_head_sha256}
                   if args.prepared_head_sha256 is not None else
                   {"preparation_receipt_sha256":
                    args.preparation_receipt_sha256}),
            )
        except TreeHeadNeedsSeal as waiting:
            print("prepared tree: the forecast starts after the "
                  f"preparation seals: {waiting}",
                  file=sys.stderr, flush=True)
            inputs = preflight_prepared_tree(
                **binding, preparation_receipt_sha256=_sealed_proof_sha256(
                    args.prepared_root, args.prepared_head_sha256,
                    on_wait=_start_seal_waits(
                        outdir, args.prepared_root, observer)))
        # An experimental component option warns on every front door it
        # can be selected through, and this runner is one of them: a
        # domain tree is not runnable through `woof go`, which refuses
        # multi-domain configs and drives the single-domain runner.
        # Without this the tree path was the one way to select an
        # experimental closure and be told nothing.  One sentence, to
        # stderr, and the run continues (owner posture: warn-not-block).
        sentence = experimental_selection_sentence(
            domain.run for domain in inputs.experiment.domains)
        if sentence is not None:
            print(f"prepared tree: {sentence}", file=sys.stderr)
    except MissingTableAssets as error:
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        return 2
    except SourceBehind as behind:
        # The forecast waited for the seal before its first step and a
        # source lead passed its late time: a source behind, not a refusal.
        return _source_behind_exit(outdir, behind, observer)
    except ValueError as error:
        # Preflight is the stage whose whole job is to refuse before the
        # GPU is touched, and every one of its refusals is a ValueError
        # naming exactly what differs.  They used to escape as tracebacks
        # exiting 1 -- the same shape the run failures take -- so a
        # config problem and a crashed forecast were indistinguishable
        # to the caller.  Nothing has run here, so no failed-run receipt.
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        return 2
    first_products = prepared_single._route_owned_first_products(
        args, outdir=outdir, observer=observer, started=started)
    columns = prepared_single._route_owned_columns(
        outdir=outdir, observer=observer,
        experiment=getattr(inputs, "experiment", None))
    run_finished = False
    interrupted = False
    # --no-memory-gate skips the resident envelope admission inside, as it
    # skips `woof go`'s own gate; the constructor floors stay.
    from woof.core.resident_admission import memory_gate_override

    def run(bound, products, restart):
        def rearm(checkpoint):
            nonlocal first_products
            from woof.first_products import halt_renders_and_wait
            if not halt_renders_and_wait(first_products):
                raise RuntimeError("a failed-leg render is still reading "
                                   "history; refusing health recovery rewind")
            retry_args = argparse.Namespace(**{**vars(args), "restart": checkpoint})
            first_products = prepared_single._route_owned_first_products(
                retry_args, outdir=outdir, observer=observer, started=started)
            return first_products

        with memory_gate_override(args.no_memory_gate):
            return run_prepared_tree(
                bound,
                output_directory=outdir,
                io_mode=args.io_mode,
                restart=restart,
                health_debug=args.health_debug,
                observer=observer,
                sealed_forcing_extension=args.sealed_forcing_extension,
                progress_options=ProgressOptions.from_args(args),
                health_retry_products=rearm,
                columns=columns,
                **({} if products is None else {"first_products": products}),
            )

    try:
        clock_changed = None
        try:
            report = run(inputs, first_products, args.restart)
        except StreamedClockChanged as changed:
            # Only the reason leaves this handler.  The exception's traceback
            # holds the attempt's frames and, through them, its whole tree
            # on the card; a sealed run started in here restored a second
            # tree beside it.
            clock_changed = str(changed)
        rerun_reason = clock_changed
        if rerun_reason is not None:
            # The head-bound attempt stepped on a clock the sealed tree does
            # not choose, so it is not this tree's forecast: its outputs are
            # set aside (kept, named) and the forecast runs again on the
            # sealed tree, bound as a launch on its proof binds it.  Not
            # through _seal_tree_inputs, which holds the sealed clock to the
            # head's and so refuses exactly this tree.
            # First the heartbeat says a new attempt starts, so a supervisor
            # takes the step and model time going back to zero as that and
            # not as a regression, and times what follows as preparation;
            # a hosting observer ends the renders it owns here too.
            restart_attempt(observer, rerun_reason)
            rearm = first_products is not None
            if rearm:
                # Ended, and waited for, before its folder moves: nothing
                # is published into a picture folder that has moved, and
                # on Windows a folder with a file open in it does not move.
                from woof.first_products import halt_renders_and_wait

                if not halt_renders_and_wait(first_products):
                    print("prepared tree: a render of the head-bound "
                          "attempt was still running after it was killed",
                          file=sys.stderr, flush=True)
                first_products = None
            prepared_single._finish_owned_columns(columns, halt=True)
            columns = None
            _release_attempt_memory()
            attempt = _set_aside_streamed_attempt(outdir)
            print(f"prepared tree: {rerun_reason}; the head-bound attempt "
                  f"is kept in {attempt} and the forecast starts again on "
                  "the sealed preparation", file=sys.stderr, flush=True)
            rerun_args = args
            if clock_changed is not None and args.restart is not None:
                # The checkpoint was stepped on the head's clock, so no
                # resume from it is the sealed tree's forecast.
                print("prepared tree: the checkpoint this attempt resumed "
                      f"from ({args.restart}) was stepped on the head's "
                      "clock, so the sealed forecast runs from its start "
                      "time", file=sys.stderr, flush=True)
                rerun_args = argparse.Namespace(
                    **{**vars(args), "restart": None})
            inputs = preflight_prepared_tree(
                **binding, preparation_receipt_sha256=_sealed_proof_sha256(
                    args.prepared_root, args.prepared_head_sha256,
                    on_wait=_start_seal_waits(
                        outdir, args.prepared_root, observer)))
            if rearm:
                first_products = prepared_single._route_owned_first_products(
                    rerun_args, outdir=outdir, observer=observer,
                    started=started)
            columns = prepared_single._route_owned_columns(
                outdir=outdir, observer=observer,
                experiment=getattr(inputs, "experiment", None))
            report = run(inputs, first_products, rerun_args.restart)
        run_finished = True
    except SourceBehind as behind:
        return _source_behind_exit(outdir, behind, observer)
    except MissingTableAssets as error:
        # A refusal, not a failed run: no failed-run-receipt, because
        # nothing ran.  One sentence naming the table and the command
        # that stages it -- the shape every other guard in this main
        # already uses.
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        return 2
    except RestartMismatchError as error:
        # Also a refusal, and the guard is doing exactly the right
        # thing -- but it used to arrive as a 40-line traceback exiting
        # 1, where every sibling refusal in this main is one sentence
        # exiting 2.  Nothing was integrated, so there is no failed run
        # to write a receipt about.
        print(f"prepared_domain_tree_forecast: --restart refused: {error}",
              file=sys.stderr)
        return 2
    except NestDivideRefusal as error:
        # A REFUSAL, and a MID-RUN one: unlike every refusal above it,
        # the tree has integrated up to this boundary, so it still owes a
        # failed-run receipt.  What it does not owe is a traceback: this
        # arrived as the 40-line shape a crashed forecast takes, exiting
        # 1, which made the one guard whose message names an arithmetic
        # collapse AND its remedy the hardest of them to read.
        _write_failed_run_receipt(outdir, error)
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        if observer is not None:
            # A host emits its terminal failure from the exception. Returning
            # only 2 replaced this cause with an empty StageExitError event.
            raise
        return 2
    except InitializationMemoryRefused as error:
        # A memory refusal taken before the first device allocation: the
        # tree does not fit the card it was about to be restored onto.  One
        # sentence naming the terms and the remedy, exit 2, and a receipt
        # saying nothing ran.
        _write_failed_run_receipt(outdir, error)
        print(f"prepared_domain_tree_forecast: {error}",
              file=sys.stderr)
        return 2
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        _write_failed_run_receipt(outdir, error)
        raise
    finally:
        # A later forecast failure must not abandon an already dispatched
        # daemon render. Success was joined while composing the run report;
        # on failure, preserve the forecast's refusal/exception even if the
        # bounded render join itself fails or is interrupted.  A stop draws
        # nothing more (the desktop kills a run 5 s after asking); a
        # failure finishes drawing the frames it wrote.  The column store
        # likewise: a stop writes nothing more, a failure writes the frames
        # it committed.
        if not run_finished:
            prepared_single._finish_owned_columns(columns, halt=interrupted)
            columns = None
        if first_products is not None and not run_finished:
            try:
                if interrupted:
                    getattr(first_products, "halt", lambda: None)()
                else:
                    from woof.runtime import _finalizing_progress

                    _finalizing_progress(observer, "finish-first-products-after-failure")
                    first_products.wait()
            except BaseException as render_error:
                print("prepared_domain_tree_forecast: first-frame plot join "
                      f"failed: {type(render_error).__name__}: {render_error}",
                      file=sys.stderr)
    prepared_single._finish_owned_columns(columns)
    if report.get("schema") == "gpuwm-ensemble-run.v1":
        print(json.dumps({"schema": report["schema"], "status": report["status"],
            "members": report["request"]["members"], "completed_seconds": report["completed_seconds"],
            "ensemble_manifest": str(outdir / "ensemble-run.json")}, sort_keys=True))
        return 0
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness": report["readiness"],
                "plan_id": report["execution_plan"]["plan_id"],
                "domain_count": report["execution_plan"]["domain_count"],
                "wall_seconds": report["wall_seconds"],
                "frame_count": report["output"]["frame_count"],
                "run_receipt": str(
                    (outdir / "evidence" / "run-receipt.json").resolve()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
