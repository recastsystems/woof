"""Generic real-case experiment runtime.

The prepare -> static -> ingest -> initialize -> integrate -> output
pipeline for the experiment path, single-domain in this task (the
multi-domain tree/schedule executor is Task 14).  Every operation here
is a pure extraction of the frozen reference-case machinery: identical
operations in identical order on identical values, with the formerly
implicit constants (bundle paths, the case start time, the eta-level
table, the forcing-coverage ceiling, ``sfcp_to_sfcp``, the trace-gas
override, the output identity) replaced by :class:`ExperimentConfig` /
:class:`CaseDataConfig` values.  Operation order and operand identity stay
stable so frozen verification profiles remain byte-inert.

Layering: this module never imports case profile modules; the frozen
reference profile imports *this* module and feeds it the frozen pinned
values.  Grid construction still consumes a declared WPS namelist
for GEOG resolution selection only; Lambert geometry comes directly from
the experiment's ``ProjectionConfig`` and registered nest layout.

Equivalence notes (the argument the A/B gate checks empirically):

- The experiment path runs each per-domain :class:`RunConfig` exactly as
  the loader built it, with ``clock_dt = 0.0`` (retired, architecture
  section C).  Every ``clock_dt`` consumer -- ``lateral_boundary_clock_dt``
  (ingest/lateral_bc.py:214-217), KF/physics ``_model_clock_dt``
  (core/kf.py:22-31, core/physics.py:86-96), the RRTMGP interval
  derivation (core/rrtmgp.py), ``_clock_scaled_diff6_factor``
  (core/dycore.py:841-854), and the npref mirror (:3903) -- resolves
  ``clock_dt <= 0`` to ``cfg.dt``, so it is bit-equivalent to the frozen
  profile's ``clock_dt = dt`` integration transform at
  ``DYNAMICS_SUBSTEPS = 1``.  The frozen profile keeps passing its
  explicit ``integration_cfg`` through this loop unchanged.
- The radiation trace-gas override travels through
  :class:`woof.core.rrtmgp.RRTMGPRadiation`'s ``trace_gas_overrides``
  hook; a frozen profile's declared value is carried without conversion.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Mapping

from woof.core.restart_request import RESTART_REQUEST_ENV, RestartRequest

import numpy as np

from woof.case_data import CaseDataConfig
from woof.ingest.water_temperature import (
    WaterTemperatureStatics, resolve_water_temperature_policy)
from woof.certify.capsule import emit_run_capsule
from woof.config import (DEFAULT_COLUMN_CHUNK, RunConfig,
                          radiation_scheme_ids, soil_layer_count)
from woof.core.grid import make_vertical_coord
from woof.core.nest_lifecycle import (admit_restart_with_lifecycle,
                                       output_episode)
from woof.core.noah import noah_initial_snow_albedo
from woof.experiment import ExperimentConfig, VerticalConfig
from woof.ingest.grib import (cached_era5_forcing,
                               clear_forcing_caches)
from woof.ingest.horiz import interpolate_era5_to_lambert
from woof.ingest.soil_downscale import (
    declared_soil_texture_downscale, soil_mesh_plan_from_case)
from woof.ingest.lateral_bc import (StateBoundaryFrames,
                                     attach_lateral_boundaries)
from woof.ingest.preprocess_backend import (
    CudaPreprocessBackend,
    release_backend_memory,
)
from woof.ingest.real import initialize_real, surface_fields_to_device
from woof.ingest.land_state import land_state_context
from woof.ingest.ruc_soil import preprocess_land_surface_soil
from woof.moisture_floor_receipt import (
    MOISTURE_FLOOR_BY_DOMAIN_KEY, moisture_floor_block,
    moisture_floor_field_names)
from woof.static.sampling_contract import (current_sampling_contract,
                                            require_relocation_sampling_contract)
from woof.static.build import (GeogSelection, build_static,
                                monthly_interp_to_date)
from woof.static.lambert import grids_from_projection_config
from woof.static.orographic import with_terrain_drag_statics
from woof.ensemble.runtime_preparation import (
    current_runtime_preparation as _runtime_preparation_source,
    runtime_input_catalog as _runtime_input_catalog,
)


#: ``mp_physics`` values whose microphysics call stages a scheme-native
#: REFL_10CM field for the history writer to consume.  Named, not inlined,
#: because this runtime carries THREE separate gates on it (case output,
#: the per-substep ``refl_due`` schedule, and the nested-tree history
#: handoff) and an inlined tuple that is updated at two of the three is a
#: silent no-radar-data output frame, not an error.
#:
#: 28 (Thompson aerosol-aware) belongs here on WRF's own structure: WRF
#: reaches ``calc_refl10cm`` from the single call site
#: ``mp_gt_driver:1458``, gated on ``diagflag .and. do_radar_ref == 1``
#: (module_mp_thompson.F:1449) and never on ``is_aerosol_aware``.  woof
#: matches it -- ``woof/core/microphysics_aerosol.py`` stages the field
#: through the same ``compute_and_stash_refl_10cm`` seam as mp=8 -- so
#: excluding 28 here does not disable a diagnostic, it strands a field
#: that the scheme has already computed and that ``refl.py``'s
#: consume-once contract then reports as an unconsumed stash.
from woof.core.physics_inventory import REFL_10CM_MICROPHYSICS
from woof.io.history_layout import metadata_history_fields as _metadata_frame

MICROPHYSICS_TRANSITION_RECEIPT_NAME = "microphysics-transitions.json"
FEEDBACK_PROVENANCE_RECEIPT_NAME = "feedback-provenance.json"
INITIAL_PERTURBATION_RECEIPT_NAME = "initial-perturbation.json"
#: The long-step derivation of a ``woof run`` whose terrain clock changed
#: a domain (:func:`woof.terrain_clock.clock_receipt`).  Absent when no
#: domain changed, so such a run directory is the one it always was.
TERRAIN_CLOCK_RECEIPT_NAME = "terrain-clock.json"
#: The acoustic derivation of a ``woof run`` whose off-centering floor
#: raised a domain's default ``epssm``
#: (:func:`woof.acoustic_adaptation.acoustic_receipt`).  Absent when no
#: domain's epssm moved, so such a run directory is the one it always was;
#: the prepared and wrfinput doors carry the same record in their receipt.
ACOUSTIC_RECEIPT_NAME = "acoustic-offcentering.json"
FEEDBACK_EXPERIMENTAL_WARNING = (
    "WARNING: feedback = 1 is EXPERIMENTAL and is not certified against "
    "stock WRF yet; the certification reference is in progress."
)
CONSERVATION_CLOSURE_RECEIPT_NAME = "conservation-closure.json"


#: This route's name in every water-temperature refusal and receipt.  Not a
#: case name and not a file name: the ROUTE, so a false or missing assembly
#: can be attributed without reading a traceback.
_WATER_ROUTE = "the ERA5 runtime route"


# ---------------------------------------------------------------------------
# Prepared-case and run-summary containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedRealCase:
    """Setup-time inputs and live initial state for a real integration.

    ``final_analysis`` is the last forcing-time horizontal analysis
    (the verification profiles score against it);
    ``initial_snow_water_kgm2`` snapshots the t=0 Noah snow water for
    the snow-mask diagnostics.  Both are carried for the profile layer;
    the integration loop itself reads only ``cfg`` / ``grid`` /
    ``static_fields`` / ``initial_result``.
    """

    cfg: RunConfig
    grid: object
    static_fields: dict[str, np.ndarray]
    initial_result: object
    final_analysis: object
    initial_snow_water_kgm2: np.ndarray
    forcing_times: tuple[datetime, ...]
    geog_selection: GeogSelection | None = None
    store_input: object | None = None
    streamed_store: object | None = None
    initialization_receipt: dict | None = None
    static_sampling_contract: str | None = None
    #: Checkpoints written from this case carry the preserved forcing-prefix
    #: contract, and a restart into it is admitted only when the live forcing
    #: inventory keeps every interval the checkpoint was written under and
    #: appends after them.  A cycling run whose forcing is renewed between
    #: legs sets it; an ordinary run leaves the exact-setup restart rule.
    preserved_forcing_prefix: bool = False


@dataclass(frozen=True)
class RealCaseRunSummary:
    """Results from the config-driven integration surface.

    Contains no oracle or gate results: those comparisons belong
    exclusively to the verification profiles (``woof verify``).
    """

    wrfout_paths: tuple[Path, ...]
    nan_free: bool
    w_max_ms: float
    boundary_w_max_ms: float
    interior_w_max_ms: float
    w_max_boundary_row: int | None
    boundary_zone_blowup: bool
    dynamics_substeps: int
    ysu_nan_guard_fires: int
    surface_forcing_updates: int
    swdown_peak_wm2: float
    swdown_peak_time: datetime
    completed_seconds: float
    #: Final accumulated convective precipitation diagnostics; zero/None
    #: when cu_physics is off.
    rainc_max_mm: float = 0.0
    rainc_max_ji: tuple[int, int] | None = None
    rainc_max_lat: float | None = None
    rainc_max_lon: float | None = None
    #: Per-domain canonical trajectory digest, or ``None`` with the
    #: instrumentation disabled (see :func:`trajectory_digest_enabled`).
    trajectory_digest: dict | None = None
    #: The path/bytes/SHA-256 record for every emitted frame, hashed once
    #: during finalization and handed on so the supervisor's success
    #: capsule does not re-read the same bytes a second time.
    frame_records: tuple[Mapping[str, object], ...] = ()
    #: The floors the run route's own capsule recorded, handed to the
    #: supervisor exactly as :class:`ExperimentRunSummary` hands them.  The
    #: single-domain route of ``run_experiment`` sets it through
    #: ``dataclass_replace``; without the field that call raised TypeError
    #: after the forecast had finished, so a single-domain config-driven run
    #: wrote every frame and then failed before rendering.
    moisture_floor_receipts: Mapping[str, object] | None = None
    hydro_receipt: Mapping[str, object] | None = None
    #: The capsule ``receipts`` fragment naming the hydro decision, cost and
    #: products (:func:`_hydro_run_receipts`), carried to the supervisor's
    #: success capsule for the reason ``moisture_floor_receipts`` is.
    hydro_receipts: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ExperimentRunSummary:
    """Tree-run handoff consumed by the CLI and supervisor."""

    wrfout_paths: tuple[Path, ...]
    completed_seconds: float
    nan_free: bool
    last_checkpoint: Path | None = None
    microphysics_transitions: tuple[Mapping[str, object], ...] = ()
    microphysics_transition_receipt: Path | None = None
    microphysics_transition_receipt_sha256: str | None = None
    feedback_provenance: Mapping[str, object] | None = None
    feedback_provenance_receipt: Path | None = None
    feedback_provenance_receipt_sha256: str | None = None
    #: Per-domain canonical trajectory digest, or ``None`` with the
    #: instrumentation disabled (see :func:`trajectory_digest_enabled`).
    trajectory_digest: dict | None = None
    #: The path/bytes/SHA-256 record for every emitted frame, hashed once
    #: during finalization and handed on so the supervisor's success
    #: capsule does not re-read the same hundreds of GiB a second time.
    frame_records: tuple[Mapping[str, object], ...] = ()
    #: The capsule ``receipts`` fragment this run's own capsule stated
    #: about its initialization -- ``{"moisture_floors_by_domain": {...}}``
    #: -- or ``None`` from a route that emitted no front-door capsule.
    #: Carried for the same reason as ``frame_records`` and for one more:
    #: `woof run` SUPERVISES by default, and the supervisor writes its
    #: success capsule into the same directory under the same fixed name
    #: AFTER this one, so what the run route recorded and did not hand
    #: back was replaced rather than kept.
    moisture_floor_receipts: Mapping[str, object] | None = None
    hydro_receipt: Mapping[str, object] | None = None
    #: The capsule ``receipts`` fragment naming the hydro decision, cost and
    #: products (:func:`_hydro_run_receipts`), carried to the supervisor's
    #: success capsule for the reason ``moisture_floor_receipts`` is.
    hydro_receipts: Mapping[str, object] | None = None
    ensemble_manifest: Path | None = None
    ensemble_manifest_sha256: str | None = None


#: Environment switch that turns the trajectory-digest instrumentation off.
#: The A4 control pair runs the same short-window config with the digest on
#: and off; if the instrumentation participated in the trajectory the two runs
#: would differ, and the shipped comparator is what says whether they do.
TRAJECTORY_DIGEST_ENV = "WOOF_TRAJECTORY_DIGEST"


def trajectory_digest_enabled() -> bool:
    """Whether the run-route trajectory digest is computed.  Default on."""
    import os

    raw = os.environ.get(TRAJECTORY_DIGEST_ENV)
    return True if raw is None else raw.strip().lower() not in {
        "0", "false", "no", "off"}


class _SingleDomainDigestClock:
    """The boundary-clock inputs the frozen single-domain loop owns.

    ``canonical_state_digest`` mixes WRF's REAL boundary accumulator
    ``dtbc`` into the digest.  The frozen single-domain loop maintains no
    such accumulator: ``dtbc`` is a domain-tree coupler property
    (``woof/core/model.py`` root-boundary section) and lateral forcing on
    this route is applied from the configured interval.  Reporting the
    ``DomainClock`` initial value keeps the digest a complete, reproducible
    function of the state this route does own, and the run summary records
    that the route does not advance it, so nobody reads the digest as
    evidence about a boundary clock that never ran.
    """

    __slots__ = ()

    #: ``woof.core.clock.DomainClock`` initialises ``dtbc_fp32`` to +0.0f.
    dtbc_fp32 = np.float32(0.0)

    #: Recorded beside the digest so the omission is legible in the capsule.
    provenance = (
        "the frozen single-domain loop maintains no WRF dtbc accumulator; "
        "the digest carries the DomainClock initial value")


def _frame_records(paths, *, progress_callback=None, completed_records=()
                   ) -> list[dict[str, object]]:
    """Retain completed writer identities, hashing legacy files as needed.

    The shared owner checks each fresh record's file revision before reuse.
    Files without a current writer proof still receive a complete stable
    hash, with progress before each file so fallback work stays observable.
    Each beat declares the file's size, the most this record can read, so
    the supervisor bounds a multi-GiB frame by its bytes and not by the
    model step.
    Missing history is named and recorded without failing a completed
    forecast. The final-state digest comes from live state independently.
    """
    from woof.output_identity import file_records

    total = len(paths)
    def beginning(index, path):
        try:
            size = os.stat(path).st_size
        except OSError:
            # file_record raises the real error for this path next.
            size = None
        _finalizing_progress(
            progress_callback, f"hash-output-frames-{index}-of-{total}",
            work_bytes=size)
    return file_records(paths, completed=completed_records,
                        before_record=beginning, allow_missing=True)


#: What the run route says for a domain whose prepared case holds no
#: moisture-floor field: a restored idealized stand-in, or a case built by
#: an ingest older than the receipt.  Never "nothing was floored" -- this
#: route did not observe that.
_RUN_FLOORS_UNRECORDED = (
    "this domain's prepared case carries no moisture-floor field, so its "
    "initial state came from a stand-in or from an ingest predating the "
    "receipt; re-run from forcing to record whether its vapour was floored "
    "on the way in")


def _run_moisture_floor_receipts(prepared_cases) -> dict[str, object]:
    """The per-domain floor blocks for a front-door run's capsule.

    WHY THE RUN ROUTE NEEDS ITS OWN.  ``proof.json`` is a prepared
    bundle's document; `woof go` and `woof run` take the experiment
    route, which writes a certification capsule and no proof at all.  The
    floors were therefore reachable only from `woof prep` and the direct
    adapters -- not from the door most runs go through -- so a forecast
    whose initial vapour was modified on the way in still looked exactly
    like one whose was not, in the only document that run produced.

    NON-FATAL, per domain.  This is assembled after the last model step,
    beside the frame hashes, and a receipt must never turn a finished
    forecast into a crash.  A domain whose block cannot be built says so
    in its own block instead of vanishing from the mapping, because a
    missing domain is the absence this receipt exists to prevent.
    """

    cases = (dict(prepared_cases) if isinstance(prepared_cases, Mapping)
             else dict(prepared_cases or {}))
    blocks: dict[str, object] = {}
    for grid_id, case in sorted(cases.items()):
        label = f"d{int(grid_id):02d}"
        try:
            blocks[label] = moisture_floor_block(
                getattr(case, "initial_result", None),
                when_unrecorded=_RUN_FLOORS_UNRECORDED)
        except Exception as error:  # noqa: BLE001 - the forecast stands
            blocks[label] = {
                "recorded": False,
                "not_recorded_because": (
                    "this domain's floor receipt could not be read from its "
                    f"prepared case: {type(error).__name__}: {error}"),
            }
    return {MOISTURE_FLOOR_BY_DOMAIN_KEY: blocks} if blocks else {}


def _emit_front_door_capsule(outdir, *, emission_site: str, exp,
                             data: CaseDataConfig, wrfout_paths,
                             trajectory_digest, io_mode: str,
                             frame_records=None,
                             progress_callback=None,
                             prepared_cases=None,
                             receipts=None) -> tuple[Path, dict[str, object]]:
    """Write the front door's certification capsule.

    Unconditional: it does not consult ``exp.feedback``, because a receipt
    that appears only on one physics tier is a receipt the other tier cannot
    be certified from.  ``receipts`` is the optional receipts-section
    mapping (the spectral seam's run receipts arrive through it).

    Returns the capsule's path AND the moisture-floor fragment it stated,
    so the run summary can carry to the supervisor exactly what this
    capsule says.  Built here and handed back rather than rebuilt by the
    caller: one builder, two documents, no chance of the two disagreeing.
    """
    run_context = {
        "runner_route_and_io_mode": {
            "route": emission_site, "io_mode": io_mode},
        "output_and_diagnostic_mode": {
            "io_mode": io_mode,
            "history_interval_seconds": float(exp.domains[0].history_interval_s)
            if getattr(exp.domains[0], "history_interval_s", None) is not None
            else None,
            "restart_interval_seconds": (
                None if exp.restart_interval_s is None
                else float(exp.restart_interval_s)),
        },
    }
    run_shape = {
        "route": emission_site,
        "domain_count": len(exp.domains),
        "run_seconds": float(exp.run_seconds),
        "start_time": exp.start_time.isoformat(),
        "output_title": data.output_title,
    }
    output = {
        "frames": (list(frame_records) if frame_records is not None
                   else _frame_records(
                       wrfout_paths, progress_callback=progress_callback)),
        "trajectory_digest": trajectory_digest,
    }
    # The initialization receipts this run's own capsule can state.  Merged
    # here rather than at each call site so a third emitting route cannot
    # be the one that forgets them.
    floors = _run_moisture_floor_receipts(prepared_cases)
    run_receipts = dict(receipts) if receipts else {}
    run_receipts.update(floors)
    capsule = emit_run_capsule(
        outdir, emission_site=emission_site, run_context=run_context,
        run_shape=run_shape, output=output,
        receipts=run_receipts or None)
    return capsule, floors


def _hydro_run_receipts(hydro_receipt) -> dict[str, object]:
    """The hydro decision, cost and products this route's receipts state.

    ``hydro_receipt`` is one :func:`woof.hydro.forecast_products.finalize_forecast`
    block (single domain) or a ``{"dNN": block}`` map (tree).  The top-level
    cost and products are the root domain's, as the prepared tree reporter
    states them; a multi-domain tree also carries every domain's block.
    Without this the ERA5/[case_data] route wrote no hydro_activation,
    hydro_cost or hydro_products anywhere (OKC 2024-05-06T18, 2026-10-20).
    """
    from woof.hydro.runtime import activation_receipt, current_settings
    fragment: dict[str, object] = {
        "hydro_activation": activation_receipt(current_settings())}
    blocks = hydro_receipt or {}
    if blocks and all(str(key).startswith("d") and isinstance(value, Mapping)
                      for key, value in blocks.items()) \
            and "hydro_cost" not in blocks:
        root = blocks[sorted(blocks)[0]]
        if len(blocks) > 1:
            fragment["hydro_by_domain"] = dict(blocks)
        blocks = root
    for key in ("hydro_cost", "hydro_products"):
        if isinstance(blocks, Mapping) and key in blocks:
            fragment[key] = blocks[key]
    return fragment


RUNTIME_REPORT_SCHEMA = "gpuwm-runtime-run-report.v1"


def _write_runtime_report(outdir: Path, exp: ExperimentConfig, summary, *,
                          wall_seconds: float) -> Path:
    """``report.json`` for the config-driven route.

    The prepared runners write ``report.json`` (status, timing_seconds,
    health, hydro_*); this route wrote none, so the doors and
    ``runplan._chain_summary`` read no status, timing or hydro record from
    an ERA5 forecast.  Same file name and the same top-level keys.
    """
    nan_free = bool(getattr(summary, "nan_free", False))
    health = {"nan_free": nan_free}
    for key in ("w_max_ms", "boundary_w_max_ms", "interior_w_max_ms",
                "boundary_zone_blowup", "ysu_nan_guard_fires"):
        value = getattr(summary, key, None)
        if value is not None:
            health[key] = value.item() if hasattr(value, "item") else value
    report = {
        "schema": RUNTIME_REPORT_SCHEMA,
        "status": "PASS" if nan_free else "FAIL",
        "route": "runtime.run_experiment",
        "experiment": exp.name,
        "start_time": exp.start_time.isoformat(),
        "run_seconds": float(exp.run_seconds),
        "completed_seconds": float(summary.completed_seconds),
        "domain_count": len(exp.domains),
        "wrfout_count": len(summary.wrfout_paths),
        "timing_seconds": {"total": float(wall_seconds)},
        "health": health,
        **(dict(summary.hydro_receipts) if summary.hydro_receipts else {}),
    }
    path = Path(outdir) / "report.json"
    tmp = path.with_name(f".report.json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
    os.replace(tmp, path)
    return path


def feedback_provenance(exp: ExperimentConfig) -> Mapping[str, object] | None:
    """Machine-readable truth label for the experimental feedback tier."""
    if int(exp.feedback) != 1:
        return None
    vertical_mapping = (
        "shared-explicit-eta-ladder-horizontal-only"
        if exp.vertical.eta_levels
        else "shared-legacy-level-count-horizontal-only")
    return {
        "schema": "gpuwm-experimental-feedback-provenance-v1",
        "feedback": "experimental",
        "feedback_value": 1,
        "stock_wrf_certification": "reference-in-progress",
        "stock_wrf_certified": False,
        "restriction": "wrf-v4.6.1-copy_fcn-horizontal",
        "vertical_mapping": vertical_mapping,
    }


def _write_feedback_provenance_receipt(
        outdir: Path, exp: ExperimentConfig, *, resumed: bool
        ) -> tuple[Path | None, str | None, Mapping[str, object] | None]:
    """Atomically stamp feedback truth into the durable run provenance."""
    provenance = feedback_provenance(exp)
    if provenance is None:
        return None, None, None
    payload = dict(provenance)
    payload.update({
        "experiment": exp.name,
        "resumed": bool(resumed),
        "domain_ids": [int(domain.grid_id) for domain in exp.domains],
    })
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n").encode("utf-8")
    path = Path(outdir) / FEEDBACK_PROVENANCE_RECEIPT_NAME
    # Unique temp and an explicit durability barrier, matching the
    # microphysics transition receipt below.  A fixed temp name is not
    # reentrant outside the supervisor's serialization, and a receipt whose
    # bytes never left the page cache is a receipt a power failure can
    # unwrite after the rename has already made it look durable.
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path, hashlib.sha256(encoded).hexdigest(), payload


def _write_initial_perturbation_receipt(
        outdir: Path, exp: ExperimentConfig, domain_receipts
        ) -> Path | None:
    """Atomically publish what the [perturbation] bubbles actually wrote.

    The treatment-proof receipt: the accepted config echoed value for
    value, plus each initialized domain's per-bubble application stats
    (cells touched, max theta delta, qv adjustment).  ``None`` -- and no
    file -- when the experiment carries no block, so an absent block
    leaves the run directory byte-identical.  Written as soon as the
    initial states exist, before integration, so even a run that dies
    mid-flight proves its arm.
    """
    if exp.perturbation is None:
        return None
    payload = {
        "schema": "gpuwm-initial-perturbation-receipt-v1",
        "experiment": exp.name,
        "config": exp.perturbation.receipt(),
        "domains": [dict(receipt) for receipt in domain_receipts],
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n").encode("utf-8")
    path = Path(outdir) / INITIAL_PERTURBATION_RECEIPT_NAME
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    for receipt in payload["domains"]:
        for row in receipt.get("bubbles", ()):
            if row.get("applied"):
                print(
                    f"initial perturbation: bubble {row['bubble']} on "
                    f"d{receipt['grid_id']:02d} touched "
                    f"{row['cells_touched']} cells, max theta delta "
                    f"{row['max_theta_added_k']:.3f} K"
                    + (", max qv delta "
                       f"{row['max_qv_delta_kg_kg']:.3e} kg/kg"
                       if "max_qv_delta_kg_kg" in row else ""))
            else:
                print(
                    f"initial perturbation: bubble {row['bubble']} on "
                    f"d{receipt['grid_id']:02d} not applied "
                    f"({row.get('reason', 'unstated')})")
    return path


def _write_microphysics_transition_receipt(
        outdir: Path, model, exp: ExperimentConfig, *, resumed: bool
        ) -> tuple[Path, str, tuple[Mapping[str, object], ...]]:
    """Atomically publish the executable edge policies and force coverage."""

    transitions = tuple(
        dict(node.coupler.transition_receipt())
        for node in model.nodes_by_grid_id.values()
        if node.coupler is not None)
    for edge in transitions:
        count = int(edge["process_force_count"])
        interval = int(edge["parent_interval_ticks"])
        start = int(edge["process_start_parent_ticks"])
        final = int(edge["final_parent_ticks"])
        first = edge["first_parent_ticks"]
        last = edge["last_parent_ticks"]
        valid_observation = (
            edge.get("current_process_coverage_complete") is True
            and interval > 0 and final >= start
            and (final - start) % interval == 0
            and ((count == 0 and first is None and last is None)
                 or (count > 0
                     and int(first) == start + interval
                     and int(last) == final
                     and int(last) - int(first) == (count - 1) * interval)))
        # EITHER form proves the invariant.  The tick arithmetic above is
        # exact while every parent step is the same size; the step-count
        # identity is exact always, and is the only one that holds under
        # an adaptive clock.  Under a fixed clock both are true, so the
        # existing path is unchanged and nothing already-green moves.
        if not valid_observation:
            valid_observation = bool(
                edge.get("force_count_matches_parent_steps"))
        if not valid_observation:
            raise RuntimeError(
                "microphysics transition force coverage is incomplete or "
                f"internally inconsistent: {edge}")
    payload = {
        "schema": "gpuwm.microphysics-transitions/v1",
        "status": "PASS",
        "experiment": exp.name,
        "experiment_fingerprint": model.experiment_fingerprint,
        "completed_seconds": model.root.clock.elapsed_seconds,
        "resumed_process": bool(resumed),
        "run_id": os.environ.get("WOOF_RUN_ID"),
        "config_digest": os.environ.get("WOOF_CONFIG_DIGEST"),
        "transitions": transitions,
    }
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8")
    path = outdir / MICROPHYSICS_TRANSITION_RECEIPT_NAME
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path, hashlib.sha256(encoded).hexdigest(), transitions


# ---------------------------------------------------------------------------
# Static/grid/vertical building blocks
# ---------------------------------------------------------------------------

_STATIC_BUILD_CACHE: dict[tuple, dict] = {}


def _cached_static_build(grid, geog_root, *,
                         selection: GeogSelection | None = None) -> dict:
    """Memoize :func:`build_static` for repeated case preparation.

    Several tests prepare a case independently; the WPS_GEOG tile build
    is pure and deterministic for a given grid, so it is shared.  Cached
    NumPy arrays are locked read-only: every consumer in this module
    derives new arrays rather than mutating the static fields, and the lock
    turns any future in-place write into an immediate ``ValueError`` instead
    of silent cross-test contamination.  Model state is still rebuilt from
    scratch on every call.
    """
    selection = (GeogSelection.fallback(geog_root) if selection is None
                 else selection)
    key = (str(geog_root), selection, grid.map_proj, grid.ref_lat,
           grid.ref_lon, grid.truelat1, grid.truelat2, grid.stand_lon,
           grid.dx, grid.e_we, grid.e_sn)
    hit = _STATIC_BUILD_CACHE.get(key)
    if hit is None:
        hit = build_static(grid, geog_root, selection=selection)
        for value in hit.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        _STATIC_BUILD_CACHE[key] = hit
    return hit


def load_source_orography(path, variable: str) -> np.ndarray:
    """Read the declared source-orography artifact (NetCDF, record 0)."""
    from woof import netcdf_bridge

    with netcdf_bridge.open_dataset(path) as ds:
        if variable not in ds.variables:
            raise ValueError(
                f"declared source-orography variable {variable!r} is not "
                f"in {path}; available: {sorted(ds.variables)}")
        return np.asarray(ds.variables[variable][0], dtype=np.float64)


def vertical_coord_for(vertical: VerticalConfig, nz: int):
    """The single-source vertical grid (F1 amendment, G4).

    ``eta_levels``/``p_top``/hybrid selectors come from the experiment's
    one :class:`VerticalConfig` -- never from per-domain RunConfig
    fields.  Two experiments differing in any eta level build two
    distinct grids from config alone (G4 completion).
    """
    if not vertical.eta_levels:
        raise ValueError(
            "the experiment runtime requires explicit eta_levels in "
            "[shared] (VerticalConfig.eta_levels); idealized nz/ztop "
            "scaffolds have no real-data vertical grid")
    eta = np.asarray(vertical.eta_levels, dtype=np.float64)
    return make_vertical_coord(nz, hybrid_opt=vertical.hybrid_opt,
                               etac=vertical.etac, eta_levels=eta)


def single_domain(exp: ExperimentConfig):
    """The one DomainConfig this task's runtime integrates (fail-loud)."""
    if len(exp.domains) != 1:
        raise NotImplementedError(
            f"experiment {exp.name!r} declares {len(exp.domains)} domains; "
            "the Task-2 runtime integrates a single domain -- the "
            "multi-domain tree/schedule executor lands in Task 14.")
    return exp.domains[0]


def experiment_grid(exp: ExperimentConfig, data: CaseDataConfig):
    """Build the case grid directly from projection/domain config."""
    dc = single_domain(exp)
    if dc.grid_id != data.output_domain:
        raise ValueError(
            f"[case_data] output_domain = {data.output_domain} does not "
            f"name the experiment's domain (grid_id = {dc.grid_id}).")
    grids = grids_from_projection_config(exp)
    grid = grids[exp.domains.index(dc)]
    cfg = dc.run
    if (cfg.nx, cfg.ny, cfg.dx, cfg.dy) != (
            grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy):
        raise ValueError(
            "experiment domain grid does not match its resolved config: "
            f"got {(cfg.nx, cfg.ny, cfg.dx, cfg.dy)}, "
            f"expected {(grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)}")
    return grid


# ---------------------------------------------------------------------------
# Forcing discovery and the coverage-derived run ceiling
# ---------------------------------------------------------------------------

def forcing_snapshots(data: CaseDataConfig, input_catalog=None) -> dict:
    source = _runtime_preparation_source()
    if source is None:
        return _forcing_snapshots(data, input_catalog)
    catalog = _runtime_input_catalog(data) if input_catalog is None else input_catalog
    return source.forcing_snapshots(data, catalog,
        build=lambda: _forcing_snapshots(data, catalog))


def _forcing_snapshots(data: CaseDataConfig, input_catalog=None) -> dict:
    """Decode forcing under one input catalog's valid-time authority.

    The catalog is built here when a caller has not already built it.  Runtime
    decode then merges all declared products together and passes the catalog's
    exact selected/excluded times into the decoder; it never re-derives a
    schedule from records in the raw files.
    """
    if input_catalog is None:
        input_catalog = _runtime_input_catalog(data)

    forcing_hashes = {
        Path(record.path).resolve(): record.sha256
        for record in getattr(input_catalog, "files", ())
        if record.role == "forcing"
    }
    forcing_identities = (
        data.forcing_identity()
        if hasattr(data, "forcing_identity") else data.forcing)
    content_sha256 = tuple(
        forcing_hashes.get(Path(path).resolve(), "")
        for path in forcing_identities
    )
    if not all(content_sha256):
        content_sha256 = None

    from woof.ingest.grib import forcing_container
    decoded = cached_era5_forcing(
        data.forcing, data.vtable, content_sha256=content_sha256,
        valid_times=input_catalog.valid_times,
        excluded_valid_times=input_catalog.excluded_valid_times,
        container=forcing_container(forcing_identities),
    )
    by_time: dict[datetime, object] = {}
    for snapshot in decoded.snapshots:
        if snapshot.valid_time in by_time:
            raise ValueError(
                f"duplicate forcing snapshot at {snapshot.valid_time}.")
        by_time[snapshot.valid_time] = snapshot
    actual = tuple(by_time)
    expected = tuple(input_catalog.valid_times)
    if actual != expected:
        raise ValueError(
            "runtime forcing decode did not consume the catalog's exact "
            f"ordered valid-time selection: expected {expected}, got {actual}; "
            "catalog exclusions: "
            f"{tuple(input_catalog.excluded_valid_times)}")
    # Optional hi-res water-temperature overlay (task #71).  The decode
    # above serves RAW snapshots from the process cache, so the declared
    # overlay is applied here, mirroring build_input_catalog's
    # application to the catalog's own served snapshots (the nested-child
    # source).  Absent key: this branch never runs.
    water_overlay_path = getattr(data, "water_temperature_overlay", None)
    if water_overlay_path is not None:
        from woof.ingest.water_overlay import (
            cached_water_temperature_overlay,
            overlay_snapshots_by_time,
        )

        overlay = cached_water_temperature_overlay(water_overlay_path)
        by_time, receipt = overlay_snapshots_by_time(by_time, overlay)
        print(
            "water-temperature overlay: replaced "
            f"{receipt['replaced_cells']} of {receipt['water_cells']} "
            f"water source cells per snapshot from {receipt['path']} "
            f"({receipt['fallback_cells']} kept ERA5 fallback)")
    return by_time


def forcing_schedule(exp: ExperimentConfig, data: CaseDataConfig,
                     available_times) -> tuple[datetime, ...]:
    """Validated forcing valid-time schedule from ``start_time`` onward.

    The run ceiling is DERIVED from validated forcing coverage: a
    ``run_seconds`` beyond the last usable forcing time is rejected here
    with the coverage stated (the frozen case's fixed-constant ceiling
    left the runtime path with this function).  A declared
    ``forcing_interval_s`` policy is enforced against the discovered
    schedule; without it, discovery accepts the file's own spacing.
    """
    times = sorted(available_times)
    if exp.start_time not in times:
        raise ValueError(
            f"forcing has no snapshot at the experiment start_time "
            f"{exp.start_time}; decoded valid times: {times}.")
    usable = tuple(t for t in times if t >= exp.start_time)
    if len(usable) < 2:
        raise ValueError(
            f"forcing declares only {len(usable)} snapshot(s) at/after "
            f"start_time {exp.start_time}; lateral-boundary forcing "
            "requires at least one interval (two valid times).")
    if data.forcing_interval_s is not None:
        for earlier, later in zip(usable, usable[1:]):
            delta = (later - earlier).total_seconds()
            if delta != data.forcing_interval_s:
                raise ValueError(
                    f"declared forcing_interval_s = "
                    f"{data.forcing_interval_s:g} but the decoded "
                    f"schedule steps {earlier} -> {later} "
                    f"({delta:g} s).")
    coverage = (usable[-1] - usable[0]).total_seconds()
    if exp.run_seconds > coverage:
        raise ValueError(
            f"run_seconds = {exp.run_seconds:g} exceeds the validated "
            f"forcing coverage of {coverage:g} s ({usable[0]} .. "
            f"{usable[-1]}); shorten the run or declare more forcing.")
    return usable


def _snapshot_host_bytes(snapshot) -> int:
    """Host bytes one decoded forcing snapshot holds, over its own arrays."""

    total = sum(int(getattr(value, "nbytes", 0))
                for value in getattr(snapshot, "fields", {}).values())
    for name in ("levels_hpa", "latitude", "longitude"):
        total += int(getattr(getattr(snapshot, name, None), "nbytes", 0))
    return total


def forcing_decode_report(exp: ExperimentConfig, snapshots) -> str:
    """What the forcing decode cost, in valid times and host bytes.

    THE COUNT IS NOT THE FORECAST'S.  Which valid times get decoded is
    decided by the input catalog, and
    :func:`woof.ingest.preflight.build_input_catalog` takes ``case_data``
    ALONE: it selects the longest contiguous run of times present in the
    forcing FILES and never sees ``run_seconds``.  So a user who fetched
    a longer window than they integrate decodes the whole window into
    host memory, shortening the run does not shorten the decode, and no
    preflight says so -- ``woof check``'s ingest itemization prices
    DEVICE memory by its own definition
    (:class:`woof.core.preflight.IngestMemoryEstimate`).  This line is
    where that window is named.

    "BEYOND THE END", NOT "UNREAD".  A time past the run's end is NOT
    surplus to the prepared case, and saying so would be false in the
    same breath that says it: :func:`forcing_schedule` returns EVERY
    decoded time at or after ``start_time`` -- it truncates at nothing --
    and :func:`prepare_real_case` loops over all of them, interpolating
    each, building a state for each, adding each to the boundary frames,
    making an interval of every consecutive pair, and keeping the LAST as
    the case's final analysis.  So the remedy is real but it is not free:
    a narrower forcing window changes ``forcing_times``, the lateral
    boundary intervals and ``final_analysis``, and this says so where the
    reader will act on it.

    ``exp.run_seconds`` is the CONFIG's run length, which is what the
    sentence calls it: :mod:`woof.ensemble.member` overrides the leg
    length with its own argument after loading the pair, and the
    experiment this is handed is still the config's.

    ``snapshots`` is the ``{valid_time: Era5Snapshot}`` mapping
    :func:`forcing_snapshots` returned, so the byte figures are the
    arrays this run is actually holding, summed -- not an estimate from
    a grid shape.
    """

    times = sorted(snapshots)
    if not times:
        return "forcing decode: no valid times decoded"
    held = {value: _snapshot_host_bytes(snapshots[value]) for value in times}
    end = exp.start_time + timedelta(seconds=float(exp.run_seconds))
    needed: list[datetime] = []
    for value in times:
        if value < exp.start_time:
            continue
        needed.append(value)
        if value >= end:
            break
    covered = set(needed)
    # The two kinds of extra, which have different consequences and so
    # are not one number: a time BEFORE start_time never enters the
    # preparation schedule at all, and a time beyond the run's end does.
    early = tuple(value for value in times if value < exp.start_time)
    beyond = tuple(value for value in times
                   if value >= exp.start_time and value not in covered)
    gib = 1024 ** 3
    lines = [
        f"forcing decode: {len(times)} valid times, "
        f"{sum(held.values()) / gib:.2f} GiB of host memory (float64, on "
        f"the source grid); this config's {exp.run_seconds:g} s run "
        f"integrates to {end.isoformat()} and needs {len(needed)} of them."
    ]
    if early:
        lines.append(
            f"  {len(early)} lie before start_time "
            f"{exp.start_time.isoformat()} and hold "
            f"{sum(held[value] for value in early) / gib:.2f} GiB nothing "
            "reads: the preparation schedule begins at start_time.")
    if beyond:
        lines.append(
            f"  {len(beyond)} lie beyond that end and hold "
            f"{sum(held[value] for value in beyond) / gib:.2f} GiB.  They "
            "are still PREPARED -- every decoded time at or after "
            "start_time is interpolated, initialized, kept as a boundary "
            "frame, and the last one is this case's final analysis -- so "
            "refetching a narrower window drops them from the prepared "
            "case as well as from memory.")
    if early or beyond:
        lines.append(
            "  The decoded window comes from the forcing FILES, not from "
            "run_seconds: build_input_catalog never sees it, so a shorter "
            "run does not shorten the decode.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prepare: static -> ingest -> initialize (the extracted case machinery)
# ---------------------------------------------------------------------------

def declared_constant_glw(exp: ExperimentConfig) -> float | None:
    """The constant downward longwave this experiment DECLARED, or None.

    A real case reaches a preparer only after
    :func:`woof.experiment.build_experiment` has run
    :func:`woof.physics_compat.constant_longwave_refusal`, so a config
    that runs a land-surface scheme with ``ra_lw_physics = 0`` and got
    this far carries
    :data:`~woof.physics_compat.CONSTANT_DOWNWARD_LONGWAVE_ACK`.  This
    turns that declaration into the number the preparer TYPES into
    :func:`~woof.core.physics.initialize_physics`, because that function
    refuses to invent one.

    The number is the experiment's own when it declared one --
    ``[experiment] constant_glw_wm2`` -- and the shipped
    :data:`~woof.core.physics.DECLARED_CONSTANT_GLW_WM2` otherwise.  The
    acknowledgement names the CLAIM (this run fabricates its downward
    longwave); the field names the NUMBER, and until it existed an
    experiment whose case radiates near 410 W m-2 had to run at 300 and
    call the difference declared.  ``initialize_physics`` accepts any
    float and the run receipt prints the value, so nothing else changes.

    ``None`` for every other experiment, which is the normal answer: a
    run with a longwave scheme has its GLW written by that scheme, and
    passing a value would only pre-fill a buffer the scheme overwrites.
    """

    from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

    if CONSTANT_DOWNWARD_LONGWAVE_ACK in tuple(exp.acknowledgements or ()):
        from woof.core.physics import DECLARED_CONSTANT_GLW_WM2
        declared = getattr(exp, "constant_glw_wm2", None)
        return (DECLARED_CONSTANT_GLW_WM2 if declared is None
                else float(declared))
    return None


def _initialize_real_case_physics(
        initial_result, cfg, initial_met, soil, soil_fields, static,
        landuse_attrs, grid, start_time, *, vertical, reconciled_soil_type,
        trace_gas_overrides=None, radiation_column_chunk=DEFAULT_COLUMN_CHUNK,
        constant_glw_wm2=None, center_lat=None, cam_ozone=None):
    """Initialize a whole case or a row window with the same operands.

    The caller owns soil reconciliation and the surface solution. A row
    loader must pass their exact windows and the whole domain's center
    latitude, so storage geometry cannot change the land-use season,
    radiation composition, or configured column chunk.
    """
    from woof.core.diagnostics import update_diagnostics
    from woof.core.landuse import (initialize_landuse,
                                    ruc_fractional_seaice,
                                    usemonalb_landuse_inputs)
    from woof.core.physics import initialize_physics
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    from woof.ingest.lake_physics import lake_physics_inputs

    # WRF interpolates GREENFRAC/LAI to the run date
    # (module_initialize_real.F:1322-1335, mid-month anchors); shdmin/
    # shdmax stay the monthly extrema (:1348-1351).  With usemonalb=false,
    # landuse_init overwrites ALBEDO12M from the table; with it true the
    # monthly field stays (usemonalb_landuse_inputs).
    from woof.ingest.vegetation import initial_vegetation_fraction
    vegfra = initial_vegetation_fraction(initial_met, static, start_time)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], start_time)
    state = initial_result.state
    # initialize_real loads prognostics but does not launch the EOS kernel.
    # Diagnose the time-zero atmosphere before the first RRTMGP call.
    update_diagnostics(state, cfg.hypsometric_opt)
    lat, lon = grid.latlon_mass()
    from woof.core.radiation_composition import make_radiation
    radiation = make_radiation(
        cfg, start_time, lat, lon, p_top=vertical.p_top,
        trace_gas_overrides=trace_gas_overrides,
        column_chunk=radiation_column_chunk)
    landuse = initialize_landuse(
        static["LU_INDEX"], urban_legend=int(getattr(cfg, "sf_urban_physics", 0)) > 0,
        soil_type=reconciled_soil_type,
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=start_time,
        cen_lat=(float(getattr(grid, "cen_lat", np.mean(lat)))
                  if center_lat is None else float(center_lat)),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=soil.soil_temperature,
        fractional_seaice=ruc_fractional_seaice(cfg),
        # The cells the soil ingest made sea ice (XICE or seaice_threshold).
        sea_ice_cells=getattr(soil, "sea_ice_cells", None),
        **usemonalb_landuse_inputs(cfg, static, start_time))
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        hydro_parent_geometry=grid,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        sst=soil_fields.get("SST", soil.tsk),
        glw=constant_glw_wm2,
        radiation=radiation,
        radiation_start_time=start_time, radiation_latitude=lat,
        radiation_longitude=lon,
        terrain_drag_static=static,
        **lake_physics_inputs(cfg, static),
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=landuse_attrs, xice=soil.xice,
            fractional_seaice=ruc_fractional_seaice(cfg)),
        **({"fire_grid_spec": grid._rust_spec()} if int(getattr(cfg, "ifire", 0)) == 2 else {}),
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))
    import cupy as cp
    from woof.core.landuse import surface_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)

    # Seed time-zero surface diagnostics from the source analysis.  The
    # first model step replaces them through SFCLAY/Noah/YSU in WRF
    # ordering.
    from woof.ingest.real import surface_fields_to_device
    met0 = surface_fields_to_device(initial_met, cp)
    driver.fields["psfc"][...] = cp.asarray(
        initial_result.surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(
        initial_result.surface_qv, dtype=cp.float32)
    driver.fields["th2"][...] = (driver.fields["t2"]
                                  * (cp.float32(100000.0)
                                     / driver.fields["psfc"])
                                  ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (met0["U10"][:, :-1]
                                        + met0["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (met0["V10"][:-1]
                                        + met0["V10"][1:])

    if getattr(cfg, "sf_surface_mosaic", 0) == 1:
        from woof.core.noah_mosaic_door import attach_noah_mosaic_to_driver
        # initialize_landuse above runs at its own default,
        # fractional_seaice=False; the tiles take the same sea-ice split.
        attach_noah_mosaic_to_driver(
            driver, cfg, landusef=static.get("LANDUSEF"), processed=False,
            landuse_attrs=landuse_attrs, landmask=static["LANDMASK"],
            fractional_seaice=False)


def case_static_fields(grid, geog_root, *, selection: GeogSelection,
                       static_highres=None, domain_id: int = 1,
                       case_date=None) -> dict:
    source = _runtime_preparation_source()
    if source is None:
        return _case_static_fields(grid, geog_root, selection=selection,
            static_highres=static_highres, domain_id=domain_id, case_date=case_date)
    return source.static_fields(grid, geog_root, selection=selection,
        static_highres=static_highres, domain_id=domain_id, case_date=case_date,
        build=lambda: _case_static_fields(grid, geog_root, selection=selection,
            static_highres=static_highres, domain_id=domain_id, case_date=case_date))


def _case_static_fields(grid, geog_root, *, selection: GeogSelection,
                       static_highres=None, domain_id: int = 1,
                       case_date=None) -> dict:
    """One domain's static fields as this route will integrate them.

    The memoized WPS_GEOG build plus the optional ``[static.highres]``
    overlay, in that order.  ONE function because two callers need the
    same answer and must not derive it twice: the preparation below, and
    the vertical-coordinate survey that has to know the run's highest
    ground before any coordinate is built
    (:mod:`woof.vertical_adaptation`).  The build is cached by geometry,
    so asking twice costs nothing the second time.
    """

    static = _cached_static_build(grid, geog_root, selection=selection)
    if static_highres is None or not getattr(static_highres, "enabled",
                                             False):
        return static
    from woof.static.highres_production import apply_highres_statics

    static, _ = apply_highres_statics(
        static, grid, config=static_highres, domain_id=domain_id,
        case_date=case_date,
        landuse_attrs=selection.landuse_global_attrs())
    return static


def prepare_real_case(cfg: RunConfig, *, grid, geog_root,
                      source_orography_path=None,
                      source_orography_variable=None,
                      vertical: VerticalConfig, sfcp_to_sfcp: bool,
                      snapshot_for, forcing_times, start_time: datetime,
                      trace_gas_overrides=None,
                      geog_selection: GeogSelection | None = None,
                      forcing_catalog=None,
                      scratch_arena=None,
                      dycore_state_workspace=None,
                      radiation_column_chunk=DEFAULT_COLUMN_CHUNK,
                      static_highres=None,
                      static_domain_id: int = 1,
                      initial_perturbation=None,
                      constant_glw_wm2: float | None = None,
                      water_temperature_policy=None,
                      soil_texture_downscale: bool = True,
                      store_request=None, cam_ozone=None,
                      smooth_cg_topo: bool = False, blend_width: int = 5,
                      preprocess_backend: str = "auto",
                      ) -> PreparedRealCase:
    """Run the real-case setup pipeline for one domain.

    Extraction of the frozen reference preparation: identical operations in
    identical order, parameterized by the formerly implicit values.
    ``snapshot_for(valid_time)`` supplies the decoded forcing snapshot
    for each entry of ``forcing_times`` (the first entry must be
    ``start_time`` -- it becomes the live initial state).
    ``trace_gas_overrides`` feeds the RRTMGP trace-gas policy hook;
    ``None`` keeps the scheme's frozen default composition.
    ``initial_perturbation`` is the validated experiment
    :class:`woof.experiment.PerturbationConfig` (or ``None``, the OFF
    contract): its bubbles are applied inside ``initialize_real`` for
    the START TIME ONLY -- later forcing times contribute unperturbed
    boundary frames, except the t=0 frame, which reads the perturbed
    state (a bubble inside the relax zone would enter it; interior
    bubbles leave every boundary strip byte-identical).  This is the
    coarse/single domain, so a bubble center outside the grid refuses.
    ``preprocess_backend`` is where the transforms run when no
    ``store_request`` names it: ``auto`` (the default) decides from the
    card, ``cuda`` and ``cpu`` pin it.
    """
    from woof.ingest.soil import (door_reconciled_soil_category,
                                   soil_source_orography)

    times = tuple(forcing_times)
    if not times or times[0] != start_time:
        raise ValueError(
            f"forcing_times must begin at start_time {start_time}; got "
            f"{times[:1]}.")
    if (cfg.nx, cfg.ny) != (grid.e_we - 1, grid.e_sn - 1) or (
            cfg.dx, cfg.dy) != (grid.dx, grid.dy):
        raise ValueError(
            "RunConfig grid does not match the supplied Lambert grid: "
            f"got {(cfg.nx, cfg.ny, cfg.dx, cfg.dy)}, expected "
            f"{(grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)}")
    from woof.core.radiation_composition import trace_gas_override_status
    trace_gas_override_status(cfg, trace_gas_overrides)

    if (source_orography_path is None) != (source_orography_variable is None):
        raise ValueError(
            "source_orography_path and source_orography_variable must be "
            "provided together")
    if (source_orography_path is not None and forcing_catalog is not None
            and "SOILGEO" in getattr(forcing_catalog, "inventory", ())):
        raise ValueError(
            "source-orography conflict: declared source_orography "
            f"{source_orography_path} variable={source_orography_variable} "
            "and forcing catalog SOILGEO via era5_z_invariant are both "
            "present; declare exactly one source")
    geog_selection = (GeogSelection.fallback(geog_root)
                      if geog_selection is None else geog_selection)
    geog_selection = with_terrain_drag_statics(geog_selection, cfg)
    perturbation_applier = None
    if initial_perturbation is not None:
        # Built once, against this (coarse/single) domain's grid, so an
        # out-of-domain bubble center refuses HERE -- before any decode,
        # static build, or device work.
        from woof.ingest.init_perturbation import (
            build_initial_state_perturbation)
        perturbation_applier = build_initial_state_perturbation(
            initial_perturbation, grid, grid_id=int(cfg.grid_id),
            require_containment=True, cfg=cfg)
    static = case_static_fields(
        grid, geog_root, selection=geog_selection,
        static_highres=static_highres, domain_id=static_domain_id,
        case_date=start_time.date())
    landuse_attrs = geog_selection.landuse_global_attrs()
    source_orography = None
    if source_orography_path is not None:
        source_orography = load_source_orography(
            source_orography_path, source_orography_variable)
    # The surface the water-temperature assembly decides on: this domain's
    # own LANDMASK and the land-use table's own ISLAKE, so a lake and the
    # ocean stay in separate provider decisions even where a coarse
    # coastline connects them on the target.  Built once for the whole
    # forcing loop, because it is invariant across forcing times.
    water_statics = WaterTemperatureStatics.for_route(
        route=_WATER_ROUTE, policy=water_temperature_policy,
        landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
        landuse_attrs=landuse_attrs)
    # Only the first time's analysis/state and the last time's analysis
    # outlive this loop; every intermediate time contributes its perimeter
    # frames and is released before the next one is built.  Retaining all
    # N of them made setup, not the forecast, the memory-binding phase.
    initial_result = None
    initial_met = None
    # The snapshot the INITIAL state came off, kept beside the interpolated
    # fields because the soil seam needs the SOURCE mesh those fields were
    # carried from and the interpolated snapshot no longer carries it.
    initial_source = None
    forcing = StateBoundaryFrames(
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    # START LAST, on both roads.  The start time is built after every other
    # time and is the only state kept; each other time contributes its
    # perimeter frames against its own position and is released before
    # the next is built (lateral_bc.start_last_forcing_order, a pure
    # reordering).  Built start first, this route held the start time's
    # whole state while every later one was built beside it: two full
    # forcing states on the card, which the preparation price and the
    # sizing both priced at one (A65, F03).  The last time's analysis is
    # still kept for ``final_analysis``.
    from woof.ingest.lateral_bc import start_last_forcing_order
    from woof.ingest.preparation_price import (
        SourceInventory, price_preparation)
    from woof.ingest.preprocess_backend import resolve_preprocess_backend
    order = tuple(start_last_forcing_order(len(times)))
    prefetched = {order[0]: snapshot_for(times[order[0]])}
    # THE FIT, BEFORE THE FIRST DEVICE ALLOCATION.  The first snapshot to
    # be interpolated is decoded on the host; its inventory prices the
    # transforms (and, off the host store, the state) this route puts on
    # the card.  auto (the default) runs the transforms on the CPU when
    # that does not fit the card's free memory, with one named line.
    first_source = prefetched[order[0]]
    preparation_price = (None if not hasattr(first_source, "fields") else
                         price_preparation(
        "experiment" if store_request is None else "experiment-host-store",
        [cfg], SourceInventory.from_snapshot(first_source),
        boundary_intervals=len(times) - 1))
    requested_backend = (preprocess_backend if store_request is None
                         else store_request.backend)
    release_backend = resolve_preprocess_backend(
        requested_backend, price=preparation_price)
    if store_request is not None:
        # The store's identity and receipts record the backend that RAN.
        from dataclasses import replace as _replace
        store_request = _replace(store_request, backend=release_backend.name)
        print(f"  initialization: {release_backend.name} transforms, "
              "host state, row-slab GPU physics")
    final_met = None
    met = result = None
    from types import SimpleNamespace

    from woof.ingest.cg_topo import RootTerrainBlend
    terrain_blend = RootTerrainBlend(
        SimpleNamespace(smooth_cg_topo=bool(smooth_cg_topo)
                        and int(static_domain_id) == 1,
                        spec_bdy_width=int(cfg.spec_bdy_width),
                        blend_width=int(blend_width)),
        static, route="case_data")
    for position, index in enumerate(order):
        valid_time = times[index]
        if position:
            del met, result
            release_backend_memory(release_backend)
        source = prefetched.pop(index, None)
        if source is None:
            source = snapshot_for(valid_time)
        # Metgrid classifies masked-field TARGET cells by the model
        # (geogrid) landmask, not by the nearest source LSM; the source-side
        # usable-point decision stays with the source LANDSEA inside the
        # masked operators.  Passing the static LANDMASK reproduces WPS and
        # keeps soil, skin, and physics on one land/water surface.
        met = interpolate_era5_to_lambert(
            source, grid, source_orography_catalog=forcing_catalog,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5,
            water_temperature_statics=water_statics,
            backend=release_backend)
        # WRF's smooth_cg_topo (woof.ingest.cg_topo), once, before the
        # first initialization reads the root terrain.
        terrain_blend.before_initialize(
            source_orography if source_orography is not None
            else met.fields.get("SOURCE_OROGRAPHY"))
        if store_request is not None:
            from woof.ingest.case_store import admit_case_initialization
            admit_case_initialization(store_request, cfg, met, times)
        coord = vertical_coord_for(vertical, cfg.nz)
        init_kwargs = dict(
            source_orography=source_orography, p_top=vertical.p_top,
            sfcp_to_sfcp=sfcp_to_sfcp, preprocess_backend=release_backend)
        if store_request is not None:
            init_kwargs.update(state_backend="cpu")
        if scratch_arena is not None:
            init_kwargs["scratch_arena"] = scratch_arena
        if dycore_state_workspace is not None:
            init_kwargs["dycore_state_workspace"] = dycore_state_workspace
        if index == 0 and perturbation_applier is not None:
            # The bubbles perturb the LIVE INITIAL STATE only; the later
            # forcing times of this loop are boundary material and stay
            # the unperturbed analysis.
            init_kwargs["initial_perturbation"] = perturbation_applier
        result = initialize_real(
            met, cfg, coord, static["HGT_M"], grid=grid,
            landmask=static["LANDMASK"],
            boundary_only=index != 0, **init_kwargs)
        f, e = grid.coriolis_m()
        # SINALPHA/COSALPHA (geo_em conventions): WRF's coriolis applies
        # the rotation terms unconditionally (module_em.F:761-769).
        sina, cosa = grid.rotation_m()
        result.state.set_map_coriolis(
            grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(), f, e,
            sina=sina, cosa=cosa)
        forcing.add_state(result.state, index=index)
        if index == len(times) - 1 and store_request is None:
            final_met = met
        if index == 0:
            initial_met = met
            initial_result = result
            initial_source = source
    # The start time's analysis and state, and the last time's analysis,
    # are held above.  Nothing between them is still resident.
    boundaries = forcing.build(times)
    attach_lateral_boundaries(initial_result.state, boundaries)
    # The chem processes' preparation-time arrays (dust statics, sulfur
    # lat/lon) onto this grid, so the prepared cache and every checkpoint
    # carry them; nothing on a chem-off state (woof/core/chem_statics.py).
    from woof.core.chem_statics import attach_chem_statics
    attach_chem_statics(initial_result.state, grid, geog_root, geog_selection)

    soil_fields = dict(initial_met.fields)
    # No lake skin override: with metgrid's masked=both SKINTEMP chain and
    # static-landmask target classification, lake cells already carry the
    # water-source skin value, and real.exe (no TAVGSFC) keeps exactly that
    # SKINTEMP wherever SST has no valid support
    # (module_initialize_real.F:2844-2866, :2898-2906).  Inland water with no
    # source water within INLAND_WATER_SOURCE_REACH_M takes this forcing
    # time's 2 m air temperature instead (woof.ingest.horiz).  The router forwards
    # this exact argument list to preprocess_noah_soil for Noah-geometry
    # schemes, so their soil state is unchanged by the LSM dispatch seam.
    # ONE RULEBOOK (ArWen's ruling, 2026-08-06).  The soil column and the
    # liquid water derived from it must be built with the SAME category the
    # physics driver integrates, so ask for the reconciled ISLTYP here
    # rather than reading the raw geogrid SCT_DOM.  WRF gets this ordering
    # for free: real.exe reconciles at module_initialize_real.F:3108-3131
    # and LSMINIT (phys/module_sf_noahdrv.F) derives SH2O afterwards.  Ours
    # ran the other way round, because initialize_landuse below needs this
    # call's own outputs (snow, xice, TSLB) and therefore cannot precede it.
    # Evidence spellings come from the one per-source table in
    # woof/ingest/soil.py, so this root call and the nested-child call in
    # woof/ingest/nest_init.py cannot drift apart again: an inline chain
    # here knew only the mapped and classic per-layer names, which is the
    # same gap that aborted the native-HRRR and nested-GFS lanes.
    soil_orography = soil_source_orography(source_orography, soil_fields)
    reconciled_soil_type = door_reconciled_soil_category(
        static, soil_fields, landuse_attrs)
    from woof.core.landuse import (
        ruc_fractional_seaice as _ruc_fractional_seaice,
        seaice_threshold_of as _seaice_threshold_of)
    soil = preprocess_land_surface_soil(
        soil_fields, sf_surface_physics=int(cfg.sf_surface_physics),
        num_soil_layers=soil_layer_count(cfg),
        # real.exe's adjust_for_seaice_pre/post keep the fraction under
        # fractional_seaice = 1 (threshold 0.02) and snap to 0/1 at 0.5
        # otherwise (module_soil_pre.F:216-219, :337-343, :392-393 of the HRRR
        # v4.1.21 fork).
        fractional_seaice=_ruc_fractional_seaice(cfg),
        seaice_threshold=_seaice_threshold_of(cfg),
        soil_type=reconciled_soil_type,
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"],
        # Land the source holds no land for takes the column the
        # router builds (woof/ingest/soil.py: island_soil_columns).
        soil_no_source_land=getattr(
            initial_met, "soil_no_source_land", None),
        # The declared artifact OR the orography the forcing carries inside
        # itself, which the horizontal stage already remapped onto this grid
        # as SOURCE_OROGRAPHY.  Resolving only the declaration silently
        # dropped WRF's adjust_soil_temp_new lapse on every case whose
        # orography rides in its GRIB -- the ordinary ERA5 route, since
        # `woof fetch --source era5` writes the invariant geopotential into
        # the combined file.  Still all-or-none: a source that declares no
        # orography at all keeps the historical no-adjustment path.
        terrain=static["HGT_M"] if soil_orography is not None else None,
        source_orography=soil_orography,
        # The finished water temperature the ingest assembled: one provider
        # per connected body of water, never a per-cell choice between two
        # differently-mapped fields.  The policy and the route name travel
        # with it, because the router refuses a raw SST/SKINTEMP pair that
        # arrives with no decision attached.
        water_temperature=getattr(
            initial_met, "water_temperature", None),
        water_temperature_policy=water_temperature_policy,
        # The moisture and deep-temperature analogue of the elevation lapse
        # two arguments up.  ``initial_source`` is the snapshot the initial
        # state came off, so the mesh described here is the mesh the soil
        # arrived on.
        soil_mesh=soil_mesh_plan_from_case(
            initial_source, grid, enabled=bool(soil_texture_downscale)),
        # Where and when, for a same-model soil donor (woof/ingest/
        # land_state.py: HRRR's RUC soil for a RUC run over CONUS).
        land_state=land_state_context(
            grid, getattr(initial_met, "valid_time", times[0])),
        route=_WATER_ROUTE)
    if store_request is not None:
        from types import SimpleNamespace
        from woof.ingest.case_store import write_case_store_input

        inputs = write_case_store_input(
            store_request, cfg=cfg, vertical=vertical, times=times,
            initial_result=initial_result, met=initial_met, soil=soil,
            soil_fields=soil_fields, reconciled_soil_type=reconciled_soil_type,
            boundaries=boundaries, landuse_attrs=landuse_attrs,
            trace_gas_overrides=trace_gas_overrides,
            radiation_column_chunk=radiation_column_chunk,
            constant_glw_wm2=constant_glw_wm2,
            **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))
        # No state or met array escapes this frame. The caller drops decoded
        # forcing before allocating the pinned store; only setup metadata and
        # the two small resolved surface planes survive beside the cache.
        #
        # THE FLOOR NAMES ARE DERIVED, not typed.  Three receipts were
        # spelled here by hand, and a hand-spelled list of receipt fields
        # is the same construction that dropped a child's aerosol receipt
        # in `nest_init`: the moisture floors this initialization applied
        # were computed, recorded on the result, and then left behind by
        # this frame -- so a store-backed run reported "not recorded" for
        # a floor that had actually fired.  Every `*_moisture_floor` field
        # comes across, so the next floor the ingest grows survives this
        # boundary with no edit.
        metadata = SimpleNamespace(**{
            name: getattr(initial_result, name, None) for name in (
                "initial_perturbation", "hydrometeor_initialization",
                "aerosol_initialization",
                *moisture_floor_field_names(initial_result))})
        return PreparedRealCase(
        static_sampling_contract=current_sampling_contract(),
            cfg=cfg, grid=grid, static_fields=static, initial_result=metadata,
            final_analysis=None, initial_snow_water_kgm2=np.array(
                soil.snow_water, dtype=np.float64, copy=True),
            forcing_times=times, geog_selection=geog_selection,
            store_input=inputs)
    source = _runtime_preparation_source()
    if source is not None:
        source.capture_root_inputs(cfg=cfg, vertical=vertical, times=times,
            initial_result=initial_result, met=initial_met, soil=soil,
            soil_fields=soil_fields, reconciled_soil_type=reconciled_soil_type,
            boundaries=boundaries, landuse_attrs=landuse_attrs,
            trace_gas_overrides=trace_gas_overrides,
            radiation_column_chunk=radiation_column_chunk,
            constant_glw_wm2=constant_glw_wm2, cam_ozone=cam_ozone,
            preprocess_backend=release_backend.name)
    _initialize_real_case_physics(
        initial_result, cfg, initial_met, soil, soil_fields, static,
        landuse_attrs, grid, start_time, vertical=vertical,
        reconciled_soil_type=reconciled_soil_type,
        trace_gas_overrides=trace_gas_overrides,
        radiation_column_chunk=radiation_column_chunk,
        constant_glw_wm2=constant_glw_wm2,
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))
    return PreparedRealCase(
        static_sampling_contract=current_sampling_contract(),
        cfg=cfg, grid=grid, static_fields=static,
        initial_result=initial_result, final_analysis=final_met,
        initial_snow_water_kgm2=np.array(
            soil.snow_water, dtype=np.float64, copy=True),
        forcing_times=times, geog_selection=geog_selection)


def prepare_root_experiment_case(exp: ExperimentConfig,
                                 data: CaseDataConfig, *,
                                 input_catalog=None,
                                 forcing_by_time=None,
                                 scratch_arena=None,
                                 dycore_state_workspace=None,
                                 store_request=None
                                 ) -> PreparedRealCase:
    source = _runtime_preparation_source()
    if source is None:
        prepared = _prepare_root_experiment_case(exp, data, input_catalog=input_catalog,
            forcing_by_time=forcing_by_time, scratch_arena=scratch_arena,
            dycore_state_workspace=dycore_state_workspace, store_request=store_request)
        return _hold_strict_theta_carrier(prepared)
    catalog = _runtime_input_catalog(data) if input_catalog is None else input_catalog
    grid = (experiment_grid(exp, data) if len(exp.domains) == 1 else grids_from_projection_config(exp)[0])
    selection = GeogSelection.from_case_data(data, domain_id=exp.root.grid_id)
    prepared = source.prepare_root(exp, data, grid=grid, selection=selection, catalog=catalog,
        scratch_arena=scratch_arena, dycore_state_workspace=dycore_state_workspace,
        store_request=store_request,
        build=lambda: _prepare_root_experiment_case(exp, data, input_catalog=catalog,
            forcing_by_time=forcing_by_time, scratch_arena=scratch_arena,
            dycore_state_workspace=dycore_state_workspace, store_request=store_request))
    return _hold_strict_theta_carrier(prepared)


def _hold_strict_theta_carrier(prepared: PreparedRealCase) -> PreparedRealCase:
    """Strict mode: the in-process prepared root holds theta as WRF's ``t = theta - 300``.

    THE BREAKAGE THIS PREVENTS (2.9.2): every strict kernel reads ``thp`` as
    WRF's ``t_2`` with ``thb == 300`` (woof.core.state.adopt_wrf_theta_carrier).
    2.9.1 re-expressed the prepared state that way only where a prepared cache
    is restored (woof.ingest.prepared_cache.restore_prepared_cache), which the
    chained GFS-series preparation passes through.  The route that builds its
    forcing in the forecast's own process (the experiment route ``woof go``
    takes on the keyless ERA5 default, SEALED_REASONS["experiment_run"]) runs
    ``prepare_real_case``'s state as built, with the preparation's 290-464 K
    base in ``thb``, so strict mode read theta up to 164 K low and stopped at
    its first step: "microphysics RAINNC contains a non-finite value" on the
    default ``woof go`` config (500x400 at 12 km, 2026-10-11T12 ERA5).

    This is the one door every in-process root passes (the single-domain
    arm, the tree root in woof.core.model, ensemble members), reached after
    every store or capture of the prepared words was written, so the caches
    keep the preparation's base.  A host-store preparation returns no state
    (its strict refusal stands: prepared_store.refuse_strict_streamed_base);
    a root restored from a cache is already re-expressed, and the adoption is
    a no-op on it.  Default mode is untouched: adopt_wrf_theta_carrier returns
    at once outside GPUWM_WRF_EXACT=1.
    """
    state = getattr(getattr(prepared, "initial_result", None), "state", None)
    if state is not None:
        from woof.core.state import adopt_wrf_theta_carrier
        adopt_wrf_theta_carrier(state)
    return prepared


def _prepare_root_experiment_case(exp: ExperimentConfig,
                                 data: CaseDataConfig, *,
                                 input_catalog=None,
                                 forcing_by_time=None,
                                 scratch_arena=None,
                                 dycore_state_workspace=None,
                                 store_request=None
                                 ) -> PreparedRealCase:
    """Prepare the root domain of a single- or multi-domain experiment."""
    dc = exp.root
    cfg = dc.run
    if len(exp.domains) == 1:
        grid = experiment_grid(exp, data)
    else:
        from woof.static.lambert import grids_from_projection_config
        grid = grids_from_projection_config(exp)[0]
    geog_selection = GeogSelection.from_case_data(
        data, domain_id=dc.grid_id)
    catalog = (_runtime_input_catalog(data) if input_catalog is None
               else input_catalog)
    snapshots = (forcing_snapshots(data, catalog)
                 if forcing_by_time is None else forcing_by_time)
    decoded_times = tuple(snapshots)
    if decoded_times != tuple(catalog.valid_times):
        raise ValueError(
            "prepared forcing snapshots do not match the input catalog's "
            f"ordered valid times: expected {catalog.valid_times}, "
            f"got {decoded_times}; catalog exclusions: "
            f"{catalog.excluded_valid_times}")
    times = forcing_schedule(exp, data, snapshots)
    # Every route that prepares a real root reaches this line exactly
    # once, and it is the last point before the expensive work at which
    # both the decode and the run length are in one scope.  The count is
    # the CATALOG's, taken from the forcing files; the run length is the
    # experiment's, and nothing had ever compared them out loud.
    print(forcing_decode_report(exp, snapshots))
    # And that every one of them is built before the forecast starts:
    # the loop runs in the model's own process, on its own workspace
    # (scratch_arena, dycore_state_workspace), so it cannot run beside it.
    from woof.ingest.boundary_stream import say_prepared_sealed
    say_prepared_sealed("experiment_run")

    def snapshot_for(valid_time):
        try:
            return snapshots[valid_time]
        except KeyError as exc:
            raise ValueError(
                f"forcing has no snapshot at {valid_time!s}") from exc

    declared_orography = data.source_orography
    from woof.core.cam_ozone import cam_ozone_setup
    cam = cam_ozone_setup(exp=exp, dc=dc, grid=grid)
    # A declared backend (``[case_data] preprocess_backend`` or ``woof run
    # --preprocess-backend``) pins this preparation, host store included;
    # silence keeps ``auto``, which reads the card and can land on either.
    pinned = getattr(data, "preprocess_backend", None)
    if pinned is not None:
        print(f"  preparation backend: {pinned}, as declared "
              "([case_data] preprocess_backend or --preprocess-backend)")
    if pinned is not None and store_request is not None:
        from dataclasses import replace as _replace
        store_request = _replace(store_request, backend=pinned)
    return prepare_real_case(
        cfg, grid=grid, geog_root=data.geog_root,
        source_orography_path=(declared_orography.path
                               if declared_orography is not None else None),
        source_orography_variable=(declared_orography.variable
                                   if declared_orography is not None else None),
        vertical=exp.vertical, sfcp_to_sfcp=data.sfcp_to_sfcp,
        snapshot_for=snapshot_for, forcing_times=times,
        start_time=exp.start_time,
        trace_gas_overrides=({"co2": data.co2_vmr}
                             if data.co2_vmr is not None else None),
        geog_selection=geog_selection, forcing_catalog=catalog,
        water_temperature_policy=resolve_water_temperature_policy(data),
        soil_texture_downscale=declared_soil_texture_downscale(data),
        scratch_arena=scratch_arena,
        dycore_state_workspace=dycore_state_workspace,
        radiation_column_chunk=exp.column_chunk,
        static_highres=getattr(data, "static_highres", None),
        static_domain_id=dc.grid_id,
        initial_perturbation=exp.perturbation,
        **({"store_request": store_request} if store_request is not None else {}),
        **({"preprocess_backend": pinned} if pinned is not None else {}),
        constant_glw_wm2=declared_constant_glw(exp),
        smooth_cg_topo=bool(exp.smooth_cg_topo and dc.parent_id == 0),
        blend_width=int(exp.blend_width),
        **({"cam_ozone": cam} if cam is not None else {}))


def prepare_experiment_case(exp: ExperimentConfig,
                            data: CaseDataConfig, *, input_catalog=None,
                            forcing_by_time=None,
                            store_request=None) -> PreparedRealCase:
    """Assemble :func:`prepare_real_case` inputs from the config pair."""
    single_domain(exp)  # Preserve Task-2's fail-loud single-domain surface.
    return prepare_root_experiment_case(
        exp, data, input_catalog=input_catalog,
        forcing_by_time=forcing_by_time,
        **({"store_request": store_request} if store_request is not None else {}))


def _child_radiation_adapter(exp: ExperimentConfig, data: CaseDataConfig,
                             dc, state, lat, lon, *,
                             radiation_workspace=None,
                             radiation_parent=None):
    """Construct one child domain's radiation adapter (shared by the
    t=0 preparer and the relocation preparer, so a relocated child's
    radiation is wired by exactly the code that wired it at start)."""
    from woof.core.radiation_composition import make_radiation, attach_modern_workspace
    from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    cfg = dc.run
    ozone_parent = None
    if (4 in radiation_scheme_ids(cfg)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY
            and cfg.o3input == 2):
        if radiation_parent is None:
            from woof.core.cam_ozone import cam_ozone_domain_ids, DriverOzoneProvider
            if exp is None or dc.grid_id not in cam_ozone_domain_ids(exp):
                raise ValueError(
                    "ra_rrtmg_variant='rrtmg_legacy' on child domain "
                    f"grid_id={dc.grid_id} requires radiation_parent= "
                    "or the experiment's shared CAM ozone carrier")
            ozone_parent = DriverOzoneProvider()
        if radiation_parent is not None:
            from woof.core.nest_interp import register_nest
            from woof.core.rrtmg_legacy import ParentOzoneProvider
            parent_dc = exp.domain(dc.parent_id)
            registration = register_nest(
                nri=dc.parent_grid_ratio, nrj=dc.parent_grid_ratio,
                i_parent_start=dc.i_parent_start,
                j_parent_start=dc.j_parent_start,
                child_nx=cfg.nx, child_ny=cfg.ny,
                parent_nx=parent_dc.run.nx, parent_ny=parent_dc.run.ny,
                stagger="", wrapper="interp")
            ozone_parent = ParentOzoneProvider(radiation_parent, registration)
    radiation = make_radiation(
        cfg, exp.start_time, lat, lon, p_top=float(state.p_top),
        trace_gas_overrides=({"co2": data.co2_vmr}
                             if data.co2_vmr is not None else None),
        column_chunk=exp.column_chunk, ozone_parent=ozone_parent)
    attach_modern_workspace(radiation, radiation_workspace)
    return radiation


def prepare_child_case(initialized, child_dc, *, exp: ExperimentConfig,
                       data: CaseDataConfig, forcing_times,
                       radiation_workspace=None,
                       radiation_parent=None) -> PreparedRealCase:
    """Attach one child's per-domain physics driver after T12 init.

    T12 returns the WRF-order atmospheric/static/Noah setup products.  This
    helper performs the same surface/radiation initialization used by the
    root, including the unblended-terrain Noah state, date-interpolated GEOG
    climatologies, and time-zero diagnostics.  It intentionally constructs a
    distinct RRTMGP adapter for the child; Task 14 attaches the one allocated
    common chunk workspace after all drivers exist.

    A nested legacy CAM consumer uses the experiment's shared retained
    ozone field. ``radiation_parent`` remains available for independent
    callers supplying the original parent adapter; either way, the child
    consumes parent-interpolated ozone rather than local climatology.
    """
    import cupy as cp

    from woof.core.diagnostics import update_diagnostics
    from woof.core.landuse import initialize_landuse
    from woof.core.physics import initialize_physics

    if (initialized.real is None or initialized.static_fields is None
            or initialized.horizontal is None or initialized.soil is None):
        raise ValueError("real experiment children require T12 real-data init")
    dc = exp.domain(child_dc.grid_id)
    if dc is not child_dc:
        raise ValueError("child_dc must be the experiment's DomainConfig")
    cfg = dc.run
    state = initialized.state
    static = initialized.static_fields
    soil = initialized.soil
    # A shared child input retains host words. Keep the fresh initializer's
    # original field precision while restoring those words on this card.
    met0 = surface_fields_to_device(initialized.horizontal, cp, preserve_dtype=True)
    real = initialized.real

    update_diagnostics(state, cfg.hypsometric_opt)
    domain_start_time = exp.domain_start_time(dc.grid_id)
    from woof.ingest.vegetation import initial_vegetation_fraction
    vegfra = initial_vegetation_fraction(
        initialized.horizontal, static, domain_start_time)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], domain_start_time)
    lat, lon = initialized.grid.latlon_mass()
    radiation = _child_radiation_adapter(
        exp, data, dc, state, lat, lon,
        radiation_workspace=radiation_workspace,
        radiation_parent=radiation_parent)
    geog_selection = GeogSelection.from_case_data(
        data, domain_id=dc.grid_id)
    if getattr(state, "chem", None) is not None:
        from woof.core.chem_statics import attach_chem_statics
        attach_chem_statics(state, initialized.grid, data.geog_root,
                            geog_selection)
    landuse_attrs = geog_selection.landuse_global_attrs()
    from woof.core.landuse import (ruc_fractional_seaice,
                                    usemonalb_landuse_inputs)
    landuse = initialize_landuse(
        static["LU_INDEX"], urban_legend=int(getattr(cfg, "sf_urban_physics", 0)) > 0,
        soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=domain_start_time,
        cen_lat=float(getattr(initialized.grid, "cen_lat", np.mean(lat))),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=soil.soil_temperature,
        fractional_seaice=ruc_fractional_seaice(cfg),
        # The cells the soil ingest made sea ice (XICE or seaice_threshold).
        sea_ice_cells=getattr(soil, "sea_ice_cells", None),
        **usemonalb_landuse_inputs(cfg, static, domain_start_time))
    from woof.core.cam_ozone import cam_ozone_setup
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    cam = cam_ozone_setup(exp=exp, dc=dc, grid=initialized.grid)
    from woof.ingest.lake_physics import lake_physics_inputs
    driver = initialize_physics(
        state, cfg, cam_ozone=cam, landuse=landuse, tsk=soil.tsk,
        hydro_parent_geometry=initialized.grid,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        glw=declared_constant_glw(exp),
        radiation=radiation, radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon,
        **lake_physics_inputs(cfg, static),
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=landuse_attrs, xice=soil.xice,
            fractional_seaice=ruc_fractional_seaice(cfg)),
        **({"fire_grid_spec": initialized.grid._rust_spec()}
           if int(getattr(cfg, "ifire", 0)) == 2 else {}))
    from woof.core.landuse import surface_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    driver.fields["psfc"][...] = cp.asarray(
        real.surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(real.surface_qv, dtype=cp.float32)
    driver.fields["th2"][...] = (
        driver.fields["t2"]
        * (cp.float32(100000.0) / driver.fields["psfc"])
        ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (met0["U10"][:, :-1]
                                         + met0["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (met0["V10"][:-1]
                                         + met0["V10"][1:])
    if getattr(cfg, "sf_surface_mosaic", 0) == 1:
        from woof.core.noah_mosaic_door import attach_noah_mosaic_to_driver
        # initialize_landuse above runs at its own default,
        # fractional_seaice=False; the tiles take the same sea-ice split.
        attach_noah_mosaic_to_driver(
            driver, cfg, landusef=static.get("LANDUSEF"), processed=False,
            landuse_attrs=landuse_attrs, landmask=static["LANDMASK"],
            fractional_seaice=False)
    from woof.core.cam_ozone import configure_cam_ozone
    configure_cam_ozone(state, cfg, exp=exp, dc=dc, grid=initialized.grid)
    return PreparedRealCase(
        static_sampling_contract=current_sampling_contract(),
        cfg=cfg, grid=initialized.grid, static_fields=dict(static),
        initial_result=real, final_analysis=initialized.horizontal,
        initial_snow_water_kgm2=np.array(
            soil.snow_water, dtype=np.float64, copy=True),
        forcing_times=tuple(forcing_times),
        geog_selection=geog_selection)


# ---------------------------------------------------------------------------
# Real-data relocation: the route-owned child preparer and runner wiring
# ---------------------------------------------------------------------------

def _relocation_host(value) -> np.ndarray:
    if hasattr(value, "__cuda_array_interface__"):
        return np.asarray(value.get())
    return np.array(value, copy=True)


def rebuild_child_driver_from_land_state(*, exp: ExperimentConfig,
                                         data: CaseDataConfig, model,
                                         initialized, child_dc, parent_node,
                                         land, landuse_attrs=None,
                                         radiation_factory=None, center_lat=None) -> float:
    """Rebuild one child's physics driver over a supplied land state.

    The operation both mid-run child events need, and the reason they can
    share it: a relocation and a spawn differ ONLY in where the land state
    comes from (an index-space transplant plus donor fill for the first,
    WRF's masked parent interpolator for the second).  Once the fields are
    in hand the rebuild is identical -- ``initialize_landuse`` /
    ``initialize_physics`` against the footprint's OWN statics, then the
    continuation fields the constructor does not take by direct overwrite
    -- and it is the same wiring the t = 0 child preparer
    (:func:`prepare_child_case`) performs.

    ``land`` maps :data:`~woof.ingest.relocation_init
    .LAND_SURFACE_CONTINUATION_FIELDS` names to arrays; a name the
    caller does not supply falls back to the cold-start default, exactly
    as it did when this lived inside the relocation preparer.
    Accumulators are NOT here: they are re-initialised at the new
    footprint, and both callers' receipts say so.

    ``landuse_attrs`` and ``radiation_factory`` are the two route seams:
    ``None`` (every case-data caller) derives both from ``data`` exactly
    as before -- the GEOG selection's land-use identity and the shared
    child radiation adapter.  The prepared tree route, which has no
    ``CaseDataConfig``, passes its own native land-use identity and a
    factory reproducing its t=0 radiation wiring
    (``radiation_factory(child_dc, state, lat, lon) -> callable|None``;
    ``None`` lets ``initialize_physics`` build the scheme from the
    RunConfig, byte-for-byte the prepared t=0 path).

    Returns the wall seconds the rebuild took.
    """
    import time as _time

    import cupy as cp

    from woof.core.landuse import initialize_landuse
    from woof.core.physics import initialize_physics

    started = _time.perf_counter()
    cfg = child_dc.run
    static = initialized.static_fields
    grid = initialized.grid
    state = initialized.state
    now = exp.start_time + timedelta(
        seconds=float(parent_node.clock.elapsed_seconds))
    # Climatology fields interpolate to the EVENT time: a child rebuilt
    # (or born) in May must not wear its January vegetation.
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], now)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], now)
    lat, lon = grid.latlon_mass()
    if getattr(state, "chem", None) is not None:
        from woof.core.chem_statics import attach_chem_statics
        chem_geog_root = None if data is None else data.geog_root
        chem_selection = (None if data is None else
                          GeogSelection.from_case_data(
                              data, domain_id=int(child_dc.grid_id)))
        attach_chem_statics(state, grid, chem_geog_root, chem_selection,
                            static_fields=static)
    if radiation_factory is None:
        parent_physics = getattr(parent_node.state, "physics", None)
        radiation = _child_radiation_adapter(
            exp, data, child_dc, state, lat, lon,
            radiation_workspace=(
                model._activation_context or {}).get("radiation_workspace"),
            radiation_parent=(None if parent_physics is None
                              else parent_physics.radiation_callable))
    else:
        radiation = radiation_factory(child_dc, state, lat, lon)
    if landuse_attrs is None:
        geog_selection = GeogSelection.from_case_data(
            data, domain_id=int(child_dc.grid_id))
        attrs = geog_selection.landuse_global_attrs()
    else:
        attrs = dict(landuse_attrs)
    from woof.core.landuse import (ruc_fractional_seaice,
                                    usemonalb_landuse_inputs)
    landuse = initialize_landuse(
        static["LU_INDEX"], urban_legend=int(getattr(cfg, "sf_urban_physics", 0)) > 0,
        soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"],
        snow=_relocation_host(land.get("snow", 0.0)),
        xice=_relocation_host(land.get("xice", 0.0)),
        valid_time=now,
        cen_lat=(float(getattr(grid, "cen_lat", np.mean(lat)))
                 if center_lat is None else float(center_lat)),
        mminlu=str(attrs["MMINLU"]),
        iswater=int(attrs["ISWATER"]),
        islake=int(attrs["ISLAKE"]),
        isice=int(attrs["ISICE"]),
        soil_temperature=(None if land.get("tslb") is None else
                          _relocation_host(land["tslb"])),
        fractional_seaice=ruc_fractional_seaice(cfg),
        **usemonalb_landuse_inputs(cfg, static, now))
    from woof.core.cam_ozone import cam_ozone_setup
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    cam = cam_ozone_setup(exp=exp, dc=child_dc, grid=grid)
    from woof.ingest.lake_physics import lake_physics_inputs
    driver = initialize_physics(
        state, cfg, landuse=landuse, hydro_parent_geometry=grid,
        tsk=land.get("tsk", 300.0),
        soil_temperature=land.get("tslb", 285.0),
        soil_moisture=land.get("smois", 0.30),
        liquid_moisture=land.get("sh2o"),
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=static["TMN"],
        xice=land.get("xice", 0.0), snow=land.get("snow", 0.0),
        snow_depth=land.get("snowh", 0.0),
        sst=land.get("tsk"),
        glw=declared_constant_glw(exp),
        cam_ozone=cam, radiation=radiation, radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon,
        **lake_physics_inputs(cfg, static),
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=attrs, xice=land.get("xice", 0.0),
            fractional_seaice=ruc_fractional_seaice(cfg)),
        **({"fire_grid_spec": grid._rust_spec()} if int(getattr(cfg, "ifire", 0)) == 2 else {}))
    from woof.core.cam_ozone import configure_cam_ozone
    configure_cam_ozone(state, cfg, exp=exp, dc=child_dc, grid=grid)
    from woof.core.landuse import surface_snow_albedo
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    # Continuation fields the constructor does not take arrive by
    # direct overwrite -- the supplied arrays, on the whole child.
    for name in ("canwat", "snowc", "snotime", "qsfc", "ust",
                 "smcrel", "psfc", "t2", "q2", "th2", "u10", "v10"):
        target = driver.fields.get(name)
        value = land.get(name)
        if target is not None and value is not None:
            target[...] = cp.asarray(value, dtype=target.dtype)
    # A rebuilt driver has no radiation call of its own yet: its shortwave
    # interpolation (swint_opt = 1) holds no fit and its held fluxes are
    # the cold driver's zeros, while the nest keeps its model time, so the
    # cadence predicate would wait a whole radt.  It calls radiation on its
    # first step instead, as a nest born at t = 0 does (itimestep == 1).
    driver.force_radiation_next_step = True
    return _time.perf_counter() - started


class RealRelocationChildPreparer:
    """Physics rebuild + land-surface continuation for a relocated child.

    This is the ``on_child_built`` the real-data route hands the
    :class:`woof.core.relocation_runner.RelocationRunner`, plus the two
    duck-typed seams the runner drives around it:

    * ``capture_outgoing(node)`` -- before the move, while the outgoing
      child is whole: snapshot the driver-held land-surface continuation
      state (:data:`~woof.ingest.relocation_init.
      LAND_SURFACE_CONTINUATION_FIELDS`), the per-column physics
      continuation and the surface-radiation carriers with their
      provenance ledger (:mod:`woof.core.physics_continuation`), and
      keep a reference to the outgoing footprint's statics.
    * ``__call__(initialized, new_dc, parent_node)`` -- the rebuild:
      FIRST assert the footprint-rebuilt statics equal the outgoing
      child's bitwise on shared ground (identical source + identical
      cells = identical bytes; any mismatch is a statics-build defect
      and refuses), then build the donor-fill plan, move the land state
      (overlap by index-space transplant, strip from nearest
      same-landmask-class donors), and rebuild the physics driver
      against the NEW statics through the same ``initialize_landuse`` /
      ``initialize_physics`` / radiation wiring the t=0 child preparer
      uses.  The rebuild re-initialises the accumulators; the physics
      continuation then carries them on the overlap, and fresh ground
      takes the parent's accumulated precipitation
      (:func:`woof.core.physics_continuation.parent_strip_accumulations`),
      and the receipt says so.
    * ``after_move(node)`` -- once the node carries the new placement:
      refresh the run's prepared-case bookkeeping and the domain's
      wrfout metadata/global attributes so later frames describe the
      footprint that produced them.
    """

    # CLASS-LEVEL, because the subclasses do not chain __init__.
    # PreparedTreeRelocationChildPreparer restates this whole body rather
    # than calling super(), so an instance attribute set only here is
    # absent on exactly the preparers the prepared-tree route actually
    # uses -- an AttributeError on the first relocation, which is how
    # this was caught.  A class attribute is inherited whatever a
    # subclass's __init__ does or does not do.
    _plan_override = None

    def __init__(self, *, exp: ExperimentConfig, data: CaseDataConfig,
                 model):
        self.exp = exp
        self.data = data
        self.model = model
        self.writers = None
        self.last_receipt = None
        self._captured = None
        self._pending_refresh = None
        self._plan_override = None

    def attach_writers(self, writers) -> None:
        self.writers = writers

    def capture_outgoing(self, node) -> None:
        from copy import deepcopy
        from tilestream.physics_inventory import carrier_scalars
        from woof.core.physics_continuation import (_capture, capture_carriers,
                                                     capture_continuation)
        from woof.ingest.relocation_init import (
            LAND_SURFACE_CONTINUATION_FIELDS)
        from woof.core import nest_accumulation

        stream = getattr(node.state, "_streamed_domain", None)
        store = None if stream is None else stream.store
        source_state = (node.state if stream is None or stream.template is None
                        else stream.template)
        driver = getattr(source_state, "physics", None)
        fields: dict[str, np.ndarray] = {}
        if driver is not None:
            for name in LAND_SURFACE_CONTINUATION_FIELDS:
                value = driver.fields.get(name)
                if value is not None:
                    if store is not None:
                        key = "fields/" + name
                        if key not in store:
                            raise ValueError(f"streamed land state is missing canonical carrier {key}")
                        value = store[key]
                    fields[name] = _capture(value)
        case = self.model._prepared_by_grid_id.get(int(node.cfg.grid_id))
        self._captured = {
            "grid_id": int(node.cfg.grid_id),
            "device": hasattr(getattr(source_state, "p", None),
                              "__cuda_array_interface__"),
            "i_parent_start": int(node.cfg.i_parent_start),
            "j_parent_start": int(node.cfg.j_parent_start),
            "fields": fields,
            "scalar_carriers": deepcopy(carrier_scalars(source_state)
                                        if stream is None else stream.scalars),
            # Driver-held per-column physics continuation (KF timers and
            # held rates, precipitation accumulators, W0AVG), captured by
            # the restart registry so it shifts with the move instead of
            # cold-restarting on the whole child (the 2026-08-16 moving-
            # nest KF artifact report).
            "continuation": capture_continuation(source_state, driver, store=store),
            "precipitation_archive": nest_accumulation.capture(source_state, store),
            # THE SURFACE-RADIATION CARRIERS and their ledger.  A carrier
            # is consumed on every surface step but produced only on the
            # radiation cadence, so a child rebuilt from cold between two
            # radiation calls holds allocation fill under a ledger that
            # says nothing ever wrote it -- and the carrier contract
            # refuses, correctly, at the first move (the 2026-08-24
            # test host B campaign's GLW refusal).  See
            # woof/core/physics_continuation.py.
            "carriers": capture_carriers(driver, store=store,
                                          scalars=None if stream is None else stream.scalars),
            "static_fields": (None if case is None
                              else case.static_fields),
        }

    def __call__(self, initialized, new_dc, parent_node) -> None:
        import time as _time

        from woof.ingest.relocation_init import LAND_SURFACE_CONTINUATION_FIELDS

        started = _time.perf_counter()
        from woof.ingest.relocation_continuation import stage_relocation_continuation
        captured = self._captured
        self._captured = None
        plan = self._plan_override
        self._plan_override = None
        static = initialized.static_fields
        staged = stage_relocation_continuation(captured, new_dc, static, plan=plan)
        plan, statics_verdict, fill, moved = (
            staged.plan, staged.statics_verdict, staged.fill, staged.land)

        driver_seconds = self._rebuild_driver(
            initialized, new_dc, parent_node, moved)
        # Physics continuation follows the move: the registry-derived
        # per-column driver state (KF NCA/held rates/PRATEC/RAINCV, the
        # RAINC/RAINNC accumulators, W0AVG) shifts in index space with
        # the same plan window as the serialised-state transplant, and
        # the freshly exposed strip cold-starts.  Before this landed the
        # whole child cold-started at every move, which is the reported
        # moving-nest KF artifact (2026-08-16).
        from woof.core.physics_continuation import (
            accumulation_ring, parent_strip_accumulations, restore_carriers,
            restore_continuation, shift_carriers, shift_continuation)

        continuation_capture = captured.get("continuation", {}) or {}
        ring = accumulation_ring(getattr(new_dc, "run", None))
        from woof.core import nest_accumulation
        archive, strip, strip_receipt = nest_accumulation.relocate(
            captured.get("precipitation_archive") or {}, continuation_capture,
            plan, ring, parent_node, new_dc)
        shifted = shift_continuation(continuation_capture, plan, strip=strip,
                                     ring=ring)
        new_state = getattr(initialized, "state", None)
        new_driver = getattr(new_state, "physics", None)
        if new_state is not None and new_driver is not None:
            continuation = restore_continuation(
                new_state, new_driver, shifted)
            nest_accumulation.install(new_state, archive)
            continuation["restored"] = True
        else:
            continuation = {
                "restored": False,
                "reason": "the rebuilt child carries no physics driver, "
                          "so its continuation state cold-starts",
            }
        # THE SURFACE-RADIATION CARRIERS.  Same plan window as the
        # serialised-state transplant, same donor fill as the
        # land-surface continuation fields above -- a carrier and the
        # skin temperature it drives reach a fresh strip cell from the
        # SAME donor column -- and the ledger travels verbatim so the
        # staleness half of the carrier guard stays armed across the
        # move.  Without this the first surface call after a move met a
        # ledger with no producer for GLW and refused (2026-08-24).
        carrier_capture = captured.get("carriers") or {}
        if new_state is not None and new_driver is not None:
            carriers = restore_carriers(
                new_driver,
                shift_carriers(carrier_capture.get("fields") or {}, plan,
                               fill),
                carrier_capture.get("contract"))
        else:
            carriers = {
                "restored": False,
                "reason": "the rebuilt child carries no physics driver, "
                          "so it holds no radiative carriers to keep",
            }
        if new_state is not None and captured.get("scalar_carriers") is not None:
            from tilestream.physics_inventory import set_carrier_scalars
            set_carrier_scalars(new_state, captured["scalar_carriers"])
        self._pending_refresh = (int(new_dc.grid_id), initialized.grid,
                                 static)
        self.last_receipt = {
            "overlap_statics": {
                "compared_cells": statics_verdict["compared_cells"],
                "mismatched_fields": statics_verdict["mismatched_fields"],
                "within_one_ulp": statics_verdict.get(
                    "within_one_ulp", {}),
                "pass": statics_verdict["pass"],
            },
            "donor_fill": dict(fill.counts),
            "fields_moved": sorted(moved),
            "fields_absent": sorted(
                set(LAND_SURFACE_CONTINUATION_FIELDS)
                - set(captured["fields"])),
            "accumulators_reinitialized": not continuation["restored"],
            "physics_continuation": continuation,
            "strip_accumulations": strip_receipt,
            "radiation_carriers": carriers,
            "driver_rebuild_seconds": driver_seconds,
            "preparer_seconds": _time.perf_counter() - started,
        }


    def prepare_windows(self, new_dc, parent_node, footprint):
        """Stage one global move, then prepare only requested device windows.

        Full-domain statics and donor choices are made before slicing. The
        returned callback is consumed by store reconstruction; it does not
        publish a node or re-enable streamed relocation admission.
        """
        from woof.core.nest_relocation import RelocationRefusal
        from woof.core.physics_continuation import (
            accumulation_ring, parent_strip_accumulations, restore_carriers,
            restore_continuation, shift_carriers, shift_continuation)
        from woof.ingest.relocation_continuation import stage_relocation_continuation
        from tilestream.physics_inventory import set_carrier_scalars

        captured, self._captured = self._captured, None
        override, self._plan_override = self._plan_override, None
        staged = stage_relocation_continuation(
            captured, new_dc, footprint.static_fields, plan=override)
        scalars = captured.get("scalar_carriers")
        if scalars is None:
            raise RelocationRefusal("window reconstruction requires canonical streamed scalar carriers")
        continuation_capture = captured.get("continuation") or {}
        ring = accumulation_ring(getattr(new_dc, "run", None))
        from woof.core import nest_accumulation
        archive, strip, strip_receipt = nest_accumulation.relocate(
            captured.get("precipitation_archive") or {}, continuation_capture,
            staged.plan, ring, parent_node, new_dc)
        shifted = shift_continuation(continuation_capture, staged.plan,
                                     strip=strip, ring=ring)
        radiation = captured.get("carriers") or {}
        radiative_fields = shift_carriers(radiation.get("fields") or {}, staged.plan, staged.fill)
        lat, _ = footprint.grid.latlon_mass()
        center_lat = float(getattr(footprint.grid, "cen_lat", np.mean(lat)))
        cfg = new_dc.run

        def crop(values, window):
            sy, sx = window
            result = {}
            for name, value in values.items():
                ny, nx = value.shape[-2:]
                if ny not in (cfg.ny, cfg.ny+1) or nx not in (cfg.nx, cfg.nx+1):
                    raise RelocationRefusal(f"continuation {name} has unsupported window shape {value.shape}")
                from woof.core.physics_continuation import _array_module
                result[name] = _array_module(value).ascontiguousarray(
                    value[..., sy.start:sy.stop+(ny-cfg.ny), sx.start:sx.stop+(nx-cfg.nx)])
            return result

        driver_seconds = 0.
        def prepare(initialized, slab_domain, window):
            nonlocal driver_seconds
            driver_seconds += self._rebuild_driver(
                initialized, slab_domain, parent_node, crop(staged.land, window),
                center_lat=center_lat)
            state, driver = initialized.state, initialized.state.physics
            continuation = restore_continuation(state, driver, crop(shifted, window))
            carriers = restore_carriers(driver, crop(radiative_fields, window), radiation.get("contract"))
            set_carrier_scalars(state, scalars)
            self.last_receipt = {
                "overlap_statics": staged.statics_verdict,
                "donor_fill": dict(staged.fill.counts),
                "fields_moved": sorted(staged.land),
                "accumulators_reinitialized": False,
                "physics_continuation": continuation,
                "strip_accumulations": strip_receipt,
                "radiation_carriers": carriers,
                "driver_rebuild_seconds": driver_seconds,
                "preparation": "global donors followed by bounded device windows",
            }

        prepare.staged = staged
        prepare.precipitation_archive = archive
        self._pending_refresh = (int(new_dc.grid_id), footprint.grid, footprint.static_fields)
        return prepare

    def _rebuild_driver(self, initialized, new_dc, parent_node,
                        moved, *, center_lat=None) -> float:
        return rebuild_child_driver_from_land_state(
            exp=self.exp, data=self.data, model=self.model,
            initialized=initialized, child_dc=new_dc,
            parent_node=parent_node, land=moved, center_lat=center_lat)

    @staticmethod
    def _recouple_moved_cumulus(node) -> None:
        """Recouple held PBL, radiation and cumulus on the new footprint.

        Both real-data routes call this after the perturbation transplant.
        Producer cadence stays unchanged; the existing physical rates are
        coupled against the relocated mass and map factors.
        """
        if getattr(node.state, "_relocation_held_physics_recoupled", False):
            return
        driver = getattr(node.state, "physics", None)
        recouple = getattr(driver, "recouple_after_relocation", None)
        if recouple is None:
            # Compatibility with caller-supplied drivers predating the
            # common PBL/radiation continuation hook.
            recouple = getattr(driver, "recouple_cumulus_tendencies", None)
        if callable(recouple):
            recouple(node.state, node.cfg.run)

    def after_move(self, node) -> None:
        import dataclasses

        self._recouple_moved_cumulus(node)
        pending = self._pending_refresh
        self._pending_refresh = None
        if pending is None:
            return
        grid_id, grid, static = pending
        case = self.model._prepared_by_grid_id.get(grid_id)
        if case is not None and dataclasses.is_dataclass(case):
            self.model._prepared_by_grid_id[grid_id] = dataclasses.replace(
                case, grid=grid, static_fields=dict(static))
        if self.writers is not None:
            self.writers.refresh_domain(grid_id, grid=grid,
                                        static_fields=static)


def build_track_writer(exp: ExperimentConfig, outdir):
    """The ``[relocation.track]`` writer for a run, or ``None``.

    Both relocation routes wire this the same way, because a track row is
    rendered from the tracker's fix and neither route contributes
    anything to it.  What the ROUTE owns is the one thing the config
    deliberately does not carry: the run's INITIAL time, which every
    row's valid-time column is measured from.

    ``None`` when the config carries no ``[relocation.track]`` table, and
    then nothing anywhere in the run changes.
    """
    relocation = getattr(exp, "relocation", None)
    config = getattr(relocation, "track", None)
    if config is None:
        return None
    from woof.core.storm_track_writer import TrackWriter
    from woof.core.storm_tracking import all_levels_of

    # The tracker's FIELD and the isobaric surfaces it watches decide
    # this file's columns, and they are fixed at open time -- so the
    # header is a statement about the CONFIG, not about what any one
    # consultation managed to find.  A rotation or echo tracker writes
    # the moving domain's centre and nothing else
    # (storm_track_writer.POSITION_ONLY_FIELDS).
    #
    # ALL the levels, not just the steering ones: a surface named in
    # [relocation.track] output_level that level_hpa does not track is
    # computed for the FILE (FollowConfig.report_level_hpa), so the
    # writer's columns are exactly the surfaces a consultation produces.
    follow = getattr(relocation, "follow", None)
    return TrackWriter(config, initial_time=exp.start_time,
                       outdir=Path(outdir),
                       levels=all_levels_of(follow),
                       tracked_field=str(
                           getattr(follow, "field", "pressure")))


def build_real_relocation_runner(exp: ExperimentConfig,
                                 data: CaseDataConfig, model, outdir, *,
                                 provider=None,
                                 receipts_name="relocation_receipts.json"):
    """Wire the real-data route's RelocationRunner, or ``None``.

    ``None`` when the config names no follow source -- bounds-only
    ``[relocation]`` stays the manual/API mechanism it always was.  With
    a follow source, this is what makes the front-door refusal
    unnecessary on THIS route: the footprint-rebuilt statics initializer
    (:func:`woof.ingest.relocation_init.real_relocation_initializer`)
    and the physics preparer (:class:`RealRelocationChildPreparer`) both
    exist here, because the route holds the input catalog with the
    static source.  Routes without a static source (the prepared domain
    tree) keep their refusal.
    """
    relocation = exp.relocation
    if not (relocation.enabled and (relocation.follow is not None
                                    or relocation.moves)):
        return None
    from woof.core.relocation_runner import RelocationRunner
    from woof.core.streamed_relocation import wire_reconstruction_runner
    from woof.ingest.relocation_init import (
        REAL_DATA_FOOTPRINT_REBUILT_STATICS, real_relocation_initializer)

    grid_id = int(relocation.grid_id)
    nodes = getattr(model, "nodes_by_grid_id", None)
    if nodes is not None and grid_id not in nodes:
        # The follow target is a DORMANT nest: it has no node, no grid
        # and no birth footprint yet, so there is nothing to anchor the
        # placement-translated statics initializer on.  The leg walk
        # rebuilds this the moment the nest is born, which is also the
        # first instant it could legally move.
        return None
    prepared = getattr(model, "_prepared_by_grid_id", {}).get(grid_id)
    require_relocation_sampling_contract(
        getattr(prepared, "static_sampling_contract", None), same_process=True)
    node = model.node(grid_id)
    child_config = node.cfg
    initializer = real_relocation_initializer(
        catalog=model._input_catalog, vertical=exp.vertical,
        child_config=child_config, reference_grid=node.grid,
        reference_i_parent_start=child_config.i_parent_start,
        reference_j_parent_start=child_config.j_parent_start)
    preparer = RealRelocationChildPreparer(exp=exp, data=data, model=model)
    if provider is None:
        return wire_reconstruction_runner(RelocationRunner.from_experiment(
            exp, schedule=model.schedule, on_child_built=preparer,
            initializer=initializer,
            static_provenance=REAL_DATA_FOOTPRINT_REBUILT_STATICS,
            track_writer=build_track_writer(exp, outdir),
            receipts_path=Path(outdir) / receipts_name))
    from woof.core.nest_reach import reach_clamp_for
    return wire_reconstruction_runner(RelocationRunner(
        config=relocation, schedule=model.schedule,
        on_child_built=preparer, provider=provider, initializer=initializer,
        static_provenance=REAL_DATA_FOOTPRINT_REBUILT_STATICS,
        track_writer=build_track_writer(exp, outdir),
        receipts_path=Path(outdir) / receipts_name,
        reach_clamp=reach_clamp_for(exp, grid_id)))


def build_real_relocation_runners(exp: ExperimentConfig,
                                  data: CaseDataConfig, model, outdir):
    """Legacy tree follower plus every currently-live per-domain follower."""
    from dataclasses import replace as _replace

    from woof.core import uh_diag
    from woof.core.relocation_runner import RelocationRunnerCollection
    from woof.core.storm_tracking import StormTracker

    # Allocate every declared consumer on its live parent before streaming
    # freezes the carrier inventory, including children that start later.
    # Adding the child's runner later must not add a new store field.
    uh_diag.allocate_declared_follower_windows(exp, model)
    runners = []
    legacy = build_real_relocation_runner(exp, data, model, outdir)
    if legacy is not None:
        runners.append(legacy)
    legacy_gid = (None if legacy is None else int(legacy.config.grid_id))
    for dc in exp.domains:
        follow = getattr(dc, "follow", None)
        gid = int(dc.grid_id)
        if follow is None or gid not in model.nodes_by_grid_id:
            continue
        if legacy_gid == gid:
            raise ValueError(
                f"d{gid:02d} has both per-domain follow and legacy "
                "tree-level [relocation]; two placement authorities for one "
                "child are refused. Keep exactly one.")
        node = model.node(gid)
        parent = node.parent
        if parent is None:
            continue
        cadence = float(follow.cadence_seconds)
        slot = uh_diag.follow_window_slot(gid)
        provider = StormTracker(follow.tracker, uh_slot=slot)
        relocation = _replace(
            exp.relocation, enabled=True, grid_id=gid,
            max_move_parent_cells=follow.max_move_parent_cells,
            min_overlap_fraction=follow.min_overlap_fraction,
            cadence_seconds=cadence, follow=follow.tracker, moves=(),
            # Each follower owns its explicitly declared track file.
            track=follow.track)
        # The same refusal the configuration load already ran, on the
        # same operands, through the same label helper: a follower that
        # reaches this door has passed it once, and a caller that built an
        # experiment in memory rather than loading a file meets it here.
        from woof.core.nest_lifecycle import FOLLOWER_TABLE, follower_label
        from woof.experiment import _refuse_unservable_follow_cadence
        _refuse_unservable_follow_cadence(
            relocation, exp.domains, follower_label(gid),
            root_dt=exp.root.run.dt, table=FOLLOWER_TABLE,
            follow_table=FOLLOWER_TABLE)
        view = _replace(exp, relocation=relocation)
        runner = build_real_relocation_runner(
            view, data, model, outdir, provider=provider,
            receipts_name=f"relocation_receipts.d{gid:02d}.json")
        if runner is not None:
            runners.append(runner)
    per_domain = any(getattr(dc, "follow", None) is not None for dc in exp.domains)
    if not per_domain:
        return legacy
    return RelocationRunnerCollection(runners)


class PreparedTreeRelocationChildPreparer(RealRelocationChildPreparer):
    """The prepared tree route's relocation preparer.

    The capture / overlap-statics assertion / donor-fill machinery is
    the case-data preparer's, inherited unchanged -- the two routes
    differ only in the seams the case route derives from its
    ``CaseDataConfig``:

    * the driver rebuild uses the NATIVE land-use identity the prepared
      route already binds at t=0 and lets ``initialize_physics`` build
      radiation from the RunConfig (byte-for-byte the t=0
      ``initialize_prepared_physics`` wiring), then re-attaches the
      shared radiation workspace exactly as the tree runner does at
      start;
    * ``after_move`` refreshes the tree runner's SimpleNamespace
      prepared-case bookkeeping (the case route's is a dataclass), so
      the NEXT move's overlap-statics assertion holds the rebuilt
      footprint's statics, not the t=0 crop.
    """

    def __init__(self, *, exp: ExperimentConfig, model,
                 radiation_workspace=None):
        self.exp = exp
        self.data = None
        self.model = model
        self.writers = None
        self.last_receipt = None
        self._captured = None
        self._pending_refresh = None
        self._radiation_workspace = radiation_workspace
        self._plan_override = None

    def _rebuild_driver(self, initialized, new_dc, parent_node,
                        moved, *, center_lat=None) -> float:
        from woof.native_wrf_contract import native_landuse_identity

        # The footprint's own static decides the land-use identity, as the
        # t=0 tree and single-domain paths do: a 20-category static (the
        # 15 arc-second MODIS set) has no lake class, and handing it the
        # constant's ISLAKE = 21 made RUC's mosaic land use index
        # LANDUSEF[20] of a 20-plane static.
        seconds = rebuild_child_driver_from_land_state(
            exp=self.exp, data=None, model=self.model,
            initialized=initialized, child_dc=new_dc,
            parent_node=parent_node, land=moved,
            center_lat=center_lat,
            landuse_attrs=native_landuse_identity(
                getattr(initialized, "static_fields", None)),
            radiation_factory=lambda _dc, _state, _lat, _lon: None)
        driver = getattr(initialized.state, "physics", None)
        radiation = (None if driver is None
                     else driver.radiation_callable)
        from woof.core.radiation_composition import attach_modern_workspace
        attach_modern_workspace(radiation, self._radiation_workspace)
        return seconds

    def after_move(self, node) -> None:
        self._recouple_moved_cumulus(node)
        pending = self._pending_refresh
        self._pending_refresh = None
        if pending is None:
            return
        grid_id, grid, static = pending
        case = self.model._prepared_by_grid_id.get(grid_id)
        if case is not None:
            # The tree runner's bookkeeping is a mutable SimpleNamespace;
            # refresh in place so the next capture_outgoing snapshots the
            # statics the child actually sits on.
            case.static_fields = dict(static)
            case.geog_selection = None
        if self.writers is not None:
            self.writers.refresh_domain(grid_id, grid=grid,
                                        static_fields=static)


def build_prepared_tree_relocation_runner(exp: ExperimentConfig, *,
                                          statics_corridor, model, outdir,
                                          radiation_workspace=None,
                                          follow_window_slot=None,
                                          receipts_name="relocation_receipts.json"):
    """Wire the prepared tree route's RelocationRunner, or ``None``.

    The prepared-route counterpart of
    :func:`build_real_relocation_runner`: same runner, same initializer,
    same preparer seams -- only the statics source differs (the sealed
    corridor crop instead of a per-footprint GEOG build), which is what
    lifts this route's follow-source refusal WHEN a verified corridor is
    on hand.  ``None`` for bounds-only ``[relocation]`` exactly as on
    the case-data route.
    """
    relocation = exp.relocation
    if not (relocation.enabled and (relocation.follow is not None
                                    or relocation.moves)):
        return None
    if statics_corridor is None:
        raise ValueError(
            "build_prepared_tree_relocation_runner requires the verified "
            "statics corridor; the preflight refusal owns the "
            "corridor-less case and must not be bypassed here")
    from woof.core.relocation_runner import RelocationRunner
    from woof.ingest.relocation_init import real_relocation_initializer
    from woof.static.corridor import (CORRIDOR_REBUILT_STATICS,
                                       corridor_footprint_statics_builder)

    corridors = (statics_corridor if isinstance(statics_corridor, dict)
                 else {int(relocation.grid_id): statics_corridor})
    for grid_id, corridor in sorted(corridors.items()):
        dc = exp.domain(int(grid_id))
        if not getattr(dc.run, "chem_sets", ""):
            continue
        from woof.chem_table import load as load_chem_table
        from woof.core.chem_statics import DUST_KEY, STATIC_VARIABLES
        if load_chem_table(dc.run).rows_for(DUST_KEY):
            missing = [name for name in STATIC_VARIABLES
                       if name not in corridor.fields]
            if missing:
                raise ValueError(
                    f"prepared moving dust chemistry d{int(grid_id):02d} "
                    f"requires sealed corridor statics {missing}; "
                    "reprepare the chemistry-enabled statics corridor "
                    "before moving the child")
    node = model.node(int(relocation.grid_id))
    child_config = node.cfg

    def _mover_statics_builder(grid_id: int):
        """Corridor crop for a mover, honouring the corridor's frame.

        A parent-anchored corridor (nothing above the mover moves) crops
        by the mover's own placement, exactly as before.  A ROOT-anchored
        one -- the [relocation.containment] case, where the mover's
        parent slides -- must crop at the mover's live origin in the
        root's frame, composed from the LIVE ancestor placements plus the
        placement being built (the tree still carries the old one while
        the initializer runs), which is why the live map is patched with
        ``new_dc`` before the origin is taken.
        """
        from woof.static.corridor import (corridor_frame_kwargs,
                                           origin_in_frame_cells)

        corridor = corridors[int(grid_id)]
        if not corridor_frame_kwargs(exp, model.node(int(grid_id)).cfg):
            return corridor_footprint_statics_builder(corridor)
        root_id = next(int(d.grid_id) for d in exp.domains
                       if int(d.parent_id) in (0, int(d.grid_id)))

        def statics_builder(grid, new_dc):
            del grid
            live = {int(n.cfg.grid_id): n.cfg
                    for n in model.walk_parent_first()}
            live[int(new_dc.grid_id)] = new_dc
            origin = origin_in_frame_cells(
                live, int(new_dc.grid_id), root_id)
            return corridor.crop_at(*origin)

        statics_builder.static_provenance = CORRIDOR_REBUILT_STATICS
        statics_builder.source_label = (
            f"statics-corridor d{int(grid_id):02d} (root frame) "
            f"sha256:{corridor.cache_sha256[:12]}")
        statics_builder.highres_applied = False
        return statics_builder

    def _root_frame_shift_resolver(grid_id: int):
        """The mover's translation in ITS OWN cells, root-frame composed.

        None when nothing above the mover moves -- the legacy
        placement-difference formula is then exact.  Under
        [relocation.containment] the mover's parent slides, so the
        placement is an offset inside a moving frame; the resolver
        composes the LIVE ancestor chain (patched with the placement
        being built) against the same chain at wiring time.  Every term
        is an integer in the mover's own cells, the same arithmetic the
        root-framed corridor crop stands on.
        """
        from woof.static.corridor import (corridor_frame_kwargs,
                                           origin_in_frame_cells)

        if not corridor_frame_kwargs(exp, model.node(int(grid_id)).cfg):
            return None
        root_id = next(int(d.grid_id) for d in exp.domains
                       if int(d.parent_id) in (0, int(d.grid_id)))
        reference = {int(n.cfg.grid_id): n.cfg
                     for n in model.walk_parent_first()}
        ref_oi, ref_oj = origin_in_frame_cells(
            reference, int(grid_id), root_id)

        def resolve(new_dc, parent_node):
            del parent_node
            live = {int(n.cfg.grid_id): n.cfg
                    for n in model.walk_parent_first()}
            live[int(new_dc.grid_id)] = new_dc
            oi, oj = origin_in_frame_cells(live, int(new_dc.grid_id),
                                           root_id)
            return oi - ref_oi, oj - ref_oj

        return resolve

    initializer = real_relocation_initializer(
        vertical=exp.vertical, child_config=child_config,
        reference_grid=node.grid,
        reference_i_parent_start=child_config.i_parent_start,
        reference_j_parent_start=child_config.j_parent_start,
        statics_builder=_mover_statics_builder(int(relocation.grid_id)),
        root_frame_shift=_root_frame_shift_resolver(
            int(relocation.grid_id)))
    preparer = PreparedTreeRelocationChildPreparer(
        exp=exp, model=model, radiation_workspace=radiation_workspace)
    owner_roots = {int(relocation.grid_id)}
    if relocation.containment is not None:
        owner_roots.add(int(relocation.containment.grid_id))
    kwargs = dict(
        schedule=model.schedule, on_child_built=preparer,
        initializer=initializer,
        static_provenance=CORRIDOR_REBUILT_STATICS,
        reground_descendant=build_prepared_tree_descendant_regrounder(
            exp, model=model, corridors=corridors,
            radiation_workspace=radiation_workspace, moving_roots=owner_roots),
        track_writer=build_track_writer(exp, outdir),
        receipts_path=Path(outdir) / receipts_name)
    if follow_window_slot is None:
        runner = RelocationRunner.from_experiment(exp, **kwargs)
    else:
        # Same declared tracker, with this consumer's generated window.
        # No programmatic provider may replace the user's follow source.
        from woof.core.storm_tracking import StormTracker
        if relocation.follow is None or relocation.moves:
            raise ValueError("a follower window requires a declared tracker")
        from woof.core.nest_reach import reach_clamp_for
        runner = RelocationRunner(
            config=relocation,
            provider=StormTracker(relocation.follow, uh_slot=follow_window_slot),
            reach_clamp=reach_clamp_for(exp, int(relocation.grid_id)),
            **kwargs)
    containment = getattr(relocation, "containment", None)
    if containment is not None:
        parent_node = model.node(int(containment.grid_id))
        runner.wire_containment(
            initializer=real_relocation_initializer(
                vertical=exp.vertical, child_config=parent_node.cfg,
                reference_grid=parent_node.grid,
                reference_i_parent_start=parent_node.cfg.i_parent_start,
                reference_j_parent_start=parent_node.cfg.j_parent_start,
                statics_builder=_mover_statics_builder(
                    int(containment.grid_id))),
            on_child_built=PreparedTreeRelocationChildPreparer(
                exp=exp, model=model,
                radiation_workspace=radiation_workspace),
            static_provenance=CORRIDOR_REBUILT_STATICS)
    from woof.core.streamed_relocation import wire_reconstruction_runner
    return wire_reconstruction_runner(runner)


def build_prepared_tree_relocation_runners(exp, *, statics_corridor, model,
                                           outdir, radiation_workspace=None):
    """Bind every live declared follower to its verified static corridor."""
    from dataclasses import replace
    from woof.core import uh_diag
    from woof.core.relocation_runner import RelocationRunnerCollection
    from woof.experiment import RelocationConfig, _refuse_unservable_follow_cadence

    legacy = build_prepared_tree_relocation_runner(
        exp, statics_corridor=statics_corridor, model=model, outdir=outdir,
        radiation_workspace=radiation_workspace)
    followers = [dc for dc in exp.domains
                 if getattr(dc, "follow", None) is not None]
    if not followers:
        return legacy
    uh_diag.allocate_declared_follower_windows(exp, model)
    runners = [] if legacy is None else [legacy]
    for dc in followers:
        gid = int(dc.grid_id)
        if legacy is not None and gid == int(legacy.config.grid_id):
            raise ValueError(f"d{gid:02d} has two placement authorities: "
                             "per-domain follow and legacy [relocation]")
        if gid not in model.nodes_by_grid_id:
            continue
        follow = dc.follow
        relocation = RelocationConfig(
            enabled=True, grid_id=gid, follow=follow.tracker,
            cadence_seconds=follow.cadence_seconds,
            max_move_parent_cells=follow.max_move_parent_cells,
            min_overlap_fraction=follow.min_overlap_fraction,
            track=follow.track)
        from woof.core.nest_lifecycle import FOLLOWER_TABLE, follower_label
        _refuse_unservable_follow_cadence(
            relocation, exp.domains, follower_label(gid),
            root_dt=exp.root.run.dt, table=FOLLOWER_TABLE,
            follow_table=FOLLOWER_TABLE)
        # Keep ALL declarations in this view: an ancestor's independent
        # follower determines the descendant corridor's coordinate frame.
        view = replace(exp, relocation=relocation)
        runners.append(build_prepared_tree_relocation_runner(
            view, statics_corridor=statics_corridor, model=model, outdir=outdir,
            radiation_workspace=radiation_workspace,
            follow_window_slot=uh_diag.follow_window_slot(gid),
            receipts_name=f"relocation_receipts.d{gid:02d}.json"))
    return RelocationRunnerCollection(runners)


def build_prepared_tree_descendant_regrounder(exp, *, model, corridors,
                                              radiation_workspace=None,
                                              moving_roots=None):
    """The mid-tree seam: re-ground one descendant of a moved domain.

    ``relocate_child`` hands this every descendant of the mover, with the
    plan whose shift is that descendant's ground displacement in its own
    cells.  The descendant's PLACEMENT is untouched -- it is an offset
    inside a parent that carried it along -- so there is no re-placement
    here; what there is, is new ground, and therefore the same rebuild
    the mover itself gets.

    ``None`` when nothing below the mover exists, which keeps a leaf
    mover on exactly the path it had before mid-tree moves existed.

    WHERE THE STATICS COME FROM.  The descendant's own corridor, cropped
    at its LIVE origin in the ROOT's frame.  That origin is read from the
    tree AFTER the mover has been re-placed, so it already includes the
    parent's displacement -- and it is counted in the descendant's own
    cells, the only frame in which it is an integer
    (:func:`woof.static.corridor.origin_in_frame_cells`).
    """
    from woof.core.nest_relocation import (_DONOR_ALIGNMENT_FIELDS,
                                            RelocationRefusal,
                                            donor_alignment_check,
                                            relocatable_attrs,
                                            release_state_arrays,
                                            snapshot_state_to_host,
                                            transplant_overlap)
    from woof.ingest.relocation_init import real_relocation_initializer
    from woof.static.corridor import (CORRIDOR_REBUILT_STATICS,
                                       origin_in_frame_cells,
                                       relocating_subtree_grid_ids)

    subtree = relocating_subtree_grid_ids(exp, moving_roots=moving_roots)
    if len(subtree) < 2:
        return None
    # ONE PREPARER PER DESCENDANT, never the mover's.
    # PreparedTreeRelocationChildPreparer holds its outgoing capture in a
    # single slot (`self._captured`), which is correct for the one domain
    # it was built to serve and wrong the moment a second domain borrows
    # it: the descendant's capture lands in the mover's slot, and the
    # mover's NEXT move then asserts its statics against the descendant's.
    # MEASURED before this was split: move 1 passed, move 2 refused with
    # 2749 mismatched cells in LANDUSEF/SOILCTOP/SOILCBOT -- a whole
    # domain's worth of disagreement, not the one-ULP kind.
    preparers = {gid: PreparedTreeRelocationChildPreparer(
        exp=exp, model=model, radiation_workspace=radiation_workspace)
        for gid in subtree}

    def attach_writers(writers) -> None:
        """Fan the run's writers out to EVERY descendant preparer.

        ``PreparedTreeRelocationChildPreparer.after_move`` refreshes a
        domain's output grid and statics through ``self.writers``, and
        returns silently when that is None.  The route attaches writers to
        the mover's preparer only (``relocation_runner.on_child_built``),
        so the per-descendant preparers introduced here would each hold
        None and refresh nothing -- the descendant would integrate on its
        new ground while every frame it wrote kept the coordinates of the
        footprint it had left.

        MEASURED before this: across six d02 moves the parent's XLONG
        marched -153.887 -> -155.438 while d03's stayed pinned at
        -152.451..-143.607, until the stale extent poked outside its own
        parent.  The values were computed on the right ground; only the
        georeferencing was wrong, which is exactly the failure that looks
        fine until someone plots it.
        """
        for member in preparers.values():
            member.attach_writers(writers)
    root_id = next(int(d.grid_id) for d in exp.domains
                   if int(d.parent_id) in (0, int(d.grid_id)))

    def reground(*, node, plan, delta_parent_cells):
        grid_id = int(node.cfg.grid_id)
        corridor = corridors[grid_id]
        preparer = preparers[grid_id]

        def statics_builder(grid, new_dc):
            del grid
            live = {int(n.cfg.grid_id): n.cfg
                    for n in model.walk_parent_first()}
            origin = origin_in_frame_cells(live, int(new_dc.grid_id), root_id)
            return corridor.crop_at(*origin)

        statics_builder.static_provenance = CORRIDOR_REBUILT_STATICS
        statics_builder.source_label = (
            f"statics-corridor d{grid_id:02d} (root frame) "
            f"sha256:{corridor.cache_sha256[:12]}")
        statics_builder.highres_applied = False

        # THE REFERENCE PLACEMENT IS OFFSET, and it has to be.  The
        # initializer derives its translation from the placement CHANGE,
        # `(new_dc.i_parent_start - ref_i) * ratio`, and a descendant's
        # placement does not change -- so declaring the real one yields a
        # zero shift and leaves the reference grid describing pre-move
        # ground, while the parent-resolved grid has already moved.  Its
        # drift gate then refuses by exactly the parent's displacement.
        #
        # Declaring `placement - delta` makes the shift come out as the
        # ground displacement, which is what actually happened to this
        # domain.  Same trick, and the same reason, as the synthetic
        # placements in `plan_descendant_reground`: the pair is a way to
        # spell a displacement, not a record of where anything sits.
        delta_i, delta_j = delta_parent_cells
        initializer = real_relocation_initializer(
            vertical=exp.vertical, child_config=node.cfg,
            reference_grid=node.grid,
            reference_i_parent_start=int(node.cfg.i_parent_start) - delta_i,
            reference_j_parent_start=int(node.cfg.j_parent_start) - delta_j,
            statics_builder=statics_builder)

        capture = getattr(preparer, "capture_outgoing", None)
        if callable(capture):
            capture(node)
        # The descendant's placement pair says nothing about how far its
        # ground moved; `plan` does.  See the plan-override comment in
        # RealRelocationChildPreparer.__call__.
        preparer._plan_override = plan
        from woof.ensemble.runtime_context import (current_member_reconstruction_owner,
                                                    bind_reconstructed_member_node)
        member_pattern_owner = current_member_reconstruction_owner(node.state)
        factory = getattr(reground, "streamed_reconstruction_factory", None)
        reconstruction = (factory(node, initializer=initializer, preparer=preparer)
                          if callable(factory) and getattr(node.state, "_streamed_domain", None) is not None
                          else None)
        if reconstruction is None:
            source_state = snapshot_state_to_host(
                node.state,
                tuple(relocatable_attrs(node.state)) + _DONOR_ALIGNMENT_FIELDS)
            release_state_arrays(node.state)
            initialized = initializer(
                node.cfg, node.parent,
                scratch_arena=getattr(model, "_scratch_arena", None),
                dycore_state_workspace=getattr(
                    model, "_dycore_state_workspace", None))
            preparer(initialized, node.cfg, node.parent)
        else:
            source_state = reconstruction.capture_source(node)
            reconstruction.release_outgoing(node)
            initialized = reconstruction.initialize(node.cfg, node.parent)

        frame_width = int(
            getattr(initializer, "donor_alignment_frame_width", 0) or 0)
        alignment = donor_alignment_check(
            source_state=source_state, target_state=initialized.state,
            plan=plan, frame_width=frame_width)
        if not alignment["pass"]:
            raise RelocationRefusal(
                f"descendant d{grid_id:02d}'s re-grounded base state does "
                f"not match its outgoing one on the overlap, so the two "
                f"footprints do not share donor cells: {alignment}. The "
                "ground displacement carried down from the mover is wrong, "
                "or its corridor is anchored to a frame that moved.")
        transplant = transplant_overlap(
            source_state=source_state, target_state=initialized.state,
            plan=plan)
        # The same two-point probe the mover gets; a descendant is the
        # control that made the mover's deficit legible in the first
        # place, so it has to be measurable the same way.
        from woof.core.nest_relocation import (
            overlap_prognostic_mismatches, relocation_probe_enabled)
        probe = None
        if relocation_probe_enabled():
            probe = {"after_transplant": overlap_prognostic_mismatches(
                source_state, initialized.state, plan)}
        post_transplant = getattr(reconstruction if reconstruction is not None else initializer,
                                  "post_transplant", None)
        post_receipt = (
            None if post_transplant is None else post_transplant(
                source_state=source_state, target_state=initialized.state,
                plan=plan))
        if probe is not None:
            probe["after_post_transplant"] = overlap_prognostic_mismatches(
                source_state, initialized.state, plan)

        # THE RK TIME-t SEEDS, which a descendant needs for exactly the
        # reason the mover does.  `real_relocation_initializer` runs WRF's
        # start_domain lineage, which seeds the time-t copies (u0, thp0,
        # qv0, qr0, ...) from the SINT-of-parent cold-start fields.  The
        # transplant then overwrites the CURRENT fields on the overlap
        # with the outgoing child's -- so without re-seeding, the first
        # RK substep reads cold-start values over ground the transplant
        # just corrected, and every prognostic starts the step with a
        # fabricated tendency equal to (transplanted - interpolated).
        #
        # `relocate_child` does this for the mover (`rk_seeds_refreshed`
        # on its receipt); nothing was doing it for the domains carried
        # along, so a descendant began each post-move step from its
        # parent's interpolation rather than from itself.  Same call,
        # same place in the sequence: after post_transplant, before the
        # node adopts the new state.
        from woof.ingest.nest_init import seed_rk_time_t_copies
        rk_seeds = (seed_rk_time_t_copies(initialized.state) if reconstruction is None
                    else reconstruction.rk_seeds)

        node.grid = getattr(initialized, "grid", node.grid)
        node.state = initialized.state
        if reconstruction is not None:
            reconstruction.commit(node)
        # THE POST-MOVE SEAM, which a descendant needs exactly as much as
        # the mover does.  RelocationRunner calls these two for the domain
        # it moved; nothing was calling them for the domains that moved
        # WITH it, so a descendant integrated on its new ground while its
        # output metadata still described the old one.
        #
        # MEASURED before this: after d02 moved [-2,-2], its wrfout
        # correctly reported I_PARENT_START 105 and XLONG shifted by
        # -0.517 deg, while d03 -- which had ridden along the same
        # distance -- still wrote the XLAT/XLONG of its pre-move
        # footprint.  Every d03 frame after a move was georeferenced to
        # ground it had already left, which is invisible in the field
        # values and obvious the moment it is plotted against its parent.
        from woof.core.state import refresh_model_time
        refresh_model_time(node.state, node.clock)
        after_move = getattr(preparer, "after_move", None)
        if callable(after_move):
            after_move(node)
        bind_reconstructed_member_node(node, previous_owner=member_pattern_owner)
        return {
            "statics": statics_builder.source_label,
            "static_fields": CORRIDOR_REBUILT_STATICS,
            "donor_alignment": alignment,
            "transplant": transplant,
            "prognostic_overlap_probe": probe,
            "rk_seeds_refreshed": len(rk_seeds),
            "post_transplant": post_receipt,
            "rebuild": getattr(initialized, "preprocess_receipt", None),
        }

    reground.attach_writers = attach_writers
    return reground


class RealSpawnChildPreparer:
    """Physics/land attachment for a NEWBORN nest on the real-data route.

    The ``on_child_built`` the route hands
    :class:`woof.core.spawn_runner.SpawnRunner`.  It is the same seam,
    and the same rule, as every leg boundary and every relocation: the
    initializer never invents driver state, the route re-initialises it
    here.

    WHERE THE LAND STATE COMES FROM.  A newborn has no prior self to
    continue from, so unlike the relocation preparer there is nothing to
    capture; and it has no ``real.exe`` product at a footprint nobody knew
    about until the trigger fired, so unlike the t = 0 child preparer
    there is no analysis-derived soil either.  What it does have is the
    live parent, and that is precisely the case WRF's own nest
    initialization is built for: ``med_nest_initial`` fills the whole
    fine grid from ``med_interp_domain(parent, nest)`` BEFORE any input
    file is consulted, and for a nest without one that interpolation is
    the initialization (Users' Guide chapter 5; share/mediation_integrate
    .F:670).  So the land state is
    :func:`~woof.ingest.nest_spawn_init.spawn_land_state_from_parent` --
    the Registry's own masked surface interpolator, run against the
    newborn's OWN-GRID land-use categories -- and the driver rebuild is
    then the shared :func:`rebuild_child_driver_from_land_state`, byte
    for byte the sequence a relocation runs.

    ACCUMULATED PRECIPITATION starts from the parent's, not zero
    (:func:`woof.core.physics_continuation.seed_birth_accumulations`): a
    living nest moves, and each move seeds its new ground with the
    parent's rain since the run began, so ground it has held since birth
    must count from the same start or every move leaves a band of extra
    rain on its leading edge.  The other accumulators are re-initialised.

    ``last_receipt`` is the duck-typed seam the runner reads (the
    relocation runner's idiom), so the land accounting reaches the spawn
    receipt instead of dying here.
    """

    def __init__(self, *, exp: ExperimentConfig, data: CaseDataConfig,
                 model):
        self.exp = exp
        self.data = data
        self.model = model
        self.prepared_by_grid_id: dict[int, object] = {}
        self.last_receipt = None

    def __call__(self, initialized, child_dc, parent_node) -> None:
        import time as _time

        from woof.ingest.nest_spawn_init import (SpawnInitRefusal,
                                                  spawn_land_state_from_parent)

        started = _time.perf_counter()
        grid_id = int(child_dc.grid_id)
        static = initialized.static_fields
        if static is None:
            raise SpawnInitRefusal(
                "the spawn initializer produced no static fields; the "
                "real-data route requires own-grid statics at the fired "
                "footprint, and the masked land interpolator has no "
                "destination land-use categories without them")
        parent_grid_id = int(parent_node.cfg.grid_id)
        parent_case = self.model._prepared_by_grid_id.get(parent_grid_id)
        parent_static = getattr(parent_case, "static_fields", None)
        if parent_static is None:
            raise SpawnInitRefusal(
                f"the parent d{parent_grid_id:02d} has no statics on "
                "record, so its land-use categories -- the SOURCE mask of "
                "WRF's masked surface interpolator -- are unavailable; a "
                "newborn's land state cannot be interpolated without them")
        geog_selection = GeogSelection.from_case_data(
            self.data, domain_id=grid_id)
        land = spawn_land_state_from_parent(
            child_dc, parent_node, static_fields=static,
            parent_static_fields=parent_static,
            landuse_attrs=geog_selection.landuse_global_attrs())
        driver_seconds = rebuild_child_driver_from_land_state(
            exp=self.exp, data=self.data, model=self.model,
            initialized=initialized, child_dc=child_dc,
            parent_node=parent_node, land=land["fields"])
        from woof.core.physics_continuation import seed_birth_accumulations

        precipitation = seed_birth_accumulations(
            initialized.state, parent_node, child_dc)
        context = self.model._activation_context or {}
        snow = land["fields"].get("snow")
        # The newborn's own "initial result" IS the materialized child:
        # the wrfout writers read `initial_result.coord` off the prepared
        # case, and the parent-frame coordinate the SINT fill produced is
        # the one this domain will integrate on.  There is no analysis
        # product to point at, and inventing one would be a lie.
        prepared = PreparedRealCase(
            static_sampling_contract=current_sampling_contract(),
            cfg=child_dc.run, grid=initialized.grid,
            static_fields=dict(static), initial_result=initialized,
            final_analysis=None,
            initial_snow_water_kgm2=(
                np.zeros(tuple(initialized.grid.latlon_mass()[0].shape),
                         dtype=np.float64) if snow is None
                else np.array(snow, dtype=np.float64, copy=True)),
            forcing_times=tuple(context.get("forcing_times", ())),
            geog_selection=geog_selection)
        self.prepared_by_grid_id[grid_id] = prepared
        self.last_receipt = {
            "land_surface": land["receipt"],
            "accumulators_reinitialized": True,
            "precipitation_accumulations": precipitation,
            "driver_rebuild_seconds": driver_seconds,
            "preparer_seconds": _time.perf_counter() - started,
        }


def build_real_spawn_runner(exp: ExperimentConfig, data: CaseDataConfig,
                            model, outdir):
    """Wire the real-data route's SpawnRunner, or ``None``.

    ``None`` when no ``[[domain]]`` declares ``spawn``.  This is what
    lifts the front-door refusal on THIS route and only here: the route
    holds the input catalog, so the newborn's own-grid statics can be
    built at the fired footprint, and it holds the case data and forcing
    calendar, so its physics driver can be attached.  Routes without
    them keep the refusal (woof.experiment.refuse_unrouted_spawn).
    """
    from woof.core.spawn_runner import RECEIPTS_SUFFIX, SpawnRunner

    preparer = RealSpawnChildPreparer(exp=exp, data=data, model=model)

    def statics_provider(child_dc, parent_node):
        from woof.ingest.nest_spawn_init import prepare_spawn_statics

        return prepare_spawn_statics(
            child_dc, parent_node, model._input_catalog,
            valid_date=exp.start_time)

    return SpawnRunner.from_experiment(
        exp, on_child_built=preparer, statics_provider=statics_provider,
        receipts_path=Path(outdir) / f"spawn_receipts{RECEIPTS_SUFFIX}")


class PreparedTreeSpawnChildPreparer:
    """Physics/land attachment for a NEWBORN nest on the prepared tree route.

    The prepared-tree twin of :class:`RealSpawnChildPreparer`, differing
    only in the seams the case route derives from its ``CaseDataConfig``,
    exactly as :class:`PreparedTreeRelocationChildPreparer` differs from
    its case-route parent: the land-use identity is the native one the
    prepared route binds at t = 0, radiation is built from the RunConfig
    and the shared radiation workspace re-attached, and the prepared-case
    bookkeeping is the tree runner's SimpleNamespace.
    """

    def __init__(self, *, exp: ExperimentConfig, model,
                 radiation_workspace=None):
        self.exp = exp
        self.model = model
        self.prepared_by_grid_id: dict[int, object] = {}
        self.last_receipt = None
        self._radiation_workspace = radiation_workspace

    def __call__(self, initialized, child_dc, parent_node) -> None:
        import time as _time
        from types import MappingProxyType, SimpleNamespace

        from woof.ingest.nest_spawn_init import (SpawnInitRefusal,
                                                  spawn_land_state_from_parent)
        from woof.native_wrf_contract import native_landuse_identity

        started = _time.perf_counter()
        grid_id = int(child_dc.grid_id)
        static = initialized.static_fields
        if static is None:
            raise SpawnInitRefusal(
                f"the spawn initializer produced no static fields for "
                f"d{grid_id:02d}; the prepared tree route crops a newborn's "
                "statics from its sealed corridor, and the masked land "
                "interpolator has no destination land-use categories "
                "without them")
        parent_grid_id = int(parent_node.cfg.grid_id)
        parent_case = self.model._prepared_by_grid_id.get(parent_grid_id)
        parent_static = getattr(parent_case, "static_fields", None)
        if parent_static is None:
            raise SpawnInitRefusal(
                f"the parent d{parent_grid_id:02d} has no statics on "
                "record, so its land-use categories -- the SOURCE mask of "
                "WRF's masked surface interpolator -- are unavailable")
        landuse_attrs = native_landuse_identity(static)
        land = spawn_land_state_from_parent(
            child_dc, parent_node, static_fields=static,
            parent_static_fields=parent_static, landuse_attrs=landuse_attrs)
        driver_seconds = rebuild_child_driver_from_land_state(
            exp=self.exp, data=None, model=self.model,
            initialized=initialized, child_dc=child_dc,
            parent_node=parent_node, land=land["fields"],
            landuse_attrs=landuse_attrs,
            radiation_factory=lambda _dc, _state, _lat, _lon: None)
        driver = getattr(initialized.state, "physics", None)
        radiation = None if driver is None else driver.radiation_callable
        from woof.core.radiation_composition import attach_modern_workspace
        attach_modern_workspace(radiation, self._radiation_workspace)
        from woof.core.physics_continuation import seed_birth_accumulations

        precipitation = seed_birth_accumulations(
            initialized.state, parent_node, child_dc)
        wrf_file_grid = getattr(parent_case, "wrf_file_grid", None)
        self.prepared_by_grid_id[grid_id] = SimpleNamespace(
            static_fields=dict(static), geog_selection=None,
            initial_result=initialized, streamed_store=None,
            wrf_file_grid=(MappingProxyType({}) if wrf_file_grid is None
                           else wrf_file_grid))
        self.last_receipt = {
            "land_surface": land["receipt"],
            "accumulators_reinitialized": True,
            "precipitation_accumulations": precipitation,
            "driver_rebuild_seconds": driver_seconds,
            "preparer_seconds": _time.perf_counter() - started,
        }


def build_prepared_tree_spawn_runner(exp: ExperimentConfig, *,
                                     statics_corridor, model, outdir,
                                     radiation_workspace=None):
    """Wire the prepared tree route's SpawnRunner, or ``None``.

    ``None`` when no ``[[domain]]`` declares ``spawn``.  A newborn's
    own-grid statics are the crop of its sealed statics corridor at the
    fired placement: a dormant slot's corridor reaches the whole parent
    frame (:func:`woof.static.corridor.corridor_reach`), so every
    placement the watch can choose has sealed ground.  A dormant slot
    with no corridor refuses here, by name, before the first step.
    """
    from woof.core.spawn_runner import RECEIPTS_SUFFIX, SpawnRunner
    from woof.experiment import dormant_domain_ids

    dormant = tuple(int(gid) for gid in dormant_domain_ids(exp))
    if not dormant:
        return None
    corridors = dict(statics_corridor or {})
    missing = [gid for gid in dormant if gid not in corridors]
    if missing:
        named = ", ".join(f"d{gid:02d}" for gid in missing)
        raise ValueError(
            f"{named} declare(s) spawn but this prepared bundle carries no "
            "statics corridor for them: the prepared tree route has no "
            "GEOG source at runtime, so a nest born at a placement chosen "
            "mid-run has no ground.  Re-prepare with --statics-corridor "
            "(woof go passes it for any config with a follow source).")
    preparer = PreparedTreeSpawnChildPreparer(
        exp=exp, model=model, radiation_workspace=radiation_workspace)

    def statics_provider(child_dc, parent_node):
        corridor = corridors[int(child_dc.grid_id)]
        fields = corridor.crop(int(child_dc.i_parent_start),
                               int(child_dc.j_parent_start))
        from woof.native_wrf_contract import native_landuse_identity
        return {"static_fields": {name: np.array(value, copy=True)
                                  for name, value in fields.items()},
                "landuse_attrs": native_landuse_identity(fields),
                "receipt": {
                    "static_source": (
                        f"statics-corridor d{int(child_dc.grid_id):02d} "
                        f"sha256:{corridor.cache_sha256[:12]}"),
                    "highres_overlay": bool(corridor.highres_applied),
                    "placement": [int(child_dc.i_parent_start),
                                  int(child_dc.j_parent_start)]}}

    return SpawnRunner.from_experiment(
        exp, on_child_built=preparer, statics_provider=statics_provider,
        receipts_path=Path(outdir) / f"spawn_receipts{RECEIPTS_SUFFIX}")


def _tree_forcing_cadence_seconds(catalog) -> float:
    """The tree builder's own LBC cadence, so a rebuilt leg clock matches.

    Imported rather than re-derived: ``resolve_clock`` must see exactly
    the interval ``build_experiment`` gave it, or a leg boundary would
    quietly re-phase the root's external-boundary calendar.
    """
    from woof.core.model import _forcing_cadence_seconds

    return _forcing_cadence_seconds(catalog)


def _spawn_leg_seconds(exp: ExperimentConfig) -> float:
    """How often the walk stops to ask whether a nest should be born.

    Coarse on purpose.  Every boundary costs one schedule rebuild, and a
    boundary is only USEFUL where the trigger could newly fire, so this
    takes the relocation cadence when the config sets one (the same
    instant the tracker is already consulted, and already validated as a
    whole number of root steps), else the root's history interval, which
    is also where the reflectivity signal is stashed.
    """
    cadence = getattr(getattr(exp, "relocation", None),
                      "cadence_seconds", None)
    if cadence:
        return float(cadence)
    # A dormant slot that says how often to look sets the leg (the shortest
    # one wins); slots that say nothing keep the history-interval leg.
    looks = [float(dc.spawn.cadence_s) for dc in exp.domains
             if getattr(getattr(dc, "spawn", None), "cadence_s", None)]
    if looks:
        return min(looks)
    history = float(getattr(exp.root, "history_interval_s", 0.0) or 0.0)
    return history if history > 0.0 else float(exp.run_seconds)


def publish_lifecycle_runners(model, *, spawn_runner=None,
                              relocation_runner=None,
                              leg_seconds=None) -> None:
    """Bind the live lifecycle runners to the tree the checkpoint sees.

    ``restart_handler`` is handed a TREE and a tick count and nothing
    else, while the runners are the route's own locals.  Without this
    binding a checkpoint could say which domains existed and nothing
    more: not which slots had fired, not how many episodes a slot had
    served, not where a follower's segment chain stood -- which is the
    whole of the state a later leg boundary reads and cannot recompute.

    Published at EVERY rebind, not once at build: the leg walk replaces
    the relocation runner as dormant follow targets come alive, and a
    checkpoint written after that point holding the pre-spawn binding
    would omit the newborn follower's history entirely.
    """
    model._spawn_runner = spawn_runner
    model._relocation_runner = relocation_runner
    if leg_seconds is not None:
        model._spawn_leg_seconds = float(leg_seconds)


def _retarget_tree_schedule(model, active_exp: ExperimentConfig,
                            end_seconds: float, lbc_interval_s) -> None:
    """Re-aim the live tree at ``end_seconds`` over ``active_exp``.

    The leg-boundary schedule surgery.  Clocks are minted fresh from the
    new domain set and then carried to the tick the tree is actually at,
    exactly as ``restore_tree_restart`` does across a checkpoint -- the
    executor derives its resume period from the root clock, so a tree
    whose clocks read the boundary resumes there rather than replaying.
    A node with no clock is a newborn: it joins at the boundary.
    """
    from dataclasses import replace as _replace

    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.state import refresh_model_time

    leg_exp = _replace(
        active_exp, run_seconds=float(end_seconds),
        domains=tuple(
            _replace(dc, run=_replace(dc.run, run_seconds=float(end_seconds)))
            for dc in active_exp.domains))
    schedule_now = getattr(model, "schedule", None)
    previous_den = (None if schedule_now is None
                    else int(schedule_now.clock.tick_den))
    tick_clock = resolve_clock(leg_exp, lbc_interval_s=lbc_interval_s,
                               tick_den_floor=previous_den)
    if previous_den is not None and int(tick_clock.tick_den) != previous_den:
        # The live clocks count in the previous tick; carrying their integer
        # ticks into a finer one would re-mean every one of them (a newborn
        # born at 1800 s read 600 s).  The startup tree takes the declared
        # tree's tick so this cannot arise; refused by name if it does.
        raise RuntimeError(
            f"spawn leg needs tick 1/{tick_clock.tick_den} s but the live tree "
            f"counts in 1/{previous_den} s; the startup clock must be resolved "
            "over the declared tree (resolve_clock tick_den_floor)")
    schedule = build_schedule(leg_exp, tick_clock)
    fresh = tick_clock.clocks()
    boundary_ticks = (0 if model.root.clock is None
                      else int(model.root.clock.ticks))
    for node in model.walk_parent_first():
        gid = int(node.cfg.grid_id)
        old = node.clock
        new = fresh[gid]
        ticks = boundary_ticks if old is None else int(old.ticks)  # noqa: E501
        new.ticks = ticks
        new.step_count = max(
            0, (ticks - new.spec.start_ticks) // new.spec.step_ticks)
        if old is not None and getattr(old, "dtbc_fp32", None) is not None:
            new.dtbc_fp32 = old.dtbc_fp32
        node.clock = new
        if old is None:
            refresh_model_time(node.state, new)
        from woof.ensemble.runtime_context import bind_reconstructed_member_node
        bind_reconstructed_member_node(node)
    model.schedule = schedule


def _attach_spawned_children(model, active_exp, record, writers,
                             preparer, lbc_interval_s,
                             coupler_factory=None) -> list[int]:
    """Give each newborn its node, clock, coupler, prepared case, writer.

    The clock is minted over the ACTIVATED domain set and carried to the
    birth tick.  ``_retarget_tree_schedule`` re-mints every clock on the
    next leg anyway, but a DomainNode is not constructible without one,
    and the newborn must read its own birth instant from the moment it
    exists rather than from the next boundary.
    """
    from types import MappingProxyType

    from woof.core.clock import resolve_clock
    from woof.core.model import DomainNode
    from woof.core.state import refresh_model_time

    if coupler_factory is None:
        from woof.core.nest import NestCoupler as coupler_factory

    boundary_ticks = int(model.root.clock.ticks)
    schedule_now = getattr(model, "schedule", None)
    fresh = resolve_clock(
        active_exp, lbc_interval_s=lbc_interval_s,
        tick_den_floor=(None if schedule_now is None
                        else int(schedule_now.clock.tick_den))).clocks()
    nodes = dict(model.nodes_by_grid_id)
    attached: list[int] = []
    for gid, child_result in sorted(record["child_results"].items()):
        gid = int(gid)
        child_dc = active_exp.domain(gid)
        parent = nodes[int(child_dc.parent_id)]
        clock = fresh[gid]
        clock.ticks = boundary_ticks
        clock.step_count = max(
            0, (boundary_ticks - clock.spec.start_ticks)
            // clock.spec.step_ticks)
        node = DomainNode(
            cfg=child_dc, grid=child_result.grid, state=child_result.state,
            clock=clock, parent=parent, children=[], coupler=None)
        node.coupler = coupler_factory(node, feedback=active_exp.feedback)
        node._started = True
        # The birth instant, so a follower does not move a nest in the same
        # boundary that spawned it (relocation_runner.on_period_begin).
        node._born_at_ticks = boundary_ticks
        node.state._nest_restart_classification = "REBUILT"
        refresh_model_time(node.state, clock)
        parent.children.append(node)
        nodes[gid] = node
        # The writers and any prepared-case consumer read the tree, so it
        # has to carry the newborn before they are touched.
        model.nodes_by_grid_id = MappingProxyType(nodes)
        prepared = preparer.prepared_by_grid_id[gid]
        model._prepared_by_grid_id[gid] = prepared
        from woof.ensemble.runtime_context import bind_reconstructed_member_node
        bind_reconstructed_member_node(node, prepared_case=prepared)
        if writers is not None:
            episodes = record.get("episode_by_grid_id", {})
            episode = int(episodes.get(str(gid), episodes.get(gid, 0)))
            # GATED on the DECLARED tables, not on the firing count.  A
            # one-shot spawn keeps the flat pathname it has always
            # written; only a declared lifecycle gets episode-numbered
            # directories.  active_exp carries the declaration through
            # _dc_replace, so the fired tree is a legal source for it.
            writers.add_domain(gid, grid=node.grid,
                               static_fields=prepared.static_fields,
                               episode=output_episode(
                                   active_exp.domain(gid), episode))
        attached.append(gid)
    model.nodes_by_grid_id = MappingProxyType(nodes)
    from woof.core import uh_diag
    declared = getattr(model, "_declared_experiment", None) or active_exp
    uh_diag.allocate_declared_follower_windows(declared, model)
    for gid in attached:
        node = model.node(gid)
        if node.cfg.follow is not None:
            # A reserved slot accumulated before birth. The newly live
            # consumer starts its own episode, never inherits that history.
            uh_diag.reset_tracker_window(node.parent.state, uh_diag.follow_window_slot(gid))
    return attached


def restore_nest_lifecycle(model, exp: ExperimentConfig, peek, *,
                           spawn_runner, lbc_interval_s=None,
                           coupler_factory=None) -> dict[int, int]:
    """Rebuild the tree the checkpoint describes, BEFORE its state lands.

    A tree checkpoint's member set is the tree that was live when it was
    written, so a resume that restores into the PRE-SPAWN tree refuses on
    a partial domain set -- and one that restores into a tree it guessed
    at restores the wrong arrays into the wrong nests.  This puts the
    tree back through the run's own seams and nothing else:

    * the runner is seeded from the block, so ``active`` presents exactly
      the domain set the checkpoint carries;
    * each live episode is materialized at its FIRED placement, because
      that is the placement ``validate_spawn_placement`` adjudicated when
      the slot fired, and a nest that has since been MOVED is brought to
      its current placement under the relocation rule instead (WP4c);
    * the newborn joins through ``_attach_spawned_children``, the same
      seam a leg boundary uses, one at a time and parent-first so a
      nested episode finds its parent already in the tree;
    * the schedule is re-aimed over the restored domain set, because the
      tick denominator is the LCM of the LIVE set's timesteps: a
      root-only schedule reads the checkpoint's tick pair as a mismatch.

    Returns ``{grid_id: output episode}`` for the writer set, so a domain
    resumed mid-episode-2 writes ``d0N/episode-002/`` from its first
    frame rather than from its next birth.
    """
    from woof.core.nest_lifecycle import output_episode
    from woof.ingest.nest_spawn_init import spawn_child_from_parent

    block = getattr(peek, "block", None)
    if block is None or spawn_runner is None or block.get("spawn") is None:
        return {}
    spawn_runner.restore_state(block["spawn"])
    active = spawn_runner.active
    episodes: dict[int, int] = {}
    # Ascending grid id is parent-before-child (the experiment loader
    # enforces that order), which is what lets a nested episode's parent
    # already be in the tree when the child is materialized from it.
    for gid in sorted(spawn_runner.spawned):
        child_dc = active.domain(gid)
        parent_node = model.node(int(child_dc.parent_id))
        statics = (None if spawn_runner.statics_provider is None
                   else spawn_runner.statics_provider(child_dc, parent_node))
        receipt = spawn_child_from_parent(
            child_dc, parent_node,
            static_fields=statics,
            blend_width=int(getattr(exp, "blend_width", 5)),
            scratch_arena=getattr(model, "_scratch_arena", None),
            dycore_state_workspace=getattr(
                model, "_dycore_state_workspace", None),
            array_module=spawn_runner.array_module,
            on_child_built=spawn_runner.on_child_built,
            vertical=getattr(exp, "vertical", None))
        # The live object the NEXT leg boundary adopts.  Without it the
        # runner's ``refresh_from_model`` has nothing to carry forward
        # and the resumed episode is invisible to its own retire watch.
        spawn_runner._child_results[gid] = receipt["child_result"]
        episode = int(spawn_runner.episodes.get(gid, 0))
        _attach_spawned_children(
            model, active,
            {"child_results": {gid: receipt["child_result"]},
             "episode_by_grid_id": {str(gid): episode}},
            None, spawn_runner.on_child_built, lbc_interval_s,
            coupler_factory)
        episodes[gid] = output_episode(child_dc, episode)
    if episodes:
        _retarget_tree_schedule(
            model, active, float(exp.run_seconds), lbc_interval_s)
    return episodes


def remark_relocation_fingerprint(model, peek) -> str:
    """Reproduce the live fingerprint of the run that wrote this set.

    A run that has relocated a nest chains each move's record digest into
    ``model.experiment_fingerprint`` (``mark_fingerprint_across_move``),
    so its later checkpoints are keyed to the move history and a FRESH
    build refuses them by construction -- which is the whole point, and
    also the reason a legitimate resume cannot get in without doing the
    same arithmetic.  Folding the header's record chain over the fresh
    build reproduces exactly that value: same base, same history, same
    mark.  Nothing else can, because the digests are the moves' own.

    The named components are re-marked the same way, for two reasons: a
    mismatch must still be able to say WHICH component moved, and the
    resumed run's OWN next move has to chain onto this history rather
    than onto the base -- without which the third segment of a moving
    run is addressed by a history that did not happen.
    """
    from woof.core.nest_relocation import mark_fingerprint_across_move

    records = tuple(getattr(peek, "relocation_records", ()) or ())
    if not records:
        return model.experiment_fingerprint
    marked = model.experiment_fingerprint
    for record_sha in records:
        marked = mark_fingerprint_across_move(marked, record_sha)
    model.experiment_fingerprint = marked
    restore_relocation_fingerprint_components(model, records)
    return marked


def restore_relocation_fingerprint_components(model, records) -> None:
    """Carry an already validated move chain into the named audit components.

    This does not change the scalar fingerprint or validate a checkpoint.
    Callers restore the canonical ordered chain after reconstructing identity;
    later moves append to that history instead of starting another list.
    """
    components = getattr(model, "_experiment_fingerprint_components", None)
    if components is not None:
        updated = dict(components)
        updated["relocation"] = {"records": list(records)}
        model._experiment_fingerprint_components = updated


def _relocate_restored_child(model, node, runner, placement) -> None:
    """Bring one restored episode to its post-move placement, in ONE hop.

    The nest was materialized at its FIRED placement, because that is
    what ``validate_spawn_placement`` adjudicated when the slot fired.
    Every move it made afterwards is one hop from there under the
    RELOCATION rule -- the follower's own admissible band and overlap
    floor -- so the placement is re-admitted by the rule that produced it
    rather than waved through.

    One hop, not N: replaying each historical move would rebuild the
    child N times for a state that is about to be overwritten wholesale,
    and would leave the segment chain at a generation the checkpoint did
    not record.  The persisted segment is restored over the hop's own
    afterwards, which is what keeps the NEXT move chaining onto the real
    predecessor.
    """
    from woof.core.nest_relocation import base_segment, relocate_child

    capture = getattr(runner.on_child_built, "capture_outgoing", None)
    if callable(capture):
        capture(node)
    relocate_child(
        node,
        i_parent_start=int(placement[0]), j_parent_start=int(placement[1]),
        segment=base_segment(node.cfg), bounds=runner.config,
        initializer=runner.initializer,
        static_provenance=runner.static_provenance,
        on_child_built=runner.on_child_built,
        scratch_arena=getattr(model, "_scratch_arena", None),
        dycore_state_workspace=getattr(
            model, "_dycore_state_workspace", None),
        staging=runner.staging)
    from woof.core.state import refresh_model_time

    refresh_model_time(node.state, node.clock)
    after_move = getattr(runner.on_child_built, "after_move", None)
    if callable(after_move):
        after_move(node)


def restore_nest_followers(model, peek, *, spawn_runner=None) -> list[int]:
    """Seed every live follower from its own checkpoint entry.

    The entry is taken WHOLE -- the runner's four keys and the writer's
    three -- because the segment chain is the part nothing can recompute:
    a follower that resumes at generation zero chains its next move's
    record onto the base preparation instead of onto its real
    predecessor, and every later checkpoint of the resumed run is then
    addressed by a history that did not happen.

    A follower this run builds that the block says nothing about is built
    FRESH and not refused: its target was still dormant when the
    checkpoint was taken, so nothing had been consulted and nothing had
    moved, which is exactly what a never-consulted runner holds.  The
    reverse -- an entry naming a follower this run does not build -- is
    refused, because the placement history it carries would be silently
    dropped and the nest would move again at the first cadence boundary.

    A restored episode whose CURRENT placement differs from its FIRED one
    is brought there in one hop first, and the persisted segment then
    lands over the hop's own.  ``spawn_runner`` supplies that split: the
    runner holds the fired placements and reports the current ones off
    its own restore.
    """
    from woof.io.restart import lifecycle_followers

    entries = peek.followers
    moved = {}
    if spawn_runner is not None:
        moved = {gid: place for gid, place
                 in getattr(spawn_runner, "restored_current_placements",
                            {}).items()
                 if tuple(place) != tuple(spawn_runner.spawned.get(gid, place))}
    if not entries:
        if moved:
            named = ", ".join(f"d{gid:02d}" for gid in sorted(moved))
            raise RuntimeError(
                f"this checkpoint says {named} sits away from the placement "
                "it fired at, but carries no follower history for it: only "
                "a follower moves a spawned nest, so a move with no "
                "follower entry means the block's two halves disagree and "
                "there is no admissible band to re-adjudicate the "
                "placement under")
        return []
    runners = lifecycle_followers(model)
    restored: list[int] = []
    for raw in sorted(entries, key=int):
        gid = int(raw)
        runner = runners.get(gid)
        if runner is None:
            raise RuntimeError(
                f"this checkpoint carries a follower history for "
                f"d{gid:02d} ({entries[raw].get('kind')}), but this run "
                "builds no follower for that domain: no per-domain "
                "[follow] table and no tree-level [relocation] naming it.  "
                "The segment chain, the executed-move count and the two "
                "cooldown anchors would be dropped, so the nest would sit "
                "at a placement the config cannot explain and would be "
                "free to move again at the first cadence boundary")
        node = model.nodes_by_grid_id.get(gid)
        placement = moved.pop(gid, None)
        if node is not None and placement is not None:
            _relocate_restored_child(model, node, runner, placement)
        # AFTER the hop: the hop's own segment counts one move off the
        # base preparation, and what the next move must chain onto is the
        # generation the checkpoint recorded.
        runner.restore_state(entries[raw])
        restored.append(gid)
    if moved:
        named = ", ".join(f"d{gid:02d}" for gid in sorted(moved))
        raise RuntimeError(
            f"this checkpoint says {named} moved off its fired placement "
            "but names no follower for it; the nest cannot be re-admitted "
            "at that placement under any band this run declares")
    return restored


def _detach_retired_children(model, grid_ids, writers, steppers) -> list[int]:
    """Detach a retired subtree at a completed leg boundary, deepest first.

    No schedule op is skipped: the schedule that referenced these nodes has
    already completed.  The next loop iteration rebuilds its op table from
    ``spawn_runner.active`` before integration resumes.
    """
    from types import MappingProxyType

    requested = {int(g) for g in grid_ids}
    if not requested:
        return []
    nodes = dict(model.nodes_by_grid_id)
    # Include any live descendants even when the policy record named only the
    # spawned root of a mixed static/spawn subtree.
    changed = True
    while changed:
        changed = False
        for gid, node in list(nodes.items()):
            if node.parent is not None and int(node.parent.cfg.grid_id) in requested and gid not in requested:
                requested.add(int(gid)); changed = True

    def depth(gid):
        d = 0; node = nodes.get(gid)
        while node is not None and node.parent is not None:
            d += 1; node = node.parent
        return d

    detached = []
    for gid in sorted(requested, key=depth, reverse=True):
        node = nodes.get(gid)
        if node is None or node.parent is None:
            continue
        if writers is not None:
            writers.remove_domain(gid)
        if steppers is not None:
            stepper = steppers.pop(gid, None)
            close = getattr(stepper, "close", None)
            if callable(close):
                close()
        node._started = False
        node.parent.children[:] = [c for c in node.parent.children
                                   if int(c.cfg.grid_id) != gid]
        nodes.pop(gid, None)
        model._prepared_by_grid_id.pop(gid, None)
        detached.append(gid)
    model.nodes_by_grid_id = MappingProxyType(nodes)
    return detached


def _exchange_consumer_planes(model, steppers, direction: str,
                              names) -> list[int]:
    """Move ONE consumer's whole-domain planes between store and state.

    A whole-domain model consumer -- the spawn trigger
    (:class:`woof.core.nest_spawn.SpawnWatch`) and the follow tracker
    (:class:`woof.core.storm_tracking.StormTracker`) -- reads ONE plane off
    ``parent_state`` through ``storm_tracking.signal_plane``, which is
    ``state.existing_scratch(slot)``.  A streamed domain's arrays live in its
    store and its ``DomainState`` stops changing at attach, so that read
    returns the plane the state was allocated with: for the UH windows, zeros
    (state.py:720), for the whole run, with no error.

    Both directions are needed and they are not symmetric bookkeeping:
    ``publish`` is how the consumer sees the domain, ``adopt`` is how the
    domain sees that the consumer zeroed its window.  Cheap by construction
    -- one ``(ny, nx)`` plane per streamed domain per LEG boundary, not per
    step, and nothing at all when no domain streams.

    ``names`` is THIS consumer's slots and no one else's, on the same
    reasoning that gave the two consumers separate windows in the first place
    (WOOF's ruling, 2026-08-07): the relocation runner resets the follow
    window on its own cadence, from inside ``execute_experiment``, and a
    spawn boundary that published or adopted that window as well could undo a
    reset the tracker had already made or hand it a window measured against a
    boundary it does not own.
    """
    if not steppers:
        return []
    from woof.core import streaming as _streaming

    names = tuple(names)
    touched: list[int] = []
    for gid, stepper in sorted(steppers.items()):
        if not _streaming.is_streaming(stepper):
            continue
        node = model.nodes_by_grid_id.get(int(gid))
        if node is None:
            continue
        # Asked of the streaming module rather than looked up here: this
        # file is not a sanctioned scratch-API site and
        # tests/test_uh_lifecycle.py's roster is right to say so.  The
        # duck-typing (a reduced state, a test double, nwp_diagnostics = 0)
        # is inside allocated_planes.
        present = _streaming.allocated_planes(node.state, names)
        if not present:
            # nwp_diagnostics = 0 allocates no window at all, so there is no
            # plane for any consumer to read and nothing to move.
            continue
        getattr(stepper, direction)(present)
        touched.append(int(gid))
    return touched


def _spawn_consumer_planes() -> tuple[str, ...]:
    """The spawn trigger's own slot, as a streaming manifest key."""
    from woof.core.uh_diag import UH_SPAWN_WINDOW_SLOT

    return (f"scratch/{UH_SPAWN_WINDOW_SLOT}",)


def _publish_consumer_planes(model, steppers, names) -> list[int]:
    return _exchange_consumer_planes(model, steppers, "publish", names)


def _adopt_consumer_planes(model, steppers, names) -> list[int]:
    return _exchange_consumer_planes(model, steppers, "adopt", names)


def _adjudicate_newborn_steppers(steppers, model, attached, factory):
    """Bind (or refuse) a stepper for every domain born this boundary.

    ``woof.core.streaming.steppers_for_tree`` walks the tree ONCE, before
    the run, and returns ``{grid_id: stepper}``; the executor resolves a
    missing grid to ``dycore.step`` (model.py's "delayed-start child" note,
    written before streaming existed).  For a DELAYED-START child that is
    right: the domain was in the tree when the mapping was built and was
    adjudicated then, it simply had not started yet.  For a SPAWNED child it
    is not: that domain was not in the tree at all when the mapping was
    built, so nothing ever asked whether it should stream.

    The two failure modes of the silent fallback are opposite and both bad.
    A big newborn that should have streamed dies at the resident allocation
    the mode was turned on to avoid -- after hours of integration, at the
    one instant the run cannot be restarted from.  A small newborn that
    should not have streamed runs correctly, which is worse, because the
    run then certifies a spawn path that has never once been exercised
    under the mode its config says it is in.

    So: with streaming engaged (a non-empty mapping is the only thing that
    says so -- ``steppers_for_tree`` returns ``{}`` when ``[tiles]`` is
    absent AND when ``auto`` decides every domain fits), a newborn either
    gets an adjudicated stepper from the route's factory, or the run
    refuses HERE, naming the grid.  Never a silent fallthrough.
    """
    if not steppers or not attached:
        return steppers
    if factory is None:
        named = ", ".join(f"d{int(gid):02d}" for gid in sorted(attached))
        raise RuntimeError(
            f"[tiles] is engaged for this run ({len(steppers)} domain(s) "
            f"stream) and {named} was born at a spawn boundary, after the "
            "stepper mapping was built.  woof.core.streaming"
            ".steppers_for_tree walks the tree once, before the run, so a "
            "domain that joins it later is not in the mapping and the "
            "executor would resolve it to woof.core.dycore.step -- "
            "integrating a newborn RESIDENT inside a streamed run, with "
            "nothing in the log to say so.  This route must pass "
            "spawned_stepper_factory (grid_id, node) -> stepper | None so a "
            "newborn is adjudicated the same way its siblings were.")
    out = dict(steppers)
    for gid in sorted(int(g) for g in attached):
        stepper = factory(gid, model.node(gid))
        if stepper is not None:
            out[gid] = stepper
    return out


def _leg_boundary_pass(model, spawn_runner, *, elapsed, writers,
                       lbc_interval_s, execute_kwargs, relocation_runner,
                       relocation_runner_factory, coupler_factory,
                       spawned_stepper_factory, leg):
    """One complete leg boundary: publish, evaluate, adopt, detach, attach.

    Hoisted out of the walk's loop so a RESUME can perform exactly this
    and nothing else.  Checkpoints are written at PERIOD_BEGIN, which is
    always BEFORE the boundary evaluation, so a checkpoint taken on the
    leg lattice captures the state the straight run was in when it
    reached this call -- and running it once at resume entry reproduces
    that run's decisions from bit-identical inputs.  Returns the (possibly
    rebound) relocation runner; ``execute_kwargs`` is mutated in place for
    the stepper mapping, as the loop did.
    """
    # The boundary instant belongs to BOTH legs: the leg that just
    # ended emitted its history there, and the next leg's pre-loop
    # emit would publish the same instant again -- which for a
    # microphysics domain also means consuming a one-frame REFL
    # handoff that no longer exists.  This is the resume-boundary
    # ownership problem the checkpoint path already solves, so it is
    # solved the same way: mark exactly the domains whose frame is
    # already durable, and the next leg suppresses them one domain at
    # a time.
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(
        gid for gid, node in model.nodes_by_grid_id.items()
        if node.clock.history_due())
    # The trigger reads the parent's WHOLE-DOMAIN plane off
    # ``node.state``; a streamed parent's arrays live in its store and
    # its state stopped changing at attach.  Publishing here -- once per
    # leg boundary, only the planes a consumer reads -- is what makes the
    # watch see the running domain instead of the attach-time zeros.
    planes = _spawn_consumer_planes()
    _publish_consumer_planes(model, execute_kwargs.get("steppers"), planes)
    record = spawn_runner.on_leg_boundary(model, t=elapsed)
    # The runner zeroed each parent's window on the STATE (its
    # "max since I last looked" reset).  The domain is the store, so the
    # zeroing has to reach it or the next fold accumulates on top of a
    # window the consumer already believes it spent.
    _adopt_consumer_planes(model, execute_kwargs.get("steppers"), planes)
    if record is not None:
        retired_ids = record.get("retired_grid_ids", ())
        if retired_ids:
            _detach_retired_children(
                model, retired_ids, writers, execute_kwargs.get("steppers"))
            # The resume-boundary marker was taken across the WHOLE
            # tree a moment ago, while the retiring subtree was still
            # in it.  Its purpose is to tell the NEXT leg which
            # domains already published this instant so it does not
            # publish twice -- and a retired domain has no next leg
            # to be told anything.  Left in, the marker names a
            # grid_id the next schedule has never heard of and the
            # executor refuses the leg by name ("committed initial
            # history domains are not in the schedule"), which is a
            # retirement that killed the run one boundary after the
            # episode it ended.
            model._resume_committed_history_grid_ids = frozenset(
                gid for gid in model._resume_committed_history_grid_ids
                if gid in model.nodes_by_grid_id)
        attached = []
        if record.get("child_results"):
            attached = _attach_spawned_children(
                model, record["experiment"], record,
                writers, spawn_runner.on_child_built, lbc_interval_s,
                coupler_factory)
            execute_kwargs["steppers"] = _adjudicate_newborn_steppers(
                execute_kwargs.get("steppers"), model, attached,
                spawned_stepper_factory)
        # A follow target that was dormant could not be wired at
        # build time; now that it exists, it can follow its storm.
        if relocation_runner_factory is not None:
            candidate = relocation_runner_factory()
            if relocation_runner is None:
                relocation_runner = candidate
            elif getattr(relocation_runner, "is_collection", False):
                relocation_runner.merge_from(candidate)
            elif candidate is not None and getattr(candidate, "is_collection", False):
                candidate.merge_from(relocation_runner)
                relocation_runner = candidate
            if relocation_runner is not None and writers is not None:
                attach_many = getattr(relocation_runner, "attach_writers", None)
                if callable(attach_many):
                    attach_many(writers)
                else:
                    attach = getattr(relocation_runner.on_child_built,
                                     "attach_writers", None)
                    if callable(attach):
                        attach(writers)
            # The binding the next leg's checkpoints must see.
            publish_lifecycle_runners(
                model, spawn_runner=spawn_runner,
                relocation_runner=relocation_runner, leg_seconds=leg)
    return relocation_runner


def resume_boundary_due(model, spawn_runner, *, leg: float, total: float,
                        tol: float = 1.0e-9) -> bool:
    """Does this resume land ON a leg boundary the run has yet to evaluate?

    Checkpoints are written at PERIOD_BEGIN, always BEFORE the boundary
    evaluation.  A checkpoint taken exactly on the leg lattice therefore
    captures the state the straight run held when it reached
    :func:`_leg_boundary_pass` -- and a resume that skips straight into
    the next leg skips that evaluation FOREVER: the nest that would have
    been born there is never born, the episode that would have retired
    never retires, and nothing anywhere says so.

    Four conditions, each of which makes the replay wrong if dropped:
    this must be a resume (a fresh run at t = 0 sits on the lattice too,
    and the straight run does not evaluate there); the lattice must
    actually be hit; there must be run left, because the last boundary of
    a completed run was already taken; and the runner must still have
    boundaries that can change the tree.
    """
    if not bool(getattr(model, "_resumed", False)):
        return False
    if not spawn_runner.needs_boundaries:
        return False
    elapsed = float(model.root.clock.elapsed_seconds)
    if elapsed <= tol or elapsed + tol >= float(total):
        return False
    leg = float(leg)
    if not math.isfinite(leg) or leg <= 0.0:
        return False
    offset = elapsed - round(elapsed / leg) * leg
    return abs(offset) <= tol * max(1.0, abs(elapsed))


def spawn_leg_boundary(spawn_runner, elapsed, *, leg, total,
                       relocation_cadence_s=None, tol=1.0e-9):
    """Where the next leg ends: the next DECISION POINT, not the next tick.

    Every boundary costs one full schedule rebuild
    (:func:`_retarget_tree_schedule`), so a boundary is only worth taking
    where the tree could actually change.  Three shapes, in order of how
    much they save:

    * nothing pending -- run straight to the end, which is what the walk
      has always done once every watch has fired;
    * a KNOWN next instant (a cooldown that has not elapsed, a window
      that has not opened, a manual trigger's ``at_s``, a minimum
      lifetime that has not run out) -- run to it.  This is the case a
      re-armable slot used to defeat entirely: it kept
      ``needs_boundaries`` true for the whole run, so a 384 h forecast
      rebuilt its schedule ~4,600 times to keep re-reading one clock;
    * an unknowable instant -- something reads the live field, or any
      trigger in the tree reads a consumer-owned window this walk would
      stop zeroing (see :meth:`~woof.core.spawn_runner.SpawnRunner
      .next_decision_time`) -- take the next tick of the leg lattice,
      unchanged.

    The returned boundary always lands ON the leg lattice: a leg length
    is validated as a whole number of root steps and the schedule builder
    needs one, so a horizon between two ticks rounds UP to the next.

    ``relocation_cadence_s`` pins the boundary to a live follower's own
    cadence.  The leg length already IS that cadence when the config sets
    one (:func:`_spawn_leg_seconds`), so for a run with a mounted
    relocation runner this preserves today's rhythm exactly rather than
    changing two things at once.
    """
    elapsed, total, leg = float(elapsed), float(total), float(leg)
    if elapsed + tol >= total:
        return total
    if not spawn_runner.needs_boundaries:
        return total
    if not math.isfinite(leg) or leg <= 0.0:
        return total
    lattice = min(total, (math.floor(elapsed / leg) + 1) * leg)
    horizon = spawn_runner.next_decision_time(elapsed)
    if horizon is None:
        boundary = lattice
    else:
        snapped = min(
            total, math.ceil((float(horizon) - tol) / leg) * leg)
        boundary = max(lattice, snapped)
    if relocation_cadence_s:
        cadence = float(relocation_cadence_s)
        if math.isfinite(cadence) and cadence > 0.0:
            boundary = min(
                boundary,
                min(total, (math.floor(elapsed / cadence) + 1) * cadence))
    return boundary


def _accumulate_leg_report(total, leg):
    """Sum one leg's executor counters into the run's (the walk's report)."""
    if total is None or leg is None or not hasattr(leg, "steps"):
        return leg if leg is not None else total
    for name in ("steps", "forces", "feedback_calls", "restarts", "lbc_resets"):
        if hasattr(leg, name) and hasattr(total, name):
            setattr(leg, name, int(getattr(total, name)) + int(getattr(leg, name)))
    histories = getattr(leg, "histories", None)
    if isinstance(histories, dict):
        merged = dict(getattr(total, "histories", {}) or {})
        for gid, count in histories.items():
            merged[gid] = int(merged.get(gid, 0)) + int(count)
        leg.histories = merged
    return leg


def walk_spawn_legs(model, exp: ExperimentConfig, data: CaseDataConfig, *,
                    spawn_runner, writers, lbc_interval_s,
                    relocation_runner=None, relocation_runner_factory=None,
                    coupler_factory=None, spawned_stepper_factory=None,
                    **execute_kwargs):
    """Integrate the run as LEGS so dormant nests can be born mid-run.

    The production consumer of
    :meth:`woof.core.spawn_runner.SpawnRunner.on_leg_boundary`.  While
    a watch is still pending the walk stops every
    :func:`_spawn_leg_seconds`, asks, and either continues or activates;
    once nothing is pending it runs straight to the end in one leg, so
    the common shape is "a few cheap root-only legs, then the whole rest
    of the forecast".

    Restart across a spawn boundary inherits the moving-nest posture
    whole, and that posture is no longer "promises nothing": a checkpoint
    written mid-episode carries the lifecycle block, and
    :func:`restore_nest_lifecycle` rebuilds the tree the checkpoint
    describes -- runner seeded, every live episode materialized at its
    FIRED placement, schedule re-aimed over the restored domain set --
    before any state lands.

    ``spawned_stepper_factory(grid_id, node) -> stepper | None`` adjudicates
    a NEWBORN's execution mode.  It is consulted only when this run is
    actually streaming something, and its absence in that case is a refusal
    rather than a fallthrough -- see :func:`_adjudicate_newborn_steppers`
    for the two ways the silent version goes wrong.  Every parent's stepper
    is left alone: a :class:`~woof.core.streaming.StreamedDomain` owns the
    domain's arrays in its store and its tile buffers, and re-attaching one
    at a leg boundary would re-copy the ATTACH-TIME state over the store and
    silently discard every step the run has taken.
    """
    from woof.core.model import execute_experiment

    leg = _spawn_leg_seconds(exp)
    total = float(exp.run_seconds)
    tol = 1.0e-9
    publish_lifecycle_runners(
        model, spawn_runner=spawn_runner,
        relocation_runner=relocation_runner, leg_seconds=leg)
    # THE RESUME BOUNDARY.  A checkpoint taken on the leg lattice was
    # taken before that boundary was evaluated, so the evaluation is
    # still owed; performing it here, once, from the restored state
    # reproduces the straight run's decisions from the same inputs.
    if resume_boundary_due(model, spawn_runner, leg=leg, total=total,
                           tol=tol):
        relocation_runner = _leg_boundary_pass(
            model, spawn_runner,
            elapsed=float(model.root.clock.elapsed_seconds),
            writers=writers, lbc_interval_s=lbc_interval_s,
            execute_kwargs=execute_kwargs,
            relocation_runner=relocation_runner,
            relocation_runner_factory=relocation_runner_factory,
            coupler_factory=coupler_factory,
            spawned_stepper_factory=spawned_stepper_factory, leg=leg)
    result = None
    while True:
        elapsed = float(model.root.clock.elapsed_seconds)
        boundary = spawn_leg_boundary(
            spawn_runner, elapsed, leg=leg, total=total, tol=tol,
            # A mounted relocation runner keeps the leg on its own
            # cadence: the follower is consulted inside
            # execute_experiment, but this lane changes the leg's LENGTH
            # and a live follower is not the place to prove that.
            relocation_cadence_s=(leg if relocation_runner is not None
                                  else None))
        _retarget_tree_schedule(
            model, spawn_runner.active, boundary, lbc_interval_s)
        # The relocation runner watches ONE grid; while that grid is
        # still dormant it is not in the tree, and consulting it would
        # ask the model for a node that does not exist.
        runner = relocation_runner
        if runner is not None and getattr(runner, "is_collection", False):
            runner.drop_absent(model)
            if not runner.target_grid_ids:
                runner = None
        elif runner is not None:
            target = getattr(runner.config, "grid_id", None)
            if target is not None and int(target) not in model.nodes_by_grid_id:
                runner = None
        leg_report = execute_experiment(
            model, relocation_runner=runner, experiment=exp,
            **execute_kwargs)
        result = _accumulate_leg_report(result, leg_report)
        elapsed = float(model.root.clock.elapsed_seconds)
        if elapsed + tol >= total:
            break
        relocation_runner = _leg_boundary_pass(
            model, spawn_runner, elapsed=elapsed, writers=writers,
            lbc_interval_s=lbc_interval_s, execute_kwargs=execute_kwargs,
            relocation_runner=relocation_runner,
            relocation_runner_factory=relocation_runner_factory,
            coupler_factory=coupler_factory,
            spawned_stepper_factory=spawned_stepper_factory, leg=leg)
    spawn_runner.close_receipt(model)
    # The last leg's execution report: a route that receipts the executor's
    # report (the prepared tree) reads the run's final period from it.
    return result


# ---------------------------------------------------------------------------
# Run schedule (whole-step counts) and output calendars
# ---------------------------------------------------------------------------

def whole_step_count(duration: float, dt: float, name: str) -> int:
    """Return an exact whole-step count or reject an ambiguous schedule."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {duration}")
    steps = int(round(duration / dt))
    tolerance = max(1.0e-9, abs(duration) * 1.0e-12)
    if steps < 1 or not np.isclose(
            steps * dt, duration, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"{name}={duration} must be an integer multiple of dt={dt}")
    return steps


def configured_run_schedule(
        cfg: RunConfig, *, run_seconds: float | None = None,
        output_interval_s: float | None = None) -> tuple[int, int]:
    """Validate and return ``(outer_steps, output_outer_steps)``.

    The run-length ceiling is NOT here: it derives from validated
    forcing coverage (:func:`forcing_schedule`) on the experiment path,
    and the frozen case profile applies its own pinned ceiling before
    delegating.
    """
    if not np.isfinite(cfg.dt) or cfg.dt <= 0.0:
        raise ValueError(f"dt must be finite and > 0, got {cfg.dt}")
    run_seconds = cfg.run_seconds if run_seconds is None else run_seconds
    output_interval_s = (cfg.output_interval_s
                         if output_interval_s is None
                         else output_interval_s)
    return (
        whole_step_count(run_seconds, cfg.dt, "run_seconds"),
        whole_step_count(
            output_interval_s, cfg.dt, "output_interval_s"),
    )


def restart_outer_steps(
        cfg: RunConfig, *, restart_interval_s: float | None = None
        ) -> int | None:
    """Restart-write cadence in outer steps; ``None`` when disabled."""
    restart_interval_s = (cfg.restart_interval_s
                          if restart_interval_s is None
                          else restart_interval_s)
    if restart_interval_s <= 0.0:
        return None
    return whole_step_count(
        restart_interval_s, cfg.dt, "restart_interval_s")


def history_output_due(outer_step: int, output_outer_steps: int, *,
                       final_outer_step: int | None = None,
                       history_begin_outer_step: int = 0,
                       history_end_outer_step: int | None = None,
                       history_interval2_steps: int | None = None,
                       history_interval_change_step: int = 0) -> bool:
    """Keep the cadence and optionally publish the completed terminal state."""
    completed = outer_step + 1
    if (completed < history_begin_outer_step
            or (history_end_outer_step is not None
                and completed > history_end_outer_step)):
        return False
    if history_interval2_steps is not None and completed > history_interval_change_step:
        return ((completed - history_interval_change_step) % history_interval2_steps == 0
                or completed == final_outer_step)
    return ((completed - history_begin_outer_step) % output_outer_steps == 0
            or completed == final_outer_step)


def refl_10cm_due(outer_step: int, substep: int,
                  output_outer_steps: int,
                  dynamics_substeps: int, *,
                  final_outer_step: int | None = None,
                  history_begin_outer_step: int = 0,
                  history_end_outer_step: int | None = None,
                  history_interval2_steps: int | None = None,
                  history_interval_change_step: int = 0) -> bool:
    """True only for the microphysics call immediately before an output.

    The final internal step owns the outer-step history frame.  Keeping this
    as a pure predicate makes the calendar wiring CPU-testable even if a
    future configuration restores more than one dynamics substep.
    """
    return (history_output_due(outer_step, output_outer_steps,
                               final_outer_step=final_outer_step,
                               history_begin_outer_step=history_begin_outer_step,
                               history_end_outer_step=history_end_outer_step,
                               history_interval2_steps=history_interval2_steps,
                               history_interval_change_step=history_interval_change_step)
            and substep + 1 == dynamics_substeps)


def single_history_window_steps(cfg, *, history_begin_s=0.0,
                                history_end_s=None,
                                grid_id: int | None = None
                                ) -> tuple[int, int | None]:
    """Resolve the frozen loop's output window on the shared domain clock.

    ``grid_id`` only labels a refusal of a window that is not a whole
    number of ticks; the caller that knows which domain it writes passes it.
    """
    from fractions import Fraction
    from types import SimpleNamespace
    from woof.core.clock import _history_window_ticks

    step = Fraction(str(cfg.dt))
    begin, end = _history_window_ticks(
        SimpleNamespace(grid_id=cfg.grid_id if grid_id is None else grid_id,
                        history_begin_s=history_begin_s,
                        history_end_s=history_end_s),
        step.numerator, step.denominator)
    return begin // step.numerator, None if end is None else end // step.numerator


# ---------------------------------------------------------------------------
# Output identity
# ---------------------------------------------------------------------------

def _global_wrf_attrs(
        grid, start_time: datetime,
        geog_selection: GeogSelection | None = None, *, domain=None,
        coord=None, feedback=None, initial_condition=None,
        configured_dt: float | None = None,
        source: str | None = None,
        simulation_start_time: datetime | None = None,
        static_fields=None) -> dict[str, object]:
    """Assemble one domain's wrfout global attributes.

    ``static_fields`` (the domain's prepared static) states the land-use
    identity when no GEOG selection does: a 20-category MODIS static (the
    15 arc-second set, with no lake class) stamps NUM_LAND_CAT = 20 and
    ISLAKE = -1, as geogrid and wrf.exe do, instead of the default table's
    21 and 21.

    ``initial_condition`` is the preparation receipt's provenance block
    (see :func:`woof.io.wrfout.initial_condition_global_attrs`).  It
    says WHAT the initial state was; ``start_time`` says only WHEN the
    model clock began, and at a nonzero forecast lead those are two
    different facts about two different times.  ``None`` writes no
    provenance attribute at all rather than asserting an analysis.

    ``start_time`` is THIS domain's start (``START_DATE``);
    ``simulation_start_time`` is the run's (``SIMULATION_START_DATE``),
    which WRF holds identical on every domain -- see
    :func:`woof.io.wrfout.wrf_global_attrs`.  They differ only for a
    delayed-start nest, and ``None`` keeps the previous single-date
    behaviour.
    """
    from woof.io.wrfout import (
        initial_condition_global_attrs, wrf_global_attrs)

    landuse_attrs = (None if geog_selection is None
                     else geog_selection.landuse_global_attrs())
    if landuse_attrs is None and static_fields is not None:
        landusef = static_fields.get("LANDUSEF") if hasattr(
            static_fields, "get") else None
        if landusef is not None:
            from woof.native_wrf_contract import native_landuse_identity
            from woof.static.geog import DEFAULT_LANDUSE_ATTRS
            categories = int(np.shape(landusef)[0])
            # Only the MODIS set without its lake class (20 planes) is
            # re-stated; any other static keeps the attributes it always had.
            if categories == 20:
                landuse_attrs = {
                    **DEFAULT_LANDUSE_ATTRS,
                    **native_landuse_identity(static_fields),
                    "NUM_LAND_CAT": categories}
    identity = {}
    if domain is not None:
        run = getattr(domain, "run", domain)
        identity.update(
            grid_id=int(getattr(domain, "grid_id", run.grid_id)),
            parent_id=int(getattr(domain, "parent_id", 0)),
            i_parent_start=int(getattr(domain, "i_parent_start", 1)),
            j_parent_start=int(getattr(domain, "j_parent_start", 1)),
            parent_grid_ratio=int(
                getattr(domain, "parent_grid_ratio", 1)),
            # wrfout's DT is ONE number per FILE, and ITIMESTEP is derived
            # from it (io/wrfout.py:1205).  Under an adaptive clock
            # ``run.dt`` is a snapshot of a value that changes every step,
            # captured whenever these attributes are built -- at setup,
            # and again at a relocation -- so two runs that build them at
            # different instants stamp different numbers forever.
            # MEASURED: a continuous run and its own resume disagreed on
            # DT for the relocating nest while all 79 of that frame's
            # variables were byte-identical.
            #
            # The CONFIGURED step is stable for the life of the run, is
            # what the namelist asked for, and is the same in a run and in
            # its resume.  A DELIBERATE DIVERGENCE from WRF, which stamps
            # grid%dt and so carries the same ambiguity: a file-level
            # attribute cannot describe a per-frame quantity, so it should
            # describe the thing that does not vary.
            dt=float(run.dt
                     if configured_dt is None
                     or not bool(getattr(run, "use_adaptive_time_step",
                                         False))
                     else configured_dt))
    if coord is not None:
        identity.update(hybrid_opt=int(coord.hybrid_opt),
                        etac=float(coord.etac))
    # ``run`` carries the resolved physics selectors, which are what let a
    # reader tell "no shallow-cumulus scheme exists" from "one ran and
    # produced nothing".
    attrs = wrf_global_attrs(
        grid, start_time, landuse_attrs=landuse_attrs,
        run=(None if domain is None else getattr(domain, "run", domain)),
        simulation_start_time=simulation_start_time,
        **identity)
    if feedback is not None:
        attrs.update(
            GPUWM_FEEDBACK=str(feedback["feedback"]),
            GPUWM_FEEDBACK_VALUE=np.int32(feedback["feedback_value"]),
            GPUWM_FEEDBACK_STOCK_WRF_CERTIFIED=np.int32(0),
            GPUWM_FEEDBACK_CERTIFICATION=str(
                feedback["stock_wrf_certification"]))
    attrs.update(initial_condition_global_attrs(
        initial_condition, source=source))
    from woof.ensemble.runtime_context import current_capture
    capture = current_capture()
    if capture is not None and capture.provenance:
        from woof.ensemble.wofs_recipe import provenance_label
        attrs["GPUWM_ENSEMBLE_PROVENANCE"] = provenance_label(capture.provenance)
    name = grid_name_for(identity.get("grid_id", 1))
    if name:
        attrs[GRID_NAME_ATTR] = name
    return attrs


#: What a run's maps call its OUTERMOST domain in place of WRF's ``d01``
#: (``nest`` -> headline ``nest 3 km``), read from the environment by
#: :func:`grid_name_for` and stamped as the ``GPUWM_GRID_NAME`` global
#: attribute, which ``rw_wrfbatch`` reads for the headline only (the domain
#: folder and the filename keep ``d01``).
#:
#: THE BREAKAGE THIS PREVENTS (WOOF Hex with Regional, 2026-10-28): a nest
#: added to a hex run (``woof hex add-nest``) is its own one-domain run, so
#: its frames say ``GRID_ID = 1`` and every nest map was headlined
#: ``d01 3 km`` while the site files the nest as d05.  The add-nest door sets
#: ``GPUWM_GRID_NAME=nest`` for its ``woof sim``.  Only d01 takes it: in a
#: run with nests of its own, d02 and beyond keep their WRF names.
GRID_NAME_ENV = "GPUWM_GRID_NAME"
GRID_NAME_ATTR = "GPUWM_GRID_NAME"


def grid_name_for(grid_id) -> str | None:
    """The ``GPUWM_GRID_NAME`` a domain's frames carry, or ``None``.

    Whitespace is collapsed; a name the renderer would not draw (over 32
    characters, a control character or ``|``) is refused by name rather
    than written and silently ignored there.
    """

    raw = os.environ.get(GRID_NAME_ENV, "")
    name = " ".join(raw.split())
    if not name or int(grid_id) != 1:
        return None
    if len(name) > 32 or "|" in name or any(not ch.isprintable() for ch in name):
        raise ValueError(f"{GRID_NAME_ENV}={raw!r}: a grid name is at most 32 printable "
                         "characters and no '|' (rw_wrfbatch's headline label rule)")
    return name


def write_case_output(prepared, output_dir: Path, valid_time: datetime, *,
                      start_time: datetime, title: str, domain_id: int = 1,
                      expect_refl_10cm: bool = True,
                      feedback=None, history_selection=None,
                      completed_records=None) -> Path | None:
    from woof.io.wrfout import (WrfoutWriter, state_frame, fire_history_attrs,
                                 wrfout_filename)

    state = prepared.initial_result.state
    streamed = getattr(state, "_streamed_domain", None)
    from woof.ensemble.runtime_context import current_capture
    capture = current_capture()
    captured_refl = None
    analysis_refl = (not expect_refl_10cm and valid_time == start_time
                     and prepared.cfg.mp_physics in REFL_10CM_MICROPHYSICS
                     and state.qv is not None)
    if analysis_refl:
        from woof.core.refl import analysis_refl_10cm
        captured_refl = analysis_refl_10cm(
            state, prepared.cfg, shape=(prepared.cfg.nz, prepared.cfg.ny, prepared.cfg.nx),
            array_module=np if streamed is not None else None)
    if capture is not None:
        if (expect_refl_10cm
                and prepared.cfg.mp_physics in REFL_10CM_MICROPHYSICS
                and state.qv is not None):
            from woof.core.refl import consume_refl_10cm
            captured_refl = consume_refl_10cm(state)
        capture.submit(
            state=state, streamed=streamed,
            metadata=_metadata_frame(prepared.grid, prepared.static_fields),
            refl_field=captured_refl, valid_time=valid_time, grid_id=domain_id)
        if not capture.keep_member_files:
            return None
    if streamed is None:
        # Output observes the completed state without re-diagnosing it.
        frame = state_frame(state, include_diagnostic_pressure=True)
    else:
        # The same StoreFrame used by the tree writer. Its arrays remain
        # valid until the next sweep; this writer closes synchronously.
        frame = streamed.history_fields()
        if (capture is None and expect_refl_10cm
                and prepared.cfg.mp_physics in REFL_10CM_MICROPHYSICS
                and state.qv is not None):
            # The field is already in StoreFrame. Retire the domain handoff
            # once, just as the tree route does, before the next due sweep.
            from woof.core.refl import consume_refl_10cm
            consume_refl_10cm(state)
    frame.update(_metadata_frame(prepared.grid, prepared.static_fields))
    if analysis_refl:
        if isinstance(captured_refl, np.ndarray):
            frame["REFL_10CM"] = captured_refl
        else:
            import cupy as cp
            frame["REFL_10CM"] = cp.asnumpy(captured_refl)
    if streamed is None:
        import cupy as cp
        frame["RAINNC"] = cp.asnumpy(state.physics.microphysics.rainnc)
    if (streamed is None and expect_refl_10cm
            and prepared.cfg.mp_physics in REFL_10CM_MICROPHYSICS
            and state.qv is not None):
        # WRF do_radar_ref=1 equivalent: consume the field computed inside
        # the output-due microphysics call from its prepared p/post-call T.
        # Missing or double-consumed handoffs are cadence bugs and fail loud.
        from woof.core.refl import consume_refl_10cm
        frame["REFL_10CM"] = cp.asnumpy(
            consume_refl_10cm(state) if capture is None else captured_refl)
    from woof.io.history_selection import resolve

    frame, history_attrs = resolve(history_selection, None).apply(frame)
    attrs = _global_wrf_attrs(prepared.grid,
                             start_time,
                             getattr(prepared, "geog_selection", None),
                             domain=prepared.cfg,
                             coord=prepared.initial_result.coord,
                             feedback=feedback,
                             static_fields=getattr(prepared, "static_fields",
                                                   None))
    attrs.update(history_attrs)
    attrs.update(fire_history_attrs(getattr(state, "physics", None)))
    path = output_dir / wrfout_filename(valid_time, domain_id)
    with WrfoutWriter(
            path, nx=prepared.cfg.nx, ny=prepared.cfg.ny, nz=prepared.cfg.nz,
            dx=prepared.cfg.dx, dy=prepared.cfg.dy,
            title=title,
            global_attrs=attrs,
            field_schema=frame,
            # The soil axis is the selected LSM's geometry.  Omitting this
            # took WrfoutWriter's old literal-4 default, so a nine-layer
            # scheme would have declared soil_layers_stag=4 here.
            soil_layers=soil_layer_count(prepared.cfg),
            _retain_identity_handle=True,
            ) as writer:
        writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)
    proof = writer.complete_output_identity()
    if capture is not None:
        capture.history_committed(proof, grid_id=domain_id, valid_time=valid_time)
    if completed_records is not None:
        completed_records.append(proof)
    return path


# ---------------------------------------------------------------------------
# Integrate: the extracted outer-step loop
# ---------------------------------------------------------------------------


def _preparation_progress(progress_callback, phase: str) -> None:
    reporter = getattr(progress_callback, "preparing", None)
    if reporter is not None:
        reporter(phase)


def _finalizing_progress(progress_callback, phase: str, *,
                         work_bytes: int | None = None) -> None:
    """Publish one named beat from the stretch after the last model step.

    Same optional-hook convention as :func:`_preparation_progress`.  It
    exists because that stretch -- drain, device synchronize, trajectory
    digest, receipts, and a SHA-256 pass over every emitted frame -- has
    no model step to beat on, and on a large run it outlasts the
    supervisor's stale-integration threshold.  Silence there is
    indistinguishable from a hang, so a completing worker was killed and
    the finished run replayed as a restart loop.

    ``work_bytes`` is what the worker will write or read before its next
    beat.  The supervisor sizes the phase's bound from it
    (:func:`woof.supervisor.finalization_stale_threshold_seconds`), so a
    beat that is followed by bulk I/O must declare it.
    """
    reporter = getattr(progress_callback, "finalizing", None)
    if reporter is None:
        return
    if work_bytes is None:
        reporter(phase)
    else:
        reporter(phase, work_bytes=int(work_bytes))


def _writing_progress(progress_callback, phase: str, *,
                      work_bytes: int | None = None):
    """:func:`woof.supervisor.writing_progress`, beside its siblings here."""
    from woof.supervisor import writing_progress

    return writing_progress(progress_callback, phase, work_bytes=work_bytes)


def _checkpoint_work_bytes(model) -> int:
    """Bytes a checkpoint of every domain writes, for its write's bound.

    A streamed domain's state is its host store; a resident domain's is the
    arrays its state holds.  Counted from the arrays themselves, so a larger
    domain declares a larger write.
    """
    total = 0
    for node in model.walk_parent_first():
        streamed = getattr(node.state, "_streamed_domain", None)
        if streamed is not None:
            # Sizes only.  On the ranked road ``store`` is a full drain that
            # waits for the history writer; the planning view has the same
            # members, shapes and dtypes without either.
            run = getattr(streamed, "_run", None)
            planning = (run.raw_store if getattr(run, "ranked", False)
                        and hasattr(run, "raw_store") else None)
            arrays = dict(planning if planning is not None
                          else streamed.store).values()
        else:
            arrays = getattr(node.state, "__dict__", {}).values()
        total += sum(int(getattr(value, "nbytes", 0) or 0) for value in arrays
                     if getattr(value, "shape", None) is not None)
    return total


def _digest_progress(progress_callback, grid_id: int):
    """The per-domain beat a trajectory digest publishes before it hashes."""
    def declare(work_bytes: int) -> None:
        _finalizing_progress(
            progress_callback, f"trajectory-digest-d{int(grid_id):02d}",
            work_bytes=work_bytes)
    return declare


def _drain_progress(progress_callback):
    """The per-domain beat the history writers' drain publishes.

    Passed as ``PerDomainWrfoutWriters.drain(before_domain=...)``.  A queued
    frame is written and then read back once for its output identity, so
    each beat declares every domain's remaining bytes: the domains share
    one NetCDF lock, and the first one's drain can wait on all of them.
    """
    def before_domain(grid_id, work_bytes):
        _finalizing_progress(
            progress_callback, f"drain-history-writers-d{int(grid_id):02d}",
            work_bytes=work_bytes)
    return before_domain


def _final_health_reports(model, progress_callback) -> list:
    """Every domain's end-of-run health report, one beat per domain.

    A streamed domain kept in host memory is scanned on the CPU, the whole
    store, so its beat declares those bytes; a device scan declares none.
    """
    from woof.core.health import health_validator_for_domain

    reports = []
    for node in model.walk_parent_first():
        grid_id = int(node.cfg.grid_id)
        validator = health_validator_for_domain(model, node)
        _finalizing_progress(
            progress_callback, f"final-health-d{grid_id:02d}",
            work_bytes=getattr(validator, "host_scan_bytes", None))
        reports.append(validator.require_healthy(
            phase=f"final-state.d{grid_id:02d}"))
    return reports


def _resumed_start_step(*, elapsed_seconds: float, dt: float,
                        outer_steps: int, run_seconds: float) -> int:
    """The first outer step after a restore on the one-domain route.

    Equal to ``outer_steps`` means the restore point IS the stop: the run
    this checkpoint came from finished, the integration loop has zero
    iterations, and the route finalizes to a summary.  PAST the stop is
    the genuine mismatch -- a checkpoint from a longer run than the one
    being resumed -- and it refuses by name.  The equality used to refuse
    with the mismatch, which is how a complete run reported rc 1.
    """
    start = whole_step_count(elapsed_seconds, dt, "restart elapsed_seconds")
    if start > outer_steps:
        raise ValueError(
            f"restart file is at {elapsed_seconds} s of model time but this "
            f"configuration stops at run_seconds={run_seconds}, so the "
            "checkpoint comes from a LONGER run than the one being resumed "
            "and there is no state to integrate backwards to.  Resume from a "
            "checkpoint at or before the stop, or raise [experiment] "
            f"run_seconds to at least {elapsed_seconds}")
    return start


def _restart_is_complete(restart_info) -> bool:
    """Whether a tree restore landed exactly on this run's stop tick.

    ``None`` is a cold start, and any restore before the stop still has
    integration to do.  A restore AT the stop has none: the schedule
    executor refuses a start period at or past the end of the schedule,
    so falling through to it would trade this named completion for a
    bare ValueError.
    """
    return bool(getattr(restart_info, "already_complete", False))


def _output_committed(progress_callback, *, domain_id: int,
                      valid_time: datetime, path: Path,
                      step_log=None) -> None:
    """Tell an interested callback that one wrfout is durable.

    Same optional-hook convention as :func:`_preparation_progress`
    above: discovered by name, absent means nothing happens, so every
    existing ``progress_callback`` is unaffected.

    The existing ``last_durable_wrfout`` field on the per-step callback
    answers "which file was most recently published", which is what a
    heartbeat needs.  It cannot answer "a file just landed, here is its
    domain and valid time" -- a consumer would have to watch that field
    for changes and re-derive the rest from the filename.  This hook is
    raised at the exact call that published the file, with the three
    facts already in scope there.
    """

    reporter = getattr(progress_callback, "output_committed", None)
    if reporter is not None:
        reporter(domain=domain_id, valid_time=valid_time, path=path)
    if step_log is not None:
        # The run step log (progress.jsonl) records the frame the same
        # way the prepared runners record theirs: an ``output_written``
        # line.  Telemetry never fails a run.
        try:
            step_log.output_committed(
                domain=domain_id, valid_time=valid_time, path=path)
        except Exception:  # noqa: BLE001 - telemetry never fails a run
            pass


def _restart_written(step_log, *, domain_id: int, valid_time: datetime,
                     path, wall_seconds: float) -> None:
    """Record one durable checkpoint in the run step log.

    The prepared single-domain and prepared tree runners raise
    ``restart_written`` themselves; the scheduled ERA5/case-data route
    wrote its hourly checkpoints without a line in ``progress.jsonl``,
    so the log read as if no checkpoint existed.  ``None`` (an ensemble
    session, or a caller without a log) does nothing.
    """
    if step_log is None or path is None:
        return
    try:
        step_log.restart_written(domain=int(domain_id), valid_time=valid_time,
                                 path=path, wall_seconds=float(wall_seconds))
    except Exception:  # noqa: BLE001 - telemetry never fails a run
        pass


class _StepLogLanding:
    """The writers' single landing slot, shared by the run's progress
    callback and its step log (``attach_progress_callback`` reads ONE
    ``output_committed`` / ``frame_submitted`` pair, so both consumers
    go behind one object).  A sink that raises is dropped for that frame.
    """

    def __init__(self, progress_callback, step_log):
        self._landed = tuple(sink for sink in (
            getattr(progress_callback, "output_committed", None),
            step_log.output_committed) if sink is not None)
        self._submitted = tuple(sink for sink in (
            getattr(progress_callback, "frame_submitted", None),
            getattr(step_log, "frame_submitted", None)) if sink is not None)

    def output_committed(self, **event) -> None:
        for sink in self._landed:
            try:
                sink(**event)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass

    def frame_submitted(self, **event) -> None:
        for sink in self._submitted:
            try:
                sink(**event)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass


def apply_single_domain_pbl_cadence(physics, cfg) -> None:
    """Single-domain-loop PBL cadence: override ONLY at configured bldt=0.

    With ``cfg.bldt == 0`` every internal dynamics step runs the
    surface/PBL stack (WRF's bldt=0 semantics), and under the retired
    compatibility integrator the INTERNAL step is the authoritative
    interval, so the driver's setup values are overwritten with
    ``bldt_seconds = cfg.dt`` / ``stepbl = 1``.  A positive configured
    bldt keeps PhysicsDriver's WRF STEPBL calendar
    (``max(nint(bldt*60/dt), 1)``, woof/core/physics.py) untouched --
    the previous unconditional override silently forced every-step PBL
    regardless of namelist bldt, a latent trap that was inert only
    because the campaign lineage runs bldt=0.
    """
    if cfg.bldt == 0.0:
        physics.bldt_seconds = cfg.dt
        physics.stepbl = 1


def _reset_streamed_up_heli_max(stepper) -> None:
    """The history-interval UP_HELI_MAX reset, applied to a streamed domain.

    ``reset_up_heli_max(state)`` zeroes the accumulator the frame just
    snapshotted.  On a streamed domain that accumulator is
    ``store["scratch/up_heli_max"]``; the loop's ``state`` holds the
    preparation copy and zeroing it changes nothing the model will read.
    ``up_heli_max`` is a SERIALIZED scratch slot
    (``restart.classify_scratch_slot`` says so), so it is one of the 229
    carriers and it goes into every checkpoint -- which means the omission
    was not merely a wrong diagnostic: it made a streamed checkpoint and a
    resident checkpoint of the same forecast differ, in exactly one member,
    at the first history interval.  A running maximum only ever grows, so
    the symptom is an UP_HELI_MAX window that never resets and a bit
    comparison that fails on one array out of 229 with everything else
    identical -- the most persuasive possible argument that the difference
    is "just a diagnostic" and can be ignored.

    ``None`` (the resident run, where ``stepper is dycore.step``) does
    nothing at all, so the unstreamed path is untouched.
    """
    if stepper is None:
        return
    buffer = stepper.store.get("scratch/up_heli_max")
    if buffer is not None:
        buffer[...] = 0.0


def integrate_prepared_case(
        output_dir, prepared, *, start_time: datetime, output_title: str,
        domain_id: int = 1, integration_cfg: RunConfig | None = None,
        restart_path=None, run_seconds: float | None = None,
        history_interval_s: float | None = None,
        history_selection=None, history_begin_s: float = 0.0,
        history_end_s: float | None = None,
        restart_interval_s: float | None = None, progress_callback=None,
        auto_epssm=None,
        write_final_output: bool = False,
        preserved_forcing_prefix: bool = False,
        health_debug: bool = False,
        feedback=None, stepper=None,
        step_observer=None, step_log=None) -> RealCaseRunSummary:
    """Integrate a prepared real case and write its configured outputs.

    Extraction of the frozen reference integration loop: intentionally free
    of oracle comparisons and plotting so the normal run surface honors
    short forecasts and arbitrary configured output cadence.

    ``integration_cfg`` is the frozen profile's compatibility hook (its
    retired substep transform sets ``clock_dt = dt``); the experiment
    path leaves it ``None`` and integrates ``prepared.cfg`` as loaded
    (``clock_dt = 0.0``, which every consumer resolves to ``dt`` -- see
    the module docstring's equivalence notes).

    ``restart_path`` resumes a run: the deterministic preparation runs
    unchanged (rebuilding the setup and the resident LBC device tables),
    then :func:`woof.io.restart.restore_restart` overwrites the full
    cross-step state and restores the clock, and the loop continues from
    the restored outer step through ``run_seconds`` (the TOTAL forecast
    length from ``start_time``, exactly as for an uninterrupted run).
    The initial start-time wrfout is not rewritten on resume.
    ``run_seconds``/``history_interval_s``/``restart_interval_s`` are the
    experiment/domain timing authority.  Legacy callers omit them and use
    the compatibility copies on ``cfg``.

    ``auto_epssm`` carries the experiment's model-chosen domain labels into
    resident and streamed checkpoints. A derived child then keeps the label
    and applies its own terrain floor, rather than refusing an inherited
    value as explicit. Legacy callers without labels keep unlabelled headers.

    ``write_final_output`` publishes the actual terminal state when a
    member leg ends between history times. The final microphysics call
    produces its output diagnostics through the ordinary history path.

    ``step_observer`` is the same per-step seam
    :func:`woof.core.model.execute_experiment` offers (the step log's
    ``domain_step``): called once per committed outer step with
    ``grid_id``, ``step_count``, ``model_seconds``, ``step_wall_seconds``
    and ``dt``, after any due wrfout or checkpoint is durable.  Until it
    was threaded here, a single-domain fixed-step runtime-route run (ERA5
    ``woof go`` or ``woof run``, one domain) opened progress.jsonl and
    wrote run_start and run_end with no step records between them, so a
    consumer billing committed step time off that log read nothing.

    ``stepper`` is what one dynamics substep is taken with.  ``None`` binds
    ``woof.core.dycore.step`` ITSELF -- not a wrapper around it -- so a run
    that configures no ``[tiles]`` executes the identical call it always
    did.  ``woof.core.streaming.make_stepper`` returns either that same
    function or a :class:`~woof.core.streaming.StreamedDomain`, which has
    the same signature and advances the same domain by the same substep out
    of a pinned host store, one tile at a time.  The loop around it --
    history cadence, restart cadence, the REFL_10CM handshake -- is not aware
    of the difference and does not need to be.

    THE OBSERVERS ARE ASKED OF THE STEPPER, NOT OF THE STATE.  That is not
    tidiness, it is a correctness fix.  Under ``[tiles] store = "host"``
    the domain lives in a pinned host store and
    ``woof.core.streaming.attach`` fills it with ``gather.pinned_copy``,
    which COPIES: the prepared ``DomainState`` this function holds is a
    snapshot of t = 0 that no sweep ever writes again.  Reducing over it --
    which is what this loop did -- meant ``nan_free`` stayed true forever,
    ``w_max`` froze at its initial value, and a domain that went non-finite
    in the store completed and wrote a checkpoint recording that it had not.
    Keeping the state current instead is not available: the premise of the
    mode is that the domain does not fit on the card.  So the reduction is
    folded per tile inside the sweep and asked of the stepper here, exactly
    as ``dycore.stability_report`` is asked of it when the domain is resident
    -- ``streaming.stability_observer`` returns THAT function itself in that
    case, so a resident run executes the identical call it always did.

    ``StateHealthValidator`` is NOT yet folded.  See the comment at its
    cadence below: it is armed for a resident run and, under a host store, it
    still validates the t = 0 snapshot.
    """
    import cupy as cp
    from woof.core import streaming as _streaming
    from woof.core.health import StateHealthValidator
    # ``dycore.stability_report`` is deliberately NOT imported here: the
    # per-substep gate goes through ``stability_observer``, which returns that
    # function ITSELF for a resident domain and the sweep's per-tile fold of
    # the store for a streamed one.  Importing it anyway would leave the
    # obvious-looking wrong call one keystroke away.
    from woof.core.dycore import step
    from woof.core.streaming import (domain_call_counts, domain_field_max,
                                      is_streaming, stability_observer)
    from woof.io.restart import (auto_epssm_header, restart_filename,
                                  restore_restart, write_restart)
    from woof.supervisor import validate_manifest_checkpoint

    stepper = step if stepper is None else stepper
    # ``woof.core.dycore.stability_report`` ITSELF for a resident domain --
    # the same object, not a wrapper round it, so the resident path has no
    # "streaming disabled" branch that could be subtly different.
    stability_report = stability_observer(stepper)
    # Whether the full-state validator observes the live domain.  The
    # condition is store = "host" specifically, NOT "is streamed": with
    # store = "device" ``attach`` makes the store the DomainState's own
    # arrays, so the sweep writes the very memory the validator reads and it
    # is armed exactly as it always was.
    store_bundle = getattr(prepared, "streamed_store", None)
    health_armed = (store_bundle is not None or not (
        is_streaming(stepper) and getattr(stepper, "host_store", False)))
    health_validations_unarmed = 0
    if health_debug and not health_armed:
        # "armed under [tiles]", not "under streaming", for the reason
        # write_case_output's refusal above carries: this sentence is read by
        # somebody who configured [tiles], and `woof stream` is a different
        # feature with a prior claim on the other word.
        raise RuntimeError(
            "health_debug asks for a full-state validation every substep, but "
            "this domain is streamed to a host store: StateHealthValidator "
            "reads the resident DomainState and the sweep never writes it, so "
            "every one of those validations would pass regardless of what the "
            "forecast did.  Refused rather than run, because an attribution "
            "mode that cannot attribute is worse than no attribution mode.  "
            "The nan / w_max / CFL gate remains armed under [tiles] -- it "
            "folds the store per tile -- so a streamed run is still guarded, "
            "just not by this.")
    # WHERE THE DOMAIN IS.  Everything below that touches model state has to
    # ask, because a streamed domain's carriers are in the stepper's store
    # and the ``state`` this loop holds has been frozen at its preparation
    # values since ``attach`` copied them out.  The restart is the case
    # where getting it wrong is silent: ``write_restart(state, cfg)`` would
    # produce a complete, self-consistent, fully validating checkpoint of
    # the INITIAL CONDITION stamped with the current clock -- every shape
    # check passes and the file resumes into a forecast that threw away
    # every step taken.  ``is_streaming`` is ``False`` for the unstreamed
    # run, where ``stepper is dycore.step``, so the resident path below is
    # the identical code it always was.
    streamed = is_streaming(stepper)
    cfg = prepared.cfg
    run_seconds = cfg.run_seconds if run_seconds is None else run_seconds
    history_interval_s = (cfg.output_interval_s
                          if history_interval_s is None
                          else history_interval_s)
    restart_interval_s = (cfg.restart_interval_s
                          if restart_interval_s is None
                          else restart_interval_s)
    outer_steps, output_outer_steps = configured_run_schedule(
        cfg, run_seconds=run_seconds,
        output_interval_s=history_interval_s)
    history_begin_step, history_end_step = single_history_window_steps(
        cfg, history_begin_s=history_begin_s, history_end_s=history_end_s,
        grid_id=domain_id)
    restart_write_steps = restart_outer_steps(
        cfg, restart_interval_s=restart_interval_s)
    restart_request = RestartRequest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    integration_cfg = cfg if integration_cfg is None else integration_cfg
    state = prepared.initial_result.state
    _preparation_progress(progress_callback, "initialize-health-validator")
    if store_bundle is None:
        health = StateHealthValidator(state)
    else:
        from woof.core.health import StoreHealthValidator
        health = StoreHealthValidator(store_bundle, cfg)
    if not health_armed:
        # SAID, not discovered.  The stability record is folded per tile and
        # is correct under streaming; this validator is not, and the failure
        # mode of a disarmed gate is that everything looks fine.  An operator
        # who is told loses nothing; an operator who is not gets a forecast
        # whose only whole-state gate has been passing on a snapshot of
        # t = 0.  See the comment at its cadence below for why it cannot
        # simply be pointed at a tile.
        import warnings

        warnings.warn(
            "[tiles] store = 'host': StateHealthValidator is bound to "
            "the prepared DomainState, which the sweep does not write, so "
            "the full-state health gate is NOT armed for this run.  Its "
            "validations are SKIPPED AND COUNTED rather than run, so a run "
            "summary cannot show validations that observed nothing.  The "
            "NaN/w_max/CFL/swdown record IS armed -- it is folded per tile "
            "out of the store (woof.core.streaming.StreamedStability).",
            RuntimeWarning, stacklevel=2)
    # Asked of the STEPPER, not of the dycore: a streamed domain takes the
    # substep as a sweep of tiles and has no single phase to observe, so it
    # declines the hook and the loop falls back to validating between
    # substeps.  With no streaming configured this is the same signature it
    # has always been, because the stepper is the same function.
    phase_hook_supported = "phase_observer" in inspect.signature(
        stepper).parameters
    # bldt=0 follows each internal dynamics step; radiation follows the
    # configured WRF STEPRA calendar on those internal steps.  A positive
    # configured bldt keeps the driver's WRF STEPBL calendar (see helper).
    apply_single_domain_pbl_cadence(state.physics, integration_cfg)
    from woof.ensemble.runtime_context import bind_current_member_state
    bind_current_member_state(prepared_case=prepared, state=state,
                              cfg=integration_cfg, grid=prepared.grid)
    if restart_write_steps is not None:
        # THE CHECKPOINT'S QUESTION, ASKED BEFORE STEP 0.  A run that will
        # write checkpoints must be able to NAME its physics setup, and
        # until audit R-046 the first time anything asked was the writer
        # itself -- so a physics callable a class name cannot bind, or a
        # scheme with no stock-class row, was discovered at the first
        # restart interval with a whole hour of forecast already spent.
        # The config half of that question is answered at plan review
        # (woof.physics_registry.require_consumer_rows, from
        # validate_run_config); the DRIVER half needs the constructed
        # driver, which exists here and nowhere earlier, and this is still
        # before the first step.  The identity itself is discarded: what is
        # bought is the refusal's placement, not the value.
        #
        # THE SAME FUNCTION THE TREE DOOR CALLS.  This site asked
        # physics_setup_identity directly and so did not carry the tree
        # door's rule that the question is defined over a PhysicsDriver and
        # over nothing else: a state carrying some other physics object was
        # skipped by execute_experiment and raised on an attribute here.
        # Both doors now go through the one function that owns the rule.
        from woof.io.restart import ask_checkpoint_physics_identity

        ask_checkpoint_physics_identity(state, integration_cfg)
    outputs = []
    completed_records = []
    nan_free = True
    w_max = 0.0
    w_max_boundary_row = None
    boundary_w_max = 0.0
    interior_w_max = 0.0
    surface_forcing_updates = 0
    swdown_peak = -np.inf
    swdown_peak_time = start_time
    start_outer_step = 0
    last_checkpoint = None
    from woof.ensemble.runtime_counters import fixed_counter_observer_for_current
    counters = fixed_counter_observer_for_current(prepared, stepper,
        start_time=start_time, domain_id=domain_id, run_seconds=run_seconds,
        outer_steps=outer_steps, output_outer_steps=output_outer_steps,
        history_begin_step=history_begin_step, history_end_step=history_end_step,
        write_final_output=write_final_output)
    if counters is not None and restart_path is None:
        counters.observe()
    if restart_path is None and history_begin_step == 0:
        # The cold-start frame writes the initial zeros without consuming
        # a microphysics handoff that cannot exist before the first step.
        _preparation_progress(progress_callback, "cold-start-wrfout")
        output_path = write_case_output(
            prepared, output_dir, start_time, start_time=start_time,
            title=output_title, domain_id=domain_id,
            expect_refl_10cm=False, feedback=feedback,
            history_selection=history_selection,
            completed_records=completed_records)
        if output_path is not None:
            outputs.append(output_path)
            _output_committed(progress_callback, domain_id=domain_id,
                              valid_time=start_time, path=output_path,
                              step_log=step_log)
        # WRF resets the nwp_diagnostics running maxima each history
        # interval (module_diag_nwp.F:246-269); woof's ratified placement
        # is immediately after the frame is durable.
        from woof.core.uh_diag import reset_up_heli_max
        reset_up_heli_max(state)
        _reset_streamed_up_heli_max(stepper if streamed else None)
        if counters is not None:
            counters.history_consumed(start_time)
    elif restart_path is not None:
        _preparation_progress(progress_callback, "validate-checkpoint")
        last_checkpoint = validate_manifest_checkpoint(restart_path)
        _preparation_progress(progress_callback, "restore-checkpoint")
        # Into the STORE for a streamed domain, and into the state for a
        # resident one.  Not a preference: the resident reader's first act
        # is to allocate a host copy of every carrier and overwrite the
        # device state with it, and above the card's ceiling that state does
        # not exist.  Below it, the state exists but is not the domain --
        # restoring into it would leave the store holding the PREPARATION
        # values and the run would resume from t=0 with the checkpoint's
        # clock.  The streamed reader applies the same refusals in the same
        # order (config echo, setup fingerprint, physics setup fingerprint,
        # clock admissibility, member classification) plus one the resident
        # reader cannot make: a resuming resident state has every slot
        # allocated by preparation, so it cannot tell a restored carrier
        # from an unrestored one, and the streamed reader requires the
        # file's member set to BE the store's.
        info = (stepper.restore_restart(last_checkpoint, cfg) if streamed
                else restore_restart(last_checkpoint, state, cfg,
                    **({'preserved_forcing_prefix': True} if preserved_forcing_prefix else {})))
        start_outer_step = _resumed_start_step(
            elapsed_seconds=info.elapsed_seconds, dt=cfg.dt,
            outer_steps=outer_steps, run_seconds=run_seconds)
        if start_outer_step == outer_steps:
            print(
                "  restart point is this configuration's stop "
                f"({run_seconds:g} s of model time): the run is already "
                "complete, finalizing without integrating")
        trackers = info.run_trackers
        if trackers is not None:
            # Summary continuity: an interrupted-and-resumed run reports
            # the same run bookkeeping as an uninterrupted one.
            nan_free = bool(trackers["nan_free"])
            w_max = float(trackers["w_max_ms"])
            w_max_boundary_row = trackers["w_max_boundary_row"]
            boundary_w_max = float(trackers["boundary_w_max_ms"])
            interior_w_max = float(trackers["interior_w_max_ms"])
            swdown_peak = float(trackers["swdown_peak_wm2"])
            swdown_peak_time = datetime.fromisoformat(
                trackers["swdown_peak_time"])
        surface_forcing_updates = domain_call_counts(stepper, state)["radiation"]
        if counters is not None:
            counters.observe()
    _preparation_progress(progress_callback, "initial-health-gate")
    if health_armed or restart_path is None:
        health.require_healthy(phase="initialized-or-restored")
    else:
        # The restored host store has its own validated checkpoint; the
        # untouched preparation state cannot certify those restored arrays.
        health_validations_unarmed += 1
    if store_bundle is not None:
        from woof.ingest.case_store import write_initialization_receipt
        write_initialization_receipt(output_dir, prepared, health.coverage)
    if progress_callback is not None:
        progress_callback(
            model_elapsed_seconds=float(stepper.scalars["elapsed_seconds"] if streamed else state.elapsed_seconds),
            outer_step=start_outer_step,
            last_durable_wrfout=(outputs[-1] if outputs else None),
            last_checkpoint=last_checkpoint, phase="initialized-or-restored",
            step_wall_seconds=0.0)
    dynamics_substeps = int(round(cfg.dt / integration_cfg.dt))
    final_output_step = outer_steps if write_final_output else None
    cadence_change = {}
    if cfg.history_interval2_s:
        cadence_change = dict(
            history_interval2_steps=whole_step_count(cfg.history_interval2_s, cfg.dt, "history_interval2_s"),
            history_interval_change_step=round(cfg.history_interval_change_s / cfg.dt))
    hydro_forecast_started = time.perf_counter()
    for outer_step in range(start_outer_step, outer_steps):
        outer_started = time.perf_counter()
        forcing_time = start_time + timedelta(seconds=outer_step * cfg.dt)
        for substep in range(dynamics_substeps):
            refl_due = (cfg.mp_physics in REFL_10CM_MICROPHYSICS
                        and refl_10cm_due(
                            outer_step, substep, output_outer_steps,
                            dynamics_substeps, final_outer_step=final_output_step,
                            history_begin_outer_step=history_begin_step,
                            history_end_outer_step=history_end_step, **cadence_change))
            phase = f"outer-{outer_step + 1}.substep-{substep + 1}"
            if health_debug and not phase_hook_supported and health_armed:
                health.require_healthy(phase=phase + ".pre-step")
            step_kwargs = {"refl_10cm_due": refl_due}
            if int(getattr(cfg, "ifire", 0)) == 2:
                step_kwargs["fire_history_due"] = refl_10cm_due(
                    outer_step, substep, output_outer_steps,
                    dynamics_substeps, final_outer_step=final_output_step,
                    history_begin_outer_step=history_begin_step,
                    history_end_outer_step=history_end_step, **cadence_change)
            if health_debug and phase_hook_supported:
                step_kwargs["phase_observer"] = health.phase_observer
            stepper(state, integration_cfg, **step_kwargs)
            if counters is not None:
                counters.observe()
            # Validator cadence (controller amendment, 2026-07-16): the
            # measured full-validation cost is 5.00% of step wall vs the
            # plan's <=2% gate, so the pre-registered remedy applies --
            # every 4th step (~1.25%) PLUS mandatory instants: the final
            # step, every output-due step, and every restart instant.
            # health_debug forces every step (attribution mode).
            step_index = outer_step * dynamics_substeps + substep
            mandatory = (
                outer_step == outer_steps - 1 and
                substep == dynamics_substeps - 1
            ) or refl_due or (
                restart_write_steps
                and (outer_step + 1) % restart_write_steps == 0
                and substep == dynamics_substeps - 1)
            if health_debug or mandatory or step_index % 4 == 0:
                # NOT YET FOLDED, and loud about it above rather than
                # silent here.  ``StateHealthValidator`` is a descriptor
                # kernel over up to 1024 WHOLE fields with one block per
                # descriptor and no windowing, so it cannot be pointed at a
                # tile's interior the way the stability reduction can.  The
                # foldable form is a per-BUFFER validation issued after the
                # gather and before the step -- at that instant the buffer
                # holds exactly the store's bytes, halo included, so the
                # union over tiles covers the domain with no false positives
                # -- but it observes the state one step behind, needs its own
                # readback point inside the sweep, and multiplies the launch
                # by the tile count.  That is a bigger change than a
                # correctness fix should smuggle in, so it is stated, not
                # done.  Under a host store this call still validates the
                # t=0 snapshot -- so under a host store it is SKIPPED
                # AND COUNTED rather than run.  Counting the skips makes an
                # unarmed validator visible in the run summary instead of
                # indistinguishable from a passing one.  The nan / w_max /
                # CFL gate below is a different observer and IS armed: it
                # folds the store.
                if health_armed:
                    health.require_healthy(phase=phase + ".post-step")
                else:
                    health_validations_unarmed += 1
            width = cfg.spec_bdy_width
            # NOT stability_report(state, ...).  Under [tiles] with a host
            # store the domain's arrays are in the store and this state is
            # never written by the sweep, so reading it here reports the
            # condition the store was FILLED from -- healthy at t=0 and
            # healthy forever, which silently disarms the nan gate below and
            # freezes w_max and the CFL at their initial values.  step_health
            # asks the stepper: the dycore's own whole-domain reduction when
            # resident, the sweep's per-tile fold of the store when streamed.
            report = stability_report(
                state, integration_cfg, boundary_width=width)
            nan_free = nan_free and not report["nan"]
            step_w_max = float(report["w_max"])
            if step_w_max > w_max:
                max_index = np.unravel_index(
                    report["w_argmax"], (cfg.nz + 1, cfg.ny, cfg.nx))
                _k, j, i = (int(index) for index in max_index)
                distance = min(j, cfg.ny - 1 - j, i, cfg.nx - 1 - i)
                # WRF/kernel boundary distance: d=0 is specified; d=1,2,3
                # are relaxation rows 1,2,3 for this case.
                w_max_boundary_row = (distance if distance < width else None)
                w_max = step_w_max
            boundary_w_max = max(
                boundary_w_max, report["boundary_w_max"])
            interior_w_max = max(
                interior_w_max, report["interior_w_max"])
            if not nan_free:
                raise RuntimeError(
                    "real-case integration produced a non-finite state at "
                    f"dynamics substep "
                    f"{dynamics_substeps * outer_step + substep + 1}")
            # ``report["nan"]`` is decided by u_max/w_max/th_max alone, so
            # it cannot see a collapsed or folded model layer: geopotential
            # is not one of those three fields, and a mass cell whose live
            # thickness went non-positive reaches this report ONLY as a
            # non-finite vertical Courant number (``health.cu`` mask bit 32
            # -> ``result[5] = nanf("")``).  Reading nothing but ``nan``
            # here left that signal with no observer on this route -- and
            # because the vertical term is a whole-domain reduction, one
            # inverted layer also replaces the real vertical maximum, so
            # the gate went blind exactly as the failure it exists to catch
            # developed.  Not folded into ``nan_free``: the state is finite,
            # the GEOMETRY is not, and the two want different sentences.
            step_cfl = report["cfl"]
            if step_cfl is not None and not math.isfinite(float(step_cfl)):
                raise RuntimeError(
                    "real-case integration produced a non-finite vertical "
                    "Courant number at dynamics substep "
                    f"{dynamics_substeps * outer_step + substep + 1}: a "
                    "model layer's live thickness is non-positive or "
                    "non-finite (a collapsed or folded geopotential "
                    "column), which no field maximum can report")
        # Both of these are the DOMAIN's, and under a host store the domain
        # is not on ``state``: its call counts live on the sweep's carried
        # clock and its swdown maximum is in the store.
        surface_forcing_updates = domain_call_counts(
            stepper, state)["radiation"]
        # Once per OUTER step over one 2-D field, so the accurate fix for the
        # streamed case is to read the store on the host rather than to add
        # another hook inside the sweep: no tile writes swdown's maximum, and
        # a 672x672 float32 plane is 1.8 MB.  Resident runs still take
        # ``cp.max`` over the state, which is what the argument names.
        step_swdown_peak = domain_field_max(
            stepper, state, "fields/swdown", state.physics.fields["swdown"])
        if step_swdown_peak > swdown_peak:
            swdown_peak = step_swdown_peak
            swdown_peak_time = forcing_time
        if history_output_due(outer_step, output_outer_steps,
                              final_outer_step=final_output_step,
                              history_begin_outer_step=history_begin_step,
                              history_end_outer_step=history_end_step, **cadence_change):
            valid = start_time + timedelta(seconds=(outer_step + 1) * cfg.dt)
            output_path = write_case_output(
                prepared, output_dir, valid, start_time=start_time,
                title=output_title, domain_id=domain_id,
                feedback=feedback, history_selection=history_selection,
                completed_records=completed_records)
            if output_path is not None:
                outputs.append(output_path)
                _output_committed(progress_callback, domain_id=domain_id,
                                  valid_time=valid, path=output_path,
                                  step_log=step_log)
            # History-interval reset of the UP_HELI_MAX window (the frame
            # above snapshotted the accumulator synchronously).
            from woof.core.uh_diag import reset_up_heli_max
            reset_up_heli_max(state)
            _reset_streamed_up_heli_max(stepper if streamed else None)
            if counters is not None:
                counters.history_consumed(valid)
        requested_restart = restart_request.pending()
        if (requested_restart or (restart_write_steps is not None
                and (outer_step + 1) % restart_write_steps == 0)):
            valid = start_time + timedelta(seconds=(outer_step + 1) * cfg.dt)
            checkpoint_path = (
                output_dir / restart_filename(valid, f"d{domain_id:02d}"))
            checkpoint_started = time.perf_counter()
            trackers = {
                "nan_free": nan_free,
                "w_max_ms": float(w_max),
                "w_max_boundary_row": w_max_boundary_row,
                "boundary_w_max_ms": float(boundary_w_max),
                "interior_w_max_ms": float(interior_w_max),
                "swdown_peak_wm2": float(swdown_peak),
                "swdown_peak_time": swdown_peak_time.isoformat(),
            }
            if streamed:
                # From the pinned store, with ZERO device-to-host copies.
                # The resident writer would not fail here -- it would write
                # this loop's ``state``, which streaming froze at t=0 -- so
                # the branch is the difference between a checkpoint and a
                # forgery that passes every check in the reader.
                last_checkpoint = stepper.write_restart(
                    checkpoint_path, cfg, run_trackers=trackers,
                    tree_header=auto_epssm_header(domain_id, auto_epssm) or None).path
            else:
                last_checkpoint = write_restart(
                    checkpoint_path, state, cfg, run_trackers=trackers,
                    tree_header=auto_epssm_header(domain_id, auto_epssm) or None,
                    **({'preserved_forcing_prefix': True} if preserved_forcing_prefix else {}))
            _restart_written(
                step_log, domain_id=domain_id, valid_time=valid,
                path=last_checkpoint,
                wall_seconds=time.perf_counter() - checkpoint_started)
            from woof.resume import retire_superseded_checkpoints
            retire_superseded_checkpoints(output_dir)
            if requested_restart:
                restart_request.acknowledge()
        # The state gate completed after the final internal step.  Publish
        # progress only after any due wrfout/checkpoint is durable, so a
        # heartbeat can never advertise unguarded or unpublished work.
        if progress_callback is not None:
            progress_callback(
                model_elapsed_seconds=float(stepper.scalars["elapsed_seconds"] if streamed else state.elapsed_seconds),
                outer_step=outer_step + 1,
                last_durable_wrfout=(outputs[-1] if outputs else None),
                last_checkpoint=last_checkpoint, phase="post-d01-sync",
                step_wall_seconds=time.perf_counter() - outer_started)
        if step_observer is not None:
            # The committed step, numbered and stamped exactly as the
            # domain-tree executor numbers it (woof/core/model.py): the
            # step just taken is outer_step + 1 at its post-step model
            # time, on this run's fixed dt.  Telemetry never fails a run,
            # so a sink that raises is dropped for that one line.
            try:
                step_observer(
                    grid_id=int(domain_id), step_count=outer_step + 1,
                    model_seconds=(outer_step + 1) * float(cfg.dt),
                    step_wall_seconds=time.perf_counter() - outer_started,
                    dt=float(cfg.dt))
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass
    _finalizing_progress(progress_callback, "synchronize-device")
    cp.cuda.runtime.deviceSynchronize()
    from woof.hydro.forecast_products import finalize_forecast as finalize_hydro
    hydro_receipt = finalize_hydro(
        state.physics, outdir=output_dir, dx_m=float(cfg.dx),
        forecast_seconds=time.perf_counter() - hydro_forecast_started,
        atmosphere_steps=(outer_steps - start_outer_step) * dynamics_substeps)

    # After the final device synchronization and after every history frame is
    # durable, so the digest observes the trajectory and cannot join it.
    trajectory_digest = None
    if trajectory_digest_enabled():
        _finalizing_progress(progress_callback, "trajectory-digest")
        from woof.state_digest import canonical_state_digest

        declare = _digest_progress(progress_callback, domain_id)
        trajectory_digest = {
            f"d{domain_id:02d}": (
                stepper.canonical_digest(_SingleDomainDigestClock(), scope="trajectory",
                                         before_hash=declare)
                if streamed else canonical_state_digest(
                    state, _SingleDomainDigestClock(), scope="trajectory",
                    before_hash=declare)),
            "boundary_clock_provenance": _SingleDomainDigestClock.provenance,
        }

    rainc_max = 0.0
    rainc_ji = None
    rainc_lat = None
    rainc_lon = None
    if state.physics.rainc is not None:
        rainc_host = (np.asarray(stepper.store["scratch/cu_rainc"])
                      if streamed else cp.asnumpy(state.physics.rainc))
        j, i = np.unravel_index(int(np.argmax(rainc_host)),
                                rainc_host.shape)
        rainc_max = float(rainc_host[j, i])
        rainc_ji = (int(j), int(i))
        lat, lon = prepared.grid.latlon_mass()
        rainc_lat = float(lat[j, i])
        rainc_lon = float(lon[j, i])
    if health_validations_unarmed:
        import warnings

        warnings.warn(
            f"{health_validations_unarmed} full-state health validations were "
            "skipped as unarmed over this streamed run; the per-substep "
            "nan / w_max / CFL gate ran on the store as normal.",
            RuntimeWarning, stacklevel=2)
    return RealCaseRunSummary(
        trajectory_digest=trajectory_digest,
        frame_records=tuple(_frame_records(
            outputs, completed_records=completed_records,
            progress_callback=progress_callback)),
        wrfout_paths=tuple(outputs), nan_free=nan_free,
        w_max_ms=w_max, boundary_w_max_ms=boundary_w_max,
        interior_w_max_ms=interior_w_max,
        w_max_boundary_row=w_max_boundary_row,
        # Same predicate as woof.verify.metrics.boundary_zone_blowup,
        # restated because the standalone preprocessing distribution omits
        # the verification tree.  The interior leg matters: max(nan, 1.0) is
        # nan, so an unmeasurable interior would otherwise switch the
        # boundary-reflection detector off instead of firing it.
        boundary_zone_blowup=(not np.isfinite(boundary_w_max)
                              or not np.isfinite(interior_w_max)
                              or boundary_w_max
                              > 5.0 * max(interior_w_max, 1.0)),
        dynamics_substeps=dynamics_substeps,
        ysu_nan_guard_fires=(int(stepper.scalars["ysu_nan_guard_fires"])
                             if streamed else state.physics.ysu_nan_guard_fires),
        surface_forcing_updates=surface_forcing_updates,
        swdown_peak_wm2=swdown_peak, swdown_peak_time=swdown_peak_time,
        completed_seconds=float(stepper.scalars["elapsed_seconds"] if streamed else state.elapsed_seconds),
        rainc_max_mm=rainc_max, rainc_max_ji=rainc_ji,
        rainc_max_lat=rainc_lat, rainc_max_lon=rainc_lon,
        hydro_receipt=hydro_receipt,
    )


# ---------------------------------------------------------------------------
# Resolved-config report (G2): every formerly implicit path/time/policy
# ---------------------------------------------------------------------------

def downward_longwave_source(exp: ExperimentConfig, cfg: RunConfig) -> str:
    """One line naming where a domain's downward longwave comes from.

    Written into every resolved-configuration report so that a run
    integrating a CONSTANT GLW says so on the receipt.  A published GLW
    row is indistinguishable from a measured one once it is in a wrfout
    file; this is the sentence that distinguishes them.

    The sentence is derived from
    :func:`woof.physics_compat.downward_longwave_disposition` -- the
    same classification the config-load guard and ``initialize_physics``
    refuse on -- so the receipt can never describe a fate the engine
    does not enact.
    """

    from woof.config import radiation_scheme_ids
    from woof.physics_compat import (CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                      downward_longwave_disposition)

    lw, sw = radiation_scheme_ids(cfg)
    kind, consumer = downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw,
        sf_surface_physics=int(cfg.sf_surface_physics))
    if kind == "scheme":
        return (f"computed every radiation call by ra_lw_physics={lw} "
                "(radt clock)")
    constant = declared_constant_glw(exp)
    if constant is None:
        if kind == "unused":
            return ("ra_lw_physics=0 and no constant declared -- nothing "
                    "reads or publishes GLW in this suite")
        # Unreachable through build_experiment, whose load guard refuses
        # exactly the consumed/published kinds without the token; stated
        # accurately anyway for a hand-assembled ExperimentConfig.
        return ("NO SOURCE: ra_lw_physics=0 with no constant declared, "
                f"yet GLW is {kind} -- this configuration is refused at "
                "config load and by initialize_physics")
    header = f"DECLARED CONSTANT {constant:g} W m-2, NOT a computed flux: "
    footer = f" (declared by {CONSTANT_DOWNWARD_LONGWAVE_ACK})"
    if kind == "consumed":
        return (header + "ra_lw_physics=0, so no scheme produces downward "
                f"longwave and the land surface ({consumer}) integrates "
                "this one number for the whole forecast" + footer)
    if kind == "published":
        return (header + "ra_lw_physics=0 and no land-surface scheme "
                "reads it, but shortwave keeps the radiation slot active, "
                "so this one number is published as the GLW row of every "
                "wrfout frame" + footer)
    return (header + "declared but UNUSED -- no land-surface scheme reads "
            "it and radiation is off, so it reaches no scheme and no "
            "wrfout row" + footer)


def resolved_config_report(exp: ExperimentConfig, data: CaseDataConfig,
                           forcing_times=None, *, input_catalog=None) -> str:
    """Human- and test-readable enumeration of every resolved value.

    Every path, time, and policy that used to be implicit in the runtime
    path appears here by name.  ``input_catalog`` supplies the forcing
    selection authority and its exclusions; ``forcing_times`` may narrow that
    selection to the schedule actually consumed by preparation.
    """
    dc = single_domain(exp)
    cfg = dc.run
    lines = [f"resolved experiment configuration -- {exp.name}"]

    def add(key, value):
        lines.append(f"  {key} = {value}")

    for record in data.resolved_inputs():
        detail = f" ({record.detail})" if record.detail else ""
        add(f"input.{record.role}", f"{record.path}{detail}")
    if data.source_orography is None:
        add("input.source_orography",
            "era5_z_invariant from forcing SOILGEO")
    add("time.start_time", exp.start_time.isoformat())
    add("time.run_seconds", f"{exp.run_seconds:g}")
    add("time.dt", f"{cfg.dt:g}")
    add("time.history_interval_s", f"{dc.history_interval_s:g}")
    add("time.restart_interval_s", f"{exp.restart_interval_s:g}")
    add("radiation.column_chunk", f"{exp.column_chunk}")
    add("radiation.downward_longwave",
        downward_longwave_source(exp, cfg))
    # THE CARRIER POLICY, always in the receipt, both values.  A reader
    # looking for "did this run integrate a sky nobody computed" gets an
    # answer whether or not the escape was taken, which is what makes the
    # answer trustworthy -- an absent line reads as "not applicable" and
    # this question is never not applicable to a run with a land surface.
    add("radiation.surface_radiation_policy",
        f"{cfg.surface_radiation_policy}"
        + ("" if cfg.surface_radiation_policy == "required" else
           " (EXPERIMENTAL FORCING: carriers with no producer are "
           "consumed at their allocation fill; not a valid configuration "
           "for a real case)"))
    add("time.forcing_interval_s",
        "discover" if data.forcing_interval_s is None
        else f"{data.forcing_interval_s:g}")
    if input_catalog is not None and forcing_times is None:
        forcing_times = input_catalog.valid_times
    if forcing_times is not None:
        times = tuple(forcing_times)
        coverage = (times[-1] - times[0]).total_seconds()
        add("time.forcing_times",
            ", ".join(t.isoformat() for t in times))
        add("time.forcing_times_consumed",
            ", ".join(t.isoformat() for t in times))
        add("time.forcing_coverage_s", f"{coverage:g}")
    if input_catalog is not None:
        exclusions = tuple(input_catalog.excluded_valid_times)
        add("time.forcing_times_excluded_by_catalog",
            (", ".join(t.isoformat() for t in exclusions)
             if exclusions else "none"))
    add("vertical.nz", f"{cfg.nz}")
    add("vertical.eta_levels",
        f"{len(exp.vertical.eta_levels)} full levels "
        f"[{exp.vertical.eta_levels[0]:g} .. "
        f"{exp.vertical.eta_levels[-1]:g}]")
    add("vertical.p_top", f"{exp.vertical.p_top:g}")
    add("vertical.hybrid_opt", f"{exp.vertical.hybrid_opt}")
    add("vertical.etac", f"{exp.vertical.etac:g}")
    add("policy.sfcp_to_sfcp", str(data.sfcp_to_sfcp))
    add("policy.co2_vmr", (f"{data.co2_vmr:g}" if data.co2_vmr is not None
                           else "date-indexed NOAA annual policy"))
    add("policy.climatology_date",
        exp.start_time.date().isoformat()
        + " (monthly GEOG fields interpolated to the run date)")
    highres = getattr(data, "static_highres", None)
    if highres is not None:
        for key, value in highres.echo().items():
            add(f"static.highres.{key}", value)
    add("output.domain_id", f"{data.output_domain}")
    add("output.title", data.output_title)
    add("output.filename_pattern",
        f"wrfout_d{data.output_domain:02d}_<YYYY-MM-DD_HH_MM_SS>")
    add("grid.nx_ny_dx", f"{cfg.nx} x {cfg.ny} @ {cfg.dx:g} m")
    if exp.projection is not None:
        proj = exp.projection
        add("grid.projection",
            f"{proj.map_proj} ref=({proj.ref_lat:g}, {proj.ref_lon:g}) "
            f"truelat=({proj.truelat1:g}, {proj.truelat2:g}) "
            f"stand_lon={proj.stand_lon:g}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment-path entry points (CLI static / ingest / run)
# ---------------------------------------------------------------------------

def _write_npz(path, fields: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    return path


def write_static(exp: ExperimentConfig, data: CaseDataConfig,
                 output) -> Path:
    """Build the experiment domain's GEOG fields into a portable NPZ."""
    dc = single_domain(exp)
    grid = experiment_grid(exp, data)
    selection = GeogSelection.from_case_data(data, domain_id=dc.grid_id)
    selection = with_terrain_drag_statics(selection, dc.run)
    fields = build_static(
        grid, data.geog_root, selection=selection)
    highres = getattr(data, "static_highres", None)
    if highres is not None and getattr(highres, "enabled", False):
        from woof.static.highres_production import apply_highres_statics
        fields, _ = apply_highres_statics(
            fields, grid, config=highres, domain_id=dc.grid_id,
            case_date=exp.start_time.date(),
            landuse_attrs=selection.landuse_global_attrs())
    return _write_npz(output, fields)


def write_ingest(exp: ExperimentConfig, data: CaseDataConfig,
                 output) -> Path:
    """Run real-data initialization and write its live FP32 state to NPZ.

    This is a stage artifact for inspection/reproducibility, not a restart
    file; ``woof run`` rebuilds the deterministic setup from the same
    config.
    """
    prepared = prepare_experiment_case(exp, data)
    result = prepared.initial_result
    state = result.state
    import cupy as cp
    names = ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr",
             "mub2d", "msft", "msfu", "msfv", "f", "e")
    fields = {name: cp.asnumpy(getattr(state, name)) for name in names}
    fields.update({
        "surface_pressure": result.surface_pressure,
        "surface_qv": result.surface_qv,
        "dry_mass": result.dry_mass,
        "dry_pressure": result.dry_pressure,
        "total_pressure": result.total_pressure,
        "total_geopotential": result.total_geopotential,
        "integrated_moisture_pressure": result.integrated_moisture_pressure,
        "case": np.asarray(exp.name),
    })
    return _write_npz(output, fields)


def _terrain_acoustics_for_case(exp, data, *, detail: bool = False):
    """The experiment with each domain's acoustic substeps derived from its ground.

    The root is read off :func:`case_static_fields`, the memoized build the
    root preparation makes a few steps later, overlay included; each nest
    off its own terrain at its own resolution (:func:`build_terrain`, or the
    full build where a ``[static.highres]`` overlay replaces the terrain),
    because a nest carries steeper ground than its parent.  A following nest
    is also read over its statics corridor, the ground it can be moved onto
    mid-run: the same frame and reach window the corridor is built on, and
    the same terrain the vertical-coordinate survey reads for it
    (:func:`woof.vertical_adaptation.run_terrain_fields`).
    """

    from woof.acoustic_adaptation import (adapt_experiment_to_terrain,
                                           fold_corridor_reading,
                                           readings_from_static)
    from woof.static.build import build_terrain
    from woof.static.corridor import (corridor_grid, moving_grid_ids,
                                       planned_corridor)

    if exp.projection is None or getattr(data, "geog_root", None) is None:
        # No projected grid or no WPS_GEOG root, so no static terrain to
        # read: the preparation below refuses such a real case by name
        # (experiment_grid, the static build), and nothing here should
        # speak before it does.
        return (exp, (), {}, (), {}) if detail else exp
    grids = tuple(grids_from_projection_config(exp))
    highres = getattr(data, "static_highres", None)
    highres_on = bool(highres is not None
                      and getattr(highres, "enabled", False))
    statics = {}
    for index, (dc, grid) in enumerate(zip(exp.domains, grids)):
        gid = int(dc.grid_id)
        selection = GeogSelection.from_case_data(data, domain_id=gid)
        if index == 0 or highres_on:
            statics[gid] = case_static_fields(
                grid, data.geog_root, selection=selection,
                static_highres=highres, domain_id=gid,
                case_date=exp.start_time.date())
        else:
            statics[gid] = {"HGT_M": build_terrain(
                grid, selection.root, selection=selection)}
    grid_by_id = {int(dc.grid_id): grid
                  for dc, grid in zip(exp.domains, grids)}
    readings = readings_from_static(exp, statics,
                                    grids_by_grid_id=grid_by_id)
    by_id = {int(dc.grid_id): dc for dc in exp.domains}
    reach = {}
    for gid in sorted(moving_grid_ids(exp)):
        dc = by_id.get(gid)
        if dc is None or int(dc.parent_id) == 0:
            continue
        corridor = corridor_grid(grid_by_id[gid],
                                 planned_corridor(exp, dc).geometry)
        selection = GeogSelection.from_case_data(data, domain_id=gid)
        if highres_on:
            terrain = case_static_fields(
                corridor, data.geog_root, selection=selection,
                static_highres=highres, domain_id=gid,
                case_date=exp.start_time.date())["HGT_M"]
        else:
            terrain = build_terrain(corridor, selection.root,
                                    selection=selection)
        fold_corridor_reading(readings, gid, dc.run, terrain)
        reach[gid] = terrain
    adapted, acoustic = adapt_experiment_to_terrain(exp, readings)
    if detail:
        terrain = {gid: static["HGT_M"] for gid, static in statics.items()}
        return adapted, acoustic, terrain, grids, reach
    return adapted


def _terrain_clock_for_case(exp, data, acoustic, terrain, grids, reach):
    """The experiment with each domain's long step fitted to its ground and
    the strongest crest-level wind its forcing carries over the window.

    Read off the decoded forcing itself, on its own grid over each
    domain's footprint, before anything is prepared: the prepared state
    builds its physics calendar from the step, so the step is settled
    first.  The decode is the keyed one the preparation reuses.  Returns
    ``(experiment, adaptations)``.
    """

    from woof.terrain_clock import (SnapshotWinds, clock_for_domains,
                                     forcing_window)

    if not acoustic or not getattr(data, "forcing", None):
        return exp, ()
    catalog = _runtime_input_catalog(data)
    window = forcing_window(forcing_snapshots(data, catalog),
                            exp.start_time, exp.run_seconds)
    starts = {}
    for dc, grid in zip(exp.domains, grids):
        lat, lon = grid.latlon_mass()
        starts[int(dc.grid_id)] = SnapshotWinds(
            f"d{int(dc.grid_id):02d}", window, np.asarray(lat),
            np.asarray(lon), exp.start_time)
    statics = {gid: {"HGT_M": field} for gid, field in terrain.items()}
    # The local-face clock (terrain_clock = "local_face", the default)
    # reads every face with the map factors the substep rule read off
    # these grids (acoustic_adaptation.readings_from_static); without them
    # its faces would not be the substep rule's and it keeps the
    # domain-wide reading.
    for dc, grid in zip(exp.domains, grids):
        gid = int(dc.grid_id)
        if (str(getattr(dc.run, "terrain_clock", "measured")) == "local_face"
                and gid in statics):
            for name, method in (("MAPFAC_U", "mapfac_u"),
                                 ("MAPFAC_V", "mapfac_v")):
                if hasattr(grid, method):
                    statics[gid][name] = getattr(grid, method)()
    return clock_for_domains(
        exp, acoustic, statics=statics, starts=starts, corridors=reach)


def _write_terrain_clock_receipt(outdir, adaptations) -> Path | None:
    """Publish the long-step derivation when it changed a domain."""

    if not any(adaptation.adapted for adaptation in adaptations):
        return None
    from woof.terrain_clock import clock_receipt

    return _publish_receipt(outdir, TERRAIN_CLOCK_RECEIPT_NAME,
                            clock_receipt(adaptations))


def _write_acoustic_receipt(outdir, adaptations, clock=None) -> Path | None:
    """Publish the acoustic derivation when the off-centering floor raised
    a domain's default epssm, so the run's own folder says which domain
    ran which value and why (A171)."""

    if not any(getattr(adaptation, "offcentering_raised", False)
               for adaptation in adaptations):
        return None
    from woof.acoustic_adaptation import acoustic_receipt

    return _publish_receipt(outdir, ACOUSTIC_RECEIPT_NAME,
                            acoustic_receipt(adaptations, clock))


def _publish_receipt(outdir, name: str, document) -> Path:
    """One receipt written whole: a partial file renamed into place."""

    encoded = (json.dumps(document, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    path = Path(outdir) / name
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def run_experiment(exp: ExperimentConfig, data: CaseDataConfig, outdir, *,
                   restart=None, progress_callback=None,
                   health_debug: bool = False
                   ) -> RealCaseRunSummary | ExperimentRunSummary:
    """Prepare and integrate a single domain or a complete domain tree.

    Opens the step log the prepared runners keep beside their outputs
    (``<outdir>/progress.jsonl``: one ``step`` record per domain per
    committed step, ``run_start``/``run_end`` around them) and hands its
    observer to the executor.  This route never did, so a consumer that
    bills committed step time off that log (WOOF's worker) read nothing
    from a finished ERA5 forecast and reported it failed: three Zhengzhou
    ERA5 27-9-3-1 runs of 2026-10-18, every frame drawn.  Inside an
    ensemble session the session wires its own log, so nothing is opened
    here.
    """
    from woof.ensemble.runtime_context import current_capture, current_session
    if current_session() is not None:
        return _run_experiment(exp, data, outdir, restart=restart,
                               progress_callback=progress_callback,
                               health_debug=health_debug)
    from woof.progress_log import ProgressOptions
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # A member run under an ensemble capture writes transient histories that
    # the ensemble retires once its products are captured, so it publishes no
    # frame markers: a marker promises a complete, readable frame, and one
    # left beside a retired history names a deleted file.  Breakage (2.9.4
    # R6, 2026-10-20): members/member-0001/ready/wrfout_*.json outlived the
    # retired member histories (test_ensemble_simulated_radar_forecast_gpu).
    step_log = ProgressOptions(progress_format="jsonl",
                               frame_markers=current_capture() is None).open(
        outdir=outdir, start_time=exp.start_time,
        run_seconds=float(exp.run_seconds),
        # One scalar for the whole run (Registry scope 1), read off the
        # root exactly as the tree runner reads it.
        adaptive_dt=bool(exp.root.run.use_adaptive_time_step))
    started = time.perf_counter()
    try:
        summary = _run_experiment(
            exp, data, outdir, restart=restart,
            progress_callback=progress_callback, health_debug=health_debug,
            step_observer=step_log.step_observer, step_log=step_log)
    except BaseException as error:
        step_log.close(status="FAIL",
                       error=f"{type(error).__name__}: {error}")
        raise
    step_log.close(status="SUCCESS")
    _write_runtime_report(outdir, exp, summary,
                          wall_seconds=time.perf_counter() - started)
    return summary


def _run_experiment(exp: ExperimentConfig, data: CaseDataConfig, outdir, *,
                    restart=None, progress_callback=None,
                    health_debug: bool = False, step_observer=None,
                    step_log=None
                    ) -> RealCaseRunSummary | ExperimentRunSummary:
    """The body of :func:`run_experiment`; ``step_observer`` reaches every
    :func:`execute_experiment` call of the run (the whole-run executor and
    each leg of a spawn walk).

    Prints the resolved-config report (G2) before any device work, then
    runs the extracted prepare/integrate pipeline with every input and
    policy drawn from the config pair.
    """
    from woof.experiment import _bind_physics_params
    exp = _bind_physics_params(exp, "woof.runtime.run_experiment")
    from woof.ensemble.runtime_context import current_session
    ensemble = current_session()
    if ensemble is not None:
        return ensemble.run_experiment(run_experiment, exp, data,
            output_directory=outdir, restart=restart,
            progress_callback=progress_callback, health_debug=health_debug)
    from woof import kernel_prewarm
    kernel_prewarm.start_plan(exp)
    from woof.io.wrfout import quarantine_orphan_wrfouts
    from woof.core.devices import refuse_unrouted_devices
    refuse_unrouted_devices(exp, "woof run")

    # THE REFUSAL THAT USED TO STAND HERE IS LIFTED.  It said this route
    # "wires no streamed-domain builder", and for two releases that was
    # true: the single-domain arm called integrate_prepared_case with
    # stepper=None and both tree arms called the executor with no
    # steppers=, so a [tiles] block was read, validated, echoed into the
    # resolved-config report and then dropped.  Both arms wire the builder
    # now (streaming.standalone_domain_builder below, builders_for_tree on
    # the tree), so the accurate answer is the run, not the refusal.
    #
    # THIS IS THE ROUTE THE UNION HAD TO LAND ON.  Two-way feedback and
    # [tiles] were disjoint: `woof run` refused [tiles] here by name,
    # and the prepared-hierarchy route refused feedback=1 at PREPARATION.
    # Only one of those two refusals was ever about wiring, and BOTH are
    # gone now -- the builders are wired below, and the prepared route
    # executes with skip_feedback_path=(feedback == 0) rather than
    # unconditionally (woof/prepared_domain_tree_forecast.py:2203,
    # woof/source_hierarchy.py:129).  So this route is no longer the
    # only address for two-way; it is the one that reaches it without a
    # preparation step.  What still refuses is the coupler, by name, for
    # the three tree shapes feedback cannot serve (woof/core/nest.py:
    # 232-249), and it refuses them identically on either route.
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    follow_configured = exp.relocation.enabled and (
        exp.relocation.follow is not None or exp.relocation.moves)
    if follow_configured and len(exp.domains) == 1:
        # A follow source needs a child to move.  Named at the front
        # door, before minutes of ingest.
        raise ValueError(
            "[relocation] configures a follow source, but this "
            "experiment has a single domain and therefore no nest to "
            "move; remove [relocation.follow]/[[relocation.move]] or "
            "add the child the follow source is for.")
    # Spawn activation is leg-boundary schedule surgery -- build_schedule
    # bakes each domain's activation tick into the per-period op lists, so
    # a trigger-driven domain cannot join a schedule already expanded, and
    # a route that reserves the nest but walks ONE execute_experiment
    # would integrate the parent alone with the nest never born.  The
    # refusal that used to stand here is lifted for the TREE path below,
    # where walk_spawn_legs drives SpawnRunner and this route's catalog
    # supplies the newborn's statics and physics.  A dormant nest is by
    # definition a second domain, so the single-domain path can never
    # carry one; it refuses here rather than reaching a walk that is not
    # wired for it.
    from woof.experiment import dormant_domain_ids
    if dormant_domain_ids(exp) and len(exp.domains) == 1:
        raise ValueError(
            "[[domain]] spawn = {...} on a single-domain experiment: a "
            "dormant nest needs a parent to be born from, and the "
            "single-domain path runs no tree walk.")
    # (The second [tiles] refusal that stood here is lifted with the first;
    # see the note at the head of this function.  What is NOT lifted is
    # streaming.make_stepper's own refusal of a STREAM decision with no
    # builder behind it -- that one is the backstop, and it stays.)
    # The land-state refusal that used to stand here is LIFTED (2026-08-07).
    # It named one gap -- "a newborn real-data nest has no defined
    # soil/land state, and how it should get one is an open physics
    # decision" -- and that decision is no longer open: it is WRF's, and
    # has been since nesting existed.  A nest with no input file of its
    # own is initialized by interpolating every field it needs from the
    # parent (Users' Guide chapter 5; med_nest_initial's unconditional
    # med_interp_domain, share/mediation_integrate.F:670), with the
    # surface/soil family going through the Registry's landmask-aware
    # interpolator (interp_mask_field:lu_index,iswater) rather than a
    # plain one -- the same operator, via the same call after each
    # shift_domain_em, that fills a moving nest's leading edge.
    # woof.ingest.nest_spawn_init.spawn_land_state_from_parent is that
    # operator, RealSpawnChildPreparer is the attachment, and both live
    # on THIS route because it holds the input catalog and the case data.
    # Routes without them keep the refusal
    # (woof.experiment.refuse_unrouted_spawn).
    experimental_feedback = feedback_provenance(exp)
    if experimental_feedback is not None:
        print(FEEDBACK_EXPERIMENTAL_WARNING)
    if exp.simulated_radar.enabled:
        # Radar admission before the fetch, the preparation and any device
        # work, for every caller of this route; the CLI door asks the same
        # inventory first (woof.config.experiment_preparation_refusals).
        from woof.simulated_radar_config import require_admitted
        require_admitted(exp)
    _preparation_progress(progress_callback, "quarantine-wrfout")
    quarantine_orphan_wrfouts(outdir)
    # THE OUTPUT DISK ADMISSION (A190), after the structural refusals above
    # so they still answer first, and before the terrain read, the
    # preparation and any device work: a run whose history, checkpoints and
    # pictures cannot fit stops here instead of partway when the disk fills.
    from woof.output_disk import require_output_space, renderer_products

    require_output_space(
        exp, outdir, restart=restart,
        render_products=renderer_products(observer=progress_callback))
    # Its own phase: reading the terrain for the acoustic substeps and the
    # long step takes seconds on a nested case, and under the phase before
    # it a run page said it was still checking the output folder.
    _preparation_progress(progress_callback, "resolve-terrain-clock")
    # THE ACOUSTIC SUBSTEPS EACH DOMAIN'S OWN GROUND NEEDS, before either
    # arm prices a tile halo or an adaptive reach from the count, and the
    # long step its ground and crest-level wind allow, before anything
    # builds a clock or a physics calendar from the step.
    exp, acoustic, terrain, grids, reach = _terrain_acoustics_for_case(
        exp, data, detail=True)
    exp, clock = _terrain_clock_for_case(exp, data, acoustic, terrain,
                                         grids, reach)
    _write_terrain_clock_receipt(outdir, clock)
    _write_acoustic_receipt(outdir, acoustic, clock)
    if len(exp.domains) == 1:
        # Frozen cardinal path: retain Task-2's exact preparation, loop,
        # output order, and single-file v3/v2 restart shims.
        _preparation_progress(progress_callback, "resolve-schedule")
        dc = single_domain(exp)
        from woof.core import streaming as _streaming
        from woof.experiment import refuse_unrouted_spectral_numerics

        if not dc.run.use_adaptive_time_step:
            refuse_unrouted_spectral_numerics(
                exp, "runtime.run_experiment:single-domain (frozen loop)")
        single_tiles = _streaming.options_for_domain(dc, exp.tiles)
        # Resolve once on the cold device. Allocation during preparation
        # must not turn a domain that fits into a different plan afterwards.
        planning_machine = _streaming.cold_planning_machine(exp)
        # THIS ARM'S ONE ADMISSION, TAKEN BEFORE THE FETCH, from the same
        # function the plan review calls
        # (:func:`woof.core.streaming.cold_single_domain_decision`).  It
        # used to be taken twice over and late: `auto` was priced from
        # `estimate_experiment` with the schedule's retained interval
        # count folded in and no device profile, AFTER the catalog was
        # built and every forcing snapshot decoded, while the review
        # priced the same domain from the shared admission estimate.
        # MEASURED on the 12 km root of the moving-nest cyclone tree on an
        # 8 GiB card, the two differ by up to 188,362,088 bytes, so there
        # was a band of budgets in which `woof check` admitted the domain
        # resident and this route then refused it once the download was
        # already spent.  The other mode took no estimate at all, which is
        # a third answer to the same question.  Whether the question is
        # asked at all is the admission function's own guard, not a
        # second copy spelled here, and the estimate is handed to the
        # decision so the route that uses it prices it once.
        resident_estimate = _streaming.cold_single_domain_admission(
            exp, machine=planning_machine, options=single_tiles)
        # The card a resident or a pinned road is admitted on, read before
        # the fetch: the planning machine when the planner already read it.
        admission_machine = _streaming.cold_admission_machine(
            planning_machine, options=single_tiles)
        single_decision = _streaming.cold_single_domain_decision(
            exp, machine=planning_machine, cfg=dc.run, options=single_tiles,
            estimate=resident_estimate)
        # A pinned tiling, priced on this card before the fetch: the
        # decision takes it from the configuration alone, and buffers that
        # cannot fit stopped in a CUDA out-of-memory while being built.
        _streaming.admit_pinned_road(dc.run, single_tiles, single_decision,
                                     machine=admission_machine)
        if not (single_decision.stream and single_decision.store == "host"):
            # THE FORECAST THIS ARM HOLDS ON THE CARD, admitted before the
            # fetch.  The preparation price admits the transforms and the
            # state; the physics driver attaches after it, and with no
            # [tiles] block nothing priced the two together, so a case whose
            # state fitted and whose physics did not (7.8 GiB more at
            # 1792x1024x55 mp=8) died in CUDA at the physics attach after
            # its download.  Same estimate the review prices.
            _streaming.admit_resident_road(
                exp, single_decision, machine=admission_machine,
                what="this run, held resident on the card")
        catalog = _runtime_input_catalog(data)
        snapshots = forcing_snapshots(data, catalog)
        times = forcing_schedule(exp, data, snapshots)
        store_direct = single_decision.stream and single_decision.store == "host"
        adaptive_clock = None
        adaptive_fingerprint = None
        if dc.run.use_adaptive_time_step:
            from woof.core.clock import resolve_clock
            from woof.core.model import experiment_fingerprint
            adaptive_clock = resolve_clock(
                exp, lbc_interval_s=_tree_forcing_cadence_seconds(catalog))
            adaptive_fingerprint = experiment_fingerprint(exp, catalog)
        print(resolved_config_report(
            exp, data, forcing_times=times, input_catalog=catalog))
        _preparation_progress(progress_callback, "prepare-case")
        if store_direct:
            from tempfile import TemporaryDirectory
            from woof.ingest.case_store import (
                CaseStoreRequest, build_case_store, initialization_resources)

            # This cache is internal and create-only. The context removes only
            # its own temporary files after the loader has closed every map.
            with TemporaryDirectory(prefix="initialization-", dir=outdir) as staging:
                prepared = prepare_experiment_case(
                    exp, data, input_catalog=catalog, forcing_by_time=snapshots,
                    store_request=CaseStoreRequest(
                        Path(staging) / "prepared",
                        resources=initialization_resources(single_tiles)))
                del snapshots, catalog
                clear_forcing_caches()
                release_backend_memory(CudaPreprocessBackend())
                single_tiles, single_decision = _refine_single_streaming_plan(
                    prepared, single_tiles, single_decision, planning_machine, resident_estimate)
                prepared, store_bundle = build_case_store(
                    prepared, valid_time=exp.start_time,
                    decision=single_decision, options=single_tiles)
        else:
            prepared = prepare_experiment_case(
                exp, data, input_catalog=catalog, forcing_by_time=snapshots)
        # The raw decode is spent here: the initial state and every
        # boundary frame are built, and this arm has no nest to re-ingest
        # for.  Both are dropped, not just the caches -- a cleared cache
        # frees nothing while a local still names the arrays.  Byte-inert:
        # these caches memoize a pure function of immutable input bytes,
        # so the only thing a later decode of the same key loses is time.
        if not store_direct:
            del snapshots, catalog
            clear_forcing_caches()
            single_tiles, single_decision = _refine_single_streaming_plan(
                prepared, single_tiles, single_decision, planning_machine, resident_estimate)
        _write_initial_perturbation_receipt(
            outdir, exp,
            ([prepared.initial_result.initial_perturbation]
             if exp.perturbation is not None else ()))
        # [tiles] on the single-domain arm.  integrate_prepared_case has
        # taken a `stepper` since the mode existed and is streaming-aware
        # throughout -- the observers are asked of the stepper, the health
        # arming, the UP_HELI_MAX reset and the restart restore all branch
        # on it -- and no caller ever supplied one.  This is that caller.
        # A domain with no tree is exactly what standalone_domain_builder
        # is for: it is a specified-boundary ROOT, the same non-nested
        # branch of prepared_domain_builder woof go's d01 takes.
        # Unconfigured, make_stepper returns woof.core.dycore.step ITSELF
        # and this run is byte-for-byte the run it always was.
        adaptive_model = None
        if adaptive_clock is not None:
            adaptive_model = _model_from_prepared_single(
                exp, prepared, adaptive_clock, adaptive_fingerprint)
        builder = (_streaming.store_domain_builder(
                       store_bundle, clock=(None if adaptive_model is None
                                            else adaptive_model.root.clock))
                   if store_direct else _streaming.standalone_domain_builder(
                       grid_id=int(dc.grid_id)))
        single_stepper = _streaming.make_stepper(
            prepared.initial_result.state, prepared.cfg, single_tiles,
            decision=single_decision, build=builder)
        if store_direct:
            prepared.initial_result.state._streamed_domain = single_stepper
        if single_tiles.enabled:
            print(single_decision.explain())
        if adaptive_model is not None:
            # Initialization above is the same resident or bounded host-store
            # route. Only the configured adaptive calendar selects this executor.
            return _run_built_experiment(
                exp, data, outdir, adaptive_model, restart=restart,
                progress_callback=progress_callback, health_debug=health_debug,
                prepared_steppers={int(dc.grid_id): single_stepper},
                prepared_decisions={int(dc.grid_id): single_decision},
                step_observer=step_observer, step_log=step_log)
        from woof.io.history_selection import resolve

        from woof.simulated_radar import run_with_radar
        summary = run_with_radar(
            integrate_prepared_case, outdir, prepared, start_time=exp.start_time,
            radar_options=exp.simulated_radar, radar_outdir=outdir,
            output_title=data.output_title, domain_id=data.output_domain,
            run_seconds=exp.run_seconds,
            history_interval_s=dc.history_interval_s,
            history_selection=resolve(exp.output, dc.output),
            history_begin_s=dc.history_begin_s, history_end_s=dc.history_end_s,
            restart_interval_s=exp.restart_interval_s,
            auto_epssm=exp.auto_epssm,
            restart_path=restart, progress_callback=progress_callback,
            health_debug=health_debug, feedback=experimental_feedback,
            stepper=single_stepper, step_observer=step_observer,
            step_log=step_log)
        _finalizing_progress(progress_callback, "provenance-receipts")
        _write_feedback_provenance_receipt(
            outdir, exp, resumed=restart is not None)
        # The fixed loop carries hashes captured before frame publication.
        frame_records = (list(summary.frame_records) if summary.frame_records else
                         _frame_records(summary.wrfout_paths,
                                        progress_callback=progress_callback))
        _finalizing_progress(progress_callback, "run-capsule")
        hydro_receipts = _hydro_run_receipts(summary.hydro_receipt)
        _, floor_receipts = _emit_front_door_capsule(
            outdir, emission_site="runtime.run_experiment:single-domain",
            exp=exp, data=data, wrfout_paths=summary.wrfout_paths,
            trajectory_digest=summary.trajectory_digest, io_mode="history",
            frame_records=frame_records,
            prepared_cases={int(dc.grid_id): prepared},
            receipts=hydro_receipts)
        return dataclass_replace(
            summary, frame_records=tuple(frame_records),
            moisture_floor_receipts=floor_receipts or None,
            hydro_receipts=hydro_receipts)

    _preparation_progress(progress_callback, "build-domain-tree")
    from woof.core.model import build_experiment

    from woof.core import streaming as _streaming
    planning_machine = _streaming.cold_planning_machine(exp)
    # THE RUN'S ONE ADMISSION, TAKEN BEFORE THE FETCH.  This route used to
    # decide the tree's roads inside the build pass below, from the model's
    # own memory LEDGER estimate against this machine -- a richer envelope
    # (the catalog's retained forcing intervals, its real lateral
    # boundaries) than the one the plan review prices from
    # (:func:`woof.core.preflight.admission_estimate`), and taken after
    # build_experiment had already fetched, decoded and ingested the whole
    # case.  MEASURED on the 12/3 km moving-nest cyclone tree those two
    # inputs move the envelope by 60,193,971 and by up to 432,788,799
    # bytes, so for any budget in between `woof check` admitted the tree
    # resident and this route then raised StreamingRefused after the
    # download was already paid for.  One call, one estimate, one budget:
    # the same function the prepared tree door and the review call, on the
    # configuration alone, before a byte is fetched, and the decision is
    # carried into the build pass rather than asked for a second time.
    cold_decisions: dict = {}
    cold_tree = _streaming.cold_tree_streaming_decision(
        exp, _streaming.cold_tree_admission_nodes(exp),
        machine=planning_machine, decisions=cold_decisions)
    if cold_tree is None:
        # NOTHING STREAMS, SO THE WHOLE TREE IS RESIDENT, and it is admitted
        # here, before the fetch and before build_experiment allocates the
        # shared workspaces and the first domain.  With no [tiles] block the
        # walk above consults nothing, and a tree too big for the card
        # stopped in a CUDA out-of-memory part way through its build.  The
        # same estimate the plan review prices (the shared admission
        # estimate, every domain's state, physics and workspaces together)
        # on the card's own profile, as the prepared tree runner and this
        # function's single-domain arm take it; --no-memory-gate skips it.
        _streaming.admit_resident_road(
            exp, None,
            machine=_streaming.cold_admission_machine(
                planning_machine, options=getattr(exp, "tiles", None)),
            what="this domain tree, held resident on the card")
    model = build_experiment(exp, data)
    print(resolved_tree_config_report(exp, data, model._input_catalog))
    return _run_built_experiment(
        exp, data, outdir, model, restart=restart,
        progress_callback=progress_callback, health_debug=health_debug,
        planning_machine=planning_machine, tree_decision=cold_tree,
        tree_decisions=cold_decisions, step_observer=step_observer,
        step_log=step_log)


def _refine_single_streaming_plan(prepared, options, decision, machine,
                                  resident_estimate=None):
    """Resolve adaptive acoustic reach on real geometry against the cold budget."""
    if not options.enabled or not prepared.cfg.use_adaptive_time_step:
        return options, decision
    from woof.core.adaptive_clock import maximum_map_factor
    from woof.core.streaming import decide

    # These are the exact FP32 map factors loaded into DomainState. A host
    # initialization has no full GPU state, so use its already resolved grid.
    factor = maximum_map_factor(geography={
        'msfu': np.asarray(prepared.grid.mapfac_u(), dtype=np.float32),
        'msfv': np.asarray(prepared.grid.mapfac_v(), dtype=np.float32),
    })
    resolved = dataclass_replace(options, acoustic_map_factor=factor)
    pricing = {} if resident_estimate is None else {"resident_estimate": resident_estimate}
    return resolved, decide(prepared.cfg, resolved, machine=machine, **pricing)


def _model_from_prepared_single(exp, prepared, tick_clock, fingerprint):
    """Bind one already initialized domain to the existing scheduled executor."""
    from types import MappingProxyType
    from woof.core.clock import build_schedule
    from woof.core.model import (DomainNode, ExperimentState,
                                  ModelRuntimeStatus, publish_declared_experiment)
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock

    dc = exp.root
    node = DomainNode(
        cfg=dc, grid=prepared.grid, state=prepared.initial_result.state,
        clock=tick_clock.clocks()[dc.grid_id], parent=None,
        children=[], coupler=None)
    node._started = True
    if prepared.streamed_store is None:
        bind_lateral_boundary_clock(node.state, node.clock)
    model = ExperimentState(
        root=node, nodes_by_grid_id=MappingProxyType({dc.grid_id: node}),
        schedule=build_schedule(exp, tick_clock), memory_ledger=None,
        experiment_fingerprint=fingerprint)
    model._scratch_arena = None
    model._dycore_state_workspace = None
    if prepared.initialization_receipt is not None:
        prepared = dataclass_replace(prepared, initialization_receipt={
            **prepared.initialization_receipt,
            'forcing_clock': 'DomainClock',
        })
    model._prepared_by_grid_id = {dc.grid_id: prepared}
    model._initial_perturbation_receipts = (
        (prepared.initial_result.initial_perturbation,)
        if exp.perturbation is not None else ())
    model._input_catalog = None
    model._runtime_status = ModelRuntimeStatus()
    model._feedback_provenance = feedback_provenance(exp)
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None
    model._last_checkpoint = None
    publish_declared_experiment(model, exp)
    return model


def _run_built_experiment(exp, data, outdir, model, *, restart=None,
                          progress_callback=None, health_debug=False,
                          prepared_steppers=None, prepared_decisions=None,
                          planning_machine=None, tree_decision=None,
                          tree_decisions=None, step_observer=None,
                          step_log=None):
    """The shared scheduled runtime, independent of initialization storage.

    ``step_observer`` reaches every :func:`execute_experiment` call (the
    whole-run executor and each leg of a spawn walk): the step log
    :func:`run_experiment` opened.
    """
    from woof.core.model import execute_experiment
    from woof.io.restart import (read_tree_lifecycle_header,
                                  restore_tree_restart, write_tree_restart)
    from woof.io.wrfout import PerDomainWrfoutWriters
    from woof.supervisor import validate_manifest_checkpoint

    # The refusal this used to be is lifted HERE and only here: this
    # route holds the input catalog (and with it the static source), so
    # the real-data relocation initializer and physics preparer exist.
    # Routes without them still refuse inside execute_experiment.
    relocation_runner = build_real_relocation_runners(
        exp, data, model, outdir)
    # Same lift, same reason, one seam over: this route holds the input
    # catalog (own-grid statics at the fired footprint) and the case data
    # and forcing calendar (the newborn's physics driver), so it can
    # activate a dormant nest instead of refusing it.
    spawn_runner = build_real_spawn_runner(exp, data, model, outdir)
    # The checkpoint writer reads the lifecycle state off the tree; this
    # route's runners are locals, and restart_handler below is handed
    # only the tree.  Published before the first checkpoint can fire.
    publish_lifecycle_runners(
        model, spawn_runner=spawn_runner,
        relocation_runner=relocation_runner,
        leg_seconds=_spawn_leg_seconds(exp))
    # ADMITTED, and said out loud.  A resume that carries lifecycle policy
    # state is not the same act as a resume that reloads arrays: it
    # reinstates which slots have fired, which are spent, how long a
    # signal has been quiet and where each follower last moved, and the
    # user who asked for it should be able to see that from the log
    # rather than infer it from a nest that did not come back.
    if admit_restart_with_lifecycle(exp, restart):
        print(f"nest lifecycle: resuming declared spawn/retire/rearm/follow "
              f"policy state from {Path(restart).name}")
    _write_initial_perturbation_receipt(
        outdir, exp, getattr(model, "_initial_perturbation_receipts", ()))
    from woof.ensemble.runtime_context import bind_current_member_model
    bind_current_member_model(model)
    restart_info = None
    lifecycle_episodes: dict[int, int] = {}
    if restart is not None:
        # VALIDATION FIRST, then the peek, then reconstruction: the tree a
        # lifecycle checkpoint restores into is the tree that checkpoint
        # describes, and the description is one JSON member off its root.
        _preparation_progress(progress_callback, "validate-checkpoint")
        restart = validate_manifest_checkpoint(restart)
        lifecycle = read_tree_lifecycle_header(restart, model)
        restart = lifecycle.root_path
        if lifecycle.block is not None:
            _preparation_progress(progress_callback, "restore-nest-lifecycle")
            lifecycle_episodes = restore_nest_lifecycle(
                model, exp, lifecycle, spawn_runner=spawn_runner,
                lbc_interval_s=_tree_forcing_cadence_seconds(
                    model._input_catalog))
            # Rebuilt over the RESTORED tree: a follow target that was
            # dormant at build time now exists, and its per-follower
            # window slot must be allocated here -- before the restore
            # applies the plane the previous run folded into it.
            relocation_runner = build_real_relocation_runners(
                exp, data, model, outdir)
            publish_lifecycle_runners(
                model, spawn_runner=spawn_runner,
                relocation_runner=relocation_runner,
                leg_seconds=_spawn_leg_seconds(exp))
            restore_nest_followers(model, lifecycle,
                                   spawn_runner=spawn_runner)
        # The live fingerprint of a run that has moved a nest is chained
        # to its move history, so a fresh build refuses its checkpoints by
        # construction.  A legitimate resume gets in by doing the same
        # arithmetic over the header's own record chain -- and by carrying
        # that chain forward, so this segment's own checkpoints stay
        # addressable by the next one.
        remark_relocation_fingerprint(model, lifecycle)
        _preparation_progress(progress_callback, "restore-tree-checkpoint")
        restart_info = restore_tree_restart(restart, model)
    # A restore whose point IS the stop tick is a finished run, not an
    # error and not an integration: the executor refuses a start period
    # at the end of the schedule, so this decides here, once.
    hydro_start_seconds = model.root.clock.elapsed_seconds
    hydro_forecast_started = time.perf_counter()
    already_complete = _restart_is_complete(restart_info)

    for node in model.walk_parent_first():
        prepared = model._prepared_by_grid_id[node.cfg.grid_id]
        if getattr(prepared, 'streamed_store', None) is not None:
            from woof.core.health import health_validator_for_domain
            from woof.ingest.case_store import write_initialization_receipt
            health = health_validator_for_domain(model, node)
            health.require_healthy(phase="initialized-or-restored")
            write_initialization_receipt(outdir, prepared, health.coverage)

    _preparation_progress(progress_callback, "initialize-domain-writers")
    with PerDomainWrfoutWriters(
            model, outdir, start_time=exp.start_time,
            title=data.output_title,
            # The tree-wide [output] history selection; each domain's own
            # `output = {...}` overrides it inside the writer set.
            history_selection=exp.output,
            simulated_radar=exp.simulated_radar, radar_output_dir=outdir,
            # A domain resumed mid-episode-2 writes d0N/episode-002/ from
            # its FIRST frame; empty on every run that is not a lifecycle
            # resume, which is the byte-inert default.
            episodes_by_grid_id=lifecycle_episodes,
            progress_callback=progress_callback) as writers:
        model._io_manager = writers
        if step_log is not None:
            # One landing slot (attach_progress_callback overwrites it), so
            # the run's callback and its step log share it: every hourly
            # wrfout gets its ``output_written`` line in progress.jsonl.
            writers.attach_progress_callback(
                _StepLogLanding(progress_callback, step_log))
        if relocation_runner is not None:
            attach_many = getattr(relocation_runner, "attach_writers", None)
            if callable(attach_many):
                attach_many(writers)
            else:
                relocation_runner.on_child_built.attach_writers(writers)
            # ...and to the descendant preparers of a MID-TREE move, which
            # are separate instances and would otherwise refresh nothing.
            # Reached through getattr twice: a runner COLLECTION carries no
            # descendant regrounder, and a run with no runner at all must
            # not be touched by this path.
            _regrounder = getattr(
                relocation_runner, "reground_descendant", None)
            _fan = getattr(_regrounder, "attach_writers", None)
            if callable(_fan):
                _fan(writers)

        def history_handler(tree, node, ticks):
            # A STREAMED domain's forecast is in its pinned host store and
            # node.state is the snapshot that filled it, so without this
            # copy every frame after the cold-start one would be the
            # initial condition under a later timestamp: correct inventory,
            # correct Times, no forecast.  The history cadence is where
            # StreamedDomain.refresh_state says the copy belongs, and it is
            # a getattr and a zero for every resident domain.
            case = model._prepared_by_grid_id[node.cfg.grid_id]
            if getattr(case, "streamed_store", None) is None:
                _streaming.refresh_streamed_state(
                    steppers.get(int(node.cfg.grid_id)), node.state)
            _submit_tree_history_frame(writers, node, ticks)

        def restart_handler(tree, ticks):
            valid = exp.start_time + timedelta(
                seconds=ticks / tree.schedule.clock.tick_den)
            # Between two model steps: its own record, sized from the
            # state it writes (see _writing_progress).
            checkpoint_started = time.perf_counter()
            with _writing_progress(
                    progress_callback, "checkpoint",
                    work_bytes=_checkpoint_work_bytes(tree)):
                tree._last_checkpoint = write_tree_restart(
                    outdir, tree, valid)
            _restart_written(
                step_log, domain_id=exp.root.grid_id, valid_time=valid,
                path=tree._last_checkpoint,
                wall_seconds=time.perf_counter() - checkpoint_started)

        # [tiles], on the SAME terms as the prepared domain-tree route
        # (woof/prepared_domain_tree_forecast.py) and for the same reason.
        # Absent -- the default -- this is an empty mapping, no planner is
        # consulted, no tilestream module is imported and the executor binds
        # woof.core.dycore.step for every grid exactly as it always did.
        # Configured, a domain the planner says will not fit resident gets a
        # streamed stepper or a loud refusal.  Before this call existed the
        # block was read, validated, echoed into the resolved-config report
        # and then IGNORED on this route: a user who wrote
        # [tiles] mode = "on" got a fully resident run with nothing
        # anywhere saying the mode never engaged.
        from woof.core import streaming as _streaming

        # THE DOOR'S DECISION, CONSUMED -- never re-taken.  ``tree_decision``
        # is the admission run_experiment took before the fetch, and the
        # decisions mapping it filled is this run's receipt source, so the
        # receipt names the road the user was shown.  ``None`` only on the
        # arms that never reached that door (the adaptive single-domain
        # hand-off below, which brings its own steppers), and then this
        # pass decides for itself exactly as it always did.
        streaming_decisions: dict = dict(tree_decisions or {})
        if prepared_steppers is None:
            steppers = _streaming.steppers_for_tree(
                model, exp.tiles,
                builders=_streaming.builders_for_tree(model, exp.tiles),
                decisions=streaming_decisions, machine=planning_machine,
                tree_decision=tree_decision,
                resident_estimate=(
                    None if tree_decision is not None else
                    getattr(model.memory_ledger, "estimate", None)))
        else:
            steppers = dict(prepared_steppers)
            streaming_decisions.update(prepared_decisions or {})
        streaming_report = _streaming.streaming_receipt(
            exp.tiles, streaming_decisions)
        if streaming_report:
            # The summary ALREADY opens with "[tiles] mode=..." -- see
            # woof.core.streaming.streaming_receipt, which builds that
            # prefix itself so every consumer of the receipt gets the tag
            # whether or not it prints one.  Adding a second here rendered
            # "[tiles] [tiles] mode='auto': ..." on the run door.  The two
            # sibling printers were never wrong because they open with
            # their own runner name instead ("prepared tree: ",
            # "prepared forecast: "), so the receipt's own tag is the one
            # thing not to repeat.
            print(f"  {streaming_report['summary']}")
        from woof.ensemble.runtime_context import (current_capture,
            observe_current_counters, observe_initial_diagnostics)
        committed_progress = progress_callback
        if current_capture() is not None:
            observe_initial_diagnostics(model)
            observe_current_counters(model, start_time=exp.start_time)
            def committed_progress(**event):
                observe_current_counters(model, start_time=exp.start_time)
                if progress_callback is not None:
                    progress_callback(**event)
        if already_complete:
            # Not a skipped run: the restore above put the finished state
            # back in memory, and everything below -- drain, digest,
            # receipts, capsule -- still runs, so this exits 0 with a
            # summary instead of refusing a run that already happened.
            print(
                "  restart point is this configuration's stop tick "
                f"({model.schedule.clock.run_ticks} ticks, "
                f"{float(exp.run_seconds):g} s of model time): the run is "
                "already complete, finalizing without integrating")
        elif spawn_runner is None:
            execute_experiment(
                model, history_handler=history_handler,
                restart_handler=restart_handler,
                progress_callback=committed_progress,
                health_debug=health_debug,
                relocation_runner=relocation_runner,
                steppers=steppers, experiment=exp,
                step_observer=step_observer)
        else:
            # The leg walk: dormant nests are born mid-run and integrate
            # from their birth boundary onward.
            walk_spawn_legs(
                model, exp, data,
                spawn_runner=spawn_runner, writers=writers,
                lbc_interval_s=_tree_forcing_cadence_seconds(
                    model._input_catalog),
                relocation_runner=relocation_runner,
                relocation_runner_factory=(
                    lambda: build_real_relocation_runners(
                        exp, data, model, outdir)),
                history_handler=history_handler,
                restart_handler=restart_handler,
                progress_callback=committed_progress,
                health_debug=health_debug,
                steppers=steppers, step_observer=step_observer)
        _finalizing_progress(progress_callback, "drain-history-writers")
        writers.drain(before_domain=_drain_progress(progress_callback))
        paths = writers.paths
    _finalizing_progress(progress_callback, "synchronize-device")
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    from woof.hydro.forecast_products import finalize_forecast as finalize_hydro
    hydro_wall = time.perf_counter() - hydro_forecast_started
    hydro_receipts = {}
    for grid_id, node in sorted(model.nodes_by_grid_id.items()):
        hydro_receipts[f"d{grid_id:02d}"] = finalize_hydro(
            node.state.physics,
            outdir=(outdir if len(model.nodes_by_grid_id) == 1 else
                    outdir / f"d{grid_id:02d}"),
            # node.cfg is the DomainConfig: its spacing and step live on .run
            # (every other reader here says dc.run.dx).  `.dx` raised
            # AttributeError at the end of every runtime-route forecast
            # (ERA5 Zhengzhou 27-9-3-1, 2026-10-19, after all 2400 steps).
            dx_m=float(node.cfg.run.dx), forecast_seconds=hydro_wall,
            atmosphere_steps=round((node.clock.elapsed_seconds - hydro_start_seconds)
                                   / float(node.cfg.run.dt)))
    # After the writer drain above and after the final device synchronization,
    # so the digest observes the trajectory and cannot participate in it.
    # A completed summary reports an observed final state, including a resume
    # already at its stop tick, which executes no new model steps. The health
    # validator follows each domain's canonical resident or streamed storage.
    final_health = _final_health_reports(model, progress_callback)
    nan_free = bool(final_health) and all(report.ok for report in final_health)
    trajectory_digest = None
    if trajectory_digest_enabled():
        _finalizing_progress(progress_callback, "trajectory-digest")
        from woof.state_digest import canonical_state_digest

        # AT THE END OF THE RUN, the other half of refresh_state's stated
        # cadence.  A whole-trajectory hash taken over a streamed domain's
        # unrefreshed DomainState is the hash of the ANALYSIS -- the digest
        # would compare equal across runs that diverged, which is the one
        # thing a digest exists to catch.  Zero and a getattr when resident.
        trajectory_digest = {}
        for grid_id, node in sorted(model.nodes_by_grid_id.items()):
            stepper = steppers.get(int(grid_id))
            declare = _digest_progress(progress_callback, grid_id)
            if _streaming.is_streaming(stepper):
                digest = stepper.canonical_digest(node.clock, scope="trajectory",
                                                  before_hash=declare)
            else:
                digest = canonical_state_digest(
                    node.state, node.clock, scope="trajectory",
                    before_hash=declare)
            trajectory_digest[f"d{grid_id:02d}"] = digest
    _finalizing_progress(progress_callback, "provenance-receipts")
    transition_path, transition_sha, transitions = \
        _write_microphysics_transition_receipt(
            outdir, model, exp, resumed=restart is not None)
    feedback_path, feedback_sha, feedback_receipt = \
        _write_feedback_provenance_receipt(
            outdir, exp, resumed=restart is not None)
    # Fresh files were hashed during writer completion. Verify those
    # revisions here and carry the records into the success capsule.
    frame_records = _frame_records(
        paths, progress_callback=progress_callback,
        completed_records=getattr(writers, "completed_records", ()))
    _finalizing_progress(progress_callback, "run-capsule")
    # Spectral run receipts bind into the capsule; a completed apply run
    # with missing step receipts refuses a clean capsule here.
    from woof.spectral_seam import seam_capsule_receipts
    hydro_run_receipts = _hydro_run_receipts(hydro_receipts)
    _, floor_receipts = _emit_front_door_capsule(
        outdir, emission_site=("runtime.run_experiment:single-domain"
                              if prepared_steppers is not None else
                              "runtime.run_experiment:domain-tree"),
        exp=exp, data=data, wrfout_paths=paths,
        trajectory_digest=trajectory_digest, io_mode="history",
        frame_records=frame_records,
        prepared_cases=getattr(model, "_prepared_by_grid_id", None),
        receipts={**seam_capsule_receipts(model),
                  "pool_trim": model._pool_trim_policy,
                  **hydro_run_receipts})
    return ExperimentRunSummary(
        wrfout_paths=paths,
        completed_seconds=model.root.clock.elapsed_seconds,
        nan_free=nan_free,
        last_checkpoint=getattr(model, "_last_checkpoint", None),
        microphysics_transitions=transitions,
        microphysics_transition_receipt=transition_path,
        microphysics_transition_receipt_sha256=transition_sha,
        feedback_provenance=feedback_receipt,
        feedback_provenance_receipt=feedback_path,
        feedback_provenance_receipt_sha256=feedback_sha,
        trajectory_digest=trajectory_digest,
        frame_records=tuple(frame_records),
        moisture_floor_receipts=floor_receipts or None,
        hydro_receipt=hydro_receipts,
        hydro_receipts=hydro_run_receipts)


def _submit_tree_history_frame(writers, node, ticks: int) -> None:
    """Production tree-history handoff, kept directly CPU-testable.

    REFL remains D2 driver-rebuilt state.  At a true period boundary the
    producing step has stashed the field, this function consumes it, and the
    restart callback drains the resulting D2H publication before writing the
    tree checkpoint.  A restored model suppresses this already-committed
    callback per due domain, so no missing stash is ever read.

    The due predicate is THE DOMAIN'S OWN, not the experiment's.  This
    read used to be ``ticks != 0``, which is the right question only for
    a domain that starts with the experiment: a nest activating later
    has its first history frame due AT its activation epoch, before any
    of its steps has run, and consuming there raised "REFL_10CM output
    is due but no microphysics-time field is stashed" and killed the run
    at that frame (#205).  ``refl_10cm_stash_is_due`` asks the same
    question against the domain's own start tick, so the root's tick-0
    analysis frame and an activating nest's activation-epoch analysis
    frame are the one case they both are -- and every frame after either
    is unchanged.
    """
    from woof.core.refl import domain_start_ticks_of, refl_10cm_stash_is_due

    refl_field = None
    if (node.state.qv is not None
            and node.state.physics is not None
            and node.state.physics.mp_physics in REFL_10CM_MICROPHYSICS):
        if refl_10cm_stash_is_due(
                ticks, domain_start_ticks=domain_start_ticks_of(node)):
            from woof.core.refl import consume_refl_10cm
            refl_field = consume_refl_10cm(node.state)
        else:
            # The domain's own analysis frame (tick 0 for the root, the
            # activation epoch for a spawned or delayed nest) writes WRF's
            # initial array, not nothing.  A newborn's birth frame used to
            # carry no REFL_10CM at all (box D 2026-10-21, storm nests d02/d03
            # at 18:15Z and 18:30Z), so its first picture had no reflectivity.
            # This also covers HRRR's GSD initial-reflectivity helper, which
            # analysis_refl_10cm consults for every domain's first frame.
            from woof.core.refl import analysis_refl_10cm
            refl_field = analysis_refl_10cm(
                node.state, shape=(node.cfg.run.nz, node.cfg.run.ny, node.cfg.run.nx))
    writers.submit(node, ticks, refl_field=refl_field)
    # History-interval reset of this domain's UP_HELI_MAX window.  Safe
    # ordering: submit's producer-stream wait_event fences the side-stream
    # D2H snapshot ahead of any later default-stream mutation, so zeroing
    # here can never race the staged copy.
    reset_history_windows_after_frame(node, ticks)


def reset_history_windows_after_frame(node, ticks: int) -> None:
    """Zero the domain's history-window maxima (UP_HELI_MAX and the hail
    window, ``uh_diag.reset_up_heli_max``) after a FULL history frame.

    A quick-look frame (``quicklook = {...}`` on the domain, so the history
    alarm rings at the quick-look cadence) does not close the window: it
    reports the running maximum and leaves it running, so the next full
    frame's maxima cover the whole full interval.  Resetting at every
    alarm made a full frame's UH and hail maxima cover only the last
    quick-look interval (release review 2.9.7 HIGH 5).  With no quick-look
    stream every frame is full and this is the reset it always was.
    """
    from woof.io.wrfout import quicklook_selection
    if quicklook_selection(node, ticks) is not None:
        return
    from woof.core.uh_diag import reset_up_heli_max
    reset_up_heli_max(node.state)


def resolved_tree_config_report(exp: ExperimentConfig,
                                data: CaseDataConfig, catalog) -> str:
    """Compact resolved report for the multi-domain run surface."""
    lines = [f"resolved experiment configuration -- {exp.name}"]
    if exp.feedback == 1:
        lines.append("  feedback = experimental")
    for record in data.resolved_inputs():
        lines.append(f"  input.{record.role} = {record.path}")
    lines.extend((
        f"  time.start_time = {exp.start_time.isoformat()}",
        f"  time.run_seconds = {exp.run_seconds:g}",
        f"  time.restart_interval_s = {exp.restart_interval_s:g}",
        f"  radiation.column_chunk = {exp.column_chunk}",
        f"  input_catalog.sha256 = {catalog.fingerprint}",
        f"  domains = {len(exp.domains)}",
    ))
    for dc in exp.domains:
        lines.append(
            f"  domain.d{dc.grid_id:02d} = parent={dc.parent_id} "
            f"start={exp.domain_start_time(dc.grid_id).isoformat()} "
            f"{dc.run.nx}x{dc.run.ny} dx={dc.run.dx:g} "
            f"dt={dc.run.dt:g} history={dc.history_interval_s:g} "
            f"mp_physics={dc.run.mp_physics} "
            "nest_microphysics_transition="
            f"{dc.run.nest_microphysics_transition}")
        # Per domain, not once for the root: child domains may resolve a
        # different ra_lw_physics, and this receipt's job is to name
        # where EACH domain's downward longwave comes from.
        lines.append(
            f"  domain.d{dc.grid_id:02d}.radiation.downward_longwave = "
            + downward_longwave_source(exp, dc.run))
    lines.append(
        "  output.filename_pattern = wrfout_d0X_<YYYY-MM-DD_HH_MM_SS>")
    return "\n".join(lines)


__all__ = [
    "CONSERVATION_CLOSURE_RECEIPT_NAME",
    "ExperimentRunSummary", "FEEDBACK_EXPERIMENTAL_WARNING",
    "FEEDBACK_PROVENANCE_RECEIPT_NAME",
    "MICROPHYSICS_TRANSITION_RECEIPT_NAME",
    "PreparedRealCase", "RealCaseRunSummary",
    "configured_run_schedule",
    "experiment_grid", "feedback_provenance",
    "forcing_decode_report", "forcing_schedule", "forcing_snapshots",
    "integrate_prepared_case", "load_source_orography",
    "prepare_child_case", "prepare_experiment_case",
    "declared_constant_glw", "downward_longwave_source",
    "case_static_fields",
    "prepare_root_experiment_case", "prepare_real_case", "refl_10cm_due",
    "resolved_config_report", "resolved_tree_config_report",
    "restart_outer_steps", "run_experiment",
    "single_domain", "vertical_coord_for", "whole_step_count",
    "write_case_output", "write_ingest", "write_static",
]
