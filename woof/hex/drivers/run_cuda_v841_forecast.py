#!/usr/bin/env python3
"""Engineering forecast driver for the MPAS-A v8.4.1 CUDA port.

DERIVED, NOT A PROOF.  This tool is a parameterized fork of the sealed proof
harness ``src/hexcore/drivers/run_cuda_v841_full_physics_x4.py``.  It exists because that
harness is a one-case proof: it pins the init file's SHA-256, its start time,
its step count and its per-case F000 surface-diagnostic bytes, and it spends
half its run comparing against a sealed native CPU authority that exists for
exactly one case.  None of that can execute a forecast from a different
initial condition.

THE FORK IS IMPORT-BASED ON PURPOSE.  Every executing model path -- mesh load,
overlays, constructor arrays, the staged two-owner composite step, the physics
configuration, the snapshot capture -- is CALLED FROM the proof module, not
copied.  Two functions are re-implemented here (``_prepare_host_execution`` and
``build_arwen_constructor_values``) because they carry case literals in their
bodies rather than in module constants; both re-implementations are faithful
transcriptions with the case assertions replaced by measurement, and the
values they hand to the model are constructed identically.  The proof module's
file bytes are never modified and its own entry point is unaffected.

WHAT IS KEPT (unchanged, still enforced, refusal on violation)
  * the engine seam pin: the sixteen seam files verified before and after
    the run by ``verify_arwen_checkout_git``, which records a git tree by
    HEAD/tree/clean and an install by version and RECORD digest; the
    receipt names the engine the bytes matched
    (:func:`measured_engine_identity`);
  * ``EXECUTION_SOURCE_PINS``: exact SHA-256 of every executing port module,
    verified before CUDA is imported and again after the forecast;
  * the mesh authority pins: grid and static file byte counts and SHA-256;
  * the rotation-aware precision-preserving mesh/static pair load, and both
    in-memory init overlays (reconstruction coefficients, edge normals) with
    every structural check they carry (topology equality with the prepared
    mesh, exact +0 padding, finiteness, placeholder provenance);
  * the full physics configuration exactly as proven:
    ``V841MpasColumnPhysicsSmagorinskyGwdoConfig`` by default (native's
    Registry deformation-based 2-D Smagorinsky horizontal mixing; a NEW
    sub-series, not bit-comparable to mixing-off arms), or the pre-mixing
    ``V841MpasColumnPhysicsGwdoConfig`` control under ``--horiz-mixing off``;
    both carry WSM6 + GF + YSU + external YSU-GWDO +
    revised-MO + NoahMP (+ glacier dispatch) + cloud fraction + legacy RRTMG,
    at the run's timestep and schedule -- MPAS-A's Registry defaults unless
    the caller names others (RK order 2, four acoustic sub-steps, three
    splits, radiation 1800 s, surface/PBL and cumulus every step;
    woof.hex.mpas_defaults) -- six-species scalar order,
    ``wsm6_hail_opt = 0``, ``xice_threshold = 0.02``;
  * the staged two-owner transaction with its rollback contract
    (``execute_composite_step``), including the no-fail commit law;
  * the surface classification receipt and the NoahMP census/glacier-path
    check at every capture (the census is now measured from this init instead
    of pinned to the proof case, but a backend census that disagrees with the
    host classification still refuses);
  * the physical snapshot gate (finite everywhere, rho/theta/pressure > 0,
    soil moisture in [0,1], non-negative precipitation, hydrometeors >= 0);
  * the exact snapshot capture and its native-grid history writer.

WHAT IS REMOVED, AND WHY (every removal deliberate; guarantees dropped)
  1. THE INIT SHA-256 PIN.  ``AUTHORITY_PINS['init']`` fixes one file.  A
     forecast from another initial condition cannot satisfy it.  REPLACED BY:
     the actual init path, byte count and SHA-256 are measured and recorded in
     the receipt, and the receipt names the init's stated source.
     GUARANTEE DROPPED: this tool cannot tell you the init is the blessed one.
     It tells you exactly which bytes it ran.
  2. THE CASE-PINNED F000 SURFACE-DIAGNOSTIC PINS.  ``t2m``/``u10``/``v10``
     array SHA-256 for the proof case.  REPLACED BY: the same three fields are
     read from the supplied init, still required FP32 [Time,nCells], finite,
     and still overlaid only onto verified exact +0 placeholders; their
     digests are recorded.  GUARANTEE DROPPED: byte identity of the F000
     diagnostic overlay to the proof case.
  3. THE NATIVE-COMPARISON STAGE.  ``compare_snapshot_to_native`` and the six
     ``native_*`` authority files (F000/F030/F001 history, validation receipt,
     launch receipt, run closure), plus their RMSE gates.  Those files are the
     CPU authority for ONE case at ONE valid time; there is no such authority
     for any other case, and comparing a new forecast against them would be
     meaningless.  GUARANTEE DROPPED: no quantitative agreement with a native
     MPAS CPU run is established for these forecasts.  Nothing here is
     verified against native MPAS.
  4. THE CHECKPOINT/RESTART PROOF STAGE.  The F030 host checkpoint, the fresh
     restart worker process, F030 rehydration identity, first-resumed-step-16
     identity and the F001 bitwise restart comparison.  A forecast runs
     uninterrupted; the restart property is a property of the port and was
     proven by the release proof.  GUARANTEE DROPPED: these runs do not
     re-establish restart bitwise identity (and produce no checkpoint).
  5. THE CASE-PINNED INITIAL-CONTENT FINGERPRINTS, each replaced by a measured
     value recorded in the receipt:
       - ``NEGATIVE_QV_PIN`` (215 negative qv values at F000 in the proof
         case).  The count is now measured from this init; the physical gate
         still requires exactly that many at F000 and exactly zero after every
         subsequent step, so a clamp regression still refuses.
       - ``INIT_RECONSTRUCTION_COEFFICIENTS_PIN.init_carrier_raw_sha256`` and
         ``INIT_EDGE_NORMAL_VECTORS_PIN.init_carrier_raw_sha256`` plus the
         edge-normal activity counts and norm envelope.  These are mesh
         geometry regenerated by ``init_atmosphere`` per init, so they are
         recorded, not pinned.  ADDED IN THEIR PLACE: an explicit physical
         bound, every edge-normal row norm within 1e-4 of unity, which the
         pinned envelope previously implied.
       - ``EXPECTED_SURFACE_CLASSIFICATION`` / ``EXPECTED_NOAHMP_CENSUS`` /
         the xland, glacier-index, sea-ice-index and threshold-delta digests.
         Land/water/sea-ice/glacier counts are date-dependent (sea ice moves).
         Measured, recorded, and the backend's own census must still equal the
         host classification at every capture.
       - ``LANDMASK_CONSTRUCTOR_CAST_PIN``.  The exactness of the int32 ->
         FP32 cast is still checked (round-trip equality and the {0,1} value
         set); only the case digests are recorded instead of pinned.
       - ``EXPECTED_ARWEN_P_TOP_PA_F32`` / ``EXPECTED_TOP_PRESSURE_RANGE_PA``.
         p_top is derived from the init's own pressure field.  This driver
         re-derives the expectation with the identical FP32 reduction and
         seeds it; the proof function then recomputes it independently and
         still asserts equality, so a transcription error refuses instead of
         passing.  The derived scalar and the per-column min/median/max are
         recorded.
       - ``START_TIME_TEXT``.  The init's ``config_start_time`` is now the
         authority; ``--start-time`` is an assertion against it, not a source.
  6. THE FIXED 30-STEP / F000-F030-F001 SCHEDULE.  Replaced by ``--hours`` and
     ``--history-every-minutes``.  dt stays 120 s and every physics cadence
     stays as proven; only the number of steps and the capture set move.
     GUARANTEE DROPPED: the proof's three-snapshot schedule and its labels.

WHAT IS ADDED
  7. THE F000 START-FRAME COMPLETION (:func:`complete_f000_surface_diagnostics`).
     The proof overlays ``t2``/``u10``/``v10`` from the init onto the seam's
     pre-first-call placeholders and publishes ``q2`` and ``psfc`` as the
     placeholders themselves (+0 and 100000 Pa), because the native F000 it
     is compared against carries the same.  A forecast's start frame is a
     product, so this driver completes it the way the regional engine writes
     its own hour 0: ``q2`` from the start file when the start file carries
     one, otherwise from the lowest model level, and ``psfc`` from the model's
     own surface pressure.  Every later frame is untouched, and so is the
     model state.
  8. THE F000 FIRST-CALL DIAGNOSTICS (:func:`complete_f000_from_first_call`,
     ``woof.hex.f000_first_call``).  The start frame is captured before step 1
     and finished after it: where the start file carries no 10 m wind or 2 m
     temperature (the HRRR road writes the -1e30 missing value), and for the
     fluxes and PBL depth (pre-first-call zeros), the frame takes the first
     physics call's diagnosis of the start state, by the forecast's own
     surface-layer scheme.

FORK-EQUIVALENCE GATE.  Because this driver reaches the model through the
proof module's own functions, a fork defect would show up as a changed
trajectory.  ``tools/gate_v841_forecast_fork_equivalence.py`` runs THIS driver
on the AUTHORITY init for 30 steps and compares its boundary fingerprints and
snapshot array digests bitwise against the release-proof uninterrupted arm.
BITWISE-IDENTICAL is required before any showcase forecast is run.

The claim, and the non-claims, are stated in the receipt.  These are
engineering forecasts on a 92-to-25 km variable-resolution global mesh.  They
establish no forecast skill.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields as dataclass_fields
from datetime import datetime, timedelta
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path
import sys
import time
import traceback
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from woof.hex.drivers import run_cuda_v841_full_physics_x4 as proof  # noqa: E402
from woof.hex import dt_admission  # noqa: E402
from woof.hex import shipped_sources  # noqa: E402
from woof.hex import history_writer, kernel_prewarm, mesh_snapshot, prep_cache  # noqa: E402
from woof.hex import time_ledger  # noqa: E402
from woof.hex import f000_first_call  # noqa: E402
from woof.hex.errors import ConfigurationRefusal  # noqa: E402
from woof.hex.species_row import species_row_for_scheme as _row_for_scheme  # noqa: E402
from woof.hex.mesh import (  # noqa: E402
    REGIONAL_BOUNDARY_MASK_NAMES,
    REGIONAL_BOUNDARY_ZONE_WIDTH,
    regional_boundary_mask_digest,
)

ROOT = proof.ROOT

SCHEMA = "mpas-port.cuda-v841-engineering-forecast/v1"

#: How an IAU increment enters, and the route a run takes when it names none (one table, the door's).
from woof.hex.analysis_door import IAU_ROUTE_DEFAULT, IAU_ROUTES  # noqa: E402
RECEIPT_MODE = "engineering-forecast"

#: The start record (which mesh and init, start, hours, cadence), written before the first step.
#: One writer for both physics routes: woof.hex.forecast_start.
from woof.hex.forecast_start import START_RECORD_NAME, START_RECORD_SCHEMA, write_start_record  # noqa: E402,F401


def restore_refused_dycore_values(rollback: Any, step: int) -> None:
    """Copy a committed boundary's dycore state VALUES back into the driver's own arrays.

    F1 follow-through (2026-10-15). CUDA graph replay returns the graph's own
    output buffers, so the committed MPAS state shares memory with the graph,
    and a refused step's replays overwrite it. The step's transaction rollback
    restores the start-state identity, not those values. Breakage it prevents:
    a --stop-on-refusal run with graphs ending on the refused step's dynamics
    (final digest, any frame written from it) where eager ends on the committed
    step. ``rollback`` is the armed TwoDeepForecastSnapshotV841, whose slot for
    ``step`` was captured before the refused step. Identities are kept; only
    bytes are copied.
    """
    slot = rollback.slots[step % 2]
    if slot.get("step") != step:
        raise RuntimeError(f"refusal restore lost committed boundary {step}")
    driver = rollback.stack["driver"]
    for part in ("state", "saved"):
        source, target = getattr(slot["atmosphere"], part), getattr(driver.atmosphere, part)
        for field in dataclass_fields(source):
            value = getattr(source, field.name)
            if isinstance(value, rollback.cp.ndarray):
                rollback.cp.copyto(getattr(target, field.name), value)


RECEIPT_NAME = "cuda-v841-forecast-receipt.json"


def _phase_release_receipt() -> dict[str, Any]:
    """What woof.hex.device_memory_release gave back over this run."""

    from woof.hex.device_memory_release import process_release

    return process_release().receipt()


DERIVED_FROM = "src/hexcore/drivers/run_cuda_v841_full_physics_x4.py"

N_CELLS = proof.N_CELLS
N_EDGES = proof.N_EDGES
N_LEVELS = proof.N_LEVELS
N_INTERFACES = proof.N_INTERFACES
N_SOIL_LEVELS = proof.N_SOIL_LEVELS
DT_SECONDS = proof.DT_SECONDS
SCALAR_NAMES = proof.SCALAR_NAMES
COLD_ZERO_SCALAR_NAMES = proof.COLD_ZERO_SCALAR_NAMES
SOURCE_SCALAR_NAMES = proof.SOURCE_SCALAR_NAMES
NOMINAL_DX_M = proof.NOMINAL_DX_M
ARWEN_XICE_THRESHOLD = proof.ARWEN_XICE_THRESHOLD

#: The run's cumulus selection, and why it was made.  ``bind_mesh`` rebinds
#: this from the BOUND mesh's own finest spacing, exactly as it rebinds
#: DT_SECONDS -- one decision, one source.  The value standing here before a
#: bind is the native x4 configuration this module's constants describe, so a
#: direct invocation with no bind behaves as it always did.
CONVECTION_DECISION: dict[str, Any] = {}

#: The run's surface/PBL cadence, and why it was chosen.  Travels the same
#: road as DT_SECONDS and CONVECTION_DECISION: ``bind_mesh`` takes the
#: decision once from the bound row's own timestep and rebinds it here, and
#: the driver refuses if its own request disagrees.  Empty before a bind
#: means the weld -- ``config_bldt_seconds = config_dt``, the proven
#: configuration -- so a direct invocation with no bind behaves as it always
#: did.  See :mod:`woof.hex.pbl_cadence`.
PBL_CADENCE_DECISION: dict[str, Any] = {}

# Mesh authority roles kept under exact-byte pins.  ``init`` is deliberately
# absent: see removal 1.
MESH_AUTHORITY_ROLES = ("grid", "static")

EDGE_NORMAL_UNIT_TOLERANCE = 1.0e-4
VERTICAL_VELOCITY_REFUSAL_M_S = 200.0

DROPPED_GUARANTEES = (
    "init identity is recorded, not pinned: this tool runs whatever init bytes "
    "it is given and reports their SHA-256",
    "no comparison against a native MPAS CPU run is performed; the sealed "
    "native GF+YSU-GWDO authority applies to one case only",
    "no checkpoint is written and restart bitwise identity is not "
    "re-established by these runs",
    "the F000 surface-diagnostic overlay, the initial negative-qv fingerprint, "
    "the init-carried mesh-geometry digests, the surface classification and "
    "NoahMP census, the landmask cast digests and p_top are measured from the "
    "supplied init and recorded rather than pinned to the proof case",
    "forecast skill is not established; this is a 92-to-25 km variable-"
    "resolution global mesh and 25 km is its fine limit, not a nest",
)

CLAIM = (
    "one uninterrupted real-initialized x4.163842 MPAS-A v8.4.2 CUDA forecast "
    "using WSM6 + GF + YSU + external YSU-GWDO + revised-MO + NoahMP "
    "(with the CUDA glacier path) + cloud fraction + legacy RRTMG, at "
    "dt = 120 s, from the init named in this receipt"
)
_FROZEN_CLAIM_SCHEMES = re.compile(
    r"^WSM6 \+ (?:(?P<cumulus>.+?) \+ )?YSU \+ external YSU-GWDO \+ .* \+ legacy RRTMG$"
)


def claim_for_physics(claim: str, physics: Any) -> str:
    """The forecast claim with its physics named as this run resolved it.

    THE BREAKAGE THIS PREVENTS: the claim sentence lists the frozen column
    batch's schemes ("WSM6 + ... + YSU + external YSU-GWDO + revised-MO +
    NoahMP + cloud fraction + legacy RRTMG").  A selection that runs on
    another seam kept that sentence: the 30 km RRTM + Dudhia and analytic
    radiation runs of 2026-10-11 both carried receipts claiming legacy RRTMG
    and the external YSU drag, neither of which ran.  A frozen-seam run's
    claim is left byte-identical.  Nothing here lists a scheme: the selectors
    are the run's own resolution (woof.hex.physics_suites) and the engine
    assignments are what the column seam was handed.
    """

    if physics is None or getattr(physics, "seam", "frozen") == "frozen":
        return claim
    head, using, tail = claim.partition(" using ")
    body, at_dt, rest = tail.partition(", at dt")
    match = _FROZEN_CLAIM_SCHEMES.match(body) if using and at_dt else None
    if match is None:
        return claim
    cumulus = match.group("cumulus") or "no cumulus scheme"
    selection = ", ".join(
        f"{key}={value}" for key, value in physics.selectors.items()
        if key != "config_convection_scheme"
    )
    engine = getattr(physics, "engine_physics", None)
    handed = "" if not engine else (
        "; the engine's column seam was handed "
        + ", ".join(f"{key}={engine[key]}" for key in sorted(engine))
    )
    return (
        f"{head} using {cumulus} for convection and the {physics.seam} seam's "
        f"physics selection {selection}{handed}{at_dt}{rest}"
    )


NONCLAIMS = proof.NONCLAIMS + (
    "not a proof: the init pin, the native comparison stage and the "
    "checkpoint/restart stage of the sealed harness are deliberately absent "
    "(see dropped_guarantees)",
)


# --------------------------------------------------------------------------
# authority verification (grid/static pinned, init recorded)
# --------------------------------------------------------------------------
def verify_forecast_authorities(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Pin the mesh, record the init."""

    if set(paths) != {"grid", "static", "init"}:
        raise ValueError("forecast authority roles are exactly grid, static, init")
    files: dict[str, Any] = {}
    for role in MESH_AUTHORITY_ROLES:
        files[role] = proof._file_record(role, paths[role], proof.AUTHORITY_PINS[role])
    init = proof._plain_absolute(paths["init"], "init")
    if not init.is_file():
        raise FileNotFoundError(f"missing init: {init}")
    files["init"] = {
        "path": str(init),
        "bytes": init.stat().st_size,
        "sha256": proof.sha256_file(init),
        "pinned": False,
        "policy": "recorded, not pinned (derived-driver removal 1)",
    }
    return {
        "files": files,
        "mesh_roles_pinned": list(MESH_AUTHORITY_ROLES),
        "init_role_pinned": False,
        "sha256": proof.canonical_json_sha256(files),
    }


# --------------------------------------------------------------------------
# case-pin relaxation: measure from this init, then rebind the proof module's
# case constants so its own functions execute unchanged against this case.
# Every substitution is recorded in the receipt.
# --------------------------------------------------------------------------
def _raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def relax_init_carrier_pins(init_path: Path) -> dict[str, Any]:
    """Rebind the two init-carried mesh-geometry digests and F000 diagnostics."""

    from netCDF4 import Dataset

    record: dict[str, Any] = {}
    with Dataset(init_path, "r") as dataset:
        coefficients = proof._read_exact_variable(
            dataset,
            "coeffs_reconstruct",
            dtype=np.float32,
            dimensions=tuple(proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN["dimensions"]),
        )
        normals = proof._read_exact_variable(
            dataset,
            "edgeNormalVectors",
            dtype=np.float32,
            dimensions=tuple(proof.INIT_EDGE_NORMAL_VECTORS_PIN["dimensions"]),
        )
        surface_diagnostics = {}
        for target, pin in proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
            value = proof._read_exact_variable(
                dataset,
                str(pin["source"]),
                dtype=np.float32,
                dimensions=("Time", "nCells"),
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{pin['source']} shape changed: {value.shape}")
            surface_diagnostics[target] = proof.array_sha256(
                np.ascontiguousarray(value[0], dtype=np.float32)
            )

    # --- reconstruction coefficients: only the init carrier digest moves.
    reconstruction = dict(proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN)
    removed_reconstruction_pin = str(reconstruction["init_carrier_raw_sha256"])
    measured_reconstruction = _raw_sha256(coefficients)
    reconstruction["init_carrier_raw_sha256"] = measured_reconstruction
    proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN = MappingProxyType(reconstruction)
    record["reconstruction_coefficients"] = {
        "removed_proof_pin_sha256": removed_reconstruction_pin,
        "init_carrier_raw_sha256": measured_reconstruction,
        "structural_checks_retained": [
            "dtype/shape",
            "finiteness",
            "topology equality with the prepared mesh",
            "no active 3-vector is the all-+0 static placeholder",
            "padding bitwise +0",
        ],
    }

    # --- edge normals: carrier digest, activity counts and norm envelope move.
    normals = np.ascontiguousarray(normals)
    nonzero = int(np.count_nonzero(normals))
    zeros = int(normals.size - nonzero)
    zero_rows = int(np.count_nonzero(np.all(normals == np.float32(0.0), axis=1)))
    normals64 = normals.astype(np.float64)
    norms = np.sqrt(np.sum(normals64 * normals64, axis=1, dtype=np.float64))
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    # Replacement physical bound for the removed exact envelope pin.
    if (
        not math.isfinite(norm_min)
        or not math.isfinite(norm_max)
        or abs(norm_min - 1.0) > EDGE_NORMAL_UNIT_TOLERANCE
        or abs(norm_max - 1.0) > EDGE_NORMAL_UNIT_TOLERANCE
    ):
        raise RuntimeError(
            "initialized edge normals are not unit vectors: "
            f"norm envelope {(norm_min, norm_max)}"
        )
    edge = dict(proof.INIT_EDGE_NORMAL_VECTORS_PIN)
    removed_edge_pin = str(edge["init_carrier_raw_sha256"])
    edge["init_carrier_raw_sha256"] = _raw_sha256(normals)
    edge["nonzero_components"] = nonzero
    edge["exact_zero_components"] = zeros
    edge["zero_rows"] = zero_rows
    edge["float64_norm_min"] = norm_min
    edge["float64_norm_max"] = norm_max
    proof.INIT_EDGE_NORMAL_VECTORS_PIN = MappingProxyType(edge)
    record["edge_normal_vectors"] = {
        "removed_proof_pin_sha256": removed_edge_pin,
        "init_carrier_raw_sha256": edge["init_carrier_raw_sha256"],
        "nonzero_components": nonzero,
        "exact_positive_zero_components": zeros,
        "zero_rows": zero_rows,
        "float64_norm_min": norm_min,
        "float64_norm_max": norm_max,
        "replacement_gate": (
            f"every row norm within {EDGE_NORMAL_UNIT_TOLERANCE} of unity"
        ),
    }

    # --- F000 surface diagnostics.
    removed_surface_pins = {
        target: str(pin["sha256"])
        for target, pin in proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items()
    }
    pins = {
        target: {"source": pin["source"], "sha256": surface_diagnostics[target]}
        for target, pin in proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items()
    }
    proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS = MappingProxyType(
        {name: MappingProxyType(value) for name, value in pins.items()}
    )
    record["f000_surface_diagnostics"] = {
        "measured": pins,
        "removed_proof_pins": removed_surface_pins,
    }
    return record


def relax_negative_qv_pin(state: Any) -> dict[str, Any]:
    """Measure this init's F000 negative-qv fingerprint and rebind the pin."""

    fingerprint = proof.negative_qv_fingerprint(np.asarray(state.scalars)[0])
    proof.NEGATIVE_QV_PIN = MappingProxyType(dict(fingerprint))
    return dict(fingerprint)


def relax_surface_classification(
    classification: Mapping[str, Any], glacier_path: str
) -> dict[str, Any]:
    """Rebind the classification/census expectations to this init's counts."""

    core = {
        name: classification[name]
        for name in proof.EXPECTED_SURFACE_CLASSIFICATION
    }
    proof.EXPECTED_SURFACE_CLASSIFICATION = MappingProxyType(dict(core))
    census = {
        "land": int(core["sflx_land_columns"]),
        "water": int(core["open_water_columns"]),
        "sea_ice": int(core["sea_ice_columns"]),
        "glacier": int(core["glacier_columns"]),
    }
    # The glacier kernel's provenance is reported by the seam only when the
    # seam ran it, which it does only when the domain HAS glacier columns.
    # MEASURED (2026-08-26, r4.75.11020): a placed swath over the Southern
    # Ocean has none, and demanding the provenance anyway refused its first
    # step with "NoahMP census/provenance changed" -- an all-ocean domain
    # failing for not naming the source file of a scheme it correctly never
    # called.  The count is still checked in both directions, so a domain
    # that HAS glaciers still has to say which kernel ran on them.
    if int(core["glacier_columns"]) > 0:
        census["glacier_path"] = glacier_path
    proof.EXPECTED_NOAHMP_CENSUS = MappingProxyType(dict(census))
    return {"surface_classification": dict(core), "noahmp_census": dict(census)}


def _suite_adapter(row: Any):
    """The row's adapter when it is a column-suite row, else None."""

    from woof.hex.physics_backend_admission import DEFAULT_BACKEND

    if row.name == DEFAULT_BACKEND:
        return None
    adapter = row.load_adapter()
    return adapter if hasattr(adapter, "soil_levels_for") else None


def _rebind_soil_levels(levels: int) -> None:
    """Rebind the soil width the init read, the snapshot writer and the
    history files size from, exactly as bind_mesh rebinds the mesh's own
    counts: one number from the selected land surface, every reader."""

    global N_SOIL_LEVELS
    N_SOIL_LEVELS = int(levels)
    proof.N_SOIL_LEVELS = int(levels)


def _suite_host(
    adapter: Any,
    row_name: str,
    values: Mapping[str, Any],
    receipt: dict[str, Any],
    *,
    orographic_statistics: Path | None,
    is_regional: bool,
    cell_area_m2=None,
) -> tuple[dict[str, Any], None, dict[str, Any]]:
    """The suite row's sealed-constructor mapping, drag statistics included.

    The frozen door's mapping (already on the padded extent for a regional
    mesh) is re-shaped by the row's adapter; the drag statistics are read
    by the engine's Rust NetCDF bridge and put on the same extent.  The
    hex-side YSU-GWDO statics are dropped: the suite's drag runs inside the
    engine's PBL slot.
    """

    suite = adapter.physics_suite_record(row_name)
    receipt["soil_geometry"] = adapter.require_init_soil_geometry(
        receipt["init_path"], row_name=row_name
    )
    oro = None
    if suite["drag_reads_statistics"]:
        if orographic_statistics is None:
            raise ConfigurationRefusal(
                "orographic_statistics",
                None,
                f"--physics-backend {row_name} runs the "
                f"{suite['selection']['drag_scheme']} drag, which reads the "
                "mesh's orographic statistics; without them the drag would run "
                "on zeros, which turns it off without saying so",
                "--orographic-statistics <mesh>.ugwp_oro_data.nc",
            )
        oro, oro_receipt = adapter.read_orographic_statistics(
            orographic_statistics, row_name=row_name, n_cells=int(N_CELLS)
        )
        if is_regional:
            from woof.hex.cuda_regional_forecast_v841 import (
                pad_regional_physics_column,
            )

            oro = {
                name: pad_regional_physics_column(value)
                for name, value in oro.items()
            }
            oro_receipt["regional_pad"] = "duplicate-last-real-column"
        receipt["orographic_statistics"] = oro_receipt
    # The suite rows' sealed mapping is the engine's own initialize_physics
    # keyword set, which has no shdmax; the frozen Noah-MP row binds it after
    # construction.  A suite row keeps the engine's fill for now and says so.
    if "shdmax" in values:
        values = {key: value for key, value in values.items() if key != "shdmax"}
        receipt["noahmp_shdmax"] = (
            "not bound on suite rows: the generic engine entry takes no "
            "shdmax (follow-up: carry it in the suite seam's statics)"
        )
    mapped = adapter.constructor_values_for(
        values, row_name=row_name, orographic_statistics=oro,
        cell_area_m2=cell_area_m2,
    )
    classification = dict(
        adapter.seal_constructor(mapped).expected_surface_classification()
    )
    receipt["surface_classification"] = classification
    receipt["physics_suite"] = dict(suite)
    return mapped, None, classification


def _suite_record_for_run(record: Mapping[str, Any], physics: Any) -> dict[str, Any]:
    """The profile record as this run executed it.

    A named suite records the MPAS spelling it was resolved from (the
    fork's bl_mynnedmf for the engine's MYNN, say), which the configuration
    also records; the profile row keeps its own spelling.  The receipt's
    record names the run's, so the two cannot disagree in one receipt.
    """

    out = dict(record)
    if physics is not None:
        out["profile_mpas_selectors"] = dict(out["mpas_selectors"])
        out["mpas_selectors"] = {
            key: value for key, value in physics.selectors.items()
            if key != "config_convection_scheme"
        }
        out["named_suite"] = physics.suite
    return out


def relax_suite_surface_classification(
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebind the classification expectation for a suite row.

    The land surface publishes its own census (RUC's land/water/lake/sea-ice
    counts), which the snapshot receipts carry under their own key; the
    Noah-MP census expectation is retired for the run (None), so no frame
    is checked against a census its land surface never writes.
    """

    proof.EXPECTED_SURFACE_CLASSIFICATION = MappingProxyType(dict(classification))
    proof.EXPECTED_NOAHMP_CENSUS = None
    return {"surface_classification": dict(classification), "noahmp_census": None}


def seed_p_top_expectation(
    *,
    pressure_base: Any,
    pressure_perturbation: Any,
    zgrid: Any,
    area_cell: Any,
) -> dict[str, Any]:
    """Re-derive the FP32 area-weighted p_top and seed the proof expectation.

    This is a faithful transcription of the reduction in
    ``proof.derive_area_weighted_p_top_v841``.  It only SEEDS the expectation;
    the proof function then recomputes the same quantity independently and
    still asserts equality, so a transcription error refuses rather than
    silently passing.
    """

    base = np.asarray(pressure_base)
    perturbation = np.asarray(pressure_perturbation)
    height = np.asarray(zgrid)
    area = np.asarray(area_cell)
    pressure = np.add(base, perturbation, dtype=np.float32)
    half = np.float32(0.5)
    one = np.float32(1.0)
    z0 = height[-1]
    z1 = np.multiply(
        half, np.add(height[-1], height[-2], dtype=np.float32), dtype=np.float32
    )
    z2 = np.multiply(
        half, np.add(height[-2], height[-3], dtype=np.float32), dtype=np.float32
    )
    w1 = np.divide(
        np.subtract(z0, z2, dtype=np.float32),
        np.subtract(z1, z2, dtype=np.float32),
        dtype=np.float32,
    )
    w2 = np.subtract(one, w1, dtype=np.float32)
    logarithm = np.add(
        np.multiply(w1, np.log(pressure[-1], dtype=np.float32), dtype=np.float32),
        np.multiply(w2, np.log(pressure[-2], dtype=np.float32), dtype=np.float32),
        dtype=np.float32,
    )
    top = np.ascontiguousarray(np.exp(logarithm, dtype=np.float32))
    if not np.all(np.isfinite(top)) or np.any(top <= 0):
        raise FloatingPointError("derived F000 top pressure is invalid for this init")
    area64 = area.astype(np.float64, copy=False)
    weighted_mean64 = float(
        np.sum(top.astype(np.float64) * area64, dtype=np.float64)
        / np.sum(area64, dtype=np.float64)
    )
    scalar = np.float32(weighted_mean64)
    observed = (float(np.min(top)), float(np.median(top)), float(np.max(top)))
    proof.EXPECTED_ARWEN_P_TOP_PA_F32 = scalar
    proof.EXPECTED_TOP_PRESSURE_RANGE_PA = observed
    return {
        "area_weighted_mean_f32_pa": float(scalar),
        "per_column_minimum_pa": observed[0],
        "per_column_median_pa": observed[1],
        "per_column_maximum_pa": observed[2],
        "seeding_policy": (
            "re-derived here, independently recomputed and asserted by the "
            "proof reduction"
        ),
    }


# --------------------------------------------------------------------------
# F000: the surface diagnostics the first physics call has not made yet
# --------------------------------------------------------------------------
#: The start-frame fields this driver completes after the proof's own overlay.
#:
#: THE BREAKAGE THIS PREVENTS: every hex forecast's start frame published
#: ``q2`` as the seam's pre-first-call +0 and ``psfc`` as its pre-first-call
#: 100000 Pa, so the hour-0 2 m dewpoint map was drawn from zero humidity (a
#: limited-area forecast's f000 dewpoint map was drawn from Q2 = 0 in every
#: cell) while every later hour was right.  The regional
#: engine's own hour 0 carries its start state's T2, Q2, PSFC, U10 and V10;
#: the proof overlay already gives this frame ``t2``/``u10``/``v10`` from the
#: start file, and these two complete it.
F000_COMPLETED_SURFACE_DIAGNOSTICS = ("q2", "psfc")


def load_f000_start_humidity(init_path: Path) -> dict[str, Any]:
    """The start file's own 2 m humidity, when it carries one.

    The init writer diagnoses ``q2`` from the source's 2 m relative humidity
    (``rh2``).  A source that delivers specific humidity instead -- the
    ``--use-spechumd yes`` road, which is how ``woof hex intermediate`` hands
    HRRR over -- reaches the writer with no ``rh2``, so ``q2`` is written as
    zero in every cell.  An all-zero ``q2`` is therefore "not carried", not a
    measurement of bone-dry air, and the start frame falls back to the model's
    lowest level (:func:`complete_f000_surface_diagnostics`).
    """

    from netCDF4 import Dataset

    with Dataset(init_path, "r") as dataset:
        if "q2" not in dataset.variables:
            return {
                "carried": False,
                "q2": None,
                "reason": "the start file has no q2 variable",
            }
        value = proof._read_exact_variable(
            dataset, "q2", dtype=np.float32, dimensions=("Time", "nCells")
        )
    field = np.ascontiguousarray(value[0], dtype=np.float32)
    carried = bool(np.any(field != np.float32(0.0)))
    return {
        "carried": carried,
        "q2": field if carried else None,
        "sha256": proof.array_sha256(field),
        "reason": (
            "the start file carries q2"
            if carried
            else (
                "the start file's q2 is zero in every cell: the init writer "
                "diagnoses q2 from the source's 2 m relative humidity, and "
                "this source delivered none"
            )
        ),
    }


def saturation_mixing_ratio(temperature_k: Any, pressure_pa: Any) -> np.ndarray:
    """Saturation water-vapour mixing ratio over liquid water, kg kg-1.

    The Tetens form (6.112 hPa, 17.27, 35.86 K) the init writer's own ``q2``
    diagnosis uses, so a ``q2`` bounded here and a ``q2`` the start file
    carried sit on the same saturation curve.  The vapour pressure is held
    below 99 % of the air pressure, as the init writer's surface pass holds
    it, because the denominator crosses zero above that.
    """

    temperature = np.asarray(temperature_k, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    vapour = 611.2 * np.exp(17.27 * (temperature - 273.16) / (temperature - 35.86))
    vapour = np.minimum(vapour, 0.99 * pressure)
    return (0.622 * vapour / (pressure - vapour)).astype(np.float32)


def _snapshot_array_record(value: Any) -> dict[str, Any]:
    """One ``receipt['arrays']`` row, computed as ``proof.capture_snapshot`` does."""

    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": proof.array_sha256(array),
        "minimum": float(array.min()) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
        "nonzero": int(np.count_nonzero(array)),
        "negative": int(np.count_nonzero(array < 0)) if array.dtype.kind == "f" else 0,
    }


def complete_f000_surface_diagnostics(
    snapshot: dict[str, Any], start_humidity: Mapping[str, Any]
) -> dict[str, Any]:
    """Give the start frame a real ``q2`` and ``psfc``; return what was done.

    ``q2``: the start file's own when it carries one (the analysis 2 m
    humidity, as the regional engine publishes at its hour 0).  When it does
    not, the lowest model level's water-vapour mixing ratio, bounded to
    [0, saturation at the frame's 2 m temperature and surface pressure]: the
    2 m value a surface layer with no flux yet would give, and the bound keeps
    the 2 m dewpoint at or below the 2 m temperature.

    ``psfc``: the model's own surface pressure at this frame, the quantity the
    first physics call receives as ``psfc``.

    Legal only on the step-0 frame and only before any physics call: after
    one, both fields are the seam's measurements and must not be replaced.
    The snapshot's ``receipt['arrays']`` rows for both fields are recomputed,
    so the published digests describe the published bytes.
    """

    receipt = snapshot["receipt"]
    if int(receipt.get("step", -1)) != 0:
        raise ValueError(
            "the F000 completion applies to the step-0 frame only; on a later "
            "frame q2 and psfc are the surface layer's own values"
        )
    execution = receipt.get("arwen_v2_surface_execution")
    if not isinstance(execution, Mapping) or execution.get("last_noahmp_census") is not None:
        raise ValueError(
            "the F000 completion found a frame the physics has already run on; "
            "replacing its q2 and psfc would overwrite the seam's measurements"
        )
    arrays = snapshot["arrays"]
    t2 = np.asarray(arrays["t2"])
    surface_pressure = np.asarray(arrays["surface_pressure"])
    qv = np.asarray(arrays["qv"])
    shape = t2.shape
    if surface_pressure.shape != shape or qv.ndim != 2 or qv.shape[1:] != shape:
        raise ValueError(
            f"F000 completion shapes disagree: t2 {t2.shape}, surface_pressure "
            f"{surface_pressure.shape}, qv {qv.shape}"
        )
    # A row whose seam exports neither field publishes it at no frame, so the
    # start frame gains nothing the later frames lack.
    completed = tuple(
        name for name in F000_COMPLETED_SURFACE_DIAGNOSTICS if name in arrays
    )
    result: dict[str, Any] = {
        "scope": "F000 history frame only; the model state is unchanged",
        "absent_from_this_row": [
            name for name in F000_COMPLETED_SURFACE_DIAGNOSTICS if name not in completed
        ],
    }
    # q2 first: its placeholder check is the one that can refuse, and a
    # refusal must leave the frame exactly as the capture produced it.
    if "q2" in completed:
        result["q2"] = _complete_f000_q2(arrays, start_humidity, t2, surface_pressure, qv)
    if "psfc" in completed:
        psfc_before = np.asarray(arrays["psfc"])
        psfc = np.array(surface_pressure, dtype=np.float32, copy=True, order="C")
        arrays["psfc"] = psfc
        result["psfc"] = {
            "source": (
                "surface_pressure (the model's own, as the first physics call "
                "receives it)"
            ),
            "placeholder_minimum": float(psfc_before.min()),
            "placeholder_maximum": float(psfc_before.max()),
            "minimum": float(psfc.min()),
            "maximum": float(psfc.max()),
            "sha256": proof.array_sha256(psfc),
        }
    rows = receipt.get("arrays")
    if isinstance(rows, dict):
        for name in completed:
            rows[name] = _snapshot_array_record(arrays[name])
    return result


def _complete_f000_q2(
    arrays: dict[str, Any],
    start_humidity: Mapping[str, Any],
    t2: np.ndarray,
    surface_pressure: np.ndarray,
    qv: np.ndarray,
) -> dict[str, Any]:
    shape = t2.shape
    placeholder = np.ascontiguousarray(np.asarray(arrays["q2"]))
    if (
        placeholder.dtype != np.dtype(np.float32)
        or placeholder.shape != shape
        or np.any(placeholder.view(np.uint32) != np.uint32(0))
    ):
        raise ValueError(
            "the start frame's q2 is not the seam's exact +0 placeholder, so "
            "something computed it and the completion would overwrite it"
        )
    if start_humidity.get("carried"):
        q2 = np.array(start_humidity["q2"], dtype=np.float32, copy=True, order="C")
        if q2.shape != shape:
            raise ValueError(
                f"the start file's q2 has shape {q2.shape} and this frame "
                f"{shape}; it was written for a different mesh"
            )
        q2_record = {
            "source": "start file q2",
            "start_file_q2_sha256": start_humidity.get("sha256"),
        }
    else:
        lowest = np.asarray(qv[0], dtype=np.float32)
        ceiling = saturation_mixing_ratio(t2, surface_pressure)
        q2 = np.ascontiguousarray(
            np.minimum(np.maximum(lowest, np.float32(0.0)), ceiling), dtype=np.float32
        )
        q2_record = {
            "source": (
                "lowest model level qv, bounded to [0, saturation at t2 and "
                "surface_pressure]"
            ),
            "why": str(start_humidity.get("reason", "")),
            "cells_bounded_at_saturation": int(np.count_nonzero(lowest > ceiling)),
            "cells_floored_at_zero": int(np.count_nonzero(lowest < 0.0)),
        }
    if not np.all(np.isfinite(q2)):
        raise FloatingPointError("the completed F000 q2 is non-finite")
    arrays["q2"] = q2
    return {
        **q2_record,
        "minimum": float(q2.min()),
        "mean": float(q2.mean(dtype=np.float64)),
        "maximum": float(q2.max()),
        "sha256": proof.array_sha256(q2),
    }


def complete_f000_from_first_call(
    snapshot: dict[str, Any], first_call: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Complete the start frame's surface-layer diagnostics from the first physics call.

    woof.hex.f000_first_call says why.  Runs after
    :func:`complete_f000_surface_diagnostics`; a first-call 2 m humidity replaces
    the lowest-level fallback, never a q2 the start file carried.  The
    ``receipt['arrays']`` rows of the replaced fields are recomputed.
    """

    receipt = snapshot["receipt"]
    if int(receipt.get("step", -1)) != 0:
        raise ValueError("the first-call completion applies to the step-0 frame only")
    arrays = snapshot["arrays"]
    record = f000_first_call.complete_arwen_start_frame(
        arrays, first_call, receipt.get("f000_completed_surface_diagnostics"))
    rows = receipt.get("arrays")
    if isinstance(rows, dict):
        for name in record.get("fields", {}):
            rows[name] = _snapshot_array_record(arrays[name])
    return record


def install_capture_labels(labels: Mapping[int, str]) -> None:
    """Rebind the proof's fixed F000/F030/F001 label table to this schedule."""

    proof.SNAPSHOT_LABELS = dict(labels)
    proof.SNAPSHOT_STEPS = tuple(sorted(labels))


# --------------------------------------------------------------------------
# constructor values (transcribed from proof.build_arwen_constructor_values,
# case literals replaced by measurement)
# --------------------------------------------------------------------------
def step_admission_receipt(config: Any) -> dict[str, Any]:
    """Record the evidence for the selected schedule.  Refuses nothing.

    RETIRED 2026-10-10 as a gate (owner ruling).  This used to raise unless a timestep anchor or a dycore step
    certificate held the configuration; it now writes, into the run's
    receipt, whether one does, and whether the run is the native x4
    reference's own schedule.  The cadences the run cannot schedule are
    refused by ``config.validate()`` (woof.hex.dt_admission.require_cadences).
    """

    from woof.hex import convection_admission
    from woof.hex.step_anchors import step_anchor_record

    cumulus_scheme = convection_admission.constructor_scheme(
        config.config_convection_scheme
    )
    record = {
        "refuses": False,
        "anchor": dt_admission.anchor_record(
            config.config_dt,
            radiation_seconds=config.config_radt_seconds,
            surface_pbl_seconds=config.config_bldt_seconds,
            cumulus_seconds=config.config_cudt_seconds,
            cumulus_scheme=cumulus_scheme,
        ),
        "dycore_anchor": step_anchor_record(config),
    }
    lane = getattr(config, "recorded_lane_match", None)
    if lane is not None:
        record["x4_recorded_schedule"] = lane()
    return record


def parse_dynamics_overrides(items: Sequence[str] | None) -> dict[str, Any]:
    """``--dynamics KEY=VALUE`` (repeatable) as configuration attributes.

    KEY is the configuration attribute, which is the MPAS namelist option's
    own name (``config_apvm_upwinding``, ``config_smdiv``,
    ``config_coef_3rd_order``, ``config_xnutr``, ``config_epssm``,
    ``config_positive_definite``, ``config_smagorinsky_coef`` ...).  VALUE is
    JSON (``0.25``, ``true``, ``"2d_smagorinsky"``), with Fortran's
    ``.true.``/``.false.`` and a bare word accepted as written.
    """

    overrides: dict[str, Any] = {}
    for item in items or ():
        key, separator, raw = str(item).partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(
                f"--dynamics {item!r} is not KEY=VALUE, so there is no "
                "option to set"
            )
        text = raw.strip()
        folded = text.casefold().strip(".")
        if folded in ("true", "false", "t", "f") and not text.startswith('"'):
            value: Any = folded in ("true", "t")
        else:
            try:
                value = json.loads(text)
            except ValueError:
                value = text
        if key in overrides and overrides[key] != value:
            raise ValueError(
                f"--dynamics sets {key} twice, to {overrides[key]!r} and "
                f"{value!r}; the run can hold one"
            )
        overrides[key] = value
    return overrides


def build_forecast_config(
    *, dynamics_overrides: Mapping[str, Any] | None = None, **arguments: Any
) -> Any:
    """The run's configuration: the flags' knobs, then ``dynamics_overrides``.

    ``arguments`` are :func:`_build_forecast_config`'s, which documents them.

    ``dynamics_overrides`` maps configuration attributes to values for the
    dynamics options that have no flag of their own.  THE BREAKAGE THIS
    CLOSES (2026-10-10 audit, measured): the certificate gates over
    ``config_apvm_upwinding``, ``config_smdiv``, ``config_coef_3rd_order``,
    ``config_xnutr``, ``config_epssm*``, ``config_positive_definite``,
    ``config_scalar_advection`` and the Smagorinsky coefficients were
    retired from the configuration type, and no run could set any of them:
    this function took none, so a ``--physics-namelist`` carrying
    ``config_apvm_upwinding = 0.25`` was refused at the door with "the run
    has no flag for it".  An attribute the run's configuration type does
    not carry is refused here by name, because setting it would change
    nothing the run executes.
    """

    config = _build_forecast_config(**arguments)
    if not dynamics_overrides:
        return config
    from dataclasses import fields, replace

    carried = {item.name for item in fields(config)}
    unknown = sorted(set(dynamics_overrides) - carried)
    if unknown:
        raise ValueError(
            f"{unknown} cannot be set on this run: its configuration "
            f"({type(config).__name__}) carries no such option, so the value "
            "would be recorded and not run"
        )
    welded = sorted(set(dynamics_overrides) & _FLAG_CARRIED_DYNAMICS)
    if welded:
        raise ValueError(
            f"{welded} each have a flag of their own "
            f"({[_FLAG_CARRIED_DYNAMICS_FLAGS[name] for name in welded]}); "
            "set them there, so one option cannot be given two values"
        )
    return replace(config, **dict(dynamics_overrides))


#: Options another argument of this driver already decides.  ``--dynamics``
#: refuses them so one option cannot arrive with two values.
_FLAG_CARRIED_DYNAMICS_FLAGS = {
    "config_dt": "--dt",
    "config_time_integration_order": "--time-integration-order",
    "config_number_of_sub_steps": "--acoustic-substeps",
    "config_monotonic": "--monotonic",
    "config_horiz_mixing": "--horiz-mixing",
    "config_mpas_cam_coef": "--cam-coefficient",
    "config_number_cam_damping_levels": "--cam-damping-levels",
    "config_rayleigh_damp_u": "--rayleigh-u",
    "config_zd": "--damping-start-m",
    "config_apply_lbcs": "--lbc-dir",
    "config_radt_seconds": "--radiation-seconds",
    "config_bldt_seconds": "--pbl-cadence",
    "config_cudt_seconds": "--convection",
    "config_convection_scheme": "--convection",
    "config_local_timestep": "--local-timestep",
    "config_local_timestep_rates": "--local-timestep-rates",
    "config_local_timestep_buffer_rings": "--local-timestep-buffer-rings",
}
_FLAG_CARRIED_DYNAMICS = frozenset(_FLAG_CARRIED_DYNAMICS_FLAGS)


def _build_forecast_config(
    *,
    dt_seconds: float,
    convection_scheme: str = "cu_grell_freitas",
    surface_pbl_seconds: float | None = None,
    horiz_mixing: str = "2d_smagorinsky",
    local_timestep: bool = False,
    local_timestep_declared_off: bool = False,
    local_timestep_rates: tuple[int, ...] = (1, 3),
    local_timestep_buffer_rings: int = 1,
    apply_lbcs: bool = False,
    time_integration_order: int = 2,
    acoustic_substeps: int = 4,
    radiation_seconds: float = 1800.0,
    monotonic: bool = True,
    cam_coefficient: float = 0.0,
    cam_damping_levels: int = 4,
    rayleigh_u: bool = False,
    damping_start_m: float = 22000.0,
) -> Any:
    """Build the run's configuration AT THE BOUND MESH'S TIMESTEP.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, the proving RTX 5090):
    this function did not exist and the configuration was constructed from
    its dataclass defaults, so ``config_dt`` was 120.0 no matter what mesh
    was bound.  ``bind_mesh`` rebinds ``DT_SECONDS`` in this module and in
    the proof module, and the sealed Arwen constructor read that rebound
    value -- but the DYCORE takes its outer step from ``config.config_dt``,
    which nothing rebound.  A mesh row declaring 100 s therefore bound
    clean, allocated 18,820 MiB, spent 285 s and died inside composite
    step 0 with ``post-RK candidate time must equal the exact step
    endpoint: 120.0 != 100.0``.

    There is now ONE timestep in this path: the bound mesh's.  It reaches
    the config here, and ``build_forecast_constructor_values`` derives the
    seam's clocks from the config rather than from the module constant, so
    the two cannot disagree by construction.  A cadence the timestep cannot
    schedule is refused by ``config.validate()`` below, on the host, before
    a mesh file is opened -- which is where a 285-second refusal belongs.

    The two cadences welded to the timestep travel with it.  ``cudt`` has
    no choice: WRF pins ``cudt = 0`` for Grell-Freitas, so the sealed
    constructor requires ``cumulus_seconds == dt``.  ``bldt`` follows dt by
    DEFAULT because that is the proven configuration's own semantics -- the
    native x4 reference ran ``bldt = dt``, i.e. surface/PBL every step --
    and ``surface_pbl_seconds`` is how an A/B arm holds it there while dt
    moves.  ``None`` is the weld and changes no run; an explicit value is an
    instrument and records itself as one.  See :mod:`woof.hex.pbl_cadence`
    for the measurement that needed it and the breakage its registry key
    prevents.
    """

    from woof.hex.config_v841 import (
        V841_GWDO_KERNEL_SHA256,
        V841MpasColumnPhysicsGwdoConfig,
        V841MpasColumnPhysicsSmagorinskyGwdoConfig,
    )
    from woof.hex.config_lts import (
        V841LocalTimestepGwdoConfig,
        V841LocalTimestepSmagorinskyGwdoConfig,
    )

    from woof.hex import convection_admission, pbl_cadence

    dt = float(dt_seconds)
    scheme = str(convection_scheme)
    if scheme not in convection_admission.ADMITTED_CONVECTION_SCHEMES:
        raise ValueError(
            f"convection_scheme must be one of "
            f"{list(convection_admission.ADMITTED_CONVECTION_SCHEMES)}, "
            f"got {convection_scheme!r}"
        )
    bldt = (
        dt
        if surface_pbl_seconds is None
        else pbl_cadence.resolve_seconds(
            dt_seconds=dt, requested=surface_pbl_seconds
        )
    )
    clocks: dict[str, Any] = {
        # Imported mesh binding specializes GWD column storage and records its
        # effective shader digest. Dataclass defaults retain the native digest.
        "gwdo_kernel_sha256": V841_GWDO_KERNEL_SHA256,
        "config_dt": dt,
        "config_bldt_seconds": bldt,
        # WRF pins cudt=0 for Grell-Freitas, so with a scheme selected the
        # cumulus cadence IS dt.  With no scheme selected there is no cadence
        # at all -- see woof.hex.convection_admission.
        "config_cudt_seconds": (
            None if scheme == convection_admission.SCHEME_OFF else dt
        ),
        "config_convection_scheme": scheme,
        # The lateral-boundary switch is NOT a user knob here: it is the
        # grid's own property, read off the bdyMask triple by the caller.
        # mpas_atm_bdy_checks refuses either mismatch by name -- boundary
        # cells with the switch off, or the switch on with none.
        "config_apply_lbcs": bool(apply_lbcs),
        "config_time_integration_order": time_integration_order,
        "config_number_of_sub_steps": acoustic_substeps,
        "config_radt_seconds": radiation_seconds,
        "config_monotonic": monotonic,
        "config_mpas_cam_coef": cam_coefficient,
        "config_rayleigh_damp_u": rayleigh_u,
        "config_zd": damping_start_m,
    }
    lts_block: dict[str, Any] = {
        "config_local_timestep": bool(local_timestep),
        "config_local_timestep_rates": tuple(local_timestep_rates),
        "config_local_timestep_buffer_rings": int(local_timestep_buffer_rings),
    }

    if horiz_mixing == "2d_smagorinsky":
        if cam_coefficient != 0 or rayleigh_u or damping_start_m != 22000 or cam_damping_levels != 4:
            if local_timestep or local_timestep_declared_off:
                raise ValueError(
                    "no configuration type carries both the upper-damping block "
                    "and the local-timestep block (V841UpperDamping*Config and "
                    "V841LocalTimestep*Config are separate subclasses), so one "
                    "of the two would be dropped without a word; run one or the other")
            from woof.hex.config_v841 import V841UpperDampingSmagorinskyConfig
            return V841UpperDampingSmagorinskyConfig(**clocks,config_number_cam_damping_levels=cam_damping_levels)
        # Default: native's Registry configuration (deformation-based 2-D
        # Smagorinsky + del4), the regime the natA/natB 24-h references
        # integrate.  Numbers under this configuration are a NEW SUB-SERIES: they are
        # not bit-comparable to any mixing-off arm.
        if local_timestep or local_timestep_declared_off:
            # Same released lane, plus the opt-in local-timestep block.  With
            # the switch off this subtype validates and executes identically
            # to its parent, which is what the default-off gate measures.
            return V841LocalTimestepSmagorinskyGwdoConfig(**clocks, **lts_block)
        return V841MpasColumnPhysicsSmagorinskyGwdoConfig(**clocks)
    if horiz_mixing == "off":
        # Explicit control arm: the pre-mixing configuration (the regime
        # native itself dies in on convective cases -- the 2026-08-17
        # reference-node control died at step 466 on case B).
        if local_timestep or local_timestep_declared_off:
            return V841LocalTimestepGwdoConfig(**clocks, **lts_block)
        return V841MpasColumnPhysicsGwdoConfig(**clocks)
    raise ValueError(
        f"horiz_mixing must be '2d_smagorinsky' or 'off', got {horiz_mixing!r}"
    )


def select_initial_scalar_names(
    variables: Mapping[str, Any], model_names: Sequence[str],
) -> tuple[str, ...]:
    """Preserve a complete carried species row or use the existing cold start."""

    names = tuple(model_names)
    if names[:len(SOURCE_SCALAR_NAMES)] != tuple(SOURCE_SCALAR_NAMES):
        raise ValueError("the model species row must begin with qv, qc and qr")
    missing_source = [name for name in SOURCE_SCALAR_NAMES if name not in variables]
    if missing_source:
        raise ValueError(f"init is missing required source species {missing_source}")
    from woof.hex.species_row import AEROSOL_MASS, species_row_for_names
    row = species_row_for_names(names)
    aerosol_names = tuple(d.name for d in row.species if d.species_class == AEROSOL_MASS)
    if aerosol_names and not any(name in variables for name in aerosol_names):
        atmospheric = tuple(name for name in names if name not in aerosol_names)
        return select_initial_scalar_names(variables, atmospheric)
    carried = [name for name in names[len(SOURCE_SCALAR_NAMES):] if name in variables]
    if not carried:
        return tuple(SOURCE_SCALAR_NAMES)
    missing = [name for name in names if name not in variables]
    if missing:
        raise ValueError(
            f"init carries a partial model species row: present {carried}, "
            f"missing {missing}; a cold start would discard carried condensate"
        )
    return names


def prepare_initial_scalar_block(
    state: Any, *, source_names: Sequence[str], model_names: Sequence[str],
    negative_qv: Mapping[str, Any],
) -> dict[str, Any]:
    """Record complete source planes without replacing them with cold zeros."""

    names = tuple(model_names)
    source = tuple(source_names)
    if source == names:
        return {
            "source_present": list(source),
            "source_absent": [],
            "source_hashes": {
                name: proof.array_sha256(state.scalars[index])
                for index, name in enumerate(source)
            },
            "scalar_order": list(names),
            "native_species_preserved": True,
            "cold_zero_planes_added": 0,
            "negative_qv": dict(negative_qv),
            "negative_qv_preserved_until_candidate_pre_wsm6_clamp": True,
        }
    if names == tuple(proof.SCALAR_NAMES):
        return proof.augment_exact_wsm6_scalars(state)
    return proof.augment_cold_scalars(state, names, source_names=source)


def supplied_undriven_scalars(
    first_lbc_path: str | Path, scalar_names: Sequence[str], driven: Sequence[str],
) -> list[str]:
    """The model species a boundary stream carries but the row does not drive.

    Metadata only, through the engine's Rust NetCDF reader (rw_netcdf).
    ``driven`` holds the stream's own ``lbc_``-prefixed names, as
    :func:`select_lbc_scalar_names` returns them, so the comparison is made on
    those names.  Comparing the bare species name against it listed every
    driven mass as undriven in the receipt (rrfs-h-h4a skeptic finding,
    2026-10-05).
    """

    from woof.netcdf_bridge import open_dataset

    dataset = open_dataset(Path(first_lbc_path))
    driven_names = set(driven)
    return sorted(
        name for name in dataset.variables
        if name.startswith("lbc_")
        and name[4:] in scalar_names
        and name not in driven_names
    )


def select_lbc_scalar_names(
    paths: Sequence[str], model_names: Sequence[str],
) -> tuple[str, ...]:
    """Select the contiguous model species every boundary endpoint carries."""

    from netCDF4 import Dataset
    from woof.hex.lbc import LBC_REQUIRED_VARIABLES, LBC_OPTIONAL_SCALAR_VARIABLES

    names = tuple(f"lbc_{name}" for name in model_names)
    supported = {**LBC_REQUIRED_VARIABLES, **LBC_OPTIONAL_SCALAR_VARIABLES}
    selected: tuple[str, ...] | None = None
    for path in paths:
        with Dataset(path, "r") as dataset:
            present = tuple(name for name in names if name in dataset.variables)
            if not present or present != names[:len(present)]:
                raise ValueError(
                    f"boundary file {path} carries noncontiguous model species "
                    f"{present}; driven arrays must follow the model's leading "
                    "species or a supplied field would drive the wrong tracer"
                )
            for name in present:
                if name not in supported:
                    raise ValueError(f"boundary species {name} has no supported reader schema")
                variable = dataset.variables[name]
                if tuple(variable.dimensions) != supported[name] or variable.dtype != np.dtype("float32"):
                    raise ValueError(
                        f"boundary species {name} in {path} must be float32 on "
                        f"{supported[name]}; a shape or precision change would "
                        "change the driving field's meaning"
                    )
        if selected is not None and present != selected:
            raise ValueError(
                f"boundary species presence changes at {path}: {selected} "
                f"versus {present}; both hourly endpoints must carry each "
                "driven tracer"
            )
        selected = present
    if selected is None:
        raise ValueError("no boundary files were supplied for tracer selection")
    return selected


def prepare_inactive_deformation(mesh: Any) -> dict[str, Any]:
    """Retain supplied legacy carriers that v8.4.1 mixing does not consume."""

    arrays = getattr(mesh, "arrays", {})
    present = [name for name in ("defc_a", "defc_b") if name in arrays]
    if not present:
        return proof.attach_inactive_zero_deformation(mesh)
    if len(present) != 2:
        raise ValueError(
            f"inactive deformation carrier pair is incomplete: present {present}; "
            "the device mesh requires both defc_a and defc_b arrays"
        )
    return {
        "shape": list(arrays["defc_a"].shape),
        "dtype": str(arrays["defc_a"].dtype),
        "active_mixing_claim": False,
        "source": "preserved supplied legacy carriers; v8.4.1 uses deformation weights",
        "defc_a_sha256": proof.array_sha256(arrays["defc_a"]),
        "defc_b_sha256": proof.array_sha256(arrays["defc_b"]),
    }


def refusal_exception_details(error: BaseException) -> dict[str, Any]:
    """Keep the numerical refusal behind a staged transaction's wrapper."""

    chain: list[dict[str, str]] = []
    current: BaseException | None = error
    relation = "raised"
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({
            "exception": type(current).__name__,
            "message": str(current),
            "relation": relation,
        })
        if current.__cause__ is not None:
            current, relation = current.__cause__, "cause"
        elif not current.__suppress_context__:
            current, relation = current.__context__, "context"
        else:
            current = None
    return {
        "exception": type(error).__name__,
        "message": str(error),
        # The refusal that started the chain.  A transaction wrapper, and a
        # rollback that itself fails, each wrap the step's own words; the
        # end of the chain is what the step refused, so a reader of the
        # receipt sees it first rather than the last wrapper's text.
        "original_refusal": {
            "exception": chain[-1]["exception"],
            "message": chain[-1]["message"],
        },
        "exception_chain": chain,
        "traceback": "".join(traceback.format_exception(
            type(error), error, error.__traceback__, chain=True,
        )),
    }


def build_forecast_constructor_values(
    *,
    init_path: Path,
    mesh: Any,
    vertical: Any,
    reference: Any,
    saved_diagnostics: Any,
    start_time_text: str | None,
    config: Any,
    n_soil_levels: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from netCDF4 import Dataset

    # The soil column is the selected land surface's: the frozen row's
    # Noah-MP integrates four layers, a suite row's RUC nine levels.  The
    # init is read at exactly that count and refused at any other, so a
    # land surface is never handed a column interpolated for another one.
    soil_levels = int(N_SOIL_LEVELS if n_soil_levels is None else n_soil_levels)

    read = proof._read_exact_variable
    with Dataset(init_path, "r") as dataset:
        select_initial_scalar_names(
            dataset.variables, tuple(_row_for_scheme(config.config_microp_scheme).names())
        )
        landmask_i = read(dataset, "landmask", dtype=np.int32, dimensions=("nCells",))
        ivgtyp = read(dataset, "ivgtyp", dtype=np.int32, dimensions=("nCells",))
        isltyp = read(dataset, "isltyp", dtype=np.int32, dimensions=("nCells",))
        xland_source = read(
            dataset, "xland", dtype=np.float32, dimensions=("Time", "nCells")
        )
        if xland_source.shape != (1, N_CELLS):
            raise ValueError(f"xland shape changed: {xland_source.shape}")
        xland = np.ascontiguousarray(xland_source[0], dtype=np.float32)
        zgrid = read(
            dataset,
            "zgrid",
            dtype=np.float32,
            dimensions=("nCells", "nVertLevelsP1"),
        )
        surface: dict[str, np.ndarray] = {}
        for source, target in (
            ("vegfra", "vegfra"),
            ("skintemp", "tsk"),
            ("tmn", "tmn"),
            ("xice", "xice"),
            ("snow", "snow"),
            ("snowh", "snow_depth"),
        ):
            value = read(
                dataset, source, dtype=np.float32, dimensions=("Time", "nCells")
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            surface[target] = np.ascontiguousarray(value[0], dtype=np.float32)
        # Noah-MP's vegetation fraction (FVEG = SHDMAX / 100 at dveg = 4).
        # The frozen engine constructor takes no shdmax, so before this read
        # every forecast ran its land surface at the engine's 90 % fill
        # (woof.hex.cuda_arwen_physics_v841, _NOAHMP_STATIC_FIELDS).  Native
        # MPAS reads the same static field.  Percent, time-invariant.
        surface["shdmax"] = np.ascontiguousarray(
            read(dataset, "shdmax", dtype=np.float32, dimensions=("nCells",)),
            dtype=np.float32,
        )
        soil: dict[str, np.ndarray] = {}
        # A selection with no land surface (sf_surface_physics=0) integrates
        # no soil column, so no scheme reads the init's tslb/smois and there
        # is no other scheme's interpolation to refuse: the init's own soil
        # count is not held against zero.  Before this, every land-surface-off
        # selection stopped at host preparation on "tslb shape (1, N, 4) is
        # not (1, N, 0)" (global 30 km, 2026-10-11).
        for source, target in (
            () if soil_levels == 0 else (
                ("tslb", "soil_temperature"),
                ("smois", "soil_moisture"),
            )
        ):
            value = read(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells", "nSoilLevels"),
            )
            if value.shape != (1, N_CELLS, soil_levels):
                raise ValueError(
                    f"{source} shape {value.shape} is not (1, {N_CELLS}, "
                    f"{soil_levels}): the selected land surface integrates "
                    f"{soil_levels} soil levels and this init carries another "
                    "count"
                )
            soil[target] = np.ascontiguousarray(value[0].T, dtype=np.float32)
        nominal_min_dc = read(dataset, "nominalMinDc", dtype=np.float32, dimensions=())
        start_text = str(getattr(dataset, "config_start_time", ""))

    if not start_text:
        raise RuntimeError("init carries no config_start_time")
    try:
        start_datetime = datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S")
    except ValueError as error:
        raise RuntimeError(f"init config_start_time is unparseable: {start_text!r}") from error
    if start_time_text is not None and start_time_text != start_text:
        raise RuntimeError(
            f"--start-time {start_time_text!r} disagrees with the init's "
            f"config_start_time {start_text!r}"
        )

    # Lake columns: the frozen physics stack carries NO lake model, and the
    # arwen vegetation parameter tables end at category 20, so a lake column
    # (MODIS category 21 -- present in with-lakes land use and in generated-mesh
    # statics, absent from the x4 proof's own landuse, whose maximum is 19)
    # routed to the land path indexes off the SLA table inside the Noah-MP cold
    # start.  WRF applies its rule at this same boundary when sf_lake_physics
    # is off: a lake column IS open water.  Fold before classification so every
    # downstream consumer -- the partition, the census, the seam -- sees one
    # consistent water column, and put the count in the receipt.  On the native
    # x4 case the mask is empty and every array passes through untouched.
    lake_mask = ivgtyp == np.int32(21)
    lake_fold = {
        "lake_category": 21,
        "lake_columns": int(np.count_nonzero(lake_mask)),
        "rule": (
            "no lake model in the frozen v8.4.1 physics: lake columns become "
            "open water before classification (ivgtyp 21->17, isltyp ->14, "
            "landmask ->0, xland ->2.0), the same conversion WRF applies when "
            "sf_lake_physics is off"
        ),
        "pre_fold_landmask_sha256": proof.array_sha256(landmask_i),
        "pre_fold_ivgtyp_sha256": proof.array_sha256(ivgtyp),
    }
    if lake_fold["lake_columns"]:
        ivgtyp = np.ascontiguousarray(np.where(lake_mask, np.int32(17), ivgtyp))
        isltyp = np.ascontiguousarray(np.where(lake_mask, np.int32(14), isltyp))
        landmask_i = np.ascontiguousarray(
            np.where(lake_mask, np.int32(0), landmask_i)
        )
        xland = np.ascontiguousarray(
            np.where(lake_mask, np.float32(2.0), xland)
        )

    source_xland_sha256 = proof.array_sha256(xland_source)
    flat_xland_sha256 = proof.array_sha256(xland)
    xland_unique, xland_counts = np.unique(xland, return_counts=True)
    # MEASURED, not pinned: the proof asserted (1.0, 2.0) with the case counts.
    # The value SET is still a hard requirement -- MPAS xland is 1 (land/ice)
    # or 2 (water) and nothing else -- but requiring BOTH values requires the
    # domain to contain a coastline, and a limited-area domain need not.
    #
    # THE BREAKAGE THIS PREVENTS: xland selects which surface scheme a column
    # runs.  A third value would send columns to neither the land-surface
    # model nor the open-water branch and the surface fluxes would be whatever
    # the uninitialised path left behind.
    #
    # THE BREAKAGE IT MUST NOT INVENT, MEASURED (2026-08-26, r4.75.11020): a
    # placed swath over the Southern Ocean is all water, so its xland set is
    # (2.0,).  Demanding a land column refuses an all-ocean domain for having
    # no coastline in it, which is a property of the case, not a defect.
    if not set(float(value) for value in xland_unique) <= {1.0, 2.0}:
        raise RuntimeError(
            f"init xland carries values outside (1.0, 2.0): {xland_unique}; "
            "MPAS xland is 1 for land or ice and 2 for water, and a column "
            "carrying anything else reaches neither surface branch"
        )
    if xland_unique.size == 0:
        raise RuntimeError("init xland is empty")
    xice = surface["xice"]
    if float(np.min(xice)) < 0.0 or float(np.max(xice)) > 1.0:
        raise RuntimeError("init xice lies outside [0,1]")
    # The partition the engine runs (xland, xice, ivgtyp), admitted as native
    # made it.  The former "category signature" also demanded landmask 0
    # under sea ice, landmask 1 under a glacier and xice 0 under a glacier;
    # native init promises none of those (its skin-temperature branch leaves
    # landmask 0 and xice untouched on the cold water columns it converts),
    # and nothing downstream routes on landmask.  That demand refused RRFS
    # v2's conus3km init over two frozen mountain-lake cells
    # (woof.hex.surface_signature has the measurement and native's rule).
    # Landmask is measured into the receipt below; what still refuses names
    # the surface branch it keeps off a sea-ice column.
    from woof.hex.surface_signature import classify_native_surface

    surface_counts, surface_masks = classify_native_surface(
        xland=xland,
        xice=xice,
        ivgtyp=ivgtyp,
        landmask=landmask_i,
        xice_threshold=float(ARWEN_XICE_THRESHOLD),
    )
    surface_classification = dict(surface_counts)
    sea_ice_mask = surface_masks["sea_ice"]
    glacier_mask = surface_masks["glacier"]
    if (
        surface_classification["sflx_land_columns"]
        + surface_classification["open_water_columns"]
        + surface_classification["sea_ice_columns"]
        + surface_classification["glacier_columns"]
        != N_CELLS
    ):
        raise RuntimeError("surface classification does not partition the mesh")
    glacier_indices = np.ascontiguousarray(np.flatnonzero(glacier_mask))
    sea_ice_indices = np.ascontiguousarray(np.flatnonzero(sea_ice_mask))
    threshold_delta_indices = np.ascontiguousarray(
        np.flatnonzero((xice >= ARWEN_XICE_THRESHOLD) & (xice < np.float32(0.5)))
    )
    surface_classification_receipt = {
        **surface_classification,
        "lake_fold": lake_fold,
        "source_field": "init xland[0,:]",
        "source_shape": list(xland_source.shape),
        "source_array_sha256": source_xland_sha256,
        "constructor_shape": list(xland.shape),
        "constructor_array_sha256": flat_xland_sha256,
        "first_glacier_column": (
            int(glacier_indices[0]) if glacier_indices.size else None
        ),
        "glacier_index_sha256": proof.array_sha256(glacier_indices),
        "sea_ice_index_sha256": proof.array_sha256(sea_ice_indices),
        "threshold_0p02_to_0p5_delta_columns": int(threshold_delta_indices.size),
        "threshold_delta_index_sha256": proof.array_sha256(threshold_delta_indices),
        "native_xland_consumed_verbatim": True,
        "counts_measured_not_pinned": True,
    }

    if np.float32(nominal_min_dc).view(np.uint32) != NOMINAL_DX_M.view(np.uint32):
        raise RuntimeError("init nominalMinDc is not exact FP32 25000 m")

    # KEPT: exactness of the int32 -> FP32 landmask cast.  Digests recorded.
    source_landmask_sha256 = proof.array_sha256(landmask_i)
    source_unique_values = tuple(int(value) for value in np.unique(landmask_i))
    if landmask_i.shape != (N_CELLS,) or source_unique_values not in ((0, 1), (0,), (1,)):
        raise RuntimeError(
            f"init landmask identity is not int32 {{0,1}}[nCells]: "
            f"{landmask_i.shape} {source_unique_values}"
        )
    landmask = np.ascontiguousarray(landmask_i, dtype=np.float32)
    target_landmask_sha256 = proof.array_sha256(landmask)
    target_uint32_values = tuple(
        int(value) for value in np.unique(landmask.view(np.uint32))
    )
    if landmask.dtype != np.dtype(np.float32) or not np.array_equal(
        landmask.astype(np.int32), landmask_i
    ):
        raise RuntimeError("landmask int32 -> FP32 constructor cast is not exact")
    landmask_receipt = {
        "source_field": "init landmask",
        "source_dimensions": ["nCells"],
        "source_shape": list(landmask_i.shape),
        "source_dtype": landmask_i.dtype.str,
        "source_array_sha256": source_landmask_sha256,
        "source_unique_values": list(source_unique_values),
        "target_field": "SealedArwenConstructorV841.landmask",
        "target_shape": list(landmask.shape),
        "target_dtype": landmask.dtype.str,
        "target_array_sha256": target_landmask_sha256,
        "target_uint32_values": list(target_uint32_values),
        "value_preserving_exact_fp32_cast": True,
        "digests_measured_not_pinned": True,
    }

    if zgrid.shape != (N_CELLS, N_INTERFACES):
        raise ValueError(f"zgrid shape changed: {zgrid.shape}")

    lat = np.asarray(proof._mesh_value(mesh, "latCell"), dtype=np.float64)
    lon = np.asarray(proof._mesh_value(mesh, "lonCell"), dtype=np.float64)
    if lat.shape != (N_CELLS,) or lon.shape != (N_CELLS,):
        raise ValueError("reconciled mesh latitude/longitude shape changed")
    latitude_deg = np.ascontiguousarray(lat * (180.0 / np.pi), dtype=np.float32)
    longitude_deg = np.ascontiguousarray(lon * (180.0 / np.pi), dtype=np.float32)
    terrain = np.ascontiguousarray(zgrid[:, 0], dtype=np.float32)
    nominal_z = np.ascontiguousarray(np.asarray(vertical.zw))
    if nominal_z.shape != (N_INTERFACES,) or nominal_z.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise TypeError("loaded vertical.zw is not the exact 56-interface host vector")
    if not np.all(np.isfinite(nominal_z)) or np.any(np.diff(nominal_z) <= 0.0):
        raise ValueError("loaded vertical.zw is not finite and strictly increasing")

    # GF's per-cell length scale, native's own construction and the same one
    # this port already feeds GWDO: len_disp/meshDensity**0.25, with a
    # non-positive config_len_disp resolved to the mesh nominalMinDc.
    from woof.hex.cuda_gwdo_v841 import native_cell_dx_m

    dx_column_m = native_cell_dx_m(
        proof._mesh_value(mesh, "meshDensity"), float(NOMINAL_DX_M)
    )
    if dx_column_m.shape != (N_CELLS,):
        raise ValueError(
            f"per-cell GF dx must have shape {(N_CELLS,)}, got {dx_column_m.shape}"
        )

    p_top_seed = seed_p_top_expectation(
        pressure_base=reference.pressure_base,
        pressure_perturbation=saved_diagnostics.pressure_perturbation,
        zgrid=vertical.zgrid,
        area_cell=np.ascontiguousarray(proof._mesh_value(mesh, "areaCell")),
    )
    p_top_pa, p_top_receipt = proof.derive_area_weighted_p_top_v841(
        pressure_base=reference.pressure_base,
        pressure_perturbation=saved_diagnostics.pressure_perturbation,
        zgrid=vertical.zgrid,
        area_cell=np.ascontiguousarray(proof._mesh_value(mesh, "areaCell")),
    )

    # The seam's four clocks come from the CONFIG, never from the module
    # constant.  That is the whole of the 2026-08-26 rebinding fix: one
    # timestep travels from the bound mesh row through config_dt to here, so
    # the dycore's outer step and the frozen seam's step are the same number
    # by construction rather than by two rebinds agreeing.
    from woof.hex import convection_admission

    cumulus_scheme = convection_admission.constructor_scheme(
        config.config_convection_scheme
    )
    gf_ishallow = convection_admission.gf_ishallow(config.config_convection_scheme)

    values: dict[str, Any] = {
        "n_levels": N_LEVELS,
        "n_columns": N_CELLS,
        "dt": float(config.config_dt),
        "radiation_seconds": float(config.config_radt_seconds),
        "surface_pbl_seconds": float(config.config_bldt_seconds),
        # The cumulus selection comes from the CONFIG, never from a literal
        # here.  it was ruled on 2026-08-26 that convection is off below 3 km,
        # so "gf" written in this mapping would have silently reinstated the
        # closure the 3 km default switches off -- the config would say off and the
        # sealed constructor would be handed on.  See
        # woof.hex.convection_admission.
        "cumulus_seconds": (
            None
            if config.config_cudt_seconds is None
            else float(config.config_cudt_seconds)
        ),
        "cumulus_scheme": cumulus_scheme,
        # The seam row comes from the CONFIG's species row, never from a
        # literal: the engine constructs a WSM6 seam when the key is absent,
        # so a P3 or mp=28 request that left it out would seal a seam that
        # refuses its own first phase-one call by name.  The config already
        # admitted the scheme through require_engine_scheme().
        "microphysics_scheme": _row_for_scheme(
            config.config_microp_scheme
        ).engine_scheme,
        "start_time": start_datetime,
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "terrain_height_m": terrain,
        "z_interface_nominal_m": nominal_z,
        "p_top_pa": p_top_pa,
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_m": dx_column_m,
        # Native MPAS v8.4.1 hardwires GF's shallow scheme on
        # (mpas_atmphys_vars.F:340).  Derived from the selection rather than
        # written as 1: the sealed constructor refuses gf_ishallow=1 with no
        # GF selected, so a literal here would refuse every convection-off
        # run at host preparation.
        "gf_ishallow": gf_ishallow,
        "landmask": landmask,
        "xland": xland,
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "ivgtyp": np.ascontiguousarray(ivgtyp, dtype=np.int32),
        "isltyp": np.ascontiguousarray(isltyp, dtype=np.int32),
        **surface,
        **soil,
        "wsm6_hail_opt": 0,
    }
    arrays = {
        name: value for name, value in values.items() if isinstance(value, np.ndarray)
    }
    receipt = {
        "source": "exact initialized x4.163842 fields from the supplied init",
        "init_path": str(Path(init_path).absolute()),
        "config_start_time": start_text,
        "mapping": {
            "latitude_deg": "reconciled mesh latCell radians -> FP32 degrees",
            "longitude_deg": "reconciled mesh lonCell radians -> FP32 degrees",
            "terrain_height_m": "init zgrid[:,0]",
            "landmask": "init int32 {0,1} -> exact FP32 sealed-constructor cast",
            "xland": "init native xland[0,:] consumed verbatim",
            "xice_threshold": "explicit MPAS config_frac_seaice threshold 0.02",
            "z_interface_nominal_m": "loaded native vertical.zw",
            "tsk": "init skintemp[0,:]",
            "snow_depth": "init snowh[0,:] (m)",
            "soil_temperature": (
                "init tslb[0,:,:].T" if soil else
                "not read: the selection integrates no soil column"),
            "soil_moisture": (
                "init smois[0,:,:].T" if soil else
                "not read: the selection integrates no soil column"),
        },
        "p_top_pa": p_top_pa,
        "p_top_seed": p_top_seed,
        "landmask_exact_cast": landmask_receipt,
        "surface_classification": surface_classification_receipt,
        "p_top_derivation": p_top_receipt,
        "p_top_policy": "exact areaCell-weighted F000 native pres2_p top",
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_policy": "native len_disp/meshDensity**0.25 per cell",
        "dx_column_min_m": float(dx_column_m.min()),
        "dx_column_max_m": float(dx_column_m.max()),
        "gf_ishallow": gf_ishallow,
        "cumulus_scheme": cumulus_scheme,
        "config_convection_scheme": config.config_convection_scheme,
        "defaults_used": False,
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": proof.array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        },
    }
    static_for_gwdo = {
        "meshDensity": np.asarray(proof._mesh_value(mesh, "meshDensity")),
        "nominalMinDc": np.asarray(NOMINAL_DX_M),
    }
    for name in ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"):
        with Dataset(init_path, "r") as dataset:
            static_for_gwdo[name] = read(
                dataset, name, dtype=np.float32, dimensions=("nCells",)
            )
    return values, receipt, static_for_gwdo, surface_classification


# --------------------------------------------------------------------------
# host preparation (transcribed from proof._prepare_host_execution, with the
# measurement/rebinding interleaved)
# --------------------------------------------------------------------------
#: Set by the same-host adapter while it is installed
#: (samehost_forecast_v841.installed): the ranks join the prepared-input
#: fingerprint after they are built, instead of main() joining it first.
DEFER_PREPARED_FINGERPRINT = False

#: Set by the resident-member roster (woof.hex.multi_member) on each member's private module: the forecast
#: copies its final-identity carriers to the host and returns with its host-only tail (history drain,
#: identity hash) as ``_deferred_tail``, so the roster can release the card first (woof.hex.card_handoff).
DEFER_TAIL = False


def join_prepared_fingerprint(host: Mapping[str, Any]) -> None:
    """Finish the prepared-input validation (cuda_dualrun.BackgroundFingerprintSeal).

    Idempotent; re-raises the validation's refusal on the caller's thread.
    """

    seal = host.get("prepared_fingerprint_seal")
    if seal is not None:
        _timed_preparation("prep.host_validation_join", seal.join)


def _timed_preparation(stage, function, *args, **kwargs):
    """Time an existing producer without changing its inputs or returned object.

    The earlier startup summary combined all host reads and transformations,
    so it could not attribute the reported startup cost to the init or static.
    These host spans include the producer's existing transformations and cache
    hits. They do not add device synchronization or field copies.
    """
    with time_ledger.stage(stage):
        return function(*args, **kwargs)


def prepare_forecast_host(
    paths: Mapping[str, Path],
    authority_receipt: Mapping[str, Any],
    *,
    start_time_text: str | None,
    horiz_mixing: str = "2d_smagorinsky",
    convection: str = "auto",
    pbl_cadence: str = "auto",
    local_timestep: bool = False,
    local_timestep_declared_off: bool = False,
    local_timestep_rates: tuple[int, ...] = (1, 3),
    local_timestep_buffer_rings: int = 1,
    lbc_paths: Sequence[str] | None = None,
    physics_backend: str = "wsm6_column",
    source_table: str | None = None,
    time_integration_order: int = 2,
    acoustic_substeps: int = 4,
    radiation_seconds: float = 1800.0,
    monotonic: bool = True,
    cam_coefficient: float = 0.0,
    cam_damping_levels: int = 4,
    rayleigh_u: bool = False,
    damping_start_m: float = 22000.0,
    invariant_path: Path | None = None,
    orographic_statistics: Path | None = None,
    physics: Any = None,
    dynamics_overrides: Mapping[str, Any] | None = None,
    physics_mode: str = "exact",
    physics_correct_skip: Sequence[str] = (),
) -> dict[str, Any]:
    from woof.hex.cuda_arwen_physics_v841 import SealedArwenConstructorV841
    from woof.hex.cuda_dualrun import BackgroundFingerprintSeal, PreparedCudaInputs
    from woof.hex.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from woof.hex.dynamics_v841 import load_v841_reference_wind_profiles
    from woof.hex.mesh_snapshot import load_dynamics_mesh

    # The claim names the physics this run resolved (claim_for_physics); the
    # bind has already named its convection decision in the same sentence.
    global CLAIM
    CLAIM = claim_for_physics(CLAIM, physics)
    relaxation: dict[str, Any] = {"init_carriers": relax_init_carrier_pins(paths["init"])}

    # DT_SECONDS is the module constant bind_mesh rebinds to the bound row's
    # declared timestep.  Reading it HERE is what closes the 2026-08-26
    # rebinding defect: before this, the config took its dataclass default
    # and the two clocks diverged silently until composite step 0.
    # The cumulus selection travels the same road as the timestep: decided
    # once at the bind from the mesh's own finest spacing, read here.  If a
    # bind happened and its request disagrees with this one, that is TWO
    # sources of the same decision -- the exact shape of the 2026-08-26 clock
    # defect -- and it is refused on the host rather than discovered in the
    # receipt of a finished run.
    from woof.hex import convection_admission as _convection

    decision = dict(CONVECTION_DECISION) if CONVECTION_DECISION else None
    if decision is not None and decision.get("requested") != convection:
        raise ValueError(
            f"the bound mesh decided its convection selection under "
            f"--convection {decision.get('requested')!r} and this run was "
            f"invoked with --convection {convection!r}.  One decision, one "
            f"source: the bind's request and the driver's must be the same "
            f"string, or the receipt would name a selection the run did not "
            f"make"
        )
    if decision is None:
        decision = _convection.convection_decision(
            nominal_dx_m=float(NOMINAL_DX_M), requested=convection
        )
    convection_scheme = decision["scheme"]
    print(
        f"[convection] {decision['scheme']} ({decision['source']}): "
        f"{decision['note']}",
        flush=True,
    )

    # The surface/PBL cadence travels the same road, for the same reason and
    # with the same refusal: one decision, one source.  ``auto`` is the weld
    # (config_bldt_seconds = config_dt), which is the proven configuration
    # and the default; an explicit cadence is an A/B arm.
    from woof.hex import pbl_cadence as _pbl

    pbl_decision = dict(PBL_CADENCE_DECISION) if PBL_CADENCE_DECISION else None
    if pbl_decision is not None and pbl_decision.get("requested") != pbl_cadence:
        raise ValueError(
            f"the bound mesh decided its surface/PBL cadence under "
            f"--pbl-cadence {pbl_decision.get('requested')!r} and this run "
            f"was invoked with --pbl-cadence {pbl_cadence!r}.  One decision, "
            f"one source: the bind's request and the driver's must be the "
            f"same string, or the receipt would name a cadence the run did "
            f"not use"
        )
    if pbl_decision is None:
        pbl_decision = _pbl.pbl_cadence_decision(
            dt_seconds=float(DT_SECONDS), requested=pbl_cadence,
            radiation_seconds=float(radiation_seconds),
        )
    print(
        f"[pbl-cadence] {pbl_decision['label']} ({pbl_decision['source']}): "
        f"{pbl_decision['note']}",
        flush=True,
    )

    config = build_forecast_config(
        dt_seconds=float(DT_SECONDS),
        convection_scheme=convection_scheme,
        surface_pbl_seconds=pbl_decision["surface_pbl_seconds"],
        horiz_mixing=horiz_mixing,
        local_timestep=local_timestep,
        local_timestep_declared_off=local_timestep_declared_off,
        local_timestep_rates=local_timestep_rates,
        local_timestep_buffer_rings=local_timestep_buffer_rings,
        apply_lbcs=bool(lbc_paths),
        time_integration_order=time_integration_order,
        acoustic_substeps=acoustic_substeps,
        radiation_seconds=radiation_seconds,
        monotonic=monotonic,
        cam_coefficient=cam_coefficient,cam_damping_levels=cam_damping_levels,
        rayleigh_u=rayleigh_u,damping_start_m=damping_start_m,
        dynamics_overrides=dynamics_overrides,
    )
    backend_row, scheme_alias, seam_options = resolve_run_row(
        physics_backend, source_table
    )
    if scheme_alias is not None:
        # A provider's row names itself as a config_microp_scheme alias
        # (woof.hex.species_row), so the dynamics driver's width check, its
        # receipts and the configuration's engine-scheme admission all
        # resolve the RUN's row from the config.
        from dataclasses import replace as _replace

        config = _replace(config, config_microp_scheme=scheme_alias)
    suite_adapter = _suite_adapter(backend_row)
    if suite_adapter is not None:
        # The suite row's configuration records the MPAS selectors the run
        # EXECUTES (MYNN, RUC, the GSL drag), not the frozen lane's literals.
        config = suite_adapter.suite_config(
            config,
            backend_row.name,
            mpas_selectors=(
                None if physics is None else {
                    key: value for key, value in physics.selectors.items()
                    if key != "config_convection_scheme"
                }
            ),
        )
        _rebind_soil_levels(suite_adapter.soil_levels_for(backend_row.name))
    elif orographic_statistics is not None:
        raise ConfigurationRefusal(
            "orographic_statistics",
            str(orographic_statistics),
            f"--physics-backend {backend_row.name} runs no drag that reads "
            "orographic statistics; the file would be accepted and never read",
            "a row whose drag reads the mesh's orographic statistics",
        )
    if physics is not None and not physics.is_default():
        # The configuration records the named suite the run was resolved
        # from (woof.hex.physics_suites).  The default suite changes nothing,
        # so a run without a suite flag builds the identical configuration.
        from dataclasses import replace as _replace_suite

        config = _replace_suite(config, config_physics_suite=physics.suite)
    from woof.hex import arwen_les
    if arwen_les.active(config):
        from dataclasses import replace
        config = replace(config, config_pbl_scheme="off",
                         config_les_surface="varying", config_mix_scalars=True)
    arwen_les.validate(config)

    run_row = _row_for_scheme(config.config_microp_scheme)
    scalar_names = tuple(run_row.names())
    surface_accumulators = tuple(item.name for item in run_row.surface_accumulators)
    # The loader's dynamics mesh and evidence, served from a verified snapshot
    # of the same grid/static bytes when one exists (woof.hex.mesh_snapshot);
    # the grid-only output mesh was discarded here and is not built.
    mesh, mesh_evidence = _timed_preparation("prep.static_read", load_dynamics_mesh,
        paths["grid"], paths["static"]
    )
    # A limited-area grid declares itself: the cull writes the
    # bdyMaskCell/Edge/Vertex triple and nothing else does.  Everything below
    # is the SAME preparation the global lane runs -- the sentinel flag says
    # "this mesh's outermost ring has one-cell edges by construction", it does
    # not select a different code path.
    is_regional = bool(getattr(mesh, "is_regional", False))
    if is_regional and not lbc_paths:
        raise ConfigurationRefusal(
            "config_apply_lbcs",
            True,
            "this grid carries a bdyMask triple, so its outermost ring is a "
            "lateral boundary that something has to drive; integrating it with "
            "no boundary series lets the interior run against whatever the "
            "initial state left on the ring, and the domain empties from the "
            "edge inward within a few hours",
            "a --lbc-dir of files rw_mpas_lbc built from the parent forecast "
            "this mesh was cut out of",
        )
    if not is_regional and lbc_paths:
        raise ConfigurationRefusal(
            "config_apply_lbcs",
            True,
            "this grid carries no bdyMask triple, so it has no boundary zone "
            "for a lateral-boundary series to drive; the files would be read "
            "and never applied, and the receipt would name a forcing the run "
            "did not use",
            "a limited-area grid cut with rw_mpas_mesh --cull-parent, or no "
            "--lbc-dir",
        )
    reconstruction_overlay = proof.overlay_exact_init_reconstruction_coefficients(
        mesh, paths["init"]
    )
    edge_normal_overlay = proof.overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    defc = prepare_inactive_deformation(mesh)
    native = _timed_preparation("prep.init_vertical", load_mpas_vertical_grid,
        paths["init"],
        mesh,
        config_coef_3rd_order=config.config_coef_3rd_order,
        allow_regional_sentinels=is_regional,
        invariant_path=invariant_path,
    )
    from netCDF4 import Dataset
    with Dataset(paths["init"], "r") as dataset:
        source_scalar_names = select_initial_scalar_names(dataset.variables, scalar_names)
    state, reference, saved = _timed_preparation("prep.init_state", load_mpas_initial_state,
        paths["init"],
        mesh,
        native.vertical_grid,
        scalar_names=source_scalar_names,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
        allow_regional_sentinels=is_regional,
        # A forecast always starts cold from its init: native MPAS-A then
        # recomputes pressure_p/pressure_base/exner (F:7590-7650) rather than
        # consuming the init program's copies.
        cold_start=True,
    )
    relaxation["negative_qv"] = relax_negative_qv_pin(state)
    scalar_receipt = prepare_initial_scalar_block(
        state, source_names=source_scalar_names, model_names=scalar_names,
        negative_qv=relaxation["negative_qv"],
    )
    les_theta_initial = None
    if arwen_les.active(config):
        from woof.netcdf_bridge import open_dataset
        with open_dataset(paths["init"]) as dataset:
            les_theta_initial = np.ascontiguousarray(
                np.asarray(dataset.variables["t_init"][:], dtype=np.float32).T)
        model_names = arwen_les.model_row(config).names()
        if "tke" in model_names:
            # Native cold initialization is zero TKE. Production uses the
            # existing native minimum viscosity, without an added seed.
            from woof.netcdf_bridge import open_dataset
            with open_dataset(paths["init"]) as dataset:
                if "tke" in dataset.variables:
                    tke = np.asarray(dataset.variables["tke"][:], dtype=np.float32)
                    if tke.ndim == 3:
                        tke = tke[0]
                    if tke.shape != (N_CELLS, N_LEVELS):
                        raise ValueError(f"init tke shape changed: {tke.shape}")
                    tke = np.ascontiguousarray(tke.T)
                    source = "initialized tke"
                else:
                    tke = np.zeros((N_LEVELS, N_CELLS), dtype=np.float32)
                    source = "native cold zero"
            state.scalars = np.concatenate((state.scalars, tke[None]), axis=0)
            scalar_receipt["tke_initialization"] = source
        scalar_names = tuple(model_names)
        scalar_receipt["scalar_order"] = list(scalar_names)
    state.validate(n_cells=N_CELLS, n_edges=N_EDGES, n_vert_levels=N_LEVELS)
    saved.validate((N_LEVELS, N_CELLS), np.dtype(np.float32), N_EDGES)
    profiles = _timed_preparation("prep.init_wind", load_v841_reference_wind_profiles,
                                 paths["init"], n_vert_levels=N_LEVELS)
    prepared = _timed_preparation("prep.host_validation", PreparedCudaInputs.validated,
        config=config,
        profile=proof.PROFILE,
        target=CLAIM,
        preparation_method=(
            "precision-preserving grid/static overlay plus exact initialized "
            "reconstruction coefficients and edge-normal vectors; initialized "
            "source species preserved in model order, with cold augmentation "
            "only when the init carries the leading three species"
        ),
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=dict(authority_receipt["files"]),
        reference_wind_profiles=profiles,
        allow_regional_sentinels=is_regional,
        seal=False,
    )
    # The execution fingerprint (the rest of the validation) runs on a thread
    # while the init's surface fields and the physics constructor are read and
    # built; it is joined, and any refusal raised, before host preparation
    # returns (cuda_dualrun.BackgroundFingerprintSeal).
    fingerprint_seal = BackgroundFingerprintSeal(prepared, config)
    f000_surface_diagnostics = _timed_preparation("prep.init_surface", proof.load_f000_initialized_surface_diagnostics,
        paths["init"]
    )
    f000_start_humidity = _timed_preparation("prep.init_humidity", load_f000_start_humidity, paths["init"])
    (
        constructor_values,
        constructor_receipt,
        gwdo_host,
        classification,
    ) = _timed_preparation("prep.physics_constructor", build_forecast_constructor_values,
        init_path=paths["init"],
        mesh=mesh,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        start_time_text=start_time_text,
        config=config,
        n_soil_levels=N_SOIL_LEVELS,
    )
    if arwen_les.active(config):
        constructor_values["pbl_scheme"] = None
        constructor_receipt["turbulence"] = "resident cuda_les_v842; WOOF surface fluxes"
    # physics.mode (woof.hex.correct_mode; MODE-KEY.md).  Exact adds nothing to
    # the mapping, so the sealed constructor and its identity are unchanged.
    from woof.hex import correct_mode as _correct_mode

    _mode, _skip = _correct_mode.validate(physics_mode, physics_correct_skip)
    if _mode != _correct_mode.PHYSICS_MODE_EXACT:
        constructor_values["physics_mode"] = _mode
        constructor_values["physics_correct_skip"] = _skip
        constructor_receipt["physics_mode"] = {
            "physics.mode": _mode, "physics.correct_skip": list(_skip)}
    if is_regional:
        # The ArWen statics move onto the PADDED extent here, BEFORE the
        # surface census is rebound, because the census counts columns and
        # the run seals nCells+1 of them.  Doing it after left the seam
        # reporting 11,021 classified columns against an expectation of
        # 11,020 and the first step receipt refused by exactly one column.
        from woof.hex.cuda_regional_forecast_v841 import pad_regional_physics_host, pad_regional_physics_column
        from woof.hex.regional_v841 import derive_regional_masks, regional_bdy_checks

        regional_masks = derive_regional_masks(mesh, np.dtype(np.float32))
        # mpas_atm_bdy_checks, on the host, before a byte moves: a mesh with
        # boundary cells and no LBCs, or LBCs and no boundary cells, is
        # refused by name here rather than discovered in a finished receipt.
        regional_bdy_checks(
            regional_masks, config_apply_lbcs=True, lbc_input_interval_valid=True
        )
        constructor_values, gwdo_host, pad_receipt = pad_regional_physics_host(
            constructor_values, gwdo_host, n_cells_solve=int(N_CELLS),
            donor_cell=getattr(mesh, "attrs", {}).get("regional_garbage_donor_cell"),
        )
        constructor_receipt["regional_physics_pad"] = pad_receipt
        if suite_adapter is None:
            classification = dict(
                SealedArwenConstructorV841.from_mapping(
                    constructor_values
                ).expected_surface_classification()
            )
    seal = SealedArwenConstructorV841.from_mapping
    if suite_adapter is not None:
        constructor_values, gwdo_host, classification = _suite_host(
            suite_adapter,
            backend_row.name,
            constructor_values,
            constructor_receipt,
            orographic_statistics=orographic_statistics,
            is_regional=is_regional,
            cell_area_m2=(
                pad_regional_physics_column(np.asarray(proof._mesh_value(mesh, "areaCell"), dtype=np.float64),
                    donor_cell=getattr(mesh, "attrs", {}).get("regional_garbage_donor_cell"))
                if is_regional else np.asarray(proof._mesh_value(mesh, "areaCell"), dtype=np.float64)),
        )
        seal = suite_adapter.seal_constructor
        relaxation["surface"] = relax_suite_surface_classification(classification)
    else:
        relaxation["surface"] = relax_surface_classification(
            classification, proof.ARWEN_GLACIER_CUDA_PROVENANCE
        )
    relaxation["p_top"] = constructor_receipt["p_top_seed"]
    sealed_constructor_audit = seal(constructor_values)
    # Belt and braces on the wiring above: the seam's clocks are DERIVED from
    # the config, so this can only fail if somebody reintroduces a second
    # source of the timestep.  It costs nothing and it is the difference
    # between a host refusal and 18,820 MiB plus 285 s on a card.
    coherence = dt_admission.require_step_clock_coherence(
        config_dt=config.config_dt,
        constructor_dt=sealed_constructor_audit.dt,
        config_radt_seconds=config.config_radt_seconds,
        constructor_radiation_seconds=constructor_values["radiation_seconds"],
        config_bldt_seconds=config.config_bldt_seconds,
        constructor_surface_pbl_seconds=constructor_values["surface_pbl_seconds"],
        config_cudt_seconds=config.config_cudt_seconds,
        constructor_cumulus_seconds=constructor_values["cumulus_seconds"],
    )
    constructor_receipt["step_clock_coherence"] = coherence
    from woof.hex import convection_admission as _convection

    constructor_receipt["dt_admission"] = step_admission_receipt(config)
    # config_bldt_seconds is part of the anchor LOOKUP key as well as the
    # comparison, so a run holding the surface/PBL cadence is recorded
    # against the row that measured that cadence and never the welded one.
    constructor_receipt["sealed_host_contract_audit"] = {
        "authority": (
            "SealedArwenConstructorV841.from_mapping"
            if suite_adapter is None
            else f"{suite_adapter.__name__}.seal_constructor"
        ),
        "accepted": True,
        "all_required_keys_dtypes_shapes_validated": True,
        "array_fields": sorted(constructor_receipt["arrays"]),
        **dict(sealed_constructor_audit.receipt()),
    }
    constructor_receipt["convection_admission"] = decision
    constructor_receipt["pbl_cadence"] = pbl_decision
    regional: dict[str, Any] | None = None
    if is_regional:
        # The species the boundary files actually carry, intersected with
        # Drive the model-ordered mass species the complete hourly series
        # supplies.  A three-species stream retains its existing treatment;
        # a native stream with ice retains those supplied fields as well.
        # A suite row names the block a boundary stream may drive (its
        # driven_scalar_names, the model's leading species); the stream's
        # other species -- an aerosol pair a native init_atmosphere writes
        # beside the masses -- are supplied and not driven, and the record
        # says so.  The frozen row keeps its exact call.
        # A row that APPENDS scalars after its driven block (a suite row, or
        # thompson_aero_column) drives only that block: an MPAS boundary
        # stream carries no ni/nr/nc, so the full model list is never a
        # prefix of what it supplies.  The frozen row appends nothing and
        # keeps its exact call.
        drives_block_only = (
            suite_adapter is not None or bool(backend_row.appended_scalar_names) or arwen_les.active(config)
        )
        driven_block = (
            tuple(backend_row.driven_scalar_names)
            if drives_block_only
            else scalar_names
        )
        if tuple(scalar_names[: len(driven_block)]) != tuple(driven_block):
            raise ValueError(
                f"row {backend_row.name} drives {driven_block}, which is not "
                f"the leading block of the model species {scalar_names}"
            )
        driven = select_lbc_scalar_names(lbc_paths or (), driven_block)
        undriven: list[str] = []
        if drives_block_only and lbc_paths:
            undriven = supplied_undriven_scalars(lbc_paths[0], scalar_names, driven)
        regional = {
            "driven_scalars": driven,
            "supplied_undriven_scalars": undriven,
            "lbc_paths": [str(path) for path in lbc_paths or ()],
            "start_time": datetime.strptime(
                constructor_receipt["config_start_time"], "%Y-%m-%d_%H:%M:%S"
            ),
            "n_cells_solve": int(N_CELLS),
            "boundary_zone_width": int(REGIONAL_BOUNDARY_ZONE_WIDTH),
            "free_interior_cells": int(
                np.count_nonzero(regional_masks.bdy_mask_cell == 0)
            ),
            "specified_zone_cells": int(regional_masks.spec_cells.size),
            "relaxation_zone_cells": int(regional_masks.relax_cells.size),
            "specified_zone_edges": int(regional_masks.spec_edges.size),
            "relaxation_zone_edges": int(regional_masks.relax_edges.size),
            "bdy_mask_sha256": regional_boundary_mask_digest(
                {
                    name: mesh.arrays[name]
                    for name in REGIONAL_BOUNDARY_MASK_NAMES
                    if name in mesh.arrays
                }
            ),
        }
    selected_physics = constructor_values.get("physics")
    if selected_physics is not None and "ishallow" in selected_physics:
        # A registry-resolved selection that names GF's shallow switch runs
        # with that value; the mesh-bind decision only knows the scheme's
        # name and would record the MPAS hardwire (1) for a run handed 0.
        decision = {**decision, "gf_ishallow": int(selected_physics["ishallow"])}
    return {
        "config": config,
        "surface_init_path": str(paths["init"]),
        "regional": regional,
        "convection": decision,
        "pbl_cadence": pbl_decision,
        "prepared": prepared,
        "constructor_values": constructor_values,
        "constructor_receipt": constructor_receipt,
        "gwdo_host": gwdo_host,
        "mesh_evidence": mesh_evidence,
        "f000_surface_diagnostics": f000_surface_diagnostics,
        "f000_start_humidity": f000_start_humidity,
        "reconstruction_coefficients": reconstruction_overlay,
        "edge_normal_vectors": edge_normal_overlay,
        "defc": defc,
        "scalar_receipt": scalar_receipt,
        "physics_backend": backend_row.name,
        "physics_resolution": None if physics is None else physics.receipt(),
        "scalar_names": scalar_names,
        "les_theta_initial": les_theta_initial,
        "surface_accumulators": surface_accumulators,
        "surface_accumulator_declarations": run_row.surface_accumulators,
        "seam_options": dict(seam_options),
        "physics_suite": (
            None
            if suite_adapter is None
            else _suite_record_for_run(
                suite_adapter.physics_suite_record(backend_row.name), physics)
        ),
        "case_pin_relaxation": relaxation,
        "start_time_text": constructor_receipt["config_start_time"],
        # Joined by join_prepared_fingerprint before the prepared inputs are
        # used: in main() for a one-card run, after the ranks are built for a
        # same-host run (samehost_forecast_v841).
        "prepared_fingerprint_seal": fingerprint_seal,
    }


# --------------------------------------------------------------------------
# per-step health gate (kept: finite/positive laws; cheap device reductions)
# --------------------------------------------------------------------------
_STATE_HEALTH_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
_SAVED_HEALTH_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)


def install_step_gate_batch(stack: Mapping[str, Any], cp: Any) -> None:
    """Read final validation and health together, before driver publication.

    Health refusals are evaluated in their existing order after the numeric
    gate. The backend commit only publishes references and bookkeeping;
    it does not write the endpoint arrays described here.
    """
    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels

    driver = stack["driver"]

    def read_endpoint(atmosphere: Any, flag: Any) -> int:
        driver._step_gate_envelope = None
        state, saved = atmosphere.state, atmosphere.saved
        extents = {int(v) + 1: int(v) for v in
                   (stack.get("solve_cells"), stack.get("solve_edges")) if v is not None}
        fields = []
        for owner, names in ((state, _STATE_HEALTH_FIELDS), (saved, _SAVED_HEALTH_FIELDS)):
            for name in names:
                array = cp.asarray(getattr(owner, name))
                if array.dtype.kind == "f":
                    fields.append((name, array, extents.get(int(array.shape[-1]))))
        scalars = cp.asarray(state.scalars)
        solve = extents.get(int(scalars.shape[-1]))
        fields.extend((("scalars[0]", scalars[0], solve),
                       ("scalars[1:]", scalars[1:], solve)))
        measured, words = SolveRegionKernels.for_cache(driver.cache).envelope_with_flags(fields, [flag])
        if words[0] == 0:
            driver._step_gate_envelope = (atmosphere, measured)
        return words[0]

    driver.final_validation_reader = read_endpoint


def may_capture_refusal_state(refusal: Mapping[str, Any]) -> bool:
    """A health-refused endpoint must never become a bonus history frame."""
    return refusal.get("refused_in") != "step health gate"


def retired_physics_seam(backend: Any) -> dict[str, Any] | None:
    """``None`` when the physics adapter sits at a committed step boundary.

    Otherwise what the refused step left: the adapter's phase and why no
    physics restart state exists.  ``restart_state`` -- and every digest
    built from it -- is legal only at a boundary, so after a refused step
    whose rollback could not rebuild the seam (phase ``broken``) or that
    armed no rollback (``rollback_not_armed``), asking for it ended the run
    with "restart_state is legal only at a step boundary" and no receipt,
    and the step's own refusal was never reported.
    """

    phase = getattr(backend, "_phase", "boundary")
    if phase == "boundary":
        return None
    return {
        "phase": str(phase),
        "restart_state": None,
        "note": (
            "the refused step left the physics adapter without a committed "
            "boundary (its rollback did not rebuild the seam), so this run "
            "has no physics restart state and no final-state digest; the "
            "MPAS state is the last committed step's"
        ),
    }


def step_health_gate(
    stack: Mapping[str, Any], step: int, cp: Any, *, trace_hot_cell: bool = False,
    measured_health: Any = None,
) -> dict[str, Any]:
    """Refuse the moment the integration stops being finite and physical.

    Deliberately allocation-light.  The frozen Arwen phase-one seam is
    documented as sensitive to the CuPy device-pool layout, so this gate
    never materializes a full-size temporary such as ``isfinite(x)``.  The
    envelope of every field -- min, max, a NaN verdict and the argmax of
    |w| -- comes from one launch pair over the solve region
    (``woof.hex.cuda_solve_region_v841``) and ONE read of a few hundred
    bytes.  min/max propagate NaN and carry +/-inf, so testing the two
    scalars is a complete finiteness test for the array, and the values are
    the ones ``cp.min``/``cp.max`` on the trimmed views returned: the
    reductions are exact and order-independent.

    Before the fused envelope this gate was 31 CuPy reductions on strided
    ``[..., :n_solve]`` views, each on one block of 512 threads and each
    read back with a drain: 78 ms of every 0.38 s step, a fifth of it
    (``evidence-gallery/hex-perf-profile-2026-09-13``).
    """

    atmosphere = stack["driver"].atmosphere
    state = atmosphere.state
    saved = atmosphere.saved
    groups = [(name, getattr(state, name)) for name in _STATE_HEALTH_FIELDS]
    groups += [(name, getattr(saved, name)) for name in _SAVED_HEALTH_FIELDS]
    # On a limited-area run every array is one element wider than the domain
    # it solves: native allocates nCells+1 (and nEdges+1) and holds pool
    # values in that element -- rho_theta, theta_m and exner are the pool
    # ZERO there by native's own rule.  This gate refuses a non-positive
    # theta_m, so reducing over the allocation instead of the domain refuses
    # every limited-area step at step 1 for a column that is not a column.
    # The bound is the model's, not a tolerance: the health of an element
    # native never solves is not a statement about the forecast.
    solve_cells = stack.get("solve_cells")
    solve_edges = stack.get("solve_edges")
    padded_extents = {
        int(value) + 1: int(value)
        for value in (solve_cells, solve_edges)
        if value is not None
    }
    owned_extents = stack.get("owned_health_extents", {})

    def _solve(array: Any) -> int | None:
        """How many of the last dimension's elements are solved (None: all)."""

        if owned_extents:
            extent = int(array.shape[-1])
            if extent not in owned_extents:
                raise ValueError("rank health field has an unknown entity extent; prevents skipping an owned forecast value")
            return int(owned_extents[extent])
        if not padded_extents:
            return None
        return padded_extents.get(int(array.shape[-1]))

    def _domain(array: Any) -> Any:
        trimmed = _solve(array)
        return array if trimmed is None else array[..., :trimmed]

    from woof.hex.cuda_solve_region_v841 import SolveRegionKernels

    fields: list[tuple[str, Any, int | None]] = []
    for name, value in groups:
        array = cp.asarray(value)
        if array.dtype.kind != "f":
            continue
        fields.append((name, array, _solve(array)))
    scalars_array = cp.asarray(state.scalars)
    scalars_solve = _solve(scalars_array)
    # qv and the hydrometeors are two more views of the same block: the
    # first species, and every species after it, each C-contiguous.
    fields.append(("scalars[0]", scalars_array[0], scalars_solve))
    fields.append(("scalars[1:]", scalars_array[1:], scalars_solve))
    pending = measured_health
    if pending is None:
        pending = getattr(stack["driver"], "_step_gate_envelope", None)
        stack["driver"]._step_gate_envelope = None
    if pending is not None and pending[0] is atmosphere and not owned_extents:
        measured = pending[1]
    else:
        measured = SolveRegionKernels.for_cache(stack["driver"].cache).envelope(fields)

    envelope: dict[str, list[float]] = {}
    nonfinite: list[str] = []
    for name, _value in groups:
        if name not in measured:
            continue
        low, high, _has_nan, _argmax = measured[name]
        envelope[name] = [low, high]
        if not (math.isfinite(low) and math.isfinite(high)):
            nonfinite.append(name)
    if nonfinite:
        raise FloatingPointError(
            f"step {step} produced non-finite {sorted(nonfinite)}: "
            + json.dumps({name: envelope[name] for name in sorted(nonfinite)})
        )
    rho_min = envelope["rho"][0]
    theta_min = envelope["theta_m"][0]
    exner_min = envelope["exner"][0]
    if rho_min <= 0.0 or theta_min <= 0.0 or exner_min <= 0.0:
        raise FloatingPointError(
            f"step {step} rho/theta_m/exner not strictly positive: "
            f"{(rho_min, theta_min, exner_min)}"
        )
    w_low, w_high = envelope["vertical_velocity"]
    w_abs_max = max(abs(w_low), abs(w_high))
    # WHERE, not only how big.  On a limited-area run the single most useful
    # fact about a growing vertical velocity is which boundary ring it sits
    # in: ring 0 is the free interior and the forecast owns it, rings 1-7 are
    # driven and a maximum there is the boundary treatment talking.  The two
    # readings cost one argmax on a state already resident.
    hot: dict[str, Any] | None = None
    if trace_hot_cell:
        # The envelope kernel already found the first-index argmax of |w|
        # over the trimmed view; the flat index is C-order inside that view,
        # exactly what cupy.argmax(cupy.abs(w[..., :n_solve])) returned.
        w_field = _domain(cp.asarray(saved.vertical_velocity))
        flat = int(measured["vertical_velocity"][3])
        cell = flat % int(w_field.shape[-1])
        hot = {
            "vertical_velocity_max_abs": w_abs_max,
            "cell": int(cell),
            "level": int(flat // int(w_field.shape[-1])),
        }
        rings = stack.get("bdy_mask_cell")
        if rings is not None and cell < len(rings):
            hot["boundary_ring"] = int(rings[cell])
            hot["zone"] = (
                "free interior"
                if int(rings[cell]) == 0
                else f"driven boundary ring {int(rings[cell])} of 7"
            )
    if w_abs_max > VERTICAL_VELOCITY_REFUSAL_M_S:
        raise FloatingPointError(
            f"step {step} vertical velocity {w_abs_max} m/s exceeds the "
            f"{VERTICAL_VELOCITY_REFUSAL_M_S} m/s divergence refusal"
            + ("" if hot is None else f"; worst column {json.dumps(hot)}")
        )
    qv_min, qv_max, _qv_nan, _qv_arg = measured["scalars[0]"]
    hydrometeor_min = measured["scalars[1:]"][0]
    if not (
        math.isfinite(qv_min) and math.isfinite(qv_max) and math.isfinite(hydrometeor_min)
    ):
        raise FloatingPointError(f"step {step} produced non-finite moisture")
    if hydrometeor_min < 0.0:
        raise FloatingPointError(
            f"step {step} left a negative hydrometeor: {hydrometeor_min}"
        )
    return {
        "step": step,
        "rho_min": rho_min,
        "theta_m_min": theta_min,
        "theta_m_max": envelope["theta_m"][1],
        "exner_min": exner_min,
        "vertical_velocity_abs_max": w_abs_max,
        "qv_min": qv_min,
        "qv_max": qv_max,
        "hydrometeor_min": hydrometeor_min,
        "hot_cell": hot,
        "finite": True,
        "entity_scope": "owned partition union" if owned_extents else "native solve region",
        "owned_health_extents": dict(owned_extents),
    }


# --------------------------------------------------------------------------
# evidence writers (partition-lane compatible)
# --------------------------------------------------------------------------
class UnarmedForecastSnapshotV841:
    """The late-health rollback of a run that has no reader for one.

    Without ``--stop-on-refusal`` a refused step ends the run, so the two
    committed boundaries :class:`TwoDeepForecastSnapshotV841` retains are
    never restored.  This takes no copy and holds no device memory; a
    restore is a defect in the caller and refuses by name.
    """

    captures = rollbacks = 0

    def capture(self, step: int, surface: Any, dynamics: Any) -> None:
        del step, surface, dynamics

    def restore(self, step: int) -> tuple[Any, Any]:
        raise RuntimeError(
            f"late health rollback to boundary {step} was requested, but this "
            "run retained no boundary: without --stop-on-refusal a refused step "
            "ends the run")

    def receipt(self) -> dict[str, Any]:
        return {"depth": 0, "device_bytes": 0, "host_bytes": 0,
                "captures": 0, "rollbacks": 0, "slot_device_bytes": [],
                "memory_scope": "no boundary retained: without --stop-on-refusal "
                                "a refused step ends the run and nothing reads one"}


class BoundaryFingerprintWriter:
    """Append-only ``step -> {atmosphere, backend}`` JSONL, partstream format."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._stream = self.path.open("w", encoding="utf-8")
        self._steps: list[int] = []

    def write(self, step: int, fingerprint: Mapping[str, Any]) -> None:
        if self._steps and step <= self._steps[-1]:
            raise ValueError(
                f"boundary fingerprints must ascend: {step} after {self._steps[-1]}"
            )
        self._steps.append(int(step))
        record = {"step": int(step), **dict(fingerprint)}
        self._stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def close(self) -> None:
        self._stream.close()


def _snapshot_q2_hash(snapshot: Mapping[str, Any]) -> str | None:
    value = snapshot["arrays"].get("q2")
    return None if value is None else proof.array_sha256(value)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
def build_schedule(
    *, hours: float, history_every_minutes: float, start_text: str, history_from_minutes: float = 0.0
) -> dict[str, Any]:
    total_seconds = float(hours) * 3600.0
    steps = total_seconds / DT_SECONDS
    if steps <= 0 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(
            f"--hours {hours} is not a whole number of {DT_SECONDS:.0f} s steps"
        )
    steps = int(round(steps))
    history_seconds = float(history_every_minutes) * 60
    # Counted against the step itself, not its integer part (see the door's
    # _schedule): an 18.75 s step truncated to 18 s gave a stride of 200
    # steps for an hour that has 192.
    raw_stride = history_seconds / float(DT_SECONDS)
    if history_seconds <= 0 or abs(raw_stride - round(raw_stride)) > 1e-9:
        raise ValueError(
            f"--history-every-minutes {history_every_minutes} is not a whole "
            f"number of {DT_SECONDS:g} s steps"
        )
    stride = int(round(raw_stride))
    first = float(history_from_minutes) * 60 / DT_SECONDS
    if first < 0 or first > steps or abs(first - round(first)) > 1e-9:
        raise ValueError('--history-from-minutes must select a whole step within the forecast')
    if (steps - int(round(first))) % stride != 0:
        raise ValueError("the history cadence does not divide the forecast length")
    capture_steps = list(range(int(round(first)), steps + 1, stride))
    start = datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S")
    labels = {
        step: (start + timedelta(seconds=step * DT_SECONDS)).strftime(
            "%Y-%m-%d_%H.%M.%S"
        )
        for step in capture_steps
    }
    return {
        "start_time": start_text,
        "dt_seconds": DT_SECONDS,
        "forecast_hours": float(hours),
        "steps": steps,
        "history_every_minutes": float(history_every_minutes),
        "history_from_minutes": float(history_from_minutes),
        "history_stride_steps": stride,
        "capture_steps": capture_steps,
        "labels": labels,
        "valid_times": {
            step: (start + timedelta(seconds=step * DT_SECONDS)).strftime(
                "%Y-%m-%d_%H:%M:%S"
            )
            for step in capture_steps
        },
    }


# --------------------------------------------------------------------------
# the DA file contract and the DA-cycling restart (RRFS v2 stage H6, T1)
# --------------------------------------------------------------------------
CHECKPOINT_SCHEMA = "gpuwm-hex.forecast-checkpoint/v1"


def cycling_options(
    args: argparse.Namespace, schedule: Mapping[str, Any], init_path: Path
) -> dict[str, Any] | None:
    """The JEDI-contract and checkpoint/restart selections, resolved to steps."""

    checkpoint_steps: set[int] = set()
    for hours in getattr(args, "checkpoint_hours", ()) or ():
        step = float(hours) * 3600.0 / DT_SECONDS
        if abs(step - round(step)) > 1e-9 or int(round(step)) not in schedule["capture_steps"]:
            raise ValueError(
                f"--checkpoint-hours {hours} is not a history frame of this run; "
                "a checkpoint is written where a frame is captured so the "
                "restart and the frame describe the same committed state"
            )
        checkpoint_steps.add(int(round(step)))
    restart = getattr(args, "restart_from", None)
    if not (getattr(args, "jedi_background", False) or checkpoint_steps or restart):
        return None
    return {
        "jedi_background": bool(getattr(args, "jedi_background", False)),
        "soil_levels": int(getattr(args, "jedi_soil_levels", 9)),
        "checkpoint_steps": checkpoint_steps,
        "restart_from": None if restart is None else Path(restart),
        "da_analysis": None if args.da_analysis is None else Path(args.da_analysis),
        "init_path": Path(init_path),
    }


def _strip_regional(array: Any, n: int) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim >= 1 and data.shape[-1] == n + 1:
        data = data[..., :n]
    return np.ascontiguousarray(data)


def download_forecast_checkpoint(
    stack: Mapping[str, Any],
    *,
    step: int,
    label: str,
    valid_time: str,
    dynamics_tendencies: Any,
) -> dict[str, Any]:
    """The full hex restart state at a committed history boundary, on the host.

    The same four parts the proof harness's F030 checkpoint carries (dycore
    state, saved diagnostics, physics backend payload, the GF advective
    forcing carrier), at any history frame instead of one fixed step.  A
    limited-area run's garbage column/edge is stripped: the restore re-pads
    it exactly as a cold start does.
    """

    from woof.hex.cuda_dualrun import fingerprint_atmosphere
    from woof.hex.driver import DrySavedDiagnostics
    from woof.hex.state import PrognosticState

    driver = stack["driver"]
    backend = stack["backend"]
    cp = driver.cp
    cp.cuda.get_current_stream().synchronize()
    if float(driver.atmosphere.state.time_seconds) != step * DT_SECONDS:
        raise RuntimeError("checkpoint clock does not equal its step")
    device_state = driver.atmosphere.state.to_host()
    saved = driver.atmosphere.saved
    n_cells, n_edges = int(N_CELLS), int(N_EDGES)
    state = PrognosticState(
        rho=_strip_regional(device_state.rho, n_cells),
        rho_theta=_strip_regional(device_state.rho_theta, n_cells),
        rho_u=_strip_regional(device_state.rho_u, n_edges),
        rho_w=_strip_regional(device_state.rho_w, n_cells),
        scalars=_strip_regional(device_state.scalars, n_cells),
        time_seconds=float(device_state.time_seconds),
    )
    host_saved = DrySavedDiagnostics(
        theta_m=_strip_regional(cp.asnumpy(saved.theta_m), n_cells),
        exner=_strip_regional(cp.asnumpy(saved.exner), n_cells),
        density_perturbation=_strip_regional(cp.asnumpy(saved.density_perturbation), n_cells),
        rho_theta_perturbation=_strip_regional(cp.asnumpy(saved.rho_theta_perturbation), n_cells),
        pressure_perturbation=_strip_regional(cp.asnumpy(saved.pressure_perturbation), n_cells),
        normal_velocity=_strip_regional(cp.asnumpy(saved.normal_velocity), n_edges),
        vertical_velocity=_strip_regional(cp.asnumpy(saved.vertical_velocity), n_cells),
    )
    gf = None
    if dynamics_tendencies is not None:
        gf = {
            "rthdynten": np.ascontiguousarray(cp.asnumpy(dynamics_tendencies.rthdynten)),
            "rqvdynten": np.ascontiguousarray(cp.asnumpy(dynamics_tendencies.rqvdynten)),
            "time_seconds": float(dynamics_tendencies.time_seconds),
        }
    backend_state = backend.restart_state()
    return {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "label": label,
        "valid_time": valid_time,
        "dt_seconds": float(DT_SECONDS),
        "n_cells": n_cells,
        "n_edges": n_edges,
        "state": state,
        "saved_diagnostics": host_saved,
        "backend_state": backend_state,
        "stochastic": (None if getattr(driver, "stochastic", None) is None
                       else driver.stochastic.process.snapshot()),
        "gf_dynamics_tendencies": gf,
        "atmosphere_fingerprint": fingerprint_atmosphere(driver.atmosphere),
        "backend_fingerprint": proof.fingerprint_nested_arrays(backend_state),
    }


def _seam_leaf_fields(backend_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    seam = backend_state.get("seam", {}) if isinstance(backend_state, Mapping) else {}
    arrays = seam.get("arrays", {}) if isinstance(seam, Mapping) else {}
    leaves: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if str(key).startswith("fields/"):
            leaves[str(key).split("/", 1)[1]] = np.asarray(value)
    return leaves


def _init_fallback_fields(init_path: Path) -> dict[str, np.ndarray]:
    # Read through the engine's Rust NetCDF reader (rw_netcdf); the JEDI
    # contract's file traffic carries no Python NetCDF library.
    from woof.netcdf_bridge import open_dataset

    out: dict[str, np.ndarray] = {}
    dataset = open_dataset(Path(init_path))
    for name in ("snowc", "snowh", "xice", "vegfra"):
        if name in dataset.variables:
            variable = dataset.variables[name]
            variable.set_auto_mask(False)
            out[name] = np.asarray(variable[0], dtype=np.float32)
    if "lai12m" in dataset.variables:
        variable = dataset.variables["lai12m"]
        variable.set_auto_mask(False)
        out["lai12m"] = np.asarray(variable[:], dtype=np.float32).T
    return out


def write_jedi_background(
    *,
    output_root: Path,
    stack: Mapping[str, Any],
    host: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    label: str,
    valid_time: str,
    soil_levels: int,
    init_fallback: Mapping[str, np.ndarray],
    init_path: Path,
    backend_state: Mapping[str, Any] | None = None,
    effective_radii: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from woof.hex import jedi_contract

    driver = stack["driver"]
    cp = driver.cp
    if backend_state is None:
        backend_state = stack["backend"].restart_state()
    valid = datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S")
    fields, provenance = jedi_contract.assemble_background_fields(
        snapshot_arrays=snapshot["arrays"],
        pressure_perturbation=cp.asnumpy(driver.atmosphere.saved.pressure_perturbation),
        pressure_base=np.asarray(host["prepared"].reference.pressure_base),
        seam_fields={
            **_seam_leaf_fields(backend_state),
            **{
                str(name): np.asarray(cp.asnumpy(value) if hasattr(value, "device") else value)
                for name, value in dict(effective_radii or {}).items()
            },
        },
        init_fallback=init_fallback,
        scalar_names=host["scalar_names"],
        n_cells=int(N_CELLS),
        n_edges=int(N_EDGES),
        n_levels=int(N_LEVELS),
        soil_levels=int(soil_levels),
        valid=valid,
    )
    record = jedi_contract.write_background(
        output_root / f"mpasout.{label}.nc",
        fields=fields,
        provenance=provenance,
        valid=valid,
        n_cells=int(N_CELLS),
        n_edges=int(N_EDGES),
        n_levels=int(N_LEVELS),
        soil_levels=int(soil_levels),
        scalar_names=host["scalar_names"],
        n_vertices=int(np.asarray(proof._mesh_value(host["prepared"].mesh, "areaTriangle")).size),
        # OU1: the init's element identity rides in the background, so the
        # analysis JEDI writes back into it can be proven to be on this run's
        # cells and edges at the DA-cycling restart.
        identity_path=Path(init_path),
    )
    return {
        "path": record.path,
        "bytes": record.bytes,
        "sha256": record.sha256,
        "receipt": record.receipt_path,
    }


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------
def _physics_cadence_proof(
    decision: Mapping[str, Any] | None,
    step_receipts: Sequence[Mapping[str, Any]],
    executed_steps: int,
) -> dict[str, Any] | None:
    """Positive evidence the declared surface/PBL cadence is the one that ran.

    A/B rule: a held-cadence arm that reproduces the welded arm exactly is
    first evidence the hold never happened, and a receipt that only repeats
    the request cannot tell the two apart.  So the receipt carries the
    seam's own counters -- how many steps the phase-one call reported
    ``surface_pbl_ran``, the engine's surface-layer count at the last step --
    against the count the declared cadence predicts
    (:func:`woof.hex.pbl_cadence.calls_in_steps`), and says whether they
    agree.  Radiation's count rides beside it because the receipt names
    every physics call rate the run used.
    """

    if decision is None or executed_steps <= 0:
        return None
    from woof.hex import pbl_cadence

    due = 0
    radiation_due = 0
    counts: Mapping[str, Any] = {}
    for receipt in step_receipts:
        cadence = receipt.get("backend", {}).get("cadence", {})
        if cadence.get("surface_pbl_ran"):
            due += 1
        if cadence.get("radiation_ran"):
            radiation_due += 1
        counts = cadence.get("call_counts", counts)
    stepbl = int(decision["steps_between_calls"])
    expected = pbl_cadence.calls_in_steps(
        steps_between_calls=stepbl, executed_steps=int(executed_steps)
    )
    # The surface layer remains active when LES replaces YSU. Older seam
    # receipts carry only the YSU counter, so retain that fallback.
    counter = "sfclay" if "sfclay" in counts else "ysu"
    engine_calls = None if counter not in counts else int(counts[counter])
    consistent = due == expected and (engine_calls in (None, expected))
    return {
        "schema": "gpuwm-hex.physics-cadence-proof/v1",
        "surface_pbl_seconds": float(decision["surface_pbl_seconds"]),
        "steps_between_calls": stepbl,
        "held": bool(decision["held"]),
        "source": decision["source"],
        "executed_steps": int(executed_steps),
        "surface_pbl_calls_expected": expected,
        "surface_pbl_steps_reported_due": due,
        "surface_pbl_steps_held": int(executed_steps) - due,
        "engine_call_counts_at_last_step": dict(counts),
        "radiation_steps_reported_due": radiation_due,
        "radiation_seconds": float(decision["radiation_seconds"]),
        "consistent": bool(consistent),
        "note": (
            f"the seam reported the surface/PBL stack due on {due} of "
            f"{int(executed_steps)} steps and held on the other "
            f"{int(executed_steps) - due}; the declared cadence "
            f"({decision['label']}) predicts {expected} calls"
            + (
                "" if engine_calls is None
                else f"; the engine's own {counter} counter reads {engine_calls}"
            )
            + (".  They agree" if consistent else ".  THEY DISAGREE")
        ),
    }


def _mixing_treatment_proof(driver: Any, executed_steps: int) -> dict[str, Any]:
    """Positive evidence the mixing treatment ran (or provably did not).

    A/B rule: exact reproduction of a mixing-off outcome without this proof
    is first evidence the treatment never ran.  Expected RK1 mixing calls =
    3 dynamics subcycles per executed step.
    """

    les = getattr(driver, "les_dissipation", None)
    if les is not None:
        return {"active": True, "lane": "resident_cuda_les_v842",
                "closure": driver.config.config_les_model,
                "surface": "WOOF surface-layer device fluxes",
                "pbl": "off", "calls": dict(les.counts),
                "scalar_order": list(les.names),
                "native_pools": False}
    cfg = getattr(driver, "mixing_config_v841", None)
    calls = int(getattr(driver, "mixing_calls_v841", 0))
    expected = 3 * int(executed_steps)
    active = cfg is not None
    proof_block: dict[str, Any] = {
        "active": active,
        "lane": "v841_2d_smagorinsky" if active else "off",
        "rk1_mixing_calls": calls,
        "expected_rk1_mixing_calls": expected if active else 0,
        "calls_match_expected": (
            calls == expected if active else calls == 0
        ),
        "deformation_weights": getattr(
            driver, "deformation_weights_receipt_v841", None
        ),
    }
    if active:
        proof_block["config"] = {
            "config_horiz_mixing": cfg.config_horiz_mixing,
            "config_len_disp": float(cfg.config_len_disp),
            "config_visc4_2dsmag": float(cfg.config_visc4_2dsmag),
            "config_smagorinsky_coef": float(cfg.config_smagorinsky_coef),
            "config_del4u_div_factor": float(cfg.config_del4u_div_factor),
            "config_h_ScaleWithMesh": bool(cfg.config_h_ScaleWithMesh),
        }
        proof_block["note"] = (
            "numbers under this configuration are a new sub-series; not "
            "bit-comparable to any mixing-off arm"
        )
    else:
        proof_block["note"] = (
            "REPORTED AS THE PRE-MIXING CONTROL CONFIGURATION: native "
            "itself dies in this regime on convective cases (reference-node "
            "control, case B, step 466)"
        )
    return proof_block


from woof.hex.nest_triggers import NEST_TRIGGERS_ENV  # noqa: E402  (one name for both physics routes)


def _arm_nest_trigger(*, host: Mapping[str, Any], schedule: Mapping[str, Any], output_root: Path,
                      backend: Any, cp: Any, start_step: int):
    """The armed ``[hex_nests]`` look, or None when the run names no table.

    The table comes from the forecast door's ``--nest-triggers FILE``, carried
    to this loop as ``WOOF_HEX_NEST_TRIGGERS`` (the door sets it for its own
    driver call only).  The fields a look can read on this route are the ones
    the physics backend forms on the card; a table naming another is refused
    here, before the first step, by :class:`woof.hex.nest_triggers.LoopTrigger`.
    The arming itself is :func:`woof.hex.nest_triggers.arm_loop`, shared with
    the MPAS-physics route.  Beside ``refl10cm`` (when the backend forms it)
    the route forms the dycore's own sources (``w``, ``cape``,
    ``updraft_helicity_max``: :mod:`woof.hex.nest_look_dycore`); until
    2026-10-29 it named ``refl10cm`` only, so a ``wmax``/``mucape``/``uh25``
    test was refused after its box was rented (owner canary
    run_9948508bd7cfdee3).  The trigger carries them as ``dycore_sources``.
    """

    table_path = host.get("nest_triggers") or os.environ.get(NEST_TRIGGERS_ENV)
    if not table_path:
        return None
    from woof.hex import nest_look_dycore, nest_triggers

    mesh = host["prepared"].mesh
    trigger = nest_triggers.arm_loop(
        table_path, run_dir=output_root, start_text=str(schedule["start_time"]), start_step=start_step,
        dt_s=float(DT_SECONDS),
        # The parent's end is always known: the schedule's own step count (a
        # schedule without forecast_hours skipped the "nest outlives the
        # parent" check silently before 2026-10-24).
        parent_seconds=(float(schedule["forecast_hours"]) * 3600.0
                        if schedule.get("forecast_hours") is not None
                        else float(schedule["steps"]) * float(DT_SECONDS)),
        mesh=mesh, owned_cells=int(mesh.n_cells),
        available_sources=(["refl10cm"] if backend.computes_reflectivity() else []) + list(nest_look_dycore.SOURCES),
        cp=cp)
    names = list(host.get("scalar_names") or ())
    trigger.dycore_sources = nest_look_dycore.for_trigger(
        trigger, cp, owned_cells=int(mesh.n_cells), qv_index=names.index("qv") if "qv" in names else None, mesh=mesh)
    return trigger


def execute_forecast(
    *,
    host: Mapping[str, Any],
    schedule: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
    arwen_checkout: Path,
    source_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    fingerprint_every: int,
    stop_on_refusal: bool = False,
    grid_path: Path | None = None,
    park_physics_tier: bool = False,
    required_free_bytes: int | None = None,
    local_timestep_classing_path: Path | None = None,
    fullstep_lts_options: Mapping[str, Any] | None = None,
    cycling: Mapping[str, Any] | None = None,
    iau: Mapping[str, Any] | None = None,
    balance_series: bool = False,
    on_history_ready: Callable[[dict[str, Any]], Any] | None = None,
    history_product_session: Callable[[], Any] | None = None,
    history_product_max_pending: int = 2,
    required_product_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the native forecast; optionally notify products of closed frames.

    Notifications carry only native file metadata and run on a separate
    worker. Consumer failures surface at the final callback drain.
    """
    if on_history_ready is not None and history_product_session is not None:
        raise ValueError("supply one history callback or one product session factory")
    if history_product_max_pending < 1:
        raise ValueError("history_product_max_pending must be positive")
    if required_product_ids and on_history_ready is None and history_product_session is None:
        raise ValueError("required products need a history consumer")
    join_prepared_fingerprint(host)
    if (fullstep_lts_options is not None
            and fullstep_lts_options.get("fullstep_lts_mode") == "active"
            and int(fullstep_lts_options.get("fullstep_lts_max_rate", 8)) != 1):
        if on_history_ready is not None or history_product_session is not None:
            raise NotImplementedError("history-ready callbacks are supported by the native executor; "
                                      "the active class executor owns a different history path")
        # Active full-step local time stepping: every rate class on its own
        # whole step.  The class executor hands a mesh that classes entirely
        # at rate one back to this function with the option cleared, so a
        # uniform mesh runs the unchanged path below.
        import sys

        from woof.hex.drivers import fullstep_lts_forecast_v841 as fullstep_forecast

        return fullstep_forecast.execute_fullstep_lts(
            sys.modules[__name__], native_execute=execute_forecast, policy=dict(fullstep_lts_options),
            host=host, schedule=schedule, cache_root=cache_root, output_root=output_root,
            arwen_checkout=arwen_checkout, source_receipt=source_receipt,
            authority_receipt=authority_receipt, fingerprint_every=fingerprint_every,
            stop_on_refusal=stop_on_refusal, grid_path=grid_path, park_physics_tier=park_physics_tier,
            required_free_bytes=required_free_bytes,
            local_timestep_classing_path=local_timestep_classing_path, cycling=cycling,
            iau=iau, balance_series=balance_series)
    from woof.hex.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    arwen_pin = dict(_pin_for_run(host, arwen_checkout, pin_arwen_physics_v841))
    # This must precede KernelCache's woof platform-binding construction.
    from woof.hex.cuda_backend import KernelCache, require_cuda

    # Any architecture NVRTC can compile the kernels for runs; this call no
    # longer pins sm_120 (``required_compute``), because that pin named no
    # breakage.  The receipt records whether the card holds an anchor
    # (``architecture`` below, woof.hex.cuda_backend.arch_admission).
    capability = require_cuda(min_compute=(12, 0), cache_dir=cache_root)
    import cupy as cp

    # ``required_free_bytes`` is the forecast door's own admission sum,
    # forwarded so this floor and the door's verdict are one number enforced
    # twice: without it, a card admitted at the door on its OWN measured row
    # would be refused here on the default model's larger fixed term, after
    # the mesh bind and the kernel compile were already paid for.  Absent,
    # the mesh-bound floor from the same admission surface applies.
    if required_free_bytes is None:
        from woof.hex.forecast_door import stochastic_array_bytes
        from woof.hex.stochastic import StochasticConfig
        stochastic_options = host.get("stochastic")
        # Price stochastic arrays only for a run that enables them; a run
        # without stochastic physics reads no prepared mesh here.
        stochastic_bytes = (stochastic_array_bytes(
            int(host["prepared"].mesh.n_cells), stochastic_options,
            levels=int(host["prepared"].vertical.n_vert_levels))
            if StochasticConfig.parse(stochastic_options).enabled else 0)
        memory = proof.gpu_memory_admission(cp, minimum=int(proof.MIN_FREE_DEVICE_BYTES) +
            stochastic_bytes)
    else:
        memory = proof.gpu_memory_admission(cp, minimum=int(required_free_bytes))
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    # RRFS v2 stage H6: a resumed run (plain restart or DA-cycling restart).
    stochastic_restart = None
    start_step = 0
    restart_kwargs: dict[str, Any] = {}
    restart_receipt: dict[str, Any] | None = None
    if cycling is not None and cycling.get("restart_from") is not None:
        import pickle

        with Path(cycling["restart_from"]).open("rb") as stream:
            checkpoint = pickle.load(stream)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("--restart-from is not a checkpoint this driver wrote")
        if float(checkpoint["dt_seconds"]) != float(DT_SECONDS) or int(checkpoint["n_cells"]) != int(N_CELLS):
            raise ValueError("checkpoint was written on a different mesh or timestep")
        start_step = int(checkpoint["step"])
        if start_step not in schedule["capture_steps"] or start_step >= int(schedule["steps"]):
            raise ValueError(
                "the checkpoint step must be a history frame of this run and "
                "before its end; --start-time/--hours name the ORIGINAL run"
            )
        if schedule["valid_times"][start_step] != checkpoint["valid_time"]:
            raise ValueError("checkpoint valid time differs from this schedule's")
        state = checkpoint["state"]
        saved = checkpoint["saved_diagnostics"]
        restart_receipt = {
            "checkpoint": str(cycling["restart_from"]),
            "checkpoint_sha256": proof.sha256_file(Path(cycling["restart_from"])),
            "step": start_step,
            "valid_time": checkpoint["valid_time"],
            "da_cycling": None,
        }
        if cycling.get("da_analysis") is not None:
            from woof.hex import jedi_contract

            prepared = host["prepared"]
            state, saved, da_receipt = jedi_contract.recouple_da_cycling(
                Path(cycling["da_analysis"]),
                init_path=Path(cycling["init_path"]),
                mesh=prepared.mesh,
                vertical=prepared.vertical,
                terrain_metrics=prepared.terrain_metrics,
                expected_reference=prepared.reference,
                # OU1: the analysis record must be the checkpoint's valid time.
                expected_valid_time=str(checkpoint["valid_time"]),
                scalar_names=host["scalar_names"],
                n_cells=int(N_CELLS),
                n_edges=int(N_EDGES),
                n_levels=int(N_LEVELS),
                allow_regional_sentinels=host.get("regional") is not None,
                time_seconds=float(state.time_seconds),
                work_dir=output_root,
            )
            restart_receipt["da_cycling"] = da_receipt
        restart_kwargs = {
            "state": state,
            "saved_diagnostics": saved,
            "backend_restart": checkpoint["backend_state"],
        }
        # The GF advective-forcing carrier is re-seeded below, not through
        # the proof's stack constructor: that constructor validates it at the
        # GLOBAL column count, while a limited-area driver forms it on its
        # padded extent (nCells + native's garbage column).  MEASURED
        # 2026-10-05: "rthdynten must be FP32 [level,cell]" for (59, 157860).
        restart_gf = checkpoint["gf_dynamics_tendencies"]
        if host.get("regional") is not None:
            host = dict(host)
            host["regional"] = dict(host["regional"])
            host["regional"]["start_time"] = datetime.strptime(
                checkpoint["valid_time"], "%Y-%m-%d_%H:%M:%S"
            )
        from woof.hex.iau_native import IAU_WINDOW_KEY, resolve_restart_window

        iau = resolve_restart_window(iau, carried=checkpoint.get(IAU_WINDOW_KEY),
                                     restart_seconds=float(start_step) * float(DT_SECONDS))
        stochastic_restart = checkpoint.get("stochastic")
        del checkpoint
    else:
        restart_gf = None
    time_ledger.emit("prep.device_upload", "begin")
    stack = proof._construct_device_stack(
        host=host,
        cache=cache,
        arwen_checkout=arwen_checkout,
        **restart_kwargs,
        backend_builder=_backend_builder(host, arwen_checkout),
        # The physics rollback is armed only when this run keeps a refused
        # step's boundary: --stop-on-refusal writes the last committed frame
        # from it.  Without it a refused step ends the run and the restored
        # seam would never be read, so the 447 MB boundary export (about
        # 24 ms of host time every step on the 43,884-cell point mesh) is not
        # taken.  No output byte depends on it.
        physics_rollback_snapshot=bool(stop_on_refusal),
        stochastic_restart=stochastic_restart,
    )
    from woof.hex.chem_transport import attach as attach_chemistry
    attach_chemistry(stack["backend"], host, stack["driver"])
    if host.get("wind_farms") not in (None, "", "off"):
        from woof.hex.wind_farm import attach_arwen
        attach_arwen(stack["backend"], host, stack["driver"], output_root=output_root)
    from woof.hex.arwen_surface_update import attach_host
    surface_receipt = attach_host(stack["backend"], host, schedule, output_root, dt=float(DT_SECONDS))
    if surface_receipt is not None:
        print(json.dumps({"surface_update": surface_receipt}), flush=True)
    time_ledger.emit("prep.device_upload", "end", attrs={"scope": "device_stack_including_kernel_load"})
    from woof.hex.cuda_column_order_v841 import attach_column_order
    column_order_receipt = attach_column_order(stack["backend"], host["prepared"].mesh)
    if column_order_receipt is not None:
        time_ledger.emit("forecast.construct", "mark", attrs={"column_order":column_order_receipt})
    # Opt-in local time stepping.  Returns None -- and rebinds nothing -- when
    # config_local_timestep is off, which is the default.
    from woof.hex.cuda_driver_lts import attach_local_timestep

    explicit_classing = None
    if local_timestep_classing_path is not None:
        # INSTRUMENT, not the shipped classing: per-cell rates from a file,
        # connectivity and the driven-zone mask from the grid file.  The
        # receipt records rate_source="explicit" so the arm cannot be
        # mistaken for the option's own behaviour.
        from woof.hex.lts_v841 import classify_from_grid_file_with_rates

        config = stack["driver"].config
        explicit_classing = classify_from_grid_file_with_rates(
            str(grid_path),
            str(local_timestep_classing_path),
            rates=tuple(int(v) for v in config.config_local_timestep_rates),
            buffer_rings=int(config.config_local_timestep_buffer_rings),
        )
    lts_attachment = attach_local_timestep(
        stack["driver"],
        grid_path=str(grid_path) if grid_path else None,
        classing=explicit_classing,
    )
    stack["local_timestep"] = lts_attachment
    fullstep_shadow = None
    if fullstep_lts_options is not None:
        from woof.hex.config_fullstep_lts import FullStepLtsConfig
        from woof.hex.fullstep_lts_dispatch import attach_fullstep_lts
        def fullstep_sink(row):
            with (output_root / "fullstep-lts-stages.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        from netCDF4 import Dataset
        if grid_path is None:
            raise ValueError("full-step shadow needs the source grid's global cell IDs")
        with Dataset(grid_path) as dataset:
            fullstep_ids = np.asarray(dataset["indexToCellID"][:], dtype=np.int64)
        fullstep_shadow = attach_fullstep_lts(stack["driver"], FullStepLtsConfig(**fullstep_lts_options), sink=fullstep_sink, global_cell_ids=fullstep_ids)
        if fullstep_shadow is not None:
            state = stack["driver"].atmosphere.state
            fullstep_shadow.observe(state.rho, state.rho_u, state.rho_w,
                stack["driver"].atmosphere.vertical.rdzw, stage=0,
                stage_dt=float(stack["driver"].config.config_dt), time_seconds=float(state.time_seconds))
    if restart_gf is not None:
        # Restore side of the step-16 restart divergence (see the proof
        # harness): the first resumed step must consume the forcing pair the
        # checkpoint step's dynamics formed.  Validated at the extent the
        # running driver forms it on.
        from woof.hex.cuda_driver import CudaV841GfDynamicsTendencies

        carrier = CudaV841GfDynamicsTendencies(
            rthdynten=cp.asarray(np.ascontiguousarray(restart_gf["rthdynten"], dtype=np.float32)),
            rqvdynten=cp.asarray(np.ascontiguousarray(restart_gf["rqvdynten"], dtype=np.float32)),
            time_seconds=float(restart_gf["time_seconds"]),
        )
        width = int(np.asarray(restart_gf["rthdynten"]).shape[-1])
        if width not in (int(N_CELLS), int(N_CELLS) + 1):
            raise ValueError(f"restored GF forcing has {width} columns for a {N_CELLS}-cell mesh")
        carrier.validate(cp=cp, n_vert_levels=int(N_LEVELS), n_cells=width)
        if float(carrier.time_seconds) != float(stack["driver"].atmosphere.state.time_seconds):
            raise RuntimeError("restored GF forcing is not stamped with the restored model time")
        stack["gf_dynamics_tendencies"] = carrier

    from woof.hex.cuda_subcycle_graphs import graph_policy, graphs_enabled
    graph_requested = graphs_enabled(int(N_CELLS))
    graph_controller = None
    if graph_requested and lts_attachment is None and fullstep_shadow is None:
        from woof.hex.cuda_subcycle_graphs import attach_subcycle_graphs

        graph_controller = attach_subcycle_graphs(stack["driver"])
    graph_full_steps = graph_partial_steps = 0

    physics_park = None
    if park_physics_tier:
        from woof.hex.cuda_physics_tier_park_v841 import CudaPhysicsTierParkV841

        physics_park = CudaPhysicsTierParkV841(
            cp, stack["backend"]._seam, diagnose=True
        )

    from woof.hex.forecast_start import RadarSnapshots
    radar = RadarSnapshots(output_root, schedule, DT_SECONDS)
    capture_steps = {step for step in schedule["capture_steps"] if step >= start_step}
    if start_step > 0:
        # The checkpoint frame itself was captured by the run that wrote it.
        capture_steps.discard(start_step)
    install_step_gate_batch(stack, cp)
    labels = schedule["labels"]
    steps = int(schedule["steps"])
    # Incremental analysis update on this executor (lane W1): one class, every cell and
    # edge, the same hook the full-step class executor uses.  None leaves the step loop
    # exactly as before.
    iau_forcing = None
    iau_receipt = None
    balance = None
    if iau is not None:
        from woof.hex.iau_native import build_iau_forcing, identity_layout

        local_cells = int(stack["driver"].atmosphere.state.rho.shape[1])
        pad = local_cells - int(N_CELLS)
        if pad not in (0, 1):
            raise ValueError(f"the driver carries {local_cells} cell columns for a {N_CELLS}-cell mesh")
        layout = identity_layout(int(N_CELLS) + pad, int(N_EDGES))
        if iau.get("route") == "native":
            layout = identity_layout(int(N_CELLS), int(N_EDGES))
        iau_forcing, iau_receipt = build_iau_forcing(
            iau, prepared=host["prepared"], scalar_names=host["scalar_names"], layouts=[layout], cp=cp,
            work_dir=cache_root, n_cells=int(N_CELLS), n_edges=int(N_EDGES), n_levels=int(N_LEVELS),
            pad_cells=pad)
        print(json.dumps({"iau": iau_receipt}, default=str), flush=True)
    if balance_series:
        from types import SimpleNamespace as _Namespace

        from woof.hex.iau_forcing import BalanceSeries

        balance = BalanceSeries.build(output_root / "balance-series.jsonl", host["scalar_names"],
                                      np.asarray(host["prepared"].vertical.dzw, dtype=np.float64), cp)
        balance_view = _Namespace(classes=[_Namespace(driver=stack["driver"], n_owned_cells=int(N_CELLS),
                                                      tables=_Namespace(rate=1))])
        balance.sample(balance_view, float(start_step) * float(DT_SECONDS), cp)
    static = proof._static_output_fields(host)
    jedi_files: dict[str, Any] = {}
    checkpoint_files: dict[str, Any] = {}
    init_fallback_cache: dict[str, Any] = {}

    fingerprint_path = output_root / "boundary-fingerprints.jsonl"
    fingerprints = (
        BoundaryFingerprintWriter(fingerprint_path) if fingerprint_every > 0 else None
    )
    snapshot_projection: dict[str, dict[str, str]] = {}
    snapshot_q2: dict[str, str | None] = {}
    snapshot_receipts: dict[str, Any] = {}
    snapshot_files: dict[str, Any] = {}
    physical_gates: dict[str, Any] = {}
    health: list[dict[str, Any]] = []
    step_receipts: list[dict[str, Any]] = []

    capture_seconds = 0.0
    write_seconds = 0.0
    write_wait_seconds = 0.0
    current_gf: dict[str, Any] = {"value": None}
    fingerprint_seconds = 0.0
    health_seconds = 0.0
    integration_seconds = 0.0
    first_step_seconds = None

    # Queue immutable D2H copies on the producer stream before a successor
    # can mutate the fields. A worker waits for their event, audits the host
    # arrays and submits the same NetCDF writer to its existing process.
    # Cycling and refusal recovery retain their synchronous state boundary.
    # WOOF_HEX_HISTORY_BACKGROUND=off also retains direct capture/writing.
    history = history_writer.open_writer(proof.write_snapshot_netcdf)
    product_ready = None
    if on_history_ready is not None or history_product_session is not None:
        from woof.hex.history_ready import HistoryReady
        product_ready = HistoryReady(on_history_ready, session_factory=history_product_session,
                                     max_pending=history_product_max_pending,
                                     required_product_ids=required_product_ids)
    # Nests that spawn on conditions ([hex_nests], woof.hex.nest_triggers): armed
    # only when the door names a table, so an unarmed run's loop is unchanged.
    nest_trigger = _arm_nest_trigger(host=host, schedule=schedule, output_root=output_root,
                                     backend=stack["backend"], cp=cp, start_step=start_step)

    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from woof.hex.host_parallel import threads as history_hash_threads
    async_capture = history is not None and cycling is None and not stop_on_refusal
    capture_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hex-capture") if async_capture else None
    capture_pending = []
    wind_frames = {}
    capture_finish_seconds = 0.0
    # F000's surface-layer diagnostics come from the first physics call
    # (woof.hex.f000_first_call): the frame is captured before step 1 and
    # finished after it.  A cycling run keeps the old frame: its step-0
    # finish reads the live restart state, which step 1 would have moved.
    # f000 only: a run resumed mid-forecast (start_step > 0) captures no f000 and
    # its frames stay as they are.  A cycling run's live-state work at step 0
    # (finish_cycling) still runs at the boundary.
    first_call = (f000_first_call.FirstCallHandoff()
                  if (0 in capture_steps and start_step == 0 and steps > 0)
                  else None)
    held_f000: list[Any] = []

    held_background: list[Any] = []

    def finish_cycling(step, snapshot, defer_background=False):
        # Reads the LIVE state at this step's boundary (restart state, checkpoint,
        # JEDI background), so it runs at the boundary even when the frame's
        # publication waits for the first physics call.
        if cycling is None:
            return
        backend_state = None
        if cycling["jedi_background"] or step in cycling["checkpoint_steps"]:
            backend_state = stack["backend"].restart_state()
        if cycling["jedi_background"]:
            if "fields" not in init_fallback_cache:
                init_fallback_cache["fields"] = _init_fallback_fields(
                    Path(cycling["init_path"])
                )
            background = dict(
                output_root=output_root,
                stack=stack,
                host=host,
                snapshot=snapshot,
                label=labels[step],
                valid_time=schedule["valid_times"][step],
                soil_levels=int(cycling["soil_levels"]),
                init_fallback=init_fallback_cache["fields"],
                init_path=Path(cycling["init_path"]),
                backend_state=backend_state,
                effective_radii=current_gf.get("radii"),
            )
            if defer_background:
                # The f000 background's surface fields come from the completed
                # frame (woof.hex.f000_first_call), written after step 1.  Its
                # live inputs are read now: the restart state is a host export
                # and the pressure perturbation is copied to the host here.
                from types import SimpleNamespace
                background["stack"] = {"driver": SimpleNamespace(cp=cp, atmosphere=SimpleNamespace(
                    saved=SimpleNamespace(pressure_perturbation=cp.asnumpy(
                        stack["driver"].atmosphere.saved.pressure_perturbation))))}
                held_background.append(background)
            else:
                jedi_files[str(step)] = write_jedi_background(**background)
        if step in cycling["checkpoint_steps"]:
            payload = download_forecast_checkpoint(
                stack,
                step=step,
                label=labels[step],
                valid_time=schedule["valid_times"][step],
                dynamics_tendencies=current_gf["value"],
            )
            if iau is not None:
                from woof.hex.iau_native import IAU_WINDOW_KEY, window_record

                payload[IAU_WINDOW_KEY] = window_record(iau)
            target = output_root / f"checkpoint.{labels[step]}.pkl"
            from woof.hex.checkpoint_io import write_checkpoint
            write_checkpoint(target, payload)
            checkpoint_files[str(step)] = {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": proof.sha256_file(target),
                "atmosphere_fingerprint_sha256": proof.canonical_json_sha256(
                    json.loads(json.dumps(payload["atmosphere_fingerprint"], sort_keys=True, default=str))
                ),
            }
            del payload
        del backend_state

    def finish_capture(step, snapshot, cycling_done=False):
        nonlocal write_seconds, capture_finish_seconds
        finish_started = time.perf_counter()
        if step in wind_frames:
            snapshot['wind_fields'], snapshot['wind_metadata'] = wind_frames.pop(step)
        time_ledger.emit("forecast.output.capture.finish", "begin", attrs={"step": step})
        if step == 0:
            if "f000_completed_surface_diagnostics" not in snapshot["receipt"]:
                snapshot["receipt"]["f000_completed_surface_diagnostics"] = (
                    complete_f000_surface_diagnostics(snapshot, host["f000_start_humidity"])
                )
            if first_call is not None:
                snapshot["receipt"]["f000_first_call_diagnostics"] = (
                    complete_f000_from_first_call(snapshot, first_call.take()))
        physical_gates[str(step)] = proof.physical_snapshot_gate(
            snapshot,
            allow_initial_negative_qv=(step == 0),
            scalar_names=host["scalar_names"],
            surface_accumulators=host["surface_accumulators"],
            surface_accumulator_declarations=host.get("surface_accumulator_declarations"),
        )
        if step == steps and hasattr(stack["backend"], "write_ledger"):
            # A seeded backend dumps its conservation ledger beside the
            # run's other receipts at the final frame (CSV + JSON).
            stack["backend"].write_ledger(output_root)
        snapshot_projection[str(step)] = proof._snapshot_hash_projection(snapshot)
        snapshot_q2[str(step)] = _snapshot_q2_hash(snapshot)
        snapshot_receipts[str(step)] = snapshot["receipt"]
        mark = time.perf_counter()
        destination = output_root / f"cuda-history.{labels[step]}.nc"
        snapshot["valid_time"] = schedule["valid_times"][step]
        radar.note_history(step)
        if history is None:
            time_ledger.emit("forecast.output.history", "begin", attrs={"step": step})
            snapshot_files[str(step)] = proof.write_snapshot_netcdf(
                destination, snapshot, static
            )
            time_ledger.emit("forecast.output.history", "end", attrs={"step": step})
        else:
            # The writer reads only the arrays.  An array the capture does
            # not solely own (a view, or a host buffer something else holds
            # and the next step could write into) is copied at hand-off.
            snapshot_files[str(step)] = history.submit(
                destination,
                # the writer reads the arrays, plus the step for the f000
                # refl10cm provenance attribute
                {"arrays": snapshot["arrays"] if async_capture else history_writer.private_arrays(snapshot["arrays"]),
                 "receipt": {"step": step},
                 **({"wind_fields": snapshot["wind_fields"], "wind_metadata": snapshot["wind_metadata"]}
                    if "wind_fields" in snapshot else {}),
                 "valid_time": snapshot["valid_time"]},
                static,
                defer_fallback=async_capture,
            )
        if product_ready is not None:
            context = {"step": step, "label": labels[step],
                       "valid_time": schedule["valid_times"][step]}
            if history is None:
                product_ready.submit(snapshot_files[str(step)], **context)
            else:
                product_ready.watch(snapshot_files[str(step)], **context)
        if cycling is not None and not cycling_done:
            finish_cycling(step, snapshot)
        while held_background:
            # the f000 JEDI background, from the completed frame
            jedi_files[str(step)] = write_jedi_background(**held_background.pop())
        write_seconds += time.perf_counter() - mark
        del snapshot
        gc.collect()
        capture_finish_seconds += time.perf_counter() - finish_started
        time_ledger.emit("forecast.output.capture.finish", "end", attrs={"step": step})

    def capture(step: int) -> None:
        nonlocal capture_seconds
        time_ledger.emit("forecast.output.capture", "begin", attrs={"step": step})
        mark = time.perf_counter()
        if capture_pending:
            previous_step = capture_pending.pop(0).result()
            # A failed writer must not move NetCDF into the capture worker.
            # Recover here before another frame can retain its host arrays;
            # healthy writes continue without a producer-side wait.
            recovered = history.recover_if_needed(snapshot_files[str(previous_step)])
            if recovered is not None and product_ready is not None:
                product_ready.submit(recovered, step=previous_step, label=labels[previous_step],
                                     valid_time=schedule["valid_times"][previous_step])
        snapshot = proof.capture_snapshot(
            label=labels[step],
            step=step,
            driver=stack["driver"],
            backend=stack["backend"],
            prep_geometry=stack["prep_geometry"],
            kernel_cache=stack["driver"].cache,
            f000_surface_diagnostics=stack["f000_surface_diagnostics"],
            expect_refl10cm=True,
            deferred=async_capture,
            solve_cells=stack.get("solve_cells"),
            scalar_names=host["scalar_names"],
            surface_accumulators=host["surface_accumulators"],
        )
        capture_seconds += time.perf_counter() - mark
        wind = getattr(stack['backend'], 'wind_farms', None)
        if wind is not None:
            wind_frames[step] = (wind.history_fields(), wind.metadata())
        if async_capture:
            def complete():
                time_ledger.emit("forecast.output.capture.materialize", "begin", attrs={"step": step})
                try:
                    completed = snapshot.result()
                finally:
                    time_ledger.emit("forecast.output.capture.materialize", "end", attrs={"step": step})
                finish_capture(step, completed)
                return step
            try:
                capture_pending.append(capture_pool.submit(copy_context().run, complete))
            except BaseException:
                # If hand-off fails, nobody else owns the queued copies.
                snapshot.transfer.wait()
                raise
        elif step == 0 and first_call is not None:
            # Host arrays already; finished once the first call has run.  The
            # start-file q2/psfc completion needs no physics call, so a cycling
            # run's boundary work (JEDI background) still sees it.
            snapshot["receipt"]["f000_completed_surface_diagnostics"] = (
                complete_f000_surface_diagnostics(snapshot, host["f000_start_humidity"]))
            finish_cycling(0, snapshot, defer_background=True)
            held_f000.append(snapshot)
        else:
            finish_capture(step, snapshot)
        time_ledger.emit("forecast.output.capture", "end", attrs={"step": step})

    # ORDER MATTERS.  ``proof._run_steps`` reads the previous surface updates
    # BEFORE capturing its start-step snapshot; capture allocates device
    # memory, so keeping this order keeps the allocation history aligned with
    # the proof arm the fork-equivalence gate compares against.
    previous = proof._previous_surface_updates(
        stack, host["surface_accumulators"]
    )
    if 0 in capture_steps:
        capture(0)
    if fingerprints is not None:
        mark = time.perf_counter()
        fingerprints.write(0, proof.fingerprint_execution_boundary(stack))
        fingerprint_seconds += time.perf_counter() - mark

    refusal: dict[str, Any] | None = None
    executed_steps = steps
    loop_started = time.perf_counter()
    # GF advective forcing carried step to step; None at step 1 is native's
    # own start state (tend_physics is zero before dynamics first forms it).
    # A resumed run starts from the carrier its checkpoint restored.
    gf_dynamics_tendencies = stack.get("gf_dynamics_tendencies") if start_step else None
    current_gf["value"] = gf_dynamics_tendencies
    last_step_clock = None
    from woof.hex.cuda_seam_snapshot_v841 import TwoDeepForecastSnapshotV841
    from woof.hex.step_receipt_writer import StepReceiptWriter
    from types import SimpleNamespace

    # The two committed boundaries are kept only for a run that reads them.
    # THE BREAKAGE THIS PREVENTS (measured 2026-10-11, RTX PRO 6000 host, one 96 GB card,
    # 30 km x1.655362, bare defaults, per-allocation hook): both slots held
    # 21,540 MiB of copies -- the seam export twice (14,925 MiB) and the MPAS
    # state, saved state and adapter carriers twice -- and the run stopped
    # at its second step with "Out of memory allocating 865,078,272 bytes"
    # inside the scalar limiter, after a door that predicted 66,086 MiB.
    # Nothing read them: without --stop-on-refusal a refused step ends the
    # run (read_pending_health raises), exactly as the per-step physics
    # boundary export is already not taken without it
    # (physics_rollback_snapshot above).
    rollback = (
        TwoDeepForecastSnapshotV841(stack, cp, observers={
            "graphs": graph_controller, "physics_park": physics_park})
        if stop_on_refusal
        else UnarmedForecastSnapshotV841()
    )
    # Allocate both slots before launching the forecast. Never fall back to
    # host snapshots or one slot when the card cannot protect a late refusal.
    rollback.capture(start_step - 1, previous, gf_dynamics_tendencies)
    rollback.capture(start_step, previous, gf_dynamics_tendencies)
    receipt_writer = StepReceiptWriter(output_root / "step-receipts.jsonl",
                                       document=output_root / "step-receipts.json",
                                       schema=SCHEMA + "/step-receipts")
    pending_health = None
    late_health_error = None
    snapshot_seconds = 0.0
    card_attrs = {"card": time_ledger.card_index(), "card_count": 1,
                  "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "mutex_cards": os.environ.get("GPU_MUTEX_CARDS")}

    def inspect_pending_health() -> None:
        nonlocal pending_health, health_seconds, late_health_error
        if pending_health is None:
            return
        checked_step, measured, atmosphere = pending_health
        pending_health = None
        started = time.perf_counter()
        checked_stack = {**stack, "driver": SimpleNamespace(
            atmosphere=atmosphere, cache=stack["driver"].cache)}
        try:
            health.append(step_health_gate(
                checked_stack, checked_step, cp,
                trace_hot_cell=stack.get("solve_cells") is not None,
                measured_health=measured))
        except FloatingPointError as error:
            # Inspection can happen while the successor is still staged.
            # Finish its transaction before restoring the committed slot.
            late_health_error = (checked_step, error)
        finally:
            health_seconds += time.perf_counter() - started

    def read_pending_health() -> bool:
        nonlocal refusal, executed_steps, previous, gf_dynamics_tendencies
        inspect_pending_health()
        if late_health_error is None:
            return True
        checked_step, error = late_health_error
        if not stop_on_refusal:
            # The run ends here; no boundary was retained to restore.
            raise error
        # The serial gate keeps the refused committed endpoint. Remove only
        # a successor, even if its own transaction already rolled back.
        if step > checked_step:
            previous, gf_dynamics_tendencies = rollback.restore(checked_step)
            time_ledger.emit("forecast.integrate", "mark", attrs={
                **card_attrs, "event": "rollback", "step": checked_step, "discarded_step": checked_step + 1,
                "reason": str(error)})
        refusal = {
            "refused": True, "refused_in": "step health gate",
            "step": checked_step, "model_seconds": checked_step * DT_SECONDS,
            **refusal_exception_details(error), "last_committed_step": checked_step,
            "note": (
                "the composite step committed but its state fails the "
                "health gate; the forecast is truncated here and the "
                "per-step health envelopes up to the previous step are "
                "retained in this receipt"),
        }
        executed_steps = checked_step
        return False

    # This hook runs after the successor's final validation kernels have
    # been enqueued and before waiting on its publication event. Inspecting
    # the previous step here overlaps device work instead of leaving the
    # card idle after execute_composite_step returns.
    endpoint_reader = stack["driver"].final_validation_reader

    def read_current_endpoint(atmosphere: Any, flag: Any) -> int:
        inspect_pending_health()
        return endpoint_reader(atmosphere, flag)

    stack["driver"].final_validation_reader = read_current_endpoint

    def hand_f000_first_call(backend: Any) -> None:
        with time_ledger.stage("forecast.output.f000_first_call"):
            first_call.put(f000_first_call.read_arwen_first_call(
                backend, cp, solve_cells=stack.get("solve_cells")))
        if held_f000:
            finish_capture(0, held_f000.pop(), cycling_done=True)

    try:
        for step in range(start_step + 1, steps + 1):
            if step == start_step + 1:
                time_ledger.emit("forecast.construct", "end")
                time_ledger.emit("forecast.integrate", "begin", attrs=card_attrs)
                time_ledger.emit("forecast.first_step", "begin", attrs=card_attrs)
            if step > start_step + 1:
                snapshot_mark = time.perf_counter()
                rollback.capture(step - 1, previous, gf_dynamics_tendencies)
                snapshot_seconds += time.perf_counter() - snapshot_mark
            mark = time.perf_counter()
            replay_before = (0 if graph_controller is None else
                             graph_controller.stats["replayed_subcycles"])
            try:
                result = proof.execute_composite_step(
                    driver=stack["driver"],
                    backend=stack["backend"],
                    scalar_names=host["scalar_names"],
                    physics_geometry=stack["physics_geometry"],
                    kernel_cache=stack["driver"].cache,
                    previous_surface_updates=previous,
                    dynamics_tendencies=gf_dynamics_tendencies,
                    physics_park=physics_park,
                    # WRF/native diagflag: the step ENDING at a history frame
                    # computes refl10cm inside its own microphysics call.
                    refl_10cm_due=(step in capture_steps or radar.due(step) or
                                   (nest_trigger is not None and nest_trigger.due(step)
                                    and "refl10cm" in nest_trigger.needed_sources())),
                    **({} if iau_forcing is None else {"held_forcing": (
                        lambda held, start, length, **context:
                        iau_forcing.add(0, held, start, length, **context))}),
                )
            except (proof.CompositeTransactionError, FloatingPointError) as error:
                # An earlier health refusal has precedence over a successor's
                # transaction refusal, exactly as in the serial loop.
                if not read_pending_health():
                    break
                if not stop_on_refusal:
                    receipt_writer.close()
                    raise
                if (graph_controller is not None and
                        graph_controller.stats["replayed_subcycles"] != replay_before):
                    # F1 follow-through (2026-10-15): this step's graph replays
                    # wrote into the buffers the committed state shares with the
                    # graph; the armed snapshot of the previous step (taken
                    # before this step) puts its values back.
                    restore_refused_dycore_values(rollback, step - 1)
                # The port refused to publish this step.  The staged two-owner
                # transaction rolled back, so the committed state is still the
                # previous step.  NO GUARD IS RELAXED: the refusal is recorded
                # verbatim, the forecast stops here, and every retained frame
                # precedes the refused step.
                refusal = {
                    "refused": True,
                    "step": step,
                    "model_seconds": step * DT_SECONDS,
                    **refusal_exception_details(error),
                    "last_committed_step": step - 1,
                    "note": (
                        "the port own numeric/geometry validation refused this "
                        "step; the forecast is truncated at the last committed "
                        "step and no unpublished state was retained"
                    ),
                }
                executed_steps = step - 1
                break
            # Numeric transaction gates remain owned by the predecessor. They
            # already close the endpoint D2H event. The main loop needs no stream
            # drain, and inspects the prior health only after the next launch.
            last_step_clock = (time.monotonic(), time.time())
            if step == start_step + 1:
                time_ledger.emit("forecast.first_step", "end", attrs=card_attrs)
            elapsed = time.perf_counter() - mark
            current_health = stack["driver"]._step_gate_envelope
            stack["driver"]._step_gate_envelope = None
            if current_health is None or current_health[0] is not stack["driver"].atmosphere:
                receipt_writer.close()
                raise RuntimeError("late health requires the predecessor endpoint envelope; "
                                   "reading mutable successor fields would check the wrong step")
            if not read_pending_health():
                break
            if graph_controller is not None:
                replays = graph_controller.stats["replayed_subcycles"] - replay_before
                if replays == int(stack["driver"].config.config_dynamics_split_steps):
                    graph_full_steps += 1
                elif replays:
                    graph_partial_steps += 1
            integration_seconds += elapsed
            if first_step_seconds is None:
                first_step_seconds = elapsed
            previous = result.committed.surface_updates
            gf_dynamics_tendencies = result.dynamics_tendencies
            current_gf["value"] = gf_dynamics_tendencies
            if balance is not None:
                balance.sample(balance_view, float(step) * float(DT_SECONDS), cp)
            current_gf["radii"] = getattr(result.committed, "effective_radii", None)
            backend_receipt = dict(result.backend_receipt)
            surface_execution = proof.require_arwen_v2_surface_execution(
                backend_receipt, executed=True, label=f"step {step} backend receipt"
            )
            receipt_writer.submit(step, elapsed, result, surface_execution)
            if nest_trigger is not None:
                if nest_trigger.due(step):
                    from woof.hex.nest_look_dycore import look_arrays
                    needed = nest_trigger.needed_sources()
                    arrays = look_arrays(
                        refl=stack["backend"].peek_history_refl10cm() if "refl10cm" in needed else None,
                        needed=needed, dycore=nest_trigger.dycore_sources, driver=stack["driver"])
                    if arrays is not None:
                        nest_trigger.look(step, arrays)
                    if "refl10cm" in needed and step not in capture_steps:
                        # Formed only for the look: clear it so no later
                        # capture is handed this step's reflectivity.
                        stack["backend"].take_history_refl10cm()
                nest_trigger.poll()
            pending_health = (step, current_health, stack["driver"].atmosphere)
            if step == 1 and first_call is not None:
                hand_f000_first_call(stack["backend"])
            # A capture re-prepares live diagnostics and consumes refl10cm. Drain
            # at that boundary so no successor changes any output byte or gate.
            output_due = (step in capture_steps or radar.due(step) or
                          (fingerprints is not None and step % fingerprint_every == 0))
            if (output_due or step == steps) and not read_pending_health():
                break
            if fingerprints is not None and step % fingerprint_every == 0:
                mark = time.perf_counter()
                fingerprints.write(step, proof.fingerprint_execution_boundary(stack))
                fingerprint_seconds += time.perf_counter() - mark
            if radar.due(step) and step not in capture_steps:
                radar.capture(step, stack, host, proof, static)
            if step in capture_steps:
                try:
                    capture(step)
                except (
                    FloatingPointError,
                    RuntimeError,
                    proof.CompositeTransactionError,
                ) as error:
                    if not stop_on_refusal:
                        raise
                    # Writing a frame re-runs the MPAS-to-physics preparation.  The
                    # state immediately before the instability already fails it, so
                    # this frame cannot be produced.  Stop here with the frames that
                    # were captured cleanly.
                    refusal = {
                        "refused": True,
                        "refused_in": "history capture",
                        "step": step,
                        "model_seconds": step * DT_SECONDS,
                        **refusal_exception_details(error),
                        "last_committed_step": step,
                        "note": (
                            "the step integrated and published, but the port own "
                            "numeric/geometry validation refused to prepare it for "
                            "physics, so no history frame exists for it; the "
                            "forecast is truncated at the last frame written"
                        ),
                    }
                    executed_steps = step
                    break
    finally:
        radar.close()
        if first_call is not None:
            first_call.put(None)  # no first call will come; never leave a writer waiting
        with time_ledger.stage("post.forecast.step_receipts"):
            step_receipts = receipt_writer.close()
    if held_f000:
        finish_capture(0, held_f000.pop(), cycling_done=True)
    loop_seconds = time.perf_counter() - loop_started
    if nest_trigger is not None:
        nest_trigger.poll(final=True)
        (output_root / "nest-trigger-receipt.json").write_text(
            json.dumps(nest_trigger.receipt(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8", newline="\n")
    if last_step_clock is not None:
        if product_ready is not None:
            product_ready.forecast_ended(*last_step_clock)
        time_ledger.emit("forecast.integrate", "end",
                         attrs={"last_step": executed_steps, "refused": refusal is not None},
                         t_mono=last_step_clock[0], t_unix=last_step_clock[1])
        time_ledger.emit("post.forecast", "begin",
                         t_mono=last_step_clock[0], t_unix=last_step_clock[1])
    if (refusal is not None and may_capture_refusal_state(refusal)
            and executed_steps > 0 and executed_steps not in capture_steps):
        # Keep the last committed state renderable even though the refusal
        # landed between scheduled captures.
        stamp = datetime.strptime(schedule["start_time"], "%Y-%m-%d_%H:%M:%S") + timedelta(
            seconds=executed_steps * DT_SECONDS
        )
        labels[executed_steps] = stamp.strftime("%Y-%m-%d_%H.%M.%S")
        schedule["valid_times"][executed_steps] = stamp.strftime("%Y-%m-%d_%H:%M:%S")
        install_capture_labels(labels)
        try:
            capture(executed_steps)
        except Exception as error:  # noqa: BLE001 - bonus capture, never fatal
            # Capturing a snapshot re-runs the MPAS-to-physics preparation, and
            # on a state the port has just refused that preparation refuses
            # again.  The last committed state is therefore not representable
            # as a history frame.  Say so; keep the scheduled frames captured
            # before the instability.
            refusal["final_state_capture_failed"] = True
            refusal["final_state_capture_error"] = str(error)
            refusal["final_state_capture_exception_details"] = refusal_exception_details(error)
            labels.pop(executed_steps, None)
            schedule["valid_times"].pop(executed_steps, None)
        else:
            capture_steps.add(executed_steps)
    retired_physics = None
    if refusal is not None:
        steps = executed_steps
        retired_physics = retired_physics_seam(stack["backend"])
        if retired_physics is not None:
            refusal["physics_seam"] = retired_physics
    if fingerprints is not None:
        from woof.hex.final_identity import carried_digest
        proof._write_exclusive_json(
            output_root / "carried-identity.json",
            {"final_state_digest": None, "physics_seam": retired_physics}
            if retired_physics is not None
            else {"final_state_digest": carried_digest(
                stack["driver"].atmosphere, stack["backend"].restart_state(),
                gf_dynamics_tendencies, previous,
                gf_required=(executed_steps > 0 and host["convection"]["scheme"] != "off"))},
        )
        fingerprints.close()
    if capture_pool is not None:
        try:
            for pending in capture_pending:
                pending.result()
        finally:
            capture_pool.shutdown(wait=True)
    def drain_output(write_wait_seconds=write_wait_seconds):
        radar.close()
        history_receipt = None
        if history is not None:
            mark = time.perf_counter()
            time_ledger.emit("forecast.output.wait", "begin")
            try:
                for key in list(snapshot_files):
                    snapshot_files[key] = history.result(snapshot_files[key])
                    if product_ready is not None:
                        step = int(key)
                        product_ready.submit(snapshot_files[key], step=step, label=labels[step],
                                             valid_time=schedule["valid_times"][step])
            finally:
                history.close()
                time_ledger.emit("forecast.output.wait", "end")
            write_wait_seconds = time.perf_counter() - mark
            history_receipt = history.receipt()
        product_ready_receipt = None
        if product_ready is not None:
            mark = time.perf_counter()
            product_ready.close()
            product_ready_receipt = product_ready.receipt()
            product_ready_receipt["final_drain_seconds"] = time.perf_counter() - mark
        return history_receipt, write_wait_seconds, product_ready_receipt

    deferred = bool(DEFER_TAIL)
    history_receipt = product_ready_receipt = None
    if not deferred:
        history_receipt, write_wait_seconds, product_ready_receipt = drain_output()

    proof._write_exclusive_json(
        output_root / "snapshot-hashes.json",
        {"projection": snapshot_projection, "q2": snapshot_q2},
    )

    if hasattr(stack["backend"], "write_ledger"):
        stack["backend"].write_ledger(output_root)

    evolution = None
    if 0 in capture_steps and steps in capture_steps:
        evolution = {
            "note": (
                "surface evolution is reported by the per-step health trace; the "
                "proof's two-snapshot evolution gate is bound to its own case"
            )
        }
    from woof.hex.cuda_backend.arch_admission import architecture_status

    from woof.hex.final_identity import final_carriers, final_digest_of
    identity_carriers = None
    if retired_physics is None:
        with time_ledger.stage("post.forecast.identity.copy" if deferred else "post.forecast.identity"):
            identity_carriers = final_carriers(
                stack, gf_dynamics_tendencies, host["surface_accumulators"],
                None if executed_steps == 0 else result.committed.surface_updates,
                to_host=deferred,
            )
    final_state_digest = None
    if not deferred:
        with time_ledger.stage("post.forecast.identity"):
            final_state_digest = None if identity_carriers is None else final_digest_of(identity_carriers)

    iau_record = None
    if iau_forcing is not None:
        iau_record = {**iau_receipt, **iau_forcing.receipt()}
        if (executed_steps * DT_SECONDS >= float(iau.get("window_start_seconds") or 0.0) + float(iau["window_seconds"])
                and not iau_record.get("complete")):
            raise RuntimeError(f"IAU window not fully applied: {iau_record}")
    returned = {
        "stochastic": (None if getattr(stack["driver"], "stochastic", None) is None
                       else stack["driver"].stochastic.receipt()),
        **({"iau": iau_record} if iau_record is not None else {}),
        **({"balance_series": {"path": str(output_root / "balance-series.jsonl"),
                               "rows": len(balance.rows)}} if balance is not None else {}),
        "final_state_digest": final_state_digest,
        "prepared_content_sha256": host["prepared"].expected_execution_fingerprint["sha256"],
        "prepared_execution_fingerprint": host["prepared"].expected_execution_fingerprint,
        "capability": capability.as_dict(),
        **({"cuda_graphs": {
            "requested": True,
            "active": graph_controller is not None,
            "fully_replayed_committed_steps": graph_full_steps,
            "partly_replayed_committed_steps": graph_partial_steps,
            "subcycles": None if graph_controller is None else graph_controller.receipt(),
            "skip_reason": "local_timestep_controller" if lts_attachment is not None else None,
        }} if graph_requested else {}),
        "cuda_graphs_policy": graph_policy(int(N_CELLS)),
        "architecture": architecture_status(capability.compute).as_dict(),
        "arwen_pre_kernel_cache_pin": arwen_pin,
        "memory_admission": memory,
        "device_memory_release": _phase_release_receipt(),
        "physics_tier_park": (
            None
            if physics_park is None
            else {
                **physics_park.receipt(),
                "window": (
                    "held on pinned host memory from after the phase-1 "
                    "tendencies are coupled until immediately before phase 2"
                ),
                "nonclaim": (
                    "not bit-identity: restored allocations sit at different "
                    "device addresses, so the payload digest of the two arms "
                    "is the only evidence that admits or refuses this"
                ),
            }
        ),
        "source_pins": source_receipt,
        "authority": authority_receipt,
        "regional": (
            None
            if host.get("regional") is None
            else {
                **host["regional"],
                "start_time": host["regional"]["start_time"].strftime(
                    "%Y-%m-%d_%H:%M:%S"
                ),
                "config_apply_lbcs": True,
                "lbc_intervals": len(host["regional"]["lbc_paths"]),
                "anchor": (
                    None
                    if getattr(stack["driver"], "regional_v841", None) is None
                    else getattr(stack["driver"].regional_v841, "anchor", None)
                ),
                "runtime": (
                    None
                    if getattr(stack["driver"], "regional_v841", None) is None
                    else stack["driver"].regional_v841.receipt()
                ),
            }
        ),
        "host_preparation": {
            "constructor": host["constructor_receipt"],
            "scalars": host["scalar_receipt"],
            "mesh_overlay": host["mesh_evidence"],
            "reconstruction_coefficients": host["reconstruction_coefficients"],
            "edge_normal_vectors": host["edge_normal_vectors"],
            "inactive_deformation": host["defc"],
            "case_pin_relaxation": host["case_pin_relaxation"],
            "f000_start_humidity": {
                key: value
                for key, value in host["f000_start_humidity"].items()
                if key != "q2"
            },
        },
        "schedule": {
            key: value for key, value in schedule.items() if key != "labels"
        },
        "history_labels": schedule["labels"],
        "walls": {
            "integration_seconds": integration_seconds,
            "integration_note": (
                "sum of the accepted per-step composite transactions, closed by "
                "the predecessor numeric gate event; EXCLUDES host preparation, device stack "
                "construction, snapshot capture, history writing, boundary "
                "fingerprinting and the per-step health gate"
            ),
            "first_step_seconds": first_step_seconds,
            "integration_seconds_excluding_first_step": (
                integration_seconds - (first_step_seconds or 0.0)
            ),
            "first_step_note": (
                "the first step carries cold NVRTC kernel compilation for this "
                "cache root"
            ),
            "seconds_per_step_after_first": (
                (integration_seconds - (first_step_seconds or 0.0)) / (steps - 1)
                if steps > 1
                else None
            ),
            "loop_seconds_including_io": loop_seconds,
            "snapshot_capture_seconds": capture_seconds,
            "snapshot_capture_pipeline": {
                "mode": "stream-ordered background capture" if async_capture else "synchronous capture",
                "event_wait_thread": "hex-capture" if async_capture else "forecast",
                "finish_seconds": capture_finish_seconds,
                "max_materializing_frames": 1 if async_capture else 0,
                "max_writer_frames": history.max_pending if history is not None else 0,
                "hash_threads": history_hash_threads(),
                "backpressure": "previous materialization before the next history boundary",
                "note": "enqueue and worker durations overlap integration; neither is device kernel time",
            },
            "history_write_seconds": write_seconds,
            "history_write_wait_after_loop_seconds": write_wait_seconds,
            "history_writer": history_receipt,
            **({"history_product_ready": product_ready_receipt}
               if product_ready_receipt is not None else {}),
            "boundary_fingerprint_seconds": fingerprint_seconds,
            "health_gate_seconds": health_seconds,
            "steps": steps,
            "forecast_seconds": steps * DT_SECONDS,
        },
        "physical_gates": physical_gates,
        "step_health": health,
        "late_health_pipeline": {**rollback.receipt(), "default_on": True,
                                 "rollback_armed": bool(stop_on_refusal),
                                 "snapshot_seconds": snapshot_seconds,
                                 "receipt_transport": "background thread compact JSONL",
                                 "phase_boundary_numeric_gates": "synchronous predecessor gates"},
        "step_receipt_count": len(step_receipts),
        "physics_cadence": _physics_cadence_proof(
            host.get("pbl_cadence"), step_receipts, executed_steps - start_step
        ),
        "physics_rollback": {
            "armed": bool(stop_on_refusal),
            "reason": (
                "--stop-on-refusal keeps a refused step's boundary and writes "
                "its frame from it"
                if stop_on_refusal
                else "no consumer: without --stop-on-refusal a refused step "
                "ends the run, so the per-step boundary export is not taken"
            ),
            "boundary_export_d2h_bytes_per_step": (
                None
                if not step_receipts
                else step_receipts[-1]["backend"]
                .get("copies", {})
                .get("transaction_boundary_snapshot_d2h_bytes")
            ),
        },
        "snapshot_receipts": snapshot_receipts,
        "snapshot_files": snapshot_files,
        "refusal": refusal,
        "steps_requested": int(schedule["steps"]),
        "steps_executed": executed_steps,
        "boundary_fingerprints": {
            "path": str(fingerprint_path) if fingerprints is not None else None,
            "every": fingerprint_every,
            "steps": fingerprints.steps if fingerprints is not None else [],
        },
        "surface_evolution": evolution,
        "local_timestep_treatment": (
            None
            if stack.get("local_timestep") is None
            else stack["local_timestep"].receipt()
        ),
        **({"fullstep_lts_treatment": fullstep_shadow.receipt()}
           if fullstep_shadow is not None else {}),
        "horizontal_mixing": _mixing_treatment_proof(
            stack["driver"], executed_steps - start_step
        ),
        "gf_deviation": {
            "mpas_dynamics_tendencies_computed": True,
            "fa35_public_api_accepts_them": False,
            "fa35_rthften_rqvften": "zero",
            "native_gf_parity_claim": False,
        },
        "full_physics_cuda_executed": True,
        "jedi_contract": (
            None
            if cycling is None
            else {
                "background_files": jedi_files,
                "soil_levels": int(cycling["soil_levels"]),
                "checkpoint_files": checkpoint_files,
                "restart": restart_receipt,
                "start_step": start_step,
            }
        ),
    }
    if deferred:
        def deferred_tail():
            """The host-only tail, run by the roster after the card is handed on: the history drain and
            the identity hash overlap (one waits on the writer, the other hashes host copies)."""
            from concurrent.futures import ThreadPoolExecutor as _Pool

            def identity():
                with time_ledger.stage("post.forecast.identity"):
                    return None if identity_carriers is None else final_digest_of(identity_carriers)

            with _Pool(max_workers=1, thread_name_prefix="hex-identity") as pool:
                digest = pool.submit(copy_context().run, identity)
                history_receipt, wait_seconds, product_ready_receipt = drain_output()
                returned["final_state_digest"] = digest.result()
            returned["walls"]["history_write_wait_after_loop_seconds"] = wait_seconds
            returned["walls"]["history_writer"] = history_receipt
            if product_ready_receipt is not None:
                returned["walls"]["history_product_ready"] = product_ready_receipt

        returned["_deferred_tail"] = deferred_tail
    return returned


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullstep-local-timestep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fullstep-lts-mode", choices=("shadow", "active"), default=None)
    parser.add_argument("--fullstep-lts-max-rate", type=int, choices=(1,2,4,8,16,32), default=None)
    parser.add_argument("--fullstep-lts-buffer-rings", type=int, default=None)
    defaults = proof.default_authority_paths()
    parser.add_argument("--grid", type=Path, default=defaults["grid"])
    parser.add_argument("--static", type=Path, default=defaults["static"])
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument(
        "--invariant", type=Path, default=None,
        help="the native invariant stream file, for a run that configures "
             "one (RRFS v2 does): the vertical grid is read from it, as "
             "MPAS-A does, instead of from the init file",
    )
    parser.add_argument(
        "--init-source",
        required=True,
        help="provenance sentence for the init (e.g. 'ERA5 2025-03-14 12Z')",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="asserted against the init's config_start_time; the init is the authority",
    )
    parser.add_argument("--hours", type=float, required=True)
    # MPAS-A v8.4.2 Registry defaults (woof.hex.mpas_defaults): order 2
    # (Registry.xml:69), four acoustic sub-steps (:104; two through v8.4.1,
    # still selectable), radiation every
    # 00:30:00 (:2316, :2321), the monotonic limiter on (:239).
    parser.add_argument("--time-integration-order", type=int, choices=(2, 3), default=2)
    parser.add_argument("--acoustic-substeps", type=int, default=4)
    parser.add_argument("--radiation-seconds", type=float, default=1800.0)
    parser.add_argument("--monotonic", action=argparse.BooleanOptionalAction, default=True,
                        help="the native flux-corrected scalar limiter (MPAS-A default on; --no-monotonic turns it off)")
    parser.add_argument("--cam-coefficient",type=float,default=0.0)
    parser.add_argument("--cam-damping-levels",type=int,default=4)
    parser.add_argument("--rayleigh-u",action="store_true")
    parser.add_argument("--damping-start-m",type=float,default=22000.0)
    parser.add_argument(
        "--dynamics", action="append", default=None, metavar="KEY=VALUE",
        help="set a dynamics option that has no flag of its own, by its "
             "MPAS namelist name (repeatable): config_apvm_upwinding=0.25, "
             "config_smdiv=0.2, config_coef_3rd_order=1.0, config_xnutr=0.2, "
             "config_positive_definite=true, config_smagorinsky_coef=0.125")
    parser.add_argument("--history-every-minutes", type=float, required=True)
    parser.add_argument("--history-from-minutes", type=float, default=0.0)
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=None,
        help="woof tree whose seam bytes run: a git clone at the pinned tag, "
             "or unset for the installed engine",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fingerprint-every",
        type=int,
        default=0,
        help="write a boundary fingerprint every N steps (0 disables; the "
        "fork-equivalence gate uses 1)",
    )
    parser.add_argument("--case-label", default=None)
    parser.add_argument(
        "--physics-mode",
        choices=("exact", "correct"),
        default="exact",
        help=(
            "physics.mode (woof.hex.correct_mode): 'exact' is the matched "
            "arithmetic; 'correct' applies every correction the engine "
            "carries (WB4 New Tiedtke dry-mass coupling, WB5 WSM6 "
            "conservative surface integral); one the engine does not carry "
            "runs as the matched arithmetic and is named as a gap in the "
            "receipt.  The default here stays 'exact' because the door "
            "passes no flag for exact; the door's own default is 'correct'"
        ),
    )
    parser.add_argument(
        "--physics-correct-skip",
        action="append",
        default=None,
        metavar="FINDING",
        help="drop one correction from correct mode (an A/B arm); repeatable",
    )
    parser.add_argument(
        "--horiz-mixing",
        choices=("2d_smagorinsky", "off"),
        default="2d_smagorinsky",
        help=(
            "horizontal mixing lane; the default is native's Registry "
            "2d_smagorinsky (deformation-based, woof.hex.mixing_v841). "
            "'off' selects the pre-mixing control lane and is reported as "
            "the configuration native itself cannot integrate on "
            "convective cases"
        ),
    )
    parser.add_argument(
        "--convection",
        default="auto",
        help=(
            "cumulus selection.  The default 'auto' switches the cumulus "
            "scheme off where the bound mesh's finest spacing is below 3 km, "
            "with no flag passed: Grell-Freitas is called every model step, "
            "so a fine mesh's short step calls it up to 24 times as often as "
            "the proven 120 s, which measured as a different solution, and "
            "convection is resolved explicitly at those spacings.  'off' and "
            "'gf' are explicit A/B arms -- they record themselves as "
            "explicit in the receipt, and an arm that overrides the default "
            "says so.  'ntiedtke' is a profile's explicit selection of New "
            "Tiedtke, which 'auto' also picks at 20 km and coarser.  "
            "See woof.hex.convection_admission"
        ),
    )
    parser.add_argument(
        "--pbl-cadence",
        default="auto",
        metavar="{auto,SECONDS}",
        help=(
            "surface/PBL cadence in seconds (config_bldt_seconds).  The "
            "default 'auto' is the PROVEN CONFIGURATION: the cadence is "
            "welded to config_dt, so the surface layer, the land-surface "
            "model and the PBL run on every model step, exactly as the "
            "native x4 v8.4.1 reference ran.  A number of seconds calls the "
            "stack once every SECONDS/dt steps and holds its tendency on "
            "the steps between (the engine's own positive-bldt path); a "
            "value that is not a whole number of steps is refused by name.  "
            "It changes the forecast and records itself as an explicit "
            "selection, with its calls per hour, in the receipt.  See "
            "hexcore.pbl_cadence"
        ),
    )
    parser.add_argument(
        "--local-timestep",
        action="store_true",
        help=(
            "OPT-IN, default off: advance coarse columns on fewer, longer "
            "acoustic sub-steps chosen from the grid file's own dcEdge. "
            "Native MPAS-A v8.4.1 has no local time stepping, so this is a "
            "DECLARED DIVERGENCE from native and the run is not bit-comparable "
            "to a default run on a variable-resolution mesh. On a quasi-uniform "
            "mesh every column lands in one class and the run is bit-identical "
            "to the default"
        ),
    )
    parser.add_argument(
        "--local-timestep-declared-off",
        action="store_true",
        help=(
            "GATE ARM, not a user feature: build the local-timestep "
            "configuration subtype with the switch OFF. The run must be "
            "bit-identical to a run with no local-timestep flag at all, which "
            "is what proves the option did not leak into the default path"
        ),
    )
    parser.add_argument(
        "--local-timestep-rates",
        default="1,3",
        help=(
            "comma-separated acoustic rate ladder; each rate must divide every "
            "RK stage's sub-step count, which for the released (1,3,6) schedule "
            "admits 1 and 3 only. Two classes by default"
        ),
    )
    parser.add_argument(
        "--local-timestep-buffer-rings",
        type=int,
        default=1,
        help="rings of cells demoted to the finer rate around a class boundary",
    )
    parser.add_argument(
        "--local-timestep-classing",
        type=Path,
        default=None,
        metavar="NPZ",
        help=(
            "A/B INSTRUMENT, not a user feature: take the per-cell acoustic "
            "rates from this .npz (key cell_rate, one rate per cell) instead of "
            "classing from dcEdge, so a class interface can be placed in the "
            "interior of a cull whose spacing would class it into one rate. "
            "The driven boundary zone is still held at rate 1 and an "
            "interface touching it is refused. Requires --local-timestep; "
            "tools/lts_forced_classing.py writes the file"
        ),
    )
    parser.add_argument(
        "--park-physics-tier",
        action="store_true",
        help=(
            "MEASURED NEGATIVE, kept only so the measurement reproduces: hold "
            "the frozen-WOOF physics seam's device residency in pinned host "
            "memory across the dynamics half of every step.  It moves 786.8 "
            "MiB and releases 735.4 MiB of it, and on x1.40962 it still made "
            "the allocator's reservation WORSE -- 4068.7 MiB with the park "
            "against 4055.3 MiB without it -- because the reservation, not "
            "the instantaneous peak, is what a card has to provide.  Alone it "
            "changes neither number, since the peak is inside phase-1 physics "
            "where the tier is being read.  Those absolutes were measured "
            "2026-08-20 at WOOF seam pin 629ddb6f0, BEFORE the Grell-Freitas "
            "local-memory frame cut; the A/B sign is what carries, and "
            "re-running the park at pin 0d04db712 is NOT MEASURED.  Do not "
            "reach for this as an optimisation.  Also not bit-identity: "
            "restored allocations sit at different device addresses, so "
            "arm-to-arm payload digests are the only evidence that admits a "
            "parked run"
        ),
    )
    parser.add_argument(
        "--required-free-bytes",
        type=int,
        default=None,
        help=(
            "free-device-byte requirement computed by the forecast door's "
            "admission (woof.hex.device_admission.required_free_bytes over "
            "the card's resolved footprint row), forwarded so the door's "
            "verdict and this driver's floor are the same number.  Default: "
            "the mesh-bound floor from the same admission surface"
        ),
    )
    parser.add_argument(
        "--lbc-dir",
        type=Path,
        default=None,
        help=(
            "directory of lateral-boundary files (lbc.*.nc, as rw_mpas_lbc "
            "writes them) for a limited-area grid.  REQUIRED when --grid "
            "carries a bdyMask triple and refused when it does not: a "
            "limited-area domain integrated with no boundary series empties "
            "from its outer ring inward, and a global domain has no boundary "
            "zone for one to drive"
        ),
    )
    parser.add_argument(
        "--physics-backend",
        default="wsm6_column",
        help=(
            "the column-physics backend row (woof.hex.physics_backend_admission); "
            "the default is the frozen lane"
        ),
    )
    parser.add_argument("--chemistry-config", type=Path, default=None)
    parser.add_argument("--wind-farms", default=None)
    parser.add_argument(
        "--source-table",
        type=Path,
        default=None,
        help="a point-source table the selected row's seam releases from",
    )
    parser.add_argument(
        "--jedi-background",
        action="store_true",
        help=(
            "at every history frame also write mpasout.<valid>.nc: the 27-field "
            "background stream RRFS v2's MPAS-JEDI reads, in native MPAS form "
            "(CDF-5, MPAS names and dimensions), with a per-field provenance "
            "receipt (woof.hex.jedi_contract)"
        ),
    )
    parser.add_argument(
        "--jedi-soil-levels",
        type=int,
        default=9,
        help=(
            "soil levels in the JEDI background.  RRFS v2's namelist declares "
            "nine (RUC); hex carries four Noah-MP layers, so 9 writes the "
            "labelled INTERIM mapping (woof.hex.jedi_contract."
            "interim_soil_levels) and 4 writes the layers as they are"
        ),
    )
    parser.add_argument(
        "--checkpoint-hours",
        default="",
        help=(
            "comma-separated forecast hours (each a history frame) at which "
            "to write checkpoint.<valid>.pkl: the full hex restart state "
            "(dycore state and saved diagnostics, physics backend payload, "
            "GF advective-forcing carrier) a --restart-from run resumes from"
        ),
    )
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help=(
            "resume from a checkpoint.<valid>.pkl this driver wrote.  "
            "--start-time/--hours keep naming the ORIGINAL run, so step "
            "numbers, labels and the boundary clock continue unbroken"
        ),
    )
    parser.add_argument(
        "--da-analysis",
        type=Path,
        default=None,
        help=(
            "DA-cycling restart (upstream config_do_DAcycling): with "
            "--restart-from, replace the checkpoint's dycore state by the "
            "analysis file's rho/theta/species/u/w and recompute the coupled "
            "fields with hex's transcription of atm_init_coupled_diagnostics; "
            "physics state resumes from the checkpoint"
        ),
    )
    parser.add_argument(
        "--physics-suite",
        default=None,
        help="the named physics suite the door resolved (woof.hex.physics_suites)",
    )
    parser.add_argument(
        "--physics",
        action="append",
        default=None,
        help="one physics override KEY=VALUE, as the door resolved it",
    )
    parser.add_argument(
        "--physics-mesh",
        default=None,
        help="the mesh row a suite's per-mesh selectors key on",
    )
    parser.add_argument(
        "--orographic-statistics",
        type=Path,
        default=None,
        help=(
            "the mesh's orographic statistics file (<mesh>.ugwp_oro_data.nc), "
            "read by a suite row whose drag uses it"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--stop-on-refusal",
        action="store_true",
        help=(
            "when the port refuses to publish a step, stop and write the "
            "receipt for the frames already committed instead of aborting "
            "with no receipt; the refusal is recorded verbatim and no "
            "validation is relaxed"
        ),
    )
    parser.add_argument(
        "--iau-analysis", type=Path, default=None,
        help="incremental analysis update: the analysis init (same mesh and cell order as --init, "
             "which is the background); analysis minus background enters as a constant tendency "
             "over --iau-window-seconds (woof.hex.iau_forcing)")
    parser.add_argument("--iau-window-seconds", type=float, default=None,
                        help="IAU window length from the start of the run")
    parser.add_argument(
        "--iau-increment", type=Path, default=None,
        help="MPAS IAU increments (the iau stream: u_amb, rho_amb, theta_amb, q*_amb), the analysis "
             "minus --init, which is the background (woof.hex.analysis_bundle); native route")
    parser.add_argument(
        "--iau-route", choices=IAU_ROUTES, default=None,
        help="native: MPAS-A v8.4.2's own IAU kernels (woof.hex.iau_v842, through woof.hex.iau_native) "
             "convert the uncoupled increments every step against that step's state; physics.mode "
             "exact runs them word for word, correct applies IN1-IN3.  hex: woof.hex.iau_forcing adds "
             "the coupled analysis minus background as a constant tendency.  Default: "
             + ", ".join(f"{route} in {mode} mode" for mode, route in IAU_ROUTE_DEFAULT.items()))
    parser.add_argument("--stochastic", default=None, metavar="JSON")
    parser.add_argument("--balance-series", action="store_true",
                        help="write balance-series.jsonl: surface-pressure tendency statistics "
                             "at every macro boundary (initialization-shock measure)")
    args = parser.parse_args(argv)
    has_increment = args.iau_analysis is not None or args.iau_increment is not None
    if has_increment != (args.iau_window_seconds is not None):
        parser.error("--iau-analysis (or --iau-increment) and --iau-window-seconds go together: an "
                     "increment without a window (or a window without an increment) has no defined forcing")
    if args.iau_window_seconds is not None and not args.iau_window_seconds > 0:
        parser.error("--iau-window-seconds must be positive; a zero window is direct insertion")
    if args.iau_route is not None and not has_increment:
        parser.error("--iau-route selects how an increment enters; it needs --iau-analysis or --iau-increment")
    if has_increment:
        args.iau_route = args.iau_route or IAU_ROUTE_DEFAULT[args.physics_mode]
        if args.iau_route == "hex" and args.iau_analysis is None:
            parser.error("--iau-route hex forms its increment from two recoupled initial-conditions files "
                         "and needs --iau-analysis; an MPAS increment file alone runs --iau-route native")
        for flag, value in (("--iau-analysis", args.iau_analysis), ("--iau-increment", args.iau_increment)):
            if value is not None and not Path(value).is_file():
                parser.error(f"{flag} {value} is not a file")
        # Retired (named gap W1-G1): a checkpoint carries its IAU window (woof.hex.iau_native.window_record),
        # and a restart continues that window or begins a new one by rule (resolve_restart_window), so the
        # increment is admitted once across a restart (IN3).
    fullstep_knobs = {name: getattr(args, name) for name in
                     ("fullstep_lts_mode", "fullstep_lts_max_rate", "fullstep_lts_buffer_rings")}
    args.fullstep_lts_options = None
    if args.fullstep_local_timestep:
        from woof.hex.config_fullstep_lts import validate_fullstep_lts_request
        validate_fullstep_lts_request(args)
        args.fullstep_lts_options = dict(fullstep_local_timestep=True,
            **{name: value for name, value in fullstep_knobs.items() if value is not None})
    elif any(value is not None for value in fullstep_knobs.values()):
        parser.error("the requested LTS policy would be ignored; enable --fullstep-local-timestep")
    if not args.preflight_only and (args.cache_root is None or args.output is None):
        parser.error("execution requires --cache-root and --output")
    if args.fingerprint_every < 0:
        parser.error("--fingerprint-every must be >= 0")
    if args.required_free_bytes is not None and args.required_free_bytes <= 0:
        parser.error(
            "--required-free-bytes must be positive: a non-positive "
            "requirement is not a measured admission, it is the memory gate "
            "switched off, and the run it admits dies inside a CuPy "
            "allocation mid-integration"
        )
    try:
        rates = tuple(
            int(piece) for piece in str(args.local_timestep_rates).split(",") if piece
        )
    except ValueError:
        parser.error("--local-timestep-rates must be comma-separated integers")
    if not rates or rates[0] != 1 or list(rates) != sorted(set(rates)):
        parser.error(
            "--local-timestep-rates must be strictly increasing and start at 1"
        )
    args.local_timestep_rates = rates
    if args.local_timestep and args.local_timestep_declared_off:
        parser.error(
            "--local-timestep and --local-timestep-declared-off contradict: "
            "the second is the gate arm that proves the option is inert when "
            "the switch is off"
        )
    if args.local_timestep_buffer_rings < 1:
        parser.error("--local-timestep-buffer-rings must be >= 1")
    try:
        args.checkpoint_hours = tuple(
            float(piece) for piece in str(args.checkpoint_hours).split(",") if piece.strip()
        )
    except ValueError:
        parser.error("--checkpoint-hours must be comma-separated numbers")
    if args.da_analysis is not None and args.restart_from is None:
        parser.error(
            "--da-analysis is a DA-cycling RESTART: it replaces the dycore "
            "state of a checkpoint, so it needs --restart-from; without one "
            "the physics state it resumes against would not exist"
        )
    if args.restart_from is not None and not Path(args.restart_from).is_file():
        parser.error(f"--restart-from {args.restart_from} is not a file")
    if args.da_analysis is not None and not Path(args.da_analysis).is_file():
        parser.error(f"--da-analysis {args.da_analysis} is not a file")
    if args.jedi_soil_levels not in (4, 9):
        parser.error("--jedi-soil-levels must be 4 (native) or 9 (RRFS v2, interim mapping)")
    if args.local_timestep_classing is not None:
        if not args.local_timestep:
            parser.error(
                "--local-timestep-classing is an instrument for the option "
                "and needs --local-timestep; without the switch the classing "
                "would be read and silently never used"
            )
        if not Path(args.local_timestep_classing).is_file():
            parser.error(
                f"--local-timestep-classing {args.local_timestep_classing} "
                "is not a file"
            )
    return args


def resolve_run_row(physics_backend: str, source_table: str | None):
    """The backend row, its scheme alias (None for the frozen row) and the
    seam options the row's adapter derives from the door's neutral inputs."""

    from woof.hex.physics_backend_admission import DEFAULT_BACKEND, resolve_backend

    row = resolve_backend(str(physics_backend))
    if row.name == DEFAULT_BACKEND:
        if source_table is not None:
            raise ConfigurationRefusal(
                "source_table",
                str(source_table),
                f"--physics-backend {row.name} carries no point source; the "
                "table would be accepted and never read",
                "a row whose adapter publishes seam_options_for",
            )
        return row, None, {}
    adapter = row.load_adapter()
    alias_for = getattr(adapter, "scheme_alias_for", None)
    options_for = getattr(adapter, "seam_options_for", None)
    if alias_for is None or options_for is None:
        raise ConfigurationRefusal(
            "physics_backend",
            row.name,
            (
                f"row {row.name!r} names adapter module {row.adapter_module}, "
                "which publishes no scheme_alias_for/seam_options_for; the "
                "driver cannot name the run's row in the configuration or "
                "derive the seam's options"
            ),
            "an adapter publishing scheme_alias_for and seam_options_for",
        )
    return row, str(alias_for(row.name)), dict(options_for(source_table=source_table))


def measured_engine_identity(checkout: Path) -> dict[str, Any]:
    """The engine the seam bytes of ``checkout`` were measured to be.

    THE BREAKAGE THIS CLOSES: through 0.3.1 the receipt's top-level
    ``arwen_commit`` and glacier digest restated the x4 proof's own sealed
    constants, so every forecast receipt named an engine commit older than
    the engine the run executed.  They now come from the identity the seam
    bytes match (the pin, or an admitted row), the one the adapter records.

    A tree that matches no measured row runs (the byte match stopped being
    a gate on 2026-10-10, :func:`woof.hex.engine_pin.identity_of`) and is
    named by the version it declares; its commit and glacier digest are
    ``None`` here because no measured row supplies them, and the tree's own
    commit is under ``arwen_git``.  ``arwen_engine_pin`` records the match,
    the moved files and the absent ones.
    """

    from woof.hex import engine_pin

    inspection = engine_pin.inspect_seam(checkout)
    identity = engine_pin.identity_of(inspection)
    record = {"arwen_engine_pin": engine_pin.measured_row_record(inspection)}
    if not identity.matches_measured_row:
        return {
            "arwen_commit": None,
            "arwen_engine_version": inspection.declared,
            "arwen_glacier_composed_tu_sha256": None,
            **record,
        }
    return {
        "arwen_commit": identity.build_commit,
        "arwen_engine_version": identity.version,
        "arwen_glacier_composed_tu_sha256": identity.glacier_composed_tu_sha256,
        **record,
    }


def _manifest_for_run(row, checkout: Path) -> dict[str, str] | None:
    """The byte manifest the run's row is bound by.

    For the frozen row on an engine a measured row names: None (the
    verifier reads that row's sixteen).  For the frozen row on any other
    engine: the sixteen paths as they are found NOW, so the verifier records
    the tree and its before/after comparison still says whether a file
    changed while the run was using it -- it no longer refuses the tree for
    not being the pinned build (retired 2026-10-10,
    :func:`woof.hex.engine_pin.identity_of`).  For a provider row: its own
    pin files plus its frozen-batch cross-pin.
    """

    from woof.hex import engine_pin
    from woof.hex.physics_backend_admission import DEFAULT_BACKEND

    if row.name == DEFAULT_BACKEND:
        inspection = engine_pin.inspect_seam(checkout)
        if engine_pin.identity_of(inspection).matches_measured_row:
            return None
        return dict(inspection.digests)
    probe = getattr(row.load_adapter(), "pin_column_batch", None)
    if probe is None:
        raise ConfigurationRefusal(
            "physics_backend",
            row.name,
            f"adapter {row.adapter_module} publishes no pin_column_batch",
            "an adapter publishing pin_column_batch",
        )
    pin = probe(checkout)["pin"]
    manifest = {relative: entry["sha256"] for relative, entry in pin["files"].items()}
    manifest.update(dict(pin.get("frozen_batch_cross_pin") or {}))
    return manifest


def _pin_for_run(host: Mapping[str, Any], checkout: Path, frozen_pin):
    """The seam pin the RUN's row is bound by: the engine manifest for the
    frozen row, the row's own adapter pin for a provider's row."""

    from woof.hex.physics_backend_admission import DEFAULT_BACKEND, resolve_backend

    row = resolve_backend(str(host.get("physics_backend", DEFAULT_BACKEND)))
    if row.name == DEFAULT_BACKEND:
        return frozen_pin(checkout)
    probe = getattr(row.load_adapter(), "pin_column_batch", None)
    if probe is None:
        raise ConfigurationRefusal(
            "physics_backend",
            row.name,
            f"adapter {row.adapter_module} publishes no pin_column_batch; the "
            "run cannot say which bytes this row would execute",
            "an adapter publishing pin_column_batch",
        )
    return probe(checkout)


def _backend_builder(host: Mapping[str, Any], checkout: Path):
    """``None`` for the frozen row; otherwise the row's own builder, handed
    the mesh's per-column area (the garbage column of a limited-area mesh
    is padded with the mean area: it holds no atmosphere and is scrubbed)."""

    from woof.hex.physics_backend_admission import DEFAULT_BACKEND, resolve_backend

    row = resolve_backend(str(host.get("physics_backend", DEFAULT_BACKEND)))
    if row.name == DEFAULT_BACKEND:
        return None

    def build(*, physics_mesh: Any, **kwargs: Any):
        area = np.asarray(proof._mesh_value(physics_mesh, "areaCell"), dtype=np.float64).reshape(-1)
        good = np.isfinite(area) & (area > 0.0)
        if not np.all(good):
            area = np.where(good, area, float(np.mean(area[good])))
        return row.build_column_backend(
            cell_area_m2=area,
            seam_options=dict(host.get("seam_options") or {}),
            **kwargs,
        )

    # A row that seals its own constructor (a suite row: its soil column and
    # drag statistics are not the frozen mapping's) hands the stack build
    # its sealer; the frozen row's sealer is the stack build's default.
    sealer = getattr(row.load_adapter(), "seal_constructor", None)
    if sealer is not None:
        build.seal_constructor = sealer
    return build


def claim_for_run(physics, convection: Mapping[str, Any], arwen_checkout) -> str:
    """The claim sentence for the physics this run executes.

    ``CLAIM`` lists the frozen suite, with the mesh and the cumulus already
    rebound at the mesh bind.  A bare default run keeps it byte for byte.
    Any other selection has its schemes named by the engine registry's
    labels (see ``mpas_mesh_binding.claim_for_selection`` for the receipt
    that named WSM6 for a Thompson run):

    * the generic seam: every label, from the mapping handed to the engine;
    * the column-suite seam: every label, from the selectors' assignments;
    * the frozen batch on another microphysics row: only the microphysics
      differs from the sentence, so only its term (the first) is renamed and
      hex's own drag and glacier terms stay as written.
    """

    if physics is None or physics.is_default():
        return CLAIM
    from woof.hex import convection_admission, engine_registry
    from woof.hex.drivers.mpas_mesh_binding import claim_for_selection, claim_scheme_terms

    checkout = engine_registry.checkout_with_registry(arwen_checkout)
    by_selector = getattr(physics, "engine_assignments", {})
    if physics.seam == "frozen":
        micro = by_selector.get("config_microp_scheme") or {}
        named = engine_registry.selection_labels(micro, checkout)
        if not named:
            return CLAIM
        return claim_for_selection(CLAIM, [*named, *claim_scheme_terms(CLAIM)[1:]])
    mapping = physics.engine_physics
    if mapping is None:
        mapping = {}
        for assignments in by_selector.values():
            for name, value in assignments.items():
                mapping.setdefault(name, value)
    labels = list(engine_registry.selection_labels(mapping, checkout, skip=("cu_physics",)))
    scheme = convection.get("scheme")
    if scheme != convection_admission.SCHEME_OFF:
        labels.insert(min(1, len(labels)), convection_admission.label(scheme))
    return claim_for_selection(CLAIM, labels)


def resolve_driver_physics(args: argparse.Namespace):
    """The named physics the door resolved, re-resolved from the same table.

    ``None`` when no suite was named (the legacy road).  The door passes the
    suite, its overrides, the mesh row the per-mesh selectors key on, and the
    backend row it decided; the driver resolves the same request with the
    same function and refuses if the row differs, so the receipt can never
    name a backend the configuration did not select.
    """

    suite = getattr(args, "physics_suite", None)
    raw = list(getattr(args, "physics", None) or ())
    if suite is None and not raw:
        return None
    from woof.hex import engine_registry
    from woof.hex.physics_suites import parse_override, resolve_physics

    # The same engine tree the run executes answers which options exist.
    overrides = dict(parse_override(text) for text in raw)
    physics = resolve_physics(
        suite, overrides, mesh=getattr(args, "physics_mesh", None),
        engine_checkout=engine_registry.checkout_with_registry(
            getattr(args, "arwen_checkout", None)))
    if physics.backend_row != str(args.physics_backend) and not (
        physics.backend_row == "wsm6_column" and args.physics_backend in (None, "wsm6_column")
    ):
        raise ConfigurationRefusal(
            "physics_backend",
            args.physics_backend,
            f"the physics suite {physics.suite!r} resolves to backend row "
            f"{physics.backend_row!r} and the driver was handed "
            f"{args.physics_backend!r}: one decision, two sources",
            f"--physics-backend {physics.backend_row}",
        )
    return physics


def resolve_physics_backend_row(args: argparse.Namespace):
    """The backend row this run selects, refused by name until routed.

    THE BREAKAGE THIS REFUSES, dated 2026-09-01: the stack build below
    constructs the frozen column batch through the frozen manifest and
    sizes its scalar block from the frozen row.  A provider's row would
    have its table accepted and never read and its appended scalars never
    allocated -- a run under a name it does not honour.  The routing
    (``PhysicsBackendRow.build_column_backend`` at the stack build, the
    row's scalar block in the driver, the row's history names) retires
    this refusal in the commit that lands it.
    """

    from woof.hex.physics_backend_admission import DEFAULT_BACKEND, resolve_backend

    if getattr(args, "chemistry_config", None) is not None:
        from woof.hex.chem_transport import configure, admit_arguments
        admit_arguments(args)
        args.physics_backend = configure(args.chemistry_config).name
    row = resolve_backend(str(args.physics_backend))
    table = getattr(args, "source_table", None)
    if table is not None and not Path(table).is_file():
        raise ConfigurationRefusal(
            "source_table",
            str(table),
            "--source-table names a file that does not exist; the seam pins "
            "the table's bytes into the run's identity",
            "an existing point-source table file",
        )
    if table is not None and row.name == DEFAULT_BACKEND:
        raise ConfigurationRefusal(
            "source_table",
            str(table),
            f"--physics-backend {row.name} carries no point source; the "
            "table would be accepted and never read",
            "a row whose adapter publishes seam_options_for",
        )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from woof.hex.speculative_contract import validate_output
    validate_output(args.output)
    time_ledger.begin_if_closed("forecast.construct", root=args.output)
    time_ledger.emit("prep.host", "begin", root=args.output)
    paths = {
        "grid": Path(args.grid).expanduser().absolute(),
        "static": Path(args.static).expanduser().absolute(),
        "init": Path(args.init).expanduser().absolute(),
    }
    arwen_checkout = proof.resolve_arwen_checkout(args.arwen_checkout)
    # Resolved first: a selection behind the column-suite seam that no
    # profile row names registers its synthesized backend row here.
    physics = resolve_driver_physics(args)
    physics_backend = resolve_physics_backend_row(args)
    proof.CHEMISTRY_CONFIG = (str(args.chemistry_config.resolve())
                              if args.chemistry_config is not None else None)
    proof.CHEMISTRY_CONFIG_SHA256 = (hashlib.sha256(args.chemistry_config.read_bytes()).hexdigest()
                                    if args.chemistry_config is not None else None)

    # This port's own sources and the engine's seam files are hashed and
    # RECORDED here, with whether each matches the x4 proof's pins; neither
    # match is a condition of running (retired 2026-10-10, see
    # proof.measure_execution_sources and engine_pin.identity_of).  The
    # same two measurements are repeated after the run, and a difference is
    # refused there: the receipt would name bytes the run did not use.
    source_before = proof.measure_execution_sources()
    run_manifest = _manifest_for_run(physics_backend, arwen_checkout)
    arwen_git_before = proof.verify_arwen_checkout_git(
        arwen_checkout, manifest=run_manifest, require_committed=False
    )
    authority_before = _timed_preparation("prep.authority_hashes", verify_forecast_authorities, paths)
    lbc_paths: list[str] | None = None
    if args.lbc_dir is not None:
        lbc_dir = Path(args.lbc_dir).expanduser().absolute()
        from woof.hex.lbc_stream import initial_paths
        lbc_paths = [str(path) for path in initial_paths(lbc_dir)]
        if not lbc_paths:
            raise ConfigurationRefusal(
                "config_apply_lbcs",
                str(lbc_dir),
                "--lbc-dir names a directory with no lbc.*.nc in it, so the "
                "run would integrate a limited-area domain against no "
                "boundary series at all",
                "the --out-dir rw_mpas_lbc wrote its boundary files into",
            )
    host = prepare_forecast_host(
        paths,
        authority_before,
        lbc_paths=lbc_paths,
        physics_backend=physics_backend.name,
        source_table=(
            None if args.source_table is None else str(args.source_table)
        ),
        start_time_text=args.start_time,
        horiz_mixing=args.horiz_mixing,
        convection=args.convection,
        pbl_cadence=args.pbl_cadence,
        time_integration_order=args.time_integration_order,
        acoustic_substeps=args.acoustic_substeps,
        radiation_seconds=args.radiation_seconds,
        monotonic=args.monotonic,
        cam_coefficient=args.cam_coefficient,cam_damping_levels=args.cam_damping_levels,
        rayleigh_u=args.rayleigh_u,damping_start_m=args.damping_start_m,
        dynamics_overrides=parse_dynamics_overrides(args.dynamics),
        orographic_statistics=(
            None
            if args.orographic_statistics is None
            else Path(args.orographic_statistics).expanduser().absolute()
        ),
        local_timestep=args.local_timestep,
        local_timestep_declared_off=args.local_timestep_declared_off,
        local_timestep_rates=args.local_timestep_rates,
        local_timestep_buffer_rings=args.local_timestep_buffer_rings,
        invariant_path=(
            None if args.invariant is None
            else Path(args.invariant).expanduser().absolute()
        ),
        physics=physics,
        physics_mode=args.physics_mode,
        physics_correct_skip=tuple(args.physics_correct_skip or ()),
    )
    from woof.hex.stochastic import resolve_options
    if getattr(args, "wind_farms", None) is not None:
        host["wind_farms"] = args.wind_farms
    host["stochastic"] = resolve_options(args.stochastic).json()
    if args.preflight_only or not DEFER_PREPARED_FINGERPRINT:
        join_prepared_fingerprint(host)
    schedule = build_schedule(
        hours=args.hours,
        history_every_minutes=args.history_every_minutes,
        history_from_minutes=args.history_from_minutes,
        start_text=host["start_time_text"],
    )
    time_ledger.emit("prep.host", "end", root=args.output)
    install_capture_labels(schedule["labels"])

    provenance = {
        "schema": SCHEMA,
        "receipt_mode": RECEIPT_MODE,
        "derived_from": DERIVED_FROM,
        "derived_from_sha256": proof.sha256_file(shipped_sources.resolve(DERIVED_FROM)),
        "case_label": args.case_label,
        "init": {
            "path": str(paths["init"]),
            "bytes": authority_before["files"]["init"]["bytes"],
            "sha256": authority_before["files"]["init"]["sha256"],
            "source": args.init_source,
            "invariant": None if args.invariant is None else {
                # The bytes the vertical grid came from, named like the init's.
                "path": str(args.invariant),
                "bytes": Path(args.invariant).stat().st_size,
                "sha256": proof.sha256_file(Path(args.invariant)),
            },
            "coupled_diagnostics": "cold-start-recomputed",
            "config_start_time": host["start_time_text"],
            "pinned": False,
        },
        "mesh": {
            role: authority_before["files"][role] for role in MESH_AUTHORITY_ROLES
        },
        **measured_engine_identity(arwen_checkout),
        "arwen_contract_document_sha256": proof.ARWEN_CONTRACT_DOCUMENT_SHA256,
        "arwen_contract_surface_sha256": proof.ARWEN_CONTRACT_SURFACE_SHA256,
        "profile": proof.PROFILE,
        "source_release": proof.SOURCE_RELEASE,
        "horiz_mixing": args.horiz_mixing,
        # Dynamics options set by name (--dynamics / a namelist the door
        # read); empty when the run holds the configuration's own values.
        "dynamics_overrides": parse_dynamics_overrides(args.dynamics),
        "convection": host["convection"],
        "local_timestep": {
            "enabled": bool(args.local_timestep),
            "declared_off_arm": bool(args.local_timestep_declared_off),
            "rates": list(args.local_timestep_rates),
            "buffer_rings": int(args.local_timestep_buffer_rings),
            "explicit_classing": (
                None
                if args.local_timestep_classing is None
                else {
                    "path": str(args.local_timestep_classing),
                    "sha256": proof.sha256_file(Path(args.local_timestep_classing)),
                    "note": "A/B instrument arm: rates from a file, not from dcEdge",
                }
            ),
            "native_equivalent": False,
            "note": (
                "native MPAS-A v8.4.1 has no local time stepping; with the "
                "option on this run is a declared divergence, and with it off "
                "the executed path is the pinned one"
            ),
        },
        "config_type": type(host["config"]).__name__,
        "dropped_guarantees": list(DROPPED_GUARANTEES),
        "claim": claim_for_run(physics, host["convection"], arwen_checkout),
        "nonclaims": list(NONCLAIMS),
        "weather_plot_policy": "native Rust/WOOF renderer only; q2 ships in the history stream and its weather-field plots go through the same renderer",
    }

    if args.preflight_only:
        source_after = proof.measure_execution_sources()
        authority_after = verify_forecast_authorities(paths)
        arwen_git_after = proof.verify_arwen_checkout_git(arwen_checkout, manifest=run_manifest, require_committed=False)
        if (
            source_after != source_before
            or authority_after != authority_before
            or arwen_git_after != arwen_git_before
        ):
            raise RuntimeError(
                "source, authority, or WOOF bytes changed during preflight"
            )
        print(
            json.dumps(
                {
                    **provenance,
                    "mode": "preflight-only; CUDA not imported",
                    "sources": source_before,
                    "arwen_git": arwen_git_before,
                    "schedule": {
                        key: value
                        for key, value in schedule.items()
                        if key != "labels"
                    },
                    "constructor": host["constructor_receipt"],
                    "scalars": host["scalar_receipt"],
                    "case_pin_relaxation": host["case_pin_relaxation"],
                },
                sort_keys=True,
                default=str,
            )
        )
        return 0

    assert args.cache_root is not None and args.output is not None
    cache_root, output_root = proof.validate_destination(
        args.cache_root, args.output, tuple(paths.values())
    )
    cache_root.mkdir(parents=False)
    output_root.mkdir(parents=False)
    time_ledger.emit("prep.destination", "mark", root=output_root)
    write_start_record(output_root, paths=paths, schedule=schedule)
    started = time.perf_counter()
    forecast = execute_forecast(
        host=host,
        schedule=schedule,
        cache_root=cache_root,
        output_root=output_root,
        arwen_checkout=arwen_checkout,
        source_receipt=source_before,
        authority_receipt=authority_before,
        fingerprint_every=int(args.fingerprint_every),
        stop_on_refusal=bool(args.stop_on_refusal),
        grid_path=paths["grid"],
        park_physics_tier=bool(args.park_physics_tier),
        required_free_bytes=args.required_free_bytes,
        local_timestep_classing_path=args.local_timestep_classing,
        fullstep_lts_options=args.fullstep_lts_options,
        cycling=cycling_options(args, schedule, paths["init"]),
        iau=(None if args.iau_window_seconds is None else {
            "route": args.iau_route, "mode": args.physics_mode,
            "analysis": (None if args.iau_analysis is None else Path(args.iau_analysis).resolve(strict=True)),
            "increment": (None if args.iau_increment is None else Path(args.iau_increment).resolve(strict=True)),
            "background": paths["init"], "window_seconds": float(args.iau_window_seconds)}),
        balance_series=bool(args.balance_series),
    )
    join_prepared_fingerprint(host)
    source_after = proof.measure_execution_sources()
    authority_after = verify_forecast_authorities(paths)
    arwen_git_after = proof.verify_arwen_checkout_git(arwen_checkout, manifest=run_manifest, require_committed=False)
    if (
        source_after != source_before
        or authority_after != authority_before
        or arwen_git_after != arwen_git_before
    ):
        raise RuntimeError("source, authority, or WOOF bytes changed during execution")
    payload = {
        **provenance,
        "status": (
            "truncated_by_model_refusal"
            if forecast.get("refusal")
            else "passed"
        ),
        "arwen_git": {"before": arwen_git_before, "after": arwen_git_after},
        "arwen_checkout_unchanged": True,
        # Which host-preparation results this run computed and which it was
        # served (woof.hex.prep_cache), with each call's key, so a reader can
        # tell a cold run from a warm one and find the stored arrays.
        "host_preparation_cache": prep_cache.receipt(),
        # Whether the dynamics mesh was loaded and validated or served from a
        # verified snapshot of the same grid/static bytes (woof.hex.mesh_snapshot).
        "mesh_snapshot": mesh_snapshot.receipt(),
        # Background replay of the previous run's NVRTC compiles
        # (woof.hex.kernel_prewarm): how many it found or compiled.
        "kernel_prewarm": kernel_prewarm.receipt(),
        "physics_backend": physics_backend.name,
        "source_table": (
            None if args.source_table is None else str(args.source_table)
        ),
        "execution_seconds": time.perf_counter() - started,
        "forecast": forecast,
        "sources_unchanged": True,
        "authorities_unchanged": True,
    }
    payload["payload_sha256"] = proof.canonical_json_sha256(
        json.loads(json.dumps(payload, sort_keys=True, default=str))
    )
    receipt = output_root / RECEIPT_NAME
    proof._write_exclusive_json(
        receipt, json.loads(json.dumps(payload, sort_keys=True, default=str))
    )
    from woof.hex.observation_scorecard import forecast_hook

    forecast_hook(output_root, init=paths["init"], grid=paths["grid"],
                  start=host["start_time_text"])
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt": str(receipt),
                "receipt_sha256": proof.sha256_file(receipt),
                "payload_sha256": payload["payload_sha256"],
                "integration_seconds": forecast["walls"]["integration_seconds"],
                "steps": forecast["walls"]["steps"],
                "history_frames": len(forecast["snapshot_files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
