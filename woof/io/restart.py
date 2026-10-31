"""wrfrst-style full-state restart files with a bit-identical contract.

``write_restart`` serializes every cross-step model field â€” prognostics,
physics/soil/snow surface state, accumulators, held slow tendencies, the
KF driver persistence, and the model clock â€” so that ``restore_restart``
into a freshly prepared process continues the trajectory FP32-bit-exactly
(Phase 4 Task 8 gate: 6 h + restart + 6 h == uninterrupted 12 h on every
state field and accumulator).

Format: NumPy NPZ (an uncompressed zip of raw little-endian ``.npy``
members).  Chosen over NetCDF for exactness and auditability: the ``.npy``
payload is the array's memory bytes, so FP32 values (including negative
zeros, denormals, and NaN payloads) round-trip bit-exactly with no
writer-side dimension/type mapping in between, and any zip tool can audit
the members.  The wrfout NetCDF writer keeps its WRF-tooling role; restart
files are a model-internal exchange, not a product.

Completeness is ENFORCED, not hoped for: ``write_restart`` walks every
``DomainState`` attribute, every ``DomainState._scratch`` slot, and every
``PhysicsDriver`` attribute and raises :class:`RestartManifestError` for
anything not explicitly classified below as serialized, rebuilt, setup, or
infrastructure.  Adding model state without updating this manifest fails
the next restart write (and the CPU manifest tests) loudly.

Classification argument (audit2 restart findings, adjudicated here):

* SERIALIZED â€” read across step boundaries and not reconstructable:
  prognostics (+ Morrison moments and effective radii, which feed the NEXT
  radiation call), ``h_diabatic`` (WRF ``rdu``, Registry.EM_COMMON:1389 â€”
  re-zeroing drops one step of retained heating), the dycore's exported
  advective forcing pair ``rthften``/``rqvften`` (WRF RTHFTEN/RQVFTEN, on
  the schemes that read them â€” same lifecycle as ``h_diabatic``: the
  producer is an RK stage that has not run yet when a resume reaches its
  first cumulus call), the km_opt=2 prognostic
  SGS TKE carrier ``tke`` (WRF ``r``, Registry.EM_COMMON:312 â€” a developed
  turbulence field with no reconstruction route), the live microphysics
  accumulators in scratch (``mp_*``; the driver's diagnostic dataclass aliases
  this canonical set), the KF driver persistence (``cu_*`` scratch,
  W0AVG), the held coupled tendencies (mu-coupled at their historical due
  step â€” recoupling at restore is NOT bit-identical), the surface/Noah
  ``fields`` dict (UST/MOL/ZNT/QSFC/HFX/QFX/PBLH/SH2O/SNOTIME/ALBEDO/EMISS
  and the rest of WRF's r-flagged surface block), ``_pending_rainbl``,
  ``microphysics_updates`` (behavior-gating counter), and the clock.
* REBUILT â€” overwritten before every read: the RK time-t copies (written
  from prognostics at each ``dycore.step`` entry), the slow-tendency slots
  (zeroed each RK stage), the acoustic perturbations (reseeded by
  ``_init_small_steps`` each stage; ``ww_pp`` is checkpoint-only instead),
  and the per-call scratch work buffers (Morrison/Kessler prep, refl,
  advection, diffusion, LBC helpers).  The driver's one-frame
  ``refl_10cm`` handoff is also rebuilt: normal output consumes it before
  any same-step restart write, and a resume never rewrites the boundary
  frame.  The bit-identity gate is the proof of this list: a missed
  cross-step dependence diverges the trajectory.
  The driver ``microphysics`` dataclass is rebuilt as aliases of the
  serialized ``scratch/mp_*`` arrays; v2 files carrying both historical
  copies are accepted only when those copies compare byte-for-byte equal.
* SETUP â€” deterministic from config + ingest (base state, coordinates,
  map factors, LBC tables) or resolved while physics is initialized
  (radiation calendar/grid/gases/ozone, Noah parameters, scheme policies,
  and coefficient assets): rebuilt by the normal preparation path and
  VALIDATED against SHA-256 fingerprints stored in the header, so a restart
  into a different setup fails loudly instead of drifting.  The dynamics
  fingerprint covers the attached lateral-boundary FORCING tables (every
  interval's side values and tendencies, byte-level).  The physics identity
  separately carries explicit versioned algorithm/policy names plus actual
  active-asset byte digests; config IDs alone are not treated as proof of
  identical physics.  The resident LBC device tables are re-uploaded by
  ``attach_lateral_boundaries`` during preparation; ``restore_restart``
  requires that attachment to exist and restores the clock LAST, after
  attach reset it to zero (audit: attach rewinds every physics calendar).
* McICA carries no RNG state: the subcolumn generator's seeds are pure
  functions of the column pressures and the fixed permuteseed
  (kernels/rrtmgp_mcica.cu:35-47, mirrored by
  ``npref.np_mcica_maxran_masks`` â€” proven there against WRF's kissvec),
  so radiation is call-time stateless and nothing is serialized for it.

Cupy is imported lazily so the module (and the CPU manifest/roundtrip
tests) stay importable without a GPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import threading
import uuid
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from woof import perf_timing
from woof.io.parallel_npz import write_npz
from woof.checkpoint_identity import (
    CONFIG_DIAGNOSTIC_FIELDS,
    CUMULUS_ALGORITHM_IDENTITIES,
    LAND_SURFACE_ALGORITHM_IDENTITIES,
    LAND_SURFACE_PARAMETER_SOURCES,
    LONGWAVE_ABOVE_ATMOSPHERE_POLICIES,
    LONGWAVE_ALGORITHM_IDENTITIES,
    MICROPHYSICS_ALGORITHM_IDENTITIES,
    NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES,
    PBL_ALGORITHM_IDENTITIES,
    RADIATION_ABOVE_ATMOSPHERE_POLICIES,
    RADIATION_ALGORITHM_IDENTITIES,
    SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES,
    SHORTWAVE_ALGORITHM_IDENTITIES,
    SURFACE_LAYER_ALGORITHM_IDENTITIES,
    URBAN_ALGORITHM_IDENTITIES,
    DYCORE_MIXING_ALGORITHM_IDENTITY,
    dycore_mixing_identity,
    UPPER_DAMPING_ALGORITHM_IDENTITY,
    upper_damping_identity,
    drop_default_diffusion_selectors,
    drop_default_spp_selectors,
    require_identifiable_checkpoint_schemes,
    unidentifiable_checkpoint_schemes,
)
from woof.config import (CORRECT_MODE_RESTART_BREAK_NOTICE,
                          MIX_ISOTROPIC_RESTART_BREAK_NOTICE,
                          TERRAIN_CLOCK_PINNED_RESTART_NOTICE,
                          TERRAIN_CLOCK_RESTART_BREAK_NOTICE,
                          radiation_scheme_ids)
from woof.supervisor import _fsync_directory, fsync_file, unique_temp_path
from woof.correct_mode import (CORRECTION_KEYS as _CORRECTION_KEYS,
                                OUTPUT_ONLY_KEYS as _OUTPUT_ONLY_CORRECTION_KEYS,
                                key_in_scope as _correct_key_in_scope)
from woof.physics_compat import (RRTMG_VARIANT_LEGACY,
                                  RRTMG_VARIANT_RTE_RRTMGP, rrtmg_variant)
from woof.core.adaptive_clock import ADAPTIVE_DERIVED_RUN_FIELDS
from woof.core.model import (ADAPTIVE_POLICY_RUN_FIELDS,
                              ADAPTIVE_TIMESTEP_RUN_FIELDS)
from woof.core.uh_diag import (
    UH_FOLLOW_WINDOW_PREFIX as _UH_FOLLOW_WINDOW_PREFIX,
    is_tracker_window_slot as _is_tracker_window_slot)
from woof.core.nssl2_contract import (
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    DEFAULT_RESTART_FIELDS as NSSL2_DEFAULT_RESTART_FIELDS,
    WRF_NAMELIST_DEFAULTS as NSSL2_WRF_NAMELIST_DEFAULTS,
    WRF_REFERENCE_COMMIT as NSSL2_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as NSSL2_WRF_REFERENCE_VERSION,
    pinned_zero_fields as nssl2_pinned_zero_fields,
    resolve_nssl2_mode_for_config,
)
#: The dycore's exported advective forcing pair (WRF RTHFTEN/RQVFTEN),
#: named so the reader's key-set refusal can say WHICH change moved the
#: layout and what to do about it instead of printing two sorted lists.
#: Present only on a state whose cu_physics is in
#: ``woof.config.CUMULUS_ADVECTIVE_FORCING_SCHEMES``; every other
#: configuration's checkpoint inventory is untouched by this pair.
#:
#: DEFINED in :mod:`woof.state_serialization_contract` and imported
#: above -- the prepared-cache side of the same tolerance needs it, and
#: that side ships in a wheel that stages no restart reader.  Re-exported
#: here under the name every reader in the tree spells.
from woof.state_serialization_contract import (
    ADVECTIVE_FORCING_STATE,
    BUILT_END_FRAME_PREFIX_SCHEMA,
    CHECKPOINT_ONLY_STATE,
    CHEM_DIAG_PREFIX,
    CHEM_STATE_PREFIX,
    CHEM_TIME_PREFIX,
    CHEM_WORK_PREFIX,
    LATERAL_BOUNDARY_PREFIX_SCHEMAS,
    REBUILT_END_FRAME_PREFIX_SCHEMAS,
    STATE_SERIALIZED_ATTRS,
    STATE_DERIVED_SETUP_ARRAYS,
    STATE_SETUP_ARRAYS,
    STATE_SETUP_SCALARS,
    lateral_boundary_prefix_identity as _lateral_boundary_prefix_identity,
    serialized_state_attrs,
    setup_core_fingerprint as _shared_setup_core_fingerprint,
    setup_fingerprint as _shared_setup_fingerprint,
)

#: Bump on any change to the key layout or classification tables; restores
#: reject unknown versions instead of guessing.  v2 pinned the lateral-
#: boundary forcing tables.  v3 removes duplicate driver/microphysics members;
#: the driver now aliases the serialized scratch/mp_* accumulator set.  v4
#: binds the resolved physics/radiation setup and every active packaged data
#: asset, including an explicit above-atmosphere radiation policy.  v5 adds
#: KF's independently held ice/snow rates and coupled snow tendency.  v6 is the
#: 2.7.0 line: the config echo gains the adaptive-timestep block and
#: ``eta_levels``, every MM5 surface-layer run carries ``fields/ustm``, and
#: every GF / New Tiedtke run carries ``held/gf_*`` -- each of which a v5 file
#: lacks, so a 2.6.5 checkpoint cannot be read by this build and is REFUSED BY
#: NAME (:data:`RETIRED_RESTART_FORMAT_VERSIONS`) instead of by a field-by-field
#: identity mismatch that reads as "your configuration changed" (ENG-010,
#: ENG-011).  The reader retains a byte-equality-checked v2 array-layout shim,
#: but an old unbound v2 file is rejected because the v4 identity header is
#: mandatory. v7 preserves the independent MF_VX_INV input words.
RESTART_FORMAT_VERSION = 7
READABLE_RESTART_FORMAT_VERSIONS = frozenset({2, RESTART_FORMAT_VERSION})

#: Format versions this build recognises and refuses with the reason, so an
#: operator holding a checkpoint from the previous release reads WHY it is
#: refused and what to do, rather than a list of "absent" configuration
#: fields.  A declared break, not a migration: the members a v6 file adds
#: are real state a v5 file never had (see the v6 note above).
RETIRED_RESTART_FORMAT_VERSIONS = {
    6: ("checkpoint format 6 lacks the independently stored MF_VX_INV map inverse; "
        "restart from the initial conditions or finish on the original build"),
    5: ("2.6.5 checkpoint format 5: 2.7.0 adds the adaptive-timestep and "
        "eta_levels configuration echo, the fields/ustm surface-layer member "
        "and the held/gf_rthblten, held/gf_rqvblten cumulus forcing members; "
        "restart from the run's initial conditions or complete it on 2.6.5"),
}


def require_readable_format_version(format_version, path) -> None:
    """Refuse a checkpoint format this build cannot read, by name.

    One gate for the single-domain reader, the tree reader and both
    ``tilestream`` readers, so a retired version is refused with the same
    sentence everywhere it can arrive.
    """
    if format_version in READABLE_RESTART_FORMAT_VERSIONS:
        return
    retired = RETIRED_RESTART_FORMAT_VERSIONS.get(format_version)
    if retired is not None:
        raise RestartMismatchError(f"restart file {path} is a {retired}")
    raise RestartMismatchError(
        f"restart file {path} has format version {format_version!r}; this "
        f"build reads {sorted(READABLE_RESTART_FORMAT_VERSIONS)}")

#: MP18 extends the existing v5 physics-identity object rather than creating
#: another archive format.  This nested contract is independently versioned:
#: an old/aliased MP18 payload cannot be mistaken for the canonical Registry
#: transport even though the outer NPZ layout remains v5.  v2 replaces the
#: hardcoded ``resolved_default_mode`` with the RESOLVED variant mode of the
#: run that wrote the file, plus its transported and absent field lists; a v1
#: payload is refused by version rather than by a confusing whole-dict
#: mismatch.
NSSL2_RESTART_CONTRACT_VERSION = 2
NSSL2_RESTART_PROGNOSTICS = NSSL2_DEFAULT_RESTART_FIELDS
NSSL2_RESTART_AUXILIARY_STATE = ("h_diabatic",)
NSSL2_RESTART_PRECIPITATION_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
    "mp_sr",
)
# The generic Morrison spellings and NSSL's historical Fortran slab argument
# names are deliberately not accepted as checkpoint identities.  Registry
# names above are the sole durable public contract.
NSSL2_LEGACY_RESTART_ALIASES = frozenset({
    "state/nc", "state/nr", "state/ni", "state/ns", "state/ng",
    "state/ccw", "state/crw", "state/cci", "state/csw", "state/chw",
    "state/chl", "state/cn", "state/vhw", "state/vhl",
    "driver/microphysics/rainnc", "driver/microphysics/rainncv",
    "driver/microphysics/snownc", "driver/microphysics/snowncv",
    "driver/microphysics/graupelnc", "driver/microphysics/graupelncv",
    "driver/microphysics/hailnc", "driver/microphysics/hailncv",
    "driver/microphysics/sr",
})

_HEADER_KEY = "__gpuwm_restart_header__"

#: RunConfig fields that may legitimately differ between the writing and
#: the resuming run (forecast length / output & restart cadence).  Every
#: other field must match exactly or the restore fails.
CONFIG_RUN_LENGTH_FIELDS = frozenset({
    "run_seconds", "output_interval_s", "history_interval2_s", "history_interval_change_s", "restart_interval_s"})

# Output-only diagnostic toggles are restart-boundary-adjustable exactly
# like the run-length fields: ``CONFIG_DIAGNOSTIC_FIELDS``, imported above
# from :mod:`woof.checkpoint_identity`, which holds the one table and the
# inertness argument for each member.  The accumulator payloads stay
# tolerant in both directions (missing in file -> zeroed with a note;
# present in file under a diagnostics-off resume -> dropped with a note).

#: An opt-in tree-checkpoint contract for restart-extend orchestration.  It
#: admits exactly one setup change: appending future root LBC intervals after
#: the sealed interval inventory recorded in the checkpoint.  Ordinary
#: restart readers never consult this marker and retain exact setup matching.
SEALED_FORCING_EXTENSION_MODE = "sealed-prefix-v1"
PRESERVED_FORCING_PREFIX_MODE = "preserved-prefix-v1"

#: Root external-LBC clock semantic identity (Davies clock bind,
#: 2026-07-28).  Which dtbc the root's external Davies consumers took is
#: invisible to the config echo, the setup fingerprint (tables only), and
#: the physics identity, yet it changes every downstream trajectory: WRF's
#: post-increment recurrence (dyn_em/solve_em.F:371-372, reset at
#: share/mediation_integrate.F:1522) versus the retired one-step-lagged
#: elapsed-based calculation.  Specified-domain headers therefore record
#: the semantic that INTEGRATED the checkpoint; restores require it to
#: match the resuming state's binding mode, and a header without the key
#: is a pre-epoch file (legacy semantics) that fails closed under bound
#: production code.
ROOT_EXTERNAL_LBC_CLOCK_IDENTITY = "wrf-postincrement-v1"
ROOT_EXTERNAL_LBC_CLOCK_LEGACY = "legacy-elapsed-v0"

#: Versioned semantic identities.  These are deliberately explicit instead
#: of inferred from scheme numbers: a trajectory-changing implementation or
#: policy change must advance its tag, causing an incompatible restart to fail
#: before restore.  Asset bytes and resolved per-run values are bound below.
#: The per-scheme identity TABLES moved to :mod:`woof.checkpoint_identity`
#: and are re-exported below.
PHYSICS_SETUP_SCHEMA_VERSION = 2
PHYSICS_DRIVER_ALGORITHM_IDENTITY = \
    "gpuwm-physics-driver-v8-strict-cache-held-cumulus-history-rates"
#: The scheme identity tables and the plan-review gate that reads them are
#: DEFINED in :mod:`woof.checkpoint_identity` and imported above -- the
#: gate is called from ``woof.config.validate_run_config``, which ships in
#: distributions that stage no ``woof/io`` at all, so the tables cannot
#: live behind this module's import.  Re-exported here under the names
#: every reader in the tree spells; these ARE those objects, not copies.

#: The mp_physics=28 state a restart may NEVER drop.  Every name here is
#: already in ``STATE_SERIALIZED_ATTRS`` and therefore already written by the
#: generic loop; this tuple exists so the *absence* of one is a refusal
#: rather than a silent, finite, wrong resume.  That failure mode is real and
#: specific to this scheme: WRF's terminal clamps (module_mp_thompson.F:
#: 3976-3982) hold nwfa/nifa at their floors and nc at 2/rho rather than
#: raising, so a checkpoint that lost the aerosol state would restore, run,
#: stay bounded, produce no NaN and no health trip -- and integrate a
#: measurably different (aerosol-inert) forecast.  ``nwfa2d``/``nifa2d`` are
#: included because they are cross-step CONSTANTS that nothing in the
#: forecast rewrites: thompson_init derives nwfa2d once at :509-510 and only
#: in the "no initial CCN" branch, so a resume that dropped them would run
#: forever with zero surface aerosol emission.  WRF agrees they belong in the
#: restart stream: Registry.EM_COMMON:492-493 gives QNWFA2D/QNIFA2D the IO
#: string ``i01{17}rhdu``, whose ``r`` is the restart stream.
THOMPSON_AEROSOL_RESTART_STATE = ("nc", "nwfa", "nifa")
THOMPSON_AEROSOL_RESTART_SURFACE_STATE = ("nwfa2d", "nifa2d")

#: The mp_physics=9 state a restart may NEVER drop: Milbrandt-Yau's six
#: hydrometeor masses and their six number moments, minus the qc/qr pair
#: every moist configuration already carries.  Each name is in
#: ``STATE_SERIALIZED_ATTRS`` and is therefore already written by the
#: generic loop; this tuple exists so the ABSENCE of one is a refusal
#: rather than a silent, finite, wrong resume.  That failure mode is
#: specific and real for a two-moment scheme: mass and number enter the
#: size distribution as a ratio, and the scheme rebuilds its mean-mass
#: diameter and slope from whatever pair it is handed, holding both inside
#: the port's own bounds (woof/core/milbrandt2.py's geometry pass).  A
#: checkpoint that lost (say) ``nh`` would therefore restore, run, stay
#: bounded, produce no NaN and no health trip -- and integrate hail on a
#: number concentration the run it claims to continue never had.  WRF
#: agrees these belong in the restart stream: the six numbers are the
#: ``scalar`` package the MILBRANDT2MOM driver arm binds as
#: qnc/qnr/qni/qns/qng/qnh (module_microphysics_driver.F:1857-1862),
#: which solve_em advects and the restart stream carries.
MILBRANDT2_RESTART_STATE = ("qi", "qs", "qg", "qh",
                            "nc", "nr", "ni", "ns", "ng", "nh")

#: The nine precipitation accumulators the mp=9 driver arm binds
#: (module_microphysics_driver.F:1868-1876), spelled as the canonical
#: scratch slots woof keeps them in.  Hail is the pair that makes this
#: list longer than a WSM6-family one, and it is the pair a scheme-blind
#: seven-slot assumption would drop: HAILNC/HAILNCV accumulate for the
#: whole run, so resuming without them restarts hail accumulation at zero
#: while rain and snow continue -- a silently wrong storm-total field.
MILBRANDT2_RESTART_PRECIPITATION_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
    "mp_sr",
)

RRTMGP_TRACE_GAS_POLICY_IDENTITY = \
    "rfmip-experiment-zero-plus-date-policy-and-overrides-v1"
#: Distinct restart identity for the exact port of WRF v4.6.1's bundled
#: legacy RRTMG (RunConfig.ra_rrtmg_variant = "rrtmg_legacy" on the 4/4
#: pair).  Deliberately NOT "rte-rrtmgp-v1": a restart written under one
#: 4/4 implementation must refuse to resume under the other.
#: LW v2 (2.8.5): each stratospheric optical-depth correction of bands 4
#: and 7 is applied once, by the thread that owns its g-point.  v1's
#: parallel band threads raced on those shared outputs, which moved
#: upper-level heating and outgoing longwave, so a v1 checkpoint may not
#: continue under v2.
RRTMG_LEGACY_LW_ALGORITHM_IDENTITY = \
    "wrf-v4.6.1-rrtmg-legacy-lw-v3-owned-gpoint-stratosphere-corrections-micron-inputs"
RRTMG_LEGACY_SW_ALGORITHM_IDENTITY = "wrf-v4.6.1-rrtmg-legacy-sw-v2-micron-inputs"
#: Legacy RRTMG extends the model column with WRF's own Cavallo buffer
#: layers (deltap = 4 mb), like RRTM option 1 but with RRTMG's tables.
RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY = \
    "wrf-v4.6.1-rrtmg-deltap-4mb-buffer-layers-v1"

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
PHYSICS_ASSET_PATHS = {
    "rrtmgp_gas_lw": Path("data/rrtmgp/rrtmgp-gas-lw-g256.nc"),
    "rrtmgp_gas_sw": Path("data/rrtmgp/rrtmgp-gas-sw-g224.nc"),
    "rrtmgp_cloud_lw": Path("data/rrtmgp/rrtmgp-clouds-lw-bnd.nc"),
    "rrtmgp_cloud_sw": Path("data/rrtmgp/rrtmgp-clouds-sw-bnd.nc"),
    # The key keeps its name: it is written into every restart manifest.
    # Since 2.8.0 the bytes RRTMGP reads are the climatology derived from
    # the RFMIP inputs (which no longer ship); see _active_asset_identity.
    "rrtmgp_rfmip": Path("data/rrtmgp/rrtmgp-trace-gas-climatology.json"),
    "wrf_rrtm_data": Path("data/wrf_radiation/RRTM_DATA"),
    "wrf_rrtmg_lw_data": Path("data/wrf_radiation/RRTMG_LW_DATA"),
    "wrf_rrtmg_lw_statics": Path("data/wrf_radiation/rrtmg_lw_statics.npz"),
    "wrf_rrtmg_sw_data": Path("data/wrf_radiation/RRTMG_SW_DATA"),
    "wrf_aer1_climatology": Path("data/wrf_radiation/rrtmg_aer1_climatology.npz"),
    "wrf_aer1_manifest": Path("data/wrf_radiation/rrtmg_aer1_manifest.json"),
    "wrf_ozone_data": Path("data/wrf_radiation/ozone.formatted"),
    "wrf_ozone_lat": Path("data/wrf_radiation/ozone_lat.formatted"),
    "wrf_ozone_plev": Path("data/wrf_radiation/ozone_plev.formatted"),
    "noah_vegparm": Path("data/noah_tables/VEGPARM.TBL"),
    "noah_soilparm": Path("data/noah_tables/SOILPARM.TBL"),
    "noah_genparm": Path("data/noah_tables/GENPARM.TBL"),
    "noah_landuse": Path("data/noah_tables/LANDUSE.TBL"),
    # Noah-MP's three tables, whose bytes woof/core/noahmp_mynn_contract.py
    # already pins against the WRF v4.6.1 tree.
    "noahmp_mptable": Path("data/noahmp/MPTABLE.TBL"),
    "noahmp_soilparm": Path("data/noahmp/SOILPARM.TBL"),
    "noahmp_genparm": Path("data/noahmp/GENPARM.TBL"),
    "kf_lutab": Path("data/kf_lutab/kf_lutab.npz"),
    "satmedmf_manifest": Path("data/satmedmf/manifest.json"),
    "eclipse_besselian": Path("data/wrf_radiation/eclipse_besselian_elements.dat"),
}


def _resolve_physics_asset(relative: Path) -> Path:
    """Where a ``PHYSICS_ASSET_PATHS`` entry's bytes actually live.

    The KEYS and the recorded ``path`` strings above stay ``gpuwm/``-
    relative and unchanged: they are written into every restart
    manifest, and a checkpoint written by one release must stay readable
    by the next.  Only the resolution moved.  Since 2.5.0 the five
    ``data/rrtmgp/`` entries ship in the ``recast-woof-data`` companion
    distribution rather than inside this wheel (see
    :mod:`woof.data_assets`), so joining them onto the package
    directory would fail to stat -- and this function's caller turns an
    unreadable asset into a hard ``RestartManifestError``, which is
    exactly the wrong verdict for "correct file, moved distribution".
    """

    from woof import data_assets

    posix = relative.as_posix()
    if posix.startswith("data/"):
        return data_assets.data_path(posix[len("data/"):])
    return _PACKAGE_DIR / relative

# --------------------------------------------------------------------------
# DomainState attribute classification.
# --------------------------------------------------------------------------

#: Cross-step state serialized under ``state/<name>`` (skipped when the
#: attribute is None for the active configuration).  p/al/alt are
#: recomputed from the prognostics at each step entry, but serializing the
#: end-of-step EOS diagnostics keeps the restored object bit-equal to the
#: live one for any pre-step consumer (e.g. output frames).
#: Overwritten before every read (see the module docstring's argument).
#: The REBUILT attributes below that ``DomainState`` does not allocate:
#: strict WRF arithmetic attaches them at their first write
#: (``woof/core/diagnostics.py`` makes ``p_perturbation`` at the first EOS
#: diagnosis under WOOF_WRF_EXACT_DIAGNOSTICS), so they sit outside
#: ``state_array_shapes`` and the allocation-equals-classification gate in
#: tests/test_preflight.py names them from here.
STRICT_LAZY_STATE_ATTRS = frozenset({"p_perturbation"})

from woof.core.fast_sbm_contract import BIN_SPECIES as _FSBM_BIN_SPECIES

STATE_REBUILT_ATTRS = frozenset({
    "qke_adv0",
    # WOOF_WRF_EXACT_DIAGNOSTICS' stored perturbation pressure: written by
    # update_diagnostics with p/al/alt at every step entry, before the
    # pressure force reads it.  Unclassified, every strict run with the
    # diagnostic control stopped at its first state digest.
    "p_perturbation",
    # RK time-t copies: dycore.step writes them from the prognostics first.
    "u0", "v0", "w0", "thp0", "php0", "mup0",
    "qv0", "qc0", "qr0", "qi0", "qs0", "qg0", "nr0", "ni0", "ns0", "ng0",
    "qh0", "qndrop0", "qnr0", "qni0", "qns0", "qng0", "qnh0",
    "volg0",
    "qnn0", "qvolg0", "qvolh0",
    # mp_physics=9 (Milbrandt-Yau) hail-number RK time-t copy, and
    # mp_physics=16 (WDM6) CCN + droplet-number RK time-t copies.  Written
    # from their prognostics by dycore.step before any reader, exactly
    # like qi0; both were unclassified at 1.9.0, which starved the shared
    # dycore-state workspace of their backings (1.9.1 D1's class).
    "nh0", "nn0",
    # km_opt=2's TKE time-t copy, written from the serialized ``tke``
    # carrier by dycore.step before any reader (core/dycore.py:2186-2187),
    # exactly like thp0 and the moist time-t copies above.  The carrier
    # itself is SERIALIZED (state_serialization_contract.py).
    "tke0", "tke_edmf0",
    # RK time-t copies of transported droplet number.  nc0 exists for the
    # schemes that TRANSPORT nc -- mp=28 (Thompson aerosol-aware), mp=9
    # (Milbrandt-Yau) and mp=16 (WDM6); mp=10 allocates nc but does not
    # transport it and so has no nc0
    # (woof/core/moist.py::THOMPSON_AERO_NUMBER_SPECIES).  nwfa0/nifa0
    # are mp=28's aerosol-number copies.
    "nc0", "nwfa0", "nifa0",
    # mp_physics=50 (P3) rime-mass and rime-volume RK time-t copies.  They
    # exist for the same reason qi0 does and are written from the
    # prognostics by dycore.step before any reader, now that
    # woof/core/moist.py::P3_SPECIES transports the pair.  The carriers
    # themselves (qir/qib) are SERIALIZED, in
    # woof/state_serialization_contract.py.
    "qir0", "qib0",
    # Slow-tendency slots, zeroed at the top of every RK stage.
    "ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t",
    # Acoustic perturbations reseeded by _init_small_steps each stage.
    # ww_pp is checkpoint-only: advance_mu_th skips its forced outer column,
    # so it must retain domain-owned values between acoustic loops.
    "u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp",
    "p_pp", "p_pp_old", "al_pp",
}) | frozenset(n+"0" for n in _FSBM_BIN_SPECIES)

#: Deterministic setup arrays covered by the header fingerprint.
#: Setup scalars folded into the fingerprint alongside the arrays.
#: Machinery: handled by dedicated sections (scratch, physics, clock) or
#: rebuilt by attach/prepare (LBC device mirrors, host caches).
STATE_INFRA_ATTRS = frozenset({
    # Rebuilt from bound analyses and the restored model clock. Held
    # tendencies are separate serialized fields when fgdt exceeds a step.
    "_spectral_nudging_graph", "_spectral_nudging_owner", "_spectral_nudging_interval",
    # A validity marker for the stage-local acoustic mass buffer, not a
    # prognostic. Stage entry rebuilds it; a fresh standalone caller seeds
    # the buffer when the marker is absent. Checkpoints are step-boundary.
    "_wrf_acoustic_muts_initialized",
    # Spectral backing is serialized through its 142 named field views.
    "fast_sbm_bins", "fast_sbm_itimestep",
    # Cached launch descriptors are rebuilt from the live state buffers.
    "_rk_copy_launch", "_rk_zero_launch", "_glue_add_launch",
    # Used only by analysis snapshot producers. The published LBC field
    # names/bytes, not this preparation selector, own forecast forcing and
    # are already covered by setup_fingerprint/lateral prefix identity.
    "_external_scalar_boundary_fields",
    # Config binds speed; restored U/V already use this frame (see restore).
    "_moving_frame_velocity", "_moving_frame_initialized",
    "_scratch", "_scratch_arena", "_phb_host", "_dz_min",
    "_host_setup_state",
    # Strict mode's record of the prepared base theta
    # (woof.core.state.adopt_wrf_theta_carrier): the prepared restore that
    # sets thb = 300 rebuilds it, and setup_fingerprint already binds it.
    "_setup_thb",
    # Rebuilt from specified/nested and spec_zone on DomainState creation.
    # It is health descriptor geometry, not integrated model state.
    "_health_spec_zone",
    # CanonicalStoreState borrows full host carriers and geography instead
    # of owning a resident DomainState. These references and proxy caches
    # hold no independent prognostics: store carriers and setup fingerprints
    # cover their arrays, while dimensions are rebuilt from the validated
    # configuration. Unknown facade attributes still use the ordinary guard.
    "_template_metadata", "_canonical_arrays", "_view_cache",
    "_canonical_store", "_canonical_geography", "_canonical_scalars",
    "_scratch_allocator", "cfg", "nx", "ny", "nz",
    "physics", "lateral_boundaries", "_lateral_boundary_device",
    # Member-local spectral forcing is serialized in its own optional
    # stochastic namespace and header. An off run adds neither.
    "_ensemble_stochastic",
    # Initial surface recipe provenance and duplicate-application guard.
    # Its numerical result is already in the serialized physics fields;
    # continued domains retain those fields and do not reapply the recipe.
    "_ensemble_surface_state",
    # Uniform-wind provenance is also a duplicate-application guard. Its
    # numerical result is in u/v; continued clocks prevent reapplication.
    "_ensemble_initial_wind",
    # Time-zero checkpoint-fill callback idempotence; contains no model state.
    # A resumed member restores its own serialized fields after initialization.
    "_ensemble_checkpoint_fill",
    "elapsed_seconds", "_nest_restart_classification",
    # The domain's ACTIVATION EPOCH in seconds, published beside
    # ``elapsed_seconds`` by woof.core.state.refresh_model_time and from
    # the same tick authority.  Same category, and for the same reason:
    # it is clock-derived, not integrated.  Every restore path rebuilds
    # the clocks and refreshes the model time before any consumer reads
    # it, so serializing it would store a second copy of a number the
    # clock spec already carries -- and a second copy is a second thing
    # that can disagree.
    "domain_start_offset",
    # mp_physics=50 (P3).  WRF's itimestep, which the adapter maintains
    # lazily on the state (woof/core/p3.py, the ``it`` block).  It is a
    # plain int, not an array, and it is CLOCK-DERIVED rather than dropped:
    # the scheme reads only ``it <= 1``, and the adapter seeds a fresh
    # attribute from ``elapsed_seconds`` -- which the header already
    # carries -- so a resumed run does not replay P3's first-step
    # saturation adjustment.  Same category as ``elapsed_seconds`` above.
    "p3_itimestep",
    # mp_physics=50 (P3), the CUDA port's per-process device workspace
    # (woof/core/p3_device.P3Workspace, cached by woof/core/p3.py::apply).
    # INFRA: it holds no value of its own -- every array in it is a
    # DomainState.scratch slot already classified in REBUILT_SCRATCH_SLOTS
    # below -- and woof/core/p3.py rebuilds it whenever the column count
    # or level count differs, so a resumed run allocates a fresh one on its
    # first P3 call.  The concrete breakage its absence caused: the FIRST
    # mp=50 device step grew this attribute on the state, and
    # classify_state_attr refuses any attribute it does not know, so
    # canonical_state_digest() and write_restart() both raised
    # RestartManifestError and no mp=50 forecast could reach its first
    # history frame or write any checkpoint at all.
    "_p3_workspace",

    # woof.core.streaming.STREAMED_SCRATCH_ATTR: {slot: array} pointing at
    # the DOMAIN's scratch arrays while they live in a streaming store rather
    # than on this state.  Infra, not state: it holds no value of its own,
    # every array in it is already covered as a carrier, and it exists only
    # so an external whole-domain write (the UP_HELI_MAX history reset and
    # the two tracker-window resets) can find the arrays the model is
    # actually integrating.  Absent on every resident state.
    "_streamed_scratch",
    # Set by woof.core.streaming.attach on the resident state whose
    # carriers it copied into a pinned host store: the back-reference the
    # history writers read to learn that this state is no longer where this
    # domain's numbers are.  INFRA and not serialized, deliberately -- the
    # mode is a property of the MACHINE a run is on, never of the forecast,
    # and streaming.identity_payload_entry contributes nothing to the
    # restart identity for the same reason: a checkpoint written streamed
    # must resume resident and vice versa, which is the operation that makes
    # the mode worth having.  On restore the resident state is rebuilt
    # unmarked and re-marked by whatever attach the new run performs.
    "_streamed_domain",
    # A streamed domain's store, published on the state by
    # woof.core.streaming.publish_store so that the readers which are not
    # the tile sweep -- the nest coupler above all -- can find the domain's
    # truth.  INFRA and not serialized: it holds the SAME carriers this
    # manifest already walks, so serializing it would write every field
    # twice and a checkpoint's identity would then depend on whether the
    # run happened to be streaming, which
    # woof.core.streaming.identity_payload_entry deliberately refuses.
    "_streamed_store",
    # Set by woof.core.streaming.prepared_tile_state_factory on every tile
    # buffer it builds, so MYNN walks the tile width the buffer is priced
    # at (woof.core.mynn_pbl_scratch.resolve_mynn_tile_column_chunk).
    # INFRA: a buffer is never checkpointed as itself, and the width never
    # changes a bit.  Without it classify_state_attr refused every streamed
    # buffer's inventory (measured on test host D's RTX 5070 Ti, 2026-09-30).
    "_tile_buffer",
    # The resident rank's admitted MYNN workspace width. It affects only
    # column batching and allocation, never forecast or checkpoint bits.
    "_mynn_rank_column_chunk",
    # woof.da.radar_tten.STATE_ATTRIBUTE: a radar latent heating forcing
    # attached for a forced period (woof.da.forecast_heating, the DA leg
    # driver).  INFRA, like boundary tables: external data rebuilt from its
    # windows by whoever attaches it, never a value the model integrated.
    # The concrete breakage its absence caused: classify_state_attr refused
    # the attribute, so a checkpoint written while the forcing was attached
    # raised RestartManifestError.  The other half of the rule lives in the
    # door: resuming INSIDE the forced period is refused there
    # (forecast_heating.resume_plan), because the slot clock restarts at
    # zero on attach and would read window 1 at minute 70.
    "radar_tten_forcing",
    # woof.da.radar_tten.SLAB_ATTRIBUTE: the (j0, i0) of the ranked slab a
    # resident rank holds, set by the door when it attaches a slab forcing.
    # INFRA: placement of a card's slab, never a forecast value.
    "_radar_tten_slab",
    # woof.da.iau.IAU_STATE_ATTRIBUTE: an analysis increment forced over an
    # IAU window (woof.da.analysis_door, the DA leg driver).  INFRA, like
    # the radar forcing above: external data the attaching door rebuilds
    # from its bundle, never a value the model integrated; its time
    # densities are on the model's own clock, so a forcing rebuilt at a
    # resume inside the window adds exactly what the unbroken run adds from
    # there.  The concrete breakage its absence caused (named gap G4 of the
    # WRF analysis door): classify_state_attr refused the attribute,
    # so any checkpoint written inside a window, and the final trajectory
    # digest of every IAU run on the tree route, raised RestartManifestError.
    "_iau_forcing",
    # woof.da.iau.IAU_AFTER_STEP_ATTRIBUTE: the analysis door's post-step
    # acts for a STREAMED root (its dry-mass share of the IAU window and its
    # balance probe, served per compute window by woof.da.iau
    # .IauSweepLease).  INFRA like the forcing above: the door rebuilds it
    # from its bundle at every attach.  Without this entry classify_state_attr
    # refuses the attribute, so every analysis-door checkpoint and the final
    # trajectory digest raise RestartManifestError.
    "_iau_after_step",
    # Strict WRF-file nest start (woof.ingest.wrfinput_nest_initial
    # .finish_strict_nest_start): the file's T a domain's FIRST frame writes
    # as WRF's th_phy_m_t0, read only before the domain's first step, and
    # the flag that the nest already took its first force round trip.  Both
    # are spent before any checkpoint can be written; INFRA like the start
    # inputs above.  Unclassified, every strict two-domain WRF-file run
    # stopped at its first state digest.
    "_wrf_start_history_t",
    "_wrf_start_diagnostics_carried",
    # The history callback's per-frame note that a strict nest frame
    # carries the next force's round trip, and the run config it reads.
    "_wrf_frame_round_trip_due",
    "_wrf_round_trip_run",
    # Strict history T (WRF's th_phy_m_t0) where it parts from t_2; a resume
    # rebuilds it at its first step, so a frame written before that step
    # writes t_2 (wrf.exe's restart carries th_phy_m_t0 itself).
    "_wrf_th_phy_m_t0",
})

# --------------------------------------------------------------------------
# DomainState._scratch slot classification.
# --------------------------------------------------------------------------

#: Persistent read-modify-write scratch: the canonical microphysics
#: accumulators (the kernels update them in place and the driver aliases them)
#: and the KF driver persistence
#: (NCA timers, PRATEC/RAINCV, stored per-column rates, RAINC).
from woof.core.hail import PERSISTENT as HAIL_PERSISTENT

from woof.core.output_diagnostics import PERSISTENT as OUTPUT_PERSISTENT

SERIALIZED_SCRATCH_SLOTS = frozenset({
    *OUTPUT_PERSISTENT,
    *HAIL_PERSISTENT,
    "ltng_ic_flashcount", "ltng_cg_flashcount",
    "ltng_ic_flashrate", "ltng_cg_flashrate", "ltng_ltngacttime",
    "ltng_iccg_in_num", "ltng_iccg_in_den", "ltng_calendar",
    "ltng_lat", "ltng_lon", "ltng_ktop_deep", "ltng_lpi",
    # WRF diff_opt=1 keeps the original thermal reference after restart.
    "diff1_theta_initial",
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_sr", "mp_kessler_sr",
    "mp_hailnc", "mp_hailncv",
    # Correct mode, finding WB1 (woof/core/thompson_cloud_booking.py): the
    # settled-cloud total accumulates for the whole run like RAINNC, so a
    # resume carries it; it exists only in a correct-mode Thompson run.
    "mp_cloudnc", "mp_cloudncv",
    "cu_rainc", "cu_nca", "cu_pratec", "cu_raincv",
    "cu_rthcuten", "cu_rqvcuten", "cu_rqccuten", "cu_rqicuten",
    "cu_rqrcuten", "cu_rqscuten",
    # WRF UP_HELI_MAX (Registry IO "rh02" -- the r is this row).  A running
    # max is a read-modify-write accumulator exactly like the mp_* totals;
    # a checkpoint written before the slot existed restores it zeroed with
    # a note, never a refusal (woof/core/uh_diag.py owns the lifecycle).
    "up_heli_max",
})

#: The WB1 settled-cloud pair (woof/core/thompson_cloud_booking.py), which
#: the physics driver allocates exactly when the run books it; restore drops
#: a stored pair the live run does not book (_apply_validated_restart).
_THOMPSON_CLOUD_BOOKING_SLOTS = frozenset({"mp_cloudnc", "mp_cloudncv"})


def _refd_max_held(config) -> bool:
    """Whether a run with this config echo allocates ``fields/refd_max``.

    The same gate as ``PhysicsDriver.__init__`` (woof/core/physics.py):
    REFD_MAX is a zero-held output plane exactly when neither the NWP nor
    the GSD diagnostic family runs.
    """
    return (int(config.get("nwp_diagnostics", 0) or 0) == 0
            and not int(config.get("gsd_diagnostics", 0) or 0))


#: Physics-driver ``fields`` planes whose PRESENCE is decided by an
#: output-only switch in CONFIG_DIAGNOSTIC_FIELDS, which a resume may flip:
#: name -> (switch, does a config echo allocate it).  Every other plane
#: keeps the exact inventory and shape check.  Named breakage (V2W WoFS,
#: 2026-10-29): a forecast branched from an analysis checkpoint written with
#: nwp_diagnostics = 0 and resumed with 1 (UP_HELI_MAX on) refused
#: "restart surface-field inventory ... extra ['refd_max']", 18 of 18
#: members rc 2.
_DIAGNOSTIC_GATED_DRIVER_FIELDS = {
    "refd_max": ("nwp_diagnostics", _refd_max_held),
}


def _diagnostic_gated_field_changes(stored_fields, header, driver) -> dict:
    """``{name: "dropped" | "initialised"}`` for the gated planes a resume's
    diagnostic switch moved.

    A plane counts as moved only when the checkpoint's own inventory agrees
    with the switch its header records; a file that disagrees with its own
    header is left to the exact inventory check, which refuses it.
    """
    config = header.get("config") or {}
    changes = {}
    for name, (_switch, allocated_by) in _DIAGNOSTIC_GATED_DRIVER_FIELDS.items():
        stored_has = name in stored_fields
        if stored_has == (name in driver.fields):
            continue
        if stored_has != allocated_by(config):
            continue
        changes[name] = "dropped" if stored_has else "initialised"
    return changes


def _diagnostic_gated_note(name, change, header) -> str:
    switch = _DIAGNOSTIC_GATED_DRIVER_FIELDS[name][0]
    written = (header.get("config") or {}).get(switch, 0)
    if change == "dropped":
        return (f"note: checkpoint written with {switch}={written} carries "
                f"fields/{name}, which this run does not hold ({switch} "
                "changed at the restart); dropped")
    return (f"note: checkpoint written with {switch}={written} has no "
            f"fields/{name}, which this run holds ({switch} changed at "
            "the restart); initialised to zero")

#: Cross-step accumulators that are deliberately NOT checkpointed.
#:
#: A THIRD class, because two questions were being answered by one word and
#: they are different questions: "does this survive a checkpoint" and "is this
#: cross-step state that a run must not lose between steps".  ``serialize``
#: says yes to both and ``rebuild`` says no to both; these say NO to the first
#: and YES to the second, and until this class existed they had to be filed
#: ``rebuild`` â€” which is read as "per-call work buffer" by everything that
#: consumes the classification for a purpose other than writing a restart
#: file.
#:
#: MEASURED CONSEQUENCE of the missing class, and the reason it now exists:
#: ``tilestream.physics_inventory.carrier_manifest`` builds the STREAMING
#: carrier set out of this classification (deliberately, so a field added to
#: woof tomorrow is streamed the day it is added).  Reading ``rebuild`` as
#: "not cross-step", it neither gathered nor scattered the two tracker
#: windows, so under a host store they were not in the store at all: each tile
#: buffer accumulated its own window over whatever tiles that buffer served,
#: at that buffer's coordinates, and the domain had no window anywhere.
#: ``tilestream/test_uh_stream.py`` is the gate; its ``carry_windows=False``
#: control reproduces the old inventory and must differ.
#:
#: The storm-following consumer windows (woof/core/uh_diag.py:
#: TRACKER_WINDOW_SLOTS) are the whole membership.  ``carry`` is their class
#: for every consumer of this table: an ORDINARY checkpoint does not write
#: one, so a run with no lifecycle to restore keeps the member set it has
#: always had, byte for byte.
#:
#: A NEST-LIFECYCLE run opts its own windows in per member instead
#: (:func:`write_tree_restart` -> :func:`_opted_in_scratch_manifest`), and
#: that is a writer decision, not a reclassification.  The old posture here
#: was that a window "restarts empty" because a checkpoint cannot know when
#: the consumer will next look; the restart-first ruling overrules it, and
#: the arithmetic says it can: the content is a max-fold since the
#: consumer's last reset, max is associative, commutative and rounding-free
#: on float32, so folding the persisted window and then the rest is bitwise
#: the unbroken fold.  An empty window at resume under-reads the next
#: spawn/retire/follow decision, which is a different trajectory, not a
#: shorter window.
CARRIED_SCRATCH_SLOTS = frozenset({
    "uh_follow_window", "uh_spawn_window",
    # Strict mode only: WRF's grid%muts from the last acoustic loop, which
    # the next step's first pressure diagnosis reads (woof/core/dycore.py
    # WRF_MUTS_SLOT). Checkpoints retain it for exact continuation too.
    "wrf_exact_muts",
    # Strict mode only: the ring of mu_2 WRF's halo holds at the next step's
    # first RK stage (woof/core/dycore.py WRF_MU_HALO_SLOT).
    "wrf_exact_mu_halo",
})

# These dycore carriers are absent in default mode. Rebuilding them from
# rounded prognostics at resume changes the next strict pressure diagnosis
# and boundary fluxes. Keep their carry classification for streaming and
# relocation, while including allocated members in ordinary checkpoints.
CHECKPOINTED_CARRIED_SCRATCH_SLOTS = frozenset({
    "wrf_exact_muts", "wrf_exact_mu_halo",
})

#: The same class, for the window family whose names are GENERATED.
#:
#: A per-domain ``[follow]`` gives each independently-cadenced child its own
#: window on the parent (``uh_diag.follow_window_slot`` ->
#: ``uh_follow_window.dNN``, allocated in woof/runtime.py), because two
#: children evaluating on different cadences must not blind each other by
#: sharing one plane.  Those names cannot be enumerated above: the set
#: depends on which children the experiment declares.
#:
#: MEASURED CONSEQUENCE of their having matched no class: ``classify_scratch
#: _slot`` is TOTAL by construction -- it raises on anything it does not
#: recognise, deliberately, so a new slot cannot be silently dropped from
#: every checkpoint -- and ``_scratch_manifest`` walks the LIVE pool through
#: it.  A run with a per-domain follow and ``restart_interval_s > 0``
#: therefore died at its first checkpoint instant, hours into the forecast,
#: having written no checkpoint at all.  ``carried_scratch_manifest`` walks
#: the same pool, so the streamed carrier set broke identically.
#:
#: Kept as a prefix tuple beside the exact-name set rather than folded into
#: it, on the REBUILT_SCRATCH_PREFIXES pattern, and taken FROM
#: ``uh_diag.UH_FOLLOW_WINDOW_PREFIX`` rather than restated: ``uh_diag
#: .is_tracker_window_slot`` already answers this question for the rest of
#: the model, and the two answering differently is what made this partial.
CARRIED_SCRATCH_PREFIXES = (_UH_FOLLOW_WINDOW_PREFIX,)

#: Output scratch whose last actual producer value is also lifecycle state.
#: Ordinary restart and transport inventories retain their existing classes:
#: streaming may prime a zero transport buffer before any diagnostic exists.
#: A lifecycle checkpoint explicitly opts in the produced held volume, once,
#: while the consumed PhysicsDriver handoff pointer remains rebuilt.
LIFECYCLE_HELD_SCRATCH_SLOTS = frozenset({"refl_10cm"})

#: Driver-manifest members a CHECKPOINT carries and a SWEEP must not.
#:
#: The exact complement of :data:`CARRIED_SCRATCH_SLOTS`, which exists because
#: "what a checkpoint carries" and "what a sweep must carry" are two questions
#: and the answer differs in BOTH directions.  That set is the slots a sweep
#: carries and a checkpoint does not; this one is the slots a checkpoint
#: carries and a sweep does not.  Named here rather than in
#: ``tilestream.physics_inventory`` for the reason stated above: this file is
#: where the classification lives, so its classes stay exhaustive.
#:
#: ``radiation/o33d_grid`` is restart-only for a standalone legacy
#: radiation adapter: it is a host output rebuilt at every radiation call.
#: A tree with nested CAM consumers attaches PhysicsDriver.o3rad instead,
#: under the SAME checkpoint key. That device field is carried by sweeps
#: because children read the parent's last radiation-time value through
#: FORCE. tilestream.physics_inventory keeps it when that owner exists;
#: no uninitialized adapter field is invented during buffer warmup.
RESTART_ONLY_DRIVER_SLOTS = frozenset({
    "radiation/o33d_grid",
})

#: Per-call work buffers overwritten before every read.  ``mp_``/``cu_``
#: rebuilds are EXACT names only â€” a future accumulator slot under those
#: prefixes must be classified explicitly instead of silently dropping.
REBUILT_SCRATCH_SLOTS = frozenset({
    "cl_zero", "output_zero", "gsd_zero", "gsd_uh_use", "gsd_pressure_w", "gsd_temperature", "gsd_density", "gsd_wind10", "gsd_wind80",
    "hail_zero", "hail_temperature", "hail_height", "hail_density",
    "hail_new_mask", "hail_new_duration",
    # Lin overwrites geometry and thermodynamics before each column call.
    "lin_theta", "lin_rho", "lin_pii", "lin_z", "lin_dz", "lin_z8w",
    "diff1_theta_work",
    "mp_th", "mp_rho", "mp_pii", "mp_z", "mp_dz8w", "mp_z8w",
    "mp_tempo_temperature", "mp_tempo_w", "mp_tempo_qb",
    "mp_thompson_temperature",
    "mp_thompson_frozen_reference_density",
    "mp_thompson_frozen_reference_temperature",
    "mp_thompson_rain_reference_density",
    "mp_thompson_snow_melt_marker",
    "mp_thompson_graupel_melt_marker",
    "mp_thompson_snow_velocity_boost",
    # WRF's private classic-graupel number exists only across one
    # output-due Thompson call and is finalized/consumed by REFL_10CM.
    "mp_thompson_graupel_number_shadow",
    # WRF's per-column no_micro flag (module_mp_thompson.F:1646, :2020),
    # mp=8 and mp=28 alike: taken from the entry state at the top of every
    # call and read by that call's terminal apply.  Nothing survives a call.
    "mp_thompson_micro_columns",
    # Classic Thompson's per-call working set (woof/core/thompson_classic.py
    # SCRATCH_* lists): the pre-microphysics theta, the plane-stacked
    # tendencies, working fields and column buffers, and the cell and column
    # flags.  WRF zeroes and recomputes all of them at the top of every
    # column call (module_mp_thompson.F:1650-2002), so nothing survives a
    # call boundary; a restored tendency would be applied twice.
    "mp_thompson_saved_theta",
    "mp_thompson_classic_ten",
    "mp_thompson_classic_work",
    "mp_thompson_classic_colbuf",
    "mp_thompson_classic_flags",
    "mp_thompson_classic_micro",
    "mp_thompson_classic_kmelt",
    # mp_physics=28 (Thompson aerosol-aware).  Listed as EXACT names, not a
    # new "mp_thompson_aero_" prefix, because the prefix rule above exists
    # precisely to stop a future accumulator from being silently dropped --
    # and three of these ARE accumulators.  They are nonetheless rebuilt,
    # not serialized: WRF zeroes ncten/nwfaten/nifaten at the top of every
    # column call (module_mp_thompson.F:1679-1681) and applies them once
    # before returning (:3972-4021), so nothing in them survives a call
    # boundary, let alone a restart.  The entry snapshots are likewise
    # re-frozen from state at every call entry (:1795-1848).  Serializing
    # any of them would be the bug: a restored non-zero tendency would be
    # applied to state a second time.
    "mp_thompson_aero_ncten",
    "mp_thompson_aero_nwfaten",
    "mp_thompson_aero_nifaten",
    "mp_thompson_aero_entry_density",
    "mp_thompson_aero_nwfa_entry_m3",
    "mp_thompson_aero_nifa_entry_m3",
    "mp_thompson_aero_tau1_density",
    "mp_thompson_aero_nwfa_work_m3",
    "mp_thompson_aero_qcten",
    "mp_thompson_aero_ni_entry",
    "mp_thompson_aero_qrten",
    "mp_thompson_aero_nrten",
    "mp_thompson_aero_qiten",
    "mp_thompson_aero_niten",
    "mp_thompson_aero_qvten",
    "mp_thompson_aero_tten",
    "mp_thompson_aero_qsten",
    "mp_thompson_aero_qgten",
    "mp_thompson_aero_ngten",
    "mp_thompson_aero_rc_entry",
    "mp_thompson_aero_nc_entry_m3",
    "mp_thompson_aero_nu_c_entry",
    "mp_thompson_aero_l_qc_entry",
    "mp_thompson_aero_condensation_rate",
    # mp_physics=50 (P3 one-category).  The adapter's three ice diagnostics
    # -- mass-weighted fall speed, mean diameter and bulk rime density --
    # are filled in full by p3_main on every call before the driver reads
    # them back (woof/core/p3.py:1749-1751, registered at
    # woof/core/preflight.py:2334).  Nothing in them survives a call
    # boundary: WRF's own diag_vmi/diag_di/diag_rhopo are ``intent(out)``
    # of p3_main (module_mp_p3.F:1965-1967), zeroed on entry (:2282-2284)
    # and re-diagnosed from the post-update ice state (:4856-4858) on every
    # call.  Serializing them would be the bug it would look like a fix for.
    "p3_vmi", "p3_di", "p3_rhopo",
    # The CUDA port's device companions (2026-08-29).  Same class and the
    # same argument as the three diagnostics above, and stated as EXACT
    # NAMES rather than a "p3_" prefix for the reason the mp=28 block gives:
    # a prefix rule would silently classify a FUTURE P3 accumulator as
    # rebuilt and drop it from every checkpoint.
    #
    # Every one of these is filled before it is read inside a single call:
    # the twelve carriers by p3k_prep/p3k_kloop1 before p3k_kloopmain reads
    # one, the six sedimentation arrays by the zeroing prologue at the top
    # of each sedimentation step, the flag pair by p3k_prep, nc by the
    # nccnst/rho respecification (module_mp_p3.F:2350) and ssat by the
    # adapter (the wrapper's :851) before k_loop_1 diagnoses it.  The
    # precipitation RATES are likewise zeroed at :2270-2271 every call; the
    # ACCUMULATORS they feed (mp_rainnc and friends) are serialized, and
    # they are a different set of names on purpose.
    #
    # P3's real cross-step carriers are th_old and qv_old, and they are
    # DomainState FIELDS this file already serializes -- the diagnosed-ssat
    # branch reads them on the next call (:2325-2337, :5018-5021).  If a
    # future build predicts ssat instead of diagnosing it, ssat becomes a
    # carrier and moves OUT of this set, and the mp=50 contract string
    # below has to move with it.
    "p3_rho", "p3_inv_rho", "p3_qvs", "p3_qvi", "p3_sup", "p3_supi",
    "p3_rhofacr", "p3_rhofaci", "p3_acn", "p3_t", "p3_tmparr1", "p3_qv_cld",
    "p3_sed_v_q", "p3_sed_v_n", "p3_sed_flux_q", "p3_sed_flux_n",
    "p3_sed_flux_qir", "p3_sed_flux_bir",
    "p3_flags", "p3_nc", "p3_ssat", "p3_effc", "p3_effi",
    "p3_prt_liq", "p3_prt_sol",
    "nssl2_driver_state", "nssl2_driver_surface_export",
    "nssl2_driver_ignored_accumulator",
    "nssl2_fused_temperature", "nssl2_primary_ice_target",
    "nssl2_nucond_ss",
    # The DA reflectivity operator's own dry-air density and diagnosed
    # temperature (woof/da/obsop.py:_nssl_reflectivity).  Both are filled
    # in full at the top of one H(x) call â€” rho from 1/alt, T from theta
    # and Exner â€” and consumed by the shared NSSL diagnostic inside that
    # same call, so no restart boundary can fall between the write and the
    # read and neither carries anything across a step.
    "da_nssl_rho", "da_nssl_t",
    "cu_expiring",
    # The UW moist-turbulence PBL's zero plane (woof/core/physics.py
    # _run_uwpbl): assigned zero at every call before the launch reads it as
    # WRF's zero QNC_CURR/WSEDL3D.  Its carried state (the diffusivities,
    # the residual stress, the held cloud fraction) lives in driver.fields,
    # which every checkpoint serializes whole.
    "uwpbl_zero",
})

REBUILT_SCRATCH_PREFIXES = (
    # Every FAST_SBM workspace is rewritten inside a call. Its spectral state is explicit.
    "fsbm_",
    "rk_", "adv_", "smag_", "diff_", "diff6_", "acoustic_", "openbc_",
    "moist_", "pd_", "morr_", "wsm3_", "wsm6_", "wsm6r_", "wdm6_", "mpcol_", "refl_", "physics_", "lbc_",
    "integration_health_", "nest_",
    # The monotonic chem stage's workspace (woof/core/chem_context.py
    # MONO_SCRATCH_SLOTS, rewritten by every launch before it is read) and
    # its zero implicit vertical flux (MONO_IMPLICIT_SLOT, allocated zero
    # and never written, so a rebuilt one is the same zeros).
    "chem_mono_",
    # UP_HELI_MAX per-step work planes (column UH + use_column flags),
    # overwritten by every launch; the accumulator itself is the exact
    # serialized name above, deliberately NOT under this prefix.
    "uh_diag_",
    # Spec-zone ring-guard snapshots live only between the capture and
    # restore inside ONE microphysics.apply call (core/microphysics.py);
    # their contents are dead at any restart boundary.
    "mp_ring_save_",
    # The km_opt=2 TKE budget's own slots (woof/core/tke_budget.py).  The
    # per-term 3-D fields and the mu totals are rewritten inside every step
    # before anything reads them.  The slab accumulator and its step counter
    # ARE carried across steps, but they are a report-only diagnostic window
    # the caller drains and resets -- never trajectory state -- so a resumed
    # run restarts the current window rather than continuing it, and the
    # drained receipt records the step count it actually covered.
    "tke_budget_",
    # MYNN's declared workspace (woof/core/mynn_pbl_scratch.py).  Every one
    # of these is rebuilt inside the call that reads it; the scheme's carried
    # state is the ten 3-D ``fields`` arrays, which are serialized as fields
    # and are deliberately not scratch.
    "mynn_pbl_",
    # Milbrandt-Yau's per-call work volumes (woof/core/milbrandt2.py).
    # Every one of them is written for every cell by milbrandt2_prelim or
    # milbrandt2_geometry at the top of the same apply() that reads it --
    # including my2_de/my2_ide, which Part 3b mutates DURING the call and
    # which the next call rebuilds from pressure and temperature before any
    # kernel touches them.  The scheme's carried state is the twelve
    # prognostic moments and the precipitation accumulators, and those are
    # serialized as fields and mp_* slots, deliberately not under this
    # prefix.
    "my2_",
    # zadvect_implicit (woof/core/ieva.py): the eta mass-flux split and
    # the three column masses, each written in full at the top of the last
    # RK substep (and the split again before the scalars) before anything
    # reads it.  Nothing in them survives a step.
    "ieva_",
)

# --------------------------------------------------------------------------
# PhysicsDriver attribute classification.
# --------------------------------------------------------------------------

#: Held coupled slow tendencies.  Serialized COUPLED, exactly as held: the
#: cumulus/radiation arrays were mu-coupled with total_mu() at their
#: historical due/expiry step, and mu has evolved since â€” recoupling
#: restored rates at restore time is not bitwise identical (audit).
DRIVER_TENDENCY_ATTRS = ("pbl_tendencies", "radiation_tendencies",
                         "cumulus_tendencies")
OPTIONAL_DRIVER_TENDENCY_ATTRS = ("fire_tendencies",)
TENDENCY_COMPONENTS = ("ru", "rv", "rtheta", "rqv", "rqc", "rqr", "rqi",
                       "rqs", "rw")
TENDENCY_REQUIRED_COMPONENTS = ("ru", "rv", "rtheta", "rqv", "rqc")

MICROPHYSICS_COMPONENTS = ("rainnc", "rainncv", "sr", "snownc", "snowncv",
                           "graupelnc", "graupelncv", "hailnc", "hailncv")
MICROPHYSICS_REQUIRED_COMPONENTS = ("rainnc", "rainncv", "sr")

#: Driver attributes serialized into the file/header.
DRIVER_SERIALIZED_ATTRS = frozenset({
    "o3rad",
    "pbl_tendencies", "radiation_tendencies", "cumulus_tendencies",
    "rthratenlw", "rthratensw", "_pending_rainbl",
    "gf_rthblten", "gf_rqvblten", "pbl_raw_rates",
    "microphysics_updates", "call_counts", "ysu_nan_guard_fires",
    "fields",
    "fire", "fire_tendencies", "hydro",
    # THE SURFACE-RADIATION CARRIER CONTRACT
    # (woof/core/radiation_carriers.py).  Serialized, not rebuilt, and
    # the distinction is the whole point: the carrier FIELDS ride the
    # surface inventory above, but WHO wrote them and WHEN cannot be
    # re-derived from the resumed configuration.  A rebuild would hand
    # the resumed run a fresh "unwritten"/"just written" record and the
    # first post-restart surface call would either refuse a healthy run
    # or admit a stale carrier -- both wrong, in opposite directions.
    # Carried in the driver HEADER (a mapping of two scalars per
    # carrier), so no array key moves and the v5 layout is untouched; a
    # checkpoint written before this contract existed has no such
    # mapping and forces a producer refresh instead (_restore_carriers).
    "carriers",
})

#: Driver attributes rebuilt by initialize_physics from config/tables, or
#: aliases of serialized storage: ``rainc``/``cu_nca``/``cu_pratec``/
#: ``cu_raincv``/``cu_rates`` reference the ``cu_*`` scratch slots (data
#: restored in place through the scratch pool, so the aliases stay live),
#: ``sfclay_result`` aliases the ``fields`` arrays (restored in place â€”
#: never rebound â€” so SFCLAY's seven WRF-inout fields stay coupled),
#: active-scheme ``microphysics`` aliases serialized ``mp_*`` scratch
#: (mp=0 rebuilds its all-zero output placeholder), ``tendencies`` is
#: recomposed by every ``compute()``, and ``last_ysu`` is refreshed before
#: any consumer.  ``refl_10cm`` is an
#: ephemeral output handoff: the due microphysics call rebuilds it, output
#: consumes it once, and restart-resume does not reproduce the boundary
#: frame (PROVENANCE.md D2).  The WSM6 SR roundoff limit, ULP count, and
#: minor-loop count are deterministic functions of the resolved
#: ``mp_physics`` and ``dt`` configuration and are rebuilt with the driver.
#: Driver members a CHECKPOINT carries and the resumed run does not
#: rebuild -- the driver's counterpart to
#: :data:`woof.state_serialization_contract.CHECKPOINT_ONLY_STATE`, in
#: its own ``diag/`` key namespace so no existing key set moves.
#:
#: ``olr`` is the whole membership.  It is WRF's top-of-atmosphere upward
#: longwave flux, published into every history frame, and it is written
#: by the LONGWAVE scheme on radiation's own cadence -- which is coarser
#: than the history cadence, so a frame between radiation calls publishes
#: the last computed field.  A resumed run rebuilt the driver's buffer as
#: ZEROS, so every frame it wrote before its first post-restart radiation
#: call published 0 W m-2 where the unbroken run published ~300.
#:
#: It was classified ``rebuild`` deliberately, on the grounds that adding
#: it to the archive "changes the v5 key layout and would reject every
#: checkpoint already on disk, which is not a price a diagnostic nothing
#: consumes gets to charge".  Two halves of that stopped being true:
#:
#: * The PRICE is not the price.  ``acoustic/``
#:   (CHECKPOINT_ONLY_STATE) established that a carrier can have its own
#:   namespace, absent-tolerant on restore -- so nothing already on disk
#:   is rejected and no ``driver/``/``fields/``/``cumulus/`` key set
#:   moves.  This follows it exactly.
#: * NOTHING CONSUMES IT is not the case.  WRF flags its own row ``rh``
#:   (Registry.EM_COMMON:1839) -- restart AND history -- so upstream
#:   carries it, and it lands in every wrfout frame here, which means a
#:   reader consumes it.  A frame reporting 0 W m-2 of outgoing longwave
#:   is not a missing value, it is a wrong one, and a reader has no way
#:   to tell: the same argument this tree already makes to refuse
#:   ``vmax`` under ``sf_sfclay_physics = 0``.
#:
#: MEASURED, and this is the whole reason the class exists: OLR was the
#: ONLY variable that differed across a restart on the tc_lowres
#: tree -- static, moving-nest, every configuration -- out of 77 in every
#: frame.  Carrying it takes a resumed run from 17-of-18 frames
#: byte-identical to 18 of 18.
#:
#: ZERO IS STILL RIGHT BEFORE THE FIRST CALL OF A RUN, which is why this
#: is a carry and not a fill value: a cold-started run's t=0 frame
#: publishes zeros because WRF's does (``misc``, zero-initialised, and
#: the time-0 write precedes the first radiation call).  A restart is not
#: the first call of a run; it is the middle of one.
#:
#: ``cldfra`` joins it on the same measured terms: the radiation cloud
#: fraction is written on radiation's cadence, published into every
#: history frame, and WRF flags its own CLDFRA row ``irh``
#: (Registry.EM_COMMON:1699), restart AND history.  Without the carry a
#: resumed run would publish an all-clear CLDFRA until its first
#: post-restart radiation call.  Absent-tolerant on restore, so every
#: checkpoint written before it existed stays restorable.
DRIVER_CHECKPOINT_ONLY_ATTRS = frozenset({"olr", "cldfra", "radiation_flux_diagnostics",
                                          "aod5503d", "aodtot",
                                          "_wrf_rthraten", "rthraten"})

#: Raw PBL rates read by GF and New Tiedtke between producer calls.
#: WRF Registry RTHBLTEN/RQVBLTEN are restart-carried for the same reason.
#: The shared driver manifest includes them in resident, tile, host-store
#: and checkpoint inventories, with stable storage from construction.
DRIVER_HELD_FORCING_ATTRS = frozenset({"gf_rthblten", "gf_rqvblten"})

DRIVER_REBUILT_ATTRS = frozenset({
    "wind_farm",  # Device tables and diagnostics rebuilt; QKE carriers persist.
    # Startup inventory/geometry decision, rebuilt from configuration and statics.
    # This receipt owns no evolving numerical state.
    "wind_farm_detection",
    # Run-local CUDA events and elapsed-time counters, recreated when hydro
    # attaches to the resumed parent. Scientific hydro state is checkpointed
    # separately; carrying timing events across processes is invalid.
    "hydro_cost",
    # Run-local source provenance; scientific lake fields are serialized.
    "lake_initialization_receipt",
    # User aerosol diagnostics are views of arrays serialized in fields.
    "aerosol2d",
    "_wrf_rthraten_coupled",
    # Lake sparse work is rebuilt from the horizontal arrays in fields.
    "lake",
    # SPP flags/shapes derive from the bound RunConfig. Patterns are borrowed
    # views owned by the stochastic provider, which restores its spectra and
    # rebinds the views before the next physics call. The driver owns no SPP
    # history, and the disabled path must add no checkpoint/carrier arrays.
    "_spp_flags", "_spp_shapes", "spp_patterns",
    "noah_mosaic",  # Rebuilt by the LANDUSEF door; arrays live in fields.
    "cam_ozone",
    # WRF's slope_rad/topo_shading carrier (woof.core.topo_radiation):
    # initialize_physics rebuilds it from the resumed RunConfig, terrain
    # and lat/lon, and every array it holds -- slope, slp_azi, the
    # radiation-time diffuse_frac/topo_coszen/hrang/topo_declin, SWNORM,
    # the shadow mask and ht_shad -- lives in ``fields``, which is
    # serialized and restored in place, so the rebuilt carrier reads the
    # checkpoint's values.  Unclassified, it ended every slope_rad run's
    # forecast at the final state digest (first real run of the port).
    "topo_shortwave",
    # Rebuilt geometry/kernel; all held undimmed arrays alias fields/*,
    # including the initialized marker used to preserve radiation cadence.
    "eclipse",
    # WRF's topo_wind / gwd_opt (woof.core.terrain_drag): initialize_physics
    # rebuilds it from the resumed RunConfig and the prepared statics; it
    # holds only static coefficients and statistics, never a value a step
    # writes, so the rebuilt carrier is the one the checkpoint ran with.
    "terrain_drag",
    # WRF's swint_opt = 1 carrier (woof.core.swint): initialize_physics
    # rebuilds it from the resumed RunConfig, latitude, longitude and start
    # time; it holds the per-step zenith-cosine scratch (a local in WRF's
    # driver) and the kernel handles, and every value a step writes --
    # the fit coefficients, the reference call, SWDDIR/SWDDIF/SWDDNI and
    # the radiation-call albedo -- lives in ``fields``, serialized and
    # restored in place, so the rebuilt carrier evaluates the checkpoint's
    # own fit on the first resumed step.
    "swint",
    # SASE: the active flag and the kernel-module tuple are re-derived
    # from the resumed RunConfig at driver init; the ledger is a
    # per-step diagnostic replaced before any consumer reads it; the
    # flux-diagnostic buffers are output-only and refilled by the first
    # step after the resume.
    "sase_active", "last_sase_ledger", "sase_flux_diag",
    "sase_nan_guard_fires",
    # The adaptive clock's radiation decision
    # (woof.core.adaptive_clock._drive_radiation_on_time).  REBUILT, and
    # it could not be anything else: it is recomputed from elapsed model
    # time before every solve, so a resumed run derives it from its own
    # clock on its first step and a serialized copy would be overwritten
    # before it was ever read.  Absent on every fixed-dt driver -- the
    # attribute only exists once an adaptive controller has set it, and
    # the predicate it feeds treats absent as "decide normally".
    "radiation_due_override",
    # Its cumulus twin, on identical terms: recomputed from elapsed model
    # time before every solve, absent on every fixed-dt driver, and
    # "absent" is what the predicate it feeds reads as "decide normally".
    "cumulus_due_override",
    # Exact adaptive surface/PBL deadline, recomputed before every solve.
    "surface_pbl_due_override",
    # The horizontal eddy-viscosity diagnostic, on the SAME terms as the
    # flux-diagnostic buffers beside it: output-only, never read back by
    # the physics, and refilled by the first step after a resume (the
    # SASE half in the closure slot, the Smagorinsky half at the next
    # output).  A resumed run's first frame therefore carries the
    # post-resume step's viscosity, which is the value that step used.
    "hmix_k_diag",
    # (OLR is NOT here.  It used to be, on the same output-only
    # grounds; see DRIVER_CHECKPOINT_ONLY_ATTRS for why the argument did
    # not survive the measurement.)
    "state", "sfclay_result", "mynn_sfclay_result",
    "mynn_sfclay_sea_result", "noah_params",
    # Reflectivity borrows the resumed RunConfig, already fingerprinted by
    # the checkpoint, and PhysicsDriver.__init__ rebinds this alias.
    "refl_config",
    # Selector-value -> runner-method receipt, re-resolved from the resumed
    # RunConfig by PhysicsDriver.__init__ (the config fingerprint already
    # binds the selector values themselves).
    "scheme_dispatch",
    # Selector-derived history mapping rebuilt by PhysicsDriver.__init__.
    # Its numerical rate buffers live in serialized fields.
    "pbl_history_rates",
    "radiation_callable", "cumulus_callable",
    # Where this domain's downward longwave came from ("scheme",
    # "declared" or "unused").  A LABEL, not state: initialize_physics
    # re-derives it on the resume from the same selectors and the same
    # caller-supplied glw, and nothing in the physics reads it.  The GLW
    # FIELD itself is serialized with the rest of the surface inventory,
    # so a resumed run carries the same numbers whatever this says.
    "glw_provenance",
    # The surface classification receipt (which source decided XLAND,
    # per-class column counts under the active xice_threshold).  A
    # RECEIPT on the same terms as glw_provenance: initialize_physics
    # rebuilds it at every construction from the serialized mask/category
    # fields, and nothing in the physics reads it back.
    "surface_classification",
    "ra_physics", "ra_lw_physics", "ra_sw_physics",
    "radiation_active", "cu_physics", "mp_physics", "surface_enabled",
    # Noah LSM option selectors: plain cfg-derived scalars reconstructed at
    # driver init, exactly like the radiation and cumulus selectors above.
    "noah_usemonalb", "noah_rdlai2d", "noah_opt_thcnd", "noah_myj",
    # seaice_noah's model step and albedo (cfg scalars, rebuilt at init) and
    # its once-per-process snow-density read-back latch: a resumed process
    # reads the carried snowpack back once more, which is the point.
    "noah_seaice_dt", "noah_seaice_albedo_default", "_noah_seaice_checked",
    # Noah-MP: the parsed tables and the solar geometry are rebuilt by
    # initialize_physics from the packaged assets and the caller's
    # start-time/latitude/longitude, and BOTH are bound into the checkpoint
    # header by _land_surface_parameters_identity -- so a resume against
    # different tables or a different date/latitude is refused rather than
    # silently continuing a different trajectory.  The four-layer thickness
    # vector is a module constant.  The per-call column census is a receipt
    # the next call overwrites.
    "noahmp_params", "noahmp_geometry", "noahmp_soil_thickness_m",
    # Prepared fixed-parent geometry is rebound from the executed inputs.
    # Hydro's numerical arrays and continuation live in the hydro manifest.
    "hydro_parent_geometry",
    "last_noahmp_census",
    # RUC: the parsed tables and the nine-level geometry are rebuilt by
    # initialize_physics from the same packaged assets, and the bundle is
    # bound into the checkpoint header by _land_surface_parameters_identity,
    # so a resume against different table bytes is refused.  Unlike Noah-MP,
    # RUC reads no solar geometry at all -- it takes GSW and GLW as forcing --
    # so there is no second identity to bind.  The per-call census is a
    # receipt the next call overwrites.
    "ruc_params", "last_ruc_census",
    # The urban canopy model (sf_urban_physics > 0).  Its ARRAYS are not
    # here: every UrbanState array lives in ``fields`` under its WRF
    # Registry name (woof/core/urban_state.py), so it rides the serialized
    # surface inventory and is restored in place.  What is rebuilt is the
    # view object over those arrays, the parsed URBPARM tables (hash-pinned
    # in woof/core/urban_tables.py), the coupler that calls the model, and
    # the config the DA refresh hands it.  The rural snapshot and the BEP
    # PBL terms are rewritten by every surface call before anything reads
    # them.
    "urban", "urban_coupler", "_urban_cfg", "surface_energy_diag",
    # Whether the last radiation call handed over SWDDIR/SWDDIF itself
    # (option 3).  Rewritten by every radiation call; between calls the
    # split rides the checkpoint in fields, so a resume needs no record of
    # who produced it.
    "_swdd_from_scheme",
    # The opt-in surface-moisture ledger (woof/core/
    # surface_moisture_ledger.py).  A DIAGNOSTIC an operator attaches to
    # one run: it writes no physics field and no restart-carried buffer,
    # and a resumed run attaches its own or does not.  Carrying it would
    # make an instrument part of the model state.
    "surface_moisture_ledger",
    # Set only when a PRE-CONTRACT checkpoint is resumed, and consumed by
    # the first surface call after the resume (which forces a producer
    # refresh).  A cold-started driver has it False, so it is rebuilt
    # rather than carried; see _restore_carriers.
    "carriers_need_producer_refresh",
    "tendencies", "last_ysu", "refl_10cm", "microphysics",
    "nssl2_binding",
    # Dynamics forcing is integrator-owned: the restart-carried state pair
    # on ARW, or buffers refilled before phase 1 on MPAS. Column spacing is
    # static constructor input. Raw PBL forcing is serialized above.
    "gf_rthdynten", "gf_rqvdynten", "gf_dx_column", "gf_active_column",
    # The Shin-Hong passenger-repair advisory latch (task #206): a
    # print-once flag, not state.  Rebuilt False at driver init, so a
    # resumed run that needs the repair says so once again -- an
    # advisory that survives a restart silently would hide the repair
    # from the operator reading the resumed log.
    "_shinhong_passenger_advisory",
    # The Shin-Hong entry-heal advisory latch (the upstream half of the
    # same class): rebuilt False for the same reason -- a resumed run
    # whose carried-in e_sgs violates the shinhonginit floor must say so
    # in ITS log.
    "_shinhong_entry_advisory",
    "bldt_seconds", "stepbl", "radt_minutes", "cudt_minutes",
    "stepra", "stepcu", "radt_seconds", "cudt_seconds",
    "rainc", "cu_nca", "cu_pratec", "cu_raincv", "cu_expiring",
    "_cu_expiry_pending",
    "cu_rates", "_sr_roundoff_upper", "_sr_roundoff_max_ulps",
    "_wsm6_minor_loops",
    # The one-step flag a mid-run driver rebuild (a spawn or a move) sets so
    # the rebuilt driver calls radiation on its first step
    # (woof.runtime.rebuild_child_driver_from_land_state); the physics step
    # clears it.  Rebuilt absent: a resumed driver takes the radiation
    # carriers the checkpoint restores, and an unclassified attribute here
    # stopped every checkpoint of a moving-nest run (2.9.4 R5, Milton).
    "force_radiation_next_step",
})

#: Scheme-callable state classification.  The walk covers DIRECT array
#: attributes, every value of dict-valued attributes, and ONE level of
#: object-container attributes; anything array-bearing outside these
#: allowlists fails the write.  Deeper nesting is out of walk scope by
#: design â€” a container holding arrays must itself be classified here.
#:
#: Arrays: the cumulus adapter's W0AVG is restart state (WRF
#: Registry.EM_COMMON:1575 r-flags it); the radiation constants are
#: setup-time (lat/lon grids, ozone climatology).
CUMULUS_CALLABLE_ARRAYS = frozenset({"w0avg"})
# G3's private column arena is overwritten before use on every call. Its
# diagnostic views belong to driver.fields and do not appear in __dict__.
CUMULUS_REBUILT_ARRAYS = frozenset({"_ws"})
RADIATION_CALLABLE_ARRAYS = frozenset({
    "latitude_deg", "longitude_deg", "_ozone_logp", "_ozone_vmr",
    # Legacy-RRTMG adapter: it retains no latitude-interpolated ozone
    # (each call interpolates from the live latitude). _o33d_grid is
    # SERIALIZED state -- WRF's O3RAD is a restart-carried field (rdf),
    # and a child domain's first post-restore radiation call consumes
    # the parent's retained o33d BEFORE the parent's next radiation
    # cadence tick, so rebuild-on-resume would orphan it (and break
    # resumed-vs-uninterrupted bit equality).  ``_o33d_host`` and
    # ``_o33d_device`` are that one field's storage behind the
    # ``_o33d_grid`` property (host copy made on first read).
    "_o33d_grid", "_o33d_host", "_o33d_device",
    # Per-call scratch: the reused host buffer the CAM ozone time
    # interpolation fills at every radiation call, rebuilt by the next one.
    "_ozmixt_buf"})
#: Containers CLASSIFIED as acceptable, deliberately (review F2 â€” no
#: silent blind spots): the RRTMGP gas/cloud table objects are
#: rebuild-on-load (module-level ``lru_cache`` loads of packaged
#: k-distribution/cloud-optics data â€” deterministic, never mutated per
#: call; their lazy ``_device`` mirrors likewise), and the KF adapter's
#: ``_history_state`` is a back-reference to the DomainState itself,
#: whose arrays the state walk already covers.
RADIATION_CALLABLE_CONTAINERS = frozenset({
    "lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables",
    "chunk_workspace",
    # Legacy-RRTMG adapter containers, all rebuild-on-load: _C (the LW
    # coefficient dict) and _sw_tables/_cuda_sw/_ozone_climo are
    # digest-checked deterministic loads of packaged assets performed at
    # construction (never mutated per call); _night_outputs is a
    # per-radiation-call product fully rebuilt before every consumption.
    "_C", "_sw_tables", "_cuda_sw", "_ozone_climo", "_night_outputs",
    # _ozone is the woof.ingest.wrf_ozone MODULE reference (its globals
    # include cached climatology arrays); modules are code, not state.
    "_ozone",
    # The RRTM+Dudhia (1/1) composition's two single-stream adapters
    # (woof/core/rrtm_lw.py::RRTMDudhiaRadiation), both rebuild-on-load:
    # a resumed run constructs a fresh composed adapter, whose
    # __post_init__ rebuilds both from the same constructor arguments the
    # radiation identity fingerprints.  The only arrays reachable through
    # them are the construction-time latitude/longitude device grids --
    # the same data as the composed adapter's own classified
    # latitude_deg/longitude_deg -- and the longwave adapter's ``_tables``
    # (digest-stable packaged RRTM coefficient loads, never mutated per
    # call).  Neither adapter holds cross-step array state: the held
    # radiative rates live on the driver (rthratenlw/rthratensw,
    # serialized by name), and RRTM's ozone is recomputed from packaged
    # O3DATA every call, unlike legacy RRTMG's restart-carried _o33d_grid.
    "longwave_adapter", "shortwave_adapter",
    # The RRTM longwave adapter's coefficient bundle (_tables): a
    # digest-stable load of the packaged module_ra_rrtm.F data performed
    # at construction and never mutated per call -- rebuild-on-load, the
    # same category as legacy RRTMG's _C.  Classified so a directly
    # registered RRTMLongwaveRadiation (any custom longwave/shortwave
    # composition) walks cleanly, not only the shipped 1/1 pair.
    "_tables"})
#: Cumulus containers, both back-references rather than state of their own:
#: the KF adapter's ``_history_state`` is the DomainState (arrays covered by
#: the state walk), and the GF adapter's ``_driver`` is the PhysicsDriver
#: THIS FUNCTION is walking -- ``GrellFreitas.bind_driver`` stores it so the
#: adapter can read the held radiative rates, and the only arrays reachable
#: through it (``rthratenlw``/``rthratensw``) are serialized by name a few
#: lines below.  It is rebuild-on-load in the strict sense: a resumed run
#: constructs a fresh adapter with ``_driver = None`` and
#: ``PhysicsDriver._run_cumulus`` rebinds it before the adapter can read it,
#: so nothing crosses the checkpoint through this attribute.
#:
#: Nothing else on the GF adapter needs classifying because there is nothing
#: else: ``cu_physics=3`` is a stateless Task-1 callable (woof/core/gf.py --
#: no NCA persistence, no trigger history, no closure memory), and every
#: quantity of GF's that DOES have memory between calls lives on the driver
#: and is already serialized there -- the held rates through ``cu_rates``'s
#: ``cu_rthcuten``/``cu_rqvcuten``/``cu_rqccuten``/``cu_rqicuten`` scratch
#: slots, and the precipitation accumulators through ``cu_rainc``/
#: ``cu_raincv``/``cu_pratec`` (SERIALIZED_SCRATCH_SLOTS).  A GF adapter that
#: ever grows real state must be serialized here, not added to this set.
CUMULUS_CALLABLE_CONTAINERS = frozenset({"_history_state", "_driver"})


class RestartManifestError(RuntimeError):
    """Model state exists that the restart manifest does not classify."""


class RestartMismatchError(ValueError):
    """The restart file does not match the resuming configuration/setup."""
    exit_code = 2


def producer_identity() -> dict[str, str]:
    """Which build wrote this file.

    A checkpoint or a history file that has been separated from the run's
    logs -- archived, mailed, or found in a directory a year later -- can
    otherwise say nothing about the code that produced it.  The version is
    the installed distribution's, not a hand-maintained constant, because
    the hand-maintained constant is exactly what went stale for four
    releases; and the restart format version rides along because a reader
    that cannot parse the payload still needs to know why.
    """
    from woof import DISTRIBUTION_NAME, __version__

    return {
        "distribution": DISTRIBUTION_NAME,
        "version": __version__,
        "restart_format_version": str(RESTART_FORMAT_VERSION),
    }


#: The header key carrying which memory road WROTE a checkpoint.
#: Provenance, never identity: see :func:`written_mode_note`.
WRITTEN_MODE_HEADER_KEY = "written_mode"

#: The two roads, in the words the note and the resume disclosure use.
RESIDENT_WRITTEN_MODE = "resident"
STREAMED_WRITTEN_MODE = "streamed"


def written_mode_note(mode: str, cfg, *, store: str | None = None) -> dict:
    """Which memory road wrote this checkpoint, and at what shape.

    RESTART IDENTITY VERSUS RESTART PROVENANCE.  The combination restart
    x memory mode is free by construction: streaming contributes nothing
    to the restart identity (``core.streaming.identity_payload_entry``),
    so a file written streamed resumes resident and one written resident
    resumes streamed, and this note may never change that.  It does not:
    nothing in this value is read by :func:`setup_fingerprint`,
    :func:`physics_setup_identity`, :func:`_require_config_match` (which
    compares ``header["config"]`` alone) or by
    ``woof.state_digest.canonical_state_digest`` and
    ``canonical_store_digest``, neither of which reads a header at all;
    they walk a ``DomainState`` and a store.  So the digest of the same
    weather is the same number whichever road wrote it, and a checkpoint
    written before this key existed resumes exactly as it did.

    What the note is FOR is the question an operator cannot otherwise
    answer from the file: a run that was killed for exhausting host RAM
    leaves a checkpoint that says which road it was on when it died.
    Without it the only mode anything can report is the mode of the run
    doing the READING, which is the question nobody asked.

    ``store`` is the road's backing store where that is a real choice
    (``"host"`` pinned RAM or ``"device"`` VRAM for a streamed write) and
    ``None`` where it is not.  The shape is the domain's, taken from
    ``cfg`` so every writer spells it the same way.

    WHO STAMPS IT TODAY.  The resident writer (:func:`_write_restart`)
    does, and it is the only writer in this package.  The streamed
    writer builds its own header in
    ``tilestream.restart_stream.write_streamed_restart`` and does not
    call this yet, so a streamed archive carries no stamp and
    :func:`header_written_mode` answers ``None`` for it.  That asymmetry
    is safe by construction: both readers check for MISSING required
    keys and neither rejects an extra one (:func:`_validate_restart` and
    ``tilestream.restart_stream.validate_streamed_restart``), so it
    costs a disclosure, never a resume.  ``streamed`` is defined here
    rather than left for later so that the streamed writer, when it is
    reached, stamps the same key with the same words through this one
    function instead of inventing a second spelling.
    """
    if mode not in (RESIDENT_WRITTEN_MODE, STREAMED_WRITTEN_MODE):
        raise ValueError(
            f"written mode must be {RESIDENT_WRITTEN_MODE!r} or "
            f"{STREAMED_WRITTEN_MODE!r}, not {mode!r}")
    note = {"mode": mode, "shape": [int(cfg.ny), int(cfg.nx)]}
    if store is not None:
        note["store"] = str(store)
    return note


def header_written_mode(header) -> str | None:
    """The road named in ``header``, or ``None`` when it names none.

    ``None`` is a fact about the FILE and never a reason to refuse: a
    checkpoint written before the stamp existed reports it, so does one
    written by the streamed writer, which does not stamp yet (see
    :func:`written_mode_note`), and every one of them still resumes.  So
    ``None`` may be read as "this file does not say", never as "this
    file was written resident".
    """
    if not isinstance(header, dict):
        return None
    note = header.get(WRITTEN_MODE_HEADER_KEY)
    if not isinstance(note, dict):
        return None
    mode = note.get("mode")
    return mode if isinstance(mode, str) and mode else None


def _admissible_elapsed_seconds(value, where: str) -> float:
    """Model-clock seconds that arithmetic can survive.

    ``float(value)`` alone admits ``NaN`` -- Python's json writes and reads
    the bare token by default -- and admits a negative clock, and both pass
    every identity check a restart makes before poisoning cadence and
    resume arithmetic downstream of them.  MP18 already refused both; this
    is the same refusal for every format and both directions.
    """
    # ``bool`` is an ``int`` and ``float("600")`` succeeds, so neither a
    # flag nor a numeric string may pass as a clock: the header field is
    # written as a JSON number and anything else is a malformed header.
    if isinstance(value, bool) or not isinstance(value, (int, float,
                                                         np.integer,
                                                         np.floating)):
        raise RestartManifestError(
            f"{where} elapsed_seconds must be a real number, got {value!r}")
    try:
        elapsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"{where} elapsed_seconds must be a real number, "
            f"got {value!r}") from exc
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RestartManifestError(
            f"{where} elapsed_seconds must be finite and non-negative, "
            f"got {elapsed!r}")
    return elapsed


def _nssl2_restart_contract_identity(cfg) -> dict[str, object]:
    """Return the exact versioned MP18 state carried by restart v5.

    ``resolved_mode`` describes THIS run, not the shipped default lane: a
    hail-off or diagnosed-CCN run resolves a different mode, keeps a
    different set of fields absent, and must not be handed a receipt
    describing someone else's configuration.  ``absent_fields`` names the
    Registry fields the variant pins to exact zero -- they are still
    written to the archive (the state allocates them), and this is the
    statement that their zeros are the contract rather than an accident.
    """
    mode = resolve_nssl2_mode_for_config(cfg)
    return {
        "schema_version": NSSL2_RESTART_CONTRACT_VERSION,
        "physics_contract_id": NSSL2_CONTRACT_ID,
        "wrf_reference": {
            "version": NSSL2_WRF_REFERENCE_VERSION,
            "commit": NSSL2_WRF_REFERENCE_COMMIT,
        },
        "resolved_mode": dataclasses.asdict(mode),
        "transported_fields": list(mode.transported_fields),
        "absent_fields": list(nssl2_pinned_zero_fields(mode)),
        "resolved_wrf_namelist_defaults": dict(NSSL2_WRF_NAMELIST_DEFAULTS),
        "state_members": [
            *(f"state/{name}" for name in NSSL2_RESTART_PROGNOSTICS),
            *(f"state/{name}" for name in NSSL2_RESTART_AUXILIARY_STATE),
        ],
        "precipitation_members": [
            f"scratch/{slot}" for slot in NSSL2_RESTART_PRECIPITATION_SLOTS
        ],
        "first_call_authority": "driver.microphysics_updates == 0",
        "clock_authority": "header.elapsed_seconds",
        "continuation_policy": "bitwise",
    }


def _require_nssl2_array(value, shape: tuple[int, ...], label: str) -> None:
    if value is None or not _is_array_like(value):
        raise RestartManifestError(
            f"MP18 restart requires array {label!r}")
    if tuple(value.shape) != shape:
        raise RestartManifestError(
            f"MP18 restart {label!r} has shape {tuple(value.shape)}, "
            f"expected {shape}")
    if np.dtype(value.dtype) != np.dtype(np.float32):
        raise RestartManifestError(
            f"MP18 restart {label!r} has dtype {value.dtype}, expected "
            "float32")


def _validate_thompson_aerosol_live_restart_state(state, cfg) -> None:
    """Fail an mp=28 WRITE that would omit any aerosol state.

    The generic writer loop already picks these up through
    ``STATE_SERIALIZED_ATTRS`` when they exist; this refuses the write when
    they DO NOT.  Without it, a build whose ``DomainState`` stopped
    allocating (say) ``nwfa2d`` would produce a checkpoint that both the
    writer and the reader consider internally consistent -- the reader's
    inventory check compares the file against the *resuming* state, so two
    equally aerosol-less endpoints agree -- and the resumed run would
    integrate with zero surface aerosol emission forever.  Adding the 28
    identity string without this guard is exactly the "restart proceeds
    while silently dropping the aerosol state" outcome that is worse than
    the previous outright refusal.
    """
    if int(cfg.mp_physics) != 28:
        return

    volume_shape = tuple(state.p.shape)
    surface_shape = tuple(state.mup.shape)
    for name, shape in (
            *((name, volume_shape) for name in
              THOMPSON_AEROSOL_RESTART_STATE),
            *((name, surface_shape) for name in
              THOMPSON_AEROSOL_RESTART_SURFACE_STATE)):
        value = getattr(state, name, None)
        if value is None or not _is_array_like(value):
            raise RestartManifestError(
                f"mp_physics=28 restart requires array 'state/{name}' "
                "(Thompson aerosol-aware carries prognostic nc plus the "
                "nwfa/nifa tracers and their nwfa2d/nifa2d surface "
                "emission); refusing to write a checkpoint that would "
                "resume with an aerosol-inert column")
        if tuple(value.shape) != shape:
            raise RestartManifestError(
                f"mp_physics=28 restart 'state/{name}' has shape "
                f"{tuple(value.shape)}, expected {shape}")
        if np.dtype(value.dtype) != np.dtype(np.float32):
            raise RestartManifestError(
                f"mp_physics=28 restart 'state/{name}' has dtype "
                f"{value.dtype}, expected float32")


def _validate_thompson_aerosol_stored_restart_state(
        stored: dict[str, np.ndarray], state, cfg, path) -> None:
    """Reject an mp=28 restart FILE that omits any aerosol state."""
    if int(cfg.mp_physics) != 28:
        return

    required = {
        f"state/{name}" for name in (
            *THOMPSON_AEROSOL_RESTART_STATE,
            *THOMPSON_AEROSOL_RESTART_SURFACE_STATE)
    }
    stored_state = {key for key in stored if key.startswith("state/")}
    missing = sorted(required - stored_state)
    if missing:
        raise RestartMismatchError(
            f"restart file {path} omits canonical mp_physics=28 aerosol "
            f"state {missing}; resuming would silently integrate an "
            "aerosol-inert Thompson column (the terminal clamps at "
            "module_mp_thompson.F:3976-3982 keep it finite and bounded, so "
            "nothing downstream would notice)")
    for key in sorted(required):
        name = key[len("state/"):]
        target = getattr(state, name, None)
        if target is None:
            # The RESUMING model has no slot for a field the file carries.
            # A DomainState built from an mp=28 RunConfig always allocates
            # all five (woof/core/state.py's mp==28 arm), so reaching this
            # means the two ends disagree about what mp=28 is -- report it
            # here rather than raising AttributeError from _check_array.
            raise RestartMismatchError(
                f"restart file {path} carries {key} but this build's "
                f"mp_physics=28 DomainState has no {name!r}")
        _check_array(stored[key], target, key)


def _validate_milbrandt2_live_restart_state(state, cfg) -> None:
    """Fail an mp=9 WRITE that would omit any Milbrandt-Yau moment.

    The generic writer loop picks these up through
    ``STATE_SERIALIZED_ATTRS`` when they exist; this refuses the write when
    they DO NOT, for the reason the mp=28 sibling gives: the reader's
    inventory check compares the file against the RESUMING state, so two
    equally moment-less endpoints agree with each other and the resumed run
    integrates a two-moment scheme on a moment it never had.  The
    precipitation slots are checked in the same pass because hail is the
    pair a seven-slot (WSM6-family) assumption drops, and a dropped hail
    accumulator restarts storm-total hail at zero while rain and snow
    continue.
    """
    if int(cfg.mp_physics) != 9:
        return

    volume_shape = tuple(state.p.shape)
    for name in MILBRANDT2_RESTART_STATE:
        value = getattr(state, name, None)
        if value is None or not _is_array_like(value):
            raise RestartManifestError(
                f"mp_physics=9 restart requires array 'state/{name}' "
                "(Milbrandt-Yau carries six hydrometeor masses and a "
                "number moment for each); refusing to write a checkpoint "
                "that would resume with a reconstituted moment")
        if tuple(value.shape) != volume_shape:
            raise RestartManifestError(
                f"mp_physics=9 restart 'state/{name}' has shape "
                f"{tuple(value.shape)}, expected {volume_shape}")
        if np.dtype(value.dtype) != np.dtype(np.float32):
            raise RestartManifestError(
                f"mp_physics=9 restart 'state/{name}' has dtype "
                f"{value.dtype}, expected float32")
    scratch = getattr(state, "_scratch", {}) or {}
    missing = [slot for slot in MILBRANDT2_RESTART_PRECIPITATION_SLOTS
               if scratch.get(slot) is None]
    if missing and len(missing) != len(
            MILBRANDT2_RESTART_PRECIPITATION_SLOTS):
        # ALL-ABSENT is the pre-first-call state and is written as such;
        # a PARTIAL set is the defect this refuses -- it means something
        # allocated the WSM6-family seven and left hail behind.
        raise RestartManifestError(
            f"mp_physics=9 restart is missing precipitation accumulators "
            f"{sorted(missing)} while carrying the rest; the mp=9 driver "
            "arm binds all nine (rain, snow, graupel, hail and SR), so a "
            "checkpoint with only some of them resumes with a storm total "
            "that restarts at zero")


def _validate_milbrandt2_stored_restart_state(
        stored: dict[str, np.ndarray], state, cfg, path) -> None:
    """Reject an mp=9 restart FILE that omits a moment or half the totals.

    Symmetric with :func:`_validate_milbrandt2_live_restart_state`, and
    deliberately so on BOTH halves.  The moment half is obvious.  The
    accumulator half exists because the generic reader's answer to a
    ``scratch/`` slot a file does not carry is to restore it
    zero-initialized with a note -- correct for a slot ADDED after the
    file was written (``up_heli_max``), and silently wrong for a file that
    dropped hail out of a nine-slot row: storm-total hail would resume at
    zero while rain and snow continued from their real totals, which is
    the exact breakage the write-side refusal names.  No gpuwm-written
    file can take that shape; a hand-edited or truncated one can, and this
    is where it is caught.  ALL-absent stays readable, matching the write
    side: that is the pre-first-call state, not a dropped slot.
    """
    if int(cfg.mp_physics) != 9:
        return

    slots = {f"scratch/{slot}"
             for slot in MILBRANDT2_RESTART_PRECIPITATION_SLOTS}
    present = slots & set(stored)
    if present and present != slots:
        raise RestartMismatchError(
            f"restart file {path} carries mp_physics=9 precipitation "
            f"accumulators {sorted(present)} but omits "
            f"{sorted(slots - present)}; the mp=9 driver arm binds all "
            "nine, and the missing ones would be restored zero-initialized"
            " -- resuming with storm-total hail back at zero while rain "
            "and snow continue from the totals this file does carry")

    required = {f"state/{name}" for name in MILBRANDT2_RESTART_STATE}
    stored_state = {key for key in stored if key.startswith("state/")}
    missing = sorted(required - stored_state)
    if missing:
        raise RestartMismatchError(
            f"restart file {path} omits canonical mp_physics=9 state "
            f"{missing}; resuming would integrate Milbrandt-Yau with a "
            "moment rebuilt from its bounds instead of the one the run "
            "that wrote this file carried, and nothing downstream would "
            "notice because the reconstituted value is finite and in "
            "range")
    for key in sorted(required):
        name = key[len("state/"):]
        target = getattr(state, name, None)
        if target is None:
            # The RESUMING model has no slot for a field the file carries.
            # A DomainState built from an mp=9 RunConfig always allocates
            # all ten (woof/core/state.py's mp==9 arm), so reaching this
            # means the two ends disagree about what mp=9 is.
            raise RestartMismatchError(
                f"restart file {path} carries {key} but this build's "
                f"mp_physics=9 DomainState has no {name!r}")
        _check_array(stored[key], target, key)


def _validate_nssl2_live_restart_state(state, cfg) -> None:
    """Fail a write unless every persistent MP18 value is canonical."""
    if int(cfg.mp_physics) != 18:
        return

    shape = tuple(state.p.shape)
    for name in (*NSSL2_RESTART_PROGNOSTICS,
                 *NSSL2_RESTART_AUXILIARY_STATE):
        _require_nssl2_array(getattr(state, name, None), shape, f"state/{name}")

    driver = getattr(state, "physics", None)
    if driver is None or int(getattr(driver, "mp_physics", -1)) != 18:
        raise RestartManifestError(
            "MP18 restart requires an attached MP18 PhysicsDriver so "
            "precipitation and first-call state cannot be omitted")
    pool = getattr(state, "_scratch", {})
    surface_shape = tuple(state.mup.shape)
    missing = [
        slot for slot in NSSL2_RESTART_PRECIPITATION_SLOTS
        if slot not in pool
    ]
    if missing:
        raise RestartManifestError(
            f"MP18 restart lacks persistent precipitation slots {missing}")
    for slot in NSSL2_RESTART_PRECIPITATION_SLOTS:
        _require_nssl2_array(pool[slot], surface_shape, f"scratch/{slot}")
    unexpected_mp = sorted(
        slot for slot in pool
        if (slot.startswith("mp_")
            and classify_scratch_slot(slot) == "serialize"
            and slot not in NSSL2_RESTART_PRECIPITATION_SLOTS))
    if unexpected_mp:
        raise RestartManifestError(
            "MP18 restart has noncanonical persistent microphysics slots "
            f"{unexpected_mp}")

    updates = getattr(driver, "microphysics_updates", None)
    if (isinstance(updates, bool) or not isinstance(updates, int)
            or updates < 0):
        raise RestartManifestError(
            "MP18 restart first-call authority microphysics_updates must be "
            f"a non-negative integer, got {updates!r}")
    try:
        _admissible_elapsed_seconds(state.elapsed_seconds, "MP18 restart")
    except RestartManifestError:
        raise RestartManifestError(
            "MP18 restart elapsed_seconds must be finite and non-negative") \
            from None


def restart_filename(valid_time: datetime, domain: str = "d01") -> str:
    """WRF ``wrfrst``-style file name for a restart valid time.

    Whole seconds only, refused rather than truncated, for the reason
    ``woof.io.wrfout.wrfout_filename`` gives: this name is the standalone
    checkpoint's whole identity and its publisher replaces, so two legal
    sub-second checkpoints used to collapse onto one file and the earlier
    one ceased to exist.
    """
    if valid_time.microsecond:
        raise ValueError(
            f"restart valid time {valid_time!r} is not on a whole second; "
            "checkpoint filenames carry whole seconds only, so distinct "
            "sub-second instants would alias onto one file and the later "
            "checkpoint would replace the earlier one")
    return valid_time.strftime(f"gpuwmrst_{domain}_%Y-%m-%d_%H_%M_%S.npz")


def _host(value) -> np.ndarray:
    """Return a host ndarray view/copy of a device or host array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _is_array_like(value) -> bool:
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and hasattr(value, "ndim"))


def _restore_carriers(driver, driver_header) -> None:
    """Re-establish the surface-radiation carrier contract on a resume.

    THE PROBLEM A RESTART CREATES.  A carrier's provenance and its age are
    the two things a checkpoint's arrays cannot carry: the GLW field is
    serialized with the rest of the surface inventory, but "a longwave
    scheme wrote this at model second 3600" is not a property of the
    numbers.  Rebuild it from the resumed configuration and the first
    post-restart surface call gets one of two wrong answers -- a fresh
    "unwritten" record refuses a perfectly healthy run, or a fresh
    "just written" record admits a carrier that has actually been stale
    since before the checkpoint.

    So it is SERIALIZED, and a checkpoint that carries it resumes with the
    provenance and the ages the uninterrupted run had at that instant.
    Restart identity therefore holds across a resume that lands on a step
    where no radiation call is due, which is exactly the step that would
    otherwise expose the difference.

    A CHECKPOINT WRITTEN BEFORE THIS CONTRACT EXISTED has no such mapping,
    and woof does not guess: the resumed driver is marked so that the
    first surface call refreshes its producers before consuming anything.
    That costs one off-cadence radiation call on a legacy resume and
    changes nothing else; guessing would cost a silent wrong answer.
    """
    stored = driver_header.get("carriers")
    if stored is None:
        driver.carriers_need_producer_refresh = True
        return
    driver.carriers.restore(stored)
    policy = driver_header.get("surface_radiation_policy")
    if policy is not None and policy != driver.carriers.policy:
        raise RestartMismatchError(
            "restart was written under surface_radiation_policy = "
            f"{policy!r} and is resuming under "
            f"{driver.carriers.policy!r}.  The carrier policy binds into "
            "run identity because it decides whether a land-surface "
            "scheme may consume a carrier no producer wrote; a resume "
            "that changed it would be a different experiment continuing "
            "under the first one's name.")


def classify_state_attr(name: str) -> str:
    """Classify one ``DomainState`` attribute name.

    Returns ``"serialize"``, ``"checkpoint_only"``, ``"rebuild"``,
    ``"setup"``, ``"derived_setup"``, or ``"infra"``;
    raises :class:`RestartManifestError` for anything unclassified so new
    state cannot silently skip the restart stream.
    """
    if name in STATE_SERIALIZED_ATTRS:
        return "serialize"
    if name in CHECKPOINT_ONLY_STATE:
        # Carried by a checkpoint, absent from the state identity.  See
        # CHECKPOINT_ONLY_STATE for why the two are different questions.
        return "checkpoint_only"
    if name in STATE_REBUILT_ATTRS:
        return "rebuild"
    if name in STATE_SETUP_ARRAYS or name in STATE_SETUP_SCALARS:
        return "setup"
    if name in STATE_DERIVED_SETUP_ARRAYS:
        # Rebuilt from a fingerprinted setup array by the same load_base
        # call that installs it, and carrying no information of its own.
        return "derived_setup"
    if name in STATE_INFRA_ATTRS:
        return "infra"
    # Chem species (woof/core/chem_state.py), named from the chem table
    # rather than listed here: ``chem_<row>`` is the species field WRF carries
    # in its restart stream; ``chem0_<row>`` is its RK time-t copy, rewritten
    # by dycore._save_time_t before any read, exactly like ``qv0``; ``chem``
    # is the ChemState object (arena and table), rebuilt with the state.
    # ``chemdiag_<name>`` is a process array or ledger total a checkpoint
    # must carry (declared ``serialize`` by its process); ``chemwork_<name>``
    # is rewritten before every read.
    if name.startswith(CHEM_STATE_PREFIX) or name.startswith(CHEM_DIAG_PREFIX):
        return "serialize"
    if name.startswith(CHEM_TIME_PREFIX) or name.startswith(CHEM_WORK_PREFIX):
        return "rebuild"
    if name == "chem":
        return "infra"
    raise RestartManifestError(
        f"DomainState attribute {name!r} is not classified in the restart "
        "manifest (woof/io/restart.py): declare it serialized (cross-step "
        "state), rebuilt (overwritten before every read), setup "
        "(fingerprint-validated), derived_setup (a function of a setup "
        "array, rebuilt with it), or infra")


def _wsm_family_rebuilt_slots() -> frozenset[str]:
    from woof.core.wsm_family import family_selectors, preparation_slots
    return frozenset(slot for mp in family_selectors()
                     for slot in preparation_slots(mp))


_WSM_FAMILY_REBUILT_SLOTS = _wsm_family_rebuilt_slots()


def classify_scratch_slot(slot: str) -> str:
    """Classify one ``DomainState.scratch`` slot name.

    Returns ``"serialize"``, ``"carry"`` or ``"rebuild"``; raises
    :class:`RestartManifestError` for unknown slots.

    ``"carry"`` is cross-step state that is deliberately not checkpointed
    (:data:`CARRIED_SCRATCH_SLOTS` and :data:`CARRIED_SCRATCH_PREFIXES`).
    Every restart path in this module tests
    ``== "serialize"`` or ``!= "serialize"``, so a ``carry`` slot is treated
    exactly like a ``rebuild`` one HERE and the restart stream is unchanged
    byte for byte.  The distinction exists for the consumers that ask this
    function a different question -- above all the streaming carrier set,
    which must move a ``carry`` slot between the store and every tile buffer
    and used to drop it.

    TOTALITY IS THE CONTRACT, not a convenience: the final ``raise`` is what
    stops a new slot being silently dropped from every checkpoint, so every
    slot a run can actually allocate has to reach a branch above it.  The
    per-follower window family is generated per declared child and so is
    matched by prefix; see :data:`CARRIED_SCRATCH_PREFIXES`.
    """
    from woof.core.nest_accumulation import is_archive_slot
    if is_archive_slot(slot):
        return "serialize"
    if slot in SERIALIZED_SCRATCH_SLOTS:
        return "serialize"
    if slot in CARRIED_SCRATCH_SLOTS:
        return "carry"
    if any(slot.startswith(prefix) for prefix in CARRIED_SCRATCH_PREFIXES):
        return "carry"
    if slot in REBUILT_SCRATCH_SLOTS or slot in _WSM_FAMILY_REBUILT_SLOTS:
        return "rebuild"
    if any(slot.startswith(prefix) for prefix in REBUILT_SCRATCH_PREFIXES):
        return "rebuild"
    raise RestartManifestError(
        f"scratch slot {slot!r} is not classified in the restart manifest "
        "(woof/io/restart.py): declare it serialized (persistent "
        "accumulator/held state), carried (cross-step but not checkpointed) "
        "or rebuilt (per-call work buffer)")


def _restorable_scratch_slot(slot: str) -> bool:
    """May a stored ``scratch/<slot>`` member be applied to a live state?

    ``serialize`` always: that class IS "this belongs in a checkpoint".

    A TRACKER WINDOW conditionally, and this is the whole reason the
    question is asked separately from the classification.  A window is
    ``carry`` -- cross-step state that no ORDINARY checkpoint writes --
    and reclassifying it would put it in every checkpoint of every run,
    including the runs with no consumer to read it.  A nest-lifecycle run
    opts its own windows in per member instead
    (:func:`write_tree_restart`), so the reader has to accept a member the
    classification alone says is not written. Strict acoustic carriers
    are checkpointed when allocated for exact continuation. A ``rebuild``
    slot is a per-call work buffer whose value between calls means
    nothing, and an unknown slot still raises out of
    :func:`classify_scratch_slot`.
    """
    return (classify_scratch_slot(slot) == "serialize"
            or slot in CHECKPOINTED_CARRIED_SCRATCH_SLOTS
            or _is_tracker_window_slot(slot)
            or slot in LIFECYCLE_HELD_SCRATCH_SLOTS)


def setup_fingerprint(state) -> str:
    """SHA-256 over the deterministic setup arrays/scalars AND the
    attached lateral-boundary forcing tables.

    A restart written on one setup (base state, coordinates, map factors)
    refuses to restore onto another: silently continuing on different
    reference profiles would not be the same trajectory.  The LBC digest
    (review F3) covers every interval's time bounds and every side's
    value/tendency bytes, so a same-config resume against a modified or
    replaced reference bundle â€” which passes the config echo â€” is
    rejected instead of silently integrating different boundary forcing.
    """
    return _shared_setup_fingerprint(
        state, error_type=RestartManifestError)


def setup_core_fingerprint(state) -> str:
    """SHA-256 over immutable setup, excluding a root's LBC inventory."""
    return _shared_setup_core_fingerprint(
        state, error_type=RestartManifestError)


def lateral_boundary_prefix_identity(state, *,
                                     rebuilt_end_frames: bool = False):
    """Compact byte identity for the root forcing interval inventory."""
    return _lateral_boundary_prefix_identity(
        state, error_type=RestartManifestError,
        rebuilt_end_frames=rebuilt_end_frames)


def _canonical_json(value) -> str:
    """Stable, strict JSON used by semantic restart fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _json_sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value, label: str):
    """Return a strict JSON value without silently stringifying objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RestartManifestError(
                f"{label} contains non-finite value {value!r}")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), label)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, Mapping):
        normalized = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise RestartManifestError(
                    f"{label} has non-string identity key {key!r}")
            normalized[key] = _json_value(value[key], f"{label}.{key}")
        return normalized
    raise RestartManifestError(
        f"{label} value {value!r} is not strict JSON restart identity; "
        "declare strings/numbers/bools/lists/mappings only")


def _array_setup_identity(value) -> dict:
    """Shape/type/content identity for a resolved setup array."""
    host = np.ascontiguousarray(_host(value))
    digest = hashlib.sha256()
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(host.tobytes(order="C"))
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "sha256": digest.hexdigest(),
    }


def _resolved_object_setup_identity(value, label: str) -> dict:
    """Canonical identity for a resolved host coefficient-table object.

    Private device mirrors are intentionally excluded: they are deterministic
    conversions of these authoritative host arrays and may be lazily absent in
    a freshly prepared process.  Every public dataclass/instance field must be
    a setup array or strict JSON scalar/container.
    """
    if value is None:
        raise RestartManifestError(f"resolved {label} object is missing")
    if dataclasses.is_dataclass(value):
        names = [field.name for field in dataclasses.fields(value)]
    else:
        names = list(getattr(value, "__dict__", {}))
    names = sorted(name for name in names if not name.startswith("_"))
    if not names:
        raise RestartManifestError(
            f"resolved {label} object {_callable_class_name(value)} has no "
            "identifiable public fields")
    arrays = {}
    values = {}
    for name in names:
        item = getattr(value, name)
        if _is_array_like(item):
            arrays[name] = _array_setup_identity(item)
        else:
            values[name] = _json_value(item, f"{label}.{name}")
    payload = {
        "class": _callable_class_name(value),
        "arrays": arrays,
        "values": values,
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def _rrtmgp_workspace_identity(workspace) -> dict:
    """Bind the optional workspace code path without hashing scratch bytes."""
    if workspace is None:
        return {"present": False}
    try:
        p_top = float(workspace.p_top)
        if not math.isfinite(p_top) or p_top < 0.0:
            raise ValueError("workspace p_top must be finite and nonnegative")
        identity = {
            "present": True,
            "class": _callable_class_name(workspace),
            "nz": int(workspace.nz),
            "column_chunk": int(workspace.column_chunk),
            "p_top": p_top,
            "nbytes": int(workspace.nbytes),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            "RRTMGP chunk_workspace lacks nz/column_chunk/p_top/nbytes "
            "identity") \
            from exc
    layouts = getattr(workspace, "_phase_layouts", None)
    identity["phase_layouts"] = (
        None if layouts is None
        else _json_value(layouts, "radiation.chunk_workspace.phase_layouts"))
    return identity


def _asset_sha256(path) -> str:
    """Digest the actual packaged bytes consumed by an active scheme.

    Once per process for an unchanged file (woof.file_digest): every
    checkpoint asks again, and a replaced asset is still hashed again."""
    from woof.file_digest import sha256_file
    return sha256_file(Path(path), block_bytes=1024 * 1024)


def _active_asset_identity(cfg, driver) -> dict[str, dict]:
    roles = []
    if getattr(driver, "eclipse", None) is not None:
        roles.append("eclipse_besselian")
    # Asset-only callers historically omit inactive physics selectors.
    if int(getattr(cfg, "bl_pbl_physics", 0)) == 901:
        roles.append("satmedmf_manifest")
    radiation = None if driver is None else driver.radiation_callable
    from woof.core.radiation_composition import modern_radiation_adapters
    if 4 in radiation_scheme_ids(cfg) and modern_radiation_adapters(radiation):
        roles.extend(("rrtmgp_gas_lw", "rrtmgp_gas_sw",
                      "rrtmgp_cloud_lw", "rrtmgp_cloud_sw",
                      "rrtmgp_rfmip"))
    if (4 in radiation_scheme_ids(cfg)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY):
        roles.extend(("wrf_rrtmg_lw_data", "wrf_rrtmg_lw_statics",
                      "wrf_rrtmg_sw_data",
                      "wrf_ozone_data", "wrf_ozone_lat",
                      "wrf_ozone_plev"))
    if int(getattr(cfg, "aer_opt", 0) or 0) == 1:
        roles.extend(("wrf_aer1_climatology", "wrf_aer1_manifest"))
    if radiation_scheme_ids(cfg)[0] == 1:
        roles.append("wrf_rrtm_data")
    if (getattr(driver, "cam_ozone", None) is not None
            and not (4 in radiation_scheme_ids(cfg)
                     and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY)):
        roles.extend(("wrf_ozone_data", "wrf_ozone_lat", "wrf_ozone_plev"))
    land_scheme = int(cfg.sf_surface_physics)
    if land_scheme != 0:
        try:
            _attribute, land_roles = \
                LAND_SURFACE_PARAMETER_SOURCES[land_scheme]
        except KeyError:
            raise RestartManifestError(
                f"land-surface scheme {land_scheme} has no packaged-asset "
                "roles in LAND_SURFACE_PARAMETER_SOURCES "
                "(woof/io/restart.py); its table bytes cannot be bound to "
                "the checkpoint") from None
        roles.extend(land_roles)
    cumulus = None if driver is None else driver.cumulus_callable
    cumulus_class = (None if cumulus is None
                     else _callable_class_name(cumulus))
    if (int(cfg.cu_physics) == 1
            and cumulus_class == "woof.core.kf.KainFritsch"):
        roles.append("kf_lutab")
    identity = {}
    for role in roles:
        relative = PHYSICS_ASSET_PATHS[role]
        path = _resolve_physics_asset(relative)
        try:
            size = path.stat().st_size
            sha256 = _asset_sha256(path)
        except OSError as exc:
            raise RestartManifestError(
                f"active physics asset {role!r} is unreadable at {path}") \
                from exc
        identity[role] = _recorded_asset_identity(
            role, relative, int(size), sha256)
    return identity


def _recorded_asset_identity(role: str, relative: Path, size: int,
                             sha256: str) -> dict:
    """The identity a restart manifest records for one active asset.

    The bytes read, except for one role.  ``rrtmgp_rfmip`` named the RFMIP
    input NetCDF until 2.8.0, when the driver switched to the climatology
    derived from it (the NetCDF no longer ships).  The pinned table holds
    exactly the float64 values the driver read from that file, so while
    the table matches its pin the role keeps recording the source file and
    every RRTMGP checkpoint written before 2.8.0 stays resumable.  Table
    bytes that miss the pin keep their own identity, and a resume refuses.
    """

    if role == "rrtmgp_rfmip":
        from woof.core.rfmip_upstream import (
            TRACE_CLIMATOLOGY_SHA256, TRACE_CLIMATOLOGY_SOURCE)
        if sha256 == TRACE_CLIMATOLOGY_SHA256:
            return dict(TRACE_CLIMATOLOGY_SOURCE)
    return {"path": relative.as_posix(), "bytes": size, "sha256": sha256}


def _scheme_algorithm(mapping: dict[int, str], scheme_id, label: str) -> str:
    try:
        return mapping[int(scheme_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"cannot identify unsupported {label} scheme {scheme_id!r}") \
            from exc


def _callable_class_name(value) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _stock_callable_class(component_id: str, selectors: dict, *,
                          label: str, variant: str | None = None,
                          missing_is_custom: bool = False) -> str | None:
    """The stock adapter class the registry names for a selection.

    ``missing_is_custom`` returns ``None`` for a selection that matches no
    registered option (a composed radiation pair), which sends the caller
    down the custom-callable path exactly as the literal dict's ``.get``
    did; without it an unknown selection is a checkpoint refusal that says
    "no stock-class row" rather than "declare a restart_identity".
    """

    from woof.physics_registry import stock_callable_class

    try:
        return stock_callable_class(component_id, selectors, variant=variant)
    except KeyError as exc:
        if missing_is_custom and "no registered" in str(exc):
            return None
        raise RestartManifestError(
            f"active {label} selection {selectors} has no stock-class row "
            "in woof/physics_registry_v2.json "
            f"(components.{component_id}.options.<option>.consumers."
            f"stock_callable_class): {exc}.  A checkpoint cannot name the "
            "adapter it serialised; give the option its row in "
            "tools/build_registry.py and regenerate the registry") from exc


def _callable_setup_identity(scheme, *, label: str,
                             expected_class: str | None) -> dict:
    """Identify a stock callable, or require a custom declaration.

    A class name alone cannot bind a closure/custom adapter's parameters.
    Non-stock callables therefore opt in with a strict-JSON
    ``restart_identity`` attribute (or zero-argument method).
    """
    if scheme is None:
        raise RestartManifestError(
            f"active {label} scheme has no callable to identify")
    class_name = _callable_class_name(scheme)
    identity = {"class": class_name}
    if expected_class is not None and class_name == expected_class:
        identity["implementation"] = "stock"
        return identity
    declared = getattr(scheme, "restart_identity", None)
    if callable(declared):
        declared = declared()
    if declared is None:
        raise RestartManifestError(
            f"custom {label} callable {class_name} must declare a strict-JSON "
            "restart_identity so restarts cannot cross incompatible code or "
            "parameters")
    identity["implementation"] = "custom"
    identity["declared_identity"] = _json_value(
        declared, f"{label}.restart_identity")
    return identity


def _float_mapping(value, label: str) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        items = value.items()
    except AttributeError as exc:
        raise RestartManifestError(f"{label} must be a mapping or None") \
            from exc
    result = {}
    for key, raw in sorted(items, key=lambda pair: str(pair[0])):
        if not isinstance(key, str):
            raise RestartManifestError(f"{label} key {key!r} is not a string")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise RestartManifestError(
                f"{label}[{key!r}] is not a numeric mole fraction: "
                f"{raw!r}") from exc
        if not np.isfinite(number):
            raise RestartManifestError(
                f"{label}[{key!r}] is non-finite: {raw!r}")
        result[key] = number
    return result


def _radiation_setup_identity(driver, cfg) -> dict:
    lw_id, sw_id = radiation_scheme_ids(cfg)
    legacy_rrtmg = ((lw_id, sw_id) == (4, 4)
                    and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY)
    if legacy_rrtmg:
        # The legacy port shares WRF scheme id 4 with the RTE+RRTMGP
        # substitution but is a different algorithm; its identity strings
        # are distinct so restarts cannot cross the two implementations.
        lw_algorithm = RRTMG_LEGACY_LW_ALGORITHM_IDENTITY
        sw_algorithm = RRTMG_LEGACY_SW_ALGORITHM_IDENTITY
        lw_policy = sw_policy = RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY
    elif lw_id == sw_id and lw_id in RADIATION_ALGORITHM_IDENTITIES:
        # Keep the historical public mapping authoritative for coupled
        # 0/0, 4/4, and 90/90 setups (including audit/test monkeypatches).
        lw_algorithm = sw_algorithm = _scheme_algorithm(
            RADIATION_ALGORITHM_IDENTITIES, lw_id, "radiation")
        lw_policy = sw_policy = _scheme_algorithm(
            RADIATION_ABOVE_ATMOSPHERE_POLICIES, lw_id,
            "radiation above-atmosphere policy")
    else:
        lw_algorithm = _scheme_algorithm(
            LONGWAVE_ALGORITHM_IDENTITIES, lw_id, "longwave radiation")
        sw_algorithm = _scheme_algorithm(
            SHORTWAVE_ALGORITHM_IDENTITIES, sw_id, "shortwave radiation")
        lw_policy = _scheme_algorithm(
            LONGWAVE_ABOVE_ATMOSPHERE_POLICIES, lw_id,
            "longwave above-atmosphere policy")
        sw_policy = _scheme_algorithm(
            SHORTWAVE_ABOVE_ATMOSPHERE_POLICIES, sw_id,
            "shortwave above-atmosphere policy")
    if 4 in (lw_id, sw_id) and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
        if lw_id == 4:
            lw_algorithm = RRTMG_LEGACY_LW_ALGORITHM_IDENTITY
            lw_policy = RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY
        if sw_id == 4:
            sw_algorithm = RRTMG_LEGACY_SW_ALGORITHM_IDENTITY
            sw_policy = RRTMG_LEGACY_ABOVE_ATMOSPHERE_POLICY
    algorithm = (lw_algorithm if lw_algorithm == sw_algorithm else
                 f"lw={lw_algorithm};sw={sw_algorithm}")
    policy = (lw_policy if lw_policy == sw_policy else
              f"lw={lw_policy};sw={sw_policy}")
    identity = {
        "scheme_id": lw_id if lw_id == sw_id else None,
        "scheme_ids": {"lw": lw_id, "sw": sw_id},
        "algorithm": algorithm,
        "algorithms": {"lw": lw_algorithm, "sw": sw_algorithm},
        "above_atmosphere_policy": policy,
        "above_atmosphere_policies": {"lw": lw_policy, "sw": sw_policy},
        "callable": None,
    }
    if not (lw_id or sw_id):
        return identity
    if driver is None:
        raise RestartManifestError(
            "active radiation cannot be restart-identified without an "
            "attached PhysicsDriver")
    scheme = driver.radiation_callable
    # The stock class for this selection comes from the REGISTRY's row
    # (``consumers.stock_callable_class``), not from a literal dict here:
    # a selection the registry knows and this dict did not was routed down
    # the custom-callable path and refused for lacking a restart_identity
    # the stock adapter never had.  A composed lw != sw pair has no registry
    # option and no stock class -- ComposedRadiation declares its own
    # identity -- so the lookup's None is the custom path, as before.  A
    # 4/4 selection names two adapters and ra_rrtmg_variant picks one.
    expected = _stock_callable_class(
        "radiation", {"ra_lw_physics": lw_id, "ra_sw_physics": sw_id},
        variant=("rrtmg_legacy" if legacy_rrtmg else "rte-rrtmgp"),
        label="radiation", missing_is_custom=True)
    callable_identity = _callable_setup_identity(
        scheme, label="radiation", expected_class=expected)
    identity["callable"] = callable_identity
    if callable_identity["implementation"] == "custom":
        declared = callable_identity["declared_identity"]
        if not isinstance(declared, Mapping):
            raise RestartManifestError(
                "custom radiation restart_identity must be a mapping with "
                "explicit algorithm and above_atmosphere_policy entries")
        missing = sorted(
            {"algorithm", "above_atmosphere_policy"} - set(declared))
        if missing:
            raise RestartManifestError(
                f"custom radiation restart_identity is missing {missing}")
        for name in ("algorithm", "above_atmosphere_policy"):
            if not isinstance(declared[name], str) or not declared[name]:
                raise RestartManifestError(
                    f"custom radiation restart_identity[{name!r}] must be "
                    "a non-empty string")
        identity["configured_slot_algorithm"] = algorithm
        identity["algorithm"] = declared["algorithm"]
        identity["above_atmosphere_policy"] = \
            declared["above_atmosphere_policy"]
        return identity
    try:
        start_time = scheme.start_time
        latitude = scheme.latitude_deg
        longitude = scheme.longitude_deg
    except AttributeError as exc:
        raise RestartManifestError(
            "active radiation callable is missing start_time/latitude_deg/"
            "longitude_deg setup required by restart identity") from exc
    if not isinstance(start_time, datetime):
        raise RestartManifestError(
            "radiation start_time is not a datetime restart identity")
    identity.update({
        "start_time": start_time.isoformat(),
        "latitude": _array_setup_identity(latitude),
        "longitude": _array_setup_identity(longitude),
    })
    if legacy_rrtmg and getattr(scheme, "trace_gas_overrides", None):
        # Stock dispatch does not consume the adapter's declared identity.
        # Add only explicit overrides, preserving historical default headers.
        identity["trace_gas_overrides"] = _float_mapping(
            scheme.trace_gas_overrides, "radiation.trace_gas_overrides")
    if (lw_id, sw_id) == (1, 1):
        # The stock classic pair's historical adapter declaration is retained
        # unchanged. It predates the custom above-atmosphere-policy contract;
        # recognize this stock implementation and bind its actual operands
        # through the same leaf identity used by mixed-spectrum compositions.
        from woof.core.radiation_composition import adapter_restart_identity
        identity["classic"] = adapter_restart_identity(scheme)
    if (lw_id, sw_id) == (4, 4) and not legacy_rrtmg:
        try:
            identity.update({
                "column_chunk": int(scheme.column_chunk),
                "validation_mode": str(scheme.validation_mode),
                "trace_gas_policy": RRTMGP_TRACE_GAS_POLICY_IDENTITY,
                "trace_gas_overrides": _float_mapping(
                    scheme.trace_gas_overrides,
                    "radiation.trace_gas_overrides"),
                "trace_vmr": _float_mapping(
                    scheme.trace_vmr, "radiation.trace_vmr"),
                "ozone_log_pressure": _array_setup_identity(
                    scheme._ozone_logp),
                "ozone_vmr": _array_setup_identity(scheme._ozone_vmr),
                "coefficient_tables": {
                    "gas_lw": _resolved_object_setup_identity(
                        scheme.lw_tables, "RRTMGP LW gas table"),
                    "gas_sw": _resolved_object_setup_identity(
                        scheme.sw_tables, "RRTMGP SW gas table"),
                    "cloud_lw": _resolved_object_setup_identity(
                        scheme.lw_cloud_tables, "RRTMGP LW cloud table"),
                    "cloud_sw": _resolved_object_setup_identity(
                        scheme.sw_cloud_tables, "RRTMGP SW cloud table"),
                },
                "chunk_workspace": _rrtmgp_workspace_identity(
                    getattr(scheme, "chunk_workspace", None)),
            })
        except AttributeError as exc:
            raise RestartManifestError(
                "RRTMGP radiation callable is missing resolved gas/ozone/"
                "execution setup required by restart identity") from exc
    elif (lw_id, sw_id) == (90, 90):
        # These are module constants used directly by the analytic proxy.
        # Pin their resolved numerical values as well as its algorithm tag.
        from woof.core.analytic_radiation import (
            CLEAR_SKY_TRANSMISSIVITY, SOLAR_CONSTANT_WM2,
            STEFAN_BOLTZMANN)
        identity["constants"] = {
            "clear_sky_transmissivity": float(CLEAR_SKY_TRANSMISSIVITY),
            "solar_constant_wm2": float(SOLAR_CONSTANT_WM2),
            "stefan_boltzmann": float(STEFAN_BOLTZMANN),
        }
    elif (lw_id, sw_id) == (0, 1):
        try:
            identity["dudhia"] = {
                "swrad_scat": float(scheme.swrad_scat),
                "icloud": int(scheme.icloud),
                "supported_path": "no-chem,no-eclipse,no-slope",
                "oracle": "WRF-v4.6.1 phys/module_ra_sw.F:SWRAD/SWPARA",
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise RestartManifestError(
                "Dudhia radiation callable is missing resolved setup "
                "required by restart identity") from exc
    return identity


def _land_surface_parameters_identity(cfg, driver) -> dict:
    """Identify the ACTIVE land-surface scheme's own parameter bundle.

    Dispatches on the scheme value through
    :data:`LAND_SURFACE_PARAMETER_SOURCES`; an unregistered scheme fails
    instead of falling through to Noah's bundle, which would write a
    checkpoint claiming Noah's tables for a run that never used them.
    """
    scheme = int(cfg.sf_surface_physics)
    try:
        attribute, _roles = LAND_SURFACE_PARAMETER_SOURCES[scheme]
    except KeyError:
        raise RestartManifestError(
            f"land-surface scheme {scheme} has no parameter-bundle row in "
            "LAND_SURFACE_PARAMETER_SOURCES (woof/io/restart.py); a "
            "checkpoint cannot identify the tables it ran with") from None
    params = getattr(driver, attribute, None)
    label = LAND_SURFACE_ALGORITHM_IDENTITIES[scheme]
    payload = _packed_parameters_identity(params, label=label,
                                          attribute=attribute)
    geometry = getattr(driver, "noahmp_geometry", None)
    if geometry is not None:
        # Noah-MP reads COSZ, XLAT, JULIAN and YR, none of which any other
        # part of the checkpoint records.  Resuming with a different start
        # time or latitude grid would silently continue a different
        # trajectory, so the geometry is bound here and the digest is
        # recomputed over the merged payload.
        declared = getattr(geometry, "restart_identity", None)
        if declared is None:
            raise RestartManifestError(
                "Noah-MP solar geometry must declare a strict-JSON "
                "restart_identity")
        payload.pop("sha256", None)
        payload["solar_geometry"] = _json_value(
            declared, f"{attribute}.solar_geometry")
        payload["sha256"] = _json_sha256(payload)
    return payload


def _packed_parameters_identity(params, *, label: str,
                                attribute: str = "noah_params") -> dict:
    if params is None:
        raise RestartManifestError(
            f"active land surface scheme {label!r} has no resolved "
            f"parameters on PhysicsDriver.{attribute}")
    class_name = _callable_class_name(params)
    if not dataclasses.is_dataclass(params):
        declared = getattr(params, "restart_identity", None)
        if callable(declared):
            declared = declared()
        if declared is None:
            raise RestartManifestError(
                f"custom land-surface parameters {class_name} must declare "
                "a strict-JSON restart_identity")
        payload = {
            "class": class_name,
            "declared_identity": _json_value(
                declared, f"{attribute}.restart_identity"),
        }
        payload["sha256"] = _json_sha256(payload)
        return payload
    arrays = {}
    values = {}
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if _is_array_like(value):
            arrays[field.name] = _array_setup_identity(value)
        else:
            values[field.name] = _json_value(
                value, f"{attribute}.{field.name}")
    payload = {"class": class_name, "arrays": arrays, "values": values}
    payload["sha256"] = _json_sha256(payload)
    return payload


#: The output-only switches the configuration digest has DROPPED since it
#: first bound them.  Every other member of CONFIG_DIAGNOSTIC_FIELDS is
#: held at its RunConfig default in the digest instead: dropping a key
#: moves the digest of every configuration, which re-pins every checkpoint
#: byte test for no change in any checkpoint's meaning, while holding it
#: at its default leaves every configuration that leaves it off with the
#: bytes it always had.  Either way the digest cannot tell two values of
#: the switch apart, which is all the restart walk needs.
_DIGEST_DROPPED_DIAGNOSTIC_FIELDS = frozenset(
    {"hailcast_opt", "haildt", "hailcast_version", "nwp_diagnostics", "tke_budget", "sase_flux_diag", "surface_energy_diag",
     # Correct mode WB1: output-only and on by default from 2.9.0, so it is
     # dropped rather than held at a default no earlier digest carried.
     "thompson_cloud_surface_booking"})


#: WRF's slope_rad / topo_shading / shadlen (woof.core.topo_radiation),
#: absent-stays-absent in the config echo and the configuration digest, the
#: rule woof.core.model.restart_identity_payload applies to the same keys:
#: off, none of the three is read, so a checkpoint written with them off is
#: byte-identical to one written before they existed.
TOPO_RADIATION_RUN_DEFAULTS = {"slope_rad": 0, "topo_shading": 0,
                               "shadlen": 25000.0}


#: WRF's topo_wind / gwd_opt (woof.core.terrain_drag) on the same
#: absent-stays-absent rule: off, a checkpoint is byte-identical to one
#: written before the two fields existed; on, they bind like any other
#: trajectory setting.
TERRAIN_DRAG_RUN_DEFAULTS = {"topo_wind": 0, "gwd_opt": 0}
#: WRF &dynamics advection orders at the values every header written
#: before the fields existed ran (vert_order 3 ladder, flux5 momentum).
ADVECTION_ORDER_RUN_DEFAULTS = {"v_sca_adv_order": 3, "v_mom_adv_order": 3,
                                "h_mom_adv_order": 5}


def _drop_inert_terrain_drag(values: dict) -> None:
    for name, default in TERRAIN_DRAG_RUN_DEFAULTS.items():
        if values.get(name, default) == default:
            values.pop(name, None)


def _drop_inert_topo_radiation(values: dict) -> None:
    _drop_inert_terrain_drag(values)
    if not values.get("slope_rad", 0):
        for name in TOPO_RADIATION_RUN_DEFAULTS:
            values.pop(name, None)
    elif not values.get("topo_shading", 0):
        values.pop("topo_shading", None)
        values.pop("shadlen", None)


def _drop_inert_mosaic(values: dict) -> None:
    """Noah mosaic's three keys, absent-stays-absent (the rule
    woof.core.model.restart_identity_payload applies): off, none is read;
    on, the urban canopy rule is written only when it is not WRF's."""
    if values.get("sf_surface_mosaic", 0) == 0:
        for name in ("sf_surface_mosaic", "mosaic_cat", "mosaic_urban_canopy"):
            values.pop(name, None)
    elif values.get("mosaic_urban_canopy") == "dominant":
        values.pop("mosaic_urban_canopy", None)


def _mosaic_checkpoint_config(config: Mapping) -> dict:
    """Keep pre-mosaic headers and comparisons byte-identical when off."""
    values = dict(config)
    from woof.wind_table import drop_inert_config
    drop_inert_config(values)
    if values.get("sf_surface_physics", 0) != 4:
        values.pop("noahmp_snow_age", None)
    if not values.get("sf_lake_physics", 0):
        for name in ("sf_lake_physics", "use_lakedepth", "lakedepth_default", "lake_min_elev"):
            values.pop(name, None)
    # WRF's ocean mixed layer (sf_ocean_physics = 1): off, no key echoes.
    from woof.surface_ocean import drop_inert_ocean
    drop_inert_ocean(values)
    _drop_inert_mosaic(values)
    drop_default_diffusion_selectors(values)
    from woof.config import drop_inert_g3_fields
    drop_inert_g3_fields(values)
    return values


def _drop_default_fork_run_keys(values: dict) -> None:
    """Use the writer's omission rules when comparing fork settings too."""
    from woof.checkpoint_identity import drop_default_cumulus_selectors
    drop_default_cumulus_selectors(values)
    if values.get("diff_6th_form", "wrf_461") == "wrf_461":
        values.pop("diff_6th_form", None)
    if values.get("diff_6th_factor2", None) is None:
        values.pop("diff_6th_factor2", None)
    if values.get("upper_wind_limiter_form", "wrf_461") == "wrf_461":
        values.pop("upper_wind_limiter_form", None)
    if not values.get("mp_zero_out", 0):
        values.pop("mp_zero_out", None)
        values.pop("mp_zero_out_thresh", None)
        values.pop("mp_zero_out_all", None)


def _drop_default_radiation_port_keys(values: dict) -> None:
    """Drop the lightning and aer_opt = 2 controls while they are read by nothing.

    Shared by the writers' echo/digest (:func:`_drop_default_off_run_keys`)
    and the resume comparison (:func:`_require_config_match`), so a 2.8.9
    checkpoint, which predates these fields, resumes under a build that has
    them, and a new checkpoint compares under the same omission rule.
    Named breakage: without the comparison half every checkpoint written
    before the fields existed was refused as "a different model".
    """
    from woof.config import drop_inactive_radiation_port_keys
    drop_inactive_radiation_port_keys(values)

    # Default-on eclipse is implicit in old non-eclipse headers. A live
    # shadow additionally binds its algorithm, table and held field inventory.
    # Explicit zero remains serialized, so switching it changes identity.
    if values.get("ra_sw_eclipse", 1) == 1:
        values.pop("ra_sw_eclipse", None)
    # seaice_threshold is read only by the initial surface (real.exe's
    # adjust_for_seaice rule), never by a resumed forecast; at the Registry
    # default 100 K it converts nothing, so older headers that predate it
    # hold the same model.  A stated threshold stays serialized.
    if float(values.get("seaice_threshold", 100.0)) == 100.0:
        values.pop("seaice_threshold", None)
    from woof.checkpoint_identity import drop_default_output_controls
    drop_default_output_controls(values)


from woof.checkpoint_identity import (  # noqa: E402
    CAMPAIGN_IMPLICIT_RUN_KEYS as _CAMPAIGN_IMPLICIT_RUN_KEYS,
    SEAICE_RUN_KEYS as _SEAICE_RUN_KEYS,
    drop_capability_campaign_keys as _drop_capability_campaign_keys,
    seaice_read as _seaice_read)


def _drop_default_off_run_keys(values: dict) -> None:
    """Drop every default-off RunConfig key that older headers omit.

    At its default none of these keys is read, so a checkpoint written with
    it there echoes and digests exactly as one written before the field
    existed; moved, it stays and binds like any other trajectory setting.
    The checkpoint echo (:func:`configuration_echo`) and the configuration
    digest (:func:`_configuration_digest_values`) both drop through here,
    so every writer drops each key the same way.  A new default-off field
    joins this list, naming the commit that added it.
    """
    from woof.wind_table import drop_inert_config
    drop_inert_config(values)
    from woof.config import drop_inert_g3_fields
    _drop_inert_nest_archive(values)
    drop_inert_g3_fields(values)
    _drop_inert_chem_block(values)
    _drop_inert_fire_block(values)
    # sf_ocean_physics and the oml_* settings (lane 291-fix-notes-and-doors-13).
    from woof.surface_ocean import drop_inert_ocean
    drop_inert_ocean(values)
    # Correct-mode keys (woof/correct_mode.py): False is the matched
    # WRF/MPAS arithmetic every earlier header ran, and outside the scheme
    # that reads it a key reaches nothing.  From 2.9.0 a key defaults ON
    # (correct mode is the default), and it echoes and digests only where
    # it is on AND read, so a default run whose schemes never read it --
    # every run without WSM6 or New Tiedtke -- echoes and digests as 2.8.9
    # did, and an on key under its scheme binds.
    for key in _CORRECTION_KEYS:
        if (key in _OUTPUT_ONLY_CORRECTION_KEYS
                or not values.get(key, False)
                or not _correct_key_in_scope(key, values)):
            values.pop(key, None)
    from woof.checkpoint_identity import drop_default_cumulus_selectors
    drop_default_cumulus_selectors(values)
    # No prescribed smoke is the source forcing every earlier header ran.
    if not values.get("rrtmg_smoke_manifest", ""):
        values.pop("rrtmg_smoke_manifest", None)
        values.pop("rrtmg_smoke_manifest_identity", None)
    # The pre-existing scalar path always used the PD operator. Preserve
    # the historical off-arm identity when that selector is its default.
    if values.get("scalar_adv_opt", 1) == 1:
        values.pop("scalar_adv_opt", None)
    # scalar_pblmix: off preserves checkpoints predating scalar diffusion.
    if not values.get("surface_energy_diag", False):
        values.pop("surface_energy_diag", None)
    if not values.get("scalar_pblmix", 0):
        values.pop("scalar_pblmix", None)
    # MYNN-EDMF's split mixing keys (lane rrfs288-gap1): off mixes nothing
    # beyond what bl_mynn_mixscalars already selects, every earlier header.
    for key in ("bl_mynn_mixaerosols", "bl_mynn_mixnumcon"):
        if not values.get(key, 0):
            values.pop(key, None)
    # TEMPO's optional coupling arms (lane rrfs-hex-tempo-preset): off is
    # what every earlier header ran; an enabled arm still binds.
    for key in ("tempo_ml_for_bl_nc", "tempo_surface_emission_before"):
        if values.get(key, False) is False:
            values.pop(key, None)
    # use_rap_aero_icbc: off is the aerosol start every earlier header ran.
    if not values.get("use_rap_aero_icbc", False):
        values.pop("use_rap_aero_icbc", None)
    # CLM lake off keeps prescribed water temperatures and the earlier
    # header.  RUC's own lake-column change at off is bound by its
    # algorithm identity (LAND_SURFACE_ALGORITHM_IDENTITIES[3]), not here.
    if not values.get("sf_lake_physics", 0):
        values.pop("sf_lake_physics", None)
        values.pop("use_lakedepth", None)
        values.pop("lakedepth_default", None)
        values.pop("lake_min_elev", None)
    # The 2.9.4 capabilities campaign's keys, omitted while they reach
    # nothing so every earlier header resumes under a default run (named
    # breakage: a 2.8.7 checkpoint refused with "lake_state_path: absent
    # from the restart file").  _require_config_match applies the same rule.
    _drop_capability_campaign_keys(values)
    # adaptive_nest_lattice (9a2f3fa63): off is the original root clock.
    if not values.get("adaptive_nest_lattice", False):
        values.pop("adaptive_nest_lattice", None)
    # zadvect_implicit (A158, e9300fa4a): 0 is the explicit vertical
    # advection every header written before the field describes.
    if not values.get("zadvect_implicit", 0):
        values.pop("zadvect_implicit", None)
    # Before the numerical-generation selector, IEVA used WRF 4.7.1.
    if values.get("zadvect_implicit_variant", "wrf_471") == "wrf_471":
        values.pop("zadvect_implicit_variant", None)
    # The RUC SOILPROP lineage: wrf_45 is omitted at its default.
    if values.get("ruc_soilprop", "wrf_45") == "wrf_45":
        values.pop("ruc_soilprop", None)
    # Before this selector every header used the older equation of time.
    # Omit only the legacy value. The new generic default must stay bound.
    if values.get("solar_generation", "wrf_461") == "wrf_461":
        values.pop("solar_generation", None)
    # The scheme-scoped WRF 4.7/4.8 fix selectors.  A header echoes one only
    # while its scheme runs under the corrected generation: with the scheme
    # off the key reaches nothing, and at the previous generation it is what
    # every earlier header (which has no such key) ran.  So a run that does
    # not use Noah-MP or the single-layer urban model, and any run pinned to
    # the previous generation, echoes and digests as 2.8.9 did.
    if (values.get("sf_surface_physics", 0) != 4
            or values.get("noahmp_snow_age", "wrf_461") == "wrf_461"):
        values.pop("noahmp_snow_age", None)
    if (int(values.get("sf_urban_physics", 0) or 0) != 1
            or values.get("urban_water_balance", "wrf_471") == "wrf_471"):
        values.pop("urban_water_balance", None)
    # The Thompson generation: wrf_461 is what every earlier header ran.
    if values.get("thompson_version", "wrf_461") == "wrf_461":
        values.pop("thompson_version", None)
    if values.get("thompson_table_dir") is None:
        values.pop("thompson_table_dir", None)
    if values.get("thompson_fork_snow_fall", "blend") == "blend":
        values.pop("thompson_fork_snow_fall", None)
    # The MYNN generation: wrf_461 is what every earlier header ran.
    if values.get("bl_mynn_version", "wrf_461") == "wrf_461":
        values.pop("bl_mynn_version", None)
    # Omitted historical headers mean legacy, even though new runs default
    # to the corrected generation. Never drop the corrected selection.
    if (values.get("bl_pbl_physics", 0) != 1
            or values.get("ysu_version", "wrf_461") == "wrf_461"):
        values.pop("ysu_version", None)
    if values.get("bl_mynn_gsd41_unsquared_qtke", False) is False:
        values.pop("bl_mynn_gsd41_unsquared_qtke", None)
    if values.get("bl_mynn_cloud_tendency_form", "wrf_461") == "wrf_461":
        values.pop("bl_mynn_cloud_tendency_form", None)
    # WRF's cycling flag: omitted at its .false. default.
    if values.get("cycling", False) is False:
        values.pop("cycling", None)
    # The MYNN surface-layer generation: wrf_461 is omitted at its default.
    if values.get("mynn_sfclay_variant", "wrf_461") == "wrf_461":
        values.pop("mynn_sfclay_variant", None)
    if values.get("urban_ucm_generation", "wrf_471") == "wrf_471":
        values.pop("urban_ucm_generation", None)
    for name in ("urban_treeoption", "urban_ttscheme", "urban_fvg"):
        if values.get(name, -1) == -1:
            values.pop(name, None)
    if values.get("urban_table_dir", "") == "":
        values.pop("urban_table_dir", None)
    # The sixth-order filter form and the fork's second factor (unset at
    # the WRF v4.6.1 form), and mp_zero_out with its threshold and array
    # switch (off), are omitted at their defaults: every earlier header
    # ran the v4.6.1 filter and no zero-out pass.
    _drop_default_fork_run_keys(values)
    # The generic RUC irrigation rule: wrf_461 is omitted at its default.
    if values.get("ruc_irrigation", "wrf_461") == "wrf_461":
        values.pop("ruc_irrigation", None)
    # The RUC QVG cold start: wrf is omitted at its default.
    if values.get("ruc_qvg_cold_start", "wrf") == "wrf":
        values.pop("ruc_qvg_cold_start", None)
    # The RUC 2 m diagnostic: flux is omitted at its default.
    if values.get("ruc_2m_diagnostic", "flux") == "flux":
        values.pop("ruc_2m_diagnostic", None)
    # The RUC snow lineage: wrf_461 is omitted at its default.
    if values.get("ruc_snow", "wrf_461") == "wrf_461":
        values.pop("ruc_snow", None)
    # Before the advection-order fields, every header ran WRF's vert_order
    # 3 ladder and the flux5 horizontal momentum stencil.
    for key, default in ADVECTION_ORDER_RUN_DEFAULTS.items():
        if int(values.get(key, default)) == default:
            values.pop(key, None)
    # w_crit_cfl (A165, c0d566414): 1.0 is the w_damp every header written
    # before the field ran.
    if float(values.get("w_crit_cfl", 1.0)) == 1.0:
        values.pop("w_crit_cfl", None)
    # slope_rad, topo_shading and shadlen (db6e9eae5), and inside it
    # topo_wind and gwd_opt (12d03dd6c, _drop_inert_terrain_drag).
    _drop_inert_topo_radiation(values)
    # sf_surface_mosaic and mosaic_cat (fa2e5efd8), mosaic_urban_canopy
    # (c517106a8).
    _drop_inert_mosaic(values)
    # diff_opt and mix_full_fields (679a5f5fe): diff_opt 2 with full-field
    # mixing is the metric operator every earlier header ran.
    drop_default_diffusion_selectors(values)
    # SPP consumers (1ef07b807): zero leaves the deterministic trajectory
    # and the pre-feature checkpoint echo and digest unchanged.
    drop_default_spp_selectors(values)
    # lightning_option and aer_opt = 2 controls (golden/ports-rad): off
    # leaves the echo and digest of every earlier header unchanged.
    _drop_default_radiation_port_keys(values)
    # terrain_clock (lane/286-fixed-step-grid): "measured" is the
    # launch-time derivation every header written before the field ran
    # under, so the echo and digest read as they did; "pinned" stays, and
    # so does "local_face" (the default from 2.8.8), which can run a
    # longer step than "measured" on the same grid.
    if values.get("terrain_clock", "measured") == "measured":
        values.pop("terrain_clock", None)
    # fractional_seaice (lane/286-veg-albedo-seaice, 654993324): 0 is the
    # 0.5 sea-ice threshold every RUC header written before the field
    # ran; 1 (0.02, WRF's fractional branch) stays and binds.
    if not values.get("fractional_seaice", 0):
        values.pop("fractional_seaice", None)
    # The frozen cloud wrapper every earlier checkpoint ran.
    if values.get("rrtmg_cloud_optics_form", "wrf_461") == "wrf_461":
        values.pop("rrtmg_cloud_optics_form", None)
    # The rural aer_opt=3 aerosol type every earlier checkpoint ran.
    if values.get("rrtmg_aer3_type_source", "rural") == "rural":
        values.pop("rrtmg_aer3_type_source", None)
    # Disabled radiation-driver options retain the trajectory and header
    # identity from before their fields were added.
    for key in ("swint_opt", "aer_opt", "alb_sol"):
        if not values.get(key, 0):
            values.pop(key, None)


def _drop_inert_nest_archive(values):
    if not values.get('nested') or not values.get('nest_accumulation_archive', False):
        values.pop('nest_accumulation_archive', None)


def configuration_echo(cfg) -> dict:
    """The run-configuration echo of every checkpoint header.

    The resident writer (:func:`_write_restart`), the streamed writer
    (``tilestream.restart_stream.write_streamed_restart``) and the
    out-of-core store writer (``tilestream.checkpoint.store_restart_header``)
    all call this one function, so one run's checkpoint describes its
    configuration the same way whichever road wrote it (A191: the streamed
    writer echoed nine default keys the resident writer drops, and two
    checkpoints of one state compared unequal with identical arrays).
    """
    values = dataclasses.asdict(cfg)
    _drop_default_off_run_keys(values)
    from woof.core.rrtmg_smoke_identity import bind_smoke_source_identity
    bind_smoke_source_identity(values)
    return values


def _drop_inert_chem_block(values: dict) -> None:
    """Remove the chem block from a config echo while chem is off, in place.

    With ``chem_sets`` empty every chem key is read by nothing
    (woof.config.validate_chem_config), so a chem-off checkpoint's header
    and configuration digest are the bytes they were before the block
    existed; :func:`_require_config_match` reads the absent keys at their
    defaults.  A chem run echoes the whole block and is compared key by
    key.
    """
    if values.get("chem_sets"):
        # The HRRR v4.1.21 smoke fork's controls, appended to the block by
        # c363fd55ff (simple_dir_fdb) and fc1e2df360: at their defaults they
        # are absent from a chem run's echo too, so every chem checkpoint
        # written before them keeps its configuration digest (named
        # breakage: the 2.8.7 sfire_smoke header's configuration_sha256,
        # tests/test_restart_287_header.py).  _require_config_match reads
        # an absent chem key at its default; a stated value binds.
        for key, default in CHEM_FORK_RUN_DEFAULTS.items():
            if values.get(key, default) == default:
                values.pop(key, None)
        return
    from woof.config import CHEM_RUN_FIELDS
    for key in CHEM_RUN_FIELDS:
        values.pop(key, None)


#: The smoke fork's chem keys at the value every earlier chem run held.
CHEM_FORK_RUN_DEFAULTS = {"simple_dir_fdb": 0, "enh_vermix": False,
                          "flam_part": -1.0, "aer_drydep_opt": 0}


def _drop_inert_fire_block(values: dict) -> None:
    """Remove the fire (SFIRE) block from a config echo while fire is off.

    With ``ifire`` 0 no fire key is read (the FireCoupler and the fire
    static are built only at ``ifire`` 2), so a fire-off checkpoint's
    header and configuration digest are the bytes they were before the
    block existed.  A fire run echoes the whole block and binds key by key.
    """
    from woof.config import inert_fire_fields
    for key in inert_fire_fields(values):
        values.pop(key, None)


def _configuration_digest_values(config: Mapping) -> dict:
    """A RunConfig echo as the configuration digest reads it.

    Run length and cadence are dropped; each output-only switch is dropped
    or held at its default, as :data:`_DIGEST_DROPPED_DIAGNOSTIC_FIELDS`
    records.
    """
    values = {key: value for key, value in dict(config).items()
              if key not in CONFIG_RUN_LENGTH_FIELDS
              and key not in _DIGEST_DROPPED_DIAGNOSTIC_FIELDS}
    # New default-off options must not move existing checkpoint digests.
    _drop_default_off_run_keys(values)
    from woof.core.rrtmg_smoke_identity import bind_smoke_source_identity
    bind_smoke_source_identity(values)
    for key in CONFIG_DIAGNOSTIC_FIELDS - _DIGEST_DROPPED_DIAGNOSTIC_FIELDS:
        if key in values:
            values[key] = _run_config_default(key)
    if int(values.get("sf_urban_physics", 0) or 0) == 0:
        # With no urban model the three urban keys reach nothing, and a
        # digest that bound them would reject every checkpoint written
        # before they existed.  They join the digest exactly when an urban
        # model runs, so a resume across urban settings is still refused.
        for key in _URBAN_DIGEST_FIELDS:
            values.pop(key, None)
    elif int(values.get("sf_urban_physics", 0)) != 1:
        values.pop("urban_water_balance", None)
    elif values.get("urban_water_balance", "wrf_471") == "wrf_471":
        # An older header omitted this key and used the rain-positive clamps.
        values.pop("urban_water_balance", None)
    return values


#: The RunConfig keys of the urban canopy selector; see
#: :func:`_configuration_digest_values`.
_URBAN_DIGEST_FIELDS = ("sf_urban_physics", "use_wudapt_lcz", "num_urban_hi",
                        "urban_water_balance",
                        "urban_ucm_generation", "urban_treeoption", "urban_ttscheme", "urban_fvg", "urban_table_dir")


def _configuration_fingerprint(cfg) -> str:
    values = _configuration_digest_values(dataclasses.asdict(cfg))
    if values.get("use_adaptive_time_step"):
        # A LIVE dt is not part of what the run IS -- it is where the
        # controller happened to be at the instant the checkpoint was
        # written, and the next instant it is different.  Binding it here
        # would make every adaptive checkpoint resumable only by a run
        # that had adapted to the identical step, i.e. by nothing.
        #
        # `use_adaptive_time_step` stays IN, so a fixed-clock
        # fingerprint still binds dt exactly and a resume that flips the
        # feature is still refused.  The live value is carried separately
        # and restored -- it is state, not identity.
        #
        # AND SO IS time_step_sound, for exactly the same reason and one
        # step further out: upstream zeroes it whenever the adaptive
        # clock is on (start_em.F:966) so that solve_em derives the
        # acoustic substep count from the LIVE dt, and this port does the
        # same in adaptive_clock._apply via wrf_num_sound_steps.  It is
        # therefore a function of dt, not a setting: a checkpoint written
        # at dt ~ 76 s carries 6 where the experiment configured 4.  Under
        # a FIXED clock it stays bound, where it really is a setting.
        #
        # THE SAME ARGUMENT REACHES FURTHER, and this fingerprint is the
        # third gate it had to be made in.  `time_step_sound` is derived
        # from the live dt every root step, so it is state for the same
        # reason dt is; the controller's targets and clamps govern future
        # steps rather than describing the run, which is why the identity
        # walk above reports a change in them instead of refusing.  A
        # fingerprint that still bound either would refuse the resume the
        # walk had just allowed -- which is exactly what happened, with a
        # message that named no field at all.
        #
        # Popped from the two published sets rather than by name here, so
        # this cannot drift from the walk the way it just did.
        for name in ADAPTIVE_DERIVED_RUN_FIELDS | ADAPTIVE_POLICY_RUN_FIELDS:
            values.pop(name, None)
    return _json_sha256(_json_value(values, "RunConfig"))


def _thompson_table_identity(path) -> dict:
    """Validate and identify every external classic-Thompson table byte.

    The runtime table owner performs the stronger parse/upload/round-trip
    gate.  Restart identity deliberately validates the source assets again:
    it must reject a same-path byte replacement before mutating live state,
    without allocating the roughly 380 MiB host table payload merely to
    construct the header.
    """
    from woof.core.thompson_contract import (
        CLASSIC_TABLE_ASSETS,
        TABLE_SET_ID,
        WRF_REFERENCE_COMMIT,
        WRF_REFERENCE_VERSION,
        validate_table_assets,
    )

    assets = validate_table_assets(path)
    if assets != CLASSIC_TABLE_ASSETS:
        raise RestartManifestError(
            "validated Thompson table assets are not the canonical set")
    return {
        "schema": 1,
        "table_set": TABLE_SET_ID,
        "wrf_version": WRF_REFERENCE_VERSION,
        "wrf_commit": WRF_REFERENCE_COMMIT,
        "assets": [
            {"filename": item.filename, "bytes": int(item.bytes),
             "sha256": item.sha256}
            for item in assets
        ],
    }


def _ferrier_setup_identity() -> dict:
    """Resolved table identity for ``mp_physics=5``.

    Every Ferrier-Aligo rain and ice rate is a lookup into
    ETAMPNEW_DATA.expanded_rain (module_mp_fer_hires.F:2969-2983), so the
    bytes are re-validated here exactly as the adapter loads them and a
    resume onto a different table is refused before restore.
    """
    from woof.core.fer_hires import (TABLE_NAME, TABLE_SHA256,
                                      FerrierTableError, load_table,
                                      table_root)

    root = table_root()
    try:
        load_table(root)
    except FerrierTableError as exc:
        raise RestartManifestError(
            f"active Ferrier table identity is invalid at {root}") from exc
    return {"table": TABLE_NAME, "sha256": TABLE_SHA256}


def _p3_setup_identity() -> dict:
    """Resolved implementation/table identity for ``mp_physics=50``.

    The table root resolves exactly as the forecast adapter resolves it
    (:func:`woof.core.p3_tables.p3_table_root`: the packaged
    ``woof/data/p3/tables`` directory unless ``WOOF_P3_TABLE_ROOT``
    overrides it), and the bytes are re-validated here -- size AND
    SHA-256 -- so the identity written into a checkpoint names the table
    the trajectory actually loaded and a same-path byte replacement is
    refused before any live state is mutated.  P3's process rates ARE
    these tables: a resume onto different bytes is a different scheme.
    """
    from woof.core.p3_tables import (
        TABLE_1_2MOM_ASSET,
        TABLE_1_2MOM_VERSION,
        _validate_asset_bytes,
        p3_table_root,
    )

    root = p3_table_root()
    path = Path(root) / TABLE_1_2MOM_ASSET.filename
    try:
        _validate_asset_bytes(path, TABLE_1_2MOM_ASSET)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"active P3 table identity is invalid at {root}") from exc
    return {
        "schema": 1,
        "table_version": TABLE_1_2MOM_VERSION,
        "assets": [{
            "filename": TABLE_1_2MOM_ASSET.filename,
            "bytes": int(TABLE_1_2MOM_ASSET.size),
            "sha256": TABLE_1_2MOM_ASSET.sha256,
        }],
        # The rain fallspeed/ventilation tables are NOT an external asset:
        # p3_init generates them in process from the module's own math
        # (module_mp_p3.F:599-670), so their identity is the code identity
        # in MICROPHYSICS_ALGORITHM_IDENTITIES[50], not a file digest.
        "generated_rain_tables": "in-process-from-p3-init-v1",
        "itimestep_policy": "gpuwm-itimestep-seeded-from-model-clock-v1",
    }


def _thompson_setup_identity(cfg=None) -> dict:
    """Resolved implementation/table identity for ``mp_physics=8``.

    The table root resolves exactly as the forecast adapter resolves it
    (:func:`woof.physics_compat.thompson_table_root`: the packaged
    ``woof_data/data/thompson/tables`` directory unless
    ``WOOF_THOMPSON_TABLE_ROOT`` overrides it), so the identity written
    into a checkpoint names the bytes the trajectory actually loaded.
    The ``admission`` token replaces the retired
    ``WOOF_EXPERIMENTAL_THOMPSON_MP8=1`` ``implementation_guard`` entry
    (mp8 promotion to first-class, product/v1 packaging lane 2026-07-28):
    checkpoints written under the guarded runtime fail the physics-setup
    equality check rather than silently continuing across the promotion.
    """
    from woof.physics_compat import thompson_table_root

    if cfg is not None and getattr(cfg,"thompson_version","wrf_461") != "wrf_461":
        from woof.core.thompson_column import table_identity
        return table_identity(cfg)

    root = thompson_table_root()
    try:
        tables = _thompson_table_identity(root)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"active Thompson table identity is invalid at {root}") from exc
    return {
        "admission": "first-class-mp8-packaged-tables-v1",
        "tables": _json_value(tables, "Thompson table identity"),
        "graupel_number_policy": (
            "wrf-private-classic-ng-reconstructed-and-transported-per-call-v1"),
        "reflectivity_policy": (
            "wrf-v4.6.1-calc-refl10cm-post-fallout-output-only-v1"),
        "snow_rime_conversion_policy": (
            "wrf-v4.6.1-prs-scw-prg-scw-png-scw-held-number-v1"),
        "snow_fall_speed_policy": (
            "wrf-v4.6.1-deposition-conditioned-vts-boost-same-call-v1"),
    }


def _thompson_aerosol_setup_identity() -> dict:
    """Resolved implementation/table identity for ``mp_physics=28``.

    mp=8 has carried a ``thompson`` sub-record since it landed, and without
    the parallel record here an mp=28 checkpoint would bind NO table bytes
    at all: it would resume against a different ``freezeH2O.dat`` or a
    different ``CCN_ACTIVATE.BIN`` with every identity check passing.  The
    scheme is table-driven to an unusual degree -- the activation fraction
    that sets droplet number is READ from ``tnccn_act``, not computed
    (module_mp_thompson.F:5229-5230 index it at fixed radius/kappa) -- so a
    silent table substitution is a silent trajectory substitution.

    Two inventories, deliberately kept apart.  ``classic_tables`` is the
    SAME four-asset set mp=8 pins, because mp=28 genuinely loads them (its
    adapter calls ``load_classic_device_tables`` and reuses the frozen mp=8
    sedimentation and classic-graupel launchers unchanged).
    ``aerosol_tables`` is the one asset only mp=28 reads, addressed by the
    path the run resolved rather than by ``root / filename``: woof ships the
    blob since 2026-08-01 but the file and root overrides let a run bind to a
    copy in a WRF ``run/`` directory instead, so its LOCATION is not part of
    the identity but its BYTES are.  Neither is the fact that it ships --
    see the note beside the record; a packaging fact in a trajectory identity
    only buys refused resumes.

    The ``aerosol_source`` token records that this build's aerosol initial
    condition is ``thompson_init``'s synthetic CCN/IN profile
    (module_mp_thompson.F:493-551) when no aerosol dataset resolved.
    Since lane/wif-default the DEFAULT real-data mp=28 aerosol state is
    WRF's monthly WIF climatology, so a restart's aerosol fields are
    normally interpolated data and only fall back to the synthetic
    profile when the dataset was unavailable -- which the run receipt
    names (woof.config.MP28_AEROSOL_SYNTHETIC_FALLBACK).
    A future WIF metgrid ingest is a different initial condition and must
    move this token rather than resume onto a checkpoint written without it.
    """
    from woof.core.thompson_aerosol_contract import (
        AEROSOL_TABLE_SET_ID,
        resolve_aerosol_table_root,
        resolve_ccn_activation_path,
        validate_ccn_activation_asset,
    )
    from woof.physics_compat import thompson_table_root

    root = thompson_table_root()
    try:
        classic = _thompson_table_identity(root)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            f"active Thompson table identity is invalid at {root}") from exc
    try:
        # Resolve from the SAME root the adapter used
        # (woof/core/microphysics_aerosol.py:182 takes
        # ``physics_compat.thompson_table_root()`` and hands it to BOTH table
        # owners, so tnccn_act and the four classic caches cannot come from
        # different WRF builds).  Passing ``root`` explicitly rather than
        # letting the aerosol contract re-resolve it keeps that guarantee
        # even if the two resolution orders ever diverge.  The
        # ``WOOF_THOMPSON_CCN_ACTIVATE`` file override is still honoured,
        # because that is the path the adapter would have loaded too.
        ccn_path = resolve_ccn_activation_path(
            None, resolve_aerosol_table_root(root))
        ccn_asset = validate_ccn_activation_asset(ccn_path)
    except (OSError, TypeError, ValueError) as exc:
        raise RestartManifestError(
            "mp_physics=28 restart identity requires the resolved "
            f"CCN activation table: {exc}") from exc
    return {
        "admission": "component-override-mp28-unvendored-ccn-table-v1",
        "classic_tables": _json_value(
            classic, "Thompson table identity"),
        "aerosol_tables": {
            "schema": 1,
            "table_set": AEROSOL_TABLE_SET_ID,
            # DELIBERATELY ABSENT: ``AEROSOL_ASSET_REDISTRIBUTED``.  This
            # dict is hashed into ``physics_setup_fingerprint``, so anything
            # placed here becomes part of the trajectory identity and a
            # change to it refuses every earlier checkpoint.  Whether the
            # blob arrives inside the wheel or from a WRF ``run/`` directory
            # is a PACKAGING fact: it cannot move a single float.  It was
            # briefly written here on 2026-08-01 and removed the same day,
            # because flipping the constant False -> True would have broken
            # resume for every mp=28 checkpoint in existence while the bound
            # bytes stayed identical.  The ``sha256`` below is what actually
            # determines the trajectory and it is unaffected by delivery.
            # Do not re-add it; the constant is published on the registry row
            # (``redistributed_by_gpuwm``), which is where a packaging fact
            # belongs.
            "assets": [{
                "filename": ccn_asset.filename,
                "bytes": int(ccn_asset.bytes),
                "sha256": ccn_asset.sha256,
            }],
        },
        "aerosol_source": (
            "wrf-v4.6.1-thompson-init-synthetic-ccn-in-profile-"
            "aer-init-opt-0-wif-input-opt-0-v1"),
        "graupel_number_policy": (
            "wrf-private-classic-ng-reconstructed-and-transported-per-call-v1"),
        "reflectivity_policy": (
            "wrf-v4.6.1-calc-refl10cm-post-fallout-output-only-v1"),
        "aerosol_tendency_policy": (
            "wrf-v4.6.1-single-terminal-ncten-nwfaten-nifaten-apply-and-"
            "clamp-then-unclamped-surface-emission-v1"),
    }


from woof.ysu_generations import ysu_algorithm_identity


def physics_algorithm_identities(cfg) -> dict:
    """The ``algorithms`` component of :func:`physics_setup_identity`.

    A function of the configuration alone: the scheme selectors pick a row
    of each identity table.  Split out so the restart door can ask what
    THIS build calls the configuration a checkpoint recorded, before any
    state exists (:func:`_require_current_algorithm_identities`).
    """
    from woof.wrf_exact import ENABLED
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ((ra_lw_physics, ra_sw_physics) == (4, 4)
            and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY):
        # The legacy port shares WRF scheme id 4 with the RTE+RRTMGP
        # substitution but is a different algorithm: the SUMMARY
        # identities must be as distinct as the detailed ones, or a
        # legacy restart would advertise itself as rte-rrtmgp-v1 at the
        # top level (integration finding, 2026-07-27).
        lw_algorithm = RRTMG_LEGACY_LW_ALGORITHM_IDENTITY
        sw_algorithm = RRTMG_LEGACY_SW_ALGORITHM_IDENTITY
    elif (ra_lw_physics == ra_sw_physics
            and ra_lw_physics in RADIATION_ALGORITHM_IDENTITIES):
        lw_algorithm = sw_algorithm = _scheme_algorithm(
            RADIATION_ALGORITHM_IDENTITIES, ra_lw_physics, "radiation")
    else:
        lw_algorithm = _scheme_algorithm(
            LONGWAVE_ALGORITHM_IDENTITIES, ra_lw_physics,
            "longwave radiation")
        sw_algorithm = _scheme_algorithm(
            SHORTWAVE_ALGORITHM_IDENTITIES, ra_sw_physics,
            "shortwave radiation")
    algorithms = {
        "physics_driver": PHYSICS_DRIVER_ALGORITHM_IDENTITY,
        "microphysics": _scheme_algorithm(
            MICROPHYSICS_ALGORITHM_IDENTITIES, cfg.mp_physics,
            "microphysics"),
        "surface_layer": _scheme_algorithm(
            SURFACE_LAYER_ALGORITHM_IDENTITIES, cfg.sf_sfclay_physics,
            "surface layer"),
        "land_surface": _scheme_algorithm(
            LAND_SURFACE_ALGORITHM_IDENTITIES, cfg.sf_surface_physics,
            "land surface"),
        "pbl": (ysu_algorithm_identity(cfg.ysu_version, strict=ENABLED)
                if cfg.bl_pbl_physics == 1 else _scheme_algorithm(
                    PBL_ALGORITHM_IDENTITIES, cfg.bl_pbl_physics, "PBL")),
        "radiation": (lw_algorithm if lw_algorithm == sw_algorithm else
                      f"lw={lw_algorithm};sw={sw_algorithm}"),
        "radiation_lw": lw_algorithm,
        "radiation_sw": sw_algorithm,
        "cumulus": _scheme_algorithm(
            CUMULUS_ALGORITHM_IDENTITIES, cfg.cu_physics, "cumulus"),
    }
    if getattr(cfg, "solar_generation", "wrf_480") != "wrf_461":
        from woof.core.solar import solar_generation
        algorithms["solar"] = solar_generation(cfg.solar_generation)["algorithm"]
    if int(cfg.sf_surface_physics) == 4:
        from woof.noahmp_generation import snow_age_generation
        generation = snow_age_generation(getattr(cfg, "noahmp_snow_age", "wrf_470"))
        algorithms["land_surface"] = NOAHMP_SNOW_AGE_ALGORITHM_IDENTITIES[generation]
    if int(getattr(cfg, "sf_urban_physics", 0)) > 0:
        # Only when an urban model runs, so every existing header is
        # unchanged; a resume across urban models is then refused here as
        # well as by the configuration fingerprint.
        algorithms["urban"] = _scheme_algorithm(
            URBAN_ALGORITHM_IDENTITIES, cfg.sf_urban_physics, "urban")
        if int(cfg.sf_urban_physics) == 1:
            from woof.config import (URBAN_WATER_BALANCE_DEFAULT,
                                      URBAN_WATER_BALANCE_GENERATIONS)
            generation = getattr(cfg, "urban_water_balance", URBAN_WATER_BALANCE_DEFAULT)
            algorithms["urban"] = URBAN_WATER_BALANCE_GENERATIONS[generation]["algorithm"]
        from woof.checkpoint_identity import urban_algorithm_identity
        algorithms["urban"] = urban_algorithm_identity(cfg)
    if int(getattr(cfg, "sf_ocean_physics", 0)):
        # Only when the mixed layer runs, so every existing header is
        # unchanged; a resume across ocean models is refused here.
        from woof.surface_ocean import ocean_option
        ocean = ocean_option(cfg.sf_ocean_physics)
        algorithms["surface_ocean"] = (ocean["algorithm_identity"] + ":" +
            hashlib.sha256(json.dumps(ocean, sort_keys=True).encode()).hexdigest())
    if cfg.sf_surface_physics and getattr(cfg, "seaice_prognostic", False):
        algorithms["seaice"] = "zero-layer-snow-fractional-stefan-v1"
    mixing = dycore_mixing_identity(cfg)
    if mixing is not None:
        # Only when the dycore mixes (km_opt 2-4, an active km_opt = 1, or
        # sixth order), so a run without mixing keeps its pre-2.8.8 header
        # and resumes; a pre-2.8.8 checkpoint of a mixing run lacks the key
        # and is refused here before restore (checkpoint_identity.
        # DYCORE_MIXING_ALGORITHM_IDENTITY names what changed).
        algorithms["dycore_mixing"] = mixing
    damping = upper_damping_identity(cfg)
    if damping is not None:
        # Only when the implicit damper runs (damp_opt = 3), so a run
        # without it keeps its pre-2.8.8 header and resumes; a pre-2.8.8
        # checkpoint of a damped run lacks the key and is refused here
        # before restore (checkpoint_identity.
        # UPPER_DAMPING_ALGORITHM_IDENTITY names what changed).
        algorithms["upper_damping"] = damping
    return algorithms


def physics_setup_identity(state, cfg) -> dict:
    """Return the complete JSON-able trajectory-defining physics setup.

    The ordinary config echo pins all configured knobs.  This resolves the
    remaining runtime inputs that config alone cannot prove: callable
    implementation/policy, radiation calendar/grid/gases/ozone, packed Noah
    parameters, selected Morrison constants, resolved driver cadence, and
    the byte digests of every packaged table active on this trajectory.
    """
    driver = getattr(state, "physics", None)
    algorithms = physics_algorithm_identities(cfg)
    microphysics = {"scheme_id": int(cfg.mp_physics)}
    if int(cfg.mp_physics) == 6:
        from woof.core.wsm6_constants import rimed_ice_constants
        selection = int(cfg.wsm6_hail_opt)
        constants = rimed_ice_constants(selection)
        microphysics["wsm6_rimed_ice"] = {
            "selection": selection,
            "n0g": float(constants.n0g),
            "deng": float(constants.deng),
            "avtg": float(constants.avtg),
            "bvtg": float(constants.bvtg),
            "lamdagmax": float(constants.lamdagmax),
        }
    if int(cfg.mp_physics) == 10:
        from woof.core.morrison_constants import rimed_ice_constants
        selection = int(cfg.morr_rimed_ice)
        constants = rimed_ice_constants(selection)
        microphysics["morrison_rimed_ice"] = {
            "selection": selection,
            "ag": float(constants.ag),
            "bg": float(constants.bg),
            "rhog": float(constants.rhog),
            "cg": float(constants.cg),
        }
    if int(cfg.mp_physics) == 8:
        microphysics["thompson"] = _thompson_setup_identity(cfg)
    if int(cfg.mp_physics) == 28:
        microphysics["thompson_aerosol"] = _thompson_aerosol_setup_identity()
    if int(cfg.mp_physics) == 18:
        microphysics["restart_contract"] = \
            _nssl2_restart_contract_identity(cfg)
    if int(cfg.mp_physics) == 50:
        microphysics["p3"] = _p3_setup_identity()
    if int(cfg.mp_physics) == 5:
        microphysics["ferrier"] = _ferrier_setup_identity()

    land_surface = {
        "scheme_id": int(cfg.sf_surface_physics),
        "parameters": None,
    }
    if int(cfg.sf_surface_physics) != 0:
        if driver is None:
            raise RestartManifestError(
                "active land-surface scheme cannot be restart-identified "
                "without an attached PhysicsDriver")
        land_surface["parameters"] = _land_surface_parameters_identity(
            cfg, driver)

    cumulus = {
        "scheme_id": int(cfg.cu_physics),
        "callable": None,
        "coefficient_table": None,
    }
    if int(cfg.cu_physics) != 0:
        if driver is None:
            raise RestartManifestError(
                "active cumulus cannot be restart-identified without an "
                "attached PhysicsDriver")
        # The stock class comes from the REGISTRY's row for the scheme
        # (``consumers.stock_callable_class``), and a scheme with no row is
        # refused HERE by name -- "no stock-class row" -- instead of being
        # sent down the custom-adapter path and refused for lacking a
        # restart_identity the stock class never had, which is the message
        # that would send a reader looking in the wrong place.  Plan
        # review (woof.physics_registry.consumer_row_gaps) asks the same
        # question before step 0, so this is the backstop.
        expected = _stock_callable_class(
            "cumulus", {"cu_physics": int(cfg.cu_physics)}, label="cumulus")
        callable_identity = _callable_setup_identity(
            driver.cumulus_callable, label="cumulus",
            expected_class=expected)
        cumulus["callable"] = callable_identity
        if (callable_identity["implementation"] == "stock"
                and int(cfg.cu_physics) == 1):
            # The coefficient table is a KF asset; GF ships none (its
            # constants are pinned words inside the kernel source, which
            # the algorithm identity string already binds).
            from woof.core.kf import load_kf_table
            cumulus["coefficient_table"] = \
                _resolved_object_setup_identity(
                    load_kf_table(), "Kain-Fritsch lookup table")

    driver_identity = {
        "attached": driver is not None,
        "class": None if driver is None else _callable_class_name(driver),
        "resolved_schemes": None,
        "cadence": None,
    }
    if driver is not None:
        driver_identity["resolved_schemes"] = {
            "mp_physics": int(driver.mp_physics),
            "ra_physics": int(driver.ra_physics),
            "ra_lw_physics": int(driver.ra_lw_physics),
            "ra_sw_physics": int(driver.ra_sw_physics),
            "radiation_active": bool(driver.radiation_active),
            "cu_physics": int(driver.cu_physics),
            "surface_enabled": bool(driver.surface_enabled),
        }
        cadence = {}
        # Under an adaptive clock the STEP COUNTS and the seconds derived
        # from them move with dt, so they describe where the run had got
        # to rather than what it is.  The MINUTES -- what the namelist
        # asked for -- stay bound in both cases, which is the part that
        # says "radiation every 12 minutes" and must not change across a
        # resume.
        derived = ("radt_seconds", "stepra", "cudt_seconds", "stepcu",
                   "bldt_seconds", "stepbl")
        adaptive = bool(getattr(cfg, "use_adaptive_time_step", False))
        for name in ("bldt_seconds", "stepbl", "radt_minutes",
                     "radt_seconds", "stepra", "cudt_minutes",
                     "cudt_seconds", "stepcu"):
            if adaptive and name in derived:
                continue
            cadence[name] = _json_value(
                getattr(driver, name), f"PhysicsDriver.{name}")
        driver_identity["cadence"] = cadence
        if int(getattr(cfg, "sf_surface_mosaic", 0)) == 1:
            from woof.checkpoint_identity import NOAH_MOSAIC_ALGORITHM_IDENTITY
            mosaic = getattr(driver, "noah_mosaic", None)
            if mosaic is None:
                raise ValueError("mosaic checkpoint has no tile setup; resuming "
                                 "would integrate the dominant category")
            # Strict JSON (the LCZ tuple becomes the list a stored header
            # reads back as), so a resumed identity compares equal.
            driver_identity["noah_mosaic"] = _json_value({
                "algorithm": NOAH_MOSAIC_ALGORITHM_IDENTITY,
                "mosaic_cat": mosaic.mosaic_cat,
                "categories": dataclasses.asdict(mosaic.categories),
                "xice_threshold": mosaic.xice_threshold,
            }, "PhysicsDriver.noah_mosaic")
        owner = getattr(driver, "cam_ozone", None)
        if owner is not None and not (4 in radiation_scheme_ids(cfg)
                and rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY and cfg.o3input == 2):
            driver_identity["cam_ozone"] = {
                "producer": owner.restart_identity,
                "latitude": _array_setup_identity(owner.latitude_deg),
                "longitude": _array_setup_identity(owner.longitude_deg),
            }
        fire = getattr(driver, "fire", None)
        if fire is not None:
            driver_identity["fire"] = _json_value(fire.setup_identity(), "SFIRE setup identity")

    identity = {
        "schema_version": PHYSICS_SETUP_SCHEMA_VERSION,
        "configuration_sha256": _configuration_fingerprint(cfg),
        "algorithms": algorithms,
        "driver": driver_identity,
        "microphysics": microphysics,
        "radiation": _radiation_setup_identity(driver, cfg),
        "land_surface": land_surface,
        "cumulus": cumulus,
        "assets": _active_asset_identity(cfg, driver),
    }
    if getattr(driver, "eclipse", None) is not None:
        from woof.eclipse_windows import _CUDA
        identity["eclipse"] = {"policy": "wrf461-besselian-current-utc-per-step-v1",
                               "kernel_sha256": hashlib.sha256(_CUDA.encode()).hexdigest()}
    urban = getattr(driver, "urban", None)
    if urban is not None and (getattr(cfg, "urban_ucm_generation", "wrf_471") == "wrf_480" or getattr(cfg, "urban_table_dir", "")):
        from woof.core.urban_ucm import pack_params
        identity["urban_parameters"] = _packed_parameters_identity(pack_params(urban.params), label="urban canopy", attribute="urban.params")
        identity["urban_parameters"]["table_sha256"] = urban.params.sha256
    # Literal-only sets have no table bundle that could otherwise bind
    # their constants. Every checkpoint writer uses this setup identity.
    # No set adds no key, preserving all earlier default setup bytes.
    from woof.physics_params import receipt as parameter_receipt
    parameters = parameter_receipt()
    if parameters is not None:
        identity["physics_params"] = parameters
    return identity


def physics_setup_fingerprint(state, cfg) -> str:
    """SHA-256 of :func:`physics_setup_identity`."""
    return _json_sha256(physics_setup_identity(state, cfg))


def ask_checkpoint_physics_identity(state, cfg) -> None:
    """Resolve one domain's checkpoint physics identity, and discard it.

    THE PLACEMENT IS THE POINT (audit R-046).  A run that will write
    checkpoints must be able to NAME its physics setup, and until R-046 the
    first thing that asked was the writer, at the first restart interval,
    with the forecast to that point already spent.  Plan review answers the
    config half (``woof.physics_registry.require_consumer_rows``, from
    ``validate_run_config``); the DRIVER half -- a physics callable whose
    class no stock-class row can bind, a land-surface parameter bundle that
    cannot be digested -- needs the constructed driver, so the earliest it
    can be asked is once a domain has one.  The identity value is thrown
    away: what is bought is the refusal's placement.

    ASKED OF A :class:`~woof.core.physics.PhysicsDriver`, and of nothing
    else.  There is exactly one production attach site --
    ``woof/core/physics.py``'s ``state.physics = driver`` -- so a state
    carrying some other object is a route this identity is not defined
    over: it reads a driver's resolved scheme ids and resolved cadence, and
    an object of another class fails on an ATTRIBUTE, naming this gate
    instead of the route that attached the object.  Such an object meets
    the restart writer at its own door, unchanged by this gate.  A domain
    with NO driver is still asked: an active cumulus scheme without an
    attached driver is exactly one of the setups a checkpoint cannot name.

    Both forecast doors call THIS function rather than each writing the
    rule out: ``woof.core.model.execute_experiment`` (once per domain,
    parent-first, before step 0 -- and at activation for a domain that
    joins the tree after the run started) and
    ``woof.runtime.integrate_prepared_case``.  They had diverged -- the
    single-domain door asked unconditionally, so the foreign object the
    tree door deliberately skips raised there on an attribute -- which is
    the sort of difference that only shows up in whichever door the user
    happens to run.
    """
    from woof.core.physics import PhysicsDriver

    driver = getattr(state, "physics", None)
    if driver is not None and not isinstance(driver, PhysicsDriver):
        return
    physics_setup_identity(state, cfg)


def _callable_state_check(scheme, allowed_arrays: frozenset,
                          allowed_containers: frozenset,
                          label: str) -> None:
    """Enforce manifest coverage for arrays on a scheme callable.

    Covers direct array attributes, dict-valued attributes (every value),
    and one level of object containers, so an adapter stashing per-call
    state in a dict or sub-object (the ``driver.cu_rates`` pattern) is
    caught instead of silently skipping the stream.  Containers that
    legitimately carry arrays must be classified in the ``*_CONTAINERS``
    allowlists above.
    """
    for name, value in getattr(scheme, "__dict__", {}).items():
        if _is_array_like(value):
            if name not in allowed_arrays:
                raise RestartManifestError(
                    f"{label} callable attribute {name!r} is an "
                    "unclassified array: add it to the restart manifest "
                    "(woof/io/restart.py) as serialized or setup state")
            continue
        if name in allowed_containers:
            continue
        if isinstance(value, dict):
            arrays = sorted(str(key) for key, item in value.items()
                            if _is_array_like(item))
            if arrays:
                raise RestartManifestError(
                    f"{label} callable dict attribute {name!r} carries "
                    f"unclassified arrays {arrays}: classify the container "
                    "in woof/io/restart.py or serialize its state")
            continue
        nested = getattr(value, "__dict__", None)
        if nested and any(_is_array_like(item) for item in nested.values()):
            raise RestartManifestError(
                f"{label} callable attribute {name!r} is an object "
                "container carrying unclassified arrays: classify it in "
                "woof/io/restart.py (rebuild-on-load) or serialize its "
                "state")


def _require_dataclass_components(container, expected, label: str) -> None:
    """Pin a serialized dataclass's fields to its component manifest.

    A field added to :class:`PhysicsTendencies` or
    :class:`MicrophysicsDiagnostics` without a manifest update would
    silently serialize nothing; this makes every write (and the CPU
    manifest tests) fail instead.
    """
    names = {field.name for field in dataclasses.fields(container)}
    expected = set(expected)
    if names != expected:
        raise RestartManifestError(
            f"{label} dataclass fields do not match the restart component "
            f"manifest (woof/io/restart.py): unclassified "
            f"{sorted(names - expected)}, stale {sorted(expected - names)}")


def state_manifest(state) -> dict[str, object]:
    """Serialized ``state/<name>`` arrays for this state's configuration.

    Walks EVERY instance attribute through :func:`classify_state_attr`, so
    an unclassified attribute raises here (and therefore in every
    ``write_restart`` call and in the manifest tests).
    """
    manifest = {}
    for name in sorted(vars(state)):
        kind = classify_state_attr(name)
        value = getattr(state, name)
        if kind == "serialize" and value is not None:
            manifest[f"state/{name}"] = value
        elif kind == "checkpoint_only" and value is not None:
            # Its own namespace, so the `state/` key-set comparison on
            # restore neither expects nor rejects it.
            manifest[f"acoustic/{name}"] = value
    return manifest


def _scratch_manifest(state) -> dict[str, object]:
    pool = getattr(state, "_scratch", {})
    manifest = {}
    for slot in sorted(pool):
        if (classify_scratch_slot(slot) == "serialize"
                or slot in CHECKPOINTED_CARRIED_SCRATCH_SLOTS):
            manifest[f"scratch/{slot}"] = pool[slot]
    return manifest


def _opted_in_scratch_manifest(state, slots) -> dict[str, object]:
    """``{scratch/<slot>: array}`` for slots the WRITER asked to serialize.

    The classification is NOT changed and must not be: a tracker window is
    ``carry`` for every other consumer of :func:`classify_scratch_slot`,
    above all ``tilestream.physics_inventory.carrier_manifest``, and
    promoting it to ``serialize`` would put a window in every checkpoint of
    every run -- including runs with no consumer to read one, whose
    checkpoints would then differ from every checkpoint they have on disk.
    A nest-lifecycle run opts its own windows in, per member, here.
    It also opts in named held output scratch: the last microphysics-time
    reflectivity volume can be consumed by a tracker before the next step.
    """
    pool = getattr(state, "_scratch", {})
    manifest: dict[str, object] = {}
    for slot in sorted(set(slots)):
        if (classify_scratch_slot(slot) != "carry"
                and slot not in LIFECYCLE_HELD_SCRATCH_SLOTS):
            raise RestartManifestError(
                f"the checkpoint writer opted scratch slot {slot!r} in, but "
                f"it classifies as {classify_scratch_slot(slot)!r}: a "
                "serialized slot is already in every manifest and a rebuilt "
                "one is a per-call work buffer whose value between calls is "
                "not state, so writing it would checkpoint a number nothing "
                "may read back")
        array = pool.get(slot)
        if array is None:
            raise RestartManifestError(
                f"the checkpoint writer opted scratch slot {slot!r} in, but "
                "this domain never allocated it; a member written from "
                "nothing would restore a plane the run does not have")
        manifest[f"scratch/{slot}"] = array
    return manifest


def carried_scratch_manifest(state) -> dict[str, object]:
    """``{scratch/<slot>: array}`` for the CARRY class -- never for a file.

    Deliberately a second function rather than a flag on
    :func:`_scratch_manifest`: nothing that writes or validates a checkpoint
    may ever pick these up by accident, and the way to guarantee that is that
    the checkpoint path cannot reach them.  The one caller is
    :func:`tilestream.physics_inventory.carrier_manifest`, which needs the
    cross-step set, not the checkpointed one.
    """
    pool = getattr(state, "_scratch", {})
    return {f"scratch/{slot}": pool[slot] for slot in sorted(pool)
            if classify_scratch_slot(slot) == "carry"}



def _any_on_owning_device(array) -> bool:
    """``bool(array.any())`` EVALUATED ON THE CARD THE ARRAY LIVES ON.

    A CuPy reduction launches on the CURRENT device, so ``array.any()`` on an
    array that lives on another card needs peer access -- and on a GeForce box
    there is none (measured on 4x RTX 5080: ``deviceCanAccessPeer`` is 0 for
    all twelve ordered pairs), so it raises instead of returning an answer.

    This one call is on the path every inventory takes:
    ``tilestream.physics_inventory.carrier_inventory`` ->
    ``carrier_manifest`` -> ``_driver_manifest``.  ``MultiGPUDomain`` builds
    each rank's inventory inside a loop whose current device is whatever the
    last iteration left, so asking rank 1 for its carriers with device 0
    current died here -- an INVENTORY call, which reads no data and should
    not care which card is current, brought the run down.  Entering the
    array's own device costs one context switch per checkpoint-manifest call
    and makes the function device-independent, which is what its callers
    already assume.
    """
    device = getattr(array, "device", None)
    if not hasattr(device, "__enter__"):
        # Host arrays take the plain reduction.  The test is "is this a
        # device CONTEXT", not "is there a device attribute": NumPy 2
        # gave every ndarray a ``.device`` -- the STRING ``"cpu"`` (array
        # API standard) -- so ``device is None`` stopped meaning "host
        # array" and ``with "cpu":`` is a TypeError.  A CuPy array's
        # ``.device`` is a ``cupy.cuda.Device``, a context manager, and
        # keeps the owning-card entry below.
        return bool(array.any())
    with device:
        return bool(array.any())


def pbl_raw_manifest(driver) -> dict[str, object]:
    """Canonical restart names, sharing GF's pair without duplicate payloads."""
    from woof.core.physics_inventory import PBL_SHARED_FORCING

    manifest = {}
    for name, value in getattr(driver, "pbl_raw_rates", {}).items():
        shared = PBL_SHARED_FORCING.get(name)
        key = (f"held/{shared}" if shared and value is getattr(driver, shared, None)
               else f"pbl/{name}")
        manifest[key] = value
    return manifest


PBL_DIAGNOSTIC_GROUPS = {
    "sase_flux_diag": ("fqv_vent", "fqv_diff", "fth_vent", "fth_diff"),
    "hmix_k_diag": ("SASE_KMH", "SASE_KHH"),
}


def pbl_diagnostic_manifest(driver) -> dict[str, object]:
    """Optional output fields whose PBL producer can skip a model step.

    The default every-step path still rebuilds these output-only values.
    Positive cadence carries the last due-call value across tiles/restart.
    """
    if not getattr(driver, "sase_active", False) or not getattr(driver, "pbl_raw_rates", {}):
        return {}
    return {f"pbl/diagnostics/{group}/{name}": value
            for group in PBL_DIAGNOSTIC_GROUPS
            for name, value in (getattr(driver, group, None) or {}).items()}


def _validate_pbl_diagnostics(stored, header, state, driver) -> None:
    prefix = "pbl/diagnostics/"
    known = {f"{prefix}{group}/{name}": group
             for group, names in PBL_DIAGNOSTIC_GROUPS.items() for name in names}
    for key in sorted(k for k in stored if k.startswith(prefix)):
        if key not in known:
            raise RestartMismatchError(f"restart carries unknown PBL diagnostic {key}")
        target = state.w if known[key] == "sase_flux_diag" else state.p
        _check_array(stored[key], target, key)
    for key in pbl_diagnostic_manifest(driver):
        # Every group here is an output-only toggle from
        # CONFIG_DIAGNOSTIC_FIELDS, which the config walk lets a resume
        # switch on; zero is then its cold value until the next due PBL
        # step refills it.  A group the checkpoint had ON must still be
        # carried, or the first off-cadence frame would publish zeros.
        group = known[key]
        newly_enabled = (group in CONFIG_DIAGNOSTIC_FIELDS
                         and not header["config"].get(group, False))
        if key not in stored and not newly_enabled:
            raise RestartMismatchError(f"restart is missing held PBL diagnostic {key}")


def _driver_manifest(driver) -> dict[str, object]:
    """Serialized driver arrays; enforces driver attribute coverage."""
    for name in sorted(vars(driver)):
        if (name not in DRIVER_SERIALIZED_ATTRS
                and name not in DRIVER_REBUILT_ATTRS
                and name not in DRIVER_CHECKPOINT_ONLY_ATTRS):
            raise RestartManifestError(
                f"PhysicsDriver attribute {name!r} is not classified in "
                "the restart manifest (woof/io/restart.py): declare it "
                "serialized, rebuilt, or checkpoint-only")
    # Normal compute() finalizes the transient KF expiry mask immediately
    # after composing this step's RK target and before Morrison.  A surviving
    # mask therefore denotes an incomplete synthetic/direct-driver transition;
    # reject it instead of serializing a state whose persistent rates and held
    # coupled tendencies have not yet received their required simultaneous
    # clear.  ``cu_expiring`` itself remains rebuild-only scratch.
    cu_expiring = getattr(driver, "cu_expiring", None)
    if (bool(getattr(driver, "_cu_expiry_pending", False))
            or (cu_expiring is not None and _any_on_owning_device(cu_expiring))):
        raise RestartManifestError(
            "cannot write restart while KF expiry finalization is pending; "
            "call PhysicsDriver.finish_step() before checkpointing")
    manifest = {
        "driver/rthratenlw": driver.rthratenlw,
        "driver/rthratensw": driver.rthratensw,
        "driver/pending_rainbl": driver._pending_rainbl,
    }
    # The checkpoint-only carriers, in their own namespace so the
    # driver/fields/cumulus key sets are untouched and a checkpoint
    # written before this existed stays restorable.  ``None`` is a real
    # answer -- the buffer exists only when the attached longwave scheme
    # declares a TOA flux -- and an absent key restores as the zeros that
    # build would have had anyway.
    for name in sorted(DRIVER_CHECKPOINT_ONLY_ATTRS):
        value = getattr(driver, name, None)
        if value is not None:
            manifest[f"diag/{name}"] = value
    for name in sorted(DRIVER_HELD_FORCING_ATTRS):
        value = getattr(driver, name, None)
        if value is not None:
            manifest[f"held/{name}"] = value
    manifest.update(pbl_raw_manifest(driver))
    manifest.update(pbl_diagnostic_manifest(driver))
    manifest.update(_fire_manifest(driver))
    manifest.update(_hydro_manifest(driver))
    for tend_name in _live_driver_tendency_attrs(driver):
        tend = getattr(driver, tend_name)
        _require_dataclass_components(
            tend, TENDENCY_COMPONENTS, f"{tend_name} (PhysicsTendencies)")
        for comp in TENDENCY_COMPONENTS:
            value = getattr(tend, comp)
            if value is not None:
                manifest[f"driver/{tend_name}/{comp}"] = value
    _require_dataclass_components(
        driver.microphysics, MICROPHYSICS_COMPONENTS,
        "MicrophysicsDiagnostics")
    from woof.core.physics import microphysics_scratch_slots
    micro_slots = dict(microphysics_scratch_slots(driver.mp_physics))
    for comp, slot in micro_slots.items():
        scratch = getattr(driver.state, "_scratch", {}).get(slot)
        if scratch is None or getattr(driver.microphysics, comp) is not scratch:
            raise RestartManifestError(
                f"driver.microphysics.{comp} does not alias canonical scratch "
                f"slot {slot!r}: restart v{RESTART_FORMAT_VERSION} writes "
                "exactly one microphysics accumulator set")
    for comp in set(MICROPHYSICS_COMPONENTS) - set(micro_slots):
        if driver.mp_physics and getattr(driver.microphysics, comp) is not None:
            raise RestartManifestError(
                f"driver.microphysics.{comp} is populated but has no canonical "
                f"scratch slot for mp_physics={driver.mp_physics}")
    if "_pwp_status" in driver.fields:
        from woof.ocean.mixed_layer import check
        check(driver.fields)
    for name in sorted(driver.fields):
        manifest[f"fields/{name}"] = driver.fields[name]
    # The in-place fields restore relies on SFClayResult aliasing the
    # fields-dict arrays; verify the alias contract (and that every
    # SFClayResult field IS a fields entry) at every write.
    for field in (() if driver.sfclay_result is None
                  else dataclasses.fields(driver.sfclay_result)):
        if driver.fields.get(field.name) is not getattr(
                driver.sfclay_result, field.name):
            raise RestartManifestError(
                f"sfclay_result.{field.name} does not alias "
                f"fields[{field.name!r}]: the in-place surface-field "
                "restore depends on that aliasing")
    for attribute, suffix in (
            ("mynn_sfclay_result", ""),
            ("mynn_sfclay_sea_result", "_sea")):
        result = getattr(driver, attribute)
        for field in (() if result is None else dataclasses.fields(result)):
            key = f"{field.name}{suffix}"
            if driver.fields.get(key) is not getattr(result, field.name):
                raise RestartManifestError(
                    f"{attribute}.{field.name} does not alias fields[{key!r}]: "
                    "the in-place MYNN surface-field restore depends on that "
                    "aliasing")
    from woof.core.radiation_composition import radiation_adapters
    # Nested engines keep the same explicit inventory checks as standalone
    # engines; composition cannot hide an unclassified array-bearing member.
    for adapter in radiation_adapters(driver.radiation_callable):
        if adapter is not driver.radiation_callable:
            _callable_state_check(adapter, RADIATION_CALLABLE_ARRAYS,
                                  RADIATION_CALLABLE_CONTAINERS, "radiation component")
    _callable_state_check(driver.radiation_callable,
                          RADIATION_CALLABLE_ARRAYS,
                          RADIATION_CALLABLE_CONTAINERS, "radiation")
    _callable_state_check(driver.cumulus_callable,
                          CUMULUS_CALLABLE_ARRAYS | CUMULUS_REBUILT_ARRAYS,
                          CUMULUS_CALLABLE_CONTAINERS, "cumulus")
    w0avg = getattr(driver.cumulus_callable, "w0avg", None)
    if w0avg is not None:
        manifest["cumulus/w0avg"] = w0avg
    o33d_grid = radiation_o33d_carrier(driver)
    if o33d_grid is not None:
        # Legacy-RRTMG root o33d field (WRF's restart-carried O3RAD
        # analogue): serialized so child-domain ozone routing resumes
        # bit-identically (see RADIATION_CALLABLE_ARRAYS note).
        manifest["radiation/o33d_grid"] = o33d_grid
    return manifest


def _live_driver_tendency_attrs(driver):
    return (*DRIVER_TENDENCY_ATTRS, *(name for name in OPTIONAL_DRIVER_TENDENCY_ATTRS
             if getattr(driver, name, None) is not None))


def _hydro_manifest(driver):
    hydro = getattr(driver, "hydro", None)
    return {} if hydro is None else {f"hydro/{key}": value for key, value in hydro.arrays.items()}


def _validate_hydro_payload(stored, header, driver):
    hydro = getattr(driver, "hydro", None)
    incoming = {key[6:]: value for key, value in stored.items() if key.startswith("hydro/")}
    metadata = (header.get("driver") or {}).get("hydro")
    if hydro is None:
        if incoming or metadata is not None:
            raise RestartMismatchError("restart carries NextGen state but the parent has no hydro driver")
        return
    try:
        hydro.validate_checkpoint(metadata, incoming)
    except (KeyError, TypeError, ValueError) as exc:
        raise RestartMismatchError(f"NextGen restart cannot continue: {exc}") from exc


def _fire_manifest(driver):
    """All native fire state, including halos, stages and moisture clocks."""
    fire = getattr(driver, "fire", None)
    if fire is None:
        return {}
    arrays = fire.arrays()
    if not isinstance(arrays, Mapping) or not arrays:
        raise RestartManifestError("SFIRE checkpoint has no fire arrays; resuming would lose the evolving fire")
    result = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or "/" in name or not name:
            raise RestartManifestError("SFIRE restart field names must be nonempty flat identifiers")
        if not _is_array_like(value) or np.dtype(value.dtype) not in (np.dtype(np.float32),np.dtype(np.int32)):
            raise RestartManifestError(f"SFIRE restart field {name!r} must carry native float32 or int32 words")
        result[f"fire/{name}"] = value
    return result


def _fire_header(driver):
    fire = getattr(driver, "fire", None)
    if fire is None:
        return {}
    identity = _json_value(fire.setup_identity(), "SFIRE setup identity")
    metadata = _json_value(fire.metadata(), "SFIRE continuation metadata")
    if not isinstance(identity, dict) or not isinstance(metadata, dict):
        raise RestartManifestError("SFIRE restart setup and continuation metadata must be mappings")
    return {"fire": metadata, "fire_setup_identity": identity,
            "fire_setup_fingerprint": _json_sha256(identity)}


def _validate_fire_payload(stored, header, driver):
    """Check complete fire inventory and static setup before any mutation."""
    fire = getattr(driver, "fire", None)
    incoming = {key: value for key, value in stored.items() if key.startswith("fire/")}
    driver_header = header.get("driver")
    driver_header = driver_header if isinstance(driver_header, dict) else {}
    metadata = driver_header.get("fire")
    if fire is None:
        if incoming or metadata is not None:
            raise RestartMismatchError("restart carries SFIRE state but the resuming driver has no coupled fire model")
        return
    expected = _fire_manifest(driver)
    if set(incoming) != set(expected):
        raise RestartMismatchError(
            "restart fire-state inventory differs from the resuming model "
            f"(missing {sorted(set(expected)-set(incoming))}, extra {sorted(set(incoming)-set(expected))}); "
            "missing fire stages or moisture fields change the resumed spread")
    for key, target in expected.items():
        _check_array(incoming[key], target, key)
    if not isinstance(metadata, dict):
        raise RestartMismatchError("restart is missing SFIRE continuation metadata and clocks")
    identity = driver_header.get("fire_setup_identity")
    fingerprint = driver_header.get("fire_setup_fingerprint")
    live = _json_value(fire.setup_identity(), "SFIRE setup identity")
    if (not isinstance(identity, dict) or fingerprint != _json_sha256(identity)
            or identity != live):
        raise RestartMismatchError("restart SFIRE static data, refinement, fuel table or ignition setup differs; resuming would change the fire trajectory")
    validator = getattr(fire, "validate_restart", None)
    if validator is not None:
        try:
            validator({key[5:]: value for key, value in incoming.items()}, metadata)
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartMismatchError("restart SFIRE continuation metadata is malformed") from exc


def radiation_o33d_carrier(driver):
    """The array a manifest records as ``radiation/o33d_grid``, without a copy.

    ``PhysicsDriver.o3rad`` when a tree owns one; otherwise the legacy
    adapter's retained field, taken on the device through ``o33d_device()``
    when the adapter keeps it there.  Reading the ``_o33d_grid`` property
    instead materializes a host copy of the whole field.  Named breakage
    (box G, 2026-10-05, 1 km HRRR physics): every streamed sweep's
    inventory runs this manifest on the forecast's main thread and then
    drops the slot as restart-only, so each radiation period paid a full
    device-to-host copy for nothing, and under host memory pressure one of
    those copies held step 551 for 27 s with both cards idle.  A restart
    writer converts the device array exactly as it converts every state
    field (``_host``), so the bytes written are unchanged.
    """
    carrier = getattr(driver, "o3rad", None)
    if carrier is not None:
        return carrier
    adapter = driver.radiation_callable
    on_device = getattr(adapter, "o33d_device", None)
    if callable(on_device):
        carrier = on_device()
        if carrier is not None:
            return carrier
    return getattr(adapter, "_o33d_grid", None)


def root_external_lbc_clock_identity(state, cfg) -> str | None:
    """The root external-LBC clock semantic active on this state.

    ``None`` for domains without an external Davies consumer (nested
    children on rolling mirrors, periodic/open cases).  For specified
    domains: :data:`ROOT_EXTERNAL_LBC_CLOCK_IDENTITY` when the attached
    external mirror is bound to a DomainClock (production tree build /
    N5S builder), :data:`ROOT_EXTERNAL_LBC_CLOCK_LEGACY` otherwise
    (legacy direct paths, or pre-attachment shim states).
    """
    if not getattr(cfg, "specified", False):
        return None
    resident = getattr(state, "_lateral_boundary_device", None)
    bound = (resident is not None and not getattr(resident, "rolling", False)
             and getattr(resident, "clock", None) is not None)
    return (ROOT_EXTERNAL_LBC_CLOCK_IDENTITY if bound
            else ROOT_EXTERNAL_LBC_CLOCK_LEGACY)


def write_restart(path, state, cfg, *, run_trackers=None,
                  tree_header: dict | None = None,
                  extra_scratch_slots=(),
                  sealed_forcing_extension: bool = False,
                  preserved_forcing_prefix: bool = False) -> Path:
    """Serialize the complete cross-step model state to ``path``.

    ``run_trackers`` (optional JSON-able dict) carries the caller's
    run-summary bookkeeping (w-max trackers, SWDOWN peak, nan flag) so a
    resumed run reports the same summary as an uninterrupted one â€” model
    evolution itself never reads them.

    ``extra_scratch_slots`` names CARRIED slots this caller wants in the
    file beside the serialized set (:func:`_opted_in_scratch_manifest`).
    Empty â€” the default â€” writes exactly the member set this function has
    always written, which is what keeps every lifecycle-free checkpoint
    byte-identical.
    """
    with perf_timing.stage("io.restart.write_restart"):
        return _write_restart(
            path, state, cfg, run_trackers=run_trackers,
            tree_header=tree_header,
            extra_scratch_slots=extra_scratch_slots,
            sealed_forcing_extension=sealed_forcing_extension,
            preserved_forcing_prefix=preserved_forcing_prefix)


def prepare_restart(path, state, cfg, *, run_trackers=None,
                    tree_header: dict | None = None,
                    extra_scratch_slots=(),
                    sealed_forcing_extension: bool = False,
                    preserved_forcing_prefix: bool = False,
                    staging=None):
    """Snapshot what :func:`write_restart` would write; publish it later.

    Every validation, the header and the host copy of every array happen
    HERE, on the caller's thread, exactly as :func:`write_restart` does
    them.  The returned callable performs only the publication (``np.savez``
    to a unique temporary, ``fsync``, atomic rename, directory ``fsync``)
    and returns the path, so a checkpoint writer thread can run it while
    the model steps on.  The snapshot OWNS its arrays: a device array is
    copied to the host (as the synchronous writer always did) and a host
    array is copied too, so a later step cannot change what is written.
    The file is the one :func:`write_restart` writes, byte for byte, apart
    from the ``created`` stamp both writers take from the wall clock.

    ``staging`` (a :class:`woof.io.checkpoint_staging.StagedSnapshot`)
    takes the device arrays' host copies asynchronously, into pinned
    memory, on the model's stream; the returned callable waits for those
    copies before it writes a byte.  Without it every array is copied
    synchronously here, as before.
    """
    with perf_timing.stage("io.restart.prepare_restart"):
        path, payload = _restart_payload(
            path, state, cfg, run_trackers=run_trackers,
            tree_header=tree_header,
            extra_scratch_slots=extra_scratch_slots,
            sealed_forcing_extension=sealed_forcing_extension,
            preserved_forcing_prefix=preserved_forcing_prefix,
            owned=True, staging=staging)
    if staging is None:
        return lambda: _publish_restart_payload(path, payload)
    held = [payload]

    def publish():
        try:
            staging.wait()
            return _publish_restart_payload(path, held[0])
        finally:
            # The pinned slab goes back to the pool for the next checkpoint.
            held.clear()
            staging.release()

    def release():
        held.clear()
        staging.release()

    publish.release = release
    return publish


def _write_restart(path, state, cfg, *, run_trackers=None,
                   tree_header: dict | None = None,
                   extra_scratch_slots=(),
                   sealed_forcing_extension: bool = False,
                   preserved_forcing_prefix: bool = False) -> Path:
    path, payload = _restart_payload(
        path, state, cfg, run_trackers=run_trackers,
        tree_header=tree_header, extra_scratch_slots=extra_scratch_slots,
        sealed_forcing_extension=sealed_forcing_extension,
        preserved_forcing_prefix=preserved_forcing_prefix)
    return _publish_restart_payload(path, payload)


def _owned_host(value) -> np.ndarray:
    """A host array no later model step can change: a copy, always."""
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.array(value, copy=True)


def _restart_payload(path, state, cfg, *, run_trackers=None,
                     tree_header: dict | None = None,
                     extra_scratch_slots=(),
                     sealed_forcing_extension: bool = False,
                     preserved_forcing_prefix: bool = False,
                     owned: bool = False, staging=None):
    if getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0):
        from woof.moving_frame import initialize_moving_frame
        initialize_moving_frame(state, cfg)
    if cfg.diff_opt == 1:
        from woof.core.dycore import initialize_coordinate_reference
        initialize_coordinate_reference(state, cfg)
    path = Path(path)
    # A root whose boundaries still stream from an unsealed preparation is
    # checkpointed over its prepared intervals, under the preserved-prefix
    # contract, and bound to the head it streams from.
    stream = None if sealed_forcing_extension else _pre_seal_intervals(state)
    view = state
    if stream is not None:
        preserved_forcing_prefix = True
        view = _ReadyPrefixState(state, stream)
    if preserved_forcing_prefix:
        if sealed_forcing_extension:
            raise ValueError('A checkpoint cannot declare two forcing continuation modes')
        _require_preservable_forcing_prefix(view, cfg, path=path,
            elapsed=_admissible_elapsed_seconds(state.elapsed_seconds, 'preserved restart write'))
    _validate_nssl2_live_restart_state(state, cfg)
    _validate_thompson_aerosol_live_restart_state(state, cfg)
    _validate_milbrandt2_live_restart_state(state, cfg)
    if sealed_forcing_extension:
        _require_sealable_forcing_prefix(
            state, cfg, path=path,
            elapsed=_admissible_elapsed_seconds(
                state.elapsed_seconds, "sealed restart write"))
    manifest: dict[str, object] = {}
    manifest.update(state_manifest(state))
    manifest.update(_scratch_manifest(state))
    stochastic_header = None
    stochastic_binding = getattr(state, "_ensemble_stochastic", None)
    if stochastic_binding is not None and stochastic_binding.enabled:
        from woof.ensemble.stochastic_execution import checkpoint_payload
        stochastic_header, stochastic_arrays = checkpoint_payload(stochastic_binding)
        manifest.update(stochastic_arrays)
    if extra_scratch_slots:
        manifest.update(_opted_in_scratch_manifest(state, extra_scratch_slots))
    driver = getattr(state, "physics", None)
    driver_header = None
    if driver is not None:
        manifest.update(_driver_manifest(driver))
        driver_header = {
            "call_counts": {key: int(value)
                            for key, value in driver.call_counts.items()},
            "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
            "microphysics_updates": int(driver.microphysics_updates),
            # Carrier provenance: source + last producer model time, per
            # carrier.  Two scalars each, in the header rather than the
            # array set, so the v5 key layout is untouched.
            "carriers": driver.carriers.state(),
            "surface_radiation_policy": driver.carriers.policy,
        }
        driver_header.update(_fire_header(driver))
        hydro = getattr(driver, "hydro", None)
        if hydro is not None:
            driver_header["hydro"] = hydro.checkpoint_metadata()
    physics_setup = physics_setup_identity(state, cfg)
    physics_setup_sha256 = _json_sha256(physics_setup)

    arrays = {}
    array_manifest = {}
    to_host = _owned_host if owned else _host
    if staging is not None:
        # Same keys, same order, same host arrays; the device ones are
        # copied asynchronously into pinned memory (checkpoint_staging).
        staged = staging.stage(manifest, _owned_host)
    for key in sorted(manifest):
        host = staged[key] if staging is not None else to_host(manifest[key])
        arrays[key] = host
        array_manifest[key] = {"shape": list(host.shape),
                               "dtype": str(host.dtype)}
    config_echo = configuration_echo(cfg)
    header = {
        "format_version": RESTART_FORMAT_VERSION,
        "case": cfg.case,
        "created": datetime.now(timezone.utc).isoformat(),
        # The producer's own identity, so a checkpoint separated from its
        # logs can still say which build wrote it.  ``woof.__version__``
        # comes from installed distribution metadata, so this is the release
        # that is speaking rather than a hand-maintained constant.
        "producer": producer_identity(),
        # Provenance, excluded from every identity by construction; see
        # written_mode_note.  The resident writer materialises a full host
        # copy of the domain, which is what "resident" names here.  The
        # streamed writer lives in another package and does not stamp
        # yet, so a header with no key says "this file does not say".
        WRITTEN_MODE_HEADER_KEY: written_mode_note(
            RESIDENT_WRITTEN_MODE, cfg),
        "elapsed_seconds": _admissible_elapsed_seconds(
            state.elapsed_seconds, "restart write"),
        "config": config_echo,
        "setup_fingerprint": setup_fingerprint(view),
        "physics_setup": physics_setup,
        "physics_setup_fingerprint": physics_setup_sha256,
        "driver": driver_header,
        "run_trackers": (None if run_trackers is None
                         else dict(run_trackers)),
        "array_manifest": array_manifest,
    }
    if stochastic_header is not None:
        header["ensemble_stochastic"] = stochastic_header
    if sealed_forcing_extension or preserved_forcing_prefix:
        header.update({
            "forcing_extension_mode": (PRESERVED_FORCING_PREFIX_MODE if preserved_forcing_prefix
                                       else SEALED_FORCING_EXTENSION_MODE),
            "setup_core_fingerprint": setup_core_fingerprint(view),
            "lateral_boundary_prefix": lateral_boundary_prefix_identity(view),
        })
    if stream is not None:
        header[BOUNDARY_STREAM_HEADER_KEY] = {
            "head_sha256": str(stream.head_sha256)}
    lbc_clock_identity = root_external_lbc_clock_identity(state, cfg)
    if lbc_clock_identity is not None:
        header["root_external_lbc_clock"] = lbc_clock_identity
    if tree_header is not None:
        overlap = set(header) & set(tree_header)
        if overlap:
            raise ValueError(
                f"tree restart header may not replace base keys "
                f"{sorted(overlap)}")
        header.update(dict(tree_header))
    # ``allow_nan=False``: Python's json emits bare ``NaN``/``Infinity``
    # tokens by default, which are not JSON, and the reader accepts them
    # back.  A header that cannot express a non-finite clock is a header
    # whose clock cannot poison resume arithmetic after every identity
    # check has already passed.
    payload = {_HEADER_KEY: np.frombuffer(
        json.dumps(header, allow_nan=False).encode("utf-8"),
        dtype=np.uint8)}
    payload.update(arrays)
    return path, payload


def _publish_restart_payload(path: Path, payload: dict) -> Path:
    """Write one prepared restart archive and make it durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish: a crash mid-write must not leave a truncated file
    # under the valid gpuwmrst name (review F4).
    # Separate concurrent writers; each publishes one complete archive.
    temp = unique_temp_path(path)
    try:
        # np.savez's archive, byte for byte, CRC'd on every core instead of
        # one (woof.io.parallel_npz: the 3 km DA legs' np.savez ran on one
        # core, 16-68 s a member when six legs ended together).
        write_npz(temp, payload)
        # Atomic visibility was already sound; durability was not.  Closing
        # a file leaves its bytes in the page cache, so a machine or volume
        # crash could expose the published name with content that never
        # reached the disk.  The wrfout writer already fsyncs before it
        # replaces; the checkpoint did not, and a checkpoint is the thing a
        # crash is supposed to leave behind.
        fsync_file(temp)
        os.replace(temp, path)
        _fsync_directory(path.parent)
        from woof.hydro.retention import pin_current
        pin_current(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


def _decode_header(data, path) -> dict:
    if _HEADER_KEY not in data.files:
        raise RestartMismatchError(
            f"{path} is not a woof restart file (missing header)")
    return json.loads(bytes(bytearray(data[_HEADER_KEY])).decode("utf-8"))


def _load_restart(path, *, with_arrays: bool):
    """Load header (and optionally arrays), wrapping corruption loudly."""
    from woof.io.restart_prefetch import cached_restart
    cached = cached_restart(path, with_arrays=with_arrays)
    if cached is not None:
        return cached
    try:
        with np.load(path, allow_pickle=False) as data:
            header = _decode_header(data, path)
            stored = ({key: data[key] for key in data.files
                       if key != _HEADER_KEY} if with_arrays else None)
    except RestartMismatchError:
        raise
    except FileNotFoundError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
        raise RestartMismatchError(
            f"woof restart file {path} is unreadable (truncated or "
            "corrupt archive â€” likely an interrupted copy or a crash "
            "mid-write; woof itself publishes restart files atomically "
            "via a .tmp rename)") from exc
    return header, stored


def read_restart_header(path) -> dict:
    """Return the JSON header (config echo, clock, manifest, trackers)."""
    return _load_restart(Path(path), with_arrays=False)[0]


def _run_config_default(key: str):
    """The ``RunConfig`` default for ``key``, or a value equal to nothing."""
    from woof.config import RunConfig

    for field in dataclasses.fields(RunConfig):
        if field.name != key:
            continue
        if field.default is not dataclasses.MISSING:
            return field.default
        if field.default_factory is not dataclasses.MISSING:
            return field.default_factory()
    return object()


def _json_default(key: str):
    """:func:`_run_config_default` in the header's JSON spelling."""
    value = _run_config_default(key)
    return list(value) if isinstance(value, tuple) else value


def _chem_run_fields() -> frozenset:
    from woof.config import CHEM_RUN_FIELDS
    return frozenset(CHEM_RUN_FIELDS)


# These historical omission values deliberately do not follow moving defaults:
# a header without a selector must keep the arithmetic it wrote.
_RESTART_ABSENT_RUN_DEFAULTS = {
    'adaptive_nest_lattice': False,
    'zadvect_implicit': 0,
    'zadvect_implicit_variant': 'wrf_471',
    'rrtmg_smoke_manifest': '',
    'bl_mynn_version': 'wrf_461',
    'bl_mynn_cloud_tendency_form': 'wrf_461',
    'bl_mynn_gsd41_unsquared_qtke': False,
    'ruc_soilprop': 'wrf_45',
    'rrtmg_cloud_optics_form': 'wrf_461',
    'rrtmg_aer3_type_source': 'rural',
    'thompson_version': 'wrf_461',
    'thompson_table_dir': None,
    'thompson_fork_snow_fall': 'blend',
    'mynn_sfclay_variant': 'wrf_461',
    'urban_ucm_generation': 'wrf_471',
    'urban_table_dir': '',
    'use_rap_aero_icbc': False,
    'ruc_snow': 'wrf_461',
    'ruc_2m_diagnostic': 'flux',
    'ruc_qvg_cold_start': 'wrf',
    'ruc_irrigation': 'wrf_461',
    'scalar_adv_opt': 1,
    'scalar_pblmix': 0,
    'tempo_ml_for_bl_nc': False,
    'tempo_surface_emission_before': False,
    'bl_mynn_mixaerosols': 0,
    'bl_mynn_mixnumcon': 0,
    'terrain_clock': 'measured',
    'spp_conv': 0,
    'spp_pbl': 0,
    'fractional_seaice': 0,
    'swint_opt': 0,
    'aer_opt': 0,
    'alb_sol': 0,
    'w_crit_cfl': 1.0,
}


def _require_config_match(stored_config: dict, cfg, path) -> None:
    CHEM_RUN_FIELDS = _chem_run_fields()
    stored_config = _mosaic_checkpoint_config(stored_config)
    live_config = _mosaic_checkpoint_config(dataclasses.asdict(cfg))
    _drop_inert_nest_archive(stored_config)
    _drop_inert_nest_archive(live_config)
    _drop_inert_chem_block(stored_config)
    _drop_inert_chem_block(live_config)
    _drop_inert_fire_block(stored_config)
    _drop_inert_fire_block(live_config)
    # The writer omits these defaults, so its own new header must compare
    # under the same rule. Active settings remain and bind value for value.
    _drop_default_fork_run_keys(stored_config)
    _drop_default_fork_run_keys(live_config)
    _drop_default_radiation_port_keys(stored_config)
    _drop_default_radiation_port_keys(live_config)
    from woof.core.rrtmg_smoke_identity import bind_smoke_source_identity
    bind_smoke_source_identity(live_config)
    absent = object()
    differences = []
    policy_changes: list[str] = []
    # THE LIVE dt LEGITIMATELY DIFFERS FROM THE CONFIGURED ONE, and only
    # under an adaptive clock.  The checkpoint's `dt` is what the model
    # was integrating with at the instant it was written; the experiment's
    # is where the ramp STARTED.  A resume that demanded they match could
    # only ever resume a run that had not adapted yet -- which is
    # the resume invariant docs/ADAPTIVE-TIMESTEP.md states, and it refused a real
    # resume at dt=36.55 against a configured 30.0.
    #
    # `time_step_sound` rides with it: adaptive_clock._apply rewrites it
    # from the live dt on every period (wrf_num_sound_steps, which is what
    # upstream's `time_step_sound = 0` under an adaptive clock asks
    # solve_em to do), so a checkpoint taken after dt grew past the
    # 4-substep floor carries a value the resuming experiment cannot have.
    # Measured: the doc's 10 km case reaches dt ~ 76 s, where the count is
    # 6 against a configured 4, and the resume was refused on a field
    # nothing had configured.
    #
    # Both sides must agree the feature is ON.  `use_adaptive_time_step`
    # itself stays compared exactly, so a fixed-clock checkpoint's dt is
    # still required to match to the bit, and a resume that flips the
    # feature is still refused.
    adaptive_both = (bool(stored_config.get("use_adaptive_time_step"))
                     and bool(live_config.get("use_adaptive_time_step")))
    for key in sorted(set(stored_config) | set(live_config)):
        if key in CONFIG_RUN_LENGTH_FIELDS:
            continue
        if key in ADAPTIVE_DERIVED_RUN_FIELDS and adaptive_both:
            # Exactly the set the adaptive clock overwrites every root
            # step, imported rather than restated -- see that constant for
            # why listing them here by hand made adaptive checkpoints
            # unrestartable.
            continue
        if key in ADAPTIVE_POLICY_RUN_FIELDS and adaptive_both:
            # CONTROLLER POLICY, not model state.  Changing a target or a
            # clamp does mean the resumed leg integrates on a different
            # clock from the leg that wrote the checkpoint -- but that is
            # exactly what a recovery resume is FOR, and refusing it made
            # restart_interval_s useless for its most valuable case: a run
            # that died because of the setting you now want to change.
            #
            # Allowed, and recorded rather than silent: the change is
            # reported so a resumed run's provenance still says the clock
            # policy moved and by how much.  `use_adaptive_time_step`
            # itself is NOT in this set -- flipping the feature leaves the
            # carried controller state meaningless and is still refused.
            stored_v = stored_config.get(key, absent)
            live_v = live_config.get(key, absent)
            if stored_v != live_v:
                # Name an ABSENT side rather than repr()-ing the
                # sentinel -- the same ruling the refusal branch below
                # already carries, and reachable here too: a header
                # written between two builds carries
                # `use_adaptive_time_step` but not every policy field
                # beside it, and then the one notice whose whole job is
                # to state the old value precisely printed
                # "<object object at 0x...>" as that value.
                was = ("absent from the restart file"
                       if stored_v is absent else repr(stored_v))
                now = ("absent from this build"
                       if live_v is absent else repr(live_v))
                policy_changes.append(f"{key}: {was} -> {now}")
            continue
        if key == "cycling":
            # A start-only control: it decides what the first MYNN call
            # (itimestep 1) keeps from the input.  A resume continues after
            # that step and never repeats it, so either value resumes the
            # same trajectory.
            continue
        if key in CONFIG_DIAGNOSTIC_FIELDS:
            continue
        if key in _SEAICE_RUN_KEYS and not _seaice_read(stored_config)                 and not _seaice_read(live_config):
            # Neither side runs prognostic sea ice: the keys reach nothing.
            continue
        stored = stored_config.get(key, absent)
        live = live_config.get(key, absent)
        if key in _CAMPAIGN_IMPLICIT_RUN_KEYS:
            default = _CAMPAIGN_IMPLICIT_RUN_KEYS[key]
            stored = default if stored is absent else stored
            live = default if live is absent else live
        if key == "noahmp_snow_age":
            if stored_config.get("sf_surface_physics", 0) != 4 and \
                    live_config.get("sf_surface_physics", 0) != 4:
                continue
            # Old headers always ran daylight-only aging.
            stored = "wrf_461" if stored is absent else stored
            live = "wrf_461" if live is absent else live
        if key in ("diff_6th_form", "diff_6th_factor2",
                   "upper_wind_limiter_form", "mp_zero_out"):
            # The writer omits these historical defaults. A current
            # checkpoint must resume under the same implicit settings;
            # an active form, factor, limiter or zero-out still differs.
            default = _run_config_default(key)
            stored = default if stored is absent else stored
            live = default if live is absent else live
        if key in ("mp_zero_out_thresh", "mp_zero_out_all") \
                and not stored_config.get("mp_zero_out", 0) \
                and not live_config.get("mp_zero_out", 0):
            # Both writers omit these unread values when zero-out is off.
            continue
        if key in ("diff_opt", "mix_full_fields") and stored is absent:
            # Checkpoints from before the coordinate selector existed used
            # exactly the metric operator with full-field scalar mixing.
            stored = 2 if key == "diff_opt" else True
        if key in _RESTART_ABSENT_RUN_DEFAULTS:
            default = _RESTART_ABSENT_RUN_DEFAULTS[key]
            stored = default if stored is absent else stored
            live = default if live is absent else live
        if key == "urban_water_balance":
            if int(live_config.get("sf_urban_physics", 0) or 0) != 1:
                continue
            stored = "wrf_471" if stored is absent else stored
            live = "wrf_480" if live is absent else live
        if key == "ysu_version":
            # Changing the entrainment cap or arithmetic changes the PBL
            # trajectory. A header without a selector is always legacy.
            if stored_config.get("bl_pbl_physics", 0) != 1 and live_config.get("bl_pbl_physics", 0) != 1:
                continue
            stored = "wrf_461" if stored is absent else stored
            live = "wrf_461" if live is absent else live
        if key == "solar_generation":
            stored = "wrf_461" if stored is absent else stored
            live = "wrf_480" if live is absent else live
        if key in ("urban_treeoption", "urban_ttscheme", "urban_fvg"):
            stored = -1 if stored is absent else stored
            live = -1 if live is absent else live
        if key in ADVECTION_ORDER_RUN_DEFAULTS:
            # Older checkpoints ran the vert_order 3 ladder and the flux5
            # horizontal momentum stencil; a moved order is refused below,
            # because it changes the trajectory.
            default = ADVECTION_ORDER_RUN_DEFAULTS[key]
            stored = default if stored is absent else stored
            live = default if live is absent else live
        if key in _CORRECTION_KEYS:
            # Correct-mode keys: False is omitted by configuration_echo and
            # is the matched arithmetic every earlier checkpoint ran (absent
            # on restore = exact, so a 2.8.9 checkpoint continues only in
            # exact mode).  A key no scheme on either side reads binds
            # nothing; under its scheme a resume across modes is refused
            # below, with CORRECT_MODE_RESTART_BREAK_NOTICE.
            if not (_correct_key_in_scope(key, stored_config)
                    or _correct_key_in_scope(key, live_config)):
                continue
            stored = False if stored is absent else stored
            live = False if live is absent else live
        if key in TERRAIN_DRAG_RUN_DEFAULTS:
            # Absent is the default (_drop_inert_terrain_drag); a moved
            # topo_wind/gwd_opt still refuses below.
            default = TERRAIN_DRAG_RUN_DEFAULTS[key]
            stored = default if stored is absent else stored
            live = default if live is absent else live
        if key in TOPO_RADIATION_RUN_DEFAULTS:
            # Absent is the default (_drop_inert_topo_radiation); a
            # flipped slope/shadow setting still refuses below.
            default = TOPO_RADIATION_RUN_DEFAULTS[key]
            stored = default if stored is absent else stored
            live = default if live is absent else live
        if key == "ra_rrtmg_variant" and stored is absent:
            # Migration rule (2026-07-27 assembly dossier): a v5 header
            # written before the radiation-variant field existed could
            # only have run the RTE+RRTMGP substitution, so restore it
            # under that value.  An explicitly STORED variant that
            # mismatches the live config stays fail-closed below --
            # never infer legacy, never widen.
            stored = RRTMG_VARIANT_RTE_RRTMGP
        if isinstance(live, tuple):
            # The header is JSON, which carries a tuple field back as a
            # list (chem_sets, chem_sources, eta_levels); compare in that
            # spelling or a tuple-valued field could never match itself.
            live = list(live)
        if (stored is absent and key in CHEM_RUN_FIELDS
                and live == _json_default(key)):
            # The chem block (woof/config.py after adaptive_nest_lattice):
            # a checkpoint written before it existed ran with chem off,
            # which is exactly what a live config holding the defaults
            # describes -- every other chem key is read by nothing while
            # chem_sets is empty.  A live config that turns chem ON against
            # such a checkpoint still refuses here (chem_sets differs).
            continue
        if stored is absent and key == "eta_levels" and live is None:
            # ABSENT STAYS ABSENT: ``eta_levels = None`` means "inherit the
            # source's ladder", which is exactly and only what every
            # checkpoint written before the field existed describes.  A
            # declared ladder still binds value for value below.
            continue
        if (stored is absent and key in ADAPTIVE_TIMESTEP_RUN_FIELDS
                and not live_config.get("use_adaptive_time_step")
                and live == _run_config_default(key)):
            # A FIXED-dt checkpoint does not need the adaptive echo: with
            # the controller off none of these fields is read by anything,
            # so a header written without them and a live config holding
            # their defaults describe one clock.  A live config that turns
            # the controller ON against such a header still refuses below
            # (the flag itself is compared), as does any non-default value.
            continue
        if stored is not live and stored != live:
            # Name an ABSENT side rather than repr()-ing the sentinel.
            # The concrete breakage this replaces: `absent` is a bare
            # object(), so a key on only one side rendered as
            # "<object object at 0x7f...>" -- a memory address, which
            # tells the operator nothing about WHY the checkpoint is
            # refused and changes between runs.  Measured on the P3 CUDA
            # port: a 2.5.8 checkpoint meeting a build that has
            # `p3_backend` is refused correctly and the field IS named,
            # but the reason (the checkpoint predates the field) was
            # unreadable.  Every RunConfig field added after a checkpoint
            # was written inherits this line, so it is fixed for all of
            # them, not for P3.
            if stored is absent:
                differences.append(
                    f"{key}: absent from the restart file, run={live!r} "
                    "-- the checkpoint was written by a build that did "
                    "not have this configuration field")
            elif live is absent:
                differences.append(
                    f"{key}: restart={stored!r}, absent from this build "
                    "-- the checkpoint was written by a build that had "
                    "this configuration field and this one does not")
            else:
                differences.append(
                    f"{key}: restart={stored!r} run={live!r}")
    if policy_changes:
        # Reported, not refused.  Retuning the controller on a resume is
        # a legitimate and often the ONLY useful recovery action, but the
        # resumed leg does integrate on a different clock from the leg
        # that wrote the checkpoint, so it must not be silent: this is
        # what tells a later reader the trajectory has a seam, and where.
        # ONE LINE, joined with "; ": woof.explain.warn collapses runs of
        # whitespace into single spaces, so a newline-separated list would
        # arrive as an unreadable run-on.  Each field still carries both
        # of its values, which is what the notice is for.
        joined = "; ".join(policy_changes)
        #
        # THE PROJECT'S OWN NOTICE CHANNEL, not warnings.warn.  A
        # warnings.warn here is a refusal in disguise under any
        # caller running -W error or simplefilter("error"): the
        # ALLOWED retune then raises UserWarning out of the middle
        # of the identity gate, so the one recovery this branch
        # exists to permit fails with a traceback instead of a
        # named refusal.  It was also routed nowhere a reader looks,
        # while the rest of the restore account already goes to
        # woof.explain.warn -- including the relocation banner this
        # notice sits beside.
        from woof.explain import warn as _explain_warn

        _explain_warn(
            f"restart file {path} was written under different adaptive "
            f"controller policy; continuing with the LIVE configuration, "
            f"which changes the clock from this point on: {joined}.  "
            f"(model state is unaffected -- these govern future "
            "steps only; flipping use_adaptive_time_step is still "
            "refused.)")
    if differences:
        message = (
            f"restart file {path} was written under a different "
            "configuration; refusing to continue a different model:\n  "
            + "\n  ".join(differences))
        if any(item.startswith('solar_generation:') for item in differences):
            old_solar = stored_config.get('solar_generation', 'wrf_461')
            message += (
                "\nChanging solar_generation changes COSZEN and HRANG, "
                "surface shortwave forcing and the interpolation fit under "
                "the saved trajectory. Resume with solar_generation="
                f"{old_solar!r}, or start a new forecast.")
        # Restart accuracy for the 2026-08-16 mixing auto-switch: a
        # checkpoint from the old anisotropic default meeting a run that
        # selects the isotropic length gets told WHY the default moved
        # under it and how to resume, not just that a field differs.  A
        # header written before the knob existed integrated the then-only
        # value 0, so the absent key reads as 0 here.
        if (stored_config.get("mix_isotropic", 0) == 0
                and live_config.get("mix_isotropic") == 1):
            message += "\n" + MIX_ISOTROPIC_RESTART_BREAK_NOTICE
        # The same accuracy for the 2.8.8 terrain-clock default: an absent
        # key is the "measured" clock every earlier checkpoint ran.
        if (stored_config.get("terrain_clock", "measured") == "measured"
                and live_config.get("terrain_clock") == "local_face"):
            message += "\n" + TERRAIN_CLOCK_RESTART_BREAK_NOTICE
        if (stored_config.get("terrain_clock", "measured") != "pinned"
                and live_config.get("terrain_clock") == "pinned"):
            message += "\n" + TERRAIN_CLOCK_PINNED_RESTART_NOTICE
        # The same accuracy for the 2.9.0 correct-mode default: an absent
        # key is the exact arithmetic every earlier checkpoint ran.
        if _correct_mode_run_flip(stored_config, live_config):
            message += "\n" + CORRECT_MODE_RESTART_BREAK_NOTICE
        raise RestartMismatchError(message)


#: The configuration selector each ``algorithms`` component is chosen by,
#: so a refusal names the setting the user wrote beside the identity that
#: moved.  ``radiation`` is the summary of the two radiation rows and is
#: left out of a list that already names them.
_ALGORITHM_SELECTORS = {
    "microphysics": ("mp_physics",),
    "surface_layer": ("sf_sfclay_physics",),
    "land_surface": ("sf_surface_physics",),
    "pbl": ("bl_pbl_physics",),
    "radiation_lw": ("ra_lw_physics",),
    "radiation_sw": ("ra_sw_physics",),
    "cumulus": ("cu_physics",),
    "urban": ("sf_urban_physics",),
    "seaice": ("seaice_prognostic",),
    "dycore_mixing": ("km_opt", "diff_6th_opt"),
    "upper_damping": ("damp_opt",),
}

#: Run-configuration keys whose ABSENCE from a checkpoint header means a
#: value other than this build's RunConfig default.  configuration_echo
#: drops a key at the value every earlier header ran; for these the
#: default has since moved, so rebuilding a RunConfig from the header must
#: put the dropped value back or the rebuilt run is not the one that wrote
#: the checkpoint.  _require_config_match reads the same absences the same
#: way.
#:
#: terrain_clock: 2.8.8 made "local_face" the default and the echo drops
#: "measured", so a header without the key ran "measured".  Without this a
#: 2.8.7 checkpoint was refused on configuration_sha256 even under a
#: configuration that set terrain_clock = "measured" as the refusal note
#: says to (stored 588c1266..., rebuilt under the default 2be20cee...).
#:
#: The four WRF 4.7/4.8 fix selectors (2.9.0): each default is the corrected
#: generation and the echo drops the previous one (and drops the two
#: scheme-scoped ones while their scheme is off), so a header without the
#: key ran the previous generation.  Without these a 2.8.9 YSU checkpoint
#: was rebuilt under ysu_version = "wrf_480" and refused as "ysu-v2 ->
#: ysu-wrf-v4.8.0 ... no configuration change resumes it", when
#: ysu_version = "wrf_461" resumes it; the same for the solar generation.
ECHO_ABSENT_RUN_VALUES = {"terrain_clock": "measured",
                          "solar_generation": "wrf_461",
                          "ysu_version": "wrf_461",
                          "noahmp_snow_age": "wrf_461",
                          "urban_water_balance": "wrf_471",
                          # The correct-mode keys (2.9.0): the default is
                          # on, the echo drops off, so a header without
                          # the key ran the exact arithmetic.
                          **{key: False for key in _CORRECTION_KEYS
                             if key not in _OUTPUT_ONLY_CORRECTION_KEYS}}


def _run_config_from_header(raw_config):
    """The RunConfig a checkpoint header's ``config`` echo describes.

    ``None`` when this build cannot rebuild one from the header (a value
    it refuses, a malformed field): callers then fall back to the stricter
    raw comparison.
    """
    if not isinstance(raw_config, Mapping):
        return None
    try:
        from woof.config import RunConfig

        names = {f.name for f in dataclasses.fields(RunConfig)}
        values = {k: v for k, v in raw_config.items() if k in names}
        for key, value in ECHO_ABSENT_RUN_VALUES.items():
            values.setdefault(key, value)
        return RunConfig(**values)
    except (TypeError, ValueError):
        return None


def _selector_values(cfg) -> dict:
    """``{selector: value}`` of a RunConfig, radiation resolved.

    The radiation pair is read through :func:`radiation_scheme_ids`, so a
    run configured by ``ra_physics`` names the scheme it ran rather than
    the -1 that defers to it.
    """
    if cfg is None:
        return {}
    values = {}
    for selectors in _ALGORITHM_SELECTORS.values():
        for name in selectors:
            value = getattr(cfg, name, None)
            if value is not None:
                values[name] = value
    try:
        values["ra_lw_physics"], values["ra_sw_physics"] = \
            radiation_scheme_ids(cfg)
    except (AttributeError, TypeError, ValueError):
        pass
    return values


def _algorithm_moves(stored, current, cfg) -> list[str]:
    """One phrase per moved algorithm identity: selector, old, new.

    ``cfg`` is the RunConfig the selectors are read from (the checkpoint's
    own, rebuilt from its header), or ``None`` to name components only.
    """
    keys = sorted(set(stored) | set(current))
    if "radiation_lw" in keys or "radiation_sw" in keys:
        keys = [key for key in keys if key != "radiation"]
    values = _selector_values(cfg)
    moves = []
    for key in keys:
        old, new = stored.get(key), current.get(key)
        if old == new:
            continue
        named = ", ".join(
            f"{name}={values[name]!r}"
            for name in _ALGORITHM_SELECTORS.get(key, ()) if name in values)
        label = (f"{named} ({key.replace('_', ' ')})" if named
                 else key.replace("_", " "))
        moves.append(
            f"{label}: {old if old is not None else 'none recorded'} -> "
            f"{new if new is not None else 'none'}")
    return moves


def _require_current_algorithm_identities(header, cfg, path) -> None:
    """Refuse FIRST a checkpoint whose schemes this build computes differently.

    The breakage: a checkpoint integrated by a scheme implementation this
    build no longer runs (2.8.8 moved YSU, the surface layers, RUC, the
    microphysics finish clamp, the dycore mixing and the upper damping
    layer under unchanged selectors) resumes under no configuration at all.  Asked after the
    configuration walk, the user was first told to change a configuration
    field (the terrain-clock note) and, having changed it, was refused
    again by a sentence naming only "algorithms, configuration_sha256".
    Asked here, before that walk, the refusal names each scheme, its
    selector and both identities.

    The identities compared are the checkpoint's recorded ones and the ones
    THIS build assigns to the checkpoint's OWN configuration, so a change
    the user made to the live configuration is left to the configuration
    walk, which names the field.  It refuses only when the recorded
    identities also differ from the live run's, which the physics setup
    gate refuses anyway, so it never refuses a resume that gate would
    allow.  A header this build cannot rebuild, or whose schemes it cannot
    name, is left to the later gates.
    """
    setup = header.get("physics_setup")
    stored = setup.get("algorithms") if isinstance(setup, Mapping) else None
    if not isinstance(stored, Mapping):
        return
    # Only an intact identity is read here: one that no longer matches its
    # own stored fingerprint, or is of another schema, is the physics
    # gate's malformed-file refusal, which must not be reworded as a
    # change of build.
    try:
        intact = (_json_sha256(setup) == header.get("physics_setup_fingerprint")
                  and setup.get("schema_version")
                  == PHYSICS_SETUP_SCHEMA_VERSION)
    except (TypeError, ValueError):
        intact = False
    if not intact:
        return
    rebuilt = _run_config_from_header(header.get("config"))
    if rebuilt is None:
        return
    try:
        current = physics_algorithm_identities(rebuilt)
        live = physics_algorithm_identities(cfg)
    except (RestartManifestError, TypeError, ValueError):
        return
    if dict(stored) == current or dict(stored) == live:
        return
    moves = _algorithm_moves(stored, current, rebuilt)
    raise RestartMismatchError(
        f"restart file {path} was integrated by scheme implementations this "
        f"build does not run: {'; '.join(moves)}.  Continuing it here would "
        "splice two implementations into one forecast, so no configuration "
        "change resumes it.  Start the run again from its initial time "
        "under this build, or resume it with the build that wrote it")


def _require_physics_setup_match(header: dict, state, cfg, path) -> None:
    """Validate stored self-consistency and live physics identity."""
    stored = header["physics_setup"]
    stored_fingerprint = header["physics_setup_fingerprint"]
    if not isinstance(stored, dict):
        raise RestartMismatchError(
            f"restart file {path} has a malformed physics setup identity")
    try:
        computed_stored_fingerprint = _json_sha256(stored)
    except (TypeError, ValueError) as exc:
        raise RestartMismatchError(
            f"restart file {path} has a malformed physics setup identity") \
            from exc
    if (not isinstance(stored_fingerprint, str)
            or stored_fingerprint != computed_stored_fingerprint):
        raise RestartMismatchError(
            f"restart file {path} physics setup fingerprint does not match "
            "its own stored identity")
    if stored.get("schema_version") != PHYSICS_SETUP_SCHEMA_VERSION:
        raise RestartMismatchError(
            f"restart file {path} physics setup schema "
            f"{stored.get('schema_version')!r} is not supported by this "
            f"build (expected {PHYSICS_SETUP_SCHEMA_VERSION})")
    try:
        live = physics_setup_identity(state, cfg)
        # Serialisability is part of the contract, and the comparison
        # below no longer hashes the live side, so it is asserted here:
        # an identity this build cannot hash is not one it can compare,
        # and that must surface as a named malformed-identity refusal
        # rather than as a bare inequality.
        _json_sha256(live)
    except RestartManifestError as exc:
        raise RestartMismatchError(
            f"resuming model has no complete physics setup identity: {exc}") \
            from exc
    # ONE RULE SET ON BOTH SIDES.  `configuration_sha256` is a hash the
    # WRITING build computed under whatever exemptions it had, so any
    # change to those exemptions silently invalidates every checkpoint
    # already on disk -- the comparison stops asking "is this the same
    # configuration" and starts asking "was it hashed by this build".
    # Measured: widening the adaptive exemptions correctly made a retuned
    # resume legal at the identity walk and this gate refused it anyway,
    # naming nothing.  Recomputing the stored side from the RunConfig the
    # header already carries puts both sides under the current rules.
    #
    # The self-consistency check above deliberately stays on the RAW
    # stored dict: that one asks whether the file is intact, which is a
    # different question and must not be normalised away.
    stored_cmp = stored
    # A header this build cannot rebuild a RunConfig from falls back to the
    # RAW stored hash, which is the STRICTER of the two: it can only refuse
    # a resume this normalisation would have allowed, never allow one it
    # would have refused.  An absent key is read as the value the header's
    # run used (ECHO_ABSENT_RUN_VALUES), not as this build's default.
    rebuilt = _run_config_from_header(header.get("config"))
    if rebuilt is not None:
        stored_cmp = dict(stored)
        stored_cmp["configuration_sha256"] = _configuration_fingerprint(
            rebuilt)

    if stored_cmp != live:
        moved = [key for key in sorted(set(stored_cmp) | set(live))
                 if stored_cmp.get(key) != live.get(key)]
        detail = ""
        if ("algorithms" in moved
                and isinstance(stored_cmp.get("algorithms"), Mapping)
                and isinstance(live.get("algorithms"), Mapping)):
            detail = " (" + "; ".join(_algorithm_moves(
                stored_cmp["algorithms"], live["algorithms"], cfg)) + ")"
        raise RestartMismatchError(
            f"restart file {path} was written under a different physics "
            f"setup; these components differ: {', '.join(moved)}{detail}.  "
            "Rebuild the identical physics preparation before restoring")


def _require_rrtmg_variant_match(header: dict, cfg, path) -> None:
    """Name a resume that swaps the 4/4 radiation IMPLEMENTATION.

    Scheme id 4 is worn by two different codes: the WRF v4.6.1 RRTMG port
    (``ra_rrtmg_variant='rrtmg_legacy'``) and the RTE+RRTMGP substitution
    that stands in for it by default.  They are refused across a resume,
    correctly and by two separate gates already -- the configuration walk
    reports ``ra_rrtmg_variant`` as a changed field, and the physics
    identity reports "radiation, algorithms" as differing components --
    and neither says WHAT breaks.

    Audit R-048: the refusal is right and its text was not.  A crossed
    resume splices two transcribed algorithms with different
    above-atmosphere treatments (the legacy port's WRF 4 mb buffer layers
    against RRTMGP's) and different pinned coefficient tables and trace-gas
    policy, and it presents the discontinuity in the heating-rate
    trajectory as a continuation.  This is the same refusal, said with the
    breakage and the two ways out, and it is asked FIRST so that it is the
    sentence the user reads rather than a field name in a list.

    Nothing new is refused: every crossed resume this names was already
    refused, and a same-variant resume -- including one from a header
    written before the field existed, which ``_require_config_match``
    restores as the RTE+RRTMGP substitution -- is untouched.
    """
    lw_id, sw_id = radiation_scheme_ids(cfg)
    if 4 not in (lw_id, sw_id):
        return
    stored_config = header.get("config")
    if not isinstance(stored_config, Mapping):
        return
    # A header written before the field existed could only have run the
    # substitution; the same migration rule _require_config_match applies,
    # and for the same reason -- never infer legacy, never widen.
    stored_variant = stored_config.get("ra_rrtmg_variant",
                                       RRTMG_VARIANT_RTE_RRTMGP)
    live_variant = rrtmg_variant(cfg)
    if stored_variant == live_variant:
        return
    names = {RRTMG_VARIANT_LEGACY: "the WRF v4.6.1 RRTMG port",
             RRTMG_VARIANT_RTE_RRTMGP: "the RTE+RRTMGP substitution"}
    raise RestartMismatchError(
        f"restart file {path} was integrated by "
        f"{names.get(stored_variant, repr(stored_variant))} "
        f"(ra_rrtmg_variant={stored_variant!r}) and this run selects "
        f"{names.get(live_variant, repr(live_variant))} "
        f"(ra_rrtmg_variant={live_variant!r}).  Radiation scheme id 4 is "
        "worn by both, and they are different codes: different transcribed "
        "algorithms, different above-atmosphere treatments (the legacy "
        "port carries WRF's 4 mb buffer layers, RRTMGP does not), and "
        "different pinned coefficient tables and trace-gas policy.  "
        "Resuming across them would splice two heating-rate trajectories "
        "and present the seam as a continuation.  Set "
        f"ra_rrtmg_variant={stored_variant!r} to continue THIS forecast, "
        "or start a new run from t = 0 under "
        f"ra_rrtmg_variant={live_variant!r}.")


def _require_nssl2_restart_contract(header: dict, cfg, path) -> None:
    """Reject an absent, stale, or extended MP18 nested schema."""
    if int(cfg.mp_physics) != 18:
        return
    setup = header.get("physics_setup")
    try:
        actual = setup["microphysics"]["restart_contract"]
    except (KeyError, TypeError) as exc:
        raise RestartMismatchError(
            f"restart file {path} has no versioned MP18 restart contract") \
            from exc
    if not isinstance(actual, dict):
        raise RestartMismatchError(
            f"restart file {path} has a malformed MP18 restart contract")
    version = actual.get("schema_version")
    if version != NSSL2_RESTART_CONTRACT_VERSION:
        raise RestartMismatchError(
            f"restart file {path} has MP18 restart contract version "
            f"{version!r}; expected {NSSL2_RESTART_CONTRACT_VERSION}")
    if actual != _nssl2_restart_contract_identity(cfg):
        raise RestartMismatchError(
            f"restart file {path} MP18 restart contract does not exactly "
            "match the canonical state/timing inventory")


def _check_array(stored: np.ndarray, target, key: str) -> None:
    if tuple(stored.shape) != tuple(target.shape):
        raise RestartMismatchError(
            f"{key}: restart shape {tuple(stored.shape)} does not match "
            f"state shape {tuple(target.shape)}")
    if stored.dtype != target.dtype:
        raise RestartMismatchError(
            f"{key}: restart dtype {stored.dtype} does not match state "
            f"dtype {target.dtype}")


# Every namespace this build can restore. Newer unknown carriers must
# refuse instead of silently losing state at a cross-version resume.
RESTART_MEMBER_NAMESPACES = (
    "state/", "acoustic/", "scratch/", "driver/", "fields/", "cumulus/",
    "diag/", "held/", "pbl/", "radiation/",
    # The dedicated stochastic validator checks the exact spectrum inventory,
    # member, recipe and process metadata before this namespace closure.
    "stochastic/", "fire/", "hydro/",
)


def _validate_member_namespaces(stored, state, driver, path, format_version) -> None:
    """Close the member inventory and validate carriers before mutation.

    Known optional diagnostics preserve their documented absence/drop
    policy. Unknown names have no restore route and must be refused.
    """
    for key in sorted(stored):
        if not key.startswith(RESTART_MEMBER_NAMESPACES):
            raise RestartMismatchError(
                f"restart file {path} carries member {key!r} under no member "
                "namespace this build knows how to restore; resume with the "
                "build that wrote it or start from prepared state")
        if key.startswith("fire/") and driver is None:
            raise RestartMismatchError("restart carries SFIRE state but the resuming state has no PhysicsDriver")
    for prefix, names, owner, description in (
            ("acoustic/", CHECKPOINT_ONLY_STATE, state, "checkpoint-only state"),
            ("diag/", DRIVER_CHECKPOINT_ONLY_ATTRS, driver, "a checkpoint-only driver"),
            ("held/", DRIVER_HELD_FORCING_ATTRS, driver, "a held physics forcing")):
        for key in sorted(key for key in stored if key.startswith(prefix)):
            name = key[len(prefix):]
            if name not in names:
                raise RestartMismatchError(
                    f"restart file {path} carries {key}, which this build "
                    f"does not classify as {description} carrier")
            target = getattr(owner, name, None)
            if target is not None:
                _check_array(stored[key], target, key)
    allowed_driver = {
        "driver/rthratenlw", "driver/rthratensw", "driver/pending_rainbl",
        *(f"driver/{group}/{component}" for group in (*DRIVER_TENDENCY_ATTRS, *OPTIONAL_DRIVER_TENDENCY_ATTRS)
          for component in TENDENCY_COMPONENTS),
    }
    if format_version == 2:
        allowed_driver.update(f"driver/microphysics/{name}"
                              for name in MICROPHYSICS_COMPONENTS)
    for key in sorted(key for key in stored if key.startswith("driver/")):
        if key not in allowed_driver:
            raise RestartMismatchError(
                f"restart file {path} carries {key}, which this build has "
                "no driver restore route for")
        if key.startswith("driver/microphysics/"):
            _check_array(stored[key], state.mup, key)
    allowed_cumulus = {f"cumulus/{name}" for name in CUMULUS_CALLABLE_ARRAYS}
    for key in sorted(key for key in stored if key.startswith("cumulus/")):
        if key not in allowed_cumulus:
            raise RestartMismatchError(
                f"restart file {path} carries {key}, which this build has "
                "no cumulus restore route for")
    for key in sorted(key for key in stored if key.startswith("radiation/")):
        if key not in RESTART_ONLY_DRIVER_SLOTS:
            raise RestartMismatchError(
                f"restart file {path} carries {key}, which is not one of "
                "this build's restart-only driver slots")
        if driver is None:
            raise RestartMismatchError(
                f"restart file {path} carries {key} but this state has no PhysicsDriver")


def _validate_nssl2_stored_restart_state(
        header: dict, stored: dict[str, np.ndarray], state, cfg,
        path, elapsed: float) -> None:
    """Hoist every MP18 inventory/timing refusal before restore mutation."""
    if int(cfg.mp_physics) != 18:
        return

    aliases = sorted(set(stored) & NSSL2_LEGACY_RESTART_ALIASES)
    if aliases:
        raise RestartMismatchError(
            f"restart file {path} uses legacy MP18 aliases {aliases}; only "
            "canonical Registry and scratch names are accepted")

    expected_state = {
        f"state/{name}" for name in serialized_state_attrs(state)
        if getattr(state, name, None) is not None
    }
    stored_state = {key for key in stored if key.startswith("state/")}
    missing_state = sorted(expected_state - stored_state)
    extra_state = sorted(stored_state - expected_state)
    if missing_state or extra_state:
        raise RestartMismatchError(
            f"restart file {path} MP18 state inventory mismatch "
            f"(missing {missing_state}, extra {extra_state})")

    required_state = {
        *(f"state/{name}" for name in NSSL2_RESTART_PROGNOSTICS),
        *(f"state/{name}" for name in NSSL2_RESTART_AUXILIARY_STATE),
    }
    missing_required = sorted(required_state - stored_state)
    if missing_required:
        raise RestartMismatchError(
            f"restart file {path} omits canonical MP18 state "
            f"{missing_required}")

    pool = getattr(state, "_scratch", {})
    expected_scratch = {
        f"scratch/{slot}" for slot in pool
        if classify_scratch_slot(slot) == "serialize"
    }
    stored_scratch = {key for key in stored if key.startswith("scratch/")}
    # A tracker window is OPTIONAL in both directions: present when the
    # writing run declared a nest lifecycle, absent otherwise, and neither
    # is an inventory defect (:func:`_restorable_scratch_slot`).  Read off
    # the FILE rather than the live pool, so a checkpoint written under
    # nwp_diagnostics = 1 and resumed under 0 reaches the generic
    # drop-with-a-note instead of refusing here on an inventory count.
    from woof.core.streaming import REFL_STORE_KEY
    optional_scratch = {key for key in stored_scratch
                        if (_is_tracker_window_slot(key[len("scratch/"):])
                            or key == REFL_STORE_KEY
                            # This shared diffusion reference is allocated
                            # lazily by the first diff_opt=1 step. It is
                            # required above and restored by the generic
                            # scratch reader, even when absent from the
                            # freshly initialized MP18 pool.
                            or key == "scratch/diff1_theta_initial")}
    missing_scratch = sorted(expected_scratch - stored_scratch)
    extra_scratch = sorted(stored_scratch - expected_scratch - optional_scratch)
    if missing_scratch or extra_scratch:
        raise RestartMismatchError(
            f"restart file {path} MP18 scratch inventory mismatch "
            f"(missing {missing_scratch}, extra {extra_scratch})")
    required_precipitation = {
        f"scratch/{slot}" for slot in NSSL2_RESTART_PRECIPITATION_SLOTS
    }
    missing_precipitation = sorted(
        required_precipitation - stored_scratch)
    if missing_precipitation:
        raise RestartMismatchError(
            f"restart file {path} omits MP18 precipitation state "
            f"{missing_precipitation}")

    for key in sorted(required_state):
        _check_array(stored[key], getattr(state, key[len("state/"):]), key)
    for key in sorted(required_precipitation):
        _check_array(stored[key], pool[key[len("scratch/"):]], key)

    driver = getattr(state, "physics", None)
    if driver is None or int(getattr(driver, "mp_physics", -1)) != 18:
        raise RestartMismatchError(
            "MP18 restart requires a prepared MP18 PhysicsDriver")
    driver_header = header.get("driver")
    updates = (None if not isinstance(driver_header, dict)
               else driver_header.get("microphysics_updates"))
    if (isinstance(updates, bool) or not isinstance(updates, int)
            or updates < 0):
        raise RestartMismatchError(
            "MP18 restart first-call authority microphysics_updates must be "
            f"a non-negative integer, got {updates!r}")
    if (isinstance(header.get("elapsed_seconds"), bool)
            or not math.isfinite(elapsed) or elapsed < 0.0):
        raise RestartMismatchError(
            "MP18 restart elapsed_seconds must be finite and non-negative")


def _asarray_like(state):
    """Return host->model-array converter for the state's array module."""
    if type(state.u).__module__.partition(".")[0] == "cupy":
        import cupy
        return cupy.asarray
    return lambda host: np.array(host, copy=True)


@dataclasses.dataclass(frozen=True)
class RestartInfo:
    """Restore result: the restored clock and the writer's run trackers."""

    elapsed_seconds: float
    run_trackers: dict | None
    header: dict


@dataclasses.dataclass(frozen=True)
class TreeRestartInfo:
    """Validated all-domain restore result."""

    elapsed_ticks: int
    tick_den: int
    phase: str
    paths_by_grid_id: dict[int, Path]
    headers_by_grid_id: dict[int, dict]
    #: The restore point IS this configuration's stop tick, so the run
    #: this checkpoint came from finished.  The state is restored exactly
    #: as it would be for any other resume; there is simply nothing left
    #: to integrate, and the route finalizes instead of stepping.
    already_complete: bool = False


@dataclasses.dataclass(frozen=True)
class _ValidatedRestart:
    """One fully loaded member whose every semantic refusal has passed.

    Tree restore retains these payloads through the complete-set validation
    pass and applies these exact arrays afterward.  Reopening a member between
    validation and mutation would recreate the partial-restore/TOCTOU hole.
    """

    path: Path
    header: dict
    stored: dict[str, np.ndarray]
    format_version: int
    elapsed: float


def require_tree_checkpoint_legal(model, *,
                                  history_ordered: bool = False
                                  ) -> tuple[int, int]:
    """Enforce the binding PERIOD_BEGIN tree checkpoint contract.

    ``history_ordered=True`` admits history frames still in flight (D2H):
    the caller snapshots now and publishes only after every frame queued
    before the snapshot is durable
    (``PerDomainWrfoutWriters.durability_barrier``), which is what this
    count protects -- a checkpoint on disk ahead of a frame that is not.
    """
    from woof.core.model import PERIOD_BEGIN

    status = getattr(model, "_runtime_status", None)
    if status is None:
        raise RestartMismatchError(
            "tree checkpoint has no model runtime phase state")
    if status.schedule_cursor != PERIOD_BEGIN:
        raise RestartMismatchError(
            "tree checkpoint is legal only at explicit PERIOD_BEGIN; "
            f"schedule_cursor={status.schedule_cursor!r}")
    pending = {
        "FORCE": int(status.pending_force),
        "FEEDBACK": int(status.pending_feedback),
        "D2H": int(status.pending_d2h),
        "mutation": int(bool(status.mutation_in_progress)),
    }
    io_manager = getattr(model, "_io_manager", None)
    if io_manager is not None:
        pending["D2H"] = max(pending["D2H"], int(io_manager.pending))
    if history_ordered:
        pending.pop("D2H")
    active = {name: count for name, count in pending.items() if count}
    if active:
        raise RestartMismatchError(
            f"tree checkpoint has pending work {active}; drain/commit it "
            "before PERIOD_BEGIN publication")
    if not bool(status.prior_feedback_committed):
        raise RestartMismatchError(
            "tree checkpoint requires all prior-period feedback committed")

    nodes = tuple(model.walk_parent_first())
    if not nodes:
        raise RestartMismatchError("tree checkpoint has no domains")
    tick_den = int(nodes[0].clock.tick_den)
    ticks = int(nodes[0].clock.ticks)
    for node in nodes:
        if int(node.clock.tick_den) != tick_den:
            raise RestartMismatchError(
                f"tree checkpoint tick denominator mismatch on "
                f"d{node.cfg.grid_id:02d}: {node.clock.tick_den} != "
                f"{tick_den}")
        if int(node.clock.ticks) != ticks:
            raise RestartMismatchError(
                f"tree checkpoint elapsed tick mismatch on "
                f"d{node.cfg.grid_id:02d}: {node.clock.ticks} != {ticks}")
    if ticks % model.schedule.period_ticks != 0:
        raise RestartMismatchError(
            f"tree checkpoint tick {ticks} is not a PERIOD_BEGIN boundary")
    return ticks, tick_den


def _fingerprint_components(model):
    """The named restart-identity components, when the route publishes them."""

    return getattr(model, "_experiment_fingerprint_components", None)


def _mix_isotropic_autoswitch_flip(stored, live) -> bool:
    """Did any domain go anisotropic-in-the-checkpoint to isotropic-live?

    Read off the named ``experiment_identity`` components, defensively:
    these are checkpoint-header payloads, and a malformed one must
    produce ``False`` (no notice), never a second failure inside a
    refusal message.  An absent stored key reads as 0 -- a checkpoint
    written before the knob existed integrated the then-only value.
    """

    try:
        stored_domains = {
            domain.get("grid_id"): domain.get("run", {})
            for domain in stored.get("experiment_identity", {})
            .get("domains", ())
            if isinstance(domain, Mapping)
            and isinstance(domain.get("run"), Mapping)}
        for domain in live.get("experiment_identity", {}).get("domains", ()):
            if not (isinstance(domain, Mapping)
                    and isinstance(domain.get("run"), Mapping)):
                continue
            counterpart = stored_domains.get(domain.get("grid_id"))
            if counterpart is None:
                continue
            if (counterpart.get("mix_isotropic", 0) == 0
                    and domain["run"].get("mix_isotropic") == 1):
                return True
    except (AttributeError, TypeError):
        return False
    return False


def _correct_mode_run_flip(stored_run, live_run) -> bool:
    """Did a correct-mode key go off (absent) in the checkpoint to on live,
    under a scheme that reads it?  The 2.9.0 default flip, by run dict."""

    for key in _CORRECTION_KEYS:
        if key in _OUTPUT_ONLY_CORRECTION_KEYS:
            continue
        if (not stored_run.get(key, False) and live_run.get(key) is True
                and _correct_key_in_scope(key, live_run)):
            return True
    return False


def _correct_mode_default_flip(stored, live) -> bool:
    """:func:`_correct_mode_run_flip` over the ``experiment_identity``
    domains, read exactly as :func:`_terrain_clock_default_flip` reads them."""

    try:
        stored_domains = {
            domain.get("grid_id"): domain.get("run", {})
            for domain in stored.get("experiment_identity", {})
            .get("domains", ())
            if isinstance(domain, Mapping)
            and isinstance(domain.get("run"), Mapping)}
        for domain in live.get("experiment_identity", {}).get("domains", ()):
            if not (isinstance(domain, Mapping)
                    and isinstance(domain.get("run"), Mapping)):
                continue
            counterpart = stored_domains.get(domain.get("grid_id"))
            if counterpart is None:
                continue
            if _correct_mode_run_flip(counterpart, domain["run"]):
                return True
    except (AttributeError, TypeError):
        return False
    return False


def _terrain_clock_default_flip(stored, live, *, to="local_face") -> bool:
    """Did any domain go "measured"-in-the-checkpoint to "local_face"-live?

    ``to="pinned"``: did any domain go from either terrain clock in the
    checkpoint to "pinned" live (the 2.8.9 WRF-folder default, D8)?

    Read off the ``experiment_identity`` components exactly as
    :func:`_mix_isotropic_autoswitch_flip` reads them; an absent stored key
    is "measured", the clock every checkpoint before 2.8.8 ran by default.
    """

    try:
        stored_domains = {
            domain.get("grid_id"): domain.get("run", {})
            for domain in stored.get("experiment_identity", {})
            .get("domains", ())
            if isinstance(domain, Mapping)
            and isinstance(domain.get("run"), Mapping)}
        for domain in live.get("experiment_identity", {}).get("domains", ()):
            if not (isinstance(domain, Mapping)
                    and isinstance(domain.get("run"), Mapping)):
                continue
            counterpart = stored_domains.get(domain.get("grid_id"))
            if counterpart is None:
                continue
            stored_clock = counterpart.get("terrain_clock", "measured")
            live_clock = domain["run"].get("terrain_clock")
            if to == "pinned":
                if stored_clock != "pinned" and live_clock == "pinned":
                    return True
            elif stored_clock == "measured" and live_clock == "local_face":
                return True
    except (AttributeError, TypeError):
        return False
    return False


def _identity_matches_under_current_rules(header, model) -> bool:
    """Do stored and live identity agree once BOTH are normalised?

    The stored components were produced by the WRITING build, under
    whatever exemptions it had.  Comparing a stored digest against one the
    reading build computes therefore asks "was this hashed by this build",
    not "is this the same run" -- so every widening of the exemptions
    silently invalidates every checkpoint already on disk.

    This is the same defect as the stored ``configuration_sha256``, met a
    second time one layer out and pointing the other way: there the STORED
    side carried a hash over unexempted fields, here the stored components
    carry the fields themselves while the live side now drops them.
    Normalising both under the current rules is the general fix, and the
    reason this helper exists rather than a third bespoke exemption.

    Conservative by construction.  It can only ever ALLOW a resume the
    digest comparison already rejected, and only when every component
    matches once the fields this build no longer binds are removed from
    BOTH sides.  Anything else still refuses, by name.
    """
    stored = header.get("experiment_fingerprint_components")
    live = _fingerprint_components(model)
    if not isinstance(stored, Mapping) or not isinstance(live, Mapping):
        return False

    def normalise(payload):
        if not isinstance(payload, Mapping):
            return payload
        out = json.loads(json.dumps(payload, default=str))
        identity = out.get("experiment_identity")
        if isinstance(identity, dict):
            for domain in identity.get("domains", ()) or ():
                run = domain.get("run") if isinstance(domain, dict) else None
                if not isinstance(run, dict):
                    continue
                # Output-only toggles: the member walk and the
                # configuration digest already let a resume change them,
                # so the tree's outer identity must too, or the tree
                # refuses the change each member accepts.
                for name in CONFIG_DIAGNOSTIC_FIELDS:
                    run.pop(name, None)
                if run.get("use_adaptive_time_step") in (True, "True"):
                    for name in ADAPTIVE_POLICY_RUN_FIELDS:
                        run.pop(name, None)
        return out

    return normalise(stored) == normalise(live)


def tree_fingerprint_mismatch_reason(gid: int, header, model) -> str:
    """Name what actually differs, and what a restart is allowed to change.

    ``woof run --restart`` publishes a tolerance -- only the forecast
    length and the output/restart cadence may differ -- and until 1.4.1
    the tree route enforced something strictly narrower without saying
    so, refusing every one of those three changes as nine words and a
    traceback.  The tolerance is honoured now; this message is what the
    remaining, genuine mismatches say.
    """

    stored = header.get("experiment_fingerprint_components")
    live = _fingerprint_components(model)
    prefix = f"tree restart d{gid:02d} was written for a different run"
    if isinstance(header.get("relocation"), Mapping):
        # The checkpoint was written AFTER a nest relocation: its
        # fingerprint is chained to the move history, no fresh build
        # computes it, and that refusal is the ruled posture rather than
        # an unexplained hash.
        moved = header["relocation"]
        # NAME WHAT DIFFERS HERE TOO.  This branch used to stop at the
        # move count, so every fingerprint mismatch on a relocated tree
        # read as "relocating runs cannot be resumed" -- which is not
        # true, and sent the diagnosis of a base-hash change into the
        # move history for hours.  The chain is replayed from the
        # checkpoint's own stored base by the tree runner before this is
        # ever reached, so by the time this speaks the move history has
        # already been ruled out.
        detail = ""
        if isinstance(stored, Mapping) and isinstance(live, Mapping):
            _moved_absent = object()
            changed = sorted(
                name for name in set(stored) | set(live)
                if stored.get(name, _moved_absent)
                != live.get(name, _moved_absent))
            detail = (f"  These components differ: {', '.join(changed)}."
                      if changed else
                      "  Every named component matches, so the digest "
                      "itself moved: the checkpoint predates this "
                      "restart-identity format.")
        return (f"{prefix}: it was written after "
                f"{moved.get('moves')} nest relocation(s) "
                f"(segment {moved.get('segment_id')!r}).{detail}  "
                f"{moved.get('posture', '')}")
    if not isinstance(stored, Mapping) or not isinstance(live, Mapping):
        return (f"{prefix} (experiment fingerprint mismatch); a checkpoint "
                "resumes only into the run that wrote it")
    _absent = object()
    differing = sorted(
        name for name in set(stored) | set(live)
        if stored.get(name, _absent) != live.get(name, _absent))
    if not differing:
        # Equal components, unequal digest: the digest itself moved, which
        # is a format change rather than a configuration change.
        return (f"{prefix} (fingerprint differs but every named component "
                "matches; the checkpoint predates this restart-identity "
                "format and must be rerun from the start)")
    named = ", ".join(differing)
    reason = (
        f"{prefix}: {named} differ(s) from the checkpoint.  A restart "
        "may change the forecast length, the output/restart cadence, "
        "each domain's history window (history_begin_s, history_end_s) "
        "and the output-only diagnostic switches "
        f"({', '.join(sorted(CONFIG_DIAGNOSTIC_FIELDS))}); "
        "everything else -- geometry, timestep, physics, nesting, "
        "prepared inputs -- must be the run that wrote the checkpoint")
    # Same restart accuracy as the single-domain door: when the moved
    # piece is the mixing length going 0 -> 1, the changed DEFAULT (the
    # 2026-08-16 auto-switch) is named beside the refusal, with the
    # remedy, instead of leaving "experiment_identity differs" to be
    # reverse-engineered.
    if ("experiment_identity" in differing
            and _mix_isotropic_autoswitch_flip(stored, live)):
        reason += "\n" + MIX_ISOTROPIC_RESTART_BREAK_NOTICE
    if ("experiment_identity" in differing
            and _terrain_clock_default_flip(stored, live)):
        reason += "\n" + TERRAIN_CLOCK_RESTART_BREAK_NOTICE
    if ("experiment_identity" in differing
            and _terrain_clock_default_flip(stored, live, to="pinned")):
        reason += "\n" + TERRAIN_CLOCK_PINNED_RESTART_NOTICE
    if ("experiment_identity" in differing
            and _correct_mode_default_flip(stored, live)):
        reason += "\n" + CORRECT_MODE_RESTART_BREAK_NOTICE
    return reason


#: Provenance on a domain's member, written ``true`` and only where that
#: domain's ``epssm`` was the model's choice (``ExperimentConfig
#: .auto_epssm``: unset or ``"auto"``).  The value that ran is in
#: ``config``, where it binds; this says who chose it, so a child derived
#: from the checkpoint (``woof downscale --point``) leaves its own
#: ``epssm`` to the model on its own ground instead of carrying its
#: parent's as a written value (A181).  Absent everywhere else, so every
#: explicit run's member keeps the keys it always had, and no identity
#: reads it.
AUTO_EPSSM_HEADER_KEY = "auto_epssm"


def auto_epssm_header(grid_id: int, labels) -> dict:
    """``{AUTO_EPSSM_HEADER_KEY: True}`` for a labelled domain, else ``{}``."""

    if int(grid_id) in {int(gid) for gid in (labels or ())}:
        return {AUTO_EPSSM_HEADER_KEY: True}
    return {}


#: Where the nest-lifecycle state lives in a tree checkpoint: the ROOT
#: member's ``tree_header``, beside the fingerprint and the domain set.
#: The root is the set's commit marker, so it is the one member a
#: whole-tree fact can live on without being written N times and without
#: a child member ever disagreeing with its parent.
NEST_LIFECYCLE_HEADER_KEY = "nest_lifecycle"

#: The block's version. A reader that does not know this string refuses by
#: name rather than reading a field it recognises out of a shape it does
#: not: a lifecycle block half-understood is a slot that silently re-fires
#: or a follower that resumes on a history that did not happen.
NEST_LIFECYCLE_CONTRACT = "gpuwm-nest-lifecycle-restart.v1"


def lifecycle_window_slots(state) -> tuple[str, ...]:
    """The consumer tracking windows one domain actually allocated.

    Sorted, so a checkpoint's audit and its member set are in the same
    order on every run.  Taken from the LIVE pool rather than a name list
    because the per-follower family (``uh_follow_window.dNN``) is
    generated per declared child; ``nwp_diagnostics = 0`` allocates none
    of them and gets an empty tuple.
    """
    pool = getattr(state, "_scratch", {})
    return tuple(sorted(slot for slot in pool
                        if _is_tracker_window_slot(slot)))


def _lifecycle_checkpoint_slots(state) -> tuple[str, ...]:
    """Actual held signals and windows selected for a lifecycle checkpoint.

    Keep ``lifecycle_window_slots`` restricted to 2-D UH windows: streaming
    uses it to allocate those windows on each tile. Reflectivity is a 3-D
    microphysics-time volume, already allocated by its producer or transport.
    No new plane is fabricated here, and the driver handoff stays consumed.
    """
    slots = set(lifecycle_window_slots(state))
    pool = getattr(state, "_scratch", {})
    slots.update(slot for slot in LIFECYCLE_HELD_SCRATCH_SLOTS
                 if pool.get(slot) is not None)
    return tuple(sorted(slots))


def _require_held_lifecycle_reflectivity(model, nodes, headers, validated,
                                        stored_slots) -> None:
    """Reject pre-persistence checkpoints before a primed zero can steer.

    After the parent's first microphysics-time history diagnostic, a UH
    fallback or reflectivity consumer requires that exact held value. A
    streamed destination's allocation is transport capacity, never evidence
    that the old checkpoint carried a producer value. Before the first due
    history, no held value exists yet and normal production creates it.

    The slot and its store key are read from ``woof.core.streaming``,
    which owns the ``refl_10cm`` handoff for every streamed route; audit
    R-052's single-spelling rule is only true if the modules outside that
    package stop typing the string, and this reader was one of the two
    that still did.
    """
    from woof.core.streaming import REFL_SCRATCH_SLOT, REFL_STORE_KEY

    parents = set()
    for gid, runner in lifecycle_followers(model).items():
        follow = getattr(runner.config, "follow", None)
        if getattr(follow, "field", None) not in ("uh", "reflectivity"):
            continue
        node = nodes.get(gid)
        if node is not None and node.parent is not None:
            parents.add(int(node.parent.cfg.grid_id))
    exp = getattr(model, "_declared_experiment", None)
    for domain in getattr(exp, "domains", ()):
        if any(getattr(getattr(domain, name, None), "trigger", None)
               == "reflectivity" for name in ("spawn", "retire")):
            parents.add(int(domain.parent_id))
    for gid in sorted(parents & set(nodes)):
        spec = nodes[gid].clock.spec
        history_ticks = getattr(spec, "history_ticks", None)
        # Reduced state-only callers have no diagnostic production clock.
        if history_ticks is None:
            continue
        first_due = (int(spec.first_history_ticks())
                     if hasattr(spec, "first_history_ticks")
                     else int(spec.start_ticks) + int(history_ticks))
        if int(headers[gid]["elapsed_ticks"]) < first_due:
            continue
        if (REFL_SCRATCH_SLOT not in stored_slots.get(str(gid), ())
                or REFL_STORE_KEY not in validated[gid].stored):
            raise RestartMismatchError(
                f"checkpoint predates held lifecycle reflectivity on d{gid:02d}: "
                "the configured consumer needs the actual microphysics-time "
                f"{REFL_STORE_KEY} volume, which this file did not persist; "
                "a primed zero or recomputed field is not a continuation. "
                "Restart from the prepared state; no domain was restored")


def lifecycle_followers(model) -> dict[int, object]:
    """``{grid_id: runner}`` for every follower this run drives.

    ``RelocationRunnerCollection`` keys its own runners by grid id; a bare
    legacy runner arrives unwrapped and is keyed by its config's target.
    """
    runner = getattr(model, "_relocation_runner", None)
    if runner is None:
        return {}
    if getattr(runner, "is_collection", False):
        return {int(gid): value for gid, value in runner.runners.items()}
    return {int(runner.config.grid_id): runner}


#: The writer's spelling, as one object: the resume seeds exactly the
#: runners the checkpoint's follower entries were taken from.
_lifecycle_followers = lifecycle_followers


def declares_nest_lifecycle(model) -> bool:
    """Does this run have a nest lifecycle to persist at all?

    The compatibility predicate, stated once: a run that constructs no
    ``SpawnRunner`` and no follow provider writes exactly the checkpoint
    this module has always written, and nothing on the lifecycle path --
    not the block, not the window members, not the streamed publish --
    may touch it.
    """
    return (getattr(model, "_spawn_runner", None) is not None
            or bool(_lifecycle_followers(model)))


#: The WRITER's spelling of the predicate, kept as one object rather than
#: one behaviour written twice.  Two spellings of "does this run have a
#: lifecycle" drift, and the drift is a checkpoint one side writes and the
#: other refuses to read.
_declares_nest_lifecycle = declares_nest_lifecycle


def _declared_domain(model, grid_id: int):
    """The ``[[domain]]`` the EXPERIMENT declares for one grid, or None.

    Read off the tree's published declaration
    (:func:`woof.core.model.publish_declared_experiment`) and nowhere
    else.  This used to reach into one route's build-time ingest bundle,
    which made checkpointing a run with a live follower work only for
    trees that route had built -- a capability restriction shaped like a
    route rather than like anything a checkpoint needs.
    """
    exp = getattr(model, "_declared_experiment", None)
    if exp is None:
        return None
    for domain in getattr(exp, "domains", ()):
        if int(getattr(domain, "grid_id", -1)) == int(grid_id):
            return domain
    return None


def _follower_entry(model, nodes_by_gid, grid_id: int, runner) -> dict:
    """One follower's checkpoint entry: the runner's four keys plus three.

    The runner owns ``segment``/``moves_executed`` and the two cooldown
    anchors -- its own history, which nothing else can state.  The other
    three describe the TREE and the runner cannot invent them: which
    surface declared this follower, where the child sits NOW (only
    ``node.cfg`` knows, because ``relocate_child`` mutates it in place),
    and where the experiment declared it.  Merged here so the entry is
    written whole and ``RelocationRunner.restore_state`` takes it whole.
    """
    entry = dict(runner.state_json())
    declared = _declared_domain(model, grid_id)
    if declared is None:
        raise RestartManifestError(
            f"this run follows d{grid_id:02d} but the tree carries no "
            "declared experiment to say where that domain was configured; "
            "the checkpoint would record a placement without saying whether "
            "it is the declared one or one the follower moved to, and a "
            "resume cannot then tell a moved nest from a reconfigured "
            "one.  Whichever route builds the tree publishes it "
            "(woof.core.model.publish_declared_experiment)")
    node = nodes_by_gid[int(grid_id)]
    entry.update({
        "kind": ("per-domain" if getattr(declared, "follow", None) is not None
                 else "legacy"),
        "current_placement": [int(node.cfg.i_parent_start),
                              int(node.cfg.j_parent_start)],
        "declared_placement": [int(declared.i_parent_start),
                               int(declared.j_parent_start)],
    })
    return entry


def _publish_streamed_lifecycle_windows(nodes) -> None:
    """Project a STREAMED domain's tracker windows onto its state first.

    A streamed domain's arrays live in its pinned store and its
    ``DomainState`` stops changing at attach, so serializing the state's
    copy at a mid-leg checkpoint instant writes the ATTACH-TIME zeros
    under the name of a live window -- and a resume then under-reads the
    next spawn/retire/follow decision with nothing anywhere saying so.
    The leg walk publishes the same planes at every leg boundary
    (``woof/runtime.py``); a checkpoint does not land on one.

    Recognised through the ``_streamed_domain`` marker
    :class:`woof.core.streaming.StreamedDomain` leaves on the state it
    took over, which is the door every reader that holds a NODE rather
    than a stepper already uses (``woof/io/wrfout.py``); the checkpoint
    writer holds nodes.

    Every allocated tracker slot uses the same carrier contract, including
    generated per-follower names. The store remains the value authority.
    """
    from woof.core import streaming as _streaming

    for node in nodes:
        streamed = getattr(node.state, "_streamed_domain", None)
        if streamed is None:
            continue
        names = tuple(f"scratch/{slot}" for slot in _lifecycle_checkpoint_slots(node.state))
        present = _streaming.allocated_planes(node.state, names)
        if present:
            streamed.publish(present)


def _nest_lifecycle_header(model, nodes) -> dict | None:
    """The ``nest_lifecycle`` block, or ``None`` for a lifecycle-free run.

    ``None`` is the compatibility contract and it is the reason the whole
    block is assembled here rather than unconditionally: a run that
    constructs no ``SpawnRunner`` and no follow provider writes exactly
    the checkpoint this module has always written, byte for byte, and
    every checkpoint already on disk keeps restoring.
    """
    if not _declares_nest_lifecycle(model):
        return None
    spawn_runner = getattr(model, "_spawn_runner", None)
    followers = _lifecycle_followers(model)
    nodes_by_gid = {int(node.cfg.grid_id): node for node in nodes}
    follower_entries: dict[str, dict] = {}
    for gid, runner in sorted(followers.items()):
        if gid in nodes_by_gid:
            follower_entries[str(gid)] = _follower_entry(
                model, nodes_by_gid, gid, runner)
            continue
        # A follow target that is still DORMANT has a runner and no
        # domain.  Nothing has been consulted and nothing has moved, so
        # the accurate statement is no entry at all -- a resume builds
        # that follower fresh, which is what a never-consulted runner is.
        # A runner that HAS moved and has no domain is a different thing
        # entirely and is refused: its segment chain addresses a nest the
        # checkpoint cannot place.
        moved = (int(getattr(runner, "moves_executed", 0) or 0)
                 or runner.state_json().get("segment") is not None)
        if moved:
            raise RestartManifestError(
                f"this run's follower for d{gid:02d} reports a move "
                "history, but that domain is not in this checkpoint's "
                "member set; the follower's CURRENT placement lives on the "
                "live node and nowhere else, so the block would carry a "
                "segment chain with no placement to chain it onto")
    leg = getattr(model, "_spawn_leg_seconds", None)
    if leg is not None:
        leg = float(leg)
        if not math.isfinite(leg) or leg <= 0.0:
            raise RestartManifestError(
                f"the leg cadence this run stops at is {leg!r}; the "
                "checkpoint cross-checks it against the resuming config's "
                "own cadence, and a non-finite one would make every resume "
                "either refuse or compare against nothing")
    return {
        "contract": NEST_LIFECYCLE_CONTRACT,
        # Cross-checked at resume against the resuming config's cadence: a
        # leg boundary is where every lifecycle decision is taken, so two
        # runs on different lattices take them at different instants.
        # ``None`` says this run had no leg walk to state a cadence for.
        "leg_seconds": leg,
        # ``None``, not ``{}``: an empty mapping reads as "a spawn runner
        # that knows nothing", which is what a slot silently re-firing
        # looks like from the reader's side.
        "spawn": (None if spawn_runner is None
                  else spawn_runner.state_json(model=model)),
        "followers": follower_entries,
        # The audit half of the window opt-in: which planes this
        # checkpoint carries, per domain, so a reader can tell "no window
        # was written" from "the window was written and it is empty".
        "window_slots": {
            str(int(node.cfg.grid_id)): list(slots)
            for node in nodes
            for slots in (_lifecycle_checkpoint_slots(node.state),) if slots},
    }


#: The block's key set, stated once.  A block with a key this build does
#: not know is refused rather than read past: an unread field is a slot
#: that silently re-fires or a follower resuming on a history that did not
#: happen, and neither says anything in the log.
NEST_LIFECYCLE_BLOCK_KEYS = ("contract", "followers", "leg_seconds",
                             "spawn", "window_slots")


@dataclasses.dataclass(frozen=True)
class TreeLifecycleHeader:
    """What a resume needs off a checkpoint BEFORE it rebuilds anything.

    One JSON member read, from the ROOT of the set: which slots fired,
    where they fired, which followers exist and what their move history
    was.  The tree is then rebuilt from this and only then is the state
    applied, because the member set a restore validates against is the
    set the reconstructed tree produces.
    """

    #: The ROOT member the block was read from, which is also the member
    #: :func:`restore_tree_restart` must be handed.
    root_path: Path
    #: The domain set this checkpoint carries, from the root's header.
    domain_ids: tuple[int, ...]
    #: The validated ``nest_lifecycle`` block, or ``None`` for a
    #: lifecycle-free checkpoint restored into a lifecycle-free run.
    block: dict | None
    #: The ``relocation`` posture block, when this set was written after
    #: at least one executed move.
    relocation: dict | None

    @property
    def spawn(self) -> dict | None:
        """The ``spawn`` sub-block, or ``None`` for a follower-only run."""
        return None if self.block is None else self.block["spawn"]

    @property
    def followers(self) -> dict:
        return {} if self.block is None else dict(self.block["followers"])

    @property
    def window_slots(self) -> dict:
        return {} if self.block is None else dict(self.block["window_slots"])

    @property
    def leg_seconds(self):
        return None if self.block is None else self.block["leg_seconds"]

    @property
    def relocation_records(self) -> tuple[str, ...]:
        """The move record digests, in the order the run chained them.

        The live fingerprint of the run that wrote this set is the fresh
        build's fingerprint folded over exactly this list
        (``mark_fingerprint_across_move``), so a legitimate resume
        reproduces it and nothing else can.
        """
        if self.relocation is None:
            return ()
        return tuple(str(sha)
                     for sha in self.relocation.get("record_sha256") or ())


def _tree_restart_root_member(path: Path) -> Path:
    """The ROOT member of the set ``path`` belongs to.

    The lifecycle block is a whole-tree fact and rides the root alone, so
    a caller handed any member (a supervisor scanning a directory finds
    whichever it finds first) still reaches it.  The root is the member
    whose declared parent is 0 -- the experiment's own root convention --
    not the lowest grid id, which is a coincidence of the shipped configs.
    """
    match = _TREE_RESTART_NAME.fullmatch(path.name)
    if match is None:
        raise RestartMismatchError(
            f"tree restart path {path} is not gpuwmrst_d0X_<instant>.npz, "
            "so the sibling members of its checkpoint set cannot be found "
            "and the whole-tree lifecycle block cannot be located")
    instant = match.group("instant")
    roots = []
    for candidate in sorted(path.parent.glob(f"gpuwmrst_d*_{instant}")):
        if _TREE_RESTART_NAME.fullmatch(candidate.name) is None:
            continue
        if int(read_restart_header(candidate).get("parent_id", -1)) == 0:
            roots.append(candidate)
    if len(roots) != 1:
        raise RestartMismatchError(
            f"the checkpoint set at instant {instant} carries "
            f"{len(roots)} root member(s) (parent_id = 0); the root is the "
            "set's commit marker and the one member the nest-lifecycle "
            "block rides, so a set without exactly one cannot say which "
            "slots fired")
    return roots[0]


def _refuse_lifecycle_grid(gid: int, reason: str) -> None:
    raise RestartMismatchError(
        f"this checkpoint's nest-lifecycle block names d{gid:02d}, {reason}")


def read_tree_lifecycle_header(path, model=None) -> TreeLifecycleHeader:
    """Peek at a tree checkpoint's lifecycle state, before reconstruction.

    Cheap on purpose: npz members are lazy, so this reads ONE JSON member
    off the root and nothing else -- which is what lets a resume decide
    what tree to build before it has built one.

    ``model`` is the run about to be resumed, and supplies the two facts
    the file cannot: whether this experiment declares a lifecycle at all
    (the same predicate the writer used, :func:`declares_nest_lifecycle`,
    as the same object) and the leg cadence it will stop on.  Omit it for
    a bare inspection: without an experiment there is nothing to compare
    against and this must not invent one.

    Every refusal below names the run it prevents, because each is a run
    that would otherwise integrate from policy state it half understands.
    """
    path = Path(path)
    header = read_restart_header(path)
    if int(header.get("parent_id", -1)) != 0:
        path = _tree_restart_root_member(path)
        header = read_restart_header(path)
    domain_ids = tuple(int(gid) for gid in header.get("domain_ids") or ())
    relocation = header.get("relocation")
    if not isinstance(relocation, Mapping):
        relocation = None
    else:
        relocation = dict(relocation)
    declares = None if model is None else declares_nest_lifecycle(model)
    block = header.get(NEST_LIFECYCLE_HEADER_KEY)

    if block is None:
        if declares:
            raise RestartMismatchError(
                f"{path.name} carries no nest-lifecycle block: this "
                "checkpoint predates lifecycle persistence; it cannot say "
                "which slots fired -- restart from t=0.  This experiment "
                "declares a dormant nest or a follower, so a resume would "
                "restore the tree and then re-fire every slot at the first "
                "leg boundary, from a policy state that is not the one the "
                "checkpoint was taken in")
        return TreeLifecycleHeader(
            root_path=path, domain_ids=domain_ids, block=None,
            relocation=relocation)

    if not isinstance(block, Mapping):
        raise RestartMismatchError(
            f"{path.name}'s nest-lifecycle block is a "
            f"{type(block).__name__}, not a mapping; nothing in it can be "
            "read, so the slots it was meant to describe would resume "
            "un-fired")
    block = dict(block)
    if declares is False:
        raise RestartMismatchError(
            f"{path.name} carries a nest-lifecycle block, but this "
            "experiment declares no dormant nest and no follower, so the "
            "run being resumed constructs neither a SpawnRunner nor a "
            "follow provider.  Every fired slot, episode count and move "
            "history in the block would be dropped on the floor and the "
            "restored tree would integrate as a plain static nest set.  "
            "Resume the experiment that wrote this checkpoint, or start "
            "from t=0")

    contract = block.get("contract")
    if contract != NEST_LIFECYCLE_CONTRACT:
        raise RestartMismatchError(
            f"{path.name}'s nest-lifecycle block states contract "
            f"{contract!r}; this build reads "
            f"{NEST_LIFECYCLE_CONTRACT!r}.  A block half-understood is a "
            "slot that silently re-fires or a follower that resumes on a "
            "history that did not happen, so an unknown version is refused "
            "rather than read for the fields whose names happen to match")
    unknown = sorted(set(block) - set(NEST_LIFECYCLE_BLOCK_KEYS))
    missing = sorted(set(NEST_LIFECYCLE_BLOCK_KEYS) - set(block))
    if unknown or missing:
        raise RestartMismatchError(
            f"{path.name}'s nest-lifecycle block carries key(s) {unknown} "
            f"this build does not know and is missing {missing}, while "
            f"stating contract {NEST_LIFECYCLE_CONTRACT!r}.  The block is "
            "honored whole or refused: a field read past is policy state "
            "the resumed run silently does not have")

    members = set(domain_ids)
    spawn = block["spawn"]
    if spawn is not None:
        if not isinstance(spawn, Mapping):
            raise RestartMismatchError(
                f"{path.name}'s nest-lifecycle spawn block is a "
                f"{type(spawn).__name__}, not a mapping or null")
        for raw in sorted(dict(spawn.get("spawned") or {})):
            gid = int(raw)
            if gid not in members:
                _refuse_lifecycle_grid(
                    gid, "as a LIVE episode, but that domain is not in this "
                    f"checkpoint's member set {sorted(members)}; its arrays "
                    "exist in no member, so the rebuilt nest would carry "
                    "the parent interpolation for the rest of the run with "
                    "nothing saying so")
        for raw in sorted(dict(spawn.get("retired") or {})):
            gid = int(raw)
            if gid in members:
                _refuse_lifecycle_grid(
                    gid, "as RETIRED, but that domain still owns a member "
                    "of this checkpoint.  Retirement detaches the subtree "
                    "at the leg boundary, before any later checkpoint, so "
                    "the two halves of this set describe different trees "
                    "and a restore would put a retired nest back in the "
                    "next leg's active set")
    for raw in sorted(dict(block["followers"]), key=int):
        gid = int(raw)
        if gid not in members:
            _refuse_lifecycle_grid(
                gid, "as a follower with a move history, but that domain "
                f"is not in this checkpoint's member set {sorted(members)}; "
                "the follower's CURRENT placement lives on the live node "
                "and nowhere else, so its segment chain addresses a nest "
                "this checkpoint cannot place")
    for raw in sorted(dict(block["window_slots"]), key=int):
        gid = int(raw)
        if gid not in members:
            _refuse_lifecycle_grid(
                gid, "as the owner of consumer tracking windows, but that "
                f"domain is not in this checkpoint's member set "
                f"{sorted(members)}; the audit and the members it audits "
                "disagree, so "
                "\"the window was written and it is empty\" cannot be told "
                "from \"no window was written\"")

    stored_leg = block["leg_seconds"]
    live_leg = None if model is None else getattr(
        model, "_spawn_leg_seconds", None)
    if stored_leg is not None and live_leg is not None:
        stored_leg = float(stored_leg)
        live_leg = float(live_leg)
        if abs(stored_leg - live_leg) > 1.0e-9 * max(1.0, abs(stored_leg)):
            raise RestartMismatchError(
                f"{path.name} was written by a run that stops every "
                f"{stored_leg:g} s to take its lifecycle decisions; this "
                f"run stops every {live_leg:g} s.  Every spawn, retire and "
                "re-arm decision is taken at a leg boundary, so the two "
                "runs evaluate the same policy at different instants and "
                "the resumed trajectory is no continuation of the one on "
                "disk.  Restore under the cadence that wrote it "
                "([relocation] cadence_seconds, else the root's "
                "history_interval_s), or start from t=0")

    return TreeLifecycleHeader(
        root_path=path, domain_ids=domain_ids, block=block,
        relocation=relocation)


def _node_placement(node):
    """``{i_parent_start, j_parent_start, parent_grid_ratio}`` or None.

    The root has no placement of its own, and a hand-assembled tree (the
    idealized cases, and every test that builds a node out of a
    ``SimpleNamespace``) may carry a config that has none either.  Both
    read as None: a header field must never be the reason a checkpoint
    cannot be written.
    """
    if getattr(node, "parent", None) is None:
        return None
    cfg = node.cfg
    fields = ("i_parent_start", "j_parent_start", "parent_grid_ratio")
    if any(getattr(cfg, name, None) is None for name in fields):
        return None
    try:
        return {name: int(getattr(cfg, name)) for name in fields}
    except (TypeError, ValueError):
        return None


class PendingTreeCheckpoint:
    """One checkpoint generation, snapshotted and not yet on disk.

    Built by ``write_tree_restart(..., defer_publish=True)``.  Every member
    was validated and its header and arrays fixed on the stepping thread;
    :meth:`publish` writes the members in the synchronous writer's order
    (children first, the root commit marker last), removes the generation's
    members on any failure, then records the root on the model and retires
    superseded sets, exactly as the synchronous writer does.

    A publish on a background writer thread passes ``retire=False`` and the
    stepping thread calls :meth:`retire_superseded` once it has joined the
    writer (``_CheckpointPublisher.wait``).  The stepping thread is the one
    that hands ``model._last_checkpoint`` to the run's progress callback,
    and the supervisor heartbeat reads that set back in full
    (``validate_manifest_checkpoint``) on that thread.  Retiring from the
    writer thread removed the set the heartbeat was still reading: with
    hourly checkpoints and ``--keep-checkpoints 1`` the 08Z publish deleted
    the 07Z set mid-read and the run died with FileNotFoundError on its own
    checkpoint 3-7 h into every 60 h GFS 12 km parent (ml-probs T1,
    2026-10-27).

    ``deferrable`` says whether :meth:`publish` may run while the model
    steps on.  A resident member owns host copies of its arrays, and a
    member of a ranked domain borrows the domain's host store, which only a
    drain or a frame download writes; both of those wait for this
    generation first (:meth:`tilestream.ranks.RankedRun.add_store_guard`).
    Any other streamed member borrows a store the next sweep writes, so a
    generation holding one must be published before stepping resumes.
    """

    def __init__(self, *, directory: Path, model, root_id: int, members,
                 lenders=(), deferrable: bool = True):
        self.directory = Path(directory)
        self._model = model
        self._root_id = int(root_id)
        self._members = list(members)
        self.deferrable = bool(deferrable)
        self._released = threading.Event()
        for streamed in lenders:
            streamed.add_store_guard(self._released.wait)

    def publish(self, *, retire: bool = True) -> Path:
        published: list[Path] = []
        paths: dict[int, Path] = {}
        try:
            for gid, member in self._members:
                paths[gid] = Path(member())
                published.append(paths[gid])
        except BaseException:
            for path in published:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        finally:
            # The borrowed store views are dropped and their guard released
            # whether or not the write succeeded: a failed checkpoint must
            # not hold the next frame download forever.
            self.abandon()
        self._model._last_checkpoint = paths[self._root_id]
        if retire:
            self.retire_superseded()
        return paths[self._root_id]

    def retire_superseded(self) -> list[Path]:
        """Retire the sets this published generation supersedes.

        The run's retention decides how many go (:mod:`woof.resume`).  Call
        it on the stepping thread: see the class docstring for why a
        background writer must not.
        """
        from woof.resume import retire_superseded_checkpoints
        return retire_superseded_checkpoints(self.directory)

    def abandon(self) -> None:
        """Release the borrowed store without writing anything."""
        members, self._members = self._members, []
        for _gid, member in members:
            # A member whose host copies are still in flight into a pinned
            # slab waits for them before the slab can go back to the pool.
            release = getattr(member, "release", None)
            if callable(release):
                release()
        self._released.set()


def write_tree_restart(directory, model, valid_time: datetime, *,
                       run_trackers_by_grid_id=None,
                       sealed_forcing_extension: bool = False,
                       auto_epssm=None, defer_publish: bool = False):
    """Publish one immutable generation per domain, with d01 last.

    ``defer_publish=True`` takes the same snapshot and returns a
    :class:`PendingTreeCheckpoint` instead of writing it; the caller
    publishes it (a checkpoint writer thread, or at once).

    ``auto_epssm`` is the grid_ids whose ``epssm`` the model chose
    (:data:`AUTO_EPSSM_HEADER_KEY`); ``None`` reads them off the tree's
    declared experiment, which a door that publishes none passes itself.

    Every generation has UUID-qualified member names.  Publishing the root
    last makes it the set's commit marker: a crash while writing children
    cannot be advertised as a durable d01 checkpoint, and a failed rewrite at
    the same valid time cannot replace members of the prior committed set.  A
    restore still validates the complete sibling set before mutating state.

    A run with a NEST LIFECYCLE -- a ``SpawnRunner`` or any follow
    provider, published on the model by
    ``woof.runtime.publish_lifecycle_runners`` -- additionally writes
    :data:`NEST_LIFECYCLE_HEADER_KEY` on the root member and the consumer
    tracking windows on each domain's own member.  A run without one
    writes exactly the member set and header keys this function has
    always written, byte for byte, because every checkpoint already on
    disk was written by such a run.
    """
    from woof.core.model import PERIOD_BEGIN

    # A deferred generation is published by a caller that orders it after
    # the history frames queued before it (the prepared door's checkpoint
    # writer waits on PerDomainWrfoutWriters.durability_barrier).
    ticks, tick_den = require_tree_checkpoint_legal(
        model, history_ordered=defer_publish)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nodes = tuple(model.walk_parent_first())
    ids = sorted(int(node.cfg.grid_id) for node in nodes)
    trackers = dict(run_trackers_by_grid_id or {})
    if auto_epssm is None:
        auto_epssm = getattr(getattr(model, "_declared_experiment", None),
                             "auto_epssm", ())
    paths: dict[int, Path] = {}
    # A checkpoint written after a nest relocation says so ON ITSELF, and
    # states the posture it actually has -- which is no longer "a restart
    # across a move promises nothing".  ``RESTART_ACROSS_MOVE_POSTURE``
    # below is the current wording and this comment used to contradict
    # it: the set resumes into the run that wrote it and reproduces that
    # run bit for bit.  The live fingerprint is chained to the move
    # history, so a fresh build still refuses this set by construction;
    # the block below is what lets a legitimate resume replay that chain
    # and what lets any reader of the member state the posture by name.
    # EVERY EVENT THAT MARKED THE FINGERPRINT, not just the mover's.
    # The containment leg (`event: "contained"`) slides the mover's
    # PARENT through the same relocate_child, so it chains its own record
    # into the live fingerprint exactly as a `"relocated"` event does.
    # Listing only the mover's moves recorded 63 marks where 70 had been
    # applied, so the replay reconstructed a different value and the
    # resume was refused -- on every checkpoint of every tree that has a
    # [relocation.containment] block, which is the shape the subsystem is
    # built around.  It also left the slid parent out of
    # `moved_grid_ids`, so the restore put it back with the MOVER's
    # initializer and failed "this initializer serves grid_id 3, asked to
    # rebuild grid_id 2".
    #
    # Receipt order is chronological and the chain is order-sensitive
    # (mark_fingerprint_across_move is a one-way fold), so taking both
    # event kinds in receipt order is what reproduces the live value.
    executed_moves = [
        entry for entry in getattr(model, "_relocation_receipts", ())
        if isinstance(entry, dict)
        and entry.get("event") in ("relocated", "contained")
        and entry.get("record_sha256")]
    # THE CHAIN IS THE RUN'S, NOT THE SEGMENT'S.  A resumed run's live
    # fingerprint is the fresh build folded over EVERY move the whole run
    # has made, including the ones an earlier segment made; a checkpoint
    # that listed only this segment's moves could not be resumed from at
    # all, because folding a short chain reproduces a different value.
    # ``_restart_crossed_relocation`` is what ``restore_tree_restart``
    # left behind for exactly this.
    inherited = getattr(model, "_restart_crossed_relocation", None)
    if not isinstance(inherited, Mapping):
        inherited = None
    prior = ([] if inherited is None
             else [str(sha) for sha in inherited.get("record_sha256") or ()])
    records = prior + [str(entry["record_sha256"])
                       for entry in executed_moves]
    runner = getattr(model, "_relocation_runner", None)
    _continuity = None
    if runner is not None and hasattr(runner, "continuity_state"):
        try:
            _continuity = runner.continuity_state()
        except Exception:            # never fail a checkpoint over a diagnostic
            _continuity = None
    relocation_header = None
    if records:
        from woof.core.nest_relocation import RESTART_ACROSS_MOVE_POSTURE

        last = executed_moves[-1] if executed_moves else inherited
        relocation_header = {
            "moves": len(records),
            "record_sha256": records,
            "segment_id": last["segment_id"],
            "grid_id": int(last["grid_id"]),
            # EVERY grid that was rebuilt, not just the last one to move.
            # A resume has to put each of them back through the same
            # rebuild, and it cannot decide that from the placement
            # NUMBER: a nest that wandered away and came back sits at its
            # original i/j with a base state that is not the original
            # bytes.  Measured: phb differs by 2**-6 between the
            # prepared-cache build and a relocation rebuild at the same
            # placement, and phb is in STATE_SETUP_ARRAYS, so the setup
            # fingerprint refuses.  The history is what decides.  Folded
            # over the WHOLE run, matching the chain above: an earlier
            # segment's rebuilt grids stay rebuilt.
            "moved_grid_ids": sorted(
                set([] if inherited is None else
                    [int(g) for g in inherited.get("moved_grid_ids") or ()])
                | {int(entry["grid_id"]) for entry in executed_moves}),
            # The tracker's hysteresis and the audit's running count.
            # Geometry alone is not continuity: a resume whose cooldown
            # anchor starts cold may move on a beat the unbroken run was
            # still suppressing, which is a different forecast produced
            # by a bookkeeping scalar.  Absent when the route built no
            # runner, and the reader treats absence as "nothing to set".
            **({} if _continuity is None else {"continuity": _continuity}),
            "posture": RESTART_ACROSS_MOVE_POSTURE,
        }
    # The nest-lifecycle state, and the window planes that go with it.
    # Assembled BEFORE the UUID is assigned and before a single member is
    # written, on the same posture the sealed prefix check below takes: a
    # runner that refuses to state its block must not leave an orphan
    # member behind that looks like part of a publish attempt.  Behind
    # the compatibility predicate in full, including the publish, which
    # WRITES to a streamed domain's state: a lifecycle-free run must not
    # even be touched by this path.
    nest_lifecycle = None
    window_slots_by_gid: dict[int, tuple[str, ...]] = {}
    if _declares_nest_lifecycle(model):
        _publish_streamed_lifecycle_windows(nodes)
        nest_lifecycle = _nest_lifecycle_header(model, nodes)
        window_slots_by_gid = {
            int(node.cfg.grid_id): _lifecycle_checkpoint_slots(node.state)
            for node in nodes}
    if sealed_forcing_extension:
        # A STREAMED domain lives in its store; ``node.state`` is the
        # snapshot ``attach`` filled the store from, and nothing writes back
        # to it.  The sealed route has no store writer
        # (``StreamedDomain.write_restart`` takes no
        # ``sealed_forcing_extension``), so below it falls through to the
        # resident writer -- which, read off that snapshot, checkpointed the
        # ANALYSIS-era state under a horizon-extension header and restored
        # a trajectory that never happened: no NaN, no shape mismatch, no
        # refusal (ENG-012).  Gather the store onto the state first
        # (``refresh_state``: the whole-domain copy a history frame already
        # costs), with the exact domain clock imposed, so both the prefix
        # validation and the writer read the domain as it is.
        sealed_elapsed = ticks / tick_den
        for node in nodes:
            streamed = getattr(node.state, "_streamed_domain", None)
            if streamed is not None:
                streamed.impose_clock(sealed_elapsed)
                streamed.refresh_state()
        # Validate the whole generation before assigning its UUID or writing
        # even a child member.  A malformed root prefix must never leave an
        # orphan child that looks like part of a publish attempt.
        for node in nodes:
            _require_sealable_forcing_prefix(
                node.state, node.cfg.run,
                path=directory / f"d{int(node.cfg.grid_id):02d}",
                elapsed=sealed_elapsed)
    checkpoint_set_id = uuid.uuid4().hex
    published: list[Path] = []
    members: list = []
    lenders: list = []
    deferrable = True
    try:
        # Children first, root commit marker last.
        for node in reversed(nodes):
            gid = int(node.cfg.grid_id)
            node.state.elapsed_seconds = ticks / tick_den
            bits = int(np.float32(node.clock.dtbc_fp32).view(np.uint32))
            started = bool(getattr(node, "_started", True))
            lifecycle = "STARTED" if started else "NOT_STARTED"
            domain_start_time = (
                model.schedule.clock.start_time
                if node.cfg.start_time is None
                else node.cfg.start_time)
            tree_header = {
                "experiment_fingerprint": model.experiment_fingerprint,
                # The named components the fingerprint is a digest of,
                # when the building route publishes them.  Stored so a
                # mismatch on restore can say WHICH one moved instead of
                # reporting that a hash differs.  Absent for routes that
                # build the fingerprint some other way; the comparison
                # below degrades to the bare-hash message.
                **({} if _fingerprint_components(model) is None else {
                    "experiment_fingerprint_components":
                        _fingerprint_components(model)}),
                "checkpoint_set_id": checkpoint_set_id,
                "grid_id": gid,
                "parent_id": int(node.cfg.parent_id),
                "domain_ids": ids,
                "elapsed_ticks": ticks,
                "tick_den": tick_den,
                "phase": PERIOD_BEGIN,
                "domain_start_time": domain_start_time.isoformat(),
                "domain_start_ticks": int(node.clock.spec.start_ticks),
                "domain_lifecycle": lifecycle,
                "nest_tables": (
                    "REBUILT" if started or node.parent is None
                    else "NOT_STARTED"),
                "dtbc_fp32_bits": bits,
                # THE LIVE CLOCK, for a resume under
                # use_adaptive_time_step.  ABSENT when the feature is off,
                # on the same convention [relocation] uses below and
                # [perturbation] uses in the identity payload: a key that
                # appears as null in every checkpoint would move every
                # pre-feature digest for a field those runs never had.
                # The reader treats absent as "the configured step",
                # which is exactly what those runs were integrating.
                #
                # step_count is STORED, not derived.  The restore path
                # recomputes it as (ticks - start) // step_ticks, which is
                # exact only while every step was the same size; after a
                # run whose step varied it is simply a wrong number, and
                # it is what the physics driver's itimestep is built from.
                **({} if not bool(node.cfg.run.use_adaptive_time_step)
                   else {"adaptive_clock": {
                       "step_ticks": int(node.clock.step_ticks),
                       "dt_fp32_bits": int(
                           np.float32(node.clock.dt_fp32).view(np.uint32)),
                       "step_count": int(node.clock.step_count),
                       # The controller's own memory, including the CFL
                       # state WRF does NOT checkpoint.
                       "controller": getattr(
                           node.clock, "adaptive_state", None),
                   }}),
                # WHERE THIS DOMAIN ACTUALLY WAS.  A nest that has moved
                # is not where its config says it is, and every setup
                # array -- terrain, map factors, base state -- belongs to
                # the placement it was at, not the one it started from.
                # Without this a restore rebuilds the tree from the
                # config, lands the nest on its ORIGINAL ground, and the
                # setup fingerprint refuses (correctly) with a base-state
                # mismatch.  The root has no placement of its own and
                # writes nulls.
                "placement": _node_placement(node),
                # The ruled posture, on the member that carries it.  This
                # block was computed here and then dropped on the floor,
                # so every checkpoint written after a move claimed
                # `relocation: null` and the restore-side warning that
                # reads it could never fire.
                **({} if relocation_header is None
                   else {"relocation": relocation_header}),
                **auto_epssm_header(gid, auto_epssm),
            }
            # D3, closed: the posture block was computed and never
            # attached, so the crossed-move warning in
            # ``restore_tree_restart`` and the posture branch of
            # ``tree_fingerprint_mismatch_reason`` could never fire from
            # this writer -- a restart across a move refused as an
            # unexplained hash.  On EVERY member, because the mismatch
            # message is produced per member.
            if relocation_header is not None:
                tree_header["relocation"] = dict(relocation_header)
            # The lifecycle block is a WHOLE-TREE fact and rides the root,
            # which is the set's commit marker; children carry nothing, so
            # a child member can never disagree with its parent.
            if nest_lifecycle is not None and node.parent is None:
                tree_header[NEST_LIFECYCLE_HEADER_KEY] = nest_lifecycle
            base = Path(restart_filename(valid_time, f"d{gid:02d}"))
            member = base.with_name(
                f"{base.stem}__{checkpoint_set_id}{base.suffix}")
            path = directory / member
            streamed = getattr(node.state, "_streamed_domain", None)
            if defer_publish:
                if streamed is not None and not sealed_forcing_extension:
                    streamed.impose_clock(ticks / tick_den)
                    publish, lends = streamed.prepare_restart(
                        path, node.cfg.run, run_trackers=trackers.get(gid),
                        tree_header=tree_header,
                        extra_scratch_slots=window_slots_by_gid.get(gid, ()))
                    members.append(
                        (gid, lambda publish=publish: publish().path))
                    if lends:
                        lenders.append(streamed)
                    else:
                        deferrable = False
                else:
                    from woof.io.checkpoint_staging import StagedSnapshot
                    members.append((gid, prepare_restart(
                        path, node.state, node.cfg.run,
                        run_trackers=trackers.get(gid), tree_header=tree_header,
                        extra_scratch_slots=window_slots_by_gid.get(gid, ()),
                        sealed_forcing_extension=sealed_forcing_extension,
                        staging=StagedSnapshot())))
                continue
            if streamed is not None and not sealed_forcing_extension:
                # Publish the canonical store, without refreshing a full GPU
                # state. Bind the exact domain time, not the tile FP32 sum.
                streamed.impose_clock(ticks / tick_den)
                paths[gid] = streamed.write_restart(
                    path, node.cfg.run, run_trackers=trackers.get(gid),
                    tree_header=tree_header,
                    extra_scratch_slots=window_slots_by_gid.get(gid, ())).path
            else:
                # Resident, or streamed under the sealed extension -- where
                # node.state was refreshed from the store above (ENG-012).
                paths[gid] = write_restart(
                    path, node.state, node.cfg.run,
                    run_trackers=trackers.get(gid), tree_header=tree_header,
                    extra_scratch_slots=window_slots_by_gid.get(gid, ()),
                    sealed_forcing_extension=sealed_forcing_extension)
            published.append(paths[gid])
    except BaseException:
        # These names are unique to this uncommitted generation, so cleanup
        # can never remove a member referenced by an older root marker.
        for path in published:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        # Snapshotted members with host copies still in flight wait for
        # them before their pinned memory can be reused.
        for _gid, member in members:
            release = getattr(member, "release", None)
            if callable(release):
                release()
        raise
    root_id = int(model.root.cfg.grid_id)
    if defer_publish:
        return PendingTreeCheckpoint(
            directory=directory, model=model, root_id=root_id,
            members=members, lenders=lenders, deferrable=deferrable)
    model._last_checkpoint = paths[root_id]
    # Only after the whole new set is published: the sets it supersedes
    # go when the run's retention says so (woof.resume), never before.
    from woof.resume import retire_superseded_checkpoints
    retire_superseded_checkpoints(directory)
    return paths[root_id]


_TREE_RESTART_NAME = re.compile(
    r"^gpuwmrst_d(?P<grid_id>[0-9]+)_(?P<instant>.+\.npz)$")


def _tree_restart_paths(path: Path, expected_ids: set[int]
                        ) -> dict[int, Path]:
    match = _TREE_RESTART_NAME.fullmatch(path.name)
    if match is None:
        raise RestartMismatchError(
            f"tree restart path {path} is not gpuwmrst_d0X_<instant>.npz")
    instant = match.group("instant")
    found: dict[int, Path] = {}
    for candidate in path.parent.glob(f"gpuwmrst_d*_{instant}"):
        parsed = _TREE_RESTART_NAME.fullmatch(candidate.name)
        if parsed is not None:
            gid = int(parsed.group("grid_id"))
            if gid in found:
                raise RestartMismatchError(
                    f"duplicate tree restart file for grid_id={gid}")
            found[gid] = candidate
    if set(found) != expected_ids:
        raise RestartMismatchError(
            "tree restart refuses a partial/mismatched domain set: "
            f"expected {sorted(expected_ids)}, found {sorted(found)} for "
            f"instant {instant}")
    return found


def tree_restart_members(path) -> dict[int, Path]:
    """Every member of the checkpoint set ``path`` belongs to, by grid id.

    ``path`` is any one member, usually the root the writer returned.
    The set is what the member's own header declares (``domain_ids``),
    found beside it by the instant in its name, and a set with a member
    missing is a refusal rather than a shorter dictionary: a caller
    copying a generation elsewhere must not carry half a tree.
    Header-only, so no array is loaded.
    """
    path = Path(path)
    header = read_restart_header(path)
    ids = header.get("domain_ids")
    if not isinstance(ids, list) or not ids:
        raise RestartMismatchError(
            f"tree restart {path} declares no domain_ids; it is not a "
            "member of a tree checkpoint set")
    return _tree_restart_paths(path, {int(gid) for gid in ids})


def checkpoint_placements(path, expected_ids) -> dict:
    """Where each domain ACTUALLY WAS when this checkpoint was written.

    Returned as ``{grid_id: {"i_parent_start": .., "j_parent_start": ..,
    "parent_grid_ratio": ..}}``, with the root and any member written
    before placements were recorded omitted.

    A caller uses this BEFORE :func:`restore_tree_restart`: a nest that
    moved is not where its config puts it, and every setup array it owns
    -- terrain, map factors, base state -- belongs to the placement it
    was at.  Restoring onto a tree built from the config alone lands the
    nest on its original ground, and :func:`setup_fingerprint` refuses
    that (correctly) as a base-state mismatch.  Reading the placement out
    first is what lets the caller rebuild the tree where the checkpoint
    says it was.

    Header-only: no arrays are loaded, so this is cheap enough to run
    before deciding anything.
    """
    members = _tree_restart_paths(Path(path), set(expected_ids))
    out = {}
    for gid, member in members.items():
        placement = read_restart_header(member).get("placement")
        if isinstance(placement, Mapping):
            out[int(gid)] = dict(placement)
    return out


def restore_tree_restart(path, model, *,
                         sealed_forcing_extension: bool = False
                         ) -> TreeRestartInfo:
    """Validate the complete current-format set, then restore all domains."""
    from woof.core.model import PERIOD_BEGIN, ModelRuntimeStatus

    path = Path(path)
    nodes = {int(node.cfg.grid_id): node
             for node in model.walk_parent_first()}
    paths = _tree_restart_paths(path, set(nodes))
    stored_lifecycle_slots = {}
    if _declares_nest_lifecycle(model):
        root_header = read_restart_header(paths[int(model.root.cfg.grid_id)])
        stored_lifecycle_slots = dict(
            (root_header.get(NEST_LIFECYCLE_HEADER_KEY) or {}).get("window_slots", {}))
    # Load every payload and hoist restore_restart's config/setup/manifest/
    # inventory/boundary refusal checks across the COMPLETE member set before
    # touching any live domain.  The validated objects retain the exact host
    # arrays that will be applied, so a later member cannot be swapped or
    # become unreadable after an earlier domain has mutated.
    if sealed_forcing_extension:
        validated = {
            gid: _validate_restart(
                paths[gid], node.state, node.cfg.run,
                sealed_forcing_extension=True)
            for gid, node in nodes.items()
        }
    else:
        # Preserve the historical default call shape as well as its exact
        # semantics.  A few out-of-tree diagnostic wrappers substitute this
        # private validator with the original three-argument signature.
        validated = {}
        for gid, node in nodes.items():
            streamed = getattr(node.state, "_streamed_domain", None)
            if streamed is not None:
                options = ({"extra_scratch_slots": tuple(stored_lifecycle_slots.get(str(gid), ()))}
                           if _declares_nest_lifecycle(model) else {})
                validated[gid] = streamed.validate_restart(
                    paths[gid], node.cfg.run, **options)
            else:
                validated[gid] = _validate_restart(paths[gid], node.state, node.cfg.run)
    headers = {gid: member.header for gid, member in validated.items()}
    _require_held_lifecycle_reflectivity(
        model, nodes, headers, validated, stored_lifecycle_slots)
    # The root audit names the held signals this generation wrote. Verify
    # those members across the complete set before applying any domain.
    for header in headers.values():
        lifecycle = header.get(NEST_LIFECYCLE_HEADER_KEY)
        if lifecycle is None:
            continue
        for raw_gid, slots in lifecycle["window_slots"].items():
            gid = int(raw_gid)
            for slot in slots:
                key = f"scratch/{slot}"
                if gid not in validated or key not in validated[gid].stored:
                    raise RestartMismatchError(
                        f"lifecycle checkpoint declares d{gid:02d} {key} "
                        "but its member is missing; no domain was restored")

    # A checkpoint set written after a nest relocation carries the ruled
    # posture on itself.  A restore that reads one is BY DEFINITION a
    # restart across a move: say so loudly BEFORE any refusal (so even
    # the failure path states the posture), and stash it on the model so
    # a run that does proceed states it in its own receipts.
    crossed = {gid: header["relocation"]
               for gid, header in headers.items()
               if isinstance(header.get("relocation"), Mapping)}
    if crossed:
        gid = sorted(crossed)[0]
        moved = crossed[gid]
        from woof.explain import warn

        warn(f"this restore crosses {moved.get('moves')} nest "
             f"relocation(s) (segment {moved.get('segment_id')!r}).  "
             f"{moved.get('posture', '')}")
        model._restart_crossed_relocation = dict(moved)

    elapsed_pairs = set()
    checkpoint_set_ids = set()
    for gid, node in nodes.items():
        header = headers[gid]
        require_readable_format_version(header.get("format_version"), paths[gid])
        if header.get("format_version") != RESTART_FORMAT_VERSION:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} must be "
                f"v{RESTART_FORMAT_VERSION}, got "
                f"{header.get('format_version')!r}; v2 is single-domain only")
        if sealed_forcing_extension and header.get(
                "forcing_extension_mode") != SEALED_FORCING_EXTENSION_MODE:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} was not intentionally sealed for "
                "forcing extension")
        if (header.get("experiment_fingerprint")
                != model.experiment_fingerprint
                and not _identity_matches_under_current_rules(header, model)):
            raise RestartMismatchError(
                tree_fingerprint_mismatch_reason(gid, header, model))
        checkpoint_set_id = header.get("checkpoint_set_id")
        if (not isinstance(checkpoint_set_id, str)
                or not checkpoint_set_id):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has no checkpoint_set_id")
        checkpoint_set_ids.add(checkpoint_set_id)
        if header.get("grid_id") != gid or header.get("parent_id") != \
                int(node.cfg.parent_id):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} grid_id/parent_id mismatch")
        if header.get("domain_ids") != sorted(nodes):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} domain set header mismatch")
        if header.get("phase") != PERIOD_BEGIN:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} phase must explicitly be "
                f"{PERIOD_BEGIN}, got {header.get('phase')!r}")
        expected_start_time = (
            model.schedule.clock.start_time
            if node.cfg.start_time is None else node.cfg.start_time)
        stored_start_time = header.get("domain_start_time")
        stored_start_ticks = header.get("domain_start_ticks")
        lifecycle = header.get("domain_lifecycle")
        # Current-format checkpoints written before delayed starts existed
        # can only represent the all-live-at-t0 lifecycle.  Admit that one
        # unambiguous migration; a delayed live config still requires every
        # new field and therefore remains fail-closed.
        if node.clock.spec.start_ticks == 0:
            if stored_start_time is None:
                stored_start_time = expected_start_time.isoformat()
            if stored_start_ticks is None:
                stored_start_ticks = 0
            if lifecycle is None:
                lifecycle = "STARTED"
        if stored_start_time != expected_start_time.isoformat():
            raise RestartMismatchError(
                f"tree restart d{gid:02d} domain_start_time mismatch")
        if stored_start_ticks != node.clock.spec.start_ticks:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} domain_start_ticks mismatch")
        if lifecycle not in ("STARTED", "NOT_STARTED"):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has invalid domain_lifecycle "
                f"{lifecycle!r}")
        ticks = header.get("elapsed_ticks")
        den = header.get("tick_den")
        if (isinstance(ticks, bool) or not isinstance(ticks, int)
                or isinstance(den, bool) or not isinstance(den, int)
                or ticks < 0 or den <= 0):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has invalid exact tick pair "
                f"{(ticks, den)!r}")
        expected_lifecycle = (
            "STARTED" if ticks >= node.clock.spec.start_ticks
            else "NOT_STARTED")
        if lifecycle != expected_lifecycle:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} lifecycle {lifecycle} disagrees "
                f"with elapsed/start ticks ({ticks}, "
                f"{node.clock.spec.start_ticks})")
        expected_tables = (
            "REBUILT" if lifecycle == "STARTED" or node.parent is None
            else "NOT_STARTED")
        if header.get("nest_tables") != expected_tables:
            raise RestartMismatchError(
                f"tree restart d{gid:02d} must classify nest tables "
                f"{expected_tables}")
        stored_seconds = header.get("elapsed_seconds")
        if (isinstance(stored_seconds, bool)
                or not isinstance(stored_seconds, (int, float))
                or float(stored_seconds) != ticks / den):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} elapsed_seconds disagrees with "
                "its exact tick pair")
        bits = header.get("dtbc_fp32_bits")
        if (isinstance(bits, bool) or not isinstance(bits, int)
                or not 0 <= bits <= 0xFFFFFFFF):
            raise RestartMismatchError(
                f"tree restart d{gid:02d} has invalid dtbc_fp32_bits")
        elapsed_pairs.add((ticks, den))
    if len(elapsed_pairs) != 1:
        raise RestartMismatchError(
            f"tree restart domain elapsed ticks mismatch: "
            f"{sorted(elapsed_pairs)}")
    if len(checkpoint_set_ids) != 1:
        raise RestartMismatchError(
            "tree restart files come from mismatched checkpoint sets")
    ticks, tick_den = elapsed_pairs.pop()
    if tick_den != model.schedule.clock.tick_den:
        raise RestartMismatchError(
            f"tree restart tick_den {tick_den} != live {model.schedule.clock.tick_den}")
    if ticks % model.schedule.period_ticks != 0:
        raise RestartMismatchError(
            f"tree restart tick {ticks} is not PERIOD_BEGIN")
    run_ticks = model.schedule.clock.run_ticks
    # A restore point BEYOND the stop is a real mismatch: the checkpoint
    # comes from a longer run than the one being resumed, and there is no
    # way to integrate backwards to the requested stop.  A restore point
    # exactly AT the stop is not -- it is the state a finished run ended
    # on, so the route finalizes and reports success instead of raising.
    # The favor-class 24 h reproduction died here: the supervisor
    # relaunched from the run's own final checkpoint and every fresh
    # worker refused at ticks == run_ticks, turning a complete run into
    # rc 1.
    already_complete = ticks == run_ticks
    if ticks > run_ticks:
        raise RestartMismatchError(
            f"tree restart is at {ticks} ticks ({ticks / tick_den:g} s of "
            f"model time) but this configuration stops at {run_ticks} ticks "
            f"({run_ticks / tick_den:g} s), so the checkpoint comes from a "
            "LONGER run than the one being resumed and there is no state to "
            "integrate backwards to.  Resume from a checkpoint at or before "
            "the stop, or raise [experiment] run_seconds to at least "
            f"{ticks / tick_den:g}")

    # The set's own topology, for the ring migration below: a nest that
    # rides inside a mover changes ground without ever being listed in
    # ``moved_grid_ids`` (that list names the grids a resume must put
    # back through a placement rebuild, which a carried nest does not
    # need), and its ring holds rain the move shifted there.
    parent_by_grid = {int(gid): int(header.get("parent_id") or 0)
                      for gid, header in headers.items()}
    # All refusal checks over all members precede the first mutation.
    for gid, node in nodes.items():
        from tilestream.restart_stream import ValidatedStreamedRestart
        if isinstance(validated[gid], ValidatedStreamedRestart):
            node.state._streamed_domain.apply_restart(validated[gid])
        else:
            _apply_validated_restart(validated[gid], node.state, node.cfg.run,
                                     parent_by_grid=parent_by_grid)
        clock = node.clock
        clock.ticks = ticks
        adaptive = headers[gid].get("adaptive_clock")
        if adaptive is None:
            clock.step_count = max(
                0, (ticks - clock.spec.start_ticks) // clock.spec.step_ticks)
        else:
            # A varying step makes that division meaningless -- it answers
            # "how many CONFIGURED steps fit in the elapsed time", which is
            # not how many were taken.  The writer stored the real count.
            clock.step_count = int(adaptive["step_count"])
            clock.step_ticks = int(adaptive["step_ticks"])
            clock.dt_fp32 = np.uint32(
                adaptive["dt_fp32_bits"]).view(np.float32)
            clock.adaptive_state = adaptive.get("controller")
        bits = headers[gid]["dtbc_fp32_bits"]
        clock.dtbc_fp32 = np.asarray(bits, dtype=np.uint32).view(np.float32)
        node.state.elapsed_seconds = ticks / tick_den
        stored_lifecycle = headers[gid].get("domain_lifecycle")
        node._started = (
            True if stored_lifecycle is None
            and node.clock.spec.start_ticks == 0
            else stored_lifecycle == "STARTED")
        if node.parent is not None:
            node.coupler.invalidate()
            reset_observation = getattr(
                node.coupler, "reset_restart_observation", None)
            if reset_observation is not None:
                reset_observation()

    model._runtime_status = ModelRuntimeStatus()
    model._resumed = True
    model._resume_committed_history_grid_ids = frozenset(
        gid for gid, node in nodes.items() if node.clock.history_due())
    root_id = int(model.root.cfg.grid_id)
    model._last_checkpoint = paths[root_id]
    # A checkpoint read beside the leg's preparation (restart_prefetch.prefetch_beside) is applied:
    # its host arrays are not held through the integration.
    from woof.io.restart_prefetch import release_beside
    release_beside(paths.values())
    return TreeRestartInfo(
        elapsed_ticks=ticks, tick_den=tick_den, phase=PERIOD_BEGIN,
        paths_by_grid_id=paths, headers_by_grid_id=headers,
        already_complete=already_complete)


def _validate_scratch_target(state, slot: str, host: np.ndarray,
                             key: str) -> None:
    """Prove a scratch copy can be applied without creating a live slot."""
    from woof.core.nest_accumulation import META
    if slot == META:
        if host.shape != (4,) or host.dtype != np.dtype(np.int64):
            raise RestartMismatchError(
                f"{key}: nest archive origin must contain four int64 coordinates")
        return
    if slot == "refl_10cm":
        # This carried diagnostic has the native mass-grid volume layout.
        # Check it even when the restored state has not allocated it yet.
        _check_array(host, state.p, key)
    target = getattr(state, "_scratch", {}).get(slot)
    if target is not None:
        _check_array(host, target, key)
        return
    arena = getattr(state, "_scratch_arena", None)
    if arena is not None and arena.has_slot(slot):
        try:
            target = arena.view(host.shape, slot, host.dtype)
        except (KeyError, TypeError, ValueError) as exc:
            raise RestartMismatchError(
                f"{key}: restart payload does not fit the live scratch arena"
            ) from exc
        _check_array(host, target, key)
        return
    # A non-arena missing slot will be allocated with the model field dtype.
    from woof.lightning_contract import INTEGER_SCRATCH_SLOTS
    dtype = np.dtype(np.int32 if slot in INTEGER_SCRATCH_SLOTS else getattr(state.u, "dtype", np.float32))
    if host.dtype != dtype:
        raise RestartMismatchError(
            f"{key}: restart dtype {host.dtype} does not match state "
            f"scratch dtype {dtype}")


def _validate_strict_radiation_total(stored, driver) -> None:
    """A rounded held total cannot be rebuilt from its published components."""
    target = getattr(driver, "_wrf_rthraten", None)
    if target is None:
        return
    key = "diag/_wrf_rthraten"
    if key not in stored:
        raise RestartMismatchError(
            "restart is missing diag/_wrf_rthraten; strict radiation and "
            "cumulus forcing would resume with zero held heating. The "
            "rounded total cannot be reconstructed from LW and SW. Use "
            "a checkpoint written by this build or start from prepared state.")
    _check_array(stored[key], target, key)


def _validate_driver_payload(stored, header, state, driver, elapsed,
                             format_version: int) -> None:
    """Hoist every PhysicsDriver refusal without mutating the driver."""
    _validate_strict_radiation_total(stored, driver)
    _validate_fire_payload(stored, header, driver)
    expected_optional = {f"driver/{name}/{component}" for name in OPTIONAL_DRIVER_TENDENCY_ATTRS
                         if getattr(driver, name, None) is not None for component in TENDENCY_COMPONENTS
                         if getattr(getattr(driver, name), component) is not None}
    stored_optional = {key for key in stored if any(key.startswith(f"driver/{name}/")
                       for name in OPTIONAL_DRIVER_TENDENCY_ATTRS)}
    if expected_optional != stored_optional:
        raise RestartMismatchError("restart held fire-tendency inventory differs; the next atmospheric step would lose coupled heating or moisture")
    expected_held = {f"held/{name}" for name in DRIVER_HELD_FORCING_ATTRS
                     if getattr(driver, name, None) is not None}
    stored_held = {key for key in stored if key.startswith("held/")}
    if stored_held != expected_held:
        raise RestartMismatchError(
            "restart held PBL forcing inventory does not match the resuming "
            f"driver (missing {sorted(expected_held - stored_held)}, "
            f"extra {sorted(stored_held - expected_held)}). "
            "GF/New Tiedtke read these raw rates between PBL calls; they "
            "cannot be reconstructed from coupled tendencies. Resume from "
            "a checkpoint written by this build or start from prepared state.")
    for key in sorted(expected_held):
        _check_array(stored[key], getattr(driver, key[5:]), key)

    raw_pbl = pbl_raw_manifest(driver)
    expected_pbl = {key for key in raw_pbl if key.startswith("pbl/")}
    stored_pbl = {key for key in stored if key.startswith("pbl/")
                  and not key.startswith("pbl/diagnostics/")}
    _validate_pbl_diagnostics(stored, header, state, driver)
    if stored_pbl != expected_pbl:
        raise RestartMismatchError(
            "restart raw PBL inventory does not match this configuration "
            f"(missing {sorted(expected_pbl - stored_pbl)}, "
            f"extra {sorted(stored_pbl - expected_pbl)}); use a checkpoint "
            "from this build or start from prepared state")
    for key, target in raw_pbl.items():
        _check_array(stored[key], target, key)

    stored_fields = {key[len("fields/"):]: value
                     for key, value in stored.items()
                     if key.startswith("fields/")}
    # A plane a resume's output-only diagnostic switch moved is dropped or
    # initialised by _restore_driver; every other plane must match exactly.
    gated = _diagnostic_gated_field_changes(stored_fields, header, driver)
    compared = set(stored_fields) - set(gated)
    expected = set(driver.fields) - set(gated)
    if compared != expected:
        raise RestartMismatchError(
            "restart surface-field inventory does not match the resuming "
            f"driver (missing {sorted(expected - compared)}, "
            f"extra {sorted(compared - expected)})")
    for name in sorted(compared):
        _check_array(stored_fields[name], driver.fields[name], f"fields/{name}")

    for name, target in (("rthratenlw", driver.rthratenlw),
                         ("rthratensw", driver.rthratensw),
                         ("pending_rainbl", driver._pending_rainbl)):
        key = f"driver/{name}"
        if key not in stored:
            raise RestartMismatchError(f"restart is missing {key}")
        _check_array(stored[key], target, key)

    for tend_name in _live_driver_tendency_attrs(driver):
        missing = [comp for comp in TENDENCY_REQUIRED_COMPONENTS
                   if f"driver/{tend_name}/{comp}" not in stored]
        if missing:
            raise RestartMismatchError(
                f"restart tendency {tend_name} is missing {missing}")
        for comp in TENDENCY_COMPONENTS:
            key = f"driver/{tend_name}/{comp}"
            host = stored.get(key)
            if format_version == RESTART_FORMAT_VERSION:
                live = getattr(getattr(driver, tend_name), comp)
                # Older SASE checkpoints could omit w only under the
                # then-mandatory every-step cadence. That next due call
                # rebuilds it before use; a skipped-call run needs it.
                rebuilt_legacy_w = (comp == "rw" and host is None
                    and tend_name == "pbl_tendencies"
                    and header["config"].get("bldt") == 0.0)
                # Historical SASE also published a zero-ice placeholder
                # as rqi even when no state ice existed. It has no
                # consumer; the every-step rebuild removes it.
                legacy_absent_ice = (comp == "rqi" and live is None
                    and tend_name == "pbl_tendencies"
                    and getattr(driver, "sase_active", False)
                    and getattr(state, "qi", None) is None
                    and header["config"].get("bldt") == 0.0)
                if ((host is not None) != (live is not None)
                        and not rebuilt_legacy_w and not legacy_absent_ice):
                    disposition = ("missing" if host is None
                                   else "unexpected")
                    raise RestartMismatchError(
                        f"restart tendency inventory has {disposition} "
                        f"canonical member {key}")
            if host is not None:
                target = (state.u if comp == "ru" else
                          state.v if comp == "rv" else
                          state.w if comp == "rw" else state.p)
                _check_array(host, target, key)

    from woof.core.physics import microphysics_scratch_slots
    slot_map = dict(microphysics_scratch_slots(driver.mp_physics))
    if not slot_map and format_version == 2:
        missing = [comp for comp in MICROPHYSICS_REQUIRED_COMPONENTS
                   if f"driver/microphysics/{comp}" not in stored]
        if missing:
            raise RestartMismatchError(
                f"v2 restart microphysics diagnostics are missing {missing}")
    for comp, slot in slot_map.items():
        scratch_key = f"scratch/{slot}"
        legacy_key = f"driver/microphysics/{comp}"
        scratch_host = stored.get(scratch_key)
        legacy_host = stored.get(legacy_key)
        if format_version == RESTART_FORMAT_VERSION:
            if legacy_host is not None:
                raise RestartMismatchError(
                    f"v{RESTART_FORMAT_VERSION} restart unexpectedly "
                    f"contains removed member {legacy_key}")
            if scratch_host is None:
                raise RestartMismatchError(
                    f"v{RESTART_FORMAT_VERSION} restart is missing "
                    f"canonical microphysics member {scratch_key}")
        elif scratch_host is not None and legacy_host is not None:
            _validate_scratch_target(
                state, slot, legacy_host, legacy_key)
            if scratch_host.tobytes() != legacy_host.tobytes():
                raise RestartMismatchError(
                    f"v2 restart's duplicate microphysics members "
                    f"{scratch_key} and {legacy_key} differ byte-for-byte; "
                    "they cannot be rebuilt as one alias without changing "
                    "the trajectory")
        source = scratch_host if scratch_host is not None else legacy_host
        if source is None:
            if comp in MICROPHYSICS_REQUIRED_COMPONENTS:
                raise RestartMismatchError(
                    f"v2 restart microphysics diagnostics are missing "
                    f"both {scratch_key} and {legacy_key}")
            empty = np.empty(state.mup.shape, dtype=state.u.dtype)
            _validate_scratch_target(state, slot, empty, scratch_key)
        else:
            _validate_scratch_target(state, slot, source, scratch_key)

    driver_header = header.get("driver")
    if not isinstance(driver_header, dict):
        raise RestartMismatchError(
            "restart has no physics-driver header but the resuming state "
            "has a PhysicsDriver attached")
    try:
        call_counts = driver_header["call_counts"]
        int(driver_header["ysu_nan_guard_fires"])
        int(driver_header["microphysics_updates"])
        if not isinstance(call_counts, dict):
            raise TypeError("call_counts is not a mapping")
        for value in call_counts.values():
            int(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise RestartMismatchError(
            "restart physics-driver header is incomplete or malformed") from exc

    if "cumulus/w0avg" in stored:
        adapter = driver.cumulus_callable
        if adapter is None or not hasattr(adapter, "w0avg"):
            raise RestartMismatchError(
                "restart carries cumulus W0AVG but the resuming driver has "
                "no trigger-history cumulus adapter")
        _check_array(stored["cumulus/w0avg"], state.p, "cumulus/w0avg")
    elif getattr(driver.cumulus_callable, "w0avg", None) is not None:
        raise RestartMismatchError(
            "restart carries no cumulus W0AVG history but the resuming "
            "driver already has live W0AVG state; prepare a fresh adapter")
    if "radiation/o33d_grid" in stored:
        if (getattr(driver, "o3rad", None) is None
                and not hasattr(driver.radiation_callable, "_o33d_grid")):
            raise RestartMismatchError(
                "restart carries the legacy-RRTMG o33d field but the "
                "resuming radiation callable has no _o33d_grid slot "
                "(variant mismatch should have refused earlier)")
        _check_array(stored["radiation/o33d_grid"], state.p,
                     "radiation/o33d_grid")
    elif (getattr(driver, "cam_ozone", None) is not None
          and int(call_counts.get("cam_ozone", call_counts.get("radiation", 0))) > 0):
        raise RestartMismatchError(
            "restart omits retained CAM ozone after its producer ran; "
            "the next child radiation call requires that held field")


def _rebuilt_frame_note(schema) -> str:
    """What a frame refusal adds for a document in the rebuilt identity."""

    if schema not in REBUILT_END_FRAME_PREFIX_SCHEMAS:
        return ""
    return (f" as the rebuilt end-frame identity ({schema}) sees it: its "
            "end frame is FP32(value + tendency * duration), which a "
            "boundary value clearing out also moves.  That identity is a "
            "checkpoint's written before 2.8.1, and a forcing series' "
            "whose builder recorded no end frame (a wrfbdy file, a "
            "prepared cache written before 2.8.1); a checkpoint written by "
            "2.8.1 or later on forcing whose builder records its end "
            "frames resumes across such a frame")


def _validated_forcing_prefix(value, *, label: str, path: Path):
    """Validate and normalize one append-only LBC identity document.

    Each interval's end frame must be the next interval's start frame: the
    refusal of a splice, a series whose interval k+1 does not start where
    interval k's tendency was built to go.  In the built end-frame identity
    (:data:`BUILT_END_FRAME_PREFIX_SCHEMA`) a row's end frame is that frame
    as its builder recorded it, so one preparation's series passes exactly,
    a hydrometeor table that clears out between two forcing times included,
    and the check applies to every checkpoint, bound to a prepared head or
    not (A140b).  A document in the rebuilt identity
    (:data:`REBUILT_END_FRAME_PREFIX_SCHEMAS`, which a checkpoint written
    before 2.8.1 carries) is checked the way its writer checked it, and its
    refusal says so, because a clear-out also breaks a rebuilt frame.
    """
    if not isinstance(value, dict):
        raise RestartMismatchError(
            f"restart file {path} has no {label} forcing inventory")
    expected_keys = {
        "schema", "spec_bdy_width", "spec_zone", "relax_zone", "intervals",
    }
    if set(value) != expected_keys:
        missing = sorted(expected_keys - set(value))
        extra = sorted(set(value) - expected_keys)
        raise RestartMismatchError(
            f"restart file {path} has malformed {label} forcing document "
            f"(missing {missing}, extra {extra})")
    if value.get("schema") not in LATERAL_BOUNDARY_PREFIX_SCHEMAS:
        raise RestartMismatchError(
            f"restart file {path} has unknown {label} forcing schema "
            f"{value.get('schema')!r}")
    controls = []
    for key in ("spec_bdy_width", "spec_zone", "relax_zone"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RestartMismatchError(
                f"restart file {path} has invalid {label} forcing "
                f"control {key}={item!r}")
        controls.append(item)
    raw_intervals = value.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise RestartMismatchError(
            f"restart file {path} has an empty/malformed {label} forcing "
            "inventory")
    intervals = []
    field_inventory = None
    for index, row in enumerate(raw_intervals):
        if not isinstance(row, dict) or set(row) != {
                "start_seconds", "end_seconds", "fields", "sha256",
                "start_frame_sha256", "end_frame_sha256"}:
            raise RestartMismatchError(
                f"restart file {path} has malformed {label} forcing "
                f"interval {index}")
        start = row["start_seconds"]
        end = row["end_seconds"]
        if (isinstance(start, bool) or not isinstance(start, (int, float))
                or isinstance(end, bool) or not isinstance(end, (int, float))
                or not math.isfinite(float(start))
                or not math.isfinite(float(end))
                or float(start) < 0.0 or float(end) <= float(start)):
            raise RestartMismatchError(
                f"restart file {path} has invalid {label} forcing bounds "
                f"at interval {index}: {(start, end)!r}")
        if intervals and float(start) != float(intervals[-1]["end_seconds"]):
            raise RestartMismatchError(
                f"restart file {path} has non-contiguous {label} forcing "
                f"inventory at interval {index}")
        fields = row["fields"]
        if (not isinstance(fields, list) or not fields
                or any(not isinstance(name, str) or not name for name in fields)
                or fields != sorted(set(fields))):
            raise RestartMismatchError(
                f"restart file {path} has malformed {label} forcing field "
                f"inventory at interval {index}")
        if field_inventory is None:
            field_inventory = fields
        elif fields != field_inventory:
            raise RestartMismatchError(
                f"restart file {path} changes {label} forcing fields at "
                f"interval {index}")
        for digest_key in (
                "sha256", "start_frame_sha256", "end_frame_sha256"):
            sha = row[digest_key]
            if (not isinstance(sha, str) or len(sha) != 64
                    or any(char not in "0123456789abcdef" for char in sha)):
                raise RestartMismatchError(
                    f"restart file {path} has invalid {label} forcing "
                    f"{digest_key} at interval {index}")
        if intervals and \
                intervals[-1]["end_frame_sha256"] != row["start_frame_sha256"]:
            raise RestartMismatchError(
                f"restart file {path} has a discontinuous {label} forcing "
                f"frame at interval {index}"
                + _rebuilt_frame_note(value.get("schema")))
        intervals.append({
            "start_seconds": start,
            "end_seconds": end,
            "fields": fields,
            "sha256": row["sha256"],
            "start_frame_sha256": row["start_frame_sha256"],
            "end_frame_sha256": row["end_frame_sha256"],
        })
    if float(intervals[0]["start_seconds"]) != 0.0:
        raise RestartMismatchError(
            f"restart file {path} {label} forcing does not begin at zero")
    return tuple(controls), intervals


def _require_sealable_forcing_prefix(state, cfg, *, path: Path,
                                     elapsed: float) -> None:
    """Refuse an invalid sealed checkpoint before any bytes are published."""
    prefix = lateral_boundary_prefix_identity(state)
    if getattr(cfg, "specified", False):
        _, intervals = _validated_forcing_prefix(
            prefix, label="live", path=path)
        if float(intervals[-1]["end_seconds"]) != elapsed:
            raise RestartMismatchError(
                f"restart file {path} forcing inventory is not sealed at "
                f"its checkpoint boundary "
                f"({intervals[-1]['end_seconds']!r} != {elapsed!r})")
        return
    if getattr(cfg, "nested", False):
        if prefix is not None:
            raise RestartMismatchError(
                f"restart file {path} nested child unexpectedly carries "
                "an external forcing prefix")
        return
    raise RestartMismatchError(
        f"restart file {path} cannot use sealed forcing extension: the "
        "domain is neither a specified external-boundary root nor a nested "
        "child")


def _live_forcing_prefix(state, stored, *, path: Path):
    """The live forcing document in the identity ``stored`` was written in.

    ``stored`` is a validated checkpoint document.  One written before
    2.8.1 is in the rebuilt end-frame identity, and the live series is
    hashed the same way, so the checkpoint reads and every row is compared
    like with like.  One in the built identity needs a live series whose
    builder recorded every end frame; a series that records none (a
    prepared cache written before 2.8.1, a wrfbdy file) is refused by name
    rather than compared under two meanings of the same digest.
    """

    schema = stored.get("schema") if isinstance(stored, dict) else None
    if schema in REBUILT_END_FRAME_PREFIX_SCHEMAS:
        return lateral_boundary_prefix_identity(
            state, rebuilt_end_frames=True)
    live = lateral_boundary_prefix_identity(state)
    if (schema == BUILT_END_FRAME_PREFIX_SCHEMA and isinstance(live, dict)
            and live.get("schema") != schema):
        raise RestartMismatchError(
            f"restart file {path} records each forcing interval's end frame "
            f"as its builder recorded it ({schema}), and the forcing it is "
            "resumed on records none (a prepared cache or boundary series "
            "made before 2.8.1, or a wrfbdy file), so their frames cannot "
            "be compared; resume on the preparation the checkpoint was "
            "written against, or prepare that forcing again")
    return live


def _require_sealed_forcing_extension(header, state, *, path: Path,
                                      elapsed: float) -> None:
    """Admit only a byte-identical sealed prefix plus contiguous future LBCs.

    The join is the shared checkpoint-boundary frame: the stored last row's
    end frame, the frame its tendency was built toward, against the first
    appended interval's start frame (A140b).
    """
    if header.get("forcing_extension_mode") != SEALED_FORCING_EXTENSION_MODE:
        raise RestartMismatchError(
            f"restart file {path} was not intentionally sealed for forcing "
            "extension")
    stored_core = header.get("setup_core_fingerprint")
    live_core = setup_core_fingerprint(state)
    if stored_core != live_core:
        raise RestartMismatchError(
            f"restart file {path} immutable setup changed while extending "
            "forcing (base state / coordinates / map factors mismatch)")
    stored_value = header.get("lateral_boundary_prefix")
    stored_controls, stored = _validated_forcing_prefix(
        stored_value, label="sealed", path=path)
    live_value = _live_forcing_prefix(state, stored_value, path=path)
    live_controls, live = _validated_forcing_prefix(
        live_value, label="live", path=path)
    if stored_controls != live_controls:
        raise RestartMismatchError(
            f"restart file {path} changes specified-boundary controls while "
            "extending forcing")
    if float(stored[-1]["end_seconds"]) != elapsed:
        raise RestartMismatchError(
            f"restart file {path} forcing inventory was not sealed at its "
            f"checkpoint boundary ({stored[-1]['end_seconds']!r} != "
            f"{elapsed!r})")
    if len(live) <= len(stored):
        raise RestartMismatchError(
            f"restart file {path} forcing extension must append at least one "
            "future interval")
    if live[:len(stored)] != stored:
        raise RestartMismatchError(
            f"restart file {path} forcing extension changed a sealed interval "
            "or its byte digest")
    suffix = live[len(stored):]
    if float(suffix[0]["start_seconds"]) != elapsed:
        raise RestartMismatchError(
            f"restart file {path} forcing suffix does not begin at the "
            "checkpoint boundary")
    if any(float(row["start_seconds"]) < elapsed for row in suffix):
        raise RestartMismatchError(
            f"restart file {path} forcing suffix is not strictly future")
    if stored[-1]["end_frame_sha256"] != suffix[0][
            "start_frame_sha256"]:
        raise RestartMismatchError(
            f"restart file {path} forcing suffix changes the shared "
            "checkpoint-boundary frame"
            + _rebuilt_frame_note(stored_value.get("schema")))


#: Header key naming the prepared head a pre-seal checkpoint was written
#: against (chained preparation, :mod:`woof.ingest.boundary_stream`).
BOUNDARY_STREAM_HEADER_KEY = "boundary_stream"


def _pre_seal_intervals(state):
    """The root's streamed boundary series while its preparation is unsealed.

    ``None`` for every eager series and for a streamed one whose tree has
    sealed: those checkpoints are written exactly as always.
    """

    boundaries = getattr(state, "lateral_boundaries", None)
    intervals = getattr(boundaries, "intervals", None)
    ready = getattr(intervals, "ready_prefix", None)
    sealed = getattr(intervals, "sealed", None)
    if not callable(ready) or not callable(sealed) or sealed():
        return None
    return intervals


class _ReadyPrefixState:
    """A state seen with only its prepared boundary intervals.

    A checkpoint written while the preparation is still producing later
    intervals is hashed over the intervals that exist (which always cover
    the checkpoint clock) under the preserved-prefix contract.  Waiting
    for the seal instead would stall the model at its first checkpoint
    for the rest of the preparation.
    """

    def __init__(self, state, intervals, count=None):
        count = int(intervals.ready_prefix() if count is None else count)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "lateral_boundaries", dataclasses.replace(
            state.lateral_boundaries,
            intervals=tuple(intervals[index] for index in range(count))))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_state"), name)


def _require_preservable_forcing_prefix(state, cfg, *, path, elapsed,
                                        prefix=None):
    """Validate the live series a preserved-prefix checkpoint records.

    ``prefix`` is that series' document when the caller already made it in
    a stored checkpoint's identity (:func:`_live_forcing_prefix`); a writer
    passes nothing and records the series in its own identity.
    """
    if not getattr(cfg, 'specified', False) or getattr(cfg, 'nested', False):
        raise RestartMismatchError(f'{path}: preserved forcing requires a specified root domain')
    controls, intervals = _validated_forcing_prefix(
        lateral_boundary_prefix_identity(state) if prefix is None else prefix,
        label='preserved', path=path)
    if elapsed > float(intervals[-1]['end_seconds']):
        raise RestartMismatchError(f'{path}: the forcing inventory ends before the checkpoint clock')
    return controls, intervals


def _stored_prefix_view(header, state, *, path):
    """The live state seen with only the intervals the checkpoint recorded.

    A checkpoint written before its preparation sealed records the
    intervals prepared at that time.  Resumed on a series still being
    prepared (the same head), the preserved-prefix check needs exactly
    those intervals, which already exist; comparing the whole series would
    wait for the seal before the first step, so a resume mid-preparation
    would lose the head start the chain gives the first run.  Every other
    live series is returned whole, as before.
    """

    intervals = _pre_seal_intervals(state)
    if intervals is None:
        return state
    _, stored = _validated_forcing_prefix(
        header.get('lateral_boundary_prefix'), label='stored preserved',
        path=path)
    if len(stored) > len(intervals):
        return state
    return _ReadyPrefixState(state, intervals, count=len(stored))


def _require_preserved_forcing_prefix(header, state, cfg, *, path, elapsed):
    if header.get('forcing_extension_mode') != PRESERVED_FORCING_PREFIX_MODE:
        raise RestartMismatchError(f'{path}: this checkpoint has no preserved forcing-prefix contract; restore its exact preparation')
    if header.get('setup_core_fingerprint') != setup_core_fingerprint(state):
        raise RestartMismatchError(f'{path}: forcing renewal changes immutable base state, coordinates or map factors')
    stored_value = header.get('lateral_boundary_prefix')
    old_controls, old = _validated_forcing_prefix(
        stored_value, label='stored preserved', path=path)
    # A renewal that appends past the stored prefix joins its last row's
    # end frame, the frame that row's tendency was built toward, to the
    # next interval's start frame, which the live document's continuity
    # check proves (A140b), a boundary value that clears out at the join
    # included.
    new_controls, new = _require_preservable_forcing_prefix(
        state, cfg, path=path, elapsed=elapsed,
        prefix=_live_forcing_prefix(state, stored_value, path=path))
    if elapsed > float(old[-1]['end_seconds']):
        raise RestartMismatchError(f'{path}: the stored forcing ends before its checkpoint clock')
    if old_controls != new_controls or new[:len(old)] != old:
        raise RestartMismatchError(f'{path}: forcing renewal changed a previously declared interval or boundary controls; retain the complete original prefix')


def _validate_ensemble_stochastic_checkpoint(header, stored, state):
    """Validate member identity and all spectra before any model-array write."""
    binding = getattr(state, "_ensemble_stochastic", None)
    enabled = binding is not None and binding.enabled
    metadata = header.get("ensemble_stochastic")
    arrays = {key: value for key, value in stored.items() if key.startswith("stochastic/")}
    if not enabled and metadata is None and not arrays:
        return
    if not enabled or metadata is None:
        raise RestartMismatchError("stochastic checkpoint enabling differs from this member")
    from woof.ensemble.stochastic_execution import decode_checkpoint_payload
    try:
        snapshot = decode_checkpoint_payload(metadata, arrays)
        binding.validate_identity(snapshot)
        binding.hook.validate_snapshot(snapshot["hook"])
    except (ValueError, TypeError, KeyError) as error:
        raise RestartMismatchError(f"stochastic checkpoint is incompatible: {error}") from error


def _validate_restart(path, state, cfg, *,
                      sealed_forcing_extension: bool = False,
                      preserved_forcing_prefix: bool = False
                      ) -> _ValidatedRestart:
    """Load one archive and perform all refusal checks without mutation."""
    path = Path(path)
    header, stored = _load_restart(path, with_arrays=True)
    if cfg.diff_opt == 1 and "scratch/diff1_theta_initial" not in stored:
        raise RestartMismatchError(
            "diff_opt=1 restart is missing its original thermal reference "
            "scratch/diff1_theta_initial; reconstructing it from evolved "
            "theta would change coordinate diffusion")
    required_header = {
        "format_version", "config", "setup_fingerprint",
        "physics_setup", "physics_setup_fingerprint",
        "array_manifest", "elapsed_seconds",
    }
    missing_header = sorted(required_header - set(header))
    if missing_header:
        raise RestartMismatchError(
            f"restart file {path} header is missing {missing_header}")
    format_version = header.get("format_version")
    require_readable_format_version(format_version, path)
    # BEFORE the configuration walk, so the crossed 4/4 radiation
    # resume is answered by the gate that says what breaks rather
    # than by a field name in a list of differences (audit R-048).
    _require_rrtmg_variant_match(header, cfg, path)
    # BEFORE the configuration walk too: a checkpoint of a scheme this
    # build computes differently resumes under no configuration, so the
    # walk's notes (the terrain clock's among them) must not be what the
    # user is sent to change first.
    _require_current_algorithm_identities(header, cfg, path)
    _require_config_match(header["config"], cfg, path)
    live_lbc_clock = root_external_lbc_clock_identity(state, cfg)
    if live_lbc_clock is not None:
        stored_lbc_clock = header.get("root_external_lbc_clock",
                                      ROOT_EXTERNAL_LBC_CLOCK_LEGACY)
        if stored_lbc_clock != live_lbc_clock:
            raise RestartMismatchError(
                f"restart file {path} was integrated under the "
                f"root_external_lbc_clock semantic {stored_lbc_clock!r} "
                f"but the resuming state runs {live_lbc_clock!r}: the "
                "root external Davies dtbc consumption differs (WRF "
                "post-increment bind vs legacy elapsed-based), so "
                "resuming would splice two different trajectories.  "
                "Regenerate the checkpoint under the current semantic "
                "(a header without the key is a pre-bind file).")
    _require_nssl2_restart_contract(header, cfg, path)
    _require_physics_setup_match(header, state, cfg, path)
    _validate_ensemble_stochastic_checkpoint(header, stored, state)
    try:
        elapsed = _admissible_elapsed_seconds(
            header["elapsed_seconds"], f"restart file {path}")
    except (TypeError, ValueError, KeyError, RestartManifestError) as exc:
        raise RestartMismatchError(
            f"restart file {path} has an invalid elapsed_seconds") from exc
    stream_header = header.get(BOUNDARY_STREAM_HEADER_KEY)
    if isinstance(stream_header, dict) and not sealed_forcing_extension:
        # A checkpoint written before its preparation sealed is validated
        # under the preserved-prefix contract: every interval it recorded
        # must be byte-identical in the live set.  A run streaming from
        # another head is refused by name before that comparison.
        live = getattr(getattr(getattr(state, "lateral_boundaries", None),
                               "intervals", None), "head_sha256", None)
        if live is not None and live != stream_header.get("head_sha256"):
            raise RestartMismatchError(
                f"restart file {path} was written against the prepared head "
                f"{stream_header.get('head_sha256')}, and this run streams "
                f"from the prepared head {live}")
        preserved_forcing_prefix = True

    def live_setup_fingerprint():
        # Only the exact and the nested-extension checks compare the whole
        # setup fingerprint.  It covers every forcing interval, so on a
        # series still being prepared it waits for the seal, which the
        # preserved-prefix check below does not need.
        return setup_fingerprint(state)

    if preserved_forcing_prefix:
        if sealed_forcing_extension:
            raise ValueError('A restore cannot declare two forcing continuation modes')
        _require_preserved_forcing_prefix(
            header, _stored_prefix_view(header, state, path=path), cfg,
            path=path, elapsed=elapsed)
    elif sealed_forcing_extension:
        if header.get("forcing_extension_mode") != \
                SEALED_FORCING_EXTENSION_MODE:
            raise RestartMismatchError(
                f"restart file {path} was not intentionally sealed for "
                "forcing extension")
        stored_prefix = header.get("lateral_boundary_prefix")
        if getattr(cfg, "specified", False):
            if stored_prefix is None:
                raise RestartMismatchError(
                    f"restart file {path} specified root has no sealed "
                    "forcing inventory")
            _require_sealed_forcing_extension(
                header, state, path=path, elapsed=elapsed)
        elif getattr(cfg, "nested", False):
            if stored_prefix is not None:
                raise RestartMismatchError(
                    f"restart file {path} nested child unexpectedly carries "
                    "a sealed external forcing inventory")
            if (header.get("setup_core_fingerprint")
                    != setup_core_fingerprint(state)
                    or header["setup_fingerprint"]
                    != live_setup_fingerprint()):
                raise RestartMismatchError(
                    f"restart file {path} changed immutable child/nest setup "
                    "while extending root forcing")
        else:
            raise RestartMismatchError(
                f"restart file {path} cannot use sealed forcing extension: "
                "the live domain is neither a specified root nor a nested "
                "child")
    elif header["setup_fingerprint"] != live_setup_fingerprint():
        raise RestartMismatchError(
            f"restart file {path} was written on a different model setup "
            "(base state / coordinates / map factors fingerprint "
            "mismatch); rebuild the identical preparation before restoring")
    manifest = header["array_manifest"]
    if not isinstance(manifest, dict):
        raise RestartMismatchError(
            f"restart file {path} has a malformed array manifest")
    if set(stored) != set(manifest):
        missing = sorted(set(manifest) - set(stored))
        extra = sorted(set(stored) - set(manifest))
        raise RestartMismatchError(
            f"restart file {path} arrays disagree with its own manifest "
            f"(missing {missing}, extra {extra})")
    for key, spec in manifest.items():
        try:
            matches = (list(stored[key].shape) == list(spec["shape"])
                       and str(stored[key].dtype) == spec["dtype"])
        except (KeyError, TypeError) as exc:
            raise RestartMismatchError(
                f"restart member {key} has a malformed manifest entry") from exc
        if not matches:
            raise RestartMismatchError(
                f"restart member {key} does not match its manifest entry")

    _validate_nssl2_stored_restart_state(
        header, stored, state, cfg, path, elapsed)
    _validate_thompson_aerosol_stored_restart_state(stored, state, cfg, path)
    _validate_milbrandt2_stored_restart_state(stored, state, cfg, path)
    stored_state = {key[len("state/"):]: value
                    for key, value in stored.items()
                    if key.startswith("state/")}
    # The chem species fields ride beside the fixed inventory, named by the
    # chem table this run loaded (woof/core/chem_state.py); a chem-off
    # state adds none, so every existing checkpoint's key set is unchanged.
    expected_state = {name for name in serialized_state_attrs(state)
                      if getattr(state, name, None) is not None}
    if set(stored_state) != expected_state:
        missing = expected_state - set(stored_state)
        detail = ""
        if missing == set(ADVECTIVE_FORCING_STATE):
            # The ONE key-layout change this release makes, named with its
            # remedy rather than left as a two-list diff the reader has to
            # difference by eye.  A checkpoint written before the dycore
            # exported the pair carries every other array this run needs.
            detail = (
                "  This checkpoint predates the dycore's advective forcing "
                f"export ({', '.join(sorted(missing))}, WRF RTHFTEN/"
                "RQVFTEN): it was written by a build whose cumulus scheme "
                "was fed hard zeros for that lane, so resuming it here "
                "would either continue a different forecast or start the "
                "lane from an unwritten buffer.  Remedy: resume from a "
                "checkpoint this build wrote, or start the run again from "
                "its prepared state.")
        raise RestartMismatchError(
            f"restart state arrays {sorted(set(stored_state))} do not "
            f"match this configuration's state {sorted(expected_state)}."
            f"{detail}")
    for name, host in stored_state.items():
        _check_array(host, getattr(state, name), f"state/{name}")

    for key, host in stored.items():
        if not key.startswith("scratch/"):
            continue
        slot = key[len("scratch/"):]
        if not _restorable_scratch_slot(slot):
            raise RestartMismatchError(
                f"restart carries non-serializable scratch slot {slot!r}")
        _validate_scratch_target(state, slot, host, key)

    driver = getattr(state, "physics", None)
    _validate_member_namespaces(stored, state, driver, path, format_version)
    _validate_hydro_payload(stored, header, driver)
    driver_keys = [key for key in stored
                   if key.startswith(("driver/", "fields/", "cumulus/", "held/", "pbl/"))]
    if driver is None:
        if driver_keys:
            raise RestartMismatchError(
                "restart carries physics-driver state but the resuming "
                "state has no PhysicsDriver attached")
        if header.get("driver") is not None:
            raise RestartMismatchError(
                "restart has a physics-driver header but the resuming state "
                "has no PhysicsDriver attached")
    else:
        _validate_driver_payload(
            stored, header, state, driver, elapsed, format_version)

    if cfg.specified:
        if (state.lateral_boundaries is None
                or getattr(state, "_lateral_boundary_device", None) is None):
            raise RestartMismatchError(
                "cfg.specified requires attach_lateral_boundaries(state, "
                "...) BEFORE restore_restart: the resident LBC device "
                "tables are rebuilt by preparation, not by the restart")
        state.lateral_boundaries.interval_at(elapsed)
    return _ValidatedRestart(
        path=path, header=header, stored=stored,
        format_version=format_version, elapsed=elapsed)


def _grid_relocated(header, cfg, *, parent_by_grid=None) -> bool:
    """Did THIS grid's ground move during the checkpointed run?

    ``moved_grid_ids`` lists the grids a relocating run moved itself.  A
    grid's ground also moves when any ANCESTOR moves and carries it along
    at the same parent-relative placement: the carried nest is rebuilt
    shifted, its interior rain lands in its ring, and it is never in the
    list, because the list is also what decides which grids a resume
    puts back through a placement rebuild.  So the answer is "this grid
    or one of its ancestors is listed", walked over ``parent_by_grid``
    (the checkpoint set's grid -> parent map) with this member's own
    ``parent_id`` as the fallback for the first step.  A nest that holds
    its ground while its parent moves (an earth-fixed child) is read as
    moved too, which keeps its accumulators exactly as stored: its ring
    was never shifted, so the stored ring is already the zero the
    migration would write.  A grid on another branch of the tree, or the
    root, is not an ancestor of the mover and keeps the migration.

    An older checkpoint can carry the relocation block without the list.
    That is read as "moved", not as "did not move", and the fallback is
    safe in both directions: the block's PRESENCE already proves a build
    far newer than the ring exclusion, so such a file has nothing to
    migrate, while normalizing it anyway would be the data loss this
    gate exists to stop.
    """
    relocation = header.get("relocation")
    if not isinstance(relocation, Mapping):
        return False
    moved = relocation.get("moved_grid_ids")
    if moved is None:
        return True
    grid_id = header.get("grid_id")
    if grid_id is None:
        grid_id = getattr(cfg, "grid_id", None)
    if grid_id is None:
        return True
    moved = {int(gid) for gid in moved}
    parents = {int(gid): int(parent or 0)
               for gid, parent in (parent_by_grid or {}).items()}
    parents.setdefault(int(grid_id), int(header.get("parent_id") or 0))
    seen: set[int] = set()
    gid = int(grid_id)
    while gid > 0 and gid not in seen:
        if gid in moved:
            return True
        seen.add(gid)
        gid = parents.get(gid, 0)
    return False


def _apply_validated_restart(validated: _ValidatedRestart,
                             state, cfg, *,
                             parent_by_grid=None) -> RestartInfo:
    """Apply an already complete-set-validated member in place."""
    header = validated.header
    stored = validated.stored
    format_version = validated.format_version
    elapsed = validated.elapsed
    driver = getattr(state, "physics", None)
    asarray = _asarray_like(state)

    stored_state = {key[len("state/"):]: value
                    for key, value in stored.items()
                    if key.startswith("state/")}
    for name, host in stored_state.items():
        getattr(state, name)[...] = asarray(host)
    # The checkpoint-only carriers.  A checkpoint written before this
    # namespace existed simply has none, and the freshly allocated zeros
    # are what that build resumed with anyway -- so absence is a note's
    # worth of information, not a refusal, and every checkpoint already
    # on disk keeps working.
    for name in CHECKPOINT_ONLY_STATE:
        target = getattr(state, name, None)
        host = stored.get(f"acoustic/{name}")
        if target is None or host is None:
            continue
        target[...] = asarray(host)
    stored_scratch = set()
    for key, host in stored.items():
        if key.startswith("scratch/"):
            slot = key[len("scratch/"):]
            stored_scratch.add(slot)
            if slot in OUTPUT_PERSISTENT and state.existing_scratch(slot) is None:
                print(f"note: checkpoint carries inactive output bucket {slot!r}; dropped")
                continue
            if slot in HAIL_PERSISTENT and state.existing_scratch(slot) is None:
                print(f"note: checkpoint {validated.path.name} carries {slot!r}; "
                      "hail diagnostics are off, so the accumulator is dropped")
                continue
            if (slot == "up_heli_max" or _is_tracker_window_slot(slot)) \
                    and state.existing_scratch(slot) is None:
                # The eagerly allocated diagnostic accumulator and the
                # consumer tracking windows beside it: a state prepared
                # with nwp_diagnostics=0 carries none of them, and
                # restoring one anyway would allocate a plane nothing
                # folds -- a stale, never-again-updated UP_HELI_MAX in
                # every frame, or a window whose consumer does not exist.
                # Dropping a diagnostic is a note, never a refusal.
                print(f"note: checkpoint {validated.path.name} carries "
                      f"{slot!r} but this run has nwp_diagnostics=0; the "
                      "diagnostic accumulator is dropped")
                continue
            if slot in _THOMPSON_CLOUD_BOOKING_SLOTS \
                    and state.existing_scratch(slot) is None:
                # Correct mode WB1: the physics driver allocates the
                # settled-cloud pair exactly when the run books it
                # (Thompson with thompson_cloud_surface_booking on, an
                # output-only key a resume may change).  A run resumed with
                # the booking off would otherwise carry the stored total in
                # every frame, never updated again.
                print(f"note: checkpoint {validated.path.name} carries "
                      f"{slot!r} but this run does not book settled cloud "
                      "(thompson_cloud_surface_booking off); the total is "
                      "dropped")
                continue
            state.scratch(host.shape, slot, dtype=host.dtype)[...] = asarray(host)
    # Old-checkpoint tolerance for accumulators that postdate the file: a
    # live serialized slot with no stored counterpart stays at its
    # zero-initialized allocation.  That is the correct diagnostic restore
    # (the first frame after resume simply covers a shortened max window),
    # so it is a NOTE, never a refusal.
    for slot in sorted(SERIALIZED_SCRATCH_SLOTS):
        if slot not in stored_scratch \
                and state.existing_scratch(slot) is not None:
            state.existing_scratch(slot)[...] = 0.0
            print(f"note: checkpoint {validated.path.name} predates the "
                  f"serialized accumulator {slot!r}; restored "
                  "zero-initialized")
    if driver is not None:
        _restore_driver(stored, header, state, driver, elapsed, asarray,
                        format_version)
    stochastic_header = header.get("ensemble_stochastic")
    if stochastic_header is not None:
        from woof.ensemble.stochastic_execution import restore_checkpoint_payload
        restore_checkpoint_payload(state._ensemble_stochastic, stochastic_header,
            {key: value for key, value in stored.items() if key.startswith("stochastic/")})
    # v5 migration normalization: checkpoints written before the
    # spec-zone ring exclusion (same format version) can carry nonzero
    # ring MP accumulators/diagnostics and stale ring h_diabatic from
    # whole-field microphysics; no WRF-valid trajectory can contain them.
    # Idempotent on post-fix checkpoints (see the function's docstring)
    # EXCEPT on a domain that relocated, whose ring holds accumulator
    # values a move shifted there out of the interior -- real history for
    # ground that is still in the domain, and zeroing it loses rain the
    # rest of the forecast never puts back.
    from woof.core.microphysics import normalize_spec_zone_ring_after_restore
    normalize_spec_zone_ring_after_restore(
        state, cfg, relocated=_grid_relocated(
            header, cfg, parent_by_grid=parent_by_grid))
    if getattr(cfg, "umove", 0.0) or getattr(cfg, "vmove", 0.0):
        from woof.moving_frame import frame_velocity
        state._moving_frame_velocity = frame_velocity(cfg)
        state._moving_frame_initialized = True
    state.elapsed_seconds = elapsed
    return RestartInfo(elapsed_seconds=elapsed,
                       run_trackers=header.get("run_trackers"),
                       header=header)


def restore_restart(path, state, cfg, *, preserved_forcing_prefix=False) -> RestartInfo:
    """Restore a restart file into a freshly PREPARED state, in place.

    The caller must have completed the normal deterministic setup first
    (base state loaded, physics initialized, lateral boundaries attached):
    restore validates the config echo and the setup fingerprint, then
    overwrites every serialized array in place â€” surface fields are never
    rebound, preserving the SFCLAY-result aliasing â€” rebinds the held
    tendency containers with the stored coupled arrays, restores the KF
    W0AVG onto the cumulus adapter (rebinding its history to this state),
    and restores ``elapsed_seconds`` LAST, after
    ``attach_lateral_boundaries`` reset it to zero.
    """
    validated = _validate_restart(path, state, cfg, preserved_forcing_prefix=preserved_forcing_prefix)
    return _apply_validated_restart(validated, state, cfg)


def _restore_driver(stored, header, state, driver, elapsed, asarray,
                    format_version: int) -> None:
    # _validate_driver_payload and _validate_member_namespaces have checked
    # every member before _apply_validated_restart mutates any model state.
    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.core.physics import PhysicsTendencies

    hydro = getattr(driver, "hydro", None)
    if hydro is not None:
        for key, target in hydro.arrays.items():
            target[...] = asarray(stored[f"hydro/{key}"])
        hydro.restore_metadata(header["driver"]["hydro"])

    # Surface/Noah fields: in place only (never rebind â€” sfclay_result and
    # the Noah launch read these exact device arrays).
    stored_fields = {key[len("fields/"):]: value
                     for key, value in stored.items()
                     if key.startswith("fields/")}
    gated = _diagnostic_gated_field_changes(stored_fields, header, driver)
    for name, host in stored_fields.items():
        if name in gated:
            continue
        target = driver.fields[name]
        target[...] = asarray(host)
    for name, change in sorted(gated.items()):
        if change == "initialised":
            driver.fields[name][...] = 0
        print(_diagnostic_gated_note(name, change, header))
    lake = getattr(driver, "lake", None)
    if lake is not None:
        lake.invalidate_columns()

    for name in ("rthratenlw", "rthratensw"):
        key = f"driver/{name}"
        target = getattr(driver, name)
        target[...] = asarray(stored[key])
    key = "driver/pending_rainbl"
    driver._pending_rainbl[...] = asarray(stored[key])

    # The checkpoint-only carriers.  Tolerant in BOTH directions, and
    # each direction is a real configuration rather than a defensive
    # shrug: a checkpoint written before this namespace existed carries
    # no key (and its build resumed with zeros anyway), and a resumed
    # configuration whose longwave scheme publishes no TOA flux has no
    # buffer to fill.
    for name in sorted(DRIVER_CHECKPOINT_ONLY_ATTRS):
        target = getattr(driver, name, None)
        host = stored.get(f"diag/{name}")
        if target is None or host is None:
            continue
        target[...] = asarray(host)

    # Preserve the constructor's carrier identities for tile buffers and
    # adapters. The complete pair was validated before any state mutation.
    for name in sorted(DRIVER_HELD_FORCING_ATTRS):
        target = getattr(driver, name, None)
        if target is not None:
            target[...] = asarray(stored[f"held/{name}"])

    for key, target in pbl_raw_manifest(driver).items():
        target[...] = asarray(stored[key])

    for key, target in pbl_diagnostic_manifest(driver).items():
        if key in stored:
            target[...] = asarray(stored[key])
        else:
            target[...] = 0.0  # newly enabled output-only diagnostic

    # Held tendencies: rebind with the stored COUPLED arrays (no
    # recoupling â€” see the manifest argument).  compute() recomposes the
    # working sum from these components before the next consumption.
    fire = getattr(driver, "fire", None)
    if fire is not None:
        fire.restore({key[5:]: asarray(value) for key, value in stored.items() if key.startswith("fire/")},
                     header["driver"]["fire"])
    for tend_name in _live_driver_tendency_attrs(driver):
        components = {}
        live_tendency = getattr(driver, tend_name)
        for comp in TENDENCY_COMPONENTS:
            comp_key = f"driver/{tend_name}/{comp}"
            legacy_absent_ice = (comp == "rqi"
                and tend_name == "pbl_tendencies"
                and getattr(driver, "sase_active", False)
                and getattr(state, "qi", None) is None
                and header["config"].get("bldt") == 0.0)
            if legacy_absent_ice:
                components[comp] = None
            elif comp_key in stored:
                components[comp] = asarray(stored[comp_key])
            elif format_version == 2 or (comp == "rw"
                    and header["config"].get("bldt") == 0.0):
                # Legacy absence meant an identically zero optional held
                # category.  Preserve the constructor's eager canonical
                # buffer so a supported v2 resume cannot grow its inventory
                # at the first scheduled physics call.
                components[comp] = getattr(live_tendency, comp)
            else:
                components[comp] = None
        setattr(driver, tend_name, PhysicsTendencies(**components))
    config = header["config"]
    reuse_pbl = bool(config.get("bl_pbl_physics")
                     and config.get("bldt") == 0.0
                     and (driver.radiation_active or driver.cu_physics))
    if (getattr(driver, "fire_tendencies", None) is None
            and (not (driver.radiation_active or driver.cu_physics) or reuse_pbl)):
        # Preserve both proven identity paths and release the constructor's
        # superseded initial PBL buffers immediately on restore. Fire needs
        # a distinct composed target: += fire would otherwise mutate held
        # PBL tendencies and replay the source on subsequent steps.
        driver.tendencies = driver.pbl_tendencies

    from woof.core.physics import microphysics_scratch_slots
    slot_map = dict(microphysics_scratch_slots(driver.mp_physics))
    components = {comp: None for comp in MICROPHYSICS_COMPONENTS}
    if not slot_map:
        # v3+ mp=0 diagnostics are deterministic zero placeholders.  Preserve
        # a v2 file's old owned values exactly for compatibility.
        if format_version == 2:
            for comp in MICROPHYSICS_COMPONENTS:
                key = f"driver/microphysics/{comp}"
                if key in stored:
                    components[comp] = asarray(stored[key])
            driver.microphysics = MicrophysicsDiagnostics(**components)
    else:
        for comp, slot in slot_map.items():
            scratch_key = f"scratch/{slot}"
            legacy_key = f"driver/microphysics/{comp}"
            scratch_host = stored.get(scratch_key)
            legacy_host = stored.get(legacy_key)
            source = scratch_host if scratch_host is not None else legacy_host
            if source is None:
                # An old pre-first-call optional can legitimately be absent;
                # the new canonical scratch slot is already zero-filled.
                target = state.scratch(state.mup.shape, slot)
            else:
                target = state.scratch(source.shape, slot, dtype=source.dtype)
                if scratch_host is None:
                    target[...] = asarray(source)
            components[comp] = target
        driver.microphysics = MicrophysicsDiagnostics(**components)

    driver_header = header["driver"]
    driver.call_counts.update(
        {key: int(value)
         for key, value in driver_header["call_counts"].items()})
    if (getattr(driver, "cam_ozone", None) is not None
            and "cam_ozone" not in driver_header["call_counts"]):
        # Older legacy checkpoints retained this same field under the same
        # key; their radiation counter is its producer/consumer history.
        driver.call_counts["cam_ozone"] = int(driver.call_counts["radiation"])
    driver.ysu_nan_guard_fires = int(driver_header["ysu_nan_guard_fires"])
    driver.microphysics_updates = int(
        driver_header["microphysics_updates"])
    _restore_carriers(driver, driver_header)

    if "cumulus/w0avg" in stored:
        adapter = driver.cumulus_callable
        restored_w0avg = asarray(stored["cumulus/w0avg"])
        if (adapter.w0avg is not None
                and adapter.w0avg.shape == restored_w0avg.shape
                and adapter.w0avg.dtype == restored_w0avg.dtype):
            adapter.w0avg[...] = restored_w0avg
        else:
            adapter.w0avg = restored_w0avg
        # Rebind the history adapter to THIS state object so the next
        # update does not re-zero W0AVG (the object-identity trap: the
        # adapter invalidates on `self._history_state is not state`).
        adapter._history_state = state
        adapter._history_time = elapsed

    if "radiation/o33d_grid" in stored:
        # Legacy-RRTMG root o33d (WRF's restart-carried O3RAD analogue):
        # reattach as HOST float32 exactly as the adapter retains it, so
        # a child domain's first post-restore radiation call interpolates
        # the identical field the uninterrupted run would have used.
        radiation = driver.radiation_callable
        restored_o33d = np.asarray(
            stored["radiation/o33d_grid"], dtype=np.float32)
        carrier = getattr(driver, "o3rad", None)
        if carrier is not None:
            carrier.set(np.ascontiguousarray(restored_o33d))
        elif radiation is not None:
            radiation._o33d_grid = np.ascontiguousarray(restored_o33d)


__all__ = [
    "ADVECTIVE_FORCING_STATE",
    "CONFIG_DIAGNOSTIC_FIELDS", "CONFIG_RUN_LENGTH_FIELDS",
    "DRIVER_REBUILT_ATTRS",
    "DRIVER_CHECKPOINT_ONLY_ATTRS",
    "DRIVER_SERIALIZED_ATTRS", "DRIVER_TENDENCY_ATTRS",
    "DRIVER_HELD_FORCING_ATTRS",
    "CUMULUS_ALGORITHM_IDENTITIES", "LAND_SURFACE_ALGORITHM_IDENTITIES",
    "LAND_SURFACE_PARAMETER_SOURCES",
    "MICROPHYSICS_COMPONENTS", "REBUILT_SCRATCH_PREFIXES",
    "MICROPHYSICS_ALGORITHM_IDENTITIES", "PBL_ALGORITHM_IDENTITIES",
    "URBAN_ALGORITHM_IDENTITIES",
    "DYCORE_MIXING_ALGORITHM_IDENTITY", "dycore_mixing_identity",
    "UPPER_DAMPING_ALGORITHM_IDENTITY", "upper_damping_identity",
    "NSSL2_LEGACY_RESTART_ALIASES", "NSSL2_RESTART_AUXILIARY_STATE",
    "NSSL2_RESTART_CONTRACT_VERSION", "NSSL2_RESTART_PRECIPITATION_SLOTS",
    "NSSL2_RESTART_PROGNOSTICS",
    "PHYSICS_ASSET_PATHS", "PHYSICS_SETUP_SCHEMA_VERSION",
    "RADIATION_ABOVE_ATMOSPHERE_POLICIES",
    "RADIATION_ALGORITHM_IDENTITIES",
    "SURFACE_LAYER_ALGORITHM_IDENTITIES",
    "READABLE_RESTART_FORMAT_VERSIONS", "REBUILT_SCRATCH_SLOTS",
    "RESTART_FORMAT_VERSION", "RESTART_MEMBER_NAMESPACES",
    "RETIRED_RESTART_FORMAT_VERSIONS", "require_readable_format_version",
    "ROOT_EXTERNAL_LBC_CLOCK_IDENTITY",
    "ROOT_EXTERNAL_LBC_CLOCK_LEGACY", "RestartInfo", "TreeRestartInfo",
    "SEALED_FORCING_EXTENSION_MODE",
    "RestartManifestError", "RestartMismatchError",
    "CARRIED_SCRATCH_SLOTS", "CARRIED_SCRATCH_PREFIXES",
    "carried_scratch_manifest",
    "RESTART_ONLY_DRIVER_SLOTS",
    "SERIALIZED_SCRATCH_SLOTS", "STATE_REBUILT_ATTRS",
    "STRICT_LAZY_STATE_ATTRS", "STATE_DERIVED_SETUP_ARRAYS",
    "STATE_SERIALIZED_ATTRS", "STATE_SETUP_ARRAYS", "STATE_SETUP_SCALARS",
    "THOMPSON_AEROSOL_RESTART_STATE",
    "THOMPSON_AEROSOL_RESTART_SURFACE_STATE",
    "MILBRANDT2_RESTART_STATE",
    "MILBRANDT2_RESTART_PRECIPITATION_SLOTS",
    "require_identifiable_checkpoint_schemes",
    "unidentifiable_checkpoint_schemes",
    "TENDENCY_COMPONENTS", "classify_scratch_slot", "classify_state_attr",
    "ask_checkpoint_physics_identity",
    "physics_setup_fingerprint", "physics_setup_identity",
    "root_external_lbc_clock_identity",
    "NEST_LIFECYCLE_BLOCK_KEYS", "NEST_LIFECYCLE_CONTRACT",
    "NEST_LIFECYCLE_HEADER_KEY", "TreeLifecycleHeader",
    "declares_nest_lifecycle", "lifecycle_followers",
    "lifecycle_window_slots",
    "read_tree_lifecycle_header",
    "read_restart_header", "require_tree_checkpoint_legal",
    "restart_filename", "restore_restart", "restore_tree_restart",
    "lateral_boundary_prefix_identity", "setup_core_fingerprint",
    "setup_fingerprint", "state_manifest", "write_restart",
    "WRITTEN_MODE_HEADER_KEY", "RESIDENT_WRITTEN_MODE",
    "STREAMED_WRITTEN_MODE", "written_mode_note", "header_written_mode",
    "checkpoint_placements",
    "tree_restart_members",
    "write_tree_restart",
]
