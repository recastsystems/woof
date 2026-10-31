"""Hash-bound, atomically published prepared real-data state caches.

The normal real-data path constructs a full 3-D state for every forcing
time even though specified lateral forcing retains only narrow boundary
strips.  A repeated benchmark therefore used to repay the complete vertical
interpolation for every forcing hour before its first model step.

This module persists the integration-ready products of that work:

* the exact time-zero prognostic/diagnostic state;
* the original FP64 vertical coordinate and base state used to load it;
* only the horizontally mapped surface fields needed to initialize physics;
* every immutable host lateral-boundary value/tendency table.

The cache is rebuildable input, not a model restart.  It is nevertheless
fail-closed: callers supply a canonical identity binding the source manifest,
static cache, configuration, namelist, and code revision.  Every array has a
shape/dtype/content digest, the complete manifest has its own digest, and a
temporary directory is renamed into place only after the header is complete.
No pickle or object arrays are accepted.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence
import uuid

import numpy as np

from woof.checkpoint_identity import CONFIG_DIAGNOSTIC_FIELDS
from woof.vertical_contract import (
    validate_coordinate_shapes,
    validate_explicit_eta_grid,
)
from woof.namelist_seal import validated_namelist_extension_invariant


PREPARED_CACHE_SCHEMA = "gpuwm-prepared-real-cache-v1"
#: File-level rename for one payload or header written aside.  Bound once,
#: apart from the directory publication rename (``os.replace`` at the call
#: sites that publish a whole bundle), so the two are separate operations.
_replace_file = os.replace
SEALED_PREPARED_EXTENSION_MODE = "sealed-prefix-v1"
_HEADER_NAME = "header.json"
#: The longest name a prepared-cache write gives a file in its directory:
#: the header written aside, then renamed onto ``header.json``.  Payloads
#: are ``aNNNNN.npy`` and theirs ``aNNNNN.npy.tmp``, a character shorter.
HEADER_PARTIAL_NAME = _HEADER_NAME + ".tmp"
_MET_REQUIRED = frozenset({
    "LANDSEA", "SKINTEMP", "T2", "U10", "V10",
})
_LEGACY_HRRR_SOIL = frozenset({"SOILT", "SOILW"})
_MET_OPTIONAL = frozenset({
    "SST", "XICE", "SEAICE", "SNOW", "SNOW_EC", "SNOWH",
    "VEGFRA",
    # The source's own PBL height (HRRR wrfnat HPBL), when it publishes
    # one: the initial physics state is seeded from it.
    "PBLH",
    # The source's low/middle/high cloud cover (HRRR wrfnat LCDC/MCDC/HCDC,
    # percent): the initial CLDFRA's cover.
    "LCDC", "MCDC", "HCDC",
})
_CANONICAL_SURFACE_REQUIRED = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})


def _creation_utc(directory):
    return datetime.now(timezone.utc).isoformat()


class PreparedCacheMismatchError(ValueError):
    """The cache is valid, but belongs to a different requested setup."""


class PreparedCacheCorruptError(ValueError):
    """The cache is incomplete, malformed, or fails a content digest."""


class PreparedCacheResourceLimitError(OSError):
    """Opening a prepared cache exhausted file handles or mapping resources."""


def _raise_if_resource_limit(error: BaseException, *, context: str) -> None:
    """Preserve resource exhaustion instead of reporting damaged data."""
    if isinstance(error, PreparedCacheResourceLimitError):
        raise error
    if not isinstance(error, OSError) or error.errno not in {
            errno.EMFILE, errno.ENFILE, errno.ENOMEM}:
        return
    if error.errno == errno.ENOMEM:
        reason = "memory or virtual-memory mapping resource limit (ENOMEM)"
        remedy = ("release unused mappings; inspect available memory, address-space "
                  "limits and Linux vm.max_map_count")
    else:
        reason = ("process open-file resource limit (EMFILE)" if error.errno == errno.EMFILE
                  else "system open-file resource limit (ENFILE)")
        remedy = ("check ulimit -n and raise the process open-file limit, or "
                  "close unused file descriptors before retrying")
        if error.errno == errno.ENFILE:
            remedy += "; also free system file handles or raise the system open-file limit"
    raise PreparedCacheResourceLimitError(
        error.errno, f"{context}: {reason}; {remedy}",
        error.filename) from error


def _canonical(value) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _plain(value):
    """Recursively unwrap read-only containers into plain JSON types.

    ``json`` serializes ``dict``/``list`` and nothing else that behaves
    like them.  A ``MappingProxyType`` -- the exact type
    :func:`restore_prepared_cache` hands back as
    ``CachedInitialResult.hydrometeor_initialization``, deliberately, so
    a restored document cannot be mutated -- is a ``Mapping`` but not a
    ``dict``, so ``json.dumps`` raised ``TypeError: Object of type
    mappingproxy is not JSON serializable``.  That turned "restore a
    prepared root, then write the child cache derived from it" into a
    crash with no mention of the field that caused it.

    Unwrapping here rather than at each call site keeps ONE normalization
    for every identity, metadata and manifest document this module
    canonicalizes, and it changes no output: a proxy serializes to the
    same object its underlying mapping does.  Tuples already normalized
    to lists through ``json``; sets and frozensets do not, and are not
    accepted -- an unordered container has no canonical serialization, so
    guessing one would put an order-dependent digest in an identity.
    """

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "prepared-cache documents cannot contain a set: an unordered "
            "container has no canonical serialization, and an identity "
            "digest must not depend on iteration order")
    return value


def _json_copy(value):
    """Normalize tuples/numpy-free JSON values and reject non-finite data."""
    return json.loads(_canonical(value))


def prepared_domain_config_identity(domain_config) -> dict[str, object]:
    """JSON-stable domain identity, including an ISO per-domain start."""
    from woof.experiment import domain_config_document

    document = domain_config_document(domain_config)
    start_time = document.get("start_time")
    if isinstance(start_time, datetime):
        document["start_time"] = start_time.isoformat()
    run = document.get("run", {})
    if int(run.get("topo_wind", 0) or 0) or int(run.get("gwd_opt", 0) or 0):
        from woof.static.rust_bridge import OROGRAPHIC_MARKER

        # Earlier orographic statics chose some source stencils in f64 and
        # scaled source words before interpolation. Those arrays remain
        # internally hash-consistent but produce different terrain drag.
        document["orographic_sampling_contract"] = OROGRAPHIC_MARKER
    # A chem domain binds the chem table it was prepared with (schema, every
    # row/set/source file, the selection), so an edited row can never serve
    # a cache prepared under the old one.  Absent -- not null -- on a
    # chem-off domain, so every existing identity is unchanged.
    run = getattr(domain_config, "run", None)
    if run is not None and getattr(run, "chem_sets", ()):
        from woof.chem_table import load as load_chem_table
        document["chem_table_identity"] = load_chem_table(run).identity
    return _json_copy(document)


#: Identity fields added to the prepared-domain document AFTER caches
#: carrying the older shape were already in the field, mapped to the
#: value that means "this feature is not in use".
#:
#: v1.1.0 gave every domain an optional per-domain ``start_time`` for
#: staggered nest starts.  The prepared-cache identity is compared by
#: strict equality, and a v1.0.1 header was serialized before the field
#: existed, so after upgrading the wheel EVERY prepared tree in the
#: field became unrunnable -- refused with "d01 cache domain config
#: differs from experiment", a sentence that points at the user's
#: experiment TOML when the cause was a package upgrade.  A test host G
#: validation run diffed the two documents: exactly one added key, and
#: zero value differences among the eleven shared keys and ~110 run
#: fields.
#:
#: Tolerating that is not weakening identity, and the narrowness is what
#: makes it true.  A field ABSENT from the header and holding its
#: documented default in the live configuration describes the same
#: prepared state as a header written before the field existed: the
#: feature is off in both.  A field absent from the header and holding a
#: NON-default value describes a different setup and is still refused,
#: as is any field the header carries and the live configuration
#: contradicts.  Entries are added here deliberately, one per field, by
#: whoever adds the field -- never by a rule that tolerates absence in
#: general.  That instruction was missed four times in a row
#: (``run.mp28_aerosol_source``, ``run.wif_climatology_path``,
#: ``run.p3_backend``, ``run.ntiedtke_tiedtke_closure``), each time
#: refusing every prepared tree written before the field, so the duty
#: no longer rests on the instruction: tests/test_prepared_cache.py pins
#: today's ``RunConfig`` against the field set at the identity header's
#: introduction and fails on any field that is in neither this table nor
#: :data:`STRICT_IDENTITY_FIELDS`.
#: v1.8 adds the per-domain ``spawn`` declaration (dormant
#: spawn-triggered nests, woof/core/nest_spawn.py).  Its not-in-use
#: value is ``None`` -- an ordinary live domain -- which is exactly the
#: state every header written before the field existed describes.  A
#: domain that really is dormant holds a SpawnConfig document instead,
#: and is refused against an older header, as it must be: the cache was
#: prepared for a nest that is not the dormant one.
#:
#: Five more fields joined ``DomainConfig`` after spawn WITHOUT joining
#: this table when they landed -- ``tiles`` (df5cf42d0), ``output``
#: (8d8855a8c), and the lifecycle trio ``retire``/``rearm``/``follow``
#: (8e663a751/21254c056) -- which re-opened the V-12 hole for every
#: tree prepared before them: an upgrade refused the tree and blamed
#: the experiment file.  Each declares an optional feature whose
#: not-in-use value is ``None``, the exact state every pre-field header
#: describes, on the same grounds as ``spawn``: none of them changes
#: the prepared initial state or the boundary tables -- ``tiles`` and
#: ``output`` say how the run writes, the lifecycle trio says what a
#: nest does after birth.  A domain that really uses one holds a
#: document instead of ``None`` and is refused against an older header.
#: ``run.ntiedtke_tiedtke_closure`` (cu_physics = 16) is the first
#: NESTED member of this table.  The walk below builds dotted paths,
#: so a ``run.`` entry has always worked here -- this is the first to
#: use it.  Its not-in-use value is ``False``, which selects New
#: Tiedtke's OWN closure: exactly what every cache prepared before
#: the flag existed was prepared with, so "absent" and "default"
#: describe the same prepared state provably rather than plausibly.
DEFAULT_TOLERANT_IDENTITY_FIELDS = frozenset({
    # 2.9.4 capability fields (lake state, mixed-layer ocean, sea-ice parameters; seaice_prognostic itself is
    # PREPARATION_INERT_RUN_FIELDS).  Each is read by the
    # forecast, not by the preparation that wrote an older tree, and its default is what such a tree was
    # run with; a non-default value is refused against an older header, which re-prepares (the safe side).
    "run.lake_state_accept_licence", "run.lake_state_path", "run.lake_state_source",
    "run.ocean_analysis", "run.ocean_analysis_source", "run.ocean_config", "run.ocean_levels",
    "run.ocean_mld_climatology", "run.ocean_s", "run.ocean_t", "run.ocean_z", "run.omdt",
    "run.seaice_initial_thickness", "run.seaice_new_ice_thickness", "run.seaice_ocean_heat_flux",
    # The operational HRRR output controls (2.9.4): interval precipitation
    # buckets and the fork's gsd_diagnostics maxima (the history cadence
    # change is a non-trajectory field).  Off (their defaults) is exactly
    # what every older tree was prepared and run with; a stated value
    # against an older header re-prepares (the safe side).
    "run.prec_acc_dt", "run.prec_acc_dt1", "run.gsd_diagnostics", "run.reset_interval1",
    "run.output_diagnostics", "run.diag_int", "run.wind_int",
    # G3 controls are absent from older trees; only their declared defaults tolerate absence.
    "run.maxiens",
    "run.maxens",
    "run.maxens2",
    "run.maxens3",
    "run.ensdim",
    "run.cu_diag",
    "run.convtrans_avglen_m",
    # 2.6.1's members: top-level DomainConfig declarations whose
    # off state is None.
    "start_time", "spawn", "tiles", "output", "retire", "rearm", "follow",
    # 2.8's per-domain follower reach bound, nested in a DECLARED follow
    # document.  Its not-in-use value is None (the engine default), the
    # state every follower header written before the key describes, and
    # the prepared initial state and boundaries never read it: it sizes
    # the statics corridor, which its own loader checks for coverage.
    "follow.reach_speed_m_s",
    # WRF's history window (cb8af5ccb), top-level DomainConfig fields
    # that joined the live document without joining this table, so the
    # routes that compare a RAW header (the stock-WRF export's walk, the
    # single-domain runner's equality) refused every bundle prepared
    # before them, 2.8.0's included.  Not-in-use is the dataclass
    # default (0.0, None): the start frame and every frame to the end,
    # which is what every earlier header's forecast wrote.  Preparation
    # reads neither, so the normalising routes drop them at any value
    # through NON_TRAJECTORY_IDENTITY_FIELDS below, as "tiles" is
    # dropped there.
    "history_begin_s", "history_end_s",
    # This branch's members, NESTED under run.  The walk builds dotted
    # paths, so these have always worked here.  Dropping them when
    # 2.6.1 restructured the table above would have re-refused every
    # tree prepared before 2.5.8 -- the exact V-12 hole this table
    # exists to close, reopened by the merge that adopted its fix.
    "run.mp28_aerosol_source", "run.wif_climatology_path",
    "run.ntiedtke_tiedtke_closure",
    # Correct-mode keys (woof/correct_mode.py, WB5/WB4): False is the
    # matched arithmetic every prepared tree written before them describes.
    # thompson_cloud_surface_booking (WB1) is output-only
    # (correct_mode.OUTPUT_ONLY_KEYS) and so is ruled on once, in
    # INERT_DIAGNOSTIC_IDENTITY_FIELDS through CONFIG_DIAGNOSTIC_FIELDS.
    "run.wsm6_conservative_surface", "run.ntiedtke_dry_mass_coupling",
    "run.wdm6_conservative_surface",
    "run.wsm3_conservative_surface", "run.wsm5_conservative_surface",
    "run.wsm7_conservative_surface",
    # HN1 (2.9.7), read by preparation (the handover closure on every
    # forcing frame).  An older tree was prepared with it off; as for the
    # keys above, its absence is tolerated at the RunConfig default, and an
    # exact-mode run (off) against an older header re-prepares.
    "run.hydrometeor_handover_numbers",
    # The New SAS knobs (cu_physics = 96 port), on the p3_backend argument
    # below: scheme-scoped (woof.config refuses a non-default value under
    # any other cumulus scheme), so a tree prepared before they existed
    # could not have selected anything but their defaults.
    "run.nsas_dx_factor", "run.sas_pgcon",
    # 2.6.1 added run.p3_backend to RunConfig without adding it here,
    # which refused EVERY tree prepared before it -- 16 of them on the
    # box that found this.  The FOURTH instance of this exact trap in
    # this one table, and the second by a different author.
    #
    # Tolerable on the strongest form of the argument: the field is
    # SCHEME-SCOPED to mp_physics = 50 (config.py:3128 refuses any
    # non-default value under every other scheme), so a cache written
    # before it existed could not have selected a P3 backend at all.
    # A tree that really did carries the value in its header and is
    # still compared strictly.
    "run.p3_backend",
    # The urban canopy keys (lane/urban-infra).  Preparation reads them
    # only when sf_urban_physics > 0 (the urban land-use legend and
    # FRC_URB2D, woof/static/highres*.py); at the default, 0, it
    # prepares the legend and tables every pre-urban tree was prepared
    # with, so "absent" and "default" are the same prepared state.  An
    # urban run against an older header carries a non-default value and
    # is still refused, as it must be: that tree has no urban legend.
    "run.sf_urban_physics", "run.use_wudapt_lcz", "run.num_urban_hi",
    # WRF's topo_wind and gwd_opt (lane/282-terrain-drag), on the urban
    # keys' argument: preparation reads them only to add the sub-grid
    # orographic statistics to the static build (woof.static.orographic);
    # at 0 it builds exactly the statics every older tree carries.  A
    # non-default value against an older header is refused, as it must be:
    # that tree has no orographic statistics.
    "run.topo_wind", "run.gwd_opt",
    # CLM lake off is the prescribed-water state in older prepared files.
    "run.sf_lake_physics", "run.use_lakedepth", "run.lakedepth_default", "run.lake_min_elev",
    # ---------------------------------------------------------------
    # The 80 other RunConfig fields that joined after the identity
    # header (1c6290410, 2026-07-19, which bound asdict(DomainConfig)
    # into every cache) and never joined this table.  Each was measured
    # against the preparation path (woof/ingest, the six direct
    # writers, DomainState's allocation) and falls into one of three
    # arguments, each of which makes "absent" and "default" the same
    # prepared state.  A non-default live value is still refused by
    # this table's narrowness; the stronger drop is
    # PREPARATION_INERT_RUN_FIELDS and needs its own ruling.
    #
    # (a) SCHEME-SCOPED: the scheme arrived in the same commit as the
    # field, so no header lacking the field describes a tree of that
    # scheme, and woof/config.py refuses a non-default value under
    # any other scheme.  mp=16 arrived with its two selectors
    # (ea7dcc296; wdm6_ccn_conc fills the cached CCN field, which only
    # an mp=16 header, always carrying the field, can hold).  mp=28
    # arrived with the aerosol pair (0ebda6608); preparation reads it
    # only on initialize_real's mp=28 arm.
    "run.wdm6_hail_opt", "run.wdm6_ccn_conc",
    "run.aer_init_opt", "run.wif_input_opt", "run.use_rap_aero_icbc",
    # (b) READ BY PREPARATION, DEFAULT IS THE PRE-FIELD BEHAVIOUR:
    # num_soil_layers (61903f124) defaults to Noah's four layers, the
    # only count preparation produced before the field; the count is
    # resolved from sf_surface_physics (config.soil_layer_count) and a
    # disagreeing value is refused, and the schemes that need another
    # count were routed after the field (RUC 6153ea505, Noah-MP
    # ab60dc8ab).
    "run.num_soil_layers",
    # (c) NEVER READ BY PREPARATION: each selects tendencies,
    # diagnostics or output the FORECAST computes, so the prepared
    # initial state and the boundary tables are the same under every
    # value (hrrr_hierarchy_direct._DOMAIN_PREPARATION_OVERRIDES
    # records the same finding for the turbulence rows).
    # MYNN PBL (e65b3ce31):
    "run.bl_mynn_closure", "run.bl_mynn_cloudpdf", "run.bl_mynn_mixlength",
    "run.bl_mynn_edmf", "run.bl_mynn_edmf_mom", "run.bl_mynn_edmf_tke",
    "run.bl_mynn_mixscalars", "run.scalar_pblmix", "run.bl_mynn_cloudmix", "run.bl_mynn_mixqt",
    "run.bl_mynn_output", "run.bl_mynn_tkeadvect", "run.icloud_bl",
    "run.scalar_adv_opt",
    # MM5/MYNN surface layer (76cd7f18b):
    "run.isftcflx", "run.iz0tlnd",
    # Noah LSM selector (9fae17c50), admitted for RUC as well. Prescribed
    # LAI and background albedo have the stronger forecast-only ruling in
    # PREPARATION_INERT_RUN_FIELDS below. The RUC sea-ice threshold switch
    # is absent-tolerant at its default 0. A stated 1 still binds: soil
    # preparation preserves the fraction and builds its ice column.
    "run.opt_thcnd", "run.fractional_seaice",
    # real.exe's seaice_threshold rule is absent-tolerant at its default
    # 100 K, which converts no water cell. A stated value still binds: soil
    # preparation turns open water colder than it into sea ice
    # (woof/ingest/soil.py preprocess_noah_soil).
    "run.seaice_threshold",
    # Noah-MP (ab60dc8ab):
    "run.dveg", "run.opt_crs", "run.opt_btr", "run.opt_run", "run.opt_sfc",
    "run.opt_frz", "run.opt_inf", "run.opt_rad", "run.opt_alb",
    "run.opt_snf", "run.opt_tbot", "run.opt_stc", "run.opt_gla",
    "run.opt_rsf", "run.opt_soil", "run.opt_pedo", "run.opt_crop",
    "run.opt_irr", "run.opt_irrm", "run.opt_infdv", "run.opt_tdrn",
    "run.soiltstep", "run.noahmp_output", "run.noahmp_acc_dt",
    # RUC (6153ea505); flag_sm_adj is documented as not applied by
    # ingest/ruc_soil.py. Disabled stochastic consumers preserve old inputs:
    "run.flag_sm_adj", "run.mosaic_lu", "run.mosaic_soil", "run.spp_lsm",
    "run.spp_conv", "run.spp_pbl",
    # Radiation (61903f124, 986db3bf2, 33694a95b, 8c0211eb0); rdmaxalb
    # is read at RESTORE by initialize_prepared_physics from the cached
    # static SNOALB, never at write:
    "run.wrf_rrtmg_compatibility", "run.ra_rrtmg_variant", "run.o3input",
    "run.use_mp_re", "run.seaice_albedo_default", "run.rdmaxalb",
    "run.surface_radiation_policy",
    # Turbulence and LES (46703df7d, 07f9daea5, 02cfd5301, 33694a95b,
    # 78c60d6b6):
    "run.c_k", "run.mix_isotropic", "run.mix_upper_bound",
    "run.tke_heat_flux", "run.tke_drag_coefficient", "run.tke_upper_bound",
    "run.isfflx", "run.moist_mix6_off",
    # SASE (604de61e3, e11e2eee2, 1a0e8a7f8); the closure selectors moved
    # to PREPARATION_INERT_RUN_FIELDS below:
    "run.km_opt_zero_acknowledgement",
    # (run.tke_budget, run.sase_flux_diag and run.hmix_k_diag are
    # output-only and are dropped at any value by
    # INERT_DIAGNOSTIC_IDENTITY_FIELDS below.)
    # Grell-Freitas (ea0cada3b): clos_choice and ishallow moved to
    # PREPARATION_INERT_RUN_FIELDS below.
    # NSSL selectors (76d96d550): mp=18 predates them, and DomainState
    # allocates the mp=18 species tuple and its CCN fill without
    # consulting them; only the scheme reads them:
    "run.nssl_2moment_on", "run.nssl_hail_on", "run.nssl_ccn_on",
    "run.nssl_density_on", "run.nssl_3moment",
    # Nest-birth policy (10c0e426f); prepared_single_domain_forecast
    # already carries a COMPATIBLE_LEGACY_DEFAULT override for it with
    # model_state_or_physics_changed False:
    "run.nest_microphysics_transition",
    # WRF zadvect_implicit (A158), argument (c): it selects how the
    # FORECAST's last RK substep advects in the vertical (woof/core/
    # ieva.py) and nothing in preparation reads it, so a header written
    # before the field describes exactly the explicit default.  A tree
    # prepared with it on carries 1 and is compared strictly.
    "run.zadvect_implicit",
    # WRF w_crit_cfl (A165), argument (c): only the FORECAST's w_damp
    # reads it, so a header written before the field describes its 1.0.
    "run.w_crit_cfl",
    # The adaptive-timestep surface (woof/core/adaptive_timestep.py,
    # WRF Registry.EM_COMMON:2269-2281).  TWELVE fields joined RunConfig
    # at once, and every one is here in the same commit that added them --
    # this table's history is four separate instances of a field arriving
    # without its entry and refusing every tree already in the field, so a
    # block this size arriving unregistered would have been the fifth and
    # by far the worst.
    #
    # Tolerable on the strongest form of the argument, and it is worth
    # stating precisely because "twelve at once" invites a shrug: with
    # `use_adaptive_time_step = False` -- the default, and the state every
    # header written before this block describes -- the controller never
    # runs, so NONE of the other eleven is read by anything.  They cannot
    # have influenced a prepared initial state or a boundary table,
    # because at preparation time no code consulted them at all.  A tree
    # that really does turn the feature on carries the values in its
    # header and is compared strictly, as it must be: the cache was
    # prepared for a different clock.
    # The FLAG alone stays here.  Its eleven companions moved to
    # PREPARATION_INERT_RUN_FIELDS, which is the stronger of the two
    # rulings and the correct one for them: tolerance forgives a field
    # ABSENT from an older header at its default, and the controller's
    # targets and clamps have to be forgiven at ANY value, in both
    # directions, or a cache prepared under one target refuses the run
    # that needs another.  The flag is not inert -- a bundle prepared
    # with the feature on describes a tree built for a different clock --
    # so it keeps the tolerant ruling it was given here.
    "run.use_adaptive_time_step",
    # Per-domain vertical ladder.  Tolerable on argument (b), in its
    # strongest form: the field's default is None, None means "inherit the
    # ladder the source carries", and inheriting the source's ladder is
    # exactly and only what preparation did before the field existed -- the
    # offline child read the parent's ZNW verbatim and the real-input child
    # read the experiment's one shared eta grid.  So "absent" and "default"
    # are the same prepared state, by construction rather than by
    # measurement.
    #
    # A tree that really does carry its own ladder is NOT forgiven: it holds
    # the ladder in its header and this table's narrowness compares it
    # strictly, which is required -- a prepared initial state and its whole
    # boundary table set are built ON the ladder, so a cache prepared for a
    # 49-level child cannot serve a 96-level one.
    "run.eta_levels",
    # The chem door, on argument (b): empty (the default) means no chem
    # species, which is exactly what every prepared state before the field
    # carried.  A chem tree holds its sets (and its enabled sources) in its
    # header and is compared strictly: its prepared state carries chem_<row>
    # fields and chem boundary tables a chem-off cache does not.
    "run.chem_sets", "run.chem_sources",
    # The downscaled child's relaxation time scale and w treatment, on
    # argument (c): both choose how the FORECAST applies the boundary
    # tables, and neither is read while the tables or the initial state
    # are built -- the tables always carry w.  Registered in the commit
    # that added them, so no tree prepared before them is refused.
    "run.relax_timescale_s", "run.relax_w",
    # WRF's horizontal diffusion form and its full-field logical
    # (lane/282-namelist-tolerance, 679a5f5fe), on argument (c): only the
    # FORECAST reads them (woof/core/dycore.py's mixing branch, the
    # diff_opt = 1 thermal reference it captures at the first step, and
    # the restart that carries it); nothing under woof/ingest, the
    # static builders or DomainState's allocation names either.  Their
    # defaults, 2 and true, are the metric full-field mixing every build
    # before the fields ran, so a header written before them describes
    # exactly the default.  A tree prepared for coordinate-surface
    # diffusion carries diff_opt = 1 and is compared strictly.
    "run.diff_opt", "run.mix_full_fields",
})

#: RunConfig fields that joined after the identity header and for which
#: an older header's silence is a REAL difference: the field shaped the
#: prepared initial state or the boundary tables from the day it existed,
#: and its default does not reproduce what preparation did before it, so
#: a tree written without it holds a different prepared state and must be
#: refused by name.  ``"run.<field>": one line saying what it changes``.
#:
#: Empty today.  Every field measured against the baseline in
#: tests/test_prepared_cache.py was either never read by preparation,
#: reads the same at its default as before it existed, or belongs to a
#: scheme that arrived with it (the three arguments above).  The table
#: exists so the next field that is none of those has a place to be
#: declared rather than a place to be forgotten: the guard test refuses a
#: RunConfig field that is in neither table, and this is the only other
#: answer it accepts.  A strict entry is not consulted by the comparison
#: -- absence already refuses -- it is the written reason the refusal is
#: right.
STRICT_IDENTITY_FIELDS: Mapping[str, str] = MappingProxyType({})


def _run_config_defaults(*names: str) -> dict[str, object]:
    """``{"run.<name>": RunConfig's default}`` for each name.

    Imported inside the call rather than at module scope: this module is
    part of the ingest path and woof.config pulls in a good deal of the
    package, so a top-level import here would tighten the import graph for
    a lookup that happens once per prepared-cache comparison.
    """
    import dataclasses

    from woof.config import RunConfig

    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    out: dict[str, object] = {}
    for name in names:
        field = fields.get(name)
        if field is None:
            raise AttributeError(
                f"RunConfig has no field {name!r}, so the tolerance for "
                f"run.{name} cannot state what its not-in-use value is. "
                f"Either the field was renamed -- update "
                f"DEFAULT_TOLERANT_IDENTITY_FIELDS with it -- or the "
                f"tolerance is stale and should be removed.")
        if field.default is dataclasses.MISSING:
            raise ValueError(
                f"RunConfig.{name} has no default, so 'absent means "
                f"default' is not a statement that can be made about it.")
        # The identity document is a JSON copy (tuples arrive as lists),
        # so the default is compared in that spelling or a tuple-valued
        # field could never match its own default.
        out[f"run.{name}"] = _json_copy(field.default)
    return out


def _domain_config_defaults(*names: str) -> dict[str, object]:
    """``{"<name>": DomainConfig's default}`` for each top-level name.

    The DomainConfig twin of :func:`_run_config_defaults`, with the same
    two refusals: a renamed field and a field with no default cannot say
    what "not in use" means.
    """
    import dataclasses

    from woof.experiment import DomainConfig

    fields = {f.name: f for f in dataclasses.fields(DomainConfig)}
    out: dict[str, object] = {}
    for name in names:
        field = fields.get(name)
        if field is None:
            raise AttributeError(
                f"DomainConfig has no field {name!r}, so the tolerance for "
                f"{name} cannot state what its not-in-use value is.")
        if field.default is dataclasses.MISSING:
            raise ValueError(
                f"DomainConfig.{name} has no default, so 'absent means "
                f"default' is not a statement that can be made about it.")
        out[name] = _json_copy(field.default)
    return out


def undelayed_identity_defaults(experiment) -> dict[str, object]:
    """What each tolerable field holds when its feature is NOT in use.

    ``start_time`` is the only member today, and its not-in-use value is
    not a constant: the loader resolves every domain's start, so a
    domain with no delayed start carries the EXPERIMENT's start rather
    than ``None``.  That is exactly the state a header written before
    delayed starts existed describes, and it is what makes tolerating
    the field's absence a statement about semantics rather than a
    shrug.  A domain that really does start late holds a different
    value, and is refused.
    """

    start = getattr(experiment, "start_time", None)
    return {"start_time": start.isoformat()
            if isinstance(start, datetime) else start,
            # A domain with no spawn declaration carries None, which is
            # the state every pre-spawn header describes.
            "spawn": None,
            # Same shape for the five fields that joined DomainConfig
            # after spawn (see DEFAULT_TOLERANT_IDENTITY_FIELDS): each is
            # an optional declaration whose off state is None.
            "tiles": None, "output": None,
            "retire": None, "rearm": None, "follow": None,
            "follow.reach_speed_m_s": None,
            # The history window: DomainConfig's own defaults, read from
            # the dataclass rather than restated (the reason is below).
            **_domain_config_defaults("history_begin_s", "history_end_s"),
            # READ FROM THE DATACLASS, not restated: two
            # hand-maintained copies of one default is the failure
            # this package has paid for repeatedly.  The NAMES come
            # from the table too: a run.* entry listed there but not
            # here would be dead, because the walk tolerates only a
            # path it finds in this map.
            **_run_config_defaults(*sorted(
                path[len("run."):]
                for path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                if path.startswith("run.")))}


#: Identity fields that describe when the run WRITES, not what it
#: integrates.  woof already publishes this partition twice, for the two
#: other places the same question is asked:
#: :data:`woof.io.restart.CONFIG_RUN_LENGTH_FIELDS` (which RunConfig
#: fields may differ between a checkpoint's writer and its resumer) and
#: :func:`woof.core.model.experiment_fingerprint` (which fields are
#: excluded from the trajectory hash).  A prepared cache is the same
#: question one stage earlier -- it holds an initial state and its
#: boundaries, and preparation writes no history or restart frame at all
#: -- so it asks the same table instead of keeping a third opinion.
#:
#: Binding these here is what refused a two-domain HRRR tree whose d02
#: history cadence was 900 s beside d01's 3600 s, and a hierarchy whose
#: namelist carried ``restart_interval = 60``: both are output cadences,
#: neither changes one model step.
#:
#: ``run.run_seconds`` belongs to the same partition and was missing
#: from it: a prepared cache is an initial state plus the boundary
#: tables for the times it decoded, and the forecast LENGTH changes
#: neither.  Its omission is what made "extend a run from its
#: checkpoint" -- the worked example in FIRST-LIGHT section 7 -- refuse
#: on the prepared routes: the length moved this identity, so the cache
#: had to be re-prepared to run one hour longer.  Whether the prepared
#: forcing actually REACHES the requested length is a separate gate and
#: is unchanged; both prepared runners already refuse a run past their
#: prepared coverage (``prepared_single_domain_forecast`` at its
#: cadence/coverage check, ``prepared_domain_tree_forecast`` at
#: "experiment duration exceeds prepared forcing hours"), so tolerating
#: the field here widens nothing.
#: The DOMAIN-level ``history_interval_s`` is the same knob as
#: ``run.output_interval_s`` under its other spelling, and this document
#: carries both.  Dropping only the ``run.`` one left the cadence bound
#: after all: a tree restarted with a changed history cadence -- one of
#: the three changes ``--restart`` publishes as permitted -- was refused
#: naming a field that says nothing about the prepared state.
NON_TRAJECTORY_IDENTITY_FIELDS = frozenset({
    # Tile layout/budgets choose the forecast execution road. Preparation
    # never reads them or changes meteorological arrays for them; the raw
    # document still records them, just as restart identity retains its
    # existing resident/streamed equivalence contract.
    "tiles",
    "run.run_seconds",
    "run.output_interval_s",
    "run.history_interval2_s", "run.history_interval_change_s",
    "run.restart_interval_s",
    "history_interval_s",
    # WRF's history_begin / history_end: when the forecast writes, like
    # the cadence beside them.
    "history_begin_s", "history_end_s",
    # Which variables reach the history tape, per domain, and the
    # quick-look stream's cadence and subset: output-only on the restart
    # identity's own terms (woof.core.model drops all three).  Binding
    # them refused a prepared tree whose d03 asked for a quick-look stream
    # ("these identity fields differ: full_history_interval_s,
    # quicklook_output"), and a per-domain `output = {...}` subset on a
    # bundle prepared without one (WOOF run-controls feasibility, defect 7),
    # although preparation reads none of them.
    "output", "full_history_interval_s", "quicklook_output",
})

#: Output-only diagnostic toggles, on the same footing as the cadences
#: above and for the same published reason: each selects which
#: diagnostic buffers a FORECAST carries, and a prepared cache predates
#: every one of them.  DERIVED from
#: :data:`woof.checkpoint_identity.CONFIG_DIAGNOSTIC_FIELDS`, the one
#: table restart already reads, rather than listed a second time: the
#: second list held only ``nwp_diagnostics``, so switching tke_budget,
#: sase_flux_diag or hmix_k_diag on or off refused an unchanged cache.
INERT_DIAGNOSTIC_IDENTITY_FIELDS = frozenset(
    f"run.{name}" for name in CONFIG_DIAGNOSTIC_FIELDS)

#: Identity fields PREPARATION never reads: they change the trajectory,
#: not the prepared state.  A prepared cache is an initial state plus
#: the EXTERNAL boundary stream for the times it decoded
#: (``run.run_seconds``'s own rationale above).  The P3 inflow-seeding
#: keys act at runtime FORCE on the child-owned rolling NEST boundary
#: tables, which preparation never computes -- so a tree prepared
#: before the field existed, or prepared with any value of it, holds
#: exactly the prepared state every value of it runs from.  Unlike the
#: two tables above these fields are trajectory-RELEVANT and stay
#: INSIDE :func:`woof.core.model.experiment_fingerprint` and the
#: restart identity; only the prepared-cache comparison drops them.
#: Entries are added deliberately, one per field, by whoever adds the
#: field, per this module's schema-growth protocol.  Provenance:
#: controller ruling 2026-08-03 under ArWen's standing delegation,
#: recorded in docs/superpowers/receipts/les/
#: INFLOW-GENERATOR-ACCEPTANCE-V2.md item 10.

# Explicit SFIRE v4.7.1 controls. Fire state and smoke emissions are initialized
# by forecast physics after atmospheric cache restore. Native atmospheric
# initialization and boundary preparation read none of these fields. This
# closed list permits inactive prior headers and fire on/off cache reuse;
# every control stays in forecast and restart identity.
SFIRE_PREPARATION_INERT_RUN_FIELDS = frozenset({
    "run.nfmc", "run.fmoist_run", "run.fmoist_interp", "run.fmoist_only",
    "run.fmoist_freq", "run.fmoist_dt", "run.fmep_decay_tlag", "run.ifire",
    "run.fire_boundary_guard", "run.fire_num_ignitions", "run.fire_ignition_ros1", "run.fire_ignition_start_lon1",
    "run.fire_ignition_start_lat1", "run.fire_ignition_end_lon1", "run.fire_ignition_end_lat1", "run.fire_ignition_radius1",
    "run.fire_ignition_start_time1", "run.fire_ignition_end_time1", "run.fire_ignition_ros2", "run.fire_ignition_start_lon2",
    "run.fire_ignition_start_lat2", "run.fire_ignition_end_lon2", "run.fire_ignition_end_lat2", "run.fire_ignition_radius2",
    "run.fire_ignition_start_time2", "run.fire_ignition_end_time2", "run.fire_ignition_ros3", "run.fire_ignition_start_lon3",
    "run.fire_ignition_start_lat3", "run.fire_ignition_end_lon3", "run.fire_ignition_end_lat3", "run.fire_ignition_radius3",
    "run.fire_ignition_start_time3", "run.fire_ignition_end_time3", "run.fire_ignition_ros4", "run.fire_ignition_start_lon4",
    "run.fire_ignition_start_lat4", "run.fire_ignition_end_lon4", "run.fire_ignition_end_lat4", "run.fire_ignition_radius4",
    "run.fire_ignition_start_time4", "run.fire_ignition_end_time4", "run.fire_ignition_ros5", "run.fire_ignition_start_lon5",
    "run.fire_ignition_start_lat5", "run.fire_ignition_end_lon5", "run.fire_ignition_end_lat5", "run.fire_ignition_radius5",
    "run.fire_ignition_start_time5", "run.fire_ignition_end_time5", "run.fire_ignition_start_x1", "run.fire_ignition_start_y1",
    "run.fire_ignition_end_x1", "run.fire_ignition_end_y1", "run.fire_ignition_start_x2", "run.fire_ignition_start_y2",
    "run.fire_ignition_end_x2", "run.fire_ignition_end_y2", "run.fire_ignition_start_x3", "run.fire_ignition_start_y3",
    "run.fire_ignition_end_x3", "run.fire_ignition_end_y3", "run.fire_ignition_start_x4", "run.fire_ignition_start_y4",
    "run.fire_ignition_end_x4", "run.fire_ignition_end_y4", "run.fire_ignition_start_x5", "run.fire_ignition_start_y5",
    "run.fire_ignition_end_x5", "run.fire_ignition_end_y5", "run.fire_lat_init", "run.fire_lon_init",
    "run.fire_ign_time", "run.fire_shape", "run.fire_sprd_mdl", "run.fire_crwn_hgt",
    "run.fire_ext_grnd", "run.fire_ext_crwn", "run.fire_sfc_flx", "run.fire_heat_peak",
    "run.fire_tg_ub", "run.fire_smk_scheme", "run.fire_smk_peak", "run.fire_smk_ext",
    "run.fire_wind_height", "run.fire_fuel_read", "run.fire_fuel_cat", "run.fire_fmc_read",
    "run.fire_print_msg", "run.fire_print_file", "run.fire_fuel_left_method", "run.fire_fuel_left_irl",
    "run.fire_fuel_left_jrl", "run.fire_grows_only", "run.fire_upwinding", "run.fire_upwind_split",
    "run.fire_viscosity", "run.fire_lfn_ext_up", "run.fire_topo_from_atm", "run.fire_advection",
    "run.fire_test_steps", "run.fire_const_time", "run.fire_const_grnhfx", "run.fire_const_grnqfx",
    "run.fire_atm_feedback", "run.fire_mountain_type", "run.fire_mountain_height", "run.fire_mountain_start_x",
    "run.fire_mountain_start_y", "run.fire_mountain_end_x", "run.fire_mountain_end_y", "run.delt_perturbation",
    "run.xrad_perturbation", "run.yrad_perturbation", "run.zrad_perturbation", "run.hght_perturbation",
    "run.stretch_grd", "run.stretch_hyp", "run.z_grd_scale", "run.sfc_full_init",
    "run.sfc_lu_index", "run.sfc_tsk", "run.sfc_tmn", "run.fire_read_lu",
    "run.fire_read_tsk", "run.fire_read_tmn", "run.fire_read_atm_ht", "run.fire_read_fire_ht",
    "run.fire_read_atm_grad", "run.fire_read_fire_grad", "run.sfc_vegfra", "run.sfc_canwat",
    "run.sfc_ivgtyp", "run.sfc_isltyp", "run.fire_lsm_reinit", "run.fire_lsm_reinit_iter",
    "run.fire_upwinding_reinit", "run.fire_is_real_perim", "run.fire_lsm_band_ngp", "run.fire_lsm_zcoupling",
    "run.fire_lsm_zcoupling_ref", "run.fire_tracer_smoke", "run.fire_viscosity_bg", "run.fire_viscosity_band",
    "run.fire_viscosity_ngp", "run.fire_slope_factor", "run.fs_array_maxsize", "run.fs_firebrand_gen_lim",
    "run.fs_firebrand_gen_dt", "run.fs_firebrand_gen_levels", "run.fs_firebrand_gen_maxhgt", "run.fs_firebrand_gen_levrand",
    "run.fs_firebrand_gen_levrand_seed", "run.fs_firebrand_gen_mom3d_dt", "run.fs_firebrand_gen_prop_diam", "run.fs_firebrand_gen_prop_effd",
    "run.fs_firebrand_gen_prop_temp", "run.fs_firebrand_gen_prop_tvel", "run.fs_firebrand_dens", "run.fs_firebrand_dens_char",
    "run.fs_firebrand_max_life_dt", "run.fs_firebrand_land_hgt", "run.fuel_crosswalk", "run.trackember",
    "run.sr_x", "run.sr_y", "run.fire_static", "run.fire_fuel_namelist",
    "run.fire_smoke",
})

# The earlier inactive fire configuration serialized an unused moisture
# filename. Only its empty value is equivalent to the current absent key.
RETIRED_INACTIVE_IDENTITY_FIELDS = {"run.fire_moisture_namelist": ""}

PREPARATION_INERT_RUN_FIELDS = SFIRE_PREPARATION_INERT_RUN_FIELDS | frozenset({
    # Turbine forcing attaches only in forecast initialize_physics; real
    # initialization and boundary preparation never read these controls.
    # Runtime overrides retain the original config bytes and record changes.
    "run.wind_farm_scheme", "run.wind_farm_table", "run.wind_farm_max_dx_m", "run.wind_farms", "run.windfarm_tke_factor",
    # The per-run smoke-off sentence (woof/smoke_execution.py): launch
    # metadata for the receipts; chem_sets carries the physics it
    # describes, so one prepared bundle serves either value.  It rides the
    # HRRR route as a forecast-TOML field (hrrr_route_inputs): a run whose
    # card could not hold the smoke envelope was refused at emission for it,
    # and its native tree authority did not reload (a 4090 host and a rented box, 2026-10-21).
    "run.smoke_off_reason",
    # Eclipse attenuation is evaluated at forecast UTC; preparation never reads it.
    "run.ra_sw_eclipse",
    # The zero-layer sea ice model (woof/ocean/seaice) is seeded by the
    # forecast's initialize_physics from the restored surface state; nothing
    # under woof/ingest names it, so one prepared bundle serves either value.
    # It stays in the restart identity (checkpoint_identity.SEAICE_RUN_KEYS).
    "run.seaice_prognostic",
    # Stability policy is read only when the forecast advances: epssm by
    # core/acoustic.py and forecast-door acoustic_adaptation.py, and the
    # sixth-order selectors by core/dycore.py. initialize_real constructs
    # the hydrostatic analysis and boundaries without an acoustic or
    # diffusion step (ingest/real.py). One verified preparation therefore
    # serves these policies. They remain strict restart/trajectory identity.
    "run.epssm", "run.diff_6th_opt", "run.diff_6th_factor",
    "run.diff_6th_slopeopt",
    # Prescribed LAI and background albedo are selected at forecast
    # physics initialization from the cached monthly static fields;
    # preparation writes those fields under either selector.
    "run.rdlai2d", "run.usemonalb",
    # Only the forecast's final RK stage reads this numerical generation
    # (core/ieva.py). Neither initial fields nor boundary tables depend on
    # it, so one prepared bundle serves both variants. Checkpoints still
    # bind the variant because switching it changes their trajectory.
    "run.zadvect_implicit_variant",
    # Only LSMRUC's SOILPROP reads the soil-water lineage
    # (core/ruc_tier.py); preparation computes no diffusivity from it.
    "run.ruc_soilprop",
    # Physics initialization and stepping read the Thompson generation;
    # prepared initial and boundary fields do not depend on it. The fork's
    # start emission is recomputed after the prepared state is loaded.
    "run.thompson_version",
    "run.thompson_fork_snow_fall",
    # The generation's table directory is read by the forecast's
    # Thompson column only (woof/core/thompson_column.py); preparation
    # computes nothing from the tables.
    "run.thompson_table_dir",
    # Only the MYNN column solver reads its generation and the gsd_41 TKE
    # option; preparation computes nothing from either.
    "run.bl_mynn_version",
    "run.bl_mynn_gsd41_unsquared_qtke",
    "run.bl_mynn_cloud_tendency_form",
    # cycling decides only what the first MYNN call keeps from the loaded
    # start; preparation computes nothing from it.
    "run.cycling",
    # Only the MYNN surface-layer column solver reads its generation;
    # preparation computes no surface exchange from it.
    "run.mynn_sfclay_variant",
    # MYNN's split number-species mixing keys select which of the PBL
    # solver's rates are coupled; preparation computes no PBL.
    "run.bl_mynn_mixaerosols",
    "run.bl_mynn_mixnumcon",
    # TEMPO's optional coupling arms act inside the forecast's physics
    # calls only; prepared fields and boundary tables do not read them.
    "run.tempo_ml_for_bl_nc",
    "run.tempo_surface_emission_before",
    # The sixth-order filter form and its second factor are read only by
    # the forecast's fixed tendencies (core/dycore.py), and mp_zero_out
    # only after the forecast's microphysics call (core/microphysics.py);
    # preparation computes neither.
    "run.diff_6th_form",
    "run.upper_wind_limiter_form",
    "run.diff_6th_factor2",
    "run.mp_zero_out",
    "run.mp_zero_out_thresh",
    "run.mp_zero_out_all",
    # Only LSMRUC's post-SFCTMP step reads the irrigation rule
    # (core/ruc_mosaic.py); preparation computes no soil water from it.
    "run.ruc_irrigation",
    # Only LSMRUC's first call reads the QVG cold start (core/ruc.py,
    # the fused prologue); preparation computes no surface humidity from it.
    "run.ruc_qvg_cold_start",
    # Only SFCDIAGS_RUCLSM reads the 2 m form (the fused epilogue,
    # core/ruc_runtime.py); preparation computes no 2 m value from it.
    "run.ruc_2m_diagnostic",
    # Only LSMRUC and SFCTMP read the snow lineage (core/ruc_tier.py, the
    # fused prologue and sfctmp); preparation computes no snow from it.
    "run.ruc_snow",
    # The WRF 4.7/4.8 upstream-fix generations and the urban canopy
    # controls are read by the forecast's physics alone: the solar
    # generation by the radiation/LSM COSZEN calls (core/solar.py), the YSU
    # generation by the PBL kernel selection (woof/ysu_generations.py), the
    # Noah-MP snow-age generation by the Noah-MP radiation/glacier units,
    # and the urban water/canopy generations, tree and vegetation controls
    # and table directory by the SLUCM/BEP column (core/urban_*); nothing
    # under woof/ingest reads any of them, so one prepared bundle serves
    # every value.  Checkpoints still bind them (io/restart.py).
    "run.solar_generation",
    "run.ysu_version",
    "run.noahmp_snow_age",
    "run.urban_water_balance",
    "run.urban_ucm_generation",
    "run.urban_treeoption",
    "run.urban_ttscheme",
    "run.urban_fvg",
    "run.urban_table_dir",
    # The vertical advection ladder and the (declaration-only) horizontal
    # momentum order are read by the flux-divergence launchers alone; no
    # initial field or boundary table depends on them, so one prepared
    # bundle serves vert_order 3 and 5.  Checkpoints still bind them.
    "run.v_sca_adv_order",
    "run.v_mom_adv_order",
    "run.h_mom_adv_order",
    "run.inflow_perturbation",
    "run.inflow_perturbation_seed",
    "run.inflow_perturbation_amplitude_scale",
    "run.inflow_perturbation_faces",
    # THE ADAPTIVE CONTROLLER'S TARGETS AND CLAMPS, on the same argument
    # one step stronger than the tolerant table could carry.  Tolerance
    # forgives a field absent from an older header AT ITS DEFAULT; these
    # have to be forgiven at any value and in both directions, because
    # the cache prepared under one target is byte-for-byte the cache
    # prepared under another.  Measured the other way: comparing them
    # here refused a cache that described exactly the state the live
    # configuration needed, so a run that died of its own controller
    # setting could not be re-prepared with the setting that would have
    # saved it.
    #
    # THE STANDING CHECK, if a field is proposed for this set: grep the
    # preparation path for a reader.  The only prepare-side module that
    # names target_cfl or min_time_step is this identity check itself.
    # If preparation reads it, it belongs in the identity instead.
    #
    # `use_adaptive_time_step` is deliberately NOT here.  It selects
    # which clock the tree was prepared for, it stays in the tolerant
    # table above, and the restart walk still refuses a resume that
    # flips it, because the carried controller state means nothing under
    # a fixed clock.
    "run.step_to_output_time", "run.adaptation_domain",
    "run.target_cfl", "run.target_hcfl", "run.max_step_increase_pct",
    "run.starting_time_step", "run.starting_time_step_den",
    "run.max_time_step", "run.max_time_step_den",
    "run.min_time_step", "run.min_time_step_den",
    # The adaptive clock's substep floor, on the same argument: only the
    # clock reads it, once per root step of the forecast, and the forecast
    # doors set it after the cache is read.
    "run.min_time_step_sound",
    # Acoustic off-centering is used by the forecast dycore only. Refusing
    # it here prevented a prepared steep-terrain nest from adopting the
    # measured stable value without rebuilding identical initial arrays.
    # It remains in experiment and restart identity because it changes
    # the forecast trajectory.
    "run.epssm",
    # Only the forecast clock reads this; prepared arrays do not change.
    "run.adaptive_nest_lattice",
    # WRF's slope_rad / topo_shading / shadlen: read only by the forecast's
    # radiation and surface calls (woof.core.topo_radiation), from the
    # prepared terrain; no prepared array depends on them.
    "run.slope_rad", "run.topo_shading", "run.shadlen",
    # THE CHEM OPERATOR'S SWITCHES (smoke / GOCART-lite / CAMS program,
    # woof/config.py after adaptive_nest_lattice).  Each is read by the
    # forecast's chem step or chem transport and by nothing that builds a
    # prepared array: which species a prepared state carries is decided by
    # run.chem_sets (tolerant table above) and the chem table's own
    # identity (prepared_domain_config_identity), never by these.
    "run.chem_adv_opt", "run.chem_mix2_off", "run.chem_mix6_off",
    "run.chemdt", "run.kemit", "run.biomass_burn_opt",
    "run.plumerisefire_frq", "run.dust_opt", "run.seas_opt",
    "run.dmsemis_opt", "run.wetscav_onoff", "run.chem_conv_tr",
    "run.vertmix_onoff", "run.aer_ra_feedback", "run.aer_op_opt",
    "run.dust_alpha", "run.dust_gamma", "run.dust_smtune", "run.dust_ustune",
    "run.mynn_chem_vertmx", "run.fire_emission_mode",
    "run.plume_fire_properties", "run.aerosol_mp_coupling",
    # simple_dir_fdb: read by the radiation factory (transported smoke into
    # the aer_opt = 3 shortwave), by nothing that builds a prepared array.
    "run.simple_dir_fdb", "run.enh_vermix", "run.flam_part",
    "run.aer_drydep_opt",
    # THE SASE CLOSURE'S SELECTORS, out of the tolerant table above for
    # the same reason as the adaptive targets: they have to be forgiven at
    # any value.  No WRF namelist spells them, so an HRRR hierarchy
    # prepares every domain from the namelist with their defaults, while
    # the forecast reads them from the experiment config: a SASE tree
    # with [shared] sase_moist_n2 = false was refused after preparation
    # for a field whose value cannot move a prepared array.  The standing
    # check holds: the only prepare-side module that names any of them
    # is this identity table.  sase_flux_diag is not among them: it is an
    # output-only switch, dropped from the restart identity as well, so
    # its one ruling is INERT_DIAGNOSTIC_IDENTITY_FIELDS above.
    "run.sase_moist_n2",
    "run.sase_stable_dissipation", "run.sase_additive_dissipation",
    # THE GRELL-FREITAS CLOSURE AND SHALLOW-ARM SELECTORS, on the same
    # argument.  Only the forecast's cumulus call reads them
    # (woof/core/gf.py packs both into the scheme's integer inputs); the
    # prepared initial state and the boundary tables are the same under
    # every value.  An HRRR hierarchy prepares its domains from WRF
    # namelists, and a tree's cumulus-off child carries 0 for both beside
    # a Grell-Freitas root that runs clos_choice = 1, so binding them
    # refused a prepared domain for a key preparation never read.  They
    # stay in the experiment fingerprint and the restart identity.
    "run.clos_choice", "run.ishallow", "run.cugd_avedx",
    # NOAH MOSAIC AND WHERE ITS URBAN CANOPY RUNS, on the same argument.
    # Preparation writes LANDUSEF whatever these say; the forecast's
    # initialization doors build the tiles from it after the cache is read
    # (woof/core/noah_mosaic_door.py) and refuse by name a cache without
    # it, and the canopy rule only fills FRC_URB2D there.  The standing
    # check holds: no prepare-side module reads any of the three.  A cache
    # prepared once serves mosaic off, WRF's dominant-urban rule and the
    # town rule alike; they stay in the experiment fingerprint and the
    # restart identity when on.
    "run.sf_surface_mosaic", "run.mosaic_cat", "run.mosaic_urban_canopy",
    # THE TERRAIN-CLOCK MODE (lane/286-fixed-step-grid), on the adaptive
    # controller's argument.  Preparation never reads it: the terrain
    # clock and the substep rule run at the forecast door, after the cache
    # is read, and bind their derivation through the terrain_clock
    # receipt.  The standing check holds: no prepare-side module names
    # it.  So a tree prepared under "measured" (or before the field) is
    # byte for byte the tree a "pinned" run needs at the same dt and
    # time_step_sound; tolerance alone forgives only the default, and it
    # refused the pinned run for a field that cannot move a prepared
    # array.  It stays in the experiment fingerprint and the restart
    # identity when pinned.
    "run.terrain_clock",
    # WRF'S RADIATION-DRIVER OPTIONS swint_opt AND aer_opt (lane
    # 286-aer-swint), on the same argument.  Only the forecast reads them:
    # the per-step surface shortwave fit (woof/core/swint.py) and the
    # legacy RRTMG shortwave's aerosol optics, formed on each radiation
    # call from the live state (woof/core/rrtmg_aerosol_optics.py).  The
    # standing check holds: nothing under woof/ingest names either, and
    # the prepared initial state and boundary tables are the same under
    # every value.  They stay in the experiment fingerprint and the
    # restart identity when moved.
    "run.swint_opt", "run.aer_opt",
    # The aer_opt = 2 user-aerosol controls (woof/config.py
    # AER_OPT2_RUN_KEYS), on the same argument: only the legacy RRTMG
    # shortwave's aerosol optics read them, on each radiation call, and
    # nothing under woof/ingest names any of them, so the prepared initial
    # state and boundary tables are the same under every value.  They stay
    # in the experiment fingerprint and the restart identity.
    "run.aer_type", "run.aer_aod550_opt", "run.aer_angexp_opt",
    "run.aer_ssa_opt", "run.aer_asy_opt", "run.aer_aod550_val",
    "run.aer_angexp_val", "run.aer_ssa_val", "run.aer_asy_val",
    "run.aer_auxinput15", "run.aer_auxinput15_interval",
    # WRF's lightning diagnostic package (lightning_option and its controls,
    # cd75865eb6e): off by default, read only by the forecast's diagnostic
    # driver on its own cadence, and nothing under woof/ingest names any of
    # them, so the prepared initial state and boundary tables are the same
    # under every value.  They stay in the experiment fingerprint.
    "run.lightning_option", "run.lightning_dt", "run.lightning_start_seconds",
    "run.flashrate_factor", "run.iccg_method", "run.iccg_prescribed_num",
    "run.iccg_prescribed_den", "run.cellcount_method", "run.cldtop_adjustment",
    "run.do_radar_ref",
    # WRF's one-dimensional ocean mixed layer (sf_ocean_physics and its
    # namelist values, 27c2bab6456): off by default and read only by the
    # forecast's surface step from the live state; nothing under
    # woof/ingest names it, so preparation writes the same state under
    # every value.  It stays in the experiment fingerprint and the restart
    # identity.
    "run.sf_ocean_physics", "run.oml_hml0", "run.oml_gamma",
    "run.oml_relaxation_time",
    # Sun-angle albedo reads live solar geometry on radiation steps; it
    # changes no prepared state or boundary table.
    "run.alb_sol", "run.rrtmg_cloud_optics_form",
    # The aer_opt=3 aerosol type reads xland on radiation calls only.
    "run.rrtmg_aer3_type_source",
    # Prescribed smoke acts only in radiation after prepared-state restore.
    # Its full member identity remains in forecast and restart identities.
    "run.rrtmg_smoke_manifest",
    # WRF's moisture pressure correction (calc_cq; no WRF namelist key).
    # Only the forecast's acoustic and big-step drivers read it
    # (woof/core/acoustic.py prepare_moist_cq and the ensemble batch
    # twins); initialize_real builds the hydrostatic state and the
    # boundary tables without an acoustic step.  The standing check
    # holds: nothing under woof/ingest names it, and the prepare-side
    # mentions are the HRRR hierarchy's per-domain override list and the
    # nest-transition contract, which read the configuration, not a
    # prepared array.  An HRRR hierarchy prepares from WRF namelists,
    # which state the importer's True, so a verification counterfactual
    # moist_cq = false was refused on that route for a value no prepared
    # array depends on.  It stays in the experiment fingerprint and the
    # restart identity.
    "run.moist_cq",
    # Placement rain history begins at forecast initialization. Neither
    # prepared prognostics nor parent boundary tables depend on its policy.
    # Forecast/restart identity still binds the policy and its archive.
    "run.nest_accumulation_archive",
})


def effective_prepared_domain_config(document):
    """Canonicalize one prepared-domain identity for comparison.

    Two documents describe the same prepared d0N when they agree here.
    Three normalizations, each of which was a real refusal:

    * cadence, inert-diagnostic and preparation-inert fields (the three
      tables above) are dropped -- the first two say when the forecast
      writes rather than what it integrates, and the third names run
      mechanisms preparation never reads;
    * ``cudt_minutes`` is pinned to 0 when ``cu_physics == 0``, because a
      cumulus interval with no cumulus scheme is dead namelist state and
      the two producers spell the dead value differently (a WRF importer
      that omits the key inherits RunConfig's 5.0; a shipped physics
      profile states 0.0);
    * ``radt`` and ``radt_minutes`` both carry the cadence the domain
      runs (:func:`woof.config.effective_radt_minutes`): a positive
      ``radt`` overrides ``radt_minutes``, and ``radt = 0`` means "not
      stated here", so the domain runs ``radt_minutes``.  The legacy
      shadow field is not a second radiation cadence.  Resolving only
      the positive case left ``radt`` raw, and a child domain that never
      mentions radiation (``radt = 0``, twelve minutes) was reported as
      differing from the WRF namelist that states ``radt = 12`` for it,
      which is the same forecast.

    A non-document is returned unchanged so callers can hand this
    whatever the header gave them.
    """

    if not isinstance(document, Mapping):
        return document
    normalized = _plain(document)
    # Top-level (domain) entries first: the output cadence lives here as
    # well as under ``run``, and a document may carry it with no ``run``
    # section at all.
    for path in sorted(NON_TRAJECTORY_IDENTITY_FIELDS
                       | INERT_DIAGNOSTIC_IDENTITY_FIELDS
                       | PREPARATION_INERT_RUN_FIELDS):
        if "." not in path:
            normalized.pop(path, None)
    run = normalized.get("run")
    if not isinstance(run, dict):
        return normalized
    for path in sorted(NON_TRAJECTORY_IDENTITY_FIELDS
                       | INERT_DIAGNOSTIC_IDENTITY_FIELDS
                       | PREPARATION_INERT_RUN_FIELDS):
        section, _, key = path.partition(".")
        if section == "run" and key:
            run.pop(key, None)
    for path, inactive in RETIRED_INACTIVE_IDENTITY_FIELDS.items():
        _, _, key = path.partition(".")
        if key in run and run[key] == inactive:
            run.pop(key)
    if run.get("radt", 0.0) > 0.0:
        run["radt_minutes"] = run["radt"]
    if "radt" in run and "radt_minutes" in run:
        # Both fields now say the one cadence the integrators run, so a
        # difference in either is a difference in the forecast.
        run["radt"] = run["radt_minutes"]
    if run.get("cu_physics") == 0:
        run["cudt_minutes"] = 0.0
    # The historical radiation aggregate is a SPELLING, not a different
    # selection: woof/config.py documents ``ra_lw_physics = ra_sw_physics
    # = -1`` as preserving "the historical woof ra_physics aggregate
    # exactly", and explicit pairs require ``ra_physics = 0``.  The two
    # producers meet here spelled differently -- a shipped 4/4 profile
    # writes the explicit (0, 4, 4) triple while the WRF namelist importer
    # emits the coupled aggregate (4, -1, -1) -- so the aggregate is
    # canonicalized to its resolved explicit form for comparison.  Mixed
    # pairs have no aggregate spelling and pass through untouched.
    if (run.get("ra_lw_physics", -1) == -1
            and run.get("ra_sw_physics", -1) == -1
            and run.get("ra_physics", 0) in (0, 4, 90)):
        aggregate = run.get("ra_physics", 0)
        run["ra_lw_physics"] = aggregate
        run["ra_sw_physics"] = aggregate
        run["ra_physics"] = 0
    return normalized


#: Where the writing woof stamps its version in a cache header.  It
#: sits OUTSIDE the hashed ``basis`` on purpose: the content digest of
#: every cache written before this release must keep verifying exactly
#: as it did, and a stamp that changed the digest would be a second
#: upgrade break in the fix for the first one.
CACHE_WRITER_KEY = "writer"

#: What a header with no stamp tells us: it was written before stamping
#: existed.  Naming that is the accurate answer; guessing a version is not.
UNSTAMPED_WRITER = "a release before 1.1.1 (which stamped no version)"

#: Preparation receipts a preparer binds into the cache's user metadata
#: ONLY when the thing they record actually happened -- the SMCDRY
#: moisture floor, the deep-soil TMN repair.  They are LOUD-when-fired,
#: absent otherwise, and the identical receipt is bound into the
#: preparation proof.
#:
#: This tuple is the single source of truth for that key set, and it
#: exists because splitting it was a bug: the writer bound
#: ``soil_moisture_floor`` while the reader's ``expected_user`` did not
#: list it, and ``user != expected_user`` is an EXACT comparison -- so
#: every preparation whose floor fired wrote a cache that it then
#: refused to read back.  A key recorded on one side of that comparison
#: and missing from the other is not leniency, it is a refusal.  Anything
#: added here must be bound from the proof by the validator, so a receipt
#: that DIFFERS from the proof still refuses.
CONDITIONAL_PREPARATION_RECEIPTS = (
    "soil_moisture_floor",
    "deep_soil_repair",
)

#: Full soil-operation receipt inventory for layouts that bind the texture
#: treatment into BOTH cache user metadata and proof. Portable adapters that
#: keep texture treatment in the proof alone retain the conditional tuple.
#: ``soil_temperature_repair`` (real.exe's TSLB reasonableness rebuild and
#: the snow-covered rebuild beside it,
#: :func:`woof.ingest.soil.soil_temperature_repair_proof`) is bound the
#: same way on the native HRRR route, and only when a land column was
#: rebuilt; the mapped routes carry it in their proof alone.
SOIL_PREPARATION_RECEIPTS = (
    *CONDITIONAL_PREPARATION_RECEIPTS,
    "soil_texture_downscale",
    "soil_temperature_repair",
)


def cache_writer_version(header) -> str:
    """The woof that wrote this cache header, or an accurate unknown."""

    writer = header.get(CACHE_WRITER_KEY) if isinstance(header, Mapping) \
        else None
    version = writer.get("gpuwm_version") if isinstance(writer, Mapping) \
        else None
    return version if isinstance(version, str) and version else UNSTAMPED_WRITER


def compare_prepared_domain_config(cached, live, *, not_in_use=None
                                   ) -> tuple[list[str], list[str]]:
    """``(tolerated, differing)`` field paths between two domain identities.

    ``tolerated`` are fields the live document has, the cached one does
    not, and whose live value is the not-in-use value the caller
    declared for them -- provably the same prepared state under a newer
    schema.  ``differing`` is everything else, including a field the
    CACHE carries and this build does not, which means the cache was
    written by a newer woof than this one.

    ``not_in_use`` comes from :func:`undelayed_identity_defaults`.
    Omitting it tolerates nothing: a caller that cannot say what "off"
    means for a field is not in a position to decide the field is off.
    """

    not_in_use = {} if not_in_use is None else dict(not_in_use)
    if not isinstance(cached, Mapping) or not isinstance(live, Mapping):
        # Both absent is not a difference; only one of them being a
        # document is.  Synthetic identities without a domain_config at
        # all are a legitimate shape for callers that bind something
        # else entirely.
        return [], ([] if cached == live else ["domain_config"])
    tolerated: list[str] = []
    differing: list[str] = []

    def walk(cached_node, live_node, prefix: str) -> None:
        for key in sorted(set(cached_node) | set(live_node)):
            path = f"{prefix}{key}"
            if (path in PREPARATION_INERT_RUN_FIELDS
                    or path in INERT_DIAGNOSTIC_IDENTITY_FIELDS):
                # Not tolerance, and not a widening of it: preparation
                # reads none of these, so a cache written under one value
                # is byte-identical to one written under another and the
                # comparison has nothing to say about them.
                # `effective_prepared_domain_config` already drops them
                # for callers that normalise first; this covers the ones
                # that hand the walk a raw header (PreparedCacheReader).
                continue
            if key not in cached_node:
                if (path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                        and path in not_in_use
                        and live_node[key] == not_in_use[path]):
                    tolerated.append(path)
                else:
                    differing.append(path)
                continue
            if key not in live_node:
                if path in RETIRED_INACTIVE_IDENTITY_FIELDS and cached_node[key] == RETIRED_INACTIVE_IDENTITY_FIELDS[path]:
                    continue
                # The mirror of the absent-in-cache rule above: a tolerable
                # field the CACHE states at its not-in-use value and the
                # live document omits (an identity that drops a default)
                # describes the same prepared state.  A nested HRRR-recipe
                # tree prepared from the namelists states the 2.9.4 sea-ice
                # parameters at their defaults while the config's identity
                # omits them, and the tree forecast refused its own fresh
                # preparation (box D, 2026-10-21 21:16Z).
                if (path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                        and path in not_in_use
                        and cached_node[key] == not_in_use[path]):
                    tolerated.append(path)
                    continue
                differing.append(path)
                continue
            old, new = cached_node[key], live_node[key]
            if isinstance(old, Mapping) and isinstance(new, Mapping):
                walk(old, new, f"{path}.")
            elif old != new:
                differing.append(path)

    walk(cached, live, "")
    return tolerated, differing


def compare_prepared_identity(cached, expected, *, not_in_use=None
                              ) -> tuple[list[str], list[str]]:
    """``(tolerated, differing)`` between a cached and a live identity.

    Only ``domain_config`` is compared default-tolerantly: it is the
    document that grows fields as the configuration schema grows.  Every
    other member of the identity -- the source, static, namelist and
    bridge digests -- is a hash of bytes and stays strictly equal, so
    nothing about what the cache was built FROM is relaxed here.
    """

    if not isinstance(cached, Mapping) or not isinstance(expected, Mapping):
        return [], ["identity"]
    tolerated, differing = compare_prepared_domain_config(
        cached.get("domain_config"), expected.get("domain_config"),
        not_in_use=not_in_use)
    for key in sorted(set(cached) | set(expected)):
        if key == "domain_config":
            continue
        if (key not in cached or key not in expected
                or cached[key] != expected[key]):
            differing.append(key)
    return tolerated, differing


def prepared_identity_refusal(*, subject: str, header, differing,
                              re_prepare: str | None = None) -> str:
    """One sentence naming the versions, the fields, and the way out.

    "d01 cache domain config differs from experiment" was true and
    useless: it named the experiment file, which was innocent, and never
    mentioned that a package upgrade had changed the identity document.
    Whatever survives the default-tolerant comparison above is a real
    mismatch, and it says which fields and between which releases.
    """

    from woof import __version__

    named = sorted(differing)
    fields = ", ".join(named) or "(none named)"
    writer = cache_writer_version(header)
    if str(writer) == str(__version__):
        # SAME VERSION ON BOTH SIDES.  Naming it twice read as a version
        # mismatch that was not one, and sent the reader looking for a
        # package upgrade that had not happened: measured on a resume
        # that had only turned the adaptive clock on.  When the versions
        # agree, the difference can only be the configuration, and the
        # sentence says that instead of implying the opposite.
        sentence = (
            f"{subject} was prepared by this same woof {__version__}, so "
            f"the difference is a configuration change rather than a "
            f"package upgrade; these identity fields differ: {fields}")
    else:
        sentence = (
            f"{subject} was prepared by {writer} and "
            f"this is woof {__version__}; these identity fields differ: "
            f"{fields}")
    # THE CLOCK IS NOT A RUNTIME SWITCH, and this is the gate that
    # actually answers when someone tries to use it as one -- the
    # restart-level refusal never gets the chance, because the cache
    # identity is compared first on every route.  A tree is prepared
    # around the configured step: its alarms and its nest step ratios are
    # built from it, so turning the adaptive clock on or off asks for a
    # different tree, not a different setting.  Saying so here is the
    # difference between a two-minute re-prepare and a hunt for a version
    # that was never wrong.
    clock = ""
    if "run.use_adaptive_time_step" in named:
        clock = ("  Turning use_adaptive_time_step on or off is what was "
                 "refused: a prepared tree is built around the configured "
                 "step, so an adaptive run needs its own prepared tree.")
    if re_prepare:
        return f"{sentence}.{clock}  Re-prepare it with: {re_prepare}"
    return (f"{sentence}.{clock}  If that difference is a package upgrade "
            f"rather than a configuration change, re-prepare the bundle "
            f"with the front door that wrote it.")


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("prepared-cache arrays must have a numeric dtype")
    return np.ascontiguousarray(array)


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b";")
    digest.update(_canonical(list(array.shape)).encode("ascii"))
    digest.update(b";")
    digest.update(memoryview(array.reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def _prepared_met_names(met, *, surface=None) -> tuple[str, ...]:
    """Validate and return the exact mapped-field persistence contract."""
    try:
        available = set(met.fields)
    except AttributeError as exc:
        raise TypeError("prepared met input must expose a fields mapping") from exc
    met_required = _MET_REQUIRED | (
        _LEGACY_HRRR_SOIL if surface is None else frozenset())
    met_names = tuple(sorted((met_required | _MET_OPTIONAL) & available))
    missing_met = sorted(met_required - set(met_names))
    if missing_met:
        raise KeyError(
            f"prepared cache physics inputs are missing {missing_met}")
    return met_names


def select_prepared_met_fields(met, *, surface=None):
    """Detach only mapped fields needed by cache writing and physics setup.

    Native horizontal interpolation produces many full-domain 3-D fields, but
    after the f00 integration state and boundary snapshot have been built only
    a small surface/soil subset remains live.  Materialize that exact contract
    on the host so callers can release the source snapshot and its device
    allocations without changing cache or physics inputs.
    """
    names = _prepared_met_names(met, surface=surface)
    selected = {
        name: np.array(
            _host(met.fields[name]), copy=True, order="C", subok=False)
        for name in names
    }
    # Surface assembly is carried beside fields on HorizontalSnapshot. The
    # native preparation path solves Noah only after this detachment, so
    # dropping that completed field silently restores the raw skin fallback.
    water = {}
    for name in ("water_temperature", "water_temperature_source"):
        value = getattr(met, name, None)
        water[name] = (None if value is None else
                       np.array(_host(value), copy=True, order="C", subok=False))
    receipt = getattr(met, "water_temperature_receipt", None)
    water["water_temperature_receipt"] = (
        None if receipt is None else _json_copy(dict(receipt)))
    return SimpleNamespace(fields=MappingProxyType(selected), **water)


class _BundleWriter:
    def __init__(self, temporary: Path):
        self.temporary = temporary
        self.manifest: dict[str, dict[str, object]] = {}
        self.payload_bytes = 0
        # Set on the first link this drive refuses: every later payload is
        # copied rather than retried as a link.
        self.copying = False
        self._pending_reuse_bytes = 0
        self._batch = None

    @contextmanager
    def immutable_batch(self):
        """Batch a caller-owned interval in which none of its arrays mutate.

        Outside this explicit scope, add() still finishes its snapshot
        before returning. Native jobs finish before scope exit and before
        any header or boundary marker can be published.
        """
        from woof.ingest.prepared_writer import batch_budget, native_writer
        if self._batch is not None:
            raise RuntimeError("prepared-array batches cannot nest")
        entry = native_writer()
        if entry is None:
            yield
            return
        workers, budget = batch_budget()
        self._batch = {"entry": entry, "workers": workers, "budget": budget,
                       "bytes": 0, "jobs": []}
        try:
            yield
            self._flush_batch()
        finally:
            self._batch = None

    def _flush_batch(self):
        from woof.ingest.prepared_writer import write_arrays
        batch = self._batch
        jobs = batch["jobs"]
        if not jobs:
            return
        arrays = [(partial, array) for _key, _path, partial, array in jobs]
        hashes = write_arrays(batch["entry"], arrays, workers=batch["workers"])
        try:
            for (key, path, partial, array), digest in zip(jobs, hashes):
                _replace_file(partial, path)
                self.manifest[key] = {
                    "file": path.name, "shape": list(array.shape),
                    "dtype": str(array.dtype), "nbytes": int(array.nbytes),
                    "sha256": digest,
                }
                self.payload_bytes += int(array.nbytes)
        finally:
            # Successful native writes created these exact private paths.
            # Renamed ones no longer exist; an interrupted publication
            # removes only the remaining files owned by this batch.
            for _key, _path, partial, _array in jobs:
                partial.unlink(missing_ok=True)
            batch["jobs"] = []
            batch["bytes"] = 0

    def _add_batched(self, key, value):
        batch = self._batch
        if any(job[0] == key for job in batch["jobs"]):
            raise ValueError(f"invalid or duplicate prepared-cache key {key!r}")
        expected = getattr(value, "nbytes", None)
        if batch["jobs"] and (expected is None
                or batch["bytes"] + int(expected) > batch["budget"]
                or len(batch["jobs"]) >= batch["workers"]):
            self._flush_batch()
        array = _host(value)
        filename = f"a{len(self.manifest) + len(batch['jobs']):05d}.npy"
        path = self.temporary / filename
        partial = path.with_name(filename + ".tmp")
        batch["jobs"].append((key, path, partial, array))
        batch["bytes"] += int(array.nbytes)

    def expect_reuse(self, reader: "PreparedCacheReader", keys) -> None:
        """Declare the payloads :meth:`link_verified` is about to place.

        So a drive that cannot link is checked ONCE, for the whole
        remaining prefix, before the first byte is copied -- not found
        full on the last file.
        """
        self._pending_reuse_bytes += sum(
            (reader.path / reader.arrays[key]["file"]).stat().st_size
            for key in keys)

    def add(self, key: str, value) -> None:
        if not isinstance(key, str) or not key or key in self.manifest:
            raise ValueError(f"invalid or duplicate prepared-cache key {key!r}")
        if self._batch is not None:
            self._add_batched(key, value)
            return
        array = _host(value)
        # Cache payload names are deliberately compact.  Prepared caches sit
        # below several transaction-owned hierarchy staging directories, so a
        # descriptive private filename can exhaust the legacy Windows path
        # budget even when the final public cache path itself is valid.
        filename = f"a{len(self.manifest):05d}.npy"
        path = self.temporary / filename
        # Written aside and renamed: a chained preparation writes its
        # boundary segments straight into a PUBLISHED cache directory, where
        # a reader must never see half an array under its final name.
        partial = path.with_name(filename + ".tmp")
        with partial.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        _replace_file(partial, path)
        self.manifest[key] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "sha256": _array_sha256(array),
        }
        self.payload_bytes += int(array.nbytes)

    def link_verified(self, key: str, reader: "PreparedCacheReader") -> None:
        """Reuse one already-verified immutable payload unchanged.

        A hard link where the drive has them: atomic, and no extra space,
        which is what an hourly append to a growing cache wants.  Where it
        has none (exFAT, or the two folders on different drives) the
        payload is copied instead and proven byte-identical to its source
        (:func:`woof.filesystem_paths.copy_verified`), after one check
        that everything still to be copied fits; the copy costs its size in
        disk, and the run says so in one line.  This used to refuse, citing
        only that cost.  Either way the completed staging bundle is
        verified against its manifest again before publication, which
        catches a predecessor that changed while it was being placed.
        """
        from woof.filesystem_paths import (
            copy_verified, links_unavailable, note_copy_instead_of_link,
            require_room_to_copy)

        if not isinstance(key, str) or not key or key in self.manifest:
            raise ValueError(f"invalid or duplicate prepared-cache key {key!r}")
        try:
            source_spec = reader.arrays[key]
        except KeyError as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache is missing array {key!r}") from exc
        filename = f"a{len(self.manifest):05d}.npy"
        source = reader.path / source_spec["file"]
        destination = self.temporary / filename
        size = source.stat().st_size
        if not self.copying:
            try:
                os.link(source, destination)
            except FileExistsError:
                raise
            except OSError as exc:
                if not links_unavailable(exc):
                    raise
                require_room_to_copy(
                    max(self._pending_reuse_bytes, size),
                    self.temporary.parent,
                    f"the prepared cache {reader.path}")
                note_copy_instead_of_link()
                self.copying = True
        if self.copying:
            copy_verified(source, destination)
        self._pending_reuse_bytes = max(0, self._pending_reuse_bytes - size)
        spec = _json_copy(source_spec)
        spec["file"] = filename
        self.manifest[key] = spec
        self.payload_bytes += int(spec["nbytes"])


def read_manifest_array(directory, key: str, spec) -> np.ndarray:
    """Read one payload and prove it against its manifest row.

    Shared by the sealed reader and the streamed boundary reader
    (:mod:`woof.ingest.boundary_stream`), so a segment read before the seal
    is held to the same shape, dtype and content digest as a sealed read.
    """

    try:
        filename = spec["file"]
    except (KeyError, TypeError) as exc:
        raise PreparedCacheCorruptError(
            f"prepared cache array entry {key!r} is malformed") from exc
    candidate = Path(filename)
    if (candidate.name != filename or candidate.is_absolute()
            or not str(filename).endswith(".npy")):
        raise PreparedCacheCorruptError(
            f"prepared cache array {key!r} has unsafe file name")
    path = Path(directory) / filename
    try:
        with path.open("rb") as stream:
            array = np.load(stream, allow_pickle=False)
    except (OSError, EOFError, ValueError) as exc:
        _raise_if_resource_limit(exc, context=f"prepared cache array {key!r}")
        raise PreparedCacheCorruptError(
            f"prepared cache array {key!r} is unreadable") from exc
    if (list(array.shape) != spec["shape"]
            or str(array.dtype) != spec["dtype"]
            or int(array.nbytes) != int(spec["nbytes"])
            or _array_sha256(array) != spec["sha256"]):
        raise PreparedCacheCorruptError(
            f"prepared cache array {key!r} fails its manifest")
    return array


def verify_manifest_arrays(read, keys, specs) -> None:
    """Read and hash-check every array in ``keys``, across the host's cores.

    Each array is checked exactly as :func:`read_manifest_array` checks it,
    on a thread pool whose bytes in flight are bounded, and the failure
    raised is the one the sorted serial walk would raise first.  Breakage
    this prevents: the end-of-run verify read and hashed a full HRRR cache
    one array at a time on one thread while the forecast's cards waited
    (V59, 8 x RTX 5090, 2026-10-21: finalize 6.7 min on one core).
    """
    from concurrent.futures import ThreadPoolExecutor

    keys = sorted(keys)
    if len(keys) <= 1:
        for key in keys:
            read(key)
        return
    workers = max(1, min(len(keys), (os.cpu_count() or 1), 32))
    budget = max(1 << 30, int(os.environ.get(
        "GPUWM_VERIFY_INFLIGHT_BYTES", 8 << 30)))
    errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="cache-verify") as pool:
        pending = []
        inflight = 0

        def settle(item):
            key, future, size = item
            try:
                future.result()
            except BaseException as error:  # noqa: BLE001 - re-raised below
                errors[key] = error
            return size

        for key in keys:
            size = int(specs[key].get("nbytes", 0) or 0)
            while pending and inflight + size > budget:
                inflight -= settle(pending.pop(0))
            pending.append((key, pool.submit(
                lambda key=key: (read(key), None)[1]), size))
            inflight += size
        for item in pending:
            settle(item)
    for key in keys:
        if key in errors:
            raise errors[key]


class PreparedCacheReader:
    """Validated cache manifest with checked, one-array-at-a-time reads."""

    def __init__(self, path, *, expected_identity):
        self.path = Path(path)
        try:
            raw = (self.path / _HEADER_NAME).read_text(encoding="utf-8")
            header = json.loads(raw)
        except (FileNotFoundError, OSError, UnicodeDecodeError,
                json.JSONDecodeError) as exc:
            _raise_if_resource_limit(exc, context=f"prepared cache header {self.path}")
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} has no readable header") from exc
        required = {
            "schema", "status", "identity", "metadata", "arrays",
            "content_sha256", "payload_bytes",
        }
        if not isinstance(header, dict) or required - set(header):
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} has a malformed header")
        if (header["schema"] != PREPARED_CACHE_SCHEMA
                or header["status"] != "READY"):
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} is not a READY "
                f"{PREPARED_CACHE_SCHEMA} bundle")
        identity = _json_copy(expected_identity)
        tolerated, differing = compare_prepared_identity(
            header["identity"], identity)
        if differing:
            raise PreparedCacheMismatchError(
                prepared_identity_refusal(
                    subject=f"prepared cache {self.path}",
                    header=header, differing=differing))
        #: Provenance: which identity fields this restore accepted as
        #: schema growth rather than as a match.  Empty is the normal
        #: case, and a caller that records it can show exactly what it
        #: tolerated and why the state is still the state it asked for.
        self.tolerated_identity_fields = tuple(tolerated)
        arrays = header["arrays"]
        if not isinstance(arrays, dict) or not arrays:
            raise PreparedCacheCorruptError(
                "prepared cache array manifest is empty or malformed")
        basis = {
            "schema": header["schema"],
            "identity": header["identity"],
            "metadata": header["metadata"],
            "arrays": arrays,
            "payload_bytes": header["payload_bytes"],
        }
        observed_content = hashlib.sha256(
            _canonical(basis).encode("utf-8")).hexdigest()
        if observed_content != header["content_sha256"]:
            raise PreparedCacheCorruptError(
                "prepared cache header content digest mismatch")
        filenames = []
        payload_bytes = 0
        for key, spec in arrays.items():
            if not isinstance(key, str) or not isinstance(spec, dict):
                raise PreparedCacheCorruptError(
                    "prepared cache contains a malformed array entry")
            try:
                filename = spec["file"]
                shape = spec["shape"]
                dtype = np.dtype(spec["dtype"])
                nbytes = int(spec["nbytes"])
                digest = spec["sha256"]
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparedCacheCorruptError(
                    f"prepared cache array entry {key!r} is malformed") from exc
            candidate = Path(filename)
            if (candidate.name != filename or candidate.is_absolute()
                    or not filename.endswith(".npy")):
                raise PreparedCacheCorruptError(
                    f"prepared cache array {key!r} has unsafe file name")
            if (not isinstance(shape, list)
                    or any(not isinstance(extent, int) or extent < 0
                           for extent in shape)
                    or dtype.hasobject
                    or nbytes != int(np.prod(shape, dtype=np.int64))
                    * dtype.itemsize
                    or not isinstance(digest, str) or len(digest) != 64):
                raise PreparedCacheCorruptError(
                    f"prepared cache array entry {key!r} is inconsistent")
            filenames.append(filename)
            payload_bytes += nbytes
        if len(set(filenames)) != len(filenames):
            raise PreparedCacheCorruptError(
                "prepared cache array manifest reuses a payload file")
        if payload_bytes != int(header["payload_bytes"]):
            raise PreparedCacheCorruptError(
                "prepared cache payload byte total is inconsistent")
        expected_files = set(filenames) | {_HEADER_NAME}
        try:
            actual_files = {entry.name for entry in self.path.iterdir()
                            if entry.is_file()}
            directories = [entry.name for entry in self.path.iterdir()
                           if entry.is_dir()]
        except OSError as exc:
            _raise_if_resource_limit(exc, context=f"prepared cache inventory {self.path}")
            raise PreparedCacheCorruptError(
                f"prepared cache {self.path} is unreadable") from exc
        if actual_files != expected_files or directories:
            raise PreparedCacheCorruptError(
                "prepared cache file inventory differs from its manifest")
        self.header = header
        self.arrays = arrays

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self.header["metadata"])

    @property
    def content_sha256(self) -> str:
        return str(self.header["content_sha256"])

    @property
    def payload_bytes(self) -> int:
        return int(self.header["payload_bytes"])

    def read_array(self, key: str) -> np.ndarray:
        try:
            spec = self.arrays[key]
        except KeyError as exc:
            raise PreparedCacheCorruptError(
                f"prepared cache is missing array {key!r}") from exc
        return read_manifest_array(self.path, key, spec)

    def verify_all(self) -> dict[str, object]:
        verify_manifest_arrays(self.read_array, self.arrays, self.arrays)
        return {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "PASS",
            "path": str(self.path.resolve()),
            "content_sha256": self.content_sha256,
            "array_count": len(self.arrays),
            "payload_bytes": self.payload_bytes,
        }



class PreparedHeadReader:
    """The start-time half of a prepared cache, read before its seal.

    A chained preparation publishes every non-boundary array of its cache
    with ``boundary-stream/head.json`` before the boundary intervals exist
    (:mod:`woof.ingest.boundary_stream`).  This reader serves exactly
    those arrays, held to the head's manifest rows, and a header view whose
    metadata carries the declared interval schedule, so a forecast can be
    restored and validated at the head.  It has no ``content_sha256``: that
    digest exists only at the seal, where the runner checks it.
    """

    def __init__(self, root, head, *, expected_identity):
        self.head = head
        cache = head["basis"]["cache"]
        self.path = Path(root) / str(cache["directory"])
        identity = _json_copy(expected_identity)
        tolerated, differing = compare_prepared_identity(
            cache["identity"], identity)
        if differing:
            raise PreparedCacheMismatchError(
                prepared_identity_refusal(
                    subject=f"prepared head {self.path}",
                    header={"identity": cache["identity"]},
                    differing=differing))
        self.tolerated_identity_fields = tuple(tolerated)
        lbc = cache.get("lbc")
        metadata = _json_copy(cache["metadata"])
        metadata["lbc"] = None if lbc is None else {
            "spec_bdy_width": lbc["spec_bdy_width"],
            "spec_zone": lbc["spec_zone"],
            "relax_zone": lbc["relax_zone"],
            "intervals": [
                {"start_seconds": float(start), "end_seconds": float(end),
                 "fields": list(lbc["fields"])}
                for start, end in lbc["schedule"]],
        }
        metadata["setup_fingerprint"] = None
        self.arrays = dict(cache["arrays"])
        self.header = {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "HEAD",
            "identity": cache["identity"],
            "metadata": metadata,
            "arrays": self.arrays,
            "payload_bytes": int(cache["payload_bytes"]),
        }
        self.setup_core_fingerprint = str(cache["setup_core_fingerprint"])

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self.header["metadata"])

    @property
    def content_sha256(self):
        return None

    @property
    def payload_bytes(self) -> int:
        return int(self.header["payload_bytes"])

    def read_array(self, key: str) -> np.ndarray:
        try:
            spec = self.arrays[key]
        except KeyError as exc:
            raise PreparedCacheCorruptError(
                f"prepared head is missing array {key!r}") from exc
        return read_manifest_array(self.path, key, spec)

    def verify_all(self) -> dict[str, object]:
        verify_manifest_arrays(self.read_array, self.arrays, self.arrays)
        return {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "HEAD_PASS",
            "path": str(self.path.resolve()),
            "content_sha256": None,
            "array_count": len(self.arrays),
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class CachedInitialResult:
    """Integration-facing subset of :class:`RealInitResult` restored cold."""

    state: object
    coord: object
    base: object
    surface_pressure: np.ndarray
    surface_qv: np.ndarray
    hydrometeor_initialization: Mapping[str, object]


@dataclass(frozen=True)
class RestoredPreparedCache:
    initial_result: CachedInitialResult
    met: object
    surface: object | None
    boundaries: object
    metadata: Mapping[str, object]
    receipt: Mapping[str, object]


def prepared_cache_identity(*, bridge_manifest_sha256: str,
                            source_manifest_sha256: str,
                            static_cache_sha256: str,
                            namelist_sha256: str, domain_config,
                            forcing_hours=None,
                            forcing_offsets_seconds=None,
                            source_identity,
                            namelist_extension_invariant=None,
                            ) -> dict[str, object]:
    """Canonical identity callers must reproduce exactly on every restore."""
    if (forcing_hours is None) == (forcing_offsets_seconds is None):
        raise ValueError(
            "prepared cache identity requires exactly one of forcing_hours "
            "or forcing_offsets_seconds")
    forcing_identity = (
        {"forcing_hours": [int(hour) for hour in forcing_hours]}
        if forcing_hours is not None else
        {"forcing_offsets_seconds": [
            int(offset) for offset in forcing_offsets_seconds]})
    identity = {
        "bridge_manifest_sha256": str(bridge_manifest_sha256).lower(),
        "source_manifest_sha256": str(source_manifest_sha256).lower(),
        "static_cache_sha256": str(static_cache_sha256).lower(),
        "namelist_sha256": str(namelist_sha256).lower(),
        "domain_config": prepared_domain_config_identity(domain_config),
        **forcing_identity,
        "source_identity": source_identity,
    }
    if namelist_extension_invariant is not None:
        identity["namelist_extension_invariant"] = \
            namelist_extension_invariant
    return _json_copy(identity)


def _coord_metadata(coord) -> dict[str, object]:
    result = {}
    for field in dataclass_fields(coord):
        value = getattr(coord, field.name)
        if not isinstance(value, np.ndarray):
            result[field.name] = value
    return _json_copy(result)


def _base_metadata(base) -> dict[str, object]:
    result = {}
    for field in dataclass_fields(base):
        value = getattr(base, field.name)
        if not isinstance(value, np.ndarray):
            result[field.name] = value
    return _json_copy(result)


def _is_nested_child_identity(identity) -> bool:
    """Return whether identity explicitly binds a parent-forced child."""

    if not isinstance(identity, Mapping):
        return False
    domain = identity.get("domain_config")
    if not isinstance(domain, Mapping):
        return False
    run = domain.get("run")
    return (isinstance(run, Mapping)
            and isinstance(domain.get("parent_id"), int)
            and int(domain["parent_id"]) > 0
            and run.get("nested") is True
            and run.get("specified") is False)


def _restore_lbc_mode(*, lbc_metadata, identity,
                      allow_nested_without_lbc: bool) -> str:
    """Resolve root/child boundary ownership without importing CuPy."""

    if not isinstance(allow_nested_without_lbc, bool):
        raise TypeError("allow_nested_without_lbc must be bool")
    if lbc_metadata is not None:
        if not isinstance(lbc_metadata, Mapping):
            raise PreparedCacheCorruptError(
                "prepared cache LBC metadata must be an object or null")
        return "external"
    if (allow_nested_without_lbc
            and _is_nested_child_identity(identity)):
        return "nested-parent-forced"
    raise PreparedCacheMismatchError(
        "nested export-only prepared cache has no external LBCs and "
        "cannot be restored as a standalone forecast root")


def _prepared_cache_staging_path(path: Path, *, nonce: str | None = None
                                 ) -> Path:
    """Return a compact, create-only sibling used for atomic publication.

    The target name must not be repeated in this private basename: caches are
    nested below other transaction staging roots and that repetition can make
    an otherwise valid published tree uncreatable on Windows.  The caller's
    ``mkdir`` remains the collision/ownership authority.
    """

    token = uuid.uuid4().hex[:10] if nonce is None else nonce
    if (not isinstance(token, str) or len(token) != 10
            or any(character not in "0123456789abcdef" for character in token)):
        raise ValueError(
            "prepared-cache staging nonce must be 10 lowercase hex characters")
    return path.with_name(f".p-{token}")


def write_prepared_cache(path, *, identity, initial_result, met,
                         boundaries, surface=None,
                         metadata=None,
                         sealed_forcing_extension: bool = False
                         ) -> dict[str, object]:
    """Write one immutable prepared-state bundle and publish atomically.

    Root-domain caches require external lateral boundaries.  A cache may omit
    them only when its identity explicitly binds a nested, non-specified child;
    that export-only cache feeds ``wrfinput_dNN`` and cannot be restored as a
    standalone forecast root.

    The whole-set call of :class:`PreparedCacheStream`: head, every interval
    as a segment, seal, in one go.  A chained preparation makes the same
    three calls with the forcing times built in between, so the one-shot
    cache and the streamed cache are the same bytes by construction.
    """

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared cache {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _prepared_cache_staging_path(path)
    temporary.mkdir()
    try:
        stream = PreparedCacheStream(
            temporary, identity=identity,
            sealed_forcing_extension=sealed_forcing_extension)
        stream.write_head(
            initial_result=initial_result, met=met, surface=surface,
            metadata=metadata, lbc=boundary_schedule(boundaries),
            # The one-shot writer has always fingerprinted the STATE's own
            # attachment; it stays the authority whenever the caller's
            # state carries some other boundary set than the one written.
            lateral_from_state=(
                getattr(initial_result.state, "lateral_boundaries", None)
                is not boundaries))
        if boundaries is not None:
            for index, interval in enumerate(boundaries.intervals):
                stream.write_segment(index, interval)
        receipt = stream.seal()
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    receipt = dict(receipt)
    receipt["path"] = str(path.resolve())
    return receipt


def boundary_schedule(boundaries, *, fields=None):
    """The head's declaration of a boundary set: controls and schedule.

    ``None`` for a cache without external boundaries.  ``fields`` names the
    inventory when the intervals do not exist yet (a chained preparation
    knows it from the start time's frame).
    """

    if boundaries is None:
        return None
    intervals = boundaries.intervals
    return {
        "spec_bdy_width": int(boundaries.spec_bdy_width),
        "spec_zone": int(boundaries.spec_zone),
        "relax_zone": int(boundaries.relax_zone),
        "schedule": [[float(interval.start_seconds),
                      float(interval.end_seconds)] for interval in intervals],
        "fields": (sorted(fields) if fields is not None
                   else sorted(intervals[0].fields) if len(intervals)
                   else []),
    }


#: A stream head's record of the user metadata keys its seal completes
#: (:meth:`PreparedCacheStream.write_head` ``seal_completes``), present
#: only when there are any.
SEAL_COMPLETES_KEY = "seal_completes_user_metadata"


class PreparedCacheStream:
    """The one prepared-cache writer: a head, one segment per interval, a seal.

    ``write_head`` writes every array the start time makes (state, coord,
    base, result, met, surface) under the file numbers the bundle has always
    used, and starts the setup fingerprint from the start state's immutable
    core.  ``write_segment(k, interval)`` writes interval k's boundary tables
    under the next file numbers and feeds the same bytes into the same
    digest the whole-set fingerprint walks.  ``seal`` writes ``header.json``
    with the unchanged ``basis``, so ``content_sha256`` is the value the
    one-shot writer produces for the same inputs.
    """

    def __init__(self, directory, *, identity,
                 sealed_forcing_extension: bool = False):
        self.directory = Path(directory)
        self.identity = _json_copy(identity)
        self.sealed_forcing_extension = bool(sealed_forcing_extension)
        self._writer = _BundleWriter(self.directory)
        self._metadata: dict[str, object] | None = None
        self._lbc: dict[str, object] | None = None
        self._digest = None
        self._fingerprint: str | None = None
        self._core_fingerprint: str | None = None
        self._state_prefix = None
        self._nested = False
        self._intervals: list[dict[str, object]] = []
        self._prefix_rows: list[dict[str, object]] = []
        #: The same rows in the rebuilt end-frame identity, kept only for a
        #: sealed-extension cache whose intervals do not all record their
        #: built end frame (its document is then wholly rebuilt).
        self._rebuilt_prefix_rows: list[dict[str, object]] = []
        self._built_end_frames = True
        self._rational = False
        self._sealed = False
        self._seal_completes: tuple[str, ...] = ()

    def move(self, directory) -> None:
        """Follow the directory after its tree was published by rename."""

        self.directory = Path(directory)
        self._writer.temporary = self.directory

    @property
    def payload_bytes(self) -> int:
        return self._writer.payload_bytes

    def write_head(self, *, initial_result, met, surface=None, metadata=None,
                   lbc=None, lateral_from_state: bool = False,
                   seal_completes: Sequence[str] = ()
                   ) -> dict[str, object]:
        """Write every array the start time makes.

        ``seal_completes`` names user ``metadata`` keys whose value is a
        mapping the start time holds only part of (the native HRRR
        preparation's per-lead ``mapping_reports``: the start lead's at the
        head, every boundary lead's once it is mapped).  :meth:`seal` adds
        the later entries (``completed_metadata``) and refuses any change
        to an entry the head already wrote, so the sealed header is the
        one-shot writer's for the same inputs.  Empty for every other
        route, whose head record is then what it always was.
        """
        from woof.state_serialization_contract import (
            _lateral_fingerprint_header,
            _update_lateral_fingerprint, _update_setup_core,
            lateral_boundary_prefix_identity, serialized_state_attrs,
        )

        if self._metadata is not None:
            raise RuntimeError("the prepared-cache head is written once")
        self.directory.mkdir(parents=True, exist_ok=True)
        writer = self._writer
        with writer.immutable_batch():
            state_names = []
            for name in serialized_state_attrs(initial_result.state):
                value = getattr(initial_result.state, name, None)
                if value is not None:
                    writer.add(f"state/{name}", value)
                    state_names.append(name)
            coord_arrays = []
            for field in dataclass_fields(initial_result.coord):
                value = getattr(initial_result.coord, field.name)
                if isinstance(value, np.ndarray):
                    writer.add(f"coord/{field.name}", value)
                    coord_arrays.append(field.name)
            base_arrays = []
            for field in dataclass_fields(initial_result.base):
                value = getattr(initial_result.base, field.name)
                if isinstance(value, np.ndarray):
                    writer.add(f"base/{field.name}", value)
                    base_arrays.append(field.name)
            writer.add("result/surface_pressure", initial_result.surface_pressure)
            writer.add("result/surface_qv", initial_result.surface_qv)

            met_names = _prepared_met_names(met, surface=surface)
            for name in met_names:
                writer.add(f"met/{name}", met.fields[name])

            surface_names = []
            if surface is not None:
                if not isinstance(surface, Mapping):
                    raise TypeError("canonical prepared surface must be a mapping")
                missing_surface = sorted(
                    _CANONICAL_SURFACE_REQUIRED - set(surface))
                if missing_surface:
                    raise KeyError(
                        "canonical prepared surface is missing "
                        f"{missing_surface}")
                surface_names = sorted(_CANONICAL_SURFACE_REQUIRED)
                for name in surface_names:
                    writer.add(f"surface/{name}", surface[name])

        if lbc is None:
            if not _is_nested_child_identity(self.identity):
                raise ValueError(
                    "omitting prepared-cache LBCs requires an identity-bound "
                    "nested non-specified child")
            if self.sealed_forcing_extension:
                raise ValueError(
                    "sealed prepared-cache forcing requires root LBCs")
        else:
            lbc = {
                "spec_bdy_width": int(lbc["spec_bdy_width"]),
                "spec_zone": int(lbc["spec_zone"]),
                "relax_zone": int(lbc["relax_zone"]),
                "schedule": [[float(start), float(end)]
                             for start, end in lbc["schedule"]],
                "fields": sorted(str(name) for name in lbc["fields"]),
            }
        self._lbc = lbc

        state = initial_result.state
        digest = hashlib.sha256()
        self._nested = _update_setup_core(digest, state, error_type=ValueError)
        self._core_fingerprint = digest.copy().hexdigest()
        if self._nested:
            self._fingerprint = digest.hexdigest()
        elif lbc is None or lateral_from_state:
            _update_lateral_fingerprint(digest, state)
            self._fingerprint = digest.hexdigest()
        else:
            _lateral_fingerprint_header(
                digest, spec_bdy_width=lbc["spec_bdy_width"],
                spec_zone=lbc["spec_zone"], relax_zone=lbc["relax_zone"],
                count=len(lbc["schedule"]))
            self._digest = digest
        if self.sealed_forcing_extension and (lateral_from_state
                                              or self._nested):
            self._state_prefix = lateral_boundary_prefix_identity(state)

        cache_metadata = {
            "user": _json_copy(metadata or {}),
            "state_names": state_names,
            "coord_arrays": coord_arrays,
            "coord_scalars": _coord_metadata(initial_result.coord),
            "base_arrays": base_arrays,
            "base_scalars": _base_metadata(initial_result.base),
            "met_fields": met_names,
            "surface_fields": surface_names,
            "hydrometeor_initialization": _json_copy(
                getattr(initial_result, "hydrometeor_initialization", {})),
        }
        # WHICH AEROSOL SOURCE THE PREPARATION CHOSE, carried so the
        # FORECAST can report it.  The choice is made here, at preparation
        # time, by initialize_real -- WRF's monthly WIF climatology or
        # thompson_init's synthetic profile -- and the run that publishes a
        # report.json is a separate process reading this bundle back cold.
        # Without this row the fact exists only in the preparing process's
        # stderr, and a forecast receipt cannot say which of two different
        # initial conditions its aerosol came from.
        #
        # ADDED ONLY WHEN NON-EMPTY, the same emptiness contract
        # ``report["tiles"]`` keeps: the receipt is empty for every scheme
        # but aerosol-aware Thompson, so a cache for any other run
        # canonicalizes to the bytes it did before this row existed and
        # every stored ``prepared_header_sha256`` stays valid.
        aerosol_initialization = _json_copy(
            getattr(initial_result, "aerosol_initialization", {}) or {})
        if aerosol_initialization:
            cache_metadata["aerosol_initialization"] = aerosol_initialization
        # The start state's W, by the same emptiness contract: the row is
        # written only when the source's vertical velocity was carried
        # (policy "carried-from-source"), so a zero-W cache canonicalizes
        # to the bytes it did before and its stored header digest stands.
        vertical_velocity_initialization = _json_copy(
            getattr(initial_result, "vertical_velocity_initialization", {}) or {})
        if vertical_velocity_initialization.get("policy") == "carried-from-source":
            cache_metadata["vertical_velocity_initialization"] = (
                vertical_velocity_initialization)
        # The water surface's receipt, and the skin temperatures taken from
        # the other surface because the source held none of the target's
        # own, ride the bundle only when there is something to say, by the
        # same emptiness contract.
        lake_receipt = getattr(met, "water_temperature_receipt", None)
        if lake_receipt and (lake_receipt.get("lake_water_mapping")
                             or lake_receipt.get("water_fill")):
            cache_metadata["water_temperature"] = _json_copy(dict(lake_receipt))
        other_surface = {
            name: int(counts.get("other_surface", 0))
            for name, counts in (
                getattr(met, "masked_field_repairs", None) or {}).items()
            if counts.get("other_surface", 0)}
        if other_surface:
            cache_metadata["surface_from_other_surface"] = other_surface
        # And the soil values of land the source holds no land for, which
        # take the column the soil router builds at the skin temperature
        # and the soil's field capacity (woof/ingest/soil.py:
        # island_soil_columns), by the same contract.
        island_soil = {
            name: int(counts.get("no_source_land", 0))
            for name, counts in (
                getattr(met, "masked_field_repairs", None) or {}).items()
            if counts.get("no_source_land", 0)}
        if island_soil:
            cache_metadata["soil_from_skin_and_field_capacity"] = island_soil
        if self.sealed_forcing_extension:
            cache_metadata.update({
                "forcing_extension_mode": SEALED_PREPARED_EXTENSION_MODE,
                "setup_core_fingerprint": self._core_fingerprint,
            })
        completes = sorted({str(key) for key in seal_completes})
        user = cache_metadata["user"]
        for key in completes:
            if not isinstance(user.get(key), Mapping):
                raise ValueError(
                    f"the head names user metadata {key!r} for its seal to "
                    "complete, and holds no mapping under it to extend")
        self._seal_completes = tuple(completes)
        self._metadata = cache_metadata
        head = {
            "identity": _json_copy(self.identity),
            "metadata": _json_copy(cache_metadata),
            "arrays": _json_copy(writer.manifest),
            "payload_bytes": int(writer.payload_bytes),
            "lbc": _json_copy(lbc),
            "setup_core_fingerprint": self._core_fingerprint,
        }
        # Only when named, so every other head (and its digest) is what it
        # was before a seal could complete metadata.
        if completes:
            head[SEAL_COMPLETES_KEY] = completes
        return head

    def write_segment(self, index: int, interval) -> dict[str, object]:
        from woof.state_serialization_contract import (
            _lateral_fingerprint_interval, built_end_frame,
            interval_has_time_law, lateral_boundary_prefix_row,
        )

        if self._metadata is None or self._sealed:
            raise RuntimeError("a segment belongs between head and seal")
        if self._lbc is None:
            raise ValueError("this cache declared no external boundaries")
        index = int(index)
        schedule = self._lbc["schedule"]
        if index != len(self._intervals):
            raise ValueError(
                f"boundary segment {index} arrived before segment "
                f"{len(self._intervals)}; segments are written in time "
                "order because that order numbers their files")
        if index >= len(schedule):
            raise ValueError(
                f"boundary segment {index} is past the declared "
                f"{len(schedule)} intervals")
        bounds = [float(interval.start_seconds),
                  float(interval.end_seconds)]
        if bounds != schedule[index]:
            raise ValueError(
                f"boundary segment {index} spans {bounds}, not the "
                f"declared {schedule[index]}")
        writer = self._writer
        before_keys = set(writer.manifest)
        before_bytes = writer.payload_bytes
        field_names = sorted(interval.fields)
        with writer.immutable_batch():
            for name in field_names:
                field = interval.fields[name]
                for side_name in ("west", "east", "south", "north"):
                    side = getattr(field, side_name)
                    prefix = f"lbc/{index}/{name}/{side_name}"
                    writer.add(f"{prefix}/value", side.value)
                    writer.add(f"{prefix}/tendency", side.tendency)
                    if side.time_law is not None:
                        for coefficient in ("quadratic", "denominator_rate"):
                            writer.add(f"{prefix}/rational_time_v1/{coefficient}",
                                       getattr(side.time_law, coefficient))
        # The frame this interval's tendency was built toward, as its
        # builder recorded it (A140b): kept in the interval's row and in its
        # segment marker, because no array holds it (the last interval's is
        # a frame no later interval starts from), and a reader restores it
        # onto the interval so the forcing row it hashes is this one.
        # Written only when recorded, so a wrfbdy file's cache keeps the
        # header it always had.
        built = built_end_frame(interval)
        interval_row = {
            "start_seconds": bounds[0],
            "end_seconds": bounds[1],
            "fields": field_names,
        }
        if built is not None:
            interval_row[END_FRAME_KEY] = built
        self._intervals.append(interval_row)
        if self._digest is not None:
            _lateral_fingerprint_interval(self._digest, interval)
        row = lateral_boundary_prefix_row(interval)
        self._prefix_rows.append(row)
        self._built_end_frames = self._built_end_frames and built is not None
        if self.sealed_forcing_extension:
            self._rebuilt_prefix_rows.append(
                row if built is None else lateral_boundary_prefix_row(
                    interval, rebuilt_end_frame=True))
        self._rational = self._rational or interval_has_time_law(interval)
        arrays = {key: _json_copy(spec) for key, spec in writer.manifest.items()
                  if key not in before_keys}
        segment = {
            "index": index,
            "start_seconds": bounds[0],
            "end_seconds": bounds[1],
            "fields": field_names,
            "arrays": arrays,
            "payload_bytes": int(writer.payload_bytes - before_bytes),
            "prefix": _json_copy(row),
        }
        if built is not None:
            segment[END_FRAME_KEY] = built
        return segment

    def seal(self, *, identity=None,
             completed_metadata: Mapping[str, object] | None = None,
             posted_user_metadata: Mapping[str, str] | None = None,
             ) -> dict[str, object]:
        """Write ``header.json``; return the one-shot writer's receipt.

        ``identity`` replaces the identity the head was written under: an
        as-posted head carries placeholders where the input manifest's
        digest goes (:mod:`woof.ingest.boundary_stream`), and the seal
        writes the one-shot identity, so ``content_sha256`` is the one-shot
        writer's for the same inputs.  Which keys may change is the
        stream's rule, checked before this is called.

        ``completed_metadata`` gives the whole value of each user metadata
        key the head named in ``seal_completes``.  Refused: a key the head
        did not name, and a value that drops or changes an entry the head
        already wrote, because a head-bound forecast checked those entries
        at the head and the seal must not take back what it checked.

        ``posted_user_metadata`` replaces user metadata strings an
        as-posted head wrote as its plan's placeholder with the digests
        its seal learned (the stream checks which keys, and against the
        sealed identity, before this is called); a value that is not a
        placeholder is refused.
        """
        from woof.state_serialization_contract import (
            lateral_boundary_prefix_document,
        )

        if self._metadata is None or self._sealed:
            raise RuntimeError(
                "the prepared cache is sealed once, after its head")
        if identity is not None:
            self.identity = _json_copy(identity)
        cache_metadata = dict(self._metadata)
        if completed_metadata:
            user = dict(cache_metadata["user"])
            for key, value in dict(completed_metadata).items():
                if key not in self._seal_completes:
                    raise ValueError(
                        f"the seal completes user metadata {key!r}, which "
                        "its head did not name")
                value = _json_copy(value)
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"completed user metadata {key!r} is not a mapping")
                changed = sorted(
                    name for name, entry in user[key].items()
                    if value.get(name) != entry)
                if changed:
                    raise ValueError(
                        f"the seal changes or drops {key} entries {changed} "
                        "the head already wrote")
                user[key] = value
            cache_metadata["user"] = user
        if posted_user_metadata:
            from woof.ingest.boundary_stream import (
                AS_POSTED_PLACEHOLDER_PREFIX)

            user = dict(cache_metadata["user"])
            for key, value in dict(posted_user_metadata).items():
                held = user.get(key)
                if (not isinstance(held, str)
                        or not held.startswith(AS_POSTED_PLACEHOLDER_PREFIX)):
                    raise ValueError(
                        f"the seal writes user metadata {key!r}, which its "
                        "head did not hold as an as-posted placeholder")
                user[key] = str(value)
            cache_metadata["user"] = user
        if self._lbc is None:
            lbc_metadata = None
        else:
            count = len(self._lbc["schedule"])
            if len(self._intervals) != count:
                raise ValueError(
                    f"the prepared cache declares {count} boundary intervals "
                    f"and {len(self._intervals)} were written")
            lbc_metadata = {
                "spec_bdy_width": self._lbc["spec_bdy_width"],
                "spec_zone": self._lbc["spec_zone"],
                "relax_zone": self._lbc["relax_zone"],
                "intervals": list(self._intervals),
            }
        cache_metadata["lbc"] = lbc_metadata
        cache_metadata["setup_fingerprint"] = (
            self._fingerprint if self._digest is None
            else self._digest.hexdigest())
        if self.sealed_forcing_extension:
            # One series, one identity: the built end-frame document when
            # every interval recorded its end frame, else the rebuilt one
            # (lateral_boundary_prefix_identity makes the same choice).
            built = bool(self._prefix_rows) and self._built_end_frames
            cache_metadata["lateral_boundary_prefix"] = (
                self._state_prefix
                if self._state_prefix is not None or self._nested
                else lateral_boundary_prefix_document(
                    spec_bdy_width=self._lbc["spec_bdy_width"],
                    spec_zone=self._lbc["spec_zone"],
                    relax_zone=self._lbc["relax_zone"],
                    rows=(self._prefix_rows if built
                          else self._rebuilt_prefix_rows),
                    rational=self._rational, built_end_frames=built))
        writer = self._writer
        basis = {
            "schema": PREPARED_CACHE_SCHEMA,
            "identity": _json_copy(self.identity),
            "metadata": cache_metadata,
            "arrays": writer.manifest,
            "payload_bytes": writer.payload_bytes,
        }
        from woof import __version__

        header = {
            **basis,
            "status": "READY",
            "created_utc": _creation_utc(self.directory),
            # Outside `basis`, so the content digest of a cache means
            # exactly what it meant before this release.  Its job is to
            # let a refusal name the release that wrote the bundle
            # instead of blaming the user's experiment file for a
            # package upgrade.
            CACHE_WRITER_KEY: {"gpuwm_version": __version__},
            "content_sha256": hashlib.sha256(
                _canonical(basis).encode("utf-8")).hexdigest(),
        }
        header_path = self.directory / _HEADER_NAME
        partial = header_path.with_name(HEADER_PARTIAL_NAME)
        partial.write_text(
            json.dumps(header, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        _replace_file(partial, header_path)
        self._sealed = True
        return {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "BUILT",
            "path": str(self.directory.resolve()),
            "content_sha256": header["content_sha256"],
            "array_count": len(writer.manifest),
            "payload_bytes": writer.payload_bytes,
        }


def _reader_boundary_side(reader, prefix):
    from woof.ingest.lateral_bc import RationalTimeLaw, SideBoundary
    keys = tuple(f"{prefix}/rational_time_v1/{name}"
                 for name in ("quadratic", "denominator_rate"))
    present = tuple(key in reader.arrays for key in keys)
    if any(present) and not all(present):
        raise PreparedCacheCorruptError(
            f"prepared cache {prefix} has an incomplete rational time law")
    law = (RationalTimeLaw(*(reader.read_array(key) for key in keys))
           if all(present) else None)
    return SideBoundary(reader.read_array(f"{prefix}/value"),
                        reader.read_array(f"{prefix}/tendency"), law)


#: The key of a prepared cache's LBC interval row (and of a boundary segment
#: marker) that records the digest of the frame the interval's tendency was
#: built toward (A140b).  Absent from every cache written before 2.8.1 and
#: from a series whose builder recorded none; such a cache still reads, and
#: its intervals hash in the rebuilt end-frame identity, as they always did.
END_FRAME_KEY = "end_frame_sha256"


def interval_built_end_frame(row, *, where: str):
    """The built end frame one LBC interval row records, or ``None``."""

    digest = row.get(END_FRAME_KEY) if isinstance(row, Mapping) else None
    if digest is None:
        return None
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise PreparedCacheCorruptError(
            f"{where} records a malformed built end frame {digest!r}")
    return digest


def _reader_boundaries(reader: PreparedCacheReader):
    from woof.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
        record_built_end_frame,
    )

    lbc = reader.header.get("metadata", {}).get("lbc")
    if not isinstance(lbc, dict) or not isinstance(lbc.get("intervals"), list):
        raise PreparedCacheMismatchError(
            f"prepared cache {reader.path} has no external LBC inventory")
    intervals = []
    for index, row in enumerate(lbc["intervals"]):
        if (not isinstance(row, dict)
                or not {"start_seconds", "end_seconds", "fields"}
                <= set(row)
                <= {"start_seconds", "end_seconds", "fields", END_FRAME_KEY}
                or not isinstance(row["fields"], list)
                or not row["fields"]):
            raise PreparedCacheCorruptError(
                f"prepared cache LBC interval {index} is malformed")
        fields = {}
        for name in row["fields"]:
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                prefix = f"lbc/{index}/{name}/{side_name}"
                sides[side_name] = _reader_boundary_side(reader, prefix)
            fields[name] = FieldBoundary(**sides)
        intervals.append(record_built_end_frame(BoundaryInterval(
            float(row["start_seconds"]), float(row["end_seconds"]), fields),
            interval_built_end_frame(
                row, where=f"prepared cache LBC interval {index}")))
    return LateralBoundaries(
        tuple(intervals), int(lbc["spec_bdy_width"]),
        int(lbc["spec_zone"]), int(lbc["relax_zone"]))


def _forcing_prefix(boundaries):
    from woof.state_serialization_contract import (
        lateral_boundary_prefix_identity,
    )

    return lateral_boundary_prefix_identity(
        SimpleNamespace(lateral_boundaries=boundaries))


def _without_run_window(value, *, omit_start_time=False):
    result = _json_copy(value)
    if isinstance(result, dict) and isinstance(result.get("run"), dict):
        result["run"].pop("run_seconds", None)
    if isinstance(result, dict) and omit_start_time:
        result.pop("start_time", None)
    return result


def _without_source_window(value, *, omit_model_start=False):
    result = _json_copy(value)
    if isinstance(result, dict):
        for key in ("source_forecast_hours", "model_forcing_hours"):
            result.pop(key, None)
        if omit_model_start:
            result.pop("model_start_time", None)
    return result


def _identity_time(value, *, label):
    if not isinstance(value, str):
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} must be an ISO timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} must be an ISO timestamp") from exc


def _domain_run_seconds(identity, *, label):
    domain = identity.get("domain_config") if isinstance(identity, dict) \
        else None
    run = domain.get("run") if isinstance(domain, dict) else None
    value = run.get("run_seconds") if isinstance(run, dict) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value)):
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} has no finite run_seconds")
    return float(value)


def _namelist_extension_invariant(identity, *, label):
    invariant = identity.get("namelist_extension_invariant") \
        if isinstance(identity, dict) else None
    try:
        return validated_namelist_extension_invariant(
            invariant, context=f"prepared-cache {label}")
    except ValueError:
        raise PreparedCacheMismatchError(
            f"prepared-cache {label} has no valid namelist extension "
            "invariant") from None


def extend_prepared_cache(path, *, predecessor, suffix, identity,
                          metadata, source_manifest_extension,
                          bridge_manifest_extension,
                          ) -> dict[str, object]:
    """Publish one cache by appending exactly one verified LBC interval.

    ``suffix`` is a normal two-time cache prepared only for the old endpoint
    and the newly arrived hour.  Initial state, static-derived setup, and all
    old forcing arrays come from ``predecessor``; only the suffix interval is
    admitted, after its shared FP32 endpoint frame matches cryptographically.
    """
    from woof import __version__
    from woof.ingest.lateral_bc import (
        BoundaryInterval, LateralBoundaries, record_built_end_frame)
    from woof.state_serialization_contract import (
        REBUILT_END_FRAME_PREFIX_SCHEMAS)

    path = Path(path)
    predecessor = Path(predecessor)
    suffix = Path(suffix)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared cache {path}")
    prior_header_path = predecessor / _HEADER_NAME
    suffix_header_path = suffix / _HEADER_NAME
    prior_header_before = hashlib.sha256(
        prior_header_path.read_bytes()).hexdigest()
    suffix_header_before = hashlib.sha256(
        suffix_header_path.read_bytes()).hexdigest()
    prior_header = json.loads(prior_header_path.read_text(encoding="utf-8"))
    suffix_header = json.loads(suffix_header_path.read_text(encoding="utf-8"))
    prior_reader = PreparedCacheReader(
        predecessor, expected_identity=prior_header.get("identity"))
    suffix_reader = PreparedCacheReader(
        suffix, expected_identity=suffix_header.get("identity"))
    prior_reader.verify_all()
    suffix_reader.verify_all()
    prior_meta = prior_reader.header["metadata"]
    if (prior_meta.get("forcing_extension_mode")
            != SEALED_PREPARED_EXTENSION_MODE):
        raise PreparedCacheMismatchError(
            "predecessor cache was not intentionally sealed for extension")

    old_identity = prior_reader.header["identity"]
    new_identity = _json_copy(identity)
    old_hours = old_identity.get("forcing_hours")
    new_hours = new_identity.get("forcing_hours")
    if (not isinstance(old_hours, list) or not old_hours
            or old_hours != list(range(len(old_hours)))
            or new_hours != old_hours + [len(old_hours)]):
        raise PreparedCacheMismatchError(
            "prepared-cache extension must append exactly one contiguous "
            "forcing hour")
    for key in ("static_cache_sha256",):
        if new_identity.get(key) != old_identity.get(key):
            raise PreparedCacheMismatchError(
                f"prepared-cache extension changes immutable {key}")
    old_namelist_invariant = _namelist_extension_invariant(
        old_identity, label="predecessor identity")
    if (_namelist_extension_invariant(
            new_identity, label="extended identity")
            != old_namelist_invariant):
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes immutable namelist fields")
    if (_without_run_window(new_identity.get("domain_config"))
            != _without_run_window(old_identity.get("domain_config"))):
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes immutable domain config")
    old_source = old_identity.get("source_identity")
    new_source = new_identity.get("source_identity")
    if _without_source_window(old_source) != _without_source_window(new_source):
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes immutable source identity")
    if (not isinstance(new_source, dict)
            or new_source.get("source_forecast_hours") != new_hours
            or new_source.get("model_forcing_hours") != new_hours):
        raise PreparedCacheMismatchError(
            "prepared-cache extension source window is not the full prefix")

    suffix_identity = suffix_reader.header["identity"]
    if suffix_identity.get("forcing_hours") != [0, 1]:
        raise PreparedCacheMismatchError(
            "prepared-cache suffix must contain exactly two source frames")
    for key in ("static_cache_sha256",):
        if suffix_identity.get(key) != old_identity.get(key):
            raise PreparedCacheMismatchError(
                f"prepared-cache suffix changes immutable {key}")
    if (_namelist_extension_invariant(
            suffix_identity, label="suffix identity")
            != old_namelist_invariant):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes immutable namelist fields")
    if suffix_identity.get("namelist_sha256") != new_identity.get(
            "namelist_sha256"):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix does not use the extended namelist bytes")
    if suffix_identity.get("source_manifest_sha256") != new_identity.get(
            "source_manifest_sha256"):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix is not bound to the extended source "
            "manifest")
    expected_bridge_extension = {
        "schema": "gpuwm-bridge-manifest-prefix-extension-v1",
        "predecessor_sha256": old_identity.get("bridge_manifest_sha256"),
        "suffix_sha256": suffix_identity.get("bridge_manifest_sha256"),
        "extended_sha256": new_identity.get("bridge_manifest_sha256"),
        "old_source_forecast_hours": old_source.get(
            "source_forecast_hours") if isinstance(old_source, dict) else None,
        "new_source_forecast_hours": new_source.get(
            "source_forecast_hours") if isinstance(new_source, dict) else None,
        "suffix_source_forecast_hours": [len(old_hours) - 1, len(old_hours)],
    }
    bridge_extension = _json_copy(bridge_manifest_extension)
    if (not isinstance(bridge_extension, dict)
            or any(bridge_extension.get(key) != value
                   for key, value in expected_bridge_extension.items())
            or not isinstance(bridge_extension.get("retained_entries"), int)
            or bridge_extension["retained_entries"] <= 0
            or not isinstance(bridge_extension.get("added_entries"), list)
            or not bridge_extension["added_entries"]):
        raise PreparedCacheMismatchError(
            "prepared-cache bridge-manifest prefix proof is malformed or "
            "belongs to another extension")
    if (_without_run_window(
            suffix_identity.get("domain_config"), omit_start_time=True)
            != _without_run_window(
                old_identity.get("domain_config"), omit_start_time=True)):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes immutable domain config")
    suffix_source = suffix_identity.get("source_identity")
    if (_without_source_window(suffix_source, omit_model_start=True)
            != _without_source_window(old_source, omit_model_start=True)):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes executable source identity")
    suffix_source_hours = [len(old_hours) - 1, len(old_hours)]
    if (not isinstance(suffix_source, dict)
            or suffix_source.get("source_forecast_hours")
            != suffix_source_hours
            or suffix_source.get("model_forcing_hours") != [0, 1]):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix does not contain the old terminal source "
            "frame and exactly one new hour")
    old_model_start = _identity_time(
        old_source.get("model_start_time") if isinstance(old_source, dict)
        else None, label="predecessor model_start_time")
    new_model_start = _identity_time(
        new_source.get("model_start_time"), label="extended model_start_time")
    suffix_model_start = _identity_time(
        suffix_source.get("model_start_time"), label="suffix model_start_time")
    if new_model_start != old_model_start:
        raise PreparedCacheMismatchError(
            "prepared-cache extension changes model time zero")
    expected_suffix_start = old_model_start + timedelta(
        hours=len(old_hours) - 1)
    if suffix_model_start != expected_suffix_start:
        raise PreparedCacheMismatchError(
            "prepared-cache suffix model start is not the old terminal hour")
    suffix_domain = suffix_identity.get("domain_config")
    if (not isinstance(suffix_domain, dict)
            or _identity_time(
                suffix_domain.get("start_time"),
                label="suffix domain start_time") != expected_suffix_start):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix domain start is not the old terminal hour")
    if (_domain_run_seconds(old_identity, label="predecessor identity")
            != float(len(old_hours) - 1) * 3600.0
            or _domain_run_seconds(new_identity, label="extended identity")
            != float(len(new_hours) - 1) * 3600.0
            or _domain_run_seconds(suffix_identity, label="suffix identity")
            != 3600.0):
        raise PreparedCacheMismatchError(
            "prepared-cache run windows do not exactly cover their forcing")
    expected_manifest_extension = {
        "schema": "gpuwm-source-manifest-prefix-extension-v1",
        "predecessor_sha256": old_identity.get("source_manifest_sha256"),
        "extended_sha256": new_identity.get("source_manifest_sha256"),
        "old_source_forecast_hours": old_source.get(
            "source_forecast_hours"),
        "new_source_forecast_hours": new_source.get(
            "source_forecast_hours"),
        "suffix_source_forecast_hours": suffix_source_hours,
    }
    manifest_extension = _json_copy(source_manifest_extension)
    if (not isinstance(manifest_extension, dict)
            or any(manifest_extension.get(key) != value
                   for key, value in expected_manifest_extension.items())
            or not isinstance(manifest_extension.get("retained_entries"), int)
            or manifest_extension["retained_entries"] <= 0
            or not isinstance(manifest_extension.get("added_entries"), list)
            or not manifest_extension["added_entries"]):
        raise PreparedCacheMismatchError(
            "prepared-cache source-manifest prefix proof is malformed or "
            "belongs to another extension")

    prior_boundaries = _reader_boundaries(prior_reader)
    suffix_boundaries = _reader_boundaries(suffix_reader)
    if len(suffix_boundaries.intervals) != 1:
        raise PreparedCacheMismatchError(
            "prepared-cache suffix must contain one boundary interval")
    old_prefix = _forcing_prefix(prior_boundaries)
    if old_prefix != prior_meta.get("lateral_boundary_prefix"):
        raise PreparedCacheMismatchError(
            "predecessor cache forcing prefix differs from its seal")
    old_end = float(len(old_hours) - 1) * 3600.0
    if float(prior_boundaries.intervals[-1].end_seconds) != old_end:
        raise PreparedCacheMismatchError(
            "predecessor forcing is not sealed at its advertised endpoint")
    suffix_interval = suffix_boundaries.intervals[0]
    if (float(suffix_interval.start_seconds) != 0.0
            or float(suffix_interval.end_seconds) != 3600.0):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix interval is not one hour")
    if (suffix_boundaries.spec_bdy_width
            != prior_boundaries.spec_bdy_width
            or suffix_boundaries.spec_zone != prior_boundaries.spec_zone
            or suffix_boundaries.relax_zone != prior_boundaries.relax_zone
            or set(suffix_interval.fields)
            != set(prior_boundaries.intervals[-1].fields)):
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes boundary geometry or fields")
    # Moved in time only, so the frame its tendency was built toward is
    # the suffix's own record.
    shifted = record_built_end_frame(BoundaryInterval(
        old_end, old_end + 3600.0, dict(suffix_interval.fields)),
        suffix_interval.end_frame_sha256)
    combined = LateralBoundaries(
        prior_boundaries.intervals + (shifted,),
        prior_boundaries.spec_bdy_width, prior_boundaries.spec_zone,
        prior_boundaries.relax_zone)
    combined_prefix = _forcing_prefix(combined)
    # The predecessor's last row records the frame its tendency was built
    # toward, so a suffix that starts from that frame joins it byte for
    # byte, a hydrometeor that clears out at the join included (A140b).
    # The rebuilt identity of a cache written before 2.8.1 cannot tell a
    # clear-out from another frame, so its refusal says which it is.
    if (old_prefix["intervals"][-1]["end_frame_sha256"]
            != combined_prefix["intervals"][-1]["start_frame_sha256"]):
        rebuilt = combined_prefix["schema"] in REBUILT_END_FRAME_PREFIX_SCHEMAS
        raise PreparedCacheMismatchError(
            "prepared-cache suffix changes the shared endpoint frame"
            + ("" if not rebuilt else
               " as the rebuilt end-frame identity sees it (value + "
               "tendency * duration in FP32), which a boundary value "
               "clearing out at the join also moves; the predecessor or the "
               "suffix records no built end frame (prepared before 2.8.1), "
               "so prepare both again to extend across such a join"))

    combined_lbc = _json_copy(prior_meta["lbc"])
    suffix_row = {
        "start_seconds": old_end,
        "end_seconds": old_end + 3600.0,
        "fields": list(prior_meta["lbc"]["intervals"][-1]["fields"]),
    }
    if shifted.end_frame_sha256 is not None:
        suffix_row[END_FRAME_KEY] = shifted.end_frame_sha256
    combined_lbc["intervals"].append(suffix_row)
    cache_metadata = _json_copy(prior_meta)
    cache_metadata.update({
        "user": _json_copy(metadata),
        "lbc": combined_lbc,
        "setup_fingerprint": None,
        "forcing_extension_mode": SEALED_PREPARED_EXTENSION_MODE,
        "lateral_boundary_prefix": combined_prefix,
        "bridge_manifest_extension": bridge_extension,
        "source_manifest_extension": manifest_extension,
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _prepared_cache_staging_path(path)
    temporary.mkdir()
    writer = _BundleWriter(temporary)
    try:
        writer.expect_reuse(prior_reader, prior_reader.arrays)
        for key in sorted(prior_reader.arrays):
            writer.link_verified(key, prior_reader)
        old_count = len(prior_boundaries.intervals)
        for key in sorted(suffix_reader.arrays):
            if not key.startswith("lbc/0/"):
                continue
            output_key = f"lbc/{old_count}/" + key[len("lbc/0/"):]
            writer.add(output_key, suffix_reader.read_array(key))
        basis = {
            "schema": PREPARED_CACHE_SCHEMA,
            "identity": new_identity,
            "metadata": cache_metadata,
            "arrays": writer.manifest,
            "payload_bytes": writer.payload_bytes,
        }
        header = {
            **basis,
            "status": "READY",
            "created_utc": _creation_utc(temporary),
            CACHE_WRITER_KEY: {"gpuwm_version": __version__},
            "content_sha256": hashlib.sha256(
                _canonical(basis).encode("utf-8")).hexdigest(),
        }
        (temporary / _HEADER_NAME).write_text(
            json.dumps(header, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        # Verify the complete staged authority after every linked or
        # copied predecessor payload and the new header exist; a concurrent
        # mutation therefore refuses publication rather than blessing bytes
        # that no longer match the inherited manifest.
        PreparedCacheReader(
            temporary, expected_identity=new_identity).verify_all()
        prior_header_after = hashlib.sha256(
            prior_header_path.read_bytes()).hexdigest()
        suffix_header_after = hashlib.sha256(
            suffix_header_path.read_bytes()).hexdigest()
        if (prior_header_after != prior_header_before
                or suffix_header_after != suffix_header_before):
            raise PreparedCacheMismatchError(
                "prepared-cache authority changed during extension")
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema": "gpuwm-prepared-cache-extension-v1",
        "status": "BUILT",
        "path": str(path.resolve()),
        "content_sha256": header["content_sha256"],
        "array_count": len(writer.manifest),
        "payload_bytes": writer.payload_bytes,
        # How the retained prefix reached the new cache: "linked" costs no
        # disk, "copied" (a drive with no hard links) costs its size.
        "predecessor_payloads": "copied" if writer.copying else "linked",
        "predecessor": {
            "path": str(predecessor.resolve()),
            "content_sha256": prior_reader.content_sha256,
            "header_sha256": prior_header_before,
        },
        "suffix": {
            "path": str(suffix.resolve()),
            "content_sha256": suffix_reader.content_sha256,
            "header_sha256": suffix_header_before,
        },
        "bridge_manifest_extension": bridge_extension,
        "bridge_manifest_sha256": new_identity["bridge_manifest_sha256"],
        "source_manifest_extension": manifest_extension,
        "appended_interval": [old_end, old_end + 3600.0],
        "sealed_prefix_sha256": hashlib.sha256(
            _canonical(combined_prefix).encode("utf-8")).hexdigest(),
    }


def reconcile_cached_state_inventory(stored_names, expected_names):
    """Return the cached state names to read, or refuse by name.

    A prepared cache carries the restart contract's serialised-state
    inventory as of the build that WROTE it, and this build's active
    configuration says what it EXPECTS.  Ordinarily those must match
    exactly: a cache from a different configuration, or from a build
    carrying state this one does not, is a different experiment.

    ONE tolerance, and it is bounded by an argument rather than by a
    version number.  A prepared cache is the t = 0 state by construction
    (``write_prepared_cache`` takes an ``initial_result``; nothing here
    carries a clock), so the dycore's advective forcing pair -- WRF
    RTHFTEN/RQVFTEN, written by an RK stage that has not run -- is
    identically zero in every cache that could contain it, which is also
    WRF's own ``start_em`` cold start.  A cache missing exactly that pair
    was therefore written by a build that had no exporter, and restoring
    the freshly allocated zeros loses nothing: the numbers are the same
    numbers.  Every prepared tree already on disk keeps working.

    A CHECKPOINT gets no such tolerance and the asymmetry is the point.
    It is mid-trajectory, the pair is genuinely non-zero there, and a
    resume that re-zeroed it would integrate a different forecast --
    ``woof.io.restart`` refuses that case by name.
    """
    from woof.state_serialization_contract import (
        ADVECTIVE_FORCING_STATE)

    stored = list(stored_names)
    expected = list(expected_names)
    if stored == expected:
        return stored
    extra = [name for name in stored if name not in expected]
    missing = [name for name in expected if name not in stored]
    tolerated = set(ADVECTIVE_FORCING_STATE)
    if (not extra and missing and set(missing) <= tolerated
            and stored == [name for name in expected
                           if name not in set(missing)]):
        return stored
    detail = []
    if missing:
        detail.append(f"the cache is missing {', '.join(missing)}")
    if extra:
        detail.append(f"the cache carries {', '.join(extra)}, "
                      "which this configuration does not allocate")
    raise PreparedCacheMismatchError(
        "prepared cache state inventory differs from the active config: "
        + "; ".join(detail)
        + ".  Prepare the case again with this build, or run it under the "
          "configuration the cache was prepared for.")


def _device_boundary_values(reader, lbc_metadata, *, lbc_mode, streamed):
    """Float32 values a restore attaches on the card as boundary tables.

    Read off the cache's own manifest -- every array under ``lbc/<i>/`` --
    so the count is the retained interval inventory the cache carries, not
    the run duration's.  A streamed restore holds one interval's slot (the
    largest); a nest forced by its parent attaches none.
    """
    if (lbc_mode == "nested-parent-forced" or not isinstance(lbc_metadata, dict)
            or not isinstance(lbc_metadata.get("intervals"), list)):
        return 0
    per_interval = [0] * len(lbc_metadata["intervals"])
    for key, spec in reader.arrays.items():
        if not key.startswith("lbc/"):
            continue
        index = key.split("/", 2)[1]
        if index.isdigit() and int(index) < len(per_interval):
            count = 1
            for extent in spec["shape"]:
                count *= int(extent)
            per_interval[int(index)] += count
    if not per_interval:
        return 0
    return max(per_interval) if streamed else sum(per_interval)


def restore_prepared_cache(path, *, expected_identity, cfg, static,
                           allow_nested_without_lbc: bool = False,
                           reader=None, boundary_source=None,
                           array_module=None,
                           scratch_arena=None, dycore_state_workspace=None,
                           ) -> RestoredPreparedCache:
    """Validate and restore an integration-ready GPU state.

    The historical/default surface restores a specified root and therefore
    requires its complete external lateral-boundary sequence.  A prepared
    hierarchy runner may explicitly opt into restoring an identity-bound
    nested child.  Such a child deliberately has no external LBC payload: its
    live :class:`woof.core.nest.NestCoupler` rebuilds rolling boundaries from
    the parent after each parent step.  The opt-in never permits a root cache
    to omit LBCs and never turns a child into a standalone root.

    ``array_module`` is :class:`DomainState`'s setup seam.  ``None`` (the
    default) restores onto the GPU for a forecast.  ``numpy`` restores a
    host setup state for a caller that only PREPARES from the root, such
    as the HRRR hierarchy stage: that stage builds the children on the CPU
    whatever the machine has, and a CUDA import there refused every
    HRRR domain tree on a CPU-only install although nothing in it needs
    a card.  A NumPy state is never a valid forecast input.

    Optional scratch and dycore workspaces are the original tree builder's
    owners. Passing them keeps restored member roots on that same ledger;
    omitted owners retain the historical constructor call.
    """
    # Resolve the pure ownership decision before importing the optional CUDA
    # runtime.  This keeps malformed caller contracts deterministic on CPU-
    # only installations and makes the hierarchy exception directly testable.
    if reader is None:
        reader = PreparedCacheReader(path, expected_identity=expected_identity)
    metadata = reader.header["metadata"]
    lbc_mode = _restore_lbc_mode(
        lbc_metadata=metadata["lbc"], identity=expected_identity,
        allow_nested_without_lbc=allow_nested_without_lbc)
    if array_module is None:
        import cupy as xp
    elif array_module is np:
        xp = np
    else:
        raise TypeError("array_module must be None (CUDA) or numpy")

    from woof.core.grid import BaseState, VerticalCoord
    from woof.core.state import DomainState
    from woof.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
        attach_lateral_boundaries, record_built_end_frame,
    )
    from woof.state_serialization_contract import (
        lateral_boundary_prefix_identity, serialized_state_attrs,
        setup_core_fingerprint, setup_fingerprint,
    )

    coord_shapes = {
        name: reader.arrays[f"coord/{name}"]["shape"]
        for name in metadata["coord_arrays"]
        if f"coord/{name}" in reader.arrays
    }
    try:
        validate_coordinate_shapes(
            coord_shapes, nz=cfg.nz, context="prepared-cache restore")
    except ValueError as exc:
        raise PreparedCacheMismatchError(str(exc)) from exc

    coord_values = dict(metadata["coord_scalars"])
    for name in metadata["coord_arrays"]:
        coord_values[name] = reader.read_array(f"coord/{name}")
    coord = VerticalCoord(**coord_values)

    base_values = dict(metadata["base_scalars"])
    for name in metadata["base_arrays"]:
        base_values[name] = reader.read_array(f"base/{name}")
    base = BaseState(**base_values)
    try:
        validate_explicit_eta_grid(
            coord.znw,
            nz=cfg.nz,
            p_top=base.p_top,
            context="prepared-cache restore",
        )
    except (TypeError, ValueError) as exc:
        raise PreparedCacheMismatchError(str(exc)) from exc

    if array_module is None:
        # ADMITTED BEFORE THE CONSTRUCTOR, whatever [tiles] says.  A resident
        # restore went straight into DomainState, so a cache too big for the
        # card stopped in a CUDA out-of-memory part way through it; the state
        # and the boundary tables this restore attaches are priced here, at
        # the retained interval count the cache actually carries.
        from woof.core.resident_admission import admit_construction

        admit_construction(
            "restoring this prepared cache onto the card", cfg,
            boundary_values=_device_boundary_values(
                reader, metadata.get("lbc"), lbc_mode=lbc_mode,
                streamed=boundary_source is not None))
    state = (DomainState(cfg, array_module=array_module)
             if scratch_arena is None and dycore_state_workspace is None else
             DomainState(cfg, scratch_arena=scratch_arena,
                         dycore_state_workspace=dycore_state_workspace,
                         array_module=array_module))
    state.load_base(coord, base)
    state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"], sina=static["SINALPHA"],
        cosa=static["COSALPHA"], msfv_inv=static.get("MF_VX_INV"))
    expected_state_names = [
        name for name in serialized_state_attrs(state)
        if getattr(state, name, None) is not None]
    stored_state_names = reconcile_cached_state_inventory(
        metadata["state_names"], expected_state_names)
    for name in stored_state_names:
        host = reader.read_array(f"state/{name}")
        target = getattr(state, name)
        if tuple(host.shape) != tuple(target.shape) or host.dtype != target.dtype:
            raise PreparedCacheMismatchError(
                f"prepared cache state/{name} shape or dtype differs from "
                "the active config")
        target[...] = xp.asarray(host)
    # Strict mode reads theta as WRF's t = theta - 300; a no-op otherwise.
    from woof.core.state import adopt_wrf_theta_carrier
    adopt_wrf_theta_carrier(state)

    lbc_meta = metadata["lbc"]
    streamed = boundary_source is not None
    if lbc_mode == "nested-parent-forced":
        boundaries = None
    elif streamed:
        # ``boundary_source`` streams the root's intervals from a prepared
        # tree (woof.ingest.boundary_stream): one device slot, reloaded
        # at each interval seam from host FP32 exactly as the eager attach
        # converts it, so nothing here waits for an interval the model has
        # not reached.
        from woof.ingest.lateral_bc import (
            attach_streaming_lateral_boundaries)

        boundaries = boundary_source
        if [[float(start), float(end)] for start, end
                in getattr(boundaries.intervals, "bounds", ())] != [
                [float(row["start_seconds"]), float(row["end_seconds"])]
                for row in lbc_meta["intervals"]]:
            raise PreparedCacheMismatchError(
                "streamed boundary schedule differs from the prepared cache")
        attach_streaming_lateral_boundaries(state, boundaries)
    else:
        intervals = []
        for index, interval_meta in enumerate(lbc_meta["intervals"]):
            field_map = {}
            for name in interval_meta["fields"]:
                sides = {}
                for side_name in ("west", "east", "south", "north"):
                    prefix = f"lbc/{index}/{name}/{side_name}"
                    sides[side_name] = _reader_boundary_side(reader, prefix)
                field_map[name] = FieldBoundary(**sides)
            intervals.append(record_built_end_frame(BoundaryInterval(
                float(interval_meta["start_seconds"]),
                float(interval_meta["end_seconds"]), field_map),
                interval_built_end_frame(
                    interval_meta,
                    where=f"prepared cache LBC interval {index}")))
        boundaries = LateralBoundaries(
            tuple(intervals), int(lbc_meta["spec_bdy_width"]),
            int(lbc_meta["spec_zone"]), int(lbc_meta["relax_zone"]))
        attach_lateral_boundaries(state, boundaries)
    if streamed:
        # The whole setup fingerprint needs every interval; at the head it
        # is checked on its immutable core, and the full fingerprint is
        # checked against the sealed header once every interval exists.
        observed_setup = None
        expected_core = getattr(reader, "setup_core_fingerprint", None)
        if expected_core is None:
            expected_core = metadata.get("setup_core_fingerprint")
        if (expected_core is not None
                and setup_core_fingerprint(state) != expected_core):
            raise PreparedCacheMismatchError(
                "prepared head reconstructed a different immutable setup "
                "core")
    elif metadata.get("forcing_extension_mode") == \
            SEALED_PREPARED_EXTENSION_MODE:
        observed_setup = setup_fingerprint(state)
        if (setup_core_fingerprint(state)
                != metadata.get("setup_core_fingerprint")):
            raise PreparedCacheMismatchError(
                "sealed prepared cache reconstructed a different immutable "
                "setup core")
        if (lateral_boundary_prefix_identity(state)
                != metadata.get("lateral_boundary_prefix")):
            raise PreparedCacheMismatchError(
                "sealed prepared cache reconstructed a different forcing "
                "inventory")
    else:
        observed_setup = setup_fingerprint(state)
        if observed_setup != metadata["setup_fingerprint"]:
            raise PreparedCacheMismatchError(
                "prepared cache reconstructed a different setup fingerprint")

    met_fields = {
        name: reader.read_array(f"met/{name}")
        for name in metadata["met_fields"]}
    surface_names = metadata.get("surface_fields", [])
    surface_fields = {
        name: reader.read_array(f"surface/{name}")
        for name in surface_names}
    result = CachedInitialResult(
        state=state, coord=coord, base=base,
        surface_pressure=reader.read_array("result/surface_pressure"),
        surface_qv=reader.read_array("result/surface_qv"),
        hydrometeor_initialization=MappingProxyType(
            metadata.get("hydrometeor_initialization", {})))
    receipt = {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "RESTORED",
        "path": str(reader.path.resolve()),
        "content_sha256": reader.content_sha256,
        "array_count": len(reader.arrays),
        "payload_bytes": reader.payload_bytes,
        "setup_fingerprint": observed_setup,
    }
    return RestoredPreparedCache(
        initial_result=result,
        met=SimpleNamespace(fields=MappingProxyType(met_fields)),
        surface=(SimpleNamespace(fields=MappingProxyType(surface_fields))
                 if surface_fields else None),
        boundaries=boundaries,
        metadata=MappingProxyType(metadata["user"]),
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CACHE_WRITER_KEY", "CachedInitialResult", "END_FRAME_KEY",
    "HEADER_PARTIAL_NAME",
    "DEFAULT_TOLERANT_IDENTITY_FIELDS",
    "INERT_DIAGNOSTIC_IDENTITY_FIELDS",
    "NON_TRAJECTORY_IDENTITY_FIELDS", "PREPARATION_INERT_RUN_FIELDS",
    "PREPARED_CACHE_SCHEMA",
    "PreparedCacheCorruptError", "PreparedCacheMismatchError",
    "PreparedCacheResourceLimitError",
    "PreparedCacheReader", "PreparedHeadReader", "RestoredPreparedCache",
    "SEALED_PREPARED_EXTENSION_MODE", "STRICT_IDENTITY_FIELDS",
    "UNSTAMPED_WRITER",
    "cache_writer_version", "compare_prepared_domain_config",
    "compare_prepared_identity", "effective_prepared_domain_config",
    "interval_built_end_frame",
    "prepared_cache_identity",
    "prepared_domain_config_identity", "prepared_identity_refusal",
    "PreparedCacheStream", "boundary_schedule", "read_manifest_array",
    "reconcile_cached_state_inventory",
    "extend_prepared_cache", "restore_prepared_cache",
    "select_prepared_met_fields", "undelayed_identity_defaults",
    "write_prepared_cache",
]
