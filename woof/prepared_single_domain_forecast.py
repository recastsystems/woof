#!/usr/bin/env python3
"""Run one GFS/ERA5/20CRv3 RW-WPS prepared cache through GPUWM.

The runner accepts a portable direct single-domain output or the external-LBC
d01 bundle in a portable hierarchy published by the public GFS and ERA5
adapters.  It reconstructs the cache identity from the caller-pinned proof,
source manifest, static geometry, experiment, and WPS namelist before importing
CuPy.  WRF-shaped history files are published at the exact cadence bound into
the experiment through woof's existing atomic writer and carry
``GPUWM_WRITE_COMPLETE=1``.
"""

from __future__ import annotations

from woof.physics_registry import canonical_template_id

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import tomllib
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from woof.core import streaming  # noqa: E402
from woof import __version__  # noqa: E402
from woof.config import EPSSM_AUTO, radiation_scheme_ids  # noqa: E402
from woof.core.nssl2_contract import (  # noqa: E402
    CONTRACT_ID as NSSL2_CONTRACT_ID,
    MP_PHYSICS as NSSL2_MP_PHYSICS,
    nssl2_contract_receipt,
    resolve_nssl2_mode_for_config,
)
from woof.core.thompson_contract import (  # noqa: E402
    AUXILIARY_TABLE_RECORDS as THOMPSON_AUXILIARY_TABLE_RECORDS,
    CLASSIC_TABLE_ASSETS as THOMPSON_CLASSIC_TABLE_ASSETS,
    GENERATED_TABLE_FILES as THOMPSON_GENERATED_TABLE_FILES,
    MP_PHYSICS as THOMPSON_MP_PHYSICS,
    TABLE_SET_ID as THOMPSON_TABLE_SET_ID,
    TRANSPORTED_SPECIES as THOMPSON_TRANSPORTED_SPECIES,
    WRF_REFERENCE_COMMIT as THOMPSON_WRF_REFERENCE_COMMIT,
    WRF_REFERENCE_VERSION as THOMPSON_WRF_REFERENCE_VERSION,
    validate_table_assets as validate_thompson_table_assets,
)
from woof.aerosol_source_receipt import (  # noqa: E402
    AEROSOL_SOURCE_KEY,
    aerosol_source_report_entry,
)
from woof.experiment import build_experiment, load_experiment  # noqa: E402
from woof.vertical_adaptation import (  # noqa: E402
    adopt_prepared_vertical, prepared_domain_coordinate_refusal)
from woof.acoustic_adaptation import (  # noqa: E402
    acoustic_receipt, adapt_experiment_to_terrain, readings_from_static)
from woof.terrain_clock import (  # noqa: E402
    clock_for_prepared_cache, clock_receipt)
from woof.explain import (  # noqa: E402
    add_explain_flag, explain_enabled, layered, render as render_explanation,
    warn,
)
from woof.kernel_compile_notice import (  # noqa: E402
    ARCHITECTURE_MISSING, COLD_CACHE, COMPILING_STATUS,
    current_compute_capability, kernel_cache_state, scan_kernel_cache,
)
from woof.ingest.prepared_cache import (  # noqa: E402
    CONDITIONAL_PREPARATION_RECEIPTS,
    SOIL_PREPARATION_RECEIPTS,
    PREPARED_CACHE_SCHEMA,
    PreparedCacheReader,
    PreparedHeadReader,
    prepared_cache_identity,
    undelayed_identity_defaults,
)
from woof.native_wrf_contract import (  # noqa: E402
    NATIVE_LANDUSE_IDENTITY,
    load_native_static_cache,
    native_landuse_identity,
    validate_native_lambert_contract,
    validate_native_lambert_contracts,
    verify_native_static_receipt,
)
from woof.progress_log import (  # noqa: E402
    ProgressOptions, add_progress_arguments)
from woof.ingest.memory_refusal import InitializationMemoryRefused  # noqa: E402
from woof.receipt_paths import receipt_basename  # noqa: E402
from woof.supervisor import atomic_write_json  # noqa: E402
from woof.physics_compat import (  # noqa: E402
    COMPOSITION_SUITE_PROFILE_IDS,
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    MULTI_DOMAIN_SELECTION_SCHEMA,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    PhysicsCapabilityError,
    RUC_PROFILE_ID,
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_TABLE_ROOT_ENV,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_SUBSTITUTION_TOKENS,
    WSM6_PROFILE_ID,
    acknowledgement_delivery,
    identify_single_domain_profile,
    physics_selection_differences,
    selection_values_one_spelling,
    single_domain_physics_selection,
    single_domain_runtime_switches,
    single_domain_verification_status,
    thompson_runtime_requirements,
    thompson_table_root,
    validate_single_domain_physics_profile,
)
from woof.certify.capsule import emit_run_capsule  # noqa: E402
from woof.physics_profile_siblings import with_profile_siblings
from woof.spectral_seam import (  # noqa: E402
    seam_capsule_receipts as _seam_capsule_receipts,
)
from woof.source_adapters import packaged_profile_sources  # noqa: E402
# Re-exported, not called here: `tests/test_prepared_single_domain_forecast`
# reads the GRIB2 profile's pins through this module because this module is
# where the certificate that enforces them lives.
from woof.source_authorities import (packaged_profile,  # noqa: E402,F401
                                      twentycrv3_authority_sha256)
from woof.table_assets import require_thompson_tables  # noqa: E402
from woof.wrf_physics_inventory import (  # noqa: E402
    stock_wrf_physics_inventory,
)


REPORT_SCHEMA = "gpuwm-prepared-single-domain-forecast-v1"
PROGRESS_SCHEMA = "gpuwm-prepared-single-domain-progress-v1"
RUNNER_CAPABILITIES_SCHEMA = "gpuwm-runner-capabilities-v1"
AUTHORITY_MATERIALIZATION_SCHEMA = (
    "gpuwm-named-source-physics-authorities-v1")
PHYSICS_PROFILE = WSM6_PROFILE_ID
KESSLER_PHYSICS_PROFILE = KESSLER_PROFILE_ID
TWENTYCRV3_WSM6_PHYSICS_PROFILE = (
    "20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1")
THOMPSON_PHYSICS_PROFILE = THOMPSON_PROFILE_ID
MORRISON_PHYSICS_PROFILE = MORRISON_PROFILE_ID
NSSL2_PHYSICS_PROFILE = NSSL2_PROFILE_ID
NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE = NSSL2_LEGACY_RRTMG_PROFILE_ID
RUC_PHYSICS_PROFILE = RUC_PROFILE_ID
MYNN_RUC_PHYSICS_PROFILE = MYNN_RUC_PROFILE_ID
MYNN_PHYSICS_PROFILE = MYNN_PROFILE_ID
NOAHMP_PHYSICS_PROFILE = NOAHMP_PROFILE_ID
MYNN_NOAHMP_PHYSICS_PROFILE = MYNN_NOAHMP_PROFILE_ID
MYNN_RTE_RRTMGP_PHYSICS_PROFILE = MYNN_RTE_RRTMGP_PROFILE_ID
MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE = MYNN_RUC_RTE_RRTMGP_PROFILE_ID
MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE = MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID
THOMPSON_MYNN_RUC_DUDHIA_PHYSICS_PROFILE = THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID
THOMPSON_MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE = (
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
PHYSICS_PROFILES = (
    PHYSICS_PROFILE,
    KESSLER_PHYSICS_PROFILE,
    TWENTYCRV3_WSM6_PHYSICS_PROFILE,
    THOMPSON_PHYSICS_PROFILE,
    MORRISON_PHYSICS_PROFILE,
    NSSL2_PHYSICS_PROFILE,
    NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
    MYNN_PHYSICS_PROFILE,
    MYNN_RTE_RRTMGP_PHYSICS_PROFILE,
    RUC_PHYSICS_PROFILE,
    MYNN_RUC_PHYSICS_PROFILE,
    THOMPSON_MYNN_RUC_DUDHIA_PHYSICS_PROFILE,
    MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE,
    NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_PHYSICS_PROFILE,
    MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE,
)
#: Sources prepared through the declarative mapped route against a
#: PACKAGED profile -- source id -> the profile id
#: :mod:`woof.source_authorities` pins.
#:
#: These share one certificate: the bundle's copied mapping, composition
#: and provenance must be byte-equal to the three documents this
#: distribution ships for that profile.  What differs between them is the
#: SHAPE OF THEIR INPUT MANIFEST, and only that: `20crv3` carries the
#: every-member GRIB2 manifest with its filename-bound member identity,
#: while any composed mapped preparation carries
#: `gpuwm-mapped-composition-inputs-v1`.  Both shapes are validated below;
#: neither pin is relaxed for the other.
#:
#: A row here is what makes a packaged source runnable end to end, and it
#: is a row -- not an arm.  A profile whose composition role is a PENDING
#: declaration (an atmosphere-only source awaiting its cross-source
#: land-surface donor) can never have prepared a bundle, so it holds no
#: row: including it would promise this stage an arm the profile itself
#: refuses to supply.
#: Sentinel for "this identity document has no such key at all", so a
#: difference report can say `absent` rather than `None`, which is a value
#: a field can legitimately hold.
_MISSING = object()

from woof.prepared_source_schemas import composed_packaged_profiles, source_schemas
_MAPPED_PACKAGED_PROFILE = composed_packaged_profiles()
_MAPPED_SOURCES = frozenset({"mapped", *_MAPPED_PACKAGED_PROFILE})
SUPPORTED_SOURCES = frozenset({"gfs", "era5", "hrrr"} | _MAPPED_SOURCES)

#: The physics-profile name that means "no named profile: bind the
#: experiment config's own physics selection" -- the ``profile_binding``
#: value the authority and physics receipts already write for that case.
#: A caller that must carry the binding as a string (a controller whose
#: argument record is replayed by its workers, and whose ensemble identity
#: names the profile) passes this name and :func:`preflight_prepared_forecast`
#: binds it as ``None``.  Breakage this prevents: the packed DA controller
#: mapped the name to ``None`` for its own preflight, so its workers read
#: ``null`` from the shared argument record and rebuilt the ensemble
#: identity with profile ``"None"`` while the controller's said
#: ``"experiment-config"``; every member leg was refused ("worker ensemble
#: identity differs from the public prepared binding", a rented box 2026-10-06
#: 18:51Z) and the cycle never ran a forecast.
EXPERIMENT_CONFIG_PROFILE = "experiment-config"


def physics_profile_binding(physics_profile):
    """``None`` for :data:`EXPERIMENT_CONFIG_PROFILE` (bind the experiment
    config's own physics), the name itself for a named profile."""
    if physics_profile is None or physics_profile == EXPERIMENT_CONFIG_PROFILE:
        return None
    return physics_profile

#: The HRRR bundle this runner reads is the one
#: ``tools/prepare_hrrr_wrf.py`` publishes, and it is NOT the portable
#: single-domain layout the other sources share.  Its artifacts keep the
#: names the native HRRR route has always written -- the preparation is
#: the certified one and renaming its outputs to suit a reader would put
#: the certification at risk for nothing -- so the reader learns the
#: layout instead.  Everything hash-bound about it is identical: the
#: same prepared-cache format, the same identity recomputation, the same
#: fail-closed comparison against the caller's pins.
HRRR_DIRECT_LAYOUT = "hrrr-native-direct-v1"

#: Where an HRRR bundle keeps each artifact, relative to --prepared-root.
HRRR_BUNDLE_PATHS = MappingProxyType({
    "static": "native-static.npz",
    "geometry_receipt": "native-geometry-receipt.json",
    "prepared_cache": "native/prepared-cache",
    "bridge_manifest": "native/native-bridge/SHA256SUMS",
})

#: ``mp_physics`` values whose microphysics call stages a scheme-native
#: REFL_10CM field, i.e. exactly ``woof.runtime.REFL_10CM_MICROPHYSICS``.
#: Duplicated rather than imported: ``woof.runtime`` pulls in the GRIB,
#: static-geography and ERA5 ingest stack, and this runner deliberately
#: imports none of it.  The duplication is held equal by
#: ``tests/test_mp28_runtime_reachability.py``, which imports both and
#: asserts they are the same set -- and which fails on ANY surviving
#: ``mp_physics in (1, 6, 8, 10, 18)`` literal anywhere under ``gpuwm/``,
#: so a fifth copy of this gate cannot be added silently.
REFL_10CM_MICROPHYSICS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 16, 18, 24, 28, 50, 106) + tuple(
    __import__('woof.microphysics_schemes', fromlist=['NAMED_MP_IDS']).NAMED_MP_IDS)
#: What a PACKAGED source reports when nothing has been measured on it:
#: the suites the generic mapped route carries for every packaged profile,
#: with the row's own limitation saying no source-specific verification is
#: claimed.  This is the route's contract, not a claim about the source.
#:
#: It used to be the empty tuple, and an empty tuple is NOT the
#: conservative reading of "nothing measured here".  tools/build_registry.py
#: completes a served source's ``source_template_ids`` from this evidence
#: row, and that key is the REACHABILITY declaration a front end reads;
#: ``woof.physics_registry.expert_template_ids_for_source`` then withholds
#: the route-wide expert list from any source whose normal list is empty.
#: So the empty row published "this source reaches no named suite at all"
#: about packaged sources this runner supports: all six composition
#: suites and all three Noah-MP expert suites disappeared, and
#: a plan naming one was warned ``template-route-evidence`` instead of
#: being handed the acknowledgement advisory the option exists to raise.
#: The plan still launched, and that is the whole defect: the warning
#: says the template is off-route for this source, which was false.
#: A source with no measurement of its own is PRICED from the most
#: conservative basis on record, which is the one every one of its
#: siblings on this route already reports.
_GENERIC_MAPPED_PROFILES = (
    PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
    MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
    NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE)
_VERIFIED_SOURCE_PHYSICS_PROFILES = {
    # A supported source must report a row of its own: it may not disappear
    # from the capability inventory, inherit another source's evidence, or
    # be emptied for want of a measurement.  A packaged profile with no row
    # written below is a packaged profile on the generic mapped route and
    # reports what that route reports; a new one in
    # woof.prepared_source_schemas is therefore table work here too, with
    # no second table to remember.
    **{source: (_GENERIC_MAPPED_PROFILES
                if source in _MAPPED_PACKAGED_PROFILE else ())
       for source in sorted(SUPPORTED_SOURCES)},
    "mapped": (),  # No packaged profile stands behind it: the composition is the caller's.
    # REPORTED METADATA, NOT A GATE (owner ruling 2026-07-31): these
    # per-source lists name the shipped profiles whose verification
    # evidence this runner can vouch for on each source.  They feed the
    # capabilities receipt and the registry drift check; nothing refuses
    # a suite for being absent from them.  The one real per-source
    # blocker they used to encode -- RUC absent from GFS because a
    # GFS-initialised RUC forecast prepared in full and then died on its
    # first surface-temperature call with `mavail must be finite` -- was
    # root-caused, not re-gated: a shoreline land column carrying the
    # water soil category (SOILTYP 14 under a land LU_INDEX) reached RUC's
    # soilvegin, which has no arm for it, and `0./0.` went into MAVAIL.
    # real.exe reconciles that column at initialization
    # (module_initialize_real.F:3108-3131) and so does every ArWen door
    # now, through woof/ingest/soil.py:door_reconciled_soil_category
    # (proven both ways by tests/test_ruc_shoreline_soil_category.py).
    # No route blocker remains and none is enforced below (ENG-009).
    # Each radiation-bearing MYNN twin follows the MYNN row it mirrors,
    # in the same order the registry route declares it: the drift check in
    # tests/test_physics_registry.py compares these lists to the route's
    # own, element for element.
    "gfs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE,
        NOAHMP_PHYSICS_PROFILE, MYNN_NOAHMP_PHYSICS_PROFILE,
        MYNN_NOAHMP_RTE_RRTMGP_PHYSICS_PROFILE),
    "era5": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE,
        RUC_PHYSICS_PROFILE, MYNN_RUC_PHYSICS_PROFILE,
        # Each Thompson member of the MYNN + RUC pair follows the WSM6 row
        # it moves the microphysics of, as the registry route declares it.
        THOMPSON_MYNN_RUC_DUDHIA_PHYSICS_PROFILE,
        MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE,
        THOMPSON_MYNN_RUC_RTE_RRTMGP_PHYSICS_PROFILE,
        "thompson-mp8-mynn-mynn-ruc-monthly-rrtmg-legacy-v1",
        # The monthly-albedo twin with alb_sol 1 (lane/286-albsol) and the
        # GSD MYNN 4.1 Thompson-aerosol suite (lane/286-fork-mynn), as the
        # registry route declares them for this source.
        "thompson-mp8-mynn-mynn-ruc-monthly-solar-rrtmg-legacy-v1",
        "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"),
    "20crv3": (
        TWENTYCRV3_WSM6_PHYSICS_PROFILE, PHYSICS_PROFILE,
        THOMPSON_PHYSICS_PROFILE, MORRISON_PHYSICS_PROFILE,
        NSSL2_PHYSICS_PROFILE, NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE),
    # The NetCDF profile decodes the same 20CRv3 model on the same
    # pressure-level/soil contract, so it reports the same suites.  These
    # rows are REPORTED METADATA, not a gate (see the note above), so this
    # is a statement about what has been verified, not a permission.
    "20crv3-cf": (
        TWENTYCRV3_WSM6_PHYSICS_PROFILE, PHYSICS_PROFILE,
        THOMPSON_PHYSICS_PROFILE, MORRISON_PHYSICS_PROFILE,
        NSSL2_PHYSICS_PROFILE, NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
        MYNN_PHYSICS_PROFILE, MYNN_RTE_RRTMGP_PHYSICS_PROFILE),
    # HRRR's own stock-WRF gate is the WSM6/YSU/MM5-91/Noah slice, and
    # woof/hrrr_route_inputs.py's SUPPORTED_MICROPHYSICS admits the
    # same scheme set the other sources report here.
    "hrrr": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged pressure-level profile decodes the same HRRR model
    # through the generic mapped route, so it reports the same verified
    # suite set the native route does (reported metadata, not a gate).
    "hrrr-prs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # One verified GEFS member through the same generic mapped route
    # (reported metadata, not a gate).
    "gefs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged GDPS profile is the same generic mapped route again
    # (reported metadata, not a gate).
    "gem-gdps": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged AIFS single profile initialises the same WSM6/YSU/
    # MM5-91/Noah slice from ECMWF's AI forecast (reported metadata, not
    # a gate); no suite has AIFS-initialised verification evidence yet.
    "aifs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged AI-ensemble member-hybrid profile is the same generic
    # mapped route again (reported metadata, not a gate); no suite has
    # member-initialised verification evidence yet.
    "aigefs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged GDAS pgrb2 profile is the GFS field catalogue through
    # the same generic mapped route (reported metadata, not a gate).
    "gdas": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged ECMWF open-data profile shares the mapped route's
    # WSM6/YSU/MM5-91/Noah contract (reported metadata, not a gate).
    "ecmwf-open-data": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # The packaged ICON-EU regular-lat-lon profile runs the same generic
    # mapped route, so it reports the same verified suite set the other
    # packaged pressure profiles do (reported metadata, not a gate).
    "icon-eu": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # RAP through the same packaged mapped route: the generic suites,
    # with no RAP-verified subset to prefer (reported metadata, not a
    # gate; the route is not stock-WRF certified and says so).
    "rap": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # RRFS through the same packaged mapped route: the generic suites,
    # with no RRFS-verified subset to prefer (reported metadata, not a
    # gate; the route is not stock-WRF certified and says so).
    "rrfs": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
    # An RRFS ensemble member with its declared companion is the same
    # packaged mapped route (reported metadata, not a gate).
    "rrfs-ens": (
        PHYSICS_PROFILE, THOMPSON_PHYSICS_PROFILE,
        MORRISON_PHYSICS_PROFILE, NSSL2_PHYSICS_PROFILE,
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE),
}

#: The three EXPERT suites of the table above.  The registry declares them
#: ROUTE-WIDE (``expert_template_ids`` under
#: ``woof.physics_registry.EXPERT_TEMPLATES_ANY_SOURCE``), so every source
#: this runner supports offers them and none of the rows above lists one by
#: hand.  They were written for ``gfs`` alone while this runner supports
#: eighteen sources, and a route that declares any expert list is
#: EXHAUSTIVE -- so on an ERA5 single domain the three were undeclared:
#: nothing offered them, and a plan that named one was told the template was
#: off-route instead of being handed the acknowledgement advisory the option
#: exists to raise.  What gates them is the route's expert acknowledgement
#: and Noah-MP's own evidence warnings, which are source-independent
#: statements: the scheme is implemented-unverified everywhere, not
#: unverified on one source and verified on another.  Appended per source
#: rather than written into each tuple so the offer cannot drift source by
#: source again.
_EXPERT_PROFILE_IDS = (
    NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
)
#: The per-source lists this runner reports: the verification rows above,
#: then the COMPOSITION suites (audit R-067), then the expert rows.
#: They are appended to EVERY source that names any suite at all, in one
#: order, because none of them reads anything source-specific: each is an
#: implemented option's only named front door, and leaving them on the
#: domain-tree route alone left SASE, the three large-eddy closures, WDM6
#: with Grell-Freitas and mp9 with New Tiedtke with no named suite on this
#: route for any source.  The expert rows stay LAST because the registry
#: keeps them in a separate ``expert_template_ids`` list that a reader
#: concatenates after the normal one, which is what makes the two orders
#: comparable element for element -- the drift check in
#: tests/test_physics_registry.py compares them that way.
#:
#: `mapped` keeps its empty tuple: it names no suite at all because the
#: composition is the caller's, and a source with no first suite gets
#: neither a composition suite nor an expert one here (which is also how
#: ``expert_template_ids_for_source`` resolves a source whose declared
#: template list is empty).
# Source-independent compositions with component evidence and no
# source-qualified WRF trajectory. The registry declares their maturity.
_GENERIC_IMPLEMENTED_PROFILE_IDS = (
    "lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1",
) + COMPOSITION_SUITE_PROFILE_IDS
_SOURCE_PHYSICS_PROFILES = MappingProxyType({
    source_id: (
        with_profile_siblings(p for p in profiles if p not in _EXPERT_PROFILE_IDS)
        + _GENERIC_IMPLEMENTED_PROFILE_IDS
        + _EXPERT_PROFILE_IDS
        if profiles else profiles)
    for source_id, profiles in _VERIFIED_SOURCE_PHYSICS_PROFILES.items()
})
# RETIRED, with a measurement.  This template carried a hand-typed
# twenty-three-value runtime row here because the shared single-domain
# menu was one other route's declared list and this template is not on
# it, so ``single_domain_runtime_switches`` refused it by name.  The menu
# is derived from every fixed-template route's own declaration now, this
# template is declared on THIS route, and the derivation reproduces the
# hand-typed row exactly -- all twenty-three keys, all twenty-three
# values, no key present in one and absent from the other.  A second copy
# of a table the registry already states can only drift from it.
#: The switch names an experiment-config-selected (custom) suite's
#: receipt records: the union of what the shipped profile rows pin,
#: in one canonical order, so two receipts for one config are
#: byte-identical.
_SUITE_RECEIPT_SWITCH_KEYS = (
    "moist", "moist_cq", "mp_physics", "top_lid", "epssm",
    "morr_rimed_ice", "wsm6_hail_opt", "ra_physics",
    "ra_lw_physics", "ra_sw_physics", "radt",
    "wrf_rrtmg_compatibility", "ra_rrtmg_variant",
    "sf_sfclay_physics", "sf_surface_physics",
    "bl_pbl_physics", "cu_physics", "cudt_minutes",
    "num_soil_layers", "terrain_opt", "km_opt", "diff_6th_opt",
    "diff_6th_factor", "diff_6th_slopeopt",
)
#: The keys a NAMED profile is allowed to own in the materialized
#: authority -- i.e. every key whose value the generated experiment.toml
#: may state on the profile's behalf.  It is deliberately WIDER than any
#: one profile's switch table (`radt_minutes` is a compatibility
#: spelling of `radt`, `ra_rrtmg_variant` is pinned only by the
#: RRTMG-family profiles), because it doubles as the mask
#: :func:`_without_materialized_physics` applies to BOTH sides of the
#: non-physics descriptor digest: a key that one profile pins and
#: another does not must be outside that digest either way, or the
#: "descriptors unchanged" proof would read a physics choice as a
#: descriptor change.
#:
#: Membership here does NOT license deleting a value the user wrote.
#: What a given materialization may rewrite is the narrower
#: :func:`_profile_pinned_physics` map, and even that only after
#: :func:`_declared_physics_conflicts` has proved the config agrees.
_MATERIALIZED_PHYSICS_KEYS = frozenset({
    'oml_gamma', 'oml_hml0', 'oml_relaxation_time', 'sf_ocean_physics',
    "moist", "moist_cq", "mp_physics", "top_lid", "epssm",
    "morr_rimed_ice", "wsm6_hail_opt", "ra_physics",
    "ra_lw_physics", "ra_sw_physics", "radt", "radt_minutes",
    "wrf_rrtmg_compatibility", "ra_rrtmg_variant", "sf_sfclay_physics",
    "sf_surface_physics", "bl_pbl_physics", "cu_physics",
    "cudt_minutes", "num_soil_layers", "terrain_opt", "km_opt",
    "diff_6th_opt", "diff_6th_factor", "diff_6th_slopeopt",
    "nest_microphysics_transition",
    # The urban canopy selector every profile states since the urban models
    # joined the registry (none = 0); a physics switch, not a descriptor.
    "sf_urban_physics", "use_wudapt_lcz", "num_urban_hi",
    # RUC mosaic selectors and MYNN scalar diffusion, stated by the
    # profiles that select those schemes since 2.8.5.
    "mosaic_lu", "mosaic_soil", "scalar_pblmix",
})
#: The nest-transition rule every materialized authority states.  No
#: profile switch table carries it, so it is pinned here, once, and read
#: from here by both the emitter and the agreement check -- a second
#: literal would let the two disagree about what the profile "is".
_NEST_MICROPHYSICS_TRANSITION = "same-scheme-only"
#: The three keys that spell ONE selection.  ``ra_lw_physics =
#: ra_sw_physics = -1`` is documented in woof/config.py as preserving
#: the historical ``ra_physics`` aggregate exactly, and explicit pairs
#: require ``ra_physics = 0``, so a config and a profile can name the
#: same two schemes through different keys.  They are compared as a
#: resolved pair, never key by key; see :func:`_declared_physics_conflicts`.
_RADIATION_SELECTION_KEYS = frozenset({
    "ra_physics", "ra_lw_physics", "ra_sw_physics"})
# The four per-source lookups below, and the two legacy companions
# after them, are DERIVED for the mapped rows and then overridden for
# the routes that genuinely differ.  They used to be literal id lists,
# and every composed packaged profile spelled the identical value in
# all six -- so adding a registry row meant remembering six more edits,
# and forgetting one surfaced as a bare `KeyError` inside the stage
# that runs a prepared bundle rather than as a refusal a user could
# act on.  The arbitrary acceptance test says a new model is table
# work; this is the seam where that used to stop being true.
#
# The overrides are the whole of what is NOT generic: `gfs`, `era5` and
# `hrrr` have their own direct runners and their own document schemas,
# and `20crv3` writes its OWN input-manifest schema because its member
# identity is sealed there -- a difference the stage reads to decide
# whether to demand a member manifest, so it must survive.
_SOURCE_SCHEMA = source_schemas(_MAPPED_SOURCES)
_PROOF_SCHEMA = {
    **{source: "gpuwm-mapped-direct-wrf-proof-v1"
       for source in _MAPPED_SOURCES},
    "gfs": "gpuwm-gfs-direct-wrf-proof-v3",
    "era5": "gpuwm-era5-direct-wrf-proof-v2",
    "hrrr": "gpuwm-hrrr-native-direct-wrf-proof-v1",
}
_LEGACY_PROOF_SCHEMAS = {
    # The mapped proof has never had an earlier revision, so no mapped
    # source accepts a legacy document.
    **{source: frozenset() for source in _MAPPED_SOURCES},
    # v2 remains independently verifiable.  It predates the explicit
    # front-door physics selection receipt and therefore cannot be promoted
    # to v3 by inference.
    "gfs": frozenset({"gpuwm-gfs-direct-wrf-proof-v2"}),
    "era5": frozenset(),
    # New in this runner as of the DA background lane: there is no
    # earlier HRRR bundle for it to have to accept.
    "hrrr": frozenset(),
}
_HIERARCHY_PROOF_SCHEMA = {
    **{source: "gpuwm-mapped-native-hierarchy-proof-v1"
       for source in _MAPPED_SOURCES},
    "gfs": "gpuwm-gfs-native-hierarchy-proof-v2",
    "era5": "gpuwm-era5-native-hierarchy-proof-v1",
    # HRRR's multi-domain route is woof.hrrr_hierarchy_direct feeding
    # woof.prepared_domain_tree_forecast, a designed division of labour
    # this lane does not widen.  Naming a schema no HRRR preparation
    # writes keeps _resolve_prepared_layout's generic path from matching
    # by accident; the explicit refusal below is what a caller sees.
    "hrrr": "gpuwm-hrrr-native-hierarchy-proof-unreachable-here",
}
_LEGACY_HIERARCHY_PROOF_SCHEMAS = {
    **{source: frozenset() for source in _MAPPED_SOURCES},
    # v1 predates the front-door physics receipt the v2 hierarchy proof
    # carries and cannot be promoted to it by inference, the same rule
    # the direct proof's v2 lives under.
    "gfs": frozenset({"gpuwm-gfs-native-hierarchy-proof-v1"}),
    "era5": frozenset(),
    "hrrr": frozenset(),
}
#: The EXACT top-level inventory of the two documents
#: :mod:`woof.mapped_direct` publishes -- the mapped route's direct
#: proof and its hierarchy proof.  ``_validate_packaged_mapped_evidence``
#: compares ``set(proof)`` against these and refuses on any difference,
#: which is right: a proof carrying a key this reader does not know
#: about is a document from a preparation this reader cannot vouch for.
#:
#: They live here, named, for one reason.  When these lists were spelled
#: inline inside the validator, the 2.5.0 soil-mesh work added
#: ``soil_texture_downscale`` to BOTH proofs in ``woof/mapped_direct.py``
#: and nothing here changed -- so on this line the forecast runner
#: rejected every mapped bundle the preparation stage produced, with
#: "mapped 20CRv3 proof top-level inventory differs", and the whole
#: mapped route's last leg was dead.  The test fixture did not catch it
#: because the fixture wrote its own proof rather than the writer's.
#: ``tests/test_prepared_single_domain_forecast.py`` now parses
#: ``woof/mapped_direct.py``'s own proof literals and asserts they equal
#: these sets, so the next key added to the writer fails a test instead
#: of a user's run.
MAPPED_DIRECT_PROOF_KEYS = frozenset({
    "schema", "status", "stock_wrf_export", "forcing_times", "soil_texture_downscale",
    "forcing_hours", "boundary_interval_seconds", "execution_inputs",
    "source_composition", "preprocessing", "static", "geometry",
    "moisture_floors",
    # The effective hybrid coordinate and the derivation behind it.  The
    # forecast ADOPTS the coordinate from this key, so a reader that
    # discarded it would run the configured etac against coefficient
    # arrays built for another one.
    "vertical_coordinate",
    "prepared_cache", "export", "timing_seconds", "proof_content_sha256",
    "preparation_parallelism", "forcing_stage_timings",
    # The prepared head the seal names (chained preparation).  A bundle
    # prepared before chaining carries none and is still accepted.
    "boundary_stream",
})
MAPPED_HIERARCHY_PROOF_KEYS = frozenset({
    "schema", "status", "stock_wrf_export", "domain_count", "forcing_times",
    "soil_texture_downscale", "forcing_hours", "vertical_coordinate",
    "boundary_interval_seconds", "target_contract", "execution_inputs",
    "source_composition", "preprocessing", "hierarchy_workers",
    "root_static", "root_geometry", "static_catalog", "source_coverage",
    "artifact_receipt", "moisture_floors_by_domain", "wrf_manifest",
    "timing_seconds", "proof_content_sha256",
    # The prepared head a chained tree's seal names.  A tree prepared in
    # one piece (chaining off, or a CUDA tree) carries none and is still
    # accepted.
    "boundary_stream",
})
#: The moisture-floor receipt keys, by document.  Named as a pair because
#: the discard below has to know which one a given schema expects, and
#: because "the key this release added" is the thing a reader of the
#: discard needs told.
MAPPED_MOISTURE_FLOOR_KEYS = frozenset({
    "moisture_floors", "moisture_floors_by_domain"})
#: The effective-vertical-coordinate receipt, on the same footing and for
#: the same reason: a bundle prepared before this release carries no such
#: key, and its forecast is still a valid preparation -- one that could
#: only ever have used the configured etac, which is exactly what
#: :func:`woof.vertical_adaptation.adopt_prepared_vertical` does with a
#: document that has none.  A bundle that HAS it keeps the exactness,
#: because the key is in the required sets above.
MAPPED_VERTICAL_COORDINATE_KEYS = frozenset({"vertical_coordinate"})
#: Completed at the direct preparation's seal. Older preparations have no
#: worker/stage measurements; their scientific identity remains readable.
MAPPED_PREPARATION_TELEMETRY_KEYS = frozenset({
    "preparation_parallelism", "forcing_stage_timings"})
#: Top-level proof keys the writer publishes ONLY when that preparation
#: performed the recorded operation, so the runner takes documents with them and
#: without them.  ``woof/mapped_direct.py`` spreads these in
#: conditionally -- ``**({...} if <receipt> is not None else {})`` --
#: which is why they cannot sit in the required sets above: a mapped
#: bundle prepared without the corridor carries byte-for-byte the proof
#: it always did, and every retained hash stays valid.  Requiring them
#: would refuse that bundle; ignoring them would refuse the opted-in one,
#: and the exactness for every OTHER key is what this inventory is for.
#: ``source_vertical_ladder`` is the same shape of key: written only when
#: the source files carried fewer levels than the mapping declares (an
#: era ladder), so a full-ladder bundle's proof is unchanged.
#: ``soil_temperature_repair`` likewise: written only when real.exe's TSLB
#: reasonableness rebuild, or the snow-covered rebuild beside it, touched
#: a land column of the root. ``soil_moisture_floor`` is present only when
#: the soil moisture floor touched a column, with the same exact receipt
#: bound in cache user metadata by CONDITIONAL_PREPARATION_RECEIPTS.
#: ``initial_perturbation`` likewise: written
#: only when the experiment carries a [perturbation] block, which the
#: prepared runner applies at start (woof.experiment
#: deferred_initial_perturbation).
#: ``posting`` likewise: written only by a preparation made as its leads
#: posted (DESIGN A136 2.4 item 6), the leads it waited for.
#: ``source_pairing`` likewise: written only when the root is its source's
#: own declared grid and pairs cell for cell (lane/286-fixed-step-grid,
#: horiz.declared_grid_pairing); without it here the runner refused every
#: such bundle as an unknown top-level inventory.
MAPPED_OPTIONAL_PROOF_KEYS = frozenset({
    # The mapped writer records the existing preparation-time land floor.
    # Its absence remains valid; rejecting the present receipt prevents a
    # correctly prepared land column from reaching the forecast reader.
    "soil_moisture_floor",
    "statics_corridor", "source_vertical_ladder", "soil_temperature_repair",
    "soil_moisture_floor",
    "initial_perturbation", "posting", "initial_source", "source_pairing"})
_SOURCE_ADAPTER = {
    # The generic mapped adapter, truthfully: a packaged profile is
    # prepared by `woof.mapped_direct` with nothing model-specific in
    # the code path, so the adapter string is the mapped one and WHICH
    # profile it was is bound where it actually lives -- the mapping and
    # composition digests, checked byte for byte above.  Writing a
    # per-model adapter id here would be a per-model label on a route
    # that has no per-model half.
    **{source: "rw-wps-mapped-composition-v2"
       for source in _MAPPED_SOURCES},
    "gfs": "gfs-pgrb2-0p25-direct-v1",
    "era5": "era5-grib1-direct-v1",
    # The one mapped route whose adapter id IS model-specific, because
    # its member identity is sealed by a packaged member manifest.
    "20crv3": "rw-wps-20crv3-member-grib2-v1",
    # The adapter identity woof/source_adapters.py declares for HRRR.
    "hrrr": "hrrr-native-state-v1",
}
_DECODER_IMPLEMENTATION = {
    "gfs": "gpuwm-all-rust-gfs-grib2-bridge",
    "era5": "gpuwm-all-rust-grib1-bridge",
}
_LANDUSE_IDENTITY = MappingProxyType(dict(NATIVE_LANDUSE_IDENTITY))
_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})
_REQUIRED_MET_FIELDS = frozenset({
    "LANDSEA", "SKINTEMP", "T2", "U10", "V10",
})
#: The dynamical fields every external LBC interval carries.  The scalar
#: half (``qv``, and for an aerosol-aware ``mp_physics = 28`` the two
#: aerosol tracers the preparation read in) comes from
#: :func:`woof.boundary_fields.external_scalar_fields`, the same function
#: the preparation writes the inventory from; a hand-typed six-field
#: constant here refused every mp=28 cache as "differs from forcing".
_LBC_DYNAMICS_FIELDS = ("mu", "phi", "theta", "u", "v")
#: The inventory every scheme other than an aerosol-reading mp=28 carries,
#: in the cache's sorted spelling; kept as the name test fixtures build
#: their headers from.
_LBC_FIELDS = tuple(sorted((*_LBC_DYNAMICS_FIELDS, "qv")))


def _lbc_field_inventories(cfg) -> tuple[list[str], ...]:
    """The LBC field lists a prepared cache may legitimately carry for ``cfg``.

    One list for most schemes.  For ``mp_physics = 28`` two: the preparation
    resolves whether aerosols were read from the dataset when it runs, and
    the cache records the outcome in its inventory, so both outcomes are
    admitted here and the aerosol receipt beside them says which happened.
    """
    from types import SimpleNamespace

    from woof.boundary_fields import external_scalar_fields

    # The four attributes the inventory reads.  A caller that hands a
    # boundary-width view rather than a RunConfig (the receipt validators
    # do) is priced as the moist, non-aerosol run every earlier cache was,
    # which is the six-field inventory this check used to hard-code.
    scheme = SimpleNamespace(
        moist=bool(getattr(cfg, "moist", True)),
        mp_physics=int(getattr(cfg, "mp_physics", 0)),
        aer_init_opt=int(getattr(cfg, "aer_init_opt", 0)),
        mp28_aerosol_source=getattr(cfg, "mp28_aerosol_source", "auto"))
    inventories = []
    for aerosol_from_input in (False, True):
        fields = sorted(
            (*_LBC_DYNAMICS_FIELDS,
             *external_scalar_fields(scheme, aerosol_from_input=aerosol_from_input)))
        if fields not in inventories:
            inventories.append(fields)
    return tuple(inventories)
_HEX = frozenset("0123456789abcdef")
_TWENTYCRV3_SOURCE = "NOAA-CIRES-DOE 20CRv3 every-member GRIB2"
_TWENTYCRV3_MEMBER_IDENTITY = "filename_memNNN_not_grib2_pdt"
_TWENTYCRV3_FILENAME = re.compile(
    r"^mem(?P<member>[0-9]{3})_(?P<time>[0-9]{10})_"
    r"(?P<role>pl|sfc)\.grb2$")
#: The two decode-route inventories a 20CRv3 member preparation can have
#: sealed, exactly: the in-process engine on the bare default, or the
#: subprocess pair on the documented Python-engine workaround.  The
#: member manifest itself declares no decoder section (it predates the
#: generic manifest and stays the route's user-facing authority), so the
#: pin here is "one of the two shipped decode routes, in full" -- never a
#: partial inventory, never a role from each.
_TWENTYCRV3_DECODER_ROLE_SETS = (
    frozenset({"gpuwm_mapped_engine"}),
    frozenset({"grib2_inventory", "grib2_dump"}),
)
_DIRECT_LAYOUTS = frozenset({
    "portable-single-domain-v2", "mapped-direct-d01-v1",
    HRRR_DIRECT_LAYOUT})
_HIERARCHY_LAYOUTS = frozenset({
    "hierarchy-d01-v1", "mapped-hierarchy-d01-v1"})
_THOMPSON_IMPLEMENTATION_FILES = (
    "woof/core/microphysics.py",
    "woof/core/refl.py",
    "woof/core/thompson.py",
    "woof/core/thompson_contract.py",
    "woof/core/thompson_runtime.py",
    "woof/core/kernels/refl.cu",
    "woof/core/kernels/thompson.cu",
)
_FORECAST_EXECUTOR_MODULES = (
    "woof.core.clock",
    "woof.core.dycore",
    "woof.core.health",
    "woof.core.model",
    "woof.core.refl",
    "woof.io.wrfout",
    "woof.state_digest",
)


def _missing_forecast_executor_modules() -> list[str]:
    missing = []
    for module in _FORECAST_EXECUTOR_MODULES:
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(module)
    return missing


def runner_capabilities() -> dict[str, object]:
    """Return the side-effect-free contract consumed by Studio doctor."""

    missing_executor_modules = _missing_forecast_executor_modules()
    forecast_available = not missing_executor_modules
    thompson = thompson_runtime_requirements()
    # The key name physics_profile_ids predates the 2026-07-31 owner
    # ruling, when these per-source lists were the ADMISSION lists.  The
    # key survives for existing consumers, but its meaning is now the
    # verification evidence this runner can vouch for on each source,
    # and each entry says so beside the list rather than relying on a
    # consumer to find the top-level physics_admission block.
    source_profile_ids_semantics = (
        "per-source verification rows plus source-independent "
        "implemented-unverified named compositions with their declared "
        "component evidence limits; the latter claim no coupled WRF trajectory. "
        "NOT an admission list -- any engine-valid suite runs from the "
        "hash-bound experiment config on every supported source (see "
        "physics_admission)")
    sources = {
        "mapped": {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": ["mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": [],
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "authority_binding": "caller mapping/composition/input receipts and sealed cache identity",
            "limitations": ["Forecast skill depends on the supplied data and selected physics; no model-specific verification is claimed."],
        },
        "gfs": {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": [
                "portable-single-domain-v2", "hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(_SOURCE_PHYSICS_PROFILES["gfs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
        },
        "era5": {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": [
                "portable-single-domain-v2", "hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(_SOURCE_PHYSICS_PROFILES["era5"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
        },
        "20crv3": {
            "readiness": (
                "NATIVE_EXACT_MEMBER_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "member_identity": _TWENTYCRV3_MEMBER_IDENTITY,
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["20crv3"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "prepared source is hash-bound to one filename member",
                "GPU forecast execution has no public 20CRv3 acceptance gate",
                "hierarchy input may be consumed only as executable d01",
            ],
        },
        "20crv3-cf": {
            "readiness": (
                "PACKAGED_NETCDF_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "member_identity": "ensemble_mean_analysis_not_a_member",
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["20crv3-cf"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "NOAA PSL's 20CRv3 NetCDF distribution is the ENSEMBLE MEAN "
                "analysis; it is smoother than any single member and is not "
                "one of the 80 trajectories",
                "orography and land mask are recovered from 20CRv3's own "
                "published fields and supplied as a provenance-bound "
                "supplement, because PSL publishes neither",
                "GPU forecast execution has no public 20CRv3 acceptance gate",
            ],
        },
        "hrrr-prs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["hrrr-prs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from HRRR's pressure-level wrfprs product, "
                "which is smoother near terrain than the native hybrid "
                "levels the certified --source hrrr route consumes",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "icon-eu": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["icon-eu"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from DWD's regular-lat-lon ICON-EU product "
                "set (field-per-file bz2 GRIB2); pressure-level moisture "
                "is derived from relative humidity because DWD publishes "
                "no specific humidity there",
                "FR_LAND and HSURF are once-per-cycle invariants, "
                "broadcast after an invariance proof; sentinel-valued "
                "ocean soil is masked from the mapping's own land "
                "fraction",
                "the native icosahedral ICON global product set (GDT "
                "101) is refused with the grid family named",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "icon-d2": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["icon-d2"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from DWD's native ICON-D2 mesh (GDT 101), "
                "remapped by gdt101_remap onto the WPS domain plus a "
                "0.25-degree halo at 0.02 degrees; pressure-level moisture "
                "is derived from relative humidity",
                "DWD's ICON-D2 pressure ladder stops at 200 hPa, so the "
                "model top can be no higher than 200 hPa",
                "the domain must lie inside lat 44.0..57.3, lon 0.0..17.3, "
                "where every cell the remap reads is published",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "gdas": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["gdas"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from the hourly f000..f009 pgrb2.0p25 set "
                "-- the field-complete analysis-cycle files, which the "
                "bytes stamp forecasts even at hour 0; the one "
                "analysis-stamped product (pgrb2.1p00.anl) publishes no "
                "soil, no land mask and no 2 m/10 m state and is not an "
                "initialization route",
                "publication lags the cycle by about seven hours (the "
                "delayed-cutoff assimilation cycle); a scheduler "
                "assuming GFS timing chases a 404",
                "the run stops at f009 -- GDAS is a background "
                "trajectory for the next assimilation window, not a "
                "public forecast product",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "gefs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["gefs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises ONE verified ensemble member from its "
                "pgrb2a+pgrb2b pair (the two level sets are exactly "
                "disjoint, so both files of the same member are "
                "mandatory per valid time); stage and verify the member "
                "with woof-member-prep first",
                "ensemble mean/spread files sharing the member "
                "directories are refused at the byte level by the "
                "PDT-1 selector pins",
                "snow is published only under a 55-percent ocean "
                "bitmap and stays policy-controlled, so runs "
                "initialise snow-free",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "gem-gdps": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["gem-gdps"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from GDPS's 15 km pressure-level product; "
                "the soil column is the single 0-10 cm ISBA layer the "
                "product publishes, anchored by skin and deep-soil "
                "temperature",
                "orography, land mask and ice analysis are the "
                "once-per-cycle analysis invariants, broadcast after an "
                "invariance proof",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "aifs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["aifs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "the published soil column reaches 0.28 m; deeper Noah "
                "layers are WRF's shallow-column interpolation anchored "
                "by the skin and static deep-soil temperatures",
                "no snow state and no sea-ice fraction are published, so "
                "runs initialise bare-ground and open-water everywhere",
                "an AI emulator's fields carry no hydrometeors and no "
                "dynamical-balance constraint",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "aigefs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["aigefs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "the member product publishes NO land-surface state: "
                "soil, land mask, orography, skin temperature and 2 m "
                "humidity are borrowed from the same cycle's physical "
                "analysis through the cross-source composition and "
                "held at their analysis values across every lead",
                "no snow state and no sea-ice fraction are published "
                "by either contributor's borrowed set, so runs "
                "initialise bare-ground and open-water everywhere",
                "an AI emulator's fields carry no hydrometeors and no "
                "dynamical-balance constraint",
                "member identity is carried by the woof-member-prep "
                "receipt, not by the byte-identical leaf filenames",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "ecmwf-open-data": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["ecmwf-open-data"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from the 0.25-degree open-data distribution "
                "(CC-BY-4.0); ECMWF's native 9 km HRES is "
                "access-restricted, a data-licensing fact rather than a "
                "capability gap",
                "snow and sea-ice fields are policy-controlled: the "
                "open data publishes them only as local-table/"
                "bitmap-masked records whose semantics are not yet bound",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "rap": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["rap"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from RAP's 32 km awip32 pressure-level "
                "product; the 13 km CONUS products are not reachable as "
                "tables (awp130pgrb has no soil state, and the native "
                "wrfprs grid is rotated lat-lon GDT 32769, outside the "
                "declared grid families)",
                "not yet accepted by unchanged stock WRF",
            ],
        },
        "rrfs": {
            "readiness": (
                "PACKAGED_GRIB2_PROFILE_PREPARATION_COMPLETE_"
                "GPU_FORECAST_NOT_ACCEPTANCE_GATED"),
            "prepared_layouts": [
                "mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(
                _SOURCE_PHYSICS_PROFILES["rrfs"]),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "limitations": [
                "initialises from the 3 km CONUS prslev+2dfld pair; "
                "prslev alone carries no surface fields and no soil, so "
                "both products must be supplied per valid time",
                "the natlev native-level product, the 3 km North-America "
                "rotated grid and every per-member ensemble file exist "
                "only in the frozen prototype bucket with no live front "
                "door, and are not claimed by this route",
                "not yet accepted by unchanged stock WRF",
            ],
        },
    }
    sources["hrrr"] = {
        "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
        "prepared_layouts": [HRRR_DIRECT_LAYOUT],
        "single_d01_gpu_execution": True,
        "physics_profile_ids": list(_SOURCE_PHYSICS_PROFILES["hrrr"]),
        "physics_profile_ids_semantics": source_profile_ids_semantics,
    }
    for source in sorted(SUPPORTED_SOURCES - sources.keys()):
        if source not in _MAPPED_SOURCES:
            raise RuntimeError(f"supported source {source!r} lacks a capability contract")
        sources[source] = {
            "readiness": "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED",
            "prepared_layouts": ["mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"],
            "single_d01_gpu_execution": True,
            "physics_profile_ids": list(_SOURCE_PHYSICS_PROFILES.get(source, ())),
            "physics_profile_ids_semantics": source_profile_ids_semantics,
            "authority_binding": "packaged mapping/composition/input receipts and sealed cache identity",
            "limitations": ["Runtime preflight remains required; no source-specific WRF verification is claimed."],
        }
    physics_profiles = {
        PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "SUPPORTED_RUNNER_PROFILE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
        },
        KESSLER_PHYSICS_PROFILE: {
            "selector": 1,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "Kessler mp1 has no woof/WRF forecast trajectory "
                "comparison on this runner's prepared sources"),
        },
        TWENTYCRV3_WSM6_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "source_scope": ["20crv3"],
            "warning": (
                "GPU execution is implemented but has no public 20CRv3 "
                "forecast acceptance gate"),
        },
        THOMPSON_PHYSICS_PROFILE: {
            "selector": THOMPSON_MP_PHYSICS,
            **thompson,
        },
        MORRISON_PHYSICS_PROFILE: {
            "selector": 10,
            "readiness": "WRF_MATCHED_RUN_RUNTIME_PROFILE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
        },
        NSSL2_PHYSICS_PROFILE: {
            "selector": NSSL2_MP_PHYSICS,
            "readiness": "WRF_MATCHED_RUN_CANDIDATE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
            "external_table_assets": [],
            "contract_id": NSSL2_CONTRACT_ID,
            "resolved_fixed_preset": True,
        },
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE: {
            "selector": NSSL2_MP_PHYSICS,
            "readiness": "WRF_MATCHED_RUN_CANDIDATE",
            "explicit_expert_consent_required": False,
            "runtime_guards": [],
            "external_table_assets": [],
            "contract_id": NSSL2_CONTRACT_ID,
            "radiation_solver": "legacy RRTMG",
            "resolved_fixed_preset": True,
        },
        MYNN_PHYSICS_PROFILE: {
            # Same microphysics selector as PHYSICS_PROFILE: what this profile
            # selects is the coupled MYNN 5/5 surface-layer/PBL pair, and
            # woof/config.py validate_run_config refuses half of it.
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "MYNN 5/5 runs a 300-step coupled forecast with a bitwise "
                "restart but has no woof/WRF trajectory comparison; the CUDA "
                "leaves also differ from the CPU references by up to 137 ULP "
                "in the diffusivities away from their oracle fixtures"),
        },
        RUC_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "RUC LSM runs with its nine-layer soil state but has no "
                "woof/WRF forecast trajectory comparison"),
        },
        MYNN_RUC_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "MYNN/MYNN/RUC follows WRF's ownership sequence but has no "
                "woof/WRF forecast trajectory comparison"),
        },
        NOAHMP_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED_EXPERT",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [
                "measured 360,000-column ceiling or explicit expert budget",
                "glacier columns refused",
            ],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "Noah-MP runs on the device but has no woof/WRF forecast "
                "trajectory comparison"),
        },
        MYNN_NOAHMP_PHYSICS_PROFILE: {
            "selector": 6,
            "readiness": "IMPLEMENTED_UNVERIFIED_EXPERT",
            "explicit_expert_consent_required": False,
            "explicit_profile_selection_required": True,
            "runtime_guards": [
                "measured 360,000-column ceiling or explicit expert budget",
                "glacier columns refused",
            ],
            "external_table_assets": [],
            "resolved_fixed_preset": True,
            "warning": (
                "MYNN/MYNN/Noah-MP follows WRF's ownership sequence but has "
                "no woof/WRF forecast trajectory comparison"),
        },
    }
    return {
        "schema": RUNNER_CAPABILITIES_SCHEMA,
        "runner": "tools.prepared_single_domain_forecast",
        "supported_sources": (
            sorted(SUPPORTED_SOURCES) if forecast_available else []),
        "physics_profile_ids": (
            list(PHYSICS_PROFILES) if forecast_available else []),
        # Owner ruling 2026-07-31: the ids above are the shipped suites
        # whose WRF-verification evidence this runner reports; they are
        # not an admission list.  Any suite the engine implements runs
        # from the hash-bound experiment config, --physics-profile is an
        # optional exactness assertion, and verification status is
        # receipt metadata, never a gate.
        "physics_admission": {
            "mode": "any-engine-valid-suite-from-hash-bound-experiment",
            "physics_profile_flag": "optional-exactness-assertion",
            "verification_status": "reported-never-gating",
        },
        "report_schema": REPORT_SCHEMA,
        "progress_schema": PROGRESS_SCHEMA,
        "readiness": (
            "FORECAST_IMPLEMENTATION_PRESENT_RUNTIME_PREFLIGHT_REQUIRED"
            if forecast_available
            else "FORECAST_EXECUTOR_OMITTED"
        ),
        "modes": {
            "forecast": {
                "available": forecast_available,
                "availability_scope": "executor-module-presence-only",
                "launch_ready": None,
                "launch_readiness_check": (
                    "validate CuPy, CUDA, GPU allocation, prepared authorities, "
                    "and profile guards before launch"
                ),
                "requires_cupy": True,
                "requires_compatible_cuda_gpu": True,
                "missing_executor_modules": missing_executor_modules,
                "included_in_standalone_rw_wps_wheel": False,
                "unavailable_reason": (
                    None
                    if forecast_available
                    else "GPUWM forecast executor modules are absent"
                ),
            },
        },
        "standalone_rw_wps_wheel": {
            "runner_included": False,
            "forecast_executor_included": False,
            "reason": "the standalone wheel is preprocessing/export only",
        },
        "source_profiles": sources if forecast_available else {},
        "physics_profiles": physics_profiles if forecast_available else {},
        "guards": {
            "claim_output_directory_create_only": True,
            "single_specified_non_nested_d01": True,
            "feedback_required": 0,
            "restart_interval_seconds_required": None,
            "hash_bound_inputs": [
                "proof",
                "source-manifest",
                "mapped-source-evidence",
                "prepared-cache-content",
                "experiment-config",
                "wps-namelist",
            ],
        },
        "window": {
            "kind": "model-relative-prepared-cache-forcing-hours",
            "minimum_frame_count": 2,
            "maximum_source_forecast_hour": None,
            "maximum_run_seconds": None,
            "limit_policy": "prepared-cache-forcing-coverage",
            "model_forcing_must_start_at_zero": True,
            "run_seconds": {
                "finite_positive_required": True,
                "whole_hour_required": True,
                "must_equal_hash_bound_experiment": False,
                "requested_length_executed": True,
                "override_mechanism": RUN_LENGTH_OVERRIDE_MECHANISM,
            },
            "source_forcing_cadence_hours": {
                "gfs": [1, 3],
                "era5": "uniform-positive-whole-hour",
                "20crv3": "manifest-bound-uniform-positive-whole-hour",
                "20crv3-cf": "uniform-positive-whole-hour",
                "hrrr-prs": "uniform-positive-whole-hour",
                "hrrr-native": "uniform-positive-whole-hour",
                "gdas": "uniform-positive-whole-hour",
                "gefs": "uniform-positive-whole-hour",
                "gem-gdps": "uniform-positive-whole-hour",
                "aifs": "uniform-positive-whole-hour",
                "aigefs": "uniform-positive-whole-hour",
                "aigfs": "uniform-positive-whole-hour",
                "ecmwf-open-data": "uniform-positive-whole-hour",
                "icon-eu": "uniform-positive-whole-hour",
                "icon-d2": "uniform-positive-whole-hour",
                "rap": "uniform-positive-whole-hour",
                "rap-native": "uniform-positive-whole-hour",
                "rrfs": "uniform-positive-whole-hour",
            },
            "forcing_must_cover_run": True,
        },
        "output": {
            "io_modes": ["history"],
            "history_interval_seconds": {
                "explicit_cli_value_required": True,
                "finite_positive_required": True,
                "must_equal_hash_bound_experiment": True,
                "must_be_whole_model_steps": True,
                "must_evenly_divide_run": False,
                "schedule_policy": (
                    "initial-and-floor-multiples-at-or-before-run-end"),
                "initial_frame_required": True,
                "frame_at_run_end_required": False,
                "last_scheduled_frame_may_precede_run_end": True,
            },
            "configurable_cadence": True,
            "restart_output": True,
        },
        "capability_query": {
            "flag": "--show-capabilities",
            "side_effect_free": True,
            "requires_cupy": False,
            "validates_gpu_or_runtime_assets": False,
        },
        "authority_materialization": {
            "available": True,
            "mode_flag": "--materialize-authorities",
            "requires_cupy": False,
            "create_only_output_directory": True,
            "atomic_file_publication": True,
            "inputs": [
                "source", "base-experiment-config", "base-wps-namelist",
                "physics-profile", "output-directory",
            ],
            "outputs": {
                "experiment_config": "experiment.toml",
                "wps_namelist": "namelist.wps",
                "receipt": "authority-receipt.json",
            },
            "receipt_schema": AUTHORITY_MATERIALIZATION_SCHEMA,
            "preserved_controls": [
                "domain topology and horizontal geometry",
                "experiment start and duration",
                "history and restart cadence",
                "explicit eta levels, p_top, hybrid_opt, and etac",
                "WPS namelist bytes",
            ],
            "source_physics_profile_ids": {
                source: list(profiles)
                for source, profiles in _SOURCE_PHYSICS_PROFILES.items()
            },
            "source_physics_profile_ids_semantics": (
                source_profile_ids_semantics),
        },
    }


@dataclass(frozen=True)
class PreparedForecastInputs:
    source: str
    layout: str
    prepared_root: Path
    domain_bundle_path: Path
    proof_path: Path
    source_manifest_path: Path
    static_path: Path
    geometry_receipt_path: Path
    prepared_cache_path: Path
    experiment_config: Path
    wps_namelist: Path
    proof: Mapping[str, object]
    source_manifest: Mapping[str, object]
    geometry_receipt: Mapping[str, object]
    cache_identity: Mapping[str, object]
    cache_identity_compatibility: Mapping[str, object]
    cache_reader: PreparedCacheReader
    experiment: object
    grid: object
    static: Mapping[str, np.ndarray]
    landuse_identity: Mapping[str, object]
    forcing_hours: tuple[int, ...]
    boundary_interval_seconds: int
    physics_receipt: Mapping[str, object]
    export_source_receipt: Mapping[str, object]
    #: The bound source-manifest identity plus the fetch level provenance
    #: it carries, for sources that publish one (GFS today).  Recorded so
    #: an acceptance harness can read back WHICH pressure ladder the run's
    #: inputs came off, rather than inferring it from a constant.
    source_manifest_receipt: Mapping[str, object] | None
    file_sha256: Mapping[str, str]
    authority_paths: Mapping[str, Path]
    source_domain_count: int
    source_member: str | None
    #: The verified ``boundary-stream/head.json`` when this forecast was
    #: bound to a prepared HEAD (``--prepared-head-sha256``) rather than to
    #: a sealed proof: the checks only the seal can answer are made there
    #: (:func:`_seal_streamed_inputs`).  ``None`` for a sealed binding.
    stream_head: Mapping[str, object] | None = None
    #: The preflight's own arguments, kept so the seal can run the same
    #: preflight again against the sealed tree.
    preflight_arguments: Mapping[str, object] | None = None
    #: What the long-step derivation read before it chose this run's
    #: clock, kept for a head-bound run only
    #: (:class:`woof.ingest.boundary_stream.ClockBasis`).
    clock_basis: object | None = None
    #: Ephemeral immutable mappings verified by this sealed direct preflight.
    #: These change no run authority or serialized receipt identity.
    cache_payload: object | None = field(default=None, repr=False, compare=False)

    @property
    def execution_plan(self):
        """This run's execution plan, as the prepared-tree inputs carry it.

        The preflight records it on the physics receipt; the ensemble
        session reads it off the inputs to admit the native member pack
        and to stamp its receipts, on both roads alike.
        """
        receipt = self.physics_receipt
        return receipt.get("execution_plan") if isinstance(receipt, Mapping) else None


@dataclass(frozen=True)
class _PreparedLayout:
    kind: str
    domain_bundle: Path
    static_path: Path
    geometry_receipt_path: Path
    prepared_cache_path: Path
    authority_paths: Mapping[str, Path]
    domain_receipt: Mapping[str, object] | None = None
    hierarchy_receipt: Mapping[str, object] | None = None
    artifact_manifest: Mapping[str, object] | None = None
    wrf_manifest: Mapping[str, object] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False)


def _history_period_count(run_seconds: float, cadence_seconds: float) -> int:
    """Return complete history periods due at or before model stop."""

    run_seconds = float(run_seconds)
    cadence_seconds = float(cadence_seconds)
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError("run-seconds must be finite and positive")
    if not math.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        raise ValueError(
            "history-interval-seconds must be finite and positive")
    return Fraction(run_seconds) // Fraction(cadence_seconds)


def _history_output_schedule(
        *, start_time: datetime, run_seconds: float, cadence_seconds: float,
        domain_id: int = 1, after_seconds: float | None = None,
        begin_seconds: float = 0.0, end_seconds: float | None = None,
        interval2_seconds: float = 0.0, change_seconds: float = 0.0,
) -> tuple[tuple[float, datetime, str], ...]:
    """Resolve exact model-relative offsets, valid times, and WRF filenames.

    ``begin_seconds`` / ``end_seconds`` are the domain's history window
    (WRF's history_begin / history_end), the begin already on the step
    lattice (:func:`woof.core.clock._history_window_ticks`): frames at
    begin + k * cadence, none after the end.  The defaults are the start
    frame and every cadence to the stop, the schedule every run had
    before the window existed.
    """

    begin = Fraction(float(begin_seconds))
    periods = _history_period_count(float(Fraction(float(run_seconds))
                                          - begin), cadence_seconds)         if begin else _history_period_count(run_seconds, cadence_seconds)
    cadence = Fraction(float(cadence_seconds))
    records = []
    for index in range(periods + 1):
        if (end_seconds is not None
                and begin + index * cadence > Fraction(float(end_seconds))):
            break
        offset_seconds = float(begin + index * cadence)
        if interval2_seconds and offset_seconds > change_seconds:
            break
        valid_time = start_time + timedelta(seconds=offset_seconds)
        if valid_time.microsecond != 0:
            raise ValueError(
                "history cadence produces sub-second valid times that cannot "
                "be represented by second-complete WRF history filenames")
        records.append((
            offset_seconds,
            valid_time,
            valid_time.strftime(
                f"wrfout_d{int(domain_id):02d}_%Y-%m-%d_%H_%M_%S"),
        ))
    if interval2_seconds and run_seconds > change_seconds:
        second = _history_output_schedule(
            start_time=start_time, run_seconds=run_seconds,
            cadence_seconds=interval2_seconds, domain_id=domain_id,
            begin_seconds=change_seconds + interval2_seconds,
            end_seconds=end_seconds) if change_seconds + interval2_seconds <= run_seconds else ()
        records.extend(row for row in second if row[0] >= begin_seconds)
    names = [record[2] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("history cadence produces duplicate WRF filenames")
    return tuple(record for record in records
                 if after_seconds is None or record[0] > after_seconds)


def _root_history_window(exp) -> tuple[float, float | None]:
    """The root's history window, using exact adaptive alarm offsets."""
    from woof.core.clock import _history_window_ticks

    domain = exp.root
    dt = exp.dt_exact(domain.grid_id)
    den = dt.denominator
    begin_ticks, end_ticks = _history_window_ticks(
        domain, int(dt * den), den,
        adaptive=bool(getattr(domain.run, "use_adaptive_time_step", False)))
    return (float(Fraction(begin_ticks, den)),
            None if end_ticks is None else float(Fraction(end_ticks, den)))


def _completed_history_inventory(records, schedule, *, captured_paths=None):
    """Report retained native files separately from synchronous capture."""
    if captured_paths is not None:
        if records:
            raise RuntimeError(
                "suppressed member histories unexpectedly retained native file records")
        if tuple(path.name for path in captured_paths) != tuple(row[2] for row in schedule):
            raise RuntimeError(
                "ensemble capture omitted or changed a hash-bound history handoff")
        frames = [{"name": name, "model_elapsed_seconds": offset,
                   "valid_time": valid_time.isoformat()}
                  for offset, valid_time, name in schedule]
        return [], {
            "output_mode": "ensemble_capture",
            "expected_frame_count": 0,
            "scheduled_frame_count": len(schedule),
            "initial_frame_verified": False,
            "last_scheduled_frame_verified": False,
            "initial_and_final_frames_verified": False,
            "all_frames_readback_verified": False,
            "capture": {
                "expected_frame_count": len(schedule),
                "frame_count": len(frames),
                "exact_schedule_verified": True,
                "history_files_retained": False,
                "frames": frames,
            },
        }
    inventory = [{**record, "model_elapsed_seconds": offset,
                  "valid_time": valid_time.isoformat(),
                  "atomic_writer_readback_verified": (
                      record["sha256"] is not None and record["bytes"] is not None)}
                 for record, (offset, valid_time, _name) in zip(records, schedule, strict=True)]
    return inventory, {}


def _hash_bound_history_cadences(domain) -> tuple[float, float | None]:
    """(alarm cadence, full-frame cadence or None) of one hash-bound domain.

    A domain with a quick-look stream (``quicklook = {...}``,
    :attr:`woof.experiment.DomainConfig.full_history_interval_s`) rings
    its history alarm at the quick-look cadence and writes FULL frames at
    the interval its TOML declares as ``history_interval_s``; every other
    domain has one cadence and ``None`` here.  Both values are the
    experiment's own stream, so a caller naming either one asks for the
    frames this run writes.
    """

    full = getattr(domain, "full_history_interval_s", None)
    return (float(domain.history_interval_s),
            None if full is None else float(full))


def _validate_hash_bound_history_cadence(
        exp, history_interval_seconds: float,
) -> dict[str, object]:
    """Cross-check CLI cadence against the immutable experiment authority."""

    requested = float(history_interval_seconds)
    domain = exp.root
    experiment_cadence, full_cadence = _hash_bound_history_cadences(domain)
    run_copy_cadence = float(domain.run.output_interval_s)
    # A quick-look domain's TOML declares the FULL cadence as
    # history_interval_s, and that is what an omitted flag defaults to
    # (_clock_defaults); the parsed alarm is the quick-look cadence.  Both
    # name this run's own stream (U-E3, 2026-10-26: a member-roster member
    # with --quicklook-interval 300 on 15-minute frames refused its own
    # default, 900 against the 300 s alarm, before the first step).
    declared_cadence = (experiment_cadence if full_cadence is None
                        else full_cadence)
    if requested not in (experiment_cadence, declared_cadence):
        # THE BREAKAGE THIS REFUSAL PREVENTS (open-defects ledger A15,
        # 2026-10-05): this check used to warn and run the experiment's
        # cadence anyway.  A 1 km dataset run that asked for 10-minute
        # history with --history-interval-seconds 600 on an hourly
        # experiment would have written hourly frames, five of every six
        # frames it asked for missing, with one warning line in a long log
        # as the only trace.  The hash-bound experiment cadence is the only
        # one this run can write (it is part of the prepared identity), so
        # an explicit flag that disagrees is refused before any step.  An
        # omitted flag defaults to the experiment's cadence and never
        # reaches this branch.
        raise ValueError(
            f"--history-interval-seconds {requested:g} asks for a history "
            f"frame every {requested:g} s, but the hash-bound experiment "
            f"declares history_interval_s = {declared_cadence:g} s on "
            f"d{int(domain.grid_id):02d}"
            + ("" if full_cadence is None else
               f" (with a {experiment_cadence:g} s quick-look stream)")
            + ", and that is the only cadence this "
            "prepared run can write; running on would silently deliver "
            "the experiment's frames instead of the ones asked for.  Omit "
            "the flag to run the experiment's cadence, or set "
            f"history_interval_s = {requested:g} in the experiment TOML "
            "and prepare again")
    if run_copy_cadence != experiment_cadence:
        warn(f"the experiment's derived output_interval_s "
             f"({run_copy_cadence:g} s) differs from its history_interval_s "
             f"({experiment_cadence:g} s); history_interval_s is "
             "authoritative and is used")
    exact_steps = Fraction(experiment_cadence) / exp.dt_exact(domain.grid_id)
    adaptive = bool(getattr(domain.run, "use_adaptive_time_step", False))
    if not adaptive and (exact_steps.denominator != 1 or exact_steps < 1):
        raise ValueError(
            "hash-bound history cadence is not a positive whole number of "
            "exact model time steps")
    begin_s, end_s = _root_history_window(exp)
    schedule = _history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=experiment_cadence,
        begin_seconds=begin_s, end_seconds=end_s,
        interval2_seconds=exp.root.run.history_interval2_s,
        change_seconds=exp.root.run.history_interval_change_s)
    periods = len(schedule) - 1
    first = schedule[0][1]
    last_offset, last_scheduled, _name = schedule[-1]
    run_end = exp.start_time + timedelta(seconds=float(exp.run_seconds))
    last_equals_run_end = (
        Fraction(float(last_offset)) == Fraction(float(exp.run_seconds)))
    window = ({} if (begin_s, end_s) == (0.0, None) else {
        "history_begin_seconds": begin_s,
        "history_end_seconds": end_s})
    return {**window,
        "schema": "gpuwm-hash-bound-history-cadence-v1",
        "requested_seconds": requested,
        "experiment_seconds": experiment_cadence,
        # The quick-look stream: the schedule above is the alarm (every
        # frame written, quick-look subsets under quicklook/ included);
        # full frames land every full_history_seconds.
        **({} if full_cadence is None else {
            "quicklook_seconds": experiment_cadence,
            "full_history_seconds": full_cadence}),
        "exact_model_steps_per_interval": None if adaptive else int(exact_steps),
        **({"adaptive_event_landing": True} if adaptive else {}),
        "complete_intervals": periods,
        "expected_frame_count": periods + 1,
        "initial_valid_time": first.isoformat(),
        "last_scheduled_offset_seconds": last_offset,
        "last_scheduled_valid_time": last_scheduled.isoformat(),
        "run_end_offset_seconds": float(exp.run_seconds),
        "run_end_valid_time": run_end.isoformat(),
        "last_scheduled_equals_run_end": last_equals_run_end,
        "initial_frame_required": not begin_s,
        "run_end_frame_scheduled": last_equals_run_end,
    }


def _profile_runtime_switches(source: str, profile: str) -> dict[str, object]:
    """Resolve one complete named physics product.

    No per-source membership refusal any more (owner ruling 2026-07-31:
    the suite choice is the user's; ``_SOURCE_PHYSICS_PROFILES`` is
    reported verification metadata, not a gate).  What remains
    fail-closed here is a NAME the shipped switch tables genuinely do
    not define, and the one real per-source blocker -- the registry's
    land-surface route declaration -- is enforced on the resolved
    selector in :func:`_validate_physics`, for named and unnamed suites
    alike.
    """

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported prepared forecast source {source!r}")
    try:
        return single_domain_runtime_switches(profile)
    except ValueError:
        raise ValueError(
            f"physics profile {profile!r} is not a shipped runner "
            f"profile; omit --physics-profile to run the hash-bound "
            f"experiment's own suite as written, or pick one of "
            f"{list(PHYSICS_PROFILES)!r}") from None


def _profile_readiness(source: str, profile: str) -> tuple[str, str | None]:
    profile = canonical_template_id(profile)
    if source in _MAPPED_SOURCES:
        suffix = {
            THOMPSON_PHYSICS_PROFILE: (
                "; Thompson MP8 also remains an experimental table-bound "
                "runtime"),
            NSSL2_PHYSICS_PROFILE: (
                "; NSSL-2 MP18 also remains a matched-run candidate"),
            NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE: (
                "; NSSL-2 MP18 plus legacy RRTMG remains a matched-run "
                "candidate"),
        }.get(profile, "")
        return "IMPLEMENTED_UNVERIFIED", (
            f"this {source} GPU forecast profile has no public acceptance gate"
            f"{suffix}")
    if profile == TWENTYCRV3_WSM6_PHYSICS_PROFILE:
        # Selectable on any prepared source now; its verification
        # standing does not improve with the source it runs on.
        return "IMPLEMENTED_UNVERIFIED", (
            "this profile has no public forecast acceptance gate")
    if profile == THOMPSON_PHYSICS_PROFILE:
        return "WRF_MATCHED_RUN_EXPERIMENTAL_RUNTIME", (
            "This legacy readiness code names an experimental runtime; "
            "the July matched run is historical and does not cover this "
            "current daytime-only suite.")
    if profile == MORRISON_PHYSICS_PROFILE:
        return "WRF_MATCHED_RUN_RUNTIME_PROFILE", (
            "This legacy readiness code is a composition exemption; no "
            "current matched-run manifest or decay tables cover this exact suite.")
    if profile in (
            NSSL2_PHYSICS_PROFILE,
            NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE,
    ):
        return "WRF_MATCHED_RUN_CANDIDATE", (
            "NSSL-2 MP18 is implemented but remains a matched-run candidate")
    if profile == MYNN_PHYSICS_PROFILE:
        return "IMPLEMENTED_UNVERIFIED", (
            "MYNN 5/5 has no woof/WRF forecast trajectory comparison")
    if profile == RUC_PHYSICS_PROFILE:
        return "IMPLEMENTED_UNVERIFIED", (
            "RUC LSM has no woof/WRF forecast trajectory comparison")
    if profile == MYNN_RUC_PHYSICS_PROFILE:
        return "IMPLEMENTED_UNVERIFIED", (
            "MYNN/MYNN/RUC has no woof/WRF forecast trajectory comparison")
    if profile in (NOAHMP_PHYSICS_PROFILE, MYNN_NOAHMP_PHYSICS_PROFILE):
        return "IMPLEMENTED_UNVERIFIED_EXPERT", (
            "Noah-MP has no woof/WRF forecast trajectory comparison; "
            "its registry acknowledgement is advisory")
    # Every other shipped template answers with its REGISTRY maturity
    # rather than a flat supported default: the fallback used to be
    # reachable only by WSM6 (maturity 'supported', which it matched),
    # and now that any registered template runs by name, a Kessler-class
    # implemented-unverified template must not read as more mature than
    # the verification block beside it in the same receipt.
    from woof.physics_registry import physics_registry

    template = physics_registry()["templates"].get(profile)
    maturity = (
        template.get("maturity") if isinstance(template, Mapping) else None)
    if maturity == "supported":
        return "SUPPORTED_RUNNER_PROFILE", None
    if maturity == "wrf-matched-run":
        return "WRF_MATCHED_RUN_RUNTIME_PROFILE", None
    return "IMPLEMENTED_UNVERIFIED", (
        f"registry maturity {maturity!r}: this profile has no "
        "woof/WRF forecast trajectory comparison on this runner")


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("physics profile contains a non-finite float")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise TypeError(f"unsupported TOML physics value {value!r}")


def _without_materialized_physics(raw: Mapping[str, object]) -> object:
    """Return a comparison copy containing every non-profile control."""

    normalized = copy.deepcopy(dict(raw))
    shared = normalized.get("shared")
    if isinstance(shared, dict):
        for key in _MATERIALIZED_PHYSICS_KEYS:
            shared.pop(key, None)
    domains = normalized.get("domain")
    if isinstance(domains, list):
        for domain in domains:
            if isinstance(domain, dict):
                for key in _MATERIALIZED_PHYSICS_KEYS:
                    domain.pop(key, None)
    return normalized


def _profile_pinned_physics(
        switches: Mapping[str, object]) -> dict[str, object]:
    """Every key THIS materialization writes on the profile's behalf.

    The profile's own switch table plus the nest-transition rule the
    emitter appends.  A key outside it -- ``radt_minutes``, which no
    switch table pins -- is one the profile writes nothing for, so the
    materializer leaves the config's own value exactly where the user
    wrote it rather than deleting a setting nothing is replacing.
    (Keeping ``radt_minutes`` is physically inert under every shipped
    profile, all of which pin a positive ``radt`` that every consumption
    site prefers; what it moves is the prepared-cache domain identity
    for a user config declaring a non-default value, disclosed in the
    CHANGELOG and measured in
    ``test_a_key_no_profile_pins_is_kept_rather_than_deleted``.)

    Unpinned is not the same as unGOVERNED: a kept key whose value the
    profile still resolves is checked for agreement -- see
    :func:`_profile_resolved_unpinned_physics`.
    """

    pinned = dict(switches)
    pinned["nest_microphysics_transition"] = _NEST_MICROPHYSICS_TRANSITION
    return pinned


def _profile_resolved_unpinned_physics(
        pinned: Mapping[str, object]) -> dict[str, object]:
    """Keys the profile does not PIN but its authority still RESOLVES.

    ``ra_rrtmg_variant`` selects which radiation IMPLEMENTATION a
    resolved RRTMG (4, 4) pair executes -- woof/core/rrtmg_legacy.py or
    woof/core/rrtmgp.py.  A profile that pins the pair without pinning
    the variant (the 20CRv3 WSM6 suite is the one shipped example)
    resolves the variant through the RunConfig default, so a config
    declaring the OTHER variant would keep it under the kept-unpinned
    rule and run a different radiation implementation under the
    profile's name -- and step 5 (:func:`_validate_profile_switches`)
    could never notice, because the key is outside the switch table.
    Such a key is governed for AGREEMENT exactly like a pinned one,
    while staying outside the strip-and-rewrite set: a declaration that
    matches the resolved value survives where the user wrote it.

    Under a profile whose pair is not (4, 4) the variant is inert and
    ungoverned here: a config coherent enough to build with the legacy
    variant declares a 4/4 pair of its own, which the radiation
    comparison already refuses by name, and a variant row against a
    no-RRTMG profile would be noise beside it.
    """

    if "ra_rrtmg_variant" in pinned \
            or _profile_radiation_pair(pinned) != (4, 4):
        return {}
    from dataclasses import fields as dataclass_fields
    from woof.config import RunConfig
    return {"ra_rrtmg_variant": next(
        field.default for field in dataclass_fields(RunConfig)
        if field.name == "ra_rrtmg_variant")}


def _profile_radiation_pair(pinned: Mapping[str, object]) -> tuple[int, int]:
    """The ``(lw, sw)`` this profile resolves to, aggregate included."""

    lw = int(pinned["ra_lw_physics"])
    sw = int(pinned["ra_sw_physics"])
    if (lw, sw) == (-1, -1):
        aggregate = int(pinned["ra_physics"])
        return aggregate, aggregate
    return lw, sw


def _physics_values_agree(declared: object, pinned: object) -> bool:
    """Whether a config's value IS the profile's value.

    ``true``/``1`` are different statements about a boolean switch and
    never agree; ``12`` and ``12.0`` are the same number written twice
    and always do, because TOML types an unsuffixed integer as ``int``
    while the switch tables carry the float the RunConfig field holds.
    """

    if isinstance(declared, bool) or isinstance(pinned, bool):
        return (isinstance(declared, bool) and isinstance(pinned, bool)
                and declared is pinned)
    if isinstance(declared, (int, float)) \
            and isinstance(pinned, (int, float)):
        return float(declared) == float(pinned)
    return type(declared) is type(pinned) and declared == pinned


def _declared_physics_conflicts(
        base_raw: Mapping[str, object], base_exp, *,
        pinned: Mapping[str, object],
) -> list[dict[str, object]]:
    """Every profile-owned physics value the config states differently.

    Only keys the config actually WRITES are considered: a key it is
    silent about is a key the profile is free to supply, which is the
    whole point of materializing an authority.  Radiation is compared as
    one resolved ``(lw, sw)`` selection rather than key by key, so a
    config that reaches the profile's exact two schemes through the
    aggregate spelling is agreement and not a conflict.  The governed
    set is the pinned keys plus the resolved-but-unpinned ones
    (:func:`_profile_resolved_unpinned_physics`): both are values the
    generated authority states on the profile's behalf, whether by
    writing them or by resolving their defaults.
    """

    governed = {**_profile_resolved_unpinned_physics(pinned), **pinned}
    expected_radiation = _profile_radiation_pair(pinned)

    def radiation_agrees(run) -> bool:
        try:
            return radiation_scheme_ids(run) == expected_radiation
        except ValueError:
            # A config whose own radiation keys are incoherent cannot
            # have reached here (build_experiment resolves them first),
            # but a resolver that raises must not read as agreement.
            return False

    runs = {int(domain.grid_id): domain.run for domain in base_exp.domains}
    scopes: list[tuple[str, Mapping[str, object], bool]] = []
    shared = base_raw.get("shared")
    if isinstance(shared, Mapping):
        # A [shared] radiation key is inherited by every domain, so it
        # agrees only when every domain resolves to the profile's pair.
        scopes.append(("[shared]", shared,
                       all(radiation_agrees(run) for run in runs.values())))
    domains = base_raw.get("domain")
    if isinstance(domains, list):
        for index, domain in enumerate(domains):
            if not isinstance(domain, Mapping):
                continue
            grid_id = int(domain.get("grid_id", index + 1))
            run = runs.get(grid_id)
            scopes.append((f"[[domain]] d{grid_id:02d}", domain,
                           run is not None and radiation_agrees(run)))

    conflicts: list[dict[str, object]] = []
    for label, table, radiation_ok in scopes:
        for key in sorted(table):
            if key not in governed:
                continue
            if key in _RADIATION_SELECTION_KEYS and radiation_ok:
                continue
            if key == "epssm" and table[key] == EPSSM_AUTO:
                # "auto" is the documented spelling of leaving epssm unset
                # (A171's importer writes it), and an unset key is the
                # profile's to supply: counting it as a conflict refused
                # every re-imported namelist pair run under a profile.
                continue
            if _physics_values_agree(table[key], governed[key]):
                continue
            conflicts.append({
                "scope": label,
                "key": key,
                "config_value": table[key],
                "profile_value": governed[key],
            })
    return conflicts


def _refuse_declared_physics_drift(
        conflicts: list[dict[str, object]], *, profile: str, origin: str,
        base_exp,
) -> None:
    """Name every overwritten key, both values, and both remedies.

    The sibling rail is the physics-fidelity axis
    (:func:`woof.experiment._reject_axis_authored_keys`), and it refuses
    a second author for a governed key even when the two AGREE.  This
    one does not, on purpose: ``physics_mode``'s remedy is "strip the
    key and let the axis write it", while a named ``--physics-profile``
    is an ASSERTION that the config IS that suite -- step 5 of the
    documented chain (:func:`_validate_profile_switches`) refuses the run
    unless the experiment states the profile's values switch for switch.
    A config that agrees is therefore the documented happy path (`woof
    domain --physics-profile P` writes P and step 2 names P), and
    refusing it would refuse the route's own output.  Disagreement is
    the whole of what is wrong here.
    """

    if not conflicts:
        return
    matched = identify_single_domain_profile(base_exp.root.run)
    rows = "\n".join(
        f"    {row['scope']} {row['key']}: config {row['config_value']!r}, "
        f"profile {row['profile_value']!r}"
        for row in conflicts)
    plural = "value" if len(conflicts) == 1 else "values"
    if matched == profile:
        # A remedy that cannot help is not a remedy.  The root already
        # resolves to the named profile, so "name the profile you meant"
        # would hand back the very flag that refused, and "edit those
        # keys to the profile values" would flatten the config's own
        # deliberate departures -- on the shipped LES trees that means
        # switching a PBL parameterization back ON over nests running
        # resolved turbulence.  The flag asserts every domain runs the
        # suite; this config says more than the suite on purpose, and
        # the one accurate instruction is to stop asserting.
        remedy = (
            f"  REMEDY: omit --physics-profile.  This config's root "
            f"domain already resolves to {profile}, and the values "
            f"above are its own deliberate departures from that suite "
            f"on other scopes (a nest running its own turbulence or "
            f"cumulus choice, for example).  Without the flag, the "
            f"config's own physics is published unchanged on every "
            f"domain and the receipt records the suite's verification "
            f"status; editing those keys to the profile values would "
            f"change the physics this config was written to run.")
    else:
        own_suite = (
            f"--physics-profile {matched} (the shipped suite this "
            f"config already IS)" if matched is not None else
            "omit --physics-profile, which publishes the config's own "
            "suite unchanged and records its verification status in "
            "the receipt")
        remedy = (
            f"  REMEDY, either: edit those keys in {origin} to the "
            f"profile values printed above, so the config and the "
            f"profile say the same thing; or name the profile you "
            f"meant -- {own_suite}.")
    raise ValueError(layered(
        f"--physics-profile {profile} contradicts the physics {origin} "
        f"declares: {len(conflicts)} {plural} would be overwritten.\n"
        f"{rows}\n"
        f"{remedy}",
        "--materialize-authorities publishes the experiment.toml every "
        "later stage binds by hash: the fetch manifest, the front door, "
        "the prepared-cache identity and the forecast all read THAT "
        "file, not the one you passed in.  Rewriting a physics value you "
        "wrote, into a file you never see, would run a different "
        "forecast than the one your config describes and would still "
        "pass every downstream check, because those checks compare the "
        "generated authority against the same profile.  Step 5 of the "
        "documented chain refuses this exact drift when a run reaches it "
        "(`experiment physics differs from the ... profile`); this "
        "refusal is the same rule applied at the first step, before a "
        "fetch and a preprocessing run have been paid for."))


def named_profile_config_conflicts(
        base_text: str, *, source: str, profile: str,
) -> list[dict[str, object]]:
    """The conflicts ``--materialize-authorities`` would refuse on.

    The seam ``woof go`` (and through it `woof run-plan`'s prepared
    route) derives its forwarded ``--physics-profile`` with.  The
    derivation used to read the ROOT domain alone
    (``identify_single_domain_profile(experiment.root.run)``) while the
    refusal above reads every ``[[domain]]`` table, so on the wizard's
    own ``--ladder`` trees -- root = the profile, nests deliberately
    departing from it -- the chain composed a stage-1 command guaranteed
    to refuse its own config.  Deriving through the SAME predicate the
    refusal runs is what keeps the two doors reading one sentence: an
    empty list here is exactly the promise that stage 1 will not raise
    the drift refusal for this (config, source, profile) triple.

    Returns the conflict rows (empty = the profile may be asserted).
    Raises what :func:`_render_materialized_experiment`'s own base
    parsing would raise for a config that does not load; callers that
    already loaded the experiment will not see that.
    """

    base_raw = tomllib.loads(base_text)
    base_exp = build_experiment(_experiment_tables(base_raw),
                                source="base named-source experiment")
    pinned = _profile_pinned_physics(
        _profile_runtime_switches(source, profile))
    return _declared_physics_conflicts(base_raw, base_exp, pinned=pinned)


def _json_safe_toml(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_toml(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_safe_toml(item) for item in value]
    if isinstance(value, datetime):
        return {"toml_datetime": value.isoformat()}
    return value


def _non_physics_descriptor_sha256(raw: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(_json_safe_toml(
        _without_materialized_physics(raw))).encode("ascii")).hexdigest()


def _experiment_tables(raw: Mapping[str, object]) -> dict[str, object]:
    """The [experiment]/[[domain]] view ``build_experiment`` accepts.

    ``[fetch]`` is an advisory acquisition-hints table that ``woof
    domain`` writes into every config it emits, and that ``woof check``
    and ``rw-wps`` both accept -- but the materializer used to hand the
    raw dict straight to ``build_experiment``, which refused it with
    ``unknown table(s) ['fetch']``.  The wizard's own output was
    therefore rejected by the one step that has to run BEFORE the front
    door.  Split it off exactly as the CLI loaders do, validating it on
    the way past rather than ignoring it.
    """

    tables = dict(raw)
    fetch_table = tables.pop("fetch", None)
    if fetch_table is not None:
        from woof.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source="materialized experiment")
    tables.pop("case_data", None)
    static_table = tables.pop("static", None)
    if static_table is not None:
        from woof.static.highres_production import parse_static_table
        # Validate the companion's schema here; its path is resolved by the
        # consuming file loader. The original TOML remains unchanged.
        parse_static_table(static_table, source="materialized experiment", base_dir=Path("."))
    return tables


def _profile_acknowledgements(switches, base_exp) -> tuple[str, ...]:
    """Governance declarations a named profile's own switches require.

    Two, and they are separate claims about the same selectors:

    * :data:`~woof.physics_compat.CONSTANT_DOWNWARD_LONGWAVE_ACK` --
      ``ra_lw_physics = 0`` under a land-surface scheme, so nothing
      computes downward longwave and the surface integrates a declared
      constant.  True at noon as much as at midnight.
    * :data:`~woof.physics_compat.ASYMMETRIC_RADIATION_NOCTURNAL_ACK`
      -- additionally, shortwave is ON and this experiment's window
      contains local night at its reference point.
    """

    from woof.config import radiation_scheme_ids_from_settings
    from woof.physics_compat import (
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK, first_local_night_time,
        settings_declared_acknowledgements)

    # RETIRED, with the reader it duplicated: this fell back to the
    # combined ``ra_physics`` only when a split key was ABSENT.  The
    # aggregate radiation option states both split keys as -1 and the
    # resolved pair in the combined key, and this route DECLARES the
    # suite on that option, so the read returned (-1, -1) and
    # downward_longwave_disposition then classified a full RTE+RRTMGP
    # suite as "nothing computes downward longwave" and demanded the
    # constant-GLW acknowledgement for it.  Measured, on the settings the
    # route resolves: (-1, -1) -> ('consumed', 'Noah LSM'); (4, 4) ->
    # ('scheme', None).  The engine's own rule answers it now.
    lw, sw = radiation_scheme_ids_from_settings(switches)
    required: list[str] = []
    if sw > 0 and lw == 0 and base_exp.projection is not None:
        if first_local_night_time(
                base_exp.start_time, float(base_exp.run_seconds),
                ref_lat=base_exp.projection.ref_lat,
                ref_lon=base_exp.projection.ref_lon) is not None:
            required.append(ASYMMETRIC_RADIATION_NOCTURNAL_ACK)
    # THE SUITE'S OWN CLAIM, from the one classifier every route reads:
    # woof.physics_compat.settings_declared_acknowledgements asks the
    # load guard's own disposition, so a materialized experiment here and
    # a namelist-routed preparation resolve the same token from the same
    # selectors, and neither can need a token the other did not attach.
    required.extend(settings_declared_acknowledgements(switches))
    return tuple(required)


def _acknowledgement_lines(merged, *, profile, added) -> list[str]:
    """The merged acknowledgements array, with why it grew."""

    lines = []
    for value in added:
        lines.append(f"# JUSTIFY {value}: required by the named physics")
        lines.append(f"# profile {profile}, which this materialized")
        lines.append("# experiment runs.  An explicit --physics-profile")
        lines.append("# selection is the declaration, written here in ink")
        lines.append("# rather than left to the base config's silence.")
        lines.append("# See docs/public/PHYSICS.md, \"Nocturnal validity\".")
    lines.append(
        "acknowledgements = ["
        + ", ".join(f'"{value}"' for value in merged) + "]")
    return lines


def _render_materialized_experiment(
        base_text: str, *, source: str, profile: str | None,
        origin: str = "the base experiment config",
) -> tuple[str, object, dict[str, object]]:
    """Patch only profile-owned TOML keys and preserve all other controls.

    ``profile=None`` (owner ruling 2026-07-31) means the base config's
    own physics IS the product: the authority pair is still published --
    later stages bind these exact bytes -- with no switch rewritten, and
    the receipt reports the suite's verification status instead of a
    profile.

    A NAMED profile supplies every profile-owned key the config is
    SILENT about, and must agree with every one it states.  It may not
    replace a physics value the user wrote: a disagreement is refused
    here by name (:func:`_refuse_declared_physics_drift`), before the
    output directory is claimed, before the fetch, and before
    preprocessing.  Contract change 2026-08-09; until then this function
    deleted all 26 profile-owned keys from the config and rewrote them
    from the profile without saying so.
    """

    base_raw = tomllib.loads(base_text)
    base_exp = build_experiment(_experiment_tables(base_raw),
                                source="base named-source experiment")
    from woof.experiment import refuse_unrouted_spawn
    refuse_unrouted_spawn(base_exp, "prepared single-domain forecast")
    if profile is None:
        base_non_physics = _non_physics_descriptor_sha256(base_raw)
        return base_text, base_exp, {
            "base_non_physics_descriptor_sha256": base_non_physics,
            "generated_non_physics_descriptor_sha256": base_non_physics,
            "profile_validation": {
                "schema": "gpuwm-prepared-physics-suite-v1",
                "source": source,
                "profile": None,
                "profile_binding": EXPERIMENT_CONFIG_PROFILE,
                "verification": single_domain_verification_status(
                    base_exp.root.run),
            },
        }
    switches = _profile_runtime_switches(source, profile)
    pinned = _profile_pinned_physics(switches)
    _refuse_declared_physics_drift(
        _declared_physics_conflicts(base_raw, base_exp, pinned=pinned),
        profile=profile, origin=origin, base_exp=base_exp)
    from woof.toml_document import iter_toml_statements
    statements = list(iter_toml_statements(base_text))
    # DECLARATIONS THE NAMED PROFILE ITSELF REQUIRES.
    #
    # Naming a profile on the command line is an explicit selection, and
    # the project's rule is that an explicit selection is declared in
    # INK rather than by silence -- the domain wizard has written the
    # same declarations into its emissions since 1.7.1.  A materialized
    # experiment is not a file anybody hand-wrote; its physics comes
    # from the profile the operator named, so the profile's governance
    # consequences are written beside it here and appear in the
    # published experiment.toml where a reader meets them.
    #
    # Through 1.8.7 this was invisible because the shipped proof configs
    # carried a standing nocturnal acknowledgement that covered every
    # profile materialized onto them.  They no longer do (they run real
    # radiation now), so the declaration is attached to the thing that
    # actually needs it.
    required_acknowledgements = _profile_acknowledgements(
        switches, base_exp)
    declared = tuple(base_raw.get("experiment", {}).get(
        "acknowledgements", ()) or ())
    merged = list(declared) + [
        value for value in required_acknowledgements if value not in declared]
    rewrite_acknowledgements = tuple(merged) != declared
    has_shared = any(kind == "table" and path == ("shared",)
                     for kind, path, _lines in statements)
    profile_lines = [
        "# RW-WPS generated exact physics authority; do not hand-edit.",
        f"# source={source} profile={profile}",
        *(
            f"{key} = {_toml_literal(value)}"
            for key, value in switches.items()
        ),
        f"nest_microphysics_transition = "
        f"{_toml_literal(_NEST_MICROPHYSICS_TRANSITION)}",
    ]
    output: list[str] = []
    section: tuple[str, ...] = ()
    shared_emitted = False

    def finish_section() -> None:
        nonlocal shared_emitted
        if section == ("shared",) and not shared_emitted:
            if output and output[-1] != "":
                output.append("")
            output.extend(profile_lines)
            shared_emitted = True

    for kind, path, lines in statements:
        if kind in {"table", "array"}:
            finish_section()
            if kind == "array" and path == ("domain",) and not has_shared:
                output.extend(["[shared]", *profile_lines, ""])
                shared_emitted = True
                has_shared = True
            section = path
            output.extend(lines)
            if rewrite_acknowledgements and section == ("experiment",):
                output.extend(_acknowledgement_lines(
                    merged, profile=profile,
                    added=required_acknowledgements))
            continue
        # ``pinned``, not ``_MATERIALIZED_PHYSICS_KEYS``: only a key this
        # profile actually states is replaced by the block above.  Every
        # surviving line has been proved to agree with the profile, so
        # dropping it changes no value -- and a key the profile does not
        # pin keeps the value the config gave it instead of vanishing.
        if (section in {("shared",), ("domain",)} and kind == "assignment"
                and len(path) == 1 and path[0] in pinned):
            continue
        if (rewrite_acknowledgements and section == ("experiment",)
                and kind == "assignment" and path == ("acknowledgements",)):
            continue
        output.extend(lines)
    finish_section()
    if not shared_emitted:
        if output and output[-1] != "":
            output.append("")
        output.extend(["[shared]", *profile_lines])
    rendered = "\n".join(output) + "\n"
    rendered_raw = tomllib.loads(rendered)
    rendered_exp = build_experiment(
        _experiment_tables(rendered_raw),
        source="materialized named-source experiment")
    # The base digest is taken with the profile's own declarations
    # already merged in.  They are a CONSEQUENCE of the named physics,
    # not a descriptor control the materializer chose to move, so the
    # guard still says exactly what it always said -- nothing else
    # changed -- while letting a no-radiation profile carry the
    # declaration it requires.
    base_for_digest = base_raw
    if rewrite_acknowledgements:
        base_for_digest = dict(base_raw)
        experiment_table = dict(base_for_digest.get("experiment", {}))
        experiment_table["acknowledgements"] = list(merged)
        base_for_digest["experiment"] = experiment_table
    base_non_physics = _non_physics_descriptor_sha256(base_for_digest)
    generated_non_physics = _non_physics_descriptor_sha256(rendered_raw)
    if generated_non_physics != base_non_physics:
        raise RuntimeError(
            "materialized experiment changed a non-physics descriptor control")
    validation = _validate_profile_switches(
        rendered_exp, source=source, profile=profile, all_domains=True)
    return rendered, rendered_exp, {
        "base_non_physics_descriptor_sha256": base_non_physics,
        "generated_non_physics_descriptor_sha256": generated_non_physics,
        "profile_validation": validation,
        # Never silent: what the named profile forced into the published
        # experiment.toml, so the receipt carries it too.
        "profile_acknowledgements": list(required_acknowledgements),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def materialize_named_source_authorities(
        *, source: str, base_experiment_config: Path,
        base_wps_namelist: Path, physics_profile: str | None,
        output_directory: Path,
) -> dict[str, object]:
    """Publish create-only per-case experiment/WPS physics authorities.

    ``physics_profile=None`` publishes the base pair with the config's
    own physics unchanged; the receipt reports its verification status.
    """

    base_experiment_config = _require_file(
        base_experiment_config, "base experiment config")
    base_wps_namelist = _require_file(
        base_wps_namelist, "base WPS namelist")
    base_text = base_experiment_config.read_text(encoding="utf-8")
    rendered, exp, validation = _render_materialized_experiment(
        base_text, source=source, profile=physics_profile,
        origin=str(base_experiment_config))
    # THE earliest point on the documented route at which the physics
    # this run will execute is known: step 2 of 6, before the fetch and
    # before preprocessing.  An mp8 suite whose lookup tables were never
    # staged is refused here, in one sentence naming the table and
    # `woof fetch-tables`, rather than eight minutes and two paid
    # stages later at the top of the GPU run.
    selected_mp = {int(domain.run.mp_physics) for domain in exp.domains}
    if 28 in selected_mp:
        from woof.core.thompson_aerosol_contract import AEROSOL_TABLE_ASSETS

        require_thompson_tables(
            assets=(*THOMPSON_CLASSIC_TABLE_ASSETS, *AEROSOL_TABLE_ASSETS))
    elif THOMPSON_MP_PHYSICS in selected_mp:
        require_thompson_tables(assets=THOMPSON_CLASSIC_TABLE_ASSETS)
    output_directory = claim_output_directory(
        output_directory, flag="--output-directory")
    experiment = output_directory / "experiment.toml"
    wps = output_directory / "namelist.wps"
    receipt_path = output_directory / "authority-receipt.json"
    _atomic_write_bytes(experiment, rendered.encode("utf-8"))
    _atomic_write_bytes(wps, base_wps_namelist.read_bytes())
    if _sha256(wps) != _sha256(base_wps_namelist):
        raise RuntimeError("materialized WPS namelist is not a byte-exact copy")
    if physics_profile is None:
        verification = validation["profile_validation"]["verification"]
        readiness = "EXPERIMENT_CONFIG_SUITE"
        warning = (
            None if verification["status"] == "wrf-verified"
            else verification["sentence"])
    else:
        readiness, warning = _profile_readiness(source, physics_profile)
    receipt = {
        "schema": AUTHORITY_MATERIALIZATION_SCHEMA,
        "status": "PASS",
        "source": source,
        "physics_profile": physics_profile,
        "readiness": readiness,
        "warning": warning,
        "warning_only": warning is not None,
        "base": {
            "experiment_config": {
                "path": str(base_experiment_config.resolve()),
                "bytes": base_experiment_config.stat().st_size,
                "sha256": _sha256(base_experiment_config),
            },
            "wps_namelist": {
                "path": str(base_wps_namelist.resolve()),
                "bytes": base_wps_namelist.stat().st_size,
                "sha256": _sha256(base_wps_namelist),
            },
        },
        "generated": {
            "experiment_config": {
                "path": str(experiment.resolve()),
                "bytes": experiment.stat().st_size,
                "sha256": _sha256(experiment),
            },
            "wps_namelist": {
                "path": str(wps.resolve()),
                "bytes": wps.stat().st_size,
                "sha256": _sha256(wps),
            },
        },
        "normalized_selected_physics": validation["profile_validation"],
        "non_physics_descriptor": {
            "base_sha256": validation[
                "base_non_physics_descriptor_sha256"],
            "generated_sha256": validation[
                "generated_non_physics_descriptor_sha256"],
            "status": "EXACT_UNCHANGED",
        },
        "preserved": {
            "domain_count": len(exp.domains),
            "run_seconds": float(exp.run_seconds),
            "start_time": exp.start_time.isoformat(),
            "mass_levels": int(exp.root.run.nz),
            "eta_levels": list(exp.vertical.eta_levels),
            "p_top": float(exp.vertical.p_top),
            "hybrid_opt": int(exp.vertical.hybrid_opt),
            "etac": float(exp.vertical.etac),
        },
    }
    _atomic_json(receipt_path, receipt)
    receipt["receipt"] = {
        "path": str(receipt_path.resolve()),
        "bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
    }
    return receipt


def _resolved_wrf_direct_contract_sha256(mp_physics: int) -> str:
    """Reproduce the exact resolved contract digest emitted by the exporter.

    The exporters write the digest with the same function
    (:func:`woof.wrf_direct.resolved_physics_contract_sha256`), so the two
    cannot drift apart again: a copy of the computation here once stayed
    on mp_physics while the exporter's took in the run's terrain-drag
    statistics, and every gwd_opt != 0 forecast ended rc=1 here.
    """

    from woof.wrf_direct import resolved_physics_contract_sha256

    return resolved_physics_contract_sha256(mp_physics)


def _chem_ledger_receipt(state, *, store=None):
    """One domain's chem mass ledger for the receipt; None when chem is off.

    Read only when the state carries chem, so a chem-off run imports
    nothing more than it did.
    """
    if getattr(state, "chem", None) is None:
        return None
    from woof.core.chem_driver import chem_ledger_receipt

    return _strict_json(chem_ledger_receipt(state, store=store))


def _strict_json(value):
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _stability_diagnosis(sample: Mapping[str, object], state, run) -> str:
    """Say which Courant term tripped, and that dt is the remedy.

    The vertical term is the maximum co-located ``dt*|w_upper|/dz_cell``;
    it never combines a global updraft with an unrelated thin layer.
    """

    def number(key, default):
        try:
            return float(sample.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    dt = float(run.dt)
    u_max = number("u_max", float("nan"))
    w_max = number("w_max", float("nan"))
    horizontal = number(
        "horizontal_cfl", dt * u_max / float(run.dx))
    vertical = number("vertical_cfl", float("nan"))
    if vertical >= horizontal:
        term = (
            f"VERTICAL Courant {vertical:.2f} "
            "(maximum co-located |w|/layer-thickness)")
    else:
        term = (f"HORIZONTAL Courant {horizontal:.2f} (u_max {u_max:.2f} "
                f"m/s over dx {float(run.dx) / 1000:.3f} km)")
    suggested = max(1.0, dt / 4.0)
    return (
        f"{term} dominates at time_step {dt:g} s "
        f"(horizontal {horizontal:.2f}, vertical {vertical:.2f}); "
        f"REMEDY: retry with a lower dt -- set time_step to about "
        f"{suggested:g} s in the experiment TOML and re-run the front "
        "door (the config is hash-bound to it).  Shortening the step is "
        "cheaper than the step count suggests: radiation and cumulus "
        "are called on wall-clock intervals, so a 4x shorter step "
        "measured +22% wall time, not 4x"
    )


def _atomic_json(path: Path, payload, *, heartbeat: bool = False) -> None:
    """Publish one JSON document atomically, via the supervisor's replace.

    ``heartbeat=True`` marks the progress publications a watcher is TOLD
    to read (``progress.json``): on Windows a reader's plain ``open()``
    denies rename over the file, so a poll racing the republish would
    otherwise raise WinError 5 out of the forecast loop and kill the run
    it reports on (measured on the tree runner's identical seam; `woof
    go`'s stopwatch heartbeat reads this file every 20 s).  Supervisor
    doctrine: bounded 0.50 s retry for everyone, then durable receipts
    fail loudly while a heartbeat quarantines its temporary and the
    worker stays up.
    """
    atomic_write_json(Path(path), _strict_json(payload),
                      _quarantine_on_permission_error=heartbeat)


def _duplicate_checked_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    path = Path(path)

    def reject_constant(value):
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _check_normalized_provenance(path: Path, profile_id: str, source: str) -> None:
    """A normalizer's provenance authority is the packaged one plus its receipt.

    Named breakage this prevents: a prepared tree that claims a packaged
    profile while carrying a SUBSTITUTED terrain provenance document, so the
    forecast's record of where terrain came from describes bytes nobody
    shipped.  A profile whose source publishes bytes the mapped engine cannot
    read runs an input-normalization stage first, and that stage carries the
    packaged declaration through with one key added -- the receipt naming
    every raw input, the converter, the plan and the normalized outputs.  So
    the bound document cannot be compared byte for byte.  It is compared
    exactly instead: the packaged content, plus that one key, and nothing
    else, with the receipt naming the normalizer the profile declares and
    pinning inside itself the digests of the authorities it ran under --
    including the very document it extends.

    The way out when this refuses: re-run the preparation.  Do not edit the
    provenance document in a prepared tree.
    """

    from woof.source_authorities import (packaged_authorities,
                                          packaged_authority_sha256,
                                          packaged_profile)
    from woof.source_normalization import NORMALIZATION_RECEIPT_KEY

    profile = packaged_profile(profile_id)
    normalizer = profile.get("input_normalizer")
    bound = _load_json_object(path, f"mapped {source} provenance authority")
    receipt = bound.pop(NORMALIZATION_RECEIPT_KEY, None)
    packaged_path = packaged_authorities(profile_id)["provenance"]
    packaged = _load_json_object(packaged_path, f"packaged {source} provenance")
    if bound != packaged:
        raise ValueError(
            f"mapped {source} provenance authority differs: the bound "
            f"document is not the packaged {profile_id} declaration extended "
            f"by its {normalizer} receipt")
    pins = packaged_authority_sha256(profile_id)
    request = receipt.get("request") if isinstance(receipt, dict) else None
    authorities = request.get("authorities") if isinstance(request, dict) else None
    if (not isinstance(authorities, dict)
            or request.get("normalizer") != normalizer
            or authorities.get("provenance") != pins["provenance"]
            or authorities.get("normalization") != pins["normalization"]):
        raise ValueError(
            f"mapped {source} normalization receipt does not name the "
            f"{normalizer} authorities this release ships")


def _require_digest(value, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in _HEX for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


class PreparationProofDigestMismatch(ValueError):
    """``--proof-sha256`` is not this preparation proof's file digest.

    Its own class because the CLI has to tell this apart from every other
    admission failure: it is the one a correct, complete, uncorrupted
    preparation produces when the reader took the digest from the wrong
    place, and the remedy is a different string to paste rather than a
    re-run of anything.
    """


def _proof_digest_refusal(
        proof_path: Path, given: str, actual: str,
) -> PreparationProofDigestMismatch:
    """The named refusal for a ``--proof-sha256`` that does not match.

    ``preparation proof SHA differs from --proof-sha256`` was the whole
    message, it arrived as an uncaught traceback, and it named neither
    which of the two digests in play is wanted nor what was given.  The
    document carries a field spelled ``proof_content_sha256`` -- the
    canonical hash of its own content WITHOUT that field -- and a reader
    hunting a digest inside a 42 KB proof finds that one first.  It can
    never equal the file digest, so the confusion is a dead end that the
    old sentence left the reader to discover.
    """

    breakage = (
        "this digest is what binds the forecast to the exact preparation "
        "that produced it, so a run past a mismatch would integrate a "
        "preparation nobody certified")
    content = None
    try:
        document = json.loads(Path(proof_path).read_text(encoding="utf-8"))
        if isinstance(document, Mapping):
            content = document.get("proof_content_sha256")
    except (OSError, UnicodeDecodeError, ValueError):
        content = None
    if isinstance(content, str) and content == given:
        what = (
            "the value given is that document's own "
            "\"proof_content_sha256\" field, which covers the proof's "
            "content WITHOUT that field and therefore never equals the "
            "file's digest")
    elif isinstance(content, str):
        what = (
            "the value given matches neither the file nor the document's "
            "own \"proof_content_sha256\" field")
    else:
        what = "the value given is not that file's digest"
    return PreparationProofDigestMismatch(
        f"--proof-sha256 refused: it must be the sha256 of the file "
        f"{proof_path}, and {what}.  {breakage[0].upper()}{breakage[1:]}.\n"
        f"  expected (sha256 of the file proof.json): {actual}\n"
        f"  given    (--proof-sha256):                {given}\n"
        f"  Remedy: pass the expected digest above.  `rw-wps` and `woof "
        f"prep` print the complete forecast command, with this digest "
        f"and the other two already filled in, when a preparation "
        f"finishes.")


def _preparation_binding_refusal(args) -> str | None:
    """Exactly one binding: a prepared head, or a sealed proof and cache.

    The breakage this prevents is a forecast bound to no preparation at
    all, or to two digests that could name different preparations.
    """

    head = getattr(args, "prepared_head_sha256", None)
    sealed = (args.proof_sha256, args.prepared_content_sha256)
    if head is not None:
        if any(value is not None for value in sealed):
            return ("--prepared-head-sha256 binds the preparation by its "
                    "head; pass it alone, or pass --proof-sha256 with "
                    "--prepared-content-sha256 for a sealed preparation")
        # Whether the head needs --source-manifest-sha256 is the head's to
        # say (an as-posted head binds its plan): the preflight reads it.
        return None
    if any(value is None for value in sealed):
        return ("a forecast binds its preparation with "
                "--prepared-head-sha256, or with both --proof-sha256 and "
                "--prepared-content-sha256")
    # A sealed binding without --source-manifest-sha256 is refused by the
    # preflight, before any file is read (``source-manifest-sha256 must be
    # a lowercase SHA-256 digest``).
    return None


def _proof_digest_refusal_at_the_door(args) -> str | None:
    """The ``--proof-sha256`` answer a door can give before doing work.

    Returns the refusal text, or ``None`` when there is nothing to say
    here -- either the digest matches, or the prepared root is not there
    at all, which is the preflight's refusal to give and not this one's.
    The check is the same digest comparison the preflight makes, so the
    two cannot disagree about what the flag means.
    """

    if args.proof_sha256 is None:
        return None
    try:
        _require_digest(args.proof_sha256, "proof-sha256")
    except ValueError as error:
        return f"--proof-sha256 refused: {error}"
    proof_path = Path(args.prepared_root) / "proof.json"
    if not proof_path.is_file():
        return None
    try:
        actual = _sha256(proof_path)
    except OSError:
        return None
    if actual == args.proof_sha256:
        return None
    return str(_proof_digest_refusal(
        proof_path.resolve(), args.proof_sha256, actual))


def _require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.resolve()


def _require_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path.resolve()


def _validated_thompson_runtime_contract() -> dict[str, object]:
    """Bind the guarded MP8 runtime to exact WRF tables and source bytes."""

    # The two process-environment gates that used to stand here predate
    # the packaging promotion: the classic tables ship as package data,
    # `woof fetch-tables` stages the externalized one, and `woof
    # doctor` byte-validates all four.  Requiring the env vars on top
    # made mp8 fail twice at runtime on a machine doctor had just called
    # clean, with neither variable documented.  thompson_table_root()
    # honours WOOF_THOMPSON_TABLE_ROOT as an override and falls back to
    # the packaged directory; either way the assets are re-validated
    # byte for byte below, which is the check that ever mattered.
    #
    # Absence is answered BEFORE the byte validation, and in a sentence
    # that names the command which fixes it.  This function runs while
    # the physics receipt is built -- at --materialize-authorities,
    # which is step 2 of 6 on the documented GFS route -- so an install
    # that never staged the externalized tables learns it here, before
    # the fetch and before preprocessing, instead of from a
    # FileNotFoundError traceback at the top of the GPU run.
    root = _require_directory(
        Path(require_thompson_tables(assets=THOMPSON_CLASSIC_TABLE_ASSETS)),
        "Thompson table root")
    assets = validate_thompson_table_assets(root)
    if tuple(assets) != tuple(THOMPSON_CLASSIC_TABLE_ASSETS):
        raise ValueError("validated Thompson assets differ from the classic contract")

    table_identity = {
        "schema": 1,
        "table_set": THOMPSON_TABLE_SET_ID,
        "wrf_version": THOMPSON_WRF_REFERENCE_VERSION,
        "wrf_commit": THOMPSON_WRF_REFERENCE_COMMIT,
        "assets": [
            {
                "filename": asset.filename,
                "bytes": int(asset.bytes),
                "sha256": asset.sha256,
            }
            for asset in assets
        ],
    }
    implementation_sha256 = {
        relative: _sha256(_require_file(
            REPO / relative, f"Thompson implementation {relative}"))
        for relative in _THOMPSON_IMPLEMENTATION_FILES
    }
    payload_bytes = sum(
        record.payload_bytes
        for records in THOMPSON_GENERATED_TABLE_FILES.values()
        for record in records
    ) + sum(
        record.payload_bytes for record in THOMPSON_AUXILIARY_TABLE_RECORDS)
    return {
        "selector": THOMPSON_MP_PHYSICS,
        "wrf_reference_version": THOMPSON_WRF_REFERENCE_VERSION,
        "wrf_reference_commit": THOMPSON_WRF_REFERENCE_COMMIT,
        "transported_fields": list(THOMPSON_TRANSPORTED_SPECIES),
        "guard": {
            # Retired: the enable gate.  What remains is the resolution
            # rule and the byte validation that always did the work.
            "table_root_environment": THOMPSON_TABLE_ROOT_ENV,
            "table_root_source": (
                "environment override" if os.environ.get(
                    THOMPSON_TABLE_ROOT_ENV) else "packaged default"),
        },
        "table_authority": {
            **table_identity,
            "root": str(root),
            "payload_bytes": int(payload_bytes),
            "identity_sha256": hashlib.sha256(
                _canonical(table_identity).encode("ascii")).hexdigest(),
        },
        "implementation_sha256": implementation_sha256,
        "native_reflectivity": {
            "field": "REFL_10CM",
            "producer": "classic-Thompson same-call graupel-number shadow",
            "consumer": "woof.core.refl.consume_refl_10cm",
            "fallback": None,
        },
    }


def _receipt_mp_physics(physics_receipt: Mapping[str, object]) -> int:
    """The resolved microphysics selector a physics receipt records."""

    resolved = physics_receipt.get("resolved")
    if not isinstance(resolved, Mapping):
        return -1
    try:
        return int(resolved.get("mp_physics"))
    except (TypeError, ValueError):
        return -1


def _thompson_authority_paths(
        physics_receipt: Mapping[str, object],
) -> dict[str, Path]:
    # Keyed on the resolved selector, not the profile id: every admitted
    # mp8 suite binds the table authority (matching the domain-tree
    # runner, which has always keyed this on mp_physics == 8).
    if _receipt_mp_physics(physics_receipt) != THOMPSON_MP_PHYSICS:
        return {}
    contract = physics_receipt.get("thompson_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("Thompson physics receipt lacks its contract")
    table = contract.get("table_authority")
    if not isinstance(table, Mapping):
        raise RuntimeError("Thompson physics receipt lacks table authority")
    root = Path(str(table.get("root"))).resolve()
    paths = {
        f"thompson_table_{asset.filename}": _require_file(
            root / asset.filename, f"Thompson table {asset.filename}")
        for asset in THOMPSON_CLASSIC_TABLE_ASSETS
    }
    paths.update({
        f"thompson_implementation_{index:02d}": _require_file(
            REPO / relative, f"Thompson implementation {relative}")
        for index, relative in enumerate(_THOMPSON_IMPLEMENTATION_FILES)
    })
    return paths


def _verify_thompson_runtime_environment(
        physics_receipt: Mapping[str, object],
) -> None:
    if _receipt_mp_physics(physics_receipt) != THOMPSON_MP_PHYSICS:
        return
    contract = physics_receipt.get("thompson_contract")
    table = contract.get("table_authority") if isinstance(contract, Mapping) else None
    expected_root = table.get("root") if isinstance(table, Mapping) else None
    actual_root = str(Path(thompson_table_root()).resolve())
    if actual_root != expected_root:
        raise RuntimeError(
            "Thompson table root changed between preflight and run: "
            f"preflight resolved {expected_root}, now {actual_root}")


def _sibling_outdir(protected: Path) -> Path:
    """A concrete --outdir the guard below will accept, beside ``protected``.

    Named in the refusal so it reads as an instruction rather than a rule,
    and matching what the GFS front door suggests, so the two surfaces
    send the user to the same directory.
    """

    protected = Path(protected)
    return protected.parent / f"{protected.name}-forecast"


def validate_output_directory(
        path: Path, *, protected_roots=(), flag: str = "--outdir",
) -> Path:
    """Resolve and protect a requested output before writing lock sidecars."""
    path = Path(path)
    # resolve() walks symlinks and raises OSError on a loop (Errno 40 /
    # "Symlink loop from ...").  This function's whole subject is turning
    # an OS-level exception into a sentence, and this was the one it let
    # through: a looping --outdir reached the reader as a traceback at
    # rc 1, from the line before the first refusal.
    resolved = _resolve_or_refuse(path, flag)
    for protected in protected_roots:
        protected = _resolve_or_refuse(Path(protected), flag)
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                f"forecast output must not be inside protected input tree "
                f"{protected}; the forecast may not write into its own "
                f"inputs.  Pass an {flag} beside them instead, for "
                f"example {_sibling_outdir(protected)}")
    return resolved


def claim_output_directory(
        path: Path, *, protected_roots=(), flag: str = "--outdir",
) -> Path:
    """Create an output directory without merging prior run contents.

    This filesystem admission also accepts a precreated empty directory.
    Concurrent WRF launches additionally hold ``output_claim`` leases for
    their whole supervisor/worker lifetime.
    """
    from woof.filesystem_paths import io_path, keep_spelling
    given = path
    path = io_path(path)
    resolved = validate_output_directory(path, protected_roots=protected_roots, flag=flag)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        # EXISTS is not the breakage; HOLDS AN EARLIER RUN is.  An empty
        # directory carries no frames to merge with and no receipt to
        # overwrite, so refusing it prevents nothing.
        #
        # And it is the normal case, not an edge one: `woof sim`
        # allocates this run's stamped folder before it dispatches here,
        # create-exclusively, because two launches inside one second must
        # not share a folder.  The door therefore hands this function a
        # directory that exists and is empty, every single time.
        # Refusing it killed the whole prepare-then-simulate route --
        # measured on test host A, where the forecast never started.
        try:
            held = sorted(child.name for child in path.iterdir())
        except OSError as probe_error:
            # Reading it failed, so this is not the empty-folder case at
            # all -- and this function's whole subject is turning an
            # OS-level exception into a sentence, so the probe must not
            # become the one that escapes.
            #
            # A looping symlink is the case that gets here, and it is the
            # one E-12 already names: `mkdir` says FileExistsError
            # because the link IS a path entry, and scandir on it says
            # ELOOP.  `resolve()` does NOT raise for it on Linux/3.14 --
            # measured -- so deferring to _resolve_or_refuse would have
            # this function silently ACCEPT an --outdir with no
            # destination, which is worse than the traceback it replaced.
            # The refusal is raised here, in E-12's own words, with the
            # probe's error as the cause.
            detail = (getattr(probe_error, "strerror", None)
                      or str(probe_error))
            errno = getattr(probe_error, "errno", None)
            where = f" (errno {errno})" if errno is not None else ""
            raise ValueError(
                f"{flag} {path} cannot be resolved to a real location: "
                f"{detail}{where}.  A symlink that points into its own "
                f"chain has no destination to create, so there is nothing "
                f"here to refuse or to reuse -- pass an {flag} that names "
                f"a real path.") from probe_error
        if held:
            from woof.ensemble.runtime_context import current_session
            session = current_session()
            if (session is not None and session.restart_roster is not None
                    and path.resolve() == session.output_directory.resolve()):
                return keep_spelling(given, _resolve_or_refuse(path, flag))
            raise FileExistsError(
                f"{resolved} already holds a run's output "
                f"({', '.join(held)[:120]}), and this runner never merges "
                f"into an earlier run's directory -- the receipt it writes "
                f"has to describe one run.  Pass a new {flag}, or remove "
                f"the old directory first.") from error
    return keep_spelling(given, _resolve_or_refuse(path, flag))


def _resolve_or_refuse(path: Path, flag: str) -> Path:
    """``path.resolve()``, with its failures delivered as sentences.

    RuntimeError is caught alongside OSError, and deliberately so: on a
    symlink loop pathlib raises ``RuntimeError("Symlink loop from ...")``
    rather than ``OSError(ELOOP)``, which an OSError-only guard misses
    entirely.  Found by running the fix on Linux against the installed
    wheel -- the Windows box this was written on cannot create a looping
    symlink without privilege, so the test skipped and the guard looked
    correct.  The catch is narrow enough to stay accurate: it wraps one
    call whose only failure mode is "this path does not resolve".
    """
    from woof.filesystem_paths import canonical_path
    try:
        return canonical_path(path)
    except (OSError, RuntimeError) as error:
        detail = getattr(error, "strerror", None) or str(error)
        errno = getattr(error, "errno", None)
        where = f" (errno {errno})" if errno is not None else ""
        raise ValueError(
            f"{flag} {path} cannot be resolved to a real location: "
            f"{detail}{where}.  A symlink that points into its own chain "
            f"has no destination to create, so there is nothing here to "
            f"refuse or to reuse -- pass an {flag} that names a real "
            f"path.") from error


def _validate_restored_source_adapter(metadata, source: str) -> None:
    """Reject a conflicting optional cache hint after identity verification.

    Direct bundles historically persist ``source_adapter`` in user metadata;
    hierarchy domain caches do not.  The mandatory source identity is already
    reconstructed and checked during preflight, so absence is valid but a
    contradictory redundant value is not.
    """

    adapter = dict(metadata).get("source_adapter")
    expected = "mapped" if source in _MAPPED_SOURCES else source
    if adapter is not None and adapter != expected:
        raise ValueError("restored cache source adapter differs from request")


def _physics_update_count(component) -> int:
    """Return zero for an explicitly disabled optional physics component."""

    if component is None:
        return 0
    value = int(component.update_count)
    if value < 0:
        raise ValueError("physics update count cannot be negative")
    return value


def _validate_restored_cache_receipt(
        receipt: Mapping[str, object], expected_content_sha256: str,
) -> None:
    """Recheck the caller's content pin immediately after cache restore."""

    if (not isinstance(receipt, Mapping)
            or receipt.get("schema") != PREPARED_CACHE_SCHEMA
            or receipt.get("status") != "RESTORED"
            or receipt.get("content_sha256") != expected_content_sha256):
        raise ValueError(
            "restored prepared cache differs from the caller-pinned content")


def _provenance_receipt() -> dict:
    """The running tree, for the report.  Never raises.

    Beside ``runtime_source_identity`` rather than instead of it: that
    hashes the forecast implementation's BYTES, which is the strongest
    binding available and stays the authority.  This says which INSTALL
    those bytes came out of -- wheel, editable or checkout, on which
    branch, dirty or clean -- which is the question a reader holding two
    receipts with different numbers actually has.
    """

    try:
        from woof.provenance_gate import receipt_block

        return receipt_block()
    except Exception as error:                          # noqa: BLE001
        return {"unavailable": f"{type(error).__name__}: {error}"}


#: Attempts per identity resolution.  The failure this exists for is
#: transient and environmental -- a ``git`` that does not spawn -- not a
#: repository answering "no".  Two retries cost milliseconds on a
#: healthy box and are not paid at all outside the one install shape
#: that can degrade this way (see :func:`_runtime_provenance`).
_IDENTITY_ATTEMPTS = 3

#: The ``identity_source`` values a CHECKOUT can only reach by falling
#: THROUGH the git branch of :func:`woof.runtime_manifest.provenance`:
#: that branch catches ``(OSError, SubprocessError)`` around its three
#: ``git`` calls and continues down the ladder rather than raising, so
#: one subprocess that fails to start re-answers the whole question from
#: a lower rung.  The manifest rung is deliberately absent -- it is
#: resolved BEFORE git and so is never a fall-through.
_GIT_FALLTHROUGH_SOURCES = frozenset({
    "installed-editable-source",
    "installed-wheel-record",
    "runtime-module-sha256-only",
})

#: The identity of a tree that is neither a manifest, nor a checkout,
#: nor an installed distribution: a source tree someone copied into
#: place.  The per-module digests in :func:`_runtime_source_identity`
#: still bind what ran.
_UNRESOLVED_PROVENANCE: Mapping[str, object] = MappingProxyType({
    "git_commit": None, "git_tree": None, "git_status_short": None,
    "identity_source": "runtime-module-sha256-only",
    "distribution_manifest_sha256": None, "installed_wheel": None,
    "installed_editable": None,
})


def _runtime_provenance() -> dict[str, object]:
    """Which install is executing, asked so a git hiccup cannot rename it.

    One resolver for all three installs (``woof.runtime_manifest``).
    Its git branch, though, is the one place a healthy checkout can
    answer as something else entirely.  ``provenance`` wraps its three
    ``git`` calls in ``except (OSError, subprocess.SubprocessError):
    pass`` and CONTINUES to the editable/wheel ladder, so a single
    ``git rev-parse`` that fails to spawn moves ``identity_source``,
    ``git_commit``, ``git_tree``, ``git_status_short`` and one of
    ``installed_wheel`` / ``installed_editable`` in one step, with
    nothing on disk changed.  That is the cheap event this run's receipt
    was destroyed by; see :func:`_runtime_source_identity_change`.

    So a checkout that did not answer ``"git"`` is asked again.  The
    precondition is a FILESYSTEM one -- ``.git`` is a name on disk --
    because :func:`woof.runtime_manifest.git_checkout_root` answers the
    same question by spawning git, and during the outage this exists for
    it would answer "not a checkout" every time and skip the retry that
    fixes it.  A wheel install has no ``.git`` and so pays nothing.
    """

    from woof.runtime_manifest import IdentityError, provenance

    identity: dict[str, object] = dict(_UNRESOLVED_PROVENANCE)
    for attempt in range(_IDENTITY_ATTEMPTS):
        try:
            identity = dict(provenance(REPO))
        except IdentityError:
            identity = dict(_UNRESOLVED_PROVENANCE)
        if identity.get("identity_source") not in _GIT_FALLTHROUGH_SOURCES:
            break
        if not (REPO / ".git").exists():
            # Not a checkout at all.  This rung's answer is the truth
            # rather than a degraded one -- pip knows exactly which
            # artifact it wrote, and RECORD says so -- so asking again
            # would only buy the same answer three times.
            break
        if attempt + 1 < _IDENTITY_ATTEMPTS:
            time.sleep(0.1 * (attempt + 1))
    # A checkout that STILL did not answer "git" has now been asked
    # _IDENTITY_ATTEMPTS times, so this is no longer a hiccup: git
    # cannot be launched at all.  The rung that answered instead reports
    # on the tree PIP WAS POINTED AT, which in a linked worktree is a
    # different tree entirely -- measured here: the worktree executing
    # is eada530d and the editable rung answered c1b32d0, the main
    # checkout's HEAD.  Neither this module nor anything in
    # ``source_sha256`` came out of that tree, so the top-level commit
    # is re-read from REPO's own ``.git``.
    if (identity.get("identity_source") in _GIT_FALLTHROUGH_SOURCES
            and (REPO / ".git").exists()):
        identity.update(_repo_head_identity())
    return identity


def _repo_head_identity() -> dict[str, object]:
    """The git half of the identity, for REPO and no other tree.

    ``git_commit`` is read out of ``.git`` bytes in a format git has not
    changed in its lifetime, so a process that cannot SPAWN git can
    still say which commit is executing.  Worktree-aware, which is what
    matters here: ``.git`` is a FILE in a linked worktree and this
    project's work happens almost entirely in them.

    ``git_tree`` and ``git_status_short`` stay ``None``, and say so: the
    tree id lives inside the commit object and the working tree cannot
    be inspected without git at all.  The comparison reads ``None`` as
    "this was not asked" rather than as an answer.
    """

    from woof.provenance import git_dir_identity

    try:
        head = git_dir_identity(REPO)
    except Exception:                                   # noqa: BLE001
        head = None
    return {
        "git_commit": None if head is None else str(head["commit_full"]),
        "git_tree": None,
        "git_status_short": None,
    }


def _tracked_status_lines(value: object) -> list[str] | None:
    """``git status --short`` with its untracked (``??``) rows dropped.

    ``None`` in, ``None`` out, because "nobody asked git" and "git found
    nothing" are different statements and the caller reads the
    difference.

    Untracked files are not the forecast implementation.  A run that
    writes anything inside the checkout creates them -- ``runs/`` is not
    ignored, so a single scratch file lands in ``git status --short`` as
    ``?? runs/...`` -- and comparing the raw list made such a run fail
    its own receipt for having produced output.  Measured in this
    worktree: 0 status lines before ``touch runs/probe``, 1 after.

    What survives the filter is every TRACKED path git reports as
    modified, added, deleted or renamed, which is the signal worth
    keeping: it catches an edit to a tracked file anywhere in the tree,
    including the many that ``source_sha256`` does not hash.
    """

    if value is None:
        return None
    return [str(line) for line in value if not str(line).startswith("??")]


def _untracked_blind(value: object) -> object:
    """One identity component with its untracked-file COUNT removed.

    ``installed_editable`` carries ``git.untracked_files`` -- an integer
    counting exactly the rows :func:`_tracked_status_lines` drops -- so
    on an editable install one scratch file moves the identity twice.
    ``git.dirty`` and ``git.dirty_files`` count TRACKED changes and are
    left alone, on purpose: they are the same signal the filtered status
    lines carry, arriving from the other install shape.
    """

    if not isinstance(value, Mapping):
        return value
    git = value.get("git")
    if not isinstance(git, Mapping) or "untracked_files" not in git:
        return value
    return {**value,
            "git": {key: item for key, item in git.items()
                    if key != "untracked_files"}}


def _runtime_source_identity_change(
        before: Mapping[str, object], after: Mapping[str, object],
) -> str | None:
    """Name the component that moved, or ``None`` if nothing did.

    A plain ``before != after`` conflated unrelated events, and the
    cheap ones destroyed the expensive one.  A finished forecast --
    every step integrated, every ``wrfout`` frame written and verified
    -- loses its receipt to ``forecast runtime implementation changed
    during run`` in two shapes that share no code with an implementation
    change:

    * one ``git rev-parse`` failing to spawn at the end of the run,
      which drops :func:`woof.runtime_manifest.provenance` through its
      git branch and re-answers ``identity_source``, ``git_commit``,
      ``git_tree``, ``git_status_short`` and ``installed_wheel`` /
      ``installed_editable`` from a lower rung, all at once (that one is
      now also retried -- see :func:`_runtime_provenance` -- so reaching
      this comparison degraded is rare);
    * the run creating ONE untracked file inside the checkout, which
      appends a ``??`` row to ``git_status_short`` and increments
      ``installed_editable.git.untracked_files``.  ``runs/`` is not
      ignored, so that is not an unusual run: it is a run that writes
      where it was told to.

    So the components are compared on their own terms:

    * ``source_sha256`` is compared ALWAYS and strictly.  It is read
      from bytes on disk, it hashes the ten modules that ARE this
      forecast implementation, it cannot fail to resolve, and it is what
      makes skipping anything else survivable.
    * ``git_commit`` and ``git_tree`` are compared whenever BOTH ends
      resolved them.  ``None`` means "this was not asked", which is an
      unanswered question rather than a changed answer.
      :func:`_runtime_provenance` guarantees both describe REPO, so they
      stay comparable even when the rung underneath them changed.
    * ``git_status_short`` is compared, both ends resolved, over TRACKED
      paths only.
    * ``distribution_manifest_sha256``, ``installed_wheel``,
      ``installed_editable`` and ``installed_source_content`` are
      compared only when the SAME rung answered at both ends.  A sealed
      manifest, a wheel RECORD, an editable git tree and a source tree's
      content answer different questions, and one rung's answer
      measured against another's is not a comparison.
    * ``identity_source`` is not compared at all.  It names which
      RESOLVER answered, not which code is executing, and the only way
      it moves while every digest above holds is the fall-through this
      exists for.  An install genuinely replaced mid-run rewrites the
      ten hashed modules and is caught by ``source_sha256``.

    That is narrower than the old test in exactly one case: a commit
    that lands mid-run, touches none of the ten hashed files and no
    tracked file at all, AND coincides with a git failure at one end.
    Every other real change still fails: a tracked edit to a hashed file
    moves ``source_sha256``, a tracked edit anywhere else moves the
    filtered ``git_status_short``, and a HEAD that moves while git works
    moves ``git_commit``.
    """

    if before == after:
        return None
    # 1. The bytes on disk.  Always, strictly.
    first = dict(before.get("source_sha256") or {})
    second = dict(after.get("source_sha256") or {})
    for name in sorted(set(first) | set(second)):
        if first.get(name) != second.get(name):
            return (f"source_sha256 {name} {first.get(name)} -> "
                    f"{second.get(name)}")
    # 2. The commit and the tree, whenever both ends resolved them.
    #    _runtime_provenance guarantees both describe REPO and no other
    #    tree, so they stay comparable even across a changed rung.
    for field in ("git_commit", "git_tree"):
        one, other = before.get(field), after.get(field)
        if one is None or other is None:
            continue
        if one != other:
            return f"{field} {one!r} -> {other!r}"
    # 3. The rung's OWN artifact, only when the same rung answered at
    #    both ends.  A wheel RECORD, an editable source tree and a
    #    sealed manifest answer three different questions; comparing one
    #    rung's answer against another's is not a comparison at all.
    if before.get("identity_source") == after.get("identity_source"):
        for field in ("distribution_manifest_sha256", "installed_wheel",
                      "installed_editable", "installed_source_content"):
            one = _untracked_blind(before.get(field))
            other = _untracked_blind(after.get(field))
            if one is None or other is None:
                continue
            if one != other:
                return f"{field} {one!r} -> {other!r}"
    # 4. The working tree, tracked paths only.
    one = _tracked_status_lines(before.get("git_status_short"))
    other = _tracked_status_lines(after.get("git_status_short"))
    if one is not None and other is not None and one != other:
        return f"git_status_short {one!r} -> {other!r}"
    return None


def _runtime_source_identity() -> dict[str, object]:
    """Bind the exact forecast implementation, not only preparation code."""

    paths = (
        REPO / "woof/core/model.py",
        REPO / "woof/ingest/hrrr_physics.py",
        REPO / "woof/ingest/hrrr_surface.py",
        REPO / "woof/ingest/lateral_bc.py",
        REPO / "woof/ingest/prepared_cache.py",
        REPO / "woof/ingest/real.py",
        REPO / "woof/io/wrfout.py",
        REPO / "woof/native_wrf_contract.py",
        REPO / "woof/source_authorities.py",
        REPO / "woof/prepared_single_domain_forecast.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"prepared forecast runtime sources are missing: {missing}")
    # ``.as_posix()`` is the one spelling every serialized identity key
    # in this product uses; ``str()`` of a relative Path is the Windows
    # bug it replaced.  Same bytes as the ``.replace`` this had.
    source_sha256 = {
        path.relative_to(REPO).as_posix(): _sha256(path)
        for path in paths
    }
    # One resolver for all three installs (woof.runtime_manifest).  The
    # branch this replaced degraded a wheel install to
    # `runtime-module-sha256-only` -- accurate, but weaker than the truth:
    # pip knows exactly which artifact it wrote, and RECORD says so.  It
    # also raised, uncaught, when site-packages happened to sit INSIDE
    # some unrelated repository, binding a stranger's commit or dying;
    # that case is now simply "not a checkout of this tree".
    #
    # The KEYS this publishes, and their meanings, are unchanged.  This
    # mapping is hashed into ``experiment_fingerprint``, which is
    # published in checkpoint headers, so a new field here would move
    # every already-sealed fingerprint and strand the legs carrying
    # them.  What improved is the RELIABILITY of the resolution
    # (:func:`_runtime_provenance`) and the end-of-run COMPARISON, which
    # moved out to :func:`_runtime_source_identity_change`.
    return {**_runtime_provenance(), "source_sha256": source_sha256}


def _durable_wrfout_inventory(output_directory: Path) -> list[dict[str, object]]:
    """Inventory atomic-complete files retained by a failed or passing run."""

    inventory = []
    for path in sorted((Path(output_directory) / "wrfout").glob("wrfout_d??_*")):
        if not path.is_file():
            continue
        inventory.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return inventory


def _hierarchy_rows(exp) -> list[dict[str, object]]:
    return [{
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
        "mp_physics": int(domain.run.mp_physics),
    } for domain in exp.domains]


def _resolve_prepared_layout(
        *, source: str, prepared_root: Path, proof: Mapping[str, object],
        source_exp, domain_bundle: Path | None,
) -> _PreparedLayout:
    """Resolve either a direct bundle or hash-bound hierarchy d01 bundle."""

    schema = proof.get("schema")
    if proof.get("status") != "READY_NOT_YET_STOCK_WRF_GATED":
        raise ValueError(
            f"{source.upper()} preparation proof is not READY for forecast")
    if source == "hrrr":
        if schema != _PROOF_SCHEMA[source]:
            raise ValueError(
                f"unsupported HRRR preparation proof schema {schema!r}; "
                "this runner reads the single-domain bundle "
                "tools/prepare_hrrr_wrf.py publishes.  A prepared HRRR "
                "domain TREE is the other route: run it with "
                "woof-prepared-tree-forecast (module form: python -m "
                "woof.prepared_domain_tree_forecast)")
        if len(source_exp.domains) != 1:
            raise ValueError(
                "the HRRR single-domain bundle requires a one-domain "
                "experiment; a domain tree runs on the tree runner")
        expected = prepared_root.resolve()
        if domain_bundle is not None \
                and Path(domain_bundle).resolve() != expected:
            raise ValueError(
                "HRRR preparation --domain-bundle must equal --prepared-root")
        return _PreparedLayout(
            kind=HRRR_DIRECT_LAYOUT,
            domain_bundle=expected,
            static_path=_require_file(
                expected / HRRR_BUNDLE_PATHS["static"],
                "native static cache"),
            geometry_receipt_path=_require_file(
                expected / HRRR_BUNDLE_PATHS["geometry_receipt"],
                "geometry receipt"),
            prepared_cache_path=_require_directory(
                expected / HRRR_BUNDLE_PATHS["prepared_cache"],
                "prepared cache"),
            authority_paths=MappingProxyType({
                "bridge_manifest":
                    expected / HRRR_BUNDLE_PATHS["bridge_manifest"],
            }),
        )
    if (schema == _PROOF_SCHEMA[source]
            or schema in _LEGACY_PROOF_SCHEMAS[source]):
        if len(source_exp.domains) != 1:
            raise ValueError(
                "single-domain preparation proof requires a one-domain "
                "experiment")
        expected = prepared_root.resolve()
        if domain_bundle is not None and Path(domain_bundle).resolve() != expected:
            raise ValueError(
                "direct preparation --domain-bundle must equal --prepared-root")
        return _PreparedLayout(
            kind=(
                "mapped-direct-d01-v1"
                if source in _MAPPED_SOURCES
                else "portable-single-domain-v2"),
            domain_bundle=expected,
            static_path=_require_file(
                expected / "native-static.npz", "native static cache"),
            geometry_receipt_path=_require_file(
                expected / "geometry-receipt.json", "geometry receipt"),
            prepared_cache_path=_require_directory(
                expected / "prepared-cache", "prepared cache"),
            authority_paths=MappingProxyType({}),
        )
    if (schema != _HIERARCHY_PROOF_SCHEMA[source]
            and schema not in _LEGACY_HIERARCHY_PROOF_SCHEMAS[source]):
        raise ValueError(
            f"unsupported {source.upper()} preparation proof schema {schema!r}")
    if len(source_exp.domains) < 2:
        raise ValueError(
            "hierarchy preparation proof requires a multi-domain experiment")
    expected_ids = list(range(1, len(source_exp.domains) + 1))
    if (proof.get("domain_count") != len(source_exp.domains)
            or [int(domain.grid_id) for domain in source_exp.domains]
            != expected_ids):
        raise ValueError(
            "hierarchy proof/domain configuration is not contiguous parent-first")

    hierarchy_root = _require_directory(
        prepared_root / "hierarchy-artifacts", "hierarchy artifact root")
    artifact_manifest_path = _require_file(
        hierarchy_root / "domain-artifacts.json",
        "hierarchy artifact manifest")
    hierarchy_receipt_path = _require_file(
        hierarchy_root / "receipt.json", "hierarchy artifact receipt")
    # Both runners restore this prepared cache. The WRF file set is an
    # optional companion; when present its redundant lineage still binds.
    export_slot = proof.get("wrf_manifest")
    without_export = _optional_stock_wrf_export(proof, export_slot)
    wrf_manifest_path = None if without_export else _require_file(
        prepared_root / "wrf-native-input" / "manifest.json",
        "hierarchy direct-WRF manifest")
    artifact_manifest = _load_json_object(
        artifact_manifest_path, "hierarchy artifact manifest")
    hierarchy_receipt = _load_json_object(
        hierarchy_receipt_path, "hierarchy artifact receipt")
    wrf_manifest = dict(export_slot) if without_export else _load_json_object(
        wrf_manifest_path, "hierarchy direct-WRF manifest")
    expected_manifest = {
        "schema": "gpuwm-native-domain-artifacts-v1",
        "domains": [{
            "grid_id": grid_id,
            "prepared_cache": f"domains/d{grid_id:02d}/prepared-cache",
            "static_cache": f"domains/d{grid_id:02d}/native-static.npz",
            "geometry_receipt": (
                f"domains/d{grid_id:02d}/geometry-receipt.json"),
        } for grid_id in expected_ids],
    }
    if artifact_manifest != expected_manifest:
        raise ValueError(
            "hierarchy artifact manifest is not the canonical domain layout")
    expected_receipt_identity = {
        "schema": "gpuwm-native-hierarchy-artifact-build-v1",
        "status": "READY",
        "domain_count": len(expected_ids),
        "grid_ids": expected_ids,
        "manifest": {
            "path": "domain-artifacts.json",
            "sha256": _sha256(artifact_manifest_path),
        },
        "boundary_inventory": {
            "external": [1],
            "nested_parent_forced": expected_ids[1:],
        },
    }
    if any(hierarchy_receipt.get(key) != value
           for key, value in expected_receipt_identity.items()):
        raise ValueError("hierarchy artifact receipt identity is inconsistent")
    domain_receipts = hierarchy_receipt.get("domains")
    if (not isinstance(domain_receipts, list)
            or len(domain_receipts) != len(expected_ids)):
        raise ValueError("hierarchy domain receipt inventory is incomplete")
    if proof.get("artifact_receipt") != hierarchy_receipt:
        raise ValueError(
            "hierarchy proof artifact receipt differs from the published tree")
    if proof.get("wrf_manifest") != wrf_manifest:
        raise ValueError(
            "hierarchy proof WRF manifest differs from the published tree")

    expected_bundle = _require_directory(
        hierarchy_root / "domains" / "d01", "hierarchy d01 bundle")
    if (domain_bundle is not None
            and Path(domain_bundle).resolve() != expected_bundle):
        raise ValueError(
            "--domain-bundle differs from the manifest-derived hierarchy d01")
    domain_receipt_path = _require_file(
        expected_bundle / "receipt.json", "hierarchy d01 receipt")
    domain_receipt = _load_json_object(
        domain_receipt_path, "hierarchy d01 receipt")
    if domain_receipt != domain_receipts[0]:
        raise ValueError(
            "hierarchy d01 receipt differs from the hierarchy authority")
    expected_domain_identity = {
        "schema": "gpuwm-native-domain-artifact-build-v1",
        "status": "READY",
        "grid_id": 1,
        "parent_id": 0,
        "boundary_mode": "external-specified",
        "valid_time": source_exp.start_time.isoformat(),
        "forcing_hours": proof.get("forcing_hours"),
    }
    if any(domain_receipt.get(key) != value
           for key, value in expected_domain_identity.items()):
        raise ValueError("hierarchy d01 receipt identity is inconsistent")
    hierarchy_authority_paths = {
        "hierarchy_artifact_manifest": artifact_manifest_path,
        "hierarchy_receipt": hierarchy_receipt_path,
        "domain_receipt": domain_receipt_path,
        **({"wrf_manifest": wrf_manifest_path} if wrf_manifest_path else {}),
    }
    if source in _MAPPED_SOURCES:
        hierarchy_authority_paths.update({
            "mapped_root_static": _require_file(
                prepared_root / "native-static.npz",
                "mapped hierarchy root static cache"),
            "mapped_root_geometry": _require_file(
                prepared_root / "geometry-receipt.json",
                "mapped hierarchy root geometry receipt"),
        })
    return _PreparedLayout(
        kind=(
            "mapped-hierarchy-d01-v1"
            if source in _MAPPED_SOURCES else "hierarchy-d01-v1"),
        domain_bundle=expected_bundle,
        static_path=_require_file(
            expected_bundle / "native-static.npz", "hierarchy d01 static cache"),
        geometry_receipt_path=_require_file(
            expected_bundle / "geometry-receipt.json",
            "hierarchy d01 geometry receipt"),
        prepared_cache_path=_require_directory(
            expected_bundle / "prepared-cache", "hierarchy d01 prepared cache"),
        authority_paths=MappingProxyType(hierarchy_authority_paths),
        domain_receipt=MappingProxyType(domain_receipt),
        hierarchy_receipt=MappingProxyType(hierarchy_receipt),
        artifact_manifest=MappingProxyType(artifact_manifest),
        wrf_manifest=MappingProxyType(wrf_manifest),
    )


def _twentycrv3_manifest_file_specs(
        manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Validate the portable copy of one exact-member 20CRv3 manifest.

    The raw GRIB paths intentionally remain historical evidence: a prepared
    tree must remain runnable after those large inputs are moved or deleted.
    Their declared byte/hash identities are nevertheless bound into the
    copied manifest, the mapped composition receipt, and the cache identity.
    """

    expected_keys = {
        "schema", "source", "member_identity", "member", "valid_times",
        "cadence_seconds", "file_count", "files", "content_sha256",
    }
    if (set(manifest) != expected_keys
            or manifest.get("schema") != _SOURCE_SCHEMA["20crv3"]
            or manifest.get("source") != _TWENTYCRV3_SOURCE
            or manifest.get("member_identity") != _TWENTYCRV3_MEMBER_IDENTITY):
        raise ValueError(
            "20CRv3 portable source manifest has an unsupported identity or "
            "top-level inventory")
    content = dict(manifest)
    declared_content_sha256 = content.pop("content_sha256")
    if declared_content_sha256 != hashlib.sha256(
            _canonical(content).encode("utf-8")).hexdigest():
        raise ValueError("20CRv3 portable source manifest content hash is stale")
    member = manifest.get("member")
    if not isinstance(member, str) or re.fullmatch(r"[0-9]{3}", member) is None:
        raise ValueError("20CRv3 manifest member must be a three-digit label")
    rows = manifest.get("files")
    if (not isinstance(rows, list)
            or isinstance(manifest.get("file_count"), bool)
            or manifest.get("file_count") != len(rows)):
        raise ValueError("20CRv3 portable source manifest file count differs")

    normalized: dict[str, dict[str, object]] = {}
    observed: dict[datetime, set[str]] = {}
    observed_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "path", "filename", "member", "valid_time", "role",
                "bytes", "sha256"}:
            raise ValueError(f"20CRv3 manifest file row {index} is malformed")
        path_value = row.get("path")
        filename = row.get("filename")
        if (not isinstance(path_value, str) or not path_value
                or not isinstance(filename, str)
                or receipt_basename(path_value) != filename):
            raise ValueError(
                f"20CRv3 manifest file row {index} has an unsafe path/name")
        match = _TWENTYCRV3_FILENAME.fullmatch(filename)
        if match is None:
            raise ValueError(
                f"20CRv3 manifest file row {index} has an invalid filename")
        try:
            valid_time = datetime.strptime(
                match.group("time"), "%Y%m%d%H")
        except ValueError as exc:
            raise ValueError(
                f"20CRv3 manifest file row {index} has an invalid time") from exc
        if (match.group("member") != member or row.get("member") != member
                or row.get("role") != match.group("role")
                or row.get("valid_time") != valid_time.isoformat()):
            raise ValueError(
                f"20CRv3 manifest file row {index} disagrees with its filename")
        byte_count = row.get("bytes")
        if (isinstance(byte_count, bool) or not isinstance(byte_count, int)
                or byte_count <= 0):
            raise ValueError(
                f"20CRv3 manifest file row {index} has invalid byte count")
        digest = _require_digest(
            row.get("sha256"), f"20CRv3 manifest row {index} sha256")
        if path_value in observed_paths or filename in normalized:
            raise ValueError("20CRv3 manifest contains duplicate inputs")
        observed_paths.add(path_value)
        observed.setdefault(valid_time, set()).add(match.group("role"))
        normalized[filename] = {
            "name": filename,
            "path": path_value,
            "member": member,
            "valid_time": valid_time.isoformat(),
            "role": match.group("role"),
            "bytes": byte_count,
            "sha256": digest,
        }
    if len(observed) < 2 or any(
            roles != {"pl", "sfc"} for roles in observed.values()):
        raise ValueError(
            "20CRv3 manifest requires at least two complete pressure/surface "
            "pairs")
    times = tuple(sorted(observed))
    valid_times = [value.isoformat() for value in times]
    if manifest.get("valid_times") != valid_times:
        raise ValueError("20CRv3 manifest valid-time inventory differs")
    deltas = {
        int((later - earlier).total_seconds())
        for earlier, later in zip(times, times[1:])
    }
    cadence_seconds = manifest.get("cadence_seconds")
    if (len(deltas) != 1 or isinstance(cadence_seconds, bool)
            or not isinstance(cadence_seconds, int)
            or cadence_seconds != next(iter(deltas))
            or cadence_seconds <= 0 or cadence_seconds % 3600 != 0):
        raise ValueError(
            "20CRv3 manifest cadence must be uniform, positive, and whole-hour")
    return normalized


#: The three fields that ARE the GFS cycle identity.  Bound strictly: a
#: manifest authored for another cycle is refused before any GPU setup.
_GFS_MANIFEST_IDENTITY_KEYS = ("model", "product", "cycle")

#: The fetch provenance a GFS front-door manifest's ``source`` object may
#: also carry: the pressure ladder the fetch actually took and the source
#: top it implies.  `woof fetch` began writing these in `1f0fc039` so the
#: preparation lane's vertical contract could stop assuming a 100 hPa
#: constant, and `woof/gfs_direct.py` reads them BEFORE the bridge runs.
#: They are real provenance, not identity -- a deeper fetch changes them
#: and changes nothing about which cycle the document names -- so folding
#: them into the identity comparison is what refused every manifest this
#: release's own fetch authors.  Enumerated rather than ignored: a field
#: nobody here knows about is still a refusal.
_GFS_MANIFEST_LEVEL_KEYS = ("pressure_levels_hpa", "top_pressure_pa")


def proof_initial_forecast_lead(proof: Mapping[str, object]) -> int:
    """The source lead the prepared run starts from, per its own proof.

    A proof written before initialization from a forecast lead existed
    carries no such block, and there is exactly one lead it can have
    meant: 0, the cycle's analysis.  Reading it as 0 is a statement of
    what those runs WERE, not a default applied to a missing field.
    """

    block = proof.get("initial_condition")
    if block is None:
        return 0
    if not isinstance(block, Mapping):
        raise ValueError(
            "preparation proof initial-condition provenance is malformed")
    lead = block.get("initial_forecast_lead_hours")
    if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
        raise ValueError(
            "preparation proof declares an unreadable initial forecast lead")
    kind = block.get("initial_condition_kind")
    process = block.get("forecast_generating_process_id")
    # A lead is never readable as an analysis, in either direction.
    if (kind != ("analysis" if lead == 0 else "forecast")
            or process != (81 if lead == 0 else 96)):
        raise ValueError(
            f"preparation proof labels forecast lead f{lead:03d} as "
            f"{kind!r} (process {process!r}); a forecast lead is not an "
            "analysis")
    return lead


def _gfs_manifest_source_receipt(
        manifest: Mapping[str, object], exp,
        proof: Mapping[str, object]) -> dict[str, object]:
    """Bind the GFS cycle identity; validate and record the level ladder.

    The identity comparison binds ``model``/``product``/``cycle`` and
    compares those three exactly.  The level fields are routed to the
    contract they serve instead: ``ingest.pressure_ladder.manifest_source_top_pa`` is
    the producing lane's own validator -- it refuses an unreadable or
    non-certified ladder and cross-checks the declared top against it --
    and the resulting source top is held to the same rule the vertical
    contract applies, that the source atmosphere must reach at least as
    high as the requested model top.  Using the producer's validator is
    deliberate: two readers of one document that disagree about it is the
    defect this function exists to close.
    """

    from woof.ingest.pressure_ladder import manifest_source_top_pa

    identity = manifest.get("source")
    if not isinstance(identity, dict):
        raise ValueError("GFS source manifest carries no source identity")
    known = set(_GFS_MANIFEST_IDENTITY_KEYS) | set(_GFS_MANIFEST_LEVEL_KEYS)
    unknown = sorted(set(identity) - known)
    if unknown:
        # Forward-compat: a NEWER fetch may stamp fields this runner
        # does not know.  Every field this runner reads is still bound
        # and checked below; extras are named and ignored.  (The old
        # closed-set refusal rejected every manifest this release's
        # own fetch authored.)
        warn(f"GFS source manifest identity carries field(s) {unknown} "
             "this runner does not read; ignoring them")
    bound = {key: identity.get(key) for key in _GFS_MANIFEST_IDENTITY_KEYS}
    # start_time is the model's time zero and the cycle is the source's;
    # they coincide only at lead 0.  Deriving the cycle by subtracting
    # the proof's declared lead keeps the binding exact at every lead
    # instead of demanding that a run start when its source did.
    lead_hours = proof_initial_forecast_lead(proof)
    cycle_time = exp.start_time - timedelta(hours=lead_hours)
    expected = {
        "model": "GFS",
        "product": "pgrb2.0p25",
        "cycle": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if bound != expected:
        raise ValueError(
            "GFS source manifest identity differs from the cycle this "
            "experiment starts from (start_time "
            f"{exp.start_time:%Y-%m-%d %H:%M:%S} at forecast lead "
            f"f{lead_hours:03d})")
    provenance = proof.get("initial_condition")
    if isinstance(provenance, Mapping) and (
            provenance.get("cycle") != expected["cycle"]
            or provenance.get("model_start_time")
            != exp.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")):
        raise ValueError(
            "preparation proof initial-condition provenance disagrees with "
            "the manifest cycle or the experiment start_time")
    # Absent levels mean a directory fetched by an older ArWen, whose only
    # possible ladder is the certified 100 hPa one; the validator returns
    # that constant, which is exactly what the preparation assumed.
    source_top_pa = manifest_source_top_pa(manifest, source="GFS")
    p_top_pa = float(exp.vertical.p_top)
    if source_top_pa > p_top_pa:
        raise ValueError(
            f"GFS source manifest records a fetch reaching {source_top_pa:g} "
            f"Pa but the experiment requests p_top {p_top_pa:g} Pa")
    levels = identity.get("pressure_levels_hpa")
    return {
        "schema": "gpuwm-gfs-source-manifest-identity-v1",
        "identity": bound,
        "pressure_levels_hpa": (
            [float(level) for level in levels]
            if isinstance(levels, list) else None),
        "source_top_pressure_pa": source_top_pa,
        "experiment_p_top_pa": p_top_pa,
    }


def _mapped_composition_manifest_file_specs(
        source: str, manifest: Mapping[str, object], *,
        planned: bool = False,
) -> dict[str, dict[str, object]]:
    """Validate the portable copy of one composed mapped input manifest.

    ``gpuwm-mapped-composition-inputs-v1`` is a different document from the
    per-source manifests: it binds the two AUTHORITIES (mapping and
    composition) plus every input file by path and digest, and it carries
    no experiment config or namelist -- those are bound by the proof's own
    ``execution_inputs`` receipt, which
    :func:`_validate_packaged_mapped_evidence` checks against the files
    the caller actually supplied.  So this reader validates the shape and
    the identity chain and returns the file inventory; it does NOT invent
    the roles the named-source manifests carry, because a role this
    document does not declare is a role nothing can bind.

    ``planned``: the manifest is an as-posted head's input plan, whose lead
    rows carry no byte count or digest until the seal writes the manifest
    (``mapped_authoring.planned_input_manifest``); such a row is bound by
    its path, with ``sha256`` ``None``, and the seal holds every row to the
    plan and to its lead's posted marker.
    """

    expected_keys = {
        "schema", "mapping_sha256", "composition_sha256", "primary_files",
        "supplements", "provenance", "decoders",
    }
    member_keys = {"member", "member_identity"} & set(manifest)
    if (member_keys and (member_keys != {"member", "member_identity"}
            or any(not isinstance(manifest[key], str) or not manifest[key]
                   for key in member_keys))):
        raise ValueError("mapped manifest member identity is incomplete")
    if (set(manifest) - member_keys != expected_keys
            or manifest.get("schema") != _SOURCE_SCHEMA[source]):
        raise ValueError(
            f"{source} portable source manifest has an unsupported schema or "
            "top-level inventory")
    for role in ("mapping_sha256", "composition_sha256"):
        _require_digest(manifest.get(role), f"mapped manifest {role}")
    normalized: dict[str, dict[str, object]] = {}

    def bind(label: str, rows: object) -> None:
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"mapped composition manifest {label} inventory is empty")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {
                    "path", "bytes", "sha256"}:
                raise ValueError(
                    f"mapped composition manifest {label} row {index} is "
                    "malformed")
            path_value = row.get("path")
            byte_count = row.get("bytes")
            pending = (planned and byte_count is None
                       and row.get("sha256") is None)
            if (not isinstance(path_value, str) or not path_value
                    or (not pending and (
                        isinstance(byte_count, bool)
                        or not isinstance(byte_count, int)
                        or byte_count <= 0))):
                raise ValueError(
                    f"mapped composition manifest {label} row {index} has an "
                    "unsafe path or byte count")
            key = f"{label}[{index}]"
            if key in normalized:
                raise ValueError("mapped composition manifest repeats a role")
            normalized[key] = {
                "name": receipt_basename(path_value),
                "path": path_value,
                "bytes": byte_count,
                "sha256": (None if pending else _require_digest(
                    row.get("sha256"), f"mapped manifest {key} sha256")),
            }

    bind("primary", manifest.get("primary_files"))
    supplements = manifest.get("supplements")
    if not isinstance(supplements, dict) or not supplements:
        raise ValueError(
            "mapped composition manifest binds no supplement; a packaged "
            "profile's terrain has to come from somewhere named")
    for role, rows in sorted(supplements.items()):
        bind(f"supplement:{role}", rows)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("mapped composition manifest binds no provenance")
    for role, row in sorted(provenance.items()):
        bind(f"provenance:{role}", [row])
    decoders = manifest.get("decoders")
    if not isinstance(decoders, dict):
        raise ValueError("mapped composition manifest decoders must be an object")
    for role, row in sorted(decoders.items()):
        bind(f"decoder:{role}", [row])
    return normalized


def _manifest_file_specs(
        source: str, manifest: Mapping[str, object], exp,
        proof: Mapping[str, object], *,
        pending=frozenset(), planned: bool = False,
) -> tuple[dict[str, dict[str, object]], Mapping[str, object] | None]:
    """Normalized role specs, plus the GFS source receipt when there is one.

    ``pending`` names the roles whose digest an as-posted head's input
    plan leaves unknown (each lead's payload and what is written from the
    lead rows): each carries ``sha256`` ``None`` until the seal writes the
    manifest, which is held to the plan and to each lead's posted marker
    (:func:`woof.ingest.boundary_stream.verify_as_posted_seal`).
    """

    if source == "20crv3":
        return _twentycrv3_manifest_file_specs(manifest), None
    if source in _MAPPED_SOURCES:
        return _mapped_composition_manifest_file_specs(
            source, manifest, planned=planned), None
    expected_schema = _SOURCE_SCHEMA[source]
    expected_keys = {"schema", "files", "source"} \
        if source in {"gfs", "hrrr"} else {"schema", "files"}
    if set(manifest) != expected_keys or manifest.get("schema") != expected_schema:
        raise ValueError(
            f"{source.upper()} portable source manifest has an unsupported "
            "schema or top-level inventory")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("portable source manifest files must be an object")
    normalized = {}
    for role, spec in files.items():
        if (not isinstance(role, str) or not role
                or not isinstance(spec, dict)
                or set(spec) != {"name", "sha256"}):
            raise ValueError(
                f"portable source manifest role {role!r} is malformed")
        name = spec.get("name")
        if (not isinstance(name, str) or not name
                or receipt_basename(name) != name
                or Path(name).is_absolute()):
            raise ValueError(
                f"portable source manifest role {role!r} has an unsafe name")
        normalized[role] = {
            "name": name,
            "sha256": (None if role in pending and spec.get("sha256") is None
                       else _require_digest(
                           spec.get("sha256"),
                           f"source manifest {role} sha256")),
        }
    common = {"bridge", "experiment_config", "wps_namelist"}
    if source == "hrrr":
        # HRRR's portable manifest binds one more authority than the
        # others: the WRF namelist.input.  It is not decoration -- the
        # prepared-cache identity's ``namelist_sha256`` IS that file's
        # digest on this route (the native preparer reads the explicit
        # eta ladder and p_top out of it), so a bundle that did not name
        # it could not have its identity recomputed here at all.
        required = common | {"namelist_input", "source_manifest"}
        optional = {"static_input", "static_receipt", "domain_spec"}
        if not required <= set(normalized) \
                or set(normalized) - required - optional:
            raise ValueError(
                "HRRR source manifest role inventory is unsupported: "
                f"{sorted(normalized)}")
        static_pair = {"static_input", "static_receipt"} & set(normalized)
        if static_pair not in (set(), {"static_input", "static_receipt"}):
            raise ValueError("HRRR source manifest static roles are incomplete")
        return normalized, None
    if source == "gfs":
        allowed = common | {"series", "static_input", "static_receipt"}
        dynamic = {role for role in normalized if role.startswith("grib-f")}
        unknown = set(normalized) - allowed - dynamic
        if unknown or not common | {"series"} <= set(normalized):
            raise ValueError(
                f"GFS source manifest role inventory is unsupported: "
                f"{sorted(normalized)}")
        static_pair = {"static_input", "static_receipt"} & set(normalized)
        if static_pair not in (set(), {"static_input", "static_receipt"}):
            raise ValueError("GFS source manifest static roles are incomplete")
        parsed_hours = []
        for role in dynamic:
            suffix = role.removeprefix("grib-f")
            if len(suffix) != 3 or not suffix.isdigit():
                raise ValueError(f"invalid GFS GRIB role {role!r}")
            parsed_hours.append(int(suffix))
        if len(parsed_hours) < 2 or len(set(parsed_hours)) != len(parsed_hours):
            raise ValueError("GFS manifest requires at least two unique GRIB hours")
        return normalized, _gfs_manifest_source_receipt(
            manifest, exp, proof)
    else:
        required = common | {"grib", "vtable"}
        optional = {"static_input", "static_receipt", "source_orography"}
        if not required <= set(normalized) or set(normalized) - required - optional:
            raise ValueError(
                f"ERA5 source manifest role inventory is unsupported: "
                f"{sorted(normalized)}")
        static_pair = {"static_input", "static_receipt"} & set(normalized)
        if static_pair not in (set(), {"static_input", "static_receipt"}):
            raise ValueError("ERA5 source manifest static roles are incomplete")
    return normalized, None


def _validate_profile_switches(
        exp, *, source: str, profile: str, all_domains: bool = False,
) -> dict[str, object]:
    """Validate descriptor physics before cache identity or GPU setup."""

    expected = _profile_runtime_switches(source, profile)
    domains = exp.domains if all_domains else (exp.root,)
    observed_domains = []
    expected_radiation = (
        (int(expected["ra_physics"]), int(expected["ra_physics"]))
        if (int(expected["ra_lw_physics"]),
            int(expected["ra_sw_physics"])) == (-1, -1)
        else (int(expected["ra_lw_physics"]),
              int(expected["ra_sw_physics"]))
    )
    labels = {
        PHYSICS_PROFILE: "supported prepared-cache profile",
        TWENTYCRV3_WSM6_PHYSICS_PROFILE: (
            "implemented-unverified 20CRv3 WSM6+KF+RTE profile"),
        THOMPSON_PHYSICS_PROFILE: "guarded Thompson MP8 profile",
        MORRISON_PHYSICS_PROFILE: "Morrison MP10 runtime profile",
        NSSL2_PHYSICS_PROFILE: "NSSL-2 wrf-matched-run-candidate profile",
        NSSL2_LEGACY_RRTMG_PHYSICS_PROFILE: (
            "NSSL-2 + legacy RRTMG wrf-matched-run-candidate profile"),
        MYNN_PHYSICS_PROFILE: "MYNN 5/5 implemented-unverified profile",
        RUC_PHYSICS_PROFILE: "RUC LSM implemented-unverified profile",
        MYNN_RUC_PHYSICS_PROFILE: (
            "MYNN 5/5 + RUC implemented-unverified profile"),
        NOAHMP_PHYSICS_PROFILE: "Noah-MP expert profile",
        MYNN_NOAHMP_PHYSICS_PROFILE: "MYNN 5/5 + Noah-MP expert profile",
    }
    for domain in domains:
        cfg = domain.run
        observed = {name: getattr(cfg, name) for name in expected}
        observed_radiation = radiation_scheme_ids(cfg)
        # Compared through the ONE spelling, the same reconciliation the
        # capability door and the prepared-cache identity both use: the
        # aggregate radiation selector and a cumulus-off cudt_minutes are
        # how a WRF namelist writes the run a shipped profile writes
        # differently, so a key-by-key comparison here refused a
        # configuration that resolves to this profile switch for switch --
        # every namelist-routed preparation of a coupled-radiation or
        # cumulus-off profile, with the resolved radiation pair printed
        # EQUAL in the same sentence that refused it.
        compared = selection_values_one_spelling(cfg, expected)
        canonical = selection_values_one_spelling(expected, expected)
        # ONE VOCABULARY WITH THE CAPABILITY DOOR.  A namelist-routed
        # run meets both refusals on the same command, and they named
        # the same idea two ways: `settings={name: {selected, expected}}`
        # there and `differs={name: {observed, expected}}` here.  The two
        # reconciliations are one function now
        # (woof.physics_compat.selection_values_one_spelling); the two
        # sentences say it the same way.
        drift = {name: {"selected": compared[name],
                        "expected": canonical[name]}
                 for name in expected if compared[name] != canonical[name]}
        if drift or observed_radiation != expected_radiation:
            label = labels.get(profile, f"named {profile} profile")
            raise ValueError(
                f"experiment physics differs from the {label} on "
                f"d{int(domain.grid_id):02d}: settings={drift}, "
                f"expected={expected}, "
                f"radiation={expected_radiation}, observed={observed}, "
                f"resolved_radiation={observed_radiation}.  A named "
                f"--physics-profile asserts the experiment IS that suite; "
                f"omit the flag to run the experiment's own physics as "
                f"written")
        observed_domains.append({
            "grid_id": int(domain.grid_id),
            **observed,
            "radiation_scheme_ids": list(observed_radiation),
        })
    readiness, warning = _profile_readiness(source, profile)
    return {
        "schema": "gpuwm-prepared-physics-profile-v1",
        "source": source,
        "profile": profile,
        "readiness": readiness,
        "warning": warning,
        "warning_only": warning is not None,
        "resolved": {
            **expected,
            "radiation_scheme_ids": list(expected_radiation),
        },
        "validated_domains": observed_domains,
    }


def _validate_physics(
        exp, profile: str | None, run_seconds: float,
        history_interval_seconds: float, *, source: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
) -> dict[str, object]:
    """Admit the experiment's physics and build its receipt.

    ``profile`` is an OPTIONAL assertion (owner ruling 2026-07-31): when
    named, the experiment must BE that shipped suite, switch for switch,
    exactly as before.  When omitted, the hash-bound experiment config is
    the authority and any suite the engine implements runs; its
    verification status -- "WRF-verified" against a shipped profile, or
    "supported, not yet WRF-verified" -- is recorded in the receipt and
    never gates.  Fail-closed refusals remain only where the code
    genuinely does not implement the request: engine selector validation
    (``woof.config.validate_run_config``, which names the exact switch)
    and the registry's land-surface route declaration.

    The registry's source-neutral tuple governance is APPLIED here too,
    on every prepared source, for named, matched and unnamed suites
    alike -- but as a WARNING, not a refusal (warn-not-block ruling):
    an expert-template tuple is implemented and individually verified,
    so an unacknowledged one runs and prints one line naming both
    published delivery spellings, ``--ack <id>`` and
    ``acknowledgements = ["<id>"]`` in the hash-bound experiment, and
    the receipt carries ``acknowledged=false`` either way.  Applying
    the governance is what keeps the removal of the profile whitelist
    from widening consent SILENTLY: pre-ruling, expert tuples could not
    reach the non-GFS lanes at all, and every run record now states
    exactly which unblessed tuple executed.  A caller who NAMES an
    expert ``--physics-profile`` has asked for that gate and still
    meets it at the front door
    (:func:`woof.physics_compat.validate_single_domain_physics_profile`).
    """

    if len(exp.domains) != 1:
        raise ValueError(
            "prepared single-domain forecast runner requires exactly one domain")
    dc = exp.root
    cfg = dc.run
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError("run-seconds must be a positive duration")
    if run_seconds % 3600.0 != 0.0:
        warn(f"run-seconds {run_seconds:g} is not a whole number of "
             "hours; continuing (forcing coverage is checked "
             "separately)")
    if float(exp.run_seconds) != float(run_seconds) \
            or float(cfg.run_seconds) != float(run_seconds):
        # The executed experiment carries the requested length
        # (_apply_run_length runs before this on every route); a
        # mismatch here is a caller that skipped it, not a user error.
        raise ValueError(
            f"run-seconds {run_seconds:g} differs from the executed "
            f"experiment run_seconds ({float(exp.run_seconds):g} s); "
            "apply the run length to the experiment before validating it")
    cadence_receipt = _validate_hash_bound_history_cadence(
        exp, history_interval_seconds)
    if (dc.grid_id != 1 or dc.parent_id != 0 or cfg.grid_id != 1
            or cfg.specified is not True or cfg.nested is not False):
        raise ValueError(
            "prepared forecast requires one specified, non-nested d01")
    if exp.feedback != 0:
        warn("feedback is inactive for a single domain; the declared value is preserved")
    matched = identify_single_domain_profile(cfg)
    effective = profile if profile is not None else matched
    # The receipt's ``source`` is the run's actual provenance.  (Until
    # this fix the 20CRv3-vouched profile rewrote it to the vouching
    # source on cross-source runs -- a wrong label on exactly the
    # cross-source admission the ruling opened.)
    validation_source = source or "gfs"
    verification = single_domain_verification_status(cfg)
    # The registry's source-neutral tuple governance, the same call the
    # front doors and the domain-tree route make: an unacknowledged
    # expert tuple WARNS here naming the acknowledgement and runs; a
    # tuple the registry has no spelling for records its blocker and
    # runs.  The acknowledgement channels are the runner's own --ack
    # flag and the hash-bound experiment's ``acknowledgements`` array,
    # and either one silences the line.
    acknowledgements, ack_provenance = acknowledgement_delivery(
        flag=tuple(expert_acknowledgements),
        toml=tuple(getattr(exp, "acknowledgements", ()) or ()),
    )
    if effective is not None:
        receipt = _validate_profile_switches(
            exp, source=validation_source, profile=effective)
        receipt["profile_binding"] = (
            "named" if profile is not None else "matched")
        # Switch drift refuses first (the named gate's own words); only
        # a config that IS the suite reaches the consent question.
        governance_selection = single_domain_physics_selection(
            cfg, expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=ack_provenance)
        domain_governance = governance_selection["domains"]["1"]
    else:
        observed = {
            name: getattr(cfg, name)
            for name in _SUITE_RECEIPT_SWITCH_KEYS
            if hasattr(cfg, name)
        }
        observed_radiation = radiation_scheme_ids(cfg)
        # Recording is not permission withheld: the registry may simply
        # have no spelling for an engine-valid tuple (the shipped
        # hierarchy descriptor's aggregate radiation-off is the
        # exhibit), and the domain-tree route has always recorded that
        # blocker and run.  Same mechanism here -- and an expert tuple
        # the registry DOES spell warns above without its
        # acknowledgement, exactly as it does on the domain-tree route.
        governance_selection = single_domain_physics_selection(
            cfg, expert_acknowledgements=acknowledgements,
            acknowledgement_provenance=ack_provenance)
        domain_governance = governance_selection["domains"]["1"]
        receipt = {
            "schema": "gpuwm-prepared-physics-suite-v1",
            "source": validation_source,
            "profile": None,
            "profile_binding": EXPERIMENT_CONFIG_PROFILE,
            "readiness": "IMPLEMENTED_SUITE_NOT_WRF_VERIFIED",
            "warning": verification["sentence"],
            "warning_only": True,
            "registry_components": domain_governance["components"],
            "registry_blocker": domain_governance["registry_blocker"],
            "resolved": {
                **observed,
                "radiation_scheme_ids": list(observed_radiation),
            },
            "validated_domains": [{
                "grid_id": int(dc.grid_id),
                **observed,
                "radiation_scheme_ids": list(observed_radiation),
            }],
        }
    receipt["registry_governance"] = {
        **domain_governance["governance"],
        "registry_sha256": governance_selection["registry_sha256"],
        "acknowledgements": list(governance_selection["acknowledgements"]),
        "acknowledgement_provenance": dict(
            governance_selection["acknowledgement_provenance"]),
    }
    receipt["verification"] = verification
    receipt["landuse_identity"] = dict(_LANDUSE_IDENTITY)
    # Scheme contracts key on the RESOLVED SELECTORS, never on the
    # profile name: an mp8 suite admitted without the Thompson profile
    # id must still bind the table authority and the preflight/run
    # table-root stability check (the domain-tree runner has always
    # keyed this on mp_physics == 8).
    mp_selector = int(cfg.mp_physics)
    if mp_selector == THOMPSON_MP_PHYSICS:
        receipt["thompson_contract"] = _validated_thompson_runtime_contract()
    elif mp_selector == 10:
        rimed_ice = int(cfg.morr_rimed_ice)
        receipt["morrison_contract"] = {
            "selector": 10,
            "morr_rimed_ice": rimed_ice,
            "rimed_ice_category": "hail" if rimed_ice == 1 else "graupel",
        }
    elif mp_selector == NSSL2_MP_PHYSICS:
        # The mode is RESOLVED FROM THIS CONFIG, never assumed to be the
        # shipped default lane: a hail-off or diagnosed-CCN run carries a
        # different mode and a different transported-field list, and a
        # receipt that describes some other configuration is worse than no
        # receipt at all.
        receipt["nssl2_contract"] = nssl2_contract_receipt(
            resolve_nssl2_mode_for_config(cfg))
    compatibility = str(getattr(cfg, "wrf_rrtmg_compatibility", "none"))
    if compatibility in WRF_RRTMG_SUBSTITUTION_TOKENS:
        scheme_ids = list(radiation_scheme_ids(cfg))
        receipt["radiation_substitution"] = {
            "contract": compatibility,
            "requested_wrf_scheme_ids": scheme_ids,
            "resolved_gpuwm_scheme_ids": scheme_ids,
            "resolved_gpuwm_solver": "RTE+RRTMGP",
        }
    elif compatibility == WRF_RRTMG_LEGACY:
        scheme_ids = list(radiation_scheme_ids(cfg))
        receipt["radiation_identity"] = {
            "contract": WRF_RRTMG_LEGACY,
            "requested_wrf_scheme_ids": scheme_ids,
            "resolved_gpuwm_scheme_ids": scheme_ids,
            "resolved_gpuwm_solver": "legacy RRTMG",
        }
    receipt["output_cadence"] = cadence_receipt
    return receipt


def _validate_front_door_physics_proof(
        proof: Mapping[str, object], *, source: str, profile: str | None, cfg,
) -> dict[str, object] | None:
    """Verify v3's explicit physics receipt without rewriting v2 history.

    THE profile-agnostic invariant of this route: the physics executed
    is the physics prepared.  The preparation proof records the front
    door's selection receipt, and this recomputes the same receipt from
    the same hash-bound experiment config and requires them to be the
    same selection: every field equal except the record-only ones
    (:data:`woof.physics_compat.SELECTION_RECEIPT_RECORD_ONLY_FIELDS`:
    the registry document digest and the maturity label), and the
    registry physics equal part by part, so a preparation made under a
    registry that differs only in citations, warnings, labels or
    admission rules resolves and one whose options, knobs, tables or
    kernel identities differ is refused naming the part (A153).  Physics
    added since the preparation (a component, option, knob or template
    entry) resolves where this config leaves it at its off value and is
    refused, named, where it selects it.  An
    admission rule that no longer admits the configuration refuses in
    the recomputation itself, naming the rule.  The proof's own receipt
    says which of the two selection spellings the preparation used -- a
    named-profile receipt or the per-domain config-derived one -- and the
    recomputation follows it; both are pure functions of (config,
    registry, acknowledgements), so equality still proves the prepared
    selection came from THIS config under THIS registry's physics.  The
    runner's own ``--physics-profile``, when named, is separately
    enforced against the config by ``_validate_physics``.
    """

    # HRRR joins this gate at its FIRST schema rather than a later one:
    # its bundle is new here, so there is no history of proofs written
    # before the receipt existed, and a new source has no reason to be
    # admitted on weaker evidence than the one beside it.
    export = proof.get("export")
    mapped_config_suite = (
        (source in _MAPPED_SOURCES or source == "era5")
        and isinstance(export, dict)
        and export.get("status") == "READY"
        and export.get("schema") == "gpuwm-native-direct-wrf-export-v3")
    if not mapped_config_suite and (
            source not in {"gfs", "hrrr"}
            or proof.get("schema") in _LEGACY_PROOF_SCHEMAS[source]):
        return None
    if proof.get("schema") != _PROOF_SCHEMA[source]:
        return None
    label = source.upper()
    # Mapped preparation binds the selection inside its export receipt.
    # Older v2 exports stay readable under their original stock contract.
    selected = export.get("physics") if mapped_config_suite else proof.get("physics")
    if not isinstance(selected, dict):
        raise ValueError(
            f"{label} v3 preparation proof physics receipt is missing")
    acknowledgements = selected.get("acknowledgements")
    acknowledgement_provenance = selected.get(
        "acknowledgement_provenance")
    if (not isinstance(acknowledgements, list)
            or any(not isinstance(value, str) for value in acknowledgements)):
        raise ValueError(
            f"{label} v3 preparation proof acknowledgements are malformed")
    if not isinstance(acknowledgement_provenance, dict):
        raise ValueError(
            f"{label} v3 preparation proof acknowledgement provenance is "
            "malformed")
    if selected.get("schema") == MULTI_DOMAIN_SELECTION_SCHEMA:
        expected = single_domain_physics_selection(
            cfg,
            expert_acknowledgements=tuple(acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    else:
        proof_profile = selected.get("profile")
        if not isinstance(proof_profile, str):
            raise ValueError(
                f"{label} v3 preparation proof physics receipt names no "
                "profile and is not a per-domain selection receipt")
        expected = validate_single_domain_physics_profile(
            proof_profile, config=cfg,
            expert_acknowledgements=tuple(acknowledgements),
            acknowledgement_provenance=acknowledgement_provenance)
    # Compared by identity, not bytes (A153): the receipt's registry
    # document digest moved with every citation edit and refused
    # byte-identical forecasts.  What differs is named.  The config is
    # passed so physics added since the preparation resolves where this
    # config leaves it off, and is named where it does not.
    differences = physics_selection_differences(
        selected, expected, settings=cfg)
    if differences:
        shown = "; ".join(differences[:8]) + (
            f"; and {len(differences) - 8} more" if len(differences) > 8
            else "")
        raise ValueError(
            f"{label} v3 preparation proof physics selection differs from "
            f"the hash-bound experiment/profile: {shown}. Prepare again with this build "
            "to bind the current scheme implementations; the old prepared state "
            "cannot be integrated under different physics.")
    return expected


def _validate_proof_content_sha256(proof: Mapping[str, object]) -> str:
    content = dict(proof)
    declared = content.pop("proof_content_sha256", None)
    expected = hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()
    if declared != expected:
        raise ValueError("mapped preparation proof content hash is stale")
    return expected


def _execution_plan_receipt(
        *, source_exp, executed_exp, profile: str,
        history_interval_seconds: float,
) -> dict[str, object]:
    def domain_receipt(domain) -> dict[str, object]:
        cfg = domain.run
        return {
            "grid_id": int(domain.grid_id),
            "mp_physics": int(cfg.mp_physics),
            "cu_physics": int(cfg.cu_physics),
            "bl_pbl_physics": int(cfg.bl_pbl_physics),
            "sf_sfclay_physics": int(cfg.sf_sfclay_physics),
            "sf_surface_physics": int(cfg.sf_surface_physics),
            "radiation_scheme_ids": list(radiation_scheme_ids(cfg)),
            "history_interval_seconds": float(domain.history_interval_s),
        }

    source_domains = [domain_receipt(domain) for domain in source_exp.domains]
    executed = domain_receipt(executed_exp.root)
    executed["history_interval_seconds"] = float(history_interval_seconds)
    overrides: list[dict[str, object]] = []
    if len(source_domains) != 1:
        overrides.append({
            "kind": "domain-selection",
            "source_grid_ids": [row["grid_id"] for row in source_domains],
            "executed_grid_ids": [1],
            "reason": "prepared forecast runner executes hash-bound d01 only",
        })
    source_history = float(source_exp.root.history_interval_s)
    # A quick-look domain's declared (full) cadence names the same stream
    # as its alarm; neither is an output-cadence override (U-E3).
    if float(history_interval_seconds) not in {
            value for value in _hash_bound_history_cadences(source_exp.root)
            if value is not None}:
        overrides.append({
            "kind": "output-cadence",
            "source_experiment_seconds": source_history,
            "executed_seconds": float(history_interval_seconds),
            "mechanism": "explicit-hash-bound-history-output-schedule-v1",
            "model_state_or_physics_changed": False,
        })
    if float(source_exp.run_seconds) != float(executed_exp.run_seconds):
        overrides.append(_run_length_override(
            float(source_exp.run_seconds), float(executed_exp.run_seconds)))
    return {
        "schema": "gpuwm-prepared-d01-execution-plan-v1",
        "profile": profile,
        "source_experiment": {
            "domain_count": len(source_domains),
            "run_seconds": float(source_exp.run_seconds),
            "domains": source_domains,
        },
        "executed_d01": {
            "run_seconds": float(executed_exp.run_seconds),
            "domain": executed,
        },
        "physics_overrides": [],
        "execution_overrides": overrides,
    }


#: The execution plan's spelling of an executed length that differs from
#: the hash-bound experiment's: the prepared state is untouched, and the
#: model integrates exactly the first ``executed_seconds`` of the run the
#: preparation was built for.
RUN_LENGTH_OVERRIDE_MECHANISM = "requested-run-length-within-prepared-forcing-v1"


def _run_length_override(source_seconds: float, executed_seconds: float) -> dict[str, object]:
    return {
        "kind": "run-length",
        "source_experiment_seconds": float(source_seconds),
        "executed_seconds": float(executed_seconds),
        "mechanism": RUN_LENGTH_OVERRIDE_MECHANISM,
        "prepared_state_changed": False,
        "model_state_or_physics_changed": False,
    }


def _apply_run_length(exp, run_seconds: float):
    """The experiment this run integrates: the hash-bound one, ``run_seconds`` long.

    ``--run-seconds`` used to be read, compared with the hash-bound
    experiment's ``run_seconds``, and then DROPPED with a warning that
    "the experiment value is authoritative": a door asked for 4200 s of
    a 12 h preparation integrated the whole 12 h (13 minutes of two
    rented cards for a byte comparison that two minutes settle, a rented box,
    2026-10-06).  Nothing the preparation binds depends on the length:
    a prepared cache is an initial state plus the boundary tables for
    the times it decoded, which is why ``run.run_seconds`` sits in
    :data:`woof.ingest.prepared_cache.NON_TRAJECTORY_IDENTITY_FIELDS`
    and never moves the cache identity.  The one thing a length CAN
    break is running past the prepared boundary forcing, and that is
    the coverage gate below (``forcing_hours[-1] * 3600 < run_seconds``),
    which names the last prepared lead.

    So the requested length is the executed length, on every route:
    both timing authorities (``ExperimentConfig.run_seconds`` and the
    derived ``RunConfig.run_seconds`` copy, asserted equal at load) are
    replaced together, every downstream reader -- the tick clock, the
    history schedule, the boundary interval plan, the seam waits, the
    report -- sees one value, and the execution plan records the
    override beside the output-cadence one.  Equal lengths return the
    experiment unchanged.
    """

    requested = float(run_seconds)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("run-seconds must be a positive duration")
    if float(exp.run_seconds) == requested \
            and float(exp.root.run.run_seconds) == requested:
        return exp, None
    from woof.config import validate_run_config
    executed_run = validate_run_config(
        replace(exp.root.run, run_seconds=requested))
    domain = replace(exp.root, run=executed_run)
    executed = replace(exp, run_seconds=requested, domains=(domain,))
    return executed, _run_length_override(float(exp.run_seconds), requested)


def _apply_runtime_run_overrides(exp, overrides):
    """Apply only run fields the prepared-state contract declares inert."""
    if overrides is None:
        return exp, []
    if not isinstance(overrides, Mapping):
        raise TypeError("runtime_run_overrides must be a mapping of RunConfig fields")
    from woof.config import RunConfig, validate_run_config
    from woof.ingest.prepared_cache import (
        PREPARATION_INERT_RUN_FIELDS, effective_prepared_domain_config,
        prepared_domain_config_identity)

    allowed = {path.removeprefix("run.")
               for path in PREPARATION_INERT_RUN_FIELDS if path.startswith("run.")}
    for name in overrides:
        if not isinstance(name, str) or name not in allowed:
            raise ValueError(
                f"runtime run override {name!r} changes a preparation input or "
                "is unknown; only PREPARATION_INERT_RUN_FIELDS may reuse "
                "the verified prepared state")
        value = overrides[name]
        declared = RunConfig.__dataclass_fields__[name].type
        valid_type = {
            "bool": type(value) is bool,
            "int": type(value) is int,
            "str": type(value) is str,
            "float": type(value) in (int, float) and math.isfinite(value),
            "float | None": value is None or (
                type(value) in (int, float) and math.isfinite(value)),
        }.get(declared, False)
        if not valid_type:
            raise ValueError(
                f"runtime run override {name} must have declared type "
                f"{declared}, got {value!r}")
    original = exp.root.run
    executed = validate_run_config(replace(original, **copy.deepcopy(dict(overrides))))
    domain = replace(exp.root, run=executed)
    if effective_prepared_domain_config(prepared_domain_config_identity(exp.root)) != \
            effective_prepared_domain_config(prepared_domain_config_identity(domain)):
        raise ValueError("runtime run overrides changed the prepared-domain identity")
    changed = [{
        "field": f"domain_config.run.{name}",
        "prepared": getattr(original, name),
        "executed": getattr(executed, name),
        "prepared_state_changed": False,
        "model_state_or_physics_changed": True,
        "reason": "preparation does not read this field; the forecast uses the executed value",
    } for name in sorted(overrides)
        if getattr(original, name) != getattr(executed, name)]
    return replace(exp, domains=(domain,)), changed


def _validate_execution_file_receipt(
        receipt, actual: Path | None, label: str,
) -> None:
    """Bind a run-control file to the bytes the preparation validated.

    IDENTITY IS THE DIGEST.  The recorded ``path`` is provenance -- where
    the preparation found the file on the machine that prepared it -- and
    it is spelled in that machine's dialect.  This used to compare
    ``Path(path_value).name`` against the supplied file's name, which
    reads a recorded string with the RUNNING platform's separator rules:
    a Windows-prepared tree records ``C:\\...\\experiment.toml``, POSIX
    finds no separator in it, ``.name`` is the whole string, and the
    comparison failed on a tree whose bytes and SHA-256 matched exactly.
    A prepared tree therefore could not cross platforms -- prepare on the
    desktop, run on the node -- and the refusal said the receipt
    "differs from supplied file", blaming content that was identical.

    A name difference over identical bytes was never a content
    difference, and it prevents no breakage, so it is not a refusal.  The
    two facts that ARE the binding are checked here, separately, and the
    refusal names whichever one moved.
    """

    if not isinstance(receipt, dict) or set(receipt) != {
            "path", "bytes", "sha256"}:
        raise ValueError(f"mapped {label} execution receipt is malformed")
    path_value = receipt.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(
            f"mapped {label} execution receipt records no source path, so "
            "the file this preparation read cannot be named in any "
            "refusal about it. Re-prepare with a distribution that writes "
            "the provenance path.")
    byte_count = receipt.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise ValueError(f"mapped {label} execution byte count is malformed")
    _require_digest(receipt.get("sha256"), f"mapped {label} execution sha256")
    if actual is None:
        # A prepared hierarchy restores its sealed cache without the raw
        # preparation files. Its caller validates typed domain configuration
        # and compares this original digest to every cache's namelist identity.
        return
    differences: list[str] = []
    actual_bytes = actual.stat().st_size
    if receipt.get("bytes") != actual_bytes:
        differences.append(
            f"the receipt records {receipt.get('bytes')!r} bytes and the "
            f"supplied file is {actual_bytes}")
    actual_sha256 = _sha256(actual)
    if receipt.get("sha256") != actual_sha256:
        differences.append(
            f"the receipt records sha256 {receipt.get('sha256')!r} and the "
            f"supplied file hashes to {actual_sha256}")
    if differences:
        raise ValueError(
            f"mapped {label} execution receipt does not describe {actual}: "
            + "; ".join(differences)
            + ". A prepared tree is bound to the exact run-control bytes "
            "it was prepared from, so running it against different bytes "
            "would integrate a configuration the preparation never "
            f"validated. The preparation read this file as {path_value!r}; "
            "supply the bytes that receipt describes, or re-prepare "
            "against the file you have.")


def _optional_stock_wrf_export(proof, export) -> bool:
    """Recognize the producer's explicit optional/off companion-file receipt."""
    if not isinstance(export, dict) or export.get("status") == "READY":
        return False
    mode = proof.get("stock_wrf_export")
    status = export.get("status")
    valid = ((mode == "off" and status == "NOT_REQUESTED")
             or (mode == "optional" and status == "REFUSED"))
    keys = {"schema", "status", "reason"}
    if status == "REFUSED":
        keys.add("unsupported")
        valid = valid and isinstance(export.get("unsupported"), dict)
    if (not valid or set(export) != keys
            or export.get("schema") not in {
                "gpuwm-native-direct-wrf-export-v2", "gpuwm-native-direct-wrf-export-v3",
                "gpuwm-native-direct-wrf-hierarchy-export-v1"}
            or not isinstance(export.get("reason"), str) or not export["reason"]):
        raise ValueError("stock-WRF export receipt differs from the requested mode")
    return True


def _validate_completed_fields(record: object, frame_count: object,
                               source: str) -> None:
    """The receipt's count of values a frame derived for its source.

    One entry per canonical field: the method, the frames that carried a
    derived value out of ``frame_count``, and the derived values out of the
    field's total.  An empty or zero entry is never written.
    """

    entry_keys = {"method", "frames", "frame_count", "values_derived",
                  "values_total"}
    if not isinstance(record, dict) or not record:
        raise ValueError(
            f"mapped {source} composition receipt completed_fields is malformed")
    for name, entry in record.items():
        counts = [entry.get(key) for key in sorted(entry_keys - {"method"})] \
            if isinstance(entry, dict) else []
        if (not isinstance(name, str) or not isinstance(entry, dict)
                or set(entry) != entry_keys
                or not isinstance(entry.get("method"), str)
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           and value >= 0 for value in counts)
                or entry["frame_count"] != frame_count
                or not 0 < entry["frames"] <= entry["frame_count"]
                or not 0 < entry["values_derived"] <= entry["values_total"]):
            raise ValueError(
                f"mapped {source} composition receipt completed_fields entry "
                f"{name!r} is malformed")


#: The composition receipt's keys on every mapped preparation.
MAPPED_COMPOSITION_RECEIPT_KEYS = frozenset({
    "schema", "status", "mapping", "composition", "input_manifest",
    "decoders", "terrain_products", "terrain_provenance", "alignment",
    "soil_layers", "frame_count", "valid_times", "frames",
    "receipt_content_sha256",
})


def mapped_composition_receipt_keys(
    receipt: object, *, declared_bindings: bool, source: str,
) -> set[str]:
    """The exact key set a mapped preparation's composition receipt has.

    ``contributing_sources`` joins it when the composition declares
    cross-source bindings.  ``completed_fields`` and ``record_aliases``
    join it only when the receipt carries them, because preparation
    writes them only when a frame derived values the source did not carry
    and when the primary's files spelled records the way an earlier
    publication of the product did (``mapping.record_aliases``); each is
    then held to its shape, or the receipt is refused as malformed.
    """

    keys = set(MAPPED_COMPOSITION_RECEIPT_KEYS)
    if declared_bindings:
        keys.add("contributing_sources")
    if isinstance(receipt, dict) and "completed_fields" in receipt:
        keys.add("completed_fields")
        _validate_completed_fields(
            receipt["completed_fields"], receipt.get("frame_count"), source)
    if isinstance(receipt, dict) and "record_aliases" in receipt:
        keys.add("record_aliases")
        aliases = receipt["record_aliases"]
        if (not isinstance(aliases, dict) or not aliases
                or any(not isinstance(name, str)
                       or isinstance(count, bool)
                       or not isinstance(count, int) or count < 1
                       for name, count in aliases.items())):
            raise ValueError(
                f"mapped {source} composition receipt is malformed")
    return keys


def _mapped_boundary_spacings(
        proof: Mapping[str, object], manifest: Mapping[str, object], *,
        member_manifest: bool,
) -> tuple[int, ...]:
    """Every boundary spacing, in seconds, a mapped preparation was made at.

    Read off the decoded frames' valid times (the series the mapping's
    target contract held at decode) and the proof's declared interval, so a
    preparation whose bound mapping resolves by its decode identity is
    still held to the spacings the packaged target takes (A166).
    """

    receipt = proof.get("source_composition")
    times = (manifest.get("valid_times") if member_manifest
             else receipt.get("valid_times") if isinstance(receipt, Mapping)
             else None)
    parsed = []
    for value in times if isinstance(times, list) else ():
        try:
            parsed.append(datetime.fromisoformat(
                str(value).replace("Z", "+00:00")))
        except ValueError:
            raise ValueError(
                f"mapped preparation valid time {value!r} cannot be read, so "
                "its boundary spacing cannot be held to the packaged "
                "mapping") from None
    spacings = {int((later - earlier).total_seconds())
                for earlier, later in zip(parsed, parsed[1:])}
    declared = proof.get("boundary_interval_seconds")
    if (isinstance(declared, int) and not isinstance(declared, bool)
            and declared > 0):
        spacings.add(declared)
    return tuple(sorted(spacings))


def _validate_packaged_mapped_evidence(
        *, prepared_root: Path, proof: Mapping[str, object],
        manifest: Mapping[str, object], manifest_sha256: str,
        experiment_config: Path | None, wps_namelist: Path | None,
        source: str = "20crv3", sealed: bool = True,
        posted: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, Path], Mapping[str, object], str | None]:
    """Validate the shared mapped evidence, including explicit model claims.

    Named profiles additionally pin their shipped authorities. ``mapped``
    binds caller-authored authorities to the manifest, composition receipt
    and sealed cache identity. Neither route requires a forecast certificate.
    The historical function name remains for existing callers.

    ``posted`` is an as-posted head's block (``basis.as_posted``): the head
    binds its input plan (``manifest`` is the plan's manifest and
    ``manifest_sha256`` its placeholder), and its seal writes the evidence
    that reads every lead -- the proof's ``seal_authored_proof_keys`` (the
    composition receipt and the vertical ladder) and the manifest copy.
    Those are checked by the sealed preflight the seal re-runs; here the
    head is held to the mapping, composition and provenance evidence, the
    execution inputs and the decoders the plan names, and the receipt
    digest its identity binds is the plan's placeholder.
    """

    from woof.source_authorities import (bound_mapping_refusal,
                                          packaged_authority_sha256)

    profile_id = _MAPPED_PACKAGED_PROFILE.get(source)
    if source not in _MAPPED_SOURCES:
        raise ValueError(f"{source} is not a mapped preparation")
    member_manifest = _SOURCE_SCHEMA[source] == _SOURCE_SCHEMA["20crv3"]

    direct_proof_keys = set(MAPPED_DIRECT_PROOF_KEYS)
    hierarchy_proof_keys = set(MAPPED_HIERARCHY_PROOF_KEYS)
    schema = proof.get("schema")
    expected_proof_keys = (
        direct_proof_keys if schema == _PROOF_SCHEMA[source]
        else hierarchy_proof_keys)
    # Older proofs predate the optional stock-WRF companion export control.
    # Their READY export still goes through the original checks below.
    if "stock_wrf_export" not in proof:
        expected_proof_keys.discard("stock_wrf_export")
    elif (not isinstance(proof["stock_wrf_export"], str)
          or proof["stock_wrf_export"] not in {"off", "optional", "required"}):
        raise ValueError("mapped stock-WRF export mode is invalid")
    # Bundles prepared before the initialization moisture-floor receipt
    # existed carry no such key, and their forecasts are still valid
    # preparations: requiring it would refuse every mapped bundle already
    # on disk.  A bundle that HAS it keeps the exactness -- the key is in
    # the required set above, so a document carrying a MALFORMED spelling
    # of it is still refused.
    expected_proof_keys -= MAPPED_MOISTURE_FLOOR_KEYS - set(proof)
    expected_proof_keys -= MAPPED_VERTICAL_COORDINATE_KEYS - set(proof)
    expected_proof_keys -= MAPPED_PREPARATION_TELEMETRY_KEYS - set(proof)
    expected_proof_keys -= {"boundary_stream"} - set(proof)
    if not sealed:
        # A prepared head's proof: the seal adds these, and the seal
        # re-runs this check with them present.
        from woof.ingest.boundary_stream import SEAL_ONLY_PROOF_KEYS

        expected_proof_keys -= SEAL_ONLY_PROOF_KEYS
    if posted is not None:
        # An as-posted head's proof leaves the keys that read every lead to
        # its seal, which re-runs this check with them present.
        expected_proof_keys -= {str(key) for key in
                                posted.get("seal_authored_proof_keys") or ()}
    # Every required key present, and nothing beyond them but the
    # declared-optional ones: a missing key and an unrecognised key are
    # both still refusals, which is the exactness this inventory exists
    # to hold.
    observed = set(proof)
    if (schema not in {
            _PROOF_SCHEMA[source], _HIERARCHY_PROOF_SCHEMA[source]}
            or not expected_proof_keys <= observed
            or observed - expected_proof_keys - MAPPED_OPTIONAL_PROOF_KEYS):
        raise ValueError(f"mapped {source} proof top-level inventory differs")
    if schema == _HIERARCHY_PROOF_SCHEMA[source] \
            and not isinstance(proof.get("target_contract"), dict):
        raise ValueError(f"mapped {source} hierarchy target contract is missing")
    if sealed:
        # A head's proof has no content digest yet; the seal re-runs this.
        _validate_proof_content_sha256(proof)
    evidence_root = _require_directory(
        prepared_root / "source-evidence", "mapped source evidence")
    if "initial_source" in proof:
        from woof.initial_source import validate_initial_evidence
        validate_initial_evidence(
            prepared_root, proof["initial_source"],
            valid_time=proof["forcing_times"][0])
    mapping_path = _require_file(
        evidence_root / "mapping.json", "mapped source mapping evidence")
    composition_path = _require_file(
        evidence_root / "composition.json",
        "mapped source composition evidence")
    # An as-posted head has no manifest copy: its seal writes it.
    copied_manifest = None if posted is not None else _require_file(
        evidence_root / "input-manifest.json",
        "mapped source manifest evidence")
    if copied_manifest is not None and copied_manifest != (
            prepared_root / "source-evidence" /
            "input-manifest.json").resolve():
        raise RuntimeError("mapped source manifest resolved unexpectedly")
    expected_authority_sha256 = (
        dict(packaged_authority_sha256(profile_id)) if profile_id else {
            "mapping": _require_digest(
                manifest.get("mapping_sha256"), "mapped manifest mapping sha256"),
            "composition": _require_digest(
                manifest.get("composition_sha256"),
                "mapped manifest composition sha256"),
        })
    if profile_id:
        # A release that adds an admission-only declaration to a packaged
        # mapping (A159's whole multiples) moves its pin while every frame
        # stays the same bytes; the preparation's own bound digest then
        # stands for the packaged mapping in every check below (A166).
        mapping_bytes = mapping_path.read_bytes()
        bound_mapping_sha256 = hashlib.sha256(mapping_bytes).hexdigest()
        if bound_mapping_sha256 != expected_authority_sha256["mapping"]:
            refusal = bound_mapping_refusal(
                profile_id, mapping_bytes,
                spacings_seconds=_mapped_boundary_spacings(
                    proof, manifest, member_manifest=member_manifest))
            if refusal is not None:
                # The one difference with a named breakage of its own: a
                # preparation from before the profile's mapping declared
                # the hydrometeors its publisher writes.
                raise ValueError(_stale_profile_mapping_refusal(
                    source=source, profile_id=profile_id,
                    mapping_bytes=mapping_bytes,
                    composition_sha256=_sha256(composition_path),
                    prepared_root=prepared_root) or (
                    f"mapped preparation does not use the packaged {source} "
                    f"authorities ({profile_id}): {refusal}"))
            expected_authority_sha256["mapping"] = bound_mapping_sha256
    if (_sha256(mapping_path) != expected_authority_sha256["mapping"]
            or _sha256(composition_path)
            != expected_authority_sha256["composition"]):
        if profile_id:
            raise ValueError(
                f"mapped preparation does not use the packaged {source} "
                f"authorities ({profile_id})")
        raise ValueError("mapped authorities differ from the input manifest")
    if copied_manifest is not None and _sha256(copied_manifest) != manifest_sha256:
        raise ValueError("mapped source manifest evidence differs from caller pin")
    if not member_manifest and any(
            manifest.get(f"{role}_sha256") != expected_authority_sha256[role]
            for role in ("mapping", "composition")):
        raise ValueError("mapped manifest authority identity differs")
    if not profile_id:
        # A packaged profile's stale preparation handed in as a caller's
        # own mapping (the stage door names a root whose mapping no
        # longer matches any pin ``mapped``) is still that profile's.
        stale = _stale_profile_mapping_refusal(
            source=source, profile_id=None,
            mapping_bytes=mapping_path.read_bytes(),
            composition_sha256=_sha256(composition_path),
            prepared_root=prepared_root)
        if stale is not None:
            raise ValueError(stale)

    composition_document = _load_json_object(
        composition_path, "mapped composition authority")
    declared_bindings = composition_document.get("field_sources") or {}
    if not isinstance(declared_bindings, dict):
        raise ValueError("mapped composition field sources must be an object")
    terrain_bindings = [binding for binding in declared_bindings.values()
                        if isinstance(binding, dict)
                        and "terrain_height" in binding.get("fields", ())]
    terrain_spec = composition_document.get("supplements", {}).get(
        "terrain_height")
    if terrain_spec is None and len(terrain_bindings) == 1:
        terrain_spec = terrain_bindings[0]
    if not isinstance(terrain_spec, dict):
        raise ValueError("mapped composition names no unique terrain provider")

    known_names = {"mapping.json", "composition.json", "input-manifest.json",
                   "composition-inputs.json"}
    provenance_candidates = [
        path.resolve() for path in evidence_root.iterdir()
        if path.is_file() and path.name not in known_names
    ]
    if member_manifest:
        if len(provenance_candidates) != 1:
            raise ValueError("mapped source evidence must contain one provenance file")
        provenance_path = provenance_candidates[0]
        if _sha256(provenance_path) != expected_authority_sha256["provenance"]:
            raise ValueError(f"mapped {source} provenance authority differs")
        provenance_paths = {"mapped_provenance": provenance_path}
    else:
        provenance_rows = manifest.get("provenance")
        if (not isinstance(provenance_rows, dict) or not provenance_rows
                or len(provenance_candidates) != len(provenance_rows)):
            raise ValueError("mapped provenance evidence inventory differs")
        provenance_paths = {}
        remaining = {path: _sha256(path) for path in provenance_candidates}
        for role, row in sorted(provenance_rows.items()):
            digest = _require_digest(row.get("sha256"), "mapped provenance sha256")
            candidates = [path for path, value in remaining.items() if value == digest]
            if not candidates:
                raise ValueError(f"mapped provenance authority differs: {role}")
            path = candidates[0]
            if path.stat().st_size != row.get("bytes"):
                raise ValueError(f"mapped provenance byte count differs: {role}")
            del remaining[path]
            provenance_paths[f"mapped_provenance:{role}"] = path
        terrain_role = terrain_spec.get("provenance_role")
        if terrain_role not in provenance_rows:
            raise ValueError("mapped terrain provenance role is missing")
        terrain_digest = provenance_rows[terrain_role]["sha256"]
        if profile_id and terrain_digest != expected_authority_sha256["provenance"]:
            if not packaged_profile(profile_id).get("input_normalizer"):
                raise ValueError(f"mapped {source} provenance authority differs")
            _check_normalized_provenance(
                provenance_paths[f"mapped_provenance:{terrain_role}"],
                profile_id, source)
        expected_authority_sha256["provenance"] = terrain_digest
        provenance_path = provenance_paths[f"mapped_provenance:{terrain_role}"]

    if posted is not None:
        return _posted_mapped_head_authority(
            proof=proof, manifest=manifest, placeholder=manifest_sha256,
            experiment_config=experiment_config, wps_namelist=wps_namelist,
            source=source, member_manifest=member_manifest,
            evidence_paths={"mapped_mapping": mapping_path,
                            "mapped_composition": composition_path,
                            **provenance_paths},
            authority_sha256=expected_authority_sha256)

    receipt = proof.get("source_composition")
    # Composition bytes are already bound above. Their declared donor roles,
    # rather than a profile allowlist or a receipt's claims, govern the shape.
    expected_receipt_keys = mapped_composition_receipt_keys(
        receipt, declared_bindings=bool(declared_bindings), source=source)
    if (not isinstance(receipt, dict) or set(receipt) != expected_receipt_keys
            or receipt.get("schema") != "gpuwm-mapped-composition-receipt-v1"
            or receipt.get("status")
            != "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED"):
        if declared_bindings and isinstance(receipt, dict) \
                and "contributing_sources" not in receipt:
            raise ValueError(
                "the composition declares contributing-source "
                "bindings but the receipt names no contributing sources; "
                "a cross-source decode always records them, so this "
                "receipt belongs to some other decode")
        raise ValueError(f"mapped {source} composition receipt is malformed")
    receipt_content = dict(receipt)
    declared_receipt_sha256 = receipt_content.pop(
        "receipt_content_sha256", None)
    expected_receipt_sha256 = hashlib.sha256(
        _canonical(receipt_content).encode("utf-8")).hexdigest()
    if declared_receipt_sha256 != expected_receipt_sha256:
        raise ValueError(f"mapped {source} composition receipt hash is stale")

    def evidence_record(record, expected_sha256: str, label: str) -> None:
        if (not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or record.get("sha256") != expected_sha256
                or not isinstance(record.get("path"), str)):
            raise ValueError(f"mapped {source} {label} receipt differs")

    evidence_record(
        receipt.get("mapping"), expected_authority_sha256["mapping"],
        "mapping")
    evidence_record(
        receipt.get("composition"), expected_authority_sha256["composition"],
        "composition")
    if member_manifest:
        # The member route seals TWO manifests: the caller pinned the
        # member manifest (its evidence copy was hashed above), and the
        # composition receipt seals the BRIDGED composition-inputs
        # document the decode actually verified.  The bridged bytes ride
        # the evidence directory precisely so this record can be
        # re-hashed here rather than taken on faith.
        bridged_manifest = _require_file(
            evidence_root / "composition-inputs.json",
            "mapped member bridged composition-inputs evidence")
        evidence_record(
            receipt.get("input_manifest"), _sha256(bridged_manifest),
            "manifest")
    else:
        evidence_record(
            receipt.get("input_manifest"), manifest_sha256, "manifest")
    provenance = receipt.get("terrain_provenance")
    if (not isinstance(provenance, dict)
            or set(provenance) != {"provenance_path", "provenance_sha256"}
            or provenance.get("provenance_sha256")
            != expected_authority_sha256["provenance"]
            or not isinstance(provenance.get("provenance_path"), str)):
        raise ValueError(f"mapped {source} terrain provenance receipt differs")

    execution = proof.get("execution_inputs")
    if not isinstance(execution, dict):
        raise ValueError(f"mapped {source} execution input receipt is missing")
    _validate_execution_file_receipt(
        execution.get("experiment_config"), experiment_config,
        "experiment config")
    _validate_execution_file_receipt(
        execution.get("wps_namelist"), wps_namelist, "WPS namelist")
    composition_decoders = receipt.get("decoders")
    execution_decoders = execution.get("decoders")
    # The decoder inventory is the manifest's own, not a constant: a GRIB2
    # profile's roles are whatever its sealed manifest declares (the
    # engine on the bare default, the subprocess pair on the workaround,
    # none for NetCDF), and BOTH sides are pinned -- the empty set is
    # checked against the manifest just as any other, so a bundle cannot
    # drop a decoder it actually used.  The MEMBER manifest declares no
    # decoder section, so its pin is the closed pair of shipped decode
    # routes instead: the recorded inventory must be one of the two, in
    # full.
    if member_manifest:
        observed_roles = (
            set(composition_decoders)
            if isinstance(composition_decoders, dict) else set())
        if frozenset(observed_roles) not in _TWENTYCRV3_DECODER_ROLE_SETS:
            raise ValueError(
                f"mapped {source} decoder inventory names neither the "
                "in-process engine nor the subprocess pair this "
                "distribution ships")
        expected_decoder_roles = observed_roles
    else:
        expected_decoder_roles = set(manifest.get("decoders") or {})
    if (not isinstance(composition_decoders, dict)
            or not isinstance(execution_decoders, dict)
            or set(composition_decoders) != expected_decoder_roles
            or set(execution_decoders) != expected_decoder_roles):
        raise ValueError(f"mapped {source} decoder inventory differs")
    decoder_sha256: dict[str, str] = {}
    for role in sorted(expected_decoder_roles):
        composed = composition_decoders[role]
        executed = execution_decoders[role]
        if (not isinstance(composed, dict)
                or set(composed) != {"path", "sha256"}
                or not isinstance(executed, dict)
                or set(executed) != {"path", "bytes", "sha256"}
                or composed.get("path") != executed.get("path")
                or composed.get("sha256") != executed.get("sha256")
                or isinstance(executed.get("bytes"), bool)
                or not isinstance(executed.get("bytes"), int)
                or executed.get("bytes") <= 0):
            raise ValueError(f"mapped {source} decoder receipt differs: {role}")
        if not member_manifest:
            declared = manifest["decoders"][role]
            if (declared.get("sha256") != executed.get("sha256")
                    or declared.get("bytes") != executed.get("bytes")
                    or receipt_basename(declared.get("path"))
                    != receipt_basename(executed.get("path"))):
                raise ValueError(f"mapped decoder manifest differs: {role}")
        decoder_sha256[role] = _require_digest(
            composed.get("sha256"), f"mapped {source} decoder {role} sha256")

    composition_document = _load_json_object(
        composition_path, f"mapped {source} composition authority")
    if member_manifest:
        valid_times = manifest.get("valid_times")
        rows = manifest.get("files")
        surface_rows = [row for row in rows if row["role"] == "sfc"]
        expected_terrain = [{
            "path": row["path"],
            "sha256": row["sha256"],
        } for row in surface_rows]
        if receipt.get("terrain_products") != expected_terrain:
            raise ValueError(f"mapped {source} terrain product receipt differs")
        alignment = receipt.get("alignment")
        # The composed exact-subset receipt, exactly as the generic
        # single-source branch validates it, PLUS the ensemble identity
        # the sealed member manifest bound -- the member route's whole
        # reason to exist.  The key set is pinned closed: the packaged
        # composition declares `valid_time_exact`, so a receipt carrying
        # broadcast keys (or any other extra) is a receipt for some other
        # decode.
        expected_alignment_keys = {
            "schema", "status", "field", "primary_shape",
            "supplement_shape", "latitude_index_range",
            "longitude_index_range", "latitude_sha256", "longitude_sha256",
            "terrain_full_sha256", "terrain_subset_sha256",
            "supplement_valid_times", "matched_primary_valid_times",
            "invariant_across_all_supplement_times",
            "latitude_index_direction", "longitude_index_direction",
            "coordinate_match", "longitude_equivalence",
            "member", "member_identity",
        }
        expected_alignment = {
            "schema": "gpuwm-mapped-exact-subset-binding-v1",
            "status": "PASS",
            "field": "terrain_height",
            "coordinate_match": "exact_equivalent_contiguous_subset",
            "invariant_across_all_supplement_times": True,
            "member": manifest["member"],
            "member_identity": _TWENTYCRV3_MEMBER_IDENTITY,
        }
        if (not isinstance(alignment, dict)
                or set(alignment) != expected_alignment_keys
                or any(alignment.get(key) != value
                       for key, value in expected_alignment.items())
                or alignment.get("matched_primary_valid_times")
                != valid_times):
            raise ValueError(f"mapped {source} member/alignment receipt differs")
        _require_digest(
            alignment.get("terrain_subset_sha256"),
            "mapped composition terrain subset sha256")
        source_member: str | None = str(manifest["member"])
    else:
        valid_times = receipt.get("valid_times")
        if not isinstance(valid_times, list) or not valid_times:
            raise ValueError(
                "mapped composition receipt names no canonical valid times")
        # The terrain products a composed preparation used are the
        # supplement files the manifest bound to the profile's declared
        # data role -- not a guess, and not "whatever the receipt says":
        # the manifest is pinned by the caller's --source-manifest-sha256,
        # so this is the caller's own binding checked against the receipt.
        supplements = manifest.get("supplements")
        if not isinstance(supplements, dict):
            raise ValueError("mapped composition supplement inventory is missing")
        field_sources = declared_bindings
        terrain_binding = terrain_binding_name = None
        for name, binding in field_sources.items():
            if "terrain_height" in binding.get("fields", ()):
                terrain_binding, terrain_binding_name = binding, name
        declared_role = terrain_spec.get("data_role")
        expected_roles = {str(binding["data_role"])
                          for binding in declared_bindings.values()}
        expected_roles.add(str(declared_role))
        if set(supplements) != expected_roles:
            raise ValueError("mapped supplement roles differ from the composition")
        supplement_rows = supplements.get(declared_role)
        if not isinstance(supplement_rows, list) or not supplement_rows:
            raise ValueError("mapped composition supplement inventory is empty")
        # NAME plus digest, not the absolute path: the portable manifest
        # stores basenames on purpose so a prepared tree survives its raw
        # inputs being moved, while the composition receipt records the
        # path the decode actually opened.  The digest is the binding; the
        # name is what makes the pairing readable in a refusal.
        expected_terrain = [
            (receipt_basename(row.get("path")), row.get("sha256"))
            for row in supplement_rows
        ]
        recorded = receipt.get("terrain_products")
        if not isinstance(recorded, list) or [
            (receipt_basename(row.get("path")), row.get("sha256"))
            for row in recorded
            if isinstance(row, dict) and set(row) == {"path", "sha256"}
        ] != expected_terrain:
            raise ValueError(f"mapped {source} terrain product receipt differs")
        alignment = receipt.get("alignment")
        # The binding receipt is composed, not predicted: its index ranges
        # and array digests come from the two grids and this runner has no
        # independent source for them.  What IS checked is every claim the
        # receipt makes ABOUT itself -- that it passed, that the
        # coordinate match was exact, and (single-source) that the terrain
        # was invariant across the supplied times -- plus the fact that
        # its bytes are inside the proof, whose content hash the caller
        # pinned.
        if terrain_binding is not None:
            # Cross-source: the top-level alignment is the terrain
            # provider's binding receipt, under the binding's own declared
            # clock; every borrowed field must carry a subset digest, and
            # matched plus carried times must be exactly the canonical
            # valid times so no lead's provenance goes unnamed.
            expected_alignment = {
                "schema": "gpuwm-cross-source-binding-v1",
                "status": "PASS",
                "binding": terrain_binding_name,
                "source_id": terrain_binding.get("source_id"),
                "grid_alignment": terrain_binding.get("grid_alignment"),
                "time_alignment": terrain_binding.get("time_alignment"),
                "coordinate_match": (
                    "bilinear_mask_aware_nearest"
                    if terrain_binding.get("grid_alignment") == "bilinear_mask_aware_nearest"
                    else "exact_equivalent_regular_strided_subset"
                    if terrain_binding.get("grid_alignment") == "exact_coordinate_strided_subset"
                    else "exact_equivalent_contiguous_subset"
                ),
            }
            matched = alignment.get("matched_primary_valid_times") \
                if isinstance(alignment, dict) else None
            carried = alignment.get("broadcast_primary_valid_times", []) \
                if isinstance(alignment, dict) else None
            if (not isinstance(alignment, dict)
                    or any(alignment.get(key) != value
                           for key, value in expected_alignment.items())
                    or not isinstance(matched, list)
                    or not isinstance(carried, list)
                    or sorted(matched + carried) != sorted(valid_times)
                    or sorted(alignment.get("fields") or ())
                    != sorted(terrain_binding.get("fields") or ())):
                raise ValueError(
                    "mapped cross-source alignment receipt differs")
            subset_digests = alignment.get("field_subset_sha256")
            if not isinstance(subset_digests, dict) or set(subset_digests) \
                    != set(terrain_binding.get("fields") or ()):
                raise ValueError(
                    "mapped cross-source subset digests do not cover the "
                    "bound fields")
            for name in sorted(subset_digests):
                _require_digest(
                    subset_digests[name],
                    f"mapped cross-source subset sha256 for {name}")
        else:
            expected_alignment = {
                "schema": "gpuwm-mapped-exact-subset-binding-v1",
                "status": "PASS",
                "field": "terrain_height",
                "coordinate_match": "exact_equivalent_contiguous_subset",
                "invariant_across_all_supplement_times": True,
            }
            if (not isinstance(alignment, dict)
                    or any(alignment.get(key) != value
                           for key, value in expected_alignment.items())
                    or alignment.get("matched_primary_valid_times")
                    != valid_times):
                raise ValueError(
                    "mapped composition alignment receipt differs")
            _require_digest(
                alignment.get("terrain_subset_sha256"),
                "mapped composition terrain subset sha256")
        if declared_bindings:
            from woof.source_authorities import packaged_contributing_sha256

            contributing_sha = (
                dict(packaged_contributing_sha256(profile_id))
                if profile_id else {
                    str(binding["mapping_role"]): _require_digest(
                        binding.get("mapping_sha256"), "contributing mapping sha256")
                    for binding in declared_bindings.values()})
            entries = receipt.get("contributing_sources")
            if (not isinstance(entries, list)
                    or len(entries) != len(declared_bindings)):
                raise ValueError(
                    "contributing source inventory differs from the "
                    "packaged composition's declared bindings")
            seen_bindings = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(
                        "contributing source receipt is malformed")
                binding_name = str(entry.get("binding"))
                binding = declared_bindings.get(binding_name)
                if binding is None or binding_name in seen_bindings:
                    raise ValueError(
                        f"contributing source receipt names binding "
                        f"{binding_name!r}, which the packaged "
                        "composition does not declare")
                seen_bindings.add(binding_name)
                mapping_role = str(binding["mapping_role"])
                mapping_record = entry.get("mapping")
                recorded_sha256 = (
                    mapping_record.get("sha256")
                    if isinstance(mapping_record, dict) else None
                )
                if (recorded_sha256 != binding["mapping_sha256"]
                        or recorded_sha256
                        != contributing_sha.get(mapping_role)):
                    raise ValueError(
                        f"contributing source {binding_name!r} mapping "
                        "hash differs from the packaged pin: the decode "
                        "did not borrow through the mapping this "
                        "distribution ships")
                if entry.get("source_id") != binding["source_id"]:
                    raise ValueError(
                        f"contributing source {binding_name!r} names a "
                        "different source than the packaged composition")
                entry_alignment = entry.get("alignment")
                # A binding whose files published none of its fields is
                # UNPUBLISHED with the field-keyed fallback recorded (HRRR
                # wrfsfc before 2020-12-02 carries no VEG); a receipt whose
                # fallback is not the table's raises here.
                from woof.runtime_surface_fetch import unpublished_binding_fallbacks
                if unpublished_binding_fallbacks(entry) is not None:
                    pass
                elif (not isinstance(entry_alignment, dict)
                        or entry_alignment.get("status") != "PASS"):
                    raise ValueError(
                        f"contributing source {binding_name!r} alignment "
                        "did not pass")
                data_rows = entry.get("data")
                declared_rows = supplements[str(binding["data_role"])]
                def portable_rows(rows):
                    if not isinstance(rows, list) or not rows:
                        raise ValueError("contributing source data inventory is empty")
                    return [(receipt_basename(row.get("path")), row.get("sha256"))
                            for row in rows if isinstance(row, dict)]
                if (not isinstance(data_rows, list)
                        or len(data_rows) != len(declared_rows)
                        or portable_rows(data_rows) != portable_rows(declared_rows)):
                    raise ValueError(
                        f"contributing source {binding_name!r} data does "
                        "not match the manifest's supplement inventory")
                provenance_record = entry.get("provenance")
                declared_provenance = manifest["provenance"].get(
                    str(binding["provenance_role"]))
                if (not isinstance(provenance_record, dict)
                        or not isinstance(declared_provenance, dict)
                        or provenance_record.get("sha256") != declared_provenance["sha256"]
                        or receipt_basename(provenance_record.get("path"))
                        != receipt_basename(declared_provenance["path"])):
                    raise ValueError("contributing source provenance differs")
        for key in ("member", "member_identity"):
            if alignment.get(key) != manifest.get(key):
                raise ValueError("mapped composition member/alignment receipt differs")
        source_member = manifest.get("member")
    if receipt.get("soil_layers") != composition_document.get("soil_layers"):
        raise ValueError(f"mapped {source} soil-layer receipt differs")
    frames = receipt.get("frames")
    if (receipt.get("frame_count") != len(valid_times)
            or receipt.get("valid_times") != valid_times
            or not isinstance(frames, list) or len(frames) != len(valid_times)):
        raise ValueError(f"mapped {source} canonical frame inventory differs")
    for index, frame in enumerate(frames):
        if (not isinstance(frame, dict)
                or set(frame) != {
                    "header_sha256", "terrain_sha256", "field_count"}
                or isinstance(frame.get("field_count"), bool)
                or not isinstance(frame.get("field_count"), int)
                or frame.get("field_count") <= 0):
            raise ValueError(
                f"mapped {source} canonical frame {index} is malformed")
        _require_digest(
            frame.get("header_sha256"),
            f"mapped {source} frame {index} header sha256")
        _require_digest(
            frame.get("terrain_sha256"),
            f"mapped {source} frame {index} terrain sha256")
    from woof.mapped_composition import composition_receipt_identity_sha256
    return MappingProxyType({
        "mapped_mapping": mapping_path,
        "mapped_composition": composition_path,
        **provenance_paths,
    }), MappingProxyType({
        "receipt_content_sha256": expected_receipt_sha256,
        # What a cache written since A151 binds instead: the receipt less
        # where the run's files sat.
        "receipt_identity_sha256": composition_receipt_identity_sha256(receipt),
        "mapping_sha256": expected_authority_sha256["mapping"],
        "composition_sha256": expected_authority_sha256["composition"],
        "decoder_sha256": decoder_sha256,
        "preprocessing": proof.get("preprocessing"),
        "target_contract": proof.get("target_contract"),
    }), source_member


def _posted_mapped_head_authority(
        *, proof: Mapping[str, object], manifest: Mapping[str, object],
        placeholder: str, experiment_config: Path | None,
        wps_namelist: Path | None, source: str, member_manifest: bool,
        evidence_paths: Mapping[str, Path],
        authority_sha256: Mapping[str, str],
) -> tuple[Mapping[str, Path], Mapping[str, object], str | None]:
    """The mapped authority an as-posted head binds, from what it carries.

    The execution inputs are held to the files this forecast runs with, and
    the decoders the preparation ran to the ones its input plan names (the
    plan's manifest seals the same decoder rows as the window's).  The
    composition receipt does not exist before the seal, so the digest the
    head's identity binds in its place is the plan's ``placeholder``; the
    sealed preflight holds the receipt and the sealed identity to each
    other, and the seal binding allows only that change
    (:func:`_require_sealed_identity`).
    """

    if member_manifest:
        # Breakage it prevents: the member route seals a bridged second
        # manifest the head cannot carry; no member route prepares as posted.
        raise ValueError(
            f"an as-posted {source} head is not bound here: the member "
            "route seals a bridged manifest only at its seal; bind the "
            "sealed preparation (--proof-sha256)")
    execution = proof.get("execution_inputs")
    if not isinstance(execution, dict):
        raise ValueError(f"mapped {source} execution input receipt is missing")
    _validate_execution_file_receipt(
        execution.get("experiment_config"), experiment_config,
        "experiment config")
    _validate_execution_file_receipt(
        execution.get("wps_namelist"), wps_namelist, "WPS namelist")
    executed_decoders = execution.get("decoders")
    declared_decoders = manifest.get("decoders") or {}
    if (not isinstance(executed_decoders, dict)
            or not isinstance(declared_decoders, dict)
            or set(executed_decoders) != set(declared_decoders)):
        raise ValueError(f"mapped {source} decoder inventory differs")
    decoder_sha256: dict[str, str] = {}
    for role in sorted(declared_decoders):
        executed = executed_decoders[role]
        declared = declared_decoders[role]
        if (not isinstance(executed, dict)
                or set(executed) != {"path", "bytes", "sha256"}
                or not isinstance(declared, dict)
                or declared.get("sha256") != executed.get("sha256")
                or declared.get("bytes") != executed.get("bytes")
                or receipt_basename(declared.get("path"))
                != receipt_basename(executed.get("path"))):
            raise ValueError(f"mapped decoder manifest differs: {role}")
        decoder_sha256[role] = _require_digest(
            executed.get("sha256"), f"mapped {source} decoder {role} sha256")
    return MappingProxyType(dict(evidence_paths)), MappingProxyType({
        "receipt_content_sha256": placeholder,
        "receipt_identity_sha256": placeholder,
        "mapping_sha256": authority_sha256["mapping"],
        "composition_sha256": authority_sha256["composition"],
        "decoder_sha256": decoder_sha256,
        "preprocessing": proof.get("preprocessing"),
        "target_contract": proof.get("target_contract"),
    }), manifest.get("member")


#: What the native HRRR preparation writes into every prepared cache's
#: ``source_identity``, beyond the install provenance block whose keys
#: differ between a git checkout, a wheel and a sealed distribution.
#: These five ARE checked, because each one changes what the arrays mean.
_HRRR_IDENTITY_REQUIRED = ("source_sha256", "source_cycle",
                           "model_start_time", "source_forecast_hours",
                           "model_forcing_hours")

#: The woof modules whose bytes decide how HRRR GRIB2 becomes model
#: state.  The preparation hashes exactly these into the cache identity;
#: this reader requires them to be present and to agree with the proof,
#: so a cache decoded by a different ingest cannot restore under a proof
#: that names this one.
_HRRR_DECODE_SOURCES = (
    "woof/hrrr_forecast.py",
    "woof/ingest/hrrr.py",
    "woof/ingest/hrrr_physics.py",
    "woof/ingest/hrrr_surface.py",
    "woof/ingest/real.py",
    "woof/ingest/soil.py",
    "woof/ingest/lateral_bc.py",
    "woof/ingest/prepared_cache.py",
    "woof/state_serialization_contract.py",
    "tools/hrrr_single_domain_benchmark.py",
)


def _posix_digest_keys(digests: object) -> dict[str, object] | None:
    """Read a serialized digest dict with machine-independent keys.

    A repository-relative path is one thing; the separator it was
    spelled with is which machine spelled it.  Returns ``None`` for
    anything that is not a dict, so callers keep their own refusal.

    A dict carrying BOTH spellings of one path is refused rather than
    merged: collapsing it would silently drop one of two digests for
    the same file, which is exactly the disagreement this check exists
    to catch.  No producer emits that; only a hand-edited file can.
    """

    if not isinstance(digests, dict):
        return None
    normalized = {str(key).replace("\\", "/"): value
                  for key, value in digests.items()}
    if len(normalized) != len(digests):
        raise ValueError(
            "HRRR prepared cache decode digests name one file under two "
            "path spellings")
    return normalized


def _validate_hrrr_source_identity(
        identity: Mapping[str, object], proof: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind an HRRR prepared cache to the proof published beside it.

    HRRR's source identity is a different SHAPE from the portable one --
    it carries per-file digests of the ingest that decoded the GRIB2 and
    the cycle/lead vocabulary that names which HRRR run it came from,
    where the portable sources carry an adapter name and a single bridge
    digest.  Both answer the same question; only one of them can be
    checked with the portable code, so this checks the other.

    The check is a comparison against the PROOF, which the caller pinned
    by digest before this ran.  A cache whose identity says something
    the proof does not is refused -- that is the whole point of
    re-deriving the front door in every worker rather than passing a
    verdict across a process boundary.

    Both digest dicts are read through :func:`_posix_digest_keys` first.
    The producers now emit ``as_posix()`` keys everywhere, but caches
    that 1.8.2 sealed on Windows carry ``woof\\hrrr_forecast.py`` --
    ``str()`` of a relative Path -- and every lookup below missed, so
    the Windows prepared route refused its own caches at the handoff.
    The digests bind the same file BYTES either way, and the separator
    is a property of the machine that wrote the JSON rather than of
    what ran, so normalizing on read keeps those sealed caches
    restorable instead of orphaning them.
    """

    missing = [key for key in _HRRR_IDENTITY_REQUIRED if key not in identity]
    if missing:
        raise ValueError(
            f"HRRR prepared cache source identity is incomplete: {missing}")
    digests = _posix_digest_keys(identity.get("source_sha256"))
    if digests is None:
        raise ValueError(
            "HRRR prepared cache source identity carries no decode digests")
    absent = [name for name in _HRRR_DECODE_SOURCES if name not in digests]
    if absent:
        raise ValueError(
            f"HRRR prepared cache decode identity omits {absent}")
    if any(not isinstance(value, str) or len(value) != 64
           for value in digests.values()):
        raise ValueError(
            "HRRR prepared cache decode digests are malformed")
    if digests != _posix_digest_keys(proof.get("source_sha256")):
        raise ValueError(
            "HRRR prepared cache decode identity differs from its proof: "
            "the cache was built by a different ingest than the proof names")
    for key in ("source_cycle", "model_start_time", "source_forecast_hours",
                "model_forcing_hours"):
        if identity.get(key) != proof.get(key):
            raise ValueError(
                f"HRRR prepared cache source identity {key} differs from "
                "the preparation proof")
    if "ingest" in identity or "ingest" in proof:
        ingest = identity.get("ingest")
        if (not isinstance(ingest, dict)
                or set(ingest) != {"soil_texture_downscale"}
                or not isinstance(ingest["soil_texture_downscale"], bool)
                or ingest != proof.get("ingest")):
            raise ValueError("prepared cache ingest settings differ from preparation proof")
    if "static_highres" in identity or "static_highres" in proof:
        if identity.get("static_highres") != proof.get("static_highres"):
            raise ValueError("prepared cache high-resolution settings differ from preparation proof")
        if identity.get("static_highres") is None:
            raise ValueError(
                "prepared cache names a high-resolution identity with no "
                "settings in it, so nothing records which statics it was "
                "prepared with; rebuild the preparation")
        from woof.static.highres_production import parse_sealed_static_highres
        parse_sealed_static_highres(identity.get("static_highres"),
                                    source="prepared high-resolution identity",
                                    base_dir=Path("."))
    if identity.get("trace_gas_overrides") != proof.get("trace_gas_overrides"):
        raise ValueError("prepared cache trace-gas settings differ from preparation proof")
    return identity


def _require_preprocessing_binding(bound, receipt) -> None:
    """Hold a cache's bound preprocessing to its proof's receipt.

    The cache binds the receipt less what the preparation measured
    (:func:`woof.ingest.preprocess_backend.preprocess_identity`, A138);
    a cache written before that bound the whole receipt and still
    restores.  Anything else was prepared by other code or another
    backend than the proof names, and restoring it would integrate
    arrays the proof does not describe.
    """

    from woof.ingest.preprocess_backend import preprocess_identity_matches

    if not preprocess_identity_matches(bound, receipt):
        raise ValueError(
            "prepared cache preprocessing identity differs from the proof")


def _physical_base_manifest_digest(manifest_sha256, manifest_files):
    """Resolve the raw authority when a portable bundle wraps its manifest.

    A native bundle can carry both its portable input-list digest and a
    separately verified original source manifest. Physical capture binds
    the latter, just as the native prepared-cache identity does.
    """
    original = manifest_files.get("source_manifest")
    if original is not None and original.get("sha256") is None:
        from woof.ingest.boundary_stream import is_as_posted_placeholder
        if is_as_posted_placeholder(manifest_sha256):
            return manifest_sha256
    return (manifest_sha256 if original is None else
            _require_digest(original.get("sha256"), "physical base source manifest"))


def _validate_source_identity(
        source: str, identity: object, manifest_sha256: str,
        manifest_files: Mapping[str, Mapping[str, object]], proof,
        *, layout: str, mapped_authority: Mapping[str, object] | None = None,
        grid_id: int = 1,
        experiment_config: Path | None = None,
        experiment_config_sha256: str | None = None,
        physical_head: Mapping[str, object] | None = None,
        ordinary_input_plan: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if not isinstance(identity, dict):
        raise ValueError("prepared cache source identity must be an object")
    from woof.ensemble.posted_preparation import BINDING_KEY, validate_posted_member_head
    bound_source = identity.get("root_preparation", identity)
    if BINDING_KEY in bound_source:
        validate_posted_member_head(bound_source[BINDING_KEY], identity={"source_identity": identity})
        baseline = dict(identity)
        if "root_preparation" in baseline:
            baseline["root_preparation"] = {key: value for key, value in bound_source.items() if key != BINDING_KEY}
        else:
            baseline.pop(BINDING_KEY)
        baseline_proof = dict(proof)
        if baseline_proof.get("source_identity") == identity:
            baseline_proof["source_identity"] = baseline
        _validate_source_identity(source, baseline, manifest_sha256, manifest_files, baseline_proof,
            layout=layout, mapped_authority=mapped_authority, grid_id=grid_id,
            experiment_config=experiment_config, experiment_config_sha256=experiment_config_sha256,
            physical_head=physical_head, ordinary_input_plan=ordinary_input_plan)
        return identity
    if "ensemble_posted_physical_input" in identity:
        from woof.ensemble.posted_physical import (
            validate_posted_physical_input, validate_posted_native_source)
        if physical_head is None or "ensemble_physical_input" in identity:
            raise ValueError("posted physical inputs need one prepared provider head authority")
        binding = identity["ensemble_posted_physical_input"]
        baseline = {key:value for key,value in identity.items() if key != "ensemble_posted_physical_input"}
        _validate_source_identity(source, baseline, manifest_sha256, manifest_files, proof,
            layout=layout, mapped_authority=mapped_authority, grid_id=grid_id,
            experiment_config=experiment_config, experiment_config_sha256=experiment_config_sha256)
        if binding != physical_head["initial_receipt"]["binding"]:
            raise ValueError("prepared source initial physical input differs from its provider head")
        validate_posted_physical_input(binding, provider_plan=physical_head["provider_plan"],
            member_index=physical_head["member_index"],
            valid_time=datetime.fromisoformat(physical_head["forcing_valid_times"][0]))
        physical_manifest = _physical_base_manifest_digest(manifest_sha256, manifest_files)
        # Some native routes bind this authority at the cache identity level.
        # Capture supplies the same verified value in its physical source view.
        validate_posted_native_source(binding,
            source_identity={"input_manifest_sha256": physical_manifest, **baseline},
            input_manifest_authority=physical_manifest,
            input_plan=ordinary_input_plan)
        return identity
    if "ensemble_physical_input" in identity:
        from woof.ensemble.physical_store import validate_physical_input_binding
        binding = identity["ensemble_physical_input"]
        baseline = {key: value for key, value in identity.items() if key != "ensemble_physical_input"}
        validate_physical_input_binding(binding, input_manifest_sha256=
                                        _physical_base_manifest_digest(manifest_sha256, manifest_files),
                                        source_identity=baseline)
        _validate_source_identity(
            source, baseline, manifest_sha256, manifest_files, proof,
            layout=layout, mapped_authority=mapped_authority, grid_id=grid_id,
            experiment_config=experiment_config,
            experiment_config_sha256=experiment_config_sha256)
        return identity
    if source in _MAPPED_SOURCES:
        if mapped_authority is None:
            raise ValueError("mapped source authority is missing")
        expected = {
            "adapter": _SOURCE_ADAPTER[source],
            "mapping_sha256": mapped_authority["mapping_sha256"],
            "composition_sha256": mapped_authority["composition_sha256"],
            "input_manifest_sha256": manifest_sha256,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "prepared cache source identity differs from the mapped authorities")
        # The receipt less where the run's files sat since A151, the whole
        # receipt's digest before; either names this proof's composition.
        bound_receipt = identity.get("composition_receipt_sha256")
        if bound_receipt not in (
                mapped_authority["receipt_content_sha256"],
                mapped_authority.get("receipt_identity_sha256")):
            raise ValueError(
                "prepared cache source identity differs from the mapped authorities")
        expected["composition_receipt_sha256"] = bound_receipt
        _require_preprocessing_binding(
            identity.get("preprocessing"), mapped_authority["preprocessing"])
        allowed = set(expected) | {"preprocessing"}
        if "initial_source" in proof or "initial_source" in identity:
            if ("initial_source" not in proof
                    or identity.get("initial_source") != proof["initial_source"]):
                raise ValueError("prepared initial analysis differs from the proof's source evidence")
            allowed.add("initial_source")
        if "static_highres" in identity:
            # Shared preflight below compares this owner-normalized block
            # with the exact experiment authority for every source family.
            allowed.add("static_highres")
        case_fields = {"preparation_case_policy", "water_temperature_overlay"}
        if case_fields & set(identity):
            if not case_fields <= set(identity):
                raise ValueError("mapped preparation case identity is incomplete")
            if experiment_config is None or experiment_config_sha256 is None:
                raise ValueError("mapped preparation case identity needs the pinned experiment authority")
            from woof.case_data import (
                optional_case_data_from_config, preparation_case_policy)
            from woof.ingest.water_overlay import overlay_file_identity
            data = optional_case_data_from_config(
                experiment_config, expected_sha256=experiment_config_sha256)
            policy = preparation_case_policy(data)
            if identity["preparation_case_policy"] != policy:
                raise ValueError("mapped preparation case policy differs from experiment authority")
            overlay = None if data is None else data.water_temperature_overlay
            expected_overlay = None if overlay is None else overlay_file_identity(overlay)
            if identity["water_temperature_overlay"] != expected_overlay:
                raise ValueError("mapped water-temperature overlay identity differs from its declared bytes")
            allowed.update(case_fields)
        if layout == "mapped-hierarchy-d01-v1":
            allowed.update({
                "target_contract", "nested_source_orography",
                "hierarchy_implementation_sha256", "grid_id"})
            if (identity.get("target_contract")
                    != mapped_authority.get("target_contract")
                    or identity.get("grid_id") != grid_id
                    or not isinstance(identity.get("nested_source_orography"), dict)
                    or not isinstance(
                        identity.get("hierarchy_implementation_sha256"), dict)):
                raise ValueError(
                    "mapped hierarchy cache source identity is incomplete")
        if set(identity) != allowed:
            raise ValueError(
                "mapped prepared cache source identity has an unsupported shape")
        return identity
    if source == "hrrr":
        return _validate_hrrr_source_identity(identity, proof)
    required = {"adapter", "input_manifest_schema", "input_manifest_sha256",
                "decoder", "preprocessing"}
    if not required <= set(identity):
        raise ValueError("prepared cache source identity is incomplete")
    if (identity.get("adapter") != _SOURCE_ADAPTER[source]
            or identity.get("input_manifest_schema") != _SOURCE_SCHEMA[source]
            or identity.get("input_manifest_sha256") != manifest_sha256):
        raise ValueError(
            "prepared cache source identity differs from the selected adapter "
            "or portable manifest")
    decoder = identity.get("decoder")
    expected_decoder = {
        "name": manifest_files["bridge"]["name"],
        "sha256": manifest_files["bridge"]["sha256"],
        "implementation": ("gpuwm-all-rust-netcdf-bridge"
            if source == "era5" and Path(manifest_files["grib"]["name"]).suffix.lower() == ".nc"
            else _DECODER_IMPLEMENTATION[source]),
    }
    if decoder != expected_decoder or proof.get("decoder_sha256") \
            != expected_decoder["sha256"]:
        raise ValueError(
            "prepared cache decoder identity differs from proof/manifest")
    _require_preprocessing_binding(
        identity.get("preprocessing"), proof.get("preprocessing"))
    if source == "gfs":
        if (identity.get("implementation_sha256")
                != proof.get("implementation_sha256")
                or identity.get("git_source_identity")
                != proof.get("git_source_identity")):
            raise ValueError(
                "GFS implementation/source identity differs from its proof")
    return identity


def _validate_artifact_record(
        record, *, expected_path: str, actual: Path, label: str,
) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} artifact record is malformed")
    expected = {
        "path": expected_path,
        "bytes": actual.stat().st_size,
        "sha256": _sha256(actual),
    }
    if record != expected:
        raise ValueError(f"{label} artifact record differs from the supplied file")


def _validate_cache_metadata(
        reader: PreparedCacheReader, *, source: str, exp, forcing_hours,
        boundary_interval_seconds: int, proof, layout: str,
        posted_placeholder: str | None = None,
        physical_catalog: Mapping[str, object] | None = None,
        boundary_publication=(), prepared_root: Path | None = None,
) -> None:
    """The cache's metadata against the source, the proof and the experiment.

    ``posted_placeholder``: an as-posted mapped head's plan placeholder,
    which its metadata binds where the composition receipt's digest goes
    until the seal writes the receipt.

    ``boundary_publication``: the hydrometeor masses the cache's source
    publishes on every frame (:func:`_boundary_publication`); a cache
    whose boundary lacks any a fresh preparation would carry is refused
    (:func:`_refuse_stale_boundary`).
    """
    metadata = reader.header.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("prepared cache metadata must be an object")
    user = metadata.get("user")
    expected_user = {
        "initial_valid_time": exp.start_time.isoformat(),
        "last_valid_time": (
            exp.start_time + timedelta(hours=forcing_hours[-1])).isoformat(),
        "forcing_hours": list(forcing_hours),
    }
    receipt_keys = CONDITIONAL_PREPARATION_RECEIPTS
    if layout == HRRR_DIRECT_LAYOUT:
        receipt_keys = SOIL_PREPARATION_RECEIPTS
        # The native preparation's own user metadata, which names the
        # HRRR cycle and both lead vocabularies rather than an adapter
        # string and a preprocessing receipt.  Compared exactly, like
        # every other layout's -- the shape differs, the strictness does
        # not.  ``mapping_reports`` is a per-field decode report whose
        # contents are the preparer's, so its PRESENCE is required and
        # its value is not re-derived here.
        expected_user.update({
            "source_cycle": proof.get("source_cycle"),
            "source_forecast_hours": proof.get("source_forecast_hours"),
            "model_forcing_hours": list(forcing_hours),
        })
        reports = user.get("mapping_reports") if isinstance(user, dict) else None
        if not isinstance(reports, dict):
            raise ValueError(
                "HRRR prepared cache carries no field mapping reports")
        expected_user["mapping_reports"] = reports
    elif layout == "portable-single-domain-v2":
        expected_user.update({
            "source_adapter": source,
            "boundary_interval_seconds": boundary_interval_seconds,
        })
        # The receipt less its measurements since A138, the whole receipt
        # before; either is this proof's preparation.
        bound = user.get("preprocessing") if isinstance(user, dict) else None
        _require_preprocessing_binding(bound, proof.get("preprocessing"))
        expected_user["preprocessing"] = bound
    elif layout in ("mapped-direct-d01-v1", "mapped-hierarchy-d01-v1"):
        # The receipt less where the run's files sat since A151, the whole
        # receipt's digest before; either is this proof's composition.
        from woof.mapped_composition import (
            composition_receipt_binding_matches)

        receipt = proof.get("source_composition") or {}
        bound = (user.get("composition_receipt_sha256")
                 if isinstance(user, dict) else None)
        if posted_placeholder is not None:
            # An as-posted head: the receipt is the seal's to write, and
            # the sealed preflight checks the digest that replaces this.
            bound = posted_placeholder
        elif not composition_receipt_binding_matches(bound, receipt):
            # Another composition's digest: the exact comparison below
            # refuses it, naming the metadata as it always did.
            bound = receipt.get("receipt_content_sha256")
        if layout == "mapped-direct-d01-v1":
            expected_user.update({
                "source_adapter": "mapped",
                "boundary_interval_seconds": boundary_interval_seconds,
                "composition_receipt_sha256": bound,
            })
        else:
            expected_user.update({
                "composition_receipt_sha256": bound,
                "mapped_target_contract": proof.get("target_contract"),
            })
    # Preparation receipts the preparer binds only when they fired.  The
    # comparison below is exact, so a receipt the writer recorded and this
    # list omitted made the front door refuse ITS OWN cache -- the SMCDRY
    # floor fires on most clipped ERA5, which is why real domains at and
    # above 448^2 could not be prepared and read back.
    #
    # Bound from the PROOF, which carries the identical receipt, rather
    # than waved through: a cache whose receipt differs from its proof
    # still refuses, and so does one that records a receipt its proof does
    # not (and vice versa) -- each of those is a genuine inconsistency.
    for key in receipt_keys:
        if key in proof or (isinstance(user, dict) and key in user):
            expected_user[key] = proof.get(key)
    if physical_catalog is not None:
        from woof.ensemble.physical_boundary import KEY
        expected_user[KEY] = dict(physical_catalog)
    if user != expected_user:
        raise ValueError(
            "prepared cache user metadata differs from source/proof/experiment")
    if set(metadata.get("surface_fields", ())) != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared cache lacks the exact source-neutral Noah surface inventory")
    if not _REQUIRED_MET_FIELDS <= set(metadata.get("met_fields", ())):
        raise ValueError("prepared cache lacks required near-surface physics fields")
    lbc = metadata.get("lbc")
    if not isinstance(lbc, dict):
        raise ValueError("prepared cache has no standalone external LBC inventory")
    cfg = exp.root.run
    if ({key: lbc.get(key) for key in (
            "spec_bdy_width", "spec_zone", "relax_zone")} != {
                "spec_bdy_width": int(cfg.spec_bdy_width),
                "spec_zone": int(cfg.spec_zone),
                "relax_zone": int(cfg.relax_zone),
            }):
        raise ValueError("prepared cache LBC width differs from experiment")
    intervals = lbc.get("intervals")
    if not isinstance(intervals, list) or len(intervals) != len(forcing_hours) - 1:
        raise ValueError("prepared cache LBC interval count differs from forcing")
    admitted_fields = _lbc_field_inventories(cfg)
    from woof.boundary_fields import admissible_boundary_inventory
    scheme_id = int(getattr(cfg, "mp_physics", 0))
    for index, interval in enumerate(intervals):
        window = {
            "start_seconds": float(forcing_hours[index] * 3600),
            "end_seconds": float(forcing_hours[index + 1] * 3600),
        }
        recorded = interval if isinstance(interval, dict) else {}
        if {key: recorded.get(key) for key in window} != window:
            raise ValueError(
                f"prepared cache LBC interval {index} spans "
                f"{recorded.get('start_seconds')}-{recorded.get('end_seconds')} s "
                f"where the forcing has {window['start_seconds']:g}-"
                f"{window['end_seconds']:g} s; re-prepare against this forcing")
        fields = recorded.get("fields")
        if (not isinstance(fields, list) or fields != sorted(fields)
                or not set(_LBC_DYNAMICS_FIELDS) <= set(fields)
                or not admissible_boundary_inventory(
                    cfg, [name for name in fields
                          if name not in _LBC_DYNAMICS_FIELDS])):
            raise ValueError(
                f"prepared cache LBC interval {index} carries fields "
                f"{fields} where mp_physics={scheme_id} forces "
                f"{' or '.join(str(f) for f in admitted_fields)}, with the "
                "analysed hydrometeors its source publishes and the numbers "
                "seeded from them; re-prepare with this configuration")
    _refuse_stale_boundary(
        cfg, metadata, published=boundary_publication, source=source,
        prepared_root=prepared_root)


def _boundary_publication(source: str, mapped_paths) -> tuple[str, ...]:
    """The hydrometeor masses a prepared root's source publishes on every frame.

    A mapped preparation answers with the mapping document it copied
    into its evidence, which the evidence check has already held to the
    packaged pin (or, for a caller's own mapping, to its manifest): that
    document's declared hydrometeors are what a fresh preparation from
    it decodes.  A native source answers with its registry row.
    """
    from woof.boundary_fields import (mapping_boundary_species,
                                       source_boundary_species)

    mapping_path = (mapped_paths or {}).get("mapped_mapping")
    if source in _MAPPED_SOURCES and mapping_path is not None:
        return mapping_boundary_species(
            _load_json_object(Path(mapping_path), "mapped mapping evidence"))
    return source_boundary_species(source)


def _reprepare_action(source: str, prepared_root: Path | None) -> str:
    """How a door that refuses a stale prepared root says to replace it."""
    root = "this root" if prepared_root is None else str(prepared_root)
    return (f"run `woof prep --source {source}` again on the inputs, "
            f"experiment config and namelist.wps {root} was prepared from "
            "(its proof and source evidence name them), into a new "
            "--output-root, and run from that root")


def _refuse_stale_boundary(cfg, metadata, *, published, source: str,
                           prepared_root: Path | None) -> None:
    """Refuse a cache whose boundary lets its analysed hydrometeors drain.

    A cache sealed before the root's specified boundary carried the
    analysed hydrometeors its source publishes (woof.boundary_fields)
    used to run on a water-vapour-only boundary with a warning.  Run, it
    drains the analysed cloud and snow out of its edges, and a warning
    left it looking fixed, so it is refused and named instead.  Read off
    the cache's own header: its start state holds a mass nonzero
    (``hydrometeor_initialization``), its source publishes it, its scheme
    carries it, and its sealed interval tables do not.  A cache whose
    source publishes none, or whose start state holds none of what its
    boundary lacks, is what a fresh preparation writes for it and is
    admitted unchanged: the GFS and ERA5 caches whose boundaries did not
    change are not touched.
    """
    from woof.boundary_fields import (analysed_start_species,
                                       sealed_boundary_species,
                                       stale_boundary_refusal,
                                       stale_boundary_species)

    lbc = (metadata or {}).get("lbc") if isinstance(metadata, Mapping) else None
    intervals = lbc.get("intervals") if isinstance(lbc, Mapping) else None
    lacking = stale_boundary_species(
        cfg, published, sealed_boundary_species(intervals),
        analysed=analysed_start_species(metadata))
    if lacking:
        raise ValueError(stale_boundary_refusal(
            subject=("this prepared cache" if prepared_root is None
                     else f"the prepared cache under {prepared_root}"),
            source=source, lacking=lacking,
            action=_reprepare_action(source, prepared_root)))


def _stale_profile_mapping_refusal(
        *, source: str, profile_id: str | None, mapping_bytes: bytes,
        composition_sha256: str, prepared_root: Path) -> str | None:
    """A packaged profile's preparation from before its mapping declared masses.

    A preparation copies the mapping it ran into its evidence.  When a
    packaged profile's mapping later declares the hydrometeors its
    publisher writes, the earlier preparation's boundary carries none of
    them, and its bound mapping no longer matches the pin.  Named as
    that profile's (``profile_id``), or, when the door was handed it as
    a caller's own ``mapped`` preparation, as the profile whose pinned
    composition it binds and whose mapping name it carries, so a stale
    root cannot run silently under the caller-mapping route either.
    ``None`` when the bound mapping lacks none of the profile's masses.
    """
    from woof.boundary_fields import (mapping_boundary_species,
                                       stale_boundary_refusal)
    from woof.source_adapters import packaged_profile_sources
    from woof.source_authorities import (packaged_authorities,
                                          packaged_authority_sha256)

    try:
        bound = json.loads(mapping_bytes)
    except (UnicodeError, ValueError):
        return None
    if not isinstance(bound, dict):
        return None
    candidates = ([(source, profile_id)] if profile_id else [
        (name, profile) for name, profile in packaged_profile_sources().items()
        if packaged_authority_sha256(profile)["composition"]
        == composition_sha256])
    carried = set(mapping_boundary_species(bound))
    for name, profile in candidates:
        packaged = json.loads(
            packaged_authorities(profile)["mapping"].read_bytes())
        if not profile_id and packaged.get("name") != bound.get("name"):
            continue
        lacking = tuple(mass for mass in mapping_boundary_species(packaged)
                        if mass not in carried)
        if lacking:
            return stale_boundary_refusal(
                subject=(f"the prepared root {prepared_root} ({profile}, "
                         "prepared from an earlier copy of its mapping)"),
                source=name, lacking=lacking,
                action=_reprepare_action(name, prepared_root))
    return None


def _validate_hierarchy_d01_artifacts(
        layout: _PreparedLayout, *, static: Mapping[str, object],
        geometry_receipt: Mapping[str, object], reader: PreparedCacheReader,
        static_sha256: str, geometry_sha256: str,
) -> None:
    receipt = layout.domain_receipt
    if receipt is None:
        raise ValueError("hierarchy d01 receipt is missing")
    expected_artifacts = {
        "prepared_cache": {
            "path": "prepared-cache",
            "content_sha256": reader.content_sha256,
            "payload_bytes": reader.payload_bytes,
            "array_count": len(reader.arrays),
        },
        "static_cache": {
            "path": "native-static.npz",
            "bytes": layout.static_path.stat().st_size,
            "sha256": static_sha256,
            "fields": sorted(static),
        },
        "geometry_receipt": {
            "path": "geometry-receipt.json",
            "sha256": geometry_sha256,
            "geometry": geometry_receipt.get("geometry"),
        },
    }
    expected_verification = {
        "schema": PREPARED_CACHE_SCHEMA,
        "status": "PASS",
        "path": "prepared-cache",
        "content_sha256": reader.content_sha256,
        "array_count": len(reader.arrays),
        "payload_bytes": reader.payload_bytes,
    }
    if (receipt.get("artifacts") != expected_artifacts
            or receipt.get("verification") != expected_verification):
        raise ValueError(
            "hierarchy d01 artifact receipt differs from its verified files")


def _validate_mapped_static_proof(
        proof: Mapping[str, object], layout: _PreparedLayout,
        *, source: str, static: Mapping[str, object],
        geometry_receipt: Mapping[str, object], static_sha256: str,
) -> None:
    """Static/geometry half of a packaged mapped bundle's proof.

    ``source`` names the profile in every refusal below.  It is a
    parameter for the same reason the function is no longer named for one
    source: this runs for every packaged mapped profile, and a refusal
    that names the wrong one sends the reader down the wrong route.
    """
    if layout.kind == "mapped-direct-d01-v1":
        expected_static = {
            "path": "native-static.npz",
            "bytes": layout.static_path.stat().st_size,
            "sha256": static_sha256,
            "fields": sorted(static),
        }
        if (proof.get("static") != expected_static
                or proof.get("geometry") != geometry_receipt):
            raise ValueError(
                f"mapped {source} direct static/geometry proof differs")
        return
    root_static = layout.authority_paths.get("mapped_root_static")
    root_geometry_path = layout.authority_paths.get("mapped_root_geometry")
    if root_static is None or root_geometry_path is None:
        raise ValueError(f"mapped {source} hierarchy root authorities are missing")
    try:
        with np.load(root_static, allow_pickle=False) as archive:
            root_fields = sorted(archive.files)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"mapped {source} hierarchy root static cache is unreadable") from exc
    expected_root_static = {
        "path": "native-static.npz",
        "bytes": root_static.stat().st_size,
        "sha256": _sha256(root_static),
        "fields": root_fields,
    }
    root_geometry = _load_json_object(
        root_geometry_path, "mapped hierarchy root geometry receipt")
    if (proof.get("root_static") != expected_root_static
            or proof.get("root_geometry") != root_geometry):
        raise ValueError(
            f"mapped {source} hierarchy root static/geometry proof differs")


def _validate_hierarchy_wrf_authority(
        layout: _PreparedLayout, *, source: str, source_exp,
        forcing_hours: tuple[int, ...], boundary_interval_seconds: int,
        source_manifest_sha256: str, decoder_sha256: object,
        preprocessing, contract_sha256: str, prepared_content_sha256: str,
        prepared_header_sha256: str, static_sha256: str,
        geometry_sha256: str,
        mapped_authority: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    wrf = layout.wrf_manifest
    if wrf is None or layout.artifact_manifest is None:
        raise ValueError("hierarchy WRF/artifact authority is missing")
    if (wrf.get("schema") != "gpuwm-native-direct-wrf-hierarchy-export-v1"
            or wrf.get("status") != "READY"
            or wrf.get("valid_time")
            != source_exp.start_time.strftime("%Y-%m-%d_%H:%M:%S")
            or wrf.get("forcing_hours") != list(forcing_hours)
            or wrf.get("boundary_interval_seconds")
            != boundary_interval_seconds
            or wrf.get("hierarchy") != _hierarchy_rows(source_exp)):
        raise ValueError(
            "hierarchy direct-WRF manifest differs from source experiment")
    source_receipt = wrf.get("source")
    if not isinstance(source_receipt, dict) \
            or source_receipt.get("contract_sha256") != contract_sha256:
        raise ValueError("hierarchy direct-WRF contract identity differs")
    domain_sources = source_receipt.get("domains")
    expected_domain_keys = {
        f"d{int(domain.grid_id):02d}" for domain in source_exp.domains}
    if not isinstance(domain_sources, dict) \
            or set(domain_sources) != expected_domain_keys:
        raise ValueError("hierarchy direct-WRF domain source inventory differs")
    expected_d01 = {
        "prepared_header_sha256": prepared_header_sha256,
        "prepared_content_sha256": prepared_content_sha256,
        "static_cache_sha256": static_sha256,
        "geometry_receipt_sha256": geometry_sha256,
        "mp_physics": int(source_exp.root.run.mp_physics),
        "microphysics": stock_wrf_physics_inventory(
            int(source_exp.root.run.mp_physics)).scheme,
        "resolved_physics_contract_sha256": (
            _resolved_wrf_direct_contract_sha256(
                source_exp.root.run.mp_physics)),
    }
    if domain_sources.get("d01") != expected_d01:
        raise ValueError(
            "hierarchy direct-WRF d01 hashes differ from preparation")
    provenance = source_receipt.get("input_provenance")
    expected_provenance: dict[str, object] = {
        "input_manifest_sha256": source_manifest_sha256,
        "decoder_sha256": decoder_sha256,
        "preprocessing": preprocessing,
        "regular_source_adapter": (
            "rw-wps-mapped"
            if source in _MAPPED_SOURCES else source),
        "native_artifact_manifest": (
            "../hierarchy-artifacts/domain-artifacts.json"),
        "native_artifact_manifest_sha256": _sha256(
            layout.authority_paths["hierarchy_artifact_manifest"]),
    }
    if source in _MAPPED_SOURCES:
        if mapped_authority is None:
            raise ValueError("20CRv3 hierarchy mapped authority is missing")
        expected_provenance.update({
            "mapping_sha256": mapped_authority["mapping_sha256"],
            "composition_sha256": mapped_authority["composition_sha256"],
            "mapped_target_contract": mapped_authority["target_contract"],
        })
    if not isinstance(provenance, dict) or any(
            provenance.get(key) != value
            for key, value in expected_provenance.items()):
        raise ValueError(
            "hierarchy direct-WRF input provenance differs from authorities")
    return MappingProxyType(expected_d01)


def _tolerated_absent_run_fields(
        observed: Mapping[str, object], run: dict) -> list[dict]:
    """Drop run fields a header written before them could not carry.

    THE V-12 HOLE, on the route the tolerant table does not cover.  A
    RunConfig field that joins the identity document refuses every
    single-domain bundle prepared before it, because the comparison below
    is a bare equality with one hand-written allowance -- and this has now
    happened five times in this package's history.  The tree route solved
    it with ingest.prepared_cache.DEFAULT_TOLERANT_IDENTITY_FIELDS, whose
    entry means "no code read this at preparation time, so a header
    written without it describes the same prepared state".  That argument
    holds identically here, so this reads THAT table rather than a second
    hand-typed list, and a field registered there is covered on both
    routes in the commit that registers it.

    Two conditions, both required: the field must be ABSENT from the
    prepared header (a header that carries a value is compared to it),
    and the current value must be the not-in-use one from
    ``undelayed_identity_defaults`` (a bundle really running the feature
    was prepared for a different clock and is refused by name).
    """
    from woof.ingest.prepared_cache import (
        DEFAULT_TOLERANT_IDENTITY_FIELDS, _run_config_defaults)

    observed_run = {}
    if isinstance(observed, Mapping):
        domain = observed.get("domain_config")
        if isinstance(domain, Mapping) and isinstance(
                domain.get("run"), Mapping):
            observed_run = domain["run"]
    names = sorted(path[len("run."):]
                   for path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                   if path.startswith("run."))
    if not names:
        return []
    not_in_use = _run_config_defaults(*names)
    overrides = []
    for name in names:
        if name in observed_run or name not in run:
            continue
        if run[name] != not_in_use[f"run.{name}"]:
            continue
        overrides.append({
            "field": f"domain_config.run.{name}",
            "prepared_identity": "field-absent",
            "current_loader_default": run.pop(name),
            "model_state_or_physics_changed": False,
            "reason": (
                "the field joined RunConfig after this bundle was "
                "prepared and holds its not-in-use value, which no code "
                "read at preparation time"),
        })
    return overrides


def _tolerated_absent_domain_fields(
        observed: Mapping[str, object], domain: dict,
        not_in_use: Mapping[str, object] | None) -> list[dict]:
    """The top-level half of :func:`_tolerated_absent_run_fields`.

    The same table's DomainConfig members (``history_begin_s`` and
    ``history_end_s`` among them) under the same two conditions: absent
    from the prepared header, and holding the not-in-use value the
    caller read from ``undelayed_identity_defaults``.  Without this, the
    history window joining the document refused every bundle prepared
    before it on this route.  No map, no tolerance.
    """
    from woof.ingest.prepared_cache import DEFAULT_TOLERANT_IDENTITY_FIELDS

    if not not_in_use:
        return []
    observed_domain = {}
    if isinstance(observed, Mapping) and isinstance(
            observed.get("domain_config"), Mapping):
        observed_domain = observed["domain_config"]
    overrides = []
    for name in sorted(path for path in DEFAULT_TOLERANT_IDENTITY_FIELDS
                       if "." not in path):
        if (name in observed_domain or name not in domain
                or name not in not_in_use
                or domain[name] != not_in_use[name]):
            continue
        overrides.append({
            "field": f"domain_config.{name}",
            "prepared_identity": "field-absent",
            "current_loader_default": domain.pop(name),
            "model_state_or_physics_changed": False,
            "reason": (
                "the field joined DomainConfig after this bundle was "
                "prepared and holds its not-in-use value, which no code "
                "read at preparation time"),
        })
    return overrides


def _dropped_preparation_inert_run_fields(
        observed: Mapping[str, object], run: dict
) -> tuple[Mapping[str, object], list[dict]]:
    """Drop run fields preparation never reads, from BOTH sides.

    The companion to :func:`_tolerated_absent_run_fields`, and the
    stronger of the two rulings.  Tolerance covers a field ABSENT from
    an older header and holding its not-in-use value; this covers a
    field whose value cannot describe a difference in the artifact at
    all, so it is dropped in both directions and at any value -- which
    is the only thing that lets a bundle prepared under one adaptive
    target be run under another.

    Same tables as the tree route's walk
    (``ingest.prepared_cache.PREPARATION_INERT_RUN_FIELDS``, the
    output-only ``INERT_DIAGNOSTIC_IDENTITY_FIELDS`` and the run-window
    ``NON_TRAJECTORY_IDENTITY_FIELDS``), so a field registered in any of
    them is covered on both routes by that one entry rather than a second
    hand-typed list here.  The third table was missing from this door:
    ``effective_prepared_domain_config`` (the tree route) dropped
    ``run.run_seconds`` under the ruling that a prepared cache is an
    initial state plus the boundary tables for the times it decoded, but
    this comparison still named it, so a run shorter than the prepared
    length was refused as "prepared cache identity differs" the moment
    the executed length stopped being silently replaced by the prepared
    one.
    """
    from woof.ingest.prepared_cache import (
        INERT_DIAGNOSTIC_IDENTITY_FIELDS, NON_TRAJECTORY_IDENTITY_FIELDS,
        PREPARATION_INERT_RUN_FIELDS)

    names = sorted(path[len("run."):]
                   for path in (PREPARATION_INERT_RUN_FIELDS
                                | INERT_DIAGNOSTIC_IDENTITY_FIELDS
                                | NON_TRAJECTORY_IDENTITY_FIELDS)
                   if path.startswith("run."))
    if not names:
        return observed, []
    observed_run = {}
    if isinstance(observed, Mapping):
        domain = observed.get("domain_config")
        if isinstance(domain, Mapping) and isinstance(
                domain.get("run"), Mapping):
            observed_run = domain["run"]
    overrides = []
    for name in names:
        in_header = name in observed_run
        in_live = name in run
        if not in_header and not in_live:
            continue
        prepared = observed_run[name] if in_header else "field-absent"
        current = run.pop(name) if in_live else "field-absent"
        if prepared == current:
            continue
        overrides.append({
            "field": f"domain_config.run.{name}",
            "prepared_identity": prepared,
            "current_loader_default": current,
            "model_state_or_physics_changed": False,
            "reason": (
                "no code on the preparation path reads this field, so the "
                "prepared artifact is the same under either value"),
        })
    # Stripped unconditionally: the expected side has already had these
    # names popped, so leaving them on the observed side would turn a
    # dropped field into a difference -- the exact refusal this removes.
    stripped = json.loads(_canonical(observed))
    stripped_run = stripped.get("domain_config", {}).get("run", {})
    for name in names:
        stripped_run.pop(name, None)
    return stripped, overrides


def _resolve_cache_identity_compatibility(
        *, source: str, observed: Mapping[str, object],
        expected: Mapping[str, object],
        not_in_use: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if observed == expected:
        return expected, MappingProxyType({
            "schema": "gpuwm-prepared-cache-identity-compatibility-v1",
            "status": "EXACT",
            "compatibility_overrides": [],
        })
    compatible = json.loads(_canonical(expected))
    run = compatible.get("domain_config", {}).get("run", {})
    # Inert first: it drops from both sides, so what follows compares
    # only fields that can actually describe the prepared artifact.
    observed_cmp, inert = _dropped_preparation_inert_run_fields(
        observed, run)
    absent = _tolerated_absent_run_fields(observed_cmp, run)
    absent += _tolerated_absent_domain_fields(
        observed_cmp, compatible.get("domain_config", {}), not_in_use)
    if (absent or inert) and observed_cmp == compatible:
        return observed, MappingProxyType({
            "schema": "gpuwm-prepared-cache-identity-compatibility-v1",
            "status": "COMPATIBLE_LEGACY_DEFAULT",
            "compatibility_overrides": inert + absent,
        })
    removed = run.pop("nest_microphysics_transition", None)
    if (source in _MAPPED_SOURCES and removed == "same-scheme-only"
            and observed_cmp == compatible):
        return observed, MappingProxyType({
            "schema": "gpuwm-prepared-cache-identity-compatibility-v1",
            "status": "COMPATIBLE_LEGACY_DEFAULT",
            "compatibility_overrides": [{
                "field": "domain_config.run.nest_microphysics_transition",
                "prepared_identity": "field-absent",
                "current_loader_default": "same-scheme-only",
                "model_state_or_physics_changed": False,
                "reason": (
                    "the hash-bound legacy config predates this explicit "
                    "same-scheme default"),
            }],
        })
    # NAME WHAT DIFFERS.  The sentence above stood alone, and a user
    # reading it could not tell an upgrade from a changed configuration --
    # measured on a bundle prepared one commit earlier, where the whole
    # difference was twelve fields that had not existed yet.  The legacy
    # field goes back first: popped only to test the allowance above, it
    # was named as differing on a bundle whose header carries it equal.
    if removed is not None:
        run["nest_microphysics_transition"] = removed
    raise ValueError(
        "prepared cache identity differs from the requested source, static "
        "data, configuration, namelist, or allowed legacy default: "
        + _describe_identity_difference(observed_cmp, compatible))


def _describe_identity_difference(observed, expected, limit: int = 8) -> str:
    """The first few differing identity paths, as ``path (a -> b)``."""
    lines: list[str] = []

    def walk(a, b, path: str) -> None:
        if len(lines) >= limit:
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, _MISSING), b.get(key, _MISSING),
                     f"{path}.{key}" if path else str(key))
            return
        if a != b:
            lines.append(
                f"{path} (prepared "
                f"{'absent' if a is _MISSING else a!r} -> requested "
                f"{'absent' if b is _MISSING else b!r})")

    walk(observed, expected, "")
    if not lines:
        return "no field-level difference (the documents differ in shape)"
    return "; ".join(lines)


def _terrain_derivations(exp, static, grid, reader, physics_receipt, *,
                         basis_out: dict | None = None):
    """The experiment with the substeps and long step this domain's ground
    and crest-level wind need, each derivation recorded in the receipt.

    The substep count reads the static terrain; the long step reads the
    same slope, the static crest and the strongest crest-level wind the
    prepared cache's start state and boundary data carry.  ``basis_out``,
    when given, receives ``basis``: what the long-step derivation read
    before it chose a clock (:class:`woof.ingest.boundary_stream
    .ClockBasis`), which a head-bound forecast asks again as each boundary
    interval arrives (:class:`woof.prepared_domain_tree_forecast
    .StreamedClockGuard`).
    """

    root = int(exp.root.grid_id)
    exp, acoustic = adapt_experiment_to_terrain(
        exp, readings_from_static(exp, {root: static},
                                  grids_by_grid_id={root: grid}))
    if basis_out is not None:
        from woof.ingest.boundary_stream import ClockBasis

        basis_out["basis"] = ClockBasis(
            experiment=exp, acoustic=acoustic,
            readers=MappingProxyType({root: reader}),
            statics=MappingProxyType({root: static}), reach=None)
    exp, clock = clock_for_prepared_cache(
        exp, acoustic, readers={root: reader}, statics={root: static})
    physics_receipt["terrain_clock"] = clock_receipt(clock)
    # Written after the clock, so its count is the one that runs.
    physics_receipt["acoustic_substeps"] = acoustic_receipt(acoustic, clock)
    return exp


def _as_posted_head_binding(source: str, as_posted: Mapping[str, object],
                            source_manifest_sha256: str | None) -> str:
    """What an as-posted head binds where a sealed one binds its manifest.

    The head binds the input plan (every lead, every object name and
    every other input; DESIGN A136 2.4 item 5 and the L3 design ruling),
    which its head digest covers, and its cache identity carries the
    plan's placeholder where the seal writes the manifest's digest.  So
    the head preflight reads that placeholder in the manifest digest's
    place, and the seal preflight reads the sealed manifest's digest
    (:func:`_sealed_arguments`).
    """

    from woof.ingest.boundary_stream import (
        as_posted_placeholder, input_plan_sha256,
    )

    # A mapped head is bound here too: the evidence its seal writes (the
    # composition receipt, its alignment and frames, the vertical ladder
    # and the manifest copy) is checked by the sealed preflight the seal
    # re-runs, and the head is held to what it already carries
    # (:func:`_validate_packaged_mapped_evidence` with ``posted``).
    if source_manifest_sha256 is not None:
        # Breakage it prevents: the digest names a manifest this head cannot
        # hold, so it would be accepted and checked against nothing.
        raise ValueError(
            "--source-manifest-sha256 names an input manifest, and this "
            "prepared head is as posted: it binds its input plan, and its "
            "seal writes the manifest, held to that plan; omit the flag")
    plan = as_posted.get("input_plan")
    digest = as_posted.get("input_plan_sha256")
    if not isinstance(plan, Mapping) or input_plan_sha256(plan) != digest:
        raise ValueError(
            "the as-posted head's input plan is not the plan its digest "
            "names")
    return as_posted_placeholder(str(digest))


def _as_posted_pending_roles(as_posted, manifest) -> frozenset:
    """The manifest roles an as-posted head's plan leaves undigested."""

    if as_posted is None:
        return frozenset()
    from woof.ingest.boundary_stream import lead_payload_lead

    prefix = str(as_posted.get("lead_role_prefix") or "")
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    leads = {str(role) for role in (files or {})
             if lead_payload_lead(role, prefix) is not None}
    return frozenset(leads | {str(role) for role in
                              as_posted.get("derived_roles") or ()})


def _posted_documents(prepared_root, as_posted) -> frozenset:
    """The files an as-posted head's seal writes from its leads (none otherwise)."""

    if as_posted is None:
        return frozenset()
    return frozenset(
        (Path(prepared_root) / str(document["path"])).resolve()
        for document in dict(as_posted.get("documents") or {}).values())


def _sealed_arguments(inputs, sealed: Mapping[str, object]) -> dict:
    """The preflight arguments that bind a head's seal.

    The head's own, with the sealed proof and content digests; for an
    as-posted head also the digest of the manifest its seal wrote, which
    :func:`woof.ingest.boundary_stream.verify_seal` has held to the head's
    input plan and to every lead's posted marker.
    """

    arguments = {**dict(inputs.preflight_arguments),
                 "proof_sha256": sealed["proof_sha256"],
                 "prepared_content_sha256": sealed["content_sha256"]}
    posted = sealed.get("as_posted")
    if posted is not None:
        arguments["source_manifest_sha256"] = posted["input_manifest_sha256"]
    return arguments


def _require_sealed_identity(inputs, sealed_inputs, head) -> None:
    """The sealed cache identity is the head's, or an as-posted head's allowed change.

    An as-posted head's identity carries its plan's placeholder where the
    seal writes the input manifest's digest; exactly those keys may change,
    and only to that digest (the L3 design ruling,
    :func:`woof.ingest.boundary_stream.check_as_posted_identity`).
    """

    posted = (head or {}).get("basis", {}).get("as_posted")
    if posted is None:
        if sealed_inputs.cache_identity != inputs.cache_identity:
            raise RuntimeError(
                "the sealed preparation differs from the head this forecast "
                "started from")
        return
    from woof.ingest.boundary_stream import (
        BoundaryStreamError, check_as_posted_identity,
    )

    from woof.ingest.boundary_stream import document_bound_digests

    try:
        check_as_posted_identity(
            inputs.cache_identity, sealed_inputs.cache_identity,
            plan_sha256=posted["input_plan_sha256"],
            manifest_sha256=sealed_inputs.file_sha256["source_manifest"],
            manifest_bound=posted["manifest_bound_identity_keys"],
            # Native HRRR: the bridge's and the fetch's SHA256SUMS, each the
            # digest the sealed manifest names (verify_seal held their rows).
            document_bound=(document_bound_digests(
                posted, sealed_inputs.source_manifest)
                if posted.get("document_bound_identity_keys") else None))
    except BoundaryStreamError as error:
        raise RuntimeError(
            "the sealed preparation differs from the as-posted head this "
            f"forecast started from: {error}") from None


def _retained_preflight_payload(reader, *, layout, head):
    """Use the additive native hasher for sealed direct immutable mappings."""
    if head is not None or layout not in _DIRECT_LAYOUTS:
        return None
    return _sealed_preflight_payload(reader)


def _sealed_preflight_payload(reader):
    """Retain a sealed cache only when the native verifier is available."""
    from woof.ingest.prepared_writer import native_hasher
    try:
        entry = native_hasher()
    except (OSError, RuntimeError):
        # Older installs without this additive bridge retain their original
        # full checked reads. Source and byte identity rules are unchanged.
        return None
    if entry is None:
        return None
    from woof.ingest.prepared_store import _CachePayload
    return _CachePayload(reader, log=lambda _: None)


def _resident_prepared_reader(inputs, boundary_source):
    """Rebind sealed verified maps, keeping each restored host array writable."""
    if boundary_source is not None:
        return inputs.cache_reader
    payload = getattr(inputs, "cache_payload", None)
    if payload is None:
        return None
    payload.require_binding(inputs.prepared_cache_path, inputs.cache_identity,
                            reader=inputs.cache_reader)
    return payload.copying_reader()


def preflight_restart_algorithms(restart: Path, cfg) -> None:
    """Check the checkpoint's scheme identities without restoring a device."""
    from woof.io.restart import read_restart_header, _require_current_algorithm_identities

    _require_current_algorithm_identities(read_restart_header(restart), cfg, restart)


def preflight_prepared_forecast(
        *, source: str, prepared_root: Path, proof_sha256: str | None = None,
        source_manifest_sha256: str,
        prepared_content_sha256: str | None = None,
        prepared_head_sha256: str | None = None,
        experiment_config: Path, wps_namelist: Path,
        physics_profile: str | None = None,
        expert_acknowledgements: tuple[str, ...] = (),
        run_seconds: float, history_interval_seconds: float,
        domain_bundle: Path | None = None,
        tiles=None, devices: int | None = None, devices_options=None,
        simulated_radar=None, radar_heating=None,
        runtime_run_overrides: Mapping[str, object] | None = None,
) -> PreparedForecastInputs:
    """Validate every portable preparation authority without importing CuPy.

    ``runtime_run_overrides`` is a Python API mapping of RunConfig fields
    declared in PREPARATION_INERT_RUN_FIELDS. The original experiment and
    WPS bytes remain bound to the preparation, and each changed runtime
    value is validated and recorded in the execution plan. Preparation
    inputs, including grid, soil geometry and microphysics selectors,
    cannot be replaced through this mapping.

    ``tiles`` is an optional :class:`~woof.core.streaming.StreamingOptions`
    that REPLACES whatever the hash-bound experiment declares, and it is
    the one execution control this stage accepts from outside the bundle.

    It has to be, on one route.  The native HRRR chain hands this runner
    the authority its PREPARER published, which is rendered from tables
    that stage builds in code -- ``{experiment, projection, shared,
    domain}``, with no ``[tiles]`` among them and no way for a user's
    block to get into one.  So a config that asked to stream was read,
    validated and reported by the front door and then integrated
    resident, because the document that reached this function had never
    heard of the table.  The mode arrives as an argument instead.

    Overlaying rather than publishing is deliberate and is the reason
    this is sound.  ``[tiles]`` contributes NOTHING to the restart
    identity on purpose (:func:`woof.core.streaming.identity_payload_entry`
    returns ``{}``): a checkpoint written resident must resume streamed
    and one written streamed must resume resident, since a forecast that
    outgrew its card resuming on the machine it outgrew is the operation
    the mode exists for.  Rendering the table into the hash-bound
    document would bind the execution mode into the prepared bundle's
    identity and refuse exactly that.
    """

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported prepared forecast source {source!r}")
    # EXPERIMENT_CONFIG_PROFILE is the string form of "no named profile";
    # a caller that carries the binding as a string (the packed DA
    # controller and its workers share one argument record and one
    # ensemble identity) passes it through unchanged and it binds here.
    physics_profile = physics_profile_binding(physics_profile)
    if runtime_run_overrides is not None:
        if not isinstance(runtime_run_overrides, Mapping):
            raise TypeError("runtime_run_overrides must be a mapping of RunConfig fields")
        runtime_run_overrides = MappingProxyType(copy.deepcopy(dict(runtime_run_overrides)))
    preflight_arguments = MappingProxyType(dict(
        source=source, prepared_root=prepared_root,
        source_manifest_sha256=source_manifest_sha256,
        experiment_config=experiment_config, wps_namelist=wps_namelist,
        physics_profile=physics_profile,
        expert_acknowledgements=tuple(expert_acknowledgements),
        run_seconds=run_seconds,
        history_interval_seconds=history_interval_seconds,
        domain_bundle=domain_bundle, tiles=tiles, devices=devices,
        devices_options=devices_options,
        **({} if runtime_run_overrides is None else
           {"runtime_run_overrides": runtime_run_overrides}),
        **({} if simulated_radar is None else {"simulated_radar": simulated_radar}),
        **({} if radar_heating is None else {"radar_heating": radar_heating})))
    head = None
    #: An as-posted head's block (``basis.as_posted``): the head binds the
    #: input plan, and the manifest does not exist until its seal.
    as_posted = None
    if prepared_head_sha256 is not None:
        if proof_sha256 is not None or prepared_content_sha256 is not None:
            raise ValueError(
                "a prepared forecast binds a head or a sealed proof, not both")
        prepared_head_sha256 = _require_digest(
            prepared_head_sha256, "prepared-head-sha256")
    else:
        proof_sha256 = _require_digest(proof_sha256, "proof-sha256")
        prepared_content_sha256 = _require_digest(
            prepared_content_sha256, "prepared-content-sha256")
    prepared_root = _require_directory(prepared_root, "prepared root")
    if prepared_head_sha256 is not None:
        # THE HEAD BINDING (chained preparation).  Everything the start time
        # made is checked here exactly as a sealed tree is; the proof's
        # seal-only keys (the cache digest, the export, the wall times) are
        # checked at the seal, which re-runs this preflight on the sealed
        # tree before the forecast publishes anything that names them.
        from woof.ingest.boundary_stream import bind_head

        head = bind_head(prepared_root, prepared_head_sha256)
        as_posted = head["basis"].get("as_posted")
    if as_posted is not None:
        source_manifest_sha256 = _as_posted_head_binding(
            source, as_posted, source_manifest_sha256)
    else:
        source_manifest_sha256 = _require_digest(
            source_manifest_sha256, "source-manifest-sha256")
    if head is not None:
        named = head["basis"].get("input_manifest_sha256")
        if named is not None and named != source_manifest_sha256:
            raise ValueError(
                "the prepared head names a different source manifest than "
                "--source-manifest-sha256")
        proof_path = None
    else:
        proof_path = _require_file(prepared_root / "proof.json",
                                   "preparation proof")
    # An as-posted head has no manifest file yet: it reads the manifest its
    # plan names (every row, each lead's payload digest not yet known).
    source_manifest_path = None if as_posted is not None else _require_file(
        prepared_root / (
            "source-evidence/input-manifest.json"
            if source in _MAPPED_SOURCES
            else "source-input-manifest.json"),
        "portable source manifest")
    experiment_config = _require_file(experiment_config, "experiment config")
    wps_namelist = _require_file(wps_namelist, "WPS namelist")

    if head is None:
        actual_proof_sha256 = _sha256(proof_path)
        if actual_proof_sha256 != proof_sha256:
            raise _proof_digest_refusal(
                proof_path, proof_sha256, actual_proof_sha256)
    if source_manifest_path is not None and \
            _sha256(source_manifest_path) != source_manifest_sha256:
        raise ValueError(
            "portable source manifest SHA differs from "
            "--source-manifest-sha256")
    proof = (dict(head["basis"]["proof_head"]) if head is not None
             else _load_json_object(proof_path, "preparation proof"))
    manifest = (dict(as_posted["input_plan"]["manifest"])
                if as_posted is not None
                else _load_json_object(source_manifest_path,
                                       "portable source manifest"))
    source_exp = load_experiment(experiment_config)
    # THE COORDINATE THE PREPARED INPUTS CARRY, before anything derived
    # from the configuration's own etac exists (see
    # woof.vertical_adaptation.adopt_prepared_vertical).
    source_exp, _prepared_vertical = adopt_prepared_vertical(
        source_exp, proof, announce=_announce_adopted_coordinate)
    from woof.experiment import refuse_unrouted_spawn
    # [perturbation] is applied by run_prepared_forecast to the restored
    # state; its refusal here was retired with that apply step.
    refuse_unrouted_spawn(source_exp, "prepared single-domain forecast")
    # NO streaming refusal here.  This route wires a streamed-domain builder
    # (streaming.builders_for_tree, below), so mode = 'on' is something it
    # can do rather than something to refuse at admission.  The refusal that
    # used to stand here outlived the wiring by a release: it was written
    # when the route passed no builders, and it went on rejecting the one
    # mode that asks for streaming unconditionally -- with a message saying
    # the route "wires no streamed-domain builder" -- while mode = 'auto'
    # went through the very builder it said did not exist.
    # refuse_unrouted_streaming is still the right answer for a route that
    # genuinely reads nothing -- see its docstring -- but `woof run` is no
    # longer such a route: it wires builders_for_tree on its tree arm and
    # standalone_domain_builder on its single-domain arm, so its own copy
    # of this refusal is gone too.
    layout = _resolve_prepared_layout(
        source=source, prepared_root=prepared_root, proof=proof,
        source_exp=source_exp, domain_bundle=domain_bundle)
    static_path = layout.static_path
    geometry_receipt_path = layout.geometry_receipt_path
    prepared_cache_path = layout.prepared_cache_path
    geometry_receipt = _load_json_object(
        geometry_receipt_path, "geometry receipt")
    if head is None:
        cache_header_path = _require_file(
            prepared_cache_path / "header.json", "prepared cache header")
        header = _load_json_object(cache_header_path, "prepared cache header")
        if (header.get("schema") != PREPARED_CACHE_SCHEMA
                or header.get("status") != "READY"
                or header.get("content_sha256") != prepared_content_sha256):
            raise ValueError(
                "prepared cache header identity/status/content differs")
    else:
        if layout.kind in _HIERARCHY_LAYOUTS:
            raise ValueError(
                "a prepared hierarchy is published sealed; bind it with "
                "--proof-sha256 and --prepared-content-sha256")
        head_cache = head["basis"]["cache"]
        if (prepared_root / str(head_cache["directory"])).resolve() \
                != Path(prepared_cache_path).resolve():
            raise ValueError(
                "the prepared head names another cache directory than the "
                "bundle layout")
        cache_header_path = None
        header = {"schema": PREPARED_CACHE_SCHEMA, "status": "READY",
                  "identity": head_cache["identity"]}

    exp = source_exp if len(source_exp.domains) == 1 else replace(
        source_exp, domains=(source_exp.root,))
    prepared_run = exp.root.run
    exp, runtime_overrides = _apply_runtime_run_overrides(exp, runtime_run_overrides)
    # The requested length is the executed length; a longer one than the
    # prepared forcing reaches refuses at the coverage gate below.
    exp, _run_length = _apply_run_length(exp, run_seconds)
    physics_receipt = _validate_physics(
        exp, physics_profile, run_seconds, history_interval_seconds,
        source=source, expert_acknowledgements=expert_acknowledgements)
    front_door_physics = _validate_front_door_physics_proof(
        proof, source=source, profile=physics_profile, cfg=prepared_run)
    physics_receipt["execution_plan"] = _execution_plan_receipt(
        source_exp=source_exp, executed_exp=exp, profile=physics_profile,
        history_interval_seconds=history_interval_seconds)
    if runtime_overrides:
        physics_receipt["execution_plan"]["physics_overrides"] = runtime_overrides
    manifest_files, source_manifest_receipt = _manifest_file_specs(
        source, manifest, source_exp, proof,
        pending=_as_posted_pending_roles(as_posted, manifest),
        planned=as_posted is not None)
    mapped_paths: Mapping[str, Path] = MappingProxyType({})
    mapped_authority: Mapping[str, object] | None = None
    source_member: str | None = None
    if source in _MAPPED_SOURCES:
        mapped_paths, mapped_authority, source_member = (
            _validate_packaged_mapped_evidence(
                prepared_root=prepared_root, proof=proof, manifest=manifest,
                manifest_sha256=source_manifest_sha256,
                experiment_config=experiment_config,
                wps_namelist=wps_namelist, source=source,
                sealed=head is None, posted=as_posted))
    else:
        for role, actual in (
                ("experiment_config", experiment_config),
                ("wps_namelist", wps_namelist)):
            spec = manifest_files[role]
            # The manifest retains its validated publication name; an
            # explicit caller path is only a locator for those exact bytes.
            # Keep that actual path below so later mutation checks rehash it.
            if spec["sha256"] != _sha256(actual):
                raise ValueError(
                    f"supplied {role} differs from the portable source manifest")

    forcing_hours = proof.get("forcing_hours")
    if (not isinstance(forcing_hours, list) or len(forcing_hours) < 2
            or any(isinstance(hour, bool) or not isinstance(hour, int)
                   for hour in forcing_hours)
            or forcing_hours[0] != 0):
        raise ValueError("preparation proof forcing hours are malformed")
    forcing_hours = tuple(forcing_hours)
    deltas = {
        later - earlier
        for earlier, later in zip(forcing_hours, forcing_hours[1:])
    }
    if len(deltas) != 1 or next(iter(deltas)) <= 0:
        raise ValueError("preparation proof forcing cadence is not uniform")
    cadence_hours = next(iter(deltas))
    if source == "hrrr" and cadence_hours != 1:
        # Not a warning like the others: HRRR publishes hourly and the
        # native preparer's whole forcing contract
        # (woof.hrrr_forecast) is built on contiguous hourly leads, so
        # a non-hourly HRRR cadence is a malformed preparation rather
        # than a coarser one.
        raise ValueError(
            f"HRRR forcing cadence of {cadence_hours} h is not hourly; the "
            "native HRRR route prepares contiguous hourly leads")
    if (source == "gfs" and cadence_hours not in {1, 3}) \
            or (source == "era5" and cadence_hours < 1) \
            or (source in _MAPPED_SOURCES
                and cadence_hours < 1):
        # The cadence is uniform (checked above) and the coverage is
        # checked below; a cadence outside the blessed set is coarser
        # boundary forcing, not a broken preparation.
        warn(f"{source.upper()} forcing cadence of {cadence_hours} h is "
             "outside the set this runner has demonstrated "
             "(1 h/3 h for GFS); continuing with it as prepared")
    boundary_interval_seconds = cadence_hours * 3600
    if forcing_hours[-1] * 3600 < run_seconds:
        # The breakage this prevents: the solver stepping past the last
        # prepared lateral-boundary tendency.  The requested length is
        # the executed length (_apply_run_length), so this is the one
        # gate on it, and it names the lead the preparation stops at.
        raise ValueError(
            f"--run-seconds {run_seconds:g} asks for {run_seconds / 3600:g} h "
            f"but the prepared boundary forcing ends at lead "
            f"{forcing_hours[-1]} h ({forcing_hours[-1] * 3600:g} s); the "
            "forecast would step past its last prepared boundary "
            "tendency.  Prepare the longer window, or run at most "
            f"--run-seconds {forcing_hours[-1] * 3600:g}")
    if proof.get("boundary_interval_seconds") != boundary_interval_seconds:
        raise ValueError("prepared forcing cadence/coverage differs from the run")
    expected_times = [
        (exp.start_time + timedelta(hours=hour)).isoformat()
        for hour in forcing_hours
    ]
    if proof.get("forcing_times") != expected_times:
        raise ValueError(
            "prepared forcing valid times differ from the experiment start")
    if source == "gfs":
        manifest_hours = sorted(
            int(role.removeprefix("grib-f"))
            for role in manifest_files if role.startswith("grib-f"))
        # Two vocabularies, bound to each other rather than assumed
        # equal: the manifest names absolute NOAA leads, the proof's
        # forcing hours are model offsets from start_time, and the
        # declared lead is what maps one onto the other.  At lead 0 this
        # is the identity comparison it has always been.
        lead_hours = proof_initial_forecast_lead(proof)
        expected_source_hours = [lead_hours + hour for hour in forcing_hours]
        declared_source_hours = proof.get("source_forecast_hours")
        if declared_source_hours is None:
            declared_source_hours = expected_source_hours
        if declared_source_hours != expected_source_hours:
            raise ValueError(
                "preparation proof source forecast hours are not its "
                "forcing offsets shifted by the declared initial lead")
        if not set(expected_source_hours) <= set(manifest_hours):
            raise ValueError(
                "GFS manifest GRIB hours do not carry the proof's forcing "
                f"window f{expected_source_hours[0]:03d}.."
                f"f{expected_source_hours[-1]:03d}")
        unused = sorted(set(manifest_hours) - set(expected_source_hours))
        if unused:
            # Bound, decoded, and accurately not used: a manifest authored
            # over a whole fetch while the run starts partway into it.
            warn("the GFS source manifest binds forecast hour(s) "
                 + ", ".join(f"f{hour:03d}" for hour in unused)
                 + " that this run's forcing window does not use")
    elif source == "hrrr":
        # Two vocabularies again, and on this route they are ALWAYS
        # different: NOAA's absolute leads (f004..f010 of one cycle) and
        # the model-relative forcing offsets (0..6) the cache and the
        # exporter use.  woof.hrrr_forecast owns the window contract;
        # this binds the proof's two lists to each other so a bundle
        # cannot claim one window and carry another.
        from woof.hrrr_forecast import validate_hrrr_source_forecast_hours

        declared = proof.get("source_forecast_hours")
        cycle_text = proof.get("source_cycle")
        if not isinstance(declared, list) or not isinstance(cycle_text, str):
            raise ValueError(
                "HRRR preparation proof names no source cycle/lead window")
        try:
            cycle = datetime.fromisoformat(cycle_text)
            source_hours = validate_hrrr_source_forecast_hours(
                declared, cycle=cycle)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"HRRR preparation proof source window is invalid: {error}"
            ) from None
        if len(source_hours) != len(forcing_hours) \
                or list(range(len(source_hours))) != list(forcing_hours):
            raise ValueError(
                "HRRR preparation proof forcing offsets are not 0..N-1 of "
                "its own source lead window")
        if (cycle + timedelta(hours=source_hours[0])) != exp.start_time:
            raise ValueError(
                f"HRRR preparation proof cycle {cycle_text} at "
                f"f{source_hours[0]:03d} is not the experiment start "
                f"{exp.start_time.isoformat()}")
        if proof.get("model_start_time") != exp.start_time.isoformat() \
                or proof.get("model_forcing_hours") != list(forcing_hours):
            raise ValueError(
                "HRRR preparation proof model start/forcing differ from "
                "the hash-bound experiment")
    elif source == "20crv3":
        if (manifest.get("valid_times") != expected_times
                or manifest.get("cadence_seconds")
                != boundary_interval_seconds):
            raise ValueError(
                "20CRv3 exact-member manifest cadence/coverage differs from "
                "proof forcing")

    # An as-posted head's proof leaves the manifest's digest and rows to
    # its seal (the head's seal-authored proof keys), where the sealed
    # preflight checks them against the sealed manifest.
    if as_posted is None and source not in _MAPPED_SOURCES \
            and proof.get("input_manifest_sha256") != source_manifest_sha256:
        raise ValueError("preparation proof input manifest hash differs")
    if layout.kind in {"portable-single-domain-v2", HRRR_DIRECT_LAYOUT}:
        expected_source_inputs = {
            "manifest_schema": _SOURCE_SCHEMA[source],
            "manifest_sha256": source_manifest_sha256,
            "files": manifest.get("files"),
        }
        if as_posted is None \
                and proof.get("source_inputs") != expected_source_inputs:
            raise ValueError(
                "preparation proof source inputs differ from manifest")
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name=source.upper())
    elif layout.kind in _HIERARCHY_LAYOUTS:
        grid = validate_native_lambert_contracts(
            source_exp, wps_namelist, source_name=source.upper())[0]
    else:
        grid = validate_native_lambert_contract(
            exp, wps_namelist, source_name=source.upper())
    verify_native_static_receipt(
        geometry_receipt_path, static_path, grid, exp.root.run)
    static = load_native_static_cache(
        static_path, grid, exp.root.run.ny, exp.root.run.nx)
    from woof.static.orographic import require_prepared_terrain_drag
    require_prepared_terrain_drag(static, exp.root.run)
    from woof.static.terrain_autosmooth import verify_receipt, run_line
    smoothing = verify_receipt(
        geometry_receipt.get("terrain_autosmooth"), static["HGT_M"],
        domain_id=exp.root.grid_id)
    if smoothing is not None:
        print(run_line(smoothing))
    static_sha256 = _sha256(static_path)
    geometry_sha256 = _sha256(geometry_receipt_path)
    header_sha256 = (None if cache_header_path is None
                     else _sha256(cache_header_path))

    header_identity = header.get("identity")
    if not isinstance(header_identity, dict):
        raise ValueError("prepared cache identity must be an object")
    physical_head = physical_catalog = None
    if "ensemble_posted_physical_input" in (header_identity.get("source_identity") or {}):
        from woof.ingest.boundary_stream import read_head, verify_seal
        from woof.ensemble import physical_boundary
        if head is None:
            physical_stream_head = read_head(prepared_root, expected_sha256=_require_digest(
                proof.get("boundary_stream",{}).get("head_sha256"), "physical prepared head"))
            verify_seal(prepared_root,head=physical_stream_head)
        else:
            physical_stream_head = head
        physical_head = physical_stream_head["basis"].get(physical_boundary.KEY)
        physical_boundary.validate_head(physical_head,
            physical_stream_head["basis"]["cache"]["lbc"]["schedule"])
        # Completed catalogs were checked against every consumed segment and
        # ordinary source seal above; early heads admit only the initial knot.
        physical_catalog = (physical_boundary.head_catalog(physical_head) if head is not None
                            else header["metadata"]["user"][physical_boundary.KEY])
    source_identity = _validate_source_identity(
        source, header_identity.get("source_identity"),
        source_manifest_sha256, manifest_files, proof, layout=layout.kind,
        mapped_authority=mapped_authority,
        experiment_config=experiment_config,
        experiment_config_sha256=_sha256(experiment_config),
        physical_head=physical_head,
        ordinary_input_plan=None if as_posted is None else as_posted["input_plan"])
    if physical_head is not None:
        from woof.ensemble.posted_physical import validate_posted_native_source
        from woof.ensemble.physical_store import physical_static_identity
        physical_manifest = _physical_base_manifest_digest(source_manifest_sha256, manifest_files)
        validate_posted_native_source(source_identity["ensemble_posted_physical_input"],
            source_identity={"input_manifest_sha256": physical_manifest, **source_identity},
            input_manifest_authority=physical_manifest,
            input_plan=None if as_posted is None else as_posted["input_plan"],
            static_identity=physical_static_identity(static, native_landuse_identity(static)))
    if "ensemble_physical_input" in source_identity:
        from woof.ensemble.physical_store import physical_static_identity, validate_physical_input_binding
        validate_physical_input_binding(
            source_identity["ensemble_physical_input"], input_manifest_sha256=
            _physical_base_manifest_digest(source_manifest_sha256, manifest_files),
            source_identity=source_identity,
            static_identity=physical_static_identity(static, native_landuse_identity(static)))
    if layout.kind in _HIERARCHY_LAYOUTS \
            and source_identity.get("grid_id") != 1:
        raise ValueError("hierarchy prepared-cache source identity is not d01")
    if "ingest" in source_identity:
        from woof.ingest.soil_downscale import declared_soil_texture_downscale
        expected_ingest = {"soil_texture_downscale":
                           declared_soil_texture_downscale(experiment_config)}
        if source_identity["ingest"] != expected_ingest:
            raise ValueError("prepared cache ingest settings differ from experiment authority")
    from woof.static.highres_production import (
        load_static_highres, prepared_highres_settings_match,
        require_sealed_urban_legend)
    if source_identity.get("static_highres") is not None:
        # Named before the whole-carrier comparison below, which would
        # refuse the same mismatch without saying which setting moved.
        require_sealed_urban_legend(
            source_identity["static_highres"], exp.root.run,
            source="the prepared cache")
    requested_highres = load_static_highres(experiment_config)
    if ("static_highres" in source_identity
            or (requested_highres is not None and requested_highres.enabled)):
        if not prepared_highres_settings_match(
                source_identity.get("static_highres"), requested_highres):
            raise ValueError("prepared cache high-resolution settings differ from experiment authority")
    if "trace_gas_overrides" in source_identity:
        from woof.case_data import trace_gas_overrides_from_config
        if source_identity["trace_gas_overrides"] != trace_gas_overrides_from_config(
                experiment_config):
            raise ValueError("prepared cache trace-gas settings differ from experiment authority")
    if layout.kind == HRRR_DIRECT_LAYOUT:
        # Three of this identity's inputs are genuinely different values
        # on the HRRR route, where for the portable sources two of them
        # happen to coincide.  The bridge manifest is the DECODED
        # intermediate's SHA256SUMS, the source manifest is the raw
        # GRIB2 set's, and the namelist is WRF's namelist.input (the
        # native preparer reads the explicit eta ladder out of it), not
        # the experiment TOML.  Each is taken from the portable manifest
        # -- pinned by --source-manifest-sha256 -- and each is verified
        # against the file in the bundle before it is used.
        source_digest = manifest_files["source_manifest"]["sha256"]
        if as_posted is not None and manifest_files["bridge"]["sha256"] is None:
            # An as-posted head (A136 L7c (b)): the decoded bridge's
            # SHA256SUMS and the source manifest are written at its seal,
            # and its cache identity carries the plan's placeholder in both
            # (source_manifest_sha256 is that placeholder here,
            # _as_posted_head_binding).  The seal preflight checks the
            # bridge file against the sealed manifest, as here.
            bridge_digest = source_digest = source_manifest_sha256
        else:
            bridge_path = layout.authority_paths["bridge_manifest"]
            bridge_digest = _sha256(
                _require_file(bridge_path, "HRRR native bridge manifest"))
            if bridge_digest != manifest_files["bridge"]["sha256"]:
                raise ValueError(
                    "HRRR native bridge manifest differs from the portable "
                    "source manifest")
        namelist_path = _require_file(
            prepared_root / manifest_files["namelist_input"]["name"],
            "HRRR WRF namelist.input")
        if _sha256(namelist_path) != manifest_files["namelist_input"]["sha256"]:
            raise ValueError(
                "bundled HRRR namelist.input differs from the portable "
                "source manifest")
        expected_identity = prepared_cache_identity(
            bridge_manifest_sha256=bridge_digest,
            source_manifest_sha256=source_digest,
            static_cache_sha256=static_sha256,
            namelist_sha256=manifest_files["namelist_input"]["sha256"],
            domain_config=exp.root,
            forcing_hours=forcing_hours,
            source_identity=source_identity,
            namelist_extension_invariant=proof.get(
                "namelist_extension_invariant"),
        )
    else:
        expected_identity = prepared_cache_identity(
            bridge_manifest_sha256=source_manifest_sha256,
            source_manifest_sha256=source_manifest_sha256,
            static_cache_sha256=static_sha256,
            namelist_sha256=_sha256(experiment_config),
            domain_config=exp.root,
            forcing_hours=forcing_hours,
            source_identity=source_identity,
        )
    cache_identity, cache_identity_compatibility = (
        _resolve_cache_identity_compatibility(
            source=source, observed=header_identity,
            expected=expected_identity,
            not_in_use=undelayed_identity_defaults(exp)))
    if head is None:
        reader = PreparedCacheReader(
            prepared_cache_path, expected_identity=cache_identity)
        cache_payload = _retained_preflight_payload(
            reader, layout=layout.kind, head=head)
        readonly_reader = (reader if cache_payload is None
                           else cache_payload.read_only_reader())
        verified_cache = readonly_reader.verify_all()
        if verified_cache["content_sha256"] != prepared_content_sha256:
            raise ValueError(
                "verified prepared cache differs from the caller pin")
    else:
        reader = PreparedHeadReader(
            prepared_root, head, expected_identity=cache_identity)
        reader.verify_all()
        cache_payload = None
        readonly_reader = reader
    _validate_cache_metadata(
        reader, source=source, exp=exp, forcing_hours=forcing_hours,
        boundary_interval_seconds=boundary_interval_seconds, proof=proof,
        layout=layout.kind, physical_catalog=physical_catalog,
        boundary_publication=_boundary_publication(source, mapped_paths),
        prepared_root=prepared_root,
        posted_placeholder=(source_manifest_sha256
                            if as_posted is not None
                            and source in _MAPPED_SOURCES else None))
    # The cache restores its OWN c1f..c4h, so the etac beside them is what
    # the model will integrate.  Hold it to the one this run adopted, and
    # hold that one to this domain's own prepared columns.
    _coordinate_refusal = prepared_domain_coordinate_refusal(
        label=f"d{int(exp.root.grid_id):02d}", vertical=exp.vertical,
        coord_scalars=(reader.header.get("metadata") or {}).get(
            "coord_scalars") or {},
        base_arrays={"mub": readonly_reader.read_array("base/mub")}
        if "base/mub" in reader.arrays else {})
    if _coordinate_refusal is not None:
        raise ValueError(_coordinate_refusal)
    if source in _MAPPED_SOURCES:
        _validate_mapped_static_proof(
            proof, layout, source=source, static=static,
            geometry_receipt=geometry_receipt,
            static_sha256=static_sha256)

    contract_path = _require_file(
        REPO / "woof" / "wrf_direct_v461_contract.json",
        "direct-WRF contract")
    if head is not None:
        # The cache receipt, the artifact records and the export all bind
        # the SEALED cache; the seal re-runs this preflight to check them.
        export_source_receipt = None
    elif layout.kind in _DIRECT_LAYOUTS:
        # An HRRR bundle keeps its artifacts where the certified native
        # preparation has always written them, so the paths its proof
        # declares are that layout's, not the portable one's.  The
        # DIGESTS compared against them are identical either way -- the
        # relative path is a label, the sha256 is the binding.
        bundle_relative = (
            dict(HRRR_BUNDLE_PATHS) if layout.kind == HRRR_DIRECT_LAYOUT
            else {"static": "native-static.npz",
                  "geometry_receipt": "geometry-receipt.json",
                  "prepared_cache": "prepared-cache"})
        cache_proof = proof.get("prepared_cache")
        expected_cache_proof = {
            "schema": PREPARED_CACHE_SCHEMA,
            "status": "BUILT",
            "path": bundle_relative["prepared_cache"],
            "content_sha256": reader.content_sha256,
            "array_count": len(reader.arrays),
            "payload_bytes": reader.payload_bytes,
        }
        if cache_proof != expected_cache_proof:
            raise ValueError(
                "proof prepared-cache receipt differs from the bundle")
        if source not in _MAPPED_SOURCES:
            artifacts = proof.get("initialization_artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError(
                    "proof initialization artifact inventory is missing")
            _validate_artifact_record(
                artifacts.get("source_manifest"),
                expected_path="source-input-manifest.json",
                actual=source_manifest_path, label="source manifest")
            _validate_artifact_record(
                artifacts.get("static_cache"),
                expected_path=bundle_relative["static"],
                actual=static_path, label="static cache")
            _validate_artifact_record(
                artifacts.get("geometry_receipt"),
                expected_path=bundle_relative["geometry_receipt"],
                actual=geometry_receipt_path, label="geometry receipt")
            expected_artifact_cache = {
                "path": bundle_relative["prepared_cache"],
                "content_sha256": reader.content_sha256,
                "payload_bytes": reader.payload_bytes,
            }
            if artifacts.get("prepared_cache") != expected_artifact_cache:
                raise ValueError(
                    "proof prepared-cache artifact differs from the bundle")
            # The proof's own digest of its preprocessing receipt binds
            # the preparation, not the stock-WRF export, so it is checked
            # whether or not the bundle carries a READY export.  Inside
            # the READY branch a bundle for a scheme with no stock-WRF
            # contract skipped it.
            preprocessing_digest = hashlib.sha256(
                _canonical(proof.get("preprocessing")).encode("utf-8")
            ).hexdigest()
            if proof.get("preprocessing_receipt_sha256") \
                    != preprocessing_digest:
                raise ValueError(
                    "preparation proof preprocessing receipt hash differs")

        # The preprocessing receipt binds how the cache was built, so it
        # is checked whether or not the unchanged-WRF files were written.
        if source not in _MAPPED_SOURCES:
            preprocessing_digest = hashlib.sha256(
                _canonical(proof.get("preprocessing")).encode("utf-8")
            ).hexdigest()
            if proof.get("preprocessing_receipt_sha256") \
                    != preprocessing_digest:
                raise ValueError(
                    "preparation proof preprocessing receipt hash differs")
        export = proof.get("export")
        without_export = _optional_stock_wrf_export(proof, export)
        if without_export:
            export_source_receipt = MappingProxyType(dict(export))
        else:
            export = proof.get("export")
            expected_dimensions = {
                "nx": int(exp.root.run.nx),
                "ny": int(exp.root.run.ny),
                "nz": int(exp.root.run.nz),
            }
            expected_export_schema = (
                "gpuwm-native-direct-wrf-export-v3"
                if front_door_physics is not None
                else "gpuwm-native-direct-wrf-export-v2"
            )
            if (not isinstance(export, dict)
                    or export.get("schema") != expected_export_schema
                    or export.get("status") != "READY"
                    or export.get("forcing_hours") != list(forcing_hours)
                    or export.get("boundary_interval_seconds")
                    != boundary_interval_seconds
                    or export.get("dimensions") != expected_dimensions
                    or export.get("valid_time")
                    != exp.start_time.strftime("%Y-%m-%d_%H:%M:%S")):
                raise ValueError(
                    "proof direct-WRF export identity differs from the run")
            # One preparation writes the export and the proof, so the
            # export repeats the receipt the proof recorded byte for byte.
            # It is held to that RECORDED receipt, which the identity check
            # above resolved against this build, and not to this build's
            # recomputation, whose registry document digest moves with every
            # citation edit (A153).  A mapped proof's recorded receipt is
            # the export's own.
            recorded_physics = (export.get("physics")
                                if source in _MAPPED_SOURCES or source == "era5"
                                else proof.get("physics"))
            if expected_export_schema.endswith("-v3") \
                    and export.get("physics") != recorded_physics:
                raise ValueError(
                    f"{source.upper()} v3 export physics receipt differs from "
                    "the preparation proof")
            export_source = export.get("source")
            expected_export_source = {
                "contract_sha256": _sha256(contract_path),
                "geometry_receipt_sha256": geometry_sha256,
                "prepared_content_sha256": reader.content_sha256,
                "prepared_header_sha256": header_sha256,
                "resolved_physics_contract_sha256": (
                    _resolved_wrf_direct_contract_sha256(
                        exp.root.run.mp_physics)),
                "static_cache_sha256": static_sha256,
            }
            if export_source != expected_export_source:
                raise ValueError(
                    "proof export source hashes differ from preparation")
            export_source_receipt = MappingProxyType(expected_export_source)
    else:
        _validate_hierarchy_d01_artifacts(
            layout, static=static, geometry_receipt=geometry_receipt,
            reader=reader, static_sha256=static_sha256,
            geometry_sha256=geometry_sha256)
        if _optional_stock_wrf_export(proof, dict(layout.wrf_manifest)):
            export_source_receipt = layout.wrf_manifest
        else:
            export_source_receipt = _validate_hierarchy_wrf_authority(
                layout, source=source, source_exp=source_exp,
                forcing_hours=forcing_hours,
                boundary_interval_seconds=boundary_interval_seconds,
                source_manifest_sha256=source_manifest_sha256,
                decoder_sha256=(
                    mapped_authority["decoder_sha256"]
                    if source in _MAPPED_SOURCES else
                    manifest_files["bridge"]["sha256"]),
                preprocessing=proof.get("preprocessing"),
                contract_sha256=_sha256(contract_path),
                prepared_content_sha256=reader.content_sha256,
                prepared_header_sha256=header_sha256,
                static_sha256=static_sha256,
                geometry_sha256=geometry_sha256,
                mapped_authority=mapped_authority)

    authority_paths = MappingProxyType({
        **({"proof": proof_path} if head is None else {
            "prepared_head": prepared_root / "boundary-stream" / "head.json"}),
        # Not at an as-posted head: its seal writes the manifest.
        **({} if source_manifest_path is None
           else {"source_manifest": source_manifest_path}),
        "static": static_path,
        "geometry_receipt": geometry_receipt_path,
        **({"cache_header": cache_header_path} if head is None else {}),
        "experiment_config": experiment_config,
        "wps_namelist": wps_namelist,
        "wrf_direct_contract": contract_path,
        # Not a document an as-posted head's seal writes (the native
        # bridge's SHA256SUMS): the seal preflight hashes it.
        **{name: path for name, path in layout.authority_paths.items()
           if Path(path).resolve() not in _posted_documents(
               prepared_root, as_posted)},
        **dict(mapped_paths),
        **_thompson_authority_paths(physics_receipt),
    })
    file_sha256 = MappingProxyType({
        name: _sha256(path) for name, path in authority_paths.items()
    })
    if ((head is None and file_sha256["proof"] != proof_sha256)
            or ("source_manifest" in file_sha256
                and file_sha256["source_manifest"] != source_manifest_sha256)):
        raise RuntimeError("preparation authorities changed during preflight")
    # THE ACOUSTIC SUBSTEPS THIS DOMAIN'S OWN GROUND NEEDS, after every
    # identity comparison above (the count binds no prepared artifact) and
    # before anything sizes, plans or steps the domain, so the tile halo,
    # the adaptive clock's floor and the dycore all read one count.
    # THE LONG STEP its ground and crest-level wind allow comes with it,
    # read from the prepared start state and boundary data.
    # A forecast bound to a head reads only the head's boundary data here
    # (none), so the clock it steps on is checked again as each interval
    # arrives (run_prepared_forecast's clock guard).
    clock_basis: dict = {}
    exp = _terrain_derivations(
        exp, static, grid, readonly_reader, physics_receipt,
        basis_out=clock_basis if head is not None else None)
    if tiles is not None:
        # LAST, after every identity comparison above.  [tiles] is not a
        # domain field and could not move one of them, but an execution
        # control that is applied before the checks it cannot affect is
        # an execution control someone will later assume was checked.
        declared = getattr(exp, "tiles", None)
        if declared is not None and declared.enabled and declared != tiles:
            # Cannot happen on the HRRR route -- its published authority
            # has no [tiles] table to declare.  Said out loud anyway,
            # because a caller supplying the mode from outside while the
            # bundle carries its own is two authorities disagreeing about
            # how this forecast integrates, and picking one silently is
            # the failure this sentence exists to prevent.
            print(f"prepared forecast: --tiles mode = '{tiles.mode}' "
                  "replaces the [tiles] the hash-bound experiment "
                  f"declares (mode = '{declared.mode}'); the flag is the "
                  "later and more specific statement, and [tiles] binds "
                  "no identity either way", file=sys.stderr)
        exp = replace(exp, tiles=tiles)
    if devices_options is not None:
        exp = replace(exp, devices=devices_options)
    if devices is not None:
        from woof.core.devices import override_device_count
        exp = replace(exp, devices=override_device_count(exp.devices, devices))
    from woof.free_threading import refuse_multi_card_under_gil
    refuse_multi_card_under_gil(exp.devices.count, "prepared forecast")
    from woof.simulated_radar_config import apply_execution_options
    exp = apply_execution_options(exp, simulated_radar)
    from woof.da import forecast_heating
    exp = forecast_heating.apply_execution_options(exp, radar_heating)
    if forecast_heating.enabled(exp):
        # This door runs the root alone; a heated root under a tree would
        # leave its children unheated and their feedback over heated cells.
        forecast_heating.refuse_tree(
            len(source_exp.domains), "prepared single-domain forecast")
    if exp.simulated_radar.enabled:
        # Before the model is built or a card allocated, not at the first
        # history: a missing or stale rw_simradar, or a scan the host
        # memory cannot hold, refuses here.
        from woof.simulated_radar_config import require_admitted
        require_admitted(exp)
    return PreparedForecastInputs(
        source=source, layout=layout.kind, prepared_root=prepared_root,
        domain_bundle_path=layout.domain_bundle,
        proof_path=proof_path, source_manifest_path=source_manifest_path,
        static_path=static_path,
        geometry_receipt_path=geometry_receipt_path,
        prepared_cache_path=prepared_cache_path,
        experiment_config=experiment_config, wps_namelist=wps_namelist,
        proof=MappingProxyType(proof),
        source_manifest=MappingProxyType(manifest),
        geometry_receipt=MappingProxyType(geometry_receipt),
        cache_identity=cache_identity,
        cache_identity_compatibility=cache_identity_compatibility,
        cache_reader=reader,
        experiment=exp, grid=grid, static=MappingProxyType(static),
        landuse_identity=MappingProxyType(native_landuse_identity(static)),
        forcing_hours=forcing_hours,
        boundary_interval_seconds=boundary_interval_seconds,
        physics_receipt=MappingProxyType(
            physics_receipt if native_landuse_identity(static)
            == dict(_LANDUSE_IDENTITY) else
            {**physics_receipt,
             "landuse_identity": native_landuse_identity(static)}),
        export_source_receipt=export_source_receipt,
        source_manifest_receipt=(
            None if source_manifest_receipt is None
            else MappingProxyType(dict(source_manifest_receipt))),
        file_sha256=file_sha256, authority_paths=authority_paths,
        source_domain_count=len(source_exp.domains),
        source_member=source_member,
        stream_head=(None if head is None else MappingProxyType(head)),
        preflight_arguments=preflight_arguments,
        clock_basis=clock_basis.get("basis"),
        cache_payload=cache_payload,
    )


def _seal_streamed_inputs(inputs: PreparedForecastInputs, *,
                          stream=None) -> PreparedForecastInputs:
    """Wait for the preparation's seal and bind the forecast to it.

    The seal is checked against the head this forecast started from
    (:func:`woof.ingest.boundary_stream.verify_seal`: the header's
    content digest over the head's arrays plus every segment, the proof
    naming this head, every consumed segment unchanged), and then the
    complete preflight runs again on the sealed tree, so every check a
    sealed binding makes at launch has been made before the forecast
    publishes a receipt.  The two preflights must agree on the cache
    identity and on every authority both hashed.
    """

    head = getattr(inputs, "stream_head", None)
    if head is None:
        return inputs
    from woof.ingest.boundary_stream import StreamedIntervals, verify_seal

    if stream is None:
        stream = StreamedIntervals(inputs.prepared_root, head=head)
    if not stream.sealed():
        print("prepared forecast: waiting for the preparation to seal "
              f"({inputs.prepared_root})", file=sys.stderr, flush=True)
    stream.wait_sealed()
    sealed = verify_seal(inputs.prepared_root, head=head,
                         consumed=stream.consumed_markers())
    sealed_inputs = preflight_prepared_forecast(
        **_sealed_arguments(inputs, sealed))
    # The cache identity may change only where an as-posted head's seal
    # writes the manifest digest; every authority both preflights hashed
    # must be unchanged (an as-posted head hashes no manifest).
    _require_sealed_identity(inputs, sealed_inputs, head)
    shared = set(inputs.file_sha256) & set(sealed_inputs.file_sha256)
    if (sealed_inputs.forcing_hours != inputs.forcing_hours
            or {key: inputs.file_sha256[key] for key in shared}
            != {key: sealed_inputs.file_sha256[key] for key in shared}):
        raise RuntimeError(
            "the sealed preparation differs from the head this forecast "
            "started from")
    from woof.ingest.boundary_stream import derived_clock

    if (derived_clock(sealed_inputs.physics_receipt.get("terrain_clock"))
            != derived_clock(inputs.physics_receipt.get("terrain_clock"))):
        # Every interval passed the clock guard as it loaded, so this is
        # a seal whose boundaries are not the ones the forecast read.
        raise RuntimeError(
            "the sealed preparation derives a different terrain clock than "
            "the one this forecast stepped on")
    return replace(sealed_inputs, stream_head=head)


def _verify_inputs_unchanged(inputs: PreparedForecastInputs) -> None:
    _verify_thompson_runtime_environment(inputs.physics_receipt)
    # Each authority file hashed on its own core (one SHA-256 stream per
    # file); the same digests the serial walk took.
    from concurrent.futures import ThreadPoolExecutor

    items = list(inputs.authority_paths.items())
    with ThreadPoolExecutor(max_workers=max(1, min(len(items) or 1, 16)),
                            thread_name_prefix="verify-authority") as pool:
        current = dict(zip((name for name, _ in items),
                           pool.map(lambda item: _sha256(item[1]), items)))
    if current != dict(inputs.file_sha256):
        raise RuntimeError("prepared forecast inputs changed during execution")
    verified = PreparedCacheReader(
        inputs.prepared_cache_path,
        expected_identity=inputs.cache_identity).verify_all()
    if verified["content_sha256"] != inputs.cache_reader.content_sha256:
        raise RuntimeError("prepared cache changed during execution")


def _consume_due_native_refl_10cm(state, ticks: int, consumer, *,
                                  domain_start_ticks: int = 0, shape=None):
    """Consume the scheme-native field staged by an output-due MP call.

    ``domain_start_ticks`` is the DOMAIN's own start tick, not the
    experiment's: the frame due there is the analysis frame, which
    precedes every step of that domain and therefore every stash.  It is
    0 on this runner -- a single domain starts with its experiment -- and
    the parameter exists so the shared predicate
    (:func:`woof.core.refl.refl_10cm_stash_is_due`) is asked the same
    question here as at the tree seam, where a nest that activates later
    made the absolute-0 reading kill the run at its first frame (#205).
    """
    from woof.core.refl import refl_10cm_stash_is_due

    if (state.qv is not None
            and state.physics.mp_physics in REFL_10CM_MICROPHYSICS):
        if refl_10cm_stash_is_due(ticks,
                                  domain_start_ticks=domain_start_ticks):
            return consumer(state)
        from woof.core.gsd_initial_reflectivity import enabled as gsd_initial_enabled
        if domain_start_ticks == 0 or gsd_initial_enabled(
                getattr(state.physics, "refl_config", None)):
            # HRRR's helper also owns an activating domain's first frame.
            from woof.core.refl import analysis_refl_10cm
            return analysis_refl_10cm(state, shape=shape)
    return None


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ModuleNotFoundError:
        if os.name != "nt":
            return 0
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.PeakWorkingSetSize)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if sys.platform == "darwin" else 1024
    return int(usage.ru_maxrss * scale)


class _CheckpointPublisher:
    """Makes snapshotted checkpoints durable on one thread of its own.

    THE BREAKAGE THIS REMOVES, measured on the 1 km CONUS run of
    2026-10-04 (2000 x 1500 x 50, 8 x RTX 5090, hourly restarts): every
    hourly checkpoint stopped the model for 31 to 37 s, about 18% of its
    195 s per forecast hour -- about 12 s waiting for that hour's history
    frame to become durable, then 18 to 21 s of ``np.savez`` and ``fsync``
    of a 25 GB archive.  Neither needs the model to stand still.  The
    snapshot is still taken between two steps (``write_tree_restart(...,
    defer_publish=True)``); waiting for the history frames queued before
    it, writing it, and the bookkeeping after it happen here.

    One checkpoint in flight at most: :meth:`submit` first waits for the
    previous one.  A failed write is raised on the stepping thread at the
    next checkpoint or at :meth:`wait`, which the runner calls before it
    reports success, so a run never claims a checkpoint it does not have.

    ``then`` (the generation's ``retire_superseded``) runs in :meth:`wait`,
    on the thread that waits -- the stepping thread -- and only after a
    successful write.  The job publishes with ``retire=False``: a set
    retired from this thread could be the one the stepping thread had just
    handed its progress callback, which reads it back in full.  The breakage
    (ml-probs T1, 2026-10-27): every 60 h GFS 12 km parent with hourly
    checkpoints and ``--keep-checkpoints 1`` died 3-7 h in with
    FileNotFoundError on the set the heartbeat was validating.
    """

    def __init__(self):
        self._thread = None
        self._failure = None
        self._then = None

    def submit(self, job, *, then=None) -> None:
        self.wait()

        def run():
            try:
                job()
            except BaseException as exc:  # noqa: BLE001 - re-raised by wait()
                self._failure = exc

        self._then = then
        self._thread = threading.Thread(
            target=run, name="gpuwm-checkpoint-writer", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()
        failure, self._failure = self._failure, None
        then, self._then = self._then, None
        if failure is not None:
            raise RuntimeError(
                "the checkpoint writer failed: "
                f"{type(failure).__name__}: {failure}") from failure
        if then is not None:
            then()


class _LandingObservers:
    """Fan the wrfout landing hook out to more than one consumer.

    ``PerDomainWrfoutWriters.attach_progress_callback`` reads ONE
    ``output_committed`` attribute off what it is handed and assigns it
    to every writer's ``landing_observer``, unconditionally -- so a
    second consumer cannot be added by calling attach twice.  The
    second call would overwrite the first, silently, and the run would
    keep publishing frames with the earlier consumer no longer watching.

    That matters here because this runner now has two: the observer of a
    caller hosting it in-process (``woof run-plan``), and its OWN early
    render when the command line asked for one.  Both are given the
    frame; neither can suppress the other; and a consumer that raises
    does not take the writer thread or its sibling down with it, which
    is the same contract the writer keeps for a single one.
    """

    def __init__(self, *sinks, on_submit=None):
        self._sinks = tuple(sink for sink in sinks if sink is not None)
        self._on_submit = on_submit

    def __bool__(self) -> bool:
        return bool(self._sinks)

    def frame_submitted(self, **event) -> None:
        """The per-step log's latency anchor (StepLog.frame_submitted)."""
        from woof.progress_log import LandingFanout
        LandingFanout(*self._sinks).frame_submitted(**event)
        if (self._on_submit is not None and
                getattr(self._on_submit, "__self__", None) not in
                [getattr(sink, "__self__", None) for sink in self._sinks]):
            try:
                self._on_submit(**event)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass

    def output_committed(self, **event) -> None:
        for sink in self._sinks:
            try:
                sink(**event)
            except Exception:  # noqa: BLE001 - telemetry never fails a run
                pass


# ---------------------------------------------------------------------------
# which initialization road a STREAMED forecast takes
# ---------------------------------------------------------------------------
#
# A streamed forecast integrates out of a pinned host store and never reads
# the resident ``DomainState`` again: ``woof.core.streaming.attach`` copies
# it carrier by carrier into that store and every sweep afterwards writes only
# the store.  Building the state first is nevertheless what this route has
# always done, and it is the CEILING.  MEASURED on a 16 GB card at nz = 49,
# the bare state costs 11 276.5 B per column and the prepared case -- state
# plus physics plus geography plus boundary tables -- about 15 780, so
# 1024 x 1024 is refused inside ``initialize_physics`` while the streamed
# forecast that state would have fed needs about 6 GiB and runs comfortably.
# The streamed engine on the same card has stepped 2624 x 2624.
#
# ``woof.ingest.prepared_store.store_from_prepared_cache`` fills the same
# store one ROW SLAB at a time -- slab-height state, physics attached to THAT,
# scattered into full-domain pinned host arrays -- so no domain-shaped device
# array is ever allocated and the domain the card can carry stops being a
# property of the card.  ``--stream-init`` chooses between the two roads.
#
# ``auto`` prefers the RESIDENT road wherever it comfortably fits, and that
# preference is deliberate rather than conservative: the resident road is the
# one every existing receipt was written by, and the two roads' agreement is
# proven by running a domain small enough for both and comparing frames.  The
# store-direct road is taken when the priced resident state does not fit,
# which is precisely the case the resident road cannot serve at all.

#: MEASURED bytes per column at nz = 49 on the lighter reference configuration,
#: not a bound for every physics configuration. Keep this measurement as a
#: floor; the configuration-aware preflight inventory below can raise it.
#: (the streamed-ceiling measurement record, section 2): a bare
#: ``DomainState`` costs 11 276.5 B/column, and the prepared case -- the same
#: state plus everything ``initialize_prepared_physics`` puts on the card
#: beside it (the physics carriers, the geography inventory and the lateral
#: boundary tables) -- about 15 780 B/column.
_MEASURED_BARE_STATE_BYTES_PER_COLUMN = 11276.5
_MEASURED_PREPARED_BYTES_PER_COLUMN = 15780.0

#: The ``nz`` both per-column measurements were taken at.  They are scaled
#: LINEARLY in ``nz`` off this reference, which is the accurate reading of one
#: measurement at one height: the prepared case is overwhelmingly 3-D arrays,
#: and the 2-D surface inventory that does not scale is a small enough share
#: that pretending it does costs a slight OVER-price below nz = 49 and a
#: slight under-price above it -- both inside the fit fraction's margin.
_MEASURED_BYTES_PER_COLUMN_NZ = 49

#: What the prepared case costs as a multiple of the state the cache carries
#: (1.40).  DERIVED from the two measurements above rather than written as a
#: bare 1.4, so that a later measurement moves the number and the evidence for
#: it in one edit and neither can drift away from the other.
#:
#: THIS RATIO IS NOT THE PRICE OF THE RESIDENT ROAD, and reading it as one is
#: the defect the pricing below was rebuilt to remove.  It relates the prepared
#: case to a BARE ``DomainState``; the cache's ``state/*`` manifest is not a
#: bare DomainState but the far smaller ``STATE_SERIALIZED_ATTRS`` subset --
#: the dynamics and microphysics prognostics only, with none of the RK and
#: dycore working arrays a ``DomainState`` allocates beside them and none of
#: what ``initialize_prepared_physics`` then puts on the card.  MEASURED on the
#: 1024 x 1024 x 49 gate cache: ``state/*`` is 4 945 485 824 B, which is
#: 4 716.4 B per column against the bare state's 11 276.5 -- the manifest
#: undercounts the state alone by 2.39x, and ``manifest x 1.40`` therefore
#: priced the whole prepared case at 6.45 GiB when the resident road went on to
#: die with 16 276 726 272 B allocated.
_PREPARED_RESIDENT_HEADROOM = (_MEASURED_PREPARED_BYTES_PER_COLUMN
                               / _MEASURED_BARE_STATE_BYTES_PER_COLUMN)

#: How much of the card's FREE memory the streamed run's whole PEAK may claim
#: before ``auto`` stops calling the fit comfortable.  A margin for what is
#: left unpriced -- the NVRTC module images and the allocator's own
#: fragmentation -- and no longer a stand-in for the tile buffers themselves.
#:
#: IT USED TO BE THAT STAND-IN, AND THAT WAS THE DEFECT.  The comment here
#: read "20% of a 16 GB card is 3.2 GiB, which covers the default buffer set
#: with room to spare", and on a 16 GB card it did.  But the tile buffers and
#: the radiation transient are ABSOLUTE costs and a fraction is a RELATIVE
#: one, so the margin shrank exactly where the card had least room to give:
#: 20% of a 10 GiB card is 1.9 GiB against a default buffer set of 5.13 GiB
#: and a 2.74 GiB radiation transient.  A 424 x 424 x 49 forecast therefore
#: priced its 2.64 GiB resident state as a comfortable fit, took the resident
#: road, and peaked at 10.51 GiB on 9.50 GiB of free memory.  Under WDDM that
#: over-commit does not raise -- the driver pages it -- so the forecast did
#: not refuse, it just stopped completing steps.
#:
#: Both of those costs are now priced explicitly (:func:`_streamed_peak_bytes`)
#: and added to the resident state before this fraction is applied, so what is
#: left for the fraction to cover is only the genuinely unpriced remainder.
_AUTO_RESIDENT_FIT_FRACTION = 0.80

#: ``--stream-init``'s values.  ``auto`` is the default and is the only one
#: that decides anything; the other two force a road so that a gate can run
#: both over the same domain and compare.
STREAM_INIT_CHOICES = ("auto", "resident", "store")

#: The schema/status ``store_from_prepared_cache`` stamps on its receipt.
#: Pinned here for the same reason ``_validate_restored_cache_receipt`` pins
#: ``PREPARED_CACHE_SCHEMA``: the receipt is re-checked against the caller's
#: content digest, and a receipt of some other shape must not be able to
#: satisfy that check by carrying the right key by accident.
_PREPARED_STORE_SCHEMA = "gpuwm-prepared-store-v1"

_GIB = float(1 << 30)


def _prepared_state_manifest_bytes(reader) -> int:
    """The EXACT device cost of restoring a prepared cache's prognostics.

    ``restore_prepared_cache`` allocates one ``DomainState`` and then runs
    ``target[...] = cp.asarray(host)`` once per ``state/*`` array, so the
    device bytes that restore takes for the trajectory are the sum of those
    arrays' own ``nbytes`` -- a number the cache HEADER already carries per
    array, and one the reader already checked against that array's ``shape``
    and ``dtype`` when it opened the bundle.

    Read, never probed.  The question is asked before anything has been
    allocated, and an answer obtained by trying the allocation would be the
    allocation it exists to avoid.
    """

    return int(sum(int(spec["nbytes"])
                   for key, spec in reader.arrays.items()
                   if str(key).startswith("state/")))


def _streamed_peak_bytes(cfg, decision) -> int | None:
    """What the STREAMED half of this forecast puts on the card, at its peak.

    The resident road does not end at the resident state.  A forecast that
    takes it and then streams allocates the tile buffers ON TOP of that state
    -- ``attach`` copies out of the state and nothing frees it
    (``woof.core.streaming.refresh_from_store``) -- and meets the radiation
    transient on top of THAT.  The card has to hold all three at once, and
    this is the second and third terms.

    Two numbers, each already measured and already carried by this project;
    neither was being read by the road decision:

    * ``streamed_envelope(...).vram_bytes`` -- ``tilestream.autoplan``'s
      footprint for the tiling this decision actually selected: the CUDA
      context, the per-process fixed cost and ``nbuffers`` compute windows.
    * ``footprint_for(cfg).radiation_transient_bytes`` -- the RRTMGP call's
      per-process working set, MEASURED at +2.74 GiB over steady state and
      zero on the rungs that run no radiation.  It is allocated and freed
      inside one radiation step, so it never appears in a steady-state
      reading; it is nevertheless on the card at the instant it matters, and
      the first radiation call is ``itimestep == 1``.

    ``None``, never zero, when this cannot be priced -- a run that does not
    stream, or a decision this function was handed without a tiling on it.
    Zero is the answer that reinstates the defect, so the caller is told the
    term is missing and decides for itself rather than being handed a
    confident nought.
    """

    if cfg is None or decision is None or not getattr(decision, "stream", False):
        return None
    try:
        from woof.core import streaming

        envelope = streaming.streamed_envelope(cfg, None, decision=decision)
        if envelope is None:
            return None
        # The envelope carries BOTH terms since the estimate surfaces
        # learned the reservation, so this reads its peak rather than
        # re-adding the transient beside it -- one place the sum lives.
        return int(envelope.peak_vram_bytes)
    except Exception:
        # The road decision must still be TAKEN.  An unpriceable streamed term
        # leaves the pre-existing resident-only rule standing, which is the
        # behaviour this route had before the term existed, rather than
        # refusing a forecast because an estimate could not be formed.
        return None


def _stream_init_pricing(state_bytes: int, free_bytes: int, total_bytes: int,
                         *, columns: int | None = None,
                         nz: int | None = None,
                         streamed_bytes: int | None = None,
                         cfg=None, resident_estimate=None,
                         forcing_intervals: int | None = None,
                         boundary_species=(),
                         ) -> dict[str, object]:
    """Price the resident road against the card, with no device work at all.

    Separated from the device query so the RULE is testable on a machine with
    no GPU: everything below is arithmetic over a handful of integers, and
    :func:`_free_device_bytes` is the only part that needs a card.

    The largest of the manifest, measured reference and configuration-aware
    resident inventories wins, because each sees costs the others miss.

    ``manifest`` -- the cache's own ``state/*`` bytes times the physics
    headroom -- is the only term that knows what THIS cache holds, so a
    configuration carrying far more prognostics than the one measured below
    (a bigger microphysics scheme, more tracers) raises the price through it.
    On its own it is not a price for the resident road at all: the manifest is
    the SERIALIZED subset, not a ``DomainState``, and it undercounted the
    1024 x 1024 gate case by 2.38x -- 6.45 GiB priced, 15.16 GiB allocated at
    the refusal.  That is the number ``auto`` used to say ``fits: true`` on and
    then die inside ``initialize_physics``.

    ``measured`` -- ``_MEASURED_PREPARED_BYTES_PER_COLUMN`` over this domain's
    columns, scaled by ``nz`` -- records the lighter reference configuration's
    state plus everything ``initialize_prepared_physics`` built beside it.
    It is a measurement, and it
    predicts the refusal it was checked against to 0.4%: 15.41 GiB priced
    against 15.35 GiB at the out-of-memory throw.

    ``configuration`` -- preflight's state, physics and lateral boundary
    tables for this cfg. The 1799 x 1059 x 50 HRRR recipe OOM on 2026-10-21
    showed the lighter reference measurement underpricing these holdings.
    Reuse the door's resident estimate when available, otherwise the same
    preflight domain estimator. Do not add its workspace/step-transient
    envelope: the streamed term already prices the tiles and radiation.

    ``max`` rather than a sum: these estimates overlap -- all are
    dominated by the same prognostics -- so adding them would double-count the
    state and refuse domains that fit.  ``columns``/``nz`` absent leaves the
    manifest term alone when no configuration is supplied. The forecast door
    supplies both the extents and the configuration.
    """

    manifest_priced = int(round(float(state_bytes)
                                * _PREPARED_RESIDENT_HEADROOM))
    measured_priced = None
    if columns and nz:
        measured_priced = int(round(
            _MEASURED_PREPARED_BYTES_PER_COLUMN * float(columns)
            * float(nz) / float(_MEASURED_BYTES_PER_COLUMN_NZ)))
    priced = (manifest_priced if measured_priced is None
              else max(manifest_priced, measured_priced))
    priced_from = ("manifest" if measured_priced is None
                   or manifest_priced >= measured_priced
                   else "measured-per-column")
    configuration_terms = None
    if resident_estimate is not None or cfg is not None:
        if resident_estimate is not None:
            domains = resident_estimate.domains
        else:
            from woof.core import preflight
            from woof.experiment import experiment_from_run_config

            domain = experiment_from_run_config(cfg, datetime(2000, 1, 1)).root
            domains = (preflight.estimate_domain(
                domain, n_lbc_intervals=preflight.lbc_intervals(
                    cfg.run_seconds, preflight.DEFAULT_FORCING_INTERVAL_SECONDS,
                    retained_intervals=forcing_intervals),
                boundary_species=boundary_species),)
        configuration_terms = {
            label: sum(int(domain.category_bytes(category)) for domain in domains)
            for label, category in (("model state", "state"),
                                    ("physics", "physics"),
                                    ("lateral boundary tables", "lbc"))}
        configuration_priced = sum(configuration_terms.values())
        if configuration_priced > priced:
            priced, priced_from = configuration_priced, "configuration"
    allowance = int(float(free_bytes) * _AUTO_RESIDENT_FIT_FRACTION)
    # THE PEAK, not the state. ``max`` is right for the resident estimates
    # because they price the same bytes two ways; this is a SUM because the
    # tiles and the radiation transient are different bytes on the same card
    # at the same instant as the state.  ``None`` leaves the pre-existing
    # resident-only rule exactly as it was.
    peak = priced if streamed_bytes is None else priced + int(streamed_bytes)
    return {
        "state_manifest_bytes": int(state_bytes),
        "physics_headroom_factor": round(_PREPARED_RESIDENT_HEADROOM, 4),
        "manifest_priced_bytes": manifest_priced,
        "measured_priced_bytes": measured_priced,
        "configuration_resident_terms": configuration_terms,
        "configuration_priced_bytes": (None if configuration_terms is None
                                       else sum(configuration_terms.values())),
        "priced_from": priced_from,
        "columns": None if not columns else int(columns),
        "nz": None if not nz else int(nz),
        "priced_resident_bytes": priced,
        "streamed_peak_bytes": (None if streamed_bytes is None
                                else int(streamed_bytes)),
        "resident_peak_bytes": int(peak),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "fit_fraction": _AUTO_RESIDENT_FIT_FRACTION,
        "fit_allowance_bytes": allowance,
        "fits": bool(peak <= allowance),
    }


def _free_device_bytes() -> tuple[int, int]:
    """``(free, total)`` device memory on this card right now.

    Not the planner's budget and not a nameplate capacity: the question the
    pricing asks is whether this allocation would succeed on this card in this
    process, and the only accurate source for that is what is free at the
    instant the decision is taken.

    Read through the same function the tiling planner reads free VRAM with
    (:func:`woof.core.preflight.device_free_and_total_bytes`).  This used
    ``cupy.cuda.Device().mem_info`` alone, which under Windows WDDM counts
    memory held by other processes as free: one streamed run printed
    "8.88 GiB free of 10.00 GiB on the card" while its own planner had
    measured 3.99 to 4.32 GiB, and the resident-road fit was priced against
    the larger figure.
    """

    from woof.core.preflight import device_free_and_total_bytes

    return device_free_and_total_bytes()


def _retained_interval_count(reader) -> int | None:
    """How many lateral boundary intervals this prepared cache retains.

    ``None`` when the reader carries no readable inventory, and the resident
    admission then prices the configured schedule instead.
    """
    try:
        count = len(reader.header["metadata"]["lbc"]["intervals"])
    except (AttributeError, KeyError, TypeError):
        return None
    return count or None


def single_urban_columns(inputs) -> dict[int, int] | None:
    """Price a prepared domain's BEM workspace from its cold-start statics."""
    from woof.core.urban_state import (bem_workspace_counted,
                                        prepared_urban_columns)
    from woof.static.corridor import relocating_subtree_grid_ids

    exp = inputs.experiment
    gid = int(exp.root.grid_id)
    if (not bem_workspace_counted(exp.root.run)
            or gid in relocating_subtree_grid_ids(exp)):
        return None
    identity = getattr(inputs, "landuse_identity", None) or {}
    if identity.get("MMINLU") is None:
        return None
    count = prepared_urban_columns(
        exp.root.run, inputs.static.get("LU_INDEX"),
        landuse_dataset=str(identity["MMINLU"]),
        frc_urb2d=inputs.static.get("FRC_URB2D"))
    return None if count is None else {gid: count}


def _priced_boundary_source(reader, source):
    """What a door prices the root's boundary tables from, before the restore.

    The hydrometeor masses the cache it is about to restore carries
    (:func:`woof.boundary_fields.sealed_boundary_species`), read off the
    same retained interval inventory as :func:`_retained_interval_count`;
    ``source``'s own row when the reader carries no readable inventory.
    Priced by the source NAME, a forecast from a user's own mapping
    (``--source mapped``, whose row publishes nothing) was weighed on water
    vapour alone while its cache held every mass the mapping declares, and
    a HRRR cache sealed before its boundary carried them was weighed on
    tables it does not hold.
    """
    from woof.boundary_fields import sealed_boundary_species

    try:
        intervals = reader.header["metadata"]["lbc"]["intervals"]
    except (AttributeError, KeyError, TypeError):
        intervals = None
    carried = sealed_boundary_species(intervals)
    return source if carried is None else carried


def _devices_init_road(exp):
    """A ranked forecast never restores the whole domain onto a card."""
    if not getattr(getattr(exp, "devices", None), "enabled", False):
        return None
    return "store", {"road": "store", "why": "[devices] forces STORE-DIRECT: "
                     "the whole domain must never be resident on a card"}


def _stream_producer_reserve(stream_head, device=0, *, identity=None):
    """The next preparation batch this physical card must still hold.

    An old head's ordinal is not a physical identity: CUDA visibility may
    differ between the producer and consumer. Without both physical
    readings every consumer card retains the bound conservatively.
    """
    decision = (stream_head or {}).get("decision") or {}
    reserve = decision.get("producer_reserve_bytes")
    if reserve is None:
        return 0
    producer = decision.get("producer_device_identity")
    if isinstance(producer, Mapping) and isinstance(identity, Mapping):
        for key in ("uuid", "pci_bus_id"):
            if producer.get(key) and identity.get(key):
                return 0 if producer[key] != identity[key] else max(0, int(reserve))
    return max(0, int(reserve))


def _devices_stream_budgets(budgets, stream_head, *, identities=None):
    """Leave a chunked producer's next batch free on its own card."""
    identities = identities or {}
    reserves = {dev: _stream_producer_reserve(
        stream_head, dev, identity=identities.get(dev)) for dev in budgets}
    if not any(reserves.values()):
        return budgets
    return {dev: max(0, int(free) - reserves[dev])
            for dev, free in budgets.items()}


def _stream_host_producer_reserve(stream_head):
    """Additional host bytes the producer may need beyond its head RSS."""
    decision = (stream_head or {}).get("decision") or {}
    return max(0, int((decision.get("host") or {}).get("producer_host_bytes") or 0))


def _stream_host_budget(available, stream_head):
    return (None if available is None else
            max(0, int(available) - _stream_host_producer_reserve(stream_head)))


def _stream_reserved_machine(machine, stream_head, *, device=0, identity=None):
    """Forecast budgets with room for the producer's next device/host peak."""
    if machine is None:
        return None
    free = int(machine.vram_bytes)
    reserved = _devices_stream_budgets(
        {device: free}, stream_head, identities={device: identity})[device]
    host = int(machine.host_bytes)
    if _stream_host_producer_reserve(stream_head):
        from woof.core.preflight import host_available_bytes

        available = host_available_bytes()
        if available is not None:
            host = min(host, int(available))
        host = _stream_host_budget(host, stream_head)
    return (machine if free == reserved and host == int(machine.host_bytes) else
            replace(machine, vram_bytes=reserved, host_bytes=host))


def _stream_reserved_options(options, machine, stream_head, *, device=0, identity=None):
    """Explicit tile budgets cannot spend the producer's reservations."""
    if options is None or machine is None:
        return options
    changes = {}
    if (options.vram_budget_bytes is not None
            and _stream_producer_reserve(stream_head, device, identity=identity)):
        changes["vram_budget_bytes"] = min(
            int(options.vram_budget_bytes), int(machine.vram_bytes))
    if options.host_budget_bytes is not None and _stream_host_producer_reserve(stream_head):
        changes["host_budget_bytes"] = min(
            int(options.host_budget_bytes), int(machine.host_bytes))
    return replace(options, **changes) if changes else options


def _stream_reserved_experiment(exp, machine, stream_head, *, device=0, identity=None):
    """Cap only the tree planner's local copy of configured tile budgets."""
    if machine is None or not (
            _stream_producer_reserve(stream_head, device, identity=identity)
            or _stream_host_producer_reserve(stream_head)):
        return exp
    return replace(exp, tiles=_stream_reserved_options(
        exp.tiles, machine, stream_head, device=device, identity=identity), domains=tuple(
            replace(domain, tiles=_stream_reserved_options(
                domain.tiles, machine, stream_head, device=device, identity=identity))
            for domain in exp.domains))


def _devices_lake_mask(cfg, *, geography=None, inventory=None, landuse=None):
    """An upper bound on where this run's ``lakemask`` can be 1, or None.

    The [devices] admission prices each rank's sparse CLM lake arrays at the
    lake-eligible columns of its window
    (:func:`woof.core.devices_memory.rank_lake_column_bounds`, open-defects
    ledger A13).  After the store loader the store carries the run's own
    ``driver/fields/lakemask`` and that is exact.  Before it, the union of
    the two ways a column becomes a lake bounds it from above: the land
    cover's lake category (``ISLAKE`` of ``landuse``, which
    ``woof.core.landuse`` records as LAKEMASK), plus any LAKEMASK record;
    and lakeini's flag-0 branch, water or sea-ice columns (all LANDMASK 0
    / XLAND 2) at or above ``lake_min_elev``.  Without terrain every water
    column counts; without the lake category, every water column counts.
    None, and the every-column price, when no land mask is readable.
    """
    import numpy as np
    if int(getattr(cfg, "sf_lake_physics", 0) or 0) != 1:
        return None
    shape = (int(cfg.ny), int(cfg.nx))

    def plane(source, *names):
        for name in names:
            value = (source or {}).get(name) if hasattr(source, "get") else None
            if value is None:
                continue
            value = np.asarray(value)
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.shape == shape:
                return value
        return None

    exact = plane(inventory, *[key for key in (inventory or {})
                               if str(key).endswith("fields/lakemask")])
    if exact is not None:
        return exact == 1
    landmask = plane(geography, "LANDMASK", "landmask")
    xland = plane(geography, "XLAND", "xland")
    if landmask is None and xland is None:
        return None
    water = (landmask == 0) if landmask is not None else (xland > 1.5)
    terrain = plane(geography, "HGT_M", "HGT", "ht")
    lu_index = plane(geography, "LU_INDEX", "lu_index")
    islake = (landuse or {}).get("ISLAKE") if hasattr(landuse, "get") else None
    if terrain is None or lu_index is None or islake is None:
        bound = water
    else:
        bound = ((water & (terrain >= np.float32(cfg.lake_min_elev)))
                 | (np.rint(lu_index) == int(islake)))
    lakes = plane(geography, "LAKEMASK", "lakemask")
    return bound if lakes is None else (bound | (lakes == 1))


def _admit_devices_forecast(exp, cfg, *, geography, forcing_intervals,
                            inventory=None, source=None, template=None,
                            boundaries=None, stream_head=None, landuse=None,
                            static=None):
    from woof.core.adaptive_clock import maximum_map_factor
    from woof.core.devices import validate_device_count, DevicesRefused
    from woof.core.preflight import estimate_devices, host_available_bytes
    from woof.core.devices_memory import (devices_gate, priced_template_live_bytes,
                                           priced_host_live_bytes)
    from woof.core.resident_admission import device_free_bytes
    import cupy as cp
    validate_device_count(exp.devices, cp.cuda.runtime.getDeviceCount())
    budgets = {}
    budget_terms = {}
    profiles = {}
    identities = {}
    from woof.core.device_probe import cuda_device_identity
    from woof.core.device_inventory import DeviceLocalMemoryProfile
    from woof.core.preflight import read_compile_platform, device_free_and_total_bytes
    for dev in set(exp.devices.device_ids()):
        with cp.cuda.Device(dev):
            free, _total = device_free_and_total_bytes(dev)
            reusable = int(cp.get_default_memory_pool().free_bytes())
            available = device_free_bytes()
            budgets[dev] = min(free + reusable, available) if available is not None else free
            budget_terms[dev] = {"device_free_bytes": free,
                                 "reusable_pool_bytes": reusable,
                                 "template_live_credit_bytes": 0}
            identities[dev] = cuda_device_identity(dev)
            props = cp.cuda.runtime.getDeviceProperties(dev)
            name = props["name"]
            profiles[dev] = DeviceLocalMemoryProfile(
                name=name.decode() if isinstance(name, bytes) else str(name),
                multiprocessor_count=int(props["multiProcessorCount"]),
                max_threads_per_multiprocessor=int(props["maxThreadsPerMultiProcessor"]),
                default_stack_limit_bytes=int(cp.cuda.runtime.deviceGetLimit(0)),
                compile_platform=read_compile_platform())
    # Leave a chunked producer's next batch free on its own card before the
    # rank widths are fitted, so the fitted widths respect that reserve.
    budgets = _devices_stream_budgets(budgets, stream_head, identities=identities)
    # [radar_heating] keeps each rank's slab slots and builds every slot on
    # the first card: priced from the same rank plan the run will build, and
    # taken out of each card's budget before the gate, as the cycle's
    # admission prices its forcing slots (woof.da.cycle_admission).
    from woof.da.forecast_heating import enabled as _heating_enabled
    if _heating_enabled(exp):
        from woof.da.forecast_heating import ranked_reservation
        reserve = ranked_reservation(
            exp.radar_heating, cfg, exp.devices,
            max_map_factor=maximum_map_factor(geography=geography))
        for dev in budgets:
            budget_terms[dev]["radar_heating_reserve_bytes"] = int(reserve.get(dev, 0))
        budgets = {dev: max(0, int(free) - int(reserve.get(dev, 0)))
                   for dev, free in budgets.items()}
    def price():
        return estimate_devices(
            replace(exp, domains=(replace(exp.root, run=cfg),)),
            max_map_factor=maximum_map_factor(geography=geography),
            forcing_intervals=forcing_intervals, inventory=inventory,
            geography=geography, source=source, profiles=profiles, budgets=budgets,
            streaming_boundaries=stream_head is not None,
            # The loader's geography carries setup arrays, not the land
            # cover; the lake bound reads the prepared statics either way.
            lake_mask=_devices_lake_mask(
                cfg, geography=geography if static is None else static,
                inventory=inventory, landuse=landuse))
    estimate = price()
    if template is not None:
        # The second admission runs after the store loader. Its retained
        # template is already excluded from free memory and already included
        # in the price. Credit only its exact priced array spans, not every
        # allocation the process happens to hold.
        first = exp.devices.device_ids()[0]
        template_price = next(row["template_bytes"] for row in estimate["cards"]
                              if row["card"] == first)
        credit = min(template_price, priced_template_live_bytes(template, cfg, device=first))
        budgets[first] += credit
        budget_terms[first]["template_live_credit_bytes"] = credit
        if credit:
            estimate = price()
    for dev, terms in budget_terms.items():
        terms["budget_bytes"] = budgets[dev]
    estimate["budget_terms"] = budget_terms
    host_free = host_available_bytes()
    host_credit = {"host_store_bytes": 0, "host_boundary_bytes": 0}
    if inventory is not None:
        # The pinned store and loaded host forcing already reduced available
        # RAM. Credit only those supplied arrays, capped by the two terms
        # that this same admission charges. Rank seam staging is not built.
        live = priced_host_live_bytes(inventory, geography, boundaries)
        host_credit = {key: min(estimate[key], value) for key, value in live.items()}
    host_budget = None if host_free is None else host_free + sum(host_credit.values())
    host_budget = _stream_host_budget(host_budget, stream_head)
    estimate["host_budget_terms"] = {
        "available_bytes": host_free,
        "store_live_credit_bytes": host_credit["host_store_bytes"],
        "boundary_live_credit_bytes": host_credit["host_boundary_bytes"],
        "budget_bytes": host_budget}
    gate = devices_gate(estimate, budgets=budgets, host_budget=host_budget)
    print(gate["verdict"], flush=True)
    if gate["refuse"]:
        from woof.core.resident_admission import memory_gate_overridden
        if memory_gate_overridden():
            # The same override the resident envelope takes (--no-memory-gate
            # or WOOF_NO_MEMORY_GATE=1): the per-card figure is a priced
            # upper bound (several slabs on ONE card each carry the process's
            # fixed terms), so the cards' own allocations decide.  Said out
            # loud, never silent.
            print("prepared forecast: [devices] memory admission OVERRIDDEN by "
                  "--no-memory-gate: the per-card figures above are priced upper "
                  "bounds and each card's own allocation now decides", flush=True)
            return estimate
        raise DevicesRefused("[devices] memory admission refused before ranks are built: "
                             "prevent a card or pinned host allocation failure\n"
                             + gate["verdict"])
    return estimate


def _devices_stepper(bundle, node, exp, decisions, *, admission=None):
    from woof.core.adaptive_clock import maximum_map_factor
    from woof.core.streaming import ranked_decision, ranked_domain_builder
    cfg = node.cfg.run
    decision = ranked_decision(
        cfg, exp.devices,
        max_map_factor=maximum_map_factor(geography=bundle.geography))
    # node.state, as steppers_for_tree passes it: on the store road it is the
    # slab-height template, the object make_stepper publishes the store on.
    # MEASURED with None here: every ranked forecast stopped in publish_store
    # ("'NoneType' object has no attribute '_streamed_store'") after the
    # slabs were built and before the first step.
    stepper = streaming.make_stepper(
        node.state, cfg, decision=decision,
        build=ranked_domain_builder(
            bundle, clock=node.clock, options=exp.devices,
            snapshot_limits=(None if admission is None else tuple(
                rank["frame_snapshot_bytes"] for rank in admission["rank_shapes"])),
            mynn_column_chunks=(None if admission is None else tuple(
                rank.get("mynn_column_chunk") for rank in admission["rank_shapes"]))))
    decisions[int(node.cfg.grid_id)] = decision
    node.state._streamed_domain = stepper
    gy, gx = exp.devices.resolved_grid(cfg.nx, cfg.ny)
    actual = getattr(stepper.tiled_run, "transport_report", None)
    actual = actual() if callable(actual) else actual
    if actual is None:
        raise RuntimeError("ranked run exposes no transport_report; refusing an "
                           "unverifiable [devices] receipt")
    print(f"prepared forecast: [devices] ON: {exp.devices.count} slabs on cards "
          f"{list(exp.devices.device_ids())}, grid {gy}x{gx}, halo {decision.halo}, "
          f"transport {actual}", flush=True)
    return {int(node.cfg.grid_id): stepper}


def _devices_step_observer(node, stepper, options, history, peaks, observer=None):
    """Retain each committed step's live clock and sample every card's pool."""
    def observe(**event):
        import cupy as cp
        row = {"step": int(event["step_count"]), "dt": float(event["dt"]),
               "time_step_sound": int(node.cfg.run.time_step_sound),
               "model_seconds": float(event["model_seconds"])}
        # Preserve the run's reported operands when it supplies them.
        for name in ("dt", "time_step_sound"):
            if name in stepper.report:
                row[name] = stepper.report[name]
        history.append(row)
        for dev in set(options.device_ids()):
            with cp.cuda.Device(dev):
                pool = cp.get_default_memory_pool()
                current = {"pool_used_bytes": int(pool.used_bytes()),
                           "pool_held_bytes": int(pool.total_bytes())}
                peak = peaks.setdefault(str(dev), {})
                for name, value in current.items():
                    peak[name] = max(peak.get(name, 0), value)
        if observer is not None:
            observer(**event)
    return observe


def _root_setup_fingerprint_for(node, stream=None):
    from woof.prepared_domain_tree_forecast import _root_setup_fingerprint

    return _root_setup_fingerprint(node, stream)


def _release_forecast_cards(stepper) -> dict:
    """Close a ranked run's slabs and free every card's pool; how much went.

    Called once the last step's history is durable and every value the end
    of the run reads from the cards has been taken.  The host store, the
    receipts and the files are untouched; what goes is the device memory
    the slabs and their staging held, so another job can admit on the
    cards while this process hashes and verifies on the host.
    """
    import cupy as cp

    run = getattr(stepper, "tiled_run", None)
    devices = tuple(dict.fromkeys(
        int(d) for d in (getattr(run, "devices", None) or ())))

    def held():
        total = 0
        for dev in devices:
            with cp.cuda.Device(dev):
                total += int(cp.get_default_memory_pool().total_bytes())
        return total

    before = held()
    started = time.perf_counter()
    if run is not None and not getattr(run, "closed", True):
        run.close()
    for dev in devices:
        with cp.cuda.Device(dev):
            cp.cuda.Device(dev).synchronize()
            cp.get_default_memory_pool().free_all_blocks()
    after = held()
    return {"cards": list(devices), "pool_bytes_before": before,
            "pool_bytes_after": after, "freed_bytes": max(0, before - after),
            "release_seconds": round(time.perf_counter() - started, 6)}


def _devices_receipt(exp, stepper, decision, admission, history, peaks):
    """Carry the ranked run's own clock, transport and peak measurements."""
    run = stepper.tiled_run
    actual = getattr(run, "transport_report", None)
    actual = actual() if callable(actual) else actual
    return {**admission, "options": exp.devices.to_json(),
            "decision": {"road": getattr(decision, "road", "ranks"),
                         "halo": decision.halo, "detail": dict(decision.detail)},
            "transport_report": actual, "step_report": dict(stepper.report),
            "dt_history": history, "peak_memory_per_card": peaks,
            "peak_memory_basis": "post-step pool samples; transient peaks may be higher",
            "ranked_peak_memory_per_card": stepper.report.get(
                "peak_memory_per_card", getattr(run, "peak_memory_per_card", None)),
            # How output reached the host: frame-only downloads (count,
            # bytes, the seconds the stepping thread spent issuing them and
            # the seconds the writer waited for them) against whole-store
            # drains and the full copies back they force.
            "output_road": _devices_output_road(run)}


def _devices_output_road(run):
    report = getattr(run, "output_report", None)
    if report is None:
        return None
    issue = list(report.get("frame_issue_seconds", ()))
    wait = list(report.get("frame_wait_seconds", ()))
    summary = {key: value for key, value in report.items()
               if key not in ("frame_issue_seconds", "frame_wait_seconds")}
    summary["full_drain_seconds"] = round(float(summary.get("full_drain_seconds", 0.0)), 6)
    for name, values in (("frame_issue_seconds", issue), ("frame_wait_seconds", wait)):
        ordered = sorted(values)
        summary[name] = {"count": len(values),
                         "median": ordered[len(ordered) // 2] if ordered else None,
                         "max": ordered[-1] if ordered else None,
                         "total": round(sum(values), 6)}
    return summary


def _choose_stream_init_road(mode: str, *, decision, reader, cfg=None,
                             device_memory=None,
                             resident_estimate=None) -> tuple[str, dict | None]:
    """Which road this forecast initializes on, and the numbers behind it.

    Returns ``("resident", None)`` -- today's road, with nothing at all
    changed and no device queried -- for every run that does not STREAM.  The
    flag has meaning only for a streamed forecast: with ``[tiles]`` off the
    resident state is not an intermediary the run could skip, it is the
    domain, and there is no second road to choose.

    ``resident`` and ``store`` force a road.  ``store`` is forced by the
    bit-parity gate, which needs the store-direct road to run on a domain that
    WOULD have fitted so the two can be compared frame for frame; ``resident``
    on a domain that cannot fit fails exactly where it fails today, inside the
    allocation, rather than being silently rescued into a mode the caller
    asked not to be in.
    """

    if mode not in STREAM_INIT_CHOICES:
        raise ValueError(
            f"--stream-init must be one of {', '.join(STREAM_INIT_CHOICES)}, "
            f"got {mode!r}")
    streams = decision is not None and bool(decision.stream)
    if not streams:
        if mode == "store":
            raise ValueError(
                "--stream-init store asks this forecast to build its domain "
                "straight into a pinned host store and integrate out of it, "
                "but this run does not stream"
                + ("" if decision is None else f" ({decision.explain()})")
                + ".  The store is only ever read by the tiling sweep, so a "
                "resident forecast initialized into one would have no domain "
                "to step.  Configure [tiles] (mode = 'on', or 'auto' on a "
                "domain that does not fit) or drop the flag.")
        return "resident", None

    from woof.boundary_fields import source_boundary_species

    pricing = _stream_init_pricing(
        _prepared_state_manifest_bytes(reader),
        *(device_memory if device_memory is not None
          else _free_device_bytes()),
        columns=(None if cfg is None else int(cfg.nx) * int(cfg.ny)),
        nz=(None if cfg is None else int(cfg.nz)),
        cfg=cfg, resident_estimate=resident_estimate,
        forcing_intervals=_retained_interval_count(reader),
        boundary_species=source_boundary_species(_priced_boundary_source(reader, None)),
        # WHAT THE RESIDENT STATE WILL HAVE TO SHARE THE CARD WITH.  Priced
        # here rather than left to the fit fraction: see
        # ``_AUTO_RESIDENT_FIT_FRACTION`` for the 10 GiB card this omission
        # paged into the ground.
        streamed_bytes=_streamed_peak_bytes(cfg, decision))
    priced = pricing["priced_resident_bytes"] / _GIB
    free = pricing["device_free_bytes"] / _GIB
    total = pricing["device_total_bytes"] / _GIB
    manifest = pricing["state_manifest_bytes"] / _GIB
    allowance = pricing["fit_allowance_bytes"] / _GIB
    # WHICH TERM SET THE PRICE, in the sentence itself.  A receipt that
    # printed only the winning number would leave a reader unable to tell a
    # cache-driven price from a route-driven one, and the two are answers to
    # different questions.
    if pricing["priced_from"] == "configuration":
        from woof.core.resident_admission import terms_text

        reference = pricing["measured_priced_bytes"]
        reference_text = ("unpriced" if reference is None
                          else f"{reference / _GIB:.2f} GiB")
        priced_as = (
            f"the prepared resident case prices at {priced:.2f} GiB "
            f"(configuration-aware preflight inventory: "
            f"{terms_text(pricing['configuration_resident_terms'])}; "
            f"manifest estimate {pricing['manifest_priced_bytes'] / _GIB:.2f} GiB, "
            f"measured reference {reference_text}) "
            f"against {free:.2f} GiB free of {total:.2f} GiB on the card")
    elif pricing["priced_from"] == "measured-per-column":
        priced_as = (
            f"the prepared resident case prices at {priced:.2f} GiB "
            f"({pricing['columns']} columns x {int(pricing['nz'])} levels at "
            f"{_MEASURED_PREPARED_BYTES_PER_COLUMN:.0f} B/column/"
            f"{_MEASURED_BYTES_PER_COLUMN_NZ} levels MEASURED on the reference configuration, "
            f"over {manifest:.2f} GiB of state/* in the cache manifest x "
            f"{_PREPARED_RESIDENT_HEADROOM:.2f} = "
            f"{pricing['manifest_priced_bytes'] / _GIB:.2f} GiB) against "
            f"{free:.2f} GiB free of {total:.2f} GiB on the card")
    else:
        priced_as = (
            f"the prepared resident case prices at {priced:.2f} GiB "
            f"({manifest:.2f} GiB of state/* in the cache manifest x "
            f"{_PREPARED_RESIDENT_HEADROOM:.2f} measured physics headroom, "
            f"over this route's measured "
            f"{_MEASURED_PREPARED_BYTES_PER_COLUMN:.0f} B/column) against "
            f"{free:.2f} GiB free of {total:.2f} GiB on the card")
    # THE OTHER TWO TERMS, said out loud whenever they were priced.  A receipt
    # that printed only the resident number would leave a reader unable to see
    # why a state that plainly fits was still sent to the store.
    if pricing["streamed_peak_bytes"] is not None:
        streamed = pricing["streamed_peak_bytes"] / _GIB
        peak = pricing["resident_peak_bytes"] / _GIB
        priced_as += (
            f", and the streamed run puts {streamed:.2f} GiB more beside it "
            f"(the tile buffers this decision selected plus the measured "
            f"radiation transient, both on the card while the resident state "
            f"is), for a {peak:.2f} GiB peak")
    if mode == "resident":
        road = "resident"
        why = (f"--stream-init resident was asked for, so the resident state "
               f"is built whether or not it fits; {priced_as}")
    elif mode == "store":
        road = "store"
        why = (f"--stream-init store was asked for, so the store is filled "
               f"slab by slab and no domain-shaped device array is "
               f"allocated; {priced_as}")
    elif pricing["fits"]:
        road = "resident"
        why = (f"--stream-init auto: {priced_as}, which fits inside the "
               f"{allowance:.2f} GiB this rule allows "
               f"({_AUTO_RESIDENT_FIT_FRACTION:.0%} of free), so the resident "
               f"road runs -- it is the faster one and it is the one the "
               f"parity proof is written against")
    else:
        road = "store"
        why = (f"--stream-init auto: {priced_as}, which does NOT fit inside "
               f"the {allowance:.2f} GiB this rule allows "
               f"({_AUTO_RESIDENT_FIT_FRACTION:.0%} of free), so the store is "
               f"filled slab by slab and no domain-shaped device array is "
               f"allocated")
    receipt = {
        "requested": mode,
        "road": road,
        "why": why,
        "measured_bare_state_bytes_per_column":
            _MEASURED_BARE_STATE_BYTES_PER_COLUMN,
        "measured_prepared_bytes_per_column":
            _MEASURED_PREPARED_BYTES_PER_COLUMN,
        **pricing,
    }
    return road, receipt


def _validate_store_bundle_receipt(
        receipt: Mapping[str, object], expected_content_sha256: str,
) -> None:
    """Recheck the caller's content pin immediately after the store is built.

    :func:`_validate_restored_cache_receipt`'s store-direct twin, at the same
    instant and for the same reason: the loader that just read the bundle
    states which bytes it read, and that statement is compared against the
    digest the caller hash-bound before the run began.  A cache swapped
    between preflight and load is refused here rather than integrated.
    """

    if (not isinstance(receipt, Mapping)
            or receipt.get("schema") != _PREPARED_STORE_SCHEMA
            or receipt.get("status") != "LOADED"
            or receipt.get("content_sha256") != expected_content_sha256):
        raise ValueError(
            "prepared store differs from the caller-pinned cache content")


def _announce_adopted_coordinate(sentence: str) -> None:
    """Say, once, which coordinate this forecast is actually integrating."""

    from woof.explain import warn

    warn(sentence,
         "The prepared inputs carry the hybrid coefficient arrays "
         "themselves, the way WRF's wrfinput carries C3H/C4H, so the model "
         "integrates them rather than rebuilding from the configuration.  "
         "The preparation derived this etac from the terrain the run can "
         "touch because the configured one could not order every column; "
         "its receipt names the governing column.  p_top is unchanged.")


def _validate_store_direct_vertical_contract(reader, cfg, bundle) -> None:
    """The host-side cache checks the slab loader does not make itself.

    ``restore_prepared_cache`` validates the vertical contract while it has
    the whole domain in hand: the coordinate arrays' shapes against ``nz``,
    and the explicit eta grid against ``p_top``.  Neither needs a device or a
    domain -- both are functions of the cache header and the 1-D coordinate
    the bundle already carries -- so both are made here rather than dropped
    on the way to the store.  What genuinely CANNOT be reproduced is named in
    the receipt instead; see ``_store_direct_gaps``.
    """

    from woof.ingest.prepared_cache import PreparedCacheMismatchError
    from woof.vertical_contract import (
        validate_coordinate_shapes, validate_explicit_eta_grid,
    )

    metadata = reader.header["metadata"]
    coord_shapes = {
        name: reader.arrays[f"coord/{name}"]["shape"]
        for name in metadata["coord_arrays"]
        if f"coord/{name}" in reader.arrays
    }
    try:
        validate_coordinate_shapes(
            coord_shapes, nz=cfg.nz, context="prepared-store load")
        validate_explicit_eta_grid(
            bundle.coord.znw, nz=cfg.nz, p_top=bundle.base.p_top,
            context="prepared-store load")
    except (TypeError, ValueError) as exc:
        raise PreparedCacheMismatchError(str(exc)) from exc


def _store_direct_gaps(reader) -> dict[str, object]:
    """What the store-direct road does NOT check, said out loud in a receipt.

    Named rather than quietly dropped, because a check that stops running
    without saying so is indistinguishable from a check that keeps passing.
    One entry only, and it is a real one: ``restore_prepared_cache`` rebuilds
    the domain's setup arrays and compares ``setup_fingerprint(state)`` with
    the cache's, which needs a domain-shaped state -- exactly the object this
    road exists not to build.

    What still holds in its place is stated beside it rather than implied: the
    reader verified the header's content digest and every array against its
    own SHA-256 as it was read, and the cache identity was compared against
    the caller's before a byte was loaded, so the BYTES and the experiment
    they belong to are both pinned.  What is not re-derived is that the active
    config reconstructs the same map factors, Coriolis and base state from
    them.
    """

    return {
        "setup_fingerprint": {
            "checked": False,
            "why": ("setup_fingerprint is taken over a whole domain's "
                    "reconstructed setup arrays, and this road never builds "
                    "a domain-shaped state"),
            "instead": ("the cache identity, the header content digest and "
                        "every array's own SHA-256 were verified on the way "
                        "into the store"),
            "cache_content_sha256": reader.content_sha256,
        },
    }


def _store_health_auxiliaries(bundle, cfg) -> dict[str, object]:
    from woof.core.health import prepared_store_health_auxiliaries

    return prepared_store_health_auxiliaries(bundle, cfg)


def _store_full_state_health(bundle, cfg, *, phase: str) -> dict[str, object]:
    """The full-state descriptor gate over the store, as a health record.

    ARMED, where this road used to declare the gate unarmed.  The reason it
    was unarmed was true and is not a reason to leave it so: the CUDA gate is
    one block per whole FIELD and needs a domain-shaped device state, which
    this road never builds -- but the domain exists, in the pinned host store,
    and :func:`woof.core.health.validate_store_fields` runs the production
    rule set over it there.  The instrument differs from the resident road's
    (NumPy over host memory rather than one kernel launch over device memory);
    the RULES do not, because both sides read them out of
    :func:`woof.core.health.rule_for_field`.

    Same key shape as the resident road's ``vars(ValidationReport)`` so that
    one reader handles both receipts, with the coverage record added rather
    than substituted: what the gate saw is a count, and what it did not see is
    a list of names.
    """

    from woof.core.health import validate_store_fields
    from woof.core.streaming import streamed_store_inventory

    template = bundle.template
    report, coverage = validate_store_fields(
        template,
        # The RUN's own inventory rule, harvested off the same template the
        # descriptors are collected from, so the two walkers join by object
        # identity.  ``streamed_store_inventory`` and not the plain carrier
        # inventory for the reason ``store_from_prepared_cache`` gives: it is
        # the rule the store itself was built by, and REFL_10CM is the key the
        # two disagreed on.
        streamed_store_inventory()(template, None),
        bundle.store,
        domain_shape=(int(cfg.ny), int(cfg.nx)),
        auxiliaries=_store_health_auxiliaries(bundle, cfg),
        # The DOMAIN's model lid, from the same base object
        # ``DomainState.load_base`` reads it out of on the resident road.
        # The theta ceiling is that lid carried down the dry adiabat, so a
        # template that never loaded a base would otherwise gate a deep-top
        # domain against the 100 hPa reference ceiling and refuse it.
        p_top=getattr(bundle.base, "p_top", None),
        phase=phase)
    record = _strict_json(vars(report))
    record["armed"] = True
    record["instrument"] = (
        "full-state descriptor gate over the pinned host store "
        "(woof.core.health.validate_store_fields)")
    record["covers"] = (
        f"{coverage['fields_checked']} gated fields over the whole domain, "
        "each under the rule woof.core.health.rule_for_field gives it -- the "
        "same per-field bounds the resident road's kernel applies")
    record["coverage"] = {
        key: value for key, value in coverage.items() if key != "covered"}
    return record


#: How long model step 1 may run before the runner says out loud that
#: something is being compiled.
#:
#: MEASURED on the reference box: a healthy first step is 1.2-1.3 s and
#: the steps after it are 0.13 s, while the compile this exists to name
#: is 52 s.  Fifteen seconds is ten times any healthy first step seen
#: and a third of the shortest compile seen, and it matches the cadence
#: of `woof go`'s own heartbeat so the two lines do not race.
FIRST_STEP_STALL_SECONDS = 15.0


class _FirstStepStallWatch:
    """Say the compile is happening WHILE it is happening.

    THE HOLE THE CENSUS ALONE DOES NOT CLOSE, measured 2026-08-16.
    Predicting the compile from the kernel cache works on a genuinely
    cold box and fails in the chain: `woof go`'s preprocessing stage
    uses the GPU too, so by the time the forecast runner starts, the
    cache already holds entries for this card -- and a reading taken
    then says "warm" even though the forecast's own kernels are all
    missing.  Staging 200 sm_120 entries against an sm_86 card and
    running the chain reproduced exactly that: silent notice, 52 s of
    compilation inside model step 1.

    So this does not predict.  It measures: if step 1 has not finished
    after :data:`FIRST_STEP_STALL_SECONDS`, the run says so, names what
    the kernel cache looked like at launch, and flips the published
    status -- and it cannot false-positive on a fast run, because a
    fast run disarms it.

    One timer thread, daemonised, cancelled by the first completed step.
    """

    def __init__(self, *, progress_path: Path, inputs, exp, step_log,
                 census, road: str = "unknown",
                 delay: float = FIRST_STEP_STALL_SECONDS,
                 capability: str | None = None, cache_census_now=None):
        self._progress_path = progress_path
        self._inputs = inputs
        self._exp = exp
        self._step_log = step_log
        self._census = census
        self._road = str(road or "unknown")
        self._delay = float(delay)
        self._capability = capability
        self._cache_census_now = (scan_kernel_cache if cache_census_now is None
                                  else cache_census_now)
        self._timer = None
        self._fired = False

    def arm(self) -> None:
        if self._timer is not None:
            return
        self._timer = threading.Timer(self._delay, self._say)
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def observe(self, **event) -> None:
        """The step observer's first call ends the wait."""

        self.disarm()

    def wrap(self, step_observer):
        """Chain :meth:`observe` in front of the real step observer."""

        if step_observer is None:
            return self.observe

        def observed(**event):
            self.disarm()
            step_observer(**event)

        return observed

    def _grown_entries(self) -> int:
        """Cubins that appeared since launch, or 0 if that is unaskable.

        The corroboration the launch census cannot give.  A cache that
        reads warm at launch can still be missing every kernel the
        FORECAST needs -- that is the measured hole in the class
        docstring -- but a compile that is genuinely running WRITES, and
        entries appearing while step 1 stalls are positive evidence no
        timer can supply.
        """

        try:
            entries, _undecodable, _architectures = self._cache_census_now()
        except Exception:            # noqa: BLE001 - telemetry never fails
            return 0
        launched = 0
        if self._census is not None:
            launched = int(self._census[0])
        return max(0, int(entries) - launched)

    def _say(self) -> None:
        if self._fired:
            return
        self._fired = True
        capability = (current_compute_capability() if self._capability is None
                      else self._capability)
        state = kernel_cache_state(compute_capability=capability,
                                   census=self._census)
        # THE STALL IS THE TRIGGER, NEVER THE EVIDENCE -- ledger #324.
        # Captured live: this printed "the one-time NVRTC compile of this
        # run's GPU kernels" while the very same line reported 388 cached
        # entries, 388 of them for this card, and not one cubin was
        # written while it stalled.  The run was transfer-bound on the
        # streamed road.  A timer knows that a reader is waiting; it
        # knows nothing whatever about what they are waiting for, so the
        # cause is claimed only where something was actually measured.
        detail = (f"the kernel cache held {state.entries} entry(s) at "
                  f"launch, {state.entries_for_capability} of them for "
                  f"sm_{state.compute_capability or '?'}")
        grown = self._grown_entries()
        head = (f"forecast: model step 1 has been running for "
                f"{self._delay:.0f} s -- ")
        if state.reason in (COLD_CACHE, ARCHITECTURE_MISSING) or grown:
            reason = state.reason or "cache_grew_during_first_step"
            if grown:
                detail += f", and {grown} more have been written since"
            print(f"{head}this is the one-time NVRTC compile of this run's "
                  f"GPU kernels for the local card ({detail}); typically "
                  "1-3 minutes, and the result is cached so later runs "
                  "skip it", flush=True)
            status = COMPILING_STATUS
        else:
            # No cause is named that was not measured.  The road IS
            # measured -- the run chose it and wrote the receipt -- so
            # the store road may say what feeds each step.  Anything
            # else says only what is known: nothing has finished yet.
            if self._road == "store":
                print(f"{head}first model step still running; streamed "
                      "transfers feed each step on this card, and no model "
                      f"step has completed ({detail})", flush=True)
            else:
                print(f"{head}first model step still running; no model step "
                      f"has completed ({detail})", flush=True)
            reason = None
            status = "RUNNING"
        try:
            _atomic_json(self._progress_path, {
                "schema": PROGRESS_SCHEMA,
                "status": status,
                "source": self._inputs.source,
                "model_elapsed_seconds": 0.0,
                "requested_run_seconds": float(self._exp.run_seconds),
            }, heartbeat=True)
        except OSError:
            pass
        # A receipt that claimed a compile on a timer was the same defect
        # written down, so it is announced on the same evidence as the
        # line -- and not at all without it.
        if self._step_log is not None and reason is not None:
            self._step_log.announce_kernel_compile(
                reason=reason,
                compute_capability=state.compute_capability,
                cached_entries=state.entries,
                cached_entries_for_this_card=state.entries_for_capability,
                detected_by=f"model step 1 exceeded {self._delay:.0f} s")


def _announce_kernel_compile(progress_path: Path, inputs, exp,
                             step_log=None, census=None) -> None:
    """Flip the published status when the NVRTC compile is actually coming.

    Physics initialization is where a first run pays its one-time NVRTC
    compile (~100 s on a modern card), and it used to pay it under the stale
    RESTORING status -- the first field run of the published wheel watched
    that silence and concluded a hang.  One line and one status flip, only
    when the kernel cache says the compile is really about to happen.

    Shared by both initialization roads: the store-direct road pays the same
    compile inside its first slab, and a road that skipped this would
    reintroduce exactly the silence for the runs most likely to be long.

    The card is asked here, not inside the notice module: this runner has
    long since initialised CUDA, and
    :func:`woof.kernel_compile_notice.kernel_cache_state` stays a pure
    filesystem predicate so a caller that must not touch the device can
    still use it.  Asking is what catches the card-swap case -- the cache
    is keyed by architecture, so 7,164 entries for the previous card are
    7,164 entries this run cannot load.

    ``census`` is the cache reading taken before this run compiled
    anything.  It is not optional in practice and the reason is
    measured: by the time this function is reached, this run's own
    first kernels are already in the cache it would otherwise scan, so
    a cache that was unusable at launch reads as warm.  See
    :func:`woof.kernel_compile_notice.kernel_cache_state`.

    ``step_log`` is told that a compile was announced.  It cannot be told
    what it COST yet -- most of the compile lands inside model step 1 --
    so the log holds the claim and emits one ``phase`` record once it has
    a measurement to attach.
    """

    state = kernel_cache_state(
        compute_capability=current_compute_capability(), census=census)
    if state.notice is None:
        return
    print(state.notice, flush=True)
    if step_log is not None:
        step_log.announce_kernel_compile(
            reason=state.reason,
            compute_capability=state.compute_capability,
            cached_entries=state.entries,
            cached_entries_for_this_card=state.entries_for_capability)
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": COMPILING_STATUS,
        "source": inputs.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": float(exp.run_seconds),
    }, heartbeat=True)


def _store_domain_extreme(store: Mapping[str, object], key: str,
                          reducer) -> float | None:
    """One reduction over a whole-domain carrier, read where the domain is.

    The receipt's ``swdown``/``rainnc`` numbers are reductions over a physics
    field, and on this road there is no domain-shaped state to take them from
    -- the only domain-shaped copy of those fields is the store, which is
    also the one the sweep keeps current.  ``None`` for a key the store does
    not carry, so a missing diagnostic is a hole in the receipt rather than
    an exception at the end of a completed forecast; the caller records WHICH
    key was missing beside it.
    """

    array = store.get(key)
    if array is None:
        return None
    return float(reducer(np.asarray(array)))


def _single_prepared_root(domain_cfg, grid, state, clock, *, store_direct):
    """Give both initialization roads the same scheduled boundary clock."""
    from woof.core.model import DomainNode
    from woof.ingest.lateral_bc import bind_lateral_boundary_clock

    node = DomainNode(domain_cfg, grid, state, clock, None, [], None)
    if not store_direct:
        # Bind before tile factories or restart setup can inherit the mirror.
        # The store road has no resident mirror; its builder receives this
        # same node.clock explicitly below.
        bind_lateral_boundary_clock(node.state, node.clock)
    return node


def _single_checkpoint_identity(inputs, runtime_source_identity):
    """Bind every sealed authority, including the unchanged stop time in TOML."""
    from woof.physics_params import document as parameter_document

    parameters = parameter_document(getattr(
        getattr(inputs, "experiment", None), "physics_params", None))
    parameter_binding = ({} if parameters is None
                         else {"physics_params": parameters})
    smoothing = (getattr(inputs, "geometry_receipt", {}) or {}).get(
        "terrain_autosmooth")
    smoothing_binding = ({"terrain_autosmooth": dict(smoothing)}
                         if smoothing is not None else {})
    stream_head = getattr(inputs, "stream_head", None)
    head_sha256 = (stream_head["head_sha256"] if stream_head is not None
                   else (dict(getattr(inputs, "proof", {}) or {}).get(
                       "boundary_stream") or {}).get("head_sha256"))
    if head_sha256 is None:
        return {
            "schema": "gpuwm.prepared-single-checkpoint.v1",
            "source": inputs.source,
            "prepared_content_sha256": inputs.cache_reader.content_sha256,
            "authority_sha256": dict(inputs.file_sha256),
            "runtime_source_identity": runtime_source_identity,
            **parameter_binding,
            **smoothing_binding,
        }
    # A chained preparation is bound by its head whichever way this run
    # binds it (the head at launch, or the sealed proof that names it), so
    # a checkpoint written before or after the seal resumes under either
    # binding.  The head binds the cache: the seal is checked against it.
    # An as-posted head binds its input plan and its seal writes the
    # manifest, held to that plan, so the manifest is the seal's too.
    sealed_only = {"proof", "cache_header", "prepared_head"}
    posting = dict(getattr(inputs, "proof", {}) or {}).get("posting")
    if ((stream_head is not None
         and stream_head["basis"].get("as_posted") is not None)
            or (isinstance(posting, Mapping) and posting.get("as_posted"))):
        sealed_only |= {"source_manifest"}
    return {
        "schema": "gpuwm.prepared-single-checkpoint.v1",
        "source": inputs.source,
        "prepared_head_sha256": head_sha256,
        **smoothing_binding,
        "authority_sha256": {
            name: digest for name, digest in inputs.file_sha256.items()
            if name not in sealed_only},
        "runtime_source_identity": runtime_source_identity,
        **parameter_binding,
    }


@contextmanager
def _checkpoint_restore_resources(writers, step_log):
    """Unwind eager output owners if checkpoint or restored health is refused."""
    try:
        yield
    except BaseException as error:
        # The closures read `failure`, not the except name: Python unbinds
        # `error` when this clause ends, so a closure over it is a NameError
        # waiting for any caller that outlives the clause, and the
        # undefined-name gate flags it (tests/test_no_undefined_names.py).
        failure = error
        for close in (
                lambda: writers.__exit__(type(failure), failure, failure.__traceback__),
                lambda: step_log.close(status="FAIL", error=f"{type(failure).__name__}: {failure}")):
            try:
                close()
            except BaseException as cleanup_error:
                error.add_note(f"restore cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
        raise


#: The attribute a forecast failure carries its model clock on, from the
#: integration that raised it to the report writer in :func:`main`.
_FAILURE_CLOCK_ATTRIBUTE = "gpuwm_model_elapsed_seconds"


def _require_next_interval(intervals, elapsed: float, *, writers, exp) -> None:
    """Wait for the interval the next step needs, between two steps.

    Called after each committed step (every history frame and checkpoint
    of that instant already written), so a wait for a lead that has not
    posted happens with the model at a clean seam rather than inside a
    step.  The interval is loaded exactly as the step would load it (the
    same cached object), so nothing the model computes changes.  A lead
    that passes its late time stops the run here: the frames through the
    seam are drained to disk first, and the refusal says where it stopped.
    """

    from woof.ingest.boundary_stream import SourceBehind
    from woof.ingest.lateral_bc import interval_index

    index = interval_index(intervals, float(elapsed))
    try:
        intervals[index]
        # The next interval loads beside the model while this one is used,
        # so the next boundary crossing finds it ready (a no-op until its
        # marker is posted, and on a non-streamed sequence).
        prefetch = getattr(intervals, "prefetch", None)
        if prefetch is not None:
            prefetch(index + 1)
    except SourceBehind as behind:
        try:
            writers.drain()
        except Exception:  # noqa: BLE001 - the refusal is what matters
            pass
        valid = exp.start_time + timedelta(seconds=float(elapsed))
        stopped = behind.at(
            model_elapsed_seconds=float(elapsed),
            model_valid_time=valid.strftime("%Y-%m-%dT%H:%M:%SZ"))
        stopped.at_seam = True
        raise stopped from None


def _stop_at_seam(error, *, model, node, exp, schedule, restart_handler,
                  checkpoint_ticks, checkpoints, seam_waits):
    """A late source lead ended the run: checkpoint the seam, say where.

    A stop raised between two steps (:func:`_require_next_interval`)
    leaves the model at a clean seam, so its state is checkpointed there
    (one write) and a relaunch resumes exactly at the seam; an instant
    already checkpointed is not written twice.  A stop raised anywhere
    else keeps the latest hourly checkpoint and writes none, because a
    state caught mid-step is not a restart state.
    """

    if seam_waits is not None:
        # The wait ended because the lead is late, not because it came.
        seam_waits.abandon()
    checkpoint = checkpoints[-1] if checkpoints else None
    if getattr(error, "at_seam", False):
        ticks = int(node.clock.ticks)
        checkpoint = checkpoint_ticks.get(ticks)
        if checkpoint is None and float(error.details.get(
                "model_elapsed_seconds") or 0.0) > 0.0:
            try:
                restart_handler(model, ticks)
                wait = getattr(restart_handler, "wait_durable", None)
                if wait is not None:
                    wait()
                checkpoint = checkpoints[-1]
            except Exception as failure:  # noqa: BLE001 - keep the refusal
                print("prepared forecast: the checkpoint at the seam could "
                      f"not be written ({type(failure).__name__}: "
                      f"{failure}); the latest hourly checkpoint stands",
                      file=sys.stderr, flush=True)
                checkpoint = checkpoints[-1] if checkpoints else None
    elif int(node.clock.ticks) > 0:
        elapsed = float(node.clock.ticks / schedule.clock.tick_den)
        valid = exp.start_time + timedelta(seconds=elapsed)
        error = error.at(model_elapsed_seconds=elapsed,
                         model_valid_time=valid.strftime("%Y-%m-%dT%H:%M:%SZ"))
    stopped = error.at(checkpoint=checkpoint)
    stopped.at_seam = getattr(error, "at_seam", False)
    return stopped


def _mark_failure_clock(error: BaseException, clock) -> None:
    """Record on ``error`` the model clock of the step that failed.

    ``progress.json`` is published on the first and every 60th step only,
    so a run that failed at model second 40 on a 2.5 s step reported the
    first step's 2.5 s in its failure report and final progress.  The
    clock is read here, while the model still exists, and never allowed
    to replace the failure being reported.
    """
    try:
        setattr(error, _FAILURE_CLOCK_ATTRIBUTE, float(clock.elapsed_seconds))
    except Exception:  # noqa: BLE001 - the failure itself is what matters
        pass


def _failure_clock(error: BaseException, heartbeat_seconds: float) -> float:
    """The model seconds a failure report states: the failed step's clock
    when the integration recorded one, else the last heartbeat's."""
    value = getattr(error, _FAILURE_CLOCK_ATTRIBUTE, None)
    try:
        return float(heartbeat_seconds if value is None else value)
    except (TypeError, ValueError):
        return float(heartbeat_seconds)


def _restore_single_checkpoint(model, restart):
    """Use the canonical tree reader after the canonical store is attached."""
    if restart is None:
        return None
    from woof.io.restart import restore_tree_restart
    from woof.supervisor import validate_manifest_checkpoint

    checkpoint = validate_manifest_checkpoint(Path(restart))
    return restore_tree_restart(checkpoint, model)


from woof.hydro.runtime import forecast_scope as _hydro_forecast_scope


@_hydro_forecast_scope
def run_prepared_forecast(
        inputs: PreparedForecastInputs, *, output_directory: Path,
        observer=None, first_products=None, stream_init: str = "auto",
        progress_options=None, preflight_seconds: float | None = None,
        kernel_cache_census=None, restart: Path | None = None,
        health_debug: bool = False,
        ensemble_bootstrap=None,
        prewarmed_execution=None,
        prestaged_boundaries=None,
        columns=None,
) -> dict[str, object]:
    """Restore, integrate, and publish hash-bound source-neutral history.

    ``observer`` is an optional second consumer of this run's progress,
    for a caller driving the runner in-process rather than reading its
    ``progress.json`` from outside (:class:`woof.runplan.RunObserver`
    is one).  It receives every per-step progress event this runner
    already builds, and -- if it carries the ``output_committed`` hook
    -- each wrfout as it becomes durable.  ``None`` leaves every byte of
    this runner's behaviour unchanged.

    ``first_products`` is a :class:`woof.first_products.FirstProducts`
    this runner OWNS, for the command line rather than for a host.  It
    exists because the feature was reachable only through a host: arming
    lives on ``RunObserver`` and on ``woof go``, so ``python -m
    woof.prepared_single_domain_forecast`` published its analysis frame
    -- fsynced, self-validated and renamed, minutes before the forecast
    ends -- with nothing watching.  MEASURED consequence: reaching a
    89 s launch-to-first-plot on this route took an EXTERNAL process
    polling the frames for ``GPUWM_WRITE_COMPLETE``, which is a watcher
    reimplementing a hook the writer already raises.  ``None`` -- and
    every invocation that names no products -- changes nothing.

    ``stream_init`` is ``--stream-init``: which road a STREAMED forecast
    builds its domain on.  ``"auto"`` (the default) prices the resident
    road against the card's free memory -- from this domain's own columns
    at the route's MEASURED cost per column, and from the cache manifest,
    whichever is larger -- and takes it wherever it comfortably fits,
    otherwise the store; ``"resident"`` and ``"store"`` force one.  A
    forecast that does not stream ignores it entirely -- there is no
    second road when the resident state IS the domain -- so a run without
    ``[tiles]`` executes byte for byte what it executed before this
    parameter existed, through the same calls.

    ``progress_options`` is :class:`woof.progress_log.ProgressOptions`,
    the four ``--progress-*`` / ``--frame-markers`` flags.  ``None`` is
    the DEFAULT SET, not silence: this runner prints WRF's per-step
    ``Timing for main:`` lines, writes ``progress.jsonl`` beside its
    outputs and publishes a frame-ready marker per durable history file
    unless a caller has asked it not to.  The reason it defaults on is
    the report this feature came from -- a run that says nothing about
    which step it is on cannot be driven from a script, and "the user
    did not pass a flag" is not a reason to withhold that.
    """

    from woof.ensemble.runtime_context import current_session
    ensemble_session = current_session()
    if ensemble_session is not None:
        return ensemble_session.run_prepared(
            run_prepared_forecast, inputs, output_directory=output_directory,
            observer=observer, first_products=first_products, stream_init=stream_init,
            progress_options=progress_options, preflight_seconds=preflight_seconds,
            kernel_cache_census=kernel_cache_census, restart=restart,
            health_debug=health_debug, ensemble_bootstrap=ensemble_bootstrap,
            **({"prestaged_boundaries": prestaged_boundaries}
               if prestaged_boundaries is not None else {}))

    # The single-domain regional route compiles the configuration's whole
    # recorded kernel set on every core now, the same call the domain-tree
    # runner makes (woof.kernel_plan).  Without it this route never bound a
    # plan: the 1x RTX PRO 6000 box's ArWen CONUS 3 km first step compiled
    # ~220 kernels on one core for 35.8 s with the card at 0 % and wrote no
    # plan for the next run (2026-10-17 04:52Z, v2 image).
    from woof import kernel_prewarm as _kernel_prewarm
    if prewarmed_execution is None:
        _kernel_prewarm.start_plan(inputs.experiment)
    else:
        from woof.standby_runtime import PreparedExecution
        if not isinstance(prewarmed_execution, PreparedExecution):
            raise TypeError("prewarmed execution must be a configuration-bound PreparedExecution")
        prewarmed_execution.verify(inputs)
    from woof.output_disk import require_output_space, renderer_products

    require_output_space(
        inputs.experiment, output_directory, restart=restart,
        render_products=renderer_products(first_products, observer))
    _verify_thompson_runtime_environment(inputs.physics_receipt)

    import cupy as cp

    from woof.core.clock import build_schedule, resolve_clock
    from woof.core.dycore import stability_gate_failed
    from woof.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher, default_cupy_probes,
        nvidia_smi_process_probes, process_memory_receipt,
    )
    from woof.core.health import StateHealthValidator
    from woof.core.model import (
        ExperimentState, ModelRuntimeStatus, execute_experiment,
    )
    from woof.core.refl import consume_refl_10cm, domain_start_ticks_of
    from woof.ingest.hrrr_physics import initialize_prepared_physics
    from woof.runtime import (
        declared_constant_glw, reset_history_windows_after_frame,
    )
    from woof.ingest.prepared_cache import restore_prepared_cache
    from woof.io.wrfout import PerDomainWrfoutWriters
    from woof.state_digest import canonical_state_digest

    from woof.progress_log import ProgressOptions

    outdir = Path(output_directory).resolve()
    progress_path = outdir / "progress.json"
    exp = inputs.experiment
    # [radar_heating]: every window is held to the run's grid identity and
    # its own end time HERE, on the host, before a card is allocated or the
    # domain is built.  Read at the attach instead, a wrong or missing
    # window cost the whole multi-card start-up before the refusal.  A
    # resume decides at the attach whether it needs the windows at all.
    radar_heating_checked = None
    if restart is None:
        from woof.da.forecast_heating import prevalidate
        radar_heating_checked = prevalidate(
            exp, prepared_root=inputs.prepared_root)
    from woof import runtime
    runtime._preparation_progress(observer, "restore-prepared-domain")
    from woof.case_data import trace_gas_overrides_from_config
    trace_gas_overrides = trace_gas_overrides_from_config(
        inputs.experiment_config, expected_sha256=inputs.file_sha256["experiment_config"])

    cfg = exp.root.run
    # Configured initial-state theta bubbles ([perturbation], PROVENANCE
    # D12), applied as the prepared domain-tree runner applies them: the
    # sealed cache stays the pure analysis, and the bubbles are added to
    # the RESTORED state before initialize_prepared_physics rederives
    # p/al/alt from the perturbed prognostics.  On the store road the
    # restore runs slab by slab, so each slab gets its rows of the bubble.
    # A resumed run applies nothing: the checkpoint trajectory carries the
    # evolved bubble.  Containment and zero-cell refusals fire here,
    # before any integration.
    perturbation = None
    perturbation_rows: list[dict] = []
    if exp.perturbation is not None and restart is None:
        from woof.ingest.init_perturbation import (
            build_initial_state_perturbation)
        perturbation = build_initial_state_perturbation(
            exp.perturbation, inputs.grid, grid_id=1,
            require_containment=True, cfg=cfg)

    def perturb_restored(result, slab_cfg, row_start):
        from woof.core.diagnostics import update_diagnostics
        # The cache serializes p as the preparation left it, so the
        # restored prognostics are diagnosed first; the bubble holds that
        # state.p while it rebalances the geopotential.
        update_diagnostics(result.state, slab_cfg.hypsometric_opt)
        if row_start is None:
            perturbation_rows.append(perturbation.apply_to_state(result.state))
            return
        window = copy.copy(perturbation)
        rows = slice(int(row_start), int(row_start) + int(slab_cfg.ny))
        window._placed = tuple(replace(
            placed, horizontal_km=(None if placed.horizontal_km is None
                                   else placed.horizontal_km[rows]))
            for placed in perturbation._placed)
        perturbation_rows.append(window.apply_to_state(
            result.state, allow_empty=True))

    def initialize_cached_physics(*args, row_start=None, domain_rows=None, **kwargs):
        if perturbation is not None:
            perturb_restored(args[0], args[1], row_start)
        return initialize_prepared_physics(
            *args, **kwargs, p_top=exp.vertical.p_top, column_chunk=exp.column_chunk,
            trace_gas_overrides=trace_gas_overrides)
    # Correct mode: a domain whose schemes carry a finding no correction
    # covers says so here, once, as well as in the receipt.
    from woof.correct_mode import uncorrected_notice
    correct_mode_line = uncorrected_notice({1: cfg})
    if correct_mode_line is not None:
        print(f"prepared forecast: {correct_mode_line}", flush=True)
    # The per-step log opens HERE, before the restore, so its first line
    # is the run announcing itself rather than the run's third minute.
    step_log = (progress_options or ProgressOptions()).open(
        outdir=outdir, start_time=exp.start_time,
        run_seconds=float(exp.run_seconds),
        # See the tree runner: scope-1 flag, read from the root, and it
        # decides both the `dt` field and the stream's schema string.
        adaptive_dt=bool(exp.root.run.use_adaptive_time_step))
    timing = {}
    # The preflight ran BEFORE this function -- it is what decided this
    # function may run at all -- so its number is handed in rather than
    # measured here.  It was previously recorded nowhere: the status
    # VALIDATING_PREPARATION was published and its duration, which
    # scales with the prepared cache, was dark.
    if preflight_seconds is not None:
        timing["preflight_verify"] = float(preflight_seconds)
    step_log.phase("preflight_verify", preflight_seconds)
    # The kernel cache as it was BEFORE this run compiled anything.
    # Pure filesystem, no device, and taken here rather than at the
    # announcement below because by then this run's own first kernels
    # are in it -- measured: a cache staged with 200 entries for
    # another card read as warm at the announcement, because 160 fresh
    # entries for THIS card had already been written into it.
    if kernel_cache_census is None:
        kernel_cache_census = scan_kernel_cache()
    total_started = time.perf_counter()
    runtime_source_identity = _runtime_source_identity()
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": "RESTORING_PREPARED_CACHE",
        "source": inputs.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": float(exp.run_seconds),
    }, heartbeat=True)

    # WHICH INITIALIZATION ROAD, decided before a byte is loaded.  The
    # decision is only ever made for a forecast that will actually stream:
    # ``exp.tiles`` unconfigured means ``decide`` is never called, no card is
    # queried, ``bundle`` stays None and everything below is the resident
    # route exactly as it was.  The decision is taken here rather than after
    # the restore because the whole point of the store road is that the
    # restore does not happen.
    # THE TABLE THAT GOVERNS THIS DOMAIN, resolved the way every other
    # surface resolves it.  This door read `exp.tiles` raw, so a domain
    # carrying its own `tiles = {...}` table was judged on the tree-wide
    # table here and on its own table at the review and at `woof run` --
    # one configuration, two answers.
    tiles_options = streaming.options_for_domain(
        exp.root, getattr(exp, "tiles", None))
    planning_machine = (None if getattr(getattr(exp, "devices", None), "enabled", False)
                        else streaming.cold_planning_machine(exp))
    # The card a resident or a pinned road is admitted on, read before the
    # restore: the planning machine when the planner already read it.
    admission_machine = (None if getattr(getattr(exp, "devices", None), "enabled", False) else
                         streaming.cold_admission_machine(
                             planning_machine, options=tiles_options))
    stream_head = getattr(inputs, "stream_head", None)
    from woof.core.device_probe import cuda_device_identity
    admission_identity = (None if admission_machine is None or not
                          _stream_producer_reserve(stream_head) else cuda_device_identity(0))
    planning_machine = _stream_reserved_machine(
        planning_machine, stream_head, identity=admission_identity)
    admission_machine = _stream_reserved_machine(
        admission_machine, stream_head, identity=admission_identity)
    tiles_options = _stream_reserved_options(
        tiles_options, admission_machine, stream_head, identity=admission_identity)
    # THE SHARED ADMISSION, not a second one.  This door used to price
    # from the restored cache's own retained lateral-boundary interval
    # count, which is the run's LEDGER question rather than the
    # admission question: the plan review that admitted this forecast
    # priced it from the configuration alone, so for any budget between
    # the two the review admitted a domain this door then refused with
    # the cache already restored.  Same function, same estimate, same
    # budget as woof check and woof run, and the same one guard over
    # whether an estimate is consulted at all.  Its boundary tables are
    # the ones the cache about to be restored carries.
    priced_boundary = _priced_boundary_source(inputs.cache_reader,
                                              inputs.source)
    from woof.core.urban_state import urban_columns_line
    urban_columns = single_urban_columns(inputs)
    urban_line = urban_columns_line(urban_columns)
    if urban_line is not None:
        print(f"prepared forecast: {urban_line}", flush=True)
    devices_admission = None
    if getattr(getattr(exp, "devices", None), "enabled", False):
        devices_admission = _admit_devices_forecast(
            exp, cfg, geography=inputs.static,
            forcing_intervals=_retained_interval_count(inputs.cache_reader),
            source=priced_boundary, stream_head=getattr(inputs, "stream_head", None),
            landuse=getattr(inputs, "landuse_identity", None))
        resident_estimate = stream_decision = None
        init_road, init_receipt = _devices_init_road(exp)
    else:
        resident_estimate = streaming.cold_single_domain_admission(
            exp, machine=planning_machine, options=tiles_options,
            source=priced_boundary, urban_columns=urban_columns)
        stream_decision = (streaming.cold_single_domain_decision(
            exp, machine=planning_machine, cfg=cfg, options=tiles_options,
            estimate=resident_estimate)
                           if tiles_options.enabled else None)
        # A PINNED TILING IS PRICED ON THIS CARD, before its buffers exist: the
        # decision above takes it from the configuration alone, and buffers
        # that cannot fit stopped in a CUDA out-of-memory while being built.
        streaming.admit_pinned_road(cfg, tiles_options, stream_decision,
                                    machine=admission_machine)
        init_road, init_receipt = _choose_stream_init_road(
            stream_init, decision=stream_decision, reader=inputs.cache_reader,
            cfg=cfg, resident_estimate=resident_estimate, device_memory=(
                None if admission_machine is None or
                not _stream_producer_reserve(stream_head, identity=admission_identity) else
                (int(admission_machine.vram_bytes), _free_device_bytes()[1])))
    if init_receipt is not None:
        print(f"prepared forecast: {init_receipt['why']}", flush=True)
    if init_road != "store":
        # A RESIDENT RELOAD IS ADMITTED TOO, before a byte is restored.  With
        # no [tiles] block (mode 'off') nothing above priced this domain, so
        # a cache too big for the card went straight into its constructor and
        # stopped in a CUDA out-of-memory.  Priced at the retained forcing
        # interval count this cache carries (one slot on a streamed head),
        # state and physics together; a decision that already admitted
        # itself (auto, a streamed tiling) passes untouched.
        streaming.admit_resident_road(
            exp, stream_decision, machine=admission_machine,
            forcing_intervals=(
                1 if getattr(inputs, "stream_head", None) is not None
                else _retained_interval_count(inputs.cache_reader)),
            source=priced_boundary, urban_columns=urban_columns,
            what="this prepared forecast, held resident on the card")

    boundary_source = None
    boundary_stream_receipt = None
    clock_guard = None
    stream_head = getattr(inputs, "stream_head", None)
    #: The model time of this process's last completed step (``None``
    #: before the first), which dates a seam wait.
    stepped = {"model_elapsed_seconds": None}
    seam_waits = None
    if stream_head is not None:
        from woof.ingest.boundary_stream import (
            PRODUCER_NAME, SeamWaits, WAIT_LOG_NAME, stream_dir,
            streamed_boundaries,
        )

        head_started = time.perf_counter()
        events = getattr(observer, "events", None)

        def wait_where(index):
            # Where the model stands while it waits: before its first
            # step in this process (``start``), or at a seam, with the
            # model time it reached.
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
            # The progress file carries the wait while it lasts and drops
            # it the moment it ends, so a reader never sees a stale wait
            # beside a stepping model.
            elapsed = stepped["model_elapsed_seconds"]
            payload = {
                "schema": PROGRESS_SCHEMA,
                "status": "INTEGRATING" if block is not None else "RUNNING",
                "source": inputs.source,
                "model_elapsed_seconds": float(elapsed or 0.0),
                "requested_run_seconds": float(exp.run_seconds),
            }
            if block is not None:
                payload["waiting"] = block
            _atomic_json(progress_path, payload, heartbeat=True)

        def say(line):
            print(line, file=sys.stderr, flush=True)

        # The model is at a seam and interval k is not there yet.  The
        # wait is said on the event stream, in progress.json and on the
        # supervisor heartbeat, with its cause (a source lead or the
        # preparation), so a watchdog times it by its own bound and a
        # reader sees what it waits for.
        seam_waits = SeamWaits(
            emit=None if events is None else events.emit,
            observer=observer, publish=publish_wait, model_time=wait_where,
            say=say, log_path=outdir / WAIT_LOG_NAME,
            producer_path=stream_dir(inputs.prepared_root) / PRODUCER_NAME)
        # The long step was derived from the head, which holds no boundary
        # interval, while a sealed run reads the crest-level winds of every
        # one (terrain_clock.clock_for_prepared_cache).  Each interval is
        # folded into that reading as it loads; one that moves the clock
        # ends this attempt (StreamedClockChanged) and main() runs the
        # forecast again on the seal.  The same guard as the tree runner's.
        if inputs.clock_basis is not None:
            from woof.prepared_domain_tree_forecast import StreamedClockGuard

            clock_guard = StreamedClockGuard(
                inputs.clock_basis,
                run_clock=inputs.physics_receipt.get("terrain_clock"),
                root_grid_id=int(exp.root.grid_id),
                run_seconds=float(exp.run_seconds))
        boundary_source = streamed_boundaries(
            inputs.prepared_root, head=stream_head,
            on_wait=seam_waits, validate=clock_guard,
            start_time=exp.start_time)
        boundary_stream_receipt = {
            "chained": True,
            "head_sha256": stream_head["head_sha256"],
            "head_decision": dict(stream_head.get("decision") or {}),
            "ready_at_start": boundary_source.intervals.ready_prefix(),
            "interval_count": len(boundary_source.intervals),
            # The lateral intervals held on the card: a chained forecast
            # streams them through one device slot (the resident admission
            # above prices exactly 1).  R15 reads it: streamed boundaries are
            # a default, and a run that held the whole schedule says so here.
            "resident_forcing_slots": 1,
        }

    if prestaged_boundaries is not None:
        boundary_source = prestaged_boundaries.for_forecast(inputs)
        if clock_guard is not None:
            boundary_source.intervals.validate = clock_guard
        if boundary_stream_receipt is not None:
            boundary_stream_receipt["ready_at_start"] = len(boundary_source.intervals)
            boundary_stream_receipt["prestage"] = prestaged_boundaries.receipt

    bundle = None
    if init_road == "store":
        # THE STORE-DIRECT ROAD.  ``store_from_prepared_cache`` reads the
        # cache one row slab at a time, attaches physics to a slab-height
        # state, and scatters the result into full-domain PINNED HOST arrays
        # -- the same store ``attach`` would have copied a resident domain
        # into, arrived at without ever allocating a domain-shaped device
        # array.  The three validations the resident road makes at this
        # instant are made here too, against the same caller-pinned digest:
        # what cannot be reproduced without a domain state is named in the
        # receipt rather than dropped in silence.
        from woof.ingest.prepared_store import store_from_prepared_cache

        _announce_kernel_compile(progress_path, inputs, exp,
                                step_log, kernel_cache_census)
        started = time.perf_counter()
        if getattr(getattr(exp, "devices", None), "enabled", False):
            cp.cuda.Device(exp.devices.device_ids()[0]).use()
        loader_memory = streaming.radiation_footprint(
            cfg, tiles_options, resident_estimate=resident_estimate,
            machine=planning_machine).prepared_memory
        loader_terms = None if loader_memory is None else loader_memory.fixed_terms()
        bundle = store_from_prepared_cache(
            inputs.prepared_cache_path,
            expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static,
            reader=(inputs.cache_reader if (boundary_source is not None
                    or inputs.cache_payload is not None)
                    else None),
            # A prestaged boundary handle makes this a streamed head, which
            # reads the cache through its reader: the retained payload is
            # refused there (prepared_store: "a streamed head cannot reuse a
            # sealed retained payload", standby --devices 8 on a rented box,
            # 2026-10-21), so the store road drops it and reads from disk.
            cache_payload=(None if boundary_source is not None else inputs.cache_payload),
            rows_per_slab=(None if loader_terms is None else loader_terms["loader_rows"]),
            retain_template_rows=(None if loader_terms is None else loader_terms["template_rows"]),
            column_chunk=exp.column_chunk,
            boundary_source=boundary_source,
            landuse_attrs=inputs.landuse_identity, grid=inputs.grid,
            valid_time=exp.start_time,
            # The user's own pinned-host budget when [tiles] carries one, so
            # the store's pre-allocation guard refuses against the number the
            # operator set rather than against the machine's whole RAM.
            budget_bytes=getattr(tiles_options, "host_budget_bytes", None),
            constant_glw_wm2=declared_constant_glw(exp),
            physics_initializer=initialize_cached_physics,
            log=lambda line: print(line, flush=True))
        # Named for the resident road's key, because it is the same work
        # measured: get the prepared case off disk and onto the machine.
        timing["restore_prepared_cache"] = time.perf_counter() - started
        timing["initialize_physics"] = 0.0
        step_log.phase("restore_prepared_cache",
                       timing["restore_prepared_cache"], road="store")
        _validate_store_bundle_receipt(
            bundle.receipt, inputs.cache_reader.content_sha256)
        _validate_store_direct_vertical_contract(
            inputs.cache_reader, cfg, bundle)
        if not inputs.cache_reader.metadata.get("surface_fields"):
            raise ValueError(
                "prepared cache has no source-neutral canonical surface state")
        if bundle.boundaries is None:
            raise ValueError(
                "prepared cache carries no lateral boundary series; a "
                "specified root cannot be integrated without one")
        # ``restored.metadata`` is the cache's ``metadata['user']`` block, and
        # the reader this route already holds is where it comes from -- so the
        # same optional source-adapter hint is cross-checked on both roads
        # without the store loader having to carry it.
        _validate_restored_source_adapter(
            inputs.cache_reader.metadata.get("user", {}), inputs.source)
        # The SLAB-HEIGHT template's driver.  Its scheme selection, its
        # radiation carrier policy and its per-carrier provenance are
        # properties of the configuration and identical at every height; its
        # ARRAYS are one slab's and are never read for a domain quantity (see
        # the receipt's swdown/rainnc reductions, which come off the store).
        driver = bundle.template.physics
        domain_state = bundle.template
    else:
        started = time.perf_counter()
        restored = restore_prepared_cache(
            inputs.prepared_cache_path,
            expected_identity=inputs.cache_identity,
            cfg=cfg, static=inputs.static,
            reader=_resident_prepared_reader(inputs, boundary_source),
            boundary_source=boundary_source)
        if clock_guard is not None:
            from woof.ingest.boundary_stream import keep_interval_check

            # The attachment above installs its own per-interval layout
            # check as the series' hook, replacing the clock guard handed
            # to the stream, so without this only interval 0 (read before
            # the attachment) was folded into the clock.  Measured on a
            # 6-interval GFS head-bound run: clock_intervals_checked 1.
            keep_interval_check(boundary_source.intervals, clock_guard)
        timing["restore_prepared_cache"] = time.perf_counter() - started
        step_log.phase("restore_prepared_cache",
                       timing["restore_prepared_cache"], road="resident")
        _validate_restored_cache_receipt(
            restored.receipt, inputs.cache_reader.content_sha256)
        if restored.surface is None:
            raise ValueError(
                "prepared cache has no source-neutral canonical surface state")
        _validate_restored_source_adapter(restored.metadata, inputs.source)

        _announce_kernel_compile(progress_path, inputs, exp,
                                step_log, kernel_cache_census)

        started = time.perf_counter()
        driver = initialize_cached_physics(
            restored.initial_result, cfg, restored.met, restored.surface,
            inputs.static, inputs.landuse_identity, inputs.grid,
            exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp))
        timing["initialize_physics"] = time.perf_counter() - started
        step_log.phase("initialize_physics", timing["initialize_physics"])
        domain_state = restored.initial_result.state

    if exp.perturbation is not None:
        # Treatment proof, before any integration: the accepted config
        # echoed value for value plus what the restored domain received
        # (summed over the store road's slabs).
        receipts = [{
            "resumed": True,
            "note": "resumed from checkpoint; the initial perturbation was "
                    "applied when the trajectory began and is not re-applied",
        }]
        if perturbation is not None:
            receipt = dict(perturbation_rows[0])
            receipt["bubbles"] = [dict(row) for row in receipt["bubbles"]]
            for index, row in enumerate(receipt["bubbles"]):
                row["cells_touched"] = sum(
                    r["bubbles"][index]["cells_touched"]
                    for r in perturbation_rows)
                if row.get("applied") and row["cells_touched"] == 0:
                    raise ValueError(
                        f"perturbation.bubbles #{index + 1} touches zero "
                        "cells in domain d01")
                for key in ("max_theta_added_k", "max_qv_delta_kg_kg"):
                    values = [r["bubbles"][index][key]
                              for r in perturbation_rows
                              if key in r["bubbles"][index]]
                    if values:
                        row[key] = max(values)
            receipts = [receipt]
        runtime._write_initial_perturbation_receipt(outdir, exp, receipts)

    if prewarmed_execution is None:
        tick_clock = resolve_clock(
            exp, lbc_interval_s=float(inputs.boundary_interval_seconds))
        schedule = build_schedule(exp, tick_clock)
    else:
        tick_clock, schedule = prewarmed_execution.clock, prewarmed_execution.schedule
    clocks = tick_clock.clocks()
    # ON THE STORE ROAD ``node.state`` IS THE SLAB-HEIGHT TEMPLATE, and this
    # is the least-invasive correct answer rather than a convenience.
    # ``execute_experiment`` requires a state object for three things and only
    # three: ``refresh_model_time`` writes ``elapsed_seconds`` onto it,
    # ``impose_clock`` reads that value back, and the stepper is called with
    # it (``StreamedDomain.__call__`` accepts it and, attached without a
    # state, checks nothing against it).  The template satisfies all three,
    # costs one slab, and carries the physics driver the REFL_10CM handoff and
    # the writer's carrier provenance need.  What it must NEVER be used for is
    # a domain-sized READ -- its arrays are 64 rows of the analysis -- and
    # every such reader on this route is routed to the store below, by name.
    node = _single_prepared_root(
        exp.root, inputs.grid, domain_state, clocks[1],
        store_direct=bundle is not None)
    checkpoint_identity = _single_checkpoint_identity(inputs, runtime_source_identity)
    fingerprint = hashlib.sha256(
        _canonical(checkpoint_identity).encode("utf-8")).hexdigest()
    model = ExperimentState(
        node, MappingProxyType({1: node}), schedule, None, fingerprint)
    model._experiment_fingerprint_components = checkpoint_identity
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._scratch_arena = None
    model._dycore_state_workspace = None
    model._io_manager = None
    model._last_checkpoint = None
    model._prepared_by_grid_id = MappingProxyType({
        # The wrfout writers read exactly two things off this: the static
        # fields for the metadata frame and ``initial_result.coord`` for the
        # global attributes.  The store road has no ``CachedInitialResult``
        # -- there is no domain state to put in one -- and both of those are
        # domain-invariant, so the bundle supplies them directly.
        1: SimpleNamespace(
            static_fields=inputs.static, geog_selection=None,
            streamed_store=bundle,
            initial_result=(
                restored.initial_result if bundle is None
                else SimpleNamespace(coord=bundle.coord, base=bundle.base,
                                     state=None))),
    })

    from woof.ensemble.runtime_context import bind_current_member_model
    bind_current_member_model(model)

    if bundle is None:
        initial_health = _strict_json(vars(
            StateHealthValidator(node.state).validate(
                phase="initialized.d01")))
        if not initial_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast initial health failed: {initial_health}")
    else:
        # ARMED ON THIS ROAD TOO.  The gate needs a DOMAIN, not a device: the
        # CUDA validator is one block per whole field and cannot be pointed at
        # the slab-height template without reporting one slab as the analysis,
        # but the domain is right here in the pinned host store, and the rule
        # set the kernel applies is the same one ``validate_fields_cpu`` reads
        # out of ``rule_for_field``.  So the per-field bounds -- the moisture
        # ranges, the coupled-mass positivity, the geopotential and specific-
        # volume limits -- are checked over the whole 1024 x 1024 analysis
        # rather than declared out of reach.  A failure is terminal here
        # exactly as it is on the resident road.
        initial_health = _store_full_state_health(
            bundle, cfg, phase="initialized.d01")
        if not initial_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast initial health failed: {initial_health}")
        coverage = initial_health["coverage"]
        print("prepared forecast: store-direct initialization -- the "
              "initialized.d01 full-state health gate IS armed, over the "
              f"store: {coverage['fields_checked']} fields, "
              f"{coverage['bytes_checked'] / _GIB:.2f} GiB, in "
              f"{coverage['seconds']:.1f}s"
              + ("" if not coverage["not_in_store"] else
                 "; gated fields the store does not carry: "
                 + ", ".join(coverage["not_in_store"])), flush=True)

    from woof.ensemble.runtime_context import (initialized_bootstrap_handoff,
        observe_current_counters, observe_initial_diagnostics)
    ensemble_report = initialized_bootstrap_handoff(
        ensemble_bootstrap, inputs=inputs, model=model, node=node,
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

    writers = PerDomainWrfoutWriters(
        model, outdir / "wrfout", start_time=exp.start_time,
        title=(f"woof {inputs.source.upper()} prepared-cache "
               f"{cfg.nx}x{cfg.ny}x{cfg.nz}"),
        # The wrfout outlives the run directory: it is what gets
        # archived, what `woof downscale` reads back, and what the
        # pictures are made from.  `report.json` alone is not where a
        # forecast-lead initialization may live -- separate the artifact
        # from its run directory and that provenance is gone.
        initial_condition=inputs.proof.get("initial_condition"),
        source=inputs.source,
        # The tree-wide [output] history selection; a single-domain run's
        # own `output = {...}` on its [[domain]] table overrides it inside
        # the writer set, exactly as it does on a tree.
        history_selection=exp.output,
        simulated_radar=exp.simulated_radar, radar_output_dir=outdir)
    model._io_manager = writers
    forecast_started = time.perf_counter()

    def history_handler(_model, current, ticks):
        # ASKED OF THE STEPPER.  ``steppers`` is bound below and read here at
        # CALL time, which is after ``streaming.steppers_for_tree`` has run.
        # It matters because the gate two lines down --
        # ``stability_gate_failed`` at CFL 10 and w 150 m/s -- is this
        # route's safety net, and under ``[tiles] store = "host"`` the
        # domain is in a pinned host store while ``current.state`` is the
        # copy taken at t = 0 that the sweep never writes.  Reducing over it
        # meant the net reported the INITIAL CFL at every history frame,
        # forever, on a run that could already have gone non-finite.
        # ``stability_observer`` returns ``dycore.stability_report`` itself
        # for a resident domain, so nothing changes for a run that
        # configures no [tiles].
        stepper = steppers.get(int(current.cfg.grid_id))
        # THE ANALYSIS FRAME ON THE STORE-DIRECT ROAD.  Before the first
        # sweep there is no fold, and ``StreamedStability`` answers that one
        # instant from the resident state instead -- correct for a domain
        # attached FROM a state, because attach copied it and neither has
        # moved since.  Here the only state is the slab-height template, so
        # that same branch would report the last slab's maxima as the
        # domain's: a plausible number that is not this domain's, in the
        # receipt, on the exact axis the fold exists to stop lying about.
        # Refused by name instead, and the gate is skipped for this one frame
        # rather than run on the wrong memory -- there is nothing to catch
        # yet, because nothing has been integrated.
        if bundle is not None and int(getattr(stepper, "steps", 0)) == 0:
            report = {
                "folded": False,
                "why": ("the store-direct road has no domain-shaped state to "
                        "reduce and the per-tile fold has no record before "
                        "the first sweep, so the analysis frame carries no "
                        "stability sample rather than the slab template's"),
            }
            gate_failed = False
        else:
            report = streaming.stability_observer(stepper)(
                current.state, current.cfg.run,
                boundary_width=current.cfg.run.spec_bdy_width)
            gate_failed = stability_gate_failed(
                report, max_cfl=10.0, max_w_ms=150.0)
        sample = {
            "ticks": int(ticks),
            "elapsed_seconds": float(current.clock.elapsed_seconds),
            **report,
        }
        history.append(_strict_json(sample))
        if gate_failed:
            raise FloatingPointError(
                "prepared forecast stability threshold failed: "
                + _stability_diagnosis(sample, current.state, current.cfg.run)
                + f"; sample {sample}")
        refl = _consume_due_native_refl_10cm(
            current.state, ticks, consume_refl_10cm,
            domain_start_ticks=domain_start_ticks_of(current),
            shape=(current.cfg.run.nz, current.cfg.run.ny, current.cfg.run.nx))
        writers.submit(current, ticks, refl_field=refl)
        # History-interval reset of the UP_HELI_MAX window
        # (module_diag_nwp.F:246-269; woof's ratified placement is
        # immediately after the frame is durable).  Every other route
        # that publishes history frames does this -- runtime.py's
        # prepared-case integrator, the downscale child, and the tree
        # runner through runtime._submit_tree_history_frame -- and THIS
        # one, the runner `woof go` uses, did not.  The running max
        # therefore never restarted: frame 2 onward reported the maximum
        # since model start rather than since the previous frame, so
        # every value after the first history period was wrong, and
        # wrong in the direction that never comes back down.
        #
        # Safe ordering is submit's, unchanged: its producer-stream
        # wait_event fences the side-stream D2H snapshot ahead of any
        # later default-stream mutation, so zeroing here cannot race the
        # staged copy.  Only a FULL frame closes the window: a quick-look
        # frame reports the running maximum and leaves it running.
        reset_history_windows_after_frame(current, ticks)
        memory_watch.sample()

    def progress_callback(**event):
        # The external observer first and unconditionally: this
        # runner's own publication is throttled to every 60th step, and
        # a caller driving the run in-process wants the cadence, not the
        # sample of it.
        observe_current_counters(model, start_time=exp.start_time)
        if observer is not None:
            observer(**event)
        outer_step = int(event["outer_step"])
        elapsed = float(event["model_elapsed_seconds"])
        if not str(event.get("phase", "")).startswith("stepping:"):
            # A committed step, not the announcement of the first one.
            stepped["model_elapsed_seconds"] = elapsed
        if outer_step == 1 or outer_step % 60 == 0 \
                or elapsed == float(exp.run_seconds):
            memory_watch.sample()
            durable_paths = writers.paths
            _atomic_json(progress_path, {
                "schema": PROGRESS_SCHEMA,
                "status": "RUNNING",
                "source": inputs.source,
                "model_elapsed_seconds": elapsed,
                "outer_step": outer_step,
                "requested_run_seconds": float(exp.run_seconds),
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                    "cuda_device_used"),
                "last_checkpoint": event.get("last_checkpoint"),
                "last_durable_wrfout": (
                    None if not durable_paths
                    else str(durable_paths[-1].resolve())),
            }, heartbeat=True)
        if seam_waits is not None and elapsed < float(exp.run_seconds):
            if clock_guard is not None:
                from woof.ingest.boundary_stream import keep_interval_check

                # Chained again after any later re-attachment (a no-op
                # while it is still in place), before the next interval
                # loads.
                keep_interval_check(boundary_source.intervals, clock_guard)
            _require_next_interval(boundary_source.intervals, elapsed,
                                   writers=writers, exp=exp)

    # Here and not at the writers' construction above: the closure this
    # binds reads `writers.paths`, so it cannot exist before the writers
    # do.  Still ahead of every submit -- the first one happens inside
    # execute_experiment below -- and attach_progress_callback refuses
    # outright if that ever stops being true.
    #
    # ONE attach, with both consumers behind it.  attach_progress_callback
    # overwrites `landing_observer` rather than appending to it, so
    # calling it twice would silently unhook the host's observer the
    # moment this runner armed a render of its own.
    #
    # No root-domain guard, unlike RunObserver.output_committed: this
    # runner integrates exactly one domain (preflight replaces a
    # multi-domain experiment with `(source_exp.root,)`), so there is one
    # writer, it is the root's, and a guard here would be a test that
    # cannot fail.  The tree runner is where that guard earns its keep.
    #
    # `getattr` for the observer's hook, exactly as attach_progress_callback
    # itself reads it: this runner's contract has always been that an
    # observer without `output_committed` is fine and simply hears
    # nothing about landings.
    landing = _LandingObservers(
        getattr(observer, "output_committed", None),
        # The per-step log's own frame hook.  It rides HERE rather than
        # anywhere else because this is the one call raised after the
        # frame is fsynced, self-validated and renamed onto its final
        # name -- which is exactly the instant a "this frame is safe to
        # read" marker is allowed to be published.
        step_log.output_committed if step_log.enabled else None,
        None if first_products is None else first_products.frame_committed,
        # The column store this runner owns when nobody hosts it (the
        # subprocess route of `woof go`): _route_owned_columns.
        _columns_sink(columns),
        on_submit=step_log.frame_submitted if step_log.enabled else None)
    if landing:
        writers.attach_progress_callback(landing)
    # Each history write between two steps beats on the run's heartbeat;
    # the landing fan-out above does not carry it.
    writers.attach_write_progress(observer)

    # [tiles]; see the same call in prepared_domain_tree_forecast.  An
    # experiment that does not configure it gets {} and the executor's own
    # dycore.step, unchanged -- builders_for_tree returns {} on the same
    # test and imports nothing either.
    #
    # The builders are the half of this seam that was missing.  Without them
    # steppers_for_tree refused every configuration it was given ("this
    # route wired no streamed-domain builder"), so [tiles] mode = "on"
    # was configurable and unreachable at the same time: no forecast this
    # CLI can launch was capable of streaming.
    #
    # `streaming_decisions` is the OTHER half.  The stepper dict cannot say
    # which way `auto` went: a grid that declined to stream is absent from
    # it, and absent is what an unconfigured run looks like too.  So an
    # `auto` forecast that quietly ran resident would be indistinguishable
    # from one that streamed -- including to a test pointed at it.  The
    # decisions are recorded per grid, reported in the receipt below as
    # report["tiles"], and printed as one line.
    streaming_decisions: dict = {}
    devices_history = []
    devices_peaks = {}
    if getattr(getattr(exp, "devices", None), "enabled", False):
        devices_admission = _admit_devices_forecast(
            exp, cfg, geography=bundle.geography,
            forcing_intervals=_retained_interval_count(inputs.cache_reader),
            inventory=bundle.store, source=priced_boundary, template=bundle.template,
            boundaries=bundle.boundaries,
            stream_head=getattr(inputs, "stream_head", None),
            landuse=getattr(inputs, "landuse_identity", None),
            static=inputs.static)
        steppers = _devices_stepper(bundle, node, exp, streaming_decisions,
                                   admission=devices_admission)
    else:
        steppers = streaming.steppers_for_tree(
            model, exp.tiles, machine=planning_machine, resident_estimate=resident_estimate,
            # ``store_domain_builder`` is ``prepared_domain_builder``'s
            # counterpart for a domain that was never resident: it reads the
            # geography, the lateral tables and the tile-state template off the
            # BUNDLE instead of off ``node.state``, and calls ``attach`` with the
            # store already built and no state at all.
            #
            # UNPARENTHESISED, deliberately.  The resident half must read as the
            # literal ``builders=streaming.builders_for_tree(``: that string is
            # what tests/test_streaming.py greps both production routes for, and
            # it greps rather than calls because the defect it guards was an
            # ABSENT argument -- a route that stops passing builders refuses
            # every streaming configuration and behaves identically otherwise.
            # Wrapping this expression in one extra paren was enough to make the
            # guard stop seeing the call it is guarding.
            builders=streaming.builders_for_tree(model, exp.tiles)
            if bundle is None
            # ``clock=node.clock`` IS NOT OPTIONAL and the builder refuses
            # without it.  ``attach(None, ...)`` has no DomainState, so the
            # lazy binding every other road relies on derived None here, latched
            # on the first buffer conversion and put the whole store-direct
            # forecast on the retired elapsed-seconds Davies recurrence -- the
            # #219 one-timestep phase error, reopened on the road the LARGEST
            # domains take.  This is the domain's own clock, the same object
            # ``bind_lateral_boundary_clock`` binds to the resident mirror.
            else {int(node.cfg.grid_id): streaming.store_domain_builder(
                bundle, clock=node.clock)},
            decisions=streaming_decisions)
    streaming_report = ({} if getattr(getattr(exp, "devices", None), "enabled", False)
                        else streaming.streaming_receipt(exp.tiles, streaming_decisions))
    if streaming_report:
        print(f"prepared forecast: {streaming_report['summary']}", flush=True)
    if bundle is not None:
        # ``decide`` was consulted twice -- once above to choose the road,
        # once inside ``steppers_for_tree`` -- and under ``auto`` it is a
        # function of the free VRAM at the instant it was taken, so the two
        # can legitimately disagree on a card whose memory moved in between.
        # On this road a disagreement is fatal rather than cosmetic: the
        # domain is in the store and there is no resident state for
        # ``dycore.step`` to integrate, so a second answer of "resident"
        # would step the slab template and publish it as the forecast.
        # Checked, not assumed.
        if int(node.cfg.grid_id) not in steppers:
            raise RuntimeError(
                "the store-direct road built this domain straight into a "
                "pinned host store, and the streaming seam then decided this "
                "grid should run RESIDENT -- there is no resident domain to "
                "run.  [tiles] mode = 'auto' re-decides against the free VRAM "
                "at the instant it is asked; pin the mode ([tiles] mode = "
                "'on') for a run that must stream.")
        # THE MARKER, and it is what routes three whole-domain consumers to
        # the store instead of to the template.  ``attach`` sets it on the
        # state it takes over, and a store-direct attach has no such state --
        # so it is set here, on the object the model actually holds:
        # ``PerDomainWrfoutWriters.submit`` asks the state for it before
        # taking a history frame (without it the frame would be the slab
        # template's), and ``streaming.live_scratch`` follows it to the
        # store's ``scratch/up_heli_max`` so the history-interval UH reset
        # lands on the accumulator the tiles are actually folding into.
        node.state._streamed_domain = steppers[int(node.cfg.grid_id)]

    if exp.nudging is not None:
        if bundle is not None:
            raise ValueError("NUDGING_FULL_HORIZONTAL_FFT: nudging needs a resident domain; tile-local FFTs change the requested wavelengths")
        from woof.spectral_nudging.prepared import allocate_held_tendencies, selected
        allocate_held_tendencies(node.state, cfg, selected(exp))
    with _checkpoint_restore_resources(writers, step_log):
        restart_info = _restore_single_checkpoint(model, restart)
        from woof.runtime import _restart_is_complete
        already_complete = _restart_is_complete(restart_info)
        restored_seconds = (None if restart_info is None else
                            float(node.clock.ticks / schedule.clock.tick_den))
        if restart_info is not None:
            from woof.core.health import health_validator_for_domain
            initial_health = _strict_json(vars(health_validator_for_domain(
                model, node).validate(phase="restored.d01")))
            if not initial_health["ok"]:
                raise FloatingPointError(
                    f"prepared forecast restored health failed: {initial_health}")

    # RADAR LATENT HEATING ([radar_heating], woof.da.forecast_heating).
    # Attached here, after the restore and before integration: every slot is
    # built against the state the run starts from, the windows were held to
    # the run's grid identity first, and a resume inside the forced period
    # is refused.  Off, this is one attribute test and nothing else.
    radar_heating_hook = None
    if not already_complete:
        from woof.da.forecast_heating import attach_forecast_heating
        radar_heating_hook = attach_forecast_heating(
            model, node, exp, prepared_root=inputs.prepared_root,
            steppers=steppers, restored_seconds=restored_seconds,
            checked=radar_heating_checked)

    nudging_hook = None
    if not already_complete and exp.nudging is not None:
        from woof.spectral_nudging.prepared import PreparedNudging, selected
        if selected(exp) is not None:
            started = time.perf_counter()
            nudging_hook = PreparedNudging(node.state, cfg, exp,
                inputs.prepared_root, inputs.cache_identity['source_manifest_sha256'])
            timing['nudging_setup'] = time.perf_counter() - started

    checkpoints = []
    checkpoint_ticks = {}
    checkpoint_publisher = _CheckpointPublisher()

    def restart_handler(tree, ticks):
        from woof.io.restart import write_tree_restart
        # At most one checkpoint in flight; a failed one fails the run here.
        checkpoint_publisher.wait()
        valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        started = time.perf_counter()
        # Snapshotted between two model steps, at the stop tick too: its own
        # record, sized from the state it writes, or the supervisor times it
        # as a step (see runtime._writing_progress).
        with runtime._writing_progress(
                observer, "checkpoint",
                work_bytes=runtime._checkpoint_work_bytes(tree)):
            # This door publishes no declared experiment, so it names the
            # model-chosen epssm itself: a child downscaled from the
            # checkpoint inherits that label (A181).
            pending = write_tree_restart(
                outdir, tree, valid, auto_epssm=exp.auto_epssm,
                defer_publish=True)
        # Taken now, on this thread: the frames up to this valid time.
        try:
            history_durable = writers.durability_barrier()
        except BaseException:
            pending.abandon()
            raise

        def publish(retire=True):
            # A checkpoint is published only after every history frame up
            # to its valid time is durable, so a resume never starts past a
            # frame that is missing.
            try:
                if history_durable is None:
                    writers.drain()
                else:
                    history_durable()
                path = pending.publish(retire=retire)
                checkpoints.append(str(Path(path).resolve()))
                checkpoint_ticks[int(ticks)] = checkpoints[-1]
                step_log.restart_written(
                    domain=exp.root.grid_id, valid_time=valid, path=path,
                    wall_seconds=time.perf_counter() - started)
            except BaseException:
                # History can fail before publish() gets to release the
                # borrowed ranked store. Never leave its next download locked.
                pending.abandon()
                raise

        if pending.deferrable and history_durable is not None:
            try:
                # Superseded sets are retired on this thread at the next
                # wait (see _CheckpointPublisher), never by the writer.
                checkpoint_publisher.submit(
                    lambda: publish(retire=False),
                    then=pending.retire_superseded)
            except BaseException:
                pending.abandon()
                raise
            return
        publish()

    # The executor's restart hook drains the history writers before calling
    # a handler; this one orders the checkpoint after the history itself.
    restart_handler.orders_history_itself = True
    restart_handler.wait_durable = checkpoint_publisher.wait

    # Armed immediately before integration and disarmed by the first
    # completed step: the compile this names is paid inside step 1, and
    # a reader watching a silent terminal has no other way to learn
    # that anything is happening at all.
    stall_watch = _FirstStepStallWatch(
        progress_path=progress_path, inputs=inputs, exp=exp,
        step_log=step_log, census=kernel_cache_census, road=init_road)
    committed_step_observer = stall_watch.wrap(
        step_log.step_observer if step_log.enabled else None)
    if getattr(getattr(exp, "devices", None), "enabled", False):
        committed_step_observer = _devices_step_observer(
            node, steppers[int(node.cfg.grid_id)], exp.devices,
            devices_history, devices_peaks, committed_step_observer)
    history_frames: list = []
    try:
        memory_watch.start()
        stall_watch.arm()
        with writers:
            if already_complete:
                from woof.prepared_domain_tree_forecast import _completed_execution_report
                execution = _completed_execution_report(model)
                stepping_ended = time.perf_counter()
            else:
                execution = execute_experiment(
                    model, history_handler=history_handler,
                    restart_handler=restart_handler,
                    progress_callback=progress_callback,
                    # The common validator follows the attached canonical store.
                    # Preserve the default store cost; explicit diagnostics enable it.
                    validate_state=(bundle is None or health_debug),
                    health_debug=health_debug,
                    skip_feedback_path=True,
                    steppers=steppers,
                    # ONE line per model time step, WRF's own bar.  Handed
                    # the bound method rather than a wrapper so the log is
                    # what the executor calls, and None when the log is off
                    # so a silenced run pays nothing per step.
                    step_observer=committed_step_observer,
                    experiment=exp)
            stepping_ended = time.perf_counter()
            # The last checkpoint is durable, or its failure is raised,
            # before anything below reports the run.
            checkpoint_publisher.wait()
            runtime._finalizing_progress(observer, "synchronize-device")
            cp.cuda.Stream.null.synchronize()
            timing["forecast_execution_with_async_io"] = (
                time.perf_counter() - forecast_started)
            drain_started = time.perf_counter()
            writers.drain(before_domain=runtime._drain_progress(observer))
            timing["final_writer_drain"] = time.perf_counter() - drain_started
            wrfout_paths = writers.paths
            history_frames = writers.frame_timings()
            for row in history_frames:
                print(f"history frame {row['path']}: stepping thread "
                      f"{row.get('submit_s')} s; writer: queued "
                      f"{row.get('queued_s')} s, copy+assemble "
                      f"{row.get('copy_and_assemble_s')} s (D2H wait "
                      f"{row.get('copy_wait_s')} s), write "
                      f"{row.get('write_s')} s, durable after "
                      f"{row.get('durable_after_s')} s", flush=True)
        timing["forecast_and_io_inclusive"] = (
            time.perf_counter() - forecast_started)
    except BaseException as error:
        _mark_failure_clock(error, node.clock)
        try:
            # A checkpoint already snapshotted still lands, so the failure
            # report and a relaunch can use it.
            checkpoint_publisher.wait()
        except BaseException as late:  # noqa: BLE001 - keep the first failure
            print("prepared forecast: the last checkpoint could not be "
                  f"written ({type(late).__name__}: {late})",
                  file=sys.stderr, flush=True)
        from woof.ingest.boundary_stream import SourceBehind
        stopped = None
        if isinstance(error, SourceBehind):
            # The same refusal, now carrying the seam checkpoint and where
            # the model stopped; it is the one `main` reports.
            stopped = _stop_at_seam(
                error, model=model, node=node, exp=exp, schedule=schedule,
                restart_handler=restart_handler,
                checkpoint_ticks=checkpoint_ticks, checkpoints=checkpoints,
                seam_waits=seam_waits)
            _mark_failure_clock(stopped, node.clock)
        # The last line a driving script reads has to say what happened.
        # Closed here, before the exception continues to `main`'s report
        # writer, because that writer can itself fail and the log must
        # still have terminated.
        said = error if stopped is None else stopped
        step_log.close(status="FAIL",
                       error=f"{type(said).__name__}: {said}")
        if stopped is not None:
            raise stopped from error
        raise
    finally:
        # A forecast that never reached step 1 must not leave a timer
        # thread waiting to announce a compile for a run that is over.
        stall_watch.disarm()
        memory_watch.stop()
        model._io_manager = None
        if radar_heating_hook is not None:
            radar_heating_hook.detach()
        if nudging_hook is not None:
            nudging_hook.detach()
    # After the drain: every frame this run will ever commit is durable
    # and has its marker, so the run_end record is true when it is read.
    step_log.close(status="SUCCESS")
    memory_watch.sample()

    cadence_seconds = float(exp.root.history_interval_s)
    cadence_receipt = _validate_hash_bound_history_cadence(
        exp, cadence_seconds)
    begin_s, end_s = _root_history_window(exp)
    output_schedule = _history_output_schedule(
        start_time=exp.start_time, run_seconds=exp.run_seconds,
        cadence_seconds=cadence_seconds, after_seconds=restored_seconds,
        begin_seconds=begin_s, end_seconds=end_s,
        interval2_seconds=exp.root.run.history_interval2_s,
        change_seconds=exp.root.run.history_interval_change_s)
    expected_frames = len(output_schedule)
    from woof.ensemble.runtime_context import current_capture
    capture = current_capture()
    cadence_paths = (writers.captured_paths
                     if capture is not None and not capture.keep_member_files
                     else wrfout_paths)
    if len(cadence_paths) != expected_frames:
        raise RuntimeError(
            f"history handoff published {len(cadence_paths)} frames, "
            f"expected {expected_frames}")
    expected_names = tuple(record[2] for record in output_schedule)
    if tuple(path.name for path in cadence_paths) != expected_names:
        raise RuntimeError(
            "WRF history output filenames/cadence differ from the "
            "hash-bound request")

    # The final health gate validates node.state and CANNOT be folded:
    # StateHealthValidator is one block per whole field with no windowing, so
    # it has no tile-interior form.  Under a host store that state is the
    # snapshot that filled the store, so the gate used to pass on the t = 0
    # ANALYSIS -- and so did canonical_state_digest below it: a receipt
    # reporting nan_free on a run that had integrated for an hour, and a
    # final digest that was the initial condition.  The note was here and
    # the fix was not; this is the fix.  Zero and a getattr on a resident
    # run, so nothing changes for a forecast that configures no [tiles].
    runtime._finalizing_progress(observer, "final-health")
    digest_thread = digest_box = None
    early_fingerprint = early_devices_receipt = cards_released = None
    final_chem_ledger = None
    if bundle is None:
        carriers_refreshed = streaming.refresh_streamed_state(
            steppers.get(int(node.cfg.grid_id)), node.state)
        final_health = _strict_json(vars(
            StateHealthValidator(node.state).validate(phase="final.d01")))
        final_stability = _strict_json(streaming.stability_observer(
            None if already_complete else steppers.get(int(node.cfg.grid_id)))(
                node.state, cfg, boundary_width=cfg.spec_bdy_width))
        if not final_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast final health failed: {final_health}")
        started = time.perf_counter()
        final_digest = canonical_state_digest(
            node.state, node.clock, scope="trajectory",
            before_hash=runtime._digest_progress(observer, node.cfg.grid_id))
        timing["canonical_final_state_digest"] = time.perf_counter() - started
    else:
        # THE STORE-DIRECT FINAL READS, none of which may touch node.state.
        # There is nothing to refresh: ``refresh_state`` copies the store onto
        # a resident domain, and this road never built one -- calling it would
        # refuse on the first carrier's shape (domain against slab), which is
        # the right refusal and the wrong question.
        stepper = steppers[int(node.cfg.grid_id)]
        final_chem_ledger = _chem_ledger_receipt(
            stepper.template_state, store=stepper._planning_store())
        carriers_refreshed = 0
        # The last sweep's record, folded per tile out of the STORE and
        # bit-equal to what ``stability_report`` returns for the same domain
        # integrated resident: max is associative and exact, the NaN classes
        # are a bitwise OR, and the argmax carries a DOMAIN flat index.  It
        # RAISES rather than returning a stale record if the sweep produced
        # none, which is the behaviour a final gate must have.
        final_stability = (
            {"available": False, "reason": "checkpoint is already at the stop; no new sweep"}
            if already_complete else _strict_json(stepper.health))
        # TWO INSTRUMENTS, BOTH ARMED, over the same store the sweep has been
        # writing.  The fold answers the question a run loop has to ask every
        # step and answers it cheaply; the descriptor gate answers the one the
        # resident road asks HERE, at the same instant, over the same fields
        # and under the same per-field bounds.  Neither substitutes for the
        # other and neither is inferred from the other's silence.
        final_health = _store_full_state_health(bundle, cfg, phase="final.d01")
        final_health["stability_fold"] = {
            "ok": None if already_complete else not bool(final_stability.get("nan")),
            "available": not already_complete,
            "instrument": "streamed store fold (StreamedDomain.health)",
            "covers": ("u, w and theta-perturbation finiteness and the two "
                       "Courant terms, over the whole domain in the store"),
        }
        if not already_complete and not final_health["stability_fold"]["ok"]:
            raise FloatingPointError(
                f"prepared forecast final stability fold failed: "
                f"{final_health['stability_fold']}; "
                f"stability {final_stability}")
        if not final_health["ok"]:
            raise FloatingPointError(
                f"prepared forecast final health failed: {final_health}; "
                f"stability {final_stability}")
        # THE DIGEST MUST BE TAKEN OVER THE STORE, or it is not this run's.
        # ``canonical_state_digest(node.state, ...)`` would hash the
        # slab-height template -- the right shape of answer over 1/16th of the
        # analysis -- and the receipt would carry a trajectory digest for a
        # trajectory that never happened.  Refused by name rather than
        # substituted, because a wrong digest is indistinguishable from a
        # right one until two runs are compared.
        digest_from_store = getattr(stepper, "canonical_digest", None)
        if digest_from_store is None:
            raise RuntimeError(
                "this build's StreamedDomain has no canonical_digest(clock, "
                "scope=...), so a store-direct run has no way to produce its "
                "final trajectory digest over the memory it integrated.  "
                "woof.state_digest.canonical_state_digest is NOT a fallback "
                "here: the only state this road holds is the slab-height "
                "template, and digesting it would publish a hash of one slab "
                "of the analysis as the forecast's trajectory.")
        capture_inputs = getattr(stepper, "canonical_digest_inputs", None)
        if getattr(stepper, "ranked", False) and capture_inputs is not None:
            # CARDS FIRST, HASHING AFTER.  Everything the end of the run still
            # reads off the cards is taken now (the digest's arrays, the
            # setup fingerprint, the devices receipt); then the ranks close
            # and every card's pool is freed, and the digest, the seal and
            # the input verify run on the host with the cards already free.
            # Breakage this prevents: V59 (8 x RTX 5090, 2026-10-21) held all
            # eight cards at 0 % for 6.7 min after its last step while one
            # thread hashed.
            digest_arrays, digest_scalars = capture_inputs()
            early_fingerprint = (
                _root_setup_fingerprint_for(node, stepper)
                if boundary_source is not None else None)
            if getattr(getattr(exp, "devices", None), "enabled", False):
                early_devices_receipt = _devices_receipt(
                    exp, stepper, streaming_decisions[int(node.cfg.grid_id)],
                    devices_admission, devices_history, devices_peaks)
            cards_released = _release_forecast_cards(stepper)
            cards_released["after_last_step_seconds"] = round(
                time.perf_counter() - stepping_ended, 6)
            freed_gib = cards_released["freed_bytes"] / 2**30
            print("cards released "
                  f"{cards_released['after_last_step_seconds']:.1f} s after "
                  f"the last step ({freed_gib:.2f} GiB freed); the digest "
                  "and the input verify run on the host", flush=True)
            from woof.state_digest import canonical_store_digest

            digest_box = {}

            def digest_on_host():
                started = time.perf_counter()
                try:
                    digest_box["digest"] = canonical_store_digest(
                        digest_arrays, digest_scalars, node.clock,
                        scope="trajectory",
                        before_hash=runtime._digest_progress(
                            observer, node.cfg.grid_id))
                except BaseException as error:  # noqa: BLE001 - joined below
                    digest_box["error"] = error
                digest_box["seconds"] = time.perf_counter() - started

            # One SHA-256 stream is serial by definition (the digest's value
            # is a single hash over every member), so it runs on its own
            # core BESIDE the seal and the verify, which use the rest.
            digest_thread = threading.Thread(
                target=digest_on_host, name="final-digest", daemon=True)
            digest_thread.start()
        else:
            started = time.perf_counter()
            final_digest = digest_from_store(node.clock, scope="trajectory",
                before_hash=runtime._digest_progress(observer, node.cfg.grid_id))
            timing["canonical_final_state_digest"] = time.perf_counter() - started

    if boundary_source is not None and prestaged_boundaries is None:
        # THE SEAL.  Every interval this run integrated was hash-checked as
        # it loaded; here the complete preparation is bound: the sealed
        # header against the head and every segment, the full preflight
        # against the sealed tree, and the setup fingerprint the header
        # records against the one the integrated state reproduces.
        from woof.prepared_domain_tree_forecast import _root_setup_fingerprint

        seal_started = time.perf_counter()
        inputs = _seal_streamed_inputs(
            inputs, stream=boundary_source.intervals)
        recorded = inputs.cache_reader.metadata.get("setup_fingerprint")
        fingerprint = (early_fingerprint if early_fingerprint is not None
                       else _root_setup_fingerprint(
                           node, steppers.get(int(node.cfg.grid_id))
                           if bundle is not None else None))
        if fingerprint != recorded:
            raise RuntimeError(
                "the sealed prepared cache records a different setup "
                "fingerprint than the streamed boundaries reproduce")
        boundary_stream_receipt.update({
            "seal_wait_seconds": time.perf_counter() - seal_started,
            "waits": [
                {"interval": index, "seconds": seconds}
                for index, seconds in boundary_source.intervals.waits],
            "wait_seconds_total": sum(
                seconds for _, seconds in boundary_source.intervals.waits),
            "proof_sha256": inputs.file_sha256["proof"],
            "prepared_content_sha256": inputs.cache_reader.content_sha256,
            "restore_to_seal_seconds": time.perf_counter() - head_started,
            "clock_intervals_checked": (
                None if clock_guard is None else clock_guard.checked),
        })
    runtime._finalizing_progress(observer, "verify-inputs",
        work_bytes=inputs.cache_reader.payload_bytes + sum(
            path.stat().st_size for path in inputs.authority_paths.values()))
    _verify_inputs_unchanged(inputs)
    if digest_thread is not None:
        digest_thread.join()
        if "error" in digest_box:
            raise digest_box["error"]
        final_digest = digest_box["digest"]
        timing["canonical_final_state_digest"] = digest_box["seconds"]
    # The same gate, told what it is looking at.  The bare ``!=`` this
    # replaced could not tell an implementation that changed from a
    # ``git`` that failed to spawn or from a scratch file this very run
    # wrote, and it destroyed the receipt of a finished forecast for
    # either -- see :func:`_runtime_source_identity_change`.
    final_source_identity = _runtime_source_identity()
    moved = _runtime_source_identity_change(
        runtime_source_identity, final_source_identity)
    if moved is not None:
        raise RuntimeError(
            f"forecast runtime implementation changed during run: {moved}")
    records = runtime._frame_records(
        wrfout_paths, completed_records=getattr(writers, "completed_records", ()),
        progress_callback=observer)
    runtime._finalizing_progress(observer, "write-receipts")
    output_inventory, capture_output = _completed_history_inventory(
        records, output_schedule,
        captured_paths=(cadence_paths if capture is not None and not capture.keep_member_files else None))
    timing["total"] = time.perf_counter() - total_started
    from woof.correct_mode import run_receipt as _correct_mode_run_receipt
    from woof.hydro.runtime import current_settings as _hydro_settings, activation_receipt
    from woof import numerics as _numerics
    from woof.free_threading import host_threads_report as _host_threads_report
    from woof.smoke_receipts import receipt as smoke_execution_receipt
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "hydro_activation": activation_receipt(_hydro_settings()),
        # The defaults this forecast actually ran with, read by the release
        # gate's defaults-active leg (R15): fast numerics unless strict was
        # asked for, and the interpreter lock off in the forecast process
        # itself (an extension can turn it back on after start-up).
        "numerics": _numerics.receipt(),
        "host_threads": _host_threads_report(),
        "pool_trim": getattr(model, "_pool_trim_policy", None),
        "source": inputs.source,
        "prepared_layout": inputs.layout,
        "scope": (
            f"single {cfg.nx}x{cfg.ny}x{cfg.nz} prepared-cache GPUWM "
            f"forecast using "
            f"{inputs.physics_receipt.get('profile') or 'the hash-bound experiment physics suite'}"),
        "gpuwm_version": __version__,
        # ``gpuwm_version`` above is what distribution METADATA claims.
        # This block is which tree actually executed -- package path,
        # install kind, branch/sha/dirt, and whether the two version
        # claims agree.  Both are kept: a reader comparing two receipts
        # needs the number they have always compared, and the reason it
        # can be wrong.
        "provenance": _provenance_receipt(),
        "runtime_source_identity": runtime_source_identity,
        # What the end-of-run recheck could actually SEE.  The identity
        # above is the one taken at launch; the gate that compares it
        # skips a rung neither end resolved, and a skipped comparison
        # that says nothing is how a weakened check goes unnoticed.
        # ``git_compared: false`` on a receipt means the run was bound
        # by ``source_sha256`` alone.
        "runtime_source_identity_recheck": {
            "identity_source_at_launch":
                runtime_source_identity["identity_source"],
            "identity_source_at_end":
                final_source_identity["identity_source"],
            "git_resolved_at_launch":
                runtime_source_identity["git_commit"] is not None,
            "git_resolved_at_end":
                final_source_identity["git_commit"] is not None,
            "git_compared": (
                runtime_source_identity["git_commit"] is not None
                and final_source_identity["git_commit"] is not None),
        },
        # physics.mode (woof/correct_mode.py) as this run's RunConfig
        # resolved it: correct by default from 2.9.0, exact when named, and
        # exact with forced_by under GPUWM_WRF_EXACT=1.
        "correct_mode": _correct_mode_run_receipt({1: cfg}),
        "domain": {
            "grid_id": 1,
            "nx": int(cfg.nx),
            "ny": int(cfg.ny),
            "nz": int(cfg.nz),
            "dx_m": float(cfg.dx),
            "dy_m": float(cfg.dy),
            "dt_seconds": float(cfg.dt),
        },
        "run_seconds": float(exp.run_seconds),
        "wall_seconds": timing["total"],
        "io_mode": "history",
        "history_interval_seconds": cadence_seconds,
        "timing_seconds": timing,
        # One row per history frame: the stepping thread's wall inside the
        # submit, and the writer thread's queue, copy, write and identity
        # walls (PerDomainWrfoutWriters.frame_timings).
        "history_frames": history_frames,
        "restart_contract": {
            "mode": "exact-sealed-setup",
            "restart_input": None if restart is None else str(Path(restart).resolve()),
            "restored_seconds": restored_seconds,
            "already_complete": already_complete,
            "interval_seconds": float(exp.restart_interval_s),
            "checkpoints_written": checkpoints,
        },
        "executor": {
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "health": {
            "initial": initial_health,
            "final": final_health,
            "final_stability": final_stability,
            "history": history,
            "limits": {"max_cfl": 10.0, "max_w_ms": 150.0},
        },
        "final_state_digest": final_digest,
        **smoke_execution_receipt(cfg),
        # The chem mass ledger (woof/core/chem_driver.py), present only on
        # a chem run so every other receipt keeps its bytes. Ranked vectors
        # are captured from the global owner before releasing the cards.
        **({} if (chem_ledger := (_chem_ledger_receipt(node.state)
                                  if bundle is None else final_chem_ledger)) is None
           else {"chem_ledger": chem_ledger}),
        "physics": {
            **dict(inputs.physics_receipt),
            "resolved_lw_sw": list(radiation_scheme_ids(cfg)),
            "radiation_update_count": _physics_update_count(
                driver.radiation_callable),
            "microphysics_update_count": int(driver.microphysics_updates),
            # READ WHERE THE DOMAIN IS.  On the resident road these are
            # reductions over the driver's own device fields, unchanged.  On
            # the store-direct road the driver is the SLAB template's, so the
            # same three expressions would report one slab's sky and one
            # slab's rain as the domain's -- three plausible numbers that are
            # not this domain's.  The store holds the full-domain arrays and
            # is what the sweep keeps current, so the reductions are taken
            # there; a key the store does not carry gives None and names
            # itself, rather than raising at the end of a finished forecast.
            **({
                "swdown_min_wm2": float(cp.min(driver.fields["swdown"]).get()),
                "swdown_max_wm2": float(cp.max(driver.fields["swdown"]).get()),
                "rainnc_max_mm": float(
                    cp.max(driver.microphysics.rainnc).get()),
            } if bundle is None else {
                "swdown_min_wm2": _store_domain_extreme(
                    bundle.store, "fields/swdown", np.min),
                "swdown_max_wm2": _store_domain_extreme(
                    bundle.store, "fields/swdown", np.max),
                "rainnc_max_mm": _store_domain_extreme(
                    bundle.store, "scratch/mp_rainnc", np.max),
                "domain_reductions_read_from": "store",
                "domain_reductions_missing_keys": sorted(
                    key for key in ("fields/swdown", "scratch/mp_rainnc")
                    if key not in bundle.store),
            }),
            # PER-CARRIER PROVENANCE (woof/core/radiation_carriers.py):
            # one row per radiative carrier -- its source, the model
            # second its producer last wrote it, and a representative
            # element read from the live buffer at receipt time.  This is
            # the run-receipt surface the contract promises: a reader
            # asking "did this run integrate a sky nobody computed" gets
            # the answer per carrier, at end of run.
            "surface_radiation_policy": str(driver.carriers.policy),
            "surface_radiation_carriers": _strict_json(
                driver.carriers.report(fields=driver.fields)),
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
            "cpu_peak_rss_bytes": _peak_rss_bytes(),
            # What each number above actually measured, how often it was
            # sampled, and whether observation stayed complete.
            "gpu_peak_sampling": memory_watch.summary(),
            # WHICH EXECUTION MODE produced the peaks above; empty, and the
            # receipt therefore unchanged, whenever [tiles] is off.
            "tiles": ({} if getattr(getattr(exp, "devices", None), "enabled", False)
                      else streaming.receipt_entry(exp.tiles, streaming_decisions)),
            # How many carriers the final health gate and the canonical
            # digest were brought up to date from the store before they ran.
            # 0 on a resident run; on a streamed one it is the whole
            # manifest, and a reader can tell the two apart from the
            # receipt rather than by trusting that the refresh happened.
            "carriers_refreshed_before_final_reads": carriers_refreshed,
        },
        "gridded_output": {
            "cadence_seconds": cadence_seconds,
            "cadence_receipt": cadence_receipt,
            "frames_per_file": 1,
            "expected_frame_count": expected_frames,
            "exact_frame_count_verified": True,
            "initial_frame_verified": (restart_info is None and bool(output_inventory)
                and output_inventory[0]["atomic_writer_readback_verified"]),
            "last_scheduled_frame_verified": (bool(output_inventory)
                and output_inventory[-1]["atomic_writer_readback_verified"]),
            "last_scheduled_offset_seconds": cadence_receipt[
                "last_scheduled_offset_seconds"],
            "last_scheduled_valid_time": cadence_receipt[
                "last_scheduled_valid_time"],
            "last_scheduled_equals_run_end": cadence_receipt[
                "last_scheduled_equals_run_end"],
            "initial_and_final_frames_verified": (restart_info is None
                and cadence_receipt["last_scheduled_equals_run_end"]
                and bool(output_inventory)
                and output_inventory[0]["atomic_writer_readback_verified"]
                and output_inventory[-1]["atomic_writer_readback_verified"]),
            "all_frames_readback_verified": all(
                item["atomic_writer_readback_verified"]
                for item in output_inventory),
            "completion_attribute": {
                "name": "GPUWM_WRITE_COMPLETE",
                "value": 1,
            },
            "frame_count": len(output_inventory),
            "total_bytes": (sum(item["bytes"] for item in output_inventory)
                if all(item["bytes"] is not None for item in output_inventory) else None),
            "files": output_inventory,
            **capture_output,
        },
        "input": {
            "prepared_root": str(inputs.prepared_root),
            "domain_bundle": str(inputs.domain_bundle_path),
            "source_domain_count": inputs.source_domain_count,
            "source_member": inputs.source_member,
            "proof_sha256": inputs.file_sha256["proof"],
            "source_manifest_sha256": inputs.file_sha256["source_manifest"],
            "source_manifest_identity": (
                None if inputs.source_manifest_receipt is None
                else dict(inputs.source_manifest_receipt)),
            "prepared_content_sha256": inputs.cache_reader.content_sha256,
            "prepared_header_sha256": inputs.file_sha256["cache_header"],
            "static_cache_sha256": inputs.file_sha256["static"],
            "geometry_receipt_sha256": inputs.file_sha256["geometry_receipt"],
            "experiment_config_sha256": inputs.file_sha256["experiment_config"],
            "wps_namelist_sha256": inputs.file_sha256["wps_namelist"],
            "wrf_direct_contract_sha256": inputs.file_sha256[
                "wrf_direct_contract"],
            "forcing_hours": list(inputs.forcing_hours),
            "boundary_interval_seconds": inputs.boundary_interval_seconds,
            "preprocessing": inputs.proof["preprocessing"],
            "source_identity": inputs.cache_identity["source_identity"],
            "cache_identity_compatibility": dict(
                inputs.cache_identity_compatibility),
            "landuse_identity": dict(inputs.landuse_identity),
            "direct_wrf_export_source": inputs.export_source_receipt,
            "authority_sha256": dict(inputs.file_sha256),
        },
    }
    # Added rather than inlined above, and only when [tiles] was
    # configured: an unconfigured run must produce the report it produced
    # before this mode existed, key for key, so that every stored receipt
    # and every hash taken over one stays valid.  Same emptiness contract as
    # streaming.identity_payload_entry.
    # [radar_heating]: present only when configured, on the same emptiness
    # contract as report["tiles"].  Every window's lead_class is in it
    # (forecast, oracle, or observed for Level II), so a run heated from a
    # nowcast issued late or from MRMS frames after the start is labelled
    # oracle in its own record.
    if radar_heating_hook is not None:
        report["radar_heating"] = radar_heating_hook.receipt()
        radar_heating_hook.require_applied()
    if nudging_hook is not None:
        report['nudging'] = nudging_hook.receipt()
    if getattr(getattr(exp, "devices", None), "enabled", False):
        report["devices"] = (
            early_devices_receipt if early_devices_receipt is not None
            else _devices_receipt(
                exp, steppers[int(node.cfg.grid_id)],
                streaming_decisions[int(node.cfg.grid_id)], devices_admission,
                devices_history, devices_peaks))
        if cards_released is not None:
            report["devices"]["cards_released"] = cards_released
    elif streaming_report:
        report["tiles"] = streaming_report
    # WHICH INITIALIZATION ROAD RAN, and the numbers it was chosen on.
    # Present only for a run that streams -- ``init_receipt`` is None for
    # every other -- on the same emptiness contract as report["tiles"]: a
    # forecast that configures no [tiles] must write the receipt it wrote
    # before this flag existed, key for key, so that every stored receipt and
    # every hash taken over one stays valid.
    if init_receipt is not None:
        report["stream_init"] = dict(init_receipt)
    if bundle is not None:
        report["stream_init"]["store"] = dict(bundle.receipt)
        report["stream_init"]["unchecked"] = _store_direct_gaps(
            inputs.cache_reader)
        # WHICH GATES WERE ARMED, said in the receipt and not only on the
        # terminal.  Four of this route's five whole-domain gates are armed;
        # the one that is not says why, with the measured number behind the
        # reason, so a reader holding this receipt can see it without
        # re-deriving it from an absence.
        one_pass = float(
            report["health"]["final"].get("coverage", {}).get("seconds", 0.0))
        report["health"]["gates"] = {
            "initialized_d01_full_state": True,
            "executor_periodic_full_state": bool(health_debug),
            "per_step_stability_fold": not already_complete,
            "final_d01_full_state": True,
            "final_stability_fold": not already_complete,
            "why": (
                "Whole-state gates read the canonical host store. "
                f"A full-store pass took {one_pass:.1f}s; periodic scans are "
                + ("enabled by --health-debug." if health_debug else
                   "off by default; --health-debug enables them.")
                + (" No step or stability fold is claimed for an already-complete restore."
                   if already_complete else " Stability folds cover every new step.")),
        }
    # The early render, collected before this process may exit.  Its
    # worker is a DAEMON thread -- it has to be, so a wedged render can
    # never hold a finished forecast open -- and a daemon thread is
    # killed at interpreter shutdown, so a runner that returned here
    # without joining would publish a receipt for pictures that were
    # never written.  `wait` returns None on every way it declined,
    # which is the same "assume nothing was published" answer the
    # finalize stage acts on.
    #
    # Added rather than inlined above, and only when something was
    # published: a run that named no products writes the report it wrote
    # before this existed, key for key.  Same emptiness contract as
    # report["tiles"].
    if first_products is not None:
        runtime._finalizing_progress(observer, "finish-first-products")
        receipt = first_products.wait()
        if receipt is not None:
            report["first_products"] = receipt
    # WHICH AEROSOL INITIAL CONDITION THIS FORECAST STARTED FROM.  For
    # mp_physics=28 that is a choice between WRF's monthly WIF climatology
    # and thompson_init's synthetic CCN/IN profile -- two different
    # forecasts -- and until now the only place it was ever said was the
    # PREPARING process's stderr, which this process does not have and no
    # reader keeps.  The preparation writes it into the cache metadata
    # (woof/ingest/prepared_cache.py) and this reads it back.
    #
    # ``report.update`` of a possibly-empty mapping, not an assignment:
    # woof.aerosol_source_receipt returns NOTHING for a scheme with no
    # aerosol number fields, so a wsm6/Morrison/Kessler forecast writes
    # the report it wrote before this existed, key for key.  The two
    # absences are the whole point and that module states the rule; do not
    # collapse this into an unconditional key.
    report.update(aerosol_source_report_entry(
        inputs.cache_reader.metadata.get(AEROSOL_SOURCE_KEY, {}),
        mp_physics=cfg.mp_physics,
        when_unrecorded=(
            "this forecast reads a prepared cache and does not itself run "
            "the real-data aerosol ingest; the cache at "
            f"{inputs.prepared_root} carries no aerosol-initialization "
            "receipt, so it was written by a preparation that predates the "
            "receipt being stored. Re-prepare to record which source filled "
            "nwfa/nifa -- the fields themselves are in the cache and the "
            "forecast is unaffected, but which dataset produced them is not "
            "recoverable from this bundle")))
    if boundary_stream_receipt is not None:
        # Only on a head binding, so a sealed run's report is unchanged.
        report["input"]["boundary_stream"] = boundary_stream_receipt
    if prestaged_boundaries is not None:
        report["input"]["boundary_prestage"] = {
            **prestaged_boundaries.receipt,
            "boundary_read_seconds": dict(boundary_source.intervals.read_seconds),
        }
    if getattr(driver, "wind_farm_detection", None) is not None:
        report["wind_farm_detection"] = driver.wind_farm_detection
    if getattr(driver, "wind_farm", None) is not None:
        report["wind_farm"] = driver.wind_farm.receipt()
        report["wind_farm"]["module_replay"] = driver.wind_farm.write_replay(outdir / "wind-replay.nc")
    # Default-on hydro: per-step cost with parallelism named (5 percent budget)
    # and the products, both at the output boundary, after device sync.
    from woof.hydro.cost import receipt as _hydro_cost_receipt
    from woof.hydro.forecast_products import publish as _hydro_products
    report["hydro_cost"] = _hydro_cost_receipt(
        driver, forecast_seconds=timing.get("forecast_execution_with_async_io", 0.0),
        atmosphere_steps=int(execution.steps))
    report["hydro_products"] = _hydro_products(
        driver, _hydro_settings(), outdir=outdir, dx_m=float(cfg.dx))
    _atomic_json(outdir / "report.json", report)
    emit_run_capsule(
        outdir, emission_site="prepared_single_domain_forecast",
        run_context={
            "runner_route_and_io_mode": {
                "route": "prepared_single_domain_forecast",
                "io_mode": "history"},
            "output_and_diagnostic_mode": {
                "io_mode": "history",
                "history_interval_seconds": cadence_seconds,
                **({"output_mode": "ensemble_capture",
                    "capture": capture_output["capture"]} if capture_output else {})},
            # DETERMINISM.md lists the configuration bytes as a
            # essential pin, and this route hash-binds the experiment
            # TOML before it runs a step -- the published wheel's capsule
            # still said "the run context did not supply this pin"
            # because the value was never handed over (the 4090 stress
            # run's accuracy finding).  Both byte pins bind here now,
            # git or no git.
            "config_bytes": {
                "path": str(inputs.experiment_config),
                "sha256": inputs.file_sha256["experiment_config"]},
            "input_artifact_bytes": dict(inputs.file_sha256),
        },
        input_bytes={"entries": {
            name: {"algorithm": "sha256", "digest": digest,
                   "path": str(inputs.authority_paths[name])}
            for name, digest in inputs.file_sha256.items()}},
        run_shape={
            "route": "prepared_single_domain_forecast",
            "domain_count": 1,
            "run_seconds": float(exp.run_seconds),
            "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
            "dt_seconds": float(cfg.dt),
        },
        output={"frames": output_inventory,
                "trajectory_digest": {"d01": final_digest}},
        # The spectral seam's run receipts merge in; an apply run whose
        # step receipts are incomplete refuses a clean capsule here.
        receipts={"report": {"path": str((outdir / "report.json").resolve())},
                  **_seam_capsule_receipts(model),
                  "pool_trim": getattr(model, "_pool_trim_policy", None)},
    )
    _atomic_json(progress_path, {
        "schema": PROGRESS_SCHEMA,
        "status": "PASS",
        "source": inputs.source,
        "model_elapsed_seconds": float(exp.run_seconds),
        "requested_run_seconds": float(exp.run_seconds),
        "report": str((outdir / "report.json").resolve()),
        "frame_count": len(output_inventory),
        "last_checkpoint": (None if model._last_checkpoint is None else
                            str(Path(model._last_checkpoint).resolve())),
    }, heartbeat=True)
    from woof.hydro.cost import refuse_if_over_budget as _refuse_hydro_cost
    _refuse_hydro_cost(report["hydro_cost"])
    return report


def _positive_finite_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return seconds


def _clock_defaults(experiment_config: Path) -> tuple[float, float]:
    """The run length and history cadence the experiment already declares.

    ``--run-seconds`` and ``--history-interval-seconds`` are validated to
    equal these exactly -- the experiment TOML is hash-bound, so no other
    value can ever be accepted.  Requiring the user to retype them bought
    nothing and cost a wasted cycle to anyone who tried to shorten a
    retry.  They are optional now and default to what the file says; when
    supplied, the exact-match guard is unchanged.
    """

    raw = tomllib.loads(Path(experiment_config).read_text(encoding="utf-8"))
    experiment = raw.get("experiment")
    domains = raw.get("domain")
    if not isinstance(experiment, Mapping) or not isinstance(domains, list):
        raise ValueError(
            f"{experiment_config} is not an [experiment]/[[domain]] TOML, "
            "so --run-seconds and --history-interval-seconds cannot be "
            "defaulted from it; pass them explicitly")
    roots = [d for d in domains
             if isinstance(d, Mapping) and int(d.get("grid_id", 0)) == 1]
    if len(roots) != 1:
        raise ValueError(
            f"{experiment_config} does not declare exactly one grid_id 1 "
            "domain; pass --run-seconds and --history-interval-seconds "
            "explicitly")
    try:
        return (float(experiment["run_seconds"]),
                float(roots[0]["history_interval_s"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{experiment_config} declares no usable run_seconds / d01 "
            "history_interval_s; pass --run-seconds and "
            "--history-interval-seconds explicitly") from error


def _resolve_clock_arguments(args) -> None:
    """Fill omitted clock flags from the hash-bound experiment TOML."""

    if args.run_seconds is not None \
            and args.history_interval_seconds is not None:
        return
    run_seconds, history_interval_seconds = _clock_defaults(
        args.experiment_config)
    if args.run_seconds is None:
        args.run_seconds = run_seconds
        print(f"prepared forecast: --run-seconds defaulted to "
              f"{run_seconds:g} from {args.experiment_config}",
              file=sys.stderr)
    if args.history_interval_seconds is None:
        args.history_interval_seconds = history_interval_seconds
        print(f"prepared forecast: --history-interval-seconds defaulted to "
              f"{history_interval_seconds:g} from {args.experiment_config}",
              file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """This runner's forecast parser, built without parsing anything.

    Separated from :func:`_parse_args` so the option surface can be READ
    -- by ``--help``, and by the docs/CLI parity test that holds every
    documented door against the flags it really defines.  A parser that
    only exists inside the function that consumes it cannot be checked
    against a document.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, help="Resume a canonical checkpoint with this exact sealed preparation/configuration.")
    parser.add_argument("--health-debug", action="store_true", help="Validate the canonical whole-domain state each step.")
    # The two MODE flags, declared rather than only intercepted.
    #
    # `main` answers both before argparse sees anything, because each is
    # a different program with a different parser and the interception
    # is what makes that possible.  Declaring them here changes no
    # behaviour on the paths that work -- a mode flag in first position
    # never reaches this parser -- and buys the thing their absence cost:
    # `--help` now NAMES them.  FIRST-LIGHT documents
    # `--materialize-authorities` as a step of the documented chain, and
    # a user who lost the doc could not rediscover the step from the
    # tool, because the only parser `--help` rendered was this one and
    # this one had never heard of it.
    modes = parser.add_argument_group(
        "mode flags (each must be the FIRST argument, and selects a "
        "different program with its own --help)")
    modes.add_argument(
        "--materialize-authorities", action="store_true",
        help=("create one hash-receipted named-source experiment/WPS "
              "authority pair for an exact physics profile, then exit.  "
              "Run it first on the line and with --help after it for "
              "that mode's own options"))
    modes.add_argument(
        "--show-capabilities", action="store_true",
        help=("print this runner's capability JSON and exit; it must be "
              "the only argument"))
    parser.add_argument("--source", choices=sorted(SUPPORTED_SOURCES), required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument(
        "--domain-bundle", type=Path,
        help=("explicit hierarchy d01 bundle; if omitted it is derived from "
              "the hash-bound domain-artifacts manifest"))
    parser.add_argument(
        "--proof-sha256", default=None,
        help=("sha256 of the sealed preparation's proof.json; with "
              "--prepared-content-sha256, binds a finished preparation"))
    parser.add_argument(
        "--source-manifest-sha256", default=None,
        help=("sha256 of the preparation's portable source manifest; "
              "required, except with --prepared-head-sha256 naming an "
              "as-posted head, which binds its input plan instead (its "
              "seal writes the manifest, held to that plan)"))
    parser.add_argument("--prepared-content-sha256", default=None)
    parser.add_argument(
        "--prepared-head-sha256", default=None,
        help=("head_sha256 of boundary-stream/head.json: binds a chained "
              "preparation at its head, so the forecast starts while the "
              "later boundary intervals are still being prepared; the "
              "proof and cache digests are checked at the seal"))
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--wps-namelist", type=Path, required=True)
    parser.add_argument(
        "--physics-profile", default=None, type=canonical_template_id,
        help=("optional assertion that the hash-bound experiment IS this "
              "shipped suite, refused on any switch drift; omitted, the "
              "experiment's own physics runs as written and its "
              "WRF-verification status is reported, never gating"))
    parser.add_argument(
        "--ack", action="append", default=[],
        help=("registry-owned expert acknowledgement id; repeat as "
              "needed.  The hash-bound experiment's acknowledgements "
              "array delivers the same consent"))
    parser.add_argument(
        "--run-seconds", type=_positive_finite_seconds, default=None,
        help=("forecast length the model integrates; defaults to the "
              "hash-bound experiment's run_seconds when omitted.  A "
              "shorter length runs the first part of the prepared "
              "forecast from the same prepared state; a length past the "
              "prepared boundary forcing refuses naming the last "
              "prepared lead"))
    parser.add_argument(
        "--history-interval-seconds", type=_positive_finite_seconds,
        default=None,
        help=("history cadence; must equal the hash-bound experiment's d01 "
              "history_interval_s, and defaults to it when omitted"))
    parser.add_argument("--io-mode", choices=("history",), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--devices-table", default=None, metavar="JSON",
                        help=argparse.SUPPRESS)
    parser.add_argument("--devices", type=int, default=None, metavar="N",
                        help="resident slab count; replaces [devices] count")
    parser.add_argument(
        "--tiles", default=None, metavar="JSON",
        help=("the [tiles] table this forecast integrates under, as a "
              "JSON object with the keys woof.core.streaming."
              "StreamingOptions takes (mode/tile_nx/tile_ny/nbuffers/"
              "halo/store/write_mode/pipeline/vram_budget_bytes/"
              "host_budget_bytes).  For the caller whose hash-bound "
              "experiment cannot carry one: the native HRRR chain hands "
              "this runner the authority its preparer BUILT, which has "
              "no [tiles] table, so a user's block had nowhere to ride.  "
              "Validated by the same StreamingOptions.from_mapping the "
              "config front door uses, and binds no identity -- omitted, "
              "the hash-bound experiment's own table (usually none) runs"))
    parser.add_argument(
        "--stream-init", choices=STREAM_INIT_CHOICES, default="auto",
        help=("which road a STREAMED forecast builds its domain on.  "
              "`resident` restores the prepared cache into one full-domain "
              "DomainState, attaches physics to it and lets the streaming "
              "seam copy it into the pinned host store -- the road with the "
              "parity proof, and the one that caps the domain at the size of "
              "the CARD rather than of the machine (MEASURED at nz = 49: the "
              "prepared case costs about 15 780 B/column, so 1024x1024 is "
              "refused on a 16 GB card while the streamed forecast it would "
              "have fed needs about 6 GiB).  `store` fills the same store one "
              "ROW SLAB at a time and never allocates a domain-shaped device "
              "array, so the ceiling is the machine's pinned RAM.  `auto`, "
              "the default, prices the resident state from the cache's own "
              "state/* manifest times the measured physics headroom and "
              "takes the resident road wherever it fits inside "
              # NOT a percent sign.  argparse interpolates every help string
              # (`self._get_help_string(action) % params`), so a literal "%"
              # here is read as a conversion -- "80% of" parses as "% o",
              # octal with the space flag, and --help dies with "%o format:
              # an integer is required, not dict" for every option in the
              # parser rather than just this one.  A fraction says the same
              # thing and cannot be misread.
              f"{_AUTO_RESIDENT_FIT_FRACTION:.2f} of the card's free "
              "memory.  "
              "Meaningful only when the run streams: with [tiles] off the "
              "resident state IS the domain and this flag changes nothing"))
    parser.add_argument(
        "--render-products", default=None, metavar="SPEC",
        help=("`woof render --products`' own spec -- a comma-separated "
              "product list, or `all`, or `none` -- for every frame this "
              "run commits, each drawn on a worker thread as it lands "
              "while the forecast is still integrating, one render at a "
              "time.  Absent is off, and off is the default: there is "
              "deliberately no second switch, so \"which products\" has "
              "one answer that cannot disagree with itself.  The first "
              "frame is the analysis at t = 0, durable before a single "
              "step is integrated"))
    from woof.product_publish import add_arguments as add_publish_arguments
    add_publish_arguments(parser)
    parser.add_argument(
        "--render-dir", type=Path, default=None, metavar="DIR",
        help=("where --render-products publishes; defaults to "
              "OUTDIR/png.  Ignored without --render-products"))
    parser.add_argument(
        "--render-section", default=None,
        metavar="lat,lon,lat,lon|FILE.json",
        help=("`woof render --section`'s own value: the line every xsec: "
              "product in --render-products is cut along.  Ignored "
              "without --render-products"))
    parser.add_argument(
        "--no-memory-gate", action="store_true", dest="no_memory_gate",
        help=("restore a forecast whose priced peak envelope exceeds this "
              "card's free memory anyway, as `woof go --no-memory-gate` "
              "does: the envelope is an upper bound and the card's own "
              "allocation then decides; a model state too big to build at "
              "all is still refused"))
    # The per-step progress surface, registered from one place so this
    # door and the tree runner's cannot drift in spelling or in help.
    add_progress_arguments(parser)
    from woof.simulated_radar_config import add_execution_argument
    add_execution_argument(parser)
    from woof.da.forecast_heating import add_execution_argument as add_heating_argument
    add_heating_argument(parser)
    return parser


#: The mode flags, and the position they are only legal in.
MODE_FLAGS = ("--materialize-authorities", "--show-capabilities")


def _parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_materialize_parser() -> argparse.ArgumentParser:
    """The ``--materialize-authorities`` mode's own parser.

    A separate program with a separate option set, exposed for the same
    reason as :func:`build_parser`: a door the documentation names has
    to be checkable against the flags it really takes.
    """

    parser = argparse.ArgumentParser(
        prog=(
            "python -m woof.prepared_single_domain_forecast "
            "--materialize-authorities"),
        description=(
            "Create one hash-receipted named-source experiment/WPS authority "
            "pair for an exact physics profile."),
    )
    parser.add_argument("--source", choices=sorted(SUPPORTED_SOURCES), required=True)
    parser.add_argument(
        "--base-experiment-config", type=Path, required=True)
    parser.add_argument("--base-wps-namelist", type=Path, required=True)
    parser.add_argument(
        "--physics-profile", default=None, type=canonical_template_id,
        help=("shipped suite to materialize into the experiment; omitted, "
              "the base config's own physics is published unchanged and "
              "its WRF-verification status is reported"))
    parser.add_argument("--output-directory", type=Path, required=True)
    # This stage now owns a LAYERED refusal (a config whose physics the
    # named profile would have overwritten), and a layered message needs
    # the flag that prints its second half -- otherwise the marker
    # woof/explain.py promises can never reach a terminal does.
    add_explain_flag(parser)
    return parser


def _parse_materialize_args(argv=None):
    return build_materialize_parser().parse_args(argv)


def _streaming_options_argument(text: str | None):
    """``--tiles`` as a validated :class:`StreamingOptions`, or ``None``.

    Through ``StreamingOptions.from_mapping`` and nothing else, so this
    flag and a ``[tiles]`` table in a config are refused by the SAME
    validator: an unknown key, a mode this build does not have, half a
    tiling, a tiling set while the mode is off -- all of them produce
    the sentence the config front door produces, rather than a second
    vocabulary a user has to learn for the flag.
    """

    if text is None:
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(
            f"--tiles must be a JSON object, got {type(payload).__name__}")
    return streaming.StreamingOptions.from_mapping(payload, source="--tiles")


def _sealed_binding(inputs: PreparedForecastInputs, *, outdir: Path,
                    observer) -> PreparedForecastInputs:
    """Wait for a head's seal, check it against that head, bind it as sealed.

    The binding a launch on the sealed proof makes, after the seal is held
    to the head this forecast was pinned to
    (:func:`woof.ingest.boundary_stream.verify_seal`).  Not through
    :func:`_seal_streamed_inputs`, which also holds the sealed clock to the
    head's and so refuses exactly the preparation this binding is for.
    The wait is said as the tree runner's seal wait is said
    (:func:`woof.prepared_domain_tree_forecast._start_seal_waits`): on the
    events, the wait log and the supervisor heartbeat, so ``woof go``'s
    watchdog reads a wait, not a worker standing still.
    """

    from woof.ingest.boundary_stream import (
        BoundaryStreamError, StreamedIntervals, verify_seal,
    )
    from woof.prepared_domain_tree_forecast import _start_seal_waits

    head = inputs.stream_head
    stream = StreamedIntervals(
        inputs.prepared_root, head=head,
        on_wait=_start_seal_waits(outdir, inputs.prepared_root, observer))
    if not stream.sealed():
        print("prepared forecast: waiting for the preparation to seal "
              f"({inputs.prepared_root})", file=sys.stderr, flush=True)
    try:
        stream.wait_sealed()
        sealed = verify_seal(inputs.prepared_root, head=head)
    except BoundaryStreamError as error:
        raise ValueError(str(error)) from None
    return preflight_prepared_forecast(**_sealed_arguments(inputs, sealed))


def _rerun_on_seal(args, inputs: PreparedForecastInputs, reason: str, *,
                   outdir: Path, observer, first_products, started):
    """Set a head-bound attempt aside and bind the sealed preparation.

    A later boundary interval moved the terrain-derived clock
    (``StreamedClockChanged``), so the attempt stepped on a clock the
    sealed preparation does not choose and is not this forecast.  Its
    outputs are kept, named, in their own folder; the supervisor heartbeat
    stays where the supervisor reads it
    (:data:`woof.prepared_domain_tree_forecast.KEPT_IN_PLACE`) and
    publishes the restart record
    (:func:`woof.supervisor.restart_attempt`), so ``woof go``'s
    watchdog reads the steps starting again as a new attempt rather than
    a regression.  Returns the sealed inputs, the re-armed first products
    and the checkpoint to resume from (none: a checkpoint the attempt
    resumed from was stepped on the head's clock).  The same recovery as
    the tree runner's.
    """

    from woof.first_products import halt_renders_and_wait
    from woof.prepared_domain_tree_forecast import (
        _release_attempt_memory, _set_aside_streamed_attempt,
    )
    from woof.supervisor import restart_attempt

    rearm = first_products is not None
    if rearm:
        # Ended, and waited for, before its folder moves: nothing is
        # published into a picture folder that has moved, and on Windows
        # a folder with a file open in it does not move.
        if not halt_renders_and_wait(first_products):
            print("prepared forecast: a render of the head-bound attempt "
                  "was still running after it was killed",
                  file=sys.stderr, flush=True)
        first_products = None
    _release_attempt_memory()
    # Said before anything of the new attempt is published; a hosting
    # observer ends the renders it owns here too.
    restart_attempt(observer, reason)
    attempt = _set_aside_streamed_attempt(outdir)
    print(f"prepared forecast: {reason}; the head-bound attempt is kept in "
          f"{attempt} and the forecast starts again on the sealed "
          "preparation", file=sys.stderr, flush=True)
    rerun_args = args
    if args.restart is not None:
        print("prepared forecast: the checkpoint this attempt resumed from "
              f"({args.restart}) was stepped on the head's clock, so the "
              "sealed forecast runs from its start time",
              file=sys.stderr, flush=True)
        rerun_args = argparse.Namespace(**{**vars(args), "restart": None})
    sealed = _sealed_binding(inputs, outdir=outdir, observer=observer)
    if rearm:
        first_products = _route_owned_first_products(
            rerun_args, outdir=outdir, observer=observer, started=started)
    return sealed, first_products, rerun_args.restart


def _route_owned_columns(*, outdir: Path, observer, experiment=None):
    """The any-point column store (rw-store v2) of a run nobody hosts, or ``None``.

    THE BREAKAGE THIS CLOSES: ``woof go`` runs this runner (and the tree
    runner) as a subprocess with no observer, and the only ordinary
    :class:`woof.live_columns.LiveColumns` arming was the one in
    ``woof run-plan`` (``RunObserver.arm_columns``, runplan.py), so a plain
    ``woof go`` run wrote its frames and no ``columns/<domain>/*.wcol`` at
    all (speed-output lane 2026-10-28: a regional cold run finished with two
    full wrfout frames, no .wcol and no columns events).  Every run writes
    its column store; this arms the same writer, cadence and switch
    (``WOOF_COLUMNS=off``) as run-plan, into ``<outdir>/columns``.

    Not armed when a host already owns the store (an observer with
    ``arm_columns``: run-plan arms and stops it itself, and two writers on one
    folder would write every frame twice) or inside an ensemble session
    (members write :class:`woof.live_columns.MemberColumns`).  A store that
    cannot start never stops the run.
    """

    from woof.ensemble.runtime_context import current_session
    if current_session() is not None:
        return None
    if observer is not None and getattr(observer, "arm_columns", None) is not None:
        return None
    from woof.live_columns import LiveColumns, StateFrames, columns_enabled
    if not columns_enabled():
        return None

    def warn(code: str, message: str, **fields) -> None:
        print(f"prepared forecast: {code}: {message}", file=sys.stderr)

    frames = None
    if experiment is not None:
        try:
            frames = StateFrames.for_experiment(experiment)
        except Exception as error:  # noqa: BLE001 - every frame on the cadence is tried instead
            warn("columns_frames_unknown", f"{type(error).__name__}: {error}")
    try:
        return LiveColumns(Path(outdir) / "columns", warn=warn, frames=frames)
    except Exception as error:  # noqa: BLE001 - a store that cannot start never stops the run
        warn("columns_not_armed",
             f"no column store for this run: {type(error).__name__}: {error}")
        return None


def _columns_sink(columns):
    """The landing hook for the store of :func:`_route_owned_columns` (``None`` when there is none)."""

    if columns is None:
        return None

    def sink(**event) -> None:
        columns.frame_committed(domain=int(event["domain"]),
                                valid_time=event["valid_time"], path=event["path"])
    return sink


def _finish_owned_columns(columns, *, halt: bool = False):
    """End the store of :func:`_route_owned_columns`.

    ``halt`` (a stop) drops what is queued; otherwise the queue and each
    grid's last frame are written, bounded as
    :meth:`woof.live_columns.LiveColumns.stop` is.  One line says what was
    written, on stderr: stdout carries only the run's JSON summary, which
    callers parse (2.9.10 pass 1: the line landed on stdout ahead of the
    summary). Nothing raised here changes the forecast's result.
    """

    if columns is None:
        return None
    try:
        summary = columns.halt() if halt else columns.stop()
    except Exception as error:  # noqa: BLE001 - the forecast's result stands
        print(f"prepared forecast: columns_not_written: {type(error).__name__}: {error}",
              file=sys.stderr)
        return None
    failed = len(summary.get("failed") or ())
    print(f"prepared forecast: column store {'halted' if halt else 'written'}: "
          f"{summary['files']} file(s), {summary['bytes']} bytes in "
          f"{summary['seconds']:.1f} s -> {summary['out']}"
          + (f" ({failed} frame(s) failed)" if failed else ""),
          file=sys.stderr, flush=True)
    return summary


def _route_owned_first_products(args, *, outdir: Path, observer,
                                started: float):
    """This runner's own renders, or ``None``.

    What comes back draws every committed frame as it lands: the first
    one through the early render, every later one, every nest's
    included, through the every-frame render behind it
    (:class:`woof.live_products.LandingRenders`).  A runner started as a
    subprocess (``woof go``, the ``--wrfinput`` doors) used to draw only
    its analysis frame while it ran.

    ``None`` for three separate reasons, and the distinction matters:

    * ``--render-products`` was not given, or was given as ``none``.
      Off by ABSENCE, decided by
      :func:`woof.first_products.early_render_requested` rather than
      here, so this runner and every other door agree on what "asked for
      pictures" means without a second switch to disagree with.
    * a HOST already armed one.  ``woof run-plan``'s HRRR chain arms
      the render on its observer before it calls this runner, with the
      dict its own finalize stage will use.  Arming a second trigger on
      the same directory would render the same frame twice, publish two
      receipts over each other, and make the finalize skip depend on
      which worker finished last.  The host's arming wins and this says
      so, because a flag that was passed and quietly did nothing is the
      failure this whole increment is about.

    The report/warn pair is what a hosted run gets from its event
    stream, spelled for a process that has none: one line on stdout at
    the instant the pictures became readable -- carrying the seconds
    from launch, which IS the time-to-first-plot number -- and one line
    on stderr for every way the render declined.  The receipt itself is
    unchanged: :mod:`woof.first_products` writes
    ``gpuwm.first-products.v1`` beside the pictures either way, so a
    finalize stage or a reader cannot tell the two arming routes apart.
    """

    from woof.ensemble.runtime_context import current_session
    if current_session() is not None:
        return None
    from woof.first_products import FirstProducts, early_render_requested
    from woof.live_products import LandingRenders

    if not early_render_requested(args.render_products):
        return None
    if getattr(observer, "first_products", None) is not None:
        print("prepared forecast: --render-products ignored; the caller "
              "hosting this runner already armed an early render, and two "
              "on one directory would draw the first frame twice",
              file=sys.stderr)
        return None
    render_dir = (outdir / "png" if args.render_dir is None
                  else Path(args.render_dir))

    def report(receipt) -> None:
        elapsed = time.perf_counter() - started
        print(f"prepared forecast: first products ready {elapsed:.1f} s "
              "after this runner started "
              f"({len(receipt['paths'])} picture(s), "
              f"--products {receipt['render_products']}): "
              + ", ".join(receipt["paths"]), flush=True)

    def warn(code: str, message: str, **fields) -> None:
        print(f"prepared forecast: {code}: {message}", file=sys.stderr)

    def report_live(entry) -> None:
        print(f"prepared forecast: pictures ready for "
              f"{Path(entry['frame']).name} ({entry['pictures']} picture(s), "
              f"drawn in {entry['render_seconds']:.1f} s)", flush=True)

    # The plan dict `go_cli.render_command` and `go_cli._render_stage`
    # both take, with `run` naming the directory this runner writes its
    # wrfout subdirectory into -- so a finalize render pointed at the
    # same --outdir composes the same command and lands in the same
    # place, which is what makes the early frame's bytes and the late
    # one's comparable at all.
    plan = {"run": outdir, "render": render_dir,
            "publish_root": getattr(args, "publish_root", None) or render_dir.parent / "www",
            "publish_cmd": getattr(args, "publish_cmd", None),
            "hydro": getattr(args, "hydro", None) or "auto",
            "render_products": args.render_products,
            "restart": getattr(args, "restart", None),
            # The section line, which only a caller that names one sets
            # (`woof go --section`); the WRF-input doors pass none.
            "render_section": getattr(args, "render_section", None)}
    if getattr(observer, "live_products", None) is not None:
        # The host already draws every frame as it lands (run-plan arms
        # that whenever its run draws at all); a second every-frame
        # render here would draw each frame twice.
        return FirstProducts(plan, report=report, warn=warn)
    return LandingRenders(plan, report=report, report_live=report_live,
                          warn=warn)


def main(argv=None, *, observer=None, prestaged_boundaries=None,
         runtime_run_overrides=None) -> int:
    """The runner's command line.

    ``observer`` is forwarded to :func:`run_prepared_forecast` and is
    otherwise inert: it lets a caller that hosts this runner in its own
    process (``woof run-plan``, through ``woof go``) receive the same
    progress this function publishes to ``progress.json``, plus each
    wrfout as it lands.  ``None`` -- every command-line invocation --
    changes nothing.

    ``runtime_run_overrides`` uses the preflight API's validated inert-field
    contract. Original preparation authorities remain byte-bound.
    """

    if argv is None:
        # Before any heavy import: a free-threaded build keeps the rank
        # threads off the interpreter lock (woof.free_threading).
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
        print(f"prepared_single_domain_forecast: {error}", file=sys.stderr)
        return 2
    # Which tree is about to integrate this forecast.  Announced before
    # any argument is interpreted, and a refusal when the version this
    # process would stamp into its report disagrees with the code
    # writing it.  ``--show-capabilities`` is answered above, untouched:
    # a front end probing the runner's capabilities gets JSON on stdout
    # and nothing else, exactly as before.
    from woof.provenance_gate import announce_for_main

    refusal = announce_for_main("woof-prepared-forecast")
    if refusal is not None:
        print(f"prepared_single_domain_forecast: {refusal}",
              file=sys.stderr)
        return 2
    if argv[:1] == ["--materialize-authorities"]:
        args = _parse_materialize_args(argv[1:])
        # Same standard as --outdir below, and for the same reason: the
        # commonest way to meet this stage twice is to re-run the
        # documented command, which lands on an --output-directory that
        # already exists.  That is a usage answer, not a crash.
        try:
            receipt = materialize_named_source_authorities(
                source=args.source,
                base_experiment_config=args.base_experiment_config,
                base_wps_namelist=args.base_wps_namelist,
                physics_profile=args.physics_profile,
                output_directory=args.output_directory,
            )
        except FileExistsError as error:
            print("prepared_single_domain_forecast: "
                  f"--output-directory refused: {error}", file=sys.stderr)
            return 2
        except (OSError, ValueError) as error:
            # A refusal this stage owns -- a profile its source cannot
            # prepare, a base config that is not there, a physics value
            # the named profile would have overwritten -- is still a
            # sentence rather than a traceback.  It is labelled as what
            # it is, not as an --output-directory problem.  Rendered,
            # because a layered message printed raw leaks the
            # ``[[explain]]`` marker onto the terminal.
            print("prepared_single_domain_forecast: refused: "
                  + render_explanation(
                      str(error), explain=explain_enabled(args),
                      command=(
                          "python -m woof.prepared_single_domain_forecast "
                          "--materialize-authorities")),
                  file=sys.stderr)
            return 2
        print(json.dumps({
            "schema": receipt["schema"],
            "status": receipt["status"],
            "source": receipt["source"],
            "physics_profile": receipt["physics_profile"],
            "experiment_config": receipt["generated"][
                "experiment_config"],
            "wps_namelist": receipt["generated"]["wps_namelist"],
            "receipt": receipt["receipt"],
        }, sort_keys=True))
        return 0
    # A mode flag that reached here was not in first position, so no
    # interception above could claim it.  Say that, rather than letting
    # the forecast parser refuse it for the unrelated reason that its own
    # required flags are missing.
    for flag in MODE_FLAGS:
        if flag in argv:
            print(f"prepared_single_domain_forecast: {flag} must be the "
                  "FIRST argument on the command line; it selects a "
                  "different program, with its own --help and its own "
                  "options", file=sys.stderr)
            return 2
    args = _parse_args(argv)
    # THE capability preflight, from the same registry `woof run` and
    # `woof go` refuse with, so this documented `python -m` door cannot
    # drift from the subcommands.  It had no handler at all: a missing
    # GPU runtime surfaced as a raw traceback AFTER this function had
    # claimed the output directory, written progress.json and run the
    # whole preparation preflight -- work a reader then has to clean up
    # for a gap that was knowable before any of it.
    from woof import capabilities
    from woof.explain import split as split_explanation

    try:
        capabilities.require(
            "python -m woof.prepared_single_domain_forecast",
            *capabilities.COMMAND_REQUIREMENTS["run"],
            before=("Refusing here, before the output directory is "
                    "claimed and before the preparation preflight runs."))
    except capabilities.CapabilityMissing as refused:
        # The action half only, with NO ``--explain`` pointer: that flag
        # is registered on the ``--materialize-authorities`` parser, not
        # on this forecast one, so a pointer at it would name a flag this
        # door rejects with a usage dump -- which is precisely the class
        # of defect this whole change exists to remove.  Measured before
        # it shipped: `python -m woof.prepared_single_domain_forecast
        # --explain ...` exits 2 on `unrecognized arguments`.
        print(split_explanation(str(refused))[0], file=sys.stderr)
        return 2
    # Before the clock defaults, before `--tiles`, and above all before
    # the output directory is claimed: `--proof-sha256` is the digest a
    # reader is most likely to take from the wrong place, and answering
    # it from inside the preflight meant a traceback AFTER a run
    # directory had been created for them to clean up.
    proof_refusal = (_preparation_binding_refusal(args)
                     or _proof_digest_refusal_at_the_door(args))
    if proof_refusal is not None:
        print(f"prepared_single_domain_forecast: {proof_refusal}",
              file=sys.stderr)
        return 2
    # After the capability gate, because the clock defaults are READ FROM
    # THE CONFIG: a reader with no runtime should hear about the runtime,
    # not about the shape of a file whose contents cannot matter yet.
    _resolve_clock_arguments(args)
    # Before the output directory is claimed: a malformed [tiles] is a
    # usage mistake, and refusing it after creating a run directory
    # leaves the caller a directory to clean up for a typo.
    try:
        tiles = _streaming_options_argument(args.tiles)
        from woof.core.devices import DeviceOptions
        devices_options = (None if args.devices_table is None else
                           DeviceOptions.from_mapping(json.loads(args.devices_table),
                                                      source="--devices-table"))
        from woof.simulated_radar_config import execution_argument
        simulated_radar = execution_argument(getattr(args, "simulated_radar_table", None))
        from woof.da.forecast_heating import execution_argument as heating_argument
        radar_heating = heating_argument(getattr(args, "radar_heating_table", None))
    except (ValueError, TypeError) as error:
        print(f"prepared_single_domain_forecast: --tiles refused: {error}",
              file=sys.stderr)
        return 2
    # A rejected --outdir is a usage mistake, not a crash: one sentence
    # naming the problem and a directory that works, never a traceback.
    try:
        outdir = claim_output_directory(
            args.outdir, protected_roots=(
                args.prepared_root, args.experiment_config, args.wps_namelist,
                *((Path(args.restart).parent,) if args.restart is not None else ())))
    except (ValueError, FileExistsError) as error:
        print(f"prepared_single_domain_forecast: --outdir refused: {error}",
              file=sys.stderr)
        return 2
    # [radar_heating] in the --experiment-config is an execution control:
    # it is cut out, the rest is bound to the preparation byte for byte as
    # before (written beside the given config, so its relative paths keep
    # resolving), and the table rides beside it exactly as
    # --radar-heating-table does.  A preparation made from the config WITH
    # the table keeps it whole.  Both at once is ambiguous.
    try:
        from woof.da.forecast_heating import detach_table_from_config
        bound_config, table_heating = detach_table_from_config(
            args.experiment_config, prepared_root=args.prepared_root)
        if table_heating is not None:
            if radar_heating is not None:
                raise ValueError(
                    "[radar_heating] is in --experiment-config and in "
                    "--radar-heating-table; give it once, or the run would "
                    "heat by whichever was read last")
            radar_heating = table_heating
            args.experiment_config = bound_config
    except ValueError as error:
        print(f"prepared_single_domain_forecast: [radar_heating] refused: {error}",
              file=sys.stderr)
        return 2
    from woof.runtime import _preparation_progress
    _preparation_progress(observer, "validate-prepared-inputs")
    started = time.perf_counter()
    # Armed here, before the preflight: the frame this renders is the
    # analysis at t = 0, and the history alarm is true at t = 0, so it is
    # durable before a single step is integrated.  Anything armed after
    # the forecast opens is already racing the thing it exists to catch.
    first_products = _route_owned_first_products(
        args, outdir=outdir, observer=observer, started=started)
    _atomic_json(outdir / "progress.json", {
        "schema": PROGRESS_SCHEMA,
        "status": "VALIDATING_PREPARATION",
        "source": args.source,
        "model_elapsed_seconds": 0.0,
        "requested_run_seconds": args.run_seconds,
    }, heartbeat=True)
    # Read before ANY of this runner's work, so it describes the cache
    # this run inherited rather than the cache this run has been
    # filling.  Pure filesystem; the device is not touched here.
    kernel_cache_census = scan_kernel_cache()
    preflight_complete = False
    columns = None
    try:
        # Timed, because it was dark and it is not free: every digest in
        # the preparation receipt is re-checked against the bytes on
        # disk here, so this scales with the prepared cache and is the
        # first thing a reader waiting on "why has nothing happened yet"
        # is actually waiting for.
        preflight_started = time.perf_counter()
        inputs = preflight_prepared_forecast(
            source=args.source, prepared_root=args.prepared_root,
            proof_sha256=args.proof_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            prepared_content_sha256=args.prepared_content_sha256,
            prepared_head_sha256=args.prepared_head_sha256,
            experiment_config=args.experiment_config,
            wps_namelist=args.wps_namelist,
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            run_seconds=args.run_seconds,
            history_interval_seconds=args.history_interval_seconds,
            domain_bundle=args.domain_bundle,
            tiles=tiles, devices=args.devices, devices_options=devices_options,
            **({} if runtime_run_overrides is None else
               {"runtime_run_overrides": runtime_run_overrides}),
            **({} if simulated_radar is None else {"simulated_radar": simulated_radar}),
            **({} if radar_heating is None else {"radar_heating": radar_heating}))
        preflight_seconds = time.perf_counter() - preflight_started
        preflight_complete = True
        verification = dict(inputs.physics_receipt).get("verification")
        if (isinstance(verification, dict)
                and verification.get("status") != "wrf-verified"
                and isinstance(verification.get("sentence"), str)):
            # One sentence, and the run continues (owner posture:
            # warn-not-block; detail lives in the receipt).
            print(f"prepared forecast: {verification['sentence']}",
                  file=sys.stderr)
        # --no-memory-gate skips the resident envelope admission inside,
        # as it skips `woof go`'s own gate; the constructor floor stays.
        from woof.core.resident_admission import memory_gate_override

        from woof.ingest.boundary_stream import StreamedClockChanged

        def run(bound, products, restart):
            with memory_gate_override(args.no_memory_gate):
                return run_prepared_forecast(
                    bound, output_directory=outdir, observer=observer,
                    first_products=products, stream_init=args.stream_init,
                    progress_options=ProgressOptions.from_args(args),
                    preflight_seconds=preflight_seconds,
                    kernel_cache_census=kernel_cache_census,
                    restart=restart, health_debug=args.health_debug,
                    columns=columns,
                    **({"prestaged_boundaries": prestaged_boundaries}
                       if prestaged_boundaries is not None else {}))

        columns = _route_owned_columns(outdir=outdir, observer=observer,
                                       experiment=getattr(inputs, "experiment", None))
        clock_changed = None
        try:
            report = run(inputs, first_products, args.restart)
        except StreamedClockChanged as changed:
            # Only the reason leaves this handler: the traceback holds the
            # attempt's frames and, through them, its state on the card.
            clock_changed = str(changed)
        if clock_changed is not None:
            _finish_owned_columns(columns, halt=True)
            columns = None
            inputs, first_products, restart = _rerun_on_seal(
                args, inputs, clock_changed, outdir=outdir,
                observer=observer, first_products=first_products,
                started=started)
            columns = _route_owned_columns(outdir=outdir, observer=observer,
                                           experiment=getattr(inputs, "experiment", None))
            report = run(inputs, first_products, restart)
    except BaseException as error:
        # A forecast that fails before its preparation seals leaves the
        # producer running: the sealed preparation is what a retry reuses
        # (woof.ingest.boundary_stream.run_chained stops it only on an
        # interrupt of the chain that owns both).
        model_elapsed_seconds = 0.0
        try:
            current_progress = _load_json_object(
                outdir / "progress.json", "forecast progress")
            model_elapsed_seconds = float(
                current_progress.get("model_elapsed_seconds", 0.0))
        except (OSError, TypeError, ValueError):
            pass
        model_elapsed_seconds = _failure_clock(error, model_elapsed_seconds)
        inventory_error = None
        try:
            durable_inventory = _durable_wrfout_inventory(outdir)
        except (OSError, ValueError) as inventory_failure:
            durable_inventory = []
            inventory_error = (
                f"{type(inventory_failure).__name__}: {inventory_failure}")
        failure = {
            "schema": REPORT_SCHEMA,
            "status": "FAIL",
            "source": args.source,
            "run_seconds": args.run_seconds,
            "io_mode": args.io_mode,
            "history_interval_seconds": args.history_interval_seconds,
            "wall_seconds": time.perf_counter() - started,
            "model_elapsed_seconds": model_elapsed_seconds,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "durable_partial_output": {
                "status": "PARTIAL_NOT_RUN_PASS",
                "frame_count": len(durable_inventory),
                "total_bytes": sum(
                    item["bytes"] for item in durable_inventory),
                "files": durable_inventory,
                "inventory_error": inventory_error,
            },
        }
        from woof.ingest.boundary_stream import (
            SOURCE_BEHIND_EXIT_CODE, WAIT_LOG_NAME, SeamWaits, SourceBehind,
        )
        behind = None
        if isinstance(error, SourceBehind):
            # A source lead passed its late time: the frames through the
            # seam and the checkpoint there are kept, and the record says
            # which lead, so the door exits 75 naming it.
            behind = error.at(frames_kept=len(durable_inventory))
            behind.details.setdefault("checkpoint", None)
            failure["source_behind"] = dict(behind.details)
            failure["exit_code"] = SOURCE_BEHIND_EXIT_CODE
        _atomic_json(outdir / "report.json", failure)
        progress_failure = {
            "schema": PROGRESS_SCHEMA,
            "status": "FAIL",
            "source": args.source,
            "model_elapsed_seconds": model_elapsed_seconds,
            "requested_run_seconds": args.run_seconds,
            "report": str((outdir / "report.json").resolve()),
            "error": str(error),
            "last_durable_wrfout": (
                None if not durable_inventory
                else durable_inventory[-1]["path"]),
        }
        if behind is not None:
            progress_failure["source_behind"] = dict(behind.details)
        _atomic_json(outdir / "progress.json", progress_failure,
                     heartbeat=True)
        if behind is not None:
            # A hosting door says `source_behind` from this record just
            # before `failed`; `woof go` relays it from the wait log.
            SeamWaits(log_path=outdir / WAIT_LOG_NAME).record_source_behind(
                behind)
            hook = getattr(observer, "source_behind", None)
            if hook is not None:
                try:
                    hook(dict(behind.details))
                except Exception:  # noqa: BLE001 - the refusal stands
                    pass
        # The column store of a run that did not finish: a stop writes
        # nothing more; a failure writes the frames it committed (as
        # run-plan's stop_columns does).
        _finish_owned_columns(columns, halt=isinstance(error, KeyboardInterrupt))
        columns = None
        if first_products is not None:
            # The renders of a run that did not finish.  A stop draws
            # nothing more (the desktop kills a run 5 s after asking); a
            # failure finishes the frames it wrote and joins the render,
            # which would otherwise die with this process mid-picture.
            try:
                if isinstance(error, KeyboardInterrupt):
                    getattr(first_products, "halt", lambda: None)()
                else:
                    from woof.runtime import _finalizing_progress

                    _finalizing_progress(observer, "finish-first-products-after-failure")
                    first_products.wait()
            except BaseException as render_error:  # noqa: BLE001
                print("prepared_single_domain_forecast: joining the "
                      f"renders failed: {type(render_error).__name__}: "
                      f"{render_error}", file=sys.stderr)
        if isinstance(error, InitializationMemoryRefused):
            # Refused before the first device allocation: this forecast
            # does not fit the card it was about to be restored onto.  The
            # failure report above records it; the terminal gets the one
            # sentence with the terms and the remedy, not a traceback.
            print(f"prepared_single_domain_forecast: {error}",
                  file=sys.stderr)
            return 2
        if not preflight_complete and isinstance(error, (ValueError, OSError)):
            print(f"prepared_single_domain_forecast: refused: {error}", file=sys.stderr)
            return 2
        from woof.io.restart import RestartMismatchError

        if isinstance(error, RestartMismatchError):
            print(f"prepared_single_domain_forecast: refused: {error}", file=sys.stderr)
            return 2
        if isinstance(error, PreparationProofDigestMismatch):
            # A run that never started, not one that failed: the door
            # check above catches every command-line spelling of this,
            # and this is the same answer for the paths that reach the
            # preflight some other way.  A traceback where a user can
            # land is the defect; the refusal names both digests.
            print(f"prepared_single_domain_forecast: {error}",
                  file=sys.stderr)
            return 2
        if behind is not None:
            # Not a failure of this run: the source is later than its
            # budget.  The sentence names the lead and what to do.
            print(f"prepared_single_domain_forecast: {behind}",
                  file=sys.stderr)
            return SOURCE_BEHIND_EXIT_CODE
        raise
    _finish_owned_columns(columns)
    if report.get("schema") == "gpuwm-ensemble-run.v1":
        print(json.dumps({"schema": report["schema"], "status": report["status"],
            "members": report["request"]["members"], "completed_seconds": report["completed_seconds"],
            "ensemble_manifest": str(outdir / "ensemble-run.json")}, sort_keys=True))
        return 0
    print(json.dumps({
        "schema": report["schema"],
        "status": report["status"],
        "source": report["source"],
        "run_seconds": report["run_seconds"],
        "history_interval_seconds": report["history_interval_seconds"],
        "frame_count": report["gridded_output"]["frame_count"],
        "prepared_content_sha256": report["input"][
            "prepared_content_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
