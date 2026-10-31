#!/usr/bin/env python3
"""Run and prove the pinned real-x4 MPAS-A v8.4.1 CUDA physics chain.

The production path is deliberately closed until every moving source below has
an exact frozen SHA-256.  Once released, one invocation performs:

* one uninterrupted 30 x 120 s full-physics CUDA forecast;
* F000/F030/F001 boundary snapshots and quantitative comparison with the
  sealed native CPU GF+YSU-GWDO authority;
* one checkpoint/restart continuation from F030 through F001; and
* an exact-byte comparison of uninterrupted and restored committed F001 state,
  saved diagnostics, Arwen persistence, surface/soil state, precipitation, and
  the six external YSU-GWDO diagnostics.

The finalized fa35 Arwen contract is the execution target.  Its documented GF
deviation is retained honestly: MPAS computes endpoint ``rthdynten/rqvdynten``,
but fa35 has no API lane for them and supplies zero GF advective forcing.  This
tool therefore never claims bitwise/source-native GF parity.  It does require
real weather evolution and records quantitative native comparisons.

CUDA and production imports are lazy.  Closed authority bytes, source pins,
constructor mappings, output scope, and the staged transaction API are checked
before a CUDA device is probed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np


#: This module ships inside the package (``woof/hex/drivers/``) so the forecast
#: runs from a wheel.  ``SRC`` is the directory holding the ``woof.hex``
#: package this file belongs to -- ``<tree>/src`` in a checkout,
#: ``site-packages`` in an install -- and ``ROOT`` is its parent, which is
#: the checkout's ``tree/`` in a checkout and only ever used for defaults
#: (the one-case proof's authority files) and for keeping outputs out of the
#: source tree.
SRC = Path(__file__).resolve().parents[2]
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Imported after the path insert, and deliberately: this resolves the frozen
# execution set against whichever hexcore the interpreter ends up with, so a
# gate never hashes one copy of a module while another copy executes.
from woof.hex import shipped_sources  # noqa: E402
from woof.hex.stochastic_timing import timed_arwen_step  # noqa: E402


SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics/v2"
SNAPSHOT_SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics-snapshot/v3"
CHECKPOINT_SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics-checkpoint/v3"
RESTART_WORKER_SCHEMA = "mpas-port.cuda-v841-real-x4-full-physics-restart-worker/v1"
PROFILE = "real-x4.163842-v8.4.1-full-physics-gf-ysu-gwdo"
SOURCE_RELEASE = "v8.4.1"
ARWEN_COMMIT = "26daaab7ef5c1104166fe61503cdd9487750f1af"
ARWEN_CONTRACT_DOCUMENT_SHA256 = (
    "5c629e23be2af20c0b1660d262443c415256126b812493f6681590bf07aff92a"
)
ARWEN_CONTRACT_SURFACE_SHA256 = (
    "45de852c6f8a8951d7e3f65b2b1e64462cf213d50707db4e52cb7eba17491ff2"
)
ARWEN_GLACIER_COMPOSED_TU_SHA256 = (
    "edafcac585d4786c0cdfddf07f8e767b64d0d40b6db0e4da3dc3b2fa8c21fb59"
)
ARWEN_XICE_THRESHOLD = np.float32(0.02)
ARWEN_GLACIER_CUDA_PROVENANCE = (
    "noahmp-glacier/cuda (woof/core/kernels/noahmp_glacier.cu)"
)

N_CELLS = 163_842
N_EDGES = 491_520
N_LEVELS = 55
N_INTERFACES = 56
N_SOIL_LEVELS = 4
SCALAR_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
CHEMISTRY_CONFIG = None
CHEMISTRY_CONFIG_SHA256 = None
SOURCE_SCALAR_NAMES = ("qv", "qc", "qr")
COLD_ZERO_SCALAR_NAMES = ("qi", "qs", "qg")
DT_SECONDS = 120.0
FULL_STEPS = 30
CHECKPOINT_STEP = 15
SNAPSHOT_STEPS = (0, CHECKPOINT_STEP, FULL_STEPS)
SNAPSHOT_LABELS = {0: "F000", CHECKPOINT_STEP: "F030", FULL_STEPS: "F001"}
START_TIME_TEXT = "2026-08-10_12:00:00"
NOMINAL_DX_M = np.float32(25_000.0)
EXPECTED_ARWEN_P_TOP_PA_F32 = np.float32(1_159.38818359375)
EXPECTED_TOP_PRESSURE_RANGE_PA = (592.24884, 1_233.00952, 1_342.08362)
# Both floors come from the ONE admission surface, re-fitted at the merged
# tip 2026-08-26 (ruling, 2026-08-26) from the measured affine row --
# woof.hex.device_admission.FLOOR_DERIVATION is the derivation of record.
# They replace the asserted ``24 GiB`` / ``22 GiB`` pair, which stood 1.2x
# above the measured 20,446 MiB x4 peak and, scaled linearly per mesh,
# admitted smaller meshes BELOW their measured peaks.  The restart worker
# executes the same device stack the run does (its own measured peak sits
# under the run's), so the same measured requirement floors both; the old
# 2 GiB split encoded margin between two asserted constants, not a measured
# difference.
from woof.hex.device_admission import native_device_floor_bytes
from woof.hex.errors import ConfigurationRefusal

MIN_FREE_DEVICE_BYTES = native_device_floor_bytes()
RESTART_WORKER_MIN_FREE_DEVICE_BYTES = MIN_FREE_DEVICE_BYTES

EXPECTED_SURFACE_CLASSIFICATION = MappingProxyType(
    {
        "xland_source": "native",
        "xland_land_columns": 61_528,
        "xland_water_columns": 102_314,
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": 4_643,
        "open_water_columns": 102_314,
        "sflx_land_columns": 53_875,
        "glacier_columns": 3_010,
    }
)
EXPECTED_NOAHMP_CENSUS = MappingProxyType(
    {
        "land": 53_875,
        "water": 102_314,
        "sea_ice": 4_643,
        "glacier": 3_010,
        "glacier_path": ARWEN_GLACIER_CUDA_PROVENANCE,
    }
)
NATIVE_XLAND_SOURCE_SHA256 = (
    "e7d081437e55646d65fd635fee48fb9b8702a25c36a6950c806809a2187748bc"
)
NATIVE_XLAND_FLAT_SHA256 = (
    "2009116b7b34d03439d47980ad6d2977583c45189bd9cdbbad3c678576b36805"
)
GLACIER_INDEX_SHA256 = (
    "1ffe16d231cca7885ad72ae438479e82f59a253e36f73937f17bd45eb5a5f4fa"
)
SEA_ICE_INDEX_SHA256 = (
    "1d8b4c34c4c2797c6c578181fd2365ee195c8a5bfbe6b5e6e1c36c2d7740182d"
)
THRESHOLD_DELTA_INDEX_SHA256 = (
    "df9a33ed4b791778e6c99b9979fe7a1ce1a447333faf53e4f2f94545a2a0c2ea"
)


F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS = MappingProxyType(
    {
        "t2": {
            "source": "t2m",
            "sha256": "1fcebcd0117a1a069e2ddc67f14e2fe2a47b030d4bf884aed2c83c996690eb3f",
        },
        "u10": {
            "source": "u10",
            "sha256": "0fb96429b062ef7d1114f26b19443c2f3c3b82d50b6c07003304c357b9cdc9da",
        },
        "v10": {
            "source": "v10",
            "sha256": "131b6b1e56a761b8c68d5abf72d61b71302c170f38c4a488acc21465f9be0960",
        },
    }
)

RECEIPT_NAME = "cuda-v841-full-physics-x4-receipt.json"
BASELINE_DIAGNOSTIC_SCHEMA = (
    "mpas-port.cuda-v841-real-x4-full-physics-baseline-diagnostic/v1"
)
BASELINE_DIAGNOSTIC_RECEIPT_NAME = (
    "cuda-v841-full-physics-x4-baseline-diagnostic-receipt.json"
)
BASELINE_DIAGNOSTIC_STATUS = (
    "uninterrupted_baseline_passed_restart_not_evaluated_non_release"
)
BASELINE_DIAGNOSTIC_WARNING = (
    "ENGINEERING BASELINE; restart proof pending; NOT RELEASE"
)
SNAPSHOT_FILE_NAMES = {
    "F000": "cuda-history.2026-08-10_12.00.00.nc",
    "F030": "cuda-history.2026-08-10_12.30.00.nc",
    "F001": "cuda-history.2026-08-10_13.00.00.nc",
    "F001_RESTART": "cuda-history-restart.2026-08-10_13.00.00.nc",
}

CLAIM = (
    "one real initialized x4.163842 MPAS-A v8.4.1 CUDA forecast through "
    "one hour using WSM6 + GF + YSU + external YSU-GWDO + revised-MO + "
    "NoahMP + cloud fraction + legacy RRTMG, with an F030 restart arm"
)
NONCLAIMS = (
    "not a bitwise or source-native MPAS full-physics reproduction",
    "fa35 GF supplies zero rthften/rqvften because its finalized public seam "
    "does not accept MPAS endpoint rthdynten/rqvdynten",
    "fa35 phase one shares moist-hydrostatic pressure among its consumers",
    "fa35 legacy RRTMG uses nominal one-dimensional interface weights and "
    "performs a host round trip on radiation-due calls",
    "fa35 accepts one scalar p_top_pa, so the exact F000 per-column native "
    "pres2_p top interface is reduced to its areaCell-weighted mean; native "
    "MPAS phase one retains per-column plrad/top-interface pressure",
    "q2 is published bitwise as the scheme produced it and is not required "
    "nonnegative (native q2 itself carries negative values); weather-field "
    "plots of it remain Rust-renderer-only under the render law",
    "does not establish forecast skill",
)

# Every execution boundary is exact-byte pinned. A future None value remains
# a hard pre-CUDA refusal, never a wildcard, so partially released edits cannot
# silently enter this proof.
# RE-DERIVED 2026-08-28 for the 0.2.0 package rename (``mpas_port`` ->
# ``woof.hex``).  Twenty-two of these twenty-three rows moved, and every one of
# the twenty-two files is byte-identical to its pre-rename self once that one
# identifier is substituted -- measured file by file, not argued from a diff.
# THE TWENTY-THIRD ROW IS NOT A RENAME MOVE AND IS THE REASON THIS TABLE NOW
# HAS A GENERATOR: ``cuda_arwen_physics_v841.py`` was edited by 8aeb582, which
# left this table naming the pre-edit digest ``d10eaeef...``, and the next run
# of the proof died at ``require_frozen_execution_sources`` before it reached a
# device -- "frozen execution source changed".  A table nothing recomputes goes
# stale the first time somebody edits a docstring.  ``tools/repin_source_tables.py``
# re-derives every row of it from the tree; the next rename is a re-run of that,
# not a hand-patch of forty-odd hex strings.
# Re-frozen 2026-10-14 for the public-release scrub: owner quotations and rented-host
# names were reworded in comments and docstrings of config_v841, cuda_arwen_physics_v841,
# cuda_ntiedtke_column_v841, cuda_seam_snapshot_v841, dt_admission and step_anchors.
# MEASURED INERT: each file's pre- and post-scrub bytes parse to the identical AST once
# docstrings are dropped and string constants normalised, with the same string-constant
# count; no number, branch or name moved.  Pins that were already stale before the scrub
# were left as they were.
# Re-derived 2026-10-18 (unify hex/next): diagnostic_export.py and history_writer.py
# are the union of the parked misc and speed sides (batched complete export, atomic
# writes and history-ready recovery); these two rows are the merged files' digests.
# No card run for this merge.
# Re-frozen 2026-10-20 (capabilities campaign landing, lane/hexcap-base) with
# tools/repin_source_tables.py: diagnostic_export.py also exports engine-held
# diag/ fields such as TOA OLR (c858be11), and cuda_arwen_physics_v841.py reads
# those diag/ keys and carries the optional mixed-layer ocean state into the
# surface restart (4ac577f7). Host-side export and restart plumbing; no CUDA
# source changed. No card run for this re-freeze.
EXECUTION_SOURCE_PINS: dict[str, str | None] = {
    "src/hexcore/diagnostic_export.py": "e8d86569ff5f592465d00a6764eb354fb2e178d297edf479185b907529c18d16",
    "src/hexcore/snapshot_transfer.py": "7064e74719933894c9d92dbfe06190b404801cd72ecc5e4fe2447b66124c5efb",
    "src/hexcore/history_ready.py": "287840b7ece3cb97185340f3e161c997bbfd9a11233ff5da4198cca42e1f9523",
    "src/hexcore/history_writer.py": "570c9124745a8d8325a35c11da17e58036bfcf2e3117a5be502dec196a9905f0",
    "src/hexcore/cuda_arwen_helper_scope_v2.py": "641e8b77427a7c2f8eff374daa37e09a034fc45271d1dac67c3e39d03e3814e9",
    "src/hexcore/radiation_source_sets.json": "4fdb41a8102a0c4891771ed476d2eb7cf870e89a66c1a2dd4a92b8cd42dac802",
    "src/hexcore/data/profiles/engine-helper-sets.v1.json": "fb9988bda21d9447b6aba277c6b394da0ffc1727e4662c894bd6279b92abda86",
    # Native upper damping: Rust and CUDA match compiled source blocks.
    "src/hexcore/native/upper_damping.rs": "5483b2c1e7da45317ffe4213c5879b583070d9e28d591ec52a9c697a052e81f1",
    "src/hexcore/upper_damping_v841.py": "947f477d5f56b41d95fb5a0b92adb21d68be95b5fac19ba524181cb373710860",
    "src/hexcore/cuda_upper_damping_v841.py": "04f0b5b450c088575e589de943b7b44fff10b442d01ae0c0acc0a5e848748545",
    # Re-frozen 2026-10-04 for monotonic scalar transport. The released
    # limiter-off default remains byte-identical. The added execution branch
    # has a native nonzero-tracer ruler and native regional mask contracts;
    # it does not qualify the coupled physics against a native endpoint.
    # Re-frozen 2026-08-26 for the limited-area PHYSICS lane.  Two changes,
    # both inert on a closed sphere.  (1) The geometry object now records
    # which of its columns is native's garbage element, read off the mesh
    # view that declared it; on a global mesh the field is None and nothing
    # reads it.  (2) When it is not None, the DERIVED physics view's garbage
    # column is overwritten with a copy of the last real column immediately
    # before validation.  MEASURED on r4.75.11020: without it the column
    # arrives as a 0 K, zero-pressure atmosphere (rho_zz is native's pool
    # 1.0, rho_theta is the pool zero) and validate_prep_v841_f32 refuses the
    # whole step, correctly -- no parameterisation has a table for it.  The
    # dycore state is not touched, because real elements gather its pool
    # values.  Every affected proof re-runs against this digest.
    # 2026-10-18: consumer-specific post-RK preparation. H60 owner replay
    # and separate device timeline are in the hex-diagnostics lane receipt.
    "src/hexcore/cuda_physics_prep_v841.py": (
        "fffb12b21f67e51e045c50dc739e2259b962ad28cf99d829211beb92c72b8ef3"
    ),
    "src/hexcore/cuda_gwdo_v841.py": (
        "8dc6e9a3d8e88c231bc8d738f80a1655839aed2478b56ccc5038ab4daba7cfb5"
    ),
    # Re-frozen 2026-10-11: commit_wsm6_state_v841_f32 takes the species
    # row's block length in place of the literal 6.  THE BREAKAGE: a
    # three-species block (Kessler, microphysics off) was copied three planes
    # past both buffers and stopped the card with cudaErrorIllegalAddress at
    # the first commit.  A six-species block runs the same six copies, so the
    # default suite's history is unchanged (measured, 60 km and 30 km).
    # Re-derived 2026-10-14 (release 0.4.0, H1 merged onto the H7 merge): the
    # SG2 series (43ccb8b0) gives the held tendencies an optional
    # diabatic_theta field, its residency check and receipt flag, and
    # with_lagged_diabatic_theta_v841.  None (the first step, physics off, a
    # backend without the producer) leaves the held tendencies as before.
    # This file equals H1's byte for byte; the H1 lane receipt's dry derecho
    # and ArWen exact identities ran on it.  No card run for this merge.
    # Re-frozen 2026-10-15 (WSM6 SR envelope): _wsm6_sr_roundoff_limit_v841
    # now prices the four per-step scalings WSM6 applies to each surface
    # accumulation (fallsum*delz/denr*dtcld*1000), 7 ULPs at one minor loop
    # where the stale copy allowed 3.  THE BREAKAGE: a default 2 h forecast
    # (60/30/15 km, dt=90 s) stopped at 4590 s on SR=1.00000048.  Only the
    # validation threshold moves; no field is written differently.
    # Re-frozen for the Morrison row: per-bucket signed-fallout validation
    # preserves native words; the default positive-sum policy is unchanged.
    # Re-derived 2026-10-27 (fix/hex-host-bound): the kernel-cache contract check reads the
    # compile-platform binding alone (compile_contract.kernel_cache_platform) instead of a full
    # manifest JSON round trip at every coupling launch.  No kernel, launch or operand changed;
    # 2 h Milton LTS forecast 0 of 1,167,818,735 words differ (HEX-HOST-BOUND-2026-10-27).
    "src/hexcore/cuda_physics_v841.py": (
        "18ac060a3adf6884eef7b853cb4b8fab76b37a6ec561c545372515d7f7e71672"
    ),
    # Re-frozen during the 12 GiB capacity work: the dynamics subcycle stopped
    # copying a scalar block nothing writes, stopped taking a private image of
    # the substep-start state for RK stage 1, and stopped asking recover_state
    # for six cell diagnostics it discards.
    # Re-frozen again for the regional CUDA lane: the two host
    # validations that already REFUSED a regional mesh -- a negative
    # cellsOnEdge entry and a nonzero specified-zone mask -- now refuse
    # through the earned-anchor gate it was ruled on 2026-08-25
    # (cuda_backend/regional_admission), so the message names the missing
    # anchor instead of declaring the lane closed/global, which the regional
    # kernels retired.  Both sites raised ConfigurationRefusal before and
    # raise ConfigurationRefusal now; no admitted global configuration can
    # reach either branch, because reaching one requires a culled mesh.
    # Every affected proof re-runs against this digest.
    # Re-frozen for the NVRTC reciprocal-rewrite lane: the flux3/flux4
    # denominator in vertical_u_flux_f32, theta_vertical_flux_f32 and
    # w_vertical_flux_f32 moved from the source literal ``12.0f`` to the
    # translation-unit constant ``mpas_third_order_denominator``.  MEASURED
    # (tree/evidence/nvrtc-reciprocal-20260826/): NVRTC rewrites a literal
    # divisor as a reciprocal multiply for every target at or above
    # compute_100, moving 33.2-33.4% of the interior flux values on the
    # reference regional bytes; a host-writable symbol cannot legally be
    # folded, so the division survives.  On targets below that boundary the
    # payload digests are byte-unchanged, measured on sm_86.  Every affected
    # proof re-runs against this digest.
    # Re-frozen again for the regional FORECAST lane: the driver
    # gained a ``regional_v841`` hook mirroring ``halo_exchanger_v841`` above
    # it, and ten guarded sites that call it -- the lbc_in interval refresh,
    # the shared tend_rho pool copy, the specified-zone/relaxation-zone
    # tendency adjust, the regional acoustic substep, the divergence-damping
    # solve count, the two halves of the specified-zone velocity overwrite,
    # the w hard-zero, the regional split transport with its boundary adjust,
    # and the end-of-step resets.  EVERY one is guarded by
    # ``self.regional_v841 is not None``, which is None on every global run,
    # and tests/test_regional_forecast_anchor.py walks the source to prove no
    # call escapes its guard.  The CUDA source string is byte-unchanged, so
    # the compile manifest, the module cache keys and the FTZ counts are
    # untouched; this is a host-Python change only.
    # Every affected proof re-runs against this digest.
    # Re-frozen 2026-08-26 for the convection ruling: the per-component
    # physics cadence table reports ``convection: None`` when no cumulus
    # scheme is selected instead of calling float() on it.  MEASURED
    # (the proving RTX 5070 Ti): the first convection-off arm bound clean,
    # admitted its timestep, built its sealed constructor and died inside
    # composite step 0 with "float() argument must be a string or a real
    # number, not 'NoneType'", surfacing as "composite step at 0.0 s was
    # aborted without publication".  Nothing changes for a run that selects
    # a scheme: the table is byte-identical there.
    # Re-frozen 2026-08-26 for the limited-area PHYSICS lane: there is now ONE
    # definition of the bdyMask digest.  ``regional_bdy_mask_digest`` used to
    # hash the three masks bare and skip any that were missing, while
    # ``mesh.regional_boundary_mask_digest`` framed each name with a NUL and
    # refused a partial triple -- two conventions, and the docstring here
    # claimed they were one.  MEASURED on r4.75.11020, same three arrays:
    # 0c2d9feb... against 2baf091d...  This function now delegates to the mesh
    # contract's definition, so an anchor row and a registry row are finally
    # comparable, and the two rows in cuda_backend/regional_admission are
    # re-minted onto the surviving spelling.  No global run reaches this
    # function at all (it returns None without a bdyMask triple), and the CUDA
    # source string is byte-unchanged, so the compile manifest, the module
    # cache keys and the FTZ counts are untouched.
    # Re-frozen 2026-08-27 for the provenance scrub.  THE BREAKAGE
    # THIS PIN MOVE PREVENTS: the public assembly rewrites provenance out of
    # the shipped source, and until this cut it did so on the COPY.  Six
    # files carrying that provenance are inside a digest gate, and the pins
    # travel with the copy, so rewriting on the copy published a tree whose
    # own gates refuse it -- including regional_admission.kernel_set_sha256,
    # which is computed at ADMISSION time and would have refused every
    # regional CUDA run at the door of the shipped package while the wheel
    # built clean and the audit passed.  The rewrite therefore happens here,
    # in the tree the pins describe.  MEASURED INERT: the pre- and post-scrub
    # bytes parse to the identical AST once docstrings are stripped and every
    # string constant is normalised, with the same string-constant count, so
    # no number, no branch and no name moved; the string constants that did
    # change are message and metadata text (a card label, a ruling quotation,
    # anchor basis prose) that nothing reads as a value.  Every affected proof
    # re-runs against this digest.
    # Re-frozen for order-2 stage orchestration. The order-3 defaults retain
    # their coefficient recomputation, timings and scalar density weights.
    # Order 2 uses the native half/half/full schedule, retains its first-stage
    # coefficients and selects the half density weight. Native schedule bits,
    # device repeatability and the scalar density contract are held separately
    # from native endpoint qualification.
    # Re-frozen 2026-10-05 for the startup lane: ``_validate_v841_host_mesh``
    # and ``_validate_v841_host_coefficients`` are served by the content-keyed
    # host-preparation cache (woof.hex.prep_cache).  MEASURED: the mesh
    # validation alone took 91.9 s of a 633 s cold start on the published
    # 30 km mesh, re-proving the same connectivity of the same bytes every
    # run.  A verdict is only replayed for a call whose every argument array,
    # scalar and object field hashes to the key it was proven under; a
    # refusal is never stored.  The CUDA source strings are byte-unchanged.
    # Re-run against this digest: the CPU battery and the same-card identity
    # pairs on conus12km and the published 30 km mesh
    # (evidence/hex-startup-20261005/).  The x4 full-physics proof and the
    # regional class mints did not re-run; their records are carried.
    # Re-frozen 2026-10-10 (certificate gates retired, owner ruling): _validate_config no longer refuses
    # config_scalar_advection=False, config_positive_definite=True (which
    # runs the FCT branch, as native does) or apvm=0 for want of a native
    # ruler; the split-three refusal names the missing code path; and a
    # limited-area cull on the global path is refused for the breakage it
    # is, whether or not a regional anchor exists.  The recorded x4 schedule
    # executes the same code.
    # WHAT RAN AGAINST THE 2026-10-10 DIGESTS, stated because an earlier
    # note here said "every affected proof re-runs against this digest"
    # and none had: the CPU battery on the 8-card RTX PRO 6000 host, and three 8-card forecasts on
    # 8-card RTX PRO 6000 host from the 2026-10-03 12Z start (15 km 24 h on bare defaults, 30 km
    # 48 h on bare defaults and on the old RK3/6/600 s schedule; receipts
    # kept with the owner's 2026-10-10 release records).  The x4
    # full-physics byte proof and the regional class mints did NOT re-run;
    # their records are carried.
    # Re-frozen again 2026-10-10 (audit of the retirement): the REGIONAL
    # scalar path takes the limited routine on the last stage when
    # config_monotonic OR config_positive_definite is on, as native
    # (mpas_atm_time_integration.F:2953) and the global path do; it tested
    # config_monotonic alone and passed positive_definite=False.  The global
    # path and every default-configuration run execute the same code as
    # before.  Re-run against this digest: the CPU battery, and the forecast
    # named in the report beside the bundle.  No regional case ran on a card.
    # Re-frozen 2026-10-13 (forecast batch, hex/gpu-forecast-batch): the host
    # mesh validation evaluates its row laws over whole tables instead of row
    # loops (tests/test_mesh_rows_vectorized.py keeps the loops and proves the
    # same verdict and message, whole and class-local meshes and corrupted
    # tables).  No CUDA source, launch or arithmetic line moved.  Re-run
    # against this digest: the derecho 3 km patch, 30 members, histories
    # bitwise equal to the separate-process runs of the previous digest.
    # Re-frozen 2026-10-13 (landing wave 2, SG2): the lagged microphysics
    # heating carry (e7be214b):
    # add_lagged_diabatic_theta_v842_f32 (mpas_atm_time_integration.F:6931-6937)
    # and the final-RK-stage removal in recover_cells_f32 (F:4436-4437).  CUDA
    # source DID move.  Both are inactive unless the physics hands the
    # dycore a diabatic_theta (has_diabatic 0 otherwise), and no producer in
    # this tree does, so every run of this tree executes the old arithmetic.
    # Re-run against this digest: the CPU pin/anchor tests only; no card run.
    # Re-frozen 2026-10-13 (integration hex/correct-mode): correct/storm-growth
    # adds the PRODUCER (lagged_heating_v841.py on the ArWen and suite
    # backends) and the specified-zone zero kernel
    # zero_speczone_lagged_diabatic_theta_v842_f32 (F:7949).  The sentence
    # above ("no producer in this tree") no longer holds: with physics on the
    # carry is active from the second step (SG2, class A, MPAS behaviour, no
    # switch).  Dry runs launch neither kernel.  This file is byte-identical to
    # correct/storm-growth 312cf586; the digest is that branch's.
    # Re-frozen 2026-10-14 (release 0.4.0, lane H1, on integrate/hex-mpas): the
    # integrate line's cuda_driver.py with the forecast-batch host mesh rows
    # above and the SG2 producer series merged in.  Re-run against this digest:
    # the dry derecho cap450 identity (fixed step 30 min, active LTS 1 h) and
    # the ArWen physics-on exact identity in the lane receipt, and G1/G2.
    # Re-derived 2026-10-14 (H1 merged onto the H7 merge): H1's file plus the
    # 0.4.0 release scrub of one comment line (an owner ruling in place of a
    # person).  No code line differs from the copy the H1 identities above
    # ran on.  No card run for this merge.
    "src/hexcore/cuda_driver.py": (
        "e8d3c8484928c85f126dfce6bacb35ba215cf59dc7005ded780b96aa639f0f42"
    ),
    # THE BREAKAGE THIS PREVENTS: cuda_driver.py now executes host
    # preparation through woof.hex.prep_cache.  Unpinned, an edit to the key
    # derivation or the result round trip could hand the frozen driver
    # different arrays with every proof still reporting frozen sources.
    # Re-frozen 2026-10-05 (start-up lane, second pass): the key now carries
    # the binding of every name a cached function reaches (a rebound
    # rkind_libm hook misses or bypasses), leaves out the mesh's file paths
    # (the same bytes elsewhere are served), carries NumPy's CPU dispatch, and
    # the disk layer is capped.  What re-ran against this digest: the CPU
    # battery files for the cache, the pin and anchor tests, and the
    # conus12km same-card identity pair on an RTX 5090, recorded in
    # evidence/hex-startup-20261005/.  The x4 full-physics proof and the
    # regional class mints did NOT re-run; their records are carried.
    # Re-derived 2026-10-14 (0.4.0 release scrub): one docstring line names
    # the card host by its hardware; docstring only, no code line moved.
    # Re-derived 2026-10-16 (single-core setup fix): the three bytecode walks
    # of the key derivation (_code_names, _code_reads, _local_hexcore_imports)
    # are memoised per code object and return frozensets.  They are pure
    # functions of the code object and package, so every key is the same; no
    # array computation moved.  Re-ran: tests/test_prep_cache.py.
    "src/hexcore/prep_cache.py": (
        "9a02aded2bf3f708589aa1006c2ac042cffa145d48e58841b062f1f253280bcf"
    ),
    # THE BREAKAGE THIS PREVENTS: the same lane that re-froze cuda_driver.py
    # also changed ``recover_state`` -- it gained ``include_pressure`` and
    # ``to_host`` gained a refusal -- and the RK stage the driver runs executes
    # this module every substep.  Unpinned, the frozen-source proof would have
    # certified a driver digest while the recovery module underneath it moved
    # freely, so a later edit to the pressure-diagnostic path could change
    # forecast bytes with every proof still reporting frozen sources.
    "src/hexcore/cuda_backend/recovery.py": (
        "9765e0d72853a2af1d4d7acf18db45e2c09e9a89ed3098e392a8ec8ebaab98ab"
    ),
    # Re-frozen for the regional CPU authority lane: the v8.4.1
    # config admits config_apply_lbcs and config_moist_physics behind the
    # driver's regional-runtime and index_qv gates (both refused without an
    # admitted LBC source), validating the base contract on a neutralized
    # view exactly as the Smagorinsky knobs already did.  No admitted global
    # configuration changes meaning; every affected proof re-runs against
    # this digest.
    # Re-frozen again 2026-08-26 (the timestep-anchor campaign, timestep admission): the three
    # literal entries that pinned config_dt / config_bldt_seconds /
    # config_cudt_seconds to 120.0 inside the `exact` map are now one
    # consultation of the earned-anchor registry woof.hex.dt_admission,
    # which holds exactly one anchor -- 120.0 s -- so the ADMITTED SET IS
    # UNCHANGED and every other value is still refused before anything is
    # allocated.  What changed is the message (it names the anchor's
    # evidence and the mint procedure instead of asserting the lane is
    # exact) and the surface a ruling would edit.  No admitted
    # configuration changes meaning; every affected proof re-runs against
    # this digest.
    # Re-frozen 2026-08-26 for the convection ruling (convection is
    # switched off below 3 km): config_convection_scheme stopped being a
    # literal in the `exact` map and is admitted from
    # woof.hex.convection_admission, config_cudt_seconds became nullable
    # because a cadence for a scheme that is never called means nothing, and
    # the two are checked as one decision on the host.  The proven
    # Grell-Freitas configuration is UNCHANGED and still the only one
    # carrying a native reference; what changed is that "off" is now a
    # second admitted configuration of the same lane, and it earns its own
    # timestep anchors rather than borrowing GF's.
    # Re-frozen 2026-08-27 for the provenance scrub.  THE BREAKAGE
    # THIS PIN MOVE PREVENTS: the public assembly rewrites provenance out of
    # the shipped source, and until this cut it did so on the COPY.  Six
    # files carrying that provenance are inside a digest gate, and the pins
    # travel with the copy, so rewriting on the copy published a tree whose
    # own gates refuse it -- including regional_admission.kernel_set_sha256,
    # which is computed at ADMISSION time and would have refused every
    # regional CUDA run at the door of the shipped package while the wheel
    # built clean and the audit passed.  The rewrite therefore happens here,
    # in the tree the pins describe.  MEASURED INERT: the pre- and post-scrub
    # bytes parse to the identical AST once docstrings are stripped and every
    # string constant is normalised, with the same string-constant count, so
    # no number, no branch and no name moved; the string constants that did
    # change are message and metadata text (a card label, a ruling quotation,
    # anchor basis prose) that nothing reads as a value.  Every affected proof
    # re-runs against this digest.
    # Re-frozen 2026-10-03 (legacy config_epssm, twice: commit c656b69 moved
    # this row without this note, and the follow-up moved it again).  The
    # ADMITTED SET WIDENS on every v8.4.1 lane, this frozen x4 lane
    # included: a nonzero config_epssm, which refused before, is now read
    # and ignored with upstream's own warning (LegacyEpssmIgnored), exactly
    # as native v8.4.1's mpas_atm_dynamics_checks does; only a non-real
    # value refuses.  No configuration admitted before changes meaning: the
    # default stays 0.0 and validates silently, and no v8.4.1 numeric path
    # reads the knob.  MEASURED on an RTX 5090 (2026-10-03): the v8.4.1 dry lane
    # on x1.2562 gives bitwise-identical states for config_epssm 0.0, 0.1
    # and 0.5 on CPU and on CUDA; tests/test_epssm_v841.py carries that run
    # and a source hold on every config_epssm read in driver.py and
    # cuda_driver.py.  Every affected proof re-runs against this digest.
    # Re-frozen for keyed order-2 dycore step certificates. Existing order-3
    # configuration bytes and full-physics anchor semantics remain unchanged.
    # Re-frozen 2026-10-10 (certificate gates retired): the column config's
    # defaults are MPAS-A v8.4.1's Registry schedule (order 2, two sub-steps,
    # radiation 1800 s, limiter on; woof.hex.mpas_defaults), the timestep is
    # admitted by cadence integrality and clock closure rather than by an
    # anchor or step certificate, and the "exact" lane pins are retired.  The
    # x4 proof passes mpas_defaults.X4_RECORDED_SCHEDULE explicitly, so the
    # recorded configuration is unchanged.
    "src/hexcore/config_v841.py": (
        "1cee9d6702785a4226805ab133aa7e7112ac6fcdb234cf3eef44a87e2c3a2933"
    ),
    # Order-2 admission depends on both the keyed certificate selector and
    # its table. Moving either without this pin would change the admitted
    # schedule while a frozen-source receipt still named the older config.
    # Re-frozen 2026-10-10: the certificate table is a record
    # (step_anchor_record), no longer a gate.
    "src/hexcore/step_anchors.py": "13f8bf3361b3fc5c25805d05d48e3c59841050e3d372c03c96f1498133399b72",
    "src/hexcore/data/dycore-step-anchors.v1.json": "f9a102a3f4fd848e95bf9fabe5a5ccd0579ba01de739de9e7da8e4ed7930e62d",
    # New pin (convection ruling, 2026-08-26): the frozen config ADMITS
    # config_convection_scheme from this module and the frozen timestep
    # registry keys on the selection it produces, so the pinned config's
    # meaning depends on these bytes -- the same reason dt_admission.py is
    # pinned.  It carries the ruling, the 3,000 m threshold, and the
    # measured breakage the gate prevents (GF is called 720 times an hour at
    # 5 s against the proven 30, and the measured |w| mean reached 87.5 m/s
    # against a control's 1.48).
    # Re-frozen 2026-08-27 for the provenance scrub.  THE BREAKAGE
    # THIS PIN MOVE PREVENTS: the public assembly rewrites provenance out of
    # the shipped source, and until this cut it did so on the COPY.  Six
    # files carrying that provenance are inside a digest gate, and the pins
    # travel with the copy, so rewriting on the copy published a tree whose
    # own gates refuse it -- including regional_admission.kernel_set_sha256,
    # which is computed at ADMISSION time and would have refused every
    # regional CUDA run at the door of the shipped package while the wheel
    # built clean and the audit passed.  The rewrite therefore happens here,
    # in the tree the pins describe.  MEASURED INERT: the pre- and post-scrub
    # bytes parse to the identical AST once docstrings are stripped and every
    # string constant is normalised, with the same string-constant count, so
    # no number, no branch and no name moved; the string constants that did
    # change are message and metadata text (a card label, a ruling quotation,
    # anchor basis prose) that nothing reads as a value.  Every affected proof
    # re-runs against this digest.
    # Re-frozen 2026-10-10: the unknown-scheme refusal names the missing seam
    # instead of an "unproven lane".  Message text only.
    # Re-frozen 2026-10-11: a variable-resolution mesh reads the bands at
    # its finest cell's scheme spacing (len_disp/meshDensity**0.25), not at
    # min(dcEdge).  THE BREAKAGE: the 48-3 km patch mesh's min(dcEdge) of
    # 2778.75 m switched cumulus off for the whole mesh, 48 km cells too.
    # The x4 proof names cu_grell_freitas explicitly, so its configuration
    # does not pass through the changed branch; uniform meshes are decided
    # byte-identically (tests/test_convection_admission_variable_mesh.py).
    "src/hexcore/convection_admission.py": (
        "33372d55a763e50428d3742af895d19c9ff8c000ef7814ee52ac757554e602f0"
    ),
    # THE BREAKAGE THIS PREVENTS: config_v841.validate() now ADMITS the
    # model timestep from this module, so the pinned config's meaning
    # depends on this file's bytes.  Unpinned, an edit here could admit a
    # timestep the frozen lane was never proven at while every proof still
    # reported frozen sources -- exactly the hole recovery.py was pinned to
    # close.
    # Re-frozen 2026-08-26 for the timestep anchors the ruling for that day named:
    # ADMITTED_TIMESTEPS gained rows at 100, 75, 20 and 5 s, each naming the
    # campaign that earned it (evidence/dt-anchors-20260826/RECEIPT.md) and
    # each recording native_reference=None, because the one native MPAS-A
    # v8.4.1 integration this program holds ran at 120 s and no other timestep
    # can ever have that half.  The pinned config's meaning depends on these
    # bytes -- V841MpasColumnPhysicsConfig.validate admits config_dt from this
    # table -- so every proof that verifies frozen sources re-runs against this
    # digest.  No executable behaviour outside the table changed.
    # Re-frozen again 2026-08-26 for the convection ruling: the registry is
    # keyed by the CONFIGURATION (dt, cumulus selection) rather than by the timestep
    # alone, because switching the closure off changes how often it is
    # called from "every step" to "never" and that is part of what an
    # anchor's forecasts measured.  Every existing row keeps its evidence
    # and its meaning; a convection-off run can no longer read one.  Two
    # convection-off rows were then EARNED on the proving RTX 5070 Ti at 20 s and 5 s
    # (evidence/convection-off-20260826/), and the 5 s Grell-Freitas row's
    # open attribution was retired against them: with the closure never
    # called the same timestep still reaches |w| max 93.96 against its
    # 102.67, so GF's call rate is NOT the cause.  Table and refusal text
    # only; no executable behaviour outside them changed.
    # Re-frozen a third time 2026-08-26 for the surface/PBL cadence campaign:
    # the registry key gained its third fragment, the cadence itself, for the
    # identical reason the second was added.  config_bldt_seconds is welded
    # to dt exactly as cudt is, so 5 s calls the surface layer, the LSM and
    # the PBL 720 times an hour against the proven 30 -- and the
    # convection-off A/B eliminated Grell-Freitas while leaving that
    # call-rate shape standing (91.4% of the 5 s excess survived the closure
    # never being called).  THE BREAKAGE THE KEY PREVENTS: without the
    # fragment a held-cadence anchor would occupy the SAME slot as the
    # welded row at the same (dt, cumulus) and silently replace it, after
    # which every ordinary welded run at that timestep would be admitted
    # against a band measured at 24x its own call rate.  require_dt_anchor
    # keeps the near-miss diagnostic: a cadence with no row of its own but a
    # welded sibling still refuses with the cadence-mismatch sentence rather
    # than "nothing is anchored here".  Table, key and refusal text only; no
    # executable behaviour outside them changed.
    # Re-frozen 2026-08-27 for the provenance scrub.  THE BREAKAGE
    # THIS PIN MOVE PREVENTS: the public assembly rewrites provenance out of
    # the shipped source, and until this cut it did so on the COPY.  Six
    # files carrying that provenance are inside a digest gate, and the pins
    # travel with the copy, so rewriting on the copy published a tree whose
    # own gates refuse it -- including regional_admission.kernel_set_sha256,
    # which is computed at ADMISSION time and would have refused every
    # regional CUDA run at the door of the shipped package while the wheel
    # built clean and the audit passed.  The rewrite therefore happens here,
    # in the tree the pins describe.  MEASURED INERT: the pre- and post-scrub
    # bytes parse to the identical AST once docstrings are stripped and every
    # string constant is normalised, with the same string-constant count, so
    # no number, no branch and no name moved; the string constants that did
    # change are message and metadata text (a card label, a ruling quotation,
    # anchor basis prose) that nothing reads as a value.  Every affected proof
    # re-runs against this digest.
    # Re-frozen 2026-10-10 (certificate gates retired): require_dt_anchor and
    # its refusal builders are replaced by anchor_record (evidence) and
    # require_cadences (cadence integrality and clock closure, the breakages
    # that remain); the candidate mint no longer gates on a sentence.  The
    # registered rows and their evidence are unchanged.
    # Re-frozen again 2026-10-10 (audit): largest_admissible_dt is deleted.
    # It applied the retired 0.90 x 125 m/s policy and nothing called it.
    # Re-frozen 2026-10-14 (release hygiene N7): the 75 s anchor's evidence
    # citations name the lab host neutrally (dt75-host-a).  String constants
    # and one comment only, measured inert by AST; no number, branch or name
    # moved.
    "src/hexcore/dt_admission.py": (
        "77abfa300cf131704d56914011771993ee4574b489f6a4b54d32a8f742b17d20"
    ),
    # New pin (surface/PBL cadence, 2026-08-26): this module decides
    # config_bldt_seconds, a knob the frozen timestep registry keys on, so
    # the pinned config's meaning depends on these bytes -- the same reason
    # convection_admission.py is pinned above.  Unpinned, an edit here could
    # silently change how often the surface/PBL stack is called while every
    # proof still reported frozen sources.  It carries the default (the weld,
    # bldt = dt, which is the proven configuration and changes no run), the
    # A/B instrument that holds it, and the measured breakage the registry
    # gate prevents.
    # Re-frozen 2026-08-27 for the provenance scrub.  THE BREAKAGE
    # THIS PIN MOVE PREVENTS: the public assembly rewrites provenance out of
    # the shipped source, and until this cut it did so on the COPY.  Six
    # files carrying that provenance are inside a digest gate, and the pins
    # travel with the copy, so rewriting on the copy published a tree whose
    # own gates refuse it -- including regional_admission.kernel_set_sha256,
    # which is computed at ADMISSION time and would have refused every
    # regional CUDA run at the door of the shipped package while the wheel
    # built clean and the audit passed.  The rewrite therefore happens here,
    # in the tree the pins describe.  MEASURED INERT: the pre- and post-scrub
    # bytes parse to the identical AST once docstrings are stripped and every
    # string constant is normalised, with the same string-constant count, so
    # no number, no branch and no name moved; the string constants that did
    # change are message and metadata text (a card label, a ruling quotation,
    # anchor basis prose) that nothing reads as a value.  Every affected proof
    # re-runs against this digest.
    # Re-frozen 2026-10-10: the radiation default beside the decision is
    # MPAS-A's 1800 s and callers pass the run's own cadence.
    "src/hexcore/pbl_cadence.py": (
        "f9c86d3ec05d98b4e80df0270d3fde1b31a5949512772f1177c4b34f7eeb82d3"
    ),
    # Re-frozen for the seam convergence: the Arwen seam pin moved
    # off the pin-only lineage onto the release line's seam-converge merge,
    # so the pinned bytes are the bytes the next public engine snapshot
    # carries. The adapter's own behavior is unchanged.
    #
    # Re-frozen again 2026-08-28 for the 2.5.8 engine repin.  THE GATE'S OWN
    # TABLE WAS THE STALE SIDE, and it was stale for one commit: 77f831b
    # rewrote ARWEN_SOURCE_MANIFEST and ARWEN_BUILD_COMMIT inside this very
    # module (four digest rows plus the build commit, to the engine PyPI
    # publishes as woof 2.5.8), which moves the module's own bytes, and did
    # not carry the new digest here.  The concrete breakage that left behind:
    # `require_frozen_execution_sources` refused BEFORE any CUDA probe with
    # "frozen execution source changed for src/hexcore/cuda_arwen_physics_v841.py",
    # so at 77f831b the x4 full-physics proof -- the proof the repin itself
    # owes -- could not launch at all, and
    # tests/test_cuda_v841_full_physics_x4.py::
    # test_exact_release_scope_and_all_execution_sources_are_frozen was RED at
    # that tip.  Only the digest moved; no behaviour in this module changed
    # beyond the manifest constants the repin deliberately rewrote.
    # Re-frozen again 2026-09-13: the checkout guard now resolves a tree to
    # the engine identity its bytes were measured as -- the public pin, or
    # a row of engine_pin.ADMITTED_ENGINES under that row's own version
    # string -- and the receipts and restart identity name that identity's
    # commit and digests.  On the public engine every value it writes is
    # unchanged; the frozen row's own execution path did not move.
    # Re-frozen again 2026-09-27: the per-step boundary export became
    # declinable (``rollback_snapshot``, default True).  A backend built
    # armed, which is every caller that does not say otherwise and this
    # proof among them, takes the same export on the same step and restores
    # it on the same failures; an unarmed one skips the export and retires
    # its seam on a failure instead of restoring it.  The forecast declines
    # it unless --stop-on-refusal reads it: 447 MB of device-to-host copies
    # and about 24 ms of host time a step on the 43,884-cell point mesh.
    # Every affected proof re-runs against this digest.
    # Re-frozen 2026-10-05 for the New Tiedtke row (RRFS v2 at 12 km): the
    # sealed constructor admits cumulus_scheme='ntiedtke' (every step, as GF),
    # _new_seam builds that row's seam from cuda_ntiedtke_column_v841 after
    # verifying the two engine files it executes, the radius guard admits
    # exactly that subclass, and the receipt names the row.  The default
    # (GF and convection-off) paths are unchanged; _ADAPTER_AUTHORITY and
    # every contract digest are unchanged.
    # Re-frozen 2026-10-10 (engine pin retired as a forecast gate, owner ruling): _verify_checkout_root no longer
    # raises on an engine whose sixteen seam files, contract surface or
    # composed glacier unit differ from the pinned engine's; it returns the
    # tree's identity as found (engine_pin.identity_of) and the pin mapping
    # and receipts record matches_measured_row, the moved and the absent
    # paths.  The radius guard no longer re-hashes the sixteen files on each
    # call.  Still refused, by breakage: a tree with no
    # woof/core/mpas_column_batch.py, a woof imported from a tree other
    # than the one named, and an engine whose column batch no longer runs
    # PhysicsDriver.compute / microphysics.apply.  On an engine that IS the
    # pin or an admitted row every recorded value is what it was;
    # _ADAPTER_AUTHORITY and every contract digest are unchanged.  Re-run
    # against this digest: the CPU battery and the forecast named in the
    # report beside the bundle (woof 2.8.9 against the 2.8.7 pin, no local
    # edits).  The x4 full-physics proof did not re-run.
    # Re-frozen 2026-10-13 for physics.mode (correct mode, findings WB4/WB5,
    # branch correct/mpas-water-budget): the sealed constructor admits the
    # optional physics_mode/physics_correct_skip keys and stores them only in
    # correct mode, the backend records the engine's correct_mode receipt.
    # Exact mode seals the same identity and passes the engine the same
    # keywords; measured: hex global 120 km 1 h histories sha256-identical to
    # 53fa27c6 + engine 4d1079249 (correct/mpas-water-budget).  The x4 proof
    # did not re-run.
    # Re-frozen 2026-10-13 (integration hex/correct-mode): the file now holds
    # both correct/mpas-water-budget's physics.mode plumbing and
    # correct/storm-growth's SG2 producer (lagged_heating_v841: stage in
    # finish_step, publish in commit_step, carried in restart state).
    # Re-frozen 2026-10-14 (release 0.4.0, lane H1): correct mode on an engine
    # whose column batch takes no mode keywords is a named gap, not a refusal:
    # the seam is built without the keywords and the receipt names WB4/WB5 as
    # gaps (_engine_correct_mode, _engine_kwargs).  Exact mode builds the seam
    # with the keywords of before.  No card run against this pin.
    # Re-derived 2026-10-14 (H1 merged onto the H7 merge): H1's file plus the
    # 0.4.0 release scrub of two docstring lines in _verify_checkout_root (an
    # owner ruling and a host's hardware in place of names).  No code line
    # differs from H1's copy.  No card run for this merge.
    # Re-frozen 2026-10-17: shared GPU surface kernels now update the ArWen
    # columns from a covering series; daily GFS words and unchanged C12
    # output are checked by the surface lane receipts. No dycore pin moved.
    # 2026-10-18: phase two requests only the columns its seam consumes.
    # Re-derived 2026-10-18 (parked-merge hex-speed): that file plus the
    # parked asynchronous history-ready seam; digest of the merged file.
    "src/hexcore/cuda_arwen_physics_v841.py": (
        "58f169eb8841f81e693bfa5297e67ee309dd61a552413e8b4da234325a50ea99"
    ),
    # New pin 2026-10-05: the New Tiedtke row's seam (the frozen column-batch
    # class subclassed, driver built at cu_physics=16, raw cumulus momentum
    # added to du/dv as native todynamics does).  Executes only when the
    # constructor selects ntiedtke.
    "src/hexcore/cuda_ntiedtke_column_v841.py": (
        "5b2016c35060ba14b17bbc7356e6cf2f8b9c59cd5fc5517f1c99ebe4620d531c"
    ),
    "src/hexcore/cuda_seam_snapshot_v841.py": (
        "a05fdda5ef82b0d77b1af63d4354953660f3d5b80470d7915bb647f41d191019"
    ),
    # The v8.4.1 horizontal-mixing execution boundary (2-D Smagorinsky):
    # CPU authorities and the CUDA operator modules the RK1 saved-Euler
    # mixing runs through.
    # Re-frozen for the regional CPU authority lane: mesh
    # mixing scaling tolerates the regional garbage-cell gather exactly as
    # native atm_compute_mesh_scaling does, and its quarter/three-quarter
    # powers follow the pinned correctly-rounded RKIND convention
    # (woof.hex.rkind_libm; float64 evaluation is a byte no-op).  Every
    # affected proof re-runs against this digest.
    "src/hexcore/mixing.py": (
        "1fd749a6c2a8599acf9bac7b2fc9764c0cb8ceb428be50b19a4e93e3a094d554"
    ),
    # Re-frozen for the regional CPU authority lane: the
    # deformation-weight halo guard now CYCLES exactly as native
    # atm_initialize_deformation_weights does (mpas_atm_core.F:1710-1716)
    # instead of raising, because a regional cull's ring-7 rows legitimately
    # reach the garbage cell and native leaves those weights zero.  Global
    # meshes have no such rows, so every admitted global configuration is
    # byte-unchanged; every affected proof re-runs against this digest.
    # The same lane also routes the three filters through the native padded
    # memory model on a regional mesh (garbage-element columns, stripped
    # after) -- a branch a global mesh never enters, since it has no
    # sentinel cellsOnEdge slot to detect.
    # Re-frozen 2026-10-05 for the startup lane: the deformation weights
    # (84.9 s of the 30 km cold start, 18.3 s of a conus12km warm start) are
    # served by woof.hex.prep_cache under the same rule as the host
    # validations above.  No arithmetic line moved.  Re-run against this
    # digest: the same set named at cuda_driver.py above; the x4
    # full-physics proof and the regional class mints did not re-run.
    # Re-frozen 2026-10-13 (forecast batch): a partition-local mesh takes each
    # deformation-weight row whose inputs are bit-identical to its whole-mesh
    # row from the (cached) whole-mesh weights and computes the rest with the
    # unchanged per-cell code; byte-identical by test and by the 30 derecho
    # member histories named at cuda_driver.py above.
    # Re-frozen 2026-10-16 for hex/global-defects e88492b4: independent
    # deformation rows run across processes, retaining the scalar expressions
    # and GNU libm. Its serial/process byte tests and speed-2's landing decks
    # are current-tree evidence. The archived x4 and regional mints did not
    # re-run and retain their original scope.
    "src/hexcore/mixing_v841.py": (
        "023df21823186b0820c21a1e15f8a9ceb662bd7c77fc4e86e469202c4ec7c749"
    ),
    "src/hexcore/cuda_horizontal.py": (
        "f94137176db65d24df0d9ff1d491c278d6c835bb08cd76dc43c5cfcbf323a345"
    ),
    "src/hexcore/cuda_horizontal_v841.py": (
        "37714ab89935af5b02d79eeea4c06cfad038de457502116c9dad27a8bdd55465"
    ),
    # The two-rank (multi-GPU) execution boundary: the scalar-transport module
    # (which gained the guarded FCT halo-exchange hook and executes in every
    # lane) and the partition modules the 2-GPU runner executes.  Pinned here
    # so a partition-module drift refuses BEFORE CUDA on every lane that
    # verifies frozen sources, exactly as the mixing boundary was pinned.
    # Re-frozen 2026-10-11: the flux-corrected limiter releases each work
    # array after the last kernel that reads it (a one-card 30 km run on
    # bare defaults ran out of device memory at step 18).  Seven `del`
    # statements; no launch, argument or arithmetic line moved.  The x4
    # recorded schedule runs with the limiter off and never enters this
    # routine.  Re-run against this digest: the CPU battery and the
    # one-card 30 km forecasts named in the report beside the bundle.
    # Re-derived 2026-10-14 (0.4.0 release scrub): one comment line names the
    # card host by its hardware; comment only, no launch, argument or
    # arithmetic line moved.
    # Re-frozen 2026-10-19 for d15ea3482cc5: restore the generic kernel
    # cache key used by the LTS interception path. Dimension-specialized
    # keys retain dimensions. CUDA source expressions did not change.
    # This is the current execution identity, not a re-mint of archived x4
    # or regional forecast evidence. Device requalification remains pending.
    "src/hexcore/cuda_transport_v841.py": (
        "178246f5032b512717edf764edaa6d1ddec6f04241f329e6cb6acffe9112acb6"
    ),
    # THE BREAKAGE THESE TWO PREVENT, found by the NVRTC reciprocal-rewrite
    # lane (the NVRTC reciprocal-rewrite defect): a pinned translation unit does not compile its own file.
    # ``cuda_transport_v841`` builds its CUDA source as
    # ``cuda_transport._CUDA_SOURCE + ...``, and every unit in the port
    # prepends ``cuda_fp32.CUDA_FTZ_HELPERS``.  Both were unpinned, so an
    # edit to either changed the compiled bytes of pinned units while every
    # pinned digest still matched and the frozen-source proof still reported
    # frozen.  That is exactly how the NVRTC reciprocal-rewrite defect would have re-entered a
    # pinned unit unnoticed -- and how its remedy landed in cuda_transport.py
    # with the battery green.  Pinned here so the file that actually supplies
    # the bytes has to move deliberately.
    "src/hexcore/cuda_transport.py": (
        "013ed46f8ef3bcc1a445b2e81135e262c23c49950e52e5b62b22e4274644b15e"
    ),
    "src/hexcore/cuda_fp32.py": (
        "ba6b962cac42454846b96d87bba73ef3ceedbf69cdf8ae6a7f880b31f25cc28e"
    ),
    "src/hexcore/partition_assets_v841.py": (
        "ddc7054ba296f5ae6d3e17a601276f698e37f622beb066e17a47f919ba79040b"
    ),
    "src/hexcore/partition_local_mesh_v841.py": (
        "3d860e5bc9695173bcf05efaaea40fdca8e545ae0f88a26a250befbacb3ee3fe"
    ),
    "src/hexcore/partition_state_v841.py": (
        "a504e6f5c5abc2014d40e4a8e3e89885f97d6456ff428c585fbaf57910eccbe8"
    ),
    # Re-frozen 2026-08-25 (stale-guard audit, finding 5): the
    # scheduler's admission floors now answer from woof.hex.device_admission
    # -- the retired 22 GiB linear partition floor and the 20 GiB
    # require_devices default are gone.  No kernel, barrier, or assignment
    # code moved; the digest moves because the pin is exact-byte by design.
    # Every affected frozen proof re-runs against this digest (named
    # follow-up: the partition-invariance gate needs the two-node pair).
    "src/hexcore/partition_device_scheduler_v841.py": (
        "0ad0d541335ffaef95d385c81fdc01c9eaa2d65dff0b3a6d8b442f2812fdfeeb"
    ),
    "src/hexcore/partition_executor_v841.py": (
        "9821e7e3b7c7238d48e40edadbc2c99afbf7e157d08b078649b51cef2c445009"
    ),
    "src/hexcore/partition_net_v841.py": (
        "707a7c455641c45a7e6c42b72c40272cfb184258177feba7a6c8e70414d45134"
    ),
}

# Every value below is a RESTATEMENT of a digest one of the four contract
# modules computes at import, and every one of them is re-derived by
# ``tools/repin_source_tables.py`` (``RESTATED_CONTRACT_DIGESTS``) rather than
# typed.  Six of the seven were wrong on 2026-08-28 and the only test that
# read them asserted them against a second copy of the same literals, so the
# table was compared to itself and passed.
#
# Four moved at the 0.2.0 package rename, because each authority document
# frames its own module name into the digest.  TWO WERE ALREADY STALE BEFORE
# IT, and are named so the next reader does not re-derive them wondering what
# happened.  ``coupling_kernel_sha256`` read a value that appears in no
# source and in no receipt in this tree, while all 564 archived receipts
# carrying that field report the computed one; and
# ``adapter_contract_sha256`` read the value that was live until the 2.5.8
# engine re-pin moved the adapter authority document under it.
KNOWN_CONTRACT_PINS = MappingProxyType(
    {
        "prep_contract_sha256": (
            "a9436205305c4127dc38399288ad47839dc797d0c786e1118e9e800cc7223c1d"
        ),
        "prep_kernel_sha256": (
            "b4caecc0c915bb72b3c677ac9199efe1d41dd6ca70481051217bdbf147a43ae4"
        ),
        "gwdo_contract_sha256": (
            "f3c92686b2b0f86b7076e10bd261a7897ef42b3ebd7f7a60789e2b8ee95e96ef"
        ),
        "gwdo_kernel_sha256": (
            "b334506b289c6f002a2660e6c7796361e39372f6652b8e60268c1400900ab9ec"
        ),
        "coupling_contract_sha256": (
            "3266241aea0d2481093bf7f1ed7617ca83301937dc89fa2229a6dcf30de11b69"
        ),
        "coupling_kernel_sha256": (
            "fb0aa7da49c1a5e7f5d6bacce6503214d1e550b9a60c4faaaeff237e2d719e4b"
        ),
        "adapter_contract_sha256": (
            "dca5e7e7c299cf0c0e2e3cd216d3665f1277c3ff3b459033d168431400089702"
        ),
        "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
    }
)

# The native authority was REGENERATED on 2026-08-20 on the reference node.
# The sealed ``...-20260810a`` bundle was lost -- it is on no reachable
# machine -- so the pins below name the 2026-08-20 rerun instead of the
# 2026-08-10 seal.
#
# The rerun is the same case, not a new one: MPAS-A v8.4.1, the surviving
# ``ifx`` binary sha256 6d33081c69b7d728530278cc2b0b2b262c9f6961dc3c0422b44aefa450d7f332,
# the same ``x4.163842.init.nc`` the ``init`` pin already names
# (sha256 f6e6f413...), 24 MPI ranks, ``config_dt`` 120 s, 30 steps,
# ``config_start_time`` 2026-08-10_12:00:00, history every 30 min,
# ``config_gwdo_scheme = bl_ysu_gwdo``, ``config_convection_scheme =
# cu_grell_freitas``; rc=0 in 173.890 s wall.
#
# It is also bit-exact against the surviving 2026-08-12 run of the identical
# setup: all three history files differ from that run in exactly nine bytes,
# and those nine bytes are the ``file_id`` global attribute MPAS fills with a
# random ten-character string per output file.
#
# The three history pins are therefore MASKED-CONTENT digests: the sha256 of
# the whole file with exactly the ``file_id`` attribute value bytes (located
# via the netCDF header, never a hardcoded offset) hashed as NULs, plus the
# exact byte count.  A bit-exact rerun of the deterministic 24-rank case
# satisfies the pin; a single flipped data byte refuses.  The whole-file
# digest is still computed and recorded in every receipt as provenance of the
# exact kept artifact, but it no longer gates execution -- pinning the random
# nonce is what made the lost 20260810a authority irreplaceable.  The kept
# copies are listed in ``mpas-authority-20260820-MANIFEST.md``.
AUTHORITY_PINS: dict[str, dict[str, Any]] = {
    "grid": {
        "relative_path": (
            "work/v841-vr-static/run-static-v841-conus-official-full-a/"
            "x4.163842.grid.nc"
        ),
        "bytes": 224_139_172,
        "sha256": "48e747157bb1f0b83b96505e268699dfb562b4c1428468cb91457fbb03b1be55",
    },
    "static": {
        "relative_path": (
            "work/v841-vr-static/run-static-v841-conus-official-full-a/"
            "x4.163842.static.nc"
        ),
        "bytes": 298_860_376,
        "sha256": "f064ee8f8d40085db4bf77a3d5fc6081cd92368b7d3dd32d98110b8b64d177e8",
    },
    "init": {
        "relative_path": (
            "work/v841-vr-static/run-real-init-v841-conus-official-full-a/"
            "x4.163842.init.nc"
        ),
        "bytes": 1_489_665_020,
        "sha256": "f6e6f41359554ad3b1103235ec4aef026409b0f085a28a7b0f7c38599b9ca2ba",
    },
    "native_f000": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/history.2026-08-10_12.00.00.nc"
        ),
        "bytes": 1_584_808_024,
        "masked_sha256": (
            "38575bfcbbe581c25ceffeec25d22061b6f22cea2308f639e8fcce093d58da17"
        ),
    },
    "native_f030": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/history.2026-08-10_12.30.00.nc"
        ),
        "bytes": 1_584_808_024,
        "masked_sha256": (
            "1cf267557cf394f0209fbce6a69350e386221d4e791c2ceefc72200d3a45da47"
        ),
    },
    "native_f001": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/history.2026-08-10_13.00.00.nc"
        ),
        "bytes": 1_584_808_024,
        "masked_sha256": (
            "2b867a3352d7580280c01120b5db7fb4e6979be528317770417dff04b9f58b4c"
        ),
    },
    "native_validation_receipt": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/native-gwdo-authority-receipt.json"
        ),
        "bytes": 24_457,
        "sha256": "2fdc8a2e0e5f0b5432773713c6d48e6d3342e59ac2abcebbbf9614fbc500a9bf",
    },
    "native_launch_receipt": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/native-gwdo-launch-receipt.json"
        ),
        "bytes": 1_811,
        "sha256": "ec36f8c9927e024faa40dd942cd0b6ba45cbcd8a90cf7ea4ec4e7576b30c87f8",
    },
    "native_closure": {
        "relative_path": (
            "work/v841-full-physics-gf-gwdo-native-authority-20260820a/"
            "native-run-a/run-closure.status"
        ),
        "bytes": 361,
        "sha256": "0a1cd6a947daa5d943551d3c17808fe85ecd11f6d29984fee5f9482b9e3b2389",
    },
}

NEGATIVE_QV_PIN = MappingProxyType(
    {
        "logical_shape": [N_LEVELS, N_CELLS],
        "full_qv_sha256": (
            "c0180afa0fa99253f414472d8f767421c85b1324f67e161fdef9f6244b565099"
        ),
        "negative_count": 215,
        "negative_indices_sha256": (
            "b622daead41e76d988e5419d92bac9b28a2390e8cfdb619d38ff19c1188a10a4"
        ),
        "negative_values_sha256": (
            "192e06fa6a1302eb06f7148e0cb85483ca61a21caf0eed4bc95ccfcb3452eb4b"
        ),
    }
)

INIT_RECONSTRUCTION_COEFFICIENTS_PIN = MappingProxyType(
    {
        "dimensions": ("nCells", "maxEdges", "R3"),
        "shape": (N_CELLS, 10, 3),
        "dtype": "<f4",
        "static_placeholder_raw_sha256": (
            "b3b09d26538fe509096884906c93ff8b8d2c794300bafcab91449eee1c7bd31c"
        ),
        "init_carrier_raw_sha256": (
            "1d25d2439a6cdcc3cc4a3cabfb5b6720730bb548f4cd88208340cca9df883350"
        ),
        "active_slots": 983_040,
        "active_components": 2_949_120,
        "nonzero_components": 2_949_120,
    }
)


INIT_EDGE_NORMAL_VECTORS_PIN = MappingProxyType(
    {
        "dimensions": ("nEdges", "R3"),
        "shape": (N_EDGES, 3),
        "dtype": "<f4",
        "static_placeholder_raw_sha256": (
            "91e0c31eb2d6776a903dc5456c5e72e1e447c2d835cebbcde87581738cac735b"
        ),
        "init_carrier_raw_sha256": (
            "25d9ef5c70b38a2e7d2c9c60456d9835a1e0fc790b60a3c98d1dbb44482d41da"
        ),
        "nonzero_components": 1_474_550,
        "exact_zero_components": 10,
        "zero_rows": 0,
        "float64_norm_min": 0.999999867111912,
        "float64_norm_max": 1.000000135860109,
    }
)

LANDMASK_CONSTRUCTOR_CAST_PIN = MappingProxyType(
    {
        "source_dimensions": ("nCells",),
        "shape": (N_CELLS,),
        "source_dtype": "<i4",
        "source_array_sha256": (
            "6aee8da961b605b9aa21362840aec214beaa384d42041e232ff8f8a9f60b22b5"
        ),
        "source_unique_values": (0, 1),
        "target_dtype": "<f4",
        "target_array_sha256": (
            "31a7e78b7f0a6b10d719df443b0e2d629e60028c8ac628c5ad8556a87c5eae65"
        ),
        "target_uint32_values": (0, 1_065_353_216),
    }
)

PHYSICS_GEOMETRY_CARRIER_PIN = MappingProxyType(
    {
        "cellsOnEdge": {
            "dimensions": ("nEdges", "TWO"),
            "shape": (N_EDGES, 2),
            "dtype": "<i4",
            "raw_sha256": (
                "a53b1c9bf9e5c0e026c7253b9026f4711bc973acd376f70a6e996adfa584b2d3"
            ),
            "source_roles": ("grid", "static", "init"),
        },
        "east_north": {
            "field_names": ("east", "north"),
            "absent_source_roles": ("grid", "static", "init"),
            "fallback_source_fields": ("lonCell", "latCell"),
            "fallback_dtype": "<f8",
        },
        "edgeNormalVectors": {
            "grid_present": False,
            "static_present": True,
            "init_present": True,
        },
    }
)

NATIVE_FIELD_MAP = MappingProxyType(
    {
        "qv": ("qv", "level_cell"),
        "qc": ("qc", "level_cell"),
        "qr": ("qr", "level_cell"),
        "qi": ("qi", "level_cell"),
        "qs": ("qs", "level_cell"),
        "qg": ("qg", "level_cell"),
        "u_zonal": ("uReconstructZonal", "level_cell"),
        "v_meridional": ("uReconstructMeridional", "level_cell"),
        "normal_u": ("u", "level_edge"),
        "w": ("w", "interface_cell"),
        "rho": ("rho", "level_cell"),
        "theta": ("theta", "level_cell"),
        "pressure": ("pressure", "level_cell"),
        "surface_pressure": ("surface_pressure", "cell"),
        "tsk": ("skintemp", "cell"),
        "t2": ("t2m", "cell"),
        "hfx": ("hfx", "cell"),
        "qfx": ("qfx", "cell"),
        "lh": ("lh", "cell"),
        "u10": ("u10", "cell"),
        "v10": ("v10", "cell"),
        "smois": ("smois", "soil_cell"),
        "tslb": ("tslb", "soil_cell"),
        "rainc": ("rainc", "cell"),
        "rainnc": ("rainnc", "cell"),
        "dusfcg": ("dusfcg", "cell"),
        "dvsfcg": ("dvsfcg", "cell"),
        "dtaux3d": ("dtaux3d", "level_cell"),
        "dtauy3d": ("dtauy3d", "level_cell"),
        "rubldiff": ("rubldiff", "level_cell"),
        "rvbldiff": ("rvbldiff", "level_cell"),
    }
)

# Broad quantitative consistency limits, not parity tolerances.  They catch
# unit/layout/sign disasters while respecting the finalized seam's documented
# pressure, RRTMG, NoahMP-atmosphere, and GF forcing differences.
NATIVE_RMSE_LIMITS = MappingProxyType(
    {
        "qv": 1.0e-2,
        "qc": 5.0e-3,
        "qr": 5.0e-3,
        "qi": 5.0e-3,
        "qs": 5.0e-3,
        "qg": 5.0e-3,
        "u_zonal": 40.0,
        "v_meridional": 40.0,
        "normal_u": 40.0,
        "w": 20.0,
        "rho": 0.5,
        "theta": 30.0,
        "pressure": 5_000.0,
        "surface_pressure": 5_000.0,
        "tsk": 30.0,
        "t2": 30.0,
        "hfx": 1_500.0,
        "qfx": 2.0e-2,
        "lh": 1_500.0,
        "u10": 35.0,
        "v10": 35.0,
        "smois": 0.5,
        "tslb": 30.0,
        "rainc": 100.0,
        "rainnc": 100.0,
        "dusfcg": 150.0,
        "dvsfcg": 150.0,
        "dtaux3d": 0.5,
        "dtauy3d": 0.5,
        "rubldiff": 0.5,
        "rvbldiff": 0.5,
    }
)


class ReleaseNotFrozen(RuntimeError):
    """The executable path has one or more unresolved exact-byte pins."""


class CompositeTransactionError(RuntimeError):
    """A staged Arwen/MPAS composite step failed without being published."""


@dataclass(frozen=True, slots=True)
class CompositeStepResult:
    committed: Any
    backend_receipt: Mapping[str, Any]
    clamp_d2h: Any
    recovery: Any
    # GF's advective forcing formed by THIS step's dynamics; the next step's
    # begin_step consumes it, exactly as native MPAS consumes rthdynten.
    dynamics_tendencies: Any = None


@dataclass(frozen=True, slots=True)
class HostDriverCheckpoint:
    state: Any
    saved_diagnostics: Any
    backend_state: Mapping[str, Any]
    atmosphere_fingerprint: Mapping[str, Any]
    backend_fingerprint: Mapping[str, Any]
    model_time_seconds: float
    # GF's advective forcing (rthdynten/rqvdynten) formed by the checkpoint
    # step's own dynamics.  It is DRIVER-OWNED per-step carried state living
    # outside both the MPAS atmosphere and the Arwen backend restart payload
    # (execute_composite_step hands it forward; the NEXT begin_step consumes
    # it).  A checkpoint without it restores a run whose first resumed step
    # feeds GF zero forcing lanes while the unbroken run feeds the real
    # step-15 forcing -- the deterministic all-arms step-16 divergence
    # measured 5/5 on the reference node (checkpoint schema v3 closes it).
    gf_dynamics_tendencies: Mapping[str, Any]
    gf_forcing_fingerprint: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    selected = Path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"required ordinary file is missing: {selected}")
    # The real SHA-256, computed once per file version (woof.hex.file_digest).
    from woof.hex.file_digest import sha256_file as file_version_sha256
    return file_version_sha256(selected)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    # A C-contiguous array's buffer IS its tobytes(order="C"); hashing the
    # buffer in place skips a full copy and lets hashlib drop the GIL, so
    # fingerprint_nested_arrays can hash arrays on several cores.  Object
    # arrays (no buffer) keep the copy.
    try:
        payload = memoryview(array).cast("B") if array.size else b""
    except (TypeError, ValueError):
        payload = array.tobytes(order="C")
    digest.update(payload)
    return digest.hexdigest()


def _fingerprint_threads() -> int:
    text = os.environ.get("WOOF_HEX_FINGERPRINT_THREADS", "16")
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError("WOOF_HEX_FINGERPRINT_THREADS must be a positive integer")
    return value


def arwen_source_manifest() -> Mapping[str, str]:
    """The sixteen-file sha-freeze of the woof source the seam executes.

    Imported lazily from the pinned adapter module, so callers must verify
    ``EXECUTION_SOURCE_PINS`` before trusting the returned constants.
    """

    from woof.hex.cuda_arwen_physics_v841 import ARWEN_SOURCE_MANIFEST

    return ARWEN_SOURCE_MANIFEST


def checkout_manifest_for(root: Path) -> dict[str, str]:
    """The sixteen-file manifest THIS checkout's seam bytes are bound by.

    The public pin's sixteen digests, unless the tree's seam bytes are, in
    full and under its own version string, the measured row of an engine
    :mod:`woof.hex.engine_pin` admits (``ADMITTED_ENGINES``): then that
    row's digests, so a run on an admitted engine is gated and provenanced
    by the bytes it executes.  The comparison is ``engine_pin.inspect_seam``,
    the same one the adapter, the forecast door and ``doctor`` run, so the
    four surfaces cannot disagree about a tree.  A tree that matches no row
    is measured against the public pin, so the refusal below names the
    digest the pin wants, as it always has.

    THE BREAKAGE THIS CLOSES (measured 2026-09-13): the frozen row's
    ``--preflight`` on an admitted private engine refused at this guard with ``woof/core/physics.py does not match the
    proven manifest (expected 898b694b..., found b3417d87...)`` while the
    adapter, the forecast door and ``doctor`` all admitted the same tree,
    because this guard read the public sixteen unconditionally.
    """

    from woof.hex import engine_pin

    inspection = engine_pin.inspect_seam(root)
    if (
        inspection.engine is None
        or inspection.absent
        or inspection.moved
        or inspection.misdeclared
    ):
        return dict(arwen_source_manifest())
    return dict(engine_pin.identity_for(inspection.engine).manifest)


_NETCDF_FILE_ID_ATTRIBUTE = "file_id"
_NC_CHAR = 2
_NC_DIMENSION = 0x0A
_NC_ATTRIBUTE = 0x0C
_NETCDF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8}


def netcdf_file_id_value_span(path: str | Path) -> tuple[int, int]:
    """Locate the byte span of the ``file_id`` global attribute VALUE.

    Walks the classic netCDF (CDF-1/2/5) header -- magic, record count,
    dimension list, then the global attribute list -- so the span follows
    the header wherever it sits.  A hardcoded offset would silently move the
    mask onto data bytes the moment the header changes shape.
    """

    selected = Path(path)
    with selected.open("rb") as stream:
        head = stream.read(1 << 22)

    if len(head) < 8 or head[:3] != b"CDF" or head[3] not in (1, 2, 5):
        raise RuntimeError(
            f"{selected} is not a classic CDF-1/2/5 netCDF file; the file_id "
            "mask is only defined against the classic header layout"
        )
    count_size = 8 if head[3] == 5 else 4

    def read_int4(offset: int) -> tuple[int, int]:
        if offset + 4 > len(head):
            raise RuntimeError(f"netCDF header of {selected} ends inside a field")
        return int.from_bytes(head[offset : offset + 4], "big"), offset + 4

    def read_count(offset: int) -> tuple[int, int]:
        if offset + count_size > len(head):
            raise RuntimeError(f"netCDF header of {selected} ends inside a count")
        return (
            int.from_bytes(head[offset : offset + count_size], "big"),
            offset + count_size,
        )

    def read_name(offset: int) -> tuple[bytes, int]:
        length, offset = read_count(offset)
        end = offset + length
        if end > len(head):
            raise RuntimeError(f"netCDF header of {selected} ends inside a name")
        return head[offset:end], end + ((-length) % 4)

    position = 4 + count_size  # magic/version, then numrecs (or STREAMING)

    tag, position = read_int4(position)
    count, position = read_count(position)
    if tag not in (0, _NC_DIMENSION) or (tag == 0 and count != 0):
        raise RuntimeError(f"netCDF dimension list of {selected} is malformed")
    for _ in range(count):
        _, position = read_name(position)
        _, position = read_count(position)  # dimension length

    tag, position = read_int4(position)
    count, position = read_count(position)
    if tag not in (0, _NC_ATTRIBUTE) or (tag == 0 and count != 0):
        raise RuntimeError(
            f"netCDF global attribute list of {selected} is malformed"
        )
    for _ in range(count):
        name, position = read_name(position)
        nc_type, position = read_int4(position)
        nelems, position = read_count(position)
        size = _NETCDF_TYPE_SIZES.get(nc_type)
        if size is None:
            raise RuntimeError(
                f"netCDF global attribute {name!r} of {selected} has "
                f"unsupported type {nc_type}"
            )
        if name == _NETCDF_FILE_ID_ATTRIBUTE.encode("ascii"):
            if nc_type != _NC_CHAR:
                raise RuntimeError(
                    f"file_id global attribute of {selected} is not NC_CHAR"
                )
            return position, nelems
        value_bytes = nelems * size
        position += value_bytes + ((-value_bytes) % 4)

    raise RuntimeError(
        "file_id global attribute is not present in the netCDF header of "
        f"{selected}"
    )


def netcdf_masked_digests(path: str | Path) -> dict[str, Any]:
    """Whole-file and file_id-masked SHA-256 of one netCDF file, in one pass.

    The masked digest hashes every byte of the file except the located
    ``file_id`` attribute value, which is hashed as NUL bytes -- exactly the
    masking the regenerated-authority digests were measured with.  MPAS
    stamps a fresh random ten-character ``file_id`` into every output file,
    so a whole-file digest can never survive a rerun while the masked digest
    identifies the content.
    """

    selected = Path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"required ordinary file is missing: {selected}")
    offset, length = netcdf_file_id_value_span(selected)
    raw = hashlib.sha256()
    masked = hashlib.sha256()
    with selected.open("rb") as stream:
        remaining = offset
        while remaining:
            block = stream.read(min(16 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError(f"{selected} ends before the file_id attribute")
            raw.update(block)
            masked.update(block)
            remaining -= len(block)
        value = stream.read(length)
        if len(value) != length:
            raise RuntimeError(f"{selected} ends inside the file_id attribute value")
        raw.update(value)
        masked.update(b"\0" * length)
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            raw.update(block)
            masked.update(block)
    return {
        "sha256": raw.hexdigest(),
        "masked_sha256": masked.hexdigest(),
        "file_id": value.decode("ascii", "replace"),
        "file_id_offset": offset,
        "file_id_len": length,
    }


def _plain_absolute(path: str | Path, label: str) -> Path:
    selected = Path(path).expanduser().absolute()
    if selected.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {selected}")
    return selected


def default_authority_paths() -> dict[str, Path]:
    return {
        role: ROOT / str(pin["relative_path"])
        for role, pin in AUTHORITY_PINS.items()
    }


def _file_record(role: str, path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    selected = _plain_absolute(path, role)
    if not selected.is_file():
        raise FileNotFoundError(f"missing {role} authority: {selected}")
    size = selected.stat().st_size
    if size != int(pin["bytes"]):
        raise RuntimeError(f"{role} byte count changed: {size} != {pin['bytes']}")
    if "masked_sha256" in pin:
        digests = netcdf_masked_digests(selected)
        if digests["masked_sha256"] != pin["masked_sha256"]:
            raise RuntimeError(
                f"{role} masked content digest changed: "
                f"{digests['masked_sha256']} != {pin['masked_sha256']} "
                "(only the random file_id nonce may differ between reruns; "
                "the authority's data bytes moved)"
            )
        return {"path": str(selected), "bytes": size, **digests}
    digest = sha256_file(selected)
    if digest != pin["sha256"]:
        raise RuntimeError(f"{role} SHA-256 changed: {digest} != {pin['sha256']}")
    return {"path": str(selected), "bytes": size, "sha256": digest}


def verify_authorities(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != set(AUTHORITY_PINS):
        raise ValueError("authority path mapping does not cover the exact closed set")
    files = {
        role: _file_record(role, paths[role], AUTHORITY_PINS[role])
        for role in AUTHORITY_PINS
    }
    validation = json.loads(paths["native_validation_receipt"].read_text("utf-8"))
    if validation.get("status") != "passed":
        raise RuntimeError("native GF+YSU-GWDO validation receipt is not passed")
    if validation.get("execution") != {"cuda_launched": False, "device": "CPU"}:
        raise RuntimeError("native authority execution identity changed")
    closure = paths["native_closure"].read_text("ascii")
    for line in (
        "status=passed",
        "gravity_wave_drag=bl_ysu_gwdo",
        "direct_gwd_activity_gate=true",
        "timesteps=30",
        "dt_seconds=120",
    ):
        if line not in closure.splitlines():
            raise RuntimeError(f"native closure lost exact line {line!r}")
    return {
        "files": files,
        "native_validation_status": "passed",
        "sha256": canonical_json_sha256(files),
    }


def unresolved_source_pins() -> tuple[str, ...]:
    return tuple(sorted(path for path, digest in EXECUTION_SOURCE_PINS.items() if digest is None))


def require_frozen_execution_sources() -> dict[str, Any]:
    unresolved = unresolved_source_pins()
    if unresolved:
        raise ReleaseNotFrozen(
            "full-physics CUDA launch remains closed until exact SHA-256 pins "
            f"are released for {list(unresolved)}"
        )
    records: dict[str, Any] = {}
    for relative, expected in EXECUTION_SOURCE_PINS.items():
        assert expected is not None
        # The wheel-reach defect.  Resolved against the hexcore package THIS PROCESS
        # IMPORTED, not against this file's checkout.  Those are the same
        # directory whenever the tool is run from a checkout, which is every
        # historical proof, so no pin moved.  They are NOT the same in the
        # only shape a wheel user can currently assemble -- the package from
        # `pip install`, `--repo` at a checkout for the drivers this file is
        # one of -- and in that shape this gate used to hash the checkout's
        # copy of a module while site-packages' copy was the one launched.
        # It printed "frozen execution sources verified: 23 modules" over
        # bytes that were never executed, which is worse than not checking.
        path = shipped_sources.resolve(relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ReleaseNotFrozen(
                f"frozen execution source changed for {relative} "
                f"(read at {path}): {actual} != {expected}"
            )
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return {"files": records, "sha256": canonical_json_sha256(records)}


def measure_execution_sources() -> dict[str, Any]:
    """Hash this port's own execution sources and record the pin match.

    The FORECAST's call, in place of :func:`require_frozen_execution_sources`.

    RETIRED 2026-10-10 as a forecast gate (owner ruling; Gate law, Refusals law): a forecast used to refuse with
    "frozen execution source changed for <file>: <digest> != <digest>"
    whenever any of this port's own modules differed from the digest
    ``EXECUTION_SOURCE_PINS`` carries.  Two digests name no breakage; they
    say the code was edited since the x4 proof last ran against it.  The
    refusal closed the forecast after every edit to a pinned module until a
    re-pin commit landed (``tools/repin_source_tables.py`` lists three
    occasions; the 2026-10-10 8-card RTX PRO 6000 host proof runs were a fourth, commit
    ``6757d6d``).  The x4 proof keeps the refusing function: byte identity
    of the code under proof is what that harness measures.  A forecast
    records the same files and digests, plus ``matches_pins`` and the moved
    paths, and runs.
    """

    records: dict[str, Any] = {}
    moved: list[str] = []
    for relative, expected in EXECUTION_SOURCE_PINS.items():
        path = shipped_sources.resolve(relative)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            moved.append(relative)
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return {
        "files": records,
        "sha256": canonical_json_sha256(records),
        # Evidence, never a permission.
        "matches_pins": not moved,
        "moved_from_pins": sorted(moved),
    }


def negative_qv_fingerprint(logical_qv: Any) -> dict[str, Any]:
    qv = np.ascontiguousarray(np.asarray(logical_qv, dtype=np.float32))
    negative = np.argwhere(qv < np.float32(0.0)).astype(np.int64, copy=False)
    values = np.ascontiguousarray(qv[qv < np.float32(0.0)])
    return {
        "logical_shape": list(qv.shape),
        "full_qv_sha256": hashlib.sha256(qv.tobytes(order="C")).hexdigest(),
        "negative_count": int(values.size),
        "negative_indices_sha256": hashlib.sha256(negative.tobytes(order="C")).hexdigest(),
        "negative_values_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def augment_cold_scalars(state: Any, scalar_names: Sequence[str], *,
                         source_names: Sequence[str] = SOURCE_SCALAR_NAMES) -> dict[str, Any]:
    """Preserve the initialized scalar prefix and append cold FP32 storage.

    The default source is qv/qc/qr. Callers with initialized ice pass the
    complete source prefix so adding aerosols cannot erase those fields.
    All appended carriers start at +0; the chemistry provider then binds
    its dated initial composition before the first history frame.
    """

    names = tuple(str(name) for name in scalar_names)
    source_names = tuple(source_names)
    if source_names[:3] != SOURCE_SCALAR_NAMES or names[:len(source_names)] != source_names:
        raise ValueError("initialized scalars must be an ordered model prefix")
    if names[: len(SOURCE_SCALAR_NAMES)] != SOURCE_SCALAR_NAMES:
        raise ValueError(
            f"the row's leading species {names[:3]} are not the init's "
            f"source species {SOURCE_SCALAR_NAMES}; the cold-start "
            "augmentation would read the init's block under the wrong names"
        )
    cold_names = names[len(source_names):]
    source = np.asarray(state.scalars)
    expected = (len(source_names), N_LEVELS, N_CELLS)
    if source.dtype != np.dtype(np.float32) or source.shape != expected:
        raise TypeError(f"source qv/qc/qr must be FP32 {expected}, got {source.dtype} {source.shape}")
    if not np.all(np.isfinite(source)):
        raise FloatingPointError("source qv/qc/qr contains non-finite values")
    source_hashes = {
        name: array_sha256(source[index]) for index, name in enumerate(source_names)
    }
    qv_fingerprint = negative_qv_fingerprint(source[0])
    cold = np.zeros((len(cold_names), N_LEVELS, N_CELLS), dtype=np.float32, order="C")
    if cold.size and np.any(cold.view(np.uint32) != np.uint32(0)):
        raise RuntimeError("cold species are not exact +0 FP32")
    augmented = np.concatenate((source, cold), axis=0)
    if augmented.shape != (len(names), N_LEVELS, N_CELLS):
        raise RuntimeError("row augmentation produced the wrong shape")
    state.scalars = np.ascontiguousarray(augmented, dtype=np.float32)
    return {
        "source_present": list(source_names),
        "source_absent": list(cold_names),
        "source_hashes": source_hashes,
        "cold_zero_plane_sha256": array_sha256(cold[0]) if cold.size else None,
        "cold_zero_planes_identical": True,
        "cold_zero_uint32": 0,
        "scalar_order": list(names),
        "negative_qv": qv_fingerprint,
        "negative_qv_preserved_until_candidate_pre_wsm6_clamp": True,
    }


def augment_exact_wsm6_scalars(state: Any) -> dict[str, Any]:
    """Append native-cold +0 FP32 qi/qs/qg without touching qv/qc/qr."""

    source = np.asarray(state.scalars)
    expected = (len(SOURCE_SCALAR_NAMES), N_LEVELS, N_CELLS)
    if source.dtype != np.dtype(np.float32) or source.shape != expected:
        raise TypeError(f"source qv/qc/qr must be FP32 {expected}, got {source.dtype} {source.shape}")
    if not np.all(np.isfinite(source)):
        raise FloatingPointError("source qv/qc/qr contains non-finite values")
    source_hashes = {
        name: array_sha256(source[index]) for index, name in enumerate(SOURCE_SCALAR_NAMES)
    }
    qv_fingerprint = negative_qv_fingerprint(source[0])
    for key, expected_value in NEGATIVE_QV_PIN.items():
        if qv_fingerprint[key] != expected_value:
            raise RuntimeError(f"stock qv fingerprint changed at {key}")
    cold = np.zeros(
        (len(COLD_ZERO_SCALAR_NAMES), N_LEVELS, N_CELLS),
        dtype=np.float32,
        order="C",
    )
    if np.any(cold.view(np.uint32) != np.uint32(0)):
        raise RuntimeError("cold WSM6 species are not exact +0 FP32")
    cold_hash = array_sha256(cold[0])
    augmented = np.concatenate((source, cold), axis=0)
    if augmented.shape != (len(SCALAR_NAMES), N_LEVELS, N_CELLS):
        raise RuntimeError("six-species WSM6 augmentation produced the wrong shape")
    state.scalars = np.ascontiguousarray(augmented, dtype=np.float32)
    return {
        "source_present": list(SOURCE_SCALAR_NAMES),
        "source_absent": list(COLD_ZERO_SCALAR_NAMES),
        "source_hashes": source_hashes,
        "cold_zero_plane_sha256": cold_hash,
        "cold_zero_planes_identical": True,
        "cold_zero_uint32": 0,
        "scalar_order": list(SCALAR_NAMES),
        "negative_qv": qv_fingerprint,
        "negative_qv_preserved_until_candidate_pre_wsm6_clamp": True,
    }


def validate_destination(cache_root: Path, output_root: Path, protected: Sequence[Path]) -> tuple[Path, Path]:
    cache = _plain_absolute(cache_root, "cache root")
    output = _plain_absolute(output_root, "output root")
    if cache == output or cache in output.parents or output in cache.parents:
        raise ValueError("cache and output roots must be disjoint")
    for label, path in (("cache root", cache), ("output root", output)):
        if path.exists():
            raise FileExistsError(f"{label} must be absent: {path}")
        if path == ROOT or ROOT in path.parents and path.parent == ROOT / "src":
            raise ValueError(f"{label} overlaps protected source scope")
        for authority in protected:
            absolute = authority.absolute()
            if path == absolute or path in absolute.parents or absolute in path.parents:
                raise ValueError(f"{label} overlaps protected authority {absolute}")
    return cache, output


def validate_baseline_diagnostic_destination(
    diagnostic_root: Path,
    *,
    cache_root: Path,
    output_root: Path,
    protected: Sequence[Path],
) -> Path:
    """Admit one fresh, isolated, explicitly non-release diagnostic root."""

    diagnostic = _plain_absolute(diagnostic_root, "baseline diagnostic output root")
    for label, other in (("cache root", cache_root), ("output root", output_root)):
        if diagnostic == other or diagnostic in other.parents or other in diagnostic.parents:
            raise ValueError(
                "baseline diagnostic output root must be disjoint from " + label
            )
    if diagnostic.exists():
        raise FileExistsError(
            f"baseline diagnostic output root must be absent: {diagnostic}"
        )
    if diagnostic == ROOT or ROOT in diagnostic.parents and diagnostic.parent == ROOT / "src":
        raise ValueError("baseline diagnostic output root overlaps protected source scope")
    for authority in protected:
        absolute = authority.absolute()
        if (
            diagnostic == absolute
            or diagnostic in absolute.parents
            or absolute in diagnostic.parents
        ):
            raise ValueError(
                "baseline diagnostic output root overlaps protected authority "
                f"{absolute}"
            )
    return diagnostic



def _mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    raise AttributeError(f"mesh has no {name!r}")


def attach_inactive_zero_deformation(mesh: Any) -> dict[str, Any]:
    counts = np.asarray(_mesh_value(mesh, "nEdgesOnCell"))
    n_cells = int(counts.size)
    edges = np.asarray(_mesh_value(mesh, "edgesOnCell"))
    max_edges = int(edges.shape[1] if edges.shape[0] == n_cells else edges.shape[0])
    shape = (n_cells, max_edges)
    arrays = getattr(mesh, "arrays", None)
    if not isinstance(arrays, dict):
        raise TypeError("precision-preserving mesh arrays must be mutable during host preparation")
    if "defc_a" in arrays or "defc_b" in arrays:
        raise ValueError("inactive deformation arrays must be generated exactly once")
    arrays["defc_a"] = np.zeros(shape, dtype=np.float32, order="C")
    arrays["defc_b"] = np.zeros(shape, dtype=np.float32, order="C")
    return {
        "shape": list(shape),
        "dtype": "float32",
        "active_mixing_claim": False,
        "defc_a_sha256": array_sha256(arrays["defc_a"]),
        "defc_b_sha256": array_sha256(arrays["defc_b"]),
    }


def _read_exact_variable(
    dataset: Any,
    name: str,
    *,
    dtype: Any,
    dimensions: tuple[str, ...],
) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"official init is missing required variable {name!r}")
    variable = dataset.variables[name]
    if np.dtype(variable.dtype) != np.dtype(dtype):
        raise TypeError(f"{name} dtype {variable.dtype} != {np.dtype(dtype)}")
    if tuple(variable.dimensions) != dimensions:
        raise ValueError(f"{name} dimensions {variable.dimensions} != {dimensions}")
    variable.set_auto_maskandscale(False)
    value = np.ascontiguousarray(np.asarray(variable[...]))
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value




def load_f000_initialized_surface_diagnostics(
    init_path: Path,
) -> dict[str, Any]:
    """Load exact init-only diagnostics absent from the frozen seam constructor."""

    from netCDF4 import Dataset

    arrays: dict[str, np.ndarray] = {}
    with Dataset(init_path, "r") as dataset:
        for target, pin in F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
            source = str(pin["source"])
            value = _read_exact_variable(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells"),
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            field = np.ascontiguousarray(value[0], dtype=np.float32)
            digest = array_sha256(field)
            if digest != pin["sha256"]:
                raise RuntimeError(
                    f"initialized F000 {target} SHA-256 changed: {digest}"
                )
            arrays[target] = field
    return {
        "arrays": arrays,
        "receipt": {
            "source": "exact hash-pinned official x4.163842 init fields",
            "policy": (
                "F000 snapshot-only replacement of frozen WOOF pre-first-call "
                "exact +0 optional diagnostic placeholders"
            ),
            "fields": {
                target: {
                    "source_variable": pin["source"],
                    "dtype": arrays[target].dtype.str,
                    "shape": list(arrays[target].shape),
                    "sha256": array_sha256(arrays[target]),
                }
                for target, pin in F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items()
            },
        },
    }


def overlay_f000_initialized_surface_diagnostics(
    arrays: dict[str, np.ndarray],
    initialized: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace only exact +0 pre-first-call placeholders with sealed init bytes."""

    expected = set(F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS)
    if set(initialized) != expected:
        raise ValueError("F000 initialized surface diagnostic inventory changed")
    receipt: dict[str, Any] = {}
    for name, pin in F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
        if name not in arrays:
            raise ValueError(f"frozen seam lacks F000 optional diagnostic {name!r}")
        placeholder = np.ascontiguousarray(np.asarray(arrays[name]))
        source = np.ascontiguousarray(np.asarray(initialized[name]))
        if (
            placeholder.dtype != np.dtype(np.float32)
            or placeholder.shape != (N_CELLS,)
            or np.any(placeholder.view(np.uint32) != np.uint32(0))
        ):
            raise ValueError(
                f"frozen seam F000 {name} is not the exact +0 FP32 placeholder"
            )
        if (
            source.dtype != np.dtype(np.float32)
            or source.shape != (N_CELLS,)
            or not np.all(np.isfinite(source))
            or array_sha256(source) != pin["sha256"]
        ):
            raise ValueError(f"sealed initialized F000 {name} identity changed")
        arrays[name] = np.array(source, copy=True, order="C")
        receipt[name] = {
            "source_variable": pin["source"],
            "placeholder_sha256": array_sha256(placeholder),
            "initialized_sha256": array_sha256(source),
            "snapshot_sha256": array_sha256(arrays[name]),
            "source_correct": True,
        }
    return {
        "applied": True,
        "scope": "F000 diagnostic snapshot only; forecast/restart state unchanged",
        "fields": receipt,
    }
def overlay_exact_init_reconstruction_coefficients(
    mesh: Any,
    init_path: Path,
) -> dict[str, Any]:
    """Replace the intentional static zero placeholder with exact init bytes."""

    from netCDF4 import Dataset

    pin = INIT_RECONSTRUCTION_COEFFICIENTS_PIN
    expected_shape = tuple(pin["shape"])
    placeholder = np.asarray(_mesh_value(mesh, "coeffs_reconstruct"))
    if placeholder.dtype != np.dtype(np.float32) or placeholder.shape != expected_shape:
        raise TypeError(
            "static coeffs_reconstruct placeholder must be exact FP32 "
            f"{expected_shape}, got {placeholder.dtype} {placeholder.shape}"
        )
    placeholder = np.ascontiguousarray(placeholder)
    placeholder_raw_sha256 = hashlib.sha256(
        placeholder.tobytes(order="C")
    ).hexdigest()
    if placeholder_raw_sha256 != pin["static_placeholder_raw_sha256"]:
        raise RuntimeError(
            "static reconstruction placeholder bytes changed: "
            f"{placeholder_raw_sha256}"
        )
    if np.any(placeholder.view(np.uint32) != np.uint32(0)):
        raise RuntimeError(
            "static coeffs_reconstruct is no longer the exact all-+0 placeholder"
        )
    prior_source = str(mesh.variable_sources.get("coeffs_reconstruct", ""))
    if prior_source != "static":
        raise RuntimeError(
            "reconstruction placeholder did not come from the sealed static file"
        )

    with Dataset(init_path, "r") as dataset:
        coefficients = _read_exact_variable(
            dataset,
            "coeffs_reconstruct",
            dtype=np.float32,
            dimensions=tuple(pin["dimensions"]),
        )
        init_counts = _read_exact_variable(
            dataset,
            "nEdgesOnCell",
            dtype=np.int32,
            dimensions=("nCells",),
        )
        init_edges_raw = _read_exact_variable(
            dataset,
            "edgesOnCell",
            dtype=np.int32,
            dimensions=("nCells", "maxEdges"),
        )
        variable = dataset.variables["coeffs_reconstruct"]
        variable_attrs = {
            name: getattr(variable, name) for name in variable.ncattrs()
        }

    if coefficients.shape != expected_shape:
        raise ValueError(
            f"initialized coeffs_reconstruct shape changed: {coefficients.shape}"
        )
    coefficients_raw_sha256 = hashlib.sha256(
        coefficients.tobytes(order="C")
    ).hexdigest()
    if coefficients_raw_sha256 != pin["init_carrier_raw_sha256"]:
        raise RuntimeError(
            "initialized reconstruction coefficient bytes changed: "
            f"{coefficients_raw_sha256}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("initialized reconstruction coefficients are non-finite")

    mesh_counts = np.ascontiguousarray(
        np.asarray(_mesh_value(mesh, "nEdgesOnCell"), dtype=np.int32)
    )
    mesh_edges = np.ascontiguousarray(
        np.asarray(_mesh_value(mesh, "edgesOnCell"), dtype=np.int32)
    )
    if mesh_counts.shape != (N_CELLS,) or mesh_edges.shape != expected_shape[:2]:
        raise ValueError("prepared mesh reconstruction topology shape changed")
    if not np.array_equal(init_counts, mesh_counts):
        raise RuntimeError("init/static nEdgesOnCell topology differs")
    if np.any(init_edges_raw < 0):
        raise RuntimeError("initialized source edgesOnCell must use MPAS one-based indexing")
    # Padding is canonicalized from nEdgesOnCell, not from the stored value.
    # THE BREAKAGE THIS PREVENTS: two padding conventions exist in valid MPAS
    # files and both are in the registry.  Native init_atmosphere zeroes the
    # unused slots of edgesOnCell; every GRID file measured -- the published
    # x1.40962 and x4.163842 as much as a generated one -- instead repeats a
    # real one-based index there, and a static built by copying a grid's
    # topology verbatim carries that through into the init minted from it.
    # Trusting the stored value turned an unused slot into a real edge and
    # refused a mesh whose topology is identical, which is exactly the
    # per-file special case a registry of arbitrary meshes cannot afford.
    # The count field is the authority for which slots are used, so this is
    # the same rule woof.hex.mesh applies at load; on a zero-padded native
    # init both rules agree entry for entry, so no proven trajectory moves.
    slot_index = np.arange(init_edges_raw.shape[1], dtype=np.int32)[None, :]
    used = slot_index < init_counts[:, None]
    init_edges = np.where(
        used & (init_edges_raw > np.int32(0)),
        init_edges_raw - np.int32(1),
        np.int32(-1),
    ).astype(np.int32, copy=False)
    init_edges = np.ascontiguousarray(init_edges)
    if np.any(used & (init_edges_raw == np.int32(0))):
        raise RuntimeError("initialized source edgesOnCell has a zero in a USED slot")
    if not np.array_equal(init_edges, mesh_edges):
        raise RuntimeError("init/static edgesOnCell topology differs after canonicalization")

    slots = np.arange(expected_shape[1], dtype=np.int32)[None, :]
    active = slots < mesh_counts[:, None]
    active_slots = int(np.count_nonzero(active))
    active_components = active_slots * expected_shape[2]
    if active_slots != int(pin["active_slots"]):
        raise RuntimeError(f"active reconstruction slot count changed: {active_slots}")
    if active_components != int(pin["active_components"]):
        raise RuntimeError(
            f"active reconstruction component count changed: {active_components}"
        )
    active_values = np.ascontiguousarray(coefficients[active])
    nonzero_components = int(np.count_nonzero(active_values))
    # THE BREAKAGE THIS PREVENTS: the init carries the static's all-+0
    # placeholder in a slot the reconstruction actually uses, so the wind
    # reconstruction there returns the zero vector and nothing says so.  The
    # unit of that breakage is the 3-VECTOR, not the component.  Requiring
    # every component to be nonzero was a property of the native x4 carrier
    # read as a law, and it is false on any mesh with exact symmetry: the
    # generated u96.64002 has 14 exactly-zero components in 1,152,000, every
    # one at a cell sitting exactly on a coordinate plane (lon 0/90/180/270
    # or lat 0), where one Cartesian component is zero because the geometry
    # says so.  A zero 3-vector is still refused, by name and by count.
    zero_vectors = int(
        np.count_nonzero(~np.any(active_values != np.float32(0.0), axis=-1))
    )
    if active_values.size != active_components:
        raise RuntimeError(
            f"active reconstruction component count changed: {active_values.size}"
        )
    if zero_vectors:
        raise RuntimeError(
            f"{zero_vectors} active initialized reconstruction 3-vectors are the "
            "all-+0 static placeholder; the init did not carry computed "
            "coefficients for every used slot"
        )
    if nonzero_components > active_components:
        raise RuntimeError(
            "initialized reconstruction nonzero component count exceeds the "
            f"active slot budget: {nonzero_components} > {active_components}"
        )
    padding_values = np.ascontiguousarray(coefficients[~active])
    if np.any(padding_values.view(np.uint32) != np.uint32(0)):
        raise RuntimeError(
            "initialized reconstruction coefficient padding must be bitwise +0"
        )

    coefficients = np.array(coefficients, copy=True, order="C")
    mesh.arrays["coeffs_reconstruct"] = coefficients
    mesh.variable_sources["coeffs_reconstruct"] = (
        "init_exact_in_memory_physics_reconstruction_overlay"
    )
    mesh.variable_dimensions["coeffs_reconstruct"] = tuple(pin["dimensions"])
    mesh.variable_attrs["coeffs_reconstruct"] = variable_attrs
    receipt = {
        "schema": "mpas-port.v841-init-reconstruction-overlay/v1",
        "field": "coeffs_reconstruct",
        "policy": "static intentional +0 placeholder -> sealed initialized carrier",
        "static_placeholder": {
            "source": prior_source,
            "shape": list(placeholder.shape),
            "dtype": placeholder.dtype.str,
            "raw_c_sha256": placeholder_raw_sha256,
            "nonzero_components": 0,
            "bitwise_positive_zero": True,
        },
        "init_carrier": {
            "path": str(Path(init_path).absolute()),
            "dimensions": list(pin["dimensions"]),
            "shape": list(coefficients.shape),
            "dtype": coefficients.dtype.str,
            "raw_c_sha256": coefficients_raw_sha256,
            "nonzero_components": nonzero_components,
        },
        "topology_identity": {
            "n_edges_on_cell_sha256": array_sha256(mesh_counts),
            "edges_on_cell_sha256": array_sha256(mesh_edges),
            "init_static_bit_identical": True,
            "active_slots": active_slots,
            "active_components": active_components,
        },
        "padding": {
            "components": int(padding_values.size),
            "bitwise_positive_zero": True,
        },
        "active_every_component_nonzero": True,
        "mesh_variable_source": mesh.variable_sources["coeffs_reconstruct"],
        "source_files_mutated": False,
        "overlay_scope": "dynamics mesh in memory only",
    }
    mesh.provenance["v841_init_reconstruction_overlay"] = receipt
    return receipt


def overlay_exact_init_edge_normal_vectors(
    mesh: Any,
    *,
    grid_path: Path,
    static_path: Path,
    init_path: Path,
) -> dict[str, Any]:
    """Bind exact initialized edge normals and receipt every physics carrier."""

    from netCDF4 import Dataset

    normal_pin = INIT_EDGE_NORMAL_VECTORS_PIN
    carrier_pin = PHYSICS_GEOMETRY_CARRIER_PIN
    expected_shape = tuple(normal_pin["shape"])
    placeholder = np.asarray(_mesh_value(mesh, "edgeNormalVectors"))
    if placeholder.dtype != np.dtype(normal_pin["dtype"]) or placeholder.shape != expected_shape:
        raise TypeError(
            "static edgeNormalVectors placeholder must be exact FP32 "
            f"{expected_shape}, got {placeholder.dtype} {placeholder.shape}"
        )
    placeholder = np.ascontiguousarray(placeholder)
    placeholder_raw_sha256 = hashlib.sha256(
        placeholder.tobytes(order="C")
    ).hexdigest()
    if placeholder_raw_sha256 != normal_pin["static_placeholder_raw_sha256"]:
        raise RuntimeError(
            "static edgeNormalVectors placeholder bytes changed: "
            f"{placeholder_raw_sha256}"
        )
    if np.any(placeholder.view(np.uint32) != np.uint32(0)):
        raise RuntimeError(
            "static edgeNormalVectors is no longer the exact all-+0 placeholder"
        )
    prior_source = str(mesh.variable_sources.get("edgeNormalVectors", ""))
    if prior_source != "static":
        raise RuntimeError("edge-normal placeholder did not come from the sealed static file")

    role_paths = {
        "grid": Path(grid_path),
        "static": Path(static_path),
        "init": Path(init_path),
    }
    raw_cells_by_role: dict[str, np.ndarray] = {}
    source_presence: dict[str, Any] = {}
    normals_by_role: dict[str, np.ndarray] = {}
    variable_attrs: dict[str, Any] = {}
    cells_pin = carrier_pin["cellsOnEdge"]
    normal_presence_pin = carrier_pin["edgeNormalVectors"]
    for role, path in role_paths.items():
        with Dataset(path, "r") as dataset:
            raw_cells = _read_exact_variable(
                dataset,
                "cellsOnEdge",
                dtype=np.int32,
                dimensions=tuple(cells_pin["dimensions"]),
            )
            if raw_cells.shape != tuple(cells_pin["shape"]):
                raise ValueError(f"{role} cellsOnEdge shape changed: {raw_cells.shape}")
            raw_cells_sha256 = hashlib.sha256(raw_cells.tobytes(order="C")).hexdigest()
            if raw_cells_sha256 != cells_pin["raw_sha256"]:
                raise RuntimeError(
                    f"{role} cellsOnEdge source bytes changed: {raw_cells_sha256}"
                )
            raw_cells_by_role[role] = raw_cells
            basis_presence = {
                name: name in dataset.variables
                for name in carrier_pin["east_north"]["field_names"]
            }
            if any(basis_presence.values()):
                raise RuntimeError(
                    f"{role} unexpectedly supplies east/north and bypasses the released fallback"
                )
            expected_normal_present = bool(normal_presence_pin[f"{role}_present"])
            actual_normal_present = "edgeNormalVectors" in dataset.variables
            if actual_normal_present != expected_normal_present:
                raise RuntimeError(
                    f"{role} edgeNormalVectors presence changed: {actual_normal_present}"
                )
            source_presence[role] = {
                "east": basis_presence["east"],
                "north": basis_presence["north"],
                "edgeNormalVectors": actual_normal_present,
            }
            if actual_normal_present:
                normals = _read_exact_variable(
                    dataset,
                    "edgeNormalVectors",
                    dtype=np.float32,
                    dimensions=tuple(normal_pin["dimensions"]),
                )
                if normals.shape != expected_shape:
                    raise ValueError(
                        f"{role} edgeNormalVectors shape changed: {normals.shape}"
                    )
                normals_by_role[role] = normals
                if role == "init":
                    variable = dataset.variables["edgeNormalVectors"]
                    variable_attrs = {
                        name: getattr(variable, name) for name in variable.ncattrs()
                    }

    for role in ("static", "init"):
        if not np.array_equal(raw_cells_by_role["grid"], raw_cells_by_role[role]):
            raise RuntimeError(f"grid/{role} raw cellsOnEdge topology differs")
    if not np.array_equal(normals_by_role["static"], placeholder):
        raise RuntimeError("mesh/static edgeNormalVectors placeholder differs")
    initialized = np.ascontiguousarray(normals_by_role["init"])
    initialized_raw_sha256 = hashlib.sha256(
        initialized.tobytes(order="C")
    ).hexdigest()
    if initialized_raw_sha256 != normal_pin["init_carrier_raw_sha256"]:
        raise RuntimeError(
            "initialized edgeNormalVectors bytes changed: "
            f"{initialized_raw_sha256}"
        )
    if not np.all(np.isfinite(initialized)):
        raise FloatingPointError("initialized edgeNormalVectors contains non-finite values")
    nonzero_components = int(np.count_nonzero(initialized))
    exact_zero_components = int(initialized.size - nonzero_components)
    zero_mask = initialized == np.float32(0.0)
    if (
        nonzero_components != int(normal_pin["nonzero_components"])
        or exact_zero_components != int(normal_pin["exact_zero_components"])
        or np.any(initialized.view(np.uint32)[zero_mask] != np.uint32(0))
    ):
        raise RuntimeError("initialized edge-normal component activity changed")
    zero_rows = int(np.count_nonzero(np.all(initialized == np.float32(0.0), axis=1)))
    if zero_rows != int(normal_pin["zero_rows"]):
        raise RuntimeError(f"initialized edge-normal zero-row count changed: {zero_rows}")
    initialized64 = initialized.astype(np.float64)
    norms = np.sqrt(np.sum(initialized64 * initialized64, axis=1, dtype=np.float64))
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    if (
        norm_min != float(normal_pin["float64_norm_min"])
        or norm_max != float(normal_pin["float64_norm_max"])
    ):
        raise RuntimeError(
            f"initialized edge-normal norm envelope changed: {(norm_min, norm_max)}"
        )

    raw_cells = raw_cells_by_role["init"]
    # A stored 0 in a ONE-BASED cellsOnEdge is MPAS's limited-area "this edge
    # has only one cell" sentinel, written by the cull on the outermost ring.
    # On a closed sphere every edge has two cells and a 0 is corruption; on a
    # regional mesh a 0 is the topology.  Zero-basing turns it into -1, which
    # is exactly what the loader already put in ``mesh.cellsOnEdge`` (it is
    # the same file), so the identity check below still has teeth either way.
    #
    # THE BREAKAGE THE REFUSAL PREVENTS: on a closed mesh, a 0 or negative
    # endpoint means the edge's second cell is missing, and every field this
    # overlay feeds -- the edge-normal basis and the cell-to-edge projection
    # of physics tendencies -- would be formed against cell index -1, reading
    # off the front of the array.  Before 2026-08-26 this was a bare
    # ``RuntimeError`` naming none of that, and it was the first thing a
    # culled mesh hit at the production door: the door reported "cellsOnEdge
    # must contain two valid one-based endpoints" when the real state of
    # affairs was that the door had no limited-area route at all.
    regional = bool(getattr(mesh, "is_regional", False))
    if np.any(raw_cells < np.int32(0)) or (
        np.any(raw_cells == np.int32(0)) and not regional
    ):
        offenders = int(np.count_nonzero(raw_cells <= np.int32(0)))
        raise ConfigurationRefusal(
            "cellsOnEdge",
            f"{offenders} endpoint(s) at or below zero",
            "this mesh declares no boundary zone (no bdyMask triple), so every "
            "edge must name two cells; an absent endpoint zero-bases to -1 and "
            "the edge-normal basis and the cell-to-edge tendency projection "
            "would both read cell -1, off the front of the array",
            "a closed global mesh, or a limited-area mesh carrying the "
            "bdyMaskCell/Edge/Vertex triple its cull wrote",
        )
    canonical_cells = np.ascontiguousarray(
        raw_cells.astype(np.int64) - np.int64(1)
    )
    mesh_cells = np.ascontiguousarray(np.asarray(_mesh_value(mesh, "cellsOnEdge")))
    if mesh_cells.shape != tuple(cells_pin["shape"]) or not np.array_equal(
        canonical_cells, mesh_cells
    ):
        raise RuntimeError("source/prepared canonical cellsOnEdge topology differs")

    arrays = getattr(mesh, "arrays", None)
    if not isinstance(arrays, dict):
        raise TypeError("precision-preserving mesh arrays must be mutable during host preparation")
    if "east" in arrays or "north" in arrays:
        raise RuntimeError("prepared mesh unexpectedly supplies east/north source arrays")
    basis_receipt: dict[str, Any] = {}
    for name in carrier_pin["east_north"]["fallback_source_fields"]:
        value = np.ascontiguousarray(np.asarray(_mesh_value(mesh, name)))
        source = str(mesh.variable_sources.get(name, ""))
        if (
            value.dtype != np.dtype(carrier_pin["east_north"]["fallback_dtype"])
            or value.shape != (N_CELLS,)
            or not np.all(np.isfinite(value))
            or not source.startswith("grid_binary64")
        ):
            raise RuntimeError(f"{name} is not the sealed binary64 grid fallback carrier")
        basis_receipt[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "array_sha256": array_sha256(value),
            "mesh_variable_source": source,
        }

    initialized = np.array(initialized, copy=True, order="C")
    mesh.arrays["edgeNormalVectors"] = initialized
    mesh.variable_sources["edgeNormalVectors"] = (
        "init_exact_in_memory_physics_edge_normal_overlay"
    )
    mesh.variable_dimensions["edgeNormalVectors"] = tuple(normal_pin["dimensions"])
    mesh.variable_attrs["edgeNormalVectors"] = variable_attrs
    receipt = {
        "schema": "mpas-port.v841-init-edge-normal-overlay/v1",
        "field": "edgeNormalVectors",
        "policy": "static intentional +0 placeholder -> sealed initialized carrier",
        "static_placeholder": {
            "source": prior_source,
            "shape": list(placeholder.shape),
            "dtype": placeholder.dtype.str,
            "raw_c_sha256": placeholder_raw_sha256,
            "nonzero_components": 0,
            "bitwise_positive_zero": True,
        },
        "init_carrier": {
            "path": str(Path(init_path).absolute()),
            "dimensions": list(normal_pin["dimensions"]),
            "shape": list(initialized.shape),
            "dtype": initialized.dtype.str,
            "raw_c_sha256": initialized_raw_sha256,
            "nonzero_components": nonzero_components,
            "exact_positive_zero_components": exact_zero_components,
            "zero_rows": zero_rows,
            "float64_norm_min": norm_min,
            "float64_norm_max": norm_max,
        },
        "cuda_physics_geometry_carrier_audit": {
            "cellsOnEdge": {
                "dimensions": list(cells_pin["dimensions"]),
                "shape": list(cells_pin["shape"]),
                "source_dtype": cells_pin["dtype"],
                "raw_c_sha256": cells_pin["raw_sha256"],
                "grid_static_init_raw_bit_identical": True,
                "prepared_mesh_zero_based_canonical": True,
                "prepared_array_sha256": array_sha256(mesh_cells),
            },
            "east_north": {
                "source_presence": source_presence,
                "absent_in_grid_static_init_and_prepared_mesh": True,
                "released_fallback": "zonal_meridional_vectors(lonCell, latCell)",
                "fallback_basis": basis_receipt,
            },
            "edgeNormalVectors": {
                "source_presence": source_presence,
                "only_zero_placeholder_trap": True,
                "resolved_by_exact_init_overlay": True,
            },
        },
        "mesh_variable_source": mesh.variable_sources["edgeNormalVectors"],
        "source_files_mutated": False,
        "overlay_scope": "dynamics mesh in memory only",
    }
    mesh.provenance["v841_init_edge_normal_overlay"] = receipt
    return receipt

def derive_area_weighted_p_top_v841(
    *,
    pressure_base: Any,
    pressure_perturbation: Any,
    zgrid: Any,
    area_cell: Any,
) -> tuple[float, dict[str, Any]]:
    """Derive fa35's scalar top pressure from exact native F000 columns.

    The per-column top interface follows the released FP32 preparation
    operations. Only the unavoidable per-column-to-scalar reduction uses
    float64: an exact areaCell weighted mean, rounded once to FP32.
    """

    base = np.asarray(pressure_base)
    perturbation = np.asarray(pressure_perturbation)
    height = np.asarray(zgrid)
    area = np.asarray(area_cell)
    mass_shape = (N_LEVELS, N_CELLS)
    interface_shape = (N_INTERFACES, N_CELLS)
    for name, value, shape in (
        ("pressure_base", base, mass_shape),
        ("pressure_perturbation", perturbation, mass_shape),
        ("zgrid", height, interface_shape),
    ):
        if value.dtype != np.dtype(np.float32) or value.shape != shape:
            raise TypeError(f"{name} must be exact FP32 {shape}")
        if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite and C-contiguous")
    if area.dtype != np.dtype(np.float32) or area.shape != (N_CELLS,):
        raise TypeError("areaCell must be exact FP32 [cell]")
    if not area.flags.c_contiguous or not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("areaCell must be finite, positive, and C-contiguous")

    pressure = np.add(base, perturbation, dtype=np.float32)
    half = np.float32(0.5)
    one = np.float32(1.0)
    z0 = height[-1]
    z1 = np.multiply(
        half,
        np.add(height[-1], height[-2], dtype=np.float32),
        dtype=np.float32,
    )
    z2 = np.multiply(
        half,
        np.add(height[-2], height[-3], dtype=np.float32),
        dtype=np.float32,
    )
    w1 = np.divide(
        np.subtract(z0, z2, dtype=np.float32),
        np.subtract(z1, z2, dtype=np.float32),
        dtype=np.float32,
    )
    w2 = np.subtract(one, w1, dtype=np.float32)
    logarithm = np.add(
        np.multiply(
            w1, np.log(pressure[-1], dtype=np.float32), dtype=np.float32
        ),
        np.multiply(
            w2, np.log(pressure[-2], dtype=np.float32), dtype=np.float32
        ),
        dtype=np.float32,
    )
    top = np.ascontiguousarray(np.exp(logarithm, dtype=np.float32))
    if not np.all(np.isfinite(top)) or np.any(top <= 0):
        raise FloatingPointError("derived native F000 top pressure is invalid")
    area64 = area.astype(np.float64, copy=False)
    weighted_mean64 = float(
        np.sum(top.astype(np.float64) * area64, dtype=np.float64)
        / np.sum(area64, dtype=np.float64)
    )
    scalar = np.float32(weighted_mean64)
    if scalar.view(np.uint32) != EXPECTED_ARWEN_P_TOP_PA_F32.view(np.uint32):
        raise RuntimeError(
            "exact F000 area-weighted p_top changed: "
            f"{float(scalar)} != {float(EXPECTED_ARWEN_P_TOP_PA_F32)}"
        )
    observed = (float(np.min(top)), float(np.median(top)), float(np.max(top)))
    for label, actual, expected in zip(
        ("minimum", "median", "maximum"),
        observed,
        EXPECTED_TOP_PRESSURE_RANGE_PA,
        strict=True,
    ):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5.0e-4):
            raise RuntimeError(f"exact F000 top pressure {label} changed: {actual}")
    receipt = {
        "source": "released FP32 prepare_mpas_to_phys pres2_p[top]",
        "per_column_shape": [N_CELLS],
        "per_column_sha256": array_sha256(top),
        "per_column_minimum_pa": observed[0],
        "per_column_median_pa": observed[1],
        "per_column_maximum_pa": observed[2],
        "area_cell_sha256": array_sha256(area),
        "area_weighted_mean_f64_pa": weighted_mean64,
        "area_weighted_mean_f32_pa": float(scalar),
        "area_weighted_mean_f32_uint32": int(scalar.view(np.uint32)),
        "reduction": "sum(float64(p_top_column)*float64(areaCell))/sum(float64(areaCell))",
        "limitation": "native per-column plrad reduced to one fa35 constructor scalar",
        "claimed_native_identity": False,
        "cavallo_buffer_layers": 3,
    }
    return float(scalar), receipt


def build_arwen_constructor_values(
    *,
    init_path: Path,
    mesh: Any,
    vertical: Any,
    reference: Any,
    saved_diagnostics: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Map exact official x4 init carriers to the finalized fa35 constructor."""

    from netCDF4 import Dataset

    with Dataset(init_path, "r") as dataset:
        missing_species = sorted(
            name for name in COLD_ZERO_SCALAR_NAMES if name in dataset.variables
        )
        if missing_species:
            raise RuntimeError(
                "official cold-start assumption changed; unexpected variables "
                f"{missing_species}"
            )
        landmask_i = _read_exact_variable(
            dataset, "landmask", dtype=np.int32, dimensions=("nCells",)
        )
        ivgtyp = _read_exact_variable(
            dataset, "ivgtyp", dtype=np.int32, dimensions=("nCells",)
        )
        isltyp = _read_exact_variable(
            dataset, "isltyp", dtype=np.int32, dimensions=("nCells",)
        )
        xland_source = _read_exact_variable(
            dataset,
            "xland",
            dtype=np.float32,
            dimensions=("Time", "nCells"),
        )
        if xland_source.shape != (1, N_CELLS):
            raise ValueError(f"xland shape changed: {xland_source.shape}")
        xland = np.ascontiguousarray(xland_source[0], dtype=np.float32)
        zgrid = _read_exact_variable(
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
            value = _read_exact_variable(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells"),
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            surface[target] = np.ascontiguousarray(value[0], dtype=np.float32)
        soil: dict[str, np.ndarray] = {}
        for source, target in (("tslb", "soil_temperature"), ("smois", "soil_moisture")):
            value = _read_exact_variable(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells", "nSoilLevels"),
            )
            if value.shape != (1, N_CELLS, N_SOIL_LEVELS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            soil[target] = np.ascontiguousarray(value[0].T, dtype=np.float32)
        nominal_min_dc = _read_exact_variable(
            dataset, "nominalMinDc", dtype=np.float32, dimensions=()
        )
        start_text = str(getattr(dataset, "config_start_time", ""))

    source_xland_sha256 = array_sha256(xland_source)
    flat_xland_sha256 = array_sha256(xland)
    xland_unique, xland_counts = np.unique(xland, return_counts=True)
    if (
        source_xland_sha256 != NATIVE_XLAND_SOURCE_SHA256
        or flat_xland_sha256 != NATIVE_XLAND_FLAT_SHA256
        or tuple(float(value) for value in xland_unique) != (1.0, 2.0)
        or tuple(int(value) for value in xland_counts) != (61_528, 102_314)
    ):
        raise RuntimeError("sealed native xland identity changed")
    xice = surface["xice"]
    sea_ice_mask = np.ascontiguousarray(xice >= ARWEN_XICE_THRESHOLD)
    open_water_mask = np.ascontiguousarray(
        (xland >= np.float32(1.5)) & ~sea_ice_mask
    )
    land_mask = np.ascontiguousarray(~(sea_ice_mask | open_water_mask))
    glacier_mask = np.ascontiguousarray(land_mask & (ivgtyp == np.int32(15)))
    sflx_land_mask = np.ascontiguousarray(land_mask & ~glacier_mask)
    surface_classification = {
        "xland_source": "native",
        "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
        "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": int(np.count_nonzero(sea_ice_mask)),
        "open_water_columns": int(np.count_nonzero(open_water_mask)),
        "sflx_land_columns": int(np.count_nonzero(sflx_land_mask)),
        "glacier_columns": int(np.count_nonzero(glacier_mask)),
    }
    if surface_classification != dict(EXPECTED_SURFACE_CLASSIFICATION):
        raise RuntimeError(
            f"real-x4 surface classification changed: {surface_classification}"
        )
    glacier_indices = np.ascontiguousarray(np.flatnonzero(glacier_mask))
    sea_ice_indices = np.ascontiguousarray(np.flatnonzero(sea_ice_mask))
    threshold_delta_indices = np.ascontiguousarray(
        np.flatnonzero(
            (xice >= ARWEN_XICE_THRESHOLD) & (xice < np.float32(0.5))
        )
    )
    if (
        glacier_indices.size != 3_010
        or int(glacier_indices[0]) != 30
        or array_sha256(glacier_indices) != GLACIER_INDEX_SHA256
        or sea_ice_indices.size != 4_643
        or array_sha256(sea_ice_indices) != SEA_ICE_INDEX_SHA256
        or threshold_delta_indices.size != 1_067
        or array_sha256(threshold_delta_indices) != THRESHOLD_DELTA_INDEX_SHA256
    ):
        raise RuntimeError("real-x4 glacier/sea-ice index identity changed")
    if not (
        np.all(ivgtyp[sea_ice_mask] == np.int32(15))
        and np.all(landmask_i[sea_ice_mask] == np.int32(0))
        and np.all(xland[sea_ice_mask] == np.float32(1.0))
        and np.all(ivgtyp[glacier_mask] == np.int32(15))
        and np.all(landmask_i[glacier_mask] == np.int32(1))
        and np.all(xland[glacier_mask] == np.float32(1.0))
        and np.all(xice[glacier_mask] == np.float32(0.0))
    ):
        raise RuntimeError("real-x4 native xland/ice category signature changed")
    surface_classification_receipt = {
        **surface_classification,
        "source_field": "init xland[0,:]",
        "source_shape": list(xland_source.shape),
        "source_array_sha256": source_xland_sha256,
        "constructor_shape": list(xland.shape),
        "constructor_array_sha256": flat_xland_sha256,
        "first_glacier_column": int(glacier_indices[0]),
        "glacier_index_sha256": array_sha256(glacier_indices),
        "sea_ice_index_sha256": array_sha256(sea_ice_indices),
        "threshold_0p02_to_0p5_delta_columns": int(threshold_delta_indices.size),
        "threshold_delta_index_sha256": array_sha256(threshold_delta_indices),
        "native_xland_consumed_verbatim": True,
    }

    if np.float32(nominal_min_dc).view(np.uint32) != NOMINAL_DX_M.view(np.uint32):
        raise RuntimeError("official x4 nominalMinDc is not exact FP32 25000 m")
    if start_text != START_TIME_TEXT:
        raise RuntimeError(f"official init start time changed: {start_text!r}")
    landmask_pin = LANDMASK_CONSTRUCTOR_CAST_PIN
    source_landmask_sha256 = array_sha256(landmask_i)
    source_unique_values = tuple(int(value) for value in np.unique(landmask_i))
    if (
        landmask_i.dtype != np.dtype(landmask_pin["source_dtype"])
        or landmask_i.shape != tuple(landmask_pin["shape"])
        or source_landmask_sha256 != landmask_pin["source_array_sha256"]
        or source_unique_values != tuple(landmask_pin["source_unique_values"])
    ):
        raise RuntimeError("sealed init landmask source identity changed")
    landmask = np.ascontiguousarray(landmask_i, dtype=np.float32)
    target_landmask_sha256 = array_sha256(landmask)
    target_uint32_values = tuple(
        int(value) for value in np.unique(landmask.view(np.uint32))
    )
    if (
        landmask.dtype != np.dtype(landmask_pin["target_dtype"])
        or target_landmask_sha256 != landmask_pin["target_array_sha256"]
        or target_uint32_values != tuple(landmask_pin["target_uint32_values"])
        or not np.array_equal(landmask.astype(np.int32), landmask_i)
    ):
        raise RuntimeError("landmask int32 -> FP32 constructor cast is not exact")
    landmask_receipt = {
        "source_field": "init landmask",
        "source_dimensions": list(landmask_pin["source_dimensions"]),
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
    }

    if zgrid.shape != (N_CELLS, N_INTERFACES):
        raise ValueError(f"zgrid shape changed: {zgrid.shape}")

    lat = np.asarray(_mesh_value(mesh, "latCell"), dtype=np.float64)
    lon = np.asarray(_mesh_value(mesh, "lonCell"), dtype=np.float64)
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

    # GF's per-cell length scale, built by native's own construction and the
    # same one this port already feeds GWDO: len_disp / meshDensity**0.25,
    # with a non-positive config_len_disp resolved to the mesh nominalMinDc.
    from woof.hex.cuda_gwdo_v841 import native_cell_dx_m

    dx_column_m = native_cell_dx_m(
        _mesh_value(mesh, "meshDensity"), float(NOMINAL_DX_M)
    )
    if dx_column_m.shape != (N_CELLS,):
        raise ValueError(
            f"per-cell GF dx must have shape {(N_CELLS,)}, got {dx_column_m.shape}"
        )

    p_top_pa, p_top_receipt = derive_area_weighted_p_top_v841(
        pressure_base=reference.pressure_base,
        pressure_perturbation=saved_diagnostics.pressure_perturbation,
        zgrid=vertical.zgrid,
        area_cell=np.ascontiguousarray(_mesh_value(mesh, "areaCell")),
    )

    values: dict[str, Any] = {
        "n_levels": N_LEVELS,
        "n_columns": N_CELLS,
        "dt": DT_SECONDS,
        "radiation_seconds": 600.0,
        "surface_pbl_seconds": DT_SECONDS,
        "cumulus_seconds": DT_SECONDS,
        "cumulus_scheme": "gf",
        # This proof IS the WSM6 authority case: its cold-start species,
        # its native CPU counterpart and its snapshot digests are all the
        # WSM6 row's, so the seam row is spelled once here beside the
        # WSM6 hail knob rather than derived from a config this driver
        # never reads.
        "microphysics_scheme": "wsm6",
        "start_time": datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S"),
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "terrain_height_m": terrain,
        "z_interface_nominal_m": nominal_z,
        "p_top_pa": p_top_pa,
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_m": dx_column_m,
        # Native MPAS v8.4.1 hardwires GF's shallow scheme on
        # (mpas_atmphys_vars.F:340).
        "gf_ishallow": 1,
        "landmask": landmask,
        "xland": xland,
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "ivgtyp": np.ascontiguousarray(ivgtyp, dtype=np.int32),
        "isltyp": np.ascontiguousarray(isltyp, dtype=np.int32),
        **surface,
        **soil,
        "wsm6_hail_opt": 0,
    }
    arrays = {name: value for name, value in values.items() if isinstance(value, np.ndarray)}
    receipt = {
        "source": "exact official x4.163842 initialized fields",
        "mapping": {
            "latitude_deg": "reconciled mesh latCell radians -> FP32 degrees",
            "longitude_deg": "reconciled mesh lonCell radians -> FP32 degrees",
            "terrain_height_m": "init zgrid[:,0]",
            "landmask": "init int32 {0,1} -> exact FP32 sealed-constructor cast",
            "xland": "init native xland[0,:] consumed verbatim",
            "xice_threshold": "explicit MPAS config_frac_seaice threshold 0.02",
            "dx_column_m": (
                "native per-cell GF length scale len_disp/meshDensity**0.25, "
                "config_len_disp=0 resolved to nominalMinDc"
            ),
            "z_interface_nominal_m": "loaded native vertical.zw",
            "tsk": "init skintemp[0,:]",
            "snow_depth": "init snowh[0,:] (m)",
            "soil_temperature": "init tslb[0,:,:].T",
            "soil_moisture": "init smois[0,:,:].T",
        },
        "p_top_pa": p_top_pa,
        "landmask_exact_cast": landmask_receipt,
        "surface_classification": surface_classification_receipt,
        "p_top_derivation": p_top_receipt,
        "p_top_policy": "exact areaCell-weighted F000 native pres2_p top",
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_policy": "native len_disp/meshDensity**0.25 per cell",
        "dx_column_min_m": float(dx_column_m.min()),
        "dx_column_max_m": float(dx_column_m.max()),
        "gf_ishallow": 1,
        "defaults_used": False,
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        },
    }
    static_for_gwdo = {
        "meshDensity": np.asarray(_mesh_value(mesh, "meshDensity")),
        "nominalMinDc": np.asarray(NOMINAL_DX_M),
    }
    for name in ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"):
        # These exact initialized statics are not necessarily retained on the
        # precision-preserving grid/static overlay, so read them from init.
        with Dataset(init_path, "r") as dataset:
            static_for_gwdo[name] = _read_exact_variable(
                dataset, name, dtype=np.float32, dimensions=("nCells",)
            )
    return values, receipt, static_for_gwdo


def _phase_from_receipt(backend: Any) -> str:
    receipt = dict(backend.step_receipt())
    phase = receipt.get("phase")
    if not isinstance(phase, str):
        raise RuntimeError("backend step receipt does not expose a phase string")
    return phase


def require_staged_backend_api(backend: Any) -> dict[str, Any]:
    required = (
        "begin_step",
        "finish_step",
        "commit_step",
        "abort_step",
        "restart_state",
        "restore_restart_state",
        "step_receipt",
        "diagnostic_snapshot",
    )
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise ReleaseNotFrozen(f"frozen adapter lacks staged API methods {missing}")
    phase = _phase_from_receipt(backend)
    if phase not in ("boundary", "restored", "complete"):
        raise RuntimeError(f"backend is not at a committed boundary: {phase!r}")
    return {"required_methods": list(required), "initial_phase": phase}


_BACKEND_TRANSACTION_IDENTITY_KEYS = (
    "schema",
    "adapter_contract_sha256",
    "coupling_contract_sha256",
    "contract_document_sha256",
    "contract_surface_sha256",
    "glacier_composed_tu_sha256",
    "arwen_commit",
    "arwen_source_manifest",
    "dependencies",
    "constructor",
)


def _backend_transaction_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: receipt[name]
        for name in _BACKEND_TRANSACTION_IDENTITY_KEYS
        if name in receipt
    }


def _drag_composed(receipt: Mapping[str, Any]) -> bool:
    """Whether the receipt says hex's own orographic drag is composed."""

    gwdo = receipt.get("gwdo")
    return isinstance(gwdo, Mapping) and gwdo.get("enabled") is True


def _verify_backend_rollback_boundary(
    backend: Any,
    *,
    expected_phase: str,
    start_time: float,
    transaction_identity: Mapping[str, Any],
) -> None:
    """Prove that an explicit or automatic adapter rollback reached its boundary."""

    receipt = dict(backend.step_receipt())
    phase = receipt.get("phase")
    if phase != expected_phase:
        raise RuntimeError(
            f"adapter rollback receipt phase {phase!r} != {expected_phase!r}"
        )
    if _backend_transaction_identity(receipt) != dict(transaction_identity):
        raise RuntimeError("adapter identity changed across the failed transaction")
    if float(receipt.get("start_time_seconds", math.nan)) != start_time:
        raise RuntimeError("adapter rollback receipt lost the exact step start time")

    # restart_state is legal only at an internal committed boundary. Requiring
    # its exact public schema prevents an automatic_rollback receipt alone
    # from being mistaken for a successfully restored seam.
    restart = backend.restart_state()
    if not isinstance(restart, Mapping) or set(restart) - {"lagged_heating", "surface_updates"} != {
        "schema",
        "identity",
        "seam",
        "adapter",
    }:
        raise RuntimeError("adapter rollback did not expose a boundary restart state")
    if restart["schema"] != receipt.get("schema"):
        raise RuntimeError("adapter rollback restart schema changed")
    restart_identity = restart["identity"]
    seam = restart["seam"]
    if not isinstance(restart_identity, Mapping) or not isinstance(seam, Mapping):
        raise RuntimeError("adapter rollback restart identity/state is malformed")
    if set(seam) != {"identity", "arrays", "scalars"}:
        raise RuntimeError("adapter rollback seam state schema changed")
    if not isinstance(seam["identity"], Mapping) or not isinstance(
        seam["scalars"], Mapping
    ):
        raise RuntimeError("adapter rollback seam identity/scalars are malformed")
    if float(seam["scalars"].get("elapsed_seconds", math.nan)) != start_time:
        raise RuntimeError("adapter rollback did not restore the exact boundary time")

    dependencies = receipt.get("dependencies", {})
    constructor = receipt.get("constructor", {})
    expected_restart_identity = {
        "adapter_contract_sha256": receipt.get("adapter_contract_sha256"),
        "arwen_commit": receipt.get("arwen_commit"),
        "arwen_source_manifest": receipt.get("arwen_source_manifest"),
        "contract_surface_sha256": receipt.get("contract_surface_sha256"),
        "glacier_composed_tu_sha256": receipt.get("glacier_composed_tu_sha256"),
        "constructor_identity_sha256": (
            constructor.get("identity_sha256")
            if isinstance(constructor, Mapping)
            else None
        ),
        "prep_contract_sha256": (
            dependencies.get("prep_contract_sha256")
            if isinstance(dependencies, Mapping)
            else None
        ),
        # The receipt's dependencies name the drag contract the adapter
        # links whether or not the drag runs; the restart identity names it
        # only when hex's drag is composed (gwdo.enabled), which is the
        # fact a rollback must preserve.  Read it under the same rule.
        # THE BREAKAGE: every rollback with hex's drag off (any suite or
        # generic-seam backend) failed here on 'gwdo_contract_sha256' and
        # buried the refusal that caused it (30 km global at TEMPO,
        # 2026-10-11).
        "gwdo_contract_sha256": (
            dependencies.get("gwdo_contract_sha256")
            if isinstance(dependencies, Mapping) and _drag_composed(receipt)
            else None
        ),
    }
    for name, expected in expected_restart_identity.items():
        if name == "gwdo_contract_sha256":
            if restart_identity.get(name) != expected:
                raise RuntimeError(
                    f"adapter rollback restart identity changed field {name!r}"
                )
            continue
        if expected is not None and restart_identity.get(name) != expected:
            raise RuntimeError(
                f"adapter rollback restart identity changed field {name!r}"
            )


@timed_arwen_step
def execute_composite_step(
    *,
    driver: Any,
    backend: Any,
    scalar_names: Sequence[str],
    physics_geometry: Any,
    kernel_cache: Any,
    previous_surface_updates: Mapping[str, Any] | None,
    dynamics_tendencies: Any = None,
    couple: Callable[..., Any] | None = None,
    clamp: Callable[..., Any] | None = None,
    recover: Callable[..., Any] | None = None,
    physics_park: Any = None,
    refl_10cm_due: bool = False,
    before_phase_two: Callable[[Any], None] | None = None,
    held_forcing: Callable[..., None] | None = None,
) -> CompositeStepResult:
    """Execute exactly one staged two-owner full-physics transaction.

    ``held_forcing`` (incremental analysis update, ``woof.hex.iau_native`` or
    ``woof.hex.iau_forcing``) adds its share of an analysis increment to the held
    physics tendencies after the coupling and before the dynamics, the slot
    MPAS-A's ``atm_add_tend_anal_incr`` writes.  ``None`` (every run without an
    IAU window) leaves the step exactly as before.

    ``refl_10cm_due`` is WRF's history-step ``diagflag`` for this step: the
    phase-two microphysics computes ``refl10cm`` at its own call time (the
    native MPAS-A v8.4.1 arrangement) and the committed field is consumed by
    the history capture that follows this step.

    ``physics_park`` is an optional
    :class:`woof.hex.cuda_physics_tier_park_v841.CudaPhysicsTierParkV841`.
    When supplied, the physics seam's device residency is moved to pinned host
    memory across ``driver.step_device_with_physics`` -- the dynamics half of
    the step, which is where the run's device high-water mark is reached and
    where nothing reads the physics tier.  Default ``None`` runs exactly as
    before, allocation for allocation.
    """

    # The block must be a DECLARED row in that row's order -- the frozen
    # WSM6 six, P3's eight, or a registered row-with-extras; an undeclared
    # block refuses by name (woof.hex.species_row).  This asserted WSM6's
    # exact six until the row generalization reached this step.
    from woof.hex.species_row import species_row_for_names

    species_row_for_names(tuple(scalar_names))
    if couple is None or clamp is None or recover is None:
        from woof.hex.cuda_physics_v841 import (
            clamp_wsm6_scalars_in_place_v841,
            couple_raw_column_physics_v841,
            recover_post_rk_wsm6_state_v841,
        )

        couple = couple or couple_raw_column_physics_v841
        clamp = clamp or clamp_wsm6_scalars_in_place_v841
        recover = recover or recover_post_rk_wsm6_state_v841

    candidate = None
    backend_active = False
    driver_committed = False
    start_atmosphere = driver.atmosphere
    start_state = start_atmosphere.state
    start_time = float(start_state.time_seconds)
    transaction_identity = _backend_transaction_identity(
        dict(backend.step_receipt())
    )
    # Device memory a physics phase does not use is given back at its two
    # boundaries when the card is short of memory (woof.hex.device_memory_release;
    # knobs WOOF_HEX_POOL_RELEASE, GPUWM_HEX_LMEM_RELEASE,
    # GPUWM_HEX_RELEASE_HEADROOM): then radiation blocks and the WSM6 frame
    # store are not held through the dycore half of the step.
    from woof.hex.device_memory_release import phase_release

    try:
        with phase_release("arwen:phase1"):
            raw = backend.begin_step(
                atmosphere=driver.atmosphere,
                scalar_names=scalar_names,
                dt=DT_SECONDS,
                dynamics_tendencies=dynamics_tendencies,
            )
        backend_active = True
        les = getattr(driver, "les_dissipation", None)
        if les is not None:
            backend.refresh_les(les)
        edge_fields = driver.horizontal.recover_edge_fields(
            start_state.rho, start_state.rho_u
        )
        held = couple(
            raw,
            state=start_state,
            scalar_names=scalar_names,
            geometry=physics_geometry,
            rho_edge=edge_fields.rho_edge,
            kernel_cache=kernel_cache,
        )
        # SG2: the previous step's microphysics heating rides every RK stage
        # of this step (MPAS rt_diabatic_tend, mpas_atm_time_integration.F:6937).
        from woof.hex.cuda_physics_v841 import with_lagged_diabatic_theta_v841
        held = with_lagged_diabatic_theta_v841(held, backend)
        stochastic = getattr(driver, "stochastic", None)
        if stochastic is not None:
            stochastic.apply(held, state=start_state, rho_edge=edge_fields.rho_edge,
                             normal_velocity=edge_fields.normal_velocity,
                             time_seconds=start_time, dt=DT_SECONDS, names=tuple(scalar_names))
        if held_forcing is not None:
            held_forcing(held, start_time, DT_SECONDS, driver=driver,
                         rho_edge=edge_fields.rho_edge,
                         normal_velocity=edge_fields.normal_velocity,
                         geometry=physics_geometry, state=start_state)
        if physics_park is not None:
            # The seam is not read again until finish_step.  Everything still
            # in flight -- the raw phase-1 tendencies, the coupled carriers,
            # the driver's own state -- is named here so the park's sharing
            # rule refuses to rebind any allocation the dycore still writes.
            physics_park.park(shared_roots=(driver, raw, held, edge_fields))
        # Coupling and optional IAU/parking have consumed these carriers.
        # In particular, neither recovered edge array belongs to the dycore
        # transaction.  Returning them to the stream's pool here avoids
        # retaining two full edge volumes across all three RK stages.
        del edge_fields, raw
        candidate = driver.step_device_with_physics(held)
        # The candidate owns the endpoint and next-step forcing.  Phase two
        # and rollback never read the phase-one conservative tendencies.
        del held
        # Read GF's next-step advective forcing here, BEFORE publication:
        # nothing after the driver commit is allowed to raise, and a dry
        # candidate legitimately carries none.  A driver that stops
        # producing the carrier mid-run is caught by the adapter's own
        # mid-run refusal at the next begin_step, not silently ignored.
        next_dynamics_tendencies = getattr(candidate, "dynamics_tendencies", None)
        clamp_d2h = clamp(
            candidate.atmosphere.state.scalars,
            scalar_names=scalar_names,
            kernel_cache=kernel_cache,
        )
        if physics_park is not None:
            physics_park.unpark()
        if before_phase_two is not None:
            before_phase_two(candidate.atmosphere)
        with phase_release("arwen:phase2"):
            update = backend.finish_step(
                atmosphere=candidate.atmosphere,
                scalar_names=scalar_names,
                dt=DT_SECONDS,
                refl_10cm_due=refl_10cm_due,
            )
        if _phase_from_receipt(backend) != "finished_unpublished":
            raise RuntimeError("adapter finish did not remain staged and unpublished")
        recovery = recover(
            candidate.atmosphere.state,
            update,
            scalar_names=scalar_names,
            kernel_cache=kernel_cache,
            phase2_dt_seconds=DT_SECONDS,
            previous_surface_updates=previous_surface_updates,
        )
        # Recovery retains the published surface/radius outputs and state.
        # Drop the consumed update before commit rebuilds saved diagnostics,
        # so its temporary dry theta need not overlap that workspace.
        del update
        committed = driver.commit_post_wsm6_candidate(candidate, recovery)
        driver_committed = True
        backend.commit_step()
        backend_active = False
        if float(committed.atmosphere.state.time_seconds) != start_time + DT_SECONDS:
            raise RuntimeError("composite transaction committed the wrong endpoint time")
        receipt = dict(backend.step_receipt())
        if receipt.get("phase") not in ("complete", "boundary"):
            raise RuntimeError("adapter commit did not publish a clean boundary")
        return CompositeStepResult(
            committed=committed,
            backend_receipt=MappingProxyType(receipt),
            clamp_d2h=clamp_d2h,
            recovery=recovery,
            dynamics_tendencies=next_dynamics_tendencies,
        )
    except Exception as error:
        rollback_errors: list[str] = []
        if physics_park is not None and physics_park.is_parked:
            # The seam's own rollback reads the arrays this park emptied.
            # Restoring first keeps the rollback path looking at the state it
            # was written against; a failure here is reported, never swallowed.
            try:
                physics_park.unpark()
            except Exception as unpark_error:  # pragma: no cover - catastrophic
                rollback_errors.append(f"physics tier unpark failed: {unpark_error}")
        if candidate is not None and not driver_committed:
            try:
                driver.abort_post_wsm6_candidate(candidate)
                if (
                    driver.atmosphere is not start_atmosphere
                    or driver.atmosphere.state is not start_state
                    or float(driver.atmosphere.state.time_seconds) != start_time
                ):
                    raise RuntimeError(
                        "MPAS driver did not restore the exact start-state identity"
                    )
            except Exception as rollback_error:  # pragma: no cover - catastrophic path
                rollback_errors.append(f"driver abort failed: {rollback_error}")
        rollback_not_armed = False
        if getattr(backend, "_phase", None) == "broken":
            # The adapter's own rollback raised while rebuilding its seam
            # (the reconstruction error is in this exception's chain).  There
            # is no transaction left to abort and no boundary to verify:
            # calling abort_step here only added "abort_step requires a begun
            # or finished transaction" over the step's real refusal, and a
            # phase-one failure was reported as a clean abort.
            rollback_errors.append(
                "the physics adapter could not rebuild its seam at the step "
                "boundary, so the seam is retired (phase 'broken')"
            )
        elif backend_active:
            try:
                phase = _phase_from_receipt(backend)
                armed = (
                    dict(backend.step_receipt()).get("transaction_rollback") or {}
                ).get("armed", True)
                if not armed:
                    # The run declined the boundary export, so there is no
                    # boundary to verify: the adapter retires its seam
                    # (abort restores the unpublished MPAS scalars, then
                    # leaves phase rollback_not_armed) and the step refuses.
                    if phase in ("begun", "finished_unpublished"):
                        backend.abort_step()
                    rollback_not_armed = True
                elif phase in ("begun", "finished_unpublished"):
                    backend.abort_step()
                    expected_phase = "rolled_back"
                elif phase == "automatic_rollback":
                    expected_phase = "automatic_rollback"
                else:
                    raise RuntimeError(
                        "adapter failure left an unrecognized transaction phase "
                        f"{phase!r}"
                    )
                if not rollback_not_armed:
                    _verify_backend_rollback_boundary(
                        backend,
                        expected_phase=expected_phase,
                        start_time=start_time,
                        transaction_identity=transaction_identity,
                    )
            except Exception as rollback_error:  # pragma: no cover - catastrophic path
                rollback_errors.append(
                    f"adapter rollback verification failed: {rollback_error}"
                )
        if driver_committed:
            raise CompositeTransactionError(
                "adapter commit failed after MPAS publication; this is an "
                "unrecoverable violation of the frozen no-fail commit contract"
            ) from error
        if rollback_errors:
            raise CompositeTransactionError(
                "composite step failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        if rollback_not_armed:
            raise CompositeTransactionError(
                f"composite step at {start_time} s was refused and aborted "
                "without publication; this run armed no physics rollback, so "
                "the physics seam is retired and the run stops at the last "
                "committed step (--stop-on-refusal arms it and keeps that "
                "step's frame)"
            ) from error
        raise CompositeTransactionError(
            f"composite step at {start_time} s was aborted without publication"
        ) from error


def _host_array(value: Any, cp: Any) -> np.ndarray:
    if isinstance(value, cp.ndarray):
        return np.ascontiguousarray(cp.asnumpy(value))
    return np.ascontiguousarray(np.asarray(value))


def _name_mapping_difference(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> str:
    """Say which keys differ and how, instead of that something differed."""

    parts: list[str] = []
    for key in sorted(set(expected) | set(observed)):
        left = expected.get(key, "<absent>")
        right = observed.get(key, "<absent>")
        if left != right:
            parts.append(f"{key}: expected {left!r}, got {right!r}")
    return "; ".join(parts) if parts else "no key differs (ordering only)"


def require_arwen_v2_surface_execution(
    payload: Mapping[str, Any], *, executed: bool, label: str
) -> dict[str, Any]:
    update = payload.get("surface_update", {})
    alarmed = update.get("enabled") is True
    expected_classification = (update["classification"] if alarmed else EXPECTED_SURFACE_CLASSIFICATION)
    expected_census = EXPECTED_NOAHMP_CENSUS
    if alarmed and executed and expected_census is not None:
        used = update["physics_classification"]
        expected_census = dict(land=used["sflx_land_columns"], water=used["open_water_columns"],
                              sea_ice=used["sea_ice_columns"], glacier=used["glacier_columns"])
        if expected_census["glacier"]:
            expected_census["glacier_path"] = ARWEN_GLACIER_CUDA_PROVENANCE
    routing = payload.get("noahmp_routing_census")
    if routing is not None and expected_census is not None:
        # Independent pre-LSM layout, validated by the backend against actual
        # execution. Static initial categories need not survive ice evolution.
        keys = ("land", "water", "sea_ice", "glacier")
        if (not isinstance(routing, Mapping) or
                any(type(routing.get(k)) is not int or routing[k] < 0 for k in keys) or
                sum(routing[k] for k in keys) != sum(expected_census[k] for k in keys) or
                set(routing) != set(keys) | ({"glacier_path"} if routing["glacier"] else set()) or
                (routing["glacier"] and routing["glacier_path"] != ARWEN_GLACIER_CUDA_PROVENANCE)):
            raise ValueError(f"{label} invalid NoahMP routing census/provenance")
        expected_census = routing
    classification = payload.get("surface_classification")
    if not isinstance(classification, Mapping):
        raise ValueError(f"{label} lacks surface_classification")
    if dict(classification) != dict(expected_classification):
        raise ValueError(
            f"{label} surface classification changed: "
            + _name_mapping_difference(
                dict(expected_classification), dict(classification)
            )
        )
    census = payload.get("last_noahmp_census")
    if expected_census is None:
        # A suite row's land surface is not Noah-MP: it publishes its own
        # census under its own key and never a Noah-MP one.  Either a
        # Noah-MP census on such a frame or a missing land census after the
        # physics ran is a frame that does not describe its own run.
        if census is not None:
            raise ValueError(f"{label} claims a NoahMP census on a non-NoahMP row")
        land = payload.get("last_land_surface_census")
        if executed and not isinstance(land, Mapping):
            raise ValueError(
                f"{label} carries no land-surface census, so nothing says how "
                "many columns the land-surface model was handed"
            )
        return {
            "surface_classification": dict(classification),
            "last_noahmp_census": None,
            "last_land_surface_census": None if land is None else dict(land),
        }
    if not executed:
        if census is not None:
            raise ValueError(f"{label} unexpectedly claims NoahMP execution")
        return {"surface_classification": dict(classification), "last_noahmp_census": None}
    if not isinstance(census, Mapping):
        raise ValueError(
            f"{label} carries no NoahMP census, so nothing says how many "
            "columns the land-surface model was handed"
        )
    if dict(census) != dict(expected_census):
        raise ValueError(
            f"{label} NoahMP census/provenance changed: "
            + _name_mapping_difference(
                dict(expected_census), dict(census)
            )
        )
    return {
        "surface_classification": dict(classification),
        "last_noahmp_census": dict(census),
    }


def retain_initial_diagnostic_snapshot() -> bool:
    """Whether the forecast stack keeps the start-time snapshot's device arrays.

    Default no (memory lane, 2026-10-15).  ``stack["initial_diagnostic"]``
    held the hour-0 ``diagnostic_snapshot()`` for the whole run: the
    zero-filled GWDO level fields (dtaux3d, dtauy3d, rubldiff, rvbldiff), the
    GWDO surface pair, and device copies of the surface, soil and
    precipitation exports -- about 40 MB at x1.40962 (1 KB per column), live
    on every device peak of route A.  Nothing reads the entry after the clock
    check that follows the snapshot, and every history frame takes a fresh
    snapshot.  The host-side metadata and receipt stay in the stack.
    ``WOOF_HEX_DIAG_KEEP_INITIAL_SNAPSHOT=1`` keeps the device groups as
    well, for A/B runs.
    """

    return os.environ.get("WOOF_HEX_DIAG_KEEP_INITIAL_SNAPSHOT", "0") == "1"


def backend_diagnostic_mapping(snapshot: Any) -> dict[str, Any]:
    """Normalize the released detached diagnostic dataclass without copies."""

    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    names = ("surface", "soil", "precipitation", "gwdo", "metadata", "receipt")
    missing = [name for name in names if not hasattr(snapshot, name)]
    if missing:
        raise TypeError(f"adapter diagnostic snapshot lacks groups {missing}")
    mapped = {name: getattr(snapshot, name) for name in names}
    if any(not isinstance(mapped[name], Mapping) for name in names):
        raise TypeError("adapter diagnostic snapshot groups must be mappings")
    return mapped


def _flatten_backend_diagnostics(snapshot: Mapping[str, Any], cp: Any, download=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(f"{prefix}/{key}" if prefix else str(key), item)
            return
        if isinstance(value, cp.ndarray) or isinstance(value, np.ndarray):
            name = prefix.rsplit("/", 1)[-1]
            if name in arrays:
                raise ValueError(f"duplicate backend diagnostic leaf {name!r}")
            arrays[name] = (download or _host_array)(value, cp)
            return
        metadata[prefix] = value

    walk("", snapshot)
    return arrays, metadata


def capture_snapshot(
    *,
    label: str,
    step: int,
    driver: Any,
    backend: Any,
    prep_geometry: Any,
    kernel_cache: Any,
    f000_surface_diagnostics: Mapping[str, Any],
    expect_refl10cm: bool = False,
    deferred: bool = False,
    solve_cells: int | None = None,
    scalar_names: Sequence[str] = SCALAR_NAMES,
    surface_accumulators: Sequence[str] = ("rainnc", "snownc", "graupelnc"),
) -> dict[str, Any]:
    """Download one committed diagnostic boundary outside step receipts.

    ``scalar_names`` and ``surface_accumulators`` are the RUN's row (the
    forecast driver hands the row it selected); the defaults are the frozen
    WSM6 lane's, so this proof harness reads exactly what it always read.

    ``expect_refl10cm`` says this frame is a history frame whose step ran
    microphysics with ``refl_10cm_due=True``: the committed one-frame field is
    consumed from the backend and captured as ``refl10cm``.  Step 0 has run no
    microphysics, so its frame carries the allocated-default zeros exactly as
    native MPAS-A writes at the initial time.  A later frame with nothing
    staged is refused by name -- a silent hole here is a history frame the
    obs referee would score without the model's reflectivity.
    """

    import cupy as cp
    from woof.hex.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841

    if label != SNAPSHOT_LABELS.get(step):
        raise ValueError("snapshot label/step mismatch")
    if float(driver.atmosphere.state.time_seconds) != step * DT_SECONDS:
        raise ValueError("snapshot model time does not equal its exact step")
    phase = _phase_from_receipt(backend)
    if phase not in ("complete", "boundary", "restored"):
        raise RuntimeError("snapshot is legal only at a committed backend boundary")
    prepared = prepare_mpas_to_phys_cuda_v841(
        driver.atmosphere,
        scalar_names=tuple(scalar_names),
        geometry=prep_geometry,
        kernel_cache=kernel_cache,
        post_rk_wsm6=False,
    )
    from woof.hex.snapshot_transfer import SnapshotTransfer, PendingSnapshot, array_records
    from contextlib import nullcontext
    with SnapshotTransfer(cp) if deferred else nullcontext(None) as transfer:
        download = _host_array if transfer is None else transfer.download
        state = driver.atmosphere.state
        saved = driver.atmosphere.saved
        arrays = {
            name: download(state.scalars[index], cp)
            for index, name in enumerate(scalar_names)
        }
        arrays.update(
            {
                "u_zonal": download(prepared.u_p, cp),
                "v_meridional": download(prepared.v_p, cp),
                "normal_u": download(saved.normal_velocity, cp),
                "rho": download(prepared.rho_dry, cp),
                "theta": download(prepared.th_p, cp),
                "pressure": download(prepared.pres_p, cp),
                "surface_pressure": download(prepared.psfc_p, cp),
                "w": download(saved.vertical_velocity, cp),
            }
        )
        diagnostic = (getattr(backend, "history_diagnostic_snapshot", backend.diagnostic_snapshot)
                      if deferred else backend.diagnostic_snapshot)
        backend_snapshot = backend_diagnostic_mapping(diagnostic())
        executed = step > 0
        require_arwen_v2_surface_execution(
            dict(backend.step_receipt()), executed=executed, label=f"{label} step receipt"
        )
        require_arwen_v2_surface_execution(
            backend_snapshot["metadata"], executed=executed, label=f"{label} metadata"
        )
        surface_execution = require_arwen_v2_surface_execution(
            backend_snapshot["receipt"], executed=executed, label=f"{label} receipt"
        )
        diagnostics, diagnostic_metadata = _flatten_backend_diagnostics(
            backend_snapshot, cp, download
        )
        arrays.update(diagnostics)
        if expect_refl10cm:
            staged_refl = backend.take_history_refl10cm()
            if staged_refl is None and step == 0 and hasattr(backend, "initial_refl10cm"):
                # Hour 0: the scheme's own reflectivity of the initial
                # hydrometeors.  Native MPAS-A writes the allocated default
                # here (its first microphysics call has not happened); hex
                # declares that divergence (initial_refl10cm says why).
                staged_refl, _ = backend.initial_refl10cm(
                    atmosphere=driver.atmosphere, scalar_names=tuple(scalar_names))
            if staged_refl is not None:
                arrays["refl10cm"] = download(staged_refl, cp)
            elif step == 0 or not getattr(backend, "computes_reflectivity", lambda: True)():
                # No microphysics in the selection, or a scheme whose
                # reflectivity the engine cannot separate from its step: the
                # allocated default, as native MPAS-A writes at the initial time.
                arrays["refl10cm"] = np.zeros((N_LEVELS, N_CELLS), dtype=np.float32)
            else:
                raise RuntimeError(
                    f"history frame {label} (step {step}) expected a committed "
                    "refl10cm and the backend holds none; the step loop failed "
                    "to pass refl_10cm_due to this step's microphysics, and "
                    "writing the frame without it would hand the obs referee a "
                    "model bundle with no reflectivity"
                )
        prepared_receipt = prepared.receipt()
        backend_receipt = dict(backend.step_receipt())
        def finish():
            nonlocal arrays
            if solve_cells is not None:
                # A limited-area device atmosphere is one column wider than the domain
                # it solves: native allocates nCells+1 and maps every absent
                # neighbour to that garbage element.  Every array downloaded above
                # carries it.  It is stripped HERE, before the physical gate, the
                # hash projection and the netCDF writer see any of them, so a
                # limited-area history frame is the same shape as the domain's own
                # cell count and a receipt from one is comparable, cell for cell,
                # with a receipt from the global run over the same ground.
                #
                # THE BREAKAGE THIS PREVENTS: without the strip, every published
                # frame carries one extra cell holding a duplicate column's physics,
                # `write_snapshot_netcdf` writes nCells+1 rows against an nCells
                # coordinate set, and the render window is sized from a mesh whose
                # last cell is not on it.
                keep = int(solve_cells)
                for name, value in list(arrays.items()):
                    array = np.asarray(value)
                    if array.ndim >= 1 and int(array.shape[-1]) == keep + 1:
                        arrays[name] = np.ascontiguousarray(array[..., :keep])
            f000_overlay = None
            if step == 0:
                if (
                    set(f000_surface_diagnostics) != {"arrays", "receipt"}
                    or not isinstance(f000_surface_diagnostics["arrays"], Mapping)
                    or not isinstance(f000_surface_diagnostics["receipt"], Mapping)
                ):
                    raise ValueError("F000 initialized surface diagnostic carrier is malformed")
                f000_overlay = {
                    "source": dict(f000_surface_diagnostics["receipt"]),
                    "overlay": overlay_f000_initialized_surface_diagnostics(
                        arrays, f000_surface_diagnostics["arrays"]
                    ),
                }
            required_diagnostics = {
                "tsk",
                # A zero-level soil row (land surface off) publishes no soil column;
                # any other row must carry both soil fields.
                *(() if N_SOIL_LEVELS == 0 else ("smois", "tslb")),
                "hfx",
                "qfx",
                "lh",
                "t2",
                "u10",
                "v10",
                "rainc",
                *surface_accumulators,
                "dusfcg",
                "dvsfcg",
                "dtaux3d",
                "dtauy3d",
                "rubldiff",
                "rvbldiff",
            }
            missing = sorted(required_diagnostics - set(arrays))
            if missing:
                raise RuntimeError(f"adapter diagnostic snapshot lacks required fields {missing}")
            if transfer is None:
                cp.cuda.get_current_stream().synchronize()
            receipt = {
                "schema": SNAPSHOT_SCHEMA,
                "label": label,
                "step": step,
                "time_seconds": step * DT_SECONDS,
                "arwen_v2_surface_execution": surface_execution,
                "audit_d2h_outside_step_receipt": True,
                "prep": prepared_receipt,
                "backend": backend_receipt,
                "backend_diagnostic_metadata": diagnostic_metadata,
                "f000_initialized_surface_diagnostics": f000_overlay,
                "arrays": array_records(arrays, array_sha256),
            }
            return {"arrays": arrays, "receipt": receipt}

        if transfer is None:
            return finish()
        transfer.seal()
        return PendingSnapshot(transfer, finish)


def physical_snapshot_gate(
    snapshot: Mapping[str, Any],
    *,
    allow_initial_negative_qv: bool,
    scalar_names: Sequence[str] = SCALAR_NAMES,
    surface_accumulators: Sequence[str] = ("rainnc", "snownc", "graupelnc"),
    surface_accumulator_declarations: Sequence[Any] | None = None,
) -> dict[str, Any]:
    arrays = snapshot["arrays"]
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise FloatingPointError(f"snapshot {name} is non-finite")
    if np.min(arrays["rho"]) <= 0.0 or np.min(arrays["theta"]) <= 0.0:
        raise FloatingPointError("snapshot rho/theta must be strictly positive")
    if np.min(arrays["pressure"]) <= 0.0 or np.min(arrays["surface_pressure"]) <= 0.0:
        raise FloatingPointError("snapshot pressure must be strictly positive")
    moisture_negative = {
        name: int(np.count_nonzero(np.asarray(arrays[name]) < 0.0))
        for name in scalar_names
    }
    expected_qv = NEGATIVE_QV_PIN["negative_count"] if allow_initial_negative_qv else 0
    if moisture_negative["qv"] != expected_qv:
        raise FloatingPointError(
            f"qv negative count {moisture_negative['qv']} != {expected_qv}"
        )
    if any(moisture_negative[name] for name in tuple(scalar_names)[1:]):
        raise FloatingPointError("a hydrometeor snapshot contains negative values")
    if N_SOIL_LEVELS == 0:
        # Land surface off: no soil column is published (capture_snapshot).
        if "smois" in arrays or "tslb" in arrays:
            raise ValueError("a zero-level soil row published a soil column")
        soil_range = None
    else:
        smois = np.asarray(arrays["smois"])
        if float(smois.min()) < 0.0 or float(smois.max()) > 1.0:
            raise FloatingPointError("soil moisture lies outside [0,1]")
        soil_range = [float(smois.min()), float(smois.max())]
    signed_fallout = set()
    if surface_accumulator_declarations is not None:
        if tuple(item.name for item in surface_accumulator_declarations) != tuple(surface_accumulators):
            raise ValueError("snapshot accumulator declarations do not match the published buckets")
        signed_fallout = {item.name for item in surface_accumulator_declarations if item.signed_fallout}
    for name in ("rainc", *(name for name in surface_accumulators if name not in signed_fallout)):
        if np.min(arrays[name]) < 0.0:
            raise FloatingPointError(f"negative accumulated precipitation in {name}")
    # Native q2 has 44 negative values.  It is published bitwise and stays
    # deliberately excluded from the nonnegativity gate; finiteness is still
    # required above like every other field.
    return {
        "finite_all_fields": True,
        "positive_rho_theta_pressure": True,
        "moisture_negative_counts": moisture_negative,
        "soil_moisture_range": soil_range,
        "precipitation_nonnegative": all(np.min(arrays[name]) >= 0.0 for name in ("rainc", *surface_accumulators)),
        "signed_fallout_accumulators": sorted(signed_fallout),
        "precipitation_matches_scheme": True,
        "q2_policy": "published bitwise; negatives allowed (native parity); finiteness gated",
    }


def _native_logical_array(variable: Any, layout: str) -> np.ndarray:
    variable.set_auto_maskandscale(False)
    raw = np.ascontiguousarray(np.asarray(variable[...]))
    if raw.dtype != np.dtype(np.float32):
        raise TypeError(f"native {variable.name} is not FP32")
    if not np.all(np.isfinite(raw)):
        raise FloatingPointError(f"native {variable.name} is non-finite")
    if layout == "cell":
        if raw.shape != (1, N_CELLS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0])
    if layout == "level_cell":
        if raw.shape != (1, N_CELLS, N_LEVELS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    if layout == "level_edge":
        if raw.shape != (1, N_EDGES, N_LEVELS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    if layout == "interface_cell":
        if raw.shape != (1, N_CELLS, N_INTERFACES):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    if layout == "soil_cell":
        if raw.shape != (1, N_CELLS, N_SOIL_LEVELS):
            raise ValueError(f"native {variable.name} shape changed: {raw.shape}")
        return np.ascontiguousarray(raw[0].T)
    raise KeyError(layout)


def comparison_metrics(cuda: Any, native: Any) -> dict[str, Any]:
    left = np.asarray(cuda)
    right = np.asarray(native)
    if left.shape != right.shape:
        raise ValueError(f"comparison shape mismatch {left.shape} != {right.shape}")
    l64 = left.astype(np.float64, copy=False)
    r64 = right.astype(np.float64, copy=False)
    difference = l64 - r64
    rmse = math.sqrt(float(np.mean(difference * difference, dtype=np.float64)))
    mae = float(np.mean(np.abs(difference), dtype=np.float64))
    authority_rms = math.sqrt(float(np.mean(r64 * r64, dtype=np.float64)))
    return {
        "count": int(left.size),
        "rmse": rmse,
        "mae": mae,
        "bias": float(np.mean(difference, dtype=np.float64)),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "native_rms": authority_rms,
        "normalized_rmse": rmse / max(authority_rms, np.finfo(np.float64).tiny),
        "cuda_sha256": array_sha256(left),
        "native_sha256": array_sha256(right),
    }


def compare_snapshot_to_native(snapshot: Mapping[str, Any], native_path: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    results: dict[str, Any] = {}
    with Dataset(native_path, "r") as dataset:
        for cuda_name, (native_name, layout) in NATIVE_FIELD_MAP.items():
            if cuda_name not in snapshot["arrays"]:
                raise RuntimeError(f"CUDA snapshot lacks comparable field {cuda_name!r}")
            if native_name not in dataset.variables:
                raise RuntimeError(f"native authority lacks field {native_name!r}")
            native = _native_logical_array(dataset.variables[native_name], layout)
            metrics = comparison_metrics(snapshot["arrays"][cuda_name], native)
            limit = float(NATIVE_RMSE_LIMITS[cuda_name])
            metrics["rmse_limit"] = limit
            metrics["broad_consistency_gate"] = metrics["rmse"] <= limit
            if not metrics["broad_consistency_gate"]:
                raise FloatingPointError(
                    f"{cuda_name} native consistency RMSE {metrics['rmse']} exceeds {limit}"
                )
            results[cuda_name] = metrics
    return {
        "comparison_kind": "broad quantitative consistency, not parity",
        "native_path": str(native_path),
        "fields": results,
        "all_broad_consistency_gates_passed": True,
        "gf_native_parity_claim": False,
        # The claim names what makes it false, measured, so a reader does
        # not have to guess whether "false" means "unmeasured" or "known".
        # The seam non-parity closure closed the SEAM non-parity (the four auxiliary forcing
        # lanes, shallow-on, per-cell dx).  What remains is not a seam gap:
        # the port's GF body is WRF v4.6.1's Freitas-2018 generation while
        # v8.4.1's module_cu_gf.mpas.F is the 2013 ensemble fork -- native
        # has no dicycle and no tau_ecmwf closures at all, and runs c0=.002
        # with a NON-precipitating shallow scheme against the port's c0=.004
        # and a precipitating one.  This is a DECLARED DIVERGENCE, not a
        # blocker: physics parity against native was retired as a goal on
        # 2026-08-20, the verification of record is obs-skill (MRMS/ASOS),
        # and docs/declared-divergences.md carries mechanism, magnitude and
        # referee.  Closing it would mean porting native's cup_gf/cup_gf_sh
        # bodies, not tuning this seam -- and whether it should close is the
        # referee's call.
        "gf_declared_divergence": (
            "GF scheme generation differs: port body is WRF v4.6.1 "
            "(dicycle + tau_ecmwf closures, c0=.004, precipitating shallow); "
            "native v8.4.1 module_cu_gf.mpas.F is the 2013 ensemble GF "
            "(Fritsch-Chappell AA0/1200s members, no dicycle, c0=.002, "
            "non-precipitating shallow). Declared divergence judged by "
            "obs-skill, not a parity blocker; see docs/declared-divergences.md"
        ),
    }


def fingerprint_nested_arrays(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Hash a detached restart/diagnostic payload without JSON-coercing arrays."""

    fields: dict[str, Any] = {}
    scalars: dict[str, Any] = {}
    leaves: list[tuple[str, np.ndarray]] = []

    def walk(path: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                walk(f"{path}/{key}" if path else str(key), item[key])
        elif isinstance(item, np.ndarray):
            fields[path] = {
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "sha256": None,
            }
            leaves.append((path, item))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            scalars[path] = item
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(f"{path}/{index}", child)
        else:
            scalars[path] = repr(item)

    walk(prefix, value)
    # THE BREAKAGE THIS PREVENTS, MEASURED (unified image, 4x RTX PRO 6000
    # box, 2026-10-16, cProfile of a 60 km 1 h forecast): the boundary
    # fingerprints hashed every restart array one after another on one
    # thread, about 15 s of one core with the card idle (the f000 boundary
    # and the close).  Each array's digest is independent, so they are taken
    # on WOOF_HEX_FINGERPRINT_THREADS threads and placed by path; the record
    # and its digest are the same bytes.
    workers = min(_fingerprint_threads(), len(leaves), os.cpu_count() or 1)
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-fingerprint") as pool:
            digests = list(pool.map(lambda leaf: array_sha256(leaf[1]), leaves))
    else:
        digests = [array_sha256(item) for _, item in leaves]
    for (path, _), digest in zip(leaves, digests):
        fields[path]["sha256"] = digest
    core = {"arrays": fields, "scalars": scalars}
    return {**core, "sha256": canonical_json_sha256(core)}


def _fingerprint_leaf_projection(value: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten exact fingerprint leaves while omitting redundant group digests."""

    leaves: dict[str, Any] = {}

    def walk(path: str, item: Any) -> None:
        if isinstance(item, Mapping):
            has_nested_mapping = any(
                isinstance(child, Mapping) for child in item.values()
            )
            for key in sorted(item):
                if key == "sha256" and has_nested_mapping:
                    continue
                walk(f"{path}/{key}" if path else str(key), item[key])
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                walk(f"{path}/{index}" if path else str(index), child)
        else:
            leaves[path] = item

    walk(prefix, value)
    return leaves


def require_fingerprint_identity(
    label: str,
    uninterrupted: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact identity and name every differing numerical leaf."""

    direct = _fingerprint_leaf_projection(uninterrupted)
    resumed = _fingerprint_leaf_projection(restored)
    missing = object()
    mismatches = sorted(
        path
        for path in set(direct) | set(resumed)
        if direct.get(path, missing) != resumed.get(path, missing)
    )
    if mismatches:
        raise RuntimeError(f"{label} differs at fingerprint paths {mismatches}")
    return {
        "bitwise_identical": True,
        "leaf_count": len(direct),
        "sha256": uninterrupted["sha256"],
    }


def fingerprint_execution_boundary(stack: Mapping[str, Any]) -> dict[str, Any]:
    """Hash all mutable MPAS and Arwen carriers at one committed boundary."""

    from woof.hex.cuda_dualrun import fingerprint_atmosphere

    return {
        "atmosphere": fingerprint_atmosphere(stack["driver"].atmosphere),
        "backend": fingerprint_nested_arrays(stack["backend"].restart_state()),
    }


def download_driver_checkpoint(
    driver: Any, backend: Any, *, dynamics_tendencies: Any
) -> HostDriverCheckpoint:
    if dynamics_tendencies is None:
        raise RuntimeError(
            "F030 checkpoint refused: the driver's GF advective-forcing "
            "carrier (rthdynten/rqvdynten) is absent.  Every full-physics "
            "step forms it, and a checkpoint written without it resumes by "
            "feeding GF zero forcing lanes at the first resumed step -- the "
            "deterministic step-16 restart divergence."
        )

    from woof.hex.cuda_dualrun import fingerprint_atmosphere
    from woof.hex.driver import DrySavedDiagnostics

    cp = driver.cp
    if _phase_from_receipt(backend) not in ("complete", "boundary"):
        raise RuntimeError("F030 checkpoint requires a committed backend boundary")
    cp.cuda.get_current_stream().synchronize()
    atmosphere_fingerprint = fingerprint_atmosphere(driver.atmosphere)
    state = driver.atmosphere.state.to_host()
    saved = driver.atmosphere.saved
    host_saved = DrySavedDiagnostics(
        theta_m=cp.asnumpy(saved.theta_m),
        exner=cp.asnumpy(saved.exner),
        density_perturbation=cp.asnumpy(saved.density_perturbation),
        rho_theta_perturbation=cp.asnumpy(saved.rho_theta_perturbation),
        pressure_perturbation=cp.asnumpy(saved.pressure_perturbation),
        normal_velocity=cp.asnumpy(saved.normal_velocity),
        vertical_velocity=cp.asnumpy(saved.vertical_velocity),
    )
    cp.cuda.get_current_stream().synchronize()
    if float(state.time_seconds) != CHECKPOINT_STEP * DT_SECONDS:
        raise RuntimeError("driver checkpoint is not at exact F030")
    host_atmosphere_fingerprint = fingerprint_atmosphere(
        SimpleNamespace(state=state, saved=host_saved)
    )
    require_fingerprint_identity(
        "F030 device-to-host MPAS checkpoint",
        atmosphere_fingerprint,
        host_atmosphere_fingerprint,
    )
    backend_state = backend.restart_state()
    backend_fingerprint = fingerprint_nested_arrays(backend_state)
    forcing_time = float(dynamics_tendencies.time_seconds)
    if forcing_time != float(state.time_seconds):
        raise RuntimeError(
            "GF advective forcing at the checkpoint must be the pair the "
            "checkpoint step's dynamics formed (stamped with its endpoint "
            f"time {float(state.time_seconds)} s); got t={forcing_time} s"
        )
    gf_host = {
        "rthdynten": np.ascontiguousarray(
            cp.asnumpy(dynamics_tendencies.rthdynten)
        ),
        "rqvdynten": np.ascontiguousarray(
            cp.asnumpy(dynamics_tendencies.rqvdynten)
        ),
        "time_seconds": forcing_time,
    }
    return HostDriverCheckpoint(
        state=state,
        saved_diagnostics=host_saved,
        backend_state=backend_state,
        atmosphere_fingerprint=atmosphere_fingerprint,
        backend_fingerprint=backend_fingerprint,
        model_time_seconds=float(state.time_seconds),
        gf_dynamics_tendencies=gf_host,
        gf_forcing_fingerprint=fingerprint_nested_arrays(gf_host),
    )


def _snapshot_hash_projection(snapshot: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: array_sha256(value)
        for name, value in sorted(snapshot["arrays"].items())
        if name != "q2"
    }


def require_bitwise_restart_identity(
    uninterrupted: Mapping[str, Any], restored: Mapping[str, Any]
) -> dict[str, Any]:
    direct = _snapshot_hash_projection(uninterrupted)
    resumed = _snapshot_hash_projection(restored)
    if direct != resumed:
        mismatches = sorted(
            name for name in set(direct) | set(resumed) if direct.get(name) != resumed.get(name)
        )
        raise RuntimeError(f"F030 restart continuation differs at F001 fields {mismatches}")
    return {
        "bitwise_identical": True,
        "field_count": len(direct),
        "fields": direct,
        "q2_excluded_from_publication_projection": True,
    }


def surface_evolution_gate(f000: Mapping[str, Any], f001: Mapping[str, Any]) -> dict[str, Any]:
    names = ("tsk", "smois", "tslb", "hfx", "qfx", "lh")
    evolved = {
        name: array_sha256(f000["arrays"][name]) != array_sha256(f001["arrays"][name])
        for name in names
    }
    if not any(evolved.values()):
        raise RuntimeError("surface and soil state did not evolve through F001")
    precipitation = sum(
        int(np.count_nonzero(f001["arrays"][name]))
        for name in ("rainc", "rainnc", "snownc", "graupelnc")
    )
    if precipitation == 0:
        raise RuntimeError("one-hour full-physics forecast produced no precipitation signal")
    gwd = {
        group: sum(int(np.count_nonzero(f001["arrays"][name])) for name in names_)
        for group, names_ in {
            "surface_stress": ("dusfcg", "dvsfcg"),
            "column_tendency": ("dtaux3d", "dtauy3d"),
            "pbl_increment": ("rubldiff", "rvbldiff"),
        }.items()
    }
    if any(count == 0 for count in gwd.values()):
        raise RuntimeError(f"external YSU-GWDO activity is absent: {gwd}")
    return {
        "surface_soil_evolved": evolved,
        "precipitation_nonzero_elements": precipitation,
        "gwd_nonzero_elements": gwd,
    }


def _write_exclusive_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def write_snapshot_netcdf(path: Path, snapshot: Mapping[str, Any], static: Mapping[str, Any],
                          *, _hash_file: bool = True) -> dict[str, Any]:
    """Write one truthful native-grid CUDA history capsule for Rust rendering."""

    from netCDF4 import Dataset
    from woof.hex.history_chunks import write_variable

    if path.exists():
        raise FileExistsError(path)
    arrays = snapshot["arrays"]
    valid_time = snapshot.get("valid_time")
    if valid_time is not None:
        # Public forecasts carry their schedule clock through both writer
        # paths. Keep capsules without a schedule clock backward compatible.
        valid_time = datetime.strptime(valid_time, "%Y-%m-%d_%H:%M:%S").strftime("%Y-%m-%d_%H:%M:%S")
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("Time", 1)
        if valid_time is not None:
            dataset.createDimension("StrLen", 64)
            clock = dataset.createVariable("xtime", "S1", ("Time", "StrLen"))
            clock.setncattr("units", "YYYY-MM-DD_hh:mm:ss")
            clock[0, :] = np.frombuffer(valid_time.ljust(64).encode("ascii"), dtype="S1")
        dataset.createDimension("nCells", N_CELLS)
        dataset.createDimension("nVertLevels", N_LEVELS)
        dataset.createDimension("nEdges", N_EDGES)
        dataset.createDimension("nVertLevelsP1", N_INTERFACES)
        if N_SOIL_LEVELS:
            # netCDF reads a zero-length dimension as UNLIMITED; a land-
            # surface-off frame has no soil column and declares no soil axis.
            dataset.createDimension("nSoilLevels", N_SOIL_LEVELS)
        dataset.setncattr("schema", SNAPSHOT_SCHEMA)
        dataset.setncattr("source", "MPAS-A v8.4.2 full-physics CUDA port")
        dataset.setncattr("arwen_commit", ARWEN_COMMIT)
        dataset.setncattr("gf_native_parity_claim", "false")
        dataset.setncattr(
            "gf_declared_divergence",
            "scheme generation: port GF is WRF v4.6.1, native v8.4.1 is the "
            "2013 ensemble fork; declared divergence judged by obs-skill "
            "(see gf_declared_divergence in the run receipt and "
            "docs/declared-divergences.md)",
        )
        dataset.setncattr("gf_seam_parity_closed", "task-231 forcing-lanes+shallow+per-cell-dx")
        dataset.setncattr("q2_products_allowed", "true")
        dataset.setncattr(
            "q2_negative_policy",
            "preserved bitwise; native q2 itself carries negative values, so "
            "consumers forming dewpoint clamp at their own boundary",
        )
        dataset.setncattr(
            "refl10cm_provenance",
            "computed inside the due step's WSM6 call from post-call "
            "temperature and unchanged prepared pressure (woof/core/refl.py "
            "transcribing WRF mp_wsm6.F90:2275-2444 with module_mp_radar "
            "melting); the model's field, not a renderer diagnostic",
        )
        for name, dtype in (("indexToCellID", "i4"), ("latCell", "f4"), ("lonCell", "f4"), ("ter", "f4")):
            variable = dataset.createVariable(name, dtype, ("nCells",))
            variable[:] = static[name]
        for name, dtype in (("indexToEdgeID", "i4"), ("latEdge", "f4"), ("lonEdge", "f4")):
            variable = dataset.createVariable(name, dtype, ("nEdges",))
            variable[:] = static[name]
        from woof.hex.species_row import species_row_for_names
        if CHEMISTRY_CONFIG is not None:
            from woof.hex.chem_transport import configure, history_metadata
            dataset.setncattr("CHEMISTRY", history_metadata(CHEMISTRY_CONFIG, CHEMISTRY_CONFIG_SHA256))
            configure(CHEMISTRY_CONFIG)
        species = species_row_for_names(SCALAR_NAMES)
        if species.engine_scheme != "wsm6":
            dataset.setncattr("refl10cm_provenance",
                "native microphysics diagnostic from the due physics call; scheme=" + species.scheme_name)
        if (snapshot.get("receipt") or {}).get("step") == 0:
            dataset.setncattr(
                "refl10cm_provenance",
                "hour 0: the active scheme's own 10 cm reflectivity of the initial hydrometeors "
                "(backend initial_refl10cm through woof.da.obsop.simulated_reflectivity, temperature "
                "from initial theta and pressure); native MPAS-A writes zero at the initial time, a "
                "declared divergence. Zero only when the selection has no microphysics or the engine "
                "cannot separate the scheme's reflectivity from its step",
            )
        units = {
            **{name: species.declaration(name).units for name in SCALAR_NAMES},
            "tke": "m^2 s^{-2}",
            "u_zonal": "m s^{-1}",
            "v_meridional": "m s^{-1}",
            "normal_u": "m s^{-1}",
            "rho": "kg m^{-3}",
            "theta": "K",
            "pressure": "Pa",
            "surface_pressure": "Pa",
            "w": "m s^{-1}",
            "tsk": "K",
            "sst": "K",
            "sstsk": "K",
            "ocean_depth": "m", "ocean_volume": "m", "ocean_area": "1",
            "ocean_ice_temp": "K", "ocean_snow_water": "m", "ocean_melt": "kg m^{-2}",
            "ocean_basal": "W m^{-2}", "ocean_net_flux": "W m^{-2}",
            "ocean_energy": "J m^{-2}", "ocean_entrainment": "J m^{-2}",
            "ocean_residual": "J m^{-2}", "ocean_initial_sst": "K", "ocean_initial_ice": "1",
            "tmn": "K",
            "t2": "K",
            "q2": "kg kg^{-1}",
            "refl10cm": "dBZ",
            "swdown": "W m^{-2}",
            "hfx": "W m^{-2}",
            "qfx": "kg m^{-2} s^{-1}",
            "lh": "W m^{-2}",
            "olr": "W m^{-2}",
            "swdown": "W m^{-2}",
            "glw": "W m^{-2}",
            "u10": "m s^{-1}",
            "v10": "m s^{-1}",
            "smois": "m3 m^{-3}",
            "tslb": "K",
            "rainc": "mm",
            "rainnc": "mm",
            "snownc": "mm",
            "graupelnc": "mm",
            "dusfcg": "Pa",
            "dvsfcg": "Pa",
            "dtaux3d": "m s^{-2}",
            "dtauy3d": "m s^{-2}",
            "rubldiff": "m s^{-2}",
            "rvbldiff": "m s^{-2}",
            "xice": "1",
            "snow": "kg m^{-2}",
            "snowh": "m",
        }
        for name, value in sorted(arrays.items()):
            array = np.asarray(value, dtype=np.float32)
            if array.shape == (N_LEVELS, N_CELLS):
                dims = ("Time", "nCells", "nVertLevels")
                payload = array.T[None]
            elif array.shape == (N_LEVELS, N_EDGES):
                dims = ("Time", "nEdges", "nVertLevels")
                payload = array.T[None]
            elif array.shape == (N_INTERFACES, N_CELLS):
                dims = ("Time", "nCells", "nVertLevelsP1")
                payload = array.T[None]
            elif N_SOIL_LEVELS and array.shape == (N_SOIL_LEVELS, N_CELLS):
                dims = ("Time", "nCells", "nSoilLevels")
                payload = array.T[None]
            elif array.shape == (N_CELLS,):
                dims = ("Time", "nCells")
                payload = array[None]
            else:
                continue
            variable = dataset.createVariable(name, "f4", dims, zlib=True, complevel=1)
            from woof.hex.chem_transport import history_units
            variable.setncattr("units", units.get(name, history_units(name) or ""))
            write_variable(variable, payload)
        if snapshot.get('wind_fields'):
            dataset.setncattr('WIND_FARMS', snapshot['wind_metadata'])
            for name, (array, dimensions, unit) in snapshot['wind_fields'].items():
                for dimension, size in zip(dimensions, array.shape):
                    if dimension not in dataset.dimensions:
                        dataset.createDimension(dimension, size)
                    elif len(dataset.dimensions[dimension]) != size:
                        raise ValueError(f'wind field {name}: dimension {dimension} differs from its history grid')
                variable = dataset.createVariable(name, 'f4', ('Time', *dimensions), zlib=True, complevel=1)
                variable.setncattr('units', unit)
                write_variable(variable, array[None])
    record = {"path": str(path), "bytes": path.stat().st_size}
    if _hash_file:
        record["sha256"] = sha256_file(path)
    return record


# atomic_write can defer just this digest until after NetCDF has closed,
# then compute it concurrently with the durable file flush. Direct calls
# retain their complete digest-bearing result.
write_snapshot_netcdf.__history_digest__ = sha256_file


def gpu_memory_admission(cp: Any, *, minimum: int = MIN_FREE_DEVICE_BYTES) -> dict[str, Any]:
    """Refuse unless this run's predicted peak still fits on the card.

    ``minimum`` is the door's own admission sum -- THIS run's predicted peak
    plus the card's margin -- so the bytes this process already holds in its
    CuPy pool are part of the peak being checked, not competition for it.
    They are added back to the card's free count.  Measured 2026-09-02 on
    the RTX 5070 Ti: two door-admitted runs (x1.40962 at 12,068 MiB free,
    r0.9.120.40520 at 9,700+ MiB free) reached this floor with 5.4 to 9.1 GiB
    of their own state already resident and were refused on the remainder
    (needed 8,011,133,287 B, saw 3,358,261,248 B; needed 10,171,315,234 B,
    saw 8,773,107,712 B), which double-counted the footprint and made every
    run over half the card unreachable.  A fresh process holds nothing, so
    the number is unchanged wherever the floor ran before any allocation.
    """

    free, total = map(int, cp.cuda.runtime.memGetInfo())
    try:
        held = int(cp.get_default_memory_pool().total_bytes())
    except Exception:  # a cupy without a default pool: nothing to add back
        held = 0
    available = free + held
    if available < minimum:
        raise MemoryError(
            f"full-physics x4 requires at least {minimum} free device bytes; got "
            f"{free} free on the card plus {held} already held by this process's "
            f"pool, {available} available"
        )
    return {
        "free_bytes": free,
        "total_bytes": total,
        "held_by_this_process_bytes": held,
        "available_bytes": available,
        "minimum": minimum,
        "admitted": True,
    }


def _prepare_host_execution(paths: Mapping[str, Path], authority_receipt: Mapping[str, Any]) -> dict[str, Any]:
    from woof.hex.config_v841 import V841MpasColumnPhysicsGwdoConfig
    from woof.hex.cuda_arwen_physics_v841 import SealedArwenConstructorV841
    from woof.hex.cuda_dualrun import PreparedCudaInputs
    from woof.hex.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from woof.hex.dynamics_v841 import load_v841_reference_wind_profiles
    from woof.hex.mesh import load_precision_preserving_mesh_pair

    # The x4 proof measures broad native full-physics consistency and hex
    # self-reproducibility, so it runs the native reference schedule
    # at, passed explicitly: since 2026-10-10 the config's defaults are
    # MPAS-A's Registry defaults (order 2, two sub-steps, radiation 1800 s,
    # limiter on), not the x4 run's (order 3, six sub-steps, 600 s, off).
    from woof.hex.mpas_defaults import X4_RECORDED_SCHEDULE

    config = V841MpasColumnPhysicsGwdoConfig(**X4_RECORDED_SCHEDULE)
    config.validate()
    mesh, output_mesh, mesh_evidence = load_precision_preserving_mesh_pair(
        paths["grid"], paths["static"]
    )
    del output_mesh
    reconstruction_overlay = overlay_exact_init_reconstruction_coefficients(
        mesh, paths["init"]
    )
    edge_normal_overlay = overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    defc = attach_inactive_zero_deformation(mesh)
    native = load_mpas_vertical_grid(
        paths["init"], mesh, config_coef_3rd_order=config.config_coef_3rd_order
    )
    state, reference, saved = load_mpas_initial_state(
        paths["init"],
        mesh,
        native.vertical_grid,
        scalar_names=SOURCE_SCALAR_NAMES,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    scalar_receipt = augment_exact_wsm6_scalars(state)
    state.validate(n_cells=N_CELLS, n_edges=N_EDGES, n_vert_levels=N_LEVELS)
    saved.validate((N_LEVELS, N_CELLS), np.dtype(np.float32), N_EDGES)
    profiles = load_v841_reference_wind_profiles(paths["init"], n_vert_levels=N_LEVELS)
    prepared = PreparedCudaInputs.validated(
        config=config,
        profile=PROFILE,
        target=CLAIM,
        preparation_method=(
            "precision-preserving grid/static overlay plus exact initialized "
            "reconstruction coefficients and edge-normal vectors; qv/qc/qr "
            "plus exact +0 qi/qs/qg"
        ),
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=dict(authority_receipt["files"]),
        reference_wind_profiles=profiles,
    )
    f000_surface_diagnostics = load_f000_initialized_surface_diagnostics(
        paths["init"]
    )
    constructor_values, constructor_receipt, gwdo_host = build_arwen_constructor_values(
        init_path=paths["init"],
        mesh=mesh,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
    )
    sealed_constructor_audit = SealedArwenConstructorV841.from_mapping(
        constructor_values
    )
    constructor_receipt["sealed_host_contract_audit"] = {
        "authority": "SealedArwenConstructorV841.from_mapping",
        "accepted": True,
        "all_required_keys_dtypes_shapes_validated": True,
        "array_fields": sorted(constructor_receipt["arrays"]),
        **dict(sealed_constructor_audit.receipt()),
    }
    return {
        "config": config,
        "prepared": prepared,
        "constructor_values": constructor_values,
        "constructor_receipt": constructor_receipt,
        "gwdo_host": gwdo_host,
        "mesh_evidence": mesh_evidence,
        "f000_surface_diagnostics": f000_surface_diagnostics,
        "reconstruction_coefficients": reconstruction_overlay,
        "edge_normal_vectors": edge_normal_overlay,
        "defc": defc,
        "scalar_receipt": scalar_receipt,
    }


def _construct_device_stack(
    *,
    host: Mapping[str, Any],
    cache: Any,
    arwen_checkout: Path,
    state: Any | None = None,
    saved_diagnostics: Any | None = None,
    backend_restart: Mapping[str, Any] | None = None,
    gf_dynamics_tendencies: Mapping[str, Any] | None = None,
    backend_builder: Callable[..., Any] | None = None,
    backend_initializer: Callable[[Any], None] | None = None,
    physics_rollback_snapshot: bool = True,
    stochastic_restart: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from woof.hex.cuda_arwen_physics_v841 import (
        PersistentTwoPhaseCudaPhysicsBackendV841,
        SealedArwenConstructorV841,
    )
    from woof.hex.cuda_driver import CudaDryDycoreDriver
    from woof.hex.cuda_gwdo_v841 import CudaYsuGwdoStaticV841
    from woof.hex.cuda_physics_prep_v841 import CudaMpasToPhysGeometryV841
    from woof.hex.cuda_physics_v841 import CudaPhysicsGeometryV841

    prepared = host["prepared"]
    selected_state = prepared.state if state is None else state
    selected_saved = prepared.saved_diagnostics if saved_diagnostics is None else saved_diagnostics
    # ONE ROUTE.  A limited-area run and a global run differ in exactly two
    # things: the device atmosphere is one column wider (native's own garbage
    # allocation, which the regional acoustic kernels index through), and the
    # driver carries a boundary runtime.  Everything downstream -- the three
    # physics geometry constructors, the sealed ArWen constructor, the
    # backend, and the composite step loop -- is the SAME code on the same
    # objects; the regional lane just hands them a mesh view with one more
    # cell in it.  Building a parallel physics construction for the regional
    # lane is what this branch exists to avoid: two constructions drift, and
    # the drift shows up as a physics difference nobody meant to make.
    regional = host.get("regional")
    from woof.hex import arwen_les
    les = None
    if arwen_les.active(host["config"]):
        from woof.hex.cuda_les_v842 import ResidentLesV842
        les = ResidentLesV842(
            prepared.mesh, prepared.vertical, prepared.reference_wind_profiles,
            host["config"], host["scalar_names"], host["les_theta_initial"])
    if regional is None:
        driver = CudaDryDycoreDriver.from_host(
            prepared.mesh,
            selected_state,
            prepared.vertical,
            prepared.reference,
            host["config"],
            saved_diagnostics=selected_saved,
            terrain_metrics=prepared.terrain_metrics,
            advection_coefficients=prepared.advection_coefficients,
            kernel_cache=cache,
            reference_wind_profiles=prepared.reference_wind_profiles,
            les_dissipation=les,
        )
        physics_mesh = prepared.mesh
    else:
        from woof.hex.cuda_regional_forecast_v841 import (
            open_regional_forecast_v841,
        )

        driver = open_regional_forecast_v841(
            prepared.mesh,
            selected_state,
            prepared.vertical,
            prepared.reference,
            selected_saved,
            prepared.terrain_metrics,
            host["config"],
            reference_wind_profiles=prepared.reference_wind_profiles,
            les_dissipation=les,
            lbc_paths=regional["lbc_paths"],
            start_time=regional["start_time"],
            kernel_cache=cache,
            # The species the boundary stream drives, in the model's own
            # order.  rw_mpas_lbc writes lbc_qv/lbc_qc/lbc_qr and the model
            # runs six WSM6 species; the three the stream carries are
            # nudged at the boundary and the three it does not are left to
            # the microphysics, which is what native does with a stream that
            # declares fewer scalars than the scheme integrates.
            scalar_names=regional["driven_scalars"],
        )
        physics_mesh = driver.regional_v841.padded
    prep_geometry = CudaMpasToPhysGeometryV841.from_host(physics_mesh)
    physics_geometry = CudaPhysicsGeometryV841.from_host(physics_mesh)
    # A suite row runs its drag inside the engine's PBL slot and hands no
    # hex-side YSU-GWDO statics (host["gwdo_host"] is None); it also seals
    # its own constructor, whose soil column and drag statistics are not the
    # frozen mapping's.  The frozen row takes both defaults, unchanged.
    gwdo_static = (
        None
        if host["gwdo_host"] is None
        else CudaYsuGwdoStaticV841.from_host(host["gwdo_host"])
    )
    sealer = getattr(backend_builder, "seal_constructor", None)
    constructor = (
        SealedArwenConstructorV841.from_mapping(host["constructor_values"])
        if sealer is None
        else sealer(host["constructor_values"])
    )
    if backend_builder is None:
        backend = PersistentTwoPhaseCudaPhysicsBackendV841(
            constructor=constructor,
            prep_geometry=prep_geometry,
            kernel_cache=cache,
            gwdo_static=gwdo_static,
            gwdo_kernel_cache=cache,
            arwen_checkout=arwen_checkout,
            # The proof keeps its per-step rollback (the default); a caller
            # that never reads a refused step's boundary declines the export
            # (woof.hex.cuda_arwen_physics_v841, rollback_snapshot).
            rollback_snapshot=bool(physics_rollback_snapshot),
        )
    else:
        # A backend ROW builds its own column backend (the forecast driver
        # hands ``PhysicsBackendRow.build_column_backend`` through here);
        # the frozen row above is what a run with no builder gets.
        backend = backend_builder(
            constructor=constructor,
            prep_geometry=prep_geometry,
            kernel_cache=cache,
            gwdo_static=gwdo_static,
            gwdo_kernel_cache=None if gwdo_static is None else cache,
            checkout=arwen_checkout,
            physics_mesh=physics_mesh,
            rollback_snapshot=bool(physics_rollback_snapshot),
        )
    from woof.hex.stochastic import StochasticConfig, StochasticPhysics
    stochastic_config = StochasticConfig.parse(host.get("stochastic"))
    if stochastic_config.enabled:
        driver.stochastic = StochasticPhysics(prepared.mesh, prepared.vertical, stochastic_config)
        if stochastic_restart is not None:
            driver.stochastic.process.restore(stochastic_restart)
        elif float(driver.atmosphere.state.time_seconds) or stochastic_config.restart_required:
            raise ValueError("stochastic restart is absent from this checkpoint")
    elif stochastic_restart is not None:
        raise ValueError("checkpoint stochastic physics must remain enabled on resume")
    api = require_staged_backend_api(backend)
    if backend_restart is not None:
        backend.restore_restart_state(backend_restart)
    if backend_initializer is not None:
        backend_initializer(backend)
    diagnostic = backend_diagnostic_mapping(backend.diagnostic_snapshot())
    if float(driver.atmosphere.state.time_seconds) != float(diagnostic["metadata"]["time_seconds"]):
        raise RuntimeError("driver and public backend diagnostic clocks differ")
    if not retain_initial_diagnostic_snapshot():
        # The start-time snapshot's device groups (GWDO level fields, surface,
        # soil and precipitation copies, about 1 KB per column) were read only
        # by the clock check above; every history frame takes its own
        # snapshot.  Keep the host groups and release the device copies.
        diagnostic = {name: diagnostic[name] for name in ("metadata", "receipt")}
    stack: dict[str, Any] = {
        "driver": driver,
        "backend": backend,
        "prep_geometry": prep_geometry,
        "physics_geometry": physics_geometry,
        "gwdo_static": gwdo_static,
        "constructor": constructor,
        "api": api,
        "f000_surface_diagnostics": host["f000_surface_diagnostics"],
        "initial_diagnostic": diagnostic,
        # The number of cells this run SOLVES.  On a global mesh it is every
        # cell; on a limited-area mesh it is every cell except native's
        # garbage element, which is the column no history frame may publish.
        "solve_cells": (
            None if regional is None else int(driver.regional_v841.n_cells_solve)
        ),
        "solve_edges": (
            None if regional is None else int(driver.regional_v841.n_edges_solve)
        ),
        # The seven-ring boundary mask, on the host, so a health reading can
        # say WHICH zone a growing field sits in rather than only how big it
        # got.  ``None`` on a global run, which has no rings.
        "bdy_mask_cell": (
            None
            if regional is None
            else np.asarray(
                driver.regional_v841.masks_host.bdy_mask_cell, dtype=np.int32
            )
        ),
    }
    if gf_dynamics_tendencies is not None:
        # Restore side of the step-16 restart divergence: re-seed the driver-owned GF advective-forcing
        # carrier the checkpoint step's dynamics formed, so the FIRST resumed
        # step's begin_step consumes exactly the pair the unbroken run's next
        # step consumes.  Without this the resumed step 16 runs GF on zero
        # rthdynten/rqvdynten lanes and every downstream field diverges.
        from woof.hex.cuda_driver import CudaV841GfDynamicsTendencies

        cp = driver.cp
        carrier = CudaV841GfDynamicsTendencies(
            rthdynten=cp.asarray(
                np.ascontiguousarray(
                    gf_dynamics_tendencies["rthdynten"], dtype=np.float32
                )
            ),
            rqvdynten=cp.asarray(
                np.ascontiguousarray(
                    gf_dynamics_tendencies["rqvdynten"], dtype=np.float32
                )
            ),
            time_seconds=float(gf_dynamics_tendencies["time_seconds"]),
        )
        carrier.validate(cp=cp, n_vert_levels=N_LEVELS, n_cells=N_CELLS)
        if float(carrier.time_seconds) != float(
            driver.atmosphere.state.time_seconds
        ):
            raise RuntimeError(
                "restored GF advective forcing must be stamped with the "
                "restored model time "
                f"{float(driver.atmosphere.state.time_seconds)} s; got "
                f"t={float(carrier.time_seconds)} s"
            )
        stack["gf_dynamics_tendencies"] = carrier
    return stack


def _previous_surface_updates(
    stack: Mapping[str, Any],
    surface_accumulators: Sequence[str] = ("rainnc", "snownc", "graupelnc"),
) -> Mapping[str, Any] | None:
    snapshot = backend_diagnostic_mapping(stack["backend"].diagnostic_snapshot())
    # Adapter diagnostic schema may group these under precipitation.  Walk by
    # leaf name without downloading the resident arrays.  The leaves are the
    # RUN's row's accumulators; the default is the frozen WSM6 lane's three.
    wanted = set(surface_accumulators)
    leaves: dict[str, Any] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in wanted:
                    leaves[key] = item
                else:
                    walk(item)

    walk(snapshot)
    return leaves if set(leaves) == wanted else None


def _run_steps(
    *,
    stack: Mapping[str, Any],
    start_step: int,
    end_step: int,
    capture_steps: set[int],
    boundary_observer: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[int, dict[str, Any]], Mapping[str, Any] | None, list[dict[str, Any]]]:
    snapshots: dict[int, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    previous = _previous_surface_updates(stack)
    if start_step in capture_steps:
        snapshots[start_step] = capture_snapshot(
            label=SNAPSHOT_LABELS[start_step],
            step=start_step,
            driver=stack["driver"],
            backend=stack["backend"],
            prep_geometry=stack["prep_geometry"],
            kernel_cache=stack["driver"].cache,
            f000_surface_diagnostics=stack["f000_surface_diagnostics"],
            solve_cells=stack.get("solve_cells"),
        )
    # GF advective forcing carried step to step.  None at the first step of
    # this leg is native's own start state (tend_physics is zero before the
    # first dynamics step forms it).
    gf_dynamics_tendencies = stack.get("gf_dynamics_tendencies")
    for step in range(start_step + 1, end_step + 1):
        result = execute_composite_step(
            driver=stack["driver"],
            backend=stack["backend"],
            scalar_names=SCALAR_NAMES,
            physics_geometry=stack["physics_geometry"],
            kernel_cache=stack["driver"].cache,
            previous_surface_updates=previous,
            dynamics_tendencies=gf_dynamics_tendencies,
        )
        gf_dynamics_tendencies = result.dynamics_tendencies
        stack["gf_dynamics_tendencies"] = gf_dynamics_tendencies
        previous = result.committed.surface_updates
        backend_receipt = dict(result.backend_receipt)
        surface_execution = require_arwen_v2_surface_execution(
            backend_receipt, executed=True, label=f"step {step} backend receipt"
        )
        receipts.append(
            {
                "step": step,
                "driver": asdict(result.committed.receipt),
                "backend": backend_receipt,
                "arwen_v2_surface_execution": surface_execution,
                "clamp_d2h": result.clamp_d2h.as_dict(),
                "recovery": result.recovery.receipt(),
            }
        )
        if boundary_observer is not None:
            boundary_observer(step, stack)
        if step in capture_steps:
            snapshots[step] = capture_snapshot(
                label=SNAPSHOT_LABELS[step],
                step=step,
                driver=stack["driver"],
                backend=stack["backend"],
                prep_geometry=stack["prep_geometry"],
                kernel_cache=stack["driver"].cache,
                f000_surface_diagnostics=stack["f000_surface_diagnostics"],
                solve_cells=stack.get("solve_cells"),
            )
    return snapshots, previous, receipts


def _static_output_fields(host: Mapping[str, Any]) -> dict[str, np.ndarray]:
    mesh = host["prepared"].mesh
    # ``prepared.mesh`` is the SOLVE mesh on both lanes; the sealed
    # constructor is one column wider on a limited-area run, because the
    # physics seam works on native's padded extent.  A history frame
    # publishes the domain, not the allocation, so the terrain carried from
    # the constructor is cut back to the mesh's own cell count here -- the
    # coordinate arrays beside it come from the mesh and are already that
    # length, and mixing the two is a netCDF write that refuses on a
    # broadcast rather than a picture that is one cell wrong.
    solve_cells = int(np.asarray(_mesh_value(mesh, "latCell")).size)
    terrain = np.ascontiguousarray(
        host["constructor_values"]["terrain_height_m"], dtype=np.float32
    )
    if terrain.size not in (solve_cells, solve_cells + 1):
        raise ValueError(
            f"terrain carries {terrain.size} columns against a mesh of "
            f"{solve_cells}; a history frame cannot say which cell each "
            "height belongs to"
        )
    return {
        "indexToCellID": np.ascontiguousarray(_mesh_value(mesh, "indexToCellID"), dtype=np.int32),
        "indexToEdgeID": np.ascontiguousarray(_mesh_value(mesh, "indexToEdgeID"), dtype=np.int32),
        "latEdge": np.ascontiguousarray(_mesh_value(mesh, "latEdge"), dtype=np.float32),
        "lonEdge": np.ascontiguousarray(_mesh_value(mesh, "lonEdge"), dtype=np.float32),
        "latCell": np.ascontiguousarray(_mesh_value(mesh, "latCell"), dtype=np.float32),
        "lonCell": np.ascontiguousarray(_mesh_value(mesh, "lonCell"), dtype=np.float32),
        "ter": np.ascontiguousarray(terrain[:solve_cells], dtype=np.float32),
    }




def require_snapshot_receipt_surface_execution(
    snapshot_receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Validate the exact capture receipt and its nested Arwen execution payload."""

    if label not in ("F000", "F030", "F001"):
        raise ValueError(f"unsupported baseline snapshot label {label!r}")
    step = {"F000": 0, "F030": CHECKPOINT_STEP, "F001": FULL_STEPS}[label]
    if (
        snapshot_receipt.get("schema") != SNAPSHOT_SCHEMA
        or snapshot_receipt.get("label") != label
        or snapshot_receipt.get("step") != step
        or snapshot_receipt.get("time_seconds") != step * DT_SECONDS
    ):
        raise ValueError(f"baseline diagnostic {label} capture receipt changed")
    nested = snapshot_receipt.get("arwen_v2_surface_execution")
    if not isinstance(nested, Mapping):
        raise ValueError(
            f"baseline diagnostic {label} lacks nested WOOF surface execution"
        )
    return require_arwen_v2_surface_execution(
        nested, executed=(step > 0), label=f"baseline diagnostic {label}"
    )


def _write_baseline_diagnostic_output(
    *,
    diagnostic_root: Path,
    host: Mapping[str, Any],
    baseline_snapshots: Mapping[int, Mapping[str, Any]],
    step_receipts: Sequence[Mapping[str, Any]],
    atmosphere_fingerprint: Mapping[str, Any],
    backend_fingerprint: Mapping[str, Any],
    physical_gates: Mapping[str, Any],
    weather_activity: Mapping[str, Any],
    native_comparisons: Mapping[str, Any],
    capability: Mapping[str, Any],
    memory_admission: Mapping[str, Any],
    arwen_pin: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish plot-capable baseline bytes without weakening release output."""

    classification = dict(host["constructor_receipt"]["surface_classification"])
    classification_core = {
        name: classification[name] for name in EXPECTED_SURFACE_CLASSIFICATION
    }
    if classification_core != dict(EXPECTED_SURFACE_CLASSIFICATION):
        raise RuntimeError("baseline diagnostic surface classification changed")
    snapshot_receipts = {
        SNAPSHOT_LABELS[step]: baseline_snapshots[step]["receipt"]
        for step in SNAPSHOT_STEPS
    }
    surface_execution = {
        label: require_snapshot_receipt_surface_execution(
            snapshot_receipts[label], label=label
        )
        for label in ("F000", "F030", "F001")
    }
    static = _static_output_fields(host)
    files = {
        label: write_snapshot_netcdf(
            diagnostic_root / SNAPSHOT_FILE_NAMES[label],
            baseline_snapshots[step],
            static,
        )
        for step, label in (
            (0, "F000"),
            (CHECKPOINT_STEP, "F030"),
            (FULL_STEPS, "F001"),
        )
    }
    proof = {
        "full_physics_cuda_executed": True,
        "uninterrupted_baseline_passed": True,
        "outer_steps": FULL_STEPS,
        "forecast_seconds": FULL_STEPS * DT_SECONDS,
        "restart_bitwise_identical": None,
        "checkpoint_restart": {
            "evaluated": False,
            "passed": False,
            "status": "not_evaluated",
        },
        "source_pins": source_receipt,
        "authority": authority_receipt,
        "capability": capability,
        "memory_admission": memory_admission,
        "arwen_pre_kernel_cache_pin": arwen_pin,
        "host_preparation": {
            "constructor": host["constructor_receipt"],
            "scalars": host["scalar_receipt"],
            "mesh_overlay": host["mesh_evidence"],
            "reconstruction_coefficients": host["reconstruction_coefficients"],
            "edge_normal_vectors": host["edge_normal_vectors"],
            "inactive_deformation": host["defc"],
        },
        "surface_classification": classification_core,
        "glacier_cuda_evidence": {
            "required_path": ARWEN_GLACIER_CUDA_PROVENANCE,
            "expected_census": dict(EXPECTED_NOAHMP_CENSUS),
            "snapshot_execution": surface_execution,
            "contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
        },
        "uninterrupted": {
            "steps": FULL_STEPS,
            "step_receipts": list(step_receipts),
            "snapshot_receipts": snapshot_receipts,
            "f001_full_state_fingerprints": {
                "atmosphere": atmosphere_fingerprint,
                "backend": backend_fingerprint,
            },
        },
        "physical_gates": physical_gates,
        "weather_activity": weather_activity,
        "native_comparisons": native_comparisons,
        "snapshot_files": files,
    }
    payload = {
        "schema": BASELINE_DIAGNOSTIC_SCHEMA,
        "status": BASELINE_DIAGNOSTIC_STATUS,
        "warning": BASELINE_DIAGNOSTIC_WARNING,
        "release_eligible": False,
        "source_release": SOURCE_RELEASE,
        "arwen_commit": ARWEN_COMMIT,
        "arwen_contract_document_sha256": ARWEN_CONTRACT_DOCUMENT_SHA256,
        "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
        "claim": "uninterrupted 30-step engineering baseline only",
        "nonclaims": [
            *NONCLAIMS,
            "checkpoint/restart identity has not been evaluated or passed",
            "this diagnostic is not a release proof",
        ],
        "proof": proof,
        "weather_plot_policy": "native Rust/WOOF renderer only; q2 ships in the history stream and its weather-field plots go through the same renderer",
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    receipt = diagnostic_root / BASELINE_DIAGNOSTIC_RECEIPT_NAME
    _write_exclusive_json(receipt, payload)
    expected_inventory = {
        BASELINE_DIAGNOSTIC_RECEIPT_NAME,
        *(SNAPSHOT_FILE_NAMES[label] for label in ("F000", "F030", "F001")),
    }
    if {path.name for path in diagnostic_root.iterdir()} != expected_inventory:
        raise RuntimeError("baseline diagnostic output inventory changed")
    return {
        "path": str(receipt),
        "bytes": receipt.stat().st_size,
        "sha256": sha256_file(receipt),
        "payload_sha256": payload["payload_sha256"],
        "status": BASELINE_DIAGNOSTIC_STATUS,
        "release_eligible": False,
    }


def _execute_full_proof(
    *,
    host: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
    arwen_checkout: Path,
    source_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    authority_paths: Mapping[str, Path],
    baseline_diagnostic_output: Path | None,
) -> dict[str, Any]:
    from woof.hex.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    arwen_pin = dict(pin_arwen_physics_v841(arwen_checkout))
    # This must precede KernelCache's woof platform-binding construction.
    from woof.hex.cuda_backend import KernelCache, require_cuda


    # No sm_120 pin (``required_compute``): it named no breakage.  A card of
    # another architecture runs the proof and its receipt says whether that
    # architecture holds an anchor; the frozen digests decide the verdict.
    capability = require_cuda(min_compute=(12, 0), cache_dir=cache_root)
    import cupy as cp

    memory = gpu_memory_admission(cp)
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    from woof.hex.cuda_dualrun import fingerprint_atmosphere
    stack = _construct_device_stack(
        host=host, cache=cache, arwen_checkout=arwen_checkout
    )
    baseline_snapshots, _, first_receipts = _run_steps(
        stack=stack,
        start_step=0,
        end_step=CHECKPOINT_STEP,
        capture_steps={0, CHECKPOINT_STEP},
    )
    checkpoint = download_driver_checkpoint(
        stack["driver"],
        stack["backend"],
        dynamics_tendencies=stack.get("gf_dynamics_tendencies"),
    )
    baseline_step16_fingerprints: dict[str, Any] = {}

    def record_baseline_step16(step: int, current_stack: Mapping[str, Any]) -> None:
        if step == CHECKPOINT_STEP + 1:
            baseline_step16_fingerprints.update(
                fingerprint_execution_boundary(current_stack)
            )

    continuation_snapshots, _, second_receipts = _run_steps(
        stack=stack,
        start_step=CHECKPOINT_STEP,
        end_step=FULL_STEPS,
        capture_steps={FULL_STEPS},
        boundary_observer=record_baseline_step16,
    )
    if set(baseline_step16_fingerprints) != {"atmosphere", "backend"}:
        raise RuntimeError("uninterrupted continuation did not capture exact step 16")
    baseline_snapshots.update(continuation_snapshots)
    baseline_f001_atmosphere = fingerprint_atmosphere(stack["driver"].atmosphere)
    baseline_f001_backend = fingerprint_nested_arrays(
        stack["backend"].restart_state()
    )

    gates = {
        SNAPSHOT_LABELS[step]: physical_snapshot_gate(
            baseline_snapshots[step], allow_initial_negative_qv=(step == 0)
        )
        for step in SNAPSHOT_STEPS
    }
    evolution = surface_evolution_gate(
        baseline_snapshots[0], baseline_snapshots[FULL_STEPS]
    )
    comparisons = {
        "F000": compare_snapshot_to_native(
            baseline_snapshots[0], authority_paths["native_f000"]
        ),
        "F030": compare_snapshot_to_native(
            baseline_snapshots[CHECKPOINT_STEP], authority_paths["native_f030"]
        ),
        "F001": compare_snapshot_to_native(
            baseline_snapshots[FULL_STEPS], authority_paths["native_f001"]
        ),
    }
    baseline_diagnostic = None
    if baseline_diagnostic_output is not None:
        baseline_diagnostic = _write_baseline_diagnostic_output(
            diagnostic_root=baseline_diagnostic_output,
            host=host,
            baseline_snapshots=baseline_snapshots,
            step_receipts=first_receipts + second_receipts,
            atmosphere_fingerprint=baseline_f001_atmosphere,
            backend_fingerprint=baseline_f001_backend,
            physical_gates=gates,
            weather_activity=evolution,
            native_comparisons=comparisons,
            capability=capability.as_dict(),
            memory_admission=memory,
            arwen_pin=arwen_pin,
            source_receipt=source_receipt,
            authority_receipt=authority_receipt,
        )

    # Release the uninterrupted device graph before the restart arm.  The
    # F030 checkpoint and F001 snapshots are detached host data and remain
    # valid.
    del stack
    # The driver/backend/seam graph carries reference cycles (state <->
    # driver <-> adapters), so its device arrays survive `del` until the
    # cycle collector runs; collect explicitly so the pools can return
    # their blocks to the driver and the fresh worker process below can
    # admit the full-physics footprint.
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # THE REAL RESTART PATH IS A FRESH PROCESS, so the restart arm runs in
    # a fresh worker process, exactly like a real resumed forecast, and
    # every bitwise identity gate below is unchanged.
    #
    # The step-16 restart divergence, closed by checkpoint schema v3: the GF advective-forcing pair
    # (rthdynten/rqvdynten) is driver-owned per-step carried state formed
    # by each step's dynamics and consumed by the NEXT begin_step.  It
    # lives outside both the MPAS atmosphere and the Arwen backend restart
    # payload, so the v2 checkpoint never carried it and every restored
    # arm re-entered step 16 on zero forcing lanes while the unbroken run
    # fed the real step-15 pair -- a deterministic all-fields divergence
    # at exactly step 16 (measured 5/5 on the reference node,
    # 2026-08-24).  The v3 checkpoint downloads the pair at F030 and the
    # restore re-seeds it, gated by its own fingerprint identity below.
    #
    # KNOWN LIMIT (measured 2026-08-12, escalated to the seam-contract
    # owner): the frozen Arwen phase-1 is not bitwise-pure with respect
    # to the CuPy device-pool layout at seam construction/run time.  With
    # a bitwise-verified F030 rehydration and identical code, perturbing
    # only the device pool before construction moved the very next
    # radiation-due phase-1: rthratenlw first diff at [g-point 0, level 0,
    # cell 64] 0x38511a95 vs 0x38511a4e (191,479 of 9,011,310 elements;
    # LW -> NoahMP -> YSU family only; SW, GF and raw dqr/dqs/dqg bitwise
    # identical; probe tools/probe_v841_lw_purity.py, arms none/none/host/
    # device).  Identical processes reproduce bit-for-bit; a step-16 FAIL
    # that survives the v3 forcing re-seed and its fingerprint gate is
    # that layout sensitivity, not a serialization defect.
    worker = _spawn_restart_worker(
        checkpoint=checkpoint,
        authority_paths=authority_paths,
        arwen_checkout=arwen_checkout,
        cache_root=cache_root,
        output_root=output_root,
    )
    restored_f030 = worker["restored_f030"]
    f030_rehydration = {
        "atmosphere": require_fingerprint_identity(
            "F030 restored MPAS atmosphere",
            checkpoint.atmosphere_fingerprint,
            restored_f030["atmosphere"],
        ),
        "backend": require_fingerprint_identity(
            "F030 restored WOOF backend",
            checkpoint.backend_fingerprint,
            restored_f030["backend"],
        ),
        "gf_forcing": require_fingerprint_identity(
            "F030 restored GF advective forcing",
            checkpoint.gf_forcing_fingerprint,
            worker["restored_gf_forcing"],
        ),
    }
    step16_identity = {
        "atmosphere": require_fingerprint_identity(
            "first resumed step 16 MPAS atmosphere",
            baseline_step16_fingerprints["atmosphere"],
            worker["step16"]["atmosphere"],
        ),
        "backend": require_fingerprint_identity(
            "first resumed step 16 WOOF backend",
            baseline_step16_fingerprints["backend"],
            worker["step16"]["backend"],
        ),
    }
    restart_f001 = worker["restart_f001"]
    restart_receipts = worker["continuation_step_receipts"]
    identity = require_bitwise_restart_identity(
        baseline_snapshots[FULL_STEPS], restart_f001
    )
    restored_f001_atmosphere = worker["f001_atmosphere_fingerprint"]
    restored_f001_backend = worker["f001_backend_fingerprint"]
    f001_full_state_identity = {
        "atmosphere": require_fingerprint_identity(
            "F030 restart continuation F001 MPAS atmosphere",
            baseline_f001_atmosphere,
            restored_f001_atmosphere,
        ),
        "backend": require_fingerprint_identity(
            "F030 restart continuation F001 WOOF backend",
            baseline_f001_backend,
            restored_f001_backend,
        ),
    }
    full_restart_identity = {
        "atmosphere": baseline_f001_atmosphere,
        "backend": baseline_f001_backend,
        "restart_arm_fresh_process": True,
        "f030_rehydration": f030_rehydration,
        "first_resumed_step16": step16_identity,
        "f001_full_state": f001_full_state_identity,
        "bitwise_identical": True,
    }

    gates["F001_RESTART"] = physical_snapshot_gate(
        restart_f001, allow_initial_negative_qv=False
    )
    static = _static_output_fields(host)
    files = {
        "F000": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F000"], baseline_snapshots[0], static
        ),
        "F030": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F030"],
            baseline_snapshots[CHECKPOINT_STEP],
            static,
        ),
        "F001": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F001"],
            baseline_snapshots[FULL_STEPS],
            static,
        ),
        "F001_RESTART": write_snapshot_netcdf(
            output_root / SNAPSHOT_FILE_NAMES["F001_RESTART"], restart_f001, static
        ),
    }
    from woof.hex.cuda_backend.arch_admission import architecture_status

    return {
        "capability": capability.as_dict(),
        "architecture": architecture_status(capability.compute).as_dict(),
        "arwen_pre_kernel_cache_pin": arwen_pin,
        "memory_admission": memory,
        "source_pins": source_receipt,
        "authority": authority_receipt,
        "host_preparation": {
            "constructor": host["constructor_receipt"],
            "scalars": host["scalar_receipt"],
            "mesh_overlay": host["mesh_evidence"],
            "reconstruction_coefficients": host["reconstruction_coefficients"],
            "edge_normal_vectors": host["edge_normal_vectors"],
            "inactive_deformation": host["defc"],
        },
        "uninterrupted": {
            "steps": FULL_STEPS,
            "step_receipts": first_receipts + second_receipts,
            "snapshot_receipts": {
                SNAPSHOT_LABELS[step]: baseline_snapshots[step]["receipt"]
                for step in SNAPSHOT_STEPS
            },
        },
        "checkpoint_restart": {
            "schema": CHECKPOINT_SCHEMA,
            "f001_restart_snapshot_receipt": restart_f001["receipt"],
            "step": CHECKPOINT_STEP,
            "time_seconds": checkpoint.model_time_seconds,
            "checkpoint_atmosphere": dict(checkpoint.atmosphere_fingerprint),
            "checkpoint_backend": dict(checkpoint.backend_fingerprint),
            "restart_execution": {
                "fresh_process": True,
                "worker_schema": RESTART_WORKER_SCHEMA,
                "worker_files": worker["files"],
                "worker_capability": worker["capability"],
                "worker_memory_admission": worker["memory_admission"],
            },
            "continuation_step_receipts": restart_receipts,
            "f001_full_state_identity": full_restart_identity,
            "f001_identity": identity,
        },
        "physical_gates": gates,
        "weather_activity": evolution,
        "native_comparisons": comparisons,
        "snapshot_files": files,
        "baseline_diagnostic_output": baseline_diagnostic,
        "gf_deviation": {
            "mpas_dynamics_tendencies_computed": True,
            "fa35_public_api_accepts_them": False,
            "fa35_rthften_rqvften": "zero",
            "native_gf_parity_claim": False,
        },
        "full_physics_cuda_executed": True,
        "forecast_seconds": FULL_STEPS * DT_SECONDS,
        "outer_steps": FULL_STEPS,
        "restart_bitwise_identical": True,
    }


def _spawn_restart_worker(
    *,
    checkpoint: HostDriverCheckpoint,
    authority_paths: Mapping[str, Path],
    arwen_checkout: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the F030 restart continuation in a genuinely fresh process."""

    worker_dir = output_root / "restart-worker"
    worker_dir.mkdir(parents=False, exist_ok=False)
    input_path = worker_dir / "restart-worker-input.pkl"
    output_path = worker_dir / "restart-worker-output.pkl"
    job = {
        "schema": RESTART_WORKER_SCHEMA,
        "arwen_commit": ARWEN_COMMIT,
        "checkpoint": checkpoint,
        "authority_paths": {
            role: str(path) for role, path in authority_paths.items()
        },
        "arwen_checkout": str(arwen_checkout),
        "cache_root": str(cache_root / "restart-worker-cache"),
    }
    with input_path.open("xb") as stream:
        pickle.dump(job, stream, protocol=4)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restart-worker-input",
            str(input_path),
            "--restart-worker-output",
            str(output_path),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh-process restart worker failed with return code "
            f"{completed.returncode}"
        )
    with output_path.open("rb") as stream:
        results = pickle.load(stream)
    if not isinstance(results, Mapping) or results.get("schema") != RESTART_WORKER_SCHEMA:
        raise RuntimeError("restart worker returned an unrecognized payload schema")
    if results.get("arwen_commit") != ARWEN_COMMIT:
        raise RuntimeError("restart worker executed against a different WOOF commit")
    if results.get("fresh_process") is not True:
        raise RuntimeError("restart worker did not attest fresh-process execution")
    required = {
        "restored_f030",
        "restored_gf_forcing",
        "step16",
        "restart_f001",
        "continuation_step_receipts",
        "f001_atmosphere_fingerprint",
        "f001_backend_fingerprint",
        "capability",
        "memory_admission",
    }
    missing = sorted(required - set(results))
    if missing:
        raise RuntimeError(f"restart worker payload lacks fields {missing}")
    results = dict(results)
    results["files"] = {
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    return results


def _execute_restart_worker(input_path: Path, output_path: Path) -> int:
    """Worker entry: restore the F030 checkpoint and continue to F001."""

    with Path(input_path).open("rb") as stream:
        job = pickle.load(stream)
    if not isinstance(job, Mapping) or job.get("schema") != RESTART_WORKER_SCHEMA:
        raise RuntimeError("restart worker input schema mismatch")
    if job.get("arwen_commit") != ARWEN_COMMIT:
        raise RuntimeError("restart worker input WOOF commit mismatch")
    checkpoint = job["checkpoint"]
    if not isinstance(checkpoint, HostDriverCheckpoint):
        raise TypeError("restart worker requires a HostDriverCheckpoint payload")
    if float(checkpoint.model_time_seconds) != CHECKPOINT_STEP * DT_SECONDS:
        raise RuntimeError("restart worker checkpoint is not at exact F030")
    forcing = getattr(checkpoint, "gf_dynamics_tendencies", None)
    if forcing is None:
        raise RuntimeError(
            "checkpoint predates the GF advective-forcing capture "
            f"({CHECKPOINT_SCHEMA}): restoring it would feed GF zero "
            "rthdynten/rqvdynten lanes at the first resumed step -- the "
            "deterministic step-16 restart divergence.  Re-run the "
            "baseline arm to mint a checkpoint that carries the pair."
        )
    paths = {role: Path(value) for role, value in dict(job["authority_paths"]).items()}
    arwen_checkout = _plain_absolute(Path(job["arwen_checkout"]), "WOOF checkout")
    # Source pins verify first: the checkout guard imports the manifest from a
    # pinned module, so that module's bytes are proven before its constants
    # are trusted.
    require_frozen_execution_sources()
    verify_arwen_checkout_git(arwen_checkout)
    authority_receipt = verify_authorities(paths)
    host = _prepare_host_execution(paths, authority_receipt)

    from woof.hex.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(arwen_checkout)
    from woof.hex.cuda_backend import KernelCache, require_cuda
    from woof.hex.cuda_backend.arch_admission import architecture_status

    cache_root = _plain_absolute(Path(job["cache_root"]), "restart worker cache root")
    cache_root.mkdir(parents=False, exist_ok=False)
    # No sm_120 pin (``required_compute``): it named no breakage.  A card of
    # another architecture runs the proof and its receipt says whether that
    # architecture holds an anchor; the frozen digests decide the verdict.
    capability = require_cuda(min_compute=(12, 0), cache_dir=cache_root)
    import cupy as cp

    memory = gpu_memory_admission(
        cp, minimum=RESTART_WORKER_MIN_FREE_DEVICE_BYTES
    )
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    from woof.hex.cuda_dualrun import fingerprint_atmosphere

    stack = _construct_device_stack(
        host=host,
        cache=cache,
        arwen_checkout=arwen_checkout,
        state=checkpoint.state,
        saved_diagnostics=checkpoint.saved_diagnostics,
        backend_restart=checkpoint.backend_state,
        gf_dynamics_tendencies=forcing,
    )
    restored_f030 = fingerprint_execution_boundary(stack)
    require_fingerprint_identity(
        "F030 restored MPAS atmosphere (fresh restart process)",
        checkpoint.atmosphere_fingerprint,
        restored_f030["atmosphere"],
    )
    require_fingerprint_identity(
        "F030 restored WOOF backend (fresh restart process)",
        checkpoint.backend_fingerprint,
        restored_f030["backend"],
    )
    restored_carrier = stack["gf_dynamics_tendencies"]
    restored_gf_forcing = fingerprint_nested_arrays(
        {
            "rthdynten": np.ascontiguousarray(
                cp.asnumpy(restored_carrier.rthdynten)
            ),
            "rqvdynten": np.ascontiguousarray(
                cp.asnumpy(restored_carrier.rqvdynten)
            ),
            "time_seconds": float(restored_carrier.time_seconds),
        }
    )
    require_fingerprint_identity(
        "F030 restored GF advective forcing (fresh restart process)",
        checkpoint.gf_forcing_fingerprint,
        restored_gf_forcing,
    )
    step16: dict[str, Any] = {}

    def capture_step16(step: int, current_stack: Mapping[str, Any]) -> None:
        if step == CHECKPOINT_STEP + 1:
            step16.update(fingerprint_execution_boundary(current_stack))

    restart_snapshots, _, restart_receipts = _run_steps(
        stack=stack,
        start_step=CHECKPOINT_STEP,
        end_step=FULL_STEPS,
        capture_steps={FULL_STEPS},
        boundary_observer=capture_step16,
    )
    if set(step16) != {"atmosphere", "backend"}:
        raise RuntimeError("restart worker did not capture exact step 16")
    results = {
        "schema": RESTART_WORKER_SCHEMA,
        "arwen_commit": ARWEN_COMMIT,
        "fresh_process": True,
        "capability": capability.as_dict(),
        "architecture": architecture_status(capability.compute).as_dict(),
        "memory_admission": memory,
        "restored_f030": restored_f030,
        "restored_gf_forcing": restored_gf_forcing,
        "step16": step16,
        "restart_f001": restart_snapshots[FULL_STEPS],
        "continuation_step_receipts": restart_receipts,
        "f001_atmosphere_fingerprint": fingerprint_atmosphere(
            stack["driver"].atmosphere
        ),
        "f001_backend_fingerprint": fingerprint_nested_arrays(
            stack["backend"].restart_state()
        ),
    }
    with Path(output_path).open("xb") as stream:
        pickle.dump(results, stream, protocol=4)
    print(json.dumps({"restart_worker": "complete", "schema": RESTART_WORKER_SCHEMA}))
    return 0


def installed_engine_provenance(root: Path) -> dict[str, Any] | None:
    """The identity of an INSTALLED woof, or ``None`` when ``root`` is not it.

    An install carries the executed bytes and no commit, so it is named by
    what pip recorded instead: the distribution's version, the SHA-256 of
    its ``RECORD`` (every installed file with its own digest, written once at
    install time) and, when pip wrote one, ``direct_url.json`` (where the
    wheel came from).  Two runs on the same install record the same three;
    a reinstall or an upgrade moves the RECORD digest.

    Only the root the interpreter would import woof from qualifies
    (:func:`woof.hex.engine_pin.installed_root`): any other directory without
    a git working tree -- an unpacked tarball, a copied site-packages -- has
    neither a commit nor a record and is still refused.
    """

    from importlib.metadata import PackageNotFoundError, distribution

    from woof.hex import engine_pin

    installed = engine_pin.installed_root()
    if installed is None or Path(installed).resolve() != Path(root).resolve():
        return None
    try:
        dist = distribution(engine_pin.DISTRIBUTION)
    except PackageNotFoundError:
        return None
    record = dist.read_text("RECORD")
    if not record:
        return None
    direct_url = dist.read_text("direct_url.json")
    return {
        "name": engine_pin.DISTRIBUTION,
        "version": dist.version,
        "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
        "direct_url": json.loads(direct_url) if direct_url else None,
    }


def verify_arwen_checkout_git(
    checkout: Path, manifest: Mapping[str, str] | None = None,
    *, require_committed: bool = True,
) -> dict[str, Any]:
    """Admit an Arwen tree iff its executed seam source is the proven one.

    The gate is ``ARWEN_SOURCE_MANIFEST`` -- the sha-freeze of the sixteen
    woof files the port executes.  Everything else recorded here is
    PROVENANCE, written into every receipt and every history file so the
    executed source can be named, and none of it gates execution: a commit
    that moves nothing the seam executes must not invalidate a proven pin.

    Two kinds of tree are admitted, and the record says which (``kind``):

    * a woof GIT WORKING TREE: its HEAD, tree and dirty paths are recorded.
      A dirty manifest file still refuses, because the receipt could not
      then name the executed bytes by commit; dirt in unrelated files is
      recorded loudly and execution proceeds.
    * the INSTALLED woof this interpreter imports: its version, the SHA-256
      of pip's RECORD and its direct_url are recorded
      (:func:`installed_engine_provenance`).  This is what lets the forecast
      run from ``pip install woof hex`` alone.  THE BREAKAGE IT CLOSES:
      through 0.3.1 an install was refused here with "not a git working
      tree" although all sixteen pinned files resolve from site-packages
      (they have since engine 2.5.8), so the forecast was unreachable
      without a woof clone at the pinned tag.

    Any other directory -- an unpacked tarball, a copied site-packages -- has
    neither a commit nor a record and is refused by name.
    """

    root = checkout.resolve(strict=True)

    try:
        probe = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
        )
    except OSError:
        # Installed wheels carry RECORD provenance even when git is absent.
        probe = None
    installed: dict[str, Any] | None = None
    if probe is None or probe.returncode != 0:
        installed = installed_engine_provenance(root)
        if installed is None:
            raise RuntimeError(
                f"{root} is neither a woof git working tree nor the woof "
                "this interpreter has installed, and the run has to be able "
                "to name the executed seam source: a git tree by its HEAD, "
                "tree and dirty paths, an install by its version and pip's "
                "RECORD digest.  Leave --gpuwm-checkout unset to run on the "
                "installed engine, or pass a clone of woof at the pinned tag."
            )

    def git(*arguments: str, strip: bool = True) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            raise RuntimeError(f"WOOF git command wrote stderr: {completed.stderr!r}")
        return completed.stdout.strip() if strip else completed.stdout

    head: str | None = None
    tree: str | None = None
    dirty_paths: list[str] = []
    if installed is None:
        top = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
        head = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        # NUL-delimited and unstripped: a porcelain entry BEGINS with a status
        # column that may be a space, which str.strip would silently eat off
        # the first entry, shifting every parsed path by one byte.
        status = git(
            "status", "--porcelain=v1", "-z", "--untracked-files=all", strip=False
        )
        if top != root:
            raise RuntimeError("WOOF checkout is not the exact requested Git root")

        expect_rename_origin = False
        for entry in status.split("\0"):
            if not entry:
                continue
            if expect_rename_origin:
                dirty_paths.append(entry)
                expect_rename_origin = False
                continue
            code, path = entry[:2], entry[3:]
            dirty_paths.append(path)
            if code and code[0] in "RC":
                expect_rename_origin = True
        dirty_paths = sorted(set(dirty_paths))

    # The bytes the RUN's row is bound by: the sixteen this tree's bytes
    # name (the public pin, or an admitted engine's own row --
    # checkout_manifest_for) unless the caller hands the manifest of the pin
    # its row runs under (a provider's row binds a different column batch
    # through its own manifest and can never match the frozen sixteen by
    # design).  Provenance is recorded either way.
    manifest = checkout_manifest_for(root) if manifest is None else dict(manifest)
    files: dict[str, str] = {}
    for relative, expected in manifest.items():
        target = root / relative
        if not target.is_file():
            raise RuntimeError(
                f"{relative} is missing from the WOOF checkout {root} â€” the "
                "seam's executed source moved; re-prove before running"
            )
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"{relative} does not match the proven manifest (expected "
                f"{expected}, found {actual}) â€” the seam's executed source "
                "moved; re-prove before running"
            )
        files[relative] = actual

    dirty_manifest = sorted(set(dirty_paths) & set(manifest))
    # ``require_committed=False`` is the FORECAST's call (2026-10-10): a
    # forecast on an engine tree with uncommitted seam edits runs, and the
    # receipt names those files by path and by the digest found
    # (``dirty_manifest_paths`` below, ``manifest.files``).  The x4 proof
    # keeps the refusal, because its receipt is a claim about a commit.
    if dirty_manifest and require_committed:
        raise RuntimeError(
            f"{dirty_manifest[0]} is dirty in the WOOF checkout â€” its bytes "
            "cannot be provenanced to any commit; commit or restore it, then "
            "re-prove before running"
        )
    if dirty_paths:
        print(
            json.dumps(
                {
                    "arwen_checkout_dirty_non_manifest_paths": dirty_paths,
                    "note": (
                        "no manifest file moved; recorded as provenance and "
                        "execution proceeds"
                    ),
                },
                sort_keys=True,
            )
        )
    record: dict[str, Any] = {
        "root": str(root),
        "kind": "git" if installed is None else "installed",
        "head": head,
        "tree": tree,
        "clean": not dirty_paths,
        "dirty_paths": dirty_paths,
        "dirty_manifest_paths": dirty_manifest,
        "provenance": (
            "head/tree/dirty-state (git) or version/RECORD digest (install) "
            "are recorded, not gates; the gate is the sixteen-file "
            "executed-source manifest"
        ),
        "manifest": {
            "files": files,
            "sha256": canonical_json_sha256(files),
        },
    }
    if installed is not None:
        record["installed_distribution"] = installed
    return record


def default_arwen_checkout() -> Path | None:
    """The woof tree a run uses when none is named: the installed engine."""

    from woof.hex import engine_pin

    return engine_pin.installed_root()


def resolve_arwen_checkout(requested: Path | str | None) -> Path:
    """The named woof tree, or the installed engine when none is named."""

    if requested is None:
        installed = default_arwen_checkout()
        if installed is None:
            raise FileNotFoundError(
                "no woof tree was named and no woof is installed in this "
                "interpreter; install the pinned engine or pass "
                "--arwen-checkout <a woof clone at the pinned tag>"
            )
        requested = installed
    checkout = _plain_absolute(requested, "WOOF checkout")
    if not checkout.is_dir():
        raise FileNotFoundError(checkout)
    return checkout


def _arwen_seam_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """The executed-seam identity that must not move during a proof.

    HEAD/tree/dirty-state are provenance: an unrelated file appearing in the
    checkout mid-run is recorded in the receipt, not turned into a refusal
    that aborts a passed proof.
    """

    return {"root": record["root"], "manifest": record["manifest"]}


def _paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        role: Path(getattr(args, role)).expanduser().absolute()
        for role in AUTHORITY_PINS
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for role, default in default_authority_paths().items():
        parser.add_argument("--" + role.replace("_", "-"), type=Path, default=default)
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
        "--baseline-diagnostic-output",
        type=Path,
        help="fresh isolated F000/F030/F001 engineering-baseline output root",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify release/source/authority bytes and host mappings without CUDA",
    )
    parser.add_argument(
        "--restart-worker-input", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--restart-worker-output", type=Path, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.restart_worker_input is not None or args.restart_worker_output is not None:
        if args.restart_worker_input is None or args.restart_worker_output is None:
            parser.error(
                "restart worker requires both --restart-worker-input and "
                "--restart-worker-output"
            )
    elif not args.preflight_only and (args.cache_root is None or args.output is None):
        parser.error("execution requires --cache-root and --output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.restart_worker_input is not None:
        assert args.restart_worker_output is not None
        return _execute_restart_worker(
            args.restart_worker_input, args.restart_worker_output
        )
    paths = _paths_from_args(args)
    arwen_checkout = resolve_arwen_checkout(args.arwen_checkout)
    # None source pins refuse here, before authority hashing, host preparation,
    # destination creation, production imports, or CUDA probing.  They also
    # precede the checkout guard, which imports the seam manifest from a
    # pinned module: the module's bytes are proven before its constants are
    # trusted.
    source_before = require_frozen_execution_sources()
    arwen_git_before = verify_arwen_checkout_git(arwen_checkout)
    authority_before = verify_authorities(paths)
    host = _prepare_host_execution(paths, authority_before)

    if args.preflight_only:
        source_after = require_frozen_execution_sources()
        authority_after = verify_authorities(paths)
        arwen_git_after = verify_arwen_checkout_git(arwen_checkout)
        if (
            source_after != source_before
            or authority_after != authority_before
            or _arwen_seam_identity(arwen_git_after)
            != _arwen_seam_identity(arwen_git_before)
        ):
            raise RuntimeError(
                "source, authority, or WOOF seam bytes changed during preflight"
            )
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "mode": "preflight-only; CUDA not imported",
                    "sources": source_before,
                    "authorities_sha256": authority_before["sha256"],
                    "arwen_git": arwen_git_before,
                    "arwen_contract_document_sha256": ARWEN_CONTRACT_DOCUMENT_SHA256,
                    "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
                    "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
                    "constructor": host["constructor_receipt"],
                    "scalars": host["scalar_receipt"],
                    "reconstruction_coefficients": host["reconstruction_coefficients"],
                    "edge_normal_vectors": host["edge_normal_vectors"],
                    "claim": CLAIM,
                    "nonclaims": list(NONCLAIMS),
                },
                sort_keys=True,
            )
        )
        return 0

    assert args.cache_root is not None and args.output is not None
    cache_root, output_root = validate_destination(
        args.cache_root, args.output, tuple(paths.values())
    )
    baseline_diagnostic_output = None
    if args.baseline_diagnostic_output is not None:
        baseline_diagnostic_output = validate_baseline_diagnostic_destination(
            args.baseline_diagnostic_output,
            cache_root=cache_root,
            output_root=output_root,
            protected=tuple(paths.values()),
        )

    cache_root.mkdir(parents=False)
    output_root.mkdir(parents=False)
    if baseline_diagnostic_output is not None:
        baseline_diagnostic_output.mkdir(parents=False)
    started = time.perf_counter()
    proof = _execute_full_proof(
        host=host,
        cache_root=cache_root,
        output_root=output_root,
        arwen_checkout=arwen_checkout,
        source_receipt=source_before,
        authority_receipt=authority_before,
        authority_paths=paths,
        baseline_diagnostic_output=baseline_diagnostic_output,
    )
    source_after = require_frozen_execution_sources()
    authority_after = verify_authorities(paths)
    arwen_git_after = verify_arwen_checkout_git(arwen_checkout)
    if (
        source_after != source_before
        or authority_after != authority_before
        or _arwen_seam_identity(arwen_git_after)
        != _arwen_seam_identity(arwen_git_before)
    ):
        raise RuntimeError(
            "source, authority, or WOOF seam bytes changed during execution"
        )
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "claim": CLAIM,
        "nonclaims": list(NONCLAIMS),
        "source_release": SOURCE_RELEASE,
        "arwen_commit": ARWEN_COMMIT,
        "arwen_contract_document_sha256": ARWEN_CONTRACT_DOCUMENT_SHA256,
        "arwen_contract_surface_sha256": ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
        "arwen_git": {"before": arwen_git_before, "after": arwen_git_after},
        "arwen_checkout_unchanged": arwen_git_after == arwen_git_before,
        "execution_seconds": time.perf_counter() - started,
        "proof": proof,
        "sources_unchanged": True,
        "authorities_unchanged": True,
        "weather_plot_policy": "native Rust/WOOF renderer only; q2 ships in the history stream and its weather-field plots go through the same renderer",
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    receipt = output_root / RECEIPT_NAME
    _write_exclusive_json(receipt, payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": str(receipt),
                "receipt_sha256": sha256_file(receipt),
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
