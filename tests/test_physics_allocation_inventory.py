"""Can a physics scheme allocate its working set outside the arena again?

This is the ratchet that ``tests/test_mynn_pbl_scratch.py`` opened, and it
exists because MYNN got all the way to a production-width timing measurement
while allocating **54,595 bytes per column in 484 pool allocations per step**
that no part of the preflight knew about.  At the 360,000 columns of a d04
nest that is 18,743.8 MiB per step against an estimate that did not move at
all.  The failure mode is not slowness.  It is that ``run_alloc_preflight``
returns a comfortable headroom number for a configuration that will run the
card out of memory, on hardware with no ECC, where the observed symptom of
running out is corrupted output rather than an exception.

WHY IT IS A SEPARATE FILE
-------------------------
Every gate here reads source with :mod:`ast` and opens no device.  It used to
live in ``tests/test_mynn_pbl_scratch.py``, whose ``_carried_hash`` helper
imports cupy -- and ``tests/conftest.py`` marks a WHOLE module ``gpu`` when any
non-test function in it imports cupy, deliberately, because which tests call a
helper is not decidable from the AST.  So a purely static, fully deterministic
gate was skipped on every ``GPUWM_NO_LOCAL_GPU=1`` run, which is every routine
run on this machine.

That is not hypothetical.  At ``cb72667`` the ratchet was red in two places at
once and no CPU run could report either:

* four ``noahmp_*_gpu.py`` modules were committed with allocation sites and
  never entered below (``92216d0`` and earlier -- the triage note in
  ``docs/tree_triage_2026-07-26.md`` names them); and
* ``woof/core/ruc_gpu.py`` grew ``ruc_tanhf_glibc`` at ``dd0df4e``, a
  fourteenth site in a module recorded with thirteen -- which is the *count
  increase* this ratchet exists to catch, and which was invisible even to a
  GPU run, because the module-set assertion fired first and stopped the test
  before the counts were compared.

Both are fixed below: the file no longer needs a device, and the two
assertions now report together instead of one masking the other.

WHAT THE INVENTORY MEANS
------------------------
It is a debt register, not a waiver.  ``mynn_pbl_gpu``'s solver draws its
working set from ``MynnPblScratch`` and its row records only the DMP
sibling's four mixscalars buffers; everything else is
recorded at the count it has today and fails on any increase.  The distinction
that decides whether a recorded row is acceptable is the one YSU's entry
states: **an allocation the preflight knows about is a cost; an allocation it
does not is a hazard** -- and the property that separates MYNN's hazard from
every row below is whether the allocation scales with the nest.  For the
Noah-MP leaf batches that property is gated, not asserted, by
:func:`test_the_noahmp_leaf_batch_transients_are_bounded_by_the_column_batch`.
"""

from __future__ import annotations

import ast
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Raw device-allocation entry points.  ``asarray``/``ascontiguousarray`` are
#: deliberately absent: they are no-ops on an array that is already
#: contiguous float32, which is what every runtime input is, and they are the
#: only sanctioned way to accept a host array from an oracle fixture.  They
#: are covered instead by the churn measurement in the module docstring of
#: woof/core/mynn_pbl_scratch.py, which counts what actually gets allocated.
_ALLOCATORS = frozenset((
    "empty", "zeros", "ones", "full",
    "empty_like", "zeros_like", "ones_like", "full_like",
))


# Vertical value conversion owns one replacement for the dtype cast buffer.
# It changes the conversion's words, not its shape or preparation peak.
_VERTICAL_CAST_ALLOCATIONS = {"woof/ingest/vert.py": 1}


def test_vertical_value_cast_has_one_replacement_buffer():
    for name, expected in _VERTICAL_CAST_ALLOCATIONS.items():
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_device_field_float32")
        calls = [node for node in ast.walk(function)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr in _ALLOCATORS]
        assert len(calls) == expected

#: The two MYNN modules the arena lane converted.  Their entries in the
#: inventory below are the enumerated exceptions, and each one is a place with
#: no ``DomainState`` to draw from that allocates once rather than per step.
_CONVERTED = ("woof/core/mynn_pbl_gpu.py", "woof/core/mynn_pbl_runtime.py")

#: The four Noah-MP leaf batches on the forecast's own path.  Every one of
#: them is reached from ``noahmp_runtime.DEVICE_LEAF_BATCH``, which is what
#: ``LEAF_BATCH_EVALUATOR`` resolves to by default, so these run per leaf per
#: staging round of every Noah-MP timestep.  (``noahmp_fluxprep_gpu`` and
#: ``noahmp_leaves_gpu``, also recorded below, are reached only from the
#: oracle tests -- nothing in ``woof`` imports them.)
_NOAHMP_LEAF_BATCHES = (
    "woof/core/noahmp_bareflux_gpu.py",
    "woof/core/noahmp_radiation_gpu.py",
    "woof/core/noahmp_sflx_pre_gpu.py",
    "woof/core/noahmp_water_gpu.py",
)

#: The same packing with the column axis vectorised, which is the Noah-MP
#: conversion in flight.  Nothing under ``gpuwm/`` imports either yet -- only
#: their own tests do -- so they are not on the forecast path today, and the
#: ``COLUMN_BATCH`` bound that covers the four above therefore does not cover
#: these.  They are gated for the same shape property anyway, because the day
#: they are wired in is the day the bound has to already be true.
_NOAHMP_SLAB_BATCHES = (
    "woof/core/noahmp_flux_slab.py",
    "woof/core/noahmp_sflx_post_slab.py",
    "woof/core/noahmp_sflx_pre_slab.py",
    "woof/core/noahmp_thermal_slab.py",
    "woof/core/noahmp_water_slab.py",
)

#: Names that ARE a grid dimension.  Used for the slab packers, where the
#: stride constants are still being written -- one per leaf, `THERMOPROP_N_IN`,
#: `TSNOSOI_N_OUT` and six more so far -- so an exhaustive allowlist of them
#: would fire on every commit of the conversion for a reason that is not a
#: defect, and a gate that cries wolf gets deleted.  The essential half of
#: the property is checked either way: the shape must read the batch width.
#: The four modules already on the forecast path keep the strict form, where
#: the name set is settled and an unrecognised name is worth a failure.
_GRID_DIMENSION_NAMES = frozenset((
    "nx", "ny", "nz", "ncol", "ncols", "ncolumn", "ncolumns",
    "ims", "ime", "jms", "jme", "kms", "kme", "its", "ite", "jts", "jte",
))

#: Every name a Noah-MP leaf batch may size a device allocation from.  The
#: whole content of the claim is that ``count``/``n`` is ``len(calls)`` and
#: nothing here is a grid dimension: with the staging loop flushing at
#: ``COLUMN_BATCH``, that makes every allocation below bounded by the batch
#: rather than by the nest.  The rest are per-column strides -- ``nfull`` is
#: ``nsnow + nsoil``, and the three tuples are output-field lists.
_BATCH_WIDTH_NAMES = frozenset(("count", "n"))
_PER_COLUMN_STRIDE_NAMES = frozenset((
    "nfull", "N_OUTPUT", "OUT_STRIDE", "_ATM_OUT", "_PHEN_OUT", "_PRCP_OUT",
    # the slab packers' spellings of the same thing
    "_NFULL", "P_STRIDE", "IN_STRIDE", "INT_STRIDE",
    # ``_batch._ATM_IN``, ``_vege.N_INPUT``, ``_bare.N_OUTPUT``: the slab
    # packers import the per-column packer's layout constants rather than
    # restating them -- deliberately, so the two cannot drift in slot order --
    # so the module handle is what the shape expression reads.
    "_batch", "_vege", "_bare",
))

#: Every ``cp.<allocator>`` call site in every physics GPU module, by the
#: function that owns it.  Written out rather than summarised so that adding
#: one is a visible diff on this line, and so that the modules nobody has
#: converted yet are an itemised debt rather than an unexamined blank.
#:
#: ``mynn_pbl_gpu`` records only the DMP sibling's four mixscalars
#: buffers: the solver itself still draws its whole working set from
#: ``MynnPblScratch``.  ``mynn_pbl_runtime`` keeps exactly two, both
#: justified above their call sites -- the six returned tendency fields on
#: the no-``DomainState`` path used by the standalone harnesses, and the
#: 256-byte validity words, built once per process.
_PHYSICS_ALLOCATION_INVENTORY = {
    # Fitch wind farms (labs/windfarm, 2.9.11), allocated once when farms
    # attach: per-turbine tables (nt x 10 hub, nt x 7 wake, the turbine,
    # curve, blade and polar rows, sized by the turbine count) and one
    # (ny, nx) map-factor plane plus one zero tendency field of the domain
    # shape; the attach entry points stage the turbine list once.
    'woof/core/wind_farm.py': {'__init__': 9, 'attach_wind_farm': 1,
                                'attach_voronoi_wind_farm': 1},
    # HRRR output windows use declared persistent scratch only.
    'woof/core/climate_diagnostics.py': {},
    'woof/core/output_diagnostics.py': {},
    # Streamed history finalization owns one replacement LTG3 plane,
    # 4*ny*nx bytes, beside the assembled history planes.
    'woof/core/gsd_diagnostics.py': {'finalize_history_fields': 1},
    # Initial GSD radar: one float32, float64 and bool column arena.
    # 87*(nz+1)*tile_threads bytes; bounded by SMs, once per initial frame,
    # priced by preflight.gsd_initial_reflectivity_workspace_bytes and
    # checked against the launcher's sizing in test_gsd_initial_reflectivity.
    'woof/core/gsd_initial_reflectivity.py': {'launch': 3},
    "woof/core/lin.py": {},  # Persistent DomainState scratch, no raw allocation sites.
    # Integer-define launchers are part of the same kernel family. Chem
    # vertical mixing and IEVA draw only declared caller/state buffers;
    # their empty rows reject any future raw allocation.  The one vertmx
    # site is the HRRR-fork smoke dry-deposition velocity (aer_drydep_opt =
    # 111 only): one (ny, nx) float32 surface plane per chemistry vertmx
    # call, 4 bytes per column, owned by the caller for that call and back
    # in the pool when it returns.  It is a surface plane, not a column
    # working set, so it does not scale with the nest's levels.
    'woof/core/chem_vertmx.py': {'smoke_deposition_velocity': 1},
    # lane/hrrr-v4-output: one history-time copy per finalized GSD field
    # (finalize_history_fields), never per step; the output-diagnostics
    # launchers draw only state carriers.
    'woof/core/gsd_diagnostics.py': {'finalize_history_fields': 1},
    'woof/core/output_diagnostics.py': {},
    'woof/core/ieva.py': {},
    # The WRF v4.4 Thompson column (thompson_version = "wrf_44") draws its
    # working fields from state.scratch slots and its coefficient tables
    # from the per-device table owner; the empty row rejects any future raw
    # allocation in its step.
    'woof/core/thompson_column.py': {},
    # Public Jacobi returns two fresh caller-owned arrays and reduces one
    # transient status vector. The CPU witness below checks actual shapes,
    # bytes, input-copy ownership and repeat-call independence.
    # _launch_global (71b192639, ensembles above 64 members): one reused
    # slab per call for the global tier, bounded by GLOBAL_SCRATCH_BYTES
    # (or GLOBAL_MIN_IN_FLIGHT matrices) and freed on return.
    'woof/core/jacobi_eigh.py': {'batched_eigh': 3, '_launch_global': 1},
    # The live GSD/precipitation steps use DomainState.scratch; their
    # scratch_shapes feed preflight.scratch_slot_registry (no raw sites).
    'woof/core/output_diagnostics.py': {},
    # Standalone assembled-frame recomputation (not called by a forecast
    # route) stages at most three float32 horizontal planes: two host
    # carriers and one output, plus CuPy reduction scalars/workspace.
    # The raw output allocation belongs to this helper, not a step arena.
    'woof/core/gsd_diagnostics.py': {'finalize_history_fields': 1},
    # SASE's pool-backed closure is priced by the two live-set phases in
    # preflight.sase_workspace_phases, including the driver-held fields.
    # The CPU allocation/lifetime witness executes its actual step with
    # numerical kernels stubbed and compares peak bytes to that bound.
    'woof/core/sase.py': {
        'launch_box_filter': 1, 'launch_structure_functions': 1,
        '_z_stencil': 1, '_thickness_args': 1, 'launch_strain': 1,
        'launch_germano_lift': 2, 'launch_model_stress': 1,
        'launch_governed_stress': 3, 'launch_dynamic_solve': 2,
        'launch_bulk_richardson_zi': 1, 'launch_w_sensor_moments': 1,
        'launch_vertical_channel': 2, 'launch_blackadar_length': 1,
        'launch_sase_step': 6, 'launch_n2': 1, 'launch_moist_n2': 1,
        'launch_plume_vent_flux': 3, 'launch_vent_deposit_scale': 1,
        'launch_scalar_mix': 2,
    },
    # Active parameter-table scaling is reached from initialize_physics,
    # never the timestep driver. The validated registry selects at most
    # 61 unique cells, independent of domain dimensions. Two explicit
    # allocations hold outputs/status (5 bytes per cell); three asarray
    # uploads hold binary64 values/factors and crop flags (17 bytes per
    # cell). Bound: 1,342 raw bytes per initialization, returned to the
    # reusable device pool on return.
    # test_parameter_scaler_has_a_registry_bound_and_an_initialization_owner
    # pins that bound and the initialization-only importer.
    "woof/core/physics_param_gpu.py": {
        "scale_physics_param_values": 2,
    },
    # CLM lake columns (sf_lake_physics = 1): per-lake-column state and
    # refresh scratch, bounded by the lake-column count.
    "woof/core/lake.py": {
        # lane/cap-lake (f9eb12244a9): one (7, ncolumns) float32 staging
        # block for the table-driven lake analysis fields, filled once at
        # initialization and bound by lake_bind_source_columns (28 B per
        # lake column).
        "initialize_lake": 7,
        "refresh_columns": 6,
        # c4ef64622 moved the CLM lake column storage out of the per-thread
        # local frame into one device arena per lake model: at most
        # LAKE_LAUNCH_COLUMNS * LAKE_ARENA_SLOTS * 8 bytes (0.31 GiB, less
        # for fewer lake columns), reused while its size holds and priced as
        # lake/arena.  Entered here, in the gate sweep, because that commit
        # did not enter its own row.
        "lake_arena": 1,
    },
    # Preparation seeds: candidate numbers span the grid, while rounded alt
    # and gathered scratch use at most 112 bytes per cell in one bounded chunk.
    "woof/ingest/closure_device.py": {
        "temperature": 2,
        "gathered_numbers": 2,
        "_rounded_alt": 1,
        "_closure_chunk": 4,
        "thompson_cold_start_moment_closure": 2,
    },
    'woof/core/acoustic.py': {},
    # The chem program (lane/aq-core).  chem_bdy and chem_outputs allocate
    # nothing through a cupy allocator (their launch tables are cp.asarray
    # uploads of a handful of pointers).  chem_driver's three sites:
    # ``_context`` builds the (rows, ny, nx) float32 deposition-velocity
    # plane ONCE per chem domain and reuses it every step; ``_masses``
    # allocates the (fields, nz*ny) float64 row partials of one ledger
    # reduction (8 * nrows * nz * ny bytes: 1.7 MiB for 8 rows at
    # 896 x 512 x 59) and a zero-length result for an empty row set.
    'woof/core/chem_bdy.py': {},
    'woof/core/chem_driver.py': {
        '_masses': 2,
        '_context': 1,
    },
    'woof/core/chem_outputs.py': {},
    # chem_prep's ``refresh`` allocates its thirteen met fields ONCE per
    # chem domain (the ``shape`` check keeps them on every later step);
    # the device inventory prices them as ``chemprep_<field>``
    # (woof/core/device_inventory.py:chem_state_array_shapes).
    'woof/core/chem_prep.py': {'refresh': 1},
    # The smoke set (lane/aq-smoke).  The fire process and the large-scale
    # washout allocate nothing through a cupy allocator: every array they
    # write is one of their ChemAllocations (DomainState allocates them and
    # device_inventory.chem_state_array_shapes prices them), and their host
    # frames are uploaded into those.  The Freitas plume unit
    # (woof/core/chem_plumerise.py) is not on this list because it compiles
    # through its own unflushed NVRTC site (chem_plumerise_cache) rather than
    # get_kernel.  Its launch allocates per plume call, every
    # plumerisefire_frq minutes, never per step: a (12, ny, nx) float32
    # property plane (20 MiB at 896 x 512), per-fire-column outputs (about
    # 1.5 KiB per fire column at 59 levels) and the bounded solver workspace,
    # PLUME_BATCH_COLUMNS x PLUME_WORKSPACE_WORDS float32 = 217 MB.  The
    # preflight does not price that transient yet: recorded as debt.
    # trailing_planes (ea9faa63c, the HRRR-Smoke fire timing): once per run,
    # the trailing window's mass and FRP sums (float64) and burning-hour
    # count (int32) on the fire source's 2-D grid, 20 bytes per source cell,
    # cached on the chem state and never reallocated per step.
    # Missing source archives take a mutually exclusive branch: one
    # float32 zero plane shared by mass/FRP (4 B/cell), below the existing
    # 20 B/cell bound; this adds a site without increasing the peak.
    'woof/core/chem_fire.py': {
        "trailing_planes": 4,
    },
    'woof/core/chem_wetdep.py': {},
    # WRF's monotonic final stage.  The forecast route hands it the six
    # scratch arrays, the zero implicit flux, mub and the map factors from
    # DomainState.scratch (woof/core/chem_transport.py), so on that route
    # the only site reached is ``launch_mono_fluxes``'s unit-map plane on a
    # domain with no map factors, one (ny, nx) plane per row launch.  The
    # other sites serve the oracle tests' standalone calls: ``_plane``
    # broadcasts a scalar mass, and the launchers allocate the scratch, the
    # zero implicit flux and a zero base mass when the caller supplies none.
    'woof/core/chem_advect_mono.py': {
        '_plane': 1,
        'launch_mono_fluxes': 3,
        'launch_mono_renorm_apply': 1,
    },
    # GOCART-lite (lane/aq-gocart).  Every launcher below takes its outputs
    # and workspaces from the caller and allocates only when handed None,
    # which the chem processes never do on the forecast path: their arrays
    # are the processes' ChemAllocations, which DomainState allocates and
    # device_inventory.chem_state_array_shapes prices.  The exceptions:
    # * chem_ageing ``launch_ageing``: the aging process hands None on the
    #   first chem step and keeps the returned REAL*8 workspace on the chem
    #   state, (3, participants, ny, nx) float64 = 96 B per column for the
    #   four GOCART carbon rows, 42 MiB at 896 x 512, once per domain.
    # * chem_dust ``step``: dust_opt = 3 (AFWA) only, once per domain: the
    #   WRF-optional DRI/NGA sentinels (four float32 2-D planes) and the
    #   scheme's TOT_DUST/VIS_DUST working columns (two float32 3-D fields,
    #   8 B per cell, 207 MiB at 59 x 512 x 896).  NOT priced by the
    #   preflight: a cost it does not know about, recorded as debt.
    # * chem_seasalt ``prepare_seasalt_table``: the sub-bin offsets and
    #   REAL*8 table, at most 100,000 words (refused above that), once per
    #   distinct dt.
    # * chem_optics ``gocart_optics``: each optical output the caller names
    #   and does not hand in, the moments and diagnostics only when asked,
    #   and the (ny, nx) int32 refractive-index mask every call.  The
    #   optics process hands in tauaer, EXTCOF55 and AOD5502D and asks for
    #   nothing else, so on the forecast path only the mask is allocated,
    #   once per radiation interval.  ``opt_out`` and ``rrtmg_sw_bands``
    #   allocate only when handed no outputs (parity tests; the radiation
    #   adapter, when wired, hands its own).  ``step``: the process's four
    #   SW wavelengths of tau, (4, nz, ny, nx) float32 = 16 B per cell,
    #   432 MiB at 59 x 512 x 896, once per domain.
    # * chem_rrtmgp_aerosol ``_outputs``: the column lookup's four
    #   (ncol, nz, nbands) outputs and its (nz, ny, nx) type mask, per call.
    #   Reached only from the parity tests today; the radiation adapter that
    #   wires it must draw these from the radiation workspace instead.
    'woof/core/chem_ageing.py': {
        'launch_ageing': 1,
    },
    'woof/core/chem_drydep_aerosol.py': {
        'launch_drydep': 1,
    },
    'woof/core/chem_dust.py': {
        'step': 5,
    },
    'woof/core/chem_inventory.py': {},
    'woof/core/chem_mp_coupling.py': {},
    'woof/core/chem_optics.py': {
        'gocart_optics': 6,
        'opt_out': 2,
        'rrtmg_sw_bands': 2,
        # lane/cap-aer-optics (177a585dd32) moved the held four-wavelength
        # tauaer work buffer (16 B per cell, once per domain) from step into
        # update_diagnostics, which the live radiation call also runs.
        'update_diagnostics': 1,
    },
    # lane/cap-aer-optics (177a585dd32): live chemistry aerosol optics into
    # both radiation engines.  Per radiation call, sized by the radiation
    # column batch: RRTMG SW three (nc, 14, nlay) band planes or one LW
    # (nc, nlay, 16); RRTMGP three (nc, nlay, nbands); finalize one to three
    # g-point arrays shaped like the gas optical depth it mixes into.
    'woof/core/chem_radiation.py': {
        'legacy': 2,
        'modern': 1,
        'finalize': 1,
    },
    'woof/core/chem_rrtmgp_aerosol.py': {
        '_outputs': 2,
    },
    'woof/core/chem_seasalt.py': {
        'prepare_seasalt_table': 2,
    },
    'woof/core/chem_settling.py': {
        'launch_settling': 2,
    },
    'woof/core/chem_sulfur.py': {
        'launch_solar_init': 2,
        'launch_solar_step': 1,
    },
    # lane/aq-cams.  The Wesely process's arrays are ChemAllocations the
    # driver owns (the launch output, one (ny, nx) plane per Wesely row, and
    # a zero plane); the one site left is the launcher's own output when a
    # caller passes no buffer (the oracle test), (nrows, ny, nx) float32.
    'woof/core/chem_drydep_gas.py': {
        'wesely_ddvel': 1,
    },
    'woof/core/advection.py': {
        '_launch': 1,
        'advect_scalar_rk3_periodic_test': 7,
    },
    # lane/282-bw-glue (ad13a2f37): ``total_theta`` builds thb + thp, the
    # theta each RK stage transports, in one four-word launch.  Its
    # ``cp.empty_like`` is the temporary the eager ``state.total_theta()``
    # (``thb + thp``, woof/core/state.py) has always drawn from CuPy's pool
    # on every call, through an operator this scanner cannot see; the lane
    # kept it on purpose ("Keep the original total_theta temporary's
    # lifetime and pool reuse", at the call), so the forecast's per-stage
    # allocations and its peak are what they were before 2.8.2.  Bound: one
    # (nz, ny, nx) float32 per call, back in the pool when the stage drops
    # it.  An unsupported layout falls back to ``state.total_theta()``.
    'woof/core/bandwidth_glue.py': {
        'total_theta': 1,
    },
    # The opt-in WRF-exact diagnostics (270038669, lane/282-wrf-exact):
    # ``update_diagnostics`` allocates ``state.p_perturbation``, one
    # (nz, ny, nx) float32 field, the first time it runs on a state that
    # has none, and reuses it after.  The line runs only when
    # GPUWM_WRF_EXACT=1 and WOOF_WRF_EXACT_DIAGNOSTICS=1 select WRF's
    # perturbation-pressure word (woof.wrf_exact.DIAGNOSTICS_ENABLED), so
    # a default forecast never reaches it, and a wrfinput start under the
    # selector seeds the field from P (woof/ingest/wrfinput.py) before any
    # step.  Bound: 4 bytes per cell, once per state, never per step.
    'woof/core/diagnostics.py': {
        # The carried-MUTS and ordinary strict arms are mutually exclusive.
        # Either creates the same p_perturbation field once per state when
        # absent, bounded by 4 bytes per cell; the first arm returns.
        'update_diagnostics': 2,
    },
    # THE STREAMED SAFETY FOLD's two buffers, from feat-safety-observers.
    # ``StreamedStability.__init__`` allocates the per-tile partial record
    # array (ntiles * 256 blocks x 9 floats) and the 8-word folded result,
    # both ONCE per streamed domain and neither per step -- which is the
    # whole point: the reduction it replaces was a whole-domain
    # ``stability_report`` issued every dynamics substep over a DomainState
    # the sweep never writes.  Bound: 9 KiB per tile plus 32 B, so 144 KiB
    # at 16 tiles, against a store measured in tens of GiB.
    'woof/core/streaming.py': {
        # Ranked health adds one partial record allocation site per card.
        '__init__': 3,
    },
    'woof/core/diffusion.py': {
        'diffuse_only_test': 1,
    },
    # ``__init__`` and ``reset`` are ``MassFluxAccumulator``'s FP64 running
    # sum, committed unlisted at 90f02a89 and entered 2026-07-31.  Each is one
    # ``cp.zeros((), dtype=cp.float64)``: zero-dimensional, 8 bytes, 512 in a
    # pool block.  It sits outside a workspace because there is no workspace
    # to sit in -- the accumulator is a diagnostic object the CALLER
    # constructs and hands to ``step``, and ``step`` never constructs one, so
    # no forecast path can reach either site per step, per domain or per
    # output interval.  (Nothing under ``gpuwm/`` constructs one at all today;
    # the live mass-flux caller, ``verify/cases/wk82.py``, takes the
    # host-syncing list observer instead.  ``reset`` has no caller anywhere.)
    #
    # The bound is one 8-byte scalar per accumulator object, and what makes
    # that the whole cost is ``add`` -- the only method the acoustic loop
    # touches, once per substep of every final RK stage.  It accumulates with
    # ``self._total += increment``, cupy's in-place ``__iadd__``, which
    # rebinds nothing.  Measured rather than assumed: 2,048 device-scalar adds
    # and 2,048 host-float adds each cost zero device mallocs and leave
    # ``data.ptr`` unchanged, and an 8-step integration leaves ``_total`` the
    # same object it was constructed as.  ``tests/test_mapped_mass_closure.py:
    # test_the_mass_flux_accumulator_allocates_once_and_never_per_substep``
    # is that measurement, and it carries the out-of-place spelling of ``add``
    # as its negative control -- one malloc per add, which is what this row
    # would be recording if the increment were folded the other way.
    'woof/core/dycore.py': {
        '__init__': 1,
        # One cached uint32 (32768, 36) CFL ring per grid, reused across
        # steps, tiles and state replacements.  cfl_inventory binds its
        # 4,718,592 bytes to estimate_domain and the tile planner; the
        # resident/per-grid charge and reuse are gated by test_cfl_memory.
        '_wrf_cfl_buffer': 1,
        'launch_coriolis_curvature': 2,
        # The identity map factors for a caller that omits one: msfu and
        # msfv, and msft since the compiled-WRF diffusion oracle
        # (83fde6032) made diff6 multiply by the field's own map factor.
        # Each is a ones array built only when its argument is None.  Both
        # forecast callers in this module pass state.msfu, state.msfv and
        # state.msft, which DomainState allocates once, so no forecast step
        # reaches them; the direct-launch tests (tests/test_diff6*.py) do.
        # Bound: (ny, nx + 1), (ny + 1, nx) and (ny, nx) float32 per such
        # call, freed on return.
        'launch_diff6': 3,
        # The NOAA WRFV3.9 fork's edge-to-edge sixth-order filter
        # (75c95b9dd, diff_6th_form = "noaa_wrf39", specified or nested
        # domains).  NOT opt-in: request-defaults.v1.toml makes it the
        # request default of every HRRR recipe source (hrrr, hrrr-native,
        # hrrr-prs) and of hrrr_wrf.nl namelists, so it runs on the
        # production HRRR route.  launch_diff6_to_edge itself allocates
        # nothing since 2.8.6 (lane cut286-fu-diff6-price): its padded
        # field, padded tendency, four padded planes and slope-taper
        # padded base geopotential are the diff6_edge_* slots of
        # preflight.scratch_slot_registry (shapes in
        # woof/core/diff6_edge_workspace.py), drawn through
        # DomainState.scratch by _diff6_edge_work and priced by the run
        # preflight and every ensemble plan that reads the registry.  The
        # one site left is the fallback for a bare-array launch with no
        # DomainState: one flat float32 buffer of two padded fields, the
        # padded planes and (slope taper only) the padded geopotential,
        # freed on return.  Only the verification callers
        # (tests/test_diff6_fork_form.py, tools/wrf_diffusion_oracle/
        # diff6_fork_oracle.py) reach it; both forecast callers
        # (prepare_fixed_tendencies, apply_diff6) pass the workspace.
        '_diff6_edge_transient_work': 1,
        'reset': 1,
    },
    # Grell-Freitas.  Scanned since 5ef5d7dba (lane speed-default-pieces),
    # whose one-pass output validation is the module's first get_kernel
    # call; the nine sites predate it and the lane added none.  All are
    # per-call transients of GF's packed GFDRV call, each scaling with the
    # column count: the rthraten sum when no driver holds the radiation
    # rates ((nz, ny, nx) float32), the zeros lane for absent auxiliary
    # forcing ((ncol, nz)), the packed inputs lvin (ncol, 15, nz), scin
    # (ncol, 9) and iin (ncol, 3) int32, the outputs lev (ncol, 16, nz),
    # sca (ncol, 9) and isc (ncol, 6) int32, and the tile workspace ws,
    # gf_workspace_floats(nz, tile) * 4 bytes, priced by
    # preflight.gf_column_workspace_bytes.  Bound: at most 132 B per cell
    # plus 108 B per column, freed when the call returns.
    # With a per-column active mask (driver gf_active_column, lane/gf-column-mask
    # 2b923cb282) the packed inputs and outputs above hold only the n launched
    # columns, and three more transients scatter them back to every column:
    # full_lev (ncol, 16, nz) and full_sca (ncol, 9) float32, full_isc (ncol, 6)
    # int32, zero where the scheme does not run.  Masked bound: at most 132 B
    # per launched cell plus 108 B per launched column, plus 64 B per cell and
    # 60 B per column, freed when the call returns; the unmasked call is unchanged.
    'woof/core/gf.py': {
        '__call__': 12,
    },
    'woof/core/health.py': {},
    # ``w0avg`` moved out of ``update_trigger_history`` and into
    # ``ensure_trigger_history`` when [tiles] landed.  The count is the same
    # one ``cp.zeros`` and the bound is unchanged -- (nz, ny, nx) float32, one
    # per KainFritsch object -- but WHERE it happens is the whole point of the
    # move: allocated on the first due cumulus call, the array joined the
    # carrier set twenty steps into a run, and a seam plan built once from the
    # inventory at construction cannot follow a carrier set that changes
    # identity mid-run.  ``ensure_trigger_history`` is idempotent and
    # construction-time, so the inventory is settled before the first step.
    # ``update_trigger_history`` keeps a ``cp.ascontiguousarray`` of the
    # half-level mean, which is not in ``_ALLOCATORS`` and never was.
    'woof/core/kf.py': {
        'ensure_trigger_history': 1,
        # The WRF 4.6.1 port (2026-10-07): the 3-D outputs (one
        # comprehension), the 2-D outputs (one loop), the optional cloud
        # diagnostics (one loop) or their 1-word placeholder, and the
        # caller-sized column workspace the frame move added (48ff6b813).
        # Every array is (nz, ny, nx) or (ny, nx) float32 except the
        # workspace, priced by preflight.kf_column_workspace_bytes.
        'launch_kf': 5,
    },
    'woof/core/microphysics.py': {},
    'woof/core/microphysics_transition.py': {},
    'woof/core/moist.py': {
        'launch_pd_fluxes': 1,
        'launch_pd_renorm_apply': 1,
    },
    # Zero, and structurally so: this module selects which compiled variant
    # of wrf_calc_n2 the launcher fetches (spec 3.3 mutation control) and
    # touches no array.  It imports cupy nowhere -- the row is entered in
    # the commit that adds the module, per the rule this file states,
    # rather than in a later sweep.
    'woof/core/moist_n2_mutation.py': {},
    # Classic Thompson (mp=8, lane/parity-mp-libm).  Zero: every working
    # array of the unit is a named DomainState.scratch slot apply_classic
    # draws (priced by preflight from thompson_classic's own slot lists).
    'woof/core/thompson_classic.py': {},
    # Milbrandt-Yau (mp_physics=9).  Zero, and deliberately so: every one of
    # the thirteen scratch volumes the six kernels pass to one another is a
    # named ``DomainState.scratch`` slot the adapter owns, so the allocation
    # gate budgets them and nothing is created per step.  The module does hold
    # ONE device buffer the scanner cannot see -- ``_constants_device()``
    # uploads the 154-entry FP32 constant table with ``cp.asarray``, which is
    # not in ``_ALLOCATORS``.  It is 616 bytes, module-global, guarded by a
    # None check and therefore once per PROCESS, not once per step, per domain
    # or per call; ``launch_milbrandt2`` reads it and never rebinds it.
    'woof/core/milbrandt2.py': {},
    'woof/core/morrison.py': {
        # The two-moment launcher's standalone fallbacks: the four radii
        # (one cp.empty_like comprehension), the WORK stack and the LTRUE
        # column flag, each only when the caller hands none.  apply() hands
        # the morr_work / morr_ltrue scratch slots, so a forecast step
        # allocates nothing here.
        'launch_morrison': 3,
    },
    # MYJ (the 2/2 pair).  Entered here by the commit that closed the MYJ
    # port's review -- the port itself added the two modules and never
    # entered them, which is exactly the "appeared or vanished" case this
    # scanner exists to catch, and it caught it.
    #
    # The PBL launcher has YSU's shape: four allocation sites building the
    # eleven per-call outputs (one cp.empty comprehension for the eight 3-D
    # fields, three cp.empty for pblh/kpbl/mixht).  They are PRICED, which
    # is what makes them a cost rather than a hazard --
    # woof/core/preflight.py's myj_output_transient_shapes enumerates
    # every one of them by name, on the ysu_output_transient_shapes
    # contract.  That enumeration is a SECOND copy of the launcher's
    # roster -- preflight cannot import a CuPy-importing module -- so it
    # is held to the first copy by tests/test_myj_port.py::
    # test_preflights_myj_output_roster_matches_the_launchers.  Without
    # that gate this row would price a roster nothing checks, which is
    # the state it shipped in until the port's re-verification.
    'woof/core/myjpbl.py': {
        'myj_pbl_step': 4,
    },
    # The Eta surface layer allocates NOTHING per call: it writes into the
    # driver's own persistent surface fields, so its row is empty and must
    # stay empty.
    'woof/core/myjsfc.py': {},
    # No longer zero: the mixscalars landing (4a0bb3f69) gave the DMP
    # sibling kernel four working buffers -- up_a_pre (ncol, nz+1, nup)
    # float32, psig_w (ncol,), plume_active (ncol,) int32 and
    # limiter_adjustment (ncol,) -- allocated once per DMP call on the
    # bl_mynn_mixscalars=1 lane only.  Recorded here in the sweep that
    # found the lane had not entered its own row; the bounds are the
    # shapes above, all ncol-scaled, none nest-persistent.
    'woof/core/mynn_pbl_gpu.py': {
        'mynn_dmp_mf_cuda': 4,
    },
    'woof/core/mynn_pbl_runtime.py': {
        # One growable int32 status block per (device, stream), not per step.
        '_validity_flags': 1,
        # One became three with the mixscalars landing (4a0bb3f69): the
        # qn-family output planes (five (nz, ny, nx) fields on the
        # bl_mynn_mixscalars=1 lane) and the qnbca zero column mp=28's
        # Registry package declares absent.  Entered in the same sweep as
        # the mynn_pbl_gpu row; ncol/plane-scaled, none nest-persistent.
        # scalar_pblmix adds one reused chunk work array (5*nz+1 floats
        # per column), never a full-domain tridiagonal workspace. The four
        # qn output planes reuse the existing allocation site above.
        # preflight.mynn_scalar_transient_shapes prices the full live peak;
        # omitting it would admit a run without space for its scalar rates.
        'mynn_pbl_step': 4,
    },
    # Born with the mixscalars landing (4a0bb3f69) and entered in the same
    # sweep as the mynn_pbl_gpu row above: the qn-family tridiagonal
    # solves (three per-call buffers) and the plume-edge flux columns
    # (two).  All (ncol,)- or (ncol, nz)-scaled, none nest-persistent.
    'woof/core/mynn_scalar_mix_gpu.py': {
        'mynn_dmp_qn_flux_columns_cuda': 2,
        'mynn_mix_scalar_columns_cuda': 3,
        # Two chunk output buffers plus an optional standalone work buffer.
        # Runtime passes its preallocated chunk work. No whole-domain solve
        # buffer is permitted to hide behind this bounded-column entry.
        'scalar_pblmix_columns_cuda': 3,
    },
    # The RUC soil-tier module allocates nothing; its columns come from
    # the caller's batch.
    'woof/core/ruc_tier.py': {},
    'woof/core/mynn_sfclay.py': {
        '_allocate_result': 1,
        '_surface_array': 1,
    },
    # ``_urban_args``' two placeholder sites (one-word int32/float32 arrays
    # bound to the urban pointer slots when no urban model ran) are gone: the
    # default path launches noah_column, which takes no urban argument, so
    # nothing is bound when no urban model runs.
    # launch_noah_seaice (lane/parity-noah, WRF's seaice_noah): one int32
    # word per launch for SFLX_SEAICE's fatal flag, read back after the
    # kernel to refuse an impossible snowpack by WRF's words.  One element,
    # never grid-scaled, never nest-persistent.
    'woof/core/noah.py': {
        'launch_noah_seaice': 1,
    },
    # ---- the four Noah-MP leaf batches -----------------------------------
    # Entered 2026-07-27, having been committed unlisted since at least
    # 92216d0.  They were checked against the question that decides whether a
    # row is a cost or a hazard, and the answer is that they are costs:
    #
    #   * every shape is ``len(calls)`` by a fixed per-column stride, gated by
    #     ``test_the_noahmp_leaf_batch_transients_are_bounded_by_the_
    #     column_batch`` below -- no grid dimension appears in any; and
    #   * ``noahmp_runtime.noahmp_lsm_step`` flushes its staging list at
    #     ``COLUMN_BATCH``, so ``len(calls)`` never exceeds 2,048 whatever the
    #     nest is.
    #
    # Derived at that width, in float32: bare_flux ``count*13``, radiation
    # ``count*19``, water ``count*73 + count`` int32, sflx_pre
    # ``n*(17+1+7+1+1+6+16)`` = ``49n``.  620 B per staged column, so 1.21 MiB
    # for all four together at COLUMN_BATCH -- against MYNN's 18,743.8 MiB,
    # which is the comparison that makes this bookkeeping rather than the
    # defect that caused this file to exist.
    #
    # It is still 1.21 MiB the preflight does not price: ``woof/core/
    # preflight.py`` has no Noah-MP transient term at all (its Noah-MP rows
    # are kernel local-memory frames and the snow stack).  That is recorded
    # here and handed over rather than fixed here, because pricing it moves
    # numbers that tests/test_preflight.py pins.
    'woof/core/noahmp_bareflux_gpu.py': {
        'evaluate_bare_flux_calls': 1,
    },
    'woof/core/noahmp_fluxprep_gpu.py': {
        'evaluate_fluxprep_leaf': 1,
    },
    'woof/core/noahmp_leaves_gpu.py': {
        'evaluate_leaf': 1,
    },
    'woof/core/noahmp_radiation_gpu.py': {
        'evaluate_radiation_calls': 1,
    },
    'woof/core/noahmp_sflx_pre_gpu.py': {
        'evaluate_sflx_pre_calls': 7,
    },
    # ---- and the same packing with the column axis vectorised ------------
    # Entered at the counts they had when they were committed (04a40cc), and
    # the Noah-MP conversion is live: expect this to move, and expect the
    # ratchet to say so, which is the point of it.
    #
    # Nothing under gpuwm/ imports either yet, so neither is on a forecast
    # path and neither inherits the COLUMN_BATCH bound.  WHEN THEY ARE WIRED
    # IN, WHATEVER CALLS THEM MUST CARRY A BOUND.  A slab that takes every
    # land column at once prices, at d04's 360,000 columns and in float32:
    # water pack (26 + 76 + 9) B/col plus evaluate (73 + 1) B/col = 185 B/col
    # = 63.5 MiB, and sflx_pre (17 + 17 + 1 + 7 + 1 + 1 + 6 + 16 + 1) = 268
    # B/col = 92.0 MiB.  155 MiB is survivable against the 29,500 MiB rail
    # and it is 128x what the staged path costs, so it is a number that has
    # to be decided rather than discovered.
    'woof/core/noahmp_flux_slab.py': {
        'evaluate_bare_flux_slab': 2,
        'evaluate_vege_flux_slab': 2,
        # b3dd58fc9f3 (2.9.3): the fixed-width masked dispatch keeps the
        # vegetated-column selection on the device.  Its two sites are the
        # sibling's two (the n == 0 empty and the (n, N_OUTPUT) slab); the
        # live mask never sizes a host allocation or a launch.
        'evaluate_vege_flux_masked_slab': 2,
        'pack_bare_flux_slab': 2,
        'pack_vege_flux_slab': 2,
    },
    'woof/core/noahmp_sflx_post_slab.py': {
        '_error_slab': 5,
        'sflx_post_seg2': 1,
    },
    'woof/core/noahmp_sflx_pre_slab.py': {
        'evaluate_sflx_pre_slab': 9,
    },
    'woof/core/noahmp_thermal_slab.py': {
        '_integers': 1,
        'evaluate_phasechange_slab': 1,
        'evaluate_radiation_slab': 1,
        'evaluate_thermoprop_slab': 1,
        'evaluate_tsnosoi_slab': 1,
        'pack_phasechange_slab': 1,
        'pack_radiation_slab': 1,
        'pack_thermoprop_slab': 1,
        'pack_tsnosoi_slab': 1,
    },
    'woof/core/noahmp_water_gpu.py': {
        'evaluate_water_calls': 2,
    },
    'woof/core/noahmp_water_slab.py': {
        'evaluate_water_slab': 2,
        'pack_water_slab': 3,
    },
    'woof/core/nssl2.py': {},
    'woof/core/nssl2_diagnostics.py': {},
    # _cached_velocity_gamma_table (c7abbdc55, lane speed-two-moment): the
    # graupel and hail fall-speed gamma table the device fills once, (2, 120)
    # float64 = 1,920 B, cached per device for the life of the process
    # (lru_cache), so it is allocated once and never per step.  It replaces
    # tgamma calls the sedimentation kernel made on every call.
    'woof/core/nssl2_driver_support.py': {
        '_cached_velocity_gamma_table': 1,
        'gather_initialize_and_sediment': 4,
    },
    'woof/core/nssl2_fused_gs.py': {},
    'woof/core/nssl2_nucond.py': {
        'launch_nucond': 1,
    },
    'woof/core/nssl2_qvexcess.py': {},
    'woof/core/nssl2_radiation.py': {},
    # analysis_refl_10cm: under GPUWM_WRF_EXACT=1 the resident tick-zero
    # frame is WRF's all-zero array, one caller-owned FP32 mass volume.
    # scratch (18dbe1d5449, the analysis-frame diagnostic): a host/store
    # analysis frame stages bounded 64-row horizontal slabs on the GPU, one
    # FP32 slab per scratch slot, dropped when the slab is done; once per
    # domain start, never per step.
    # launch_refl10cm_wsm6: the WSM6 reflectivity column workspace (lane/292-mem-cuts 052440f32b4), three
    # per-call arrays (5 float32, 4 float64 and 1 uint8 rows per level) sized for the columns in flight, one
    # tile (wsm6_tile_columns: the card's resident capacity), returned to the pool when the call ends.
    'woof/core/refl.py': {'analysis_refl_10cm': 1, 'launch_refl10cm_wsm6': 3,
                           'scratch': 1},
    # WRF v4.7.1 slope_rad / topo_shading (lane 281-namelist-gaps), only on a
    # domain that turns slope_rad on.  ``__init__`` allocates the carrier's
    # held planes once per domain -- diffuse_frac, topo_coszen, hrang and
    # SWNORM (four (ny, nx) float32) -- beside ``slope_geometry``'s two
    # static planes (slope, slp_azi) and, with topo_shading, the int32
    # shadow mask and ht_shad that ``terrain_shadow`` builds from the
    # edge-extended (ny+4, nx+4) terrain.  Per radiation call:
    # ``diffuse_fraction`` one (ny, nx) plane and ``after_radiation`` the
    # one-word declination; per land call ``before_land`` saves SWDOWN and
    # GSW (two planes) and ``adjust_surface_shortwave`` writes two.  Bound:
    # about a dozen (ny, nx) float32 planes per domain with slope_rad on,
    # none otherwise.
    'woof/core/topo_radiation.py': {
        '__init__': 4,
        'adjust_surface_shortwave': 2,
        'after_radiation': 1,
        'before_land': 2,
        'diffuse_fraction': 1,
        'slope_geometry': 2,
        'terrain_shadow': 2,
    },
    # RRTMGP chunk scratch is per (device, stream); allocation sites are unchanged.
    'woof/core/rrtmgp.py': {
        # Four flux arrays still allocate once each per call. The LW switch
        # adds mutually exclusive zero-initializer branches beside lw_up's
        # and lw_dn's empty-initializer branches (four AST sites), and
        # sw_up and sw_dn allocate through ``sw_allocate = cp.empty_like
        # if every column is daylit else cp.zeros_like`` (7f4d1a2b3, the
        # shortwave pass skips dark columns), one alias call each, which
        # the scanner reads as a site since it resolves allocator aliases.
        # Plus the two existing hydrometeor defaults, and chunk_up and
        # chunk_dn: one (chunk_ncol, nz+1) float32 pair per shortwave chunk
        # on the compacted-daylight path only, bounded by column_chunk
        # rather than by the domain.  All four float32 (ncol, nz+1) flux
        # arrays remain priced as columns/{lw_up,lw_dn,sw_up,sw_dn} by
        # rrtmgp_column_shapes.  The eleventh is the surface direct beam
        # BEP+BEM reads as SWDDIR (sf_urban_physics = 3): one float32 per
        # column, (ncol,), allocated only when the surface fields carry
        # SWDDIR or the driver asked for the surface diffuse flux WRF's
        # slope_rad reads (topo_radiation.request_surface_diffuse; lane
        # 281-namelist-gaps), beside the four (ncol, nz+1) flux arrays.
        # Both consumers share that one plane, so the count stays 11.
        '__call__': 11,
        '_cloud_optics': 1,
        '_finalize_cloud_optics': 2,
        '_gas_optics': 5,
        '_gas_vmr': 1,
        '_interface_temperatures': 1,
        '_interpolation_metadata': 2,
        # The block-fold refactor left each solver TWO allocation paths:
        # flux outputs when `out` is None, plus the partial-flux arrays on
        # the non-folding branch (which production never takes -- its
        # allocations exist textually, not at runtime).
        '_lw_rte': 5,
        '_planck_sources': 3,
        '_sw_rte': 6,
        '_validation_flags_device': 1,
        '_workspace_output': 1,
        'delta_scale': 1,
        'field': 1,
        'hydrometeor_paths': 2,
    },
    'woof/core/ruc_gpu.py': {
        '_soil_phase_partition_cuda': 1,
        'ruc_qsn_cuda': 1,
        'ruc_sea_ice_step_cuda': 3,
        'ruc_snow_preparation_cuda': 3,
        'ruc_snow_sea_ice_step_cuda': 3,
        'ruc_snow_soil_step_cuda': 11,
        'ruc_snow_temperature_step_cuda': 3,
        'ruc_soil_moisture_step_cuda': 2,
        'ruc_soil_properties_cuda': 1,
        'ruc_soil_step_cuda': 6,
        'ruc_soil_temperature_step_cuda': 2,
        'ruc_surface_parameters_cuda': 4,
        # Added at dd0df4e with the whole-column device path, and the one row
        # here that DOES scale with the nest: the argument is the snow-covered
        # subset of the RUC batch, so ``module_sf_ruclsm.F:2087``'s rebuild
        # allocates one float32 output per snow-covered column.  Fully
        # snow-covered d04 is 360,000 columns = 1.37 MiB, once per snow-cover
        # rebuild, and the input copy beside it is the same again.  Recorded
        # rather than removed because the alternative spelling is the Python
        # ``for`` loop over fdlibm's reduction that this replaced.
        'ruc_tanhf_glibc': 1,
        'ruc_transpiration_cuda': 2,
        # lane/speed-ruc.  58e52d010 (the full-width sfctmp reference
        # contract): one float32 output per RucSurfaceTemperatureStep field
        # scattered back to the full RUC batch; the reference a fused kernel
        # is graded against, reached from tests/test_ruc_sfctmp_fused.py and
        # not from the forecast.
        'ruc_sfctmp_full_width_reference': 1,
        # 1a0b7dccf (one RUC call is six kernels).  _sfctmp_scratch builds
        # the fused sfctmp's working set once per (device, stream, batch
        # width, soil levels) and caches it in _SFCTMP_SCRATCH: one float32
        # slab of (nzs x profile slots + scalar slots) x n, the device
        # pointer table, the uint64 check flags and an n-byte alive mask.
        # It scales with the RUC land batch, once per domain shape, never
        # per step.
        '_sfctmp_scratch': 4,
        # One per-call site builds the outputs (one array per
        # _SFCTMP_OUTPUTS field, (nzs, n) or (n,)), which the orchestration
        # it replaced also returned fresh.  The other seven are refusal
        # placeholders, reached only when an input is missing or malformed
        # so the refusal can be recorded in the device flags and raised
        # with the orchestration's first message.
        'ruc_sfctmp_full_width_fused': 8,
    },
    'woof/core/sfclay.py': {
        # Three cold-start surface outputs now receive zeros before a
        # flux-off call. They have the same caller surface shape as the
        # returned outputs, replacing uninitialized values rather than
        # adding a domain-volume workspace.
        '_allocate_result': 2,
        '_surface_array': 1,
    },
    # Shin-Hong (bl_pbl_physics=11): the same shape as YSU's row below --
    # five allocation sites building the thirteen per-call outputs (one
    # empty_like comprehension for the nine 3-D fields, four cp.empty for
    # the 2-D ones), all raw allocations, priced when Phase D wires the
    # scheme into the driver/preflight the way ysu_output_transient_shapes
    # prices YSU's.
    #
    # The sixth is the column workspace of the frame move (878435c39,
    # lane speed-default-pieces): ws = cp.empty in launch_shinhong, bound
    # shinhong_workspace_floats(nz, tile) * 4 bytes for the columns in
    # flight, priced by preflight.shinhong_column_workspace_bytes.  It
    # replaces a 17,160 B per-thread local frame the driver charged at the
    # card's full resident-thread capacity.
    'woof/core/shinhong.py': {
        'launch_shinhong': 6,
    },
    # Not a physics scheme: DomainState became a kernel-launching module
    # at lane/speed-dycore-host 66af318bc, when mu_at_u_faces and
    # mu_at_v_faces moved to one face_mass launch.  The one site is that
    # launch's 2-D (ny, nx + 1) or (ny + 1, nx) float32 output, one per
    # call, which replaces the roll/add/multiply/concatenate chain's
    # temporaries of the same shape.  Bound: one surface face array.
    'woof/core/state.py': {
        '_mass_faces_device': 1,
    },
    'woof/core/thompson.py': {},
    # THE SIX mp_physics=28 MODULES.  Entered here by the commit that merged
    # the port onto this line, which is the commit that first made them
    # visible to this scanner; the port itself predates the inventory.
    #
    # Every site below is inside a ``probe_*`` or ``_empty_like_*`` entry
    # point, and none of them is on the forecast path.  The probes exist so
    # the aerosol oracle harness can call one device helper at a time against
    # its WRF counterpart (tools/thompson_wrf461_oracle/probe_aero_*.F90),
    # and they allocate their own output because their caller is a test with
    # no scratch arena.  The forecast path itself allocates nothing: the two
    # modules that run every step, thompson_aerosol_state.py and
    # thompson_aerosol_sed.py, are EMPTY rows, and those two empty rows are
    # the assertion this block exists to make.
    'woof/core/thompson_aerosol_cold.py': {
        'probe_cold_warm_loop': 3,
    },
    'woof/core/thompson_aerosol_launch.py': {
        '_empty_like_float': 1,
        '_empty_like_int': 1,
        'probe_cloud_dist': 1,
        'probe_constant_tables': 1,
        # The libm-word probe (tests/test_thompson_aerosol_libm.py): one
        # output plus two one-element dummies for the unused argument pair.
        'probe_libm': 3,
        # The pair re-enforcement probe
        # (tests/test_thompson_aerosol_clipping_gpu.py): the two outputs.
        'probe_reenforce_pair': 2,
    },
    'woof/core/thompson_aerosol_sat.py': {
        'probe_droplet_evaporation_indices': 5,
        'probe_rain_evaporation_rates': 3,
    },
    'woof/core/thompson_aerosol_sed.py': {},
    'woof/core/thompson_aerosol_state.py': {},
    'woof/core/thompson_aerosol_warm.py': {
        'probe_frozen_constants': 1,
        'probe_warm_frozen_rates': 2,
        'probe_warm_rates': 3,
    },
    # The km_opt=2 TKE budget accumulates term by term into diagnostic
    # scratch the caller already owns and reduces on device once per step,
    # so the scanner finds no bare allocation.  Like uh_diag below, this
    # empty row asserts that rather than reserving a slot to fill later.
    # It is entered late, and that is the defect this row closes: the
    # budget landed in 02cfd530 without its inventory row, so the module
    # set drifted and this gate reported "appeared or vanished" instead of
    # a bound.  The rule the gate states -- a new module's row belongs in
    # the commit that adds the allocation -- was not followed then.
    'woof/core/tke_budget.py': {},
    # UP_HELI_MAX's every-step maximum runs entirely inside the caller's
    # persistent slots -- the scanner finds no bare allocation, and this
    # empty row is the assertion of that, not an unfilled placeholder.
    'woof/core/uh_diag.py': {},
    # WDM6 (mp_physics=16).  ZERO, and the bound is WSM6's, because the
    # adapter is WSM6's shape: every array it works in is a named
    # ``state.scratch`` slot from the static registry (wdm6_theta/rho/pii/
    # dz/z8w plus the seven shared mp_* surface slots), so preflight already
    # prices all of them and the scanner correctly finds no bare allocation.
    #
    # Two per-call transients the AST scanner does NOT see are stated here
    # rather than left implied, because "empty row" must mean measured-zero
    # and not looked-away-from:
    #   * ``cp.power(state.p / P0, RCP)`` on the Exner line, which is two
    #     (nz,ny,nx) FP32 temporaries per call, freed before the kernel
    #     launches.  It is byte-for-byte the line ``woof/core/wsm6.py:111``
    #     already runs under the same empty row, and the estimator's
    #     microphysics transient envelope is what prices it.
    #   * ``cp.ascontiguousarray(xland)``, which is at most ONE (ny,nx) FP32
    #     plane and in practice zero: ``initialize_physics`` builds
    #     ``fields["xland"]`` C-contiguous (physics.py:3774-3779), so the
    #     call returns the same object.  The 3-D transients above dominate
    #     it by a factor of 2*nz either way.
    #
    # This row belongs logically to ea7dcc296, the commit that added
    # woof/core/wdm6.py; that commit cannot be amended (forward commits
    # only), so it is entered here with its bound rather than left to a
    # later sweep -- which is exactly what this gate's message asks for.
    #
    # ONE site since a3f158bef (2026-09-13): launch_wdm6 takes the
    # substep-count status word as an argument and allocates
    # ``cp.empty((1,), uint32)`` only when a direct caller passes none.
    # Bound: one 4-byte word per call.  The adapter passes the priced
    # ``wdm6_count_status`` slot from the static registry
    # (preflight.scratch_slot_registry, the same commit), so on the run
    # path the count is zero and the site is the direct callers' (the
    # count-safety tests).  Entered here with its bound because a3f158bef
    # did not, and the gate was red for it on every box since
    # (proof/node-reds-276).
    'woof/core/wdm6.py': {'launch_wdm6': 1},
    # launch_wsm6: the column workspace that replaced the per-thread local frame (lane/292-mem-cuts
    # 052440f32b4), one per call, wsm6_workspace_rows(nz) float32 rows for the columns in flight (one tile,
    # at most the card's resident capacity), returned to the pool when the call ends.
    'woof/core/wsm6.py': {'launch_wsm6': 1},
    # WSM3 init words are one 25-word constant table per device,
    # uploaded with cp.asarray, which this raw allocator scanner excludes.
    # All timestep fields use DomainState scratch.
    'woof/core/wsm3.py': {},
    # ---- the urban canopy models (sf_urban_physics 1-3), 2026-09-30 --------
    # MYJURB (MYJ under BEP/BEM): ``launch_myjurb``'s per-call bundle, the MYJ
    # roster plus EXCH_M -- eight (nz, ny, nx) outputs from one comprehension,
    # PBLH/MIXHT/KPBL planes, RQIBLTEN's published zero and HT when a caller
    # passes none (physics.py passes the state's).  Priced by preflight's
    # myj_output_transient_shapes under sf_urban_physics 2/3, held to the
    # launcher by tests/test_myjurb_wrf471_parity.py.
    'woof/core/myjurb.py': {
        'launch_myjurb': 6,
    },
    # BEP: ``_class_for_table`` builds the class tables once per (device,
    # table) and caches them (CLASS_BYTES, 11 x 795 view words and the
    # 11 x 8,192 x 32-word view-factor scratch, 11 MiB, priced as
    # bep_class_scratch); ``launch_bep_columns`` allocates any output a caller
    # did not pass and the column workspace, capped at BEP_WORKSPACE_BYTES
    # (256 MiB, priced as bep_column_workspace); ``_bep_out`` is the four
    # surface planes and the error-flag plane, held on the urban state and
    # allocated once.
    'woof/core/urban_bep.py': {
        '_bep_out': 2,
        '_class_for_table': 3,
        'launch_bep_columns': 2,
    },
    'woof/core/urban_bep_couple.py': {},
    # UCM: ``_status_word`` is one uint32 per urban state, allocated once and
    # zeroed on the device per call; ``run_columns`` is the oracle-test
    # entry (``ucm_column_test``), not the forecast path, and sizes to the
    # columns a test hands it.
    # ``_run_columns480`` is the same oracle-test entry for the explicit
    # WRF 4.8 canopy generation (urban_ucm_generation = "wrf_480"): one
    # status word per test column, never the forecast path.
    'woof/core/urban_ucm.py': {
        '_status_word': 1,
        'run_columns': 2,
        '_run_columns480': 1,
    },
    'woof/core/ysu.py': {
        # The precedent this rule follows and the reason it is a ratchet
        # rather than a ban: YSU's nine per-call outputs are raw allocations,
        # but they are priced -- preflight.py's ysu_output_transient_shapes
        # enumerates every one of them by name.  An allocation the preflight
        # knows about is a cost; an allocation it does not is a hazard.
        # MYNN's were the second kind.
        #
        # The tenth is the column workspace of the frame move (17cf943ef):
        # ws = cp.empty at woof/core/ysu.py:162, bound
        # ysu_workspace_floats(nz, tile) * 4 bytes, priced by the same
        # commit's preflight row.
        'launch_ysu': 10,
    },

    # Physics constructor allocations are fixed domain/soil/history carriers,
    # priced by physics_inventory_census. Default WSM6 delegates its kernels
    # to the common loader and has no per-call allocation.
    # _prepare_atmosphere_fused (lane/radiation-fusion): the five phy_prep
    # fields the unfused ufunc chain allocated implicitly, now named.
    'woof/core/physics.py': {'couple_sase_w_tendency': 1, '_held_zero_tendency_views': 1, 'scalar': 1, 'zero': 2, 'zeros': 1, 'materialize': 2, 'couple_column_tendencies': 1, '_physics_exner': 1, '_prepare_atmosphere': 1, '_prepare_atmosphere_fused': 5, '__init__': 22, 'zero_surface': 1, '_run_cumulus': 8, '_compose_tendencies': 1, '_run_sfclay': 4, '_run_myj_pbl': 1, '_run_ysu': 2, '_scalar_pblmix_rates': 1, '_run_shinhong': 2, '_run_sase': 2, '_zero_accumulator': 1, '_attach_urban': 2, 'initialize_physics': 37, '_run_radiation': 1},

    # Physics constructor allocations are fixed domain/soil/history carriers,
    # priced by physics_inventory_census. Default WSM6 delegates its kernels
    # to the common loader and has no per-call allocation.
}


def _tracked_core_modules() -> frozenset[str] | None:
    """``woof/core`` files git knows about, or ``None`` if git cannot say.

    The ratchet is a claim about what this repository ships, and an untracked
    file cannot enter a commit.  Scoping to tracked paths is therefore not a
    loosening -- ``git ls-files`` lists staged files too, so a module is
    covered from the moment anyone prepares it for a commit, which is strictly
    before it can be pushed.

    It is here because this tree is shared: a second lane converting Noah-MP
    wrote ``noahmp_sflx_pre_slab.py`` and ``noahmp_water_slab.py`` into
    ``woof/core`` while this file was being written, and pinning per-function
    counts for another lane's half-finished module would record a number that
    is wrong by the time it is read.  When git cannot answer, everything on
    disk is scanned, which is the strict direction.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", "woof/core/*.py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):       # pragma: no cover
        return None
    if completed.returncode != 0:                       # pragma: no cover
        return None
    tracked = frozenset(line.strip() for line in completed.stdout.splitlines()
                        if line.strip())
    return tracked or None


def _kernel_factory_modules(sources):
    """Follow imported aliases and return-only wrappers of kernel loaders."""
    trees = {module: ast.parse(source) for module, source in sources.items()}
    imports = {}
    for module, tree in trees.items():
        names = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    parts = module.split(".")[:-node.level]
                    prefix = ".".join(parts + ([prefix] if prefix else []))
                for name in node.names:
                    names[name.asname or name.name] = prefix + "." + name.name
            elif isinstance(node, ast.Import):
                for name in node.names:
                    names[name.asname or name.name.split(".")[0]] = (
                        name.name if name.asname else name.name.split(".")[0])
        imports[module] = names

    def symbol(module, node):
        if isinstance(node, ast.Name):
            return imports[module].get(node.id, module + "." + node.id)
        if isinstance(node, ast.Attribute):
            parent = symbol(module, node.value)
            return parent + "." + node.attr if parent else None
        return None

    factories = {"woof.core.kernels.get_kernel",
                 "woof.core.kernels.get_kernel_int_defines"}
    changed = True
    while changed:
        changed = False
        for module, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = module + "." + node.name
                if name in factories:
                    continue
                if any(isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call)
                       and symbol(module, ret.value.func) in factories
                       for ret in ast.walk(node)):
                    factories.add(name)
                    changed = True
    return {module for module, tree in trees.items()
            if any(isinstance(node, ast.Call) and symbol(module, node.func) in factories
                   for node in ast.walk(tree))}


def _physics_gpu_modules() -> tuple[str, ...]:
    """Every tracked ``woof/core`` module that launches a kernel, by AST.

    Discovered rather than listed, so a scheme added tomorrow inherits the
    gate without anyone remembering to add it here.
    """
    tracked = _tracked_core_modules()
    sources = {}
    paths = {}
    for path in sorted((ROOT / "woof" / "core").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if tracked is not None and rel not in tracked:
            continue
        module = "woof.core." + path.stem
        sources[module] = path.read_text(encoding="utf-8")
        paths[module] = rel
    found = [paths[module] for module in sorted(_kernel_factory_modules(sources))]
    found.append("woof/ingest/closure_device.py")
    return tuple(found)


def test_kernel_discovery_keeps_imported_and_local_loader_wrappers():
    sources = {
        "woof.core.factory": "from woof.core.kernels import get_kernel, get_kernel_int_defines\n"
            "def loader(module, name, fork):\n"
            "    if fork: return get_kernel_int_defines(module, name, {'FORK': 1})\n"
            "    return get_kernel(module, name)\n",
        "woof.core.consumer": "from woof.core.factory import loader as selected\n"
            "def local(name): return selected('physics', name, False)\n"
            "def step(): local('body')((1,), (1,), ())\n",
        "woof.core.qualified": "import woof.core.factory as factory\n"
            "def step(): factory.loader('physics', 'body', True)((1,), (1,), ())\n",
        "woof.core.relative": "from . import kernels as k\n"
            "def step(): k.get_kernel('physics', 'body')((1,), (1,), ())\n",
        "woof.core.host": "def loader(): return object()\ndef step(): loader()\n",
    }
    assert _kernel_factory_modules(sources) == {
        "woof.core.factory", "woof.core.consumer", "woof.core.qualified",
        "woof.core.relative"}
    # A new allocation in a transitive caller remains visible to the
    # function-by-function ratchet; discovering wrappers is not an exemption.
    allocated = sources["woof.core.consumer"] + (
        "def unpriced():\n    work = cp.empty((columns, levels), dtype=cp.float32)\n")
    assert _inventory(allocated, "woof/core/consumer.py") == {"unpriced": 1}


def test_sase_actual_pool_peak_is_within_its_priced_phase(monkeypatch):
    """Allocation ownership only: numerical kernels are outside this CPU test."""
    import gc
    import weakref
    import numpy as np
    from woof.config import RunConfig, SASE_PBL_SCHEME
    from woof.core import preflight as pf, sase

    class Allocations:
        live = 0
        peak = 0
        rows = []

        def __getattr__(self, name):
            return getattr(np, name)

        asnumpy = staticmethod(np.asarray)

        def release(self, size):
            self.live -= size

        def record(self, result):
            self.live += result.nbytes
            self.peak = max(self.peak, self.live)
            self.rows.append((result.shape, result.dtype.itemsize, result.nbytes))
            weakref.finalize(result, self.release, result.nbytes)
            return result

        def empty(self, shape, dtype=np.float64):
            return self.record(np.zeros(shape, dtype))

        zeros = empty

        def empty_like(self, value, **kwargs):
            return self.record(np.zeros_like(value, **kwargs))

    allocations = Allocations()
    monkeypatch.setattr(sase, 'cp', allocations)
    monkeypatch.setattr(sase, '_kern', lambda name: lambda *args, **kwargs: None)
    # A zero-filled kernel stub produces zero z_i. The host scalar bound has
    # no allocation, so replace it with a finite value for this storage test.
    monkeypatch.setattr(sase, 'partition_cap', lambda *args, **kwargs: 1.0)
    cfg = RunConfig(nx=9, ny=8, nz=6, dx=1000., dy=1000., ztop=6000., dt=1.,
                    run_seconds=1., bl_pbl_physics=SASE_PBL_SCHEME)
    shape = (cfg.nz, cfg.ny, cfg.nx)
    # Match the model driver's seven held work fields and surface rho plane.
    held = [allocations.empty(shape, np.float32) for _ in range(7)]
    rho = allocations.empty(shape[1:], np.float32)
    state = [np.ones(shape, np.float32) for _ in range(5)]
    dz_col = np.full(shape, 100., np.float32)
    result = sase.launch_sase_step(*state, dx=1000., dy=1000., dz=100., delta=1000.,
        dt=1., n2=held[4], n2_moist=held[5], dz_col=dz_col, additive_dissipation=True)
    phases = pf.sase_workspace_phases(cfg)
    sizes = {name: sum(math.prod(dimensions) * width for dimensions, width in phase.values())
             for name, phase in phases.items()}
    assert allocations.peak == 102_400
    assert sizes == {'solve': 102_848, 'apply': 71_000}
    assert allocations.peak <= max(sizes.values())
    priced = {(tuple(dimensions), width) for phase in phases.values()
              for dimensions, width in phase.values()}
    assert all((dimensions, width) in priced for dimensions, width, _ in allocations.rows)
    assert all(value.dtype == np.dtype('float32') and value.shape == shape
               for value in (result['kv'], result['km_h']))
    assert sum(value.nbytes for value in (result['kv'], result['km_h'])) == 2 * 4 * math.prod(shape)
    del result, held, rho
    gc.collect()
    assert allocations.live == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('count', [0, 3])
@pytest.mark.parametrize('contiguous', [False, True])
def test_jacobi_outputs_are_caller_owned_and_status_is_transient(
        monkeypatch, dtype, count, contiguous):
    import sys
    from types import SimpleNamespace
    from woof.core import jacobi_eigh as module

    allocations, copies, launches = [], [], []

    def empty(shape, dtype):
        value = np.empty(shape, dtype=dtype)
        allocations.append(value)
        return value

    def array(value):
        copied = np.ascontiguousarray(value)
        if copied is not value:
            copies.append(copied)
        return copied

    def kernel(defines, shared_bytes):
        def launch(grid, block, args, *, shared_mem):
            source, w, v, status, n = args
            assert source.flags.c_contiguous and n == count
            assert shared_mem == shared_bytes
            launches.append((grid, block, defines))
            w.fill(7)
            v.fill(9)
            status.fill(2)
        return launch

    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(
        empty=empty, ascontiguousarray=array, int32=np.int32))
    monkeypatch.setattr(module, '_kernel', kernel)
    original = np.arange(count * 25, dtype=dtype).reshape(count, 5, 5)
    source = original if contiguous else original[:, :, ::-1]
    before = source.tobytes()
    w, v, sweeps = module.batched_eigh(source, return_sweeps=True)
    assert source.tobytes() == before and len(allocations) == 3
    assert w is allocations[0] and v is allocations[1]
    status = allocations[2]
    assert [(value.shape, value.dtype) for value in allocations] == [
        ((count, 5), np.dtype(dtype)), ((count, 5, 5), np.dtype(dtype)),
        ((count,), np.dtype(np.int32))]
    assert sum(value.nbytes for value in allocations) == count * (30 * np.dtype(dtype).itemsize + 4)
    assert all(not np.shares_memory(value, source) for value in allocations)
    assert not np.shares_memory(w, v)
    if count:
        assert sweeps == 2 and len(launches) == 1
        assert launches[0][2] == module.plan(5, np.dtype(dtype).str).defines
        assert len(copies) == int(not contiguous)
        assert sum(value.nbytes for value in copies) == int(not contiguous) * source.nbytes
    else:
        assert sweeps == 0 and launches == []
    retained = w.copy()
    module.batched_eigh(source)
    assert np.array_equal(w, retained)
    assert allocations[3] is not w and allocations[4] is not v


def _allocator_names(node) -> tuple[str, ...]:
    """The ``cp`` allocators an expression can evaluate to, if any.

    ``cp.empty_like`` is one, and ``cp.empty_like if full else
    cp.zeros_like`` is either; a call through a name bound to it allocates
    once whichever branch ran.
    """
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id in ("cp", "cupy") and node.attr in _ALLOCATORS):
        return (node.attr,)
    if isinstance(node, ast.IfExp):
        return tuple(sorted(set(_allocator_names(node.body)
                                + _allocator_names(node.orelse))))
    return ()


def _scope_bindings(scope) -> dict[str, tuple[str, ...]]:
    """Every plain name ``scope``'s own body binds, with its allocators.

    Nested functions and classes are their own scopes and are left out.  A
    name bound to anything else maps to ``()``, so it shadows an allocator
    alias of the same name in an enclosing scope.
    """
    bound: dict[str, tuple[str, ...]] = {}
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = scope.args
        for argument in (arguments.posonlyargs + arguments.args
                         + arguments.kwonlyargs
                         + [arguments.vararg, arguments.kwarg]):
            if argument is not None:
                bound[argument.arg] = ()
    pending = list(ast.iter_child_nodes(scope))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            names = (_allocator_names(node.value)
                     if node.value is not None else ())
            for target in targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = tuple(sorted(
                        set(bound.get(target.id, ()) + names)))
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))
    return bound


def _allocation_sites(source: str, rel: str):
    """Every ``cp.<allocator>(...)`` call in ``source``, with its function.

    A call through a local alias of an allocator counts as a site too,
    labelled with every allocator the alias can be.  7f4d1a2b3 (the
    RTE-RRTMGP shortwave pass skips dark columns) allocated ``sw_up`` and
    ``sw_dn`` through ``sw_allocate = cp.empty_like if ... else
    cp.zeros_like``, and a scanner that read only ``cp.<allocator>(...)``
    lost both per-call arrays while they still allocated.
    """
    found = []
    tree = ast.parse(source)

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = ["<module>"]
            self.bindings = [_scope_bindings(tree)]

        def _scoped(self, node):
            self.stack.append(node.name)
            self.bindings.append(_scope_bindings(node))
            self.generic_visit(node)
            self.bindings.pop()
            self.stack.pop()

        visit_FunctionDef = _scoped
        visit_AsyncFunctionDef = _scoped

        def visit_ClassDef(self, node):
            self._scoped(node)

        def _alias(self, name: str) -> tuple[str, ...]:
            for scope in reversed(self.bindings):
                if name in scope:
                    return scope[name]
            return ()

        def visit_Call(self, node):
            func = node.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id in ("cp", "cupy")
                    and func.attr in _ALLOCATORS):
                found.append((rel, self.stack[-1], func.attr, node.lineno,
                              node))
            elif isinstance(func, ast.Name) and self._alias(func.id):
                found.append((rel, self.stack[-1],
                              "|".join(self._alias(func.id)), node.lineno,
                              node))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def _inventory(source: str, rel: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in _allocation_sites(source, rel):
        function = site[1]
        counts[function] = counts.get(function, 0) + 1
    return counts


def test_real_device_helper_allocation_inventory():
    # Setup helpers selected by initialize_real's device column route.
    # Each output is caller-shaped FP64, except the split's FP32 full levels;
    # _flag and integrate's missing-column index are single uint32 words.
    # Casts, broadcast packing, reductions and base upload also allocate;
    # preparation_price includes their measured device-column workspace.
    rel = "woof/ingest/real_device.py"
    expected = {
        "_flag": 1,
        "widen": 1,
        "float32": 1,
        "_fp32_probe": 1,
        "base_residual": 1,
        "_specific_humidity_to_mixing_ratio": 1,
        "_cap_stratospheric_qv": 1,
        "_integrate_moisture": 3,
        "_pressure_at": 1,
        "dry_pressure_ladder": 1,
        "_rebalance_moist_pressure": 1,
        "_fp32_geopotential_split": 1,
        "_thermo": 1,
        "surface_pressure_from_surface": 1,
    }
    assert _inventory((ROOT / rel).read_text(encoding="utf-8"), rel) == expected


# Column adapters allocate their documented table and in-flight column slabs.
# reflectivity (18dbe1d5449): the analysis-frame radar diagnostic for
# mp=5, once per domain start and never per step: a ones dz volume, one
# (ny, nx) zero surface plane and one unused output volume, each sized to
# the state it is handed (a 64-row slab on the host/store road).
_PHYSICS_ALLOCATION_INVENTORY.update({'woof/core/fer_hires.py': {'reflectivity': 3}})
# golden/ports-rad (2.9.0 round 2): the radiation and switch ports' sites,
# entered with their bounds when the round merged them (the lane left them out).
_PHYSICS_ALLOCATION_INVENTORY.update({
    # aer_opt = 2: the column AOD, single-scattering albedo, asymmetry and
    # the two band arrays of one radiation call, each (ncol, nz) or (ncol,)
    # float32 and released with the call; bounded by the column chunk.
    'woof/core/aer_opt2.py': {'optics_device': 5},
    # Lightning: the flash-rate planes (ny, nx) of one due call and the
    # history extension; nothing scales with nz.
    'woof/core/lightning.py': {'extend': 1, 'launch_lightning': 2,
                                'update_lightning': 3},
    # New Goddard: its column inputs and the two heating/flux outputs of one
    # radiation call, (ncol, nz) or (ncol, nz+1) float32, released with it.
    'woof/core/radiation_composition.py': {'__call__': 15},
})
# icloud = 2: cal_cldfra2 returns one (ncol, nz) float32 cloud fraction, the
# same output cal_cldfra1 allocates on the icloud = 1 path.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['cal_cldfra2'] = 1
# The WRF 4.8 solar constant stays on the card so a radiation-due step can be
# captured as a graph (2.9.0 round 2): one zero-length float32 coordinate pair
# per radiation call, 0 bytes; the radconst output it reads is solar.py's.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['_solar_constant_device'] = 1
# OLR at the top of the radiation column (2.9.1, rrtmgp-edge-olr): one
# (ncol,) float32 per longwave call, released with it.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['__call__'] += 1
# The Morrison edge-ring mask (2.9.1, rrtmgp-edge-olr): one (ny, nx) bool per
# domain shape, built once and cached on the adapter; the ring columns the
# microphysics never runs on size their cloud from their own moments.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['_microphysics_columns'] = 1
# WRF's top/bottom boundary fluxes on RTE+RRTMGP (2.9.2, rrtmgp-rows): one
# (rows, ncol) float32 stack per call, 8 rows per spectrum run, released with
# the call (columns/boundary_fluxes); and the clear-sky SW solve's zero band
# cloud optics, one (chunk, nlay, nband) float32 per SW chunk
# (columns/clear_sky_cloud).  The clear-sky solves' flux outputs allocate
# inside _lw_rte/_sw_rte (out=None), already counted there.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['__call__'] += 1
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['_clear_sky_cloud'] = 1
# The paired solvers (rad-history, 2026-10-18): on the prepared route the
# clear-sky top/surface rows come out of the same kernel as the all-sky
# fluxes, so _lw_rte and _sw_rte each allocate one (ncol, nlev) float32
# clear up/down pair per call on that path -- the pair the separate clear
# solve used to allocate through its own out=None outputs.  The finalized
# optics and staged Planck cubes that path no longer builds are gone, so
# no priced term grows.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['_lw_rte'] += 2
_PHYSICS_ALLOCATION_INVENTORY['woof/core/rrtmgp.py']['_sw_rte'] += 2
# WRF's ocean mixed layer (2.9.1): initialize_ocean fills an absent (ny, nx)
# mixed-layer field once at domain start, a persistent driver field.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/surface_ocean.py'] = {'initialize_ocean': 1}
# lane/cap-slab-ocean (aa6bdeed308): the coupled-ocean SST and SST-change history planes,
# two (ny, nx) float32 allocated once on first diagnostic update.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/surface_ocean.py']['update_ocean_diagnostics'] = 1
# lane/cap-solar (04e65faa5ff): the native SW direct/diffuse/DNI history
# split, three (ny, nx) float32 planes at driver construction (__init__ 21 ->
# 22, one loop site); _run_radiation's DNI site allocates only when a caller
# removed that plane.
# lane/cap-hail (97e9a3ca061): hail.py draws all its working set from the
# state scratch arena (21 hail_*/hailcast_* slots); no pool allocation.
_PHYSICS_ALLOCATION_INVENTORY['woof/core/hail.py'] = {}


def _live_inventory() -> dict[str, dict[str, int]]:
    return {rel: _inventory((ROOT / rel).read_text(encoding="utf-8"), rel)
            for rel in _physics_gpu_modules()}


def test_this_gate_opens_no_device_and_must_never_start():
    """The reason this file is not part of ``test_mynn_pbl_scratch.py``.

    ``conftest._cupy_scope`` is the same function that
    ``pytest_collection_modifyitems`` uses to decide what gets marked
    ``gpu`` and skipped under
    ``GPUWM_NO_LOCAL_GPU=1``.  Asked about this module it must answer "no cupy
    anywhere", because the moment it answers otherwise every assertion below
    stops running on the only machine that routinely runs the suite -- which
    is exactly how four Noah-MP modules and one RUC function stayed missing
    from the inventory.
    """
    from conftest import _cupy_scope

    whole, functions = _cupy_scope(str(Path(__file__).resolve()))
    assert (whole, functions) == (False, frozenset()), (
        "this module now opens a device somewhere, so conftest will mark it "
        "gpu and skip the whole ratchet on every CPU run: "
        f"whole={whole} functions={sorted(functions)}")


def test_the_ast_gate_catches_a_raw_allocation():
    """The failing form, first, in both a converted and an unconverted module.

    Without this the passing tests below could be passing because the
    scanner is broken.  The injected line is the exact shape the old solver
    used -- an ``(ncol, nz)`` work array inside the leaf that needs it -- and
    it is injected twice: into ``mynn_pbl_gpu``, whose recorded count is
    zero, and into ``ysu``, whose recorded count is nine.  A ratchet that
    only notices the first-from-zero case would not be a ratchet.
    """
    converted = ROOT / "woof/core/mynn_pbl_gpu.py"
    source = converted.read_text(encoding="utf-8")
    anchor = "    blocks = (ncol + _TPB - 1) // _TPB\n"
    assert anchor in source
    injected = source.replace(
        anchor, "    sneaky = cp.empty((ncol, nz), dtype=DTYPE)\n" + anchor, 1)
    sites = _allocation_sites(injected, "woof/core/mynn_pbl_gpu.py")
    assert sites, "the scanner did not see an injected cp.empty"
    assert all(site[2] == "empty" for site in sites)
    # And the same scanner over the real file sees exactly the recorded
    # DMP-sibling buffers (4a0bb3f69), nothing more.
    real_sites = _allocation_sites(source, "woof/core/mynn_pbl_gpu.py")
    assert len(real_sites) == 4
    assert {site[1] for site in real_sites} == {"mynn_dmp_mf_cuda"}

    unconverted = ROOT / "woof/core/ysu.py"
    ysu_source = unconverted.read_text(encoding="utf-8")
    recorded = _PHYSICS_ALLOCATION_INVENTORY["woof/core/ysu.py"]
    assert _inventory(ysu_source, "woof/core/ysu.py") == recorded
    ysu_anchor = "        rthraten = cp.zeros_like(theta)\n"
    assert ysu_anchor in ysu_source
    ysu_injected = ysu_source.replace(
        ysu_anchor, ysu_anchor + "        extra = cp.zeros_like(theta)\n", 1)
    grown = _inventory(ysu_injected, "woof/core/ysu.py")
    assert grown != recorded, "the ratchet did not see an added allocation"


def test_the_ast_gate_sees_an_allocation_behind_an_alias():
    """A call through an allocator alias is a site, one per call.

    The form 7f4d1a2b3 used for rrtmgp's sw_up/sw_dn, injected into ysu
    beside an existing site: the count must grow by exactly the two alias
    calls, and a same-named plain function in a nested scope must not be
    read as the alias.  Then the real rrtmgp ``__call__`` must show its two.
    """
    rel = "woof/core/ysu.py"
    source = (ROOT / rel).read_text(encoding="utf-8")
    recorded = _PHYSICS_ALLOCATION_INVENTORY[rel]
    anchor = "        rthraten = cp.zeros_like(theta)\n"
    assert anchor in source
    function = next(site[1] for site in _allocation_sites(source, rel)
                    if site[3] == source[:source.index(anchor)].count("\n") + 1)
    injected = source.replace(anchor, anchor + (
        "        grow = cp.empty_like if theta.size else cp.zeros_like\n"
        "        extra_a = grow(theta)\n"
        "        extra_b = grow(theta)\n"
        "        def _shadow(grow=len):\n"
        "            return grow(theta)\n"), 1)
    grown = _inventory(injected, rel)
    assert grown == {**recorded, function: recorded[function] + 2}
    labels = {site[2] for site in _allocation_sites(injected, rel)
              if site[1] == function}
    assert "empty_like|zeros_like" in labels

    rrtmgp = "woof/core/rrtmgp.py"
    aliased = [site for site in _allocation_sites(
        (ROOT / rrtmgp).read_text(encoding="utf-8"), rrtmgp)
        if site[1] == "__call__" and site[2] == "empty_like|zeros_like"]
    assert [ast.unparse(site[4]) for site in aliased] == [
        "sw_allocate(lw_up)", "sw_allocate(lw_up)"]


def test_no_physics_gpu_module_allocates_outside_its_workspace():
    """The property, not the instance: scan every physics GPU module.

    MYNN's converted solver must stay at zero, and no other scheme may add
    a site.  The inventory is compared function by function, so moving an
    allocation between functions is a diff too.

    Both halves are collected before anything is asserted.  They used to be
    two statements, and the first one masked the second for a whole session:
    ``ruc_tanhf_glibc`` was a fourteenth site in a module recorded with
    thirteen from ``dd0df4e`` onward, and no run ever reported it, because the
    module-set assertion above it was already failing on four unrelated
    Noah-MP files.
    """
    live = _live_inventory()
    problems = []
    difference = set(live) ^ set(_PHYSICS_ALLOCATION_INVENTORY)
    if difference:
        problems.append(
            f"physics GPU modules appeared or vanished: {sorted(difference)}"
            " -- entering a new module's row belongs in the commit that adds"
            " the allocation, not in a later sweep; state its bound beside it")
    for rel in sorted(set(live) & set(_PHYSICS_ALLOCATION_INVENTORY)):
        recorded = _PHYSICS_ALLOCATION_INVENTORY[rel]
        if live[rel] == recorded:
            continue
        moved = sorted(set(recorded) | set(live[rel]))
        problems.append(f"{rel}: " + ", ".join(
            f"{name} {recorded.get(name, 0)} -> {live[rel].get(name, 0)}"
            for name in moved if recorded.get(name) != live[rel].get(name)))
    assert not problems, "\n".join(problems)
    for rel in _CONVERTED:
        assert rel in live, rel
    assert live["woof/core/mynn_pbl_gpu.py"] == {
        "mynn_dmp_mf_cuda": 4}


def test_the_allocation_allowlist_has_not_rotted():
    """Every enumerated exception must still be a real site."""
    for rel in _CONVERTED:
        live = _inventory((ROOT / rel).read_text(encoding="utf-8"), rel)
        stale = set(_PHYSICS_ALLOCATION_INVENTORY[rel]) - set(live)
        assert not stale, (
            f"{rel}: enumerated but no longer allocating: {stale}")


def test_parameter_scaler_has_a_registry_bound_and_an_initialization_owner():
    """A table edit must not grow into an undeclared grid-sized allocation."""
    from woof.physics_params import registry

    cells = {(section, category, row.column)
             for row in registry().values() if row.table == "ruc-vegparm"
             for section, categories in row.categories.items()
             for category in categories}
    assert len(cells) == 61, "update the parameter-scaling allocation bound with registry growth"
    helper = ast.parse((ROOT / "woof/core/physics_param_gpu.py").read_text(encoding="utf-8"))
    function = next(node for node in helper.body if isinstance(node, ast.FunctionDef)
                    and node.name == "scale_physics_param_values")
    count = next(node for node in function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "count"
                         for target in node.targets))
    assert ast.unparse(count.value) == "len(values)"
    allocations = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == "cp"
                   and node.func.attr in _ALLOCATORS]
    assert len(allocations) == 2
    assert all(ast.unparse(node.args[0]) == "count" for node in allocations)

    owners = []
    for path in sorted((ROOT / "woof").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom)
                    and node.module == "woof.core.physics_param_gpu"):
                continue
            containers = [parent.name for parent in ast.walk(tree)
                          if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and node in ast.walk(parent)]
            owners.append((path.relative_to(ROOT).as_posix(), tuple(containers)))
    assert owners == [("woof/core/physics.py", ("initialize_physics",))], (
        "parameter-scaling allocations gained a caller outside physics initialization", owners)


def _shape_names(node: ast.Call) -> set[str]:
    """Every bare name the first positional argument of ``node`` reads.

    That argument is the shape.  ``dtype=`` is skipped deliberately: it names
    ``cp``/``np``, which says nothing about how big the array is, and so is
    anything in callee position -- ``len(_ATM_OUT)`` is the field count, and
    ``len`` is not a dimension.
    """
    if not node.args:
        return set()
    shape = node.args[0]
    callees = {id(inner.func) for inner in ast.walk(shape)
               if isinstance(inner, ast.Call)}
    return {inner.id for inner in ast.walk(shape)
            if isinstance(inner, ast.Name) and id(inner) not in callees}


def _grid_sized_sites(source: str, rel: str, *, strict: bool = True):
    """Allocation sites in ``rel`` that are not sized by the batch.

    A shape that reads no name at all is a literal -- ``cp.empty(0)`` on the
    empty-batch early return -- and cannot scale with anything, so it passes.
    Everything else must name the batch width.

    ``strict`` additionally requires every other name to be a recognised
    per-column stride, which is the right rule where the name set is settled.
    Without it the test is that no name is a grid dimension, which is the
    essential half.
    """
    bad = []
    for _rel, function, allocator, lineno, node in _allocation_sites(
            source, rel):
        names = _shape_names(node)
        if not names:
            continue
        if strict:
            wrong = names - _BATCH_WIDTH_NAMES - _PER_COLUMN_STRIDE_NAMES
        else:
            wrong = names & _GRID_DIMENSION_NAMES
        if wrong or not (names & _BATCH_WIDTH_NAMES):
            bad.append(f"{rel}:{lineno} {function} cp.{allocator} "
                       f"shape reads {sorted(names)}")
    return bad


def test_the_noahmp_leaf_batch_transients_are_bounded_by_the_column_batch():
    """What makes the four Noah-MP rows above a cost and not a hazard.

    The four modules are on the forecast's own path -- ``DEVICE_LEAF_BATCH``
    is what ``LEAF_BATCH_EVALUATOR`` resolves to unless
    ``WOOF_NOAHMP_HOST_LEAVES=1`` is set -- so "it is only a fixture" is not
    available as a defence, the way it is for ``noahmp_leaves_gpu`` and
    ``noahmp_fluxprep_gpu``, which nothing under ``gpuwm/`` imports.

    What IS available is that none of them can grow with the nest.  Two facts
    together give that, and both are checked here rather than believed:

    1. every allocation's shape is ``len(calls)`` times a fixed per-column
       stride, with no grid dimension anywhere in it; and
    2. ``noahmp_runtime.noahmp_lsm_step`` flushes its staging list at
       ``COLUMN_BATCH``, so ``len(calls)`` is bounded by that constant and not
       by ``nx*ny``.

    Both are read out of the source rather than imported.  ``noahmp_runtime``
    imports all eight leaf batches at module scope and several of those import
    cupy, so importing it here would put a device back into a file whose whole
    purpose is not to need one.

    At ``COLUMN_BATCH`` the four together are 620 B per staged column, or
    1.21 MiB.  MYNN's unpriced set, which is why any of this exists, was
    18,743.8 MiB at d04.  That is the whole difference between the two
    findings, and it is the reason the four rows above were entered rather
    than reported as a defect.

    The two ``*_slab.py`` packers are checked for the same shape property and
    are NOT covered by (2): nothing under ``gpuwm/`` imports them yet, so they
    are on no forecast path and inherit no bound.  Whatever eventually calls
    them has to carry one -- see the note beside their inventory rows for what
    an unbounded slab costs at d04.

    The failing form is shown first, in the module with the most sites: a
    grid-shaped allocation must be named, and a batch-shaped one must not.
    """
    # -- the failing form ---------------------------------------------------
    rel = "woof/core/noahmp_sflx_pre_gpu.py"
    source = (ROOT / rel).read_text(encoding="utf-8")
    anchor = "    d_troot = cp.zeros(n, dtype=cp.float32)\n"
    assert anchor in source, "the anchor moved; re-point this negative control"
    injected = source.replace(
        anchor, "    grid = cp.zeros((ny, nx), dtype=cp.float32)\n" + anchor,
        1)
    named = _grid_sized_sites(injected, rel)
    assert len(named) == 1 and "'nx', 'ny'" in named[0].replace('"', "'"), (
        f"the shape check did not name an (ny, nx) allocation: {named}")

    # -- the failing form again, in the form the slab packers are held to ---
    named = _grid_sized_sites(injected, rel, strict=False)
    assert len(named) == 1 and "'nx', 'ny'" in named[0].replace('"', "'"), (
        f"the non-strict shape check did not name it either: {named}")

    # -- and the real files -------------------------------------------------
    for rel in _NOAHMP_LEAF_BATCHES:
        source = (ROOT / rel).read_text(encoding="utf-8")
        bad = _grid_sized_sites(source, rel)
        assert not bad, (
            "a Noah-MP leaf batch now sizes a device allocation from "
            "something that is not its call batch, so the COLUMN_BATCH bound "
            "recorded in _PHYSICS_ALLOCATION_INVENTORY no longer holds.  If "
            "the shape really is batch-sized and only the stride constant is "
            "new, add its name to _PER_COLUMN_STRIDE_NAMES and say so:\n"
            + "\n".join(bad))
    for rel in _NOAHMP_SLAB_BATCHES:
        source = (ROOT / rel).read_text(encoding="utf-8")
        bad = _grid_sized_sites(source, rel, strict=False)
        assert not bad, (
            "a Noah-MP slab packer sizes a device allocation from a grid "
            "dimension, or from nothing that names its batch width:\n"
            + "\n".join(bad))

    # -- and the bound itself ----------------------------------------------
    runtime_source = (ROOT / "woof/core/noahmp_runtime.py").read_text(
        encoding="utf-8")
    assert "if len(staged) >= COLUMN_BATCH:" in runtime_source, (
        "the staging loop no longer flushes at COLUMN_BATCH, so nothing "
        "bounds len(calls) and the four Noah-MP rows above become a hazard "
        "rather than a cost")
    batch = None
    for node in ast.walk(ast.parse(runtime_source)):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "COLUMN_BATCH"
                        for t in node.targets)
                and isinstance(node.value, ast.Constant)):
            batch = node.value.value
    assert isinstance(batch, int) and batch > 0, (
        "COLUMN_BATCH is no longer a module-level integer constant: "
        f"{batch!r}")
    # 620 B per staged column: 52 + 76 + 296 + 196, derived above.
    assert batch * 620 < 4 * 1024 ** 2, (
        f"COLUMN_BATCH={batch} now prices the Noah-MP leaf transients at "
        f"{batch * 620 / 1024 ** 2:.2f} MiB, above the 4 MiB this row was "
        "entered under; re-derive the per-column figure and say what it costs")
