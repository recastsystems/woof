#!/usr/bin/env python3
"""Build the complete Linux RW-WPS runtime from one clean source checkout.

The lower-level distribution builder deliberately accepts already-built
artifacts so it can verify each one independently.  This command is the
end-user entry point: it builds the Python wheel and every vendored Rust
decoder/CPU backend offline, then delegates to that verifier and packager.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.build_native_wrf_distribution import (  # noqa: E402
    _tracked_source_paths,
    build_distribution,
)
from woof.native_wrf_distribution import (  # noqa: E402
    BRIDGE_WORKSPACES,
    BUNDLED_BRIDGES,
    CPU_BACKEND_LIBRARY,
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
)


_TOP_LEVEL_EXCLUDES = {
    "cli.py",
    # ``python -m woof`` is the woof console script by another name: its
    # only import is woof.cli, excluded on the line above, so it leaves
    # with it.  The standalone project has no such door.
    "__main__.py",
    # The spectral-numerics seam: the hook the MODEL LOOP calls once per
    # completed slow large step, plus receipt binding into the run
    # capsule.  Pure runtime -- a preprocessing wheel completes no model
    # step, so nothing here is reachable from any staged door -- and it
    # imports woof.spectral_ops, which this wheel does not stage.
    "spectral_seam.py",
    # gpuwm-product front doors (CUDA forecast side): the domain wizard
    # imports the memory preflight + woof.cli, and the downscale front
    # door drives the offline CUDA child.  Neither belongs in the
    # standalone RW-WPS preprocessing wheel.
    "domain_wizard.py",
    # The wizard's prompt session, for the same reason and one more: it
    # reaches woof.domain_wizard and woof.core.preflight, both absent
    # here, so staging it fails this builder's own unresolved-import
    # scan.  A preprocessing wheel has no domain to size.
    "domain_interactive.py",
    # `woof domain --recipe --region` (2.9.0, box/country-recipe): its only
    # importer is domain_wizard.py, excluded above, and it imports the wizard
    # back, so staging it failed this builder's unresolved-import scan and
    # every standalone-package test with it.  A preprocessing wheel lays out
    # no domain.
    "region_recipe.py",
    # The versioned Python domain API (fix/ds-api, 2.9.11): it builds a
    # DomainState and steps the dycore (woof.core.dycore), the forecast
    # executor this wheel never carries; nothing staged imports it, and
    # staging it failed this builder's unresolved-import scan.
    "domain.py",
    # `woof go --smoke-radiation` (labs/smoke-optics, 2.9.11): a go-door
    # restatement that imports woof.go_cli; its one importer is go_cli.
    "smoke_radiation.py",
    # The warm-standby and race doors (2.9.4, `woof standby` / `woof
    # race`): a standby process holds a prepared forecast warm and starts
    # it the moment a source cycle posts; race drives standby (or `woof
    # go`) against the operational product.  Their importers are woof.cli
    # (excluded above), race_cli -> race -> standby_cli -> standby /
    # standby_runtime -> boundary_prestage, and the forecast runner
    # prepared_single_domain_forecast (not staged).  standby_cli reaches
    # woof.region_recipe (excluded above), so staging it refused this
    # builder's unresolved-import scan and every standalone-package test
    # with it.  A preprocessing wheel holds no forecast warm and races
    # nothing.
    "race.py", "race_cli.py", "standby.py", "standby_cli.py",
    "standby_runtime.py", "boundary_prestage.py",
    # `woof go --storm-nests` / [storm_nests] (2.9.4): authoring of dormant
    # storm-following nest slots, priced by the planner through the domain
    # wizard and woof.core.preflight, both absent here; its only importer is
    # go_cli.py (excluded below).  The forecast's nest-lives summary lives in
    # woof/nest_lives.py, which stages.  A preprocessing wheel authors no
    # nests.
    "storm_nests.py",
    # `woof go` drives tools/prepared_single_domain_forecast.py through
    # the GPU forecast.  Its imports happen to resolve against what RW-WPS
    # stages, so the scan would not have caught it -- but the runner it
    # exists to call is not in this wheel, so shipping it would offer a
    # command that cannot run.
    "go_cli.py",
    # Launch-time smoke capacity decisions rebuild forecast companion setups.
    # Preparation shares only the CPU receipt helpers in smoke_receipts.py.
    "smoke_execution.py",
    # Hub-height wind products (cap-hubwind, 2.9.4): a render door that
    # reaches woof.render (excluded below); its importers are render.py,
    # go_cli.py and cli.py, all excluded.  A preprocessing wheel renders
    # no forecast products.
    "hubwind.py",
    # The in-run tropical-cyclone tracker (cap-storm-tracks, 2.9.4): it binds
    # to a forecast's output through woof.go_cli and the ensemble runtime
    # context (woof.runtime); its only importer is go_cli.py, excluded
    # above.  A preprocessing wheel tracks no running storm.
    "tc_accumulator.py",
    # Input-cycle checks belong to go_cli/runplan, both excluded here.
    # Their checkpoint verification imports the forecast-only io.restart;
    # no standalone preparation door imports this module.
    "input_cycle.py",
    # Nested health recovery (lane/prod-stability) restores a forecast
    # from a checkpoint and re-arms the supervisor.  Its only importer,
    # prepared_domain_tree_forecast.py, is the forecast runner this wheel
    # does not carry; a preprocessing wheel runs no forecast to recover.
    "stability_recovery.py",
    # Its only importer, go_cli.py, is excluded, as is its dependency
    # woof.supervisor. A preprocessing wheel does not run forecasts.
    "forecast_supervisor.py",
    # `woof warm-kernels` compiles the forecast's GPU kernels by stepping
    # a synthetic domain through woof.core.dycore, moist and physics,
    # none of which this wheel stages; its only importer is woof.cli,
    # excluded above.  Staged, it was the one module reaching for those
    # three, and this builder's unresolved-import scan refused the whole
    # staging.  A preprocessing wheel runs no forecast to warm.
    "warm_kernels.py",
    # Both commands execute coupled fire forecasts and require the omitted
    # physics/dycore runners. Fire-static preparation stays in woof.static.
    "fire_ideal.py", "sfire_debug.py",
    # The live renderer that draws each frame as a forecast writes it.  Its
    # importers are go_cli.py, first_products.py and runplan.py, all
    # excluded above, and it reaches for woof.first_products, woof.render,
    # woof.render_receipts and woof.go_cli, none of which this wheel
    # stages, so the unresolved-import scan refused the whole staging.  A
    # preprocessing wheel runs no forecast whose frames it could draw.
    "live_products.py",
    # The any-point column store (rw-store v2) a forecast writes as its frames
    # land.  Its importers are runplan.py, offline_child_run.py and
    # woof/ensemble/batch_product_output.py, none staged here, and it reads
    # resident frames through woof.io.wrfout, which this wheel excludes, so
    # staging it refused the whole build.  Preparation writes no frames.
    "live_columns.py",
    # Forecast-history comparison calls the excluded render front door.
    # Staging it leaves an unresolved woof.render import in a package
    # whose preparation commands do not create forecast history.
    "render_compare.py",
    # The physics catalog behind `woof physics-catalog` and New forecast's
    # Physics step.  Its importers are woof/cli.py and woof/runplan.py,
    # both excluded, and it reaches for woof.domain_wizard,
    # woof.companion_domains and woof.core.pace, none of which this wheel
    # stages, so the unresolved-import scan refused the whole staging.  A
    # preprocessing wheel picks no physics for a forecast.
    "physics_catalog.py",
    # `woof go`'s own telemetry writer: the append-only events.jsonl a
    # go chain emits per stage.  Its ONLY importer anywhere in gpuwm/ or
    # tools/ is woof/go_cli.py:1674 -- excluded directly above -- and
    # the three modules it reaches inside its functions are
    # woof.progress_log, woof.runplan and woof.first_products, all
    # three already excluded below.  Staging it therefore put a module
    # in the wheel reaching for three deliberately absent ones, and
    # this builder's own unresolved-import scan refused the staging
    # outright: at that point `tools/build_rw_wps_release.py` could not
    # stage, build or publish the standalone wheel at all.  A
    # preprocessing wheel runs no `woof go` chain, so it has no stage
    # walls to record.
    "chain_events.py",
    # The early-render worker `woof go` and run-plan arm to draw the
    # analysis frame while the forecast runs.  It is reached only from a
    # running forecast, its only two referrers are go_cli.py and
    # runplan.py -- both excluded directly above and below -- and it
    # imports woof.go_cli itself, so staging it put a module in the
    # wheel reaching for one deliberately absent from it and this
    # builder's own unresolved-import scan refused the staging.  A
    # preprocessing wheel renders no forecast frames.
    "first_products.py",
    # The two prepared-cache GPU forecast runners.  Their substance
    # moved out of tools/ and into the package so a `pip install recast-woof`
    # can finish a forecast; that makes them top-level modules here, and
    # they are exactly what this preprocessing wheel does not do.  They
    # reach woof.core.model and the whole CUDA side, which RW-WPS does
    # not stage.
    "prepared_single_domain_forecast.py",
    "prepared_domain_tree_forecast.py",
    # Per-model-time-step progress for those two runners: the WRF-shaped
    # `Timing for main:` line, progress.jsonl and the frame-ready
    # markers.  A preprocessing wheel takes no model time steps and
    # renders no history frames, so it has nothing to report; the only
    # importers are the two runners immediately above, both excluded.
    # It also reaches woof.supervisor for the atomic marker write, and
    # supervisor is excluded below -- which is how this surfaced: the
    # staging scan refused on an unresolved internal import rather than
    # shipping a module that would raise at first use.
    "progress_log.py",
    # The machine front door.  It is an envelope over the run routes, so it
    # reaches every one of them -- woof.cli, woof.go_cli, woof.supervisor,
    # woof.domain_wizard, woof.certify.capsule and woof.core.preflight --
    # and all six are deliberately absent here.  A preprocessing wheel has no
    # run to plan, and staging it would offer a command that cannot execute
    # one, which is exactly the reason go_cli.py is excluded above.
    "runplan.py",
    # The run's disk projection and the download and preparation pricing it
    # reads.  Their only importer in the package is woof/runplan.py,
    # excluded directly above (tools/wiki_seed reads them too, and is not
    # staged), and download_budget.request_from_arguments reads a fetch argv
    # through woof.cli's parser, which this wheel does not carry.  Staged,
    # download_budget was a module reaching for a deliberately absent one:
    # this builder's own unresolved-import scan refused the whole staging and
    # the 2.8.0 cut battery went red on it.  Its measured table
    # (woof/data/download-bytes.v1.json) is not staged either, so the module
    # could not have priced anything here.  A preprocessing wheel plans no
    # run, so it has no run disk to price.
    "disk_budget.py",
    "download_budget.py",
    # Direct forecast disk admission reads the same excluded pricing,
    # checkpoint-retention policy and restart reader. No preprocessing
    # entry point integrates a forecast or writes its history output.
    "output_disk.py",
    # `woof speedrun` and the capsule module behind it.  A speedrun
    # times the SHIPPED chain end to end by driving `woof go` as a
    # subprocess -- prepare, forecast, render -- and seals a capsule of
    # the walls; `woof go` is excluded directly above, so the door
    # here would offer a command whose only action is to invoke a
    # command this wheel does not carry.  A preprocessing wheel runs no
    # forecast, so it has no course to run and no clock to seal.
    #
    # It is also how this surfaced: woof/speedrun.py reaches
    # woof.verify.cases._repo_config to find a course's config, and
    # woof.verify is the developer verification package RW-WPS omits
    # entirely -- so the staging scan refused rather than shipping a
    # module that would raise at first use.
    "speedrun.py",
    "speedrun_cli.py",
    # `woof sources` -- the human listing of the source registry.  It is
    # a CLI door and nothing else: its only importer anywhere is
    # woof/cli.py, excluded at the top of this list, and it reaches
    # woof.runplan (excluded above) for the inventory it prints.  A
    # wheel with no CLI has no door to register it on, so staging it put
    # a module in the wheel reaching for a deliberately absent one and
    # this builder's own unresolved-import scan refused the staging.
    "sources_cli.py",
    # These product orchestrators both execute forecasts: multi-run dispatches
    # isolated supervisor workers, while stream extends forecasts through
    # restart checkpoints.  Neither exposes an RW-WPS preprocessing surface,
    # and both import executor modules deliberately absent from this wheel.
    "multi_run.py",
    "stream.py",
    "downscale.py",
    "offline_child.py",
    # Own-geography preparation belongs to the offline forecast child;
    # its parent-history readers are excluded with that route.
    "offline_child_geography.py",
    "offline_child_run.py",
    "offline_child_smoke.py",
    # Moving nest-downs (`woof downscale --follow`, `--warm-from`): the
    # moving child orchestrator, its warm start from a parent restart and the
    # follow-target detector.  Their only importers are downscale.py and
    # offline_child_run.py (excluded above), and they reach woof.offline_child,
    # woof.io.restart and woof.io.netcdf_serialization, none staged here.
    "moving_downscale.py",
    "offline_child_warm.py",
    "setup_targets.py",
    # `woof resume` locates a forecast checkpoint and hands it to the
    # supervisor's `run --restart` dispatch.  It reached this staging
    # only because it is a top-level module: nothing RW-WPS ships
    # imports it (the one importer is the excluded woof.cli), no
    # standalone entry point exposes it, and its own two lookups are the
    # restart validator in woof.supervisor and the header reader in
    # woof.io.restart -- one forbidden here, the other not staged at
    # all.  A preprocessing wheel has no checkpoints to resume from.
    "resume.py",
    # `woof branch` is resume's other half: a NEW run seeded from an
    # existing run's checkpoint (woof.branch).  It reached this staging
    # the same way resume.py did -- it is a top-level module and nothing
    # RW-WPS ships imports it (the one importer is the excluded
    # woof.cli) -- and its own lookups are the forecast executor in
    # woof.core.model and the checkpoint locator in woof.resume, one
    # forbidden here and the other excluded directly above, so staging
    # it fails this builder's own unresolved-import scan on four
    # imports.  A preprocessing wheel has no checkpoints to branch from.
    "branch.py",
    # The saved history a resumed forecast's first pictures read from
    # beside its checkpoint.  Its importers are go_cli.py, live_products.py
    # and first_products.py, all excluded above, and it reaches
    # woof.io.restart (refused below), woof.resume and
    # woof.live_products (both excluded above).  Staged, it was a module
    # reaching for three deliberately absent ones, and this builder's own
    # unresolved-import scan refused the whole staging.  A preprocessing
    # wheel resumes no forecast and draws no pictures.
    "restart_render.py",
    "runtime.py",
    # Forecast-history consumers stay in WOOF; preparation still parses
    # their metadata through simulated_radar_config.py.
    "simulated_radar.py",
    "state_digest.py",
    "supervisor.py",
    # Forecast input runners and terminal job/catalog entry points have no
    # standalone preprocessing command. Their readers remain staged below.
    "metem_forecast.py", "wrfinput_forecast.py",
    # The resident member-leg server (2.9.2, woof/leg_server.py) runs
    # WRF-input forecast legs in warm engine processes: it imports the
    # excluded woof.wrfinput_forecast on both of its paths, nothing
    # RW-WPS ships imports it, and a preprocessing wheel runs no leg.
    # Staged, its import of the excluded executor refused this builder's
    # own unresolved-import scan and with it the whole wheel staging.
    "leg_server.py",
    # The warm per-card leg executor (sweep 5, woof/leg_executor.py) is
    # the same kind of process: started as its own worker, it runs each
    # leg through the excluded woof.wrfinput_forecast's main.  No module
    # imports it.  Staged, its wrfinput_forecast import refused this
    # builder's unresolved-import scan and every standalone-package test.
    # Its companion woof/checkpoint_schedule.py (`--checkpoint-at`
    # instants and a worker's restart request) is imported only by the
    # forecast runners wrfinput_forecast and prepared_domain_tree_forecast,
    # both excluded.  A preprocessing wheel runs no leg and writes no
    # checkpoint.
    "leg_executor.py", "checkpoint_schedule.py",
    # The physics-tuning campaign (sweep 5, woof/tuning*.py): it prepares
    # member forecasts through woof.go_cli and woof.stage_cli, runs each
    # member through woof.prepared_domain_tree_forecast in a
    # `python -m woof.tuning_member` child, then collects, scores and
    # reports.  Only the tuning modules import one another; staged,
    # tuning_member's runner import refused this builder's
    # unresolved-import scan and every standalone-package test.  A
    # preprocessing wheel runs no tuning members.
    "tuning.py", "tuning_collect.py", "tuning_member.py",
    "tuning_observations.py", "tuning_prepare.py", "tuning_report.py",
    "tuning_retention.py", "tuning_update.py",
    # The kernel pre-warm (2.9.2, woof/kernel_prewarm.py) compiles a run's
    # CUDA kernel set beside the fetch; its importers are the forecast
    # runners (wrfinput_forecast, prepared_*_forecast) and warm_kernels,
    # and it reaches woof.core.preflight, which is not staged and reads
    # woof.io.restart (refused below).  A preprocessing wheel compiles no
    # kernels.
    "kernel_prewarm.py",
    # The capability lanes' forecast/product doors (2.9.4): hub-height wind
    # maps are a render product (lane/cap-hubwind; importers woof.render,
    # woof.go_cli, woof.cli) and the tropical-cyclone accumulator is a
    # `woof go` request loader (lane/cap-cyclone-go; importer woof.go_cli).
    # Both reach excluded doors; a preprocessing wheel renders and runs
    # nothing.
    "hubwind.py", "tc_accumulator.py",
    # mp_tend_radar's live attach (2.9.1): the dycore and the device kernel
    # on running domains; preparation reads the tendency in
    # woof/ingest/wrfinput_radar.py, which stays staged.
    "wrfinput_radar_forcing.py",
    "launchpad_api.py", "tui_worker.py",
    # These forecast UI doors size research runs, dispatch remote workers,
    # select runtime products or validate starter run plans. They reach the
    # excluded wizard, supervisor, restart and runplan modules; no RW-WPS
    # entry point or staged preparation module consumes them.
    "remote_cli.py", "remote_worker.py", "research_workspaces.py",
    "starter_template.py", "tui_products.py",
    "case_catalog.py", "case_catalog_import.py",
    # The companion's query/edit/fit doors are registered only by woof.cli;
    # the remote modules transfer and supervise ArWen jobs. Configuration
    # recovery is consumed by those excluded run/catalog doors. None belongs
    # to source_cli's standalone preprocessing argument surface.
    "companion_query.py", "companion_domains.py", "companion_forcing.py",
    # The saved-setups door (`woof companion-setup save|start`) is the same
    # surface one step on: a library of FORECAST configurations a person
    # copies and re-times.  Its only importer is woof/cli.py:374, and its
    # own work is done through woof.companion_domains (the native editor
    # excluded directly above), woof.starter_template and woof.fetch_routes
    # -- so staging it put a module in the wheel reaching for a deliberately
    # absent one and this builder's own unresolved-import scan refused the
    # staging outright.  A preprocessing wheel prepares inputs for a forecast
    # it does not run; it has no forecast setups to save or start from.
    "companion_setups.py",
    "configuration_recovery.py", "remote_artifacts.py",
    "remote_input_transfer.py", "remote_plan.py", "remote_processed.py",
    # Companion forecast configuration and remote rendering/preparation
    # orchestrators belong to the same excluded ArWen UI/worker surface.
    # Standalone source_cli has no entry point or importer for these doors.
    "companion_physics.py", "cyclone_setup.py", "remote_native_plots.py",
    "remote_preparation_v2.py", "remote_processed_cache_v2.py",
    "remote_processed_v2.py",
    # The product renderer publishes supervisor/first-products receipts.
    # Its callers are the excluded ArWen CLI/go/run-plan/remote doors and
    # unstaged DA/cells/verification packages. RW-WPS has no render command;
    # keep the renderer and its receipt owner together in full ArWen, rather
    # than staging a renderer whose result publication cannot be imported.
    "render.py", "render_receipts.py",
    # Analysis-cycle orchestration and cyclone setup are full ArWen entry
    # points. Their shared source facts remain in source_drivability; the
    # standalone source_cli never invokes an analysis or a tracking setup.
    "background_contract.py", "regional_preparation.py",
    # Ensemble orchestration binds member fetches before calling the native
    # preparer. The standalone preparation wheel receives already fetched
    # inputs and does not ship its run-plan or domain-wizard callers.
    "era5_preparation.py",
    "local_da.py", "local_da_fetch.py", "local_da_observations.py",
    "local_da_runtime.py", "cyclone_seed.py", "cyclone_sources.py",
    # This CLI-only trajectory selector launches cyclone forecasts through
    # the excluded wizard/setup doors; standalone preprocessing never calls it.
    "tc_strata.py",
    # The Earth-2 job door runs tracking, forecasts and rendering. Fetch
    # and the model inventory remain in earth2_run for input acquisition.
    "earth2_door.py", "earth2_job.py", "earth2_diagnostics.py",
    # The rapid-cycling controller and the nowcast scorer belong to the same
    # analysis-cycle surface as the doors above, and both are reached only
    # from them (`woof/local_da.py` and `woof/local_da_runtime.py`, already
    # excluded here).  Each is unstageable on its own terms as well: the
    # controller reaches woof.ensemble, woof.mcp and woof.supervisor, none
    # of which this wheel stages, and the scorer reaches woof.verify.obs
    # nine times -- the verification package this wheel omits by design.
    # Staging either put a module in the wheel that ImportErrors the moment
    # it is reached, and this builder's own staging scans refused outright.
    "local_da_controller.py", "local_da_score.py",
    # Terrain preparation now calls acoustic_adaptation.steepest_slope and
    # terrain_clock_local.candidate_map through static.terrain_autosmooth.
    # candidate_map also reads terrain_clock.measured_map. These CPU readers
    # stay staged with their measured JSON tables; neither module executes a
    # forecast, and the model, dycore and physics executors remain excluded.
}
_CORE_MODULES = {
    "__init__.py",
    # The FAST_SBM (mp_physics=30) Registry state contract: tuples of bin,
    # mass and number names transcribed from WRF registry.sbm:318, no
    # imports at all.  device_inventory, wrf_direct, wrfinput, lateral_bc,
    # state and the WRF physics inventory (all staged) read it at module
    # scope; without it the wheel refused staging on ten unresolved imports.
    "fast_sbm_contract.py",
    # aer_opt = 2's option and value validation: woof.config and the
    # namelist importer call validate_options when a namelist selects the
    # user aerosol.  Module scope is stdlib and numpy; its device optics
    # import the kernels only inside optics_device (recorded below).
    "aer_opt2.py",
    # The namelist importer collapses equal adaptive-clock bounds using
    # wrf_num_sound_steps. Its module imports adaptive_timestep at module
    # scope; both use stdlib and numpy without initializing a forecast.
    # The physics cadence refresh is function-local and recorded below.
    "adaptive_clock.py", "adaptive_timestep.py",
    "constants.py",
    "diagnostics.py",
    # The C-library transcriptions the CPU paths hash through (A126):
    # diagnostics.py imports noahmp_libm.powf_array, and
    # woof/static/lambert.py and projection.py import host_libm.  Both
    # need only math, struct and numpy.
    "noahmp_libm.py",
    "host_libm.py",
    # The portable transcendentals the host preparation takes (A130):
    # real.py, nest_init.py, grid.py, nest_interp.py and the other
    # preparation paths import it.  numpy, ctypes and the stdlib at
    # module scope; the CPU preprocessing library it calls is staged.
    "portable_math.py",
    # The per-card cache keys the staged noah.py and milbrandt2_constants.py
    # take for their device tables (lane 281-multigpu-2card): functools and
    # threading at module scope, CuPy only inside the call.
    "device_cache.py",
    # The [devices] options table config.py and experiment.py parse beside
    # [tiles] (lane 281-multigpu-2card): dataclasses and typing at module
    # scope; the rank machinery it names is imported only by the forecast.
    "devices.py",
    "track_boundary.py",  # NumPy-only boundary diagnostic used by storm_tracking.
    # Prepared/wrfinput initialization imports the lazy tile door, and the
    # CPU fit estimator reads the import-free mosaic array inventory.
    "noah_mosaic.py", "noah_mosaic_door.py",
    "grid.py",
    "landuse.py",
    # landuse.py reads the LCZ categories through urban_tables under the
    # urban land-use legend; urban_tables imports only noahmp_libm and noah,
    # both staged here.
    "urban_tables.py",
    "microphysics_transition.py",
    # The chem species arena (woof/core/chem_state.py) and the process
    # interface it sizes process arrays from (chem_context.py).  DomainState
    # (staged) allocates the chem_<row> fields when a run names chem sets,
    # and a prepared chem state (smoke initial conditions) is built on the
    # host by this package.  Both are CuPy-free: the array module is passed
    # in, and a process module is imported only when a chem run names it.
    "chem_state.py",
    "chem_context.py",
    # The chem processes' preparation-time arrays (dust statics sampled from
    # the WPS geography, the sulfur chemistry's lat/lon): the staged real
    # preparation routes (mapped_direct, era5_direct, gfs_direct) write them
    # into a chem state before it is sealed.  Self-contained: numpy, the chem
    # table, chem_context/chem_state above and woof.static (staged whole);
    # no process module and no kernel.
    "chem_statics.py",
    # The process modules in this build, because attach_chem_state (and the
    # device inventory pricing it) reads a named process's ALLOCATES when a
    # prepared chem state names one: without them a preparation of any
    # chem set with a process raised ModuleNotFoundError.  CuPy-free at
    # import by the process interface's contract; the card path of step()
    # is function-local and never reached by a preparation.
    "chem_vertmx.py",
    # The smoke set's processes (lane/aq-smoke), for the same reason: a
    # preparation of the smoke set reads their ALLOCATES.  chem_plumerise
    # imports its parameter and cadence module at module scope.
    "chem_fire.py",
    "chem_plumerise.py",
    "chem_plumerise_cache.py",
    "chem_wetdep.py",
    # The GOCART process modules (lane/aq-gocart), for the same reason:
    # a prepared GOCART state names them and attach_chem_state reads their
    # ALLOCATES.  CuPy-free at import; each launcher and step() imports the
    # card path inside the function, and chem_column holds their shared
    # array checks.
    "chem_ageing.py",
    "chem_column.py",
    "chem_drydep_aerosol.py",
    "chem_drydep_gas.py",
    "chem_dust.py",
    "chem_inventory.py",
    "chem_mp_coupling.py",
    "chem_optics.py",
    "chem_seasalt.py",
    "chem_settling.py",
    "chem_sulfur.py",
    # The Milbrandt-Yau constant table, reached by microphysics_transition
    # above when a mixed nest edge enters mp=9 and the kernel needs the
    # scheme's own ck vector.  Pure numpy, no kernels and no scheme: the
    # forecast module woof/core/milbrandt2.py stays out of this wheel and
    # reads its device cache from here.
    "milbrandt2_constants.py",
    # The three mp=28 scalars the same mixed-edge module seeds a
    # Thompson-aerosol child from (NT_C and the two aerosol floors).
    # Import-free constants: the table contract that re-exports them
    # (thompson_aerosol_contract.py) stays out, as recorded below.
    "thompson_aerosol_constants.py",
    # The card probe `--preprocess-backend auto` reads before it keeps a
    # certified card: free memory, utilization and the CUDA error of a
    # card too full to open, from a short-lived subprocess.  Without it
    # this package had no load reading and prepared on a card another
    # program held busy or nearly full, where woof moves that
    # preparation to the CPU.  A leaf: stdlib at module scope plus a
    # function-local woof.local_gpu (staged); the forecast memory
    # preflight that re-exports it stays out.
    "device_probe.py",
    # The inventories a CUDA preparation is priced from before its first
    # device allocation: the exact DomainState allocation list, one
    # boundary interval's side tables and the card's CUDA context.
    # woof/ingest/preparation_price.py (staged) reads all three for every
    # GFS, ERA5 and mapped preparation; while they lived in the forecast
    # memory preflight, which stays out, this builder refused the staging
    # on three unresolved imports and a staged wheel raised
    # ModuleNotFoundError at the price.  Module scope is stdlib plus
    # woof.config (staged), woof.boundary_fields (staged) is
    # function-local, and preflight re-exports every name.  No CuPy, no
    # forecast executor.
    "device_inventory.py",
    # Configuration-only prognostic names used by state and device inventory.
    "physics_inventory.py",
    # The resident admission the prepared-cache restore, the wrfinput
    # restore and init_at_rest (all staged) take before their DomainState
    # constructor: the exact state inventory and the boundary tables
    # against the card's free memory, refused by name.  Module scope is
    # stdlib plus woof.ingest.memory_refusal (staged); CuPy and
    # device_inventory are function-local, and the one forecast import
    # (the loader slab's estimate) is recorded in _OPTIONAL_STAGED_IMPORTS.
    "resident_admission.py",
    # The sea-level pressure reduction and its nine-point smoother,
    # reached by storm_tracking.py below when a follow block tracks
    # `field = 'pressure'`.  Same shape as sase_limits.py further down:
    # the arithmetic lives in woof/core rather than in its historical
    # home under woof/verify because that tree is developer
    # verification and this distribution omits it -- staging
    # storm_tracking.py while it reached across that boundary is what
    # the verification-import scan below refused.  Leaf module, numpy at
    # module scope and nothing internal, so it carries no CuPy and no
    # forecast executor.
    "mslp.py",
    "nest_interp.py",
    # Shared config/initialization contracts, without memory or GPU executors.
    "nest_fields.py", "ozone_contract.py", "inflow_perturbation.py",
    # The config reader validates attribute-following fields and source
    # domains through this module; storm_tracking also imports its grammar.
    # Its imports are NumPy and the already-staged streaming options module.
    # Its device reduction is function-local and is never called by config
    # validation, so importing it requires no CuPy or forecast executor.
    "attribute_tracking.py",
    # Config VALIDATION for the two storm-following blocks, which is
    # preprocessing work: `woof/experiment.py` is staged, and it calls
    # `build_follow_config` for `[relocation.follow]` and
    # `build_spawn_config` for a `[[domain]]` `spawn` table while LOADING a
    # config, long before any state exists.  Excluding them would make this
    # wheel refuse to read a perfectly valid storm-following TOML.  Both are
    # stdlib + numpy (nest_spawn imports only storm_tracking beyond that),
    # so they carry no CuPy and no forecast executor.
    "storm_tracking.py",
    "nest_spawn.py",
    # The SAME config-validation reason, for the `[relocation.track]`
    # table: `woof/experiment.py` calls `build_track_config` while
    # LOADING a config, and reads `POSITION_ONLY_FIELDS` and
    # `SURFACE_LEVEL` from here to refuse a track block whose columns the
    # configured tracker could never fill.  Those three imports sit at
    # function scope inside the loader, so leaving the module behind made
    # this wheel refuse to read a valid storm-following TOML -- and the
    # staging scan below refused the wheel outright rather than shipping
    # a half-readable one.  Module scope is stdlib plus numpy, and its one
    # internal lookup is `woof.core.storm_tracking`, staged directly
    # above for the same reason.  No CuPy, no forecast executor: the CSV
    # this module writes is driven from `woof/runtime.py`, excluded.
    "storm_track_writer.py",
    # The SAME reason as the two directly above, for the three lifecycle
    # tables that sit beside `spawn` on a `[[domain]]`: `woof/experiment.py`
    # calls `build_retire_config` for `retire`, `build_rearm_config` for
    # `rearm` and `build_domain_follow_config` for a per-domain `follow`
    # while LOADING a config, so a wheel without this module refuses to read
    # a valid storm-following TOML -- and, because those three imports sit
    # at function scope inside the loader, the staging scan below refused
    # the wheel outright rather than shipping a half-readable one.
    # Module scope is math/dataclasses plus numpy, and its only two internal
    # lookups are woof.core.uh_diag and woof.core.storm_tracking, both
    # already staged for the same config-validation reason.  No CuPy, no
    # forecast executor: the tree surgery a retirement decision feeds is
    # woof/runtime.py's, and that module is excluded above.
    "nest_lifecycle.py",
    # Reached by ALL THREE above -- storm_tracking, nest_spawn and
    # nest_lifecycle -- for the tracking-window slot
    # NAMES: the windows are one source of truth rather than string
    # literals copied into each consumer, so validating a spawn, retire or
    # follow block imports the module that owns them.  It costs nothing this
    # wheel refuses -- module scope is numpy plus core/constants.py
    # (already staged), and the only CUDA reference is a function-local
    # get_kernel inside the device fold, which no config-validation path
    # ever calls.
    "uh_diag.py",
    # The same config-validation reason again: `woof/experiment.py` and
    # nest_lifecycle call `validate_reach_speed` for a follow block's
    # `reach_speed_m_s` while LOADING a config, and the statics corridor
    # (woof/static/corridor.py, staged) sizes a moving nest from its
    # reach.  Stdlib only: no CuPy, no forecast executor.
    "nest_reach.py",
    # The [tiles] option surface, and NOT the streamed transport.  Module
    # scope here is dataclasses plus typing: every tilestream import in the
    # file is function-local, so staging it carries no forecast executor
    # and no CuPy.  It is not optional either -- woof/experiment.py holds
    # `tiles: StreamingOptions = OFF` as a field default, so the class
    # cannot be defined without it, and a wheel that stages experiment.py
    # and not this one fails on `import woof.era5_direct`.  That is how
    # the tilestream port broke this wheel: the scan below read
    # `from woof.core import streaming` as an import of `woof.core`
    # alone, which IS staged, so nothing refused the staging.  It now
    # checks woof.core.streaming too (A129).
    "streaming.py",
    "nssl2_contract.py",
    "noah.py",
    # state.py allocates the SASE prognostic and reads its realizability
    # floor from here.  A dependency-free constants module, which is why
    # those two values live in woof/core rather than in the closure's
    # authority module under woof/verify -- that tree is developer
    # verification and this distribution omits it.
    # The benchmark's mp=50 table-authority arm validates the P3 lookup
    # table at profile binding -- the same stage and the same reason as
    # the Thompson arm beside it.  Leaf module: stdlib + numpy at module
    # scope, no CuPy, no forecast executor.
    "p3_tables.py",
    "sase_limits.py",
    "state.py",
    # Shared clock bindings contain only scalar bookkeeping; state.py uses
    # them when publishing a domain clock, including prepared host state.
    "sfire_clock.py",
    "thompson_contract.py",
    # The mp=28 real-data cold start closes cloud droplet, rain and ice
    # number over the analyzed mass through the scheme's own entry block
    # (woof/ingest/real.py, staged above), and that block is the host
    # mirror woof/core/thompson_entry.py: numpy at module scope, its
    # powers from noahmp_libm.py and host_libm.py (staged above), its
    # gamma moments from thompson_aerosol_contract.py, whose only other
    # internal reaches are thompson_contract.py (staged) and, function-
    # locally, woof.physics_compat (staged); the contract's exp/log are
    # correctly_rounded_libm.py, 75 lines of the decimal module.  Three
    # leaf modules, no CuPy and no forecast executor, so a preparation-only
    # install starts an mp=28 case with the same numbers the full one does.
    "thompson_entry.py",
    "thompson_aerosol_contract.py",
    "correctly_rounded_libm.py",
    # state.py allocates WDM6's three number prognostics by NAME and reads
    # that tuple from `wdm6_constants.WDM6_NUMBER_SPECIES` rather than
    # copying the spelling, so an mp_physics=16 config cannot be loaded
    # without it.  Same shape as sase_limits.py above: a leaf constants
    # module, stdlib-only at module scope (dataclasses/functools/math), no
    # CuPy and no forecast executor.  It reaches one sibling --
    # `wsm6_constants.py`, staged just below -- because WDM6's cold half IS
    # WSM6's and the coefficients are imported rather than duplicated.
    # `woof/core/microphysics.py` and `woof/core/wdm6.py` import it too,
    # but both are already excluded from this wheel, so state.py is the
    # only module that forces it here.
    "wdm6_constants.py",
    # Reached only by wdm6_constants.py, for the shared rimed-ice
    # constants and the Weierstrass Gamma.  Also stdlib-only at module
    # scope; it is staged as that module's dependency, not on its own
    # account -- nothing else in this wheel names it.
    "wsm6_constants.py",
    "wsm_family_constants.py",
}
#: `nest_spawn_init.py` and `relocation_init.py` are forecast-time child
#: initializers, not preprocessing: one builds a nest that is born mid-run,
#: the other rebuilds one that just moved, and both need a LIVE parent state
#: to do it.  They reach `woof.core.nest_relocation`, `woof.core.health`
#: and `woof.ensemble.state_sha`, none of which this wheel stages.  Their
#: only importers are `woof/core/spawn_runner.py` (not in _CORE_MODULES)
#: and `woof/runtime.py` (already excluded above), so leaving them behind
#: strands nothing.
_INGEST_EXCLUDES = {"preflight.py", "nest_spawn_init.py",
                    "relocation_init.py", "relocation_continuation.py",
                    "case_store.py"}
#: `woof/obs/sources.py` is the seam between the ingest lane and the scoring
#: lane: it builds the scorer's dataclasses and reaches
#: `woof.verify.obs.contracts` to do it.  RW-WPS ships no verification
#: package, and that boundary is absolute rather than exemptible, so the one
#: module that crosses it stays behind.  Nothing else in `woof/obs` imports
#: it -- `__init__` pulls radar_grid, superob, sweeps and target_grid -- so
#: the radar front door `woof doctor` checks for is unaffected.
# GOES window acquisition publishes cycle/ensemble manifests. The lower-level
# observation decoders remain available to standalone preprocessing.
# ``aq_airnow.py`` is the air-quality station source of the same scoring
# seam: it builds the scorer's StationObsSet through woof.obs.sources, so it
# stays behind with it (its Rust decoder rw_airnow is a verification tool).
#: `precipitation.py` subclasses `woof.obs.sources._GriddedSource` at module
#: scope, so it crosses the same verification boundary `sources.py` does and
#: stays behind with it.  It is the scorer's precipitation timeline, and its
#: only importer is the observation battery's scoring tool, which this wheel
#: does not stage.
_OBS_EXCLUDES = {"sources.py", "goes_window.py", "precipitation.py", "aq_airnow.py"}
#: The PREPARATION SIDE of ``woof/ensemble``, an allowlist like
#: ``_IO_MODULES`` and for the same reason: the package is mostly member
#: orchestration and the batched forecast, which this wheel exists not to
#: carry.
#:
#: What forces a name here is that ordinary preparation reaches it.  Every
#: streamed preparation asks ``posted_preparation`` whether a posted member
#: input is bound (``woof/ingest/boundary_stream.py``, ``woof/gfs_direct.py``,
#: ``woof/mapped_direct.py``, ``woof/ingest/nest_init.py``), every child
#: preparation asks ``runtime_preparation`` the same about a runtime member
#: (``nest_init.py``), and every namelist import reads ``stochastic`` and
#: ``stochastic_seeds`` through ``woof/namelist_stochastic.py``.  With the
#: package absent each of those is a ModuleNotFoundError in a preparation
#: that binds no member at all, which is how the ensemble line broke this
#: wheel's staging.
#:
#: The rest is the physical member input those doors accept
#: (``--physical-input-store``, ``--physical-output-store``, a posted
#: provider): the store, its field contract, the boundary record a prepared
#: head binds, the source recipes and the three per-source contracts and
#: reuse writers.  Module scope across the set is the standard library plus
#: numpy; every reach into the forecast executor is function-local and has
#: its reason in ``_OPTIONAL_STAGED_IMPORTS``.
#:
#: ``__init__.py`` is staged as the package door.  Its exports are lazy, so
#: importing one of these modules imports none of the orchestration the
#: door also names (``cycle`` and ``engine`` reach the forecast executor).
_ENSEMBLE_MODULES = {
    "__init__.py",
    "gfs_physical_contract.py", "gfs_posted_reuse.py",
    "hrrr_physical_contract.py",
    "mapped_physical_contract.py", "mapped_posted_reuse.py",
    "physical_boundary.py", "physical_fields.py", "physical_store.py",
    "posted_native.py", "posted_physical.py", "posted_preparation.py",
    "member_variants.py", "recipes.py", "runtime_preparation.py", "seeds.py",
    "stochastic.py", "stochastic_seeds.py", "surface_controls.py",
}
#: The only two files of ``woof/io`` this wheel stages -- named
#: individually rather than by excluding the rest of the package,
#: because the package is the forecast executor's output side and the
#: allowed set is the small half.
#:
#: ``woof/doctor.py`` imports ``woof.io.nc_writer_bridge`` to report
#: on the NetCDF writer, and that check RETURNS A BLOCKING ``missing``
#: Check on ImportError.  So the two ways of leaving woof/io out
#: entirely both break something concrete: not staging it at all fails
#: this builder's unresolved-import scan, and allowlisting the import
#: instead ships a wheel whose `rw-wps doctor` always reports the
#: NetCDF writer missing -- on a wheel that BUNDLES the netcdf_writer
#: cdylib as one of its artifacts, and whose own suite states the
#: doctor belongs here precisely because a preprocessing install is
#: the one that needs to be told which bridge is missing.
#:
#: Staging the whole subpackage is what does not work: measured, it
#: cascades to 26 fresh unresolved imports, because ``io/restart.py``
#: reaches gpuwm.core.model/physics/microphysics/kf and
#: woof.supervisor, and ``io/wrfout.py`` reaches woof.runtime and the
#: physics runtimes -- the forecast executor this wheel exists not to
#: carry.  ``nc_writer_bridge.py`` is the seam that does not: module
#: scope is ctypes/os/pathlib/typing plus numpy, and its two internal
#: lookups (woof.bridges, woof.rustwx) are function-local and both
#: already staged.  ``__init__.py`` is empty and comes along only so
#: ``woof.io`` is a package in the wheel.
#:
#: ``classic_tape.py`` is the third, and it is here for the same reason
#: and by the same measurement.  ``woof/wrf_direct.py`` -- the
#: wrfinput/wrfbdy writer, which IS preprocessing and which this wheel
#: therefore stages -- imports ``ClassicTape`` at module scope, so
#: leaving the module behind fails this builder's own unresolved-import
#: scan and the wheel cannot be staged at all.  It does not cascade:
#: module scope is pathlib plus numpy, and its one internal lookup
#: (``woof.io.nc_writer_bridge``) is function-local and already staged.
#: It carries no forecast executor -- it is the shared classic-NetCDF
#: tape both NetCDF products write through, nothing more.
#: The woof.io modules this wheel stages.  An ALLOWLIST, and the
#: comment on ``_FORBIDDEN_STAGED_FILES`` below says why that is a
#: hazard: woof/io is partially staged, so "add one more name" is a
#: mistake somebody can make.  The bar for a name here is that the
#: module is DATA or GRAMMAR -- no CuPy, no netCDF4, no runtime, no
#: forecast executor -- and that something the wheel actually does
#: needs it.
#:
#: ``wrf_output_schema`` and ``history_selection`` joined on 2026-08-20
#: and both clear that bar: the schema module is a transcribed WRF
#: Registry table importing nothing but the standard library, and
#: history_selection is the ``[output]`` block's grammar, importing only
#: it and woof.explain.  What needs them is the CONFIG READER --
#: ``woof.config.load_history_selection`` resolves ``[output]`` at the
#: one load every front door shares, and ``woof.render`` reads the same
#: module's product-input table at module scope.  Without them staged, a
#: preprocessing install could not read a config carrying an ``[output]``
#: block at all.  Neither reaches ``woof.io.wrfout``: the Registry
#: metadata table the vocabulary is built from moved INTO
#: wrf_output_schema the same day, precisely so that reading a config
#: does not import the forecast output writer (and woof.supervisor and
#: netCDF4 behind it).  tests/test_history_selection.py pins that
#: boundary against the artifact.
#:
#: ``classic_product`` is the whole-file classic NetCDF writer the staged
#: woof.obs product writers open by default
#: (``woof.obs.grid_product.open_obs_grid_product``), so
#: ``woof.obs.write_radar_grid``, which the staged ``woof.obs`` exports,
#: needs it.  Missing, that call raised ImportError in this package.
#: Module scope is pathlib, typing and numpy, and its one internal lookup
#: (``woof.io.nc_writer_bridge``) is function-local and staged.
#:
#: ``netcdf_serialization`` holds the deferred ``netCDF4`` handle and the
#: process-wide netCDF4 session lock (D-10, 2.8.8).  The staged
#: ``woof.wrf_direct`` imports it at module scope, so without it this
#: package's staging refused ("unresolved internal imports") and an
#: installed wheel could not import the domain-artifact reader.  Module
#: scope is contextlib and threading; netCDF4 itself loads on first use.
_IO_MODULES = {"__init__.py", "classic_product.py", "classic_tape.py",
               "history_selection.py", "nc_writer_bridge.py",
               "netcdf_serialization.py",
               "wrf_output_schema.py", "sfire_schema.py",
               # Import-free WRF registry data used by input and output schemas.
               "wrf_input_history_schema.py"}
_ROOT_DATA = {
    # woof.fetch_pool sizes its transfer pool from this table at import (2.8.9).
    "fetch-width-rows.json",
    "native_wrf_support_v1.json",
    "physics_params_registry_v1.json",
    "physics_registry_v2.json",
    "wrf_direct_v461_contract.json",
    "terrain_clock_map.json",
    "terrain_clock_map_local_face.json",
    "terrain_clock_adaptive_local_face.json",
}
_TOOL_FILES = {"__init__.py", *HRRR_HELPERS}
_FORBIDDEN_STAGED_FILES = {
    f"woof/{name}" for name in _TOP_LEVEL_EXCLUDES
} | {
    "woof/core/model.py",
    "woof/core/dycore.py",
    "woof/core/physics.py",
    # The forecast executor's OUTPUT side, named here for the same
    # belt-and-braces reason as the three above.  `_IO_MODULES` makes
    # woof/io a partially-staged package, so "add one more name to the
    # allowlist" is now a mistake somebody can make, and these two are
    # what it must not reach: measured, staging them cascades to 26
    # fresh unresolved imports (restart.py reaches woof.core.model /
    # physics / microphysics / kf and woof.supervisor, wrfout.py
    # reaches woof.runtime and the physics runtimes).
    #
    # The unresolved-import scan below would catch that TODAY, so this
    # is redundant today -- deliberately.  The scan only objects while
    # those imports stay at module scope; the tilestream port already
    # broke this wheel exactly once by making the scan blind (see
    # streaming.py in _CORE_MODULES), and a later edit that made
    # wrfout.py's executor imports function-local would ship the
    # forecast output side of a preprocessing wheel with nothing
    # refusing.  This line refuses on the FILE, which no import
    # rearrangement can talk out of.
    "woof/io/restart.py",
    "woof/io/wrfout.py",
}

#: The forecast-only half of woof/ingest/wrfrst.py, shared by its six rows
#: below: apply_warm_start_physics and apply_warm_start_clock overlay a WRF
#: restart file onto a LIVE forecast state, driver and model clock
#: (woof/wrfinput_forecast.py and woof/prepared_domain_tree_forecast.py,
#: neither staged).  Preparation reads and checks a wrfrst
#: (discover_restart_files, read_wrfrst, check_restart_physics) without
#: building a state, so none of these imports runs here.
_WRFRST_LIVE_OVERLAY = ("apply_warm_start_physics and apply_warm_start_clock "
                        "overlay a WRF restart file onto a live forecast "
                        "state; preparation only reads and checks the file")

_OPTIONAL_STAGED_IMPORTS = {
    ("woof/ocean/seaice/__init__.py", "woof.core.physics"):
        "capture_precipitation reads the live microphysics snow ratio inside a "
        "forecast step; preparation stages the module only for its OUTPUT "
        "history table, which woof.io.wrf_output_schema reads at import",
    ("woof/eclipse_windows.py", "woof.core.physics"):
        "EclipseRadiation.apply couples the dimmed shortwave heating into a live "
        "forecast step; preparation reads only the pinned Besselian windows "
        "(eclipses_during) to record the ra_sw_eclipse namelist decision",
    ("woof/earth2_fetch_cache.py", "woof.domain_wizard"):
        "check_domain_coverage sizes a forecast domain in earth2_job; "
        "standalone preparation only receives completed input artifacts",
    ("woof/cyclone_tracks.py", "woof.cyclone_sources"):
        "select_track is a forecast setup operation; preparation only reads "
        "validate_lineage and never selects a tracked forecast",
    ("woof/cyclone_tracks.py", "woof.domain_wizard"):
        "layout and placement_report author forecast domains; preparation "
        "only validates the recorded track lineage",
    ("woof/cyclone_tracks.py", "woof.cyclone_setup"):
        "layout and placement_report author forecast domains; preparation "
        "only validates the recorded track lineage",
    ("woof/earth2_run.py", "woof.domain_wizard"):
        "prepare_launch authors a forecast configuration for the woof go "
        "door, which is excluded from the standalone preparation wheel; "
        "fetch and verification never call this forecast-only function",
    ("woof/certify/pins.py", "woof.supervisor"):
        "the installed-code fingerprint is resolved only when a forecast "
        "capsule asks resolve_pins; the preparation wheel imports the "
        "certification package for CPU compiler identity, without resolving "
        "a forecast's installed-code pins or importing its supervisor",
    ("woof/certify/capsule.py", "woof.supervisor"):
        "forecast capsule code_identity reads the enclosing checkout through "
        "the supervisor. The CPU kernel pack needs certify.compile_platform's "
        "compiler fingerprint, whose package initializer imports capsule, but "
        "standalone preparation emits no forecast certification capsule and "
        "never calls code_identity; importing the compiler helpers does not "
        "import the supervisor",
    ("woof/core/aer_opt2.py", "woof.core.kernels"):
        "forecast-only aer_opt = 2 device optics (optics_device); standalone "
        "configuration only validates the user-aerosol options and values",
    ("woof/ingest/wrfrst.py", "woof.io.restart"):
        "the checkpoint manifest carriers the overlay fills, inside apply_warm_start_physics; "
        + _WRFRST_LIVE_OVERLAY,
    ("woof/ingest/wrfrst.py", "woof.core.physics"):
        "the held-tendency coupling and physics cadence, inside apply_warm_start_physics; "
        + _WRFRST_LIVE_OVERLAY,
    ("woof/ingest/wrfrst.py", "woof.core.physics_inventory"):
        "the PBL composition predicate (apply_warm_start_physics) and the restart-instant REFL_10CM microphysics roster (apply_warm_start_clock); "
        + _WRFRST_LIVE_OVERLAY,
    ("woof/ingest/wrfrst.py", "woof.core.radiation_carriers"):
        "the held radiation carriers, inside apply_warm_start_physics; "
        + _WRFRST_LIVE_OVERLAY,
    ("woof/ingest/wrfrst.py", "woof.core.model"):
        "the model clock's period marker, inside apply_warm_start_clock; "
        + _WRFRST_LIVE_OVERLAY,
    ("woof/ingest/wrfrst.py", "woof.core.refl"):
        "the restart-instant REFL_10CM frame, inside apply_warm_start_clock; "
        + _WRFRST_LIVE_OVERLAY,
    ("woof/ingest/lateral_bc.py", "woof.core.dycore"):
        "WRF_MUTS_SLOT inside _wrf_final_muts_arg, called only by the CUDA "
        "_launch_finalize_field forecast boundary finalizer in strict mode; "
        "preparation reads or writes boundary forcing without stepping a "
        "state boundary or running an acoustic loop",
    ("woof/core/chem_ageing.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/chem_drydep_aerosol.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/chem_dust.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/chem_optics.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/chem_seasalt.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/chem_settling.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/chem_sulfur.py", "woof.core.kernels"):
        "CUDA launch only; preparation reads process allocation rows "
        "without compiling or stepping a chemistry kernel",
    ("woof/core/adaptive_clock.py", "woof.core.physics"):
        "the physics cadence helpers inside _refresh_physics_cadence, "
        "called by the executing adaptive driver. The function returns "
        "before this import without a live physics driver; namelist "
        "translation calls only wrf_num_sound_steps and standalone "
        "preparation advances no forecast clock",
    ("woof/config.py", "woof.ensemble.request"):
        "the [ensemble] table's validator, imported only when a RunConfig "
        "TOML carries the table.  The request reads the batched forecast's "
        "product table (woof.ensemble.batch_products), which this wheel "
        "does not stage.  load_config asks "
        "woof.ensemble.request_installed() first: preparation consumes "
        "nothing from the table, so a preparation-only install leaves it "
        "for the ensemble door to validate where the forecast is installed",
    ("woof/experiment.py", "woof.ensemble.request"):
        "the same [ensemble] validator at the experiment door, behind the "
        "same request_installed() check as woof/config.py above",
    ("woof/core/streaming.py", "woof.ensemble.stochastic_streaming"):
        "the stochastic hook of an executing forecast tile step; "
        "StreamingOptions and config validation reach no step",
    ("woof/stage_cli.py", "woof.ensemble.calibration_admission"):
        "the calibrated-perturbation admission inside sim_main, a forecast "
        "door.  This package's one route into stage_cli is the preparation "
        "handoff, which resolves no forecast (see "
        "woof.prepared_single_domain_forecast below)",
    ("woof/ensemble/posted_physical.py", "woof.ensemble.physical_recenter"):
        "the recentered recipe's member operator, imported inside "
        "PostedPhysicalProvider.prepare only for recipe kind 'recentered'.  "
        "It runs real initialization over a donor population on the card "
        "(woof.ensemble.recentered, woof.ensemble.native_preparation); "
        "direct posted recipes never take the branch",
    ("woof/ensemble/posted_native.py", "woof.prepared_single_domain_forecast"):
        "the ordinary source head preflight a posted provider's member runs "
        "with its original pin (checked_source_inputs).  It is the forecast "
        "runner's preflight and this wheel carries no forecast runner; the "
        "function asks woof.stage_cli.missing_forecast_runners() first and "
        "refuses a provider member here by name.  Sealed physical stores do "
        "not reach it",
    ("woof/ensemble/runtime_preparation.py", "woof.ingest.preflight"):
        "the runtime member's input catalog (runtime_input_catalog and "
        "RuntimePreparation.catalog), called by the forecast runtime's own "
        "preparation, excluded with woof/ingest/preflight.py above.  A "
        "standalone preparation asks only current_runtime_preparation(), "
        "which answers None with no runtime member bound",
    ("woof/ensemble/runtime_preparation.py", "woof.ingest.case_store"):
        "capture_root_inputs seals a runtime member's cold source words, "
        "reached only inside a root build scope the forecast runtime opens",
    ("woof/ensemble/runtime_preparation.py", "woof.runtime"):
        "prepare_root and _restore_root build or restore a runtime member's "
        "root through the forecast runtime, excluded above",
    ("tools/hrrr_single_domain_benchmark.py", "tools.hrrr_posted_reuse"):
        "the shared posted HRRR producer of a provider member, imported only "
        "with --physical-input-provider and beside "
        "woof.ensemble.posted_native.checked_source_inputs, which refuses a "
        "provider member on a preparation-only install by name before the "
        "producer is used",
    ("woof/ingest/prepared_store.py", "woof.core.terrain_drag"):
        "the topo_wind=1 halo refresh of a forecast slab's physics "
        "initialization (CuPy, imported inside the function); standalone "
        "preparation initializes no forecast physics",
    ("woof/config.py", "woof.offline_child_geography"):
        "the [static] parser runs only when load_config admits child_static, "
        "which only the excluded offline child route requests; ordinary "
        "RunConfig loading refuses [static] before reaching this import",
    ("woof/core/chem_plumerise_cache.py", "woof.certify.kernel_manifest"):
        "the kernel manifest record written beside the plume unit's card "
        "compile, which only a running forecast's plume process performs; "
        "a preparation reads the smoke processes' ALLOCATES and never "
        "compiles",
    ("woof/core/chem_vertmx.py", "woof.core.kernels"):
        "the card launch of the vertmx process's step(), which only a "
        "running forecast's chem operator calls; a preparation allocates "
        "the process's arrays and never steps it",
    ("woof/core/streaming.py", "woof.core.preflight"):
        "forecast tree admission and execution estimates; standalone preparation "
        "only reads StreamingOptions and does not call these planners",
    ("woof/stage_cli.py", "woof.prepared_single_domain_forecast"):
        "the forecast runner, imported inside _schema_index (behind "
        "resolve_bundle and resolve_head_bundle), streaming_flags, sim_main "
        "and register_cli.  This package's one route into stage_cli is "
        "woof.prep_output's handoff after a finished preparation, which asks "
        "missing_forecast_runners() first and, with both runners absent "
        "here, prints the preparation-only line and resolves no bundle",
    ("woof/stage_cli.py", "woof.prepared_domain_tree_forecast"):
        "the tree forecast runner, imported inside _schema_index and "
        "sim_main.  Same route and same reason as "
        "woof.prepared_single_domain_forecast directly above: the "
        "preparation handoff asks missing_forecast_runners() before it "
        "resolves a bundle, so a finished preparation here imports neither",
    ("woof/core/streaming.py", "woof.core.pace"):
        "the step-cost sentence of an auto [tiles] refusal "
        "(_refused_tiling_clause), reached only from decide() after "
        "tilestream.autoplan refused a tiling.  decide imports tilestream "
        "first, which this package does not carry, and its one staged "
        "caller prices a downscaled forecast child "
        "(downscale_pricing.price_child).  The clause also catches a failed "
        "import and states the refusal without the cost",
    ("woof/core/streaming.py", "woof.io.restart"):
        "live forecast tile builder inventories restart tracker slots; "
        "standalone preparation constructs no tile stepper",
    ("woof/input_cycle.py", "woof.io.restart"):
        "the checkpoint clock (read_restart_header and "
        "_admissible_elapsed_seconds), imported inside verify only when a "
        "--restart checkpoint is the run's input.  This package resumes no "
        "checkpoint and does not stage the restart reader, so a --cycle "
        "check here reads a prepared bundle's or declared forcing's start",
    ("woof/ingest/stream_resume.py", "woof.io.restart"):
        "the checkpoint clock of `seal-prefix --checkpoint` "
        "(read_restart_header), imported inside main only when a forecast "
        "checkpoint names the prefix to seal (lane/spot-resume).  This "
        "package writes and resumes no checkpoint; a standalone "
        "preparation seals by --checkpoint-seconds, and its staged "
        "callers (fetch_as_posted, hrrr_hierarchy_direct) import only the "
        "posted-marker helpers",
    ("woof/core/streaming.py", "woof.core.streamed_relocation"):
        "forecast-only replacement/adoption of a child store after a move",
    ("woof/core/streaming.py", "woof.core.physics_step_control"):
        "per-step physics cadence in an executing tile; no config path calls it",
    ("woof/core/streaming.py", "woof.core.streamed_state"):
        "running-state publication into a forecast tile store",
    ("woof/core/streaming.py", "woof.core.nest_relocation"):
        "the relocation host-snapshot term of the tree admission, priced "
        "only when a forecast tree carries a moving nest; standalone "
        "preparation decides no tree and moves no nest",
    ("woof/ingest/boundary_stream.py", "woof.runplan"):
        "the interrupt test in _is_stop, reached only from run_chained, "
        "whose callers (woof/go_cli.py and woof/runplan.py) run a "
        "forecast and are not staged; standalone preparation never "
        "chains, so it never calls run_chained",
    ("woof/hrrr_route_inputs.py", "woof.companion_domains"):
        "the doors' WPS namelist renderer (candidate_wps_text) inside "
        "run_route_inputs, reached only when a native hrrr configuration "
        "has no namelist.wps beside it.  run_route_inputs writes the set "
        "a run of `woof go` or `woof run-plan` hands the HRRR chain, "
        "and its only callers are woof/runplan.py's plan resolution and "
        "HRRR chain, excluded above; this wheel plans and chains no run",
    ("woof/ingest/boundary_stream.py", "woof.core.preflight"):
        "the forecast's memory admission (chained_admission) and the host "
        "RAM reader (host_admission), both reached only by a "
        "PreparedTreeWriter that chains, i.e. publishes its head for a "
        "forecast to start on while boundaries are still built.  Its "
        "constructor declines chaining when forecast_installed() is false, "
        "as it is here without woof.core.preflight and woof.core.model, "
        "so an era5, gfs or mapped preparation publishes at its seal and "
        "neither import runs",
    ("woof/ingest/host_decode_window.py", "woof.core.preflight"):
        "the nominal GFS field count (source_analysis_fields_per_time) "
        "inside plan_window, the planner's pre-fetch view that only "
        "woof/go_cli.py's memory line calls; go_cli is excluded above.  "
        "The staged preparation path, woof/gfs_direct.py, imports only "
        "decode_window, available_host_bytes, threads_available and "
        "gfs_decoded_lead_bytes, which price the window from the fetched "
        "files themselves and reach no preflight import",
    ("woof/doctor.py", "woof.core.tempo_rust"):
        "the TEMPO initialization library's doctor line (2.9.0), imported "
        "function-locally inside _tempo_library_check, which returns no "
        "line at all when the seam is absent.  TEMPO is forecast physics: "
        "this wheel stages none of it and bundles no TEMPO library, so the "
        "line belongs only to an install that can run the scheme",
    ("woof/ingest/boundary_stream.py", "woof.core.urban_state"):
        "the BEM workspace count reader in prepared_head_urban_columns; "
        "forecast_installed() is checked before this import, so a "
        "preparation-only installation reads no forecast workspace count "
        "and publishes at the seal",
    ("woof/ingest/boundary_stream.py", "woof.core.devices_memory"):
        "ranked forecast memory admission, reached through PreparedTreeWriter.admit "
        "only after forecast_installed() confirms a forecast executor; a "
        "preparation-only installation returns before pricing any forecast",
    ("woof/ingest/boundary_stream.py", "woof.prepared_domain_tree_forecast"):
        "the prepared tree's land-cover count reader, reused for chained "
        "head admission after forecast_installed() confirms this package "
        "can forecast; standalone preparation has no forecast executor "
        "and returns before importing this reader",
    ("woof/downscale_pricing.py", "woof.core.preflight"):
        "the downscaled child's memory admission (estimate_experiment) "
        "inside price_child, reached only by the downscale door's plan "
        "review and the offline child runner, both forecast routes; the "
        "preparation wheel prices no child",
    ("woof/core/streaming.py", "woof.core.cam_ozone"):
        "live tile physics initialization; pure ozone config dependencies "
        "are staged separately in ozone_contract",
    ("woof/stage_reuse.py", "woof.prepared_single_domain_forecast"):
        "claim_run_output delegates the output-directory refusal to the "
        "forecast runner's own claim_output_directory rather than "
        "restating it; the import is function-local and only a run door "
        "claims an output directory.  This wheel prepares inputs and "
        "claims none",
    ("woof/metem_door.py", "woof.core.preflight"):
        "metgrid_memory_admission is called only by metem_forecast's GPU "
        "run door; metadata and analyzed-field validation use staged contracts",
    ("woof/ingest/hrrr_physics.py", "woof.core.radiation_composition"):
        "GPU physics initialization after the standalone --prepare-only return, "
        "on the same boundary as this module's existing core.physics import",
    ("woof/ingest/wrfinput.py", "woof.core.ruc_runtime"):
        "initialize_wrfinput_physics performs RUC cold initialization only "
        "after importing CuPy and creating the executing physics driver; "
        "host input reading and preparation do not call this function",
    ("woof/ingest/wrfinput.py", "woof.core.physics"):
        "initialize_wrfinput_physics imports CuPy first and initializes an "
        "executing forecast; file reading and host state restoration do not call it",
    ("woof/ingest/wrfinput.py", "woof.core.ruc_runtime"):
        "RUC's cold start (ruc_cold_start, 2.9.0 aa76ba05b) inside "
        "initialize_wrfinput_physics, on the same boundary as the "
        "woof.core.physics import above: it runs on the forecast's "
        "constructed physics driver after the native file's fields are "
        "restored onto it; file reading and host state restoration do not call it",
    ("woof/experiment.py", "woof.spectral_ops.config"):
        "the [spectral_numerics] TABLE PARSER, imported function-locally "
        "and only when a config actually carries the table.  The "
        "subsystem is model runtime and this wheel stages none of it; "
        "experiment.py guards the import and refuses such a config by "
        "name (install the full woof distribution to run spectral "
        "numerics), so a preparation-only install parses every config "
        "it can act on and refuses the one table it cannot.",
    ("woof/physics_menu.py", "woof.domain_wizard"):
        "the WIZARD'S PROSE, and only that: physics_summary() inside "
        "profile_facts() and _radiation_words() inside "
        "nocturnal_remedy().  Their only callers anywhere are "
        "woof/runplan.py (`woof run-plan --physics-profiles`) and "
        "woof/domain_wizard.py itself, and both modules are excluded "
        "above -- a preprocessing wheel plans no run and builds no "
        "domain.  The route a preparation-only install DOES reach this "
        "module on is physics_compat.nocturnal_radiation_refusal() -> "
        "universally_admissible_profile(), which runs at the one "
        "experiment load every front door shares; that path reads "
        "woof.physics_compat and woof.source_adapters, both staged, "
        "and touches neither name above.  It reached the wizard until "
        "2026-08-20, when the suite list, the declared default and "
        "profile_route_blocker moved to woof.physics_compat for "
        "exactly this reason: a table the whole product reads must not "
        "sit behind a door this wheel does not have",
    ("woof/core/kernels/__init__.py", "woof.certify.kernel_manifest"):
        "certification kernel-manifest recording, reached only after the "
        "CuPy import inside the loader; RW-WPS stages no forecast executor "
        "and compiles no CUDA module",
    ("woof/core/kernels/__init__.py", "woof.core.noahmp_kernel_sources"):
        "the one Noah-MP compile site, reached only after the CuPy import "
        "inside the loader and only for a module whose name starts with "
        "noahmp_; RW-WPS stages no forecast executor and compiles no CUDA "
        "module, so the loader never takes that branch here",
    ("woof/core/kernels/__init__.py", "woof.core.fast_sbm_cuda"):
        "FAST_SBM's own direct NVRTC loader (709a2e078), reached only "
        "after the CuPy import inside load_module and only when the name "
        "is fast_sbm; RW-WPS stages no forecast executor and compiles no "
        "CUDA module, so the loader never takes that branch here",
    ("woof/core/nest_interp.py", "woof.certify.kernel_manifest"):
        "certification kernel-manifest recording, reached only after the "
        "CuPy import inside the loader; RW-WPS stages no forecast executor "
        "and compiles no CUDA module",
    ("woof/core/noah_mosaic.py", "woof.certify.kernel_manifest"):
        "certification kernel-manifest recording inside the Noah mosaic "
        "tile loop's own compile (_mosaic_module, _mosaic_ucm_module), "
        "reached only after the CuPy import when a forecast launches the "
        "loop; RW-WPS stages no forecast executor and compiles no CUDA "
        "module.  The module is staged for the doors' tile initialization "
        "and the fit estimator's import-free array inventory",
    ("woof/core/noah_mosaic.py", "woof.core.urban_ucm"):
        "the urban canopy's packed tables and state pointers, imported "
        "function-locally in launch_noah_mosaic's urban arm, which only a "
        "forecast's surface step calls; a preparation-only install never "
        "launches the tile loop",
    ("woof/physics_compat.py", "woof.core.ruc_contract"):
        "RUC'S SOIL GEOMETRY COUNTS, and only those.  The import is "
        "function-local inside pending_wrf_physics_components' "
        "sf_surface_physics == 3 arm (physics_compat.py, the deferred "
        "block), so importing this module in a preprocessing wheel does "
        "not touch it.  It is reached only when a caller asks whether a "
        "RUC forecast column is admissible -- a forecast-side question a "
        "preparation-only install has no arm to answer, and one that "
        "woof.config defers for exactly the same reason.  Staging "
        "woof/core/ruc_contract.py instead would pull "
        "woof.core.noahmp_mynn_contract and woof.ingest.ruc_soil in "
        "behind it for two integer tuples.",
    ("woof/core/state.py", "woof.core.preflight"):
        "CUDA forecast scratch-preflight path; RW-WPS constructs host state",
    ("woof/core/resident_admission.py", "woof.core.preflight"):
        "the loader slab's device estimate (estimate_domain), reached only "
        "by store_from_prepared_cache building a forecast's pinned host "
        "store; standalone preparation builds no store, and the constructor "
        "floor it does reach reads only the staged device_inventory",
    ("woof/core/uh_diag.py", "woof.core.streaming"):
        "the streamed view of the tracking accumulators, reached only from "
        "_zero_domain_slot -- i.e. only when a run RESETS a running max "
        "after publishing a history frame. uh_diag is staged for the slot "
        "NAMES that [relocation.follow] and [[domain]].spawn validation "
        "read while LOADING a config, and no config-validation path calls "
        "the reset. Kept as an OPTIONAL entry after woof.core.streaming "
        "joined _CORE_MODULES, because what it records is that uh_diag "
        "does not need the module at import time and the record is worth "
        "more than the redundancy: the module is staged for the option "
        "surface experiment.py cannot define itself without, and if that "
        "reason ever goes away this line says uh_diag does not supply a "
        "second one",
    ("woof/core/streaming.py", "woof.core.dycore"):
        "the tile sweep's stepper and its stability record, imported inside "
        "the four functions that run a tile: attach, step, and the health "
        "fold over the store. woof.core.streaming is staged for the "
        "[tiles] option surface alone -- the dataclass experiment.py holds "
        "as a field default -- and every route that reaches these imports "
        "is the forecast executor this wheel omits",
    ("woof/core/streaming.py", "woof.core.refl"):
        "the reflectivity stash the streamed history writer consumes, "
        "imported inside the publish path. Same reason as woof.core.dycore "
        "directly above: no config-loading path reaches it",
    ("woof/core/streaming.py", "woof.core.chem_history"):
        "completed forecast chemistry history, imported by the live tile "
        "history hook; standalone preparation reads StreamingOptions and "
        "does not publish a running tile's history",
    ("woof/core/streaming.py", "woof.sfire_debug"):
        "fire-only diagnostic forecast sweeps, imported only when the "
        "live streamed runner executes fire_test_steps; preparation "
        "validates tile options without stepping a fire domain",
    ("woof/ingest/wrfinput.py", "woof.core.chem_sfire"):
        "native smoke conversion in initialize_wrfinput's forecast state "
        "constructor; preparation reads raw fire-static and smoke fields "
        "without constructing the coupled CUDA forecast",
    ("woof/core/streaming.py", "woof.core.nest_stream"):
        "the NEST streamed transport, imported inside three functions that "
        "each require a forecast: prepared_domain_builder.build wires the "
        "tile hook only for a domain whose decision STREAMS, _tree_reservations "
        "prices a live tree's coupling corridors against a VRAM budget, and "
        "steppers_for_tree hands its result to "
        "woof.core.model.execute_experiment -- a module _FORBIDDEN_STAGED_FILES "
        "refuses outright. This is the boundary the _CORE_MODULES entry for "
        "streaming.py already draws in its own words: that file is staged for "
        "the [tiles] OPTION SURFACE and NOT the streamed transport. Staging "
        "nest_stream would cross it, and would pay for the crossing twice -- "
        "the module's own nested imports reach woof.core.preflight, which "
        "this wheel deliberately omits, so admitting it would need a second "
        "exception to cover the first. Nothing staged here needs a NAME from "
        "it: experiment.py cannot define StreamingOptions without streaming.py, "
        "but no class definition and no config-validation path reaches "
        "nest_stream, and its only other referrer, woof/core/nest.py, is not "
        "staged at all",
    ("woof/core/streaming.py", "woof.core.prepared_tile_memory"):
        "prepared forecast VRAM pricing, imported function-locally inside "
        "radiation_footprint after tilestream.autoplan. RW-WPS stages "
        "streaming.py for StreamingOptions and config validation; no "
        "standalone preprocessing path calls the forecast footprint planner",
    ("woof/core/streaming.py", "woof.state_digest"):
        "the canonical end-of-run digest taken from the STORE, imported "
        "inside StreamedDomain.canonical_digest. woof/state_digest.py is on "
        "_TOP_LEVEL_EXCLUDES above by deliberate choice, and what reaches it "
        "here is evidence a run records after it has integrated: the method "
        "refuses outright on a domain carrying no scalars. A preprocessing "
        "wheel steps nothing and so digests nothing",
    ("woof/hrrr_prepared_bundle.py",
     "woof.prepared_single_domain_forecast"):
        "physics-selection helpers on the opt-in portable-bundle branch. "
        "publish_hrrr_prepared_bundle runs only when prepare_hrrr_wrf is "
        "given --wps-namelist, and the three documents it writes exist "
        "for a config-driven FORECAST stage to bind -- which is exactly "
        "what this preprocessing wheel does not ship. The certified "
        "native preparation above that branch never reaches it",
    ("tools/hrrr_single_domain_benchmark.py", "woof.core.clock"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.core.dycore"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.core.health"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.core.model"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.core.refl"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.io.wrfout"):
        "forecast-only output branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.state_digest"):
        "forecast-only evidence branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "woof.runtime"):
        "forecast-only physics setup after the public --prepare-only "
        "return: declared_constant_glw supplies initialize_hrrr_physics "
        "with the declared downward longwave, and nothing on the "
        "--prepare-only path calls it. The tool itself cannot be dropped "
        "instead -- it is a member of native_wrf_distribution.HRRR_HELPERS, "
        "so the sealed-runtime receipt refuses an installation that is "
        "missing it",
    ("woof/ingest/hrrr_physics.py", "woof.core.physics"):
        "forecast-only physics setup after the public --prepare-only return",
    # 399b3c5abe seeds the f00 cloud fraction with the radiation's own
    # cal_cldfra / legacy cloud-fraction flags, in the same function as the
    # woof.core.physics._prepare_atmosphere import directly above, so it is
    # the same forecast-only branch.  Followed, the closure staged
    # woof.core.rrtmgp, whose RadiationResult import reaches the forbidden
    # woof.core.physics and refused the whole staging.
    ("woof/ingest/hrrr_physics.py", "woof.core.rrtmgp"):
        "forecast-only f00 cloud-fraction seeding beside the "
        "woof.core.physics import of the same function",
    ("woof/ingest/hrrr_physics.py", "woof.core.rrtmg_legacy"):
        "forecast-only f00 cloud-fraction seeding beside the "
        "woof.core.physics import of the same function",
    # 8c8de91ad6 (restored in 2.9.4): the mapped route asks the standby
    # door for a prestaged boundary handle; the standalone package has no
    # standby door, and mapped_direct takes the ImportError as "none".
    ("woof/mapped_direct.py", "woof.boundary_prestage"):
        "standby's prestaged boundary handle; absent in the preparation "
        "package, where mapped_direct reads it as no prestage",
    ("woof/source_availability.py", "woof.domain_wizard"):
        "availability() prices the download a run of a given length makes with "
        "the wizard's fetch_window; it answers the web page's calendar and the "
        "case catalog, neither staged here. The module stays because "
        "woof/source_adapters.py reads ArchiveWindow from it at import",
    ("woof/static/corridor.py", "woof.core.nest_relocation"):
        "the older-corridor check inside load_child_statics_corridor, which "
        "compares a corridor sealed under an earlier build contract with "
        "the tree's own child statics at the prepared placement. Its only "
        "caller is the prepared-tree forecast runner "
        "(woof/prepared_domain_tree_forecast.py, excluded above) when it "
        "loads a corridor for a moving nest; preparation seals corridors "
        "under the current contract and loads none. The module stays "
        "because woof/source_hierarchy.py and "
        "woof/hrrr_hierarchy_direct.py build and seal corridors through "
        "it at module scope",
    ("woof/static/corridor.py", "woof.ingest.relocation_init"):
        "the overlap rule the same older-corridor check reads, beside "
        "woof.core.nest_relocation directly above and on the same "
        "forecast-only route; relocation_init.py is a forecast-time child "
        "initializer and stays out with _INGEST_EXCLUDES",
    ("woof/ingest/init_perturbation.py", "woof.core.rrtmgp"):
        "radiation_temperature_ceiling reads the RTE+RRTMGP gas-table span "
        "when a forecast runner builds the initial-state perturbation "
        "(woof/runtime.py, the prepared-tree runner and a nest's "
        "initialization in woof/core/model.py, none staged here); "
        "preparation never builds one. The module ships with the rest of "
        "woof/ingest, and woof/ingest/nest_init.py imports it on that same "
        "forecast-only branch",
    ("woof/fetch.py", "tools.download_gfs_native_subset"):
        "the GFS subset transport (the NOMADS crop and whole-object "
        "downloads, their level ladder and record counts), imported inside "
        "the locked bodies of fetch_gfs and fetch_gfs_fullfile and five level "
        "and record helpers.  Their callers are the `woof fetch` door "
        "(fetch_main and the route fetch under it, registered only by "
        "woof/cli.py, excluded above), woof/core/preflight.py (not staged) "
        "and woof/download_budget.py (excluded above).  This package "
        "has no fetch door: rw-wps --source gfs prepares a series already "
        "fetched and authors its front-door manifest from the fetch receipt "
        "on disk, which reads no transport",
}


def _names_verify(module: str) -> bool:
    return module == "woof.verify" or module.startswith("woof.verify.")


def _staged_verification_imports(destination: Path, *, scan=None) -> list[dict[str, object]]:
    """Return developer-only ``woof.verify`` imports in staged runtime code.

    RW-WPS intentionally excludes the verification package.  Scanning every
    staged Python module closes the package boundary at build time instead of
    relying on a hand-maintained list of public entry points.  Direct imports
    and literal dynamic imports are both rejected.
    """
    if scan is None:
        with _StagedImportScan() as scan:
            return _staged_verification_imports(destination, scan=scan)
    scan.refresh(destination)
    violations: list[dict[str, object]] = []
    for source in sorted(destination.rglob("*.py")):
        for node in scan.facts[source][1]:
            candidates: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                candidates.extend((alias.name, "import") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                candidates.append((node.module, "from"))
                # `from woof import verify` imports woof.verify.
                if not _names_verify(node.module):
                    candidates.extend(
                        (f"{node.module}.{alias.name}", "from")
                        for alias in node.names if alias.name != "*")
            elif isinstance(node, ast.Call) and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    function = node.func
                    dynamic = (
                        isinstance(function, ast.Name)
                        and function.id == "__import__"
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    )
                    if dynamic:
                        candidates.append((argument.value, "dynamic"))
            for module, kind in candidates:
                if _names_verify(module):
                    violations.append({
                        "path": source.relative_to(destination).as_posix(),
                        "line": int(getattr(node, "lineno", 0)),
                        "module": module,
                        "kind": kind,
                    })
    return violations


def _is_source_module(source_root: Path, module: str) -> bool:
    """Is ``module`` a module or a regular package in the source tree?"""

    path = source_root.joinpath(*module.split("."))
    return (path.with_suffix(".py").is_file()
            or (path / "__init__.py").is_file())


def _complete_staged_import_closure(destination: Path, derived_modules=None, *, scan=None) -> list[str]:
    """Stage required package imports to a fixed point without importing code.

    The existing manifest defines the preprocessing doors and their optional
    forecast boundaries. New required modules follow their imports, including
    imports inside functions and package aliases. An excluded executor, missing
    source or verification dependency still refuses the build.
    """
    if scan is None:
        with _StagedImportScan() as scan:
            return _complete_staged_import_closure(destination, derived_modules, scan=scan)
    derived_modules = set() if derived_modules is None else derived_modules
    added: list[str] = []
    requested_imports = _staged_internal_imports(destination, source_root=REPO, scan=scan)
    while missing := _staged_internal_imports(
            destination, source_root=REPO, derived_modules=derived_modules, scan=scan):
        pending: dict[str, Path] = {}
        for record in missing:
            module = str(record["module"])
            parts = module.split(".")
            for count in range(1, len(parts) + 1):
                path = REPO.joinpath(*parts[:count])
                source = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
                relative = source.relative_to(REPO).as_posix()
                if relative in _FORBIDDEN_STAGED_FILES or _names_verify(module):
                    raise RuntimeError(
                        "RW-WPS wheel staging has unresolved internal imports: "
                        f"excluded executor or verification module: {record}; "
                        f"required by {requested_imports}")
                if not source.is_file():
                    raise RuntimeError(
                        "RW-WPS wheel staging has unresolved internal imports: "
                        f"no package source: {record}; required by {requested_imports}")
                if not (destination / relative).is_file():
                    pending[relative] = source
        if not pending:
            raise RuntimeError(f"RW-WPS import closure made no progress: {missing}")
        for relative, source in sorted(pending.items()):
            _copy_source(source, destination / relative)
            added.append(relative)
            if source.name != "__init__.py":
                derived_modules.add(".".join(Path(relative).with_suffix("").parts))
    return added


def _analyze_staged_source(item):
    """Parse one content version and retain compact import facts, in a worker."""
    source, text, owner, is_package, keep_tree = item
    tree = ast.parse(text, filename=str(source))
    nodes = []
    type_checking_nodes = set()
    incoming = {}
    package = owner if is_package else owner.rpartition(".")[0]

    def retain(module, names=None):
        if module not in incoming:
            incoming[module] = names
        elif incoming[module] is not None:
            if names is None:
                incoming[module] = None
            else:
                incoming[module].update(names)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        if isinstance(node, ast.If):
            marker = node.test
            if ((isinstance(marker, ast.Name) and marker.id == "TYPE_CHECKING")
                    or (isinstance(marker, ast.Attribute) and marker.attr == "TYPE_CHECKING")):
                for statement in node.body:
                    type_checking_nodes.update(ast.walk(statement))
        if isinstance(node, ast.Import):
            for alias in node.names:
                retain(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = node.module or ""
            if node.level:
                parts = package.split(".")
                resolved = ".".join(parts[:len(parts) - node.level + 1]
                                    + ([resolved] if resolved else []))
            retain(resolved, None if any(alias.name == "*" for alias in node.names)
                   else {alias.name for alias in node.names})
            for alias in node.names:
                retain(f"{resolved}.{alias.name}")
        elif isinstance(node, ast.Call) and node.args:
            argument = node.args[0]
            # The symbol closure deliberately treats ANY literal module lookup
            # conservatively, not only import_module and __import__ calls.
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                retain(argument.value)
                function = node.func
                if ((isinstance(function, ast.Name) and function.id == "__import__")
                        or (isinstance(function, ast.Attribute)
                            and function.attr == "import_module")):
                    nodes.append(node)
    # Only derived modules need function bodies for symbol reachability. Sending
    # every original AST back would serialize the stage on Python unpickling.
    return (tree if keep_tree else None, nodes,
            type_checking_nodes.intersection(nodes), incoming)


class _StagedImportScan:
    """One stage's content cache and parallel parser, never shared across stages.

    Only added or changed files are parsed. Importers can change a derived
    module's reachable symbols at any iteration, so every pass rechecks the
    compact facts against the current module set and importer index.
    """

    def __init__(self):
        self.contents = {}
        self.facts = {}
        self.sources = []
        self.incoming = {}
        self.source_modules = {}
        self.unreached = {}
        self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown()

    def refresh(self, destination, derived_modules=()):
        sources = []
        changed = []
        for source in sorted(destination.rglob("*.py")):
            parts = list(source.relative_to(destination).with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            owner = ".".join(parts)
            sources.append((source, owner, is_package))
            text = source.read_text(encoding="utf-8")
            keep_tree = owner in (derived_modules or ())
            if (self.contents.get(source) != text
                    or (keep_tree and self.facts[source][0] is None)):
                self.contents[source] = text
                changed.append((source, text, owner, is_package, keep_tree))
        if changed:
            if self.pool is None and len(changed) > 1:
                cores = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                         else os.cpu_count() or 1)
                workers = min(cores, len(changed))
                if os.name == "nt":
                    # ProcessPoolExecutor's Windows wait API admits at most 61 workers.
                    workers = min(workers, 61)
                # spawn on every platform: forkserver binds an AF_UNIX socket
                # under the temporary directory, and a long TMPDIR (CI runner
                # work trees, HPC scratch paths) exceeds the 107-byte socket
                # path limit ("AF_UNIX path too long").
                self.pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"))
            results = (self.pool.map(_analyze_staged_source, changed)
                       if self.pool is not None else map(_analyze_staged_source, changed))
            for item, facts in zip(changed, results):
                self.facts[item[0]] = facts
        self.sources = [(source, owner, package) for source, owner, package in sources if owner]
        self.incoming = {}
        for source, owner, is_package in self.sources:
            relative = owner.replace(".", "/") + ("/__init__.py" if is_package else ".py")
            for module, names in self.facts[source][3].items():
                if (relative, module) in _OPTIONAL_STAGED_IMPORTS:
                    continue
                if module not in self.incoming:
                    self.incoming[module] = None if names is None else set(names)
                elif self.incoming[module] is not None:
                    if names is None:
                        self.incoming[module] = None
                    else:
                        self.incoming[module].update(names)

    def unreached_function_nodes(self, module, tree):
        """Cache reachability by AST identity and the current imported symbols."""
        imported = self.incoming.get(module)
        key = (id(tree), module, None if imported is None else frozenset(imported))
        hit = self.unreached.get(key)
        if hit is None or hit[0] is not tree:
            hit = (tree, _unreached_function_nodes(module, tree, self.incoming))
            self.unreached[key] = hit
        return hit[1]

    def is_source_module(self, source_root, module):
        key = (source_root, module)
        if key not in self.source_modules:
            self.source_modules[key] = _is_source_module(source_root, module)
        return self.source_modules[key]


def _unreached_function_nodes(module, tree, incoming):
    """Functions outside an added module's imported symbol closure.

    A module import, star import or dynamic module lookup is conservative and
    retains every function. Named imports retain module initialization and
    all directly referenced local functions. Source bytes remain unchanged.
    """
    imported = incoming.get(module)
    if imported is None:
        return set()
    names = set(imported)
    if not names:
        return set()
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    active = [node for node in tree.body if node not in definitions.values()]
    for definition in definitions.values():
        active.extend(definition.decorator_list)
        if isinstance(definition, ast.ClassDef):
            active.extend(definition.bases)
            active.extend(definition.keywords)
            active.extend(node for node in definition.body if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)))
            for method in definition.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    active.extend(method.decorator_list)
                    active.extend(method.args.defaults)
                    active.extend(value for value in method.args.kw_defaults
                                  if value is not None)
        else:
            active.extend(definition.args.defaults)
            active.extend(value for value in definition.args.kw_defaults if value is not None)
    for name in names & definitions.keys():
        active.append(definitions.pop(name))
    while active:
        node = active.pop()
        for used in ast.walk(node):
            if isinstance(used, ast.Call) and isinstance(used.func, ast.Name) and used.func.id in {
                    "globals", "locals", "eval", "exec"}:
                return set()
            if isinstance(used, ast.Name):
                names.add(used.id)
        for name in names & definitions.keys():
            active.append(definitions.pop(name))
    unreached = set()
    for definition in definitions.values():
        functions = ([definition] if not isinstance(definition, ast.ClassDef)
                     else [node for node in ast.walk(definition)
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
        for function in functions:
            for statement in function.body:
                unreached.update(ast.walk(statement))
    return unreached


def _staged_internal_imports(
    destination: Path, *, source_root: Path, optional: bool = False,
    derived_modules=None, scan=None,
) -> list[dict[str, object]]:
    """Return imports whose internal module is absent from wheel staging.

    ``from package import name`` imports ``package.name`` as a module
    whenever the source tree has one by that name, so each such alias is
    checked as well as ``package``: ``from woof.core import host_libm``
    needs ``woof/core/host_libm.py`` staged, not only
    ``woof/core/__init__.py``.  Read as an import of ``woof.core`` alone
    it passed this scan while host_libm was not staged, and the staged
    map projections could not be imported.  Only the source
    tree (``source_root``) can say whether a name is a submodule or an
    attribute of its package, which is why the scan is given one.
    """
    if scan is None:
        with _StagedImportScan() as scan:
            return _staged_internal_imports(
                destination, source_root=source_root, optional=optional,
                derived_modules=derived_modules, scan=scan)
    scan.refresh(destination, derived_modules)
    sources = scan.sources
    modules = {module for _, module, _ in sources}
    violations: list[dict[str, object]] = []
    for source, current_module, is_package in sources:
        tree, nodes, type_checking_nodes, _ = scan.facts[source]
        unreached = (scan.unreached_function_nodes(current_module, tree)
                     if current_module in (derived_modules or ()) else set())
        current_package = (
            current_module if is_package else current_module.rpartition(".")[0]
        )
        for node in nodes:
            if node in type_checking_nodes:
                continue
            candidates: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                candidates.extend((alias.name, "import") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = current_package.split(".") if current_package else []
                    ascend = node.level - 1
                    if ascend > len(package_parts):
                        resolved = ""
                    else:
                        base = package_parts[:len(package_parts) - ascend]
                        if node.module:
                            base.extend(node.module.split("."))
                        resolved = ".".join(base)
                else:
                    resolved = node.module or ""
                if resolved:
                    candidates.append((resolved, "from"))
                    # A missing package is reported as itself; its
                    # submodules cannot be staged without it.
                    if resolved in modules:
                        candidates.extend(
                            (f"{resolved}.{alias.name}", "from")
                            for alias in node.names
                            if alias.name != "*" and scan.is_source_module(
                                source_root, f"{resolved}.{alias.name}"))
            elif isinstance(node, ast.Call) and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    function = node.func
                    dynamic = (
                        isinstance(function, ast.Name)
                        and function.id == "__import__"
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    )
                    if dynamic:
                        candidates.append((argument.value, "dynamic"))
            for module, kind in candidates:
                root = module.partition(".")[0]
                if root not in {"woof", "tools"}:
                    continue
                # The folded hex model is a component beside the preparation package, as gpuwm-hex is to
                # the engine: its lazy doors (the woof hex store in woof/output_formats.py) never reach it.
                if module == "woof.hex" or module.startswith("woof.hex."):
                    continue
                if module not in modules:
                    relative_path = source.relative_to(destination).as_posix()
                    reason = _OPTIONAL_STAGED_IMPORTS.get(
                        (relative_path, module))
                    if reason is None and node in unreached:
                        reason = (
                            "function outside the named imports from staged preprocessing "
                            "modules; module imports and dynamic lookup retain all functions")
                    if (reason is not None) != optional:
                        continue
                    record = {
                        "path": relative_path,
                        "line": int(getattr(node, "lineno", 0)),
                        "module": module,
                        "kind": kind,
                    }
                    if reason is not None:
                        record["optional_reason"] = reason
                    violations.append(record)
    return violations

#: The token :func:`_standalone_pyproject` replaces with this release.
_STANDALONE_VERSION_PLACEHOLDER = "0.0.0+placeholder"

#: The standalone RW-WPS wheel's own pyproject.  Its version is a
#: placeholder that :func:`_standalone_pyproject` fills in from the
#: running package, because it is not free to be anything else: the
#: sealed-runtime receipt refuses unless the installed ``rw-wps``
#: distribution version equals ``woof.__version__``
#: (``native_wrf_distribution._installed_record_receipt``).  It read
#: ``0.1.1`` and agreed with the stale package constant by coincidence;
#: the moment that constant started telling the truth, a hardcoded
#: version here would have failed the very seal it feeds.
_STANDALONE_PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rw-wps"
version = "0.0.0+placeholder"
description = "Native parallel preprocessing and stock-WRF initialization"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "netCDF4>=1.6", "threadpoolctl>=3.1"]
keywords = ["WRF", "WPS", "GRIB", "NetCDF", "weather"]
classifiers = ["License :: OSI Approved :: Apache Software License"]

[tool.setuptools]
license-files = ["LICENSE", "NOTICE", "licenses/LICENSE-rust-libm.txt", "licenses/LICENSE-FDLIBM-SunPro.txt"]

[project.optional-dependencies]
# [ctk] and the 14.0 floor for the same measured reason as the parent
# project's own GPU extras (see pyproject.toml): a CuPy wheel carries the
# compiler and no CUDA headers, so on a driver-only box every device call
# dies with "Failed to find CUDA headers", and [ctk] -- a CuPy 14 extra --
# is what supplies a matching toolkit as wheels.
gpu = ["cupy-cuda12x[ctk]>=14.0"]
geog = ["rasterio>=1.3", "pyproj>=3.6"]

[project.scripts]
rw-wps = "woof.source_cli:main"
gpuwm-sfire-static = "woof.static.sfire:main"
woof-wrf-init = "woof.source_cli:main"
woof-wrf-runtime-check = "woof.native_wrf_distribution:main"
woof-mapped-inspect = "woof.mapped_source:main"

[tool.setuptools.packages.find]
include = ["woof*", "tools"]

[tool.setuptools.package-data]
woof = [
  "native_wrf_support_v1.json",
  "physics_params_registry_v1.json",
  "physics_registry_v2.json",
  "wrf_direct_v461_contract.json",
  "terrain_clock_map.json",
  "terrain_clock_map_local_face.json",
  "terrain_clock_adaptive_local_face.json",
  "authorities/*.json",
  "core/kernels/*.cu",
  "core/kernels/*.cuh",
  "data/noah_tables/*.TBL",
  "data/noah_tables/*.md",
  "data/surface_ocean.json",
  "data/output_formats.json",
  "data/chem/*.json",
  "data/chem/*.md",
  "data/chem/species/*.json",
  "data/chem/sets/*.json",
  "data/chem/sources/*.json",
  "data/chem/diagnostics/*.json",
  "data/chem/wesely/*.json",
  "data/chem/wesely/*.md",
  "data/thompson/fork-build/*.F90",
  "data/wsm_family/schemes.json",
]
tools = ["*.sh"]
"""


def _standalone_pyproject() -> str:
    """The standalone wheel's pyproject, stamped with THIS release.

    A plain substitution rather than ``str.format``: the template is
    TOML and carries braces of its own (``license = { file = ... }``).
    """

    from woof import __version__

    stamped = _STANDALONE_PYPROJECT.replace(
        _STANDALONE_VERSION_PLACEHOLDER, __version__)
    if _STANDALONE_VERSION_PLACEHOLDER in stamped:
        raise RuntimeError(
            "the standalone pyproject template lost its version placeholder")
    return stamped


class NotAGitCheckout(RuntimeError):
    """The tree cannot answer whether a file is tracked.

    Distinct from "this file is untracked", which is a refusal.  A
    snapshot extracted from a tarball or a `git archive` has no index at
    all, so the question has no answer there -- and the two are not the
    same finding.  Conflating them cost a Linux gate run a 40-line
    `CalledProcessError` traceback ending in `exit status 128`, which
    reads as a staging bug rather than as "this directory is not a
    checkout".
    """


def _tree_is_a_git_checkout() -> bool:
    """Does ``REPO`` have a git index to ask?  Asked once, cached."""

    global _GIT_CHECKOUT
    if _GIT_CHECKOUT is None:
        probe = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--git-dir"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _GIT_CHECKOUT = probe.returncode == 0
    return _GIT_CHECKOUT


#: Memo for :func:`_tree_is_a_git_checkout` (None = not yet asked).
_GIT_CHECKOUT: bool | None = None


def _require_tracked(source: Path) -> None:
    relative = source.resolve().relative_to(REPO).as_posix()
    if not _tree_is_a_git_checkout():
        raise NotAGitCheckout(
            f"{REPO} is not a git checkout, so whether {relative} is "
            "tracked cannot be answered.  RW-WPS wheel staging copies "
            "tracked files only; run it from a clone (CI's "
            "actions/checkout leaves one), not from an extracted "
            "archive")
    if relative not in _tracked_source_paths(REPO):
        raise subprocess.CalledProcessError(
            1, ["git", "-C", str(REPO), "ls-files", "--error-unmatch", relative])


def _copy_source(source: Path, destination: Path) -> None:
    _require_tracked(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _stage_rw_wps_python_project(destination: Path) -> dict[str, object]:
    """Create the source-matched, forecast-executor-free wheel project."""

    destination.mkdir()
    package = destination / "woof"
    for source in sorted((REPO / "woof").glob("*.py")):
        if source.name not in _TOP_LEVEL_EXCLUDES:
            _copy_source(source, package / source.name)
    for name in sorted(_ROOT_DATA):
        _copy_source(REPO / "woof" / name, package / name)
    for subpackage, excludes in (
        ("ingest", _INGEST_EXCLUDES),
        # Radar observation ingest.  `woof doctor` checks for the NEXRAD
        # front door by name, unconditionally, so that a green report can
        # never mean "radar is fine" on a machine that cannot read a radar
        # volume -- and that check imports `woof.obs.nexrad`.  Staging the
        # subpackage is what keeps the check unconditional here too: making
        # the import lazy instead would reintroduce, in a new place, exactly
        # the silent-green hole the check exists to close.  It costs nothing
        # a preprocessing wheel should refuse -- the whole subpackage is
        # stdlib plus numpy, imports only `woof.bridges` and
        # `woof.static.projection` (both already staged), and contains no
        # CuPy and no forecast executor.  Turning radar bytes into gridded
        # observation files is preprocessing, which is what this wheel is.
        ("obs", _OBS_EXCLUDES),
        ("static", set()),
        # The rental time ledger (2.8.9): fetch, geography, table and
        # composition stages emit timing records through it. Stdlib only;
        # the waterfall's matplotlib import is function-local.
        ("timeline", set()),
    ):
        for source in sorted((REPO / "woof" / subpackage).glob("*.py")):
            if source.name not in excludes:
                _copy_source(source, package / subpackage / source.name)
    for name in sorted(_IO_MODULES):
        _copy_source(
            REPO / "woof" / "io" / name,
            package / "io" / name,
        )
    for name in sorted(_ENSEMBLE_MODULES):
        _copy_source(
            REPO / "woof" / "ensemble" / name,
            package / "ensemble" / name,
        )
    for name in sorted(_CORE_MODULES):
        _copy_source(
            REPO / "woof" / "core" / name,
            package / "core" / name,
        )
    _copy_source(
        REPO / "woof" / "core" / "kernels" / "__init__.py",
        package / "core" / "kernels" / "__init__.py",
    )
    for name in CUDA_KERNEL_SOURCES:
        _copy_source(
            REPO / "woof" / "core" / "kernels" / name,
            package / "core" / "kernels" / name,
        )
    for source in sorted((REPO / "woof" / "authorities").glob("*.json")):
        _copy_source(source, package / "authorities" / source.name)
    for source in sorted((REPO / "woof" / "data" / "noah_tables").iterdir()):
        if source.is_file():
            _copy_source(
                source,
                package / "data" / "noah_tables" / source.name,
            )
    _copy_source(REPO / "woof/data/wsm_family/schemes.json",
                 package / "data/wsm_family/schemes.json")
    # Configuration loads the ocean selector/default table before any
    # forecast allocation, including in the preparation-only package.
    _copy_source(REPO / "woof" / "data" / "surface_ocean.json",
                 package / "data" / "surface_ocean.json")
    # The output format table (Icechunk, 2.9.10): woof.io.history_selection
    # reads it through woof.output_formats at import, so every config load
    # does; without it the staged package could not load a configuration.
    _copy_source(REPO / "woof" / "data" / "output_formats.json",
                 package / "data" / "output_formats.json")
    # The ocean product rows (2.9.4 ocean lanes): woof.io.wrf_output_schema
    # merges them into the history vocabulary at import, so reading any
    # config with an [output] block reads this file.
    _copy_source(REPO / "woof" / "ocean" / "rows.json",
                 package / "ocean" / "rows.json")
    # The lake state and lake ice source tables (cap-lake, 2.9.4):
    # woof.config.validate_run_config reads the lake-state rows on every
    # run config, and woof.ingest.lake_ice reads the ice rows at preparation.
    for source in sorted((REPO / "woof" / "data" / "surface_obs").glob("*.toml")):
        _copy_source(source, package / "data" / "surface_obs" / source.name)
    # The chem table (woof.chem_table, staged with the top-level modules):
    # its schema, rows, sets, sources and diagnostics, which a chem
    # preparation loads by name.  Not the oracle fixtures, which only the
    # engine's tests read.
    chem_root = REPO / "woof" / "data" / "chem"
    for pattern in ("*.json", "*.md", "species/*.json", "sets/*.json",
                    "sources/*.json", "diagnostics/*.json",
                    "wesely/*.json", "wesely/*.md"):
        for source in sorted(chem_root.glob(pattern)):
            if source.is_file():
                _copy_source(source, package / "data" / "chem"
                             / source.relative_to(chem_root))
    # Fork tables are generated from pinned public inputs at actual first use.
    # Ship only the small original CPU harness, never the 345 MB coefficient set.
    for source in sorted((REPO / "woof" / "data" / "thompson" / "fork-build").glob("*.F90")):
        _copy_source(source, package / "data" / "thompson" / "fork-build" / source.name)
    for name in sorted(_TOOL_FILES):
        _copy_source(REPO / "tools" / name, destination / "tools" / name)
    _copy_source(REPO / "README.md", destination / "README.md")
    _copy_source(REPO / "LICENSE", destination / "LICENSE")
    _copy_source(REPO / "NOTICE", destination / "NOTICE")
    for name in ("LICENSE-rust-libm.txt", "LICENSE-FDLIBM-SunPro.txt"):
        _copy_source(REPO / "licenses" / name, destination / "licenses" / name)
    (destination / "pyproject.toml").write_text(
        _standalone_pyproject(),
        encoding="utf-8",
        newline="\n",
    )

    with _StagedImportScan() as scan:
        derived_modules = set()
        import_closure_files = _complete_staged_import_closure(destination, derived_modules, scan=scan)
        files = sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*") if path.is_file()
        )
        forbidden = sorted(_FORBIDDEN_STAGED_FILES & set(files))
        if forbidden:
            raise RuntimeError(
                f"RW-WPS wheel staging contains forecast executors: {forbidden}"
            )
        verification_imports = _staged_verification_imports(destination, scan=scan)
        if verification_imports:
            raise RuntimeError(
                "RW-WPS wheel staging imports omitted developer verification "
                f"modules: {verification_imports}"
            )
        internal_imports = _staged_internal_imports(
            destination, source_root=REPO, derived_modules=derived_modules, scan=scan)
        if internal_imports:
            raise RuntimeError(
                "RW-WPS wheel staging has unresolved internal imports: "
                f"{internal_imports}"
            )
        optional_internal_imports = _staged_internal_imports(
            destination, source_root=REPO, optional=True, derived_modules=derived_modules, scan=scan)
        return {
            "distribution": PYTHON_DISTRIBUTION,
            "file_count": len(files),
            "files": files,
            "import_closure_files": import_closure_files,
            "forecast_executor_files": forbidden,
            "verification_imports": verification_imports,
            "unresolved_internal_imports": internal_imports,
            "optional_internal_imports": optional_internal_imports,
        }


def _run(
    argv: list[str], *, cwd: Path = REPO,
    env: dict[str, str] | None = None,
) -> None:
    # ``main`` reserves stdout for its one machine-readable JSON receipt.
    # pip and Cargo both write progress to stdout, so inheriting their streams
    # produced a file that looked like a receipt but could not be parsed as
    # JSON.  Preserve all build diagnostics on stderr while keeping stdout a
    # strict, single-document interface suitable for atomic capture.
    subprocess.run(
        argv, cwd=cwd, env=env, check=True,
        stdout=sys.stderr, stderr=sys.stderr,
    )


def _cargo_release_environment(
    *, source_root: Path, target_dir: Path, source_date_epoch: str,
) -> dict[str, str]:
    """Return a path-stable environment for the sealed Rust build.

    Rust dependencies can encode absolute source filenames in panic-location
    tables even when application code never prints them.  A release assembled
    from a different clean checkout would then have different bridge hashes.
    Use Cargo's encoded flag channel so checkout paths containing whitespace
    are handled as one rustc argument, and discard caller-provided rust flags
    that would otherwise make the sealed artifact host-dependent.
    """
    environment = os.environ.copy()
    environment.pop("RUSTFLAGS", None)
    rust_flags = [
        f"--remap-path-prefix={source_root.resolve()}=/usr/src/rw-wps",
    ]
    if platform.system() == "Windows":
        # MSVC's linker otherwise writes the wall-clock link time into the PE
        # header and every debug-directory entry. /Brepro replaces those
        # timestamps and the CodeView identity with content-derived values.
        rust_flags.extend((
            "-C", "link-arg=/Brepro",
            "-C", "target-feature=+crt-static",
        ))
    environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(rust_flags)
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    environment["SOURCE_DATE_EPOCH"] = source_date_epoch
    return environment


def _stage_decoder_cargo_registry(destination: Path) -> None:
    """Copy the standalone decoder's locked registry closure for its build."""
    registry = REPO / "tools/rustwx/vendor/crates-io"
    lock = tomllib.loads((REPO / "tools/grib1_bridge/Cargo.lock").read_text())
    selected = {}
    for package in lock["package"]:
        if not package.get("source", "").startswith("registry+"):
            continue
        source = registry / f"{package['name']}-{package['version']}"
        if not source.is_dir():
            source = registry / package["name"]
        metadata = tomllib.loads((source / "Cargo.toml").read_text())["package"]
        checksum = json.loads((source / ".cargo-checksum.json").read_text())["package"]
        if ((metadata["name"], metadata["version"]) != (package["name"], package["version"])
                or checksum != package["checksum"]):
            raise ValueError(f"{source}: registry identity or checksum differs from the decoder lock")
        selected[source.relative_to(REPO).as_posix()] = source.name
    # Copy tracked members only: an ignored target or cache must not become
    # part of a standalone build merely because it exists in the checkout.
    for relative in sorted(_tracked_source_paths(REPO)):
        parent = Path(relative).parts[:5]
        name = selected.get("/".join(parent))
        if name is not None:
            source = REPO / relative
            target = destination / name / Path(relative).relative_to("/".join(parent))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def build_release(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() not in {
        "x86_64", "AMD64",
    }:
        raise RuntimeError("RW-WPS release assembly requires Linux x86-64")
    if sys.version_info < (3, 11):
        raise RuntimeError("RW-WPS release assembly requires Python 3.11+")

    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output directory and archive must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("refusing to build RW-WPS from a dirty source tree")

    source_date_epoch = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "-s", "--format=%ct", "HEAD"],
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="rw-wps-release-build-") as raw:
        temporary = Path(raw)
        python_project = temporary / "python-project"
        python_inventory = _stage_rw_wps_python_project(python_project)
        wheel_dir = temporary / "wheel"
        wheel_dir.mkdir()
        wheel_environment = os.environ.copy()
        wheel_environment["SOURCE_DATE_EPOCH"] = source_date_epoch
        wheel_environment["PYTHONHASHSEED"] = "0"
        _run([
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(python_project),
        ], env=wheel_environment)
        wheels = sorted(wheel_dir.glob("rw_wps-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"wheel build produced {len(wheels)} RW-WPS wheels: {wheels}")

        cargo_target = temporary / "cargo-target"
        environment = _cargo_release_environment(
            source_root=REPO,
            target_dir=cargo_target,
            source_date_epoch=source_date_epoch,
        )
        decoder_registry = temporary / "decoder-registry"
        _stage_decoder_cargo_registry(decoder_registry)
        # One build per workspace the bundle's declaration names, into
        # one target directory, which is the shape the release bridge
        # builder already uses.  Reading the workspaces off the table
        # instead of hard-coding the decoder one is what makes a bridge
        # that lives elsewhere -- rw_fetch does -- actually get built:
        # this command used to compile only tools/grib1_bridge and then
        # hand the packager a namespace with no rw_fetch in it at all.
        for workspace in BRIDGE_WORKSPACES:
            manifest = REPO / workspace / "Cargo.toml"
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"the standalone bundle declares bridges from "
                    f"{workspace} and this checkout has no {manifest}")
            command = [
                "cargo",
                "build",
                "--manifest-path",
                str(manifest),
                "--release",
                "--locked",
                "--offline",
            ]
            if workspace == "tools/grib1_bridge":
                command += ["--config", "source.vendored-sources.directory="
                            + json.dumps(str(decoder_registry))]
            _run(command, cwd=manifest.parent, env=environment)
        native = cargo_target / "release"
        distribution_args = argparse.Namespace(
            wheel=wheels[0],
            cpu_backend=native / CPU_BACKEND_LIBRARY,
            output_dir=output,
            archive=archive,
            **{bridge.dest: native / bridge.name
               for bridge in BUNDLED_BRIDGES},
        )
        result = build_distribution(distribution_args)

    result["build_interface"] = "tools/build_rw_wps_release.py"
    result["rust_build"] = "cargo-release-locked-offline"
    result["python_build"] = "pip-wheel-no-build-isolation-no-deps"
    result["python_package_inventory"] = python_inventory
    result["source_date_epoch"] = source_date_epoch
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        build_release(args), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
