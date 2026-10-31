"""Typed multi-domain experiment schema and TOML loader (Phase 5, lane L1).

Implements section A of the RATIFIED Phase-5 nesting architecture
(docs/superpowers/specs/2026-07-16-phase5-nesting-architecture.md): frozen
:class:`DomainConfig`/:class:`ExperimentConfig` dataclasses over the
existing :class:`woof.config.RunConfig`, the ``[experiment]`` /
``[shared]`` / ``[[domain]]`` / ``[projection]`` TOML shape, and every
load-time validation rule of the ratified design -- all fail-loud.

Clock fields (architecture section C): the ROOT domain carries WRF's
exact integer-rational model step -- ``time_step`` (integer seconds) plus
the optional ``time_step_fract_num``/``time_step_fract_den`` correction,
mirroring WRF v4.6.1 Registry.EM_COMMON:2245-2246::

    rconfig   integer time_step           namelist,domains  1   -1  ih ...
    rconfig   integer time_step_fract_num namelist,domains  1    0  ih ...
    rconfig   integer time_step_fract_den namelist,domains  1    1  ih ...

Child ``dt`` and ``dx`` are NEVER hand-typed: ``dt_child = dt_parent /
parent_time_step_ratio`` and ``dx_child = dx_parent / parent_grid_ratio``
as exact rationals -- WRF divides the parent's rational time interval by
``parent_time_step_ratio`` EXACTLY (share/set_timekeeping.F:366-368,
``stepTime = domain_get_time_step(parents(1)) / parent_time_step_ratio``).
Explicitly supplied child values are cross-checked and a mismatch is a
hard error: the namelist chain is authoritative (the bundle's d04 runs at
exactly 1000/3 m and 5/3 s -- never a hand-typed "500 m", never 1.6667).

The legacy single-domain path is untouched: ``load_config()`` and the
``[grid]``/``[dynamics]``/``[run]`` tables resolve byte-identically
(pinned by tests/test_config_freeze.py); frozen verify cases construct
``RunConfig`` directly and never migrate.
"""

from __future__ import annotations

import difflib
import math
import io
import tomllib
from dataclasses import dataclass, fields, replace as _dc_replace
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

from woof import physics_mode as physics_mode_module
from woof.core.devices import DeviceOptions, DEVICES_OFF, validate_device_road
from woof.core import streaming as streaming_module
from woof.io import history_selection as history_selection_module
from woof import simulated_radar_config as simulated_radar_module
from woof.config_keys import KeyRow, key_rows
from woof.config import (DEFAULT_COLUMN_CHUNK, RADAR_HEATING_OFF, RadarHeatingConfig,
                          EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT,
                          EPSSM_AUTO,
                          GRELL_FAMILY_CU_PHYSICS, GRELL_FAMILY_DEFAULTS,
                          SAS_FAMILY_DEFAULTS, SAS_PGCON_SCHEMES,
                          MIX_ISOTROPIC_AUTO, SASE_FAIL_CLOSED_DEFAULTS,
                          SASE_PBL_SCHEME, RunConfig,
                          anisotropic_w_mixing_ratio,
                          auto_mix_isotropic_selection, radiation_enabled,
                          validate_run_config, warn_anisotropic_w_mixing)
from woof.explain import layered, warn, warn_once
from woof.static.projection import footprint_contains_pole

#: Relative tolerance for cross-checking hand-typed child dx/dt against the
#: exact chain-derived rational.  Wide enough for a truncated decimal of
#: the true value (the bundle namelist's ``dx = 333.333333`` vs the exact
#: 1000/3 m, relative error ~1e-9); a wrong value (a hand-typed "500 m"
#: d04 against ratio 3 from 1 km) misses by ~0.5 and is a hard error.
_REL_TOL = 1.0e-6

# Shared by both experiment validators and the namelist support report.
FEEDBACK_OPTIONS = (0, 1)
SMOOTH_OPTION_OPTIONS = (0, 1, 2)

#: WRF moving-nest namelist controls (Registry.EM_COMMON &domains).  Any
#: appearance is still rejected loudly, and the reason is unchanged: every
#: key here drives CONTINUOUS, per-step nest motion, which invalidates the
#: SINT donor index/weight tables inside the integration.  This tree offers
#: DISCRETE relocation instead -- whole parent cells at cycle boundaries,
#: donor tables rebuilt once per placement generation -- which preserves
#: the premise ("tables precomputed once at setup") that this rejection
#: exists to protect.  The discrete surface is :class:`RelocationConfig`
#: under ``[relocation]``.  Unlike WRF's per-step ``move_interval``, its
#: ``cadence_seconds``/``[[relocation.move]]`` schedule names cycle-boundary
#: OPPORTUNITIES for the discrete mechanism (ArWen's storm-following order,
#: leg 2, 2026-08-06 -- superseding leg 1's no-schedule stance for the
#: runner while the per-step keys stay rejected).
_MOVING_NEST_KEYS = frozenset({
    "num_moves", "move_id", "move_interval", "move_cd_x", "move_cd_y",
    "vortex_interval", "max_vortex_speed", "corral_dist", "track_level",
    "time_to_move",
})

#: Keys accepted in ``[relocation]``.  ``enabled`` is mandatory and every
#: other key is refused while it is false, so a config cannot acquire a
#: movable nest by inheriting a block someone left behind.  ``follow`` is
#: the nested ``[relocation.follow]`` storm-tracking table
#: (:mod:`woof.core.storm_tracking`); ``move`` is the manual itinerary
#: (``[[relocation.move]]`` rows); ``cadence_seconds`` is how often the
#: runner consults whichever source is configured.  All three admissible
#: only while enabled.
_RELOCATION_KEYS = frozenset({
    "enabled", "grid_id", "mode", "max_move_parent_cells",
    "min_overlap_fraction", "cadence_seconds", "follow", "move",
    "containment", "track", "reach_speed_m_s",
})

#: Rows for the [relocation] keys no typed field declares
#: (:mod:`woof.config_keys`).
_RELOCATION_KEY_ROWS = key_rows(
    KeyRow("track", "table", None,
           "a single table, not an array of tables, naming the one file "
           "the tracked vortex is written to (path, interval_seconds, "
           "output_level); absent writes none"),
)

#: Keys accepted in ``[relocation.containment]`` -- the ancestor that
#: slides to keep the tracked mover contained (see
#: :class:`ContainmentConfig`).
_RELOCATION_CONTAINMENT_KEYS = frozenset({
    "grid_id", "deadband_cells", "max_move_parent_cells",
    "cadence_seconds",
})

#: Keys accepted in each ``[[relocation.move]]`` row (the manual follow
#: itinerary; the config-driven stand-in for the tracker seam).
_RELOCATION_MOVE_KEYS = frozenset({
    "at_seconds", "di_parent_cells", "dj_parent_cells",
})

#: The only relocation mode implemented.  Named (rather than implied) so a
#: future second mode has to be selected deliberately.
DISCRETE_RELOCATION_MODE = "discrete-cycle-boundary"

#: Nesting guard keys accepted in [shared] with their WRF Registry
#: defaults; any OTHER value is rejected loudly (only the default
#: machinery is implemented).  Declared as rows (:mod:`woof.config_keys`)
#: so a front end reads each one's type and default.
_GUARD_KEY_ROWS = key_rows(
    # Registry.EM_COMMON:2301: default 2 = SINT, the only implemented
    # horizontal nest interpolator (bilinear/NN/quadratic rejected).
    KeyRow("interp_method_type", "integer", 2,
           "nest horizontal interpolator; only 2 (SINT) is implemented"),
    # Registry.EM_COMMON:2300: 0 = standard eta-level interpolation;
    # 1 = isobaric re-interpolation, not implemented.
    KeyRow("nest_interp_coord", "integer", 0,
           "nest vertical coordinate; only 0 (eta levels) is implemented"),
    # Vertical nest refinement: rejected (identical vertical grid on all
    # domains; WRF only calls init_domain_vert_nesting when e_vert
    # differs, share/mediation_integrate.F:666).
    KeyRow("vert_refine_method", "integer", 0,
           "vertical nest refinement; only 0 (none) is implemented"),
    # High-resolution child terrain ingest: rejected (children SINT the
    # parent terrain and blend it, dyn_em/nest_init_utils.F).
    KeyRow("input_from_hires", "boolean", False,
           "high-resolution nest terrain input; only false is implemented"),
)
_GUARD_DEFAULTS = {name: row.default for name, row in _GUARD_KEY_ROWS.items()}

#: Rows for the [shared] keys that are not RunConfig fields
#: (:mod:`woof.config_keys`); ``eta_levels`` and ``p_top`` are fields.
_SHARED_KEY_ROWS = key_rows(
    KeyRow("e_vert", "integer", None,
           "WRF's full-level count, nz + 1; give nz, e_vert or eta_levels"),
    *_GUARD_KEY_ROWS.values(),
)

_EXPERIMENT_KEYS = frozenset({
    "name", "start_time", "run_seconds", "feedback", "smooth_option",
    "blend_width", "spec_bdy_width", "restart_interval_s", "smooth_cg_topo",
    "column_chunk", "acknowledgements", "constant_glw_wm2",
    # The physics-fidelity axis (woof/physics_mode.py).  Experiment-scope
    # because a tree whose domains ran different ledger entries could not be
    # compared across its own nest boundary, and because the whole point of
    # the axis is that one run is one arm.
    "physics_mode", "patchset", "patches",
})

#: Rows for the [experiment] keys no typed field declares
#: (:mod:`woof.config_keys`): the fidelity axis is resolved into a
#: :class:`woof.physics_mode.PhysicsModeResolution`, not stored by name.
_EXPERIMENT_KEY_ROWS = key_rows(
    KeyRow("physics_mode", "string", None,
           "the physics-fidelity axis, 'wrf-faithful' or 'arwen-patched'; "
           "absent leaves every physics key as the config writes it"),
    KeyRow("patchset", "string", physics_mode_module.DEFAULT_PATCHSET,
           "the frozen patch-set version the fidelity axis resolves; "
           "needs physics_mode"),
    KeyRow("patches", "array", None,
           "divergence-ledger entry ids to apply instead of the whole "
           "patch set; needs physics_mode", items="string"),
)

_EXPERIMENT_REQUIRED = ("name", "start_time", "run_seconds",
                        "restart_interval_s")

_PROJECTION_KEYS = ("map_proj", "ref_lat", "ref_lon", "truelat1",
                    "truelat2", "stand_lon")

#: [perturbation] carries exactly one key: the [[perturbation.bubbles]]
#: array of tables.  Scalars may join later; today a stray scalar here is
#: most likely a bubble key that escaped its double-bracket table.
_PERTURBATION_KEYS = ("bubbles",)
_BUBBLE_KEYS = frozenset({
    "center_lat", "center_lon", "center_height_m", "radius_km",
    "depth_m", "amplitude_k", "rh_preserve",
})
_BUBBLE_REQUIRED = ("center_lat", "center_lon", "center_height_m",
                    "radius_km", "depth_m", "amplitude_k")
#: WRF's own idealized warm bubble peak (em_quarter_ss,
#: module_initialize_ideal.F): the reference a large amplitude is named
#: against.
WRF_IDEALIZED_BUBBLE_AMPLITUDE_K = 3.0
#: Above this peak theta perturbation a bubble is past an initiation
#: nudge and rewrites the analysis near its center.  That is a choice,
#: not a breakage, so it runs: the amplitude is named in a warning and
#: recorded in the perturbation receipt.  The amplitude itself carries
#: no upper refusal.  What a large bubble can break depends on the
#: analysis under it, so it is refused where that analysis is seen, per
#: domain and before integration (woof.ingest.init_perturbation): a
#: layer heated past the top of the radiation's temperature table, and
#: rh_preserve building more water vapour than a forecast was measured
#: to survive.  Both application points keep the initial state
#: hydrostatic at the analysed pressure: initialize_real writes the
#: bubble before the specific volume and geopotential are formed, and
#: the prepared domain-tree runner re-integrates the geopotential after.
BUBBLE_AMPLITUDE_WARNING_K = 10.0

#: RunConfig keys that may NOT appear in [shared]: they are per-domain
#: (derived or [[domain]]-owned), experiment-owned, or retired in the
#: experiment path.
_SHARED_FORBIDDEN = {
    "nx": "[[domain]] (e_we/nx)", "ny": "[[domain]] (e_sn/ny)",
    "dx": "[[domain]] (root only; children derive dx_parent/ratio)",
    "dy": "[[domain]] (root only)",
    "dt": "the root [[domain]] time_step (children derive)",
    "clock_dt": "nowhere -- retired in the experiment path (always 0.0; "
                "it persists solely in a frozen legacy verification "
                "profile)",
    "run_seconds": "[experiment]", "restart_interval_s": "[experiment]",
    "spec_bdy_width": "[experiment]",
    "output_interval_s": "[[domain]] history_interval_s",
    "case": "nowhere -- the experiment path does not use the legacy "
            "case registry",
    "specified": "[[domain]]", "nested": "[[domain]]",
    "grid_id": "[[domain]]",
}

#: Per-domain RunConfig scalars a [[domain]] table may override
#: (architecture section A: cu_physics/cudt on d01 only per the bundle
#: namelist, radt 12/3/1/1, bldt, epssm's Registry-default tail, and
#: diff_6th_factor 0.12/0.10/0.08/0.06).  The turbulence row (km_opt
#: through tke_drag_coefficient) is the per-domain closure selection the
#: design doc reserved ("turbulence treatment stays configurable per
#: domain"): a PBL parent may carry a PBL-off Smagorinsky child.  Every
#: per-domain RunConfig still passes the full validate_run_config battery,
#: so cross-key refusals (PBL requires a surface layer, km_opt=3/4
#: excludes khdif/kvdif, isfflx=0/2 need their consumer) apply per domain
#: with the same messages as a single-domain config.  Per-domain LES
#: selection is a configuration capability, implemented-unverified for
#: nested LES children: no evidence covers a nested, specified-boundary,
#: moist, or terrain-following LES domain.
#: One entry here is a woof capability WRF cannot express, and it is
#: called out rather than left to be rediscovered: ``isfflx`` is
#: ``nentries=1`` in WRF (Registry.EM_COMMON:2644) -- a SCALAR namelist
#: rconfig, where ``km_opt`` (:2993) and ``bl_pbl_physics`` (:2617) are
#: ``max_domains`` columns.  ``isfflx = 1, 1, 0`` is therefore not a WRF
#: namelist at all: a Fortran namelist read of a list into a scalar is an
#: error, not a per-domain selection.  Admitting it per domain HERE is a
#: deliberate gpuwm-over-WRF extension of the TOML schema, and it is
#: reachable only by writing the TOML directly -- ``woof.namelist_import``
#: reads ``isfflx`` as the scalar WRF spells (commit a0ef9d29 reverted the
#: column reader for inventing a spelling WRF does not have), and
#: ``hrrr_hierarchy_direct``'s certified raw runtime contract pins
#: ``("physics", "isfflx")`` to the scalar ``[1]`` and refuses a column.
#: So no namelist-driven route can produce a per-domain isfflx, and none
#: should: a config that uses it has left WRF-expressible territory and
#: cannot be round-tripped back to a namelist.
from woof.sfire_config import FIRE_DOMAIN_FIELDS

_DOMAIN_RUN_OVERRIDES = (
    "hailcast_opt", "haildt", "hailcast_version",
    "lightning_option", "lightning_dt", "lightning_start_seconds",
    "flashrate_factor", "iccg_method", "iccg_prescribed_num",
    "iccg_prescribed_den", "cellcount_method", "cldtop_adjustment", "do_radar_ref",
    # clos_choice/ishallow ride with cu_physics: per-domain because the
    # scheme they configure is, and inert (validated zero) on any domain
    # that does not select cu_physics = 3.  On a tree that mixes
    # Grell-Freitas with another cumulus choice a [shared] value reaches
    # the Grell-Freitas domains only (see tree_runs_grell below).
    # cugd_avedx joins them for Grell 3D (cu_physics = 5) on the same terms.
    "cu_physics", "cudt_minutes", "clos_choice", "ishallow", "cugd_avedx",
    "maxiens", "maxens", "maxens2", "maxens3", "ensdim", "cu_diag", "convtrans_avglen_m",
    # The SAS-family keys, on the Grell keys' terms (tree_runs_sas below):
    # per-domain because the scheme they configure is.  sas_pgcon is a WRF
    # max_domains key; nsas_dx_factor is WRF scope 1, and the namelist door
    # writes the one value onto every New SAS domain, so a per-domain
    # difference is reachable only by writing the TOML directly.
    "sas_pgcon", "nsas_dx_factor",
    "radt", "radt_minutes", "bldt",
    # Every domain owns its radiation driver. Spectrum composition, CAM
    # parent transport and shared workspace sizing resolve these choices
    # per domain; requiring them to match has no remaining runtime basis.
    "ra_physics", "ra_lw_physics", "ra_sw_physics", "ra_rrtmg_variant",
    "wrf_rrtmg_compatibility", "o3input", "use_mp_re", "swrad_scat",
    "diff_6th_factor", "epssm", "spec_exp", "mp_physics", "moist",
    "moist_cq", "nest_microphysics_transition",
    # WRF SPP consumer switches are max_domains. Each original driver
    # owns its enabled parameter patterns and checkpoint state.
    # spp_lsm stays tree-wide: its consumer is the land surface, which
    # this loader keeps shared, and a per-domain row would change the
    # registry physics every earlier preparation was bound to.
    "spp_conv", "spp_pbl",
    "km_opt", "bl_pbl_physics", "sf_sfclay_physics", "c_s", "c_k",
    # WRF Registry.EM_COMMON:2889 declares moist_mix6_off max_domains, so it
    # is per domain here for the same reason diff_6th_factor is.
    "moist_mix6_off",
    # The NOAA WRFV3.9 fork declares diff_6th_factor2 max_domains
    # (Registry.EM_COMMON:2629), beside diff_6th_factor.
    "diff_6th_factor2",
    "diff_6th_opt", "mix_isotropic", "mix_upper_bound", "isfflx",
    "tke_heat_flux", "tke_drag_coefficient", "tke_upper_bound",
    # The rest of the numerics WRF declares `max_domains`, added because
    # the split was arbitrary rather than principled: `diff_6th_opt` and
    # `diff_6th_factor` were per domain while `diff_6th_slopeopt` and
    # `diff_6th_thresh` -- the same knob -- were not, and `epssm` was
    # while `emdiv`/`smdiv` were not.  A tree cannot tune damping or
    # diffusion on the nest that needs it without moving the parent too,
    # and on a 10/2/0.667 km tree those want different values: the
    # relaxation sponge alone is 40 km on the root and 2.7 km on the
    # inner nest at the same cell count.
    #
    # Additive and backward compatible: a domain that names none of these
    # still takes the `[shared]` value, so no existing experiment moves.
    #
    # DELIBERATELY ABSENT.  Geometry (`dx`, `dy`, `ztop`, `grid_id`,
    # `nested`, `specified`) is authored by the domain tree, not chosen.
    # The scheme SELECTORS WRF also scopes `max_domains` --
    # `sf_surface_physics` and the `bl_mynn_*` block remain outside this
    # table. Radiation selectors are resolved independently above.
    "diff_6th_slopeopt", "diff_6th_thresh",
    "dampcoef", "zdamp",
    "emdiv", "smdiv",
    "khdif", "kvdif", "diff_opt", "mix_full_fields",
    "h_sca_adv_order", "moist_adv_opt",
    # WRF declares the remaining advection orders max_domains too
    # (operational HRRR runs vert_order 5 on d01 and 3 on its nests).
    "v_sca_adv_order", "v_mom_adv_order", "h_mom_adv_order",
    "tke_budget",
    # Output-only, and per domain because its cost scales with the grid:
    # four extra (nz+1, ny, nx) planes per frame, so the finest domains
    # of a tree can be left off while the domains whose subgrid fluxes
    # are being read carry it.  The two SASE PHYSICS selectors stay
    # [shared], so every SASE domain of a tree runs one variant; on a
    # tree that mixes SASE with another PBL scheme the loader applies
    # them to the SASE domains only (see tree_runs_sase below).
    "sase_flux_diag",
    # Output-only on the same terms: two extra (nz, ny, nx) planes per
    # frame, so a tree can carry the horizontal viscosity on the domain
    # whose mixing is being read and leave it off the rest.
    "hmix_k_diag",
    "surface_energy_diag",
    # LES-nest inflow seeding (P3): per-domain because the mechanism IS
    # per-edge -- it perturbs one child's rolling nest-boundary tables,
    # validate_run_config refuses it on a non-nested domain, and like
    # per-domain isfflx these are TOML-only gpuwm-over-WRF keys with no
    # namelist spelling (stock 4.6.1's boundary perturbation is the
    # stoch-package perturb_bdy pattern route, not a cell-perturbation
    # column; PROVENANCE.md D10).
    "inflow_perturbation", "inflow_perturbation_seed",
    "inflow_perturbation_amplitude_scale", "inflow_perturbation_faces",
    # The adaptive-timestep keys WRF declares `max_domains`
    # (Registry.EM_COMMON:2269-2277).  Per domain because the constraint
    # is: a 10 km parent and a 2 km nest reach target_cfl at different
    # timesteps, and upstream's own guidance runs max_step_increase_pct
    # at 5 on a parent and 51 on a nest.
    #
    # `use_adaptive_time_step`, `step_to_output_time` and
    # `adaptation_domain` are DELIBERATELY ABSENT: WRF declares those
    # scope `1`, one scalar for the whole run, so leaving them out of
    # this tuple is what makes "[shared] only" enforced by the loader
    # rather than merely documented.
    "target_cfl", "target_hcfl", "max_step_increase_pct",
    "starting_time_step", "starting_time_step_den",
    "max_time_step", "max_time_step_den",
    "min_time_step", "min_time_step_den",
    # Per domain for the reason the steep-terrain rules set it: one
    # domain's ground needs six substeps and its neighbour's does not.
    "min_time_step_sound",
    # WRF declares slope_rad and topo_shading max_domains
    # (Registry.EM_COMMON), and a namelist commonly shades only the nests
    # fine enough to resolve the ridges (topo_shading = 0, 1, 1).  shadlen
    # is scope 1 in WRF and stays [shared].
    "slope_rad", "topo_shading",
    # Where Noah mosaic runs the urban canopy (woof/config.py
    # MOSAIC_URBAN_CANOPY_RULES): a woof key with no WRF spelling, per
    # domain so a tree can run the town rule on the nest fine enough to
    # see its towns while the parent keeps WRF's.  sf_surface_mosaic and
    # sf_urban_physics stay [shared]; every domain's RunConfig still passes
    # validate_noah_mosaic_config with them.
    "mosaic_urban_canopy",
    # WRF Registry scopes CLM lake selection per domain.
    "sf_lake_physics", "use_lakedepth", "lakedepth_default", "lake_min_elev",
    "lake_state_source", "lake_state_path", "lake_state_accept_licence",
    # WRF declares topo_wind and gwd_opt max_domains too
    # (woof.core.terrain_drag); the GSL suite tapers itself with each
    # domain's grid length, so a column is the natural shape.
    "topo_wind", "gwd_opt",
    "wind_farm_scheme", "wind_farm_table", "wind_farm_max_dx_m", "wind_farms", "windfarm_tke_factor",
    "history_interval2_s", "history_interval_change_s",
    "prec_acc_dt", "prec_acc_dt1", "gsd_diagnostics", "reset_interval1",
) + FIRE_DOMAIN_FIELDS

#: Per-domain vertical keys are REJECTED outright (F1 amendment: the
#: vertical grid is single-sourced from ExperimentConfig.vertical, so
#: vertical nesting is impossible by construction).
_DOMAIN_VERTICAL_KEYS = ("nz", "e_vert", "eta_levels", "p_top", "ztop",
                         "hybrid_opt", "etac")

#: Rows for the [[domain]] keys no typed field declares
#: (:mod:`woof.config_keys`).
_DOMAIN_KEY_ROWS = key_rows(
    KeyRow("quicklook", "table", None,
           "optional additional history cadence and variable selection"),
    KeyRow("static", "table", None, "per-domain terrain smooth_option, smooth_passes and smooth_precision"),
    KeyRow("start_time", "datetime", None,
           "when this nest starts, an offset-free TOML date-time inside "
           "the run; absent starts it with [experiment].start_time"),
    KeyRow("e_we", "integer", None,
           "WRF's staggered west-east point count, nx + 1; give e_we or nx"),
    KeyRow("e_sn", "integer", None,
           "WRF's staggered south-north point count, ny + 1; give e_sn or ny"),
)

_DOMAIN_KEYS = frozenset({
    "grid_id", "parent_id", "i_parent_start", "j_parent_start",
    "parent_grid_ratio", "parent_time_step_ratio", "history_interval_s",
    # WRF's history_begin_* / history_end_* (share/set_timekeeping_alarms
    # .m4), as seconds from this domain's start: the first history frame
    # is written at start + history_begin_s, then every
    # history_interval_s, and none after start + history_end_s.
    "history_begin_s", "history_end_s",
    "start_time", "static",
    "e_we", "e_sn", "nx", "ny", "dx", "dy", "dt",
    "time_step", "time_step_fract_num", "time_step_fract_den",
    "specified", "nested",
    # The dormant-nest declaration (woof/core/nest_spawn.py): a child
    # carrying `spawn = {...}` is declared, reserved in the memory plan,
    # and integrates nothing until its trigger fires mid-run.
    "spawn", "retire", "rearm", "follow",
    # The per-domain [tiles] override (woof/core/streaming.py): a domain
    # carrying `tiles = {...}` chooses its OWN road, and the tree-wide
    # [tiles] table is the default for every domain that does not.  This
    # is how "stream the parent, keep the child resident" -- and its
    # inverse -- are said explicitly rather than left to the planner.
    "tiles",
    # The per-domain [output] override (woof/io/history_selection.py): a
    # domain carrying `output = {...}` selects its OWN history variables,
    # and the tree-wide [output] table is the default for every domain
    # that does not.  This is how "keep the full inventory on the parent,
    # trim the 1 km child that writes ten times the bytes" is said
    # explicitly rather than applied to the whole tree.
    "output",
    # The quick-look stream (`quicklook = { every_s = 300, history_vars =
    # [...] }`): a domain whose full history is every history_interval_s
    # also writes a small variable subset every every_s in between, to
    # quicklook/wrfout_dNN_<time> (see DomainConfig.full_history_interval_s).
    "quicklook",
    *_DOMAIN_RUN_OVERRIDES,
})
_DOMAIN_REQUIRED = ("grid_id", "parent_id", "i_parent_start",
                    "j_parent_start", "parent_grid_ratio",
                    "parent_time_step_ratio", "history_interval_s")


@dataclass(frozen=True)
class VerticalConfig:
    """The single shared vertical coordinate (F1 amendment, §A).

    Frozen/hashable; serializes verbatim in the resolved-TOML round trip
    and sits INSIDE the experiment fingerprint (``eta_levels`` as the
    exact FP64 tuple).  Every domain shares this one object -- vertical
    nesting is rejected by construction (per-domain vertical keys are a
    load error; WRF only calls init_domain_vert_nesting when a nest
    refines the vertical grid, share/mediation_integrate.F:666).
    ``eta_levels = ()`` (with ``p_top = 0``) marks an idealized/legacy
    configuration whose vertical grid is built from nz/ztop instead.

    Invariants live HERE so programmatic construction cannot bypass them
    (shadow review S3): finiteness before ordering, 1.0 -> 0.0 strictly
    decreasing eta, finite non-negative p_top, hybrid_opt in (0, 1, 2),
    finite etac in [0, 1].
    """

    eta_levels: tuple[float, ...]
    p_top: float
    hybrid_opt: int
    etac: float

    @property
    def mass_level_count(self) -> int | None:
        """Return the derived mass-level count for an explicit eta grid.

        Idealized legacy configurations deliberately carry no explicit eta
        interfaces, so their count remains owned by ``RunConfig.nz`` and this
        property returns ``None``.  Real-data callers must require a concrete
        count rather than silently substituting a project-default profile.
        """

        return len(self.eta_levels) - 1 if self.eta_levels else None

    @property
    def interface_level_count(self) -> int | None:
        """Return WRF ``e_vert`` for an explicit eta grid, if present."""

        return len(self.eta_levels) if self.eta_levels else None

    def __post_init__(self):
        for i, value in enumerate(self.eta_levels):
            if not math.isfinite(value):
                raise ValueError(
                    f"eta_levels[{i}] = {value!r} is not finite; every "
                    "eta full level must be a finite float.")
        if self.eta_levels:
            if len(self.eta_levels) < 2:
                raise ValueError(
                    f"eta_levels needs at least 2 full levels, got "
                    f"{len(self.eta_levels)}.")
            if self.eta_levels[0] != 1.0 or self.eta_levels[-1] != 0.0:
                raise ValueError(
                    "eta_levels must run from 1.0 (surface) to 0.0 "
                    f"(top), got {self.eta_levels[0]!r} .. "
                    f"{self.eta_levels[-1]!r}.")
            if any(b >= a for a, b in zip(self.eta_levels,
                                          self.eta_levels[1:])):
                raise ValueError("eta_levels must be strictly decreasing.")
        if not math.isfinite(self.p_top) or self.p_top < 0.0:
            raise ValueError(
                f"p_top = {self.p_top!r} must be a finite non-negative "
                "pressure in Pa.")
        if self.hybrid_opt not in (0, 1, 2):
            raise ValueError(
                f"hybrid_opt must be 0/1 (B(eta) = eta) or 2 (WRF v4 "
                f"cubic-B hybrid), got {self.hybrid_opt!r}.")
        if not math.isfinite(self.etac) or not 0.0 <= self.etac <= 1.0:
            raise ValueError(
                f"etac = {self.etac!r} must be a finite eta value in "
                "[0, 1].")


#: WPS map_proj string -> WRF integer convention (1=lambert, 2=polar
#: stereographic, 3=mercator); the [shared] map_proj gate below.
_MAP_PROJ_WRF_CODES = {"lambert": 1, "polar": 2, "mercator": 3}


@dataclass(frozen=True)
class ProjectionConfig:
    """``[projection]`` table: the WPS &geogrid projection parameter set
    ``grids_from_wps_namelist`` already consumes (woof/static/
    projection.py); resolved per the F1 amendment and consumed by Task 5.
    Frozen/hashable, inside the experiment fingerprint (every parameter
    exact).  Finiteness/range invariants here so programmatic
    construction cannot bypass them (shadow review S3).

    ``map_proj`` is the WPS string: ``lambert``, ``mercator``, or
    ``polar``.  All six keys are always present; for Mercator,
    ``truelat2`` and ``stand_lon`` do not enter the projection math
    (module_llxy semantics) but stay in the fingerprint.  Genuine-limit
    refusals: a Lambert cone spanning both hemispheres is
    ill-conditioned (lc_cone is written for same-signed true
    latitudes), and true latitudes at the pole/equator degenerate the
    respective projections."""

    map_proj: str
    ref_lat: float
    ref_lon: float
    truelat1: float
    truelat2: float
    stand_lon: float

    def __post_init__(self):
        if self.map_proj not in _MAP_PROJ_WRF_CODES:
            normalized = self.map_proj.lower().replace("_", "-")
            latlon = normalized in {
                "lat-lon", "latlon", "regular-ll", "rotated-lat-lon",
                "rotated-ll",
            }
            blocker = (
                " Regular/rotated latitude-longitude needs angular dx/dy "
                "rather than metre spacing and WRF's global/pole polar "
                "filter; rotated grids also need pole_lat/pole_lon state "
                "and the map_proj == 6 curvature branch."
                if latlon else "")
            raise ValueError(
                f"map_proj = {self.map_proj!r} is not supported "
                "(implemented: 'lambert', 'mercator', 'polar')."
                f"{blocker}")
        for key, lo, hi in (("ref_lat", -90.0, 90.0),
                            ("truelat1", -90.0, 90.0),
                            ("truelat2", -90.0, 90.0),
                            ("ref_lon", -180.0, 180.0),
                            ("stand_lon", -180.0, 180.0)):
            value = getattr(self, key)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(
                    f"{key} = {value!r} in [projection] must be a "
                    f"finite value in [{lo}, {hi}] degrees.")
        if self.map_proj == "lambert":
            if self.truelat1 * self.truelat2 < 0.0:
                raise ValueError(
                    f"Lambert true latitudes {self.truelat1!r} and "
                    f"{self.truelat2!r} span both hemispheres; the "
                    "lc_cone secant formula is defined for a cone on "
                    "one side of the equator.")
            if max(abs(self.truelat1), abs(self.truelat2)) >= 90.0 \
                    or min(abs(self.truelat1), abs(self.truelat2)) <= 0.0:
                raise ValueError(
                    "Lambert true latitudes must lie strictly between "
                    "the equator and the pole, got "
                    f"{self.truelat1!r}/{self.truelat2!r}; use map_proj "
                    "= 'mercator' toward the equator or 'polar' toward "
                    "the pole.")
            if max(abs(self.truelat1), abs(self.truelat2)) >= 89.9 \
                    or min(abs(self.truelat1), abs(self.truelat2)) <= 0.1:
                warn("Lambert true latitudes "
                     f"{self.truelat1!r}/{self.truelat2!r} sit inside "
                     "0.1 degrees of the projection's degenerate limits "
                     "(equator/pole); the cone is ill-conditioned there "
                     "-- mercator or polar is the better tool")
        elif self.map_proj == "mercator":
            if abs(self.truelat1) >= 90.0:
                raise ValueError(
                    f"Mercator truelat1 = {self.truelat1!r} is at the "
                    "pole; the projection degenerates (cos(truelat1) "
                    "-> 0).")
            if abs(self.truelat1) >= 89.9:
                warn(f"Mercator truelat1 = {self.truelat1!r} is within "
                     "0.1 degrees of the pole; cos(truelat1) is nearly "
                     "0 and the scale factor is extreme")
        elif self.map_proj == "polar":
            if abs(self.truelat1) <= 0.1:
                raise ValueError(
                    f"polar stereographic truelat1 = {self.truelat1!r} "
                    "does not select a hemisphere; the pole (from the "
                    "sign of truelat1) is the projection centre.")

    @property
    def wrf_map_proj(self) -> int:
        """WRF integer convention (1=lambert, 2=polar, 3=mercator)."""
        return _MAP_PROJ_WRF_CODES[self.map_proj]


@dataclass(frozen=True)
class BubbleConfig:
    """One validated [[perturbation.bubbles]] entry.

    A cosine-squared warm bubble added ONCE to the initial potential
    temperature after the base real-data state is final: peak
    ``amplitude_k`` Kelvin at (``center_lat``, ``center_lon``,
    ``center_height_m`` AGL), falling to exactly zero at the ellipse
    ``sqrt((r_h/radius_km)^2 + ((z-zc)/depth_m)^2) = 1`` -- the WRF
    em_quarter_ss shape (module_initialize_ideal.F; the port's own
    transcription is woof/verify/cases/moist_bubble.py).  ``depth_m``
    is the vertical HALF-depth.  ``rh_preserve`` keeps relative humidity
    constant through the theta change by adjusting qv inside the bubble;
    the default leaves qv byte-untouched.

    Geometry validation (center inside the coarse domain, at least one
    cell touched) needs the projected grid and therefore happens at
    prepare time (:mod:`woof.ingest.init_perturbation`), before any
    integration step.
    """

    center_lat: float
    center_lon: float
    center_height_m: float
    radius_km: float
    depth_m: float
    amplitude_k: float
    rh_preserve: bool = False

    def __post_init__(self):
        for name in ("center_lat", "center_lon", "center_height_m",
                     "radius_km", "depth_m", "amplitude_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                    value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"{name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if not -90.0 <= self.center_lat <= 90.0:
            raise ValueError(
                f"center_lat = {self.center_lat!r} is outside [-90, 90] "
                "degrees")
        if not -180.0 <= self.center_lon <= 180.0:
            raise ValueError(
                f"center_lon = {self.center_lon!r} is outside [-180, 180] "
                "degrees")
        if self.center_height_m < 0.0:
            raise ValueError(
                f"center_height_m = {self.center_height_m!r} is below the "
                "surface; the bubble center is metres AGL and must be "
                ">= 0")
        for name in ("radius_km", "depth_m", "amplitude_k"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"{name} = {getattr(self, name)!r} must be positive; "
                    "a nonpositive bubble is a no-op wearing the name of "
                    "a perturbation")
        if not isinstance(self.rh_preserve, bool):
            raise ValueError(
                f"rh_preserve must be a boolean, got {self.rh_preserve!r}")
        note = self.amplitude_warning()
        if note is not None:
            warn(note, once=True)

    def amplitude_warning(self) -> str | None:
        """The warning a peak above the nudge level carries, or ``None``."""
        if self.amplitude_k <= BUBBLE_AMPLITUDE_WARNING_K:
            return None
        return (
            f"perturbation bubble amplitude_k = {self.amplitude_k:g} K is "
            f"above {BUBBLE_AMPLITUDE_WARNING_K:g} K and "
            f"{self.amplitude_k / WRF_IDEALIZED_BUBBLE_AMPLITUDE_K:.1f} "
            "times WRF's idealized warm bubble "
            f"({WRF_IDEALIZED_BUBBLE_AMPLITUDE_K:g} K, em_quarter_ss): it "
            "replaces the analysis near its center rather than nudging "
            "it. It runs as configured, and this warning is recorded in "
            "the perturbation receipt.")

    def receipt(self) -> dict:
        """Every accepted value, echoed in the shape it resolved to.

        A bubble above the nudge level also carries its ``warning``; one
        at or below it carries no such key, so its receipt is unchanged.
        """
        receipt = {
            "center_lat": float(self.center_lat),
            "center_lon": float(self.center_lon),
            "center_height_m": float(self.center_height_m),
            "radius_km": float(self.radius_km),
            "depth_m": float(self.depth_m),
            "amplitude_k": float(self.amplitude_k),
            "rh_preserve": bool(self.rh_preserve),
        }
        note = self.amplitude_warning()
        if note is not None:
            receipt["warning"] = note
        return receipt


@dataclass(frozen=True)
class PerturbationConfig:
    """The validated [perturbation] block: one or more theta bubbles.

    Absent block = ``ExperimentConfig.perturbation is None`` = zero
    behavior change, byte-identical prepared state (the
    inflow_perturbation OFF discipline).  Present, the block is either
    honored on the experiment runtime path, deferred by a preparation to
    the prepared forecast runner that applies it, single-domain or tree
    (:func:`deferred_initial_perturbation`), or refused by name on
    routes that do not thread it (:func:`refuse_unrouted_perturbation`);
    it is never dropped.
    """

    bubbles: tuple[BubbleConfig, ...]

    def __post_init__(self):
        if not self.bubbles:
            raise ValueError(
                "[perturbation] must carry at least one "
                "[[perturbation.bubbles]] entry; an empty block is "
                "either a stray table or a disabled feature wearing an "
                "enabled name -- delete the block to disable.")

    def receipt(self) -> dict:
        return {
            "schema": "gpuwm-initial-perturbation-v1",
            "bubbles": [bubble.receipt() for bubble in self.bubbles],
        }


def _build_perturbation(table, source: str) -> PerturbationConfig:
    """Validate [perturbation] / [[perturbation.bubbles]] of ``source``."""
    if not isinstance(table, dict):
        raise ValueError(
            f"[perturbation] of {source} must be a table carrying "
            f"[[perturbation.bubbles]] entries, got {table!r}.")
    _reject_unknown_keys("perturbation", table, _PERTURBATION_KEYS, source)
    _require_keys("perturbation", table, _PERTURBATION_KEYS, source)
    entries = table["bubbles"]
    if not isinstance(entries, list) or not entries or any(
            not isinstance(entry, dict) for entry in entries):
        raise ValueError(
            f"bubbles in [perturbation] of {source} must be an array of "
            "tables (one [[perturbation.bubbles]] block per bubble), got "
            f"{entries!r}.")
    bubbles = []
    for index, entry in enumerate(entries):
        label = f"perturbation.bubbles #{index + 1}"
        _reject_unknown_keys(label, entry, _BUBBLE_KEYS, source)
        _require_keys(label, entry, _BUBBLE_REQUIRED, source)
        try:
            bubbles.append(BubbleConfig(**entry))
        except ValueError as err:
            raise ValueError(
                f"[[{label}]] of {source}: {err}") from None
    return PerturbationConfig(bubbles=tuple(bubbles))


def refuse_unrouted_spectral_numerics(exp, route: str) -> None:
    """Fail loud where an ACTIVE [spectral_numerics] block would be dropped.

    The Level-2 hook is wired at exactly one seam -- the
    ``execute_experiment`` slow-large-step commit point.  A loop that
    integrates outside that seam and silently ignores an active block
    would run the forecast with the operator absent while the config, the
    restart identity and the capsule all say it was there: receipts the
    user expects would never be written, and an ``apply`` campaign would
    be a void arm wearing a treatment label.  ``mode = "off"`` passes --
    off is defined as the absence of the operator, which every loop
    provides for free.
    """
    config = getattr(exp, "spectral_numerics", None)
    if config is None or getattr(config, "mode", "off") == "off":
        return
    raise RuntimeError(
        f"the {route} route does not carry [spectral_numerics] "
        f"(mode = {config.mode!r}) to the wired slow-large-step seam; "
        "refused rather than ignored, because a forecast integrated with "
        "the operator silently absent would still be stamped with its "
        "config identity.  Run this configuration through the "
        "execute_experiment routes (woof run / the prepared tree "
        "runners), or set mode = \"off\".")


def refuse_unrouted_simulated_radar(exp, route: str) -> None:
    """Fail loud where an enabled [simulated_radar] would be dropped.

    The radar listens on the forecast's history landings through
    :class:`woof.simulated_radar.LiveSimulatedRadar`, which the
    ``woof run``, ``woof go`` and prepared-runner routes attach.  A route
    that integrates without it (the ensemble member and local cycling
    member legs) would accept the table, finish, and write no radar volume:
    a requested product silently absent from a run that reports success.
    ``enabled = false`` passes; it asks for nothing.
    """
    options = getattr(exp, "simulated_radar", None)
    if options is None or not getattr(options, "enabled", False):
        return
    raise ValueError(
        f"the {route} route does not attach [simulated_radar] to its history "
        "landings; refused rather than ignored, because the run would finish "
        "with no radar volume while its configuration asked for one.  Run "
        "this configuration through woof go, woof run or woof sim, or set "
        "[simulated_radar] enabled = false for the members.")


def refuse_unrouted_perturbation(exp, route: str) -> None:
    """Fail loud where a [perturbation] block would otherwise be dropped.

    The governance rule for this block is the same as for every key:
    honored or refused, never ignored.  Routes that do not thread the
    perturbation into their initialization call this immediately after
    loading the experiment.
    """
    if getattr(exp, "perturbation", None) is None:
        return
    raise ValueError(layered(
        f"the {route} route does not apply [perturbation] blocks; "
        "refused rather than ignored, because a dropped perturbation "
        "would run an unperturbed state under the name of your bubbles.",
        "Initial-state theta bubbles are applied by the experiment "
        "runtime path (woof run / woof ingest, through "
        "woof.ingest.real.initialize_real) and by the prepared "
        "single-domain and domain-tree forecast runners (applied to the "
        "restored states, woof.prepared_single_domain_forecast and "
        "woof.prepared_domain_tree_forecast)."))


#: The receipt a preparation writes when it prepares a domain tree whose
#: experiment carries a [perturbation] block.  One spelling for every
#: source: the deferral is the same fact whichever source prepared the
#: tree, because the runner that applies the bubbles restores every
#: source's tree the same way.
DEFERRED_PERTURBATION_SCHEMA = "gpuwm-initial-perturbation-deferred-v1"


def deferred_initial_perturbation(exp, route: str, *,
                                  announce: bool = True) -> dict | None:
    """What a preparation does with the experiment's [perturbation] block.

    No preparation writes a bubble into the arrays it prepares.  The
    prepared forecast runners apply the configured bubbles to the
    restored states at experiment start
    (:mod:`woof.prepared_domain_tree_forecast` for a tree,
    :mod:`woof.prepared_single_domain_forecast` for one domain),
    whatever source prepared the bundle, so a preparation keeps its
    initial and boundary arrays unperturbed and returns this receipt for
    its proof.  Every source route calls this one function; none has its
    own path.  An absent block returns ``None`` and says nothing.

    ``announce`` says the deferral once on stderr.  A stage that only
    needs the early refusal passes False, so a preparation that reads
    its configuration in two processes says it once.
    """

    perturbation = getattr(exp, "perturbation", None)
    if perturbation is None:
        return None
    runner = ("woof.prepared_domain_tree_forecast" if len(exp.domains) > 1
              else "woof.prepared_single_domain_forecast")
    if announce:
        warn(
            "initial perturbation is deferred to prepared forecast "
            "initialization",
            "Prepared initial and boundary arrays remain unperturbed, and "
            "the preparation receipt records the configured bubbles as "
            "deferred. The prepared forecast runner applies them once to "
            "the restored states at experiment start, never on restart or "
            "to a nest born later. Companion stock-WRF export cannot "
            "represent this deferred mutation.")
    return {
        "schema": DEFERRED_PERTURBATION_SCHEMA,
        "status": "DEFERRED_TO_FORECAST_INITIALIZATION",
        "prepared_arrays": "unperturbed source initial and boundary states",
        "application_route": runner,
        "application_point": "restored states at experiment start time",
        "applied_on_restart": False,
        "applied_to_delayed_domains": False,
        "config": perturbation.receipt(),
    }


def deferred_perturbation_config(receipt) -> PerturbationConfig:
    """The [perturbation] block a deferral receipt recorded, revalidated.

    For a stage that builds its experiment without the configuration
    (the native HRRR hierarchy builds its children from namelists, which
    spell no such block) and must still know the tree carries bubbles,
    so that the companion stock-WRF export refuses a state it cannot
    hold instead of writing the unperturbed one under the tree's name.
    Rebuilt through :func:`_build_perturbation`, the one validator; a
    receipt's own annotations (``warning``) are not configuration.
    """

    if (not isinstance(receipt, dict)
            or receipt.get("schema") != DEFERRED_PERTURBATION_SCHEMA):
        raise ValueError(
            "a [perturbation] deferral must be a "
            f"{DEFERRED_PERTURBATION_SCHEMA} receipt, got {receipt!r}")
    bubbles = [{key: value for key, value in bubble.items()
                if key in _BUBBLE_KEYS}
               for bubble in receipt["config"]["bubbles"]]
    return _build_perturbation(
        {"bubbles": bubbles}, f"a {DEFERRED_PERTURBATION_SCHEMA} receipt")


@dataclass(frozen=True)
class DomainConfig:
    """One domain of a nested experiment (architecture section A).

    ``run`` is the fully resolved per-domain :class:`RunConfig` carrying
    nx/ny/nz/dx/dy, the per-domain physics and diffusion selections, the
    ``specified``/``nested`` flags, and ``clock_dt = 0.0`` (retired in the
    experiment path); ``run.dt`` is the binary64 image of the CHAINED
    single-precision WRF REAL dt (``np.float32`` division down the ratio
    chain, share/set_timekeeping.F:368 -- for d04 1.6666666269302368,
    NOT ``float(dt_exact)``); the exact rational lives on
    :meth:`ExperimentConfig.dt_exact`.  (Stale wording fixed per the
    p5t9 adversarial review, finding 4.)

    The root domain (``parent_id == 0``) carries the WRF rational clock
    keys ``time_step`` (integer seconds) + ``time_step_fract_num`` /
    ``time_step_fract_den`` (Registry.EM_COMMON:2245-2246); children carry
    ``time_step = None`` and derive their dt from the ratio chain.
    """

    grid_id: int
    parent_id: int
    i_parent_start: int
    j_parent_start: int
    parent_grid_ratio: int
    parent_time_step_ratio: int
    history_interval_s: float
    run: RunConfig
    time_step: int | None = None
    #: WRF's history_begin: seconds after this domain's start before its
    #: first history frame (0 writes the start frame, WRF's default).
    #: Output-only, so a restart may change it (RESTART_TOLERATED_DOMAIN
    #: _FIELDS in woof/core/model.py).
    history_begin_s: float = 0.0
    #: WRF's history_end: no history frame after start + this many
    #: seconds; ``None`` writes to the end of the run.
    history_end_s: float | None = None
    time_step_fract_num: int = 0
    time_step_fract_den: int = 1
    start_time: datetime | None = None
    #: The validated ``spawn`` table (:class:`woof.core.nest_spawn
    #: .SpawnConfig`), or ``None`` for an ordinary live domain.  ``None``
    #: is the OFF contract and is omitted from the restart-identity
    #: payload and tolerated by the prepared-cache identity, so every
    #: pre-feature fingerprint stays byte-identical.  Non-``None`` marks
    #: this domain DORMANT: declared and memory-reserved from startup,
    #: integrating nothing until the trigger fires; ``i_parent_start`` /
    #: ``j_parent_start`` are then the PLACEHOLDER placement (the memory
    #: plan's and the manual time-trigger's), and the fired placement is
    #: chosen at trigger time (:func:`active_experiment`).
    spawn: "object | None" = None
    #: Optional episode retirement policy for a spawned child.
    retire: "object | None" = None
    #: Optional slot re-arm policy; absent means the historical one-shot spawn.
    rearm: "object | None" = None
    #: Optional per-domain storm-follow policy.  Tree-level [relocation] remains
    #: the backwards-compatible single-follower surface.
    follow: "object | None" = None
    #: This domain's own ``[tiles]`` road (:class:`woof.core.streaming
    #: .StreamingOptions`), or ``None`` to take the tree-wide table.
    #: ``None`` is the inherit contract and drops out of the restart
    #: identity, so every experiment written before the per-domain
    #: surface existed keeps its exact fingerprint.
    #:
    #: Unlike ``spawn``, a DECLARED value binds nothing either: ``[tiles]``
    #: is an execution choice whose entire claim is that it changes no
    #: bytes, so a domain that streamed must resume resident and a domain
    #: that ran resident must resume streamed.  See
    #: ``streaming.identity_payload_entry``.
    tiles: "object | None" = None
    #: This domain's own ``[output]`` history-variable selection
    #: (:class:`woof.io.history_selection.HistorySelection`), or ``None``
    #: to take the tree-wide table.  ``None`` is the inherit contract and
    #: drops out of the restart identity, so every experiment written
    #: before the surface existed keeps its exact fingerprint.
    #:
    #: Like ``tiles`` and unlike ``spawn``, a DECLARED value binds nothing
    #: either: the selection decides which variables reach the HISTORY
    #: tape and changes no number the model computes, so a run that
    #: trimmed its history must resume from a checkpoint written by one
    #: that did not.  Checkpoints are separate files written from model
    #: state (:mod:`woof.io.restart`), never from the history frame.
    output: "object | None" = None
    #: The QUICK-LOOK stream (``quicklook = { every_s, history_vars }`` on
    #: a [[domain]]).  When set, ``history_interval_s`` above is the
    #: quick-look cadence -- the alarm every history consumer (the clock,
    #: the adaptive driver, the divisibility checks) already reads -- and
    #: this is the cadence of the FULL frames, a whole multiple of it.  A
    #: frame on the full lattice is an ordinary wrfout; every other frame
    #: is written with :attr:`quicklook_output` only, to its own folder,
    #: ``quicklook/wrfout_dNN_<time>`` (the renderer needs the wrfout
    #: name; the folder marks the subset).  ``None``
    #: (every domain without the key) writes every frame full, exactly as
    #: before.  Output-only like ``output``: it leaves the restart identity.
    full_history_interval_s: float | None = None
    #: The quick-look frames' variable selection
    #: (:class:`woof.io.history_selection.HistorySelection`).
    quicklook_output: "object | None" = None


#: The quick-look subset when a ``quicklook`` table names no variables: the
#: maps a minute-by-minute look needs (composite reflectivity, updraft
#: helicity, 10 m wind, 2 m temperature and moisture, surface pressure and
#: accumulated precipitation).  Measured 13 B per cell against 172 for a
#: full frame (WOOF run-controls feasibility, 2026-10-23).
QUICKLOOK_DEFAULT_VARS = ("REFL_10CM", "UP_HELI_MAX", "U10", "V10", "T2",
                          "Q2", "PSFC", "RAINNC", "RAINC")


def _quicklook(dom: dict, *, grid_id: int, source: str,
               history_interval_s: float):
    """(alarm cadence, full cadence, selection) for one domain's table.

    No ``quicklook`` key: ``(history_interval_s, None, None)``, the
    domain as it always was.  Refused, by name: a quick-look cadence that
    is not below the full one, or that does not divide it -- a full frame
    must land on a quick-look alarm, or the full tape would drift off its
    own interval.
    """
    table = dom.get("quicklook")
    if table is None:
        return history_interval_s, None, None
    where = f"quicklook on [[domain]] grid_id = {grid_id} of {source}"
    if not isinstance(table, dict):
        raise ValueError(f"{where} must be an inline TABLE, e.g. quicklook = "
                         "{ every_s = 300.0 }, got " + repr(table))
    unknown = sorted(set(table) - {"every_s", "history_vars"})
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; known keys: "
                         "['every_s', 'history_vars']")
    every = table.get("every_s")
    if (isinstance(every, bool) or not isinstance(every, (int, float))
            or not math.isfinite(float(every)) or float(every) <= 0):
        raise ValueError(f"{where}: every_s must be a positive number of "
                         f"seconds, got {every!r}")
    every = float(every)
    full = Fraction(str(history_interval_s))
    quick = Fraction(str(every))
    if quick >= full or (full / quick).denominator != 1:
        raise ValueError(
            f"{where}: every_s = {every:g} s must be shorter than the "
            f"domain's history_interval_s = {history_interval_s:g} s and "
            "divide it exactly, so every full frame lands on a quick-look "
            "time")
    names = table.get("history_vars", list(QUICKLOOK_DEFAULT_VARS))
    checked = history_selection_module.HistorySelection.from_mapping(
        {"history_vars": list(names)}, source=where)
    # The quick-look form of the same selection: T and PHB only when named
    # (history_selection.QUICKLOOK_UNFORCED_FIELDS says why).
    selection = history_selection_module.QuickLookSelection(
        history_vars=checked.history_vars, source=checked.source)
    return every, float(history_interval_s), selection


@dataclass(frozen=True)
class ScheduledRelocationMove:
    """One row of the manual follow itinerary: when, and how far.

    The shift is in whole PARENT cells, the same unit and sign convention
    the tracker's plan provider returns, so the manual mode exercises the
    exact interface the tracker must satisfy.
    """

    at_seconds: float
    di_parent_cells: int = 0
    dj_parent_cells: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.at_seconds)) or float(
                self.at_seconds) <= 0.0:
            raise ValueError(
                f"at_seconds = {self.at_seconds!r} must be a finite, "
                "positive model time; t = 0 is initial placement, not a "
                "move")
        for name in ("di_parent_cells", "dj_parent_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(
                    f"{name} = {value!r} must be a whole number of parent "
                    "cells; discrete relocation has no fractional moves")

    def to_json(self) -> dict[str, object]:
        return {"at_seconds": float(self.at_seconds),
                "di_parent_cells": int(self.di_parent_cells),
                "dj_parent_cells": int(self.dj_parent_cells)}


@dataclass(frozen=True)
class ContainmentConfig:
    """An ancestor that slides to keep the tracked mover contained.

    The two-mover shape one storm actually needs: the TRACKED mover
    (``[relocation] grid_id``) follows the vortex in its own parent's
    cells, and one strict ancestor of it -- this block -- slides in
    whole cells of ITS parent whenever the mover has drifted more than
    ``deadband_cells`` (mover's-parent cells) from the centered
    placement.  While the ancestor slides, the tracked mover is
    EARTH-FIXED: its placement is compensated by ``-shift x ratio`` so
    the storm's frame does not move at all
    (:func:`woof.core.nest_relocation.relocate_child`,
    ``earth_fixed_descendants``).  This is deliberately not a second
    storm tracker -- two trackers can disagree and fight; a containment
    policy reads only placements, so it cannot.

    ``cadence_seconds`` ``None`` means the mover's own cadence.
    """

    grid_id: int
    deadband_cells: int = 8
    max_move_parent_cells: int | None = 1
    cadence_seconds: float | None = None

    def __post_init__(self) -> None:
        if int(self.grid_id) < 2:
            raise ValueError(
                f"containment grid_id = {self.grid_id!r} names the root "
                "or an invalid id; only a nest can slide")
        if int(self.deadband_cells) < 1:
            raise ValueError(
                f"containment deadband_cells = {self.deadband_cells!r} "
                "must be at least 1 mover's-parent cell; a zero dead-band "
                "slides on every consultation")
        if (self.max_move_parent_cells is not None
                and int(self.max_move_parent_cells) < 1):
            raise ValueError(
                "containment max_move_parent_cells must be at least 1")
        if self.cadence_seconds is not None and (
                not math.isfinite(float(self.cadence_seconds))
                or float(self.cadence_seconds) <= 0.0):
            raise ValueError(
                f"containment cadence_seconds = {self.cadence_seconds!r} "
                "must be a finite, positive number of model seconds")

    def to_json(self) -> dict[str, object]:
        return {
            "grid_id": int(self.grid_id),
            "deadband_cells": int(self.deadband_cells),
            "max_move_parent_cells": (
                None if self.max_move_parent_cells is None
                else int(self.max_move_parent_cells)),
            "cadence_seconds": (
                None if self.cadence_seconds is None
                else float(self.cadence_seconds)),
        }


@dataclass(frozen=True)
class RelocationConfig:
    """Bounds and (optionally) a follow source for discrete relocation.

    Three layers, deliberately separable:

    * The MECHANISM bounds -- which child may move (``grid_id``), how far
      in one event (``max_move_parent_cells``), and how much ground it
      must keep (``min_overlap_fraction``).  A caller driving
      :func:`woof.core.nest_relocation.relocate_child` directly uses
      only these.
    * The FOLLOW SOURCE -- at most one of: ``follow``, the validated
      ``[relocation.follow]`` storm-tracking block
      (:class:`woof.core.storm_tracking.FollowConfig`, the plan-provider
      seam), or ``moves``, the manual ``[[relocation.move]]`` itinerary
      (testable without a tracker, through the SAME provider contract).
      Neither present means the runner schedules nothing and relocation
      stays a manual/API mechanism, exactly as leg 1 shipped it.
    * The CADENCE -- ``cadence_seconds`` names the cycle-boundary
      opportunities at which the follow source is consulted.  ``None``
      with a source configured means EVERY complete cycle boundary (the
      tracker's own cooldown/dead-band are then the rate limit).

    ``None`` on either bound means unbounded, which is admissible but is
    a deliberate choice a config has to make.
    """

    enabled: bool = False
    grid_id: int | None = None
    mode: str = DISCRETE_RELOCATION_MODE
    max_move_parent_cells: int | None = None
    min_overlap_fraction: float | None = None
    cadence_seconds: float | None = None
    #: The validated ``[relocation.follow]`` storm-tracking block
    #: (:class:`woof.core.storm_tracking.FollowConfig`), or ``None`` when
    #: the config carries none.
    follow: "object | None" = None
    #: The manual itinerary (``[[relocation.move]]`` rows), or empty.
    moves: tuple[ScheduledRelocationMove, ...] = ()
    #: The ``[relocation.containment]`` ancestor-slide block
    #: (:class:`ContainmentConfig`), or ``None``.
    containment: "ContainmentConfig | None" = None
    #: The ``[relocation.track]`` output file
    #: (:class:`woof.core.storm_track_writer.TrackConfig`), or ``None``.
    #: A DIAGNOSTIC: it reads the vortex fix the tracker already
    #: produced and writes it where other tooling can read it.  Absent
    #: means no track file and a byte-identical forecast.  ONE file,
    #: because a run has one vortex.
    track: "object | None" = None
    #: How fast, on average since the start, the tracked nest may travel
    #: (m/s); ``None`` takes :data:`woof.core.nest_reach
    #: .DEFAULT_REACH_SPEED_M_S`.  The runner clamps a move that would
    #: pass it, and the statics corridor is sized to it
    #: (:mod:`woof.core.nest_reach`).
    reach_speed_m_s: float | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            if self.follow is not None:
                raise ValueError(
                    "[relocation.follow] on a disabled [relocation] block "
                    "is refused: a tracker with no mechanism to drive is a "
                    "config that half-opted-in; set enabled = true "
                    "deliberately, or delete the follow table")
            if self.moves:
                raise ValueError(
                    "[[relocation.move]] rows on a disabled [relocation] "
                    "block are refused: an itinerary with no mechanism to "
                    "drive is a config that half-opted-in; set enabled = "
                    "true deliberately, or delete the rows")
            if self.containment is not None:
                raise ValueError(
                    "[relocation.containment] on a disabled [relocation] "
                    "block is refused: an ancestor slide with no mover to "
                    "contain is a config that half-opted-in")
            if self.track is not None:
                raise ValueError(
                    "[relocation.track] on a disabled [relocation] block "
                    "is refused: the track is the TRACKER's answer written "
                    "out, and a disabled block runs no tracker")
            if self.reach_speed_m_s is not None:
                raise ValueError(
                    "reach_speed_m_s on a disabled [relocation] block is "
                    "refused: it bounds a tracker, and a disabled block "
                    "runs none")
            return
        if self.mode != DISCRETE_RELOCATION_MODE:
            raise ValueError(
                f"relocation mode {self.mode!r} is not implemented; the "
                f"only mode is {DISCRETE_RELOCATION_MODE!r} (whole parent "
                "cells at cycle boundaries)")
        if self.grid_id is None:
            raise ValueError(
                "[relocation] with enabled = true must name the grid_id of "
                "the child that may move; a tree-wide 'something moves' is "
                "not a placement")
        if int(self.grid_id) < 2:
            raise ValueError(
                f"relocation grid_id = {self.grid_id!r} names the root "
                "domain or an invalid id; only a child can be relocated, "
                "because a placement is a position inside a parent")
        if (self.max_move_parent_cells is not None
                and int(self.max_move_parent_cells) < 1):
            raise ValueError(
                "max_move_parent_cells must be at least 1 parent cell; a "
                "relocation moves whole parent cells, and 0 is the null "
                "move, which needs no bound")
        if self.min_overlap_fraction is not None and not (
                0.0 <= float(self.min_overlap_fraction) <= 1.0):
            raise ValueError(
                "min_overlap_fraction is a fraction of the child's cells "
                f"and must lie in [0, 1], got {self.min_overlap_fraction!r}")
        if self.follow is not None and self.moves:
            raise ValueError(
                "[relocation.follow] and [[relocation.move]] are two "
                "follow sources; with both present one would be silently "
                "ignored, so both are refused -- keep the tracker or the "
                "manual itinerary, not both")
        if self.containment is not None:
            if self.follow is None and not self.moves:
                raise ValueError(
                    "[relocation.containment] slides an ancestor to keep "
                    "the tracked mover contained, and this [relocation] "
                    "tracks nothing ([relocation.follow] or "
                    "[[relocation.move]]); add a follow source, or delete "
                    "the containment table")
            if int(self.containment.grid_id) == int(self.grid_id):
                raise ValueError(
                    "[relocation.containment] grid_id equals the mover's "
                    "own; containment names a STRICT ANCESTOR of the "
                    "tracked mover")
        from woof.core.nest_reach import validate_reach_speed
        object.__setattr__(self, "reach_speed_m_s", validate_reach_speed(
            self.reach_speed_m_s, "[relocation]"))
        if self.reach_speed_m_s is not None and self.follow is None:
            raise ValueError(
                "reach_speed_m_s bounds how far the TRACKER may take the "
                "nest, and this [relocation] has no [relocation.follow] "
                "block: a [[relocation.move]] itinerary's rows already say "
                "exactly where the nest goes, so the key would bound "
                "nothing.  Delete it, or add a follow block.")
        if self.track is not None and self.follow is None:
            raise ValueError(
                "[relocation.track] writes the vortex position the "
                "TRACKER finds, and this [relocation] has no "
                "[relocation.follow] block -- a scripted "
                "[[relocation.move]] itinerary knows where to put the "
                "nest but nothing about where the storm is. Add a follow "
                "block, or delete the track table.")
        cadence = self.cadence_seconds
        if cadence is not None:
            if self.follow is None and not self.moves:
                raise ValueError(
                    "cadence_seconds names how often the follow source is "
                    "consulted, and this [relocation] has no follow source "
                    "([relocation.follow] or [[relocation.move]]); add "
                    "one, or delete the key")
            if not math.isfinite(float(cadence)) or float(cadence) <= 0.0:
                raise ValueError(
                    f"cadence_seconds = {cadence!r} must be a finite, "
                    "positive number of model seconds")
        if self.moves:
            previous = 0.0
            for move in self.moves:
                at = float(move.at_seconds)
                if at <= previous:
                    raise ValueError(
                        "[[relocation.move]] rows must be strictly "
                        f"increasing in at_seconds; {at} follows {previous}")
                if cadence is not None:
                    multiples = at / float(cadence)
                    if abs(multiples - round(multiples)) > _REL_TOL * max(
                            1.0, abs(multiples)):
                        raise ValueError(
                            f"at_seconds = {at} is not a whole number of "
                            f"cadence_seconds = {float(cadence)}; a move "
                            "can only fire at a cadence opportunity")
                previous = at

    def receipt(self) -> dict[str, object]:
        """The accepted configuration, echoed value for value."""
        return {
            "enabled": bool(self.enabled),
            "grid_id": (None if self.grid_id is None else int(self.grid_id)),
            "mode": self.mode,
            "max_move_parent_cells": (
                None if self.max_move_parent_cells is None
                else int(self.max_move_parent_cells)),
            "min_overlap_fraction": (
                None if self.min_overlap_fraction is None
                else float(self.min_overlap_fraction)),
            "cadence_seconds": (
                None if self.cadence_seconds is None
                else float(self.cadence_seconds)),
            "follow": (None if self.follow is None
                       else self.follow.to_json()),
            "moves": [move.to_json() for move in self.moves],
            "containment": (None if self.containment is None
                            else self.containment.to_json()),
            "track": (None if self.track is None
                      else self.track.to_json()),
            # Echoed when set; absent it is the default, which the
            # corridor receipt and the runner's receipt state as a number.
            **({"reach_speed_m_s": float(self.reach_speed_m_s)}
               if self.reach_speed_m_s is not None else {}),
        }


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    """A validated experiment: domains stored parent-before-child.

    ``vertical``/``projection`` are the F1-amendment value objects:
    frozen/hashable, serialized verbatim in the resolved-TOML round trip,
    and inside the section-B experiment fingerprint -- two experiments
    differing in any eta level or projection parameter compare (and
    hash) distinct.  ``projection`` is required for real TOMLs (shared
    ``map_proj`` nonzero, WRF convention 1=lambert/2=polar/3=mercator);
    ``None`` is reserved for wrapped idealized configs
    (:func:`experiment_from_run_config`).

    TIMING AUTHORITY (F14 amendment): ``run_seconds`` /
    ``restart_interval_s`` here and each domain's ``history_interval_s``
    are the ONLY authoritative timing values; the embedded per-domain
    RunConfig copies are derived and asserted equal at load.

    ``blend_width`` defaults to 5 per Registry.EM_COMMON:2324 (``rconfig
    integer blend_width ... 5 "width of cg fg terrain blended zone"``).
    ``feedback`` is a tree-wide switch: 0 is the production one-way path
    and 1 enables the experimental child-to-parent restriction path.
    ``smooth_option`` selects WRF's post-feedback parent smoother
    (interp_fcn.F:3794-4014): 0 none, 1 ``sm121``, 2 ``smdsm`` (WRF's
    Registry default).  It is read only when feedback = 1, exactly as
    WRF's ``SUBROUTINE smoother`` returns before dispatching when
    feedback is off (:3823-3824).
    """

    name: str
    start_time: datetime
    run_seconds: float
    vertical: VerticalConfig
    projection: ProjectionConfig | None = None
    feedback: int = 0
    smooth_option: int = 0
    blend_width: int = 5
    spec_bdy_width: int = 5
    restart_interval_s: float
    domains: tuple[DomainConfig, ...]
    column_chunk: int = DEFAULT_COLUMN_CHUNK
    acknowledgements: tuple[str, ...] = ()
    #: The downward longwave a constant-GLW experiment DECLARES, W m-2.
    #: ``None`` means the shipped default
    #: (:data:`woof.core.physics.DECLARED_CONSTANT_GLW_WM2`).  Read only
    #: when the acknowledgement that fabricates GLW is present -- the
    #: refusal it lifts is unchanged, and this is not a way to lift it --
    #: and it exists because the acknowledgement could name the CLAIM but
    #: not the NUMBER: an experiment whose case radiates near 410 W m-2
    #: had to run at 300 and call the difference declared.  The engine
    #: takes any float here already; the receipt prints the value.
    constant_glw_wm2: float | None = None
    #: Discrete-relocation admissibility bounds.  Default-disabled, so an
    #: experiment that never mentions [relocation] carries a static nest
    #: and is byte-for-byte the experiment it was before this existed.
    relocation: RelocationConfig = RelocationConfig()
    #: The resolved physics-fidelity axis (:mod:`woof.physics_mode`).  The
    #: resolved values are ALSO baked into each domain's RunConfig, so the
    #: fingerprint would bind the physics without this field; it is carried
    #: anyway because a receipt that says "arwen-patched, patchset v1,
    #: patches [L4]" is readable and a pair of selector values is not.
    #: ``physics_mode.UNGOVERNED`` is the state of every configuration that
    #: does not name the axis, and it authors nothing.
    physics_mode: physics_mode_module.PhysicsModeResolution = (
        physics_mode_module.UNGOVERNED)
    #: The validated [perturbation] block, or ``None`` when the config
    #: does not carry one.  ``None`` is the OFF contract: no applier is
    #: built, no initialization branch runs, and the restart-identity
    #: payload omits the key entirely so absent-block fingerprints stay
    #: byte-identical to pre-feature ones.
    perturbation: "PerturbationConfig | None" = None
    #: The resolved [tiles] block (:mod:`woof.core.streaming`).  An
    #: experiment that never mentions it carries ``StreamingOptions.OFF``,
    #: whose stepper IS ``dycore.step`` -- there is no disabled-streaming
    #: code path, only the absence of one.  Excluded from the restart
    #: identity on purpose: see ``streaming.identity_payload_entry``.
    tiles: "streaming_module.StreamingOptions" = streaming_module.OFF
    devices: "DeviceOptions" = DEVICES_OFF
    #: The resolved tree-wide [output] block
    #: (:mod:`woof.io.history_selection`).  An experiment that never
    #: mentions it carries ``HistorySelection.FULL``, which writes every
    #: variable the run produces and stamps no attribute -- the default
    #: is what every run before this surface existed did, byte for byte.
    #: Excluded from the restart identity for the same reason ``tiles``
    #: is: it changes no number the model computes.
    output: "object" = history_selection_module.FULL
    #: Radar products observe saved history and do not change restart state.
    simulated_radar: simulated_radar_module.SimulatedRadarOptions = simulated_radar_module.OFF
    #: The resolved [radar_heating] table (:class:`woof.config.RadarHeatingConfig`).
    #: OFF when absent, and then absent from every document and identity.
    #: Present, it binds the restart identity (minus the windows path), so a
    #: checkpoint written heated cannot resume under other heating or none.
    radar_heating: "RadarHeatingConfig" = RADAR_HEATING_OFF
    #: grid_ids whose ``mix_isotropic`` was CHOSEN BY THE MODEL because
    #: the config left it unset or wrote the ``"auto"`` sentinel (ArWen's
    #: 2026-08-16 auto-switch ruling; ``resolve_auto_mix_isotropic``).
    #: A provenance LABEL, not a trajectory input: the chosen value sits
    #: on each domain's ``run.mix_isotropic``, which is what the restart
    #: identity binds, so this field leaves ``restart_identity_payload``
    #: and the sealed-extension identity unconditionally -- an
    #: auto-selected 1 and a written 1 are the same run, and each must
    #: resume the other's checkpoints.  The empty default keeps every
    #: code-constructed experiment (``experiment_from_run_config``, the
    #: frozen verify fixtures) on explicit semantics: what its author
    #: set is what runs.
    auto_mix_isotropic: tuple[int, ...] = ()
    #: grid_ids whose ``epssm`` is the MODEL'S CHOICE: the config left it
    #: unset or wrote the ``"auto"`` sentinel, which is also what
    #: ``woof import-namelist`` writes where WRF's Registry default fills
    #: a domain the namelist did not assign.  Such a domain is raised to
    #: the measured off-centering floor over steep ground
    #: (:func:`woof.acoustic_adaptation.adapt_experiment_acoustics`); a
    #: written value below the floor is refused instead.  A provenance
    #: LABEL like ``auto_mix_isotropic``: the value that runs sits on
    #: ``run.epssm`` and binds there, so this field leaves the restart and
    #: sealed-extension identities.  The empty default keeps every
    #: code-constructed experiment on explicit semantics.
    auto_epssm: tuple[int, ...] = ()
    #: The validated [spectral_numerics] block
    #: (:class:`woof.spectral_ops.config.SpectralNumericsConfig`), or
    #: ``None`` when the config does not carry one.  ``None`` is the OFF
    #: contract: no hook is built, the slow-step seam pays one ``is None``
    #: test, and the restart-identity payload omits the key so
    #: absent-block fingerprints stay byte-identical to pre-feature ones.
    #: A PRESENT block binds the restart identity value for value (the
    #: Level-2 contract: restart/config identity includes the resolved
    #: operator config) and is honored only at the
    #: ``execute_experiment`` slow-large-step commit point; every other
    #: integrating loop refuses an active block
    #: (:func:`refuse_unrouted_spectral_numerics`).
    spectral_numerics: object | None = None
    #: Opt-in analysis relaxation, selected per domain. Absent means no nudging.
    nudging: object | None = None
    #: WRF's &domains smooth_cg_topo (Registry.EM_COMMON:2301, scope 1,
    #: default .false.): domain 1's outer rows take the input model's
    #: terrain and ramp back to the high-resolution terrain over
    #: blend_width rows (woof.ingest.cg_topo).  False, the default, builds
    #: every root terrain exactly as before and drops out of the restart
    #: identity; True binds it, because the terrain is part of the run.
    smooth_cg_topo: bool = False
    #: The validated [physics_params] set
    #: (:class:`woof.physics_params.PhysicsParamSet`), or ``None`` when the
    #: config does not carry one.  ``None`` is the default contract: every
    #: kernel literal and parameter table is WRF's/today's and the restart
    #: identity omits the key.  A PRESENT set binds the restart identity
    #: value for value, because a resume under other constants is another
    #: trajectory.  Last field on purpose: a mid-dataclass field would move
    #: every positional construction after it.
    physics_params: object | None = None

    def __post_init__(self):
        if self.feedback not in FEEDBACK_OPTIONS:
            raise ValueError(
                "feedback must be 0 (one-way) or 1 (experimental "
                f"two-way), got {self.feedback!r}.")
        if (any(dc.run.fire_smoke for dc in self.domains)
                and not any(dc.run.ifire == 2 for dc in self.domains)):
            raise ValueError("bulk SFIRE smoke needs an actual ifire=2 source domain in the tree")
        if self.smooth_option not in SMOOTH_OPTION_OPTIONS:
            raise ValueError(
                "smooth_option must be 0 (none), 1 (sm121) or 2 (smdsm), "
                f"got {self.smooth_option!r}.")
        if (isinstance(self.column_chunk, bool)
                or not isinstance(self.column_chunk, int)
                or self.column_chunk < 1):
            raise ValueError(
                "column_chunk must be a positive integer number of "
                f"radiation columns, got {self.column_chunk!r}.")
        if (
            not isinstance(self.acknowledgements, tuple)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in self.acknowledgements
            )
        ):
            raise ValueError(
                "acknowledgements must be a tuple of non-empty ids, got "
                f"{self.acknowledgements!r}.")
        if self.constant_glw_wm2 is not None:
            import math as _math
            if (isinstance(self.constant_glw_wm2, bool)
                    or not isinstance(self.constant_glw_wm2, (int, float))
                    or not _math.isfinite(float(self.constant_glw_wm2))
                    or float(self.constant_glw_wm2) <= 0.0):
                raise ValueError(
                    "constant_glw_wm2 must be a finite positive downward "
                    "longwave flux in W m-2, got "
                    f"{self.constant_glw_wm2!r}.")
            from woof.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK
            if CONSTANT_DOWNWARD_LONGWAVE_ACK not in self.acknowledgements:
                raise ValueError(
                    "constant_glw_wm2 names the fabricated downward "
                    "longwave a run consumes for its whole forecast, so it "
                    "is only readable where that fabrication is declared: "
                    "this experiment does not carry "
                    f"acknowledgements = [{CONSTANT_DOWNWARD_LONGWAVE_ACK!r}]. "
                    "A run with a longwave scheme has its GLW written by "
                    "that scheme every radiation step and would ignore "
                    "this number, which is exactly the setting that looks "
                    "like it took effect and did not. Remove the key, or "
                    "declare the fabrication.")

        from dataclasses import replace
        from woof.config import radiation_scheme_ids
        from woof.core.ozone_contract import cam_ozone_domain_ids
        ozone_domains = cam_ozone_domain_ids(self)
        if ozone_domains or any(radiation_scheme_ids(dc.run)[0] == 1 for dc in self.domains):
            context = streaming_module.RadiationMemoryContext(
                self.column_chunk, self.vertical.p_top,
                self.root.grid_id in ozone_domains, ozone_domains)
            object.__setattr__(self, "tiles", replace(self.tiles, radiation_context=context))
            object.__setattr__(self, "domains", tuple(
                replace(dc, tiles=replace(dc.tiles, radiation_context=context))
                if dc.tiles is not None else dc for dc in self.domains))

        from woof.core.uh_diag import declared_follower_slots
        follower_slots = declared_follower_slots(self.domains)
        context = (streaming_module.FollowerWindowMemoryContext(
            tuple(sorted(follower_slots.items())),
            follower_slots.get(int(self.root.grid_id), ())) if follower_slots else None)
        validate_device_road(self.devices, self.tiles, self.domains)
        if context is not None or self.tiles.follower_context is not None:
            object.__setattr__(self, "tiles", replace(self.tiles, follower_context=context))
        object.__setattr__(self, "domains", tuple(
            replace(dc, tiles=replace(dc.tiles, follower_context=context))
            if dc.tiles is not None and (context is not None or dc.tiles.follower_context is not None)
            else dc for dc in self.domains))

        if (self.tiles.mode == "auto"
                or any(dc.tiles is not None and dc.tiles.mode == "auto"
                       for dc in self.domains)
                or (len(self.domains) == 1 and (self.tiles.enabled
                    or any(dc.tiles is not None and dc.tiles.enabled
                           for dc in self.domains)))):
            # Retain the actual scientific/forcing configuration, without a
            # reference back through its tiles options. Pricing stays lazy.
            snapshot = replace(self, tiles=streaming_module.OFF,
                               domains=tuple(replace(dc, tiles=None)
                                             for dc in self.domains))
            context = streaming_module.ResidentAdmissionContext(snapshot)
            object.__setattr__(self, "tiles", replace(self.tiles, resident_context=context))
            object.__setattr__(self, "domains", tuple(
                replace(dc, tiles=replace(dc.tiles, resident_context=context))
                if dc.tiles is not None else dc for dc in self.domains))

    @property
    def root(self) -> DomainConfig:
        return self.domains[0]

    def domain(self, grid_id: int) -> DomainConfig:
        for dc in self.domains:
            if dc.grid_id == grid_id:
                return dc
        raise KeyError(
            f"no domain with grid_id={grid_id}; experiment {self.name!r} "
            f"has {[dc.grid_id for dc in self.domains]}")

    def children_of(self, grid_id: int) -> tuple[DomainConfig, ...]:
        self.domain(grid_id)  # KeyError on unknown ids
        return tuple(dc for dc in self.domains if dc.parent_id == grid_id)

    def domain_start_time(self, grid_id: int) -> datetime:
        """Absolute valid time at which one configured domain becomes live.

        ``None`` is retained only for programmatic legacy fixtures created
        before per-domain starts entered the schema; it means the experiment
        start.  Every TOML-loaded domain carries an explicit resolved value.
        """
        value = self.domain(grid_id).start_time
        return self.start_time if value is None else value

    def domain_start_offset_exact(self, grid_id: int) -> Fraction:
        """Exact seconds from the experiment start to a domain start."""
        delta = self.domain_start_time(grid_id) - self.start_time
        return (Fraction(delta.days * 86400 + delta.seconds)
                + Fraction(delta.microseconds, 1_000_000))

    def dt_exact(self, grid_id: int) -> Fraction:
        """The domain's model step as an EXACT rational (seconds).

        Root: ``time_step + time_step_fract_num/time_step_fract_den``
        (Registry.EM_COMMON:2245-2246).  Child: the parent's rational dt
        divided by ``parent_time_step_ratio`` exactly, as WRF's
        ``stepTime = parent stepTime / parent_time_step_ratio``
        (share/set_timekeeping.F:366-368).  Bundle chain (1,4,3,3):
        60, 15, 5, 5/3 s.
        """
        dc = self.domain(grid_id)
        if dc.parent_id == 0:
            return (Fraction(dc.time_step)
                    + Fraction(dc.time_step_fract_num,
                               dc.time_step_fract_den))
        return self.dt_exact(dc.parent_id) / dc.parent_time_step_ratio

    def dx_exact(self, grid_id: int) -> Fraction:
        """The domain's grid spacing as an EXACT rational (metres):
        root ``dx``, children ``dx_parent / parent_grid_ratio`` exactly.
        Bundle chain (1,4,3,3): 12000, 3000, 1000, 1000/3 m."""
        dc = self.domain(grid_id)
        if dc.parent_id == 0:
            return Fraction(dc.run.dx)
        return self.dx_exact(dc.parent_id) / dc.parent_grid_ratio


#: :func:`config_path_kind`'s answer for a zero-byte file.  Its own
#: sentence names a remedy the other kinds have no use for, so the
#: readers that need to tell it apart compare against this rather than
#: matching prose.
EMPTY_CONFIG_FILE = "is empty"

#: The opening words of the kind reported when ``is_file()`` ITSELF
#: failed -- a symlink loop, a permission wall, a dead mount.  The rest
#: of that kind is the errno's own text, so it cannot be a constant.
_UNREADABLE_CONFIG_PREFIX = "cannot be read as a configuration file"


def config_path_kind(path: str | Path) -> str | None:
    """``None`` when ``path`` is a configuration file that can be opened,
    otherwise the words for what it is instead.

    ONE vocabulary for "that is not a config file", because two doors ask
    the question.  :func:`readable_config_path` asks it to refuse, and
    ``woof resume`` asks it of each rung of its resolution ladder to say
    what each candidate was instead -- and a resume whose ladder reported
    "missing" for a directory, while the loader one call later called the
    same path "a directory", would be two answers about one file.

    The kind is decided BEFORE anything opens the path.  ``is_file()`` is
    what separates a regular file from a directory, a FIFO, a device and
    a broken symlink in one call, and it is the guard the fleet's
    hostile-input node asked for by name.
    """

    path = Path(path)
    try:
        if path.is_file():
            if path.stat().st_size == 0:
                return EMPTY_CONFIG_FILE
            return None
    except OSError as error:
        return f"{_UNREADABLE_CONFIG_PREFIX} ({error.strerror or error})"
    if path.is_dir():
        return "is a directory"
    if path.exists():
        return "is not a regular file (a device, socket or FIFO)"
    return "does not exist"


def readable_config_path(path: str | Path) -> Path:
    """``path`` as a readable regular file, or a refusal saying which
    kind of thing it actually is.

    A mistyped config path is the commonest error there is, and in
    v1.4.0 every wrong-KIND of path reached ``open()`` and came back as
    a Python traceback at exit 1: ``FileNotFoundError`` for a typo,
    ``IsADirectoryError`` for a directory, an eight-argument
    ``TypeError: RunConfig.__init__()`` for a zero-byte file, ``OSError
    [Errno 40]`` for a symlink loop -- and, for a FIFO, no return at
    all, because ``tomllib`` reads it forever.  Every other refusal in
    this product is one sentence at exit 2.

    So the kind is decided BEFORE anything opens the path.  ``is_file()``
    is what separates a regular file from a directory, a FIFO, a device
    and a broken symlink in one call, and it is the guard the fleet's
    hostile-input node asked for by name.
    """

    path = Path(path)
    kind = config_path_kind(path)
    if kind is None:
        return path
    if kind == EMPTY_CONFIG_FILE:
        raise ValueError(layered(
            f"{path} is empty, so there is no configuration in "
            "it to run.\n"
            "  remedy: woof domain ... --out "
            f"{path}   # re-author it",
            "A zero-byte TOML parses to an empty table, which "
            "used to reach the RunConfig constructor and come "
            "back as its argument list."))
    if kind.startswith(_UNREADABLE_CONFIG_PREFIX):
        # A symlink loop, a permission wall, a dead mount: is_file()
        # itself is what failed, and its errno is the diagnosis.
        #
        # It carries a remedy for the same reason the other three kinds
        # do.  This one used to end at the errno and a full stop, so the
        # reader with the LEAST to go on -- the kind of the path is not
        # even known here -- was the one given nothing to do next.
        raise ValueError(layered(
            f"{path} {kind}, so what it is could not be decided.\n"
            "  remedy: resolve the path and check this account can read "
            "it (a symbolic-link loop, a permission wall and a mount "
            "that is gone all answer this way), then pass the "
            "experiment .toml that `woof domain` wrote.",
            "The errno is quoted as the operating system gave it, and it "
            "comes from the KIND check rather than from a read: nothing "
            "is opened until the path is known to be a readable regular "
            "file, because a FIFO would block this process forever."))
    raise ValueError(layered(
        f"{path} {kind}; pass the experiment .toml that `woof domain` "
        "wrote.",
        "Nothing is opened until the path is known to be a readable "
        "regular file: a FIFO would block this process forever, and a "
        "directory or a missing file used to arrive as a Python "
        "traceback rather than as a refusal."))


def is_experiment_toml(path: str | Path) -> bool:
    """True when ``path`` is an experiment TOML (``[experiment]`` or
    ``[[domain]]`` present) rather than a legacy RunConfig TOML."""
    from woof.config_authority import read_config_authority

    return is_experiment_toml_bytes(read_config_authority(path).payload)


def is_experiment_toml_bytes(payload: bytes) -> bool:
    """Byte-oriented route detection for one already captured config."""

    raw = tomllib.load(io.BytesIO(payload))
    return "experiment" in raw or "domain" in raw


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML (fail-loud, section A).

    The one-file case schema's companion tables are split off and
    VALIDATED here -- never silently dropped, and never refused as
    unknown.  ``[fetch]`` (advisory hints, schema owned by
    :func:`woof.fetch.validate_fetch_hints`) always was; ``[case_data]``
    and ``[static]`` now are too, against their own owners' schemas
    (:func:`woof.case_data.build_case_data` without the input-existence
    check -- this loader answers "what experiment is this", which must
    not require the declared inputs to be fetched yet -- and
    :func:`woof.static.highres_production.parse_static_table`).  Before
    task #204 this loader refused the wizard's own ERA5 emission -- the
    exact file ``woof domain --source era5`` writes -- as "does not
    have a table 'case_data'", from every front door that loads through
    here.  Callers that CONSUME the case declarations load through
    :func:`woof.case_data.load_experiment_case` instead; the experiment
    schema itself stays strict.
    """
    from woof.config_authority import read_config_authority

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    source = str(authority.source)
    from woof.config import _anchor_config_file_paths
    _anchor_config_file_paths(raw.get("shared", {}), authority.source,
                              keys=("rrtmg_smoke_manifest", "wind_farm_table", "wind_farms"))
    for domain in raw.get("domain", []):
        _anchor_config_file_paths(domain, authority.source, keys=("wind_farm_table", "wind_farms"))
    base_dir = Path(authority.source).parent
    experiment = build_experiment_from_config_tables(
        raw, source=source, base_dir=base_dir)
    review_root_footprint(experiment, source)
    return experiment


def review_root_footprint(experiment: "ExperimentConfig", source: str) -> None:
    """Keep a Lambert root clear of its singular cone tip.

    Polar stereographic is regular through its hemisphere's pole. Its
    source and static windows include periodic donor halos, so a cap is
    valid through file-backed loading as well as wizard fitting.
    """

    projection = experiment.projection
    if projection is None or projection.map_proj == "polar":
        return
    root = experiment.root.run
    if not footprint_contains_pole(
            {"map_proj": projection.map_proj, "ref_lat": projection.ref_lat,
             "ref_lon": projection.ref_lon, "truelat1": projection.truelat1,
             "truelat2": projection.truelat2,
             "stand_lon": projection.stand_lon},
            root.nx, root.ny, root.dx):
        return
    pole = "north" if projection.truelat1 >= 0.0 else "south"
    raise ValueError(
        f"experiment config {source}: the root domain ({root.nx} x "
        f"{root.ny} mass points at {root.dx / 1000:g} km on "
        f"{projection.map_proj!r}) contains or touches the {pole} pole; "
        "Lambert longitude is singular there. Set [projection] map_proj "
        "to 'polar' for a pole-enclosing domain, or move ref_lat "
        "away from the pole or shrink the root "
        "(nx/ny/dx), until the footprint clears it.")


def build_experiment_from_config_tables(raw: dict, *, source: str,
                                       base_dir: Path) -> ExperimentConfig:
    """Validate companion owners and build the experiment from parsed authority.

    Consume a private table copy, leaving the caller's publication authority
    intact. This is the same schema boundary used for file-backed loading.
    """
    import copy

    from woof.vertical_contract import validate_config_source_top

    validate_config_source_top(raw, context=source)
    raw = copy.deepcopy(raw)
    from woof.config import _anchor_config_file_paths
    for table in [raw.get("shared", {}), *raw.get("domain", [])]:
        _anchor_config_file_paths(table, base_dir / "experiment.toml",
                                  keys=("wind_farm_table", "wind_farms"))
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from woof.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=source)
        # A run that starts from HRRR with smoke on takes HRRR-Smoke's
        # smoke as its start state and edge values (the source's row).
        from woof.config import apply_route_chem_sources
        apply_route_chem_sources(raw, fetch_table)
        # A configuration that names an operational-fork source (hrrr*,
        # rap*) and omits a scheme-generation selector runs that source's
        # generation, not the generic WRF v4.6.1 one: the door's carried
        # experiment.toml predated these selectors and silently ran the
        # v4.6.1 surface layer and Thompson against HRRR's own analysis
        # (woof.physics_source_defaults.GENERATION_SELECTORS).  Written
        # keys are kept as written.
        from woof.physics_source_defaults import fill_omitted_generation_selectors
        fill_omitted_generation_selectors(raw, fetch_table)
    case_table = raw.pop("case_data", None)
    if case_table is not None:
        from woof.case_data import build_case_data
        build_case_data(case_table, source=source, base_dir=base_dir,
                        require_inputs=False)
    static_table = raw.pop("static", None)
    if static_table is not None:
        from woof.static.highres_production import parse_static_table
        parse_static_table(static_table, source=source, base_dir=base_dir)
    ingest_table = raw.pop("ingest", None)
    if ingest_table is not None:
        from woof.ingest.soil_downscale import parse_ingest_table
        parse_ingest_table(ingest_table, source=source)
    return build_experiment(raw, source=source)


def drop_unreached_sase_selectors(raw: dict, kept_runs) -> None:
    """Remove [shared] SASE selectors a cut-down tree has no domain for.

    ``raw`` holds a tree's tables already cut to some of its domains, and
    ``kept_runs`` are those domains' RunConfigs as the WHOLE tree resolved
    them.  On a tree that mixes SASE with another PBL scheme the loader
    applies the [shared] SASE selectors to the SASE domains only, so when
    none of the kept domains runs SASE the selectors reached none of them:
    each kept domain ran the defaults.  Left in [shared], the cut-down
    tree would be refused for a key naming a seam it does not have -- the
    HRRR root preparation of a YSU root under a SASE nest stopped that
    way.  With a SASE domain kept, the selectors stay and reach it.
    """
    if any(run.bl_pbl_physics == SASE_PBL_SCHEME for run in kept_runs):
        return
    shared = raw.get("shared")
    if isinstance(shared, dict):
        for key in SASE_FAIL_CLOSED_DEFAULTS:
            shared.pop(key, None)


def drop_unreached_grell_selectors(raw: dict, kept_runs) -> None:
    """Remove [shared] Grell-family keys a cut-down tree has no domain for.

    The Grell-Freitas twin of :func:`drop_unreached_sase_selectors`: on a
    tree that mixes Grell-Freitas with another cumulus choice the loader
    applies a [shared] ``clos_choice`` or ``ishallow`` to the
    Grell-Freitas domains only, so a cut-down tree keeping none of them
    ran the Registry defaults, and left in [shared] the keys would refuse
    it.  With a Grell-Freitas domain kept, they stay and reach it.
    """
    shared = raw.get("shared")
    if not any(run.cu_physics in SAS_PGCON_SCHEMES for run in kept_runs):
        # The SAS-family keys, on the same terms.
        if isinstance(shared, dict):
            for key in SAS_FAMILY_DEFAULTS:
                shared.pop(key, None)
    if any(run.cu_physics in GRELL_FAMILY_CU_PHYSICS for run in kept_runs):
        return
    if isinstance(shared, dict):
        for key in GRELL_FAMILY_DEFAULTS:
            shared.pop(key, None)


def drop_unreached_relocation(raw: dict, kept_grid_ids) -> None:
    """Remove a [relocation] table whose mover a cut-down tree has no domain for.

    ``raw`` holds a tree's tables already cut to some of its domains, and
    ``kept_grid_ids`` are the grid ids it kept.  Everything in
    [relocation] belongs to the one child its ``grid_id`` names: the
    tracker or itinerary that moves it, the ancestor slide that keeps it
    contained, the track file of the vortex it follows.  With that child
    cut away the table has nothing left to drive, and left in place the
    loader refuses the cut-down tree for a ``grid_id`` it does not have:
    the HRRR root preparation of every storm-following layout stopped
    that way.  A disabled [relocation] must be empty, so the table goes
    whole rather than switched off.  With the mover kept, the table stays
    and the loader checks it as it checks any tree.
    """
    table = raw.get("relocation")
    if not isinstance(table, dict) or table.get("grid_id") is None:
        return
    try:
        mover = int(table["grid_id"])
    except (TypeError, ValueError):
        # Not a grid id at all: the loader refuses it by name.
        return
    if mover not in {int(grid_id) for grid_id in kept_grid_ids}:
        del raw["relocation"]


def experiment_from_run_config(cfg: RunConfig,
                               start_time: datetime) -> ExperimentConfig:
    """Wrap a scalar :class:`RunConfig` as a one-domain experiment.

    The RunConfig is carried VERBATIM (no flag rewriting: a periodic
    idealized config wraps as-is -- the multi-domain root-flag rule binds
    only at TOML load).  ``cfg.dt`` decomposes exactly into the WRF
    rational clock keys: ``time_step`` integer seconds plus the exact
    binary remainder as ``time_step_fract_num/den``.
    """
    dt = Fraction(cfg.dt)
    whole = dt.numerator // dt.denominator
    rem = dt - whole
    dom = DomainConfig(
        grid_id=cfg.grid_id, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=float(cfg.output_interval_s), run=cfg,
        time_step=int(whole), time_step_fract_num=rem.numerator,
        time_step_fract_den=rem.denominator, start_time=start_time)
    return ExperimentConfig(
        name=cfg.case or "run_config", start_time=start_time,
        run_seconds=float(cfg.run_seconds),
        # eta_levels/p_top are unknown to a scalar RunConfig (legacy
        # cases own their vertical grids); the hybrid selectors are the
        # compatibility copies.  projection = None is RESERVED for these
        # wrapped idealized/legacy configs (F1 amendment).
        vertical=VerticalConfig(eta_levels=(), p_top=0.0,
                                hybrid_opt=cfg.hybrid_opt, etac=cfg.etac),
        projection=None,
        restart_interval_s=float(cfg.restart_interval_s),
        domains=(dom,), spec_bdy_width=cfg.spec_bdy_width)


# ---------------------------------------------------------------------------
# Loader internals
# ---------------------------------------------------------------------------

def _reject_moving_nest_keys(table_name: str, entries: dict,
                             source: str) -> None:
    present = sorted(_MOVING_NEST_KEYS & set(entries))
    if present:
        raise ValueError(
            f"WRF moving-nest key(s) {present} in [{table_name}] of "
            f"{source} are rejected: they drive nest motion from INSIDE "
            "the integration -- WRF's specified moves land on whole parent "
            "cells too, but on the model's own schedule, rebuilding the "
            "donor tables as it runs -- and woof's SINT donor "
            "index/weight tables are built once per placement generation, "
            "which is what makes a move's overlap transplant bitwise and "
            "its donor alignment checkable. A nest that follows weather is "
            "expressed here as DISCRETE relocation instead -- whole parent "
            "cells at cycle boundaries -- through the [relocation] table "
            "(enabled = true): [[relocation.move]] for a specified "
            "itinerary, or [relocation.follow] with [relocation.track] for "
            "storm following. The vortex controls have no equivalent at "
            "all: woof's tracker is field/threshold/cooldown shaped, not "
            "corral/max-speed shaped, so translating them would substitute "
            "different physics under the same key. Remove the key(s).")


def _build_relocation(raw: dict, source: str, domains,
                      run_seconds: float) -> RelocationConfig:
    """Validate ``[relocation]``, refusing every key while it is off."""
    if "relocation" not in raw:
        return RelocationConfig()
    table = dict(raw["relocation"])
    _reject_moving_nest_keys("relocation", table, source)
    _reject_unknown_keys("relocation", table, _RELOCATION_KEYS, source)
    enabled = table.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"enabled in [relocation] of {source} must be a boolean, got "
            f"{enabled!r}.")
    if not enabled:
        stray = sorted(set(table) - {"enabled"})
        if stray:
            raise ValueError(
                f"[relocation] of {source} carries {stray} while enabled "
                "is false or absent. A relocation surface that is off must "
                "be empty, so a nest cannot start moving because a block "
                "was inherited or a flag was flipped somewhere else; set "
                "enabled = true deliberately, or delete the key(s).")
        return RelocationConfig()
    domain_ids = [dc.grid_id for dc in domains]
    grid_id = table.get("grid_id")
    if grid_id is not None and int(grid_id) not in set(domain_ids):
        raise ValueError(
            f"grid_id = {grid_id!r} in [relocation] of {source} is not a "
            f"domain of this experiment (have {sorted(domain_ids)}).")
    rows = table.get("move", [])
    if not isinstance(rows, list):
        raise ValueError(
            f"move in [relocation] of {source} must be written as "
            "[[relocation.move]] array-of-tables rows, got "
            f"{type(rows).__name__}.")
    moves = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"[[relocation.move]] row {index} of {source} is not a "
                "table.")
        row = dict(row)
        _reject_unknown_keys(
            f"relocation.move[{index}]", row, _RELOCATION_MOVE_KEYS, source)
        _require_keys(f"relocation.move[{index}]", row, ("at_seconds",),
                      source)
        try:
            moves.append(ScheduledRelocationMove(
                at_seconds=float(row["at_seconds"]),
                di_parent_cells=row.get("di_parent_cells", 0),
                dj_parent_cells=row.get("dj_parent_cells", 0)))
        except ValueError as err:
            raise ValueError(
                f"[[relocation.move]] row {index} of {source}: {err}"
            ) from None
    follow = None
    if "follow" in table:
        if not isinstance(table["follow"], dict):
            raise ValueError(
                f"follow in [relocation] of {source} must be the "
                "[relocation.follow] TABLE (the storm-tracking block), got "
                f"{table['follow']!r}.")
        from woof.core.storm_tracking import build_follow_config
        follow = build_follow_config(dict(table["follow"]), source)
    containment = None
    if "containment" in table:
        if not isinstance(table["containment"], dict):
            raise ValueError(
                f"containment in [relocation] of {source} must be the "
                "[relocation.containment] TABLE (the ancestor-slide "
                f"block), got {table['containment']!r}.")
        crow = dict(table["containment"])
        _reject_unknown_keys("relocation.containment", crow,
                             _RELOCATION_CONTAINMENT_KEYS, source)
        _require_keys("relocation.containment", crow, ("grid_id",), source)
        try:
            containment = ContainmentConfig(
                grid_id=int(crow["grid_id"]),
                deadband_cells=int(crow.get("deadband_cells", 8)),
                max_move_parent_cells=(
                    None if crow.get("max_move_parent_cells") is None
                    else int(crow["max_move_parent_cells"])),
                cadence_seconds=(
                    None if crow.get("cadence_seconds") is None
                    else float(crow["cadence_seconds"])))
        except ValueError as err:
            raise ValueError(
                f"[relocation.containment] of {source}: {err}") from None
    track = _RELOCATION_KEY_ROWS["track"].get(
        table, where=f"[relocation] of {source}")
    if track is not None:
        from woof.core.storm_track_writer import build_track_config
        track = build_track_config(track, source)
    try:
        relocation = RelocationConfig(
            enabled=True,
            grid_id=(None if grid_id is None else int(grid_id)),
            mode=str(table.get("mode", DISCRETE_RELOCATION_MODE)),
            max_move_parent_cells=(
                None if table.get("max_move_parent_cells") is None
                else int(table["max_move_parent_cells"])),
            min_overlap_fraction=(
                None if table.get("min_overlap_fraction") is None
                else float(table["min_overlap_fraction"])),
            cadence_seconds=(
                None if table.get("cadence_seconds") is None
                else float(table["cadence_seconds"])),
            follow=follow,
            moves=tuple(moves),
            containment=containment,
            track=track,
            reach_speed_m_s=table.get("reach_speed_m_s"))
    except ValueError as err:
        raise ValueError(f"[relocation] of {source}: {err}") from None
    if containment is not None:
        # A strict ancestor, proven against the tree rather than trusted:
        # the whole design (compensated ride-along under the slide) rests
        # on the mover being INSIDE the sliding domain.
        by_id = {int(dc.grid_id): dc for dc in domains}
        if int(containment.grid_id) not in by_id:
            raise ValueError(
                f"[relocation.containment] grid_id = "
                f"{containment.grid_id} of {source} is not a domain of "
                f"this experiment (have {sorted(by_id)}).")
        mover_parent = int(by_id[int(relocation.grid_id)].parent_id)
        if int(containment.grid_id) != mover_parent:
            raise ValueError(
                f"[relocation.containment] grid_id = "
                f"{containment.grid_id} of {source} is not the tracked "
                f"mover's parent (d{relocation.grid_id:02d}'s parent is "
                f"d{mover_parent:02d}); containment slides the frame the "
                "mover moves inside, and its dead-band is measured in "
                "that frame's cells.")
    # The runner fires at complete cycle boundaries (root steps, where
    # every parent-child pair is synchronized), so a cadence or a
    # scheduled move that does not land on one can never fire.  Refused
    # here BY NAME rather than discovered as an eternally stationary nest.
    root_dc = domains[0]
    root_dt = (Fraction(root_dc.time_step)
               + Fraction(root_dc.time_step_fract_num,
                          root_dc.time_step_fract_den))

    def _whole_root_steps(seconds: float) -> bool:
        steps = float(seconds) / float(root_dt)
        return abs(steps - round(steps)) <= _REL_TOL * max(1.0, abs(steps))

    if (relocation.cadence_seconds is not None
            and not _whole_root_steps(relocation.cadence_seconds)):
        raise ValueError(
            f"cadence_seconds = {relocation.cadence_seconds} in "
            f"[relocation] of {source} is not a whole number of root "
            f"steps (root dt = {float(root_dt)} s); relocations execute "
            "at parent-step boundaries, and this cadence never lands on "
            "one.")
    if (relocation.containment is not None
            and relocation.containment.cadence_seconds is not None
            and not _whole_root_steps(
                relocation.containment.cadence_seconds)):
        raise ValueError(
            f"cadence_seconds = "
            f"{relocation.containment.cadence_seconds} in "
            f"[relocation.containment] of {source} is not a whole number "
            f"of root steps (root dt = {float(root_dt)} s); it can never "
            "fire.")
    for move in relocation.moves:
        if not _whole_root_steps(move.at_seconds):
            raise ValueError(
                f"[[relocation.move]] at_seconds = {move.at_seconds} in "
                f"{source} is not a whole number of root steps (root dt "
                f"= {float(root_dt)} s); it can never fire.")
        if float(move.at_seconds) >= float(run_seconds):
            raise ValueError(
                f"[[relocation.move]] at_seconds = {move.at_seconds} "
                f"in {source} is at or past the end of the run "
                f"(run_seconds = {run_seconds}); it can never fire.")
    if relocation.follow is not None:
        _refuse_unservable_follow_cadence(
            relocation, domains, source, root_dt=root_dt)
    if relocation.track is not None:
        _refuse_unservable_track(relocation, domains, source,
                                 root_dt=root_dt,
                                 whole_root_steps=_whole_root_steps)
        relocation = _with_report_levels(relocation, source)
    return relocation


def _report_levels_for(relocation) -> tuple:
    """Surfaces ``output_level`` names that ``level_hpa`` does not steer.

    Order is ``output_level``'s own, so the track file's extra columns
    appear where the config asked for them; the surface block and any
    already-steered surface drop out because they are not extra.
    """
    from woof.core.storm_track_writer import SURFACE_LEVEL
    from woof.core.storm_tracking import levels_of

    track = getattr(relocation, "track", None)
    follow = getattr(relocation, "follow", None)
    if track is None or follow is None or track.output_level is None:
        return ()
    known = {round(float(SURFACE_LEVEL), 6)} | {
        round(float(v), 6) for v in levels_of(follow)}
    out, seen = [], set()
    for value in track.output_level:
        key = round(float(value), 6)
        if key in known or key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return tuple(out)


def _attach_report_levels(relocation, extra, source):
    """Build the follow config the extras imply, to surface its refusals
    at LOAD -- the caller keeps the refusal, ``_with_report_levels``
    keeps the result, and both go through this one constructor so they
    cannot disagree about what is admissible."""
    try:
        return _dc_replace(relocation.follow,
                           report_level_hpa=tuple(extra))
    except ValueError as err:
        raise ValueError(
            f"[relocation.track] output_level of {source}: {err}") from None


def _with_report_levels(relocation, source):
    """``relocation`` with the report-only surfaces on its follow block."""
    extra = _report_levels_for(relocation)
    if not extra:
        return relocation
    return _dc_replace(
        relocation, follow=_attach_report_levels(relocation, extra, source))


def _refuse_unservable_track(relocation, domains, source, *, root_dt,
                             whole_root_steps) -> None:
    """A track stream that cannot produce accurate records refuses AT LOAD.

    Two ways to configure one that cannot, each refused by name here
    rather than discovered as an empty file after a twelve-hour run:

    **An interval that can never fire.**  The runner only acts at
    complete cycle boundaries, so an interval that is not a whole number
    of root steps rounds to whichever boundary happens to follow it.
    Same contract ``cadence_seconds`` holds, same refusal.

    **A finer interval on a stash-backed field THE ROW READS.**  ``uh``
    and ``reflectivity`` are not reduced on demand: they are planes
    somebody else folded into a scratch slot on somebody else's rhythm,
    and asking for one between those instants gets an answer that does
    not mean what the tracker's answer means.

    That argument only bites if the row is derived from the plane, and
    for those two fields it no longer is: a rotation or echo tracker
    writes POSITION ONLY -- the mover's own centre, off the mover's own
    grid, exact at every cycle boundary -- so the runner skips locating
    on a track-only boundary and there is nothing stale to read.  The
    refusal is kept, expressed as the fields that are stash-backed AND
    not position-only, so it comes back on its own if either set ever
    changes; today that difference is empty.

    ``refl_10cm`` is stashed by the microphysics inside its
    ``refl_10cm_due`` branch, which follows the HISTORY cadence -- so a
    track emission off that cadence reads a plane from some earlier
    instant, or none at all, and issue #111's refusal
    (:func:`_refuse_unservable_follow_cadence`) is the same contract one
    step further out.  The UH follow window is reset once per relocation
    cadence by the runner, so a read between resets sees a PARTIAL
    accumulation: the track row would report a centre derived from two
    minutes of rotation while the nest was steered by six, and the file
    would disagree with the model it claims to describe.  "Two formats,
    one fix" is the whole design, and a partial window breaks it.

    Note what this refusal is NOT: reading a stash does not perturb the
    forecast.  The reset belongs to the mover's consultation and stays
    there, so a track stream cannot move the nest however often it
    looks.  This is about the accuracy of the record, not its safety.

    A ``pressure`` tracker reduces from the live prognostic column, which
    is valid at every cycle boundary, so it is exempt -- the same line
    :data:`woof.core.storm_tracking.STASH_BACKED_FIELDS` already draws
    for the follow cadence.

    THE 10-M WIND REFUSAL IS SCOPED THE SAME WAY, and for the same
    reason: it exists because ``vmax`` would be written as a plausible
    0.00 m/s under ``sf_sfclay_physics = 0``, and a position-only row has
    no ``vmax`` column to be wrong about.  Note this is a refusal on
    :data:`woof.core.storm_tracking.STASH_BACKED_FIELDS`' cousin
    question, not on the tracker itself:
    :func:`_refuse_unservable_follow_cadence` still applies in full,
    because the NEST is steered by the plane whatever the file says.
    """
    from woof.core.storm_track_writer import POSITION_ONLY_FIELDS
    from woof.core.storm_tracking import STASH_BACKED_FIELDS

    # BOTH refusals below are about COLUMNS the row would carry, so both
    # are scoped to the shape of the row.  A rotation or echo tracker
    # writes POSITION ONLY -- a clock and the mover's own centre, read
    # from the mover's own grid -- so it carries no peak wind to be wrong
    # about and reads no stash to be stale.  See POSITION_ONLY_FIELDS.
    tracked_field = getattr(relocation.follow, "field", None)
    position_only = tracked_field in POSITION_ONLY_FIELDS

    # WHICH BLOCKS, checked against what the tracker actually produces.
    # A column of all-NaN nobody can fill is the same defect as a track
    # file nobody configured, so a block the run cannot write refuses by
    # name here rather than appearing as a dead column after twelve
    # hours.
    if relocation.track.output_level is not None:
        from woof.core.storm_track_writer import SURFACE_LEVEL
        from woof.core.storm_tracking import levels_of

        if position_only:
            raise ValueError(
                f"[relocation.track] of {source} refuses output_level "
                f"under [relocation.follow] field = {tracked_field!r}: "
                "that file is a clock and the moving domain's own centre, "
                "with no surface block and no isobaric surfaces, so there "
                "is nothing to choose between. Delete output_level, or "
                "track on field = 'pressure'.")
        # A SURFACE NAMED HERE AND NOT IN level_hpa IS REPORT-ONLY.
        #
        # The two keys answer different questions -- level_hpa is what
        # STEERS the nest, output_level is what the FILE carries -- and
        # tying the second to the first cost one or the other.  A nest
        # wants a curated handful (850/700/500 is the classic steering
        # set; averaging in a 200 hPa outflow centre would drag it off
        # the eyewall), while a forecaster reading vortex tilt wants the
        # whole profile.  So the extras are computed, given their own
        # centre search and their own columns, and kept OUT of the
        # steering mean (storm_tracking.centre_over_levels).
        #
        # Each costs one more plane and one more centre search per
        # consultation -- 2.3 ms on this box at 378x378x49, flat in the
        # number of surfaces -- so a twenty-surface profile is 0.15% of a
        # three-day run at the relocation cadence.  That is why there is
        # no cap on how many may be named.
        tracked = levels_of(relocation.follow)
        steer = {round(float(v), 6) for v in tracked}
        extra, seen = [], set()
        for value in relocation.track.output_level:
            key = round(float(value), 6)
            if key == round(float(SURFACE_LEVEL), 6) or key in steer:
                continue
            if key in seen:
                continue
            seen.add(key)
            extra.append(float(value))
        if extra:
            if position_only:
                raise ValueError(                      # unreachable today
                    "position-only trackers refuse output_level above")
            if not tracked:
                raise ValueError(
                    "[relocation.track] output_level of "
                    f"{source} names {', '.join(f'{v:g}' for v in extra)}, "
                    "but [relocation.follow] tracks SEA LEVEL "
                    f"(level_hpa = {SURFACE_LEVEL:g}). The surface block "
                    "and an isobaric profile are different reductions in "
                    "different units, and this run computes only the "
                    "first. Track on an isobaric surface -- level_hpa = "
                    "850 -- to report others beside it.")
            # The surfaces themselves are validated where they are
            # ATTACHED (_attach_report_levels, below): this function
            # refuses and does not build, and a value that has to survive
            # the call cannot be set on a local here.
            _attach_report_levels(relocation, extra, source)

    # VMAX is a 10-m wind, and a 10-m wind is a SURFACE-LAYER product.
    # woof.core.physics allocates every SFCLAY_OUTPUTS name
    # unconditionally and fills them only when a scheme runs, so under
    # sf_sfclay_physics = 0 u10/v10 exist and are identically zero.  A
    # track row would then carry VMAX = 0 kt, which is not a missing
    # value -- it is a plausible-looking number of the wrong quantity,
    # and a deck reader has no way to tell.  Refused on the SELECTOR, at
    # load, naming the domain and the knob.
    by_id = {int(dc.grid_id): dc for dc in domains}
    mover = by_id.get(int(relocation.grid_id))
    wind_sources = set()
    if not position_only:
        wind_sources = {int(mover.parent_id)} if mover is not None else set()
        refine = getattr(relocation.follow, "refine_grid_id", None)
        if refine is not None:
            wind_sources.add(int(refine))
    for gid in sorted(wind_sources):
        dc = by_id.get(gid)
        if dc is None:
            continue
        holder = getattr(dc, "run", None) or dc
        selector = getattr(holder, "sf_sfclay_physics", None)
        # None is "this object does not carry a resolved physics
        # selector", which is a different statement from "this domain
        # runs no surface layer" and must not be reported as one.  The
        # config loader always resolves it, so only a hand-built domain
        # reaches here with None.
        if selector is not None and int(selector) == 0:
            raise ValueError(
                f"[relocation.track] of {source} needs a 10-m wind for "
                f"the peak wind, and d{gid:02d} -- which the tracker can read the "
                f"vortex off -- runs sf_sfclay_physics = 0. No "
                "surface-layer scheme means u10/v10 are allocated and "
                "never filled, so the wind would be written as 0.00 m/s: "
                "not a "
                "missing value, a wrong one. Set sf_sfclay_physics on "
                f"d{gid:02d}, or delete the track table.")

    consult = relocation.cadence_seconds
    if consult is None:
        consult = float(root_dt)
    cfg = relocation.track
    interval = cfg.interval_seconds
    if interval is not None:
        if not whole_root_steps(interval):
            raise ValueError(
                f"[relocation.track] interval_seconds = {interval} of "
                f"{source} is not a whole number of root steps (root dt = "
                f"{float(root_dt)} s); the track writer emits at complete "
                "cycle boundaries, and this interval never lands on one.")
        if interval < float(consult) - _REL_TOL:
            # Only a field that is BOTH stash-backed and actually read by
            # the row.  Written as the difference rather than as a literal
            # list so the guard keeps its meaning if either set changes:
            # today it is empty, because the two stash-backed fields are
            # exactly the two whose row no longer reads a plane.
            field = tracked_field
            if field in set(STASH_BACKED_FIELDS) - set(POSITION_ONLY_FIELDS):
                raise ValueError(
                    f"[relocation.track] interval_seconds = {interval} "
                    f"of {source} is finer than the {float(consult)} s "
                    f"cadence the tracker is consulted on, and "
                    f"[relocation.follow] field = {field!r} is served from "
                    "a SCRATCH STASH rather than reduced on demand. "
                    "reflectivity exists only at history instants, and the "
                    "uh window is reset once per relocation cadence -- so "
                    "between consultations a track row would be computed "
                    "from a stale plane or a partial accumulation, and "
                    "would report a centre the nest was never steered by. "
                    "(It would not MOVE the nest: the window reset belongs "
                    "to the mover's consultation. This is about the record "
                    "being true, not the run being safe.) Either set "
                    "interval_seconds >= cadence_seconds, or track on "
                    "field = 'pressure', which is reduced from the live "
                    "prognostic column and is valid at every boundary.")


def _refuse_moving_slope_radiation(domains, relocation, source) -> None:
    """slope_rad on a nest that moves: refused until the move carries it.

    A relocation rebuilds the nest cold at its new placement and carries
    only the radiation carriers the land-surface consumer matrix names
    (woof.core.physics_continuation.relocatable_carriers).  The held
    slope-radiation state (the diffuse fraction, the radiation-time solar
    geometry and the shadow of woof.core.topo_radiation) is not among
    them, so after every move the land surface would take the flat flux
    until the next radiation call instead of the one WRF gives it.
    """
    moving = set()
    if relocation is not None and getattr(relocation, "enabled", False)             and (getattr(relocation, "moves", ())
                 or getattr(relocation, "follow", None) is not None):
        moving.add(int(relocation.grid_id))
    moving.update(int(dc.grid_id) for dc in domains
                  if getattr(dc, "follow", None) is not None)
    for dc in domains:
        if int(dc.grid_id) in moving and int(dc.run.slope_rad) == 1:
            raise ValueError(
                f"[[domain]] grid_id = {int(dc.grid_id)} of {source} moves "
                "and sets slope_rad = 1: a move rebuilds the nest and "
                "carries only the land-surface radiation carriers "
                "(woof.core.physics_continuation), not the held slope "
                "radiation state, so after every move its land surface "
                "would take the flat shortwave until the next radiation "
                "call.  Set slope_rad = 0 on the moving nest.")


def _refuse_rebuilt_nest_shortwave_interpolation(domains, relocation,
                                                 source) -> None:
    """RETIRED: swint_opt = 1 on a nest rebuilt mid-run is served.

    The defect this refused -- a spawned or moved nest's driver rebuilt cold
    with no shortwave-interpolation fit, so its surface shortwave read zero
    in daylight until its next radiation call a whole radt away -- is fixed
    at its cause: woof.runtime.rebuild_child_driver_from_land_state marks
    the rebuilt driver to call radiation on its first step
    (``force_radiation_next_step``), exactly as a nest born at t = 0 does,
    so the fit restarts from a real call.  Kept as a named no-op so the
    load sequence that called it reads the same.
    """
    del domains, relocation, source


def _refuse_rebuilt_nest_noah_mosaic(domains, relocation, source) -> None:
    """Noah mosaic on a nest whose physics is rebuilt mid-run: refused.

    A move and a spawn both rebuild the nest's physics driver from a land
    state (woof.runtime.rebuild_child_driver_from_land_state), and that
    rebuild does not build the land-use tiles: only the initialization
    doors do (woof/core/noah_mosaic_door.py).  WRF carries them through a
    move by re-deriving them with interp_mask_land_field on lu_index
    (Registry.EM_COMMON:1873-1913).  Accepted, the nest would run until the
    move or the spawn and then stop on its first Noah step with no tile
    state, hours into the run; refused here, before anything is prepared.
    """
    rebuilt = {}
    if relocation is not None and getattr(relocation, "enabled", False) \
            and (getattr(relocation, "moves", ())
                 or getattr(relocation, "follow", None) is not None):
        rebuilt[int(relocation.grid_id)] = "moves"
    for dc in domains:
        if getattr(dc, "follow", None) is not None:
            rebuilt[int(dc.grid_id)] = "moves"
        elif getattr(dc, "spawn", None) is not None:
            rebuilt.setdefault(int(dc.grid_id), "is spawned mid-run")
    for dc in domains:
        how = rebuilt.get(int(dc.grid_id))
        if how is not None and int(getattr(dc.run, "sf_surface_mosaic", 0)) == 1:
            raise ValueError(
                f"[[domain]] grid_id = {int(dc.grid_id)} of {source} {how} "
                "and runs Noah mosaic (sf_surface_mosaic = 1): a move or a "
                "spawn rebuilds the nest's physics from its land state and "
                "that rebuild does not build the land-use tiles (WRF "
                "re-derives them with interp_mask_land_field, "
                "Registry.EM_COMMON:1873-1913), so the nest would stop at "
                "its first Noah step after the event with no tile state.  "
                "Keep this nest still and present from the start, or set "
                "sf_surface_mosaic = 0.")


def _refuse_rebuilt_nest_lake(domains, relocation, source) -> None:
    """A rebuilt lake nest must not silently cold-start its heat storage."""
    rebuilt = set()
    if relocation is not None and getattr(relocation, "enabled", False) \
            and (getattr(relocation, "moves", ())
                 or getattr(relocation, "follow", None) is not None):
        rebuilt.add(int(relocation.grid_id))
    rebuilt.update(int(dc.grid_id) for dc in domains
                   if getattr(dc, "follow", None) is not None
                   or getattr(dc, "spawn", None) is not None)
    for dc in domains:
        if int(dc.grid_id) in rebuilt and int(getattr(dc.run, "sf_lake_physics", 0)) == 1:
            raise ValueError(
                f"[[domain]] grid_id = {int(dc.grid_id)} of {source} moves "
                "or is spawned mid-run with sf_lake_physics = 1: the nest "
                "rebuild does not carry lake water, ice, snow and sediment "
                "heat storage. Keep this lake domain fixed and present "
                "from the start; otherwise its lake state would be reset.")


def _refuse_child_terrain_drag(domains, source) -> None:
    """topo_wind / gwd_opt on a child domain: refused until its statics route
    builds the orographic statistics.

    The root's preparation adds VAR_SSO, CON, VAR, OA1-4, OL1-4 and the GSL
    sets to its static build (woof.static.orographic); a child's statics
    come from the nest initializers (woof.ingest.nest_init and its
    siblings), which do not ask for them yet, so the child's physics would
    find no statistics to drag with and the run would stop at the child's
    birth instead of before the first step.
    """
    for dc in domains:
        if getattr(dc, "parent_id", None) in (None, 0)                 or int(dc.grid_id) == 1:
            continue
        for name in ("topo_wind", "gwd_opt"):
            value = int(getattr(dc.run, name, 0) or 0)
            if value:
                raise ValueError(
                    f"[[domain]] grid_id = {int(dc.grid_id)} of {source} "
                    f"sets {name} = {value}: a child's statics are built by "
                    "the nest initializer, which does not add the sub-grid "
                    "orographic statistics the terrain drag reads, so the "
                    "child would stop at its birth.  Set it on the root "
                    "(grid_id = 1), or run the child as its own root.")


def _refuse_windowed_stash_watch(domains, source) -> None:
    """A spawn or retire watch cannot read a parent with a history window.

    The reflectivity and UH planes a lifecycle trigger reads are stashed
    at the watched parent's HISTORY times (woof.core.refl); a parent that
    sets history_begin_s / history_end_s has no stash outside that window,
    so the watch would read a plane that was never written and the run
    would refuse mid-flight.
    """
    from woof.core.storm_tracking import STASH_BACKED_FIELDS

    by_id = {int(dc.grid_id): dc for dc in domains}
    for dc in domains:
        for name in ("spawn", "retire"):
            policy = getattr(dc, name, None)
            trigger = getattr(policy, "trigger", None)
            if trigger not in STASH_BACKED_FIELDS:
                continue
            parent = by_id.get(int(dc.parent_id))
            if parent is None:
                continue
            if (float(getattr(parent, "history_begin_s", 0.0) or 0.0),
                    getattr(parent, "history_end_s", None)) == (0.0, None):
                continue
            raise ValueError(
                f"[[domain]] grid_id = {int(dc.grid_id)} of {source} "
                f"watches {trigger} on its parent grid_id = "
                f"{int(parent.grid_id)} for its {name}, and the parent sets "
                "history_begin_s / history_end_s: the watched plane is "
                "stashed only at the parent's history times, so outside "
                "that window the watch has nothing to read and the run "
                "refuses mid-flight.  Remove the offsets from the parent.")


def _refuse_unservable_follow_cadence(relocation, domains, source,
                                      *, root_dt,
                                      table: str = "[relocation]",
                                      follow_table: str = "[relocation.follow]",
                                      ) -> None:
    """The reflectivity stash must be able to serve every evaluation.

    The tracker's composite-reflectivity plane is not a diagnostic it can
    ask for on demand: ``refl_10cm`` is stashed by the microphysics
    drivers inside their ``refl_10cm_due`` branch, which follows the
    HISTORY cadence.  So an evaluation cadence that is not a whole
    multiple of the watched domain's ``history_interval_s`` asks for a
    plane at instants where it does not exist, and the run discovers
    that mid-flight as a ``TrackerRefusal`` -- at the first cadence where
    UH is under threshold and the echo fallback is consulted, which may
    be hours in and is exactly the moment a storm-following nest is
    supposed to be working.

    This is issue #111, and the contract is not new: the shipped
    ``moving_nest_20110427_follow_2km.toml`` states it in a comment above
    ``cadence_seconds`` and nothing enforced it.  Refused here, at
    admission, naming both knobs and the multiple that would work.

    It applies to ``field = "uh"`` as much as to ``field =
    "reflectivity"``: the echo handoff is automatic, not opt-in, so a
    UH-primary tracker whose cadence the stash cannot serve is a run that
    refuses the first time rotation is absent.

    ``table`` and ``follow_table`` name the tables the knobs actually
    live in, because this refusal serves two shapes of configuration: a
    whole-run ``[relocation]`` and a per-domain ``[[domain]].follow``.
    Naming the wrong one sends the reader to a table their configuration
    does not have, and since the per-domain followers are refused at load
    this is the message most readers meet.

    The watched domain is the PARENT of ``grid_id``.  ``grid_id`` names
    the child that MOVES; ``RelocationRunner`` hands the provider
    ``node.parent.state``, so the stash whose cadence matters belongs to
    the parent.
    """

    from woof.core.storm_tracking import STASH_BACKED_FIELDS

    # A pressure tracker reduces its plane from the live prognostic
    # column (woof.core.storm_tracking.mslp_hpa_from_state), which is
    # valid at EVERY cycle boundary.  There is no stash to miss, so the
    # whole contract below is inapplicable -- and enforcing it anyway
    # would refuse a perfectly servable cadence and send the reader to a
    # knob that has nothing to do with it.  Gated on the field rather
    # than deleted: uh and reflectivity still depend on the stash, and
    # the uh echo handoff is automatic, so both keep the refusal.
    if getattr(relocation.follow, "field", None) not in STASH_BACKED_FIELDS:
        return
    by_id = {int(dc.grid_id): dc for dc in domains}
    try:
        child = by_id[int(relocation.grid_id)]
        parent = by_id[int(child.parent_id)]
        stash = float(parent.history_interval_s)
    except (KeyError, TypeError, ValueError, AttributeError):
        # A child with no resolvable parent, or a domain missing the
        # fields this reads.  Tree integrity and per-domain schema are
        # other validators' refusals; preempting them with a cadence
        # message would send the reader to the wrong knob.
        return
    if not math.isfinite(stash) or stash <= 0.0:
        return
    window = (float(getattr(parent, "history_begin_s", 0.0) or 0.0),
              getattr(parent, "history_end_s", None))
    if window != (0.0, None):
        raise ValueError(
            f"{table} of {source} configures a {follow_table} tracker "
            f"watching [[domain]] grid_id = {int(parent.grid_id)}, which "
            f"sets history_begin_s = {window[0]:g}"
            + ("" if window[1] is None
               else f" / history_end_s = {float(window[1]):g}")
            + ": the tracker's reflectivity signal is stashed only at that "
            "domain's history times (woof.core.refl), so outside the "
            "window it has no plane to read and the run refuses "
            "mid-flight.  Remove the begin/end offsets from the watched "
            "domain.")
    where = (f"history_interval_s = {stash} s on the domain the tracker "
             f"watches ([[domain]] grid_id = {int(parent.grid_id)}, the "
             f"parent of the relocating grid_id = "
             f"{int(relocation.grid_id)})")
    why = ("the tracker's composite-reflectivity signal is stashed by the "
           "microphysics at history cadence (woof.core.refl), so a "
           "consultation off that cadence asks for a refl_10cm plane that "
           "does not exist and the run refuses mid-flight")
    if relocation.cadence_seconds is None:
        raise ValueError(
            f"{table} of {source} configures a {follow_table} "
            f"tracker but no cadence_seconds, which means EVERY complete "
            f"cycle boundary (root dt = {float(root_dt)} s), and {where} "
            f"cannot serve that: {why}. Set cadence_seconds to a whole "
            f"multiple of {stash} (the history interval itself, {stash}, "
            f"is the usual choice).")
    cadence = float(relocation.cadence_seconds)
    multiples = cadence / stash
    if abs(multiples - round(multiples)) > _REL_TOL * max(
            1.0, abs(multiples)):
        lower = max(1, int(multiples)) * stash
        raise ValueError(
            f"cadence_seconds = {cadence} in {table} of {source} is "
            f"not a whole multiple of {where}: {why}. Use a whole multiple "
            f"of {stash} (nearest below/above: {lower} / "
            f"{lower + stash}), or set that domain's history_interval_s "
            f"to a value {cadence} divides into.")


def _require_keys(table_name: str, entries: dict, required, source: str):
    missing = [key for key in required if key not in entries]
    if missing:
        raise ValueError(
            f"[{table_name}] of {source} is missing required key(s) "
            f"{missing}; present: {sorted(entries)}.")


def did_you_mean(key: str, known) -> str:
    """`` (did you mean 'dampcoef'?)`` for a near miss, else ``""``.

    Public because every schema in this package that refuses an unknown
    key owes the reader the same second half of the sentence.  A second
    copy of this three-liner in :mod:`woof.case_data` would be a second
    cutoff to keep in step with this one.
    """

    close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.7)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _reject_axis_authored_keys(table_name: str, entries: dict,
                               resolution, source: str) -> None:
    """Refuse a key the physics-fidelity axis is already the author of.

    The axis is sugar over a resolved patch vector, and sugar that MERGES
    with a hand-written key is how a run ends up integrating a value nobody
    chose: two authors, one key, and a receipt that can name only one of
    them.  So the second author is refused by name, in either mode -- there
    is no "they happen to agree" exemption, because agreeing today is a
    property of the value, not of the configuration, and the ledger's
    faithful/patched edge can move under a new patch-set version while the
    hand-written value stays put.

    The remedy is composition, not a wider schema: a battery arm strips the
    governed keys and lets the ``[experiment]`` overlay write them
    (``woof.physics_mode.governed_keys`` is the exact list to strip).
    """

    if not resolution.governed:
        return
    authored = {patch.key: patch for patch in resolution.resolved
                if patch.key}
    present = sorted(set(entries) & set(authored))
    if not present:
        return
    named = ", ".join(
        f"{key!r} (divergence-ledger {authored[key].entry_id}, resolved to "
        f"{authored[key].value!r})" for key in present)
    raise ValueError(layered(
        f"[{table_name}] of {source} sets {named}, but physics_mode = "
        f"{resolution.mode!r} is already the author of "
        f"{'that key' if len(present) == 1 else 'those keys'}; a key with "
        "two authors runs a value neither of them can be shown to have "
        "chosen.",
        "Remove the key(s) and let the axis write them, or remove "
        "physics_mode and configure the keys directly. The axis governs "
        f"{sorted(authored)} under physics_mode = {resolution.mode!r}, "
        f"patchset = {resolution.patchset!r}; "
        "woof.physics_mode.governed_keys() is the same list for a "
        "composer that has to strip them."))


def _reject_unknown_keys(table_name: str, entries: dict, known,
                         source: str) -> None:
    """Refuse keys this schema does not know, naming each one.

    This used to warn and drop, on the reasoning that the required-key
    and value checks around it catch every misspelling that matters.
    They do not, and cannot: every key with a default is optional by
    construction, so ``damp_coef = 0.4`` passes every one of them and
    the run integrates with ``dampcoef`` at its built-in 0.2.  The
    fleet's hostile-input node caught exactly that -- one stderr line
    among a hundred, then three wrfout frames, 159 product images and
    ``forecast validity PASS`` at exit 0, computed from a coefficient
    the user never chose.

    Warn-not-block is for the run that will almost certainly work.  A
    dropped key is the other case: the answer is confident, complete,
    and not the answer that was asked for.  So it refuses, it names the
    key, and it names the key it thinks was meant.  The full known-key
    list is mechanism, so it sits behind ``--explain``.
    """
    unknown = sorted(set(entries) - set(known))
    if not unknown:
        return
    named = ", ".join(
        f"{key!r}{did_you_mean(key, known)}" for key in unknown)
    raise ValueError(layered(
        f"[{table_name}] of {source} does not have a key {named}; "
        "no key is ignored, because a dropped key runs a default "
        "under the name of your value.",
        f"Known [{table_name}] keys: {sorted(known)}."))


def _positive_int(table_name: str, key: str, value, source: str,
                  minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{key} in [{table_name}] of {source} must be an integer, "
            f"got {value!r}.")
    if value < minimum:
        raise ValueError(
            f"{key} in [{table_name}] of {source} must be >= {minimum}, "
            f"got {value}.")
    return value


def _mass_dims(dom: dict, grid_id, source: str) -> tuple[int, int]:
    """nx/ny mass dimensions from e_we/e_sn (staggered) or nx/ny.

    WRF's e_we/e_sn count STAGGERED u/v points; woof dimensions count
    mass points, one fewer (the frozen reference case case represents the
    251 x 201 namelist domain as 250 x 200).
    """
    dims = []
    for stag_key, mass_key in (("e_we", "nx"), ("e_sn", "ny")):
        stag = _DOMAIN_KEY_ROWS[stag_key].get(
            dom, where=f"[[domain]] grid_id={grid_id} of {source}")
        mass = dom.get(mass_key)
        if stag is None and mass is None:
            raise ValueError(
                f"[[domain]] grid_id={grid_id} of {source} must carry "
                f"{stag_key} (staggered) or {mass_key} (mass points).")
        if stag is not None:
            stag = _positive_int("domain", stag_key, stag, source, 2)
            if mass is not None and mass != stag - 1:
                raise ValueError(
                    f"[[domain]] grid_id={grid_id} of {source} carries "
                    f"inconsistent dimensions: {stag_key}={stag} "
                    f"(staggered) implies {mass_key}={stag - 1} mass "
                    f"points but {mass_key}={mass} was supplied.")
            dims.append(stag - 1)
        else:
            dims.append(_positive_int("domain", mass_key, mass, source, 1))
    return dims[0], dims[1]


def _reject_misplaced_run_keys(dom: dict, grid_id, source: str) -> None:
    """A real run setting written into [[domain]] refuses, never drops.

    ``_reject_unknown_keys`` warns and drops, and for a typo or a
    leftover that is right.  A key that IS a ``RunConfig`` field is
    neither: the user wrote a setting this model really has, in a table
    that does not carry it, and dropping it silently would run the
    OTHER value -- the one in ``[shared]`` -- while the file on disk
    says otherwise.  That is a confidently-delivered wrong answer, and
    the one class of configuration mistake this loader refuses rather
    than warns about.

    Measured case: ``bl_pbl_physics`` on a ``[[domain]]`` table used to
    warn and drop, so a tree that named an experimental PBL closure on
    one nest ran the shared scheme on every nest and reported success.
    Nothing in the run receipt would have contradicted the file.
    """
    known = {f.name for f in fields(RunConfig)}
    misplaced = sorted(key for key in dom
                       if key in known and key not in _DOMAIN_KEYS)
    if misplaced:
        raise ValueError(
            f"key(s) {misplaced} on [[domain]] grid_id={grid_id} of "
            f"{source} are run settings that are NOT per domain: they "
            f"belong in [shared], where they apply to every domain of "
            f"the tree.  Refused rather than ignored -- a physics or "
            f"run key that parsed here and was dropped would run the "
            f"[shared] value while this file said otherwise.  Per-domain "
            f"keys are {sorted(_DOMAIN_RUN_OVERRIDES)}.")


#: Named by every inline vertical refusal.  The refusals below guard the
#: INLINE nest corridor, which still shares one ladder across the tree; a
#: per-domain ladder IS shipped, on the offline route, and a refusal that
#: did not say so would describe the product as less capable than it is.
_OFFLINE_LADDER_DOOR = (
    "A per-domain vertical ladder IS available on the OFFLINE downscale "
    "route: `woof downscale --child-levels N,STRETCH` prepares the child "
    "once on the host through a conservative vertical remap "
    "(woof/vertical_remap.py, mass- and water-conserving) and runs it as a "
    "standalone specified=True domain.")


def _reject_domain_vertical_keys(dom: dict, grid_id, source: str) -> None:
    """ANY per-domain vertical key is rejected (F1 amendment, §A).

    The vertical grid is single-sourced from the one
    ``ExperimentConfig.vertical`` -- vertical nesting is rejected by
    construction (WRF only invokes vertical nest machinery when a nest
    refines the vertical grid: ``if (nest%e_vert /= parent%e_vert)``
    guards init_domain_vert_nesting, share/mediation_integrate.F:666).
    """
    present = [key for key in _DOMAIN_VERTICAL_KEYS if key in dom]
    if present:
        raise ValueError(
            f"vertical key(s) {present} on [[domain]] grid_id={grid_id} "
            f"of {source} are rejected: the vertical grid is "
            "single-sourced from [shared] (ExperimentConfig.vertical) -- "
            "vertical nesting is rejected by construction for an INLINE "
            "nest tree (WRF only calls init_domain_vert_nesting when a "
            "nest refines the vertical grid, "
            "share/mediation_integrate.F:666), because the inline "
            "corridor interpolates boundaries every parent step on the "
            "GPU and has no vertical operator there.  "
            + _OFFLINE_LADDER_DOOR)


def _bad_mix_isotropic_sentinel(value, where: str, source: str) -> str:
    """The refusal for a string that is not the ``"auto"`` sentinel."""

    return (
        f"mix_isotropic = {value!r} in {where} of {source} must be 0 "
        f"(anisotropic mixing lengths), 1 (isotropic (dx*dy*dz)^(1/3)) "
        f"or the string \"{MIX_ISOTROPIC_AUTO}\" -- the same meaning as "
        f"leaving the key unset: the model selects isotropic where "
        f"mix_upper_bound*(dz_max/dx)^2 exceeds "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT} and the WRF-default "
        f"anisotropic form otherwise.")


def _bad_epssm_sentinel(value, where: str, source: str) -> str:
    """The refusal for an ``epssm`` string that is not ``"auto"``.

    A string reaching RunConfig would fail its number check with no word
    about the sentinel the string was probably meant to be.
    """

    return (
        f"epssm = {value!r} in {where} of {source} must be a number in "
        f"(0, 1] or the string \"{EPSSM_AUTO}\" -- the same meaning as "
        f"leaving the key unset: WRF's default 0.1, raised to the measured "
        f"off-centering floor over ground 0.1 was not measured stable on "
        f"(woof/acoustic_adaptation.py).")


def _cross_check(kind: str, grid_id, supplied: float, derived: Fraction,
                 chain: str, source: str) -> None:
    """Hand-typed child dx/dt cross-check: the namelist chain is
    authoritative; a mismatch is a hard error (kills the 333.33 m vs
    "500 m" prose confusion -- e.g. a hand-typed 500 m d04 against
    ratio 3 from 1 km)."""
    derived_f = float(derived)
    if not math.isfinite(float(supplied)) or abs(
            float(supplied) - derived_f) > _REL_TOL * abs(derived_f):
        raise ValueError(
            f"[[domain]] grid_id={grid_id} of {source} hand-types "
            f"{kind}={supplied!r} but the parent chain derives {kind} = "
            f"{derived} ({chain}): child {kind} is never hand-typed -- "
            "the namelist chain is authoritative; remove the key or fix "
            "the chain.")


def _history_window(dom: dict, *, grid_id: int, source: str,
                    run_seconds: float) -> tuple[float, float | None]:
    """``[[domain]] history_begin_s`` / ``history_end_s``, validated.

    Whole seconds, as WRF states them (the D/H/M/S integers of
    history_begin_* / history_end_*).  A begin that is not on the domain's
    step lattice rings at the first step after it, as WRF's alarm does
    (ESMF_ClockAdvance rings once RingTime <= CurrTime); woof.core.clock
    rounds it the same way.
    """

    def seconds(key: str) -> float | None:
        if key not in dom:
            return None
        value = dom[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0
                or float(value) != int(float(value))):
            raise ValueError(
                f"{key} = {value!r} on [[domain]] grid_id = {grid_id} of "
                f"{source} must be a whole, non-negative number of seconds "
                "(WRF states it in whole days, hours, minutes and seconds).")
        return float(value)

    begin = seconds("history_begin_s") or 0.0
    end = seconds("history_end_s")
    if begin >= run_seconds:
        raise ValueError(
            f"history_begin_s = {begin:g} on [[domain]] grid_id = {grid_id} "
            f"of {source} is at or past the run's end ({run_seconds:g} s): "
            "the domain would write no history, and every product drawn "
            "from its frames would be missing.")
    if end is not None and end <= begin:
        raise ValueError(
            f"history_end_s = {end:g} on [[domain]] grid_id = {grid_id} of "
            f"{source} is not after history_begin_s = {begin:g}: the domain "
            "would write no history.")
    return begin, end


def _check_cadence(label: str, seconds: Fraction, dt: Fraction, grid_id,
                   source: str) -> None:
    if seconds <= 0:
        return  # 0 = every step (WRF convention for radt/cudt/bldt)
    steps = seconds / dt
    if steps.denominator != 1:
        raise ValueError(
            f"{label} = {float(seconds):g} s on domain grid_id={grid_id} "
            f"of {source} is not a whole number of that domain's steps: "
            f"dt = {dt} s exactly, {seconds}/({dt}) = {steps}.")


def wrf_physics_cadences(run, dt: Fraction) -> dict[str, dict[str, object]]:
    """The physics cadences of one fixed-step domain that wrf.exe rounds.

    WRF takes ``radt``, ``cudt`` and ``bldt`` in minutes and runs each
    scheme on whole steps: ``STEPRA/STEPCU/STEPBL = MAX(NINT(minutes*60/DT),
    1)`` (phys/module_physics_init.F).  woof's clock does the same
    (:func:`woof.core.clock.resolve_clock`); this names the cadences
    that are not already a whole number of steps, each with its whole
    step count and the minutes it actually runs at, for the loader's
    warning and the run receipt.  Empty when every cadence divides, so a
    receipt is unchanged for every configuration that already ran.

    Before 2.9.10 the loader refused such a cadence ("300/(45) = 20/3")
    where wrf.exe runs it: a WRF user's ``cudt = 5`` at a 45 s step
    (7 steps, 5.25 min) could not be written as it is in WRF.
    """

    rows: dict[str, dict[str, object]] = {}
    radt = run.radt if run.radt > 0.0 else run.radt_minutes
    for label, minutes, read in (
            ("radt", radt, radiation_enabled(run)),
            ("cudt_minutes", run.cudt_minutes, run.cu_physics != 0),
            ("bldt", run.bldt, (run.bl_pbl_physics != 0
                                or run.sf_sfclay_physics != 0
                                or run.sf_surface_physics != 0))):
        if not read or minutes <= 0.0:
            continue
        exact = Fraction(str(minutes)) * 60 / dt
        if exact.denominator == 1:
            continue
        # Fortran NINT rounds half away from zero; every value is positive.
        steps = max(int(exact + Fraction(1, 2)), 1)
        rows[label] = {
            "requested_minutes": float(minutes),
            "step_seconds": float(dt),
            "steps": steps,
            "effective_minutes": float(Fraction(steps) * dt / 60),
        }
    return rows


def _warn_wrf_physics_cadences(run, dt: Fraction, grid_id, source: str) -> None:
    """Say, once, which cadences run as wrf.exe rounds them."""

    for label, row in wrf_physics_cadences(run, dt).items():
        warn(f"{label} = {row['requested_minutes']:g} min on domain "
             f"grid_id={grid_id} of {source} is "
             f"{Fraction(str(row['requested_minutes'])) * 60 / dt} steps of "
             f"{dt} s; it runs every {row['steps']} steps "
             f"({row['effective_minutes']:g} min), as wrf.exe runs it",
             why="phys/module_physics_init.F: STEPRA/STEPCU/STEPBL = "
                 "MAX(NINT(minutes*60/DT), 1); the scheme fires on whole "
                 "steps. The run receipt records the effective minutes "
                 "(experiment.domains[].physics_cadences).",
             once=True)


def _check_run_length_on_step_grid(run_seconds: Fraction, dt: Fraction,
                                   grid_id, source: str) -> None:
    """Refuse a forecast length that is not a whole number of d01 steps.

    ``woof.core.clock.resolve_clock`` has always enforced this -- runs
    end on a root-domain boundary -- but it enforces it at clock
    resolution, which every route reaches only after preparation.  So a
    half-step ``run_seconds`` passed ``woof check`` and then died in an
    unhandled ``ValueError`` at forecast, exiting 1 through a traceback,
    while the identical arithmetic error on ``history_interval_s`` or
    ``restart_interval_s`` was refused here in one line at admission,
    exit 2, before anything was fetched or prepared.  Same error class,
    same config file, two different user experiences.

    This admits nothing new and refuses nothing that ran: it is the
    check ``resolve_clock`` already applies, moved to where its siblings
    live so the refusal arrives before the work rather than after it.
    """
    if run_seconds <= 0:
        return
    steps = run_seconds / dt
    if steps.denominator != 1:
        raise ValueError(
            f"run_seconds = {float(run_seconds):g} s in [experiment] of "
            f"{source} is not a whole number of root-domain "
            f"(grid_id={grid_id}) steps: dt = {dt} s exactly, "
            f"{run_seconds}/({dt}) = {steps}.  Runs end on a d01 "
            "boundary, so the forecast length must land on the step "
            "grid.")


def _check_whole_second_cadence(label: str, seconds: Fraction, grid_id,
                                source: str) -> None:
    """Refuse a publication cadence the file names cannot express.

    History and standalone-restart instants become file names and a
    ``Times`` record at whole-second resolution, and both publishers
    replace.  A quarter-second history interval is divisible by a
    quarter-second step, so it used to pass admission -- and then three
    distinct legal frames formatted to one name, each replacing the last,
    with no exception raised anywhere.  Whole seconds are the widest
    cadence the on-disk contract can name, so they are the widest cadence
    admitted; the alternative is a run that quietly loses its own output.
    """
    if seconds <= 0:
        return
    if seconds.denominator != 1:
        raise ValueError(
            f"{label} = {float(seconds):g} s on domain grid_id={grid_id} "
            f"of {source} is not a whole number of seconds. History and "
            "restart files are named, and their Times record written, to "
            "whole-second resolution, so distinct sub-second instants "
            "would alias onto one file name and the later one would "
            "replace the earlier.")


def validate_boundary_timing(
        exp: ExperimentConfig, boundary_interval_seconds: int, *,
        source: str = "boundary forcing",
        live_born_children=()) -> None:
    """Validate the structural hierarchy/forcing timing contract.

    There is no whole-hour requirement.  The decoded boundary cadence must
    be a positive integer number of seconds and an exact number of root
    steps, because the root Davies clock resets only at top-of-step interval
    seams.  A delayed child start must additionally be an exact parent-step
    boundary and an exact boundary-forcing seam.  History output cadence is
    independent and is deliberately absent from this contract.

    ``live_born_children`` names the grid ids of children that are
    initialized from their LIVE parent at their start instant rather
    than from a forcing snapshot: the seam rule exists because a
    declared late start reads the forcing at that instant and so has to
    land on one, and a child that reads no snapshot has nothing to land
    on.  A spawned nest is the built-in case; a nest a cycling analysis
    attaches to its own trajectory declares itself here.  The
    parent-step alignment is not relaxed for either.
    """
    if (isinstance(boundary_interval_seconds, bool)
            or not isinstance(boundary_interval_seconds, int)
            or boundary_interval_seconds <= 0):
        raise ValueError(
            "boundary_interval_seconds must be a positive integer number "
            f"of seconds for {source}, got {boundary_interval_seconds!r}.")
    interval = Fraction(boundary_interval_seconds)
    root_steps = interval / exp.dt_exact(exp.root.grid_id)
    if root_steps.denominator != 1:
        raise ValueError(
            f"boundary_interval_seconds = {boundary_interval_seconds} s "
            f"for {source} is not a whole number of root-domain steps: "
            f"d{exp.root.grid_id:02d} dt = "
            f"{exp.dt_exact(exp.root.grid_id)} s exactly, cadence/dt = "
            f"{root_steps}.")
    live_born = {int(grid_id) for grid_id in live_born_children}
    for dc in exp.domains[1:]:
        offset = exp.domain_start_offset_exact(dc.grid_id)
        parent_offset = exp.domain_start_offset_exact(dc.parent_id)
        parent_dt = exp.dt_exact(dc.parent_id)
        parent_steps = (offset - parent_offset) / parent_dt
        if parent_steps.denominator != 1:
            raise ValueError(
                f"delayed start_time for d{dc.grid_id:02d} "
                f"({exp.domain_start_time(dc.grid_id).isoformat()}; offset "
                f"{float(offset):g} s) is not aligned to its parent step "
                f"boundary: d{dc.parent_id:02d} dt = {parent_dt} s "
                f"exactly, (child-parent start offset)/dt = "
                f"{parent_steps}.")
        if getattr(dc, "spawn", None) is not None:
            # A SPAWNED nest's epoch is its trigger's instant, and a
            # trigger fires on the weather rather than on the forcing
            # calendar.  The seam rule below exists because a domain
            # whose late start is DECLARED is initialized from the
            # forcing at that instant and so must land on a snapshot; a
            # spawned nest reads no snapshot at its birth -- it is SINT
            # from its LIVE parent (that is the whole point of spawning
            # it inside the storm the trigger saw), and its lateral
            # boundaries come from that parent afterwards.  Requiring a
            # forcing seam here would confine every spawn to the LBC
            # cadence, which for a six-hourly analysis is four instants a
            # day.  The parent-step alignment above is NOT relaxed: it is
            # the rule the spawn instant is already validated against.
            continue
        if int(dc.grid_id) in live_born:
            # The same argument, declared by the caller: this child is
            # SINT from its live parent at its start and reads no
            # forcing snapshot there, so a forcing seam has nothing to
            # bind.  The parent-step alignment above still applies.
            continue
        forcing_seams = offset / interval
        if forcing_seams.denominator != 1:
            raise ValueError(
                f"delayed start_time for d{dc.grid_id:02d} "
                f"({exp.domain_start_time(dc.grid_id).isoformat()}; offset "
                f"{float(offset):g} s) is not aligned to the "
                f"boundary-forcing cadence: boundary_interval_seconds = "
                f"{boundary_interval_seconds} s, offset/cadence = "
                f"{forcing_seams}.")


def _parent_before_child(domain_tables: list, source: str) -> list:
    """Reorder [[domain]] tables so parents precede their children.

    Declaration order carries no information the tree does not: when
    every table has integer grid_id/parent_id, a wave sort (roots, then
    children of placed domains, preserving declaration order inside a
    wave) recovers the only admissible order.  Anything unresolvable --
    a parent id naming no table, a cycle -- is left as declared for the
    loop's own checks to refuse by name.
    """

    try:
        ids = [int(dom["grid_id"]) for dom in domain_tables]
        parents = [int(dom["parent_id"]) for dom in domain_tables]
    except (KeyError, TypeError, ValueError):
        return domain_tables
    if len(set(ids)) != len(ids):
        return domain_tables
    # Already parent-before-child?  Then the declared order stands --
    # it is a valid order the author chose, and downstream receipts
    # key on domain sequence.
    seen: set[int] = set()
    ordered = True
    for index in range(len(domain_tables)):
        if parents[index] != 0 and parents[index] not in seen:
            ordered = False
            break
        seen.add(ids[index])
    if ordered:
        return domain_tables
    # Repair: emit the earliest declarable table each pass, preserving
    # declaration order among the tables that are ready.
    placed: set[int] = set()
    remaining = list(range(len(domain_tables)))
    order: list[int] = []
    while remaining:
        wave = [i for i in remaining
                if parents[i] == 0 or parents[i] in placed]
        if not wave:
            return domain_tables  # unresolvable; let the loop refuse
        order.extend(wave)
        placed.update(ids[i] for i in wave)
        remaining = [i for i in remaining if i not in wave]
    warn(f"[[domain]] tables in {source} were declared out of "
         "parent-before-child order; reordered by grid_id/parent_id")
    return [domain_tables[i] for i in order]


def build_experiment(raw: dict, source: str) -> ExperimentConfig:
    """Validate a parsed experiment TOML dict and build the config."""
    known_tables = ("experiment", "shared", "projection", "domain",
                    "relocation", "perturbation", "tiles", "devices", "output", "simulated_radar",
                    "spectral_numerics", "nudging", "spectral_nudging", "physics_params", "radar_heating")
    # Setup lineage is provenance, with no atmosphere or follower controls.
    if isinstance(raw, dict) and "lineage" in raw:
        from woof.cyclone_tracks import validate_lineage
        raw = dict(raw)
        validate_lineage(raw.pop("lineage"))
    # [ingest] is INGEST POLICY, and it is validated-and-dropped HERE
    # rather than added to the companion list above.  The companion
    # tables declare INPUTS: dropping one loses a setting, so the caller
    # must consume it and reaching this seam with one present is a
    # routing defect worth refusing.  [ingest] declares neither an input
    # nor anything this builder consumes -- its value is read from the
    # config by whoever ingests (woof.ingest.soil_downscale.
    # declared_soil_texture_downscale, and woof.case_data's loader,
    # which puts it on CaseDataConfig) -- so requiring every one of the
    # dozen callers of this function to split it would only mean the
    # switch is unreachable from whichever door someone forgot.  It is
    # still VALIDATED on the way past: a typo refuses here, it does not
    # silently run the default under the name of your setting.
    if isinstance(raw, dict) and "ingest" in raw:
        from woof.ingest.soil_downscale import parse_ingest_table
        raw = dict(raw)
        parse_ingest_table(raw.pop("ingest"), source=source)
    # Missing random-field/parameter consumers refuse before preparation:
    # otherwise their requested increments would silently stay absent.
    from woof.ensemble_admission import refuse_configured_random
    refuse_configured_random(raw)
    if isinstance(raw, dict) and "ensemble" in raw:
        from woof import ensemble
        raw = dict(raw)
        table = raw.pop("ensemble")
        # The same boundary as woof.config.load_config: a preparation-only
        # install stages no ensemble request, and the table is the ensemble
        # door's to validate.
        if ensemble.request_installed():
            from woof.ensemble.request import EnsembleRequest
            EnsembleRequest.from_mapping(table)
    # Companion tables of the ONE-FILE case schema: real, documented
    # tables that belong to other owners (woof.case_data, woof.fetch,
    # woof.static.highres_production) and are split off by every file
    # loader before this builder runs.  Reaching this dict-level seam
    # with one still present is a routing defect in the CALLER, and it
    # must never be reported as the table being unknown: task #204 was
    # exactly that -- the ERA5 adapter told a user their wizard-written
    # config "does not have a table 'case_data'" while the table sat in
    # the file, present and valid.
    companion_tables = ("case_data", "fetch", "static")
    present_companions = [name for name in raw if name in companion_tables]
    if present_companions:
        named = ", ".join(f"[{name}]" for name in present_companions)
        raise ValueError(layered(
            f"experiment config {source} carries {named}, which is part "
            "of the one-file case schema but was handed to the "
            "experiment-table builder unsplit.  The table is PRESENT and "
            "its name is valid; this loading path simply does not consume "
            "it, which is a defect in the calling code, not in the "
            "config.",
            "Every file loader splits the companion tables off first: "
            "woof.experiment.load_experiment validates and detaches "
            "[case_data]/[fetch]/[static], and "
            "woof.case_data.load_experiment_case does the same while "
            "also returning the case declarations.  A caller building "
            "from a raw dict must do the same before build_experiment."))
    unknown_tables = [name for name in raw if name not in known_tables]
    if unknown_tables:
        # A whole stray table is the same defect as a stray key, one
        # order of magnitude larger: `[dynamics]` is the LEGACY config's
        # table name, so pasting a familiar block into an experiment
        # config used to drop every setting in it behind one line.
        named = ", ".join(
            f"{name!r}{did_you_mean(name, known_tables)}"
            for name in unknown_tables)
        raise ValueError(layered(
            f"experiment config {source} does not have a table {named}; "
            "no table is ignored, because a dropped table runs defaults "
            "under the name of your settings.",
            f"Known tables: {list(known_tables)}. The [grid]/[dynamics]/"
            "[run] table names belong to the single-domain legacy config "
            "schema; an experiment config carries [experiment], [shared], "
            "[projection], one [[domain]] per nest, and optionally "
            "[perturbation] (initial-state theta bubbles)."))
    if "experiment" not in raw:
        raise ValueError(
            f"experiment config {source} must carry an [experiment] "
            "table.")
    domain_tables = raw.get("domain", [])
    if not isinstance(domain_tables, list) or not domain_tables:
        raise ValueError(
            f"experiment config {source} must carry at least one "
            "[[domain]] table (array of tables).")

    # ---- [experiment] ------------------------------------------------
    exp = raw["experiment"]
    _reject_moving_nest_keys("experiment", exp, source)
    _reject_unknown_keys("experiment", exp, _EXPERIMENT_KEYS, source)
    _require_keys("experiment", exp, _EXPERIMENT_REQUIRED, source)
    name = exp["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"name in [experiment] of {source} must be a non-empty "
            f"string, got {name!r}.")
    start_time = exp["start_time"]
    if not isinstance(start_time, datetime) or start_time.tzinfo is not None:
        raise ValueError(
            f"start_time in [experiment] of {source} must be an "
            "offset-free TOML datetime (e.g. 1974-04-03T12:00:00), got "
            f"{start_time!r}.")
    run_seconds = float(exp["run_seconds"])
    if not math.isfinite(run_seconds) or run_seconds <= 0.0:
        raise ValueError(
            f"run_seconds in [experiment] of {source} must be a finite "
            f"positive duration in seconds, got {exp['run_seconds']!r}.")
    feedback = exp.get("feedback", 0)
    if feedback not in FEEDBACK_OPTIONS:
        raise ValueError(
            f"feedback = {feedback!r} in [experiment] of {source} is "
            "rejected: feedback must be 0 (one-way) or 1 "
            "(experimental two-way child-to-parent restriction).")
    smooth_option = exp.get("smooth_option", 0)
    if smooth_option not in SMOOTH_OPTION_OPTIONS:
        raise ValueError(
            f"smooth_option = {smooth_option!r} in [experiment] of "
            f"{source} is rejected: WRF's post-feedback parent smoother "
            "options are 0 (none), 1 (sm121) or 2 (smdsm, the WRF "
            "Registry default); it applies only when feedback = 1.")
    blend_width = _positive_int("experiment", "blend_width",
                                exp.get("blend_width", 5), source, 0)
    spec_bdy_width = _positive_int("experiment", "spec_bdy_width",
                                   exp.get("spec_bdy_width", 5), source, 1)
    smooth_cg_topo = exp.get("smooth_cg_topo", False)
    if not isinstance(smooth_cg_topo, bool):
        raise ValueError(
            f"smooth_cg_topo = {smooth_cg_topo!r} in [experiment] of {source} "
            "must be true or false (WRF's &domains logical).")
    restart_interval_s = float(exp["restart_interval_s"])
    if not math.isfinite(restart_interval_s) or restart_interval_s < 0.0:
        raise ValueError(
            f"restart_interval_s in [experiment] of {source} must be a "
            "finite non-negative interval in seconds (0 disables restart "
            f"writing), got {exp['restart_interval_s']!r}.")
    column_chunk = _positive_int(
        "experiment", "column_chunk",
        exp.get("column_chunk", DEFAULT_COLUMN_CHUNK), source)
    raw_constant_glw = exp.get("constant_glw_wm2")
    if raw_constant_glw is not None:
        if (isinstance(raw_constant_glw, bool)
                or not isinstance(raw_constant_glw, (int, float))):
            raise ValueError(
                f"constant_glw_wm2 in [experiment] of {source} must be a "
                "downward longwave flux in W m-2 (a number), got "
                f"{raw_constant_glw!r}.")
        raw_constant_glw = float(raw_constant_glw)
    raw_acknowledgements = exp.get("acknowledgements", [])
    if not isinstance(raw_acknowledgements, list):
        raise ValueError(
            f"acknowledgements in [experiment] of {source} must be an array "
            f"of non-empty ids, got {raw_acknowledgements!r}.")
    acknowledgements: list[str] = []
    for index, value in enumerate(raw_acknowledgements):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"acknowledgements[{index}] in [experiment] of {source} "
                f"must be a non-empty string, got {value!r}.")
        acknowledgements.append(value)

    # ---- the physics-fidelity axis (woof/physics_mode.py) -------------
    # ABSENT is not the same state as present-and-"wrf-faithful".  Absent
    # authors nothing, which is what every configuration written before the
    # axis existed means and what keeps them byte-identical; present makes
    # the axis the author of every ledger key, in EITHER mode, which is what
    # lets one case file serve every battery arm through a two-line overlay.
    if "physics_mode" in exp or "patchset" in exp or "patches" in exp:
        if "physics_mode" not in exp:
            named = sorted(set(exp) & {"patchset", "patches"})
            raise ValueError(
                f"[experiment] of {source} sets {named} without "
                "physics_mode; a qualifier on the fidelity axis does not "
                "select it, and implying the patched mode from one would "
                "make the most consequential line of a config the one that "
                "is not written. Add physics_mode = "
                f"{physics_mode_module.PHYSICS_MODE_WRF_FAITHFUL!r} or "
                f"{physics_mode_module.PHYSICS_MODE_ARWEN_PATCHED!r}.")
        where = f"[experiment] of {source}"
        physics_mode = physics_mode_module.resolve(
            *(_EXPERIMENT_KEY_ROWS[key].get(exp, where=where)
              for key in ("physics_mode", "patchset", "patches")),
            source=source)
    else:
        physics_mode = physics_mode_module.UNGOVERNED

    # ---- [projection] ------------------------------------------------
    projection = None
    if "projection" in raw:
        proj = raw["projection"]
        _reject_unknown_keys("projection", proj, _PROJECTION_KEYS, source)
        _require_keys("projection", proj, _PROJECTION_KEYS, source)
        try:
            projection = ProjectionConfig(
                map_proj=str(proj["map_proj"]).lower(),
                ref_lat=float(proj["ref_lat"]),
                ref_lon=float(proj["ref_lon"]),
                truelat1=float(proj["truelat1"]),
                truelat2=float(proj["truelat2"]),
                stand_lon=float(proj["stand_lon"]))
        except ValueError as err:
            raise ValueError(f"[projection] of {source}: {err}") from None

    # ---- [perturbation] ----------------------------------------------
    # ABSENT authors nothing: perturbation = None, byte-identical prepared
    # state, and the restart-identity payload omits the key so every
    # pre-feature fingerprint is preserved (woof/core/model.py).
    perturbation = None
    if "perturbation" in raw:
        perturbation = _build_perturbation(raw["perturbation"], source)

    # ---- [spectral_numerics] -----------------------------------------
    # ABSENT authors nothing (spectral_numerics = None; the restart
    # identity omits the key so pre-feature fingerprints are preserved).
    # PRESENT, the block is validated by its owner
    # (woof.spectral_ops.config.from_mapping, fail-closed on unknown
    # keys) and then either honored at the one wired seam -- the
    # execute_experiment slow-large-step commit point -- or refused by
    # routes that do not thread it (refuse_unrouted_spectral_numerics),
    # the [perturbation] governance applied unchanged.
    spectral_numerics = None
    if "spectral_numerics" in raw:
        try:
            from woof.spectral_ops.config import from_mapping as \
                _spectral_from_mapping
        except ImportError:
            # The standalone preprocessing wheel stages none of the
            # spectral-numerics runtime (tools/build_rw_wps_release.py
            # excludes it by name); a config that carries the table on
            # such an install is refused with the remedy rather than a
            # bare ImportError.  A config without the table never takes
            # this branch, so preparation-only installs parse everything
            # they can act on.
            raise ValueError(
                f"[spectral_numerics] of {source}: this installation "
                "carries no spectral-numerics runtime (the standalone "
                "preprocessing distribution excludes it); run this "
                "configuration with the full woof distribution, or "
                "remove the table") from None
        try:
            spectral_numerics = _spectral_from_mapping(
                raw["spectral_numerics"])
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"[spectral_numerics] of {source}: {err}") from None

    devices = DeviceOptions.from_mapping(raw.get("devices"), source=source)

    # ---- [physics_params] ----------------------------------------------
    # ABSENT authors nothing: physics_params = None, every kernel source and
    # parameter table is today's, and the restart identity omits the key so
    # pre-feature fingerprints are preserved.  PRESENT, the set is validated
    # against woof/physics_params_registry_v1.json (unknown constants and
    # out-of-range values refuse) and bound to this process below, once the
    # domains are known.
    physics_params = None
    if "physics_params" in raw:
        from woof.physics_params import PhysicsParamsError, parse_table
        base_dir = Path(source).parent if Path(source).is_file() else None
        try:
            physics_params = parse_table(
                raw["physics_params"], source=source, base_dir=base_dir)
        except PhysicsParamsError as err:
            raise ValueError(str(err)) from None

    # ---- [tiles] ---------------------------------------------------
    # ABSENT is the OFF contract, and it is the shared StreamingOptions.OFF
    # object rather than a fresh one: the mode is an EXECUTION choice, it
    # authors no computed value, and
    # woof.core.streaming.identity_payload_entry deliberately contributes
    # nothing to the restart identity -- a checkpoint written resident must
    # resume streamed and one written streamed must resume resident, which
    # is the operation the mode exists for.  Proven in four legs across a
    # file, bit-exact in all of them.
    tiles = streaming_module.StreamingOptions.from_mapping(
        raw.get("tiles"), source=source)

    # ---- [output] ---------------------------------------------------
    # ABSENT is the FULL contract, and it is the shared FULL object
    # rather than a fresh one: writing every variable the run produces is
    # what every run before this surface existed did, byte for byte, and
    # a trimmed tape is a thing a user asks for.  Like [tiles] it
    # contributes nothing to the restart identity -- see the `output`
    # field's docstring on ExperimentConfig.
    output = history_selection_module.HistorySelection.from_mapping(
        raw.get("output"), source=source)
    simulated_radar = simulated_radar_module.SimulatedRadarOptions.from_mapping(
        raw.get("simulated_radar"), source=source)
    # ---- [radar_heating] ---------------------------------------------
    # Absent is OFF and changes no document, fingerprint or argv.  Present,
    # it is honored by the deterministic prepared door
    # (woof.da.forecast_heating) and refused by every route that does not
    # attach it (forecast_heating.require_routed, asked by
    # execute_experiment), never ignored.  A relative windows root is
    # anchored at the config file's own directory.
    source_path = Path(str(source))
    radar_heating = RadarHeatingConfig.from_mapping(
        raw.get("radar_heating"), source=source,
        base_dir=source_path.parent if source_path.is_file() else None)

    # ---- [shared] ------------------------------------------------------
    shared = dict(raw.get("shared", {}))
    # Captured from the RAW table, before anything defaults it: whether
    # the author CHOSE radiation at all.  ra_physics defaults to 0 and
    # ra_lw/sw_physics to -1, so by RunConfig time "wrote ra_physics = 0"
    # and "wrote nothing" are the same object -- and the radiation-off
    # land-surface refusal has to tell those two readers apart.  Same
    # idiom, and the same reason, as ``declared_map_proj`` below.
    from woof.physics_compat import declared_radiation_selectors
    declared_radiation = declared_radiation_selectors(shared)
    _reject_moving_nest_keys("shared", shared, source)
    run_field_names = {f.name for f in fields(RunConfig)}
    for key, where in _SHARED_FORBIDDEN.items():
        if key in shared:
            raise ValueError(
                f"key {key!r} in [shared] of {source} is not a shared "
                f"key; it belongs in {where}.")
    shared_known = ((run_field_names - set(_SHARED_FORBIDDEN))
                    | {"e_vert", "eta_levels", "p_top"}
                    | set(_GUARD_DEFAULTS))
    _reject_unknown_keys("shared", shared, shared_known, source)
    _reject_axis_authored_keys("shared", shared, physics_mode, source)

    # mix_isotropic's "auto" sentinel (ArWen's 2026-08-16 auto-switch
    # ruling): the string means the same as leaving the key unset -- the
    # model chooses -- so it is stripped HERE and its absence is what the
    # domain loop below reads.  Any other string is refused by name;
    # integers flow through to validate_run_config's 0/1 battery
    # untouched, so every explicit config keeps its exact meaning.
    if isinstance(shared.get("mix_isotropic"), str):
        if shared["mix_isotropic"] != MIX_ISOTROPIC_AUTO:
            raise ValueError(_bad_mix_isotropic_sentinel(
                shared["mix_isotropic"], "[shared]", source))
        del shared["mix_isotropic"]
    # epssm's "auto" sentinel, on the same rule: the model chooses, so it
    # is stripped here and the domain loop reads its absence.
    if isinstance(shared.get("epssm"), str):
        if shared["epssm"] != EPSSM_AUTO:
            raise ValueError(_bad_epssm_sentinel(
                shared["epssm"], "[shared]", source))
        del shared["epssm"]

    # Compared with the one implemented value, not checked against the
    # row's type.  Every other value is refused just below with the reason
    # it cannot run, and a spelling of the implemented value
    # (input_from_hires = 0, interp_method_type = 2.0) runs exactly what
    # the default runs, so a type refusal of it would name no breakage.
    # The rows still carry each key's type, default and meaning for a
    # front end, and the default is read from them (_GUARD_DEFAULTS).
    for key, default in _GUARD_DEFAULTS.items():
        value = shared.pop(key, default)
        if value == default:
            continue
        if key == "interp_method_type":
            raise ValueError(
                f"interp_method_type = {value!r} in [shared] of {source} "
                "is rejected: only SINT (interp_method_type = 2, the WRF "
                "default per Registry.EM_COMMON:2301) is implemented; "
                "bilinear (1), nearest-neighbour (3) and quadratic (4) "
                "are not.")
        if key == "nest_interp_coord":
            raise ValueError(
                f"nest_interp_coord = {value!r} in [shared] of {source} "
                "is rejected: isobaric nest re-interpolation "
                "(nest_interp_coord = 1) is not implemented; only the "
                "standard eta-level interpolation (0).")
        if key == "vert_refine_method":
            raise ValueError(
                f"vert_refine_method = {value!r} in [shared] of {source} "
                "is rejected: vertical nest refinement is not "
                "implemented -- e_vert/eta_levels/p_top must be identical "
                "across domains (WRF only calls init_domain_vert_nesting "
                "when refining, share/mediation_integrate.F:666).")
        raise ValueError(
            f"input_from_hires = {value!r} in [shared] of {source} is "
            "rejected: high-resolution child terrain input is not "
            "implemented (children SINT + blend the parent terrain, "
            "dyn_em/nest_init_utils.F:712-785).")

    # Accept either woof's mass-level count (nz), WRF's full-interface count
    # (e_vert), or derive both from an explicit eta array.  Every combination
    # is cross-checked before a RunConfig is constructed; no fixed vertical
    # profile is injected when the user omits one spelling.
    eta_levels = tuple(float(v) for v in shared.pop("eta_levels", ()))
    supplied_nz = shared.get("nz")
    supplied_e_vert = _SHARED_KEY_ROWS["e_vert"].get(
        shared, where=f"[shared] of {source}")
    shared.pop("e_vert", None)
    if supplied_nz is None and supplied_e_vert is None and not eta_levels:
        raise ValueError(
            f"[shared] of {source} must carry nz or e_vert, or provide "
            "explicit eta_levels from which the level count can be derived.")
    nz = (None if supplied_nz is None else
          _positive_int("shared", "nz", supplied_nz, source, 4))
    e_vert = (None if supplied_e_vert is None else
              _positive_int("shared", "e_vert", supplied_e_vert, source, 5))
    if nz is not None and e_vert is not None and e_vert != nz + 1:
        raise ValueError(
            f"inconsistent vertical counts in [shared] of {source}: "
            f"nz={nz} mass levels require e_vert={nz + 1}, got {e_vert}.")
    if nz is None:
        nz = (e_vert - 1 if e_vert is not None else len(eta_levels) - 1)
    if e_vert is None:
        e_vert = nz + 1
    if eta_levels and len(eta_levels) != e_vert:
        raise ValueError(
            f"eta_levels in [shared] of {source} has "
            f"{len(eta_levels)} entries but nz = {nz} mass levels "
            f"require e_vert = nz + 1 = {e_vert} full levels.")
    shared["nz"] = nz
    if "ztop" not in shared:
        raise ValueError(
            f"[shared] of {source} must carry ztop (RunConfig requires "
            "it for the vertical grid scaffold).")

    # F1 amendment: nz/eta_levels/p_top/hybrid selectors resolve into the
    # ONE VerticalConfig (its __post_init__ owns the finiteness/ordering
    # invariants); hybrid_opt/etac additionally flow into each RunConfig
    # as derived compatibility copies, asserted equal below.
    try:
        vertical = VerticalConfig(
            eta_levels=eta_levels,
            p_top=float(shared.pop("p_top", 0.0)),
            hybrid_opt=shared.get("hybrid_opt", RunConfig.hybrid_opt),
            etac=float(shared.get("etac", RunConfig.etac)))
    except ValueError as err:
        raise ValueError(f"[shared] of {source}: {err}") from None

    # [projection] is required for real experiments and only meaningful
    # there: a nonzero [shared] map_proj (WRF convention: 1 = Lambert,
    # 2 = polar stereographic, 3 = Mercator) <=> a [projection] table,
    # and the integer must agree with the table's WPS string.
    map_proj = shared.get("map_proj", RunConfig.map_proj)
    if map_proj != 0 and projection is None:
        raise ValueError(
            f"[shared] of {source} sets map_proj = {map_proj} but no "
            "[projection] table is present; real experiments must carry "
            "the projection parameter set (F1 amendment).")
    # The [projection] table carries the full parameter set, so a config
    # that simply omits the [shared] integer is completed from the table
    # and nothing is contradicted -- the user wrote one projection, once.
    #
    # A config that DECLARES the integer and disagrees with the table is a
    # different thing, and it used to be a one-line warning that then ran
    # the table's projection anyway.  That is a wrong answer, not an
    # inconvenience: writing map_proj = 3 beside a "lambert" table gave a
    # complete Mercator-labelled request integrated on a Lambert grid, at
    # exit 0 with `woof check` PASS.  Nothing downstream can recover the
    # discarded intent, because shared["map_proj"] is overwritten here.
    # Warn-not-block covers reports we are confident about; it does not
    # cover "your configuration says two different things and I picked
    # one".  Its sibling checks already agree -- a nonzero map_proj with
    # no table raises four lines above, and native_wrf_contract.py raises
    # on this same disagreement between a WPS namelist and the table.
    declared_map_proj = "map_proj" in shared
    if projection is not None and not declared_map_proj:
        map_proj = projection.wrf_map_proj
        shared["map_proj"] = map_proj
    elif projection is not None and map_proj != projection.wrf_map_proj:
        raise ValueError(
            f"[shared] map_proj = {map_proj} in {source} contradicts the "
            f"[projection] table, which selects {projection.map_proj!r} "
            f"(WRF code {projection.wrf_map_proj}).  No value is chosen "
            f"for you, because the one that would be dropped is the one "
            f"naming the projection your grid is defined on.  Set "
            f"[shared] map_proj = {projection.wrf_map_proj} to keep "
            f"{projection.map_proj!r}, or change [projection] map_proj to "
            f"the projection you meant; removing the [shared] key "
            f"entirely also works, and lets the table speak alone.")

    spec_zone = shared.get("spec_zone", RunConfig.spec_zone)
    relax_zone = shared.get("relax_zone", RunConfig.relax_zone)
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError(
            f"spec_bdy_width = {spec_bdy_width} in [experiment] of "
            f"{source} must cover spec_zone + relax_zone = "
            f"{spec_zone} + {relax_zone}.")

    # ---- [[domain]] ----------------------------------------------------
    # Declaration order used to be a refusal ("parent-before-child").
    # The tree is fully declared by grid_id/parent_id, so when every id
    # parses, the loader orders the tables itself and says so; only a
    # genuinely unresolvable tree (unknown parent, cycle) still refuses,
    # via the existing checks inside the loop.
    domain_tables = _parent_before_child(domain_tables, source)
    domains: list[DomainConfig] = []
    auto_mix_ids: list[int] = []
    auto_epssm_ids: list[int] = []
    by_id: dict[int, DomainConfig] = {}
    dt_by_id: dict[int, Fraction] = {}
    # Each built domain's adaptive-clock step bounds (exact), so a child whose
    # bounds come from [shared] takes its parent's over its time-step ratio.
    adaptive_bounds_by_id: dict[int, dict[str, Fraction]] = {}
    dx_by_id: dict[int, Fraction] = {}
    fp32_by_id: dict[int, np.float32] = {}
    # A tree that mixes SASE with another PBL scheme.  The SASE selectors
    # are [shared] keys, and validate_run_config refuses a non-default
    # value on a domain with no SASE closure for it to change.  On such a
    # tree a [shared] value speaks for the SASE domains: it reaches them
    # and leaves every other domain on the default.  Without this, no
    # spelling selected a SASE variant on a mixed tree -- [shared] was
    # refused on the non-SASE domain and [[domain]] as a misplaced run
    # key.  A tree with no SASE domain keeps the refusal, as a single
    # domain does, and a value written into a non-SASE domain's own table
    # is checked as written.
    tree_runs_sase = any(
        physics_mode.settings.get(
            "bl_pbl_physics",
            dom.get("bl_pbl_physics",
                    shared.get("bl_pbl_physics", RunConfig.bl_pbl_physics)))
        == SASE_PBL_SCHEME
        for dom in domain_tables)
    # The same rule for the Grell-family keys.  clos_choice and ishallow
    # are read only where cu_physics = 3, and the wizard writes them in
    # [shared]; on a tree whose cumulus-off child sits under a
    # Grell-Freitas root, [shared] was refused on the child.  On a tree
    # with a Grell-Freitas domain a [shared] value reaches those domains
    # and leaves the others on the Registry defaults; a tree with none,
    # a single domain, and a value written into a non-Grell domain's own
    # table keep the refusal.
    tree_runs_grell = any(
        physics_mode.settings.get(
            "cu_physics",
            dom.get("cu_physics",
                    shared.get("cu_physics", RunConfig.cu_physics)))
        in GRELL_FAMILY_CU_PHYSICS
        for dom in domain_tables)
    # And for the SAS-family keys (nsas_dx_factor, sas_pgcon), read only
    # where cu_physics is a SAS scheme: a [shared] value reaches the SAS
    # domains and leaves the others on the Registry defaults.
    tree_runs_sas = any(
        physics_mode.settings.get(
            "cu_physics",
            dom.get("cu_physics",
                    shared.get("cu_physics", RunConfig.cu_physics)))
        in SAS_PGCON_SCHEMES
        for dom in domain_tables)
    for index, dom in enumerate(domain_tables):
        _reject_moving_nest_keys("domain", dom, source)
        _reject_domain_vertical_keys(dom, dom.get("grid_id", index + 1),
                                     source)
        _reject_misplaced_run_keys(dom, dom.get("grid_id", index + 1),
                                   source)
        if "devices" in dom:
            raise ValueError("devices on [[domain]] is refused: a tree has one split, "
                             "the experiment [devices] table, whose domains key names "
                             "the grids that run split")
        _reject_unknown_keys("domain", dom, _DOMAIN_KEYS, source)
        _reject_axis_authored_keys(
            f"domain grid_id={dom.get('grid_id', index + 1)}",
            dom, physics_mode, source)
        _require_keys(f"domain #{index + 1}", dom, _DOMAIN_REQUIRED, source)
        grid_id = _positive_int("domain", "grid_id", dom["grid_id"], source)
        if "static" in dom:
            from woof.static.terrain_smoothing import parse_domain_static
            parse_domain_static(dom["static"], source=source, grid_id=grid_id)
        if grid_id in by_id:
            raise ValueError(
                f"duplicate grid_id = {grid_id} in [[domain]] tables of "
                f"{source}.")
        # This domain's mix_isotropic provenance: AUTO when neither this
        # table nor [shared] writes an integer for it (a per-domain
        # "auto" string overrides a [shared] integer, and is stripped
        # here so only resolved integers ever reach RunConfig).
        if isinstance(dom.get("mix_isotropic"), str):
            if dom["mix_isotropic"] != MIX_ISOTROPIC_AUTO:
                raise ValueError(_bad_mix_isotropic_sentinel(
                    dom["mix_isotropic"],
                    f"[[domain]] grid_id={grid_id}", source))
            dom = dict(dom)
            del dom["mix_isotropic"]
            mix_isotropic_auto = True
        else:
            mix_isotropic_auto = ("mix_isotropic" not in dom
                                  and "mix_isotropic" not in shared)
        # epssm's provenance, on the same rule: the model's choice when
        # neither this table nor [shared] writes a number for it.
        if isinstance(dom.get("epssm"), str):
            if dom["epssm"] != EPSSM_AUTO:
                raise ValueError(_bad_epssm_sentinel(
                    dom["epssm"], f"[[domain]] grid_id={grid_id}", source))
            dom = dict(dom)
            del dom["epssm"]
            epssm_auto = True
        else:
            epssm_auto = "epssm" not in dom and "epssm" not in shared
        parent_id = dom["parent_id"]
        is_root = parent_id == 0
        if index == 0 and not is_root:
            raise ValueError(
                f"the first [[domain]] of {source} must be the root "
                f"(parent_id = 0), got parent_id = {parent_id!r} on "
                f"grid_id = {grid_id} (domains are stored "
                "parent-before-child).")
        if index > 0 and is_root:
            raise ValueError(
                f"exactly one root domain (parent_id = 0) is allowed; "
                f"grid_id = {grid_id} of {source} is a second root.")
        if not is_root and parent_id not in by_id:
            raise ValueError(
                f"parent_id = {parent_id!r} of grid_id = {grid_id} in "
                f"{source} does not name a previously declared domain "
                f"(domains must be listed parent-before-child); declared "
                f"so far: {sorted(by_id)}.")

        domain_start = _DOMAIN_KEY_ROWS["start_time"].get(
            dom, where=f"[[domain]] grid_id = {grid_id} of {source}")
        if domain_start is None:
            domain_start = start_time
        if (not isinstance(domain_start, datetime)
                or domain_start.tzinfo is not None):
            raise ValueError(
                f"start_time on [[domain]] grid_id = {grid_id} of {source} "
                "must be an offset-free TOML datetime, got "
                f"{domain_start!r}.")
        if is_root and domain_start != start_time:
            warn(f"root [[domain]] start_time = "
                 f"{domain_start.isoformat()} in {source} is ignored; "
                 f"[experiment].start_time = {start_time.isoformat()} "
                 "is authoritative")
            domain_start = start_time
        # Construct through integer microseconds so the exact offset check
        # below cannot inherit a floating duration.
        run_microseconds = Fraction(str(run_seconds)) * 1_000_000
        if run_microseconds.denominator != 1:
            raise ValueError(
                f"run_seconds = {run_seconds!r} in {source} cannot be "
                "represented on Python's microsecond datetime lattice.")
        stop_time = start_time + timedelta(
            microseconds=int(run_microseconds))
        if domain_start < start_time or domain_start >= stop_time:
            raise ValueError(
                f"start_time = {domain_start.isoformat()} on d{grid_id:02d} "
                f"of {source} must lie in the experiment window "
                f"[{start_time.isoformat()}, {stop_time.isoformat()}).")
        if not is_root:
            parent_start = by_id[parent_id].start_time
            if parent_start is None:
                parent_start = start_time
            if domain_start < parent_start:
                raise ValueError(
                    f"start_time = {domain_start.isoformat()} on "
                    f"d{grid_id:02d} of {source} precedes parent "
                    f"d{parent_id:02d} start_time = "
                    f"{parent_start.isoformat()}.")
            delta = domain_start - parent_start
            offset = (Fraction(delta.days * 86400 + delta.seconds)
                      + Fraction(delta.microseconds, 1_000_000))
            parent_steps = offset / dt_by_id[parent_id]
            if parent_steps.denominator != 1:
                raise ValueError(
                    f"delayed start_time for d{grid_id:02d} "
                    f"({domain_start.isoformat()}; offset "
                    f"{float(offset):g} s) is not aligned to its parent "
                    f"step boundary: d{parent_id:02d} dt = "
                    f"{dt_by_id[parent_id]} s exactly, "
                    f"(child-parent start offset)/dt = "
                    f"{parent_steps}.")
            # Task #205 RETIRED the categorical refusal that stood here.
            # It named a real breakage -- a delayed child's first history
            # frame consumed a REFL_10CM stash no step of that domain had
            # produced, and the run died at the activation epoch -- and
            # that breakage is fixed at the consume sites
            # (woof.core.refl.refl_10cm_stash_is_due asks the DOMAIN's
            # own start tick).  A delayed start is a supported shape on
            # the experiment-tree route again, proven by a bare-default
            # GPU run whose d02 activates 6 h in.  The structural
            # refusals above (window, parent precedence, step alignment)
            # are untouched, and the routes that still lack activation
            # machinery refuse BY ROUTE through
            # :func:`refuse_delayed_activation` rather than here.

        # --- spawn: the dormant-nest declaration ------------------------
        spawn_cfg = None
        if "spawn" in dom:
            if is_root:
                raise ValueError(
                    f"spawn on the root [[domain]] grid_id = {grid_id} of "
                    f"{source} is refused: a spawn materializes a CHILD "
                    "inside its parent at trigger time, and the root has "
                    "no parent to be placed in.")
            if not isinstance(dom["spawn"], dict):
                raise ValueError(
                    f"spawn on [[domain]] grid_id = {grid_id} of {source} "
                    "must be an inline TABLE, e.g. spawn = { trigger = "
                    f"\"uh\", ... }}, got {dom['spawn']!r}.")
            if "start_time" in dom:
                raise ValueError(
                    f"[[domain]] grid_id = {grid_id} of {source} declares "
                    "both spawn and start_time; a dormant nest's "
                    "activation time belongs to its trigger, and a "
                    "delayed start beside it would be a second author of "
                    "the same instant. Remove one.")
            from woof.core.nest_spawn import build_spawn_config
            spawn_cfg = build_spawn_config(
                dict(dom["spawn"]), source, grid_id=grid_id)

        # --- retire/rearm/follow: per-domain lifecycle -------------------
        retire_cfg = rearm_cfg = follow_cfg = None
        if "retire" in dom:
            if spawn_cfg is None:
                raise ValueError(
                    f"retire on [[domain]] grid_id = {grid_id} of {source} "
                    "requires spawn: retirement is episode policy for a "
                    "trigger-spawned slot, not an alternate end_time")
            if not isinstance(dom["retire"], dict):
                raise ValueError(f"retire on d{grid_id:02d} must be an inline TABLE")
            from woof.core.nest_lifecycle import build_retire_config
            retire_cfg = build_retire_config(dict(dom["retire"]), source, grid_id=grid_id)
        if "rearm" in dom:
            if spawn_cfg is None or retire_cfg is None:
                raise ValueError(
                    f"rearm on [[domain]] grid_id = {grid_id} of {source} "
                    "requires both spawn and retire; without retirement "
                    "there is no legal boundary at which to re-arm")
            if not isinstance(dom["rearm"], dict):
                raise ValueError(f"rearm on d{grid_id:02d} must be an inline TABLE")
            from woof.core.nest_lifecycle import build_rearm_config
            rearm_cfg = build_rearm_config(dict(dom["rearm"]), source, grid_id=grid_id)
        if "follow" in dom:
            if is_root:
                raise ValueError(f"follow on root d{grid_id:02d} is refused: only a child has a placement inside a parent")
            if not isinstance(dom["follow"], dict):
                raise ValueError(f"follow on d{grid_id:02d} must be an inline TABLE")
            from woof.core.nest_lifecycle import build_domain_follow_config
            follow_cfg = build_domain_follow_config(
                dict(dom["follow"]), source, grid_id=grid_id)

        # --- tiles: this domain's own road -------------------------------
        # Validated by the SAME parser the tree-wide table uses, so one
        # vocabulary of keys, one set of value refusals, one place a new
        # knob is added.  A domain that says nothing takes the tree-wide
        # table; a domain that speaks overrides it entirely rather than
        # merging key-by-key, because a half-inherited tiling ("mode from
        # the tree, store from the domain") is a configuration nobody can
        # read off the file.
        domain_tiles = None
        if "tiles" in dom:
            if not isinstance(dom["tiles"], dict):
                raise ValueError(
                    f"tiles on [[domain]] grid_id = {grid_id} of {source} "
                    "must be an inline TABLE, e.g. tiles = { mode = "
                    f"\"auto\" }}, got {dom['tiles']!r}.")
            # The two budget keys name a CARD, not a domain, and the tree
            # decision subtracts every domain's claim from one number.
            # Per-domain copies of it would be several answers to "how
            # much VRAM is there", and the walk would have to pick one --
            # so they are refused here, where the fix is obvious, rather
            # than silently overridden where it is not.
            card_keys = sorted({"vram_budget_bytes", "host_budget_bytes"}
                               & set(dom["tiles"]))
            if card_keys:
                raise ValueError(
                    f"tiles on [[domain]] grid_id = {grid_id} of {source} "
                    f"sets {card_keys}, which name the CARD and not this "
                    "domain: the tree decision prices every domain against "
                    "one budget, so a per-domain copy would be a second "
                    "answer to how much VRAM there is.  Set them on the "
                    "tree-wide [tiles] table instead; the per-domain table "
                    "chooses this domain's ROAD.")
            domain_tiles = streaming_module.StreamingOptions.from_mapping(
                dict(dom["tiles"]),
                source=f"[[domain]] grid_id = {grid_id} of {source}")

        # --- output: this domain's own history variables -----------------
        # Same parser as the tree-wide table, same vocabulary, same
        # refusals, and the same all-or-nothing override rule: a domain
        # that speaks replaces the tree-wide selection entirely rather
        # than merging key by key, because "the preset from the tree, the
        # drop list from the domain" is a configuration nobody can read
        # off the file.
        domain_output = None
        if "output" in dom:
            if not isinstance(dom["output"], dict):
                raise ValueError(
                    f"output on [[domain]] grid_id = {grid_id} of {source} "
                    "must be an inline TABLE, e.g. output = { preset = "
                    f"\"minimal\" }}, got {dom['output']!r}.")
            domain_output = (
                history_selection_module.HistorySelection.from_mapping(
                    dict(dom["output"]),
                    source=f"[[domain]] grid_id = {grid_id} of {source}"))

        # Flags: root takes external specified LBCs, children are nested
        # -- WRF &bdy_control, bundle namelist.input specified=T,F,F,F /
        # nested=F,T,T,T; mutually exclusive by construction.
        specified = dom.get("specified", is_root)
        nested = dom.get("nested", not is_root)
        if is_root and (specified is not True or nested is not False):
            warn(f"root domain grid_id = {grid_id} of {source}: "
                 f"specified = {specified!r}, nested = {nested!r} "
                 "corrected to specified = true, nested = false",
                 why="WRF &bdy_control: the head grid takes external "
                     "lateral boundaries; the flags derive from "
                     "parent_id and are never a real choice.")
            specified, nested = True, False
        if not is_root and (nested is not True or specified is not False):
            warn(f"child domain grid_id = {grid_id} of {source}: "
                 f"specified = {specified!r}, nested = {nested!r} "
                 "corrected to nested = true, specified = false",
                 why="WRF &bdy_control: children are forced by their "
                     "parent (bundle specified=T,F,F,F / nested=F,T,T,T); "
                     "the flags derive from parent_id and are never a "
                     "real choice.")
            specified, nested = False, True

        ratio = _positive_int("domain", "parent_grid_ratio",
                              dom["parent_grid_ratio"], source)
        tratio = _positive_int("domain", "parent_time_step_ratio",
                               dom["parent_time_step_ratio"], source)
        i_start = _positive_int("domain", "i_parent_start",
                                dom["i_parent_start"], source)
        j_start = _positive_int("domain", "j_parent_start",
                                dom["j_parent_start"], source)
        if is_root:
            corrected = [key for key, value in
                         (("parent_grid_ratio", ratio),
                          ("parent_time_step_ratio", tratio),
                          ("i_parent_start", i_start),
                          ("j_parent_start", j_start)) if value != 1]
            if corrected:
                warn(f"root domain grid_id = {grid_id} of {source}: "
                     f"{', '.join(corrected)} corrected to 1 (a root has "
                     "no parent, so these keys carry no information)")
                ratio = tratio = i_start = j_start = 1
        else:
            for key, value in (("parent_grid_ratio", ratio),
                               ("parent_time_step_ratio", tratio)):
                if value < 2:
                    raise ValueError(
                        f"{key} = {value} on child domain grid_id = "
                        f"{grid_id} of {source} must be >= 2 (a ratio-1 "
                        "child is not a refinement).")

        nx, ny = _mass_dims(dom, grid_id, source)

        # --- exact-rational dt/dx (never hand-typed on children) -------
        if is_root:
            if "time_step" not in dom:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of {source} "
                    "must carry time_step as integer seconds (plus "
                    "optional time_step_fract_num/time_step_fract_den, "
                    "Registry.EM_COMMON:2245-2246).")
            time_step = _positive_int("domain", "time_step",
                                      dom["time_step"], source, 0)
            fract_num = _positive_int(
                "domain", "time_step_fract_num",
                dom.get("time_step_fract_num", 0), source, 0)
            fract_den = _positive_int(
                "domain", "time_step_fract_den",
                dom.get("time_step_fract_den", 1), source)
            dt_ex = Fraction(time_step) + Fraction(fract_num, fract_den)
            if dt_ex <= 0:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of "
                    f"{source} resolves a non-positive model step: "
                    f"time_step = {time_step} + {fract_num}/{fract_den} "
                    f"s.")
            # A root `dt` used to warn "is ignored" and be discarded with
            # no comparison at all, so `dt = 60.0` beside `time_step = 5`
            # ran a 5 s step at exit 0 -- and the SAME warning was
            # emitted when the two agreed, which made it noise on the
            # harmless case and silent in effect on the harmful one.
            # Both spellings are the user's and they name one quantity,
            # so the disagreement is checked exactly the way a child's
            # hand-typed dt already is, ten lines below.
            if "dt" in dom:
                _cross_check(
                    "dt", grid_id, dom["dt"], dt_ex,
                    f"time_step {time_step} + {fract_num}/{fract_den} s "
                    "on this root domain", source)
            # WRF-REAL head-grid dt: single-precision evaluation of the
            # rational namelist keys (Registry.EM_COMMON:2245-2246).
            dt_fp32 = (np.float32(time_step)
                       + np.float32(fract_num) / np.float32(fract_den))
            if "dx" not in dom:
                raise ValueError(
                    f"the root [[domain]] grid_id = {grid_id} of {source} "
                    "must carry dx (metres); children derive theirs from "
                    "the ratio chain.")
            # Finiteness BEFORE Fraction(), not after.  Fraction(nan)
            # raises ValueError and reaches the front door as rc 2, but
            # Fraction(inf) raises OverflowError -- one branch apart, and
            # only the first is caught -- so `dx = inf` and `dx = nan`
            # produced a sentence and a traceback for the same mistake.
            dx_value = float(dom["dx"])
            if not math.isfinite(dx_value) or dx_value <= 0:
                raise ValueError(
                    f"dx = {dom['dx']!r} on the root domain of {source} "
                    "must be a positive, finite grid spacing in metres.")
            dx_ex = Fraction(dx_value)
            if "dy" in dom and float(dom["dy"]) != float(dom["dx"]):
                raise ValueError(
                    f"dy = {dom['dy']!r} on grid_id = {grid_id} of "
                    f"{source} must equal dx = {dom['dx']!r} (Lambert "
                    "grids are isotropic).")
        else:
            time_step, fract_num, fract_den = None, 0, 1
            dt_ex = dt_by_id[parent_id] / tratio
            # `time_step` on a child was warned-and-dropped while `dt` on
            # the same table refuses through _cross_check ten lines below
            # -- the same quantity, the same number, one spelling running
            # a different step at exit 0.  The old warning's own `why=`
            # advertised the check it declined to perform ("A hand-typed
            # child dt key is still cross-checked against the chain"): it
            # was describing the sibling branch, not itself.  That WRF
            # has no per-child time_step is the reason to refuse, not to
            # warn: the model cannot deliver what was asked.
            if ("time_step" in dom or "time_step_fract_num" in dom
                    or "time_step_fract_den" in dom):
                child_step = (
                    Fraction(_positive_int(
                        "domain", "time_step", dom.get("time_step", 0),
                        source, 0))
                    + Fraction(
                        _positive_int(
                            "domain", "time_step_fract_num",
                            dom.get("time_step_fract_num", 0), source, 0),
                        _positive_int(
                            "domain", "time_step_fract_den",
                            dom.get("time_step_fract_den", 1), source)))
                # float(), not the Fraction: _cross_check reprs what it
                # is given, and "hand-types time_step=Fraction(5, 1)"
                # shows the reader this loader's internals rather than
                # the number they typed.
                _cross_check(
                    "time_step", grid_id, float(child_step), dt_ex,
                    f"parent dt {dt_by_id[parent_id]} s / "
                    f"parent_time_step_ratio {tratio}", source)
            # CHAINED float32 division down the ratio chain -- WRF's REAL
            # grid%dt = parent%dt / parent_time_step_ratio at every tree
            # edge (share/set_timekeeping.F:368), so the kernel dt
            # matches WRF bit-for-bit on ANY ratio chain (§C; the bundle
            # chain lands np.float32(60)/4/3/3 = 1.6666666, 0x3FD55555).
            # Cadence/tick arithmetic keeps using the exact rational.
            dt_fp32 = fp32_by_id[parent_id] / np.float32(tratio)
            dx_ex = dx_by_id[parent_id] / ratio
            if "dt" in dom:
                _cross_check(
                    "dt", grid_id, dom["dt"], dt_ex,
                    f"parent dt {dt_by_id[parent_id]} s / "
                    f"parent_time_step_ratio {tratio}", source)
            if "dx" in dom:
                _cross_check(
                    "dx", grid_id, dom["dx"], dx_ex,
                    f"parent dx {dx_by_id[parent_id]} m / "
                    f"parent_grid_ratio {ratio}", source)
            if "dy" in dom:
                _cross_check(
                    "dy", grid_id, dom["dy"], dx_ex,
                    f"parent dx {dx_by_id[parent_id]} m / "
                    f"parent_grid_ratio {ratio}", source)

        # --- footprint: alignment + parent-row clearance ----------------
        if not is_root:
            parent = by_id[parent_id]
            for axis, size, start, parent_size in (
                    ("west-east", nx, i_start, parent.run.nx),
                    ("south-north", ny, j_start, parent.run.ny)):
                if size % ratio != 0:
                    raise ValueError(
                        f"child domain grid_id = {grid_id} of {source}: "
                        f"{axis} extent {size} mass cells is not an "
                        f"integer multiple of parent_grid_ratio = "
                        f"{ratio} (WPS requires e_we/e_sn = "
                        "n * ratio + 1).")
                span = size // ratio
                near = start - 1
                far = parent_size - (start + span - 1)
                need = spec_bdy_width + blend_width
                for side, clearance in ((f"{axis} low", near),
                                        (f"{axis} high", far)):
                    if clearance < need:
                        raise ValueError(
                            f"child domain grid_id = {grid_id} of "
                            f"{source} violates the parent-row clearance "
                            f"rule: {side} clearance is {clearance} "
                            f"parent rows but spec_bdy_width + "
                            f"blend_width = {spec_bdy_width} + "
                            f"{blend_width} = {need} rows are required "
                            "(the child boundary must clear the parent's "
                            "own Davies and terrain-blend zones).")

        # --- per-domain RunConfig ---------------------------------------
        history_interval_s = float(dom["history_interval_s"])
        if not math.isfinite(history_interval_s) or history_interval_s <= 0:
            raise ValueError(
                f"history_interval_s = {dom['history_interval_s']!r} on "
                f"domain grid_id = {grid_id} of {source} must be a "
                "finite positive interval in seconds.")
        history_begin_s, history_end_s = _history_window(
            dom, grid_id=grid_id, source=source,
            run_seconds=run_seconds)
        # The quick-look stream: the alarm becomes its cadence, and the
        # full frames keep the declared interval (DomainConfig docs).
        history_interval_s, full_history_interval_s, quicklook_output = (
            _quicklook(dom, grid_id=grid_id, source=source,
                       history_interval_s=history_interval_s))
        kw = dict(shared)
        kw.update(
            nx=nx, ny=ny, nz=nz, dx=float(dx_ex), dy=float(dx_ex),
            # run.dt carries the chained-FP32 WRF kernel dt exactly (its
            # binary64 image); the exact rational stays on dt_exact().
            dt=float(dt_fp32), clock_dt=0.0, run_seconds=run_seconds,
            output_interval_s=history_interval_s,
            specified=bool(specified), nested=bool(nested),
            grid_id=grid_id, spec_bdy_width=spec_bdy_width,
            # F14 timing authority: derived compatibility copy on every
            # domain (equality asserted below); restart ALARMS evaluate
            # on the d01 clock only (section C).
            restart_interval_s=restart_interval_s,
        )
        for key in _DOMAIN_RUN_OVERRIDES:
            if key in dom:
                kw[key] = dom[key]
        # WRF's adaptive nest steps at its parent's step over
        # parent_time_step_ratio.  The starting/min/max bounds are per-domain
        # keys, and a [shared] value reached every child verbatim: under the
        # HRRR recipe (all three pinned at 20 s) a 1 km nest stepped at the
        # parent's 20 s and went non-finite within eight steps (box D,
        # 2026-10-21).  A child that does not set a bound itself takes its
        # parent's over the ratio, as an exact fraction.
        if parent_id != 0 and kw.get("use_adaptive_time_step"):
            parent_bounds = adaptive_bounds_by_id.get(int(parent_id), {})
            for bound_key in ("starting_time_step", "max_time_step", "min_time_step"):
                if bound_key in dom or f"{bound_key}_den" in dom:
                    continue
                inherited = parent_bounds.get(bound_key)
                if inherited is None:
                    continue
                value = inherited / int(dom["parent_time_step_ratio"])
                kw[bound_key] = int(value.numerator)
                kw[f"{bound_key}_den"] = int(value.denominator)
        # The axis writes its resolved vector LAST and onto every domain:
        # it is the author of these keys (the [shared]/[[domain]] refusal
        # above guarantees nobody else wrote them), and one experiment is
        # one arm -- a tree whose domains ran different ledger entries
        # could not be compared across its own nest boundary.
        kw.update(physics_mode.settings)
        if tree_runs_sase and kw.get("bl_pbl_physics") != SASE_PBL_SCHEME:
            for key, default in SASE_FAIL_CLOSED_DEFAULTS.items():
                if key not in dom:
                    kw[key] = default
        if (tree_runs_grell and kw.get("cu_physics", RunConfig.cu_physics)
                not in GRELL_FAMILY_CU_PHYSICS):
            for key, default in GRELL_FAMILY_DEFAULTS.items():
                if key not in dom:
                    kw[key] = default
        # G3-only shared controls reach only G3 domains. Explicit domain
        # controls remain validated, so an off-scheme value is not discarded.
        from woof.config import G3_RUN_DEFAULTS
        if tree_runs_grell and kw.get("cu_physics", RunConfig.cu_physics) != 5:
            for key, default in G3_RUN_DEFAULTS.items():
                if key not in dom:
                    kw[key] = default
        if (tree_runs_sas and kw.get("cu_physics", RunConfig.cu_physics)
                not in SAS_PGCON_SCHEMES):
            for key, default in SAS_FAMILY_DEFAULTS.items():
                if key not in dom:
                    kw[key] = default
        # bl_pbl_physics is per-domain for every scheme, SASE included:
        # each domain's PhysicsDriver runs its own closure on its own
        # state, and SASE's prognostic e_sgs is not a nest-forced field
        # (woof/core/nest_fields.py), so a SASE domain cold-starts it
        # whatever its parent runs.  The multi-domain SASE warning below
        # covers this tree too.
        # Nonzero spec_exp on a NESTED domain is forced to 0 with a
        # warning: WRF's nested lbc_fcx_gcx branch has NO exponential
        # sponge term -- only the specified branch applies spec_exp
        # (dyn_em/module_bc_em.F lbc_fcx_gcx :1297-1341: specified
        # branch spongeweight :1320, nested branch :1325-1337 with the
        # sponge lines commented out).  woof's nested Davies weights
        # DO read cfg.spec_exp, so zeroing here is what keeps them the
        # exact spec_exp = 0 transliteration of WRF's nested branch
        # (the N1 child_spec_exp pin: children force with spec_exp=0).
        if not is_root and float(kw.get("spec_exp", 0.0)) != 0.0:
            warn(f"spec_exp = {kw['spec_exp']!r} on nested domain "
                 f"grid_id = {grid_id} of {source} is forced to 0, "
                 "matching WRF (the sponge applies on specified "
                 "boundaries only); continuing",
                 why="dyn_em/module_bc_em.F:1297-1341: the nested "
                     "lbc_fcx_gcx branch has no sponge term (:1325); "
                     "children force with spec_exp = 0, exactly as WRF "
                     "would run this namelist.")
            kw["spec_exp"] = 0.0
        # An AUTO domain starts on the RunConfig default (0, WRF's
        # Registry value) so the whole validation battery and the
        # exposure computation below see a resolved integer; the
        # criterion-driven switch to 1 happens in ONE place,
        # resolve_auto_mix_isotropic, after the tree is assembled and
        # the layer depths are knowable.  The pop also covers a
        # per-domain "auto" under a [shared] integer.
        if mix_isotropic_auto:
            kw.pop("mix_isotropic", None)
            auto_mix_ids.append(grid_id)
        # An AUTO epssm starts on the RunConfig default (WRF's 0.1); the
        # floor that raises it over steep ground needs the terrain, so it
        # is applied where the terrain is read
        # (woof.acoustic_adaptation.adapt_experiment_acoustics).
        if epssm_auto:
            kw.pop("epssm", None)
            auto_epssm_ids.append(grid_id)
        # The chem name lists may arrive from TOML as arrays; RunConfig
        # stores them as one comma-separated string (as load_config does).
        # [shared]-only: every domain of a tree carries the same chem sets,
        # because a nest's chem boundary is its parent's field.
        from woof.config import apply_chem_set_defaults, join_chem_name_lists
        join_chem_name_lists(kw)
        # The sets' own scheme defaults (smoke: washout on, hourly plume),
        # for keys this configuration leaves unset, as load_config does.
        apply_chem_set_defaults(kw)
        # The full legacy invariant battery applies to every per-domain
        # RunConfig (p5t1 review F1): same checks, same messages as
        # load_config.
        from woof.surface_ocean import apply_ocean_defaults
        apply_ocean_defaults(kw)
        run = validate_run_config(RunConfig(**kw))
        bounds = {}
        for bound_key in ("starting_time_step", "max_time_step", "min_time_step"):
            num = getattr(run, bound_key, None)
            if num is None or (isinstance(num, (int, float)) and num <= 0):
                continue
            den = int(getattr(run, f"{bound_key}_den", 0) or 0) or 1
            bounds[bound_key] = Fraction(num) / den
        adaptive_bounds_by_id[int(grid_id)] = bounds

        # --- cadence divisibility (integer domain steps) ----------------
        # TOML decimal minutes use the same decimal clock as run_seconds;
        # a binary float Fraction would make 2.4 minutes fail on 72 s steps.
        # The adaptive driver shortens the root step to every domain's
        # history alarm and divides that step down the hierarchy. Its
        # varying steps need not divide the declared history interval.
        if not run.use_adaptive_time_step:
            _check_cadence("history_interval_s", Fraction(history_interval_s),
                           dt_ex, grid_id, source)
        _check_whole_second_cadence(
            "history_interval_s", Fraction(history_interval_s), grid_id,
            source)
        # radt / cudt / bldt run on wrf.exe's whole steps (NINT), and the
        # clock resolves them the same way; one that is not a whole
        # number of steps is said, not refused.
        if not run.use_adaptive_time_step:
            _warn_wrf_physics_cadences(run, dt_ex, grid_id, source)
        if is_root:
            _check_cadence("restart_interval_s",
                           Fraction(restart_interval_s), dt_ex, grid_id,
                           source)
            _check_whole_second_cadence(
                "restart_interval_s", Fraction(restart_interval_s), grid_id,
                source)
            # The forecast length is on the same step grid its output
            # cadences are, and is refused in the same place and the
            # same way.
            _check_run_length_on_step_grid(
                Fraction(str(run_seconds)), dt_ex, grid_id, source)

        if spawn_cfg is not None:
            # Timing sanity against THIS tree's clocks.  The manual
            # trigger's instant must land on the parent's step lattice
            # (spawning is a cycle-boundary operation); a field window
            # opening at or after the end of the run can never fire and
            # is a disabled feature wearing an enabled name.
            parent_dt = dt_by_id[parent_id]
            if spawn_cfg.at_s is not None:
                steps = Fraction(str(float(spawn_cfg.at_s))) / parent_dt
                if steps.denominator != 1:
                    raise ValueError(
                        f"spawn at_s = {spawn_cfg.at_s:g} s on [[domain]] "
                        f"grid_id = {grid_id} of {source} is not a whole "
                        f"number of parent d{parent_id:02d} steps: dt = "
                        f"{parent_dt} s exactly. A spawn is a "
                        "cycle-boundary operation and its manual instant "
                        "must land on the parent step grid.")
                if float(spawn_cfg.at_s) >= run_seconds:
                    raise ValueError(
                        f"spawn at_s = {spawn_cfg.at_s:g} s on [[domain]] "
                        f"grid_id = {grid_id} of {source} is at or past "
                        f"run_seconds = {run_seconds:g}; the nest could "
                        "never spawn, so its declaration (and its VRAM "
                        "reservation) would buy nothing.")
            if (spawn_cfg.earliest_s is not None
                    and float(spawn_cfg.earliest_s) >= run_seconds):
                raise ValueError(
                    f"spawn earliest_s = {spawn_cfg.earliest_s:g} s on "
                    f"[[domain]] grid_id = {grid_id} of {source} is at or "
                    f"past run_seconds = {run_seconds:g}; the window can "
                    "never open, so the nest could never spawn.")

        dc = DomainConfig(
            grid_id=grid_id, parent_id=parent_id, i_parent_start=i_start,
            j_parent_start=j_start, parent_grid_ratio=ratio,
            parent_time_step_ratio=tratio,
            history_interval_s=history_interval_s, run=run,
            time_step=time_step, time_step_fract_num=fract_num,
            time_step_fract_den=fract_den, start_time=domain_start,
            history_begin_s=history_begin_s, history_end_s=history_end_s,
            spawn=spawn_cfg, retire=retire_cfg, rearm=rearm_cfg,
            follow=follow_cfg, tiles=domain_tiles, output=domain_output,
            full_history_interval_s=full_history_interval_s,
            quicklook_output=quicklook_output)
        domains.append(dc)
        by_id[grid_id] = dc
        dt_by_id[grid_id] = dt_ex
        dx_by_id[grid_id] = dx_ex
        fp32_by_id[grid_id] = dt_fp32

    # Resolve every directed physics edge before constructing the experiment.
    # Same-scheme trees retain their original behavior; mixed schemes require
    # an explicit GPUWM transition instead of reaching a missing parent field.
    from woof.core.microphysics_transition import (
        mixed_edge_entry_note, resolve_microphysics_transition,
    )
    for dc in domains[1:]:
        contract = resolve_microphysics_transition(
            by_id[dc.parent_id].run, dc.run)
        # A mixed edge that seeds a moment from a ported default rather
        # than from the parent's own field runs, and says which mapping
        # it used, in one line here.
        note = mixed_edge_entry_note(contract)
        if note is not None:
            warn(f"domain grid_id = {dc.grid_id}: {note}")

    # km_opt=2 on a nest child, under any parent.
    #
    # WRF gives tke no ``i`` (nest-interpolation) and no ``f`` (feedback)
    # Registry flag (Registry.EM_COMMON:312), so a child cold-starts its own
    # TKE and never returns it.  woof does exactly that for every parent:
    # tke is not in woof/core/nest_fields.py's forced-field inventory, so
    # neither the child's boundary forcing nor the feedback reads or writes
    # it, and each domain's DomainState allocates its own zero TKE from its
    # own km_opt.  Under a parent carrying no TKE this tree has been run --
    # a 402x402 250 m km_opt=2 PBL-off child under a km_opt=4 750 m parent,
    # 7 h, status PASS (docs/superpowers/receipts/les/
    # nested-les-km2-2026-08-02.md).  Under a km_opt=2 parent the child
    # does the same thing; the parent's TKE simply stays on the parent, as
    # it does in WRF.  That tree is admitted and the run says it is
    # implemented but not yet verified against a reference.
    for dc in domains[1:]:
        parent = by_id[dc.parent_id].run
        if dc.run.km_opt == 2 and parent.km_opt == 2:
            warn_once(
                f"nested-km2-under-km2-{dc.run.grid_id}-{parent.grid_id}",
                f"domain grid_id = {dc.run.grid_id} runs km_opt=2 under a "
                f"km_opt=2 parent (grid_id = {parent.grid_id}): the child "
                "cold-starts its own TKE and the parent keeps its own, as "
                "in WRF, which neither interpolates TKE to a nest nor "
                "feeds it back. Implemented, not yet verified against a "
                "reference run.",
                why="Registry.EM_COMMON:312 declares tke with no i and no "
                    "f flag; woof/core/nest_fields.py carries no tke "
                    "row. The measured nested km_opt=2 tree had a "
                    "km_opt=4 parent "
                    "(docs/superpowers/receipts/les/"
                    "nested-les-km2-2026-08-02.md).")

    # SASE on a domain tree, uniform or per domain.  Each SASE domain runs
    # its own closure on its own state: the prognostic subgrid energy
    # e_sgs is allocated per DomainState and is not a nest-forced field,
    # so a SASE nest cold-starts it at the realizability floor (as a
    # single domain does at step 0) and never feeds it back.  Its lateral
    # edges take the specified-domain boundary policy (e_sgs held at the
    # floor across spec_bdy_width rows, the same rows excluded from the
    # dynamic solve; PhysicsDriver._run_sase).  The closure's calibration
    # is single-domain, so the run says the tree is implemented but not
    # verified.
    if len(domains) > 1:
        sase_ids = [dc.grid_id for dc in domains
                    if dc.run.bl_pbl_physics == SASE_PBL_SCHEME]
        if sase_ids:
            listed = ", ".join(str(i) for i in sase_ids)
            warn_once(
                f"nested-sase-{listed}-of-{len(domains)}",
                f"bl_pbl_physics = {SASE_PBL_SCHEME} (SASE) runs on "
                f"grid_id {listed} of a {len(domains)}-domain tree: each "
                "SASE nest cold-starts its own subgrid energy e and holds "
                "it at the floor across its boundary rows, and no domain "
                "hands e to another. Implemented, not yet verified "
                "against a reference run.",
                why="e_sgs is allocated per DomainState "
                    "(woof/core/state.py) and is not in "
                    "woof/core/nest_fields.py; the SASE closure's "
                    "calibration cases (GABLS1 and the real-data "
                    "confirmation runs) are single-domain.")

    # A dormant nest must be a LEAF: a child declared under it would need
    # cascading activation (its parent does not exist until a trigger
    # fires), which no runner implements.  Refused rather than deferred,
    # because the child's own reservation would otherwise price a domain
    # that could never legally start.
    dormant_ids = {dc.grid_id for dc in domains if dc.spawn is not None}
    for dc in domains:
        if dc.parent_id in dormant_ids:
            raise ValueError(
                f"[[domain]] grid_id = {dc.grid_id} of {source} declares "
                f"dormant d{dc.parent_id:02d} as its parent; a nest under "
                "a spawn-triggered nest would need cascading activation, "
                "which is not implemented. Declare the spawn on the leaf, "
                "or make the parent an ordinary domain.")

    if any(getattr(dc.run, "umove", 0.0) or getattr(dc.run, "vmove", 0.0) for dc in domains):
        if len(domains) != 1 or raw.get("relocation", {}).get("enabled", False):
            raise ValueError("umove/vmove with nests or moving nests needs nested ground geometry translated each step")
    relocation = _build_relocation(raw, source, domains, run_seconds)
    from woof.core.nest_lifecycle import validate_follow_tracks
    domains = validate_follow_tracks(domains, relocation, source)
    _refuse_windowed_stash_watch(domains, source)
    _refuse_moving_slope_radiation(domains, relocation, source)
    _refuse_rebuilt_nest_noah_mosaic(domains, relocation, source)
    _refuse_rebuilt_nest_shortwave_interpolation(domains, relocation, source)
    _refuse_rebuilt_nest_lake(domains, relocation, source)
    _refuse_child_terrain_drag(domains, source)
    from woof.core.attribute_tracking import validate_attribute_domains
    validate_attribute_domains(domains, relocation)
    experiment = ExperimentConfig(
        name=name, start_time=start_time, run_seconds=run_seconds,
        vertical=vertical, projection=projection,
        feedback=feedback, smooth_option=smooth_option,
        blend_width=blend_width, spec_bdy_width=spec_bdy_width,
        smooth_cg_topo=smooth_cg_topo,
        restart_interval_s=restart_interval_s, domains=tuple(domains),
        column_chunk=column_chunk,
        acknowledgements=tuple(acknowledgements),
        constant_glw_wm2=raw_constant_glw,
        relocation=relocation,
        physics_mode=physics_mode,
        perturbation=perturbation,
        tiles=tiles, devices=devices, output=output, simulated_radar=simulated_radar,
        radar_heating=radar_heating,
        spectral_numerics=spectral_numerics,
        nudging=_nudging_from_tables(raw, domains),
        auto_epssm=tuple(sorted(auto_epssm_ids)),
        physics_params=physics_params)
    from woof.static.terrain_smoothing import refuse_moving_reach
    refuse_moving_reach(domain_tables, experiment, source=source)
    # The mixing-length auto-switch runs HERE, at the one load every
    # front door shares, so run/go/check, both prepared runners and the
    # wizard's candidate loop all execute (and announce) the same
    # selection; the advisory pass further down then sees the RESOLVED
    # tree and warns only about written-out exposures.
    experiment = resolve_auto_mix_isotropic(experiment, auto_mix_ids, source)
    # THE PLAN-TIME DEPENDENCY CHECK for [output].  woof owns the render
    # product catalog, so it can resolve which products die with which
    # dropped variables and say so HERE -- at the one load every front
    # door shares -- rather than leaving the user to discover it hours
    # later at render time with a directory of missing PNGs.  It is a
    # WARNING and not a refusal: shedding a product to get the disk back
    # is exactly what the surface is for.  A full-default tree resolves
    # to FULL on every domain and this loop says nothing.
    for dc in experiment.domains:
        selection = history_selection_module.resolve(
            experiment.output, dc.output)
        simulated_radar_module.validate_history_selection(
            simulated_radar, selection, where=f"d{dc.grid_id:02d} of {source}")
        selection.warn_lost_products(
            history_selection_module.HISTORY_VOCABULARY,
            where=f"d{dc.grid_id:02d} of {source}")
    # [tiles] against the TREE, and it needs the assembled experiment
    # because neither half of the question is answerable alone: the block
    # is parsed hundreds of lines above, the parent_id edges hundreds of
    # lines below, and only here are both in hand.  mode = 'on' streams
    # every grid, so on a tree every coupling edge would have BOTH ends
    # streamed -- the one concurrent-nesting shape the coupler refuses --
    # and the combination is refused from the config text, at the one load
    # every front door shares, rather than at the first FORCE, which on a
    # prepared tree is a fetch, two preparations and a whole tree
    # downstream of a fact that was legible in the TOML.  mode = 'auto' is
    # deliberately NOT refused: streamed children and streamed parents are
    # both legal roads, the planner prices each domain against the budget
    # its predecessors left, and steppers_for_tree refuses a both-streamed
    # edge at decision time, before anything is built.
    streaming_module.refuse_streamed_nests(experiment, source=source)
    # The UCM's first-level fatal (module_sf_urban.F:825), refused from the
    # config text at the one load every front door shares instead of at
    # step 1 after a fetch and a preparation.
    ucm_refusal = ucm_first_level_refusal(experiment, source=source)
    if ucm_refusal is not None:
        raise ValueError(ucm_refusal)
    from woof.physics_compat import (
        constant_longwave_refusal,
        nocturnal_radiation_refusal,
        radiation_off_land_surface_refusal,
        validate_resolved_physics_vertical_levels,
    )
    for domain in experiment.domains:
        validate_resolved_physics_vertical_levels(
            domain.run, p_top=experiment.vertical.p_top)
    # The nocturnal-radiation guard (2026-08-06): a real case (it has a
    # [projection], so it has a place and a clock) whose window includes
    # local night may not run shortwave with longwave OFF undeclared --
    # the surface would radiate all night with no downward longwave and
    # the skin temperature and 2 m moisture collapse.  Guarded HERE, at
    # the one load every front door shares (run/go/check, both prepared
    # runners, the DA drivers, the wizard's candidate loop), so no door
    # can miss it.  [experiment].acknowledgements carries the
    # declared-experiment override; the refusal names it.
    if experiment.projection is not None:
        # The provenance guard (task #125) shares this seam and this
        # voice: a real case may not load under an install whose
        # metadata version contradicts the version its own source
        # declares -- that is the "plots say 1.6.2 while 1.8.7
        # executes" shape, and the refusal names both numbers, both
        # locations and the remedy.  A BORROWED number that agrees --
        # every worktree beside an editable install -- warns one line
        # naming the install the number came from, and never refuses;
        # a wheel has one version claim and passes in silence.
        # Idealized loads (no projection, no place, no clock) are not
        # a disagreement, exactly as they are not this nocturnal
        # guard's business.
        from woof.provenance_gate import require_version_identity
        require_version_identity(source)
        refusal = nocturnal_radiation_refusal(
            [dc.run for dc in experiment.domains],
            start_time=experiment.start_time,
            run_seconds=experiment.run_seconds,
            ref_lat=experiment.projection.ref_lat,
            ref_lon=experiment.projection.ref_lon,
            acknowledgements=experiment.acknowledgements)
        if refusal is not None:
            raise ValueError(f"experiment config {source}: {refusal}")
        # THE TWO RADIATION-ABSENCE GUARDS, in order of how much they
        # diagnose.  Both landed in this release, both read the same
        # selectors, and they overlap on exactly one class -- both
        # streams off under a land-surface scheme -- which is the class
        # BOTH of them are about.  Where they overlap, the radiation-off
        # guard speaks first, because "you have no radiation at all" is
        # the larger fact and its message carries the three remedies;
        # being told "your longwave is a declared constant" while
        # shortwave is also off would answer a smaller question than the
        # one the reader has.  A config in that overlap needs BOTH
        # declarations, which is a feature and not an accident: they are
        # two claims -- "nothing computes my sky" and "the number my land
        # surface integrates is one I typed" -- and a file that means
        # both says both.
        #
        # The radiation-OFF land-surface guard (2026-08-09), the other
        # half of the same hole.  The nocturnal guard tests sw > 0 and
        # lw == 0, so a suite with BOTH streams off walked past it while
        # initialize_physics attached no radiation adapter at all -- and
        # Noah/Noah-MP/RUC read fields["glw"] every surface step anyway.
        # Same door, same declaration idiom, no clock: a sky nothing
        # computes is wrong at noon as well as at midnight.  The
        # declared-selector set travels with it because radiation
        # defaults to OFF: a config that simply never named a selector is
        # in this class too, and gets told THAT rather than being quoted
        # two zeros it does not contain.
        refusal = radiation_off_land_surface_refusal(
            [dc.run for dc in experiment.domains],
            acknowledgements=experiment.acknowledgements,
            declared_selectors=declared_radiation)
        if refusal is not None:
            raise ValueError(f"experiment config {source}: {refusal}")
        # The constant-GLW guard (2026-08-09), the nocturnal guard's
        # companion and deliberately a SECOND question rather than a
        # clause of the first: that one asks whether the window is
        # survivable, this one asks whether the downward longwave
        # exists at all.  Kept separate because the nocturnal
        # acknowledgement was checked before any physics was inspected,
        # so a config carrying it never had its GLW source examined --
        # which is how ten shipped configs came to integrate a frozen
        # 300 W m-2.  Same load, same front doors, its own token.  It
        # refuses EXACTLY the set initialize_physics refuses, so a config
        # that passes here runs rather than dying mid-preparation.
        refusal = constant_longwave_refusal(
            [dc.run for dc in experiment.domains],
            acknowledgements=experiment.acknowledgements)
        if refusal is not None:
            raise ValueError(f"experiment config {source}: {refusal}")
    # Outside the projection gate above: inflow seeding is a nest
    # mechanism, and an idealized nested tree can declare it too.
    _refuse_inflow_seeding_under_a_pbl_off_parent(experiment, source)
    _advise_anisotropic_w_mixing(experiment, source)
    _assert_derived_copies(experiment, source)
    return _bind_physics_params(experiment, source)


def _bind_physics_params(experiment, source: str):
    """Refuse a [physics_params] set that acts on no domain, then bind the
    experiment's set (or its absence) to this process.

    Binding at the load every front door shares is what makes the set reach
    every route: the kernels read the process's set when they compile and
    the forecast's RUC tables read it when they are built.  One process runs
    one set, so a second experiment that declares another is refused here
    rather than running under the first one's constants.

    A set named by ``WOOF_PHYSICS_PARAMS`` (members sharing one prepared
    root) is attached to an experiment that carries no table, so the run's
    identity and receipts name it exactly as if the file had; a table that
    names a different set refuses.
    """
    from dataclasses import replace as _replace
    from woof.physics_params import (
        PhysicsParamsError, check_schemes, declare, environment_set)
    pset = getattr(experiment, "physics_params", None)
    try:
        from_env = environment_set()
        if from_env is not None:
            if pset is None:
                experiment = _replace(experiment, physics_params=from_env)
                pset = from_env
            elif pset.sha256() != from_env.sha256():
                raise PhysicsParamsError(
                    f"{source} carries physics parameter set {pset.name!r} "
                    f"but WOOF_PHYSICS_PARAMS names {from_env.name!r}; two "
                    "sources for one run's constants cannot both be honoured")
        if pset is not None:
            check_schemes(pset, [dc.run for dc in experiment.domains],
                          source=source)
        declare(pset, source=source)
    except PhysicsParamsError as err:
        raise ValueError(str(err)) from None
    return experiment


class ExposedMixing(NamedTuple):
    """What :func:`anisotropic_w_mixing_exposure` hands its callers.

    ``dz_max`` -- deepest base-state layer of the vertical coordinate, in
    metres, or ``inf`` when the coordinate cannot be resolved at all.
    ``inf`` is a sentinel a caller must route on, not a depth to compute
    with; see :func:`anisotropic_w_mixing_exposure` for what it selects.
    ``domains`` -- the DomainConfigs on the exposed path, possibly empty.
    ``ladder`` -- how ``dz_max`` was obtained, EMPTY when the config
    wrote its own eta interfaces and non-empty when they were resolved
    for it; callers put it in the advisory so a derived number never
    reads like a declared one.
    """

    dz_max: float
    domains: tuple
    ladder: str


def anisotropic_w_mixing_exposure(experiment: ExperimentConfig
                                  ) -> ExposedMixing:
    """Layer depths and the domains that would be judged against them.

    This is the only place both halves of the question are known -- the
    shared vertical coordinate owns the layer depths, each domain owns
    its horizontal spacing and its mixing selectors -- so it is where the
    layer depths meet the selectors.  It computes no verdict: callers
    hand the result to :func:`woof.config.anisotropic_w_mixing_advice`,
    which owns the criterion and the wording.

    A COORDINATE WITH NO EXPLICIT ETA INTERFACES IS RESOLVED, NOT
    SKIPPED.  Skipping it was a hole: ``km_opt = 3`` with
    ``mix_isotropic = 0`` at ``dx = 100`` m and no ``eta_levels`` is the
    exact configuration the criterion exists for, and until 2026-08-09 it
    came back empty and every door went quiet -- silence read as a pass
    by anything downstream, which is the opposite of what the absence of
    a number means.  When the ladder is not written down it is rebuilt
    the way the model rebuilds it: the uniform interfaces
    :func:`woof.core.grid.make_vertical_coord` produces from ``nz``
    (every idealized/legacy route in the tree calls it with no
    ``stretch``), with the model top expressed as a pressure by
    :func:`woof.core.grid.analytic_base_pressure`, whose base state is
    the one :func:`~woof.core.grid.base_layer_depths` already inverts --
    so the resolved column spans exactly ``[0, ztop]``.

    ``nz``/``ztop`` come off the first domain because they cannot differ
    across domains: they are in ``_DOMAIN_VERTICAL_KEYS``, so a
    per-domain spelling is a load error and the vertical grid is shared
    by construction.

    ``dz_max = inf`` when even that fails (a model top above the analytic
    base state's ~24.6 km ceiling).  Infinity is a SENTINEL, not a large
    depth: no ratio is computed from it -- ``anisotropic_w_mixing_ratio``
    returns ``None`` for a non-finite depth, because there is no number
    and inventing one would put a fictitious depth into a stability
    criterion.  What it does instead is select a different sentence:
    :func:`woof.config.anisotropic_w_mixing_advice` recognises the
    non-finite depth on an otherwise-exposed domain and advises that the
    criterion CANNOT BE EVALUATED here, quoting ``ladder`` for the reason.
    That advisory reaches the same three doors the numeric one does --
    the load-time warning, ``woof check``, and the repository screen in
    ``tests/test_shipped_configs_mixing_stability.py``, which reports the
    domain with an infinite ratio exactly as its legacy branch does.

    Until 2026-08-09 that sentinel went nowhere: the ``None`` it produced
    was the same ``None`` a config OFF the exposed path produces, so all
    three doors read it as "not applicable" and went quiet.  Anything
    here that consumes the ratio alone must not read its absence as a
    pass -- ask for the advice.
    """

    exposed = tuple(d for d in experiment.domains
                    if d.run.km_opt in (2, 3) and d.run.mix_isotropic == 0)
    if not exposed:
        return ExposedMixing(0.0, (), "")
    dz_max, ladder = _mixing_layer_depths(experiment)
    return ExposedMixing(dz_max, exposed, ladder)


#: The warmest lowest-layer mean temperature the UCM plan check allows a
#: column.  A layer's depth scales with its temperature, so the first mass
#: level can sit higher than the base state puts it; the plan refuses only
#: when even a lowest layer this warm leaves it inside the canopy.
UCM_PLAN_WARMEST_LAYER_K = 320.0

#: What each URBPARM.TBL class is (URBPARM.TBL's header); an LCZ table
#: class k is Local Climate Zone k.
_URBPARM_CLASS_NAMES = {1: "low-density residential",
                        2: "high-density residential",
                        3: "commercial and industrial"}
_LCZ_CLASS_NAMES = {1: "compact high-rise", 2: "compact mid-rise",
                    3: "compact low-rise", 4: "open high-rise",
                    5: "open mid-rise", 6: "open low-rise",
                    7: "lightweight low-rise", 8: "large low-rise",
                    9: "sparsely built", 10: "heavy industry",
                    11: "bare rock or paved"}


def _first_layer_depth(eta2: float, vertical, base_temp: float) -> float:
    """The base-state depth (m) of a lowest layer ending at ``eta2``."""
    from woof.core.grid import base_layer_depths
    znw = np.asarray((1.0, float(eta2), 0.0), dtype=np.float64)
    return float(base_layer_depths(znw, vertical.hybrid_opt, vertical.etac,
                                   vertical.p_top, base_temp)[0])


def ucm_first_level_refusal(experiment: ExperimentConfig, *,
                            source: str = "experiment") -> str | None:
    """Refuse, before anything is fetched or prepared, a UCM run whose
    table classes' canopy reaches the first model level.

    THE BREAKAGE IT PREVENTS.  ``module_sf_urban.F:825`` is a WRF
    ``FATAL_ERROR``: on any urban cell where ``ZDC + Z0C + 2 m`` reaches
    the first mass level (``0.5 * DZ8W(1)``) the UCM's log profile is
    undefined and the model stops.  woof's UCM kernel carries the same
    stop, so such a run used to spend its fetch, its preparation and its
    card warm-up and then die at step 1 (the 750 m San Francisco and Los
    Angeles grids with Local Climate Zone land cover did exactly that on
    2026-09-30: their first level is about 25 m, and LCZ 1 and LCZ 4 need
    about 34 and 31 m).

    The canopy per class is the table row's (:func:`woof.core.
    urban_tables.ucm_canopy_heights`), and the first level's height is
    the configured eta ladder's lowest layer in WRF's base state
    (:func:`woof.core.grid.base_layer_depths`), let rise to what a
    :data:`UCM_PLAN_WARMEST_LAYER_K` lowest layer would give, so a grid
    that is only marginal is left to the exact step-1 check.  The plan
    cannot see which classes the land cover will put in the domain; it
    refuses on the classes the selected table carries.
    """
    vertical = experiment.vertical
    for domain in experiment.domains:
        run = domain.run
        if int(getattr(run, "sf_urban_physics", 0) or 0) != 1:
            continue
        eta = getattr(run, "eta_levels", None) or vertical.eta_levels
        if not eta:
            continue  # idealized: no real-data ladder; the step-1 check stands
        from woof.core.urban_tables import ucm_canopy_heights
        lcz = int(run.use_wudapt_lcz)
        base_temp = float(run.base_temp)
        za_base = 0.5 * _first_layer_depth(float(eta[1]), vertical, base_temp)
        za_warm = za_base * UCM_PLAN_WARMEST_LAYER_K / base_temp
        table_options = {
            "generation": getattr(run, "urban_ucm_generation", "wrf_471"),
            "tbl_dir": getattr(run, "urban_table_dir", "") or None,
        }
        heights = ucm_canopy_heights(lcz, **table_options)
        over = {k: h for k, h in heights.items() if h >= za_warm}
        if not over:
            continue
        names = _LCZ_CLASS_NAMES if lcz else _URBPARM_CLASS_NAMES
        label = "LCZ " if lcz else "URBPARM class "
        listed = ", ".join(f"{label}{k} ({names.get(k, 'urban')}, "
                           f"{h:.1f} m)" for k, h in sorted(over.items()))
        tallest = max(heights.values())
        # A first level above the tallest canopy even in a column 10
        # percent colder than the base state: the eta that gives it.
        target = 2.0 * tallest / 0.9
        lo, hi = 0.5, float(eta[0])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _first_layer_depth(mid, vertical, base_temp) >= target:
                lo = mid
            else:
                hi = mid
        tallest_urbparm = max(ucm_canopy_heights(0, **table_options).values())
        remedies = [
            (f"land cover whose urban classes stay below the level: NLCD "
             f"with use_wudapt_lcz = 0 (the three URBPARM classes, the "
             f"tallest canopy {tallest_urbparm:.1f} m)") if lcz else None,
            (f"a first model level above the canopy: a lowest layer at "
             f"least {target:.0f} m deep, which on this grid is "
             f"eta_levels[1] = {lo:.4f} or lower (now {float(eta[1]):.4f})"),
            ("sf_urban_physics = 2 or 3 (BEP, BEP+BEM), which spread the "
             "buildings over the model levels and have no such limit"),
        ]
        remedy = "; ".join(r for r in remedies if r)
        return (
            f"{source}: d{domain.grid_id:02d} runs the single-layer urban "
            f"canopy model (sf_urban_physics = 1) with use_wudapt_lcz = "
            f"{lcz}, and WRF stops the model at its first step on any urban "
            f"cell whose canopy, ZDC + Z0C + 2 m, reaches the first model "
            f"level (module_sf_urban.F:825, a FATAL_ERROR).  On this "
            f"vertical grid the first level is {za_base:.1f} m above the "
            f"ground in the base state and no higher than {za_warm:.1f} m "
            f"even under a {UCM_PLAN_WARMEST_LAYER_K:.0f} K lowest layer, "
            f"and the table's {listed} reach it.  The land cover is not "
            f"read until preparation, so the plan refuses on the table's "
            f"classes rather than spend the fetch, the preparation and the "
            f"card on a run that stops at step 1.  Remedies: {remedy}.")
    return None


def _mixing_layer_depths(experiment: ExperimentConfig) -> tuple[float, str]:
    """``(dz_max, ladder)`` -- THE layer-depth resolution, in one place.

    Extracted from :func:`anisotropic_w_mixing_exposure` so the
    auto-switch resolver (:func:`resolve_auto_mix_isotropic`) and the
    switched-domain reporter (:func:`auto_selected_isotropic_mixing`)
    read exactly the depths the advisory reads -- one implementation of
    the criterion's inputs, never a copy.  The ``inf`` sentinel and the
    ladder provenance keep their exposure-era meanings.
    """

    from woof.core.grid import (analytic_base_pressure, base_layer_depths,
                                 make_vertical_coord)

    vertical = experiment.vertical
    if vertical.eta_levels:
        znw = vertical.eta_levels
        p_top = vertical.p_top
        ladder = ""
    else:
        run = experiment.domains[0].run
        p_top = analytic_base_pressure(run.ztop)
        if p_top is None:
            return (
                math.inf,
                f"no eta_levels, and ztop = {float(run.ztop):g} m is above "
                f"the analytic base state's representable ceiling, so no "
                f"layer depth can be resolved for this grid at all")
        znw = make_vertical_coord(int(run.nz)).znw
        ladder = (f"resolved ladder -- this config declares no eta_levels, "
                  f"so the depths are the uniform nz = {int(run.nz)} "
                  f"interfaces the model builds, spanning ztop = "
                  f"{float(run.ztop):g} m")

    dz_max = float(base_layer_depths(
        znw, vertical.hybrid_opt, vertical.etac, p_top).max())
    return dz_max, ladder


def resolve_auto_mix_isotropic(experiment: ExperimentConfig, auto_ids,
                               source: str) -> ExperimentConfig:
    """Apply the mixing-length auto-switch (project ruling, 2026-08-16) to one tree.

    ``auto_ids`` are the grid_ids whose config left ``mix_isotropic``
    unset (or wrote ``"auto"``).  Each such domain that selects the
    per-axis path (``km_opt`` 2/3) is judged by the one criterion
    implementation -- :func:`woof.config.anisotropic_w_mixing_ratio`
    over :func:`_mixing_layer_depths` -- and where the ratio exceeds
    :data:`woof.config.EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT` the domain
    runs ``mix_isotropic = 1``, announced by one loud
    :func:`woof.explain.warn` line per switched domain.

    Everything else is untouched, deliberately and observably:

    * a domain that satisfies the criterion keeps the WRF-default 0 and
      is byte-identical (restart identity included) to the same file
      with ``mix_isotropic = 0`` written out;
    * a WRITTEN 0 or 1 is never here (``auto_ids`` excludes it) -- an
      explicit danger-zone 0 keeps the anisotropic form and gets the
      forced-override advisory instead;
    * an unresolvable layer depth switches nothing: the criterion has
      no number there, and the no-number advisory (not this resolver)
      is what speaks.

    The re-validated replacement RunConfig keeps the p5t1 battery
    binding on what actually runs.
    """

    auto_ids = tuple(sorted({int(grid_id) for grid_id in auto_ids}))
    if not auto_ids:
        return experiment
    experiment = _dc_replace(experiment, auto_mix_isotropic=auto_ids)
    candidates = [dc for dc in experiment.domains
                  if dc.grid_id in set(auto_ids)
                  and dc.run.km_opt in (2, 3)
                  and dc.run.mix_isotropic == 0]
    if not candidates:
        return experiment
    dz_max, ladder = _mixing_layer_depths(experiment)
    switched: dict[int, float] = {}
    for dc in candidates:
        ratio = anisotropic_w_mixing_ratio(
            km_opt=dc.run.km_opt, mix_isotropic=0,
            mix_upper_bound=dc.run.mix_upper_bound,
            dx=dc.run.dx, dy=dc.run.dy, dz_max=dz_max)
        if (ratio is not None
                and ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT):
            switched[dc.grid_id] = float(ratio)
    if not switched:
        return experiment
    experiment = _dc_replace(experiment, domains=tuple(
        _dc_replace(dc, run=validate_run_config(
            _dc_replace(dc.run, mix_isotropic=1)))
        if dc.grid_id in switched else dc
        for dc in experiment.domains))
    for grid_id in sorted(switched):
        warn(auto_mix_isotropic_selection(
            where=f"domain grid_id = {grid_id} of {source}",
            ratio=switched[grid_id], ladder=ladder),
            why="With mix_isotropic = 0 the vertical exchange coefficient "
                "is built and capped on the LAYER DEPTH and then diffuses "
                "w over the HORIZONTAL spacing; past "
                "mix_upper_bound*(dz_max/dx)^2 = 1/4 that operator can "
                "invert a 2dx mode and past 1/2 grow one. The config "
                "declined to choose a length, so the model chooses the "
                "one that is stable on this grid.")
    return experiment


def auto_selected_isotropic_mixing(experiment: ExperimentConfig
                                   ) -> tuple[tuple[tuple[int, float], ...],
                                              str]:
    """``((grid_id, ratio), ...)`` the model switched, plus the ladder.

    The reporting face of :func:`resolve_auto_mix_isotropic`, for doors
    that see only the RESOLVED experiment (``woof check`` first among
    them): a domain is reported here exactly when its ``mix_isotropic =
    1`` was the model's choice, with the ratio the anisotropic path
    would have reached -- recomputed through the same
    :func:`_mixing_layer_depths` /
    :func:`woof.config.anisotropic_w_mixing_ratio` pair the resolver
    used, so the two can never tell different stories.  A WRITTEN 1 is
    never reported: that configuration is legitimate and quiet.
    """

    auto = set(getattr(experiment, "auto_mix_isotropic", ()) or ())
    selected = [dc for dc in experiment.domains
                if dc.grid_id in auto and dc.run.km_opt in (2, 3)
                and dc.run.mix_isotropic == 1]
    if not selected:
        return (), ""
    dz_max, ladder = _mixing_layer_depths(experiment)
    out = []
    for dc in selected:
        ratio = anisotropic_w_mixing_ratio(
            km_opt=dc.run.km_opt, mix_isotropic=0,
            mix_upper_bound=dc.run.mix_upper_bound,
            dx=dc.run.dx, dy=dc.run.dy, dz_max=dz_max)
        if (ratio is not None
                and ratio > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT):
            out.append((dc.grid_id, float(ratio)))
    return tuple(out), ladder


def _refuse_inflow_seeding_under_a_pbl_off_parent(
        experiment: ExperimentConfig, source: str) -> None:
    """A seeded child needs a parent that DIAGNOSES a boundary layer.

    ``inflow_perturbation`` sets its vertical extent from the parent's
    ``pblh`` diagnostic, so a parent running ``bl_pbl_physics = 0`` has
    nothing to define it with and
    :func:`woof.core.inflow_perturbation.build_inflow_perturbation`
    refuses.  That refusal is correct, it is ratified as G4 in
    ``docs/superpowers/specs/P6-LES-DECISIONS-RATIFIED-2026-08-05.md``,
    and it STAYS where it is -- a spawned or relocated nest is built
    outside the config path and must still hit it.

    What it could not do is arrive in time.  It lives at ``NestCoupler``
    construction, downstream of a fetch, two preparations and a whole
    prepared tree, so G4's own record is a four-domain LES tree admitted
    to the card and killed twelve seconds later on a fact that was
    legible in the TOML.  And ``configs/`` shipped a file in exactly
    that state, because the case it belongs to was parked before G4's
    sweep reached it.  Refused HERE, at the one load every front door
    shares (run/go/check, both prepared runners, the LES route in
    ``hrrr_hierarchy_direct``, ``core.preflight``, the domain wizard,
    the namelist importer), so a door nobody remembered to wire still
    inherits it -- the same seam and the same voice as the nocturnal,
    radiation-absence and follow-cadence refusals above.

    One copy of the predicate: the sentence comes from
    :func:`woof.core.inflow_perturbation.parent_pbl_refusal`, which the
    construction-time guard reads too.
    """

    from woof.core.inflow_perturbation import parent_pbl_refusal

    by_id = {int(dc.grid_id): dc for dc in experiment.domains}
    for dc in experiment.domains:
        if not getattr(dc.run, "inflow_perturbation", False):
            continue
        # The root carries parent_id == 0, which is not a grid id, so a
        # rootless lookup answers None -- the same "no PBLH to read"
        # case, and parent_pbl_refusal says so in the same words.
        parent = by_id.get(int(dc.parent_id))
        refusal = parent_pbl_refusal(
            None if parent is None else parent.run.bl_pbl_physics)
        if refusal is None:
            continue
        parent_named = (
            f"its parent ([[domain]] grid_id = {int(parent.grid_id)}) runs "
            f"bl_pbl_physics = {int(parent.run.bl_pbl_physics)}"
            if parent is not None else
            "it declares no parent in this tree (parent_id = "
            f"{int(dc.parent_id)})")
        raise ValueError(layered(
            f"experiment config {source}: [[domain]] grid_id = "
            f"{int(dc.grid_id)} sets inflow_perturbation = true, but "
            f"{parent_named}.  {refusal}.  REMEDY: "
            f"inflow_perturbation = false on grid_id = {int(dc.grid_id)}.",
            "A PBL-off parent is itself an LES domain, and its RESOLVED "
            "eddies already ARE this child's inflow turbulence -- "
            "seeding here would double-count them.  The seeding belongs "
            "on the FIRST domain whose parent parameterizes turbulence "
            "(bl_pbl_physics != 0), which is where G4 rules it ON "
            "(docs/superpowers/specs/"
            "P6-LES-DECISIONS-RATIFIED-2026-08-05.md).  Refused at the "
            "config load rather than at the first nest bind because the "
            "same pairing used to be admitted, prepared and placed on "
            "the card, and then killed the run twelve seconds in at "
            "NestCoupler construction -- which is G4's own record."))


def _advise_anisotropic_w_mixing(experiment: ExperimentConfig,
                                 source: str) -> None:
    """Advisory pass: per-axis mixing lengths against the explicit limit.

    Warn, never block: see
    :func:`woof.config.warn_anisotropic_w_mixing` for the ruling and
    the numbers behind it.  ``woof check`` repeats the same sentence in
    its advisory list, because this line is emitted at config load and
    the report is where a reader is looking.
    """

    dz_max, exposed, ladder = anisotropic_w_mixing_exposure(experiment)
    auto = set(getattr(experiment, "auto_mix_isotropic", ()) or ())
    for domain in exposed:
        # Post-resolution, an exposed domain either WROTE its 0 (the
        # forced-override state, said as such) or left it unset on a
        # depth the criterion could not evaluate (the no-number
        # advisory, which the forced flag does not touch).
        warn_anisotropic_w_mixing(
            where=f"domain grid_id = {domain.grid_id} of {source}",
            km_opt=domain.run.km_opt,
            mix_isotropic=domain.run.mix_isotropic,
            mix_upper_bound=domain.run.mix_upper_bound,
            dx=domain.run.dx, dy=domain.run.dy, dz_max=dz_max,
            ladder=ladder, forced=domain.grid_id not in auto)


def _assert_derived_copies(experiment: ExperimentConfig,
                           source: str) -> None:
    """F14 timing authority + F1 vertical single-sourcing assertions.

    ``ExperimentConfig.run_seconds``/``restart_interval_s`` and each
    ``DomainConfig.history_interval_s`` are the ONLY authoritative
    timing values; the embedded RunConfig copies are derived and must
    match exactly.  Likewise ``hybrid_opt``/``etac`` on every RunConfig
    are compatibility copies of ``experiment.vertical``.  By
    construction these hold; the assertion guards future drift loudly.
    """
    for dc in experiment.domains:
        for label, copy, authority in (
                ("run_seconds", dc.run.run_seconds,
                 experiment.run_seconds),
                ("output_interval_s", dc.run.output_interval_s,
                 dc.history_interval_s),
                ("restart_interval_s", dc.run.restart_interval_s,
                 experiment.restart_interval_s),
                ("hybrid_opt", dc.run.hybrid_opt,
                 experiment.vertical.hybrid_opt),
                ("etac", dc.run.etac, experiment.vertical.etac)):
            if copy != authority:
                raise ValueError(
                    f"derived RunConfig copy diverged on grid_id = "
                    f"{dc.grid_id} of {source}: run.{label} = {copy!r} "
                    f"but the authoritative experiment value is "
                    f"{authority!r} (F14 timing authority / F1 vertical "
                    "single-sourcing).")


# ---------------------------------------------------------------------------
# Dormant nests: the active-tree views the spawn runner consumes
# ---------------------------------------------------------------------------

def dormant_domain_ids(exp: ExperimentConfig) -> tuple[int, ...]:
    """grid_ids of every declared spawn-triggered (dormant) nest."""
    return tuple(dc.grid_id for dc in exp.domains
                 if getattr(dc, "spawn", None) is not None)


def validate_spawn_placement(exp: ExperimentConfig, grid_id: int,
                             i_parent_start: int,
                             j_parent_start: int) -> None:
    """Re-run the loader's placement admission for a trigger-chosen spot.

    A spawned placement bypasses :func:`build_experiment` (the tree was
    admitted with the placeholder placement), so the SAME parent-row
    clearance rule is applied here, with the same numbers and the same
    sentence shape.  ``register_nest``'s +-2 SINT stencil refusal still
    has the final word at materialization.
    """
    dc = exp.domain(grid_id)
    parent = exp.domain(dc.parent_id)
    need = int(exp.spec_bdy_width) + int(exp.blend_width)
    for axis, size, start, parent_size in (
            ("west-east", dc.run.nx, int(i_parent_start), parent.run.nx),
            ("south-north", dc.run.ny, int(j_parent_start), parent.run.ny)):
        if start < 1:
            raise ValueError(
                f"spawn placement {start} on the {axis} axis of "
                f"d{grid_id:02d} is not 1-based WRF namelist semantics")
        span = size // dc.parent_grid_ratio
        near = start - 1
        far = parent_size - (start + span - 1)
        for side, clearance in ((f"{axis} low", near),
                                (f"{axis} high", far)):
            if clearance < need:
                raise ValueError(
                    f"spawned placement ({i_parent_start}, "
                    f"{j_parent_start}) for d{grid_id:02d} violates the "
                    f"parent-row clearance rule: {side} clearance is "
                    f"{clearance} parent rows but spec_bdy_width + "
                    f"blend_width = {exp.spec_bdy_width} + "
                    f"{exp.blend_width} = {need} rows are required.")


def active_experiment(exp: ExperimentConfig,
                      spawned: dict | None = None, *,
                      retired=(), birth_times=None) -> ExperimentConfig:
    """The tree the executor integrates: dormant nests out, spawned in.

    ``spawned`` maps grid_id -> ``(i_parent_start, j_parent_start)`` for
    every dormant nest whose trigger has fired; those domains join the
    active tree AT the fired placement (their ``spawn`` declaration is
    KEPT, so the activated experiment's identity binds the fact and the
    terms of the spawn).  Dormant nests not in ``spawned`` are removed
    -- they cost their memory-plan reservation and zero compute, which
    is the declared contract.  With no dormant nests this is the
    identity, byte-for-byte the same object.

    This is the schedule-surgery seam the leg runner consumes: legs
    before a trigger integrate ``active_experiment(exp)``, and the leg
    after a fire integrates ``active_experiment(exp, {gid: (i, j)})``.

    ``birth_times`` maps grid_id -> seconds since the experiment start at
    which that slot's CURRENT episode began, and it is what gives the
    newborn an ACTIVATION EPOCH.  A late-activating nest is not a new
    shape -- a domain may declare ``start_time`` and join the run hours
    in -- and every consumer of that epoch reads it off the resolved
    clock's ``start_ticks``: the history alarm, the REFL_10CM handoff,
    the schedule's activation bookkeeping, and the step counter that
    decides whether a domain's first step runs radiation.  A spawn used
    to record none of it, so the newborn resolved ``start_ticks = 0``
    and claimed to have been running since t = 0.  Two failures came
    straight out of that: its first frame consumed a REFL_10CM stash no
    step of that domain had produced, and its first step was numbered as
    though it were mid-run, so radiation was not due and the land
    surface then consumed a GLW buffer nothing had written.  Recording
    the epoch here rather than at the clock-minting sites is deliberate:
    every leg re-mints its clocks from THIS view, so an epoch stored
    anywhere else would be correct for one leg and lost at the next.
    """
    spawned = {} if spawned is None else dict(spawned)
    birth_times = {} if birth_times is None else {
        int(gid): float(t) for gid, t in birth_times.items()}
    retired = {int(gid) for gid in retired}
    dormant = set(dormant_domain_ids(exp))
    bad_retired = sorted(retired - dormant)
    if bad_retired:
        raise ValueError(
            f"retired grid_id(s) {bad_retired} are not declared dormant "
            "spawn slots; lifecycle retirement only applies to spawned episodes")
    # A retired parent cannot leave an integrated orphan.  Expand the seeds
    # over the DECLARED tree before selecting the next leg's domain set.
    blocked = set(retired)
    changed = True
    while changed:
        changed = False
        for dc in exp.domains:
            if int(dc.parent_id) in blocked and int(dc.grid_id) not in blocked:
                blocked.add(int(dc.grid_id)); changed = True
    unknown = sorted(set(spawned) - dormant)
    if unknown:
        raise ValueError(
            f"spawned grid_id(s) {unknown} are not declared dormant "
            f"nests of experiment {exp.name!r} (dormant: "
            f"{sorted(dormant)}); only a declared spawn can activate.")
    if not dormant:
        return exp
    for grid_id, position in spawned.items():
        i_start, j_start = (int(position[0]), int(position[1]))
        validate_spawn_placement(exp, grid_id, i_start, j_start)
    domains = []
    for dc in exp.domains:
        if dc.grid_id in blocked:
            continue
        if dc.grid_id in spawned:
            i_start, j_start = spawned[dc.grid_id]
            fields = {"i_parent_start": int(i_start),
                      "j_parent_start": int(j_start)}
            birth = birth_times.get(int(dc.grid_id))
            if birth is not None:
                # Whole microseconds: the spawn instant is already
                # validated as a whole number of PARENT steps, and the
                # datetime lattice this lands on is microseconds.
                fields["start_time"] = exp.start_time + timedelta(
                    microseconds=int(round(birth * 1_000_000)))
            domains.append(_dc_replace(dc, **fields))
        elif dc.grid_id in dormant:
            continue
        else:
            domains.append(dc)
    return _dc_replace(exp, domains=tuple(domains))


def pre_spawn_experiment(exp: ExperimentConfig) -> ExperimentConfig:
    """The startup tree: every dormant nest removed, nothing spawned."""
    return active_experiment(exp, None)


def delayed_domain_ids(exp: ExperimentConfig) -> tuple[int, ...]:
    """grid_ids of every domain that activates later than the experiment."""
    return tuple(dc.grid_id for dc in exp.domains
                 if exp.domain_start_time(dc.grid_id) != exp.start_time)


def refuse_delayed_activation(exp: ExperimentConfig, route: str) -> None:
    """Fail loud where a delayed nest start has no activation machinery.

    Delayed activation is a supported shape on the experiment-tree route
    (``woof run``): that route holds an activation context, and the
    executor's ``on_domain_start`` initializes the child from the
    analysis at its activation time before its first step.  A route that
    builds its domains from prepared caches holds neither -- every node
    is started at the experiment's t = 0 and no callback exists to bring
    one to life later -- so a delayed child there stays at tick 0 while
    the rest of the tree advances, and the executor's tick-exact sync
    check raises ``tick-exact sync violated after period 0: grid_id=2 at
    0 ticks, boundary N``.  A traceback with no remedy in it is what this
    refusal replaces.
    """
    delayed = delayed_domain_ids(exp)
    if not delayed:
        return
    named = ", ".join(
        f"d{gid:02d} at {exp.domain_start_time(gid).isoformat()}"
        for gid in delayed)
    raise ValueError(layered(
        f"the {route} route does not implement delayed nest activation; "
        f"{named} start(s) later than [experiment].start_time = "
        f"{exp.start_time.isoformat()}, and this route has no way to "
        "bring a domain to life mid-run. Start every [[domain]] at "
        "[experiment].start_time here, or run the config through "
        "`woof run`, which activates delayed nests.",
        "A delayed nest is initialized at its activation epoch from the "
        "analysis valid at that instant, by "
        "woof.core.model.execute_experiment's on_domain_start using the "
        "activation context woof.core.model.build_experiment attaches "
        "(the input catalog, the case data and the radiation "
        "workspace).  This route restores each domain from a prepared "
        "cache instead: there is no catalog to re-ingest from and no "
        "activation callback, so the domain would sit at tick 0 while "
        "its parent advanced and the run would die at the first period "
        "boundary with 'tick-exact sync violated'."))


def refuse_unrouted_spawn(exp: ExperimentConfig, route: str) -> None:
    """Fail loud where a spawn declaration would otherwise be dropped.

    Same governance as [perturbation]: honored or refused, never
    ignored.  A route that neither reserves the dormant nest nor
    evaluates its trigger calls this immediately after loading the
    experiment, so the user learns at admission -- not after a run that
    silently integrated without the nest they declared.
    """
    dormant = dormant_domain_ids(exp)
    if not dormant:
        return
    named = ", ".join(f"d{gid:02d}" for gid in dormant)
    raise ValueError(layered(
        f"the {route} route does not implement spawn-triggered nests; "
        f"{named} declare(s) spawn tables that this route would neither "
        "reserve nor watch, so the run would quietly integrate without "
        "the nest(s) you declared.",
        "Dormant nests are reserved and spawned on the experiment tree "
        "path (woof.core.model.build_experiment reserves; the "
        "leg-boundary spawn runner activates through "
        "woof.experiment.active_experiment and "
        "woof.ingest.nest_spawn_init)."))


def _public_config_value(value):
    """Like asdict, without traversing derived execution contexts."""
    from copy import deepcopy
    from dataclasses import fields, is_dataclass
    from woof.core.streaming import StreamingOptions
    from woof.core.storm_tracking import FollowConfig, ATTRIBUTE_KEYS
    if isinstance(value, DeviceOptions):
        return value.to_mapping()
    if isinstance(value, StreamingOptions):
        return value.to_mapping()
    if is_dataclass(value):
        from woof.config import CHEM_RUN_FIELDS
        inert_chem = (frozenset(CHEM_RUN_FIELDS)
                      if isinstance(value, RunConfig) and not value.chem_sets
                      else frozenset())
        if isinstance(value, RunConfig):
            from woof.config import inert_fire_fields
            inert_chem = inert_chem | frozenset(inert_fire_fields(value))
        document = {item.name: _public_config_value(getattr(value, item.name))
                    for item in fields(value)
                    if item.name not in inert_chem
                    and not (isinstance(value, FollowConfig)
                            and ((value.field != "attribute" and item.name in ATTRIBUTE_KEYS)
                                 or (value.field != "pressure" and item.name in {
                                     "pressure_center", "fallback_level_hpa", "fallback_deficit_m"})))}
        from woof.io.history_selection import HistorySelection
        if isinstance(value, HistorySelection):
            if value.formats == ("netcdf",):
                document.pop("formats", None)
            if value.icechunk_store is None:
                document.pop("icechunk_store", None)
        if (isinstance(value, HistorySelection)
                and not document.get("full_history_interval_s")):
            # The between-hours subset (S4, HRRR-shaped sub-hourly history)
            # stays absent while no full cadence is set: every frame is
            # written with the selection, as before, and a default document
            # keeps its pre-feature bytes.
            for name in ("full_history_interval_s", "between_preset",
                         "between_history_add"):
                document.pop(name, None)
        if isinstance(value, DomainConfig):
            # A diagnostic filename is not an output control. Publishing
            # the same quick-look selection beside a prepared bundle must
            # preserve its identity when the authority moves there.
            if document.get("quicklook_output") is not None:
                document["quicklook_output"].pop("source", None)
            # The quick-look stream (run controls, 2.9.7) stays absent while a
            # domain has none, so a default document keeps its pre-feature
            # bytes (and every experiment fingerprint read from it).
            for name in ("full_history_interval_s", "quicklook_output"):
                if document.get(name) is None:
                    document.pop(name, None)
        run_document = (document if isinstance(value, RunConfig) else
                        document.get("run") if isinstance(value, DomainConfig) else None)
        if isinstance(run_document, dict):
            from woof.config import drop_inert_g3_fields
            drop_inert_g3_fields(run_document)
            from woof.checkpoint_identity import drop_default_cumulus_selectors
            drop_default_cumulus_selectors(run_document)
            # The generic forms existed before these selectors. Keep
            # their public bytes; an explicitly moved form remains bound.
            for name, default in (("ruc_irrigation", "wrf_461"),
                                  ("ruc_qvg_cold_start", "wrf"),
                                  ("ruc_2m_diagnostic", "flux"),
                                  ("ruc_snow", "wrf_461"),
                                  ("swint_opt", 0), ("aer_opt", 0),
                                  ("alb_sol", 0),
                                  ("thompson_version", "wrf_461"),
                                  ("thompson_table_dir", None),
                                  ("thompson_fork_snow_fall", "blend"),
                                  ("rrtmg_cloud_optics_form", "wrf_461"),
                                  ("rrtmg_aer3_type_source", "rural"),
                                  ("rrtmg_smoke_manifest", ""),
                                  ("surface_energy_diag", False),
                                  ("cycling", False),
                                  ("bl_mynn_mixaerosols", 0),
                                  ("bl_mynn_mixnumcon", 0),
                                  ("tempo_ml_for_bl_nc", False),
                                  ("tempo_surface_emission_before", False),
                                  # 2.9.0 (golden/wrf-fixes).  WRF's one
                                  # scalar advection operator and the WRF
                                  # 4.7.1 urban canopy at their defaults.
                                  ("scalar_adv_opt", 1),
                                  ("urban_ucm_generation", "wrf_471"),
                                  ("urban_treeoption", -1),
                                  ("urban_ttscheme", -1),
                                  ("urban_fvg", -1.0),
                                  ("urban_table_dir", ""),
                                  # The four WRF 4.7/4.8 fix selectors: a
                                  # document written before them ran the
                                  # PREVIOUS generation, so that value is
                                  # the absent one (as in a checkpoint
                                  # header, restart.ECHO_ABSENT_RUN_VALUES)
                                  # and the corrected default is written.
                                  ("solar_generation", "wrf_461"),
                                  ("ysu_version", "wrf_461"),
                                  ("noahmp_snow_age", "wrf_461"),
                                  ("urban_water_balance", "wrf_471")):
                if run_document.get(name, default) == default:
                    run_document.pop(name, None)
            # ... and the scheme-scoped ones reach nothing while their
            # scheme is off, so they leave such a document whatever they say.
            if run_document.get("bl_pbl_physics", 0) != 1:
                run_document.pop("ysu_version", None)
            if run_document.get("sf_surface_physics", 0) != 4:
                run_document.pop("noahmp_snow_age", None)
            if int(run_document.get("sf_urban_physics", 0) or 0) != 1:
                run_document.pop("urban_water_balance", None)
            # The ocean mixed layer (2.9.1): off, its selector and settings
            # leave the document, which keeps the bytes it had before them.
            if int(run_document.get("sf_ocean_physics", 0) or 0) == 0:
                from woof.surface_ocean import ocean_namelist_defaults
                for name, default in (("sf_ocean_physics", 0),
                                      *ocean_namelist_defaults().items()):
                    if run_document.get(name, default) == default:
                        run_document.pop(name, None)
            # The correct-mode keys (woof/correct_mode.py), by the
            # checkpoint echo's rule: written only where on AND read by a
            # selected scheme, so a default document without WSM6 or New
            # Tiedtke keeps its bytes under the 2.9.0 correct-mode default,
            # and an exact-mode document keeps the bytes it had before them.
            from woof.correct_mode import (CORRECTION_KEYS,
                                            OUTPUT_ONLY_KEYS, key_in_scope)
            for name in CORRECTION_KEYS:
                if (name in OUTPUT_ONLY_KEYS
                        or not run_document.get(name, False)
                        or not key_in_scope(name, run_document)):
                    run_document.pop(name, None)
            # Lightning and aer_opt = 2 controls (golden/ports-rad): absent
            # while unread, so default public documents keep their bytes.
            from woof.config import drop_inactive_radiation_port_keys
            drop_inactive_radiation_port_keys(run_document)
            # The 2.9.4 capabilities campaign, by the checkpoint echo's rule
            # (woof.checkpoint_identity.drop_capability_campaign_keys), plus
            # lane/cap-eclipse's default-on shadow (absent reads 1) and
            # lane/cap-slab-ocean's PWP controls at their defaults while the
            # ocean is off.
            from woof.checkpoint_identity import drop_capability_campaign_keys
            drop_capability_campaign_keys(run_document)
            from woof.checkpoint_identity import drop_default_output_controls
            drop_default_output_controls(run_document)
            # Fitch wind farms (labs/windfarm, 2.9.11): farm physics off leaves
            # the document as it was before the keys, by the checkpoint
            # echo's rule (woof.wind_table.drop_inert_config).
            from woof.wind_table import drop_inert_config as drop_inert_wind_farms
            drop_inert_wind_farms(run_document)
            if run_document.get("ra_sw_eclipse", 1) == 1:
                run_document.pop("ra_sw_eclipse", None)
            # tc/nest's fixed-frame archive (2.9.7) is read only on a nested
            # domain, and off is the historical counter every earlier
            # document ran: absent there, by the checkpoint echo's rule
            # (woof.io.restart._drop_inert_nest_archive).
            if (not run_document.get("nested")
                    or not run_document.get("nest_accumulation_archive", False)):
                run_document.pop("nest_accumulation_archive", None)
            if int(run_document.get("sf_ocean_physics", 0) or 0) == 0:
                for name, default in (("ocean_levels", 30),
                                      ("ocean_analysis", ""),
                                      ("ocean_analysis_source", "rtofs"),
                                      ("ocean_mld_climatology", "woa18"),
                                      ("omdt", 1.0)):
                    if run_document.get(name, default) == default:
                        run_document.pop(name, None)
                for name in ("ocean_z", "ocean_t", "ocean_s"):
                    if not run_document.get(name, ()):
                        run_document.pop(name, None)
        return document
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_public_config_value(item) for item in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_public_config_value(item) for item in value)
    if isinstance(value, dict):
        return type(value)((_public_config_value(key), _public_config_value(item))
                           for key, item in value.items())
    return deepcopy(value)


def domain_config_document(domain: DomainConfig) -> dict[str, object]:
    """Resolved domain fields with public execution controls only."""
    return _public_config_value(domain)


def _nudging_from_tables(raw, domains):
    if "nudging" in raw and "spectral_nudging" in raw:
        raise ValueError("choose one [nudging] table; [spectral_nudging] is its legacy alias")
    row = raw.get("nudging", raw.get("spectral_nudging"))
    if row is None:
        return None
    from woof.spectral_nudging.config import from_mapping
    return from_mapping(row, tuple(dc.grid_id for dc in domains))


def experiment_config_document(exp: ExperimentConfig) -> dict[str, object]:
    """Resolved experiment fields, preserving the domain document contract."""
    document = _public_config_value(exp)
    if getattr(exp, 'nudging', None) is None:
        document.pop("nudging", None)
    # A newly introduced execution control stays absent when off, so default
    # public snapshots and plans retain their pre-feature bytes.
    if exp.physics_params is None:
        document.pop("physics_params", None)
    else:
        from woof.physics_params import document as parameter_document
        document["physics_params"] = parameter_document(exp.physics_params)
    if not exp.devices.enabled:
        document.pop("devices", None)
    if not exp.simulated_radar.enabled:
        document.pop("simulated_radar", None)
    heating = getattr(exp, "radar_heating", None)
    if heating is None or not heating.enabled:
        document.pop("radar_heating", None)
    else:
        document["radar_heating"] = heating.to_mapping()
    return document
