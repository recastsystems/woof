# WRF 4.6.1 moist-theta source audit

This inventories the moist prognostic paths and retained gaps. A one-hour
WRF 4.6.1 forecast with strict numerics matches all 190 common floating
fields at seven history times, with zero of 49,148,792 words differing.
The scope is 90 by 81 by 39 mass cells, 10 km, 30-second steps, native four
MPI processes (2 by 2), and one engine GPU process, using Thompson 8,
RRTMG 4/4, MYJ 2, Eta 2, Noah 2 and Tiedtke 6. The receipt is
`WRF-SETTINGS-2026-10-31/wrfs-theta/strict-whole-hour-receipt.json`.
The two-domain strict hour also matches every native floating field: 190 fields at seven history times in each domain, 120,975,463 words with zero differences. The child has 111 by 96 by 39 mass cells and a 10-second step; the parent has the single-domain geometry above. Receipts: `nest-feedback-d01-hour-compare.json` and `nest-feedback-d02-hour-compare.json` in the same evidence folder. Live surface, soil and radiation feedback follows native Registry operators on the moist route; the legacy dry route is preserved.

Engine default and default numerics remain dry and fast.

Engine baseline: `34951539d98`. Native source: WRF 4.6.1
`d66e442fccc04111067e29274c9f9eaccc3cef28`.

Theta is dry potential temperature. Theta_m is
`theta * (1 + (Rv/Rd) qv)`, where qv is water vapour mixing ratio and
Rv/Rd is the ratio of gas constants. EOS means equation of state.
A prognostic field is advanced in time.

## Source inventory

Command: `python3 tools/check_theta_m_inventory.py /path/to/WRF-v4.6.1`. `proof/theta-m-port/wrf461-sites.json` keeps the source line and file SHA-256 for each row. The checker rescans the named directories and verifies every occurrence and every listed file hash. Generated `.f90` duplicates are excluded. Counts here are from the verified commit, not assumed from the brief. Some occurrences are declarations, calls or comments rather than arithmetic branches.

The engine column describes the current implementation, not the baseline. Paths are relative to `gpuwm/`. Each row names its WRF source location and engine implementation or concrete missing path. Component checks are recorded in the lane receipt; this inventory does not establish native whole-forecast agreement.

| WRF site | Role | Engine implementation and remaining scope |
| --- | --- | --- |
| `dyn_em/module_big_step_utilities_em.F:958` (source) | Pass the thermal choice into calc_p_rho_phi. | core/diagnostics.py passes the selected-variable convention by choosing moist=False at use_theta_m=1; core/state.py carries the scalar selector. Declaration/call wiring is implemented and covered by the scoped whole-forecast comparison above. |
| `dyn_em/module_big_step_utilities_em.F:968` (source) | Declare the thermal choice argument. | core/diagnostics.py passes the selected-variable convention by choosing moist=False at use_theta_m=1; core/state.py carries the scalar selector. Declaration/call wiring is implemented and covered by the scoped whole-forecast comparison above. |
| `dyn_em/module_big_step_utilities_em.F:1061` (source) | Nonhydrostatic pressure: omit the vapour multiplier when t already carries theta_m. | core/diagnostics.py:update_diagnostics and ensemble/batch_diagnostics.py omit the second vapour multiplier when prognostic theta already includes vapour; existing dry arithmetic stays selected at 0. |
| `dyn_em/module_big_step_utilities_em.F:1114` (source) | Hydrostatic top-level inverse density: omit the extra vapour multiplier. | Retained refusal: namelist_import.py:DYNAMICS_REQUIRED_VALUES names the missing hydrostatic calc_p_rho_phi density and small_step_prep geopotential solves. The engine still advances nonhydrostatic vertical momentum; this hydrostatic branch is not implemented. |
| `dyn_em/module_big_step_utilities_em.F:1136` (source) | Hydrostatic lower-level inverse density: omit the extra vapour multiplier. | Retained refusal: namelist_import.py:DYNAMICS_REQUIRED_VALUES names the missing hydrostatic calc_p_rho_phi density and small_step_prep geopotential solves. The engine still advances nonhydrostatic vertical momentum; this hydrostatic branch is not implemented. |
| `dyn_em/module_big_step_utilities_em.F:4830` (source) | phy_prep: divide full theta_m by 1+Rv/Rd*qv to obtain dry theta for physics. | core/state.py:total_theta supplies dry theta to core/physics.py. core/theta_m_physics.py converts thermal and vapour tendencies; raw selected theta remains available as prognostic_theta. |
| `dyn_em/module_big_step_utilities_em.F:5323` (source) | Cumulus preparation: remove the vapour tendency contribution from the moist thermal forcing. | core/theta_m_physics.py:dry_cumulus_theta_tendency removes the vapour forcing term before physics consumes dry advective forcing. core/physics.py invokes this only for selected moist theta. |
| `dyn_em/module_big_step_utilities_em.F:5505` (source) | Microphysics preparation: save dry theta and old vapour. | core/theta_m_physics.py:save_vapour and scheme adapters preserve incoming qv and dry theta on device. Classic and aerosol-aware Thompson adapters and ensemble/batch_physics.py use the same convention. |
| `dyn_em/module_big_step_utilities_em.F:5735` (source) | Microphysics finish: use old/new vapour and limited dry heating to rebuild theta_m and h_diabatic. | core/theta_m_physics.py:finish_microphysics applies old/new vapour factors, limited dry heating and h_diabatic. Scheme adapters, including Thompson aerosol-aware, invoke it only at 1. Component replay and the scoped whole-forecast comparison above cover classic Thompson; aerosol-aware Thompson uses the same selected conversion. |
| `dyn_em/module_diffusion_em.F:4294` (source) | Specified surface heat flux: multiply by the vapour factor and add the 1.61*theta*qfx term. | core/kernels/smag2d.cu:wrf_smag_surface_heat_const selects the moist vapour multiplier and 1.61*dry_theta*qfx term with GPUWM_USE_THETA_M; core/dycore.py and ensemble/batch_mixing.py bind the added operands. |
| `dyn_em/module_diffusion_em.F:4311` (source) | Computed surface heat flux: multiply by the vapour factor and add the 1.61*theta*qfx term. | core/kernels/smag2d.cu:wrf_smag_surface_scalars applies the moist heat and vapour flux terms. core/dycore.py passes dry thermal physics and selected thermal diffusion consistently; batch surface ABI is bound where that graph has admitted flux inputs. |
| `dyn_em/module_diffusion_em.F:8116` (source) | km_opt=5 implicit mixing bottom row: convert heat and vapour surface fluxes. | Implemented in core/kernels/km5.cu implicit scalar solve: the real use_theta_m setting selects the moist bottom-row heat and vapor terms; geometry supplies dry theta. |
| `dyn_em/module_diffusion_em.F:8127` (source) | km_opt=5 implicit interior mixing: scale nonlocal flux by the surface vapour factor. | Implemented in core/kernels/km5.cu: the implicit nonlocal interior heat flux uses the surface vapor factor when use_theta_m=1. |
| `dyn_em/module_first_rk_step_part2.F:993` (source) | Convert accumulated dry physics tendencies to moist theta before diffusion | core/dycore.py converts a copied summed physics rtheta with the previous saved thermal value and current step-entry qv0 before diffusion. The thermal value is read before thp0 is overwritten, matching native t_1 ordering. core/theta_m_physics.py:convert_physics_theta_tendency includes the qv tendency. ensemble/batch_moist_dycore.py converts member-owned copies, preserving held dry producer rates. |
| `dyn_em/module_initialize_fire.F:1115` (source) | Convert ideal initial dry theta to moist theta | No direct claim for this native ideal-case initializer. core/state.py ideal constructors start with qv=0, where dry and moist theta coincide. ingest/init_perturbation.py converts dry thermal and vapour increments back together. A separately supplied moist ideal profile must initialize selected theta consistently; these native cases have no whole-run proof. |
| `dyn_em/module_initialize_heldsuarez.F:565` (source) | Convert ideal initial dry theta to moist theta | No direct claim for this native ideal-case initializer. core/state.py ideal constructors start with qv=0, where dry and moist theta coincide. ingest/init_perturbation.py converts dry thermal and vapour increments back together. A separately supplied moist ideal profile must initialize selected theta consistently; these native cases have no whole-run proof. |
| `dyn_em/module_initialize_ideal.F:1726` (source) | Convert ideal initial dry theta to moist theta | No direct claim for this native ideal-case initializer. core/state.py ideal constructors start with qv=0, where dry and moist theta coincide. ingest/init_perturbation.py converts dry thermal and vapour increments back together. A separately supplied moist ideal profile must initialize selected theta consistently; these native cases have no whole-run proof. |
| `dyn_em/module_initialize_real.F:4904` (source) | Convert t_2 from dry to moist after preserving dry history T; boundary construction consumes this t_2. | ingest/real.py:initialize_real converts dry theta to theta_m after dry initialization at option 1. ingest/wrfinput.py:restore_domain_state instead retains native THM directly for a matching USE_THETA_M=1 input. Dry T remains a diagnostic file field. |
| `dyn_em/module_initialize_real.F:5165` (source) | Hydrostatic initialization top-level inverse density uses selected theta. | ingest/real.py computes the initial balanced dry thermodynamic column before final conversion to theta_m. This is initialization balance, not hydrostatic time integration. The non_hydrostatic=false runtime refusal remains for the missing density/geopotential solves. |
| `dyn_em/module_initialize_real.F:5183` (source) | Hydrostatic initialization lower-level inverse density uses selected theta. | ingest/real.py computes the initial balanced dry thermodynamic column before final conversion to theta_m. This is initialization balance, not hydrostatic time integration. The non_hydrostatic=false runtime refusal remains for the missing density/geopotential solves. |
| `dyn_em/module_initialize_scm_xy.F:696` (source) | Convert ideal initial dry theta to moist theta | No direct claim for this native ideal-case initializer. core/state.py ideal constructors start with qv=0, where dry and moist theta coincide. ingest/init_perturbation.py converts dry thermal and vapour increments back together. A separately supplied moist ideal profile must initialize selected theta consistently; these native cases have no whole-run proof. |
| `dyn_em/module_initialize_tropical_cyclone.F:1144` (source) | Convert ideal initial dry theta to moist theta | No direct claim for this native ideal-case initializer. core/state.py ideal constructors start with qv=0, where dry and moist theta coincide. ingest/init_perturbation.py converts dry thermal and vapour increments back together. A separately supplied moist ideal profile must initialize selected theta consistently; these native cases have no whole-run proof. |
| `dyn_em/nest_init_utils.F:814` (source) | Pass the thermal choice to adjust_tempqv. | ingest/nest_init.py forwards cfg.use_theta_m to core/nest_interp.py:adjust_tempqv or verify/npref.py; ingest/wrfinput_nest_initial.py implements the same selected branch for file startup. |
| `dyn_em/nest_init_utils.F:837` (source) | Declare the thermal choice argument. | ingest/nest_init.py forwards cfg.use_theta_m to core/nest_interp.py:adjust_tempqv or verify/npref.py; ingest/wrfinput_nest_initial.py implements the same selected branch for file startup. |
| `dyn_em/nest_init_utils.F:852` (source) | Compute old relative humidity from dry physical temperature. | core/nest_interp.py and ingest/wrfinput_nest_initial.py:adjust_tempqv_fp32 divide incoming theta_m by 1+Rv/Rd*qv to compute dry physical temperature and old relative humidity. |
| `dyn_em/nest_init_utils.F:869` (source) | Recover dry theta for the terrain correction. | Terrain adjustment uses the recovered dry theta; ingest/nest_init.py first passes raw prognostic theta rather than the dry physics view at option 1. |
| `dyn_em/nest_init_utils.F:876` (source) | Rebuild moist theta using the old vapour before updating vapour to conserve relative humidity. | core/nest_interp.py and ingest/wrfinput_nest_initial.py rebuild theta_m using OLD qv before replacing qv to conserve relative humidity, preserving native operation order. The real two-domain strict hour matches all native floating fields as scoped above. |
| `dyn_em/solve_em.F:509` (comment) | Comment explaining the old-vapour seed needed before first physics. | core/dycore.py saves time-entry thp0 and qv0 before physics; the moist tendency conversion reads those old fields, including when a physics scheme mutates its current fields. This source row is a comment, not an additional arithmetic operation. |
| `dyn_em/solve_em.F:2965` (source) | Diagnose pressure/density after acoustic advancement using the selected variable. | core/dycore.py advances raw selected thp; core/diagnostics.py refreshes pressure and inverse density with the appropriate vapour multiplier. Ensemble diagnostic and acoustic callers use the same convention. The scoped single-domain and nested full-field proofs above pass. |
| `dyn_em/solve_em.F:4191` (source) | Diagnose pressure/density near the end of the model step using the selected variable. | core/dycore.py advances raw selected thp; core/diagnostics.py refreshes pressure and inverse density with the appropriate vapour multiplier. Ensemble diagnostic and acoustic callers use the same convention. The scoped single-domain and nested full-field proofs above pass. |
| `dyn_em/start_em.F:588` (source) | Reject old initial files without the selected thermal prognostic. | ingest/wrfinput.py:restore_domain_state requires USE_THETA_M=1 and a THM field when the requested selector is 1. Missing selected prognostic or incompatible file metadata is refused explicitly. |
| `dyn_em/start_em.F:821` (source) | WRFPLUS startup pressure: choose the vapour multiplier; outside this lane. | WRFPLUS tangent-linear/adjoint startup has no engine counterpart and is outside the authorized feature scope. Normal forward startup is covered at line 831. |
| `dyn_em/start_em.F:831` (source) | Normal startup pressure: use multiplier 1 for moist theta. | ingest/wrfinput_nest_initial.py:start_perturbation_pressure reads THM and uses vapour multiplier 1 at option 1. core/diagnostics.py applies that convention in the forward run. |
| `dyn_em/start_em.F:2140` (source) | Pass the thermal choice into traditional diagnostic fields. | core/state.py:total_theta and dry history T feed traditional diagnostics. core/gsd_diagnostics.py and output consumers recover dry temperature; the native diagnostics signature itself has no separate thermal arithmetic branch. |
| `dyn_em/start_em.F:2386` (source) | Hydrostatic moist-column top inverse density uses the selected theta. | Retained refusal: namelist_import.py:DYNAMICS_REQUIRED_VALUES names the missing hydrostatic calc_p_rho_phi density and small_step_prep geopotential solves. The engine still advances nonhydrostatic vertical momentum; this hydrostatic branch is not implemented. |
| `dyn_em/start_em.F:2411` (source) | Hydrostatic moist-column lower inverse density uses the selected theta. | Retained refusal: namelist_import.py:DYNAMICS_REQUIRED_VALUES names the missing hydrostatic calc_p_rho_phi density and small_step_prep geopotential solves. The engine still advances nonhydrostatic vertical momentum; this hydrostatic branch is not implemented. |
| `dyn_em/start_em.F:2463` (source) | Alternate hydrostatic top inverse density uses the selected theta. | Retained refusal: namelist_import.py:DYNAMICS_REQUIRED_VALUES names the missing hydrostatic calc_p_rho_phi density and small_step_prep geopotential solves. The engine still advances nonhydrostatic vertical momentum; this hydrostatic branch is not implemented. |
| `dyn_em/start_em.F:2480` (source) | Alternate hydrostatic lower inverse density uses the selected theta. | Retained refusal: namelist_import.py:DYNAMICS_REQUIRED_VALUES names the missing hydrostatic calc_p_rho_phi density and small_step_prep geopotential solves. The engine still advances nonhydrostatic vertical momentum; this hydrostatic branch is not implemented. |
| `share/dfi.F:714` (source) | First filter pass: convert theta_m to physical temperature for saturation. | Retained refusal: namelist_import.py:DFI_REFUSAL names the absent backward/forward digital-filter pass and its state combination. The engine has no DFI route to receive this thermal branch; no DFI support is claimed. |
| `share/dfi.F:963` (source) | Second filter pass: convert theta_m to physical temperature for saturation. | Retained refusal: namelist_import.py:DFI_REFUSAL names the absent backward/forward digital-filter pass and its state combination. The engine has no DFI route to receive this thermal branch; no DFI support is claimed. |
| `share/dfi.F:3301` (source) | Filtered-state hydrostatic top-level inverse density. | Retained refusal: namelist_import.py:DFI_REFUSAL names the absent backward/forward digital-filter pass and its state combination. The engine has no DFI route to receive this thermal branch; no DFI support is claimed. |
| `share/dfi.F:3324` (source) | Filtered-state hydrostatic lower-level inverse density. | Retained refusal: namelist_import.py:DFI_REFUSAL names the absent backward/forward digital-filter pass and its state combination. The engine has no DFI route to receive this thermal branch; no DFI support is claimed. |
| `share/input_wrf.F:57` (source) | Declare the file thermal metadata variable. | ingest/wrfinput.py reads USE_THETA_M metadata and retains both dry T and native THM. namelist_import.py resolves the requested selector before restoring the selected prognostic. |
| `share/input_wrf.F:200` (source) | Read the file USE_THETA_M attribute. | ingest/wrfinput.py reads USE_THETA_M metadata and retains both dry T and native THM. namelist_import.py resolves the requested selector before restoring the selected prognostic. |
| `share/input_wrf.F:211` (source) | Check that file and namelist thermal choices agree. | namelist_import.py compares the requested scalar with the producing file metadata and refuses a mismatch. A matching option-1 pair retains native THM and moist wrfbdy T; there is no dry conversion on that selected route. |
| `share/input_wrf.F:212` (source) | Report the file choice on mismatch. | namelist_import.py compares the requested scalar with the producing file metadata and refuses a mismatch. A matching option-1 pair retains native THM and moist wrfbdy T; there is no dry conversion on that selected route. |
| `share/input_wrf.F:214` (source) | Report the namelist choice on mismatch. | namelist_import.py compares the requested scalar with the producing file metadata and refuses a mismatch. A matching option-1 pair retains native THM and moist wrfbdy T; there is no dry conversion on that selected route. |
| `share/input_wrf.F:216` (source) | Report the consistency error on mismatch. | namelist_import.py compares the requested scalar with the producing file metadata and refuses a mismatch. A matching option-1 pair retains native THM and moist wrfbdy T; there is no dry conversion on that selected route. |
| `share/input_wrf.F:1069` (comment) | Comment table for old input compatibility. | ingest/wrfinput.py:restore_domain_state refuses requested theta_m when USE_THETA_M is not 1 or THM is absent. Old-input compatibility comments do not add a separate arithmetic branch. The established option-0 import conversion remains available. |
| `share/input_wrf.F:1080` (comment) | Comment documenting old-input restriction. | ingest/wrfinput.py:restore_domain_state refuses requested theta_m when USE_THETA_M is not 1 or THM is absent. Old-input compatibility comments do not add a separate arithmetic branch. The established option-0 import conversion remains available. |
| `share/input_wrf.F:1085` (source) | Reject older input data for a requested moist integration. | ingest/wrfinput.py:restore_domain_state refuses requested theta_m when USE_THETA_M is not 1 or THM is absent. Old-input compatibility comments do not add a separate arithmetic branch. The established option-0 import conversion remains available. |
| `share/input_wrf.F:1086` (source) | Report the old-input rejection. | ingest/wrfinput.py:restore_domain_state refuses requested theta_m when USE_THETA_M is not 1 or THM is absent. Old-input compatibility comments do not add a separate arithmetic branch. The established option-0 import conversion remains available. |
| `share/mediation_integrate.F:752` (source) | Pass child thermal choice to terrain adjustment | ingest/nest_init.py passes cfg.use_theta_m to terrain correction. experiment.py and domain.py reject parent/child selector disagreement because forcing and feedback must carry one prognostic variable. |
| `share/output_wrf.F:42` (source) | Declare the output thermal metadata variable. | io/wrfout.py writes USE_THETA_M=1 when selected; frame T is dry perturbation theta and THM is selected prognostic perturbation. Dry output metadata and arithmetic are preserved at 0. Restart files retain selected state with fingerprinted configuration. |
| `share/output_wrf.F:194` (source) | Read the selected option for output metadata. | io/wrfout.py writes USE_THETA_M=1 when selected; frame T is dry perturbation theta and THM is selected prognostic perturbation. Dry output metadata and arithmetic are preserved at 0. Restart files retain selected state with fingerprinted configuration. |
| `share/output_wrf.F:679` (source) | Write USE_THETA_M to file metadata. | io/wrfout.py writes USE_THETA_M=1 when selected; frame T is dry perturbation theta and THM is selected prognostic perturbation. Dry output metadata and arithmetic are preserved at 0. Restart files retain selected state with fingerprinted configuration. |
| `share/track_driver.F:123` (source) | Convert prognostic theta to physical track temperature | core/attribute_tracking.py converts selected theta to dry theta before theta-based track attributes. Products from history consume dry T. Native storm-track equivalence is not claimed. |
| `share/wrf_timeseries.F:386` (source) | Surface time series: recover dry perturbation theta. | io/wrfout.py emits dry T. live_columns.py and simulated_radar.py consume frame fields with that convention; state-based thermodynamic consumers use total_theta. Column/sounding real-forecast comparison remains outstanding. |
| `share/wrf_timeseries.F:455` (source) | Column time series: recover full dry theta. | io/wrfout.py emits dry T. live_columns.py and simulated_radar.py consume frame fields with that convention; state-based thermodynamic consumers use total_theta. Column/sounding real-forecast comparison remains outstanding. |
| `phys/module_diag_trad_fields.F:18` (source) | Accept option argument in traditional fields signature. | The WRF argument is declared but has no active branch in this file. Engine diagnostic consumers receive dry theta through core/state.py:total_theta or history T, while raw THM remains separate. |
| `phys/module_diag_trad_fields.F:43` (source) | Declare option argument; no branch uses it in this file. | The WRF argument is declared but has no active branch in this file. Engine diagnostic consumers receive dry theta through core/state.py:total_theta or history T, while raw THM remains separate. |
| `phys/module_diagnostics_driver.F:309` (source) | Pass the option into traditional diagnostics; t argument is already dry. | core/gsd_diagnostics.py and the output diagnostic pipeline use dry thermodynamic inputs. io/wrfout.py keeps T dry and THM prognostic, so file-based traditional products do not reinterpret theta_m as dry theta. |
| `phys/module_diagnostics_driver.F:1252` (source) | Refresh th_phy_m_t0, the dry T field written to history. | core/dycore.py and core/nest_round_trip.py save the strict dry step-entry T view for history; io/wrfout.py uses that retained view when required by strict numerics. Fast history derives dry T from the current selected state. |
| `phys/module_firebrand_spotting.F:2248` (source) | Convert moist prognostic theta to dry physics theta | core/sfire_spotting.py:prepare_atmosphere forwards state.use_theta_m to prepare_native_fields, whose device preparation converts selected total theta to dry physics theta. No new fire feature or native whole fire forecast is claimed. |
| `Registry/Registry.EM_COMMON:211` (source) | THM binds the selected prognostic and its boundary interpolation, while T remains dry. | io/wrfout.py separates dry T from selected THM; ingest/wrfinput.py retains matching native THM and selected wrfbdy T. Nest interpolation, relaxation, specified zones and feedback carry raw selected thermal fields with qv, not the dry diagnostic view. |
| `Registry/Registry.EM_COMMON:2860` (source) | Declare scalar use_theta_m with stock default 1. | config.py defines scalar use_theta_m with engine default 0. Namelist import resolves an omitted value from the file WRF line, logs that decision, and emits the selected config value. Python/TOML and the shared go configuration carry it. Fast numerics remains default. The scoped strict whole-forecast comparison above passes; this does not claim all configurations are bitwise exact. |

Total: 66 literal sites. Counts by file:

- `Registry/Registry.EM_COMMON`: 2.
- `dyn_em/module_big_step_utilities_em.F`: 9.
- `dyn_em/module_diffusion_em.F`: 4.
- `dyn_em/module_first_rk_step_part2.F`: 1.
- `dyn_em/module_initialize_fire.F`: 1.
- `dyn_em/module_initialize_heldsuarez.F`: 1.
- `dyn_em/module_initialize_ideal.F`: 1.
- `dyn_em/module_initialize_real.F`: 3.
- `dyn_em/module_initialize_scm_xy.F`: 1.
- `dyn_em/module_initialize_tropical_cyclone.F`: 1.
- `dyn_em/nest_init_utils.F`: 5.
- `dyn_em/solve_em.F`: 3.
- `dyn_em/start_em.F`: 8.
- `phys/module_diag_trad_fields.F`: 2.
- `phys/module_diagnostics_driver.F`: 2.
- `phys/module_firebrand_spotting.F`: 1.
- `share/dfi.F`: 4.
- `share/input_wrf.F`: 10.
- `share/mediation_integrate.F`: 1.
- `share/output_wrf.F`: 3.
- `share/track_driver.F`: 1.
- `share/wrf_timeseries.F`: 2.

## Sites without a literal use_theta_m

- `Registry/Registry.EM_COMMON:209-211`: T is dry perturbation theta; THM is the selected prognostic perturbation. `config.py`, `experiment.py` and `domain.py` enforce one scalar thermal choice across the nest tree. `io/wrfout.py` keeps the two history fields distinct.
- `dyn_em/module_big_step_utilities_em.F:6674-6707`: `conv_t_tendf_to_moist` applies `(1+rvovrd*qv_old)*t_tendf + rvovrd*dry_theta_old*qv_tend`. `core/theta_m_physics.py` and `core/dycore.py` implement this before diffusion, using the previous saved t_1 thermal value, current moist_old vapour and a copied tendency. Native conversion precedes small_step_prep replacing t_1 with t_2; the engine preserves that ordering. `ensemble/batch_moist_dycore.py` converts admitted member-owned copies.
- Acoustic preparation and finish: `core/kernels/dycore.cu`, `core/dycore.py` and `core/bandwidth_glue.py` couple and uncouple the raw selected thermal field. `prognostic_theta()` is used for transport; `total_theta()` is reserved for dry physics/diagnostics. `ensemble/batch_glue.py` forms the same raw total thermal field. No separate acoustic source branch is required by `module_small_step_em.F`.
- Acoustic coefficients: `core/acoustic.py:calc_coefs` and `core/kernels/acoustic.cu` form c2a, alpha and the vertical solve coefficients from selected-branch pressure and inverse density. `core/diagnostics.py` removes the extra vapour multiplier at 1 before those coefficients are evaluated. `advance_mu_th`, `advance_w_phi`, pressure increments and small-step finish continue consuming raw selected thp/thb and their coupled increments. The ensemble acoustic/big-step bindings preserve these operands.
- Water loading and buoyancy: `core/acoustic.py:calc_cq` retains all active scheme water mass. `core/kernels/dycore.cu:pg_buoy_w` retains its water-loading contribution; switching thermal variables does not remove liquid, ice or vapour mass. The only EOS multiplier removed at 1 is the duplicated vapour contribution already carried in theta_m.
- Diffusion stability and closures: `core/kernels/smag2d.cu` recovers dry theta for N2 and TKE thermodynamics under `GPUWM_USE_THETA_M`, while transported thermal diffusion uses selected theta. `core/dycore.py` binds the appropriate variant without changing the dry arm. Strict moist mixing seeds physics and preserves native vertical-then-horizontal accumulation order. The existing prepared ensemble graph has no km_opt=2/3 closure binding and no admitted moist surface flux inputs without PBL; its concrete refusals remain.
- `main/real_em.F` has no literal use_theta_m at this commit. Its boundary construction consumes selected `grid%t_2` from `module_initialize_real.F:4909`. The matching native option-1 file door retains THM and moist wrfbdy thermal strips. Generic preparation converts the completed dry initialized state once at setup; no native executable is on a user route.
- Online nests and relocation: `ingest/nest_init.py`, `core/nest_round_trip.py`, `core/nest_operands.py` and `core/microphysics_transition.py` preserve the selector and raw selected thermal field with qv through initialization, force, feedback, copied overlap and moving windows. Device transfer carries those arrays and selector without a thermal conversion. The two-domain strict hour above passes. Ideal moving-nest integration passes at both selectors, including exact overlap retention of the previous thermal state at 1. The follow-up real moving forecast below uses a fetched terrain corridor. Real multi-card execution is not measured.
- Offline archive children: `offline_child.py` reads native THM for selected option 1, interpolates/remaps it alongside qv, and recovers dry T for terrain and number closures. Own-terrain adjustment regenerates THM; initial state and both boundary routes use selected THM. Missing USE_THETA_M=1 metadata or THM is refused because dry archive T cannot supply selected-variable interpolation. Focused real-file reader checks cover inherited/vertical/own-terrain preparation, not a real nested forecast.
- Restart and checkpoint: `io/restart.py` fingerprints the thermal configuration and preserves raw selected arrays; legacy absent selectors mean 0. Member views and packed Thompson carry the selector. `io/wrfout.py` supplies dry T to products, live columns and simulated radar. `core/sfire_spotting.py` forwards the selected convention explicitly. A checkpoint at 30 minutes continues to the same one-hour result: zero of 35,738,754 typed history words differ from uninterrupted execution (`restart-clean-all-types-compare.json`). The six-hour fast 0/1/native comparison and observation-sheet review are recorded in the lane RESULT.md. Native-file restart and multi-card execution have separate limits; checkpoint equality does not establish those routes.
- Combined settings: the km5 and averaged-flux branches are integrated. The two literal implicit-mixing sites above use the real thermal selector. Combined proof scope is recorded separately from the earlier single-feature numbers in `docs/wrf-settings.md`.

## Decisions

- Keep `use_theta_m=0` as the engine default and fast numerics as the default for either selector. Strict numerics is used only for named proof runs.
- `run --wrfinput` with matching file and requested values of 1 retains THM and moist boundary T as WRF does. The dry diagnostic T field does not replace that prognostic. A requested option-1 input without native THM or compatible metadata is refused.
- Namelist import resolves an omitted value from the file's WRF line and records the choice in its log. Shared Python/TOML and go configuration carry that value; there is no per-case adapter.
- The temporary admission gate was removed after the integrated route completed a real one-hour forecast. Hydrostatic integration and DFI retain concrete missing-solve/filter refusals. Source coverage, component comparisons and an implemented path are not a completed-port or support claim.

The six-hour fast comparison covers 2019-09-04 12 to 18 UTC on a 73 by 60 by 44 mesh. At hour six selected-one versus native-one root-mean-square differences are 0.03479 K for 2 m temperature, 0.06507 K for dewpoint, 0.05684 m/s for 10 m wind speed and 0.67094 dBZ for column-maximum reflectivity. All 37 maps and time-series figures were inspected. Moisture and surface-pressure differences change sign rather than diverging in one direction. MRMS hybrid-scan observations support a broad rain-pattern comparison, not equality to simulated column maximum. Evidence: `WRF-SETTINGS-2026-10-31/wrfs-theta/doors/MODERN-VISUAL-REVIEW.md`.


## Initial small-grid timing and restart scope

The fast whole-forecast timing pair completed 120 steps per setting on the
same shared device, serialized with the other lanes under the approved timing
procedure. Median step wall time after the first step was 9.841 ms at 0 and
10.173 ms at 1, a ratio of 1.0337, below the 1.2 limit. Every measured stage
median was within 1.045 of its dry counterpart. Event measurements added no
synchronization. Configuration, commands, occupancy, all samples and stage
measurements are in `WRF-SETTINGS-2026-10-31/wrfs-theta/TIMING.md` and
`fast-whole-stage-timing-receipt.json`. This measures the 90 by 81 by 39
single-domain case above with RADT=10 and fast numerics, not all mesh sizes.

Native WRF restart-file continuation completes in both numerical modes, but
its strict continuation is not bitwise exact. Retaining native restart
pressure and inverse density fixed one initialization defect. The remaining
comparison has 41 of 187 common floating fields differing after ten minutes
and 74 after twenty and thirty minutes. At thirty minutes maximum differences
are 0.146545 K in 2 m temperature and 1.039063 Pa in surface pressure; every
floating output is finite. The remaining cause is not localized. Evidence:
`native-restart-retained-eos-summary.json` and
`native-restart-retained-eos-finite.json` in the same evidence folder.
The exact engine-checkpoint proof above is a separate result.

The real moving-nest follow-up below resolves the initial terrain-corridor
limit through the normal geography fetch door. Real multi-card execution
is not measured; multiple permitted physical devices were unavailable. Native feedback replay covers resident storage, either stored domain,
and both stored domains: 73 fields and 619,650 words per combination, with
zero differences (`feedback-storage-replay.json`). This does not substitute
for a multi-card forecast. The combined km_opt=5 and averaged-flux integration is documented separately
in `docs/wrf-settings.md`.


## Real-size follow-up and dry restart comparison

The required real-size timing pair uses the public NCAR CONUS 12 km case,
2019-11-26 12 UTC, 424 by 299 by 49 mass cells, 100 steps of 36 seconds,
schemes 8/4/4/2/2/2/6, diffusion 1, mixing 4 and RADT=15. Exactly one
back-to-back pair ran with fast numerics under the authorized shared-device
procedure. Median step wall times were 84.598 ms at 0 and 84.667 ms at 1:
ratio 1.000816, below 1.2. This supersedes the small-grid acceptance number.
Per-stage samples, commands, hashes and occupancy are in
`WRF-SETTINGS-2026-10-31/wrfs-theta/real-size-timing-receipt.json`.

All 198 floating output fields in ten frames were finite. Twelve RustWx
comparison sheets and the domain-water/surface-pressure series were opened.
Broad weather patterns agree and no new boundary ring is visible. At one
hour surface-temperature RMS difference is 0.0223 K, with one interior cell
exceeding 1 K (maximum 3.137 K); domain-water difference is 3.52 parts per
million and mean surface-pressure difference 0.01066 Pa. These compare the
two thermal settings, not native WRF equality. Field hashes, reduced arrays
and review are in `real-size-retained/` and `REAL-SIZE-TIMING.md`.

Exactly one dry native-restart continuation was then run in the same
90 by 81 by 39 strict configuration as the retained moist continuation.
At ten/twenty/thirty minutes, 74/75/75 of 187 common floating fields differ
from native at 0, versus 41/74/74 at 1. No field differs only in the moist
run. Final maximum 2 m temperature differences are 0.147034/0.146545 K,
and surface-pressure differences 1.046875/1.039063 Pa, for 0/1 respectively.
The accumulated sensible-heat errors also evolve comparably. The dry
importer's separate RADT=9.9 to 10 rounding defect affects solar fields;
the moist run retains 9.9 and matches native COSZEN. These results establish
an existing warm-start door limitation, not identical causes or error words.
Reproduction commands, namelists and fieldwise comparisons are retained in
`doors/stage2/` in the evidence folder.

A user with a stock moist-theta WRF restart file gets a completed forecast
with finite fields, but its continuation is not bitwise equal to native WRF.
Engine checkpoint continuation retains its separate exact proof. The dry
arm is unchanged in this follow-up; the release owner decides whether to
apply the generic corrections listed in the evidence's `Defects for the lead`.


## Real moving-nest follow-up

One 30-minute real forecast completed through `woof run CONFIG` with
`use_theta_m=1`, fast numerics, public ERA5 for 2001-06-11 12 UTC and
geography from `woof fetch-geog --datasets wrf`. The parent is 84 by 68
at 12 km, the child 90 by 90 at 4 km, both with 49 mass levels and
30/10-second steps. Physics is Morrison 10, YSU 1, MM5 surface layer 91,
Noah 2 and legacy RRTMG 4/4, with feedback off. At 15 minutes the child
moved one parent cell east. All fourteen histories are finite. The move
retained 96.6667 percent overlap in 26 fields, skipped none, left the
parent unchanged in words and matched all 759,510 checked static cells.

All eight RustWx sheets were opened, covering temperature, dewpoint, wind
and reflectivity in both domains at 0/10/15/20/30 minutes. The moved
footprint appears in the 20-minute history. No new straight edge band,
perimeter ring or abrupt wind seam is visible. Reflectivity remained
blank, so this is not a moving rain-seam proof. Initial lake and terrain
adjustment is visible; normal initialization retained forcing for 107
lake cells outside the separate lake-state coverage. The changed footprint
also changes the domain-mean pressure, which is not a fixed-area drift test.
Commands, configuration, reductions, hashes and figures are in
`WRF-SETTINGS-2026-10-31/wrfs-theta/REAL-MOVING-RANKS-PROOF.md`,
`moving-normal-config/`, `moving1-normal-retained/` and `moving-normal-maps/`.

## Generic corrections deferred for the dry route

The follow-up makes no source changes. The release owner decides whether
to apply these existing moist-only repairs at `use_theta_m=0`. Full exact
namelists, evidence, effects on dry runs and the remaining native-restart
limitation are under `Defects for the lead` in the lane's `RESULT.md` and
`doors/stage2/DEFECTS-FOR-LEAD.md`:

| Existing dry defect | Moist repair commit | Consequence of extending the repair to dry runs |
| --- | --- | --- |
| RADT=9.9 becomes 10 for solar geometry. | `f61de8cc71651a1fe8e1228b3aff9e7402e72547` | Solar-angle and subsequent forecast words change. |
| Cold Eta startup leaves U10E/V10E unseeded. | `bfc7f4d09db6595f6eab22ce53dca28ef72a2014` | Initial wind diagnostic words change; this is not a rotation-formula repair. |
| Initial child feedback leaves parent P_HYD stale. | `bfc7f4d09db6595f6eab22ce53dca28ef72a2014` | Initial parent pressure diagnostic words change. |
| Runtime feedback omits live surface, soil, radiation, rainfall and carried physics fields. | `e8354638a074098606779cd0f2e1b4ca71d371cc`, `f66accad677cc275e33c9a24dc8aeedceeaf01b3`, `64c7cd8f76bd31be9bdba53d3b3f428e2a247e01` | Parent fields inside the child footprint and later forecasts change. |
| Native warm initialization recomputes saved pressure and inverse density as if starting cold. | `b1894e6f287cae9186244096f00520a206ef98df` | Initial warm physics inputs and later forecast words change. |
| Strict diffusion option 2 adds mixing and physics in a different order from native. | `bfc7f4d09db6595f6eab22ce53dca28ef72a2014` | Rounding and boundary tendencies can change; this is source-established, not a new measured dry forecast failure. |
| Fast native restart can reject valid signed transport residuals. | `e8354638a074098606779cd0f2e1b4ca71d371cc` | Admission and recorded normalization policy change; the new dry warm check used strict numerics. |

Shared warm-file mapping repairs already applying at 0 are listed separately
in that ledger. Cold-run byte comparisons do not certify native warm-file
continuation. Thermal-variable conversion itself is not a generic dry defect.


## Ranked history conversion

The first real two-rank comparison exposed a history-only defect: stored
history wrote the moist prognostic into both `T` and `THM`. Every other
field, including THM and vapour, matched the resident forecast. The stored
and case-frame writers now derive dry `T` from selected moist theta and
vapour, leaving `THM` prognostic. A thermal-only history request includes
vapour in its downloaded dependencies. The dry-selector branch is unchanged.
The StoreFrame regression suite passes 28 checks. The corrected whole
ranked forecast at `f364db15e7c7346275c7e04ba6fca74104d664b4` matches the
unaffected resident control at 0/5/10 minutes: zero of 23,997,339 typed
words differ, including all 23,963,007 floating words, with identical
schemas. All three history files also have identical SHA-256 hashes.

Scope: the moving case's 84 by 68 by 49 root, 20 steps of 30 seconds,
fast numerics, the same sealed ERA5 preparation and free-threaded Python,
using `woof prep` then `woof sim --devices-table`. Two ranks use a
1 by 2 layout with `ids=[0,0]` and configured host transport on one physical
card. Co-located ranks actually use one local transfer leg and no host legs.
This is not a physical multi-card proof; real multi-card is not measured.
Direct CONFIG execution refuses `[devices]`, so the documented prepared
route was used. One completed CONFIG resident check precedes the compared
prepared control, and only the affected two-rank forecast was repeated
after fixing T. Pre-step interpreter and memory refusals are retained.
The optional column store was not written because `rw_soundings` was absent;
complete history fields were compared, not columns or soundings.
Evidence: `rank-one-two1-prepared-fixed-all-words.json`, the two retained
run reports, exact command scripts and `REAL-MOVING-RANKS-PROOF.md` in the
lane evidence folder. These are correctness runs on a shared card, not
additional speed claims.
