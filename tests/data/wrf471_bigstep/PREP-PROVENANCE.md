Compiled WRF v4.7.1 geopotential and physics preparation

The reference calls the unchanged rhs_ph and phy_prep routines extracted from
dyn_em/module_big_step_utilities_em.F at commit
f52c197ed39d12e087d02c50f412d90d418f6186. The entire source is pinned to
bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815.
The full module_model_constants.F is compiled. The actual generated
grid_config_rec_type interface is imported, and the actual generated
module_state_description.F is compiled. No numerical routine is stubbed.

The reference build uses gfortran 15.2.0, -O0, -ffp-contract=off and
-fcheck=bounds, with WRF's word-size and EM_CORE preprocessor definitions.
prep_build.py records exact extracted routine hashes. The drivers preserve
the actual argument lists, i/k/j memory order, staggered wind faces, all 49
mass levels, 50 interfaces and four horizontal halo rows on each side.

state-real.npz preserves stored float32 words from a 2024-05-25 18Z WRF real
initialization, input SHA256
29f54e83c7bfed342da29e46b8d520376778aae5560be657e7b2fa519d658d16.
The Rust NetCDF reader and crop_state.rs selected x=70..81 and y=70..79.
state-real.json records every decoded array's shape and word hash.

Eight cases retain those words or apply declared transformations: periodic
and specified boundaries, advection orders 2 and 5, a 1000 m terrain ridge,
map factors from 0.3 to 3.0, zero motion and near-zero motion. Omega is the
actual compiled calc_ww_cp output on the same real wind state, retained in
coupling.npz. The four real cases keep the stored vertical velocity. Terrain
and map stress cases add a smooth 0.5 m/s vertical-velocity perturbation.
The engine's documented total-mass face averaging
is supplied to both rhs_ph calls so that routine is tested independently of
calc_mu_uv. The latter has its own compiled oracle.

The fused slow_geopotential production launcher is graded against the whole
ph_tend array. Its prior order-2 boundary implementation retained the interior
half-face contribution where WRF skips the complete normal-direction term.
The correction applies to the production and supplied-face launchers and to
the old float64 mirror. The exact base kernel fails the corner regression.
Order 2 is now exact: since 2582c065c99 (port/dyn-options, folded into
rel/3.0.0 at 6a79a9fb05; h_sca_adv_order 2 qualified at 63d5c6b7136 with
strict native replays 0/3321496) h_sca_adv_order 2 runs the open-boundary
geopotential kernel, and all six
order-2 cases (real periodic and specified, steep terrain, map extremes,
zero and near-zero motion) match every ph_tend word of compiled WRF
(0 of 6000 differing each, prep_measure.py, RTX PRO 6000 Blackwell,
NVRTC 13.2.51, strict, 2026-11-02). Before it the order-2 maximum was 4608
ULP (near-zero, absolute at most 6.47e-27; real state 0.0703125, steep
terrain 9). Order 5 is unchanged: 1024 ULP, real-state maximum absolute
difference 0.0703125. WRF subtracts PH and PHB separately before summing; the kernel
subtracts their rounded totals, groups mass*g*w differently and subtracts x
before y rather than y before x. The unchanged-routine total-geopotential
decomposition control is saved separately and is not the accepted reference.
tools/bigstep_wrf471_oracle/momentum_rhs_diagnostic.py compiles a transient
copy of the kernel in WRF's operation order with FMA off. Against that
total-geopotential control it matches every ph_tend word in all eight
cases, orders 2 and 5 included, so the remaining native difference is the
PH/PHB split plus float32 evaluation order and nothing else. Receipt:
tools/bigstep_wrf471_oracle/receipts/rhs-arithmetic-attribution.json.

All 15 phy_prep output arrays are retained. Twelve corresponding production
coupling products are measured through physics._prepare_atmosphere and the
existing radiation _t8w_columns adapter: eight are bit identical, density and
Exner are at most 1 ULP, temperature and interface temperature at most 2 ULP.
Density divides (1+qv) once instead of multiplying it by a rounded 1/ALT;
Exner uses CUDA power instead of scalar Fortran/libm power. Interface
interpolation itself is word-identical when given the compiled temperature.
Three WRF diagnostics, th_phy_m_t0, p8w and z, are not returned by the engine
coupling interface and are explicitly ungraded, rather than filled by a
verification transcription. This is not whole-routine phy_prep parity.

The primary gates assert exact per-case measurements and both complete
output hashes. Fixture maxima are measurements, not arbitrary tolerances or
universal error bounds. Unassigned native diagnostic memory is retained as a
sentinel and excluded only where no physical output is defined.

Rebuild with tools/bigstep_wrf471_oracle/prep_build.sh, then prep_generate.py.
Reproduce GPU measurements with prep_measure.py and run the focused tests
in tests/test_bigstep_prep_wrf471_parity.py. The source fixture, reference
executable and exact build flags are recorded in prep-receipt.json and the
build receipts delivered with the comparison report.
