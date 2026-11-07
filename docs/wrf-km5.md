# SMS-3DTKE (`km_opt=5`)

SMS-3DTKE is WRF 4.6.1's scale-adaptive, three-dimensional turbulence
closure. It evolves turbulent kinetic energy (TKE), computes horizontal
and vertical mixing, and solves vertical transport implicitly. The closure
and column solves run on the device. Fast numerics is the default;
`GPUWM_WRF_EXACT=1` selects strict arithmetic for native replay proofs.

Select it in a normal experiment TOML:

```toml
[shared]
km_opt = 5
diff_opt = 2
bl_pbl_physics = 0
sf_sfclay_physics = 1
```

Run with `woof go experiment.toml`. The same WRF names work in
`RunConfig` and in imported WRF namelists. Domain tables can override the
closure and its per-domain controls. The support report names the
`sms-3dtke` turbulence option and its current evidence level.

WRF requires `diff_opt=2`, `bl_pbl_physics=0`, and a surface-layer selector
in `{0, 1, 5, 91}`. It also rejects `cu_physics=11`, whose scale-aware
Kain-Fritsch implementation requires a separate boundary-layer scheme.
Other physics pairings still follow their own validation rules.

Native WRF also has a numerical failure with a disabled surface layer and
positive prescribed heating: `sf_sfclay_physics=0`, `isfflx=0`,
`tke_heat_flux=0.05` left `wspd=0` and produced nonfinite nonlocal momentum
terms in the native first call. The port preserves that equation; use an
active compatible surface layer for heated real forecasts. No additional
WRF pairing rule is invented for this source defect.

| WRF control | Default | Meaning on this route |
| --- | --- | --- |
| `c_k` | 0.15 | TKE exchange-coefficient constant |
| `c_s` | 0.25 | Horizontal Smagorinsky constant |
| `tke_upper_bound` | 1000 | Upper bound for TKE after each Runge-Kutta stage, m2/s2 |
| `isfflx` | 1 | Surface flux source: calculated fluxes at 1, prescribed turbulence heat flux at 0 or 2 |
| `tke_heat_flux` | 0 | Prescribed kinematic heat flux, K m/s |
| `tke_drag_coefficient` | 0 | Prescribed surface drag contribution to TKE production |
| `mix_full_fields` | true | Mix full fields; false uses WRF's base profiles, verified zero for admitted real inputs |
| `mix_isotropic` | 0 | Retained WRF setting; SMS-3DTKE does not consume this length-scale switch |
| `mix_upper_bound` | 0.1 | Retained WRF setting; used by other closures, not SMS-3DTKE |
| `tke_mix2_off` | false | WRF ignores this switch for km_opt=5 and always mixes TKE horizontally |
| `tke_mix6_off` | false | Disables the separate sixth-order TKE filter when that filter is selected |
| `tke_adv_opt` | 1 | Positive-definite TKE transport; other TKE transport algorithms are not implemented |
| `sfs_opt` | 0 | Nonlinear backscatter stresses are not implemented |
| `m_opt` | 0 | Additional WRF stress-output arrays are not implemented |

The last three restrictions are implementation limits, not extra native
SMS-3DTKE pairing rules. Existing moist-advection, damping, hybrid-coordinate,
map-factor and nudging paths remain shared. No new restriction on them is
introduced by selecting this closure.

The closure supports either thermal setting. With `use_theta_m=1`, its
geometry and buoyancy diagnostics use dry theta while implicit thermal
transport carries moist theta. The bottom heat and vapor fluxes and the
interior nonlocal heat flux use WRF's moist factors. Add `do_avgflx_em=1`
to retain the averaged acoustic fluxes. See [combined settings](wrf-settings.md)
for the shared configuration and continuation limits.

TKE and the retained nonlocal momentum terms `gamu` and `gamv` are saved
in restart files. The heat-flux profile, mixing lengths and mesoscale
viscosity are rebuilt each step. Preflight includes both persistent fields
and device workspaces. History includes `TKE`; `rw_compare` can compare
`tke_lowest` and `w1km` alongside the usual weather fields.

WRF's Registry gives TKE and these new fields no parent interpolation or
feedback flags. Each child therefore owns its turbulence state; adding
parent TKE forcing would change the native algorithm. Existing flow-dependent
TKE boundary handling is shared with km_opt=2. Movement and decomposition
must preserve the retained state through the common state inventory.

The retained oracle in `tests/data/oracles/km5_wrf461_strict239` checks
real native calls at steps 1 and 100, at both theta settings. Full native
capture replay has zero differing words over 192 calls with strict
numerics. This does not assert whole-forecast equality. Whole-run, timing,
restart and domain-decomposition receipts are recorded in the lane's
`RESULT.md`. A paired six-hour 3 km forecast, restart continuation and
a five-minute off-case comparison have been completed. The measured
median full-step ratio is 1.128 relative to the fastest km_opt=4 control
under the recorded shared-device timing conditions. The optional strict
whole-forecast attempt stopped before stepping at the existing native
math-library qualification check. Logical halo decomposition and one
scheduled nest move were exercised; physical multi-card execution was not.
