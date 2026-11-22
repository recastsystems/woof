# First-step U forcing and moisture pressure correction

An unmatched moist physics selection must retain WRF's moisture pressure
correction. The implicit-switch authority now derives `moist_cq = true`
from active microphysics when no shipped profile matches. Explicit values
in matching shipped profiles remain authoritative, and unmatched dry
selections keep the correction off. The importer, catalog and domain
writer use the same authority so their configurations still round trip.

Previously, changing the surface-layer selector could make an otherwise
moist suite miss its profile and inherit the dry `RunConfig` default.
For Thompson, YSU, Noah and legacy RRTMG, surface layer 91 matched a
profile with moisture correction on; revised surface layer 1 did not.
The latter forecast omitted the factor from pressure forcing and the
acoustic updates, although compiled WRF used it. The correction ships
by default through `physics_compat.implicit_runtime_switches`.

A measured first-step budget at the first east U relaxation point gives
the following coupled tendencies on RK stage 1:

| Term | Compiled WRF | Previous WOOF-ArWen |
| --- | ---: | ---: |
| Horizontal advection | -21.24713898 | -21.24713707 |
| Vertical advection | 0.23670197 | 0.23670197 |
| Big-step horizontal pressure gradient | 227.67422485 | 325.39910698 |
| Coriolis and curvature, combined | 44.64031982 | 44.64031982 |
| Held Davies relaxation | -0.36744955 | -0.36744955 |

The full budget retains each RK stage, every acoustic update, horizontal
diffusion, PBL forcing, prescribed boundaries and the mass factors. Both
models' forcing sums, acoustic chains, mass seeds and finishes close to
the observed stored words. The pressure-gradient term contains the first
large local difference, but a local forcing difference does not establish
the cause of the forecast discrepancy.

Identical-input compiled replays separate the causes. On native input
words the pressure-gradient replay reproduces the live WRF result.
On WOOF-ArWen input words it reproduces the larger force to within 0.003
coupled units. Replacing pressure alone accounts for 99.90 percent of the
input-induced difference. A separate native-input control measures the
omitted moisture correction as about 3.906 coupled units at this point.

The pressure rewrite comes from the declared precision choices in
`kernels/diagnostics.cu`: separate geopotential differences with a base
thickness residual, and `log1p` of a preserved pressure drop instead of
the rounded pressure ratio. The actual prognostic and pressure carriers
in this experiment are FP32. Those precision choices remain the default.
The first refresh also precedes physics in WOOF-ArWen, while WRF retains its
imported pressure until after the first acoustic stage.

The six-hour comparison uses unchanged input files and the original
compiled-WRF control spread. Ratios are domain-wide RMSE divided by the
larger WRF O2 decomposition or O3 compiler-control RMSE:

| Field | Previous default | Moisture correction fixed |
| --- | ---: | ---: |
| T2 | 4.1139 | 0.8742 |
| U | 2.6917 | 0.9670 |
| Q2 | 5.2155 | 1.1292 |
| RAINNC | 8.1224 | 1.2127 |

The fix reduces first-hour U from 32.6057 to 1.7547 times that spread.
A separate verification-only EOS control retains the previous moisture
setting and changes just the EOS arithmetic plus the first-stage refresh
timing. Its first-hour U remains 32.6029; its six-hour T2, U, Q2 and rain
ratios are 4.7176, 2.8066, 5.5381 and 9.6814. This control does not improve
the forecast comparison. Its effects are measured on the preceding
default, not on the corrected moisture setting, and are not additive.

The large forecast difference is therefore localized to the missing
moisture pressure correction, despite the larger first-stage local EOS
effect. The residual ratios near the control spread do not identify a
new defect or assign the remainder to any one declared choice. These
measurements establish numerical behavior in the tested forecast, not
observation-based forecast skill.
