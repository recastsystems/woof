# Constant moving frame

`RunConfig.umove` and `vmove` are constant x/y domain speeds in m/s,
both defaulting to `0.0`. Use them to follow steadily moving features on a
flat idealized Cartesian domain. Smaller frame-relative transport speeds
can reduce advection error and the area needed to contain a feature.

```python
from woof.config import RunConfig
from woof.moving_frame import initialize_moving_frame, ground_relative_winds

cfg = RunConfig(nx=64, ny=64, nz=40, dx=1000., dy=1000.,
                ztop=10000., dt=1., run_seconds=60., umove=12., vmove=-8.)
# Build the state normally and fill initial u/v in ground coordinates.
# state = ...
initialize_moving_frame(state, cfg)
u_ground, v_ground = ground_relative_winds(state)
```

Call initialization after filling initial winds, before consuming them.
The dycore, physics initialization, and state history capture also initialize
the frame on first use. Initialization is idempotent: never subtract the
speed yourself. Thereafter `state.u/v` are frame-relative; advection and
lateral inflow/outflow tests use them. Surface stress, fluxes and PBL physics
use ground-relative winds. Wind tendencies need no translation. The upper
Rayleigh layer damps only vertical velocity, so it has no horizontal wind
target to shift. Restarts retain the frame through their binding config and
restore frame-relative winds without subtracting again.

Supply lateral tables in ground coordinates using the existing builders.
Attachment converts their mass-coupled wind values **and tendencies** once;
`lateral_boundaries_in_frame(state, tables)` exposes that immutable
conversion. `domain_boundary_snapshot` supplies ground-relative coupled
winds, including when called on a moving-frame state. Reattaching converted
tables is safe. Complete tables with linear boundary mass are supported;
lazy tables and rational mass laws are refused because their current
representation cannot carry the converted momentum law faithfully.

History `U`, `V`, `U10`, `V10` and wind diagnostics are ground-relative.
`U10/V10` come from the ground-relative surface calculation and are not
shifted again. The asynchronous history writer records `UMOVE`, `VMOVE`,
`DOMAIN_X_OFFSET_M = umove*t`, `DOMAIN_Y_OFFSET_M = vmove*t` and
`WIND_REFERENCE_FRAME = "ground"` on each output. Time is seconds from
simulation start, including after restart. Direct `WrfoutWriter` users
pass `moving_frame_attrs(state)` as `global_attrs` beside `state_frame(state)`.
Both zero speeds skip conversion entirely and add no frame attributes.

Use resident stepping: streamed tile scatter does not carry the one-time
wind-frame initialization marker and would subtract translation on reuse.

Terrain, real-data geography, nests and moving nests are refused: their
ground-fixed geometry would need translation every step. Prescribed surface
properties must be spatially uniform. Evolving soil, urban, lake, ocean and
fire states are also refused because those ground fields would need
translation. Coriolis and map curvature are refused because shifting winds
without shifting the geostrophic balance changes the acceleration.

These conventions follow the translation and relative-wind definitions in
the [CM1 namelist guide](https://cm1.readthedocs.io/en/latest/README.namelist/).
The [CM1 governing equations](https://www2.mmm.ucar.edu/people/bryan/cm1/cm1_equations.pdf),
sections 2, 4 and 8, distinguish transport, wall stress and geostrophic balance.
See also the 2002 paper
[A Benchmark Simulation for Moist Nonhydrostatic Numerical Models](https://doi.org/10.1175/1520-0493(2002)130%3C2917:ABSFMN%3E2.0.CO;2),
for the moist benchmark formulation. This option does not imply numerical
equivalence to CM1.
