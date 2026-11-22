# Condensate loading at initialization

`hex init` completes hydrostatic balance with cloud water, rain, ice, snow and
graupel by default under fast numerics, after the producer has populated those
fields, whatever model the first guess came from. Under strict numerics, the
proof mode where the init must be the producer's own (MPAS-A case 7, or NOAA's
RRFS fork), the default is no, so a GFS start there still matches NOAA's RRFS
init. Both the standard and RRFS producer routes, and the blend's inside init,
use `woof.hex.init_balance.complete_init`. An explicit `--condensate-balance yes`
or `no` overrides either default. The provenance receipt records the selection,
what chose it (`chosen_by`), changed column count, maximum pressure and theta
changes, and completion time before hashing the final output.

Vapour is used as the producer wrote it, neither clipped nor written. The RRFS
producer can leave slightly negative `qv` (16 points down to -6.9e-6 kg/kg in a
GFS start for 2024-05-21 12Z); the balance takes it as the forecast will, and
refuses only vapour that breaks the moist equation of state. The completion
writes block by block, so a refusal removes the incomplete init and names why.

The hydrostatic stencil is MPAS's perturbation-pressure relation. Its weight
includes `qv + qc + qr + qi + qs + qg`, consistent with the forecast dynamics.
Number concentrations and graupel volume are not mass mixing ratios. The stock
MPAS case-7 initialization uses only vapour because it does not import these
hydrometeors; adding them later requires completing the pressure balance.

The completion holds physical dry density, every water species and the vertical
grid fixed. It preserves dry and total moist column mass exactly. It anchors
pressure at the top mass level and integrates downward using the saved native
`fzm`, `fzp` and `rdzu` coefficients. It then inverts the cold-reader equation of
state for theta; changing only the saved pressure would be overwritten by that
reader. Base pressure used by the solve is reconstructed from `rho_base`,
`theta_base` and `zz`, as it is on a cold start. Saved `pressure_p` remains relative
to the saved `pressure_base`, keeping the public diagnostic sum consistent.

Columns with no condensate are not written. Pressure below an elevated cloud
must change even in its clear layers. This completion intentionally changes
temperature instead of density: simply putting condensate into a bottom-anchored
pressure/density iteration would change column mass. It applies to boundary
columns too; it is a cold initialization operation, not an interior analysis
increment or a lateral boundary update.

The arithmetic is double precision in bounded host NumPy blocks. No time-step
kernel or GPU route is changed. The separate experimental `init_gpu.physics`
path currently emits zero condensate, and the CPU `assemble_prognostic_state`
reference currently accepts only vapour. Their vapour-only arithmetic stays
unchanged. Direct use of an external `rw_mpas_init` executable bypasses the
`hex init` completion; this release's command is the supported entry point.

Double precision column residuals vanish to round-off. A float32 init stores a
rounded theta, so the cold reader recovers pressure with float32 storage error.
The fix does not guarantee a stationary three-dimensional state: horizontal
pressure gradients, boundary adjustment and evolving condensate remain. It does
not claim improved storm survival or forecast skill. CPU tests exercise the
completion and both command routes. Two-hour hrrrv5 forecasts from HRRR starts
on a 3 km storm box (2022-03-05 and 2024-05-21) ran with it and scored level
with the unbalanced start against radar and surface reports.

Authority: public MPAS-Model v8.4.2, commit
`f34984b2c89353cf31c235b08d3b4acd96b37bb3`, BSD-3-Clause (the notice is in
`licenses/LICENSE-MPAS-BSD-3-Clause.txt`). The hydrostatic stencil is in
`src/core_init_atmosphere/mpas_init_atm_cases.F`, lines 5316–5319. Total moist
mass, its vertical force and pressure-gradient stencil are in
`src/core_atmosphere/dynamics/mpas_atm_time_integration.F`, lines 3238–3256,
6467 and 6784–6786. The mass-preserving completion is an extension, not the
unmodified MPAS case-7 initializer.
