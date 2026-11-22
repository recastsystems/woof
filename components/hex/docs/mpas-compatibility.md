# MPAS compatibility

woof hex is an independent line-for-line GPU implementation of
MPAS-Atmosphere v8.4.x dynamical numerics. The current upstream pin is
v8.4.2, commit `f34984b2c89353cf31c235b08d3b4acd96b37bb3`.
The source citations and hashes are in `docs/source-profiles.md`,
`woof.hex.mpas_defaults.UPSTREAM_PIN`, and `tools/native_dry_reference/REFERENCE-PINS.json`.
Implementation of the numerics is distinct from equality to a particular
compiled executable: compilers and GPUs round and contract expressions differently.

Internal module names containing `v841` identify the original implementation
path. They do not override the v8.4.2 source pin or its four-sub-step default.

## What agrees, and how tightly

**The claim.** The dycore is byte-identical to native MPAS-A v8.4.2 built
with the pinned strict recipe: GNU Fortran, single precision, no FP
contraction or vectorised reassociation (`gnu-strict`:
`-O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations`;
`gnu-o0` is accepted as the same arithmetic). "Byte-identical" means every
32-bit word of every prognostic and saved-diagnostic field, and of each
transported scalar, is equal. There is no tolerance.

**Measured scope (and nothing outside it is claimed):**

| Case | Hex arm | Result | Evidence |
|---|---|---|---|
| Public `x1.2562`, 26 levels, dry Jablonowski-Williamson, one 1,800 s step, RK2/4, RK3/6 and RK2/2 | NumPy (the CUDA backend of this door was not re-measured on the 0.4.0 tree) | 0 differing words in all 12 fields against `gnu-o0` | `woof hex verify-dycore` (reproduce below) |
| Public `x1.40962`, 55 levels, real terrain, 2023-03-31 12Z and 2022-12-22, dry, 5 days, 21 six-hourly samples | CUDA, one card | 0 differing words, 0 RMS at every sample | dycore-match receipts (kept with the release evidence); reproduce with `tools/native_dry_reference/` |
| `x1.40962` real terrain, water vapour transported, 5 days; water vapour plus five condensates | CUDA, 1, 2 and 4 ranks | 0 differing words at all 20 outputs | identity-extend receipts |
| `x1.163842` (60 km), water vapour, one step and 5 days | CUDA, one card | 0 differing words | identity-extend receipts |

**The identity is per host C library.** Hex evaluates the single-precision
`sinf`, `cosf`, `asinf`, `atan2f`, `powf` and related functions through the
host's GNU libm (`woof.hex.native_geometry_math`, `woof.hex.native_powf`),
exactly as the native executable does. Eight float libm functions differ
between glibc 2.39 and 2.43, so hex and the native reference must run on
the same host, and every receipt names that host's glibc. The measurements
above used glibc 2.39 and GNU Fortran 13.3.0.

**Against an optimised native build** (`gnu-o3`, or Intel) hex differs by
exactly as much as two native builds differ from each other: it sits inside
native MPAS's own build-to-build spread. The verify door holds those
comparisons to the per-field bounds below, because a byte-identity bound
would fail native against native.

| Field | Bound against an optimised native | Units |
|---|---:|---|
| Density, potential temperature, Exner pressure | 6 | binary32 ULP |
| Density times potential temperature, derived by the loader | 6 | binary32 ULP |
| Dry tracer, exactly zero in this test | 0 | binary32 ULP |
| Horizontal normal wind | 0.0002 | m/s |
| Vertical wind | 0.00003 | m/s |
| Horizontal mass flux | 0.00008 | native `ru` units |
| Vertical mass flux | 0.000008 | native `rw` units |
| Density perturbation | 0.0000001 | native `rho_p` units |
| Density-temperature perturbation | 0.000025 | native `rtheta_p` units |
| Pressure perturbation | 0.025 | Pa |

The printed table always includes the number of differing values, the raw
ULP distance, the absolute difference, the bound and PASS/FAIL. It refuses
non-finite values and missing or mismatched arrays, exits nonzero on a
failure, and never changes a bound to fit a new compiler.

**Not measured, so not claimed:** physics on; terrain-heavy or
variable-resolution meshes beyond the cases above; multi-card beyond the
1/2/4-rank moist runs; local time stepping (a different time scheme by
design, never called identical); another host C library without a fresh
measurement on that host.

**Declared residue: where native MPAS reads undefined memory.** A few native
MPAS-A physics routines read memory they never set, so the pinned strict
binary itself writes different words there from run to run. For the MPAS
physics suites such words are left out of an identity check only when that
run-to-run variation has been measured on the pinned strict binary, a
bounds-checked build names the read, and the project owner has signed off on
that one case. Each case is listed in `tools/native_undefined_exclusions.json`
with its exact word set and witnesses, is called a declared residue and never
identical, and every other word is still compared with no tolerance at a
stated process count. One case is registered: in the ozone climatology setup
(`mpas_atmphys_o3climatology.F:164-184`), a cell lying exactly on the last
ozone-table latitude makes native read past the end of the table (line 165),
so the 708 `ozmixm` words of that cell (12 months by 59 levels) are excluded.
No selected real case reaches it; in the synthetic case that does, the other
1,813,188 words of the field agree, at one native and one hex process.

## What is the same

The implemented dynamics follows the MPAS finite-volume, C-grid/TRiSK
operators, terrain-following vertical coordinate, split dynamics/transport
RK integration, acoustic sub-steps, scalar transport and selected mixing
and upper-damping paths. Supported options retain upstream source citations.
An unsupported option is refused by name; this is not all of MPAS.

The mesh reader accepts MPAS NetCDF connectivity, coordinates and metric
arrays, native static fields, and native init-class files containing
`rho`, `theta`, `u`, `w`, `qv`, base-state fields and vertical metrics.
Connectivity is converted from MPAS's one-based indexing internally.
Limited-area meshes require their boundary masks and boundary forcing.
The independent initialization path also accepts WPS intermediate data.
See the mesh and initialization chapters of the manual for the required
variables, supported vertical declarations and explicit compatibility mode.

Output uses MPAS-style unstructured NetCDF dimensions and field names,
including pressure, temperature, reconstructed winds and history times.
Hex checkpoint/restart is its own format; a native MPAS restart is not a
supported substitute. Native MPAS `streams.atmosphere` is not interpreted
by the forecast door. Output selection and interval are hex CLI options.

The forecast door honors supported `&nhyd_model`, `&damping` and
`&limited_area` options through `--dynamics KEY=VALUE` and
`--physics-namelist`. The default schedule follows v8.4.2: RK order 2,
**four acoustic sub-steps**, three dynamics splits and the monotonic
limiter. Two sub-steps remains selectable. Radiation defaults to 1,800 s;
surface/PBL and convection are called every step. The timestep rule is
6 s per km of nominal mesh spacing, subject to the mesh stability checks.
Legacy `config_epssm` is read and ignored, as upstream does.
I/O, MPI decomposition, native restart, printout, soundings and assimilation
namelist groups are not implemented by that importer. LES
(`config_les_model` `3d_smagorinsky` or `prognostic_1.5_order`, with
`config_les_surface`, `config_mix_scalars` and the surface flux and drag
values) runs on the MPAS route (`--physics-backend mpasphys`), MPAS-A
v8.4.2's dissipation module on the GPU; other routes refuse it by name. Use `woof hex physics import` to see
each option's admitted, ignored, outside-scope or refused disposition.

## What is different

The execution is GPU resident CUDA/CuPy, with a NumPy dry reference path.
It has different compiler and floating-point expression evaluation from
native Fortran, even when the operators and schedule are the same.

Physics comes from the WOOF/WOOF-ArWen suite set, shared with the `woof`
distribution. Its verification standard is observations, including MRMS
and ASOS, with case-specific results and limitations in `docs/obs-referee.md`
and `docs/declared-divergences.md`. This dry comparison does not establish
observation-based forecast skill, and choosing an MPAS suite name does not
substitute native MPAS physics.

| Available suite choice | Implementations selected |
|---|---|
| `arwen_mpas_column`, default | WSM6, mesh-dependent convection, YSU, revised MM5 surface layer, Noah-MP, YSU gravity-wave drag, legacy RRTMG, Xu-Randall cloud fraction |
| `arwen_thompson_column` | The default with aerosol-aware Thompson |
| `rrfs_v2_dev` | Aerosol-aware Thompson, MYNN-EDMF/MYNN surface layer, RUC, GSL drag and legacy RRTMG/MYNN cloud; declared RRFS differences accompany the receipt |
| `mesoscale_reference`, `convection_permitting`, `hrrrv5` | MPAS-like selector resolution into the available WOOF implementations; unsupported choices are explicitly refused |

The complete suite and override table is in `docs/physics-suites.md`.
Optional local time stepping is **off by default**. Enabling it changes the
integration schedule and is outside the native agreement claim. Multi-node
self-reproducibility evidence does not establish a native multi-node match.

## Reproduce the one-step check in three commands

Use Linux, a source distribution unpacked into `woof hex`, Python 3.11+,
a supported CUDA device/runtime for the GPU arm, and GNU Fortran plus MPI
development tools (`mpicc`, `mpif90`, `mpirun`), `make`, `git`, `curl`, and
the C build toolchain on PATH. Intel oneAPI can be selected instead with
`--compiler ifx`. Allocate one card using your site's GPU allocation protocol.
Run these commands from the unpacked source tree. The build runs only on
that Linux host, never on a desktop. The public bridge bundle must be
available for your platform; otherwise build `rw_mpas_hostprep` from the
public engine source as described by `woof hex doctor --explain` and set
`WOOF_HEX_RW_MPAS_HOSTPREP` to the resulting binary before the check.

```sh
python -m pip install -e '.[gpu]'
woof fetch-bridges
woof hex verify-dycore --work-root ./native-reference --compiler gnu --backend cuda
```

Use `--backend cpu` for the independent NumPy path or `both` for both.
The command builds or locates the pinned native strict and optimized
executables, downloads the public small mesh, prepares one shared init,
runs all three schedules and prints the agreement and native-spread tables.
It fixes the documented standard-case settings for its native runs, including
the timestep and initial file, even if custom forecast variables are present
in the shell environment. Custom experiments use the lower-level helpers.
Source commits, compiler flags, executable hashes, input hashes, raw
measurements and summaries are saved in the work root. Builds from another
processor/compiler can have different data digests; executable reuse is
checked against its local build receipt and the upstream source pin.
Source validation includes staged and unstaged Git changes. Altered native
numerics are refused, and Makefile changes must match the recorded build flags.
Completed runs are also checked against receipts binding the executable,
initial file, namelists and output bytes. If raw histories were cleaned up,
the command preserves their old logs and regenerates them. Partial runs or
changed cached bytes fail with a named reason and a fresh-work-root remedy.
Initialization receipts bind the pinned public mesh, init executable, decks
and initial state. Cached init and run decks must also match the advertised
JW case and schedule; a self-consistent custom-case receipt cannot stand in
for this test. Each cached executable and initial state must match the
selected reference. Caches made before init receipts were introduced need
a fresh work root.
The CPU tier runs a live comparison when `HEX_NATIVE_DRY_REFERENCE` points
to this work root and otherwise prints an explicit missing-reference skip.

## Five-day divergence check

`tools/native_dry_reference/forecast_divergence.py` requires matching native
strict and optimized six-hour histories from the same real initial file,
through five days, with physics off and the same mixing/damping in all arms.
`run_forecast_pair.sh` builds or locates GNU strict (`-O3`, contraction and
vectorization disabled) and stock optimized (`-O3`) native v8.4.2, then runs
the pair with MPAS 2-D Smagorinsky mixing and Registry damping values.
The one-step standard remains the separate mixing-off case. It runs the hex dry driver
from native strict time zero and compares all three trajectories. Every arm
uses the same diagnostic reconstruction and log-pressure interpolation;
RMS values are weighted by cell area. The wind statistic is the RMS vector
wind difference, not the difference of scalar wind speeds.

The chart and CSV report 500 hPa height and 250 hPa wind growth. Native
spread is a measured comparator, not an adjustable acceptance tolerance.
The summaries distinguish an error envelope at every sample from fitted
log-RMS growth over days 1-5. A one-step PASS does not earn the sentence
that hex diverges no faster than two native builds. Any measured conclusion
must name its mesh, initial state, builds, schedule and dry configuration.

### Measured five-day result, 2026-10-11

Public `x1.40962`, 40,962 cells, 55 levels, GFS-derived 2023-03-31 12Z state
with common regridded terrain, 360 s timestep, RK2/four acoustic sub-steps,
three splits, physics off, identical Smagorinsky mixing and damping. GNU
13.3.0 native builds used the strict `gnu-strict` recipe and stock `gnu-o3`.
Hex ran on one RTX PRO 6000.

| Day-five RMS difference | Native strict vs optimized | Hex vs strict | Hex vs optimized |
|---|---:|---:|---:|
| 500 hPa geometric height, m | 0.000284180 | 0 | 0.000284180 |
| 250 hPa vector wind, m/s | 0.0000954801 | 0 | 0.0000954801 |

Hex against native strict is 0 at all 21 samples for both diagnostics, on
2023-03-31 12Z and on 2022-12-22; hex against the optimised build equals the
native pair at every sample. Zero error has no logarithmic growth fit. The
earlier result of the same day (hex 0.00142 m and 0.000358 m/s against
strict, growing faster than the native pair) was measured before the
operation-order work that closed it. Its receipts, and the closing
measurements change by change, are kept with the release evidence; the
reference they were taken against is rebuilt by `tools/native_dry_reference/`.

The analysis rebuilds the MPAS RBF wind reconstruction from geometry, since
an init file can carry zero placeholders. Height uses scalar-layer geometric
midpoints and interpolation at a unique downward pressure crossing. Outside
or ambiguous crossings are excluded using the same mask for all pairs.
