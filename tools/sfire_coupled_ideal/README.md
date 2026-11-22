# Coupled SFIRE ideal verification

The reference is the complete `test/em_fire` case from WRF 4.7.1. Its
original inputs specify a one-hour run, 50 m atmosphere cells, 12.5 m fire
cells, a terrain hill, moist sounding, line ignition and two-way feedback.
Build and run in separate directories:

```sh
bash tools/sfire_coupled_ideal/build_reference.sh WRF_SOURCE BUILD INSTALL 2
bash tools/sfire_coupled_ideal/run_reference.sh WRF_SOURCE BUILD NEW_RUN 21600
python tools/sfire_coupled_ideal/collect_receipt.py \
  WRF_SOURCE ORIGINAL_SOURCE BUILD NEW_RUN reference-receipt.json
```

Launch long jobs detached with stdout logs and a completion marker. The
runner records the exact executables, inputs, initializer product, output
hashes and exit status. It refuses to overwrite an existing reference.
The build uses scalar optimized arithmetic with contraction and vector
math disabled. The routine corpus remains the separately compiled scalar
reference; compiler controls quantify any optimized-run differences.

`initial_state.read_ideal_initial` reads the compiled initializer through
the native `rw_netcdf` bridge and supplies the existing atmospheric
restoration objects. It requires the moist sounding and native base and
eta-coordinate fields. `NFUEL_CAT` in the initializer is zero when
`fire_fuel_read=0`; the fire driver applies the configured constant fuel
category during its initialization pass.
The native ideal initializer sets directional `MAPFAC_MX/MY/UX/UY/VX/VY`
to one and leaves legacy `MAPFAC_M/U/V` zero. Import restores the native
directions after proving their X and Y values equal and retains the legacy
words for evidence. Atmospheric integrals use those same directional
factors and refuse a nonpositive map factor or nonfinite integral.

Build and run the independent native comparison tool on a CPU host:

```sh
cd tools/sfire_coupled_ideal/rust
cargo build --offline --release
cargo test --offline
target/release/sfire-ideal-compare REFERENCE CANDIDATE 4 4 comparison.json
target/release/sfire-ideal-compare REFERENCE - 4 4 reference-summary.json
```

Both inputs must contain genuine `wrfout` files with matching complete
`XTIME` timelines. Decoding uses the existing pure Rust `netcrust` reader.
Fire summaries crop the nonphysical fire extension to atmosphere extent
times refinement. They include `FIRE_AREA`, sensible and latent power,
remaining fuel, and the `LFN=0` perimeter from the Rust contour engine.
Atmospheric summaries include dry mass, vapor mass, mass-weighted theta,
vertical mean theta and moisture, and maximum upward velocity by level.
Each stored field receives FP32 word counts, maximum ULP, maximum
absolute error, bias and RMSE. Geometry comparison includes burned-cell
intersection over union, symmetric nearest-segment distance averaged by
perimeter length at segment midpoints, and the maximum distance sampled
at vertices and midpoints. The latter is a sampled Hausdorff estimate.
Heat integrals use the trapezoidal rule at stored history times and are
identified as estimates rather than every-step heat accumulation.

The temperature integral is a dry-air `Cp*T` proxy. It excludes latent
energy, potential energy, pressure work and open-boundary energy fluxes;
it is not a closed-domain energy budget. Comparison is implementation
verification. Observed wildfire perimeters remain the skill reference.

`run_arwen.py --reference REFERENCE --out NEW_OUTPUT` restores the compiled
initializer, runs the production dycore and attached fire physics on an
integer domain clock, and publishes Rust-written histories. The full
native window is the default; `--run-seconds 2` and `300` are bounded
precursors. Ordinary arithmetic is the default. `--arithmetic wrf-exact`
selects an explicit diagnostic control and is recorded in its receipt.
The runner keeps fifth-order open-boundary transport, the native hill,
moist sounding, open top and moisture pressure coupling. Native ideal
loading forces `hypsometric_opt=1` despite its pre-import metadata of 2.
Inactive package defaults and identical binary32 registry defaults are
recorded without falsely selecting disabled physics arms. WOOF-ArWen's dry
potential-temperature formulation is declared separately from native
WRF's `use_theta_m=1` numerics.

`package_snapshot.py CHECKOUT NEW_ARCHIVE` packages committed engine and
harness paths for node qualification. It records the branch, tip and
complete engine code manifest without carrying Git or environment files.
Unpack it into a distinct node checkout that retains the required native
IO artifacts. The runner verifies that manifest before execution and
records its own harness hashes. `--allow-dirty` labels an exploratory
snapshot explicitly; final qualification requires committed paths.
`--committed-tree` reads the exact tracked HEAD blobs, excluding later
workspace edits and untracked modules. Its receipt records those excluded
edits separately. This permits later-phase development while qualifying a
fixed committed snapshot.

The original complete reference is retained. A distinct corrected source
and executable can isolate reproduced native defects:

```sh
python tools/sfire_coupled_ideal/correct_native_reference.py \
  WRF_SOURCE CORRECTED_SOURCE
python tools/sfire_coupled_ideal/build_incremental_reference.py \
  CORRECTED_SOURCE BASELINE_BUILD CORRECTED_BUILD
bash tools/sfire_coupled_ideal/run_reference.sh \
  CORRECTED_SOURCE CORRECTED_BUILD CORRECTED_RUN 21600
```

The pinned source hashes and unique replacement blocks are verified before
editing the copied source. Native `solve_em` increments the step counter
before fire physics, and the fire driver then uses `itimestep*dt` as the
starting time and advances another `dt`. The corrected driver starts at
`max(itimestep-1,0)*dt`, preserving time zero during initialization and
advancing each additional diagnostic test step correctly. This reference
correction is for a constant timestep; WOOF-ArWen uses its actual calendar.
The open-boundary top geopotential donor is not corrected: WOOF-ArWen follows
native WRF there (`rhs_ph_open/README.md`). Incremental build
receipts hash the unchanged native libraries, compile each changed whole
module with the baseline flags, and relink separate full `ideal` and `wrf`
executables. Baseline sources and binaries remain untouched.
