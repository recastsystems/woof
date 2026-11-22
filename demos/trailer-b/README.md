# Moving-source handoff and 111 m tree

This lane implements the source door and provides fallback compute evidence. It does **not** yet establish survival of a mature Xenia storm: the producer has not supplied moving segment histories or a track from which to select its two-hour interval and fixed box. The available 2026-10-16 global source is clear in the full LES footprint. A separate 3 km handoff follows its Texas precipitation band and develops convection. These are import, stability and performance benchmarks.

## Build and verification

From this checkout, with the project's Rust toolchain and Python environment:

```sh
export GPUWM_BRIDGE_SOURCE_REV="$(git rev-parse HEAD)"
cargo build --manifest-path tools/rustwx/Cargo.toml --release --locked --offline -j 8 \
  -p rw-netcdf --bin rw_netcdf -p rw-mpas --bin rw_mpas_convert
cargo build --manifest-path tools/rw_wps/Cargo.toml --release --locked --offline -j 8 \
  -p mapped-engine --bin gpuwm_mapped_engine
export WOOF_RW_NETCDF="$PWD/tools/rustwx/target/release/rw_netcdf"
export WOOF_MAPPED_ENGINE_BIN="$PWD/tools/rw_wps/target/release/gpuwm_mapped_engine"
# Retain your normal rw_wrfbatch, native preprocess/static/write bridges and recast-woof-data installation.
```

The measured checks passed: 77 focused Python ingest/mapping/boundary tests, 206 mapped-engine Rust tests, 438 rw-mpas Rust tests (five ignored), and 45 rw-netcdf tests. These include real NetCDF duplicate-boundary composition, terrain/cadence/vertical-axis refusals, signed W averaging and diagnostic name protection. Exact logs accompany the delivery. `build_gallery.py --root DIRECTORY` builds an offline browser of the existing Rust-rendered PNGs; it performs no weather-array processing.

## Configuration

`les-111m.toml` is the Xenia candidate, starting at 1974-04-03 12Z pending actual track selection. It runs 7,200 seconds with fixed nests and five-minute histories. Dimensions are mass cells; the matching WPS dimensions include the extra endpoint.

| Domain | Grid | Dimensions | Footprint | dt | Closure |
|---|---:|---:|---:|---:|---|
| root | 1,000 m | 120 × 120 × 96 | 120 km | 5 s | YSU, km_opt=4 |
| middle | 1,000/3 m | 180 × 180 × 96 | 60 km | 5/3 s | Smagorinsky, km_opt=3 |
| fine | 1,000/9 m | 270 × 270 × 96 | 30 km | 5/9 s | Smagorinsky, km_opt=3 |

Both nest ratios are three; feedback is off. `mix_isotropic` is omitted to exercise the current default: it selects isotropic mixing on the 111 m grid. The 333 m grid retains the default selection appropriate to its aspect ratio. Physics remains active: Morrison, RRTMGP longwave/shortwave, Noah and surface exchange. WSM6 currently refuses 96 levels because its implementation supports at most 80. Changing from source WSM6 to Morrison is a declared physics change, not a checkpoint continuation. Source WSM6 publishes no number moments; Morrison initializes any absent moments through its normal package policy.

The vertical grid uses 96 layers, an approximately 20 m first layer and a nominal 400 m upper-layer cap; the candidate has 36 mass levels below 1.7 km and 40 below 2 km in its design profile. Actual geometric heights depend on the prepared atmospheric column. The reduced benchmark uses 48/72/108 horizontal cells (48/24/12 km), the same vertical/physics/timestep settings and 600 seconds.

The 30 km box remains fixed. A following nest is extra integration work and will be needed only if the measured storm leaves it during the selected two hours. The mature interval cannot be selected from a case label.

## Export and composition contract

Export every segment with its **own** mesh and init on one explicit geographic grid:

```sh
rw_mpas_convert --history SEGMENT_HISTORY --mesh SEGMENT_GRID --init SEGMENT_INIT \
  --field-set full --window latlon:SOUTH,NORTH,WEST,EAST,SPACING_DEGREES \
  --simulation-start 1974-04-03_12:00:00 --remap product --jobs 8 --out-dir SEGMENT_EXPORT
# Every subsequent segment uses the first exported endpoint as terrain authority:
rw_mpas_convert ... --terrain-authority FIRST_EXPORTED_FRAME
```

This new `latlon:` window fixes the axes independently of each mesh's resolution. Extents must be exact multiples of spacing. `--terrain-authority` binds HGT on those axes; it preserves the other atmospheric columns and their absolute geopotential heights. It is geographic terrain consistency, not a balanced restart or a claim that the source ground interface is unchanged. Export receipts retain mesh/init/history hashes and remapping details. State fields use barycentric remapping; accumulated fields use sampled conservative remapping.

Copy `segments.example.json` and replace its paths, window and endpoint lists. Every file must have one complete valid-time record. All frames must have identical geographic axes, terrain and field inventories, complete finite columns with declared units and vertical axes, and exact five-minute endpoint-inclusive coverage. A two-hour interval needs 25 unique frames. Shared boundary instants are retained from the later manifest segment; the receipt records both discarded and retained column ranges. Frame lists can be unordered. Overlap away from shared endpoints, missing times, grid changes and disappearing fields are refused. No time interpolation or invented five-minute files occurs.

```sh
woof compose-source --manifest segments.json --out composed --jobs 8 --prepare \
  --source hex-global --author-input-manifest source-inputs.json \
  --experiment-config les-111m.toml --wps-namelist les-111m.namelist.wps \
  --geog-root WPS_GEOG --output-root prepared \
  --preprocess-backend cpu --preprocess-workers 8 --hierarchy-workers 8
woof sim prepared --experiment-config les-111m.toml --outdir forecast --render-products refl
```

`composed/inputs.txt` and the supplement bindings feed one existing generic mapped prep source. Rust performs validation/composition; Python launches commands. Prep seals the original full-field input bytes. There is no producer-specific Python adapter.

## Field audit

The shared field table now exports `W` on interfaces and six number columns when published: `nc/ni/nr/ns/ng/nh` → `QNCLOUD/QNICE/QNRAIN/QNSNOW/QNGRAUPEL/QNHAIL` (number per kg). Absent moments stay absent, and a composition cannot silently lose a published field.

The generic `mass_from_interfaces` mapping averages adjacent signed W interfaces onto the source mass levels, then normal prep interpolation reconstructs target interfaces with its normal impermeable endpoint policy. This retains resolved vertical motion, not byte-exact interface values. W was previously zero-filled by this source authority, and even a supplied W was skipped in later boundary-only initialization. Both losses are fixed, including a published-updraft boundary regression. The canonical atmospheric-window path now retains W/omega and all number fields.

SGS TKE is not carried. The chosen km_opt=3 LES closure diagnoses mixing locally and has no prognostic SGS TKE state to import. A prognostic TKE closure requires a declared source field or initialization policy before any continuity claim. Mass moisture species, thermodynamics, geopotential heights, Earth-relative winds and surface/soil inputs continue through the existing table/composition authority. Surface fields are not a replacement for full-column forcing.

## Diagnostics and pictures

```sh
# One path per line; diagnose each source/run separately.
woof storm-diagnostics --input-list frames.txt --out diagnostics \
  --heights 500,1000,3000 --gravity 9.81 --jobs 8 --render --label 'WOOF-ArWen LES'
# The hex export writes geopotential using 9.80665 instead:
woof storm-diagnostics --input-list hex-frames.txt --out hex-diagnostics \
  --heights 500,1000,3000 --gravity 9.80665 --jobs 8 --render --label 'hex source'
python chart_series.py --series 'hex=hex-diagnostics/series.json' \
  --series 'WOOF-ArWen=diagnostics/series.json' --out time-series.png
```

All weather-array reads, vertical interpolation, wind rotation, spherical vertical curl, terrain-slope correction, reflectivity column maximum and statistics run in Rust. Vorticity is relative vertical vorticity in Earth coordinates, evaluated at constant geometric height using samples at the named AGL height. No Coriolis parameter is added. Statistics exclude a two-cell rim; updraft area uses the geographic grid Jacobian and w ≥ 5 m/s. The three time series are max w, max absolute vertical vorticity and updraft area at each named height. Their raw values and source paths are in `series.json`.

Weather maps run through `woof.rustwx` → `rw_wrfbatch`; matplotlib only draws the analysis time series. `chart_series.py --height 500`, `1000` or `3000` selects the named height (default 1000). Generic map legends use the renderer's scale chosen separately for each frame; compare printed values, not colors alone. Diagnostic frames contain available 2-D fields only, so a renderer notice about omitted pressure-volume products is expected and is retained in the receipt. The requested diagnostic maps remain available. [CAPTIONS.md](CAPTIONS.md) records which representative pictures were actually inspected and their interpretation.

## Measured and projected

The delivery evidence contains exact commands, stage logs, passive five-second process/GPU samples and timestamps. VRAM values are sampled occupancy, not exact allocation peaks. Queue time and file-transfer time are separate from command timing.

| Stage / hardware | Result | Wall |
|---|---|---:|
| 60 km source → 3 km prep, eight workers | passed | 9.6 s |
| 3 km WOOF-ArWen, two model hours, PRO 6000 | 314 adaptive steps, 25 frames; 4,922 MiB sampled VRAM | 24.3 s model; 43.72 s command/render |
| reduced 96-level tree prep, eight workers | passed | 26.3 s |
| reduced tree, ten model minutes, PRO 6000 | 120/360/1080 fixed steps; 3,616 MiB sampled VRAM | 30.5 s model; 33.33 s command/render |
| reduced tree, ten model minutes, an RTX 4090 | passed, 120/360/1080 fixed steps; one per-process sample 3,164 MiB; another GPU job present | 79.1 s model; 81.83 s command/render |
| full tree prep, eight workers | passed | 29.6 s |
| full 120/180/270 × 96 tree, two model hours, PRO 6000 GPU 0 | PASS; 1440/4320/12960 fixed steps, 75 frames; 9,454 MiB observed per-process NVML peak | 1,810.6 s model; 1,823.57 s command/render |
| full-tree diagnostics, all 75 frames, eight readers | finite named-height fields and time series | 9.22 s command |
| full-tree selected maps, ten domain/time frames | 70 weather pictures, 1400 × 1000 | 1.61 s command |
| Texas band: 60 km source → 3 km prep, eight workers | 03/06Z full-column forcing, 300 × 240 × 49 target | 9.7 s prep; 10.67 s command |
| Texas band: two model hours, PRO 6000 GPU 0 | PASS; 314 adaptive steps, 25 frames; 4,912 MiB per-process NVML peak | 40.2 s model; 50.98 s command/render |
| Texas band: diagnostics / 28 selected weather maps | all 25 diagnostic times / four selected map times | 3.40 / 0.76 s command |
| regional 316,527-cell/55-level export, first frame, optimized Rust | passed; target 201 × 201; 157 MB output | 3.21 s |
| regional two-frame composition, two readers | passed, unchanged 3 h cadence | 0.60 s processor |
| regional export on the RTX 4090 host, committed optimized Rust, first / second frame | real retained lts-off histories, own mesh/init; fixed 201 × 201 axes and terrain | 4.10 / 4.48 s command; 4.02 / 4.40 s processor |
| RTX 4090 host regional two-frame composition | passed, unchanged 3 h cadence | 1.93 s command; 0.605 s processor |
| regional composed source → reduced tree prep | passed | 27.8 s prep |
| regional five-minute coverage check | correctly refused: 2 frames supplied, 37 needed | see receipt |

The full-tree memory estimator projects a 10.925 GiB peak envelope (7.410 GiB allocation estimate), reduced 4.741 GiB (1.860 GiB allocation). These are estimates. The measured full run's own NVML process peak was 9,454 MiB (9.232 GiB; 894 two-second readings); CuPy reserved-pool peak was 8,101,403,136 bytes. There is no measured capacity requirement for a 96 GB card. Full size on a 4090 remains unmeasured; the reduced 4090 result is measured. Do not extrapolate a mature storm from a calm fallback.

The full run acquired GPU 0 through its mutex; another lane subsequently ran forecasts on that card. The receipt measures 1,540.4 seconds with another process present during 1,808.1 seconds of forecast execution. Timing therefore includes contention. The device-wide 24.51 GB peak includes other processes and must not be attributed to this tree. The hierarchy receipt totals 1,813.08 s including restore/initialization; the model log's 1,810.6 s and command's 1,823.57 s measure different scopes. All 75 domain/time health samples were finite, maximum sampled CFL 0.109584 (vertical 0.103696), recovery status `NOT_NEEDED`, no retries or timestep reductions. These samples do not prove stability under a mature storm.

At two hours, the fine-grid horizontal maxima at 500/1000/3000 m AGL were w = 0.4896/0.3182/0.2285 m/s and |vertical vorticity| = 0.001318/0.001071/0.0003660 s⁻¹, with zero area at w ≥ 5 m/s. The maps show weak adjustment patterns and clear reflectivity. Full-column WRF output, the prepared tree, its matching metadata snapshot, final checkpoints and receipts are mirrored to the RTX 4090 host under its `trailer-les/full-mirror` scratch folder; final raw-file hashes accompany delivery.

All 75 full-tree raw-file SHA256 values match between rental and the RTX 4090 host. The RTX 4090 host's mirror contains three final checkpoints. The copied prepared setup and metadata snapshot retain their original seals; relocating an absolute input path requires the ordinary source-authoring workflow, not editing sealed manifests.

## Separate precipitation-band handoff

The old source's 03:00 UTC Texas band provides a more useful weather example than the calm LES box. `commands/prep-band.sh` and `commands/run-band.sh` in the delivery contain the exact executed commands; `configs/band-3km.*` records the 300 × 240 × 49 target, 3 km spacing, center 32°N/98.5°W, Morrison/RRTMGP physics and adaptive 15–24 s stepping. This two-hour run starts at 2026-10-16 03Z with the 03/06Z source full columns. It is separate from the 96-level fixed-step LES benchmark.

The same-time 1,000 m w pictures visibly retain the source's broad Texas updraft pattern after initialization. The source reflectivity band remains visible after WOOF-ArWen's first five-minute physics diagnostic and develops into a roughly 50–60 dBZ line through 05Z. Initial WOOF-ArWen reflectivity is zero before that diagnostic; it must not be interpreted as lost condensate. At 05Z, maxima at 500/1000/3000 m are w = 2.884/5.314/10.216 m/s, |vertical vorticity| = 0.002664/0.002250/0.002994 s⁻¹, and updraft area = 0/9.261/92.600 km². The charts compare two native source samples with 25 WOOF-ArWen samples; connecting lines do not create measured intermediate source times. Footprints differ. This supports precipitation-system continuity and convective development, not a tracked rotating storm or a moving 1 km handoff.

All 25 band health samples were finite; maximum sampled CFL was 3.58169 and final CFL 1.46867 under this adaptive single-domain configuration. The report's hierarchy-independent total is 40.95 s, while its forecast execution is 39.08 s. Another process shared the card for 38.5 s; the measured 4,912 MiB belongs to this process. Its prepared setup and raw run are mirrored separately on the RTX 4090 host under `band-mirror`. The first proposed southern box failed source coverage, and the first simulation command omitted its required WPS namelist and refused before stepping; both attempts remain in evidence.

One debug regional exporter spent about 49 seconds at roughly 110% CPU despite an eight-thread pool. It was stopped. All-branch history inspection found a parallel physics-pool input reader, but that is a different executable and does not establish converter parallelism. Rebuilding optimized Rust reduced the same first-frame export to 3.21 seconds. An optional broader registry enumeration was also stopped after 89.73 s at about 101% CPU (73 OS threads); 78 tests had passed. This is existing exhaustive Python test enumeration, not a model stage. Its serial validation cost is filed in `trailer-les-serial-test-found.txt`; the 77 focused checks completed in 2.43 s. The debug delay and optimized rebuild are preserved in `single-core-found.txt` and the export receipts. The rental's single-core sentinel remains active; GPU-feeding CPU work is assessed with GPU utilization rather than thread count alone.

The first full prep attempt used the reduced WPS geometry and correctly refused; the matching full namelist succeeded in a new output directory. Cache reuse also refused a later provenance-text edit. Benchmarks using the original cache use its matching pre-audit metadata snapshot, with identical numerical changes; the current regional prep uses the updated authority. A development machine provisioning failures and retries are retained rather than counted as model timings.

A development machine regional inputs and exports are retained under its `trailer-les/{regional-source,regional-export-node1}` scratch folder. Its committed exporter was rebuilt from `1cbcc73e1c5b60a95fbde9b4c332c1a6acc95cdf` with eight Cargo jobs in 43.63 s, SHA256 `fce759e1a62b7eaca2143ce601e6869b4c0092edd217bf63c2552e89e7ff2fb0`. Both full-column exports and the public composition CLI passed. Export CPU time divided by wall was about 1.8 cores for each short stage; neither stage exceeded five seconds. The retained lts-off histories use the same regional mesh/init as the rental proof. The original rental export's `/work/fix-lattice/before` history paths were later removed by their owner; its hashed receipts and exported CF files survive.

## Remaining science work

Actual moving Xenia exports, emitted mesh spacing, a mature object track, common geographic forcing coverage and the chosen two-hour interval remain producer dependencies. The regional fallback's mesh includes a sub-km core, but this measured export window samples its coarse region (mean nearest-cell distance 22.94 km, maximum 37.60 km); it is **not** a 1 km handoff. Its WSM6 history publishes no number moments. The global fallback has only three-hour source samples. Neither fallback can establish same-storm five-minute survival or whether a following nest is needed. Before/after maps and time series must be repeated on the real storm, using identical diagnostic footprints and readable shared interpretation of scales.
