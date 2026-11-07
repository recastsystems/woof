# Surface forcing and lagged initial conditions

`surface-series` drives `rw_sfc_series`. It fetches selected GRIB records concurrently,
decodes independent timestamps and GRIB fields concurrently inside Rust, and
remaps independent cells with rayon using the MPAS-A v8.4.2 case-8 arithmetic.
No weather-field arrays cross the Python CLI.

Build from a read-only engine source (the case-8 port's public `rw-mpas` library is required):

```sh
tools/build_sfc_series.sh /path/to/gpuwm /path/to/owned/build
export WOOF_HEX_RW_SFC_SERIES=/path/to/owned/build/rw_sfc_series
woof hex surface-series --source gfs --cycle 2023-03-01T00 --hours 720 \
  --static /path/to/native.static.nc --out surface_series.nc
```

The default uses available CPU threads and up to 16 concurrent fetches. `--threads`
and `--jobs` set those counts explicitly. Decode workers are independent Rust
processes using the bundled generic rw-wps decoder. The thread budget is divided
among active timestamp workers; each worker decodes independent fields with rayon.
The producer carries its decoder, so a separately installed intermediate bridge is
not required for surface series. The receipt reports the worker count, rayon threads, bytes,
source URL or CDS request, SHA-256 and fetch/decode/remap/write/wall seconds.
Downloaded GRIB and decoded intermediates are removed with sizes in that receipt.
Failed decoder logs survive in the invocation's scratch folder.

The parent Cargo workspace contains `crates/rw-surface` and explicitly excludes
`crates/rw-sfc-series`. Preserve the reader's release profile, dependency-free
crate and `static-netcdf` feature. The producer declares its own `[workspace]`
and uses its own checked-in lockfile. The parent lock contains only the reader,
so reader builds do not resolve absent engine dependencies. This layout follows
the reader owner's superseding workspace agreement in commit
`6ebc8bbe602b12df75e37c0bf014e70084eabc24`.
The shared reader and locator are
imported unchanged from surface-update commit
`c8be9a9a1bedfeec0610df191659e1fd43185776`. The reader
C ABI source seed is `tests/surface_monthly_input_v842.rs` and its filename
locator is `woof/hex/surface_series_reader.py`.

The earlier producer-only handoff archive remains pinned to its original
standalone layout. Retain its nested `[workspace]` when placing it beside the
reader. The current producer builder also supports handoffs without that
declaration by adding it only when absent in the staged manifest.

The producer's `rw-mpas`, `rw-store`, `netcrust` and `grib-core` path
dependencies come from the engine source. The isolated producer builder stages
or rebases those paths before invoking Cargo without editing the engine source.
Build the parent reader workspace directly, or use the isolated reader builder,
without any engine dependencies:

```sh
tools/build_surface_reader.sh /path/to/owned/reader-build
export WOOF_HEX_SURFACE_READER=/path/to/owned/reader-build/target/release/librw_surface.so
```

The bridge container uses the same isolated reader builder with
`--features static-netcdf` and bundles both the reader library and producer.
An old i5 reader library is unnecessary. Source wheel builds use this reader
builder too; prebuilt bundles verify library and reader source hashes.

For a ten-day proof, use the existing shared init or static file for the target
mesh. Its `latCell`, `lonCell` and `landmask` arrays define the remap; the cell
count alone does not establish mesh identity. Generate the forcing in the run
directory, using the forecast's cycle:

```sh
woof hex surface-series --source gfs --cycle "$cycle" --hours 240 \
  --static "$shared_init_or_static" --out "$run_dir/surface_series.nc" \
  --receipt "$run_dir/surface_series.receipt.json" --threads 24 --jobs 32
```

This makes 41 records at six-hour spacing, including both endpoints. The
consumer reads `surface_series.nc` directly and selects the latest record at
or before the model clock. Keep this prepared file only while the proof uses it,
then retain its receipt and delete it.

The update lane completed the the reference GPU host proof using `2023-03-31T12`
through `2023-04-10T12`, with 41 analyses against its registered native static.
The consumer reports 1,440 steps and 11 daily frames, with integration taking
515.024540165 seconds. All six maps were inspected. All 901,164 daily SST and
ice words equal series records after the shared 0.02 ice cutoff. Its producer
receipt reports 41 records, 24 decode workers, 32 fetch jobs and 4.637917485
seconds. A fresh-process restart from six through twelve hours also passed
bitwise. It also reports a passing static bridge build and 59 GPU surface
tests, plus 581,948,715 C12 written numeric values with zero differences
against native or base. These results were reported by the consumer and were
not independently rerun by this lane. Consumer-relative receipts are
`n1out/ten-day-series.json`, `n1out/ten-day-final-receipt.json.gz`,
`surface-proof/daily.json` and `delivery/REPORT.md`. The supplied producer receipt,
exact forecast CLI and acceptance summary are mirrored in the lane's source
handoff. Full daily/final comparison receipts and the exact restart command
were not included. This
supersedes the earlier 00Z window and unpopulated
`/workspace/shared-prepared/s2s-global120-gfs-2023033100/` reservation.
The consumer has consumed and disposed of the proof outputs and prepared
forcing. Do not regenerate these completed proof artifacts. Keep the receipts.
Surface-update owns `rw-surface`, the shared workspace and forecast runtime;
the producer owns `rw-sfc-series`, acquisition, remap and the writer.

The producer also runs directly with a pinned wheel that does not have the new
`surface-series` CLI. From this source tree, using the exact init supplied to
the forecast:

```sh
tools/run_s2s_surface_proof.sh /path/to/rw_sfc_series \
  "$CONSUMER_INIT" "$CONSUMER_SERIES_DIR" 2023-03-31T12 240 24 32
```

This calls the Rust executable directly, writes the exact command and input/output
SHA-256 receipts, and retains a `.keep` marker until the consumer confirms
consumption. It refuses to overwrite an existing series or receipt. Remove the
marker and prepared data after acceptance and keep the receipts. The mesh input
is read in its original `latCell`, `lonCell`, `landmask` order. If the forecast
uses a reordered init, pass that final init to this command. Do not use a mesh
with the same cell count as a substitute for the actual forecast init.
Pass the 12Z cycle explicitly: the already handed-off runner's optional cycle
default remains the earlier 00Z example. Producer runtime, binary and format
are unchanged; the workspace isolation change affects Cargo build layout only.
The producer CLI module is
`woof/hex/surface_series.py`; the update lane's reader locator is
`woof/hex/surface_series_reader.py`, avoiding a file ownership collision.

If no matching GFS init already exists, the generic init door can prepare it
using the existing shared grid and forecast static. `SHARED_GRID` and
`SHARED_STATIC` must name the files used by the node's passing smoke. Native
init and geometry bridges must already be installed or supplied through
`WOOF_HEX_RW_MPAS_INIT` and `WOOF_HEX_RW_MPAS_GEOMETRY`:

```sh
export WOOF_HEX_RW_SFC_SERIES=/path/to/rw_sfc_series
shared=${CONSUMER_INIT_DIR:?use the consumer-owned shared init directory}
mkdir -p "$shared"
printf '%s\n' '#!/bin/sh' 'exec "${WOOF_HEX_RW_SFC_SERIES:?}" --decode "$@"' \
  > "$shared/parallel_met_intermediate"
chmod +x "$shared/parallel_met_intermediate"
RAYON_NUM_THREADS=24 woof hex init-source --source gfs --cycle 2023-03-31T12 \
  --out-dir "$shared" --grid "$SHARED_GRID" --static "$SHARED_STATIC" \
  --vertical-spec "$HEX_SOURCE/src/hexcore/data/countries/tc55-v1.json" \
  --writer "$shared/parallel_met_intermediate" --jobs 16 --stop-after init
```

The decoder wrapper selects the bundled parallel Rust decoder. The init command
must use the full patched hex source or an installed version with the generic
`init-source` door. These are reproducible handoff commands, not a claim that
the reference GPU host preparation has run. The retained lane executable is x86-64 Linux with
a maximum required GLIBC symbol version of 2.39; rebuild with the source recipe
above on a node with an older C library. The portable source archive contains
this crate, builder, direct runner, table and Vtables; the builder also needs
the read-only engine dependency tree described above.

Source rows declare `surface_series.interval_hours`, `products`, and optional
`decode` overrides. GFS and CFS supply six-hourly analyses. ERA5 supports hourly
analyses; pass `--interval-hours 6` to request six-hourly records. CDS authentication
comes from `CDS_API_KEY` and optionally `CDS_API_URL`; the tool never opens credential
files. The CDS account must already have accepted the dataset terms. `--staged DIR`
uses product basenames from an existing shared prepared copy where available.

The output is native NetCDF with unlimited `Time`, `nCells`, `StrLen=64`,
`xtime(Time,StrLen)` containing padded `YYYY-MM-DD_HH:MM:SS`, float32
`sst(Time,nCells)` in K and `xice(Time,nCells)` in [0,1]. It includes optional
`snowh(Time,nCells)` in metres, `Time` in seconds from the first record, and
`record_provenance(Time,ProvenanceLen=4096)` containing padded JSON. The JSON
receipt repeats provenance. The surface-update reader consumes this file directly,
using its latest record at or before the model clock. No conversion is required.
Snow and provenance are accepted extras; the current update consumes SST and ice.
Fields retain the original forecast cell order.
The run filename agreed with the surface-update consumer is `surface_series.nc`.
A run window includes the initial timestamp and each cadence through its endpoint.
An endpoint between cadence boundaries includes the next record to cover the full
window. Choose a fresh output path for each invocation. The producer owns this
writer; the surface-update lane owns the `rw-surface` C ABI reader package.

`cfs` is an init-source table row, with `Vtable.CFS` and the existing Rust
`ncep-cdas-cfsv2` rules. It selects the NCEI CFSv2 operational pressure analysis,
including all four soil layers and near-surface RH. Forecasts select pressure and
flux products, with the source cycle in the directory and valid time in the filename.
The first guess contains 37 pressure levels plus the surface level. No CFS adapter
exists. Example:

```sh
woof hex init-source --source cfs --cycle 2023-03-31T00 --out-dir cfs-init \
  --grid grid.nc --static native.static.nc --vertical-spec tc55-v1.json
```

Earlier source cycles can provide extra inputs at the same valid time:

```sh
woof hex global --row global-120km --source gfs --cycle 2023-03-31T00 \
  --lagged 0,6,12,18 --work-dir lagged-run
woof hex global --row global-120km --source gefs --cycle 2023-03-31T00 \
  --members 1-5 --lagged 0,6,12,18 --work-dir twenty-run
```

For global, `--members` names native members and the lag list expands their
Cartesian product. A deterministic source needs only `--lagged`. For
`ensemble-prepare`, `--members 20` is the total requested count; members are balanced
over the selected lags, with one extra native member on the earliest selected lags
when the total is not divisible by the number of lags. Each input has a distinct directory and manifest provenance naming
`source`, `source_cycle`, `source_member`, `lead_hours`, `lag_hours`, and `valid_time`.
The earlier cycle's lead equals the base lead plus its lag. Unpublished cycles,
out-of-range leads and duplicate lags refuse before acquisition. Sources need a
`forecast` cadence and maximum lead to admit positive lags. `--source-hour` remains
available for ensemble preparation; the generic init-source door uses `--lead-hours`.

Measured source-init acceptance on the reference GPU host uses the generic CLI, Rust init and
geometry bridges, and Rust native static with GWD: 40,962 cells and 55 levels.
The CFS analysis at `2023-03-31T12` prepared in 13.824490974 seconds and completed
a 24-hour forecast with 288 steps at 300 seconds. Integration took
165.766243401 seconds including cold kernel compilation; command wall was
193.184295570 seconds. All 288 step-health records are finite, and the final
physical gate passes.

`ensemble-prepare --source gfs --source-cycle 2023-03-31T12 --members 4
--lagged 0,6,12,18 --jobs 4` prepared four 120 km inputs in 35.484128208 seconds
with six Rayon threads per member. Each earlier cycle's forecast is valid at
`2023-03-31T12`. All four theta and combined meteorology digests differ. Each
member then completed its own sequential 24-hour GPU forecast: 288 steps, two
history frames, and finite health checks throughout. Integration took
71.918869545, 72.952658760, 72.253815379 and 73.523855980 seconds. These are
sequential single-card runs from the manifest, with CUDA graphs disabled and
the default fast numerics, using gpuwm 2.9.0 and CuPy 14.2.0.

The lane retains exact commands, source-cycle/lead provenance, per-field hashes,
CPU/GPU samples and complete forecast receipts in `n1out/series-source-proofs.json`,
`n1out/series-source-acceptance.json`, `n1out/source-proof-receipts/` and
`receipts/source-acceptance-audit.json`. Prepared inputs, history frames, caches
and the temporary runtime were removed after verification. The first CFS run
exposed a runtime packaging issue: a CUDA 13 kernel pack was refused by the
CUDA 12 proof runtime, causing 212 inline compiles taking 98.699 seconds while
the GPU waited. This remains filed with the runtime owner in
`the cold-compile defect receipt`; the later member runs used the cache.
