# Native history in Icechunk

Add an Icechunk store to a forecast with `--output-format icechunk`. The normal netCDF history remains available. A separate Rust process reads each published history frame, writes lossless Zarr v3 arrays, and commits the whole valid time in one Icechunk snapshot. The model continues stepping while this happens. The command drains the writer and reports export failures before returning.

```sh
woof go forecast.toml --output-format icechunk
woof run forecast.toml --outdir run --output-format icechunk
woof sim prepared --experiment-config forecast.toml --outdir run --output-format icechunk
woof resume forecast.toml --outdir run --output-format icechunk
```

`go`, `run`, `sim`, `resume`, `branch`, `downscale` and `ensemble` accept the same output options. An experiment can select them without CLI flags:

```toml
[output]
formats = ["netcdf", "icechunk"]
# Optional base directory or S3 prefix:
# icechunk_store = "/data/forecast-stores"
```

Both engines also accept `--output-config outputs.toml` containing this block. Explicit CLI values override the file. The default row remains `netcdf`. The packaged `woof/data/output_formats.json` declares the writers and their chunk policies; woof hex reads that same table.

The table's `scheduling` block keeps the writer below the model's CPU priority: nice +10 on POSIX, or below-normal priority on Windows. Where `ionice` is available it selects best-effort I/O priority 7. `cpu_fraction = 0.5` caps both Rayon and Tokio worker pools at half the available CPUs (at least one), including process affinity limits. Smaller inherited `RAYON_NUM_THREADS` or `TOKIO_WORKER_THREADS` settings are respected. These settings apply to the writer process only; the command still waits for its final output drain before returning.

## Repositories and arrays

Each history directory gains `icechunk/d01`, `icechunk/d02`, and so on. Native hex cell history uses `icechunk/mesh`. Each ensemble member and each chained segment has its own repositories beneath its normal output directory. Selecting a store retains WOOF ensemble member history instead of its usual aggregate-only output.

`--icechunk-store /data/stores` or `--icechunk-store s3://bucket/prefix` sets a base. The writer appends relative member and segment directories, then the domain. Intermediate directory names `run`, `history`, `wrfout` and `chain` are omitted. The receipt lists the exact repository URLs, so consumers need not reconstruct the directory layout. A simple `run --outdir run --icechunk-store /data/stores` writes `/data/stores/d01`.

All native numeric fields retain their names, precision, units, grid staggering and level dimensions. There is no interpolation. Native `Time` becomes `time`; static fields also gain `time` so a moving domain keeps each frame's own coordinates. Native character arrays become uint8 character bytes. The derived time coordinate has CF epoch units and a calendar; projected WRF output gains a CF `crs` grid mapping. Original global attributes and field inventories for each frame are retained in the root's `icechunk_output.frames` metadata. A field absent at an intermediate output time has an unwritten fill chunk.

Compression is lossless zstd level 3. Time chunks contain one frame; spatial and vertical chunk sizes come from the table, clipped to each array's dimensions. Keep the same policy when resuming a repository.

## Read a local store

Install `icechunk>=2.3.1,<3`, `zarr>=3.1` and `xarray` in the reader environment. These are not needed by the forecast writer.

```python
import icechunk
import xarray as xr

repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage("run/icechunk/d01")
)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False, mask_and_scale=False)
# Native model levels and grid dimensions:
map_field = ds["T"].isel(time=0, bottom_top=0)
ml_tile = ds["T"].isel(time=0, south_north=slice(0, 64), west_east=slice(0, 64))
```

`mask_and_scale=False` preserves the native stored numbers, including integer flags and fill values. Omit it when CF masking and scale decoding are wanted. Coordinates and time still decode normally. For a raw numeric comparison to netCDF, also pass `decode_times=False`. Independent aligned members can be combined with `xr.concat(member_datasets, dim="member")`.

## Write and read S3

```sh
woof go forecast.toml --output-format icechunk --icechunk-store s3://bucket/forecast
```

The writer uses the standard AWS SDK credential environment and profile chain. For an S3-compatible endpoint, set `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL`. An explicit endpoint uses path-style addressing; an `http://` endpoint permits HTTP for a local test server. Credentials are not placed in requests or receipts.

```python
import os
import icechunk
import xarray as xr

endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", os.environ.get("AWS_ENDPOINT_URL"))
storage = icechunk.s3_storage(
    bucket="bucket", prefix="forecast/d01", from_env=True,
    endpoint_url=endpoint, force_path_style=endpoint is not None,
    allow_http=bool(endpoint and endpoint.startswith("http://")),
)
repo = icechunk.Repository.open(storage)
ds = xr.open_zarr(repo.readonly_session("main").store,
                  consolidated=False, mask_and_scale=False)
```

## Snapshots, resume and receipts

The branch is `main`. Each frame creates one commit named `frame <UTC valid time>`. A reader pinned to a snapshot sees only that snapshot's complete frames. Repeated times with identical values and metadata are skipped when a run resumes. A changed committed time or an out-of-order insertion is refused rather than overwriting history. Successful completion adds an immutable `final-<snapshot ID>` tag. A later resumed run can create a new final tag while the old tag remains readable. `go --restart CHECKPOINT` reuses an existing local store beside the checkpoint under `wrfout/icechunk`; for S3, keep the same configured store base. Prepared continuation directories named `chain` do not create another repository.

```python
first = xr.open_zarr(repo.readonly_session(snapshot_id=first_snapshot_id).store,
                    consolidated=False, mask_and_scale=False)
finished = xr.open_zarr(repo.readonly_session(tag=final_tag).store,
                       consolidated=False, mask_and_scale=False)
```

`RUN/output-formats/progress.jsonl` records committed frames during the run. `receipt.json` beside it records the output stage, store URLs, snapshot IDs, final tags and writer wall times. A failed model does not finalize its store. A failed writer makes the requested output stage fail and retains already committed snapshots.

## Installation and capability

The platform wheel stages `rw_icechunk` beside the other Rust bridges. `woof doctor` checks its contract. The installed interpreter can report the table and writer availability:

```python
from woof.output_formats import capabilities
print(capabilities())
```

A row with `id="icechunk"` and `available=True` means that an ABI-compatible writer is installed. Missing binaries cause an early refusal when the row is selected. Source builds use their own Cargo workspace and lockfile, keeping S3 dependencies out of the other tools. Its crates come from the vendored registry, so build from inside the crate (that is where Cargo finds `.cargo/config.toml`) and no network is needed:

```sh
(cd tools/rw_icechunk && cargo build --release --locked --offline)
export WOOF_RW_ICECHUNK="$PWD/tools/rw_icechunk/target/release/rw_icechunk"
```

The writer requires Rust 1.94 or newer. Its dependency notices ship in `licenses/RW-ICECHUNK-THIRD-PARTY.txt`.
