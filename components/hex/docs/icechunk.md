# Icechunk forecast output

Use `--output-format icechunk` to retain native history in an atomic Zarr v3 store alongside the ordinary netCDF output. The writer and output-format table ship with woof and are shared by both engines.

```sh
woof hex global --row global-120km --cycle latest --hours 1 \
  --work-dir run --output-format icechunk
woof hex forecast --mesh MESH --grid grid.nc --static static.nc --init init.nc \
  --hours 1 --out forecast --scratch scratch --output-format icechunk
```

`forecast` (including regional and point-mesh routes), `global`, `ensemble`, `moving run`, `moving fixed`, `moving-ensemble`, `pair`, `regional`, `country`, `mesh-plan --run`, `global-cycle`, `s2s` and `add-nest` accept `--output-format`, `--icechunk-store` and `--output-config`. Output selection follows the forecast into child processes. Use a base such as `--icechunk-store s3://bucket/forecast` for S3, or a local path. Ensemble members and moving segments have independent repositories. The per-forecast `output-formats/receipt.json` lists exact store URLs and snapshots. An `add-nest` child writes its WOOF domain repository.

A common configuration file works on both engines:

```toml
[output]
formats = ["netcdf", "icechunk"]
# icechunk_store = "s3://bucket/forecast"
```

Pass it with `--output-config outputs.toml`; CLI values override the file. Native MPAS arrays keep their cell, edge and vertical dimensions, names and numeric precision. Static coordinates gain time so each moving frame retains its location. There is one atomic commit per complete output time. Resume skips identical committed times and successful completion tags the final snapshot.

```python
import icechunk
import xarray as xr

repo = icechunk.Repository.open(
    icechunk.local_filesystem_storage("run/forecast/icechunk/mesh")
)
ds = xr.open_zarr(repo.readonly_session("main").store,
                  consolidated=False, mask_and_scale=False)
```

For S3-compatible storage:

```python
import os
import icechunk
import xarray as xr

endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", os.environ.get("AWS_ENDPOINT_URL"))
repo = icechunk.Repository.open(icechunk.s3_storage(
    bucket="bucket", prefix="forecast/mesh", from_env=True,
    endpoint_url=endpoint, force_path_style=endpoint is not None,
    allow_http=bool(endpoint and endpoint.startswith("http://")),
))
ds = xr.open_zarr(repo.readonly_session("main").store,
                  consolidated=False, mask_and_scale=False)
```

Install `icechunk>=2.3.1,<3`, `zarr>=3.1` and `xarray` for reading. Forecast writing uses the bundled Rust `rw_icechunk`, found through woof's normal bridge resolver. `woof doctor` checks it; `woof.output_formats.capabilities()` reports installed row availability. The default remains netCDF. See the woof manual's `ICECHUNK.md` for snapshot reads, full layout, receipts and source-build instructions.
