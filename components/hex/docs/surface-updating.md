# Surface series on the ArWen physics route

A forecast discovers `surface_series.nc` in its output directory or the
containing run directory. A series covering the complete forecast window
turns on SST, fractional sea ice, the MPAS diurnal SST skin model, monthly
greenness and background albedo, and deep soil temperature updating.
Without a series, the physics receipt records `enabled: false` and the
reason. An incomplete or malformed series is an input error.

The file has native MPAS axes `Time`, `nCells`, and `StrLen=64`:

| Variable | Type and dimensions |
| --- | --- |
| `xtime` | character `(Time, StrLen)`, padded `YYYY-MM-DD_HH:MM:SS` |
| `sst` | float32 `(Time, nCells)`, K |
| `xice` | float32 `(Time, nCells)`, fraction |

Records have a uniform positive interval, normally six hours. Cells follow
the original init file order. The consumer gathers reordered and partitioned
columns on the GPU. It reads monthly `greenfrac`, `albedo12m`, and `landmask`
from the native init file.

Optional `snowh` and per-record provenance JSON sidecars or character
variables may accompany the series. This consumer reads only `sst` and
`xice` from each record; it does not apply the optional snow field.

The implementation shares `SurfaceAlarms`, `MonthlyCalendar`, the Rust
record reader, and the CUDA surface kernels with the MPAS physics route.
The analysis alarm selects the latest record at or before the model clock.
It runs before a coincident history capture, so a daily SST history contains
that day's analysis. Physics execution receipts retain the category census
used by the preceding physics call when ice changes at that boundary.

Each record is decoded by the source-built Rust `rw-surface` library and
uploaded as one packed SST/ice transfer. Field interpolation, ice category
and soil conversions, skin SST, and deep soil updates run on the GPU.
The receipt records each read's decode time, transfer enqueue time and bytes.
Its zero device-to-host reader count concerns record ingestion; category
validation separately reduces six device counts to the host per alarm.

The public wheel carries `woof/hex/libexec/librw_surface.so` on Linux, built
by `tools/build_engine_bridges.sh` from `tools/rustwx/crates/rw-surface`.
The bridge build uses the pinned static NetCDF/HDF5/zlib archive. Development
source builds require NetCDF-C development files. `WOOF_HEX_SURFACE_READER`
can explicitly select a source-built library for a proof.

Restart state includes the alarm phase, monthly calendar, skin and deep soil
stores, and stream clock identity. A changed stream clock or column extent
is rejected. Fixed-mesh whole-column, partitioned same-host, and full-step
LTS drivers bind the service. Moving-mesh surface state transfer has not
been validated by the ten-day fixed-mesh proof.
