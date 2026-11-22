# earth2studio-arwen

WOOF-ArWen (a CUDA port of WRF-ARW, package `woof`) and hex (a CUDA port of
MPAS-Atmosphere, package `woof hex`) for [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio).

- **`WoofHistory`** is an Earth2Studio data source over a woof or woof hex
  run. It reads the run's history through woof's Earth-2 exporter
  (`woof export --earth2`) in Earth2Studio's own variable names
  (`u10m`, `t2m`, `msl`, `z500`, `q850`, ...) on the 0.25 degree lattice
  Earth2Studio's global models use, so any Earth2Studio diagnostic,
  downscaler or tracker can run on a physics forecast.
- **`ArWen`** is an Earth2Studio prognostic model. Each step runs one WOOF-ArWen
  forecast over a latitude-longitude box, started from the state Earth2Studio
  hands it, with lateral boundaries from any Earth2Studio data source, and
  returns WOOF-ArWen's state in the same variables one boundary interval later.

Earth2Studio has no plugin registry; its extension points are the
`DataSource` and `PrognosticModel` protocols. Both classes satisfy them
(`isinstance(src, DataSource)`, `isinstance(model, PrognosticModel)`) and go
straight into `earth2studio.run.deterministic`, `diagnostic` and `ensemble`.

## Install

Earth2Studio and woof usually live in separate environments (each brings its
own CUDA stack). Install this package beside Earth2Studio and tell it where
woof is:

```sh
pip install earth2studio-arwen            # in the Earth2Studio environment
export WOOF_PYTHON=/path/to/gpuwm-env/bin/python   # where `pip install recast-woof[all-cu13]` ran
export GPUWM_CASE_DATA_ROOT=/path/to/case-data      # woof's static geography (WPS_GEOG) lives here
```

## A WOOF run as an Earth2Studio data source

```python
import earth2studio.run as run
from collections import OrderedDict
from earth2studio.io import NetCDF4Backend
from earth2studio.models.dx import WindgustAFNO
from earth2studio.models.px import Persistence
from earth2studio_arwen import WoofHistory

src = WoofHistory("hex-run/forecast", consumers="windgust-afno")   # channels WindgustAFNO needs
dx = WindgustAFNO.load_model(WindgustAFNO.load_default_package())
ic = dx.input_coords()
px = Persistence(list(ic["variable"]), OrderedDict(lat=ic["lat"], lon=ic["lon"]))
times = ["2026-10-17T00", "2026-10-17T06", "2026-10-17T12", "2026-10-17T18"]
run.diagnostic(times, 0, px, dx, src, NetCDF4Backend("gust.nc"))
```

`consumers` names rows of woof's Earth-2 table (`windgust-afno`,
`precip-afno`, `climatenet`, `tc-tracker`, `stormcast-conus`); `channels` adds
any other Earth2Studio variable id. The export runs once per run folder and
is cached beside it.

## WOOF-ArWen as an Earth2Studio prognostic model

```python
import earth2studio.run as run
from earth2studio.data import ARCO
from earth2studio.io import NetCDF4Backend
from earth2studio_arwen import WOOF-ArWen

model = WOOF-ArWen((30, -105, 44, -89), boundary=ARCO(), dx_km=3.0, margin_deg=2.0)
run.deterministic(["2024-05-21T12"], 1, model, ARCO(), NetCDF4Backend("arwen.nc"))
```

The state is z, t, u, v, q on 13 pressure levels plus t2m, u10m, v10m and
msl (the variables woof's `earth2-pl13-q-netcdf-v1` profile reads), on the
0.25 degree points of the box. woof fits a WOOF-ArWen domain inside the box,
takes the land surface from the ERA5 analysis at the initialization (keyless),
and runs the forecast on the GPU. Inside the domain the output is WOOF-ArWen's
forecast; the ring between the domain and the box edge is the boundary
source's state at the output time. A run whose inputs are byte-identical to a
finished one is reused.

## Licence

Apache-2.0. Earth2Studio and woof are Apache-2.0; this package calls them
through their public interfaces and copies no code from either. Model weights
an Earth2Studio model downloads carry their own licences.
