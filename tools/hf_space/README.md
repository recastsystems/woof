---
title: WOOF weather samples
emoji: 🌦️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: "3.13"
app_file: app.py
license: apache-2.0
---
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WOOF weather samples

Browse forecasts from WOOF-ArWen, the CUDA WRF-ARW model (`woof`), and hex,
the CUDA MPAS model (`woof hex`). Choose a sample, variable and lead time.
Maps are drawn by the engine's Rust renderer and saved before you open the viewer.

| Sample | Model grid | Initial conditions | Valid times |
| --- | --- | --- | --- |
| WOOF-ArWen CONUS 3 km | 796 × 636 cells, 49 levels, 3 km | GFS, 17 October 2026 at 00 UTC | 00 and 01 UTC |
| hex 60 km global | 163,842 cells, 55 levels, nominal 60 km | GFS, 17 October 2026 at 00 UTC | 00 and 01 UTC |

The WOOF-ArWen sample covers part of CONUS. Its display grid is 602 × 1065 at
0.03 degree, about 3.336 km. Points outside the source domain are masked.
The hex display grid is 664 × 333 at about 60.277 km. Display grids describe
the saved maps; the table gives each model's grid.

The Soundings tab shows WOOF-ArWen profiles at 35.30 N, 97.50 W and
40.00 N, 100.00 W. Each window includes a skew-T, hodograph, wind barbs
and diagnostic tables. Open fullscreen to read the tables. Profile and
diagnostic JSON is available to download.

## Run your own

Create a configuration:

```sh
woof domain --point=35.3,-97.5 --card 24gb --ladder 12 \
  --source gfs --cycle latest --hours 6 \
  --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 \
  --out configs/myarea.toml
```

Launch the forecast in your CUDA-capable engine environment:

```sh
woof go configs/myarea.toml
```

Create a sounding from your run:

```sh
woof soundings ./run --point point-a=35,-97 --mode det --out ./render
```

## Run the viewer

Use Python 3.13 and install `requirements.txt`. Generate the images and profile
JSON with `prepare_assets.py` before starting the viewer. Assets are generated
separately and are not stored with the viewer code. The builder reads a local
source table and uses the engine's Rust map and sounding renderers. Building
surface tensor views also requires NumPy and netCDF4 1.7.4.

`catalog.json` lists the samples and relative asset paths. The app displays
these saved files. It does not run a forecast or render maps on request.

```sh
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7861 python app.py
```

Open `http://127.0.0.1:7861`.

The viewer and sample graphics are licensed under Apache-2.0.
See `LICENSE`, `NOTICE-arwen` and `NOTICE-hex` for attribution.
