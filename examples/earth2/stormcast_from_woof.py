"""StormCast-CONUS conditioned on a WOOF forecast instead of GFS.

StormCast-CONUS (NVIDIA Earth-2, weights nvidia/stormcast-conus on Hugging Face,
OpenMDW-1.1, no account) starts from the HRRR analysis and is driven hour by hour
by a coarse "conditioning" forecast, GFS by default. Here the conditioning is a
WOOF (WOOF-ArWen or hex) forecast exported on the HRRR grid with

    woof export RUN/wrfout --earth2 e2s-hrrr-3km --for stormcast-conus --out STORE

and read through Earth2Studio's own InferenceOutputSource. woof.earth2_io's
pregridded_conditioning() lets Earth2Studio 0.19 take a source that is already on
the HRRR grid (it otherwise asks for coordinate names its own HRRR source lacks).

usage: python stormcast_from_woof.py --init 2024-05-21T18 --hours 6 --cond STORE/d01.zarr --out refc.nc
       (--cond gfs runs the stock GFS-conditioned model for comparison;
        --window y0,y1,x0,x1 runs a sub-window of the HRRR grid on a smaller card)
Output: CF NetCDF of refc, t2m, u10m, v10m, msl by valid time, drawn with
    woof render --cf refc.nc --variable refc --valid T --observation MRMS_...grib2.gz
Needs: earth2studio[stormcast-conus]==0.19.0 (natten wheel matching torch), about 24 GB of
GPU memory for the full CONUS domain, 7.5 GB for a 512x768 window.
"""
import argparse, json, time
from datetime import datetime
import numpy as np, torch, xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True)
ap.add_argument("--hours", type=int, default=6)
ap.add_argument("--cond", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--window", default=None, help="y0,y1,x0,x1 HRRR index window (StormCastCONUS hrrr_lat_lim/hrrr_lon_lim)")
a = ap.parse_args()

from earth2studio.data import HRRR, GFS_FX, InferenceOutputSource
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import StormCastCONUS

t_start = time.time()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
import os
if os.environ.get("E2C_VRAM_MIB"):
    torch.cuda.set_per_process_memory_fraction(int(os.environ["E2C_VRAM_MIB"]) * 2**20 / torch.cuda.get_device_properties(0).total_memory)
init = np.datetime64(datetime.fromisoformat(a.init), "ns")
cond = GFS_FX() if a.cond == "gfs" else InferenceOutputSource(a.cond, engine="zarr")
pkg = StormCastCONUS.load_default_package()
kw = {}
if a.window:
    y0, y1, x0, x1 = map(int, a.window.split(","))
    kw = {"hrrr_lat_lim": (y0, y1), "hrrr_lon_lim": (x0, x1)}
model = StormCastCONUS.load_model(pkg, conditioning_data_source=cond, **kw).to(dev)
if a.cond == "gfs":
    model.conditioning_init_time = init - np.timedelta64(int(init.astype("datetime64[h]").astype(int) % 6), "h")
ic = model.input_coords()
x, coords = fetch_data(HRRR(), time=np.array([init]), variable=ic["variable"],
                       lead_time=np.array([np.timedelta64(0, "h")]), device=dev)
from earth2studio.utils.coords import map_coords
x, coords = map_coords(x, coords, ic)
t_load = time.time()
keep = ["refc", "t2m", "u10m", "v10m", "msl"]
frames, leads = [], []
import contextlib
ctx = contextlib.nullcontext()
if a.cond != "gfs":
    from woof.earth2_io import pregridded_conditioning  # export already on the HRRR grid
    ctx = pregridded_conditioning()
stack = contextlib.ExitStack(); stack.enter_context(ctx)
it = model.create_iterator(x, coords)
for step, (y, yc) in enumerate(it):
    vi = [list(yc["variable"]).index(k) for k in keep]
    frames.append(y[..., vi, :, :].detach().float().cpu().numpy().reshape(len(keep), *y.shape[-2:]))
    leads.append(yc["lead_time"][0])
    print(json.dumps({"step": step, "lead": str(yc["lead_time"][0]), "refc_max": float(frames[-1][0].max())}), flush=True)
    if step >= a.hours:
        break
t_end = time.time()
lat, lon = model.lat, model.lon
arr = np.stack(frames)
units = {"refc": ("dBZ", "equivalent_reflectivity_factor"), "t2m": ("K", "air_temperature"),
         "u10m": ("m s-1", "eastward_wind"), "v10m": ("m s-1", "northward_wind"), "msl": ("Pa", "air_pressure_at_mean_sea_level")}
valid = (init + np.array(leads).astype("timedelta64[ns]")).astype("datetime64[ns]")
label = "GFS" if a.cond == "gfs" else f"WOOF export {a.cond}"
ds = xr.Dataset({k: (("time", "y", "x"), arr[:, i].astype("float32"), {"units": units[k][0], "standard_name": units[k][1]}) for i, k in enumerate(keep)},
                coords={"time": ("time", valid, {"standard_name": "time", "axis": "T"}),
                        "lat": (("y", "x"), np.asarray(lat, "float32"), {"units": "degrees_north", "standard_name": "latitude"}),
                        "lon": (("y", "x"), np.asarray(lon, "float32"), {"units": "degrees_east", "standard_name": "longitude"})},
                attrs={"source": f"StormCast-CONUS conditioned on {label}", "seed": a.seed, "init": str(init)})
ds.to_netcdf(a.out)
print(json.dumps({"load_s": round(t_load - t_start, 1), "forecast_s": round(t_end - t_load, 1),
                  "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "out": a.out}))
