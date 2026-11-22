"""Build woof/data/geog_repairs/gmted2010_e180_glo30.npz (GMTED EAST OF 180, woof.geog_assets): every MODIS 20-class
land pixel that the WPS_GEOG GMTED2010 30" terrain stores as 0 m in its tile column east of 180 (x 21601-22800, 180 to
190 E) gets the mean Copernicus GLO-30 height of its 30" cell (an absent GLO-30 tile is water: left out).

usage: python tools/build_gmted_e180_patch.py <WPS_GEOG root> <out.npz>      (needs rasterio and network access to the
public copernicus-dem-30m bucket; about 10 minutes)"""
import sys, os, math, json, urllib.request
import numpy as np
import rasterio
from rasterio.windows import from_bounds
G = os.path.expanduser(sys.argv[1])
out = sys.argv[2]
xs, ys = [], []
for j in range(18):
    y = f"{j*1200+1:05d}-{j*1200+1200:05d}"
    fe, fl = f"{G}/topo_gmted2010_30s/21601-22800.{y}", f"{G}/modis_landuse_20class_30s/00001-01200.{y}"
    if not (os.path.exists(fe) and os.path.exists(fl)):
        continue
    E = np.fromfile(fe, ">i2").reshape(1206, 1206)[3:-3, 3:-3]
    L = np.fromfile(fl, np.uint8).reshape(1200, 1200)
    r, c = np.nonzero((L != 17) & (L != 0) & (E == 0))
    xs.append(21601 + c); ys.append(j * 1200 + 1 + r)
X, Y = np.concatenate(xs), np.concatenate(ys)
d = 1 / 120.0
lon_w = (X - 1) * d + 0.004166667 - d / 2 - 360.0      # cell west edge in -180..180 (x=1 centre at 0.00417 E)
lat_s = (Y - 1) * d - 89.99583 - d / 2
H = np.zeros(X.size, np.int16)
tiles = {}
for i in range(X.size):
    key = (math.floor(lat_s[i] + 1e-9), math.floor(lon_w[i] + 1e-9))
    tiles.setdefault(key, []).append(i)
stat = {"pixels": int(X.size), "tiles": len(tiles), "absent": 0, "filled": 0}
for (la, lo), idx in sorted(tiles.items()):
    ns = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"; ew = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    url = f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    try:
        ds = rasterio.open(url)
    except Exception:
        stat["absent"] += 1
        continue
    with ds:
        a = ds.read(1).astype(np.float64)
        tr = ds.transform
        for i in idx:
            w = from_bounds(lon_w[i], lat_s[i], lon_w[i] + d, lat_s[i] + d, tr).round_offsets().round_lengths()
            r0, c0 = max(int(w.row_off), 0), max(int(w.col_off), 0)
            blk = a[r0:r0 + max(int(w.height), 1), c0:c0 + max(int(w.width), 1)]
            blk = blk[np.isfinite(blk) & (blk > -1000)]
            if blk.size:
                H[i] = int(round(max(blk.mean(), 0.0)))
    print(name, len(idx), "max", int(H[idx].max()), flush=True)
keep = H > 0
stat["filled"] = int(keep.sum())
np.savez_compressed(out, x=X[keep].astype(np.int32), y=Y[keep].astype(np.int32), h=H[keep])
print(json.dumps(stat))
