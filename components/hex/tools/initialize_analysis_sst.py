"""Initialize a fixed-SST and PWP pair from the same dated profile analysis.

Run once before either forecast. The owned init artifact is replaced atomically;
source and result digests are retained. No forecast state or guards are patched.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, time
import numpy as np

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--start', required=True)
    p.add_argument('--source', default='hycom', choices=['hycom', 'rtofs', 'cf'])
    p.add_argument('--levels', type=int, default=40)
    p.add_argument('--layer-m', type=float, default=5.)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    import cupy as cp
    from netCDF4 import Dataset
    from woof.ocean.mixed_layer.analysis import load
    def digest(path):
        with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
    before = time.perf_counter()
    original = digest(a.init)
    with Dataset(a.init) as ds:
        lat = cp.asarray(ds['latCell'][:].ravel(), dtype=cp.float32)
        lon = cp.asarray(ds['lonCell'][:].ravel(), dtype=cp.float32)
        wet = cp.asarray(ds['landmask'][:].ravel() == 0)
        wet &= cp.asarray(ds['xice'][:].ravel() == 0)
    land = cp.where(wet, cp.float32(2), cp.float32(1))
    mapped = load(a.analysis, latitude=cp.rad2deg(lat), longitude=cp.rad2deg(lon),
                  land=land, levels=a.levels, layer_m=a.layer_m, source=a.source,
                  rotation=(cp.zeros_like(land), cp.ones_like(land)),
                  expected_time=a.start, uncovered='prescribed')
    active = cp.asnumpy(mapped['om_active'] > .5)
    surface = cp.asnumpy(mapped['om_tmp'][0])
    temporary = a.init.with_suffix('.analysis-sst.part.nc')
    if temporary.exists(): raise FileExistsError(temporary)
    shutil.copyfile(a.init, temporary)
    with Dataset(temporary, 'r+') as ds:
        old = np.asarray(ds['sst'][:])
        new = old.copy().reshape(-1)
        new[active] = surface[active]
        ds['sst'][:] = new.reshape(old.shape)
        ds.ocean_surface_analysis = str(a.analysis)
        ds.ocean_surface_analysis_sha256 = digest(a.analysis)
    os.replace(temporary, a.init)
    a.receipt.write_text(json.dumps(dict(schema='paired-ocean-initial-sst-v1',
        init=str(a.init), analysis=str(a.analysis), source_sha256=digest(a.analysis),
        original_init_sha256=original, initialized_sha256=digest(a.init),
        active_cells=int(active.sum()), unchanged_cells=int((~active).sum()),
        max_change_k=float(np.max(np.abs(new-old.reshape(-1)))),
        wall_seconds=time.perf_counter()-before,
        policy='same conservative top-layer mean as PWP; uncovered and icy cells unchanged'), indent=2))
if __name__ == '__main__': main()
