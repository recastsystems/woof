"""Compare sea-ice forecasts with daily NOAA/NIC IMS extent analyses.

NetCDF decoding/writing and IMS sampling use existing Rust bridges. CUDA
reductions form the metrics; Python handles paths, metadata and orchestration.
IMS is an extent classification, not observed ice thickness or concentration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def compare(history: Path, analysis: Path, out: Path, *, radius: int = 3,
            threshold: float = .15, render: bool = False,
            initial_history: Path | None = None) -> dict:
    import cupy as cp
    from woof.netcdf_bridge import Dataset
    from woof.ingest.lake_ice import sample_ims_codes
    from woof.io.nc_writer_bridge import ClassicSchema
    from woof.rustwx import find_renderer, run_cf_renderer
    if radius < 0 or not 0 < threshold < 1:
        raise ValueError('radius must be nonnegative and extent threshold between zero and one')
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    names = ('XLAT', 'XLONG', 'XLAND', 'LAKEMASK', 'SEAICE', 'ICETHICK',
             'ICEVOLUME', 'ICESNOWMASS', 'ICEENERGY', 'ICEENERGYERROR',
             'ICECONCCHANGE', 'ICETHICKCHANGE')
    with Dataset(history) as ds:
        ds.prefetch(names)
        # Rust supplies decoded planes; these reshape operations are views.
        host = {k: ds.variables[k][0] for k in names}
        valid = history.name.split('_', 2)[-1].replace('_', 'T', 1).replace('_', ':')+'Z'
    counts, _, sample_receipt = sample_ims_codes(analysis, host['XLAT'], host['XLONG'], radius=radius)
    cp.cuda.runtime.deviceSynchronize()
    f = {k: cp.asarray(v, dtype=cp.float32) for k, v in host.items()}
    codes = cp.asarray(counts).reshape((*f['SEAICE'].shape, 5))
    water_count = codes[..., 1] + codes[..., 3]
    # Subtract the accumulated area change to recover the initial ice mask.
    initial = f['SEAICE'] - f['ICECONCCHANGE']
    ocean = ((f['XLAND'] > 1.5) | (initial > 0)) & (f['LAKEMASK'] < .5)
    observed = ocean & (water_count > 0)
    pixel_fraction = codes[..., 3] / cp.maximum(water_count, 1)
    # Majority IMS water pixels define extent; the model uses the declared
    # operational extent threshold. No land/snow pixels count as ocean ice.
    obs_ice = pixel_fraction >= .5
    forecast_ice = f['SEAICE'] >= threshold
    def count(mask):
        return int(cp.count_nonzero(mask).item())
    tp = count(observed & obs_ice & forecast_ice)
    tn = count(observed & ~obs_ice & ~forecast_ice)
    fp = count(observed & ~obs_ice & forecast_ice)
    fn = count(observed & obs_ice & ~forecast_ice)
    n = tp+tn+fp+fn
    if not n:
        raise ValueError('IMS and the forecast have no overlapping non-lake ocean cells')
    persistence = initial >= threshold
    ptp = count(observed & obs_ice & persistence)
    ptn = count(observed & ~obs_ice & ~persistence)
    pfp = count(observed & ~obs_ice & persistence)
    pfn = count(observed & obs_ice & ~persistence)
    def stats(plane, mask):
        return dict(min=float(cp.min(plane[mask]).item()),
                    mean=float(cp.mean(plane[mask], dtype=cp.float64).item()),
                    max=float(cp.max(plane[mask]).item()))
    finite = all(bool(cp.all(cp.isfinite(f[k][ocean])).item()) for k in names[4:])
    bounded = bool(cp.all((f['SEAICE'][ocean] >= 0) & (f['SEAICE'][ocean] <= 1)).item())
    if not finite or not bounded:
        raise ValueError('nonfinite ice output or concentration outside [0,1]')
    receipt = dict(schema='arwen.seaice-extent-verification.v1', valid=valid,
        history=str(history), analysis=str(analysis),
        analysis_sha256=hashlib.sha256(analysis.read_bytes()).hexdigest(),
        reference='NOAA/NIC IMS G02156 Version 1 daily ice extent (public domain US government data)',
        reference_doi='10.7265/N52R3PMC', reference_is_not='thickness or fractional-concentration truth',
        sampling=sample_receipt | dict(radius_pixels=radius, land_snow_pixels_excluded=True),
        model_extent_threshold=threshold, ims_water_pixel_extent_threshold=.5,
        ocean_cells=count(ocean), matched_cells=n, true_positive=tp, true_negative=tn,
        false_positive=fp, false_negative=fn, accuracy=(tp+tn)/n,
        critical_success_index=tp/(tp+fp+fn) if tp+fp+fn else None,
        initial_extent_persistence=dict(true_positive=ptp, true_negative=ptn,
            false_positive=pfp, false_negative=pfn, accuracy=(ptp+ptn)/n,
            critical_success_index=ptp/(ptp+pfp+pfn) if ptp+pfp+pfn else None),
        forecast_ice_cells=tp+fp, observed_ice_cells=tp+fn,
        stats={k:stats(f[k], ocean) for k in names[4:]}, finite=finite, concentration_bounded=bounded,
        computational_backend=dict(netcdf='rw_netcdf/Rust', ims='gpuwm_preprocess_cpu/Rust', metrics='CUDA', writer='netcdf_writer/Rust'))
    if initial_history is not None:
        with Dataset(initial_history) as ds:
            ds.prefetch(('ICEVOLUME', 'ICESNOWMASS'))
            v0 = cp.asarray(ds.variables['ICEVOLUME'][0], cp.float64)
            s0 = cp.asarray(ds.variables['ICESNOWMASS'][0], cp.float64)
        actual = -333700*(917*(f['ICEVOLUME'].astype(cp.float64)-v0)
                          + f['ICESNOWMASS'].astype(cp.float64)-s0)
        closure = actual - f['ICEENERGY'].astype(cp.float64)
        receipt['solid_enthalpy_closure'] = dict(initial_history=str(initial_history),
            residual_j_m2=stats(closure, ocean),
            max_absolute_j_m2=float(cp.max(cp.abs(closure[ocean])).item()),
            max_relative_to_latent_energy_scale=float(cp.max(
                cp.abs(closure[ocean])/cp.maximum(
                    333700*(917*v0[ocean]+s0[ocean])+cp.abs(f['ICEENERGY'][ocean]), 1)).item()),
            relative_scale='initial solid latent energy plus absolute accumulated external enthalpy, minimum 1 J m-2')
    planes = dict(FORECASTICE=cp.where(observed, f['SEAICE'], cp.nan),
                  OBSERVEDICE=cp.where(observed, obs_ice.astype(cp.float32), cp.nan))
    nc = out/'extent.nc'
    ny, nx = f['SEAICE'].shape
    schema = ClassicSchema()
    y = schema.def_dim('y', ny); x = schema.def_dim('x', nx)
    t = schema.def_dim('time', 1)
    for key, value in dict(Conventions='CF-1.8', title='Thermodynamic sea ice and NOAA/NIC IMS extent',
        source='ERA5 initialized WOOF-ArWen forecast; NOAA/NIC IMS G02156 Version 1',
        references='https://doi.org/10.7265/N52R3PMC', time_coverage_start=valid).items():
        schema.put_global_attr(key, value)
    tv = schema.def_var('time', 'f8', (t,))
    schema.put_var_attr(tv, 'standard_name', 'time')
    schema.put_var_attr(tv, 'units', 'seconds since '+valid.replace('T', ' ').removesuffix('Z'))
    lat = schema.def_var('lat', 'f4', (y,x))
    lon = schema.def_var('lon', 'f4', (y,x))
    for var, standard, units in ((lat, 'latitude', 'degrees_north'), (lon, 'longitude', 'degrees_east')):
        schema.put_var_attr(var, 'standard_name', standard)
        schema.put_var_attr(var, 'units', units)
    variables = {}
    for key in planes:
        variables[key] = schema.def_var(key, 'f4', (t,y,x))
        schema.put_var_attr(variables[key], 'units', '1')
        schema.put_var_attr(variables[key], 'coordinates', 'lat lon')
        schema.put_var_attr(variables[key], 'long_name', key)
    with schema.create(nc) as writer:
        writer.write_var(tv, cp.asnumpy(cp.zeros(1, cp.float64)))
        writer.write_var(lat, cp.asnumpy(f['XLAT']))
        writer.write_var(lon, cp.asnumpy(f['XLONG']))
        for key, plane in planes.items():
            writer.write_var(variables[key], cp.asnumpy(plane))
    if render:
        west, east = float(cp.min(f['XLONG']).item()), float(cp.max(f['XLONG']).item())
        south, north = float(cp.min(f['XLAT']).item()), float(cp.max(f['XLAT']).item())
        bounds = f'{west},{east},{south},{north}' if east-west < 180 else None
        table = out/'cf-products.json'
        table.write_text(json.dumps([dict(names=[key], standard_names=[], product=key.lower(),
            title=title, recipe=None, default_units='1', display_units='1', unit_factors={'1':1.},
            contour_level=None, observation_product=None) for key,title in (
                ('FORECASTICE','Forecast ice concentration'), ('OBSERVEDICE','NOAA/NIC IMS ice extent'))], indent=2)+'\n')
        renderer = find_renderer()
        if renderer is None:
            raise RuntimeError("rw_wrfbatch is required for ice verification maps")
        for key in planes:
            rc = run_cf_renderer(renderer, nc, variable=key, out_dir=out/key.lower(),
                valid=valid, bounds=bounds, table=table, size=(1100,850), receipt=out/(key.lower()+'-render.json'))
            if rc:
                raise RuntimeError(f'Rust CF renderer failed for {key}: {rc}')
    receipt['wall_seconds'] = time.monotonic()-started
    receipt['final_used_gpu_pool_bytes'] = cp.get_default_memory_pool().used_bytes()
    (out/'verification.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=.15)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--initial-history', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.history, args.analysis, args.out, radius=args.radius,
                             threshold=args.threshold, render=args.render,
                             initial_history=args.initial_history), indent=2))

if __name__ == '__main__':
    main()
