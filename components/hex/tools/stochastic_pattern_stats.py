"""Measure the stochastic pattern's length scale, decorrelation time and spectrum on the GPU.

    PYTHONPATH=src python tools/stochastic_pattern_stats.py --grid x1.40962.grid.nc \
        [--lam-grid lam.grid.nc] --length-km 450 --time-h 6 --out stats.json [--plots DIR]

The requested pattern law (woof.hex.stochastic) is a spatial correlation
exp(-r**2/(2 L**2)) and a lag correlation exp(-dt/tau). This tool draws
independent realisations of the GPU pattern at mesh cells and measures:

* point variance (should be 1 everywhere, boundaries included);
* the correlation against great-circle distance from random cell pairs, the
  distance where it falls to exp(-1/2) (the measured L), and a least-squares
  Gaussian fit, beside the exact truncated Legendre series;
* the angular power spectrum of a pattern projected back on the global mesh
  (area-weighted quadrature), against the requested spectrum;
* the lag autocorrelation of an AR(1) run and the time where it falls to 1/e;
* on-the-fly synthesis against a stored-basis matrix product (same words up to
  float rounding) and the synthesis kernel time.

Every array operation runs on the GPU (CuPy); the host only bins small
summaries. Charts are matplotlib analysis charts, not weather fields.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _mesh_points(path, cp):
    from woof.hex.mesh import Mesh
    mesh = Mesh.from_netcdf(path, validate=False)
    a = mesh.arrays
    n = int(mesh.n_cells)
    out = dict(mesh=mesh, lat=cp.asarray(a['latCell'][:n], dtype=cp.float64),
               lon=cp.asarray(a['lonCell'][:n], dtype=cp.float64),
               area=cp.asarray(a['areaCell'][:n], dtype=cp.float64))
    if 'bdyMaskCell' in a:
        out['bdy'] = cp.asarray(a['bdyMaskCell'][:n])
    return out


def _theory(cp, length_m, degree, radius, distances):
    """Exact correlation of the truncated spectrum at the given distances (Legendre series)."""
    from woof.hex.stochastic import spectral_exponent
    a = spectral_exponent(length_m, radius)
    ell = cp.arange(1, degree + 1, dtype=cp.float64)
    c = (2*ell + 1)*cp.exp(-(ell*(ell + 1) - 2)*a)
    x = cp.cos(cp.asarray(distances, dtype=cp.float64)/radius)
    p_prev, p_cur = cp.ones_like(x), x.copy()
    total = c[0]*p_cur
    for l in range(1, degree):
        p_next = ((2*l + 1)*x*p_cur - l*p_prev)/(l + 1)
        p_prev, p_cur = p_cur, p_next
        total += c[l]*p_cur
    return total/cp.sum(c)


def spatial_stats(cp, points, length_m, *, samples, pairs, seed, label, degree=0, max_km=None):
    from woof.hex.stochastic import PointHarmonics, SphericalPattern, StochasticConfig, pattern_degree
    cfg = StochasticConfig(sppt=True, member=1, seed=seed, length_m=length_m, time_s=3600., degree=degree)
    deg = pattern_degree(cfg, length_m)
    resource = PointHarmonics(points['lat'], points['lon'], deg)
    pattern = SphericalPattern(resource, cfg, channels=1)
    n = resource.cells
    rng = cp.random.default_rng(seed)
    i = rng.integers(0, n, pairs)
    j = rng.integers(0, n, pairs)
    lat, lon = points['lat'], points['lon']
    cosd = (cp.sin(lat[i])*cp.sin(lat[j]) + cp.cos(lat[i])*cp.cos(lat[j])*cp.cos(lon[i] - lon[j]))
    dist = resource.radius*cp.arccos(cp.clip(cosd, -1, 1))
    max_m = (max_km*1000. if max_km else 4*length_m)
    bins = 60
    width = max_m/bins
    which = cp.floor(dist/width).astype(cp.int64)
    keep = which < bins
    which = cp.where(keep, which, bins)
    sxy = cp.zeros(bins + 1); sxx = cp.zeros(bins + 1); syy = cp.zeros(bins + 1)
    count = cp.bincount(which, minlength=bins + 1).astype(cp.float64)
    var_sum = cp.zeros(n); mean_sum = cp.zeros(n)
    started = cp.cuda.Event(); ended = cp.cuda.Event()
    synth_ms = []
    for k in range(samples):
        started.record()
        # 100 decorrelation times apart: independent draws (phi = exp(-100)).
        field = pattern.advance(k*100*cfg.time_s)[:, 0].astype(cp.float64)
        ended.record(); ended.synchronize()
        synth_ms.append(cp.cuda.get_elapsed_time(started, ended))
        x, y = field[i], field[j]
        sxy += cp.bincount(which, weights=x*y, minlength=bins + 1)
        sxx += cp.bincount(which, weights=x*x, minlength=bins + 1)
        syy += cp.bincount(which, weights=y*y, minlength=bins + 1)
        var_sum += field*field; mean_sum += field
    corr = (sxy/cp.sqrt(sxx*syy))[:bins]
    centers = (cp.arange(bins) + 0.5)*width
    theory = _theory(cp, length_m, deg, resource.radius, centers)
    gaussian = cp.exp(-centers**2/(2*length_m**2))
    ok = count[:bins] > 50
    corr_h, centers_h = cp.asnumpy(corr), cp.asnumpy(centers)
    # Measured L: first crossing of exp(-1/2), linear interpolation.
    target = math.exp(-0.5)
    measured = None
    for b in range(1, bins):
        if corr_h[b - 1] >= target > corr_h[b]:
            f = (corr_h[b - 1] - target)/(corr_h[b - 1] - corr_h[b])
            measured = float(centers_h[b - 1] + f*(centers_h[b] - centers_h[b - 1]))
            break
    # Least-squares Gaussian length over d <= 2L where correlation is positive.
    sel = ok & (centers <= 2*length_m) & (corr > 0.05)
    d2 = centers[sel]**2; lnr = -cp.log(corr[sel])
    fit = float(cp.sqrt(cp.sum(d2*d2)/(2*cp.sum(d2*lnr)))) if int(cp.sum(sel)) > 2 else None
    variance = var_sum/samples - (mean_sum/samples)**2
    out = dict(label=label, points=n, degree=deg, samples=samples, pairs=pairs,
               requested_length_km=length_m/1000,
               measured_length_km_at_exp_minus_half=None if measured is None else measured/1000,
               gaussian_fit_length_km=None if fit is None else fit/1000,
               point_variance=dict(mean=float(cp.mean(variance)), p01=float(cp.percentile(variance, 1)),
                                   p99=float(cp.percentile(variance, 99)),
                                   expected_sampling_sd=math.sqrt(2/samples)),
               max_abs_corr_minus_theory=float(cp.max(cp.abs(corr - theory)[ok])),
               synthesis_ms_median=sorted(synth_ms[1:])[len(synth_ms[1:])//2] if len(synth_ms) > 1 else None,
               curve=dict(distance_km=[float(v)/1000 for v in centers_h],
                          measured=[float(v) for v in corr_h],
                          theory=[float(v) for v in cp.asnumpy(theory)],
                          gaussian=[float(v) for v in cp.asnumpy(gaussian)],
                          pairs=[int(v) for v in cp.asnumpy(count[:bins])]))
    if 'bdy' in points:
        # Variance by boundary row: on-the-fly harmonics have no boundary inflation.
        rows = {}
        for row in range(0, 8):
            m = points['bdy'] == row
            if int(cp.sum(m)):
                rows[str(row)] = float(cp.mean(variance[m]))
        out['point_variance_by_bdyMaskCell'] = rows
    return out, pattern, resource


def spectrum_stats(cp, points, length_m, *, samples, seed):
    """Angular power spectrum of patterns projected back with area-weighted quadrature."""
    from woof.hex.stochastic import PointHarmonics, SphericalPattern, StochasticConfig, pattern_degree, spectral_exponent
    cfg = StochasticConfig(sppt=True, member=2, seed=seed, length_m=length_m, time_s=3600.)
    deg = pattern_degree(cfg, length_m)
    resource = PointHarmonics(points['lat'], points['lon'], deg)
    pattern = SphericalPattern(resource, cfg, channels=1)
    weights = (points['area']/cp.sum(points['area'])*4*math.pi).astype(cp.float32)
    basis = resource.basis
    power = cp.zeros(deg)
    for k in range(samples):
        field = pattern.advance(k*100*cfg.time_s)[:, 0]
        est = basis.T @ (weights*field)
        band = cp.bincount((resource.ell - 1).astype(cp.int64), weights=(est.astype(cp.float64))**2,
                           minlength=deg)
        power += band
    power /= samples
    ell = cp.arange(1, deg + 1, dtype=cp.float64)
    a = spectral_exponent(length_m, resource.radius)
    target = (2*ell + 1)*cp.exp(-(ell*(ell + 1) - 2)*a)
    target *= 4*math.pi/cp.sum(target)
    # Synthesis check: on-the-fly harmonic_sum against the stored-basis product.
    direct = basis @ pattern.coefficients[:, 0]
    diff = float(cp.max(cp.abs(direct - pattern.pattern[:, 0])))
    scale = float(cp.max(cp.abs(direct)))
    rel = cp.abs(power - target)/target
    big = target > 0.01*cp.max(target)
    return dict(degree=deg, samples=samples,
                power_by_degree=[float(v) for v in cp.asnumpy(power)],
                target_by_degree=[float(v) for v in cp.asnumpy(target)],
                max_relative_error_where_target_above_1pct_of_peak=float(cp.max(rel[big])),
                expected_sampling_relative_sd_low_degree=math.sqrt(2/(3*samples)),
                on_the_fly_vs_stored_basis=dict(max_abs_difference=diff, max_abs_value=scale))


def temporal_stats(cp, points, length_m, time_s, *, dt, steps, seed, max_lag):
    from woof.hex.stochastic import PointHarmonics, SphericalPattern, StochasticConfig, pattern_degree
    cfg = StochasticConfig(sppt=True, member=3, seed=seed, length_m=length_m, time_s=time_s)
    deg = pattern_degree(cfg, length_m)
    resource = PointHarmonics(points['lat'], points['lon'], deg)
    pattern = SphericalPattern(resource, cfg, channels=1)
    series = cp.empty((steps, resource.cells), dtype=cp.float32)
    for k in range(steps):
        series[k] = pattern.advance(k*dt)[:, 0]
    # The process mean is zero by construction. Removing each point's sample
    # mean would bias every lag low by about 2*tau/T (0.1 for T = 20 tau).
    var = cp.mean(series*series)
    lags = list(range(0, max_lag + 1))
    acf = [float(cp.mean(series[:steps - lag]*series[lag:])/var) for lag in lags]
    target = 1/math.e
    measured = None
    for k in range(1, len(acf)):
        if acf[k - 1] >= target > acf[k]:
            f = (acf[k - 1] - target)/(acf[k - 1] - acf[k])
            measured = (k - 1 + f)*dt
            break
    lag1 = acf[1]
    return dict(dt_s=dt, steps=steps, requested_time_h=time_s/3600,
                measured_time_h_at_exp_minus_one=None if measured is None else measured/3600,
                lag1=lag1, expected_lag1=math.exp(-dt/time_s),
                time_from_lag1_h=(-dt/math.log(lag1))/3600 if 0 < lag1 < 1 else None,
                acf=dict(lag_h=[lag*dt/3600 for lag in lags], measured=acf,
                         expected=[math.exp(-lag*dt/time_s) for lag in lags]))


def synthesis_timing(cp, points_count, degree, repeats=5):
    """harmonic_sum kernel time at a RRFS conus3km-sized point count (cells+vertices)."""
    from woof.hex.stochastic import PointHarmonics, SphericalPattern, StochasticConfig
    n = int(points_count)
    lat = cp.arcsin(1 - 2*(cp.arange(n, dtype=cp.float64) + 0.5)/n)
    lon = cp.arange(n, dtype=cp.float64)*(math.pi*(3 - math.sqrt(5)))
    resource = PointHarmonics(lat, lon, degree)
    out = {}
    for channels in (1, 3):
        pattern = SphericalPattern(resource, StochasticConfig(sppt=True, degree=degree), channels=channels)
        pattern.advance(0)
        times = []
        for k in range(repeats):
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            resource.synthesize(pattern.coefficients, pattern.pattern, degree, n, channels)
            e.record(); e.synchronize()
            times.append(cp.cuda.get_elapsed_time(s, e))
        out[f'channels_{channels}_ms'] = sorted(times)[len(times)//2]
    return dict(points=n, degree=degree, **out)


def plots(stats, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for key, colour in (('global', 'C0'), ('lam', 'C1')):
        if key not in stats:
            continue
        c = stats[key]['curve']
        axes[0].plot(c['distance_km'], c['measured'], 'o', ms=3, color=colour,
                     label=f"{key} measured ({stats[key]['points']} cells)")
        axes[0].plot(c['distance_km'], c['theory'], '-', color=colour, lw=1, label=f'{key} truncated series')
    if 'global' in stats:
        c = stats['global']['curve']
        axes[0].plot(c['distance_km'], c['gaussian'], 'k--', lw=1, label='exp(-r^2/2L^2)')
        L = stats['global']['requested_length_km']
        axes[0].axvline(L, color='grey', lw=.6); axes[0].axhline(math.exp(-.5), color='grey', lw=.6)
    axes[0].set_xlabel('distance (km)'); axes[0].set_ylabel('correlation'); axes[0].legend(fontsize=7)
    axes[0].set_title('Spatial correlation vs distance')
    if 'spectrum' in stats:
        s = stats['spectrum']
        ell = list(range(1, len(s['power_by_degree']) + 1))
        axes[1].semilogy(ell, s['power_by_degree'], 'o', ms=3, label='measured (projected)')
        axes[1].semilogy(ell, s['target_by_degree'], '-', label='requested')
        axes[1].set_xlabel('degree l'); axes[1].set_ylabel('variance per degree'); axes[1].legend(fontsize=7)
        axes[1].set_title('Variance spectrum (' + stats['global']['label'] + ')')
    if 'temporal' in stats:
        t = stats['temporal']['acf']
        axes[2].plot(t['lag_h'], t['measured'], 'o', ms=3, label='measured')
        axes[2].plot(t['lag_h'], t['expected'], '-', label='exp(-lag/tau)')
        axes[2].axhline(1/math.e, color='grey', lw=.6)
        axes[2].set_xlabel('lag (h)'); axes[2].set_ylabel('autocorrelation'); axes[2].legend(fontsize=7)
        axes[2].set_title('Lag autocorrelation')
    fig.tight_layout()
    fig.savefig(directory/'pattern-statistics.png', dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--lam-grid', type=Path, default=None)
    parser.add_argument('--region', default=None, metavar='LAT,LON,RADIUS_KM',
                        help='a regional subset of the global grid cells (the points a limited-area mesh holds)')
    parser.add_argument('--length-km', type=float, default=450.)
    parser.add_argument('--time-h', type=float, default=6.)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--pairs', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--timing-points', type=int, default=5_870_000,
                        help='synthesis timing point count (conus3km cells + vertices, about 5.87 million)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plots', type=Path, default=None)
    args = parser.parse_args(argv)
    import cupy as cp
    started = time.perf_counter()
    length = args.length_km*1000.
    stats = dict(schema='gpuwm-hex.stochastic-pattern-stats/v1', device=cp.cuda.runtime.getDeviceProperties(
        cp.cuda.Device().id)['name'].decode(), law='spatial exp(-r^2/(2L^2)), temporal exp(-dt/tau)')
    glob = _mesh_points(args.grid, cp)
    stats['global'], _, _ = spatial_stats(cp, glob, length, samples=args.samples, pairs=args.pairs,
                                          seed=args.seed, label=args.grid.name)
    stats['spectrum'] = spectrum_stats(cp, glob, length, samples=min(args.samples, 100), seed=args.seed + 1)
    stats['temporal'] = temporal_stats(cp, glob, length, args.time_h*3600., dt=600., steps=2160,
                                       seed=args.seed + 2, max_lag=144)
    if args.lam_grid is None and args.region:
        clat, clon, radius_km = (float(v) for v in args.region.split(','))
        clat, clon = math.radians(clat), math.radians(clon)
        cosd = (cp.sin(glob['lat'])*math.sin(clat) + cp.cos(glob['lat'])*math.cos(clat)*cp.cos(glob['lon'] - clon))
        inside = cp.arccos(cp.clip(cosd, -1, 1))*6371.22 <= radius_km
        region = {k: glob[k][inside] for k in ('lat', 'lon', 'area')}
        stats['lam'], _, _ = spatial_stats(cp, region, length, samples=args.samples, pairs=args.pairs,
                                           seed=args.seed + 3, label=f'region {args.region} of {args.grid.name}',
                                           max_km=2*length/1000)
    if args.lam_grid is not None:
        lam = _mesh_points(args.lam_grid, cp)
        stats['lam'], _, _ = spatial_stats(cp, lam, length, samples=args.samples, pairs=args.pairs,
                                           seed=args.seed + 3, label=args.lam_grid.name, max_km=2*length/1000)
    from woof.hex.stochastic import auto_degree
    stats['synthesis_timing'] = synthesis_timing(cp, args.timing_points, auto_degree(length))
    stats['wall_seconds'] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=1) + '\n')
    if args.plots is not None:
        plots(stats, args.plots)
    brief = {k: stats[k] for k in ('device', 'wall_seconds')}
    for key in ('global', 'lam'):
        if key in stats:
            brief[key] = {k: v for k, v in stats[key].items() if k != 'curve'}
    brief['spectrum'] = {k: v for k, v in stats['spectrum'].items() if not k.endswith('degree')}
    brief['temporal'] = {k: v for k, v in stats['temporal'].items() if k != 'acf'}
    brief['synthesis_timing'] = stats['synthesis_timing']
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main()
