"""Predictive storm tracker and moving-region placement policy (CPU only).

Everything here reads MODEL STATE ONLY: per-point latitude/longitude plus
fields diagnosed from the running forecast (or from its history files, or from
any gridded forecast for scoring). No observed or oracle track enters at run
time; the only external input is the storm's position at the analysis time,
which the analysis/advisory itself provides.

Pieces
------
* ``LocalLattice``    resample scattered cells (any mesh or lat/lon grid) onto
                      a local azimuthal-equidistant lattice about a guess.
* ``TCFixer``         tropical-cyclone centre: smoothed MSLP minimum and
                      850 (and 700) hPa relative-vorticity maximum, each refined
                      by a weighted centroid, combined by consensus, gated by
                      intensity and by continuity with the predicted position.
* ``ObjectFixer``     severe-convection centre: the dominant thresholded object
                      of a 2-D field (updraft helicity, composite reflectivity)
                      near the predicted position.
* setup targets       follow a severe-weather SETUP rather than a cyclone:
                      ``low`` (a surface low: the TC consensus on synoptic
                      scales), ``cape_shear`` (the dominant area of mixed-layer
                      CAPE x 0-6 km bulk shear), ``boundary`` (a dryline or
                      front: the dominant 2 m dewpoint / theta-e / temperature
                      gradient segment, oriented like the segment the user
                      traced) and ``path`` (a path the user traced: the region goes
                      where the user said, by time; it is not a tracker).
* ``MotionModel``     recency-weighted least-squares motion from the fix
                      history, blended with the deep-layer steering flow
                      diagnosed from the same state while the history is short.
* ``PlacementPolicy`` when to move and where: keep the storm core inside the
                      constant-spacing fine interior with an uncertainty margin
                      that grows with lead; move only when the current region
                      would fail before the next possible check; place the new
                      centre (snapped to a lattice) at the predicted position
                      that maximises the time before the next move.
* ``TrackSession``    the run-time object a controller drives:
                      observe(t, state) -> Fix ; decide(t, region) -> Decision.

Storm and case specifics (initial position, which fixer, radii) are config.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import itertools
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.229
G = 9.80616
RD = 287.0
CP = 1003.0


# --------------------------------------------------------------------------- geometry

def xyz(lat_deg, lon_deg):
    lat, lon = np.deg2rad(np.asarray(lat_deg, float)), np.deg2rad(np.asarray(lon_deg, float))
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=-1)


def latlon(v):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.rad2deg(np.arcsin(np.clip(v[..., 2], -1, 1))), np.rad2deg(np.arctan2(v[..., 1], v[..., 0]))


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(np.asarray(lon2) - np.asarray(lon1)) / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def project(lat0, lon0, lat, lon):
    """Azimuthal-equidistant (x east, y north) km about (lat0, lon0)."""
    p0, l0 = math.radians(lat0), math.radians(lon0)
    p, l = np.deg2rad(lat), np.deg2rad(lon)
    cosc = np.clip(math.sin(p0) * np.sin(p) + math.cos(p0) * np.cos(p) * np.cos(l - l0), -1, 1)
    c = np.arccos(cosc)
    k = np.where(c > 1e-12, c / np.where(np.sin(c) == 0, 1, np.sin(c)), 1.0)
    x = k * np.cos(p) * np.sin(l - l0)
    y = k * (math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(l - l0))
    return EARTH_RADIUS_KM * x, EARTH_RADIUS_KM * y


def unproject(lat0, lon0, x, y):
    p0, l0 = math.radians(lat0), math.radians(lon0)
    x, y = np.asarray(x, float) / EARTH_RADIUS_KM, np.asarray(y, float) / EARTH_RADIUS_KM
    rho = np.hypot(x, y)
    c = rho
    safe = np.where(rho > 1e-15, rho, 1.0)
    lat = np.arcsin(np.clip(np.cos(c) * math.sin(p0) + y * np.sin(c) * math.cos(p0) / safe, -1, 1))
    lon = l0 + np.arctan2(x * np.sin(c), rho * math.cos(p0) * np.cos(c) - y * math.sin(p0) * np.sin(c))
    lat = np.where(rho > 1e-15, lat, p0)
    lon = np.where(rho > 1e-15, lon, l0)
    return np.rad2deg(lat), (np.rad2deg(lon) + 180.0) % 360.0 - 180.0


def destination(lat, lon, bearing_deg, dist_km):
    """Great-circle point at bearing (deg clockwise from north) and distance."""
    b = math.radians(bearing_deg)
    return unproject(lat, lon, dist_km * math.sin(b), dist_km * math.cos(b))


def wrap_lon(lon):
    return (np.asarray(lon, float) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- resampling

class PointCloud:
    """Scattered points (cell centres or grid points) with a KD-tree on the unit sphere."""

    def __init__(self, lat_deg, lon_deg):
        from scipy.spatial import cKDTree
        self.lat = np.asarray(lat_deg, float).ravel()
        self.lon = wrap_lon(np.asarray(lon_deg, float).ravel())
        self.xyz = xyz(self.lat, self.lon)
        self.tree = cKDTree(self.xyz)
        self.n = self.lat.size

    def near(self, lat, lon, radius_km):
        return np.asarray(self.tree.query_ball_point(xyz(lat, lon), 2 * math.sin(radius_km / EARTH_RADIUS_KM / 2)), int)


class LocalLattice:
    """Regular azimuthal-equidistant lattice about a centre, sampled by inverse-distance weights."""

    def __init__(self, cloud: PointCloud, lat0, lon0, half_width_km, spacing_km, k=4):
        self.lat0, self.lon0, self.spacing = float(lat0), float(wrap_lon(lon0)), float(spacing_km)
        n = int(math.ceil(half_width_km / spacing_km))
        axis = np.arange(-n, n + 1) * spacing_km
        self.x, self.y = np.meshgrid(axis, axis)  # [row=y, col=x]
        self.lat, self.lon = unproject(self.lat0, self.lon0, self.x, self.y)
        d, i = cloud.tree.query(xyz(self.lat, self.lon).reshape(-1, 3), k=k)
        d = np.atleast_2d(d.T).T if k == 1 else d
        i = i.reshape(-1, k) if k > 1 else i.reshape(-1, 1)
        d = d.reshape(-1, k) * EARTH_RADIUS_KM
        w = 1.0 / np.maximum(d, 1e-3) ** 2
        self.index, self.weight = i, w / w.sum(axis=1, keepdims=True)
        self.nearest_km = d[:, 0].reshape(self.x.shape)
        self.shape = self.x.shape

    def sample(self, values):
        v = np.asarray(values, float)[self.index]
        good = np.isfinite(v)
        w = np.where(good, self.weight, 0.0)
        s = w.sum(axis=1)
        out = np.where(s > 0, (np.where(good, v, 0.0) * w).sum(axis=1) / np.where(s > 0, s, 1), np.nan)
        return out.reshape(self.shape)

    def radius_from(self, x0, y0):
        return np.hypot(self.x - x0, self.y - y0)

    def to_latlon(self, x, y):
        lat, lon = unproject(self.lat0, self.lon0, x, y)
        return float(lat), float(lon)

    def to_xy(self, lat, lon):
        x, y = project(self.lat0, self.lon0, lat, lon)
        return float(x), float(y)


def smooth(field, sigma_px):
    """NaN-aware Gaussian smoothing (normalised convolution)."""
    from scipy.ndimage import gaussian_filter
    f = np.asarray(field, float)
    good = np.isfinite(f)
    if sigma_px <= 0:
        return f
    num = gaussian_filter(np.where(good, f, 0.0), sigma_px, mode='nearest')
    den = gaussian_filter(good.astype(float), sigma_px, mode='nearest')
    return np.where(den > 0.2, num / np.where(den > 0, den, 1), np.nan)


def vorticity(lattice: LocalLattice, u, v):
    """Relative vorticity dv/dx - du/dy (s-1) on the lattice; u,v in m/s along local east/north.

    The local east/north frame differs from the projection axes by a rotation that is
    below 2 degrees within 600 km at tropical latitudes; it is ignored here.
    """
    h = lattice.spacing * 1000.0
    dvdx = np.gradient(v, h, axis=1)
    dudy = np.gradient(u, h, axis=0)
    return dvdx - dudy


# --------------------------------------------------------------------------- state -> fields

def interpolate_to_pressure(pressure, values, target_pa):
    """Log-pressure linear interpolation per column. pressure/values: (n, nlev), level 0 at the bottom."""
    p = np.asarray(pressure, float)
    f = np.asarray(values, float)
    above = p < target_pa
    k1 = np.argmax(above, axis=1)  # first level above the target
    ok = above.any(axis=1) & (k1 > 0)
    k1 = np.where(ok, k1, 1)
    k0 = k1 - 1
    rows = np.arange(p.shape[0])
    lp0, lp1 = np.log(p[rows, k0]), np.log(p[rows, k1])
    w = (math.log(target_pa) - lp0) / np.where(lp1 != lp0, lp1 - lp0, 1)
    out = f[rows, k0] + w * (f[rows, k1] - f[rows, k0])
    return np.where(ok, out, np.nan)


def sea_level_pressure(psfc, terrain_m, t_low_k):
    """Hypsometric reduction with a 6.5 K/km mean-layer temperature; equals psfc over the sea."""
    t_mean = np.asarray(t_low_k, float) + 0.0065 * np.asarray(terrain_m, float) / 2.0
    return np.asarray(psfc, float) * np.exp(G * np.asarray(terrain_m, float) / (RD * np.maximum(t_mean, 150.0)))


STEERING_LEVELS_PA = (85000.0, 70000.0, 50000.0, 30000.0, 20000.0)
STEERING_WEIGHTS = (0.15, 0.2, 0.25, 0.2, 0.2)


@dataclass
class StateFields:
    """Per-point arrays the tracker consumes; any optional field may be None."""
    lat: np.ndarray
    lon: np.ndarray
    mslp_pa: np.ndarray | None = None
    u_p: Mapping[float, np.ndarray] | None = None   # pressure (Pa) -> zonal wind
    v_p: Mapping[float, np.ndarray] | None = None
    wind10: np.ndarray | None = None
    object_field: np.ndarray | None = None          # e.g. updraft helicity or composite reflectivity
    land: np.ndarray | None = None
    # Setup targets. ``columns``: per-point model columns (n, nlev), level 0 at the bottom: pressure (Pa),
    # theta (K), qv (kg/kg), u, v (m/s), plus terrain (m) and psfc (Pa) as (n,). ``surface``: t2 (K), q2 (kg/kg),
    # psfc (Pa), each (n,). Either may be None; the fixer that needs one says so in its miss reason.
    columns: Mapping[str, np.ndarray] | None = None
    surface: Mapping[str, np.ndarray] | None = None
    cloud: PointCloud | None = None
    diagnostic_arrays: Mapping[str, np.ndarray] | None = None

    def points(self) -> PointCloud:
        if self.cloud is None:
            self.cloud = PointCloud(self.lat, self.lon)
        return self.cloud


def fields_from_columns(lat, lon, *, psfc, terrain, t_low, pressure=None, u=None, v=None, u10=None, v10=None,
                        levels_pa=(85000.0, 70000.0) + STEERING_LEVELS_PA[2:], object_field=None, subset=None,
                        cloud=None, columns=None, surface=None):
    """Build StateFields from model columns (history file or live arrays).

    ``subset`` (index array) limits the expensive 3-D interpolation to cells near the storm;
    points outside it get NaN upper-air fields.
    """
    n = np.asarray(lat).size
    out = StateFields(lat=np.asarray(lat, float), lon=wrap_lon(lon), cloud=cloud,
                      mslp_pa=sea_level_pressure(psfc, terrain, t_low),
                      land=np.asarray(terrain, float) > 1.0)
    if u10 is not None and v10 is not None:
        out.wind10 = np.hypot(np.asarray(u10, float), np.asarray(v10, float))
    if pressure is not None and u is not None and v is not None:
        idx = np.arange(n) if subset is None else np.asarray(subset, int)
        P, U, V = np.asarray(pressure)[idx], np.asarray(u)[idx], np.asarray(v)[idx]
        out.u_p, out.v_p = {}, {}
        for level in sorted(set(levels_pa)):
            uu, vv = np.full(n, np.nan), np.full(n, np.nan)
            uu[idx] = interpolate_to_pressure(P, U, level)
            vv[idx] = interpolate_to_pressure(P, V, level)
            out.u_p[level], out.v_p[level] = uu, vv
    if object_field is not None:
        out.object_field = np.asarray(object_field, float)
    if columns is not None:
        out.columns = {k: np.asarray(v, float) for k, v in columns.items()}
    if surface is not None:
        out.surface = {k: np.asarray(v, float) for k, v in surface.items()}
    return out


# --------------------------------------------------------------------------- fixes

@dataclass
class Fix:
    seconds: float
    ok: bool
    lat: float | None = None
    lon: float | None = None
    reason: str = ''
    mslp_min_hpa: float | None = None
    depression_hpa: float | None = None
    vort850_max: float | None = None
    vmax10_ms: float | None = None
    components: dict = field(default_factory=dict)
    guess: list | None = None
    search_km: float | None = None
    steering_ms: list | None = None
    field_fix: dict | None = None
    field_fix_error: str | None = None

    def as_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TCConfig:
    lattice_spacing_km: float = 5.0
    search_km: float = 150.0          # base continuity radius about the predicted position
    search_growth_kmh: float = 40.0   # added per hour since the last good fix
    smooth_km: float = 20.0
    centroid_km: float = 100.0
    agree_km: float = 120.0
    min_depression_hpa: float = 1.0   # MSLP minimum below the ring mean (ring_inner_km..ring_outer_km)
    ring_inner_km: float = 250.0
    ring_outer_km: float = 450.0
    min_vort850: float = 5.0e-5       # hemisphere-signed, s-1, on the smoothed lattice
    min_components: int = 1
    vmax_radius_km: float = 250.0
    steering_inner_km: float = 0.0
    steering_outer_km: float = 500.0
    weights: dict = field(default_factory=lambda: {'mslp': 1.0, 'vort850': 1.0, 'vort700': 0.5})
    # fix = prediction + fix_blend * (consensus - prediction). 1.0 for a cyclone (its centre is sharp); a broad
    # synoptic low's smoothed minimum jumps 100-200 km between half-hourly frames (IFS f360 lee low, box L
    # 2026-10-24: fixes 44.9N -> 46.6N -> 43.4N in 1.5 h, the region thrown 264 km off), so the low target damps it.
    fix_blend: float = 1.0


@dataclass
class ObjectConfig:
    field_name: str = 'refl10cm'      # state/history variable (column maximum taken for 3-D fields)
    lattice_spacing_km: float = 3.0
    search_km: float = 100.0
    search_growth_kmh: float = 25.0   # about the predicted position: motion uncertainty, not the maximum speed
    smooth_km: float = 6.0
    threshold: float = 35.0           # field units (dBZ for reflectivity, m2/s2 for UH)
    min_area_km2: float = 200.0
    distance_scale_km: float = 60.0   # objects far from the prediction are strongly down-weighted
    size_power: float = 0.5           # score = (exceedance mass)**size_power: a large distant system must not win
    steering_inner_km: float = 0.0
    steering_outer_km: float = 250.0


def _centroid(lattice, weight, x0, y0, radius):
    r = lattice.radius_from(x0, y0)
    w = np.where((r <= radius) & np.isfinite(weight), np.maximum(weight, 0.0), 0.0)
    s = w.sum()
    if s <= 0:
        return x0, y0
    return float((w * lattice.x).sum() / s), float((w * lattice.y).sum() / s)


def _ring_mean(lattice, f, x0, y0, r_in, r_out):
    r = lattice.radius_from(x0, y0)
    m = (r >= r_in) & (r <= r_out) & np.isfinite(f)
    return float(np.nanmean(f[m])) if m.any() else float('nan')


def _consensus(points: dict, weights: dict, agree_km: float):
    """Largest set of fixes all within agree_km of their weighted mean (ties: larger total weight)."""
    keys = list(points)
    best = None
    for size in range(len(keys), 0, -1):
        for combo in itertools.combinations(keys, size):
            w = np.array([weights.get(k, 1.0) for k in combo])
            p = np.array([points[k] for k in combo])
            mean = (w[:, None] * p).sum(0) / w.sum()
            if all(np.hypot(*(points[k] - mean)) <= agree_km for k in combo):
                score = (size, w.sum())
                if best is None or score > best[0]:
                    best = (score, combo, mean)
        if best is not None:
            return best[1], best[2]
    return (), None


def steering(fields: StateFields, lat, lon, inner_km, outer_km):
    """Deep-layer (850-200 hPa) mean wind over a disc/annulus about the centre (m/s, east/north)."""
    if not fields.u_p:
        return None
    cloud = fields.points()
    idx = cloud.near(lat, lon, outer_km)
    if inner_km > 0:
        idx = idx[distance_km(lat, lon, cloud.lat[idx], cloud.lon[idx]) >= inner_km]
    if idx.size == 0:
        return None
    # Equal-area mean: sample on a lattice so variable cell density does not bias the mean.
    lattice = LocalLattice(cloud, lat, lon, outer_km, max(outer_km / 25.0, 5.0), k=1)
    r = np.hypot(lattice.x, lattice.y)
    disc = (r <= outer_km) & (r >= inner_km)
    su = sv = sw = 0.0
    for level, weight in zip(STEERING_LEVELS_PA, STEERING_WEIGHTS):
        if level not in fields.u_p:
            continue
        uu, vv = lattice.sample(fields.u_p[level]), lattice.sample(fields.v_p[level])
        m = disc & np.isfinite(uu) & np.isfinite(vv)
        if m.sum() < 10:
            continue
        su += weight * float(uu[m].mean()); sv += weight * float(vv[m].mean()); sw += weight
    return None if sw == 0 else [su / sw, sv / sw]


class TCFixer:
    def __init__(self, config: TCConfig | None = None):
        self.c = config or TCConfig()

    def fix(self, seconds, fields: StateFields, guess, search_km) -> Fix:
        c = self.c
        cloud = fields.points()
        glat, glon = float(guess[0]), float(wrap_lon(guess[1]))
        half = search_km + max(c.ring_outer_km, c.centroid_km * 2)
        lat_ = LocalLattice(cloud, glat, glon, half, c.lattice_spacing_km)
        sig = c.smooth_km / c.lattice_spacing_km
        r0 = lat_.radius_from(0.0, 0.0)
        inside = r0 <= search_km
        sign = 1.0 if glat >= 0 else -1.0
        comps, info = {}, {}
        fx = Fix(seconds=float(seconds), ok=False, guess=[glat, glon], search_km=float(search_km))
        p = None
        if fields.mslp_pa is not None:
            p = smooth(lat_.sample(fields.mslp_pa), sig) / 100.0
            pin = np.where(inside, p, np.nan)
            if np.isfinite(pin).any():
                j = np.nanargmin(pin)
                x0, y0 = float(lat_.x.flat[j]), float(lat_.y.flat[j])
                ring = _ring_mean(lat_, p, x0, y0, c.ring_inner_km, c.ring_outer_km)
                depression = ring - float(pin.flat[j])
                info['mslp'] = dict(min_hpa=float(pin.flat[j]), depression_hpa=depression)
                if depression >= c.min_depression_hpa:
                    xc, yc = _centroid(lat_, ring - p, x0, y0, c.centroid_km)
                    xc, yc = _centroid(lat_, ring - p, xc, yc, c.centroid_km)
                    comps['mslp'] = np.array([xc, yc])
                fx.depression_hpa = depression
        for name, level in (('vort850', 85000.0), ('vort700', 70000.0)):
            if not fields.u_p or level not in fields.u_p:
                continue
            u, v = lat_.sample(fields.u_p[level]), lat_.sample(fields.v_p[level])
            z = sign * smooth(vorticity(lat_, smooth(u, sig), smooth(v, sig)), 0)
            zin = np.where(inside, z, np.nan)
            if not np.isfinite(zin).any():
                continue
            j = np.nanargmax(zin)
            zmax = float(zin.flat[j])
            info[name] = dict(max=zmax)
            if name == 'vort850':
                fx.vort850_max = zmax
            if zmax >= c.min_vort850 * (1.0 if name == 'vort850' else 0.8):
                x0, y0 = float(lat_.x.flat[j]), float(lat_.y.flat[j])
                xc, yc = _centroid(lat_, z - 0.5 * zmax, x0, y0, c.centroid_km * 0.75)
                comps[name] = np.array([xc, yc])
        fx.components = {k: dict(**info.get(k, {}), lat=lat_.to_latlon(*comps[k])[0], lon=lat_.to_latlon(*comps[k])[1])
                         for k in comps}
        for k in info:
            fx.components.setdefault(k, dict(**info[k], rejected=True))
        if not comps:
            fx.reason = 'no component passed its intensity threshold within the search radius'
            return fx
        used, mean = _consensus(comps, c.weights, c.agree_km)
        if len(used) < c.min_components:
            fx.reason = f'only {len(used)} agreeing components'
            return fx
        if np.hypot(*mean) > search_km:
            fx.reason = f'consensus centre {np.hypot(*mean):.0f} km from the prediction exceeds the search radius'
            return fx
        fx.lat, fx.lon = lat_.to_latlon(*(c.fix_blend * np.asarray(mean, float)))
        fx.ok = True
        fx.reason = 'consensus of ' + '+'.join(used)
        # Position consensus still drives motion. Report intensity separately
        # from the unsmoothed water-cell fix; a 20 km smoother erases the eye.
        if fields.diagnostic_arrays is not None:
            from woof.tc.field_products import diagnose_arrays
            try:
                fx.field_fix = diagnose_arrays(fields.diagnostic_arrays, (fx.lat,fx.lon), unstructured=True)
            except ValueError as error:
                # A storm that has moved inland may have no nearby water cell.
                # Keep its motion fix and report missing intensity explicitly.
                if 'no finite water-cell pressure' not in str(error):
                    raise
                fx.field_fix_error = str(error)
            if fx.field_fix is not None:
                fx.mslp_min_hpa = fx.field_fix['mslp_hpa']
                fx.vmax10_ms = fx.field_fix['vmax10_m_s']
        fx.steering_ms = steering(fields, fx.lat, fx.lon, c.steering_inner_km, c.steering_outer_km)
        return fx


class ObjectFixer:
    def __init__(self, config: ObjectConfig | None = None):
        self.c = config or ObjectConfig()

    def fix(self, seconds, fields: StateFields, guess, search_km) -> Fix:
        from scipy.ndimage import label
        c = self.c
        fx = Fix(seconds=float(seconds), ok=False, guess=[float(guess[0]), float(wrap_lon(guess[1]))],
                 search_km=float(search_km))
        if fields.object_field is None:
            fx.reason = 'no object field in the state'
            return fx
        cloud = fields.points()
        lat_ = LocalLattice(cloud, guess[0], guess[1], search_km + 150.0, c.lattice_spacing_km)
        f = smooth(lat_.sample(fields.object_field), c.smooth_km / c.lattice_spacing_km)
        mask = np.isfinite(f) & (f >= c.threshold)
        labels, n = label(mask)
        if n == 0:
            fx.reason = 'no object above threshold'
            return fx
        cell = c.lattice_spacing_km ** 2
        excess = np.where(mask, f - c.threshold + 1e-6, 0.0)
        best = None
        for k in range(1, n + 1):
            m = labels == k
            area = m.sum() * cell
            if area < c.min_area_km2:
                continue
            w = excess * m
            xc, yc = float((w * lat_.x).sum() / w.sum()), float((w * lat_.y).sum() / w.sum())
            d = math.hypot(xc, yc)
            if d > search_km:
                continue
            score = float(w.sum() * cell) ** c.size_power * math.exp(-(d / c.distance_scale_km) ** 2)
            if best is None or score > best[0]:
                best = (score, xc, yc, area, float(f[m].max()))
        if best is None:
            fx.reason = 'no object of sufficient area within the search radius'
            return fx
        _, xc, yc, area, peak = best
        fx.lat, fx.lon = lat_.to_latlon(xc, yc)
        fx.ok = True
        fx.reason = 'dominant object'
        fx.components = dict(object=dict(area_km2=area, peak=peak, objects=int(n)))
        fx.steering_ms = steering(fields, fx.lat, fx.lon, c.steering_inner_km, c.steering_outer_km)
        return fx


# --------------------------------------------------------------------------- setup targets

def low_config(**overrides) -> 'TCConfig':
    """A synoptic surface low: the TC consensus (smoothed MSLP minimum + 850 hPa vorticity) on synoptic scales.

    Wider lattice, smoothing, centroid and depression ring than a cyclone, and a weaker vorticity floor (a
    mid-latitude low's smoothed 850 hPa vorticity is a few 1e-5 s-1, not a cyclone's 1e-4)."""
    base = dict(lattice_spacing_km=15.0, search_km=200.0, search_growth_kmh=30.0, smooth_km=100.0,
                centroid_km=200.0, agree_km=250.0, min_depression_hpa=1.0, ring_inner_km=500.0,
                ring_outer_km=900.0, min_vort850=2.0e-5, min_components=1, vmax_radius_km=500.0,
                steering_inner_km=0.0, steering_outer_km=800.0,
                weights={'mslp': 1.0, 'vort850': 0.5, 'vort700': 0.0}, fix_blend=0.5)
    base.update(overrides)
    return TCConfig(**base)


@dataclass
class EnvConfig:
    """The dominant area of a severe-weather ingredient product: mixed-layer CAPE x 0-6 km bulk shear.

    index = (MLCAPE / 1000 J kg-1) * (S06 / 20 m s-1); 1 means 1000 J/kg with 20 m/s of deep-layer shear."""
    lattice_spacing_km: float = 15.0
    search_km: float = 300.0
    search_growth_kmh: float = 50.0
    smooth_km: float = 40.0
    threshold: float = 0.25           # index units: e.g. 500 J/kg with 10 m/s, or 250 J/kg with 20 m/s
    min_area_km2: float = 5000.0
    distance_scale_km: float = 250.0
    size_power: float = 0.5
    mixed_layer_pa: float = 10000.0   # the lowest 100 hPa
    shear_depth_m: float = 6000.0
    fix_blend: float = 0.6            # fix = prediction + blend * (area centre - prediction): an area's centroid
                                      # wanders with the diurnal cycle; the blend keeps the region's moves clean
    core_fraction: float = 0.6        # the target is the area's core: the part at or above core_fraction x its peak.
                                      # Breakage it prevents: GFS 2026-10-23 18Z f354 (box K, 2026-10-24) grew one
                                      # connected 0.25 area to 570,000 km2 over Texas and Mexico, its centroid
                                      # drifted 700 km to Sinaloa and the region left the storms on the Rio Grande.
    # Storms are part of the setup: deep convection uses up the CAPE it grows from, so the index is lowest
    # exactly where the setup is producing storms, and an index-only target walks away from them. Breakage it
    # prevents: GFS 2026-10-23 18Z f354 cape_shear (box L, 2026-10-24) left the storm complex on the Rio Grande
    # and Edwards Plateau after 4 h and slid 800 km WSW, ending on a storm-free Pacific CAPE plume off Sinaloa.
    # Column-maximum reflectivity at or above storm_dbz near the area pulls the centre with weight
    # storm_weight x the area's peak index per fully covered cell; with no storms the target is the index core.
    storm_field: str = 'refl10cm'
    storm_dbz: float = 35.0
    storm_weight: float = 1.0
    storm_reach_km: float = 150.0     # storms within this distance of the CAPE x shear area belong to it
    # A setup moves with the weather, not across a country in an hour. When the storms died after dark on the same
    # run (+16 h), the one connected 0.25 area stretched from Coahuila to the Pacific, its peak was a storm-free
    # ocean plume off Sinaloa, and the fixes ran 100-150 km per 30 min check (900 km in 3.5 h). So the core is taken
    # from the area's part within core_reach_km of the prediction, and one fix moves at most max_step_km from it.
    core_reach_km: float | None = 200.0
    max_step_km: float | None = 45.0
    steering_inner_km: float = 0.0
    steering_outer_km: float = 500.0


@dataclass
class BoundaryConfig:
    """A dryline or front: the dominant gradient segment of a surface scalar near the prediction."""
    field: str = 'dewpoint'           # 'dewpoint' (dryline), 'theta_e' or 'temperature' (fronts), from t2/q2/psfc
    lattice_spacing_km: float = 6.0
    search_km: float = 200.0
    search_growth_kmh: float = 40.0
    smooth_km: float = 18.0
    threshold: float = 4.0            # K per 100 km
    min_area_km2: float = 2000.0
    distance_scale_km: float = 150.0
    size_power: float = 0.5
    segment: list | None = None       # [[lat, lon], [lat, lon]]: the segment the user traced at the start
    orientation_weight: float = 1.0   # score *= |cos(angle to the segment's orientation)| ** weight
    fix_blend: float = 0.7
    steering_inner_km: float = 0.0
    steering_outer_km: float = 400.0


@dataclass
class PathConfig:
    """A path the user traced: rows of [seconds since the run's start, lat, lon]. Not a tracker."""
    rows: list = field(default_factory=list)
    search_km: float = 0.0
    search_growth_kmh: float = 0.0


def _virtual(t, q):
    return t * (1.0 + 0.608 * q)


def _saturation_mixing_ratio(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / np.maximum(t - 29.65, 1.0))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _theta_e(t, p, r, t_lcl):
    """Bolton (1980) eq. 43; r in kg/kg."""
    rg = r * 1000.0
    return (t * (100000.0 / p) ** (0.2854 * (1.0 - 0.00028 * rg))
            * np.exp((3.376 / t_lcl - 0.00254) * rg * (1.0 + 0.00081 * rg)))


def _lcl_temperature(t, r, p):
    """Bolton (1980) eq. 21 from the vapour pressure (hPa)."""
    e = np.maximum(r * p / (0.622 + r), 1.0) / 100.0
    return 2840.0 / (3.5 * np.log(t) - np.log(e) - 4.805) + 55.0


def mlcape_and_shear(columns: Mapping[str, np.ndarray], mixed_layer_pa=10000.0, shear_depth_m=6000.0):
    """Mixed-layer CAPE (J/kg: positive buoyancy area of a pseudo-adiabatic parcel, Bolton theta-e, virtual
    temperature) and the 0-shear_depth_m bulk shear (m/s) of model columns, vectorised over columns.

    columns: pressure, theta, qv, u, v as (n, nlev) with level 0 at the bottom, and terrain, psfc as (n,)."""
    p = np.asarray(columns['pressure'], float)
    th = np.asarray(columns['theta'], float)
    qv = np.maximum(np.asarray(columns['qv'], float), 1e-8)
    u, v = np.asarray(columns['u'], float), np.asarray(columns['v'], float)
    psfc = np.asarray(columns['psfc'], float)
    kappa = RD / CP
    t = th * (p / 1.0e5) ** kappa
    tv = _virtual(t, qv)
    # heights above the ground (hypsometric, from the surface pressure)
    z = np.empty_like(p)
    z[:, 0] = RD * tv[:, 0] / G * np.log(np.maximum(psfc, p[:, 0] + 1.0) / p[:, 0])
    z[:, 1:] = z[:, :1] + np.cumsum(RD * 0.5 * (tv[:, 1:] + tv[:, :-1]) / G * np.log(p[:, :-1] / p[:, 1:]), axis=1)
    # the mixed layer: mean theta and mixing ratio of the levels within mixed_layer_pa of the surface
    ml = p >= (psfc[:, None] - mixed_layer_pa)
    ml[:, 0] = True
    count = ml.sum(axis=1)
    th_ml = (th * ml).sum(axis=1) / count
    r_ml = ((qv / (1.0 - qv)) * ml).sum(axis=1) / count
    t_start = th_ml * (psfc / 1.0e5) ** kappa
    t_lcl = _lcl_temperature(t_start, r_ml, psfc)
    p_lcl = psfc * (t_lcl / t_start) ** (1.0 / kappa)
    thetae = _theta_e(t_start, psfc, r_ml, t_lcl)
    # parcel temperature on every level: dry below the LCL, pseudo-adiabatic (theta-e conserved) above
    dry = th_ml[:, None] * (p / 1.0e5) ** kappa
    lo, hi = np.full_like(p, 150.0), np.full_like(p, 350.0)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        warm = _theta_e(mid, p, _saturation_mixing_ratio(mid, p), mid) > thetae[:, None]
        hi = np.where(warm, mid, hi)
        lo = np.where(warm, lo, mid)
    moist_t = 0.5 * (lo + hi)
    above = p < p_lcl[:, None]
    tp = np.where(above, moist_t, dry)
    rp = np.where(above, _saturation_mixing_ratio(moist_t, p), r_ml[:, None])
    tvp = tp * (1.0 + 0.608 * rp / (1.0 + rp))
    buoyancy = G * (tvp - tv) / tv
    dz = np.diff(z, axis=1, prepend=0.0)
    useful = (p >= 10000.0) & (z > 0)
    cape = np.where(useful, np.maximum(buoyancy, 0.0) * dz, 0.0).sum(axis=1)
    # 0-depth bulk shear: the lowest model level to depth above the ground
    reach = z >= shear_depth_m
    k1 = np.argmax(reach, axis=1)
    ok = reach.any(axis=1) & (k1 > 0)
    k1 = np.where(ok, k1, 1)
    k0 = k1 - 1
    rows = np.arange(p.shape[0])
    w = (shear_depth_m - z[rows, k0]) / np.maximum(z[rows, k1] - z[rows, k0], 1e-3)
    ut = u[rows, k0] + w * (u[rows, k1] - u[rows, k0])
    vt = v[rows, k0] + w * (v[rows, k1] - v[rows, k0])
    shear = np.where(ok, np.hypot(ut - u[:, 0], vt - v[:, 0]), np.nan)
    return np.where(np.isfinite(cape), cape, np.nan), shear


def surface_scalar(surface: Mapping[str, np.ndarray], name: str):
    """2 m dewpoint, theta-e or temperature (K) from t2 (K), q2 (kg/kg) and psfc (Pa)."""
    t2 = np.asarray(surface['t2'], float)
    if name == 'temperature':
        return t2
    q = np.maximum(np.asarray(surface['q2'], float), 1e-8)
    p = np.asarray(surface['psfc'], float)
    r = q / (1.0 - q)
    e = np.maximum(r * p / (0.622 + r), 1.0)
    if name == 'dewpoint':
        ln = np.log(e / 611.2)
        return 273.15 + 243.5 * ln / (17.67 - ln)
    if name == 'theta_e':
        return _theta_e(t2, p, r, _lcl_temperature(t2, r, p))
    raise ValueError(f'unknown boundary field {name!r} (dewpoint, theta_e, temperature)')


def _dominant_object(lattice, f, threshold, min_area_km2, search_km, distance_scale_km, size_power,
                     extra_score=None):
    """The thresholded object of ``f`` with the best size x nearness score: ((score, x, y, area, peak, mask), n)."""
    from scipy.ndimage import label
    mask = np.isfinite(f) & (f >= threshold)
    labels, n = label(mask)
    if n == 0:
        return None, 0
    cell = lattice.spacing ** 2
    excess = np.where(mask, f - threshold + 1e-6, 0.0)
    best = None
    for k in range(1, n + 1):
        m = labels == k
        area = m.sum() * cell
        if area < min_area_km2:
            continue
        w = excess * m
        xc, yc = float((w * lattice.x).sum() / w.sum()), float((w * lattice.y).sum() / w.sum())
        d = math.hypot(xc, yc)
        if d > search_km:
            continue
        score = float(w.sum() * cell) ** size_power * math.exp(-(d / distance_scale_km) ** 2)
        if extra_score is not None:
            score *= extra_score(m, w, xc, yc)
        if best is None or score > best[0]:
            best = (score, xc, yc, area, float(f[m].max()), m)
    return best, n


def _axis_deg(lattice, w):
    """Principal axis of a weighted object, degrees clockwise from north in [0, 180)."""
    s = w.sum()
    x0, y0 = (w * lattice.x).sum() / s, (w * lattice.y).sum() / s
    dx, dy = lattice.x - x0, lattice.y - y0
    cxx, cyy, cxy = (w * dx * dx).sum() / s, (w * dy * dy).sum() / s, (w * dx * dy).sum() / s
    angle = 0.5 * math.atan2(2 * cxy, cxx - cyy)        # from +x (east), counter-clockwise
    return (90.0 - math.degrees(angle)) % 180.0


class EnvFixer:
    """``cape_shear``: the dominant area of MLCAPE x 0-6 km bulk shear near the prediction."""

    def __init__(self, config: EnvConfig | None = None):
        self.c = config or EnvConfig()

    def fix(self, seconds, fields: StateFields, guess, search_km) -> Fix:
        c = self.c
        fx = Fix(seconds=float(seconds), ok=False, guess=[float(guess[0]), float(wrap_lon(guess[1]))],
                 search_km=float(search_km))
        if fields.columns is None:
            fx.reason = 'no model columns in the state (cape_shear needs pressure, theta, qv, u, v)'
            return fx
        cloud = fields.points()
        lattice = LocalLattice(cloud, guess[0], guess[1], search_km + 2 * c.smooth_km, c.lattice_spacing_km, k=1)
        idx = lattice.index[:, 0]
        cols = {k: np.asarray(v)[idx] for k, v in fields.columns.items()}
        cape, shear = mlcape_and_shear(cols, c.mixed_layer_pa, c.shear_depth_m)
        index = (cape / 1000.0) * (shear / 20.0)
        f = smooth(index.reshape(lattice.shape), c.smooth_km / c.lattice_spacing_km)
        best, n = _dominant_object(lattice, f, c.threshold, c.min_area_km2, search_km, c.distance_scale_km,
                                   c.size_power)
        if best is None:
            fx.reason = ('no CAPE x shear area above threshold' if n == 0
                         else 'no CAPE x shear area of sufficient size within the search radius')
            fx.components = dict(cape_shear=dict(
                max_index=float(np.nanmax(f)) if np.isfinite(f).any() else None, rejected=True))
            return fx
        _, xc, yc, area, peak, m = best
        local = m if c.core_reach_km is None else m & (np.hypot(lattice.x, lattice.y) <= c.core_reach_km)
        if local.any():
            peak = float(f[local].max())
        else:
            local = m
        core_level = max(c.threshold, c.core_fraction * peak)
        core = local & (f >= core_level)
        w = np.where(core, f - core_level + 1e-6, 0.0) if core.any() else np.where(local, f - c.threshold + 1e-6, 0.0)
        storm_km2, storm_share = 0.0, 0.0
        if fields.object_field is not None and c.storm_weight > 0:
            from scipy.ndimage import distance_transform_edt
            hot = (np.asarray(fields.object_field, float)[idx] >= c.storm_dbz).reshape(lattice.shape)
            near = (distance_transform_edt(~m) * c.lattice_spacing_km <= c.storm_reach_km) & (
                np.hypot(lattice.x, lattice.y) <= search_km)
            cover = smooth(hot.astype(float), c.smooth_km / c.lattice_spacing_km)
            ws = np.where(near, c.storm_weight * peak * cover, 0.0)
            storm_km2 = float((hot & near).sum() * c.lattice_spacing_km ** 2)
            if ws.sum() > 0:
                storm_share = float(ws.sum() / (ws.sum() + w.sum()))
                w = w + ws
        xc, yc = float((w * lattice.x).sum() / w.sum()), float((w * lattice.y).sum() / w.sum())
        dx, dy = c.fix_blend * xc, c.fix_blend * yc
        step = math.hypot(dx, dy)
        capped = c.max_step_km is not None and step > c.max_step_km
        if capped:
            dx, dy = dx * c.max_step_km / step, dy * c.max_step_km / step
        fx.lat, fx.lon = lattice.to_latlon(dx, dy)
        fx.ok = True
        fx.reason = ('core of the dominant CAPE x shear area with its storms' if storm_share > 0
                     else 'core of the dominant CAPE x shear area')
        cape2, shear2 = cape.reshape(lattice.shape), shear.reshape(lattice.shape)
        fx.components = dict(cape_shear=dict(
            area_km2=area, core_km2=float(core.sum() * c.lattice_spacing_km ** 2), peak_index=peak, objects=int(n),
            storm_km2=storm_km2, storm_share=storm_share, step_km=step, step_capped=bool(capped),
            raw_centre=list(lattice.to_latlon(xc, yc)),
            max_mlcape=float(np.nanmax(np.where(m, cape2, np.nan))),
            median_shear06=float(np.nanmedian(np.where(m, shear2, np.nan)))))
        fx.steering_ms = steering(fields, fx.lat, fx.lon, c.steering_inner_km, c.steering_outer_km)
        return fx


class BoundaryFixer:
    """``boundary``: the dominant dryline / front segment (surface-scalar gradient) near the prediction."""

    def __init__(self, config: BoundaryConfig | None = None):
        self.c = config or BoundaryConfig()
        self.orientation = None
        if self.c.segment:
            (a_lat, a_lon), (b_lat, b_lon) = self.c.segment
            x, y = project(a_lat, a_lon, b_lat, b_lon)
            self.orientation = math.degrees(math.atan2(float(x), float(y))) % 180.0

    def fix(self, seconds, fields: StateFields, guess, search_km) -> Fix:
        c = self.c
        fx = Fix(seconds=float(seconds), ok=False, guess=[float(guess[0]), float(wrap_lon(guess[1]))],
                 search_km=float(search_km))
        if fields.surface is None:
            fx.reason = 'no surface fields in the state (boundary needs t2, q2, psfc)'
            return fx
        cloud = fields.points()
        lattice = LocalLattice(cloud, guess[0], guess[1], search_km + 3 * c.smooth_km, c.lattice_spacing_km)
        scalar = smooth(lattice.sample(surface_scalar(fields.surface, c.field)), c.smooth_km / c.lattice_spacing_km)
        gy, gx = np.gradient(scalar, lattice.spacing / 100.0)     # K per 100 km
        grad = np.hypot(gx, gy)
        orientation = self.orientation

        def aligned(m, w, xc, yc):
            if orientation is None or c.orientation_weight <= 0:
                return 1.0
            return abs(math.cos(math.radians(_axis_deg(lattice, w) - orientation))) ** c.orientation_weight + 1e-3

        best, n = _dominant_object(lattice, grad, c.threshold, c.min_area_km2, search_km, c.distance_scale_km,
                                   c.size_power, extra_score=aligned)
        if best is None:
            fx.reason = (f'no {c.field} gradient above {c.threshold:g} K/100 km' if n == 0
                         else f'no {c.field} gradient segment of sufficient size within the search radius')
            return fx
        _, xc, yc, area, peak, m = best
        axis = _axis_deg(lattice, np.where(m, grad - c.threshold, 0.0))
        self.orientation = axis          # a boundary turns slowly: follow its own orientation from here on
        fx.lat, fx.lon = lattice.to_latlon(c.fix_blend * xc, c.fix_blend * yc)
        fx.ok = True
        fx.reason = f'dominant {c.field} gradient segment'
        fx.components = dict(boundary=dict(field=c.field, area_km2=area, peak_k_per_100km=peak, objects=int(n),
                                           axis_deg=axis, raw_centre=list(lattice.to_latlon(xc, yc))))
        fx.steering_ms = steering(fields, fx.lat, fx.lon, c.steering_inner_km, c.steering_outer_km)
        return fx


def path_position(rows, seconds):
    t = np.array([float(r[0]) for r in rows])
    lat = np.array([float(r[1]) for r in rows])
    lon = np.degrees(np.unwrap(np.radians([float(r[2]) for r in rows])))   # the short way across 180
    return float(np.interp(seconds, t, lat)), float(wrap_lon(np.interp(seconds, t, lon)))


class PathFixer:
    """``path``: the user's own path, by time. The region goes where the user said; nothing is diagnosed."""

    def __init__(self, config: PathConfig):
        if len(config.rows) < 1:
            raise ValueError('a path target needs at least one [seconds, lat, lon] row')
        self.c = config

    def fix(self, seconds, fields, guess, search_km) -> Fix:
        lat, lon = path_position(self.c.rows, seconds)
        ahead = path_position(self.c.rows, seconds + 1800.0)
        x, y = project(lat, lon, ahead[0], ahead[1])
        return Fix(seconds=float(seconds), ok=True, lat=lat, lon=lon, guess=[float(guess[0]), float(guess[1])],
                   reason="the user's path (not a tracker)",
                   steering_ms=[float(x) * 1000.0 / 1800.0, float(y) * 1000.0 / 1800.0])


TRACKER_MODES = ('tc', 'object', 'low', 'cape_shear', 'boundary', 'path')


# --------------------------------------------------------------------------- motion

@dataclass
class MotionConfig:
    window_hours: float = 6.0
    recency_hours: float = 3.0
    full_history_hours: float = 6.0   # history span at which the fit alone is trusted
    max_speed_kmh: float = 80.0


class MotionModel:
    def __init__(self, config: MotionConfig | None = None):
        self.c = config or MotionConfig()

    def velocity(self, fixes: Sequence[Fix]):
        """(east, north) km/h, and the fraction from the track fit."""
        good = [f for f in fixes if f.ok]
        if not good:
            return np.zeros(2), 0.0
        last = good[-1]
        steer = np.array(last.steering_ms if last.steering_ms else [0.0, 0.0]) * 3.6
        recent = [f for f in good if last.seconds - f.seconds <= self.c.window_hours * 3600]
        span = (last.seconds - recent[0].seconds) / 3600.0
        if len(recent) < 2 or span <= 0:
            v, alpha = steer, 0.0
        else:
            t = np.array([(f.seconds - last.seconds) / 3600.0 for f in recent])
            x, y = project(last.lat, last.lon, np.array([f.lat for f in recent]), np.array([f.lon for f in recent]))
            w = np.exp(t / self.c.recency_hours)
            A = np.stack([np.ones_like(t), t], 1) * np.sqrt(w)[:, None]
            fit = np.array([np.linalg.lstsq(A, q * np.sqrt(w), rcond=None)[0][1] for q in (x, y)])
            alpha = min(1.0, span / self.c.full_history_hours)
            v = alpha * fit + (1 - alpha) * steer
        speed = float(np.hypot(*v))
        if speed > self.c.max_speed_kmh:
            v = v * self.c.max_speed_kmh / speed
        return v, alpha

    def predict(self, fixes, seconds_ahead):
        good = [f for f in fixes if f.ok]
        last = good[-1]
        v, _ = self.velocity(fixes)
        h = seconds_ahead / 3600.0
        lat, lon = unproject(last.lat, last.lon, v[0] * h, v[1] * h)
        return float(lat), float(lon)


# --------------------------------------------------------------------------- placement

@dataclass
class PlacementConfig:
    interior_radius_km: float          # constant-spacing fine interior (cap radius minus its ramp)
    core_margin_km: float = 150.0      # storm core that must stay inside the interior
    check_minutes: float = 30.0        # decision granularity; whole macro steps
    max_dwell_minutes: float = 360.0   # longest a segment-mode controller waits between checks
    max_lead_minutes: float = 900.0
    snap_deg: float = 0.25
    fix_error_km: float = 25.0
    motion_error_kmh: float = 6.0
    min_move_km: float = 20.0
    max_move_km: float | None = None    # optional cap on one move (smaller remap zones, more frequent moves)
    follow: str = 'lazy'                # 'lazy': move only when the region would fail before the next check
                                        # 'continuous': re-centre at every check where the best (capped) centre is
                                        # >= min_move_km from the predicted position continuous_lead_minutes ahead
    continuous_lead_minutes: float = 120.0
    prepared_tolerance_km: float = 40.0 # continuous mode: an available centre this close to the ideal step is used
    prefer_prepared_fraction: float = 0.75
    max_misses: int = 3

    def __post_init__(self):
        # THE BREAKAGE THIS PREVENTS (TC-the storm storm-following case, 2026-10-26: a 146 km interior
        # under the default 150 km core margin): with no safe radius every dwell is unsafe, so the
        # case was accepted here and failed only after its first member preparation had used the card.
        if self.safe_km() <= 0:
            raise ValueError(
                f'storm-following placement has no safe radius: interior_radius_km '
                f'{self.interior_radius_km:g} minus core_margin_km {self.core_margin_km:g} is '
                f'{self.safe_km():g} km, so the storm core can never be inside the fine interior; set '
                f'tracker.placement.core_margin_km below the interior radius (or use a larger interior)')

    def safe_km(self):
        return self.interior_radius_km - self.core_margin_km


@dataclass
class Decision:
    seconds: float
    move: bool
    centre: list
    next_check_seconds: float
    dwell_minutes: float
    reason: str
    predicted_path: list = field(default_factory=list)
    candidates_considered: int = 0
    used_prepared: bool = False
    status: str = 'tracking'

    def as_dict(self):
        return asdict(self)


def snap(lat, lon, step):
    if step <= 0:
        return float(lat), float(wrap_lon(lon))
    return float(round(lat / step) * step), float(wrap_lon(round(lon / step) * step))


class PlacementPolicy:
    SUBSTEP_MIN = 10.0

    def __init__(self, config: PlacementConfig, motion: MotionModel | None = None):
        self.c = config
        self.motion = motion or MotionModel()

    def _path(self, origin, velocity, minutes):
        h = np.asarray(minutes, float) / 60.0
        lat, lon = unproject(origin[0], origin[1], velocity[0] * h, velocity[1] * h)
        return np.atleast_1d(lat), np.atleast_1d(lon)

    def dwell(self, centre, origin, velocity, start_minutes=0.0, horizon_minutes=None):
        """Largest whole number of check periods for which the predicted core stays safe (minutes; -1 if unsafe now).

        Capped at horizon_minutes (default max_dwell_minutes)."""
        c = self.c
        horizon = c.max_dwell_minutes if horizon_minutes is None else horizon_minutes
        s = np.arange(0.0, horizon + 1e-9, self.SUBSTEP_MIN)
        lat, lon = self._path(origin, velocity, start_minutes + s)
        sigma = c.fix_error_km + c.motion_error_kmh * (start_minutes + s) / 60.0
        ok = distance_km(centre[0], centre[1], lat, lon) + sigma <= c.safe_km()
        if not ok[0]:
            return -1.0
        bad = np.flatnonzero(~ok)
        limit = horizon if bad.size == 0 else s[bad[0]] - self.SUBSTEP_MIN
        return float(math.floor(limit / c.check_minutes + 1e-9) * c.check_minutes)

    def capped_step(self, current, target):
        """Lattice point nearest the target, no further than max_move_km from current (target itself if no cap)."""
        c = self.c
        tlat, tlon = target
        d = float(distance_km(current[0], current[1], tlat, tlon))
        if c.max_move_km and d > c.max_move_km:
            x, y = project(current[0], current[1], tlat, tlon)
            k = (c.max_move_km - 1e-6) / d
            tlat, tlon = (float(v) for v in unproject(current[0], current[1], x * k, y * k))
        step = c.snap_deg if c.snap_deg > 0 else 1e-9
        corners = [(float(fa(tlat / step) * step), float(wrap_lon(fb(tlon / step) * step)))
                   for fa in (math.floor, math.ceil) for fb in (math.floor, math.ceil)]
        if c.max_move_km:
            inside = [q for q in corners if distance_km(current[0], current[1], *q) <= c.max_move_km + 1e-6]
            corners = inside or [tuple(map(float, current))]
        return min(corners, key=lambda q: float(distance_km(tlat, tlon, *q)))

    @staticmethod
    def _excluded(q, excluded):
        return any(distance_km(q[0], q[1], e[0], e[1]) < 1.0 for e in excluded)

    def choose(self, origin, velocity, current, available=(), start_minutes=0.0, excluded=()):
        """Best centre for a move made start_minutes after origin's time (never an excluded centre)."""
        c = self.c
        leads = np.arange(0.0, c.max_lead_minutes + 1e-9, c.check_minutes)
        lat, lon = self._path(origin, velocity, start_minutes + leads)
        cands = {snap(a, b, c.snap_deg) for a, b in zip(lat, lon)}
        cands |= {tuple(map(float, a)) for a in available}
        if c.max_move_km:
            # Also offer the furthest allowed step toward each predicted position, then drop longer moves.
            for a, b in zip(lat, lon):
                d = float(distance_km(current[0], current[1], a, b))
                if d > c.max_move_km:
                    x, y = project(current[0], current[1], a, b)
                    k = (c.max_move_km - 1e-6) / d
                    qa, qb = unproject(current[0], current[1], x * k, y * k)
                    # All four surrounding lattice points: snapping to the nearest alone can overshoot the cap.
                    step = c.snap_deg if c.snap_deg > 0 else 1e-9
                    for fa in (math.floor, math.ceil):
                        for fb in (math.floor, math.ceil):
                            cands.add((float(fa(float(qa) / step) * step), float(wrap_lon(fb(float(qb) / step) * step))))
            limited = {q for q in cands if distance_km(current[0], current[1], q[0], q[1]) <= c.max_move_km + 1e-6}
            cands = limited or cands
        if excluded:
            # Members that failed (e.g. no converged overlap operator) and their exact lattice points are skipped;
            # add the lattice neighbours of each excluded point so a nearby alternative is always on offer.
            step = c.snap_deg if c.snap_deg > 0 else 0.25
            for e in excluded:
                for da in (-step, 0.0, step):
                    for db in (-step, 0.0, step):
                        q = (float(e[0] + da), float(wrap_lon(e[1] + db)))
                        if not c.max_move_km or distance_km(current[0], current[1], *q) <= c.max_move_km + 1e-6:
                            cands.add(q)
            cands = {q for q in cands if not self._excluded(q, excluded)} or cands
            available = [q for q in available if not self._excluded(q, excluded)]
        scored = []
        # Rank by the uncapped time to the next forced move (fewest moves), so a centre that
        # leads the storm wins over one that merely contains it until the check cap.
        for cand in cands:
            d = self.dwell(cand, origin, velocity, start_minutes, c.max_lead_minutes)
            moved = float(distance_km(current[0], current[1], cand[0], cand[1]))
            scored.append((d, -moved, cand in {tuple(map(float, a)) for a in available}, cand))
        best = max(scored, key=lambda r: (r[0], r[1]))
        prepared = [r for r in scored if r[2] and r[0] >= max(2 * c.check_minutes, c.prefer_prepared_fraction * best[0])]
        if prepared:
            pick = max(prepared, key=lambda r: (r[0], r[1]))
            return pick[3], min(pick[0], c.max_dwell_minutes), len(scored), True
        return best[3], min(best[0], c.max_dwell_minutes), len(scored), False

    def decide(self, seconds, fixes: Sequence[Fix], region, available=(), misses=0, excluded=()) -> Decision:
        c = self.c
        region = [float(region[0]), float(wrap_lon(region[1]))]
        if misses > c.max_misses:
            return Decision(seconds, False, region, seconds + c.check_minutes * 60, 0.0,
                            f'storm lost: {misses} consecutive failed fixes; region frozen', status='lost')
        good = [f for f in fixes if f.ok]
        if not good:
            return Decision(seconds, False, region, seconds + c.check_minutes * 60, 0.0, 'no fix yet; holding',
                            status='searching')
        velocity, _ = self.motion.velocity(fixes)
        last = good[-1]
        lag = (seconds - last.seconds) / 60.0  # minutes since the last good fix (dead reckoning when > 0)
        origin = (last.lat, last.lon)
        path = [list(map(float, p)) for p in zip(*self._path(origin, velocity, lag + np.array([0, 60, 180, 360])))]
        if misses:
            # A missed fix: never move on dead reckoning alone; re-check at the next opportunity.
            return Decision(seconds, False, region, seconds + c.check_minutes * 60, 0.0,
                            f'fix missed ({misses}/{c.max_misses}); holding the region', path, status='holding')
        stay = self.dwell(region, origin, velocity, lag)
        if c.follow == 'continuous':
            # Keep the storm near the centre, a short lead ahead of it, in steps of at most max_move_km.
            target = self._path(origin, velocity, lag + c.continuous_lead_minutes)
            target = (float(target[0][0]), float(target[1][0]))
            centre = self.capped_step(region, target)
            if excluded and self._excluded(centre, excluded):
                centre, _, _, _ = self.choose(origin, velocity, region, available, lag, excluded)
            prepared = False
            ideal = centre
            ideal_step = float(distance_km(region[0], region[1], ideal[0], ideal[1]))
            to_target = float(distance_km(region[0], region[1], *target))
            # A prepared member may replace the ideal step only when the ideal step itself is due, and only if it
            # brings the region closer to the target (otherwise two members can alternate: measured live).
            for q in (available if ideal_step >= c.min_move_km else ()):
                if excluded and self._excluded(q, excluded):
                    continue
                q = (float(q[0]), float(wrap_lon(q[1])))
                step = float(distance_km(region[0], region[1], *q))
                if (distance_km(ideal[0], ideal[1], *q) <= c.prepared_tolerance_km and step >= c.min_move_km
                        and distance_km(q[0], q[1], *target) < to_target - 0.5 * c.min_move_km
                        and (not c.max_move_km or step <= c.max_move_km + 1e-6)
                        and self.dwell(q, origin, velocity, lag) >= c.check_minutes):
                    if not prepared or distance_km(ideal[0], ideal[1], *q) < distance_km(ideal[0], ideal[1], *centre):
                        centre, prepared = q, True
            moved = float(distance_km(region[0], region[1], centre[0], centre[1]))
            dwell = self.dwell(centre, origin, velocity, lag)
            if moved >= c.min_move_km and dwell >= c.check_minutes:
                return Decision(seconds, True, [float(centre[0]), float(centre[1])], seconds + c.check_minutes * 60,
                                dwell, f'continuous follow: move {moved:.0f} km toward the storm '
                                       f'{c.continuous_lead_minutes:.0f} min ahead' + (' (prepared member)' if prepared else ''),
                                path, 1, prepared)
            if stay >= c.check_minutes:
                return Decision(seconds, False, region, seconds + c.check_minutes * 60, stay,
                                f'continuous follow: lead target {moved:.0f} km away; holding', path, 1)
            # Otherwise fall through to the lazy rule (the capped step cannot hold the core): best centre now.
        if stay >= c.check_minutes:
            return Decision(seconds, False, region, seconds + stay * 60, stay,
                            f'storm safe in the current region for {stay:.0f} min', path)
        centre, dwell, n, prepared = self.choose(origin, velocity, region, available, lag, excluded)
        moved = float(distance_km(region[0], region[1], centre[0], centre[1]))
        if moved < c.min_move_km and stay >= 0:
            return Decision(seconds, False, region, seconds + c.check_minutes * 60, max(stay, 0.0),
                            f'best centre only {moved:.0f} km away; holding', path, n)
        reason = f'current region safe for {max(stay, 0):.0f} min < one check; move {moved:.0f} km'
        if dwell < c.check_minutes:
            reason += '; WARNING new region cannot hold the predicted core for one check period'
        return Decision(seconds, True, [float(centre[0]), float(centre[1])],
                        seconds + max(dwell, c.check_minutes) * 60, dwell, reason, path, n, prepared)

    def anticipate_chain(self, decision: Decision, fixes, region, count=2):
        """The next `count` moves the policy expects if the storm follows the prediction (each from the previous)."""
        out = []
        d = decision
        for _ in range(count):
            a = self.anticipate(d, fixes, d.centre)
            if a is None:
                break
            out.append(a)
            d = Decision(a['at_seconds'], True, a['centre'], a['at_seconds'], 0.0, 'anticipated')
        return out

    def anticipate(self, decision: Decision, fixes, region, *, velocity=None):
        """Where and when the next move will most likely happen, if the storm follows the prediction.

        The time is the uncapped predicted dwell of the region in effect after this decision (up to
        max_lead_minutes), not the segment check cap, so a controller whose member preparation takes many
        forecast hours can start that member now. Returns dict(centre, at_seconds, dwell_minutes) or None."""
        good = [f for f in fixes if f.ok]
        if not good or decision.status != 'tracking':
            return None
        c = self.c
        if velocity is None:
            velocity, _ = self.motion.velocity(fixes)
        last = good[-1]
        origin = (last.lat, last.lon)
        lag = (decision.seconds - last.seconds) / 60.0
        current = decision.centre
        if c.follow == 'continuous':
            # Next check at which the lead target is at least min_move_km away.
            for k in range(1, int(c.max_lead_minutes // c.check_minutes) + 1):
                ahead = lag + k * c.check_minutes + c.continuous_lead_minutes
                path = self._path(origin, velocity, ahead)
                centre = self.capped_step(current, (float(path[0][0]), float(path[1][0])))
                if distance_km(current[0], current[1], *centre) >= c.min_move_km:
                    return dict(centre=list(centre), at_seconds=decision.seconds + k * c.check_minutes * 60.0,
                                dwell_minutes=self.dwell(centre, origin, velocity, lag + k * c.check_minutes),
                                move_in_minutes=k * c.check_minutes)
            return None
        move_in = max(self.dwell(current, origin, velocity, lag, c.max_lead_minutes), 0.0)
        centre, dwell, _, _ = self.choose(origin, velocity, current, (), lag + move_in)
        return dict(centre=list(centre), at_seconds=decision.seconds + move_in * 60.0, dwell_minutes=dwell,
                    move_in_minutes=move_in)


# --------------------------------------------------------------------------- session

@dataclass
class TrackerConfig:
    mode: str = 'tc'                  # one of TRACKER_MODES
    initial_center_deg: list | None = None
    tc: TCConfig = field(default_factory=TCConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    low: TCConfig = field(default_factory=low_config)
    cape_shear: EnvConfig = field(default_factory=EnvConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    path: PathConfig = field(default_factory=PathConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    placement: PlacementConfig | None = None

    @classmethod
    def from_dict(cls, d: Mapping):
        d = dict(d)
        out = cls(mode=d.get('mode', 'tc'), initial_center_deg=d.get('initial_center_deg'))
        out.tc = TCConfig(**d.get('tc', {}))
        out.object = ObjectConfig(**d.get('object', {}))
        out.low = low_config(**d.get('low', {}))
        out.cape_shear = EnvConfig(**d.get('cape_shear', {}))
        out.boundary = BoundaryConfig(**d.get('boundary', {}))
        out.path = PathConfig(**d.get('path', {}))
        out.motion = MotionConfig(**d.get('motion', {}))
        if d.get('placement'):
            out.placement = PlacementConfig(**d['placement'])
        return out

    def target(self):
        """The active mode's own config block."""
        return getattr(self, self.mode)

    def as_dict(self):
        return asdict(self)


class TrackSession:
    """Run-time tracker state. A controller calls observe() on model state, then decide()."""

    def __init__(self, config: TrackerConfig):
        if config.initial_center_deg is None:
            raise ValueError('the analysis-time storm position is required')
        if config.mode not in TRACKER_MODES:
            raise ValueError(f'unknown tracker mode {config.mode} (one of {", ".join(TRACKER_MODES)})')
        self.config = config
        self.fixer = {'tc': TCFixer, 'low': TCFixer, 'object': ObjectFixer, 'cape_shear': EnvFixer,
                      'boundary': BoundaryFixer, 'path': PathFixer}[config.mode](config.target())
        self.motion = MotionModel(config.motion)
        self.policy = PlacementPolicy(config.placement, self.motion) if config.placement else None
        self.fixes: list[Fix] = []
        self.decisions: list[Decision] = []
        self.misses = 0
        self.lost = False

    def predicted(self, seconds):
        good = [f for f in self.fixes if f.ok]
        if not good:
            return tuple(self.config.initial_center_deg), 0.0
        return self.motion.predict(self.fixes, seconds - good[-1].seconds), (seconds - good[-1].seconds) / 3600.0

    def search_radius(self, hours_since_fix):
        c = self.config.target()
        return c.search_km + c.search_growth_kmh * max(hours_since_fix, 0.0)

    def observe(self, seconds, fields: StateFields) -> Fix:
        guess, since = self.predicted(seconds)
        fix = self.fixer.fix(seconds, fields, guess, self.search_radius(since))
        self.fixes.append(fix)
        if fix.ok:
            self.misses = 0
        else:
            self.misses += 1
            if self.policy and self.misses > self.policy.c.max_misses:
                self.lost = True
        return fix

    def decide(self, seconds, region, available=(), excluded=()) -> Decision:
        if self.policy is None:
            raise ValueError('no placement config')
        d = self.policy.decide(seconds, self.fixes, region, available, misses=(10 ** 6 if self.lost else self.misses),
                               excluded=excluded)
        self.decisions.append(d)
        return d

    def anticipate(self, decision, region):
        return self.policy.anticipate(decision, self.fixes, region) if self.policy else None

    def anticipate_chain(self, decision, region, count=2):
        return self.policy.anticipate_chain(decision, self.fixes, region, count) if self.policy else []

    def to_json(self):
        return dict(config=self.config.as_dict(), fixes=[f.as_dict() for f in self.fixes],
                    decisions=[d.as_dict() for d in self.decisions], lost=self.lost)

    def save(self, path):
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_json(), indent=1, default=float) + '\n')

    @classmethod
    def restore(cls, data: Mapping):
        """Rebuild a session from to_json() output (a controller resuming between segments)."""
        s = cls(TrackerConfig.from_dict(data['config']))
        for f in data.get('fixes', []):
            f = dict(f)
            s.fixes.append(Fix(**{k: f.get(k) for k in Fix.__dataclass_fields__ if k in f}))
        for d in data.get('decisions', []):
            s.decisions.append(Decision(**d))
        s.lost = bool(data.get('lost'))
        s.misses = 0
        for f in reversed(s.fixes):
            if f.ok:
                break
            s.misses += 1
        return s


def history_needs(config: TrackerConfig) -> dict:
    """What a mode reads from a history frame beyond the cyclone fields: its object field, model columns, 2 m fields.

    Breakage it prevents: a setup target whose fields the controller never read answers 'no model columns in the
    state' at every frame and the region never moves (the object-mode lesson, derecho 1 km run, 2026-10-17)."""
    return dict(object_field=config.object.field_name if config.mode == 'object' else None,
                columns=config.mode == 'cape_shear', surface=config.mode == 'boundary',
                state=config.mode != 'path',
                # cape_shear reads its storms too, when the frame has the field (EnvConfig.storm_field)
                storm_field=config.cape_shear.storm_field if config.mode == 'cape_shear' else None)


def interior_radius_from_mesh(mesh: Mapping) -> float:
    """Constant-spacing fine interior radius from a case 'mesh' block (cap radius minus its ramp)."""
    return float(mesh['fine_radius_km']) - float(mesh.get('fine_transition_km', 0.0))
