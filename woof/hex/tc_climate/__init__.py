"""Opt-in tropical cyclone diagnostics. No atmospheric state is modified.

GPI is a diagnostic index, not a calibrated storm count or probability.
The tracker consumes device fields in native mesh order, at six-hour cadence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math


def genesis_potential_index(xp, absolute_vorticity_850, rh600_percent,
                            potential_intensity_ms, shear_850_200_ms):
    """Camargo et al. (2007), Eq. 1, doi:10.1175/JCLI4282.1.

    Inputs are s^-1, percent, m/s and m/s. PI must be supplied by a documented
    profile calculation, never inferred from an SST threshold. Invalid humidity,
    negative speeds and missing inputs yield NaN. Zero PI yields zero GPI.
    NumPy supports reference tests; the forecast caller supplies CuPy.
    """
    eta, rh, pi, shear = (xp.asarray(a) for a in (
        absolute_vorticity_850, rh600_percent, potential_intensity_ms, shear_850_200_ms))
    if any(a.shape != eta.shape for a in (rh, pi, shear)):
        raise ValueError("GPI fields must have identical native-cell shapes")
    valid = (xp.isfinite(eta) & xp.isfinite(rh) & xp.isfinite(pi) & xp.isfinite(shear)
             & (rh >= 0) & (rh <= 100) & (pi >= 0) & (shear >= 0))
    value = (xp.abs(1e5 * eta) ** 1.5 * (rh / 50) ** 3 *
             (xp.maximum(pi, 0) / 70) ** 3 / (1 + .1 * xp.maximum(shear, 0)) ** 2)
    return xp.where(valid, value, xp.nan)


@dataclass(frozen=True)
class Recipe:
    """Thresholds are loaded from a row; no storm or basin lives in code."""
    pressure_delta_pa: float
    pressure_radius_deg: float
    thickness_delta_m: float
    thickness_radius_deg: float
    warm_offset_deg: float
    merge_deg: float
    wind_radius_deg: float
    link_deg: float
    max_gap_hours: float
    min_duration_hours: float
    min_wind_ms: float
    max_abs_lat_deg: float
    max_terrain_m: float
    min_qualified_samples: int
    cadence_hours: float
    candidate_capacity: int
    flood_capacity: int
    pi_ckcd: float
    pi_wind_reduction: float
    pi_top_hpa: float

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for name in ('candidate_capacity', 'flood_capacity', 'min_qualified_samples'):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.flood_capacity & (self.flood_capacity - 1):
            raise ValueError("flood_capacity must be a power of two")
        if self.flood_capacity > 65536 or self.candidate_capacity > 4096:
            raise ValueError("capacities exceed bounded diagnostic workspace")
        if any(getattr(self, k) >= 180 for k in asdict(self) if k.endswith('_deg')):
            raise ValueError("angular radii must be less than 180 degrees")
        if self.max_gap_hours < self.cadence_hours:
            raise ValueError("max_gap_hours must cover one sampling interval")
        if self.max_abs_lat_deg > 90 or self.pi_wind_reduction > 1:
            raise ValueError("invalid latitude or PI wind reduction")
        if not math.isclose(24 / self.cadence_hours, round(24 / self.cadence_hours)):
            raise ValueError("tracker cadence must divide the daily GPI interval")

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else Path(__file__).with_name('recipes.json')
        return cls(**json.loads(path.read_text())['recipe'])


def workspace_allowance(cells, levels, samples, recipe):
    """Conservative planning allowance, NOT a measured lease budget.

    Includes profile copies/interpolation temporaries, graph floods, pair sort,
    all retained compact records, and a compiler/allocator reserve. A smoke
    must still measure the actual process peak before any production lease.
    """
    c, f = recipe.candidate_capacity, recipe.flood_capacity
    return int(80*cells*levels + 64*cells + 16*c*f + 64*c*c + 64*c*samples + 64*1024**2)


class DeviceTracker:
    """Bounded GPU graph detector and online one-to-one track association.

    All work arrays remain on device. observe() does not fetch scalars, compact
    on the host or synchronize. Download only at an existing output boundary.
    Overflow status bits: 1 candidate list, 2 flood queue, 4 active tracks.
    Any nonzero status invalidates that diagnostic batch.
    """
    def __init__(self, cells_on_cell, n_edges_on_cell, latitude_rad, longitude_rad,
                 terrain_m, recipe=None):
        import cupy as cp
        self.cp = cp
        self.recipe = recipe or Recipe.load()
        r = self.recipe
        self.n = int(latitude_rad.size)
        self.graph = cp.ascontiguousarray(cp.asarray(cells_on_cell), dtype=cp.int32)
        self.counts = cp.ascontiguousarray(cp.asarray(n_edges_on_cell), dtype=cp.int32)
        self.lat = cp.ascontiguousarray(cp.asarray(latitude_rad), dtype=cp.float32)
        self.lon = cp.ascontiguousarray(cp.asarray(longitude_rad), dtype=cp.float32)
        self.terrain = cp.ascontiguousarray(cp.asarray(terrain_m), dtype=cp.float32)
        if self.graph.ndim != 2 or self.graph.shape[0] != self.n:
            raise ValueError('cellsOnCell must be zero-based [cell, neighbor], padding -1')
        if any(a.shape != (self.n,) for a in (self.counts, self.lat, self.lon, self.terrain)):
            raise ValueError('geometry fields must share native cell order')
        # Geometry validation is setup-only and has one explicit transfer.
        bad = (((self.counts < 0) | (self.counts > self.graph.shape[1])).any()
               | (~cp.isfinite(self.lat) | ~cp.isfinite(self.lon)
                  | ~cp.isfinite(self.terrain) | (cp.abs(self.lat) > math.pi/2)).any())
        if bool(bad):
            raise ValueError('invalid geometry or nEdgesOnCell outside adjacency width')
        self.xyz = cp.ascontiguousarray(cp.stack((cp.cos(self.lat)*cp.cos(self.lon),
                        cp.cos(self.lat)*cp.sin(self.lon), cp.sin(self.lat)), axis=1))
        self.module = cp.RawModule(code=Path(__file__).with_name('kernels.cu').read_text(),
                                   options=('--std=c++17',))
        c, f = r.candidate_capacity, r.flood_capacity
        self.mask = cp.empty(self.n, cp.int32)
        self.candidates = cp.full(c, -1, cp.int32)
        self.accepted = cp.zeros(c, cp.int32)
        self.keep = cp.zeros(c, cp.int32)
        self.vmax = cp.empty(c, cp.float32)
        self.queues = cp.empty((c, f), cp.int32)
        self.hashes = cp.empty((c, 2*f), cp.int32)
        self.errors = cp.zeros(1, cp.int32)
        self.lastcell = cp.full(c, -1, cp.int32)
        self.lasttime = cp.zeros(c, cp.float64)
        self.active_id = cp.full(c, -1, cp.int64)
        self.nextid = cp.zeros(1, cp.int64)
        self.used = cp.zeros(c, cp.int32)
        self.ids = cp.full(c, -1, cp.int64)
        self.distance = cp.empty(c*c, cp.float32)
        self.previous_time = None

    def _call(self, name, size, args):
        self.module.get_function(name)(((size+127)//128,), (128,), args)

    def observe(self, time_hours, mslp_pa, thickness_300_500_m, wind10_ms):
        cp, r = self.cp, self.recipe
        if not math.isfinite(time_hours):
            raise ValueError('time must be finite hours since run origin')
        if self.previous_time is not None and not math.isclose(
                time_hours-self.previous_time, r.cadence_hours, abs_tol=1e-6):
            raise ValueError('tracker calls must follow the configured cadence, including empty frames')
        fields = (mslp_pa, thickness_300_500_m, wind10_ms)
        if any(not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or
               a.shape != (self.n,) or not a.flags.c_contiguous for a in fields):
            raise ValueError('tracker fields must be contiguous device float32 [cell]')
        self.previous_time = time_hours
        c, f, n, d = r.candidate_capacity, r.flood_capacity, self.n, self.graph.shape[1]
        cosine = lambda x: cp.float32(math.cos(math.radians(x)))
        self.candidates.fill(-1)
        self._call('minima', n, (n,d,self.graph,self.counts,mslp_pa,self.mask))
        prefix = cp.cumsum(self.mask, dtype=cp.int32)
        self._call('compact', n, (n,c,self.mask,prefix,self.candidates,self.errors))
        self._call('detect', c, (n,d,c,f,self.graph,self.counts,self.xyz,*fields,
            self.candidates,cp.float32(r.pressure_delta_pa),cosine(r.pressure_radius_deg),
            cp.float32(r.thickness_delta_m),cosine(r.thickness_radius_deg),
            cosine(r.warm_offset_deg),cosine(r.wind_radius_deg),self.queues,self.hashes,
            self.accepted,self.vmax,self.errors))
        self._call('merge', c, (c,self.candidates,self.accepted,self.xyz,mslp_pa,
                               cosine(r.merge_deg),self.keep))
        self._call('pairs', c*c, (c,self.candidates,self.keep,self.lastcell,self.lasttime,
            cp.float64(time_hours),cp.float64(r.max_gap_hours),self.xyz,cosine(r.link_deg),self.distance))
        # Stable ordering fixes equal-distance assignments by track slot then cell.
        order = cp.argsort(self.distance, kind='stable').astype(cp.int64, copy=False)
        self.module.get_function('stitch')((1,), (1,), (c,self.candidates,self.keep,
            self.distance,order,cp.float64(time_hours),cp.float64(r.max_gap_hours),
            self.lastcell,self.lasttime,self.active_id,self.nextid,self.used,self.ids,self.errors))
        index = cp.maximum(self.candidates, 0)
        return dict(time_hours=float(time_hours), candidate_count=prefix[-1:].copy(), ids=self.ids.copy(), cells=self.candidates.copy(),
            latitude_deg=cp.rad2deg(self.lat[index]), longitude_deg=cp.rad2deg(self.lon[index]),
            mslp_pa=mslp_pa[index], wind10_ms=self.vmax.copy(), terrain_m=self.terrain[index],
            errors=self.errors.copy())


def qualify_tracks(records, recipe):
    """Final small-table reduction outside the model loop.

    Online association is a deterministic nearest-pair variant, not a claim of
    byte-identical TempestExtremes trajectories. Qualification follows the 2021
    recipe's separate per-track threshold counts and elapsed duration.
    """
    groups = {}
    for record in records:
        groups.setdefault(int(record['track_id']), []).append(record)
    kept = []
    for tid, points in sorted(groups.items()):
        points.sort(key=lambda p: p['time_hours'])
        duration = points[-1]['time_hours']-points[0]['time_hours']
        checks = [sum(p['wind10_ms'] >= recipe.min_wind_ms for p in points),
                  sum(p['latitude_deg'] <= recipe.max_abs_lat_deg for p in points),
                  sum(p['latitude_deg'] >= -recipe.max_abs_lat_deg for p in points),
                  sum(p['terrain_m'] <= recipe.max_terrain_m for p in points)]
        if duration >= recipe.min_duration_hours and min(checks) >= recipe.min_qualified_samples:
            kept.append(dict(track_id=tid, duration_hours=duration, points=points))
    return kept
