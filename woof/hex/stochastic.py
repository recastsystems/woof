"""GPU stochastic physics on spherical Voronoi meshes, written from the papers.

SPPT: Buizza, Miller & Palmer (1999, QJRMS 125, 2887-2908) and Palmer et al.
(2009, ECMWF Tech. Memo. 598): the net physics tendency is multiplied by
(1+r), r a smooth random pattern clipped so 1+r stays in [0,2], tapered to
zero near the surface and in the stratosphere, and reverted to the
unperturbed tendency wherever the perturbed humidity tendency would make the
air supersaturated or the humidity negative. Several independent length and
time scales can be summed (Palmer et al. 2009, multi-scale SPPT).

SKEB: Shutts (2005, QJRMS 131, 3079-3102) and Berner et al. (2009, JAS 66,
603-626): a streamfunction forcing pattern whose rotational wind injects a
fraction of the dissipated kinetic energy back into the resolved flow.

Pattern: an AR(1)-in-time field of real spherical-harmonic coefficients
with a Gaussian variance spectrum, so the spatial correlation is
exp(-r**2/(2*length_m**2)) (the Gaussian correlation model on the sphere of
Weaver & Courtier 2001, QJRMS 127, 1815-1846) and the lag correlation is
exp(-dt/time_s). The harmonics are summed on the fly at each mesh point on
the GPU, so the same law holds on global, variable-resolution and
limited-area meshes without a stored basis. Philox4x32-10 follows the
Random123 BSD notice in licenses/LICENSE-Random123.txt.

No code or registry from NOAA's stochastic_physics package (LGPL) was read,
downloaded or used; namelist option names are read only as configuration.
All field arithmetic is CUDA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import threading
import time
import weakref

import numpy as np

RESOURCE_LOCK = threading.Lock()
MESH_RESOURCES = weakref.WeakValueDictionary()
SCHEMA = 'gpuwm-hex.stochastic/v3'


def backscatter_rate_scale(dt, time_s, interval_s=0.0):
    """Rate variance per unit energy for a correlated, held AR(1) pattern.

    With update period h and phi=exp(-h/tau), the integrated autocovariance
    is h/2 * (1+phi)/(1-phi). Thus sigma**2 = 2*D/h*tanh(h/(2*tau)).
    Using the white-noise 2*D/dt with a six-hour pattern at dt=600 s
    overstates its long-time energy input by about 72. Spatial clipping and
    normalization make this a nominal budget, not an exact net KE budget.
    """
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(time_s) or time_s <= 0:
        raise ValueError('backscatter dt and time_s must be positive and finite')
    if not math.isfinite(interval_s) or interval_s < 0:
        raise ValueError('backscatter interval_s must be nonnegative and finite')
    period = max(dt, interval_s)
    return 2 * math.tanh(period / (2*time_s)) / period


EARTH_RADIUS_M = 6371220.0
#: Omitted variance fraction of the automatic truncation.
TRUNCATION_TAIL = 1.0e-4


def spectral_exponent(length_m, radius=EARTH_RADIUS_M):
    """a in the variance spectrum exp(-l(l+1)*a) of the correlation exp(-r**2/(2 L**2)).

    The heat kernel exp(t*Laplacian) on the sphere has spectrum exp(-l(l+1)t/R**2)
    and correlation exp(-r**2/(4t)) for small t/R**2, so 4t = 2 L**2 gives
    a = L**2/(2 R**2) (Weaver & Courtier 2001, section 3).
    """
    ratio = float(length_m)/float(radius)
    return ratio*ratio/2  # a product overflows to inf, a power would raise


def auto_degree(length_m, radius=EARTH_RADIUS_M, tail=TRUNCATION_TAIL):
    """Smallest degree N whose omitted variance exp(-N(N+1)a) is below ``tail`` (8..512)."""
    a = spectral_exponent(length_m, radius)
    n = math.ceil((-1 + math.sqrt(1 + 4*math.log(1/tail)/a)) / 2)
    return int(min(512, max(8, n)))


def pattern_degree(config, length_m, radius=EARTH_RADIUS_M):
    return config.degree if config.degree else auto_degree(length_m, radius)


def config_max_degree(config, radius=EARTH_RADIUS_M):
    lengths = [config.length_m] + [scale[1] for scale in config.sppt_scales]
    return max(pattern_degree(config, length, radius) for length in lengths)


@dataclass(frozen=True)
class StochasticConfig:
    sppt: bool = False
    skeb: bool = False
    member: int = 0
    seed: int = 53
    # 0 chooses the truncation from the shortest length (auto_degree).
    degree: int = 0
    length_m: float = 500000.0
    time_s: float = 21600.0
    amplitude: float = 0.5
    clip: float = 1.0
    skeb_fraction: float = 0.1
    thermal_fraction: float = 0.01
    surface_m: float = 500.0
    top_start_m: float = 18000.0
    top_end_m: float = 25000.0
    # Each optional scale is (amplitude, length metres, time seconds, seed).
    sppt_scales: tuple = ()
    logit: bool = False
    surface_limit: bool = True
    interval_s: float = 0.0
    restart_required: bool = False
    surface_profile: tuple = ()
    # Palmer et al. (2009): no perturbed humidity tendency may supersaturate.
    moist_guard: bool = True

    def __post_init__(self):
        for name in ('sppt', 'skeb', 'logit', 'surface_limit', 'restart_required', 'moist_guard'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be logical')
        for name in ('member', 'seed'):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError(f'{name} must be an unsigned 64-bit integer')
        if type(self.degree) is not int or not 0 <= self.degree <= 512:
            raise ValueError('degree must be 0 (automatic) or 1..512 for the normalized harmonic recurrence')
        for name in ('length_m', 'time_s', 'amplitude', 'surface_m', 'top_end_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('skeb_fraction', 'thermal_fraction', 'top_start_m'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if not math.isfinite(self.clip) or not 0 < self.clip <= 1:
            raise ValueError('clip must be in (0,1] so SPPT cannot reverse a tendency')
        if self.top_end_m <= self.top_start_m:
            raise ValueError('top_end_m must exceed top_start_m')
        if max(self.skeb_fraction, self.thermal_fraction) > 1:
            raise ValueError('backscatter fractions must not exceed the dissipated energy')
        if not math.isfinite(self.interval_s) or self.interval_s < 0:
            raise ValueError('interval_s must be nonnegative and finite')
        scales = tuple(tuple(scale) for scale in self.sppt_scales)
        if len(scales) > 5:
            raise ValueError('SPPT supports at most five independent spectral scales')
        for scale in scales:
            if len(scale) != 4:
                raise ValueError('SPPT scale needs amplitude, length_m, time_s, seed')
            amp, length, tau, seed = scale
            if not all(math.isfinite(v) and v > 0 for v in (amp, length, tau)):
                raise ValueError('SPPT scale amplitude, length and time must be positive and finite')
            if type(seed) is not int or not 0 <= seed < 2**64:
                raise ValueError('SPPT scale seed must be an unsigned 64-bit integer')
        object.__setattr__(self, 'sppt_scales', scales)
        profile = tuple(self.surface_profile)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in profile):
            raise ValueError('surface_profile weights must be finite in [0,1]')
        object.__setattr__(self, 'surface_profile', profile)

    @property
    def enabled(self):
        return self.sppt or self.skeb

    @property
    def member_seed(self):
        payload = f'{SCHEMA}:{self.member}:{self.seed}'.encode('ascii')
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        options = json.loads(value) if isinstance(value, str) else value
        if not isinstance(options, Mapping):
            raise ValueError('stochastic configuration must be a JSON object')
        try:
            return cls(**options)
        except TypeError as error:
            raise ValueError(f'invalid stochastic configuration: {error}') from error

    def json(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

    def state_json(self):
        controls = asdict(self)
        controls.pop('restart_required')
        return json.dumps(controls, sort_keys=True, separators=(',', ':'))


def member_options(member, *, seed=53, source_members=1, ensemble_size=1, **options):
    """Default ON for every ensemble of more than one member.

    Measured basis (feat/stochastic-physics, 2026-10-29): GEFS 10 members,
    2026-10-27 00Z, hex 120 km MPAS physics, +6 h against the GFS analysis.
    SPPT+SKEB raised the area-weighted spread of z500 from 3.21 to 3.50 m and
    of t2m from 0.44 to 0.51 K. The ensemble-mean RMSE moved from 4.065 to
    4.078 m (z500) and from 1.698 to 1.692 K (t2m). So the spread/skill ratio
    rose from 0.79 to 0.86 and from 0.26 to 0.30, at under 1 % of the step.
    Before that run the default was on only below ten source members.
    ``source_members`` is still validated and recorded by the callers.
    Explicit per-member booleans override the default in either direction.
    """
    if type(source_members) is not int or source_members < 1:
        raise ValueError('source_members must be positive')
    if type(ensemble_size) is not int or ensemble_size < 1:
        raise ValueError('ensemble_size must be positive')
    on = ensemble_size > 1
    return asdict(StochasticConfig(**{**dict(sppt=on, skeb=on, member=member, seed=seed), **options}))


def member_number(identity):
    """Stable identity for named rosters, independent of selection and grouping."""
    if type(identity) is int:
        return identity
    return int.from_bytes(hashlib.sha256(str(identity).encode('utf-8')).digest()[:8], 'little')


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    source = Path(__file__).with_name('stochastic.cu').read_text()
    return cp.RawModule(code=source, options=('--std=c++17',))


def _kernel(name):
    import cupy as cp
    return _module(cp.cuda.Device().id).get_function(name)


def _field_strides(field, levels, columns, label):
    """Validate raw-kernel fields using metadata only; return strides in words."""
    if (field.dtype != np.float32 or field.ndim != 2
            or field.shape[0] != levels or field.shape[1] < columns
            or any(stride % 4 for stride in field.strides)):
        raise ValueError(f'stochastic {label} requires float32[level,{columns} or more]')
    return tuple(np.int64(stride//4) for stride in field.strides)


def _valid_indices(cp, values, count):
    """int32 topology with every missing or padded neighbour as -1.

    A limited-area mesh marks the neighbours outside its region either with a
    negative index or with the padded garbage entity ``count``; both become -1.
    """
    indices = cp.asarray(values, dtype=cp.int32)
    return cp.ascontiguousarray(cp.where((indices < 0) | (indices >= count), -1, indices).astype(cp.int32))


class PointHarmonics:
    """Harmonic synthesis at a set of points on the sphere (cells, then any vertices).

    Holds the point coordinates and the three-term recurrence tables up to
    ``degree``; no (points x modes) basis is stored. Also the resource for
    pattern statistics at arbitrary points (verification tools and tests).
    """
    def __init__(self, lat, lon, degree, radius=EARTH_RADIUS_M):
        import cupy as cp
        if type(degree) is not int or not 1 <= degree <= 512:
            raise ValueError('harmonic degree must be 1..512')
        self.degree = degree
        self.radius = float(radius)
        self.lat = cp.ascontiguousarray(cp.asarray(lat, dtype=cp.float64))
        self.lon = cp.ascontiguousarray(cp.asarray(lon, dtype=cp.float64))
        self.cells = int(self.lat.size)
        self.vertices = 0
        # Recurrence tables at each (l, m>=0) cosine slot l*l+l-1+m.
        self.modes = (degree + 1)**2 - 1
        index = cp.arange(1, self.modes+1, dtype=cp.float64)
        self.ell = cp.floor(cp.sqrt(index))
        ell = self.ell
        order = cp.arange(self.modes, dtype=cp.float64) - (ell*ell + ell - 1)
        lsq, msq = ell*ell, order*order
        valid = (order >= 0) & (order < ell)
        safe = cp.where(valid, lsq - msq, 1)
        ra = cp.where(valid, cp.sqrt((4*lsq - 1)/safe), 0)
        rb = cp.where(valid, cp.sqrt(cp.maximum((2*ell + 1)*((ell - 1)**2 - msq), 0)
                                     / cp.where(valid, (2*ell - 3)*(lsq - msq), 1)), 0)
        self.ra = cp.ascontiguousarray(ra.astype(cp.float32))
        self.rb = cp.ascontiguousarray(rb.astype(cp.float32))
        self._basis = None

    @property
    def basis(self):
        """Stored (points x modes) basis at full degree, for verification only."""
        import cupy as cp
        if self._basis is None:
            points = int(self.lat.size)
            self._basis = cp.empty((points, self.modes), dtype=cp.float32)
            count = points * (self.degree + 1)
            _kernel('harmonics')(((count+127)//128,), (128,),
                (self.lat, self.lon, self._basis, np.int32(points), np.int32(self.degree)))
        return self._basis

    def synthesize(self, coefficients, out, degree, points, channels):
        """out[:points, :channels] = sum over modes of Y(point) * coefficients."""
        if degree > self.degree or points > self.lat.size:
            raise ValueError('pattern degree or points exceed this harmonic resource')
        _kernel('harmonic_sum')(((points+127)//128,), (128,),
            (self.lat, self.lon, coefficients, self.ra, self.rb, out,
             np.int32(points), np.int32(degree), np.int32(channels)))
        return out


class MeshHarmonics(PointHarmonics):
    """Immutable mesh resource, built on GPU and shared across member streams.

    Patterns are synthesised on the fly by the ``harmonic_sum`` kernel at the
    mesh cells and vertices. A limited-area mesh is accepted: missing
    neighbours are -1, edges touching them carry no SKEB curl, and cells of
    the lateral-boundary zone (``bdyMaskCell`` > 0) are masked.
    """
    def __init__(self, mesh, degree):
        import cupy as cp
        started = time.perf_counter()
        cells_count = mesh.n_cells
        a = mesh.arrays
        self.build_started = cp.cuda.Event()
        self.build_started.record()
        lat_vertex = cp.asarray(a['latVertex'], dtype=cp.float64)
        super().__init__(
            cp.concatenate((cp.asarray(a['latCell'], dtype=cp.float64)[:cells_count], lat_vertex)),
            cp.concatenate((cp.asarray(a['lonCell'], dtype=cp.float64)[:cells_count],
                            cp.asarray(a['lonVertex'], dtype=cp.float64))),
            degree, float(mesh.attrs.get('sphere_radius', EARTH_RADIUS_M)))
        self.cells = cells_count
        self.vertices = int(lat_vertex.size)
        self.edges = mesh.n_edges
        self.coe = _valid_indices(cp, a['cellsOnEdge'], self.cells)[:self.edges]
        self.voe = _valid_indices(cp, a['verticesOnEdge'], self.vertices)[:self.edges]
        self.cov = _valid_indices(cp, a['cellsOnVertex'], self.cells)
        self.limited = bool(cp.any(self.coe < 0)) or bool(cp.any(self.cov < 0))
        area = cp.asarray(a['areaCell'], dtype=cp.float32)[:self.cells]
        self.area = area / cp.sum(area)
        self.dv = cp.ascontiguousarray(cp.asarray(a['dvEdge'], dtype=cp.float32)[:self.edges])
        # Orient the dual tangent against r cross the primal edge normal.
        def xyz(kind, count):
            return cp.stack([cp.asarray(a[c+kind], dtype=cp.float64)[:count] for c in ('x','y','z')], axis=1)
        cells, vertices = xyz('Cell', self.cells), xyz('Vertex', self.vertices)
        coe, voe = cp.maximum(self.coe, 0), cp.maximum(self.voe, 0)
        normal = cells[coe[:,1]] - cells[coe[:,0]]
        radial = cells[coe[:,1]] + cells[coe[:,0]]
        tangent = vertices[voe[:,1]] - vertices[voe[:,0]]
        complete = cp.all(self.coe >= 0, axis=1) & cp.all(self.voe >= 0, axis=1)
        # Edges between two existing cells: the only ones whose edge density
        # and kinetic-energy sink are defined on a limited-area mesh.
        self.edge_interior = cp.ascontiguousarray(cp.all(self.coe >= 0, axis=1).astype(cp.float32))
        self.orientation = cp.ascontiguousarray(cp.where(complete,
            cp.sign(cp.sum(cp.cross(radial, normal)*tangent, axis=1)), 0).astype(cp.float32))
        self.edge_weight = cp.asarray(a['dcEdge'], dtype=cp.float32)[:self.edges] * self.dv
        self.edge_weight /= cp.sum(self.edge_weight)
        # Lateral-boundary zone of a limited-area mesh: the boundary series
        # drives those cells, so no stochastic forcing is applied there.
        mask = a['bdyMaskCell'] if 'bdyMaskCell' in a else None
        self.cell_mask = (None if mask is None else
            cp.ascontiguousarray((cp.asarray(mask)[:self.cells] == 0).astype(cp.float32)))
        if self.cell_mask is not None and bool(cp.all(self.cell_mask == 1)):
            self.cell_mask = None
        self.ready = cp.cuda.Event()
        self.ready.record()
        self.receipt = dict(parallelism='GPU', block_threads=128,
                            synthesis='on the fly at each point (harmonic_sum kernel), no stored basis',
                            basis_bytes=0, recurrence_table_bytes=int(self.ra.nbytes + self.rb.nbytes),
                            degree=degree, limited_area=self.limited,
                            boundary_zone_cells_masked=(0 if self.cell_mask is None else
                                                        int(self.cells - int(cp.sum(self.cell_mask)))),
                            construction_enqueue_seconds=time.perf_counter()-started,
                            timing_note='enqueue wall, not GPU elapsed time')

    @classmethod
    def cached(cls, mesh, degree):
        import cupy as cp
        key = (cp.cuda.Device().id, degree)
        with RESOURCE_LOCK:
            resources = getattr(mesh, '_stochastic_harmonics', None)
            if resources is None:
                resources = mesh._stochastic_harmonics = {}
            if key not in resources:
                # Resident member preparation interns identical immutable host
                # arrays. Share the resource without hashing the mesh on a CPU.
                owners = tuple(mesh.arrays[name] for name in ('latCell', 'lonCell',
                    'latVertex', 'lonVertex', 'areaCell', 'cellsOnEdge', 'verticesOnEdge',
                    'cellsOnVertex', 'dvEdge', 'dcEdge', 'xCell', 'yCell', 'zCell',
                    'xVertex', 'yVertex', 'zVertex'))
                shared_key = (*key, float(mesh.attrs.get('sphere_radius', EARTH_RADIUS_M)),
                              int(mesh.n_cells), int(mesh.n_edges),
                              *(id(array) for array in owners))
                resource = MESH_RESOURCES.get(shared_key)
                if resource is None:
                    resource = cls(mesh, degree)
                    resource.host_owners = owners
                    MESH_RESOURCES[shared_key] = resource
                resources[key] = resource
            resource = resources[key]
        cp.cuda.get_current_stream().wait_event(resource.ready)
        return resource


class SphericalPattern:
    """Stationary AR(1) coefficients, independent Philox stream per process.

    A snapshot contains only coefficients and counters, never a basis.
    Repeated calls at the same clock return the same pattern without drawing.
    ``points`` are the first mesh points synthesised (cells, then vertices);
    ``channels`` how many of the three coefficient channels are summed.
    """
    def __init__(self, resource, config, *, degree=None, points=None, channels=3):
        import cupy as cp
        self.resource, self.config = resource, config
        self.degree = int(degree if degree is not None else
                          pattern_degree(config, config.length_m, resource.radius))
        self.modes = (self.degree + 1)**2 - 1
        self.points = int(resource.lat.size if points is None else points)
        self.channels = int(channels)
        self.coefficients = cp.zeros((self.modes, 3), dtype=cp.float32)
        ell = resource.ell[:self.modes]
        scale = config.length_m/resource.radius/math.sqrt(2)
        # Variance spectrum exp(-l(l+1)a), a = scale**2: correlation
        # exp(-r**2/(2 L**2)). Cancel the common l=1 exponent before
        # normalizing, so a very long correlation length keeps a finite dipole
        # limit instead of 0/0; multiply by scale twice on the GPU so the
        # dipole exponent stays zero even when scale**2 would overflow.
        variance = cp.exp(-((ell*(ell+1)-2)*scale)*scale)
        # Addition theorem: sum_m Y_lm(x)^2 = (2l+1)/(4 pi), unit point variance.
        self.weights = cp.sqrt(variance*(4*math.pi)/cp.sum(variance)).astype(cp.float32)
        # RMS of one component of the rotational wind k x grad(psi) of this
        # pattern: E|grad psi|**2 = sum_modes w**2 l(l+1)/(4 pi R**2), half per
        # component (isotropy). SKEB normalises by this expected value, not by a
        # sample over the mesh, so a limited area much smaller than the length
        # scale is not amplified into O(1) curls (lam7k, 2026-10-29).
        self.curl_rms = float(cp.sqrt(cp.sum(self.weights.astype(cp.float64)**2 * ell*(ell+1))
                                      / (8*math.pi))) / resource.radius
        self.last_time = None
        self.draw = 0
        self.pattern = cp.zeros((self.points, 3), dtype=cp.float32)

    def _synthesize(self):
        if self.channels > 0:
            self.resource.synthesize(self.coefficients, self.pattern, self.degree,
                                     self.points, self.channels)

    def advance(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('pattern clock must be finite and nonnegative')
        if self.last_time == seconds:
            return self.pattern
        if self.last_time is not None and seconds < self.last_time:
            raise ValueError('pattern clock moved backwards; restore its checkpoint')
        if self.last_time is None:
            phi, innovation = 0.0, 1.0
        else:
            elapsed = (seconds-self.last_time)/self.config.time_s
            phi = math.exp(-elapsed)
            # Use elapsed directly: exp(-elapsed) can round to one while
            # the innovation is still representable by a float32 word.
            innovation = math.sqrt(-math.expm1(-2*elapsed))
        n = self.modes*3
        _kernel('ar1')(((n+127)//128,), (128,),
            (self.coefficients, self.weights, np.int32(n), np.float32(phi),
             np.float32(innovation),
             np.uint64(self.config.member_seed), np.uint64(self.draw)))
        self._synthesize()
        self.last_time, self.draw = seconds, self.draw+1
        return self.pattern

    def snapshot(self, *, device=False):
        import cupy as cp
        return dict(schema=SCHEMA, config=self.config.state_json(), time=self.last_time,
                    draw=self.draw, coefficients=(self.coefficients.copy() if device
                                                 else cp.asnumpy(self.coefficients)))

    def restore(self, value):
        import cupy as cp
        if value['schema'] != SCHEMA or value['config'] != self.config.state_json():
            raise ValueError('stochastic checkpoint configuration differs')
        clock, draw = value['time'], value['draw']
        if (type(draw) is not int or not 0 <= draw < 2**64
                or (clock is None and draw != 0)
                or (clock is not None and (not math.isfinite(clock) or clock < 0 or draw == 0))):
            raise ValueError('invalid stochastic checkpoint clock or draw counter')
        a = value['coefficients']
        if a.shape != self.coefficients.shape or a.dtype != np.float32:
            raise ValueError('stochastic checkpoint coefficient layout or values differ')
        a = cp.asarray(a)
        if not bool(cp.isfinite(a).all()):
            raise ValueError('stochastic checkpoint coefficient layout or values differ')
        self.coefficients[...] = a
        self.last_time, self.draw = clock, draw
        self._synthesize()


class ForcingPatterns:
    """Independent SPPT scales and two backscatter streams on one mesh resource.

    The base process carries SKEB's streamfunction (vertices) and thermal
    (cells) channels, and SPPT when no separate scales are named. Only the
    channels and points a run uses are synthesised.
    """
    def __init__(self, resource, config):
        self.config = config
        cells = resource.cells
        if config.skeb:
            points, channels = resource.cells + resource.vertices, 3
        elif config.sppt and not config.sppt_scales:
            points, channels = cells, 1
        else:
            points, channels = cells, 0
        self.base = SphericalPattern(resource, config, points=points, channels=channels)
        self.scales = [(amp, SphericalPattern(resource, replace(config,
            length_m=length, time_s=tau, seed=seed, sppt_scales=()),
            degree=pattern_degree(config, length, resource.radius), points=cells, channels=1))
            for amp, length, tau, seed in config.sppt_scales if config.sppt]

    def advance(self, seconds):
        import cupy as cp
        interval = self.config.interval_s
        if interval:
            # Decimal boundary clocks such as .3/.1 can divide one ULP
            # below the integral tick. Admit that rounding error before
            # flooring, so the update is not held for another whole interval.
            quotient = seconds/interval
            tick = math.floor(quotient)
            if quotient != tick and math.nextafter(quotient, math.inf) == tick+1:
                tick += 1
            seconds = tick*interval
        base = self.base.advance(seconds)
        if self.scales:
            sppt = cp.zeros(self.base.resource.cells, dtype=cp.float32)
            for amplitude, process in self.scales:
                sppt += amplitude*process.advance(seconds)[:, 0]
        else:
            sppt = self.config.amplitude*base[:self.base.resource.cells, 0]
        return sppt, base[:, 1], base[:, 2]

    def snapshot(self, *, device=False):
        return dict(schema=SCHEMA, config=self.config.state_json(), base=self.base.snapshot(device=device),
                    scales=[process.snapshot(device=device) for _, process in self.scales])

    def restore(self, value):
        if value['schema'] != SCHEMA or value['config'] != self.config.state_json():
            raise ValueError('stochastic checkpoint configuration differs')
        if len(value['scales']) != len(self.scales):
            raise ValueError('stochastic checkpoint scale count differs')
        self.base.restore(value['base'])
        for (_, process), saved in zip(self.scales, value['scales']):
            process.restore(saved)


def _skeb_patterns(cp, physics, streamfunction, temperature, nc):
    """SKEB's vertex streamfunction, its curl normaliser and the thermal pattern.

    Both are unit-variance pattern channels used as drawn (Berner et al. 2009:
    an unclipped streamfunction pattern). The curl is normalised by the
    pattern's expected component RMS (``SphericalPattern.curl_rms``). The
    thermal pattern keeps unit variance; on a closed mesh its area mean is
    removed so it adds no net heat, while a limited area keeps its slice of
    the global pattern.
    """
    psi = cp.ascontiguousarray(streamfunction[nc:])
    thermal = temperature[:nc].copy()
    r = physics.resource
    if not getattr(r, 'limited', False):
        thermal -= cp.sum(thermal*r.area)
    return psi, physics.skeb_norm, thermal


def _kinetic_sink(cp, wind, momentum, edge_density, interior):
    """Positive kinetic-energy sink of the physics momentum source, per level and edge.

    Only edges between two existing cells with a positive edge density and
    finite words contribute. A limited-area boundary edge borrows the padded
    garbage cell for its density and tendency, which native leaves undefined.
    THE BREAKAGE THIS PREVENTS: one such word made the per-level dissipation
    NaN, and SKEB then turned every wind NaN (lam7k hrrrv5 --sppt --skeb,
    2026-10-29). A product with a zero mask keeps a NaN, so the mask selects.
    """
    keep = edge_density > 0
    raw = -wind*momentum/cp.where(keep, edge_density, 1)
    keep &= cp.isfinite(raw)
    if interior is not None:
        keep &= interior[None, :] > 0
    return cp.where(keep, cp.maximum(raw, 0), 0)


def _guard_fields(owner, state, names, nc, levels, has_qv, theta, *, zz=None, own_metric=True):
    """Moisture-guard state views and an optional revert counter (first calls only).

    ``zz`` is the metric of the cells being forced; an LTS class passes its
    own gathered columns (``own_metric=False``), never the member's whole mesh.
    """
    import cupy as cp
    if zz is None and own_metric:
        zz = getattr(owner, 'zz', None)
    guard = bool(owner.config.sppt and owner.config.moist_guard and has_qv and zz is not None)
    if not guard:
        zero = (np.int64(0), np.int64(0))
        return False, theta, theta, zero, zero, np.uint64(0)
    rtheta_state = state.rho_theta[:, :nc]
    qv_state = state.scalars[names.index('qv'), :, :nc]
    gs = _field_strides(rtheta_state, levels, nc, 'theta state')
    vs = _field_strides(qv_state, levels, nc, 'qv state')
    counter = None
    counts = owner.__dict__.setdefault('guard_counts', [])
    if len(counts) < 4:
        counter = cp.zeros(1, dtype=cp.uint32)
        counts.append((counter, levels*nc))
    return True, rtheta_state, qv_state, gs, vs, (np.uint64(0) if counter is None else counter)


class StochasticPhysics:
    """Apply SPPT and dissipation-budget SKEB to held dry-mass sources.

    SKEB uses the resolved kinetic-energy sink of the unperturbed physics
    momentum tendency. Its globally normalized dual-edge curl is rotational.
    This is a mesh adaptation, not WRF's prescribed constant-dissipation mode.
    """
    def __init__(self, mesh, vertical, config, *, partition=None):
        import cupy as cp
        self.config = StochasticConfig.parse(config)
        radius = float(mesh.attrs.get('sphere_radius', EARTH_RADIUS_M))
        self.resource = MeshHarmonics.cached(mesh, config_max_degree(self.config, radius))
        self.process = ForcingPatterns(self.resource, self.config)
        self.skeb_norm = cp.asarray([self.process.base.curl_rms], dtype=cp.float32)
        # ``partition`` (a same-host MPAS-physics rank): ``mesh`` is the rank's
        # local mesh, owned cells and edges first; ``allreduce`` sums a float64
        # vector over the ranks. The spectral process is the same on every rank
        # (same member seed and draws), so the pattern is one global pattern;
        # the normalisations below are global owned-entity sums. Halo words of
        # the forced sources are replaced by their owners' after forcing.
        self.partition = partition
        if partition is not None:
            r, a = self.resource, mesh.arrays
            oc, oe = int(partition.owned_cells), int(partition.owned_edges)
            area = cp.asarray(a['areaCell'], dtype=cp.float32)[:oc]
            weight = (cp.asarray(a['dcEdge'], dtype=cp.float32) * cp.asarray(a['dvEdge'], dtype=cp.float32))[:oe]
            totals = partition.allreduce(np.array([float(cp.sum(area, dtype=cp.float64)),
                                                   float(cp.sum(weight, dtype=cp.float64))]))
            self.owned_area = cp.ascontiguousarray(area / cp.float32(totals[0]))
            self.owned_edge_weight = cp.ascontiguousarray(weight / cp.float32(totals[1]))
        z = cp.asarray(vertical.zgrid, dtype=cp.float32)
        n = mesh.n_cells
        if z.ndim != 2 or z.shape[0] != vertical.n_vert_levels+1 or z.shape[1] < n:
            raise ValueError('stochastic taper requires zgrid[interface,cell]')
        z = z[:, :n]
        height = (z[1:]+z[:-1])*0.5 - z[0]
        c = self.config
        # Palmer et al. (2009): SPPT is tapered to zero near the surface and
        # in the stratosphere. Here a linear ramp from the ground to surface_m
        # (the lowest level is always zero) and from top_start_m to top_end_m.
        lower = cp.clip(height/c.surface_m,0,1) if c.surface_limit else 1
        self.taper = cp.ascontiguousarray(cp.minimum(lower,
            cp.clip((c.top_end_m-height)/(c.top_end_m-c.top_start_m),0,1)).astype(cp.float32))
        if c.surface_limit:
            self.taper[0,:] = 0
            if c.surface_profile:
                count = min(len(c.surface_profile), int(height.shape[0]))
                self.taper[:count,:] = cp.minimum(self.taper[:count,:],
                    cp.asarray(c.surface_profile[:count],dtype=cp.float32)[:,None])
        if self.resource.cell_mask is not None:
            self.taper *= self.resource.cell_mask[None, :]
        # Vertex taper: mean over the vertex's cells that exist (a limited-area
        # boundary vertex has fewer than three).
        cov = self.resource.cov
        present = (cov >= 0).astype(cp.float32)
        gathered = self.taper[:, cp.maximum(cov, 0)] * present[None, :, :]
        self.vertex_taper = cp.ascontiguousarray(cp.sum(gathered, axis=2)
            / cp.maximum(cp.sum(present, axis=1), 1)[None, :])
        del gathered
        # Moisture guard: MPAS's zz metric (rho_d = zz*rho_zz) for the pressure.
        zz = getattr(vertical, 'zz', None)
        self.zz = (None if zz is None else
                   cp.ascontiguousarray(cp.asarray(zz, dtype=cp.float32)[:, :n]))
        self.guard_counts = []
        self.calls = 0
        self.timing_events = []
        from .stochastic_timing import StepTimings
        self.step_timing = StepTimings(cp)

    def apply(self, held, *, state, rho_edge, normal_velocity, time_seconds, dt, names):
        import cupy as cp
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('stochastic dt must be positive')
        r, c = self.resource, self.config
        sample = None
        step_timing = getattr(self, 'step_timing', None)
        if len(self.timing_events) < 4 or (step_timing is not None and step_timing.active is not None):
            sample = tuple(cp.cuda.Event() for _ in range(3))
            sample[0].record()
        sppt, streamfunction, temperature = self.process.advance(float(time_seconds))
        if sample is not None:
            sample[1].record()
        nc, ne = r.cells, r.edges
        energy_scale = backscatter_rate_scale(dt, c.time_s, c.interval_s)
        # Regional physics has a padded garbage column; only real cells/edges participate.
        theta = held.rho_theta[:, :nc]
        momentum = held.rho_u[:, :ne]
        density = state.rho[:, :nc]
        edge_density = rho_edge[:, :ne]
        wind = normal_velocity[:, :ne]
        levels = self.taper.shape[0]
        has_qv = c.sppt and 'qv' in names
        qv = held.scalars[names.index('qv'),:,:nc] if has_qv else theta
        ts = _field_strides(theta, levels, nc, 'theta tendency')
        qs = _field_strides(qv, levels, nc, 'qv tendency')
        rs = _field_strides(density, levels, nc, 'cell density')
        ms = _field_strides(momentum, levels, ne, 'momentum tendency')
        es = _field_strides(edge_density, levels, ne, 'edge density')
        # Unused pointers keep one kernel ABI for SPPT-only and SKEB-only calls.
        psi = thermal = dissipation = norm = sppt
        if c.skeb and getattr(self, 'partition', None) is not None:
            dissipation, psi, norm, thermal = self._partition_budget(
                streamfunction, temperature, momentum, edge_density, wind, nc, levels)
        elif c.skeb:
            sink = _kinetic_sink(cp, wind, momentum, edge_density, getattr(r, 'edge_interior', None))
            dissipation = cp.sum(sink*r.edge_weight[None,:], axis=1, keepdims=True)
            psi, norm, thermal = _skeb_patterns(cp, self, streamfunction, temperature, nc)
        guard, rtheta_state, qv_state, gs, vs, counter = self._guard_fields(
            state, names, nc, levels, has_qv, theta)
        # Fuse taper, clipping, SPPT and SKEB writes. No level-by-edge curl,
        # backscatter, gathered SPPT factors or level-by-cell heat allocation.
        _kernel('force_cells')(((levels*nc+127)//128,), (128,),
            (theta, qv, density, self.taper, sppt, thermal, dissipation,
             *ts, *qs, *rs, np.int32(nc), np.int32(levels),
             np.int32(c.sppt), np.int32(c.skeb), np.int32(has_qv),
             np.int32(c.logit), np.float32(c.clip),
             np.float32(energy_scale*c.thermal_fraction*300/1004),
             np.int32(guard), rtheta_state, qv_state, self.zz if guard else theta,
             *gs, *vs, np.float32(dt), counter))
        _kernel('force_edges')(((levels*ne+127)//128,), (128,),
            (momentum, edge_density, self.taper, self.vertex_taper, sppt, psi,
             dissipation, norm, r.coe, r.voe, r.orientation, r.dv,
             *ms, *es, np.int32(nc), np.int32(self.vertex_taper.shape[1]),
             np.int32(ne), np.int32(levels), np.int32(c.sppt), np.int32(c.skeb),
             np.int32(c.logit), np.float32(c.clip), np.float32(energy_scale*c.skeb_fraction)))
        if sample is not None:
            sample[2].record()
            self.timing_events.append(sample)
            if step_timing is not None:
                step_timing.forcing(sample, cold=self.calls == 0)
        self.calls += 1
        return held

    def _guard_fields(self, state, names, nc, levels, has_qv, theta, *, zz=None, own_metric=True):
        return _guard_fields(self, state, names, nc, levels, has_qv, theta, zz=zz, own_metric=own_metric)

    def guard_receipt(self):
        import cupy as cp
        samples = [dict(reverted=int(cp.asnumpy(count)[0]), points=points)
                   for count, points in getattr(self, 'guard_counts', [])]
        return dict(rule='Palmer et al. (2009): perturbed tendencies revert where the perturbed '
                         'humidity would be supersaturated (Bolton 1980 over water) or negative',
                    enabled=bool(self.config.sppt and self.config.moist_guard
                                 and getattr(self, 'zz', None) is not None),
                    first_calls=samples)

    def _partition_budget(self, streamfunction, temperature, momentum, edge_density, wind, nc, levels):
        """SKEB's global sums on a rank: owned-entity partials, one reduction per step.

        The one-card sums over every cell and edge of the closed mesh; a rank
        holds halo copies, so only owned cells and edges (first in local order)
        contribute. Thermal spread is E[t^2] - E[t]^2 over the unit total area.
        """
        import cupy as cp
        r, c, part = self.resource, self.config, self.partition
        oc, oe = int(part.owned_cells), int(part.owned_edges)
        w, area = self.owned_edge_weight, self.owned_area
        interior = getattr(r, 'edge_interior', None)
        sink = _kinetic_sink(cp, wind[:, :oe], momentum[:, :oe], edge_density[:, :oe],
                             None if interior is None else interior[:oe])
        psi = cp.ascontiguousarray(streamfunction[nc:])
        thermal = temperature[:nc].copy()
        owned_t = thermal[:oc].astype(cp.float64)
        partial = cp.concatenate((
            cp.sum(sink.astype(cp.float64)*w[None, :], axis=1),
            cp.sum(owned_t*area)[None]))
        total = part.allreduce(cp.asnumpy(partial))
        dissipation = cp.asarray(total[:levels].astype(np.float32).reshape(levels, 1))
        if not getattr(r, 'limited', False):
            thermal -= cp.float32(total[levels])
        return dissipation, psi, self.skeb_norm, thermal

    def receipt(self):
        import cupy as cp
        from .device_admission import stochastic_array_ledger
        basis = dict(self.resource.receipt)
        if self.resource.ready.done:
            basis['gpu_elapsed_ms'] = cp.cuda.get_elapsed_time(
                self.resource.build_started, self.resource.ready)
        timing = [dict(sample=i, pattern_ms=cp.cuda.get_elapsed_time(start, pattern),
                       forcing_total_ms=cp.cuda.get_elapsed_time(start, end))
                  for i, (start, pattern, end) in enumerate(self.timing_events) if end.done]
        processes = [self.process.base, *(process for _, process in self.process.scales)]
        buffers = dict(taper=int(self.taper.nbytes), vertex_taper=int(self.vertex_taper.nbytes),
                       moisture_metric=(0 if getattr(self, 'zz', None) is None
                                        else int(self.zz.nbytes)),
                       coefficients=sum(int(p.coefficients.nbytes) for p in processes),
                       spectral_weights=sum(int(p.weights.nbytes) for p in processes),
                       pattern_outputs=sum(int(p.pattern.nbytes) for p in processes))
        return dict(schema=SCHEMA, config=asdict(self.config), member_seed=self.config.member_seed,
                    member_seed_hex=f'{self.config.member_seed:016x}',
                    basis=basis, calls=self.calls, gpu_timing_samples=timing,
                    step_cost=self.step_timing.receipt(),
                    admission_arrays=stochastic_array_ledger(
                        self.resource.cells, int(self.taper.shape[0]),
                        getattr(self.resource, 'degree', None) or config_max_degree(self.config),
                        extra_scales=len(self.process.scales), skeb=self.config.skeb),
                    pattern_degrees=dict(base=getattr(self.process.base, 'degree', None),
                                         scales=[getattr(process, 'degree', None)
                                                 for _, process in self.process.scales]),
                    correlation='exp(-r**2/(2*length_m**2)) in space, exp(-dt/time_s) in time',
                    moist_guard=self.guard_receipt(),
                    admission_note='logical array floor; construction scratch, intermediate expressions, allocator placement and opaque library workspaces excluded; not measured peak VRAM',
                    member_buffer_bytes=buffers,
                    member_buffer_note='logical retained arrays only; excludes mesh resource, CuPy pool and transient reductions; not peak VRAM',
                    timing_note='CUDA stream elapsed including dispatch gaps; cold calls excluded from matched warm step-cost grading',
                    evaluation='GPU on-the-fly harmonic synthesis into reused output; GPU Philox and fused cell/edge forcing (128 threads per block)',
                    dissipation='positive kinetic-energy sink of unperturbed physics momentum sources',
                    backscatter_normalization='integrated AR(1) covariance: 2/h*tanh(h/(2*time_s)), h=max(dt,interval_s)',
                    scientific_status='spread-skill qualification required',
                    **({} if getattr(self, 'partition', None) is None else dict(partition=dict(
                        ranks=int(self.partition.ranks), owned_cells=int(self.partition.owned_cells),
                        owned_edges=int(self.partition.owned_edges),
                        pattern='one spectral process on every rank (same member seed and draws), '
                                'evaluated at local cells and vertices',
                        budget='owned-entity float64 partial sums reduced across ranks once per step',
                        halos='forced halo words replaced by their owners at the source exchange'))))


class LtsStochasticPhysics:
    """One GPU spectral process and global budget for all classes of a member.

    Fine-first LTS visits class clocks out of order. Patterns are therefore
    evaluated once at a synchronized macro start and held over that macro.
    Unperturbed owned-edge sinks are integrated over each class's own dt,
    converted to area-weighted owned-cell dissipation at the macro endpoint,
    and used by every class in the next macro. The first macro has zero SKEB
    budget. This explicit one-macro lag avoids an order-dependent reduction.
    """
    schema = 'gpuwm-hex.lts-stochastic/v1'

    def __init__(self, mesh, vertical, config, layouts):
        import cupy as cp
        # Harmonic caches must not attach GPU basis arrays to the host mesh
        # serialized by moving checkpoints. Keep its read-only array owners,
        # but put runtime-only cache attributes on a private shallow view.
        import copy
        runtime_mesh = copy.copy(mesh)
        runtime_mesh.__dict__.pop('_stochastic_harmonics', None)
        self.global_physics = StochasticPhysics(runtime_mesh, vertical, config)
        self.config = self.global_physics.config
        self.resource = self.global_physics.resource
        self.process = self.global_physics.process
        self.levels = int(vertical.n_vert_levels)
        self.layouts = []
        for layout in layouts:
            cells = cp.asarray(layout.cell_l2g, dtype=cp.int32)
            edges = cp.asarray(layout.edge_l2g, dtype=cp.int32)
            self.layouts.append(dict(cells=cells, edges=edges,
                owned_edges=int(layout.n_owned_edges),
                zz=(None if getattr(self.global_physics, 'zz', None) is None else
                    cp.ascontiguousarray(self.global_physics.zz[:, cells]))))
        self.eoc = cp.ascontiguousarray(cp.asarray(mesh.arrays['edgesOnCell'], dtype=cp.int32))
        self.nec = cp.ascontiguousarray(cp.asarray(mesh.arrays['nEdgesOnCell'], dtype=cp.int32))
        self.sink_integral = cp.zeros((self.levels, mesh.n_edges), dtype=cp.float32)
        self.cell_sink = cp.empty((self.levels, mesh.n_cells), dtype=cp.float32)
        self.dissipation = cp.zeros((self.levels, 1), dtype=cp.float32)
        self.boundary_seconds = None
        self.macro = None
        self.calls = [0] * len(layouts)
        self.macro_calls = 0

    def begin_macro(self, seconds, duration):
        import cupy as cp
        if (self.macro is not None or not math.isfinite(seconds) or seconds < 0
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError('invalid or overlapping stochastic macro interval')
        if self.boundary_seconds is not None and seconds != self.boundary_seconds:
            raise ValueError('stochastic macro does not start at its carried boundary')
        self.macro = (float(seconds), float(duration))
        self.class_elapsed = [0.0] * len(self.layouts)
        r, c, g = self.resource, self.config, self.global_physics
        sppt, streamfunction, temperature = self.process.advance(float(seconds))
        self.sppt = cp.ascontiguousarray(sppt)
        self.psi, self.norm, thermal = _skeb_patterns(cp, g, streamfunction, temperature, r.cells)
        self.local_fields = [(
            cp.ascontiguousarray(g.taper[:, row['cells']]),
            cp.ascontiguousarray(self.sppt[row['cells']]),
            cp.ascontiguousarray(thermal[row['cells']])) for row in self.layouts]
        self.sink_integral.fill(0)

    def apply(self, index, held, *, state, rho_edge, normal_velocity, time_seconds, dt, names):
        import cupy as cp
        if self.macro is None:
            raise ValueError('stochastic class step is outside a synchronized macro')
        start, duration = self.macro
        if (not math.isfinite(dt) or dt <= 0 or time_seconds < start
                or time_seconds + dt > start + duration + 1e-8
                or abs(time_seconds - (start + self.class_elapsed[index])) > 1e-8):
            raise ValueError('stochastic class step straddles its macro interval')
        row, r, c = self.layouts[index], self.resource, self.config
        nc, ne = int(row['cells'].size), int(row['edges'].size)
        theta, momentum = held.rho_theta[:, :nc], held.rho_u[:, :ne]
        density, edge_density = state.rho[:, :nc], rho_edge[:, :ne]
        # Only edge owners contribute. Halo copies receive identical global
        # patterns but never duplicate the member's dissipation reduction.
        owned = row['owned_edges']
        if c.skeb:
            sink = cp.maximum(-normal_velocity[:, :owned] * momentum[:, :owned]
                              / edge_density[:, :owned], 0)
            self.sink_integral[:, row['edges'][:owned]] += sink * cp.float32(dt)
        taper, sppt, thermal = self.local_fields[index]
        has_qv = c.sppt and 'qv' in names
        qv = held.scalars[names.index('qv'), :, :nc] if has_qv else theta
        ts = _field_strides(theta, self.levels, nc, 'theta tendency')
        qs = _field_strides(qv, self.levels, nc, 'qv tendency')
        rs = _field_strides(density, self.levels, nc, 'cell density')
        ms = _field_strides(momentum, self.levels, ne, 'momentum tendency')
        es = _field_strides(edge_density, self.levels, ne, 'edge density')
        scale = backscatter_rate_scale(duration, c.time_s, c.interval_s)
        guard, rtheta_state, qv_state, gs, vs, counter = _guard_fields(
            self, state, names, nc, self.levels, has_qv, theta, zz=row.get('zz'), own_metric=False)
        _kernel('force_cells')(((self.levels * nc + 127) // 128,), (128,),
            (theta, qv, density, taper, sppt, thermal, self.dissipation,
             *ts, *qs, *rs, np.int32(nc), np.int32(self.levels),
             np.int32(c.sppt), np.int32(c.skeb), np.int32(has_qv),
             np.int32(c.logit), np.float32(c.clip),
             np.float32(scale * c.thermal_fraction * 300 / 1004),
             np.int32(guard), rtheta_state, qv_state, row.get('zz') if guard else theta,
             *gs, *vs, np.float32(dt), counter))
        # Every local edge, including halos, reads the ORIGINAL global dual
        # vertices and cell tapers. No truncated class topology enters curl.
        _kernel('force_lts_edges')(((self.levels * ne + 127) // 128,), (128,),
            (momentum, edge_density, self.global_physics.taper,
             self.global_physics.vertex_taper, self.sppt, self.psi,
             self.dissipation, self.norm, r.coe, r.voe, r.orientation, r.dv, row['edges'],
             *ms, *es, np.int32(r.cells), np.int32(self.global_physics.vertex_taper.shape[1]),
             np.int32(ne), np.int32(self.levels), np.int32(c.sppt), np.int32(c.skeb),
             np.int32(c.logit), np.float32(c.clip), np.float32(scale * c.skeb_fraction)))
        self.calls[index] += 1
        self.class_elapsed[index] += dt
        return held

    def end_macro(self):
        import cupy as cp
        if self.macro is None:
            raise ValueError('no stochastic macro to finish')
        start, duration = self.macro
        if any(abs(value - duration) > 1e-8 for value in self.class_elapsed):
            raise ValueError('stochastic macro is missing an owned class interval')
        if self.config.skeb:
            r = self.resource
            _kernel('lts_sink_cells')(((self.levels * r.cells + 127) // 128,), (128,),
                (self.sink_integral, self.cell_sink, self.eoc, self.nec, r.edge_weight,
                 np.int32(r.cells), np.int32(r.edges), np.int32(self.eoc.shape[1]),
                 np.int32(self.levels), np.float32(duration)))
            self.dissipation[...] = cp.sum(self.cell_sink * r.area[None, :], axis=1, keepdims=True)
        self.boundary_seconds = start + duration
        self.macro = None
        self.local_fields = []
        self.macro_calls += 1

    def snapshot(self, *, device=True):
        import cupy as cp
        if self.macro is not None:
            raise ValueError('stochastic checkpoint requires a completed macro boundary')
        return dict(schema=self.schema, config=self.config.state_json(),
            process=self.process.snapshot(device=device), boundary_seconds=self.boundary_seconds,
            dissipation=(self.dissipation.copy() if device else cp.asnumpy(self.dissipation)),
            macro_calls=self.macro_calls)

    def restore(self, value):
        import cupy as cp
        if value['schema'] != self.schema or value['config'] != self.config.state_json():
            raise ValueError('LTS stochastic checkpoint configuration differs')
        if self.macro is not None:
            raise ValueError('cannot restore a stochastic process inside an active macro')
        budget = cp.asarray(value['dissipation'])
        boundary = value['boundary_seconds']
        macro_calls = value['macro_calls']
        if (type(macro_calls) is not int or macro_calls < 0
                or (boundary is None) != (macro_calls == 0)):
            raise ValueError('invalid LTS stochastic checkpoint macro counter')
        if (budget.shape != self.dissipation.shape or budget.dtype != cp.float32
                or not bool(cp.all(cp.isfinite(budget) & (budget >= 0)))
                or (boundary is not None and (not math.isfinite(boundary) or boundary < 0))):
            raise ValueError('invalid LTS stochastic checkpoint budget or boundary')
        self.process.restore(value['process'])
        if boundary is not None and not 0 <= self.process.base.last_time < boundary:
            raise ValueError('stochastic process clock is outside its carried macro boundary')
        self.dissipation[...] = budget
        self.boundary_seconds = boundary
        self.macro_calls = macro_calls

    def receipt(self):
        return dict(schema=self.schema, config=asdict(self.config), device='GPU',
            spectral_processes=1, pattern_clock='one global draw at each macro start',
            draws=self.process.base.draw, macro_calls=self.macro_calls, calls_by_class=self.calls,
            budget='area-weighted global owned-cell mean of time-integrated owned-edge sinks',
            budget_lag_macros=1, startup_budget='zero before the first accepted macro',
            curl='global dual-vertex field, gathered by global edge ID including halos',
            checkpoint='mesh-independent spectral coefficients, counters and per-level dissipation',
            basis=self.resource.receipt, scientific_status='spread-skill qualification required')


#: NOAA's published RRFS v2 ensemble SPPT settings (rrfs-v2-hrrrv5 namelist, &nam_stochy; this tree's
#: verification/source-namelists): one scale of amplitude 0.8, 450 km and 6 h, the smooth bound, the surface
#: limit, the 15-27 km stratospheric taper and an update every step; SKEB off. The base the RRFS suite route
#: resolves --sppt/--skeb from (a namelist's own values and --stochastic JSON still win, in that order).
#: The seed is this module's default: each member's stream comes from (member, seed).
RRFS_V2_ENSEMBLE = dict(sppt_scales=((0.8, 450000.0, 21600.0, 53),), logit=True, surface_limit=True,
                        top_start_m=15000.0, top_end_m=27000.0, interval_s=0.0)


def _json_options(value):
    if value is None:
        return {}
    options = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(options, Mapping):
        raise ValueError('stochastic configuration must be a JSON object')
    return dict(options)


def resolve_options(value=None, namelist=None, *, base=None):
    """One validated configuration for the public door and either physics route.

    Precedence: ``base`` (a route's defaults, e.g. :data:`RRFS_V2_ENSEMBLE`), then the namelist's
    ``&nam_stochy``, then ``value`` (``--stochastic`` JSON and the ``--sppt/--skeb`` switches).
    The namelist's scale and taper controls apply whenever the run turns a scheme on, by the
    namelist's own ``do_sppt/do_skeb`` or by the switches.
    """
    explicit = _json_options(value)
    options = dict(base or {})
    if namelist is not None:
        from .config import parse_fortran_namelist
        groups = namelist if isinstance(namelist, Mapping) else parse_fortran_namelist(namelist)
        for group in groups.values():
            for key in ('do_sppt','do_skeb','config_stochini'):
                if key in group:
                    if type(group[key]) is not bool:
                        raise ValueError(f'{key} must be logical')
                    if key == 'config_stochini':
                        options['restart_required'] = group[key]
                    else:
                        options[{'do_sppt':'sppt','do_skeb':'skeb'}[key]] = group[key]
        params = {key: val for group in groups.values() for key,val in group.items()
                  if key.startswith('config_sppt') or key.startswith('config_iseed_sppt')}
        if (options.get('sppt') or options.get('skeb')
                or explicit.get('sppt') is True or explicit.get('skeb') is True):
            mapping = {'config_sppt_hgt_top1':'top_start_m','config_sppt_hgt_top2':'top_end_m',
                       'config_sppt_logit':'logit', 'config_sppt_sfclimit':'surface_limit',
                       'config_spptint':'interval_s'}
            defaults = StochasticConfig()
            scales = []
            indexed = {f'config_sppt_{suffix}_{i}' for suffix in ('lscale', 'tau') for i in range(1, 6)}
            indexed.update(f'config_sppt_{i}' for i in range(1, 6))
            indexed.update(f'config_iseed_sppt{i}' for i in range(1, 6))
            explicit_scales = bool(indexed.intersection(params))
            consumed = set(mapping)
            for index in range(1, 6):
                keys = (f'config_sppt_{index}', f'config_sppt_lscale_{index}',
                        f'config_sppt_tau_{index}', f'config_iseed_sppt{index}')
                consumed.update(keys)
                # Switches and taper/cadence controls do not declare independent
                # scales. Keep the base pattern so later scalar JSON controls apply.
                if not explicit_scales or (index != 1 and not any(key in params for key in keys)):
                    continue
                amplitude = params.get(keys[0], defaults.amplitude if index == 1 else 0.0)
                if not math.isfinite(amplitude) or amplitude < 0:
                    raise ValueError(f'{keys[0]} must be nonnegative and finite')
                if amplitude:
                    seed = params.get(keys[3], defaults.seed+index-1)
                    # MPAS templates quote the decimal int64 seed. Preserve its
                    # exact integer value without a float conversion.
                    if isinstance(seed, str) and seed.isascii() and seed.isdecimal():
                        seed = int(seed)
                    scales.append((amplitude, params.get(keys[1], defaults.length_m),
                        params.get(keys[2], defaults.time_s), seed))
            if scales:
                options['sppt_scales'] = tuple(scales)
            elif any(key.startswith('config_sppt_') and key[-1:].isdigit() for key in params):
                options['sppt'] = False
            for key,val in params.items():
                if key in mapping:
                    # config_sppt_sfclimit selects this module's own height
                    # ramp (surface_m); config_sppt_logit its smooth bound.
                    options[mapping[key]] = val
                elif key in consumed:
                    pass
                else:
                    raise ValueError(f'{key}={val!r} is not implemented by the hex stochastic forcing; '
                                     'use --stochastic JSON for its length, time, amplitude and taper controls')
    options.update(explicit)
    return StochasticConfig.parse(options)
