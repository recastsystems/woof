"""CUDA driver only. All spectral transforms and physics execute on the GPU."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

@dataclass(frozen=True)
class SpectralGrid:
    frequencies: int = 36
    directions: int = 36
    first_hz: float = 0.035
    ratio: float = 1.1

    def __post_init__(self):
        if (not 16 <= self.frequencies <= 128 or not 12 <= self.directions <= 180
                or not math.isfinite(self.first_hz) or self.first_hz <= 0
                or not math.isfinite(self.ratio) or not 1.02 <= self.ratio <= 1.25):
            raise ValueError("spectral grid needs 16..128 frequencies, 12..180 directions, positive Hz, ratio 1.02..1.25")
        if self.first_hz * self.ratio ** (self.frequencies - 1) < 0.4:
            raise ValueError("frequency grid must reach 0.4 Hz to resolve wind generation and dissipation")

class WaveModel:
    """One-way earth-relative U10/V10 input; no CPU fallback.

    Device arrays are float32 C contiguous, spatial axes south to north and west
    to east. Spectrum is surface-elevation variance density E(f,theta), m2/Hz/rad,
    theta travelling counterclockwise from east, shape [ny,nx,nf,nd].
    Dry depth <= 0 absorbs waves. Open edges have zero incident action unless
    supplied by ``set_boundary``. Static bathymetry and no ocean current.
    """
    def __init__(self, depth, latitude, longitude, *, spectral=SpectralGrid()):
        import cupy as cp
        self.cp = cp
        for a in (depth, latitude, longitude):
            if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or not a.flags.c_contiguous:
                raise TypeError("wave geometry must be contiguous CUDA float32 arrays")
        if depth.ndim != 2 or latitude.ndim != 1 or longitude.ndim != 1 or depth.shape != (latitude.size, longitude.size):
            raise ValueError("depth [latitude,longitude] and one-dimensional axes required")
        self.ny, self.nx = depth.shape
        if min(self.ny, self.nx) < 2:
            raise ValueError("propagation needs at least two cells per spatial axis")
        self.depth, self.latitude, self.longitude = depth, latitude, longitude
        for axis in (latitude, longitude):
            diff = cp.diff(axis)
            if not bool(cp.all(cp.isfinite(axis))) or not bool(cp.all(diff > 0)) or not bool(cp.allclose(diff, diff[0], atol=1e-5)):
                raise ValueError("spatial axes must be finite, regular and increasing")
        if not bool(cp.all(cp.isfinite(depth))) or not bool(cp.any(depth > 0)):
            raise ValueError("finite depth with at least one wet cell required")
        if float(cp.max(cp.abs(latitude))) >= 89:
            raise ValueError("regular lat-lon propagation excludes the coordinate singularity within 1 degree of either pole")
        self.dx, self.dy = float(longitude[1] - longitude[0]), float(latitude[1] - latitude[0])
        self.spectral = spectral
        self.nf, self.nd = spectral.frequencies, spectral.directions
        self.nc, self.ns = depth.size, self.nf * self.nd
        self.frequency = spectral.first_hz * cp.power(cp.float32(spectral.ratio), cp.arange(self.nf, dtype=cp.float32))
        self.df = self.frequency * cp.float32(math.sqrt(spectral.ratio) - 1 / math.sqrt(spectral.ratio))
        self.spectrum = cp.zeros((self.ny, self.nx, self.nf, self.nd), dtype=cp.float32)
        self.work = cp.empty_like(self.spectrum)
        self.rate = cp.zeros_like(self.spectrum)
        self.k = cp.empty((self.nc, self.nf), cp.float32)
        self.cg = cp.empty_like(self.k)
        self.turn = cp.empty_like(self.spectrum)
        self.cfl = cp.empty_like(self.spectrum)
        self.moments = cp.zeros((self.nc, 10), cp.float32)
        self.wind = cp.zeros((self.nc, 2), cp.float32)
        self.limiter = cp.ones(self.nc, cp.float32)
        self.bounds = cp.zeros(self.nc, cp.float32)
        self.budgets = cp.zeros((self.ny, self.nx, self.nf, self.nd, 4), cp.float32)
        self.diagnostics = cp.empty((7, self.ny, self.nx), cp.float32)
        source = Path(__file__).with_name("kernels.cu").read_text()
        from woof.certify.kernel_manifest import record_module
        self.module = cp.RawModule(code=source, options=("--std=c++17", f"-DWAVES_DIRECTIONS={self.nd}"))
        record_module(f"woof.ocean.waves:kernels:directions={self.nd}", source=source,
                      options=("--std=c++17", f"-DWAVES_DIRECTIONS={self.nd}"), module=self.module)
        self._functions = {name: self.module.get_function(name) for name in (
            "geometry", "propagate", "refract", "moments", "sources", "source_integrate", "dia_apply", "diagnose", "input_bound")}
        # The DIA exchange gathers in a fixed order (ordered.cu); its atomicAdd scatter differed run to run.
        from .ordered import OrderedDia, compile_module
        _, _, self.ordered_module = compile_module(key=f"woof.ocean.waves:ordered:directions={self.nd}",
                                                   fast_math=False)
        self.dia = OrderedDia(self.ordered_module, self.frequency, self.df, self.nd,
                              math.log(self.spectral.ratio), diagonal=False)
        self._launch("geometry", self.nc * self.ns, (depth, latitude, self.frequency,
            self.k, self.cg, self.turn, self.cfl, self.i(self.nx), self.i(self.ny),
            self.i(self.nf), self.i(self.nd), self.f(self.dx), self.f(self.dy)))
        self.max_dt = 0.8 / float(cp.max(self.cfl))
        self.timing = {k: 0.0 for k in ("wind_and_moments", "propagation", "local_sources", "DIA", "diagnostics")}
        self._events = []
        self.boundary = None

    i = staticmethod(np.int32)
    f = staticmethod(np.float32)

    def _launch(self, name, n, args, *, blocks=False):
        self._functions[name]((n if blocks else (n + 127) // 128,), (128,), args)

    def _timed(self, name, action):
        cp = self.cp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        action()
        end.record()
        self._events.append((name, start, end))

    def collect_timing(self):
        self.cp.cuda.get_current_stream().synchronize()
        for name, start, end in self._events:
            self.timing[name] += self.cp.cuda.get_elapsed_time(start, end) / 1000
        self._events.clear()

    def _moments(self, u0, v0, u1, v1, alpha):
        self._launch("moments", self.nc, (self.spectrum, self.frequency, self.df,
            self.k, self.depth, u0, v0, u1, v1, self.f(alpha), self.wind,
            self.moments, self.i(self.nc), self.i(self.nf), self.i(self.nd)), blocks=True)

    def set_boundary(self, spectrum):
        """Prescribed incident spectra on domain edges, GPU [ny,nx,nf,nd].

        Interior values are ignored. Only edge bins travelling into the domain
        are imposed after each propagation step. No default artificial swell.
        """
        cp = self.cp
        if not isinstance(spectrum, cp.ndarray) or spectrum.shape != self.spectrum.shape or spectrum.dtype != cp.float32:
            raise ValueError("boundary must match the CUDA spectrum")
        if not bool(cp.all(cp.isfinite(spectrum))) or not bool(cp.all(spectrum >= 0)):
            raise ValueError("boundary spectrum must be finite and nonnegative")
        self.boundary = spectrum.copy()

    def _apply_boundary(self):
        if self.boundary is None:
            return
        cp = self.cp
        angle = cp.arange(self.nd, dtype=cp.float32) * self.f(2 * math.pi / self.nd)
        for selector, incoming in (( (slice(None), 0), cp.cos(angle)>0),
                                   ( (slice(None), -1), cp.cos(angle)<0),
                                   ( (0, slice(None)), cp.sin(angle)>0),
                                   ( (-1, slice(None)), cp.sin(angle)<0)):
            self.spectrum[selector] = cp.where(incoming, self.boundary[selector], self.spectrum[selector])
        self.spectrum *= (self.depth > 0)[:, :, None, None]

    def step(self, u0, v0, u1, v1, *, alpha, dt_limit):
        cp = self.cp
        if not math.isfinite(dt_limit) or dt_limit <= 0 or not 0 <= alpha <= 1:
            raise ValueError("positive finite dt and forcing interpolation in [0,1] required")
        for a in (u0, v0, u1, v1):
            if not isinstance(a, cp.ndarray) or a.dtype != cp.float32 or a.shape != self.depth.shape or not a.flags.c_contiguous:
                raise ValueError("winds must match the CUDA spatial grid")
        self._timed("wind_and_moments", lambda: self._moments(u0, v0, u1, v1, alpha))
        self._launch("input_bound", self.nc, (self.frequency, self.k, self.depth,
            self.wind, self.moments, self.bounds, self.i(self.nc), self.i(self.nf), self.i(self.nd)), blocks=True)
        growth = float(cp.max(self.bounds))
        dt = min(dt_limit, self.max_dt, 0.2 / max(growth, 1e-20))
        def propagate():
            self._launch("propagate", self.nc * self.ns, (self.spectrum, self.work,
                self.depth, self.latitude, self.cg, self.turn, self.i(self.nx), self.i(self.ny),
                self.i(self.nf), self.i(self.nd), self.f(self.dx), self.f(self.dy), self.f(dt)))
            self.spectrum, self.work = self.work, self.spectrum
            self._launch("refract", self.nc * self.nf, (self.spectrum, self.turn,
                self.depth, self.i(self.nc), self.i(self.nf), self.i(self.nd), self.f(dt)))
            self._apply_boundary()
        self._timed("propagation", propagate)
        self._timed("wind_and_moments", lambda: self._moments(u0, v0, u1, v1, alpha))
        def local():
            self._launch("sources", self.nc * self.ns, (self.spectrum, self.frequency,
                self.k, self.depth, self.wind, self.moments, self.i(self.nc), self.i(self.nf),
                self.i(self.nd), self.f(dt), self.budgets))
        self._timed("local_sources", local)
        # Recompute spectral moments after growth for finite-depth DIA scaling.
        self._timed("wind_and_moments", lambda: self._moments(u0, v0, u1, v1, alpha))
        def nonlinear():
            self.rate.fill(0)
            self.dia.launch(self.spectrum, self.rate, None, self.frequency, self.df, self.depth,
                            self.moments, self.nc, dt=dt, limited=True)
            self._launch("dia_apply", self.nc, (self.spectrum, self.rate, self.limiter,
                self.i(self.nc), self.i(self.nf), self.i(self.nd), self.f(dt)), blocks=True)
        self._timed("DIA", nonlinear)
        return dt

    def output(self, u, v):
        self._moments(u, v, u, v, 0.0)
        self._timed("diagnostics", lambda: self._launch("diagnose", self.nc,
            (self.spectrum, self.df, self.frequency, self.moments, self.depth,
             self.diagnostics, self.i(self.nc), self.i(self.nf), self.i(self.nd))))
        return self.diagnostics

    def coupling_fields(self):
        """Device-only waves.surface.v1 interface, earth-relative grid.

        Call output() at the coupling time first. Effective z0 includes wave
        stress. No extra host copies or wave physics path is needed by a caller.
        """
        return {"schema": "arwen.waves.surface.v1", "hs_m": self.diagnostics[0],
                "tp_s": self.diagnostics[1], "mean_from_direction_deg": self.diagnostics[2],
                "windsea_hs_m": self.diagnostics[3], "swell_hs_m": self.diagnostics[4],
                "roughness_m": self.diagnostics[5], "friction_velocity_ms": self.diagnostics[6],
                "wet_mask": self.depth > 0}
