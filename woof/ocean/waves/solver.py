"""Shared ocean ABI: persistent CUDA state and asynchronous coupled stepping."""
from dataclasses import dataclass
import math
from .model import WaveModel, SpectralGrid

@dataclass(frozen=True)
class WaveOutputs:
    hs: object
    tp: object
    direction: object

class CoupledWaveSolver:
    def __init__(self, grid, bathymetry, config):
        import cupy as cp
        self.cp, self.shape = cp, (grid.ny, grid.nx)
        def plane(a, name):
            if not hasattr(a, '__cuda_array_interface__'):
                raise TypeError(f'{name} must be a CUDA plane')
            a = cp.asarray(a)
            if a.shape != self.shape or a.dtype != cp.float32 or not a.flags.c_contiguous:
                raise ValueError(f'{name} must be CUDA float32 C contiguous {self.shape}')
            return a
        self._plane = plane
        lat, lon = plane(grid.latitude,'latitude'), plane(grid.longitude,'longitude')
        if not bool(cp.allclose(lat,lat[:,0:1])) or not bool(cp.allclose(lon,lon[0:1,:])):
            raise ValueError('spectral waves require a regular geographic grid; curvilinear/projected wave propagation is not implemented')
        self.bathymetry = bathymetry
        self.base_depth = plane(bathymetry.depth,'depth').copy()
        self.wet = plane(bathymetry.wet,'wet') > .5
        depth = cp.where(self.wet,self.base_depth,cp.float32(0))
        self.model = WaveModel(depth,cp.ascontiguousarray(lat[:,0]),cp.ascontiguousarray(lon[0,:]),
                               spectral=SpectralGrid(**config.get('spectral',{})))
        # A conservative finite-depth group-speed envelope, independent of eta.
        # Constructor synchronization is allowed; step/setters never read scalars.
        latitude_max = float(cp.max(cp.abs(lat)))
        dx = 6371000*math.cos(math.radians(latitude_max))*math.radians(self.model.dx)
        dy = 6371000*math.radians(self.model.dy)
        c_bound = 9.80665/(2*math.pi*self.model.spectral.first_hz)
        cap = float(config.get('max_transport_step_seconds',60))
        if not math.isfinite(cap) or cap<=0:
            raise ValueError('max_transport_step_seconds must be positive and finite')
        self.transport_step = min(cap,.8/(2*c_bound*(1/dx+1/dy)))
        self.forcing = None
        self._outputs = WaveOutputs(*(cp.zeros(self.shape,cp.float32) for _ in range(3)))

    @property
    def outputs(self):
        return self._outputs

    def set_forcing(self, forcing):
        self._plane(forcing.u10,'u10'); self._plane(forcing.v10,'v10')
        self.forcing = forcing

    def set_water_level(self, eta):
        cp, m = self.cp, self.model
        eta = self._plane(eta,'eta')
        # Preserve the prepared ocean-connected coastline. Inland flooding is
        # the surge component's prognostic domain, not a new ocean connection.
        m.depth[:] = cp.where(self.wet,cp.maximum(self.base_depth+eta,0),0)
        m._launch('geometry',m.nc*m.ns,(m.depth,m.latitude,m.frequency,m.k,m.cg,m.turn,m.cfl,
            m.i(m.nx),m.i(m.ny),m.i(m.nf),m.i(m.nd),m.f(m.dx),m.f(m.dy)))

    def step(self, dt):
        if self.forcing is None:
            raise RuntimeError('set_forcing must precede a wave step')
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('wave step must be positive and finite')
        m, cp = self.model, self.cp
        u,v = cp.asarray(self.forcing.u10),cp.asarray(self.forcing.v10)
        pieces=math.ceil(dt/self.transport_step); h=dt/pieces
        for _ in range(pieces):
            m._launch('propagate',m.nc*m.ns,(m.spectrum,m.work,m.depth,m.latitude,m.cg,m.turn,
                m.i(m.nx),m.i(m.ny),m.i(m.nf),m.i(m.nd),m.f(m.dx),m.f(m.dy),m.f(h)))
            m.spectrum,m.work=m.work,m.spectrum
            m._launch('refract',m.nc*m.nf,(m.spectrum,m.turn,m.depth,m.i(m.nc),m.i(m.nf),m.i(m.nd),m.f(h)))
            m._apply_boundary()
            m._launch('source_integrate',m.nc,(m.spectrum,m.frequency,m.df,m.k,m.depth,u,v,m.wind,
                m.moments,m.i(m.nc),m.i(m.nf),m.i(m.nd),m.f(h),m.budgets),blocks=True)
            m._moments(u,v,u,v,0)
            m.rate.fill(0)
            m._launch('dia',m.nc*m.ns,(m.spectrum,m.rate,m.frequency,m.df,m.depth,m.moments,
                m.i(m.nc),m.i(m.nf),m.i(m.nd),m.f(math.log(m.spectral.ratio)),m.f(h)))
            m._launch('dia_apply',m.nc,(m.spectrum,m.rate,m.limiter,m.i(m.nc),m.i(m.nf),m.i(m.nd),m.f(h)),blocks=True)
        m._moments(u,v,u,v,0)
        m._launch('diagnose',m.nc,(m.spectrum,m.df,m.frequency,m.moments,m.depth,
            m.diagnostics,m.i(m.nc),m.i(m.nf),m.i(m.nd)))
        wet = m.depth>0
        # Match diagnose's variance cutoff for every shared output. A separate
        # Hs cutoff admitted waves whose period/direction were undefined. Keep
        # invalid wet diagnostics visible: NaN variance is not calm water.
        calm = m.moments[:,0].reshape(self.shape) <= m.f(1e-15)
        moving = wet & ~calm
        self._outputs.hs[:]=cp.where(moving,m.diagnostics[0],0)
        self._outputs.tp[:]=cp.where(moving,m.diagnostics[1],0)
        # Shared ABI travels TOWARDS clockwise from north; model is FROM.
        self._outputs.direction[:]=cp.where(moving,(m.diagnostics[2]+180)%360,0)


def create_solver(grid, bathymetry, *, config=None):
    from .action import ActionSolver
    return ActionSolver(grid,bathymetry,config or {})
