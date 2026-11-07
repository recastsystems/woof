"""PWP ocean columns on the MPAS physics route (a hex extension, not part of MPAS-A).

Selected by the same door flag as the WOOF route (``--ocean-mode pwp``), with the same column
physics (Price, Weller and Pinkel 1986, doi:10.1029/JC091iC07p08411; ``kernels/ocean_pwp.cuh``)
and the same parameter table (``data/ocean-pwp.v1.json``, ``--pwp-options``). On this route the
columns take the OML1D slot's place and timing in the MPAS driver: after the surface layer and
before the land surface, once per ``dt_pbl`` (mpas_atmphys_driver.F, ``run("oml")``), forced by
MPAS's own pool fluxes:

- heat: ``gsw`` (net shortwave, absorbed with depth), ``sfc_emiss*glw - sfc_emiss*sigma*T^4``,
  ``-hfx - lh``;
- momentum: ``rho_air*ust^2`` along the lowest-level reconstructed wind, as the OML1D driver
  takes its wind;
- freshwater: ``(raincv + rainncv)/dt - qfx`` (virtual salt flux).

The top layer is published as ``sst`` and ``skintemp`` (what the surface layer reads next step) and
as MPAS's own mixed-layer diagnostics ``t_oml``/``h_oml``/``hu_oml``/``hv_oml``. Columns step only on
open water (``xland`` water, no sea ice, not a preserved lake), as on the WOOF route where PWP is
the tropical open-water option and ice stays with the sea-ice scheme.

With PWP off nothing here runs, so the matched MPAS configuration is unchanged. The columns are
carried by the moving/checkpoint carrier (``moving_v842``; on several cards ``moving_mpi_v842``
assembles the rank-owned columns and hands each rank its owned cells back): same-mesh restarts copy
them, moves take each target cell's coherent surface donor, and a wet cell with no started donor
starts from its own SST with the configured thermocline prior.
"""
from __future__ import annotations

from datetime import datetime
from importlib.resources import files
import hashlib
import os

import numpy as np

STEFAN_BOLTZMANN = 5.67051e-8

CUDA_SOURCE = r"""
extern "C" __global__ void pwp_mpas_step(const int n, const double dt, const double rain_dt,
    const double omega, const double mixed_depth, const double gradient,
    const float* xland, const float* xice, const float* lake,
    const float* hfx, const float* lh, const float* qfx, const float* gsw, const float* glw,
    const float* emiss, const float* ust, const float* u, const long long us,
    const float* v, const long long vs, const float* psfc, const float* t2m,
    const float* rain_c, const float* rain_nc, const float* lat,
    double* columns, const double* pp, double* diagnostics,
    float* sst, float* tsk, float* t_oml, float* h_oml, float* hu_oml, float* hv_oml)
{
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  if (c >= n) return;
  if (!(xland[c] > 1.5f) || xice[c] > 0.f || (lake && lake[c] > 0.5f)) return;
  const int nz = int(pp[0]);
  const double dz = pp[1];
  double* x = columns + (long long)c*4*nz;
  double* d = diagnostics + 4*(long long)c;
  if (d[3] < 0.5) {
    // A column that has not started: this cell's SST over the configured thermocline prior
    // (the WOOF route's PWP initialization), fixed salinity, at rest.
    const double top = double(sst[c]);
    for (int k = 0; k < nz; k++) {
      const double z = (k + 0.5)*dz;
      x[k] = top - fmax(0.0, z - mixed_depth)*gradient;
      x[nz + k] = pp[13];
      x[2*nz + k] = 0.0;
      x[3*nz + k] = 0.0;
    }
    d[0] = mixed_depth; d[1] = 0.0; d[3] = 1.0;
  }
  const double t = x[0];
  const double rho_air = double(psfc[c])/(287.04*fmax(180.0, double(t2m[c])));
  const double uu = u[c*us], vv = v[c*vs];
  const double speed = fmax(1.e-10, hypot(uu, vv));
  const double stress = rho_air*double(ust[c])*double(ust[c]);
  const double qsw = double(gsw[c]);
  const double e = double(emiss[c]);
  const double qnon = e*double(glw[c]) - e*5.67051e-8*t*t*t*t - double(hfx[c]) - double(lh[c]);
  double rain = 0.0;
  if (rain_c) rain += fmax(0.0, double(rain_c[c]));
  if (rain_nc) rain += fmax(0.0, double(rain_nc[c]));
  const double freshwater = rain/1000.0/rain_dt - double(qfx[c])/1000.0;
  const double f = 2.0*omega*sin(double(lat[c]));
  pwp_advance(x, pp, dt, f, stress*uu/speed, stress*vv/speed, qsw, qnon, freshwater, d);
  const float top = float(x[0]);
  sst[c] = top; tsk[c] = top; t_oml[c] = top;
  h_oml[c] = float(d[0]);
  hu_oml[c] = float(x[2*nz]*d[0]);
  hv_oml[c] = float(x[3*nz]*d[0]);
}
"""


def _kernel():
    import cupy as cp
    from .. import numerics
    source = files("woof.hex").joinpath("kernels/ocean_pwp.cuh").read_text() + CUDA_SOURCE
    opts = tuple(v for v in numerics.options(("--std=c++17",)) if v.lstrip("-").split("=", 1)[0] != "ftz")
    return cp.RawKernel(source, "pwp_mpas_step", options=opts)


def _vector(pools, pool, name, n, *, required=True):
    import cupy as cp
    try:
        array = pools[pool][name]
    except KeyError:
        if required:
            raise ValueError(f"PWP on the MPAS route needs {pool}.{name}, which this suite does not allocate")
        return None
    if not isinstance(array, cp.ndarray) or array.dtype != np.float32:
        raise TypeError(f"PWP {pool}.{name} must be a CUDA float32 array")
    if array.ndim != 1 or not array.flags.c_contiguous or array.size < n:
        raise ValueError(f"PWP {pool}.{name} must be a contiguous cell vector with at least {n} cells")
    return array


def _pointer(array):
    """An optional device vector: the array, or a null pointer the kernel tests."""
    return np.uint64(0) if array is None else array


def carried_columns(state, donors=None):
    """A carrier's PWP state on the target mesh: as it is (same mesh), or each target cell's coherent
    surface donor column after a move. Iteration-limit counts and energy residuals stay with the donor's
    history; the carried mixed depth and started flag decide the column."""
    if state["columns"] is None:
        return dict(state)
    columns, diagnostics = np.asarray(state["columns"]), np.asarray(state["diagnostics"])
    if donors is not None:
        donors = np.asarray(donors)
        columns, diagnostics = columns[donors], diagnostics[donors]
    return dict(state, columns=np.ascontiguousarray(columns), diagnostics=np.ascontiguousarray(diagnostics))


def assemble_owned(states, partitions):
    """Rank-owned PWP states (multi-card moving carrier) as one global-cell state, never reading a halo.

    Each rank holds columns[n_owned, 4, levels] and diagnostics[n_owned, 4] in its local owned order."""
    from .moving_mpi_v842 import assemble_columns
    if all(state is None for state in states):
        return None
    if any(state is None for state in states):
        raise ValueError("PWP ocean is selected on some MPAS ranks and not on others")
    first = states[0]
    for state in states[1:]:
        if (state["identity"], state["levels"], state["layer_m"]) != (first["identity"], first["levels"], first["layer_m"]):
            raise ValueError("MPAS ranks carry PWP columns with different identities or layering")
    started = [state["columns"] is not None for state in states]
    if not any(started):
        return dict(first)
    if not all(started):
        raise ValueError("PWP columns started on some MPAS ranks and not on others")
    columns = assemble_columns([np.moveaxis(np.asarray(state["columns"]), 0, -1) for state in states],
                               partitions, entity="cell")
    diagnostics = assemble_columns([np.asarray(state["diagnostics"]).T for state in states], partitions, entity="cell")
    return dict(first, columns=np.ascontiguousarray(np.moveaxis(columns, -1, 0)),
                diagnostics=np.ascontiguousarray(diagnostics.T))


def project_owned(state, partition):
    """A global PWP state restricted to one rank's owned cells, in its local owned order."""
    if state is None or state["columns"] is None:
        return state
    ids = partition.cell_l2g[:partition.n_owned_cells]
    return dict(state, columns=np.ascontiguousarray(np.asarray(state["columns"])[ids]),
                diagnostics=np.ascontiguousarray(np.asarray(state["diagnostics"])[ids]))


class OceanWords:
    """Host PWP state for owner assembly on the multi-card MPAS moving carrier (moving_mpi_v842)."""

    def __init__(self, state=None):
        self.state = state

    def restart_state(self):
        return self.state

    def restore_state(self, state, donors=None):
        if state is None:
            raise ValueError("this run selects PWP but the carrier holds no PWP ocean state (the case's ocean "
                             "selection changed between segments); a carried run keeps one ocean model")
        self.state = carried_columns(state, donors)


class MpasPwpOcean:
    """Device PWP columns for one rank's owned cells, stepped from the OML slot."""

    def __init__(self, provider, options):
        from ..ocean_pwp import KEYS
        self.provider = provider
        self.options = dict(options)
        self.columns = None
        self.diagnostics = None
        self.restored = None
        self.steps = 0
        self.initialization = None
        self.parameters = None
        self.kernel = None
        self._keys = KEYS
        self.lake = None

    # ------------------------------------------------------------------ carrier
    def restart_state(self):
        """Always a dict when PWP is selected, so a carrier tells "not started yet" from "no PWP"."""
        if self.columns is None:
            if self.restored is not None:
                return self.restored
            return dict(identity=self.options["identity"], levels=int(self.options["levels"]),
                        layer_m=float(self.options["layer_m"]), columns=None, diagnostics=None,
                        initialization=None)
        cp = self.provider.cp
        return dict(identity=self.options["identity"], levels=int(self.options["levels"]),
                    layer_m=float(self.options["layer_m"]), columns=cp.asnumpy(self.columns),
                    diagnostics=cp.asnumpy(self.diagnostics), initialization=self.initialization)

    def restore_state(self, state, donors=None):
        """Carried columns (same mesh as they are; a move by coherent surface donor)."""
        if state is None:
            raise ValueError("this run selects PWP but the carrier holds no PWP ocean state (the case's ocean "
                             "selection changed between segments); a carried run keeps one ocean model")
        if (state["identity"], int(state["levels"]), float(state["layer_m"])) != (
                self.options["identity"], int(self.options["levels"]), float(self.options["layer_m"])):
            raise ValueError("the carrier's PWP columns have a different identity or layering than this run's "
                             "--pwp-options")
        carried = carried_columns(state, donors)
        # Sealed before the first OML-slot call: start from this member's SST.
        self.restored = None if carried["columns"] is None else carried

    # ------------------------------------------------------------------ setup
    def _start(self, pools, n):
        import cupy as cp
        p = self.options
        nz = int(p["levels"])
        self.parameters = cp.asarray([p[k] for k in self._keys], dtype=cp.float64)
        self.kernel = _kernel()
        native = self.provider.native
        lake = getattr(native, "tc_lakemask", None)
        if lake is not None:
            self.lake = cp.ascontiguousarray(cp.asarray(np.asarray(lake)[:n], dtype=cp.float32))
        if self.restored is not None:
            columns, diagnostics = self.restored["columns"], self.restored["diagnostics"]
            if columns.shape[0] < n or columns.shape[1:] != (4, nz) or diagnostics.shape[0] < n:
                raise ValueError(f"carried PWP columns {columns.shape} do not cover this member's {n} cells")
            self.columns = cp.ascontiguousarray(cp.asarray(columns[:n], dtype=cp.float64))
            self.diagnostics = cp.ascontiguousarray(cp.asarray(diagnostics[:n], dtype=cp.float64))
            self.initialization = self.restored.get("initialization")
            self.restored = None
            return
        self.columns = cp.zeros((n, 4, nz), dtype=cp.float64)
        self.diagnostics = cp.zeros((n, 4), dtype=cp.float64)
        self.initialization = "SST with configured mixed depth and thermocline prior; not an observed subsurface analysis"
        profile = os.environ.get("WOOF_HEX_PWP_PROFILE")
        analysis = os.environ.get("WOOF_HEX_PWP_ANALYSIS")
        if profile and analysis:
            raise ValueError("PWP analysis and pre-remapped profile cannot both supply the initial ocean")
        wet = self._wet(pools, n)
        if analysis:
            from woof.ocean.mixed_layer.analysis import load
            mesh = pools["mesh"]
            start = str(getattr(self.provider.session, "start_time", "")).replace("_", "T")
            mapped = load(analysis, latitude=cp.rad2deg(mesh["latCell"][:n]),
                          longitude=cp.rad2deg(mesh["lonCell"][:n]),
                          land=cp.where(wet, cp.float32(2), cp.float32(1)), levels=nz, layer_m=p["layer_m"],
                          source=os.environ.get("WOOF_HEX_PWP_ANALYSIS_SOURCE", "hycom"),
                          rotation=(cp.zeros(n, cp.float32), cp.ones(n, cp.float32)),
                          expected_time=datetime.fromisoformat(start), uncovered="prescribed")
            covered = wet & (mapped["om_active"] > .5)
            for variable, key in enumerate(("om_tmp", "om_s", "om_u", "om_v")):
                self.columns[:, variable, :] = cp.where(covered[:, None], mapped[key].T, 0)
            self.diagnostics[:, 3] = covered
            self.diagnostics[:, 0] = p["initial_mixed_depth_m"]
            with open(analysis, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.initialization = ("dated ocean analysis, conservative layer means, SHA256=" + digest
                                   + "; uncovered water starts from SST with the configured prior")
        elif profile:
            with np.load(profile, allow_pickle=False) as data:
                host = data["columns"]
            if host.shape[0] < n or host.shape[1:] != (4, nz) or not np.isfinite(host[:n]).all():
                raise ValueError(f"PWP profile must contain finite columns[nCells,4,{nz}] in this mesh's order")
            self.columns[...] = cp.asarray(host[:n])
            self.diagnostics[:, 3] = wet
            self.diagnostics[:, 0] = p["initial_mixed_depth_m"]
            self.initialization = "supplied T/S/u/v profile SHA256=" + hashlib.sha256(open(profile, "rb").read()).hexdigest()

    def _wet(self, pools, n):
        import cupy as cp
        wet = (_vector(pools, "sfc_input", "xland", n)[:n] > 1.5) & ~(_vector(pools, "sfc_input", "xice", n)[:n] > 0)
        if self.lake is not None:
            wet &= ~(self.lake > .5)
        return cp.ascontiguousarray(wet)

    # ------------------------------------------------------------------ step
    def launch_slot(self, call):
        """The OML slot's place in mpas_atmphys_driver.F, PWP columns instead of OML1D."""
        import cupy as cp
        pools = call.pools
        mesh = pools["mesh"]
        n = int(mesh["nCellsSolve"])
        if self.columns is None:
            self._start(pools, n)
        if self.columns.shape[0] != n:
            raise ValueError(f"PWP columns hold {self.columns.shape[0]} cells, this rank solves {n}")
        diag = pools["diag"]
        winds = [diag[name] for name in ("uReconstructZonal", "uReconstructMeridional")]
        for wind in winds:
            if wind.ndim != 2 or wind.shape[0] < 1 or wind.dtype != np.float32:
                raise ValueError("PWP lowest-level winds must be float32 (levels, cells)")
        u, v = winds[0][0], winds[1][0]
        vec = lambda pool, name, required=True: _vector(pools, pool, name, n, required=required)
        sfc = {k: vec("sfc_input", k) for k in ("xland", "xice", "sst", "skintemp")}
        dp = {k: vec("diag_physics", k) for k in ("hfx", "lh", "qfx", "gsw", "glw", "sfc_emiss", "ust", "t2m",
                                                    "t_oml", "h_oml", "hu_oml", "hv_oml")}
        rain_c = vec("diag_physics", "raincv", required=False)
        rain_nc = vec("diag_physics", "rainncv", required=False)
        psfc = vec("diag", "surface_pressure")
        lat = mesh["latCell"]
        p = self.options
        dynamics_dt = float(getattr(self.provider.session, "dynamics_dt", None) or call.dt)
        if n:
            self.kernel(((n + 127)//128,), (128,), (
                np.int32(n), np.float64(call.dt), np.float64(dynamics_dt), np.float64(7.292e-5),
                np.float64(p["initial_mixed_depth_m"]), np.float64(p["initial_gradient_k_m"]),
                sfc["xland"], sfc["xice"], _pointer(self.lake),
                dp["hfx"], dp["lh"], dp["qfx"], dp["gsw"], dp["glw"], dp["sfc_emiss"], dp["ust"],
                u, np.int64(u.strides[0]//4), v, np.int64(v.strides[0]//4), psfc, dp["t2m"],
                _pointer(rain_c), _pointer(rain_nc), lat, self.columns, self.parameters, self.diagnostics,
                sfc["sst"], sfc["skintemp"], dp["t_oml"], dp["h_oml"], dp["hu_oml"], dp["hv_oml"]))
        self.steps += 1

    def receipt(self):
        active = None if self.diagnostics is None else int((self.diagnostics[:, 3] > .5).sum().item())
        failures = None if self.diagnostics is None else int(self.diagnostics[:, 2].sum().item())
        return dict(mode="pwp", route="mpas-physics", slot="oml (dt_pbl, after the surface layer)",
                    options=self.options, steps=self.steps, active_columns=active,
                    mixing_iteration_limit_hits=failures, initialization=self.initialization,
                    source="HYCOM PWP technical specification; DOI 10.1029/JC091iC07p08411",
                    forcing="MPAS pool gsw, glw*sfc_emiss, hfx, lh, ust, raincv+rainncv, qfx; lowest-level wind",
                    scope="open-water columns (no sea ice, no preserved lake); no horizontal ocean dynamics")


def configure(config, environ=os.environ):
    """The DriverConfig for --ocean-mode pwp on this route: the OML slot on, PWP in its place."""
    from dataclasses import replace
    if environ.get("WOOF_HEX_OCEAN_MODE") != "pwp":
        return config
    from ..ocean_pwp import configuration
    if "oml" not in config.selectors:
        raise ValueError("this MPAS suite table has no ocean-mixed-layer slot for the PWP columns to run in")
    return replace(config, ocean_pwp=configuration())


def register(config, provider):
    """Install the PWP columns as the OML slot's CUDA callback (device_pools_v842.register_cuda_slots)."""
    from .driver_v842 import Backend
    if config.backends.get("oml") != Backend.HEX_CUDA:
        raise ValueError("PWP on the MPAS route runs on the GPU in the OML slot; this run puts that slot on "
                         f"{config.backends.get('oml')}")
    ocean = MpasPwpOcean(provider, config.ocean_pwp)
    provider.session.ocean_pwp = ocean
    return ocean
