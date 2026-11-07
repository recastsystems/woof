"""Ordinary go binding for committed-step, fixed-Earth cyclone diagnostics.

The surface gust is the rw-post similarity diagnostic, evaluated on CUDA at
all committed steps. It is a diagnostic gust, not a resolved three-second gust.
"""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import json
import math
import time

_ACTIVE = ContextVar("gpuwm_go_cyclone", default=None)
PRODUCTS = ("max-wind", "gust", "min-pressure", "track", "rain")


def utc_datetime(value):
    """Model clocks without an offset are UTC, as in experiment TOML."""
    from datetime import datetime, timezone
    value = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_request(path, config):
    from woof.go_cli import GoRefusal
    path = Path(path).resolve()
    try:
        request = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise GoRefusal(f"--cyclone-products cannot read {path}: {error}; no cyclone diagnostics can be bound") from None
    if not isinstance(request, dict):
        raise GoRefusal("--cyclone-products needs a JSON object with a common_grid")
    allowed = {"common_grid", "grid_id", "products", "experiment_label", "analysis_time",
               "onshore_threshold_m_s"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise GoRefusal("--cyclone-products unknown request keys: " + ", ".join(unknown)
                        + "; these would otherwise be silently ignored")
    products = validate_swath_request(request)
    import tomllib
    payload = tomllib.loads(Path(config).read_text())
    domains = payload.get("domain", [])
    grid_id = request.get("grid_id", 2 if len(domains) > 1 else 1)
    if type(grid_id) is not int or grid_id not in {d["grid_id"] for d in domains}:
        raise GoRefusal(f"--cyclone-products grid_id {grid_id!r} is absent from CONFIG; no committed steps could be observed")
    # [tiles] mode = "auto" (what cyclone-setup authors) keeps a domain resident whenever
    # it fits the card; only a domain the forecast actually streams lacks the complete
    # every-step plane, and committed_step refuses that one by name.  "on" streams every
    # domain, so it can never supply the plane.
    if payload.get("tiles", {}).get("mode", "off") not in ("off", "auto"):
        raise GoRefusal("--cyclone-products requires resident columns: [tiles] mode = 'on' streams every domain, and a streamed state holds only one tile and cannot supply a complete every-step pressure/gust plane; use mode 'auto' or 'off'")
    if payload.get("ensemble", {}).get("members", 1) > 1:
        raise GoRefusal("--cyclone-products ordinary go currently binds one trajectory; multiple members would overwrite its single-track receipt. Use the branch ensemble cyclone product route")
    request = {**request, "grid_id": grid_id, "products": products}
    request.setdefault("experiment_label", payload["experiment"]["name"])
    request.setdefault("analysis_time", payload["experiment"]["start_time"].isoformat() + ("Z" if payload["experiment"]["start_time"].tzinfo is None else ""))
    try:
        request['analysis_time'] = utc_datetime(request['analysis_time']).isoformat()
    except (ValueError, TypeError, AttributeError):
        raise GoRefusal("--cyclone-products analysis_time must be an ISO timestamp; an invalid epoch would misdate the whole track") from None
    return request


def validate_swath_request(request):
    """Validate the shared fixed-frame geometry and supported products."""
    from woof.go_cli import GoRefusal
    products = request.get("products", list(PRODUCTS))
    if not isinstance(products, list) or not products or any(p not in PRODUCTS for p in products):
        raise GoRefusal(f"--cyclone-products unknown or empty products {products!r}; supported: {', '.join(PRODUCTS)}")
    grid = request.get("common_grid")
    if not isinstance(grid, dict) or set(grid) != {"south", "north", "west", "east", "nx", "ny"}:
        raise GoRefusal("--cyclone-products common_grid needs exactly south,north,west,east,nx,ny; cannot locate the Earth-fixed swath")
    if any(type(grid[k]) is not int or grid[k] < 2 for k in ("nx", "ny")):
        raise GoRefusal("--cyclone-products nx and ny must be integers >= 2; bilinear sampling needs two points per axis")
    try:
        valid = all(math.isfinite(grid[k]) for k in ("south", "north", "west", "east"))
        valid &= -90 <= grid["south"] < grid["north"] <= 90
        valid &= -180 <= grid["west"] < grid["east"] <= 180
    except TypeError:
        valid = False
    if not valid:
        raise GoRefusal("--cyclone-products needs finite ordered latitude/longitude bounds; this request cannot define a fixed Earth grid")
    return products


def bind_output(path):
    capture = _ACTIVE.get()
    if capture is not None:
        capture.root = Path(path) / "cyclone"


def active():
    return _ACTIVE.get() is not None


def _ensemble_call(args):
    """Leave roster diagnostics to the ensemble door that owns the members."""
    if (getattr(args, "command", None) == "ensemble"
            or (getattr(args, "members", None) or 1) > 1
            or getattr(args, "recipe", None) is not None
            or getattr(args, "trajectories", None) is not None):
        return True
    source = getattr(args, "source", None)
    if source:
        from woof.ensemble.wofs_recipe import parse_source_spec
        if len(parse_source_spec(source)) > 1:
            return True
    config = getattr(args, "config", None)
    if config is not None and Path(config).is_file():
        import tomllib
        value = tomllib.loads(Path(config).read_text()).get("ensemble", {})
        if isinstance(value, dict):
            return (value.get("members", 1) > 1 or value.get("recipe") is not None
                    or bool(value.get("trajectories")))
        return isinstance(value, int) and value > 1
    return False


def go_scope(function):
    @wraps(function)
    def invoke(args, **kwargs):
        path = getattr(args, "cyclone_products", None)
        if path is None or getattr(args, "branch_from", None) is not None or _ensemble_call(args):
            return function(args, **kwargs)
        from woof.go_cli import GoRefusal
        request = load_request(path, args.config)
        if getattr(args, "tiles", None) not in (None, 'off', 'auto'):
            raise GoRefusal("--cyclone-products requires --tiles off or auto: --tiles on streams every domain, and a streamed state cannot supply a complete every-step pressure/gust plane")
        if getattr(args, "prepare_only", False) or getattr(args, "readiness", False) or getattr(args, "no_probe", False):
            raise GoRefusal("--cyclone-products needs forecast steps; this stage-only command would produce none")
        print(f"go: cyclone products {', '.join(request['products'])}; d{request['grid_id']:02d}; every committed step on a fixed Earth grid")
        if getattr(args, "dry_run", False):
            return function(args, **kwargs)
        from woof import rustwx
        renderer = rustwx.find_renderer()
        if renderer is None:
            raise GoRefusal("--cyclone-products requires rw_wrfbatch; otherwise the requested diagnostic maps cannot be rendered")
        import subprocess
        try:
            probe = subprocess.run([str(renderer), '--tc-accumulator-abi'], capture_output=True,
                                   text=True, timeout=15, env=rustwx.renderer_env())
            compatible = probe.returncode == 0 and probe.stdout.strip() == 'gpuwm.tc-accumulator.v1'
        except (OSError, subprocess.TimeoutExpired):
            compatible = False
        if not compatible:
            raise GoRefusal("--cyclone-products rw_wrfbatch lacks the every-step accumulator contract; this binary could not render the requested swaths. Rebuild rw_wrfbatch from this source tree")
        from woof.ensemble.runtime_context import MemberOutputCapture, current_capture, member_output_scope
        if current_capture() is not None:
            raise GoRefusal("--cyclone-products cannot replace an active ensemble capture: doing so would lose the ensemble's diagnostics")
        capture = Capture(request)
        token = _ACTIVE.set(capture)
        try:
            with member_output_scope(MemberOutputCapture(capture, 0, keep_member_files=True,
                                                        initialize_callback=capture.initialize)):
                code = function(args, **kwargs)
                if not code:
                    capture.finish()
                return code
        finally:
            _ACTIVE.reset(token)
    return invoke


class Capture:
    def __init__(self, request):
        self.request, self.root, self.model = request, None, None
        self.common = None
        self.fixes = []
        self.diagnostic_seconds = 0.

    def initialize(self, *, model=None, **kwargs):
        if model is not None:
            self.model = model
            experiment = getattr(model, '_declared_experiment', None)
            if experiment is not None:
                self.request['analysis_time'] = utc_datetime(experiment.start_time).isoformat()

    def __call__(self, *, grid_id, valid_time=None, **kwargs):
        # Observe initialized/restored analysis at its normal history handoff.
        if self.model is not None and grid_id == self.request['grid_id']:
            node = next(n for n in self.model.walk_parent_first() if n.cfg.grid_id == grid_id)
            if valid_time is not None:
                from datetime import timedelta
                valid = utc_datetime(valid_time)
                self.request['analysis_time'] = (valid-timedelta(seconds=float(node.state.elapsed_seconds))).isoformat()
            self.committed_step(node=node, model=self.model)

    def committed_step(self, *, node, model):
        if node.cfg.grid_id != self.request['grid_id']:
            return
        if getattr(node.state, '_streamed_domain', None) is not None:
            raise RuntimeError(
                f"--cyclone-products: [tiles] mode 'auto' streamed d{node.cfg.grid_id:02d} on this card, "
                "and a streamed state holds only one tile, so it cannot supply the complete every-step "
                "pressure/gust plane; run with --tiles off or on a card where the domain fits resident")
        if self.common is None:
            from woof.ensemble.cyclone_products import CommonWind
            self.common = CommonWind(self.request)
            cp = self.common.cp
            self.gust = cp.zeros(self.common.lat.shape, cp.float32)
            self.pressure = cp.full(self.common.lat.shape, cp.inf, cp.float32)
            self.gust_visited = cp.zeros(self.common.lat.shape, cp.bool_)
            self.pressure_visited = cp.zeros(self.common.lat.shape, cp.bool_)
        common, cp = self.common, self.common.cp
        if common.last_seconds == float(node.state.elapsed_seconds):
            return
        started = time.perf_counter()
        common.observe(node=node, model=model)
        gust, pressure = diagnose(node.state)
        sampled_gust, sampled_pressure = common.sample(gust), common.sample(pressure)
        gust_valid = common.inside & cp.isfinite(sampled_gust)
        pressure_valid = common.inside & cp.isfinite(sampled_pressure)
        self.gust = cp.maximum(self.gust, cp.where(gust_valid, sampled_gust, 0))
        self.pressure = cp.minimum(self.pressure, cp.where(pressure_valid, sampled_pressure, cp.inf))
        self.gust_visited |= gust_valid
        self.pressure_visited |= pressure_valid
        # Only a six-number device row per step; no host synchronization in
        # the step path. Download and timestamp formatting happen at finish.
        from woof.io.history_layout import metadata_history_fields
        if getattr(self, 'native_key', None) != common.key:
            meta = metadata_history_fields(node.grid, model._prepared_by_grid_id[node.cfg.grid_id].static_fields)
            self.native_coordinates = tuple(cp.asarray(meta[k], dtype=cp.float32) for k in ('XLAT', 'XLONG'))
            self.native_key = common.key
        lat, lon = self.native_coordinates
        index = cp.argmin(cp.where(cp.isfinite(pressure), pressure, cp.inf))
        fields = node.state.physics.fields
        from woof.tc.intensity import wind_within_radius
        self.fixes.append(cp.stack((cp.asarray(float(node.state.elapsed_seconds)),
            lat.ravel()[index], lon.ravel()[index], pressure.ravel()[index],
            wind_within_radius(node.state, lat, lon, index), cp.max(gust))))
        self.diagnostic_seconds += time.perf_counter()-started

    def finish(self):
        if self.root is None or self.common is None or self.common.samples < 2:
            raise RuntimeError("--cyclone-products forecast ended without two committed samples on the requested grid")
        from datetime import timedelta
        from woof import rustwx
        cp, common = self.common.cp, self.common
        self.root.mkdir(parents=True, exist_ok=True)
        common_receipt = common.finish(self.root/'fields')
        for name, array, valid in (("time_max_gust", self.gust, self.gust_visited),
                                   ("time_min_mslp", self.pressure, self.pressure_visited)):
            cp.asnumpy(cp.where(valid, array, cp.nan).astype('<f4')).tofile(self.root/'fields'/f'{name}.f32')
        analysis = utc_datetime(self.request['analysis_time'])
        fixes = []
        for seconds, lat, lon, pressure, wind, gust in cp.asnumpy(cp.stack(self.fixes)):
            if not all(math.isfinite(float(v)) for v in (seconds, lat, lon, pressure, wind, gust)):
                raise RuntimeError("nonfinite every-step cyclone track; refusing a success receipt")
            fixes.append(dict(valid_time=(analysis+timedelta(seconds=float(seconds))).isoformat(),
                latitude=float(lat), longitude=float(lon), mslp_hpa=float(pressure),
                max_wind_m_s=float(wind), max_gust_m_s=float(gust)))
        receipt = dict(schema="gpuwm.tc-accumulator.v1", request=self.request,
            common_wind=common_receipt, fixes=fixes, out=str(self.root/'maps'),
            diagnostic_enqueue_seconds=self.diagnostic_seconds,
            sampling="every committed selected-domain step before relocation; initialized state when history starts at analysis",
            gust_method="rw-post surface-layer similarity (Panofsky/IFS), CUDA; diagnostic gust, not a resolved three-second observation",
            pressure_method="raw sea-level surface-pressure minimum over water; no smoothing; wind within 150 km of the field centre")
        path = self.root/'receipt.json'
        path.write_text(json.dumps(receipt, indent=2)+'\n')
        started = time.perf_counter()
        receipt['render'] = rustwx.run_cyclone_product_renderer(rustwx.find_renderer(), path)
        receipt['render_seconds'] = time.perf_counter()-started
        receipt['status'] = 'completed'
        path.write_text(json.dumps(receipt, indent=2)+'\n')
        print(f"go: every-step cyclone receipt: {path}")


def diagnose(state):
    """Launch rw-post-seeded surface-gust and pressure kernels on live columns."""
    import cupy as cp
    import numpy as np
    fields = state.physics.fields
    if not getattr(state.physics, 'surface_enabled', True):
        raise ValueError("every-step cyclone gust requires active surface physics; initialized surface carriers are not a gust diagnostic")
    required = ('u10', 'v10', 'ust', 'hfx', 'qfx', 't2', 'th2', 'q2', 'pblh', 'psfc')
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"every-step cyclone gust lacks surface physics carriers: {missing}")
    p = cp.ascontiguousarray(state.p, dtype=cp.float32)
    nz, ny, nx = p.shape
    phi = cp.ascontiguousarray(state.php + (state.phb[:, None, None] if state.phb.ndim == 1 else state.phb), dtype=cp.float32)
    theta = cp.ascontiguousarray(state.total_theta(), dtype=cp.float32)
    qv = cp.zeros_like(p) if state.qv is None else cp.ascontiguousarray(state.qv, dtype=cp.float32)
    gust, pressure = cp.empty((ny, nx), cp.float32), cp.empty((ny, nx), cp.float32)
    from woof.certify.kernel_manifest import record_module
    module = cp.RawModule(code=CUDA, options=())
    record_module('woof.tc_accumulator:fields', source=CUDA, options=(),
                  module=module)
    kernel = module.get_function('tc_fields')
    kernel(((nx*ny+255)//256,), (256,), (p, theta, qv, phi,
        *(cp.ascontiguousarray(fields[k], dtype=cp.float32) for k in required),
        np.int32(nx*ny), np.int32(nz), gust, pressure))
    from woof.tc.intensity import surface_pressure_hpa
    pressure = surface_pressure_hpa(state)
    return gust, pressure


# Seeded from this tree's rw-post group_e/column.rs (similarity gust) and
# wrf-core pressure reduction via core/mslp.py. CUDA, not host field processing.
CUDA = r'''
extern "C" __global__ void tc_fields(const float*p,const float*th,const float*q,const float*phi,
 const float*u,const float*v,const float*ust,const float*hfx,const float*qfx,const float*t2,
 const float*th2,const float*q2,const float*pbl,const float*psfc,int n,int nz,float*gust,float*slp){
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 const double ps=p[i], rd=287., cp=1004.5, eps=287./461.6, g=9.81, lapse=.0065, tc=290.66;
 int k=1; while(k<nz && ps-p[k*n+i]<10000.)++k;
 if(k==nz){slp[i]=__int_as_float(0x7fffffff);}else{
  int a=k*n+i,b=(k-1)*n+i;
  double pa=p[a],pb=p[b],ta=th[a]*pow(pa/100000.,rd/cp)*(1.+.608*fmax((double)q[a],0.));
  double tb=th[b]*pow(pb/100000.,rd/cp)*(1.+.608*fmax((double)q[b],0.));
  double za=.5*(phi[a]+phi[a+n])/9.80665,zb=.5*(phi[b]+phi[b+n])/9.80665;
  double f=(log(ps-10000.)-log(pb))/(log(pa)-log(pb));
  double tr=fabs(pa-pb)<1.?ta:tb+f*(ta-tb),zr=fabs(pa-pb)<1.?za:zb+f*(za-zb);
  double ts=tr*pow(ps/(ps-10000.),lapse*rd/g),tl=tr+lapse*zr;
  tl=(ts<=tc && tl>=tc)?tc:tc-.005*(ts-tc)*(ts-tc);
  slp[i]=.01*ps*exp(2.*g*(.5*(phi[i]+phi[n+i])/9.80665)/(rd*(tl+ts)));
 }
 if(!isfinite(u[i]) || !isfinite(v[i]) || !isfinite(ust[i]) || !isfinite(hfx[i])
    || !isfinite(qfx[i]) || !isfinite(t2[i]) || !isfinite(th2[i]) || !isfinite(q2[i])
    || !isfinite(psfc[i]) || t2[i]<=0. || th2[i]<=0. || psfc[i]<=0.){
   gust[i]=__int_as_float(0x7fffffff);return;
 }
 double r=q2[i],vf=(1.+r/eps)/(1.+r),rho=psfc[i]/(rd*t2[i]*vf);
 double wth=hfx[i]/(rho*cp),wq=qfx[i]/rho,c61=1./eps-1.;
 double flux=wth*(1.+c61*r/(1.+r))+c61*th2[i]*wq;
 double sig=fmax((double)ust[i],0.);
 if(flux>0. && pbl[i]>0.)sig=cbrt(sig*sig*sig+pbl[i]*.4*9.81*flux/(24.*th2[i]*vf));
 gust[i]=hypot((double)u[i],(double)v[i])+7.71*sig;
}
extern "C" __global__ void tc_smooth(const float*in,float*out,int nx,int ny){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=nx*ny)return;int x=i%nx,y=i/nx;
 double value=3.*in[i];for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
 int xx=max(0,min(nx-1,x+dx)), yy=max(0,min(ny-1,y+dy));value+=in[yy*nx+xx];}
 out[i]=value/12.;
}
'''
