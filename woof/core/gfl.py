"""Grell-Freitas-Li, WRF v4.8.0's ``cu_physics=3``: the engine callable.

WRF v4.8.0 deleted the old Grell-Freitas driver (module_cu_gf_wrfdrv.F, GFDRV)
and points ``cu_physics=3`` (now ``GFLSCHEME``) at GFL, the NOAA GSL Grell-Freitas-
Li scheme (phys/GFL submodule 873b414). This adapter runs it when
``cu_gf_version = "gfl_873b414"``; ``"wrf_461"`` keeps the previous scheme
(:class:`woof.core.gf.GrellFreitas`, WRF v4.6.1/v4.7.1 GFDRV) byte for byte.

The scheme is ``woof.core.gfl_873b414``: a generated, complete CUDA
transcription of module_cu_gfl_deep, module_cu_gfl_sh and the WRF driver
module_cu_gfl (WRF_CHEM=0), lowered from the GNU trees of the WRF v4.8.0
strict referee's own compilation units (tools/gfl_873b414). The deep and
shallow bodies are the same lowered code woof hex runs for RRFS GFL bf6a4b0.
One thread runs one column with WRF's own bounds (kms=1, kme=nz+1, kts=1,
kte=nz).

What the WRF v4.8.0 cumulus driver hands the routine, and what this seam
passes (module_cumulus_driver.F:1274-1296, first_rk_step_part1.F:1562-1700):
  * U, V, T, QV, P (p_hyd), PI, RHO, DZ8W at mass levels; W (w_2) and P8W
    (p_hyd_w) at the nz+1 faces. The top mass-level word of the kme
    allocation is never read by the routine; this seam passes 0 there.
  * RTHFTEN/RQVFTEN, the dycore's pure advective rates, and RTHRATEN,
    RTHBLTEN, RQVBLTEN: the same lanes, the same WRF-exact mass decoupling and
    the same moist-theta conversion as the GF adapter (GFL sums them itself,
    module_cu_gfl_wrf.F90:311-312, so the export stays pure advection).
  * HT, XLAND, HFX, QFX, GSW (GFL copies GSW into a local it never reads), KPBL.
  * DX (one value; ``gf_dx_column`` from the driver gives one per column), DT
    (the model clock dt), XLV, CP, G, R_V (module_model_constants),
    ICHOICE_DEEP = clos_choice, ISHALLOW_G3 = ishallow.
Outputs: RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN, RUCUTEN, RVCUTEN, RAINCV
(pratec * dt), PRATEC, HTOP, HBOT, GD_CLOUD/GD_CLOUD2 and the shallow k22/kbcon/
ktop/xmb words. Unlike GFDRV, GFL has no four-cell edge band and convects in
every column of the tile, so no edge mask is applied.

Not carried by WRF v4.8.0's GFL:
  * SPP (``spp_conv``): the GFL call passes no pattern. validate_spp_config
    refuses spp_conv unless cu_gf_version = "wrf_461" (binding SPP never
    switches the scheme, so an ensemble's members all run one scheme).
  * PR92 lightning from the cumulus cloud top (``lightning_option = 11``):
    GFL never writes KTOP_DEEP, so WRF v4.8.0 reads a level that stays 0
    there (an upstream defect, module_cumulus_driver.F:1274-1296);
    ``RunConfig.__post_init__`` keeps such a run on ``wrf_461``.
"""

from __future__ import annotations

import numpy as np

from woof.core.gf import _active_index
from woof.core.kf import _model_clock_dt

DTYPE = np.float32
VERSION = "gfl_873b414"

#: Level inputs of the routine, in the generated FIELDS order's terms.
_LEVEL_IN = ("u", "v", "w", "t", "q", "rho", "p", "pi", "p8w", "dz8w",
             "rqvften", "rthften", "rthblten", "rqvblten", "rthraten")
_LEVEL_OUT = ("gdc", "gdc2", "rthcuten", "rqvcuten", "rqccuten", "rqicuten",
              "rucuten", "rvcuten")
_SURF_IN = ("ht", "xland", "hfx", "qfx", "gsw")
_SURF_OUT = ("htop", "hbot", "xmb_shallow", "raincv", "pratec")
_INT_OUT = ("k22_shallow", "kbcon_shallow", "ktop_shallow")


class GrellFreitasLi:
    """Stateless ``cu_physics=3`` (WRF v4.8.0 GFL) callable for :class:`PhysicsDriver`.

    The driver remains the cadence authority; each due call returns fresh
    rates and the RAINCV the routine computed (pratec * dt), as for GF.
    """

    _active_cache = None

    #: The restart manifest's identity for this callable (woof/io/restart.py
    #: _callable_setup_identity): the scheme and its pins.
    restart_identity = {"scheme": "cu_grell_freitas_li", "cu_gf_version": VERSION,
                        "wrf": "06d4240ae989", "gfl": "873b414670323ae766ec4fe1f4b6b5ab891358ed"}

    def __init__(self):
        self._driver = None
        self._scalars = {}

    def _scalar(self, name, value, dtype):
        """A one-word device argument kept for the run; a new value is written on the device (no host copy)."""
        import cupy as cp
        held = self._scalars.get(name)
        if held is None or held[1] != value or held[0].dtype != dtype:
            word = cp.empty(1, dtype=dtype) if held is None or held[0].dtype != dtype else held[0]
            word.fill(value)
            held = self._scalars[name] = (word, value)
        return held[0]

    def bind_driver(self, driver) -> None:
        """Optional driver hook: the held radiation and forcing lanes (as GF)."""
        self._driver = driver

    def release(self) -> int:
        """Drop the driver back-reference (the GF adapter's cycle); owns no workspace."""
        self._driver = None
        return 0

    def _lanes(self, *, state, cfg, nz, ny, nx, cols, zeros_lev):
        """The forcing lanes, exactly as :class:`woof.core.gf.GrellFreitas` builds them."""
        import cupy as cp
        from woof.wrf_exact import ENABLED as WRF_EXACT
        forcing_mass = None

        def mass(source):
            # The first lane to need it fixes the coupling mass, as in GF: the
            # advective lane reads the carried MUTS, the moist-theta vapour lane
            # the step's WRF_MUTS scratch (gf.py, GrellFreitas.__call__).
            nonlocal forcing_mass
            if forcing_mass is None:
                if source == "carried":
                    from woof.state_serialization_contract import carried_wrf_muts
                    mut = carried_wrf_muts(state)
                else:
                    from woof.state_serialization_contract import WRF_MUTS_SLOT
                    mut = state.existing_scratch(WRF_MUTS_SLOT)
                if mut is None:
                    mut = state.total_mu()
                forcing_mass = (state.c1h[:, None, None] * mut[None]
                                + state.c2h[:, None, None])
            return forcing_mass

        def held_lane(name):
            lane = (None if self._driver is None
                    else getattr(self._driver, name, None))
            if lane is None:
                return zeros_lev
            if lane.shape != (nz, ny, nx):
                raise ValueError(
                    f"driver {name} must be [nz, ny, nx]={nz, ny, nx}, "
                    f"got {lane.shape}")
            owned = {"gf_rthdynten": "rthften", "gf_rqvdynten": "rqvften"}
            if (WRF_EXACT and name in owned
                    and lane is getattr(state, owned[name], None)
                    and not (name == "gf_rthdynten" and cfg.use_theta_m)):
                lane = lane / mass("carried")
            if (name == "gf_rthdynten" and getattr(state, "use_theta_m", 0)
                    and lane is not getattr(state, "rthften", None)):
                from woof.core.theta_m_physics import dry_cumulus_theta_tendency
                vapour = getattr(self._driver, "gf_rqvdynten", None)
                if vapour is None:
                    vapour = cp.zeros_like(lane)
                elif WRF_EXACT and vapour is getattr(state, "rqvften", None):
                    vapour = vapour / mass("scratch")
                lane = lane.copy()
                dry_cumulus_theta_tendency(state, lane, vapour)
            return cols(lane)

        return {"rthften": held_lane("gf_rthdynten"), "rqvften": held_lane("gf_rqvdynten"),
                "rthblten": held_lane("gf_rthblten"), "rqvblten": held_lane("gf_rqvblten")}

    def __call__(self, *, atmosphere, fields, state, cfg,
                 spp_conv: int | None = None, pattern_spp_conv=None):
        import cupy as cp
        from woof.core import gfl_873b414 as gfl
        from woof.core.gf import _native_result_type

        flag = getattr(cfg, "spp_conv", 0) if spp_conv is None else spp_conv
        if flag:
            raise ValueError(
                "spp_conv with cu_gf_version='gfl_873b414': WRF v4.8.0's GFL call carries no SPP "
                "pattern, so the perturbation would be silently dropped; select cu_gf_version='wrf_461'")
        nz, ny, nx = state.p.shape
        ncol = ny * nx
        active = (None if self._driver is None
                  else getattr(self._driver, "gf_active_column", None))
        index = None if active is None else _active_index(self, active, ny, nx)
        n = ncol if index is None else int(index.size)

        def cols(a):
            # A level source stays in WRF's [k, ny, nx] order; the fused pack reads it there.
            return a

        def flat(a, dtype=DTYPE):
            view = cp.asarray(a, dtype=dtype).reshape(ncol)
            return view if index is None else view[index]

        kme = nz + 1
        # Level fields in the kernel's lane placement ([slab][level][lane], gfl_lanes); the
        # top mass-level word, absent lanes and the pad columns stay 0.
        level = cp.zeros((len(_LEVEL_IN) + len(_LEVEL_OUT),) + gfl.lane_shape(n, kme), dtype=DTYPE)
        f = {name: level[i] for i, name in enumerate(_LEVEL_IN + _LEVEL_OUT)}
        zeros_lev = _ABSENT
        sources = {}
        if self._driver is not None:
            total = getattr(self._driver, "radiation_theta_forcing", None)
            rthraten = (total() if total is not None else
                        self._driver.rthratenlw + self._driver.rthratensw)
        else:
            rthraten = cp.zeros((nz, ny, nx), dtype=DTYPE)
        sources["u"] = atmosphere["u"]
        sources["v"] = atmosphere["v"]
        sources["w"] = state.w                              # w_2 at the nz+1 faces
        sources["t"] = atmosphere["temperature"]
        sources["q"] = atmosphere["qv"]
        sources["rho"] = atmosphere["rho"]
        sources["p"] = atmosphere["pressure"]
        sources["pi"] = atmosphere["exner"]
        sources["p8w"] = atmosphere["p_interface"]          # p_hyd_w at the nz+1 faces
        sources["dz8w"] = atmosphere["dz"]
        sources["rthraten"] = rthraten
        for name, lane in self._lanes(state=state, cfg=cfg, nz=nz, ny=ny, nx=nx,
                                      cols=cols, zeros_lev=zeros_lev).items():
            if lane is not _ABSENT:
                sources[name] = lane
        _pack_lanes(level, _LEVEL_IN, sources, ncol=ncol, n=n, index=index, nz=nz)
        surf = cp.zeros((len(_SURF_IN) + len(_SURF_OUT), n), dtype=DTYPE)
        for i, name in enumerate(_SURF_IN + _SURF_OUT):
            f[name] = surf[i]
        f["ht"][:] = flat(state.ht)
        f["xland"][:] = flat(fields["xland"])
        f["hfx"][:] = flat(fields["hfx"])
        f["qfx"][:] = flat(fields["qfx"])
        gsw = fields.get("gsw") if hasattr(fields, "get") else None
        if gsw is not None:
            f["gsw"][:] = flat(gsw)
        ints = cp.zeros((1 + len(_INT_OUT), n), dtype=cp.int32)
        f["kpbl"] = ints[0]
        f["kpbl"][:] = flat(fields["kpbl"], cp.int32)
        for i, name in enumerate(_INT_OUT):
            f[name] = ints[1 + i]
        dx_column = (None if self._driver is None
                     else getattr(self._driver, "gf_dx_column", None))
        if dx_column is None:
            f["dxcell"] = cp.asarray([cfg.dx], dtype=DTYPE)
        else:
            if dx_column.shape != (ny, nx):
                raise ValueError(f"driver gf_dx_column must be [ny, nx]={ny, nx}, got {dx_column.shape}")
            f["dxcell"] = cp.ascontiguousarray(flat(dx_column))
        from woof.core.constants import CP, G, RV, XLV
        scalars = {"itimestep": (_itimestep(state, cfg), cp.int32),
                   "dt": (_model_clock_dt(cfg), DTYPE), "xlv": (XLV, DTYPE), "cp": (CP, DTYPE),
                   "g": (G, DTYPE), "r_v": (RV, DTYPE),
                   "ichoice_deep": (int(cfg.clos_choice), cp.int32),
                   "ishallow_g3": (int(cfg.ishallow), cp.int32)}
        for name, (value, dtype) in scalars.items():
            f[name] = self._scalar(name, value, dtype)
        gfl.launch_lane_columns(f, ncol=n, nz=nz)

        # Every level output back to WRF's [nz, ny, nx] in one fused unpack (inactive columns 0).
        lev = _unpack_lanes(level, len(_LEVEL_IN), _LEVEL_OUT, ncol=ncol, n=n, index=index,
                            nz=nz, ny=ny, nx=nx)
        sca = {name: f[name] for name in _SURF_OUT}

        def grid(values, dtype=DTYPE):
            return values

        def plane(values):
            if index is None:
                return cp.ascontiguousarray(values.reshape(ny, nx))
            full = cp.zeros(ncol, dtype=values.dtype)
            full[index] = values
            return full.reshape(ny, nx)

        rthcuten = grid(lev["rthcuten"])
        rqvcuten = grid(lev["rqvcuten"])
        rqc = grid(lev["rqccuten"])
        rqi = grid(lev["rqicuten"])
        history_rates = {"rqicuten": rqi, "rucuten": grid(lev["rucuten"]),
                         "rvcuten": grid(lev["rvcuten"])}
        if getattr(state, "qi", None) is None:
            from woof.wrf_exact import ENABLED
            # As for GF: stock cumulus_driver passes RQICUTEN even without prognostic
            # ice; addtendc leaves that cold slot unused under F_QI.
            if not ENABLED:
                rqc = cp.ascontiguousarray(rqc + rqi)
            rqi = None
        rainc = plane(sca["raincv"])
        if cfg.cu_rad_feedback:
            from woof.core.g3 import run_g3_diag
            from woof.core.physics_inventory import cumulus_state_layout
            for key, (field_shape, dtype) in cumulus_state_layout(cfg).items():
                if key not in fields:
                    fields[key] = cp.zeros(field_shape, dtype=dtype)
            averages = {key: fields[key] for key in
                        ('gd_cloud_a', 'qc_cu', 'gd_cloud2_a', 'qi_cu', 'gd_cldfr', 'raincv_a', 'raincv_b')}
            averages['clock'] = fields['g3_diag_clock']
            run_g3_diag(averages, cloud=grid(lev["gdc"]), ice=grid(lev["gdc2"]), rain=rainc,
                        temperature=atmosphere['temperature'], pressure=atmosphere['pressure'],
                        qv=atmosphere['qv'], qc=atmosphere['qc'], qi=atmosphere['qi'],
                        fqi=int(getattr(state, 'qi', None) is not None),
                        dt=float(_model_clock_dt(cfg)), curr_secs=float(state.elapsed_seconds),
                        avg_minutes=float(cfg.convtrans_avglen_m), adaptive=cfg.use_adaptive_time_step)
        rrtmg = 99 in (cfg.ra_lw_physics, cfg.ra_sw_physics)
        return _native_result_type()(
            owner=self,
            rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqc,
            rqicuten=rqi, rainc=rainc, history_rates=history_rates,
            # GFL's bounds carry the same .001 level marker GFDRV's do.
            cloud_base=plane(sca["hbot"]) if rrtmg else None,
            cloud_top=plane(sca["htop"]) if rrtmg else None)


#: A forcing lane the driver does not hold: its lane-placement words stay 0.
_ABSENT = object()

_PACK = r"""
extern "C" __global__ void gfl_pack_lanes(const unsigned long long* sources, const int* source_levels,
    float* store, const long long* index, long long ncol, long long n, int kme, int fields, long long total){
  // store[f][slab][k][lane] <- source_f[k][column]: a level source is WRF's [k, ny, nx], column =
  // index[slab*128+lane] (the active-column list) or slab*128+lane. Exact word copies; words a
  // source does not have (its levels, absent sources, pad columns) are left as they are (0).
  const long long slabs=(n+127)/128;
  for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=(long long)gridDim.x*blockDim.x){
    const long long lane=i%128, k=(i/128)%kme, slab=i/(128LL*kme);
    const long long f=slab/slabs, c=(slab%slabs)*128+lane;
    const float* source=(const float*)sources[f];
    if(source==nullptr||c>=n||k>=source_levels[f])continue;
    const long long column=index?index[c]:c;
    store[i]=source[k*ncol+column];
  }
}
extern "C" __global__ void gfl_unpack_lanes(const float* store, float* grid, const long long* index,
    long long ncol, long long n, int kme, int nz, int fields, long long total){
  // grid[f][k][column] <- store[f][slab][k][lane] for k < nz (the routine's kts:kte words).
  for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=(long long)gridDim.x*blockDim.x){
    const long long c=i%n, k=(i/n)%nz, f=i/(n*(long long)nz);
    const long long slabs=(n+127)/128;
    const long long column=index?index[c]:c;
    grid[(f*nz+k)*ncol+column]=store[((f*slabs+c/128)*kme+k)*128+c%128];
  }
}
"""


#: The lane pack/unpack copies compile with CuPy's defaults, stated so the
#: kernel manifest records exactly what they compile (copies, no arithmetic).
_PACK_OPTIONS = ()


def _kernels():
    import cupy as cp
    device = cp.cuda.runtime.getDevice()
    held = _KERNELS.get(device)
    if held is None:
        from woof.certify.kernel_manifest import record_module
        module = cp.RawModule(code=_PACK, options=_PACK_OPTIONS)
        record_module("woof.core.gfl:pack_lanes", source=_PACK,
                      options=_PACK_OPTIONS, module=module)
        held = _KERNELS[device] = (module.get_function("gfl_pack_lanes"),
                                   module.get_function("gfl_unpack_lanes"))
    return held


_KERNELS = {}
_TABLES = {}


def _device_table(key, dtype):
    """A small device table (source pointers, levels), uploaded once per distinct key."""
    import cupy as cp
    held = _TABLES.get((key, dtype))
    if held is None:
        if len(_TABLES) > 64:
            _TABLES.clear()
        held = _TABLES[(key, dtype)] = cp.asarray(list(key), dtype=dtype)
    return held


def _pack_lanes(store, names, sources, *, ncol, n, index, nz):
    """Every level input into the lane placement in one launch, gathering active columns."""
    import cupy as cp
    import numpy as np
    kme = store.shape[2]
    pointers, levels, keep = [], [], []
    for name in names:
        source = sources.get(name)
        if source is None:
            pointers.append(0); levels.append(0); continue
        source = cp.ascontiguousarray(source, dtype=DTYPE)
        if source.size % ncol or source.size // ncol > kme:
            raise ValueError(f"GFL level source {name} must be [k <= {kme}, {ncol}] columns, got {source.shape}")
        keep.append(source)
        pointers.append(source.data.ptr); levels.append(source.size // ncol)
    total = len(names) * store.shape[1] * kme * 128
    pack, _ = _kernels()
    pack((min(65535, -(-total // 256)),), (256,),
         (_device_table(tuple(pointers), cp.uint64), _device_table(tuple(levels), cp.int32), store,
          index if index is not None else np.uint64(0), np.int64(ncol), np.int64(n), np.int32(kme),
          np.int32(len(names)), np.int64(total)))
    return keep


def _unpack_lanes(store, first, names, *, ncol, n, index, nz, ny, nx):
    """The level outputs as WRF [nz, ny, nx] grids in one launch (inactive columns 0)."""
    import cupy as cp
    import numpy as np
    grids = (cp.zeros if index is not None else cp.empty)((len(names), nz, ny, nx), dtype=DTYPE)
    total = len(names) * nz * n
    _, unpack = _kernels()
    if total:
        unpack((min(65535, -(-total // 256)),), (256,),
               (store[first:], grids, index if index is not None else np.uint64(0), np.int64(ncol), np.int64(n),
                np.int32(store.shape[2]), np.int32(nz), np.int32(len(names)), np.int64(total)))
    return {name: grids[i] for i, name in enumerate(names)}


def _itimestep(state, cfg) -> int:
    """WRF's grid%itimestep for this call: completed steps + 1 (the step being taken).

    GFL reads it only for total_time = itimestep*dt, the cold-start blend of
    f_thresh, which the WRF driver then sets to 0 (module_cu_gfl_wrf.F90:268-271).
    """
    dt = float(_model_clock_dt(cfg))
    return int(round(float(getattr(state, "elapsed_seconds", 0.0)) / dt)) + 1 if dt > 0 else 1
