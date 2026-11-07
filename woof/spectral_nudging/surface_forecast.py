"""Native auxinput9 surface FDDA on resident regional forecast states.

Native records are decoded by Rust once at setup. CUDA selects the current
record from the device clock. All column work and RK coupling stay on device.
Each domain/member owns its forcing, graph and restart-carried tendencies.
"""
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

import numpy as np

FASDAS_HELD = ("sda_hfx", "sda_qfx", "qnorm", "hfx_both", "qfx_both")

HELD = ("surface_fdda_u", "surface_fdda_v", "surface_fdda_theta", "surface_fdda_qv")


def upper_enabled(controls):
    return any(controls.coefficient(name) > 0 for name in
               ("u", "v", "temperature", "qv"))


def memory_shapes(cfg, controls, run_seconds):
    """Resident allocations made below, also consumed by admission pricing."""
    import math
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    volume, plane = (nz, ny+1, nx+1), (ny+1, nx+1)
    mass, u, v = (nz, ny, nx), (nz, ny, nx+1), (nz, ny+1, nx)
    window = min(run_seconds, controls.sgfdda_end_h*3600. +
                 (max(controls.dtramp_min, 0.)*60. if controls.if_ramping else 0.))
    records = max(1, math.ceil(window/(controls.sgfdda_interval_m*60.)))
    shapes = {f"surface_fdda.{name}": shape for name, shape in zip(
        (*HELD, "hfx_fdda"), (u, v, mass, mass, mass))}
    shapes.update({f"surface_fdda.field.{name}": volume for name in
                   ("u", "v", "theta", "temperature", "qv", "pressure", "z")})
    shapes.update({f"surface_fdda.field.{name}": plane for name in
                   ("ht", "regime", "znt", "odis_old", "odis_new", "sda_hfx", "sda_qfx")})
    shapes.update({"surface_fdda.old_analysis": (4, *plane),
                   "surface_fdda.new_analysis": (4, *plane),
                   "surface_fdda.wpbl": (4, *volume),
                   "surface_fdda.tendencies": (4, *volume),
                   "surface_fdda.time_factors": (2,),
                   "surface_fdda.wzfac": (4, nz),
                   "surface_fdda.analyses": (records, 2, 4, *plane),
                   "surface_fdda.densities": (records, 2, *plane),
                   "surface_fdda.operator_density": (4, *plane),
                   "surface_fdda.clock_and_controls": (2*records+5,)})
    for group in ("rates", "combined"):
        shapes.update({f"surface_fdda.{group}.{name}": shape for name, shape in
                       zip(("ru", "rv", "rtheta", "rqv", "rqc"), (u, v, mass, mass, mass))})
    if upper_enabled(controls):
        upper_window = min(run_seconds, (controls.gfdda_end_h or 0.)*3600. +
                           (max(controls.dtramp_min, 0.)*60. if controls.if_ramping else 0.))
        upper_records = max(1, math.ceil(upper_window/(controls.gfdda_interval_m*60.)))
        shapes["surface_fdda.upper_analyses"] = (upper_records, 2, 4, *volume)
        shapes["surface_fdda.upper_offsets"] = (2*upper_records,)
    if controls.grid_sfdda == 2:
        for name in (*FASDAS_HELD,"heat","moisture"):
            shapes[f"surface_fdda.{name}"]=(ny,nx)
    return shapes


def allocate(state, fasdas=False):
    import cupy as cp
    for name, template in zip((*HELD, "hfx_fdda"), (state.u, state.v, state.thp, state.qv, state.thp)):
        if getattr(state, name, None) is None:
            setattr(state, name, cp.zeros_like(template))
    if fasdas:
        for name in FASDAS_HELD:
            if getattr(state,name,None) is None:
                setattr(state,name,cp.zeros_like(state.mup))


class SurfaceForecast:
    def __init__(self, state, cfg, controls, path, start_time, run_seconds, *, upper_path=None):
        import cupy as cp
        from woof.netcdf_bridge import Dataset
        from woof.file_digest import sha256_file
        from woof.core.physics import PhysicsTendencies
        from woof.spectral_nudging.surface_gpu import SurfaceOperator, SurfaceCoefficients, _strict
        from woof.spectral_nudging.surface_inputs import read_native_surface
        from woof.wrf_exact import ENABLED as wrf_exact

        if controls.grid_sfdda not in (1,2):
            raise ValueError("SurfaceForecast requires grid_sfdda=1 or 2")
        self.fasdas = controls.grid_sfdda == 2
        self.carried = (*HELD,"hfx_fdda",*(FASDAS_HELD if self.fasdas else ()))
        if cfg.use_adaptive_time_step:
            raise ValueError("SURFACE_FDDA_ADAPTIVE_CLOCK: the graph requires a fixed model step")
        if not (cfg.specified or cfg.nested):
            raise ValueError("surface FDDA needs a regional specified root or nest")
        self.state, self.cfg, self.controls = state, cfg, controls
        self.path = Path(path)
        self.sha256 = sha256_file(self.path)
        with Dataset(self.path) as dataset:
            timestamps = dataset.stored_words(("Times",))["Times"]
        offsets = [(datetime.fromisoformat(row.tobytes().decode("ascii").rstrip("\x00 ").replace("_", "T"))
                    - start_time.replace(tzinfo=None)).total_seconds() for row in timestamps]
        interval = controls.sgfdda_interval_m*60.
        needed = min(run_seconds, controls.sgfdda_end_h*3600. +
                     (max(controls.dtramp_min, 0.)*60. if controls.if_ramping else 0.))
        if (not offsets or offsets[0] != 0 or any(abs(b-a-interval)>1.e-6 for a,b in zip(offsets,offsets[1:]))
                or offsets[-1]+interval < needed):
            raise ValueError("SURFACE_FDDA_ANALYSIS_COVERAGE: native OLD/NEW records must bracket the requested surface window from model start")
        selected = [i for i,offset in enumerate(offsets) if offset < max(needed, 1.)]
        offsets = [offsets[i] for i in selected]
        nz, ny, nx = state.thp.shape
        shape, surface = (nz,ny+1,nx+1), (ny+1,nx+1)
        self.fields = f = {name:cp.empty(shape,cp.float32) for name in
                          ("u","v","theta","temperature","qv","pressure","z")}
        f.update({name:cp.zeros(surface,cp.float32) for name in
                  ("ht","regime","znt","odis_old","odis_new","sda_hfx","sda_qfx")})
        f.update(old_analysis=cp.empty((4,*surface),cp.float32),
                 new_analysis=cp.empty((4,*surface),cp.float32),
                 wpbl=cp.empty((4,*shape),cp.float32),
                 tendencies=cp.zeros((4,*shape),cp.float32),time_factors=cp.zeros(2,cp.float32))
        weights=np.ones((4,nz),np.float32)
        for n, suffix in enumerate(("uv","uv","t","q")):
            if getattr(controls,"if_zfac_"+suffix):
                kz=getattr(controls,"k_zfac_"+suffix)
                weights[n,:min(kz,nz)]=0
                if kz<nz:weights[n,kz]=.1
        f["wzfac"]=cp.asarray(weights)
        self.operator=SurfaceOperator(SurfaceCoefficients(cfg.dx,controls.rinblw,
                    controls.guv_sfc,controls.gt_sfc,controls.gq_sfc),f,fasdas=self.fasdas)
        self.analyses=cp.empty((len(selected),2,4,*surface),cp.float32)
        self.densities=cp.empty((len(selected),2,*surface),cp.float32)
        for out, record in enumerate(selected):
            inputs=read_native_surface(self.path,record=record).fields
            for e,endpoint in enumerate(("OLD","NEW")):
                for n,name in enumerate(("U10","V10","TH2","Q2")):
                    value=inputs[f"{name}_NDG_{endpoint}"]
                    expected=(ny,nx+1) if n==0 else ((ny+1,nx) if n==1 else (ny,nx))
                    if value.shape != expected:
                        raise ValueError(f"surface FDDA {name} shape {value.shape} differs from domain {expected}")
                    padded=cp.pad(cp.asarray(value),((0,surface[0]-value.shape[0]),(0,surface[1]-value.shape[1])),mode="edge")
                    self.analyses[out,e,n]=padded
                f["odis_old" if e==0 else "odis_new"][...]=cp.pad(
                    cp.asarray(inputs[f"ODIS_NDG_{endpoint}"]),((0,1),(0,1)),mode="edge")
            self.operator.prepare_density()
            self.densities[out]=self.operator.density
        self.offsets=cp.asarray(offsets,dtype=cp.float64)
        self.upper_path = None
        self.upper_args = None
        allocate(state,self.fasdas)
        self.rates=PhysicsTendencies.zeros(state)
        self.combined=PhysicsTendencies.zeros(state)
        self.clock=cp.asarray([state.elapsed_seconds],dtype=cp.float64)
        self.control=cp.zeros(3,cp.int32)
        source=Path(__file__).with_suffix(".cu").read_text()
        options=("--std=c++17","--fmad=false") if _strict(None) else ("--std=c++17",)
        self.module=cp.RawModule(code=source,options=options)
        from woof.certify.kernel_manifest import record_module
        record_module("woof.spectral_nudging.surface_forecast",source=source,options=options,module=self.module)
        self.kernels={name:self.module.get_function("sf_"+name) for name in ("clock","select","pack","couple","add","diagnostic","upper","sda_commit","noah_inputs","qnorm")}
        self.grid,self.block=((surface[0]*surface[1]+127)//128,),(128,)
        self.volume_grid=((max(state.u.size,state.v.size)+255)//256,)
        self.ints=tuple(np.int32(v) for v in (nz,ny,nx,state.thb.ndim==3))
        p=state.physics.fields
        if self.fasdas:
            self.fasdas_fields={name:cp.empty_like(state.mup) for name in ("heat","moisture")}
            self.fasdas_fields["qnorm"]=state.qnorm
            self.sda_args=(self.control,f["sda_hfx"],f["sda_qfx"],state.sda_hfx,state.sda_qfx,np.int32(ny),np.int32(nx))
            if getattr(cfg,"sf_surface_physics",0)==2:
                from woof.core.noah import noah_category_ids
                _,isice=noah_category_ids(state.physics.noah_params)
                if bool(cp.any((p["ivgtyp"]==isice)&(p["xland"]<1.5)).item()):
                    raise ValueError("FASDAS_NATIVE_UNDEFINED_GLACIER_CAPACITY: WRF SFLX_GLACIAL never assigns HCPCT_FASDAS before its skin correction")
        # Only these native surface layers write REGIME. The native FDDA
        # driver diagnoses its PBL-depth fallback for all other layers.
        native_regime=cfg.sf_sfclay_physics in (1,5,7,91)
        self.pack_args=(state.u,state.v,state.thp,state.thb,state.p,state.p,state.qv,
                        state.php,state.phb,state.ht,p["pblh"],p.get("regime",p["pblh"]),p["znt"],
                        *(f[name] for name in ("u","v","theta","temperature","qv","pressure","z","ht","regime","znt","wpbl")),
                        *self.ints,np.int32(native_regime))
        self.diagnostic_args=(self.control,f["tendencies"],f["pressure"],f["theta"],
                              state.php,state.phb,state.hfx_fdda,*self.ints)
        self.couple_args=(self.control,f["tendencies"],state.mup,state.mub2d,state.c1h,state.c2h,
                          state.msfu,state.msfv,state.msft,*(getattr(state,name) for name in HELD),
                          self.rates.ru,self.rates.rv,self.rates.rtheta,self.rates.rqv,
                          *self.ints[:3],np.int32(state.has_msf),np.int32(wrf_exact))
        self.clock_args=(self.clock,f["time_factors"],self.control,self.offsets,np.int32(len(selected)),
                         np.float64(cfg.dt),np.float32(controls.sgfdda_interval_m),np.float32(controls.sgfdda_end_h),
                         np.float32(controls.gfdda_end_h or 0),np.int32(controls.if_ramping),
                         np.float32(controls.dtramp_min),np.int32(controls.cadence_steps(cfg.dt)),np.int32(controls.fgdtzero))
        if upper_path is not None:
            self.upper_path = Path(upper_path)
            self.upper_sha256 = sha256_file(self.upper_path)
            with Dataset(self.upper_path) as dataset:
                stamps = dataset.stored_words(("Times",))["Times"]
                upper_offsets = [(datetime.fromisoformat(row.tobytes().decode("ascii").rstrip("\x00 ").replace("_", "T"))
                                  - start_time.replace(tzinfo=None)).total_seconds() for row in stamps]
                upper_interval = controls.gfdda_interval_m*60.
                window = min(run_seconds, (controls.gfdda_end_h or 0.)*3600. +
                             (max(controls.dtramp_min, 0.)*60. if controls.if_ramping else 0.))
                if (not upper_offsets or upper_offsets[0] != 0 or
                        any(abs(b-a-upper_interval)>1.e-6 for a,b in zip(upper_offsets,upper_offsets[1:])) or
                        upper_offsets[-1]+upper_interval < window):
                    raise ValueError("GRID_FDDA_ANALYSIS_COVERAGE: native upper OLD/NEW records must bracket the grid nudging window")
                indices = [i for i,offset in enumerate(upper_offsets) if offset < max(window,1.)]
                self.upper_analyses = cp.empty((len(indices),2,4,*shape),cp.float32)
                names = tuple(f"{name}_NDG_{end}" for end in ("OLD","NEW") for name in ("U","V","T","Q"))
                for out,record in enumerate(indices):
                    inputs = dataset.stored_words(names,record=record)
                    for e,end in enumerate(("OLD","NEW")):
                        for n,name in enumerate(("U","V","T","Q")):
                            value = inputs[f"{name}_NDG_{end}"][0]
                            expected = (nz,ny,nx+1) if n==0 else ((nz,ny+1,nx) if n==1 else (nz,ny,nx))
                            if value.dtype != np.dtype("float32") or value.shape != expected:
                                raise ValueError(f"native grid FDDA {name} must have float32 C-grid shape {expected}")
                            self.upper_analyses[out,e,n] = cp.pad(cp.asarray(value),
                                ((0,0),(0,shape[1]-value.shape[1]),(0,shape[2]-value.shape[2])),mode="edge")
            self.upper_offsets = cp.asarray([upper_offsets[i] for i in indices],cp.float64)
            self.upper_args = (self.clock,self.upper_offsets,np.int32(len(indices)),self.upper_analyses,
                               *(f[name] for name in ("u","v","theta","qv","wpbl","wzfac","tendencies")),
                               *self.ints[:3],np.float64(cfg.dt),np.float32(controls.gfdda_interval_m),
                               np.float32(controls.gfdda_end_h or 0.),np.float32(controls.sgfdda_end_h),
                               np.int32(controls.if_ramping),np.float32(controls.dtramp_min),
                               np.float32(controls.coefficient("u")),np.float32(controls.coefficient("temperature")),
                               np.float32(controls.coefficient("qv")))
        cp.cuda.get_current_stream().synchronize()  # setup, before capture
        self.stream=cp.cuda.Stream(non_blocking=True)
        with self.stream:
            # Compile kernels without advancing the forecast's carried state.
            saved=[getattr(state,name).copy() for name in self.carried]
            self.record()
            self.stream.synchronize()
            self.clock.set(np.asarray([state.elapsed_seconds],np.float64))
            for name,value in zip(self.carried,saved):getattr(state,name)[...]=value
            self.stream.synchronize()
            self.stream.begin_capture()
            try:self.record()
            finally:self.graph=self.stream.end_capture()
        self.steps=0
        state._surface_fdda_owner=self

    def record(self):
        f=self.fields
        self.kernels["clock"]((1,),(1,),self.clock_args)
        self.kernels["select"](self.grid,self.block,(self.control,self.analyses,self.densities,
                        f["old_analysis"],f["new_analysis"],self.operator.density,np.int32(self.grid_size)))
        self.kernels["pack"](self.grid,self.block,self.pack_args)
        self.operator.record()
        if self.fasdas:
            self.kernels["sda_commit"](self.grid,self.block,self.sda_args)
        if self.upper_args is not None:
            self.kernels["upper"](self.grid,self.block,self.upper_args)
        self.kernels["diagnostic"](self.volume_grid,(256,),self.diagnostic_args)
        self.kernels["couple"](self.volume_grid,(256,),self.couple_args)

    @property
    def grid_size(self):
        return self.fields["ht"].size

    def fold(self,base):
        self.graph.launch()
        self.steps+=1
        if base is None:return self.rates
        a,b,o=base,self.rates,self.combined
        self.kernels["add"](self.volume_grid,(256,),(
            a.ru,a.rv,a.rtheta,a.rqv,b.ru,b.rv,b.rtheta,b.rqv,
            o.ru,o.rv,o.rtheta,o.rqv,np.int32(o.ru.size),np.int32(o.rv.size),np.int32(o.rtheta.size)))
        for field in dataclass_fields(base):
            if field.name not in ("ru","rv","rtheta","rqv"):
                setattr(o,field.name,getattr(base,field.name))
        o.extra_scalars=getattr(base,"extra_scalars",None)
        return o

    def update_qnorm(self, dqv, qv):
        """Native YSU moisture normalization, after its PBL call."""
        n=self.cfg.nx*self.cfg.ny
        self.kernels["qnorm"](((n+127)//128,),(128,),
            (dqv,qv,self.state.qnorm,np.float32(self.cfg.dt),np.int32(n)))

    def noah_inputs(self, atmosphere):
        f=self.state.physics.fields
        self.kernels["noah_inputs"](self.grid,self.block,(
            self.state.sda_hfx,self.state.sda_qfx,self.state.qnorm,
            atmosphere["rho"][0],f["cpm"],atmosphere["dz"][0],f["xland"],
            f["hfx"],f["qfx"],self.fasdas_fields["heat"],self.fasdas_fields["moisture"],
            self.state.hfx_both,self.state.qfx_both,np.int32(self.cfg.ny*self.cfg.nx)))
        return self.fasdas_fields

    def check_at_seam(self):
        # The only clock read is at an existing synchronized output seam.
        if not np.isfinite(self.clock.get()[0]):
            raise RuntimeError("surface FDDA device clock became nonfinite")

    def receipt(self):
        row = dict(enabled=True,config=self.controls.to_dict(),input=str(self.path),
                    input_sha256=self.sha256,records=len(self.offsets),steps=self.steps,
                    data_path="Rust numeric record read; CUDA graph columns and staggered RK coupling")
        if self.upper_path is not None:
            row["native_upper"] = dict(input=str(self.upper_path),input_sha256=self.upper_sha256,
                                       records=len(self.upper_offsets))
        return row

    def detach(self):
        self.state._surface_fdda_owner=None
