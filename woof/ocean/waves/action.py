"""Fixed-interval GPU action-balance driver implementing CONTRACT.md.

Python binds CUDA kernels, retains buffers and schedules explicit clocks only.
No device-to-host scalar, conversion or synchronization occurs in step().
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import time
import numpy as np
from .model import SpectralGrid

PACKAGE=Path(__file__).parent
CONTRACT=json.loads((PACKAGE/"spectral_contract.json").read_text())
PRODUCTS=tuple(CONTRACT["products"])+tuple(CONTRACT["extra_device_outputs"])


class ActionSolver:
    def __init__(self, grid, bathymetry, config):
        import cupy as cp
        from .backends import StructuredBackend
        self.cp,self.config=cp,dict(config)
        self.spectral=SpectralGrid(**config.get("spectral",{}))
        self.nf,self.nd=self.spectral.frequencies,self.spectral.directions
        self.source_pack=config.get("source_pack","st4")
        packs=json.loads((PACKAGE/"source_packs.json").read_text())
        if self.source_pack not in ("st4","janssen"):
            raise ValueError("source_pack must be st4 or janssen")
        self.pack_id=0 if self.source_pack=="st4" else 1
        self.source_parameter_sha256=hashlib.sha256((PACKAGE/"source_packs.json").read_bytes()).hexdigest()
        self.backend=config.get("grid_backend")
        if self.backend is None:
            kind=config.get("backend","latlon")
            if kind=="voronoi":
                raise ValueError("voronoi requires its constructed config['grid_backend']")
            self.backend=StructuredBackend(grid,bathymetry,kind=kind,map_factor=config.get("map_factor"),
                gse_spreading_seconds=config.get("gse_spreading_seconds",3600.),spectral=config.get("spectral",{}))
        b=self.backend
        self.shape=tuple(b.full_shape)
        self.nfull=math.prod(self.shape);self.nc=b.ncell;self.ns=self.nf*self.nd
        self.state=cp.zeros((self.nc,self.nf,self.nd),cp.float32)
        self.base_depth=b.depth.copy()
        self.water_level=cp.zeros(self.nc,cp.float32)
        self.previous_depth=b.depth.copy();self.depth_rate=cp.zeros(self.nc,cp.float32)
        self.frequency=self.spectral.first_hz*cp.power(cp.float32(self.spectral.ratio),cp.arange(self.nf,dtype=cp.float32))
        self.df=self.frequency*cp.float32(math.sqrt(self.spectral.ratio)-1/math.sqrt(self.spectral.ratio))
        self.nfx=self.nf+math.ceil(math.log(1.25/.75)/math.log(self.spectral.ratio))+2
        if self.nc*self.nfx*self.nd>=2**31:
            raise ValueError("DIA expanded array exceeds the inherited int32 kernel index range")
        self.frequency_ext=self.spectral.first_hz*cp.power(cp.float32(self.spectral.ratio),cp.arange(self.nfx,dtype=cp.float32))
        self.df_ext=self.frequency_ext*cp.float32(math.sqrt(self.spectral.ratio)-1/math.sqrt(self.spectral.ratio))
        self.k=cp.empty((self.nc,self.nf),cp.float32);self.cg=cp.empty_like(self.k)
        self.fdot=cp.zeros_like(self.k)
        if hasattr(b,"bind_workspace"):
            self._transport_workspace=tuple(cp.empty_like(self.state) for _ in range(3))
            b.bind_workspace(*self._transport_workspace)
            self.turn=self._transport_workspace[0]
        else:
            self._transport_workspace=()
            self.turn=cp.empty_like(self.state)
        self.wind_beta=self.turn  # Transport scratch is free throughout source/diagnostic stages.
        self.u=cp.zeros(self.nc,cp.float32);self.v=cp.zeros_like(self.u)
        self.rho=cp.full(self.nc,1.225,cp.float32)
        self._wind_endpoints=cp.zeros((4,self.nfull),cp.float32)
        self._wind_times=cp.asarray([0.,1.],dtype=cp.float64);self._wind_bracket=False
        self.moments=cp.zeros((self.nc,16),cp.float32)
        self.moments[:,7]=.1;self.moments[:,8]=1e-4
        self.shelter=cp.zeros((self.nc,self.nf,2),cp.float32)
        # Source and partition work is cell-local. Reuse one bounded arena
        # instead of retaining a full-domain spectrum for every intermediate.
        self.batch_cells=min(self.nc,int(config.get("source_batch_cells",1024)))
        if self.batch_cells<1:raise ValueError("source_batch_cells must be positive")
        bins=self.batch_cells*self.ns;extended=self.batch_cells*self.nfx*self.nd
        self._source_workspace=cp.empty(3*bins+3*extended,cp.float32)
        self.beta=self._source_workspace[:bins].reshape(self.batch_cells,self.nf,self.nd)
        self.loss=self._source_workspace[bins:2*bins].reshape(self.beta.shape)
        self.seed=self._source_workspace[2*bins:3*bins].reshape(self.beta.shape)
        self.dia_energy=self._source_workspace[3*bins:3*bins+extended].reshape(self.batch_cells,self.nfx,self.nd)
        self.dia_rate=self._source_workspace[3*bins+extended:3*bins+2*extended].reshape(self.dia_energy.shape)
        self.dia_diagonal=self._source_workspace[3*bins+2*extended:].reshape(self.dia_energy.shape)
        self.saturation=self.dia_energy.ravel()[:bins].reshape(self.beta.shape)
        self.wind_input_mask=cp.zeros((self.nc,(self.ns+31)//32),cp.uint32)
        self.source_limiter=cp.ones(self.nc,cp.float32)
        self.depth_breaking=cp.empty(self.nc,cp.float32)
        self.energy=self.state  # Reconstruction followed by in-place E -> N.
        self.legacy_moments=cp.zeros((self.batch_cells,10),cp.float32)
        self.smooth=self.beta
        self.labels=self.loss.view(cp.int32);self.labels_work=self.seed.view(cp.int32)
        self.basin_variance=self.saturation
        self.basin_wind=self.dia_rate.ravel()[:bins].reshape(self.beta.shape)
        self.partition_workspace=cp.empty(7*bins if self.ns>1600 else 0,cp.float32)
        self.selected=cp.empty((self.nc,3),cp.int32)
        self.bulk=cp.zeros((len(PRODUCTS),self.nfull),cp.float32)
        self._outputs=SimpleNamespace(**{key:self.bulk[j].reshape(self.shape) for j,key in enumerate(PRODUCTS)})
        self._budget=None
        self._boundary=None
        self.boundary_action=cp.zeros((b.boundary_cells.size,self.nf,self.nd),cp.float32)
        self.boundary_e0=cp.zeros_like(self.boundary_action);self.boundary_e1=cp.zeros_like(self.boundary_action)
        self.boundary_times=cp.asarray([0.,1.],dtype=cp.float64)
        self._station_full=cp.empty(0,cp.int32);self._station_compact=cp.empty(0,cp.int32)
        self._station_spectra=cp.empty((0,self.nf,self.nd),cp.float32)
        self._station_bulk=cp.empty((len(PRODUCTS),0),cp.float32)
        self.errors=cp.zeros(1,cp.int32)
        self._forcing=None;self._initialized=False
        self.time_seconds=float(config.get("time_seconds",0))
        self.device_clock=cp.asarray([self.time_seconds],dtype=cp.float64)
        self.device_steps=cp.zeros(1,cp.int64)
        shapes=PACKAGE/"partition_shapes.v1.json"
        default_shape_hash=(hashlib.sha256(json.dumps(json.loads(shapes.read_text()),sort_keys=True,
            separators=(",",":")).encode()).hexdigest() if shapes.is_file() else "none")
        self.partition_shapes_sha256=config.get("partition_shapes_sha256",default_shape_hash)
        self._required_shape_hash=config.get("partition_shapes_sha256")
        self.stage_wall={};self.gpu_seconds={};self.steps=0
        self.stage_names=("sources_first_half","propagation","sources_second_half","diagnostics")
        self.device_timing=cp.zeros((4,2),cp.uint64)
        # All source comes from this repository. Compile and register the full
        # module including the inherited conservative DIA and implicit turning.
        source="\n".join((PACKAGE/name).read_text() for name in ("kernels.cu","physics.cu","diagnostics.cu","budget.cu"))
        options=("--std=c++17","--use_fast_math",f"-DNF={self.nf}",f"-DND={self.nd}",
                 f"-DWAVES_BACKEND_HANDLES_TURN={int(bool(getattr(b,'turning_handled',False)))}",
                 f"-DWAVES_DIRECTIONS={self.nd}",f"-DDIA_COEFFICIENT={packs[self.source_pack]['dia_coefficient']}f")
        self.module=cp.RawModule(code=source,options=options)
        from woof.certify.kernel_manifest import record_module
        record_module(f"woof.ocean.waves:action:{self.source_pack}:{self.nf}:{self.nd}:{b.name}",
                      source=source,options=options,module=self.module)
        names=("wave_geometry","wave_gather","wave_water_level","wave_depth_rate","wave_convert",
               "wave_boundary","wave_frequency","wave_moments","wave_wind","wave_wind_fused","wave_shelter",
               "wave_saturation","wave_depth_breaking","wave_breaking_weight","wave_source_rates","wave_sources","wave_reconstruct",
               "wave_smooth","wave_watershed","wave_label_jump","wave_basin_energy",
               "wave_choose_basins","wave_merge_basins","wave_diagnose","wave_station_spectra","wave_station_bulk",
               "wave_dia_moments","wave_zero","wave_copy","wave_clock","wave_timer","wave_forcing_bracket",
               "wave_tail","wave_extend_dia","wave_contract_dia","refract","wave_dia_rates",
               "wave_budget_capture","wave_budget_sources","wave_turn","wave_refract","wave_refract_geometry","wave_wind_mask","wave_limiter_reset","wave_copy_group")
        self.kernels={name:self.module.get_function(name) for name in names}
        # All views are constructed once, before graph capture. No per-step
        # device allocations or host scalar reads are needed for tiling.
        self._batches=[]
        for first in range(0,self.nc,self.batch_cells):
            stop=min(first+self.batch_cells,self.nc)
            arrays={name:getattr(self,name)[first:stop] for name in
                ("state","k","cg","u","v","rho","moments","shelter","wind_input_mask","source_limiter","selected","wind_beta","depth_breaking")}
            arrays.update(depth=b.depth[first:stop],wet=b.wet_to_full[first:stop],nc=stop-first,first=first)
            self._batches.append(SimpleNamespace(**arrays))
        # At fixed intrinsic sigma, cg=(g/(2*sigma))*(tanh x+x*sech(x)^2).
        # Its maximum has x*tanh(x)=1 and cg<0.6*g/sigma for every depth.
        # Use that bound rather than the phase-speed bound g/sigma.
        self.transport_step=b.max_dt(.6*9.80665/(2*math.pi*self.spectral.first_hz))
        if not math.isfinite(self.transport_step) or self.transport_step<=0:
            raise ValueError("backend returned invalid geographic CFL")
        self.transport_groups=(b.frequency_steps(self.spectral.first_hz,self.spectral.ratio,self.nf)
            if hasattr(b,"frequency_steps") and hasattr(b,"advect_group") else ((0,self.nf,self.transport_step),))
        self._geometry()
        # Allocate/compile backend scratch on an all-zero state before capture.
        # A positive interval also works with backends that reject dt=0. The
        # returned scratch is discarded, so no prognostic clock/state advances.
        b.advect(self.state,self.cg,min(1.,self.transport_step),cp.cuda.get_current_stream(),None)
        # Constructor-only copies for immutable geometry fingerprinting. They
        # protect restarts from silently loading into another compact cell order.
        self.wet_order_sha256=self._hash_device(b.wet_to_full)
        h=hashlib.sha256()
        for a in (b.lon,b.lat,self.base_depth,b.area):h.update(cp.asnumpy(a).tobytes())
        for name in ("neighbor","left","right","normal_east","normal_north","length","distance","ca","sa",
                     "neighbor_angle","left_edge_angle","left_right_angle"):
            if hasattr(b,name):
                h.update(name.encode());h.update(cp.asnumpy(getattr(b,name)).tobytes())
        h.update(json.dumps({"name":b.name,"shape":self.shape,
            "turning_handled":bool(getattr(b,"turning_handled",False))},sort_keys=True).encode())
        self.geometry_sha256=h.hexdigest()

    def _hash_device(self,array):
        return hashlib.sha256(self.cp.asnumpy(array).tobytes()).hexdigest()

    def _array(self,array,shape,name,dtype="float32"):
        cp=self.cp
        if not hasattr(array,"__cuda_array_interface__"):
            raise TypeError(f"{name} must be a CUDA array")
        a=cp.asarray(array)
        if a.shape!=tuple(shape) or a.dtype!=cp.dtype(dtype) or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous CUDA {dtype} {shape}")
        if a.device.id!=self.state.device.id:
            raise ValueError(f"{name} belongs to another CUDA device")
        return a

    def _launch(self,name,n,args,*,blocks=False):
        if n:self.kernels[name]((n if blocks else (n+127)//128,),(128,),args)

    def _zero(self,array):
        self._launch("wave_zero",array.size,(array,np.int64(array.size)))

    def _geometry(self):
        b=self.backend
        self._launch("wave_geometry",self.nc*self.nf,
            (b.depth,b.lat,b.depth_gradient_east,b.depth_gradient_north,self.depth_rate,
             self.frequency,self.k,self.cg,self.fdot,np.int32(self.nc)))

    @property
    def outputs(self):return self._outputs

    def set_forcing(self,forcing):
        self._wind_bracket=False
        for key,target in (("u10",self.u),("v10",self.v),("rho_air",self.rho)):
            a=self._array(getattr(forcing,key),self.shape,key)
            self._launch("wave_gather",self.nc,(a,self.backend.wet_to_full,target,np.int32(self.nc)))
        self._forcing=forcing

    def set_wind_bracket(self,u0,v0,u1,v1,*,time0,time1):
        if not math.isfinite(time0) or not math.isfinite(time1) or time1<=time0:
            raise ValueError("forcing times must be finite and increasing")
        for j,array in enumerate((u0,v0,u1,v1)):
            self.cp.copyto(self._wind_endpoints[j].reshape(self.shape),self._array(array,self.shape,"wind endpoint"))
        self._wind_times.set(np.asarray([time0,time1],dtype=np.float64),stream=self.cp.cuda.get_current_stream())
        self._wind_bracket=True;self._forcing=True

    def set_water_level(self,eta):
        a=self._array(eta,self.shape,"eta")
        self._launch("wave_water_level",self.nc,(a,self.base_depth,self.backend.wet_to_full,
            self.backend.depth,self.water_level,self.previous_depth,np.int32(self.nc)))
        self.backend.set_depth(self.backend.depth,self.cp.cuda.get_current_stream())

    def initialize_surface(self,forcing):
        """Diagnose initial stress and partitions without advancing action.

        Initial wave products must classify the supplied wind sea with the
        initialization wind, and the first atmosphere step needs its spectral
        roughness. This is initialization only, never an output-cadence solve.
        """
        if self.steps:
            raise ValueError("wave surface initialization must precede time stepping")
        self.set_forcing(forcing)
        self._moments();self._wind();self.refresh_outputs()

    def _table(self,table):
        # A packed table is metadata. It contains no spatial or spectral fields.
        raw=json.dumps(table,sort_keys=True,separators=(",",":")).encode()
        digest=hashlib.sha256(raw).hexdigest()
        if self._required_shape_hash is not None and digest!=self._required_shape_hash:
            raise ValueError("partition shape metadata does not match the required restart identity")
        self.partition_shapes_sha256=digest
        if getattr(self,'_shape_rows_digest',None)==digest:return self._shape_rows
        rows=table.get("rows")
        if not rows:
            raise ValueError("partition shape metadata requires nonempty rows")
        packed=[]
        for row in rows:
            kind=row["kind"]
            kind={"windsea":0,"swell":1}.get(kind,kind)
            upper=row.get("tp_max_seconds")
            values=[kind,row.get("tp_min_seconds",0),1e9 if upper is None else upper,row["gamma"],
                    row["sigma_a"],row["sigma_b"],row["s"]]
            if kind not in (0,1) or not all(math.isfinite(float(v)) for v in values) or not (0<=values[1]<values[2] and values[3]>0 and values[4]>0 and values[5]>0 and values[6]>=0):
                raise ValueError("invalid partition shape row")
            packed.append(values)
        self._shape_rows=self.cp.asarray(packed,dtype=self.cp.float32)
        self._shape_rows_digest=digest
        return self._shape_rows

    def _reconstruct(self,partitions,out,shape_table,point_map=None):
        rows=self._table(shape_table)
        mapping=np.uint64(0) if point_map is None else point_map
        self._launch("wave_reconstruct",out.shape[0],(partitions,mapping,self.frequency,self.df,
            rows,out,self.errors,np.int32(out.shape[0]),np.int32(rows.shape[0])),blocks=True)
        # Keep table storage alive through stream completion without a sync.
        self._shape_rows=rows

    def initialize_partitions(self,partitions,*,valid_time,shape_table):
        p=self._array(partitions,(self.nfull,4,4),"initial partitions")
        if not math.isfinite(valid_time):raise ValueError("initial valid_time must be finite")
        self._reconstruct(p,self.energy,shape_table,self.backend.wet_to_full)
        self._launch("wave_convert",self.state.size,(self.energy,self.state,self.frequency,np.int32(self.nc),np.int32(0)))

        self.time_seconds=float(valid_time);self._initialized=True
        self.device_clock.fill(self.time_seconds)
        self.refresh_outputs()

    def set_boundary_spectra(self,e0,e1,*,time0,time1):
        if not math.isfinite(time0) or not math.isfinite(time1) or time1<=time0:
            raise ValueError("boundary times must be finite and increasing")
        self.cp.copyto(self.boundary_e0,self._array(e0,self.boundary_action.shape,"boundary start"))
        self.cp.copyto(self.boundary_e1,self._array(e1,self.boundary_action.shape,"boundary end"))
        self.boundary_times.set(np.asarray([time0,time1],dtype=np.float64),stream=self.cp.cuda.get_current_stream())
        self._boundary=(self.boundary_e0,self.boundary_e1,float(time0),float(time1))

    def set_boundary_partitions(self,p0,p1,*,time0,time1,shape_table):
        shape=(self.boundary_action.shape[0],4,4)
        p0=self._array(p0,shape,"boundary partitions start");p1=self._array(p1,shape,"boundary partitions end")
        self._reconstruct(p0,self.boundary_e0,shape_table)
        self._reconstruct(p1,self.boundary_e1,shape_table)
        self.set_boundary_spectra(self.boundary_e0,self.boundary_e1,time0=time0,time1=time1)

    def _moments(self,batch=None):
        a=self if batch is None else batch
        self._launch("wave_moments",a.nc,(a.state,self.frequency,self.df,a.k,a.u,a.v,
            a.wind_input_mask,a.moments,np.int32(a.nc),np.float32(self.spectral.ratio)),blocks=True)

    def _wind(self,batch=None):
        a=self if batch is None else batch
        # Fixed nonlinear surface-layer iterations, not time substeps.
        # A warp per cell avoids three idle warps at each sequential tail ring.
        # All directions/frequencies remain covered by the kernel's stride loops.
        self.kernels["wave_wind_fused"]((a.nc,),(32,),
            (a.state,self.frequency,self.df,a.k,a.u,a.v,a.rho,a.shelter,a.moments,
             a.wind_beta,np.int32(a.nc),np.int32(self.pack_id)))
        self._launch("wave_wind_mask",a.wind_input_mask.size,(a.wind_beta,a.wind_input_mask,np.int32(a.nc)))

    def _sources(self,dt):
        self._launch("wave_limiter_reset",self.nc,(self.source_limiter,np.int32(self.nc)))
        # Cell-local stress needs enough independent cells to fill the GPU.
        # Its output borrows the full transport scratch; expensive DIA/saturation
        # intermediates remain tiled. No extra domain spectrum is allocated.
        self._moments();self._wind()
        self._launch("wave_tail",self.nc,(self.state,self.frequency,self.moments,np.int32(self.nc),
            np.int32(self.pack_id),np.float32(math.log(self.spectral.ratio))),blocks=True)
        self._budget_capture(0)
        self._moments();self._wind()
        self._launch("wave_depth_breaking",self.nc,(self.backend.depth,self.moments,self.depth_breaking,np.int32(self.nc)))
        for a in self._batches:self._sources_batch(a,dt)
        self._budget_capture(1)
        self._moments()
        self._launch("wave_tail",self.nc,(self.state,self.frequency,self.moments,np.int32(self.nc),
            np.int32(self.pack_id),np.float32(math.log(self.spectral.ratio))),blocks=True)

        self._budget_capture(2)

    def _sources_batch(self,a,dt):
        if self.pack_id==0:
            self._launch("wave_saturation",a.state.size,(a.state,self.frequency,a.k,a.cg,self.saturation,np.int32(a.nc)))
            self._launch("wave_breaking_weight",a.state.size,(self.saturation,self.frequency,self.df,a.k,a.cg,
                a.depth,self.beta,np.int32(a.nc)))
        self._launch("wave_source_rates",a.state.size,(a.state,self.frequency,self.df,a.k,a.cg,
            a.depth,a.u,a.v,a.rho,a.moments,a.wind_beta,self.saturation,self.beta,a.depth_breaking,self.loss,self.seed,
            np.int32(a.nc),np.int32(self.pack_id),
            getattr(a,"budget_map",np.uintp(0)),self._budget["components"] if self._budget else np.uintp(0),
            np.int32(self._budget["count"] if self._budget else 0)))
        self._launch("wave_dia_moments",a.nc,(a.moments,self.legacy_moments,np.int32(a.nc)))
        self._launch("wave_extend_dia",a.nc*self.nfx*self.nd,(a.state,self.frequency_ext,self.dia_energy,
            np.int32(a.nc),np.int32(self.nfx)))
        self._zero(self.dia_rate)
        self._zero(self.dia_diagonal)
        self._launch("wave_dia_rates",a.nc*self.nfx*self.nd,(self.dia_energy,self.dia_rate,self.dia_diagonal,self.frequency_ext,self.df_ext,a.depth,
            self.legacy_moments,np.int32(a.nc),np.int32(self.nfx),np.int32(self.nd),
            np.float32(math.log(self.spectral.ratio))))
        if self._budget and a.budget_active:
            b=self._budget
            self._launch("wave_budget_sources",b["previous"].size,(a.state,self.frequency,a.moments,a.wind_beta,self.loss,
                self.seed,self.dia_rate,self.dia_diagonal,b["components"],b["cells"],b["totals"],
                np.int32(b["count"]),np.int32(a.first),np.int32(a.nc),np.int32(self.nfx),np.float32(dt)))
        self._launch("wave_sources",a.state.size,(a.state,self.frequency,a.moments,a.wind_beta,self.loss,
            self.seed,self.dia_rate,self.dia_diagonal,a.source_limiter,np.int32(a.nc),np.int32(self.nfx),np.float32(dt)))

    def _stage(self,name,fn):
        stage=self.stage_names.index(name);wall=time.perf_counter()
        self._launch("wave_timer",1,(self.device_timing,np.int32(stage),np.int32(0)))
        fn()
        self._launch("wave_timer",1,(self.device_timing,np.int32(stage),np.int32(1)))
        self.stage_wall[name]=self.stage_wall.get(name,0)+time.perf_counter()-wall

    def _interpolate_wind(self):
        if self._wind_bracket:
            self._launch("wave_forcing_bracket",self.nc,(*tuple(self._wind_endpoints[j] for j in range(4)),
                self.backend.wet_to_full,self.device_clock,self._wind_times,self.u,self.v,self.errors,np.int32(self.nc)))

    def step(self,dt):
        if self._forcing is None:raise RuntimeError("set_forcing must precede step")
        if not math.isfinite(dt) or dt<=0:raise ValueError("wave step must be positive and finite")
        self._interpolate_wind()
        if self._boundary is not None:
            _,_,t0,t1=self._boundary
            if self.time_seconds<t0-1e-6 or self.time_seconds+dt>t1+1e-6:
                raise ValueError("wave step crosses the boundary time bracket; supply adjacent endpoint spectra")
        self._launch("wave_depth_rate",self.nc,(self.backend.depth,self.previous_depth,self.depth_rate,
            np.int32(self.nc),np.float32(dt)))
        self._geometry()
        self._stage("sources_first_half",lambda:self._sources(dt*.5))
        def propagate():
            for first,count,bound in self.transport_groups:
                pieces=math.ceil(dt/bound);h=dt/pieces
                for j in range(pieces):
                    boundary=None
                    if self._boundary is not None:
                        e0,e1,t0,t1=self._boundary
                        self._launch("wave_boundary",self.boundary_action.size,(e0,e1,self.boundary_action,
                            self.frequency,np.int32(self.boundary_action.shape[0]),self.device_clock,
                            self.boundary_times,np.float64((j+.5)*h),self.errors))
                        boundary=self.boundary_action
                    stream=self.cp.cuda.get_current_stream()
                    if hasattr(self.backend,"advect_group"):
                        advanced=self.backend.advect_group(self.state,self.cg,h,stream,boundary,first,count)
                    else:
                        advanced=self.backend.advect(self.state,self.cg,h,stream,boundary)
                    self._launch("wave_copy_group",self.nc*count*self.nd,(advanced,self.state,
                        np.int32(self.nc),np.int32(first),np.int32(count)))
                    self._budget_capture(3)
                    turn_threads=1 << (self.nd-1).bit_length()
                    self.kernels["wave_refract_geometry"]((self.nc*count,),(turn_threads,),
                        (self.state,self.backend.depth,self.backend.lat,self.backend.depth_gradient_east,
                         self.backend.depth_gradient_north,self.frequency,self.k,self.cg,np.int32(self.nc),np.int32(self.nf),
                         np.int32(self.nd),np.float32(h),np.int32(first),np.int32(count)),shared_mem=5*turn_threads*4)
                    self._budget_capture(4)
            self._launch("wave_frequency",self.nc*self.nd,(self.state,self.fdot,self.df,np.int32(self.nc),np.float32(dt)))
            self._budget_capture(5)
        self._stage("propagation",propagate)
        self._stage("sources_second_half",lambda:self._sources(dt*.5))
        self._stage("diagnostics",lambda:self.refresh_outputs(refresh_stress=True))
        self._launch("wave_clock",1,(self.device_clock,self.device_steps,np.float64(dt)))
        self.time_seconds+=dt;self.steps+=1

    def refresh_outputs(self,*,refresh_stress=False):
        if refresh_stress:self._interpolate_wind()
        self._moments()
        if refresh_stress:self._wind()
        for a in self._batches:self._diagnose_batch(a)

    def _diagnose_batch(self,a):
        self._launch("wave_smooth",a.state.size,(a.state,self.frequency,self.smooth,np.int32(a.nc)))
        self._launch("wave_watershed",a.state.size,(self.smooth,self.labels,np.int32(a.nc)))
        labels=self.labels
        self.kernels["wave_merge_basins"]((a.nc,),(256,),
            (a.state,self.smooth,labels,self.frequency,self.df,a.k,a.u,a.v,self.partition_workspace,np.int32(a.nc)))
        self._launch("wave_diagnose",a.nc,(a.state,labels,self.frequency,self.df,self.basin_variance,
            self.basin_wind,a.selected,a.moments,a.wet,self.bulk,np.int32(a.nc),
            np.int32(self.nfull)),blocks=True)

    def enable_point_budget(self,indices):
        """Observe cumulative E increments at selected wet cells, without changing physics."""
        cp=self.cp;cells=cp.asarray(indices,dtype=cp.int32)
        if bool(cp.any((cells<0)|(cells>=self.nc))):raise ValueError("invalid budget wet cells")
        count=cells.size;mapping=cp.full(self.nc,-1,cp.int32);mapping[cells]=cp.arange(count,dtype=cp.int32)
        terms=["tail_before","source_actual","tail_after","geographic_including_boundary","angular","frequency","reserved",
               *["raw_"+x for x in ("wind","seed","dia","bottom","depth_breaking","whitecapping","cumulative_breaking","swell")],
               *["applied_"+x for x in ("wind","seed","dia","bottom","depth_breaking","whitecapping","cumulative_breaking","swell")]]
        self._budget={"cells":cells,"count":count,"terms":terms,"map":mapping,
                      "previous":cp.zeros((count,self.nf,self.nd),cp.float64),
                      "totals":cp.zeros((len(terms),count,self.nf,self.nd),cp.float64),
                      "components":cp.zeros((5,count,self.nf,self.nd),cp.float32)}
        host_cells=cp.asnumpy(cells).tolist()
        for a in self._batches:
            a.budget_map=mapping[a.first:a.first+a.nc]
            a.budget_active=any(a.first<=c<a.first+a.nc for c in host_cells)
        self._budget_capture(-1)

    def _budget_capture(self,slot):
        if self._budget:
            b=self._budget
            self._launch("wave_budget_capture",b["previous"].size,(self.state,self.frequency,b["cells"],
                b["previous"],b["totals"],np.int32(b["count"]),np.int32(slot)))

    def set_stations(self,indices):
        a=self._array(indices,(indices.size,),"station full indices","int32")
        cp=self.cp
        if bool(cp.any((a<0)|(a>=self.nfull))):raise ValueError("station outside full grid")
        compact=self.backend.full_to_wet[a]
        if bool(cp.any(compact<0)):raise ValueError("station must identify a wet cell")
        self._station_full=a.copy();self._station_compact=compact.copy()
        self._station_spectra=cp.empty((a.size,self.nf,self.nd),cp.float32)
        self._station_bulk=cp.empty((len(PRODUCTS),a.size),cp.float32)

    def station_spectra(self):
        self._launch("wave_station_spectra",self._station_spectra.size,(self.state,self._station_compact,
            self.frequency,self._station_spectra,np.int32(self._station_full.size)))
        return self._station_spectra

    def station_outputs(self):
        self._launch("wave_station_bulk",self._station_bulk.size,(self.bulk,self._station_full,
            self._station_bulk,np.int32(self._station_full.size),np.int32(self.nfull)))
        return {name:self._station_bulk[j] for j,name in enumerate(PRODUCTS)}

    def source_diagnostics(self,cell):
        """Re-evaluate terms at one retained state without advancing it.

        Returned views share the bounded arena and must be consumed before
        another solver launch. Surface-iteration state is copied for this
        output-cadence operation, so inspection cannot alter continuation.
        """
        if not 0<=cell<self.nc:raise ValueError("source diagnostic cell outside wet order")
        original=self._batches[cell//self.batch_cells]
        a=SimpleNamespace(**vars(original))
        for name in ("moments","shelter","wind_input_mask"):
            setattr(a,name,getattr(original,name).copy())
        self._moments(a);self._wind(a)
        self._launch("wave_depth_breaking",a.nc,(a.depth,a.moments,a.depth_breaking,np.int32(a.nc)))
        if self.pack_id==0:
            self._launch("wave_saturation",a.state.size,(a.state,self.frequency,a.k,a.cg,self.saturation,np.int32(a.nc)))
            self._launch("wave_breaking_weight",a.state.size,(self.saturation,self.frequency,self.df,a.k,a.cg,
                a.depth,self.beta,np.int32(a.nc)))
        self._launch("wave_source_rates",a.state.size,(a.state,self.frequency,self.df,a.k,a.cg,
            a.depth,a.u,a.v,a.rho,a.moments,a.wind_beta,self.saturation,self.beta,a.depth_breaking,self.loss,self.seed,
            np.int32(a.nc),np.int32(self.pack_id),
            getattr(a,"budget_map",np.uintp(0)),self._budget["components"] if self._budget else np.uintp(0),
            np.int32(self._budget["count"] if self._budget else 0)))
        self._launch("wave_dia_moments",a.nc,(a.moments,self.legacy_moments,np.int32(a.nc)))
        self._launch("wave_extend_dia",a.nc*self.nfx*self.nd,(a.state,self.frequency_ext,self.dia_energy,np.int32(a.nc),np.int32(self.nfx)))
        self._zero(self.dia_rate);self._zero(self.dia_diagonal)
        self._launch("wave_dia_rates",a.nc*self.nfx*self.nd,(self.dia_energy,self.dia_rate,self.dia_diagonal,
            self.frequency_ext,self.df_ext,a.depth,self.legacy_moments,np.int32(a.nc),np.int32(self.nfx),
            np.int32(self.nd),np.float32(math.log(self.spectral.ratio))))
        local=cell%self.batch_cells
        return {"wind_beta_per_s":a.wind_beta[local],"loss_per_s":self.loss[local],
                "seed_efth_per_s":self.seed[local],"dia_efth_per_s":self.dia_rate[local,:self.nf],
                "dia_diagonal_per_s":self.dia_diagonal[local,:self.nf],"keepalive":a}

    def allocation_receipt(self):
        """Deduplicate aliased CUDA allocation owners, without copying fields."""
        owners={}
        def visit(value):
            if isinstance(value,self.cp.ndarray):
                memory=value.data.mem;owners[int(memory.ptr)]=int(memory.size)
            elif isinstance(value,(tuple,list)):
                for item in value:visit(item)
            elif isinstance(value,dict):
                for item in value.values():visit(item)
            elif isinstance(value,SimpleNamespace):visit(vars(value))
        visit(vars(self));visit(vars(self.backend))
        total=sum(owners.values())
        return {"unique_array_bytes":total,"bytes_per_wet_cell":total/self.nc,
                "source_workspace_bytes":self._source_workspace.nbytes,"source_batch_cells":self.batch_cells,
                "full_domain_spectra":1+len(self._transport_workspace) if self._transport_workspace else None,
                "pool_used_bytes":self.cp.get_default_memory_pool().used_bytes(),
                "pool_reserved_bytes":self.cp.get_default_memory_pool().total_bytes()}

    def collect(self):
        """Explicit output-cadence barrier, validation and completed GPU timings."""
        cp=self.cp;cp.cuda.get_current_stream().synchronize()
        if int(self.errors[0]):raise ValueError("wave input error: invalid partition reconstruction or expired boundary bracket")
        if not bool(cp.all(cp.isfinite(self.state))) or float(cp.min(self.state)) < -1e-8:
            raise RuntimeError("wave action is nonfinite or negative")
        if not bool(cp.all(cp.isfinite(self.bulk))):raise RuntimeError("wave bulk diagnostics are nonfinite")
        elapsed=cp.asnumpy(self.device_timing)
        self.gpu_seconds={name:int(elapsed[j,1])*1e-9 for j,name in enumerate(self.stage_names)}
        self.time_seconds=float(self.device_clock[0]);actual_steps=int(self.device_steps[0])
        return {"steps":actual_steps,"time_seconds":self.time_seconds,"gpu_seconds":dict(self.gpu_seconds),
                "timing_scope":"CUDA globaltimer stage wall totals, including every graph replay and shared-card stalls",
                "enqueue_wall_seconds":dict(self.stage_wall),"nwet":self.nc,"nf":self.nf,"nd":self.nd,
                "source_limiter_min":float(cp.min(self.source_limiter)),"dia_limiter_min":None,
                "source_integration":"joint diagonal implicit; DIA derivative included; HJ increment and positivity limit",
                "pool_bytes":cp.get_default_memory_pool().total_bytes(),"source_pack":self.source_pack,
                "allocation":self.allocation_receipt(),
                "source_parameter_sha256":self.source_parameter_sha256,"geometry_sha256":self.geometry_sha256}

    def _restart_metadata(self):
        return {"schema":"gpuwm.waves.restart/v1","time_seconds":float(self.device_clock[0]),
                "spectral_contract":CONTRACT["schema"],"spectral":asdict(self.spectral),"source_pack":self.source_pack,
                "source_parameter_sha256":self.source_parameter_sha256,"backend":self.backend.name,
                "geometry_sha256":self.geometry_sha256,"wet_order_sha256":self.wet_order_sha256,
                "partition_shapes_sha256":self.partition_shapes_sha256}

    def restart_state(self):
        return {"metadata":self._restart_metadata(),"arrays":self.checkpoint_arrays()}

    def restore_state(self,payload):
        metadata=payload["metadata"];arrays=payload["arrays"]
        self.validate_checkpoint(metadata,arrays)
        for key,target in self.checkpoint_arrays().items():self.cp.copyto(target,arrays[key])
        self.time_seconds=float(metadata["time_seconds"]);self._initialized=True;self._forcing=None
        self.backend.set_depth(self.backend.depth,self.cp.cuda.get_current_stream())
        self._geometry()

    def checkpoint_arrays(self):
        return {"action":self.state,"depth":self.backend.depth,"water_level":self.water_level,
                "depth_previous":self.previous_depth,"surface_iteration":self.moments,
                "shelter_iteration":self.shelter,"wind_input_iteration":self.wind_input_mask,
                "bulk_outputs":self.bulk,"clock":self.device_clock,"steps":self.device_steps}

    def checkpoint_metadata(self):
        return self._restart_metadata()

    def validate_checkpoint(self,metadata,arrays):
        expected=self._restart_metadata()
        for key in ("schema","spectral_contract",*CONTRACT["restart_match"]):
            if metadata.get(key)!=expected[key]:raise ValueError(f"wave restart mismatch: {key}")
        if not math.isfinite(metadata["time_seconds"]):raise ValueError("invalid restart clock")
        if set(arrays)!=set(self.checkpoint_arrays()):raise ValueError("wave restart array inventory differs")
        for key,target in self.checkpoint_arrays().items():
            self._array(arrays[key],target.shape,f"restart {key}",str(target.dtype))
        if float(arrays["clock"][0])!=metadata["time_seconds"] or int(arrays["steps"][0])<0:
            raise ValueError("wave restart clock/count disagrees with metadata")

    def restore_metadata(self,metadata):
        self.time_seconds=float(metadata["time_seconds"]);self._initialized=True;self._forcing=None
        self.backend.set_depth(self.backend.depth,self.cp.cuda.get_current_stream())
        self._geometry()

    def resident_arrays(self):
        """Persistent solver/backend allocations, excluding borrowed forcing."""
        result={}
        def visit(prefix,value):
            if isinstance(value,self.cp.ndarray):result[prefix]=value
            elif isinstance(value,(list,tuple)):
                for j,item in enumerate(value):visit(f"{prefix}/{j}",item)
        for owner,obj in (("solver",self),("backend",self.backend)):
            for name,value in vars(obj).items():
                if name!="_forcing":visit(f"{owner}/{name}",value)
        return result
