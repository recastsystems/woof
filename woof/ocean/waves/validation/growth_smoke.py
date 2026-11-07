"""One bounded GPU smoke, both published packs at 10 and 20 m/s.

All model calculations and profile extraction are CUDA. The small JSON and
analysis charts are receipts, not weather-field maps. Run under the GPU pool.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import cupy as cp
import numpy as np
from woof.ocean.waves.action import ActionSolver
from woof.ocean.waves.backends.structured import StructuredBackend


class HomogeneousBackend:
    """Exact geographic limit for a spatially uniform deep ocean."""
    name="homogeneous"
    ncell=1
    full_shape=(1,)
    def __init__(self):
        self.wet_to_full=cp.zeros(1,cp.int32);self.full_to_wet=cp.zeros(1,cp.int32)
        self.lon=cp.zeros(1,cp.float32);self.lat=cp.zeros(1,cp.float32)
        self.depth=cp.full(1,1000,cp.float32);self.area=cp.full(1,1e12,cp.float32)
        self.depth_gradient_east=cp.zeros(1,cp.float32);self.depth_gradient_north=cp.zeros(1,cp.float32)
        self.boundary_cells=cp.empty(0,cp.int32);self.work=None
    def max_dt(self,cg_max):return 1e20
    def set_depth(self,depth,stream):cp.copyto(self.depth,depth)
    def advect(self,state,group_velocity,dt,stream,incident_boundary):
        if self.work is None:self.work=cp.empty_like(state)
        with stream:cp.copyto(self.work,state)
        return self.work


def fetch_backend():
    """One-dimensional fetch experiment, periodic across the wind.

    A 2 km first cell grows geometrically by 3 percent over 128 cells, giving
    a roughly 2.9 Mm fetch without losing short-fetch resolution. Four periodic
    rows remove the finite-width fetch contamination of the earlier channel.
    Geometry is a laboratory Cartesian plane; no weather map is generated.
    """
    ny,nx=4,128;shape=(ny,nx);nc=nx*ny;dy=25000.
    width=2000*cp.power(cp.float32(1.03),cp.arange(nx,dtype=cp.float32))
    centers=cp.cumsum(width)-.5*width
    grid=SimpleNamespace(nx=nx,ny=ny,dx=2000.,dy=dy,
        latitude=cp.zeros(shape,cp.float32),longitude=cp.broadcast_to(centers[None,:]/6371000*180/math.pi,shape).copy(),
        cosalpha=cp.ones(shape,cp.float32),sinalpha=cp.zeros(shape,cp.float32))
    bathy=SimpleNamespace(depth=cp.full(shape,1000,cp.float32),wet=cp.ones(shape,cp.float32))
    b=StructuredBackend(grid,bathy,kind="curvilinear",map_factor=cp.ones(shape,cp.float32))
    b.name="idealized_fetch"
    cells=cp.arange(nc,dtype=cp.int32);y,x=cp.divmod(cells,nx)
    b.neighbor[:,2]=((y-1)%ny)*nx+x;b.neighbor[:,3]=((y+1)%ny)*nx+x
    b.boundary_cells=cp.flatnonzero((x==0)|(x==nx-1)).astype(cp.int32)
    slots=cp.full(nc,-1,cp.int32);slots[b.boundary_cells]=cp.arange(b.boundary_cells.size,dtype=cp.int32)
    b.boundary_slot.fill(-1)
    b.boundary_slot[:,0]=cp.where(x==0,slots,-1);b.boundary_slot[:,1]=cp.where(x==nx-1,slots,-1)
    w=width[x];wl=width[cp.maximum(x-1,0)];wr=width[cp.minimum(x+1,nx-1)]
    b.area=(w*dy).astype(cp.float32)
    b.length=cp.stack((cp.full(nc,dy,cp.float32),cp.full(nc,dy,cp.float32),w,w),axis=1)
    b.distance=cp.stack((.5*(w+wl),.5*(w+wr),cp.full(nc,dy,cp.float32),cp.full(nc,dy,cp.float32)),axis=1)
    b._cfl_length=float(cp.min(b.area/cp.sum(b.length,axis=1)))
    b._diffusion_metric=float(cp.max(cp.sum(b.length/b.distance,axis=1)/b.area))
    b.fetch_centers=centers
    return b


def main():
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path)
    parser.add_argument("--duration-hours",type=int,default=72)
    parser.add_argument("--dt-seconds",type=int,default=600)
    parser.add_argument("--only-duration",action="store_true")
    parser.add_argument("--budget-seconds",type=float,default=570.)
    args=parser.parse_args()
    if not 30<=args.budget_seconds<=570:parser.error("budget must be between30 and570 seconds")
    if args.duration_hours<24:parser.error("duration must cover at least 24 hours for convergence sampling")
    if not 1<=args.dt_seconds<=600 or 3600%args.dt_seconds:
        parser.error("fixed dt must divide one hour and lie between 1 and 600 seconds")
    per_hour=3600//args.dt_seconds
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cpu_start=time.process_time()
    receipt={"schema":"gpuwm.waves.growth-smoke/v1","status":"RUNNING","results":[],
        "started_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,"-m","woof.ocean.waves.validation.growth_smoke",*sys.argv[1:]],
        "git_commit":subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True,timeout=10).stdout.strip(),
        "references":{"jonswap":"Hasselmann et al. 1973, DHZ supplement A8(12), epsilon=1.6e-7 chi, fp U/g=3.5 chi^(-1/3)",
        "composite":"Kahma and Calkoen 1992, JPO 22, epsilon=5.2e-7 chi^.9, omega_p U/g=13.7 chi^(-.27)",
        "duration":"equivalent fetch obtained by integrating the group speed of each published fetch-peak law; diagnostic scaling, not independent observed duration data"},
        "config":{"dt_seconds":args.dt_seconds,"duration_hours":args.duration_hours,"fetch_hours":72,"only_duration":args.only_duration,"nf":36,"nd":36,"fetch_grid":[4,128],"first_cell_m":2000,"cell_width_ratio":1.03,
                  "crosswind_boundary":"periodic","geometry":"idealized Cartesian channel",
                  "depth_m":1000,"gse_spreading_seconds":3600,"initial":"calm, idealised smoke only"}}
    path=args.output/"receipt.json"
    def save():path.write_text(json.dumps(receipt,indent=2)+"\n")
    save()
    try:
        receipt["gpu"]=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        for pack in ("st4","janssen"):
            for speed in (10.,20.):
                for mode in (("duration",) if args.only_duration else ("duration","fetch")):
                    if time.perf_counter()-start>args.budget_seconds-20:raise TimeoutError("ten-minute smoke budget reached")
                    item={"source_pack":pack,"wind_ms":speed,"mode":mode,"hourly":[]}
                    began=time.perf_counter();shape=(1,)
                    if mode=="duration":
                        backend=HomogeneousBackend()
                        model=ActionSolver(None,None,{"grid_backend":backend,"source_pack":pack})
                    else:
                        backend=fetch_backend();shape=backend.full_shape
                        model=ActionSolver(None,None,{"source_pack":pack,"grid_backend":backend})
                    forcing=SimpleNamespace(u10=cp.full(shape,speed,cp.float32),v10=cp.zeros(shape,cp.float32),
                        rho_air=cp.full(shape,1.225,cp.float32),time_seconds=0.)
                    model.set_forcing(forcing)
                    # No warmup integration: capture builds one actual repeated step.
                    stream=cp.cuda.Stream(non_blocking=True)
                    cp.cuda.get_current_stream().synchronize()
                    with stream:
                        stream.begin_capture();model.step(float(args.dt_seconds));graph=stream.end_capture()
                    cp.cuda.get_current_stream().synchronize()
                    hours=args.duration_hours if mode=="duration" else 72
                    steps=hours*per_hour
                    for step in range(1,steps+1):
                        graph.launch(stream=cp.cuda.get_current_stream())
                        if step%per_hour==0:
                            stats=model.collect()
                            center=(0,) if mode=="duration" else (shape[0]//2,shape[1]//2)
                            item["hourly"].append({"hour":step/per_hour,"hs_m":float(model.outputs.hs[center]),
                                "tp_s":float(model.outputs.tp[center]),"ustar":float(model.outputs.ustar[center]),
                                "z0":float(model.outputs.z0[center]),"charnock":float(model.outputs.charnock[center]),
                                "tau_wave_east":float(model.outputs.tau_wave_east[center]),
                                "tau_wave_north":float(model.outputs.tau_wave_north[center])})
                            if step in (steps-24*per_hour,steps-12*per_hour,steps):
                                item.setdefault("stationarity",[]).append({"hour":step/per_hour,
                                    "hs":cp.asnumpy(model.outputs.hs[shape[0]//2] if mode=="fetch" else model.outputs.hs).tolist(),
                                    "tp":cp.asnumpy(model.outputs.tp[shape[0]//2] if mode=="fetch" else model.outputs.tp).tolist()})
                            if time.perf_counter()-start>args.budget_seconds:raise TimeoutError("ten-minute smoke budget reached")
                    if mode=="fetch":
                        item["fetch_m"]=cp.asnumpy(backend.fetch_centers).tolist()
                        item["hs_m"]=cp.asnumpy(model.outputs.hs[shape[0]//2]).tolist()
                        item["tp_s"]=cp.asnumpy(model.outputs.tp[shape[0]//2]).tolist()
                    item["summary"]=model.collect()
                    # Preserve the spectrum and term tendencies behind the gate,
                    # with no extra integrations or parameter arms.
                    cell=0 if mode=="duration" else (shape[0]//2)*shape[1]+shape[1]//2
                    terms=model.source_diagnostics(cell)
                    item["spectral_diagnosis"]={"cell":cell,
                        "frequency_hz":cp.asnumpy(model.frequency).tolist(),
                        "efth":cp.asnumpy(model.state[cell]*(2*math.pi*model.frequency[:,None])).tolist(),
                        **{key:cp.asnumpy(value).tolist() for key,value in terms.items() if key!="keepalive"},
                        "dia_update_factor":None,
                        "budget_scope":"all terms evaluated at the same final spectrum; no prognostic advance"}
                    if item["summary"]["steps"]!=steps or item["summary"]["time_seconds"]!=hours*3600:
                        raise AssertionError("captured device step count/clock did not advance exactly")
                    item["wall_seconds"]=time.perf_counter()-began
                    receipt["results"].append(item);save();print(json.dumps({"finished":mode,"pack":pack,"wind":speed,"wall_seconds":item["wall_seconds"],"last":item["hourly"][-1]}),flush=True)
                    del terms,graph,model,forcing,backend
                    cp.cuda.get_current_stream().synchronize();cp.get_default_memory_pool().free_all_blocks()
        receipt["status"]="DONE"
    except BaseException as exc:
        receipt["status"]="FAILED";receipt["error"]=f"{type(exc).__name__}: {exc}";raise
    finally:
        receipt["wall_seconds"]=time.perf_counter()-start;receipt["cpu_seconds"]=time.process_time()-cpu_start;save()
    charts(args.output,receipt)


def charts(output,receipt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mode in ("fetch","duration"):
        if not any(item["mode"]==mode for item in receipt["results"]):continue
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for column,speed in enumerate((10.,20.)):
            axh,axf=axes[0,column],axes[1,column]
            for item in receipt["results"]:
                if item["mode"]!=mode or item["wind_ms"]!=speed:continue
                if mode=="fetch":
                    x=np.asarray(item["fetch_m"])*9.80665/speed**2
                    hs=np.asarray(item["hs_m"]);tp=np.asarray(item["tp_s"])
                else:
                    x=np.asarray([r["hour"]*3600 for r in item["hourly"]])*9.80665/speed
                    hs=np.asarray([r["hs_m"] for r in item["hourly"]]);tp=np.asarray([r["tp_s"] for r in item["hourly"]])
                axh.loglog(x,hs*9.80665/speed**2,label=item["source_pack"])
                axf.loglog(x,np.divide(speed,9.80665*tp,out=np.full_like(tp,np.nan),where=tp>0),label=item["source_pack"])
                if mode=="fetch":
                    for profile in item["stationarity"][:-1]:
                        axh.loglog(x,np.asarray(profile["hs"])*9.80665/speed**2,":",alpha=.4,
                            label=f"{item['source_pack']} {profile['hour']:g} h")
                        axf.loglog(x,speed/(9.80665*np.asarray(profile["tp"])),":",alpha=.4)
            xr=np.geomspace(max(1,float(x.min())),float(x.max()),200)
            for label,a,p,b,q in (("JONSWAP",1.6e-7,1.,3.5,1/3),("KC composite",5.2e-7,.9,13.7/(2*math.pi),.27)):
                chi=xr if mode=="fetch" else (xr*(1-q)/(4*math.pi*b))**(1/(1-q))
                hcurve=4*np.sqrt(a*chi**p);fcurve=b*chi**(-q)
                valid=(hcurve<=.27)&(fcurve>=.13)
                axh.loglog(xr,np.where(valid,hcurve,np.nan),"--",label=label)
                axf.loglog(xr,np.where(valid,fcurve,np.nan),"--",label=label)
            axh.axhspan(.24,.27,color="grey",alpha=.18,label="PM reference (not an ST4 gate; see GROWTH-REFEREES.md)")
            axf.axhline(.13,color="black",ls=":",label="PM peak limit")
            axh.set_title(f"{speed:g} m/s, {mode}-limited")
            axh.set_ylabel("Hs g / U10 squared");axf.set_ylabel("fp U10 / g")
            axf.set_xlabel("fetch g / U10 squared" if mode=="fetch" else "time g / U10")
            for ax in (axh,axf):ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=8)
        hours=receipt["config"].get("fetch_hours",72) if mode=="fetch" else receipt["config"]["duration_hours"]
        fig.suptitle(f"{hours} h candidate growth, 36 x 36 bins; references stop at full development"+("\nDuration references use equivalent-fetch scaling" if mode=="duration" else ""))
        fig.savefig(output/f"growth-{mode}.png",dpi=140);plt.close(fig)


if __name__=="__main__":main()
