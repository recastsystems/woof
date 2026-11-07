"""Standalone prepared-input door: Rust bytes/NetCDF, CUDA physics and sampling."""
from __future__ import annotations
import bisect
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import time
import tomllib
from types import SimpleNamespace
import numpy as np
from .action import ActionSolver,PACKAGE,PRODUCTS


class NativeIO:
    def __init__(self):
        from .cli import native_tools
        _,path=native_tools();self.lib=ctypes.CDLL(str(path));p=ctypes.c_void_p;s=ctypes.c_size_t
        self.lib.waves_load_bytes.argtypes=[ctypes.c_char_p,p,s]
        self.lib.waves_v1_snapshot.argtypes=[ctypes.c_char_p,p,p,p,s,s,ctypes.c_double,p]
        self.lib.waves_v1_points.argtypes=[ctypes.c_char_p,p,p,p,p,p,s,s,s,s,s,ctypes.c_double,p,p]
        for name in ("waves_load_bytes","waves_v1_snapshot","waves_v1_points"):
            getattr(self.lib,name).restype=ctypes.c_int

    def load(self,root,entry):
        path=(root/entry["path"]).resolve()
        if not path.is_relative_to(root) or path.suffix!=".bin":
            raise ValueError("prepared arrays must be .bin files inside the prepared folder")
        dtype=entry["dtype"]
        if dtype not in ("<f4","<i4"):raise ValueError("prepared array dtype must be little-endian float32 or int32")
        shape=tuple(entry["shape"])
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in shape):
            raise ValueError("invalid prepared shape")
        count=math.prod(shape);host=np.empty(shape,dtype=dtype)
        if host.nbytes!=entry["bytes"] or path.stat().st_size!=host.nbytes:
            raise ValueError(f"prepared byte count mismatch: {entry['path']}")
        # Hash verification is metadata integrity, not a field transform.
        with path.open("rb") as file:digest=hashlib.file_digest(file,"sha256").hexdigest()
        if digest!=entry["sha256"]:raise ValueError(f"prepared hash mismatch: {entry['path']}")
        if count and self.lib.waves_load_bytes(os.fsencode(path),host.ctypes.data,host.nbytes):
            raise RuntimeError("Rust prepared byte read failed")
        return host


def run(config_path):
    import cupy as cp
    path=Path(config_path).resolve()
    config=tomllib.loads(path.read_text()) if path.suffix.lower()==".toml" else json.loads(path.read_text())
    cfg=config.get("waves",config)
    prepared=(path.parent/cfg["prepared"]).resolve()
    manifest_path=prepared/"manifest.json";manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="gpuwm.waves.prepared/v1" or manifest.get("status")!="complete":
        raise ValueError("complete gpuwm.waves.prepared/v1 input required")
    requested=(path.parent/cfg["output"]).resolve();output=requested
    for suffix in range(10000):
        output=requested if suffix==0 else requested.with_name(f"{requested.name}-{suffix:03}")
        try:output.mkdir(parents=True,exist_ok=False);break
        except FileExistsError:continue
    else:raise RuntimeError("output suffix space exhausted")
    receipt={"schema":"gpuwm.waves.run/v1","status":"RUNNING","command":["python","-m","woof.ocean.waves","run",str(path)],
        "config":config,"prepared_manifest":str(manifest_path),"prepared_manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frames":[],"stages":{},"output":str(output),"source_pack_status":"candidate, no inferred skill qualification"}
    receipt_file=output/"receipt.json";started=time.perf_counter();cpu_start=time.process_time()
    def save():
        temporary=receipt_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt,indent=2)+"\n")
        temporary.replace(receipt_file)
    def interrupted(signum,frame):
        raise InterruptedError(f"wave run interrupted by signal {signum}")
    old_sigterm=signal.signal(signal.SIGTERM,interrupted)
    save();print(json.dumps({"output":str(output)}),flush=True)
    try:
        io=NativeIO();stage=time.perf_counter()
        host={name:io.load(prepared,entry) for name,entry in manifest["arrays"].items()}
        device={name:cp.asarray(a) for name,a in host.items()}
        info=manifest["grid"];nx,ny=int(info["nx"]),int(info["ny"]);shape=(ny,nx)
        if nx*ny!=info["ncell"]:raise ValueError("structured prepared shape mismatch")
        if info["kind"] not in ("latlon","curvilinear"):
            raise ValueError("standalone structured door requires latlon or curvilinear; mesh caller supplies its backend")
        grid=SimpleNamespace(nx=nx,ny=ny,dx=float(info["dx"]),dy=float(info["dy"]),
            latitude=device["lat"].reshape(shape),longitude=device["lon"].reshape(shape),
            cosalpha=device["cosalpha"].reshape(shape),sinalpha=device["sinalpha"].reshape(shape))
        wet=device["wet"].astype(cp.float32).reshape(shape)
        bathy=SimpleNamespace(depth=device["depth"].reshape(shape),wet=wet)
        options={key:cfg[key] for key in ("spectral","source_pack","gse_spreading_seconds") if key in cfg}
        options.update(backend=info["kind"],map_factor=device["map_factor"].reshape(shape),time_seconds=manifest["initial_valid_time"])
        model=ActionSolver(grid,bathy,options)
        table=json.loads((PACKAGE/"partition_shapes.v1.json").read_text())
        model.initialize_partitions(device["initial_partitions"],valid_time=manifest["initial_valid_time"],shape_table=table)
        # Incident spectra order follows the backend, independently of file order.
        boundary_index=device["boundary_cells"]
        lookup=cp.full(nx*ny,-1,cp.int32);lookup[boundary_index]=cp.arange(boundary_index.size,dtype=cp.int32)
        order=lookup[model.backend.wet_to_full[model.backend.boundary_cells]]
        if bool(cp.any(order<0)):raise ValueError("prepared boundary omits a backend open boundary cell")
        boundary=cp.ascontiguousarray(device["boundary_partitions"][:,order])
        stations=list(manifest["stations"])
        budget_points=[]
        for point in cfg.get("budget_points",[]):
            # Geographic selection is a CUDA reduction; prepared bytes came from Rust.
            dlat=cp.deg2rad(device["lat"]-float(point["lat"]))
            dlon=cp.deg2rad(device["lon"]-float(point["lon"]))
            distance=cp.sin(dlat/2)**2+cp.cos(cp.deg2rad(device["lat"]))*math.cos(math.radians(point["lat"]))*cp.sin(dlon/2)**2
            distance=cp.where(device["wet"]>0,distance,cp.inf)
            cell=int(cp.argmin(distance))
            selected={**point,"cell":cell,"lat":float(host["lat"][cell]),"lon":float(host["lon"][cell])}
            budget_points.append(selected);stations.append(selected)
        if budget_points:
            model.enable_point_budget(model.backend.full_to_wet[cp.asarray([p["cell"] for p in budget_points],dtype=cp.int32)])
            (output/"budget").mkdir()
            receipt["budget"]={"points":budget_points,"terms":model._budget["terms"],
                "scope":"Cumulative actual split-operator increments and raw/applied per-term source increments at fixed Eulerian cells; transport includes boundary influence. No per-step host synchronization."}

        model.set_stations(cp.asarray([s["cell"] for s in stations],dtype=cp.int32))
        times=[float(v) for v in manifest["forcing_times"]]
        btimes=[float(v) for v in manifest["boundary_times"]]
        if len(times)<2 or len(btimes)<2 or any(b<=a for a,b in zip(times,times[1:])) or any(b<=a for a,b in zip(btimes,btimes[1:])):
            raise ValueError("forcing/boundary times must strictly increase")
        when=float(manifest["initial_valid_time"]);end=min(times[-1],btimes[-1])
        if when<max(times[0],btimes[0]) or end<=when:raise ValueError("no common forcing/boundary time coverage")
        interval=float(cfg.get("output_seconds",21600));step=float(cfg.get("dt_seconds",600))
        if not math.isfinite(interval) or not math.isfinite(step) or min(interval,step)<=0:
            raise ValueError("step and output intervals must be finite and positive")
        output_times=[when+j*interval for j in range(math.floor((end-when)/interval)+1)]
        if output_times[-1]<end-1e-6:output_times.append(end)
        cp.cuda.get_current_stream().synchronize()
        receipt["stages"]["initialize_wall_seconds"]=time.perf_counter()-stage
        receipt["grid"]={**info,"nwet":model.nc,"nf":model.nf,"nd":model.nd,"geographic_max_dt_seconds":model.transport_step}
        receipt["gpu"]=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        receipt["initial_state"]="GFS-Wave partition reconstruction";receipt["boundary"]="time-interpolated incident spectra"
        receipt["source_parameter_sha256"]=model.source_parameter_sha256
        receipt["shape_table_sha256"]=model.partition_shapes_sha256
        receipt["git_commit"]=subprocess.run(["git","rev-parse","HEAD"],cwd=PACKAGE,check=True,capture_output=True,text=True,timeout=10).stdout.strip()
        receipt["science"]=json.loads((PACKAGE/"source_packs.json").read_text())["choices"]
        lat=host["lat"];lon=host["lon"];wet_host=cp.asnumpy(wet)
        names=[]
        for station in stations:
            encoded=station["id"].encode("ascii")
            if len(encoded)>16:raise ValueError("station identifier exceeds 16 bytes")
            names.append(encoded.ljust(16,b" "))
        ids=b"".join(names)
        station_lat=np.asarray([s["lat"] for s in stations],dtype=np.float32)
        station_lon=np.asarray([s["lon"] for s in stations],dtype=np.float32)
        frequency=cp.asnumpy(model.frequency)
        directions=cp.asnumpy((270-cp.arange(model.nd,dtype=cp.float32)*(360/model.nd))%360)
        forcing_index=boundary_cursor=-1;graphs={};frame=0;write_wall=0.;run_start=time.perf_counter()
        while True:
            fi=min(len(times)-2,max(0,bisect.bisect_right(times,when)-1))
            bi=min(len(btimes)-2,max(0,bisect.bisect_right(btimes,when)-1))
            if fi!=forcing_index:
                model.set_wind_bracket(device["u10_east"][fi].reshape(shape),device["v10_north"][fi].reshape(shape),
                    device["u10_east"][fi+1].reshape(shape),device["v10_north"][fi+1].reshape(shape),time0=times[fi],time1=times[fi+1])
                forcing_index=fi
            if bi!=boundary_cursor:
                model.set_boundary_partitions(boundary[bi],boundary[bi+1],time0=btimes[bi],time1=btimes[bi+1],shape_table=table)
                boundary_cursor=bi
            if abs(when-output_times[frame])<1e-6:
                # Classify partitions against the wind at this valid time,
                # including t=0 before any prognostic step has run.
                model.refresh_outputs(refresh_stress=True)
                summary=model.collect();stage=time.perf_counter();bulk=cp.asnumpy(model.bulk)
                leaf=f"waves-{frame:04}.nc"
                if io.lib.waves_v1_snapshot(os.fsencode(output/leaf),lat.ctypes.data,lon.ctypes.data,
                    wet_host.ctypes.data,ny,nx,when,bulk.ctypes.data):raise RuntimeError("Rust CF snapshot write failed")
                if stations:
                    model.station_outputs();station_bulk=cp.asnumpy(model._station_bulk);efth=cp.asnumpy(model.station_spectra())
                    if io.lib.waves_v1_points(os.fsencode(output/"points.nc"),ctypes.cast(ctypes.c_char_p(ids),ctypes.c_void_p),
                        station_lat.ctypes.data,station_lon.ctypes.data,frequency.ctypes.data,directions.ctypes.data,
                        len(stations),len(output_times),model.nf,model.nd,frame,when,station_bulk.ctypes.data,efth.ctypes.data):
                        raise RuntimeError("Rust station write failed")
                if budget_points:
                    b=model._budget;cells=b["cells"]
                    angle=cp.arange(model.nd,dtype=cp.float64)*360/model.nd
                    heading=cp.asarray([p["heading_math_deg"] for p in budget_points])
                    sector=(cp.abs((angle[None,:]-heading[:,None]+180)%360-180)<=30)
                    totals=b["totals"]
                    spectral=cp.sum(totals,axis=-1)*(2*math.pi/model.nd)
                    directional=cp.sum(totals*sector[None,:,None,:],axis=-1)*(2*math.pi/model.nd)
                    efth=b["previous"]
                    item={"epoch":when,"frequency_hz":frequency.tolist(),"df_hz":cp.asnumpy(model.df).tolist(),
                        "points":budget_points,"terms":b["terms"],"cumulative_omni_m2_per_hz":cp.asnumpy(spectral).tolist(),
                        "cumulative_sector_m2_per_hz":cp.asnumpy(directional).tolist(),
                        "efth_m2_hz_rad":cp.asnumpy(efth).tolist(),"sector_halfwidth_deg":30,
                        "u10_east":cp.asnumpy(model.u[cells]).tolist(),"v10_north":cp.asnumpy(model.v[cells]).tolist(),
                        "moments":cp.asnumpy(model.moments[cells]).tolist(),
                        "bulk":{name:bulk[j,[p["cell"] for p in budget_points]].tolist() for j,name in enumerate(PRODUCTS)}}
                    (output/"budget"/f"{frame:04}.json").write_text(json.dumps(item)+"\n")
                write_wall+=time.perf_counter()-stage
                entry={"path":leaf,"epoch":when,"valid":datetime.fromtimestamp(when,timezone.utc).isoformat(),
                    "max_hs_m":float(cp.max(model.outputs.hs)),"summary":summary}
                receipt["frames"].append(entry);save();print(json.dumps(entry),flush=True);frame+=1
                if when>=end-1e-6:break
            dt=min(step,end-when,output_times[frame]-when,times[fi+1]-when,btimes[bi+1]-when)
            if dt<=0:raise RuntimeError("prepared scheduling did not advance")
            if dt not in graphs:
                # One asynchronous graph per actual interval. All fields, clocks,
                # endpoint storage and output buffers keep stable device addresses.
                stream=cp.cuda.Stream(non_blocking=True)
                stream.wait_event(cp.cuda.get_current_stream().record())
                with stream:
                    stream.begin_capture();model.step(dt);graph=stream.end_capture()
                cp.cuda.get_current_stream().wait_event(stream.record())
                graphs[dt]=graph
            graphs[dt].launch(stream=cp.cuda.get_current_stream());when+=dt
        summary=model.collect();receipt["stages"].update(simulate_wall_seconds=time.perf_counter()-run_start,
            write_wall_seconds=write_wall,device_wall_seconds=summary["gpu_seconds"])
        receipt["summary"]=summary;receipt["status"]="DONE"
        # Product/verification modules are lane-owned. An explicit configured
        # postprocess command is an argv list, never a shell string.
        for command in cfg.get("postprocess",[]):
            if not isinstance(command,list) or not all(isinstance(s,str) for s in command):raise ValueError("postprocess requires argv lists")
            subprocess.run([s.replace("{output}",str(output)).replace("{prepared}",str(prepared)) for s in command],check=True,timeout=1800)
    except BaseException as exc:
        receipt["status"]="FAILED";receipt["error"]=f"{type(exc).__name__}: {exc}";raise
    finally:
        receipt["wall_seconds"]=time.perf_counter()-started;receipt["cpu_seconds"]=time.process_time()-cpu_start;save()
        signal.signal(signal.SIGTERM,old_sigterm)
    return 0
