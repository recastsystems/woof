"""Coupled CF output collection and registration in the normal render tree."""
from __future__ import annotations

from datetime import datetime,timezone
import json
from pathlib import Path
import time


class WaveOutput:
    def __init__(self, folder, coupler):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.coupler=coupler;self.frames=[];self.times=[];self.point_spectra=[];self.point_fields={}
        self.wall_seconds=0.

    def collect(self):
        import cupy as cp
        import numpy as np
        from woof.io.nc_writer_bridge import ClassicSchema
        from .door import rows
        solver=self.coupler.waves
        epoch=float(solver.time_seconds)
        if self.times and epoch==self.times[-1]:return
        started=time.perf_counter();grid=self.coupler.grid
        path=self.folder/f"waves-{len(self.frames):05d}.nc"
        schema=ClassicSchema("cdf2")
        t=schema.def_dim("time",1);y=schema.def_dim("y",grid.ny);x=schema.def_dim("x",grid.nx)
        schema.put_global_attr("Conventions","CF-1.10")
        schema.put_global_attr("wave_direction_convention","FROM clockwise from true north; direction retains legacy TOWARD convention")
        ti=schema.def_var("time","f8",(t,));schema.put_var_attr(ti,"units","seconds since 1970-01-01 00:00:00")
        lat=schema.def_var("latitude","f4",(y,x));lon=schema.def_var("longitude","f4",(y,x))
        schema.put_var_attr(lat,"units","degrees_north");schema.put_var_attr(lon,"units","degrees_east")
        wet=schema.def_var("wet","f4",(y,x))
        fields={key:getattr(solver.outputs,key) for key in rows()["products"] if hasattr(solver.outputs,key)}
        fields.update(u10_east=self.coupler.buffers['u10'],v10_north=self.coupler.buffers['v10'])
        ids={}
        for key in fields:
            ids[key]=schema.def_var(key,"f4",(t,y,x))
            row=rows()["products"].get(key,{"units":"m/s","title":key})
            schema.put_var_attr(ids[key],"units",row.get("storage_units",row["units"]))
            schema.put_var_attr(ids[key],"long_name",row["title"])
            schema.put_var_attr(ids[key],"coordinates","time latitude longitude")
            if key=="hs":schema.put_var_attr(ids[key],"standard_name","sea_surface_wave_significant_height")
        with schema.create(path) as writer:
            writer.write_var(ti,np.asarray([epoch],dtype=np.float64))
            writer.write_var(lat,cp.asnumpy(grid.latitude));writer.write_var(lon,cp.asnumpy(grid.longitude))
            writer.write_var(wet,cp.asnumpy(self.coupler.bathymetry.wet))
            for key,array in fields.items():writer.write_var(ids[key],cp.asnumpy(array))
        if self.coupler.station_metadata:
            self.point_spectra.append(solver.station_spectra().copy())
            for key,array in solver.station_outputs().items():
                if key in rows()["products"]:self.point_fields.setdefault(key,[]).append(array.copy())
        self.times.append(epoch)
        self.frames.append({"path":str(path),"valid_time":datetime.fromtimestamp(epoch,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
        self.wall_seconds+=time.perf_counter()-started
        self._manifest()

    def _manifest(self):
        (self.folder/"frames.json").write_text(json.dumps({"schema":"gpuwm.waves.coupled-output/v1",
            "frames":self.frames,"stations":self.coupler.station_metadata,"output_wall_seconds":self.wall_seconds,
            "prepared_identity":getattr(self.coupler,"prepared_identity",None),
            "preparation":getattr(self.coupler,"preparation_receipt",None)},indent=2)+"\n")

    def points(self, path):
        if not self.point_spectra:return None
        import cupy as cp
        import numpy as np
        from woof.io.nc_writer_bridge import ClassicSchema
        from .door import rows
        stations=self.coupler.station_metadata;solver=self.coupler.waves
        schema=ClassicSchema("cdf2")
        st=schema.def_dim("station",len(stations));t=schema.def_dim("time",len(self.times))
        f=schema.def_dim("frequency",solver.nf);d=schema.def_dim("direction",solver.nd)
        width=max(len(row["id"].encode()) for row in stations);ch=schema.def_dim("id_strlen",width)
        schema.put_global_attr("Conventions","CF-1.10");schema.put_global_attr("featureType","timeSeries")
        si=schema.def_var("station_id","S1",(st,ch));schema.put_var_attr(si,"cf_role","timeseries_id")
        ti=schema.def_var("time","f8",(t,));schema.put_var_attr(ti,"units","seconds since 1970-01-01 00:00:00")
        lat=schema.def_var("latitude","f4",(st,));lon=schema.def_var("longitude","f4",(st,))
        schema.put_var_attr(lat,"units","degrees_north");schema.put_var_attr(lon,"units","degrees_east")
        freq=schema.def_var("frequency","f4",(f,));schema.put_var_attr(freq,"units","Hz")
        direction=schema.def_var("direction","f4",(d,));schema.put_var_attr(direction,"units","degree")
        schema.put_var_attr(direction,"direction_convention","FROM clockwise from true north")
        efth=schema.def_var("efth","f4",(st,t,f,d));schema.put_var_attr(efth,"units","m2 Hz-1 rad-1")
        ids={}
        for key in self.point_fields:
            # The legacy bulk 'direction' would collide with the spectral
            # coordinate. Keep its documented meaning under an explicit name.
            name="legacy_direction" if key=="direction" else key
            ids[key]=schema.def_var(name,"f4",(st,t));schema.put_var_attr(ids[key],"units",rows()["products"][key]["units"])
            schema.put_var_attr(ids[key],"coordinates","latitude longitude time")
        with schema.create(path) as writer:
            raw=b"".join(row["id"].encode().ljust(width,b" ") for row in stations)
            writer.write_var(si,np.frombuffer(raw,dtype="S1"))
            writer.write_var(ti,np.asarray(self.times,dtype=np.float64))
            writer.write_var(lat,np.asarray([row["lat"] for row in stations],dtype=np.float32))
            writer.write_var(lon,np.asarray([row["lon"] for row in stations],dtype=np.float32))
            writer.write_var(freq,cp.asnumpy(solver.frequency))
            writer.write_var(direction,cp.asnumpy((270-cp.arange(solver.nd,dtype=cp.float32)*(360/solver.nd))%360))
            writer.write_var(efth,cp.asnumpy(cp.stack(self.point_spectra,axis=1)))
            for key,arrays in self.point_fields.items():writer.write_var(ids[key],cp.asnumpy(cp.stack(arrays,axis=1)))
        return str(path)


def select_render_frames(frames, times, *, origin, cadence):
    """Keep forecast-relative cadence across a checkpoint continuation."""
    return [row for row, epoch in zip(frames, times)
            if abs((epoch-origin)/cadence-round((epoch-origin)/cadence))<1e-8
            or row is frames[-1]]


def publish(physics, *, outdir, domain="d01", render=True,source_label="coupled atmosphere and waves"):
    ocean=getattr(physics,"ocean",None)
    collector=getattr(ocean,"output",None)
    if collector is None:return {"status":"not-run"}
    root=Path(outdir)/domain/"waves";root.mkdir(parents=True,exist_ok=True)
    points=collector.points(root/"points.nc")
    cadence=getattr(ocean,"render_interval",21600.)
    first=collector.times[0] if collector.times else 0.
    origin=getattr(ocean,"prepared_identity",{}).get("initial_valid_time",first)
    frames=select_render_frames(collector.frames,collector.times,origin=origin,cadence=cadence)
    request={"operation":"products","domain":domain,"output":str(root/"render-data"),
             "join_existing_output":True,
             "render_output":str(Path(outdir)),"source_label":source_label,
             "rows":str(Path(__file__).with_name("rows.json")),
             "products":["hs","tp","tm01","tm02","dspr","ws_hs","sw1_hs","sw2_hs","sw3_hs"],
             "frames":[{"path":row["path"],"valid":row["valid_time"]} for row in frames]}
    request_path=root/"products-request.json"
    request_path.write_text(json.dumps(request,indent=2)+"\n")
    result={"status":"collected","frames":len(collector.frames),"render_frames":len(frames),
            "render_interval_seconds":cadence,"points":points,
            "prepared_identity":getattr(ocean,"prepared_identity",None),
            "preparation":getattr(ocean,"preparation_receipt",None),
            "request":str(root/"products-request.json"),"output_wall_seconds":collector.wall_seconds}
    if collector.frames:
        from .waves.data import invoke
        work=invoke(request_path,operation="products")
        result["products"]=json.loads((work/"receipt.json").read_text())
        if render:
            from .door import render_ocean
            render_ocean(work/"render-request.json")
            rendered_path=root/'render-receipt.json'
            rendered=json.loads(rendered_path.read_text())
            maps=[row['path'] for row in rendered['products']]
            if len(maps)!=len(frames)*len(request['products']) or any(
                    not Path(p).is_file() or not Path(p).resolve().is_relative_to(root.resolve()) for p in maps):
                raise RuntimeError('wave renderer did not fill this run\'s complete product tree')
            result.update(renderer_receipt=str(rendered_path),maps=maps)
        result["status"]="rendered" if render else "registered"
    (root/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
