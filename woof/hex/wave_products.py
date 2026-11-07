"""Wave native-cell and station output through the shared Rust NetCDF writer."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .waves import PRODUCTS


def units(name):
    if name.endswith("hs") or name == "z0":
        return "m"
    if name.endswith("tp") or name in ("tm01", "tm02"):
        return "s"
    if name in ("dp", "dir", "dspr", "direction") or name.endswith("dp"):
        return "degree"
    return "m s-1" if name == "ustar" else "1"


def render_workdir(forecast_dir):
    """The render scratch folder, as a resolved path.

    THE BREAKAGE THIS FIXES (hex 0.4.14): woof's NativeIO.load resolves each array path and requires it inside the
    folder it was given, which it does not resolve.  Under an output path with a symlink in it (a test host's and the
    files inside the prepared folder", after the forecast had finished (the door's products stage exited 2).
    """
    import tempfile
    return Path(tempfile.mkdtemp(prefix="render-", dir=Path(forecast_dir) / "waves")).resolve()


def write_frame(service):
    """Output cadence; device transfers complete before timing collection."""
    from woof.io.nc_writer_bridge import ClassicSchema
    cp = service.cp
    core_receipt = service.solver.collect()
    epoch = service.start_epoch+service.clock
    valid = datetime.fromtimestamp(epoch, timezone.utc)
    folder = service.output_root / "waves" / "native"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (valid.strftime("%Y%m%dT%H%M%SZ") + ".nc")
    schema = ClassicSchema()
    t = schema.def_dim("time", 1)
    c = schema.def_dim("nCells", service.n)
    schema.put_global_attr("Conventions", "CF-1.10")
    schema.put_global_attr("source", "woof spectral waves on native Voronoi cells")
    schema.put_global_attr("prepared_sha256", service.prepared.sha256)
    payloads = []

    def variable(name, unit, data, dims, dtype="f4", standard=None):
        index = schema.def_var(name, dtype, dims)
        schema.put_var_attr(index, "units", unit)
        if standard:
            schema.put_var_attr(index, "standard_name", standard)
        if dims == (t,c):
            schema.put_var_attr(index, "coordinates", "time latitude longitude")
        payloads.append((index, data))

    variable("time", "seconds since 1970-01-01 00:00:00", np.asarray([epoch], np.float64), (t,), "f8", "time")
    # Restore original mesh-file order so the ordinary hex converter can
    # consume this file beside that mesh, including column-reordered runs.
    lat = cp.rad2deg(cp.asarray(service.mesh.arrays["latCell"], dtype=cp.float64)).astype(cp.float32)
    lon = cp.rad2deg(cp.asarray(service.mesh.arrays["lonCell"], dtype=cp.float64)).astype(cp.float32)
    variable("latitude", "degrees_north", cp.asnumpy(lat[service.inverse_order]), (c,), standard="latitude")
    variable("longitude", "degrees_east", cp.asnumpy(lon[service.inverse_order]), (c,), standard="longitude")
    variable("area", "m2", cp.asnumpy(cp.asarray(service.mesh.arrays["areaCell"], dtype=cp.float32)[service.inverse_order]), (c,))
    variable("wet", "1", cp.asnumpy(service.bathy.wet[service.inverse_order]), (c,))
    for name in PRODUCTS:
        data = getattr(service.solver.outputs, name)
        standard = "sea_surface_wave_from_direction" if name in ("dp", "dir") else None
        variable(name, units(name), cp.asnumpy(data[service.inverse_order])[None], (t,c), standard=standard)
    with schema.create(path) as writer:
        for index, data in payloads:
            writer.write_var(index, data)
    write_points(service)
    service.collect_timing()
    receipt = service.receipt()
    receipt['core'] = core_receipt
    receipt.update(last_frame=str(path), valid_time=valid.isoformat(), frames=len(service.point_frames))
    (service.output_root / "waves" / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return path


def write_points(service):
    from woof.io.nc_writer_bridge import ClassicSchema
    cp = service.cp
    stations = service.prepared.manifest["stations"]
    if not stations:
        return
    frames = service.point_frames
    nf, nd = service.options["spectral"]["frequencies"], service.options["spectral"]["directions"]
    schema = ClassicSchema()
    st = schema.def_dim("station", len(stations))
    tm = schema.def_dim("time", len(frames))
    fr = schema.def_dim("frequency", nf)
    dr = schema.def_dim("direction", nd)
    width = max(1,max(len(s["id"].encode("ascii")) for s in stations))
    ch = schema.def_dim("id_strlen", width)
    schema.put_global_attr("Conventions", "CF-1.10")
    schema.put_global_attr("featureType", "timeSeries")
    payload = []

    def var(name, dtype, dims, data, unit=None, role=None):
        number = schema.def_var(name, dtype, dims)
        if unit:
            schema.put_var_attr(number, "units", unit)
        if role:
            schema.put_var_attr(number, "cf_role", role)
        payload.append((number,data))

    ids = b"".join(s["id"].encode("ascii").ljust(width,b" ") for s in stations)
    var("station_id", "S1", (st,ch), ids, role="timeseries_id")
    var("time", "f8", (tm,), np.asarray([f["epoch"] for f in frames],np.float64), "seconds since 1970-01-01 00:00:00")
    for name,key,unit in (("latitude","lat","degrees_north"),("longitude","lon","degrees_east")):
        var(name,"f4",(st,),np.asarray([s[key] for s in stations],np.float32),unit)
    var("model_cell", "i4", (st,), np.asarray([s["cell"] for s in stations], np.int32), "1")
    var("model_cell_distance", "f4", (st,), np.asarray([s.get("distance_km", float("nan")) for s in stations], np.float32), "km")
    schema.put_global_attr("station_mapping", json.dumps(stations, sort_keys=True))
    spectral=service.options["spectral"]
    frequency=cp.float32(spectral["first_hz"])*cp.power(cp.float32(spectral["ratio"]),cp.arange(nf,dtype=cp.float32))
    direction=(cp.float32(270)-cp.arange(nd,dtype=cp.float32)*cp.float32(360/nd))%cp.float32(360)
    var("frequency","f4",(fr,),cp.asnumpy(frequency),"Hz")
    var("direction","f4",(dr,),cp.asnumpy(direction),"degree")
    for key in frames[0]["fields"]:
        # The spectral coordinate owns the CF name direction. Preserve the
        # legacy toward diagnostic under an unambiguous station variable.
        name = "wave_toward_direction" if key == "direction" else key
        var(name,"f4",(st,tm),cp.asnumpy(cp.stack([f["fields"][key] for f in frames],axis=1)),units(key))
    var("efth","f4",(st,tm,fr,dr),cp.asnumpy(cp.stack([f["efth"] for f in frames],axis=1)),"m2 Hz-1 rad-1")
    path=service.output_root/"waves"/"points.nc"
    partial=path.with_suffix(".partial.nc")
    with schema.create(partial) as writer:
        for number,data in payload:
            writer.write_var(number,data)
    partial.replace(path)


def mapper_binary():
    from .engines import bundled_program, executable_name
    explicit = os.environ.get("WOOF_HEX_RW_WAVE_PRODUCTS")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError("named hex wave product mapper is absent: " + explicit)
        return path
    packaged = bundled_program("rw_hex_wave_products")
    if packaged:
        return Path(packaged)
    crate = Path(__file__).resolve().parents[2] / "rust" / "wave-products"
    path = crate / "target" / "release" / executable_name("rw_hex_wave_products")
    if path.is_file():
        return path
    found = shutil.which("rw_hex_wave_products")
    if found:
        return Path(found)
    raise FileNotFoundError(f"hex wave products need rw_hex_wave_products; build cargo build --release --manifest-path {crate / 'Cargo.toml'}")


def render(forecast_dir, mesh_path, out_case, *, domain="global", bounds=(-180,180,-89.5,89.5), size=(1200,750), table_path=None, products=None):
    """Native polygon lookup in Rust, followed by the normal CF Rust renderer."""
    available = ("hs", "tp", "ws_hs", "sw1_hs", "sw2_hs", "sw3_hs")
    products = available if products is None else tuple(products)
    if not products or len(set(products)) != len(products) or set(products) - set(available):
        raise ValueError("wave map products must be unique names from " + ", ".join(available))
    from woof.netcdf_bridge import open_dataset
    from woof.io.nc_writer_bridge import ClassicSchema
    from woof.ocean.waves.prepared_run import NativeIO
    from woof.rustwx import find_renderer, renderer_remedy, run_cf_renderer
    from importlib.resources import files
    forecast_dir, out_case = Path(forecast_dir), Path(out_case)
    frames = sorted((forecast_dir/"waves"/"native").glob("*.nc"))
    if not frames:
        return None
    mapper = mapper_binary()
    renderer = find_renderer()
    if renderer is None:
        raise FileNotFoundError(renderer_remedy())
    table = Path(table_path) if table_path else files("woof.hex").joinpath("data/wave-products.v1.json")
    native = NativeIO()
    work = render_workdir(forecast_dir)
    made = []
    deleted = []

    def raw(name, values, dtype="<f4"):
        path = work/(name+".bin")
        # The NetCDF reader may expose every numeric type as float64. Pack
        # the declared wire type explicitly; Rust owns all field transforms.
        path.write_bytes(np.ascontiguousarray(values, dtype=dtype).tobytes())
        made.append(path)
        return str(path)

    with open_dataset(mesh_path) as ds:
        polygons = {"vertices_on_cell": raw("vertices_on_cell", ds.variables["verticesOnCell"][:], "<i4"),
                    "edge_counts": raw("edge_counts", ds.variables["nEdgesOnCell"][:], "<i4"),
                    "vertex_latitude": raw("vertex_latitude", ds.variables["latVertex"][:], "<f8"),
                    "vertex_longitude": raw("vertex_longitude", ds.variables["lonVertex"][:], "<f8")}
    results = []
    began = time.perf_counter()
    for frame in frames:
        with open_dataset(frame) as ds:
            epoch = float(ds.variables["time"][:][0])
            if frame not in (frames[0],frames[-1]) and abs(epoch/21600-round(epoch/21600))>1.e-8:
                continue
            lat, lon, wet = (ds.variables[k][:] for k in ("latitude", "longitude", "wet"))
            fields = {key:raw(key,ds.variables[key][:]) for key in dict.fromkeys((*products,"hs","dp"))}
        valid = datetime.fromtimestamp(epoch,timezone.utc)
        grid = work / "grid"
        grid.mkdir(exist_ok=True)
        request = dict(latitude=raw("latitude",lat), longitude=raw("longitude",lon), wet=raw("wet",wet),
            **polygons, fields={k:fields[k] for k in products}, units={k:units(k) for k in products},
            height=fields["hs"], peak_direction=fields["dp"], bounds=None if bounds is None else list(bounds), nx=721, ny=361,
            output=str(grid), threads=os.cpu_count() or 1)
        request_path = work / "request.json"
        request_path.write_text(json.dumps(request))
        log = work / (frame.stem+".mapper.log")
        with log.open("w") as handle:
            subprocess.run([str(mapper),str(request_path)], stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        receipt = json.loads((grid/"receipt.json").read_text())
        # NativeIO's Rust loader verifies raw grid bytes; use the same entry
        # point without pretending these scratch products are prepared forcing.
        def load_grid(name,shape):
            import hashlib
            p=grid/(name+".bin")
            with p.open("rb") as stream:
                digest=hashlib.file_digest(stream,"sha256").hexdigest()
            return native.load(grid,dict(path=p.name,dtype="<f4",shape=shape,bytes=p.stat().st_size,sha256=digest))
        nx,ny=receipt["nx"],receipt["ny"]
        schema=ClassicSchema()
        t=schema.def_dim("time",1);y=schema.def_dim("lat",ny);x=schema.def_dim("lon",nx)
        schema.put_global_attr("Conventions","CF-1.10")
        schema.put_global_attr("source","Native-cell spectral waves")
        values=[]
        for name,unit,dims,data,dtype in (
            ("time","seconds since 1970-01-01 00:00:00",(t,),np.asarray([epoch],np.float64),"f8"),
            ("lat","degrees_north",(y,),load_grid("latitude",[ny]),"f4"),
            ("lon","degrees_east",(x,),load_grid("longitude",[nx]),"f4")):
            i=schema.def_var(name,dtype,dims);schema.put_var_attr(i,"units",unit);values.append((i,data))
        for name in products:
            i=schema.def_var(name,"f4",(t,y,x));schema.put_var_attr(i,"units",units(name))
            schema.put_var_attr(i,"coordinates","time lat lon")
            values.append((i,load_grid(name,[1,ny,nx])))
        plane=work/(frame.stem+".cf.nc")
        with schema.create(plane) as writer:
            for i,data in values:writer.write_var(i,data)
        for product in products:
            dest=out_case/domain/"waves"/product/valid.strftime("%Y-%m-%d")
            dest.mkdir(parents=True,exist_ok=True)
            render_receipt=dest/(frame.stem+".render.json")
            rendered=work/"rendered"/product/frame.stem
            code=run_cf_renderer(renderer,plane,variable=product,out_dir=rendered,
                valid=valid.isoformat(),bounds=",".join(map(str,receipt["bounds"])),table=Path(str(table)),size=size,
                overlays=grid/"directions.json" if product=="hs" else None,receipt=render_receipt)
            if code:
                raise RuntimeError(f"wave map renderer failed for {product}: exit {code}, {render_receipt}")
            result=json.loads(render_receipt.read_text())
            original_outputs=result["outputs"]
            published=[]
            for output in original_outputs:
                source=Path(output).resolve()
                if not source.is_relative_to(rendered.resolve()) or source.suffix != ".png":
                    raise ValueError("wave renderer output is outside its owned PNG folder")
                target=dest/source.name
                source.replace(target)
                published.append(str(target))
            result["renderer_outputs"]=original_outputs
            result["outputs"]=published
            render_receipt.write_text(json.dumps(result,indent=2)+"\n")
            results.append(dict(valid=valid.isoformat(),product=product,receipt=str(render_receipt),outputs=result["outputs"]))
        # These are owned transient render bytes, already represented by the
        # native CF frame and renderer receipts. Keep no per-frame duplicates.
        transient=[plane,*grid.glob("*.bin")]
        for path in transient:
            deleted.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    for path in set(made):
        if path.exists():
            deleted.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    record=dict(schema="gpuwm-hex.wave-products/v1",frames=len(frames),maps=results,
        seconds=time.perf_counter()-began,deleted_files=deleted,
        layout="<case>/<domain>/waves/<product>/<valid-day>/",
        remap="Rust native Voronoi cell constant, spherical polygon mask, no smoothing",
        renderer=str(renderer),mapper=str(mapper))
    target=out_case/domain/"waves"/"products-receipt.json"
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(record,indent=2)+"\n")
    points=forecast_dir/"waves"/"points.nc"
    if points.is_file():
        shutil.copy2(points,target.parent/"points.nc")
    return record
