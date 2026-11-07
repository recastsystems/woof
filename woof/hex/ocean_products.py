"""Native ocean budget reductions on CUDA, Rust NetCDF and CF maps."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import subprocess

from .s2s_door import write_json, S2SError
from .s2s_products import validate_history, product_grid, TIME_FORMAT


def physical_renderer():
    from .engines import resolve, RENDERER
    renderer = resolve(RENDERER)
    result = subprocess.run([str(renderer), "--cf", "unused", "--help"], capture_output=True, text=True)
    if "CF physical-scale-v1" not in result.stdout + result.stderr:
        raise S2SError("ocean maps require engine-patches/cf-physical-scale.patch and rebuilt rw_wrfbatch")
    return renderer


def write_plane(path, cp, lat, lon, valid, fields, *, source="Coupled mixed-layer ocean and sea ice"):
    import numpy as np
    from woof.io.nc_writer_bridge import ClassicSchema
    schema = ClassicSchema()
    t = schema.def_dim("time", 1)
    y = schema.def_dim("lat", lat.shape[0])
    x = schema.def_dim("lon", lat.shape[1])
    schema.put_global_attr("Conventions", "CF-1.8")
    schema.put_global_attr("source", source)
    payloads = []
    def variable(name, units, data, dims, standard=None):
        number = schema.def_var(name, "f4", dims)
        schema.put_var_attr(number, "units", units)
        if standard:
            schema.put_var_attr(number, "standard_name", standard)
        payloads.append((number, np.ascontiguousarray(cp.asnumpy(data), dtype=np.float32)))
    variable("time", "seconds since "+valid.strftime("%Y-%m-%d %H:%M:%S"), cp.asarray([0]), (t,), "time")
    variable("lat", "degrees_north", lat[:,0], (y,), "latitude")
    variable("lon", "degrees_east", lon[0], (x,), "longitude")
    for name, (units, plane) in fields.items():
        variable(name, units, plane[None], (t,y,x))
    with schema.create(path) as writer:
        for number, payload in payloads:
            writer.write_var(number, payload)


def build_ocean_products(value, state, *, render=True):
    import cupy as cp
    from woof.netcdf_bridge import open_dataset
    from woof.rustwx import run_cf_renderer
    from .engines import resolve, RENDERER
    renderer = physical_renderer() if render else None
    started = time.monotonic()
    if value["surface"].get("ocean_mode") != "prognostic":
        raise S2SError("ocean response products require prognostic coupling")
    output = Path(value["paths"]["work"])/"ocean-products"
    output.mkdir(parents=True, exist_ok=True)
    with open_dataset(value["paths"]["grid"]) as ds:
        lat, lon = ds.variables["latCell"][:], ds.variables["lonCell"][:]
        weights = cp.asarray(ds.variables["areaCell"][:], cp.float64).ravel()
    ylats, xlons, index = product_grid(cp, lat, lon, 1)
    shape = ylats.shape
    colors = [[5,48,97],[33,102,172],[67,147,195],[146,197,222],[209,229,240],
              [247,247,247],[253,219,199],[244,165,130],[214,96,77],[178,24,43],[103,0,31]]
    table = output/"cf-ocean-products.json"
    rows = []
    for name, title, units, levels in (
        ("sst_change", "SST change from initialization", "K", [-8,-4,-2,-1,-.5,-.1,.1,.5,1,2,4,8]),
        ("ice_change", "Sea ice fraction change from initialization", "%", [-100,-50,-25,-10,-5,-1,1,5,10,25,50,100])):
        rows.append(dict(names=[name],standard_names=[],product=name,title=title,
                         default_units=units,display_units=units,unit_factors={units:1},
                         scale=dict(levels=levels,colors=colors)))
    write_json(table, rows)
    summaries, maps = [], []
    start = datetime.strptime(value["start_time"], TIME_FORMAT)
    for member in value["members"]:
        # The native initialization carries this mesh-static mask unchanged.
        # Prepared atmospheric input may already have been disposed of after
        # forecast startup; response products must not require that large file.
        with open_dataset(value["paths"]["static"]) as ds:
            ocean = cp.asarray(ds.variables["landmask"][:]).ravel() == 0
        w = cp.where(ocean, weights, 0)
        total = w.sum()
        schedule = validate_history(member,state["attempts"], start,value["days"],value["history_every_minutes"]//60)
        names = ("sst","ocean_area","ocean_volume","ocean_depth","ocean_melt",
                 "ocean_energy","ocean_entrainment","ocean_residual","ocean_initial_sst","ocean_initial_ice")
        for valid, path in schedule:
            with open_dataset(path) as ds:
                fields = {k: cp.asarray(ds.variables[k][:]).ravel() for k in names}
            if any(not bool(cp.isfinite(v[ocean]).all()) for v in fields.values()):
                raise S2SError(f"nonfinite ocean state in {path}")
            if bool(((fields["ocean_area"]<0)|(fields["ocean_area"]>1)|(fields["ocean_volume"]<0)).any()):
                raise S2SError(f"unphysical ice area or volume in {path}")
            delta = fields["sst"]-fields["ocean_initial_sst"]
            ice = 100*(fields["ocean_area"]-fields["ocean_initial_ice"])
            day = (valid-start).total_seconds()/86400
            row = dict(member=member["run_id"],day=day,valid=valid.isoformat(),cells=int(ocean.sum()),
                sst_change_min_K=float(delta[ocean].min()),sst_change_max_K=float(delta[ocean].max()),
                sst_change_mean_K=float((cp.where(ocean,delta,0)*w).sum()/total),
                ice_change_min_percent=float(ice[ocean].min()),ice_change_max_percent=float(ice[ocean].max()),
                ice_fraction_mean=float((fields["ocean_area"]*w).sum()/total),
                ice_volume_mean_m=float((fields["ocean_volume"]*w).sum()/total),
                depth_max_m=float(fields["ocean_depth"][ocean].max()),
                energy_residual_max_J_m2=float(cp.abs(fields["ocean_residual"][ocean]).max()),
                energy_residual_mean_J_m2=float((fields["ocean_residual"]*w).sum()/total),
                external_energy_mean_J_m2=float((fields["ocean_energy"]*w).sum()/total),
                entrainment_energy_mean_J_m2=float((fields["ocean_entrainment"]*w).sum()/total))
            summaries.append(row)
            # Every weekly endpoint and the final endpoint uses identical ladders.
            if render and (day in (7,14,21,28) or day == value["days"]):
                prefix = output/f"{member['run_id']}-day{day:02.0f}"
                data = prefix.with_suffix(".nc")
                write_plane(data,cp,ylats,xlons,valid,{
                    "sst_change":("K",cp.where(ocean,delta,cp.nan)[index].reshape(shape)),
                    "ice_change":("%",cp.where(ocean,ice,cp.nan)[index].reshape(shape))})
                for variable in ("sst_change","ice_change"):
                    receipt=output/f"{prefix.name}-{variable}.render.json"
                    images=output/f"{prefix.name}-{variable}"
                    result=run_cf_renderer(renderer,data,variable=variable,out_dir=images,
                        valid=valid.isoformat()+"Z",bounds="-180,180,-89,89",table=table,
                        size=(1100,700),receipt=receipt)
                    pictures = [str(p) for p in images.rglob("*.png")]
                    if result != 0 or not pictures:
                        raise S2SError(f"ocean renderer failed for {variable} at {valid}: rc={result}, images={pictures}")
                    maps.append(dict(member=member["run_id"],day=day,variable=variable,
                        receipt=str(receipt),images=pictures,
                        renderer_result=str(result)))
    with (output/"daily-ocean-budget.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    result=dict(schema="hex-ocean-response.v1",seconds=time.monotonic()-started,
        parallelism="CUDA reductions/remapping; Rust NetCDF; rw_wrfbatch through woof.rustwx",
        ocean_mask_source=str(value["paths"]["static"]),
        frames=len(summaries),maps=maps,daily=summaries,
        scope="coupled response demonstration; no seasonal skill or observed ice thickness qualification")
    write_json(output/"ocean-receipt.json", result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args()
    build_ocean_products(json.loads((args.work_dir/"s2s-plan.json").read_text()),
                         json.loads((args.work_dir/"s2s-state.json").read_text()))

if __name__ == "__main__":
    main()
