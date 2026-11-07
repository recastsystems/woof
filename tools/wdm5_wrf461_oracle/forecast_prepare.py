"""Prepare a WDM family CLI check from the shared public analysis inputs.

NetCDF payloads are read and written by Rust. The original shared inputs
stay read-only. Added fields are WRF's zero-if-source-absent cold start.
The public WRF em_real namelist is retained as the template authority;
geometry, dates and physics are stated in the generated namelist receipt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import urllib.request
import numpy as np
from woof.netcdf_bridge import Dataset
from woof.io.nc_writer_bridge import ClassicSchema

URL="https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/test/em_real/namelist.input"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(shared, output, mp):
    output.mkdir(parents=True,exist_ok=True)
    template=output/"public-namelist.input"
    template.write_bytes(urllib.request.urlopen(URL,timeout=60).read())
    assert "mp_physics" in template.read_text()
    source=shared/"inputs/iowa-convective/A071"
    records=[]
    for name in ["wrfinput_d01","wrfbdy_d01"]:
        src=source/name;dst=output/name
        with Dataset(src) as ds:
            ds.set_auto_maskandscale(False)
            schema=ClassicSchema()
            dimids={n:schema.def_dim(n,len(d),unlimited=d.isunlimited()) for n,d in ds.dimensions.items()}
            for n in ds.ncattrs():
                v=ds.getncattr(n)
                schema.put_global_attr(n,np.int32(mp) if n=="MP_PHYSICS" else v)
            rows={}
            for n,v in ds.variables.items():
                dtype=np.dtype("S1") if v.is_character else v.dtype
                vid=schema.def_var(n,dtype,[dimids[d] for d in v.dimensions])
                for a in v.ncattrs():schema.put_var_attr(vid,a,v.getncattr(a))
                rows[n]=(vid,v.dimensions,dtype,v,None)
            added=[]
            active=["QICE","QSNOW","QNCLOUD","QNRAIN","QNCCN"]
            if mp==16:active.append("QGRAUP")
            if name.startswith("wrfinput"):
                proto=ds.variables["QCLOUD"]
                for n in active:
                    if n in rows:continue
                    vid=schema.def_var(n,proto.dtype,[dimids[d] for d in proto.dimensions])
                    for a in proto.ncattrs():
                        if a not in ("description","units"):schema.put_var_attr(vid,a,proto.getncattr(a))
                    schema.put_var_attr(vid,"description",f"WRF source-absent cold start {n}")
                    schema.put_var_attr(vid,"units","  kg(-1)" if n.startswith("QN") else "kg kg-1")
                    rows[n]=(vid,proto.dimensions,proto.dtype,None,proto.shape);added.append(n)
            else:
                for old,v in ds.variables.items():
                    if not old.startswith("QCLOUD_"):continue
                    for prefix in ["QICE","QSNOW"]+(["QGRAUP"] if mp==16 else []):
                        n=prefix+old[len("QCLOUD"):]
                        if n in rows:continue
                        vid=schema.def_var(n,v.dtype,[dimids[d] for d in v.dimensions])
                        for a in v.ncattrs():schema.put_var_attr(vid,a,v.getncattr(a))
                        rows[n]=(vid,v.dimensions,v.dtype,None,v.shape);added.append(n)
            with schema.create(dst) as writer:
                for n,(vid,dims,dtype,var,shape) in rows.items():
                    a=np.zeros(shape,dtype=dtype) if var is None else np.asarray(var[:],dtype=dtype)
                    if dims and ds.dimensions[dims[0]].isunlimited():
                        for r in range(a.shape[0]):writer.write_record(r,vid,np.ascontiguousarray(a[r]))
                    else:writer.write_var(vid,np.ascontiguousarray(a))
            # Native read-back checks every existing field after the copy.
            with Dataset(dst) as copied:
                for n in ds.variables:
                    dtype=np.dtype("S1") if ds.variables[n].is_character else ds.variables[n].dtype
                    a=np.asarray(ds.variables[n][:],dtype=dtype)
                    b=np.asarray(copied.variables[n][:],dtype=dtype)
                    assert a.tobytes()==b.tobytes(),n
                for n in added:assert not np.any(copied.variables[n][:]),n
            records.append(dict(source=str(src),source_sha256=sha(src),output=str(dst),
                                output_sha256=sha(dst),added_zero_fields=added,
                                original_fields_byte_equal=True))
    sweep=shared/"provenance/sweep"
    combos=json.loads((sweep/"combos.json").read_text())
    row=next(c for c in combos["combos"] if c["id"]=="A071")
    row["namelist"]["physics"]["mp_physics"]=mp
    row["namelist"]["physics"]["do_radar_ref"]=1
    row["namelist"]["physics"]["ccn_conc"]=1e8
    # Keep the matched public input's other physics and geometry settings.
    selected=output/"selected-combo.json"
    selected.write_text(json.dumps({"combos":[row]},indent=2))
    nml=subprocess.check_output(["python",str(sweep/"mknml.py"),str(selected),
                                str(sweep/"cases.json"),"A071","iowa-convective","51"],text=True)
    # Only initial and final history frames are needed for this check.
    import re
    nml=re.sub(r"(?m)^( history_interval_s\s*=).*",r"\1 300,",nml)
    nml=re.sub(r"(?m)^ iofields_filename.*\n","",nml)
    from woof.namelist_import import parse_namelist
    # Start from the public template, then overlay the exact forcing and
    # package settings. Retain every public-template key not overridden.
    generated=output/"generated-namelist.input"
    generated.write_text(nml)
    groups=parse_namelist(template)
    overrides=parse_namelist(generated)
    for group,values in overrides.items():groups.setdefault(group,{}).update(values)
    # Optional I/O customizations in the public template need no extra file.
    groups["time_control"].pop("iofields_filename",None)
    # WRF adds the bare minute cadence to its second component.
    for key in ("history_interval", "history_interval_d", "history_interval_h", "history_interval_m"):
        groups["time_control"][key] = [0]
    groups["time_control"]["history_interval_s"] = [300]
    groups["physics"].pop("physics_suite",None)  # explicit per-component selectors replace the suite
    def fmt(v):
        if isinstance(v,bool):return ".true." if v else ".false."
        if isinstance(v,str):return "'"+v.replace("'","''")+"'"
        return repr(v)
    lines=[]
    for group,values in groups.items():
        lines.append("&"+group)
        for key,values in values.items():lines.append(" "+key+" = "+", ".join(fmt(v) for v in values)+",")
        lines.extend(["/", ""])
    (output/"namelist.input").write_text("\n".join(lines))
    receipt=dict(public_template_url=URL,public_template_sha256=sha(template),
                 configuration_sha256=sha(output/"namelist.input"),mp_physics=mp,
                 geometry=[40,40,50],interval_seconds=15,run_seconds=300,
                 source="shared WRF real.exe input from public HRRR analysis",
                 source_absent_policy="ice, snow and number absent from the original warm-rain package start at zero",
                 fields=records)
    (output/"prepare-receipt.json").write_text(json.dumps(receipt,indent=2))
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("shared",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--mp",type=int,choices=[14,16],default=14)
    a=p.parse_args();print(json.dumps(prepare(a.shared,a.output,a.mp),indent=2))
