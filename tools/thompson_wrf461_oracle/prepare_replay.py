"""Build a package-complete replay input through the Rust NetCDF bridges."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
import numpy as np
from woof import netcdf_bridge
from woof.io.nc_writer_bridge import ClassicSchema
from woof.wrf_physics_inventory import stock_wrf_physics_inventory

def augment(source: Path, target: Path, additions: dict, attrs: dict):
    with netcdf_bridge.open_dataset(source) as data:
        schema = ClassicSchema()
        dims = {name: schema.def_dim(name, len(dim), unlimited=dim.isunlimited())
                for name, dim in data.dimensions.items()}
        for key in data.ncattrs():
            value = attrs.get(key, data.getncattr(key))
            if isinstance(value, (int,float)): value = np.float32(value) if isinstance(value,float) else np.int32(value)
            schema.put_global_attr(key, value)
        variables={}
        for name, variable in data.variables.items():
            ident = schema.def_var(name, np.dtype("S1") if variable.is_character else variable.dtype, [dims[d] for d in variable.dimensions])
            for key in variable.ncattrs():
                value=variable.getncattr(key)
                if isinstance(value,(int,float)): value=np.float32(value) if isinstance(value,float) else np.int32(value)
                schema.put_var_attr(ident,key,value)
            variables[name]=(ident,np.dtype("S1") if variable.is_character else variable.dtype,variable.dimensions,variable[:])
        for name,(shape,units,description) in additions.items():
            if name in variables:continue
            ident=schema.def_var(name,np.float32,[dims[d] for d in shape])
            schema.put_var_attr(ident,"units",units)
            schema.put_var_attr(ident,"description",description)
            schema.put_var_attr(ident,"stagger","")
            variables[name]=(ident,np.dtype("f4"),shape,np.zeros(tuple(len(data.dimensions[d]) for d in shape),np.float32))
        with schema.create(target) as writer:
            for ident,dtype,shape,values in variables.values():
                values=np.asarray(values,dtype=dtype)
                if shape and data.dimensions[shape[0]].isunlimited():
                    for rec in range(values.shape[0]):writer.write_record(rec,ident,values[rec])
                else:writer.write_var(ident,values)
    with netcdf_bridge.open_dataset(source) as old, netcdf_bridge.open_dataset(target) as new:
        pairs={}
        for name,var in old.variables.items():
            got=np.asarray(new.variables[name][:],dtype=np.dtype("S1") if var.is_character else var.dtype)
            want=np.asarray(var[:],dtype=np.dtype("S1") if var.is_character else var.dtype)
            assert np.array_equal(got,want,equal_nan=True) if got.dtype.kind in "f" else np.array_equal(got,want),name
            pairs[name]=hashlib.sha256(got.tobytes()).hexdigest()
    return pairs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shared-input",type=Path,required=True)
    p.add_argument("--namelist",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    volume=("Time","bottom_top","south_north","west_east")
    fields={}
    for field in stock_wrf_physics_inventory(38).wrfinput_fields:
        if field.collection in ("moist","scalar"):
            fields[field.netcdf_name]=(volume,field.units,field.registry_name)
    for name in ("QNWFA2D","QNIFA2D"):
        fields[name]=(("Time","south_north","west_east"),"kg-1 s-1",name)
    hashes=augment(args.shared_input/"wrfinput_d01",args.out/"wrfinput_d01",fields,{"MP_PHYSICS":38})
    os.symlink(args.shared_input/"wrfbdy_d01",args.out/"wrfbdy_d01")
    text=args.namelist.read_text()
    text=re.sub(r"(mp_physics\s*=\s*)1,",r"\g<1>38,",text)
    text=re.sub(r"(history_interval_s\s*=\s*)15,",r"\g<1>300,",text)
    # The input is synthetic-aerosol initialization from WRF's own profile.
    text=text.replace("&physics","&physics\n use_aero_icbc = .false.,")
    (args.out/"namelist.input").write_text(text)
    receipt={"source":str(args.shared_input),"source_namelist":str(args.namelist),
             "unchanged_source_fields":hashes,"added_species":sorted(set(fields)-set(hashes)),
             "initialization":"source fields retained; absent package state starts at zero",
             "boundary":"original shared boundary file, symlinked read-only"}
    (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
if __name__=="__main__":main()
