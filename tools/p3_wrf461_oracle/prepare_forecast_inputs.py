"""Add source-absent P3 package fields through the Rust NetCDF seams."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from woof.netcdf_bridge import Dataset
from woof.io.classic_tape import ClassicTape
from woof.wrf_physics_inventory import _INVENTORIES


def extend(source, target, mp, boundary=False):
    receipt={}
    with Dataset(source) as src:
        src.set_auto_maskandscale(False)
        out=ClassicTape(target)
        for name,dim in src.dimensions.items():
            out.createDimension(name,None if name=='Time' else len(dim))
        attrs={name:src.getncattr(name) for name in src.ncattrs()}
        attrs['MP_PHYSICS']=np.int32(mp)
        out.setncatts(attrs)
        values={}
        for name,var in src.variables.items():
            value=var[:]
            v=out.createVariable(name,value.dtype if var.is_character else var.dtype,var.dimensions)
            v.setncatts({key:var.getncattr(key) for key in var.ncattrs()})
            values[name]=value
            receipt[name]=dict(policy='unchanged',sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
        for row in _INVENTORIES[mp].wrfinput_fields:
            suffixes=('_BXS','_BXE','_BYS','_BYE','_BTXS','_BTXE','_BTYS','_BTYE') if boundary else ('',)
            for suffix in suffixes:
                name=row.netcdf_name+suffix
                if name in values: continue
                ref=src.variables['QVAPOR'+suffix] if boundary else src.variables['QVAPOR']
                v=out.createVariable(name,'f4',ref.dimensions)
                v.setncatts(dict(units=row.units,description=row.registry_name,MemoryOrder='XYZ',stagger='',FieldType=np.int32(104)))
                value=np.zeros(ref.shape,dtype=np.float32)
                values[name]=value
                receipt[name]=dict(policy='source-absent-zero',words=int(value.size),sha256=hashlib.sha256(value.tobytes()).hexdigest())
        for name,value in values.items():
            if 'Time' in out.variables[name].dimensions:
                for t in range(value.shape[0]): out.variables[name][t]=value[t]
            else: out.variables[name][:]=value
        out.close()
    with Dataset(target) as dst:
        dst.set_auto_maskandscale(False)
        for name,row in receipt.items():
            digest=hashlib.sha256(np.ascontiguousarray(dst.variables[name][:], dtype=values[name].dtype).tobytes()).hexdigest()
            if digest!=row['sha256']: raise RuntimeError(f'Rust round-trip changed {name}')
    return receipt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    ap.add_argument('namelist',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--mp-physics',type=int,choices=(50,52),required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    import re
    text=a.namelist.read_text()
    for key,value in (("mp_physics",a.mp_physics),("cu_physics",0),
                      ("history_interval_s",300),("run_seconds",600)):
        text,count=re.subn(r"(?m)^(\s*"+key+r"\s*=\s*)[^,\n]+",
                           lambda m:m[1]+str(value),text)
        if count!=1: raise ValueError(f"namelist must contain exactly one {key}")
    text=re.sub(r"(?m)^.*iofields_filename.*\n","",text)
    (a.out/'namelist.input').write_text(text)
    result=dict(mp_physics=a.mp_physics,source=str(a.source),variables={})
    for name in ('wrfinput_d01','wrfbdy_d01'):
        result['variables'][name]=extend(a.source/name,a.out/name,a.mp_physics,name.startswith('wrfbdy'))
    (a.out/'input-receipt.json').write_text(json.dumps(result,indent=1))


if __name__=='__main__': main()
