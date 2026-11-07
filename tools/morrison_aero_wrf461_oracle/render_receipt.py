"""Prepare a first-level diagnostic view with the existing Rust I/O seams.

Usage: render_receipt.py WRFOUT RENDER_INPUT RECEIPT.json
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from woof.netcdf_bridge import open_dataset
from woof.io.nc_writer_bridge import ClassicSchema

source=Path(sys.argv[1]); target=Path(sys.argv[2])
receipt={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'fields':{}}
with open_dataset(source) as src:
    src.set_auto_maskandscale(False)
    schema=ClassicSchema()
    dims={name:schema.def_dim(name,0 if dim.isunlimited() else len(dim),unlimited=dim.isunlimited()) for name,dim in src.dimensions.items()}
    for name in src.ncattrs():schema.put_global_attr(name,src.getncattr(name))
    entries={}
    for name,var in src.variables.items():
        dtype=np.dtype('S1') if var.is_character else var.dtype
        idx=schema.def_var(name,dtype,[dims[d] for d in var.dimensions])
        for attr in var.ncattrs():schema.put_var_attr(idx,attr,var.getncattr(attr))
        entries[name]=(idx,var.dimensions,dtype,var)
    diagnostic=src.variables['CCN1_GS']
    viewdims=tuple(d for d in diagnostic.dimensions if d!='bottom_top')
    idx=schema.def_var('CCN1_GS_LEVEL1','f4',[dims[d] for d in viewdims])
    schema.put_var_attr(idx,'description','CCN at S=0.02%, first mass level; inactive with aercu_opt=0')
    schema.put_var_attr(idx,'units','#/cm-3')
    schema.put_var_attr(idx,'FieldType',np.int32(104))
    schema.put_var_attr(idx,'stagger','')
    schema.put_var_attr(idx,'coordinates','XLONG XLAT XTIME')
    with schema.create(target) as writer:
        for name,(varid,vdims,dtype,var) in entries.items():
            data=np.ascontiguousarray(var[...],dtype=dtype)
            receipt['fields'][name]={'sha256':hashlib.sha256(data.tobytes()).hexdigest(),'shape':list(data.shape),'dtype':str(dtype)}
            if dtype.kind=='f':
                receipt['fields'][name].update(nonfinite=int(np.count_nonzero(~np.isfinite(data))),min=float(np.nanmin(data)),max=float(np.nanmax(data)))
            if vdims and src.dimensions[vdims[0]].isunlimited():
                for recno in range(data.shape[0]):writer.write_record(recno,varid,np.ascontiguousarray(data[recno]))
            else:writer.write_var(varid,data)
        values=np.ascontiguousarray(diagnostic[...],dtype=np.float32)
        axis=diagnostic.dimensions.index('bottom_top')
        # View selection only. Rust decoded the field and writes the result.
        selection=[slice(None)]*values.ndim; selection[axis]=0
        view=np.ascontiguousarray(values[tuple(selection)])
        for recno in range(view.shape[0]):writer.write_record(recno,idx,np.ascontiguousarray(view[recno]))
        receipt['first_level_view_sha256']=hashlib.sha256(view.tobytes()).hexdigest()
with open_dataset(target) as dst:
    dst.set_auto_maskandscale(False)
    for name,var in dst.variables.items():
        data=np.ascontiguousarray(var[...],dtype=np.dtype('S1') if var.is_character else var.dtype)
        want=receipt['first_level_view_sha256'] if name=='CCN1_GS_LEVEL1' else receipt['fields'][name]['sha256']
        if hashlib.sha256(data.tobytes()).hexdigest()!=want:raise RuntimeError('render preparation changed '+name)
receipt['render_input_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
receipt['preserved_fields']=len(receipt['fields'])
receipt['all_finite']=all(field.get('nonfinite',0)==0 for field in receipt['fields'].values())
Path(sys.argv[3]).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'all_finite':receipt['all_finite'],'preserved_fields':receipt['preserved_fields'],'ccn':receipt['fields']['CCN1_GS'],'rain':receipt['fields']['RAINNC'],'refl':receipt['fields'].get('REFL_10CM')}))
