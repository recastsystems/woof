
"""Orchestrate Rust NetCDF I/O and its CUDA-seeded stored-word selection.

No meteorological quantity is recomputed. The selection operation is seeded
from tools/rustwx/crates/netcdf-writer/src/bin/nc_rewrite.rs::select_words.
Run the CUDA operation under the card queue. All changes are explicit JSON
metadata: dimension indices and integer global attributes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def select_words(value, axis, indices):
    import cupy as cp
    from woof.core.kernels import get_kernel
    a = np.ascontiguousarray(value)
    source = a.shape[axis]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= source:
        raise ValueError('selected indices must be unique and inside the dimension')
    shape = list(a.shape); shape[axis] = len(indices)
    inner = int(np.prod(a.shape[axis+1:]))
    size = int(np.prod(shape)) * a.dtype.itemsize
    src = cp.asarray(a.view(np.uint8).reshape(-1))
    dst = cp.empty(size,dtype=cp.uint8)
    ix = cp.asarray(indices,dtype=cp.int32)
    get_kernel('record_select','record_select_words')(((size+255)//256,), (256,),
        (src,dst,ix,np.uint64(size),np.int32(a.dtype.itemsize),np.int32(source),
         np.int32(len(indices)),np.uint64(inner)))
    return dst.get().view(a.dtype).reshape(shape)

def reconfigure(source, output, plan, receipt):
    from woof import netcdf_bridge
    from woof.io.nc_writer_bridge import ClassicSchema
    selections = plan.get('dimension_indices',{})
    attributes = plan.get('global_ints',{})
    with netcdf_bridge.open_dataset(source) as ds:
        ds.set_auto_maskandscale(False)
        schema = ClassicSchema("cdf5")
        dimensions = {}
        for key, dim in ds.dimensions.items():
            if key in selections and dim.isunlimited():
                raise ValueError('record dimensions cannot be selected')
            dimensions[key] = schema.def_dim(key, len(selections.get(key,range(len(dim)))),
                                             unlimited=dim.isunlimited())
        for key in selections:
            if key not in dimensions:
                raise ValueError('selection names a missing dimension: '+key)
        for key in ds.ncattrs():
            if key not in attributes:
                schema.put_global_attr(key,ds.getncattr(key))
        for key,value in attributes.items():
            schema.put_global_attr(key,np.array([value],dtype=np.int32))
        variables = {}
        for key,var in ds.variables.items():
            variables[key] = schema.def_var(key,"S1" if var.is_character else var.dtype,[dimensions[d] for d in var.dimensions])
            for attr in var.ncattrs():
                schema.put_var_attr(variables[key],attr,var.getncattr(attr))
        rows = {}
        with schema.create(output) as writer:
            for key,var in ds.variables.items():
                value = np.ascontiguousarray(var[:],dtype=("S1" if var.is_character else var.dtype))
                original = hashlib.sha256(value.tobytes()).hexdigest()
                for axis,dimension in enumerate(var.dimensions):
                    if dimension in selections:
                        value = select_words(value,axis,selections[dimension])
                if key in plan.get('variable_values', {}):
                    value = np.asarray(plan['variable_values'][key],dtype=var.dtype).reshape(value.shape)
                rows[key] = dict(shape=list(value.shape),source_sha256=original,
                                 output_sha256=hashlib.sha256(value.tobytes()).hexdigest())
                if var.dimensions and ds.dimensions[var.dimensions[0]].isunlimited():
                    for record in range(value.shape[0]):
                        writer.write_record(record,variables[key],value[record])
                else:
                    writer.write_var(variables[key],value)
    receipt.write_text(json.dumps(dict(source=str(source),output=str(output),plan=plan,fields=rows),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path); ap.add_argument('output',type=Path)
    ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args(); reconfigure(a.source,a.output,json.loads(a.plan.read_text()),a.receipt)
