"""Build a declared source-absent package extension through Rust NetCDF I/O.

This check utility preserves every supplied input and boundary word. New
package fields start at zero. It does not claim that real.exe was rerun.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from woof.netcdf_bridge import open_dataset
from woof.io.nc_writer_bridge import ClassicSchema
from woof.wrf_physics_inventory import stock_wrf_physics_inventory
from woof.namelist_import import parse_namelist

parser=argparse.ArgumentParser()
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--recorded-namelist',type=Path,required=True)
parser.add_argument('--public-template',type=Path,required=True)
parser.add_argument('--mp-physics',type=int,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
inventory=stock_wrf_physics_inventory(args.mp_physics)
new_fields={field.netcdf_name:field for field in inventory.wrfinput_fields}
receipt={'mp_physics':args.mp_physics,'source_absent_initialization':'zero only for new package variables, including all lateral boundary values and tendencies','input_word_hashes':{},'added':{}}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def rewrite(filename,boundary):
    path=args.inputs/filename
    receipt[filename+'_source_sha256']=sha(path)
    with open_dataset(path) as src:
        src.set_auto_maskandscale(False)
        schema=ClassicSchema()
        dims={name:schema.def_dim(name,0 if dim.isunlimited() else len(dim),unlimited=dim.isunlimited()) for name,dim in src.dimensions.items()}
        for name in src.ncattrs():
            schema.put_global_attr(name,np.int32(args.mp_physics) if name=='MP_PHYSICS' else src.getncattr(name))
        entries={}
        for name,var in src.variables.items():
            dtype=np.dtype("S1") if var.is_character else var.dtype
            idx=schema.def_var(name,dtype,[dims[d] for d in var.dimensions])
            for attr in var.ncattrs():schema.put_var_attr(idx,attr,var.getncattr(attr))
            entries[name]=(idx,var.dimensions,dtype,var)
        additions=[]
        for name,field in new_fields.items():
            for suffix in ('BXS','BXE','BYS','BYE','BTXS','BTXE','BTYS','BTYE') if boundary else ('',):
                target=name+'_'+suffix if suffix else name
                if target in entries:continue
                prototype=src.variables['QVAPOR_'+suffix] if boundary else src.variables['QVAPOR']
                idx=schema.def_var(target,'f4',[dims[d] for d in prototype.dimensions])
                for attr in prototype.ncattrs():
                    value=prototype.getncattr(attr)
                    if attr=='description':value='source-absent '+name
                    if attr=='units':value=field.units
                    schema.put_var_attr(idx,attr,value)
                entries[target]=(idx,prototype.dimensions,np.dtype('f4'),None)
                additions.append(target)
        with schema.create(args.out/filename) as writer:
            for name,(idx,vdims,dtype,var) in entries.items():
                if var is None:
                    shape=tuple(len(src.dimensions[d]) for d in vdims)
                    data=np.zeros(shape,dtype=dtype)
                else:data=np.ascontiguousarray(var[...],dtype=dtype)
                if vdims and src.dimensions[vdims[0]].isunlimited():
                    for recno in range(data.shape[0]):writer.write_record(recno,idx,np.ascontiguousarray(data[recno]))
                else:writer.write_var(idx,data)
                if var is not None:receipt['input_word_hashes'][filename+':'+name]=hashlib.sha256(data.tobytes()).hexdigest()
        receipt['added'][filename]=additions
    # Independent read-back through Rust for every original variable.
    with open_dataset(args.out/filename) as dst:
        dst.set_auto_maskandscale(False)
        for name,var in dst.variables.items():
            key=filename+':'+name
            if key not in receipt['input_word_hashes']:continue
            data=np.ascontiguousarray(var[...],dtype=np.dtype("S1") if var.is_character else var.dtype)
            if hashlib.sha256(data.tobytes()).hexdigest()!=receipt['input_word_hashes'][key]:
                raise RuntimeError('supplied field changed during package extension: '+key)
    receipt[filename+'_derived_sha256']=sha(args.out/filename)

rewrite('wrfinput_d01',False)
rewrite('wrfbdy_d01',True)

# Public WRF template, with the supplied case's grid/time/settings as explicit overrides.
public=parse_namelist(args.public_template)
recorded=parse_namelist(args.recorded_namelist)
for group,settings in recorded.items():public.setdefault(group,{}).update(settings)
receipt['removed_template_settings']={'physics_suite':public['physics'].pop('physics_suite',None)}
public['physics']['mp_physics']=[args.mp_physics]
public['physics']['aercu_opt']=[0]
public['time_control']['history_interval_s']=[300]
public['time_control'].pop('history_interval',None)
public['time_control'].pop('iofields_filename',None)
public['time_control'].pop('ignore_iofields_warning',None)
def literal(value):
    if isinstance(value,bool):return '.true.' if value else '.false.'
    if isinstance(value,str):return "'"+value.replace("'","''")+"'"
    return str(value)
text=''
for group,settings in public.items():
    text+='&'+group+'\n'
    for key,values in settings.items():text+=' '+key+' = '+', '.join(literal(value) for value in values)+',\n'
    text+='/\n'
(args.out/'namelist.input').write_text(text)
receipt['public_template_sha256']=sha(args.public_template)
receipt['recorded_namelist_sha256']=sha(args.recorded_namelist)
receipt['public_template']='https://github.com/wrf-model/WRF/blob/v4.6.1/test/em_real/namelist.input'
receipt['override_groups']=sorted(recorded)
(args.out.parent/'forecast-input-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'added':receipt['added'],'preserved_fields':len(receipt['input_word_hashes'])}))
