"""Orchestrate existing Rust NetCDF reader/writer for explicit test statics."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
from woof.netcdf_bridge import Dataset
from woof.io.nc_writer_bridge import ClassicSchema

source=Path(sys.argv[1]); namelist=Path(sys.argv[2]); root=Path(sys.argv[3])
root.mkdir(exist_ok=True,parents=True)
receipt={'source_sha256':hashlib.sha256((source/'wrfinput_d01').read_bytes()).hexdigest(),'fixture':'public recorded forecast input plus explicitly supplied full irrigation fraction and phase zero','options':{}}
for option in (1,2,3):
    target=root/str(option); target.mkdir(exist_ok=True)
    with Dataset(source/'wrfinput_d01') as old:
        schema=ClassicSchema(); dims={name:schema.def_dim(name,len(dim)) for name,dim in old.dimensions.items()}
        for name in old.ncattrs():schema.put_global_attr(name,old.getncattr(name))
        ids={}
        for name,v in old.variables.items():
            ids[name]=schema.def_var(name,'S1' if v.is_character else v.dtype,[dims[d] for d in v.dimensions])
            for attr in v.ncattrs():schema.put_var_attr(ids[name],attr,v.getncattr(attr))
        axis=['Time','south_north','west_east']; shape=tuple(len(old.dimensions[d]) for d in axis)
        ids['IRRIGATION']=schema.def_var('IRRIGATION','f4',[dims[d] for d in axis])
        ids['IRR_PH_FIELD']=schema.def_var('IRR_PH_FIELD','i4',[dims[d] for d in axis])
        with schema.create(target/'wrfinput_d01') as writer:
            for name,v in old.variables.items():
                value=np.asarray(v[:],dtype='S1' if v.is_character else v.dtype)
                if v.is_character:
                    value=np.asarray(value,dtype='S1')
                writer.write_var(ids[name],value)
            writer.write_var(ids['IRRIGATION'],np.ones(shape,np.float32))
            writer.write_var(ids['IRR_PH_FIELD'],np.zeros(shape,np.int32))
    shutil.copyfile(source/'wrfbdy_d01',target/'wrfbdy_d01')
    text=namelist.read_text(); text=text.replace('&physics',f'&physics\n sf_surf_irr_scheme = {option},\n irr_daily_amount = 25.0,\n irr_start_hour = 0,\n irr_num_hours = 24,\n irr_start_julianday = 0,\n irr_end_julianday = 366,\n irr_freq = 1,\n irr_ph = 0,',1)
    (target/'namelist.input').write_text(text)
    receipt['options'][str(option)]={'namelist_sha256':hashlib.sha256(text.encode()).hexdigest(),'wrfinput_sha256':hashlib.sha256((target/'wrfinput_d01').read_bytes()).hexdigest()}
(root/'fixture-receipt.json').write_text(json.dumps(receipt,indent=2))
