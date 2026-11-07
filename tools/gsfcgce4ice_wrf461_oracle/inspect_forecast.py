from pathlib import Path
import hashlib,json,sys
import numpy as np
from woof.netcdf_bridge import open_dataset

root=Path(sys.argv[1]); output=Path(sys.argv[2])
files=sorted(root.glob('wrfout_d01*'))
required=['QHAIL','HAILNC','HAILNCV','PHYSC','PHYSE','PHYSD','PHYSS','PHYSM','PHYSF',
          'ACPHYSC','ACPHYSE','ACPHYSD','ACPHYSS','ACPHYSM','ACPHYSF',
          'RE_CLOUD_GSFC','RE_RAIN_GSFC','RE_ICE_GSFC','RE_SNOW_GSFC',
          'RE_GRAUPEL_GSFC','RE_HAIL_GSFC','PRECI3D','PRECS3D','PRECG3D','PRECH3D','PRECR3D',
          'PHYS_TOT','ACPHYS_TOT']
report={'files':[]}
for path in files:
    with open_dataset(path) as ds:
        fields={}
        for name,var in ds.variables.items():
            if var.is_character: continue
            a=np.ascontiguousarray(var[...], dtype=var.dtype)
            fields[name]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                          'nonfinite':int((~np.isfinite(a)).sum()),
                          'min':float(a.min()),'max':float(a.max()), 'units':var.attributes.get('units')}
        report['files'].append({'path':str(path),'mp_physics':ds.global_attributes.get('MP_PHYSICS'),
                                'fields':fields})
assert len(files) >= 2, 'initial and forecast history frames are required'
last=report['files'][-1]
assert last['mp_physics']==7
assert not (set(required)-set(last['fields'])),set(required)-set(last['fields'])
assert 'REFL_10CM' in last['fields']
assert all(rec['nonfinite']==0 for rec in last['fields'].values())
output.write_text(json.dumps(report,indent=1)+'\n')
print(json.dumps({'frames':len(files),'fields':len(last['fields']),
                 'mp_physics':7,'required_fields_present':True,'nonfinite':0}))
