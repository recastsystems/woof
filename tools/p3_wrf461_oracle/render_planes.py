"""Wrap Rust-computed P3 planes in a Rust-written WRF history view."""
import argparse,json
from pathlib import Path
import numpy as np
from woof.netcdf_bridge import Dataset
from woof.io.classic_tape import ClassicTape

def main():
 ap=argparse.ArgumentParser();ap.add_argument('wrfout',type=Path);ap.add_argument('planes',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
 meta=json.loads((a.planes/'planes.json').read_text())
 with Dataset(a.wrfout) as src:
  src.set_auto_maskandscale(False);out=ClassicTape(a.out);values={}
  for name,d in src.dimensions.items():out.createDimension(name,None if name=='Time' else len(d))
  out.setncatts({name:src.getncattr(name) for name in src.ncattrs()})
  for name,v in src.variables.items():
   dst=out.createVariable(name,np.dtype('S1') if v.is_character else v.dtype,v.dimensions);dst.setncatts({k:v.getncattr(k) for k in v.ncattrs()})
   data=v[:];values[name]=data[meta['time_index']:meta['time_index']+1] if 'Time' in v.dimensions else data
  for name,units,definition in zip(meta['names'],meta['units'],meta['definitions']):
   dst=out.createVariable(name,'f4',('Time','south_north','west_east'));dst.setncatts(dict(units=units,description=definition,MemoryOrder='XY',stagger='',FieldType=np.int32(104)))
   values[name]=np.fromfile(a.planes/(name+'.bin'),dtype='<f4').reshape(1,*meta['shape'])
  for name,value in values.items():
   out.variables[name][0 if 'Time' in out.variables[name].dimensions else slice(None)]=value[0] if 'Time' in out.variables[name].dimensions else value
  out.close()
if __name__=='__main__':main()
