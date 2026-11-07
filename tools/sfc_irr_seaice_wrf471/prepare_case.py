"""Prepare a copied real CONUS analysis with named controlled option inputs."""
from pathlib import Path
import argparse,shutil
import numpy as np
import re
from netCDF4 import Dataset
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('native',type=Path);p.add_argument('out',type=Path);p.add_argument('--irrigation',type=int,default=0);p.add_argument('--ua',action='store_true');p.add_argument('--fractional',action='store_true');p.add_argument('--land',type=int,default=2);p.add_argument('--albedo',type=int,default=0);p.add_argument('--thickness',type=int,default=0);p.add_argument('--snowdepth',type=int,default=0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
for src in (a.native/'run').iterdir():
 if src.is_file() and src.name not in ('namelist.input','wrf.exe'):
  target=a.out/src.name
  if not target.exists():target.symlink_to(src.resolve())
for name in ('wrfinput_d01','wrfbdy_d01'):shutil.copyfile(a.source/name,a.out/name)
with Dataset(a.out/'wrfinput_d01','r+') as d:
 d.setncattr('SF_SURFACE_PHYSICS', a.land)
 dims=('Time','south_north','west_east');shape=d.variables['TSK'].shape
 def field(name,value,desc):
  v=d.createVariable(name,'f4',dims) if name not in d.variables else d.variables[name]
  v[:]=np.broadcast_to(np.asarray(value,np.float32),shape);v.FieldType=104;v.MemoryOrder='XY ';v.description=desc;v.units='';v.stagger='';v.coordinates='XLONG XLAT'
 ice=np.asarray(d.variables['SEAICE'][:]);
 if a.fractional:
  ice=np.where(ice>=.02,np.float32(.5),ice);d.variables['SEAICE'][:]=ice
 snowh=np.asarray(d.variables['SNOWH'][:]);lu=np.asarray(d.variables['LU_INDEX'][:])
 if a.irrigation:
  field('IRRIGATION',np.where(np.isin(lu,[12,14]),20,0),'CONTROLLED irrigated land percentage, crop categories only')
  v=d.createVariable('IRR_PH_FIELD','i4',dims);v[:]=0;v.FieldType=106;v.MemoryOrder='XY ';v.description='CONTROLLED phase zero';v.stagger=''
 def found(name):
  v=d.variables[name] if name in d.variables else d.createVariable(name,'i4',('Time',))
  v[:]=1
 if a.albedo==2:
  field('ALBSI',.65,'CONTROLLED prescribed ice albedo');found('FNDALBSI')
 if a.thickness:
  field('ICEDEPTH',3.,'CONTROLLED prescribed sea ice thickness, m');found('FNDICEDEPTH')
 if a.snowdepth:
  field('SNOWSI',np.maximum(snowh,.001),'CONTROLLED prescribed snow depth from real SNOWH, m');found('FNDSNOWSI')
 print('analysis ice cells (controlled fraction .5 when requested)',int((ice>=.02).sum()),'fractional ice cells',int(((ice>=.02)&(ice<1)).sum()),'analysis snow cells',int((snowh>0).sum()))
text=(a.source/'namelist.input').read_text()
def update(section, **values):
 global text
 m=re.search(r'(?ms)^\s*&'+section+r'\b(.*?)^\s*/',text);assert m,section
 body=m.group(1)
 for key,value in values.items():
  if isinstance(value,bool): value='.true.' if value else '.false.'
  elif isinstance(value,str): value=repr(value)
  else: value=str(value)
  line=' '+key+' = '+value+',\n'
  pattern=r'(?m)^\s*'+key+r'\s*=.*$'
  body=re.sub(pattern,line.rstrip(),body) if re.search(pattern,body) else body+line
 text=text[:m.start(1)]+body+text[m.end(1):]
update('time_control',run_days=0,run_hours=0,run_minutes=0,run_seconds=60,end_day=26,end_hour=12,end_minute=1,end_second=0,history_interval_m=1,frames_per_outfile=2,restart=False)
update('domains',time_step=20,nproc_x=4,nproc_y=4)
update('physics',physics_suite='none',mp_physics=8,ra_lw_physics=4,ra_sw_physics=4,sf_sfclay_physics=1,sf_surface_physics=a.land,bl_pbl_physics=1,cu_physics=6,num_soil_layers=4,ua_phys=a.ua,fractional_seaice=int(a.fractional),seaice_albedo_opt=a.albedo,seaice_thickness_opt=a.thickness,seaice_snowdepth_opt=a.snowdepth)
if a.irrigation:update('physics',sf_surf_irr_scheme=a.irrigation,irr_daily_amount=5.,irr_start_hour=12,irr_num_hours=1,irr_start_julianday=1,irr_end_julianday=366,irr_freq=1,irr_ph=0)
(a.out/'namelist.input').write_text(text);(a.out/'captures').mkdir(exist_ok=True)
