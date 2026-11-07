"""Adapt the public WRF em_real namelist to an existing native input grid."""
from pathlib import Path
from datetime import datetime,timedelta
import argparse,json
from woof.namelist_import import parse_namelist
from woof import netcdf_bridge
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--metgrid',type=Path);a=p.parse_args()
groups=parse_namelist(a.source)
groups={k:{key:val[:1] for key,val in v.items()} for k,v in groups.items()}
with netcdf_bridge.open_dataset(a.input) as ds:
 sizes={k:len(v) for k,v in ds.dimensions.items()}
 attrs={k:ds.getncattr(k) for k in ds.ncattrs()}
 start=datetime.strptime(str(attrs['START_DATE'])[:19],'%Y-%m-%d_%H:%M:%S')
 top=float(ds.variables['P_TOP'][:].reshape(-1)[0])
 eta=ds.variables['ZNW'][:].reshape(-1).tolist()
end=start+timedelta(hours=1)
def put(section,**fields):
 target=groups.setdefault(section,{})
 for k,v in fields.items():target[k]=v if isinstance(v,list) else [v]
put('time_control',run_days=0,run_hours=0,run_minutes=0,run_seconds=60,interval_seconds=3600,input_from_file=True,history_interval_s=30,frames_per_outfile=3,restart=False)
groups['time_control'].pop('history_interval',None)
for prefix,time in [('start',start),('end',end)]:
 for key in ['year','month','day','hour','minute','second']:put('time_control',**{prefix+'_'+key:getattr(time,key)})
put('domains',max_dom=1,time_step=15,time_step_fract_num=0,time_step_fract_den=1,e_we=sizes['west_east']+1,e_sn=sizes['south_north']+1,e_vert=sizes['bottom_top']+1,dx=float(attrs['DX']),dy=float(attrs['DY']),p_top_requested=top,eta_levels=eta,grid_id=1,parent_id=0,parent_grid_ratio=1,parent_time_step_ratio=1,i_parent_start=1,j_parent_start=1,feedback=0,smooth_option=0,num_metgrid_soil_levels=sizes['soil_layers_stag'])
groups['physics'].pop('physics_suite',None)
put('physics',mp_physics=6,ra_lw_physics=99,ra_sw_physics=99,co2tf=1,cu_physics=0,sf_sfclay_physics=1,sf_surface_physics=2,bl_pbl_physics=1,radt=0,bldt=0,num_soil_layers=4)
if a.metgrid is not None:
 with netcdf_bridge.open_dataset(a.metgrid) as met:
  put('domains',num_metgrid_levels=len(met.dimensions['num_metgrid_levels']),num_metgrid_soil_levels=int(met.getncattr('NUM_METGRID_SOIL_LEVELS')))
put('dynamics',mix_full_fields=True,use_theta_m=0,hybrid_opt=int(attrs.get('HYBRID_OPT',2)))
put('bdy_control',specified=True,nested=False,spec_bdy_width=5,spec_zone=1,relax_zone=4)
def fmt(v):
 if type(v) is bool:return '.true.' if v else '.false.'
 if isinstance(v,str):return "'"+v+"'"
 return str(v)
a.output.write_text('\n'.join('&'+g+'\n'+'\n'.join(' '+k+' = '+', '.join(fmt(x) for x in values)+',' for k,values in entries.items())+'\n/' for g,entries in groups.items())+'\n')
print(json.dumps({'source':str(a.source),'start':start.isoformat(),'layers':sizes['bottom_top'],'grid':[sizes['south_north'],sizes['west_east']],'selectors':[99,99],'co2tf':1,'run_seconds':60}))
