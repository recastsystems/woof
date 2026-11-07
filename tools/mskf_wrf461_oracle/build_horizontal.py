"""Build the horizontal forcing oracle directly from WRF's interp_vec_cu."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--wrf',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--wrap-longitude',action='store_true');a=ap.parse_args();a.work.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
src=(a.wrf/'phys/module_physics_init.F').read_text();start=src.index('subroutine interp_vec_cu(');sub=src[start:src.index('end subroutine interp_vec_cu',start)+len('end subroutine interp_vec_cu')]
rng=np.random.default_rng(271);lat=np.array([-75,-25,25,75],dtype='f4');lon=np.array([0,72,144,216,288],dtype='f4');lon=np.where(lon>=180,lon-360,lon).astype('f4') if a.wrap_longitude else lon;pm=np.broadcast_to(np.linspace(100,90000,30,dtype='f4')[None,:,None,None],(12,30,4,5)).copy();monthly=(rng.random((10,12,30,4,5))*1e5).astype('f4');coords=np.stack([np.linspace(-89,89,64,dtype='f4'),np.linspace(-190,370,64,dtype='f4')]);coords[:,0:5]=np.array([[-75,-25,25,75,0],[0,72,144,288,360]],dtype='f4')
np.concatenate([lat,lon,pm.ravel(),monthly.ravel(),coords.ravel()]).astype('<f4').tofile(a.work/'horizontal-input.bin')
body=f'''program horizontal
implicit none
integer,parameter:: nx=5,ny=4,nz=30,nc=64
real::lat(ny),lon(nx),pm(nx,ny,nz,12),aero(nx,ny,nz,12,10),coords(nc,2),result(nc,nz,12,11)
real::wlat1,wlat2,wlon1,wlon2
integer::c,k,m,s,i1,i2,j1,j2
open(10,file='{a.work}/horizontal-input.bin',form='unformatted',access='stream');read(10)lat,lon,pm,aero,coords;close(10)
open(10,file='{a.out}/toy-cesm.dat',form='unformatted');write(10)nx,ny,nz;write(10)lat;write(10)lon;write(10)pm;write(10)aero;close(10)
do c=1,nc
call interp_vec_cu(lat,coords(c,1),.true.,j1,j2,wlat1,wlat2)
call interp_vec_cu(lon,coords(c,2),.true.,i1,i2,wlon1,wlon2)
do m=1,12
do k=1,nz
result(c,k,m,1)=wlon1*(wlat1*pm(i1,j1,k,m)+wlat2*pm(i1,j2,k,m))+wlon2*(wlat1*pm(i2,j1,k,m)+wlat2*pm(i2,j2,k,m))
do s=1,10
result(c,k,m,s+1)=wlon1*(wlat1*aero(i1,j1,k,m,s)+wlat2*aero(i1,j2,k,m,s))+wlon2*(wlat1*aero(i2,j1,k,m,s)+wlat2*aero(i2,j2,k,m,s))
enddo
enddo
enddo
enddo
open(10,file='{a.work}/horizontal-output.bin',form='unformatted',access='stream',status='replace');write(10)result;close(10)
contains
{sub}
end program
'''
p=a.work/'horizontal.f90';p.write_text(body);subprocess.run(['gfortran','-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize','-ffree-line-length-none',str(p),'-o',str(a.work/'horizontal')],check=True);subprocess.run([str(a.work/'horizontal')],check=True);expected=np.fromfile(a.work/'horizontal-output.bin','f4').reshape(11,12,30,64);np.savez_compressed(a.out/'horizontal.npz',coordinates=coords,expected=expected);(a.out/'horizontal.json').write_text(json.dumps({'wrf_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'columns':64,'fixture_sha256':hashlib.sha256((a.out/'horizontal.npz').read_bytes()).hexdigest(),'toy_sha256':hashlib.sha256((a.out/'toy-cesm.dat').read_bytes()).hexdigest()},indent=2)+'\n');print('HORIZONTAL_ORACLE',expected.size)
