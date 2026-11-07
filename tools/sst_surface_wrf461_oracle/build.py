"""Compile the pinned public WRF surface block and unmodified modules."""
import argparse
import hashlib
from pathlib import Path
import subprocess

FLOATS = 'xland glw gsw hfx qfx tsk ust emiss sst dtw sstsk tmn xice xicem xice_save vegfra albedo albbck embck albsi tsk_save snow snowc snowh lakemask tyr tyra tdly'.split()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    pins = __import__('json').loads(Path(__file__).with_name('sources.json').read_text())
    for row in pins:
        if row['path'] not in ('phys/module_surface_driver.F', 'phys/module_sf_sstskin.F', 'phys/module_sf_tmnupdate.F'):
            continue
        got = hashlib.sha256((args.source / row['path']).read_bytes()).hexdigest()
        if got != row['sha256']:
            raise SystemExit(f"source pin mismatch: {row['path']}: {got}")
    s = (args.source/'phys/module_surface_driver.F').read_text()
    update = s.split('! Update SST\n',1)[1].split('  IF(PRESENT(SST_SKIN))THEN',1)[0]
    skin = s.split('  IF(PRESENT(SST_SKIN))THEN',1)[1].split('  IF(PRESENT(TMN_UPDATE))THEN',1)[0].rstrip()
    assert skin.endswith('ENDIF')
    skin = skin[:-len('ENDIF')]
    tmn = s.split('  IF(PRESENT(TMN_UPDATE))THEN',1)[1].split('! Modified for adaptive time step',1)[0].strip().rstrip('!').rstrip()
    assert tmn.endswith('ENDIF')
    tmn = tmn[:-len('ENDIF')]
    header = '''subroutine surface_stage(n,ns,nl,h,p,f,ivgtyp,isltyp,soil,tlag,nday,nyear,stage) bind(C)
use iso_c_binding
use module_sf_sstskin, only: sst_skin_update
use module_sf_tmnupdate, only: tmnupdate
implicit none
integer(c_int),value :: n,ns,nl,stage
integer(c_int) :: h(13),ivgtyp(n,1),isltyp(n,1),nyear
real(c_float) :: p(4),f(n,28),soil(n,ns,1,3),tlag(n,nl,1),nday
integer :: i,j,k,nk,ij,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,kts,kte
integer :: i_start(1),i_end(1),j_start(1),j_end(1),num_tiles,num_soil_layers,lagday,yr
integer :: sst_update,sst_skin,tmn_update,fractional_seaice,seaice_albedo_opt,iswater,isice,lakemodel
real :: dt,julian_in,stbolt,seaice_albedo_default,xice_threshold
ids=1;ide=n+1;jds=1;jde=2;kds=1;kde=1
ims=1;ime=n;jms=1;jme=1;kms=1;kme=1;kts=1;kte=1
num_tiles=1;num_soil_layers=ns;lagday=nl;yr=h(4)
i_start=1;i_end=n;j_start=1;j_end=1
sst_update=h(6);sst_skin=h(7);tmn_update=h(8);fractional_seaice=h(9)
seaice_albedo_opt=h(10);iswater=h(11);isice=h(12);lakemodel=h(13)
dt=p(1);julian_in=p(2);stbolt=p(3);seaice_albedo_default=p(4)
xice_threshold=.5
if(fractional_seaice==1) xice_threshold=.02
'''
    aliases = [f'{name}=>f(:,{i}:{i})' for i,name in enumerate(FLOATS,1)]
    aliases += ['tslb=>soil(:,:,:,1)','smois=>soil(:,:,:,2)','sh2o=>soil(:,:,:,3)']
    body = header + 'associate( &\n' + ', &\n'.join(aliases) + ')\nselect case(stage)\ncase(1)\n'+update+'\ncase(2)\n'+skin+'\ncase(3)\n'+tmn+'\nend select\nend associate\nend subroutine\n'
    body += '''subroutine wrf_debug(level,message)
integer :: level
character(*) :: message
end subroutine
subroutine wrf_error_fatal(message)
character(*) :: message
print *,message
stop 1
end subroutine
'''
    args.output.mkdir(parents=True,exist_ok=True)
    wrapper=args.output/'surface_stage.F90'; wrapper.write_text(body)
    stub=args.output/'module_wrf_error.F90'
    # This module exports no names used by sst_skin_update. No numerical
    # dependency is replaced; the native source merely USEs its namespace.
    stub.write_text('module module_wrf_error\nend module\n')
    command=['gfortran','-cpp','-DEM_CORE=1','-ffree-form','-ffree-line-length-none','-O2',
             '-ffp-contract=off','-fno-tree-vectorize','-fPIC','-shared','-J',str(args.output),'-I',str(args.output),str(stub),
             str(args.source/'phys/module_sf_sstskin.F'),str(args.source/'phys/module_sf_tmnupdate.F'),
             str(wrapper),'-o',str(args.output/'libsurface_stage.so')]
    subprocess.run(command,check=True)
    print(__import__('json').dumps(dict(command=command,wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),library_sha256=hashlib.sha256((args.output/'libsurface_stage.so').read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
