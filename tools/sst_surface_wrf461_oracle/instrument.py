from pathlib import Path
import argparse
ap=argparse.ArgumentParser(description='Capture real native surface inputs and outputs at steps 1,360,361,720')
ap.add_argument('source',type=Path)
root=ap.parse_args().source
p=root/'phys/module_surface_driver.F';s=p.read_text()
floats=['xland','glw','gsw','hfx','qfx','tsk','ust','emiss','sst','dtw','sstsk','tmn','xice','xicem','xice_save','vegfra','albedo','albbck','embck','albsi','tsk_save','snow','snowc','snowh','lakemask','tyr','tyra','tdly']
ints=['ivgtyp','isltyp']
def capture(arm):
 lines=['BLOCK','INTEGER :: pc_u,pc_n,pc_k,pc_l,pc_i(16),pc_j(16)','CHARACTER(LEN=256) :: pc_name',
 'IF (itimestep==1 .OR. itimestep==360 .OR. itimestep==361 .OR. itimestep==720) THEN',
 'DO pc_k=1,16',
 'pc_i(pc_k)=i_start(1)+(pc_k-1)*(i_end(1)-i_start(1))/15',
 'pc_j(pc_k)=j_start(1)+MOD(pc_k*7,16)*(j_end(1)-j_start(1))/15','ENDDO',
 f'WRITE(pc_name,\'(A,I6.6,A,I4.4,A,I4.4,A)\') "capture-",itimestep,"-",i_start(1),"-",j_start(1),"-{arm}.bin"',
 'OPEN(NEWUNIT=pc_u,FILE=TRIM(pc_name),ACCESS="stream",FORM="unformatted",STATUS="replace",CONVERT="little_endian")',
 'WRITE(pc_u) 16,num_soil_layers,lagday,yr,itimestep,sst_update,sst_skin,tmn_update, &',
 'fractional_seaice,seaice_albedo_opt,iswater,isice,lakemodel',
 'WRITE(pc_u) dt,julian_in,stbolt,seaice_albedo_default']
 for name in floats:lines += [f'WRITE(pc_u) ({name}(pc_i(pc_k),pc_j(pc_k)),pc_k=1,16)']
 for name in ints:lines += [f'WRITE(pc_u) ({name}(pc_i(pc_k),pc_j(pc_k)),pc_k=1,16)']
 for name in ['tslb','smois','sh2o']:lines += [f'WRITE(pc_u) (({name}(pc_i(pc_k),pc_l,pc_j(pc_k)),pc_k=1,16),pc_l=1,num_soil_layers)']
 lines += ['WRITE(pc_u) ((tlag(pc_i(pc_k),pc_l,pc_j(pc_k)),pc_k=1,16),pc_l=1,lagday)','WRITE(pc_u) nday,nyear','CLOSE(pc_u)','ENDIF','END BLOCK']
 return '\n'.join(lines)+'\n'
assert s.count('! Update SST\n')==1
s=s.replace('! Update SST\n',capture('pre')+'! Update SST\n')
needle='! Modified for adaptive time step'
assert s.count(needle)==1
s=s.replace(needle,capture('post')+needle)
p.write_text(s)
p=root/'configure.wrf';s=p.read_text().replace('-O2 -ftree-vectorize -funroll-loops','-O2 -ffp-contract=off -fno-tree-vectorize');p.write_text(s)
for name in ['module_sf_tmnupdate.F','module_sf_sstskin.F']:(root/'phys'/name).touch()
print('instrumented',p)
