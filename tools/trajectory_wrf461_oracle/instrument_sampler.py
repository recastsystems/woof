"""Capture actual first-step trajectory sampler inputs; keep native math intact."""
from pathlib import Path
import hashlib,json,sys

def instrument(root):
    path=root/'share/module_trajectory.F'
    original=path.read_bytes()
    manifest=json.loads(Path(__file__).with_name('source_manifest.json').read_text())
    row=next(x for x in manifest['sources'] if x['path']=='share/module_trajectory.F')
    assert hashlib.sha256(original).hexdigest()==row['sha256']
    text=original.decode();begin=text.index('   subroutine trajectory_driver( grid )')
    end=text.index('   end subroutine trajectory_driver',begin)
    section=text[begin:end]
    anchor='   integer :: m, mu, n, ndx, n_vars, n_traj'
    section=section.replace(anchor,'   integer :: capture_unit\n   character(len=64) :: capture_file\n'+anchor,1)
    section=section.replace('   use module_state_description, only : num_moist, num_tracer, param_first_scalar',
       '   use module_state_description, only : num_moist, num_tracer, param_first_scalar, P_QV',1)
    anchor='!-----------------------------------------------------------------------------\n!  is trajectory in time interval?'
    capture="""     if (grid%itimestep == 1) then
       write(capture_file,'(A,I2.2,A)') 'trajectory-sampler-',dm,'.in.bin'
       open(newunit=capture_unit,file=trim(capture_file),access='stream', &
            form='unformatted',status='replace')
       write(capture_unit) 1,dm,n_traj,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme
       write(capture_unit) grid%traj_i,grid%traj_j,grid%traj_k,grid%traj_long,grid%traj_lat
       write(capture_unit) grid%moist(:,:,:,P_QV),grid%tracer(:,:,:,2),grid%tracer(:,:,:,9)
       write(capture_unit) grid%p,grid%pb,grid%t_2,grid%ph_2,grid%phb,grid%u_2,grid%v_2,grid%w_2
       close(capture_unit)
     endif
"""
    assert section.count(anchor)==1
    section=section.replace(anchor,capture+anchor)
    path.write_text(text[:begin]+section+text[end:])
    (root/'trajectory-sampler-instrumentation.json').write_text(json.dumps(dict(path=row['path'],original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':instrument(Path(sys.argv[1]))
