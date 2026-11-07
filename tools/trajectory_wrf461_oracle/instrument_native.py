"""Capture real native module_em trajectory arguments around unchanged math."""
from pathlib import Path
import hashlib,json,sys

def instrument(root):
    path=root/'dyn_em/module_em.F';original=path.read_bytes()
    manifest=json.loads(Path(__file__).with_name('source_manifest.json').read_text())
    row=next(x for x in manifest['sources'] if x['path']=='dyn_em/module_em.F')
    assert hashlib.sha256(original).hexdigest()==row['sha256']
    text=original.decode()
    anchor='  logical :: has_proj_map\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'  integer :: cap_unit\n  character(80) :: cap_file\n')
    anchor='traj_loop: &\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,CAPTURE+anchor)
    anchor='END SUBROUTINE trajectory\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,OUTPUT+anchor)
    path.write_text(text)
    (root/'trajectory-instrumentation.json').write_text(json.dumps(dict(path=row['path'],original_sha256=row['sha256'],instrumented_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')

CAPTURE="   IF (itimestep == 1) THEN\n     WRITE(cap_file,'(A,I2.2)') 'trajectory-',dm\n     OPEN(newunit=cap_unit,file=TRIM(cap_file)//'.in.bin',form='unformatted',access='stream',status='replace')\n     WRITE(cap_unit) 1,dm,traj_cnt(dm),ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte\n     WRITE(cap_unit) dt,rdx,rdy\n     WRITE(cap_unit) traj_i,traj_j,traj_k,traj_long,traj_lat\n     WRITE(cap_unit) ru_m,rv_m,ww_m,mut,muu,muv,msft,msfu,msfv\n     WRITE(cap_unit) c1h,c2h,c1f,c2f,grid%znw\n     WRITE(cap_unit) proj%code,proj%hemi,proj%rebydx,proj%truelat1,proj%truelat2,proj%stdlon,proj%polei,proj%polej, &\n                    proj%cone,proj%rsw,proj%dlon,proj%lat1,proj%lon1,proj%latinc,proj%loninc\n     CLOSE(cap_unit)\n   END IF\n"
OUTPUT="   IF (itimestep == 1) THEN\n     OPEN(newunit=cap_unit,file=TRIM(cap_file)//'.out.bin',form='unformatted',access='stream',status='replace')\n     WRITE(cap_unit) traj_i,traj_j,traj_k,traj_long,traj_lat\n     CLOSE(cap_unit)\n   END IF\n"

if __name__=='__main__':instrument(Path(sys.argv[1]))
