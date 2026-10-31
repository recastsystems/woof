! Numerical oracle: pristine MPAS bl_ugwp.F:15-1066, with a C ABI only.
module ugwp_oro_oracle
use iso_c_binding
use bl_ugwp, only: bl_ugwp_run,bl_ugwp_init,bl_ugwp_final,bl_ugwp_timestep_init,bl_ugwp_timestep_final
use module_bl_ugwp_gwdo, only: gwdo_ugwp
implicit none
contains
subroutine oracle_run(nc,nz,a,p,s,kpbl,c,diags,o,so,err) bind(c)
integer(c_int),value :: nc,nz,diags
real(c_float),intent(in) :: a(nc,nz,10),p(nc,nz+1),s(nc,26),c(7)
integer(c_int),intent(in) :: kpbl(nc)
real(c_float),intent(out) :: o(nc,nz,12),so(nc,10)
integer(c_int),intent(out) :: err
character(len=256) :: msg
! MPAS explicitly adds to existing PBL tendencies; candidate outputs start
! from those inputs. Optional arrays are provided in both diagnostic modes.
o(:,:,1)=a(:,:,9)
o(:,:,2)=a(:,:,10)
o(:,:,3:12)=0.
so=0.
call bl_ugwp_run(sina=s(:,1),cosa=s(:,2),rublten=o(:,:,1),rvblten=o(:,:,2), &
  dtaux3d=o(:,:,3),dtauy3d=o(:,:,4), &
  dtaux3d_ls=o(:,:,5),dtauy3d_ls=o(:,:,6),dtaux3d_bl=o(:,:,7),dtauy3d_bl=o(:,:,8), &
  dtaux3d_ss=o(:,:,9),dtauy3d_ss=o(:,:,10),dtaux3d_fd=o(:,:,11),dtauy3d_fd=o(:,:,12), &
  dusfcg=so(:,1),dvsfcg=so(:,2),dusfc_ls=so(:,3),dvsfc_ls=so(:,4), &
  dusfc_bl=so(:,5),dvsfc_bl=so(:,6),dusfc_ss=so(:,7),dvsfc_ss=so(:,8), &
  dusfc_fd=so(:,9),dvsfc_fd=so(:,10),ugwp_diags=(diags/=0), &
  uproj=a(:,:,1),vproj=a(:,:,2),t1=a(:,:,3),q1=a(:,:,4), &
  prsi=p,prsl=a(:,:,5),prslk=a(:,:,6),zl=a(:,:,7),dz=a(:,:,8), &
  hpbl=s(:,3),kpbl=kpbl,br1=s(:,4),xland1=s(:,5),dxmeter=s(:,6), &
  var=s(:,7),oc1=s(:,8),oa2d1=s(:,9),oa2d2=s(:,10),oa2d3=s(:,11),oa2d4=s(:,12), &
  ol2d1=s(:,13),ol2d2=s(:,14),ol2d3=s(:,15),ol2d4=s(:,16), &
  varss=s(:,17),oc1ss=s(:,18),oa2d1ss=s(:,19),oa2d2ss=s(:,20), &
  oa2d3ss=s(:,21),oa2d4ss=s(:,22),ol2d1ss=s(:,23),ol2d2ss=s(:,24), &
  ol2d3ss=s(:,25),ol2d4ss=s(:,26), &
  g_=c(1),cp_=c(2),rd_=c(3),rv_=c(4),fv_=c(5),pi_=c(6),deltim=c(7), &
  its=1,ite=nc,kte=nz,kme=nz+1,errmsg=msg,errflg=err)
end subroutine

! The complete pristine MPAS wrapper module_bl_ugwp_gwdo.F:15-469;
! ngw_scheme=false leaves p15's optional branch unexecuted.
subroutine oracle_wrapper(nc,nz,a,p,s,kpbl,c,diags,o,so,err) bind(c)
integer(c_int),value :: nc,nz,diags
real(c_float),intent(in) :: a(nc,nz,10),p(nc,nz+1),s(nc,26),c(7)
integer(c_int),intent(in) :: kpbl(nc)
real(c_float),intent(out) :: o(nc,nz,12),so(nc,10)
integer(c_int),intent(out) :: err
real :: aa(nc,nz+1,1,10),oo(nc,nz+1,1,12),ss(nc,1,10)
real :: pp(nc,nz+1,1),zi(nc,nz+1,1),th(nc,nz+1,1)
character(len=256) :: msg
integer :: m
aa=0.;oo=0.;ss=0.;zi=0.;th=0.123
pp(:,:,1)=p
do m=1,10
 aa(:,1:nz,1,m)=a(:,:,m)
enddo
oo(:,:,1,1)=aa(:,:,1,9)
oo(:,:,1,2)=aa(:,:,1,10)
call gwdo_ugwp(u3d=aa(:,:,:,1),v3d=aa(:,:,:,2),t3d=aa(:,:,:,3),qv3d=aa(:,:,:,4), &
 p3d=aa(:,:,:,5),p3di=pp,pi3d=aa(:,:,:,6),z=aa(:,:,:,7),dz=aa(:,:,:,8),zi=zi, &
 rublten=oo(:,:,:,1),rvblten=oo(:,:,:,2),rthblten=th, &
 dtaux3d=oo(:,:,:,3),dtauy3d=oo(:,:,:,4), &
 dtaux3d_ls=oo(:,:,:,5),dtauy3d_ls=oo(:,:,:,6),dtaux3d_bl=oo(:,:,:,7),dtauy3d_bl=oo(:,:,:,8), &
 dtaux3d_ss=oo(:,:,:,9),dtauy3d_ss=oo(:,:,:,10),dtaux3d_fd=oo(:,:,:,11),dtauy3d_fd=oo(:,:,:,12), &
 dusfcg=ss(:,:,1),dvsfcg=ss(:,:,2),dusfc_ls=ss(:,:,3),dvsfc_ls=ss(:,:,4), &
 dusfc_bl=ss(:,:,5),dvsfc_bl=ss(:,:,6),dusfc_ss=ss(:,:,7),dvsfc_ss=ss(:,:,8), &
 dusfc_fd=ss(:,:,9),dvsfc_fd=ss(:,:,10), &
 ugwp_diags=(diags/=0),ngw_scheme=.false.,ntau_d1y=1,ntau_d2t=1,r_DoY=0., &
 sina=reshape(s(:,1),[nc,1]),cosa=reshape(s(:,2),[nc,1]), &
 pblh=reshape(s(:,3),[nc,1]),kpbl2d=reshape(kpbl,[nc,1]), &
 br1=reshape(s(:,4),[nc,1]),xland=reshape(s(:,5),[nc,1]),dx=reshape(s(:,6),[nc,1]), &
 var2dls=reshape(s(:,7),[nc,1]),oc12dls=reshape(s(:,8),[nc,1]), &
 oa2d1ls=reshape(s(:,9),[nc,1]),oa2d2ls=reshape(s(:,10),[nc,1]), &
 oa2d3ls=reshape(s(:,11),[nc,1]),oa2d4ls=reshape(s(:,12),[nc,1]), &
 ol2d1ls=reshape(s(:,13),[nc,1]),ol2d2ls=reshape(s(:,14),[nc,1]), &
 ol2d3ls=reshape(s(:,15),[nc,1]),ol2d4ls=reshape(s(:,16),[nc,1]), &
 var2dss=reshape(s(:,17),[nc,1]),oc12dss=reshape(s(:,18),[nc,1]), &
 oa2d1ss=reshape(s(:,19),[nc,1]),oa2d2ss=reshape(s(:,20),[nc,1]), &
 oa2d3ss=reshape(s(:,21),[nc,1]),oa2d4ss=reshape(s(:,22),[nc,1]), &
 ol2d1ss=reshape(s(:,23),[nc,1]),ol2d2ss=reshape(s(:,24),[nc,1]), &
 ol2d3ss=reshape(s(:,25),[nc,1]),ol2d4ss=reshape(s(:,26),[nc,1]), &
 cp=c(2),g=c(1),rd=c(3),rv=c(4),ep1=c(5),pi=c(6),dt=c(7),itimestep=1, &
 ids=1,ide=nc+1,jds=1,jde=2,kds=1,kde=nz+1, &
 ims=1,ime=nc,jms=1,jme=1,kms=1,kme=nz+1, &
 its=1,ite=nc,jts=1,jte=1,kts=1,kte=nz,errmsg=msg,errflg=err)
do m=1,12
 o(:,:,m)=oo(:,1:nz,1,m)
enddo
so=ss(:,1,:)
if(any(th/=0.123)) error stop 'orographic wrapper changed rthblten'
end subroutine

! MPAS driver expressions: mpas_atmphys_driver_gwdo.F:534,679-680.
subroutine oracle_driver(n,density,len_disp,before,after,dx,diff) bind(c)
integer(c_int),value :: n
real(c_float),value :: len_disp
real(c_float),intent(in) :: density(n),before(n),after(n)
real(c_float),intent(out) :: dx(n),diff(n)
dx=len_disp/density**0.25
diff=after-before
end subroutine
function oracle_lifecycle() result(err) bind(c)
integer(c_int) :: err
character(len=256) :: msg
integer :: status
call bl_ugwp_init(msg,status)
err=status
call bl_ugwp_final(msg,status)
err=err+status
call bl_ugwp_timestep_init(msg,status)
err=err+status
call bl_ugwp_timestep_final(msg,status)
err=err+status
end function
end module
