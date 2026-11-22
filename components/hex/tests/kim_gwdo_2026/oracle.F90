! MPAS PR #1409 driver_gwdo bl_kim_gwdo slice (mpas_atmphys_driver_gwdo.F:525-550 dx, :758-781 call).
! Test-only C binding. bl_gwdo.F90 is pristine MMM-physics 63c806deef6a; module_bl_gwdo.F is
! MPAS-A v8.4.2 (9784f201) with exactly the PR #1409 hunk applied (elvmax, dx_factor, if_nonhyd).
module kim_gwdo_test_oracle
contains
subroutine kim_gwdo_oracle(nlev,ncol,dt,len_disp,dx_factor,nonhyd,u,v,t,q,p,prsi,exner,z,ru,rv, &
                           md,var,con,oa1,oa2,oa3,oa4,ol1,ol2,ol3,ol4,elvmax,outputs,err) bind(C)
  use iso_c_binding
  use module_bl_gwdo, only: gwdo
  use bl_gwdo, only: bl_gwdo_init, bl_gwdo_finalize
  implicit none
  integer(c_int), value :: nlev,ncol,nonhyd
  real(c_float), value :: dt,len_disp,dx_factor
  real(c_float), intent(in) :: u(ncol,nlev),v(ncol,nlev),t(ncol,nlev),q(ncol,nlev)
  real(c_float), intent(in) :: p(ncol,nlev),prsi(ncol,nlev+1),exner(ncol,nlev),z(ncol,nlev)
  real(c_float), intent(in) :: ru(ncol,nlev),rv(ncol,nlev)
  real(c_float), intent(in) :: md(ncol),var(ncol),con(ncol),elvmax(ncol)
  real(c_float), intent(in) :: oa1(ncol),oa2(ncol),oa3(ncol),oa4(ncol)
  real(c_float), intent(in) :: ol1(ncol),ol2(ncol),ol3(ncol),ol4(ncol)
  real(c_float), intent(out) :: outputs(ncol*nlev*6+ncol*2)
  integer(c_int), intent(out) :: err
  real :: up(ncol,nlev+1,1),vp(ncol,nlev+1,1),tp(ncol,nlev+1,1),qp(ncol,nlev+1,1)
  real :: pp(ncol,nlev+1,1),ep(ncol,nlev+1,1),zp(ncol,nlev+1,1)
  real :: rup(ncol,nlev+1,1),rvp(ncol,nlev+1,1),du(ncol,nlev+1,1),dv(ncol,nlev+1,1)
  real :: dx(ncol,1),sina(ncol,1),cosa(ncol,1),dus(ncol,1),dvs(ncol,1)
  real :: rdif(ncol,nlev),vdif(ncol,nlev)
  integer :: kpbl(ncol,1),i,words
  logical :: if_nonhyd
  character(len=256) :: errmsg
  real, parameter :: g=9.80616, rd=287., rv_const=461.6, cp=7.*rd/2., ep1=rv_const/rd-1., pi=3.14159265358979323846
  call bl_gwdo_init(errmsg,err)
  if (err /= 0) return
  if_nonhyd = nonhyd /= 0
  up=0.;vp=0.;tp=0.;qp=0.;pp=0.;ep=0.;zp=0.;rup=0.;rvp=0.
  up(:,1:nlev,1)=u;vp(:,1:nlev,1)=v;tp(:,1:nlev,1)=t;qp(:,1:nlev,1)=q
  pp(:,1:nlev,1)=p;ep(:,1:nlev,1)=exner;zp(:,1:nlev,1)=z
  rup(:,1:nlev,1)=ru;rvp(:,1:nlev,1)=rv
  do i=1,ncol
    sina(i,1)=0.;cosa(i,1)=1.
    dx(i,1)=len_disp/md(i)**0.25
  enddo
  kpbl=2
  call gwdo(u3d=up,v3d=vp,t3d=tp,qv3d=qp,p3d=pp,p3di=reshape(prsi,[ncol,nlev+1,1]),pi3d=ep,z=zp, &
    rublten=rup,rvblten=rvp,dtaux3d=du,dtauy3d=dv,dusfcg=dus,dvsfcg=dvs, &
    var2d=reshape(var,[ncol,1]),oc12d=reshape(con,[ncol,1]), &
    oa2d1=reshape(oa1,[ncol,1]),oa2d2=reshape(oa2,[ncol,1]),oa2d3=reshape(oa3,[ncol,1]),oa2d4=reshape(oa4,[ncol,1]), &
    ol2d1=reshape(ol1,[ncol,1]),ol2d2=reshape(ol2,[ncol,1]),ol2d3=reshape(ol3,[ncol,1]),ol2d4=reshape(ol4,[ncol,1]), &
    elvmax=reshape(elvmax,[ncol,1]), &
    sina=sina,cosa=cosa,cp=cp,g=g,rd=rd,rv=rv_const,ep1=ep1,pi=pi,dt=dt,dx=dx,kpbl2d=kpbl,itimestep=1, &
    dx_factor=dx_factor,if_nonhyd=if_nonhyd, &
    ids=1,ide=ncol+1,jds=1,jde=2,kds=1,kde=nlev+1,ims=1,ime=ncol,jms=1,jme=1,kms=1,kme=nlev+1, &
    its=1,ite=ncol,jts=1,jte=1,kts=1,kte=nlev,errmsg=errmsg,errflg=err)
  if (err /= 0) return
  rdif=rup(:,1:nlev,1)-ru;vdif=rvp(:,1:nlev,1)-rv
  words=ncol*nlev
  outputs(1:words)=reshape(rup(:,1:nlev,1),[words])
  outputs(words+1:2*words)=reshape(rvp(:,1:nlev,1),[words])
  outputs(2*words+1:3*words)=reshape(du(:,1:nlev,1),[words])
  outputs(3*words+1:4*words)=reshape(dv(:,1:nlev,1),[words])
  outputs(4*words+1:5*words)=reshape(rdif,[words])
  outputs(5*words+1:6*words)=reshape(vdif,[words])
  outputs(6*words+1:6*words+ncol)=dus(:,1)
  outputs(6*words+ncol+1:)=dvs(:,1)
  call bl_gwdo_finalize(errmsg,err)
end subroutine
end module
