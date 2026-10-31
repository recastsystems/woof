! Interface-only p15 dependency. The pristine wrapper calls this only when
! ngw_scheme is true; these orographic tests explicitly select false. Fail
! loudly if the stub is ever reached, so it cannot simulate missing physics.
module bl_ugwpv1_ngw
implicit none
contains
subroutine ugwpv1_ngw_run(xlatd,raincv,rainncv,ddy_j1tau,ddy_j2tau,jindx1_tau,jindx2_tau, &
 r_DoY,kdt,dtp,ugrs,vgrs,tgrs,q1,prsl,prslk,prsi,zl,zi,ntau_d2t,days_limb,tau_limb, &
 rublten,rvblten,rthblten,ugwp_diags,dudt_ngw,dvdt_ngw,dtdt_ngw,its,ite,levs)
integer,intent(in) :: kdt,ntau_d2t,its,ite,levs
integer,intent(in) :: jindx1_tau(:),jindx2_tau(:)
logical,intent(in) :: ugwp_diags
real,intent(in) :: r_DoY,dtp
real,intent(in) :: xlatd(:),raincv(:),rainncv(:),ddy_j1tau(:),ddy_j2tau(:)
real,intent(in) :: ugrs(:,:),vgrs(:,:),tgrs(:,:),q1(:,:),prsl(:,:),prslk(:,:),prsi(:,:),zl(:,:),zi(:,:)
real,intent(in) :: days_limb(:),tau_limb(:,:)
real,intent(inout) :: rublten(:,:),rvblten(:,:),rthblten(:,:)
real,intent(out) :: dudt_ngw(:,:),dvdt_ngw(:,:),dtdt_ngw(:,:)
error stop 'p15 test-only interface reached: nonstationary physics is not simulated'
end subroutine
end module
