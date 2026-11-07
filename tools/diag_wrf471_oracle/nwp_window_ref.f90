! Dependency surface for expression calls of unchanged module_diag_nwp.
! Numeric selector values are generated from the pinned Registry.EM_COMMON.
! idx_bg1 is not read by compared non-PSD expressions. Compared PSD cases
! either supply GH volume or do not consume the resulting density index.
module module_configure
  type grid_config_rec_type
    integer :: nssl_2moment_on=1,nssl_hail_on=0
  end type
end module
module module_mp_thompson
  integer :: idx_bg1=5
end module
subroutine wrf_debug(level,message)
  integer :: level
  character(*) :: message
end subroutine

! Wrapper is compiled after WRF's module (the build splits this file).
subroutine nwp_window_ref(nx,ny,nz,fields,znw,surface,out,dt,xtime,history,step,adaptive,maxstep) bind(C)
  use iso_c_binding
  use module_configure
  use module_diag_nwp,only: diagnostic_output_nwp
  implicit none
  integer(c_int),value :: nx,ny,nz,history,step,adaptive,maxstep
  real(c_float),value :: dt,xtime
  real(c_float),intent(in) :: fields(nx,nz+1,ny,7),znw(nz+1),surface(nx,ny,2)
  real(c_float),intent(inout) :: out(nx,ny,12)
  type(grid_config_rec_type) :: cfg
  integer :: ib(1),ie(1),jb(1),je(1)
  ib=1;ie=nx;jb=1;je=ny
  call diagnostic_output_nwp(config_flags=cfg, &
    ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1, &
    ims=1,ime=nx,jms=1,jme=ny,kms=1,kme=nz+1, &
    ips=1,ipe=nx,jps=1,jpe=ny,kps=1,kpe=nz+1, &
    i_start=ib,i_end=ie,j_start=jb,j_end=je,kts=1,kte=nz+1,num_tiles=1, &
    u=fields(:,:,:,1),v=fields(:,:,:,1),temp=fields(:,:,:,1),p8w=fields(:,:,:,2), &
    dt=dt,xtime=xtime,sbw=0,mphysics_opt=1, &
    gsfcgce_hail=0,gsfcgce_2ice=0,mpuse_hail=0, &
    nssl_cnoh=0.,nssl_cnohl=0.,nssl_rho_qh=0.,nssl_rho_qhl=0.,nssl_alphah=0.,nssl_alphahl=0., &
    nwp_diagnostics=1,diagflag=.true.,history_interval=history,itimestep=step, &
    u10=surface(:,:,1),v10=surface(:,:,2),w=fields(:,:,:,1), &
    wspd10max=out(:,:,1),up_heli_max=out(:,:,2),w_up_max=out(:,:,3),w_dn_max=out(:,:,4), &
    znw=znw,w_colmean=out(:,:,10),numcolpts=out(:,:,11),w_mean=out(:,:,5), &
    grpl_max=out(:,:,6),grpl_colint=out(:,:,12),refd_max=out(:,:,7),refl_10cm=fields(:,:,:,7), &
    hail_maxk1=out(:,:,8),hail_max2d=out(:,:,9),qg_curr=fields(:,:,:,3), &
    rho=fields(:,:,:,4),ph=fields(:,:,:,5),phb=fields(:,:,:,6),g=9.81, &
    max_time_step=maxstep,adaptive_ts=adaptive/=0)
end subroutine
