! Test adapter only. Calls the unchanged pinned native module.
subroutine nwp_hail_ref(nx,ny,nz,fields,out,mp,flags,options,constants) bind(C)
  use iso_c_binding
  use module_configure
  use module_diag_nwp,only: diagnostic_output_nwp
  implicit none
  integer(c_int),value :: nx,ny,nz,mp
  integer(c_int),intent(in) :: flags(4),options(5)
  real(c_float),intent(in) :: constants(6)
  real(c_float),target,intent(in) :: fields(nx,nz+1,ny,8)
  real(c_float),intent(out) :: out(nx,ny,2)
  real(c_float),pointer :: qh(:,:,:),ng(:,:,:),nh(:,:,:),qb(:,:,:)
  real(c_float) :: scratch(nx,ny,12),zero(nx,nz+1,ny),znw(nz+1),sfc(nx,ny)
  type(grid_config_rec_type) :: cfg
  integer :: ib(1),ie(1),jb(1),je(1)
  nullify(qh,ng,nh,qb)
  if(flags(1)/=0) qh=>fields(:,:,:,3)
  if(flags(2)/=0) ng=>fields(:,:,:,4)
  if(flags(3)/=0) nh=>fields(:,:,:,5)
  if(flags(4)/=0) qb=>fields(:,:,:,6)
  cfg%nssl_2moment_on=options(4)
  cfg%nssl_hail_on=options(5)
  ib=1;ie=nx;jb=1;je=ny
  scratch=0.;zero=0.;znw=0.6;sfc=0.
  call diagnostic_output_nwp(config_flags=cfg, &
    ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1, &
    ims=1,ime=nx,jms=1,jme=ny,kms=1,kme=nz+1, &
    ips=1,ipe=nx,jps=1,jpe=ny,kps=1,kpe=nz+1, &
    i_start=ib,i_end=ie,j_start=jb,j_end=je,kts=1,kte=nz+1,num_tiles=1, &
    u=zero,v=zero,temp=zero,p8w=zero,dt=20.,xtime=0.,sbw=0,mphysics_opt=mp, &
    gsfcgce_hail=options(2),gsfcgce_2ice=options(3),mpuse_hail=options(1), &
    nssl_cnoh=constants(1),nssl_cnohl=constants(2),nssl_rho_qh=constants(3), &
    nssl_rho_qhl=constants(4),nssl_alphah=constants(5),nssl_alphahl=constants(6), &
    nwp_diagnostics=1,diagflag=.true.,history_interval=1,itimestep=1, &
    u10=sfc,v10=sfc,w=zero,wspd10max=scratch(:,:,1),up_heli_max=scratch(:,:,2), &
    w_up_max=scratch(:,:,3),w_dn_max=scratch(:,:,4),znw=znw, &
    w_colmean=scratch(:,:,10),numcolpts=scratch(:,:,11),w_mean=scratch(:,:,5), &
    grpl_max=scratch(:,:,6),grpl_colint=scratch(:,:,12),refd_max=scratch(:,:,7),refl_10cm=zero, &
    hail_maxk1=scratch(:,:,8),hail_max2d=scratch(:,:,9),qg_curr=fields(:,:,:,1), &
    rho=fields(:,:,:,2),ph=zero,phb=zero,g=9.81,max_time_step=20,adaptive_ts=.false., &
    qh_curr=qh,ng_curr=ng,nh_curr=nh,qvolg_curr=qb,qr_curr=fields(:,:,:,7),nr_curr=fields(:,:,:,8))
  out(:,:,1)=scratch(:,:,8)
  out(:,:,2)=scratch(:,:,9)
end subroutine

! Reproduce only the native grid construction, using the native precision.
! These source constants are generated offline, never in a forecast loop.
subroutine nwp_hail_bins_ref(d,width) bind(C)
  use iso_c_binding
  implicit none
  real(c_double),intent(out) :: d(50),width(50)
  real(c_double) :: edges(51)
  integer :: n
  edges(1)=500.d-6
  edges(51)=0.075d0
  do n=2,50
    edges(n)=dexp(dfloat(n-1)/dfloat(50)*dlog(edges(51)/edges(1))+dlog(edges(1)))
  enddo
  do n=1,50
    d(n)=dsqrt(edges(n)*edges(n+1))
    width(n)=edges(n+1)-edges(n)
  enddo
end subroutine
