! Test adapter only. The referee is unmodified WRF SFDDAGD.
subroutine native_surface(nx,ny,nz,fasdas,dx,radius,coef,tfac,guv,gt,gq, &
    u,v,theta,temp,qv,p,z,ht,regime,znt,old,new,odis_old,odis_new,wpbl,wzfac,out,sda_hfx,sda_qfx) bind(C)
  use iso_c_binding
  use module_fdda_psufddagd, only: sfddagd
  implicit none
  integer(c_int),value :: nx,ny,nz,fasdas
  real(c_float),value :: dx,radius,coef,tfac,guv,gt,gq
  real(c_float) :: u(nx,nz,ny),v(nx,nz,ny),theta(nx,nz,ny),temp(nx,nz,ny)
  real(c_float) :: qv(nx,nz,ny),p(nx,nz,ny),z(nx,nz,ny)
  real(c_float) :: ht(nx,ny),regime(nx,ny),znt(nx,ny),old(nx,ny,4),new(nx,ny,4)
  real(c_float) :: odis_old(nx,ny),odis_new(nx,ny)
  real(c_float) :: wpbl(nx,nz,ny,4),wzfac(nz,4),out(nx,nz,ny,4)
  real(c_float) :: sda_hfx(nx,ny),sda_qfx(nx,ny),mu(nx,ny),dummy(nx,ny)
  real :: time_factor,actual_end
  dummy=0.;mu=0.;time_factor=tfac;actual_end=3600.
  ! A unit-minute interval preserves an already captured coefficient exactly.
  call sfddagd(1,dx,10.,coef,1,1,60, &
    guv,gt,gq,radius,u,v,theta,temp,qv,p,p, &
    old(:,:,1),old(:,:,2),dummy,old(:,:,3),old(:,:,4),dummy,dummy,dummy,dummy,odis_old, &
    new(:,:,1),new(:,:,2),dummy,new(:,:,3),new(:,:,4),dummy,dummy,dummy,dummy,odis_new, &
    out(:,:,:,1),out(:,:,:,2),out(:,:,:,3),out(:,:,:,4),mu, &
    dummy,ht,regime,znt,z,z, &
    1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz, &
    wpbl,wzfac,0,0.,actual_end,time_factor,fasdas,sda_hfx,sda_qfx)
end subroutine

logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor=.false.
end function
subroutine get_wrf_debug_level(level)
  integer :: level
  level=0
end subroutine
subroutine wrf_message(message)
  character(*) :: message
end subroutine
subroutine wrf_debug(level,message)
  integer :: level
  character(*) :: message
end subroutine
subroutine wrf_error_fatal(message)
  character(*) :: message
  print *,message
  stop 9
end subroutine
