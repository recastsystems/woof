! WRF 4.6.1 public-source column oracle. The wrapper changes no arithmetic.
subroutine column_tmn(ncol,lagday,tsk,tmn,tlag,tyr,tyra,tdly,nday,nyear,julian,dt,yr) bind(C)
 use iso_c_binding
 use module_sf_tmnupdate, only: tmnupdate
 implicit none
 integer(c_int),value :: ncol,lagday,yr
 real(c_float),value :: julian,dt
 real(c_float) :: tsk(ncol,1),tmn(ncol,1),tlag(ncol,lagday,1)
 real(c_float) :: tyr(ncol,1),tyra(ncol,1),tdly(ncol,1),nday
 integer(c_int) :: nyear
 integer :: ist(1),ien(1),jst(1),jen(1)
 ist=1; ien=ncol; jst=1; jen=1
 call tmnupdate(tsk,tmn,tlag,tyr,tyra,tdly,nday,nyear,lagday,julian,dt,yr, &
  1,ncol,1,1,1,1,1,ncol,1,1,1,1,ist,ien,jst,jen,1,1,1)
end subroutine
