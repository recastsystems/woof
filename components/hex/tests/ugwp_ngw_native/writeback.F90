! Exact numerical slice from mpas_atmphys_driver_gwdo.F:648-657,676-683.
subroutine oracle_writeback(im,levs,diags,ru_in,rv_in,rt_in,du_in,dv_in,dt_in, &
ru_before,rv_before,ru,rv,rt,rudiff,rvdiff,du,dv,dt) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::im,levs,diags
real(c_float)::ru_in(im,levs),rv_in(im,levs),rt_in(im,levs)
real(c_float)::du_in(im,levs),dv_in(im,levs),dt_in(im,levs)
real(c_float)::ru_before(im,levs),rv_before(im,levs)
real(c_float)::ru(im,levs),rv(im,levs),rt(im,levs),rudiff(im,levs),rvdiff(im,levs)
real(c_float)::du(im,levs),dv(im,levs),dt(im,levs)
integer::i,k
if(diags/=0) then
 do k=1,levs
 do i=1,im
  du(i,k)=du_in(i,k)
  dv(i,k)=dv_in(i,k)
  dt(i,k)=dt_in(i,k)
 enddo
 enddo
endif
 do k=1,levs
 do i=1,im
  rudiff(i,k)=ru_in(i,k)-ru_before(i,k)
  rvdiff(i,k)=rv_in(i,k)-rv_before(i,k)
  ru(i,k)=ru_in(i,k)
  rv(i,k)=rv_in(i,k)
  rt(i,k)=rt_in(i,k)
 enddo
 enddo
end subroutine
