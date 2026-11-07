! Check EM QNSESFCINIT cold start and restart carry on 64 columns.
program init_state_probe
  use module_sf_qnsesfc
  implicit none
  integer, parameter :: n=64
  integer :: low(n,1), veg(n,1), i
  real :: ust(n,1), z0(n,1), sea(n,1), ice(n,1), saved(n,1)
  do i=1,n
    ust(i,1)=real(i)*0.003
    z0(i,1)=real(i)*0.001
    sea(i,1)=1.+real(mod(i,2))
    ice(i,1)=0.
    veg(i,1)=mod(i,24)+1
  enddo
  saved=z0
  low=99
  call QNSESFCINIT(low,ust,z0,sea,ice,veg,.false.,.true., &
       1,n+1,1,2,1,3,1,n,1,1,1,3,1,n,1,1,1,2)
  if(any(low/=1)) stop 1
  if(any(transfer(ust,[(0,i=1,n)])/=transfer(0.1,0))) stop 2
  if(any(transfer(z0,[(0,i=1,n)])/=transfer(saved,[(0,i=1,n)]))) stop 3
  do i=1,n
    ust(i,1)=real(i)*0.003
  enddo
  saved=ust
  call QNSESFCINIT(low,ust,z0,sea,ice,veg,.true.,.true., &
       1,n+1,1,2,1,3,1,n,1,1,1,3,1,n,1,1,1,2)
  if(any(transfer(ust,[(0,i=1,n)])/=transfer(saved,[(0,i=1,n)]))) stop 4
  print '(a)', 'PASS: 64 EM columns, cold USTAR=0.1, LOWLYR=1, Z0 unchanged, restart USTAR unchanged'
end program init_state_probe
