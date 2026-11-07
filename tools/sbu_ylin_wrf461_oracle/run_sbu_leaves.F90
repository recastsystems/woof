program run_sbu_leaves
  use module_mp_sbu_ylin, only : RSIF, RSLF, ggamma
  implicit none
  integer :: i
  real :: p, t, x, a, b, c
  open(unit=10,file='sbu-ylin-leaves.txt',status='replace')
  write(10,'(I0)') 64
  do i=1,64
     p=20000.0+real(i)*1200.0
     t=180.0+real(i)*2.0
     x=0.1+real(i)*0.17
     a=RSIF(p,t)
     b=RSLF(p,t)
     c=ggamma(x)
     write(10,'(6(Z8.8,1X))') transfer(p,0),transfer(t,0),transfer(x,0), &
       transfer(a,0),transfer(b,0),transfer(c,0)
  enddo
  close(10)
end program
