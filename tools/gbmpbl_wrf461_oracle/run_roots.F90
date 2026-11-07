program run_roots_oracle
  implicit none
  integer :: i, unit
  real :: a, b, c, r1, r2
  open(newunit=unit,file='roots-oracle.bin',access='stream',form='unformatted',status='replace')
  do i=0,127
    a=real(mod(i,7)-3)
    b=real(mod(i,11)-5)*0.75
    c=real(mod(i,13)-6)*0.125
    call roots(a,b,c,r1,r2)
    write(unit) a,b,c,r1,r2
  enddo
  close(unit)
end program run_roots_oracle
