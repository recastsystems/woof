! Pins two compiler behaviours the SBU-YLin port reproduces, at the
! reference flags, instead of assuming them:
!   1. AMAX1/AMIN1 on a pair of signed zeros, both argument orders (the
!      scheme clips every condensate with amax1(0.0, q) at entry, and the
!      oracle cases carry -0.0 inputs).
!   2. REAL**INTEGER with a run-time exponent (ggamma's TEMP**K1), printed
!      next to the repeated-squaring product the port computes, for every
!      K1 the scheme uses.
program probe_lowering
  implicit none
  real :: pz, nz, x, y, r
  integer :: k
  pz = 0.0
  nz = -0.0
  write(*,'(A,Z8.8)') 'amax1(+0,-0) ', transfer(amax1(pz, nz), 0_4)
  write(*,'(A,Z8.8)') 'amax1(-0,+0) ', transfer(amax1(nz, pz), 0_4)
  write(*,'(A,Z8.8)') 'amin1(+0,-0) ', transfer(amin1(pz, nz), 0_4)
  write(*,'(A,Z8.8)') 'amin1(-0,+0) ', transfer(amin1(nz, pz), 0_4)
  x = 0.8137291
  do k = 1, 8
    r = x**k
    y = sqchain(x, k)
    write(*,'(A,I0,1X,Z8.8,1X,Z8.8)') 'powi k=', k, transfer(r, 0_4), &
        transfer(y, 0_4)
  end do
contains
  real function sqchain(base, n)
    real, intent(in) :: base
    integer, intent(in) :: n
    real :: b
    integer :: m
    m = n
    b = base
    if (mod(m, 2) == 1) then
      sqchain = b
    else
      sqchain = 1.0
    end if
    m = m / 2
    do while (m > 0)
      b = b * b
      if (mod(m, 2) == 1) sqchain = sqchain * b
      m = m / 2
    end do
  end function sqchain
end program probe_lowering
