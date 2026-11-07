! How gfortran 13 at -O0 lowers the one integer power in module_sf_slab.F,
! UPFLUX = RADSWTCH*STBOLT*TG0**4 (SLAB1D loop 70), measured rather than
! guessed.  For each skin temperature it prints the bits of TG0**4 beside
! the candidate spellings ((x*x)*(x*x), ((x*x)*x)*x, powf(x,4.0)); the port
! spells the one whose column always matches.  The real-exponent powers
! (**ROVCP) stay powf calls; `nm -u module_sf_slab.o` is the receipt.
program pow_probe
  implicit none
  interface
    function powf(x, y) bind(c, name='powf')
      use iso_c_binding, only: c_float
      real(c_float), value :: x, y
      real(c_float) :: powf
    end function powf
  end interface
  integer, parameter :: n = 4001
  integer :: i, nsq, nlin, npow
  real :: x, ref, a, b, c
  nsq = 0; nlin = 0; npow = 0
  do i = 0, n - 1
    x = 180.0 + 0.0412345 * real(i)
    ref = x**4
    a = (x * x) * (x * x)
    b = ((x * x) * x) * x
    c = powf(x, 4.0)
    if (transfer(ref, 0) == transfer(a, 0)) nsq = nsq + 1
    if (transfer(ref, 0) == transfer(b, 0)) nlin = nlin + 1
    if (transfer(ref, 0) == transfer(c, 0)) npow = npow + 1
  end do
  write(*, '(A,I0)') 'samples ', n
  write(*, '(A,I0)') 'match (x*x)*(x*x) ', nsq
  write(*, '(A,I0)') 'match ((x*x)*x)*x ', nlin
  write(*, '(A,I0)') 'match powf(x,4.0) ', npow
end program pow_probe
