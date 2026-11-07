module init_probe_m
  ! Does -finit-logical reach an AUTOMATIC logical array?  That is the
  ! storage class of COMPUTE_ENTR_DETR's GTEST_LOCAL_LCL2
  ! (module_bl_mfshconvpbl.F:730-731), which WRF reads at :997 without
  ! ever assigning it on the paths that do not find an LCL.  build.sh runs
  ! this probe under -finit-logical=true and =false and requires the two
  ! answers to differ, or the flag proves nothing about the oracle.
  implicit none
contains
  subroutine probe(n, out)
    integer, intent(in) :: n
    integer, intent(out) :: out
    logical, dimension(n) :: auto_l
    real, dimension(n) :: noise
    noise = 1.0
    out = count(auto_l)
  end subroutine probe

  subroutine scribble(n)
    ! Leave nonzero bytes on the stack where probe's array will live.
    integer, intent(in) :: n
    integer, dimension(n) :: junk
    junk = -1
    if (junk(1) == 7) print *, 'never'
  end subroutine scribble
end module init_probe_m

program init_probe
  use init_probe_m
  implicit none
  integer :: c
  call scribble(64)
  call probe(64, c)
  write(*, '(A,I0)') 'true_count=', c
end program init_probe
