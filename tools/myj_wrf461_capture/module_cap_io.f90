! Capture helper for WRF 4.6.1 scheme wrappers (MYJSFC/MYJPBL replay).
! Writes the patch region of each named argument to a per-tile stream file.
module module_cap_io
  implicit none
  integer, save :: cap_u = 0
  integer, save :: cap_nsteps = -1
  integer, save :: cap_steps(64)
contains
  subroutine cap_init()
    character(len=512) :: s
    integer :: n, ios, p, q
    if (cap_nsteps >= 0) return
    cap_nsteps = 0
    call get_environment_variable('WRF_CAP_STEPS', s, n)
    if (n <= 0) return
    p = 1
    do while (p <= n .and. cap_nsteps < 64)
      q = index(s(p:n), ',')
      if (q == 0) then
        q = n + 1
      else
        q = p + q - 1
      end if
      cap_nsteps = cap_nsteps + 1
      read(s(p:q-1), *, iostat=ios) cap_steps(cap_nsteps)
      if (ios /= 0) cap_nsteps = cap_nsteps - 1
      p = q + 1
    end do
  end subroutine cap_init

  logical function cap_want(itimestep)
    integer, intent(in) :: itimestep
    integer :: m
    call cap_init()
    cap_want = .false.
    do m = 1, cap_nsteps
      if (cap_steps(m) == itimestep) cap_want = .true.
    end do
  end function cap_want

  subroutine cap_open(tag, phase, itimestep, its, jts)
    character(len=*), intent(in) :: tag, phase
    integer, intent(in) :: itimestep, its, jts
    character(len=256) :: fn, dir
    integer :: n
    call get_environment_variable('WRF_CAP_DIR', dir, n)
    if (n <= 0) dir = '.'
    write(fn, '(a,"/",a,"_",a,"_s",i5.5,"_i",i4.4,"_j",i4.4,".bin")') trim(dir), trim(tag), trim(phase), itimestep, its, jts
    open(newunit=cap_u, file=trim(fn), access='stream', form='unformatted', status='replace', convert='little_endian')
  end subroutine cap_open

  subroutine cap_close()
    if (cap_u /= 0) close(cap_u)
    cap_u = 0
  end subroutine cap_close

  subroutine cap_hdr(name, kind, i0, i1, k0, k1, j0, j1)
    character(len=*), intent(in) :: name
    integer, intent(in) :: kind, i0, i1, k0, k1, j0, j1
    character(len=16) :: nm
    nm = name
    write(cap_u) nm, kind, i0, i1, k0, k1, j0, j1
  end subroutine cap_hdr

  subroutine cap_r3(name, a, ims, ime, kms, kme, jms, jme, i0, i1, k0, k1, j0, j1)
    character(len=*), intent(in) :: name
    integer, intent(in) :: ims, ime, kms, kme, jms, jme, i0, i1, k0, k1, j0, j1
    real, intent(in) :: a(ims:ime, kms:kme, jms:jme)
    call cap_hdr(name, 3, i0, i1, k0, k1, j0, j1)
    write(cap_u) a(i0:i1, k0:k1, j0:j1)
  end subroutine cap_r3

  subroutine cap_r2(name, a, ims, ime, jms, jme, i0, i1, j0, j1)
    character(len=*), intent(in) :: name
    integer, intent(in) :: ims, ime, jms, jme, i0, i1, j0, j1
    real, intent(in) :: a(ims:ime, jms:jme)
    call cap_hdr(name, 2, i0, i1, 1, 1, j0, j1)
    write(cap_u) a(i0:i1, j0:j1)
  end subroutine cap_r2

  subroutine cap_i2(name, a, ims, ime, jms, jme, i0, i1, j0, j1)
    character(len=*), intent(in) :: name
    integer, intent(in) :: ims, ime, jms, jme, i0, i1, j0, j1
    integer, intent(in) :: a(ims:ime, jms:jme)
    call cap_hdr(name, 5, i0, i1, 1, 1, j0, j1)
    write(cap_u) a(i0:i1, j0:j1)
  end subroutine cap_i2

  subroutine cap_rs(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    call cap_hdr(name, 1, 1, 1, 1, 1, 1, 1)
    write(cap_u) x
  end subroutine cap_rs

  subroutine cap_is(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x
    call cap_hdr(name, 4, 1, 1, 1, 1, 1, 1)
    write(cap_u) x
  end subroutine cap_is
end module module_cap_io
