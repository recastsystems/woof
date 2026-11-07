! Fixture writer for the CAM Zhang-McFarlane oracle.
!
! One call to zmio_open(stem) opens <stem>.bin (raw little-endian words,
! appended back to back) and <stem>.manifest (one text line per array:
! "name dtype count byte_offset").  dtype is f8, f4 or i4.  The Python side
! is gpuwm.verify.camzm_oracle.load.  Nothing here converts a value: every
! word is written as the program holds it.
module zmio
  implicit none
  integer, parameter, private :: r8 = selected_real_kind(12)
  integer, private :: ubin = 0, uman = 0
  logical, private :: isopen = .false.
  integer(8), private :: offset = 0
  character(len=128), public :: zmio_prefix = ''
contains
  subroutine zmio_open(stem)
    character(len=*), intent(in) :: stem
    open(newunit=ubin, file=trim(stem)//'.bin', access='stream', &
         form='unformatted', status='replace', action='write')
    open(newunit=uman, file=trim(stem)//'.manifest', status='replace', &
         action='write')
    offset = 0
    isopen = .true.
  end subroutine zmio_open

  subroutine zmio_close()
    if (isopen) then
       close(ubin)
       close(uman)
    end if
    isopen = .false.
  end subroutine zmio_close

  subroutine note(name, dtype, n, nbytes)
    character(len=*), intent(in) :: name, dtype
    integer, intent(in) :: n, nbytes
    write(uman, '(a,1x,a,1x,i0,1x,i0)') trim(zmio_prefix)//trim(name), dtype, n, offset
    offset = offset + int(n, 8) * int(nbytes, 8)
  end subroutine note

  subroutine zmio_put_r8(name, arr, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    real(r8), intent(in) :: arr(n)
    if (.not. isopen) return
    call note(name, 'f8', n, 8)
    write(ubin) arr(1:n)
  end subroutine zmio_put_r8

  subroutine zmio_put_r4(name, arr, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    real(4), intent(in) :: arr(n)
    if (.not. isopen) return
    call note(name, 'f4', n, 4)
    write(ubin) arr(1:n)
  end subroutine zmio_put_r4

  subroutine zmio_put_i4(name, arr, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    integer(4), intent(in) :: arr(n)
    if (.not. isopen) return
    call note(name, 'i4', n, 4)
    write(ubin) arr(1:n)
  end subroutine zmio_put_i4

  subroutine zmio_put1_r8(name, x)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: x
    real(r8) :: a(1)
    a(1) = x
    call zmio_put_r8(name, a, 1)
  end subroutine zmio_put1_r8

  subroutine zmio_put1_r4(name, x)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: x
    real(4) :: a(1)
    a(1) = x
    call zmio_put_r4(name, a, 1)
  end subroutine zmio_put1_r4

  subroutine zmio_put1_i4(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x
    integer(4) :: a(1)
    a(1) = x
    call zmio_put_i4(name, a, 1)
  end subroutine zmio_put1_i4
end module zmio
