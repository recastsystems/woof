module module_configure
 implicit none
 type grid_config_rec_type
  integer :: co2tf=1
 end type
end module
logical function wrf_dm_on_monitor()
 wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_bytes(a,n)
 integer :: a(*),n
end subroutine
subroutine wrf_dm_bcast_real(a,n)
 real :: a(* )
 integer :: n
end subroutine
subroutine wrf_error_fatal(msg)
 character(*) :: msg
 write(*,'(A)') msg
 error stop 1
end subroutine
subroutine wrf_message(msg)
 character(*) :: msg
end subroutine
subroutine wrf_debug(level,msg)
 integer :: level
 character(*) :: msg
end subroutine
