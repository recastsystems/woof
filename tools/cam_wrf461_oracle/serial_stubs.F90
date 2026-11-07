! Serial oracle services. No radiation arithmetic is replaced.
module module_cam_support
contains
  subroutine endrun(message)
    character(*),optional :: message
    if(present(message)) print *, message
    error stop 'CAM oracle endrun'
  end subroutine
end module
module module_wrf_error
  integer,parameter :: IWORDSIZE=4
  character(len=512) :: wrf_err_message
contains
  subroutine wrf_error_fatal(message)
    character(*) :: message
    print *, message
    error stop 'CAM oracle fatal'
  end subroutine
  subroutine wrf_debug(level,message)
    integer :: level
    character(*) :: message
  end subroutine
end module
module module_state_description
  integer,parameter :: SSIBSCHEME=8, CLMSCHEME=5
  integer,parameter :: P_SUL=1, P_SSLT=2,P_OCPHO=3,P_BCPHO=4,P_OCPHI=5,P_BCPHI=6
  integer,parameter :: P_DUST1=7,P_BG=11,P_VOLC=12
end module

module module_dm
contains
  integer function local_communicator()
    local_communicator=0
  end function
end module
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_bytes(buffer,count)
  integer :: buffer(*),count
end subroutine
subroutine wrf_message(message)
  character(*) :: message
end subroutine
subroutine wrf_error_fatal(message)
  character(*) :: message
  print *,message
  error stop 'CAM external fatal'
end subroutine
subroutine wrf_debug(level,message)
  integer :: level
  character(*) :: message
end subroutine

integer function get_unused_unit()
get_unused_unit=37
end function
