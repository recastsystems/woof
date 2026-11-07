! Calls the original extracted WRF routines, not a Fortran reimplementation.
program run_dfi_weights
  implicit none
  integer :: steps, nf, unit, ios
  real :: dt, cutoff
  real, allocatable :: h(:)
  character(256) :: argument
  call get_command_argument(1, argument)
  read(argument,*) steps
  call get_command_argument(2, argument)
  read(argument,*) nf
  call get_command_argument(3, argument)
  read(argument,*) dt
  call get_command_argument(4, argument)
  read(argument,*) cutoff
  call get_command_argument(5, argument)
  allocate(h(steps+1))
  h = -huge(1.)
  call dfcoef(steps, dt, cutoff, nf, h)
  open(newunit=unit, file=trim(argument), status='replace', &
       access='stream', form='unformatted', action='write', iostat=ios)
  if (ios /= 0) stop 2
  write(unit) h
  close(unit)
end program

subroutine wrf_message(message)
  character(*), intent(in) :: message
end subroutine
