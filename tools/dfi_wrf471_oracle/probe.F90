program probe
 implicit none
 integer :: steps, nf, u
 real :: dt, cutoff
 real, allocatable :: h(:)
 character(512) :: arg, output, mode
 call get_command_argument(1, mode)
 call get_command_argument(2, arg)
 read(arg,*) steps
 call get_command_argument(3, arg)
 read(arg,*) nf
 call get_command_argument(4, arg)
 read(arg,*) dt
 call get_command_argument(5, arg)
 read(arg,*) cutoff
 call get_command_argument(6, output)
 if (trim(mode)=='safe') then
   allocate(h(101))
   call rhofil(dt,cutoff,2,steps,2,h,100)
 else
   allocate(h(steps+1))
   call dfcoef(steps,dt,cutoff,nf,h)
 endif
 open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
 write(u) h
 close(u)
end program
subroutine wrf_message(message)
 character(*) :: message
end subroutine
