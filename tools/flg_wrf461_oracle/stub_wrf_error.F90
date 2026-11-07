subroutine wrf_error_fatal(message)
 character(*),intent(in)::message
 write(*,'(A)') trim(message)
 error stop 1
end subroutine
