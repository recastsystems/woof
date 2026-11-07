program dump_radar
  use module_mp_morr_two_moment_aero, only: morr_two_moment_init_aero
  use module_mp_radar
  use iso_fortran_env, only: int64
  implicit none
  integer :: rimed, n
  character(len=32) :: arg
  call get_command_argument(1,arg)
  read(arg,*) rimed
  call morr_two_moment_init_aero(rimed)
  write(*,'(A,1X,Z8.8)') 'xam_r', transfer(xam_r,0)
  write(*,'(A,1X,Z8.8)') 'xam_s', transfer(xam_s,0)
  write(*,'(A,1X,Z8.8)') 'xam_g', transfer(xam_g,0)
  write(*,'(A,1X,Z8.8)') 'xorg2', transfer(xorg2,0)
  write(*,'(A,1X,Z8.8)') 'xosg2', transfer(xosg2,0)
  write(*,'(A,1X,Z8.8)') 'xogg2', transfer(xogg2,0)
  write(*,'(A,1X,Z8.8)') 'xobmr', transfer(xobmr,0)
  write(*,'(A,1X,Z8.8)') 'xobms', transfer(xobms,0)
  write(*,'(A,1X,Z8.8)') 'xobmg', transfer(xobmg,0)
  write(*,'(A,1X,Z8.8)') 'xocms', transfer(xocms,0)
  write(*,'(A,1X,Z8.8)') 'xocmg', transfer(xocmg,0)
  write(*,'(A,1X,Z16.16)') 'k_w', transfer(k_w,0_int64)
  write(*,'(A,1X,Z16.16)') 'pi5', transfer(pi5,0_int64)
  write(*,'(A,1X,Z16.16)') 'lamda4', transfer(lamda4,0_int64)
  do n=1,4
    write(*,'(A,I0,1X,Z8.8)') 'xcrg_',n,transfer(xcrg(n),0)
  enddo
  do n=1,4
    write(*,'(A,I0,1X,Z8.8)') 'xcsg_',n,transfer(xcsg(n),0)
  enddo
  do n=1,4
    write(*,'(A,I0,1X,Z8.8)') 'xcgg_',n,transfer(xcgg(n),0)
  enddo
  do n=1,4
    write(*,'(A,I0,1X,Z8.8)') 'xcre_',n,transfer(xcre(n),0)
  enddo
  do n=1,4
    write(*,'(A,I0,1X,Z8.8)') 'xcse_',n,transfer(xcse(n),0)
  enddo
  do n=1,4
    write(*,'(A,I0,1X,Z8.8)') 'xcge_',n,transfer(xcge(n),0)
  enddo
  write(*,'(A,1X,Z16.16)') 'm_w_0_re', transfer(real(m_w_0,8),0_int64)
  write(*,'(A,1X,Z16.16)') 'm_w_0_im', transfer(aimag(m_w_0),0_int64)
  write(*,'(A,1X,Z16.16)') 'm_i_0_re', transfer(real(m_i_0,8),0_int64)
  write(*,'(A,1X,Z16.16)') 'm_i_0_im', transfer(aimag(m_i_0),0_int64)
  do n=1,50
    write(*,'(A,I0,1X,Z16.16)') 'xxds_',n,transfer(xxDs(n),0_int64)
  enddo
  do n=1,50
    write(*,'(A,I0,1X,Z16.16)') 'xdts_',n,transfer(xdts(n),0_int64)
  enddo
  do n=1,50
    write(*,'(A,I0,1X,Z16.16)') 'xxdg_',n,transfer(xxDg(n),0_int64)
  enddo
  do n=1,50
    write(*,'(A,I0,1X,Z16.16)') 'xdtg_',n,transfer(xdtg(n),0_int64)
  enddo
  do n=1,50
    write(*,'(A,I0,1X,Z16.16)') 'simpson_',n,transfer(simpson(n),0_int64)
  enddo
end program dump_radar
