program clock_oracle
use module_date_time, only: get_julgmt,split_date_char
implicit none
character(len=24) :: start_date,valid_date
integer :: c,year,day,ny,nm,nd,nh,ni,ns,nt,ios,days,whole
real :: gmt,unused,jul,xt
real(kind=8) :: elapsed,seconds,day_r8,fraction
do
 read(*,*,iostat=ios)c,start_date,valid_date,elapsed
 if(ios/=0)exit
 call get_julgmt(start_date,year,day,gmt)
 call get_julgmt(valid_date,year,day,unused)
 call split_date_char(valid_date,ny,nm,nd,nh,ni,ns,nt)
 seconds=real((day-1)*86400,8)+real(nh*3600+ni*60+ns,8)+real(nt,8)/10000.d0
 day_r8=seconds/86400.d0+1.d0
 jul=real(day_r8)-1.0
 days=int(elapsed/86400.d0)
 whole=int(elapsed-real(days,8)*86400.d0)
 xt=(real(days)*24.*60.)+(real(whole)/60.)
 fraction=elapsed-real(days,8)*86400.d0-real(whole,8)
 if(fraction/=0.d0)xt=xt+real(fraction)/60.
 write(*,'(i0,3(1x,es24.16e3))')c,jul,xt,gmt
enddo
end program
