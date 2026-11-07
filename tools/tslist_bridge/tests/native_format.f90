! Formats transcribed unchanged from pinned WRF wrf_timeseries.F.
! This exercises the compiler's actual formatted WRITE, not a text fixture.
program native_format
  use, intrinsic :: ieee_arithmetic
  implicit none
  real :: hour, vals(14)
  integer :: n, u
  character(25) :: description
  character(5) :: prefix
  character(26) :: profile_format
  character(7) :: fields(6)
  real :: coordinates(6)
  description='Native format oracle'
  prefix='proof'
  hour=1.0/3.0
  vals=[300.12345,0.006789,-10.125,20.5,99876.5,400.,500.,-0.0,0.000005, &
        1.e20,-1.e20,ieee_value(0.0,ieee_quiet_nan), &
        ieee_value(0.0,ieee_positive_inf),ieee_value(0.0,ieee_negative_inf)]
  open(newunit=u,file='native-values.bin',access='stream',form='unformatted',status='new')
  write(u) hour,vals
  close(u)
  open(newunit=u,file='native-text.txt',form='formatted',status='new')
  write(u,'(A26,I2,I3,A6,A2,F7.3,A1,F8.3,A3,I4,A1,I4,A3,F7.3,A1,F8.3,A2,F6.1,A7,A2,A19)') &
    description//' ',2,17,' '//prefix, &
    ' (',35.125,',',-97.125,') (',35,',',30,') (',35.123,',',-97.123,') ', &
    321.5,' meters','  ','2000-01-24_12:00:00'
  write(u,'(i2,f13.6,i5,i5,i5,1x,14(f13.5,1x))') 2,hour,17,35,30,vals
  profile_format='(f13.6,1x,000(f13.5,1x))'
  write(profile_format(12:14),'(I3.3)') 14
  write(u,profile_format) hour,vals
  close(u)
  fields=['12345-6',' 12D-2 ','-1.2+1 ','3 51 25','1.25E-3','  -0   ']
  do n=1,6
    read(fields(n),'(f7.3)') coordinates(n)
  end do
  open(newunit=u,file='native-read.bin',access='stream',form='unformatted',status='new')
  write(u) coordinates
  close(u)
end program
