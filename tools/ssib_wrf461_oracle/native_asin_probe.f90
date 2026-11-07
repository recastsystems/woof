program native_asin_probe
 use iso_fortran_env,only:int32,real32
 implicit none
 integer(int32) :: words(3), result(3)
 real(real32) :: values(3)
 integer :: k
 words=[int(z'bd137442',int32),int(z'bd222103',int32),int(z'bd289002',int32)]
 values=transfer(words,values)
 do k=1,3
   values(k)=asin(values(k))
 enddo
 result=transfer(values,result)
 write(*,'(3(z8.8,1x))') result
end program
