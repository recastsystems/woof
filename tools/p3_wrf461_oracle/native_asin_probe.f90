program probe
 implicit none
 integer :: i, words(3), output(3)
 real :: values(3), result(3)
 words = [int(z'bd137442'),int(z'bd222103'),int(z'bd289002')]
 values = transfer(words,values)
 do i=1,3
  result(i) = asin(values(i))
 end do
 output = transfer(result, output)
 write(*,'(3(z8.8,1x))') output
end program
