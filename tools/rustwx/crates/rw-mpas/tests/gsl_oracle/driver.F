program replay
 use mpas_init_atm_gwd_gsl
 use mpas_framework
 implicit none
 type(domain_type) :: domain
 type(block_type),target :: block
 character(1024) :: input,output
 integer :: me, ierr
 call get_command_argument(1,input)
 call get_command_argument(2,geog)
 call get_command_argument(3,output)
 open(10,file=trim(input),access='stream',form='unformatted',status='old',convert='little_endian')
 read(10) nc,ne,me,radius,scaling
 allocate(lat(nc),lon(nc),dc(ne),counts(nc),edges(me,nc),ss(10,nc),ls(10,nc))
 read(10) lat,lon,dc,counts,edges
 close(10)
 block%structs=>pool
 domain%blocklist=>block
 domain%configs=>pool
 ierr=99
 call calc_gsl_oro_data(domain,ierr)
 if(ierr/=0) stop 16
 open(11,file=trim(output),access='stream',form='unformatted',status='replace',convert='little_endian')
 write(11) ls,ss
 close(11)
end program
