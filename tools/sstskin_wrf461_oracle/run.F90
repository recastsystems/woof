program column_oracle
use module_sf_sstskin
implicit none
integer :: n, ns, i, s, update
real :: dt, stbolt
real, allocatable :: x(:,:), glw(:,:), sw(:,:), hfx(:,:), qfx(:,:), tsk(:,:), ust(:,:), emiss(:,:), sst(:,:), dtw(:,:), skin(:,:), soil(:,:)
character(1024) :: input, output
call get_command_argument(1,input)
call get_command_argument(2,output)
open(10,file=trim(input),access='stream',form='unformatted',status='old')
open(11,file=trim(output),access='stream',form='unformatted',status='replace')
read(10) n,ns,update,dt,stbolt
allocate(x(n,1),glw(n,1),sw(n,1),hfx(n,1),qfx(n,1),tsk(n,1),ust(n,1),emiss(n,1),sst(n,1),dtw(n,1),skin(n,1),soil(n,1))
read(10) x,tsk,ust,emiss,sst,dtw,skin,soil
do s=1,ns
read(10) glw,sw,hfx,qfx
do i=1,n
if(x(i,1)>1.5.and.update/=1) then
tsk(i,1)=sst(i,1)
soil(i,1)=sst(i,1)
endif
enddo
call sst_skin_update(x,glw,sw,hfx,qfx,tsk,ust,emiss,dtw,skin,dt,stbolt,1,n,1,1,1,1,1,n,1,1,1,1,1,n,1,1,1,1)
do i=1,n
if(x(i,1)>1.5) tsk(i,1)=skin(i,1)
enddo
write(11) skin,dtw,tsk,soil
enddo
close(10)
close(11)
end program
