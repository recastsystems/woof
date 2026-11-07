program columns
 use module_ra_hs
 use module_damping_em
 implicit none
 integer,parameter :: nx=8,ny=8,nz=8
 real :: p(0:nx+1,nz+1,0:ny+1),pb(0:nx+1,nz+1,0:ny+1),pw(0:nx+1,nz+1,0:ny+1)
 real :: pi(0:nx+1,nz+1,0:ny+1),t(0:nx+1,nz+1,0:ny+1),lat(0:nx+1,0:ny+1)
 real :: rt(0:nx+1,nz+1,0:ny+1),ru(0:nx+1,nz+1,0:ny+1),rv(0:nx+1,nz+1,0:ny+1)
 real :: ut(0:nx+1,nz+1,0:ny+1),vt(0:nx+1,nz+1,0:ny+1),r0(0:nx+1,nz+1,0:ny+1)
 real :: u0(0:nx+1,nz+1,0:ny+1),v0(0:nx+1,nz+1,0:ny+1)
 real :: sigma(nz+1),degrad
 integer :: i,j,k
 sigma=(/1.,.95,.8,.70001,.7,.69999,.3,.01,.001/)
 degrad=3.14159265358979323846/180.
 do j=0,ny+1
 do i=0,nx+1
 lat(i,j)=-90.+180.*real(modulo((j-1)*nx+i-1,nx*ny))/real(nx*ny-1)
 if(i==4.and.j==4) lat(i,j)=0.
 do k=1,nz+1
 pb(i,k,j)=60000.*sigma(k)
 p(i,k,j)=(40000.+real(17*i+31*j))*sigma(k)
 pw(i,k,j)=100000.+real(17*i+31*j)
 pi(i,k,j)=((p(i,k,j)+pb(i,k,j))/100000.)**(287./1004.5)
 t(i,k,j)=185.+real(modulo(7*i+11*j+19*k,150))
 rt(i,k,j)=real(i-2*j+k)*1.e-5
 ru(i,k,j)=real(3*i-2*j+k)*70000.
 rv(i,k,j)=real(2*i+3*j-k)*65000.
 ut(i,k,j)=real(i+j-k)*.01
 vt(i,k,j)=real(i-j+k)*.02
 end do
 end do
 end do
 r0=rt;u0=ut;v0=vt
 call hsinit(rt,.true.,1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz)
 if(any(rt/=r0)) stop 10
 call HSRAD(rt,pw,p+pb,pi,pw,t,pw,pw,287.,9.81,1004.5,60.,lat,degrad, &
 1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz)
 call held_suarez_damp(ut,vt,ru,rv,p,pb, &
 1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz)
 open(10,file='columns.hex',status='replace')
 write(10,'(A)') '# i j k lat p pb p8w pi t r0 rth ru rv u0 v0 ut vt, float32 hex words'
 do j=0,ny+1
 do k=1,nz+1
 do i=0,nx+1
 write(10,'(3(I3,1X),15(Z8.8,1X))') i,j,k,transfer(lat(i,j),0),transfer(p(i,k,j),0), &
 transfer(pb(i,k,j),0),transfer(pw(i,k,j),0),transfer(pi(i,k,j),0),transfer(t(i,k,j),0), &
 transfer(r0(i,k,j),0),transfer(rt(i,k,j),0),transfer(ru(i,k,j),0),transfer(rv(i,k,j),0), &
 transfer(u0(i,k,j),0),transfer(v0(i,k,j),0),transfer(ut(i,k,j),0),transfer(vt(i,k,j),0),transfer(degrad,0)
 end do
 end do
 end do
 close(10)
 rt=r0
 call hsinit(rt,.false.,1,nx+1,1,ny+1,1,nz+1,0,nx+1,0,ny+1,1,nz+1,1,nx,1,ny,1,nz)
 if(any(rt(1:nx,1:nz,1:ny)/=0.)) stop 11
 print *, 'PASS 64 columns x 8 levels; hsinit restart and cold start'
end program
