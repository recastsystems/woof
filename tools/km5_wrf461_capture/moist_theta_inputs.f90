! Proof-only conversion of the saved dry-theta native inputs. Preserve the
! two forcing endpoints, then difference the moist coupled endpoints.
program moist_theta_inputs
  use netcdf
  implicit none
  integer :: nc, nb, id, nx, ny, nz, n, side, i, k, b, x, y
  character(2), parameter :: sides(4) = ['XS','XE','YS','YE']
  real, parameter :: rvovrd=461.6/287.0, interval=21600.
  real, allocatable :: t(:,:,:),q(:,:,:),mub(:,:),c1(:),c2(:)
  real, allocatable :: tb(:,:,:),qb(:,:,:),td(:,:,:),qd(:,:,:),mu(:,:),mud(:,:)
  real :: mass0,mass1,t0,t1,q0,q1,v0,v1
  call check(nf90_open('wrfinput_d01',nf90_write,nc))
  call dimension(nc,'west_east',nx)
  call dimension(nc,'south_north',ny)
  call dimension(nc,'bottom_top',nz)
  allocate(t(nx,ny,nz),q(nx,ny,nz),mub(nx,ny),c1(nz),c2(nz))
  call check(nf90_inq_varid(nc,'T',id));call check(nf90_get_var(nc,id,t))
  call check(nf90_inq_varid(nc,'QVAPOR',id));call check(nf90_get_var(nc,id,q))
  t=(1.+rvovrd*q)*(t+300.)-300.
  call check(nf90_inq_varid(nc,'THM',id));call check(nf90_put_var(nc,id,t))
  call check(nf90_inq_varid(nc,'MUB',id));call check(nf90_get_var(nc,id,mub))
  call check(nf90_inq_varid(nc,'C1H',id));call check(nf90_get_var(nc,id,c1))
  call check(nf90_inq_varid(nc,'C2H',id));call check(nf90_get_var(nc,id,c2))
  call check(nf90_put_att(nc,nf90_global,'USE_THETA_M',1))
  call check(nf90_close(nc))
  call check(nf90_open('wrfbdy_d01',nf90_write,nb))
  do side=1,4
    n=ny
    if(side>2)n=nx
    allocate(tb(n,nz,5),qb(n,nz,5),td(n,nz,5),qd(n,nz,5),mu(n,5),mud(n,5))
    call get3('T_B'//sides(side),tb);call get3('T_BT'//sides(side),td)
    call get3('QVAPOR_B'//sides(side),qb);call get3('QVAPOR_BT'//sides(side),qd)
    call check(nf90_inq_varid(nb,'MU_B'//sides(side),id));call check(nf90_get_var(nb,id,mu))
    call check(nf90_inq_varid(nb,'MU_BT'//sides(side),id));call check(nf90_get_var(nb,id,mud))
    do b=1,5
      do k=1,nz
        do i=1,n
          select case(side)
            case(1);x=b;y=i
            case(2);x=nx-b+1;y=i
            case(3);x=i;y=b
            case(4);x=i;y=ny-b+1
          end select
          mass0=c1(k)*(mu(i,b)+mub(x,y))+c2(k)
          mass1=c1(k)*(mu(i,b)+interval*mud(i,b)+mub(x,y))+c2(k)
          t0=tb(i,k,b)/mass0;t1=(tb(i,k,b)+interval*td(i,k,b))/mass1
          q0=qb(i,k,b)/mass0;q1=(qb(i,k,b)+interval*qd(i,k,b))/mass1
          v0=((1.+rvovrd*q0)*(t0+300.)-300.)*mass0
          v1=((1.+rvovrd*q1)*(t1+300.)-300.)*mass1
          tb(i,k,b)=v0;td(i,k,b)=(v1-v0)/interval
        enddo
      enddo
    enddo
    call check(nf90_inq_varid(nb,'T_B'//sides(side),id));call check(nf90_put_var(nb,id,tb))
    call check(nf90_inq_varid(nb,'T_BT'//sides(side),id));call check(nf90_put_var(nb,id,td))
    deallocate(tb,qb,td,qd,mu,mud)
  enddo
  call check(nf90_put_att(nb,nf90_global,'USE_THETA_M',1))
  call check(nf90_close(nb))
contains
  subroutine check(status)
    integer :: status
    if(status/=nf90_noerr)then
      print *,trim(nf90_strerror(status));stop 1
    endif
  end subroutine
  subroutine dimension(file,name,length)
    integer :: file,length,d
    character(*) :: name
    call check(nf90_inq_dimid(file,name,d))
    call check(nf90_inquire_dimension(file,d,len=length))
  end subroutine
  subroutine get3(name,a)
    character(*) :: name
    real :: a(:,:,:)
    integer :: v
    call check(nf90_inq_varid(nb,name,v));call check(nf90_get_var(nb,v,a))
  end subroutine
end program
