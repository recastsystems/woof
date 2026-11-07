! Test-only native forcing generator. Endpoints come from the real initial
! analysis, with explicit perturbations at NEW. This is constructed forcing,
! not an observation dataset and not a qualifying whole-forecast proof.
program make_forcing
  use netcdf
  implicit none
  integer :: src,dst,nx,ny,nattr,i,j,e,v,dimx,dimy,dimxs,dimys,dimt,dimc,tdims(2),dims(3)
  integer :: varid,attid,field,ids(22)
  real,allocatable :: u(:,:),vv(:,:),t(:,:),th(:,:),q(:,:),p(:,:),snow(:,:),a(:,:),b(:,:)
  character(1024) :: input,output,upperair
  character(128) :: att,name
  character(19) :: stamp
  character(4),parameter :: base(11)=[character(4)::'U10','V10','T2','TH2','Q2','RH','PSL','PS','TOB','ODIS','SN']
  character(3),parameter :: endpoint(2)=[character(3)::'OLD','NEW']
  call get_command_argument(1,input);call get_command_argument(2,output)
  call get_command_argument(3,upperair)
  call ck(nf90_open(trim(input),NF90_NOWRITE,src))
  call ck(nf90_inq_dimid(src,'west_east',v));call ck(nf90_inquire_dimension(src,v,len=nx))
  call ck(nf90_inq_dimid(src,'south_north',v));call ck(nf90_inquire_dimension(src,v,len=ny))
  allocate(u(nx,ny),vv(nx,ny),t(nx,ny),th(nx,ny),q(nx,ny),p(nx,ny),snow(nx,ny))
  call read2('U10',u);call read2('V10',vv);call read2('T2',t);call read2('TH2',th)
  call read2('Q2',q);call read2('PSFC',p);call read2('SNOW',snow)
  call ck(nf90_inq_varid(src,'Times',v));call ck(nf90_get_var(src,v,stamp))
  call ck(nf90_create(trim(output),ior(NF90_CLOBBER,NF90_64BIT_OFFSET),dst))
  call ck(nf90_inquire(src,nAttributes=nattr))
  do attid=1,nattr
    call ck(nf90_inq_attname(src,NF90_GLOBAL,attid,att))
    call ck(nf90_copy_att(src,NF90_GLOBAL,trim(att),dst,NF90_GLOBAL))
  end do
  call ck(nf90_put_att(dst,NF90_GLOBAL,'TITLE','WRF SFCFDDA INPUT'))
  call ck(nf90_put_att(dst,NF90_GLOBAL,'SURFACE_ORACLE_FORCING', &
    'Constructed from wrfinput; NEW wind +0.5 m/s, T/TH +1 K, Q *0.95; ODIS 20 km'))
  call ck(nf90_def_dim(dst,'west_east',nx,dimx));call ck(nf90_def_dim(dst,'south_north',ny,dimy))
  call ck(nf90_def_dim(dst,'west_east_stag',nx+1,dimxs))
  call ck(nf90_def_dim(dst,'south_north_stag',ny+1,dimys))
  call ck(nf90_def_dim(dst,'Time',NF90_UNLIMITED,dimt));call ck(nf90_def_dim(dst,'DateStrLen',19,dimc))
  tdims=[dimc,dimt];call ck(nf90_def_var(dst,'Times',NF90_CHAR,tdims,varid))
  do e=1,2
    do field=1,11
      name=trim(base(field))//'_NDG_'//endpoint(e)
      dims=[dimx,dimy,dimt]
      if(field==1) dims(1)=dimxs
      if(field==2) dims(2)=dimys
      call ck(nf90_def_var(dst,trim(name),NF90_FLOAT,dims,ids((e-1)*11+field)))
      ! Native WRF's input inventory uses FieldType/MemoryOrder, not only
      ! the NetCDF dimensions. Missing metadata can silently skip a field.
      call ck(nf90_put_att(dst,ids((e-1)*11+field),'FieldType',104))
      call ck(nf90_put_att(dst,ids((e-1)*11+field),'MemoryOrder','XY '))
      att=''
      if(field==1) att='X'
      if(field==2) att='Y'
      call ck(nf90_put_att(dst,ids((e-1)*11+field),'stagger',trim(att)))
      call ck(nf90_put_att(dst,ids((e-1)*11+field),'description',trim(name)))
    end do
  end do
  call ck(nf90_enddef(dst))
  call ck(nf90_put_var(dst,varid,stamp))
  do e=1,2
    do field=1,11
      if(field==1) then
        allocate(a(nx+1,ny));a(1,:)=u(1,:);a(nx+1,:)=u(nx,:)
        do i=2,nx
          a(i,:)=0.5*(u(i-1,:)+u(i,:))
        end do
      else if(field==2) then
        allocate(a(nx,ny+1));a(:,1)=vv(:,1);a(:,ny+1)=vv(:,ny)
        do j=2,ny
          a(:,j)=0.5*(vv(:,j-1)+vv(:,j))
        end do
      else
        allocate(a(nx,ny))
        select case(field)
        case(3);a=t
        case(4);a=th
        case(5);a=q
        case(7,8);a=p
        case(10);a=20.
        case(11);a=snow
        case default;a=0.
        end select
      end if
      if(e==2) then
        if(field==1.or.field==2) a=a+0.5
        if(field==3.or.field==4) a=a+1.
        if(field==5) a=a*0.95
      end if
      call ck(nf90_put_var(dst,ids((e-1)*11+field),a))
      print *,trim(base(field)),endpoint(e),minval(a),maxval(a)
      deallocate(a)
    end do
  end do
  call ck(nf90_close(dst))
  if(len_trim(upperair)>0) call make_upperair(trim(upperair))
  call ck(nf90_close(src))
contains
  subroutine ck(status)
    integer,intent(in)::status
    if(status/=NF90_NOERR) then
      print *,trim(nf90_strerror(status));stop 1
    end if
  end subroutine
  subroutine read2(name,a)
    character(*),intent(in)::name
    real,intent(out)::a(:,:)
    integer :: var
    call ck(nf90_inq_varid(src,name,var));call ck(nf90_get_var(src,var,a))
  end subroutine
  subroutine make_upperair(path)
    character(*),intent(in)::path
    integer :: dest,nd,n,d,sz,at,rank,srcvar,dstvar,ne,which,nva
    integer :: source_dims(NF90_MAX_VAR_DIMS),new_dims(NF90_MAX_VAR_DIMS),mapping(NF90_MAX_DIMS)
    integer :: lengths(NF90_MAX_VAR_DIMS)
    character(128) :: dname,aname,vname
    character(6),parameter :: original(6)=[character(6)::'U','V','T','QVAPOR','PH','MU']
    character(6),parameter :: ndg(6)=[character(6)::'U_NDG','V_NDG','T_NDG','Q_NDG','PH_NDG','MU_NDG']
    real,allocatable :: b2(:,:),b3(:,:,:)
    call ck(nf90_create(path,ior(NF90_CLOBBER,NF90_64BIT_OFFSET),dest))
    call ck(nf90_inquire(src,nDimensions=nd,nAttributes=n))
    do at=1,n
      call ck(nf90_inq_attname(src,NF90_GLOBAL,at,aname))
      call ck(nf90_copy_att(src,NF90_GLOBAL,trim(aname),dest,NF90_GLOBAL))
    end do
    call ck(nf90_put_att(dest,NF90_GLOBAL,'TITLE','WRF V4.7.1 FDDA INPUT'))
    do d=1,nd
      call ck(nf90_inquire_dimension(src,d,name=dname,len=sz))
      if(trim(dname)=='Time') sz=NF90_UNLIMITED
      call ck(nf90_def_dim(dest,trim(dname),sz,mapping(d)))
    end do
    call ck(nf90_def_var(dest,'Times',NF90_CHAR,[mapping(dim_id('DateStrLen')),mapping(dim_id('Time'))],dstvar))
    call ck(nf90_enddef(dest));call ck(nf90_put_var(dest,dstvar,stamp))
    do which=1,6
      call ck(nf90_inq_varid(src,trim(original(which)),srcvar))
      call ck(nf90_inquire_variable(src,srcvar,ndims=rank,dimids=source_dims))
      do d=1,rank
        new_dims(d)=mapping(source_dims(d))
        call ck(nf90_inquire_dimension(src,source_dims(d),len=lengths(d)))
      end do
      if(rank==4) then
        allocate(b3(lengths(1),lengths(2),lengths(3)))
        call ck(nf90_get_var(src,srcvar,b3))
      else if(rank==3) then
        allocate(b2(lengths(1),lengths(2)))
        call ck(nf90_get_var(src,srcvar,b2))
      else
        stop 'unexpected native upperair variable rank'
      end if
      do ne=1,2
        call ck(nf90_redef(dest))
        vname=trim(ndg(which))//'_'//endpoint(ne)
        call ck(nf90_def_var(dest,trim(vname),NF90_FLOAT,new_dims(:rank),dstvar))
        call ck(nf90_inquire_variable(src,srcvar,natts=nva))
        do at=1,nva
          call ck(nf90_inq_attname(src,srcvar,at,aname))
          call ck(nf90_copy_att(src,srcvar,trim(aname),dest,dstvar))
        end do
        call ck(nf90_enddef(dest))
        if(rank==4) then
          call ck(nf90_put_var(dest,dstvar,b3))
        else
          call ck(nf90_put_var(dest,dstvar,b2))
        end if
      end do
      if(allocated(b3)) deallocate(b3)
      if(allocated(b2)) deallocate(b2)
    end do
    call ck(nf90_close(dest))
  end subroutine
  integer function dim_id(name)
    character(*),intent(in)::name
    call ck(nf90_inq_dimid(src,name,dim_id))
  end function
end program
