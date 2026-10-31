! Framework-only stubs. The three GSL modules themselves are pristine.
module mpas_kind_types
 implicit none
 integer,parameter :: RKIND=RK, R4KIND=4, StrKIND=1024
end module
module mpas_derived_types
 use mpas_kind_types
 implicit none
 integer,parameter :: MPAS_LOG_ERR=1
 type mpas_pool_type
  integer :: unused
 end type
 type block_type
  type(mpas_pool_type),pointer :: structs
 end type
 type domain_type
  type(block_type),pointer :: blocklist
  type(mpas_pool_type),pointer :: configs
 end type
end module
module mpas_framework
 use mpas_derived_types
 implicit none
 integer,target :: nc,ne
 integer,allocatable,target :: counts(:),edges(:,:)
 real(kind=RKIND),target :: radius,scaling
 real(kind=RKIND),allocatable,target :: lat(:),lon(:),dc(:),ss(:,:),ls(:,:)
 character(len=StrKIND),target :: geog
 type(mpas_pool_type),target :: pool
 interface mpas_pool_get_array
  module procedure get_real,get_int1,get_int2
 end interface
 interface mpas_pool_get_config
  module procedure get_config_real,get_config_string
 end interface
 contains
 subroutine mpas_pool_get_subpool(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  type(mpas_pool_type),pointer :: out
  out=>pool
 end subroutine
 subroutine mpas_pool_get_dimension(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  integer,pointer :: out
  select case(name)
  case('nCells'); out=>nc
  case('nEdges'); out=>ne
  case default; stop 10
  end select
 end subroutine
 subroutine get_config_real(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  real(kind=RKIND),pointer :: out
  select case(name)
  case('sphere_radius'); out=>radius
  case('config_gwd_cell_scaling'); out=>scaling
  case default; stop 11
  end select
 end subroutine
 subroutine get_config_string(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  character(len=StrKIND),pointer :: out
  if(name/='config_geog_data_path') stop 12
  out=>geog
 end subroutine
 subroutine get_int1(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  integer,pointer :: out(:)
  if(name/='nEdgesOnCell') stop 13
  out=>counts
 end subroutine
 subroutine get_int2(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  integer,pointer :: out(:,:)
  if(name/='edgesOnCell') stop 14
  out=>edges
 end subroutine
 subroutine get_real(p,name,out)
  type(mpas_pool_type) :: p
  character(*) :: name
  real(kind=RKIND),pointer :: out(:)
  select case(name)
  case('latCell');out=>lat
  case('lonCell');out=>lon
  case('dcEdge');out=>dc
  case('var2dss');out=>ss(1,:)
  case('conss');out=>ss(2,:)
  case('oa1ss');out=>ss(3,:)
  case('oa2ss');out=>ss(4,:)
  case('oa3ss');out=>ss(5,:)
  case('oa4ss');out=>ss(6,:)
  case('ol1ss');out=>ss(7,:)
  case('ol2ss');out=>ss(8,:)
  case('ol3ss');out=>ss(9,:)
  case('ol4ss');out=>ss(10,:)
  case('var2dls');out=>ls(1,:)
  case('conls');out=>ls(2,:)
  case('oa1ls');out=>ls(3,:)
  case('oa2ls');out=>ls(4,:)
  case('oa3ls');out=>ls(5,:)
  case('oa4ls');out=>ls(6,:)
  case('ol1ls');out=>ls(7,:)
  case('ol2ls');out=>ls(8,:)
  case('ol3ls');out=>ls(9,:)
  case('ol4ls');out=>ls(10,:)
  case default; stop 15
  end select
 end subroutine
end module
module mpas_log
 use mpas_kind_types
 contains
 subroutine mpas_log_write(message,messageType,realArgs)
  character(*) :: message
  integer,optional :: messageType
  real(kind=RKIND),optional :: realArgs(:)
 end subroutine
end module
module mpas_c_interfacing
 use iso_c_binding
 contains
 subroutine mpas_f_to_c_string(input,output)
  character(*) :: input
  character(kind=c_char) :: output(:)
  integer :: i
  do i=1,len_trim(input)
   output(i)=input(i:i)
  end do
  output(len_trim(input)+1)=c_null_char
 end subroutine
end module
module mpas_timekeeping
end module
module mpas_timer
end module
module mpas_stream_manager
end module
