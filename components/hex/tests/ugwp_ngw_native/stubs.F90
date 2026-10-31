module ccpp_kind_types
integer, parameter :: kind_phys=4
end module
module mpas_derived_types
integer, parameter :: MPAS_LOG_ERR=1,MPAS_LOG_CRIT=2
end module
module mpas_log
contains
subroutine mpas_log_write(msg,realArgs,intArgs,messageType,masterOnly)
character(len=*),intent(in)::msg
real,optional::realArgs(:)
integer,optional::intArgs(:),messageType
logical,optional::masterOnly
end subroutine
end module
module mpas_atmphys_constants
real,parameter::P0=1.e5
end module
module mpas_constants
real,parameter::pii=3.141592653589793,gravity=9.80616,omega=7.29212e-5,a=6371229.,rgas=287.,rv=461.6,cp=7.*rgas/2.
end module
