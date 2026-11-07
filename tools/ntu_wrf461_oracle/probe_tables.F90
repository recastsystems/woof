program probe
 use module_mp_ntu
 implicit none
 integer :: mode, unit
 character(len=256) :: name, arg
 call get_command_argument(1,arg); read(arg,*) mode
 call get_command_argument(2,name)
 call AERO_CONST(mode)
 open(newunit=unit,file=trim(name),access='stream',form='unformatted',status='replace')
 write(unit) IBAER,TBLRC,TBLXF,DNC0,DNC1,DNC2,DNC3,SENS,ASH,CMAS,DNAS,AVAN,BETA1,RXMIN,ZCCN,CNMOD,CNSTD,WMAS,RFACT
 close(unit)
end program
