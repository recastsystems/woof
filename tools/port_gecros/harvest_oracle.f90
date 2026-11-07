program oracle_harvest
implicit none
real :: s(60),dt,crop,ds1,ds2,extra2
integer :: ios
character(len=256):: input,output
call get_command_argument(1,input)
call get_command_argument(2,output)
open(10,file=trim(input),access='stream',form='unformatted',status='old')
open(11,file=trim(output),access='stream',form='unformatted',status='replace')
do
 read(10,iostat=ios)dt,crop,s
 if(ios/=0)exit
 if(nint(crop)==1)then
 ds1=2.1;ds2=2.;extra2=10.
 else
 ds1=1.78;ds2=1.63;extra2=14.
 endif
 if(s(1)>=ds1.and.s(42)<0)then
 if(checkIfHarvest(s,dt,ds1,ds2,0.,extra2)==1)call gecros_reinit(s)
 endif
 write(11)s
enddo
close(10);close(11)
contains
SUBROUTINE gecros_reinit(STATE_GECROS)   
implicit none
REAL, DIMENSION(1:60), INTENT(INOUT) :: STATE_GECROS

  !Re-inititalization of Gecros variables after harvest
  STATE_GECROS(1) = 0.               !DS
  STATE_GECROS(2) = 0.               !CTDU
  STATE_GECROS(3) = 0.               !CVDU
  STATE_GECROS(4) = STATE_GECROS(53) !CLV 
  STATE_GECROS(5) = 0.               !CLVD  
  STATE_GECROS(6) = 0.               !CSST  
  STATE_GECROS(7) = 0.               !CSO
  STATE_GECROS(8) = STATE_GECROS(54) !CRT 
  STATE_GECROS(9) = 0.               !CRTD 
  STATE_GECROS(10) = 0.              !CLVDS 
  STATE_GECROS(11) = STATE_GECROS(55)!NRT
  STATE_GECROS(12) = 0.              !NST
  STATE_GECROS(13) = STATE_GECROS(56)!NLV 
  STATE_GECROS(14) = 0.              !NSO
  STATE_GECROS(15) = STATE_GECROS(56)!TNLV
  STATE_GECROS(16) = 0.              !NLVD
  STATE_GECROS(17) = 0.              !NRTD
  STATE_GECROS(18) = 0.              !CRVS
  STATE_GECROS(19) = 0.              !CRVR
  STATE_GECROS(20) = 0.              !NREOE
  STATE_GECROS(21) = 0.              !NREOF
  STATE_GECROS(22) = 0.              !DCDSR
  STATE_GECROS(23) = 0.              !DCDTR
  STATE_GECROS(24) = STATE_GECROS(57)!SLNB
  STATE_GECROS(25) = STATE_GECROS(58)!LAIC
  STATE_GECROS(26) = 0.              !RMUL
  STATE_GECROS(27) = 0.              !NDEMP
  STATE_GECROS(28) = 0.              !NSUPP
  STATE_GECROS(29) = 0.              !NFIXT
  STATE_GECROS(30) = 0.              !NFIXR
  STATE_GECROS(31) = 0.              !DCDTP
  STATE_GECROS(32) = STATE_GECROS(51)!HT
  STATE_GECROS(33) = STATE_GECROS(52)!ROOTD
  STATE_GECROS(34) = 0.              !TPCAN
  STATE_GECROS(35) = 0.              !TRESP
  STATE_GECROS(36) = 0.              !TNUPT
  STATE_GECROS(37) = 0.              !LITNT
  STATE_GECROS(38) = 0.              !daysSinceDS1
  STATE_GECROS(39) = 0.              !daysSinceDS2
  STATE_GECROS(40) = -1.             !drilled -1:false, 1:true
  STATE_GECROS(41) = -1.             !emerged -1:false, 1:true
  STATE_GECROS(42) = 1.              !harvested -1:false, 1:true
  STATE_GECROS(43) = 0.              !TTEM
  STATE_GECROS(45) = 0.              !WSO
  STATE_GECROS(46) = 0.              !WSTRAW
  STATE_GECROS(47) = 0.              !GrainNC
  STATE_GECROS(48) = 0.              !StrawNC
  STATE_GECROS(49) = 0.01            !GLAI
  STATE_GECROS(50) = 0.01            !TLAI
  
END SUBROUTINE gecros_reinit
function checkIfHarvest(STATE_GECROS, DT, harvestDS1, harvestDS2, harvestDS1ExtraDays, harvestDS2ExtraDays) 
implicit none
real :: DT, harvestDS1, harvestDS2
real :: daysSinceDS1, daysSinceDS2
real :: harvestDS1ExtraDays, harvestDS2ExtraDays
integer :: checkIfHarvest
REAL, DIMENSION(1:60), INTENT(INOUT) :: STATE_GECROS


 !***check whether maturity (DS1) has been reached
 if (STATE_GECROS(1) >= harvestDS1) then
 
    if (STATE_GECROS(38) >= harvestDS1ExtraDays) then
        checkIfHarvest=1
 !if we are > DS1, but not over the limit, increase the counter of days
    else
        STATE_GECROS(38) = STATE_GECROS(38) + DT/86400.
    endif
 else 
 
 !if maturity has not been reached, but we are close (> DS2)
 !check the number of days for which we have been > DS2
 !and harvest in case we are over the limit given for that stage 
 !(in case that maturity will not be reached at all)
 
 checkIfHarvest=0
 if (STATE_GECROS(1) >= harvestDS2 ) then

       if (STATE_GECROS(39) >= harvestDS2ExtraDays) then
           checkIfHarvest=1
       else !if we are > DS2, but not over the limit, increase the counter of days
           STATE_GECROS(39) = STATE_GECROS(39) + DT/86400.
           checkIfHarvest=0
      endif
 endif
 endif
 return
end function checkIfHarvest
end program
