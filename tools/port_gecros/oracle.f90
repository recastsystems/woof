program oracle
use module_sf_gecros
implicit none
integer :: i,n,k,ios
real :: x(18),s(60),y(6),fpro,fcar,laii,slnbi
character(len=256)::input,output
call get_command_argument(1,input)
call get_command_argument(2,output)
open(10,file=trim(input),access='stream',form='unformatted',status='old')
open(11,file=trim(output),access='stream',form='unformatted',status='replace')
               hti    = 0.01
               rdi    = 10.
               fpro   = 6.25*seednc
               lncmin = sla0*slnmin
               fcar   = 1.-fpro-ffat-flig-foac-fmin
               cfo    = 0.444*fcar+0.531*fpro+0.774*ffat+0.667*flig+0.368*foac
               clvi   = npl * seedw * cfo * eg * fcrsh
               crti   = npl * seedw * cfo * eg * (1.-fcrsh)
               ygo    = cfo/(1.275*fcar+1.887*fpro+3.189*ffat+2.231*flig+0.954* &
                        foac)*30./12.
               nlvi   = min(0.75 * npl * seedw * eg * seednc, lnci * clvi/cfv)
               laii   = clvi/cfv*sla0
               nrti   = npl * seedw * eg * seednc - nlvi
               slnbi  = nlvi/laii
    

do
 read(10,iostat=ios)x,s
 if(ios/=0)exit
 call gecros(x(1),x(2),1,x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10),x(11),x(12),x(13),x(14),x(15),x(16),x(17),s,y(1),y(2),y(3),y(4),y(5),y(6))
 write(11)s,y
enddo
close(10);close(11)
open(12,file='initial.bin',access='stream',form='unformatted',status='replace')
s=0.0
call gecros_init(40.,hti,rdi,clvi,crti,nlvi,laii,nrti,slnbi,s)
write(12)s
close(12)
contains
SUBROUTINE gecros_init(xlat,hti,rdi,clvi,crti,nlvi,laii,nrti,slnbi,state_gecros)   
implicit none
REAL, INTENT(IN)     :: HTI
REAL, INTENT(IN)     :: RDI
REAL, INTENT(IN)     :: CLVI
REAL, INTENT(IN)     :: CRTI
REAL, INTENT(IN)     :: NLVI
REAL, INTENT(IN)     :: LAII
REAL, INTENT(IN)     :: NRTI
REAL, INTENT(IN)     :: SLNBI
REAL, INTENT(IN)     :: XLAT
REAL, DIMENSION(1:60), INTENT(INOUT) :: STATE_GECROS

  !Inititalization of Gecros variables  
  STATE_GECROS(1) = 0.      !DS
  STATE_GECROS(2) = 0.      !CTDURDI, HTI, CLVI, CRTI, NLVI, LAII, NRTI, SLNBI,
  STATE_GECROS(3) = 0.      !CVDU
  STATE_GECROS(4) = CLVI    !CLV 
  STATE_GECROS(5) = 0.      !CLVD  
  STATE_GECROS(6) = 0.      !CSST  
  STATE_GECROS(7) = 0.      !CSO
  STATE_GECROS(8) = CRTI    !CSRT 
  STATE_GECROS(9) =  0.     !CRTD 
  STATE_GECROS(10) = 0.     !CLVDS 
  STATE_GECROS(11) = NRTI   !NRT
  STATE_GECROS(12) = 0.     !NST
  STATE_GECROS(13) = NLVI   !NLV 
  STATE_GECROS(14) = 0.     !NSO
  STATE_GECROS(15) = NLVI   !TNLV
  STATE_GECROS(16) = 0.     !NLVD
  STATE_GECROS(17) = 0.     !NRTD
  STATE_GECROS(18) = 0.     !CRVS
  STATE_GECROS(19) = 0.     !CRVR
  STATE_GECROS(20) = 0.     !NREOE
  STATE_GECROS(21) = 0.     !NREOF
  STATE_GECROS(22) = 0.     !DCDSR
  STATE_GECROS(23) = 0.     !DCDTR
  STATE_GECROS(24) = SLNBI  !SLNB
  STATE_GECROS(25) = LAII   !LAIC
  STATE_GECROS(26) = 0.     !RMUL
  STATE_GECROS(27) = 0.     !NDEMP
  STATE_GECROS(28) = 0.     !NSUPP
  STATE_GECROS(29) = 0.     !NFIXT
  STATE_GECROS(30) = 0.     !NFIXR
  STATE_GECROS(31) = 0.     !DCDTP
  STATE_GECROS(32) = 0.01   !HTI
  STATE_GECROS(33) = RDI    !RDI
  STATE_GECROS(34) = 0.     !TPCAN
  STATE_GECROS(35) = 0.     !TRESP
  STATE_GECROS(36) = 0.     !TNUPT
  STATE_GECROS(37) = 0.     !LITNT
  STATE_GECROS(38) = 0.     !daysSinceDS1
  STATE_GECROS(39) = 0.     !daysSinceDS2
  STATE_GECROS(40) = -1.    !drilled -1:false, 1:true
  STATE_GECROS(41) = -1.    !emerged -1:false, 1:true
  STATE_GECROS(42) = -1.    !harvested -1:false, 1:true
  STATE_GECROS(43) = 0.     !TTEM
  STATE_GECROS(44) = XLAT   !GLAT
  STATE_GECROS(45) = 0.     !WSO
  STATE_GECROS(46) = 0.     !WSTRAW
  STATE_GECROS(47) = 0.     !GrainNC
  STATE_GECROS(48) = 0.     !StrawNC
  STATE_GECROS(49) = 0.01   !GLAI
  STATE_GECROS(50) = 0.01   !TLAI
  STATE_GECROS(51) = HTI    !Fields 51-58 set for reinitialization  
  STATE_GECROS(52) = RDI    
  STATE_GECROS(53) = CLVI   
  STATE_GECROS(54) = CRTI   
  STATE_GECROS(55) = NRTI   
  STATE_GECROS(56) = NLVI   
  STATE_GECROS(57) = SLNBI  
  STATE_GECROS(58) = LAII   
    
END SUBROUTINE gecros_init
end program oracle
