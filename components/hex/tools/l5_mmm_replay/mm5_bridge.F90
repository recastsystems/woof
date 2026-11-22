! L5 referee bridge (test tool, never on a run path): bind(c) entry to pristine NCAR/MMM-physics sf_sfclayrev_run.
! Column layout a(n,48), constants c(13): hex tests/sfclay_mm5_oracle/oracle_wrapper.F90 oracle_run (the MPAS v8.4.2
! driver's argument set, every optional present). With -DMMM550 the three MMM 550b5b494758 arguments are passed:
! varf (orographic std-dev, MPAS var2d), if_kim_tofd and tofd_factor (sf_sfclayrev.F90:85,99,110,124).
module l5_mm5_bridge
use iso_c_binding
use sf_sfclayrev
implicit none
contains
subroutine l5_mm5_init() bind(c)
character(256):: msg
integer:: err
call sf_sfclayrev_init(msg,err)
end subroutine
subroutine l5_mm5_run(a,c,n,varf,tofd,tofd_factor) bind(c)
integer(c_int),value:: n,tofd
real(c_float),value:: tofd_factor
real(c_float),intent(inout):: a(n,48)
real(c_float),intent(in):: c(13), varf(n)
character(256):: msg
integer:: err
call sf_sfclayrev_run( &
 ux=a(:,35), vx=a(:,36), t1d=a(:,37), qv1d=a(:,38), p1d=a(:,39), dz8w1d=a(:,40), &
 cp=c(1), g=c(2), rovcp=c(3), r=c(4), xlv=c(5), &
 psfcpa=a(:,41), chs=a(:,20), chs2=a(:,21), cqs2=a(:,22), cpm=a(:,30), pblh=a(:,43), rmol=a(:,26), &
 znt=a(:,1), ust=a(:,2), mavail=a(:,44), zol=a(:,8), mol=a(:,4), regime=a(:,9), psim=a(:,10), psih=a(:,11), &
 fm=a(:,12), fh=a(:,13), xland=a(:,45), hfx=a(:,5), qfx=a(:,6), tsk=a(:,42), u10=a(:,15), v10=a(:,16), &
 th2=a(:,17), t2=a(:,18), q2=a(:,19), flhc=a(:,23), flqc=a(:,24), qgh=a(:,25), qsfc=a(:,7), lh=a(:,14), &
 gz1oz0=a(:,29), wspd=a(:,27), br=a(:,28), dx=a(:,47), &
#ifdef MMM550
 varf=varf, if_kim_tofd=(tofd/=0), tofd_factor=tofd_factor, &
#endif
 svp1=c(6), svp2=c(7), svp3=c(8), svpt0=c(9), ep1=c(10), ep2=c(11), karman=c(12), p1000mb=c(13), &
 lakemask=a(:,46), water_depth=a(:,48), ustm=a(:,3), ck=a(:,31), cka=a(:,32), cd=a(:,33), cda=a(:,34), &
 isfflx=.true.,isftcflx=0,iz0tlnd=0,scm_force_flux=.false.,shalwater_z0=.false.,its=1,ite=n,errmsg=msg,errflg=err)
end subroutine
end module
