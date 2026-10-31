#ifndef ORACLE_ENTRY
#define ORACLE_ENTRY oracle
#define ORACLE_KDT 2
#define ORACLE_DIAGS .true.
#endif
subroutine ORACLE_ENTRY(levs,im,dtp,amp,day,rdzw,dzu,lat,taulat,days,tau,ntlat,ntday, &
                  u,v,t,q,p,exner,pi,z,zi,rc,rn,ru,rv,rt,du,dv,dt,dk,zh,state,j1,j2,w1,w2) bind(C)
use iso_c_binding
use bl_ugwpv1_ngw
use cires_ugwpv1_solv2
use ugwp_common,only:pi2
use ugwp_wmsdis_init,only:zci,zdci,zcosang,zsinang
use cires_ugwpv1_module,only:kvg,kion,cires_ugwp_dealloc
implicit none
integer(c_int),value::levs,im,ntlat,ntday
real(c_float),value::dtp,amp,day
real(c_float)::rdzw(levs),dzu(levs),lat(im),taulat(ntlat),days(ntday),tau(ntlat,ntday)
real(c_float)::u(im,levs),v(im,levs),t(im,levs),q(im,levs),p(im,levs),exner(im,levs),pi(im,levs+1)
real(c_float)::z(im,levs),zi(im,levs+1),rc(im),rn(im),ru(im,levs),rv(im,levs),rt(im,levs)
real(c_float)::du(im,levs),dv(im,levs),dt(im,levs),dk(im,levs),zh(im),state(4*58+2*(levs+1))
integer(c_int)::j1(im),j2(im)
real(c_float)::w1(im),w2(im),xlat(im),tau_ngw(im),tau_ddd(im),rain(im)
call ugwpv1_ngw_init(lat,levs,dtp,rdzw,dzu,ntlat,amp,taulat,j1,j2,w1,w2)
state(1:25)=zci
state(26:50)=zdci
state(51:54)=zcosang
state(55:58)=zsinang
state(59:59+levs)=kvg
state(60+levs:60+2*levs)=kion
block
use ugwp_conv_init,only:ch_conv,spf_conv,xaz_conv,yaz_conv
use ugwp_fjet_init,only:ch_fjet,spf_fjet,xaz_fjet,yaz_fjet
use ugwp_okw_init,only:ch_okwp,spf_okwp,xaz_okwp,yaz_okwp
integer::off
off=58+2*(levs+1)
state(off+1:off+25)=ch_conv
state(off+26:off+50)=spf_conv
state(off+51:off+54)=xaz_conv
state(off+55:off+58)=yaz_conv
off=off+58
state(off+1:off+25)=ch_fjet
state(off+26:off+50)=spf_fjet
state(off+51:off+54)=xaz_fjet
state(off+55:off+58)=yaz_fjet
off=off+58
state(off+1:off+25)=ch_okwp
state(off+26:off+50)=spf_okwp
state(off+51:off+54)=xaz_okwp
state(off+55:off+58)=yaz_okwp
end block
xlat=lat*(180./3.1415926)
call ugwpv1_ngw_run(xlatd=xlat,ddy_j1tau=w1,ddy_j2tau=w2,raincv=rc,rainncv=rn, &
ugrs=u,vgrs=v,tgrs=t,q1=q,prsl=p,prslk=exner,prsi=pi,zl=z,zi=zi,r_DoY=day, &
kdt=ORACLE_KDT,dtp=dtp,ntau_d2t=ntday,days_limb=days,tau_limb=tau,ugwp_diags=ORACLE_DIAGS, &
jindx1_tau=j1,jindx2_tau=j2,rublten=ru,rvblten=rv,rthblten=rt,its=1,ite=im,levs=levs, &
dudt_ngw=du,dvdt_ngw=dv,dtdt_ngw=dt)
! Inspect all solver outputs as well as real driver outputs.
block
use cires_tauamf_data,only:tau_amf_interp
use cires_ugwpv1_triggers,only:slat_geos5_2020
use cires_ugwpv1_module,only:ngwflux_update
call tau_amf_interp(im,day,j1,j2,w1,w2,ntday,days,tau,tau_ddd)
call slat_geos5_2020(im,amp,xlat,tau_ngw)
rain=(rc+rn)*.001
call ngwflux_update(im,tau_ddd,xlat,rain,tau_ngw)
call cires_ugwpv1_ngw_solv2(im,levs,ORACLE_KDT,dtp,tau_ngw,t,u,v,q,p,pi,z,zi,exner,xlat,du,dv,dt,dk,zh)
end block
call cires_ugwp_dealloc
! Pristine finalize omits these SAVE allocatables; clear for repeated test calls.
deallocate(zci,zdci,zcosang,zsinang)
block
use ugwp_wmsdis_init,only:zci2,zci3,zci4,lzmet,czmet,mkzmet,dczmet,dmkz
deallocate(zci2,zci3,zci4,lzmet,czmet,mkzmet,dczmet,dmkz)
end block
end subroutine
