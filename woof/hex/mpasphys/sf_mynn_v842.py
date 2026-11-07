"""MPAS-A v8.4.2 MYNN surface layer, single-precision CUDA transcription.

Source: physics_mmm/sf_mynn.F90:1-2237 (MMM a4baf7f),
physics_wrf/sf_mynn_pre.F:1-169, physics_wrf/module_sf_mynn.F:19-390,
and mpas_atmphys_driver_sfclayer.F:352-796,839-1088 (MPAS f34984b).
The original WOOF surface backend remains independently selectable.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target
from woof.hex import numerics as _hex_numerics

from collections import ChainMap
from collections.abc import Mapping
from importlib.resources import files
import numpy as np

SCHEME = "sf_mynn"
BACKEND = "hex-cuda"
FIELDS = ('u1d', 'v1d', 't1d', 'qv1d', 'p1d', 'dz8w1d', 'rho1d', 'u1d2', 'v1d2', 'dz2w1d', 'psfcpa', 'chs', 'chs2', 'cqs2', 'cpm', 'pblh', 'rmol', 'znt', 'ust', 'mavail', 'zol', 'mol', 'regime', 'psim', 'psih', 'xland', 'hfx', 'qfx', 'tsk', 'u10', 'v10', 'th2', 't2', 'q2', 'flhc', 'flqc', 'snowh', 'qgh', 'qsfc', 'lh', 'gz1oz0', 'wspd', 'br', 'dx', 'ch', 'qcg', 'wstar', 'qstar', 'ustm', 'ck', 'cka', 'cd', 'cda', 'rstoch1d')
OUTPUTS = ('chs', 'chs2', 'cqs2', 'cpm', 'rmol', 'znt', 'ust', 'zol', 'mol', 'regime', 'psim', 'psih', 'hfx', 'qfx', 'u10', 'v10', 'th2', 't2', 'q2', 'flhc', 'flqc', 'qgh', 'qsfc', 'lh', 'gz1oz0', 'wspd', 'br', 'ch', 'wstar', 'qstar', 'ustm', 'ck', 'cka', 'cd', 'cda')
CONSTANTS = ('cp', 'g', 'rovcp', 'r', 'xlv', 'svp1', 'svp2', 'svp3', 'svpt0', 'ep1', 'ep2', 'karman')
OPTIONAL = ('isftcflx', 'iz0tlnd', 'ck', 'cka', 'cd', 'cda', 'ustm')
SOURCE = r"""
__device__ float psim_stab[1001];
__device__ float psim_unstab[1001];
__device__ float psih_stab[1001];
__device__ float psih_unstab[1001];
// MPAS physics_mmm/sf_mynn.F90:130-996 sf_mynn_run
__device__ __noinline__ void m_sf_mynn_run(float u1d, float v1d, float t1d, float qv1d, float p1d, float dz8w1d, float rho1d, float u1d2, float v1d2, float dz2w1d, float cp, float g, float rovcp, float r, float xlv, float psfcpa, float &chs, float &chs2, float &cqs2, float &cpm, float pblh, float &rmol, float &znt, float &ust, float mavail, float &zol, float &mol, float &regime, float &psim, float &psih, float xland, float &hfx, float &qfx, float tsk, float &u10, float &v10, float &th2, float &t2, float &q2, float &flhc, float &flqc, float snowh, float &qgh, float &qsfc, float &lh, float &gz1oz0, float &wspd, float &br, int isfflx, float dx, float svp1, float svp2, float svp3, float svpt0, float ep1, float ep2, float karman, float &ch, float qcg, int itimestep, float &wstar, float &qstar, float &ustm, float &ck, float &cka, float &cd, float &cda, bool spp_pbl, float rstoch1d, int isftcflx, int iz0tlnd, int its, int ite, bool has_isftcflx, bool has_iz0tlnd, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm);
// MPAS physics_mmm/sf_mynn.F90:999-1073 zilitinkevich_1995
__device__ __noinline__ void m_zilitinkevich_1995(float z_0, float &zt, float &zq, float restar, float ustar, float karman, float landsea, int iz0tlnd2, bool spp_pbl, float rstoch, bool has_iz0tlnd2);
// MPAS physics_mmm/sf_mynn.F90:1076-1109 davis_etal_2008
__device__ __noinline__ void m_davis_etal_2008(float &z_0, float ustar);
// MPAS physics_mmm/sf_mynn.F90:1112-1141 taylor_yelland_2001
__device__ __noinline__ void m_taylor_yelland_2001(float &z_0, float ustar, float wsp10);
// MPAS physics_mmm/sf_mynn.F90:1144-1172 charnock_1955
__device__ __noinline__ void m_charnock_1955(float &z_0, float ustar, float wsp10, float visc, float zu);
// MPAS physics_mmm/sf_mynn.F90:1175-1208 edson_etal_2013
__device__ __noinline__ void m_edson_etal_2013(float &z_0, float ustar, float wsp10, float visc, float zu);
// MPAS physics_mmm/sf_mynn.F90:1211-1250 garratt_1992
__device__ __noinline__ void m_garratt_1992(float &zt, float &zq, float z_0, float ren, float landsea);
// MPAS physics_mmm/sf_mynn.F90:1253-1302 fairall_etal_2003
__device__ __noinline__ void m_fairall_etal_2003(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl);
// MPAS physics_mmm/sf_mynn.F90:1305-1336 fairall_etal_2014
__device__ __noinline__ void m_fairall_etal_2014(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl);
// MPAS physics_mmm/sf_mynn.F90:1339-1400 yang_2008
__device__ __noinline__ void m_yang_2008(float z_0, float &zt, float &zq, float ustar, float tstar, float qst, float ren, float visc, float landsea);
// MPAS physics_mmm/sf_mynn.F90:1403-1457 andreas_2002
__device__ __noinline__ void m_andreas_2002(float z_0, float bvisc, float ustar, float &zt, float &zq);
// MPAS physics_mmm/sf_mynn.F90:1460-1499 psi_hogstrom_1996
__device__ __noinline__ void m_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// MPAS physics_mmm/sf_mynn.F90:1502-1544 psi_dyerhicks
__device__ __noinline__ void m_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za);
// MPAS physics_mmm/sf_mynn.F90:1547-1581 psi_beljaars_holtslag_1991
__device__ __noinline__ void m_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1584-1618 psi_zilitinkevich_esau_2007
__device__ __noinline__ void m_psi_zilitinkevich_esau_2007(float &psi_m, float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1621-1656 psi_businger_1971
__device__ __noinline__ void m_psi_businger_1971(float &psi_m, float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1659-1695 psi_suselj_sood_2010
__device__ __noinline__ void m_psi_suselj_sood_2010(float &psi_m, float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1698-1718 psi_cb2005
__device__ __noinline__ void m_psi_cb2005(float &psim1, float &psih1, float zl, float z0l);
// MPAS physics_mmm/sf_mynn.F90:1721-1788 li_etal_2010
__device__ __noinline__ void m_li_etal_2010(float &zl, float rib, float zaz0, float z0zt);
// MPAS physics_mmm/sf_mynn.F90:1791-1844 zolri
__device__ __noinline__ float m_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:1847-1891 zolri2
__device__ __noinline__ float m_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:1894-1964 zolrib
__device__ __noinline__ float m_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:1967-2004 psi_init
__device__ __noinline__ void m_psi_init(int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:2009-2016 psim_stable_full
__device__ __noinline__ float m_psim_stable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2019-2026 psih_stable_full
__device__ __noinline__ float m_psih_stable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2029-2043 psim_unstable_full
__device__ __noinline__ float m_psim_unstable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2046-2060 psih_unstable_full
__device__ __noinline__ float m_psih_unstable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2066-2076 psim_stable_full_gfs
__device__ __noinline__ float m_psim_stable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2079-2089 psih_stable_full_gfs
__device__ __noinline__ float m_psih_stable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2092-2109 psim_unstable_full_gfs
__device__ __noinline__ float m_psim_unstable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2112-2129 psih_unstable_full_gfs
__device__ __noinline__ float m_psih_unstable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2135-2157 psim_stable
__device__ __noinline__ float m_psim_stable(float zolf, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:2160-2182 psih_stable
__device__ __noinline__ float m_psih_stable(float zolf, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:2185-2207 psim_unstable
__device__ __noinline__ float m_psim_unstable(float zolf, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:2210-2232 psih_unstable
__device__ __noinline__ float m_psih_unstable(float zolf, int psi_opt);
// MPAS physics_mmm/sf_mynn.F90:130-996 sf_mynn_run
__device__ __noinline__ void m_sf_mynn_run(float u1d, float v1d, float t1d, float qv1d, float p1d, float dz8w1d, float rho1d, float u1d2, float v1d2, float dz2w1d, float cp, float g, float rovcp, float r, float xlv, float psfcpa, float &chs, float &chs2, float &cqs2, float &cpm, float pblh, float &rmol, float &znt, float &ust, float mavail, float &zol, float &mol, float &regime, float &psim, float &psih, float xland, float &hfx, float &qfx, float tsk, float &u10, float &v10, float &th2, float &t2, float &q2, float &flhc, float &flqc, float snowh, float &qgh, float &qsfc, float &lh, float &gz1oz0, float &wspd, float &br, int isfflx, float dx, float svp1, float svp2, float svp3, float svpt0, float ep1, float ep2, float karman, float &ch, float qcg, int itimestep, float &wstar, float &qstar, float &ustm, float &ck, float &cka, float &cd, float &cda, bool spp_pbl, float rstoch1d, int isftcflx, int iz0tlnd, int its, int ite, bool has_isftcflx, bool has_iz0tlnd, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm) {
float prt = 0x1.0000000000000p+0f;
float xka = 0x1.92a7380000000p-16f;
int n;
int i;
int k;
int l;
int yesno;
float ep3;
float pl;
float thcon;
float tvcon;
float e1;
float dthvdz;
float dthvm;
float vconv;
float zol2;
float zol10;
float zolza;
float zolz0;
float zolzt;
float dtg;
float psix;
float dtthx;
float dthdz;
float psix10;
float psit;
float psit2;
float psiq;
float psiq2;
float psiq10;
float dzdt;
float fluxc;
float vsgd;
float restar;
float visc;
float dqg;
float oldust;
float oldtst;
float za;
float za2;
float thv1d;
float th1d;
float tc1d;
float tv1d;
float qvsh;
float psih2;
float psim10;
float psih10;
float wspdi;
float z_q;
float z_t;
float zntstoch;
float govrth;
float thgb;
float thvgb;
float psfc;
float qsfcmr;
float gz2oz0;
float gz10oz0;
float gz2ozt;
float gz10ozt;
float gz1ozt;
float zratio;
ep3 = __fsub_rn(0x1.0000000000000p+0f,ep2);
{ // independent MPAS column loop
psfc = __fdiv_rn(psfcpa,0x1.f400000000000p+9f);
thgb = __fmul_rn(tsk,SL_POW(__fdiv_rn(0x1.9000000000000p+6f,psfc),rovcp));
pl = __fdiv_rn(p1d,0x1.f400000000000p+9f);
thcon = SL_POW(__fdiv_rn(0x1.9000000000000p+6f,pl),rovcp);
th1d = __fmul_rn(t1d,thcon);
tc1d = __fsub_rn(t1d,0x1.1126660000000p+8f);
qvsh = __fdiv_rn(qv1d,__fadd_rn(0x1.0000000000000p+0f,qv1d));
tvcon = __fadd_rn(0x1.0000000000000p+0f,__fmul_rn(ep1,qvsh));
thv1d = __fmul_rn(th1d,tvcon);
tv1d = __fmul_rn(t1d,tvcon);
za = __fmul_rn(0x1.0000000000000p-1f,dz8w1d);
za2 = __fadd_rn(dz8w1d,__fmul_rn(0x1.0000000000000p-1f,dz2w1d));
govrth = __fdiv_rn(g,th1d);
}
{ // independent MPAS column loop
if ((tsk<0x1.1126660000000p+8f)) {
e1 = __fmul_rn(svp1,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(4648,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,tsk))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,tsk)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,tsk)))));
} else {
e1 = __fmul_rn(svp1,SL_EXP(__fdiv_rn(__fmul_rn(svp2,__fsub_rn(tsk,svpt0)),__fsub_rn(tsk,svp3))));
}
if (((xland>0x1.8000000000000p+0f)||(qsfc<=0x0.0p+0f))) {
qsfc = __fdiv_rn(__fmul_rn(ep2,e1),__fsub_rn(psfc,__fmul_rn(ep3,e1)));
qsfcmr = __fdiv_rn(__fmul_rn(ep2,e1),__fsub_rn(psfc,e1));
} else {
qsfcmr = __fdiv_rn(qsfc,__fsub_rn(0x1.0000000000000p+0f,qsfc));
}
if ((tsk<0x1.1126660000000p+8f)) {
e1 = __fmul_rn(svp1,SL_EXP(__fadd_rn(__fsub_rn(__fmul_rn(4648,__fsub_rn(0x1.dfda840000000p-9f,__fdiv_rn(0x1.0000000000000p+0f,t1d))),__fmul_rn(0x1.747ae20000000p+3f,SL_LOG(__fdiv_rn(0x1.1126660000000p+8f,t1d)))),__fmul_rn(0x1.7318fc0000000p-6f,__fsub_rn(0x1.1126660000000p+8f,t1d)))));
} else {
e1 = __fmul_rn(svp1,SL_EXP(__fdiv_rn(__fmul_rn(svp2,__fsub_rn(t1d,svpt0)),__fsub_rn(t1d,svp3))));
}
pl = __fdiv_rn(p1d,0x1.f400000000000p+9f);
qgh = __fdiv_rn(__fmul_rn(ep2,e1),__fsub_rn(pl,e1));
cpm = __fmul_rn(cp,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.ae147a0000000p-1f,qv1d)));
}
{ // independent MPAS column loop
wspd = SL_SQRT(__fadd_rn(__fmul_rn(u1d,u1d),__fmul_rn(v1d,v1d)));
thvgb = __fmul_rn(thgb,__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(ep1,qsfc)));
dthdz = __fsub_rn(th1d,thgb);
dthvdz = __fsub_rn(thv1d,thvgb);
fluxc = fmaxf(__fadd_rn(__fdiv_rn(__fdiv_rn(hfx,rho1d),cp),__fdiv_rn(__fmul_rn(__fmul_rn(ep1,thvgb),qfx),rho1d)),0x0.0p+0f);
if (((xland>0x1.8000000000000p+0f)||(qsfc<=0x0.0p+0f))) {
wstar = __fmul_rn(0x1.4000000000000p+0f,SL_POW(__fmul_rn(__fmul_rn(__fdiv_rn(g,tsk),pblh),fluxc),0x1.51eb860000000p-2f));
} else {
wstar = __fmul_rn(0x1.4000000000000p+0f,SL_POW(__fmul_rn(__fmul_rn(__fdiv_rn(g,tsk),fminf(__fmul_rn(0x1.8000000000000p+0f,pblh),0x1.f400000000000p+11f)),fluxc),0x1.51eb860000000p-2f));
}
vsgd = __fmul_rn(0x1.47ae140000000p-2f,SL_POW(fmaxf(__fsub_rn(__fdiv_rn(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.51eb860000000p-2f));
wspd = SL_SQRT(__fadd_rn(__fadd_rn(__fmul_rn(wspd,wspd),__fmul_rn(wstar,wstar)),__fmul_rn(vsgd,vsgd)));
wspd = fmaxf(wspd,0x1.99999a0000000p-4f);
br = __fdiv_rn(__fmul_rn(__fmul_rn(govrth,za),dthvdz),__fmul_rn(wspd,wspd));
if ((itimestep==1)) {
br = fmaxf(br,-0x1.0000000000000p+1f);
br = fminf(br,0x1.0000000000000p+1f);
} else {
br = fmaxf(br,-0x1.0000000000000p+2f);
br = fminf(br,0x1.0000000000000p+2f);
}
}
{ // independent MPAS column loop
visc = __fmul_rn(0x1.bcee880000000p-17f,__fsub_rn(__fadd_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.acbc8c0000000p-8f,tc1d)),__fmul_rn(__fmul_rn(0x1.16890c0000000p-17f,tc1d),tc1d)),__fmul_rn(__fmul_rn(__fmul_rn(0x1.4c9a2e0000000p-28f,tc1d),tc1d),tc1d)));
if ((__fsub_rn(xland,0x1.8000000000000p+0f)>=0)) {
if (has_isftcflx) {
if ((isftcflx==0)) {
if (true) {
m_charnock_1955(znt,ust,wspd,visc,za);
} else {
m_edson_etal_2013(znt,ust,wspd,visc,za);
}
} else if (((isftcflx==1)||(isftcflx==2))) {
m_davis_etal_2008(znt,ust);
} else if ((isftcflx==3)) {
m_taylor_yelland_2001(znt,ust,wspd);
} else if ((isftcflx==4)) {
if (true) {
m_charnock_1955(znt,ust,wspd,visc,za);
} else {
m_edson_etal_2013(znt,ust,wspd,visc,za);
}
}
} else {
if (true) {
m_charnock_1955(znt,ust,wspd,visc,za);
} else {
m_edson_etal_2013(znt,ust,wspd,visc,za);
}
}
if (spp_pbl) {
zntstoch = fmaxf(__fadd_rn(znt,__fmul_rn(__fmul_rn(znt,0x1.0000000000000p+0f),rstoch1d)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = fmaxf(__fdiv_rn(__fmul_rn(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if (has_isftcflx) {
if ((isftcflx==0)) {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
} else if ((isftcflx==1)) {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
} else if ((isftcflx==2)) {
m_garratt_1992(z_t,z_q,zntstoch,restar,xland);
} else if ((isftcflx==3)) {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
}
} else {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
}
} else {
if (spp_pbl) {
zntstoch = fmaxf(__fadd_rn(znt,__fmul_rn(__fmul_rn(znt,0x1.0000000000000p+0f),rstoch1d)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = fmaxf(__fdiv_rn(__fmul_rn(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if ((snowh>=0x1.99999a0000000p-4f)) {
m_andreas_2002(zntstoch,visc,ust,z_t,z_q);
} else {
if (has_iz0tlnd) {
if ((iz0tlnd<=1)) {
m_zilitinkevich_1995(zntstoch,z_t,z_q,restar,ust,karman,xland,iz0tlnd,spp_pbl,rstoch1d,true);
} else if ((iz0tlnd==2)) {
m_yang_2008(zntstoch,z_t,z_q,ust,mol,qstar,restar,visc,xland);
} else if ((iz0tlnd==3)) {
m_garratt_1992(z_t,z_q,zntstoch,restar,xland);
}
} else {
m_zilitinkevich_1995(zntstoch,z_t,z_q,restar,ust,karman,xland,0,spp_pbl,rstoch1d,true);
}
}
}
zratio = __fdiv_rn(zntstoch,z_t);
gz1oz0 = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),zntstoch));
gz1ozt = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),z_t));
gz2oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),z_t));
gz10oz0 = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),z_t));
if ((br>0x0.0p+0f)) {
if ((br>0x1.99999a0000000p-3f)) {
regime = 0x1.0000000000000p+0f;
} else {
regime = 0x1.0000000000000p+1f;
}
if ((itimestep<=1)) {
m_li_etal_2010(zol,br,__fdiv_rn(za,zntstoch),zratio);
} else {
zol = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(za,karman),g),mol),__fmul_rn(th1d,fmaxf(__fmul_rn(ust,ust),0x1.a36e2e0000000p-14f)));
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
}
zol = m_zolrib(br,za,zntstoch,z_t,gz1oz0,gz1ozt,zol,0);
zol = fmaxf(zol,0x0.0p+0f);
zol = fminf(zol,0x1.4000000000000p+4f);
zolzt = __fdiv_rn(__fmul_rn(zol,z_t),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
if ((__fsub_rn(xland,0x1.8000000000000p+0f)>=0)) {
psim = __fsub_rn(m_psim_stable(zolza,0),m_psim_stable(zolz0,0));
psih = __fsub_rn(m_psih_stable(zolza,0),m_psih_stable(zolzt,0));
psim10 = __fsub_rn(m_psim_stable(zol10,0),m_psim_stable(zolz0,0));
psih10 = __fsub_rn(m_psih_stable(zol10,0),m_psih_stable(zolz0,0));
psih2 = __fsub_rn(m_psih_stable(zol2,0),m_psih_stable(zolz0,0));
} else {
psim = __fsub_rn(m_psim_stable(zolza,0),m_psim_stable(zolz0,0));
psih = __fsub_rn(m_psih_stable(zolza,0),m_psih_stable(zolzt,0));
psim10 = __fsub_rn(m_psim_stable(zol10,0),m_psim_stable(zolz0,0));
psih10 = __fsub_rn(m_psih_stable(zol10,0),m_psih_stable(zolz0,0));
psih2 = __fsub_rn(m_psih_stable(zol2,0),m_psih_stable(zolz0,0));
}
rmol = __fdiv_rn(zol,za);
} else if ((br==0x0.0p+0f)) {
regime = 0x1.8000000000000p+1f;
psim = 0x0.0p+0f;
psih = psim;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = __fdiv_rn(zol,za);
} else if ((br<0x0.0p+0f)) {
regime = 0x1.0000000000000p+2f;
if ((itimestep<=1)) {
m_li_etal_2010(zol,br,__fdiv_rn(za,zntstoch),zratio);
} else {
zol = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(za,karman),g),mol),__fmul_rn(th1d,fmaxf(__fmul_rn(ust,ust),0x1.0624de0000000p-10f)));
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
}
zol = m_zolrib(br,za,zntstoch,z_t,gz1oz0,gz1ozt,zol,0);
zol = fmaxf(zol,-0x1.4000000000000p+4f);
zol = fminf(zol,0x0.0p+0f);
zolzt = __fdiv_rn(__fmul_rn(zol,z_t),za);
zolz0 = __fdiv_rn(__fmul_rn(zol,zntstoch),za);
zolza = __fdiv_rn(__fmul_rn(zol,__fadd_rn(za,zntstoch)),za);
zol10 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = __fdiv_rn(__fmul_rn(zol,__fadd_rn(0x1.0000000000000p+1f,zntstoch)),za);
if ((__fsub_rn(xland,0x1.8000000000000p+0f)>=0)) {
psim = __fsub_rn(m_psim_unstable(zolza,0),m_psim_unstable(zolz0,0));
psih = __fsub_rn(m_psih_unstable(zolza,0),m_psih_unstable(zolzt,0));
psim10 = __fsub_rn(m_psim_unstable(zol10,0),m_psim_unstable(zolz0,0));
psih10 = __fsub_rn(m_psih_unstable(zol10,0),m_psih_unstable(zolz0,0));
psih2 = __fsub_rn(m_psih_unstable(zol2,0),m_psih_unstable(zolz0,0));
} else {
psim = __fsub_rn(m_psim_unstable(zolza,0),m_psim_unstable(zolz0,0));
psih = __fsub_rn(m_psih_unstable(zolza,0),m_psih_unstable(zolzt,0));
psim10 = __fsub_rn(m_psim_unstable(zol10,0),m_psim_unstable(zolz0,0));
psih10 = __fsub_rn(m_psih_unstable(zol10,0),m_psih_unstable(zolz0,0));
psih2 = __fsub_rn(m_psih_unstable(zol2,0),m_psih_unstable(zolz0,0));
}
psih = fminf(psih,__fmul_rn(0x1.cccccc0000000p-1f,gz1ozt));
psim = fminf(psim,__fmul_rn(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = fminf(psih2,__fmul_rn(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = fminf(psim10,__fmul_rn(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = fminf(psih10,__fmul_rn(0x1.cccccc0000000p-1f,gz10ozt));
rmol = __fdiv_rn(zol,za);
}
psix = __fsub_rn(gz1oz0,psim);
psix10 = __fsub_rn(gz10oz0,psim10);
oldust = ust;
ust = __fadd_rn(__fmul_rn(0x1.0000000000000p-1f,ust),__fdiv_rn(__fmul_rn(__fmul_rn(0x1.0000000000000p-1f,karman),wspd),psix));
wspdi = fmaxf(SL_SQRT(__fadd_rn(__fmul_rn(u1d,u1d),__fmul_rn(v1d,v1d))),0x1.99999a0000000p-4f);
if (has_ustm) {
ustm = __fadd_rn(__fmul_rn(0x1.0000000000000p-1f,ustm),__fdiv_rn(__fmul_rn(__fmul_rn(0x1.0000000000000p-1f,karman),wspdi),psix));
}
if ((__fsub_rn(xland,0x1.8000000000000p+0f)<0x0.0p+0f)) {
ust = fmaxf(ust,0x1.47ae140000000p-8f);
if (has_ustm) ustm = ust;
}
gz1ozt = SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),z_t));
gz2ozt = SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),z_t));
psit = fmaxf(__fsub_rn(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmaxf(__fsub_rn(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,zntstoch),z_q)),psih),0x1.0000000000000p+0f);
psiq2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,zntstoch),z_q)),psih2),0x1.0000000000000p+0f);
psiq10 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,zntstoch),z_q)),psih10),0x1.0000000000000p+0f);
dtg = __fsub_rn(thv1d,thvgb);
oldtst = mol;
mol = __fdiv_rn(__fdiv_rn(__fmul_rn(karman,dtg),psit),0x1.0000000000000p+0f);
dqg = __fmul_rn(__fsub_rn(qvsh,qsfc),0x1.f400000000000p+9f);
qstar = __fdiv_rn(__fdiv_rn(__fmul_rn(karman,dqg),psiq),0x1.0000000000000p+0f);
}
{ // independent MPAS column loop
psix = __fsub_rn(gz1oz0,psim);
psix10 = __fsub_rn(gz10oz0,psim10);
psit = fmaxf(__fsub_rn(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmaxf(__fsub_rn(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z_q),z_q)),psih),0x1.0000000000000p+0f);
psiq2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+1f,z_q),z_q)),psih2),0x1.0000000000000p+0f);
psiq10 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(0x1.4000000000000p+3f,z_q),z_q)),psih10),0x1.0000000000000p+0f);
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho1d,mavail),ust),karman),psiq);
flhc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rho1d,cpm),ust),karman),psit);
qfx = __fmul_rn(flqc,__fsub_rn(qsfcmr,qv1d));
qfx = fmaxf(qfx,-0x1.47ae140000000p-6f);
lh = __fmul_rn(xlv,qfx);
if ((__fsub_rn(xland,0x1.8000000000000p+0f)>0x0.0p+0f)) {
hfx = __fmul_rn(flhc,__fsub_rn(thgb,th1d));
if (has_isftcflx) {
if ((isftcflx!=0)) {
hfx = __fadd_rn(hfx,__fmul_rn(__fmul_rn(__fmul_rn(rho1d,ustm),ustm),wspdi));
}
}
} else if ((__fsub_rn(xland,0x1.8000000000000p+0f)<0x0.0p+0f)) {
hfx = __fmul_rn(flhc,__fsub_rn(thgb,th1d));
hfx = fmaxf(hfx,-0x1.f400000000000p+7f);
}
chs = __fdiv_rn(__fmul_rn(ust,karman),psit);
ch = __fdiv_rn(flhc,__fmul_rn(cpm,rho1d));
cqs2 = __fdiv_rn(__fmul_rn(ust,karman),psiq2);
chs2 = __fdiv_rn(__fmul_rn(ust,karman),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = __fmul_rn(__fdiv_rn(karman,psix10),__fdiv_rn(karman,psiq10));
cd = __fmul_rn(__fdiv_rn(karman,psix10),__fdiv_rn(karman,psix10));
cka = __fmul_rn(__fdiv_rn(karman,psix),__fdiv_rn(karman,psiq));
cda = __fmul_rn(__fdiv_rn(karman,psix),__fdiv_rn(karman,psix));
}
}
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u1d2;
v10 = v1d2;
} else {
u10 = __fdiv_rn(__fmul_rn(u1d,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v1d,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = __fdiv_rn(__fmul_rn(u1d,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
v10 = __fdiv_rn(__fmul_rn(v1d,SL_LOG(__fdiv_rn(0x1.4000000000000p+3f,zntstoch))),SL_LOG(__fdiv_rn(za,zntstoch)));
} else {
u10 = __fdiv_rn(__fmul_rn(u1d,psix10),psix);
v10 = __fdiv_rn(__fmul_rn(v1d,psix10),psix);
}
dtg = __fsub_rn(th1d,thgb);
th2 = __fadd_rn(thgb,__fdiv_rn(__fmul_rn(dtg,psit2),psit));
if ((((th1d>thgb)&&((th2<thgb)||(th2>th1d)))||((th1d<thgb)&&((th2>thgb)||(th2<th1d))))) {
th2 = __fadd_rn(thgb,__fdiv_rn(__fmul_rn(0x1.0000000000000p+1f,__fsub_rn(th1d,thgb)),za));
}
t2 = __fmul_rn(th2,SL_POW(__fdiv_rn(psfc,0x1.9000000000000p+6f),rovcp));
q2 = __fadd_rn(qsfcmr,__fdiv_rn(__fmul_rn(__fsub_rn(qv1d,qsfcmr),psiq2),psiq));
q2 = fmaxf(q2,fminf(qsfcmr,qv1d));
q2 = fminf(q2,__fmul_rn(0x1.0ccccc0000000p+0f,qv1d));
if (false) {
yesno = 0;
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho1d<0x0.0p+0f)||(rho1d>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((__fmul_rn(qsfc,0x1.f400000000000p+9f)<0x0.0p+0f)||(__fmul_rn(qsfc,0x1.f400000000000p+9f)>0x1.4000000000000p+5f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// MPAS physics_mmm/sf_mynn.F90:999-1073 zilitinkevich_1995
__device__ __noinline__ void m_zilitinkevich_1995(float z_0, float &zt, float &zq, float restar, float ustar, float karman, float landsea, int iz0tlnd2, bool spp_pbl, float rstoch, bool has_iz0tlnd2) {
float czil;
if ((__fsub_rn(landsea,0x1.8000000000000p+0f)>0)) {
if ((restar<0x1.99999a0000000p-4f)) {
zt = __fmul_rn(z_0,SL_EXP(__fmul_rn(karman,0x1.0000000000000p+1f)));
zt = fminf(zt,0x1.f751040000000p-15f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
zq = __fmul_rn(z_0,SL_EXP(__fmul_rn(karman,0x1.8000000000000p+1f)));
zq = fminf(zq,0x1.f751040000000p-15f);
zq = fmaxf(zq,0x1.12e0be0000000p-29f);
} else {
zt = __fmul_rn(z_0,SL_EXP(__fmul_rn((-karman),__fsub_rn(__fmul_rn(0x1.0000000000000p+2f,SL_SQRT(restar)),0x1.99999a0000000p+1f))));
zt = fminf(zt,0x1.f751040000000p-15f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
zq = __fmul_rn(z_0,SL_EXP(__fmul_rn((-karman),__fsub_rn(__fmul_rn(0x1.0000000000000p+2f,SL_SQRT(restar)),0x1.0ccccc0000000p+2f))));
zq = fminf(zt,0x1.f751040000000p-15f);
zq = fmaxf(zt,0x1.12e0be0000000p-29f);
}
} else {
if ((iz0tlnd2==1)) {
czil = SL_POW(0x1.4000000000000p+3f,__fmul_rn(-0x1.99999a0000000p-2f,__fdiv_rn(z_0,0x1.1eb8520000000p-4f)));
} else {
czil = 0x1.5c28f60000000p-4f;
}
zt = __fmul_rn(z_0,SL_EXP(__fmul_rn(__fmul_rn((-karman),czil),SL_SQRT(restar))));
zt = fminf(zt,__fmul_rn(0x1.8000000000000p-1f,z_0));
zq = __fmul_rn(z_0,SL_EXP(__fmul_rn(__fmul_rn((-karman),czil),SL_SQRT(restar))));
zq = fminf(zq,__fmul_rn(0x1.8000000000000p-1f,z_0));
if (spp_pbl) {
zt = __fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch));
zt = fmaxf(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
}
}
// MPAS physics_mmm/sf_mynn.F90:1076-1109 davis_etal_2008
__device__ __noinline__ void m_davis_etal_2008(float &z_0, float ustar) {
float zw;
float zn1;
float zn2;
float g = 0x1.39eb860000000p+3f;
float ozo = 0x1.0ac1fa0000000p-16f;
zw = fminf(SL_POW(__fdiv_rn(ustar,0x1.0f5c280000000p+0f),0x1.3333340000000p-2f),0x1.0000000000000p+0f);
zn1 = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(0x1.6872b00000000p-7f,ustar),ustar),0x1.39eb860000000p+3f),0x1.0ac1fa0000000p-16f);
zn2 = __fadd_rn(__fmul_rn(0x1.4000000000000p+3f,SL_EXP(__fmul_rn(-0x1.3000000000000p+3f,SL_POW(ustar,-0x1.554c980000000p-2f)))),__fdiv_rn(0x1.baeb220000000p-20f,fmaxf(ustar,0x1.47ae140000000p-7f)));
z_0 = __fadd_rn(__fmul_rn(__fsub_rn(0x1.0000000000000p+0f,zw),zn1),__fmul_rn(zw,zn2));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1112-1141 taylor_yelland_2001
__device__ __noinline__ void m_taylor_yelland_2001(float &z_0, float ustar, float wsp10) {
float g = 0x1.39eb860000000p+3f;
float pi = 0x1.921fb60000000p+1f;
float hs;
float tp;
float lp;
hs = __fmul_rn(0x1.9652be0000000p-6f,__fmul_rn(wsp10,wsp10));
tp = __fmul_rn(0x1.753f7c0000000p-1f,fmaxf(wsp10,0x1.99999a0000000p-4f));
lp = __fdiv_rn(__fmul_rn(0x1.39eb860000000p+3f,__fmul_rn(tp,tp)),0x1.921fb60000000p+2f);
z_0 = __fmul_rn(__fmul_rn(0x1.2c00000000000p+10f,hs),SL_POW(__fdiv_rn(hs,lp),0x1.2000000000000p+2f));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1144-1172 charnock_1955
__device__ __noinline__ void m_charnock_1955(float &z_0, float ustar, float wsp10, float visc, float zu) {
float g = 0x1.39eb860000000p+3f;
float czo2 = 0x1.6872b00000000p-7f;
float czc;
float wsp10m;
wsp10m = __fdiv_rn(__fmul_rn(wsp10,0x1.7069e20000000p+3f),SL_LOG(__fdiv_rn(zu,0x1.a36e2e0000000p-14f)));
czc = __fadd_rn(0x1.6872b00000000p-7f,__fmul_rn(0x1.cac0840000000p-8f,fminf(fmaxf(__fdiv_rn(__fsub_rn(wsp10m,0x1.4000000000000p+3f),0x1.0000000000000p+3f),0x0.0p+0f),0x1.0000000000000p+0f)));
z_0 = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(czc,ustar),ustar),0x1.39eb860000000p+3f),__fdiv_rn(__fmul_rn(0x1.c28f5c0000000p-4f,visc),fmaxf(ustar,0x1.99999a0000000p-5f)));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1175-1208 edson_etal_2013
__device__ __noinline__ void m_edson_etal_2013(float &z_0, float ustar, float wsp10, float visc, float zu) {
float g = 0x1.39eb860000000p+3f;
float m = 0x1.bda5120000000p-10f;
float b = -0x1.47ae140000000p-8f;
float czc;
float wsp10m;
wsp10m = __fdiv_rn(__fmul_rn(wsp10,0x1.7069e20000000p+3f),SL_LOG(__fdiv_rn(zu,0x1.a36e2e0000000p-14f)));
wsp10m = fminf(0x1.3000000000000p+4f,wsp10m);
czc = __fadd_rn(__fmul_rn(0x1.bda5120000000p-10f,wsp10m),-0x1.47ae140000000p-8f);
czc = fmaxf(czc,0x0.0p+0f);
z_0 = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(czc,ustar),ustar),0x1.39eb860000000p+3f),__fdiv_rn(__fmul_rn(0x1.c28f5c0000000p-4f,visc),fmaxf(ustar,0x1.1eb8520000000p-4f)));
z_0 = fmaxf(z_0,0x1.10bafe0000000p-23f);
z_0 = fminf(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1211-1250 garratt_1992
__device__ __noinline__ void m_garratt_1992(float &zt, float &zq, float z_0, float ren, float landsea) {
float rq;
float e = 0x1.5bf0a80000000p+1f;
if ((__fsub_rn(landsea,0x1.8000000000000p+0f)>0)) {
zt = __fmul_rn(z_0,SL_EXP(__fsub_rn(0x1.0000000000000p+1f,__fmul_rn(0x1.3d70a40000000p+1f,SL_POW(ren,0x1.0000000000000p-2f)))));
zq = __fmul_rn(z_0,SL_EXP(__fsub_rn(0x1.0000000000000p+1f,__fmul_rn(0x1.23d70a0000000p+1f,SL_POW(ren,0x1.0000000000000p-2f)))));
zq = fminf(zq,0x1.cd5f9a0000000p-15f);
zq = fmaxf(zq,0x1.12e0be0000000p-29f);
zt = fminf(zt,0x1.cd5f9a0000000p-15f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
} else {
zq = __fdiv_rn(z_0,0x1.d8e64a0000000p+2f);
zt = zq;
}
}
// MPAS physics_mmm/sf_mynn.F90:1253-1302 fairall_etal_2003
__device__ __noinline__ void m_fairall_etal_2003(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl) {
if ((ren<=0x1.0000000000000p+1f)) {
zt = __fmul_rn(0x1.cd5f9a0000000p-15f,SL_POW(ren,-0x1.3333340000000p-1f));
zq = zt;
} else {
zt = __fmul_rn(0x1.cd5f9a0000000p-15f,SL_POW(ren,-0x1.3333340000000p-1f));
zq = zt;
}
if (spp_pbl) {
zt = __fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch));
zq = zt;
}
zt = fminf(zt,0x1.a36e2e0000000p-14f);
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
zq = fminf(zt,0x1.a36e2e0000000p-14f);
zq = fmaxf(zt,0x1.12e0be0000000p-29f);
}
// MPAS physics_mmm/sf_mynn.F90:1305-1336 fairall_etal_2014
__device__ __noinline__ void m_fairall_etal_2014(float &zt, float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl) {
zt = fminf(0x1.4f8b580000000p-13f,__fdiv_rn(0x1.e68a0e0000000p-15f,SL_POW(ren,0x1.70a3d80000000p-1f)));
zq = zt;
if (spp_pbl) {
zt = fmaxf(__fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch)),0x1.12e0be0000000p-29f);
zq = fmaxf(__fadd_rn(zt,__fmul_rn(__fmul_rn(zt,0x1.0000000000000p-1f),rstoch)),0x1.12e0be0000000p-29f);
} else {
zt = fmaxf(zt,0x1.12e0be0000000p-29f);
zq = fmaxf(zt,0x1.12e0be0000000p-29f);
}
}
// MPAS physics_mmm/sf_mynn.F90:1339-1400 yang_2008
__device__ __noinline__ void m_yang_2008(float z_0, float &zt, float &zq, float ustar, float tstar, float qst, float ren, float visc, float landsea) {
float ht;
float tstar2;
float qstar2;
float z_02;
float renc2;
float renc = 0x1.2c00000000000p+8f;
float beta = 0x1.8000000000000p+0f;
float m = 0x1.5400000000000p+7f;
float b = 0x1.5980000000000p+9f;
z_02 = fminf(z_0,0x1.0000000000000p-1f);
z_02 = fmaxf(z_02,0x1.47ae140000000p-5f);
renc2 = __fadd_rn(0x1.5980000000000p+9f,__fmul_rn(0x1.5400000000000p+7f,SL_LOG(z_02)));
ht = __fdiv_rn(__fmul_rn(renc2,visc),fmaxf(ustar,0x1.47ae140000000p-7f));
tstar2 = fminf(tstar,0x0.0p+0f);
qstar2 = fminf(qst,0x0.0p+0f);
zt = __fmul_rn(ht,SL_EXP(__fmul_rn(__fmul_rn(-0x1.8000000000000p+0f,SL_POW(ustar,0x1.0000000000000p-1f)),fabsf(tstar2))));
zq = __fmul_rn(ht,SL_EXP(__fmul_rn(__fmul_rn(-0x1.8000000000000p+0f,SL_POW(ustar,0x1.0000000000000p-1f)),fabsf(qstar2))));
zt = fminf(zt,__fdiv_rn(z_0,0x1.0000000000000p+1f));
zq = fminf(zq,__fdiv_rn(z_0,0x1.0000000000000p+1f));
}
// MPAS physics_mmm/sf_mynn.F90:1403-1457 andreas_2002
__device__ __noinline__ void m_andreas_2002(float z_0, float bvisc, float ustar, float &zt, float &zq) {
float ren2;
float zntsno;
float bt0_s = 0x1.4000000000000p+0f;
float bt0_t = 0x1.3126ea0000000p-3f;
float bt0_r = 0x1.449ba60000000p-2f;
float bt1_s = 0x0.0p+0f;
float bt1_t = -0x1.19999a0000000p-1f;
float bt1_r = -0x1.2147ae0000000p-1f;
float bt2_s = 0x0.0p+0f;
float bt2_t = 0x0.0p+0f;
float bt2_r = -0x1.76c8b40000000p-3f;
float bq0_s = 0x1.9c28f60000000p+0f;
float bq0_t = 0x1.676c8c0000000p-2f;
float bq0_r = 0x1.9581060000000p-2f;
float bq1_s = 0x0.0p+0f;
float bq1_t = -0x1.4189380000000p-1f;
float bq1_r = -0x1.0624de0000000p-1f;
float bq2_s = 0x0.0p+0f;
float bq2_t = 0x0.0p+0f;
float bq2_r = -0x1.70a3d80000000p-3f;
zntsno = __fadd_rn(__fdiv_rn(__fmul_rn(0x1.147ae20000000p-3f,bvisc),ustar),__fmul_rn(__fdiv_rn(__fmul_rn(0x1.1eb8520000000p-5f,__fmul_rn(ustar,ustar)),0x1.39999a0000000p+3f),__fadd_rn(__fmul_rn(0x1.4000000000000p+2f,SL_EXP(__fmul_rn(-0x1.0000000000000p+0f,__fmul_rn(__fdiv_rn(__fsub_rn(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f),__fdiv_rn(__fsub_rn(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f))))),0x1.0000000000000p+0f)));
ren2 = __fdiv_rn(__fmul_rn(ustar,zntsno),bvisc);
if ((ren2>0x1.f400000000000p+9f)) ren2 = 0x1.f400000000000p+9f;
if ((ren2<=0x1.147ae20000000p-3f)) {
zt = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.4000000000000p+0f,__fmul_rn(0x0.0p+0f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
zq = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.9c28f60000000p+0f,__fmul_rn(0x0.0p+0f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
} else if (((ren2>0x1.147ae20000000p-3f)&&(ren2<0x1.4000000000000p+1f))) {
zt = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.3126ea0000000p-3f,__fmul_rn(-0x1.19999a0000000p-1f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
zq = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.676c8c0000000p-2f,__fmul_rn(-0x1.4189380000000p-1f,SL_LOG(ren2))),__fmul_rn(0x0.0p+0f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
} else {
zt = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.449ba60000000p-2f,__fmul_rn(-0x1.2147ae0000000p-1f,SL_LOG(ren2))),__fmul_rn(-0x1.76c8b40000000p-3f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
zq = __fmul_rn(zntsno,SL_EXP(__fadd_rn(__fadd_rn(0x1.9581060000000p-2f,__fmul_rn(-0x1.0624de0000000p-1f,SL_LOG(ren2))),__fmul_rn(-0x1.70a3d80000000p-3f,__fmul_rn(SL_LOG(ren2),SL_LOG(ren2))))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1460-1499 psi_hogstrom_1996
__device__ __noinline__ void m_psi_hogstrom_1996(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.5333340000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.0000000000000p+3f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1502-1544 psi_dyerhicks
__device__ __noinline__ void m_psi_dyerhicks(float &psi_m, float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = __fdiv_rn(__fmul_rn(z_0,zl),za);
zhl = __fdiv_rn(__fmul_rn(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zml));
psi_h = __fmul_rn(-0x1.4000000000000p+2f,__fsub_rn(zl,zhl));
} else {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),__fadd_rn(0x1.0000000000000p+0f,x0)))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x0,x0))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x0)));
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),__fadd_rn(0x1.0000000000000p+0f,y0))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1547-1581 psi_beljaars_holtslag_1991
__device__ __noinline__ void m_psi_beljaars_holtslag_1991(float &psi_m, float &psi_h, float zl) {
float a = 0x1.0000000000000p+0f;
float b = 0x1.54fdf40000000p-1f;
float c = 0x1.4000000000000p+2f;
float d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-__fadd_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+0f,zl),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-__fsub_rn(__fadd_rn(__fadd_rn(SL_POW(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),__fmul_rn(__fmul_rn(0x1.54fdf40000000p-1f,__fsub_rn(zl,0x1.c924920000000p+3f)),SL_EXP(__fmul_rn(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// MPAS physics_mmm/sf_mynn.F90:1584-1618 psi_zilitinkevich_esau_2007
__device__ __noinline__ void m_psi_zilitinkevich_esau_2007(float &psi_m, float &psi_h, float zl) {
float cm = 0x1.8000000000000p+1f;
float ct = 0x1.4000000000000p+1f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = __fmul_rn(-0x1.8000000000000p+1f,SL_POW(zl,0x1.aaaaaa0000000p-1f));
psi_h = __fmul_rn(-0x1.4000000000000p+1f,SL_POW(zl,0x1.99999a0000000p-1f));
}
}
// MPAS physics_mmm/sf_mynn.F90:1621-1656 psi_businger_1971
__device__ __noinline__ void m_psi_businger_1971(float &psi_m, float &psi_h, float zl) {
float x;
float y;
float pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = __fadd_rn(__fsub_rn(__fadd_rn(SL_LOG(__fmul_rn(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(x,x)),0x1.0000000000000p+1f))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
psi_h = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = __fmul_rn(-0x1.2ccccc0000000p+2f,zl);
psi_h = __fmul_rn(-0x1.967c880000000p+2f,zl);
}
}
// MPAS physics_mmm/sf_mynn.F90:1659-1695 psi_suselj_sood_2010
__device__ __noinline__ void m_psi_suselj_sood_2010(float &psi_m, float &psi_h, float zl) {
float rfc = 0x1.851eb80000000p-3f;
float ric = 0x1.76c8b40000000p-3f;
float phit = 0x1.99999a0000000p-1f;
if ((zl>0x0.0p+0f)) {
psi_m = (-__fadd_rn(__fdiv_rn(zl,0x1.851eb80000000p-3f),__fmul_rn(0x1.1f4f0e0000000p+0f,SL_EXP(__fsub_rn(0x1.0000000000000p+0f,__fdiv_rn(0x1.aaa64c0000000p+0f,zl))))));
psi_h = (-__fadd_rn(__fdiv_rn(__fmul_rn(zl,0x1.76c8b40000000p-3f),0x1.71a9fc0000000p-3f),__fmul_rn(0x1.c5c2900000000p+2f,SL_POW(zl,0x1.1bedfa0000000p+0f))));
} else {
psi_m = __fmul_rn(0x1.fb15b60000000p-1f,SL_LOG(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.c872b00000000p+3f,zl))));
psi_h = __fmul_rn(0x1.02a3060000000p+0f,SL_LOG(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.04e7d60000000p+4f,zl))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1698-1718 psi_cb2005
__device__ __noinline__ void m_psi_cb2005(float &psim1, float &psih1, float zl, float z0l) {
psim1 = __fsub_rn(__fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),__fmul_rn(0x1.8666660000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = __fsub_rn(__fmul_rn(-0x1.6000000000000p+2f,SL_LOG(__fadd_rn(zl,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),__fmul_rn(0x1.6000000000000p+2f,SL_LOG(__fadd_rn(z0l,SL_POW(__fadd_rn(0x1.0000000000000p+0f,SL_POW(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// MPAS physics_mmm/sf_mynn.F90:1721-1788 li_etal_2010
__device__ __noinline__ void m_li_etal_2010(float &zl, float rib, float zaz0, float z0zt) {
float alfa;
float beta;
float zaz02;
float z0zt2;
float au11 = 0x1.70a3d80000000p-5f;
float bu11 = 0x1.89374c0000000p-9f;
float bu12 = 0x1.82a9940000000p-8f;
float bu21 = -0x1.5326180000000p-4f;
float bu22 = 0x1.c4dd300000000p-1f;
float bu31 = 0x1.6425ae0000000p-3f;
float bu32 = -0x1.d7b4a20000000p-1f;
float bu33 = -0x1.b0f27c0000000p-4f;
float aw11 = 0x1.25c91e0000000p-1f;
float aw12 = -0x1.c275260000000p-2f;
float aw21 = -0x1.39a9fc0000000p+2f;
float aw22 = 0x1.a400000000000p+5f;
float bw11 = -0x1.b98c7e0000000p-5f;
float bw12 = 0x1.8a3d700000000p+0f;
float bw21 = -0x1.56872c0000000p-1f;
float bw22 = -0x1.a418940000000p+1f;
float as11 = 0x1.817c1c0000000p-1f;
float as21 = 0x1.de147a0000000p+3f;
float bs11 = 0x1.4154ca0000000p-3f;
float bs21 = -0x1.3c84b60000000p-2f;
float bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = SL_LOG(zaz02);
beta = SL_LOG(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = __fadd_rn(__fmul_rn(__fmul_rn(0x1.70a3d80000000p-5f,alfa),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),__fmul_rn(alfa,alfa)),__fmul_rn(__fadd_rn(__fmul_rn(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),__fadd_rn(__fadd_rn(__fmul_rn(0x1.6425ae0000000p-3f,__fmul_rn(beta,beta)),__fmul_rn(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = fmaxf(zl,-0x1.e000000000000p+3f);
zl = fminf(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),__fadd_rn(__fmul_rn(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),__fmul_rn(rib,rib)),__fmul_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),__fadd_rn(__fmul_rn(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = fminf(zl,0x1.0000000000000p+2f);
zl = fmaxf(zl,0x0.0p+0f);
} else {
zl = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),__fmul_rn(0x1.4154ca0000000p-3f,alfa)),__fmul_rn(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = fminf(zl,0x1.4000000000000p+4f);
zl = fmaxf(zl,0x1.0000000000000p+0f);
}
}
// MPAS physics_mmm/sf_mynn.F90:1791-1844 zolri
__device__ __noinline__ float m_zolri(float ri, float za, float z0, float zt, float zol1, int psi_opt) {
float result;
int n;
int nmax = 20;
float x1;
float x2;
float fx1;
float fx2;
if ((ri<0x0.0p+0f)) {
x1 = __fsub_rn(zol1,0x1.47ae140000000p-6f);
x2 = 0x0.0p+0f;
} else {
x1 = 0x0.0p+0f;
x2 = __fadd_rn(zol1,0x1.47ae140000000p-6f);
}
n = 0;
fx1 = m_zolri2(x1,ri,za,z0,zt,psi_opt);
fx2 = m_zolri2(x2,ri,za,z0,zt,psi_opt);
while (((fabsf(__fsub_rn(x1,x2))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((fabsf(fx2)<fabsf(fx1))) {
x1 = __fsub_rn(x1,__fmul_rn(__fdiv_rn(fx1,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx1 = m_zolri2(x1,ri,za,z0,zt,psi_opt);
result = x1;
} else {
x2 = __fsub_rn(x2,__fmul_rn(__fdiv_rn(fx2,__fsub_rn(fx2,fx1)),__fsub_rn(x2,x1)));
fx2 = m_zolri2(x2,ri,za,z0,zt,psi_opt);
result = x2;
}
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(x1,x2))>=0x1.47ae140000000p-7f))) {
m_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:1847-1891 zolri2
__device__ __noinline__ float m_zolri2(float &zol2, float ri2, float za, float z0, float zt, int psi_opt) {
float result;
float zol20;
float zol3;
float psim1;
float psih1;
float psix2;
float psit2;
float zolt;
if ((__fmul_rn(zol2,ri2)<0x0.0p+0f)) {
zol2 = 0x0.0p+0f;
}
zol20 = __fdiv_rn(__fmul_rn(zol2,z0),za);
zol3 = __fadd_rn(zol2,zol20);
zolt = __fdiv_rn(__fmul_rn(zol2,zt),za);
if ((ri2<0)) {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_psih_unstable(zol3,psi_opt),m_psih_unstable(zolt,psi_opt))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_psim_unstable(zol3,psi_opt),m_psim_unstable(zol20,psi_opt))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),zt)),__fsub_rn(m_psih_stable(zol3,psi_opt),m_psih_stable(zolt,psi_opt))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(SL_LOG(__fdiv_rn(__fadd_rn(za,z0),z0)),__fsub_rn(m_psim_stable(zol3,psi_opt),m_psim_stable(zol20,psi_opt))),0x1.0000000000000p+0f);
}
result = __fsub_rn(__fdiv_rn(__fmul_rn(zol2,psit2),__fmul_rn(psix2,psix2)),ri2);
return result;
}
// MPAS physics_mmm/sf_mynn.F90:1894-1964 zolrib
__device__ __noinline__ float m_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, float &zol1, int psi_opt) {
float result;
int n;
int nmax = 20;
float zol20;
float zol3;
float zolt;
float zolold;
float psit2;
float psix2;
if ((__fmul_rn(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = __fdiv_rn(__fmul_rn(zolold,z0),za);
zol3 = __fadd_rn(zolold,zol20);
zolt = __fdiv_rn(__fmul_rn(zolold,zt),za);
if ((ri<0)) {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_psih_unstable(zol3,psi_opt),m_psih_unstable(zolt,psi_opt))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_psim_unstable(zol3,psi_opt),m_psim_unstable(zol20,psi_opt))),0x1.0000000000000p+0f);
} else {
psit2 = fmaxf(__fsub_rn(logzt,__fsub_rn(m_psih_stable(zol3,psi_opt),m_psih_stable(zolt,psi_opt))),0x1.0000000000000p+0f);
psix2 = fmaxf(__fsub_rn(logz0,__fsub_rn(m_psim_stable(zol3,psi_opt),m_psim_stable(zol20,psi_opt))),0x1.0000000000000p+0f);
}
result = __fdiv_rn(__fmul_rn(ri,__fmul_rn(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(fabsf(__fsub_rn(zolold,result))>0x1.47ae140000000p-7f))) {
m_li_etal_2010(result,ri,__fdiv_rn(za,z0),__fdiv_rn(z0,zt));
} else {
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:1967-2004 psi_init
__device__ __noinline__ void m_psi_init(int psi_opt) {
int n;
float zolf;
if ((psi_opt==0)) {
for (n=0; n<=1000; ++n) {
zolf = __fmul_rn((float)(n),0x1.47ae140000000p-7f);
psim_stab[n] = m_psim_stable_full(zolf);
psih_stab[n] = m_psih_stable_full(zolf);
zolf = __fmul_rn((-(float)(n)),0x1.47ae140000000p-7f);
psim_unstab[n] = m_psim_unstable_full(zolf);
psih_unstab[n] = m_psih_unstable_full(zolf);
}
} else {
for (n=0; n<=1000; ++n) {
zolf = __fmul_rn((float)(n),0x1.47ae140000000p-7f);
psim_stab[n] = m_psim_stable_full_gfs(zolf);
psih_stab[n] = m_psih_stable_full_gfs(zolf);
zolf = __fmul_rn((-(float)(n)),0x1.47ae140000000p-7f);
psim_unstab[n] = m_psim_unstable_full_gfs(zolf);
psih_unstab[n] = m_psih_unstable_full_gfs(zolf);
}
}
}
// MPAS physics_mmm/sf_mynn.F90:2009-2016 psim_stable_full
__device__ __noinline__ float m_psim_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.8666660000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(1,SL_POW(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2019-2026 psih_stable_full
__device__ __noinline__ float m_psih_stable_full(float zolf) {
float result;
result = __fmul_rn(-0x1.5333340000000p+2f,SL_LOG(__fadd_rn(zolf,SL_POW(__fadd_rn(1,SL_POW(zolf,0x1.19999a0000000p+0f)),0x1.d1745c0000000p-1f))));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2029-2043 psim_unstable_full
__device__ __noinline__ float m_psim_unstable_full(float zolf) {
float result;
float x;
float ym;
float psimc;
float psimk;
x = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = __fadd_rn(__fsub_rn(__fadd_rn(__fmul_rn(2,SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(1,x)))),SL_LOG(__fmul_rn(0x1.0000000000000p-1f,__fadd_rn(1,__fmul_rn(x,x))))),__fmul_rn(0x1.0000000000000p+1f,SL_ATAN(x))),0x1.921fb60000000p+0f);
ym = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+3f,zolf)),0x1.51eb860000000p-2f);
psimc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fdiv_rn(__fadd_rn(__fadd_rn(__fmul_rn(ym,ym),ym),0x1.0000000000000p+0f),0x1.8000000000000p+1f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psimk,__fmul_rn(__fmul_rn(zolf,zolf),psimc)),__fadd_rn(1,__fmul_rn(zolf,zolf)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2046-2060 psih_unstable_full
__device__ __noinline__ float m_psih_unstable_full(float zolf) {
float result;
float y;
float yh;
float psihc;
float psihk;
y = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = __fmul_rn(0x1.0000000000000p+1f,SL_LOG(__fdiv_rn(__fadd_rn(1,y),0x1.0000000000000p+1f)));
yh = SL_POW(__fsub_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.1000000000000p+5f,zolf)),0x1.51eb860000000p-2f);
psihc = __fadd_rn(__fsub_rn(__fmul_rn(0x1.8000000000000p+0f,SL_LOG(__fdiv_rn(__fadd_rn(__fadd_rn(__fmul_rn(yh,yh),yh),0x1.0000000000000p+0f),0x1.8000000000000p+1f))),__fmul_rn(0x1.bb67ae0000000p+0f,SL_ATAN(__fdiv_rn(__fadd_rn(__fmul_rn(0x1.0000000000000p+1f,yh),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = __fdiv_rn(__fadd_rn(psihk,__fmul_rn(__fmul_rn(zolf,zolf),psihc)),__fadd_rn(1,__fmul_rn(zolf,zolf)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2066-2076 psim_stable_full_gfs
__device__ __noinline__ float m_psim_stable_full_gfs(float zolf) {
float result;
float aa;
float alpha4 = 0x1.4000000000000p+4f;
aa = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,aa),SL_LOG(__fadd_rn(aa,0x1.0000000000000p+0f)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2079-2089 psih_stable_full_gfs
__device__ __noinline__ float m_psih_stable_full_gfs(float zolf) {
float result;
float bb;
float alpha4 = 0x1.4000000000000p+4f;
bb = SL_SQRT(__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(0x1.4000000000000p+4f,zolf)));
result = __fadd_rn(__fmul_rn(-0x1.0000000000000p+0f,bb),SL_LOG(__fadd_rn(bb,0x1.0000000000000p+0f)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2092-2109 psim_unstable_full_gfs
__device__ __noinline__ float m_psim_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0 = -0x1.fccccc0000000p+1f;
float a1 = 0x1.8a3d700000000p+3f;
float b1 = -0x1.f051ec0000000p+2f;
float b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fccccc0000000p+1f,__fmul_rn(0x1.8a3d700000000p+3f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.f051ec0000000p+2f,__fmul_rn(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fsub_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p+1f,SL_SQRT(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2112-2129 psih_unstable_full_gfs
__device__ __noinline__ float m_psih_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0p = -0x1.fc39580000000p+2f;
float a1p = 0x1.8c00000000000p+4f;
float b1p = -0x1.168f5c0000000p+3f;
float b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = __fdiv_rn(__fmul_rn(__fadd_rn(-0x1.fc39580000000p+2f,__fmul_rn(0x1.8c00000000000p+4f,hl1)),hl1),__fadd_rn(0x1.0000000000000p+0f,__fmul_rn(__fadd_rn(-0x1.168f5c0000000p+3f,__fmul_rn(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = __fdiv_rn(0x1.0000000000000p+0f,SL_SQRT(hl1));
result = __fadd_rn(__fadd_rn(SL_LOG(hl1),__fmul_rn(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2135-2157 psim_stable
__device__ __noinline__ float m_psim_stable(float zolf, int psi_opt) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = __fadd_rn(psim_stab[nzol],__fmul_rn(rzol,__fsub_rn(psim_stab[(nzol+1)],psim_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_psim_stable_full(zolf);
} else {
result = m_psim_stable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2160-2182 psih_stable
__device__ __noinline__ float m_psih_stable(float zolf, int psi_opt) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn(zolf,0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = __fadd_rn(psih_stab[nzol],__fmul_rn(rzol,__fsub_rn(psih_stab[(nzol+1)],psih_stab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_psih_stable_full(zolf);
} else {
result = m_psih_stable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2185-2207 psim_unstable
__device__ __noinline__ float m_psim_unstable(float zolf, int psi_opt) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = __fadd_rn(psim_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psim_unstab[(nzol+1)],psim_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_psim_unstable_full(zolf);
} else {
result = m_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2210-2232 psih_unstable
__device__ __noinline__ float m_psih_unstable(float zolf, int psi_opt) {
float result;
int nzol;
float rzol;
nzol = (int)(__fmul_rn((-zolf),0x1.9000000000000p+6f));
rzol = __fsub_rn(__fmul_rn((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = __fadd_rn(psih_unstab[nzol],__fmul_rn(rzol,__fsub_rn(psih_unstab[(nzol+1)],psih_unstab[nzol])));
} else {
if ((psi_opt==0)) {
result = m_psih_unstable_full(zolf);
} else {
result = m_psih_unstable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:130-996
extern "C" __global__ void sf_mynn_columns(float *fields, const float *constants, int itimestep, int isfflx, int spp_pbl, int isftcflx, int iz0tlnd, unsigned present, int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
m_sf_mynn_run(fields[0*n+i],fields[1*n+i],fields[2*n+i],fields[3*n+i],fields[4*n+i],fields[5*n+i],fields[6*n+i],fields[7*n+i],fields[8*n+i],fields[9*n+i],constants[0],constants[1],constants[2],constants[3],constants[4],fields[10*n+i],fields[11*n+i],fields[12*n+i],fields[13*n+i],fields[14*n+i],fields[15*n+i],fields[16*n+i],fields[17*n+i],fields[18*n+i],fields[19*n+i],fields[20*n+i],fields[21*n+i],fields[22*n+i],fields[23*n+i],fields[24*n+i],fields[25*n+i],fields[26*n+i],fields[27*n+i],fields[28*n+i],fields[29*n+i],fields[30*n+i],fields[31*n+i],fields[32*n+i],fields[33*n+i],fields[34*n+i],fields[35*n+i],fields[36*n+i],fields[37*n+i],fields[38*n+i],fields[39*n+i],fields[40*n+i],fields[41*n+i],fields[42*n+i],isfflx,fields[43*n+i],constants[5],constants[6],constants[7],constants[8],constants[9],constants[10],constants[11],fields[44*n+i],fields[45*n+i],itimestep,fields[46*n+i],fields[47*n+i],fields[48*n+i],fields[49*n+i],fields[50*n+i],fields[51*n+i],fields[52*n+i],spp_pbl,fields[53*n+i],isftcflx,iz0tlnd,0,0, (present & (1u<<0))!=0, (present & (1u<<1))!=0, (present & (1u<<2))!=0, (present & (1u<<3))!=0, (present & (1u<<4))!=0, (present & (1u<<5))!=0, (present & (1u<<6))!=0);
}
// MPAS physics_mmm/sf_mynn.F90:1967-2004 psi_init
extern "C" __global__ void sf_mynn_tables() { if(blockIdx.x==0 && threadIdx.x==0) m_psi_init(0); }
"""

_MODULES = {}


def _cuda_source(corrections=frozenset(), waves=False):
    """Assemble sf_mynn.F90:130-2232 with shared glibc 2.43 helpers."""
    kernels = files('woof').joinpath('core/kernels')
    libm = kernels.joinpath('mynn_libm.cuh').read_text()
    trig = kernels.joinpath('glibc_trig_flt32.cuh').read_text()
    # The older MYNN header retains fdlibm atanf from glibc 2.39.
    # Native glibc 2.43 uses CORE-MATH; reuse the existing shared header.
    preamble = '''typedef float real;
#define SL_LOG mynn_glibc_logf
#define SL_EXP mynn_glibc_expf
#define SL_POW mynn_glibc_powf
#define SL_ATAN glibc_atanf
#define SL_SQRT __fsqrt_rn
'''
    source = preamble + libm + trig + SOURCE + _driver_source(corrections) + LEAF_SOURCE
    if waves:
        from ..wave_surface import mpas_mynn_source
        source = mpas_mynn_source(source)
    return source


CUDA_OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true')


def _module(corrections=frozenset(), waves=False):
    """Initialize MPAS sf_mynn_init (sf_mynn.F90:92-106) tables per device."""
    import cupy as cp
    from cupy.cuda import function
    from .strict_cache_v842 import compile_image
    device = cp.cuda.runtime.getDevice()
    key = (device, frozenset(corrections).intersection(CORRECT_MODE_SOURCE), waves)
    if key not in _MODULES:
        # CuPy 14.2's RawModule cache appends -ftz=true after caller options.
        # NATIVE-DEFECT-POLICY.md's shared FTZ defect requires bypassing that
        # cache compiler: retain subnormal MPAS sea-ice blend words (:672-747).
        from ..cuda_backend.aot import direct_image
        binary = direct_image(_cuda_source(key[1], waves=waves), _hex_numerics.options(CUDA_OPTIONS),
                              _compilation_target(device, xp=cp))
        module = function.Module()
        module.load(binary)
        module.get_function('sf_mynn_tables')((1,), (1,), ())
        _MODULES[key] = module
    return _MODULES[key]


def run_packed(fields, constants, *, itimestep, isfflx=1, spp_pbl=False,
               isftcflx=0, iz0tlnd=0, present=127, ocean_z0=None):
    """Execute sf_mynn_run (sf_mynn.F90:130-996) on field-major device storage.

    FIELDS and CONSTANTS give the exact MPAS argument order. MPAS driver
    passes initflag=1 only on a first non-restart call, otherwise zero.
    ``present`` is the OPTIONAL bitmask; every optional argument is present
    in MPAS's surface driver. Storage is updated in place on the active stream.
    """
    import cupy as cp
    if not isinstance(fields, cp.ndarray) or fields.dtype != cp.float32:
        raise TypeError('MYNN RKIND field storage must be float32 on the device')
    if fields.ndim != 2 or fields.shape[0] != len(FIELDS) or not fields.flags.c_contiguous:
        raise ValueError('MYNN field storage must be contiguous (len(FIELDS), nCells)')
    if constants.shape != (len(CONSTANTS),) or constants.dtype != cp.float32:
        raise ValueError('MYNN constants must match the float32 MPAS argument order')
    n = fields.shape[1]
    wave_args = ()
    if ocean_z0 is not None:
        from ..wave_surface import validate
        validate(ocean_z0, n)
        wave_args = (ocean_z0,)
    if n:
        _module(waves=ocean_z0 is not None).get_function('sf_mynn_columns')(
            ((n + 63) // 64,), (64,),
            (fields, constants, np.int32(itimestep), np.int32(isfflx),
             np.int32(spp_pbl), np.int32(isftcflx), np.int32(iz0tlnd),
             np.uint32(present), np.int32(n)) + wave_args)
    return fields


def run_columns(values: Mapping, constants: Mapping, **options):
    """Launch sf_mynn.F90:130-996; values retain MPAS argument names."""
    import cupy as cp
    first = values['u1d']
    packed = cp.empty((len(FIELDS), first.size), dtype=cp.float32)
    for index, name in enumerate(FIELDS):
        packed[index] = values[name]
    scalar_values = cp.asarray([constants[name] for name in CONSTANTS], dtype=cp.float32)
    run_packed(packed, scalar_values, **options)
    return {name: packed[FIELDS.index(name)] for name in OUTPUTS}


def sf_mynn_init():
    """MPAS sf_mynn_init (sf_mynn.F90:92-106), including psi table setup."""
    _module()
    return ' ', 0


def sf_mynn_finalize():
    """MPAS sf_mynn_finalize (sf_mynn.F90:112-124) has no numerical work."""
    return ' ', 0

# MPAS physics_wrf/sf_mynn_pre.F:58-165; profile storage is (level, cell),
# matching Fortran's cell-contiguous (cell, level) arrays.
DRIVER_SOURCE = r'''
// MPAS physics_wrf/sf_mynn_pre.F:58-165
extern "C" __global__ void sf_mynn_pre_columns(
    float *f, const float *u, const float *v, const float *t,
    const float *qv, const float *qc, const float *p, const float *rho,
    const float *dz, const float *pattern, float *qc1d,
    int first, int spp, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    F(u1d)=u[i]; F(v1d)=v[i]; F(t1d)=t[i]; F(qv1d)=qv[i];
    F(p1d)=p[i]; F(rho1d)=rho[i]; F(dz8w1d)=dz[i]; qc1d[i]=qc[i];
    F(u1d2)=u[n+i]; F(v1d2)=v[n+i]; F(dz2w1d)=dz[n+i];
    F(rstoch1d)=spp ? pattern[i] : 0.0f;
    if(first) {
        F(ust)=fmaxf(__fmul_rn(0.04f,__fsqrt_rn(__fadd_rn(
            __fmul_rn(u[i],u[i]),__fmul_rn(v[i],v[i])))),0.001f);
        F(mol)=0.0f; F(qsfc)=__fdiv_rn(qv[i],__fadd_rn(1.0f,qv[i]));
        F(qstar)=0.0f;
    }
}
// MPAS mpas_atmphys_driver_sfclayer.F:429-438,470-538.
extern "C" __global__ void sf_mynn_from_mpas(
    float *f, float *sea, const float *density, const float *sst,
    const float *ice, float len_disp, float threshold, int fraction, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    F(dx)=__fdiv_rn(len_disp,SL_POW(density[i],0.25f));
    if(fraction) {
        for(int j=0;j<FIELD_COUNT;++j) sea[j*n+i]=f[j*n+i];
        sea[IDX_qsfc*n+i]=0.0f;
        if(ice[i]>=threshold && ice[i]<=1.0f) {
            sea[IDX_xland*n+i]=2.0f;
            sea[IDX_mavail*n+i]=1.0f;
            sea[IDX_znt*n+i]=0.0001f;
            sea[IDX_tsk*n+i]=fmaxf(sst[i],271.4f);
        }
    }
}
// MPAS mpas_atmphys_driver_sfclayer.F:672-747,773-788.
extern "C" __global__ void sf_mynn_to_mpas(
    float *f, const float *sea, const float *ice, const float *regime_hold,
    float threshold, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || ice[i]<threshold || ice[i]>1.0f) return;
    float x=ice[i];
    float y=__fsub_rn(1.0f,x);
    for(int j=0;j<FIELD_COUNT;++j) {
        if(BLEND_FIELD(j)) f[j*n+i]=__fadd_rn(__fmul_rn(f[j*n+i],x),
                                            __fmul_rn(y,sea[j*n+i]));
    }
    if(x>=0.5f) F(regime)=regime_hold[i];
}
'''

# Only these native arrays are weighted. Fluxes/cpm/qsfc/znt keep the ice-call
# values; other MPAS driver stages own their handling.
_BLEND = ('br','flhc','flqc','gz1oz0','mol','psih','psim','rmol','ust','wspd',
          'zol','q2','t2','th2','u10','v10','cd','cda','ck','cka','ustm','ch')
_PROFILE = ('u3d','v3d','t3d','qv3d','qc3d','p3d','rho3d','dz8w')
_SURFACE = {'tsk':'skintemp','xland':'xland','snowh':'snowh','psfcpa':'psfc'}
_DIAG_INPUT = ('br','cpm','chs','chs2','cqs2','flhc','flqc','gz1oz0','hfx','qfx',
               'qgh','qsfc','lh','mol','psim','psih','regime','rmol','ust','ustm',
               'wspd','znt','zol','mavail','ch','qcg')
_DIAG_NAMES = {'t2':'t2m','th2':'th2m'}
# sfclayer.F:155-170,245-314 deliberately retains these ocean-call arrays;
# driver_seaice.F:390-399 consumes them after the land-surface call.
_SEA_HANDOFF = ('chs','chs2','cqs2','cpm','hfx','qfx','qgh','qsfc','lh')


# RP3: seaice_component_tsk in the native correct-mode referee (lines 60-103),
# called by corrected sfclayer_from_MPAS after its fractional sea-ice copy.
# Keep the exact text intact: correct_source changes only the selected variant.
CORRECT_MODE_SOURCE = {
    'RP3': ((
        '            sea[IDX_tsk*n+i]=fmaxf(sst[i],271.4f);',
        '''            sea[IDX_tsk*n+i]=fmaxf(sst[i],271.4f);
            // RP3 referee seaice_component_tsk, lines 87-98.
            float blended=F(tsk);
            float recovered=__fdiv_rn(__fsub_rn(blended,
                __fmul_rn(__fsub_rn(1.0f,ice[i]),sea[IDX_tsk*n+i])),ice[i]);
            F(tsk)=fmaxf(recovered,221.4f);
            if(ice[i]<0.2f && blended<253.15f) F(tsk)=253.15f;
            if(ice[i]<0.1f && blended<263.15f) F(tsk)=263.15f;''', 1),)
}


def _driver_source(corrections=frozenset()):
    """Bind mpas_atmphys_driver_sfclayer.F:352-796,839-1088 fields."""
    macros = '\n'.join('#define IDX_'+name+' '+str(i) for i,name in enumerate(FIELDS))
    macros += '\n#define FIELD_COUNT '+str(len(FIELDS))
    macros += '\n#define F(name) f[IDX_##name*n+i]\n'
    macros += '#define BLEND_FIELD(j) ('+' || '.join('(j)==IDX_'+k for k in _BLEND)+')\n'
    from .driver_v842 import correct_source
    return macros+correct_source(DRIVER_SOURCE, CORRECT_MODE_SOURCE, corrections,
                                 'sf_mynn_v842 driver')


def preprocess(packed, profiles: Mapping, *, itimestep, spp_pbl=False, pattern_spp=None):
    """MPAS sf_mynn_pre_run (sf_mynn_pre.F:58-165), including first-call seed.

    Returns qc1d, a preprocessing output unused by sf_mynn_run. All profile
    arrays must hold at least two levels with cells contiguous at each level.
    """
    import cupy as cp
    n=packed.shape[1]
    for key in _PROFILE:
        value=profiles[key]
        if value.dtype!=cp.float32 or value.ndim!=2 or value.shape[0]<2 or value.shape[1]!=n or not value.flags.c_contiguous:
            raise ValueError('MYNN profile '+key+' must be float32 (levels>=2, nCells), contiguous')
    active = bool(spp_pbl == 1 and pattern_spp is not None)
    pattern = pattern_spp if active else packed[0]
    qc1d=cp.empty(n,cp.float32)
    if n:
        _module().get_function('sf_mynn_pre_columns')(((n+63)//64,),(64,),
            (packed,*(profiles[k] for k in _PROFILE),pattern,qc1d,
             np.int32(itimestep==1),np.int32(active),np.int32(n)))
    return qc1d


def from_mpas(mesh: Mapping, diagnostics: Mapping, surface: Mapping, constants: Mapping,
              *, len_disp, xice_threshold, config_frac_seaice=False,
              corrections=frozenset()):
    """sfclayer_from_MPAS, mpas_atmphys_driver_sfclayer.F:352-592.

    Framework pools supply device arrays with MPAS field names. Atmosphere
    profiles are prepared by the enclosing physics driver and passed separately.
    """
    import cupy as cp
    n=surface['xland'].size
    packed=cp.zeros((len(FIELDS),n),cp.float32)
    for name in _DIAG_INPUT:
        packed[FIELDS.index(name)]=diagnostics[name]
    packed[FIELDS.index('pblh')]=diagnostics['hpbl']
    for name,source in _SURFACE.items():
        packed[FIELDS.index(name)]=surface[source]
    sea=cp.empty_like(packed) if config_frac_seaice else packed
    ice=surface['xice'] if config_frac_seaice else packed[0]
    sst=surface['sst'] if config_frac_seaice else packed[0]
    if n:
        _module(corrections).get_function('sf_mynn_from_mpas')(((n+63)//64,),(64,),
            (packed,sea,mesh['meshDensity'],sst,ice,np.float32(len_disp),
             np.float32(xice_threshold),np.int32(config_frac_seaice),np.int32(n)))
    return packed, sea


def to_mpas(packed, diagnostics: Mapping, surface: Mapping, *, sea=None,
            xice_threshold, regime_hold=None):
    """sfclayer_to_MPAS, mpas_atmphys_driver_sfclayer.F:595-796."""
    import cupy as cp
    n=packed.shape[1]
    if sea is not None and n:
        _module().get_function('sf_mynn_to_mpas')(((n+63)//64,),(64,),
            (packed,sea,surface['xice'],regime_hold,np.float32(xice_threshold),np.int32(n)))
    for name in OUTPUTS:
        # wstar and qstar are local to the WRF wrapper, not MPAS pool outputs.
        if name not in ('wstar','qstar'):
            diagnostics[_DIAG_NAMES.get(name,name)][...] = packed[FIELDS.index(name)]
    return diagnostics


def driver(mesh: Mapping, diagnostics: Mapping, surface: Mapping, profiles: Mapping,
           constants: Mapping, *, itimestep, config_do_restart=False, len_disp,
           xice_threshold, config_frac_seaice=False, isfflx=1, isftcflx=0,
           iz0tlnd=0, spp_pbl=0, pattern_spp=None, sea_state=None,
           corrections=frozenset(), ocean_z0=None):
    """MPAS MYNN driver slice, mpas_atmphys_driver_sfclayer.F:839-1088.

    The MPAS call site passes no pattern_spp_pbl, so stochastic perturbations
    remain inactive there. Supplying a pattern here reproduces the wrapper's
    existing optional-argument path (module_sf_mynn.F:223-388).
    MPAS fixes isftcflx=0 and iz0tlnd=0 in
    mpas_atmphys_driver_sfclayer.F:31-32; their arguments here also allow
    standalone module-oracle coverage of the source's other defined paths.
    """
    import cupy as cp
    initflag=1 if not config_do_restart and itimestep<=1 else 0
    hold=diagnostics['regime'].copy() if config_frac_seaice else None
    packed,sea=from_mpas(mesh,diagnostics,surface,constants,len_disp=len_disp,
                         xice_threshold=xice_threshold,config_frac_seaice=config_frac_seaice,
                         corrections=corrections)
    scalar_values=cp.asarray([constants[k] for k in CONSTANTS],cp.float32)
    options=dict(itimestep=initflag,isfflx=isfflx,isftcflx=isftcflx,iz0tlnd=iz0tlnd,
                 spp_pbl=bool(spp_pbl==1 and pattern_spp is not None))
    preprocess(packed,profiles,itimestep=initflag,spp_pbl=spp_pbl,pattern_spp=pattern_spp)
    run_packed(packed,scalar_values,ocean_z0=ocean_z0,**options)
    if config_frac_seaice:
        preprocess(sea,profiles,itimestep=initflag,spp_pbl=spp_pbl,pattern_spp=pattern_spp)
        run_packed(sea,scalar_values,ocean_z0=ocean_z0,**options)
        # MPAS sfclayer.F:245-314 retains these arrays across deallocation;
        # driver_seaice.F:390-399 blends their unweighted ocean-call values.
        # Copy into the adapter's prepared device views before it commits.
        if sea_state is not None:
            for name in _SEA_HANDOFF:
                sea_state[name+'_sea'][...] = sea[FIELDS.index(name)]
    return to_mpas(packed,diagnostics,surface,sea=sea if config_frac_seaice else None,
                   xice_threshold=xice_threshold,regime_hold=hold)

# Private routines are exposed for whole-module oracle replay only.

_NATIVE_PROFILE_INPUTS = (
    'u_p', 'v_p', 't_p', 'qv_p', 'qc_p', 'pres_hyd_p', 'rho_p', 'dz_p',
)


def _discard_native_profile_inputs(provider):
    """Retire input copies after MYNN; sf_mynn_pre.F:73-86 INTENT(IN).

    Only the const profile arguments to sf_mynn_pre_columns are retired.
    Surface diagnostics and the fractional-seaice handoff remain writable.
    Native storage remains authoritative; each later slot reads it afresh.
    Standalone callers may supply ordinary mappings without native buffers.
    """
    buffers = getattr(provider, 'buffers', None)
    if buffers is not None:
        for name in _NATIVE_PROFILE_INPUTS:
            buffers.pop(('prepared', name), None)


class _SurfaceProfileViews(Mapping):
    """Read-only lowest levels; sf_mynn_pre.F:125-140 reads kts and kts+1.

    Keep native host copies alive through both land/ocean preprocessing calls.
    These inputs never enter the generic write-back buffer. The native getter
    still copies complete profiles; only their device uploads are narrowed.
    """
    def __init__(self, profiles):
        self.profiles = profiles
        self.buffers = {}

    def __iter__(self):
        return iter(self.profiles)

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, name):
        if name in self.buffers:
            return self.buffers[name][1]
        owner = self.profiles.owner
        native_name = self.profiles.aliases[name]
        if getattr(owner, 'resident', False):
            return owner.prepared(native_name)[:2]
        value = owner.native.get_variable(native_name)
        if value is None:
            raise RuntimeError(f'native prepared array {native_name} is not allocated before CUDA slot')
        host, lower = value
        count = owner.session.layout.dimensions['nCellsSolve']
        if (host.ndim != 3 or host.shape[2] != 1 or lower[0] != 1
                or host.shape[0] < count or host.shape[1] < 2):
            raise ValueError(f'native MYNN profile {native_name} must have i,k,one-j bounds and at least two levels')
        packed = owner.cp.ascontiguousarray(owner.cp.asarray(host[:count, :2, 0].T))
        self.buffers[name] = (host, packed)
        return packed


def launch_slot(call):
    """f4 SlotCall boundary, mpas_atmphys_driver_sfclayer.F:839-1088.

    The forecast adapter supplies solve-cell device views in named MPAS pools:
    mesh, diag_physics, sfc_input, and prepared atm_input. configs carries the
    framework config_len_disp/config_do_restart values; constants uses the
    sf_mynn_run scalar argument names. scalar_state carries the manager's
    xice_threshold (mpas_atmphys_manager.F:739-744). Common preparation and
    halo handling belong to the enclosing adapter, not this scheme callback.
    """
    if (call.slot, call.scheme, str(call.backend)) != ('surface_layer', SCHEME, BACKEND):
        raise ValueError('MYNN launcher requires surface_layer/sf_mynn/hex-cuda')
    pools = call.pools
    # Resolve every pool/scalar before any device work, so a missing adapter
    # binding cannot leave a partially updated physics pool.
    mesh, diagnostics, surface, profiles, constants, configs, state = (
        pools[name] for name in ('mesh', 'diag_physics', 'sfc_input', 'atm_input',
                                'constants', 'configs', 'scalar_state'))
    len_disp = configs['config_len_disp']
    restart = configs['config_do_restart']
    threshold = state['xice_threshold']
    fraction = call.options['config_frac_seaice']
    # mpas_atmphys_interface.F:521 prepares psfc_p from diag.surface_pressure;
    # mpas_atmphys_driver_sfclayer.F:1017,1048 passes it to both MYNN calls.
    # It is a prepared global, not a field in the native sfc_input pool.
    surface = ChainMap({'psfc': pools['prepared']['psfc_p']}, surface)
    from .device_pools_v842 import _ProfileViews
    if isinstance(profiles, _ProfileViews):
        profiles = _SurfaceProfileViews(profiles)
    # Native parameters at sfclayer.F:30-32; the driver passes no stochastic
    # pattern to this wrapper, so sf_mynn_pre.F:142-156 leaves it inactive.
    from ..wave_surface import feedback
    result = driver(mesh, diagnostics, surface, profiles, constants,
                  itimestep=call.itimestep, config_do_restart=restart,
                  len_disp=len_disp, xice_threshold=threshold,
                  config_frac_seaice=fraction, isfflx=1, isftcflx=0,
                  iz0tlnd=0, spp_pbl=0, sea_state=pools['prepared'],
                  corrections=getattr(call, 'corrections', frozenset()), ocean_z0=feedback(pools))
    _discard_native_profile_inputs(pools)
    return result


def register_backend(registry):
    """Register MYNN only; mpas_atmphys_driver_sfclayer.F:970-1088."""
    registry.register('surface_layer', SCHEME, BACKEND, launch_slot,
                      corrections=CORRECT_MODE_SOURCE)

LEAF_SOURCE = r"""
// MPAS physics_mmm/sf_mynn.F90:1002-2232
extern "C" __global__ void sf_mynn_leaves(float *data,int which,int psi,int n) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
switch(which) {
case 0: m_zilitinkevich_1995(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i],data[6*n+i],psi,data[9*n+i]>0.5f,data[7*n+i],true); break;
case 1: m_davis_etal_2008(data[0*n+i],data[1*n+i]); break;
case 2: m_taylor_yelland_2001(data[0*n+i],data[1*n+i],data[2*n+i]); break;
case 3: m_charnock_1955(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i]); break;
case 4: m_edson_etal_2013(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i]); break;
case 5: m_garratt_1992(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i]); break;
case 6: m_fairall_etal_2003(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i],data[9*n+i]>0.5f); break;
case 7: m_fairall_etal_2014(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i],data[9*n+i]>0.5f); break;
case 8: m_yang_2008(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i],data[6*n+i],data[7*n+i],data[8*n+i]); break;
case 9: m_andreas_2002(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i]); break;
case 10: m_psi_hogstrom_1996(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i]); break;
case 11: m_psi_dyerhicks(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i]); break;
case 12: m_psi_beljaars_holtslag_1991(data[0*n+i],data[1*n+i],data[2*n+i]); break;
case 13: m_psi_zilitinkevich_esau_2007(data[0*n+i],data[1*n+i],data[2*n+i]); break;
case 14: m_psi_businger_1971(data[0*n+i],data[1*n+i],data[2*n+i]); break;
case 15: m_psi_suselj_sood_2010(data[0*n+i],data[1*n+i],data[2*n+i]); break;
case 16: m_psi_cb2005(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i]); break;
case 17: m_li_etal_2010(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i]); break;
case 18: data[10*n+i]=m_zolri(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],psi); break;
case 19: data[10*n+i]=m_zolri2(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],psi); break;
case 20: data[10*n+i]=m_zolrib(data[0*n+i],data[1*n+i],data[2*n+i],data[3*n+i],data[4*n+i],data[5*n+i],data[6*n+i],psi); break;
case 21: data[10*n+i]=m_psim_stable_full(data[0*n+i]); break;
case 22: data[10*n+i]=m_psih_stable_full(data[0*n+i]); break;
case 23: data[10*n+i]=m_psim_unstable_full(data[0*n+i]); break;
case 24: data[10*n+i]=m_psih_unstable_full(data[0*n+i]); break;
case 25: data[10*n+i]=m_psim_stable_full_gfs(data[0*n+i]); break;
case 26: data[10*n+i]=m_psih_stable_full_gfs(data[0*n+i]); break;
case 27: data[10*n+i]=m_psim_unstable_full_gfs(data[0*n+i]); break;
case 28: data[10*n+i]=m_psih_unstable_full_gfs(data[0*n+i]); break;
case 29: data[10*n+i]=m_psim_stable(data[0*n+i],psi); break;
case 30: data[10*n+i]=m_psih_stable(data[0*n+i],psi); break;
case 31: data[10*n+i]=m_psim_unstable(data[0*n+i],psi); break;
case 32: data[10*n+i]=m_psih_unstable(data[0*n+i],psi); break;
}
}
"""

LEAF_SOURCE += r'''
// MPAS physics_mmm/sf_mynn.F90:1967-2004 psi_init
extern "C" __global__ void sf_mynn_leaf_tables(int psi) {
    if(blockIdx.x==0 && threadIdx.x==0) m_psi_init(psi);
}
'''
