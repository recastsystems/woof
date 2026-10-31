"""MPAS-A v8.4.2 MM5 revised/classic surface layer and driver slice.

One CUDA thread per MPAS cell; all RKIND data are float32. Atmospheric
inputs are the lowest model level prepared by MPAS's physics driver.
Packed storage has shape (len(FIELDS), nCells), with contiguous rows.
Every optional argument supplied by MPAS is carried, including old USTM.
The standalone replay also preserves optional-argument absence semantics.
WOOF kernels and suite defaults are untouched.

Sources: physics_mmm/sf_sfclayrev.F90:1-1119;
physics_wrf/module_sf_sfclay.F:1-968; module_sf_sfclayrev.F:1-281;
sf_sfclayrev_pre.F:1-101; mpas_atmphys_driver_sfclayer.F:352-1088.
"""
from __future__ import annotations
from woof.hex import numerics as _hex_numerics

from functools import lru_cache
from pathlib import Path
from typing import Mapping

OUTPUTS = ('znt', 'ust', 'ustm', 'mol', 'hfx', 'qfx', 'qsfc', 'zol', 'regime', 'psim', 'psih', 'fm', 'fh', 'lh', 'u10', 'v10', 'th2', 't2', 'q2', 'chs', 'chs2', 'cqs2', 'flhc', 'flqc', 'qgh', 'rmol', 'wspd', 'br', 'gz1oz0', 'cpm', 'ck', 'cka', 'cd', 'cda')
INPUTS = ('ux', 'vx', 't1d', 'qv1d', 'p1d', 'dz8w1d', 'psfcpa', 'tsk', 'pblh', 'mavail', 'xland', 'lakemask', 'dx', 'water_depth')
FIELDS = OUTPUTS + INPUTS
CONSTANTS = ('cp', 'g', 'rovcp', 'r', 'xlv', 'svp1', 'svp2', 'svp3', 'svpt0',
             'ep1', 'ep2', 'karman', 'p1000mb')
SCHEMES = ('sf_monin_obukhov_rev', 'sf_monin_obukhov')
OPTIONAL_ARGUMENTS = ('isftcflx', 'iz0tlnd', 'scm_force_flux', 'ustm',
                      'ck', 'cd', 'cka', 'cda')
ALL_OPTIONAL = 255
COMPILE_OPTIONS = ('--std=c++17', '--fmad=false', '--ftz=false',
                   '--prec-div=true', '--prec-sqrt=true')

CUDA_SOURCE = r"""
// MPAS-A v8.4.2 surface layer: source association, RKIND=4.
// Copyright (c) UCAR and Los Alamos National Security, LLC. BSD-3-Clause.
// Device libm comes from the shared pinned engine headers.

struct MM5State { float znt; float ust; float ustm; float mol; float hfx; float qfx; float qsfc; float zol; float regime; float psim; float psih; float fm; float fh; float lh; float u10; float v10; float th2; float t2; float q2; float chs; float chs2; float cqs2; float flhc; float flqc; float qgh; float rmol; float wspd; float br; float gz1oz0; float cpm; float ck; float cka; float cd; float cda; float ux; float vx; float t1d; float qv1d; float p1d; float dz8w1d; float psfcpa; float tsk; float pblh; float mavail; float xland; float lakemask; float dx; float water_depth; };
struct MM5Constants { float cp; float g; float rovcp; float r; float xlv; float svp1; float svp2; float svp3; float svpt0; float ep1; float ep2; float karman; float p1000mb; };
// MPAS physics_mmm/sf_sfclayrev.F90:987-992 psim_stable_full
__device__ float mm5_psim_stable_full(float zolf, const float *tables) {
float result;

result = FMUL(-0x1.8666660000000p+2f, mm5_log(FADD(zolf, mm5_pow(FADD(0x1.0000000000000p+0f, mm5_pow(zolf, 0x1.4000000000000p+1f)), 0x1.99999a0000000p-2f)))); // MPAS:989
return result; // MPAS:991
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:995-1000 psih_stable_full
__device__ float mm5_psih_stable_full(float zolf, const float *tables) {
float result;

result = FMUL(-0x1.5333340000000p+2f, mm5_log(FADD(zolf, mm5_pow(FADD(0x1.0000000000000p+0f, mm5_pow(zolf, 0x1.19999a0000000p+0f)), 0x1.d1745c0000000p-1f)))); // MPAS:997
return result; // MPAS:999
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1003-1015 psim_unstable_full
__device__ float mm5_psim_unstable_full(float zolf, const float *tables) {
float result;
float psimc;
float psimk;
float x;
float y;
float ym;
x = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.0000000000000p+4f, zolf)), 0x1.0000000000000p-2f); // MPAS:1006
psimk = FADD(FSUB(FADD(FMUL(0x1.0000000000000p+1f, mm5_log(FMUL(0x1.0000000000000p-1f, FADD(0x1.0000000000000p+0f, x)))), mm5_log(FMUL(0x1.0000000000000p-1f, FADD(0x1.0000000000000p+0f, FMUL(x, x))))), FMUL(0x1.0000000000000p+1f, mm5_atan(x))), 0x1.921fb60000000p+0f); // MPAS:1007
ym = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.4000000000000p+3f, zolf)), 0x1.51eb860000000p-2f); // MPAS:1009
psimc = FADD(FSUB(FMUL(0x1.8000000000000p+0f, mm5_log(FDIV(FADD(FADD(FMUL(ym, ym), ym), 0x1.0000000000000p+0f), 0x1.8000000000000p+1f))), FMUL(0x1.bb67ae0000000p+0f, mm5_atan(FDIV(FADD(FMUL(0x1.0000000000000p+1f, ym), 0x1.0000000000000p+0f), 0x1.bb67ae0000000p+0f)))), 0x1.d0552a0000000p+0f); // MPAS:1010
result = FDIV(FADD(psimk, FMUL(FMUL(zolf, zolf), psimc)), FADD(0x1.0000000000000p+0f, FMUL(zolf, zolf))); // MPAS:1012
return result; // MPAS:1014
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1018-1030 psih_unstable_full
__device__ float mm5_psih_unstable_full(float zolf, const float *tables) {
float result;
float psihc;
float psihk;
float y;
float yh;
y = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.0000000000000p+4f, zolf)), 0x1.0000000000000p-1f); // MPAS:1021
psihk = FMUL(0x1.0000000000000p+1f, mm5_log(FDIV(FADD(0x1.0000000000000p+0f, y), 0x1.0000000000000p+1f))); // MPAS:1022
yh = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.1000000000000p+5f, zolf)), 0x1.51eb860000000p-2f); // MPAS:1024
psihc = FADD(FSUB(FMUL(0x1.8000000000000p+0f, mm5_log(FDIV(FADD(FADD(FMUL(yh, yh), yh), 0x1.0000000000000p+0f), 0x1.8000000000000p+1f))), FMUL(0x1.bb67ae0000000p+0f, mm5_atan(FDIV(FADD(FMUL(0x1.0000000000000p+1f, yh), 0x1.0000000000000p+0f), 0x1.bb67ae0000000p+0f)))), 0x1.d0552a0000000p+0f); // MPAS:1025
result = FDIV(FADD(psihk, FMUL(FMUL(zolf, zolf), psihc)), FADD(0x1.0000000000000p+0f, FMUL(zolf, zolf))); // MPAS:1027
return result; // MPAS:1029
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1034-1047 psim_stable
__device__ float mm5_psim_stable(float zolf, const float *tables) {
float result;
int nzol;
float rzol;
nzol = int(FMUL(zolf, 0x1.9000000000000p+6f)); // MPAS:1038
rzol = FSUB(FMUL(zolf, 0x1.9000000000000p+6f), nzol); // MPAS:1039
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1040
result = FADD(tables[0*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[0*1001 + (int)(FADD(nzol, 0x1.0000000000000p+0f))], tables[0*1001 + (int)(nzol)]))); // MPAS:1041
} else { // MPAS:1042
result = mm5_psim_stable_full(zolf, tables); // MPAS:1043
} // MPAS:1044
return result; // MPAS:1046
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1050-1063 psih_stable
__device__ float mm5_psih_stable(float zolf, const float *tables) {
float result;
int nzol;
float rzol;
nzol = int(FMUL(zolf, 0x1.9000000000000p+6f)); // MPAS:1054
rzol = FSUB(FMUL(zolf, 0x1.9000000000000p+6f), nzol); // MPAS:1055
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1056
result = FADD(tables[1*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[1*1001 + (int)(FADD(nzol, 0x1.0000000000000p+0f))], tables[1*1001 + (int)(nzol)]))); // MPAS:1057
} else { // MPAS:1058
result = mm5_psih_stable_full(zolf, tables); // MPAS:1059
} // MPAS:1060
return result; // MPAS:1062
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1066-1079 psim_unstable
__device__ float mm5_psim_unstable(float zolf, const float *tables) {
float result;
int nzol;
float rzol;
nzol = int(FMUL(-(zolf), 0x1.9000000000000p+6f)); // MPAS:1070
rzol = FSUB(FMUL(-(zolf), 0x1.9000000000000p+6f), nzol); // MPAS:1071
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1072
result = FADD(tables[2*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[2*1001 + (int)(FADD(nzol, 0x1.0000000000000p+0f))], tables[2*1001 + (int)(nzol)]))); // MPAS:1073
} else { // MPAS:1074
result = mm5_psim_unstable_full(zolf, tables); // MPAS:1075
} // MPAS:1076
return result; // MPAS:1078
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1082-1095 psih_unstable
__device__ float mm5_psih_unstable(float zolf, const float *tables) {
float result;
int nzol;
float rzol;
nzol = int(FMUL(-(zolf), 0x1.9000000000000p+6f)); // MPAS:1086
rzol = FSUB(FMUL(-(zolf), 0x1.9000000000000p+6f), nzol); // MPAS:1087
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1088
result = FADD(tables[3*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[3*1001 + (int)(FADD(nzol, 0x1.0000000000000p+0f))], tables[3*1001 + (int)(nzol)]))); // MPAS:1089
} else { // MPAS:1090
result = mm5_psih_unstable_full(zolf, tables); // MPAS:1091
} // MPAS:1092
return result; // MPAS:1094
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:960-981 zolri2
__device__ float mm5_zolri2(float &zol2, float ri2, float z, float z0, const float *tables) {
float result;
float psih2;
float psix2;
float zol20;
float zol3;
if ((FMUL(zol2, ri2) < 0x0.0p+0f)) zol2 = 0x0.0p+0f; // MPAS:965
zol20 = FDIV(FMUL(zol2, z0), z); // MPAS:967
zol3 = FADD(zol2, zol20); // MPAS:968
if ((ri2 < 0x0.0p+0f)) { // MPAS:970
psix2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psim_unstable(zol3, tables), mm5_psim_unstable(zol20, tables))); // MPAS:971
psih2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psih_unstable(zol3, tables), mm5_psih_unstable(zol20, tables))); // MPAS:972
} else { // MPAS:973
psix2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psim_stable(zol3, tables), mm5_psim_stable(zol20, tables))); // MPAS:974
psih2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psih_stable(zol3, tables), mm5_psih_stable(zol20, tables))); // MPAS:975
} // MPAS:976
result = FSUB(FDIV(FMUL(zol2, psih2), FMUL(psix2, psix2)), ri2); // MPAS:978
return result; // MPAS:980
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:922-957 zolri
__device__ float mm5_zolri(float ri, float z, float z0, const float *tables) {
float result;
int iter;
float fx1;
float fx2;
float x1;
float x2;
if ((ri < 0x0.0p+0f)) { // MPAS:929
x1 = -0x1.4000000000000p+2f; // MPAS:930
x2 = 0x0.0p+0f; // MPAS:931
} else { // MPAS:932
x1 = 0x0.0p+0f; // MPAS:933
x2 = 0x1.4000000000000p+2f; // MPAS:934
} // MPAS:935
fx1 = mm5_zolri2(x1, ri, z, z0, tables); // MPAS:937
fx2 = mm5_zolri2(x2, ri, z, z0, tables); // MPAS:938
iter = 0x0.0p+0f; // MPAS:939
while ((fabsf(FSUB(x1, x2)) > 0x1.47ae140000000p-7f)) {
if ((iter == 0x1.4000000000000p+3f)) return result; // MPAS:941
if ((fx1 == fx2)) return result; // MPAS:943
if ((fabsf(fx2) < fabsf(fx1))) { // MPAS:944
x1 = FSUB(x1, FMUL(FDIV(fx1, FSUB(fx2, fx1)), FSUB(x2, x1))); // MPAS:945
fx1 = mm5_zolri2(x1, ri, z, z0, tables); // MPAS:946
result = x1; // MPAS:947
} else { // MPAS:948
x2 = FSUB(x2, FMUL(FDIV(fx2, FSUB(fx2, fx1)), FSUB(x2, x1))); // MPAS:949
fx2 = mm5_zolri2(x2, ri, z, z0, tables); // MPAS:950
result = x2; // MPAS:951
} // MPAS:952
iter = FADD(iter, 0x1.0000000000000p+0f); // MPAS:953
}
return result; // MPAS:956
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1098-1115 depth_dependent_z0
__device__ float mm5_depth_dependent_z0(float water_depth, float z0, float ust, const float *tables) {
float result;
float depth_b;
float effective_depth;
if ((water_depth < 0x1.4000000000000p+3f)) { // MPAS:1102
effective_depth = 0x1.4000000000000p+3f; // MPAS:1103
} else if ((water_depth > 0x1.9000000000000p+6f)) { // MPAS:1104
effective_depth = 0x1.9000000000000p+6f; // MPAS:1105
} else { // MPAS:1106
effective_depth = water_depth; // MPAS:1107
} // MPAS:1108
depth_b = FMUL(0x1.1111120000000p-5f, mm5_log(FDIV(0x1.3b00000000000p+10f, effective_depth))); // MPAS:1110
result = mm5_exp(FDIV(FSUB(FMUL(0x1.59999a0000000p+1f, ust), FDIV(0x1.cccccc0000000p+0f, depth_b)), FADD(ust, FDIV(0x1.5c28f60000000p-3f, depth_b)))); // MPAS:1111
result = fminf(result, 0x1.99999a0000000p-4f); // MPAS:1112
return result; // MPAS:1114
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:79-920
__device__ void mm5_revised(MM5State &s, const MM5Constants &c, const float *tables, int isfflx, int isftcflx, int iz0tlnd, int scm_force_flux, int shalwater_z0, unsigned int optional_mask = 255) {
float pl, thcon, tvcon, e1, zl, tskv, dthvdz, dthvm, vconv, rzol, rzol2, rzol10, zol2, zol10, dtg, psix, dtthx, psix10, psit, psit2, psiq, psiq2, psiq10, fluxc, vsgd, z0q, visc, restar, czil, gz0ozq, gz0ozt, zw, zn1, zn2, zolzz, zol0, zl2, zl10, z0t, za, thvx, zqkl, zqklp1, thx, qx, psih2, psim2, psih10, psim10, denomq, denomq2, denomt2, wspdi, gz2oz0, gz10oz0, rhox, govrth, tgdsa, scr3, scr4, thgb, psfc, pq, pq2, pq10, czc;
int nzol, nzol2, nzol10;
psfc = FDIV(s.psfcpa, 0x1.f400000000000p+9f); // MPAS:221
tgdsa = s.tsk; // MPAS:227
thgb = FMUL(s.tsk, mm5_pow(FDIV(c.p1000mb, s.psfcpa), c.rovcp)); // MPAS:230
L5:;
L10:;
L26:;
pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:255
scr3 = s.t1d; // MPAS:256
thcon = mm5_pow(FDIV(FMUL(c.p1000mb, 0x1.0624de0000000p-10f), pl), c.rovcp); // MPAS:258
thx = FMUL(scr3, thcon); // MPAS:259
scr4 = scr3; // MPAS:260
thvx = thx; // MPAS:261
qx = 0x0.0p+0f; // MPAS:262
L30:;
s.qgh = 0x0.0p+0f; // MPAS:266
s.flhc = 0x0.0p+0f; // MPAS:267
s.flqc = 0x0.0p+0f; // MPAS:268
s.cpm = c.cp; // MPAS:269
qx = s.qv1d; // MPAS:274
tvcon = FADD(0x1.0000000000000p+0f, FMUL(c.ep1, qx)); // MPAS:275
thvx = FMUL(thx, tvcon); // MPAS:276
scr4 = FMUL(scr3, tvcon); // MPAS:277
L50:;
e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(tgdsa, c.svpt0)), FSUB(tgdsa, c.svp3)))); // MPAS:281
if (((s.xland > 0x1.8000000000000p+0f) && (s.lakemask == 0x0.0p+0f))) e1 = FMUL(e1, 0x1.f5c2900000000p-1f); // MPAS:283
if (((s.xland > 0x1.8000000000000p+0f) || (s.qsfc <= 0x0.0p+0f))) s.qsfc = FDIV(FMUL(c.ep2, e1), FSUB(psfc, e1)); // MPAS:285
e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(s.t1d, c.svpt0)), FSUB(s.t1d, c.svp3)))); // MPAS:288
pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:289
s.qgh = FDIV(FMUL(c.ep2, e1), FSUB(pl, e1)); // MPAS:290
s.cpm = FMUL(c.cp, FADD(0x1.0000000000000p+0f, FMUL(0x1.99999a0000000p-1f, qx))); // MPAS:291
L60:;
L80:;
zqklp1 = 0x0.0p+0f; // MPAS:299
rhox = FDIV(FMUL(psfc, 0x1.f400000000000p+9f), FMUL(c.r, scr4)); // MPAS:300
L90:;
zqkl = FADD(s.dz8w1d, zqklp1); // MPAS:304
L110:;
za = FMUL(0x1.0000000000000p-1f, FADD(zqkl, zqklp1)); // MPAS:308
L120:;
govrth = FDIV(c.g, thx); // MPAS:312
L160:;
s.gz1oz0 = mm5_log(FDIV(FADD(za, s.znt), s.znt)); // MPAS:318
gz2oz0 = mm5_log(FDIV(FADD(0x1.0000000000000p+1f, s.znt), s.znt)); // MPAS:319
gz10oz0 = mm5_log(FDIV(FADD(0x1.4000000000000p+3f, s.znt), s.znt)); // MPAS:320
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:321
zl = s.znt; // MPAS:322
} else { // MPAS:323
zl = 0x1.47ae140000000p-7f; // MPAS:324
} // MPAS:325
s.wspd = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:326
tskv = FMUL(thgb, FADD(0x1.0000000000000p+0f, FMUL(c.ep1, s.qsfc))); // MPAS:328
dthvdz = FSUB(thvx, tskv); // MPAS:329
if ((s.xland < 0x1.8000000000000p+0f)) { // MPAS:336
fluxc = fmaxf(FADD(FDIV(FDIV(s.hfx, rhox), c.cp), FDIV(FMUL(FMUL(c.ep1, tskv), s.qfx), rhox)), 0x0.0p+0f); // MPAS:337
vconv = FMUL(0x1.0000000000000p+0f, mm5_pow(FMUL(FMUL(FDIV(c.g, tgdsa), s.pblh), fluxc), 0x1.51eb860000000p-2f)); // MPAS:339
} else { // MPAS:340
if ((-(dthvdz) >= 0x0.0p+0f)) { // MPAS:341
dthvm = -(dthvdz); // MPAS:342
} else { // MPAS:343
dthvm = 0x0.0p+0f; // MPAS:344
} // MPAS:345
vconv = FSQRT(dthvm); // MPAS:348
} // MPAS:349
vsgd = FMUL(0x1.47ae140000000p-2f, mm5_pow(fmaxf(FSUB(FDIV(s.dx, 0x1.3880000000000p+12f), 0x1.0000000000000p+0f), 0x0.0p+0f), 0x1.51eb860000000p-2f)); // MPAS:351
s.wspd = FSQRT(FADD(FADD(FMUL(s.wspd, s.wspd), FMUL(vconv, vconv)), FMUL(vsgd, vsgd))); // MPAS:352
s.wspd = fmaxf(s.wspd, 0x1.99999a0000000p-4f); // MPAS:353
s.br = FDIV(FMUL(FMUL(govrth, za), dthvdz), FMUL(s.wspd, s.wspd)); // MPAS:354
if ((s.mol < 0x0.0p+0f)) s.br = fminf(s.br, 0x0.0p+0f); // MPAS:356
s.rmol = FMUL(FMUL(FMUL(-(govrth), dthvdz), za), c.karman); // MPAS:357
L260:;
if ((s.br > 0x0.0p+0f)) { // MPAS:381
if ((s.br > 0x1.f400000000000p+7f)) { // MPAS:382
s.zol = mm5_zolri(0x1.f400000000000p+7f, za, s.znt, tables); // MPAS:383
} else { // MPAS:384
s.zol = mm5_zolri(s.br, za, s.znt, tables); // MPAS:385
} // MPAS:386
} // MPAS:387
if ((s.br < 0x0.0p+0f)) { // MPAS:389
if ((s.ust < 0x1.0624de0000000p-10f)) { // MPAS:390
s.zol = FMUL(s.br, s.gz1oz0); // MPAS:391
} else { // MPAS:392
if ((s.br < -0x1.f400000000000p+7f)) { // MPAS:393
s.zol = mm5_zolri(-0x1.f400000000000p+7f, za, s.znt, tables); // MPAS:394
} else { // MPAS:395
s.zol = mm5_zolri(s.br, za, s.znt, tables); // MPAS:396
} // MPAS:397
} // MPAS:398
} // MPAS:399
zolzz = FDIV(FMUL(s.zol, FADD(za, s.znt)), za); // MPAS:403
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, s.znt)), za); // MPAS:404
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, s.znt)), za); // MPAS:405
zol0 = FDIV(FMUL(s.zol, s.znt), za); // MPAS:406
zl2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.zol); // MPAS:407
zl10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.zol); // MPAS:408
if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:410
zl = FMUL(FDIV(0x1.47ae140000000p-7f, za), s.zol); // MPAS:411
} else { // MPAS:412
zl = zol0; // MPAS:413
} // MPAS:414
if ((s.br < 0x0.0p+0f)) goto L310; // MPAS:416
if ((s.br == 0x0.0p+0f)) goto L280; // MPAS:417
s.regime = 0x1.0000000000000p+0f; // MPAS:421
s.psim = FSUB(mm5_psim_stable(zolzz, tables), mm5_psim_stable(zol0, tables)); // MPAS:425
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:426
psim10 = FSUB(mm5_psim_stable(zol10, tables), mm5_psim_stable(zol0, tables)); // MPAS:428
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:429
psim2 = FSUB(mm5_psim_stable(zol2, tables), mm5_psim_stable(zol0, tables)); // MPAS:431
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:432
pq = FSUB(mm5_psih_stable(s.zol, tables), mm5_psih_stable(zl, tables)); // MPAS:436
pq2 = FSUB(mm5_psih_stable(zl2, tables), mm5_psih_stable(zl, tables)); // MPAS:437
pq10 = FSUB(mm5_psih_stable(zl10, tables), mm5_psih_stable(zl, tables)); // MPAS:438
s.rmol = FDIV(s.zol, za); // MPAS:441
goto L320; // MPAS:443
L280:;
s.regime = 0x1.8000000000000p+1f; // MPAS:447
s.psim = 0x0.0p+0f; // MPAS:448
s.psih = s.psim; // MPAS:449
psim10 = 0x0.0p+0f; // MPAS:450
psih10 = psim10; // MPAS:451
psim2 = 0x0.0p+0f; // MPAS:452
psih2 = psim2; // MPAS:453
pq = s.psih; // MPAS:457
pq2 = psih2; // MPAS:458
pq10 = 0x0.0p+0f; // MPAS:459
s.zol = 0x0.0p+0f; // MPAS:461
s.rmol = FDIV(s.zol, za); // MPAS:462
goto L320; // MPAS:464
L310:;
s.regime = 0x1.0000000000000p+2f; // MPAS:469
s.psim = FSUB(mm5_psim_unstable(zolzz, tables), mm5_psim_unstable(zol0, tables)); // MPAS:473
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:474
psim10 = FSUB(mm5_psim_unstable(zol10, tables), mm5_psim_unstable(zol0, tables)); // MPAS:476
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:477
psim2 = FSUB(mm5_psim_unstable(zol2, tables), mm5_psim_unstable(zol0, tables)); // MPAS:479
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:480
pq = FSUB(mm5_psih_unstable(s.zol, tables), mm5_psih_unstable(zl, tables)); // MPAS:484
pq2 = FSUB(mm5_psih_unstable(zl2, tables), mm5_psih_unstable(zl, tables)); // MPAS:485
pq10 = FSUB(mm5_psih_unstable(zl10, tables), mm5_psih_unstable(zl, tables)); // MPAS:486
s.psih = fminf(s.psih, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:490
s.psim = fminf(s.psim, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:491
psih2 = fminf(psih2, FMUL(0x1.cccccc0000000p-1f, gz2oz0)); // MPAS:492
psim10 = fminf(psim10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:493
psih10 = fminf(psih10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:496
s.rmol = FDIV(s.zol, za); // MPAS:497
L320:;
dtg = FSUB(thx, thgb); // MPAS:505
psix = FSUB(s.gz1oz0, s.psim); // MPAS:506
psix10 = FSUB(gz10oz0, psim10); // MPAS:507
psit = FSUB(s.gz1oz0, s.psih); // MPAS:512
psit2 = FSUB(gz2oz0, psih2); // MPAS:513
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:515
zl = s.znt; // MPAS:516
} else { // MPAS:517
zl = 0x1.47ae140000000p-7f; // MPAS:518
} // MPAS:519
psiq = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), za), 0x1.92a7380000000p-16f), FDIV(za, zl))), pq); // MPAS:521
psiq2 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.0000000000000p+1f), 0x1.92a7380000000p-16f), FDIV(0x1.0000000000000p+1f, zl))), pq2); // MPAS:522
psiq10 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.4000000000000p+3f), 0x1.92a7380000000p-16f), FDIV(0x1.4000000000000p+3f, zl))), pq10); // MPAS:525
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:529
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:530
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:531
z0t = FMUL(0x1.cd5f9a0000000p-15f, mm5_pow(restar, -0x1.3333340000000p-1f)); // MPAS:532
z0t = fminf(z0t, 0x1.a36e2e0000000p-14f); // MPAS:533
z0t = fmaxf(z0t, 0x1.12e0be0000000p-29f); // MPAS:534
z0q = z0t; // MPAS:535
zolzz = FDIV(FMUL(s.zol, FADD(za, z0t)), za); // MPAS:538
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0t)), za); // MPAS:539
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0t)), za); // MPAS:540
zol0 = FDIV(FMUL(s.zol, z0t), za); // MPAS:541
if ((s.zol > 0x0.0p+0f)) { // MPAS:543
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:544
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:545
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:546
} else { // MPAS:547
if ((s.zol == 0x0.0p+0f)) { // MPAS:548
s.psih = 0x0.0p+0f; // MPAS:549
psih10 = 0x0.0p+0f; // MPAS:550
psih2 = 0x0.0p+0f; // MPAS:551
} else { // MPAS:552
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:553
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:554
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:555
} // MPAS:556
} // MPAS:557
psit = FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih); // MPAS:558
psit2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2); // MPAS:559
zolzz = FDIV(FMUL(s.zol, FADD(za, z0q)), za); // MPAS:561
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0q)), za); // MPAS:562
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0q)), za); // MPAS:563
zol0 = FDIV(FMUL(s.zol, z0q), za); // MPAS:564
if ((s.zol > 0x0.0p+0f)) { // MPAS:566
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:567
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:568
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:569
} else { // MPAS:570
if ((s.zol == 0x0.0p+0f)) { // MPAS:571
s.psih = 0x0.0p+0f; // MPAS:572
psih10 = 0x0.0p+0f; // MPAS:573
psih2 = 0x0.0p+0f; // MPAS:574
} else { // MPAS:575
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:576
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:577
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:578
} // MPAS:579
} // MPAS:580
psiq = FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih); // MPAS:582
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2); // MPAS:583
psiq10 = FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10); // MPAS:584
} // MPAS:585
if ((optional_mask & 1u) == 1u) { // MPAS:587
if (((isftcflx == 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:588
z0q = 0x1.a36e2e0000000p-14f; // MPAS:594
zolzz = FDIV(FMUL(s.zol, FADD(za, z0q)), za); // MPAS:598
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0q)), za); // MPAS:599
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0q)), za); // MPAS:600
zol0 = FDIV(FMUL(s.zol, z0q), za); // MPAS:601
if ((s.zol > 0x0.0p+0f)) { // MPAS:603
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:604
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:605
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:606
} else { // MPAS:607
if ((s.zol == 0x0.0p+0f)) { // MPAS:608
s.psih = 0x0.0p+0f; // MPAS:609
psih10 = 0x0.0p+0f; // MPAS:610
psih2 = 0x0.0p+0f; // MPAS:611
} else { // MPAS:612
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:613
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:614
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:615
} // MPAS:616
} // MPAS:617
psiq = FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih); // MPAS:619
psit = psiq; // MPAS:620
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2); // MPAS:621
psiq10 = FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10); // MPAS:622
psit2 = psiq2; // MPAS:623
} // MPAS:624
if (((isftcflx == 0x1.0000000000000p+1f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:625
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:631
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:633
gz0ozt = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.af6b3a0000000p-1f), 0x1.4000000000000p+2f)); // MPAS:634
z0t = FDIV(s.znt, mm5_exp(gz0ozt)); // MPAS:638
zolzz = FDIV(FMUL(s.zol, FADD(za, z0t)), za); // MPAS:640
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0t)), za); // MPAS:641
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0t)), za); // MPAS:642
zol0 = FDIV(FMUL(s.zol, z0t), za); // MPAS:643
if ((s.zol > 0x0.0p+0f)) { // MPAS:645
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:646
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:647
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:648
} else { // MPAS:649
if ((s.zol == 0x0.0p+0f)) { // MPAS:650
s.psih = 0x0.0p+0f; // MPAS:651
psih10 = 0x0.0p+0f; // MPAS:652
psih2 = 0x0.0p+0f; // MPAS:653
} else { // MPAS:654
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:655
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:656
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:657
} // MPAS:658
} // MPAS:659
psit = FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih); // MPAS:663
psit2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2); // MPAS:664
gz0ozq = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.8c97f00000000p-1f), 0x1.4000000000000p+2f)); // MPAS:666
z0q = FDIV(s.znt, mm5_exp(gz0ozq)); // MPAS:667
zolzz = FDIV(FMUL(s.zol, FADD(za, z0q)), za); // MPAS:669
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0q)), za); // MPAS:670
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0q)), za); // MPAS:671
zol0 = FDIV(FMUL(s.zol, z0q), za); // MPAS:672
if ((s.zol > 0x0.0p+0f)) { // MPAS:674
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:675
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:676
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:677
} else { // MPAS:678
if ((s.zol == 0x0.0p+0f)) { // MPAS:679
s.psih = 0x0.0p+0f; // MPAS:680
psih10 = 0x0.0p+0f; // MPAS:681
psih2 = 0x0.0p+0f; // MPAS:682
} else { // MPAS:683
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:684
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:685
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:686
} // MPAS:687
} // MPAS:688
psiq = FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih); // MPAS:690
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2); // MPAS:691
psiq10 = FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10); // MPAS:692
} // MPAS:696
} // MPAS:697
if ((optional_mask & 240u) == 240u) { // MPAS:698
s.ck = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psiq10)); // MPAS:699
s.cd = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psix10)); // MPAS:700
s.cka = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psiq)); // MPAS:701
s.cda = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psix)); // MPAS:702
} // MPAS:703
if ((optional_mask & 2u) == 2u) { // MPAS:704
if (((iz0tlnd >= 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) <= 0x0.0p+0f))) { // MPAS:705
zl = s.znt; // MPAS:706
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:708
restar = FDIV(FMUL(s.ust, zl), visc); // MPAS:709
if ((iz0tlnd == 0x1.0000000000000p+0f)) { // MPAS:713
czil = mm5_pow(0x1.4000000000000p+3f, FMUL(-0x1.99999a0000000p-2f, FDIV(zl, 0x1.1eb8520000000p-4f))); // MPAS:714
} else if ((iz0tlnd == 0x1.0000000000000p+1f)) { // MPAS:715
czil = 0x1.99999a0000000p-4f; // MPAS:716
} // MPAS:717
z0t = FDIV(s.znt, mm5_exp(FMUL(FMUL(czil, c.karman), FSQRT(restar)))); // MPAS:721
zolzz = FDIV(FMUL(s.zol, FADD(za, z0t)), za); // MPAS:723
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0t)), za); // MPAS:724
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0t)), za); // MPAS:725
zol0 = FDIV(FMUL(s.zol, z0t), za); // MPAS:726
if ((s.zol > 0x0.0p+0f)) { // MPAS:728
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:729
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:730
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:731
} else { // MPAS:732
if ((s.zol == 0x0.0p+0f)) { // MPAS:733
s.psih = 0x0.0p+0f; // MPAS:734
psih10 = 0x0.0p+0f; // MPAS:735
psih2 = 0x0.0p+0f; // MPAS:736
} else { // MPAS:737
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:738
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:739
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:740
} // MPAS:741
} // MPAS:742
psiq = FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih); // MPAS:744
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2); // MPAS:745
psit = psiq; // MPAS:746
psit2 = psiq2; // MPAS:747
} // MPAS:753
} // MPAS:754
s.ust = FADD(FMUL(0x1.0000000000000p-1f, s.ust), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), s.wspd), psix)); // MPAS:756
wspdi = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:758
if ((optional_mask & 8u) == 8u) { // MPAS:759
s.ustm = FADD(FMUL(0x1.0000000000000p-1f, s.ustm), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), wspdi), psix)); // MPAS:760
} // MPAS:761
s.u10 = FDIV(FMUL(s.ux, psix10), psix); // MPAS:763
s.v10 = FDIV(FMUL(s.vx, psix10), psix); // MPAS:764
s.th2 = FADD(thgb, FDIV(FMUL(dtg, psit2), psit)); // MPAS:765
s.q2 = FADD(s.qsfc, FDIV(FMUL(FSUB(qx, s.qsfc), psiq2), psiq)); // MPAS:766
s.t2 = FMUL(s.th2, mm5_pow(FDIV(s.psfcpa, c.p1000mb), c.rovcp)); // MPAS:767
if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:769
s.ust = fmaxf(s.ust, 0x1.0624de0000000p-10f); // MPAS:770
} // MPAS:771
s.mol = FDIV(FDIV(FMUL(c.karman, dtg), psit), 0x1.0000000000000p+0f); // MPAS:772
denomq = psiq; // MPAS:773
denomq2 = psiq2; // MPAS:774
denomt2 = psit2; // MPAS:775
s.fm = psix; // MPAS:776
s.fh = psit; // MPAS:777
L330:;
L335:;
if ((optional_mask & 4u) == 4u) { // MPAS:783
if (scm_force_flux) goto L350; // MPAS:784
} // MPAS:785
s.qfx = 0x0.0p+0f; // MPAS:787
s.hfx = 0x0.0p+0f; // MPAS:788
L350:;
if (!(isfflx)) goto L410; // MPAS:792
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:796
if (shalwater_z0) { // MPAS:800
s.znt = mm5_depth_dependent_z0(s.water_depth, s.znt, s.ust, tables); // MPAS:801
} else { // MPAS:802
s.znt = FADD(FDIV(FMUL(FMUL(0x1.2f1aa00000000p-6f, s.ust), s.ust), c.g), FDIV(0x1.baeb220000000p-20f, s.ust)); // MPAS:804
s.znt = fminf(s.znt, 0x1.758e220000000p-9f); // MPAS:806
} // MPAS:807
if ((optional_mask & 1u) == 1u) { // MPAS:813
if ((isftcflx != 0x0.0p+0f)) { // MPAS:814
zw = fminf(mm5_pow(FDIV(s.ust, 0x1.0f5c280000000p+0f), 0x1.3333340000000p-2f), 0x1.0000000000000p+0f); // MPAS:821
zn1 = FADD(FDIV(FMUL(FMUL(0x1.6872b00000000p-7f, s.ust), s.ust), c.g), 0x1.0ac1fa0000000p-16f); // MPAS:822
zn2 = FADD(FMUL(0x1.4000000000000p+3f, mm5_exp(FMUL(-0x1.3000000000000p+3f, mm5_pow(s.ust, -0x1.554c980000000p-2f)))), FDIV(0x1.baeb220000000p-20f, fmaxf(s.ust, 0x1.47ae140000000p-7f))); // MPAS:823
s.znt = FADD(FMUL(FSUB(0x1.0000000000000p+0f, zw), zn1), FMUL(zw, zn2)); // MPAS:825
s.znt = fminf(s.znt, 0x1.758e220000000p-9f); // MPAS:826
s.znt = fmaxf(s.znt, 0x1.10bafe0000000p-23f); // MPAS:827
} // MPAS:828
} // MPAS:829
zl = s.znt; // MPAS:830
} else { // MPAS:831
zl = 0x1.47ae140000000p-7f; // MPAS:832
} // MPAS:833
s.flqc = FDIV(FMUL(FMUL(FMUL(rhox, s.mavail), s.ust), c.karman), denomq); // MPAS:834
dtthx = fabsf(FSUB(thx, thgb)); // MPAS:837
if ((dtthx > 0x1.4f8b580000000p-17f)) { // MPAS:838
s.flhc = FDIV(FMUL(FMUL(FMUL(s.cpm, rhox), s.ust), s.mol), FSUB(thx, thgb)); // MPAS:839
} else { // MPAS:842
s.flhc = 0x0.0p+0f; // MPAS:843
} // MPAS:844
L360:;
if ((optional_mask & 4u) == 4u) { // MPAS:852
if (scm_force_flux) goto L405; // MPAS:853
} // MPAS:854
s.qfx = FMUL(s.flqc, FSUB(s.qsfc, qx)); // MPAS:857
s.lh = FMUL(c.xlv, s.qfx); // MPAS:859
L370:;
L390:;
if ((FSUB(s.xland, 0x1.8000000000000p+0f) > 0x0.0p+0f)) { // MPAS:866
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:867
} else if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:874
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:875
} // MPAS:877
L400:;
L405:;
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:883
zl = s.znt; // MPAS:884
} else { // MPAS:885
zl = 0x1.47ae140000000p-7f; // MPAS:886
} // MPAS:887
s.chs = FDIV(FMUL(s.ust, c.karman), denomq); // MPAS:891
s.cqs2 = FDIV(FMUL(s.ust, c.karman), denomq2); // MPAS:900
s.chs2 = FDIV(FMUL(s.ust, c.karman), denomt2); // MPAS:901
L410:;
}
// MPAS physics_wrf/module_sf_sfclay.F:265-946
__device__ void mm5_classic(MM5State &s, const MM5Constants &c, const float *tables, int isfflx, int isftcflx, int iz0tlnd, int scm_force_flux, int shalwater_z0, unsigned int optional_mask = 255) {
float pl, thcon, tvcon, e1, zl, tskv, dthvdz, dthvm, vconv, rzol, rzol2, rzol10, zol2, zol10, dtg, psix, dtthx, psix10, psit, psit2, psiq, psiq2, psiq10, fluxc, vsgd, z0q, visc, restar, czil, gz0ozq, gz0ozt, zw, zn1, zn2, zolzz, zol0, zl2, zl10, z0t, za, thvx, zqkl, zqklp1, thx, qx, psih2, psim2, psih10, psim10, denomq, denomq2, denomt2, wspdi, gz2oz0, gz10oz0, rhox, govrth, tgdsa, scr3, scr4, thgb, psfc, pq, pq2, pq10, czc;
int nzol, nzol2, nzol10;
psfc = FDIV(s.psfcpa, 0x1.f400000000000p+9f); // MPAS:393
tgdsa = s.tsk; // MPAS:399
thgb = FMUL(s.tsk, mm5_pow(FDIV(c.p1000mb, s.psfcpa), c.rovcp)); // MPAS:402
L5:;
L10:;
L26:;
pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:427
scr3 = s.t1d; // MPAS:428
thcon = mm5_pow(FDIV(FMUL(c.p1000mb, 0x1.0624de0000000p-10f), pl), c.rovcp); // MPAS:430
thx = FMUL(scr3, thcon); // MPAS:431
scr4 = scr3; // MPAS:432
thvx = thx; // MPAS:433
qx = 0x0.0p+0f; // MPAS:434
L30:;
s.qgh = 0x0.0p+0f; // MPAS:438
s.flhc = 0x0.0p+0f; // MPAS:439
s.flqc = 0x0.0p+0f; // MPAS:440
s.cpm = c.cp; // MPAS:441
qx = s.qv1d; // MPAS:446
tvcon = FADD(0x1.0000000000000p+0f, FMUL(c.ep1, qx)); // MPAS:447
thvx = FMUL(thx, tvcon); // MPAS:448
scr4 = FMUL(scr3, tvcon); // MPAS:449
L50:;
e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(tgdsa, c.svpt0)), FSUB(tgdsa, c.svp3)))); // MPAS:453
if (((s.xland > 0x1.8000000000000p+0f) || (s.qsfc <= 0x0.0p+0f))) s.qsfc = FDIV(FMUL(c.ep2, e1), FSUB(psfc, e1)); // MPAS:455
e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(s.t1d, c.svpt0)), FSUB(s.t1d, c.svp3)))); // MPAS:458
pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:459
s.qgh = FDIV(FMUL(c.ep2, e1), FSUB(pl, e1)); // MPAS:460
s.cpm = FMUL(c.cp, FADD(0x1.0000000000000p+0f, FMUL(0x1.99999a0000000p-1f, qx))); // MPAS:461
L60:;
L80:;
zqklp1 = 0x0.0p+0f; // MPAS:469
rhox = FDIV(FMUL(psfc, 0x1.f400000000000p+9f), FMUL(c.r, scr4)); // MPAS:470
L90:;
zqkl = FADD(s.dz8w1d, zqklp1); // MPAS:474
L110:;
za = FMUL(0x1.0000000000000p-1f, FADD(zqkl, zqklp1)); // MPAS:478
L120:;
govrth = FDIV(c.g, thx); // MPAS:482
L160:;
s.gz1oz0 = mm5_log(FDIV(za, s.znt)); // MPAS:489
gz2oz0 = mm5_log(FDIV(0x1.0000000000000p+1f, s.znt)); // MPAS:490
gz10oz0 = mm5_log(FDIV(0x1.4000000000000p+3f, s.znt)); // MPAS:491
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:492
zl = s.znt; // MPAS:493
} else { // MPAS:494
zl = 0x1.47ae140000000p-7f; // MPAS:495
} // MPAS:496
s.wspd = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:497
tskv = FMUL(thgb, FADD(0x1.0000000000000p+0f, FMUL(c.ep1, s.qsfc))); // MPAS:499
dthvdz = FSUB(thvx, tskv); // MPAS:500
if ((s.xland < 0x1.8000000000000p+0f)) { // MPAS:507
fluxc = fmaxf(FADD(FDIV(FDIV(s.hfx, rhox), c.cp), FDIV(FMUL(FMUL(c.ep1, tskv), s.qfx), rhox)), 0x0.0p+0f); // MPAS:508
vconv = FMUL(0x1.0000000000000p+0f, mm5_pow(FMUL(FMUL(FDIV(c.g, tgdsa), s.pblh), fluxc), 0x1.51eb860000000p-2f)); // MPAS:510
} else { // MPAS:511
if ((-(dthvdz) >= 0x0.0p+0f)) { // MPAS:512
dthvm = -(dthvdz); // MPAS:513
} else { // MPAS:514
dthvm = 0x0.0p+0f; // MPAS:515
} // MPAS:516
vconv = FSQRT(dthvm); // MPAS:519
} // MPAS:520
vsgd = FMUL(0x1.47ae140000000p-2f, mm5_pow(fmaxf(FSUB(FDIV(s.dx, 0x1.3880000000000p+12f), 0x1.0000000000000p+0f), 0x0.0p+0f), 0x1.51eb860000000p-2f)); // MPAS:522
s.wspd = FSQRT(FADD(FADD(FMUL(s.wspd, s.wspd), FMUL(vconv, vconv)), FMUL(vsgd, vsgd))); // MPAS:523
s.wspd = fmaxf(s.wspd, 0x1.99999a0000000p-4f); // MPAS:524
s.br = FDIV(FMUL(FMUL(govrth, za), dthvdz), FMUL(s.wspd, s.wspd)); // MPAS:525
if ((s.mol < 0x0.0p+0f)) s.br = fminf(s.br, 0x0.0p+0f); // MPAS:527
s.rmol = FMUL(FMUL(FMUL(-(govrth), dthvdz), za), c.karman); // MPAS:529
L260:;
if ((s.br < 0x0.0p+0f)) goto L310; // MPAS:562
if ((s.br < 0x1.99999a0000000p-3f)) goto L270; // MPAS:566
s.regime = 0x1.0000000000000p+0f; // MPAS:567
s.psim = FMUL(-0x1.4000000000000p+3f, s.gz1oz0); // MPAS:568
s.psim = fmaxf(s.psim, -0x1.4000000000000p+3f); // MPAS:570
s.psih = s.psim; // MPAS:571
psim10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.psim); // MPAS:572
psim10 = fmaxf(psim10, -0x1.4000000000000p+3f); // MPAS:573
psih10 = psim10; // MPAS:574
psim2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.psim); // MPAS:575
psim2 = fmaxf(psim2, -0x1.4000000000000p+3f); // MPAS:576
psih2 = psim2; // MPAS:577
if ((s.ust < 0x1.47ae140000000p-7f)) { // MPAS:580
s.rmol = FMUL(s.br, s.gz1oz0); // MPAS:581
} else { // MPAS:582
s.rmol = FDIV(FMUL(FMUL(FMUL(c.karman, govrth), za), s.mol), FMUL(s.ust, s.ust)); // MPAS:583
} // MPAS:584
s.rmol = fminf(s.rmol, 0x1.3ff7ce0000000p+3f); // MPAS:585
s.rmol = FDIV(s.rmol, za); // MPAS:586
goto L320; // MPAS:588
L270:;
if ((s.br == 0x0.0p+0f)) goto L280; // MPAS:592
s.regime = 0x1.0000000000000p+1f; // MPAS:593
s.psim = FDIV(FMUL(FMUL(-0x1.4000000000000p+2f, s.br), s.gz1oz0), FSUB(0x1.19999a0000000p+0f, FMUL(0x1.4000000000000p+2f, s.br))); // MPAS:594
s.psim = fmaxf(s.psim, -0x1.4000000000000p+3f); // MPAS:596
s.psih = s.psim; // MPAS:598
psim10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.psim); // MPAS:599
psim10 = fmaxf(psim10, -0x1.4000000000000p+3f); // MPAS:600
psih10 = psim10; // MPAS:601
psim2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.psim); // MPAS:602
psim2 = fmaxf(psim2, -0x1.4000000000000p+3f); // MPAS:603
psih2 = psim2; // MPAS:604
s.zol = FDIV(FMUL(s.br, s.gz1oz0), FSUB(0x1.0000a80000000p+0f, FMUL(0x1.4000000000000p+2f, s.br))); // MPAS:610
if ((s.zol > 0x1.0000000000000p-1f)) { // MPAS:612
s.zol = FADD(FMUL(FMUL(FADD(FMUL(0x1.e3d70a0000000p+0f, s.gz1oz0), 0x1.61999a0000000p+5f), s.br), s.br), FMUL(FSUB(FMUL(0x1.2e147a0000000p+0f, s.gz1oz0), 0x1.5eb8520000000p+0f), s.br)); // MPAS:616
s.zol = fminf(s.zol, 0x1.3ff7ce0000000p+3f); // MPAS:618
} // MPAS:619
s.rmol = FDIV(s.zol, za); // MPAS:622
goto L320; // MPAS:624
L280:;
s.regime = 0x1.8000000000000p+1f; // MPAS:628
s.psim = 0x0.0p+0f; // MPAS:629
s.psih = s.psim; // MPAS:630
psim10 = 0x0.0p+0f; // MPAS:631
psih10 = psim10; // MPAS:632
psim2 = 0x0.0p+0f; // MPAS:633
psih2 = psim2; // MPAS:634
if ((s.ust < 0x1.47ae140000000p-7f)) { // MPAS:637
s.zol = FMUL(s.br, s.gz1oz0); // MPAS:638
} else { // MPAS:639
s.zol = FDIV(FMUL(FMUL(FMUL(c.karman, govrth), za), s.mol), FMUL(s.ust, s.ust)); // MPAS:640
} // MPAS:641
s.rmol = FDIV(s.zol, za); // MPAS:643
goto L320; // MPAS:645
L310:;
s.regime = 0x1.0000000000000p+2f; // MPAS:650
if ((s.ust < 0x1.47ae140000000p-7f)) { // MPAS:651
s.zol = FMUL(s.br, s.gz1oz0); // MPAS:652
} else { // MPAS:653
s.zol = FDIV(FMUL(FMUL(FMUL(c.karman, govrth), za), s.mol), FMUL(s.ust, s.ust)); // MPAS:654
} // MPAS:655
zol10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.zol); // MPAS:656
zol2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.zol); // MPAS:657
s.zol = fminf(s.zol, 0x0.0p+0f); // MPAS:658
s.zol = fmaxf(s.zol, -0x1.3fff2e0000000p+3f); // MPAS:659
zol10 = fminf(zol10, 0x0.0p+0f); // MPAS:660
zol10 = fmaxf(zol10, -0x1.3fff2e0000000p+3f); // MPAS:661
zol2 = fminf(zol2, 0x0.0p+0f); // MPAS:662
zol2 = fmaxf(zol2, -0x1.3fff2e0000000p+3f); // MPAS:663
nzol = int(FMUL(-(s.zol), 0x1.9000000000000p+6f)); // MPAS:664
rzol = FSUB(FMUL(-(s.zol), 0x1.9000000000000p+6f), nzol); // MPAS:665
nzol10 = int(FMUL(-(zol10), 0x1.9000000000000p+6f)); // MPAS:666
rzol10 = FSUB(FMUL(-(zol10), 0x1.9000000000000p+6f), nzol10); // MPAS:667
nzol2 = int(FMUL(-(zol2), 0x1.9000000000000p+6f)); // MPAS:668
rzol2 = FSUB(FMUL(-(zol2), 0x1.9000000000000p+6f), nzol2); // MPAS:669
s.psim = FADD(tables[4*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[4*1001 + (int)(FADD(nzol, 0x1.0000000000000p+0f))], tables[4*1001 + (int)(nzol)]))); // MPAS:670
s.psih = FADD(tables[5*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[5*1001 + (int)(FADD(nzol, 0x1.0000000000000p+0f))], tables[5*1001 + (int)(nzol)]))); // MPAS:671
psim10 = FADD(tables[4*1001 + (int)(nzol10)], FMUL(rzol10, FSUB(tables[4*1001 + (int)(FADD(nzol10, 0x1.0000000000000p+0f))], tables[4*1001 + (int)(nzol10)]))); // MPAS:672
psih10 = FADD(tables[5*1001 + (int)(nzol10)], FMUL(rzol10, FSUB(tables[5*1001 + (int)(FADD(nzol10, 0x1.0000000000000p+0f))], tables[5*1001 + (int)(nzol10)]))); // MPAS:673
psim2 = FADD(tables[4*1001 + (int)(nzol2)], FMUL(rzol2, FSUB(tables[4*1001 + (int)(FADD(nzol2, 0x1.0000000000000p+0f))], tables[4*1001 + (int)(nzol2)]))); // MPAS:674
psih2 = FADD(tables[5*1001 + (int)(nzol2)], FMUL(rzol2, FSUB(tables[5*1001 + (int)(FADD(nzol2, 0x1.0000000000000p+0f))], tables[5*1001 + (int)(nzol2)]))); // MPAS:675
s.psih = fminf(s.psih, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:681
s.psim = fminf(s.psim, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:682
psih2 = fminf(psih2, FMUL(0x1.cccccc0000000p-1f, gz2oz0)); // MPAS:683
psim10 = fminf(psim10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:684
psih10 = fminf(psih10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:686
s.rmol = FDIV(s.zol, za); // MPAS:688
L320:;
dtg = FSUB(thx, thgb); // MPAS:696
psix = FSUB(s.gz1oz0, s.psim); // MPAS:697
psix10 = FSUB(gz10oz0, psim10); // MPAS:698
psit = fmaxf(FSUB(s.gz1oz0, s.psih), 0x1.0000000000000p+1f); // MPAS:701
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:703
zl = s.znt; // MPAS:704
} else { // MPAS:705
zl = 0x1.47ae140000000p-7f; // MPAS:706
} // MPAS:707
psiq = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), za), 0x1.92a7380000000p-16f), FDIV(za, zl))), s.psih); // MPAS:708
psit2 = FSUB(gz2oz0, psih2); // MPAS:709
psiq2 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.0000000000000p+1f), 0x1.92a7380000000p-16f), FDIV(0x1.0000000000000p+1f, zl))), psih2); // MPAS:710
psiq10 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.4000000000000p+3f), 0x1.92a7380000000p-16f), FDIV(0x1.4000000000000p+3f, zl))), psih10); // MPAS:712
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:716
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:717
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:720
z0t = FMUL(0x1.cd5f9a0000000p-15f, mm5_pow(restar, -0x1.3333340000000p-1f)); // MPAS:721
z0t = fminf(z0t, 0x1.a36e2e0000000p-14f); // MPAS:722
z0t = fmaxf(z0t, 0x1.12e0be0000000p-29f); // MPAS:723
z0q = z0t; // MPAS:724
psiq = fmaxf(FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih), 0x1.0000000000000p+1f); // MPAS:726
psit = fmaxf(FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih), 0x1.0000000000000p+1f); // MPAS:727
psiq2 = fmaxf(FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2), 0x1.0000000000000p+1f); // MPAS:728
psit2 = fmaxf(FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2), 0x1.0000000000000p+1f); // MPAS:729
psiq10 = fmaxf(FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10), 0x1.0000000000000p+1f); // MPAS:730
} // MPAS:731
if ((optional_mask & 1u) == 1u) { // MPAS:733
if (((isftcflx == 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:734
z0q = 0x1.a36e2e0000000p-14f; // MPAS:740
psiq = FSUB(mm5_log(FDIV(za, z0q)), s.psih); // MPAS:741
psit = psiq; // MPAS:742
psiq2 = FSUB(mm5_log(FDIV(0x1.0000000000000p+1f, z0q)), psih2); // MPAS:743
psiq10 = FSUB(mm5_log(FDIV(0x1.4000000000000p+3f, z0q)), psih10); // MPAS:744
psit2 = psiq2; // MPAS:745
} // MPAS:746
if (((isftcflx == 0x1.0000000000000p+1f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:747
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:753
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:755
gz0ozt = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.af6b3a0000000p-1f), 0x1.4000000000000p+2f)); // MPAS:756
gz0ozq = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.8c97f00000000p-1f), 0x1.4000000000000p+2f)); // MPAS:757
psit = FADD(FSUB(s.gz1oz0, s.psih), gz0ozt); // MPAS:758
psiq = FADD(FSUB(s.gz1oz0, s.psih), gz0ozq); // MPAS:759
psit2 = FADD(FSUB(gz2oz0, psih2), gz0ozt); // MPAS:760
psiq2 = FADD(FSUB(gz2oz0, psih2), gz0ozq); // MPAS:761
psiq10 = FADD(FSUB(gz10oz0, s.psih), gz0ozq); // MPAS:762
} // MPAS:763
} // MPAS:764
if ((optional_mask & 240u) == 240u) { // MPAS:765
s.ck = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psiq10)); // MPAS:766
s.cd = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psix10)); // MPAS:767
s.cka = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psiq)); // MPAS:768
s.cda = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psix)); // MPAS:769
} // MPAS:770
if ((optional_mask & 2u) == 2u) { // MPAS:771
if (((iz0tlnd >= 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) <= 0x0.0p+0f))) { // MPAS:772
zl = s.znt; // MPAS:773
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:775
restar = FDIV(FMUL(s.ust, zl), visc); // MPAS:776
if ((iz0tlnd == 0x1.0000000000000p+0f)) { // MPAS:780
czil = mm5_pow(0x1.4000000000000p+3f, FMUL(-0x1.99999a0000000p-2f, FDIV(zl, 0x1.1eb8520000000p-4f))); // MPAS:781
} else if ((iz0tlnd == 0x1.0000000000000p+1f)) { // MPAS:782
czil = 0x1.99999a0000000p-4f; // MPAS:783
} // MPAS:784
psit = FADD(FSUB(s.gz1oz0, s.psih), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:786
psiq = FADD(FSUB(s.gz1oz0, s.psih), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:787
psit2 = FADD(FSUB(gz2oz0, psih2), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:788
psiq2 = FADD(FSUB(gz2oz0, psih2), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:789
} // MPAS:791
} // MPAS:792
s.ust = FADD(FMUL(0x1.0000000000000p-1f, s.ust), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), s.wspd), psix)); // MPAS:794
wspdi = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:796
if ((optional_mask & 8u) == 8u) { // MPAS:797
s.ustm = FADD(FMUL(0x1.0000000000000p-1f, s.ustm), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), wspdi), psix)); // MPAS:798
} // MPAS:799
s.u10 = FDIV(FMUL(s.ux, psix10), psix); // MPAS:800
s.v10 = FDIV(FMUL(s.vx, psix10), psix); // MPAS:801
s.th2 = FADD(thgb, FDIV(FMUL(dtg, psit2), psit)); // MPAS:802
s.q2 = FADD(s.qsfc, FDIV(FMUL(FSUB(qx, s.qsfc), psiq2), psiq)); // MPAS:803
s.t2 = FMUL(s.th2, mm5_pow(FDIV(s.psfcpa, c.p1000mb), c.rovcp)); // MPAS:805
if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:812
s.ust = fmaxf(s.ust, 0x1.99999a0000000p-4f); // MPAS:813
} // MPAS:814
s.mol = FDIV(FDIV(FMUL(c.karman, dtg), psit), 0x1.0000000000000p+0f); // MPAS:815
denomq = psiq; // MPAS:816
denomq2 = psiq2; // MPAS:817
denomt2 = psit2; // MPAS:818
s.fm = psix; // MPAS:819
s.fh = psit; // MPAS:820
L330:;
L335:;
if ((optional_mask & 4u) == 4u) { // MPAS:826
if ((scm_force_flux == 0x1.0000000000000p+0f)) goto L350; // MPAS:827
} // MPAS:828
s.qfx = 0x0.0p+0f; // MPAS:830
s.hfx = 0x0.0p+0f; // MPAS:831
L350:;
if ((isfflx == 0x0.0p+0f)) goto L410; // MPAS:835
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:840
s.znt = FADD(FDIV(FMUL(FMUL(0x1.2f1aa00000000p-6f, s.ust), s.ust), c.g), FDIV(0x1.baeb220000000p-20f, s.ust)); // MPAS:843
s.znt = fminf(s.znt, 0x1.758e220000000p-9f); // MPAS:845
if ((optional_mask & 1u) == 1u) { // MPAS:851
if ((isftcflx != 0x0.0p+0f)) { // MPAS:852
zw = fminf(mm5_pow(FDIV(s.ust, 0x1.0f5c280000000p+0f), 0x1.3333340000000p-2f), 0x1.0000000000000p+0f); // MPAS:859
zn1 = FADD(FDIV(FMUL(FMUL(0x1.6872b00000000p-7f, s.ust), s.ust), c.g), 0x1.0ac1fa0000000p-16f); // MPAS:860
zn2 = FADD(FMUL(0x1.4000000000000p+3f, mm5_exp(FMUL(-0x1.3000000000000p+3f, mm5_pow(s.ust, -0x1.554c980000000p-2f)))), FDIV(0x1.baeb220000000p-20f, fmaxf(s.ust, 0x1.47ae140000000p-7f))); // MPAS:861
s.znt = FADD(FMUL(FSUB(0x1.0000000000000p+0f, zw), zn1), FMUL(zw, zn2)); // MPAS:863
s.znt = fminf(s.znt, 0x1.758e220000000p-9f); // MPAS:864
s.znt = fmaxf(s.znt, 0x1.10bafe0000000p-23f); // MPAS:865
} // MPAS:866
} // MPAS:867
zl = s.znt; // MPAS:868
} else { // MPAS:869
zl = 0x1.47ae140000000p-7f; // MPAS:870
} // MPAS:871
s.flqc = FDIV(FMUL(FMUL(FMUL(rhox, s.mavail), s.ust), c.karman), denomq); // MPAS:872
dtthx = fabsf(FSUB(thx, thgb)); // MPAS:875
if ((dtthx > 0x1.4f8b580000000p-17f)) { // MPAS:876
s.flhc = FDIV(FMUL(FMUL(FMUL(s.cpm, rhox), s.ust), s.mol), FSUB(thx, thgb)); // MPAS:877
} else { // MPAS:880
s.flhc = 0x0.0p+0f; // MPAS:881
} // MPAS:882
L360:;
if ((optional_mask & 4u) == 4u) { // MPAS:889
if ((scm_force_flux == 0x1.0000000000000p+0f)) goto L405; // MPAS:890
} // MPAS:891
s.qfx = FMUL(s.flqc, FSUB(s.qsfc, qx)); // MPAS:894
s.qfx = fmaxf(s.qfx, 0x0.0p+0f); // MPAS:895
s.lh = FMUL(c.xlv, s.qfx); // MPAS:896
L370:;
L390:;
if ((FSUB(s.xland, 0x1.8000000000000p+0f) > 0x0.0p+0f)) { // MPAS:903
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:904
} else if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:911
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:912
s.hfx = fmaxf(s.hfx, -0x1.f400000000000p+7f); // MPAS:913
} // MPAS:914
L400:;
L405:;
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:920
zl = s.znt; // MPAS:921
} else { // MPAS:922
zl = 0x1.47ae140000000p-7f; // MPAS:923
} // MPAS:924
s.chs = FDIV(FMUL(s.ust, c.karman), denomq); // MPAS:925
s.cqs2 = FDIV(FMUL(s.ust, c.karman), denomq2); // MPAS:930
s.chs2 = FDIV(FMUL(s.ust, c.karman), denomt2); // MPAS:931
L410:;
}
// MPAS physics_mmm/sf_sfclayrev.F90:27-55; physics_wrf/module_sf_sfclay.F:949-964
extern "C" __global__ void mm5_init_tables(float *tables) {
int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>1000) return;
float zolf=FMUL(float(n), 0.01f);
tables[n]=mm5_psim_stable_full(zolf,tables);
tables[1001+n]=mm5_psih_stable_full(zolf,tables);
zolf=FMUL(-float(n), 0.01f);
tables[2002+n]=mm5_psim_unstable_full(zolf,tables);
tables[3003+n]=mm5_psih_unstable_full(zolf,tables);
float x=mm5_pow(FSUB(1.f, FMUL(16.f,zolf)), 0.25f);
tables[4004+n]=FADD(FSUB(FADD(FMUL(2.f,mm5_log(FMUL(0.5f,FADD(1.f,x)))),mm5_log(FMUL(0.5f,FADD(1.f,FMUL(x,x))))),FMUL(2.f,mm5_atan(x))), 1.57079637050628662109375f);
float y=mm5_pow(FSUB(1.f,FMUL(16.f,zolf)),0.5f);
tables[5005+n]=FMUL(2.f, mm5_log(FMUL(0.5f, FADD(1.f,y))));
}
// MPAS physics_mmm/sf_sfclayrev.F90:79-920; physics_wrf/module_sf_sfclay.F:265-946.
extern "C" __global__ void mm5_columns(float *a, const float *constants, const float *tables, int n, int revised, int isfflx, int isftcflx, int iz0tlnd, int scm_force_flux, int shalwater_z0, unsigned int optional_mask = 255) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
MM5State s; MM5Constants c;
s.znt=a[0*n+i];
s.ust=a[1*n+i];
s.ustm=a[2*n+i];
s.mol=a[3*n+i];
s.hfx=a[4*n+i];
s.qfx=a[5*n+i];
s.qsfc=a[6*n+i];
s.zol=a[7*n+i];
s.regime=a[8*n+i];
s.psim=a[9*n+i];
s.psih=a[10*n+i];
s.fm=a[11*n+i];
s.fh=a[12*n+i];
s.lh=a[13*n+i];
s.u10=a[14*n+i];
s.v10=a[15*n+i];
s.th2=a[16*n+i];
s.t2=a[17*n+i];
s.q2=a[18*n+i];
s.chs=a[19*n+i];
s.chs2=a[20*n+i];
s.cqs2=a[21*n+i];
s.flhc=a[22*n+i];
s.flqc=a[23*n+i];
s.qgh=a[24*n+i];
s.rmol=a[25*n+i];
s.wspd=a[26*n+i];
s.br=a[27*n+i];
s.gz1oz0=a[28*n+i];
s.cpm=a[29*n+i];
s.ck=a[30*n+i];
s.cka=a[31*n+i];
s.cd=a[32*n+i];
s.cda=a[33*n+i];
s.ux=a[34*n+i];
s.vx=a[35*n+i];
s.t1d=a[36*n+i];
s.qv1d=a[37*n+i];
s.p1d=a[38*n+i];
s.dz8w1d=a[39*n+i];
s.psfcpa=a[40*n+i];
s.tsk=a[41*n+i];
s.pblh=a[42*n+i];
s.mavail=a[43*n+i];
s.xland=a[44*n+i];
s.lakemask=a[45*n+i];
s.dx=a[46*n+i];
s.water_depth=a[47*n+i];
c.cp=constants[0];
c.g=constants[1];
c.rovcp=constants[2];
c.r=constants[3];
c.xlv=constants[4];
c.svp1=constants[5];
c.svp2=constants[6];
c.svp3=constants[7];
c.svpt0=constants[8];
c.ep1=constants[9];
c.ep2=constants[10];
c.karman=constants[11];
c.p1000mb=constants[12];
if(revised) mm5_revised(s,c,tables,isfflx,isftcflx,iz0tlnd,scm_force_flux,shalwater_z0,optional_mask);
else mm5_classic(s,c,tables,isfflx,isftcflx,iz0tlnd,scm_force_flux,shalwater_z0,optional_mask);
a[0*n+i]=s.znt;
a[1*n+i]=s.ust;
a[2*n+i]=s.ustm;
a[3*n+i]=s.mol;
a[4*n+i]=s.hfx;
a[5*n+i]=s.qfx;
a[6*n+i]=s.qsfc;
a[7*n+i]=s.zol;
a[8*n+i]=s.regime;
a[9*n+i]=s.psim;
a[10*n+i]=s.psih;
a[11*n+i]=s.fm;
a[12*n+i]=s.fh;
a[13*n+i]=s.lh;
a[14*n+i]=s.u10;
a[15*n+i]=s.v10;
a[16*n+i]=s.th2;
a[17*n+i]=s.t2;
a[18*n+i]=s.q2;
a[19*n+i]=s.chs;
a[20*n+i]=s.chs2;
a[21*n+i]=s.cqs2;
a[22*n+i]=s.flhc;
a[23*n+i]=s.flqc;
a[24*n+i]=s.qgh;
a[25*n+i]=s.rmol;
a[26*n+i]=s.wspd;
a[27*n+i]=s.br;
a[28*n+i]=s.gz1oz0;
a[29*n+i]=s.cpm;
a[30*n+i]=s.ck;
a[31*n+i]=s.cka;
a[32*n+i]=s.cd;
a[33*n+i]=s.cda;
}

// MPAS mpas_atmphys_driver_sfclayer.F:429-534,554-562 sfclayer_from_MPAS
extern "C" __global__ void mm5_from_mpas(float *a, float *sea, const float *meshDensity, const float *sst, const float *xice, int n, float len_disp, float xice_threshold, int frac) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
a[46*n+i]=FDIV(len_disp, mm5_pow(meshDensity[i],0.25f));
a[18*n+i]=0.0f;
a[17*n+i]=0.0f;
a[16*n+i]=0.0f;
a[14*n+i]=0.0f;
a[15*n+i]=0.0f;
a[32*n+i]=0.0f;
a[33*n+i]=0.0f;
a[30*n+i]=0.0f;
a[31*n+i]=0.0f;
a[45*n+i]=0.0f;
a[47*n+i]=0.0f;
if(!frac) return;
for(int j=0;j<48;j++) sea[j*n+i]=a[j*n+i];
sea[6*n+i]=0.f;
if(xice[i]>=xice_threshold && xice[i]<=1.0f) {
sea[44*n+i]=2.f;
sea[43*n+i]=1.f;
sea[0*n+i]=0.0001f;
sea[41*n+i]=fmaxf(sst[i],271.4f);
}
}
// MPAS mpas_atmphys_driver_sfclayer.F:693-771 sfclayer_to_MPAS
extern "C" __global__ void mm5_to_mpas(float *a, const float *sea, const float *xice, const float *regime_hold, int n, float xice_threshold) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
float ice=xice[i]; if(!(ice>=xice_threshold && ice<=1.f)) return;
float ocean=FSUB(1.f,ice);
a[27*n+i]=FADD(FMUL(a[27*n+i],ice),FMUL(ocean,sea[27*n+i]));
a[22*n+i]=FADD(FMUL(a[22*n+i],ice),FMUL(ocean,sea[22*n+i]));
a[23*n+i]=FADD(FMUL(a[23*n+i],ice),FMUL(ocean,sea[23*n+i]));
a[28*n+i]=FADD(FMUL(a[28*n+i],ice),FMUL(ocean,sea[28*n+i]));
a[3*n+i]=FADD(FMUL(a[3*n+i],ice),FMUL(ocean,sea[3*n+i]));
a[10*n+i]=FADD(FMUL(a[10*n+i],ice),FMUL(ocean,sea[10*n+i]));
a[9*n+i]=FADD(FMUL(a[9*n+i],ice),FMUL(ocean,sea[9*n+i]));
a[25*n+i]=FADD(FMUL(a[25*n+i],ice),FMUL(ocean,sea[25*n+i]));
a[1*n+i]=FADD(FMUL(a[1*n+i],ice),FMUL(ocean,sea[1*n+i]));
a[26*n+i]=FADD(FMUL(a[26*n+i],ice),FMUL(ocean,sea[26*n+i]));
a[7*n+i]=FADD(FMUL(a[7*n+i],ice),FMUL(ocean,sea[7*n+i]));
a[18*n+i]=FADD(FMUL(a[18*n+i],ice),FMUL(ocean,sea[18*n+i]));
a[17*n+i]=FADD(FMUL(a[17*n+i],ice),FMUL(ocean,sea[17*n+i]));
a[16*n+i]=FADD(FMUL(a[16*n+i],ice),FMUL(ocean,sea[16*n+i]));
a[14*n+i]=FADD(FMUL(a[14*n+i],ice),FMUL(ocean,sea[14*n+i]));
a[15*n+i]=FADD(FMUL(a[15*n+i],ice),FMUL(ocean,sea[15*n+i]));
a[32*n+i]=FADD(FMUL(a[32*n+i],ice),FMUL(ocean,sea[32*n+i]));
a[33*n+i]=FADD(FMUL(a[33*n+i],ice),FMUL(ocean,sea[33*n+i]));
a[30*n+i]=FADD(FMUL(a[30*n+i],ice),FMUL(ocean,sea[30*n+i]));
a[31*n+i]=FADD(FMUL(a[31*n+i],ice),FMUL(ocean,sea[31*n+i]));
a[2*n+i]=FADD(FMUL(a[2*n+i],ice),FMUL(ocean,sea[2*n+i]));
a[12*n+i]=FADD(FMUL(a[12*n+i],ice),FMUL(ocean,sea[12*n+i]));
a[11*n+i]=FADD(FMUL(a[11*n+i],ice),FMUL(ocean,sea[11*n+i]));
if(ice>=0.5f) a[8*n+i]=regime_hold[i];
}
// MPAS mpas_atmphys_sfc_diagnostics.F:79-91,101-116 and
// physics_wrf/module_sf_sfcdiags.F:47-64: GNU scalar arithmetic preserves
// an input NaN's payload and quiets signaling NaNs. CUDA arithmetic instead
// canonicalizes NaNs. Retain the source operand word; finite arithmetic
// continues to use the shared rounded primitives, without a tolerance.
__device__ float mm5_diag_quiet(float a) {
return __uint_as_float(__float_as_uint(a) | 0x00400000u);
}
// MPAS physics_wrf/module_sf_sfcdiags.F:47-64 scalar product semantics.
__device__ float mm5_diag_mul(float a, float b) {
if(a!=a) return mm5_diag_quiet(a);
if(b!=b) return mm5_diag_quiet(b);
return FMUL(a,b);
}
// MPAS physics_wrf/module_sf_sfcdiags.F:47-64 scalar division semantics.
__device__ float mm5_diag_div(float a, float b) {
if(a!=a) return mm5_diag_quiet(a);
if(b!=b) return mm5_diag_quiet(b);
return FDIV(a,b);
}
// MPAS physics_wrf/module_sf_sfcdiags.F:47-64 scalar subtraction semantics.
__device__ float mm5_diag_sub(float a, float b) {
if(a!=a) return mm5_diag_quiet(a);
if(b!=b) return mm5_diag_quiet(b);
return FSUB(a,b);
}
// MPAS mpas_atmphys_sfc_diagnostics.F:27-123 atmphys_sfc_diagnostics
// The default LSM arm performs no stores, exactly as MPAS's SELECT CASE.
extern "C" __global__ void mm5_sfc_diagnostics(float *a, const float *xice, const float *q2mxy, const float *t2mxy, const float *constants, int n, int lsm, float xice_threshold) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n || lsm==0) return;
MM5Constants c;
c.cp=constants[0];
c.g=constants[1];
c.rovcp=constants[2];
c.r=constants[3];
c.xlv=constants[4];
c.svp1=constants[5];
c.svp2=constants[6];
c.svp3=constants[7];
c.svpt0=constants[8];
c.ep1=constants[9];
c.ep2=constants[10];
c.karman=constants[11];
c.p1000mb=constants[12];
float psfcpa=a[40*n+i];
float tsk=a[41*n+i];
float xland=a[44*n+i];
float qsfc=a[6*n+i];
float qfx=a[5*n+i];
float hfx=a[4*n+i];
float cqs2=a[21*n+i];
float chs2=a[20*n+i];
float rho=mm5_diag_div(psfcpa,mm5_diag_mul(c.r,tsk));
float q2,t2;
if(lsm==1 || (mm5_diag_sub(xland,1.5f)>0.f || (mm5_diag_sub(xland,1.5f)<=0.f && xice[i]>=xice_threshold))) {
q2=cqs2<1.e-5f ? qsfc : mm5_diag_sub(qsfc,mm5_diag_div(qfx,mm5_diag_mul(rho,cqs2)));
t2=chs2<1.e-5f ? tsk : mm5_diag_sub(tsk,mm5_diag_div(hfx,mm5_diag_mul(mm5_diag_mul(rho,c.cp),chs2)));
} else { q2=q2mxy[i]; t2=t2mxy[i]; }
a[18*n+i]=q2;
a[17*n+i]=t2;
a[16*n+i]=mm5_diag_mul(t2,mm5_pow(mm5_diag_div(c.p1000mb,psfcpa),c.rovcp));
}
// MPAS physics_wrf/module_sf_sfcdiags.F:7-69 SFCDIAGS
extern "C" __global__ void mm5_wrf_sfcdiags(float *a, const float *constants, int n, int ua_phys) {
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
float psfcpa=a[40*n+i];
float tsk=a[41*n+i];
float qsfc=a[6*n+i];
float qfx=a[5*n+i];
float hfx=a[4*n+i];
float cqs2=a[21*n+i];
float chs2=a[20*n+i];
float chs=a[19*n+i];
float qv1d=a[37*n+i];
float t1d=a[36*n+i];
float cp=constants[0],r=constants[3],rovcp=constants[2];
float rho=mm5_diag_div(psfcpa,mm5_diag_mul(r,tsk));
float q2=qsfc; if(!(cqs2<1.e-5f)) q2=ua_phys ? mm5_diag_sub(qsfc,mm5_diag_mul(mm5_diag_div(chs,cqs2),mm5_diag_sub(qsfc,qv1d))) : mm5_diag_sub(qsfc,mm5_diag_div(qfx,mm5_diag_mul(rho,cqs2)));
float t2=tsk; if(!(chs2<1.e-5f)) t2=ua_phys ? mm5_diag_sub(tsk,mm5_diag_mul(mm5_diag_div(chs,chs2),mm5_diag_sub(tsk,t1d))) : mm5_diag_sub(tsk,mm5_diag_div(hfx,mm5_diag_mul(mm5_diag_mul(rho,cp),chs2)));
a[18*n+i]=q2;
a[17*n+i]=t2;
a[16*n+i]=mm5_diag_mul(t2,mm5_pow(mm5_diag_div(1.e5f,psfcpa),rovcp));
}

"""


# RP3 referee: correct/mpas-rad-pbl-lsm-seaice/mpas-patched/
# correct_referee/mpas_atmphys_lsm_shared.F lines 62-105 and
# correct_referee/mpas_atmphys_driver_sfclayer.F lines 535-546 (native build 42d7e5f2).
# Preserve the stored blend for both low-fraction tests, as the referee does.
CORRECT_MODE_SOURCE = {
    'RP3': (('sea[41*n+i]=fmaxf(sst[i],271.4f);',
             'sea[41*n+i]=fmaxf(sst[i],271.4f);\n   float tsk_blend=a[41*n+i];\n   float tsk_ice=FDIV(FSUB(tsk_blend,FMUL(FSUB(1.f,xice[i]),sea[41*n+i])),xice[i]);\n   tsk_ice=fmaxf(tsk_ice,221.4f);\n   if(xice[i]<0.2f && tsk_blend<253.15f) tsk_ice=253.15f;\n   if(xice[i]<0.1f && tsk_blend<263.15f) tsk_ice=263.15f;\n   a[41*n+i]=tsk_ice;', 1),),
}


def _cuda_text(corrections=()):
    """MPAS mpas_atmphys_driver_sfclayer.F:428-546; RP3 referee cited above."""
    from .driver_v842 import correct_source
    return correct_source(CUDA_SOURCE, CORRECT_MODE_SOURCE, corrections, __name__)


def _shared_math_source() -> str:
    """Reuse the pinned engine's shared glibc header, without a private libm.

    MPAS physics_mmm/sf_sfclayrev.F90:79-920 requires these column functions.
    libm_identity.py/rkind_libm.py at the lane base expose host functions;
    the shared device header is still supplied by the admitted engine.
    """
    from woof.core import kernels
    directory = Path(kernels.__file__).parent
    return ('typedef float real;\n' +
            (directory / 'glibc_flt32.cuh').read_text() + '\n' +
            (directory / 'glibc_trig_flt32.cuh').read_text() + '\n' +
            '#define mm5_log gfk_log\n#define mm5_exp gfk_exp\n'
            '#define mm5_pow gfk_pow\n#define mm5_atan glibc_atanf\n')


def _read_cuda_artifact(path, architecture, corrections=()):
    """Validate the compiled source for physics_mmm/sf_sfclayrev.F90:27-55 tables.

    Reject stale source/options/CUBIN: otherwise a replay could silently execute
    a different kernel from the production module whose hash it records.
    """
    import hashlib
    import json
    path = Path(path)
    seal = json.loads(path.read_text())
    source = _shared_math_source() + _cuda_text(corrections)
    expected = (*_hex_numerics.options(COMPILE_OPTIONS), '--gpu-architecture=sm_' + architecture)
    if (seal.get('schema') != 'mpas.mm5-cuda-artifact/v1'
            or seal.get('cuda_source_sha256') != hashlib.sha256(source.encode()).hexdigest()
            or tuple(seal.get('options', ())) != expected):
        raise ValueError('MM5 compiled artifact source/options do not match this module/device')
    filename = seal.get('cubin_file', '')
    if not filename or Path(filename).name != filename:
        raise ValueError('MM5 compiled artifact CUBIN filename is invalid')
    cubin = path.parent / filename
    if hashlib.sha256(cubin.read_bytes()).hexdigest() != seal.get('cubin_sha256'):
        raise ValueError('MM5 compiled artifact CUBIN hash changed')
    return cubin


def _nvrtc_cubin(program):
    """Retrieve complete kernels for physics_mmm/sf_sfclayrev.F90:27-55.

    CuPy's NVRTC wrapper omits the last binary byte. Use the size/get API
    from the exact loaded NVRTC library; a truncated ELF cannot be loaded.
    """
    import ctypes as ct
    loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if '/libnvrtc.so' in line}
    if len(loaded) != 1:
        raise RuntimeError('Cannot identify the exact NVRTC library used for this program')
    library = ct.CDLL(loaded.pop())
    library.nvrtcGetCUBINSize.argtypes = (ct.c_void_p, ct.POINTER(ct.c_size_t))
    library.nvrtcGetCUBINSize.restype = ct.c_int
    library.nvrtcGetCUBIN.argtypes = (ct.c_void_p, ct.c_void_p)
    library.nvrtcGetCUBIN.restype = ct.c_int
    size = ct.c_size_t()
    status = library.nvrtcGetCUBINSize(program, ct.byref(size))
    if status or not size.value:
        raise RuntimeError('nvrtcGetCUBINSize failed: ' + str(status))
    buffer = ct.create_string_buffer(size.value)
    status = library.nvrtcGetCUBIN(program, buffer)
    if status:
        raise RuntimeError('nvrtcGetCUBIN failed: ' + str(status))
    return buffer.raw


@lru_cache(maxsize=None)
def _kernels(device: int, corrections=()):
    """Build column kernels/tables for physics_mmm/sf_sfclayrev.F90:27-55."""
    import cupy as cp
    import os
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        artifact = os.environ.get('MPAS_MM5_CORRECT_CUDA_ARTIFACT' if corrections
                                  else 'MPAS_MM5_CUDA_ARTIFACT')
        if artifact:
            cubin = _read_cuda_artifact(artifact, cp.cuda.Device(device).compute_capability, corrections)
            module.load_file(str(cubin))
        else:
            from ..cuda_backend.aot import direct_image
            # RawModule's cache compiler appends -ftz=true after caller flags.
            # Compile directly, as the shared subnormal defect witness requires.
            architecture = cp.cuda.Device(device).compute_capability
            module.load(direct_image(_shared_math_source() + _cuda_text(corrections),
                                    _hex_numerics.options(COMPILE_OPTIONS), architecture,
                                    filename='sfclay_mm5_v842.cu'))
        table = cp.empty(6006, dtype=cp.float32)
        module.get_function('mm5_init_tables')((8,), (128,), (table,))
    return module, table


def launch_columns(packed, constants, *, scheme: str, isfflx: int = 1,
                   isftcflx: int = 0, iz0tlnd: int = 0,
                   scm_force_flux: int = 0, shalwater_z0: int = 0,
                   optional_mask: int = ALL_OPTIONAL) -> None:
    """In-place column call with the argument values MPAS supplies.

    MPAS physics_mmm/sf_sfclayrev.F90:79-919;
    physics_wrf/module_sf_sfclay.F:265-946. Revised pre_run and the WRF
    wrappers select the lowest model level; the packed input rows represent
    that exact interface. ``optional_mask`` uses OPTIONAL_ARGUMENTS order;
    the MPAS driver supplies all eight. Absent argument rows retain their
    input words. No atmosphere, flux or previous state is inferred.
    """
    if not 0 <= optional_mask <= ALL_OPTIONAL:
        raise ValueError('optional_mask must contain only the eight source argument bits')
    # MPAS physics_mmm/sf_sfclayrev.F90:713-721 and
    # physics_wrf/module_sf_sfclay.F:781-789 leave CZIL undefined for >2.
    # Standalone native witnesses record history-dependent words in those
    # source alternatives. Keep executing, as native does; the actual MPAS
    # driver supplies IZ0TLND=0 and never reaches the unassigned value.
    import cupy as cp
    import numpy as np
    if scheme not in SCHEMES:
        raise ValueError(f'unknown MM5 MPAS scheme: {scheme}')
    if (not isinstance(packed, cp.ndarray) or packed.dtype != cp.float32 or
            packed.ndim != 2 or packed.shape[0] != len(FIELDS) or
            not packed.flags.c_contiguous):
        raise ValueError('packed must be a contiguous float32 device array (48, nCells)')
    if (not isinstance(constants, cp.ndarray) or constants.dtype != cp.float32 or
            constants.shape != (len(CONSTANTS),) or not constants.flags.c_contiguous):
        raise ValueError('constants must be a contiguous float32 device array (13,)')
    if packed.device.id != constants.device.id:
        raise ValueError('columns and constants must reside on the same CUDA device')
    n = packed.shape[1]
    if n == 0:
        return
    with cp.cuda.Device(packed.device.id):
        module, tables = _kernels(packed.device.id)
        module.get_function('mm5_columns')(
            ((n + 127) // 128,), (128,),
            (packed, constants, tables, np.int32(n),
             np.int32(scheme == SCHEMES[0]), np.int32(isfflx),
             np.int32(isftcflx), np.int32(iz0tlnd), np.int32(scm_force_flux),
             np.int32(shalwater_z0), np.uint32(optional_mask)))


_MPAS_NAMES = {'t2': 't2m', 'th2': 'th2m'}


def sfclayer_from_mpas(atmosphere: Mapping, diag_physics: Mapping,
                      sfc_input: Mapping, mesh_density, *, config_len_disp,
                      config_frac_seaice: bool, xice_threshold, corrections=(), driver_module=None):
    """Pack the MPAS pools; return main, sea, and incoming regime buffers.

    MPAS mpas_atmphys_driver_sfclayer.F:352-592. ``atmosphere`` supplies
    ux,vx,t1d,qv1d,p1d,dz8w1d,psfcpa as contiguous device vectors: the
    first-level u_p,v_p,t_p,qv_p,pres_hyd_p,dz_p and psfc_p from the common
    physics preparation. ``diag_physics`` and ``sfc_input`` use MPAS pool
    names. ``config_len_disp`` and ``xice_threshold`` come from MPAS.
    Revised lake mask and water depth are zero in this driver.
    """
    import cupy as cp
    import numpy as np
    n = atmosphere['ux'].size
    main = cp.empty((len(FIELDS), n), dtype=cp.float32)
    for j, name in enumerate(OUTPUTS):
        if name in ('q2', 't2', 'th2', 'u10', 'v10', 'cd', 'cda', 'ck', 'cka'):
            main[j].fill(0)
        else:
            main[j] = diag_physics[_MPAS_NAMES.get(name, name)]
    sources = {**atmosphere, 'tsk': sfc_input['skintemp'],
               'xland': sfc_input['xland'], 'pblh': diag_physics['hpbl'],
               'mavail': diag_physics['mavail']}
    for name in INPUTS:
        if name in ('dx', 'lakemask', 'water_depth'):
            continue
        main[FIELDS.index(name)] = sources[name]
    sea = cp.empty_like(main) if config_frac_seaice else main
    held = diag_physics['regime'].copy()
    sst = sfc_input['sst'] if config_frac_seaice else main[0]
    xice = sfc_input['xice'] if config_frac_seaice else main[0]
    if n:
        module = driver_module
        if module is None:
            module, _ = _kernels(main.device.id, tuple(sorted(set(corrections) & set(CORRECT_MODE_SOURCE))))
        module.get_function('mm5_from_mpas')(
            ((n + 127) // 128,), (128,),
            (main, sea, mesh_density, sst, xice, np.int32(n),
             np.float32(config_len_disp), np.float32(xice_threshold),
             np.int32(config_frac_seaice)))
    return main, sea, held


def sfclayer_to_mpas(main, sea, held, diag_physics: Mapping,
                    sfc_input: Mapping, *, config_frac_seaice: bool,
                    xice_threshold, driver_module=None) -> None:
    """MPAS mpas_atmphys_driver_sfclayer.F:595-796 pool stores and ice blend.

    Preserve MPAS's selective blending: HFX, QFX, LH, QSFC, ZNT, CPM and the
    LSM exchange coefficients retain the primary surface calculation.
    The incoming regime is restored when the ice fraction is at least 0.5.
    """
    import numpy as np
    n = main.shape[1]
    if config_frac_seaice and n:
        module = driver_module
        if module is None:
            module, _ = _kernels(main.device.id)
        module.get_function('mm5_to_mpas')(
            ((n + 127) // 128,), (128,),
            (main, sea, sfc_input['xice'], held, np.int32(n),
             np.float32(xice_threshold)))
    for j, name in enumerate(OUTPUTS):
        diag_physics[_MPAS_NAMES.get(name, name)][...] = main[j]


def run_surface_layer(atmosphere: Mapping, diag_physics: Mapping,
                      sfc_input: Mapping, mesh_density, constants, *,
                      scheme: str, config_len_disp,
                      config_frac_seaice: bool, xice_threshold,
                      sea_exchange: Mapping | None = None, corrections=()) -> None:
    """MPAS mpas_atmphys_driver_sfclayer.F:839-1088 MM5 driver branches.

    Driver switches are the source's fixed values ISFFLX=1, ISFTCFLX=0,
    IZ0TLND=0, SCM_FORCE_FLUX=0; MPAS's shallow-water flag is fixed at zero
    (mpas_atmphys_vars.F:604). Fractional ice invokes a second full column
    calculation, including cells outside the blend mask, as MPAS does.
    """
    main, sea, held = sfclayer_from_mpas(
        atmosphere, diag_physics, sfc_input, mesh_density,
        config_len_disp=config_len_disp, config_frac_seaice=config_frac_seaice,
        xice_threshold=xice_threshold, corrections=corrections)
    launch_columns(main, constants, scheme=scheme)
    if config_frac_seaice:
        launch_columns(sea, constants, scheme=scheme)
        # MPAS mpas_atmphys_driver_sfclayer.F:260-310 retains these arrays;
        # mpas_atmphys_driver_seaice.F:390-399 consumes the open-water results.
        if sea_exchange is not None:
            for name in SEA_EXCHANGE:
                sea_exchange[name][...] = sea[FIELDS.index(name)]
    sfclayer_to_mpas(main, sea, held, diag_physics, sfc_input,
                    config_frac_seaice=config_frac_seaice,
                    xice_threshold=xice_threshold)


def surface_diagnostics(packed, constants, *, lsm_scheme: str,
                        xice, xice_threshold, q2mxy=None, t2mxy=None) -> None:
    """MPAS mpas_atmphys_sfc_diagnostics.F:27-123, after the land model.

    Noah-MP land diagnostics use q2mxy/t2mxy; water and ice use surface fluxes.
    The default SELECT CASE arm leaves all diagnostic words untouched.
    """
    import numpy as np
    lsm = {'sf_noah': 1, 'sf_noahmp': 2}.get(lsm_scheme, 0)
    if not lsm or not packed.shape[1]:
        return
    if lsm == 2 and (q2mxy is None or t2mxy is None):
        raise ValueError('Noah-MP diagnostics require q2mxy and t2mxy')
    if q2mxy is None:
        q2mxy = packed[0]
    if t2mxy is None:
        t2mxy = packed[0]
    module, _ = _kernels(packed.device.id)
    n = packed.shape[1]
    module.get_function('mm5_sfc_diagnostics')(
        ((n + 127) // 128,), (128,),
        (packed, xice, q2mxy, t2mxy, constants, np.int32(n), np.int32(lsm),
         np.float32(xice_threshold)))


def wrf_surface_diagnostics(packed, constants, *, ua_phys: bool = False) -> None:
    """MPAS physics_wrf/module_sf_sfcdiags.F:7-69, including UA_PHYS."""
    import numpy as np
    n = packed.shape[1]
    if n:
        module, _ = _kernels(packed.device.id)
        module.get_function('mm5_wrf_sfcdiags')(
            ((n + 127) // 128,), (128,),
            (packed, constants, np.int32(n), np.int32(ua_phys)))


def register_hex_cuda(registry, arguments) -> None:
    """Bind f4 SlotCall callbacks to the MPAS surface-layer driver.

    MPAS mpas_atmphys_driver.F:277-285;
    mpas_atmphys_driver_sfclayer.F:839-1010. ``arguments(call)`` supplies
    atmosphere, diag_physics, sfc_input, mesh_density, constants and
    xice_threshold from the common prepared/native pools. Resolved source
    options own the two config values. The provider does no scheme maths.
    """
    def invoke(call):
        """Dispatch the MPAS mpas_atmphys_driver_sfclayer.F:839-1010 slice."""
        from .driver_v842 import slot_corrections
        return run_surface_layer(
            **arguments(call), corrections=slot_corrections(call, CORRECT_MODE_SOURCE), scheme=call.scheme,
            config_len_disp=call.options['config_len_disp'],
            config_frac_seaice=call.options['config_frac_seaice'])
    for scheme in SCHEMES:
        registry.register('surface_layer', scheme, 'hex-cuda', invoke,
                          corrections=CORRECT_MODE_SOURCE)


def run_surface_diagnostics(atmosphere: Mapping, diag_physics: Mapping,
                            sfc_input: Mapping, constants, *,
                            lsm_scheme: str, xice_threshold,
                            q2mxy=None, t2mxy=None) -> None:
    """Apply the MPAS diagnostic pool boundary after the land model.

    MPAS mpas_atmphys_sfc_diagnostics.F:27-123. Common preparation supplies
    psfcpa, and the source pools supply skintemp/xland/xice and fluxes.
    Only q2, t2m and th2m are stored; all other pool fields are retained.
    The default SELECT CASE branch returns before reading unused inputs.
    """
    if lsm_scheme not in ('sf_noah', 'sf_noahmp'):
        return
    import cupy as cp
    n = atmosphere['psfcpa'].size
    packed = cp.empty((len(FIELDS), n), dtype=cp.float32)
    for name in ('hfx', 'qfx', 'qsfc', 'chs2', 'cqs2'):
        packed[FIELDS.index(name)] = diag_physics[name]
    packed[FIELDS.index('psfcpa')] = atmosphere['psfcpa']
    packed[FIELDS.index('tsk')] = sfc_input['skintemp']
    packed[FIELDS.index('xland')] = sfc_input['xland']
    surface_diagnostics(packed, constants, lsm_scheme=lsm_scheme,
                        xice=sfc_input['xice'], xice_threshold=xice_threshold,
                        q2mxy=q2mxy, t2mxy=t2mxy)
    for name in ('q2', 't2', 'th2'):
        diag_physics[_MPAS_NAMES.get(name, name)][...] = packed[FIELDS.index(name)]


# Registry entries for the integrator's f4 adapter; not a CLI registration.
HEX_CUDA_BACKENDS = {scheme: run_surface_layer for scheme in SCHEMES}


def _native_len_disp(native):
    """Read live RKIND config; src/framework/mpas_pool_routines.F:4761-4791.

    Native initialization resolves zero to nominalMinDc. Read the exported
    pool word rather than the original namelist option. The direct ABI path
    permits the additive f3 export before its Python convenience method lands.
    """
    if callable(getattr(native, 'get_real_config', None)):
        return native.get_real_config('config_len_disp')
    import ctypes as ct
    import numpy as np
    from .native_backend import _LOCK
    with _LOCK:
        native._check()
        function = native._bind('native_config_get_real',
            [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_float)])
        value = ct.c_float()
        function(native.pool('configs'), b'config_len_disp', ct.byref(value))
        return np.float32(value.value)


# MPAS mpas_atmphys_driver_sfclayer.F:260-310 retained native globals;
# mpas_atmphys_driver_seaice.F:390-399 delayed fractional-ice blending.
SEA_EXCHANGE = ('chs', 'chs2', 'cqs2', 'cpm', 'hfx', 'qfx', 'qgh', 'qsfc', 'lh')


def bind_native_pools(provider, call):
    """MPAS mpas_atmphys_driver_sfclayer.F:352-592,839-1088 native slot ABI.

    allocate_sfclayer allocates without from/to conversion; common
    MPAS_to_physics has prepared the atmospheric globals. Slice their first
    model level and leave all conversion/blending to run_surface_layer.
    """
    cp = provider.cp
    aliases = {'g': 'gravity', 'rovcp': 'rcp', 'r': 'R_d',
               'ep1': 'ep_1', 'ep2': 'ep_2', 'p1000mb': 'P0'}
    constants = cp.asarray([provider.native.get_constant(aliases.get(n, n))
                            for n in CONSTANTS], dtype=cp.float32)
    profiles = dict(ux='u_p', vx='v_p', t1d='t_p', qv1d='qv_p',
                    p1d='pres_hyd_p', dz8w1d='dz_p')
    atmosphere = {name: provider.prepared(global_name)[0]
                  for name, global_name in profiles.items()}
    atmosphere['psfcpa'] = provider.prepared('psfc_p')
    return dict(atmosphere=atmosphere, diag_physics=provider['diag_physics'],
                sfc_input=provider['sfc_input'],
                mesh_density=provider.field('mesh', 'meshDensity'),
                constants=constants, config_len_disp=_native_len_disp(provider.native),
                xice_threshold=provider['scalar_state']['xice_threshold'],
                sea_exchange=({name: provider.prepared(name + '_sea')
                               for name in SEA_EXCHANGE}
                              if call.options['config_frac_seaice'] else None))


def run_slot(call):
    """MPAS mpas_atmphys_driver_sfclayer.F:839-1088 SlotCall dispatch."""
    from .driver_v842 import slot_corrections
    return run_surface_layer(**call.pools, scheme=call.scheme,
                             corrections=slot_corrections(call, CORRECT_MODE_SOURCE),
                             config_frac_seaice=call.options['config_frac_seaice'])


def run_native_diagnostics(provider, call):
    """MPAS mpas_atmphys_sfc_diagnostics.F:27-123 post-LSM transaction.

    The pressure comes from the current diag pool after LSM/sea-ice, rather
    than a possibly stale common preparation array. Only diagnostics change.
    """
    lsm = call.options['config_lsm_scheme']
    if lsm not in ('sf_noah', 'sf_noahmp'):
        return
    provider.begin(call, {})
    cp = provider.cp
    aliases = {'g': 'gravity', 'rovcp': 'rcp', 'r': 'R_d',
               'ep1': 'ep_1', 'ep2': 'ep_2', 'p1000mb': 'P0'}
    constants = cp.asarray([provider.native.get_constant(aliases.get(n, n))
                            for n in CONSTANTS], dtype=cp.float32)
    extra = ({name: provider.field('output_noahmp', name)
              for name in ('q2mxy', 't2mxy')} if lsm == 'sf_noahmp' else {})
    run_surface_diagnostics(
        atmosphere={'psfcpa': provider.field('diag', 'surface_pressure')},
        diag_physics=provider['diag_physics'], sfc_input=provider['sfc_input'],
        constants=constants, lsm_scheme=lsm,
        xice_threshold=provider['scalar_state']['xice_threshold'], **extra)
    provider.commit()
