// WRF v4.7.1 module_sf_noahdrv.F:3206-4175, mosaic column.
// SFLX and its subtree are mapped from module_sf_noahlsm.F.
// D1 uses the corrected 4*t+ns soil index; D2 weights cell increments;
// D3 RC/LAI mosaic consumer outputs are absent. CuPy FTZ may flush inputs.
// FASDAS and WRF-Hydro are not ported.
// UCM is composed only under NOAH_MOSAIC_UCM; the plain unit stays inert.
// Qualified against the committed scalar WRF column fixtures; FTZ pairs pinned in tests.
/* Correctly-rounded radix-10 logarithm function for binary32 value.

Copyright (c) 2022-2023 Alexei Sibidanov.

This file is part of the CORE-MATH project
project (file src/binary32/log10/log10f.c, revision bc385c2).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// glibc 2.43 e_log10f.c, CORE-MATH revision bc385c2.
// Internal double arithmetic is the declared libm implementation, not physics.
  __device__ __constant__ double noah_log10_tr[] =
    {
      0x1p+0,         0x1.f81f82p-1,  0x1.f07c1fp-1,  0x1.e9131acp-1,
      0x1.e1e1e1ep-1, 0x1.dae6077p-1, 0x1.d41d41dp-1, 0x1.cd85689p-1,
      0x1.c71c71cp-1, 0x1.c0e0704p-1, 0x1.bacf915p-1, 0x1.b4e81b5p-1,
      0x1.af286bdp-1, 0x1.a98ef6p-1,  0x1.a41a41ap-1, 0x1.9ec8e95p-1,
      0x1.999999ap-1, 0x1.948b0fdp-1, 0x1.8f9c19p-1,  0x1.8acb90fp-1,
      0x1.8618618p-1, 0x1.8181818p-1, 0x1.7d05f41p-1, 0x1.78a4c81p-1,
      0x1.745d174p-1, 0x1.702e05cp-1, 0x1.6c16c17p-1, 0x1.6816817p-1,
      0x1.642c859p-1, 0x1.605816p-1,  0x1.5c9882cp-1, 0x1.58ed231p-1,
      0x1.5555555p-1, 0x1.51d07ebp-1, 0x1.4e5e0a7p-1, 0x1.4afd6ap-1,
      0x1.47ae148p-1, 0x1.446f865p-1, 0x1.4141414p-1, 0x1.3e22cbdp-1,
      0x1.3b13b14p-1, 0x1.3813814p-1, 0x1.3521cfbp-1, 0x1.323e34ap-1,
      0x1.2f684bep-1, 0x1.2c9fb4ep-1, 0x1.29e412ap-1, 0x1.27350b9p-1,
      0x1.2492492p-1, 0x1.21fb781p-1, 0x1.1f7047ep-1, 0x1.1cf06aep-1,
      0x1.1a7b961p-1, 0x1.1811812p-1, 0x1.15b1e5fp-1, 0x1.135c811p-1,
      0x1.1111111p-1, 0x1.0ecf56cp-1, 0x1.0c9715p-1,  0x1.0a6810ap-1,
      0x1.0842108p-1, 0x1.0624dd3p-1, 0x1.041041p-1,  0x1.0204081p-1,
      0.5
    };
  __device__ __constant__ double noah_log10_tl[] =
    {
      -0x1.d45fd6237ebe3p-47, 0x1.b947689311b6ep-8, 0x1.b5e909c96d7d5p-7,
      0x1.45f4f59ed2165p-6,   0x1.af5f92cbd8f1ep-6, 0x1.0ba01a606de8cp-5,
      0x1.3ed119b9a2b7bp-5,   0x1.714834298eec2p-5, 0x1.a30a9d98357fbp-5,
      0x1.d41d512670813p-5,   0x1.02428c0f65519p-4, 0x1.1a23444eecc3ep-4,
      0x1.31b30543f4cb4p-4,   0x1.48f3ed39bfd04p-4, 0x1.5fe8049a0e423p-4,
      0x1.769140a6aa008p-4,   0x1.8cf1836c98cb3p-4, 0x1.a30a9d55541a1p-4,
      0x1.b8de4d1ee823ep-4,   0x1.ce6e4202ca2e6p-4, 0x1.e3bc1accace07p-4,
      0x1.f8c9683b5abd4p-4,   0x1.06cbd68ca9a6ep-3, 0x1.11142f19df73p-3,
      0x1.1b3e71fa7a97fp-3,   0x1.254b4d37a46e3p-3, 0x1.2f3b6912cbf07p-3,
      0x1.390f683115886p-3,   0x1.42c7e7fffc5a8p-3, 0x1.4c65808c78d3cp-3,
      0x1.55e8c50751c55p-3,   0x1.5f52445dec3d8p-3, 0x1.68a288c3f12p-3,
      0x1.71da17bdf0d19p-3,   0x1.7af973608afd9p-3, 0x1.84011952a2579p-3,
      0x1.8cf1837a7ea6p-3,    0x1.95cb2891e43d6p-3, 0x1.9e8e7b0f869ep-3,
      0x1.a73beaa5db18dp-3,   0x1.afd3e394558d3p-3, 0x1.b856cf060d9f1p-3,
      0x1.c0c5134de1ffcp-3,   0x1.c91f1371bc99fp-3, 0x1.d1652ffcd3f53p-3,
      0x1.d997c6f635e75p-3,   0x1.e1b733ab90f3bp-3, 0x1.e9c3ceadac856p-3,
      0x1.f1bdeec43a305p-3,   0x1.f9a5e7a5fa3fep-3, 0x1.00be05ac02f2bp-2,
      0x1.04a054d81a2d4p-2,   0x1.087a0835957fbp-2, 0x1.0c4b457099517p-2,
      0x1.101431aa1fe51p-2,   0x1.13d4f08b98dd8p-2, 0x1.178da53edb892p-2,
      0x1.1b3e71e9f9d58p-2,   0x1.1ee777defdeedp-2, 0x1.2288d7b48e23bp-2,
      0x1.2622b0f52e49fp-2,   0x1.29b522a4c6314p-2, 0x1.2d404b0e30f8p-2,
      0x1.30c4478f3fbe5p-2,   0x1.34413509f7915p-2
    };
__device__ __constant__ unsigned noah_log10_st[] = {0x3f800000u,0x41200000u,0x42c80000u,0x447a0000u,0x461c4000u,0x47c35000u,0x49742400u,0x4b189680u,0x4cbebc20u,0x4e6e6b28u,0x501502f9u,0x0u,0x0u,0x0u,0x0u,0x0u};
  __device__ __constant__ double noah_log10_b[] =
    {
      0x1.bcb7b15c5a2f8p-2, -0x1.bcbb1dbb88ebap-3, 0x1.2871c39d521c6p-3
    };
  __device__ __constant__ double noah_log10_c[] =
    {
      0x1.bcb7b1526e50ep-2,  -0x1.bcb7b1526e53dp-3, 0x1.287a7636f3fa2p-3,
      -0x1.bcb7b146a14b3p-4, 0x1.63c627d5219cbp-4,  -0x1.2880736c8762dp-4,
      0x1.fc1ecf913961ap-5
    };

__device__ static float noah_log10_special(float x) {
 unsigned ux=__float_as_uint(x),ax=ux<<1;
 if(ux==0x7f800000u)return x;
 if(ax==0u)return __uint_as_float(0xff800000u);
 if(ax>0xff000000u)return FADD(x,x);
 return FDIV(FSUB(x,x),FSUB(x,x));
}
__device__ static float noah_log10(float x) {
  unsigned ux = __float_as_uint(x);
  if ( (ux < (1 << 23) || ux >= 0x7f800000u))
    {
      if (ux == 0 || ux >= 0x7f800000u)
	return noah_log10_special(x);
      /* subnormal */
      int n = __clz (ux) - 8;
      ux <<= n;
      ux -= n << 23;
    }
  unsigned m = ux & ((1 << 23) - 1), j = (m + (1 << (23 - 7))) >> (23 - 6);
  double ix = noah_log10_tr[j], l = noah_log10_tl[j];
  int e = ((int) ux >> 23) - 127;
  unsigned je = e + 1;
  je = (je * 0x4d104d4) >> 28;
  if ( (ux == noah_log10_st[je]))
    return je;

  double tz = __longlong_as_double (((long long) m | ((long long) 1023 << 23)) << (52 - 23));
  double z = tz * ix - 1, z2 = z * z;
  double r
      = ((e * 0x1.34413509f79ffp-2 + l) + z * noah_log10_b[0]) + z2 * (noah_log10_b[1] + z * noah_log10_b[2]);
  float ub = r, lb = r + 0x1.b008p-34;
  if ( (ub != lb))
    {
      double f = z
		 * ((noah_log10_c[0] + z * noah_log10_c[1])
		    + z2
			  * ((noah_log10_c[2] + z * noah_log10_c[3])
			     + z2 * (noah_log10_c[4] + z * noah_log10_c[5] + z2 * noah_log10_c[6])));
      f -= 0x1.0cee0ed4ca7e9p-54 * e;
      f += l - noah_log10_tl[0];
      double el = e * 0x1.34413509f7ap-2;
      r = el + f;
      ub = r;
      tz = r;
      if ( (!((__double_as_longlong(tz) & ((1 << 28) - 1)))))
	{
	  double dr = (el - r) + f;
	  r += dr * 32;
	  ub = r;
	}
    }
  return ub;
}

#if !NOAH_MATH_ONLY
#undef RCP
#define RCP __uint_as_float(0x3e924925u) // WRF CAPA=R_D/CP
#define NSOIL 4

#define NRD      287.04f      // noahlsm module RD
#define NSIGMA   5.67e-8f     // noahlsm SIGMA
#define NCPH2O   4.218e+3f    // noahlsm CPH2O
#define NCPICE   2.106e+3f    // noahlsm CPICE
#define NLSUBF   3.335e+5f    // noahlsm LSUBF
#define NEMISSI_S 0.95f       // noahlsm EMISSI_S
#define NTFREEZ  273.15f      // SFLX TFREEZ
#define NLVH2O   2.501e+6f    // SFLX LVH2O
#define NLSUBS   2.83e+6f     // SFLX/PENMAN/SNOPAC LSUBS
#define NR_SHEAT 287.04f      // SFLX local R
#define NCP_PEN  1004.6f      // PENMAN's local CP
#define NELCP    2.4888e+3f   // PENMAN ELCP
#define NLSUBC   2.501e+6f    // PENMAN/SNOPAC LSUBC
#define NSTBOLT  5.67051e-8f  // module_model_constants STBOLT (NOAHRES)
#define NXLF     3.50e+5f     // module_model_constants XLF (SNOPCX)

// packed-table column indices (gpuwm/core/noah.py VEG_COLS/SOIL_COLS/GEN)
#define NVEGC 15
#define VG_NROOT 0
#define VG_RSMIN 1
#define VG_RGL 2
#define VG_HS 3
#define VG_SNUP 4
#define VG_LAIMIN 5
#define VG_LAIMAX 6
#define VG_EMISSMIN 7
#define VG_EMISSMAX 8
#define VG_ALBEDOMIN 9
#define VG_ALBEDOMAX 10
#define VG_Z0MIN 11
#define VG_Z0MAX 12
#define VG_SHDTBL 13
#define VG_MAXALB 14
#define NSOILC 10
#define SO_BEXP 0
#define SO_SMCDRY 1
#define SO_F1 2
#define SO_SMCMAX 3
#define SO_SMCREF 4
#define SO_PSISAT 5
#define SO_DKSAT 6
#define SO_DWSAT 7
#define SO_SMCWLT 8
#define SO_QUARTZ 9
#define GEN_TOPT 0
#define GEN_CMCMAX 1
#define GEN_CFACTR 2
#define GEN_RSMAX 3
#define GEN_SBETA 4
#define GEN_FXEXP 5
#define GEN_CSOIL 6
#define GEN_SALP 7
#define GEN_REFDK 8
#define GEN_REFKDT 9
#define GEN_FRZK 10
#define GEN_ZBOT 11
#define GEN_LVCOEF 12
#define GEN_SLOPE 13
#define GEN_BARE 14
#define GEN_NATURAL 15

// ---------------------------------------------------------------- FRH2O
// module_sf_noahlsm.F:1447
__device__ static real noah_frh2o(real tkelv, real smc, real sh2o,
                                  real smcmax, real bexp, real psis)
{
    const real ck = 8.0f, blim = 5.5f, error = 0.005f;
    const real hlice = 3.335e5f, gs = 9.81f, t0 = 273.15f;
    real bx = (bexp <= blim) ? bexp : blim;
    int nlog = 0, kcount = 0;
    if (tkelv > (__uint_as_float(0x43889312u) /* gfortran15.2 fold_000 */)) return smc;
    real swl = FSUB(smc, sh2o);
    if (swl > (FSUB(smc, 0.02f))) swl = FSUB(smc, 0.02f);
    if (swl < 0.0f) swl = 0.0f;
    while (nlog < 10 && kcount == 0) {
        nlog += 1;
        real df = FSUB(gfk_log(FMUL(FMUL((FDIV(FMUL(psis, gs), hlice)), gfk_pow(FADD(1.0f, FMUL(ck, swl)), 2.0f)), gfk_pow(FDIV(smcmax, (FSUB(smc, swl))), bx))), gfk_log(FDIV(-(FSUB(tkelv, t0)), tkelv)));
        real denom = FADD(FDIV(__uint_as_float(0x41800000u) /* gfortran15.2 fold_001 */, (FADD(1.0f, FMUL(ck, swl)))), FDIV(bx, (FSUB(smc, swl))));
        real swlk = FSUB(swl, FDIV(df, denom));
        if (swlk > (FSUB(smc, 0.02f))) swlk = FSUB(smc, 0.02f);
        if (swlk < 0.0f) swlk = 0.0f;
        real dswl = fabsf(FSUB(swlk, swl));
        swl = swlk;
        if (dswl <= error) kcount += 1;
    }
    real freew = FSUB(smc, swl);
    if (kcount == 0) {                 // Flerchinger explicit fallback
        real fk = FMUL(gfk_pow(FMUL((FDIV(hlice, (FMUL(gs, (-psis))))), (FDIV((FSUB(tkelv, t0)), tkelv))), FDIV(-1.0f, bx)), smcmax);
        if (fk < 0.02f) fk = 0.02f;
        freew = fminf(fk, smc);
    }
    return freew;
}

// ---------------------------------------------------------------- CSNOW
// module_sf_noahlsm.F:1148
__device__ static real noah_csnow(real dsnow)
{
    return FMUL(__uint_as_float(0x3e6e33f0u) /* gfortran15.2 fold_002 */, (FMUL(0.328f, gfk_pow(10.0f, FMUL(2.25f, dsnow)))));
}

// ------------------------------------------------------------- SNOW_NEW
// module_sf_noahlsm.F:3601
__device__ static void noah_snow_new(real temp, real newsn, real& snowh,
                                     real& sndens)
{
    real snowhc = FMUL(snowh, 100.0f);
    real newsnc = FMUL(newsn, 100.0f);
    real tempc = FSUB(temp, 273.15f);
    real dsnew;
    if (tempc <= -15.0f) dsnew = 0.05f;
    else dsnew = FADD(0.05f, FMUL(0.0017f, gfk_pow(FADD(tempc, 15.0f), 1.5f)));
    real hnewc = FDIV(newsnc, dsnew);
    if (FADD(snowhc, hnewc) < 1.0e-3f) sndens = fmaxf(dsnew, sndens);
    else sndens = FDIV((FADD(FMUL(snowhc, sndens), FMUL(hnewc, dsnew))), (FADD(snowhc, hnewc)));
    snowhc = FADD(snowhc, hnewc);
    snowh = FMUL(snowhc, 0.01f);
}

// --------------------------------------------------------------- SNFRAC
// module_sf_noahlsm.F:2817
__device__ static real noah_snfrac(real sneqv, real snup, real salp)
{
    if (sneqv < snup) {
        real rsnow = FDIV(sneqv, snup);
        return FSUB(1.0f, (FSUB(gfk_exp(FMUL(-salp, rsnow)), FMUL(rsnow, gfk_exp(-salp)))));
    }
    return 1.0f;
}

// --------------------------------------------------------------- ALCALC
// module_sf_noahlsm.F:891
__device__ static void noah_alcalc(real alb, real snoalb, real embrd,
                                   real sncovr, real dt, bool snowng,
                                   real& snotime1, real lvcoef,
                                   real& albedo, real& emissi)
{
    const real snacca = 0.94f, snaccb = 0.58f;
    albedo = FADD(alb, FMUL(sncovr, (FSUB(snoalb, alb))));
    emissi = FADD(embrd, FMUL(sncovr, (FSUB(NEMISSI_S, embrd))));
    real snoalb1 = FADD(snoalb, FMUL(lvcoef, (FSUB(0.85f, snoalb))));
    real snoalb2 = snoalb1;
    if (snowng) {
        snotime1 = 0.0f;
    } else {
        snotime1 = FADD(snotime1, dt);
        snoalb2 = FMUL(snoalb1, gfk_pow(snacca,
                                 gfk_pow(FDIV(snotime1, 86400.0f), snaccb)));
    }
    snoalb2 = fmaxf(snoalb2, alb);
    albedo = FADD(alb, FMUL(sncovr, (FSUB(snoalb2, alb))));
    if (albedo > snoalb2) albedo = snoalb2;
}

// --------------------------------------------------------------- TDFCND
// module_sf_noahlsm.F:4124
__device__ static real noah_tdfcnd(real smc, real qz, real smcmax,
                                   real sh2o, real bexp, real psisat,
                                   int soiltyp, int opt_thcnd)
{
    if (opt_thcnd == 1 || (opt_thcnd == 2 && soiltyp != 4
                           && soiltyp != 3)) {
        real satratio = FDIV(smc, smcmax);
        const real thkice = 2.2f, thkw = 0.57f, thko = 2.0f,
                   thkqtz = 7.7f;
        real thks = FMUL(gfk_pow(thkqtz, qz), gfk_pow(thko, FSUB(1.0f, qz)));
        real xunfroz = FDIV(sh2o, smc);
        real xu = FMUL(xunfroz, smcmax);
        real thksat = FMUL(FMUL(gfk_pow(thks, FSUB(1.0f, smcmax)), gfk_pow(thkice, FSUB(smcmax, xu))), gfk_pow(thkw, xu));
        real gammd = FMUL((FSUB(1.0f, smcmax)), 2700.0f);
        real thkdry = FDIV((FADD(FMUL(0.135f, gammd), 64.7f)), (FSUB(2700.0f, FMUL(0.947f, gammd))));
        real akei = satratio;
        real akel;
        if (satratio > 0.1f) akel = FADD(noah_log10(satratio), 1.0f);
        else akel = 0.0f;
        real ake = FDIV((FADD(FMUL((FSUB(smc, sh2o)), akei), FMUL(sh2o, akel))), smc);
        return FADD(FMUL(ake, (FSUB(thksat, thkdry))), thkdry);
    }
    real psif = FMUL(FMUL(psisat, 100.0f), gfk_pow(FDIV(smcmax, smc), bexp));
    real pf = noah_log10(fabsf(psif));
    if (pf <= 5.1f) return FMUL(420.0f, gfk_exp(-(FADD(pf, 2.7f))));
    return 0.1744f;
}

// --------------------------------------------------------------- SNOWZ0
// module_sf_noahlsm.F:3552
__device__ static real noah_snowz0(real sncovr, real z0brd, real snowh)
{
    const real z0s = 0.001f;
    real burial = FSUB(FMUL(7.0f, z0brd), snowh);
    real z0eff;
    if (burial <= 0.0007f) z0eff = z0s;
    else z0eff = FDIV(burial, 7.0f);
    return FADD(FMUL((FSUB(1.0f, sncovr)), z0brd), FMUL(sncovr, z0eff));
}

// --------------------------------------------------------------- PENMAN
// module_sf_noahlsm.F:2195
__device__ static void noah_penman(real sfctmp, real sfcprs, real ch,
                                   real t2v, real th2, real prcp,
                                   real fdown, real ssoil, real q2,
                                   real q2sat, real dqsdt2, bool snowng,
                                   bool frzgra, real emissi_in,
                                   real sncovr, real& etp, real& rch,
                                   real& rr, real& epsca, real& t24,
                                   real& flx2)
{
    real emissi = emissi_in;
    real elcp1 = FADD(FMUL((FSUB(1.0f, sncovr)), NELCP), FDIV(FMUL(FMUL(sncovr, NELCP), NLSUBS), NLSUBC));
    real lvs = FADD(FMUL((FSUB(1.0f, sncovr)), NLSUBC), FMUL(sncovr, NLSUBS));
    flx2 = 0.0f;
    real delta = FMUL(elcp1, dqsdt2);
    t24 = FMUL(FMUL(FMUL(sfctmp, sfctmp), sfctmp), sfctmp);
    rr = FADD(FDIV(FMUL(FMUL(emissi, t24), 6.48e-8f), (FMUL(sfcprs, ch))), 1.0f);
    real rho = FDIV(sfcprs, (FMUL(NRD, t2v)));
    rch = FMUL(FMUL(rho, NCP_PEN), ch);
    if (!snowng) {
        if (prcp > 0.0f) rr = FADD(rr, FDIV(FMUL(NCPH2O, prcp), rch));
    } else {
        rr = FADD(rr, FDIV(FMUL(NCPICE, prcp), rch));
    }
    real fnet = FSUB(FSUB(fdown, FMUL(FMUL(emissi, NSIGMA), t24)), ssoil);
    if (frzgra) {
        flx2 = FMUL(-NLSUBF, prcp);
        fnet = FSUB(fnet, flx2);
    }
    real rad = FSUB(FADD(FDIV(fnet, rch), th2), sfctmp);
    real a = FMUL(elcp1, (FSUB(q2sat, q2)));
    epsca = FDIV((FADD(FMUL(a, rr), FMUL(rad, delta))), (FADD(delta, rr)));
    etp = FDIV(FMUL(epsca, rch), lvs);           // AOASIS = 1 without the UCM
}

// --------------------------------------------------------------- CANRES
// module_sf_noahlsm.F:1009
__device__ static void noah_canres(real solar, real ch, real sfctmp,
                                   real q2, const real* sh2o,
                                   const real* zsoil, real smcwlt,
                                   real smcref, real rsmin, int nroot,
                                   real q2sat, real dqsdt2, real topt,
                                   real rsmax, real rgl, real hs,
                                   real xlai, real sfcprs, real emissi,
                                   real& rc, real& pc)
{
    const real slv = 2.501000e6f;
    real ff = FDIV(FMUL(__uint_as_float(0x3f8ccccdu) /* gfortran15.2 fold_003 */, solar), (FMUL(rgl, xlai)));
    real rcs = FDIV((FADD(ff, FDIV(rsmin, rsmax))), (FADD(1.0f, ff)));
    rcs = fmaxf(rcs, 0.0001f);
    real rct = FSUB(1.0f, FMUL(0.0016f, gfk_pow(FSUB(topt, sfctmp), 2.0f)));
    rct = fmaxf(rct, 0.0001f);
    real rcq = FDIV(1.0f, (FADD(1.0f, FMUL(hs, (FSUB(q2sat, q2))))));
    rcq = fmaxf(rcq, 0.01f);
    real rcsoil = 0.0f;
    real gx = FDIV((FSUB(sh2o[0], smcwlt)), (FSUB(smcref, smcwlt)));
    gx = fminf(fmaxf(gx, 0.0f), 1.0f);
    real part[NSOIL];
    part[0] = FMUL((FDIV(zsoil[0], zsoil[nroot - 1])), gx);
    for (int k = 1; k < nroot; ++k) {
        gx = FDIV((FSUB(sh2o[k], smcwlt)), (FSUB(smcref, smcwlt)));
        gx = fminf(fmaxf(gx, 0.0f), 1.0f);
        part[k] = FMUL((FDIV((FSUB(zsoil[k], zsoil[k - 1])), zsoil[nroot - 1])), gx);
    }
    for (int k = 0; k < nroot; ++k) rcsoil = FADD(rcsoil, part[k]);
    rcsoil = fmaxf(rcsoil, 0.0001f);
    rc = FDIV(rsmin, (FMUL(FMUL(FMUL(FMUL(xlai, rcs), rct), rcq), rcsoil)));
    real rr2 = FADD(FDIV(FMUL((FDIV(FMUL(FMUL(FMUL(4.0f, emissi), NSIGMA), NRD), CP)), gfk_pow(sfctmp, 4.0f)), (FMUL(sfcprs, ch))), 1.0f);
    real delta = FMUL((__uint_as_float(0x451b9cbcu) /* gfortran15.2 fold_004 */), dqsdt2);
    pc = FDIV((FADD(rr2, delta)), (FADD(FMUL(rr2, (FADD(1.0f, FMUL(rc, ch)))), delta)));
}

// ---------------------------------------------------------------- DEVAP
// module_sf_noahlsm.F:1189
__device__ static real noah_devap(real etp1, real smc, real shdfac,
                                  real smcmax, real smcdry, real fxexp)
{
    real sratio = FDIV((FSUB(smc, smcdry)), (FSUB(smcmax, smcdry)));
    real fx;
    if (sratio > 0.0f) {
        fx = gfk_pow(sratio, fxexp);
        fx = fmaxf(fminf(fx, 1.0f), 0.0f);
    } else {
        fx = 0.0f;
    }
    return FMUL(FMUL(fx, (FSUB(1.0f, shdfac))), etp1);
}

// --------------------------------------------------------------- TRANSP
// module_sf_noahlsm.F:4355
__device__ static void noah_transp(real* et, real etp1,
                                   const real* sh2o, real cmc,
                                   real shdfac, real smcwlt,
                                   real cmcmax, real pc, real cfactr,
                                   real smcref, int nroot,
                                   const real* rtdis)
{
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    real etp1a;
    if (cmc != 0.0f)
        etp1a = FMUL(FMUL(FMUL(shdfac, pc), etp1), (FSUB(1.0f, gfk_pow(FDIV(cmc, cmcmax), cfactr))));
    else
        etp1a = FMUL(FMUL(shdfac, pc), etp1);
    real gx[NSOIL];
    real sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = FDIV((FSUB(sh2o[i], smcwlt)), (FSUB(smcref, smcwlt)));
        gx[i] = fmaxf(fminf(gx[i], 1.0f), 0.0f);
        sgx = FADD(sgx, gx[i]);
    }
    sgx = FDIV(sgx, (real)nroot);
    real denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        real rtx = FSUB(FADD(rtdis[i], gx[i]), sgx);
        gx[i] = FMUL(gx[i], fmaxf(rtx, 0.0f));
        denom = FADD(denom, gx[i]);
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = FDIV(FMUL(etp1a, gx[i]), denom);
}

// ---------------------------------------------------------------- EVAPO
// module_sf_noahlsm.F:1323
__device__ static void noah_evapo(real& eta1, const real* smc, real cmc,
                                  real etp1, real dt, const real* sh2o,
                                  real smcmax, real pc, real smcwlt,
                                  real smcref, real shdfac, real cmcmax,
                                  real smcdry, real cfactr, int nroot,
                                  const real* rtdis, real fxexp,
                                  real& edir, real& ec, real* et,
                                  real& ett)
{
    edir = 0.0f;
    ec = 0.0f;
    ett = 0.0f;
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    if (etp1 > 0.0f) {
        if (shdfac < 1.0f)
            edir = noah_devap(etp1, smc[0], shdfac, smcmax, smcdry,
                              fxexp);
        if (shdfac > 0.0f) {
            noah_transp(et, etp1, sh2o, cmc, shdfac, smcwlt, cmcmax,
                        pc, cfactr, smcref, nroot, rtdis);
            for (int k = 0; k < NSOIL; ++k) ett = FADD(ett, et[k]);
            if (cmc > 0.0f)
                ec = FMUL(FMUL(shdfac, gfk_pow(FDIV(cmc, cmcmax), cfactr)), etp1);
            else
                ec = 0.0f;
            real cmc2ms = FDIV(cmc, dt);
            ec = fminf(cmc2ms, ec);
        }
    }
    eta1 = FADD(FADD(edir, ett), ec);
}

// -------------------------------------------------------------- FAC2MIT
// module_sf_noahlsm.F:1424
__device__ static real noah_fac2mit(real smcmax)
{
    real flimit = 0.90f;
    if (smcmax == 0.395f) flimit = 0.59f;
    else if (smcmax == 0.434f || smcmax == 0.404f) flimit = 0.85f;
    else if (smcmax == 0.465f || smcmax == 0.406f) flimit = 0.86f;
    else if (smcmax == 0.476f || smcmax == 0.439f) flimit = 0.74f;
    else if (smcmax == 0.200f || smcmax == 0.464f) flimit = 0.80f;
    return flimit;
}

// --------------------------------------------------------------- WDFCND
// module_sf_noahlsm.F:4461
__device__ static void noah_wdfcnd(real& wdf, real& wcnd, real smc,
                                   real smcmax, real bexp, real dksat,
                                   real dwsat, real sicemax)
{
    real factr1 = FDIV(0.05f, smcmax);
    real factr2 = FDIV(smc, smcmax);
    factr1 = fminf(factr1, factr2);
    real expon = FADD(bexp, 2.0f);
    wdf = FMUL(dwsat, gfk_pow(factr2, expon));
    if (sicemax > 0.0f) {
        real vkwgt = FDIV(1.0f, (FADD(1.0f, gfk_pow(FMUL(500.0f, sicemax), 3.0f))));
        wdf = FADD(FMUL(vkwgt, wdf), FMUL(FMUL((FSUB(1.0f, vkwgt)), dwsat), gfk_pow(factr1, expon)));
    }
    expon = FADD(FMUL(2.0f, bexp), 3.0f);
    wcnd = FMUL(dksat, gfk_pow(factr2, expon));
}

// ------------------------------------------------------------------ SRT
// module_sf_noahlsm.F:3803, CVFRZ-J is INTEGER, not a real exponent.
// libgcc __powisf2 order: square base, then multiply result for odd bits.
__device__ static real mosaic_powi(real x,int n) {
 unsigned count=n<0?(unsigned)(-n):(unsigned)n;
 real y=(count&1u)?x:1.0f;
 while((count>>=1u)){x=FMUL(x,x);if(count&1u)y=FMUL(y,x);}
 return n<0?FDIV(1.0f,y):y;
}
// module_sf_noahlsm.F:3653
__device__ static void noah_srt(real* rhstt, real edir, const real* et,
                                const real* sh2o, const real* sh2oa,
                                real pcpdrp, const real* zsoil,
                                real dwsat, real dksat, real smcmax,
                                real bexp, real dt, real smcwlt,
                                real slope, real kdt, real frzx,
                                const real* sice, real* ai, real* bi,
                                real* ci, real& runoff1, real& runoff2)
{
    const int cvfrz = 3;
    real dmax[NSOIL];
    real sicemax = 0.0f;
    for (int ks = 0; ks < NSOIL; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    real pddum = pcpdrp;
    runoff1 = 0.0f;
    runoff2 = 0.0f;
    if (pcpdrp != 0.0f) {
        real dt1 = FDIV(dt, 86400.0f);
        real smcav = FSUB(smcmax, smcwlt);
        dmax[0] = FMUL(-zsoil[0], smcav);
        real dice = FMUL(-zsoil[0], sice[0]);
        dmax[0] = FMUL(dmax[0], (FSUB(1.0f, FDIV((FSUB(FADD(sh2oa[0], sice[0]), smcwlt)), smcav))));
        real dd = dmax[0];
        for (int ks = 1; ks < NSOIL; ++ks) {
            dice = FADD(dice, FMUL((FSUB(zsoil[ks - 1], zsoil[ks])), sice[ks]));
            dmax[ks] = FMUL((FSUB(zsoil[ks - 1], zsoil[ks])), smcav);
            dmax[ks] = FMUL(dmax[ks], (FSUB(1.0f, FDIV((FSUB(FADD(sh2oa[ks], sice[ks]), smcwlt)), smcav))));
            dd = FADD(dd, dmax[ks]);
        }
        real val = FSUB(1.0f, gfk_exp(FMUL(-kdt, dt1)));
        real ddt = FMUL(dd, val);
        real px = FMUL(pcpdrp, dt);
        if (px < 0.0f) px = 0.0f;
        real infmax = FDIV((FMUL(px, (FDIV(ddt, (FADD(px, ddt)))))), dt);
        real fcr = 1.0f;
        if (dice > 1.0e-2f) {
            real acrt = FDIV(FMUL((real)cvfrz, frzx), dice);
            real ssum = 1.0f;
            int ialp1 = cvfrz - 1;
            for (int j = 1; j <= ialp1; ++j) {
                int k = 1;
                for (int jj = j + 1; jj <= ialp1; ++jj) k = k * jj;
                ssum = FADD(ssum, FDIV(mosaic_powi(acrt, cvfrz - j), (real)k));
            }
            fcr = FSUB(1.0f, FMUL(gfk_exp(-acrt), ssum));
        }
        infmax = FMUL(infmax, fcr);
        real wdf0, wcnd0;
        noah_wdfcnd(wdf0, wcnd0, sh2oa[0], smcmax, bexp, dksat, dwsat,
                    sicemax);
        infmax = fmaxf(infmax, wcnd0);
        infmax = fminf(infmax, FDIV(px, dt));
        if (pcpdrp > infmax) {
            runoff1 = FSUB(pcpdrp, infmax);
            pddum = infmax;
        }
    }
    real wdf, wcnd;
    noah_wdfcnd(wdf, wcnd, sh2oa[0], smcmax, bexp, dksat, dwsat,
                sicemax);
    real ddz = FDIV(1.0f, (FMUL(-0.5f, zsoil[1])));
    ai[0] = 0.0f;
    bi[0] = FDIV(FMUL(wdf, ddz), (-zsoil[0]));
    ci[0] = -bi[0];
    real dsmdz = FDIV((FSUB(sh2o[0], sh2o[1])), (FMUL(-0.5f, zsoil[1])));
    rhstt[0] = FDIV((FADD(FADD(FSUB(FADD(FMUL(wdf, dsmdz), wcnd), pddum), edir), et[0])), zsoil[0]);
    real ddz2 = 0.0f;
    for (int k = 1; k < NSOIL; ++k) {
        real denom2 = FSUB(zsoil[k - 1], zsoil[k]);
        real wdf2, wcnd2, dsmdz2, slopx;
        if (k != NSOIL - 1) {
            slopx = 1.0f;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[k], smcmax, bexp, dksat,
                        dwsat, sicemax);
            real denom = FSUB(zsoil[k - 1], zsoil[k + 1]);
            dsmdz2 = FDIV((FSUB(sh2o[k], sh2o[k + 1])), (FMUL(denom, 0.5f)));
            ddz2 = FDIV(2.0f, denom);
            ci[k] = FDIV(FMUL(-wdf2, ddz2), denom2);
        } else {
            slopx = slope;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[NSOIL - 1], smcmax, bexp,
                        dksat, dwsat, sicemax);
            dsmdz2 = 0.0f;
            ci[k] = 0.0f;
        }
        real numer = FADD(FSUB(FSUB(FADD(FMUL(wdf2, dsmdz2), FMUL(slopx, wcnd2)), FMUL(wdf, dsmdz)), wcnd), et[k]);
        rhstt[k] = FDIV(numer, (-denom2));
        ai[k] = FDIV(FMUL(-wdf, ddz), denom2);
        bi[k] = -(FADD(ai[k], ci[k]));
        if (k == NSOIL - 1) runoff2 = FMUL(slopx, wcnd2);
        if (k != NSOIL - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

// --------------------------------------------------------------- ROSR12
// module_sf_noahlsm.F:2537
__device__ static void noah_rosr12(real* p, const real* a,
                                   const real* b, const real* c_in,
                                   const real* d)
{
    real c_[NSOIL], delta[NSOIL];
    for (int k = 0; k < NSOIL; ++k) c_[k] = c_in[k];
    c_[NSOIL - 1] = 0.0f;
    p[0] = FDIV(-c_[0], b[0]);
    delta[0] = FDIV(d[0], b[0]);
    for (int k = 1; k < NSOIL; ++k) {
        p[k] = FMUL(-c_[k], (FDIV(1.0f, (FADD(b[k], FMUL(a[k], p[k - 1]))))));
        delta[k] = FMUL((FSUB(d[k], FMUL(a[k], delta[k - 1]))), (FDIV(1.0f, (FADD(b[k], FMUL(a[k], p[k - 1]))))));
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 1; k < NSOIL; ++k) {
        int kk = NSOIL - k - 1;
        p[kk] = FADD(FMUL(p[kk], p[kk + 1]), delta[kk]);
    }
}

// ---------------------------------------------------------------- SSTEP
// module_sf_noahlsm.F:3950
__device__ static void noah_sstep(real* sh2oout, const real* sh2oin,
                                  real& cmc, const real* rhstt_in,
                                  real rhsct, real dt, real smcmax,
                                  real cmcmax, const real* zsoil,
                                  real* smc, const real* sice,
                                  const real* ai_in, const real* bi_in,
                                  const real* ci_in, real& runoff3,
                                  bool update_cmc)
{
    real rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], p[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhstt[k] = FMUL(rhstt_in[k], dt);
        ai[k] = FMUL(ai_in[k], dt);
        bi[k] = FADD(1.0f, FMUL(bi_in[k], dt));
        ci[k] = FMUL(ci_in[k], dt);
    }
    noah_rosr12(p, ai, bi, ci, rhstt);
    real wplus = 0.0f;
    real ddz = -zsoil[0];
    for (int k = 0; k < NSOIL; ++k) {
        if (k != 0) ddz = FSUB(zsoil[k - 1], zsoil[k]);
        sh2oout[k] = FADD(FADD(sh2oin[k], p[k]), FDIV(wplus, ddz));
        real stot = FADD(sh2oout[k], sice[k]);
        if (stot > smcmax) {
            if (k == 0) ddz = -zsoil[0];
            else ddz = FADD(-zsoil[k], zsoil[k - 1]);
            wplus = FMUL((FSUB(stot, smcmax)), ddz);
        } else {
            wplus = 0.0f;
        }
        smc[k] = fmaxf(fminf(stot, smcmax), 0.02f);
        sh2oout[k] = fmaxf(FSUB(smc[k], sice[k]), 0.0f);
    }
    runoff3 = wplus;
    if (update_cmc) {
        cmc = FADD(cmc, FMUL(dt, rhsct));
        if (cmc < 1.0e-20f) cmc = 0.0f;
        cmc = fminf(cmc, cmcmax);
    }
}

// ---------------------------------------------------------------- SMFLX
// module_sf_noahlsm.F:2669
__device__ static void noah_smflx(real* smc, real& cmc, real dt,
                                  real prcp1, const real* zsoil,
                                  real* sh2o, real slope, real kdt,
                                  real frzfact, real smcmax, real bexp,
                                  real smcwlt, real dksat, real dwsat,
                                  real shdfac, real cmcmax, real edir,
                                  real ec, const real* et,
                                  real& runoff1, real& runoff2,
                                  real& runoff3, real& drip)
{
    real rhsct = FSUB(FMUL(shdfac, prcp1), ec);
    drip = 0.0f;
    real trhsct = FMUL(dt, rhsct);
    real excess = FADD(cmc, trhsct);
    if (excess > cmcmax) drip = FSUB(excess, cmcmax);
    real pcpdrp = FADD(FMUL((FSUB(1.0f, shdfac)), prcp1), FDIV(drip, dt));
    real sice[NSOIL];
    for (int i = 0; i < NSOIL; ++i) sice[i] = FSUB(smc[i], sh2o[i]);
    real fac2 = 0.0f;
    for (int i = 0; i < NSOIL; ++i)
        fac2 = fmaxf(fac2, FDIV(sh2o[i], smcmax));
    real flimit = noah_fac2mit(smcmax);
    real rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];
    if ((FMUL(pcpdrp, dt)) > (FMUL(FMUL(__uint_as_float(0x3dccccccu) /* gfortran15.2 fold_005 */, (-zsoil[0])), smcmax))
        || (fac2 > flimit)) {
        real sh2ofg[NSOIL], sh2oa[NSOIL], sh2onew[NSOIL];
        real dummy = 0.0f;
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2ofg, sh2o, dummy, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   false);
        for (int k = 0; k < NSOIL; ++k)
            sh2oa[k] = FMUL((FADD(sh2o[k], sh2ofg[k])), 0.5f);
        noah_srt(rhstt, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < NSOIL; ++k) sh2o[k] = sh2onew[k];
    } else {
        real sh2onew[NSOIL];
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < NSOIL; ++k) sh2o[k] = sh2onew[k];
    }
}

// ------------------------------------------------------------------ TBND
// module_sf_noahlsm.F:4080
__device__ static real noah_tbnd(real tu, real tb, const real* zsoil,
                                 real zbot, int k)
{
    real zup, zb;
    if (k == 0) zup = 0.0f;
    else zup = zsoil[k - 1];
    if (k == NSOIL - 1) zb = FSUB(FMUL(2.0f, zbot), zsoil[k]);
    else zb = zsoil[k + 1];
    return FADD(tu, FDIV(FMUL((FSUB(tb, tu)), (FSUB(zup, zsoil[k]))), (FSUB(zup, zb))));
}

// ---------------------------------------------------------------- TMPAVG
// module_sf_noahlsm.F:4250
__device__ static real noah_tmpavg(real tup, real tm, real tdn,
                                   const real* zsoil, int k)
{
    const real t0 = 2.7315e2f;
    real dz;
    if (k == 0) dz = -zsoil[0];
    else dz = FSUB(zsoil[k - 1], zsoil[k]);
    real dzh = FMUL(dz, 0.5f);
    if (tup < t0) {
        if (tm < t0) {
            if (tdn < t0) {
                return FDIV((FADD(FADD(tup, FMUL(2.0f, tm)), tdn)), 4.0f);
            }
            real x0 = FDIV(FMUL((FSUB(t0, tm)), dzh), (FSUB(tdn, tm)));
            return FDIV(FMUL(0.5f, (FADD(FADD(FMUL(tup, dzh), FMUL(tm, (FADD(dzh, x0)))), FMUL(t0, (FSUB(FMUL(2.0f, dzh), x0)))))), dz);
        }
        if (tdn < t0) {
            real xup = FDIV(FMUL((FSUB(t0, tup)), dzh), (FSUB(tm, tup)));
            real xdn = FSUB(dzh, FDIV(FMUL((FSUB(t0, tm)), dzh), (FSUB(tdn, tm))));
            return FDIV(FMUL(0.5f, (FADD(FADD(FMUL(tup, xup), FMUL(t0, (FSUB(FSUB(FMUL(2.0f, dz), xup), xdn)))), FMUL(tdn, xdn)))), dz);
        }
        real xup = FDIV(FMUL((FSUB(t0, tup)), dzh), (FSUB(tm, tup)));
        return FDIV(FMUL(0.5f, (FADD(FMUL(tup, xup), FMUL(t0, (FSUB(FMUL(2.0f, dz), xup)))))), dz);
    }
    if (tm < t0) {
        if (tdn < t0) {
            real xup = FSUB(dzh, FDIV(FMUL((FSUB(t0, tup)), dzh), (FSUB(tm, tup))));
            return FDIV(FMUL(0.5f, (FADD(FADD(FMUL(t0, (FSUB(dz, xup))), FMUL(tm, (FADD(dzh, xup)))), FMUL(tdn, dzh)))), dz);
        }
        real xup = FSUB(dzh, FDIV(FMUL((FSUB(t0, tup)), dzh), (FSUB(tm, tup))));
        real xdn = FDIV(FMUL((FSUB(t0, tm)), dzh), (FSUB(tdn, tm)));
        return FDIV(FMUL(0.5f, (FADD(FMUL(t0, (FSUB(FSUB(FMUL(2.0f, dz), xup), xdn))), FMUL(tm, (FADD(xup, xdn)))))), dz);
    }
    if (tdn < t0) {
        real xdn = FSUB(dzh, FDIV(FMUL((FSUB(t0, tm)), dzh), (FSUB(tdn, tm))));
        return FDIV((FADD(FMUL(t0, (FSUB(dz, xdn))), FMUL(FMUL(0.5f, (FADD(t0, tdn))), xdn))), dz);
    }
    return FDIV((FADD(FADD(tup, FMUL(2.0f, tm)), tdn)), 4.0f);
}

// ---------------------------------------------------------------- SNKSRC
// module_sf_noahlsm.F:2922
__device__ static real noah_snksrc(real qtot, real tavg, real smc,
                                   real& sh2o, const real* zsoil,
                                   real smcmax, real psisat, real bexp,
                                   real dt, int k)
{
    const real dh2o = 1.0000e3f, hlice = 3.3350e5f;
    real dz;
    if (k == 0) dz = -zsoil[0];
    else dz = FSUB(zsoil[k - 1], zsoil[k]);
    real freew = noah_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    real xh2o = FADD(sh2o, FDIV(FMUL(qtot, dt), (FMUL(__uint_as_float(0x4d9f0673u) /* gfortran15.2 fold_006 */, dz))));
    if (xh2o < sh2o && xh2o < freew) {
        if (freew > sh2o) xh2o = sh2o;
        else xh2o = freew;
    }
    if (xh2o > sh2o && xh2o > freew) {
        if (freew < sh2o) xh2o = sh2o;
        else xh2o = freew;
    }
    if (xh2o < 0.0f) xh2o = 0.0f;
    if (xh2o > smc) xh2o = smc;
    real tsnsr = FDIV(FMUL(FMUL(__uint_as_float(0xcd9f0673u) /* gfortran15.2 fold_007 */, dz), (FSUB(xh2o, sh2o))), dt);
    sh2o = xh2o;
    return tsnsr;
}

// ------------------------------------------------------------------ HRT
// module_sf_noahlsm.F:1588
__device__ static void noah_hrt(real* rhsts, const real* stc,
                                const real* smc, real smcmax,
                                const real* zsoil, real yy, real zz1,
                                real tbot, real zbot, real psisat,
                                real* sh2o, real dt, real bexp,
                                int soiltyp, int opt_thcnd, real df1,
                                real quartz, real csoil, int vegtyp,
                                int isurban, real* ai, real* bi,
                                real* ci)
{
    const real t0 = 273.15f, cair = 1004.0f, cice = 2.106e6f,
               ch2o = 4.2e6f;
    real csoil_loc;
    if (vegtyp == isurban) csoil_loc = 3.0e6f;
    else csoil_loc = csoil;
    real hcpct = FADD(FADD(FADD(FMUL(sh2o[0], ch2o), FMUL((FSUB(1.0f, smcmax)), csoil_loc)), FMUL((FSUB(smcmax, smc[0])), cair)), FMUL((FSUB(smc[0], sh2o[0])), cice));
    real ddz = FDIV(1.0f, (FMUL(-0.5f, zsoil[1])));
    ai[0] = 0.0f;
    ci[0] = FDIV((FMUL(df1, ddz)), (FMUL(zsoil[0], hcpct)));
    bi[0] = FADD(-ci[0], FDIV(df1, (FMUL(FMUL(FMUL(FMUL(0.5f, zsoil[0]), zsoil[0]), hcpct), zz1))));
    real dtsdz = FDIV((FSUB(stc[0], stc[1])), (FMUL(-0.5f, zsoil[1])));
    real ssoil = FDIV(FMUL(df1, (FSUB(stc[0], yy))), (FMUL(FMUL(0.5f, zsoil[0]), zz1)));
    real denom = FMUL(zsoil[0], hcpct);
    rhsts[0] = FDIV((FSUB(FMUL(df1, dtsdz), ssoil)), denom);
    real qtot = FMUL(FMUL(-1.0f, rhsts[0]), denom);
    real sice = FSUB(smc[0], sh2o[0]);
    real tsurf = FDIV((FADD(yy, FMUL((FSUB(zz1, 1.0f)), stc[0]))), zz1);
    real tbk = noah_tbnd(stc[0], stc[1], zsoil, zbot, 0);
    if (sice > 0.0f || stc[0] < t0 || tsurf < t0 || tbk < t0) {
        real tavg = noah_tmpavg(tsurf, stc[0], tbk, zsoil, 0);
        real tsnsr = noah_snksrc(qtot, tavg, smc[0], sh2o[0], zsoil,
                                 smcmax, psisat, bexp, dt, 0);
        rhsts[0] = FSUB(rhsts[0], FDIV(tsnsr, denom));
    }
    real ddz2 = 0.0f;
    real df1k = df1;
    real dtsdz2;
    for (int k = 1; k < NSOIL; ++k) {
        hcpct = FADD(FADD(FADD(FMUL(sh2o[k], ch2o), FMUL((FSUB(1.0f, smcmax)), csoil_loc)), FMUL((FSUB(smcmax, smc[k])), cair)), FMUL((FSUB(smc[k], sh2o[k])), cice));
        real df1n, tbk1;
        if (k != NSOIL - 1) {
            df1n = noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                               psisat, soiltyp, opt_thcnd);
            if (vegtyp == isurban) df1n = 3.24f;
            denom = FMUL(0.5f, (FSUB(zsoil[k - 1], zsoil[k + 1])));
            dtsdz2 = FDIV((FSUB(stc[k], stc[k + 1])), denom);
            ddz2 = FDIV(2.0f, (FSUB(zsoil[k - 1], zsoil[k + 1])));
            ci[k] = FDIV(FMUL(-df1n, ddz2), (FMUL((FSUB(zsoil[k - 1], zsoil[k])), hcpct)));
            tbk1 = noah_tbnd(stc[k], stc[k + 1], zsoil, zbot, k);
        } else {
            df1n = noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                               psisat, soiltyp, opt_thcnd);
            if (vegtyp == isurban) df1n = 3.24f;
            denom = FSUB(FMUL(0.5f, (FADD(zsoil[k - 1], zsoil[k]))), zbot);
            dtsdz2 = FDIV((FSUB(stc[k], tbot)), denom);
            ci[k] = 0.0f;
            tbk1 = noah_tbnd(stc[k], tbot, zsoil, zbot, k);
        }
        denom = FMUL((FSUB(zsoil[k], zsoil[k - 1])), hcpct);
        rhsts[k] = FDIV((FSUB(FMUL(df1n, dtsdz2), FMUL(df1k, dtsdz))), denom);
        qtot = FMUL(FMUL(-1.0f, denom), rhsts[k]);
        sice = FSUB(smc[k], sh2o[k]);
        real tavg = noah_tmpavg(tbk, stc[k], tbk1, zsoil, k);
        if (sice > 0.0f || stc[k] < t0 || tbk < t0 || tbk1 < t0) {
            real tsnsr = noah_snksrc(qtot, tavg, smc[k], sh2o[k],
                                     zsoil, smcmax, psisat, bexp, dt,
                                     k);
            rhsts[k] = FSUB(rhsts[k], FDIV(tsnsr, denom));
        }
        ai[k] = FDIV(FMUL(-df1k, ddz), (FMUL((FSUB(zsoil[k - 1], zsoil[k])), hcpct)));
        bi[k] = -(FADD(ai[k], ci[k]));
        tbk = tbk1;
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

// ---------------------------------------------------------------- HSTEP
// module_sf_noahlsm.F:1853
__device__ static void noah_hstep(real* stcout, const real* stcin,
                                  const real* rhsts_in, real dt,
                                  const real* ai_in, const real* bi_in,
                                  const real* ci_in)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], p[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhsts[k] = FMUL(rhsts_in[k], dt);
        ai[k] = FMUL(ai_in[k], dt);
        bi[k] = FADD(1.0f, FMUL(bi_in[k], dt));
        ci[k] = FMUL(ci_in[k], dt);
    }
    noah_rosr12(p, ai, bi, ci, rhsts);
    for (int k = 0; k < NSOIL; ++k) stcout[k] = FADD(stcin[k], p[k]);
}

// ---------------------------------------------------------------- SHFLX
// module_sf_noahlsm.F:2600
__device__ static void noah_shflx(real* stc, const real* smc,
                                  real smcmax, real& t1, real dt,
                                  real yy, real zz1, const real* zsoil,
                                  real tbot, real zbot, real psisat,
                                  real* sh2o, real bexp, real df1,
                                  real quartz, real csoil, int vegtyp,
                                  int isurban, int soiltyp,
                                  int opt_thcnd, real& ssoil)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], stcf[NSOIL];
    noah_hrt(rhsts, stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot,
             psisat, sh2o, dt, bexp, soiltyp, opt_thcnd, df1, quartz,
             csoil, vegtyp, isurban, ai, bi, ci);
    noah_hstep(stcf, stc, rhsts, dt, ai, bi, ci);
    for (int i = 0; i < NSOIL; ++i) stc[i] = stcf[i];
    t1 = FDIV((FADD(yy, FMUL((FSUB(zz1, 1.0f)), stc[0]))), zz1);
    ssoil = FDIV(FMUL(df1, (FSUB(stc[0], t1))), (FMUL(0.5f, zsoil[0])));
}

// -------------------------------------------------------------- SNOWPACK
// module_sf_noahlsm.F:3417
__device__ static void noah_snowpack(real esd, real dtsec, real& snowh,
                                     real& sndens, real tsnow,
                                     real tsoil
#if NOAH_UA_PHYS
,real snomlt=-1.0f
#endif
)
{
    const real c1k = 0.01f, c2k = 21.0f;
    real snowhc = FMUL(snowh, 100.0f);
    real esdc = FMUL(esd, 100.0f);
    real dthr = FDIV(dtsec, 3600.0f);
    real tsnowc = FSUB(tsnow, 273.15f);
    real tsoilc = FSUB(tsoil, 273.15f);
    real tavgc = FMUL(0.5f, (FADD(tsnowc, tsoilc)));
    real esdcx;
    if (esdc > 1.0e-2f) esdcx = esdc;
    else esdcx = 1.0e-2f;
    real bfac = FMUL(FMUL(dthr, c1k), gfk_exp(FSUB(FMUL(0.08f, tavgc), FMUL(c2k, sndens))));
    const int ipol = 4;
    real pexp = 0.0f;
    for (int j = ipol; j >= 1; --j)
        pexp = FDIV(FMUL(FMUL((FADD(1.0f, pexp)), bfac), esdcx), (real)(j + 1));
    pexp = FADD(pexp, 1.0f);
    real dsx = FMUL(sndens, pexp);
    if (dsx > 0.40f) dsx = 0.40f;
    if (dsx < 0.05f) dsx = 0.05f;
    sndens = dsx;
    if (tsnowc >= 0.0f) {
        real dw = FDIV(FMUL(0.13f, dthr), 24.0f);

#if NOAH_UA_PHYS
        if(tsoilc>=0.0f && snomlt>=0.0f)dw=noah_ua_retention(dw,snomlt,esdcx);
#endif
        sndens = FADD(FMUL(sndens, (FSUB(1.0f, dw))), dw);
        if (sndens >= 0.40f) sndens = 0.40f;
    }
    snowhc = FDIV(esdc, sndens);
    snowh = FMUL(snowhc, 0.01f);
}

// ---------------------------------------------------------------- kernel
struct MosaicState {
#if NOAH_UA_PHYS
 real flx4,fvb,fbur,fgsn;
#endif

#if NOAH_MOSAIC_UCM
 bool urban_tile; int natural; real rural_t1, rural_q1, rural_initial;
#endif
 int ivgtyp,isltyp;
 real psfc;
 real sfcprs;
 real sfctmp;
 real qv1;
 real qgh;
 real dz8w1;
 real glw;
 real swdown;
 real rainbl;
 real sr;
 real chs;
 real cqs2;
 real chs2;
 real rib;
 real vegfra;
 real shdmin;
 real shdmax;
 real tmn;
 real xland;
 real xice;
 real snoalb;
 real embck;
 real tsk;
 real hfx;
 real qfx;
 real lh;
 real grdflx;
 real qsfc;
 real canwat;
 real snow;
 real snowc;
 real snowh;
 real albedo;
 real albbck;
 real emiss;
 real znt;
 real z0;
 real snotime;
 real lai;
 real smstav;
 real smstot;
 real sfcrunoff;
 real udrunoff;
 real acsnow;
 real acsnom;
 real snopcx;
 real potevp;
 real noahres;
 real chklowq;
 real smois[4];
 real tslb[4];
 real sh2o[4];
 real smcrel[4];
};
// module_sf_noahlsm_glacial_only.F:565-707
__device__ static void mosaic_hrtice(real* rhs,const real* stc,real tbot,
 const real* zsoil,real yy,real zz1,real df1,real* ai,real* bi,real* ci) {
 real hcpct=FMUL(1.e6f, (FSUB(0.8194f, FMUL(__uint_as_float(0x3d860aa6u) /* gfortran15.2 fold_008 */, zsoil[0]))));
 real df1k=df1,ddz=FDIV(1.0f, (FMUL(-0.5f, zsoil[1])));
 ai[0]=0.0f;ci[0]=FDIV((FMUL(df1, ddz)), (FMUL(zsoil[0], hcpct)));
 bi[0]=FADD(-ci[0], FDIV(df1, (FMUL(FMUL(FMUL(FMUL(0.5f, zsoil[0]), zsoil[0]), hcpct), zz1))));
 real dtsdz=FDIV((FSUB(stc[0], stc[1])), (FMUL(-0.5f, zsoil[1])));
 real ssoil=FDIV(FMUL(df1, (FSUB(stc[0], yy))), (FMUL(FMUL(0.5f, zsoil[0]), zz1)));
 rhs[0]=FDIV((FSUB(FMUL(df1, dtsdz), ssoil)), (FMUL(zsoil[0], hcpct)));
 real ddz2=0.0f,df1n=df1;
 for(int k=1;k<4;++k) {
  real zmd=FMUL(0.5f, (FADD(zsoil[k], zsoil[k-1])));
  hcpct=FMUL(1.e6f, (FSUB(0.8194f, FMUL(0.1309f, zmd))));
  df1n=FSUB(0.32333f, (FMUL(0.10073f, zmd)));
  real dtsdz2,denom;
  if(k!=3) {
   denom=FMUL(0.5f, (FSUB(zsoil[k-1], zsoil[k+1])));
   dtsdz2=FDIV((FSUB(stc[k], stc[k+1])), denom);
   ddz2=FDIV(2.0f, (FSUB(zsoil[k-1], zsoil[k+1])));
   ci[k]=FDIV(FMUL(-df1n, ddz2), (FMUL((FSUB(zsoil[k-1], zsoil[k])), hcpct)));
  } else {
   dtsdz2=FDIV((FSUB(stc[k], tbot)), (FSUB(FMUL(0.5f, (FADD(zsoil[k-1], zsoil[k]))), (-25.0f))));
   ci[k]=0.0f;
  }
  denom=FMUL((FSUB(zsoil[k], zsoil[k-1])), hcpct);
  rhs[k]=FDIV((FSUB(FMUL(df1n, dtsdz2), FMUL(df1k, dtsdz))), denom);
  ai[k]=FDIV(FMUL(-df1k, ddz), (FMUL((FSUB(zsoil[k-1], zsoil[k])), hcpct)));
  bi[k]=-(FADD(ai[k], ci[k]));df1k=df1n;dtsdz=dtsdz2;ddz=ddz2;
 }
}
// module_sf_noahlsm_glacial_only.F:824-853, HSTEP:710-754 calls Noah ROSR12
__device__ static void mosaic_shflx_ice(real* stc,real dt,real yy,real zz1,
 const real* zsoil,real tbot,real df1) {
 real rhs[4],ai[4],bi[4],ci[4],stcf[4];
 mosaic_hrtice(rhs,stc,tbot,zsoil,yy,zz1,df1,ai,bi,ci);
 noah_hstep(stcf,stc,rhs,dt,ai,bi,ci);
 for(int k=0;k<4;++k)stc[k]=stcf[k];
}
// module_sf_noahdrv.F:3251-3729; module_sf_noahlsm_glacial_only.F:32-409
__device__ static void noah_mosaic_tile_glacial(MosaicState& s,
 const real* vegtbl,const real* soiltbl,const real* genp,const real* dzs,
 real dt,int lucats,int slcats,int isurban,int isice,real xice_threshold,
 int itimestep,int frpcpn,int usemonalb,int rdlai2d,int opt_thcnd)
{
 const size_t idx=0,plane=1;
 const int* ivgtyp=&s.ivgtyp; const int* isltyp=&s.isltyp;
 real* psfc_a=&s.psfc;
 real* sfcprs_a=&s.sfcprs;
 real* sfctmp_a=&s.sfctmp;
 real* qv1_a=&s.qv1;
 real* qgh_a=&s.qgh;
 real* dz8w1_a=&s.dz8w1;
 real* glw_a=&s.glw;
 real* swdown_a=&s.swdown;
 real* rainbl_a=&s.rainbl;
 real* sr_a=&s.sr;
 real* chs_a=&s.chs;
 real* cqs2_a=&s.cqs2;
 real* chs2_a=&s.chs2;
 real* rib_a=&s.rib;
 real* vegfra_a=&s.vegfra;
 real* shdmin_a=&s.shdmin;
 real* shdmax_a=&s.shdmax;
 real* tmn_a=&s.tmn;
 real* xland_a=&s.xland;
 real* xice_a=&s.xice;
 real* snoalb_a=&s.snoalb;
 real* embck_a=&s.embck;
 real* tsk_a=&s.tsk;
 real* hfx_a=&s.hfx;
 real* qfx_a=&s.qfx;
 real* lh_a=&s.lh;
 real* grdflx_a=&s.grdflx;
 real* qsfc_a=&s.qsfc;
 real* canwat_a=&s.canwat;
 real* snow_a=&s.snow;
 real* snowc_a=&s.snowc;
 real* snowh_a=&s.snowh;
 real* albedo_a=&s.albedo;
 real* albbck_a=&s.albbck;
 real* emiss_a=&s.emiss;
 real* znt_a=&s.znt;
 real* z0_a=&s.z0;
 real* snotime_a=&s.snotime;
 real* lai_a=&s.lai;
 real* smstav_a=&s.smstav;
 real* smstot_a=&s.smstot;
 real* sfcrunoff_a=&s.sfcrunoff;
 real* udrunoff_a=&s.udrunoff;
 real* acsnow_a=&s.acsnow;
 real* acsnom_a=&s.acsnom;
 real* snopcx_a=&s.snopcx;
 real* potevp_a=&s.potevp;
 real* noahres_a=&s.noahres;
 real* chklowq_a=&s.chklowq;
 real* smois_a=s.smois;
 real* tslb_a=s.tslb;
 real* sh2o_a=s.sh2o;
 real* smcrel_a=s.smcrel;
    // ================= driver (module_sf_noahdrv.F lsm) prep =========
    // module_sf_noahdrv.F:809-814 writes CHKLOWQ for EVERY column, before the
    // XLAND land/sea branch in mosaic prep -- so open water, sea ice and land ice all
    // leave the driver with CHKLOWQ = 1.  This kernel used to return from the
    // three skip paths below without writing it, leaving whatever the caller
    // had in the array; the WRF oracle's water and sea-ice rows found it (0.0
    // against 1.0).  The `myj` arm that can set 0 instead is not ported --
    // launch_noah has no myj argument -- so the value is unconditionally 1.
    chklowq_a[idx] = 1.0f;
    if ((FSUB(xland_a[idx], 1.5f)) >= 0.0f) return;        // open water
    if (xice_a[idx] >= xice_threshold) {              // sea ice
        for (int k = 0; k < NSOIL; ++k) sh2o_a[k * plane + idx] = 1.0f;
        lai_a[idx] = 0.01f;
        return;
    }
    int vegtyp = ivgtyp[idx];
    int soiltyp = isltyp[idx];


    const real tresh = 0.95f, a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = __uint_as_float(0x45867529u) /* gfortran15.2 fold_009 */;
    real psfc = psfc_a[idx];
    real sfcprs = sfcprs_a[idx];
    real q2k = FDIV(qv1_a[idx], (FADD(1.0f, qv1_a[idx])));
    real q2sat = FDIV(qgh_a[idx], (FADD(1.0f, qgh_a[idx])));
    real sfctmp = sfctmp_a[idx];
    real apes = gfk_pow(FDIV(1.0e5f, psfc), RCP);             // CAPA = R_d/CP
    real apelm = gfk_pow(FDIV(1.0e5f, sfcprs), RCP);
    real sfcth2 = FMUL(sfctmp, apelm);
    real th2 = FDIV(sfcth2, apes);
    real emissi = emiss_a[idx];
    real lwdn = FMUL(glw_a[idx], emissi);
    real soldn = swdown_a[idx];
    real solnet = FMUL(soldn, (FSUB(1.0f, albedo_a[idx])));
    real prcp = FDIV(rainbl_a[idx], dt);
    real shdfac = FDIV(vegfra_a[idx], 100.0f);
    if (vegtyp == 25 || vegtyp == 26 || vegtyp == 27) shdfac = 0.0f;
    real t1 = tsk_a[idx];
    real chk = chs_a[idx];
    real shmin = FDIV(shdmin_a[idx], 100.0f);
    real shmax = FDIV(shdmax_a[idx], 100.0f);
    real sneqv = FMUL(snow_a[idx], 0.001f);
    real snowhk = snowh_a[idx];
    real sncovr = snowc_a[idx];
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;

    real dqsdt2 = FDIV(FMUL(q2sat, a23m4), (FMUL((FSUB(sfctmp, a4)), (FSUB(sfctmp, a4)))));
    if (snow_a[idx] > 0.0f) {
        real sfctsno = sfctmp;
        real e2sat = FMUL(611.2f, gfk_exp(FMUL(6174.0f, (FSUB(__uint_as_float(0x3b6fed42u) /* gfortran15.2 fold_011 */, FDIV(1.0f, sfctsno))))));
        real q2sati = FDIV(FMUL(0.622f, e2sat), (FSUB(sfcprs, e2sat)));
        q2sati = FDIV(q2sati, (FADD(1.0f, q2sati)));
        if (t1 > 273.14f) {
            q2sat = FADD(FMUL(q2sat, (FSUB(1.0f, snowc_a[idx]))), FMUL(q2sati, snowc_a[idx]));
            dqsdt2 = FADD(FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx]))), FMUL(FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno))), snowc_a[idx]));
        } else {
            q2sat = q2sati;
            dqsdt2 = FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno)));
        }
        if (t1 > 273.0f && snowc_a[idx] > 0.0f)
            dqsdt2 = FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx])));
    }
    real tbot = tmn_a[idx];
    if (soiltyp == 14 && xice_a[idx] == 0.0f) soiltyp = 7;
    real snoalb1 = snoalb_a[idx];
    real cmc = FDIV(canwat_a[idx], 1000.0f);
    real alb = albbck_a[idx];
    real z0brd = z0_a[idx];
    real embrd = embck_a[idx];
    real snotime1 = snotime_a[idx];
    real ribb = rib_a[idx];
    real smc[NSOIL], stc[NSOIL], swc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = smois_a[k * plane + idx];
        stc[k] = tslb_a[k * plane + idx];
        swc[k] = sh2o_a[k * plane + idx];
    }
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = FMUL(5.0f, sneqv);
    real xlai = lai_a[idx];
    if (rdlai2d) {
        if (shdfac > 0.0f && xlai <= 0.0f) xlai = 0.01f;
    }

 // module_sf_noahlsm_glacial_only.F:208-406, SFLX_GLACIAL
 real zsoil[4];zsoil[0]=-dzs[0];
 for(int k=1;k<4;++k)zsoil[k]=FADD(-dzs[k], zsoil[k-1]);
 if(sneqv<0.10f){sneqv=0.10f;snowhk=0.50f;}
 real sndens=FDIV(sneqv, snowhk);
 real sncond=noah_csnow(sndens);
 bool snowng=false,frzgra=false;
 if(prcp>0.0f){if(ffrozp>0.5f)snowng=true;else if(t1<=NTFREEZ)frzgra=true;}
 real prcpf;
 if(snowng||frzgra){
  real sn_new=FMUL(FMUL(prcp, dt), 0.001f);sneqv=FADD(sneqv, sn_new);prcpf=0.0f;
  noah_snow_new(sfctmp,sn_new,snowhk,sndens);
  if(sncovr>0.99f){
   if(stc[0]<(__uint_as_float(0x43861333u) /* gfortran15.2 fold_012 */))sndens=0.2f;
   if(snowng && t1<273.0f && sfctmp<273.0f)sndens=0.2f;
  }
  sncond=noah_csnow(sndens);
 } else prcpf=prcp;
 sncovr=1.0f;
 real albedo;
 // glacial ALCALC:412-523 has the same aging expression, SNCOVR fixed at 1.
 noah_alcalc(alb,snoalb1,embrd,1.0f,dt,snowng,snotime1,genp[GEN_LVCOEF],albedo,emissi);
 real df1=sncond;
 real dsoil=-(FMUL(0.5f, zsoil[0]));
 real dtot=FADD(snowhk, dsoil);
 real frcsno=FDIV(snowhk, dtot),frcsoi=FDIV(dsoil, dtot);
 df1=FADD(FMUL(frcsno, sncond), FMUL(frcsoi, df1));
 if(dtot>FMUL(2.0f, dsoil))dtot=FMUL(2.0f, dsoil);
 real ssoil=FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
 // glacial SNOWZ0:1202-1226: complete cover, no ordinary-land blend.
 real burial=FSUB(FMUL(7.0f, z0brd), snowhk);
 real z0k=burial<=0.0007f?0.001f:FDIV(burial, 7.0f);
 real fdown=FADD(solnet, lwdn);
 real t2v=FMUL(sfctmp, (FADD(1.0f, FMUL(0.61f, q2k))));
 real rho=FDIV(sfcprs, (FMUL(NRD, t2v)));
 real rch=FMUL(FMUL(rho, 1004.6f), chk);
 real t24=FMUL(FMUL(FMUL(sfctmp, sfctmp), sfctmp), sfctmp);
 // glacial PENMAN:756-821 differs from ordinary PENMAN's ELCP1 choice.
 real elcp1,lvs;
 if(t1>273.15f){elcp1=NELCP;lvs=NLSUBC;}
 else {elcp1=__uint_as_float(0x4530031fu) /* gfortran15.2 fold_013 */;lvs=NLSUBS;}
 real delta=FMUL(elcp1, dqsdt2);
 real a=FMUL(elcp1, (FSUB(q2sat, q2k)));
 real rr=FADD(FDIV(FMUL(FMUL(emissi, t24), 6.48e-8f), (FMUL(sfcprs, chk))), 1.0f);
 if(!snowng){if(prcp>0.0f)rr=FADD(rr, FDIV(FMUL(NCPH2O, prcp), rch));}
 else rr=FADD(rr, FDIV(FMUL(NCPICE, prcp), rch));
 real flx2=frzgra?FMUL(-NLSUBF, prcp):0.0f;
 real fnet=FSUB(FSUB(FSUB(fdown, (FMUL(FMUL(emissi, NSIGMA), t24))), ssoil), flx2);
 real rad=FSUB(FADD(FDIV(fnet, rch), th2), sfctmp);
 real epsca=FDIV((FADD(FMUL(a, rr), FMUL(rad, delta))), (FADD(delta, rr)));
 real etp=FDIV(FMUL(epsca, rch), lvs);
 // glacial SNOPAC:856-1073
 real snomlt=0.0f,dew=0.0f,esnow=0.0f,esnow1=0.0f,esnow2=0.0f;
 real etanrg;
 if(etp<=0.0f){
  if(ribb>=0.1f && fdown>150.0f)
   etp=FDIV((FADD(FDIV(fminf(FMUL(etp, (FSUB(1.0f, ribb))),0.0f), 0.980f), FMUL(etp, (__uint_as_float(0xbca3d700u) /* gfortran15.2 fold_015 */)))), 0.980f);
  real etp1=FMUL(etp, 0.001f);dew=-etp1;esnow2=FMUL(etp1, dt);etanrg=FMUL(etp, NLSUBS);
 } else {esnow=etp;esnow1=FMUL(esnow, 0.001f);esnow2=FMUL(esnow1, dt);etanrg=FMUL(esnow, NLSUBS);}
 real flx1=0.0f;
 if(snowng)flx1=FMUL(FMUL(NCPICE, prcp), (FSUB(t1, sfctmp)));
 else if(prcp>0.0f)flx1=FMUL(FMUL(NCPH2O, prcp), (FSUB(t1, sfctmp)));
 dsoil=-(FMUL(0.5f, zsoil[0]));dtot=FADD(snowhk, dsoil);
 real denom=FADD(1.0f, FDIV(df1, (FMUL(FMUL(dtot, rr), rch))));
 real t12a=FDIV((FSUB(FSUB(FADD(FDIV((FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t24))), rch), th2), sfctmp), FDIV(etanrg, rch))), rr);
 real t12b=FDIV(FMUL(df1, stc[0]), (FMUL(FMUL(dtot, rr), rch)));
 real t12=FDIV((FADD(FADD(sfctmp, t12a), t12b)), denom);
 real flx3=0.0f,ex=0.0f;
 if(t12<=NTFREEZ){
  t1=t12;ssoil=FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
  sneqv=fmaxf(0.0f,FSUB(sneqv, esnow2));flx3=0.0f;ex=0.0f;snomlt=0.0f;
 } else {
  t1=NTFREEZ;if(dtot>FMUL(2.0f, dsoil))dtot=FMUL(2.0f, dsoil);
  ssoil=FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
  if(FSUB(sneqv, esnow2)<=1.e-6f){sneqv=0.0f;ex=0.0f;snomlt=0.0f;flx3=0.0f;}
  else {
   sneqv=FSUB(sneqv, esnow2);
   real seh=FMUL(rch, (FSUB(t1, th2)));
   real t14=FMUL((FMUL(t1, t1)), (FMUL(t1, t1)));
   flx3=FSUB(FSUB(FSUB(FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t14)), ssoil), seh), etanrg);
   if(flx3<=0.0f)flx3=0.0f;
   ex=FDIV(FMUL(flx3, 0.001f), NLSUBF);snomlt=FMUL(ex, dt);
   if(FSUB(sneqv, snomlt)>=1.e-6f)sneqv=FSUB(sneqv, snomlt);
   else {ex=FDIV(sneqv, dt);flx3=FMUL(FMUL(ex, 1000.0f), NLSUBF);snomlt=sneqv;sneqv=0.0f;}
  }
 }
 real zz1=1.0f;
 real yy=FSUB(stc[0], FDIV(FMUL(FMUL(FMUL(0.5f, ssoil), zsoil[0]), zz1), df1));
 mosaic_shflx_ice(stc,dt,yy,zz1,zsoil,tbot,df1);
 // glacial SNOWPACK:1076-1199 shares the ordinary routine's expression order.
 if(sneqv>=0.10f)noah_snowpack(sneqv,dt,snowhk,sndens,t1,yy);
 else {sneqv=0.10f;snowhk=0.50f;}
 real eta_kin=etp;
 real q1=FADD(q2k, FDIV(FMUL(eta_kin, CP), rch));
 real sheat=FMUL(FDIV(-(FMUL(FMUL(chk, CP), sfcprs)), (FMUL(NR_SHEAT, t2v))), (FSUB(th2, t1)));
 real esnow_e=FMUL(esnow, NLSUBS),etp_e=FMUL(etp, NLSUBS);
 real eta_e=etp_e>0.0f?esnow_e:etp_e;
 real ssoil_out=FMUL(-1.0f, ssoil);
 real runoff1=FDIV(snomlt, dt);
 // noahdrv:3605-3729 and 4062-4080. SOILW retains its previous tile value.
 s.lai=0.01f;s.canwat=FMUL(cmc, 1000.0f);s.snow=FMUL(sneqv, 1000.0f);s.snowh=snowhk;
 s.albedo=albedo;s.albbck=alb;s.z0=z0brd;s.emiss=emissi;s.znt=z0k;
 s.tsk=t1;s.hfx=sheat;s.potevp=FADD(s.potevp, FMUL(etp_e, (FDIV(dt, (__uint_as_float(0x4f1502f9u) /* gfortran15.2 fold_016 */)))));
 s.qfx=eta_kin;s.lh=eta_e;s.grdflx=ssoil_out;s.snowc=sncovr;
 s.chs2=s.cqs2;s.snotime=snotime1;
 if(q1>s.qsfc)s.cqs2=s.chs;
 s.qsfc=FDIV(q1, (FSUB(1.0f, q1)));
 for(int k=0;k<4;++k){s.smois[k]=1.0f;s.sh2o[k]=1.0f;s.smcrel[k]=1.0f;s.tslb[k]=stc[k];}
 real t14=FMUL((FMUL(t1, t1)), (FMUL(t1, t1)));
 s.noahres=FSUB(FSUB(FSUB(FSUB(FSUB(FADD(FSUB((FADD(solnet, lwdn)), sheat), ssoil_out), eta_e), (FMUL(FMUL(emissi, NSTBOLT), t14))), flx1), flx2), flx3);
 s.smstot=0.0f;s.sfcrunoff=FADD(s.sfcrunoff, FMUL(FMUL(runoff1, dt), 1000.0f));
 if(ffrozp>0.5f)s.acsnow=FADD(s.acsnow, FMUL(prcp, dt));
 if(s.snow>0.0f){s.acsnom=FADD(s.acsnom, FMUL(snomlt, 1000.0f));s.snopcx=FSUB(s.snopcx, FDIV(snomlt, (FDIV(dt, __uint_as_float(0x4da6e49cu) /* gfortran ROWLIW=ROW*ELIW */))));}
}
// module_sf_noahdrv.F:3251-3729; module_sf_noahlsm.F:69-888
__device__ static void noah_mosaic_tile(MosaicState& s,
 const real* vegtbl,const real* soiltbl,const real* genp,const real* dzs,
 real dt,int lucats,int slcats,int isurban,int isice,real xice_threshold,
 int itimestep,int frpcpn,int usemonalb,int rdlai2d,int opt_thcnd
#if NOAH_UA_PHYS
,const real* canopy
#endif
)
{
 const size_t idx=0,plane=1;
 const int* ivgtyp=&s.ivgtyp; const int* isltyp=&s.isltyp;
 real* psfc_a=&s.psfc;
 real* sfcprs_a=&s.sfcprs;
 real* sfctmp_a=&s.sfctmp;
 real* qv1_a=&s.qv1;
 real* qgh_a=&s.qgh;
 real* dz8w1_a=&s.dz8w1;
 real* glw_a=&s.glw;
 real* swdown_a=&s.swdown;
 real* rainbl_a=&s.rainbl;
 real* sr_a=&s.sr;
 real* chs_a=&s.chs;
 real* cqs2_a=&s.cqs2;
 real* chs2_a=&s.chs2;
 real* rib_a=&s.rib;
 real* vegfra_a=&s.vegfra;
 real* shdmin_a=&s.shdmin;
 real* shdmax_a=&s.shdmax;
 real* tmn_a=&s.tmn;
 real* xland_a=&s.xland;
 real* xice_a=&s.xice;
 real* snoalb_a=&s.snoalb;
 real* embck_a=&s.embck;
 real* tsk_a=&s.tsk;
 real* hfx_a=&s.hfx;
 real* qfx_a=&s.qfx;
 real* lh_a=&s.lh;
 real* grdflx_a=&s.grdflx;
 real* qsfc_a=&s.qsfc;
 real* canwat_a=&s.canwat;
 real* snow_a=&s.snow;
 real* snowc_a=&s.snowc;
 real* snowh_a=&s.snowh;
 real* albedo_a=&s.albedo;
 real* albbck_a=&s.albbck;
 real* emiss_a=&s.emiss;
 real* znt_a=&s.znt;
 real* z0_a=&s.z0;
 real* snotime_a=&s.snotime;
 real* lai_a=&s.lai;
 real* smstav_a=&s.smstav;
 real* smstot_a=&s.smstot;
 real* sfcrunoff_a=&s.sfcrunoff;
 real* udrunoff_a=&s.udrunoff;
 real* acsnow_a=&s.acsnow;
 real* acsnom_a=&s.acsnom;
 real* snopcx_a=&s.snopcx;
 real* potevp_a=&s.potevp;
 real* noahres_a=&s.noahres;
 real* chklowq_a=&s.chklowq;
 real* smois_a=s.smois;
 real* tslb_a=s.tslb;
 real* sh2o_a=s.sh2o;
 real* smcrel_a=s.smcrel;
    // ================= driver (module_sf_noahdrv.F lsm) prep =========
    // module_sf_noahdrv.F:809-814 writes CHKLOWQ for EVERY column, before the
    // XLAND land/sea branch in mosaic prep -- so open water, sea ice and land ice all
    // leave the driver with CHKLOWQ = 1.  This kernel used to return from the
    // three skip paths below without writing it, leaving whatever the caller
    // had in the array; the WRF oracle's water and sea-ice rows found it (0.0
    // against 1.0).  The `myj` arm that can set 0 instead is not ported --
    // launch_noah has no myj argument -- so the value is unconditionally 1.
    chklowq_a[idx] = 1.0f;
    if ((FSUB(xland_a[idx], 1.5f)) >= 0.0f) return;        // open water
    if (xice_a[idx] >= xice_threshold) {              // sea ice
        for (int k = 0; k < NSOIL; ++k) sh2o_a[k * plane + idx] = 1.0f;
        lai_a[idx] = 0.01f;
        return;
    }
    int vegtyp = ivgtyp[idx];
    int soiltyp = isltyp[idx];


    const real tresh = 0.95f, a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = __uint_as_float(0x45867529u) /* gfortran15.2 fold_009 */;
    real psfc = psfc_a[idx];
    real sfcprs = sfcprs_a[idx];
    real q2k = FDIV(qv1_a[idx], (FADD(1.0f, qv1_a[idx])));
    real q2sat = FDIV(qgh_a[idx], (FADD(1.0f, qgh_a[idx])));
    real sfctmp = sfctmp_a[idx];
    real apes = gfk_pow(FDIV(1.0e5f, psfc), RCP);             // CAPA = R_d/CP
    real apelm = gfk_pow(FDIV(1.0e5f, sfcprs), RCP);
    real sfcth2 = FMUL(sfctmp, apelm);
    real th2 = FDIV(sfcth2, apes);
    real emissi = emiss_a[idx];
    real lwdn = FMUL(glw_a[idx], emissi);
    real soldn = swdown_a[idx];
    real solnet = FMUL(soldn, (FSUB(1.0f, albedo_a[idx])));
    real prcp = FDIV(rainbl_a[idx], dt);
    real shdfac = FDIV(vegfra_a[idx], 100.0f);
    if (vegtyp == 25 || vegtyp == 26 || vegtyp == 27) shdfac = 0.0f;
    real t1 = tsk_a[idx];
    real chk = chs_a[idx];
    real shmin = FDIV(shdmin_a[idx], 100.0f);
    real shmax = FDIV(shdmax_a[idx], 100.0f);
    real sneqv = FMUL(snow_a[idx], 0.001f);
    real snowhk = snowh_a[idx];
    real sncovr = snowc_a[idx];
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;

    real dqsdt2 = FDIV(FMUL(q2sat, a23m4), (FMUL((FSUB(sfctmp, a4)), (FSUB(sfctmp, a4)))));
    if (snow_a[idx] > 0.0f) {
        real sfctsno = sfctmp;
        real e2sat = FMUL(611.2f, gfk_exp(FMUL(6174.0f, (FSUB(__uint_as_float(0x3b6fed42u) /* gfortran15.2 fold_011 */, FDIV(1.0f, sfctsno))))));
        real q2sati = FDIV(FMUL(0.622f, e2sat), (FSUB(sfcprs, e2sat)));
        q2sati = FDIV(q2sati, (FADD(1.0f, q2sati)));
        if (t1 > 273.14f) {
            q2sat = FADD(FMUL(q2sat, (FSUB(1.0f, snowc_a[idx]))), FMUL(q2sati, snowc_a[idx]));
            dqsdt2 = FADD(FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx]))), FMUL(FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno))), snowc_a[idx]));
        } else {
            q2sat = q2sati;
            dqsdt2 = FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno)));
        }
        if (t1 > 273.0f && snowc_a[idx] > 0.0f)
            dqsdt2 = FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx])));
    }
    real tbot = tmn_a[idx];
    if (soiltyp == 14 && xice_a[idx] == 0.0f) soiltyp = 7;
    real snoalb1 = snoalb_a[idx];
    real cmc = FDIV(canwat_a[idx], 1000.0f);
    real alb = albbck_a[idx];
    real z0brd = z0_a[idx];
    real embrd = embck_a[idx];
    real snotime1 = snotime_a[idx];
    real ribb = rib_a[idx];
    real smc[NSOIL], stc[NSOIL], swc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = smois_a[k * plane + idx];
        stc[k] = tslb_a[k * plane + idx];
        swc[k] = sh2o_a[k * plane + idx];
    }
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = FMUL(5.0f, sneqv);
    real xlai = lai_a[idx];
    if (rdlai2d) {
        if (shdfac > 0.0f && xlai <= 0.0f) xlai = 0.01f;
    }

#if NOAH_MOSAIC_UCM
    // noahdrv.F:3455-3474. No FRC_URB2D test on the rural temperature.
    if(s.urban_tile) {
        vegtyp=s.natural;
        shdfac=vegtbl[(s.natural-1)*NVEGC+VG_SHDTBL];
        albedo_a[idx]=0.2f; alb=0.2f; emissi=0.98f;
        lwdn=FMUL(glw_a[idx],emissi);
        solnet=FMUL(soldn,FSUB(1.0f,albedo_a[idx]));
        t1=s.rural_initial;
    }
#endif
    // ======================== SFLX ===================================
    real sldpth[NSOIL], zsoil[NSOIL];
    for (int k = 0; k < NSOIL; ++k) sldpth[k] = dzs[k];
    zsoil[0] = -sldpth[0];
    for (int kz = 1; kz < NSOIL; ++kz)
        zsoil[kz] = FADD(-sldpth[kz], zsoil[kz - 1]);

    // REDPRM from the packed tables (SLOPETYP = 1 as the driver sets)
    const real* sv = soiltbl + (size_t)(soiltyp - 1) * NSOILC;
    const real* vv = vegtbl + (size_t)(vegtyp - 1) * NVEGC;
    real csoil = genp[GEN_CSOIL];
    real bexp = sv[SO_BEXP];
    real dksat = sv[SO_DKSAT];
    real dwsat = sv[SO_DWSAT];
    real f1 = sv[SO_F1];
    real psisat = sv[SO_PSISAT];
    real quartz = sv[SO_QUARTZ];
    real smcdry = sv[SO_SMCDRY];
    real smcmax = sv[SO_SMCMAX];
    real smcref = sv[SO_SMCREF];
    real smcwlt = sv[SO_SMCWLT];
    real zbot = genp[GEN_ZBOT];
    real salp = genp[GEN_SALP];
    real sbeta = genp[GEN_SBETA];
    real refdk = genp[GEN_REFDK];
    real frzk = genp[GEN_FRZK];
    real fxexp = genp[GEN_FXEXP];
    real refkdt = genp[GEN_REFKDT];
    real kdt = FDIV(FMUL(refkdt, dksat), refdk);
    real slope = genp[GEN_SLOPE];
    real lvcoef = genp[GEN_LVCOEF];
    real frzfact = FMUL((FDIV(smcmax, smcref)), (__uint_as_float(0x3f615e16u) /* gfortran15.2 fold_017 */));
    real frzx = FMUL(frzk, frzfact);
    real topt = genp[GEN_TOPT];
    real cmcmax = genp[GEN_CMCMAX];
    real cfactr = genp[GEN_CFACTR];
    real rsmax = genp[GEN_RSMAX];
    int nroot = (int)vv[VG_NROOT];
    real snup = vv[VG_SNUP];
    real rsmin = vv[VG_RSMIN];
    real rgl = vv[VG_RGL];
    real hs = vv[VG_HS];
    real emissmin = vv[VG_EMISSMIN];
    real emissmax = vv[VG_EMISSMAX];
    real laimin = vv[VG_LAIMIN];
    real laimax = vv[VG_LAIMAX];
    real z0min = vv[VG_Z0MIN];
    real z0max = vv[VG_Z0MAX];
    real albedomin = vv[VG_ALBEDOMIN];
    real albedomax = vv[VG_ALBEDOMAX];
    int bare = (int)genp[GEN_BARE];
    if (vegtyp == bare) shdfac = 0.0f;
    real rtdis[NSOIL];
    for (int i = 0; i < NSOIL; ++i) rtdis[i] = 0.0f;
    for (int i = 0; i < nroot; ++i)
        rtdis[i] = FDIV(-sldpth[i], zsoil[nroot - 1]);

    // urban parameter overrides (plain Noah, no UCM)
    if (vegtyp == isurban) {
        shdfac = 0.05f;
        rsmin = 400.0f;
        smcmax = 0.45f;
        smcref = 0.42f;
        smcwlt = 0.40f;
        smcdry = 0.40f;
    }

    // background emissivity / LAI / albedo / roughness interpolation
    real embrd_o;
    if (shdfac >= shmax) {
        embrd_o = emissmax;
        if (!rdlai2d) xlai = laimax;
        if (!usemonalb) alb = albedomin;
        z0brd = z0max;
    } else if (shdfac <= shmin) {
        embrd_o = emissmin;
        if (!rdlai2d) xlai = laimin;
        if (!usemonalb) alb = albedomax;
        z0brd = z0min;
    } else {
        if (shmax > shmin) {
            real interp_fraction = FDIV((FSUB(shdfac, shmin)), (FSUB(shmax, shmin)));
            interp_fraction = fminf(interp_fraction, 1.0f);
            interp_fraction = fmaxf(interp_fraction, 0.0f);
            embrd_o = FADD(FMUL((FSUB(1.0f, interp_fraction)), emissmin), FMUL(interp_fraction, emissmax));
            if (!rdlai2d) xlai = FADD(FMUL((FSUB(1.0f, interp_fraction)), laimin), FMUL(interp_fraction, laimax));
            if (!usemonalb) alb = FADD(FMUL((FSUB(1.0f, interp_fraction)), albedomax), FMUL(interp_fraction, albedomin));
            z0brd = FADD(FMUL((FSUB(1.0f, interp_fraction)), z0min), FMUL(interp_fraction, z0max));
        } else {
            embrd_o = FADD(FMUL(0.5f, emissmin), FMUL(0.5f, emissmax));
            if (!rdlai2d) xlai = FADD(FMUL(0.5f, laimin), FMUL(0.5f, laimax));
            if (!usemonalb) alb = FADD(FMUL(0.5f, albedomin), FMUL(0.5f, albedomax));
            z0brd = FADD(FMUL(0.5f, z0min), FMUL(0.5f, z0max));
        }
    }
    embrd = embrd_o;

    // snowpack density / precipitation type
    bool snowng = false, frzgra = false;
    real sndens, sncond;
    if (sneqv <= 1.0e-7f) {
        sneqv = 0.0f;
        sndens = 0.0f;
        snowhk = 0.0f;
        sncond = 1.0f;
    } else {
        sndens = FDIV(sneqv, snowhk);
        sncond = noah_csnow(sndens);
    }
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= NTFREEZ) frzgra = true;
    }
    real prcpf;
    if (snowng || frzgra) {
        real sn_new = FMUL(FMUL(prcp, dt), 0.001f);
        sneqv = FADD(sneqv, sn_new);
        prcpf = 0.0f;
        noah_snow_new(sfctmp, sn_new, snowhk, sndens);
        sncond = noah_csnow(sndens);
    } else {
        prcpf = prcp;
    }


#if NOAH_UA_PHYS
    real flx4=0.0f,fvb=0.0f,gama=0.0f,fbur=0.0f,fgsn=0.0f,etpn=0.0f;
#endif
    // snow cover fraction, snow albedo, emissivity
    real albedo;
    if (sneqv == 0.0f) {
        sncovr = 0.0f;
        albedo = alb;
        emissi = embrd;
    } else {
        sncovr = noah_snfrac(sneqv, snup, salp);
#if NOAH_UA_PHYS
        noah_ua_cover(sneqv,snowhk,shdfac,canopy[2*(vegtyp-1)],canopy[2*(vegtyp-1)+1],xlai,fvb,gama,fbur,fgsn);
        chk=noah_ua_exchange(sfctmp,t1,shdfac,fgsn,xlai,chk);
#endif

        sncovr = fminf(sncovr, 0.98f);
        noah_alcalc(alb, snoalb1, embrd, sncovr, dt, snowng, snotime1,
                    lvcoef, albedo, emissi);
    }

    // surface thermal conductivity + first-guess soil heat flux
    real df1 = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                           psisat, soiltyp, opt_thcnd);
    if (vegtyp == isurban) df1 = 3.24f;
    df1 = FMUL(df1, gfk_exp(FMUL(sbeta, shdfac)));
    if (sncovr > 0.97f) df1 = sncond;
    real dsoil = -(FMUL(0.5f, zsoil[0]));
    real dtot = 0.0f;
    real ssoil;
    if (sneqv == 0.0f) {
        ssoil = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dsoil);
    } else {
        dtot = FADD(snowhk, dsoil);
        real frcsno = FDIV(snowhk, dtot);
        real frcsoi = FDIV(dsoil, dtot);
        real df1a = FADD(FMUL(frcsno, sncond), FMUL(frcsoi, df1));
        df1 = FADD(FMUL(df1a, sncovr), FMUL(df1, (FSUB(1.0f, sncovr))));
        ssoil = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
    }

    // roughness over snow
    real z0k;
    if (sncovr > 0.0f) z0k = noah_snowz0(sncovr, z0brd, snowhk);
    else z0k = z0brd;


#if NOAH_UA_PHYS
    z0k=noah_ua_roughness(shmax,fbur,fgsn,z0brd);
#endif
    // Penman potential evaporation
    real fdown = FADD(solnet, lwdn);
    real t2v = FMUL(sfctmp, (FADD(1.0f, FMUL(0.61f, q2k))));
    real etp, rch, rr, epsca, t24, flx2;
    noah_penman(sfctmp, sfcprs, chk, t2v, th2, prcp, fdown, ssoil, q2k,
                q2sat, dqsdt2, snowng, frzgra, emissi, sncovr, etp,
                rch, rr, epsca, t24, flx2);
#if NOAH_UA_PHYS
    noah_ua_adjust_penman(sfctmp,th2,emissi,t24,fdown,ssoil,flx2,rch,rr,q2k,q2sat,dqsdt2,sncovr,sneqv,albedo,soldn,fvb,gama,stc[0],etpn,flx4);
#endif


    // canopy resistance
    // Complete burial skips CANRES. WRF leaves PC undefined; retain zero.
    real rc = 0.0f, pc = 0.0f;
    if (shdfac > 0.0f && xlai > 0.0f)
        noah_canres(soldn, chk, sfctmp, q2k, swc, zsoil, smcwlt,
                    smcref, rsmin, nroot, q2sat, dqsdt2, topt, rsmax,
                    rgl, hs, xlai, sfcprs, emissi, rc, pc);

    // ---- NOPAC / SNOPAC -------------------------------------------
    real eta_kin, beta, flx1, flx3, ssoil_pac;
    real runoff1, runoff2, runoff3, dew, drip;
    real edir1k = 0.0f, ec1k = 0.0f, ett1k = 0.0f;
    real et1k[NSOIL];
    for (int k = 0; k < NSOIL; ++k) et1k[k] = 0.0f;
    real etns = 0.0f, esnow = 0.0f, snomlt = 0.0f;
    int ebal_case;


    if (sneqv == 0.0f) {
        // =================== NOPAC ===================================
        ebal_case = 0;
        real prcp1 = FMUL(prcp, 0.001f);
        real etp1 = FMUL(etp, 0.001f);
        dew = 0.0f;
        real eta = 0.0f;
        if (etp > 0.0f) {
            real eta1;
            noah_evapo(eta1, smc, cmc, etp1, dt, swc, smcmax, pc,
                       smcwlt, smcref, shdfac, cmcmax, smcdry, cfactr,
                       nroot, rtdis, fxexp, edir1k, ec1k, et1k, ett1k);
            noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip);
            eta = FMUL(eta1, 1000.0f);
        } else {
            dew = -etp1;
            prcp1 = FADD(prcp1, dew);
            noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip);
        }
        if (etp <= 0.0f) {
            beta = 0.0f;
            eta = etp;
            if (etp < 0.0f) beta = 1.0f;
        } else {
            beta = FDIV(eta, etp);
        }
        real df1n = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                                psisat, soiltyp, opt_thcnd);
        if (vegtyp == isurban) df1n = 3.24f;
        df1n = FMUL(df1n, gfk_exp(FMUL(sbeta, shdfac)));
        real yynum = FSUB(fdown, FMUL(FMUL(emissi, NSIGMA), t24));
        real yy = FADD(sfctmp, FDIV((FSUB(FSUB(FADD(FDIV(yynum, rch), th2), sfctmp), FMUL(beta, epsca))), rr));
        real zz1 = FADD(FDIV(df1n, (FMUL(FMUL(FMUL(-0.5f, zsoil[0]), rch), rr))), 1.0f);
        noah_shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1n, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil_pac);
        flx1 = FMUL(FMUL(NCPH2O, prcp), (FSUB(t1, sfctmp)));
        flx3 = 0.0f;
        eta_kin = eta;
                               // filled below (needs eta_e)
    } else {
        // =================== SNOPAC ==================================
        const real esdmin = 1.0e-6f, snoexp = 2.0f;
        real esd = sneqv;
        dew = 0.0f;
        real etns1 = 0.0f;
        real esnow1, esnow2;
        esnow = 0.0f;
        esnow1 = 0.0f;
        esnow2 = 0.0f;
        real prcp1 = FMUL(prcpf, 0.001f);
        beta = 1.0f;
        real etanrg, etp1;
        if (etp <= 0.0f) {
            if (ribb >= 0.1f && fdown > 150.0f) {
                etp = FDIV((FADD(FDIV(FMUL(fminf(FMUL(etp, (FSUB(1.0f, ribb))), 0.0f), sncovr), 0.980f), FMUL(etp, (FSUB(0.980f, sncovr))))), 0.980f);
            }
            if (etp == 0.0f) beta = 0.0f;
            etp1 = FMUL(etp, 0.001f);
            dew = -etp1;
            esnow2 = FMUL(etp1, dt);
            etanrg = FMUL(etp, (FADD(FMUL((FSUB(1.0f, sncovr)), NLSUBC), FMUL(sncovr, NLSUBS))));
        } else {
            etp1 = FMUL(etp, 0.001f);
            if (sncovr < 1.0f) {
                noah_evapo(etns1, smc, cmc, etp1, dt, swc, smcmax, pc,
                           smcwlt, smcref, shdfac, cmcmax, smcdry,
                           cfactr, nroot, rtdis, fxexp, edir1k, ec1k,
                           et1k, ett1k);
                edir1k = FMUL(edir1k, (FSUB(1.0f, sncovr)));
                ec1k = FMUL(ec1k, (FSUB(1.0f, sncovr)));
                for (int k = 0; k < NSOIL; ++k)
                    et1k[k] = FMUL(et1k[k], (FSUB(1.0f, sncovr)));
                ett1k = FMUL(ett1k, (FSUB(1.0f, sncovr)));
                etns1 = FMUL(etns1, (FSUB(1.0f, sncovr)));
                etns = FMUL(etns1, 1000.0f);
            }
            esnow = FMUL(etp, sncovr);
#if NOAH_UA_PHYS
            esnow=FMUL(etpn,sncovr);
#endif

            esnow1 = FMUL(esnow, 0.001f);
            esnow2 = FMUL(esnow1, dt);
            etanrg = FADD(FMUL(esnow, NLSUBS), FMUL(etns, NLSUBC));
        }
        flx1 = 0.0f;
        if (snowng) flx1 = FMUL(FMUL(NCPICE, prcp), (FSUB(t1, sfctmp)));
        else if (prcp > 0.0f) flx1 = FMUL(FMUL(NCPH2O, prcp), (FSUB(t1, sfctmp)));
        real dsoil2 = FMUL(-0.5f, zsoil[0]);
        dtot = FADD(snowhk, dsoil2);
        real denom = FADD(1.0f, FDIV(df1, (FMUL(FMUL(dtot, rr), rch))));
        real t12a = FDIV((FSUB(FSUB(FADD(FDIV((FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t24))), rch), th2), sfctmp), FDIV(etanrg, rch))), rr);
        real t12b = FDIV(FMUL(df1, stc[0]), (FMUL(FMUL(dtot, rr), rch)));
        real t12 = FDIV((FADD(FADD(sfctmp, t12a), t12b)), denom);
        real stc1_old = stc[0];
        real ex = 0.0f;
        snomlt = 0.0f;
        if (t12 <= NTFREEZ) {
            ebal_case = 1;
            t1 = t12;
            ssoil_pac = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
            esd = fmaxf(0.0f, FSUB(esd, esnow2));
            flx3 = 0.0f;
#if NOAH_UA_PHYS
            flx4=0.0f;
#endif


        } else {
            t1 = FADD(FMUL(NTFREEZ, fmaxf(0.01f, gfk_pow(sncovr, snoexp))), FMUL(t12, (FSUB(1.0f, fmaxf(0.01f, gfk_pow(sncovr, snoexp))))));
            beta = 1.0f;
            ssoil_pac = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
            bool clipped = false;
            if (FSUB(esd, esnow2) <= esdmin) {
                esd = 0.0f;
                ex = 0.0f;
                snomlt = 0.0f;
                flx3 = 0.0f;
#if NOAH_UA_PHYS
            flx4=0.0f;
#endif

                clipped = true;
            } else {
                esd = FSUB(esd, esnow2);
                real seh = FMUL(rch, (FSUB(t1, th2)));
                real t14 = FMUL(t1, t1);
                t14 = FMUL(t14, t14);
                flx3 = FSUB(FSUB(FSUB(FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t14)), ssoil_pac), seh), etanrg);
                real flx3_def = flx3;
                if (flx3 <= 0.0f) flx3 = 0.0f;
                if (flx3 != flx3_def) clipped = true;

#if NOAH_UA_PHYS
                noah_ua_melt_flux(flx3,flx4);
#endif
                ex = FDIV(FMUL(flx3, 0.001f), NLSUBF);
                snomlt = FMUL(ex, dt);
                if (FSUB(esd, snomlt) >= esdmin) {
                    esd = FSUB(esd, snomlt);
                } else {
                    ex = FDIV(esd, dt);
                    flx3 = FMUL(FMUL(ex, 1000.0f), NLSUBF);
                    snomlt = esd;
                    esd = 0.0f;
                    clipped = true;
                }
            }
            ebal_case = clipped ? 3 : 2;
            prcp1 = FADD(prcp1, ex);
            real seh = FMUL(rch, (FSUB(t1, th2)));
            real t14 = FMUL(t1, t1);
            t14 = FMUL(t14, t14);

        }
        noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                   frzx, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                   cmcmax, edir1k, ec1k, et1k, runoff1, runoff2,
                   runoff3, drip);
        real zz1 = 1.0f;
        real yy = FSUB(stc[0], FDIV(FMUL(FMUL(FMUL(0.5f, ssoil_pac), zsoil[0]), zz1), df1));
        real t11 = t1;
        real ssoil1;
        noah_shflx(stc, smc, smcmax, t11, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil1);
        if (esd > 0.0f) {
            noah_snowpack(esd, dt, snowhk, sndens, t1, yy
#if NOAH_UA_PHYS
,snomlt
#endif
);
        } else {
            esd = 0.0f;
            snowhk = 0.0f;
            sndens = 0.0f;
            sncovr = 0.0f;
        }
        sneqv = esd;
        eta_kin = FSUB(FADD(esnow, etns), FMUL(1000.0f, dew));
    }

    real q1 = FADD(q2k, FDIV(FMUL(eta_kin, CP), rch));
    real sheat = FMUL(FDIV(-(FMUL(FMUL(chk, CP), sfcprs)), (FMUL(NR_SHEAT, t2v))), (FSUB(th2, t1)));


#if NOAH_UA_PHYS
    sheat=FADD(sheat,flx4);
#endif
    // kinematic -> energy conversions (SFLX epilogue)
    real edir_e = FMUL((FMUL(edir1k, 1000.0f)), NLVH2O);
    real ec_e = FMUL((FMUL(ec1k, 1000.0f)), NLVH2O);
    real ett_e = FMUL((FMUL(ett1k, 1000.0f)), NLVH2O);
    real esnow_e = FMUL(esnow, NLSUBS);
    real etp_e = FMUL(etp, (FADD(FMUL((FSUB(1.0f, sncovr)), NLVH2O), FMUL(sncovr, NLSUBS))));
    real eta_e;
    if (etp_e > 0.0f) eta_e = FADD(FADD(FADD(edir_e, ec_e), ett_e), esnow_e);
    else eta_e = etp_e;
    if (etp_e == 0.0f) beta = 0.0f;
    else beta = FDIV(eta_e, etp_e);

    if (ebal_case == 0) {

    }

    real ssoil_out = FMUL(-1.0f, ssoil_pac);
    runoff3 = FDIV(runoff3, dt);
    runoff2 = FADD(runoff2, runoff3);
    real soilm = FMUL(FMUL(-1.0f, smc[0]), zsoil[0]);
    for (int k = 1; k < NSOIL; ++k)
        soilm = FADD(soilm, FMUL(smc[k], (FSUB(zsoil[k - 1], zsoil[k]))));
    real soilwm = FMUL(FMUL(-1.0f, (FSUB(smcmax, smcwlt))), zsoil[0]);
    real soilww = FMUL(FMUL(-1.0f, (FSUB(smc[0], smcwlt))), zsoil[0]);
    real smav[NSOIL];
    for (int k = 0; k < NSOIL; ++k)
        smav[k] = FDIV((FSUB(smc[k], smcwlt)), (FSUB(smcmax, smcwlt)));
    if (nroot >= 2) {
        for (int k = 1; k < nroot; ++k) {
            soilwm = FADD(soilwm, FMUL((FSUB(smcmax, smcwlt)), (FSUB(zsoil[k - 1], zsoil[k]))));
            soilww = FADD(soilww, FMUL((FSUB(smc[k], smcwlt)), (FSUB(zsoil[k - 1], zsoil[k]))));
        }
    }
    real soilw;
    if (soilwm < 1.0e-6f) {
        soilwm = 0.0f;
        soilw = 0.0f;
        soilm = 0.0f;
    } else {
        soilw = FDIV(soilww, soilwm);
    }

    // ================= driver post-SFLX updates ======================
    lai_a[idx] = xlai;
    canwat_a[idx] = FMUL(cmc, 1000.0f);
    snow_a[idx] = FMUL(sneqv, 1000.0f);
    snowh_a[idx] = snowhk;
    albedo_a[idx] = albedo;
    albbck_a[idx] = alb;
    z0_a[idx] = z0brd;
    emiss_a[idx] = emissi;
    znt_a[idx] = z0k;
#if NOAH_MOSAIC_UCM
    s.rural_t1=t1; s.rural_q1=q1;
#endif
    tsk_a[idx] = t1;
    hfx_a[idx] = sheat;
    potevp_a[idx] = FADD(potevp_a[idx], FMUL(etp_e, (FDIV(dt, (__uint_as_float(0x4f1502f9u) /* gfortran15.2 fold_016 */)))));
    qfx_a[idx] = eta_kin;
    lh_a[idx] = eta_e;
    grdflx_a[idx] = ssoil_out;
    snowc_a[idx] = sncovr;
    // WRF module_sf_noahdrv.F:3691: ordinary-land Noah makes the 2 m
    // thermal exchange coefficient identical to its moisture coefficient.
    chs2_a[idx] = cqs2_a[idx];
    snotime_a[idx] = snotime1;
    if (q1 > qsfc_a[idx]) cqs2_a[idx] = chs_a[idx];
    qsfc_a[idx] = FDIV(q1, (FSUB(1.0f, q1)));
    for (int k = 0; k < NSOIL; ++k) {
        smois_a[k * plane + idx] = smc[k];
        tslb_a[k * plane + idx] = stc[k];
        sh2o_a[k * plane + idx] = swc[k];
        smcrel_a[k * plane + idx] = smav[k];
    }
    real t14 = FMUL(t1, t1); t14 = FMUL(t14, t14);
    noahres_a[idx] = FSUB(FSUB(FSUB(FSUB(FSUB(FADD(FSUB((FADD(solnet, lwdn)), sheat), ssoil_out), eta_e), (FMUL(FMUL(emissi, NSTBOLT), t14))), flx1), flx2), flx3);

#if NOAH_UA_PHYS
    noahres_a[idx]=FSUB(noahres_a[idx],flx4);
    s.flx4=flx4;s.fvb=fvb;s.fbur=fbur;s.fgsn=fgsn;
#endif
    chklowq_a[idx] = 1.0f;
    smstav_a[idx] = soilw;
    smstot_a[idx] = FMUL(soilm, 1000.0f);
    sfcrunoff_a[idx] = FADD(sfcrunoff_a[idx], FMUL(FMUL(runoff1, dt), 1000.0f));
    udrunoff_a[idx] = FADD(udrunoff_a[idx], FMUL(FMUL(runoff2, dt), 1000.0f));
    if (ffrozp > 0.5f)
        acsnow_a[idx] = FADD(acsnow_a[idx], FMUL(prcp, dt));
    if (snow_a[idx] > 0.0f) {
        acsnom_a[idx] = FADD(acsnom_a[idx], FMUL(snomlt, 1000.0f));
        snopcx_a[idx] = FSUB(snopcx_a[idx], FDIV(snomlt, (FDIV(dt, __uint_as_float(0x4da6e49cu) /* gfortran ROWLIW=ROW*ELIW */))));
    }
}
struct MosaicSums {
 real tsk;
 real znt;
 real z0;
 real area;
 real r1;
 real r2;
 real pe;
 real sm;
 real sx;
 real qsfc;
 real canwat;
 real snow;
 real snowh;
 real snowc;
 real albedo;
 real albbck;
 real emiss;
 real embck;
 real hfx;
 real qfx;
 real lh;
 real grdflx;
 real soil[3][4];
};
// module_sf_noahdrv.F:4139-4175, D1 corrected soil indices; D2 weighted increments.
__device__ static void noah_mosaic_accumulate(MosaicState& s,const MosaicState& tile,
 MosaicSums& avg,real area) {
 real t4=FMUL(tile.tsk, tile.tsk);t4=FMUL(t4, t4);avg.tsk=FADD(avg.tsk, FMUL((FMUL(tile.emiss, t4)), area));
 avg.qsfc=FADD(avg.qsfc, FMUL(tile.qsfc, area));
 avg.canwat=FADD(avg.canwat, FMUL(tile.canwat, area));
 avg.snow=FADD(avg.snow, FMUL(tile.snow, area));
 avg.snowh=FADD(avg.snowh, FMUL(tile.snowh, area));
 avg.snowc=FADD(avg.snowc, FMUL(tile.snowc, area));
 avg.albedo=FADD(avg.albedo, FMUL(tile.albedo, area));
 avg.albbck=FADD(avg.albbck, FMUL(tile.albbck, area));
 avg.emiss=FADD(avg.emiss, FMUL(tile.emiss, area));
 avg.embck=FADD(avg.embck, FMUL(tile.embck, area));
 avg.hfx=FADD(avg.hfx, FMUL(tile.hfx, area));
 avg.qfx=FADD(avg.qfx, FMUL(tile.qfx, area));
 avg.lh=FADD(avg.lh, FMUL(tile.lh, area));
 avg.grdflx=FADD(avg.grdflx, FMUL(tile.grdflx, area));
 avg.znt=FADD(avg.znt, FMUL(gfk_log(tile.znt), area));avg.z0=FADD(avg.z0, FMUL(gfk_log(tile.z0), area));avg.area=FADD(avg.area, area);
 for(int k=0;k<4;++k)avg.soil[0][k]=FADD(avg.soil[0][k], FMUL(tile.smois[k], area));
 for(int k=0;k<4;++k)avg.soil[1][k]=FADD(avg.soil[1][k], FMUL(tile.tslb[k], area));
 for(int k=0;k<4;++k)avg.soil[2][k]=FADD(avg.soil[2][k], FMUL(tile.sh2o[k], area));
 avg.r1=FADD(avg.r1, FMUL(tile.sfcrunoff, area));avg.r2=FADD(avg.r2, FMUL(tile.udrunoff, area));
 avg.pe=FADD(avg.pe, FMUL(tile.potevp, area));avg.sm=FADD(avg.sm, FMUL(tile.acsnom, area));avg.sx=FADD(avg.sx, FMUL((-tile.snopcx), area));

#if NOAH_UA_PHYS
 s.flx4=tile.flx4;s.fvb=tile.fvb;s.fbur=tile.fbur;s.fgsn=tile.fgsn;
#endif
 s.cqs2=tile.cqs2;s.chs2=tile.chs2;s.lai=tile.lai;s.snotime=tile.snotime;
 s.smstav=tile.smstav;s.smstot=tile.smstot;s.noahres=tile.noahres;s.chklowq=tile.chklowq;
 for(int k=0;k<4;++k)s.smcrel[k]=tile.smcrel[k];
}
// module_sf_noahdrv.F:4181-4213; D2 adds corrected increments once per cell.
__device__ static void noah_mosaic_mean(MosaicState& s,const MosaicSums& avg,
 real dt,int frpcpn) {
s.qsfc=avg.qsfc;
 s.canwat=avg.canwat;
 s.snow=avg.snow;
 s.snowh=avg.snowh;
 s.snowc=avg.snowc;
 s.albedo=avg.albedo;
 s.albbck=avg.albbck;
 s.emiss=avg.emiss;
 s.embck=avg.embck;
 s.hfx=avg.hfx;
 s.qfx=avg.qfx;
 s.lh=avg.lh;
 s.grdflx=avg.grdflx;
 for(int k=0;k<4;++k)s.smois[k]=avg.soil[0][k];
 for(int k=0;k<4;++k)s.tslb[k]=avg.soil[1][k];
 for(int k=0;k<4;++k)s.sh2o[k]=avg.soil[2][k];
 s.tsk=gfk_pow(FDIV(avg.tsk, s.emiss),0.25f);s.znt=gfk_exp(FDIV(avg.znt, avg.area));s.z0=gfk_exp(FDIV(avg.z0, avg.area));
 s.sfcrunoff=FADD(s.sfcrunoff, avg.r1);s.udrunoff=FADD(s.udrunoff, avg.r2);s.potevp=FADD(s.potevp, avg.pe);
 s.acsnom=FADD(s.acsnom, avg.sm);s.snopcx=FSUB(s.snopcx, avg.sx);
 real frozen=frpcpn?s.sr:(s.sfctmp<=273.15f?1.0f:0.0f);
 if(frozen>0.5f)s.acsnow=FADD(s.acsnow, FMUL((FDIV(s.rainbl, dt)), dt));
}
#if NOAH_MOSAIC_UCM
extern "C" __global__ void noah_mosaic_ucm_column(
#else
extern "C" __global__ void noah_mosaic_column(
#endif
 int* ivgtyp,
 int* isltyp,
 real* psfc_a,
 real* sfcprs_a,
 real* sfctmp_a,
 real* qv1_a,
 real* qgh_a,
 real* dz8w1_a,
 real* glw_a,
 real* swdown_a,
 real* rainbl_a,
 real* sr_a,
 real* chs_a,
 real* cqs2_a,
 real* chs2_a,
 real* rib_a,
 real* vegfra_a,
 real* shdmin_a,
 real* shdmax_a,
 real* tmn_a,
 real* xland_a,
 real* xice_a,
 real* snoalb_a,
 real* embck_a,
 real* tsk_a,
 real* hfx_a,
 real* qfx_a,
 real* lh_a,
 real* grdflx_a,
 real* qsfc_a,
 real* canwat_a,
 real* snow_a,
 real* snowc_a,
 real* snowh_a,
 real* albedo_a,
 real* albbck_a,
 real* emiss_a,
 real* znt_a,
 real* z0_a,
 real* snotime_a,
 real* lai_a,
 real* smstav_a,
 real* smstot_a,
 real* sfcrunoff_a,
 real* udrunoff_a,
 real* acsnow_a,
 real* acsnom_a,
 real* snopcx_a,
 real* potevp_a,
 real* noahres_a,
 real* chklowq_a,
 real* smois_a,
 real* tslb_a,
 real* sh2o_a,
 real* smcrel_a,
 real* tsk_mosaic,
 real* qsfc_mosaic,
 real* canwat_mosaic,
 real* snow_mosaic,
 real* snowh_mosaic,
 real* snowc_mosaic,
 real* albedo_mosaic,
 real* albbck_mosaic,
 real* emiss_mosaic,
 real* embck_mosaic,
 real* znt_mosaic,
 real* z0_mosaic,
 real* hfx_mosaic,
 real* qfx_mosaic,
 real* lh_mosaic,
 real* grdflx_mosaic,
 real* snotime_mosaic,
 real* tslb_mosaic,
 real* smois_mosaic,
 real* sh2o_mosaic,
 const int* mosaic_cat_index,
 const real* landusef2,
 const int* lcz,
 int nlcz,
 const real* vegtbl,
 const real* soiltbl,
 const real* genp,
 const real* dzs,
 real dt,
 int lucats,
 int slcats,
 int isurban,
 int isice,
 real xice_threshold,
 int itimestep,
 int frpcpn,
 int usemonalb,
 int rdlai2d,
 int opt_thcnd,
 int mosaic_cat,
 int ny,
 int nx
#if NOAH_UA_PHYS
,const real* canopy,real* flx4_a,real* fvb_a,real* fbur_a,real* fgsn_a
#endif

#if NOAH_MOSAIC_UCM
 , const unsigned long long* urban_state,
 const unsigned long long* urban_tiles,
 const real* tab, const real* glob, const int* isw,
 const real* frc, const real* u1, const real* v1, const real* hrang,
 real* ust, int natural, int use_lcz, int jmonth, unsigned int* err
#endif
 ) {
 int col=blockIdx.x*blockDim.x+threadIdx.x; if(col>=ny*nx)return;
 const size_t idx=col,plane=(size_t)ny*nx;
 MosaicState s;
#if NOAH_UA_PHYS
 s.flx4=flx4_a[idx];s.fvb=fvb_a[idx];s.fbur=fbur_a[idx];s.fgsn=fgsn_a[idx];
#endif

#if NOAH_MOSAIC_UCM
 s.urban_tile=false; s.natural=natural;
#endif
 s.ivgtyp=ivgtyp[idx];s.isltyp=isltyp[idx];
 s.psfc=psfc_a[idx];
 s.sfcprs=sfcprs_a[idx];
 s.sfctmp=sfctmp_a[idx];
 s.qv1=qv1_a[idx];
 s.qgh=qgh_a[idx];
 s.dz8w1=dz8w1_a[idx];
 s.glw=glw_a[idx];
 s.swdown=swdown_a[idx];
 s.rainbl=rainbl_a[idx];
 s.sr=sr_a[idx];
 s.chs=chs_a[idx];
 s.cqs2=cqs2_a[idx];
 s.chs2=chs2_a[idx];
 s.rib=rib_a[idx];
 s.vegfra=vegfra_a[idx];
 s.shdmin=shdmin_a[idx];
 s.shdmax=shdmax_a[idx];
 s.tmn=tmn_a[idx];
 s.xland=xland_a[idx];
 s.xice=xice_a[idx];
 s.snoalb=snoalb_a[idx];
 s.embck=embck_a[idx];
 s.tsk=tsk_a[idx];
 s.hfx=hfx_a[idx];
 s.qfx=qfx_a[idx];
 s.lh=lh_a[idx];
 s.grdflx=grdflx_a[idx];
 s.qsfc=qsfc_a[idx];
 s.canwat=canwat_a[idx];
 s.snow=snow_a[idx];
 s.snowc=snowc_a[idx];
 s.snowh=snowh_a[idx];
 s.albedo=albedo_a[idx];
 s.albbck=albbck_a[idx];
 s.emiss=emiss_a[idx];
 s.znt=znt_a[idx];
 s.z0=z0_a[idx];
 s.snotime=snotime_a[idx];
 s.lai=lai_a[idx];
 s.smstav=smstav_a[idx];
 s.smstot=smstot_a[idx];
 s.sfcrunoff=sfcrunoff_a[idx];
 s.udrunoff=udrunoff_a[idx];
 s.acsnow=acsnow_a[idx];
 s.acsnom=acsnom_a[idx];
 s.snopcx=snopcx_a[idx];
 s.potevp=potevp_a[idx];
 s.noahres=noahres_a[idx];
 s.chklowq=chklowq_a[idx];
 for(int k=0;k<4;++k)s.smois[k]=smois_a[k*plane+idx];
 for(int k=0;k<4;++k)s.tslb[k]=tslb_a[k*plane+idx];
 for(int k=0;k<4;++k)s.sh2o[k]=sh2o_a[k*plane+idx];
 for(int k=0;k<4;++k)s.smcrel[k]=smcrel_a[k*plane+idx];
 // module_sf_noahdrv.F:3118-3157
 if(itimestep==1) {
  if(s.xland>=1.5f) {
   s.smstav=1.0f;s.smstot=1.0f;
   for(int k=0;k<4;++k){s.smois[k]=1.0f;s.tslb[k]=273.16f;s.smcrel[k]=1.0f;}
  } else if(s.xice>=xice_threshold) {
   s.smstav=1.0f;s.smstot=1.0f;
   for(int k=0;k<4;++k){s.smois[k]=1.0f;s.smcrel[k]=1.0f;}
  }
 }
 s.chklowq=1.0f;
 if(s.xland<1.5f && s.xice<xice_threshold) {
 MosaicSums avg={};
 real carried_soilw=0.0f;
 for(int t=mosaic_cat-1;t>=0;--t){
 size_t off=t*plane+idx;
 MosaicState tile=s;
 tile.ivgtyp=mosaic_cat_index[off];
#if NOAH_MOSAIC_UCM
 int utype=tile.ivgtyp==isurban?(use_lcz?5:2):0;
 for(int l=0;l<nlcz;++l)if(tile.ivgtyp==lcz[l])utype=l+1;
 tile.urban_tile=utype!=0;
 // Preserve the original tile category for UTYPE; remap only VEGTYP.
#else
 for(int l=0;l<nlcz;++l)if(tile.ivgtyp==lcz[l])tile.ivgtyp=isurban;
#endif
 tile.tsk=tsk_mosaic[off];
 tile.qsfc=qsfc_mosaic[off];
 tile.canwat=canwat_mosaic[off];
 tile.snow=snow_mosaic[off];
 tile.snowh=snowh_mosaic[off];
 tile.snowc=snowc_mosaic[off];
 tile.albedo=albedo_mosaic[off];
 tile.albbck=albbck_mosaic[off];
 tile.emiss=emiss_mosaic[off];
 tile.embck=embck_mosaic[off];
 tile.znt=znt_mosaic[off];
 tile.z0=z0_mosaic[off];
 tile.hfx=hfx_mosaic[off];
 tile.qfx=qfx_mosaic[off];
 tile.lh=lh_mosaic[off];
 tile.grdflx=grdflx_mosaic[off];
 tile.snotime=snotime_mosaic[off];
 for(int k=0;k<4;++k)tile.tslb[k]=tslb_mosaic[(4*t+k)*plane+idx];
 for(int k=0;k<4;++k)tile.smois[k]=smois_mosaic[(4*t+k)*plane+idx];
 for(int k=0;k<4;++k)tile.sh2o[k]=sh2o_mosaic[(4*t+k)*plane+idx];
 tile.sfcrunoff=0.0f;tile.udrunoff=0.0f;tile.potevp=0.0f;tile.acsnom=0.0f;tile.snopcx=0.0f;tile.acsnow=0.0f;
 tile.smstav=carried_soilw;
#if NOAH_MOSAIC_UCM
 if(tile.urban_tile)tile.rural_initial=PLANE(urban_tiles,7)[off];
#endif
 if(tile.ivgtyp==isice)noah_mosaic_tile_glacial(tile,vegtbl,soiltbl,genp,dzs,dt,lucats,slcats,isurban,isice,xice_threshold,itimestep,frpcpn,usemonalb,rdlai2d,opt_thcnd);
 else noah_mosaic_tile(tile,vegtbl,soiltbl,genp,dzs,dt,lucats,slcats,isurban,isice,xice_threshold,itimestep,frpcpn,usemonalb,rdlai2d,opt_thcnd
#if NOAH_UA_PHYS
,canopy
#endif
);
#if NOAH_MOSAIC_UCM
 if(tile.urban_tile) {
  // noahdrv.F:3733-4057. Roof/wall/road stacks tile; SMR/TGRL stay grid.
  UcmCol c; ucm_load_state(urban_state,idx,plane,c);
  c.tr=PLANE(urban_tiles,0)[off];c.tb=PLANE(urban_tiles,1)[off];
  c.tg=PLANE(urban_tiles,2)[off];c.tc=PLANE(urban_tiles,3)[off];
  c.qc=PLANE(urban_tiles,4)[off];c.uc=PLANE(urban_tiles,5)[off];
  for(int k=0;k<4;++k){size_t a=(4*t+k)*plane+idx;
   c.trl[k]=PLANE(urban_tiles,12)[a];c.tbl[k]=PLANE(urban_tiles,13)[a];c.tgl[k]=PLANE(urban_tiles,14)[a];}
  c.utype=utype;c.jmonth=jmonth;c.ta=tile.sfctmp;
  c.qa=FDIV(tile.qv1,FADD(1.0f,tile.qv1));
  c.u1=u1[idx];c.v1=v1[idx];c.ua=ucm_wind(c.u1,c.v1);
  c.ssg=tile.swdown;c.llg=tile.glw;c.rain=FMUL(FDIV(tile.rainbl,dt),3600.0f);
  c.rhoo=FDIV(tile.sfcprs,FMUL(FMUL(287.04f,c.ta),FADD(1.0f,FMUL(0.61f,c.qa))));
  c.za=FMUL(0.5f,tile.dz8w1);c.delt=dt;c.omg=hrang[idx];c.znt=tile.znt;
  ucm_floor_exchange(&tile.chs,&tile.chs2,&tile.cqs2,0);
  c.chs=tile.chs;c.chs2=tile.chs2;
  int code=ucm_urban(tab,glob,isw,c);
  if(code!=UCM_OK){ucm_flag(err,code);return;}
  real f=frc[idx],omf=FSUB(1.0f,f);
  tile.albedo=FADD(FMUL(f,c.alb),FMUL(omf,tile.albedo));
  tile.hfx=FADD(FMUL(f,c.sh),FMUL(omf,tile.hfx));
  tile.qfx=FADD(FMUL(f,c.lh_kin),FMUL(omf,tile.qfx));
  tile.lh=FADD(FMUL(f,c.lh),FMUL(omf,tile.lh));
  tile.grdflx=FADD(FMUL(f,c.g),FMUL(omf,tile.grdflx));
  tile.tsk=FADD(FMUL(f,c.ts),FMUL(omf,tile.rural_t1));
  real q1=FADD(FMUL(f,c.qs),FMUL(omf,tile.rural_q1));
  tile.qsfc=FDIV(q1,FSUB(1.0f,q1));
  ust[idx]=FADD(FMUL(f,c.ust),FMUL(omf,ust[idx]));
  tile.znt=gfk_exp(FADD(FMUL(f,gfk_log(c.znt)),FMUL(omf,gfk_log(tile.znt))));
  ucm_store_state(urban_state,idx,plane,c);
  PLANE(urban_tiles,0)[off]=c.tr;PLANE(urban_tiles,1)[off]=c.tb;
  PLANE(urban_tiles,2)[off]=c.tg;PLANE(urban_tiles,3)[off]=c.tc;
  PLANE(urban_tiles,4)[off]=c.qc;PLANE(urban_tiles,5)[off]=c.uc;
  PLANE(urban_tiles,6)[off]=c.ts;PLANE(urban_tiles,7)[off]=tile.rural_t1;
  PLANE(urban_tiles,8)[off]=c.sh;PLANE(urban_tiles,9)[off]=c.lh;
  PLANE(urban_tiles,10)[off]=c.g;PLANE(urban_tiles,11)[off]=c.rn;
  for(int k=0;k<4;++k){size_t a=(4*t+k)*plane+idx;
   PLANE(urban_tiles,12)[a]=c.trl[k];PLANE(urban_tiles,13)[a]=c.tbl[k];PLANE(urban_tiles,14)[a]=c.tgl[k];}
  s.chs=tile.chs;
 }
#endif
 carried_soilw=tile.smstav;
 real area=landusef2[off];
 tsk_mosaic[off]=tile.tsk;
 qsfc_mosaic[off]=tile.qsfc;
 canwat_mosaic[off]=tile.canwat;
 snow_mosaic[off]=tile.snow;
 snowh_mosaic[off]=tile.snowh;
 snowc_mosaic[off]=tile.snowc;
 albedo_mosaic[off]=tile.albedo;
 albbck_mosaic[off]=tile.albbck;
 emiss_mosaic[off]=tile.emiss;
 embck_mosaic[off]=tile.embck;
 znt_mosaic[off]=tile.znt;
 z0_mosaic[off]=tile.z0;
 hfx_mosaic[off]=tile.hfx;
 qfx_mosaic[off]=tile.qfx;
 lh_mosaic[off]=tile.lh;
 grdflx_mosaic[off]=tile.grdflx;
 snotime_mosaic[off]=tile.snotime;
 for(int k=0;k<4;++k)tslb_mosaic[(4*t+k)*plane+idx]=tile.tslb[k];
 for(int k=0;k<4;++k)smois_mosaic[(4*t+k)*plane+idx]=tile.smois[k];
 for(int k=0;k<4;++k)sh2o_mosaic[(4*t+k)*plane+idx]=tile.sh2o[k];
 noah_mosaic_accumulate(s,tile,avg,area);
 }
 noah_mosaic_mean(s,avg,dt,frpcpn);
 } else if(s.xland<1.5f && s.xice>=xice_threshold){
  for(int k=0;k<4;++k)s.sh2o[k]=1.0f;s.lai=0.01f;
 }
 psfc_a[idx]=s.psfc;
 sfcprs_a[idx]=s.sfcprs;
 sfctmp_a[idx]=s.sfctmp;
 qv1_a[idx]=s.qv1;
 qgh_a[idx]=s.qgh;
 dz8w1_a[idx]=s.dz8w1;
 glw_a[idx]=s.glw;
 swdown_a[idx]=s.swdown;
 rainbl_a[idx]=s.rainbl;
 sr_a[idx]=s.sr;
 chs_a[idx]=s.chs;
 cqs2_a[idx]=s.cqs2;
 chs2_a[idx]=s.chs2;
 rib_a[idx]=s.rib;
 vegfra_a[idx]=s.vegfra;
 shdmin_a[idx]=s.shdmin;
 shdmax_a[idx]=s.shdmax;
 tmn_a[idx]=s.tmn;
 xland_a[idx]=s.xland;
 xice_a[idx]=s.xice;
 snoalb_a[idx]=s.snoalb;
 embck_a[idx]=s.embck;
 tsk_a[idx]=s.tsk;
 hfx_a[idx]=s.hfx;
 qfx_a[idx]=s.qfx;
 lh_a[idx]=s.lh;
 grdflx_a[idx]=s.grdflx;
 qsfc_a[idx]=s.qsfc;
 canwat_a[idx]=s.canwat;
 snow_a[idx]=s.snow;
 snowc_a[idx]=s.snowc;
 snowh_a[idx]=s.snowh;
 albedo_a[idx]=s.albedo;
 albbck_a[idx]=s.albbck;
 emiss_a[idx]=s.emiss;
 znt_a[idx]=s.znt;
 z0_a[idx]=s.z0;
 snotime_a[idx]=s.snotime;
 lai_a[idx]=s.lai;
 smstav_a[idx]=s.smstav;
 smstot_a[idx]=s.smstot;
 sfcrunoff_a[idx]=s.sfcrunoff;
 udrunoff_a[idx]=s.udrunoff;
 acsnow_a[idx]=s.acsnow;
 acsnom_a[idx]=s.acsnom;
 snopcx_a[idx]=s.snopcx;
 potevp_a[idx]=s.potevp;

#if NOAH_UA_PHYS
 flx4_a[idx]=s.flx4;fvb_a[idx]=s.fvb;fbur_a[idx]=s.fbur;fgsn_a[idx]=s.fgsn;
#endif
 noahres_a[idx]=s.noahres;
 chklowq_a[idx]=s.chklowq;
 for(int k=0;k<4;++k)smois_a[k*plane+idx]=s.smois[k];
 for(int k=0;k<4;++k)tslb_a[k*plane+idx]=s.tslb[k];
 for(int k=0;k<4;++k)sh2o_a[k*plane+idx]=s.sh2o[k];
 for(int k=0;k<4;++k)smcrel_a[k*plane+idx]=s.smcrel[k];
}

#endif // !NOAH_MATH_ONLY
