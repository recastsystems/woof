// NTU multi-moment microphysics, WRF mp_physics=56.
// Authority: unmodified WRF 4.7.1 phys/module_mp_ntu.F, public domain.
// SHA256 c864483fa572d42ab1dc51cc538f8062dc2f582369f597261e8c850c49ce34dd.
// Mechanical operation-order transcription: tools/ntu_wrf461_oracle/f2cu.py.
// Native capture/replay and licence pin: tools/ntu_wrf471_oracle/.
// The cold-start driver completes the witnessed uninitialized top density.
// No recovered F1-F6 source change is applied to the transcribed module.

// ---------------------------------------------------------------------------
// NTU multi-moment microphysics (mp_physics = 56): device vocabulary.
//
// Every arithmetic operation in the transcription below is one IEEE binary32
// (or binary64 where WRF declares DOUBLE PRECISION) operation in WRF's own
// order.  The strict arm is compiled with --fmad=false --ftz=false so that a
// written a*b+c stays two roundings, exactly as gfortran -ffp-contract=off.
// The transcendental words are WOOF's own math routines (expf/logf/powf and
// the binary64 exp/log); gfortran lowers WRF's REAL**REAL to powf except for
// the exponents 2.0, 1.0 and -1.0, which it folds (ntu_sq, identity, 1/x).
// ---------------------------------------------------------------------------
#ifndef NTU_DEV
#define NTU_DEV static __device__ __forceinline__
#endif
// The four long WRF routines stay out of line so NVRTC compiles them once.
#ifndef NTU_DEV_BIG
#define NTU_DEV_BIG static __device__ __noinline__
#endif
#ifndef NTU_CONST_TABLE
#define NTU_CONST_TABLE __device__ const
#endif
#ifndef NTU_KMAX
#define NTU_KMAX 64
#endif

// Status word bits (one per launch, OR-reduced): the column hit a WRF
// fatal-error branch (STOP / wrf_error_fatal) and its result is not WRF's.
#define NTU_ERR_STOP  1u
#define NTU_ERR_FATAL 2u
#define NTU_ERR_KMAX  4u

// Native LOG10F rounds the logarithm once to binary32.
#ifndef NTU_HOST
// Native glibc 2.43 LOG10F rounds the logarithm once to binary32.
// The recovered split-float route produced 17 wrong real-capture words.
NTU_DEV float ntu_log10(float x) {
    return (float)__dmul_rn(glibc_log((double)x),0.43429448190325182765);
}
#endif

// WOOF ERF: exp(-x*x) times the positive integral recurrence.
// Two-word arithmetic preserves the recurrence at binary64 boundaries.
// The 2/sqrt(pi) split is computed from the Machin identity with
// Decimal precision 90; no external math source or runtime asset is used.
#ifndef NTU_HOST
struct NtuDD {double hi,lo;};
NTU_DEV NtuDD ntu_dd_norm(double a,double b){double h=__dadd_rn(a,b);return {h,__dsub_rn(b,__dsub_rn(h,a))};}
NTU_DEV NtuDD ntu_dd_add(NtuDD a,NtuDD b){double h=__dadd_rn(a.hi,b.hi),v=__dsub_rn(h,a.hi);double e=__dadd_rn(__dsub_rn(a.hi,__dsub_rn(h,v)),__dsub_rn(b.hi,v));e=__dadd_rn(e,__dadd_rn(a.lo,b.lo));return ntu_dd_norm(h,e);}
NTU_DEV NtuDD ntu_dd_mul(NtuDD a,NtuDD b){double h=__dmul_rn(a.hi,b.hi);double e=__fma_rn(a.hi,b.hi,-h);e=__dadd_rn(e,__dadd_rn(__dmul_rn(a.hi,b.lo),__dmul_rn(a.lo,b.hi)));return ntu_dd_norm(h,e);}
NTU_DEV NtuDD ntu_dd_div(NtuDD a,double b){double q=__ddiv_rn(a.hi,b);double r=__dadd_rn(__fma_rn(-q,b,a.hi),a.lo);return ntu_dd_norm(q,__ddiv_rn(r,b));}
NTU_DEV_BIG NtuDD ntu_dd_exp(NtuDD x){
 // ln(2) split and a short reduced Taylor series, retaining both words.
 const NtuDD ln2={0.6931471805599453,2.3190468138462996e-17};
 int k=(int)round(x.hi*1.4426950408889634);
 NtuDD r=ntu_dd_add(x,ntu_dd_mul({-(double)k,0},ln2));
 NtuDD term={1,0},sum=term;
 for(int n=1;n<64;++n){term=ntu_dd_div(ntu_dd_mul(term,r),(double)n);sum=ntu_dd_add(sum,term);if(fabs(term.hi)<fabs(sum.hi)*0x1p-110)break;}
 return {ldexp(sum.hi,k),ldexp(sum.lo,k)};
}
NTU_DEV_BIG double ntu_erf(double x){
 if(x!=x)return x+x;
 double a=x<0?-x:x;
 if(a>=6)return x<0?-1:1;
 if(a==0)return x;
 NtuDD y=ntu_dd_mul({a,0},{a,0}),z=ntu_dd_mul(y,{2,0}),term={a,0},sum=term;
 for(int n=1;n<240;++n){term=ntu_dd_div(ntu_dd_mul(term,z),(double)(2*n+1));sum=ntu_dd_add(sum,term);if(term.hi<sum.hi*0x1p-105)break;}
 NtuDD expx=ntu_dd_exp({-y.hi,-y.lo});
 NtuDD answer=ntu_dd_mul(ntu_dd_mul({1.1283791670955126,1.533545961316588e-17},expx),sum);
 double result=__dadd_rn(answer.hi,answer.lo);if(result>1)result=1;return x<0?-result:result;
}
NTU_DEV_BIG float ntu_erff(float x){return (float)ntu_erf((double)x);}
#endif

#ifndef NTU_HOST
// WRF STOP and wrf_error_fatal terminate the model. A device trap does so
// asynchronously; polling a flag with asnumpy here synchronized every step.
// There is no shared module-global status pointer, so independent ensemble
// streams cannot overwrite one another's error target.
#define NTU_FATAL(code) do { asm volatile("trap;"); } while (0)
#define NTU_EXPF(x)  gfk_exp(x)
#define NTU_LOGF(x)  gfk_log(x)
#define NTU_POWF(x, y) gfk_pow((x), (y))
#define NTU_EXP(x)   glibc_exp(x)
#define NTU_LOG(x)   glibc_log(x)
// ERF of a REAL retains the binary64-to-binary32 route in the strict arm.
#define NTU_LOG10F(x) ntu_log10(x)
#define NTU_ERFF(x)   ntu_erff(x)
#define NTU_ERF(x)    ntu_erf(x)
#define NTU_COSF(x)   glibc_cosf(x)
#define NTU_SINF(x)   glibc_sinf(x)
#define NTU_TANHF(x)  tanhf(x)
#endif

NTU_DEV unsigned int ntu_f2u(float f) {
    union { float f; unsigned int u; } v; v.f = f; return v.u;
}
NTU_DEV float ntu_bits_f(unsigned int u) {
    union { unsigned int u; float f; } v; v.u = u; return v.f;
}
NTU_DEV double ntu_bits_d(unsigned long long u) {
    union { unsigned long long u; double f; } v; v.u = u; return v.f;
}
// gfortran folds REAL**2.0 to one multiplication.
NTU_DEV float ntu_sq(float x) { return x * x; }
NTU_DEV double ntu_sqd(double x) { return x * x; }

// Fortran MIN/MAX: a NaN operand propagates; on a tie the larger zero wins
// for MAX and the smaller for MIN (gfortran -O2's IEEE minimum/maximum).
NTU_DEV float ntu_fmax(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a > b) return a;
    if (b > a) return b;
    return (ntu_f2u(a) & 0x80000000u) ? b : a;
}
NTU_DEV float ntu_fmin(float a, float b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a < b) return a;
    if (b < a) return b;
    return (ntu_f2u(a) & 0x80000000u) ? a : b;
}
NTU_DEV double ntu_dmax(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return (a > b) ? a : ((b > a) ? b : (a == 0.0 && b == 0.0 && a < 1.0 / a ? b : a));
}
NTU_DEV double ntu_dmin(double a, double b) {
    if (a != a) return a;
    if (b != b) return b;
    return (a < b) ? a : ((b < a) ? b : a);
}
NTU_DEV int ntu_imax(int a, int b) { return a > b ? a : b; }
NTU_DEV int ntu_imin(int a, int b) { return a < b ? a : b; }
// NINT: round half away from zero.
NTU_DEV int ntu_nint(float x) { return (int)roundf(x); }

// ---- BEGIN transcription (f2cu.py) ----
// ---- module PARAMETERs, exact bits from gfortran's own folding ----
#define NTU_ID_NH42SO4 1
#define NTU_ID_DUST 0
#define NTU_ID_IN 2
#define NTU_INSPEC 2
#define NTU_ICE_SHAPE 1
#define NTU_AGG_SHAPE 1
#define NTU_ICE_RHOI 1
#define NTU_ICE_RHOS 1
#define NTU_ICE_RHOG 1
#define NTU_ICE_VENT 2
#define NTU_HAIL_VENT 1
#define NTU_HWET_MODE 1
#define NTU_LIQ_VTC 1
#define NTU_LIQ_VTR 1
#define NTU_ICE_VTI 1
#define NTU_ICE_VTS 1
#define NTU_ICE_VTG 1
#define NTU_ICE_VTH 1
#define NTU_AFAC_3M 1
#define NTU_AFAR_3M 1
#define NTU_AFAI_3M 1
#define NTU_AFAS_3M 1
#define NTU_AFAG_3M 1
#define NTU_AFAH_3M 1
#define NTU_SAT_ADJ 0
#define NTU_NCCN 3
#define NTU_NAER 2
#define NTU_MAER 4
#define NTU_NAER1 4
#define NTU_NAER2 1
#define NTU_NAERT 5
#define NTU_NTBXA 25
#define NTU_DTMIN ntu_bits_f(0x3C23D70Au) /* 0.01 */
#define NTU_DT20S ntu_bits_f(0x41A00000u) /* 20. */
#define NTU_PI ntu_bits_f(0x40490FDBu) /* 3.1415926535897932384626434 */
#define NTU_SQRTPI ntu_bits_f(0x3F6B3F8Eu) /* 9.189385332046727417803297E-1 */
#define NTU_SQRT2 ntu_bits_f(0x3FB504F3u) /* 1.4142135623730950488016887 */
#define NTU_THRD ntu_bits_f(0x3EAAAAABu) /* (1./3.) */
#define NTU_C4PI3 ntu_bits_f(0x40860A92u) /* ((4.*PI)/3.) */
#define NTU_CP ntu_bits_f(0x447B5D71u) /* 1.00546E3 */
#define NTU_TK0C ntu_bits_f(0x43889333u) /* 2.7315E2 */
#define NTU_R ntu_bits_f(0x438F876Du) /* 2.87058E2 */
#define NTU_RV ntu_bits_f(0x43E6BF5Cu) /* 4.61495E2 */
#define NTU_CPI ntu_bits_f(0x4502D000u) /* 2.093E3 */
#define NTU_CPW ntu_bits_f(0x4583D000u) /* 4.218E3 */
#define NTU_CMW ntu_bits_f(0x3C939431u) /* 1.8015E-2 */
#define NTU_RHOSU ntu_bits_f(0x3F8AB6ACu) /* (8.5E4/(2.8715E2*2.7315E2)) */
#define NTU_RHOW ntu_bits_f(0x44794000u) /* 9.97E2 */
#define NTU_RHOG1 ntu_bits_f(0x43C80000u) /* 4.E2 */
#define NTU_RHOI0 ntu_bits_f(0x44638000u) /* 9.1E2 */
#define NTU_IRHOI0 ntu_bits_f(0x3A900901u) /* (1./RHOI0) */
#define NTU_RHOS0 ntu_bits_f(0x42C80000u) /* 1.E2 */
#define NTU_RHOI1 ntu_bits_f(0x43FA0000u) /* 5.E2 */
#define NTU_RHOG0 ntu_bits_f(0x44207FFFu) /* (((0.078+(0.184*6.))-(0.015*36.))*1000) */
#define NTU_RHOH ntu_bits_f(0x44610000u) /* 9.E2 */
#define NTU_IRHOH ntu_bits_f(0x3A91A2B4u) /* (1./RHOH) */
#define NTU_RHOIMIN ntu_bits_f(0x42480000u) /* 5.E1 */
#define NTU_RHOIMAX ntu_bits_f(0x44638000u) /* RHOI0 */
#define NTU_C4PI3W ntu_bits_f(0x458281CBu) /* (((4.*PI)*RHOW)/3.) */
#define NTU_IAMI0 ntu_bits_f(0x3B098B1Fu) /* (6./(PI*RHOI0)) */
#define NTU_G ntu_bits_f(0x411CE560u) /* 9.806 */
#define NTU_IAPW ntu_bits_f(0x3AA76362u) /* (4./(PI*RHOW)) */
#define NTU_AAW ntu_bits_f(0x3F490FDBu) /* (PI/4.) */
#define NTU_V2M3 ntu_bits_f(0x3FF47645u) /* (6./PI) */
#define NTU_BMW ntu_bits_f(0x40400000u) /* 3. */
#define NTU_AMW ntu_bits_f(0x440281CBu) /* ((PI*RHOW)/6.) */
#define NTU_BMI0 ntu_bits_f(0x40400000u) /* 3. */
#define NTU_AMI0 ntu_bits_f(0x43EE3CC9u) /* ((PI*RHOI0)/6.) */
#define NTU_BMS ntu_bits_f(0x40400000u) /* 3. */
#define NTU_AMS0 ntu_bits_f(0x42517084u) /* ((PI*RHOS0)/6.) */
#define NTU_BMG ntu_bits_f(0x40400000u) /* 3. */
#define NTU_AMG0 ntu_bits_f(0x43517084u) /* ((PI*RHOG1)/6.) */
#define NTU_BMH ntu_bits_f(0x40400000u) /* 3. */
#define NTU_AMH ntu_bits_f(0x43EB9E94u) /* ((PI*RHOH)/6.) */
#define NTU_IAMW ntu_bits_f(0x3AFB1512u) /* (1./AMW) */
#define NTU_IAMH ntu_bits_f(0x3B0B125Cu) /* (1./AMH) */
#define NTU_AIMM ntu_bits_f(0x3F28F5C3u) /* 6.6E-1 */
#define NTU_BIMM ntu_bits_f(0x42C80000u) /* 1.E2 */
#define NTU_AVSG ntu_bits_f(0x3F5C28F6u) /* 8.6E-1 */
#define NTU_BVSG ntu_bits_f(0x3E8F5C29u) /* 2.8E-1 */
#define NTU_AVRH ntu_bits_f(0x3F47AE14u) /* 7.8E-1 */
#define NTU_BVRH ntu_bits_f(0x3E9DB22Du) /* 3.08E-1 */
#define NTU_AVIS ntu_bits_f(0x3F800000u) /* 1. */
#define NTU_BVIS ntu_bits_f(0x3E0F5C29u) /* 1.4E-1 */
#define NTU_VENC1 ntu_bits_f(0x3CFD21FFu) /* 3.09E-2 */
#define NTU_VENC2 ntu_bits_f(0x3E142C3Du) /* 1.447E-1 */
#define NTU_VENP1 ntu_bits_f(0x3C2C0831u) /* 1.05E-2 */
#define NTU_VENP2 ntu_bits_f(0x3CBAC711u) /* 2.28E-2 */
#define NTU_VENH1 ntu_bits_f(0x3E6538EFu) /* 0.22385 */
#define NTU_VENH2 ntu_bits_f(0x3A8461FAu) /* 0.00101 */
#define NTU_NSMALL ntu_bits_f(0x3C23D70Au) /* 1.E-2 */
#define NTU_NSMAL1 ntu_bits_f(0x42C80000u) /* 1.E2 */
#define NTU_RLIMIT ntu_bits_f(0x0A4FB11Fu) /* 1.E-32 */
#define NTU_SLIMIT ntu_bits_f(0x3C23D70Au) /* 1.E-2 */
#define NTU_QSMAL1 ntu_bits_f(0x3089705Fu) /* 1.E-9 */
#define NTU_QSMALL ntu_bits_f(0x283424DCu) /* 1.E-14 */
#define NTU_QLIMIT ntu_bits_f(0x358637BDu) /* 1.E-6 */
#define NTU_RSMALL ntu_bits_f(0x1E3CE508u) /* 1.E-20 */
#define NTU_ASMALL ntu_bits_f(0x2B8CBCCCu) /* 1.E-12 */
#define NTU_ISMALL ntu_bits_f(0x233877AAu) /* 1.E-17 */
#define NTU_BOLTZ ntu_bits_f(0x19857725u) /* 1.38E-23 */
#define NTU_MLIMIT ntu_bits_f(0x3C23D70Au) /* 1.E-2 */
#define NTU_VTZ0 ntu_bits_f(0x40BA8F5Cu) /* 5.83 */
#define NTU_VTC0 ntu_bits_f(0x3F19999Au) /* 0.6 */
#define NTU_VTC1 ntu_bits_f(0x3E1B93CDu) /* 0.151931 */
#define NTU_VTC2 ntu_bits_f(0x4107F4A2u) /* ((VTZ0**2.)/4.) */
#define NTU_VTA0 ntu_bits_f(0x3ADED289u) /* 1.7E-3 */
#define NTU_VTB0 ntu_bits_f(0x00000000u) /* 0. */
#define NTU_AVC0 ntu_bits_f(0x4BE4E1C0u) /* 3.E7 */
#define NTU_BVC0 ntu_bits_f(0x40000000u) /* 2. */
#define NTU_AVR0 ntu_bits_f(0x44527FCFu) /* 841.997 */
#define NTU_BVR0 ntu_bits_f(0x3F4CCCCDu) /* 0.8 */
#define NTU_AVI0 ntu_bits_f(0x442F0000u) /* 700. */
#define NTU_BVI0 ntu_bits_f(0x3F800000u) /* 1. */
#define NTU_AVS0 ntu_bits_f(0x413B851Fu) /* 11.72 */
#define NTU_BVS0 ntu_bits_f(0x3ED1EB85u) /* 0.41 */
#define NTU_AVG0 ntu_bits_f(0x419A6666u) /* 19.3 */
#define NTU_BVG0 ntu_bits_f(0x3EBD70A4u) /* 0.37 */
#define NTU_AVH0 ntu_bits_f(0x434EE3D7u) /* 206.89 */
#define NTU_BVH0 ntu_bits_f(0x3F236E2Fu) /* 0.6384 */
#define NTU_VTCMAX ntu_bits_f(0x3F800000u) /* 1. */
#define NTU_VTSMAX ntu_bits_f(0x41200000u) /* 10. */
#define NTU_VTIMAX ntu_bits_f(0x41200000u) /* 10. */
#define NTU_VTRMAX ntu_bits_f(0x41700000u) /* 15. */
#define NTU_VTGMAX ntu_bits_f(0x41A00000u) /* 20. */
#define NTU_VTHMAX ntu_bits_f(0x41C80000u) /* 25. */
#define NTU_AFAC0 ntu_bits_f(0x00000000u) /* 0. */
#define NTU_AFAR0 ntu_bits_f(0x00000000u) /* 0. */
#define NTU_AFAI0 ntu_bits_f(0x40400000u) /* 3. */
#define NTU_AFAS0 ntu_bits_f(0x00000000u) /* 0. */
#define NTU_AFAG0 ntu_bits_f(0x00000000u) /* 0. */
#define NTU_AFAH0 ntu_bits_f(0x00000000u) /* 0. */
#define NTU_SASMAX ntu_bits_f(0x3F800000u) /* 1. */
#define NTU_SASMIN ntu_bits_f(0x3A83126Fu) /* 1.E-3 */
#define NTU_KCCMIN ntu_bits_f(0x3E645A1Du) /* 0.223 */
#define NTU_KCCMAX ntu_bits_f(0x3F7FBE77u) /* 0.999 */
#define NTU_KCRMIN ntu_bits_f(0x3E645A1Du) /* 0.223 */
#define NTU_KCRMAX ntu_bits_f(0x3F7FBE77u) /* 0.999 */
#define NTU_KCIMIN ntu_bits_f(0x3F0E5604u) /* 0.556 */
#define NTU_KCIMAX ntu_bits_f(0x3F7FBE77u) /* 0.999 */
#define NTU_KCSMIN ntu_bits_f(0x3E645A1Du) /* 0.223 */
#define NTU_KCSMAX ntu_bits_f(0x3F7FBE77u) /* 0.999 */
#define NTU_KCGMIN ntu_bits_f(0x3E645A1Du) /* 0.223 */
#define NTU_KCGMAX ntu_bits_f(0x3F7FBE77u) /* 0.999 */
#define NTU_KCHMIN ntu_bits_f(0x3E645A1Du) /* 0.223 */
#define NTU_KCHMAX ntu_bits_f(0x3F7FBE77u) /* 0.999 */
#define NTU_AFAMAX ntu_bits_f(0x46EA6000u) /* 3.E4 */
#define NTU_AFAMIN ntu_bits_f(0x00000000u) /* 0. */
#define NTU_AFU ntu_bits_f(0x3EA00000u) /* 3.125E-1 */
#define NTU_BFU ntu_bits_f(0x3A8A4EA3u) /* 1.0552E-3 */
#define NTU_CFU ntu_bits_f(0xC019BF48u) /* (-2.4023) */
#define NTU_DCMIN ntu_bits_f(0x33D6BF95u) /* 1.E-7 */
#define NTU_DCMAX ntu_bits_f(0x38D1B717u) /* 1.E-4 */
#define NTU_DIMIN ntu_bits_f(0x358637BDu) /* 1.E-6 */
#define NTU_DIMAX ntu_bits_f(0x3BA3D70Au) /* 5.E-3 */
#define NTU_DRMIN ntu_bits_f(0x37FBA882u) /* 3.E-5 */
#define NTU_DRMAX ntu_bits_f(0x3BC49BA6u) /* 6.E-3 */
#define NTU_DSMIN ntu_bits_f(0x37A7C5ACu) /* 2.E-5 */
#define NTU_DSMAX ntu_bits_f(0x3C23D70Au) /* 1.E-2 */
#define NTU_DGMIN ntu_bits_f(0x3851B717u) /* 5.E-5 */
#define NTU_DGMAX ntu_bits_f(0x3CA3D70Au) /* 2.E-2 */
#define NTU_DHMIN ntu_bits_f(0x3A83126Fu) /* 1.E-3 */
#define NTU_DHMAX ntu_bits_f(0x3D23D70Au) /* 4.E-2 */
#define NTU_RCMIN ntu_bits_f(0x3356BF95u) /* (DCMIN/2.) */
#define NTU_RCMAX ntu_bits_f(0x3851B717u) /* (DCMAX/2.) */
#define NTU_RRMIN ntu_bits_f(0x377BA882u) /* (DRMIN/2.) */
#define NTU_RRMAX ntu_bits_f(0x3B449BA6u) /* (DRMAX/2.) */
#define NTU_DI0 ntu_bits_f(0x36C9539Cu) /* 6.E-6 */
#define NTU_DCR ntu_bits_f(0x38D1B717u) /* 100.E-6 */
#define NTU_MI0 ntu_bits_f(0x29E7C086u) /* (AMI0*(DI0**BMI0)) */
#define NTU_SIG1 ntu_bits_f(0xC0C8978Du) /* (-6.2685) */
#define NTU_SIG2 ntu_bits_f(0xBE8BD662u) /* (-2.7312E-1) */
#define NTU_SIG3 ntu_bits_f(0x3E677C46u) /* 2.2606E-1 */
#define NTU_MNR1 ntu_bits_f(0xC012B021u) /* (-2.2920) */
#define NTU_MNR2 ntu_bits_f(0xBEB4024Bu) /* (-3.5158E-1) */
#define NTU_MNR3 ntu_bits_f(0x3EB1B478u) /* 3.4708E-1 */
#define NTU_EFC1 ntu_bits_f(0xBFA0C49Cu) /* (-1.2560) */
#define NTU_EFC2 ntu_bits_f(0xBC92AB69u) /* (-1.7904E-02) */
#define NTU_EFC3 ntu_bits_f(0x3F5AF8DFu) /* 8.5536E-01 */
#define NTU_EFR1 ntu_bits_f(0xBDCB31C2u) /* (-9.9216E-02) */
#define NTU_EFR2 ntu_bits_f(0x3CF19503u) /* 2.9490E-02 */
#define NTU_EFR3 ntu_bits_f(0x3F7E0C9Eu) /* 9.9238E-01 */
// ---- constant subexpressions gfortran folds at compile time ----
#define NTU_K0 ntu_bits_d(0x3FF6A09E667F3BCDull) /* DSQRT(2.D+0) */
#define NTU_K1 ntu_bits_d(0x3FEFFFFECC35D0DFull) /* DERF((5.D+0/DSQRT(2.D+0))) */
#define NTU_K2 ntu_bits_d(0x400921FB54442D18ull) /* DACOS((-1.D+0)) */
#define NTU_K3 ntu_bits_f(0x257907F4u) /* (DI0**3.) */
#define NTU_K4 ntu_bits_d(0xC034B927F32BFFB8ull) /* DLOG(1.D-9) */
#define NTU_K5 ntu_bits_d(0xC01BA18A998FFFA0ull) /* DLOG(1.D-3) */
#define NTU_K6 ntu_bits_d(0x3FF6A09E667F3BCDull) /* DSQRT(DBLE(2.)) */
#define NTU_K7 ntu_bits_f(0x30897061u) /* (DHMIN**BMH) */
#define NTU_K8 ntu_bits_f(0x30897061u) /* ((1.E-3)**BMW) */
// ---- module DATA tables ----
NTU_CONST_TABLE int NTU_NAERN[2] = {NTU_NAER1, NTU_NAER2};
NTU_CONST_TABLE float NTU_ITBLE[121] = {1.000000f, 0.979490f, 0.959401f, 0.939723f, 0.920450f, 0.899498f, 0.879023f, 0.857038f, 0.833681f, 0.810961f, 0.783430f, 0.755092f, 0.703072f, 0.537032f, 0.467735f, 0.524807f, 0.630957f, 0.812831f, 1.096478f, 1.479108f, 1.905461f, 2.089296f, 2.290868f, 2.398833f, 2.454709f, 2.426610f, 2.371374f, 2.290868f, 2.137962f, 1.995262f, 1.862087f, 1.737801f, 1.621810f, 1.513561f, 1.396368f, 1.288250f, 1.188502f, 1.096478f, 1.000000f, 0.922571f, 0.851138f, 0.785236f, 0.724436f, 0.668344f, 0.616595f, 0.575440f, 0.537032f, 0.501187f, 0.467735f, 0.436516f, 0.407380f, 0.380189f, 0.354813f, 0.331131f, 0.316228f, 0.301995f, 0.291743f, 0.285102f, 0.281838f, 0.278612f, 0.275423f, 0.278612f, 0.281838f, 0.285102f, 0.291743f, 0.298538f, 0.309030f, 0.319890f, 0.331131f, 0.346737f, 0.367282f, 0.393550f, 0.426580f, 0.457088f, 0.489779f, 0.524807f, 0.562341f, 0.609537f, 0.660693f, 0.716143f, 0.785236f, 0.860994f, 0.954993f, 1.047129f, 1.148154f, 1.258925f, 1.380384f, 1.496236f, 1.603245f, 1.698244f, 1.778279f, 1.840772f, 1.883649f, 1.905461f, 1.905461f, 1.883649f, 1.862087f, 1.840772f, 1.798871f, 1.737801f, 1.698244f, 1.640590f, 1.584893f, 1.548817f, 1.513561f, 1.475707f, 1.452112f, 1.428894f, 1.412538f, 1.393157f, 1.377209f, 1.361445f, 1.348963f, 1.336596f, 1.327394f, 1.318257f, 1.309182f, 1.303167f, 1.294196f, 1.288250f, 1.279381f};
NTU_CONST_TABLE float NTU_IECC[168] = {0.00f, 0.03f, 0.29f, 0.25f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.10f, 0.42f, 0.50f, 0.47f, 0.21f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.15f, 0.49f, 0.56f, 0.55f, 0.46f, 0.10f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.20f, 0.52f, 0.61f, 0.62f, 0.59f, 0.49f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.25f, 0.62f, 0.72f, 0.75f, 0.74f, 0.71f, 0.68f, 0.57f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.30f, 0.70f, 0.80f, 0.84f, 0.85f, 0.85f, 0.84f, 0.83f, 0.81f, 0.77f, 0.69f, 0.10f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.35f, 0.75f, 0.85f, 0.89f, 0.91f, 0.92f, 0.92f, 0.91f, 0.90f, 0.89f, 0.87f, 0.85f, 0.82f, 0.71f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.40f, 0.80f, 0.89f, 0.93f, 0.94f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.94f, 0.93f, 0.92f, 0.91f, 0.88f, 0.80f, 0.15f, 0.00f, 0.00f};
NTU_CONST_TABLE float NTU_IEPC[168] = {0.00f, 0.00f, 0.13f, 0.41f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.25f, 0.54f, 0.56f, 0.39f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.08f, 0.57f, 0.75f, 0.83f, 0.85f, 0.86f, 0.84f, 0.78f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.10f, 0.65f, 0.80f, 0.87f, 0.90f, 0.91f, 0.91f, 0.91f, 0.88f, 0.78f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.12f, 0.67f, 0.81f, 0.88f, 0.91f, 0.92f, 0.93f, 0.93f, 0.92f, 0.91f, 0.89f, 0.81f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.15f, 0.68f, 0.82f, 0.89f, 0.91f, 0.93f, 0.94f, 0.94f, 0.95f, 0.95f, 0.95f, 0.94f, 0.91f, 0.83f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.18f, 0.69f, 0.82f, 0.89f, 0.92f, 0.94f, 0.95f, 0.95f, 0.96f, 0.96f, 0.96f, 0.96f, 0.95f, 0.94f, 0.91f, 0.80f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.22f, 0.70f, 0.83f, 0.90f, 0.93f, 0.95f, 0.96f, 0.97f, 0.98f, 0.98f, 0.98f, 0.98f, 0.97f, 0.96f, 0.94f, 0.92f, 0.84f, 0.00f, 0.00f, 0.00f};
NTU_CONST_TABLE float NTU_AMS1[45] = {2.728E+01f, 1.258E+00f, 6.452E-02f, 6.259E-03f, 8.658E-04f, 4.657E+01f, 1.193E+00f, 3.790E-02f, 2.688E-03f, 3.101E-04f, 2.260E+01f, 1.000E+00f, 5.059E-02f, 4.889E-03f, 6.826E-04f, 6.176E+00f, 6.002E-01f, 5.889E-02f, 8.892E-03f, 1.647E-03f, 2.982E+00f, 1.795E-01f, 1.261E-02f, 1.647E-03f, 3.128E-04f, 2.783E+00f, 2.179E-01f, 1.901E-02f, 2.825E-03f, 5.699E-04f, 1.630E+00f, 3.267E-01f, 6.223E-02f, 1.493E-02f, 3.902E-03f, 1.145E+01f, 3.346E-02f, 1.178E-04f, 7.701E-05f, 2.463E-05f, 6.606E+01f, 1.799E-03f, 1.178E-04f, 7.701E-05f, 2.463E-05f};
NTU_CONST_TABLE float NTU_BMS1[45] = {2.792f, 2.455f, 2.085f, 1.748f, 1.411f, 2.846f, 2.449f, 2.015f, 1.618f, 1.221f, 2.773f, 2.429f, 2.053f, 1.710f, 1.367f, 2.642f, 2.371f, 2.073f, 1.802f, 1.530f, 2.556f, 2.254f, 1.923f, 1.621f, 1.320f, 2.549f, 2.276f, 1.977f, 1.704f, 1.431f, 2.495f, 2.322f, 2.133f, 1.960f, 1.787f, 2.686f, 2.064f, 1.382f, 1.382f, 1.382f, 2.863f, 1.732f, 1.382f, 1.382f, 1.382f};
NTU_CONST_TABLE float NTU_AAS1[45] = {1.782E+00f, 2.778E-01f, 4.794E-02f, 1.241E-02f, 4.069E-03f, 2.571E+00f, 2.635E-01f, 3.180E-02f, 6.517E-03f, 1.879E-03f, 1.910E+00f, 2.522E-01f, 3.759E-02f, 8.905E-03f, 2.787E-03f, 6.238E-01f, 1.697E-01f, 4.686E-02f, 1.662E-02f, 6.761E-03f, 5.604E-01f, 9.194E-02f, 1.726E-02f, 5.030E-03f, 1.927E-03f, 2.802E-01f, 8.345E-02f, 2.602E-02f, 1.042E-02f, 4.854E-03f, 1.269E-01f, 1.106E-01f, 8.453E-02f, 5.913E-02f, 3.763E-02f, 3.569E-01f, 2.314E-02f, 1.664E-03f, 1.397E-03f, 8.521E-04f, 6.339E-01f, 5.656E-03f, 1.664E-03f, 1.397E-03f, 8.521E-04f};
NTU_CONST_TABLE float NTU_BAS1[45] = {2.133f, 1.938f, 1.725f, 1.531f, 1.337f, 2.170f, 1.932f, 1.671f, 1.432f, 1.194f, 2.140f, 1.927f, 1.693f, 1.480f, 1.267f, 2.027f, 1.882f, 1.722f, 1.576f, 1.431f, 2.011f, 1.821f, 1.612f, 1.422f, 1.232f, 1.941f, 1.810f, 1.666f, 1.534f, 1.403f, 1.861f, 1.842f, 1.821f, 1.801f, 1.782f, 1.960f, 1.669f, 1.350f, 1.350f, 1.350f, 2.018f, 1.509f, 1.350f, 1.350f, 1.350f};
// ---- module SAVE state set by AERO_CONST (lives in NtuAeroConst) ----
struct NtuAeroConst {
    int IBAER[2];
    double TBLRC[25];
    double TBLXF[50];
    float DNC0;
    float DNC1;
    float DNC2;
    float DNC3;
    float SENS[2];
    float ASH[2];
    float CMAS[2];
    float DNAS[2];
    float AVAN[2];
    float BETA1[2];
    float RXMIN[2];
    float ZCCN[6];
    float CNMOD[6];
    float CNSTD[6];
    float WMAS[6];
    float RFACT[6];
};

NTU_DEV double ntu_dpdf(double X);

NTU_DEV double ntu_dyequ(double X, double* WEGHT, double* XLBAR, double* SIGMA);

NTU_DEV double ntu_guess_rc(double XAFRC, double* TBLXA, double* TBLRC);

NTU_DEV double ntu_pdf(double X);

NTU_DEV double ntu_yequ(double X, double XAFRC, double* WEGHT, double* XLBAR, double* SIGMA);

NTU_DEV_BIG void ntu_find_rc0(double XAFRC, float* CMODE, float* CSTDV, float* WMAS, float& RC, double* TBLXA, double* TBLRC);

NTU_DEV float ntu_polysvp(float T, int TYPE);

NTU_DEV float ntu_gamln(float XX);

NTU_DEV void ntu_cfg(float& GAMMCF, float A, float X, float& GLN);

NTU_DEV void ntu_gser(float& GAMSER, float A, float X, float& GLN);

NTU_DEV float ntu_gammp(float A, float X);

NTU_DEV_BIG void ntu_solve_afac(NtuAeroConst* __restrict__ AC, float TK1D, float QC1D, float& NC1D, float& LAMC, float& MVDC, float& AFAC);

NTU_DEV void ntu_solve_afar(float TK1D, float QR1D, float& NR1D, float& LAMR, float& MVDR, float& AFAR);

NTU_DEV_BIG void ntu_solve_afai(float TK1D, float P1D, float RHO, float QV1D, float QI1D, float& NI1D, float& VI1D, float& FI1D, float& I2M1D, float& I3M1D, float& ADAGR, float& ZETA, float& LAMI, float& AFAI, float& MVDI, float& RHOI, float& AMI, float& BMI, float& AVI, float& BVI, float& BEST);

NTU_DEV_BIG void ntu_solve_afas(float TK1D, float RHO, float QS1D, float QC1D, float& NS1D, float& VS1D, float& FS1D, float& S2M1D, float& AFAS, float& LAMS, float& MVDS, float& RHOS, float& SASPR, float& AMS, float& AVS, float& BVS);

NTU_DEV_BIG void ntu_solve_afag(float TK1D, float RHO, float QG1D, float QC1D, float& NG1D, float& VG1D, float& G2M1D, float& LAMG, float& AFAG, float& MVDG, float& RHOG, float& AMG, float& AVG, float& BVG);

NTU_DEV_BIG void ntu_solve_afah(float TK1D, float RHO, float QH1D, float& NH1D, float& H2M1D, float& LAMH, float& AFAH, float& MVDH, float& AVH, float& BVH);

NTU_DEV_BIG void ntu_aero_const(NtuAeroConst* __restrict__ AC, int CCNTY);

NTU_DEV void ntu_ccnwhitby(float W1D, float RSX, float RS9, float& ZCN9, float& RACT, float BETA1, float ALPHA, float* ZCCN, float* CNMOD, float* CNSTD);

NTU_DEV double ntu_dlnx(float RX, float XMODE, float SIGMA, int N);

NTU_DEV void ntu_rswhitby(float W1D, float RSX, float& Z, float& RACT, float BETA1, float ALPHA, float* ZCCN, float* CNMOD, float* CNSTD);

NTU_DEV_BIG void ntu_activa(NtuAeroConst* __restrict__ AC, float TK1D, float W1D, float& NC1D, float& NR1D, float* ABCD, float& QACAC, float& QACAR, float& RX0, float* ZCCNS, float SSW, int IAE, bool& DO_FR);

NTU_DEV double ntu_dlnx2(double DRX, float XMODE, float SIGMA, int N);

NTU_DEV_BIG void ntu_deactiva(NtuAeroConst* __restrict__ AC, float NACCV, float QC1D, float& NC1D, float& DCN, float TCN, float& WCN, float& RC, float* ZCCNS, int IAE, bool& DO_FR);

NTU_DEV void ntu_flflux(float* Q1D, float* VT1D, float* RHO, float* DZ, int NK, float DT);

NTU_DEV_BIG void ntu_icenu(float& TK1D, float P1D, float DT, float RHO, float& QV1D, float& QI1D, float& NI1D, float& VI1D, float& FI1D, float& AI1D, float& I3M1D, float& QNIN);

NTU_DEV_BIG void ntu_large_dt(NtuAeroConst* __restrict__ AC, float DT, float& TK1D, float QV1D, float P1D, float RHO, float& QC1D, float& QR1D, float& QI1D, float& QS1D, float& QG1D, float& QH1D, float& NC1D, float& NR1D, float& NI1D, float& NS1D, float& NG1D, float& NH1D, float& VI1D, float& VS1D, float& VG1D, float& FI1D, float& FS1D, float& AI1D, float& AS1D, float& AG1D, float& AH1D, float& I3M1D, float& SASPR, float& GQCTR);

NTU_DEV void ntu_ptflux(float* Q1D, float* VT1D, float* RHO, float* DZ, int NK, float DT, float DTMN, float& PRT1D);

NTU_DEV_BIG void ntu_sedi_fall(NtuAeroConst* __restrict__ AC, float TK1D, float P1D, float QV1D, float QC1D, float QR1D, float QI1D, float QS1D, float QG1D, float QH1D, float& NC1D, float& NR1D, float& NI1D, float& NS1D, float& NG1D, float& NH1D, float& VI1D, float& VS1D, float& VG1D, float& FI1D, float& FS1D, float& AI1D, float& AS1D, float& AG1D, float& AH1D, float& I3M1D, float& VTQC, float& VTQR, float& VTQI, float& VTQS, float& VTQG, float& VTQH, float& VTNC, float& VTNR, float& VTNI, float& VTNS, float& VTNG, float& VTNH, float& VTVI, float& VTVS, float& VTVG, float& VTFI, float& VTFS, float& VTAI, float& VTAS, float& VTAG, float& VTAH, float& VTI3M, float& SASPR, float RHO);

NTU_DEV_BIG void ntu_small_dt(NtuAeroConst* __restrict__ AC, float DT, float& DTS, float SDTS, float DTKDT, float DQVDT, float DPDT, float RHO, float& TK1D, float P1D, float& QV1D, float& QC1D, float& QR1D, float& QI1D, float& QS1D, float& QG1D, float& QH1D, float& NC1D, float& NR1D, float& NI1D, float& NS1D, float& NG1D, float& NH1D, float& VI1D, float& VS1D, float& VG1D, float& FI1D, float& FS1D, float& AI1D, float& AS1D, float& AG1D, float& AH1D, float& I3M1D, float& SASPR, float& XDNC, float& XDNR, float& GQCTR);

NTU_DEV_BIG void ntu_ntu_micro(NtuAeroConst* __restrict__ AC, float* TK3D, float* QV3D, float* DZ3D, float* P3D, float* S3D, float* W3D, float& SR, float& ICENCV, float& VINCV, float& VSNCV, float& VGNCV, float& FINCV, float& FSNCV, float& AINCV, float& ASNCV, float& AGNCV, float& AHNCV, float& I3MNCV, float& CLODNCV, float& RAINNC, float& RAINNCV, float& SNOWNC, float& SNOWNCV, float& GRAPNC, float& GRAPNCV, float& HAILNC, float& HAILNCV, float DT, float DTMN, float* QC3D, float* QR3D, float* QI3D, float* QS3D, float* QG3D, float* QH3D, float* NC3D, float* NR3D, float* NI3D, float* NS3D, float* NG3D, float* NH3D, float* VI3D, float* VS3D, float* VG3D, float* FI3D, float* FS3D, float* AI3D, float* AS3D, float* AG3D, float* AH3D, float* I3M3D, float* QDCN3D, float* QTCN3D, float* QCCN3D, float* QRCN3D, float* QNIN3D, int KTS, int KTE);

// WRF FUNCTION DPDF (module_mp_ntu.F:453-459)
NTU_DEV double ntu_dpdf(double X) {
    double DPI = 0.0;
    double NTU_RET = 0;
    DPI = NTU_K2;
    NTU_RET = (NTU_EXP((-(X * X))) / sqrt(DPI));
    return NTU_RET;
}

// WRF FUNCTION DYEQU (module_mp_ntu.F:431-444)
NTU_DEV double ntu_dyequ(double X, double* WEGHT, double* XLBAR, double* SIGMA) {
    double DX = 0.0, DSQRT2 = 0.0;
    int I = 0;
    double NTU_RET = 0;
    DSQRT2 = NTU_K0;
    NTU_RET = 0.E+0;
    { const int ntu_hi_i = NTU_NCCN;
    for (I = 1; I <= ntu_hi_i; ++I) {
        DX = (((X - XLBAR[(I) - (1)]) / SIGMA[(I) - (1)]) / DSQRT2);
        NTU_RET = (NTU_RET + (((WEGHT[(I) - (1)] * ntu_dpdf(DX)) / SIGMA[(I) - (1)]) / DSQRT2));
    } }
    return NTU_RET;
}

// WRF FUNCTION GUESS_RC (module_mp_ntu.F:381-411)
NTU_DEV double ntu_guess_rc(double XAFRC, double* TBLXA, double* TBLRC) {
    int IBNG = 0, IEND = 0, I = 0;
    double NTU_RET = 0;
    if ((XAFRC >= TBLXA[(NTU_NTBXA) - (1)])) {
        NTU_RET = TBLRC[(NTU_NTBXA) - (1)];
        return NTU_RET;
    } else if ((XAFRC < TBLXA[(1) - (1)])) {
        NTU_RET = TBLRC[(1) - (1)];
        return NTU_RET;
    }
    if ((XAFRC < TBLXA[((NTU_NTBXA / 4)) - (1)])) {
        IBNG = 1;
        IEND = ((NTU_NTBXA / 4) - 1);
    } else if ((XAFRC < TBLXA[(((NTU_NTBXA * 2) / 4)) - (1)])) {
        IBNG = (NTU_NTBXA / 4);
        IEND = (((NTU_NTBXA * 2) / 4) - 1);
    } else if ((XAFRC < TBLXA[(((NTU_NTBXA * 3) / 4)) - (1)])) {
        IBNG = ((NTU_NTBXA * 2) / 4);
        IEND = (((NTU_NTBXA * 3) / 4) - 1);
    } else {
        IBNG = ((NTU_NTBXA * 3) / 4);
        IEND = (NTU_NTBXA - 1);
    }
    { const int ntu_hi_i = IBNG;
    for (I = IEND; I >= ntu_hi_i; --I) {
        if ((XAFRC >= TBLXA[(I) - (1)])) {
            goto L_111;
        }
    } }
    L_111:;
    NTU_RET = TBLRC[(I) - (1)];
    return NTU_RET;
}

// WRF FUNCTION PDF (module_mp_ntu.F:446-451)
NTU_DEV double ntu_pdf(double X) {
    double NTU_RET = 0;
    NTU_RET = ((1.E+0 + NTU_ERF(X)) * 5.E-1);
    return NTU_RET;
}

// WRF FUNCTION YEQU (module_mp_ntu.F:415-429)
NTU_DEV double ntu_yequ(double X, double XAFRC, double* WEGHT, double* XLBAR, double* SIGMA) {
    double DX = 0.0, DSQRT2 = 0.0;
    int I = 0;
    double NTU_RET = 0;
    DSQRT2 = NTU_K0;
    NTU_RET = 0.E+0;
    { const int ntu_hi_i = NTU_NCCN;
    for (I = 1; I <= ntu_hi_i; ++I) {
        DX = (((X - XLBAR[(I) - (1)]) / SIGMA[(I) - (1)]) / DSQRT2);
        NTU_RET = (NTU_RET + (WEGHT[(I) - (1)] * ntu_pdf(DX)));
    } }
    NTU_RET = (NTU_RET - XAFRC);
    return NTU_RET;
}

// WRF SUBROUTINE FIND_RC0 (module_mp_ntu.F:318-377)
NTU_DEV_BIG void ntu_find_rc0(double XAFRC, float* CMODE, float* CSTDV, float* WMAS, float& RC, double* TBLXA, double* TBLRC) {
    const int ITERMAX = 50;
    const float RC_MAX = 99.0f;
    const float RC_MIN = 1.0E-9f;
    const float XTOR = 1.0E-5f;
    double DWMAS[3] = {};
    double DMODE[3] = {};
    double DSTDV[3] = {};
    double DRC = 0.0, UL = 0.0, X1MAFRC = 0.0, Y0 = 0.0, Y = 0.0, DY = 0.0, DX = 0.0;
    int IM = 0, I = 0;
    { const int ntu_hi_im = NTU_NCCN;
    for (IM = 1; IM <= ntu_hi_im; ++IM) {
        DWMAS[(IM) - (1)] = (double)(WMAS[(IM) - (1)]);
        DSTDV[(IM) - (1)] = (double)(CSTDV[(IM) - (1)]);
        DMODE[(IM) - (1)] = (NTU_LOG((double)(CMODE[(IM) - (1)])) + (double)(((3.0f * CSTDV[(IM) - (1)]) * CSTDV[(IM) - (1)])));
    } }
    UL = (((double)(1.0f) - NTU_K1) * (double)(0.5f));
    X1MAFRC = (1.E+0 - XAFRC);
    if ((XAFRC < 1.E-10)) {
        RC = RC_MIN;
        return;
    }
    if ((X1MAFRC < UL)) {
        RC = RC_MAX;
        return;
    }
    DRC = ntu_guess_rc(XAFRC, TBLXA, TBLRC);
    Y0 = ntu_yequ(DRC, XAFRC, DWMAS, DMODE, DSTDV);
    { const int ntu_hi_i = ITERMAX;
    for (I = 1; I <= ntu_hi_i; ++I) {
        Y = Y0;
        DY = ntu_dyequ(DRC, DWMAS, DMODE, DSTDV);
        if ((DY <= 1.E-50)) {
            /* PRINT *,'IN FIND_RC0.F DY IS',DY */
            /* PRINT *,I,DX,Y,DY,DEXP(DRC),DRC,XAFRC,X1MAFRC,DMODE,DSTDV, C */
            NTU_FATAL(NTU_ERR_STOP);
        }
        DX = (-(Y / DY));
        if ((fabs(DX) > 3.E+0)) {
            DX = (DX / fabs((DX / (double)(0.4f))));
        }
        if ((fabs(DX) < (double)(XTOR))) {
            RC = (float)(NTU_EXP((DRC + DX)));
            RC = ntu_fmin(RC_MAX, ntu_fmax(RC, RC_MIN));
            return;
        }
        Y0 = ntu_yequ((DRC + DX), XAFRC, DWMAS, DMODE, DSTDV);
        if (((Y * Y0) < (double)(0.0f))) {
            DX = (DX * 5.E-1);
            Y0 = ntu_yequ((DRC + DX), XAFRC, DWMAS, DMODE, DSTDV);
        }
        DRC = (DRC + DX);
    } }
    RC = (float)(NTU_EXP(DRC));
}

// WRF FUNCTION POLYSVP (module_mp_ntu.F:607-638)
NTU_DEV float ntu_polysvp(float T, int TYPE) {
    const float A0I = 6.11147274f;
    const float A1I = 0.503160820f;
    const float A2I = 0.188439774E-1f;
    const float A3I = 0.420895665E-3f;
    const float A4I = 0.615021634E-5f;
    const float A5I = 0.602588177E-7f;
    const float A6I = 0.385852041E-9f;
    const float A7I = 0.146898966E-11f;
    const float A8I = 0.252751365E-14f;
    const float A0 = 6.11239921f;
    const float A1 = 0.443987641f;
    const float A2 = 0.142986287E-1f;
    const float A3 = 0.264847430E-3f;
    const float A4 = 0.302950461E-5f;
    const float A5 = 0.206739458E-7f;
    const float A6 = 0.640689451E-10f;
    const float A7 = -0.952447341E-13f;
    const float A8 = -0.976195544E-15f;
    float DUM = 0.0f, DT = 0.0f;
    float NTU_RET = 0;
    if ((TYPE == 1)) {
        DT = ntu_fmax((-80.0f), (T - 273.16f));
        NTU_RET = (A0I + (DT * (A1I + (DT * (A2I + (DT * (A3I + (DT * (A4I + (DT * (A5I + (DT * (A6I + (DT * (A7I + (A8I * DT))))))))))))))));
        NTU_RET = (NTU_RET * 100.0f);
    }
    if ((TYPE == 0)) {
        DT = ntu_fmax((-80.0f), (T - 273.16f));
        NTU_RET = (A0 + (DT * (A1 + (DT * (A2 + (DT * (A3 + (DT * (A4 + (DT * (A5 + (DT * (A6 + (DT * (A7 + (A8 * DT))))))))))))))));
        NTU_RET = (NTU_RET * 100.0f);
    }
    return NTU_RET;
}

// WRF FUNCTION GAMLN (module_mp_ntu.F:642-670)
NTU_DEV float ntu_gamln(float XX) {
    const double STP = 2.5066282746310005E0;
    const double COF[6] = {76.18009172947146E0, -86.50532032941677E0, 24.01409824083091E0, -1.231739572450155E0, .1208650973866179E-2, -.5395239384953E-5};
    double SER = 0.0, TMP = 0.0, X = 0.0, Y = 0.0;
    int J = 0;
    float NTU_RET = 0;
    X = (double)(XX);
    Y = X;
    TMP = (X + 5.5E0);
    TMP = (((X + 0.5E0) * NTU_LOG(TMP)) - TMP);
    SER = 1.000000000190015E0;
    { const int ntu_hi_j = 6;
    for (J = 1; J <= ntu_hi_j; ++J) {
        Y = (Y + 1.E0);
        SER = (SER + (COF[(J) - (1)] / Y));
    } }
    NTU_RET = (float)((TMP + NTU_LOG(((STP * SER) / X))));
    return NTU_RET;
}

// WRF SUBROUTINE CFG (module_mp_ntu.F:734-767)
NTU_DEV void ntu_cfg(float& GAMMCF, float A, float X, float& GLN) {
    const int ITMAX = 500;
    const float EPS = 3.0E-7f;
    const float FPMIN = 1.0E-30f;
    float AN = 0.0f, B = 0.0f, C = 0.0f, D = 0.0f, DE1 = 0.0f, H = 0.0f;
    int I = 0;
    GLN = ntu_gamln(A);
    B = ((X + 1.0f) - A);
    C = (1.0f / FPMIN);
    D = (1.0f / B);
    H = D;
    { const int ntu_hi_i = ITMAX;
    for (I = 1; I <= ntu_hi_i; ++I) {
        AN = (-((float)(I) * ((float)(I) - A)));
        B = (B + 2.0f);
        D = ((AN * D) + B);
        if ((fabsf(D) < FPMIN)) {
            D = FPMIN;
        }
        C = (B + (AN / C));
        if ((fabsf(C) < FPMIN)) {
            C = FPMIN;
        }
        D = (1.0f / D);
        DE1 = (D * C);
        H = (H * DE1);
        if ((fabsf((DE1 - 1.0f)) < EPS)) {
            goto L_888;
        }
    } }
    NTU_FATAL(NTU_ERR_FATAL); /* 'Warning : ITMAX too small in gcf' */
    L_888:;
    GAMMCF = (NTU_EXPF((((-X) + (A * NTU_LOGF(X))) - GLN)) * H);
    return;
}

// WRF SUBROUTINE GSER (module_mp_ntu.F:702-732)
NTU_DEV void ntu_gser(float& GAMSER, float A, float X, float& GLN) {
    const int ITMAX = 500;
    const float EPS = 3.0E-7f;
    float AP = 0.0f, DE1 = 0.0f, SUMM = 0.0f;
    int N = 0;
    GLN = ntu_gamln(A);
    if ((X <= 0.0f)) {
        if ((X < 0.0f)) {
            NTU_FATAL(NTU_ERR_FATAL); /*  'WARNING: X <0 in GSER'  */
        }
        GAMSER = 0.0f;
        return;
    }
    AP = A;
    SUMM = (1.0f / A);
    DE1 = SUMM;
    { const int ntu_hi_n = ITMAX;
    for (N = 1; N <= ntu_hi_n; ++N) {
        AP = (AP + 1.0f);
        DE1 = ((DE1 * X) / AP);
        SUMM = (SUMM + DE1);
        if ((fabsf(DE1) < (fabsf(SUMM) * EPS))) {
            goto L_777;
        }
    } }
    NTU_FATAL(NTU_ERR_FATAL); /* 'Warning : ITMAX too small in GSER' */
    L_777:;
    GAMSER = (SUMM * NTU_EXPF((((-X) + (A * NTU_LOGF(X))) - GLN)));
    return;
}

// WRF FUNCTION GAMMP (module_mp_ntu.F:683-700)
NTU_DEV float ntu_gammp(float A, float X) {
    float GAMMCF = 0.0f, GAMSER = 0.0f, GLN = 0.0f;
    float NTU_RET = 0;
    if (((X < 0.0f) || (A <= 0.0f))) {
        NTU_FATAL(NTU_ERR_FATAL); /* 'warning : bad arguments in gammq' */
    }
    if ((X < (A + 1.0f))) {
        ntu_gser(GAMSER, A, X, GLN);
        NTU_RET = GAMSER;
    } else {
        ntu_cfg(GAMMCF, A, X, GLN);
        NTU_RET = (1.0f - GAMMCF);
    }
    return NTU_RET;
    return NTU_RET;
}

// WRF SUBROUTINE SOLVE_AFAC (module_mp_ntu.F:771-816)
NTU_DEV_BIG void ntu_solve_afac(NtuAeroConst* __restrict__ AC, float TK1D, float QC1D, float& NC1D, float& LAMC, float& MVDC, float& AFAC) {
    float GC1 = 0.0f, LAMCMIN = 0.0f, LAMCMAX = 0.0f, C3M1D = 0.0f, SIGC = 0.0f, MNRC = 0.0f, MVRC = 0.0f, EFRC = 0.0f, KDX = 0.0f, LTK = 0.0f, LQC = 0.0f;
    if ((NC1D < NTU_NSMALL)) {
        LTK = NTU_LOGF(TK1D);
        LQC = (-(1.0f * NTU_LOGF(QC1D)));
        NC1D = NTU_EXPF(((((AC->DNC0 + (AC->DNC1 * LTK)) + (AC->DNC2 * ntu_sq(LTK))) + (AC->DNC3 * NTU_POWF(LTK, 3.0f))) - (0.25f * LQC)));
    }
    if (((QC1D >= NTU_QSMAL1) && (NC1D < NTU_NSMAL1))) {
        LTK = NTU_LOGF(TK1D);
        LQC = (-(1.0f * NTU_LOGF(QC1D)));
        NC1D = NTU_EXPF(((((AC->DNC0 + (AC->DNC1 * LTK)) + (AC->DNC2 * ntu_sq(LTK))) + (AC->DNC3 * NTU_POWF(LTK, 3.0f))) - (0.25f * LQC)));
    }
    C3M1D = (__fdiv_rn((QC1D * NTU_V2M3), NTU_RHOW));
    MVRC = NTU_POWF((__fdiv_rn((QC1D / NC1D), NTU_C4PI3W)), NTU_THRD);
    MVRC = ntu_fmin(ntu_fmax(MVRC, NTU_RCMIN), NTU_RCMAX);
    EFRC = NTU_EXPF(((NTU_EFC1 + (NTU_EFC2 * NTU_LOGF(NC1D))) + (NTU_EFC3 * NTU_LOGF(MVRC))));
    KDX = ntu_fmax(NTU_KCCMIN, ntu_fmin(NTU_KCCMAX, NTU_POWF((MVRC / EFRC), 3.0f)));
    AFAC = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
    AFAC = ntu_fmin(ntu_fmax(AFAC, NTU_AFAMIN), NTU_AFAMAX);
    GC1 = ntu_gamln((AFAC + 1.0f));
    LAMC = NTU_POWF(((NTU_EXPF((ntu_gamln((AFAC + 4.0f)) - GC1)) * NC1D) / C3M1D), NTU_THRD);
    LAMCMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAC + 4.0f)) - GC1)), NTU_THRD), NTU_DCMAX));
    LAMCMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAC + 4.0f)) - GC1)), NTU_THRD), NTU_DCMIN));
    if ((LAMC < LAMCMIN)) {
        LAMC = LAMCMIN;
        NC1D = (C3M1D * NTU_EXPF(((ntu_gamln((AFAC + 1.0f)) - ntu_gamln((AFAC + 4.0f))) + (3.0f * NTU_LOGF(LAMC)))));
    } else if ((LAMC > LAMCMAX)) {
        LAMC = LAMCMAX;
        NC1D = (C3M1D * NTU_EXPF(((ntu_gamln((AFAC + 1.0f)) - ntu_gamln((AFAC + 4.0f))) + (3.0f * NTU_LOGF(LAMC)))));
    }
    MVDC = (NTU_POWF(NTU_EXPF((ntu_gamln((AFAC + 4.0f)) - ntu_gamln((AFAC + 1.0f)))), NTU_THRD) / LAMC);
}

// WRF SUBROUTINE SOLVE_AFAR (module_mp_ntu.F:820-859)
NTU_DEV void ntu_solve_afar(float TK1D, float QR1D, float& NR1D, float& LAMR, float& MVDR, float& AFAR) {
    float GR1 = 0.0f, LAMRMIN = 0.0f, LAMRMAX = 0.0f, R3M1D = 0.0f, BDR = 0.0f, MVRR = 0.0f, EFRR = 0.0f, KDX = 0.0f, LTK = 0.0f, LQR = 0.0f;
    if ((NR1D < NTU_NSMALL)) {
        LTK = NTU_LOGF(TK1D);
        LQR = (-(1.0f * NTU_LOGF(QR1D)));
        NR1D = NTU_EXPF((((((-5793.7852f) + (3191.1171f * LTK)) - (582.73279f * ntu_sq(LTK))) + (35.346854f * NTU_POWF(LTK, 3.0f))) - (0.25f * LQR)));
    }
    R3M1D = (__fdiv_rn((QR1D * NTU_V2M3), NTU_RHOW));
    MVRR = ntu_fmin(NTU_RRMAX, ntu_fmax(NTU_RRMIN, NTU_POWF((__fdiv_rn((QR1D / NR1D), NTU_C4PI3W)), NTU_THRD)));
    EFRR = NTU_EXPF(((NTU_EFR1 + (NTU_EFR2 * NTU_LOGF(NR1D))) + (NTU_EFR3 * NTU_LOGF(MVRR))));
    KDX = ntu_fmax(NTU_KCRMIN, ntu_fmin(NTU_KCRMAX, NTU_POWF((MVRR / EFRR), 3.0f)));
    AFAR = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
    AFAR = ntu_fmin(ntu_fmax(AFAR, NTU_AFAMIN), NTU_AFAMAX);
    GR1 = ntu_gamln((AFAR + 1.0f));
    LAMR = NTU_POWF(((NTU_EXPF((ntu_gamln((AFAR + 4.0f)) - GR1)) * NR1D) / R3M1D), NTU_THRD);
    LAMRMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAR + 4.0f)) - GR1)), NTU_THRD), NTU_DRMAX));
    LAMRMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAR + 4.0f)) - GR1)), NTU_THRD), NTU_DRMIN));
    if ((LAMR < LAMRMIN)) {
        LAMR = LAMRMIN;
        NR1D = (R3M1D * NTU_EXPF(((ntu_gamln((AFAR + 1.0f)) - ntu_gamln((AFAR + 4.0f))) + (3.0f * NTU_LOGF(LAMR)))));
    } else if ((LAMR > LAMRMAX)) {
        LAMR = LAMRMAX;
        NR1D = (R3M1D * NTU_EXPF(((ntu_gamln((AFAR + 1.0f)) - ntu_gamln((AFAR + 4.0f))) + (3.0f * NTU_LOGF(LAMR)))));
    }
    MVDR = (NTU_POWF(NTU_EXPF((ntu_gamln((AFAR + 4.0f)) - ntu_gamln((AFAR + 1.0f)))), NTU_THRD) / LAMR);
}

// WRF SUBROUTINE SOLVE_AFAI (module_mp_ntu.F:863-1100)
NTU_DEV_BIG void ntu_solve_afai(float TK1D, float P1D, float RHO, float QV1D, float QI1D, float& NI1D, float& VI1D, float& FI1D, float& I2M1D, float& I3M1D, float& ADAGR, float& ZETA, float& LAMI, float& AFAI, float& MVDI, float& RHOI, float& AMI, float& BMI, float& AVI, float& BVI, float& BEST) {
    float GI1 = 0.0f, TC1D = 0.0f, ESW = 0.0f, ESI = 0.0f, QVSI = 0.0f, INHGR = 0.0f, I3M0 = 0.0f, BEST0 = 0.0f, LAMIMIN = 0.0f, LAMIMAX = 0.0f, IPF = 0.0f, IPG = 0.0f, KDX = 0.0f, ZETA2 = 0.0f, ZETA4 = 0.0f, KINV = 0.0f, AAI = 0.0f, BAI = 0.0f, IBA1 = 0.0f, C1X2 = 0.0f, VTA1 = 0.0f, VTB1 = 0.0f, IPH2 = 0.0f, IPG2 = 0.0f, AVIA0 = 0.0f, BDI = 0.0f, FDI = 0.0f, LTK = 0.0f, LQI = 0.0f, MDI = 0.0f;
    int HID = 0;
    if ((VI1D < NTU_ISMALL)) {
        TC1D = (TK1D - NTU_TK0C);
        HID = ntu_imax(ntu_imin(ntu_nint((fabsf(TC1D) / 0.25f)), 120), 0);
        ESW = ntu_fmin((0.99f * P1D), ntu_polysvp(TK1D, 0));
        ESI = ntu_fmin((0.99f * P1D), ntu_polysvp(TK1D, 1));
        if ((ESI > ESW)) {
            ESI = ESW;
        }
        QVSI = ((0.622f * ESI) / (P1D - ESI));
        INHGR = NTU_ITBLE[HID];
        RHOI = (NTU_RHOI0 * NTU_EXPF((-((3.0f * ntu_fmax(((QV1D - QVSI) - 5.0E-5f), 0.0f)) / INHGR))));
        VI1D = (QI1D / RHOI);
    }
    RHOI = (QI1D / VI1D);
    if ((RHOI < NTU_RHOIMIN)) {
        RHOI = NTU_RHOIMIN;
    } else if ((RHOI > NTU_RHOIMAX)) {
        RHOI = NTU_RHOIMAX;
    }
    VI1D = (QI1D / RHOI);
    if ((NI1D < NTU_NSMALL)) {
        LTK = NTU_LOGF(TK1D);
        LQI = (-(1.0f * NTU_LOGF(QI1D)));
        MDI = (__fdiv_rn(NTU_EXPF((((-3.2653646f) + (2.0539073f * LTK)) - (0.25f * LQI))), 1.0E3f));
        NI1D = ((((1.0E9f * QI1D) * NTU_V2M3) / RHOI) / NTU_POWF(MDI, 3.0f));
    }
    if (((QI1D >= NTU_QSMAL1) && (NI1D < NTU_NSMAL1))) {
        LTK = NTU_LOGF(TK1D);
        LQI = (-(1.0f * NTU_LOGF(QI1D)));
        MDI = (__fdiv_rn(NTU_EXPF((((-3.2653646f) + (2.0539073f * LTK)) - (0.25f * LQI))), 1.0E3f));
        NI1D = ((((1.0E9f * QI1D) * NTU_V2M3) / RHOI) / NTU_POWF(MDI, 3.0f));
    }
    if (((I2M1D >= NTU_ASMALL) && (I3M1D >= NTU_ISMALL))) {
        KDX = ntu_fmax(0.0f, (NTU_POWF(I2M1D, 3.0f) / (ntu_sq(I3M1D) * NI1D)));
        if ((KDX > NTU_KCIMAX)) {
            KDX = NTU_KCIMIN;
            I2M1D = NTU_POWF(((KDX * NI1D) * ntu_sq(I3M1D)), NTU_THRD);
        } else if ((KDX < NTU_KCIMIN)) {
            KDX = NTU_KCIMIN;
            I2M1D = NTU_POWF(((KDX * NI1D) * ntu_sq(I3M1D)), NTU_THRD);
        }
        AFAI = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
        AFAI = ntu_fmin(ntu_fmax(AFAI, NTU_AFAMIN), NTU_AFAMAX);
        LAMI = (((AFAI + 3.0f) * I2M1D) / I3M1D);
        LAMIMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD), NTU_DIMAX));
        LAMIMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD), NTU_DIMIN));
        if ((LAMI < LAMIMIN)) {
            LAMI = LAMIMIN;
            NI1D = (I3M1D * NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - ntu_gamln((AFAI + 4.0f))) + (3.0f * NTU_LOGF(LAMI)))));
        } else if ((LAMI > LAMIMAX)) {
            LAMI = LAMIMAX;
            NI1D = (I3M1D * NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - ntu_gamln((AFAI + 4.0f))) + (3.0f * NTU_LOGF(LAMI)))));
        }
    } else if (((I2M1D < NTU_ASMALL) && (I3M1D >= NTU_ISMALL))) {
        AFAI = NTU_AFAI0;
        I2M1D = NTU_POWF(((NTU_KCIMIN * NI1D) * ntu_sq(I3M1D)), NTU_THRD);
        LAMI = NTU_POWF(((NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))) * NI1D) / I3M1D), NTU_THRD);
        LAMIMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD), NTU_DIMAX));
        LAMIMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD), NTU_DIMIN));
        if ((LAMI < LAMIMIN)) {
            LAMI = LAMIMIN;
            NI1D = (I3M1D * NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - ntu_gamln((AFAI + 4.0f))) + (3.0f * NTU_LOGF(LAMI)))));
        } else if ((LAMI > LAMIMAX)) {
            LAMI = LAMIMAX;
            NI1D = (I3M1D * NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - ntu_gamln((AFAI + 4.0f))) + (3.0f * NTU_LOGF(LAMI)))));
        }
    } else {
        BDI = (NTU_POWF((((QI1D * NTU_V2M3) / NI1D) / RHOI), NTU_THRD) * 1.0E3f);
        BDI = ntu_fmax(ntu_fmin((NTU_DIMAX * 1.0E3f), BDI), (NTU_DIMIN * 1.0E3f));
        FDI = (((((7.4015986E-2f + (7.9866676E-1f * BDI)) - (9.4468892E-3f * NTU_LOGF(NI1D))) + (3.8235092E-1f * ntu_sq(BDI))) + (2.9811542E-4f * ntu_sq(NTU_LOGF(NI1D)))) + ((1.9052614E-2f * BDI) * NTU_LOGF(NI1D)));
        KDX = ntu_fmax(NTU_KCIMIN, ntu_fmin(NTU_KCIMAX, NTU_POWF((BDI / FDI), 3.0f)));
        AFAI = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
        AFAI = ntu_fmin(ntu_fmax(AFAI, NTU_AFAMIN), NTU_AFAMAX);
        KDX = (((ntu_sq(AFAI) + (3.0f * AFAI)) + 2.0f) / ((ntu_sq(AFAI) + (6.0f * AFAI)) + 9.0f));
        I2M1D = NTU_POWF(((KDX * NI1D) * ntu_sq(((QI1D * NTU_V2M3) / RHOI))), NTU_THRD);
        GI1 = ntu_gamln((AFAI + 1.0f));
        LAMI = NTU_POWF((__fdiv_rn(((((NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - GI1)) * NTU_PI) * RHOI) * NI1D) / QI1D), 6.0f)), NTU_THRD);
        LAMIMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD), NTU_DIMAX));
        LAMIMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD), NTU_DIMIN));
        if ((LAMI < LAMIMIN)) {
            LAMI = LAMIMIN;
            NI1D = (((QI1D * NTU_V2M3) / RHOI) * NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - ntu_gamln((AFAI + 4.0f))) + (3.0f * NTU_LOGF(LAMI)))));
        } else if ((LAMI > LAMIMAX)) {
            LAMI = LAMIMAX;
            NI1D = (((QI1D * NTU_V2M3) / RHOI) * NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - ntu_gamln((AFAI + 4.0f))) + (3.0f * NTU_LOGF(LAMI)))));
        }
    }
    MVDI = (NTU_POWF(NTU_EXPF((ntu_gamln((AFAI + 4.0f)) - ntu_gamln((AFAI + 1.0f)))), NTU_THRD) / LAMI);
    if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
        I3M0 = (NI1D * NTU_K3);
        if ((MVDI > (NTU_DI0 + 1.0E-7f))) {
            ZETA = (NTU_LOGF((FI1D / I3M1D)) / NTU_LOGF((I3M1D / I3M0)));
            if ((ZETA > 0.4f)) {
                ZETA = 0.4f;
                FI1D = (NTU_POWF((I3M1D / I3M0), ZETA) * I3M1D);
            } else if ((ZETA < (-0.4f))) {
                ZETA = (-0.4f);
                FI1D = (NTU_POWF((I3M1D / I3M0), ZETA) * I3M1D);
            }
            ADAGR = ((1.0f + (2.0f * ZETA)) / (1.0f - ZETA));
        } else {
            FI1D = I3M1D;
            ZETA = 0.0f;
            ADAGR = 1.0f;
        }
    } else {
        I3M1D = ((QI1D * NTU_V2M3) / RHOI);
        TC1D = (TK1D - NTU_TK0C);
        HID = ntu_imax(ntu_imin(ntu_nint((fabsf(TC1D) / 0.25f)), 120), 0);
        ADAGR = NTU_POWF(ntu_fmax(ntu_fmin(NTU_ITBLE[HID], 2.0f), 0.5f), NTU_THRD);
        ZETA = ((ADAGR - 1.0f) / (ADAGR + 2.0f));
        I3M0 = (NI1D * NTU_K3);
        FI1D = (NTU_POWF((I3M1D / I3M0), ZETA) * I3M1D);
    }
    if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
        AMI = (__fdiv_rn(((NTU_PI * RHOI) * NTU_POWF(NTU_DI0, (2.0f - (2.0f / ADAGR)))), 6.0f));
        BMI = ((2.0f / ADAGR) + 1.0f);
    } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
        AMI = (__fdiv_rn(((NTU_PI * RHOI) * NTU_POWF(NTU_DI0, (1.0f - ADAGR))), 6.0f));
        BMI = (ADAGR + 2.0f);
    } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
        AMI = (__fdiv_rn((NTU_PI * RHOI), 6.0f));
        BMI = NTU_BMI0;
    }
    KINV = (((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((__fdiv_rn(TK1D, NTU_TK0C)), 1.5f)) / RHO);
    BEST0 = (((2.0f * NTU_G) * NI1D) / ntu_sq(KINV));
    if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
        AAI = ((NTU_PI / 4.0f) / NTU_POWF(NTU_DI0, ZETA));
        BAI = ((3.0f * (ADAGR + 1.0f)) / (ADAGR + 2.0f));
        IPH2 = (((2.0f * 3.0f) * ADAGR) / (ADAGR + 2.0f));
        ZETA4 = ((4.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
        IBA1 = ((BMI + IPH2) - BAI);
        BEST = (((BEST0 * AMI) * NTU_EXPF(((ntu_gamln(((IBA1 + AFAI) + 1.0f)) - ntu_gamln((AFAI + 1.0f))) - (IBA1 * NTU_LOGF(LAMI))))) / (AAI * NTU_POWF(NTU_DI0, ZETA4)));
        C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
        VTB1 = ((((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f) - ((((NTU_VTA0 * NTU_VTB0) * NTU_POWF(BEST, NTU_VTB0)) / NTU_VTC2) / ntu_sq((sqrtf((1.0f + C1X2)) - 1.0f))));
        VTA1 = ntu_fmax((((NTU_VTC2 * ntu_sq((NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f))) - (NTU_VTA0 * NTU_POWF(BEST, NTU_VTB0))) / NTU_POWF(BEST, VTB1)), 0.0f);
        AVIA0 = (VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1))));
        AVI = (AVIA0 * NTU_POWF((((2.0f * AMI) * NTU_G) / (AAI * NTU_POWF(NTU_DI0, ZETA4))), VTB1));
        BVI = ((VTB1 * ((BMI + IPH2) - BAI)) - 1.0f);
    } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
        ZETA2 = ((2.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
        AAI = ((NTU_PI * NTU_POWF(NTU_DI0, ZETA2)) / 4.0f);
        BAI = ((2.0f * 3.0f) / (ADAGR + 2.0f));
        IPG2 = ((2.0f * 3.0f) / (ADAGR + 2.0f));
        IBA1 = ((BMI + IPG2) - BAI);
        BEST = ((((BEST0 * AMI) * NTU_POWF(NTU_DI0, ZETA2)) * NTU_EXPF(((ntu_gamln(((IBA1 + AFAI) + 1.0f)) - ntu_gamln((AFAI + 1.0f))) - (IBA1 * NTU_LOGF(LAMI))))) / AAI);
        C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
        VTB1 = ((((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f) - ((((NTU_VTA0 * NTU_VTB0) * NTU_POWF(BEST, NTU_VTB0)) / NTU_VTC2) / ntu_sq((sqrtf((1.0f + C1X2)) - 1.0f))));
        VTA1 = ntu_fmax((((NTU_VTC2 * ntu_sq((NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f))) - (NTU_VTA0 * NTU_POWF(BEST, NTU_VTB0))) / NTU_POWF(BEST, VTB1)), 0.0f);
        AVIA0 = (VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1))));
        AVI = (AVIA0 * NTU_POWF(((((2.0f * AMI) * NTU_G) * NTU_POWF(NTU_DI0, ZETA2)) / AAI), VTB1));
        BVI = ((VTB1 * ((BMI + IPG2) - BAI)) - 1.0f);
    } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
        AAI = (NTU_PI / 4.0f);
        BAI = 2.0f;
        IBA1 = BMI;
        BEST = (__fdiv_rn(((BEST0 * AMI) * NTU_EXPF(((ntu_gamln(((IBA1 + AFAI) + 1.0f)) - ntu_gamln((AFAI + 1.0f))) - (IBA1 * NTU_LOGF(LAMI))))), AAI));
        C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
        VTB1 = ((((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f) - ((((NTU_VTA0 * NTU_VTB0) * NTU_POWF(BEST, NTU_VTB0)) / NTU_VTC2) / ntu_sq((sqrtf((1.0f + C1X2)) - 1.0f))));
        VTA1 = ntu_fmax((((NTU_VTC2 * ntu_sq((NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f))) - (NTU_VTA0 * NTU_POWF(BEST, NTU_VTB0))) / NTU_POWF(BEST, VTB1)), 0.0f);
        AVIA0 = (VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1))));
        AVI = (AVIA0 * NTU_POWF((__fdiv_rn(((2.0f * AMI) * NTU_G), AAI)), VTB1));
        BVI = ((VTB1 * BMI) - 1.0f);
    }
}

// WRF SUBROUTINE SOLVE_AFAS (module_mp_ntu.F:1104-1262)
NTU_DEV_BIG void ntu_solve_afas(float TK1D, float RHO, float QS1D, float QC1D, float& NS1D, float& VS1D, float& FS1D, float& S2M1D, float& AFAS, float& LAMS, float& MVDS, float& RHOS, float& SASPR, float& AMS, float& AVS, float& BVS) {
    float LAMSMIN = 0.0f, LAMSMAX = 0.0f, KINV = 0.0f, C1X2 = 0.0f, VTA1 = 0.0f, VTB1 = 0.0f, BEST = 0.0f, S3M1D = 0.0f, KDX = 0.0f, TC1D = 0.0f, SBA1 = 0.0f, AMS2 = 0.0f, AAS2 = 0.0f, AMS3 = 0.0f, BMS3 = 0.0f, AAS3 = 0.0f, BAS3 = 0.0f, LAMD = 0.0f, BDS = 0.0f, FDS = 0.0f, LTK = 0.0f, LQS = 0.0f, MDS = 0.0f;
    int TBIN = 0, DBIN = 0;
    RHOS = (QS1D / (VS1D + NTU_ISMALL));
    if ((RHOS > NTU_RHOIMAX)) {
        RHOS = NTU_RHOIMAX;
    } else if ((RHOS < NTU_RHOIMIN)) {
        RHOS = NTU_RHOIMIN;
    }
    VS1D = (QS1D / RHOS);
    if ((NS1D < NTU_NSMALL)) {
        LTK = NTU_LOGF(TK1D);
        LQS = (-(1.0f * NTU_LOGF(QS1D)));
        MDS = (__fdiv_rn(NTU_EXPF((((((-123.23898f) + (40.74706f * LTK)) - (3.0333477f * ntu_sq(LTK))) - (0.31219981f * LQS)) + (0.0012798222f * ntu_sq(LQS)))), 1.0E3f));
        NS1D = ((((1.0E9f * QS1D) * NTU_V2M3) / RHOS) / NTU_POWF(MDS, 3.0f));
    }
    AMS = (__fdiv_rn((NTU_PI * RHOS), 6.0f));
    S3M1D = (QS1D / AMS);
    SASPR = (FS1D / S3M1D);
    if ((SASPR < NTU_SASMIN)) {
        SASPR = NTU_SASMIN;
        FS1D = (SASPR * S3M1D);
    } else if ((SASPR > NTU_SASMAX)) {
        SASPR = NTU_SASMAX;
        FS1D = (SASPR * S3M1D);
    }
    if (((S2M1D >= NTU_ASMALL) && (S3M1D >= NTU_ISMALL))) {
        KDX = ntu_fmax(0.0f, (NTU_POWF(S2M1D, 3.0f) / (ntu_sq(S3M1D) * NS1D)));
        if ((KDX > NTU_KCSMAX)) {
            KDX = NTU_KCSMIN;
            S2M1D = NTU_POWF(((KDX * NS1D) * ntu_sq(S3M1D)), NTU_THRD);
        } else if ((KDX < NTU_KCSMIN)) {
            KDX = NTU_KCSMIN;
            S2M1D = NTU_POWF(((KDX * NS1D) * ntu_sq(S3M1D)), NTU_THRD);
        }
        AFAS = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
        AFAS = ntu_fmin(ntu_fmax(AFAS, NTU_AFAMIN), NTU_AFAMAX);
        LAMS = (((AFAS + 3.0f) * S2M1D) / S3M1D);
    } else {
        AFAS = NTU_AFAS0;
        S2M1D = NTU_POWF(((NTU_KCSMIN * NS1D) * ntu_sq(S3M1D)), NTU_THRD);
        LAMS = NTU_POWF(((NTU_EXPF((ntu_gamln((AFAS + 4.0f)) - ntu_gamln((AFAS + 1.0f)))) * NS1D) / S3M1D), NTU_THRD);
    }
    LAMSMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAS + 4.0f)) - ntu_gamln((AFAS + 1.0f)))), NTU_THRD), NTU_DSMAX));
    LAMSMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAS + 4.0f)) - ntu_gamln((AFAS + 1.0f)))), NTU_THRD), NTU_DSMIN));
    if ((LAMS < LAMSMIN)) {
        LAMS = LAMSMIN;
        NS1D = (S3M1D * NTU_EXPF(((ntu_gamln((AFAS + 1.0f)) - ntu_gamln((AFAS + 4.0f))) + (3.0f * NTU_LOGF(LAMS)))));
    } else if ((LAMS > LAMSMAX)) {
        LAMS = LAMSMAX;
        NS1D = (S3M1D * NTU_EXPF(((ntu_gamln((AFAS + 1.0f)) - ntu_gamln((AFAS + 4.0f))) + (3.0f * NTU_LOGF(LAMS)))));
    }
    MVDS = (NTU_POWF(NTU_EXPF((ntu_gamln((AFAS + 4.0f)) - ntu_gamln((AFAS + 1.0f)))), NTU_THRD) / LAMS);
    KINV = (((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((__fdiv_rn(TK1D, NTU_TK0C)), 1.5f)) / RHO);
    TC1D = (TK1D - NTU_TK0C);
    TBIN = ntu_imin(ntu_imax(((int)((__fdiv_rn(fabsf(TC1D), 5.0f))) - 4), 0), 8);
    if ((MVDS < 1.5E-4f)) {
        DBIN = 0;
    }
    if (((MVDS >= 1.5E-4f) && (MVDS < 3.0E-4f))) {
        DBIN = 1;
    }
    if (((MVDS >= 3.0E-4f) && (MVDS < 1.0E-3f))) {
        DBIN = 2;
    }
    if (((MVDS >= 1.0E-3f) && (MVDS < 3.0E-3f))) {
        DBIN = 3;
    }
    if ((MVDS >= 3.0E-3f)) {
        DBIN = 4;
    }
    DBIN = ntu_imin(ntu_imax(DBIN, 0), 4);
    AMS3 = NTU_AMS1[((TBIN * 5) + DBIN)];
    BMS3 = NTU_BMS1[((TBIN * 5) + DBIN)];
    AAS3 = NTU_AAS1[((TBIN * 5) + DBIN)];
    BAS3 = NTU_BAS1[((TBIN * 5) + DBIN)];
    LAMD = ntu_fmin(NTU_POWF((((AMS3 * NTU_POWF(LAMS, 3.0f)) / AMS) * NTU_EXPF((ntu_gamln(((BMS3 + AFAS) + 1.0f)) - ntu_gamln((AFAS + 4.0f))))), (1.0f / BMS3)), LAMS);
    SBA1 = ((BMS3 + 2.0f) - BAS3);
    BEST = (((((2.0f * NTU_G) * NS1D) * AMS3) * NTU_EXPF(((ntu_gamln(((SBA1 + AFAS) + 1.0f)) - ntu_gamln((AFAS + 1.0f))) - (SBA1 * NTU_LOGF(LAMD))))) / (ntu_sq(KINV) * AAS3));
    C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
    VTB1 = (((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f);
    VTA1 = ((NTU_VTC2 * ntu_sq((NTU_POWF(((float)(1) + C1X2), 5.0E-1f) - 1.0f))) / NTU_POWF(BEST, VTB1));
    AVS = ((VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1)))) * NTU_POWF((((2.0f * AMS3) * NTU_G) / AAS3), VTB1));
    BVS = ((VTB1 * ((BMS3 + 2.0f) - BAS3)) - 1.0f);
}

// WRF SUBROUTINE SOLVE_AFAG (module_mp_ntu.F:1266-1372)
NTU_DEV_BIG void ntu_solve_afag(float TK1D, float RHO, float QG1D, float QC1D, float& NG1D, float& VG1D, float& G2M1D, float& LAMG, float& AFAG, float& MVDG, float& RHOG, float& AMG, float& AVG, float& BVG) {
    float G3M1D = 0.0f, BEST0 = 0.0f, LAMGMIN = 0.0f, LAMGMAX = 0.0f, KDX = 0.0f, KINV = 0.0f, GG1 = 0.0f, C1X2 = 0.0f, VTA1 = 0.0f, VTB1 = 0.0f, BEST = 0.0f, GMLR = 0.0f, BDG = 0.0f, FDG = 0.0f, LTK = 0.0f, LQG = 0.0f, MDG = 0.0f;
    RHOG = (QG1D / (VG1D + NTU_ISMALL));
    if ((TK1D >= NTU_TK0C)) {
        if ((RHOG > NTU_RHOH)) {
            RHOG = NTU_RHOH;
        }
    } else {
        if ((RHOG > NTU_RHOG0)) {
            RHOG = NTU_RHOG0;
        }
    }
    if ((RHOG < NTU_RHOIMIN)) {
        RHOG = NTU_RHOIMIN;
    }
    VG1D = (QG1D / RHOG);
    if ((NG1D < NTU_NSMALL)) {
        LTK = NTU_LOGF(TK1D);
        LQG = (-(1.0f * NTU_LOGF(QG1D)));
        MDG = (__fdiv_rn(NTU_EXPF((((((-2205.8027f) + (1225.8046f * LTK)) - (226.27995f * ntu_sq(LTK))) + (13.929644f * NTU_POWF(LTK, 3.0f))) - (0.25f * LQG))), 1.0E3f));
        NG1D = ((((1.0E9f * QG1D) * NTU_V2M3) / RHOG) / NTU_POWF(MDG, 3.0f));
    }
    AMG = (__fdiv_rn((NTU_PI * RHOG), 6.0f));
    G3M1D = ((QG1D * NTU_V2M3) / RHOG);
    if (((G2M1D >= NTU_ASMALL) && (G3M1D >= NTU_ISMALL))) {
        KDX = ntu_fmax(0.0f, (NTU_POWF(G2M1D, 3.0f) / (ntu_sq(G3M1D) * NG1D)));
        if ((KDX > NTU_KCGMAX)) {
            KDX = NTU_KCGMIN;
            G2M1D = NTU_POWF(((KDX * NG1D) * ntu_sq(G3M1D)), NTU_THRD);
        } else if ((KDX < NTU_KCGMIN)) {
            KDX = NTU_KCGMIN;
            G2M1D = NTU_POWF(((KDX * NG1D) * ntu_sq(G3M1D)), NTU_THRD);
        }
        AFAG = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
        AFAG = ntu_fmin(ntu_fmax(AFAG, NTU_AFAMIN), NTU_AFAMAX);
        LAMG = (((AFAG + 3.0f) * G2M1D) / G3M1D);
    } else {
        AFAG = NTU_AFAG0;
        G2M1D = NTU_POWF(((NTU_KCGMIN * NG1D) * ntu_sq(G3M1D)), NTU_THRD);
        LAMG = NTU_POWF(((NTU_EXPF((ntu_gamln((AFAG + 4.0f)) - ntu_gamln((AFAG + 1.0f)))) * NG1D) / G3M1D), NTU_THRD);
    }
    LAMGMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAG + 4.0f)) - ntu_gamln((AFAG + 1.0f)))), NTU_THRD), NTU_DGMAX));
    LAMGMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAG + 4.0f)) - ntu_gamln((AFAG + 1.0f)))), NTU_THRD), NTU_DGMIN));
    if ((LAMG < LAMGMIN)) {
        LAMG = LAMGMIN;
        NG1D = (G3M1D * NTU_EXPF(((ntu_gamln((AFAG + 1.0f)) - ntu_gamln((AFAG + 4.0f))) + (3.0f * NTU_LOGF(LAMG)))));
    } else if ((LAMG > LAMGMAX)) {
        LAMG = LAMGMAX;
        NG1D = (G3M1D * NTU_EXPF(((ntu_gamln((AFAG + 1.0f)) - ntu_gamln((AFAG + 4.0f))) + (3.0f * NTU_LOGF(LAMG)))));
    }
    MVDG = (NTU_POWF(NTU_EXPF((ntu_gamln((AFAG + 4.0f)) - ntu_gamln((AFAG + 1.0f)))), NTU_THRD) / LAMG);
    KINV = (((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((__fdiv_rn(TK1D, NTU_TK0C)), 1.5f)) / RHO);
    BEST0 = (((2.0f * NTU_G) * NG1D) / ntu_sq(KINV));
    GG1 = ntu_gamln((AFAG + 1.0f));
    BEST = (__fdiv_rn(((BEST0 * AMG) * NTU_EXPF(((ntu_gamln(((NTU_BMG + AFAG) + 1.0f)) - GG1) - (NTU_BMG * NTU_LOGF(LAMG))))), NTU_AAW));
    C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
    VTB1 = (((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f);
    VTA1 = ((NTU_VTC2 * ntu_sq((NTU_POWF(((float)(1) + C1X2), 5.0E-1f) - 1.0f))) / NTU_POWF(BEST, VTB1));
    AVG = ((VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1)))) * NTU_POWF((__fdiv_rn(((2.0f * AMG) * NTU_G), NTU_AAW)), VTB1));
    BVG = ((VTB1 * NTU_BMG) - 1.0f);
}

// WRF SUBROUTINE SOLVE_AFAH (module_mp_ntu.F:1376-1455)
NTU_DEV_BIG void ntu_solve_afah(float TK1D, float RHO, float QH1D, float& NH1D, float& H2M1D, float& LAMH, float& AFAH, float& MVDH, float& AVH, float& BVH) {
    float H3M1D = 0.0f, BEST0 = 0.0f, GH1 = 0.0f, KDX = 0.0f, LAMHMIN = 0.0f, LAMHMAX = 0.0f, KINV = 0.0f, C1X2 = 0.0f, VTA1 = 0.0f, VTB1 = 0.0f, BEST = 0.0f, BDH = 0.0f, FDH = 0.0f, LTK = 0.0f, LQH = 0.0f;
    if ((NH1D < NTU_NSMALL)) {
        LTK = NTU_LOGF(TK1D);
        LQH = (-(1.0f * NTU_LOGF(QH1D)));
        NH1D = NTU_EXPF(((((22.929406f - (4.2328364f * LTK)) + (0.30647567f * ntu_sq(LTK))) - (0.009233271f * NTU_POWF(LTK, 3.0f))) - (0.25f * LQH)));
    }
    H3M1D = (__fdiv_rn((QH1D * NTU_V2M3), NTU_RHOH));
    if (((H2M1D >= NTU_ASMALL) && (H3M1D >= NTU_ISMALL))) {
        KDX = ntu_fmax(0.0f, (NTU_POWF(H2M1D, 3.0f) / (ntu_sq(H3M1D) * NH1D)));
        if ((KDX > NTU_KCHMAX)) {
            KDX = NTU_KCHMIN;
            H2M1D = NTU_POWF(((KDX * NH1D) * ntu_sq(H3M1D)), NTU_THRD);
        } else if ((KDX < NTU_KCHMIN)) {
            KDX = NTU_KCHMIN;
            H2M1D = NTU_POWF(((KDX * NH1D) * ntu_sq(H3M1D)), NTU_THRD);
        }
        AFAH = ((((6.0f * KDX) - 3.0f) + sqrtf(((8.0f * KDX) + 1.0f))) / (2.0f - (2.0f * KDX)));
        AFAH = ntu_fmin(ntu_fmax(AFAH, NTU_AFAMIN), NTU_AFAMAX);
        LAMH = (((AFAH + 3.0f) * H2M1D) / H3M1D);
    } else {
        AFAH = NTU_AFAH0;
        H2M1D = NTU_POWF(((NTU_KCHMIN * NH1D) * ntu_sq(H3M1D)), NTU_THRD);
        LAMH = NTU_POWF(((NTU_EXPF((ntu_gamln((AFAH + 4.0f)) - ntu_gamln((AFAH + 1.0f)))) * NH1D) / H3M1D), NTU_THRD);
    }
    LAMHMIN = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAH + 4.0f)) - ntu_gamln((AFAH + 1.0f)))), NTU_THRD), NTU_DHMAX));
    LAMHMAX = (__fdiv_rn(NTU_POWF(NTU_EXPF((ntu_gamln((AFAH + 4.0f)) - ntu_gamln((AFAH + 1.0f)))), NTU_THRD), NTU_DHMIN));
    if ((LAMH < LAMHMIN)) {
        LAMH = LAMHMIN;
        NH1D = (H3M1D * NTU_EXPF(((ntu_gamln((AFAH + 1.0f)) - ntu_gamln((AFAH + 4.0f))) + (3.0f * NTU_LOGF(LAMH)))));
    } else if ((LAMH > LAMHMAX)) {
        LAMH = LAMHMAX;
        NH1D = (H3M1D * NTU_EXPF(((ntu_gamln((AFAH + 1.0f)) - ntu_gamln((AFAH + 4.0f))) + (3.0f * NTU_LOGF(LAMH)))));
    }
    MVDH = (NTU_POWF(NTU_EXPF((ntu_gamln((AFAH + 4.0f)) - ntu_gamln((AFAH + 1.0f)))), NTU_THRD) / LAMH);
    KINV = (((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((__fdiv_rn(TK1D, NTU_TK0C)), 1.5f)) / RHO);
    BEST0 = (((2.0f * NTU_G) * NH1D) / ntu_sq(KINV));
    GH1 = ntu_gamln((AFAH + 1.0f));
    BEST = (__fdiv_rn(((BEST0 * NTU_AMH) * NTU_EXPF(((ntu_gamln(((NTU_BMH + AFAH) + 1.0f)) - GH1) - (NTU_BMH * NTU_LOGF(LAMH))))), NTU_AAW));
    C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
    VTB1 = (((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f);
    VTA1 = ((NTU_VTC2 * ntu_sq((NTU_POWF(((float)(1) + C1X2), 5.0E-1f) - 1.0f))) / NTU_POWF(BEST, VTB1));
    AVH = ((VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1)))) * NTU_POWF((((2.0f * NTU_AMH) * NTU_G) / NTU_AAW), VTB1));
    BVH = ((VTB1 * NTU_BMH) - 1.0f);
}

// WRF SUBROUTINE AERO_CONST (module_mp_ntu.F:1459-1563)
NTU_DEV_BIG void ntu_aero_const(NtuAeroConst* __restrict__ AC, int CCNTY) {
    const int ID_SENS = 1;
    const int N1 = NTU_NCCN;
    const int N2 = 5;
    int IDAER[2] = {};
    float XMASS[2] = {};
    const float ASHX[5] = {8.0E2f, 3.57E3f, 3.57E3f, 2.0E3f, 1.0E4f};
    const float CMASX[5] = {0.13214f, 0.13214f, 0.13214f, 0.13214f, 0.13214f};
    const float DNASX[5] = {1769.0f, 1769.0f, 1769.0f, 1769.0f, 1769.0f};
    const float AVANX[5] = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const float ZCCNX[15] = {3.4E8f, 6.0E7f, 3.0E6f, 1.0E9f, 8.0E8f, 7.2E5f, 6.4E9f, 2.3E9f, 3.2E6f, 1.06E11f, 3.2E10f, 5.4E6f, 4.02E-11f, 4.5E9f, 5.4E6f};
    const float CNMODX[15] = {5.0E-9f, 3.55E-8f, 3.1E-7f, 8.0E-9f, 3.3E-8f, 4.6E-7f, 7.5E-9f, 3.8E-8f, 5.1E-7f, 7.0E-9f, 2.7E-8f, 4.3E-7f, 4.5E-9f, 2.35E-8f, 5.5E-7f};
    const float CNSTDX[15] = {0.47f, 0.69f, 0.99f, 0.47f, 0.74f, 0.79f, 0.53f, 0.69f, 0.77f, 0.59f, 0.77f, 0.79f, 0.53f, 0.75f, 0.74f};
    const float SENX[15] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float XLOG3 = 0.0f, XMTOT = 0.0f, XMAS = 0.0f, DUST_IN0 = 0.0f, DUST_IN = 0.0f, DAIR = 0.0f, DZ0 = 0.0f, ZH = 0.0f;
    double DMODE = 0.0, D2STDV = 0.0, DLNXX = 0.0;
    int IA = 0, I = 0, K = 0, J = 0, IM = 0, IV = 0, IB = 0;
    if ((CCNTY == 1)) {
        AC->DNC0 = 46635.361f;
        AC->DNC1 = (-25567.933f);
        AC->DNC2 = 4674.3534f;
        AC->DNC3 = (-284.84377f);
    } else if ((CCNTY == 2)) {
        AC->DNC0 = (-48575.888f);
        AC->DNC1 = 26634.802f;
        AC->DNC2 = (-4862.961f);
        AC->DNC3 = 295.7886f;
    } else if ((CCNTY == 3)) {
        AC->DNC0 = (-63646.587f);
        AC->DNC1 = 34778.66f;
        AC->DNC2 = (-6326.6327f);
        AC->DNC3 = 383.28849f;
    } else if ((CCNTY == 4)) {
        AC->DNC0 = (-6069.3697f);
        AC->DNC1 = 3667.889f;
        AC->DNC2 = (-725.3412f);
        AC->DNC3 = 47.260735f;
    }
    AC->IBAER[(1) - (1)] = 1;
    { const int ntu_hi_i = NTU_NAER;
    for (I = 2; I <= ntu_hi_i; ++I) {
        AC->IBAER[(I) - (1)] = (AC->IBAER[((I - 1)) - (1)] + NTU_NAERN[((I - 1)) - (1)]);
    } }
    AC->TBLRC[(1) - (1)] = NTU_K4;
    AC->TBLRC[(NTU_NTBXA) - (1)] = NTU_K5;
    { const int ntu_hi_i = (NTU_NTBXA - 1);
    for (I = 2; I <= ntu_hi_i; ++I) {
        AC->TBLRC[(I) - (1)] = (AC->TBLRC[(1) - (1)] + (((double)((I - 1)) / (double)((NTU_NTBXA - 1))) * (AC->TBLRC[(NTU_NTBXA) - (1)] - AC->TBLRC[(1) - (1)])));
    } }
    IDAER[0] = CCNTY; IDAER[1] = 5;
    /* PRINT *, 'AEROSOL ID =',IDAER */
    { const int ntu_hi_j = NTU_NAER;
    for (J = 1; J <= ntu_hi_j; ++J) {
        AC->ASH[(J) - (1)] = ASHX[(IDAER[(J) - (1)]) - (1)];
        AC->DNAS[(J) - (1)] = DNASX[(IDAER[(J) - (1)]) - (1)];
        AC->CMAS[(J) - (1)] = CMASX[(IDAER[(J) - (1)]) - (1)];
        AC->AVAN[(J) - (1)] = AVANX[(IDAER[(J) - (1)]) - (1)];
        AC->BETA1[(J) - (1)] = (((NTU_CMW * AC->DNAS[(J) - (1)]) * AC->AVAN[(J) - (1)]) / (AC->CMAS[(J) - (1)] * NTU_RHOW));
        XMTOT = 0.0f;
        { const int ntu_hi_im = N1;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            AC->ZCCN[((IM) - (1)) + 3 * ((J) - (1))] = (ZCCNX[((IM) - (1)) + 3 * ((IDAER[(J) - (1)]) - (1))] * 1.0f);
            AC->CNMOD[((IM) - (1)) + 3 * ((J) - (1))] = CNMODX[((IM) - (1)) + 3 * ((IDAER[(J) - (1)]) - (1))];
            AC->CNSTD[((IM) - (1)) + 3 * ((J) - (1))] = CNSTDX[((IM) - (1)) + 3 * ((IDAER[(J) - (1)]) - (1))];
            XLOG3 = ((AC->CNMOD[((IM) - (1)) + 3 * ((J) - (1))] * AC->CNMOD[((IM) - (1)) + 3 * ((J) - (1))]) * AC->CNMOD[((IM) - (1)) + 3 * ((J) - (1))]);
            AC->WMAS[((IM) - (1)) + 3 * ((J) - (1))] = ((AC->ZCCN[((IM) - (1)) + 3 * ((J) - (1))] * XLOG3) * NTU_EXPF((4.5f * ntu_sq(AC->CNSTD[((IM) - (1)) + 3 * ((J) - (1))]))));
            AC->RFACT[((IM) - (1)) + 3 * ((J) - (1))] = (1.0f / (((NTU_C4PI3 * AC->DNAS[(J) - (1)]) * XLOG3) * NTU_EXPF((4.5f * ntu_sq(AC->CNSTD[((IM) - (1)) + 3 * ((J) - (1))])))));
            XMTOT = (XMTOT + AC->WMAS[((IM) - (1)) + 3 * ((J) - (1))]);
        } }
        { const int ntu_hi_im = N1;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            AC->WMAS[((IM) - (1)) + 3 * ((J) - (1))] = (AC->WMAS[((IM) - (1)) + 3 * ((J) - (1))] / XMTOT);
        } }
        { const int ntu_hi_i = NTU_NTBXA;
        for (I = 1; I <= ntu_hi_i; ++I) {
            AC->TBLXF[((I) - (1)) + 25 * ((J) - (1))] = (double)(0.0f);
        } }
        { const int ntu_hi_im = N1;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            DMODE = (NTU_LOG((double)(AC->CNMOD[((IM) - (1)) + 3 * ((J) - (1))])) + (((double)(3.0f) * (double)(AC->CNSTD[((IM) - (1)) + 3 * ((J) - (1))])) * (double)(AC->CNSTD[((IM) - (1)) + 3 * ((J) - (1))])));
            D2STDV = (NTU_K6 * (double)(AC->CNSTD[((IM) - (1)) + 3 * ((J) - (1))]));
            { const int ntu_hi_i = NTU_NTBXA;
            for (I = 1; I <= ntu_hi_i; ++I) {
                DLNXX = ((AC->TBLRC[(I) - (1)] - DMODE) / D2STDV);
                AC->TBLXF[((I) - (1)) + 25 * ((J) - (1))] = (AC->TBLXF[((I) - (1)) + 25 * ((J) - (1))] + (((double)(AC->WMAS[((IM) - (1)) + 3 * ((J) - (1))]) * (double)(0.5f)) * (1.E+0 + NTU_ERF(DLNXX))));
            } }
        } }
        { const int ntu_hi_i = NTU_NTBXA;
        for (I = 1; I <= ntu_hi_i; ++I) {
            AC->TBLXF[((I) - (1)) + 25 * ((J) - (1))] = ntu_dmax(1.E-20, AC->TBLXF[((I) - (1)) + 25 * ((J) - (1))]);
        } }
        ntu_find_rc0(1.E-6, &AC->CNMOD[((1) - (1)) + 3 * ((J) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((J) - (1))], &AC->WMAS[((1) - (1)) + 3 * ((J) - (1))], AC->RXMIN[(J) - (1)], &AC->TBLXF[((1) - (1)) + 25 * ((J) - (1))], AC->TBLRC);
        /* WRITE(*,*)'FOR DRY AEROSOL FRACTION BE > 1.E-6' */
        /* WRITE(*,'(A30,I1,A8,F6.4,A3)')'MINIMUN R-CUTOFF FOR AEROSOL' */
        /* WRITE(*,*)'SENSITIVITY TEST ID:',ID_SENS */
        AC->SENS[(J) - (1)] = SENX[((IDAER[(J) - (1)]) - (1)) + 5 * ((ID_SENS) - (1))];
    } }
}

// WRF SUBROUTINE CCNWHITBY (module_mp_ntu.F:2846-2875)
NTU_DEV void ntu_ccnwhitby(float W1D, float RSX, float RS9, float& ZCN9, float& RACT, float BETA1, float ALPHA, float* ZCCN, float* CNMOD, float* CNSTD) {
    float RWS0 = 0.0f, TMPX = 0.0f, TMP1 = 0.0f, TMP2 = 0.0f, TMP3 = 0.0f, TMP4 = 0.0f, TMP5 = 0.0f, TMP6 = 0.0f;
    int IM = 0;
    RWS0 = sqrtf(((BETA1 * NTU_POWF(RS9, 3.0f)) / ALPHA));
    TMPX = NTU_LOG10F((RSX * 1.0E6f));
    TMP1 = ((-0.61425115f) + (((-0.66624878f) - (0.17367658f * TMPX)) * TMPX));
    RACT = ((NTU_POWF(10.0f, TMP1) * RWS0) * NTU_POWF(W1D, (-0.11782f)));
    ZCN9 = 0.0f;
    { const int ntu_hi_im = NTU_NCCN;
    for (IM = 1; IM <= ntu_hi_im; ++IM) {
        TMP2 = (ZCCN[(IM) - (1)] / 2.0f);
        TMP3 = (NTU_SQRT2 * CNSTD[(IM) - (1)]);
        TMP4 = (NTU_LOGF((RSX / CNMOD[(IM) - (1)])) / TMP3);
        TMP5 = (NTU_LOGF((RS9 / CNMOD[(IM) - (1)])) / TMP3);
        TMP6 = (TMP2 * (NTU_ERFF(TMP4) - NTU_ERFF(TMP5)));
        ZCN9 = (ZCN9 + TMP6);
    } }
}

// WRF FUNCTION DLNX (module_mp_ntu.F:2793-2804)
NTU_DEV double ntu_dlnx(float RX, float XMODE, float SIGMA, int N) {
    double DRX = 0.0, DXMODE = 0.0, DSTDV = 0.0;
    double NTU_RET = 0;
    DRX = (double)(RX);
    DXMODE = (double)(XMODE);
    DSTDV = (double)(SIGMA);
    NTU_RET = ((NTU_LOG((DRX / DXMODE)) - (ntu_sqd(DSTDV) * (double)(N))) / (NTU_K0 * DSTDV));
    return NTU_RET;
}

// WRF SUBROUTINE RSWHITBY (module_mp_ntu.F:2819-2844)
NTU_DEV void ntu_rswhitby(float W1D, float RSX, float& Z, float& RACT, float BETA1, float ALPHA, float* ZCCN, float* CNMOD, float* CNSTD) {
    float RWS0 = 0.0f, TEMP = 0.0f, TEMP1 = 0.0f, TEMP2 = 0.0f, TMPX = 0.0f, TEMP4 = 0.0f;
    int IM = 0;
    Z = 0.0f;
    { const int ntu_hi_im = NTU_NCCN;
    for (IM = 1; IM <= ntu_hi_im; ++IM) {
        TEMP1 = (ZCCN[(IM) - (1)] / 2.0f);
        TEMP2 = (NTU_SQRT2 * CNSTD[(IM) - (1)]);
        TEMP = (NTU_LOGF((RSX / CNMOD[(IM) - (1)])) / TEMP2);
        Z = (Z + (TEMP1 * (1.0f - NTU_ERFF(TEMP))));
    } }
    RWS0 = sqrtf(((BETA1 * NTU_POWF(RSX, 3.0f)) / ALPHA));
    TMPX = NTU_LOG10F((RSX * 1.0E6f));
    TEMP4 = ((-0.61425115f) + (((-0.66624878f) - (0.17367658f * TMPX)) * TMPX));
    RACT = ((NTU_POWF(10.0f, TEMP4) * RWS0) * NTU_POWF(W1D, (-0.11782f)));
}

// WRF SUBROUTINE ACTIVA (module_mp_ntu.F:2690-2789)
NTU_DEV_BIG void ntu_activa(NtuAeroConst* __restrict__ AC, float TK1D, float W1D, float& NC1D, float& NR1D, float* ABCD, float& QACAC, float& QACAR, float& RX0, float* ZCCNS, float SSW, int IAE, bool& DO_FR) {
    const float RS10 = 10.0E-6f;
    const float RS15 = 15.0E-6f;
    const float RS20 = 20.0E-6f;
    const float RS25 = 25.0E-6f;
    float NACAC = 0.0f, NACAR = 0.0f, GAMAA = 0.0f, ALPHA = 0.0f, SSRW = 0.0f, ES0 = 0.0f, RX9 = 0.0f, RX1 = 0.0f, XMASS = 0.0f, RACT = 0.0f, ZCN9 = 0.0f, ZCN99 = 0.0f, ZCCN10 = 0.0f, ZCCN15 = 0.0f, ZCCN20 = 0.0f, ZCCN25 = 0.0f, RACT10 = 0.0f, RACT15 = 0.0f, RACT20 = 0.0f, RACT25 = 0.0f, WR10 = 0.0f, WR15 = 0.0f, WR20 = 0.0f, WR25 = 0.0f, WR10TO15 = 0.0f, WR15TO20 = 0.0f, WR20TO25 = 0.0f, WR25PLUS = 0.0f;
    double X1 = 0.0, X3 = 0.0, DMASS = 0.0;
    int IM = 0;
    QACAC = 0.0f;
    QACAR = 0.0f;
    NACAC = 0.0f;
    NACAR = 0.0f;
    if (DO_FR) {
        ntu_find_rc0((double)((ABCD[(1) - (1)] / ABCD[(2) - (1)])), &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->WMAS[((1) - (1)) + 3 * ((IAE) - (1))], RX0, &AC->TBLXF[((1) - (1)) + 25 * ((IAE) - (1))], AC->TBLRC);
        DO_FR = false;
    }
    if ((RX0 == 1.0E-9f)) {
        return;
    }
    RX1 = RX0;
    SSRW = ntu_fmin(SSW, 0.03f);
    ES0 = (0.0761f - (1.55E-4f * (TK1D - NTU_TK0C)));
    ALPHA = ((2.0f * ES0) / ((NTU_RV * TK1D) * NTU_RHOW));
    GAMAA = ((4.0f * NTU_POWF(ALPHA, 3.0f)) / (27.0f * AC->BETA1[(IAE) - (1)]));
    RX9 = NTU_POWF((GAMAA / ntu_sq(SSRW)), NTU_THRD);
    RX9 = ntu_fmax(RX9, AC->RXMIN[(IAE) - (1)]);
    if ((RX1 <= RX9)) {
        return;
    }
    if ((RX1 > RS10)) {
        DMASS = 0.E+0;
        { const int ntu_hi_im = NTU_NCCN;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            X3 = ntu_dlnx(RX1, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 3);
            X1 = ntu_dlnx(RS10, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 3);
            DMASS = (DMASS + ((((double)(ZCCNS[(IM) - (1)]) / (double)(AC->RFACT[((IM) - (1)) + 3 * ((IAE) - (1))])) * (NTU_ERF(X3) - NTU_ERF(X1))) / 2.E+0));
        } }
        XMASS = ntu_fmax(0.0f, (float)(DMASS));
        XMASS = ntu_fmin(ABCD[(1) - (1)], XMASS);
        if ((NTU_NAERN[(IAE) - (1)] >= 4)) {
            ABCD[(4) - (1)] = (ABCD[(4) - (1)] + XMASS);
        } else if ((NTU_NAERN[(IAE) - (1)] == 3)) {
            ABCD[(3) - (1)] = (ABCD[(3) - (1)] + XMASS);
        }
        ABCD[(1) - (1)] = (ABCD[(1) - (1)] - XMASS);
        ntu_rswhitby(W1D, RS10, ZCCN10, RACT10, AC->BETA1[(IAE) - (1)], ALPHA, ZCCNS, &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))]);
        ntu_rswhitby(W1D, RS15, ZCCN15, RACT15, AC->BETA1[(IAE) - (1)], ALPHA, ZCCNS, &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))]);
        ntu_rswhitby(W1D, RS20, ZCCN20, RACT20, AC->BETA1[(IAE) - (1)], ALPHA, ZCCNS, &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))]);
        ntu_rswhitby(W1D, RS25, ZCCN25, RACT25, AC->BETA1[(IAE) - (1)], ALPHA, ZCCNS, &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))]);
        WR10 = (NTU_C4PI3W * NTU_POWF(RACT10, 3.0f));
        WR15 = (NTU_C4PI3W * NTU_POWF(RACT15, 3.0f));
        WR20 = (NTU_C4PI3W * NTU_POWF(RACT20, 3.0f));
        WR25 = (NTU_C4PI3W * NTU_POWF(RACT25, 3.0f));
        WR10TO15 = ((0.5f * (WR10 + WR15)) * (ZCCN10 - ZCCN15));
        WR15TO20 = ((0.5f * (WR15 + WR20)) * (ZCCN15 - ZCCN20));
        WR20TO25 = ((0.5f * (WR20 + WR25)) * (ZCCN20 - ZCCN25));
        WR25PLUS = (WR25 * ZCCN25);
        NACAR = ZCCN10;
        QACAR = (((WR10TO15 + WR15TO20) + WR20TO25) + WR25PLUS);
        RX1 = RS10;
        NR1D = (NR1D + NACAR);
    }
    if ((RX1 > RX9)) {
        DMASS = 0.E+0;
        { const int ntu_hi_im = NTU_NCCN;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            X3 = ntu_dlnx(RX1, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 3);
            X1 = ntu_dlnx(RX9, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 3);
            DMASS = (DMASS + ((((double)(ZCCNS[(IM) - (1)]) / (double)(AC->RFACT[((IM) - (1)) + 3 * ((IAE) - (1))])) * (NTU_ERF(X3) - NTU_ERF(X1))) / 2.E+0));
        } }
        XMASS = ntu_fmax(0.0f, (float)(DMASS));
        XMASS = ntu_fmin(XMASS, ABCD[(1) - (1)]);
        if ((NTU_NAERN[(IAE) - (1)] >= 3)) {
            ABCD[(3) - (1)] = (ABCD[(3) - (1)] + XMASS);
        }
        ABCD[(1) - (1)] = (ABCD[(1) - (1)] - XMASS);
        ntu_ccnwhitby(W1D, RS10, RX9, ZCN99, RACT, AC->BETA1[(IAE) - (1)], ALPHA, ZCCNS, &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))]);
        ntu_ccnwhitby(W1D, RX1, RX9, ZCN9, RACT, AC->BETA1[(IAE) - (1)], ALPHA, ZCCNS, &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))]);
        NACAC = ntu_fmin(ZCN9, ntu_fmax((ZCN99 - NC1D), 0.0f));
        QACAC = ((NACAC * NTU_C4PI3W) * NTU_POWF(RACT, 3.0f));
        NC1D = (NC1D + NACAC);
        RX1 = RX9;
    }
    RX0 = RX1;
}

// WRF FUNCTION DLNX2 (module_mp_ntu.F:2806-2815)
NTU_DEV double ntu_dlnx2(double DRX, float XMODE, float SIGMA, int N) {
    double DXMODE = 0.0, DSTDV = 0.0;
    double NTU_RET = 0;
    DXMODE = (double)(XMODE);
    DSTDV = (double)(SIGMA);
    NTU_RET = ((NTU_LOG((DRX / DXMODE)) - (ntu_sqd(DSTDV) * (double)(N))) / (NTU_K0 * DSTDV));
    return NTU_RET;
}

// WRF SUBROUTINE DEACTIVA (module_mp_ntu.F:2879-2951)
NTU_DEV_BIG void ntu_deactiva(NtuAeroConst* __restrict__ AC, float NACCV, float QC1D, float& NC1D, float& DCN, float TCN, float& WCN, float& RC, float* ZCCNS, int IAE, bool& DO_FR) {
    double DNACCV = 0.0, DMASS = 0.0, DSQRT2 = 0.0, X0 = 0.0, X1 = 0.0, X2 = 0.0, X3 = 0.0, S1 = 0.0, S2 = 0.0;
    int I = 0, IM = 0;
    DSQRT2 = NTU_K0;
    DNACCV = (double)(ntu_fmin(NACCV, NC1D));
    if ((QC1D < NTU_RSMALL)) {
        DNACCV = (double)(NC1D);
        NC1D = NTU_RLIMIT;
    }
    DMASS = 0.E+0;
    if ((DNACCV > 1.E+1)) {
        if (DO_FR) {
            ntu_find_rc0((double)((DCN / TCN)), &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->WMAS[((1) - (1)) + 3 * ((IAE) - (1))], RC, &AC->TBLXF[((1) - (1)) + 25 * ((IAE) - (1))], AC->TBLRC);
            DO_FR = false;
        }
        S1 = 0.E+0;
        { const int ntu_hi_im = NTU_NCCN;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            X0 = ntu_dlnx(RC, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 0);
            S1 = (S1 + (((double)(ZCCNS[(IM) - (1)]) * (1.E+0 - NTU_ERF(X0))) * 5.E-1));
        } }
        if ((DNACCV > S1)) {
            DCN = (DCN + WCN);
            WCN = 0.0f;
            ntu_find_rc0((double)((DCN / TCN)), &AC->CNMOD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->CNSTD[((1) - (1)) + 3 * ((IAE) - (1))], &AC->WMAS[((1) - (1)) + 3 * ((IAE) - (1))], RC, &AC->TBLXF[((1) - (1)) + 25 * ((IAE) - (1))], AC->TBLRC);
            return;
        }
        S1 = (double)(0.0f);
        { const int ntu_hi_im = NTU_NCCN;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            X0 = ntu_dlnx(RC, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 0);
            S1 = (S1 + (((double)((ZCCNS[(IM) - (1)] / AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))])) / DSQRT2) * ntu_dpdf(X0)));
        } }
        X1 = (double)(RC);
        X2 = ((double)(RC) * NTU_EXP((DNACCV / S1)));
        { const int ntu_hi_i = 4;
        for (I = 1; I <= ntu_hi_i; ++I) {
            S1 = 0.E+0;
            S2 = 0.E+0;
            { const int ntu_hi_im = NTU_NCCN;
            for (IM = 1; IM <= ntu_hi_im; ++IM) {
                X0 = ntu_dlnx2(X1, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 0);
                X3 = ntu_dlnx2(X2, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 0);
                S1 = (S1 + (((double)((ZCCNS[(IM) - (1)] / AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))])) / DSQRT2) * ntu_dpdf(X3)));
                S2 = (S2 + (((double)(ZCCNS[(IM) - (1)]) * (NTU_ERF(X3) - NTU_ERF(X0))) * 5.E-1));
            } }
            DNACCV = (DNACCV - S2);
            X1 = X2;
            X2 = (X2 * NTU_EXP((DNACCV / S1)));
        } }
        X1 = (double)(RC);
        X2 = ntu_dmax(X1, X2);
        { const int ntu_hi_im = NTU_NCCN;
        for (IM = 1; IM <= ntu_hi_im; ++IM) {
            X0 = ntu_dlnx2(X1, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 3);
            X3 = ntu_dlnx2(X2, AC->CNMOD[((IM) - (1)) + 3 * ((IAE) - (1))], AC->CNSTD[((IM) - (1)) + 3 * ((IAE) - (1))], 3);
            DMASS = (DMASS + ((((double)(ZCCNS[(IM) - (1)]) / (double)(AC->RFACT[((IM) - (1)) + 3 * ((IAE) - (1))])) * (NTU_ERF(X3) - NTU_ERF(X0))) * 5.E-1));
        } }
        DCN = (DCN + ntu_fmin(WCN, (float)(DMASS)));
        WCN = (WCN - ntu_fmin(WCN, (float)(DMASS)));
        RC = (float)(X2);
    }
}

// WRF SUBROUTINE FLFLUX (module_mp_ntu.F:3240-3263)
NTU_DEV void ntu_flflux(float* Q1D, float* VT1D, float* RHO, float* DZ, int NK, float DT) {
    const int MAXSTP = 1000;
    float DQDT[NTU_KMAX] = {};
    int NSTEP = 0, K = 0, NS = 0;
    NSTEP = 1;
    { const int ntu_hi_k = NK;
    for (K = 1; K <= ntu_hi_k; ++K) {
        NSTEP = ntu_imax(NSTEP, (int)((((VT1D[(K) - (1)] * DT) / DZ[(K) - (1)]) + 1.0f)));
    } }
    { const int ntu_hi_ns = NSTEP;
    for (NS = 1; NS <= ntu_hi_ns; ++NS) {
        DQDT[(1) - (1)] = (-((VT1D[(1) - (1)] * Q1D[(1) - (1)]) / DZ[(1) - (1)]));
        { const int ntu_hi_k = NK;
        for (K = 2; K <= ntu_hi_k; ++K) {
            DQDT[(K) - (1)] = ((((VT1D[((K - 1)) - (1)] * RHO[((K - 1)) - (1)]) * Q1D[((K - 1)) - (1)]) - ((VT1D[(K) - (1)] * RHO[(K) - (1)]) * Q1D[(K) - (1)])) / (DZ[(K) - (1)] * RHO[(K) - (1)]));
        } }
        { const int ntu_hi_k = NK;
        for (K = 1; K <= ntu_hi_k; ++K) {
            Q1D[(K) - (1)] = (Q1D[(K) - (1)] + ((DQDT[(K) - (1)] * DT) / (float)(NSTEP)));
        } }
    } }
}

// WRF SUBROUTINE ICENU (module_mp_ntu.F:2955-3053)
NTU_DEV_BIG void ntu_icenu(float& TK1D, float P1D, float DT, float RHO, float& QV1D, float& QI1D, float& NI1D, float& VI1D, float& FI1D, float& AI1D, float& I3M1D, float& QNIN) {
    float QVSI = 0.0f, XXLS = 0.0f, ESI = 0.0f, QNDVI = 0.0f, NNDVI = 0.0f, VNDVI = 0.0f, FNDVI = 0.0f, ANDVI = 0.0f, INDVI = 0.0f, NVI0 = 0.0f, EPA = 0.0f, RGDEP = 0.0f, SFCTNV = 0.0f, ICED = 0.0f, GGDEP = 0.0f, SRI = 0.0f, ARDEP0 = 0.0f, ARDEP = 0.0f, COSM1 = 0.0f, GEOF1 = 0.0f, IJDEP0 = 0.0f, IJDEP = 0.0f, DC1 = 0.0f, DC2 = 0.0f, DC3 = 0.0f, TC1D = 0.0f, INR0 = 0.0f, DANGLE = 0.0f, DACTE = 0.0f, QVDMAX = 0.0f, NNUMAX = 0.0f, SSRI = 0.0f, CPM = 0.0f;
    int IDEPNU = 0;
    QNDVI = 0.0f;
    NNDVI = 0.0f;
    VNDVI = 0.0f;
    FNDVI = 0.0f;
    ANDVI = 0.0f;
    INDVI = 0.0f;
    IDEPNU = 8;
    TC1D = (TK1D - NTU_TK0C);
    XXLS = ((3.15E6f - (2.37E3f * TK1D)) + 0.3337E6f);
    ESI = ntu_fmin((0.99f * P1D), ntu_polysvp(TK1D, 1));
    QVSI = ((0.622f * ESI) / (P1D - ESI));
    SRI = (QV1D / QVSI);
    SSRI = ((QV1D / QVSI) - 1.0f);
    QVDMAX = ((QV1D - QVSI) / (1.0f + ((ntu_sq(XXLS) * QV1D) / ((NTU_CP * NTU_RV) * ntu_sq(TK1D)))));
    NNUMAX = ((0.98f * QVDMAX) / NTU_MI0);
    ICED = ((916.7f - (0.175f * TC1D)) - (5.0E-4f * ntu_sq(TC1D)));
    SRI = (QV1D / QVSI);
    SFCTNV = (((76.1f - (0.155f * TC1D)) + (28.5f + (0.25f * TC1D))) * 1.0E-3f);
    if ((IDEPNU == 0)) {
        NNDVI = 0.0f;
    } else if ((IDEPNU == 1)) {
        NVI0 = (1.0E-2f * NTU_EXPF((0.6f * (NTU_TK0C - ntu_fmax(TK1D, 2.46E2f)))));
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, ((NVI0 / RHO) - NI1D)));
    } else if ((IDEPNU == 2)) {
        NVI0 = (5.0f * NTU_EXPF((0.304f * (NTU_TK0C - ntu_fmax(TK1D, 2.46E2f)))));
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, ((NVI0 / RHO) - NI1D)));
    } else if ((IDEPNU == 3)) {
        NVI0 = ((1007.08f * NTU_POWF(SSRI, 4.5f)) * 1.0E3f);
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, ((NVI0 / RHO) - NI1D)));
    } else if ((IDEPNU == 4)) {
        NVI0 = (1.0E3f * NTU_EXPF(((1.296E1f * SSRI) - 6.39E-1f)));
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, ((NVI0 / RHO) - NI1D)));
    } else if ((IDEPNU == 6)) {
        NVI0 = ((QNIN * 4.337E-1f) * NTU_EXPF((-(2.0E6f / (ntu_sq(NTU_LOGF(SRI)) * NTU_POWF(TK1D, 3.0f))))));
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, (NVI0 * DT)));
    } else if ((IDEPNU == 7)) {
        DC1 = (-0.3353619f);
        DC2 = 1.990979f;
        DC3 = 2.175539f;
        INR0 = 1.75E-7f;
        DANGLE = 5.06f;
        DACTE = 3.35E-20f;
        EPA = (ESI * SRI);
        RGDEP = (((2.0f * 2.99E-26f) * SFCTNV) / (((ICED * NTU_BOLTZ) * TK1D) * NTU_LOGF(SRI)));
        GGDEP = ((NTU_C4PI3 * SFCTNV) * ntu_sq(RGDEP));
        ARDEP0 = ((NTU_POWF(SFCTNV, 5.0E-1f) * NTU_POWF((NTU_BOLTZ * TK1D), (-1.5f))) / 1.0E13f);
        ARDEP = ((ntu_sq(EPA) * ARDEP0) / ICED);
        COSM1 = NTU_LOGF((1.0f - NTU_COSF(((DANGLE * NTU_PI) / 1.8E2f))));
        GEOF1 = ntu_fmin(1.0f, NTU_EXPF(((DC1 + (DC2 * COSM1)) + ((DC3 * RGDEP) / INR0))));
        IJDEP0 = NTU_EXPF((((-(1.0f * DACTE)) - (GEOF1 * GGDEP)) / (NTU_BOLTZ * TK1D)));
        IJDEP = (((ARDEP * ntu_sq(INR0)) * NTU_POWF(GEOF1, 5.0E-1f)) * IJDEP0);
        NVI0 = (QNIN * (1.0f - NTU_EXPF((-(IJDEP * DT)))));
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, NVI0));
    } else if ((IDEPNU == 8)) {
        NVI0 = ntu_fmax(0.0f, (((5.94E-5f * NTU_POWF((NTU_TK0C - TK1D), 3.33f)) * NTU_POWF((1.0E6f / 1.0E6f), ((0.0264f * (NTU_TK0C - TK1D)) + 3.3E-3f))) * 1.0E3f));
        NNDVI = ntu_fmin(ntu_fmin(QNIN, NNUMAX), ntu_fmax(0.0f, ((NVI0 / RHO) - NI1D)));
    }
    QNDVI = (NNDVI * NTU_MI0);
    VNDVI = (QNDVI * NTU_IRHOI0);
    FNDVI = ((QNDVI * 1.0f) * NTU_IAMI0);
    ANDVI = NTU_POWF(((NTU_KCIMIN * NNDVI) * ntu_sq(INDVI)), NTU_THRD);
    INDVI = (QNDVI * NTU_IAMI0);
    QNIN = ntu_fmax(0.0f, (QNIN - NNDVI));
    QV1D = ntu_fmax(0.0f, (QV1D - QNDVI));
    QI1D = ntu_fmax(0.0f, (QI1D + QNDVI));
    NI1D = ntu_fmax(0.0f, (NI1D + NNDVI));
    VI1D = ntu_fmax(0.0f, (VI1D + VNDVI));
    FI1D = ntu_fmax(0.0f, (FI1D + FNDVI));
    I3M1D = ntu_fmax(0.0f, (I3M1D + INDVI));
    AI1D = ntu_fmax(0.0f, (AI1D + ANDVI));
    CPM = (NTU_CP * (1.0f + (0.887f * QV1D)));
    XXLS = ((3.15E6f - (2370.0f * TK1D)) + 0.3337E6f);
    TK1D = (TK1D + ((XXLS * QNDVI) / CPM));
}

// WRF SUBROUTINE LARGE_DT (module_mp_ntu.F:4369-6903)
NTU_DEV_BIG void ntu_large_dt(NtuAeroConst* __restrict__ AC, float DT, float& TK1D, float QV1D, float P1D, float RHO, float& QC1D, float& QR1D, float& QI1D, float& QS1D, float& QG1D, float& QH1D, float& NC1D, float& NR1D, float& NI1D, float& NS1D, float& NG1D, float& NH1D, float& VI1D, float& VS1D, float& VG1D, float& FI1D, float& FS1D, float& AI1D, float& AS1D, float& AG1D, float& AH1D, float& I3M1D, float& SASPR, float& GQCTR) {
    const float AVTC = 8.8462E+02f;
    const float BVTC = 9.7593E+07f;
    const float CVTC = (-3.4249E+11f);
    const float AVTR = 2.1454E+00f;
    const float BVTR = (-2.2812E-04f);
    const float CVTR = 2.9676E-09f;
    const float CQC1 = 2.0901E+01f;
    const float CQC2 = 9.9111E-01f;
    const float CQC3 = 4.4182E+00f;
    const float CNC1 = 1.8276E+01f;
    const float CNC2 = 1.0015E+00f;
    const float CNC3 = 1.9838E+00f;
    const float CQR1 = 1.5943E+01f;
    const float CQR2 = 1.1898E+00f;
    const float CQR3 = 4.0073E+00f;
    const float CNR1 = 9.4791E+00f;
    const float CNR2 = 9.7607E-01f;
    const float CNR3 = 1.0858E+00f;
    const float AQ1 = 6.6793E+0f;
    const float BQ1 = 1.0090E+0f;
    const float CQ1 = 1.4095E+0f;
    const float AQ2 = 9.9912E+0f;
    const float BQ2 = (-4.7678E-1f);
    const float CQ2 = (-3.1388E-2f);
    const float AN3 = (-4.3561E0f);
    const float BN3 = 1.9934E0f;
    const float CN3 = 1.6465E-2f;
    const float AN4 = (-4.0731E1f);
    const float BN4 = 5.3720E5f;
    const float CN4 = (-2.0139E-5f);
    const float AQ4 = (-2.1370E1f);
    const float BQ4 = 1.9899E9f;
    const float AN5 = 1.5519E1f;
    const float BN5 = 3.1491E-0f;
    const float CN5 = 4.3989E-1f;
    const float AQ5 = 2.0090E1f;
    const float BQ5 = 2.9626E0f;
    const float CQ5 = 3.2358E0f;
    const float AN6 = (-1.8239E1f);
    const float BN6 = 2.2956E0f;
    const float CN6 = (-2.3261E-4f);
    const float AN7 = (-1.7431E2f);
    const float BN7 = 2.6031E5f;
    const float CN7 = (-9.3613E7f);
    const float AN8 = (-1.6185E2f);
    const float BN8 = 2.2786E5f;
    const float CN8 = (-7.6988E7f);
    const float AQ8 = (-2.3531E1f);
    const float BQ8 = 9.8271E-1f;
    const float CQ8 = (-1.3202E-1f);
    const float DSHED = 1.0E-3f;
    const float XISP = 0.25f;
    const float ISEPL = 0.6f;
    const float ISEPS = 0.6f;
    const float ZC1 = 0.69509913f;
    const float ZC2 = -0.46685819f;
    const float ZC3 = 0.30490087f;
    const float ZC4 = 1.62148100f;
    const float ZP1 = 0.36793126f;
    const float ZP2 = 1.82782890f;
    const float ZP3 = 0.63206874f;
    const float ZP4 = -1.00164090f;
    float TC1D = 0.0f, QNIN = 0.0f, AFAC = 0.0f, AFAR = 0.0f, AFAI = 0.0f, AFAS = 0.0f, AFAG = 0.0f, AFAH = 0.0f, ADAGR = 0.0f, RHOI = 0.0f, RHOS = 0.0f, RHOG = 0.0f, RHOIS = 0.0f, IRHOI = 0.0f, IRHOS = 0.0f, IRHOG = 0.0f, ZETA = 0.0f, ZETA2 = 0.0f, ZETA3 = 0.0f, IAMI = 0.0f, IAMG = 0.0f, DI0Z1 = 0.0f, DI0Z2 = 0.0f, DI0Z3 = 0.0f, DI0Z4 = 0.0f, IPG = 0.0f, IPH = 0.0f, IPF = 0.0f, DNRI = 0.0f, DNIRM = 0.0f, DNIAG = 0.0f, DNSAG = 0.0f, DNSAC = 0.0f, DNGAC = 0.0f, DNSRM = 0.0f, DNGRM = 0.0f, AMI = 0.0f, BMI = 0.0f, AMG = 0.0f, AMS = 0.0f, IRHO = 0.0f, AVI = 0.0f, BVI = 0.0f, AVS = 0.0f, BVS = 0.0f, AVG = 0.0f, BVG = 0.0f, AVH = 0.0f, BVH = 0.0f, CAPS = 0.0f, SASR1 = 0.0f, SASR2 = 0.0f, SASR3 = 0.0f, SASR4 = 0.0f, SASP1 = 0.0f, SASP2 = 0.0f, SASP3 = 0.0f, QHOCI = 0.0f, QHORG = 0.0f, QNMCI = 0.0f, QNMRG = 0.0f, QNCCI = 0.0f, QFZCI = 0.0f, QFZRG = 0.0f, QMLIR = 0.0f, QMLIC = 0.0f, QMLSR = 0.0f, QMLGR = 0.0f, QMLHR = 0.0f, QIMII = 0.0f, QIMCSI = 0.0f, QIMRSI = 0.0f, QIMCGI = 0.0f, QIMRGI = 0.0f, QBKRC = 0.0f, QINIG = 0.0f, QINSG = 0.0f, QCNCR = 0.0f, QCNIS = 0.0f, QCNGH = 0.0f, QRMCI = 0.0f, QRMCS = 0.0f, QRMCG = 0.0f, QRMCH = 0.0f, QRMRI = 0.0f, QRMRS = 0.0f, QRMRG = 0.0f, QRMRH = 0.0f, QCLCR = 0.0f, QCLIR = 0.0f, QCLIS = 0.0f, QCLIG = 0.0f, QCLIH = 0.0f, QCLSR = 0.0f, QCLSG = 0.0f, QCLSH = 0.0f, QCLGR = 0.0f, QCLIRG = 0.0f, QCLSRG = 0.0f, QCLGRG = 0.0f, QHWSH = 0.0f, QHWML = 0.0f, QHDRM = 0.0f, QCLSS = 0.0f, NHOCI = 0.0f, NHORG = 0.0f, NNMCI = 0.0f, NNMRG = 0.0f, NNCCI = 0.0f, NFZCI = 0.0f, NFZRG = 0.0f, NMLIR = 0.0f, NMLIC = 0.0f, NMLSR = 0.0f, NMLGR = 0.0f, NMLHR = 0.0f, NIMII = 0.0f, NIMCSI = 0.0f, NIMRSI = 0.0f, NIMCGI = 0.0f, NIMRGI = 0.0f, NBKRC = 0.0f, NBKRR = 0.0f, NINIG = 0.0f, NINSG = 0.0f, NCNCR = 0.0f, NICNIS = 0.0f, NGCNGH = 0.0f, NHCNGH = 0.0f, NRMCI = 0.0f, NRMCS = 0.0f, NRMCG = 0.0f, NRMCH = 0.0f, NRMRI = 0.0f, NRMRS = 0.0f, NRMRG = 0.0f, NRMRH = 0.0f, NCLCC = 0.0f, NCLCR = 0.0f, NCLRR = 0.0f, NCLIR = 0.0f, NCLIS = 0.0f, NCLIG = 0.0f, NCLIH = 0.0f, NCLSR = 0.0f, NCLSS = 0.0f, NCLSG = 0.0f, NCLSH = 0.0f, NCLGR = 0.0f, NCLIRG = 0.0f, NCLSRG = 0.0f, NCLGRG = 0.0f, NHWSH = 0.0f, NSCNIS = 0.0f, VIMII = 0.0f, VCLIS = 0.0f, VICLIS = 0.0f, VSCLIS = 0.0f, VCLIG = 0.0f, VICLIG = 0.0f, VGCLIG = 0.0f, VCLIH = 0.0f, VICNIS = 0.0f, VSCNIS = 0.0f, VIINIG = 0.0f, VGINIG = 0.0f, VFZCI = 0.0f, VFZRG = 0.0f, VIMCSI = 0.0f, VMLIR = 0.0f, VMLIC = 0.0f, VRMCI = 0.0f, VCLIR = 0.0f, VHOCI = 0.0f, VHORG = 0.0f, VNMCI = 0.0f, VNMRG = 0.0f, VNCCI = 0.0f, VIMCGI = 0.0f, VIMRSI = 0.0f, VIMRGI = 0.0f, VCLSS = 0.0f, VCLSG = 0.0f, VSCLSG = 0.0f, VGCLSG = 0.0f, VCLSH = 0.0f, VSINSG = 0.0f, VGINSG = 0.0f, VMLSR = 0.0f, VRMCS = 0.0f, VCLSR = 0.0f, VMLGR = 0.0f, VRMCG = 0.0f, VCNGH = 0.0f, VCLGR = 0.0f, VCLIRG = 0.0f, VCLSRG = 0.0f, VCLGRG = 0.0f, FHOCI = 0.0f, FNMCI = 0.0f, FNCCI = 0.0f, FRMCI = 0.0f, FMLIR = 0.0f, FMLIC = 0.0f, FCLIR = 0.0f, FICLIS = 0.0f, FCLIG = 0.0f, FIMCSI = 0.0f, FCLIH = 0.0f, FICNIS = 0.0f, FINIG = 0.0f, FFZCI = 0.0f, FIMII = 0.0f, FIMCGI = 0.0f, FIMRSI = 0.0f, FIMRGI = 0.0f, FSCNIS = 0.0f, FSCLIS = 0.0f, FCLSG = 0.0f, FCLSH = 0.0f, FINSG = 0.0f, FMLSR = 0.0f, FRMCS = 0.0f, FCLSR = 0.0f, FCLSS = 0.0f, IHOCI = 0.0f, INMCI = 0.0f, INCCI = 0.0f, IRMCI = 0.0f, IMLIR = 0.0f, IMLIC = 0.0f, ICLIR = 0.0f, ICLIS = 0.0f, ICLIG = 0.0f, IIMCSI = 0.0f, ICLIH = 0.0f, ICNIS = 0.0f, IINIG = 0.0f, IFZCI = 0.0f, IIMII = 0.0f, IIMCGI = 0.0f, IIMRSI = 0.0f, IIMRGI = 0.0f, AHOCI = 0.0f, ANMCI = 0.0f, ANCCI = 0.0f, ARMCI = 0.0f, AMLIR = 0.0f, AMLIC = 0.0f, ACLIR = 0.0f, AICLIS = 0.0f, AICLIG = 0.0f, AIMCSI = 0.0f, AICLIH = 0.0f, AICNIS = 0.0f, ASCNIS = 0.0f, AIINIG = 0.0f, AGINIG = 0.0f, AFZCI = 0.0f, AIMII = 0.0f, AIMCGI = 0.0f, AIMRSI = 0.0f, AIMRGI = 0.0f, AMLSR = 0.0f, ASCLIS = 0.0f, ASCLSG = 0.0f, ASCLSH = 0.0f, ARMCS = 0.0f, ACLSR = 0.0f, AGCLIG = 0.0f, ASINSG = 0.0f, AGINSG = 0.0f, AMLGR = 0.0f, ARMCG = 0.0f, AGCNGH = 0.0f, AHCNGH = 0.0f, AGCLSG = 0.0f, ACLGR = 0.0f, ACLIRG = 0.0f, ACLSRG = 0.0f, ACLGRG = 0.0f, AHORG = 0.0f, ANMRG = 0.0f, AFZRG = 0.0f, AMLHR = 0.0f, ACLSS = 0.0f, AHCLIH = 0.0f, AHCLSH = 0.0f, ARMCH = 0.0f, ARMRG = 0.0f, ARMRH = 0.0f, AHDRM = 0.0f, AHWML = 0.0f, ACLSS1 = 0.0f, QCSOUR = 0.0f, QCSINK = 0.0f, QRSOUR = 0.0f, QRSINK = 0.0f, QISOUR = 0.0f, QISINK = 0.0f, QSSOUR = 0.0f, QSSINK = 0.0f, QGSOUR = 0.0f, QGSINK = 0.0f, QHSOUR = 0.0f, QHSINK = 0.0f, NCSOUR = 0.0f, NCSINK = 0.0f, NRSOUR = 0.0f, NRSINK = 0.0f, NISOUR = 0.0f, NISINK = 0.0f, NSSOUR = 0.0f, NSSINK = 0.0f, NGSOUR = 0.0f, NGSINK = 0.0f, NHSOUR = 0.0f, NHSINK = 0.0f, FISOUR = 0.0f, FISINK = 0.0f, FSSOUR = 0.0f, FSSINK = 0.0f, VISOUR = 0.0f, VISINK = 0.0f, VSSOUR = 0.0f, VSSINK = 0.0f, VGSOUR = 0.0f, VGSINK = 0.0f, AISOUR = 0.0f, AISINK = 0.0f, ASSOUR = 0.0f, ASSINK = 0.0f, AGSOUR = 0.0f, AGSINK = 0.0f, AHSOUR = 0.0f, AHSINK = 0.0f, IISOUR = 0.0f, IISINK = 0.0f, RHOAJ = 0.0f, CPM = 0.0f, SSRI = 0.0f, XXLS = 0.0f, XXLF = 0.0f, XXLV = 0.0f, QVSI = 0.0f, ESI = 0.0f, ABI = 0.0f, IDT = 0.0f, SSRW = 0.0f, KAP = 0.0f, DV = 0.0f, MUA = 0.0f, SCN = 0.0f, ACTW = 0.0f, ESW = 0.0f, SFCTNW = 0.0f, SFCTNV = 0.0f, EPA = 0.0f, QVSW = 0.0f, TQCI = 0.0f, TQRSG = 0.0f, FF = 0.0f, RFZ = 0.0f, ICED = 0.0f, VOLMC = 0.0f, IJHOF = 0.0f, AREAC = 0.0f, RGIMF = 0.0f, GGIMF = 0.0f, ARIMF0 = 0.0f, ARIMF = 0.0f, COSM2 = 0.0f, GEOF2 = 0.0f, IJIMF0 = 0.0f, IJIMF = 0.0f, GGCNT = 0.0f, NGCNT0 = 0.0f, NGCNT1 = 0.0f, NGCNT = 0.0f, TCC = 0.0f, CNTGG0 = 0.0f, CNTGG = 0.0f, PSIA = 0.0f, KDIFF = 0.0f, KAPA = 0.0f, CNTKN = 0.0f, CNTFT0 = 0.0f, CNTFT1 = 0.0f, CNTF1 = 0.0f, CNTF2 = 0.0f, CNTFT = 0.0f, IJCNT1 = 0.0f, IJCNT2 = 0.0f, IJCNT3 = 0.0f, DC1 = 0.0f, DC2 = 0.0f, DC3 = 0.0f, INR0 = 0.0f, FANGLE = 0.0f, FACTE = 0.0f, DACTE = 0.0f, RATIO = 0.0f, WNRE = 0.0f, CNRE = 0.0f, PNRE = 0.0f, MRATO = 0.0f, MVRC = 0.0f, MVRR = 0.0f, MVDC = 0.0f, MVDR = 0.0f, MVDI = 0.0f, MVDS = 0.0f, MVDG = 0.0f, MVDH = 0.0f, DSMM = 0.0f, LMVRC = 0.0f, LMVRR = 0.0f, TMP1 = 0.0f, GUC = 0.0f, GUR = 0.0f, RMCSQ = 0.0f, RMRSQ = 0.0f, RMCGQ = 0.0f, RMRGQ = 0.0f, RMCHQ = 0.0f, RMRHQ = 0.0f, RMCSA = 0.0f, RMRSA = 0.0f, RMCGA = 0.0f, RMRGA = 0.0f, RMCHA = 0.0f, RMRHA = 0.0f, VTQ0 = 0.0f, VTQC = 0.0f, VTQR = 0.0f, VTQI = 0.0f, VTQS = 0.0f, VTQG = 0.0f, VTQH = 0.0f, VTN0 = 0.0f, VTNC = 0.0f, VTNR = 0.0f, VTNI = 0.0f, VTNS = 0.0f, VTNG = 0.0f, VTNH = 0.0f, VTAX = 0.0f, VTAC = 0.0f, VTAR = 0.0f, VTAI = 0.0f, VTAS = 0.0f, VTAG = 0.0f, VTAH = 0.0f, VTV0 = 0.0f, VTVI = 0.0f, VTVS = 0.0f, VTVG = 0.0f, VTF0 = 0.0f, VTFI = 0.0f, VTI3M = 0.0f, VTQIC = 0.0f, VTQSC = 0.0f, VTQGC = 0.0f, VTQHC = 0.0f, VTQRI = 0.0f, VTQRS = 0.0f, VTQRG = 0.0f, VTQRH = 0.0f, VTQIS = 0.0f, VTQIG = 0.0f, VTQIH = 0.0f, VTQSG = 0.0f, VTQSH = 0.0f, VTNIC = 0.0f, VTNSC = 0.0f, VTNGC = 0.0f, VTNHC = 0.0f, VTNRI = 0.0f, VTNRS = 0.0f, VTNRG = 0.0f, VTNRH = 0.0f, VTNIS = 0.0f, VTNIG = 0.0f, VTNIH = 0.0f, VTNSG = 0.0f, VTNSH = 0.0f, VTARI = 0.0f, VTARS = 0.0f, VTARG = 0.0f, VTARH = 0.0f, VTAIC = 0.0f, VTASC = 0.0f, VTAGC = 0.0f, VTAHC = 0.0f, VTAIS = 0.0f, VTAIG = 0.0f, VTAIH = 0.0f, VTASG = 0.0f, VTASH = 0.0f, VTVRI = 0.0f, VTVIC = 0.0f, VTVSC = 0.0f, VTVIS = 0.0f, VTVIG = 0.0f, VTVIH = 0.0f, VTVSG = 0.0f, VTFRI = 0.0f, VTFIC = 0.0f, VTFIS = 0.0f, VTFIG = 0.0f, VTFIH = 0.0f, EIS = 0.0f, EIG = 0.0f, EIH = 0.0f, ESG = 0.0f, ESH = 0.0f, EII = 0.0f, ESS = 0.0f, ECI = 0.0f, ECS = 0.0f, ECG = 0.0f, ECH = 0.0f, ERI = 0.0f, ERS = 0.0f, ERG = 0.0f, ERH = 0.0f, EII1 = 0.0f, EII2 = 0.0f, EIS1 = 0.0f, EIS2 = 0.0f, ESS1 = 0.0f, ESS2 = 0.0f, SMLTQ = 0.0f, GMLTQ = 0.0f, HMLTQ = 0.0f, SMLTA = 0.0f, GMLTA = 0.0f, HMLTA = 0.0f, VENQS = 0.0f, VENQG = 0.0f, VENQH = 0.0f, VENAS = 0.0f, VENAG = 0.0f, VENAH = 0.0f, STOKE = 0.0f, DSLL = 0.0f, NGTAL = 0.0f, MVDX = 0.0f, ABW = 0.0f, FSQC = 0.0f, FSNC = 0.0f, FSQR = 0.0f, FSNR = 0.0f, FSQI = 0.0f, FSNI = 0.0f, FSQS = 0.0f, FSNS = 0.0f, FSQG = 0.0f, FSNG = 0.0f, FSQH = 0.0f, FSNH = 0.0f, FSAC = 0.0f, FSAR = 0.0f, FSAI = 0.0f, FSAS = 0.0f, FSAG = 0.0f, FSAH = 0.0f, FSVI = 0.0f, FSVS = 0.0f, FSVG = 0.0f, FSFI = 0.0f, DMWDT = 0.0f, DMIDT = 0.0f, HIDQV = 0.0f, HSDQV = 0.0f, HGDQV = 0.0f, HHDQV = 0.0f, HCWQV = 0.0f, HRWQV = 0.0f, HSWQV = 0.0f, HGWQV = 0.0f, HHWQV = 0.0f, HHDCD = 0.0f, HHDRM = 0.0f, HHDCL = 0.0f, HHDTT = 0.0f, HSWCD = 0.0f, HGWCD = 0.0f, HHWCD = 0.0f, HSWRM = 0.0f, HGWRM = 0.0f, HHWRM = 0.0f, VDMAX = 0.0f, SBMAX = 0.0f, EVMAX = 0.0f, SUMDEP = 0.0f, SUMSUB = 0.0f, SUMEVP = 0.0f, SUMCND = 0.0f, MLMAX = 0.0f, ESW0 = 0.0f, ESI0 = 0.0f, QVSW0 = 0.0f, QVSI0 = 0.0f, XXLF0 = 0.0f, MLWM = 0.0f, MWT = 0.0f, MLWC = 0.0f, DH9 = 0.0f, MCORE = 0.0f, ICOR1 = 0.0f, DIT = 0.0f, BEST = 0.0f, VENQI = 0.0f, QTMP0 = 0.0f, QTMP1 = 0.0f, QTMP2 = 0.0f, QTMP3 = 0.0f, QTMP4 = 0.0f, QTMP5 = 0.0f, QTMP6 = 0.0f, QTMP7 = 0.0f, QTMP8 = 0.0f, QTMP9 = 0.0f, ATMP1 = 0.0f, ATMP2 = 0.0f, ATMP3 = 0.0f, ATMP4 = 0.0f, NCLS2 = 0.0f, BTMP = 0.0f, BSTMP = 0.0f, BGTMP = 0.0f, BHTMP = 0.0f, H2Z = 0.0f, H4Z = 0.0f, QRMC1 = 0.0f, NRMC1 = 0.0f, FRMC1 = 0.0f, IRMC1 = 0.0f, ARMC1 = 0.0f, QRMR1 = 0.0f, NRMR1 = 0.0f, ARMR1 = 0.0f, QFZC1 = 0.0f, NFZC1 = 0.0f, QFZR1 = 0.0f, NFZR1 = 0.0f, AFZR1 = 0.0f, QCLI1 = 0.0f, NCLI1 = 0.0f, QCNI1 = 0.0f, NCNI1 = 0.0f, ACNI1 = 0.0f, FCNI1 = 0.0f, QCLG1 = 0.0f, NCLG1 = 0.0f, ACLG1 = 0.0f, ACLI1 = 0.0f, FCLI1 = 0.0f, QCLS1 = 0.0f, NCLS1 = 0.0f, ACLS1 = 0.0f, LIM1 = 0.0f, LIM2 = 0.0f, LIMA = 0.0f, LIMB = 0.0f, LIMC = 0.0f, LIMD = 0.0f, LIME = 0.0f, LIMF = 0.0f, DICC = 0.0f, DICA = 0.0f, DIAC = 0.0f, DIAA = 0.0f, DIC0 = 0.0f, DIA0 = 0.0f, DIF0 = 0.0f, DSS0 = 0.0f, DSL0 = 0.0f, DIC2 = 0.0f, DIA2 = 0.0f, DILSV = 0.0f, DSLSV = 0.0f, DGLSV = 0.0f, DILSF = 0.0f, DSLSF = 0.0f, SMLR = 0.0f, GMLR = 0.0f, SMLF = 0.0f, GMLF = 0.0f, HMLF = 0.0f, AVWSG = 0.0f, BVWSG = 0.0f, CAPWS = 0.0f, WSAPR = 0.0f, RHOWS = 0.0f, RHOWG = 0.0f, VENWS = 0.0f, VENWG = 0.0f, SSRW0 = 0.0f, SSRI0 = 0.0f, LLMI = 0.0f, LLMS = 0.0f, LLMG = 0.0f, LLMH = 0.0f, KDX = 0.0f, RHOIW = 0.0f, RHOSW = 0.0f, RHOGW = 0.0f, RHOIG = 0.0f, RHOIH = 0.0f, RHOSG = 0.0f, RHOSH = 0.0f, RHOHW = 0.0f, KINV = 0.0f, BEST0 = 0.0f, C1X2 = 0.0f, VTB1 = 0.0f, VTA1 = 0.0f, LIM3 = 0.0f, LIM4 = 0.0f, LIM5 = 0.0f, LIM6 = 0.0f, LIM7 = 0.0f, LIM8 = 0.0f, LAMC = 0.0f, LAMR = 0.0f, LAMS = 0.0f, LAMG = 0.0f, LAMH = 0.0f, LAMI = 0.0f, GC2 = 0.0f, GC3 = 0.0f, GC4 = 0.0f, GC5 = 0.0f, GC6 = 0.0f, GC7 = 0.0f, GR2 = 0.0f, GR3 = 0.0f, GR4 = 0.0f, GR5 = 0.0f, GR6 = 0.0f, GR7 = 0.0f, GI1 = 0.0f, GI2 = 0.0f, GI2G1 = 0.0f, GI2G2 = 0.0f, GI2H1 = 0.0f, GI3 = 0.0f, GI3H1 = 0.0f, GIF1 = 0.0f, GIF2 = 0.0f, GIG1 = 0.0f, GIG2 = 0.0f, GIH1 = 0.0f, GIH2 = 0.0f, GIM1 = 0.0f, GIM2 = 0.0f, GI4 = 0.0f, GI5 = 0.0f, GIM2H1 = 0.0f, GI2H3 = 0.0f, GI2HG1 = 0.0f, GIH2G1 = 0.0f, GIF3 = 0.0f, GI2G3 = 0.0f, GI3G1 = 0.0f, GIZM2H1 = 0.0f, GIG3 = 0.0f, GIM2G1 = 0.0f, GIZM2G1 = 0.0f, GIM3 = 0.0f, GIMF1 = 0.0f, GIZMF1 = 0.0f, GIMG1 = 0.0f, GIZMG1 = 0.0f, GIMH1 = 0.0f, GIZMH1 = 0.0f, GIZM1 = 0.0f, GIZ1 = 0.0f, GIZ2G1 = 0.0f, GIZF1 = 0.0f, GIZG1 = 0.0f, GIZH1 = 0.0f, GIH3 = 0.0f, GS2 = 0.0f, GS3 = 0.0f, GS4 = 0.0f, GS5 = 0.0f, GSM1 = 0.0f, GSM2 = 0.0f, GSM3 = 0.0f, GG2 = 0.0f, GG3 = 0.0f, GG4 = 0.0f, GG5 = 0.0f, GGM1 = 0.0f, GGM2 = 0.0f, GGM3 = 0.0f, GH2 = 0.0f, GH3 = 0.0f, GH4 = 0.0f, GH5 = 0.0f, Z32G = 0.0f, Z32H = 0.0f, Z3BMI = 0.0f;
    int I = 0, WBIN = 0, CBIN = 0, PBIN = 0, TBIN = 0, DBIN = 0;
    DC1 = (-0.3353619f);
    DC2 = 1.990979f;
    DC3 = 2.175539f;
    INR0 = 1.75E-7f;
    FANGLE = 30.98f;
    FACTE = 15.7E-20f;
    DACTE = 3.35E-20f;
    MVDC = 0.0f;
    MVDR = 0.0f;
    MVDI = 0.0f;
    MVDS = 0.0f;
    MVDG = 0.0f;
    MVDH = 0.0f;
    MVRC = 0.0f;
    MVRR = 0.0f;
    SMLF = 0.0f;
    GMLF = 0.0f;
    HMLF = 0.0f;
    FF = 0.0f;
    DMWDT = 0.0f;
    DMIDT = 0.0f;
    ACLSS1 = 0.0f;
    HIDQV = 0.0f;
    HSDQV = 0.0f;
    HGDQV = 0.0f;
    HHDQV = 0.0f;
    HCWQV = 0.0f;
    HRWQV = 0.0f;
    HSWQV = 0.0f;
    HGWQV = 0.0f;
    HHWQV = 0.0f;
    HHDTT = 0.0f;
    QHOCI = 0.0f;
    QHORG = 0.0f;
    QNMCI = 0.0f;
    QNMRG = 0.0f;
    QNCCI = 0.0f;
    QFZCI = 0.0f;
    QFZRG = 0.0f;
    QMLIR = 0.0f;
    QMLIC = 0.0f;
    QMLSR = 0.0f;
    QMLGR = 0.0f;
    QMLHR = 0.0f;
    QIMII = 0.0f;
    QIMCSI = 0.0f;
    QIMRSI = 0.0f;
    QIMCGI = 0.0f;
    QIMRGI = 0.0f;
    QBKRC = 0.0f;
    QINIG = 0.0f;
    QINSG = 0.0f;
    QCNCR = 0.0f;
    QCNIS = 0.0f;
    QCNGH = 0.0f;
    QRMCI = 0.0f;
    QRMCS = 0.0f;
    QRMCG = 0.0f;
    QRMCH = 0.0f;
    QRMRI = 0.0f;
    QRMRS = 0.0f;
    QRMRG = 0.0f;
    QRMRH = 0.0f;
    QCLCR = 0.0f;
    QCLIR = 0.0f;
    QCLIS = 0.0f;
    QCLIG = 0.0f;
    QCLIH = 0.0f;
    QCLSR = 0.0f;
    QCLSG = 0.0f;
    QCLSH = 0.0f;
    QCLGR = 0.0f;
    QCLIRG = 0.0f;
    QCLSRG = 0.0f;
    QCLGRG = 0.0f;
    QHWSH = 0.0f;
    QHDRM = 0.0f;
    QHWML = 0.0f;
    NHOCI = 0.0f;
    NHORG = 0.0f;
    NNMCI = 0.0f;
    NNMRG = 0.0f;
    NNCCI = 0.0f;
    NFZCI = 0.0f;
    NFZRG = 0.0f;
    NMLIR = 0.0f;
    NMLIC = 0.0f;
    NMLSR = 0.0f;
    NMLGR = 0.0f;
    NMLHR = 0.0f;
    NIMII = 0.0f;
    NIMCSI = 0.0f;
    NIMRSI = 0.0f;
    NIMCGI = 0.0f;
    NIMRGI = 0.0f;
    NBKRC = 0.0f;
    NBKRR = 0.0f;
    NINIG = 0.0f;
    NINSG = 0.0f;
    NCNCR = 0.0f;
    NICNIS = 0.0f;
    NGCNGH = 0.0f;
    NRMCI = 0.0f;
    NRMCS = 0.0f;
    NRMCG = 0.0f;
    NRMCH = 0.0f;
    NRMRI = 0.0f;
    NRMRS = 0.0f;
    NRMRG = 0.0f;
    NRMRH = 0.0f;
    NCLCC = 0.0f;
    NCLCR = 0.0f;
    NCLRR = 0.0f;
    NCLIR = 0.0f;
    NCLIS = 0.0f;
    NCLIG = 0.0f;
    NCLIH = 0.0f;
    NCLSR = 0.0f;
    NCLSS = 0.0f;
    NHCNGH = 0.0f;
    NCLSG = 0.0f;
    NCLSH = 0.0f;
    NCLGR = 0.0f;
    NCLIRG = 0.0f;
    NCLSRG = 0.0f;
    NCLGRG = 0.0f;
    NHWSH = 0.0f;
    NSCNIS = 0.0f;
    VCLSS = 0.0f;
    VICNIS = 0.0f;
    VIMII = 0.0f;
    VICLIS = 0.0f;
    VSCLIS = 0.0f;
    VICLIG = 0.0f;
    VGCLIG = 0.0f;
    VCLIH = 0.0f;
    VCLSG = 0.0f;
    VSCNIS = 0.0f;
    VIINIG = 0.0f;
    VGINIG = 0.0f;
    VFZCI = 0.0f;
    VFZRG = 0.0f;
    VMLIR = 0.0f;
    VMLIC = 0.0f;
    VRMCI = 0.0f;
    VCLIR = 0.0f;
    VHOCI = 0.0f;
    VHORG = 0.0f;
    VNMCI = 0.0f;
    VNMRG = 0.0f;
    VNCCI = 0.0f;
    VIMCSI = 0.0f;
    VIMCGI = 0.0f;
    VIMRSI = 0.0f;
    VIMRGI = 0.0f;
    VCLSR = 0.0f;
    VSCLSG = 0.0f;
    VGCLSG = 0.0f;
    VCLSH = 0.0f;
    VSINSG = 0.0f;
    VGINSG = 0.0f;
    VMLSR = 0.0f;
    VRMCS = 0.0f;
    VMLGR = 0.0f;
    VRMCG = 0.0f;
    VCNGH = 0.0f;
    VCLGR = 0.0f;
    VCLIRG = 0.0f;
    VCLSRG = 0.0f;
    VCLGRG = 0.0f;
    VCLIS = 0.0f;
    VCLIG = 0.0f;
    FIMCSI = 0.0f;
    FIMCGI = 0.0f;
    FIMRSI = 0.0f;
    FIMRGI = 0.0f;
    FCLSS = 0.0f;
    FIMII = 0.0f;
    FHOCI = 0.0f;
    FNMCI = 0.0f;
    FNCCI = 0.0f;
    FFZCI = 0.0f;
    FMLIR = 0.0f;
    FCLIR = 0.0f;
    FICLIS = 0.0f;
    FCLIG = 0.0f;
    FCLIH = 0.0f;
    FRMCI = 0.0f;
    FICNIS = 0.0f;
    FINIG = 0.0f;
    FMLIC = 0.0f;
    FSCNIS = 0.0f;
    FSCLIS = 0.0f;
    FCLSG = 0.0f;
    FCLSH = 0.0f;
    FINSG = 0.0f;
    FMLSR = 0.0f;
    FRMCS = 0.0f;
    FCLSR = 0.0f;
    IINIG = 0.0f;
    IIMCSI = 0.0f;
    IIMCGI = 0.0f;
    IIMRSI = 0.0f;
    IIMRGI = 0.0f;
    IIMII = 0.0f;
    IHOCI = 0.0f;
    INMCI = 0.0f;
    INCCI = 0.0f;
    IFZCI = 0.0f;
    IMLIR = 0.0f;
    IMLIC = 0.0f;
    ICLIR = 0.0f;
    ICLIS = 0.0f;
    ICLIG = 0.0f;
    ICLIH = 0.0f;
    IRMCI = 0.0f;
    ICNIS = 0.0f;
    ACLSS = 0.0f;
    AIMCSI = 0.0f;
    AIMCGI = 0.0f;
    AIMRSI = 0.0f;
    AIMRGI = 0.0f;
    AIMII = 0.0f;
    AHOCI = 0.0f;
    ANMCI = 0.0f;
    ANCCI = 0.0f;
    AFZCI = 0.0f;
    AMLIC = 0.0f;
    AMLIR = 0.0f;
    ACLIR = 0.0f;
    AICLIS = 0.0f;
    AICLIG = 0.0f;
    AICLIH = 0.0f;
    ARMCI = 0.0f;
    AICNIS = 0.0f;
    AIINIG = 0.0f;
    AGINIG = 0.0f;
    ASINSG = 0.0f;
    AGINSG = 0.0f;
    ASCLIS = 0.0f;
    ASCLSG = 0.0f;
    ASCLSH = 0.0f;
    ARMCS = 0.0f;
    ACLSR = 0.0f;
    AMLSR = 0.0f;
    AGCLIG = 0.0f;
    AGCLSG = 0.0f;
    AMLGR = 0.0f;
    ARMCG = 0.0f;
    AGCNGH = 0.0f;
    ACLGR = 0.0f;
    ACLIRG = 0.0f;
    ACLSRG = 0.0f;
    ACLGRG = 0.0f;
    AHORG = 0.0f;
    ANMRG = 0.0f;
    AFZRG = 0.0f;
    AMLHR = 0.0f;
    AHCLIH = 0.0f;
    AHCLSH = 0.0f;
    ARMCH = 0.0f;
    ARMRG = 0.0f;
    ARMRH = 0.0f;
    AHDRM = 0.0f;
    AHWML = 0.0f;
    ASCNIS = 0.0f;
    AHCNGH = 0.0f;
    IDT = (1.0f / DT);
    IRHO = (1.0f / RHO);
    TC1D = (TK1D - NTU_TK0C);
    if (((TK1D >= 265.0f) && (TK1D < 268.0f))) {
        FF = ((TC1D + 8.0f) * NTU_THRD);
    }
    if (((TK1D >= 268.0f) && (TK1D <= 270.0f))) {
        FF = (((-3.0f) - TC1D) * 0.5f);
    }
    RHOAJ = NTU_POWF((NTU_RHOSU / RHO), 0.54f);
    CPM = (NTU_CP * (1.0f + (0.887f * QV1D)));
    ESW = ntu_polysvp(TK1D, 0);
    ESI = ntu_polysvp(TK1D, 1);
    QVSW = ((0.622f * ESW) / (P1D - ESW));
    QVSI = ((0.622f * ESI) / (P1D - ESI));
    SSRW = ((QV1D / QVSW) - 1.0f);
    SSRI = ((QV1D / QVSI) - 1.0f);
    ESW0 = ntu_polysvp(NTU_TK0C, 0);
    QVSW0 = ((0.622f * ESW0) / (P1D - ESW0));
    ESI0 = ntu_polysvp(NTU_TK0C, 1);
    QVSI0 = ((0.622f * ESI0) / (P1D - ESI0));
    XXLV = (3.1484E6f - (2370.0f * TK1D));
    XXLS = ((3.15E6f - (2370.0f * TK1D)) + 0.3337E6f);
    XXLF = (2836310.8f - (3.1484E6f - (2370.0f * TK1D)));
    XXLF0 = (2836310.8f - (3.1484E6f - (2370.0f * NTU_TK0C)));
    SSRW0 = ((QV1D / QVSW0) - 1.0f);
    SSRI0 = ((QV1D / QVSI0) - 1.0f);
    DV = ((2.11E-5f * NTU_POWF((TK1D / NTU_TK0C), 1.94f)) * (101325.0f / P1D));
    KAP = (2.3971E-2f + (0.0078E-2f * TC1D));
    MUA = ((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((TK1D / NTU_TK0C), 1.5f));
    SCN = NTU_POWF((MUA / (RHO * DV)), NTU_THRD);
    ICED = ((916.7f - (0.175f * TC1D)) - (5.0E-4f * ntu_sq(TC1D)));
    ABW = ((((TK1D * NTU_RV) / ESW) / DV) + (((XXLV * (((XXLV / TK1D) / NTU_RV) - 1.0f)) / TK1D) / KAP));
    ABI = ((((TK1D * NTU_RV) / ESI) / DV) + (((XXLS * (((XXLS / TK1D) / NTU_RV) - 1.0f)) / TK1D) / KAP));
    SFCTNV = (((0.095f * TC1D) + 104.6f) * 1.0E-3f);
    SFCTNW = ((28.0f + (0.25f * TC1D)) * 1.0E-3f);
    COSM2 = NTU_LOGF((1.0f - NTU_COSF(((FANGLE * NTU_PI) / 180.0f))));
    ACTW = NTU_LOGF(((1.0f * ESW) / ESI));
    EIH = ntu_fmin(1.0f, (0.01f * NTU_EXPF((0.1f * TC1D))));
    ESH = ntu_fmin(1.0f, (0.01f * NTU_EXPF((0.1f * TC1D))));
    if ((QC1D >= NTU_QSMALL)) {
        ntu_solve_afac(AC, TK1D, QC1D, NC1D, LAMC, MVDC, AFAC);
        GC2 = NTU_EXPF(((ntu_gamln((AFAC + 2.0f)) - ntu_gamln((AFAC + 1.0f))) - NTU_LOGF(LAMC)));
        GC3 = NTU_EXPF(((ntu_gamln((AFAC + 3.0f)) - ntu_gamln((AFAC + 1.0f))) - (2.0f * NTU_LOGF(LAMC))));
        GC4 = NTU_EXPF(((ntu_gamln((AFAC + 4.0f)) - ntu_gamln((AFAC + 1.0f))) - (3.0f * NTU_LOGF(LAMC))));
        GC5 = NTU_EXPF(((ntu_gamln((AFAC + 5.0f)) - ntu_gamln((AFAC + 1.0f))) - (4.0f * NTU_LOGF(LAMC))));
        GC6 = NTU_EXPF(((ntu_gamln((AFAC + 6.0f)) - ntu_gamln((AFAC + 1.0f))) - (5.0f * NTU_LOGF(LAMC))));
        GC7 = NTU_EXPF(((ntu_gamln((AFAC + 7.0f)) - ntu_gamln((AFAC + 1.0f))) - (6.0f * NTU_LOGF(LAMC))));
        MVRC = ntu_fmin(ntu_fmax(NTU_POWF(((QC1D / NC1D) / NTU_C4PI3W), NTU_THRD), NTU_RCMIN), NTU_RCMAX);
        LMVRC = NTU_LOGF(MVRC);
        GUC = NTU_EXPF(NTU_EXPF(((NTU_AFU + (NTU_BFU * NTU_POWF(LMVRC, 3.0f))) + (NTU_CFU * NTU_POWF(sqrtf(RHO), 3.0f)))));
        VTQC = ((NTU_EXPF(((CQC1 + (CQC2 * NTU_LOGF(NC1D))) + (CQC3 * NTU_LOGF(MVRC)))) * GUC) / QC1D);
        VTNC = ((NTU_EXPF(((CNC1 + (CNC2 * NTU_LOGF(NC1D))) + (CNC3 * NTU_LOGF(MVRC)))) * GUC) / NC1D);
        VTAC = (NTU_POWF(VTQC, (3.0f / 4.0f)) * NTU_POWF(VTNC, (1.0f / 4.0f)));
        VTQC = ntu_fmin(VTQC, NTU_VTCMAX);
        VTNC = ntu_fmin(VTNC, NTU_VTCMAX);
        VTAC = ntu_fmin(VTAC, NTU_VTCMAX);
        QRMC1 = ((((NTU_PI * NTU_PI) * NTU_RHOW) * NC1D) / 24.0f);
        NRMC1 = ((NTU_PI * NC1D) / 4.0f);
        FRMC1 = ((NTU_PI * NTU_RHOW) * NC1D);
        IRMC1 = (((NTU_PI * NTU_RHOW) * NC1D) / 4.0f);
        ARMC1 = ((NTU_PI * NC1D) / 4.0f);
        QFZC1 = (((((NTU_PI * NTU_PI) * NTU_RHOW) * NC1D) * GC7) / 36.0f);
        NFZC1 = (((NTU_PI * NC1D) * GC4) / 6.0f);
        HCWQV = ((SSRW0 * NTU_EXPF(((AQ1 + (BQ1 * NTU_LOGF(NC1D))) + (CQ1 * LMVRC)))) / ABW);
    }
    if ((QR1D >= NTU_QSMALL)) {
        ntu_solve_afar(TK1D, QR1D, NR1D, LAMR, MVDR, AFAR);
        GR2 = NTU_EXPF(((ntu_gamln((AFAR + 2.0f)) - ntu_gamln((AFAR + 1.0f))) - NTU_LOGF(LAMR)));
        GR3 = NTU_EXPF(((ntu_gamln((AFAR + 3.0f)) - ntu_gamln((AFAR + 1.0f))) - (2.0f * NTU_LOGF(LAMR))));
        GR4 = NTU_EXPF(((ntu_gamln((AFAR + 4.0f)) - ntu_gamln((AFAR + 1.0f))) - (3.0f * NTU_LOGF(LAMR))));
        GR5 = NTU_EXPF(((ntu_gamln((AFAR + 5.0f)) - ntu_gamln((AFAR + 1.0f))) - (4.0f * NTU_LOGF(LAMR))));
        GR6 = NTU_EXPF(((ntu_gamln((AFAR + 6.0f)) - ntu_gamln((AFAR + 1.0f))) - (5.0f * NTU_LOGF(LAMR))));
        GR7 = NTU_EXPF(((ntu_gamln((AFAR + 6.0f)) - ntu_gamln((AFAR + 1.0f))) - (6.0f * NTU_LOGF(LAMR))));
        MVRR = ntu_fmin(ntu_fmax(NTU_POWF(((QR1D / NR1D) / NTU_C4PI3W), NTU_THRD), NTU_RRMIN), NTU_RRMAX);
        LMVRR = NTU_LOGF(MVRR);
        GUR = NTU_EXPF(NTU_EXPF(((NTU_AFU + (NTU_BFU * NTU_POWF(LMVRR, 3.0f))) + (NTU_CFU * NTU_POWF(sqrtf(RHO), 3.0f)))));
        VTQR = ((NTU_EXPF(((CQR1 + (CQR2 * NTU_LOGF(NR1D))) + (CQR3 * NTU_LOGF(MVRR)))) * GUR) / QR1D);
        VTNR = ((NTU_EXPF(((CNR1 + (CNR2 * NTU_LOGF(NR1D))) + (CNR3 * NTU_LOGF(MVRR)))) * GUR) / NR1D);
        VTAR = ((NTU_POWF(VTQR, (3.0f / 4.0f)) * VTNR) * (1.0f / 4.0f));
        VTQR = ntu_fmin(VTQR, NTU_VTRMAX);
        VTNR = ntu_fmin(VTNR, NTU_VTRMAX);
        VTAR = ntu_fmin(VTAR, NTU_VTRMAX);
        QRMR1 = ((((NTU_PI * NTU_PI) * NTU_RHOW) * NR1D) / 24.0f);
        NRMR1 = ((NTU_PI * NR1D) / 4.0f);
        ARMR1 = ((NTU_PI * NR1D) / 4.0f);
        QFZR1 = (((((NTU_PI * NTU_PI) * NTU_RHOW) * NR1D) * GR7) / 36.0f);
        NFZR1 = (((NTU_PI * NR1D) * GR4) / 6.0f);
        AFZR1 = (((NTU_PI * NR1D) * GR6) / 6.0f);
        HRWQV = (((SSRW0 * NR1D) * NTU_EXPF((AQ2 + ((BQ2 + (CQ2 * LMVRR)) * ntu_sq(LMVRR))))) / ABW);
    }
    if ((QI1D >= NTU_QSMALL)) {
        ntu_solve_afai(TK1D, P1D, RHO, QV1D, QI1D, NI1D, VI1D, FI1D, AI1D, I3M1D, ADAGR, ZETA, LAMI, AFAI, MVDI, RHOI, AMI, BMI, AVI, BVI, BEST);
        IPG = (3.0f / (ADAGR + 2.0f));
        IPH = ((3.0f * ADAGR) / (ADAGR + 2.0f));
        IPF = ((3.0f * (ADAGR + 1.0f)) / (ADAGR + 2.0f));
        ZETA2 = ((2.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
        ZETA3 = ((3.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
        DI0Z1 = NTU_POWF(NTU_DI0, ZETA);
        DI0Z2 = NTU_POWF(NTU_DI0, ZETA2);
        DI0Z3 = NTU_POWF(NTU_DI0, ZETA3);
        DI0Z4 = NTU_POWF(NTU_DI0, (4.0f * ZETA));
        IRHOI = (1.0f / RHOI);
        IAMI = (1.0f / AMI);
        QCLI1 = (((NTU_PI * AMI) * NI1D) / 4.0f);
        QCNI1 = ((((NTU_PI * XISP) * AMI) * NI1D) / 6.0f);
        NCLI1 = ((NTU_PI * NI1D) / 4.0f);
        NCNI1 = (((NTU_PI * XISP) * NI1D) / 6.0f);
        ACLI1 = ((NTU_PI * NI1D) / 4.0f);
        ACNI1 = (((NTU_PI * XISP) * NI1D) / 6.0f);
        FCLI1 = (((NTU_PI * NI1D) / 4.0f) / DI0Z3);
        FCNI1 = (((AMI * XISP) * IRHOI) * NI1D);
        Z32G = (ZETA3 + (2.0f * IPG));
        Z32H = (ZETA3 + (2.0f * IPH));
        Z3BMI = (ZETA3 + BMI);
        LLMI = NTU_LOGF(LAMI);
        GI1 = ntu_gamln((AFAI + 1.0f));
        GI2 = NTU_EXPF(((ntu_gamln((AFAI + 2.0f)) - GI1) - LLMI));
        GI3 = NTU_EXPF(((ntu_gamln((AFAI + 3.0f)) - GI1) - (2.0f * LLMI)));
        GI4 = NTU_EXPF(((ntu_gamln((AFAI + 4.0f)) - GI1) - (3.0f * LLMI)));
        GI5 = NTU_EXPF(((ntu_gamln((AFAI + 5.0f)) - GI1) - (4.0f * LLMI)));
        GIM1 = NTU_EXPF(((ntu_gamln(((AFAI + BMI) + 1.0f)) - GI1) - (BMI * LLMI)));
        GIM2 = NTU_EXPF(((ntu_gamln(((AFAI + BMI) + 2.0f)) - GI1) - ((BMI + 1.0f) * LLMI)));
        GIM3 = NTU_EXPF(((ntu_gamln(((AFAI + BMI) + 3.0f)) - GI1) - ((BMI + 2.0f) * LLMI)));
        GIF1 = NTU_EXPF(((ntu_gamln(((AFAI + IPF) + 1.0f)) - GI1) - (IPF * LLMI)));
        GIF2 = NTU_EXPF(((ntu_gamln(((AFAI + IPF) + 2.0f)) - GI1) - ((IPF + 1.0f) * LLMI)));
        GIF3 = NTU_EXPF(((ntu_gamln(((AFAI + IPF) + 3.0f)) - GI1) - ((IPF + 2.0f) * LLMI)));
        GIG1 = NTU_EXPF(((ntu_gamln(((AFAI + IPG) + 1.0f)) - GI1) - (IPG * LLMI)));
        GIG2 = NTU_EXPF(((ntu_gamln(((AFAI + IPG) + 2.0f)) - GI1) - ((IPG + 1.0f) * LLMI)));
        GIG3 = NTU_EXPF(((ntu_gamln(((AFAI + IPG) + 3.0f)) - GI1) - ((IPG + 2.0f) * LLMI)));
        GIH1 = NTU_EXPF(((ntu_gamln(((AFAI + IPH) + 1.0f)) - GI1) - (IPH * LLMI)));
        GIH2 = NTU_EXPF(((ntu_gamln(((AFAI + IPH) + 2.0f)) - GI1) - ((IPH + 1.0f) * LLMI)));
        GIH3 = NTU_EXPF(((ntu_gamln(((AFAI + IPH) + 3.0f)) - GI1) - ((IPH + 2.0f) * LLMI)));
        GIZ1 = NTU_EXPF(((ntu_gamln(((AFAI + ZETA3) + 1.0f)) - GI1) - (ZETA3 * LLMI)));
        GI2G1 = NTU_EXPF(((ntu_gamln(((AFAI + (2.0f * IPG)) + 1.0f)) - GI1) - ((2.0f * IPG) * LLMI)));
        GI2G2 = NTU_EXPF(((ntu_gamln(((AFAI + (2.0f * IPG)) + 2.0f)) - GI1) - (((2.0f * IPG) + 1.0f) * LLMI)));
        GI2G3 = NTU_EXPF(((ntu_gamln(((AFAI + (2.0f * IPG)) + 3.0f)) - GI1) - (((2.0f * IPG) + 2.0f) * LLMI)));
        GI3G1 = NTU_EXPF(((ntu_gamln(((AFAI + (3.0f * IPG)) + 1.0f)) - GI1) - ((3.0f * IPG) * LLMI)));
        GI2H1 = NTU_EXPF(((ntu_gamln(((AFAI + (2.0f * IPH)) + 1.0f)) - GI1) - ((2.0f * IPH) * LLMI)));
        GI2H3 = NTU_EXPF(((ntu_gamln(((AFAI + (2.0f * IPH)) + 3.0f)) - GI1) - (((2.0f * IPH) + 2.0f) * LLMI)));
        GI3H1 = NTU_EXPF(((ntu_gamln(((AFAI + (3.0f * IPH)) + 1.0f)) - GI1) - ((3.0f * IPH) * LLMI)));
        GIMF1 = NTU_EXPF(((ntu_gamln((((AFAI + BMI) + IPF) + 1.0f)) - GI1) - ((BMI + IPF) * LLMI)));
        GIMG1 = NTU_EXPF(((ntu_gamln((((AFAI + BMI) + IPG) + 1.0f)) - GI1) - ((BMI + IPG) * LLMI)));
        GIMH1 = NTU_EXPF(((ntu_gamln((((AFAI + BMI) + IPH) + 1.0f)) - GI1) - ((BMI + IPH) * LLMI)));
        GIZM1 = NTU_EXPF(((ntu_gamln((((AFAI + ZETA3) + BMI) + 1.0f)) - GI1) - ((ZETA3 + BMI) * LLMI)));
        GIZF1 = NTU_EXPF(((ntu_gamln((((AFAI + ZETA3) + IPF) + 1.0f)) - GI1) - ((ZETA3 + IPF) * LLMI)));
        GIZG1 = NTU_EXPF(((ntu_gamln((((AFAI + ZETA3) + IPG) + 1.0f)) - GI1) - ((ZETA3 + IPG) * LLMI)));
        GIZH1 = NTU_EXPF(((ntu_gamln((((AFAI + ZETA3) + IPH) + 1.0f)) - GI1) - ((ZETA3 + IPH) * LLMI)));
        GI2HG1 = NTU_EXPF(((ntu_gamln((((AFAI + (2.0f * IPH)) + IPG) + 1.0f)) - GI1) - (((2.0f * IPH) + IPG) * LLMI)));
        GIH2G1 = NTU_EXPF(((ntu_gamln((((AFAI + IPH) + (2.0f * IPG)) + 1.0f)) - GI1) - ((IPH + (2.0f * IPG)) * LLMI)));
        GIM2G1 = NTU_EXPF(((ntu_gamln((((AFAI + BMI) + (2.0f * IPG)) + 1.0f)) - GI1) - ((BMI + (2.0f * IPG)) * LLMI)));
        GIM2H1 = NTU_EXPF(((ntu_gamln((((AFAI + BMI) + (2.0f * IPH)) + 1.0f)) - GI1) - ((BMI + (2.0f * IPH)) * LLMI)));
        GIZ2G1 = NTU_EXPF(((ntu_gamln(((AFAI + Z32G) + 1.0f)) - GI1) - (Z32G * LLMI)));
        GIZMF1 = NTU_EXPF(((ntu_gamln((((AFAI + Z3BMI) + IPF) + 1.0f)) - GI1) - ((Z3BMI + IPF) * LLMI)));
        GIZMG1 = NTU_EXPF(((ntu_gamln((((AFAI + Z3BMI) + IPG) + 1.0f)) - GI1) - ((Z3BMI + IPG) * LLMI)));
        GIZMH1 = NTU_EXPF(((ntu_gamln((((AFAI + Z3BMI) + IPH) + 1.0f)) - GI1) - ((Z3BMI + IPH) * LLMI)));
        GIZM2G1 = NTU_EXPF(((ntu_gamln((((AFAI + Z32G) + BMI) + 1.0f)) - GI1) - ((Z32G + BMI) * LLMI)));
        GIZM2H1 = NTU_EXPF(((ntu_gamln((((AFAI + Z32H) + BMI) + 1.0f)) - GI1) - ((Z32H + BMI) * LLMI)));
        FSQI = NTU_EXPF(((ntu_gamln((((BVI + BMI) + AFAI) + 1.0f)) - ntu_gamln(((BMI + AFAI) + 1.0f))) - (BVI * LLMI)));
        FSNI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 1.0f)) - GI1) - (BVI * LLMI)));
        FSVI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 4.0f)) - ntu_gamln((AFAI + 4.0f))) - (BVI * LLMI)));
        VTQI = ntu_fmin(((RHOAJ * FSQI) * AVI), NTU_VTIMAX);
        VTNI = ntu_fmin(((RHOAJ * FSNI) * AVI), NTU_VTIMAX);
        VTVI = ntu_fmin(((RHOAJ * FSVI) * AVI), NTU_VTIMAX);
        if ((AI1D >= NTU_ASMALL)) {
            FSAI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 3.0f)) - ntu_gamln((AFAI + 3.0f))) - (BVI * LLMI)));
            VTAI = ntu_fmin(((RHOAJ * FSAI) * AVI), NTU_VTIMAX);
        }
        if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
            FSFI = NTU_EXPF(((ntu_gamln((((BVI + ZETA3) + AFAI) + 4.0f)) - ntu_gamln(((ZETA3 + AFAI) + 4.0f))) - (BVI * LLMI)));
            FSVI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 4.0f)) - ntu_gamln((AFAI + 4.0f))) - (BVI * LLMI)));
            VTFI = ntu_fmin(((RHOAJ * FSFI) * AVI), NTU_VTIMAX);
            VTI3M = ntu_fmin(((RHOAJ * FSVI) * AVI), NTU_VTIMAX);
        }
        if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
            H2Z = (ZC2 * ZETA);
            H4Z = (ZC4 * ZETA);
            QTMP0 = NTU_EXPF(((ntu_gamln(((H2Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H2Z + 1.0f))));
            QTMP1 = NTU_EXPF(((ntu_gamln(((H4Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H4Z + 1.0f))));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP2 = (LLMI * (((H2Z + BVI) + IPH) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + BVI) + IPH) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + BVI) + IPH) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + BVI) + IPH) + AFAI) + 2.0f)) - GI1) - QTMP3));
                VENQI = ((((((NTU_AVIS * ZC1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVIS * ZC3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVIS * ZC1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA2)))) + ((((NTU_BVIS * ZC3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA2))));
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP2 = (LLMI * (((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP3));
                VENQI = ((((((NTU_AVSG * ZC1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVSG * ZC3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVSG * ZC1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA)))) + ((((NTU_BVSG * ZC3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA))));
            }
        } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
            H2Z = (ZP2 * ZETA);
            H4Z = (ZP4 * ZETA);
            QTMP0 = NTU_EXPF(((ntu_gamln(((H2Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H2Z + 1.0f))));
            QTMP1 = NTU_EXPF(((ntu_gamln(((H4Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H4Z + 1.0f))));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP2 = (LLMI * (((H2Z + BVI) + IPG) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + BVI) + IPG) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + BVI) + IPG) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + BVI) + IPG) + AFAI) + 2.0f)) - GI1) - QTMP3));
                VENQI = ((((((NTU_AVIS * ZP1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVIS * ZP3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVIS * ZP1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z - ZETA)))) + ((((NTU_BVIS * ZP3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z - ZETA))));
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP2 = (LLMI * (((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP3));
                VENQI = ((((((NTU_AVSG * ZP1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVSG * ZP3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVSG * ZP1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z - (ZETA / 2.0f))))) + ((((NTU_BVSG * ZP3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z - (ZETA / 2.0f)))));
            }
        } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
            QTMP0 = NTU_EXPF(((ntu_gamln((AFAI + 2.0f)) - GI1) - NTU_LOGF(LAMI)));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP1 = (LLMI * (BVI + 2.0f));
                QTMP2 = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 3.0f)) - GI1) - QTMP1));
                VENQI = ((NTU_AVIS * QTMP0) + ((NTU_BVIS * BTMP) * QTMP2));
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP1 = (LLMI * (1.5f + (BVI / 2.0f)));
                QTMP2 = NTU_EXPF(((ntu_gamln((((BVI / 2.0f) + AFAI) + 2.5f)) - GI1) - QTMP1));
                VENQI = ((NTU_AVSG * QTMP0) + ((NTU_BVSG * BTMP) * QTMP2));
            }
        }
        HIDQV = ((((((2.0f * NTU_PI) * NI1D) * VENQI) * XXLS) * SSRI0) / ABI);
    }
    if ((QS1D >= NTU_QSMALL)) {
        ntu_solve_afas(TK1D, RHO, QS1D, QC1D, NS1D, VS1D, FS1D, AS1D, AFAS, LAMS, MVDS, RHOS, SASPR, AMS, AVS, BVS);
        LLMS = NTU_LOGF(LAMS);
        GS2 = NTU_EXPF(((ntu_gamln((AFAS + 2.0f)) - ntu_gamln((AFAS + 1.0f))) - LLMS));
        GS3 = NTU_EXPF(((ntu_gamln((AFAS + 3.0f)) - ntu_gamln((AFAS + 1.0f))) - (2.0f * LLMS)));
        GS4 = NTU_EXPF(((ntu_gamln((AFAS + 4.0f)) - ntu_gamln((AFAS + 1.0f))) - (3.0f * LLMS)));
        GS5 = NTU_EXPF(((ntu_gamln((AFAS + 5.0f)) - ntu_gamln((AFAS + 1.0f))) - (4.0f * LLMS)));
        GSM1 = NTU_EXPF(((ntu_gamln(((AFAS + NTU_BMS) + 1.0f)) - ntu_gamln((AFAS + 1.0f))) - (NTU_BMS * LLMS)));
        GSM2 = NTU_EXPF(((ntu_gamln(((AFAS + NTU_BMS) + 2.0f)) - ntu_gamln((AFAS + 1.0f))) - ((NTU_BMS + 1.0f) * LLMS)));
        GSM3 = NTU_EXPF(((ntu_gamln(((AFAS + NTU_BMS) + 3.0f)) - ntu_gamln((AFAS + 1.0f))) - ((NTU_BMS + 2.0f) * LLMS)));
        IRHOS = (1.0f / RHOS);
        FSQS = NTU_EXPF(((ntu_gamln((((BVS + NTU_BMS) + AFAS) + 1.0f)) - ntu_gamln(((NTU_BMS + AFAS) + 1.0f))) - (BVS * LLMS)));
        FSNS = NTU_EXPF(((ntu_gamln(((BVS + AFAS) + 1.0f)) - ntu_gamln((AFAS + 1.0f))) - (BVS * LLMS)));
        FSVS = NTU_EXPF(((ntu_gamln(((BVS + AFAS) + 4.0f)) - ntu_gamln((AFAS + 4.0f))) - (BVS * NTU_LOGF(LAMS))));
        VTQS = ntu_fmin(((RHOAJ * FSQS) * AVS), NTU_VTSMAX);
        VTNS = ntu_fmin(((RHOAJ * FSNS) * AVS), NTU_VTSMAX);
        VTVS = ntu_fmin(((RHOAJ * FSVS) * AVS), NTU_VTSMAX);
        if ((AS1D >= NTU_ASMALL)) {
            FSAS = NTU_EXPF(((ntu_gamln(((BVS + AFAS) + 3.0f)) - ntu_gamln((AFAS + 3.0f))) - (BVS * LLMS)));
            VTAS = ntu_fmin(((RHOAJ * FSAS) * AVS), NTU_VTSMAX);
        }
        QCLS1 = (((NTU_PI * AMS) * NS1D) / 4.0f);
        NCLS1 = ((NTU_PI * NS1D) / 4.0f);
        ACLS1 = ((NTU_PI * NS1D) / 4.0f);
        BSTMP = (SCN * sqrtf(((AVS * RHOAJ) / MUA)));
        QTMP1 = (LLMS * (1.5f + (BVS / 2.0f)));
        QTMP2 = NTU_EXPF(((ntu_gamln((AFAS + 2.0f)) - ntu_gamln((AFAS + 1.0f))) - LLMS));
        QTMP3 = NTU_EXPF(((ntu_gamln((((BVS / 2.0f) + AFAS) + 2.5f)) - ntu_gamln((AFAS + 1.0f))) - QTMP1));
        CAPS = ((ZP1 * NTU_POWF(SASPR, (ZP2 / 3.0f))) + (ZP3 * NTU_POWF(SASPR, (ZP4 / 3.0f))));
        VENQS = (((NTU_AVSG * QTMP2) * CAPS) + (((NTU_BVSG * BSTMP) * QTMP3) * CAPS));
        SASR1 = NTU_POWF(SASPR, (-(1.0f / 3.0f)));
        SASR2 = NTU_POWF(SASPR, (-(2.0f / 3.0f)));
        SASR3 = NTU_POWF(SASPR, (1.0f / 3.0f));
        SASR4 = NTU_POWF(SASPR, (2.0f / 3.0f));
        HSDQV = ((((((2.0f * NTU_PI) * NS1D) * VENQS) * XXLS) * SSRI0) / ABI);
        HSWQV = ((((((2.0f * NTU_PI) * NS1D) * VENQS) * XXLV) * SSRW0) / ABW);
    }
    if ((QG1D >= NTU_QSMALL)) {
        ntu_solve_afag(TK1D, RHO, QG1D, QC1D, NG1D, VG1D, AG1D, LAMG, AFAG, MVDG, RHOG, AMG, AVG, BVG);
        LLMG = NTU_LOGF(LAMG);
        GG2 = NTU_EXPF(((ntu_gamln((AFAG + 2.0f)) - ntu_gamln((AFAG + 1.0f))) - LLMG));
        GG3 = NTU_EXPF(((ntu_gamln((AFAG + 3.0f)) - ntu_gamln((AFAG + 1.0f))) - (2.0f * LLMG)));
        GG4 = NTU_EXPF(((ntu_gamln((AFAG + 4.0f)) - ntu_gamln((AFAG + 1.0f))) - (3.0f * LLMG)));
        GG5 = NTU_EXPF(((ntu_gamln((AFAG + 5.0f)) - ntu_gamln((AFAG + 1.0f))) - (4.0f * LLMG)));
        GGM1 = NTU_EXPF(((ntu_gamln(((AFAG + NTU_BMG) + 1.0f)) - ntu_gamln((AFAG + 1.0f))) - (NTU_BMG * LLMG)));
        GGM2 = NTU_EXPF(((ntu_gamln(((AFAG + NTU_BMG) + 2.0f)) - ntu_gamln((AFAG + 1.0f))) - ((NTU_BMG + 1.0f) * LLMG)));
        GGM3 = NTU_EXPF(((ntu_gamln(((AFAG + NTU_BMG) + 3.0f)) - ntu_gamln((AFAG + 1.0f))) - ((NTU_BMG + 2.0f) * LLMG)));
        IAMG = (1.0f / AMG);
        IRHOG = (1.0f / RHOG);
        FSQG = NTU_EXPF(((ntu_gamln((((BVG + NTU_BMG) + AFAG) + 1.0f)) - ntu_gamln(((NTU_BMG + AFAG) + 1.0f))) - (BVG * LLMG)));
        FSNG = NTU_EXPF(((ntu_gamln(((BVG + AFAG) + 1.0f)) - ntu_gamln((AFAG + 1.0f))) - (BVG * LLMG)));
        FSVG = NTU_EXPF(((ntu_gamln(((BVG + AFAG) + 4.0f)) - ntu_gamln((AFAG + 4.0f))) - (BVG * NTU_LOGF(LAMG))));
        VTQG = ntu_fmin(((RHOAJ * FSQG) * AVG), NTU_VTGMAX);
        VTNG = ntu_fmin(((RHOAJ * FSNG) * AVG), NTU_VTGMAX);
        VTVG = ntu_fmin(((RHOAJ * FSVG) * AVG), NTU_VTGMAX);
        if ((AG1D >= NTU_ASMALL)) {
            FSAG = NTU_EXPF(((ntu_gamln(((BVG + AFAG) + 3.0f)) - ntu_gamln((AFAG + 3.0f))) - (BVG * LLMG)));
            VTAG = ntu_fmin(((RHOAJ * FSAG) * AVG), NTU_VTGMAX);
        }
        QCLG1 = (((NTU_PI * AMG) * NG1D) / 4.0f);
        NCLG1 = ((NTU_PI * NG1D) / 4.0f);
        ACLG1 = ((NTU_PI * NG1D) / 4.0f);
        BGTMP = (SCN * sqrtf(((AVG * RHOAJ) / MUA)));
        QTMP1 = (LLMG * (1.5f + (BVG / 2.0f)));
        QTMP2 = NTU_EXPF(((ntu_gamln((AFAG + 2.0f)) - ntu_gamln((AFAG + 1.0f))) - LLMG));
        QTMP3 = NTU_EXPF(((ntu_gamln((((BVG / 2.0f) + AFAG) + 2.5f)) - ntu_gamln((AFAG + 1.0f))) - QTMP1));
        VENQG = ((NTU_AVSG * QTMP2) + ((NTU_BVSG * BGTMP) * QTMP3));
        HGDQV = ((((((2.0f * NTU_PI) * NG1D) * VENQG) * XXLS) * SSRI0) / ABI);
        HGWQV = ((((((2.0f * NTU_PI) * NG1D) * VENQG) * XXLV) * SSRW0) / ABW);
    }
    if ((QH1D >= NTU_QSMALL)) {
        ntu_solve_afah(TK1D, RHO, QH1D, NH1D, AH1D, LAMH, AFAH, MVDH, AVH, BVH);
        LLMH = NTU_LOGF(LAMH);
        GH2 = NTU_EXPF(((ntu_gamln((AFAH + 2.0f)) - ntu_gamln((AFAH + 1.0f))) - LLMH));
        GH3 = NTU_EXPF(((ntu_gamln((AFAH + 3.0f)) - ntu_gamln((AFAH + 1.0f))) - (2.0f * LLMH)));
        GH4 = NTU_EXPF(((ntu_gamln((AFAH + 4.0f)) - ntu_gamln((AFAH + 1.0f))) - (3.0f * LLMH)));
        GH5 = NTU_EXPF(((ntu_gamln((AFAH + 5.0f)) - ntu_gamln((AFAH + 1.0f))) - (4.0f * LLMH)));
        FSQH = NTU_EXPF(((ntu_gamln((((BVH + NTU_BMH) + AFAH) + 1.0f)) - ntu_gamln(((NTU_BMH + AFAH) + 1.0f))) - (BVH * LLMH)));
        FSNH = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 1.0f)) - ntu_gamln((AFAH + 1.0f))) - (BVH * LLMH)));
        VTQH = ntu_fmin(((RHOAJ * FSQH) * AVH), NTU_VTHMAX);
        VTNH = ntu_fmin(((RHOAJ * FSNH) * AVH), NTU_VTHMAX);
        if ((AH1D >= NTU_ASMALL)) {
            FSAH = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 3.0f)) - ntu_gamln((AFAH + 3.0f))) - (BVH * LLMH)));
            VTAH = ntu_fmin(((RHOAJ * FSAH) * AVH), NTU_VTHMAX);
        }
        BHTMP = (SCN * sqrtf(((AVH * RHOAJ) / MUA)));
        QTMP1 = (LLMH * (1.5f + (BVH / 2.0f)));
        QTMP2 = NTU_EXPF(((ntu_gamln((AFAH + 2.0f)) - ntu_gamln((AFAH + 1.0f))) - NTU_LOGF(LAMH)));
        QTMP3 = NTU_EXPF(((ntu_gamln((((BVH / 2.0f) + AFAH) + 2.5f)) - ntu_gamln((AFAH + 1.0f))) - QTMP1));
        QTMP4 = (LLMH * (2.0f + BVH));
        QTMP5 = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 3.0f)) - ntu_gamln((AFAH + 1.0f))) - QTMP4));
        ATMP1 = (LLMH * (0.5f + (BVH / 2.0f)));
        ATMP2 = NTU_EXPF(((ntu_gamln((((BVH / 2.0f) + AFAH) + 1.5f)) - ntu_gamln((AFAH + 1.0f))) - ATMP1));
        ATMP3 = (LLMH * (1.0f + BVH));
        ATMP4 = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 2.0f)) - ntu_gamln((AFAH + 1.0f))) - ATMP3));
        VENQH = ((QTMP2 + ((NTU_VENH1 * BHTMP) * QTMP3)) + ((NTU_VENH2 * ntu_sq(BHTMP)) * QTMP5));
        VENAH = ((1.0f + ((NTU_VENH1 * BHTMP) * ATMP2)) + ((NTU_VENH2 * ntu_sq(BHTMP)) * ATMP4));
        HHDQV = ((((((2.0f * NTU_PI) * NH1D) * VENQH) * XXLS) * SSRI0) / ABI);
        HHWQV = ((((((2.0f * NTU_PI) * NH1D) * VENQH) * XXLV) * SSRW0) / ABW);
    }
    if ((QC1D >= NTU_QSMALL)) {
        NCLCC = (NTU_EXPF(((AN3 + (BN3 * NTU_LOGF(NC1D))) + (CN3 * NTU_POWF(LMVRC, 3.0f)))) * GUC);
        NCNCR = (((NC1D * NC1D) * NTU_EXPF(((AN4 + (BN4 * MVRC)) + (CN4 / MVRC)))) * GUC);
        QCNCR = (NCNCR * NTU_EXPF((AQ4 + ((BQ4 * QC1D) / NC1D))));
        NCNCR = ntu_fmin(NCNCR, (NC1D * IDT));
        QCNCR = ntu_fmin(QCNCR, (QC1D * IDT));
    }
    if ((QR1D >= NTU_QSMALL)) {
        NCLRR = ntu_fmin((NTU_EXPF(((AN6 + (BN6 * NTU_LOGF(NR1D))) + (CN6 / MVRR))) * GUR), (NR1D * IDT));
        NBKRR = (NCLRR * NTU_EXPF((AN7 + ((BN7 + (CN7 * MVRR)) * MVRR))));
        NBKRC = (NCLRR * NTU_EXPF((AN8 + ((BN8 + (CN8 * MVRR)) * MVRR))));
        QBKRC = ntu_fmin((NTU_POWF(NBKRC, BQ8) * NTU_EXPF((AQ8 + (CQ8 * LMVRR)))), (QR1D * IDT));
        NBKRC = ntu_fmin(NBKRC, (NR1D * IDT));
    }
    if (((QC1D >= NTU_QSMALL) && (QR1D >= NTU_QSMALL))) {
        NCLCR = (((NC1D * NR1D) * GUR) * NTU_EXPF(((AN5 + (BN5 * LMVRR)) + (CN5 * LMVRC))));
        QCLCR = (((NC1D * NR1D) * GUR) * NTU_EXPF(((AQ5 + (BQ5 * LMVRR)) + (CQ5 * LMVRC))));
        NCLCR = ntu_fmin(NCLCR, (NC1D * IDT));
        QCLCR = ntu_fmin(QCLCR, (QC1D * IDT));
    }
    if ((TK1D < NTU_TK0C)) {
        if (((((HIDQV + HSDQV) + HGDQV) + HHDQV) >= NTU_QSMALL)) {
            VDMAX = (((XXLS * (QV1D - QVSI0)) / (1.0f + ((ntu_sq(XXLS) * QV1D) / ((CPM * NTU_RV) * ntu_sq(TK1D))))) * IDT);
            SUMDEP = (((HIDQV + HSDQV) + HGDQV) + HHDQV);
            if (((SUMDEP > VDMAX) && (VDMAX >= NTU_QSMALL))) {
                RATIO = ntu_fmin(1.0f, (VDMAX / (SUMDEP + NTU_QSMALL)));
                HIDQV = (HIDQV * RATIO);
                HSDQV = (HSDQV * RATIO);
                HGDQV = (HGDQV * RATIO);
                HHDQV = (HHDQV * RATIO);
            }
        }
        if (((((HIDQV + HSDQV) + HGDQV) + HHDQV) < 0.0f)) {
            SBMAX = (((XXLS * (QV1D - QVSI0)) / (1.0f + ((ntu_sq(XXLS) * QV1D) / ((CPM * NTU_RV) * ntu_sq(TK1D))))) * IDT);
            SUMSUB = (((HIDQV + HSDQV) + HGDQV) + HHDQV);
            if (((SBMAX < 0.0f) && (SUMSUB < (SBMAX * 0.9999f)))) {
                HIDQV = (HIDQV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                HSDQV = (HSDQV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                HGDQV = (HGDQV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                HHDQV = (HHDQV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
            }
        }
        if ((QC1D >= NTU_QSMALL)) {
            TMP1 = (TC1D * TC1D);
            VOLMC = (NTU_C4PI3 * NTU_POWF((MVRC * 1.0E6f), 3.0f));
            IJHOF = NTU_POWF(10.0f, ntu_fmax((-20.0f), (((((-606.3952f) - (52.6611f * TC1D)) - (1.7439f * TMP1)) - ((0.0265f * TMP1) * TC1D)) - (1.536E-4f * ntu_sq(TMP1)))));
            RFZ = (1.0f - NTU_EXPF((-((IJHOF * VOLMC) * DT))));
            if ((TK1D > 243.15f)) {
                RFZ = 0.0f;
            }
            if ((TK1D < 223.15f)) {
                RFZ = 1.0f;
            }
            NHOCI = ((RFZ * NC1D) * IDT);
            QHOCI = ((RFZ * QC1D) * IDT);
            VHOCI = (QHOCI * NTU_IRHOI0);
            FHOCI = ((QHOCI * 1.0f) * NTU_IAMI0);
            IHOCI = (QHOCI * NTU_IAMI0);
            AHOCI = NTU_POWF(((NTU_KCIMIN * NHOCI) * ntu_sq(IHOCI)), NTU_THRD);
        }
        if (((TK1D < 269.15f) && (QC1D >= NTU_QSMALL))) {
            QNMCI = ((QFZC1 * NTU_BIMM) * NTU_EXPF(((NTU_AIMM * (NTU_TK0C - TK1D)) - 1.0f)));
            NNMCI = ((NFZC1 * NTU_BIMM) * NTU_EXPF(((NTU_AIMM * (NTU_TK0C - TK1D)) - 1.0f)));
            QNMCI = ntu_fmin(QNMCI, (QC1D * IDT));
            NNMCI = ntu_fmin(NNMCI, (NC1D * IDT));
            VNMCI = (QNMCI * NTU_IRHOI0);
            FNMCI = ((QNMCI * 1.0f) * NTU_IAMI0);
            INMCI = (QNMCI * NTU_IAMI0);
            ANMCI = NTU_POWF(((NTU_KCIMIN * NNMCI) * ntu_sq(INMCI)), NTU_THRD);
        }
        if (((QC1D >= NTU_QSMALL) && (TK1D < 271.15f))) {
            RGIMF = (((2.0f * 2.99E-26f) * SFCTNW) / (((ICED * NTU_BOLTZ) * TK1D) * ACTW));
            GGCNT = ((NTU_C4PI3 * SFCTNV) * ntu_sq(RGIMF));
            EPA = ((ESI * QV1D) / QVSI);
            GEOF2 = NTU_EXPF(((DC1 + (DC2 * COSM2)) + ((DC3 * RGIMF) / INR0)));
            NGCNT0 = NTU_EXPF((((-DACTE) - (GEOF2 * GGCNT)) / (NTU_BOLTZ * TK1D)));
            NGCNT1 = (EPA / (1.61E-11f * NTU_POWF(TK1D, 0.5f)));
            CNTGG0 = ((ntu_sq(XXLV) - ((XXLV * NTU_RV) * TK1D)) / ((KAP * NTU_RV) * ntu_sq(TK1D)));
            CNTGG = ((1.0f / (((NTU_RV * TK1D) / (ESW * DV)) + CNTGG0)) / NTU_RHOW);
            KDIFF = (((9.1018E-11f * ntu_sq(TK1D)) + (8.8197E-8f * TK1D)) - 1.0654E-5f);
            TCC = (TC1D + (((CNTGG * SSRW) * XXLV) / KDIFF));
            NGCNT = NTU_EXPF((4.11f - (0.262f * TCC)));
            CNTKN = (((6.6E-8f * TK1D) * 101325.0f) / ((293.15f * P1D) * INR0));
            PSIA = (-((((1.0f * NTU_BOLTZ) * TCC) * (1.0f + CNTKN)) / (((6.0f * NTU_PI) * INR0) * MUA)));
            KAPA = 5.39E5f;
            CNTFT0 = (0.4f * ((1.0f + (1.45f * CNTKN)) + ((0.4f * CNTKN) * NTU_EXPF((-(1.0f / CNTKN))))));
            CNTFT1 = ((1.0f + (3.0f * CNTKN)) * (((2.0f * KAP) + ((5.0f * KAPA) * CNTKN)) + KAPA));
            CNTFT = ((CNTFT0 * (KAP + ((2.5f * CNTKN) * KAPA))) / CNTFT1);
            CNTF1 = ((((4.0f * NTU_PI) * NGCNT) * NC1D) * MVRC);
            CNTF2 = ((KAP * (TC1D - TCC)) / P1D);
            IJCNT1 = (-(((((1.0f * CNTF1) * CNTF2) * NTU_RV) * TK1D) / (XXLV * RHO)));
            IJCNT2 = (((CNTF1 * CNTF2) * CNTFT) * IRHO);
            IJCNT3 = (CNTF1 * PSIA);
        }
        if (((TK1D <= 233.15f) && (QR1D >= NTU_QSMALL))) {
            QHORG = (QR1D * IDT);
            NHORG = (NR1D * IDT);
            VHORG = (QHORG / NTU_RHOG0);
            AHORG = NTU_POWF(((NTU_KCGMAX * NHORG) * ntu_sq(((QHORG * NTU_V2M3) / NTU_RHOG0))), NTU_THRD);
        }
        if (((TK1D < 269.15f) && (QR1D >= NTU_QSMALL))) {
            QNMRG = ((QFZR1 * NTU_BIMM) * NTU_EXPF(((NTU_AIMM * (NTU_TK0C - TK1D)) - 1.0f)));
            NNMRG = ((NFZR1 * NTU_BIMM) * NTU_EXPF(((NTU_AIMM * (NTU_TK0C - TK1D)) - 1.0f)));
            QNMRG = ntu_fmin(QNMRG, (QR1D * IDT));
            NNMRG = ntu_fmin(NNMRG, (NR1D * IDT));
            VNMRG = (QNMRG / NTU_RHOG0);
            ANMRG = ((AFZR1 * NTU_BIMM) * NTU_EXPF(((NTU_AIMM * (NTU_TK0C - TK1D)) - 1.0f)));
        }
        if (((QI1D >= NTU_QSMALL) && (MVDI >= 1.0E-5f))) {
            EII1 = NTU_POWF(10.0f, ((3.5E-2f * TC1D) - 0.7f));
            EII2 = (1.0f - (RHOI / NTU_RHOI0));
            EII = ntu_fmin(ntu_fmax(ntu_fmax(EII1, EII2), 0.0f), 1.0f);
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                QCNIS = ((((QCNI1 * VTQI) * EII) * NI1D) * (((GIM2H1 / DI0Z4) + ((2.0f * GIMF1) / DI0Z1)) + (DI0Z2 * GIM2G1)));
                NICNIS = ((((NCNI1 * VTNI) * EII) * NI1D) * (((GI2H1 / DI0Z4) + ((2.0f * GIF1) / DI0Z1)) + (DI0Z2 * GI2G1)));
                if ((AI1D >= NTU_ASMALL)) {
                    AICNIS = ((((ACNI1 * VTAI) * EII) * NI1D) * (((GI2H3 / DI0Z4) + ((2.0f * GIF3) / DI0Z1)) + (DI0Z2 * GI2G3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FICNIS = (((((FCNI1 * VTFI) * EII) * NI1D) / DI0Z3) * (((GIZM2H1 / DI0Z4) + ((2.0f * GIZMF1) / DI0Z1)) + (DI0Z2 * GIZM2G1)));
                    ICNIS = ((((FCNI1 * VTI3M) * EII) * NI1D) * (((GIM2H1 / DI0Z4) + ((2.0f * GIMF1) / DI0Z1)) + (DI0Z2 * GIM2G1)));
                }
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                QCNIS = ((((((QCNI1 * VTQI) * EII) * NI1D) * 4.0f) * DI0Z2) * GIM2G1);
                NICNIS = ((((((NCNI1 * VTNI) * EII) * NI1D) * 4.0f) * DI0Z2) * GI2G1);
                if ((AI1D >= NTU_ASMALL)) {
                    AICNIS = ((((((ACNI1 * VTAI) * EII) * NI1D) * 4.0f) * DI0Z2) * GI2G3);
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FICNIS = (((((((FCNI1 * VTFI) * EII) * NI1D) * 4.0f) * DI0Z2) * GIZM2G1) / DI0Z3);
                    ICNIS = ((((((FCNI1 * VTI3M) * EII) * NI1D) * 4.0f) * DI0Z2) * GIM2G1);
                }
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                QCNIS = (((((QCNI1 * VTQI) * EII) * NI1D) * 4.0f) * GIM3);
                NICNIS = (((((NCNI1 * VTNI) * EII) * NI1D) * 4.0f) * GI3);
                if ((AI1D >= NTU_ASMALL)) {
                    AICNIS = (((((ACNI1 * VTAI) * EII) * NI1D) * 4.0f) * GI5);
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FICNIS = (((((FCNI1 * VTFI) * EII) * NI1D) * 4.0f) * GIM3);
                    ICNIS = (((((FCNI1 * VTI3M) * EII) * NI1D) * 4.0f) * GIM3);
                }
            }
            QCNIS = ntu_fmin((2.0f * QCNIS), (QI1D * IDT));
            NSCNIS = ntu_fmin(NICNIS, (NI1D * IDT));
            NICNIS = ntu_fmin((2.0f * NICNIS), (NI1D * IDT));
            AICNIS = ntu_fmin((2.0f * AICNIS), (AI1D * IDT));
            FICNIS = ntu_fmin((2.0f * FICNIS), (FI1D * IDT));
            ICNIS = ntu_fmin((2.0f * ICNIS), (I3M1D * IDT));
            VICNIS = ntu_fmin((QCNIS * IRHOI), (VI1D * IDT));
            LIM1 = (0.7f * (1.0f - (RHOI / NTU_RHOI0)));
            LIM2 = sqrtf((1.0f - (LIM1 * LIM1)));
            LIM3 = ((((((1.0f + (2.0f * ISEPS)) + ntu_sq(ISEPS)) + ISEPL) + ((2.0f * ISEPS) * ISEPL)) + (ntu_sq(ISEPS) * ISEPL)) / 8.0f);
            LIM4 = ((2.0f * LIM3) * LIM2);
            LIM5 = (LIM3 * ntu_sq(LIM2));
            LIM6 = (LIM3 * LIM2);
            LIM7 = ((2.0f * LIM3) * ntu_sq(LIM2));
            LIM8 = (LIM3 * NTU_POWF(LIM2, 3.0f));
            LIMA = ((((((1.0f + (2.0f * ISEPL)) + ntu_sq(ISEPL)) + ISEPS) + ((2.0f * ISEPL) * ISEPS)) + (ntu_sq(ISEPL) * ISEPS)) / 8.0f);
            LIMB = (LIMA * LIM2);
            LIMC = ((2.0f * LIMA) * LIM2);
            LIMD = ((2.0f * LIMA) * ntu_sq(LIM2));
            LIME = (LIMA * ntu_sq(LIM2));
            LIMF = (LIMA * NTU_POWF(LIM2, 3.0f));
            DICC = (NTU_POWF(NTU_DI0, (-(6.0f * ZETA))) * GI3H1);
            DICA = (GI2HG1 / DI0Z3);
            DIAC = GIH2G1;
            DIAA = (DI0Z3 * GI3G1);
            DIA0 = (DI0Z1 * GIG1);
            DIC0 = (GIH1 / DI0Z2);
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                DILSV = ((((((LIMA + LIMB) + LIMC) + LIMD) + LIME) + LIMF) * DICA);
                DILSF = ((((((LIM3 + LIM4) + LIM5) + LIM6) + LIM7) + LIM8) * DIAC);
                SASP1 = (ntu_fmin(((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)) * DIA0), ((((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2)) * DIC0)) / ntu_fmax(((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)) * DIA0), ((((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2)) * DIC0)));
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                DILSV = ((((((LIMA + LIMB) + LIMC) + LIMD) + LIME) + LIMF) * DIAC);
                DILSF = ((((((LIM3 + LIM4) + LIM5) + LIM6) + LIM7) + LIM8) * DICA);
                SASP1 = (ntu_fmin(((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)) * DIC0), ((((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2)) * DIA0)) / ntu_fmax(((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)) * DIC0), ((((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2)) * DIA0)));
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                DILSV = ((((((LIMA + LIMB) + LIMC) + LIMD) + LIME) + LIMF) * GI4);
                DILSF = ((((((LIM3 + LIM4) + LIM5) + LIM6) + LIM7) + LIM8) * GI4);
                SASP1 = (ntu_fmin((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)), (((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2))) / ntu_fmax((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)), (((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2))));
            }
            DNIAG = ((((2.0f * AMI) * GIM1) * NTU_V2M3) / DILSV);
            VSCNIS = ((NICNIS * DILSV) / NTU_V2M3);
            FSCNIS = (NICNIS * DILSF);
            RATIO = NTU_POWF((RHOI / DNIAG), NTU_THRD);
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                ASCNIS = ((((((ACNI1 * VTAI) * EII) * NI1D) * (((GI2H3 / DI0Z4) + ((2.0f * GIF3) / DI0Z1)) + (DI0Z2 * GI2G3))) * 1.5874f) * ntu_sq(RATIO));
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                ASCNIS = ((((((((ACNI1 * VTAI) * EII) * NI1D) * 4.0f) * DI0Z2) * GI2G3) * 1.5874f) * ntu_sq(RATIO));
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                ASCNIS = (((((((ACNI1 * VTAI) * EII) * NI1D) * 4.0f) * GI5) * 1.5874f) * ntu_sq(RATIO));
            }
        }
        if ((QG1D >= NTU_QSMALL)) {
            DSLL = ((2.0f * 1.0E-2f) * (NTU_EXPF(ntu_fmin(20.0f, (-(TC1D / (((1.1E4f * (QC1D + QR1D)) - (1.3E3f * QG1D)) + 1.0f))))) - 1.0f));
            DSLL = ntu_fmin(1.0f, ntu_fmax(1.0E-4f, DSLL));
            if ((AFAG <= 20.0f)) {
                RATIO = ntu_fmin(1.0f, ntu_fmax(0.0f, fabsf(ntu_gammp((AFAG + 1.0f), (DSLL * LAMG)))));
                NGTAL = (NG1D * (1.0f - RATIO));
                if ((NGTAL >= NTU_NSMALL)) {
                    QCNGH = ntu_fmin((((1.0f - RATIO) * QG1D) * IDT), (QG1D * IDT));
                    NGCNGH = ntu_fmin((((1.0f - RATIO) * NG1D) * IDT), (NG1D * IDT));
                    VCNGH = ntu_fmin((((1.0f - RATIO) * VG1D) * IDT), (VG1D * IDT));
                    if ((AG1D >= NTU_ASMALL)) {
                        AGCNGH = ntu_fmin((((1.0f - RATIO) * AG1D) * IDT), (AG1D * IDT));
                    }
                }
            } else {
                if ((MVDG >= DSLL)) {
                    QCNGH = (QG1D * IDT);
                    NGCNGH = (NG1D * IDT);
                    VCNGH = (VG1D * IDT);
                    if ((AG1D >= NTU_ASMALL)) {
                        AGCNGH = (AG1D * IDT);
                    }
                }
            }
            NHCNGH = (QCNGH / (NTU_AMH * NTU_K7));
            AHCNGH = NTU_POWF(((NTU_KCHMIN * NHCNGH) * ntu_sq((QCNGH * NTU_IAMH))), NTU_THRD);
        }
        if (((QI1D >= NTU_QSMALL) && (QC1D >= NTU_QSMALL))) {
            VTQ0 = (VTQI - VTQC);
            VTN0 = (VTNI - VTNC);
            VTAX = (VTAI - VTAC);
            VTV0 = (VTVI - VTQC);
            VTF0 = (VTFI - VTQC);
            VTQIC = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQI) * VTQC)));
            VTNIC = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNI) * VTNC)));
            VTAIC = sqrtf(((VTAX * VTAX) + ((0.04f * VTAI) * VTAC)));
            VTVIC = sqrtf(((VTV0 * VTV0) + ((0.04f * VTVI) * VTQC)));
            VTFIC = sqrtf(((VTF0 * VTF0) + ((0.04f * VTFI) * VTQC)));
            DNRI = ntu_fmin(ntu_fmax((-(((5.0E5f * MVDC) * VTQIC) / TC1D)), 0.0f), 6.0f);
            DNIRM = (1.0E3f * ((0.078f + (0.184f * DNRI)) - (0.015f * ntu_sq(DNRI))));
            DNIRM = ntu_fmin(ntu_fmax(DNIRM, NTU_RHOIMIN), NTU_RHOG0);
            RHOIW = ((QI1D + QC1D) / (((QI1D / RHOI) + (QC1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOIW / RHOI), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDI, 3.0f)), NTU_THRD) * RATIO), MVDI);
            if ((((ADAGR - 1.0f) >= NTU_SLIMIT) && (MVDI >= 1.0E-4f))) {
                WBIN = ntu_imin(ntu_imax(ntu_nint(((MVDC * 1.0E6f) / 10.0f)), 0), 20);
                CNRE = ((((VTQI * NTU_POWF(NTU_DI0, (-(2.0f * ZETA)))) * NTU_POWF(MVDI, IPH)) * RHO) / MUA);
                if ((CNRE < 0.4f)) {
                    CBIN = 0;
                }
                if (((CNRE >= 0.4f) && (CNRE < 0.6f))) {
                    CBIN = 1;
                }
                if (((CNRE >= 0.6f) && (CNRE < 0.9f))) {
                    CBIN = 2;
                }
                if (((CNRE >= 0.9f) && (CNRE < 1.5f))) {
                    CBIN = 3;
                }
                if (((CNRE >= 1.5f) && (CNRE < 3.5f))) {
                    CBIN = 4;
                }
                if (((CNRE >= 3.5f) && (CNRE < 7.5f))) {
                    CBIN = 5;
                }
                if (((CNRE >= 7.5f) && (CNRE < 15.0f))) {
                    CBIN = 6;
                }
                if ((CNRE >= 15.0f)) {
                    CBIN = 7;
                }
                CBIN = ntu_imin(ntu_imax(CBIN, 0), 7);
                ECI = ntu_fmax(0.0f, NTU_IECC[((CBIN * 21) + WBIN)]);
                QRMCI = ((((QRMC1 * ECI) * VTQIC) * NI1D) * (((((GIF1 * GC4) / DI0Z1) + ((GIG1 * GC5) * DI0Z1)) + ((GIH1 * GC5) / DI0Z2)) + GC6));
                NRMCI = ((((NRMC1 * ECI) * VTNIC) * NI1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GC2) * DI0Z1)) + ((GIH1 * GC2) / DI0Z2)) + GC3));
                if ((AI1D >= NTU_ASMALL)) {
                    ARMCI = ((((((ntu_sq(MVDX) - ntu_sq(MVDI)) * NRMC1) * ECI) * VTAIC) * NI1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GC2) * DI0Z1)) + ((GIH1 * GC2) / DI0Z2)) + GC3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FRMCI = (((((((FRMC1 * ECI) * VTFIC) * NI1D) / DNIRM) / 8.0f) / DI0Z3) * (((((GIZF1 * GC4) / DI0Z1) + ((GIZG1 * GC5) * DI0Z1)) + ((GIZH1 * GC5) / DI0Z2)) + (GIZ1 * GC6)));
                    IRMCI = (((((IRMC1 * ECI) * VTVIC) * NI1D) / DNIRM) * (((((GIF1 * GC4) / DI0Z1) + ((GIG1 * GC5) * DI0Z1)) + ((GIH1 * GC5) / DI0Z2)) + GC6));
                }
            } else if ((((1.0f - ADAGR) >= NTU_SLIMIT) && (MVDI >= 1.5E-4f))) {
                WBIN = ntu_imin(ntu_imax(ntu_nint(((MVDC * 1.0E6f) / 10.0f)), 0), 20);
                PNRE = ((((VTQI * NTU_POWF(NTU_DI0, ZETA)) * NTU_POWF(MVDI, IPG)) * RHO) / MUA);
                if ((PNRE < 1.5f)) {
                    PBIN = 0;
                }
                if (((PNRE >= 1.5f) && (PNRE < 6.0f))) {
                    PBIN = 1;
                }
                if (((PNRE >= 6.0f) && (PNRE < 15.0f))) {
                    PBIN = 2;
                }
                if (((PNRE >= 15.0f) && (PNRE < 28.0f))) {
                    PBIN = 3;
                }
                if (((PNRE >= 28.0f) && (PNRE < 43.0f))) {
                    PBIN = 4;
                }
                if (((PNRE >= 43.0f) && (PNRE < 70.0f))) {
                    PBIN = 5;
                }
                if (((PNRE >= 70.0f) && (PNRE < 105.0f))) {
                    PBIN = 6;
                }
                if ((PNRE >= 105.0f)) {
                    PBIN = 7;
                }
                PBIN = ntu_imin(ntu_imax(PBIN, 0), 7);
                ECI = ntu_fmax(0.0f, NTU_IEPC[((PBIN * 21) + WBIN)]);
                QRMCI = ((((QRMC1 * ECI) * VTQIC) * NI1D) * ((((GI2G1 * GC4) * DI0Z2) + (((2.0f * GIG1) * GC5) * DI0Z1)) + GC6));
                NRMCI = ((((NRMC1 * ECI) * VTNIC) * NI1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GC2) * DI0Z1)) + GC3));
                if ((AI1D >= NTU_ASMALL)) {
                    ARMCI = ((((((ntu_sq(MVDX) - ntu_sq(MVDI)) * NRMC1) * ECI) * VTAIC) * NI1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GC2) * DI0Z1)) + GC3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FRMCI = (((((((FRMC1 * ECI) * VTFIC) * NI1D) / DNIRM) / 2.0f) / DI0Z3) * ((((GIZ2G1 * GC4) * DI0Z2) + (((2.0f * GIZG1) * GC5) * DI0Z1)) + (GIZ1 * GC6)));
                    IRMCI = (((((IRMC1 * ECI) * VTVIC) * NI1D) / DNIRM) * ((((GI2G1 * GC4) * DI0Z2) + (((2.0f * GIG1) * GC5) * DI0Z1)) + GC6));
                }
            } else if (((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT) && (MVDI >= 2.0E-4f))) {
                ECI = 0.5f;
                QRMCI = ((((QRMC1 * ECI) * VTQIC) * NI1D) * (((GI3 * GC4) + ((2.0f * GI2) * GC5)) + GC6));
                NRMCI = ((((NRMC1 * ECI) * VTNIC) * NI1D) * ((GI3 + ((2.0f * GI2) * GC2)) + GC3));
                if ((AI1D >= NTU_ASMALL)) {
                    ARMCI = ((((((ntu_sq(MVDX) - ntu_sq(MVDI)) * NRMC1) * ECI) * VTAIC) * NI1D) * ((GI3 + ((2.0f * GI2) * GC2)) + GC3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FRMCI = ((((((FRMC1 * ECI) * VTFIC) * NI1D) / DNIRM) / 4.0f) * (((GI3 * GC4) + ((2.0f * GI2) * GC5)) + GC6));
                    IRMCI = (((((IRMC1 * ECI) * VTVIC) * NI1D) / DNIRM) * (((GI3 * GC4) + ((2.0f * GI2) * GC5)) + GC6));
                }
            }
            QRMCI = ntu_fmin(QRMCI, (QC1D * IDT));
            NRMCI = ntu_fmin(NRMCI, (NC1D * IDT));
            VRMCI = (QRMCI / DNIRM);
            FRMCI = ntu_fmin(FRMCI, ((QC1D * IDT) / NTU_AMW));
            IRMCI = ntu_fmin(IRMCI, ((QC1D * IDT) / NTU_AMW));
            ARMCI = ntu_fmax(0.0f, ntu_fmin(ARMCI, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
            if ((QRMCI > 0.0f)) {
                QINIG = ntu_fmin((2.0f * QRMCI), (QI1D * IDT));
                NINIG = ntu_fmin(((QINIG * NI1D) / QI1D), (NI1D * IDT));
                VIINIG = ntu_fmin((VRMCI + ((QRMCI * VI1D) / QI1D)), (VI1D * IDT));
                VGINIG = (2.0f * VRMCI);
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FINIG = ntu_fmin((FRMCI + ((QRMCI * FI1D) / QI1D)), (FI1D * IDT));
                    IINIG = ntu_fmin((IRMCI + ((QRMCI * I3M1D) / QI1D)), (I3M1D * IDT));
                }
                if ((AI1D >= NTU_ASMALL)) {
                    AIINIG = ntu_fmin((ARMCI + ((QRMCI * AI1D) / QI1D)), (AI1D * IDT));
                    AGINIG = NTU_POWF(((NTU_KCGMAX * NINIG) * ntu_sq(((QINIG * NTU_V2M3) / DNIRM))), NTU_THRD);
                }
            }
        }
        if ((((QS1D >= NTU_QSMALL) && (QC1D >= NTU_QSMALL)) && (MVDS >= 1.5E-4f))) {
            ECS = ((ntu_fmin(MVDC, 3.0E-5f) * 3.333E4f) * sqrtf((ntu_fmin(MVDS, 1.0E-3f) * 1.0E3f)));
            VTQ0 = (VTQS - VTQC);
            VTN0 = (VTNS - VTNC);
            VTAX = (VTAS - VTAC);
            VTV0 = (VTVS - VTQC);
            VTQSC = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQS) * VTQC)));
            VTNSC = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNS) * VTNC)));
            VTASC = sqrtf(((VTAX * VTAX) + ((0.04f * VTAS) * VTAC)));
            VTVSC = sqrtf(((VTV0 * VTV0) + ((0.04f * VTVS) * VTQC)));
            DNRI = ntu_fmin(ntu_fmax((-(((5.0E5f * MVDC) * VTQSC) / TC1D)), 0.0f), 6.0f);
            DNSRM = (1.0E3f * ((0.078f + (0.184f * DNRI)) - (0.015f * ntu_sq(DNRI))));
            DNSRM = ntu_fmin(ntu_fmax(DNSRM, NTU_RHOIMIN), NTU_RHOG0);
            RHOSW = ((QS1D + QC1D) / (((QS1D / RHOS) + (QC1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOSW / RHOS), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDS, 3.0f)), NTU_THRD) * RATIO), MVDS);
            QRMCS = ((((QRMC1 * ECS) * VTQSC) * NS1D) * ((((SASR2 * GS3) * GC4) + (((SASR1 * 2.0f) * GS2) * GC5)) + GC6));
            NRMCS = ((((NRMC1 * ECS) * VTNSC) * NS1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GS2) * GC2)) + GC3));
            QRMCS = ntu_fmin(QRMCS, (QC1D * IDT));
            NRMCS = ntu_fmin(NRMCS, (NC1D * IDT));
            VRMCS = (QRMCS / DNSRM);
            FRMCS = (((((((FRMC1 * ECS) * VTVSC) * NS1D) * SASPR) / DNSRM) / 2.0f) * ((((SASR2 * GS3) * GC4) + (((SASR1 * 2.0f) * GS2) * GC5)) + GC6));
            FRMCS = ntu_fmin(FRMCS, ((QC1D * IDT) / NTU_AMW));
            if ((AS1D >= NTU_ASMALL)) {
                ARMCS = ((((((ntu_sq(MVDX) - ntu_sq(MVDS)) * NRMC1) * ECS) * VTASC) * NS1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GS2) * GC2)) + GC3));
                ARMCS = ntu_fmax(0.0f, ntu_fmin(ARMCS, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
            }
            if ((QRMCS > 0.0f)) {
                NIMCSI = ((3.5E8f * QRMCS) * FF);
                QIMCSI = ntu_fmin((NIMCSI * NTU_MI0), QRMCS);
                VIMCSI = (QIMCSI * NTU_IRHOI0);
                FIMCSI = ((QIMCSI * 1.0f) * NTU_IAMI0);
                IIMCSI = (QIMCSI * NTU_IAMI0);
                AIMCSI = NTU_POWF(((NTU_KCIMIN * NIMCSI) * ntu_sq(IIMCSI)), NTU_THRD);
                QINSG = ntu_fmin((2.0f * QRMCS), (QS1D * IDT));
                NINSG = ntu_fmin(((QINSG * NS1D) / QS1D), (NS1D * IDT));
                VSINSG = ntu_fmin((VRMCS + ((QRMCS * VS1D) / QS1D)), (VS1D * IDT));
                VGINSG = (2.0f * VRMCS);
                FINSG = ntu_fmin((FRMCS + ((QRMCS * FS1D) / QS1D)), (FS1D * IDT));
                if ((AS1D >= NTU_ASMALL)) {
                    ASINSG = ntu_fmin((ARMCS + ((QRMCS * AS1D) / QS1D)), (AS1D * IDT));
                    AGINSG = NTU_POWF(((NTU_KCGMAX * NINSG) * ntu_sq(((QINSG * NTU_V2M3) / DNSRM))), NTU_THRD);
                }
            }
        }
        if ((((QG1D >= NTU_QSMALL) && (QC1D >= NTU_QSMALL)) && (MVDG >= 2.0E-4f))) {
            STOKE = (((NTU_RHOW * VTQG) * ntu_sq(MVDC)) / ((9.0f * MUA) * MVDG));
            STOKE = ntu_fmax(1.5f, ntu_fmin(10.0f, STOKE));
            ECG = (5.5E-1f * NTU_LOG10F((2.51f * STOKE)));
            VTQ0 = (VTQG - VTQC);
            VTN0 = (VTNG - VTNC);
            VTAX = (VTAG - VTAC);
            VTQGC = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQG) * VTQC)));
            VTNGC = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNG) * VTNC)));
            VTAGC = sqrtf(((VTAX * VTAX) + ((0.04f * VTAG) * VTAC)));
            DNRI = ntu_fmin(ntu_fmax((-(((5.0E5f * MVDC) * VTQGC) / TC1D)), 0.0f), 6.0f);
            DNGRM = (1.0E3f * ((0.078f + (0.184f * DNRI)) - (0.015f * ntu_sq(DNRI))));
            DNGRM = ntu_fmin(ntu_fmax(DNGRM, NTU_RHOIMIN), NTU_RHOG0);
            RHOGW = ((QG1D + QC1D) / (((QG1D / RHOG) + (QC1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOGW / RHOG), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD) * RATIO), MVDG);
            QRMCG = ((((QRMC1 * ECG) * VTQGC) * NG1D) * (((GG3 * GC4) + ((2.0f * GG2) * GC5)) + GC6));
            NRMCG = ((((NRMC1 * ECG) * VTNGC) * NG1D) * ((GG3 + ((2.0f * GG2) * GC2)) + GC3));
            QRMCG = ntu_fmin(QRMCG, (QC1D * IDT));
            NRMCG = ntu_fmin(NRMCG, (NC1D * IDT));
            VRMCG = (QRMCG / DNGRM);
            if ((AG1D >= NTU_ASMALL)) {
                ARMCG = ((((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NRMC1) * ECG) * VTAGC) * NG1D) * ((GG3 + ((2.0f * GG2) * GC2)) + GC3));
                ARMCG = ntu_fmax(0.0f, ntu_fmin(ARMCG, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
            }
            if ((QRMCG > 0.0f)) {
                NIMCGI = ((3.5E8f * QRMCG) * FF);
                QIMCGI = ntu_fmin((NIMCGI * NTU_MI0), QRMCG);
                VIMCGI = (QIMCGI * NTU_IRHOI0);
                FIMCGI = ((QIMCGI * 1.0f) * NTU_IAMI0);
                IIMCGI = (QIMCGI * NTU_IAMI0);
                AIMCGI = NTU_POWF(((NTU_KCIMIN * NIMCGI) * ntu_sq(IIMCGI)), NTU_THRD);
            }
        }
        if (((QH1D >= NTU_QSMALL) && (QC1D >= NTU_QSMALL))) {
            ECH = NTU_EXPF((-((8.68E-7f * NTU_POWF(MVDC, (-1.6f))) * MVDH)));
            VTQ0 = (VTQH - VTQC);
            VTN0 = (VTNH - VTNC);
            VTAX = (VTAH - VTAC);
            VTQHC = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQH) * VTQC)));
            VTNHC = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNH) * VTNC)));
            VTAHC = sqrtf(((VTAX * VTAX) + ((0.04f * VTAH) * VTAC)));
            RHOHW = ((QH1D + QC1D) / (((QH1D / NTU_RHOH) + (QC1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOHW / NTU_RHOH), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDH, 3.0f)), NTU_THRD) * RATIO), MVDH);
            QRMCH = ((((QRMC1 * ECH) * VTQHC) * NH1D) * (((GH3 * GC4) + ((2.0f * GH2) * GC5)) + GC6));
            NRMCH = ((((NRMC1 * ECH) * VTNHC) * NH1D) * ((GH3 + ((2.0f * GH2) * GC2)) + GC3));
            QRMCH = ntu_fmin(QRMCH, (QC1D * IDT));
            NRMCH = ntu_fmin(NRMCH, (NC1D * IDT));
            if ((AH1D >= NTU_ASMALL)) {
                ARMCH = ((((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NRMC1) * ECH) * VTAHC) * NH1D) * ((GH3 + ((2.0f * GH2) * GC2)) + GC3));
                ARMCH = ntu_fmax(0.0f, ntu_fmin(ARMCH, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
            }
        }
        if (((QR1D >= NTU_QSMALL) && (QI1D >= NTU_QSMALL))) {
            ERI = 1.0f;
            VTQ0 = (VTQR - VTQI);
            VTN0 = (VTNR - VTNI);
            VTAX = (VTAR - VTAI);
            VTV0 = (VTQR - VTVI);
            VTF0 = (VTQR - VTFI);
            VTQRI = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQR) * VTQI)));
            VTNRI = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNR) * VTNI)));
            VTARI = sqrtf(((VTAX * VTAX) + ((0.04f * VTAR) * VTAI)));
            VTVRI = sqrtf(((VTV0 * VTV0) + ((0.04f * VTQR) * VTVI)));
            VTFRI = sqrtf(((VTF0 * VTF0) + ((0.04f * VTQR) * VTFI)));
            if ((((ADAGR - 1.0f) >= NTU_SLIMIT) && (MVDR >= MVDI))) {
                QCLIR = ((((QCLI1 * ERI) * VTQRI) * NR1D) * ((((GIMF1 / DI0Z1) + ((GIMG1 * GR2) * DI0Z1)) + ((GIMH1 * GR2) / DI0Z2)) + (GIM1 * GR3)));
                NCLIR = ((((NCLI1 * ERI) * VTNRI) * NR1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GR2) * DI0Z1)) + ((GIH1 * GR2) / DI0Z2)) + GR3));
                if ((AI1D >= NTU_ASMALL)) {
                    ACLIR = ((((ACLI1 * ERI) * VTARI) * NR1D) * ((((GIF3 / DI0Z1) + ((GIG3 * GR2) * DI0Z1)) + ((GIH3 * GR2) / DI0Z2)) + (GI3 * GR3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIR = ((((FCLI1 * ERI) * VTFRI) * NR1D) * ((((GIZMF1 / DI0Z1) + ((GIZMG1 * GR2) * DI0Z1)) + ((GIZMH1 * GR2) / DI0Z2)) + (GIZM1 * GR3)));
                    ICLIR = ((((NCLI1 * ERI) * VTVRI) * NR1D) * ((((GIMF1 / DI0Z1) + ((GIMG1 * GR2) * DI0Z1)) + ((GIMH1 * GR2) / DI0Z2)) + (GIM1 * GR3)));
                }
            } else if ((((1.0f - ADAGR) >= NTU_SLIMIT) && (MVDR >= MVDI))) {
                QCLIR = ((((QCLI1 * ERI) * VTQRI) * NR1D) * (((GIM2G1 * DI0Z2) + (((2.0f * GIMG1) * GR2) * DI0Z1)) + (GIM1 * GR3)));
                NCLIR = ((((NCLI1 * ERI) * VTNRI) * NR1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GR2) * DI0Z1)) + GR3));
                if ((AI1D >= NTU_ASMALL)) {
                    ACLIR = ((((ACLI1 * ERI) * VTARI) * NR1D) * (((GI2G3 * DI0Z2) + (((2.0f * GIG3) * GR2) * DI0Z1)) + (GI3 * GR3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIR = ((((FCLI1 * ERI) * VTFRI) * NR1D) * (((GIZM2G1 * DI0Z2) + (((2.0f * GIZMG1) * GR2) * DI0Z1)) + (GIZM1 * GR3)));
                    ICLIR = ((((NCLI1 * ERI) * VTVRI) * NR1D) * (((GIM2G1 * DI0Z2) + (((2.0f * GIMG1) * GR2) * DI0Z1)) + (GIM1 * GR3)));
                }
            } else if (((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT) && (MVDR >= MVDI))) {
                QCLIR = ((((QCLI1 * ERI) * VTQRI) * NR1D) * ((GIM3 + ((2.0f * GIM2) * GR2)) + (GIM1 * GR3)));
                NCLIR = ((((NCLI1 * ERI) * VTNRI) * NR1D) * ((GI3 + ((2.0f * GI2) * GR2)) + GR3));
                if ((AI1D >= NTU_ASMALL)) {
                    ACLIR = ((((ACLI1 * ERI) * VTARI) * NR1D) * ((GI5 + ((2.0f * GI4) * GR2)) + (GI3 * GR3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIR = ((((NCLI1 * ERI) * VTFRI) * NR1D) * ((GIM3 + ((2.0f * GIM2) * GR2)) + (GIM1 * GR3)));
                    ICLIR = ((((NCLI1 * ERI) * VTVRI) * NR1D) * ((GIM3 + ((2.0f * GIM2) * GR2)) + (GIM1 * GR3)));
                }
            }
            QCLIR = ntu_fmin(QCLIR, (QI1D * IDT));
            NCLIR = ntu_fmin(NCLIR, (NI1D * IDT));
            VCLIR = ntu_fmin((QCLIR * IRHOI), (VI1D * IDT));
            ACLIR = ntu_fmin(ACLIR, (AI1D * IDT));
            FCLIR = ntu_fmin(FCLIR, (FI1D * IDT));
            ICLIR = ntu_fmin(ICLIR, (I3M1D * IDT));
            DNRI = ntu_fmin(ntu_fmax((-(((5.0E5f * MVDR) * VTQRI) / TC1D)), 0.0f), 6.0f);
            DNIRM = (1.0E3f * ((0.078f + (0.184f * DNRI)) - (0.015f * ntu_sq(DNRI))));
            DNIRM = ntu_fmin(ntu_fmax(DNIRM, NTU_RHOIMIN), NTU_RHOG0);
            RHOIW = ((QI1D + QR1D) / (((QI1D / RHOI) + (QR1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOIW / DNIRM), NTU_THRD);
            if ((MVDI >= MVDR)) {
                if ((((ADAGR - 1.0f) >= NTU_SLIMIT) && (MVDI >= 1.0E-4f))) {
                    QRMRI = ((((QRMR1 * ERI) * VTQRI) * NI1D) * (((((GIF1 * GR4) / DI0Z1) + ((GIG1 * GR5) * DI0Z1)) + ((GIH1 * GR5) / DI0Z2)) + GR6));
                    NRMRI = ((((NRMR1 * ERI) * VTNRI) * NI1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GR2) * DI0Z1)) + ((GIH1 * GR2) / DI0Z2)) + GR3));
                } else if ((((1.0f - ADAGR) >= NTU_SLIMIT) && (MVDI >= 1.5E-4f))) {
                    QRMRI = ((((QRMR1 * ERI) * VTQRI) * NI1D) * ((((GI2G1 * GR4) * DI0Z2) + (((2.0f * GIG1) * GR5) * DI0Z1)) + GR6));
                    NRMRI = ((((NRMR1 * ERI) * VTNRI) * NI1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GR2) * DI0Z1)) + GR3));
                } else if (((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT) && (MVDI >= 2.0E-4f))) {
                    QRMRI = ((((QRMR1 * ERI) * VTQRI) * NI1D) * (((GI3 * GR4) + ((2.0f * GI2) * GR5)) + GR6));
                    NRMRI = ((((NRMR1 * ERI) * VTNRI) * NI1D) * ((GI3 + ((2.0f * GI2) * GR2)) + GR3));
                }
                QRMRI = ntu_fmin(QRMRI, (QR1D * IDT));
                NRMRI = ntu_fmin(NRMRI, (NR1D * IDT));
            }
            MVDX = ntu_fmax(ntu_fmax((NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDI, 3.0f)), NTU_THRD) * RATIO), MVDI), MVDR);
            QCLIRG = (QRMRI + QCLIR);
            NCLIRG = ((QCLIRG * NTU_V2M3) / (DNIRM * NTU_POWF((RATIO * MVDX), 3.0f)));
            VCLIRG = (QCLIRG / DNIRM);
            ACLIRG = NTU_POWF(((NTU_KCIMIN * NCLIRG) * ntu_sq(((QCLIRG * NTU_V2M3) / DNIRM))), NTU_THRD);
        }
        if (((QR1D >= NTU_QSMALL) && (QS1D >= NTU_QSMALL))) {
            ERS = 1.0f;
            VTQ0 = (VTQR - VTQS);
            VTN0 = (VTNR - VTNS);
            VTAX = (VTAR - VTAS);
            VTQRS = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQR) * VTQS)));
            VTNRS = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNR) * VTNS)));
            VTARS = sqrtf(((VTAX * VTAX) + ((0.04f * VTAR) * VTAS)));
            DNRI = ntu_fmin(ntu_fmax((-(((5.0E5f * MVDR) * VTQRS) / TC1D)), 0.0f), 6.0f);
            DNSRM = (1.0E3f * ((0.078f + (0.184f * DNRI)) - (0.015f * ntu_sq(DNRI))));
            DNSRM = ntu_fmin(ntu_fmax(DNSRM, NTU_RHOIMIN), NTU_RHOG0);
            RHOSW = ((QS1D + QR1D) / (((QS1D / RHOS) + (QR1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOSW / DNSRM), NTU_THRD);
            if (((MVDS >= 2.0E-4f) && (MVDS >= MVDR))) {
                QRMRS = ((((QRMR1 * ERS) * VTQRS) * NS1D) * ((((SASR2 * GS3) * GR4) + (((SASR1 * 2.0f) * GS2) * GR5)) + GR6));
                NRMRS = ((((NRMR1 * ERS) * VTNRS) * NS1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GS2) * GR2)) + GR3));
                QRMRS = ntu_fmin(QRMRS, (QR1D * IDT));
                NRMRS = ntu_fmin(NRMRS, (NR1D * IDT));
            }
            if ((MVDR >= MVDS)) {
                QCLSR = ((((QCLS1 * ERS) * VTQRS) * NR1D) * (((SASR2 * GSM3) + (((SASR1 * 2.0f) * GR2) * GSM2)) + (GR3 * GSM1)));
                NCLSR = ((((NCLS1 * ERS) * VTNRS) * NR1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GR2) * GS2)) + GR3));
                QCLSR = ntu_fmin(QCLSR, (QS1D * IDT));
                NCLSR = ntu_fmin(NCLSR, (NS1D * IDT));
                VCLSR = ntu_fmin((QCLSR * IRHOS), (VS1D * IDT));
                FCLSR = ntu_fmin((((QCLSR * IRHOS) * SASPR) * NTU_V2M3), (FS1D * IDT));
                if ((AS1D >= NTU_ASMALL)) {
                    ACLSR = ((((ACLS1 * ERS) * VTARS) * NR1D) * (((SASR2 * GS5) + (((SASR1 * 2.0f) * GR2) * GS4)) + (GR3 * GS3)));
                    ACLSR = ntu_fmin(ACLSR, (AS1D * IDT));
                }
            }
            if ((QRMRS > 0.0f)) {
                NIMRSI = ((3.5E8f * QRMRS) * FF);
                QIMRSI = ntu_fmin((NIMRSI * NTU_MI0), QRMRS);
                VIMRSI = (QIMRSI * NTU_IRHOI0);
                FIMRSI = ((QIMRSI * 1.0f) * NTU_IAMI0);
                IIMRSI = (QIMRSI * NTU_IAMI0);
                AIMRSI = NTU_POWF(((NTU_KCIMIN * NIMRSI) * ntu_sq(IIMRSI)), NTU_THRD);
            }
            MVDX = ntu_fmax(ntu_fmax((NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDS, 3.0f)), NTU_THRD) * RATIO), MVDR), MVDS);
            QCLSRG = (QRMRS + QCLSR);
            NCLSRG = ((QCLSRG * NTU_V2M3) / (DNSRM * NTU_POWF((RATIO * MVDX), 3.0f)));
            VCLSRG = (QCLSRG / DNSRM);
            ACLSRG = NTU_POWF(((NTU_KCIMIN * NCLSRG) * ntu_sq(((QCLSRG * NTU_V2M3) / DNSRM))), NTU_THRD);
        }
        if (((QR1D >= NTU_QSMALL) && (QG1D >= NTU_QSMALL))) {
            VTQ0 = (VTQR - VTQG);
            VTN0 = (VTNR - VTNG);
            VTAX = (VTAR - VTAG);
            VTQRG = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQR) * VTQG)));
            VTNRG = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNR) * VTNG)));
            VTARG = sqrtf(((VTAX * VTAX) + ((0.04f * VTAR) * VTAG)));
            RHOGW = ((QG1D + QR1D) / (((QG1D / RHOG) + (QR1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOGW / DNGRM), NTU_THRD);
            STOKE = (((NTU_RHOW * fabsf(VTQ0)) * ntu_sq(MVDR)) / ((9.0f * MUA) * MVDG));
            STOKE = ntu_fmax(1.5f, ntu_fmin(10.0f, STOKE));
            ERG = (5.5E-1f * NTU_LOG10F((2.51f * STOKE)));
            DNRI = ntu_fmin(ntu_fmax((-(((5.0E5f * MVDR) * VTQRG) / TC1D)), 0.0f), 6.0f);
            DNGRM = (1.0E3f * ((0.078f + (0.184f * DNRI)) - (0.015f * ntu_sq(DNRI))));
            DNGRM = ntu_fmin(ntu_fmax(DNGRM, NTU_RHOIMIN), NTU_RHOG0);
            if (((MVDG >= 2.0E-4f) && (MVDG >= MVDR))) {
                QRMRG = ((((QRMR1 * ERG) * VTQRG) * NG1D) * (((GG3 * GR4) + ((2.0f * GG2) * GR5)) + GR6));
                NRMRG = ((((NRMR1 * ERG) * VTNRG) * NG1D) * ((GG3 + ((2.0f * GG2) * GR2)) + GR3));
                QRMRG = ntu_fmin(QRMRG, (QR1D * IDT));
                NRMRG = ntu_fmin(NRMRG, (NR1D * IDT));
                if ((AG1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD) * RATIO), MVDG);
                    ARMRG = ((((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NRMR1) * ERG) * VTARG) * NG1D) * ((GG3 + ((2.0f * GG2) * GR2)) + GR3));
                    ARMRG = ntu_fmin(ARMRG, (((QR1D * IDT) * NTU_IAPW) / MVDR));
                }
            }
            if ((MVDR >= MVDG)) {
                QCLGR = ((((QCLG1 * ERG) * VTQRG) * NR1D) * ((GGM3 + ((2.0f * GR2) * GGM2)) + (GR3 * GGM1)));
                NCLGR = ((((NCLG1 * ERG) * VTNRG) * NR1D) * ((GG3 + ((2.0f * GR2) * GG2)) + GR3));
                QCLGR = ntu_fmin(QCLGR, (QG1D * IDT));
                NCLGR = ntu_fmin(NCLGR, (NG1D * IDT));
                VCLGR = ntu_fmin((QCLGR * IRHOG), (VG1D * IDT));
                if ((AG1D >= NTU_ASMALL)) {
                    ACLGR = ((((ACLG1 * ERG) * VTARG) * NR1D) * ((GG5 + ((2.0f * GR2) * GG4)) + (GR3 * GG3)));
                    ACLGR = ntu_fmin(ACLGR, (AG1D * IDT));
                }
            }
            if ((QRMRG > 0.0f)) {
                NIMRGI = ((3.5E8f * QRMRG) * FF);
                QIMRGI = ntu_fmin((NIMRGI * NTU_MI0), QRMRG);
                VIMRGI = (QIMRGI * NTU_IRHOI0);
                FIMRGI = ((QIMRGI * 1.0f) * NTU_IAMI0);
                IIMRGI = (QIMRGI * NTU_IAMI0);
                AIMRGI = NTU_POWF(((NTU_KCIMIN * NIMRGI) * ntu_sq(IIMRGI)), NTU_THRD);
            }
            MVDX = ntu_fmax(ntu_fmax((NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD) * RATIO), MVDR), MVDG);
            QCLGRG = (QRMRG + QCLGR);
            NCLGRG = ((QCLGRG * NTU_V2M3) / (DNGRM * NTU_POWF((RATIO * MVDX), 3.0f)));
            VCLGRG = (QCLGRG / DNGRM);
            ACLGRG = (ARMRG + ACLGR);
        }
        if ((((QH1D >= NTU_QSMALL) && (QR1D >= NTU_QSMALL)) && (MVDH >= MVDR))) {
            ERH = 1.0f;
            VTQ0 = (VTQR - VTQH);
            VTN0 = (VTNR - VTNH);
            VTAX = (VTAR - VTAH);
            VTQRH = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQR) * VTQH)));
            VTNRH = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNR) * VTNH)));
            VTARH = sqrtf(((VTAX * VTAX) + ((0.04f * VTAR) * VTAH)));
            RHOHW = ((QH1D + QR1D) / (((QH1D / NTU_RHOH) + (QR1D / NTU_RHOW)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOHW / NTU_RHOH), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDH, 3.0f)), NTU_THRD) * RATIO), MVDH);
            QRMRH = ((((QRMR1 * ERH) * VTQRH) * NH1D) * (((GH3 * GR4) + ((2.0f * GH2) * GR5)) + GR6));
            NRMRH = ((((NRMR1 * ERH) * VTNRH) * NH1D) * ((GH3 + ((2.0f * GH2) * GR2)) + GR3));
            QRMRH = ntu_fmin(QRMRH, (QR1D * IDT));
            NRMRH = ntu_fmin(NRMRH, (NR1D * IDT));
            if ((AH1D >= NTU_ASMALL)) {
                ARMRH = ((((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NRMR1) * ERH) * VTARH) * NH1D) * ((GH3 + ((2.0f * GH2) * GR2)) + GR3));
                ARMRH = ntu_fmin(ARMRH, (((QR1D * IDT) * NTU_IAPW) / MVDR));
            }
        }
        if ((((QS1D >= NTU_QSMALL) && (QI1D >= NTU_QSMALL)) && (MVDS >= MVDI))) {
            EIS1 = NTU_POWF(10.0f, ((3.5E-2f * TC1D) - 0.7f));
            RHOIS = ((QI1D + QS1D) / (((QI1D / RHOI) + (QS1D / RHOS)) + NTU_ISMALL));
            EIS2 = (1.0f - (RHOIS / NTU_RHOI0));
            EIS = ntu_fmin(ntu_fmax(ntu_fmax(EIS1, EIS2), 0.0f), 1.0f);
            RATIO = NTU_POWF((RHOIS / RHOS), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDI, 3.0f) + NTU_POWF(MVDS, 3.0f)), NTU_THRD) * RATIO), MVDS);
            VTQ0 = (VTQS - VTQI);
            VTN0 = (VTNS - VTNI);
            VTV0 = (VTVS - VTVI);
            VTF0 = (VTVS - VTFI);
            VTQIS = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQI) * VTQS)));
            VTNIS = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNI) * VTNS)));
            VTVIS = sqrtf(((VTV0 * VTV0) + ((0.04f * VTVI) * VTVS)));
            VTFIS = sqrtf(((VTF0 * VTF0) + ((0.04f * VTFI) * VTVS)));
            if (((AS1D >= NTU_ASMALL) || (AI1D >= NTU_ASMALL))) {
                VTAX = (VTAS - VTAI);
                VTAIS = sqrtf(((VTAX * VTAX) + ((0.04f * VTAI) * VTAS)));
            }
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                QCLIS = ((((QCLI1 * EIS) * VTQIS) * NS1D) * ((((GIMF1 / DI0Z1) + (((SASR1 * GIMG1) * GS2) * DI0Z1)) + (((SASR1 * GIMH1) * GS2) / DI0Z2)) + ((SASR2 * GIM1) * GS3)));
                NCLIS = ((((NCLI1 * EIS) * VTNIS) * NS1D) * ((((GIF1 / DI0Z1) + (((SASR1 * GIG1) * GS2) * DI0Z1)) + (((SASR1 * GIH1) * GS2) / DI0Z2)) + (SASR2 * GS3)));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIS = ((((ACLI1 * EIS) * VTAIS) * NS1D) * ((((GIF3 / DI0Z1) + (((SASR1 * GIG3) * GS2) * DI0Z1)) + (((SASR1 * GIH3) * GS2) / DI0Z2)) + ((SASR2 * GI3) * GS3)));
                }
                if ((AS1D >= NTU_ASMALL)) {
                    ASCLIS = ((((((ntu_sq(MVDX) - ntu_sq(MVDS)) * NCLI1) * EIS) * VTAIS) * NS1D) * ((((GIF1 / DI0Z1) + (((SASR1 * GIG1) * GS2) * DI0Z1)) + (((SASR1 * GIH1) * GS2) / DI0Z2)) + (SASR2 * GS3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FICLIS = ((((FCLI1 * EIS) * VTFIS) * NS1D) * ((((GIZMF1 / DI0Z1) + (((SASR1 * GIZMG1) * GS2) * DI0Z1)) + (((SASR1 * GIZMH1) * GS2) / DI0Z2)) + ((SASR2 * GIZM1) * GS3)));
                    ICLIS = ((((NCLI1 * EIS) * VTVIS) * NS1D) * ((((GIMF1 / DI0Z1) + (((SASR1 * GIMG1) * GS2) * DI0Z1)) + (((SASR1 * GIMH1) * GS2) / DI0Z2)) + ((SASR2 * GIM1) * GS3)));
                }
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                QCLIS = ((((QCLI1 * EIS) * VTQIS) * NS1D) * (((GIM2G1 * DI0Z2) + ((((SASR1 * 2.0f) * GIMG1) * GS2) * DI0Z1)) + ((SASR2 * GIM1) * GS3)));
                NCLIS = ((((NCLI1 * EIS) * VTNIS) * NS1D) * (((GI2G1 * DI0Z2) + ((((SASR1 * 2.0f) * GIG1) * GS2) * DI0Z1)) + (SASR2 * GS3)));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIS = ((((ACLI1 * EIS) * VTAIS) * NS1D) * (((GI2G3 * DI0Z2) + ((((SASR1 * 2.0f) * GIG3) * GS2) * DI0Z1)) + ((SASR2 * GI3) * GS3)));
                }
                if ((AS1D >= NTU_ASMALL)) {
                    ASCLIS = ((((((ntu_sq(MVDX) - ntu_sq(MVDS)) * NCLI1) * EIS) * VTAIS) * NS1D) * (((GI2G1 * DI0Z2) + ((((SASR1 * 2.0f) * GIG1) * GS2) * DI0Z1)) + (SASR2 * GS3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FICLIS = ((((FCLI1 * EIS) * VTFIS) * NS1D) * (((GIZM2G1 * DI0Z2) + ((((SASR1 * 2.0f) * GIZMG1) * GS2) * DI0Z1)) + ((SASR2 * GIZM1) * GS3)));
                    ICLIS = ((((NCLI1 * EIS) * VTVIS) * NS1D) * (((GIM2G1 * DI0Z2) + ((((SASR1 * 2.0f) * GIMG1) * GS2) * DI0Z1)) + ((SASR2 * GIM1) * GS3)));
                }
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                QCLIS = ((((QCLI1 * EIS) * VTQIS) * NS1D) * ((GIM3 + (((SASR1 * 2.0f) * GIM2) * GS2)) + ((SASR2 * GIM1) * GS3)));
                NCLIS = ((((NCLI1 * EIS) * VTNIS) * NS1D) * ((GI3 + (((SASR1 * 2.0f) * GI2) * GS2)) + (SASR2 * GS3)));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIS = ((((ACLI1 * EIS) * VTAIS) * NS1D) * ((GI5 + (((SASR1 * 2.0f) * GI4) * GS2)) + ((SASR2 * GI3) * GS3)));
                }
                if ((AS1D >= NTU_ASMALL)) {
                    ASCLIS = ((((((ntu_sq(MVDX) - ntu_sq(MVDS)) * NCLI1) * EIS) * VTAIS) * NS1D) * ((GI3 + (((SASR1 * 2.0f) * GI2) * GS2)) + (SASR2 * GS3)));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FICLIS = ((((NCLI1 * EIS) * VTFIS) * NS1D) * ((GIM3 + (((SASR1 * 2.0f) * GIM2) * GS2)) + ((SASR2 * GIM1) * GS3)));
                    ICLIS = ((((NCLI1 * EIS) * VTVIS) * NS1D) * ((GIM3 + (((SASR1 * 2.0f) * GIM2) * GS2)) + ((SASR2 * GIM1) * GS3)));
                }
            }
            QCLIS = ntu_fmin(QCLIS, (QI1D * IDT));
            NCLIS = ntu_fmin(NCLIS, (NI1D * IDT));
            AICLIS = ntu_fmin(AICLIS, (AI1D * IDT));
            VICLIS = ntu_fmin((QCLIS * IRHOI), (VI1D * IDT));
            FICLIS = ntu_fmin(FICLIS, (FI1D * IDT));
            ICLIS = ntu_fmin(ICLIS, (I3M1D * IDT));
            LIM1 = (0.7f * (1.0f - (RHOI / NTU_RHOI0)));
            LIM2 = sqrtf((1.0f - (LIM1 * LIM1)));
            LIM3 = ((((((1.0f + (2.0f * ISEPS)) + ntu_sq(ISEPS)) + ISEPL) + ((2.0f * ISEPS) * ISEPL)) + (ntu_sq(ISEPS) * ISEPL)) / 8.0f);
            LIM4 = ((2.0f * LIM3) * LIM2);
            LIM5 = (LIM3 * ntu_sq(LIM2));
            LIM6 = (LIM3 * LIM2);
            LIM7 = ((2.0f * LIM3) * ntu_sq(LIM2));
            LIM8 = (LIM3 * NTU_POWF(LIM2, 3.0f));
            LIMA = ((((((1.0f + (2.0f * ISEPL)) + ntu_sq(ISEPL)) + ISEPS) + ((2.0f * ISEPL) * ISEPS)) + (ntu_sq(ISEPL) * ISEPS)) / 8.0f);
            LIMB = (LIMA * LIM2);
            LIMC = ((2.0f * LIMA) * LIM2);
            LIMD = ((2.0f * LIMA) * ntu_sq(LIM2));
            LIME = (LIMA * ntu_sq(LIM2));
            LIMF = (LIMA * NTU_POWF(LIM2, 3.0f));
            DSS0 = (SASR4 * GS2);
            DSL0 = (SASR1 * GS2);
            DICC = (NTU_POWF(NTU_DI0, (-(6.0f * ZETA))) * GI3H1);
            DICA = (GI2HG1 / DI0Z3);
            DIAC = GIH2G1;
            DIAA = (DI0Z3 * GI3G1);
            DIA0 = (DI0Z1 * GIG1);
            DIC0 = (GIH1 / DI0Z2);
            DIF0 = (GIF1 / DI0Z1);
            DIC2 = (GI2H1 / DI0Z4);
            DIA2 = (DI0Z2 * GI2G1);
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                DSLSV = ((((((LIMA * GS4) + (((LIMB * DIA0) * GS3) * SASR2)) + (((LIMC * DIC0) * GS3) * SASR3)) + (((LIMD * DIF0) * GS2) * SASR1)) + (((LIME * DIC2) * GS2) * SASR4)) + (LIMF * DICA));
                DSLSF = (((((((LIM3 * GS4) * SASPR) + (((LIM4 * DIA0) * GS3) * SASR3)) + (((LIM5 * DIA2) * GS2) * SASR1)) + ((((LIM6 * DIC0) * GS3) * SASR4) * SASR4)) + (((LIM7 * DIF0) * GS2) * SASR4)) + (LIM8 * DIAC));
                VCLIS = (((((NCLI1 * EIS) * VTVIS) * NS1D) * (((((GIF1 / DI0Z1) * GSM1) + (((SASR1 * GIG1) * GSM2) * DI0Z1)) + (((SASR1 * GIH1) * GSM2) / DI0Z2)) + (SASR2 * GSM3))) / NTU_V2M3);
                SASP2 = ntu_fmin(1.0f, (((1.0f + ISEPS) * (DSS0 + (LIM2 * DIA0))) / ((1.0f + ISEPL) * (DSL0 + (LIM2 * DIC0)))));
                DSLSF = ntu_fmax(DSLSF, (ntu_fmax(DSLSV, GS4) * SASP2));
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                DSLSV = ((((((LIMA * GS4) + (((LIMB * GS3) * SASR2) * DIC0)) + (((LIMC * DIA0) * GS3) * SASR3)) + (((LIMD * DIF0) * GS2) * SASR1)) + (((LIME * DIA2) * GS2) * SASR4)) + (LIMF * DIAC));
                DSLSF = (((((((LIM3 * GS4) * SASPR) + (((LIM4 * DIC0) * GS3) * SASR3)) + (((LIM5 * DIC2) * GS2) * SASR1)) + ((((LIM6 * DIA0) * GS3) * SASR4) * SASR4)) + (((LIM7 * DIF0) * GS2) * SASR4)) + (LIM8 * DICA));
                VCLIS = (((((NCLI1 * EIS) * VTVIS) * NS1D) * ((((GI2G1 * DI0Z2) * GSM1) + ((((SASR1 * 2.0f) * GIG1) * GSM2) * DI0Z1)) + (SASR2 * GSM3))) / NTU_V2M3);
                SASP2 = ntu_fmin(1.0f, (((1.0f + ISEPS) * (DSS0 + (LIM2 * DIC0))) / ((1.0f + ISEPL) * (DSL0 + (LIM2 * DIA0)))));
                DSLSF = ntu_fmax(DSLSF, (ntu_fmax(DSLSV, GS4) * SASP2));
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                DSLSV = ((((((LIMA * GS4) + (((LIMB * GS3) * SASR2) * GI2)) + (((LIMC * GI2) * GS3) * SASR3)) + (((LIMD * GI3) * GS2) * SASR1)) + (((LIME * GI3) * GS2) * SASR4)) + (LIMF * GI4));
                DSLSF = (((((((LIM3 * GS4) * SASPR) + (((LIM4 * GI2) * GS3) * SASR3)) + (((LIM5 * GI3) * GS2) * SASR1)) + ((((LIM6 * GI2) * GS3) * SASR4) * SASR4)) + (((LIM7 * GI3) * GS2) * SASR4)) + (LIM8 * GI4));
                VCLIS = (((((NCLI1 * EIS) * VTVIS) * NS1D) * (((GI3 * GSM1) + (((SASR1 * 2.0f) * GI2) * GSM2)) + (SASR2 * GSM3))) / NTU_V2M3);
                SASP2 = ntu_fmin(1.0f, (((1.0f + ISEPS) * (DSS0 + (LIM2 * GI2))) / ((1.0f + ISEPL) * (DSL0 + (LIM2 * GI2)))));
                DSLSF = ntu_fmax(DSLSF, (ntu_fmax(DSLSV, GS4) * SASP2));
            }
            DNSAC = ((((AMI * GIM1) * NTU_V2M3) + (RHOS * GSM1)) / ntu_fmax(DSLSV, GS4));
            DNSAC = ntu_fmin(DNSAC, RHOIS);
            VSCLIS = ((NCLIS * ntu_fmax((DSLSV - GS4), 0.0f)) / NTU_V2M3);
            FSCLIS = (NCLIS * (DSLSF - (GS4 * SASPR)));
        }
        if (((QG1D >= NTU_QSMALL) && (QI1D >= NTU_QSMALL))) {
            EIG = 0.0f;
            RHOIG = ((QI1D + QG1D) / (((QI1D / RHOI) + (QG1D / RHOG)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOIG / RHOG), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDI, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD) * RATIO), MVDG);
            VTQ0 = (VTQG - VTQI);
            VTN0 = (VTNG - VTNI);
            VTV0 = (VTVG - VTVI);
            VTF0 = (VTVG - VTFI);
            VTQIG = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQI) * VTQG)));
            VTNIG = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNI) * VTNG)));
            VTVIG = sqrtf(((VTV0 * VTV0) + ((0.04f * VTVI) * VTVG)));
            VTFIG = sqrtf(((VTF0 * VTF0) + ((0.04f * VTFI) * VTVG)));
            if (((AG1D >= NTU_ASMALL) || (AI1D >= NTU_ASMALL))) {
                VTAX = (VTAG - VTAI);
                VTAIG = sqrtf(((VTAX * VTAX) + ((0.04f * VTAI) * VTAG)));
            }
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                QCLIG = ((((QCLI1 * EIG) * VTQIG) * NG1D) * ((((GIMF1 / DI0Z1) + ((GIMG1 * GG2) * DI0Z1)) + ((GIMH1 * GG2) / DI0Z2)) + (GIM1 * GG3)));
                NCLIG = ((((NCLI1 * EIG) * VTNIG) * NG1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GG2) * DI0Z1)) + ((GIH1 * GG2) / DI0Z2)) + GG3));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIG = ((((ACLI1 * EIG) * VTAIG) * NG1D) * ((((GIF3 / DI0Z1) + ((GIG3 * GG2) * DI0Z1)) + ((GIH3 * GG2) / DI0Z2)) + (GI3 * GG3)));
                }
                if ((AG1D >= NTU_ASMALL)) {
                    AGCLIG = ((((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NCLI1) * EIG) * VTAIG) * NG1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GG2) * DI0Z1)) + ((GIH1 * GG2) / DI0Z2)) + GG3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIG = ((((FCLI1 * EIG) * VTFIG) * NG1D) * ((((GIZMF1 / DI0Z1) + ((GIZMG1 * GG2) * DI0Z1)) + ((GIZMH1 * GG2) / DI0Z2)) + (GIZM1 * GG3)));
                    ICLIG = ((((NCLI1 * EIG) * VTVIG) * NG1D) * ((((GIMF1 / DI0Z1) + ((GIMG1 * GG2) * DI0Z1)) + ((GIMH1 * GG2) / DI0Z2)) + (GIM1 * GG3)));
                }
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                QCLIG = ((((QCLI1 * EIG) * VTQIG) * NG1D) * (((GIM2G1 * DI0Z2) + (((2.0f * GIMG1) * GG2) * DI0Z1)) + (GIM1 * GG3)));
                NCLIG = ((((NCLI1 * EIG) * VTNIG) * NG1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GG2) * DI0Z1)) + GG3));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIG = ((((ACLI1 * EIG) * VTAIG) * NG1D) * (((GI2G3 * DI0Z2) + (((2.0f * GIG3) * GG2) * DI0Z1)) + (GI3 * GG3)));
                }
                if ((AG1D >= NTU_ASMALL)) {
                    AGCLIG = ((((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NCLI1) * EIG) * VTAIG) * NG1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GG2) * DI0Z1)) + GG3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIG = ((((FCLI1 * EIG) * VTFIG) * NG1D) * (((GIZM2G1 * DI0Z2) + (((2.0f * GIZMG1) * GG2) * DI0Z1)) + (GIZM1 * GG3)));
                    ICLIG = ((((NCLI1 * EIG) * VTVIG) * NG1D) * (((GIM2G1 * DI0Z2) + (((2.0f * GIMG1) * GG2) * DI0Z1)) + (GIM1 * GG3)));
                }
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                QCLIG = ((((QCLI1 * EIG) * VTQIG) * NG1D) * ((GIM3 + ((2.0f * GIM2) * GG2)) + (GIM1 * GG3)));
                NCLIG = ((((NCLI1 * EIG) * VTNIG) * NG1D) * ((GI3 + ((2.0f * GI2) * GG2)) + GG3));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIG = ((((ACLI1 * EIG) * VTAIG) * NG1D) * ((GI5 + ((2.0f * GI4) * GG2)) + (GI3 * GG3)));
                }
                if ((AG1D >= NTU_ASMALL)) {
                    AGCLIG = ((((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NCLI1) * EIG) * VTAIG) * NG1D) * ((GI3 + ((2.0f * GI2) * GG2)) + GG3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIG = ((((NCLI1 * EIG) * VTFIG) * NG1D) * ((GIM3 + ((2.0f * GIM2) * GG2)) + (GIM1 * GG3)));
                    ICLIG = ((((NCLI1 * EIG) * VTVIG) * NG1D) * ((GIM3 + ((2.0f * GIM2) * GG2)) + (GIM1 * GG3)));
                }
            }
            QCLIG = ntu_fmin(QCLIG, (QI1D * IDT));
            NCLIG = ntu_fmin(NCLIG, (NI1D * IDT));
            AICLIG = ntu_fmin(AICLIG, (AI1D * IDT));
            AGCLIG = ntu_fmin(AGCLIG, (AI1D * IDT));
            VICLIG = ntu_fmin((QCLIG * IRHOI), (VI1D * IDT));
            FCLIG = ntu_fmin(FCLIG, (FI1D * IDT));
            ICLIG = ntu_fmin(ICLIG, (I3M1D * IDT));
            LIM1 = (0.7f * (1.0f - (RHOI / NTU_RHOI0)));
            LIM2 = sqrtf((1.0f - (LIM1 * LIM1)));
            LIMA = ((((((1.0f + (2.0f * ISEPL)) + ntu_sq(ISEPL)) + ISEPS) + ((2.0f * ISEPL) * ISEPS)) + (ntu_sq(ISEPL) * ISEPS)) / 8.0f);
            LIMB = (LIMA * LIM2);
            LIMC = ((2.0f * LIMA) * LIM2);
            LIMD = ((2.0f * LIMA) * ntu_sq(LIM2));
            LIME = (LIMA * ntu_sq(LIM2));
            LIMF = (LIMA * NTU_POWF(LIM2, 3.0f));
            DICC = (NTU_POWF(NTU_DI0, (-(6.0f * ZETA))) * GI3H1);
            DICA = (GI2HG1 / DI0Z3);
            DIAC = GIH2G1;
            DIAA = (DI0Z3 * GI3G1);
            DIA0 = (DI0Z1 * GIG1);
            DIC0 = (GIH1 / DI0Z2);
            DIF0 = (GIF1 / DI0Z1);
            DIC2 = (GI2H1 / DI0Z4);
            DIA2 = (DI0Z2 * GI2G1);
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                DGLSV = ((((((LIMA * GG4) + ((LIMB * DIA0) * GG3)) + ((LIMC * DIC0) * GG3)) + ((LIMD * DIF0) * GG2)) + ((LIME * DIC2) * GG2)) + (LIMF * DICA));
                VCLIG = (((((NCLI1 * EIG) * VTVIG) * NG1D) * (((((GIF1 / DI0Z1) * GGM1) + ((GIG1 * GGM2) * DI0Z1)) + ((GIH1 * GGM2) / DI0Z2)) + GGM3)) / NTU_V2M3);
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                DGLSV = ((((((LIMA * GG4) + ((LIMB * DIC0) * GG3)) + ((LIMC * DIA0) * GG3)) + ((LIMD * DIF0) * GG2)) + ((LIME * DIA2) * GG2)) + (LIMF * DIAC));
                VCLIG = (((((NCLI1 * EIG) * VTVIG) * NG1D) * ((((GI2G1 * DI0Z2) * GGM1) + (((2.0f * GIG1) * GGM2) * DI0Z1)) + GGM3)) / NTU_V2M3);
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                DGLSV = ((((((LIMA * GG4) + ((LIMB * GI2) * GG3)) + ((LIMC * GI2) * GG3)) + ((LIMD * GI3) * GG2)) + ((LIME * GI3) * GG2)) + (LIMF * GI4));
                VCLIG = (((((NCLI1 * EIG) * VTVIG) * NG1D) * (((GI3 * GGM1) + ((2.0f * GI2) * GGM2)) + GGM3)) / NTU_V2M3);
            }
            DNGAC = ((((AMI * GIM1) * NTU_V2M3) + (RHOG * GG4)) / ntu_fmax(DGLSV, GG4));
            DNGAC = ntu_fmax(ntu_fmin(DNGAC, RHOIG), NTU_RHOIMIN);
            VGCLIG = ntu_fmax(0.0f, ((QCLIG / DNGAC) + (((RHOG / DNGAC) - 1.0f) * VCLIG)));
        }
        if (((QH1D >= NTU_QSMALL) && (QI1D >= NTU_QSMALL))) {
            RHOIH = ((QI1D + QH1D) / (((QI1D / RHOI) + (QH1D / NTU_RHOH)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOIH / NTU_RHOH), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDI, 3.0f) + NTU_POWF(MVDH, 3.0f)), NTU_THRD) * RATIO), MVDH);
            VTQ0 = (VTQH - VTQI);
            VTN0 = (VTNH - VTNI);
            VTV0 = (VTQH - VTVI);
            VTF0 = (VTQH - VTFI);
            VTQIH = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQI) * VTQH)));
            VTNIH = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNI) * VTNH)));
            VTVIH = sqrtf(((VTV0 * VTV0) + ((0.04f * VTVI) * VTQH)));
            VTFIH = sqrtf(((VTF0 * VTF0) + ((0.04f * VTFI) * VTQH)));
            if (((AH1D >= NTU_ASMALL) || (AI1D >= NTU_ASMALL))) {
                VTAX = (VTAH - VTAI);
                VTAIH = sqrtf(((VTAX * VTAX) + ((0.04f * VTAI) * VTAH)));
            }
            if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
                QCLIH = ((((QCLI1 * EIH) * VTQIH) * NH1D) * ((((GIMF1 / DI0Z1) + ((GIMG1 * GH2) * DI0Z1)) + ((GIMH1 * GH2) / DI0Z2)) + (GIM1 * GH3)));
                NCLIH = ((((NCLI1 * EIH) * VTNIH) * NH1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GH2) * DI0Z1)) + ((GIH1 * GH2) / DI0Z2)) + GH3));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIH = ((((ACLI1 * EIH) * VTAIH) * NH1D) * ((((GIF3 / DI0Z1) + ((GIG3 * GH2) * DI0Z1)) + ((GIH3 * GH2) / DI0Z2)) + (GI3 * GH3)));
                }
                if ((AH1D >= NTU_ASMALL)) {
                    AHCLIH = ((((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NCLI1) * EIH) * VTAIH) * NH1D) * ((((GIF1 / DI0Z1) + ((GIG1 * GH2) * DI0Z1)) + ((GIH1 * GH2) / DI0Z2)) + GH3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIH = ((((FCLI1 * EIH) * VTFIH) * NH1D) * ((((GIZMF1 / DI0Z1) + ((GIZMG1 * GH2) * DI0Z1)) + ((GIZMH1 * GH2) / DI0Z2)) + (GIZM1 * GH3)));
                    ICLIH = ((((NCLI1 * EIH) * VTVIH) * NH1D) * ((((GIMF1 / DI0Z1) + ((GIMG1 * GH2) * DI0Z1)) + ((GIMH1 * GH2) / DI0Z2)) + (GIM1 * GH3)));
                }
            } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
                QCLIH = ((((QCLI1 * EIH) * VTQIH) * NH1D) * (((GIM2G1 * DI0Z2) + (((2.0f * GIMG1) * GH2) * DI0Z1)) + (GIM1 * GH3)));
                NCLIH = ((((NCLI1 * EIH) * VTNIH) * NH1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GH2) * DI0Z1)) + GH3));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIH = ((((ACLI1 * EIH) * VTAIH) * NH1D) * (((GI2G3 * DI0Z2) + (((2.0f * GIG3) * GH2) * DI0Z1)) + (GI3 * GH3)));
                }
                if ((AH1D >= NTU_ASMALL)) {
                    AHCLIH = ((((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NCLI1) * EIH) * VTAIH) * NH1D) * (((GI2G1 * DI0Z2) + (((2.0f * GIG1) * GH2) * DI0Z1)) + GH3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIH = ((((FCLI1 * EIH) * VTFIH) * NH1D) * (((GIZM2G1 * DI0Z2) + (((2.0f * GIZMG1) * GH2) * DI0Z1)) + (GIZM1 * GH3)));
                    ICLIH = ((((NCLI1 * EIH) * VTVIH) * NH1D) * (((GIM2G1 * DI0Z2) + (((2.0f * GIMG1) * GH2) * DI0Z1)) + (GIM1 * GH3)));
                }
            } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
                QCLIH = ((((QCLI1 * EIH) * VTQIH) * NH1D) * ((GIM3 + ((2.0f * GIM2) * GH2)) + (GIM1 * GH3)));
                NCLIH = ((((NCLI1 * EIH) * VTNIH) * NH1D) * ((GI3 + ((2.0f * GI2) * GH2)) + GH3));
                if ((AI1D >= NTU_ASMALL)) {
                    AICLIH = ((((ACLI1 * EIH) * VTAIH) * NH1D) * ((GI5 + ((2.0f * GI4) * GH2)) + (GI3 * GH3)));
                }
                if ((AH1D >= NTU_ASMALL)) {
                    AHCLIH = ((((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NCLI1) * EIH) * VTAIH) * NH1D) * ((GI3 + ((2.0f * GI2) * GH2)) + GH3));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FCLIH = ((((NCLI1 * EIH) * VTFIH) * NH1D) * ((GIM3 + ((2.0f * GIM2) * GH2)) + (GIM1 * GH3)));
                    ICLIH = ((((NCLI1 * EIH) * VTVIH) * NH1D) * ((GIM3 + ((2.0f * GIM2) * GH2)) + (GIM1 * GH3)));
                }
            }
            QCLIH = ntu_fmin(QCLIH, (QI1D * IDT));
            NCLIH = ntu_fmin(NCLIH, (NI1D * IDT));
            VCLIH = ntu_fmin((QCLIH * IRHOI), (VI1D * IDT));
            AICLIH = ntu_fmin(AICLIH, (AI1D * IDT));
            AHCLIH = ntu_fmin(AHCLIH, (AI1D * IDT));
            FCLIH = ntu_fmin(FCLIH, (FI1D * IDT));
            ICLIH = ntu_fmin(ICLIH, (I3M1D * IDT));
        }
        if (((QS1D >= NTU_QSMALL) && (QG1D >= NTU_QSMALL))) {
            ESG = 0.0f;
            RHOSG = ((QS1D + QG1D) / (((QS1D / RHOS) + (QG1D / RHOG)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOSG / RHOG), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDS, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD) * RATIO), MVDG);
            VTQ0 = (VTQG - VTQS);
            VTN0 = (VTNG - VTNS);
            VTV0 = (VTVG - VTVS);
            VTQSG = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQS) * VTQG)));
            VTNSG = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNS) * VTNG)));
            VTVSG = sqrtf(((VTV0 * VTV0) + ((0.04f * VTVS) * VTVG)));
            if (((AS1D >= NTU_ASMALL) || (AG1D >= NTU_ASMALL))) {
                VTAX = (VTAG - VTAS);
                VTASG = sqrtf(((VTAX * VTAX) + ((0.04f * VTAS) * VTAG)));
            }
            QCLSG = ((((QCLS1 * ESG) * VTQSG) * NG1D) * (((SASR2 * GSM3) + (((SASR1 * 2.0f) * GG2) * GSM2)) + (GSM1 * GG3)));
            NCLSG = ((((NCLS1 * ESG) * VTNSG) * NG1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GS2) * GG2)) + GG3));
            QCLSG = ntu_fmin(QCLSG, (QS1D * IDT));
            NCLSG = ntu_fmin(NCLSG, (NS1D * IDT));
            VSCLSG = ntu_fmin((QCLSG * IRHOS), (VS1D * IDT));
            LIM1 = (0.7f * (1.0f - (RHOS / NTU_RHOI0)));
            LIM2 = sqrtf((1.0f - (LIM1 * LIM1)));
            LIMA = ((((((1.0f + (2.0f * ISEPL)) + ntu_sq(ISEPL)) + ISEPS) + ((2.0f * ISEPL) * ISEPS)) + (ntu_sq(ISEPL) * ISEPS)) / 8.0f);
            LIMB = (LIMA * LIM2);
            LIMC = ((2.0f * LIMA) * LIM2);
            LIMD = ((2.0f * LIMA) * ntu_sq(LIM2));
            LIME = (LIMA * ntu_sq(LIM2));
            LIMF = (LIMA * NTU_POWF(LIM2, 3.0f));
            DGLSV = ((((LIMA * GG4) + (((LIMB + LIMC) * GG3) * GS2)) + (((LIMD + LIME) * GG2) * GS3)) + (LIMF * GS4));
            DNGAC = ((((AMS * GSM1) + (AMG * GGM1)) * NTU_V2M3) / ntu_fmax(DGLSV, GG4));
            DNGAC = ntu_fmax(ntu_fmin(DNGAC, RHOSG), NTU_RHOIMIN);
            VCLSG = ((((NCLS1 * ESG) * VTVSG) * NG1D) * ((((SASR2 * GS3) * GGM1) + (((SASR1 * 2.0f) * GS2) * GGM2)) + GGM3));
            VGCLSG = ntu_fmax(0.0f, ((QCLSG / DNGAC) + (((RHOG / DNGAC) - 1.0f) * VCLSG)));
            FCLSG = ntu_fmin((((QCLSG * IRHOS) * SASPR) * NTU_V2M3), (FS1D * IDT));
            if ((AS1D >= NTU_ASMALL)) {
                ASCLSG = ((((ACLS1 * ESG) * VTASG) * NG1D) * (((SASR2 * GS5) + (((SASR1 * 2.0f) * GG2) * GS4)) + (GG3 * GS3)));
                ASCLSG = ntu_fmin(ASCLSG, (AS1D * IDT));
            }
            if ((AG1D >= NTU_ASMALL)) {
                AGCLSG = ((((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NCLS1) * ESG) * VTASG) * NG1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GS2) * GG2)) + GG3));
            }
        }
        if (((QH1D >= NTU_QSMALL) && (QS1D >= NTU_QSMALL))) {
            RHOSH = ((QS1D + QH1D) / (((QS1D / RHOS) + (QH1D / NTU_RHOH)) + NTU_ISMALL));
            RATIO = NTU_POWF((RHOSH / NTU_RHOH), NTU_THRD);
            MVDX = ntu_fmax((NTU_POWF((NTU_POWF(MVDS, 3.0f) + NTU_POWF(MVDH, 3.0f)), NTU_THRD) * RATIO), MVDH);
            VTQ0 = (VTQH - VTQS);
            VTN0 = (VTNH - VTNS);
            VTQSH = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQS) * VTQH)));
            VTNSH = sqrtf(((VTN0 * VTN0) + ((0.04f * VTNS) * VTNH)));
            if (((AH1D >= NTU_ASMALL) || (AS1D >= NTU_ASMALL))) {
                VTAX = (VTAH - VTAS);
                VTASH = sqrtf(((VTAX * VTAX) + ((0.04f * VTAS) * VTAH)));
            }
            QCLSH = ((((QCLS1 * ESH) * VTQSH) * NH1D) * (((SASR2 * GSM3) + (((SASR1 * 2.0f) * GH2) * GSM2)) + (GH3 * GSM1)));
            NCLSH = ((((NCLS1 * ESH) * VTNSH) * NH1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GH2) * GS2)) + GH3));
            QCLSH = ntu_fmin(QCLSH, (QS1D * IDT));
            NCLSH = ntu_fmin(NCLSH, (NS1D * IDT));
            VCLSH = ntu_fmin((QCLSH * IRHOS), (VS1D * IDT));
            FCLSH = ntu_fmin((((QCLSH * IRHOS) * SASPR) * NTU_V2M3), (FS1D * IDT));
            if ((AS1D >= NTU_ASMALL)) {
                ASCLSH = ((((ACLS1 * ESH) * VTASH) * NH1D) * (((SASR2 * GS5) + (((SASR1 * 2.0f) * GH2) * GS4)) + (GH3 * GS3)));
                ASCLSH = ntu_fmin(ASCLSH, (AS1D * IDT));
            }
            if ((AH1D >= NTU_ASMALL)) {
                AHCLSH = ((((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NCLS1) * ESH) * VTASH) * NH1D) * (((SASR2 * GS3) + (((SASR1 * 2.0f) * GH2) * GS2)) + GH3));
            }
        }
        if (((QS1D >= NTU_QSMALL) && (MVDS >= 2.0E-4f))) {
            ESS1 = (0.1f * ntu_fmin(1.0f, (0.05f * NTU_EXPF((0.1f * TC1D)))));
            ESS2 = (1.0f - (RHOS / NTU_RHOI0));
            ESS = ESS1;
            QCLSS = (((((((((2.0f * NTU_PI) * XISP) * VTQS) * ESS) * SASR2) / 3.0f) * ntu_sq(NS1D)) * GSM3) * AMS);
            NCLSS = (-((((((((2.0f * NTU_PI) * XISP) * VTNS) * ESS) * SASR2) / 3.0f) * ntu_sq(NS1D)) * GS3));
            LIM1 = (0.7f * (1.0f - (RHOS / NTU_RHOI0)));
            LIM2 = sqrtf((1.0f - (LIM1 * LIM1)));
            LIM3 = ((((((1.0f + (2.0f * ISEPS)) + ntu_sq(ISEPS)) + ISEPL) + ((2.0f * ISEPS) * ISEPL)) + (ntu_sq(ISEPS) * ISEPL)) / 8.0f);
            LIM4 = ((2.0f * LIM3) * LIM2);
            LIM5 = (LIM3 * ntu_sq(LIM2));
            LIM6 = (LIM3 * LIM2);
            LIM7 = ((2.0f * LIM3) * ntu_sq(LIM2));
            LIM8 = (LIM3 * NTU_POWF(LIM2, 3.0f));
            LIMA = ((((((1.0f + (2.0f * ISEPL)) + ntu_sq(ISEPL)) + ISEPS) + ((2.0f * ISEPL) * ISEPS)) + (ntu_sq(ISEPL) * ISEPS)) / 8.0f);
            LIMB = (LIMA * LIM2);
            LIMC = ((2.0f * LIMA) * LIM2);
            LIMD = ((2.0f * LIMA) * ntu_sq(LIM2));
            LIME = (LIMA * ntu_sq(LIM2));
            LIMF = (LIMA * NTU_POWF(LIM2, 3.0f));
            DSLSV = ((((((LIMA + LIMB) + LIMC) + LIMD) + LIME) + LIMF) * GS4);
            DSLSF = (((((((LIM3 + LIM4) + LIM5) + LIM6) + LIM7) + LIM8) * GS4) * SASPR);
            SASP3 = (ntu_fmin(((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)) * SASR4), ((((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2)) * SASR1)) / ntu_fmax(((((1.0f + LIM2) + ISEPS) + (ISEPS * LIM2)) * SASR4), ((((1.0f + LIM2) + ISEPL) + (ISEPL * LIM2)) * SASR1)));
            DNSAG = ntu_fmax(NTU_RHOIMIN, (((2.0f * RHOS) * GSM1) / DSLSV));
            VCLSS = ((2.0f * QCLSS) * ntu_fmax(((1.0f / DNSAG) - IRHOS), 0.0f));
            FCLSS = (((2.0f * QCLSS) * NTU_V2M3) * ntu_fmax(((SASP3 / DNSAG) - (SASPR * IRHOS)), 0.0f));
            RATIO = NTU_POWF((RHOS / DNSAG), NTU_THRD);
            if ((AS1D >= NTU_ASMALL)) {
                ACLSS = (((((((((2.0f * NTU_PI) * XISP) * VTAS) * ESS) * SASR2) / 3.0f) * ntu_sq(NS1D)) * GS5) * ((1.5874f * ntu_sq(RATIO)) - 2.0f));
                if ((ACLSS < 0.0f)) {
                    ACLSS1 = ntu_fmin(ACLSS, 0.0f);
                    ACLSS1 = ntu_fmax(ACLSS1, (-((1.0f * AS1D) * IDT)));
                    ACLSS = 0.0f;
                }
            }
        }
    } else {
        if ((((((HCWQV + HRWQV) + HSWQV) + HGWQV) + HHWQV) >= NTU_QSMALL)) {
            VDMAX = (((XXLV * (QV1D - QVSW0)) / (1.0f + ((ntu_sq(XXLV) * QV1D) / ((CPM * NTU_RV) * ntu_sq(TK1D))))) * IDT);
            SUMCND = ((((HCWQV + HRWQV) + HSWQV) + HGWQV) + HHWQV);
            if (((SUMCND > VDMAX) && (VDMAX >= NTU_QSMALL))) {
                RATIO = ntu_fmin(1.0f, (VDMAX / (SUMCND + NTU_QSMALL)));
                HCWQV = (HCWQV * RATIO);
                HRWQV = (HRWQV * RATIO);
                HSWQV = (HSWQV * RATIO);
                HGWQV = (HGWQV * RATIO);
                HHWQV = (HHWQV * RATIO);
            }
        }
        if ((((((HCWQV + HRWQV) + HSWQV) + HGWQV) + HHWQV) < 0.0f)) {
            EVMAX = (((XXLV * (QV1D - QVSW0)) / (1.0f + ((ntu_sq(XXLV) * QV1D) / ((CPM * NTU_RV) * ntu_sq(TK1D))))) * IDT);
            SUMEVP = ((((HCWQV + HRWQV) + HSWQV) + HGWQV) + HHWQV);
            if (((EVMAX < 0.0f) && (SUMEVP < (EVMAX * 0.9999f)))) {
                HCWQV = (HCWQV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                HRWQV = (HRWQV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                HSWQV = (HSWQV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                HGWQV = (HGWQV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                HHWQV = (HHWQV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
            }
        }
        if ((QI1D >= NTU_QSMALL)) {
            if ((MVDI > NTU_DCR)) {
                QMLIR = (-((1.0f * QI1D) * IDT));
                VMLIR = (-((1.0f * VI1D) * IDT));
                if ((AI1D >= NTU_ASMALL)) {
                    AMLIR = (-((1.0f * AI1D) * IDT));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FMLIR = (-((1.0f * FI1D) * IDT));
                    IMLIR = (-((1.0f * I3M1D) * IDT));
                }
            } else {
                QMLIC = (-((1.0f * QI1D) * IDT));
                VMLIC = (-((1.0f * VI1D) * IDT));
                if ((AI1D >= NTU_ASMALL)) {
                    AMLIC = (-((1.0f * AI1D) * IDT));
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FMLIC = (-((1.0f * FI1D) * IDT));
                    IMLIC = (-((1.0f * I3M1D) * IDT));
                }
            }
            NMLIR = ntu_fmin(((QMLIR * NI1D) / QI1D), 0.0f);
            NMLIC = ntu_fmin(((QMLIC * NI1D) / QI1D), 0.0f);
        }
        if ((QS1D >= NTU_QSMALL)) {
            RMCSQ = 0.0f;
            RMRSQ = 0.0f;
            RMCSA = 0.0f;
            RMRSA = 0.0f;
            if ((QC1D >= NTU_QSMALL)) {
                ECS = 1.0f;
                VTQ0 = (VTQS - VTQC);
                VTQSC = sqrtf(((VTQ0 * VTQ0) + ((4.0E-2f * VTQS) * VTQC)));
                RMCSQ = ((((QRMC1 * ECS) * VTQSC) * NS1D) * (((GS3 * GC4) + ((2.0f * GS2) * GC5)) + GC6));
                RMCSQ = ntu_fmin(RMCSQ, (QC1D * IDT));
                if ((AS1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax(NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDS, 3.0f)), NTU_THRD), MVDS);
                    RMCSA = (((((ntu_sq(MVDX) - ntu_sq(MVDS)) * NRMC1) * VTQSC) * NS1D) * ((GS3 + ((2.0f * GS2) * GC2)) + GC3));
                    RMCSA = ntu_fmax(0.0f, ntu_fmin(RMCSA, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
                }
            }
            if ((QR1D >= NTU_QSMALL)) {
                ERS = 1.0f;
                VTQ0 = (VTQR - VTQS);
                VTQRS = sqrtf(((VTQ0 * VTQ0) + ((4.0E-2f * VTQR) * VTQS)));
                RMRSQ = ((((QRMR1 * ERS) * VTQRS) * NS1D) * (((GS3 * GR4) + ((2.0f * GS2) * GR5)) + GR6));
                RMRSQ = ntu_fmin(RMRSQ, (QR1D * IDT));
                if ((AS1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax(NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDS, 3.0f)), NTU_THRD), MVDS);
                    RMRSA = (((((ntu_sq(MVDX) - ntu_sq(MVDS)) * NRMR1) * VTQRS) * NS1D) * ((GS3 + ((2.0f * GS2) * GR2)) + GR3));
                    RMRSA = ntu_fmax(0.0f, ntu_fmin(RMRSA, (((QR1D * IDT) * NTU_IAPW) / MVDR)));
                }
            }
            MLMAX = (1.0f - ntu_fmin(1.0f, (RHOS / NTU_RHOIMAX)));
            HSWCD = (((((2.0f * NTU_PI) * NS1D) * KAP) * VENQS) * (NTU_TK0C - TK1D));
            HSWRM = ((NTU_CPW * (NTU_TK0C - TK1D)) * (RMRSQ + RMCSQ));
            SMLF = (ntu_fmax(0.0f, (-(((1.0f * ((HSWCD + HSWQV) + HSWRM)) * DT) / XXLF0))) / QS1D);
            if ((SMLF < NTU_MLIMIT)) {
                SMLF = 0.0f;
                SMLR = 0.0f;
            } else {
                SMLF = ntu_fmin(MLMAX, ntu_fmax(0.0f, SMLF));
                SMLR = (0.01195f * NTU_EXPF((4.411f * SMLF)));
            }
            AVWSG = ((SMLR * NTU_AVRH) + ((1.0f - SMLR) * NTU_AVSG));
            BVWSG = ((SMLR * NTU_BVRH) + ((1.0f - SMLR) * NTU_BVSG));
            RHOWS = ((SMLF * NTU_RHOW) + ((1.0f - SMLF) * RHOS));
            if ((SMLF >= NTU_MLIMIT)) {
                KINV = (((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((TK1D / NTU_TK0C), 1.5f)) / RHO);
                BEST0 = (((2.0f * NTU_G) * NS1D) / ntu_sq(KINV));
                BEST = ((((2.0f * BEST0) * RHOWS) * NTU_EXPF(((ntu_gamln(((NTU_BMS + AFAS) + 1.0f)) - ntu_gamln((AFAS + 1.0f))) - (NTU_BMS * NTU_LOGF(LAMS))))) / 3.0f);
                C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
                VTB1 = (((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f);
                VTA1 = ((NTU_VTC2 * ntu_sq((NTU_POWF(((float)(1) + C1X2), 5.0E-1f) - 1.0f))) / NTU_POWF(BEST, VTB1));
                AVS = ((VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1)))) * NTU_POWF((((4.0f * RHOWS) * NTU_G) / 3.0f), VTB1));
                BVS = ((VTB1 * NTU_BMS) - 1.0f);
            } else {
                AVS = AVS;
                BVS = BVS;
            }
            BSTMP = (SCN * sqrtf(((AVS * RHOAJ) / MUA)));
            QTMP1 = (LLMS * (1.5f + (BVS / 2.0f)));
            QTMP2 = NTU_EXPF(((ntu_gamln((AFAS + 2.0f)) - ntu_gamln((AFAS + 1.0f))) - LLMS));
            QTMP3 = NTU_EXPF(((ntu_gamln((((BVS / 2.0f) + AFAS) + 2.5f)) - ntu_gamln((AFAS + 1.0f))) - QTMP1));
            DSMM = (MVDS * 1.0E3f);
            WSAPR = ((SMLF * ((((0.9951f + (2.51E-2f * DSMM)) - (3.644E-2f * ntu_sq(DSMM))) + (5.303E-3f * NTU_POWF(DSMM, 3.0f))) - (2.492E-4f * NTU_POWF(DSMM, 4.0f)))) + ((1.0f - SMLF) * SASPR));
            CAPWS = ((ZP1 * NTU_POWF(WSAPR, (ZP2 / 3.0f))) + (ZP3 * NTU_POWF(WSAPR, (ZP4 / 3.0f))));
            VENWS = (((AVWSG * QTMP2) * CAPWS) + (((BVWSG * BSTMP) * QTMP3) * CAPWS));
            SMLTQ = (((NTU_CPW / XXLF) * (NTU_TK0C - TK1D)) * (RMCSQ + RMRSQ));
            QMLSR = (((((((2.0f * NTU_PI) * NS1D) * KAP) * (NTU_TK0C - TK1D)) / XXLF) * VENWS) + SMLTQ);
            QMLSR = ntu_fmax(ntu_fmin(QMLSR, 0.0f), (-((1.0f * QS1D) * IDT)));
            NMLSR = ntu_fmin(((QMLSR * NS1D) / QS1D), 0.0f);
            VMLSR = ((((QS1D + (QMLSR * DT)) / RHOWS) - VS1D) * IDT);
            VMLSR = ntu_fmax(VMLSR, (-((1.0f * VS1D) * IDT)));
            FMLSR = ntu_fmax((((-(1.0f * FS1D)) + (WSAPR * (((QS1D / RHOWS) * NTU_V2M3) + (((QMLSR / RHOWS) * NTU_V2M3) * DT)))) * IDT), (-((1.0f * FS1D) * IDT)));
            if ((AS1D >= NTU_ASMALL)) {
                ATMP1 = (NTU_LOGF(LAMS) * (0.5f + (BVS / 2.0f)));
                ATMP2 = NTU_EXPF(((ntu_gamln((((BVS / 2.0f) + AFAS) + 1.5f)) - ntu_gamln((AFAS + 1.0f))) - ATMP1));
                VENAS = ((AVWSG * CAPWS) + (((BVWSG * BSTMP) * ATMP2) * CAPWS));
                SMLTA = (((NTU_CPW / XXLF) * (NTU_TK0C - TK1D)) * (RMCSA + RMRSA));
                AMLSR = (((((((8.0f * NS1D) * KAP) * (NTU_TK0C - TK1D)) / XXLF) / RHOWS) * VENAS) + SMLTA);
                AMLSR = ntu_fmax(ntu_fmin(AMLSR, 0.0f), (-((1.0f * AS1D) * IDT)));
            }
        }
        if ((QG1D >= NTU_QSMALL)) {
            RMCGQ = 0.0f;
            RMRGQ = 0.0f;
            RMCGA = 0.0f;
            RMRGA = 0.0f;
            if ((QC1D >= NTU_QSMALL)) {
                ECG = 1.0f;
                VTQ0 = (VTQG - VTQC);
                VTQGC = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQG) * VTQC)));
                RMCGQ = ((((QRMC1 * ECG) * VTQGC) * NG1D) * (((GG3 * GC4) + ((2.0f * GG2) * GC5)) + GC6));
                RMCGQ = ntu_fmin(RMCGQ, (QC1D * IDT));
                if ((AG1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax(NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD), MVDG);
                    RMCGA = (((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NRMC1) * VTQGC) * NG1D) * ((GG3 + ((2.0f * GG2) * GR2)) + GR3));
                    RMCGA = ntu_fmax(0.0f, ntu_fmin(RMCGA, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
                }
            }
            if ((QR1D >= NTU_QSMALL)) {
                ERG = 1.0f;
                VTQ0 = (VTQR - VTQG);
                VTQRG = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQR) * VTQG)));
                RMRGQ = ((((QRMR1 * ERG) * VTQRG) * NG1D) * (((GG3 * GR4) + ((2.0f * GG2) * GR5)) + GR6));
                RMRGQ = ntu_fmin(RMRGQ, (QR1D * IDT));
                if ((AG1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax(NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDG, 3.0f)), NTU_THRD), MVDG);
                    RMRGA = (((((ntu_sq(MVDX) - ntu_sq(MVDG)) * NRMR1) * VTQRG) * NG1D) * ((GG3 + ((2.0f * GG2) * GR2)) + GR3));
                    RMRGA = ntu_fmax(0.0f, ntu_fmin(RMRGA, (((QR1D * IDT) * NTU_IAPW) / MVDR)));
                }
            }
            MLMAX = (1.0f - ntu_fmin(1.0f, (RHOG / NTU_RHOIMAX)));
            HGWCD = (((((2.0f * NTU_PI) * NG1D) * KAP) * VENQG) * (NTU_TK0C - TK1D));
            HGWRM = ((NTU_CPW * (NTU_TK0C - TK1D)) * (RMRGQ + RMCGQ));
            GMLF = (ntu_fmax(0.0f, (-(((1.0f * ((HGWCD + HGWQV) + HGWRM)) * DT) / XXLF0))) / QG1D);
            if ((GMLF < NTU_MLIMIT)) {
                GMLF = 0.0f;
                GMLR = 0.0f;
            } else {
                GMLF = ntu_fmin(MLMAX, ntu_fmax(0.0f, GMLF));
                GMLR = (0.01195f * NTU_EXPF((4.411f * GMLF)));
            }
            AVWSG = ((GMLR * NTU_AVRH) + ((1.0f - GMLR) * NTU_AVSG));
            BVWSG = ((GMLR * NTU_BVRH) + ((1.0f - GMLR) * NTU_BVSG));
            RHOWG = ((GMLF * NTU_RHOW) + ((1.0f - GMLF) * RHOG));
            KINV = (((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((TK1D / NTU_TK0C), 1.5f)) / RHO);
            BEST0 = (((2.0f * NTU_G) * NG1D) / ntu_sq(KINV));
            BEST = ((((2.0f * BEST0) * RHOWG) * NTU_EXPF(((ntu_gamln(((NTU_BMG + AFAG) + 1.0f)) - ntu_gamln((AFAG + 1.0f))) - (NTU_BMG * NTU_LOGF(LAMG))))) / 3.0f);
            C1X2 = (NTU_VTC1 * NTU_POWF(BEST, 5.0E-1f));
            VTB1 = (((C1X2 / NTU_POWF((1.0f + C1X2), 5.0E-1f)) / (NTU_POWF((1.0f + C1X2), 5.0E-1f) - 1.0f)) / 2.0f);
            VTA1 = ((NTU_VTC2 * ntu_sq((NTU_POWF(((float)(1) + C1X2), 5.0E-1f) - 1.0f))) / NTU_POWF(BEST, VTB1));
            AVG = ((VTA1 * NTU_POWF(KINV, (1.0f - (2.0f * VTB1)))) * NTU_POWF((((4.0f * RHOWG) * NTU_G) / 3.0f), VTB1));
            BVG = ((VTB1 * NTU_BMG) - 1.0f);
            BGTMP = (SCN * sqrtf(((AVG * RHOAJ) / MUA)));
            QTMP1 = (LLMG * (1.5f + (BVG / 2.0f)));
            QTMP2 = NTU_EXPF(((ntu_gamln((AFAG + 2.0f)) - ntu_gamln((AFAG + 1.0f))) - LLMG));
            QTMP3 = NTU_EXPF(((ntu_gamln((((BVG / 2.0f) + AFAG) + 2.5f)) - ntu_gamln((AFAG + 1.0f))) - QTMP1));
            VENWG = ((AVWSG * QTMP2) + ((BVWSG * BGTMP) * QTMP3));
            GMLTQ = (((NTU_CPW / XXLF) * (NTU_TK0C - TK1D)) * (RMCGQ + RMRGQ));
            QMLGR = (((((((2.0f * NTU_PI) * NG1D) * KAP) * (NTU_TK0C - TK1D)) / XXLF) * VENWG) + GMLTQ);
            QMLGR = ntu_fmax(ntu_fmin(QMLGR, 0.0f), (-((1.0f * QG1D) * IDT)));
            NMLGR = ntu_fmin(((QMLGR * NG1D) / QG1D), 0.0f);
            VMLGR = ntu_fmax(((((QG1D + (QMLGR * DT)) / RHOWG) - VG1D) * IDT), (-((1.0f * VG1D) * IDT)));
            if ((AG1D >= NTU_ASMALL)) {
                ATMP1 = (NTU_LOGF(LAMG) * (0.5f + (BVG / 2.0f)));
                ATMP2 = NTU_EXPF(((ntu_gamln((((BVG / 2.0f) + AFAG) + 1.5f)) - ntu_gamln((AFAG + 1.0f))) - ATMP1));
                VENAG = (AVWSG + ((BVWSG * BGTMP) * ATMP2));
                GMLTA = (((NTU_CPW / XXLF) * (NTU_TK0C - TK1D)) * (RMCGA + RMRGA));
                AMLGR = (((((((8.0f * NG1D) * KAP) * (NTU_TK0C - TK1D)) / XXLF) / RHOWG) * VENAG) + GMLTA);
                AMLGR = ntu_fmax(ntu_fmin(AMLGR, 0.0f), (-((1.0f * AG1D) * IDT)));
            }
        }
        if ((QH1D >= NTU_QSMALL)) {
            RMCHQ = 0.0f;
            RMRHQ = 0.0f;
            RMCHA = 0.0f;
            RMRHA = 0.0f;
            if ((QC1D >= NTU_QSMALL)) {
                ECH = 1.0f;
                VTQ0 = (VTQH - VTQC);
                VTQHC = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQH) * VTQC)));
                RMCHQ = ((((QRMC1 * ECH) * VTQHC) * NH1D) * (((GH3 * GC4) + ((2.0f * GH2) * GC5)) + GC6));
                RMCHQ = ntu_fmin(RMCHQ, (QC1D * IDT));
                if ((AH1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax(NTU_POWF((NTU_POWF(MVDC, 3.0f) + NTU_POWF(MVDH, 3.0f)), NTU_THRD), MVDH);
                    RMCHA = (((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NRMC1) * VTQHC) * NH1D) * ((GH3 + ((2.0f * GH2) * GR2)) + GR3));
                    RMCHA = ntu_fmax(0.0f, ntu_fmin(RMCHA, (((QC1D * IDT) * NTU_IAPW) / MVDC)));
                }
            }
            if ((QR1D >= NTU_QSMALL)) {
                ERH = 1.0f;
                VTQ0 = (VTQH - VTQR);
                VTQRH = sqrtf(((VTQ0 * VTQ0) + ((0.04f * VTQH) * VTQR)));
                RMRHQ = ((((QRMR1 * ERH) * VTQRH) * NH1D) * (((GH3 * GR4) + ((2.0f * GH2) * GR5)) + GR6));
                RMRHQ = ntu_fmin(RMRHQ, (QR1D * IDT));
                if ((AH1D >= NTU_ASMALL)) {
                    MVDX = ntu_fmax(NTU_POWF((NTU_POWF(MVDR, 3.0f) + NTU_POWF(MVDH, 3.0f)), NTU_THRD), MVDH);
                    RMRHA = (((((ntu_sq(MVDX) - ntu_sq(MVDH)) * NRMR1) * VTQRH) * NH1D) * ((GH3 + ((2.0f * GH2) * GR2)) + GR3));
                    RMRHA = ntu_fmax(0.0f, ntu_fmin(RMRHA, (((QR1D * IDT) * NTU_IAPW) / MVDR)));
                }
            }
            HMLTQ = (((NTU_CPW / XXLF) * (NTU_TK0C - TK1D)) * (RMRHQ + RMCHQ));
            QMLHR = (((((((2.0f * NTU_PI) * NH1D) * KAP) * (NTU_TK0C - TK1D)) / XXLF) * VENQH) + HMLTQ);
            QMLHR = ntu_fmax(ntu_fmin(QMLHR, 0.0f), (-((1.0f * QH1D) * IDT)));
            NMLHR = ntu_fmin(((QMLHR * NH1D) / QH1D), 0.0f);
            if ((AH1D >= NTU_ASMALL)) {
                HMLTA = (((NTU_CPW / XXLF) * (NTU_TK0C - TK1D)) * (RMRHA + RMCHA));
                AMLHR = (((((((8.0f * NH1D) * KAP) * (NTU_TK0C - TK1D)) / XXLF) / NTU_RHOH) * VENAH) + HMLTA);
                AMLHR = ntu_fmax(ntu_fmin(AMLHR, 0.0f), (-((1.0f * AH1D) * IDT)));
            }
        }
    }
    DMWDT = (QRMCH + QRMRH);
    DMIDT = (QCLIH + QCLSH);
    if (((QH1D >= NTU_QLIMIT) && (DMWDT >= NTU_QSMALL))) {
        HHDCD = (((((2.0f * NTU_PI) * NH1D) * KAP) * VENQH) * (TK1D - NTU_TK0C));
        HHDRM = (DMWDT * (XXLF + (NTU_CPW * (TK1D - NTU_TK0C))));
        HHDCL = ((DMIDT * NTU_CPI) * (TK1D - NTU_TK0C));
        HHDTT = (((HHDCD + HHDQV) + HHDRM) + HHDCL);
        if ((HHDTT > 0.0f)) {
            QCLIH = (QCLIH / EIH);
            NCLIH = (NCLIH / EIH);
            VCLIH = (VCLIH / EIH);
            FCLIH = (FCLIH / EIH);
            ICLIH = (ICLIH / EIH);
            QCLSH = (QCLSH / ESH);
            NCLSH = (NCLSH / ESH);
            VCLSH = (VCLSH / ESH);
            AICLIH = (AICLIH / EIH);
            AHCLIH = (AHCLIH / EIH);
            ASCLSH = (ASCLSH / ESH);
            AHCLSH = (AHCLSH / ESH);
            MCORE = (((QH1D * IDT) + QCLIH) + QCLSH);
            HHWCD = (((((2.0f * NTU_PI) * NH1D) * KAP) * VENQH) * (NTU_TK0C - TK1D));
            HHWRM = ((NTU_CPW * (NTU_TK0C - TK1D)) * DMWDT);
            MLWM = ntu_fmax(0.0f, (-(((1.0f * ((HHWCD + HHWQV) + HHWRM)) * DT) / XXLF0)));
            HMLF = ntu_fmin(0.9f, ntu_fmax(0.0f, (MLWM / QH1D)));
            if ((MLWM > DMWDT)) {
                QHWML = ntu_fmax((-(1.0f * MCORE)), (DMWDT - MLWM));
                AHWML = ((((QHWML * 4.0f) * NTU_IRHOH) / NTU_PI) / MVDH);
            }
            MLWC = (0.268f + ((0.1389f * MCORE) * 1.0E3f));
            if (((MLWM * 1.0E3f) > MLWC)) {
                DH9 = ((MVDH * 1.0E2f) - 9.0E-1f);
                MWT = (((((QH1D * IDT) + QCLIH) + QCLSH) + DMWDT) * 1.0E3f);
                ICOR1 = (MWT / (((1.0f + (10.67f * DH9)) - (10.81f * ntu_sq(DH9))) + (10.26f * NTU_POWF(DH9, 3.0f))));
                QHWSH = ntu_fmax(0.0f, (MLWM - ntu_fmax(0.0f, (ICOR1 * 1.0E-3f))));
                NHWSH = (QHWSH / (NTU_AMW * NTU_K8));
            }
        } else {
            QHDRM = DMWDT;
            AHDRM = (ARMCH + ARMRH);
        }
    }
    QCSOUR = (QC1D + ((QBKRC - QMLIC) * DT));
    QCSINK = (((((((((((QHOCI + QNMCI) + QNCCI) + QRMCI) + QRMCS) + QRMCG) + QRMCH) + QCLCR) + QCNCR) + QIMCSI) + QIMCGI) * DT);
    if (((QCSINK > QCSOUR) && (QCSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QCSOUR / (QCSINK + NTU_QSMALL)));
        QHOCI = (QHOCI * RATIO);
        QNMCI = (QNMCI * RATIO);
        QNCCI = (QNCCI * RATIO);
        QRMCI = (QRMCI * RATIO);
        QRMCS = (QRMCS * RATIO);
        QRMCG = (QRMCG * RATIO);
        QRMCH = (QRMCH * RATIO);
        QCLCR = (QCLCR * RATIO);
        QCNCR = (QCNCR * RATIO);
        QIMCSI = (QIMCSI * RATIO);
        QIMCGI = (QIMCGI * RATIO);
    }
    NCSOUR = (NC1D + ((NBKRC - NMLIC) * DT));
    NCSINK = ((((((((((((NHOCI + NNMCI) + NNCCI) + NRMCI) + NRMCS) + NRMCG) + NRMCH) + NCLCR) + NCLCC) + NCNCR) + NIMCSI) + NIMCGI) * DT);
    if (((NCSINK > NCSOUR) && (NCSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NCSOUR / (NCSINK + NTU_NSMALL)));
        NHOCI = (NHOCI * RATIO);
        NNMCI = (NNMCI * RATIO);
        NNCCI = (NNCCI * RATIO);
        NRMCI = (NRMCI * RATIO);
        NRMCS = (NRMCS * RATIO);
        NRMCG = (NRMCG * RATIO);
        NRMCH = (NRMCH * RATIO);
        NCLCR = (NCLCR * RATIO);
        NCLCC = (NCLCC * RATIO);
        NCNCR = (NCNCR * RATIO);
        NIMCSI = (NIMCSI * RATIO);
        NIMCGI = (NIMCGI * RATIO);
    }
    QRSOUR = (QR1D + ((((((((QCLCR + QCNCR) - QMLIR) - QMLSR) - QMLGR) - QMLHR) + QHWSH) - QHWML) * DT));
    QRSINK = (((((((((QHORG + QNMRG) + QRMRI) + QRMRS) + QRMRG) + QRMRH) + QBKRC) + QIMRSI) + QIMRGI) * DT);
    if (((QRSINK > QRSOUR) && (QRSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QRSOUR / (QRSINK + NTU_QSMALL)));
        QHORG = (QHORG * RATIO);
        QNMRG = (QNMRG * RATIO);
        QRMRI = (QRMRI * RATIO);
        QRMRS = (QRMRS * RATIO);
        QRMRG = (QRMRG * RATIO);
        QRMRH = (QRMRH * RATIO);
        QBKRC = (QBKRC * RATIO);
        QIMRSI = (QIMRSI * RATIO);
        QIMRGI = (QIMRGI * RATIO);
    }
    NRSOUR = (NR1D + ((((((((NCNCR + NBKRR) - NMLIR) - NMLSR) - NMLGR) - NMLHR) + NCLCR) + NHWSH) * DT));
    NRSINK = ((((((((((NHORG + NNMRG) + NRMRI) + NRMRS) + NRMRG) + NRMRH) + NCLRR) + NBKRC) + NIMRSI) + NIMRGI) * DT);
    if (((NRSINK > NRSOUR) && (NRSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NRSOUR / (NRSINK + NTU_NSMALL)));
        NHORG = (NHORG * RATIO);
        NNMRG = (NNMRG * RATIO);
        NRMRI = (NRMRI * RATIO);
        NRMRS = (NRMRS * RATIO);
        NRMRG = (NRMRG * RATIO);
        NRMRH = (NRMRH * RATIO);
        NCLRR = (NCLRR * RATIO);
        NBKRC = (NBKRC * RATIO);
        NIMRSI = (NIMRSI * RATIO);
        NIMRGI = (NIMRGI * RATIO);
    }
    QISOUR = (QI1D + ((((((((QIMCSI + QIMCGI) + QIMRSI) + QIMRGI) + QHOCI) + QNMCI) + QNCCI) + QRMCI) * DT));
    QISINK = (((((((((-QMLIR) - QMLIC) + QCLIR) + QCLIS) + QCLIG) + QCLIH) + QINIG) + QCNIS) * DT);
    if (((QISINK > QISOUR) && (QISOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QISOUR / (QISINK + NTU_QSMALL)));
        QMLIR = (QMLIR * RATIO);
        QMLIC = (QMLIC * RATIO);
        QCLIR = (QCLIR * RATIO);
        QCLIS = (QCLIS * RATIO);
        QCLIG = (QCLIG * RATIO);
        QCLIH = (QCLIH * RATIO);
        QINIG = (QINIG * RATIO);
        QCNIS = (QCNIS * RATIO);
    }
    NISOUR = (NI1D + (((((((NIMCSI + NIMCGI) + NIMRSI) + NIMRGI) + NHOCI) + NNMCI) + NNCCI) * DT));
    NISINK = (((((((((-NMLIR) - NMLIC) + NCLIR) + NCLIS) + NCLIG) + NCLIH) + NINIG) + NICNIS) * DT);
    if (((NISINK > NISOUR) && (NISOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NISOUR / (NISINK + NTU_NSMALL)));
        NCLIS = (NCLIS * RATIO);
        NCLIG = (NCLIG * RATIO);
        NCLIH = (NCLIH * RATIO);
        NINIG = (NINIG * RATIO);
        NMLIR = (NMLIR * RATIO);
        NMLIC = (NMLIC * RATIO);
        NCLIR = (NCLIR * RATIO);
        NICNIS = (NICNIS * RATIO);
    }
    VISOUR = (VI1D + ((((((((VIMCSI + VIMCGI) + VIMRSI) + VIMRGI) + VHOCI) + VNMCI) + VNCCI) + VRMCI) * DT));
    VISINK = (((((((((-VMLIR) - VMLIC) + VCLIR) + VICLIS) + VICLIG) + VCLIH) + VIINIG) + VICNIS) * DT);
    if (((VISINK > VISOUR) && (VISOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (VISOUR / (VISINK + NTU_ISMALL)));
        VMLIR = (VMLIR * RATIO);
        VMLIC = (VMLIC * RATIO);
        VCLIR = (VCLIR * RATIO);
        VICLIS = (VICLIS * RATIO);
        VICLIG = (VICLIG * RATIO);
        VCLIH = (VCLIH * RATIO);
        VIINIG = (VIINIG * RATIO);
        VICNIS = (VICNIS * RATIO);
    }
    FISOUR = (FI1D + ((((((((FIMCSI + FIMCGI) + FIMRSI) + FIMRGI) + FHOCI) + FNMCI) + FNCCI) + FRMCI) * DT));
    FISINK = (((((((((-FMLIR) - FMLIC) + FCLIR) + FICLIS) + FCLIG) + FCLIH) + FINIG) + FICNIS) * DT);
    if (((FISINK > FISOUR) && (FISOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (FISOUR / (FISINK + NTU_ISMALL)));
        FMLIR = (FMLIR * RATIO);
        FMLIC = (FMLIC * RATIO);
        FCLIR = (FCLIR * RATIO);
        FICLIS = (FICLIS * RATIO);
        FCLIG = (FCLIG * RATIO);
        FCLIH = (FCLIH * RATIO);
        FINIG = (FINIG * RATIO);
        FICNIS = (FICNIS * RATIO);
    }
    AISOUR = (AI1D + ((((((((AIMCSI + AIMCGI) + AIMRSI) + AIMRGI) + AHOCI) + ANMCI) + ANCCI) + ARMCI) * DT));
    AISINK = ((((((((ACLIR - AMLIR) - AMLIC) + AICLIS) + AICLIG) + AICLIH) + AIINIG) + AICNIS) * DT);
    if (((AISINK > AISOUR) && (AISOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (AISOUR / (AISINK + NTU_ASMALL)));
        AMLIR = (AMLIR * RATIO);
        AMLIC = (AMLIC * RATIO);
        ACLIR = (ACLIR * RATIO);
        AICLIS = (AICLIS * RATIO);
        AICLIG = (AICLIG * RATIO);
        AICLIH = (AICLIH * RATIO);
        AIINIG = (AIINIG * RATIO);
        AICNIS = (AICNIS * RATIO);
    }
    IISOUR = (I3M1D + ((((((((IIMCSI + IIMCGI) + IIMRSI) + IIMRGI) + IHOCI) + INMCI) + INCCI) + IRMCI) * DT));
    IISINK = (((((((((-IMLIR) - IMLIC) + ICLIR) + ICLIS) + ICLIG) + ICLIH) + IINIG) + ICNIS) * DT);
    if (((IISINK > IISOUR) && (IISOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (IISOUR / (IISINK + NTU_ISMALL)));
        IMLIR = (IMLIR * RATIO);
        IMLIC = (IMLIC * RATIO);
        ICLIR = (ICLIR * RATIO);
        ICLIS = (ICLIS * RATIO);
        ICLIG = (ICLIG * RATIO);
        ICLIH = (ICLIH * RATIO);
        IINIG = (IINIG * RATIO);
        ICNIS = (ICNIS * RATIO);
    }
    QSSOUR = (QS1D + (((QCNIS + QCLIS) + QRMCS) * DT));
    QSSINK = ((((((-QMLSR) + QCLSR) + QCLSG) + QCLSH) + QINSG) * DT);
    if (((QSSINK > QSSOUR) && (QSSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QSSOUR / (QSSINK + NTU_QSMALL)));
        QMLSR = (QMLSR * RATIO);
        QCLSG = (QCLSG * RATIO);
        QCLSH = (QCLSH * RATIO);
        QINSG = (QINSG * RATIO);
        QCLSR = (QCLSR * RATIO);
    }
    NSSOUR = (NS1D + (NSCNIS * DT));
    NSSINK = (((((((-NMLSR) + NCLSR) + NCLSG) + NCLSH) - NCLSS) + NINSG) * DT);
    if (((NSSINK > NSSOUR) && (NSSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NSSOUR / (NSSINK + NTU_NSMALL)));
        NMLSR = (NMLSR * RATIO);
        NCLSG = (NCLSG * RATIO);
        NCLSH = (NCLSH * RATIO);
        NCLSS = (NCLSS * RATIO);
        NINSG = (NINSG * RATIO);
        NCLSR = (NCLSR * RATIO);
    }
    VSSOUR = (VS1D + ((((VSCNIS + VSCLIS) + VRMCS) + VCLSS) * DT));
    VSSINK = ((((((-VMLSR) + VCLSR) + VSCLSG) + VCLSH) + VSINSG) * DT);
    if (((VSSINK > VSSOUR) && (VSSOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (VSSOUR / (VSSINK + NTU_ISMALL)));
        VMLSR = (VMLSR * RATIO);
        VSCLSG = (VSCLSG * RATIO);
        VCLSH = (VCLSH * RATIO);
        VSINSG = (VSINSG * RATIO);
        VCLSR = (VCLSR * RATIO);
    }
    FSSOUR = (FS1D + ((((FSCNIS + FSCLIS) + FRMCS) + FCLSS) * DT));
    FSSINK = ((((((-FMLSR) + FCLSR) + FCLSG) + FCLSH) + FINSG) * DT);
    if (((FSSINK > FSSOUR) && (FSSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (FSSOUR / (FSSINK + NTU_QSMALL)));
        FMLSR = (FMLSR * RATIO);
        FCLSG = (FCLSG * RATIO);
        FCLSH = (FCLSH * RATIO);
        FINSG = (FINSG * RATIO);
        FCLSR = (FCLSR * RATIO);
    }
    ASSOUR = (AS1D + ((((ASCNIS + ASCLIS) + ARMCS) + ACLSS) * DT));
    ASSINK = (((((((-AMLSR) + ACLSR) + ASCLSG) + ASCLSH) + ASINSG) - ACLSS1) * DT);
    if (((ASSINK > ASSOUR) && (ASSOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (ASSOUR / (ASSINK + NTU_ASMALL)));
        AMLSR = (AMLSR * RATIO);
        ASINSG = (ASINSG * RATIO);
        ACLSR = (ACLSR * RATIO);
        ASCLSG = (ASCLSG * RATIO);
        ASCLSH = (ASCLSH * RATIO);
        ACLSS1 = (ACLSS1 * RATIO);
    }
    QGSOUR = (QG1D + ((((((((((QHORG + QNMRG) + QCLIG) + QCLSG) + QINIG) + QINSG) + QRMCG) + QCLIRG) + QCLSRG) + QCLGRG) * DT));
    QGSINK = ((((-QMLGR) + QCNGH) + QCLGR) * DT);
    if (((QGSINK > QGSOUR) && (QGSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QGSOUR / (QGSINK + NTU_QSMALL)));
        QMLGR = (QMLGR * RATIO);
        QCNGH = (QCNGH * RATIO);
        QCLGR = (QCLGR * RATIO);
    }
    NGSOUR = (NG1D + (((((((NHORG + NNMRG) + NINIG) + NINSG) + NCLIRG) + NCLSRG) + NCLGRG) * DT));
    NGSINK = ((((-NMLGR) + NGCNGH) + NCLGR) * DT);
    if (((NGSINK > NGSOUR) && (NGSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NGSOUR / (NGSINK + NTU_NSMALL)));
        NMLGR = (NMLGR * RATIO);
        NGCNGH = (NGCNGH * RATIO);
        NCLGR = (NCLGR * RATIO);
    }
    VGSOUR = (VG1D + ((((((((((VHORG + VNMRG) + VGCLIG) + VGCLSG) + VGINIG) + VGINSG) + VRMCG) + VCLIRG) + VCLSRG) + VCLGRG) * DT));
    VGSINK = ((((-VMLGR) + VCNGH) + VCLGR) * DT);
    if (((VGSINK > VGSOUR) && (VGSOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (VGSOUR / (VGSINK + NTU_ISMALL)));
        VMLGR = (VMLGR * RATIO);
        VCNGH = (VCNGH * RATIO);
        VCLGR = (VCLGR * RATIO);
    }
    AGSOUR = (AG1D + ((((((((((AHORG + ANMRG) + AGCLIG) + AGCLSG) + AGINIG) + AGINSG) + ARMCG) + ACLIRG) + ACLSRG) + ACLGRG) * DT));
    AGSINK = ((((-AMLGR) + AGCNGH) + ACLGR) * DT);
    if (((AGSINK > AGSOUR) && (AGSOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (AGSOUR / (AGSINK + NTU_ASMALL)));
        AMLGR = (AMLGR * RATIO);
        AGCNGH = (AGCNGH * RATIO);
        ACLGR = (ACLGR * RATIO);
    }
    QHSOUR = (QH1D + ((((QCNGH + QCLIH) + QCLSH) + QHDRM) * DT));
    QHSINK = (((-QMLHR) - QHWML) * DT);
    if (((QHSINK > QHSOUR) && (QHSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QHSOUR / (QHSINK + NTU_QSMALL)));
        QMLHR = (QMLHR * RATIO);
        QHWML = (QHWML * RATIO);
    }
    NHSOUR = (NH1D + (NHCNGH * DT));
    NHSINK = ((-NMLHR) * DT);
    if (((NHSINK > NHSOUR) && (NHSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NHSOUR / (NHSINK + NTU_NSMALL)));
        NMLHR = (NMLHR * RATIO);
    }
    AHSOUR = (AH1D + ((((AHCNGH + AHCLIH) + AHCLSH) + AHDRM) * DT));
    AHSINK = (((-AMLHR) - AHWML) * DT);
    if (((AHSINK > AHSOUR) && (AHSOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (AHSOUR / (AHSINK + NTU_ASMALL)));
        AMLHR = (AMLHR * RATIO);
        AHWML = (AHWML * RATIO);
    }
    QFZCI = ((QHOCI + QNMCI) + QNCCI);
    NFZCI = ((NHOCI + NNMCI) + NNCCI);
    VFZCI = ((VHOCI + VNMCI) + VNCCI);
    FFZCI = ((FHOCI + FNMCI) + FNCCI);
    AFZCI = ((AHOCI + ANMCI) + ANCCI);
    IFZCI = ((IHOCI + INMCI) + INCCI);
    QFZRG = (QHORG + QNMRG);
    NFZRG = (NHORG + NNMRG);
    VFZRG = (VHORG + VNMRG);
    AFZRG = (AHORG + ANMRG);
    QIMII = (((QIMCSI + QIMCGI) + QIMRSI) + QIMRGI);
    NIMII = (((NIMCSI + NIMCGI) + NIMRSI) + NIMRGI);
    VIMII = (((VIMCSI + VIMCGI) + VIMRSI) + VIMRGI);
    AIMII = (((AIMCSI + AIMCGI) + AIMRSI) + AIMRGI);
    FIMII = (((FIMCSI + FIMCGI) + FIMRSI) + FIMRGI);
    IIMII = (((IIMCSI + IIMCGI) + IIMRSI) + IIMRGI);
    GQCTR = (((((((((((((((QRMCS + QRMCG) + QRMCH) + QCLIR) + QCLCR) + QCLIS) + QCLIG) + QCLIH) + QCNCR) + QCNIS) + QINIG) - QBKRC) - QIMCSI) - QIMCGI) - QMLIR) * DT);
    QC1D = ntu_fmax(0.0f, (QC1D + (((((((((((QBKRC - QCLCR) - QCNCR) - QFZCI) - QMLIC) - QRMCI) - QRMCS) - QRMCG) - QRMCH) - QIMCSI) - QIMCGI) * DT)));
    QR1D = ntu_fmax(0.0f, (QR1D + ((((((((((((((((QCLCR + QCNCR) - QBKRC) - QFZRG) - QIMRSI) - QIMRGI) - QMLIR) - QMLSR) - QMLGR) - QMLHR) - QRMRI) - QRMRS) - QRMRG) - QRMRH) + QHWSH) - QHWML) * DT)));
    QI1D = ntu_fmax(0.0f, (QI1D + (((((((((((QIMII - QCNIS) - QCLIS) - QCLIG) - QCLIH) - QINIG) + QFZCI) + QMLIR) + QMLIC) + QRMCI) - QCLIR) * DT)));
    QS1D = ntu_fmax(0.0f, (QS1D + ((((((((QCNIS + QCLIS) - QCLSG) - QCLSH) - QINSG) + QMLSR) + QRMCS) - QCLSR) * DT)));
    QG1D = ntu_fmax(0.0f, (QG1D + ((((((((((((QFZRG + QCLIG) + QCLSG) + QINIG) + QINSG) + QMLGR) + QRMCG) - QCNGH) - QCLGR) + QCLIRG) + QCLSRG) + QCLGRG) * DT)));
    QH1D = ntu_fmax(0.0f, (QH1D + ((((((QMLHR + QCNGH) + QHDRM) + QCLIH) + QCLSH) + QHWML) * DT)));
    NC1D = ntu_fmax(0.0f, (NC1D + ((((((((((((NBKRC - NCLCR) - NCLCC) - NCNCR) - NFZCI) - NRMCI) - NRMCS) - NRMCG) - NRMCH) - NMLIC) - NIMCSI) - NIMCGI) * DT)));
    NR1D = ntu_fmax(0.0f, (NR1D + ((((((((((((((((NCNCR + NBKRR) - NBKRC) - NCLRR) - NIMRSI) - NIMRGI) - NFZRG) - NMLIR) - NMLSR) - NMLGR) - NMLHR) - NRMRI) - NRMRS) - NRMRG) - NRMRH) + NHWSH) * DT)));
    NI1D = ntu_fmax(0.0f, (NI1D + ((((((((((NIMII - NICNIS) - NCLIS) - NCLIG) - NCLIH) - NINIG) + NFZCI) + NMLIR) + NMLIC) - NCLIR) * DT)));
    NS1D = ntu_fmax(0.0f, (NS1D + (((((((NSCNIS - NCLSG) - NCLSH) + NCLSS) - NINSG) + NMLSR) - NCLSR) * DT)));
    NG1D = ntu_fmax(0.0f, (NG1D + (((((((((NFZRG + NINIG) + NINSG) + NMLGR) - NGCNGH) - NCLGR) + NCLIRG) + NCLSRG) + NCLGRG) * DT)));
    NH1D = ntu_fmax(0.0f, (NH1D + ((NMLHR + NHCNGH) * DT)));
    VI1D = ntu_fmax(0.0f, (VI1D + (((((((((((VIMII - VICNIS) - VICLIS) - VICLIG) - VCLIH) - VIINIG) + VFZCI) + VMLIR) + VMLIC) + VRMCI) - VCLIR) * DT)));
    VS1D = ntu_fmax(0.0f, (VS1D + (((((((((VSCNIS + VSCLIS) - VSCLSG) - VCLSH) - VSINSG) + VMLSR) + VRMCS) - VCLSR) + VCLSS) * DT)));
    VG1D = ntu_fmax(0.0f, (VG1D + ((((((((((((VFZRG + VGCLIG) + VGCLSG) + VGINIG) + VGINSG) + VMLGR) + VRMCG) - VCNGH) - VCLGR) + VCLIRG) + VCLSRG) + VCLGRG) * DT)));
    FI1D = ntu_fmax(0.0f, (FI1D + (((((((((((FIMII - FICNIS) - FICLIS) - FCLIG) - FCLIH) - FINIG) + FFZCI) + FMLIR) + FMLIC) + FRMCI) - FCLIR) * DT)));
    FS1D = ntu_fmax(0.0f, (FS1D + (((((((((FSCNIS + FSCLIS) - FCLSG) - FCLSH) - FINSG) + FMLSR) + FRMCS) - FCLSR) + FCLSS) * DT)));
    I3M1D = ntu_fmax(0.0f, (I3M1D + (((((((((((IIMII - ICNIS) - ICLIS) - ICLIG) - ICLIH) - IINIG) + IFZCI) + IMLIR) + IMLIC) + IRMCI) - ICLIR) * DT)));
    AI1D = ntu_fmax(0.0f, (AI1D + (((((((((((AIMII - AICNIS) - AICLIS) - AICLIG) - AICLIH) - AIINIG) + AFZCI) + AMLIR) + AMLIC) + ARMCI) - ACLIR) * DT)));
    AS1D = ntu_fmax(0.0f, (AS1D + ((((((((((ASCNIS + ASCLIS) - ASCLSG) - ASCLSH) - ASINSG) + AMLSR) + ARMCS) - ACLSR) + ACLSS) + ACLSS1) * DT)));
    AG1D = ntu_fmax(0.0f, (AG1D + ((((((((((((AFZRG + AGCLIG) + AGCLSG) + AGINIG) + AGINSG) + AMLGR) + ARMCG) - AGCNGH) - ACLGR) + ACLIRG) + ACLSRG) + ACLGRG) * DT)));
    AH1D = ntu_fmax(0.0f, (AH1D + ((((((AMLHR + AHCNGH) + AHDRM) + AHCLIH) + AHCLSH) + AHWML) * DT)));
    TK1D = (TK1D + ((((((((((((((((((QFZCI + QFZRG) + QMLIR) + QMLIC) + QMLSR) + QMLGR) + QMLHR) + QRMCI) + QRMCS) + QRMCG) + QRMRI) + QRMRS) + QRMRG) + QIMII) + QHDRM) + QHWML) * XXLF) / CPM) * DT));
    if (((QC1D >= NTU_QSMALL) && (NC1D >= NTU_NSMALL))) {
        MVDC = NTU_POWF(((QC1D * NTU_IAMW) / NC1D), NTU_THRD);
        if ((MVDC > NTU_DCR)) {
            QR1D = (QR1D + QC1D);
            NR1D = (NR1D + NC1D);
            QC1D = 0.0f;
            NC1D = 0.0f;
        }
    }
    if (((QR1D >= NTU_QSMALL) && (NR1D >= NTU_NSMALL))) {
        MVDR = NTU_POWF(((QR1D * NTU_IAMW) / NR1D), NTU_THRD);
        if ((MVDR < NTU_DCR)) {
            QC1D = (QC1D + QR1D);
            NC1D = (NC1D + NR1D);
            QR1D = 0.0f;
            NR1D = 0.0f;
        }
    }
    if (((QH1D >= NTU_QSMALL) && (NH1D >= NTU_NSMALL))) {
        MVDH = NTU_POWF(((QH1D * NTU_IAMH) / NH1D), NTU_THRD);
        if ((MVDH < NTU_DHMIN)) {
            QG1D = (QG1D + QH1D);
            NG1D = (NG1D + NH1D);
            VG1D = (VG1D + (QH1D / NTU_RHOG0));
            QH1D = 0.0f;
            NH1D = 0.0f;
            if (((AH1D >= NTU_ASMALL) && (NTU_AFAG_3M == 1))) {
                AG1D = (AG1D + AH1D);
                AH1D = 0.0f;
            }
        }
    }
    if ((GQCTR > 0.0f)) {
        TQCI = (QC1D + QI1D);
        if ((TQCI > (NTU_RLIMIT * 2.0f))) {
            GQCTR = ntu_fmin(1.0f, (GQCTR / TQCI));
        } else {
            GQCTR = 0.0f;
        }
    } else {
        TQRSG = (((QR1D + QS1D) + QG1D) + QH1D);
        if ((TQRSG > (NTU_RLIMIT * 3.0f))) {
            GQCTR = ntu_fmax((-1.0f), (GQCTR / TQRSG));
        } else {
            GQCTR = 0.0f;
        }
    }
}

// WRF SUBROUTINE PTFLUX (module_mp_ntu.F:3203-3236)
NTU_DEV void ntu_ptflux(float* Q1D, float* VT1D, float* RHO, float* DZ, int NK, float DT, float DTMN, float& PRT1D) {
    const int MAXSTP = 1000;
    float DQDT[NTU_KMAX] = {};
    int K = 0, NS = 0, NSTEP = 0;
    NSTEP = 1;
    { const int ntu_hi_k = NK;
    for (K = 1; K <= ntu_hi_k; ++K) {
        NSTEP = ntu_imax(NSTEP, (int)((((VT1D[(K) - (1)] * DT) / DZ[(K) - (1)]) + 1.0f)));
        if ((NSTEP > 80)) {
            /* PRINT *,'IN PREFLUX',K,NSTEP,DT,VT1D(K),DZ(K),Q1D(K),RHO(K) */
        }
    } }
    if ((NSTEP > MAXSTP)) {
        /* PRINT *,'NSTEP FOR PRECIP. IS: ',NSTEP,VT1D,DT,DZ */
        NTU_FATAL(NTU_ERR_STOP);
    }
    { const int ntu_hi_ns = NSTEP;
    for (NS = 1; NS <= ntu_hi_ns; ++NS) {
        DQDT[(1) - (1)] = (-((VT1D[(1) - (1)] * Q1D[(1) - (1)]) / DZ[(1) - (1)]));
        { const int ntu_hi_k = NK;
        for (K = 2; K <= ntu_hi_k; ++K) {
            DQDT[(K) - (1)] = ((((VT1D[((K - 1)) - (1)] * RHO[((K - 1)) - (1)]) * Q1D[((K - 1)) - (1)]) - ((VT1D[(K) - (1)] * RHO[(K) - (1)]) * Q1D[(K) - (1)])) / (DZ[(K) - (1)] * RHO[(K) - (1)]));
        } }
        PRT1D = (PRT1D + (((((VT1D[(NK) - (1)] * RHO[(NK) - (1)]) * Q1D[(NK) - (1)]) * DTMN) * 60.0f) / (float)(NSTEP)));
        { const int ntu_hi_k = NK;
        for (K = 1; K <= ntu_hi_k; ++K) {
            Q1D[(K) - (1)] = (Q1D[(K) - (1)] + ((DQDT[(K) - (1)] * DT) / (float)(NSTEP)));
        } }
    } }
}

// WRF SUBROUTINE SEDI_FALL (module_mp_ntu.F:3057-3199)
NTU_DEV_BIG void ntu_sedi_fall(NtuAeroConst* __restrict__ AC, float TK1D, float P1D, float QV1D, float QC1D, float QR1D, float QI1D, float QS1D, float QG1D, float QH1D, float& NC1D, float& NR1D, float& NI1D, float& NS1D, float& NG1D, float& NH1D, float& VI1D, float& VS1D, float& VG1D, float& FI1D, float& FS1D, float& AI1D, float& AS1D, float& AG1D, float& AH1D, float& I3M1D, float& VTQC, float& VTQR, float& VTQI, float& VTQS, float& VTQG, float& VTQH, float& VTNC, float& VTNR, float& VTNI, float& VTNS, float& VTNG, float& VTNH, float& VTVI, float& VTVS, float& VTVG, float& VTFI, float& VTFS, float& VTAI, float& VTAS, float& VTAG, float& VTAH, float& VTI3M, float& SASPR, float RHO) {
    const float AVTC = 8.8462E+02f;
    const float BVTC = 9.7593E+07f;
    const float CVTC = (-3.4249E+11f);
    const float AVTR = 2.1454E+00f;
    const float BVTR = (-2.2812E-04f);
    const float CVTR = 2.9676E-09f;
    const float CQC1 = 2.0901E+01f;
    const float CQC2 = 9.9111E-01f;
    const float CQC3 = 4.4182E+00f;
    const float CNC1 = 1.8276E+01f;
    const float CNC2 = 1.0015E+00f;
    const float CNC3 = 1.9838E+00f;
    const float CQR1 = 1.5943E+01f;
    const float CQR2 = 1.1898E+00f;
    const float CQR3 = 4.0073E+00f;
    const float CNR1 = 9.4791E+00f;
    const float CNR2 = 9.7607E-01f;
    const float CNR3 = 1.0858E+00f;
    float QRHO = 0.0f, RHOAJ = 0.0f, MVRC = 0.0f, MVRR = 0.0f, GUC = 0.0f, GUR = 0.0f, LAMC = 0.0f, LAMR = 0.0f, LAMI = 0.0f, LAMS = 0.0f, LAMG = 0.0f, LAMH = 0.0f, RHOI = 0.0f, RHOS = 0.0f, RHOG = 0.0f, AFAC = 0.0f, AFAR = 0.0f, AFAI = 0.0f, AFAS = 0.0f, AFAG = 0.0f, AFAH = 0.0f, ADAGR = 0.0f, ZETA3 = 0.0f, AMI = 0.0f, BMI = 0.0f, AMS = 0.0f, AMG = 0.0f, AVI = 0.0f, BVI = 0.0f, AVS = 0.0f, BVS = 0.0f, AVG = 0.0f, BVG = 0.0f, AVH = 0.0f, BVH = 0.0f, ZETA = 0.0f, FSQC = 0.0f, FSQR = 0.0f, FSQI = 0.0f, FSQS = 0.0f, FSQG = 0.0f, FSQH = 0.0f, FSNC = 0.0f, FSNR = 0.0f, FSNI = 0.0f, FSNS = 0.0f, FSNG = 0.0f, FSNH = 0.0f, FSVI = 0.0f, FSVS = 0.0f, FSVG = 0.0f, FSFI = 0.0f, FSAI = 0.0f, FSAS = 0.0f, FSAG = 0.0f, FSAH = 0.0f, MVDC = 0.0f, MVDR = 0.0f, MVDI = 0.0f, MVDS = 0.0f, MVDG = 0.0f, MVDH = 0.0f, BEST = 0.0f;
    QRHO = sqrtf(RHO);
    RHOAJ = NTU_POWF((NTU_RHOSU / RHO), 0.54f);
    if ((QC1D >= NTU_QSMALL)) {
        ntu_solve_afac(AC, TK1D, QC1D, NC1D, LAMC, MVDC, AFAC);
        MVRC = ntu_fmin(ntu_fmax(NTU_POWF(((QC1D / NC1D) / NTU_C4PI3W), NTU_THRD), NTU_RCMIN), NTU_RCMAX);
        GUC = NTU_EXPF(NTU_EXPF(((NTU_AFU + (NTU_BFU * NTU_POWF(NTU_LOGF(MVRC), 3.0f))) + (NTU_CFU * NTU_POWF(QRHO, 3.0f)))));
        VTQC = ((NTU_EXPF(((CQC1 + (CQC2 * NTU_LOGF(NC1D))) + (CQC3 * NTU_LOGF(MVRC)))) * GUC) / QC1D);
        VTNC = ((NTU_EXPF(((CNC1 + (CNC2 * NTU_LOGF(NC1D))) + (CNC3 * NTU_LOGF(MVRC)))) * GUC) / NC1D);
        VTQC = ntu_fmin(VTQC, NTU_VTCMAX);
        VTNC = ntu_fmin(VTNC, NTU_VTCMAX);
    }
    if ((QR1D >= NTU_QSMALL)) {
        ntu_solve_afar(TK1D, QR1D, NR1D, LAMR, MVDR, AFAR);
        MVRR = ntu_fmin(ntu_fmax(NTU_POWF(((QR1D / NR1D) / NTU_C4PI3W), NTU_THRD), NTU_RRMIN), NTU_RRMAX);
        GUR = NTU_EXPF(NTU_EXPF(((NTU_AFU + (NTU_BFU * NTU_POWF(NTU_LOGF(MVRR), 3.0f))) + (NTU_CFU * NTU_POWF(QRHO, 3.0f)))));
        VTQR = ((NTU_EXPF(((CQR1 + (CQR2 * NTU_LOGF(NR1D))) + (CQR3 * NTU_LOGF(MVRR)))) * GUR) / QR1D);
        VTNR = ((NTU_EXPF(((CNR1 + (CNR2 * NTU_LOGF(NR1D))) + (CNR3 * NTU_LOGF(MVRR)))) * GUR) / NR1D);
        VTQR = ntu_fmin(VTQR, NTU_VTRMAX);
        VTNR = ntu_fmin(VTNR, NTU_VTRMAX);
    }
    if ((QI1D >= NTU_QSMALL)) {
        ntu_solve_afai(TK1D, P1D, RHO, QV1D, QI1D, NI1D, VI1D, FI1D, AI1D, I3M1D, ADAGR, ZETA, LAMI, AFAI, MVDI, RHOI, AMI, BMI, AVI, BVI, BEST);
        FSQI = NTU_EXPF(((ntu_gamln((((BVI + BMI) + AFAI) + 1.0f)) - ntu_gamln(((BMI + AFAI) + 1.0f))) - (BVI * NTU_LOGF(LAMI))));
        FSNI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 1.0f)) - ntu_gamln((AFAI + 1.0f))) - (BVI * NTU_LOGF(LAMI))));
        FSVI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 4.0f)) - ntu_gamln((AFAI + 4.0f))) - (BVI * NTU_LOGF(LAMI))));
        VTQI = ntu_fmin(((RHOAJ * FSQI) * AVI), NTU_VTIMAX);
        VTNI = ntu_fmin(((RHOAJ * FSNI) * AVI), NTU_VTIMAX);
        VTVI = ntu_fmin(((RHOAJ * FSVI) * AVI), NTU_VTIMAX);
        if ((AI1D >= NTU_ASMALL)) {
            FSAI = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 3.0f)) - ntu_gamln((AFAI + 3.0f))) - (BVI * NTU_LOGF(LAMI))));
            VTAI = ntu_fmin(((RHOAJ * FSAI) * AVI), NTU_VTIMAX);
        }
        if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
            ZETA3 = ((3.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            FSFI = NTU_EXPF(((ntu_gamln((((BVI + ZETA3) + AFAI) + 4.0f)) - ntu_gamln(((ZETA3 + AFAI) + 4.0f))) - (BVI * NTU_LOGF(LAMI))));
            VTFI = ntu_fmin(((RHOAJ * FSFI) * AVI), NTU_VTIMAX);
            VTI3M = ntu_fmin(((RHOAJ * FSVI) * AVI), NTU_VTIMAX);
        }
    }
    if ((QS1D >= NTU_QSMALL)) {
        ntu_solve_afas(TK1D, RHO, QS1D, QC1D, NS1D, VS1D, FS1D, AS1D, AFAS, LAMS, MVDS, RHOS, SASPR, AMS, AVS, BVS);
        FSQS = NTU_EXPF(((ntu_gamln((((BVS + NTU_BMS) + AFAS) + 1.0f)) - ntu_gamln(((NTU_BMS + AFAS) + 1.0f))) - (BVS * NTU_LOGF(LAMS))));
        FSNS = NTU_EXPF(((ntu_gamln(((BVS + AFAS) + 1.0f)) - ntu_gamln((AFAS + 1.0f))) - (BVS * NTU_LOGF(LAMS))));
        FSVS = NTU_EXPF(((ntu_gamln(((BVS + AFAS) + 4.0f)) - ntu_gamln((AFAS + 4.0f))) - (BVS * NTU_LOGF(LAMS))));
        VTQS = ntu_fmin(((RHOAJ * FSQS) * AVS), NTU_VTSMAX);
        VTNS = ntu_fmin(((RHOAJ * FSNS) * AVS), NTU_VTSMAX);
        VTVS = ntu_fmin(((RHOAJ * FSVS) * AVS), NTU_VTSMAX);
        VTFS = VTVS;
        if ((AS1D >= NTU_ASMALL)) {
            FSAS = NTU_EXPF(((ntu_gamln(((BVS + AFAS) + 3.0f)) - ntu_gamln((AFAS + 3.0f))) - (BVS * NTU_LOGF(LAMS))));
            VTAS = ntu_fmin(((RHOAJ * FSAS) * AVS), NTU_VTSMAX);
        }
    }
    if ((QG1D >= NTU_QSMALL)) {
        ntu_solve_afag(TK1D, RHO, QG1D, QC1D, NG1D, VG1D, AG1D, LAMG, AFAG, MVDG, RHOG, AMG, AVG, BVG);
        FSQG = NTU_EXPF(((ntu_gamln((((BVG + NTU_BMG) + AFAG) + 1.0f)) - ntu_gamln(((NTU_BMG + AFAG) + 1.0f))) - (BVG * NTU_LOGF(LAMG))));
        FSNG = NTU_EXPF(((ntu_gamln(((BVG + AFAG) + 1.0f)) - ntu_gamln((AFAG + 1.0f))) - (BVG * NTU_LOGF(LAMG))));
        FSVG = NTU_EXPF(((ntu_gamln(((BVG + AFAG) + 4.0f)) - ntu_gamln((AFAG + 4.0f))) - (BVG * NTU_LOGF(LAMG))));
        VTQG = ntu_fmin(((RHOAJ * FSQG) * AVG), NTU_VTGMAX);
        VTNG = ntu_fmin(((RHOAJ * FSNG) * AVG), NTU_VTGMAX);
        VTVG = ntu_fmin(((RHOAJ * FSVG) * AVG), NTU_VTGMAX);
        if ((AG1D >= NTU_ASMALL)) {
            FSAG = NTU_EXPF(((ntu_gamln(((BVG + AFAG) + 3.0f)) - ntu_gamln((AFAG + 3.0f))) - (BVG * NTU_LOGF(LAMG))));
            VTAG = ntu_fmin(((RHOAJ * FSAG) * AVG), NTU_VTGMAX);
        }
    }
    if ((QH1D >= NTU_QSMALL)) {
        ntu_solve_afah(TK1D, RHO, QH1D, NH1D, AH1D, LAMH, AFAH, MVDH, AVH, BVH);
        FSQH = NTU_EXPF(((ntu_gamln((((BVH + NTU_BMH) + AFAH) + 1.0f)) - ntu_gamln(((NTU_BMH + AFAH) + 1.0f))) - (BVH * NTU_LOGF(LAMH))));
        FSNH = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 1.0f)) - ntu_gamln((AFAH + 1.0f))) - (BVH * NTU_LOGF(LAMH))));
        VTQH = ntu_fmin(((RHOAJ * FSQH) * AVH), NTU_VTHMAX);
        VTNH = ntu_fmin(((RHOAJ * FSNH) * AVH), NTU_VTHMAX);
        if ((AH1D >= NTU_ASMALL)) {
            FSAH = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 3.0f)) - ntu_gamln((AFAH + 3.0f))) - (BVH * NTU_LOGF(LAMH))));
            VTAH = ntu_fmin(((RHOAJ * FSAH) * AVH), NTU_VTHMAX);
        }
    }
}

// WRF SUBROUTINE SMALL_DT (module_mp_ntu.F:3267-4365)
NTU_DEV_BIG void ntu_small_dt(NtuAeroConst* __restrict__ AC, float DT, float& DTS, float SDTS, float DTKDT, float DQVDT, float DPDT, float RHO, float& TK1D, float P1D, float& QV1D, float& QC1D, float& QR1D, float& QI1D, float& QS1D, float& QG1D, float& QH1D, float& NC1D, float& NR1D, float& NI1D, float& NS1D, float& NG1D, float& NH1D, float& VI1D, float& VS1D, float& VG1D, float& FI1D, float& FS1D, float& AI1D, float& AS1D, float& AG1D, float& AH1D, float& I3M1D, float& SASPR, float& XDNC, float& XDNR, float& GQCTR) {
    const float TORR = 1.0E-2f;
    const float AQ1 = 6.6793E+0f;
    const float BQ1 = 1.0090E+0f;
    const float CQ1 = 1.4095E+0f;
    const float AQ2 = 9.9912E+0f;
    const float BQ2 = (-4.7678E-1f);
    const float CQ2 = (-3.1388E-2f);
    const float AN9 = (-1.0593E+0f);
    const float BN9 = 8.9774E-1f;
    const float CN9 = (-2.8403E-1f);
    const float DN9 = 1.6328E+0f;
    const float AN10 = 8.2841E+0f;
    const float BN10 = 9.7219E-1f;
    const float CN10 = (-5.0808E-1f);
    const float ZC1 = 0.69509913f;
    const float ZC2 = -0.46685819f;
    const float ZC3 = 0.30490087f;
    const float ZC4 = 1.62148100f;
    const float ZP1 = 0.36793126f;
    const float ZP2 = 1.82782890f;
    const float ZP3 = 0.63206874f;
    const float ZP4 = -1.00164090f;
    float CPM = 0.0f, ESW = 0.0f, ESI = 0.0f, QVSW = 0.0f, QVSI = 0.0f, SSRW = 0.0f, SSRI = 0.0f, XXLV = 0.0f, XXLS = 0.0f, XXLF = 0.0f, AFAC = 0.0f, AFAR = 0.0f, AFAI = 0.0f, AFAS = 0.0f, AFAG = 0.0f, AFAH = 0.0f, ADAGR = 0.0f, ZETA = 0.0f, AMI = 0.0f, BMI = 0.0f, AMS = 0.0f, AMG = 0.0f, AVI = 0.0f, BVI = 0.0f, AVS = 0.0f, BVS = 0.0f, AVG = 0.0f, BVG = 0.0f, AVH = 0.0f, BVH = 0.0f, DV = 0.0f, MUA = 0.0f, KAP = 0.0f, SCN = 0.0f, TC1D = 0.0f, IDT = 0.0f, ELCLD = 0.0f, ELDLD = 0.0f, ELCLC = 0.0f, CAPS = 0.0f, QVTEND = 0.0f, QCTEND = 0.0f, QRTEND = 0.0f, QITEND = 0.0f, QSTEND = 0.0f, QGTEND = 0.0f, QHTEND = 0.0f, RATIO = 0.0f, QVSOUR = 0.0f, QCSOUR = 0.0f, QRSOUR = 0.0f, QISOUR = 0.0f, QSSOUR = 0.0f, QGSOUR = 0.0f, QHSOUR = 0.0f, QVSINK = 0.0f, QCSINK = 0.0f, QRSINK = 0.0f, QISINK = 0.0f, QSSINK = 0.0f, QGSINK = 0.0f, QHSINK = 0.0f, NCSOUR = 0.0f, NISOUR = 0.0f, NCSINK = 0.0f, NISINK = 0.0f, NSSOUR = 0.0f, NSSINK = 0.0f, FISOUR = 0.0f, FISINK = 0.0f, FSSOUR = 0.0f, FSSINK = 0.0f, VISOUR = 0.0f, VISINK = 0.0f, VSSOUR = 0.0f, VSSINK = 0.0f, VGSOUR = 0.0f, VGSINK = 0.0f, AISOUR = 0.0f, AISINK = 0.0f, ASSOUR = 0.0f, ASSINK = 0.0f, AGSOUR = 0.0f, AGSINK = 0.0f, AHSOUR = 0.0f, AHSINK = 0.0f, IISOUR = 0.0f, IISINK = 0.0f, QVWTEND = 0.0f, QVITEND = 0.0f, DRHIDT = 0.0f, DRHWDT = 0.0f, DTSI = 0.0f, DTSW = 0.0f, QACCV = 0.0f, QACRC = 0.0f, QVDVC = 0.0f, QVDVR = 0.0f, QVDVI = 0.0f, QVDVS = 0.0f, QVDVG = 0.0f, QVDVH = 0.0f, QSBIV = 0.0f, QSBSV = 0.0f, QSBGV = 0.0f, QSBHV = 0.0f, QEVCV = 0.0f, QEVRV = 0.0f, QEVSV = 0.0f, QEVGV = 0.0f, QEVHV = 0.0f, NACCV = 0.0f, NACRC = 0.0f, NSBIV = 0.0f, NSBSV = 0.0f, NSBGV = 0.0f, NSBHV = 0.0f, VVDVI = 0.0f, VSBIV = 0.0f, VVDVG = 0.0f, VSBGV = 0.0f, VEVGV = 0.0f, FVDVI = 0.0f, VVDVS = 0.0f, VSBSV = 0.0f, VEVSV = 0.0f, FSBIV = 0.0f, FVDVS = 0.0f, FSBSV = 0.0f, FEVSV = 0.0f, AVDVI = 0.0f, ASBIV = 0.0f, AVDVS = 0.0f, ASBSV = 0.0f, AEVSV = 0.0f, AVDVG = 0.0f, ASBGV = 0.0f, AEVGV = 0.0f, AVDVH = 0.0f, ASBHV = 0.0f, AEVHV = 0.0f, IVDVI = 0.0f, ISBIV = 0.0f, MVDC = 0.0f, MVDR = 0.0f, MVDI = 0.0f, MVDS = 0.0f, MVDG = 0.0f, MVDH = 0.0f, MVRC = 0.0f, MVRR = 0.0f, BTMP = 0.0f, BSTMP = 0.0f, BGTMP = 0.0f, BHTMP = 0.0f, ABW = 0.0f, ABI = 0.0f, RHOAJ = 0.0f, LMVRC = 0.0f, LMVRR = 0.0f, RHOI = 0.0f, RHOS = 0.0f, RHOG = 0.0f, VENQS = 0.0f, VENQG = 0.0f, VENQH = 0.0f, SUMDEP = 0.0f, VDMAX = 0.0f, EVMAX = 0.0f, SBMAX = 0.0f, SUMCND = 0.0f, SUMEVP = 0.0f, SUMSUB = 0.0f, H2Z = 0.0f, H4Z = 0.0f, ZETA2 = 0.0f, ZETA3 = 0.0f, ZETA4 = 0.0f, ZETA5 = 0.0f, GVHAB = 0.0f, IPH = 0.0f, IPG = 0.0f, GI1 = 0.0f, INHGR = 0.0f, BEST = 0.0f, DNIVD = 0.0f, DNSVD = 0.0f, DNGVD = 0.0f, QTMP0 = 0.0f, QTMP1 = 0.0f, QTMP2 = 0.0f, QTMP3 = 0.0f, QTMP4 = 0.0f, QTMP5 = 0.0f, QTMP6 = 0.0f, QTMP7 = 0.0f, QTMP8 = 0.0f, QTMP9 = 0.0f, FTMP0 = 0.0f, FTMP1 = 0.0f, FTMP2 = 0.0f, FTMP3 = 0.0f, FTMP4 = 0.0f, FTMP5 = 0.0f, FTMP6 = 0.0f, FTMP7 = 0.0f, FTMP8 = 0.0f, FTMP9 = 0.0f, ATMP0 = 0.0f, ATMP1 = 0.0f, ATMP2 = 0.0f, ATMP3 = 0.0f, ATMP4 = 0.0f, ATMP5 = 0.0f, ATMP6 = 0.0f, ATMP7 = 0.0f, ATMP8 = 0.0f, ATMP9 = 0.0f, LAMC = 0.0f, LAMR = 0.0f, LAMI = 0.0f, LAMS = 0.0f, LAMG = 0.0f, LAMH = 0.0f, VENQI = 0.0f, VENFI = 0.0f, VENAI = 0.0f, VENAS = 0.0f, VENAG = 0.0f, VENAH = 0.0f, VENIC = 0.0f, VENIA = 0.0f, VENQI0 = 0.0f, VENAI0 = 0.0f, RAT1 = 0.0f, RAT2 = 0.0f, LLMI = 0.0f, LLMS = 0.0f, LLMG = 0.0f, LLMH = 0.0f;
    int I = 0, HID = 0;
    QVTEND = 0.0f;
    QVWTEND = 0.0f;
    QVITEND = 0.0f;
    QCTEND = 0.0f;
    QRTEND = 0.0f;
    QITEND = 0.0f;
    QSTEND = 0.0f;
    QGTEND = 0.0f;
    QHTEND = 0.0f;
    MVRC = 0.0f;
    MVRR = 0.0f;
    MVDC = 0.0f;
    MVDR = 0.0f;
    MVDI = 0.0f;
    MVDS = 0.0f;
    MVDG = 0.0f;
    MVDH = 0.0f;
    QACCV = 0.0f;
    QACRC = 0.0f;
    QVDVC = 0.0f;
    QVDVR = 0.0f;
    QVDVI = 0.0f;
    QVDVS = 0.0f;
    QVDVG = 0.0f;
    QVDVH = 0.0f;
    QSBIV = 0.0f;
    QSBSV = 0.0f;
    QSBGV = 0.0f;
    QSBHV = 0.0f;
    QEVCV = 0.0f;
    QEVRV = 0.0f;
    QEVSV = 0.0f;
    QEVGV = 0.0f;
    QEVHV = 0.0f;
    NACCV = 0.0f;
    NACRC = 0.0f;
    NSBIV = 0.0f;
    NSBSV = 0.0f;
    NSBGV = 0.0f;
    NSBHV = 0.0f;
    VVDVI = 0.0f;
    VSBIV = 0.0f;
    VVDVS = 0.0f;
    VSBSV = 0.0f;
    VEVSV = 0.0f;
    VVDVG = 0.0f;
    VSBGV = 0.0f;
    VEVGV = 0.0f;
    FVDVI = 0.0f;
    FSBIV = 0.0f;
    FVDVS = 0.0f;
    FSBSV = 0.0f;
    FEVSV = 0.0f;
    IVDVI = 0.0f;
    ISBIV = 0.0f;
    AVDVI = 0.0f;
    ASBIV = 0.0f;
    AVDVS = 0.0f;
    ASBSV = 0.0f;
    AEVSV = 0.0f;
    AVDVG = 0.0f;
    ASBGV = 0.0f;
    AEVGV = 0.0f;
    AVDVH = 0.0f;
    ASBHV = 0.0f;
    AEVHV = 0.0f;
    ESW = ntu_fmin((0.99f * P1D), ntu_polysvp(TK1D, 0));
    ESI = ntu_fmin((0.99f * P1D), ntu_polysvp(TK1D, 1));
    if ((ESI > ESW)) {
        ESI = ESW;
    }
    QVSW = ((0.622f * ESW) / (P1D - ESW));
    QVSI = ((0.622f * ESI) / (P1D - ESI));
    SSRW = ((QV1D / QVSW) - 1.0f);
    SSRI = ((QV1D / QVSI) - 1.0f);
    XXLV = (3.1484E6f - (2370.0f * TK1D));
    XXLS = ((3.15E6f - (2370.0f * TK1D)) + 0.3337E6f);
    XXLF = (2836310.8f - (3.1484E6f - (2370.0f * TK1D)));
    ELDLD = (1.0f + (((XXLS * XXLS) * QV1D) / ((NTU_CP * NTU_RV) * ntu_sq(TK1D))));
    ELCLD = (1.0f + (((XXLV * XXLS) * QV1D) / ((NTU_CP * NTU_RV) * ntu_sq(TK1D))));
    ELCLC = (1.0f + (((XXLV * XXLV) * QV1D) / ((NTU_CP * NTU_RV) * ntu_sq(TK1D))));
    DRHWDT = ((1.0f + SSRW) * (((DPDT / P1D) + (DQVDT / QV1D)) - ((XXLV * DTKDT) / (NTU_RV * ntu_sq(TK1D)))));
    DRHIDT = ((1.0f + SSRI) * (((DPDT / P1D) + (DQVDT / QV1D)) - ((XXLS * DTKDT) / (NTU_RV * ntu_sq(TK1D)))));
    if ((fabsf(DRHWDT) < NTU_RLIMIT)) {
        DTSW = 1.0E7f;
    } else {
        DTSW = (-((SSRW / DRHWDT) * 1.01f));
    }
    if ((fabsf(DRHIDT) < NTU_RLIMIT)) {
        DTSI = 1.0E7f;
    } else {
        DTSI = (-((SSRI / DRHIDT) * 1.01f));
    }
    if ((TK1D > NTU_TK0C)) {
        if ((DTSW > 0.0f)) {
            DTS = ntu_fmin(ntu_fmin(DT, DTSW), (DT - SDTS));
        }
    } else {
        if ((DTSI > 0.0f)) {
            DTS = ntu_fmin(ntu_fmin(DT, DTSI), (DT - SDTS));
        }
        if ((DTSW > 0.0f)) {
            DTS = ntu_fmin(ntu_fmin(ntu_fmin(DT, DTS), DTSW), (DT - SDTS));
        }
    }
    DTS = ntu_fmax(NTU_DTMIN, ntu_fmin(DT, (DT - SDTS)));
    IDT = (1.0f / DTS);
    TC1D = (TK1D - NTU_TK0C);
    DV = ((2.11E-5f * NTU_POWF((TK1D / NTU_TK0C), 1.94f)) * (101325.0f / P1D));
    MUA = ((1.72E-5f * (393.0f / (TK1D + 120.0f))) * NTU_POWF((TK1D / NTU_TK0C), 1.5f));
    KAP = (2.3971E-2f + (0.0078E-2f * TC1D));
    SCN = NTU_POWF((MUA / (RHO * DV)), NTU_THRD);
    CPM = (NTU_CP * (1.0f + (0.887f * QV1D)));
    ABW = ((((TK1D * NTU_RV) / ESW) / DV) + (((XXLV * (((XXLV / TK1D) / NTU_RV) - 1.0f)) / TK1D) / KAP));
    ABI = ((((TK1D * NTU_RV) / ESI) / DV) + (((XXLS * (((XXLS / TK1D) / NTU_RV) - 1.0f)) / TK1D) / KAP));
    RHOAJ = NTU_POWF((NTU_RHOSU / RHO), 0.54f);
    HID = ntu_imax(ntu_imin(ntu_nint((fabsf(TC1D) / 0.25f)), 120), 0);
    INHGR = NTU_ITBLE[HID];
    GVHAB = (((INHGR - 1.0f) / (INHGR + 2.0f)) + 1.0f);
    if ((QC1D >= NTU_QSMALL)) {
        ntu_solve_afac(AC, TK1D, QC1D, NC1D, LAMC, MVDC, AFAC);
        MVRC = ntu_fmin(ntu_fmax(NTU_POWF(((QC1D / NC1D) / NTU_C4PI3W), NTU_THRD), NTU_RCMIN), NTU_RCMAX);
        LMVRC = NTU_LOGF(MVRC);
    }
    if ((QR1D >= NTU_QSMALL)) {
        ntu_solve_afar(TK1D, QR1D, NR1D, LAMR, MVDR, AFAR);
        MVRR = ntu_fmin(ntu_fmax(NTU_POWF(((QR1D / NR1D) / NTU_C4PI3W), NTU_THRD), NTU_RRMIN), NTU_RRMAX);
        LMVRR = NTU_LOGF(MVRR);
    }
    if ((QI1D >= NTU_QSMALL)) {
        ntu_solve_afai(TK1D, P1D, RHO, QV1D, QI1D, NI1D, VI1D, FI1D, AI1D, I3M1D, ADAGR, ZETA, LAMI, AFAI, MVDI, RHOI, AMI, BMI, AVI, BVI, BEST);
        GI1 = ntu_gamln((AFAI + 1.0f));
        LLMI = NTU_LOGF(LAMI);
        if (((ADAGR - 1.0f) >= NTU_SLIMIT)) {
            IPH = ((3.0f * ADAGR) / (ADAGR + 2.0f));
            IPG = (3.0f / (ADAGR + 2.0f));
            ZETA2 = ((2.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            ZETA3 = ((3.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            ZETA4 = ((4.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            ZETA5 = ((5.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            H2Z = (ZC2 * ZETA);
            H4Z = (ZC4 * ZETA);
            QTMP0 = NTU_EXPF(((ntu_gamln(((H2Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H2Z + 1.0f))));
            QTMP1 = NTU_EXPF(((ntu_gamln(((H4Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H4Z + 1.0f))));
            FTMP0 = NTU_EXPF(((ntu_gamln((((H2Z + ZETA3) + AFAI) + 2.0f)) - GI1) - (LLMI * ((H2Z + ZETA3) + 1.0f))));
            FTMP1 = NTU_EXPF(((ntu_gamln((((H4Z + ZETA3) + AFAI) + 2.0f)) - GI1) - (LLMI * ((H4Z + ZETA3) + 1.0f))));
            ATMP0 = NTU_EXPF(((ntu_gamln(((H2Z + AFAI) + 1.0f)) - GI1) - (LLMI * H2Z)));
            ATMP1 = NTU_EXPF(((ntu_gamln(((H4Z + AFAI) + 1.0f)) - GI1) - (LLMI * H4Z)));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP2 = (LLMI * (((H2Z + BVI) + IPH) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + BVI) + IPH) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + BVI) + IPH) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + BVI) + IPH) + AFAI) + 2.0f)) - GI1) - QTMP3));
                FTMP2 = (LLMI * ((((H2Z + BVI) + IPH) + ZETA3) + 1.0f));
                FTMP3 = (LLMI * ((((H4Z + BVI) + IPH) + ZETA3) + 1.0f));
                FTMP4 = NTU_EXPF(((ntu_gamln((((((H2Z + BVI) + IPH) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP2));
                FTMP5 = NTU_EXPF(((ntu_gamln((((((H4Z + BVI) + IPH) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP3));
                ATMP2 = (LLMI * ((H2Z + BVI) + IPH));
                ATMP3 = (LLMI * ((H4Z + BVI) + IPH));
                ATMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + BVI) + IPH) + AFAI) + 1.0f)) - GI1) - ATMP2));
                ATMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + BVI) + IPH) + AFAI) + 1.0f)) - GI1) - ATMP3));
                VENQI = ((((((NTU_AVIS * ZC1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVIS * ZC3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVIS * ZC1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA2)))) + ((((NTU_BVIS * ZC3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA2))));
                VENFI = ((((((NTU_AVIS * ZC1) * FTMP0) / NTU_POWF(NTU_DI0, (ZETA3 + H2Z))) + (((NTU_AVIS * ZC3) * FTMP1) / NTU_POWF(NTU_DI0, (H4Z + ZETA3)))) + ((((NTU_BVIS * ZC1) * BTMP) * FTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA5)))) + ((((NTU_BVIS * ZC3) * BTMP) * FTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA5))));
                VENAI = ((((((NTU_AVIS * ZC1) * ATMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVIS * ZC3) * ATMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVIS * ZC1) * BTMP) * ATMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA2)))) + ((((NTU_BVIS * ZC3) * BTMP) * ATMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA2))));
                QTMP6 = (LLMI * (BVI + IPH));
                QTMP7 = (LLMI * (BVI + IPG));
                QTMP8 = NTU_EXPF(((ntu_gamln((((BVI + IPH) + AFAI) + 1.0f)) - GI1) - QTMP6));
                QTMP9 = NTU_EXPF(((ntu_gamln((((BVI + IPG) + AFAI) + 1.0f)) - GI1) - QTMP7));
                VENIC = (NTU_AVIS + (((NTU_BVIS * BTMP) * QTMP8) / NTU_POWF(NTU_DI0, ZETA2)));
                VENIA = (NTU_AVIS + (((NTU_BVIS * BTMP) * QTMP9) * NTU_POWF(NTU_DI0, ZETA)));
                INHGR = ((INHGR * VENIC) / VENIA);
                GVHAB = (((INHGR - 1.0f) / (INHGR + 2.0f)) + 1.0f);
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP2 = (LLMI * (((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP3));
                FTMP2 = (LLMI * ((((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3) + 1.0f));
                FTMP3 = (LLMI * ((((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3) + 1.0f));
                FTMP4 = NTU_EXPF(((ntu_gamln((((((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP2));
                FTMP5 = NTU_EXPF(((ntu_gamln((((((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP3));
                ATMP2 = (LLMI * (((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3));
                ATMP3 = (LLMI * (((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3));
                ATMP4 = NTU_EXPF(((ntu_gamln((((((H2Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3) + AFAI) + 1.0f)) - GI1) - ATMP2));
                ATMP5 = NTU_EXPF(((ntu_gamln((((((H4Z + (BVI / 2.0f)) + (IPH / 2.0f)) + ZETA3) + AFAI) + 1.0f)) - GI1) - ATMP3));
                VENQI = ((((((NTU_AVSG * ZC1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVSG * ZC3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVSG * ZC1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA)))) + ((((NTU_BVSG * ZC3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA))));
                VENFI = ((((((NTU_AVSG * ZC1) * FTMP0) / NTU_POWF(NTU_DI0, (ZETA3 + H2Z))) + (((NTU_AVSG * ZC3) * FTMP1) / NTU_POWF(NTU_DI0, (H4Z + ZETA3)))) + ((((NTU_BVSG * ZC1) * BTMP) * FTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA4)))) + ((((NTU_BVSG * ZC3) * BTMP) * FTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA4))));
                VENAI = ((((((NTU_AVSG * ZC1) * ATMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVSG * ZC3) * ATMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVSG * ZC1) * BTMP) * ATMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA)))) + ((((NTU_BVSG * ZC3) * BTMP) * ATMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA))));
                QTMP6 = (LLMI * ((BVI / 2.0f) + (IPH / 2.0f)));
                QTMP7 = (LLMI * ((BVI / 2.0f) + (IPG / 2.0f)));
                QTMP8 = NTU_EXPF(((ntu_gamln(((((BVI / 2.0f) + (IPH / 2.0f)) + AFAI) + 1.0f)) - GI1) - QTMP6));
                QTMP9 = NTU_EXPF(((ntu_gamln(((((BVI / 2.0f) + (IPG / 2.0f)) + AFAI) + 1.0f)) - GI1) - QTMP7));
                VENIC = (NTU_AVSG + (((NTU_BVSG * BTMP) * QTMP8) / NTU_POWF(NTU_DI0, ZETA)));
                VENIA = (NTU_AVSG + (((NTU_BVSG * BTMP) * QTMP9) * NTU_POWF(NTU_DI0, (ZETA / 2.0f))));
                INHGR = ((INHGR * VENIC) / VENIA);
                GVHAB = (((INHGR - 1.0f) / (INHGR + 2.0f)) + 1.0f);
            }
        } else if (((1.0f - ADAGR) >= NTU_SLIMIT)) {
            IPG = (3.0f / (ADAGR + 2.0f));
            IPH = ((3.0f * ADAGR) / (ADAGR + 2.0f));
            ZETA2 = ((2.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            ZETA3 = ((3.0f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            ZETA4 = ((2.5f * (ADAGR - 1.0f)) / (ADAGR + 2.0f));
            H2Z = (ZP2 * ZETA);
            H4Z = (ZP4 * ZETA);
            QTMP0 = NTU_EXPF(((ntu_gamln(((H2Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H2Z + 1.0f))));
            QTMP1 = NTU_EXPF(((ntu_gamln(((H4Z + AFAI) + 2.0f)) - GI1) - (LLMI * (H4Z + 1.0f))));
            FTMP0 = NTU_EXPF(((ntu_gamln((((H2Z + ZETA3) + AFAI) + 2.0f)) - GI1) - (LLMI * ((H2Z + ZETA3) + 1.0f))));
            FTMP1 = NTU_EXPF(((ntu_gamln((((H4Z + ZETA3) + AFAI) + 2.0f)) - GI1) - (LLMI * ((H4Z + ZETA3) + 1.0f))));
            ATMP0 = NTU_EXPF(((ntu_gamln(((H2Z + AFAI) + 1.0f)) - GI1) - (LLMI * H2Z)));
            ATMP1 = NTU_EXPF(((ntu_gamln(((H4Z + AFAI) + 1.0f)) - GI1) - (LLMI * H4Z)));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP2 = (LLMI * (((H2Z + BVI) + IPG) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + BVI) + IPG) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + BVI) + IPG) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + BVI) + IPG) + AFAI) + 2.0f)) - GI1) - QTMP3));
                FTMP2 = (LLMI * ((((H2Z + BVI) + IPG) + ZETA3) + 1.0f));
                FTMP3 = (LLMI * ((((H4Z + BVI) + IPG) + ZETA3) + 1.0f));
                FTMP4 = NTU_EXPF(((ntu_gamln((((((H2Z + BVI) + IPG) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP2));
                FTMP5 = NTU_EXPF(((ntu_gamln((((((H4Z + BVI) + IPG) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP3));
                ATMP2 = (LLMI * ((H2Z + BVI) + IPG));
                ATMP3 = (LLMI * ((H4Z + BVI) + IPG));
                ATMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + BVI) + IPG) + AFAI) + 1.0f)) - GI1) - ATMP2));
                ATMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + BVI) + IPG) + AFAI) + 1.0f)) - GI1) - ATMP3));
                VENQI = ((((((NTU_AVIS * ZP1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVIS * ZP3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVIS * ZP1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z - ZETA)))) + ((((NTU_BVIS * ZP3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z - ZETA))));
                VENFI = ((((((NTU_AVIS * ZP1) * FTMP0) / NTU_POWF(NTU_DI0, (ZETA3 + H2Z))) + (((NTU_AVIS * ZP3) * FTMP1) / NTU_POWF(NTU_DI0, (H4Z + ZETA3)))) + ((((NTU_BVIS * ZP1) * BTMP) * FTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA2)))) + ((((NTU_BVIS * ZP3) * BTMP) * FTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA2))));
                VENAI = ((((((NTU_AVIS * ZP1) * ATMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVIS * ZP3) * ATMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVIS * ZP1) * BTMP) * ATMP4) / NTU_POWF(NTU_DI0, (H2Z - ZETA)))) + ((((NTU_BVIS * ZP3) * BTMP) * ATMP5) / NTU_POWF(NTU_DI0, (H4Z - ZETA))));
                QTMP6 = (LLMI * (BVI + IPH));
                QTMP7 = (LLMI * (BVI + IPG));
                QTMP8 = NTU_EXPF(((ntu_gamln((((BVI + IPH) + AFAI) + 1.0f)) - GI1) - QTMP6));
                QTMP9 = NTU_EXPF(((ntu_gamln((((BVI + IPG) + AFAI) + 1.0f)) - GI1) - QTMP7));
                VENIC = (NTU_AVIS + (((NTU_BVIS * BTMP) * QTMP8) / NTU_POWF(NTU_DI0, ZETA2)));
                VENIA = (NTU_AVIS + (((NTU_BVIS * BTMP) * QTMP9) * NTU_POWF(NTU_DI0, ZETA)));
                INHGR = ((INHGR * VENIC) / VENIA);
                GVHAB = (((INHGR - 1.0f) / (INHGR + 2.0f)) + 1.0f);
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP2 = (LLMI * (((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + 1.0f));
                QTMP3 = (LLMI * (((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + 1.0f));
                QTMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP2));
                QTMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + AFAI) + 2.0f)) - GI1) - QTMP3));
                FTMP2 = (LLMI * ((((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + ZETA3) + 1.0f));
                FTMP3 = (LLMI * ((((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + ZETA3) + 1.0f));
                FTMP4 = NTU_EXPF(((ntu_gamln((((((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP2));
                FTMP5 = NTU_EXPF(((ntu_gamln((((((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + ZETA3) + AFAI) + 2.0f)) - GI1) - FTMP3));
                ATMP2 = (LLMI * ((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)));
                ATMP3 = (LLMI * ((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)));
                ATMP4 = NTU_EXPF(((ntu_gamln(((((H2Z + (BVI / 2.0f)) + (IPG / 2.0f)) + AFAI) + 1.0f)) - GI1) - ATMP2));
                ATMP5 = NTU_EXPF(((ntu_gamln(((((H4Z + (BVI / 2.0f)) + (IPG / 2.0f)) + AFAI) + 1.0f)) - GI1) - ATMP3));
                VENQI = ((((((NTU_AVSG * ZP1) * QTMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVSG * ZP3) * QTMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVSG * ZP1) * BTMP) * QTMP4) / NTU_POWF(NTU_DI0, (H2Z - (ZETA / 2.0f))))) + ((((NTU_BVSG * ZP3) * BTMP) * QTMP5) / NTU_POWF(NTU_DI0, (H4Z - (ZETA / 2.0f)))));
                VENFI = ((((((NTU_AVSG * ZP1) * FTMP0) / NTU_POWF(NTU_DI0, (ZETA3 + H2Z))) + (((NTU_AVSG * ZP3) * FTMP1) / NTU_POWF(NTU_DI0, (H4Z + ZETA3)))) + ((((NTU_BVSG * ZP1) * BTMP) * FTMP4) / NTU_POWF(NTU_DI0, (H2Z + ZETA4)))) + ((((NTU_BVSG * ZP3) * BTMP) * FTMP5) / NTU_POWF(NTU_DI0, (H4Z + ZETA4))));
                VENAI = ((((((NTU_AVSG * ZP1) * ATMP0) / NTU_POWF(NTU_DI0, H2Z)) + (((NTU_AVSG * ZP3) * ATMP1) / NTU_POWF(NTU_DI0, H4Z))) + ((((NTU_BVSG * ZP1) * BTMP) * ATMP4) / NTU_POWF(NTU_DI0, (H2Z - (ZETA / 2.0f))))) + ((((NTU_BVSG * ZP3) * BTMP) * ATMP5) / NTU_POWF(NTU_DI0, (H4Z - (ZETA / 2.0f)))));
                QTMP6 = (LLMI * ((BVI / 2.0f) + (IPH / 2.0f)));
                QTMP7 = (LLMI * ((BVI / 2.0f) + (IPG / 2.0f)));
                QTMP8 = NTU_EXPF(((ntu_gamln(((((BVI / 2.0f) + (IPH / 2.0f)) + AFAI) + 1.0f)) - GI1) - QTMP6));
                QTMP9 = NTU_EXPF(((ntu_gamln(((((BVI / 2.0f) + (IPG / 2.0f)) + AFAI) + 1.0f)) - GI1) - QTMP7));
                VENIC = (NTU_AVSG + (((NTU_BVSG * BTMP) * QTMP8) / NTU_POWF(NTU_DI0, ZETA)));
                VENIA = (NTU_AVSG + (((NTU_BVSG * BTMP) * QTMP9) * NTU_POWF(NTU_DI0, (ZETA / 2.0f))));
                INHGR = ((INHGR * VENIC) / VENIA);
                GVHAB = (((INHGR - 1.0f) / (INHGR + 2.0f)) + 1.0f);
            }
        } else if ((fabsf((ADAGR - 1.0f)) < NTU_SLIMIT)) {
            QTMP0 = NTU_EXPF(((ntu_gamln((AFAI + 2.0f)) - GI1) - NTU_LOGF(LAMI)));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP2 = (LLMI * (2.0f + BVI));
                QTMP3 = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 3.0f)) - GI1) - QTMP2));
                ATMP2 = (LLMI * (1.0f + BVI));
                ATMP3 = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 2.0f)) - GI1) - ATMP2));
                VENQI = ((NTU_AVIS * QTMP0) + ((NTU_BVIS * BTMP) * QTMP3));
                VENFI = VENQI;
                VENAI = (NTU_AVIS + ((NTU_BVIS * BTMP) * ATMP3));
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP2 = (LLMI * (1.5f + (BVI / 2.0f)));
                QTMP3 = NTU_EXPF(((ntu_gamln((((BVI / 2.0f) + AFAI) + 2.5f)) - GI1) - QTMP2));
                ATMP2 = (LLMI * (0.5f + (BVI / 2.0f)));
                ATMP3 = NTU_EXPF(((ntu_gamln((((BVI / 2.0f) + AFAI) + 1.5f)) - GI1) - ATMP2));
                VENQI = ((NTU_AVSG * QTMP0) + ((NTU_BVSG * BTMP) * QTMP3));
                VENFI = VENQI;
                VENAI = (NTU_AVSG + ((NTU_BVSG * BTMP) * ATMP3));
            }
        }
        if (((NTU_ICE_SHAPE == 1) && (AFAI <= 20.0f))) {
            QTMP0 = NTU_EXPF(((ntu_gamln((AFAI + 2.0f)) - GI1) - NTU_LOGF(LAMI)));
            ATMP0 = NTU_EXPF(((ntu_gamln((AFAI + 1.0f)) - GI1) - NTU_LOGF(LAMI)));
            if ((BEST <= 1.0f)) {
                BTMP = (ntu_sq(SCN) * ((AVI * RHOAJ) / MUA));
                QTMP2 = (LLMI * (2.0f + BVI));
                QTMP3 = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 3.0f)) - GI1) - QTMP2));
                ATMP2 = (LLMI * (1.0f + BVI));
                ATMP3 = NTU_EXPF(((ntu_gamln(((BVI + AFAI) + 2.0f)) - GI1) - ATMP2));
                VENQI0 = ((NTU_AVIS * QTMP0) + ((NTU_BVIS * BTMP) * QTMP3));
                VENAI0 = ((NTU_AVIS * ATMP0) + ((NTU_BVIS * BTMP) * ATMP3));
            } else if ((BEST > 1.0f)) {
                BTMP = (SCN * sqrtf(((AVI * RHOAJ) / MUA)));
                QTMP2 = (LLMI * (1.5f + (BVI / 2.0f)));
                QTMP3 = NTU_EXPF(((ntu_gamln((((BVI / 2.0f) + AFAI) + 2.5f)) - GI1) - QTMP2));
                ATMP2 = (LLMI * (0.5f + (BVI / 2.0f)));
                ATMP3 = NTU_EXPF(((ntu_gamln((((BVI / 2.0f) + AFAI) + 1.5f)) - GI1) - ATMP2));
                VENQI0 = ((NTU_AVSG * QTMP0) + ((NTU_BVSG * BTMP) * QTMP3));
                VENAI0 = ((NTU_AVSG * ATMP0) + ((NTU_BVSG * BTMP) * ATMP3));
            }
        }
    } else {
        ADAGR = 1.0f;
        ZETA = 0.0f;
    }
    if ((QS1D >= NTU_QSMALL)) {
        ntu_solve_afas(TK1D, RHO, QS1D, QC1D, NS1D, VS1D, FS1D, AS1D, AFAS, LAMS, MVDS, RHOS, SASPR, AMS, AVS, BVS);
        LLMS = NTU_LOGF(LAMS);
        BSTMP = (SCN * sqrtf(((AVS * RHOAJ) / MUA)));
        QTMP1 = (LLMS * (1.5f + (BVS / 2.0f)));
        QTMP2 = NTU_EXPF(((ntu_gamln((AFAS + 2.0f)) - ntu_gamln((AFAS + 1.0f))) - NTU_LOGF(LAMS)));
        QTMP3 = NTU_EXPF(((ntu_gamln((((BVS / 2.0f) + AFAS) + 2.5f)) - ntu_gamln((AFAS + 1.0f))) - QTMP1));
        ATMP1 = (LLMS * (0.5f + (BVS / 2.0f)));
        ATMP2 = NTU_EXPF(((ntu_gamln((((BVS / 2.0f) + AFAS) + 1.5f)) - ntu_gamln((AFAS + 1.0f))) - ATMP1));
        CAPS = ((ZP1 * NTU_POWF(SASPR, (ZP2 / 3.0f))) + (ZP3 * NTU_POWF(SASPR, (ZP4 / 3.0f))));
        VENQS = (((NTU_AVSG * QTMP2) * CAPS) + (((NTU_BVSG * BSTMP) * QTMP3) * CAPS));
        VENAS = ((NTU_AVSG * CAPS) + (((NTU_BVSG * BSTMP) * ATMP2) * CAPS));
    }
    if ((QG1D >= NTU_QSMALL)) {
        ntu_solve_afag(TK1D, RHO, QG1D, QC1D, NG1D, VG1D, AG1D, LAMG, AFAG, MVDG, RHOG, AMG, AVG, BVG);
        LLMG = NTU_LOGF(LAMG);
        BGTMP = (SCN * sqrtf(((AVG * RHOAJ) / MUA)));
        QTMP1 = (LLMG * (1.5f + (BVG / 2.0f)));
        QTMP2 = NTU_EXPF(((ntu_gamln((AFAG + 2.0f)) - ntu_gamln((AFAG + 1.0f))) - NTU_LOGF(LAMG)));
        QTMP3 = NTU_EXPF(((ntu_gamln((((BVG / 2.0f) + AFAG) + 2.5f)) - ntu_gamln((AFAG + 1.0f))) - QTMP1));
        ATMP1 = (LLMG * (0.5f + (BVG / 2.0f)));
        ATMP2 = NTU_EXPF(((ntu_gamln((((BVG / 2.0f) + AFAG) + 1.5f)) - ntu_gamln((AFAG + 1.0f))) - ATMP1));
        VENQG = ((NTU_AVSG * QTMP2) + ((NTU_BVSG * BGTMP) * QTMP3));
        VENAG = (NTU_AVSG + ((NTU_BVSG * BGTMP) * ATMP2));
    }
    if ((QH1D >= NTU_QSMALL)) {
        ntu_solve_afah(TK1D, RHO, QH1D, NH1D, AH1D, LAMH, AFAH, MVDH, AVH, BVH);
        LLMH = NTU_LOGF(LAMH);
        BHTMP = (SCN * sqrtf(((AVH * RHOAJ) / MUA)));
        QTMP1 = (LLMH * (1.5f + (BVH / 2.0f)));
        QTMP2 = NTU_EXPF(((ntu_gamln((AFAH + 2.0f)) - ntu_gamln((AFAH + 1.0f))) - NTU_LOGF(LAMH)));
        QTMP3 = NTU_EXPF(((ntu_gamln((((BVH / 2.0f) + AFAH) + 2.5f)) - ntu_gamln((AFAH + 1.0f))) - QTMP1));
        QTMP4 = (LLMH * (2.0f + BVH));
        QTMP5 = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 3.0f)) - ntu_gamln((AFAH + 1.0f))) - QTMP4));
        ATMP1 = (LLMH * (0.5f + (BVH / 2.0f)));
        ATMP2 = NTU_EXPF(((ntu_gamln((((BVH / 2.0f) + AFAH) + 1.5f)) - ntu_gamln((AFAH + 1.0f))) - ATMP1));
        ATMP3 = (LLMH * (1.0f + BVH));
        ATMP4 = NTU_EXPF(((ntu_gamln(((BVH + AFAH) + 2.0f)) - ntu_gamln((AFAH + 1.0f))) - ATMP3));
        VENQH = ((QTMP2 + ((NTU_VENH1 * BHTMP) * QTMP3)) + ((NTU_VENH2 * ntu_sq(BHTMP)) * QTMP5));
        VENAH = ((1.0f + ((NTU_VENH1 * BHTMP) * ATMP2)) + ((NTU_VENH2 * ntu_sq(BHTMP)) * ATMP4));
    }
    if ((QC1D >= NTU_QSMALL)) {
        if ((SSRW > NTU_RSMALL)) {
            QVDVC = ((SSRW * NTU_EXPF(((AQ1 + (BQ1 * NTU_LOGF(NC1D))) + (CQ1 * LMVRC)))) / ABW);
        } else if ((SSRW < (-(1.0f * NTU_RSMALL)))) {
            QEVCV = ntu_fmin(((SSRW * NTU_EXPF(((AQ1 + (BQ1 * NTU_LOGF(NC1D))) + (CQ1 * LMVRC)))) / ABW), 0.0f);
            NACCV = (-(1.0f * NTU_EXPF((((AN9 + (BN9 * NTU_LOGF(NC1D))) + (CN9 * LMVRC)) + (DN9 * NTU_LOGF((-SSRW)))))));
            QACCV = (NACCV * 4.1888E-15f);
            QACCV = ntu_fmax(QACCV, (-((1.0f * QC1D) * IDT)));
            NACCV = ntu_fmax(NACCV, (-((1.0f * NC1D) * IDT)));
        }
    }
    if ((QR1D >= NTU_QSMALL)) {
        if ((SSRW > NTU_RSMALL)) {
            QVDVR = (((SSRW * NR1D) * NTU_EXPF((AQ2 + ((BQ2 + (CQ2 * LMVRR)) * ntu_sq(LMVRR))))) / ABW);
        } else if ((SSRW < (-(1.0f * NTU_RSMALL)))) {
            QEVRV = (((SSRW * NR1D) * NTU_EXPF((AQ2 + ((BQ2 + (CQ2 * LMVRR)) * ntu_sq(LMVRR))))) / ABW);
            NACRC = ((SSRW * NTU_EXPF(((AN10 + (BN10 * NTU_LOGF(NR1D))) + (CN10 * LMVRR)))) / ABW);
            QACRC = (NACRC * 5.236E-10f);
            QEVRV = ntu_fmin(QEVRV, 0.0f);
            QACRC = ntu_fmax(QACRC, (-((1.0f * QR1D) * IDT)));
            NACRC = ntu_fmax(NACRC, (-((1.0f * NR1D) * IDT)));
        }
    }
    if ((TK1D < NTU_TK0C)) {
        if ((QI1D >= NTU_QSMALL)) {
            DNIVD = (NTU_RHOI0 * NTU_EXPF((-((3.0f * ntu_fmax(((QV1D - QVSI) - 5.0E-5f), 0.0f)) / INHGR))));
            DNIVD = ntu_fmin(ntu_fmax(DNIVD, NTU_RHOIMIN), NTU_RHOIMAX);
            if (((NTU_ICE_SHAPE == 1) && (AFAI <= 20.0f))) {
                RAT1 = ntu_fmin(1.0f, ntu_fmax(0.0f, fabsf(ntu_gammp((AFAI + 1.0f), (NTU_DI0 * LAMI)))));
                RAT2 = (1.0f - RAT1);
                QVDVI = (((((2.0f * NTU_PI) * NI1D) * ((RAT2 * VENQI) + (RAT1 * VENQI0))) * SSRI) / ABI);
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FVDVI = (((((12.0f * NI1D) * (((RAT2 * VENFI) * GVHAB) + (RAT1 * VENQI0))) * SSRI) / ABI) / DNIVD);
                    IVDVI = (((((12.0f * NI1D) * ((RAT2 * VENQI) + (RAT1 * VENQI0))) * SSRI) / ABI) / DNIVD);
                }
                if ((AI1D >= NTU_ASMALL)) {
                    AVDVI = (((((8.0f * NI1D) * ((RAT2 * VENAI) + (RAT1 * VENAI0))) * SSRI) / ABI) / DNIVD);
                }
            } else {
                QVDVI = (((((2.0f * NTU_PI) * NI1D) * VENQI) * SSRI) / ABI);
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FVDVI = ((((((12.0f * NI1D) * GVHAB) * VENFI) * SSRI) / ABI) / DNIVD);
                    IVDVI = (((((12.0f * NI1D) * VENQI) * SSRI) / ABI) / DNIVD);
                }
                if ((AI1D >= NTU_ASMALL)) {
                    AVDVI = (((((8.0f * NI1D) * VENAI) * SSRI) / ABI) / DNIVD);
                }
            }
            VVDVI = (QVDVI / DNIVD);
            if ((QVDVI < 0.0f)) {
                QSBIV = ntu_fmin(QVDVI, 0.0f);
                VSBIV = ntu_fmin((QVDVI / RHOI), 0.0f);
                QVDVI = 0.0f;
                VVDVI = 0.0f;
                if ((AI1D >= NTU_ASMALL)) {
                    ASBIV = ntu_fmin(((AVDVI * DNIVD) / RHOI), 0.0f);
                    AVDVI = 0.0f;
                }
                if (((I3M1D >= NTU_ISMALL) && (FI1D >= NTU_ISMALL))) {
                    FSBIV = ntu_fmin(((FVDVI * DNIVD) / RHOI), 0.0f);
                    ISBIV = ntu_fmin(((IVDVI * DNIVD) / RHOI), 0.0f);
                    FVDVI = 0.0f;
                    IVDVI = 0.0f;
                }
            }
        }
        if ((QS1D >= NTU_QSMALL)) {
            DNSVD = (RHOS * NTU_EXPF((-((3.0f * ntu_fmax(((QV1D - QVSI) - 5.0E-5f), 0.0f)) / 1.0f))));
            DNSVD = ntu_fmin(ntu_fmax(DNSVD, NTU_RHOIMIN), NTU_RHOIMAX);
            QVDVS = (((((2.0f * NTU_PI) * NS1D) * VENQS) * SSRI) / ABI);
            VVDVS = (QVDVS / DNSVD);
            FVDVS = (((QVDVS / DNSVD) * SASPR) * NTU_V2M3);
            if ((AS1D >= NTU_ASMALL)) {
                AVDVS = (((((8.0f * NS1D) * VENAS) * SSRI) / ABI) / DNSVD);
            }
            if ((QVDVS < 0.0f)) {
                QSBSV = ntu_fmin(QVDVS, 0.0f);
                VSBSV = ntu_fmin((QVDVS / RHOS), 0.0f);
                FSBSV = ntu_fmin(((FVDVS * DNSVD) / RHOS), 0.0f);
                QVDVS = 0.0f;
                VVDVS = 0.0f;
                FVDVS = 0.0f;
                if ((AS1D >= NTU_ASMALL)) {
                    ASBSV = ntu_fmin(((AVDVS * DNSVD) / RHOS), 0.0f);
                    AVDVS = 0.0f;
                }
            }
        }
        if ((QG1D >= NTU_QSMALL)) {
            DNGVD = (RHOG * NTU_EXPF((-(3.0f * ntu_fmax(((QV1D - QVSI) - 5.0E-5f), 0.0f)))));
            DNGVD = ntu_fmin(ntu_fmax(DNGVD, NTU_RHOIMIN), NTU_RHOG0);
            QVDVG = (((((2.0f * NTU_PI) * NG1D) * VENQG) * SSRI) / ABI);
            VVDVG = (QVDVG / DNGVD);
            if ((AG1D >= NTU_ASMALL)) {
                AVDVG = (((((8.0f * NG1D) * VENAG) * SSRI) / ABI) / DNGVD);
            }
            if ((QVDVG < 0.0f)) {
                QSBGV = ntu_fmin(QVDVG, 0.0f);
                QVDVG = 0.0f;
                VSBGV = ntu_fmin((QVDVG / RHOG), 0.0f);
                VVDVG = 0.0f;
                if ((AG1D >= NTU_ASMALL)) {
                    ASBGV = ntu_fmin(((AVDVG * DNGVD) / RHOG), 0.0f);
                    AVDVG = 0.0f;
                }
            }
        }
        if ((QH1D >= NTU_QSMALL)) {
            QVDVH = (((((2.0f * NTU_PI) * NH1D) * VENQH) * SSRI) / ABI);
            if ((AH1D >= NTU_ASMALL)) {
                AVDVH = (((((8.0f * NH1D) * VENAH) * SSRI) / ABI) / NTU_RHOH);
            }
            if ((QVDVH < 0.0f)) {
                QSBHV = ntu_fmin(QVDVH, 0.0f);
                QVDVH = 0.0f;
                if ((AH1D >= NTU_ASMALL)) {
                    ASBHV = ntu_fmin(AVDVH, 0.0f);
                    AVDVH = 0.0f;
                }
            }
        }
        if (((((QSBIV + QSBSV) + QSBGV) + QSBHV) < 0.0f)) {
            SBMAX = (((QV1D - QVSI) / (1.0f + ((ntu_sq(XXLS) * QV1D) / ((CPM * NTU_RV) * ntu_sq(TK1D))))) * IDT);
            SUMSUB = (((QSBIV + QSBSV) + QSBGV) + QSBHV);
            if (((SBMAX < 0.0f) && (SUMSUB < (SBMAX * 0.9999f)))) {
                QSBIV = (QSBIV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                QSBSV = (QSBSV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                QSBGV = (QSBGV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                QSBHV = (QSBHV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                VSBIV = (VSBIV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                VSBSV = (VSBSV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                VSBGV = (VSBGV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                FSBIV = (FSBIV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                FSBSV = (FSBSV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                ISBIV = (ISBIV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                ASBIV = (ASBIV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                ASBSV = (ASBSV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                ASBGV = (ASBGV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
                ASBHV = (ASBHV * ntu_fmin(1.0f, ((0.9999f * SBMAX) / SUMSUB)));
            }
        }
    } else {
        if (((QS1D >= NTU_QSMALL) && (SSRW < (-(1.0f * NTU_RSMALL))))) {
            QEVSV = (((((2.0f * NTU_PI) * NS1D) * VENQS) * SSRW) / ABW);
            QEVSV = ntu_fmax(ntu_fmin(QEVSV, 0.0f), (-((1.0f * QS1D) * IDT)));
            VEVSV = ntu_fmax(ntu_fmin((QEVSV / RHOS), 0.0f), (-((1.0f * VS1D) * IDT)));
            FEVSV = ntu_fmax((((QEVSV / RHOS) * SASPR) * NTU_V2M3), (-((1.0f * FS1D) * IDT)));
            if ((AS1D >= NTU_ASMALL)) {
                AEVSV = (((((8.0f * NS1D) * VENAS) * SSRW) / ABW) / RHOS);
                AEVSV = ntu_fmax(ntu_fmin(AEVSV, 0.0f), (-((1.0f * AS1D) * IDT)));
            }
        }
        if (((QG1D >= NTU_QSMALL) && (SSRW < (-(1.0f * NTU_RSMALL))))) {
            QEVGV = (((((2.0f * NTU_PI) * NG1D) * VENQG) * SSRW) / ABW);
            QEVGV = ntu_fmax(ntu_fmin(QEVGV, 0.0f), (-((1.0f * QG1D) * IDT)));
            VEVGV = ntu_fmax(ntu_fmin((QEVGV / RHOG), 0.0f), (-((1.0f * VG1D) * IDT)));
            if ((AG1D >= NTU_ASMALL)) {
                AEVGV = (((((8.0f * NG1D) * VENAG) * SSRW) / ABW) / RHOG);
                AEVGV = ntu_fmax(ntu_fmin(AEVGV, 0.0f), (-((1.0f * AG1D) * IDT)));
            }
        }
        if (((QH1D >= NTU_QSMALL) && (SSRW < (-(1.0f * NTU_RSMALL))))) {
            QEVHV = (((((2.0f * NTU_PI) * NH1D) * VENQH) * SSRW) / ABW);
            QEVHV = ntu_fmax(ntu_fmin(QEVHV, 0.0f), (-((1.0f * QH1D) * IDT)));
            if ((AH1D >= NTU_ASMALL)) {
                AEVHV = (((((8.0f * NH1D) * VENAH) * SSRW) / ABW) / NTU_RHOH);
                AEVHV = ntu_fmax(ntu_fmin(AEVHV, 0.0f), (-((1.0f * AH1D) * IDT)));
            }
        }
        if ((((((QEVCV + QEVRV) + QEVSV) + QEVGV) + QEVHV) < 0.0f)) {
            EVMAX = (((QV1D - QVSW) / (1.0f + ((ntu_sq(XXLV) * QV1D) / ((CPM * NTU_RV) * ntu_sq(TK1D))))) * IDT);
            SUMEVP = ((((QEVCV + QEVRV) + QEVSV) + QEVGV) + QEVHV);
            if (((EVMAX < 0.0f) && (SUMEVP < (EVMAX * 0.9999f)))) {
                QEVCV = (QEVCV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                QEVRV = (QEVRV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                QEVSV = (QEVSV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                QEVGV = (QEVGV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                QEVHV = (QEVHV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                VEVSV = (VEVSV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                VEVGV = (VEVGV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                FEVSV = (FEVSV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                AEVSV = (AEVSV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                AEVGV = (AEVGV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
                AEVHV = (AEVHV * ntu_fmin(1.0f, ((0.9999f * EVMAX) / SUMEVP)));
            }
        }
    }
    QVWTEND = ((((((((-QACCV) - QVDVC) - QVDVR) - QEVCV) - QEVRV) - QEVSV) - QEVGV) - QEVHV);
    QVITEND = ((((((((-QVDVI) - QVDVS) - QVDVG) - QVDVH) - QSBIV) - QSBSV) - QSBGV) - QSBHV);
    QVTEND = (QVWTEND + QVITEND);
    QCTEND = (((QACCV - QACRC) + QVDVC) + QEVCV);
    QRTEND = ((QACRC + QVDVR) + QEVRV);
    QITEND = (QVDVI + QSBIV);
    QSTEND = ((QVDVS + QEVSV) + QSBSV);
    QGTEND = ((QVDVG + QEVGV) + QSBGV);
    QHTEND = ((QVDVH + QEVHV) + QSBHV);
    DTS = ntu_fmin(ntu_fmin(ntu_fmin(ntu_fmin(ntu_fmin(ntu_fmin(ntu_fmin(ntu_fmin(ntu_fmin(((TORR * ntu_fmax(QV1D, 1.0E-7f)) / ntu_fmax(fabsf(QVTEND), NTU_QSMALL)), ((TORR * ntu_fmax(QC1D, NTU_QLIMIT)) / ntu_fmax(fabsf(QCTEND), NTU_QSMALL))), ((TORR * ntu_fmax(QR1D, NTU_QLIMIT)) / ntu_fmax(fabsf(QRTEND), NTU_QSMALL))), ((TORR * ntu_fmax(QI1D, NTU_QLIMIT)) / ntu_fmax(fabsf(QITEND), NTU_QSMALL))), ((TORR * ntu_fmax(QS1D, NTU_QLIMIT)) / ntu_fmax(fabsf(QSTEND), NTU_QSMALL))), ((TORR * ntu_fmax(QG1D, NTU_QLIMIT)) / ntu_fmax(fabsf(QGTEND), NTU_QSMALL))), ((TORR * ntu_fmax(QH1D, NTU_QLIMIT)) / ntu_fmax(fabsf(QHTEND), NTU_QSMALL))), DT), DTS), (DT - SDTS));
    DTS = ntu_fmin((DT - SDTS), ntu_fmax(NTU_DTMIN, DTS));
    DTSI = (-((QVWTEND * ELCLD) + (QVITEND * ELDLD)));
    DTSW = (-((QVWTEND * ELCLC) + (QVITEND * ELCLD)));
    if ((fabsf(DTSI) < NTU_RLIMIT)) {
        DTSI = 1.0E5f;
    } else {
        DTSI = ((SSRI * QV1D) / ((1.0f + SSRI) * DTSI));
    }
    if ((fabsf(DTSW) < NTU_RLIMIT)) {
        DTSW = 1.0E5f;
    } else {
        DTSW = ((SSRW * QV1D) / ((1.0f + SSRW) * DTSW));
    }
    if ((TK1D > NTU_TK0C)) {
        if ((DTSW > 0.0f)) {
            DTS = ntu_fmax(ntu_fmin(DTS, DTSW), NTU_DTMIN);
        }
    } else {
        if ((DTSI > 0.0f)) {
            DTS = ntu_fmax(ntu_fmin(DTS, DTSI), NTU_DTMIN);
        }
        if ((DTSW > 0.0f)) {
            DTS = ntu_fmax(ntu_fmin(DTS, DTSW), NTU_DTMIN);
        }
    }
    QVSOUR = (QV1D + (((((((((((-QACCV) - QEVCV) - QEVRV) - QEVSV) - QEVGV) - QEVHV) - QSBIV) - QSBSV) - QSBGV) - QSBHV) * DTS));
    QVSINK = ((((((QVDVC + QVDVR) + QVDVI) + QVDVS) + QVDVG) + QVDVH) * DTS);
    if (((QVSINK > QVSOUR) && (QVSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QVSOUR / (QVSINK + NTU_QSMALL)));
        QVDVC = (QVDVC * RATIO);
        QVDVR = (QVDVR * RATIO);
        QVDVI = (QVDVI * RATIO);
        QVDVS = (QVDVS * RATIO);
        QVDVG = (QVDVG * RATIO);
        QVDVH = (QVDVH * RATIO);
    }
    QCSOUR = (QC1D + ((QVDVC - QACRC) * DTS));
    QCSINK = (((-QACCV) - QEVCV) * DTS);
    if (((QCSINK > QCSOUR) && (QCSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QCSOUR / (QCSINK + NTU_QSMALL)));
        QACCV = (QACCV * RATIO);
        QEVCV = (QEVCV * RATIO);
    }
    QRSOUR = (QR1D + (QVDVR * DTS));
    QRSINK = (((-QACRC) - QEVRV) * DTS);
    if (((QRSINK > QRSOUR) && (QRSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QRSOUR / (QRSINK + NTU_QSMALL)));
        QACRC = (QACRC * RATIO);
        QEVRV = (QEVRV * RATIO);
    }
    QISOUR = (QI1D + (QVDVI * DTS));
    QISINK = ((-QSBIV) * DTS);
    if (((QISINK > QISOUR) && (QISOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QISOUR / (QISINK + NTU_QSMALL)));
        QSBIV = (QSBIV * RATIO);
    }
    QSSOUR = (QS1D + (QVDVS * DTS));
    QSSINK = (((-QEVSV) - QSBSV) * DTS);
    if (((QSSINK > QSSOUR) && (QSSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QSSOUR / (QSSINK + NTU_QSMALL)));
        QSBSV = (QSBSV * RATIO);
        QEVSV = (QEVSV * RATIO);
    }
    QGSOUR = (QG1D + (QVDVG * DTS));
    QGSINK = (((-QSBGV) - QEVGV) * DTS);
    if (((QGSINK > QGSOUR) && (QGSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QGSOUR / (QGSINK + NTU_QSMALL)));
        QSBGV = (QSBGV * RATIO);
        QEVGV = (QEVGV * RATIO);
    }
    QHSOUR = (QH1D + (QVDVH * DTS));
    QHSINK = (((-QSBHV) - QEVHV) * DTS);
    if (((QHSINK > QHSOUR) && (QHSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (QHSOUR / (QHSINK + NTU_QSMALL)));
        QSBHV = (QSBHV * RATIO);
        QEVHV = (QEVHV * RATIO);
    }
    if ((QI1D >= NTU_QSMALL)) {
        NSBIV = ntu_fmin(((QSBIV * NI1D) / QI1D), 0.0f);
    }
    if ((QS1D >= NTU_QSMALL)) {
        NSBSV = ntu_fmin(((QSBSV * NS1D) / QS1D), 0.0f);
    }
    if ((QG1D >= NTU_QSMALL)) {
        NSBGV = ntu_fmin(((QSBGV * NG1D) / QG1D), 0.0f);
    }
    if ((QH1D >= NTU_QSMALL)) {
        NSBHV = ntu_fmin(((QSBHV * NH1D) / QH1D), 0.0f);
    }
    NCSOUR = (NC1D + ((-NACRC) * DTS));
    NCSINK = ((-NACCV) * DTS);
    if (((NCSINK > NCSOUR) && (NCSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NCSOUR / (NCSINK + NTU_NSMALL)));
        NACCV = (NACCV * RATIO);
    }
    NISOUR = NI1D;
    NISINK = ((-NSBIV) * DTS);
    if (((NISINK > NISOUR) && (NISOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NISOUR / (NISINK + NTU_NSMALL)));
        NSBIV = (NSBIV * RATIO);
    }
    NSSOUR = NS1D;
    NSSINK = ((-NSBSV) * DTS);
    if (((NSSINK > NSSOUR) && (NSSOUR >= NTU_NSMALL))) {
        RATIO = ntu_fmin(1.0f, (NSSOUR / (NSSINK + NTU_NSMALL)));
        NSBSV = (NSBSV * RATIO);
    }
    VISOUR = (VI1D + (VVDVI * DTS));
    VISINK = ((-VSBIV) * DTS);
    if (((VISINK > VISOUR) && (VISOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (VISOUR / (VISINK + NTU_ISMALL)));
        VSBIV = (VSBIV * RATIO);
    }
    VSSOUR = (VS1D + (VVDVS * DTS));
    VSSINK = (((-VEVSV) - VSBSV) * DTS);
    if (((VSSINK > VSSOUR) && (VSSOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (VSSOUR / (VSSINK + NTU_ISMALL)));
        VSBSV = (VSBSV * RATIO);
        VEVSV = (VEVSV * RATIO);
    }
    VGSOUR = (VG1D + (VVDVG * DTS));
    VGSINK = (((-VSBGV) - VEVGV) * DTS);
    if (((VGSINK > VGSOUR) && (VGSOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (VGSOUR / (VGSINK + NTU_ISMALL)));
        VSBGV = (VSBGV * RATIO);
        VEVGV = (VEVGV * RATIO);
    }
    FISOUR = (FI1D + (FVDVI * DTS));
    FISINK = ((-FSBIV) * DTS);
    if (((FISINK > FISOUR) && (FISOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (FISOUR / (FISINK + NTU_ISMALL)));
        FSBIV = (FSBIV * RATIO);
    }
    FSSOUR = (FS1D + (FVDVS * DTS));
    FSSINK = (((-FEVSV) - FSBSV) * DTS);
    if (((FSSINK > FSSOUR) && (FSSOUR >= NTU_QSMALL))) {
        RATIO = ntu_fmin(1.0f, (FSSOUR / (FSSINK + NTU_QSMALL)));
        FSBSV = (FSBSV * RATIO);
        FEVSV = (FEVSV * RATIO);
    }
    AISOUR = (AI1D + (AVDVI * DTS));
    AISINK = ((-ASBIV) * DTS);
    if (((AISINK > AISOUR) && (AISOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (AISOUR / (AISINK + NTU_ASMALL)));
        ASBIV = (ASBIV * RATIO);
    }
    ASSOUR = (AS1D + (AVDVS * DTS));
    ASSINK = (((-ASBSV) - AEVSV) * DTS);
    if (((ASSINK > ASSOUR) && (ASSOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (ASSOUR / (ASSINK + NTU_ASMALL)));
        ASBSV = (ASBSV * RATIO);
        AEVSV = (AEVSV * RATIO);
    }
    AGSOUR = (AG1D + (AVDVG * DTS));
    AGSINK = (((-ASBGV) - AEVGV) * DTS);
    if (((AGSINK > AGSOUR) && (AGSOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (AGSOUR / (AGSINK + NTU_ASMALL)));
        ASBGV = (ASBGV * RATIO);
        AEVGV = (AEVGV * RATIO);
    }
    AHSOUR = (AH1D + (AVDVH * DTS));
    AHSINK = (((-ASBHV) - AEVHV) * DTS);
    if (((AHSINK > AHSOUR) && (AHSOUR >= NTU_ASMALL))) {
        RATIO = ntu_fmin(1.0f, (AHSOUR / (AHSINK + NTU_ASMALL)));
        ASBHV = (ASBHV * RATIO);
        AEVHV = (AEVHV * RATIO);
    }
    IISOUR = (I3M1D + (IVDVI * DTS));
    IISINK = ((-ISBIV) * DTS);
    if (((IISINK > IISOUR) && (IISOUR >= NTU_ISMALL))) {
        RATIO = ntu_fmin(1.0f, (IISOUR / (IISINK + NTU_ISMALL)));
        ISBIV = (ISBIV * RATIO);
    }
    QV1D = ntu_fmax(0.0f, (QV1D + (((((((((((((((((-QACCV) - QVDVC) - QVDVR) - QEVCV) - QEVRV) - QEVSV) - QEVGV) - QEVHV) - QVDVI) - QVDVS) - QVDVG) - QVDVH) - QSBIV) - QSBSV) - QSBGV) - QSBHV) * DTS)));
    QC1D = ntu_fmax(0.0f, (QC1D + ((((QACCV - QACRC) + QVDVC) + QEVCV) * DTS)));
    QR1D = ntu_fmax(0.0f, (QR1D + (((QACRC + QVDVR) + QEVRV) * DTS)));
    QI1D = ntu_fmax(0.0f, (QI1D + ((QVDVI + QSBIV) * DTS)));
    QS1D = ntu_fmax(0.0f, (QS1D + (((QVDVS + QSBSV) + QEVSV) * DTS)));
    QG1D = ntu_fmax(0.0f, (QG1D + (((QVDVG + QSBGV) + QEVGV) * DTS)));
    QH1D = ntu_fmax(0.0f, (QH1D + (((QVDVH + QSBHV) + QEVHV) * DTS)));
    NC1D = ntu_fmax(0.0f, (NC1D + ((NACCV - NACRC) * DTS)));
    NR1D = ntu_fmax(0.0f, (NR1D + (NACRC * DTS)));
    NI1D = ntu_fmax(0.0f, (NI1D + (NSBIV * DTS)));
    NS1D = ntu_fmax(0.0f, (NS1D + (NSBSV * DTS)));
    NG1D = ntu_fmax(0.0f, (NG1D + (NSBGV * DTS)));
    NH1D = ntu_fmax(0.0f, (NH1D + (NSBHV * DTS)));
    VI1D = ntu_fmax(0.0f, (VI1D + ((VVDVI + VSBIV) * DTS)));
    VS1D = ntu_fmax(0.0f, (VS1D + (((VVDVS + VSBSV) + VEVSV) * DTS)));
    VG1D = ntu_fmax(0.0f, (VG1D + (((VVDVG + VSBGV) + VEVGV) * DTS)));
    FI1D = ntu_fmax(0.0f, (FI1D + ((FVDVI + FSBIV) * DTS)));
    FS1D = ntu_fmax(0.0f, (FS1D + (((FVDVS + FSBSV) + FEVSV) * DTS)));
    I3M1D = ntu_fmax(0.0f, (I3M1D + ((IVDVI + ISBIV) * DTS)));
    AI1D = ntu_fmax(0.0f, (AI1D + ((AVDVI + ASBIV) * DTS)));
    AS1D = ntu_fmax(0.0f, (AS1D + (((AVDVS + ASBSV) + AEVSV) * DTS)));
    AG1D = ntu_fmax(0.0f, (AG1D + (((AVDVG + ASBGV) + AEVGV) * DTS)));
    AH1D = ntu_fmax(0.0f, (AH1D + (((AVDVH + ASBHV) + AEVHV) * DTS)));
    XDNC = (XDNC + (NACCV * DTS));
    XDNR = (XDNR + (NACRC * DTS));
    GQCTR = (QACRC * DTS);
    TK1D = (TK1D + ((((((((((((QACCV + QACRC) + QVDVC) + QVDVR) + QEVCV) + QEVRV) + QEVSV) + QEVGV) + QEVHV) * XXLV) + ((((((((QVDVI + QVDVS) + QVDVG) + QVDVH) + QSBIV) + QSBSV) + QSBGV) + QSBHV) * XXLS)) / CPM) * DTS));
    if (((QC1D >= NTU_QSMALL) && (NC1D >= NTU_NSMALL))) {
        MVDC = NTU_POWF(((QC1D * NTU_IAMW) / NC1D), NTU_THRD);
        if ((MVDC > NTU_DCR)) {
            QR1D = (QR1D + QC1D);
            NR1D = (NR1D + NC1D);
            QC1D = 0.0f;
            NC1D = 0.0f;
        }
    }
    if (((QR1D >= NTU_QSMALL) && (NR1D >= NTU_NSMALL))) {
        MVDR = NTU_POWF(((QR1D * NTU_IAMW) / NR1D), NTU_THRD);
        if ((MVDR < NTU_DCR)) {
            QC1D = (QC1D + QR1D);
            NC1D = (NC1D + NR1D);
            QR1D = 0.0f;
            NR1D = 0.0f;
        }
    }
    if (((QH1D >= NTU_QSMALL) && (NH1D >= NTU_NSMALL))) {
        MVDH = NTU_POWF(((QH1D * NTU_IAMH) / NH1D), NTU_THRD);
        if ((MVDH < NTU_DHMIN)) {
            QG1D = (QG1D + QH1D);
            NG1D = (NG1D + NH1D);
            VG1D = (VG1D + (QH1D / NTU_RHOG0));
            QH1D = 0.0f;
            NH1D = 0.0f;
            if (((AH1D >= NTU_ASMALL) && (NTU_AFAG_3M == 1))) {
                AG1D = (AG1D + AH1D);
                AH1D = 0.0f;
            }
        }
    }
}

// WRF SUBROUTINE NTU_MICRO (module_mp_ntu.F:1813-2686)
NTU_DEV_BIG void ntu_ntu_micro(NtuAeroConst* __restrict__ AC, float* TK3D, float* QV3D, float* DZ3D, float* P3D, float* S3D, float* W3D, float& SR, float& ICENCV, float& VINCV, float& VSNCV, float& VGNCV, float& FINCV, float& FSNCV, float& AINCV, float& ASNCV, float& AGNCV, float& AHNCV, float& I3MNCV, float& CLODNCV, float& RAINNC, float& RAINNCV, float& SNOWNC, float& SNOWNCV, float& GRAPNC, float& GRAPNCV, float& HAILNC, float& HAILNCV, float DT, float DTMN, float* QC3D, float* QR3D, float* QI3D, float* QS3D, float* QG3D, float* QH3D, float* NC3D, float* NR3D, float* NI3D, float* NS3D, float* NG3D, float* NH3D, float* VI3D, float* VS3D, float* VG3D, float* FI3D, float* FS3D, float* AI3D, float* AS3D, float* AG3D, float* AH3D, float* I3M3D, float* QDCN3D, float* QTCN3D, float* QCCN3D, float* QRCN3D, float* QNIN3D, int KTS, int KTE) {
    int HID[NTU_KMAX] = {};
    float TC3D[NTU_KMAX] = {};
    float TAIR[NTU_KMAX] = {};
    float PRES[NTU_KMAX] = {};
    float QVAP[NTU_KMAX] = {};
    float ADAGR[NTU_KMAX] = {};
    float AMI[NTU_KMAX] = {};
    float BMI[NTU_KMAX] = {};
    float AVI[NTU_KMAX] = {};
    float BVI[NTU_KMAX] = {};
    float AMS[NTU_KMAX] = {};
    float AVS[NTU_KMAX] = {};
    float BVS[NTU_KMAX] = {};
    float AMG[NTU_KMAX] = {};
    float AVG[NTU_KMAX] = {};
    float BVG[NTU_KMAX] = {};
    float AVH[NTU_KMAX] = {};
    float BVH[NTU_KMAX] = {};
    float INHGR[NTU_KMAX] = {};
    float DCLDMT[NTU_KMAX] = {};
    float DCLDMC[NTU_KMAX] = {};
    float DCLDMR[NTU_KMAX] = {};
    float DTAIR[NTU_KMAX] = {};
    float DPDT[NTU_KMAX] = {};
    float DQVAP[NTU_KMAX] = {};
    float QACAC[NTU_KMAX] = {};
    float QACAR[NTU_KMAX] = {};
    float VT_QC[NTU_KMAX] = {};
    float VT_QR[NTU_KMAX] = {};
    float VT_QI[NTU_KMAX] = {};
    float VT_QS[NTU_KMAX] = {};
    float VT_QG[NTU_KMAX] = {};
    float VT_QH[NTU_KMAX] = {};
    float VT_NC[NTU_KMAX] = {};
    float VT_NR[NTU_KMAX] = {};
    float VT_NI[NTU_KMAX] = {};
    float VT_NS[NTU_KMAX] = {};
    float VT_NG[NTU_KMAX] = {};
    float VT_NH[NTU_KMAX] = {};
    float VT_VI[NTU_KMAX] = {};
    float VT_VS[NTU_KMAX] = {};
    float VT_VG[NTU_KMAX] = {};
    float VT_FI[NTU_KMAX] = {};
    float VT_FS[NTU_KMAX] = {};
    float VT_AI[NTU_KMAX] = {};
    float VT_AS[NTU_KMAX] = {};
    float VT_AG[NTU_KMAX] = {};
    float VT_AH[NTU_KMAX] = {};
    float VTI3M[NTU_KMAX] = {};
    float TK0[NTU_KMAX] = {};
    float P40[NTU_KMAX] = {};
    float RHO[NTU_KMAX] = {};
    float CPM[NTU_KMAX] = {};
    float GQCTR[NTU_KMAX] = {};
    float ESW[NTU_KMAX] = {};
    float ESI[NTU_KMAX] = {};
    float QVSW[NTU_KMAX] = {};
    float QVSI[NTU_KMAX] = {};
    float SSRW[NTU_KMAX] = {};
    float SSRI[NTU_KMAX] = {};
    float XXLV[NTU_KMAX] = {};
    float XXLS[NTU_KMAX] = {};
    float XXLF[NTU_KMAX] = {};
    float XDNC[NTU_KMAX] = {};
    float XDNR[NTU_KMAX] = {};
    float IPF[NTU_KMAX] = {};
    float IPG[NTU_KMAX] = {};
    float I3M0[NTU_KMAX] = {};
    float ZETA[NTU_KMAX] = {};
    float BEST[NTU_KMAX] = {};
    float MVRC[NTU_KMAX] = {};
    float MVRR[NTU_KMAX] = {};
    float SIGC[NTU_KMAX] = {};
    float MNRC[NTU_KMAX] = {};
    float KDX[NTU_KMAX] = {};
    float BDR[NTU_KMAX] = {};
    float BDI[NTU_KMAX] = {};
    float BDS[NTU_KMAX] = {};
    float BDG[NTU_KMAX] = {};
    float BDH[NTU_KMAX] = {};
    float R3M3D[NTU_KMAX] = {};
    float S3M3D[NTU_KMAX] = {};
    float G3M3D[NTU_KMAX] = {};
    float H3M3D[NTU_KMAX] = {};
    float FDI[NTU_KMAX] = {};
    float FDS[NTU_KMAX] = {};
    float FDG[NTU_KMAX] = {};
    float FDH[NTU_KMAX] = {};
    float RHOI[NTU_KMAX] = {};
    float RHOS[NTU_KMAX] = {};
    float RHOG[NTU_KMAX] = {};
    float IASPR[NTU_KMAX] = {};
    float SASPR[NTU_KMAX] = {};
    float MVDC[NTU_KMAX] = {};
    float MVDR[NTU_KMAX] = {};
    float MVDI[NTU_KMAX] = {};
    float MVDS[NTU_KMAX] = {};
    float MVDG[NTU_KMAX] = {};
    float MVDH[NTU_KMAX] = {};
    float EFRC[NTU_KMAX] = {};
    float EFRR[NTU_KMAX] = {};
    float AFAC[NTU_KMAX] = {};
    float AFAR[NTU_KMAX] = {};
    float AFAI[NTU_KMAX] = {};
    float AFAS[NTU_KMAX] = {};
    float AFAG[NTU_KMAX] = {};
    float AFAH[NTU_KMAX] = {};
    float LAMC[NTU_KMAX] = {};
    float LAMR[NTU_KMAX] = {};
    float LAMI[NTU_KMAX] = {};
    float LAMS[NTU_KMAX] = {};
    float LAMG[NTU_KMAX] = {};
    float LAMH[NTU_KMAX] = {};
    float LTK[NTU_KMAX] = {};
    float LQC[NTU_KMAX] = {};
    float LQR[NTU_KMAX] = {};
    float LQI[NTU_KMAX] = {};
    float LQS[NTU_KMAX] = {};
    float LQG[NTU_KMAX] = {};
    float LQH[NTU_KMAX] = {};
    float MDI[NTU_KMAX] = {};
    float MDS[NTU_KMAX] = {};
    float MDG[NTU_KMAX] = {};
    float VTMEAN[NTU_KMAX * 3] = {};
    bool DO_FR[NTU_KMAX * 2] = {};
    float RX0[NTU_KMAX * 2] = {};
    float AERO[NTU_KMAX * 5] = {};
    float ABCD[4] = {};
    float ZCCNS[3 * NTU_KMAX * 2] = {};
    float SDTL = 0.0f, SDTS = 0.0f, DTL = 0.0f, DTS = 0.0f, CNDMAX = 0.0f, TMP = 0.0f, XMASS = 0.0f, TEMPC = 0.0f, TEMPR = 0.0f, TEMPT = 0.0f;
    int K = 0, A = 0, IV = 0, IV0 = 0, IV1 = 0, IV2 = 0, IV3 = 0, IV4 = 0, IM = 0, IT = 0, J = 0;
    { const int ntu_hi_k = KTE;
    for (K = KTS; K <= ntu_hi_k; ++K) {
        AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = ntu_fmax(0.0f, QDCN3D[(K) - (KTS)]);
        AERO[((K) - (KTS)) + NTU_KMAX * ((2) - (1))] = ntu_fmax(0.0f, QTCN3D[(K) - (KTS)]);
        AERO[((K) - (KTS)) + NTU_KMAX * ((3) - (1))] = ntu_fmax(0.0f, QCCN3D[(K) - (KTS)]);
        AERO[((K) - (KTS)) + NTU_KMAX * ((4) - (1))] = ntu_fmax(0.0f, QRCN3D[(K) - (KTS)]);
        AERO[((K) - (KTS)) + NTU_KMAX * ((5) - (1))] = ntu_fmax(0.0f, QNIN3D[(K) - (KTS)]);
        TC3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - NTU_TK0C);
        ESW[(K) - (KTS)] = ntu_fmin((0.99f * P3D[(K) - (KTS)]), ntu_polysvp(TK3D[(K) - (KTS)], 0));
        ESI[(K) - (KTS)] = ntu_fmin((0.99f * P3D[(K) - (KTS)]), ntu_polysvp(TK3D[(K) - (KTS)], 1));
        if ((ESI[(K) - (KTS)] > ESW[(K) - (KTS)])) {
            ESI[(K) - (KTS)] = ESW[(K) - (KTS)];
        }
        SSRW[(K) - (KTS)] = ((QV3D[(K) - (KTS)] / ((0.622f * ESW[(K) - (KTS)]) / (P3D[(K) - (KTS)] - ESW[(K) - (KTS)]))) - 1.0f);
        SSRI[(K) - (KTS)] = ((QV3D[(K) - (KTS)] / ((0.622f * ESI[(K) - (KTS)]) / (P3D[(K) - (KTS)] - ESI[(K) - (KTS)]))) - 1.0f);
        XXLV[(K) - (KTS)] = (3.1484E6f - (2370.0f * TK3D[(K) - (KTS)]));
        XXLS[(K) - (KTS)] = ((3.15E6f - (2370.0f * TK3D[(K) - (KTS)])) + 0.3337E6f);
        XXLF[(K) - (KTS)] = (2836310.8f - (3.1484E6f - (2370.0f * TK3D[(K) - (KTS)])));
        CPM[(K) - (KTS)] = (NTU_CP * (1.0f + (0.887f * QV3D[(K) - (KTS)])));
        HID[(K) - (KTS)] = ntu_imax(ntu_imin(ntu_nint((fabsf(TC3D[(K) - (KTS)]) / 0.25f)), 120), 0);
        if (((QC3D[(K) - (KTS)] >= NTU_QSMALL) && (NC3D[(K) - (KTS)] < NTU_NSMALL))) {
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQC[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QC3D[(K) - (KTS)])));
            NC3D[(K) - (KTS)] = NTU_EXPF(((((AC->DNC0 + (AC->DNC1 * LTK[(K) - (KTS)])) + (AC->DNC2 * ntu_sq(LTK[(K) - (KTS)]))) + (AC->DNC3 * NTU_POWF(LTK[(K) - (KTS)], 3.0f))) - (0.25f * LQC[(K) - (KTS)])));
            MVRC[(K) - (KTS)] = NTU_POWF(((QC3D[(K) - (KTS)] / NC3D[(K) - (KTS)]) / NTU_C4PI3W), NTU_THRD);
            MVRC[(K) - (KTS)] = ntu_fmin(ntu_fmax(MVRC[(K) - (KTS)], NTU_RCMIN), NTU_RCMAX);
            EFRC[(K) - (KTS)] = NTU_EXPF(((NTU_EFC1 + (NTU_EFC2 * NTU_LOGF(NC3D[(K) - (KTS)]))) + (NTU_EFC3 * NTU_LOGF(MVRC[(K) - (KTS)]))));
            KDX[(K) - (KTS)] = ntu_fmax(NTU_KCCMIN, ntu_fmin(NTU_KCCMAX, NTU_POWF((MVRC[(K) - (KTS)] / EFRC[(K) - (KTS)]), 3.0f)));
            KDX[(K) - (KTS)] = ntu_fmin(NTU_KCCMAX, ntu_fmax(NTU_KCCMIN, KDX[(K) - (KTS)]));
            AFAC[(K) - (KTS)] = ((((6.0f * KDX[(K) - (KTS)]) - 3.0f) + sqrtf(((8.0f * KDX[(K) - (KTS)]) + 1.0f))) / (2.0f - (2.0f * KDX[(K) - (KTS)])));
            AFAC[(K) - (KTS)] = ntu_fmin(ntu_fmax(AFAC[(K) - (KTS)], NTU_AFAMIN), NTU_AFAMAX);
        }
        if (((QR3D[(K) - (KTS)] >= NTU_QSMALL) && (NR3D[(K) - (KTS)] < NTU_NSMALL))) {
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQR[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QR3D[(K) - (KTS)])));
            NR3D[(K) - (KTS)] = NTU_EXPF((((((-5793.7852f) + (3191.1171f * LTK[(K) - (KTS)])) - (582.73279f * ntu_sq(LTK[(K) - (KTS)]))) + (35.346854f * NTU_POWF(LTK[(K) - (KTS)], 3.0f))) - (0.25f * LQR[(K) - (KTS)])));
            MVRR[(K) - (KTS)] = NTU_POWF(((QR3D[(K) - (KTS)] / NR3D[(K) - (KTS)]) / NTU_C4PI3W), NTU_THRD);
            MVRR[(K) - (KTS)] = ntu_fmin(NTU_RRMAX, ntu_fmax(NTU_RRMIN, MVRR[(K) - (KTS)]));
            EFRR[(K) - (KTS)] = NTU_EXPF(((NTU_EFR1 + (NTU_EFR2 * NTU_LOGF(NR3D[(K) - (KTS)]))) + (NTU_EFR3 * NTU_LOGF(MVRR[(K) - (KTS)]))));
            KDX[(K) - (KTS)] = ntu_fmax(NTU_KCRMIN, ntu_fmin(NTU_KCRMAX, NTU_POWF((MVRR[(K) - (KTS)] / EFRR[(K) - (KTS)]), 3.0f)));
            KDX[(K) - (KTS)] = ntu_fmin(NTU_KCRMAX, ntu_fmax(NTU_KCRMIN, KDX[(K) - (KTS)]));
            AFAR[(K) - (KTS)] = ((((6.0f * KDX[(K) - (KTS)]) - 3.0f) + sqrtf(((8.0f * KDX[(K) - (KTS)]) + 1.0f))) / (2.0f - (2.0f * KDX[(K) - (KTS)])));
            AFAR[(K) - (KTS)] = ntu_fmin(ntu_fmax(AFAR[(K) - (KTS)], NTU_AFAMIN), NTU_AFAMAX);
        }
        if (((QI3D[(K) - (KTS)] >= NTU_QSMALL) && (NI3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVSI[(K) - (KTS)] = ((0.622f * ESI[(K) - (KTS)]) / (P3D[(K) - (KTS)] - ESI[(K) - (KTS)]));
            INHGR[(K) - (KTS)] = NTU_ITBLE[HID[(K) - (KTS)]];
            RHOI[(K) - (KTS)] = (NTU_RHOI0 * NTU_EXPF((-((3.0f * ntu_fmax(((QV3D[(K) - (KTS)] - QVSI[(K) - (KTS)]) - 5.0E-5f), 0.0f)) / INHGR[(K) - (KTS)]))));
            VI3D[(K) - (KTS)] = (QI3D[(K) - (KTS)] / RHOI[(K) - (KTS)]);
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQI[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QI3D[(K) - (KTS)])));
            MDI[(K) - (KTS)] = (NTU_EXPF((((-3.2653646f) + (2.0539073f * LTK[(K) - (KTS)])) - (0.25f * LQI[(K) - (KTS)]))) / 1.0E3f);
            NI3D[(K) - (KTS)] = ((((1.0E9f * QI3D[(K) - (KTS)]) * NTU_V2M3) / RHOI[(K) - (KTS)]) / NTU_POWF(MDI[(K) - (KTS)], 3.0f));
            I3M3D[(K) - (KTS)] = ((QI3D[(K) - (KTS)] * NTU_V2M3) / RHOI[(K) - (KTS)]);
            ADAGR[(K) - (KTS)] = NTU_POWF(ntu_fmax(ntu_fmin(NTU_ITBLE[HID[(K) - (KTS)]], 2.0f), 0.5f), NTU_THRD);
            ZETA[(K) - (KTS)] = ((ADAGR[(K) - (KTS)] - 1.0f) / (ADAGR[(K) - (KTS)] + 2.0f));
            I3M0[(K) - (KTS)] = (NI3D[(K) - (KTS)] * NTU_K3);
            FI3D[(K) - (KTS)] = (NTU_POWF((I3M3D[(K) - (KTS)] / I3M0[(K) - (KTS)]), ZETA[(K) - (KTS)]) * I3M3D[(K) - (KTS)]);
            BDI[(K) - (KTS)] = (NTU_POWF((I3M3D[(K) - (KTS)] / NI3D[(K) - (KTS)]), NTU_THRD) * 1.0E3f);
            BDI[(K) - (KTS)] = ntu_fmin(ntu_fmax(BDI[(K) - (KTS)], (NTU_DIMIN * 1.0E3f)), (NTU_DIMAX * 1.0E3f));
            FDI[(K) - (KTS)] = (((((7.4015986E-2f + (0.79866676f * BDI[(K) - (KTS)])) - (9.4468892E-3f * NTU_LOGF(NI3D[(K) - (KTS)]))) + (0.38235092f * ntu_sq(BDI[(K) - (KTS)]))) + (2.9811542E-4f * ntu_sq(NTU_LOGF(NI3D[(K) - (KTS)])))) + ((1.9052614E-2f * BDI[(K) - (KTS)]) * NTU_LOGF(NI3D[(K) - (KTS)])));
            KDX[(K) - (KTS)] = ntu_fmax(NTU_KCIMIN, ntu_fmin(NTU_KCIMAX, NTU_POWF((BDI[(K) - (KTS)] / FDI[(K) - (KTS)]), 3.0f)));
            AFAI[(K) - (KTS)] = ((((6.0f * KDX[(K) - (KTS)]) - 3.0f) + sqrtf(((8.0f * KDX[(K) - (KTS)]) + 1.0f))) / (2.0f - (2.0f * KDX[(K) - (KTS)])));
            AFAI[(K) - (KTS)] = ntu_fmin(ntu_fmax(AFAI[(K) - (KTS)], NTU_AFAMIN), NTU_AFAMAX);
            KDX[(K) - (KTS)] = (((ntu_sq(AFAI[(K) - (KTS)]) + (3.0f * AFAI[(K) - (KTS)])) + 2.0f) / ((ntu_sq(AFAI[(K) - (KTS)]) + (6.0f * AFAI[(K) - (KTS)])) + 9.0f));
            AI3D[(K) - (KTS)] = NTU_POWF(((KDX[(K) - (KTS)] * NI3D[(K) - (KTS)]) * ntu_sq(I3M3D[(K) - (KTS)])), NTU_THRD);
        }
        if (((QS3D[(K) - (KTS)] >= NTU_QSMALL) && (NS3D[(K) - (KTS)] < NTU_NSMALL))) {
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQS[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QS3D[(K) - (KTS)])));
            if ((TK3D[(K) - (KTS)] < NTU_TK0C)) {
                RHOS[(K) - (KTS)] = (((((15740.702f - (6098.0087f * LTK[(K) - (KTS)])) + (503.33089f * LQS[(K) - (KTS)])) + (594.29913f * ntu_sq(LTK[(K) - (KTS)]))) + (1.9033961f * ntu_sq(LQS[(K) - (KTS)]))) - ((94.950429f * LTK[(K) - (KTS)]) * LQS[(K) - (KTS)]));
            } else {
                RHOS[(K) - (KTS)] = NTU_EXPF(((((((-64808.666f) + (23113.508f * LTK[(K) - (KTS)])) - (36.46632f * LQS[(K) - (KTS)])) - (2060.6024f * ntu_sq(LTK[(K) - (KTS)]))) - (0.005729458f * ntu_sq(LQS[(K) - (KTS)]))) + ((6.5057411f * LTK[(K) - (KTS)]) * LQS[(K) - (KTS)])));
            }
            RHOS[(K) - (KTS)] = ntu_fmin(NTU_RHOIMAX, ntu_fmax(NTU_RHOIMIN, RHOS[(K) - (KTS)]));
            VS3D[(K) - (KTS)] = (QS3D[(K) - (KTS)] / RHOS[(K) - (KTS)]);
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQS[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QS3D[(K) - (KTS)])));
            MDS[(K) - (KTS)] = (NTU_EXPF((((((-123.23898f) + (40.74706f * LTK[(K) - (KTS)])) - (3.0333477f * ntu_sq(LTK[(K) - (KTS)]))) - (0.31219981f * LQS[(K) - (KTS)])) + (0.0012798222f * ntu_sq(LQS[(K) - (KTS)])))) / 1.0E3f);
            NS3D[(K) - (KTS)] = ((((1.0E9f * QS3D[(K) - (KTS)]) * NTU_V2M3) / RHOS[(K) - (KTS)]) / NTU_POWF(MDS[(K) - (KTS)], 3.0f));
            S3M3D[(K) - (KTS)] = ((QS3D[(K) - (KTS)] * NTU_V2M3) / RHOS[(K) - (KTS)]);
            SASPR[(K) - (KTS)] = 0.7f;
            FS3D[(K) - (KTS)] = (SASPR[(K) - (KTS)] * S3M3D[(K) - (KTS)]);
            BDS[(K) - (KTS)] = (NTU_POWF((S3M3D[(K) - (KTS)] / NS3D[(K) - (KTS)]), NTU_THRD) * 1.0E3f);
            BDS[(K) - (KTS)] = ntu_fmin(ntu_fmax(BDS[(K) - (KTS)], (NTU_DSMIN * 1.0E3f)), (NTU_DSMAX * 1.0E3f));
            if ((TK3D[(K) - (KTS)] >= NTU_TK0C)) {
                FDS[(K) - (KTS)] = ((((((-0.21911541f) + (1.2739845f * BDS[(K) - (KTS)])) + (0.10141003f * NTU_LOGF(NS3D[(K) - (KTS)]))) + (0.30063818f * ntu_sq(BDS[(K) - (KTS)]))) - (4.3857765E-3f * ntu_sq(NTU_LOGF(NS3D[(K) - (KTS)])))) - ((0.078801732f * BDS[(K) - (KTS)]) * NTU_LOGF(NS3D[(K) - (KTS)])));
            } else {
                if ((QC3D[(K) - (KTS)] >= 1.0E-8f)) {
                    FDS[(K) - (KTS)] = ((((((-1.1527014f) + (2.9067645f * BDS[(K) - (KTS)])) + (0.25316062f * NTU_LOGF(NS3D[(K) - (KTS)]))) - (0.17768557f * ntu_sq(BDS[(K) - (KTS)]))) - (0.013117292f * ntu_sq(NTU_LOGF(NS3D[(K) - (KTS)])))) - ((0.17020429f * BDS[(K) - (KTS)]) * NTU_LOGF(NS3D[(K) - (KTS)])));
                } else {
                    FDS[(K) - (KTS)] = ((((((-0.2813929f) + (1.7275463f * BDS[(K) - (KTS)])) + (0.045550156f * NTU_LOGF(NS3D[(K) - (KTS)]))) - (0.16526226f * ntu_sq(BDS[(K) - (KTS)]))) - (1.7699916E-3f * ntu_sq(NTU_LOGF(NS3D[(K) - (KTS)])))) - ((4.6441257E-2f * BDS[(K) - (KTS)]) * NTU_LOGF(NS3D[(K) - (KTS)])));
                }
            }
            KDX[(K) - (KTS)] = ntu_fmax(NTU_KCSMIN, ntu_fmin(NTU_KCSMAX, NTU_POWF((BDS[(K) - (KTS)] / FDS[(K) - (KTS)]), 3.0f)));
            AFAS[(K) - (KTS)] = ((((6.0f * KDX[(K) - (KTS)]) - 3.0f) + sqrtf(((8.0f * KDX[(K) - (KTS)]) + 1.0f))) / (2.0f - (2.0f * KDX[(K) - (KTS)])));
            AFAS[(K) - (KTS)] = ntu_fmin(ntu_fmax(AFAS[(K) - (KTS)], NTU_AFAMIN), NTU_AFAMAX);
            KDX[(K) - (KTS)] = (((ntu_sq(AFAS[(K) - (KTS)]) + (3.0f * AFAS[(K) - (KTS)])) + 2.0f) / ((ntu_sq(AFAS[(K) - (KTS)]) + (6.0f * AFAS[(K) - (KTS)])) + 9.0f));
            AS3D[(K) - (KTS)] = NTU_POWF(((KDX[(K) - (KTS)] * NS3D[(K) - (KTS)]) * ntu_sq(S3M3D[(K) - (KTS)])), NTU_THRD);
        }
        if (((QG3D[(K) - (KTS)] >= NTU_QSMALL) && (NG3D[(K) - (KTS)] < NTU_NSMALL))) {
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQG[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QG3D[(K) - (KTS)])));
            if ((TK3D[(K) - (KTS)] < NTU_TK0C)) {
                RHOG[(K) - (KTS)] = NTU_EXPF(((((((-509.23219f) + (187.98419f * LTK[(K) - (KTS)])) + (1.5493489f * LQG[(K) - (KTS)])) - (17.147561f * ntu_sq(LTK[(K) - (KTS)]))) - (0.005829723f * ntu_sq(LQG[(K) - (KTS)]))) - ((0.26664109f * LTK[(K) - (KTS)]) * LQG[(K) - (KTS)])));
            } else {
                RHOG[(K) - (KTS)] = NTU_EXPF(((((((-2894.6795f) + (1050.5173f * LTK[(K) - (KTS)])) - (11.536497f * LQG[(K) - (KTS)])) - (95.054373f * ntu_sq(LTK[(K) - (KTS)]))) - (0.001986964f * ntu_sq(LQG[(K) - (KTS)]))) + ((2.0502147f * LTK[(K) - (KTS)]) * LQG[(K) - (KTS)])));
            }
            VG3D[(K) - (KTS)] = (QG3D[(K) - (KTS)] / RHOG[(K) - (KTS)]);
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQG[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QG3D[(K) - (KTS)])));
            MDG[(K) - (KTS)] = (NTU_EXPF((((((-2205.8027f) + (1225.8046f * LTK[(K) - (KTS)])) - (226.27995f * ntu_sq(LTK[(K) - (KTS)]))) + (13.929644f * NTU_POWF(LTK[(K) - (KTS)], 3.0f))) - (0.25f * LQG[(K) - (KTS)]))) / 1.0E3f);
            NG3D[(K) - (KTS)] = ((((1.0E9f * QG3D[(K) - (KTS)]) * NTU_V2M3) / RHOG[(K) - (KTS)]) / NTU_POWF(MDG[(K) - (KTS)], 3.0f));
            G3M3D[(K) - (KTS)] = ((QG3D[(K) - (KTS)] * NTU_V2M3) / RHOG[(K) - (KTS)]);
            BDG[(K) - (KTS)] = (NTU_POWF((G3M3D[(K) - (KTS)] / NG3D[(K) - (KTS)]), NTU_THRD) * 1.0E3f);
            BDG[(K) - (KTS)] = ntu_fmin(ntu_fmax(BDG[(K) - (KTS)], (NTU_DGMIN * 1.0E3f)), (NTU_DGMAX * 1.0E3f));
            if ((TK3D[(K) - (KTS)] >= NTU_TK0C)) {
                FDG[(K) - (KTS)] = (((((0.58006354f + (0.79661229f * BDG[(K) - (KTS)])) - (0.18394382f * NTU_LOGF(NG3D[(K) - (KTS)]))) + (6.7371044E-2f * ntu_sq(BDG[(K) - (KTS)]))) + (9.832945E-3f * ntu_sq(NTU_LOGF(NG3D[(K) - (KTS)])))) + ((0.12433055f * BDG[(K) - (KTS)]) * NTU_LOGF(NG3D[(K) - (KTS)])));
            } else {
                if ((QC3D[(K) - (KTS)] >= 1.0E-8f)) {
                    FDG[(K) - (KTS)] = (((((0.17363469f + (1.5044291f * BDG[(K) - (KTS)])) - (0.050639722f * NTU_LOGF(NG3D[(K) - (KTS)]))) + (1.5101052E-2f * ntu_sq(BDG[(K) - (KTS)]))) + (2.5974719E-3f * ntu_sq(NTU_LOGF(NG3D[(K) - (KTS)])))) + ((0.01961464f * BDG[(K) - (KTS)]) * NTU_LOGF(NG3D[(K) - (KTS)])));
                } else {
                    FDG[(K) - (KTS)] = (((((0.59259317f - (0.89933515f * BDG[(K) - (KTS)])) + (2.0893032f * ntu_sq(BDG[(K) - (KTS)]))) - (0.50305755f * NTU_POWF(BDG[(K) - (KTS)], 3.0f))) - (2.2446793E-2f * NTU_LOGF(NG3D[(K) - (KTS)]))) + (2.7589047E-3f * ntu_sq(NTU_LOGF(NG3D[(K) - (KTS)]))));
                }
            }
            KDX[(K) - (KTS)] = ntu_fmax(NTU_KCGMIN, ntu_fmin(NTU_KCGMAX, NTU_POWF((BDG[(K) - (KTS)] / FDG[(K) - (KTS)]), 3.0f)));
            AFAG[(K) - (KTS)] = ((((6.0f * KDX[(K) - (KTS)]) - 3.0f) + sqrtf(((8.0f * KDX[(K) - (KTS)]) + 1.0f))) / (2.0f - (2.0f * KDX[(K) - (KTS)])));
            AFAG[(K) - (KTS)] = ntu_fmin(ntu_fmax(AFAG[(K) - (KTS)], NTU_AFAMIN), NTU_AFAMAX);
            KDX[(K) - (KTS)] = (((ntu_sq(AFAG[(K) - (KTS)]) + (3.0f * AFAG[(K) - (KTS)])) + 2.0f) / ((ntu_sq(AFAG[(K) - (KTS)]) + (6.0f * AFAG[(K) - (KTS)])) + 9.0f));
            AG3D[(K) - (KTS)] = NTU_POWF(((KDX[(K) - (KTS)] * NG3D[(K) - (KTS)]) * ntu_sq(G3M3D[(K) - (KTS)])), NTU_THRD);
        }
        if (((QH3D[(K) - (KTS)] >= NTU_QSMALL) && (NH3D[(K) - (KTS)] < NTU_NSMALL))) {
            LTK[(K) - (KTS)] = NTU_LOGF(TK3D[(K) - (KTS)]);
            LQH[(K) - (KTS)] = (-(1.0f * NTU_LOGF(QH3D[(K) - (KTS)])));
            NH3D[(K) - (KTS)] = NTU_EXPF(((((22.929406f - (4.2328364f * LTK[(K) - (KTS)])) + (0.30647567f * ntu_sq(LTK[(K) - (KTS)]))) - (0.009233271f * NTU_POWF(LTK[(K) - (KTS)], 3.0f))) - (0.25f * LQH[(K) - (KTS)])));
            H3M3D[(K) - (KTS)] = ((QH3D[(K) - (KTS)] * NTU_V2M3) / NTU_RHOH);
            BDH[(K) - (KTS)] = (NTU_POWF((H3M3D[(K) - (KTS)] / NH3D[(K) - (KTS)]), NTU_THRD) * 1.0E3f);
            BDH[(K) - (KTS)] = ntu_fmin(ntu_fmax(BDH[(K) - (KTS)], (NTU_DHMIN * 1.0E3f)), (NTU_DHMAX * 1.0E3f));
            if ((TK3D[(K) - (KTS)] >= NTU_TK0C)) {
                FDH[(K) - (KTS)] = (((((1.157754f + (0.37852874f * BDH[(K) - (KTS)])) - (0.11129737f * NTU_LOGF(NH3D[(K) - (KTS)]))) + (0.13929599f * ntu_sq(BDH[(K) - (KTS)]))) + (8.1105237E-3f * ntu_sq(NTU_LOGF(NH3D[(K) - (KTS)])))) + ((5.7432113E-2f * BDH[(K) - (KTS)]) * NTU_LOGF(NH3D[(K) - (KTS)])));
            } else {
                FDH[(K) - (KTS)] = ((((((-0.48246793f) + (2.0407077f * BDH[(K) - (KTS)])) + (0.022262969f * NTU_LOGF(NH3D[(K) - (KTS)]))) - (0.158389f * ntu_sq(BDH[(K) - (KTS)]))) - (5.5545804E-3f * ntu_sq(NTU_LOGF(NH3D[(K) - (KTS)])))) + ((2.9443577E-2f * BDH[(K) - (KTS)]) * NTU_LOGF(NH3D[(K) - (KTS)])));
            }
            KDX[(K) - (KTS)] = ntu_fmax(NTU_KCHMIN, ntu_fmin(NTU_KCHMAX, NTU_POWF((BDH[(K) - (KTS)] / FDH[(K) - (KTS)]), 3.0f)));
            AFAH[(K) - (KTS)] = ((((6.0f * KDX[(K) - (KTS)]) - 3.0f) + sqrtf(((8.0f * KDX[(K) - (KTS)]) + 1.0f))) / (2.0f - (2.0f * KDX[(K) - (KTS)])));
            AFAH[(K) - (KTS)] = ntu_fmin(ntu_fmax(AFAH[(K) - (KTS)], NTU_AFAMIN), NTU_AFAMAX);
            KDX[(K) - (KTS)] = (((ntu_sq(AFAH[(K) - (KTS)]) + (3.0f * AFAH[(K) - (KTS)])) + 2.0f) / ((ntu_sq(AFAH[(K) - (KTS)]) + (6.0f * AFAH[(K) - (KTS)])) + 9.0f));
            AH3D[(K) - (KTS)] = NTU_POWF(((KDX[(K) - (KTS)] * NH3D[(K) - (KTS)]) * ntu_sq(H3M3D[(K) - (KTS)])), NTU_THRD);
        }
        if ((SSRW[(K) - (KTS)] < (-1.0E-1f))) {
            if ((QR3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QV3D[(K) - (KTS)] = (QV3D[(K) - (KTS)] + QR3D[(K) - (KTS)]);
                TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QR3D[(K) - (KTS)] * XXLV[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QR3D[(K) - (KTS)] = 0.0f;
                NR3D[(K) - (KTS)] = 0.0f;
            }
            if ((QC3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QV3D[(K) - (KTS)] = (QV3D[(K) - (KTS)] + QC3D[(K) - (KTS)]);
                TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QC3D[(K) - (KTS)] * XXLV[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QC3D[(K) - (KTS)] = 0.0f;
                NC3D[(K) - (KTS)] = 0.0f;
            }
        }
        if ((SSRI[(K) - (KTS)] < (-1.0E-1f))) {
            if ((QI3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QV3D[(K) - (KTS)] = (QV3D[(K) - (KTS)] + QI3D[(K) - (KTS)]);
                TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QI3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QI3D[(K) - (KTS)] = 0.0f;
                NI3D[(K) - (KTS)] = 0.0f;
                I3M3D[(K) - (KTS)] = 0.0f;
                FI3D[(K) - (KTS)] = 0.0f;
                VI3D[(K) - (KTS)] = 0.0f;
                AI3D[(K) - (KTS)] = 0.0f;
            }
            if ((QS3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QV3D[(K) - (KTS)] = (QV3D[(K) - (KTS)] + QS3D[(K) - (KTS)]);
                TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QS3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QS3D[(K) - (KTS)] = 0.0f;
                NS3D[(K) - (KTS)] = 0.0f;
                VS3D[(K) - (KTS)] = 0.0f;
                AS3D[(K) - (KTS)] = 0.0f;
                FS3D[(K) - (KTS)] = 0.0f;
            }
            if ((QG3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QV3D[(K) - (KTS)] = (QV3D[(K) - (KTS)] + QG3D[(K) - (KTS)]);
                TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QG3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QG3D[(K) - (KTS)] = 0.0f;
                NG3D[(K) - (KTS)] = 0.0f;
                VG3D[(K) - (KTS)] = 0.0f;
                AG3D[(K) - (KTS)] = 0.0f;
            }
            if ((QH3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QV3D[(K) - (KTS)] = (QV3D[(K) - (KTS)] + QH3D[(K) - (KTS)]);
                TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QH3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QH3D[(K) - (KTS)] = 0.0f;
                NH3D[(K) - (KTS)] = 0.0f;
                AH3D[(K) - (KTS)] = 0.0f;
            }
        }
        if (((TK3D[(K) - (KTS)] >= NTU_TK0C) && (QS3D[(K) - (KTS)] < NTU_QLIMIT))) {
            QR3D[(K) - (KTS)] = (QR3D[(K) - (KTS)] + QS3D[(K) - (KTS)]);
            NR3D[(K) - (KTS)] = (NR3D[(K) - (KTS)] + NS3D[(K) - (KTS)]);
            TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QS3D[(K) - (KTS)] * XXLF[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QS3D[(K) - (KTS)] = 0.0f;
            NS3D[(K) - (KTS)] = 0.0f;
            VS3D[(K) - (KTS)] = 0.0f;
            AS3D[(K) - (KTS)] = 0.0f;
            FS3D[(K) - (KTS)] = 0.0f;
        }
        if (((TK3D[(K) - (KTS)] >= NTU_TK0C) && (QG3D[(K) - (KTS)] < NTU_QLIMIT))) {
            QR3D[(K) - (KTS)] = (QR3D[(K) - (KTS)] + QG3D[(K) - (KTS)]);
            NR3D[(K) - (KTS)] = (NR3D[(K) - (KTS)] + NG3D[(K) - (KTS)]);
            TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QG3D[(K) - (KTS)] * XXLF[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QG3D[(K) - (KTS)] = 0.0f;
            NG3D[(K) - (KTS)] = 0.0f;
            VG3D[(K) - (KTS)] = 0.0f;
            AG3D[(K) - (KTS)] = 0.0f;
        }
        if (((TK3D[(K) - (KTS)] >= NTU_TK0C) && (QH3D[(K) - (KTS)] < NTU_QLIMIT))) {
            QR3D[(K) - (KTS)] = (QR3D[(K) - (KTS)] + QH3D[(K) - (KTS)]);
            NR3D[(K) - (KTS)] = (NR3D[(K) - (KTS)] + NH3D[(K) - (KTS)]);
            TK3D[(K) - (KTS)] = (TK3D[(K) - (KTS)] - ((QH3D[(K) - (KTS)] * XXLF[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QH3D[(K) - (KTS)] = 0.0f;
            NH3D[(K) - (KTS)] = 0.0f;
            AH3D[(K) - (KTS)] = 0.0f;
        }
    } }
    { const int ntu_hi_k = KTE;
    for (K = KTS; K <= ntu_hi_k; ++K) {
        TAIR[(K) - (KTS)] = TK3D[(K) - (KTS)];
        QVAP[(K) - (KTS)] = QV3D[(K) - (KTS)];
        DTAIR[(K) - (KTS)] = (-((W3D[(K) - (KTS)] * NTU_G) / NTU_CP));
        DQVAP[(K) - (KTS)] = 0.0f;
        PRES[(K) - (KTS)] = P3D[(K) - (KTS)];
        TK0[(K) - (KTS)] = (TAIR[(K) - (KTS)] - (DTAIR[(K) - (KTS)] * DT));
        P40[(K) - (KTS)] = (PRES[(K) - (KTS)] * NTU_POWF((TK0[(K) - (KTS)] / TAIR[(K) - (KTS)]), (NTU_CP / NTU_R)));
        TAIR[(K) - (KTS)] = TK0[(K) - (KTS)];
        IV1 = 1;
        { const int ntu_hi_iv = NTU_NAER;
        for (IV = 1; IV <= ntu_hi_iv; ++IV) {
            IV2 = (IV1 + 1);
            DO_FR[((K) - (KTS)) + NTU_KMAX * ((IV) - (1))] = true;
            RX0[((K) - (KTS)) + NTU_KMAX * ((IV) - (1))] = 99.0f;
            if ((NTU_NAERN[(IV) - (1)] >= 2)) {
                AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] = ntu_fmin(AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))], AERO[((K) - (KTS)) + NTU_KMAX * ((IV2) - (1))]);
                { const int ntu_hi_im = NTU_NCCN;
                for (IM = 1; IM <= ntu_hi_im; ++IM) {
                    ZCCNS[((IM) - (1)) + 3 * (((K) - (KTS)) + NTU_KMAX * ((IV) - (1)))] = ((AERO[((K) - (KTS)) + NTU_KMAX * ((IV2) - (1))] * AC->WMAS[((IM) - (1)) + 3 * ((IV) - (1))]) * AC->RFACT[((IM) - (1)) + 3 * ((IV) - (1))]);
                } }
            } else {
                { const int ntu_hi_im = NTU_NCCN;
                for (IM = 1; IM <= ntu_hi_im; ++IM) {
                    ZCCNS[((IM) - (1)) + 3 * (((K) - (KTS)) + NTU_KMAX * ((IV) - (1)))] = ((AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] * AC->WMAS[((IM) - (1)) + 3 * ((IV) - (1))]) * AC->RFACT[((IM) - (1)) + 3 * ((IV) - (1))]);
                } }
            }
            IV1 = (IV1 + NTU_NAERN[(IV) - (1)]);
        } }
    } }
    SDTL = 0.0f;
    { const int ntu_hi_it = ((int)(((DT - (NTU_DT20S * 0.01f)) / NTU_DT20S)) + 1);
    for (IT = 1; IT <= ntu_hi_it; ++IT) {
        DTL = ntu_fmin(NTU_DT20S, (DT - SDTL));
        { const int ntu_hi_k = KTE;
        for (K = KTS; K <= ntu_hi_k; ++K) {
            DTS = DTL;
            SDTS = 0.0f;
            TMP = (((W3D[(K) - (KTS)] * NTU_G) / NTU_CP) / TK0[(K) - (KTS)]);
            XDNC[(K) - (KTS)] = 0.0f;
            XDNR[(K) - (KTS)] = 0.0f;
            GQCTR[(K) - (KTS)] = 0.0f;
            ADAGR[(K) - (KTS)] = 1.0f;
            RHOI[(K) - (KTS)] = NTU_RHOI0;
            RHOS[(K) - (KTS)] = NTU_RHOS0;
            RHOG[(K) - (KTS)] = NTU_RHOG0;
            AFAC[(K) - (KTS)] = NTU_AFAC0;
            AFAR[(K) - (KTS)] = NTU_AFAR0;
            AFAI[(K) - (KTS)] = NTU_AFAI0;
            AFAS[(K) - (KTS)] = NTU_AFAS0;
            AFAG[(K) - (KTS)] = NTU_AFAG0;
            AFAH[(K) - (KTS)] = NTU_AFAH0;
            AVI[(K) - (KTS)] = NTU_AVI0;
            BVI[(K) - (KTS)] = NTU_BVI0;
            AVS[(K) - (KTS)] = NTU_AVS0;
            BVS[(K) - (KTS)] = NTU_BVS0;
            AVG[(K) - (KTS)] = NTU_AVG0;
            BVG[(K) - (KTS)] = NTU_BVG0;
            AVH[(K) - (KTS)] = NTU_AVH0;
            BVH[(K) - (KTS)] = NTU_BVH0;
            AMI[(K) - (KTS)] = NTU_AMI0;
            BMI[(K) - (KTS)] = NTU_BMI0;
            AMS[(K) - (KTS)] = NTU_AMS0;
            AMG[(K) - (KTS)] = NTU_AMG0;
            IASPR[(K) - (KTS)] = 1.0f;
            SASPR[(K) - (KTS)] = 1.0f;
            QACAC[(K) - (KTS)] = 0.0f;
            QACAR[(K) - (KTS)] = 0.0f;
            L_111:;
            PRES[(K) - (KTS)] = (P40[(K) - (KTS)] * NTU_POWF((1.0f - (TMP * (SDTL + SDTS))), (NTU_CP / NTU_R)));
            DPDT[(K) - (KTS)] = (-(((NTU_G * W3D[(K) - (KTS)]) * PRES[(K) - (KTS)]) / ((NTU_R * TK0[(K) - (KTS)]) * (1.0f - (TMP * (SDTL + SDTS))))));
            ESW[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 0));
            ESI[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 1));
            if ((ESI[(K) - (KTS)] > ESW[(K) - (KTS)])) {
                ESI[(K) - (KTS)] = ESW[(K) - (KTS)];
            }
            QVSW[(K) - (KTS)] = ((0.622f * ESW[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESW[(K) - (KTS)]));
            QVSI[(K) - (KTS)] = ((0.622f * ESI[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESI[(K) - (KTS)]));
            SSRW[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / QVSW[(K) - (KTS)]) - 1.0f);
            SSRI[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / QVSI[(K) - (KTS)]) - 1.0f);
            XXLV[(K) - (KTS)] = (3.1484E6f - (2370.0f * TAIR[(K) - (KTS)]));
            CPM[(K) - (KTS)] = (NTU_CP * (1.0f + (0.887f * QVAP[(K) - (KTS)])));
            RHO[(K) - (KTS)] = ((PRES[(K) - (KTS)] / (TAIR[(K) - (KTS)] * (1.0f + (0.61f * QVAP[(K) - (KTS)])))) / NTU_R);
            CNDMAX = ((QVAP[(K) - (KTS)] - QVSW[(K) - (KTS)]) / (1.0f + ((ntu_sq(XXLV[(K) - (KTS)]) * QVAP[(K) - (KTS)]) / ((CPM[(K) - (KTS)] * NTU_RV) * ntu_sq(TAIR[(K) - (KTS)])))));
            if (((SSRW[(K) - (KTS)] > NTU_RSMALL) && (SSRW[(K) - (KTS)] > S3D[(K) - (KTS)]))) {
                DCLDMT[(K) - (KTS)] = 0.0f;
                DCLDMC[(K) - (KTS)] = 0.0f;
                DCLDMR[(K) - (KTS)] = 0.0f;
                IV1 = 1;
                { const int ntu_hi_iv = NTU_NAER;
                for (IV = 1; IV <= ntu_hi_iv; ++IV) {
                    { const int ntu_hi_j = NTU_NAERN[(IV) - (1)];
                    for (J = 1; J <= ntu_hi_j; ++J) {
                        ABCD[(J) - (1)] = AERO[((K) - (KTS)) + NTU_KMAX * ((((IV1 - 1) + J)) - (1))];
                    } }
                    if ((((AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] > 1.0E-20f) && (NTU_NAERN[(IV) - (1)] >= 2)) && (W3D[(K) - (KTS)] > 1.0E-5f))) {
                        ntu_activa(AC, TAIR[(K) - (KTS)], W3D[(K) - (KTS)], NC3D[(K) - (KTS)], NR3D[(K) - (KTS)], ABCD, QACAC[(K) - (KTS)], QACAR[(K) - (KTS)], RX0[((K) - (KTS)) + NTU_KMAX * ((IV) - (1))], &ZCCNS[((1) - (1)) + 3 * (((K) - (KTS)) + NTU_KMAX * ((IV) - (1)))], SSRW[(K) - (KTS)], IV, DO_FR[((K) - (KTS)) + NTU_KMAX * ((IV) - (1))]);
                        DCLDMC[(K) - (KTS)] = (DCLDMC[(K) - (KTS)] + QACAC[(K) - (KTS)]);
                        DCLDMR[(K) - (KTS)] = (DCLDMR[(K) - (KTS)] + QACAR[(K) - (KTS)]);
                        { const int ntu_hi_j = NTU_NAERN[(IV) - (1)];
                        for (J = 1; J <= ntu_hi_j; ++J) {
                            AERO[((K) - (KTS)) + NTU_KMAX * ((((IV1 - 1) + J)) - (1))] = ABCD[(J) - (1)];
                        } }
                    }
                    IV1 = (IV1 + NTU_NAERN[(IV) - (1)]);
                } }
                DCLDMT[(K) - (KTS)] = (DCLDMC[(K) - (KTS)] + DCLDMR[(K) - (KTS)]);
                if ((DCLDMT[(K) - (KTS)] > NTU_RLIMIT)) {
                    if ((DCLDMT[(K) - (KTS)] > CNDMAX)) {
                        DCLDMC[(K) - (KTS)] = (DCLDMC[(K) - (KTS)] * (CNDMAX / DCLDMT[(K) - (KTS)]));
                        DCLDMR[(K) - (KTS)] = (DCLDMR[(K) - (KTS)] * (CNDMAX / DCLDMT[(K) - (KTS)]));
                        DCLDMT[(K) - (KTS)] = (DCLDMC[(K) - (KTS)] + DCLDMR[(K) - (KTS)]);
                    }
                    QVAP[(K) - (KTS)] = ntu_fmax(0.0f, (QVAP[(K) - (KTS)] - DCLDMT[(K) - (KTS)]));
                    QC3D[(K) - (KTS)] = ntu_fmax(0.0f, (QC3D[(K) - (KTS)] + DCLDMC[(K) - (KTS)]));
                    QR3D[(K) - (KTS)] = ntu_fmax(0.0f, (QR3D[(K) - (KTS)] + DCLDMR[(K) - (KTS)]));
                    CPM[(K) - (KTS)] = (NTU_CP * (1.0f + (0.887f * QVAP[(K) - (KTS)])));
                    XXLV[(K) - (KTS)] = (3.1484E6f - (2370.0f * TAIR[(K) - (KTS)]));
                    TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] + ((XXLV[(K) - (KTS)] * DCLDMT[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                }
                ESW[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 0));
                ESI[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 1));
                if ((ESI[(K) - (KTS)] > ESW[(K) - (KTS)])) {
                    ESI[(K) - (KTS)] = ESW[(K) - (KTS)];
                }
                SSRW[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / ((0.622f * ESW[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESW[(K) - (KTS)]))) - 1.0f);
                SSRI[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / ((0.622f * ESI[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESI[(K) - (KTS)]))) - 1.0f);
                RHO[(K) - (KTS)] = ((PRES[(K) - (KTS)] / (TAIR[(K) - (KTS)] * (1.0f + (0.61f * QVAP[(K) - (KTS)])))) / NTU_R);
            }
            S3D[(K) - (KTS)] = ntu_fmax(S3D[(K) - (KTS)], SSRW[(K) - (KTS)]);
            if (((TAIR[(K) - (KTS)] < (NTU_TK0C - 2.0f)) && (SSRI[(K) - (KTS)] > 1.0E-4f))) {
                ntu_icenu(TAIR[(K) - (KTS)], PRES[(K) - (KTS)], DTS, RHO[(K) - (KTS)], QVAP[(K) - (KTS)], QI3D[(K) - (KTS)], NI3D[(K) - (KTS)], VI3D[(K) - (KTS)], FI3D[(K) - (KTS)], AI3D[(K) - (KTS)], I3M3D[(K) - (KTS)], AERO[((K) - (KTS)) + NTU_KMAX * ((5) - (1))]);
                ESW[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 0));
                ESI[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 1));
                if ((ESI[(K) - (KTS)] > ESW[(K) - (KTS)])) {
                    ESI[(K) - (KTS)] = ESW[(K) - (KTS)];
                }
                SSRW[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / ((0.622f * ESW[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESW[(K) - (KTS)]))) - 1.0f);
                SSRI[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / ((0.622f * ESI[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESI[(K) - (KTS)]))) - 1.0f);
                RHO[(K) - (KTS)] = ((PRES[(K) - (KTS)] / (TAIR[(K) - (KTS)] * (1.0f + (0.61f * QVAP[(K) - (KTS)])))) / NTU_R);
            }
            if (((((((QC3D[(K) - (KTS)] >= NTU_QSMALL) || (QR3D[(K) - (KTS)] >= NTU_QSMALL)) || (QI3D[(K) - (KTS)] >= NTU_QSMALL)) || (QS3D[(K) - (KTS)] >= NTU_QSMALL)) || (QG3D[(K) - (KTS)] >= NTU_QSMALL)) || (QH3D[(K) - (KTS)] >= NTU_QSMALL))) {
                ntu_small_dt(AC, DTL, DTS, SDTS, DTAIR[(K) - (KTS)], DQVAP[(K) - (KTS)], DPDT[(K) - (KTS)], RHO[(K) - (KTS)], TAIR[(K) - (KTS)], PRES[(K) - (KTS)], QVAP[(K) - (KTS)], QC3D[(K) - (KTS)], QR3D[(K) - (KTS)], QI3D[(K) - (KTS)], QS3D[(K) - (KTS)], QG3D[(K) - (KTS)], QH3D[(K) - (KTS)], NC3D[(K) - (KTS)], NR3D[(K) - (KTS)], NI3D[(K) - (KTS)], NS3D[(K) - (KTS)], NG3D[(K) - (KTS)], NH3D[(K) - (KTS)], VI3D[(K) - (KTS)], VS3D[(K) - (KTS)], VG3D[(K) - (KTS)], FI3D[(K) - (KTS)], FS3D[(K) - (KTS)], AI3D[(K) - (KTS)], AS3D[(K) - (KTS)], AG3D[(K) - (KTS)], AH3D[(K) - (KTS)], I3M3D[(K) - (KTS)], SASPR[(K) - (KTS)], XDNC[(K) - (KTS)], XDNR[(K) - (KTS)], GQCTR[(K) - (KTS)]);
            }
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] + (DTAIR[(K) - (KTS)] * DTS));
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + (DQVAP[(K) - (KTS)] * DTS));
            SDTS = (SDTS + DTS);
            if ((SDTS >= DTL)) {
                goto L_333;
            }
            goto L_111;
            L_333:;
            if ((XDNR[(K) - (KTS)] > 1.0f)) {
                IV0 = 0;
                { const int ntu_hi_iv = NTU_NAER;
                for (IV = 1; IV <= ntu_hi_iv; ++IV) {
                    if ((NTU_NAERN[(IV) - (1)] == 4)) {
                        IV3 = (IV0 + 3);
                        IV4 = (IV0 + 4);
                        if ((NR3D[(K) - (KTS)] > 1.0f)) {
                            XMASS = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))] * ntu_fmin(1.0f, (XDNR[(K) - (KTS)] / NR3D[(K) - (KTS)])));
                        } else {
                            XMASS = AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))];
                        }
                        AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] + XMASS);
                        AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))] - XMASS);
                    }
                    IV0 = (IV0 + NTU_NAERN[(IV) - (1)]);
                } }
            }
            if ((XDNC[(K) - (KTS)] > 1.0f)) {
                IV1 = 1;
                { const int ntu_hi_iv = NTU_NAER;
                for (IV = 1; IV <= ntu_hi_iv; ++IV) {
                    IV2 = (IV1 + 1);
                    IV3 = (IV2 + 1);
                    if ((NTU_NAERN[(IV) - (1)] >= 3)) {
                        if ((NC3D[(K) - (KTS)] > 1.0f)) {
                            XMASS = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] * ntu_fmin(1.0f, (XDNC[(K) - (KTS)] / NC3D[(K) - (KTS)])));
                        } else {
                            XMASS = AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))];
                        }
                        AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] + XMASS);
                        AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] - XMASS);
                        AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] = ntu_fmin(AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))], AERO[((K) - (KTS)) + NTU_KMAX * ((IV2) - (1))]);
                    } else if ((NTU_NAERN[(IV) - (1)] == 2)) {
                        ntu_deactiva(AC, XDNC[(K) - (KTS)], QC3D[(K) - (KTS)], NC3D[(K) - (KTS)], AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))], AERO[((K) - (KTS)) + NTU_KMAX * ((IV2) - (1))], AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))], RX0[((K) - (KTS)) + NTU_KMAX * ((IV) - (1))], &ZCCNS[((1) - (1)) + 3 * (((K) - (KTS)) + NTU_KMAX * ((IV) - (1)))], IV, DO_FR[((K) - (KTS)) + NTU_KMAX * ((IV) - (1))]);
                    } else {
                        XMASS = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] * ntu_fmin(1.0f, (XDNC[(K) - (KTS)] / (ZCCNS[((2) - (1)) + 3 * (((K) - (KTS)) + NTU_KMAX * ((IV) - (1)))] + ZCCNS[((3) - (1)) + 3 * (((K) - (KTS)) + NTU_KMAX * ((IV) - (1)))]))));
                        AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV1) - (1))] + XMASS);
                    }
                    IV1 = (IV1 + NTU_NAERN[(IV) - (1)]);
                } }
            }
            PRES[(K) - (KTS)] = (P40[(K) - (KTS)] * NTU_POWF((1.0f - (TMP * (SDTL + SDTS))), (NTU_CP / NTU_R)));
            RHO[(K) - (KTS)] = ((PRES[(K) - (KTS)] / (TAIR[(K) - (KTS)] * (1.0f + (0.61f * QVAP[(K) - (KTS)])))) / NTU_R);
        } }
        { const int ntu_hi_k = KTE;
        for (K = KTS; K <= ntu_hi_k; ++K) {
            if (((((((QC3D[(K) - (KTS)] >= NTU_QSMALL) || (QR3D[(K) - (KTS)] >= NTU_QSMALL)) || (QI3D[(K) - (KTS)] >= NTU_QSMALL)) || (QS3D[(K) - (KTS)] >= NTU_QSMALL)) || (QG3D[(K) - (KTS)] >= NTU_QSMALL)) || (QH3D[(K) - (KTS)] >= NTU_QSMALL))) {
                ntu_large_dt(AC, DTL, TAIR[(K) - (KTS)], QVAP[(K) - (KTS)], PRES[(K) - (KTS)], RHO[(K) - (KTS)], QC3D[(K) - (KTS)], QR3D[(K) - (KTS)], QI3D[(K) - (KTS)], QS3D[(K) - (KTS)], QG3D[(K) - (KTS)], QH3D[(K) - (KTS)], NC3D[(K) - (KTS)], NR3D[(K) - (KTS)], NI3D[(K) - (KTS)], NS3D[(K) - (KTS)], NG3D[(K) - (KTS)], NH3D[(K) - (KTS)], VI3D[(K) - (KTS)], VS3D[(K) - (KTS)], VG3D[(K) - (KTS)], FI3D[(K) - (KTS)], FS3D[(K) - (KTS)], AI3D[(K) - (KTS)], AS3D[(K) - (KTS)], AG3D[(K) - (KTS)], AH3D[(K) - (KTS)], I3M3D[(K) - (KTS)], SASPR[(K) - (KTS)], GQCTR[(K) - (KTS)]);
            }
            IV0 = 0;
            { const int ntu_hi_iv = NTU_NAER;
            for (IV = 1; IV <= ntu_hi_iv; ++IV) {
                if ((NTU_NAERN[(IV) - (1)] >= 4)) {
                    IV3 = (IV0 + 3);
                    IV4 = (IV0 + 4);
                    if ((GQCTR[(K) - (KTS)] > 0.0f)) {
                        XMASS = (GQCTR[(K) - (KTS)] * AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))]);
                    } else {
                        XMASS = (GQCTR[(K) - (KTS)] * AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))]);
                    }
                    AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV3) - (1))] - XMASS);
                    AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))] = (AERO[((K) - (KTS)) + NTU_KMAX * ((IV4) - (1))] + XMASS);
                }
                IV0 = (IV0 + NTU_NAERN[(IV) - (1)]);
            } }
        } }
        SDTL = (SDTL + DTL);
    } }
    { const int ntu_hi_k = KTE;
    for (K = KTS; K <= ntu_hi_k; ++K) {
        ESW[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 0));
        ESI[(K) - (KTS)] = ntu_fmin((0.99f * PRES[(K) - (KTS)]), ntu_polysvp(TAIR[(K) - (KTS)], 1));
        if ((ESI[(K) - (KTS)] > ESW[(K) - (KTS)])) {
            ESI[(K) - (KTS)] = ESW[(K) - (KTS)];
        }
        SSRW[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / ((0.622f * ESW[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESW[(K) - (KTS)]))) - 1.0f);
        SSRI[(K) - (KTS)] = ((QVAP[(K) - (KTS)] / ((0.622f * ESI[(K) - (KTS)]) / (PRES[(K) - (KTS)] - ESI[(K) - (KTS)]))) - 1.0f);
        XXLV[(K) - (KTS)] = (3.1484E6f - (2370.0f * TAIR[(K) - (KTS)]));
        XXLS[(K) - (KTS)] = ((3.15E6f - (2.37E3f * TAIR[(K) - (KTS)])) + 0.3337E6f);
        CPM[(K) - (KTS)] = (NTU_CP * (1.0f + (0.887f * QVAP[(K) - (KTS)])));
        if ((SSRW[(K) - (KTS)] < (-1.0E-1f))) {
            if ((QR3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QR3D[(K) - (KTS)]);
                TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QR3D[(K) - (KTS)] * XXLV[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QR3D[(K) - (KTS)] = 0.0f;
                NR3D[(K) - (KTS)] = 0.0f;
            }
            if ((QC3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QC3D[(K) - (KTS)]);
                TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QC3D[(K) - (KTS)] * XXLV[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QC3D[(K) - (KTS)] = 0.0f;
                NC3D[(K) - (KTS)] = 0.0f;
            }
        }
        if ((SSRI[(K) - (KTS)] < (-1.0E-1f))) {
            if ((QI3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QI3D[(K) - (KTS)]);
                TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QI3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QI3D[(K) - (KTS)] = 0.0f;
                NI3D[(K) - (KTS)] = 0.0f;
                I3M3D[(K) - (KTS)] = 0.0f;
                FI3D[(K) - (KTS)] = 0.0f;
                VI3D[(K) - (KTS)] = 0.0f;
                AI3D[(K) - (KTS)] = 0.0f;
            }
            if ((QS3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QS3D[(K) - (KTS)]);
                TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QS3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QS3D[(K) - (KTS)] = 0.0f;
                NS3D[(K) - (KTS)] = 0.0f;
                VS3D[(K) - (KTS)] = 0.0f;
                AS3D[(K) - (KTS)] = 0.0f;
                FS3D[(K) - (KTS)] = 0.0f;
            }
            if ((QG3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QG3D[(K) - (KTS)]);
                TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QG3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QG3D[(K) - (KTS)] = 0.0f;
                NG3D[(K) - (KTS)] = 0.0f;
                VG3D[(K) - (KTS)] = 0.0f;
                AG3D[(K) - (KTS)] = 0.0f;
            }
            if ((QH3D[(K) - (KTS)] < NTU_QSMAL1)) {
                QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QH3D[(K) - (KTS)]);
                TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QH3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
                QH3D[(K) - (KTS)] = 0.0f;
                NH3D[(K) - (KTS)] = 0.0f;
                AH3D[(K) - (KTS)] = 0.0f;
            }
        }
        if (((TAIR[(K) - (KTS)] > NTU_TK0C) && (QI3D[(K) - (KTS)] > 0.0f))) {
            QR3D[(K) - (KTS)] = (QR3D[(K) - (KTS)] + QI3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QI3D[(K) - (KTS)] * XXLF[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            NR3D[(K) - (KTS)] = (NR3D[(K) - (KTS)] + NI3D[(K) - (KTS)]);
            QI3D[(K) - (KTS)] = 0.0f;
            NI3D[(K) - (KTS)] = 0.0f;
            I3M3D[(K) - (KTS)] = 0.0f;
            FI3D[(K) - (KTS)] = 0.0f;
            VI3D[(K) - (KTS)] = 0.0f;
            AI3D[(K) - (KTS)] = 0.0f;
        }
        if (((QC3D[(K) - (KTS)] < NTU_QSMALL) || (NC3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QC3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QC3D[(K) - (KTS)] * XXLV[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QC3D[(K) - (KTS)] = 0.0f;
            NC3D[(K) - (KTS)] = 0.0f;
        }
        if (((QR3D[(K) - (KTS)] < NTU_QSMALL) || (NR3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QR3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QR3D[(K) - (KTS)] * XXLV[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QR3D[(K) - (KTS)] = 0.0f;
            NR3D[(K) - (KTS)] = 0.0f;
        }
        if (((QI3D[(K) - (KTS)] < NTU_QSMALL) || (NI3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QI3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QI3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QI3D[(K) - (KTS)] = 0.0f;
            NI3D[(K) - (KTS)] = 0.0f;
            I3M3D[(K) - (KTS)] = 0.0f;
            FI3D[(K) - (KTS)] = 0.0f;
            VI3D[(K) - (KTS)] = 0.0f;
            AI3D[(K) - (KTS)] = 0.0f;
        }
        if (((QS3D[(K) - (KTS)] < NTU_QSMALL) || (NS3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QS3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QS3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QS3D[(K) - (KTS)] = 0.0f;
            NS3D[(K) - (KTS)] = 0.0f;
            VS3D[(K) - (KTS)] = 0.0f;
            AS3D[(K) - (KTS)] = 0.0f;
            FS3D[(K) - (KTS)] = 0.0f;
        }
        if (((QG3D[(K) - (KTS)] < NTU_QSMALL) || (NG3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QG3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QG3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QG3D[(K) - (KTS)] = 0.0f;
            NG3D[(K) - (KTS)] = 0.0f;
            VG3D[(K) - (KTS)] = 0.0f;
            AG3D[(K) - (KTS)] = 0.0f;
        }
        if (((QH3D[(K) - (KTS)] < NTU_QSMALL) || (NH3D[(K) - (KTS)] < NTU_NSMALL))) {
            QVAP[(K) - (KTS)] = (QVAP[(K) - (KTS)] + QH3D[(K) - (KTS)]);
            TAIR[(K) - (KTS)] = (TAIR[(K) - (KTS)] - ((QH3D[(K) - (KTS)] * XXLS[(K) - (KTS)]) / CPM[(K) - (KTS)]));
            QH3D[(K) - (KTS)] = 0.0f;
            NH3D[(K) - (KTS)] = 0.0f;
            AH3D[(K) - (KTS)] = 0.0f;
        }
        RHO[(K) - (KTS)] = ((PRES[(K) - (KTS)] / (TAIR[(K) - (KTS)] * (1.0f + (0.61f * QVAP[(K) - (KTS)])))) / NTU_R);
        VT_QC[(K) - (KTS)] = 0.0f;
        VT_NC[(K) - (KTS)] = 0.0f;
        VT_QR[(K) - (KTS)] = 0.0f;
        VT_NR[(K) - (KTS)] = 0.0f;
        VT_QI[(K) - (KTS)] = 0.0f;
        VT_NI[(K) - (KTS)] = 0.0f;
        VT_QS[(K) - (KTS)] = 0.0f;
        VT_NS[(K) - (KTS)] = 0.0f;
        VT_QG[(K) - (KTS)] = 0.0f;
        VT_NG[(K) - (KTS)] = 0.0f;
        VT_QH[(K) - (KTS)] = 0.0f;
        VT_NH[(K) - (KTS)] = 0.0f;
        VT_VI[(K) - (KTS)] = 0.0f;
        VT_VS[(K) - (KTS)] = 0.0f;
        VT_VG[(K) - (KTS)] = 0.0f;
        VT_FI[(K) - (KTS)] = 0.0f;
        VT_FS[(K) - (KTS)] = 0.0f;
        VT_AI[(K) - (KTS)] = 0.0f;
        VT_AS[(K) - (KTS)] = 0.0f;
        VT_AG[(K) - (KTS)] = 0.0f;
        VT_AH[(K) - (KTS)] = 0.0f;
        VTI3M[(K) - (KTS)] = 0.0f;
        if (((((((QC3D[(K) - (KTS)] >= NTU_QSMALL) || (QR3D[(K) - (KTS)] >= NTU_QSMALL)) || (QI3D[(K) - (KTS)] >= NTU_QSMALL)) || (QS3D[(K) - (KTS)] >= NTU_QSMALL)) || (QG3D[(K) - (KTS)] >= NTU_QSMALL)) || (QH3D[(K) - (KTS)] >= NTU_QSMALL))) {
            ntu_sedi_fall(AC, TAIR[(K) - (KTS)], PRES[(K) - (KTS)], QVAP[(K) - (KTS)], QC3D[(K) - (KTS)], QR3D[(K) - (KTS)], QI3D[(K) - (KTS)], QS3D[(K) - (KTS)], QG3D[(K) - (KTS)], QH3D[(K) - (KTS)], NC3D[(K) - (KTS)], NR3D[(K) - (KTS)], NI3D[(K) - (KTS)], NS3D[(K) - (KTS)], NG3D[(K) - (KTS)], NH3D[(K) - (KTS)], VI3D[(K) - (KTS)], VS3D[(K) - (KTS)], VG3D[(K) - (KTS)], FI3D[(K) - (KTS)], FS3D[(K) - (KTS)], AI3D[(K) - (KTS)], AS3D[(K) - (KTS)], AG3D[(K) - (KTS)], AH3D[(K) - (KTS)], I3M3D[(K) - (KTS)], VT_QC[(K) - (KTS)], VT_QR[(K) - (KTS)], VT_QI[(K) - (KTS)], VT_QS[(K) - (KTS)], VT_QG[(K) - (KTS)], VT_QH[(K) - (KTS)], VT_NC[(K) - (KTS)], VT_NR[(K) - (KTS)], VT_NI[(K) - (KTS)], VT_NS[(K) - (KTS)], VT_NG[(K) - (KTS)], VT_NH[(K) - (KTS)], VT_VI[(K) - (KTS)], VT_VS[(K) - (KTS)], VT_VG[(K) - (KTS)], VT_FI[(K) - (KTS)], VT_FS[(K) - (KTS)], VT_AI[(K) - (KTS)], VT_AS[(K) - (KTS)], VT_AG[(K) - (KTS)], VT_AH[(K) - (KTS)], VTI3M[(K) - (KTS)], SASPR[(K) - (KTS)], RHO[(K) - (KTS)]);
        }
        if (((QC3D[(K) - (KTS)] >= NTU_QSMALL) && (NC3D[(K) - (KTS)] >= NTU_NSMALL))) {
            MVDC[(K) - (KTS)] = NTU_POWF(((QC3D[(K) - (KTS)] * NTU_IAMW) / NC3D[(K) - (KTS)]), NTU_THRD);
            if ((MVDC[(K) - (KTS)] > NTU_DCR)) {
                QR3D[(K) - (KTS)] = (QR3D[(K) - (KTS)] + QC3D[(K) - (KTS)]);
                NR3D[(K) - (KTS)] = (NR3D[(K) - (KTS)] + NC3D[(K) - (KTS)]);
                QC3D[(K) - (KTS)] = 0.0f;
                NC3D[(K) - (KTS)] = 0.0f;
            }
        }
        if (((QR3D[(K) - (KTS)] >= NTU_QSMALL) && (NR3D[(K) - (KTS)] >= NTU_NSMALL))) {
            MVDR[(K) - (KTS)] = NTU_POWF(((QR3D[(K) - (KTS)] * NTU_IAMW) / NR3D[(K) - (KTS)]), NTU_THRD);
            if ((MVDR[(K) - (KTS)] < NTU_DCR)) {
                QC3D[(K) - (KTS)] = (QC3D[(K) - (KTS)] + QR3D[(K) - (KTS)]);
                NC3D[(K) - (KTS)] = (NC3D[(K) - (KTS)] + NR3D[(K) - (KTS)]);
                QR3D[(K) - (KTS)] = 0.0f;
                NR3D[(K) - (KTS)] = 0.0f;
            }
        }
        if (((QH3D[(K) - (KTS)] >= NTU_QSMALL) && (NH3D[(K) - (KTS)] >= NTU_NSMALL))) {
            MVDH[(K) - (KTS)] = NTU_POWF(((QH3D[(K) - (KTS)] * NTU_IAMH) / NH3D[(K) - (KTS)]), NTU_THRD);
            if ((MVDH[(K) - (KTS)] < NTU_DHMIN)) {
                QG3D[(K) - (KTS)] = (QG3D[(K) - (KTS)] + QH3D[(K) - (KTS)]);
                NG3D[(K) - (KTS)] = (NG3D[(K) - (KTS)] + NH3D[(K) - (KTS)]);
                VG3D[(K) - (KTS)] = (VG3D[(K) - (KTS)] + (QH3D[(K) - (KTS)] / NTU_RHOG0));
                QH3D[(K) - (KTS)] = 0.0f;
                NH3D[(K) - (KTS)] = 0.0f;
                if (((AH3D[(K) - (KTS)] >= NTU_ASMALL) && (NTU_AFAG_3M == 1))) {
                    AG3D[(K) - (KTS)] = (AG3D[(K) - (KTS)] + AH3D[(K) - (KTS)]);
                    AH3D[(K) - (KTS)] = 0.0f;
                }
            }
        }
        TEMPC = (QC3D[(K) - (KTS)] + QI3D[(K) - (KTS)]);
        if ((TEMPC > 0.0f)) {
            VTMEAN[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = (((VT_QC[(K) - (KTS)] * QC3D[(K) - (KTS)]) + (VT_QI[(K) - (KTS)] * QI3D[(K) - (KTS)])) / TEMPC);
        } else {
            VTMEAN[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = 0.0f;
        }
        TEMPR = (((QR3D[(K) - (KTS)] + QS3D[(K) - (KTS)]) + QG3D[(K) - (KTS)]) + QH3D[(K) - (KTS)]);
        if ((TEMPR > 0.0f)) {
            VTMEAN[((K) - (KTS)) + NTU_KMAX * ((2) - (1))] = (((((VT_QR[(K) - (KTS)] * QR3D[(K) - (KTS)]) + (VT_QS[(K) - (KTS)] * QS3D[(K) - (KTS)])) + (VT_QG[(K) - (KTS)] * QG3D[(K) - (KTS)])) + (VT_QH[(K) - (KTS)] * QH3D[(K) - (KTS)])) / TEMPR);
        } else {
            VTMEAN[((K) - (KTS)) + NTU_KMAX * ((2) - (1))] = 0.0f;
        }
        TEMPT = (TEMPC + TEMPR);
        if ((TEMPT > 0.0f)) {
            VTMEAN[((K) - (KTS)) + NTU_KMAX * ((3) - (1))] = (((VTMEAN[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] * TEMPC) + (VTMEAN[((K) - (KTS)) + NTU_KMAX * ((2) - (1))] * TEMPR)) / TEMPT);
        } else {
            VTMEAN[((K) - (KTS)) + NTU_KMAX * ((3) - (1))] = 0.0f;
        }
    } }
    ntu_ptflux(&QC3D[(1) - (KTS)], &VT_QC[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, CLODNCV);
    ntu_ptflux(&QR3D[(1) - (KTS)], &VT_QR[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, RAINNCV);
    ntu_ptflux(&QI3D[(1) - (KTS)], &VT_QI[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, ICENCV);
    ntu_ptflux(&QS3D[(1) - (KTS)], &VT_QS[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, SNOWNCV);
    ntu_ptflux(&QG3D[(1) - (KTS)], &VT_QG[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, GRAPNCV);
    ntu_ptflux(&QH3D[(1) - (KTS)], &VT_QH[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, HAILNCV);
    RAINNC = ((((((RAINNC + CLODNCV) + RAINNCV) + ICENCV) + SNOWNCV) + GRAPNCV) + HAILNCV);
    SNOWNC = ((SNOWNC + ICENCV) + SNOWNCV);
    GRAPNC = (GRAPNC + GRAPNCV);
    HAILNC = (HAILNC + HAILNCV);
    SR = ((((ICENCV + SNOWNCV) + GRAPNCV) + HAILNCV) / ((((((CLODNCV + RAINNCV) + ICENCV) + SNOWNCV) + GRAPNCV) + HAILNCV) + 1.0E-12f));
    ntu_flflux(&NC3D[(1) - (KTS)], &VT_NC[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
    ntu_flflux(&NR3D[(1) - (KTS)], &VT_NR[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
    ntu_flflux(&NI3D[(1) - (KTS)], &VT_NI[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
    ntu_flflux(&NS3D[(1) - (KTS)], &VT_NS[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
    ntu_flflux(&NG3D[(1) - (KTS)], &VT_NG[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
    ntu_flflux(&NH3D[(1) - (KTS)], &VT_NH[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
    ntu_ptflux(&VI3D[(1) - (KTS)], &VT_VI[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, VINCV);
    ntu_ptflux(&VS3D[(1) - (KTS)], &VT_VS[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, VSNCV);
    ntu_ptflux(&VG3D[(1) - (KTS)], &VT_VG[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, VGNCV);
    ntu_ptflux(&FI3D[(1) - (KTS)], &VT_FI[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, FINCV);
    ntu_ptflux(&FS3D[(1) - (KTS)], &VT_FS[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, FSNCV);
    ntu_ptflux(&AI3D[(1) - (KTS)], &VT_AI[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, AINCV);
    ntu_ptflux(&AS3D[(1) - (KTS)], &VT_AS[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, ASNCV);
    ntu_ptflux(&AG3D[(1) - (KTS)], &VT_AG[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, AGNCV);
    ntu_ptflux(&AH3D[(1) - (KTS)], &VT_AH[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, AHNCV);
    ntu_ptflux(&I3M3D[(1) - (KTS)], &VTI3M[(1) - (KTS)], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT, DTMN, I3MNCV);
    IV0 = 0;
    { const int ntu_hi_iv = NTU_NAER;
    for (IV = 1; IV <= ntu_hi_iv; ++IV) {
        if ((NTU_NAERN[(IV) - (1)] > 3)) {
            ntu_flflux(&AERO[((1) - (KTS)) + NTU_KMAX * (((IV0 + 3)) - (1))], &VTMEAN[((1) - (KTS)) + NTU_KMAX * ((1) - (1))], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
            ntu_flflux(&AERO[((1) - (KTS)) + NTU_KMAX * (((IV0 + 4)) - (1))], &VTMEAN[((1) - (KTS)) + NTU_KMAX * ((2) - (1))], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
        } else if ((NTU_NAERN[(IV) - (1)] == 3)) {
            ntu_flflux(&AERO[((1) - (KTS)) + NTU_KMAX * (((IV0 + 3)) - (1))], &VTMEAN[((1) - (KTS)) + NTU_KMAX * ((3) - (1))], &RHO[(1) - (KTS)], &DZ3D[(1) - (KTS)], KTE, DT);
        }
        IV0 = (IV0 + NTU_NAERN[(IV) - (1)]);
    } }
    { const int ntu_hi_k = KTE;
    for (K = KTS; K <= ntu_hi_k; ++K) {
        TK3D[(K) - (KTS)] = TAIR[(K) - (KTS)];
        QV3D[(K) - (KTS)] = QVAP[(K) - (KTS)];
        if (((QC3D[(K) - (KTS)] >= NTU_QSMALL) && (NC3D[(K) - (KTS)] >= NTU_NSMALL))) {
            ntu_solve_afac(AC, TK3D[(K) - (KTS)], QC3D[(K) - (KTS)], NC3D[(K) - (KTS)], LAMC[(K) - (KTS)], MVDC[(K) - (KTS)], AFAC[(K) - (KTS)]);
        } else {
            QC3D[(K) - (KTS)] = 0.0f;
            NC3D[(K) - (KTS)] = 0.0f;
            MVDC[(K) - (KTS)] = 0.0f;
            AFAC[(K) - (KTS)] = 0.0f;
            LAMC[(K) - (KTS)] = 0.0f;
        }
        if (((QR3D[(K) - (KTS)] >= NTU_QSMALL) && (NR3D[(K) - (KTS)] >= NTU_NSMALL))) {
            ntu_solve_afar(TK3D[(K) - (KTS)], QR3D[(K) - (KTS)], NR3D[(K) - (KTS)], LAMR[(K) - (KTS)], MVDR[(K) - (KTS)], AFAR[(K) - (KTS)]);
        } else {
            QR3D[(K) - (KTS)] = 0.0f;
            NR3D[(K) - (KTS)] = 0.0f;
            MVDR[(K) - (KTS)] = 0.0f;
            LAMR[(K) - (KTS)] = 0.0f;
            AFAR[(K) - (KTS)] = 0.0f;
        }
        if (((QI3D[(K) - (KTS)] >= NTU_QSMALL) && (NI3D[(K) - (KTS)] >= NTU_NSMALL))) {
            ntu_solve_afai(TK3D[(K) - (KTS)], PRES[(K) - (KTS)], RHO[(K) - (KTS)], QV3D[(K) - (KTS)], QI3D[(K) - (KTS)], NI3D[(K) - (KTS)], VI3D[(K) - (KTS)], FI3D[(K) - (KTS)], AI3D[(K) - (KTS)], I3M3D[(K) - (KTS)], ADAGR[(K) - (KTS)], ZETA[(K) - (KTS)], LAMI[(K) - (KTS)], AFAI[(K) - (KTS)], MVDI[(K) - (KTS)], RHOI[(K) - (KTS)], AMI[(K) - (KTS)], BMI[(K) - (KTS)], AVI[(K) - (KTS)], BVI[(K) - (KTS)], BEST[(K) - (KTS)]);
            IASPR[(K) - (KTS)] = (FI3D[(K) - (KTS)] / I3M3D[(K) - (KTS)]);
        } else {
            QI3D[(K) - (KTS)] = 0.0f;
            NI3D[(K) - (KTS)] = 0.0f;
            MVDI[(K) - (KTS)] = 0.0f;
            I3M3D[(K) - (KTS)] = 0.0f;
            FI3D[(K) - (KTS)] = 0.0f;
            VI3D[(K) - (KTS)] = 0.0f;
            AFAI[(K) - (KTS)] = 0.0f;
            IASPR[(K) - (KTS)] = 1.0f;
            ADAGR[(K) - (KTS)] = 1.0f;
            RHOI[(K) - (KTS)] = 0.0f;
            AI3D[(K) - (KTS)] = 0.0f;
            LAMI[(K) - (KTS)] = 0.0f;
        }
        if ((AS3D[(K) - (KTS)] < NTU_ASMALL)) {
            AS3D[(K) - (KTS)] = 0.0f;
        }
        if (((QS3D[(K) - (KTS)] >= NTU_QSMALL) && (NS3D[(K) - (KTS)] >= NTU_NSMALL))) {
            ntu_solve_afas(TK3D[(K) - (KTS)], RHO[(K) - (KTS)], QS3D[(K) - (KTS)], QC3D[(K) - (KTS)], NS3D[(K) - (KTS)], VS3D[(K) - (KTS)], FS3D[(K) - (KTS)], AS3D[(K) - (KTS)], AFAS[(K) - (KTS)], LAMS[(K) - (KTS)], MVDS[(K) - (KTS)], RHOS[(K) - (KTS)], SASPR[(K) - (KTS)], AMS[(K) - (KTS)], AVS[(K) - (KTS)], BVS[(K) - (KTS)]);
        } else {
            QS3D[(K) - (KTS)] = 0.0f;
            NS3D[(K) - (KTS)] = 0.0f;
            VS3D[(K) - (KTS)] = 0.0f;
            AS3D[(K) - (KTS)] = 0.0f;
            MVDS[(K) - (KTS)] = 0.0f;
            RHOS[(K) - (KTS)] = 0.0f;
            AFAS[(K) - (KTS)] = 0.0f;
            LAMS[(K) - (KTS)] = 0.0f;
            FS3D[(K) - (KTS)] = 0.0f;
            SASPR[(K) - (KTS)] = 1.0f;
        }
        if ((AG3D[(K) - (KTS)] < NTU_ASMALL)) {
            AG3D[(K) - (KTS)] = 0.0f;
        }
        if (((QG3D[(K) - (KTS)] >= NTU_QSMALL) && (NG3D[(K) - (KTS)] >= NTU_NSMALL))) {
            ntu_solve_afag(TK3D[(K) - (KTS)], RHO[(K) - (KTS)], QG3D[(K) - (KTS)], QC3D[(K) - (KTS)], NG3D[(K) - (KTS)], VG3D[(K) - (KTS)], AG3D[(K) - (KTS)], LAMG[(K) - (KTS)], AFAG[(K) - (KTS)], MVDG[(K) - (KTS)], RHOG[(K) - (KTS)], AMG[(K) - (KTS)], AVG[(K) - (KTS)], BVG[(K) - (KTS)]);
        } else {
            QG3D[(K) - (KTS)] = 0.0f;
            NG3D[(K) - (KTS)] = 0.0f;
            MVDG[(K) - (KTS)] = 0.0f;
            VG3D[(K) - (KTS)] = 0.0f;
            AG3D[(K) - (KTS)] = 0.0f;
            AFAG[(K) - (KTS)] = 0.0f;
            RHOG[(K) - (KTS)] = 0.0f;
            LAMG[(K) - (KTS)] = 0.0f;
        }
        if ((AH3D[(K) - (KTS)] < NTU_ASMALL)) {
            AH3D[(K) - (KTS)] = 0.0f;
        }
        if (((QH3D[(K) - (KTS)] >= NTU_QSMALL) && (NH3D[(K) - (KTS)] >= NTU_NSMALL))) {
            ntu_solve_afah(TK3D[(K) - (KTS)], RHO[(K) - (KTS)], QH3D[(K) - (KTS)], NH3D[(K) - (KTS)], AH3D[(K) - (KTS)], LAMH[(K) - (KTS)], AFAH[(K) - (KTS)], MVDH[(K) - (KTS)], AVH[(K) - (KTS)], BVH[(K) - (KTS)]);
        } else {
            QH3D[(K) - (KTS)] = 0.0f;
            NH3D[(K) - (KTS)] = 0.0f;
            MVDH[(K) - (KTS)] = 0.0f;
            LAMH[(K) - (KTS)] = 0.0f;
            AH3D[(K) - (KTS)] = 0.0f;
            AFAH[(K) - (KTS)] = 0.0f;
        }
        if (((QC3D[(K) - (KTS)] + QR3D[(K) - (KTS)]) < NTU_QSMALL)) {
            AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = ntu_fmax(NTU_RLIMIT, (AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] + AERO[((K) - (KTS)) + NTU_KMAX * ((3) - (1))]));
            AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = ntu_fmin(AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))], AERO[((K) - (KTS)) + NTU_KMAX * ((2) - (1))]);
            AERO[((K) - (KTS)) + NTU_KMAX * ((3) - (1))] = 0.0f;
        }
        if (((((QR3D[(K) - (KTS)] + QS3D[(K) - (KTS)]) + QG3D[(K) - (KTS)]) + QH3D[(K) - (KTS)]) < NTU_QSMALL)) {
            AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = ntu_fmax(NTU_RLIMIT, (AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] + AERO[((K) - (KTS)) + NTU_KMAX * ((4) - (1))]));
            AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))] = ntu_fmin(AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))], AERO[((K) - (KTS)) + NTU_KMAX * ((2) - (1))]);
            AERO[((K) - (KTS)) + NTU_KMAX * ((4) - (1))] = 0.0f;
        }
        QDCN3D[(K) - (KTS)] = AERO[((K) - (KTS)) + NTU_KMAX * ((1) - (1))];
        QTCN3D[(K) - (KTS)] = AERO[((K) - (KTS)) + NTU_KMAX * ((2) - (1))];
        QCCN3D[(K) - (KTS)] = AERO[((K) - (KTS)) + NTU_KMAX * ((3) - (1))];
        QRCN3D[(K) - (KTS)] = AERO[((K) - (KTS)) + NTU_KMAX * ((4) - (1))];
        QNIN3D[(K) - (KTS)] = AERO[((K) - (KTS)) + NTU_KMAX * ((5) - (1))];
    } }
}

// ---- END transcription ----

// ---------------------------------------------------------------------------
// Column drivers: WRF MP_NTU (module_mp_ntu.F:1689-1811) and NTU_INIT /
// INIT_AEROSOL (:255-314, :1569-1685), one GPU thread per column.
//
// Field layout is the engine's: (nz, ny, nx) contiguous, k = 0 at the
// surface; a column is c = j*nx + i and level k of it lives at k*ncol + c.
// WRF's MP_NTU turns the column over (K = 1 is the model top); the gather
// below does the same flip, NK = KTE-K+1.
// ---------------------------------------------------------------------------
enum NtuField3 {
    NTU_F_TH = 0, NTU_F_P, NTU_F_DZ, NTU_F_PII,
    NTU_F_QV, NTU_F_QC, NTU_F_QR, NTU_F_QI, NTU_F_QS, NTU_F_QG, NTU_F_QH,
    NTU_F_NC, NTU_F_NR, NTU_F_NI, NTU_F_NS, NTU_F_NG, NTU_F_NH,
    NTU_F_QDCN, NTU_F_QTCN, NTU_F_QCCN, NTU_F_QRCN, NTU_F_QNIN,
    NTU_F_FI, NTU_F_FS, NTU_F_VI, NTU_F_VS, NTU_F_VG,
    NTU_F_AI, NTU_F_AS, NTU_F_AG, NTU_F_AH, NTU_F_I3M,
    NTU_F_W,                      // (nz+1) w levels
    NTU_NFIELD3
};
enum NtuField2 {
    NTU_S_RAINNC = 0, NTU_S_RAINNCV, NTU_S_SNOWNC, NTU_S_SNOWNCV,
    NTU_S_GRAPNC, NTU_S_GRAPNCV, NTU_S_HAILNC, NTU_S_HAILNCV, NTU_S_SR,
    NTU_NFIELD2
};

// Audit-only original layout. Production uses ntu_parallel.cuh.
#ifdef NTU_REFERENCE_COLUMN_LAYOUT
// MP_NTU for one column.
NTU_DEV void ntu_mp_column(NtuAeroConst* __restrict__ AC,
                           float* const* __restrict__ f3,
                           float* const* __restrict__ f2,
                           size_t c, size_t ncol, int nz, float DT_MP)
{
    const int KTS = 1;
    const int KTE = nz;
    float TK1D[NTU_KMAX], P1D[NTU_KMAX], W1D[NTU_KMAX], S1D[NTU_KMAX],
          DZ1D[NTU_KMAX], QV1D[NTU_KMAX], QC1D[NTU_KMAX], QR1D[NTU_KMAX],
          QI1D[NTU_KMAX], QS1D[NTU_KMAX], QG1D[NTU_KMAX], QH1D[NTU_KMAX],
          NC1D[NTU_KMAX], NR1D[NTU_KMAX], NI1D[NTU_KMAX], NS1D[NTU_KMAX],
          NG1D[NTU_KMAX], NH1D[NTU_KMAX], VI1D[NTU_KMAX], VS1D[NTU_KMAX],
          VG1D[NTU_KMAX], FI1D[NTU_KMAX], FS1D[NTU_KMAX], AI1D[NTU_KMAX],
          AS1D[NTU_KMAX], AG1D[NTU_KMAX], AH1D[NTU_KMAX], I3M1D[NTU_KMAX],
          QDCN1D[NTU_KMAX], QTCN1D[NTU_KMAX], QCCN1D[NTU_KMAX],
          QRCN1D[NTU_KMAX], QNIN1D[NTU_KMAX];
    float* TH = f3[NTU_F_TH];
    const float* PII = f3[NTU_F_PII];
    const float* W = f3[NTU_F_W];
    float DT = DT_MP;
    float DTMN = DT_MP / 60.0f;
    float CLODNCV = 0.0f, ICENCV = 0.0f, VINCV = 0.0f, VSNCV = 0.0f,
          VGNCV = 0.0f, FINCV = 0.0f, FSNCV = 0.0f, AINCV = 0.0f,
          ASNCV = 0.0f, AGNCV = 0.0f, AHNCV = 0.0f, I3MNCV = 0.0f;
    float& RAINNC = f2[NTU_S_RAINNC][c];
    float& RAINNCV = f2[NTU_S_RAINNCV][c];
    float& SNOWNC = f2[NTU_S_SNOWNC][c];
    float& SNOWNCV = f2[NTU_S_SNOWNCV][c];
    float& GRAPNC = f2[NTU_S_GRAPNC][c];
    float& GRAPNCV = f2[NTU_S_GRAPNCV][c];
    float& HAILNC = f2[NTU_S_HAILNC][c];
    float& HAILNCV = f2[NTU_S_HAILNCV][c];
    float& SR = f2[NTU_S_SR][c];
    RAINNCV = 0.0f; SNOWNCV = 0.0f; GRAPNCV = 0.0f; HAILNCV = 0.0f;
    SR = 0.0f;
    for (int K = KTS; K <= KTE; ++K) {
        const int NK = KTE - K + 1;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        const size_t above = (size_t)NK * ncol + c;
        const int k = K - 1;
        S1D[k] = 0.0f;
        TK1D[k] = TH[at] * PII[at];
        W1D[k] = 0.5f * (W[at] + W[above]);
        QV1D[k] = f3[NTU_F_QV][at];
        DZ1D[k] = f3[NTU_F_DZ][at];
        P1D[k] = f3[NTU_F_P][at];
        QC1D[k] = f3[NTU_F_QC][at];
        QR1D[k] = f3[NTU_F_QR][at];
        QI1D[k] = f3[NTU_F_QI][at];
        QS1D[k] = f3[NTU_F_QS][at];
        QG1D[k] = f3[NTU_F_QG][at];
        QH1D[k] = f3[NTU_F_QH][at];
        NC1D[k] = f3[NTU_F_NC][at];
        NR1D[k] = f3[NTU_F_NR][at];
        NI1D[k] = f3[NTU_F_NI][at];
        NS1D[k] = f3[NTU_F_NS][at];
        NG1D[k] = f3[NTU_F_NG][at];
        NH1D[k] = f3[NTU_F_NH][at];
        VI1D[k] = f3[NTU_F_VI][at];
        VS1D[k] = f3[NTU_F_VS][at];
        VG1D[k] = f3[NTU_F_VG][at];
        FI1D[k] = f3[NTU_F_FI][at];
        FS1D[k] = f3[NTU_F_FS][at];
        AI1D[k] = f3[NTU_F_AI][at];
        AS1D[k] = f3[NTU_F_AS][at];
        AG1D[k] = f3[NTU_F_AG][at];
        AH1D[k] = f3[NTU_F_AH][at];
        I3M1D[k] = f3[NTU_F_I3M][at];
        QDCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QDCN][at]);
        QTCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QTCN][at]);
        QCCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QCCN][at]);
        QRCN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QRCN][at]);
        QNIN1D[k] = ntu_fmax(NTU_RLIMIT, f3[NTU_F_QNIN][at]);
    }
    ntu_ntu_micro(AC, TK1D, QV1D, DZ1D, P1D, S1D, W1D, SR, ICENCV, VINCV,
                  VSNCV, VGNCV, FINCV, FSNCV, AINCV, ASNCV, AGNCV, AHNCV,
                  I3MNCV, CLODNCV, RAINNC, RAINNCV, SNOWNC, SNOWNCV, GRAPNC,
                  GRAPNCV, HAILNC, HAILNCV, DT, DTMN, QC1D, QR1D, QI1D, QS1D,
                  QG1D, QH1D, NC1D, NR1D, NI1D, NS1D, NG1D, NH1D, VI1D, VS1D,
                  VG1D, FI1D, FS1D, AI1D, AS1D, AG1D, AH1D, I3M1D, QDCN1D,
                  QTCN1D, QCCN1D, QRCN1D, QNIN1D, KTS, KTE);
    for (int K = KTS; K <= KTE; ++K) {
        const int NK = KTE - K + 1;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        const int k = K - 1;
        TH[at] = TK1D[k] / PII[at];
        f3[NTU_F_QV][at] = ntu_fmax(0.0f, QV1D[k]);
        f3[NTU_F_QC][at] = ntu_fmax(0.0f, QC1D[k]);
        f3[NTU_F_QR][at] = ntu_fmax(0.0f, QR1D[k]);
        f3[NTU_F_QI][at] = ntu_fmax(0.0f, QI1D[k]);
        f3[NTU_F_QS][at] = ntu_fmax(0.0f, QS1D[k]);
        f3[NTU_F_QG][at] = ntu_fmax(0.0f, QG1D[k]);
        f3[NTU_F_QH][at] = ntu_fmax(0.0f, QH1D[k]);
        f3[NTU_F_NC][at] = ntu_fmax(0.0f, NC1D[k]);
        f3[NTU_F_NR][at] = ntu_fmax(0.0f, NR1D[k]);
        f3[NTU_F_NI][at] = ntu_fmax(0.0f, NI1D[k]);
        f3[NTU_F_NS][at] = ntu_fmax(0.0f, NS1D[k]);
        f3[NTU_F_NG][at] = ntu_fmax(0.0f, NG1D[k]);
        f3[NTU_F_NH][at] = ntu_fmax(0.0f, NH1D[k]);
        f3[NTU_F_VI][at] = ntu_fmax(0.0f, VI1D[k]);
        f3[NTU_F_VS][at] = ntu_fmax(0.0f, VS1D[k]);
        f3[NTU_F_VG][at] = ntu_fmax(0.0f, VG1D[k]);
        f3[NTU_F_FI][at] = ntu_fmax(0.0f, FI1D[k]);
        f3[NTU_F_FS][at] = ntu_fmax(0.0f, FS1D[k]);
        f3[NTU_F_AI][at] = ntu_fmax(0.0f, AI1D[k]);
        f3[NTU_F_AS][at] = ntu_fmax(0.0f, AS1D[k]);
        f3[NTU_F_AG][at] = ntu_fmax(0.0f, AG1D[k]);
        f3[NTU_F_AH][at] = ntu_fmax(0.0f, AH1D[k]);
        f3[NTU_F_I3M][at] = ntu_fmax(0.0f, I3M1D[k]);
        f3[NTU_F_QDCN][at] = ntu_fmax(0.0f, QDCN1D[k]);
        f3[NTU_F_QTCN][at] = ntu_fmax(0.0f, QTCN1D[k]);
        f3[NTU_F_QCCN][at] = ntu_fmax(0.0f, QCCN1D[k]);
        f3[NTU_F_QRCN][at] = ntu_fmax(0.0f, QRCN1D[k]);
        f3[NTU_F_QNIN][at] = ntu_fmax(0.0f, QNIN1D[k]);
    }
}

#endif // NTU_REFERENCE_COLUMN_LAYOUT

// NTU_INIT + INIT_AEROSOL for one column.  phb/ph are on the nz+1 w levels,
// p/pb/alt/qv on the nz mass levels; the five aerosol fields are written in
// place (left as given when restart != 0, exactly as WRF).
//
// Declared divergence (fix): INIT_AEROSOL's first level loop runs
// K = KTS..KTE with KTE = kde, one level past the nz mass levels NTU_INIT
// filled, so WRF compares an uninitialised pressure word against 850 hPa and
// can move the boundary-layer top K_PBL to KTE-2.  The loop here stops at the
// last mass level; every other statement is WRF's.
NTU_DEV void ntu_init_column(NtuAeroConst* __restrict__ AC,
                             const float* __restrict__ phb,
                             const float* __restrict__ ph,
                             const float* __restrict__ p,
                             const float* __restrict__ pb,
                             const float* __restrict__ alt,
                             const float* __restrict__ qv,
                             float* const* __restrict__ aero,
                             size_t c, size_t ncol, int nz, int restart)
{
    const int KTS = 1;
    const int KTE = nz + 1;            // NTU_INIT receives kte = kde
    float DZ[NTU_KMAX + 1], DZ8W[NTU_KMAX + 1], RHO[NTU_KMAX + 1],
          P_PHY[NTU_KMAX + 1];
    float QAERO[NTU_KMAX + 1][5];
    for (int K = KTS; K <= KTE; ++K) {
        const size_t at = (size_t)(K - 1) * ncol + c;
        DZ[K - 1] = __fdiv_rn((phb[at] + ph[at]), NTU_G);
    }
    for (int K = KTS; K <= KTE - 1; ++K) {
        const int NK = KTE - K;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        DZ8W[K - 1] = DZ[NK] - DZ[NK - 1];
        RHO[K - 1] = (1.0f / alt[at]) * (1.0f + qv[at]);
        P_PHY[K - 1] = p[at] + pb[at];
        for (int s = 0; s < 5; ++s) QAERO[K - 1][s] = aero[s][at];
    }
    if (!restart) {
        const float P_PBL = 85000.0f;
        const float BACKIN = 4.0E5f;
        float XMASS[NTU_NAER];
        float RASH[NTU_NAER];
        for (int IA = 1; IA <= NTU_NAERT; ++IA)
            for (int K = KTS; K <= KTE - 1; ++K) QAERO[K - 1][IA - 1] = 0.0f;
        for (int J = 1; J <= NTU_NAER; ++J) {
            XMASS[J - 1] = 0.0f;
            for (int IM = 1; IM <= NTU_NCCN; ++IM) {
                const int ij = (IM - 1) + 3 * (J - 1);
                float XMAS = (AC->ZCCN[ij] * NTU_POWF(AC->CNMOD[ij], 3.0f))
                             * NTU_EXPF(4.5f * ntu_sq(AC->CNSTD[ij]));
                XMASS[J - 1] = XMASS[J - 1] + XMAS;
            }
            XMASS[J - 1] = ((XMASS[J - 1] * NTU_C4PI3) * AC->DNAS[J - 1])
                           * AC->SENS[J - 1];
        }
        // ID_DUST*ID_IN == 0: no dust, background ice nuclei.
        float DUST_IN0 = BACKIN;
        float DUST_IN = 0.0f, DAIR = 0.0f;
        int K_PBL = 1;
        for (int K = KTS; K <= KTE - 1; ++K) {
            int IV = 1;
            for (int IA = 1; IA <= NTU_NAER; ++IA) {
                QAERO[K - 1][IV - 1] = XMASS[IA - 1] / RHO[K - 1];
                RASH[IA - 1] = 1.0f / AC->ASH[IA - 1];
                DUST_IN = DUST_IN0;
                if (IA == NTU_ID_IN) {
                    // ID_DUST == 0: background value, doubled scale height.
                    DUST_IN = BACKIN;
                    RASH[IA - 1] = RASH[IA - 1] * 0.5f;
                    DAIR = RHO[K - 1];
                    DUST_IN = ntu_fmax(BACKIN, DUST_IN);
                    QAERO[K - 1][IV - 1] = DUST_IN / DAIR;
                }
                IV = IV + NTU_NAERN[IA - 1];
            }
            if (P_PHY[K - 1] < P_PBL) K_PBL = K;
        }
        K_PBL = ntu_imin(K_PBL, KTE - 3 + 1);
        float ZH = 0.0f, DZ0 = 0.0f;
        for (int K = K_PBL - 1; K >= 1; --K) {
            DAIR = RHO[K - 1];
            ZH = ZH + (DZ0 + DZ8W[K - 1]) * 0.5f;
            DZ0 = DZ8W[K - 1];
            int IV = 1;
            for (int IA = 1; IA <= NTU_NAER; ++IA) {
                QAERO[K - 1][IV - 1] = QAERO[K - 1][IV - 1]
                                       * NTU_EXPF(-(RASH[IA - 1] * ZH));
                IV = IV + NTU_NAERN[IA - 1];
            }
        }
        int IV = 0;
        for (int IA = 1; IA <= NTU_NAER; ++IA) {
            const int top = ntu_imin(2, NTU_NAERN[IA - 1]);
            for (int IB = 2; IB <= top; ++IB)
                for (int K = KTS; K <= KTE - 1; ++K)
                    QAERO[K - 1][IV + IB - 1] = QAERO[K - 1][IV];
            IV = IV + NTU_NAERN[IA - 1];
        }
        (void)DAIR;
    }
    for (int K = KTS; K <= KTE - 1; ++K) {
        const int NK = KTE - K;
        const size_t at = (size_t)(NK - 1) * ncol + c;
        for (int s = 0; s < 5; ++s) aero[s][at] = QAERO[K - 1][s];
    }
}

// ---------------------------------------------------------------------------
// 10 cm radar reflectivity from NTU's own predicted size distributions.
//
// WRF's NTU arm writes no REFL_10CM; this diagnostic is WOOF's.  Each
// category's gamma distribution n(D) ~ D^AFA exp(-LAM D) comes from the
// scheme's own SOLVE_AFA* routines (run on copies, so the state is not
// touched), and contributes its Rayleigh sixth moment
//   M6 = N Gamma(AFA+7) / (Gamma(AFA+1) LAM^6)        [m^6 per kg of air].
// Frozen categories scale by |K_ice|^2/|K_w|^2 (0.176/0.93) and by the
// square of their predicted bulk density over solid ice (917 kg m-3), the
// mass-squared form that holds for any particle shape.  The floor is the
// -35 dBZ the reflectivity family uses.
NTU_DEV double ntu_m6(float N, float AFA, float LAM) {
    return (double)N * exp((double)(ntu_gamln(AFA + 7.0f) - ntu_gamln(AFA + 1.0f))
                           - 6.0 * log((double)LAM));
}

NTU_DEV float ntu_refl_cell(NtuAeroConst* __restrict__ AC,
                            float* const* __restrict__ f3, size_t at)
{
    const double KICE = 0.176 / 0.93;
    const float T = f3[NTU_F_TH][at] * f3[NTU_F_PII][at];
    const float P = f3[NTU_F_P][at];
    const float QV = f3[NTU_F_QV][at];
    const float RHO = __fdiv_rn(P / (T * (1.0f + 0.61f * QV)), NTU_R);
    double z = 0.0;
    float q, n, lam = 0.0f, mvd = 0.0f, afa = 0.0f;
    q = f3[NTU_F_QC][at]; n = f3[NTU_F_NC][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        ntu_solve_afac(AC, T, q, n, lam, mvd, afa);
        if (lam > 0.0f) z += ntu_m6(n, afa, lam);
    }
    q = f3[NTU_F_QR][at]; n = f3[NTU_F_NR][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        ntu_solve_afar(T, q, n, lam, mvd, afa);
        if (lam > 0.0f) z += ntu_m6(n, afa, lam);
    }
    q = f3[NTU_F_QI][at]; n = f3[NTU_F_NI][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float vi = f3[NTU_F_VI][at], fi = f3[NTU_F_FI][at];
        float ai = f3[NTU_F_AI][at], i3 = f3[NTU_F_I3M][at];
        float adagr = 1.0f, zeta = 0.0f, rhoi = 0.0f, ami = 0.0f, bmi = 0.0f,
              avi = 0.0f, bvi = 0.0f, best = 0.0f;
        ntu_solve_afai(T, P, RHO, QV, q, n, vi, fi, ai, i3, adagr, zeta, lam,
                       afa, mvd, rhoi, ami, bmi, avi, bvi, best);
        if (lam > 0.0f) {
            const double r = (double)rhoi / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    q = f3[NTU_F_QS][at]; n = f3[NTU_F_NS][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float vs = f3[NTU_F_VS][at], fs = f3[NTU_F_FS][at],
              as2 = f3[NTU_F_AS][at];
        float rhos = 0.0f, saspr = 1.0f, ams = 0.0f, avs = 0.0f, bvs = 0.0f;
        ntu_solve_afas(T, RHO, q, f3[NTU_F_QC][at], n, vs, fs, as2, afa, lam,
                       mvd, rhos, saspr, ams, avs, bvs);
        if (lam > 0.0f) {
            const double r = (double)rhos / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    q = f3[NTU_F_QG][at]; n = f3[NTU_F_NG][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float vg = f3[NTU_F_VG][at], ag = f3[NTU_F_AG][at];
        float rhog = 0.0f, amg = 0.0f, avg = 0.0f, bvg = 0.0f;
        ntu_solve_afag(T, RHO, q, f3[NTU_F_QC][at], n, vg, ag, lam, afa, mvd,
                       rhog, amg, avg, bvg);
        if (lam > 0.0f) {
            const double r = (double)rhog / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    q = f3[NTU_F_QH][at]; n = f3[NTU_F_NH][at];
    if (q >= NTU_QSMALL && n >= NTU_NSMALL) {
        float ah = f3[NTU_F_AH][at], avh = 0.0f, bvh = 0.0f;
        ntu_solve_afah(T, RHO, q, n, ah, lam, afa, mvd, avh, bvh);
        if (lam > 0.0f) {
            const double r = (double)NTU_RHOH / 917.0;
            z += KICE * r * r * ntu_m6(n, afa, lam);
        }
    }
    const double zmm = z * (double)RHO * 1.0e18;   // mm^6 m^-3
    return zmm > 3.1622776601683794e-4 ? (float)(10.0 * log10(zmm)) : -35.0f;
}

#ifndef NTU_HOST
extern "C" __global__ void ntu_init_kernel(
    NtuAeroConst* AC, const float* phb, const float* ph, const float* p,
    const float* pb, const float* alt, const float* qv,
    float* const* aero, long long ncol, int nz, int restart,
    unsigned int* status)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    if (nz > NTU_KMAX) { atomicOr(status, NTU_ERR_KMAX); NTU_FATAL(NTU_ERR_KMAX); return; }
    ntu_init_column(AC, phb, ph, p, pb, alt, qv, aero, (size_t)c,
                    (size_t)ncol, nz, restart);
}

extern "C" __global__ void ntu_refl_kernel(
    NtuAeroConst* AC, float* const* f3, long long ncell, float* refl)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    refl[i] = ntu_refl_cell(AC, f3, (size_t)i);
}

#ifdef NTU_REFERENCE_COLUMN_LAYOUT
extern "C" __global__ void ntu_mp_kernel(
    NtuAeroConst* AC, float* const* f3, float* const* f2, long long ncol,
    int nz, float dt_mp, unsigned int* status)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    if (nz > NTU_KMAX) { atomicOr(status, NTU_ERR_KMAX); NTU_FATAL(NTU_ERR_KMAX); return; }
    ntu_mp_column(AC, f3, f2, (size_t)c, (size_t)ncol, nz, dt_mp);
}
#endif // NTU_REFERENCE_COLUMN_LAYOUT
#endif
