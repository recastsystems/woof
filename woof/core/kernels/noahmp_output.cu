// WRF v4.6.1 d66e442f, phys/module_diag_misc.F:641-846.
// Original notice: licenses/LICENSE-WRF-public-domain.txt.
// Generated offline from pinned expressions. Native FP32 tree retained.
#ifdef WRF_EXACT
#define AD(a,b) __fadd_rn(a,b)
#define SU(a,b) __fsub_rn(a,b)
#define MU(a,b) __fmul_rn(a,b)
#define DV(a,b) __fdiv_rn(a,b)
#else
#define AD(a,b) ((a)+(b))
#define SU(a,b) ((a)-(b))
#define MU(a,b) ((a)*(b))
#define DV(a,b) ((a)/(b))
#endif
extern "C" __global__ void noahmp_output_calc(
float* acsagv,
float* acsagb,
float* acirg,
float* acshg,
float* acevg,
float* acghv,
float* acpahg,
float* acirb,
float* acshb,
float* acevb,
float* acghb,
float* acpahb,
float* acsav,
float* acirc,
float* acshc,
float* acevc,
float* actr,
float* acpahv,
float* acswdnlsm,
float* acswuplsm,
float* aclwdnlsm,
float* aclwuplsm,
float* acshflsm,
float* aclhflsm,
float* acghflsm,
float* acpahlsm,
float* acints,
float* acintr,
float* acdrips,
float* acdripr,
float* acthros,
float* acthror,
float* acetlsm,
float* acsnmelt,
float* acsnsub,
float* acsnfro,
float* acsubc,
float* acfroc,
float* acevac,
float* acdewc,
float* acfrzc,
float* acmeltc,
float* acsnbot,
float* acponding,
float* acrainlsm,
float* acsnowlsm,
float* acrainsnow,
float* acrunsb,
float* acrunsf,
float* acqlat,
float* acqrf,
float* acrech,
float* acqspring,
float* acecan,
float* acetran,
float* acedir,
float* aceflxb,
float* accanhs,
const float* sag,
const float* irg,
const float* shg,
const float* evg,
const float* ghv,
const float* pahg,
const float* irb,
const float* shb,
const float* evb,
const float* ghb,
const float* pahb,
const float* sav,
const float* ircmp,
const float* shc,
const float* evc,
const float* tr,
const float* pahv,
const float* swdown,
const float* albedo,
const float* emiss,
const float* glw,
const float* trad,
const float* qints,
const float* qintr,
const float* qdrips,
const float* qdripr,
const float* qthros,
const float* qthror,
const float* qsnsub,
const float* qsnfro,
const float* qsubc,
const float* qfroc,
const float* qevac,
const float* qdewc,
const float* qfrzc,
const float* qmeltc,
const float* qsnbot,
const float* ponding,
const float* rainlsm,
const float* snowlsm,
const float* runsb,
const float* runsf,
const float* fpice,
const float* fveg,
const float* qlat,
const float* qrf,
const float* deeprech,
const float* qspring,
const float* ecan,
const float* etran,
const float* edir,
const float* eflxb,
const float* canhs,
const float* qmelt,
const int* isnow,
float dt,
int n,
int reset,
int opt_run,
int groundwater_due){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
if(reset){
acsagv[c]=0.0f;
acsagb[c]=0.0f;
acirg[c]=0.0f;
acshg[c]=0.0f;
acevg[c]=0.0f;
acghv[c]=0.0f;
acpahg[c]=0.0f;
acirb[c]=0.0f;
acshb[c]=0.0f;
acevb[c]=0.0f;
acghb[c]=0.0f;
acpahb[c]=0.0f;
acsav[c]=0.0f;
acirc[c]=0.0f;
acshc[c]=0.0f;
acevc[c]=0.0f;
actr[c]=0.0f;
acpahv[c]=0.0f;
acswdnlsm[c]=0.0f;
acswuplsm[c]=0.0f;
aclwdnlsm[c]=0.0f;
aclwuplsm[c]=0.0f;
acshflsm[c]=0.0f;
aclhflsm[c]=0.0f;
acghflsm[c]=0.0f;
acpahlsm[c]=0.0f;
acints[c]=0.0f;
acintr[c]=0.0f;
acdrips[c]=0.0f;
acdripr[c]=0.0f;
acthros[c]=0.0f;
acthror[c]=0.0f;
acetlsm[c]=0.0f;
acsnmelt[c]=0.0f;
acsnsub[c]=0.0f;
acsnfro[c]=0.0f;
acsubc[c]=0.0f;
acfroc[c]=0.0f;
acevac[c]=0.0f;
acdewc[c]=0.0f;
acfrzc[c]=0.0f;
acmeltc[c]=0.0f;
acsnbot[c]=0.0f;
acponding[c]=0.0f;
acrainlsm[c]=0.0f;
acsnowlsm[c]=0.0f;
acrainsnow[c]=0.0f;
acrunsb[c]=0.0f;
acrunsf[c]=0.0f;
acqlat[c]=0.0f;
acqrf[c]=0.0f;
acrech[c]=0.0f;
acqspring[c]=0.0f;
acecan[c]=0.0f;
acetran[c]=0.0f;
acedir[c]=0.0f;
aceflxb[c]=0.0f;
accanhs[c]=0.0f;
}
float kjconvert=DV(dt,1000.0f);
acsagb[c]=AD(acsagb[c],MU(MU(kjconvert,SU(1.0f,fveg[c])),sag[c]));
acsagv[c]=AD(acsagv[c],MU(MU(kjconvert,fveg[c]),sag[c]));
acirg[c]=AD(acirg[c],MU(MU(kjconvert,fveg[c]),irg[c]));
acshg[c]=AD(acshg[c],MU(MU(kjconvert,fveg[c]),shg[c]));
acevg[c]=AD(acevg[c],MU(MU(kjconvert,fveg[c]),evg[c]));
acghv[c]=AD(acghv[c],MU(MU(kjconvert,fveg[c]),ghv[c]));
acpahg[c]=AD(acpahg[c],MU(MU(kjconvert,fveg[c]),pahg[c]));
acirb[c]=AD(acirb[c],MU(MU(kjconvert,SU(1.0f,fveg[c])),irb[c]));
acshb[c]=AD(acshb[c],MU(MU(kjconvert,SU(1.0f,fveg[c])),shb[c]));
acevb[c]=AD(acevb[c],MU(MU(kjconvert,SU(1.0f,fveg[c])),evb[c]));
acghb[c]=AD(acghb[c],MU(MU(kjconvert,SU(1.0f,fveg[c])),ghb[c]));
acpahb[c]=AD(acpahb[c],MU(MU(kjconvert,SU(1.0f,fveg[c])),pahb[c]));
acsav[c]=AD(acsav[c],MU(kjconvert,sav[c]));
acirc[c]=AD(acirc[c],MU(kjconvert,ircmp[c]));
acshc[c]=AD(acshc[c],MU(kjconvert,shc[c]));
acevc[c]=AD(acevc[c],MU(kjconvert,evc[c]));
actr[c]=AD(actr[c],MU(kjconvert,tr[c]));
aceflxb[c]=AD(aceflxb[c],DV(eflxb[c],1000.0f));
accanhs[c]=AD(accanhs[c],MU(kjconvert,canhs[c]));
acpahv[c]=AD(acpahv[c],MU(kjconvert,pahv[c]));
acswdnlsm[c]=AD(acswdnlsm[c],MU(kjconvert,swdown[c]));
acswuplsm[c]=AD(acswuplsm[c],MU(MU(kjconvert,swdown[c]),albedo[c]));
aclwdnlsm[c]=AD(aclwdnlsm[c],MU(MU(kjconvert,glw[c]),emiss[c]));
aclwuplsm[c]=AD(aclwuplsm[c],MU(MU(MU(MU(MU(MU(kjconvert,5.67e-08f),emiss[c]),trad[c]),trad[c]),trad[c]),trad[c]));
acshflsm[c]=AD(AD(acshg[c],acshc[c]),acshb[c]);
aclhflsm[c]=AD(AD(AD(acevg[c],acevc[c]),actr[c]),acevb[c]);
acghflsm[c]=AD(acghv[c],acghb[c]);
acpahlsm[c]=AD(AD(acpahv[c],acpahg[c]),acpahb[c]);
acints[c]=AD(acints[c],MU(dt,qints[c]));
acintr[c]=AD(acintr[c],MU(dt,qintr[c]));
acdrips[c]=AD(acdrips[c],MU(dt,qdrips[c]));
acdripr[c]=AD(acdripr[c],MU(dt,qdripr[c]));
acthros[c]=AD(acthros[c],MU(dt,qthros[c]));
acthror[c]=AD(acthror[c],MU(dt,qthror[c]));
acsnsub[c]=AD(acsnsub[c],MU(dt,qsnsub[c]));
acsnfro[c]=AD(acsnfro[c],MU(dt,qsnfro[c]));
acsubc[c]=AD(acsubc[c],MU(dt,qsubc[c]));
acfroc[c]=AD(acfroc[c],MU(dt,qfroc[c]));
acevac[c]=AD(acevac[c],MU(dt,qevac[c]));
acdewc[c]=AD(acdewc[c],MU(dt,qdewc[c]));
acfrzc[c]=AD(acfrzc[c],MU(dt,qfrzc[c]));
acmeltc[c]=AD(acmeltc[c],MU(dt,qmeltc[c]));
acsnbot[c]=AD(acsnbot[c],MU(dt,qsnbot[c]));
acsnmelt[c]=AD(acsnmelt[c],MU(dt,qmelt[c]));
acrunsf[c]=AD(acrunsf[c],runsf[c]);
acrunsb[c]=AD(acrunsb[c],runsb[c]);
acecan[c]=AD(acecan[c],MU(dt,ecan[c]));
acetran[c]=AD(acetran[c],MU(dt,etran[c]));
acedir[c]=AD(acedir[c],MU(dt,edir[c]));
acetlsm[c]=AD(acetlsm[c],MU(dt,AD(AD(etran[c],edir[c]),ecan[c])));
acponding[c]=AD(acponding[c],ponding[c]);
acrainlsm[c]=AD(acrainlsm[c],MU(dt,rainlsm[c]));
acsnowlsm[c]=AD(acsnowlsm[c],MU(dt,snowlsm[c]));
if(opt_run==5){
if(groundwater_due){
acqlat[c]=AD(acqlat[c],MU(qlat[c],1000.0f));
acqrf[c]=AD(acqrf[c],MU(qrf[c],1000.0f));
acrech[c]=AD(acrech[c],deeprech[c]);
acqspring[c]=AD(acqspring[c],qspring[c]);
}
}
if(isnow[c]<0 && isnow[c]>-5){
acrainsnow[c]=AD(acrainsnow[c],MU(dt,AD(qdripr[c],qthror[c])));
}
}
