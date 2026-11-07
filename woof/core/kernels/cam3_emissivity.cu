#ifdef CAM_LONGWAVE_MODE
#define CAM_LONGWAVE (CAM_LONGWAVE_MODE==1)
#else
#define CAM_LONGWAVE (*dolw)
#endif
#ifdef CAM_SHORTWAVE_MODE
#define CAM_SHORTWAVE (CAM_SHORTWAVE_MODE==1)
#else
#define CAM_SHORTWAVE (*dosw)
#endif
#ifndef CAM_LEVELS
#define CAM_LEVELS 32
#endif
// Generated from pinned WRF 4.6.1 numeric statements by the lane tools.
// WOOF's own math routines provide exp, log and pow. No other math source is included.
typedef int integer;
typedef int logical;
typedef int ftnlen;
typedef float real;
typedef double doublereal;
#define TRUE_ 1
#define FALSE_ 0
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define abs(a) ((a)>=0?(a):-(a))
// Fast forecasts use CUDA binary64 libm. Strict replay retains the pinned math.
#if defined(GPUWM_WRF_EXACT) && GPUWM_WRF_EXACT
#define cam_exp64 glibc_exp
#define cam_log64 glibc_log
#define cam_pow64 glibc_pow
#else
#define cam_exp64 exp
#define cam_log64 log
#define cam_pow64 pow
#endif
__device__ double d_int(double *x){return trunc(*x);}
__device__ int cam_floor__(double *x){return int(floor(*x));}
__device__ double pow_dd(double *x,double *y){return cam_pow64(*x,*y);}
__device__ double pow_di(double *a,int *b){int n=*b;double x=*a,result=1;unsigned int m=n<0?-n:n;if(n<0)x=1/x;while(m){if(m&1)result*=x;m>>=1;if(m)x*=x;}return result;}
// WOOF base-ten reduction: log10(x)=exponent*log10(2)+log(m)/log(10).
// Keep the mantissa on the same side of unity as x to limit cancellation.
// The low term is the residual after rounding log10(2) to the chosen high word.
__device__ double d_lg10(double *x){
#if !defined(GPUWM_WRF_EXACT) || !GPUWM_WRF_EXACT
 return log10(*x);
#else
 int e;double m=frexp(*x,&e);if(e>0){m*=2;--e;}
 const double hi=0x1.3441350900000p-2,lo=0x1.ef3fde623e256p-35;
 return double(e)*hi+(double(e)*lo+0.43429448190325182765*cam_log64(m));
#endif
}
__device__ int cam_overlap_failure__(){asm("trap;");return 0;}
__device__ int endrun_(){asm("trap;");return 0;}
__device__ int cam_optics_warning__(){printf("RADCSWMX: aerosol optics outside physical range\n");return 0;}
struct CamState {
    doublereal *cg_abari__, *cg_abarl__, *cg_abco2__, *cg_abh2o__,
        *cg_abo2__, *cg_abo3__, *cg_abs_cff_mss_aer__,
        *cg_ah2onw__    , *cg_ah2ow__   , cg_amco2__, cg_amd__,
        cg_amo__, *cg_bbari__, *cg_bbarl__, cg_bgscl_rf__, cg_c16__,
        cg_c17__, cg_c26__, cg_c27__, cg_c28__, cg_c29__, cg_c30__,
        cg_c31__, cg_cappa__, cg_carscl__, cg_carscl_rf__, *cg_cbari__,
        *cg_cbarl__, cg_ch4vmr__, *cg_cn_ah2ow__    , *cg_cn_eh2ow__    , *cg_co2__, cg_co2mmr__, cg_co2vmr__, *cg_coefh__
        , *cg_coefj__   , *cg_coefk__   , cg_cpair__, cg_cplol__, cg_cplos__,
        cg_dayspy__, *cg_dbari__, *cg_dbarl__, cg_delta__,
        cg_dlp_h2o__, cg_dlu_h2o__, cg_dpfco2__, cg_dpfo3__, cg_drh_h2o__,
         cg_dte_h2o__, cg_dtp_h2o__, cg_dustscl__, cg_dustscl_rf__,
        *cg_ebari__, *cg_ebarl__, *cg_eh2onw__  , cg_epsilo__, cg_epsqs__, *cg_estbl__,
        *cg_estblh2o__, cg_f11vmr__, cg_f12vmr__, *cg_fat__ , *cg_fbari__, *cg_fbarl__, cg_fc1__, *cg_fet__ , *cg_frcsol__, cg_fwc1__, cg_fwc2__,
        cg_fwcoef__, *cg_gbg__, *cg_gcb__, *cg_gcphil__ , *cg_gcphob__, *cg_gdst__  , cg_gravit__, cg_gravmks__, *cg_gsslt__    , *cg_gsul__    , *cg_gvolc__,
        cg_inf__, *cg_kbg__, *cg_kcb__, *cg_kcphil__    , *cg_kcphob__, *cg_kdst__  , *cg_ksslt__   , *cg_ksul__
            , *cg_kvolc__, cg_latice__,
        cg_latvap__, *cg_ln_ah2ow__ ,
         *cg_ln_eh2ow__ ,
        cg_max_lp_h2o__, cg_max_lu_h2o__, cg_max_rh_h2o__,
        cg_max_te_h2o__, cg_max_tp_h2o__, cg_min_lp_h2o__,
        cg_min_lu_h2o__, cg_min_p_h2o__, cg_min_rh_h2o__, cg_min_te_h2o__,
         cg_min_tp_h2o__, cg_min_u_h2o__, cg_mwch4__, cg_mwco2__,
        cg_mwdry__, cg_mwf11__, cg_mwf12__, cg_mwh2o__, cg_mwn2o__,
        cg_n2ovmr__, *cg_nirwgt__, cg_o2mmr__, cg_p0__, *cg_pcf__,
        *cg_pco2__, *cg_ph2o__, cg_pie__, *cg_po2__,
        cg_r_universal__, cg_rair__, *cg_raytau__, *cg_retab__,
        cg_rga__, cg_rgsslp__, cg_scon__, cg_sslp__, cg_ssltscl__,
        cg_ssltscl_rf__, cg_stebol__, cg_sulscl__, cg_sulscl_rf__,
        cg_tauback__, cg_tmax__, cg_tmelt__, cg_tmin__, cg_v_abo3_35__,
        cg_v_abo3_64__, cg_v_raytau_35__, cg_v_raytau_64__, cg_volcscl__,
        cg_volcscl_rf__, *cg_wavmax__, *cg_wavmin__, *cg_wbg__,
        *cg_wcb__, *cg_wcphil__ ,
        *cg_wcphob__, *cg_wdst__    , *cg_wsslt__   , *cg_wsul__    , *cg_wvolc__, cg_zvir__;
    logical cg_have_ozone__;
    integer cg_idxbcphi__, cg_idxbcpho__, cg_idxbg__, cg_idxcarbonfirst__,
        cg_idxdustfirst__, cg_idxocphi__, cg_idxocpho__, cg_idxsslt__,
        cg_idxsul__, cg_idxvolc__;
    logical cg_indirect__;
    integer cg_ixcldice__, cg_ixcldliq__, cg_levsiz_ozone_save__;
    logical cg_masterproc__;
    integer cg_mxaerl__, cg_ntoplw__;
    logical cg_ozncyc__, cg_radforce__, cg_strat_volcanic__, cg_trace_gas__;
    integer *cg_yrdata__;
double driver_gases[5]; unsigned char* scratch; unsigned long long scratch_offset; double* sw_flux; int sw_flux_offset,sw_flux_stride;
void** abs_args; double* abs_cache; int abs_phase, abs_lane, abs_stride, abs_padding;
};
#define CAM_FULL_WORK_BYTES (0ULL+((24488+((64+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+(((CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7)+((((CAM_LEVELS+1)-(1)+1))*4+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1))*4+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+((((CAM_LEVELS+1)-(1)+1))*8+7)+((((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1)*((CAM_LEVELS+1)-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7)+((((CAM_LEVELS+1)+1-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7)+((((CAM_LEVELS+1)-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1))*4+7)+((((CAM_LEVELS+1)-(1)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1))*4+7)+((((CAM_LEVELS+1)-(0)+1))*4+7)+(((CAM_LEVELS-(0)+1))*8+7)+(((CAM_LEVELS-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((4-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+(((19-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+((((CAM_LEVELS+1)-(0)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7)+(((CAM_LEVELS-(1)+1))*4+7)+(((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(11-(1)+1))*8+7)+7)&~7))))
#define CAM_SW_FLUX_VALUES (((CAM_LEVELS+1)-(0)+1)*(15-(1)+1)+((CAM_LEVELS+1)-(0)+1)*(15-(1)+1)+((CAM_LEVELS+1)-(0)+1)*(15-(1)+1)+((CAM_LEVELS+1)-(0)+1)*(15-(1)+1)+((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))
#define CAM_WORK_BYTES (0ULL+((CAM_FULL_WORK_BYTES-CAM_SW_FLUX_VALUES*8)))

static_assert(sizeof(CamState)<=2048,"CAM private state slot");
struct CamScratchFrame{CamState* state;unsigned long long saved;__device__ CamScratchFrame(CamState* s):state(s),saved(s->scratch_offset){} __device__ ~CamScratchFrame(){state->scratch_offset=saved;}};
template<typename T> __device__ T* cam_alloc(CamState* s,unsigned long long n){unsigned long long p=(s->scratch_offset+7)&~7ULL;s->scratch_offset=p+n*sizeof(T);T* out=(T*)(s->scratch+p);for(unsigned long long i=0;i<n;++i)out[i]=T{};return out;}

__device__ double r_mod(float* a,float* b){return fmodf(*a,*b);}
__device__ integer i_dnnt(double* x){return int(round(*x));}
__device__ int cam_driver_failure__(){printf("CAM driver: invalid ozone pressure order, interpolation factors or aerosol state\n");asm("trap;");return 0;}
__device__ int read_camgases__(CamState* state,int* year,float* julian,int* read,char* label,double* co2,double* n2o,double* ch4,double* f11,double* f12,ftnlen length){*co2=state->driver_gases[0];*n2o=state->driver_gases[1];*ch4=state->driver_gases[2];*f11=state->driver_gases[3];*f12=state->driver_gases[4];return 0;}

template<typename T> __device__ T* cam_flux_alloc(CamState* s,int n){
        T* p=(T*)(s->sw_flux+s->sw_flux_offset*s->sw_flux_stride);
        s->sw_flux_offset+=n;
        for(int at=0;at<n;++at)p[at*s->sw_flux_stride*(8/sizeof(T))]=0;
        return p;
    }
__device__ int radems_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *s2c, doublereal *tcg,
    doublereal *w, doublereal *tplnke, doublereal *plh2o, doublereal *pnm,
     doublereal *plco2, doublereal *tint, doublereal *tint4, doublereal *
    tlayr, doublereal *tlayr4, doublereal *plol, doublereal *plos,
    doublereal *ucfc11, doublereal *ucfc12, doublereal *un2o0, doublereal
    *un2o1, doublereal *uch4, doublereal *uco211, doublereal *uco212,
    doublereal *uco213, doublereal *uco221, doublereal *uco222,
    doublereal *uco223, doublereal *uptype, doublereal *bn2o0, doublereal
    *bn2o1, doublereal *bch4, doublereal *co2em, doublereal *co2eml,
    doublereal *co2t, doublereal *h2otr, doublereal *abplnk1, doublereal *
    abplnk2, doublereal *emstot, doublereal *plh2ob, doublereal *wb,
    doublereal *aer_trn_ttl__);
__device__ int trcplk_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *tint, doublereal *tlayr,
    doublereal *tplnke, doublereal *emplnk, doublereal *abplnk1,
    doublereal *abplnk2);
__device__ int trcems_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pverp, integer *k, doublereal *co2t, doublereal *pnm,
    doublereal *ucfc11, doublereal *ucfc12, doublereal *un2o0, doublereal
    *un2o1, doublereal *bn2o0, doublereal *bn2o1, doublereal *uch4,
    doublereal *bch4, doublereal *uco211, doublereal *uco212, doublereal *
    uco213, doublereal *uco221, doublereal *uco222, doublereal *uco223,
    doublereal *uptype, doublereal *w, doublereal *s2c, doublereal *up2,
    doublereal *emplnk, doublereal *th2o, doublereal *tco2, doublereal *
    to3, doublereal *emstrc, doublereal *aer_trn_ttl__);
__device__ doublereal psi_(CamState* state, doublereal *tpx, integer *iband);
__device__ doublereal phi_(CamState* state, doublereal *tpx, integer *iband);
__device__ doublereal fh2oself_(CamState* state, doublereal *temp);
__device__ int radabs_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *pbr, doublereal *pnm,
    doublereal *co2em, doublereal *co2eml, doublereal *tplnka, doublereal
    *s2c, doublereal *tcg, doublereal *w, doublereal *h2otr, doublereal *
    plco2, doublereal *plh2o, doublereal *co2t, doublereal *tint,
    doublereal *tlayr, doublereal *plol, doublereal *plos, doublereal *
    pmln, doublereal *piln, doublereal *ucfc11, doublereal *ucfc12,
    doublereal *un2o0, doublereal *un2o1, doublereal *uch4, doublereal *
    uco211, doublereal *uco212, doublereal *uco213, doublereal *uco221,
    doublereal *uco222, doublereal *uco223, doublereal *uptype,
    doublereal *bn2o0, doublereal *bn2o1, doublereal *bch4, doublereal *
    abplnk1, doublereal *abplnk2, doublereal *abstot, doublereal *absnxt,
    doublereal *plh2ob, doublereal *wb, doublereal *aer_mpp__, doublereal
    *aer_trn_ttl__);
__device__ int trcab_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pverp, integer *k1, integer *k2, doublereal *ucfc11,
    doublereal *ucfc12, doublereal *un2o0, doublereal *un2o1, doublereal *
    uch4, doublereal *uco211, doublereal *uco212, doublereal *uco213,
    doublereal *uco221, doublereal *uco222, doublereal *uco223,
    doublereal *bn2o0, doublereal *bn2o1, doublereal *bch4, doublereal *
    to3co2, doublereal *pnm, doublereal *dw, doublereal *pnew, doublereal
    *s2c, doublereal *uptype, doublereal *dplh2o, doublereal *abplnk1,
    doublereal *tco2, doublereal *th2o, doublereal *to3, doublereal *
    abstrc, doublereal *aer_trn_ttl__);
__device__ int trcabn_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pverp, integer *k2, integer *kn, doublereal *ucfc11,
    doublereal *ucfc12, doublereal *un2o0, doublereal *un2o1, doublereal *
    uch4, doublereal *uco211, doublereal *uco212, doublereal *uco213,
    doublereal *uco221, doublereal *uco222, doublereal *uco223,
    doublereal *tbar, doublereal *bplnk, doublereal *winpl, doublereal *
    pinpl, doublereal *tco2, doublereal *th2o, doublereal *to3,
    doublereal *uptype, doublereal *dw, doublereal *s2c, doublereal *up2,
    doublereal *pnew, doublereal *abstrc, doublereal *uinpl, doublereal *
    aer_trn_ngh__);
__device__ int radclwmx_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *lwupcgs, doublereal *tnm,
    doublereal *qnm, doublereal *o3vmr, doublereal *pmid, doublereal *
    pint, doublereal *pmln, doublereal *piln, doublereal *n2o, doublereal
    *ch4, doublereal *cfc11, doublereal *cfc12, doublereal *cld,
    doublereal *emis, doublereal *pmxrgn, integer *nmxrgn, doublereal *
    qrl, doublereal *qrlcs, logical *doabsems, doublereal *abstot,
    doublereal *absnxt, doublereal *emstot, doublereal *flns, doublereal *
    flnt, doublereal *flnsc, doublereal *flntc, doublereal *flwds,
    doublereal *flut, doublereal *flutc, doublereal *flup, doublereal *
    flupc, doublereal *fldn, doublereal *fldnc, doublereal *aer_mass__);
__device__ int radtpl_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *tnm, doublereal *lwupcgs,
    doublereal *qnm, doublereal *pnm, doublereal *plco2, doublereal *
    plh2o, doublereal *tplnka, doublereal *s2c, doublereal *tcg,
    doublereal *w, doublereal *tplnke, doublereal *tint, doublereal *
    tint4, doublereal *tlayr, doublereal *tlayr4, doublereal *pmln,
    doublereal *piln, doublereal *plh2ob, doublereal *wb);
__device__ int radoz2_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *o3vmr, doublereal *pint,
    doublereal *plol, doublereal *plos, integer *ntoplw);
__device__ int aer_pth_(CamState* state, doublereal *aer_mass__, doublereal *aer_mpp__,
    integer *ncol, integer *pcols, integer *plev, integer *plevp);
__device__ int aer_trn_(CamState* state, doublereal *aer_mpp__, doublereal *
    aer_trn_ttl__, integer *pcols, integer *plev, integer *plevp);
__device__ int sortarray_(CamState* state, integer *n, doublereal *ain, integer *indxa);
__device__ int trcpth_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *tnm, doublereal *pnm,
    doublereal *cfc11, doublereal *cfc12, doublereal *n2o, doublereal *
    ch4, doublereal *qnm, doublereal *ucfc11, doublereal *ucfc12,
    doublereal *un2o0, doublereal *un2o1, doublereal *uch4, doublereal *
    uco211, doublereal *uco212, doublereal *uco213, doublereal *uco221,
    doublereal *uco222, doublereal *uco223, doublereal *bn2o0, doublereal
    *bn2o1, doublereal *bch4, doublereal *uptype);
__device__ int radcswmx_(CamState* state, integer *jj, integer *lchnk, integer *ncol,
    integer *pcols, integer *pver, integer *pverp, doublereal *pint,
    doublereal *pmid, doublereal *h2ommr, doublereal *rh, doublereal *
    o3mmr, doublereal *aermmr, doublereal *cld, doublereal *cicewp,
    doublereal *cliqwp, doublereal *rel, doublereal *rei, doublereal *
    tauxcl, doublereal *tauxci, doublereal *eccf, doublereal *coszrs,
    doublereal *scon, doublereal *solin, real *solcon, doublereal *asdir,
    doublereal *asdif, doublereal *aldir, doublereal *aldif, integer *
    nmxrgn, doublereal *pmxrgn, doublereal *qrs, doublereal *qrscs,
    doublereal *fsnt, doublereal *fsntc, doublereal *fsntoa, doublereal *
    fsntoac, doublereal *fsnirtoa, doublereal *fsnrtoac, doublereal *
    fsnrtoaq, doublereal *fsns, doublereal *fsnsc, doublereal *fsdsc,
    doublereal *fsds, doublereal *sols, doublereal *soll, doublereal *
    solsd, doublereal *solld, doublereal *frc_day__, doublereal *fsup,
    doublereal *fsupc, doublereal *fsdn, doublereal *fsdnc, doublereal *
    fsdndir, doublereal *fsdncdir, doublereal *fsdndif, doublereal *
    fsdncdif, doublereal *fsdsdir, doublereal *fsdsdif, doublereal *
    fsdscdir, doublereal *fsdscdif, doublereal *aertau, doublereal *
    aerssa, doublereal *aerasm, doublereal *aerfwd);
__device__ int raddedmx_(CamState* state, integer *pver, integer *pverp, integer *pcols,
    doublereal *coszrs, integer *ndayc, integer *idayc, doublereal *abh2o,
     doublereal *abo3, doublereal *abco2, doublereal *abo2, doublereal *
    uh2o, doublereal *uo3, doublereal *uco2, doublereal *uo2, doublereal *
    trayoslp, doublereal *pflx, integer *ns, doublereal *tauxcl,
    doublereal *wcl, doublereal *gcl, doublereal *fcl, doublereal *tauxci,
     doublereal *wci, doublereal *gci, doublereal *fci, doublereal *
    tauxar, doublereal *wa, doublereal *ga, doublereal *fa, doublereal *
    rdir, doublereal *rdif, doublereal *tdir, doublereal *tdif,
    doublereal *explay, doublereal *rdirc, doublereal *rdifc, doublereal *
    tdirc, doublereal *tdifc, doublereal *explayc);
__device__ integer findvalue_(CamState* state, integer *ix, integer *n, doublereal *ain, integer *indxa);
__device__ int param_cldoptics_calc_(CamState* state, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, integer *pverr, integer *pverrp,
    integer *ppcnst, doublereal *q, doublereal *cldn, doublereal *
    landfrac, doublereal *landm, doublereal *icefrac, doublereal *pdel,
    doublereal *t, doublereal *ps, doublereal *pmid, doublereal *pint,
    doublereal *cicewp, doublereal *cliqwp, doublereal *emis, doublereal *
    rel, doublereal *rei, doublereal *pmxrgn, integer *nmxrgn, doublereal
    *snowh);
__device__ int radctl_(CamState* state, integer *j, integer *lchnk, integer *ncol,
    integer *pcols, integer *pver, integer *pverp, integer *pverr,
    integer *pverrp, integer *ppcnst, integer *pcnst, doublereal *lwups,
    doublereal *emis, doublereal *pmid, doublereal *pint, doublereal *
    pmln, doublereal *piln, doublereal *pdel, doublereal *t, doublereal *
    qm1, doublereal *cld, doublereal *cicewp, doublereal *cliqwp,
    doublereal *tauxcl, doublereal *tauxci, doublereal *coszrs,
    doublereal *clat, doublereal *asdir, doublereal *asdif, doublereal *
    aldir, doublereal *aldif, real *solcon, real *gmt, integer *julday,
    real *julian, real *dt, real *xtime, doublereal *pin, doublereal *
    ozmixmj, doublereal *ozmix, integer *levsiz, integer *num_months__,
    integer *ghg_input__, doublereal *m_psp__, doublereal *m_psn__,
    doublereal *aerosoljp, doublereal *aerosoljn, doublereal *m_hybi__,
    integer *paerlev, integer *naer_c__, doublereal *pmxrgn, integer *
    nmxrgn, logical *dolw, logical *dosw, logical *doabsems, doublereal *
    abstot, doublereal *absnxt, doublereal *emstot, doublereal *fsup,
    doublereal *fsupc, doublereal *fsdn, doublereal *fsdnc, doublereal *
    fsdndir, doublereal *fsdncdir, doublereal *fsdndif, doublereal *
    fsdncdif, doublereal *flup, doublereal *flupc, doublereal *fldn,
    doublereal *fldnc, doublereal *swcf, doublereal *lwcf, doublereal *
    flut, doublereal *fsns, doublereal *fsnt, doublereal *flns,
    doublereal *flnt, doublereal *qrs, doublereal *qrscs, doublereal *qrl,
     doublereal *qrlcs, doublereal *flwds, doublereal *rel, doublereal *
    rei, doublereal *sols, doublereal *soll, doublereal *solsd,
    doublereal *solld, doublereal *n2ovmr, doublereal *ch4vmr, doublereal
    *f11vmr, doublereal *f12vmr, doublereal *landfrac, doublereal *zm,
    doublereal *fsds, doublereal *fsdsdir, doublereal *fsdsdif);
__device__ int camrad_(CamState* state, real *rthratenlw, real *rthratensw, real *
    rthratenlwc, real *rthratenswc, logical *dolw, logical *dosw, real *
    swupt, real *swuptc, real *swdnt, real *swdntc, real *lwupt, real *
    lwuptc, real *lwdnt, real *lwdntc, real *swupb, real *swupbc, real *
    swdnb, real *swdnbc, real *lwupb, real *lwupbc, real *lwdnb, real *
    lwdnbc, real *swcf, real *lwcf, real *olr, real *cemiss, real *
    taucldc, real *taucldi, real *coszr, real *gsw, real *glw, real *xlat,
     real *xlong, real *albedo, real *t_phy__, real *tsk, real *emiss,
    real *qv3d, real *qc3d, real *qr3d, real *qi3d, real *qs3d, real *
    qg3d, real *alswvisdir, real *alswvisdif, real *alswnirdir, real *
    alswnirdif, real *swvisdir, real *swvisdif, real *swnirdir, real *
    swnirdif, integer *sf_surface_physics__, real *swddir, real *swddif,
    real *swddni, logical *f_qv__, logical *f_qc__, logical *f_qr__,
    logical *f_qi__, logical *f_qs__, logical *f_qg__, real *f_ice_phy__,
    real *f_rain_phy__, real *p_phy__, real *p8w, real *z__, real *
    pi_phy__, real *rho_phy__, real *dz8w, real *cldfra, real *xland,
    real *xice, real *snow, real *ozmixm, real *pin0, integer *levsiz,
    integer *num_months__, real *m_psp__, real *m_psn__, real *aerosolcp,
    real *aerosolcn, real *m_hybi0__, integer *cam_abs_dim1__, integer *
    cam_abs_dim2__, integer *paerlev, integer *naer_c__, real *gmt,
    integer *julday, real *julian, integer *yr, real *dt, real *xtime,
    real *declin, real *solcon, real *radt, real *degrad, integer *
    n_cldadv__, real *abstot_3d__, real *absnxt_3d__, real *emstot_3d__,
    logical *doabsems, integer *ghg_input__, integer *ids, integer *ide,
    integer *jds, integer *jde, integer *kds, integer *kde, integer *ims,
    integer *ime, integer *jms, integer *jme, integer *kms, integer *kme,
    integer *its, integer *ite, integer *jts, integer *jte, integer *kts,
    integer *kte, real *coszen);
__device__ int radinp_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *pmid, doublereal *pint,
    doublereal *o3vmr, doublereal *pmidrd, doublereal *pintrd, doublereal
    *eccf, doublereal *o3mmr);
__device__ int radozn_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, doublereal *pmid, doublereal *pin, integer *levsiz,
    doublereal *ozmix, doublereal *o3vmr);
__device__ int oznint_(CamState* state, integer *julday, real *julian, real *dt, real *
    gmt, real *xtime, doublereal *ozmixmj, doublereal *ozmix, integer *
    levsiz, integer *num_months__, integer *pcols);
__device__ int get_aerosol_(CamState* state, integer *c__, integer *julday, real *
    julian, real *dt, real *gmt, real *xtime, doublereal *m_psp__,
    doublereal *m_psn__, doublereal *aerosoljp, doublereal *aerosoljn,
    doublereal *m_hybi__, integer *paerlev, integer *naer_c__, doublereal
    *pint, integer *pcols, integer *pver, integer *pverp, integer *pverr,
    integer *pverrp, doublereal *aerosolt, doublereal *scale);
__device__ int aerosol_indirect_(CamState* state, integer *ncol, integer *lchnk,
    integer *pcols, integer *pver, integer *ppcnst, doublereal *landfrac,
    doublereal *pmid, doublereal *t, doublereal *qm1, doublereal *cld,
    doublereal *zm, doublereal *rel);
__device__ int trcmix_clwrf_(CamState* state, integer *lchnk, integer *ncol, integer *
    pcols, integer *pver, doublereal *pmid, doublereal *clat, doublereal *
    n2ovmr, doublereal *ch4vmr, doublereal *f11vmr, doublereal *f12vmr,
    doublereal *n2o, doublereal *ch4, doublereal *cfc11, doublereal *
    cfc12);
__device__ int aqsat_(CamState* state, doublereal *t, doublereal *p, doublereal *es,
    doublereal *qs, integer *ii, integer *ilen, integer *kk, integer *
    kstart, integer *kend);
__device__ int cldefr_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *landfrac, doublereal *t,
    doublereal *rel, doublereal *rei, doublereal *ps, doublereal *pmid,
    doublereal *landm, doublereal *icefrac, doublereal *snowh);
__device__ int cldems_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *clwp, doublereal *fice,
    doublereal *rei, doublereal *emis);
__device__ int cldovrlap_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
     integer *pver, integer *pverp, doublereal *pint, doublereal *cld,
    integer *nmxrgn, doublereal *pmxrgn);
__device__ int getfactors_(CamState* state, logical *cycflag, integer *np1, doublereal *
    cdayminus, doublereal *cdayplus, doublereal *cday, doublereal *fact1,
    doublereal *fact2);
__device__ int get_rf_scales_(CamState* state, doublereal *scales);
__device__ int get_int_scales_(CamState* state, doublereal *scales);
__device__ int vert_interpolate_(CamState* state, doublereal *match_ps__, doublereal *
    aerosolc, doublereal *m_hybi__, integer *paerlev, integer *naer_c__,
    doublereal *pint, integer *n, doublereal *aerosol_mmr__, integer *
    pcols, integer *pver, integer *pverp, integer *ncol, integer *c__);
__device__ int background_(CamState* state, integer *lchnk, integer *ncol, doublereal *
    pint, integer *pcols, integer *pverr, integer *pverrp, doublereal *
    mmr);
__device__ int scale_aerosols_(CamState* state, doublereal *aerosolt, integer *pcols,
    integer *pver, integer *ncol, integer *lchnk, doublereal *scale);
__device__ doublereal estblf_(CamState* state, doublereal *td);
__device__ logical validfactors_(CamState* state, doublereal *fact1, doublereal *fact2);
__device__ int reltab_(CamState* state, integer *ncol, integer *pcols, integer *pver,
    doublereal *t, doublereal *landfrac, doublereal *landm, doublereal *
    icefrac, doublereal *rel, doublereal *snowh);
__device__ int reitab_(CamState* state, integer *ncol, integer *pcols, integer *pver,
    doublereal *t, doublereal *re);
__device__ int trcmix_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, doublereal *pmid, doublereal *clat, doublereal *n2o,
    doublereal *ch4, doublereal *cfc11, doublereal *cfc12);




__device__ integer radems_c__1 = 1;
__device__ integer radems_c__2 = 2;
__device__ doublereal radems_c_b22 = .699999988079071045;


__device__ int radems_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *s2c, doublereal *tcg,
    doublereal *w, doublereal *tplnke, doublereal *plh2o, doublereal *pnm,
     doublereal *plco2, doublereal *tint, doublereal *tint4, doublereal *
    tlayr, doublereal *tlayr4, doublereal *plol, doublereal *plos,
    doublereal *ucfc11, doublereal *ucfc12, doublereal *un2o0, doublereal
    *un2o1, doublereal *uch4, doublereal *uco211, doublereal *uco212,
    doublereal *uco213, doublereal *uco221, doublereal *uco222,
    doublereal *uco223, doublereal *uptype, doublereal *bn2o0, doublereal
    *bn2o1, doublereal *bch4, doublereal *co2em, doublereal *co2eml,
    doublereal *co2t, doublereal *h2otr, doublereal *abplnk1, doublereal *
    abplnk2, doublereal *emstot, doublereal *plh2ob, doublereal *wb,
    doublereal *aer_trn_ttl__)
{
 CamScratchFrame scratch_frame(state);

    integer s2c_dim1 = 0,s2c_offset = 0,tcg_dim1 = 0,tcg_offset = 0,w_dim1 = 0,w_offset = 0,plh2o_dim1 = 0,plh2o_offset = 0,pnm_dim1 = 0,pnm_offset = 0,plco2_dim1 = 0,plco2_offset = 0,tint_dim1 = 0,tint_offset = 0,tint4_dim1 = 0,tint4_offset = 0,tlayr_dim1 = 0,tlayr_offset = 0,tlayr4_dim1 = 0,tlayr4_offset = 0,plol_dim1 = 0,plol_offset = 0,plos_dim1 = 0,plos_offset = 0,plh2ob_dim2 = 0,plh2ob_offset = 0,wb_dim2 = 0,wb_offset = 0,aer_trn_ttl_dim1 = 0,aer_trn_ttl_dim2 = 0,aer_trn_ttl_dim3 = 0,aer_trn_ttl_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,bn2o0_dim1 = 0,bn2o0_offset = 0,bn2o1_dim1 = 0,bn2o1_offset = 0,bch4_dim1 = 0,bch4_offset = 0,uptype_dim1 = 0,uptype_offset = 0,emstot_dim1 = 0,emstot_offset = 0,co2em_dim1 = 0,co2em_offset = 0,co2eml_dim1 = 0,co2eml_offset = 0,co2t_dim1 = 0,co2t_offset = 0,h2otr_dim1 = 0,h2otr_offset = 0,abplnk1_dim2 = 0,abplnk1_offset = 0,abplnk2_dim2 = 0,abplnk2_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0;

    doublereal d__1 = 0,d__2 = 0,d__3 = 0;

    doublereal pnew_mks__ = 0,sslp_mks__ = 0;

    integer i__ = 0,k = 0;

    doublereal u[1] = {};

    integer k1 = 0;

    doublereal u1 = 0,u2 = 0,u7 = 0,u8 = 0,u9 = 0,fe = 0;

    integer ib = 0;

    doublereal k21[1] = {},k22[1] = {},ub[2] = {},pi = 0,et = 0,u13 = 0,ex = 0,te = 0;

    integer iu = 0,ip = 0;

    doublereal wp = 0,wu = 0,r300 = 0,uc1[1] = {},et2 = 0,t1i = 0,et4 = 0,r250 = 0,te1 = 0,te2 = 0,to3[1] = {},tr1[1] = {},tr2[1] = {},tr3[1] = {},tr4[1] = {},tr7[1] = {},tr8[1] = {},te3 = 0,te4 = 0,te5 = 0;

    integer iu1 = 0,ip1 = 0;

    doublereal wu1 = 0,wp1 = 0,t_e__ = 0;

    integer iuc = 0;

    doublereal t_p__ = 0;

    integer ite = 0,irh = 0;

    doublereal fwk = 0;

    integer itp = 0;

    doublereal dtx[1] = {},dty[1] = {},wuc = 0,wte = 0,sum[1] = {},wrh = 0,esx = 0;

    doublereal wtp = 0,qsx = 0;

    integer iuc1 = 0,ite1 = 0,irh1 = 0;

    doublereal tco2[1] = {},wco2 = 0,th2o[1] = {};

    integer itp1 = 0;

    doublereal wuc1 = 0,wte1 = 0,tmp1 = 0,tmp2 = 0,tmp3 = 0,wrh1 = 0,wtp1 = 0,w00_00__ = 0,w00_01__ = 0,w00_10__ = 0,w00_11__ = 0,f1co2 = 0,f2co2 = 0,f3co2 = 0,beta = 0,w01_00__ = 0,w01_01__ = 0,w01_10__ = 0,w01_11__ = 0,w10_00__ = 0,t1co2 = 0,emis[2] = {},phat = 0,omet = 0,pnew = 0,dvar = 0,uscl = 0,uvar = 0,sqti = 0,w10_01__ = 0,w10_10__ = 0,w10_11__ = 0,w_0_00__ = 0,w_0_01__ = 0,w_0_10__ = 0,w_0_11__ = 0,w_1_00__ = 0,sqwp = 0,w_1_01__ = 0,w_1_10__ = 0,w_1_11__ = 0,w11_00__ = 0,w11_01__ = 0,w11_10__ = 0,w11_11__ = 0,fch2o = 0,fdif = 0;

    integer iest = 0;

    doublereal pch2o, uch2o, o3ems[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}    , term7[2] = {}
        , term8[2] = {}  , w_0_0___, w_0_1___,
        w_1_0___, w_1_1___, log_p__, oneme, log_u__, pnewb[2] = {}, tpath,
        dbvtt[1] = {}, omeps, rsqti, rsslp, posqt, o3bndi, rbeta7, co2ems[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}
        , rbeta8, co2plk[1] = {}, rbeta9, h2oems[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}   , troco2[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}  , exm1sq, tlayr5,
        f1sqwp, absbnd, rbeta13, tcrfac, log_uc__, alphat, c_star__,
        e_star__, tlocal, q_path__, l_star__, tpathe, emplnk[14] = {}    , realnu, emstrc[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}  ;

    doublereal rh_path__ = 0;


    --tplnke;
    co2eml_dim1 = 1;
    co2eml_offset = 1 + co2eml_dim1;
    co2eml -= co2eml_offset;
    aer_trn_ttl_dim1 = 1;
    aer_trn_ttl_dim2 = (CAM_LEVELS+1);
    aer_trn_ttl_dim3 = (CAM_LEVELS+1);
    aer_trn_ttl_offset = 1 + aer_trn_ttl_dim1 * (1 + aer_trn_ttl_dim2 * (1 +
        aer_trn_ttl_dim3));
    aer_trn_ttl__ -= aer_trn_ttl_offset;
    wb_dim2 = 1;
    wb_offset = 1 + 2 * (1 + wb_dim2);
    wb -= wb_offset;
    plh2ob_dim2 = 1;
    plh2ob_offset = 1 + 2 * (1 + plh2ob_dim2);
    plh2ob -= plh2ob_offset;
    emstot_dim1 = 1;
    emstot_offset = 1 + emstot_dim1;
    emstot -= emstot_offset;
    abplnk2_dim2 = 1;
    abplnk2_offset = 1 + 14 * (1 + abplnk2_dim2);
    abplnk2 -= abplnk2_offset;
    abplnk1_dim2 = 1;
    abplnk1_offset = 1 + 14 * (1 + abplnk1_dim2);
    abplnk1 -= abplnk1_offset;
    h2otr_dim1 = 1;
    h2otr_offset = 1 + h2otr_dim1;
    h2otr -= h2otr_offset;
    co2t_dim1 = 1;
    co2t_offset = 1 + co2t_dim1;
    co2t -= co2t_offset;
    co2em_dim1 = 1;
    co2em_offset = 1 + co2em_dim1;
    co2em -= co2em_offset;
    bch4_dim1 = 1;
    bch4_offset = 1 + bch4_dim1;
    bch4 -= bch4_offset;
    bn2o1_dim1 = 1;
    bn2o1_offset = 1 + bn2o1_dim1;
    bn2o1 -= bn2o1_offset;
    bn2o0_dim1 = 1;
    bn2o0_offset = 1 + bn2o0_dim1;
    bn2o0 -= bn2o0_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;
    plos_dim1 = 1;
    plos_offset = 1 + plos_dim1;
    plos -= plos_offset;
    plol_dim1 = 1;
    plol_offset = 1 + plol_dim1;
    plol -= plol_offset;
    tlayr4_dim1 = 1;
    tlayr4_offset = 1 + tlayr4_dim1;
    tlayr4 -= tlayr4_offset;
    tlayr_dim1 = 1;
    tlayr_offset = 1 + tlayr_dim1;
    tlayr -= tlayr_offset;
    tint4_dim1 = 1;
    tint4_offset = 1 + tint4_dim1;
    tint4 -= tint4_offset;
    tint_dim1 = 1;
    tint_offset = 1 + tint_dim1;
    tint -= tint_offset;
    plco2_dim1 = 1;
    plco2_offset = 1 + plco2_dim1;
    plco2 -= plco2_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;
    plh2o_dim1 = 1;
    plh2o_offset = 1 + plh2o_dim1;
    plh2o -= plh2o_offset;
    w_dim1 = 1;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    tcg_dim1 = 1;
    tcg_offset = 1 + tcg_dim1;
    tcg -= tcg_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;


    r250 = .0040000000000000001f;
    r300 = .0033333333333333335f;
    rsslp = 1.f / state->cg_sslp__;
    fdif = 1.65999996662139893f;
    sslp_mks__ = state->cg_sslp__ / 10.f;
    omeps = 1.f - state->cg_epsilo__;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    ex = cam_exp64(960.f / tplnke[i__]);

    d__1 = tplnke[i__], d__1 *= d__1;
    co2plk[i__ - 1] = 5e8f / (d__1 * d__1 * (ex - 1.f));
    co2t[i__ + state->cg_ntoplw__ * co2t_dim1] = tplnke[i__];
    sum[i__ - 1] = co2t[i__ + state->cg_ntoplw__ * co2t_dim1] * pnm[
        i__ + state->cg_ntoplw__ * pnm_dim1];

    }

    k = state->cg_ntoplw__;
    i__1 = state->cg_ntoplw__ + 1;
    for (k1 = (CAM_LEVELS+1); k1 >= i__1; --k1) {
    ++k;
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        sum[i__ - 1] += tlayr[i__ + k * tlayr_dim1] * (pnm[i__ + k *
            pnm_dim1] - pnm[i__ + (k - 1) * pnm_dim1]);
        ex = cam_exp64(960.f / tlayr[i__ + k1 * tlayr_dim1]);
        tlayr5 = tlayr[i__ + k1 * tlayr_dim1] * tlayr4[i__ + k1 *
            tlayr4_dim1];

        d__1 = ex - 1.f;
        co2eml[i__ + (k1 - 1) * co2eml_dim1] = ex * 120000004096.f / (
            tlayr5 * (d__1 * d__1));
        co2t[i__ + k * co2t_dim1] = sum[i__ - 1] / pnm[i__ + k * pnm_dim1]
            ;

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    dbvtt[i__ - 1] = (-2.89113668259233236e-4f + (2.37712515627208631e-6f
        + 1.13051887296045095e-10f * tplnke[i__]) * tplnke[i__]) / (
        1.f + (-.0061364821158349514f + 1.5550320313195698e-5f *
        tplnke[i__]) * tplnke[i__]);

    }

    trcplk_(state, lchnk, ncol, pcols, pver, pverp, &tint[tint_offset], &tlayr[
        tlayr_offset], &tplnke[1], emplnk, &abplnk1[abplnk1_offset], &
        abplnk2[abplnk2_offset]);
    i__1 = (CAM_LEVELS+1);
    for (k1 = state->cg_ntoplw__; k1 <= i__1; ++k1) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        u[i__ - 1] = plh2o[i__ + k1 * plh2o_dim1];
        pnew = u[i__ - 1] / w[i__ + k1 * w_dim1];
        pnew_mks__ = pnew * sslp_mks__;
        uc1[i__ - 1] = (s2c[i__ + k1 * s2c_dim1] + plh2o[i__ + k1 *
            plh2o_dim1] * .00170000002253800631f) * (s2c[i__ + k1 *
            s2c_dim1] * 2.f + 1.f) / (s2c[i__ + k1 * s2c_dim1] * 15.f
            + 1.f);
        pch2o = s2c[i__ + k1 * s2c_dim1];
        tpathe = tcg[i__ + k1 * tcg_dim1] / w[i__ + k1 * w_dim1];

        d__1 = max(tpathe,state->cg_min_tp_h2o__);
        t_p__ = min(d__1,state->cg_max_tp_h2o__);
        iest = (integer) (cam_floor__(&t_p__) -
            state->cg_min_tp_h2o__);
        esx = state->cg_estblh2o__[iest] + (
            state->cg_estblh2o__[iest + 1] -
            state->cg_estblh2o__[iest]) * (t_p__ -
            state->cg_min_tp_h2o__ - iest);
        qsx = state->cg_epsilo__ * esx / (pnew_mks__ - omeps * esx);
        q_path__ = w[i__ + k1 * w_dim1] / pnm[i__ + k1 * pnm_dim1] /
            state->cg_rga__;
        ub[0] = plh2ob[(i__ + k1 * plh2ob_dim2 << 1) + 1] / psi_(state, &t_p__, &
            radems_c__1);
        ub[1] = plh2ob[(i__ + k1 * plh2ob_dim2 << 1) + 2] / psi_(state, &t_p__, &
            radems_c__2);
        pnewb[0] = ub[0] / wb[(i__ + k1 * wb_dim2 << 1) + 1] * phi_(state, &
            t_p__, &radems_c__1);
        pnewb[1] = ub[1] / wb[(i__ + k1 * wb_dim2 << 1) + 2] * phi_(state, &
            t_p__, &radems_c__2);
        dtx[i__ - 1] = tplnke[i__] - 250.f;
        dty[i__ - 1] = tpathe - 250.f;
        te1 = tplnke[i__];
        te2 = te1 * te1;
        te3 = te2 * te1;
        te4 = te3 * te1;
        te5 = te4 * te1;
        dvar = (t_p__ - state->cg_min_tp_h2o__) /
            state->cg_dtp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        itp = min(i__3,9);
        itp1 = itp + 1;
        wtp = dvar - cam_floor__(&dvar);
        wtp1 = 1.f - wtp;


        d__2 = tplnke[i__] - t_p__;
        d__1 = max(d__2,state->cg_min_te_h2o__);
        t_e__ = min(d__1,state->cg_max_te_h2o__);
        dvar = (t_e__ - state->cg_min_te_h2o__) /
            state->cg_dte_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ite = min(i__3,20);
        ite1 = ite + 1;
        wte = dvar - cam_floor__(&dvar);
        wte1 = 1.f - wte;


        d__2 = q_path__ / qsx;
        d__1 = max(d__2,state->cg_min_rh_h2o__);
        rh_path__ = min(d__1,state->cg_max_rh_h2o__);
        dvar = (rh_path__ - state->cg_min_rh_h2o__) /
            state->cg_drh_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        irh = min(i__3,6);
        irh1 = irh + 1;
        wrh = dvar - cam_floor__(&dvar);
        wrh1 = 1.f - wrh;
        w_0_0___ = wtp * wte;
        w_0_1___ = wtp * wte1;
        w_1_0___ = wtp1 * wte;
        w_1_1___ = wtp1 * wte1;
        w_0_00__ = w_0_0___ * wrh;
        w_0_01__ = w_0_0___ * wrh1;
        w_0_10__ = w_0_1___ * wrh;
        w_0_11__ = w_0_1___ * wrh1;
        w_1_00__ = w_1_0___ * wrh;
        w_1_01__ = w_1_0___ * wrh1;
        w_1_10__ = w_1_1___ * wrh;
        w_1_11__ = w_1_1___ * wrh1;
        fch2o = fh2oself_(state, &t_p__);
        uch2o = pch2o * state->cg_epsilo__ / (q_path__ * fch2o);
        ib = 1;
        uvar = ub[ib - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        iu = min(i__3,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnewb[ib - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ip = min(i__3,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;
        fe = state->cg_fet__[ib * 6 - 6] + state->cg_fet__[ib
            * 6 - 5] * te1 + state->cg_fet__[ib * 6 - 4] * te2
            + state->cg_fet__[ib * 6 - 3] * te3 +
            state->cg_fet__[ib * 6 - 2] * te4 +
            state->cg_fet__[ib * 6 - 1] * te5;
        e_star__ = state->cg_eh2onw__[ip + (itp + (iu + (ite + irh *
             21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1 +
            state->cg_eh2onw__[ip + (itp + (iu + (ite + irh1 *
            21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1 +
            state->cg_eh2onw__[ip + (itp + (iu + (ite1 + irh *
            21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1 +
            state->cg_eh2onw__[ip + (itp + (iu + (ite1 + irh1 *
            21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1 +
            state->cg_eh2onw__[ip + (itp + (iu1 + (ite + irh *
            21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_eh2onw__[ip + (itp + (iu1 + (ite + irh1 *
            21) * 25) * 10) * 10 - 55111] * w11_10__ * wu +
            state->cg_eh2onw__[ip + (itp + (iu1 + (ite1 + irh *
            21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_eh2onw__[ip + (itp + (iu1 + (ite1 + irh1 *
             21) * 25) * 10) * 10 - 55111] * w11_00__ * wu +
            state->cg_eh2onw__[ip + (itp1 + (iu + (ite + irh *
            21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1 +
            state->cg_eh2onw__[ip + (itp1 + (iu + (ite + irh1 *
            21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1 +
            state->cg_eh2onw__[ip + (itp1 + (iu + (ite1 + irh *
            21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1 +
            state->cg_eh2onw__[ip + (itp1 + (iu + (ite1 + irh1 *
             21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1 +
            state->cg_eh2onw__[ip + (itp1 + (iu1 + (ite + irh *
            21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_eh2onw__[ip + (itp1 + (iu1 + (ite + irh1 *
             21) * 25) * 10) * 10 - 55111] * w10_10__ * wu +
            state->cg_eh2onw__[ip + (itp1 + (iu1 + (ite1 + irh *
             21) * 25) * 10) * 10 - 55111] * w10_01__ * wu +
            state->cg_eh2onw__[ip + (itp1 + (iu1 + (ite1 + irh1
            * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu +
            state->cg_eh2onw__[ip1 + (itp + (iu + (ite + irh *
            21) * 25) * 10) * 10 - 55111] * w01_11__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp + (iu + (ite + irh1 *
            21) * 25) * 10) * 10 - 55111] * w01_10__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp + (iu + (ite1 + irh *
            21) * 25) * 10) * 10 - 55111] * w01_01__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp + (iu + (ite1 + irh1 *
             21) * 25) * 10) * 10 - 55111] * w01_00__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp + (iu1 + (ite + irh *
            21) * 25) * 10) * 10 - 55111] * w01_11__ * wu +
            state->cg_eh2onw__[ip1 + (itp + (iu1 + (ite + irh1 *
             21) * 25) * 10) * 10 - 55111] * w01_10__ * wu +
            state->cg_eh2onw__[ip1 + (itp + (iu1 + (ite1 + irh *
             21) * 25) * 10) * 10 - 55111] * w01_01__ * wu +
            state->cg_eh2onw__[ip1 + (itp + (iu1 + (ite1 + irh1
            * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu +
            state->cg_eh2onw__[ip1 + (itp1 + (iu + (ite + irh *
            21) * 25) * 10) * 10 - 55111] * w00_11__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp1 + (iu + (ite + irh1 *
             21) * 25) * 10) * 10 - 55111] * w00_10__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp1 + (iu + (ite1 + irh *
             21) * 25) * 10) * 10 - 55111] * w00_01__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp1 + (iu + (ite1 + irh1
            * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wu1 +
            state->cg_eh2onw__[ip1 + (itp1 + (iu1 + (ite + irh *
             21) * 25) * 10) * 10 - 55111] * w00_11__ * wu +
            state->cg_eh2onw__[ip1 + (itp1 + (iu1 + (ite + irh1
            * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu +
            state->cg_eh2onw__[ip1 + (itp1 + (iu1 + (ite1 + irh
            * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu +
            state->cg_eh2onw__[ip1 + (itp1 + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wu;


        d__2 = fe * (1.f - (1.f - e_star__) * aer_trn_ttl__[i__ + (k1 + (
            ib * aer_trn_ttl_dim3 + 1) * aer_trn_ttl_dim2) *
            aer_trn_ttl_dim1]);
        d__1 = max(d__2,0.);
        emis[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
        uscl = uvar / state->cg_min_u_h2o__;
        emis[i__ + ib - 2] *= uscl;
        }
        ib = 2;
        uvar = ub[ib - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        iu = min(i__3,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnewb[ib - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ip = min(i__3,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;


        d__3 = uch2o * fdif;
        d__2 = max(d__3,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_uc__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_uc__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        iuc = min(i__3,24);
        iuc1 = iuc + 1;
        wuc = dvar - cam_floor__(&dvar);
        wuc1 = 1.f - wuc;
        fe = state->cg_fet__[ib * 6 - 6] + state->cg_fet__[ib
            * 6 - 5] * te1 + state->cg_fet__[ib * 6 - 4] * te2
            + state->cg_fet__[ib * 6 - 3] * te3 +
            state->cg_fet__[ib * 6 - 2] * te4 +
            state->cg_fet__[ib * 6 - 1] * te5;
        l_star__ = state->cg_ln_eh2ow__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp + (iu + (ite + irh1 *
             21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp + (iu + (ite1 + irh *
             21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp + (iu + (ite1 + irh1
            * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp + (iu1 + (ite + irh *
             21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ln_eh2ow__[ip + (itp + (iu1 + (ite + irh1
            * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu +
            state->cg_ln_eh2ow__[ip + (itp + (iu1 + (ite1 + irh
            * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ln_eh2ow__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu + (ite + irh *
             21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu + (ite + irh1
            * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu + (ite1 + irh
            * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1 +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu1 + (ite + irh
            * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu +
            state->cg_ln_eh2ow__[ip + (itp1 + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu + (ite + irh *
             21) * 25) * 10) * 10 - 55111] * w01_11__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu + (ite + irh1
            * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu + (ite1 + irh
            * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu1 + (ite + irh
            * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu + (ite + irh
            * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wu1 +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu +
            state->cg_ln_eh2ow__[ip1 + (itp1 + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wu;
        c_star__ = state->cg_cn_eh2ow__[ip + (itp + (iuc + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp + (iuc + (ite + irh1
            * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp + (iuc + (ite1 + irh
            * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp + (iuc + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp + (iuc1 + (ite + irh
            * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp + (iuc1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp + (iuc1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp + (iuc1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc + (ite + irh
            * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wuc1 +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wuc +
            state->cg_cn_eh2ow__[ip + (itp1 + (iuc1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc + (ite + irh
            * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp + (iuc1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wuc1 +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wuc +
            state->cg_cn_eh2ow__[ip1 + (itp1 + (iuc1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wuc;


        d__2 = fe * (1.f - l_star__ * c_star__ * aer_trn_ttl__[i__ + (k1
            + (ib * aer_trn_ttl_dim3 + 1) * aer_trn_ttl_dim2) *
            aer_trn_ttl_dim1]);
        d__1 = max(d__2,0.);
        emis[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
        uscl = uvar / state->cg_min_u_h2o__;
        emis[i__ + ib - 2] *= uscl;
        }
        h2oems[i__ + k1 - 2] = emis[i__ - 1] + emis[i__];

    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        term7[i__ - 1] = state->cg_coefj__[0] +
            state->cg_coefj__[1] * dty[i__ - 1] * (
            state->cg_c16__ * dty[i__ - 1] + 1.f);
        term8[i__ - 1] = state->cg_coefk__[0] +
            state->cg_coefk__[1] * dty[i__ - 1] * (
            state->cg_c17__ * dty[i__ - 1] + 1.f);
        term7[i__] = state->cg_coefj__[3] + state->cg_coefj__[
            4] * dty[i__ - 1] * (state->cg_c26__ * dty[i__ - 1]
            + 1.f);
        term8[i__] = state->cg_coefk__[3] + state->cg_coefk__[
            4] * dty[i__ - 1] * (state->cg_c27__ * dty[i__ - 1]
            + 1.f);

    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        k21[i__ - 1] = term7[i__ - 1] + term8[i__ - 1] / ((
            state->cg_c30__ + state->cg_c31__ * (dty[i__
            - 1] - 10.f) * (dty[i__ - 1] - 10.f)) * sqrt(u[i__ - 1])
            + 1.f);
        k22[i__ - 1] = term7[i__] + term8[i__] / ((state->cg_c28__
            + state->cg_c29__ * (dty[i__ - 1] - 10.f)) * sqrt(u[
            i__ - 1]) + 1.f);
        fwk = state->cg_fwcoef__ + state->cg_fwc1__ / (
            state->cg_fwc2__ * u[i__ - 1] + 1.f);
        tr1[i__ - 1] = cam_exp64(-(k21[i__ - 1] * (sqrt(u[i__ - 1]) +
            state->cg_fc1__ * fwk * u[i__ - 1])));
        tr2[i__ - 1] = cam_exp64(-(k22[i__ - 1] * (sqrt(u[i__ - 1]) +
            state->cg_fc1__ * fwk * u[i__ - 1])));
        tr1[i__ - 1] *= aer_trn_ttl__[i__ + (k1 + ((aer_trn_ttl_dim3 << 2)
             + 1) * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
        tr2[i__ - 1] *= aer_trn_ttl__[i__ + (k1 + (aer_trn_ttl_dim3 * 3 +
            1) * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
        tr3[i__ - 1] = cam_exp64(-((state->cg_coefh__[0] +
            state->cg_coefh__[1] * dtx[i__ - 1]) * uc1[i__ - 1])
            );
        tr4[i__ - 1] = cam_exp64(-((state->cg_coefh__[2] +
            state->cg_coefh__[3] * dtx[i__ - 1]) * uc1[i__ - 1])
            );
        tr7[i__ - 1] = tr1[i__ - 1] * tr3[i__ - 1];
        tr8[i__ - 1] = tr2[i__ - 1] * tr4[i__ - 1];
        troco2[i__ + k1 - 2] = tr7[i__ - 1] * .64999997615814209f + tr8[
            i__ - 1] * .349999994039535522f;
        th2o[i__ - 1] = tr8[i__ - 1];

    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        t1i = cam_exp64(-480.f / co2t[i__ + k1 * co2t_dim1]);
        sqti = sqrt(co2t[i__ + k1 * co2t_dim1]);
        rsqti = 1.f / sqti;
        et = t1i;
        et2 = et * et;
        et4 = et2 * et2;
        omet = 1.f - et2 * 1.5f;
        f1co2 = omet * 899.70001220703125f * (et * 1.94773995876312256f +
            1.f + et2 * 4.73485994338989258f) * rsqti;
        sqwp = sqrt(plco2[i__ + k1 * plco2_dim1]);
        f1sqwp = f1co2 * sqwp;
        t1co2 = 1.f / (omet * 245.17999267578125f * sqwp * rsqti + 1.f);
        oneme = 1.f - et2;

        d__1 = oneme;
        alphat = d__1 * (d__1 * d__1) * rsqti;
        wco2 = state->cg_co2vmr__ * 2.52209997177124023f * pnm[i__
            + k1 * pnm_dim1] * state->cg_rga__;
        u7 = alphat * 49411.f * et2 * wco2;
        u8 = alphat * 39744.f * et4 * wco2;
        u9 = alphat * 104470.f * et4 * et2 * wco2;
        u13 = alphat * 2838.800048828125f * et4 * wco2;
        tpath = co2t[i__ + k1 * co2t_dim1];
        tlocal = tplnke[i__];
        tcrfac = sqrt(tlocal * r250 * (tpath * r300));
        pi = pnm[i__ + k1 * pnm_dim1] * rsslp + state->cg_dpfco2__ *
             2.f * tcrfac;
        posqt = pi / (sqti * 2.f);
        rbeta7 = 1.f / (posqt * 5.32880020141601562f);
        rbeta8 = 1.f / (posqt * 10.6576004028320312f);
        rbeta9 = rbeta7;
        rbeta13 = rbeta9;
        f2co2 = u7 / sqrt(u7 * (rbeta7 + 1.f) + 4.f) + u8 / sqrt(u8 * (
            rbeta8 + 1.f) + 4.f) + u9 / sqrt(u9 * (rbeta9 + 1.f) +
            4.f);
        f3co2 = u13 / sqrt(u13 * (rbeta13 + 1.f) + 4.f);
        tmp1 = cam_log64(f1sqwp + 1.f);
        tmp2 = cam_log64(f2co2 + 1.f);
        tmp3 = cam_log64(f3co2 + 1.f);
        absbnd = (tmp1 + t1co2 * 2.f * tmp2 + tmp3 * 2.f) * sqti;
        tco2[i__ - 1] = 1.f / (u7 / sqrt(u7 * (rbeta7 + 1.f) + 4.f) *
            10.f + 1.f);
        co2ems[i__ + k1 - 2] = troco2[i__ + k1 - 2] * absbnd * co2plk[i__
            - 1];
        ex = cam_exp64(960.f / tint[i__ + k1 * tint_dim1]);

        d__1 = ex - 1.f;
        exm1sq = d__1 * d__1;
        co2em[i__ + k1 * co2em_dim1] = ex * 120000004096.f / (tint[i__ +
            k1 * tint_dim1] * tint4[i__ + k1 * tint4_dim1] * exm1sq);

    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        h2otr[i__ + k1 * h2otr_dim1] = cam_exp64(s2c[i__ + k1 * s2c_dim1] *
            -12.f);
        h2otr[i__ + k1 * h2otr_dim1] *= aer_trn_ttl__[i__ + (k1 + (
            aer_trn_ttl_dim3 * 6 + 1) * aer_trn_ttl_dim2) *
            aer_trn_ttl_dim1];
        d__1 = co2t[i__ + k1 * co2t_dim1] / 293.f;
        te = pow_dd(&d__1, &radems_c_b22);
        u1 = plos[i__ + k1 * plos_dim1] * 18.2900009155273438f / te;
        u2 = plos[i__ + k1 * plos_dim1] * .564899981021881104f / te;
        phat = plos[i__ + k1 * plos_dim1] / plol[i__ + k1 * plol_dim1];
        tlocal = tplnke[i__];
        tcrfac = sqrt(tlocal * r250) * te;
        beta = (1.f / phat + state->cg_dpfo3__ * tcrfac) *
            3.1201249372919126f;
        realnu = 1.f / beta * te;
        o3bndi = te * 74.f * (tplnke[i__] / 375.f) * cam_log64(u1 / sqrt(4.f +
            u1 * (1.f + realnu)) + 1.f + u2 / sqrt(4.f + u2 * (1.f +
            realnu)));
        o3ems[i__ + k1 - 2] = dbvtt[i__ - 1] * h2otr[i__ + k1 *
            h2otr_dim1] * o3bndi;
        to3[i__ - 1] = 1.f / (u1 / sqrt(4.f + u1 * (1.f + realnu)) *
            .100000001490116119f + 1.f + u2 / sqrt(4.f + u2 * (1.f +
            realnu)) * .100000001490116119f);

    }

    trcems_(state, lchnk, ncol, pcols, pverp, &k1, &co2t[co2t_offset], &pnm[
        pnm_offset], &ucfc11[ucfc11_offset], &ucfc12[ucfc12_offset], &
        un2o0[un2o0_offset], &un2o1[un2o1_offset], &bn2o0[
        bn2o0_offset], &bn2o1[bn2o1_offset], &uch4[uch4_offset], &
        bch4[bch4_offset], &uco211[uco211_offset], &uco212[
        uco212_offset], &uco213[uco213_offset], &uco221[uco221_offset]
        , &uco222[uco222_offset], &uco223[uco223_offset], &uptype[
        uptype_offset], &w[w_offset], &s2c[s2c_offset], u, emplnk,
        th2o, tco2, to3, emstrc, &aer_trn_ttl__[aer_trn_ttl_offset]);
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        emstot[i__ + k1 * emstot_dim1] = h2oems[i__ + k1 - 2] + co2ems[
            i__ + k1 - 2] + o3ems[i__ + k1 - 2] + emstrc[i__ + k1 - 2]
            ;

    }


    }

    return 0;
}






__device__ doublereal trcplk_c_b2 = 4.;
__device__ doublereal trcplk_c_b7 = 5.;
__device__ doublereal trcplk_c_b8 = 2.;


__device__ int trcplk_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *tint, doublereal *tlayr,
    doublereal *tplnke, doublereal *emplnk, doublereal *abplnk1,
    doublereal *abplnk2)
{


    doublereal f1[14] = { 585713024.,794950016.,1470089984.,
        1400310016.,134852992.,1051580032.,335369984.,399600992.,
        535993984.,842955008.,463681984.,5.18944e8,883201984.,1032790016.
        };
    doublereal f2[14] = { 202493001728.,304286007296.,690697994240.,
        647333019648.,28574400512.,441861996544.,96278003712.,
        121617997824.,179905003520.,329028993024.,148293992448.,
        172315000832.,350140006400.,431364014080. };
    doublereal f3[14] = { 1383.,1531.,1879.,1849.,848.,1681.,1148.,
        1217.,1343.,1561.,1279.,1328.,1586.,1671. };

    integer tint_dim1 = 0,tint_offset = 0,tlayr_dim1 = 0,tlayr_offset = 0,abplnk1_dim2 = 0,abplnk1_offset = 0,abplnk2_dim2 = 0,abplnk2_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,k = 0,wvl = 0;


    emplnk -= 15;
    --tplnke;
    abplnk2_dim2 = 1;
    abplnk2_offset = 1 + 14 * (1 + abplnk2_dim2);
    abplnk2 -= abplnk2_offset;
    abplnk1_dim2 = 1;
    abplnk1_offset = 1 + 14 * (1 + abplnk1_dim2);
    abplnk1 -= abplnk1_offset;
    tlayr_dim1 = 1;
    tlayr_offset = 1 + tlayr_dim1;
    tlayr -= tlayr_offset;
    tint_dim1 = 1;
    tint_offset = 1 + tint_dim1;
    tint -= tint_offset;


    for (wvl = 1; wvl <= 14; ++wvl) {
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        emplnk[wvl + i__ * 14] = f1[wvl - 1] / (pow_dd(&tplnke[i__], &
            trcplk_c_b2) * (cam_exp64(f3[wvl - 1] / tplnke[i__]) - 1.f));

    }


    }

    for (wvl = 1; wvl <= 14; ++wvl) {
    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        d__1 = cam_exp64(f3[wvl - 1] / tint[i__ + k * tint_dim1]) - 1.f;
        abplnk1[wvl + (i__ + k * abplnk1_dim2) * 14] = f2[wvl - 1] *
            cam_exp64(f3[wvl - 1] / tint[i__ + k * tint_dim1]) / (
            pow_dd(&tint[i__ + k * tint_dim1], &trcplk_c_b7) * pow_dd(&
            d__1, &trcplk_c_b8));
        d__1 = cam_exp64(f3[wvl - 1] / tlayr[i__ + k * tlayr_dim1]) - 1.f;
        abplnk2[wvl + (i__ + k * abplnk2_dim2) * 14] = f2[wvl - 1] *
            cam_exp64(f3[wvl - 1] / tlayr[i__ + k * tlayr_dim1]) / (
            pow_dd(&tlayr[i__ + k * tlayr_dim1], &trcplk_c_b7) * pow_dd(&
            d__1, &trcplk_c_b8));

        }


    }


    }

    return 0;
}





__device__ int trcems_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pverp, integer *k, doublereal *co2t, doublereal *pnm,
    doublereal *ucfc11, doublereal *ucfc12, doublereal *un2o0, doublereal
    *un2o1, doublereal *bn2o0, doublereal *bn2o1, doublereal *uch4,
    doublereal *bch4, doublereal *uco211, doublereal *uco212, doublereal *
    uco213, doublereal *uco221, doublereal *uco222, doublereal *uco223,
    doublereal *uptype, doublereal *w, doublereal *s2c, doublereal *up2,
    doublereal *emplnk, doublereal *th2o, doublereal *tco2, doublereal *
    to3, doublereal *emstrc, doublereal *aer_trn_ttl__)
{
 CamScratchFrame scratch_frame(state);

    doublereal g1[6] = { .0468555986881256104,.0397454015910625458,
        .0407663993537425995,.0304380003362894058,.0540397986769676208,
        .0321962013840675354 };

    doublereal g2[6] = { 14.4832000732421875,4.30242013931274414,
        5.2352299690246582,3.25342011451721191,.698934972286224365,
        16.5599002838134766 };

    doublereal g3[6] = { 26.1898002624511719,18.4475994110107422,
        15.3633003234863281,12.1927003860473633,9.14992046356201172,
        8.07091999053955078 };

    doublereal g4[6] = { .0261781997978687286,.036951601505279541,
        .0307266004383563995,.024385400116443634,.0182932000607252121,
        .0161418002098798752 };

    doublereal ab[6] = { .0308570004999637604,.0235239993780851364,
        .017309999093413353,.0266609992831945419,.0280740000307559967,
        .0229150000959634781 };

    doublereal bb[6] = { -1.35120004415512085e-4,
        -6.83199978084303439e-5,-3.26090012094937265e-5,
        -1.02280000646715052e-5,-9.57430020207539201e-5,
        -1.03040001704357564e-4 };

    doublereal abp[6] = { .0291290003806352615,.0241010002791881561,
        .0198209993541240692,.0269039999693632126,.0294579993933439255,
        .0198919996619224548 };

    doublereal bbp[6] = { -1.3138999929651618e-4,
        -5.56880004296544939e-5,-4.63799988210666925e-5,
        -8.03619986982084811e-5,-1.01149998954497278e-4,
        -8.80610023159533739e-5 };

    integer co2t_dim1 = 0,co2t_offset = 0,pnm_dim1 = 0,pnm_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,uptype_dim1 = 0,uptype_offset = 0,bn2o0_dim1 = 0,bn2o0_offset = 0,bn2o1_dim1 = 0,bn2o1_offset = 0,bch4_dim1 = 0,bch4_offset = 0,s2c_dim1 = 0,s2c_offset = 0,w_dim1 = 0,w_offset = 0,aer_trn_ttl_dim1 = 0,aer_trn_ttl_dim2 = 0,aer_trn_ttl_dim3 = 0,aer_trn_ttl_offset = 0,emstrc_dim1 = 0,emstrc_offset = 0,i__1 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,l = 0;

    doublereal p1 = 0,w1 = 0,u01 = 0,u11 = 0,u02 = 0,u12 = 0,u03 = 0,tt[1] = {},tw[6] = {},tlw = 0,ech4 = 0,phi1 = 0,tch4 = 0,psi1 = 0,eco21 = 0,eco22 = 0,en2o1 = 0,en2o2 = 0,en2o3 = 0,sqti[1] = {},ecfc1 = 0,ecfc2 = 0,ecfc3 = 0,ecfc4 = 0,ecfc5 = 0,ecfc6 = 0,ecfc7 = 0,ecfc8 = 0,tcfc3 = 0,tcfc4 = 0,tcfc6 = 0,tcfc7 = 0,tcfc8 = 0,beta01 = 0,beta11 = 0,beta02 = 0,beta03 = 0,betac = 0,betac1 = 0,betac2 = 0;


    --to3;
    --tco2;
    --th2o;
    emplnk -= 15;
    --up2;
    aer_trn_ttl_dim1 = 1;
    aer_trn_ttl_dim2 = (CAM_LEVELS+1);
    aer_trn_ttl_dim3 = (CAM_LEVELS+1);
    aer_trn_ttl_offset = 1 + aer_trn_ttl_dim1 * (1 + aer_trn_ttl_dim2 * (1 +
        aer_trn_ttl_dim3));
    aer_trn_ttl__ -= aer_trn_ttl_offset;
    emstrc_dim1 = 1;
    emstrc_offset = 1 + emstrc_dim1;
    emstrc -= emstrc_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;
    w_dim1 = 1;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    bch4_dim1 = 1;
    bch4_offset = 1 + bch4_dim1;
    bch4 -= bch4_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    bn2o1_dim1 = 1;
    bn2o1_offset = 1 + bn2o1_dim1;
    bn2o1 -= bn2o1_offset;
    bn2o0_dim1 = 1;
    bn2o0_offset = 1 + bn2o0_dim1;
    bn2o0 -= bn2o0_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;
    co2t_dim1 = 1;
    co2t_offset = 1 + co2t_dim1;
    co2t -= co2t_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    sqti[i__ - 1] = sqrt(co2t[i__ + *k * co2t_dim1]);
    tt[i__ - 1] = (d__1 = co2t[i__ + *k * co2t_dim1] - 250.f, abs(d__1));

    }

    for (l = 1; l <= 6; ++l) {
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        psi1 = cam_exp64(abp[l - 1] * tt[i__ - 1] + bbp[l - 1] * tt[i__ - 1] *
            tt[i__ - 1]);
        phi1 = cam_exp64(ab[l - 1] * tt[i__ - 1] + bb[l - 1] * tt[i__ - 1] * tt[
            i__ - 1]);
        p1 = pnm[i__ + *k * pnm_dim1] * (psi1 / phi1) /
            state->cg_sslp__;
        w1 = w[i__ + *k * w_dim1] * phi1;
        tw[i__ + l - 2] = cam_exp64(-g1[l - 1] * p1 * (sqrt(g2[l - 1] * (w1 /
            p1) + 1.f) - 1.f) - g3[l - 1] * s2c[i__ + *k * s2c_dim1]
            - g4[l - 1] * uptype[i__ + *k * uptype_dim1]);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tw[i__ - 1] *= aer_trn_ttl__[i__ + (*k + ((aer_trn_ttl_dim3 << 2) + 1)
         * aer_trn_ttl_dim2) * aer_trn_ttl_dim1] *
        .699999988079071045f + aer_trn_ttl__[i__ + (*k + (
        aer_trn_ttl_dim3 * 5 + 1) * aer_trn_ttl_dim2) *
        aer_trn_ttl_dim1] * .300000011920928955f;
    tw[i__] *= aer_trn_ttl__[i__ + (*k + (aer_trn_ttl_dim3 * 5 + 1) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 1] *= aer_trn_ttl__[i__ + (*k + (aer_trn_ttl_dim3 * 5 + 1) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 2] *= aer_trn_ttl__[i__ + (*k + (aer_trn_ttl_dim3 * 5 + 1) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 3] *= aer_trn_ttl__[i__ + (*k + (aer_trn_ttl_dim3 * 6 + 1) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 4] *= aer_trn_ttl__[i__ + (*k + (aer_trn_ttl_dim3 * 6 + 1) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];

    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tcfc3 = cam_exp64(ucfc11[i__ + *k * ucfc11_dim1] * -175.0050048828125f);
    tcfc4 = cam_exp64(ucfc11[i__ + *k * ucfc11_dim1] * -1202.1800537109375f);
    tcfc6 = cam_exp64(ucfc12[i__ + *k * ucfc12_dim1] * -5786.72998046875f);
    tcfc7 = cam_exp64(ucfc12[i__ + *k * ucfc12_dim1] * -2873.510009765625f);
    tcfc8 = cam_exp64(ucfc12[i__ + *k * ucfc12_dim1] * -2085.590087890625f);
    ecfc1 = (1.f - cam_exp64(ucfc11[i__ + *k * ucfc11_dim1] *
        -54.0900001525878906f)) * 50.f * tw[i__ - 1] * emplnk[i__ *
        14 + 7];
    ecfc2 = (1.f - cam_exp64(ucfc11[i__ + *k * ucfc11_dim1] *
        -5130.02978515625f)) * 60.f * tw[i__] * emplnk[i__ * 14 + 8];
    ecfc3 = (1.f - tcfc3) * 60.f * tw[i__ + 2] * tcfc6 * emplnk[i__ * 14
        + 9];
    ecfc4 = (1.f - tcfc4) * 100.f * tw[i__ + 3] * emplnk[i__ * 14 + 10];
    ecfc5 = (1.f - cam_exp64(ucfc12[i__ + *k * ucfc12_dim1] *
        -1272.3499755859375f)) * 45.f * tw[i__ + 1] * emplnk[i__ * 14
        + 11];
    ecfc6 = (1.f - tcfc6) * 50.f * tw[i__ + 2] * emplnk[i__ * 14 + 12];
    ecfc7 = (1.f - tcfc7) * 80.f * tw[i__ + 3] * tcfc4 * emplnk[i__ * 14
        + 13];
    ecfc8 = (1.f - tcfc8) * 70.f * tw[i__ + 4] * emplnk[i__ * 14 + 14];
    tlw = cam_exp64(sqrt(up2[i__]) * -1.f);
    tlw *= aer_trn_ttl__[i__ + (*k + (aer_trn_ttl_dim3 * 7 + 1) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    betac = bch4[i__ + *k * bch4_dim1] / uch4[i__ + *k * uch4_dim1];
    ech4 = sqti[i__ - 1] * 6.0044398307800293f * cam_log64(uch4[i__ + *k *
        uch4_dim1] / sqrt(4.f + uch4[i__ + *k * uch4_dim1] * (1.f +
        1.f / betac)) + 1.f) * tlw * emplnk[i__ * 14 + 3];
    tch4 = 1.f / (uch4[i__ + *k * uch4_dim1] / sqrt(4.f + uch4[i__ + *k *
        uch4_dim1] * (1.f + 1.f / betac)) * .0199999995529651642f +
        1.f);
    u01 = un2o0[i__ + *k * un2o0_dim1];
    u11 = un2o1[i__ + *k * un2o1_dim1];
    beta01 = bn2o0[i__ + *k * bn2o0_dim1] / un2o0[i__ + *k * un2o0_dim1];
    beta11 = bn2o1[i__ + *k * bn2o1_dim1] / un2o1[i__ + *k * un2o1_dim1];
    en2o1 = sqti[i__ - 1] * 2.35558009147644043f * cam_log64(u01 / sqrt(4.f +
        u01 * (1.f + 1.f / beta01)) + 1.f + u11 / sqrt(4.f + u11 * (
        1.f + 1.f / beta11))) * tlw * tch4 * emplnk[i__ * 14 + 4];
    u02 = u01 * .100089997053146362f;
    u12 = u11 * .0992745980620384216f;
    beta02 = beta01 * .964281976222991943f;
    en2o2 = sqti[i__ - 1] * 2.65581011772155762f * cam_log64(u02 / sqrt(4.f +
        u02 * (1.f + 1.f / beta02)) + 1.f + u12 / sqrt(4.f + u12 * (
        1.f + 1.f / beta02))) * tco2[i__] * th2o[i__] * emplnk[i__ *
        14 + 5];
    u03 = u01 * .0333767011761665344f;
    beta03 = beta01 * .982142984867095947f;
    en2o3 = sqti[i__ - 1] * 2.54033994674682617f * cam_log64(u03 / sqrt(4.f +
        u03 * (1.f + 1.f / beta03)) + 1.f) * tw[i__ + 4] * tcfc8 *
        emplnk[i__ * 14 + 6];
    betac1 = pnm[i__ + *k * pnm_dim1] * 2.97557997703552246f / (
        state->cg_sslp__ * sqti[i__ - 1]);
    betac2 = betac1 * 2.f;
    eco21 = sqti[i__ - 1] * 3.75710010528564453f * cam_log64(uco211[i__ + *k *
        uco211_dim1] / sqrt(4.f + uco211[i__ + *k * uco211_dim1] * (
        1.f + 1.f / betac1)) + 1.f + uco212[i__ + *k * uco212_dim1] /
        sqrt(4.f + uco212[i__ + *k * uco212_dim1] * (1.f + 1.f /
        betac2)) + uco213[i__ + *k * uco213_dim1] / sqrt(4.f + uco213[
        i__ + *k * uco213_dim1] * (1.f + 1.f / betac2))) * to3[i__] *
        tw[i__ + 3] * tcfc4 * tcfc7 * emplnk[i__ * 14 + 2];
    eco22 = sqti[i__ - 1] * 3.8443000316619873f * cam_log64(uco221[i__ + *k *
        uco221_dim1] / sqrt(4.f + uco221[i__ + *k * uco221_dim1] * (
        1.f + 1.f / betac1)) + 1.f + uco222[i__ + *k * uco222_dim1] /
        sqrt(4.f + uco222[i__ + *k * uco222_dim1] * (1.f + 1.f /
        betac1)) + uco223[i__ + *k * uco223_dim1] / sqrt(4.f + uco223[
        i__ + *k * uco223_dim1] * (1.f + 1.f / betac2))) * tw[i__ + 2]
         * tcfc3 * tcfc6 * emplnk[i__ * 14 + 1];
    emstrc[i__ + *k * emstrc_dim1] = ecfc1 + ecfc2 + ecfc3 + ecfc4 +
        ecfc5 + ecfc6 + ecfc7 + ecfc8 + en2o1 + en2o2 + en2o3 + ech4
        + eco21 + eco22;

    }

    return 0;
}





__device__ doublereal psi_(CamState* state, doublereal *tpx, integer *iband)
{


    doublereal psi_r0__[2] = { .565308451652526855,-73.0087890625 };
    doublereal psi_r1__[2] = { .00407519005239009857,
        1.22199547290802002 };
    doublereal psi_r2__[2] = { -1.0434723662910983e-5,
        -.00712256226688623428 };
    doublereal psi_r3__[2] = { 1.23765353521321231e-8,
        1.47852824738947675e-5 };

    doublereal ret_val = 0;

    ret_val = ((psi_r3__[(0 + (0 + (*iband - 1 << 3))) / 8] * *tpx + psi_r2__[
        (0 + (0 + (*iband - 1 << 3))) / 8]) * *tpx + psi_r1__[(0 + (0 + (*
        iband - 1 << 3))) / 8]) * *tpx + psi_r0__[(0 + (0 + (*iband - 1 <<
         3))) / 8];
    return ret_val;
}





__device__ doublereal phi_(CamState* state, doublereal *tpx, integer *iband)
{


    doublereal phi_r0__[2] = { .96091771125793457,-22.1031341552734375
        };
    doublereal phi_r1__[2] = { 4.86076751258224249e-4,
        .424062609672546387 };
    doublereal phi_r2__[2] = { -1.84806265224324306e-6,
        -.00295543414540588856 };
    doublereal phi_r3__[2] = { 2.1123995885830027e-9,
        7.52470896259183064e-6 };

    doublereal ret_val = 0;

    ret_val = ((phi_r3__[(0 + (0 + (*iband - 1 << 3))) / 8] * *tpx + phi_r2__[
        (0 + (0 + (*iband - 1 << 3))) / 8]) * *tpx + phi_r1__[(0 + (0 + (*
        iband - 1 << 3))) / 8]) * *tpx + phi_r0__[(0 + (0 + (*iband - 1 <<
         3))) / 8];
    return ret_val;
}






__device__ doublereal fh2oself_c_b2 = 2.07274842262268066;


__device__ doublereal fh2oself_(CamState* state, doublereal *temp)
{
    doublereal ret_val = 0,d__1 = 0;

    d__1 = (296.f - *temp) / 36.f;
    ret_val = pow_dd(&fh2oself_c_b2, &d__1);
    return ret_val;
}






__device__ integer radabs_c__1 = 1;
__device__ integer radabs_c__2 = 2;
__device__ doublereal radabs_c_b43 = .699999988079071045;


__device__ int radabs_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *pbr, doublereal *pnm,
    doublereal *co2em, doublereal *co2eml, doublereal *tplnka, doublereal
    *s2c, doublereal *tcg, doublereal *w, doublereal *h2otr, doublereal *
    plco2, doublereal *plh2o, doublereal *co2t, doublereal *tint,
    doublereal *tlayr, doublereal *plol, doublereal *plos, doublereal *
    pmln, doublereal *piln, doublereal *ucfc11, doublereal *ucfc12,
    doublereal *un2o0, doublereal *un2o1, doublereal *uch4, doublereal *
    uco211, doublereal *uco212, doublereal *uco213, doublereal *uco221,
    doublereal *uco222, doublereal *uco223, doublereal *uptype,
    doublereal *bn2o0, doublereal *bn2o1, doublereal *bch4, doublereal *
    abplnk1, doublereal *abplnk2, doublereal *abstot, doublereal *absnxt,
    doublereal *plh2ob, doublereal *wb, doublereal *aer_mpp__, doublereal
    *aer_trn_ttl__)
{
    if (state->abs_phase == 1) {
        state->abs_args[0] = pbr;
        state->abs_args[1] = pnm;
        state->abs_args[2] = co2em;
        state->abs_args[3] = co2eml;
        state->abs_args[4] = tplnka;
        state->abs_args[5] = s2c;
        state->abs_args[6] = tcg;
        state->abs_args[7] = w;
        state->abs_args[8] = h2otr;
        state->abs_args[9] = plco2;
        state->abs_args[10] = plh2o;
        state->abs_args[11] = co2t;
        state->abs_args[12] = tint;
        state->abs_args[13] = tlayr;
        state->abs_args[14] = plol;
        state->abs_args[15] = plos;
        state->abs_args[16] = pmln;
        state->abs_args[17] = piln;
        state->abs_args[18] = ucfc11;
        state->abs_args[19] = ucfc12;
        state->abs_args[20] = un2o0;
        state->abs_args[21] = un2o1;
        state->abs_args[22] = uch4;
        state->abs_args[23] = uco211;
        state->abs_args[24] = uco212;
        state->abs_args[25] = uco213;
        state->abs_args[26] = uco221;
        state->abs_args[27] = uco222;
        state->abs_args[28] = uco223;
        state->abs_args[29] = uptype;
        state->abs_args[30] = bn2o0;
        state->abs_args[31] = bn2o1;
        state->abs_args[32] = bch4;
        state->abs_args[33] = abplnk1;
        state->abs_args[34] = abplnk2;
        state->abs_args[35] = abstot;
        state->abs_args[36] = absnxt;
        state->abs_args[37] = plh2ob;
        state->abs_args[38] = wb;
        state->abs_args[39] = aer_mpp__;
        state->abs_args[40] = aer_trn_ttl__;
        return 0;
    }
    if (state->abs_phase == 2) {
        for (int i=0;i<(CAM_LEVELS+1)*(CAM_LEVELS+1);++i) abstot[i]=state->abs_cache[i];
        for (int i=0;i<CAM_LEVELS*4;++i) absnxt[i]=state->abs_cache[(CAM_LEVELS+1)*(CAM_LEVELS+1)+i];
        return 0;
    }

 CamScratchFrame scratch_frame(state);

    integer pbr_dim1 = 0,pbr_offset = 0,pnm_dim1 = 0,pnm_offset = 0,co2em_dim1 = 0,co2em_offset = 0,co2eml_dim1 = 0,co2eml_offset = 0,tplnka_dim1 = 0,tplnka_offset = 0,s2c_dim1 = 0,s2c_offset = 0,tcg_dim1 = 0,tcg_offset = 0,w_dim1 = 0,w_offset = 0,h2otr_dim1 = 0,h2otr_offset = 0,plco2_dim1 = 0,plco2_offset = 0,plh2o_dim1 = 0,plh2o_offset = 0,co2t_dim1 = 0,co2t_offset = 0,tint_dim1 = 0,tint_offset = 0,tlayr_dim1 = 0,tlayr_offset = 0,plol_dim1 = 0,plol_offset = 0,plos_dim1 = 0,plos_offset = 0,pmln_dim1 = 0,pmln_offset = 0,piln_dim1 = 0,piln_offset = 0,plh2ob_dim2 = 0,plh2ob_offset = 0,wb_dim2 = 0,wb_offset = 0,aer_mpp_dim1 = 0,aer_mpp_offset = 0,aer_trn_ttl_dim1 = 0,aer_trn_ttl_dim2 = 0,aer_trn_ttl_dim3 = 0,aer_trn_ttl_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,uptype_dim1 = 0,uptype_offset = 0,bn2o0_dim1 = 0,bn2o0_offset = 0,bn2o1_dim1 = 0,bn2o1_offset = 0,bch4_dim1 = 0,bch4_offset = 0,abplnk1_dim2 = 0,abplnk1_offset = 0,abplnk2_dim2 = 0,abplnk2_offset = 0,abstot_dim1 = 0,abstot_dim2 = 0,abstot_offset = 0,absnxt_dim1 = 0,absnxt_dim2 = 0,absnxt_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0,i__5 = 0;

    doublereal d__1 = 0,d__2 = 0,d__3 = 0;

    doublereal pnew_mks__ = 0,sslp_mks__ = 0;

    integer i__ = 0,k = 0;

    doublereal u[1] = {};

    integer k1 = 0,k2 = 0;

    doublereal u1 = 0,u2 = 0,p1 = 0,p2 = 0,u7[1] = {},u8 = 0,u9 = 0,fa = 0;

    integer ib = 0;

    doublereal k21 = 0,k22 = 0,ub[2] = {};

    integer kn = 0;

    doublereal pi = 0,et = 0,u13 = 0,dw[1] = {},te = 0;

    integer iu = 0,ip = 0;

    doublereal wp = 0,wu = 0,r300 = 0,uc1[1] = {},et2 = 0,et4 = 0,r293 = 0,r250 = 0,te1 = 0,te2 = 0,te3 = 0,to3[1] = {},tr1 = 0,tr2 = 0,te4 = 0,tr5 = 0,tr6 = 0,te5 = 0,tr9[1] = {};

    integer iu1 = 0,ip1 = 0;

    doublereal wu1 = 0,wp1 = 0,t_e__ = 0,r3205 = 0,emm[4] = {};

    integer iuc = 0;

    doublereal t_p__ = 0;

    integer ite = 0,irh = 0;

    doublereal tr10[1] = {},fwk[1] = {};

    integer itp = 0;

    doublereal dtx[1] = {},dty[1] = {},wuc = 0,wte = 0,wrh = 0,esx = 0;

    integer wvl = 0;

    doublereal wtp = 0,qsx = 0,aer_pth_ngh__[1] = {};

    doublereal aer_pth_dlt__ = 0,aer_trn_ngh__[7] = {},ds2c = 0;

    integer iuc1 = 0,ite1 = 0,irh1 = 0;

    doublereal tco2[1] = {},wco2 = 0,th2o[1] = {};

    integer itp1 = 0;

    doublereal wuc1 = 0,wte1 = 0,tmp1 = 0,tmp2 = 0,tmp3 = 0,wrh1 = 0,wtp1 = 0,w00_00__ = 0,w00_01__ = 0,w00_10__ = 0,w00_11__ = 0,f1co2 = 0,f2co2[1] = {},f3co2[1] = {},beta = 0,w01_00__ = 0,w01_01__ = 0,w01_10__ = 0,w01_11__ = 0,abso[4] = {},tbar[4] = {},t1co2[1] = {},dpnm[1] = {},dvar = 0,pnew[1] = {},omet = 0,fwku[1] = {},sqti[1] = {},uvar = 0,uscl = 0,w10_00__ = 0,w_0_00__ = 0,w_0_01__ = 0,w_0_10__ = 0,w_0_11__ = 0,w_1_00__ = 0,sqwp = 0,w_1_01__ = 0,w_1_10__ = 0,w_1_11__ = 0,w10_01__ = 0,w10_10__ = 0,w10_11__ = 0,w11_00__ = 0,w11_01__ = 0,w11_10__ = 0,w11_11__ = 0,fch2o = 0,fdif = 0;

    integer iest = 0;

    doublereal o3emm[4] = {},trab2[1] = {},pch2o = 0,uch2o = 0,odap_aer_ttl__ = 0,term7[2] = {},term8[2] = {},w_0_0___ = 0,w_0_1___ = 0,w_1_0___ = 0,w_1_1___ = 0;

    doublereal log_p__, denom, oneme, log_u__, bplnk[56] = {}    , to3co2[1] = {}, pnewb[2] = {}, dplol, to3h2o[1] = {}, rphat, dplos,
        pinpl[4] = {}    , tpath, rdpnm, uinpl[4] = {}    , dtym10, winpl[4] = {}  , omeps,
        pnmsq[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {} , zinpl[4] = {}  ,
        rsqti, rsslp, posqt, o3bndi, sqrtu[1] = {}, dplco2, rbeta7[1] = {}, rbeta8,
        rbeta9, dplh2o[1] = {}, temh2o[4] = {}    , f1sqwp[1] = {},
        r2sslp, absbnd, rbeta13, tcrfac, log_uc__, alphat, a_star__,
        c_star__, q_path__;

    doublereal abstrc[1] = {}, tlocal, tpatha, l_star__, realnu, dbvtit[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}
        , dbvtly[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  ;

    integer cam_s1_0__ = 0,cam_s2_0__ = 0,cam_s3_0__ = 0,cam_s4_1__ = 0,bnd_idx__ = 0;

    doublereal rh_path__ = 0,rdpnmsq = 0;


    absnxt_dim1 = 1;
    absnxt_dim2 = CAM_LEVELS;
    absnxt_offset = 1 + absnxt_dim1 * (1 + absnxt_dim2);
    absnxt -= absnxt_offset;
    pmln_dim1 = 1;
    pmln_offset = 1 + pmln_dim1;
    pmln -= pmln_offset;
    co2eml_dim1 = 1;
    co2eml_offset = 1 + co2eml_dim1;
    co2eml -= co2eml_offset;
    pbr_dim1 = 1;
    pbr_offset = 1 + pbr_dim1;
    pbr -= pbr_offset;
    aer_trn_ttl_dim1 = 1;
    aer_trn_ttl_dim2 = (CAM_LEVELS+1);
    aer_trn_ttl_dim3 = (CAM_LEVELS+1);
    aer_trn_ttl_offset = 1 + aer_trn_ttl_dim1 * (1 + aer_trn_ttl_dim2 * (1 +
        aer_trn_ttl_dim3));
    aer_trn_ttl__ -= aer_trn_ttl_offset;
    aer_mpp_dim1 = 1;
    aer_mpp_offset = 1 + aer_mpp_dim1;
    aer_mpp__ -= aer_mpp_offset;
    wb_dim2 = 1;
    wb_offset = 1 + 2 * (1 + wb_dim2);
    wb -= wb_offset;
    plh2ob_dim2 = 1;
    plh2ob_offset = 1 + 2 * (1 + plh2ob_dim2);
    plh2ob -= plh2ob_offset;
    abstot_dim1 = 1;
    abstot_dim2 = (CAM_LEVELS+1);
    abstot_offset = 1 + abstot_dim1 * (1 + abstot_dim2);
    abstot -= abstot_offset;
    abplnk2_dim2 = 1;
    abplnk2_offset = 1 + 14 * (1 + abplnk2_dim2);
    abplnk2 -= abplnk2_offset;
    abplnk1_dim2 = 1;
    abplnk1_offset = 1 + 14 * (1 + abplnk1_dim2);
    abplnk1 -= abplnk1_offset;
    bch4_dim1 = 1;
    bch4_offset = 1 + bch4_dim1;
    bch4 -= bch4_offset;
    bn2o1_dim1 = 1;
    bn2o1_offset = 1 + bn2o1_dim1;
    bn2o1 -= bn2o1_offset;
    bn2o0_dim1 = 1;
    bn2o0_offset = 1 + bn2o0_dim1;
    bn2o0 -= bn2o0_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;
    piln_dim1 = 1;
    piln_offset = 1 + piln_dim1;
    piln -= piln_offset;
    plos_dim1 = 1;
    plos_offset = 1 + plos_dim1;
    plos -= plos_offset;
    plol_dim1 = 1;
    plol_offset = 1 + plol_dim1;
    plol -= plol_offset;
    tlayr_dim1 = 1;
    tlayr_offset = 1 + tlayr_dim1;
    tlayr -= tlayr_offset;
    tint_dim1 = 1;
    tint_offset = 1 + tint_dim1;
    tint -= tint_offset;
    co2t_dim1 = 1;
    co2t_offset = 1 + co2t_dim1;
    co2t -= co2t_offset;
    plh2o_dim1 = 1;
    plh2o_offset = 1 + plh2o_dim1;
    plh2o -= plh2o_offset;
    plco2_dim1 = 1;
    plco2_offset = 1 + plco2_dim1;
    plco2 -= plco2_offset;
    h2otr_dim1 = 1;
    h2otr_offset = 1 + h2otr_dim1;
    h2otr -= h2otr_offset;
    w_dim1 = 1;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    tcg_dim1 = 1;
    tcg_offset = 1 + tcg_dim1;
    tcg -= tcg_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;
    tplnka_dim1 = 1;
    tplnka_offset = 1 + tplnka_dim1;
    tplnka -= tplnka_offset;
    co2em_dim1 = 1;
    co2em_offset = 1 + co2em_dim1;
    co2em -= co2em_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;


    i__1 = state->cg_ntoplw__ - 1;
    for (k2 = 1; k2 <= i__1; ++k2) {
    i__2 = state->cg_ntoplw__ - 1;
    for (k1 = 1; k1 <= i__2; ++k1) {
        i__3 = 1 - 1;
        for (cam_s1_0__ = 0; cam_s1_0__ <= i__3; ++cam_s1_0__) {
        abstot[cam_s1_0__ + 1 + (k1 + k2 * abstot_dim2) * abstot_dim1]
             = state->cg_inf__;

        }


    }


    }

    for (k2 = 1; k2 <= 4; ++k2) {
    i__1 = state->cg_ntoplw__ - 1;
    for (k1 = 1; k1 <= i__1; ++k1) {
        i__2 = 1 - 1;
        for (cam_s2_0__ = 0; cam_s2_0__ <= i__2; ++cam_s2_0__) {
        absnxt[cam_s2_0__ + 1 + (k1 + k2 * absnxt_dim2) * absnxt_dim1]
             = state->cg_inf__;

        }


    }


    }

    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1 - 1;
    for (cam_s3_0__ = 0; cam_s3_0__ <= i__2; ++cam_s3_0__) {
        abstot[cam_s3_0__ + 1 + (k + k * abstot_dim2) * abstot_dim1] =
            state->cg_inf__;

    }


    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        dbvtly[i__ + k - 2] = (-2.89113668259233236e-4f + (
            2.37712515627208631e-6f + 1.13051887296045095e-10f *
            tlayr[i__ + (k + 1) * tlayr_dim1]) * tlayr[i__ + (k + 1) *
             tlayr_dim1]) / (1.f + (-.0061364821158349514f +
            1.5550320313195698e-5f * tlayr[i__ + (k + 1) * tlayr_dim1]
            ) * tlayr[i__ + (k + 1) * tlayr_dim1]);
        dbvtit[i__ + k - 2] = (-2.89113668259233236e-4f + (
            2.37712515627208631e-6f + 1.13051887296045095e-10f * tint[
            i__ + k * tint_dim1]) * tint[i__ + k * tint_dim1]) / (1.f
            + (-.0061364821158349514f + 1.5550320313195698e-5f * tint[
            i__ + k * tint_dim1]) * tint[i__ + k * tint_dim1]);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    dbvtit[i__ + (CAM_LEVELS+1) - 2] = (-2.89113668259233236e-4f + (
        2.37712515627208631e-6f + 1.13051887296045095e-10f * tint[i__
        + (CAM_LEVELS+1) * tint_dim1]) * tint[i__ + (CAM_LEVELS+1) * tint_dim1]) / (
        1.f + (-.0061364821158349514f + 1.5550320313195698e-5f * tint[
        i__ + (CAM_LEVELS+1) * tint_dim1]) * tint[i__ + (CAM_LEVELS+1) * tint_dim1]);

    }

    r293 = .0034129692832764505f;
    r250 = .0040000000000000001f;
    r3205 = 3.1201249372919126f;
    r300 = .0033333333333333335f;
    rsslp = 1.f / state->cg_sslp__;
    r2sslp = 1.f / (state->cg_sslp__ * 2.f);
    fdif = 1.65999996662139893f;
    sslp_mks__ = state->cg_sslp__ / 10.f;
    omeps = 1.f - state->cg_epsilo__;
    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {

        d__1 = pnm[i__ + k * pnm_dim1];
        pnmsq[i__ + k - 2] = d__1 * d__1;
        dtx[i__ - 1] = tplnka[i__ + k * tplnka_dim1] - 250.f;

    }


    }

    i__1 = state->cg_ntoplw__;
    for (k1 = (CAM_LEVELS+1); k1 >= i__1; --k1) {
    i__2 = state->cg_ntoplw__;
    for (k2 = (CAM_LEVELS+1); k2 >= i__2; --k2) {
        if (k1 == k2) {
        goto L9030;
        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        dplh2o[i__ - 1] = plh2o[i__ + k1 * plh2o_dim1] - plh2o[i__ +
            k2 * plh2o_dim1];
        u[i__ - 1] = (d__1 = dplh2o[i__ - 1], abs(d__1));
        sqrtu[i__ - 1] = sqrt(u[i__ - 1]);
        ds2c = (d__1 = s2c[i__ + k1 * s2c_dim1] - s2c[i__ + k2 *
            s2c_dim1], abs(d__1));
        dw[i__ - 1] = (d__1 = w[i__ + k1 * w_dim1] - w[i__ + k2 *
            w_dim1], abs(d__1));
        uc1[i__ - 1] = (ds2c + u[i__ - 1] * .00170000002253800631f) *
            (ds2c * 2.f + 1.f) / (ds2c * 15.f + 1.f);
        pch2o = ds2c;
        pnew[i__ - 1] = u[i__ - 1] / dw[i__ - 1];
        pnew_mks__ = pnew[i__ - 1] * sslp_mks__;
        tpatha = (d__1 = tcg[i__ + k1 * tcg_dim1] - tcg[i__ + k2 *
            tcg_dim1], abs(d__1)) / dw[i__ - 1];

        d__1 = max(tpatha,state->cg_min_tp_h2o__);
        t_p__ = min(d__1,state->cg_max_tp_h2o__);
        iest = (integer) (cam_floor__(&t_p__) -
            state->cg_min_tp_h2o__);
        esx = state->cg_estblh2o__[iest] + (
            state->cg_estblh2o__[iest + 1] -
            state->cg_estblh2o__[iest]) * (t_p__ -
            state->cg_min_tp_h2o__ - iest);
        qsx = state->cg_epsilo__ * esx / (pnew_mks__ - omeps *
            esx);
        q_path__ = dw[i__ - 1] / (d__1 = pnm[i__ + k1 * pnm_dim1] -
            pnm[i__ + k2 * pnm_dim1], abs(d__1)) /
            state->cg_rga__;
        ub[0] = (d__1 = plh2ob[(i__ + k1 * plh2ob_dim2 << 1) + 1] -
            plh2ob[(i__ + k2 * plh2ob_dim2 << 1) + 1], abs(d__1))
            / psi_(state, &t_p__, &radabs_c__1);
        ub[1] = (d__1 = plh2ob[(i__ + k1 * plh2ob_dim2 << 1) + 2] -
            plh2ob[(i__ + k2 * plh2ob_dim2 << 1) + 2], abs(d__1))
            / psi_(state, &t_p__, &radabs_c__2);
        pnewb[0] = ub[0] / (d__1 = wb[(i__ + k1 * wb_dim2 << 1) + 1]
            - wb[(i__ + k2 * wb_dim2 << 1) + 1], abs(d__1)) *
            phi_(state, &t_p__, &radabs_c__1);
        pnewb[1] = ub[1] / (d__1 = wb[(i__ + k1 * wb_dim2 << 1) + 2]
            - wb[(i__ + k2 * wb_dim2 << 1) + 2], abs(d__1)) *
            phi_(state, &t_p__, &radabs_c__2);
        dtx[i__ - 1] = tplnka[i__ + k2 * tplnka_dim1] - 250.f;
        dty[i__ - 1] = tpatha - 250.f;
        fwk[i__ - 1] = state->cg_fwcoef__ +
            state->cg_fwc1__ / (state->cg_fwc2__ * u[
            i__ - 1] + 1.f);
        fwku[i__ - 1] = fwk[i__ - 1] * u[i__ - 1];
        te1 = tplnka[i__ + k2 * tplnka_dim1];
        te2 = te1 * te1;
        te3 = te2 * te1;
        te4 = te3 * te1;
        te5 = te4 * te1;
        dvar = (t_p__ - state->cg_min_tp_h2o__) /
            state->cg_dtp_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        itp = min(i__4,9);
        itp1 = itp + 1;
        wtp = dvar - cam_floor__(&dvar);
        wtp1 = 1.f - wtp;


        d__2 = tplnka[i__ + k2 * tplnka_dim1] - t_p__;
        d__1 = max(d__2,state->cg_min_te_h2o__);
        t_e__ = min(d__1,state->cg_max_te_h2o__);
        dvar = (t_e__ - state->cg_min_te_h2o__) /
            state->cg_dte_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        ite = min(i__4,20);
        ite1 = ite + 1;
        wte = dvar - cam_floor__(&dvar);
        wte1 = 1.f - wte;


        d__2 = q_path__ / qsx;
        d__1 = max(d__2,state->cg_min_rh_h2o__);
        rh_path__ = min(d__1,state->cg_max_rh_h2o__);
        dvar = (rh_path__ - state->cg_min_rh_h2o__) /
            state->cg_drh_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        irh = min(i__4,6);
        irh1 = irh + 1;
        wrh = dvar - cam_floor__(&dvar);
        wrh1 = 1.f - wrh;
        w_0_0___ = wtp * wte;
        w_0_1___ = wtp * wte1;
        w_1_0___ = wtp1 * wte;
        w_1_1___ = wtp1 * wte1;
        w_0_00__ = w_0_0___ * wrh;
        w_0_01__ = w_0_0___ * wrh1;
        w_0_10__ = w_0_1___ * wrh;
        w_0_11__ = w_0_1___ * wrh1;
        w_1_00__ = w_1_0___ * wrh;
        w_1_01__ = w_1_0___ * wrh1;
        w_1_10__ = w_1_1___ * wrh;
        w_1_11__ = w_1_1___ * wrh1;
        fch2o = fh2oself_(state, &t_p__);
        uch2o = pch2o * state->cg_epsilo__ / (q_path__ * fch2o);
        ib = 1;
        uvar = ub[ib - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        iu = min(i__4,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnewb[ib - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        ip = min(i__4,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        a_star__ = state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ *
            wu + state->cg_ah2onw__[ip1 + (itp + (iu + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ *
            wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu1 + state->cg_ah2onw__[ip1 + (itp
            + (iu1 + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_11__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_10__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_01__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_11__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_10__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_01__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
             w00_00__ * wu1 + state->cg_ah2onw__[ip1 + (
            itp1 + (iu1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 -
            55111] * w00_10__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 -
            55111] * w00_01__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10
            - 55111] * w00_00__ * wu;


        d__2 = fa * (1.f - (1.f - a_star__) * aer_trn_ttl__[i__ + (k1
            + (k2 + ib * aer_trn_ttl_dim3) * aer_trn_ttl_dim2) *
            aer_trn_ttl_dim1]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }
        ib = 2;
        uvar = ub[ib - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        iu = min(i__4,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnewb[ib - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        ip = min(i__4,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;


        d__3 = uch2o * fdif;
        d__2 = max(d__3,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_uc__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_uc__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        iuc = min(i__4,24);
        iuc1 = iuc + 1;
        wuc = dvar - cam_floor__(&dvar);
        wuc1 = 1.f - wuc;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        l_star__ = state->cg_ln_ah2ow__[ip + (itp + (iu + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ *
            wu1 + state->cg_ln_ah2ow__[ip + (itp + (iu + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__
            * wu1 + state->cg_ln_ah2ow__[ip + (itp + (iu + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w11_01__
            * wu1 + state->cg_ln_ah2ow__[ip + (itp + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w11_00__ * wu1 + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w11_11__ * wu + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
             w11_10__ * wu + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
             w11_01__ * wu + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111]
            * w11_00__ * wu + state->cg_ln_ah2ow__[ip + (
            itp1 + (iu + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w10_11__ * wu1 + state->cg_ln_ah2ow__[
            ip + (itp1 + (iu + (ite + irh1 * 21) * 25) * 10) * 10
            - 55111] * w10_10__ * wu1 +
            state->cg_ln_ah2ow__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ln_ah2ow__[ip + (itp1 + (iu + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ *
            wu1 + state->cg_ln_ah2ow__[ip + (itp1 + (iu1 + (
            ite + irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ *
             wu + state->cg_ln_ah2ow__[ip + (itp1 + (iu1 + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__
            * wu + state->cg_ln_ah2ow__[ip + (itp1 + (iu1 +
            (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w10_01__ * wu + state->cg_ln_ah2ow__[ip + (itp1
            + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111]
            * w10_00__ * wu + state->cg_ln_ah2ow__[ip1 + (
            itp + (iu + (ite + irh * 21) * 25) * 10) * 10 - 55111]
             * w01_11__ * wu1 + state->cg_ln_ah2ow__[ip1 + (
            itp + (iu + (ite + irh1 * 21) * 25) * 10) * 10 -
            55111] * w01_10__ * wu1 + state->cg_ln_ah2ow__[
            ip1 + (itp + (iu + (ite1 + irh * 21) * 25) * 10) * 10
            - 55111] * w01_01__ * wu1 +
            state->cg_ln_ah2ow__[ip1 + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu1
            + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu
            + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 + (ite
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ *
            wu + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__
            * wu + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu + state->cg_ln_ah2ow__[ip1 + (
            itp1 + (iu + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu1 + state->cg_ln_ah2ow__[
            ip1 + (itp1 + (iu + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wu1 +
            state->cg_ln_ah2ow__[ip1 + (itp1 + (iu + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ *
            wu1 + state->cg_ln_ah2ow__[ip1 + (itp1 + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wu1 + state->cg_ln_ah2ow__[ip1 + (
            itp1 + (iu1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu + state->cg_ln_ah2ow__[
            ip1 + (itp1 + (iu1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wu +
            state->cg_ln_ah2ow__[ip1 + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu
            + state->cg_ln_ah2ow__[ip1 + (itp1 + (iu1 + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wu;
        c_star__ = state->cg_cn_ah2ow__[ip + (itp + (iuc + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ *
            wuc1 + state->cg_cn_ah2ow__[ip + (itp + (iuc + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__
            * wuc1 + state->cg_cn_ah2ow__[ip + (itp + (iuc
            + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w11_01__ * wuc1 + state->cg_cn_ah2ow__[ip + (
            itp + (iuc + (ite1 + irh1 * 21) * 25) * 10) * 10 -
            55111] * w11_00__ * wuc1 + state->cg_cn_ah2ow__[
            ip + (itp + (iuc1 + (ite + irh * 21) * 25) * 10) * 10
            - 55111] * w11_11__ * wuc +
            state->cg_cn_ah2ow__[ip + (itp + (iuc1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wuc
            + state->cg_cn_ah2ow__[ip + (itp + (iuc1 + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w11_01__
            * wuc + state->cg_cn_ah2ow__[ip + (itp + (iuc1
            + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w11_00__ * wuc + state->cg_cn_ah2ow__[ip + (
            itp1 + (iuc + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w10_11__ * wuc1 + state->cg_cn_ah2ow__[
            ip + (itp1 + (iuc + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w10_10__ * wuc1 +
            state->cg_cn_ah2ow__[ip + (itp1 + (iuc + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ *
            wuc1 + state->cg_cn_ah2ow__[ip + (itp1 + (iuc +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w10_00__ * wuc1 + state->cg_cn_ah2ow__[ip + (
            itp1 + (iuc1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w10_11__ * wuc + state->cg_cn_ah2ow__[
            ip + (itp1 + (iuc1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w10_10__ * wuc +
            state->cg_cn_ah2ow__[ip + (itp1 + (iuc1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ *
            wuc + state->cg_cn_ah2ow__[ip + (itp1 + (iuc1 +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w10_00__ * wuc + state->cg_cn_ah2ow__[ip1 + (
            itp + (iuc + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w01_11__ * wuc1 + state->cg_cn_ah2ow__[
            ip1 + (itp + (iuc + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w01_10__ * wuc1 +
            state->cg_cn_ah2ow__[ip1 + (itp + (iuc + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ *
            wuc1 + state->cg_cn_ah2ow__[ip1 + (itp + (iuc +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wuc1 + state->cg_cn_ah2ow__[ip1 + (
            itp + (iuc1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w01_11__ * wuc + state->cg_cn_ah2ow__[
            ip1 + (itp + (iuc1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w01_10__ * wuc +
            state->cg_cn_ah2ow__[ip1 + (itp + (iuc1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ *
            wuc + state->cg_cn_ah2ow__[ip1 + (itp + (iuc1 +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wuc + state->cg_cn_ah2ow__[ip1 + (
            itp1 + (iuc + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wuc1 + state->cg_cn_ah2ow__[
            ip1 + (itp1 + (iuc + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wuc1 +
            state->cg_cn_ah2ow__[ip1 + (itp1 + (iuc + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ *
            wuc1 + state->cg_cn_ah2ow__[ip1 + (itp1 + (iuc
            + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wuc1 + state->cg_cn_ah2ow__[ip1 + (
            itp1 + (iuc1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wuc + state->cg_cn_ah2ow__[
            ip1 + (itp1 + (iuc1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wuc +
            state->cg_cn_ah2ow__[ip1 + (itp1 + (iuc1 + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__
            * wuc + state->cg_cn_ah2ow__[ip1 + (itp1 + (
            iuc1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wuc;


        d__2 = fa * (1.f - l_star__ * c_star__ * aer_trn_ttl__[i__ + (
            k1 + (k2 + ib * aer_trn_ttl_dim3) * aer_trn_ttl_dim2)
            * aer_trn_ttl_dim1]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }

        }

        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        term7[i__ - 1] = state->cg_coefj__[0] +
            state->cg_coefj__[1] * dty[i__ - 1] * (
            state->cg_c16__ * dty[i__ - 1] + 1.f);
        term8[i__ - 1] = state->cg_coefk__[0] +
            state->cg_coefk__[1] * dty[i__ - 1] * (
            state->cg_c17__ * dty[i__ - 1] + 1.f);
        term7[i__] = state->cg_coefj__[3] +
            state->cg_coefj__[4] * dty[i__ - 1] * (
            state->cg_c26__ * dty[i__ - 1] + 1.f);
        term8[i__] = state->cg_coefk__[3] +
            state->cg_coefk__[4] * dty[i__ - 1] * (
            state->cg_c27__ * dty[i__ - 1] + 1.f);

        }

        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        k21 = term7[i__ - 1] + term8[i__ - 1] / ((
            state->cg_c30__ + state->cg_c31__ * (dty[
            i__ - 1] - 10.f) * (dty[i__ - 1] - 10.f)) * sqrtu[i__
            - 1] + 1.f);
        k22 = term7[i__] + term8[i__] / ((state->cg_c28__ +
            state->cg_c29__ * (dty[i__ - 1] - 10.f)) *
            sqrtu[i__ - 1] + 1.f);
        tr1 = cam_exp64(-(k21 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr2 = cam_exp64(-(k22 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr1 *= aer_trn_ttl__[i__ + (k1 + (k2 + (aer_trn_ttl_dim3 << 2)
            ) * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
        tr2 *= aer_trn_ttl__[i__ + (k1 + (k2 + aer_trn_ttl_dim3 * 3) *
             aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
        tr5 = cam_exp64(-((state->cg_coefh__[4] +
            state->cg_coefh__[5] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr6 = cam_exp64(-((state->cg_coefh__[6] +
            state->cg_coefh__[7] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr9[i__ - 1] = tr1 * tr5;
        tr10[i__ - 1] = tr2 * tr6;
        th2o[i__ - 1] = tr10[i__ - 1];
        trab2[i__ - 1] = tr9[i__ - 1] * .64999997615814209f + tr10[
            i__ - 1] * .349999994039535522f;

        }

        if (k2 < k1) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            to3h2o[i__ - 1] = h2otr[i__ + k1 * h2otr_dim1] / h2otr[
                i__ + k2 * h2otr_dim1];

        }

        } else {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            to3h2o[i__ - 1] = h2otr[i__ + k2 * h2otr_dim1] / h2otr[
                i__ + k1 * h2otr_dim1];

        }

        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        dpnm[i__ - 1] = pnm[i__ + k1 * pnm_dim1] - pnm[i__ + k2 *
            pnm_dim1];
        to3co2[i__ - 1] = (pnm[i__ + k1 * pnm_dim1] * co2t[i__ + k1 *
            co2t_dim1] - pnm[i__ + k2 * pnm_dim1] * co2t[i__ + k2
            * co2t_dim1]) / dpnm[i__ - 1];
        d__1 = to3co2[i__ - 1] * r293;
        te = pow_dd(&d__1, &radabs_c_b43);
        dplos = plos[i__ + k1 * plos_dim1] - plos[i__ + k2 *
            plos_dim1];
        dplol = plol[i__ + k1 * plol_dim1] - plol[i__ + k2 *
            plol_dim1];
        u1 = abs(dplos) * 18.2900009155273438f / te;
        u2 = abs(dplos) * .564899981021881104f / te;
        rphat = dplol / dplos;
        tlocal = tint[i__ + k2 * tint_dim1];
        tcrfac = sqrt(tlocal * r250) * te;
        beta = r3205 * (rphat + state->cg_dpfo3__ * tcrfac);
        realnu = te / beta;
        tmp1 = u1 / sqrt(u1 * (realnu + 1.f) + 4.f);
        tmp2 = u2 / sqrt(u2 * (realnu + 1.f) + 4.f);
        o3bndi = te * 74.f * cam_log64(tmp1 + 1.f + tmp2);
        abso[i__ + 1] = o3bndi * to3h2o[i__ - 1] * dbvtit[i__ + k2 -
            2];
        to3[i__ - 1] = 1.f / (tmp1 * .100000001490116119f + 1.f +
            tmp2 * .100000001490116119f);

        }

        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        sqwp = sqrt((d__1 = plco2[i__ + k1 * plco2_dim1] - plco2[i__
            + k2 * plco2_dim1], abs(d__1)));
        et = cam_exp64(-480.f / to3co2[i__ - 1]);
        sqti[i__ - 1] = sqrt(to3co2[i__ - 1]);
        rsqti = 1.f / sqti[i__ - 1];
        et2 = et * et;
        et4 = et2 * et2;
        omet = 1.f - et2 * 1.5f;
        f1co2 = omet * 899.70001220703125f * (et *
            1.94773995876312256f + 1.f + et2 *
            4.73485994338989258f) * rsqti;
        f1sqwp[i__ - 1] = f1co2 * sqwp;
        t1co2[i__ - 1] = 1.f / (omet * 245.17999267578125f * sqwp *
            rsqti + 1.f);
        oneme = 1.f - et2;

        d__1 = oneme;
        alphat = d__1 * (d__1 * d__1) * rsqti;
        pi = (d__1 = dpnm[i__ - 1], abs(d__1));
        wco2 = state->cg_co2vmr__ * 2.52209997177124023f * pi *
            state->cg_rga__;
        u7[i__ - 1] = alphat * 49411.f * et2 * wco2;
        u8 = alphat * 39744.f * et4 * wco2;
        u9 = alphat * 104470.f * et4 * et2 * wco2;
        u13 = alphat * 2838.800048828125f * et4 * wco2;
        tpath = to3co2[i__ - 1];
        tlocal = tint[i__ + k2 * tint_dim1];
        tcrfac = sqrt(tlocal * r250 * tpath * r300);
        posqt = ((pnm[i__ + k2 * pnm_dim1] + pnm[i__ + k1 * pnm_dim1])
             * r2sslp + state->cg_dpfco2__ * tcrfac) *
            rsqti;
        rbeta7[i__ - 1] = 1.f / (posqt * 5.3228001594543457f);
        rbeta8 = 1.f / (posqt * 10.6576004028320312f);
        rbeta9 = rbeta7[i__ - 1];
        rbeta13 = rbeta9;
        f2co2[i__ - 1] = u7[i__ - 1] / sqrt(u7[i__ - 1] * (rbeta7[i__
            - 1] + 1.f) + 4.f) + u8 / sqrt(u8 * (rbeta8 + 1.f) +
            4.f) + u9 / sqrt(u9 * (rbeta9 + 1.f) + 4.f);
        f3co2[i__ - 1] = u13 / sqrt(u13 * (rbeta13 + 1.f) + 4.f);

        }

        if (k2 >= k1) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            sqti[i__ - 1] = sqrt(tlayr[i__ + k2 * tlayr_dim1]);

        }

        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        tmp1 = cam_log64(f1sqwp[i__ - 1] + 1.f);
        tmp2 = cam_log64(f2co2[i__ - 1] + 1.f);
        tmp3 = cam_log64(f3co2[i__ - 1] + 1.f);
        absbnd = (tmp1 + t1co2[i__ - 1] * 2.f * tmp2 + tmp3 * 2.f) *
            sqti[i__ - 1];
        abso[i__ + 2] = trab2[i__ - 1] * co2em[i__ + k2 * co2em_dim1]
            * absbnd;
        tco2[i__ - 1] = 1.f / (u7[i__ - 1] / sqrt(u7[i__ - 1] * (
            rbeta7[i__ - 1] + 1.f) + 4.f) * 10.f + 1.f);

        }

        trcab_(state, lchnk, ncol, pcols, pverp, &k1, &k2, &ucfc11[ucfc11_offset]
            , &ucfc12[ucfc12_offset], &un2o0[un2o0_offset], &un2o1[
            un2o1_offset], &uch4[uch4_offset], &uco211[uco211_offset],
             &uco212[uco212_offset], &uco213[uco213_offset], &uco221[
            uco221_offset], &uco222[uco222_offset], &uco223[
            uco223_offset], &bn2o0[bn2o0_offset], &bn2o1[bn2o1_offset]
            , &bch4[bch4_offset], to3co2, &pnm[pnm_offset], dw, pnew,
            &s2c[s2c_offset], &uptype[uptype_offset], u, &abplnk1[
            abplnk1_offset], tco2, th2o, to3, abstrc, &aer_trn_ttl__[
            aer_trn_ttl_offset]);
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        abstot[i__ + (k1 + k2 * abstot_dim2) * abstot_dim1] = abso[
            i__ - 1] + abso[i__] + abso[i__ + 1] + abso[i__ + 2]
            + abstrc[i__ - 1];

        }

L9030:
        ;
    }


    }

    i__1 = state->cg_ntoplw__;
    for (k2 = CAM_LEVELS; k2 >= i__1; --k2) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        tbar[i__ - 1] = (tint[i__ + (k2 + 1) * tint_dim1] + tlayr[i__ + (
            k2 + 1) * tlayr_dim1]) * .5f;
        emm[i__ - 1] = (co2em[i__ + (k2 + 1) * co2em_dim1] + co2eml[i__ +
            k2 * co2eml_dim1]) * .5f;
        tbar[i__] = (tlayr[i__ + (k2 + 1) * tlayr_dim1] + tint[i__ + k2 *
            tint_dim1]) * .5f;
        emm[i__] = (co2em[i__ + k2 * co2em_dim1] + co2eml[i__ + k2 *
            co2eml_dim1]) * .5f;
        tbar[i__ + 1] = (tbar[i__] + tbar[i__ - 1]) * .5f;
        emm[i__ + 1] = emm[i__ - 1];
        tbar[i__ + 2] = tbar[i__ + 1];
        emm[i__ + 2] = emm[i__];
        o3emm[i__ - 1] = (dbvtit[i__ + (k2 + 1) - 2] + dbvtly[i__ + k2 -
            2]) * .5f;
        o3emm[i__] = (dbvtit[i__ + k2 - 2] + dbvtly[i__ + k2 - 2]) * .5f;
        o3emm[i__ + 1] = o3emm[i__ - 1];
        o3emm[i__ + 2] = o3emm[i__];
        temh2o[i__ - 1] = tbar[i__ - 1];
        temh2o[i__] = tbar[i__];
        temh2o[i__ + 1] = tbar[i__ - 1];
        temh2o[i__ + 2] = tbar[i__];
        dpnm[i__ - 1] = pnm[i__ + (k2 + 1) * pnm_dim1] - pnm[i__ + k2 *
            pnm_dim1];

    }

    for (wvl = 1; wvl <= 14; ++wvl) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        bplnk[wvl + (i__ + 1) * 14 - 29] = (abplnk1[wvl + (i__ + (k2
            + 1) * abplnk1_dim2) * 14] + abplnk2[wvl + (i__ + k2 *
             abplnk2_dim2) * 14]) * .5f;
        bplnk[wvl + (i__ + 2) * 14 - 29] = (abplnk1[wvl + (i__ + k2 *
            abplnk1_dim2) * 14] + abplnk2[wvl + (i__ + k2 *
            abplnk2_dim2) * 14]) * .5f;
        bplnk[wvl + (i__ + 3) * 14 - 29] = bplnk[wvl + (i__ + 1) * 14
            - 29];
        bplnk[wvl + (i__ + 4) * 14 - 29] = bplnk[wvl + (i__ + 2) * 14
            - 29];

        }


    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        rdpnmsq = 1.f / (pnmsq[i__ + (k2 + 1) - 2] - pnmsq[i__ + k2 - 2]);
        rdpnm = 1.f / dpnm[i__ - 1];
        p1 = (pbr[i__ + k2 * pbr_dim1] + pnm[i__ + (k2 + 1) * pnm_dim1]) *
             .5f;
        p2 = (pbr[i__ + k2 * pbr_dim1] + pnm[i__ + k2 * pnm_dim1]) * .5f;

        d__1 = p1;
        uinpl[i__ - 1] = (pnmsq[i__ + (k2 + 1) - 2] - d__1 * d__1) *
            rdpnmsq;

        d__1 = p2;
        uinpl[i__] = -(pnmsq[i__ + k2 - 2] - d__1 * d__1) * rdpnmsq;

        d__1 = p1;
        uinpl[i__ + 1] = -(pnmsq[i__ + k2 - 2] - d__1 * d__1) * rdpnmsq;

        d__1 = p2;
        uinpl[i__ + 2] = (pnmsq[i__ + (k2 + 1) - 2] - d__1 * d__1) *
            rdpnmsq;
        winpl[i__ - 1] = (pnm[i__ + (k2 + 1) * pnm_dim1] - pbr[i__ + k2 *
            pbr_dim1]) * .5f * rdpnm;
        winpl[i__] = (-pnm[i__ + k2 * pnm_dim1] + pbr[i__ + k2 * pbr_dim1]
            ) * .5f * rdpnm;
        winpl[i__ + 1] = ((pnm[i__ + (k2 + 1) * pnm_dim1] + pbr[i__ + k2 *
             pbr_dim1]) * .5f - pnm[i__ + k2 * pnm_dim1]) * rdpnm;
        winpl[i__ + 2] = ((-pnm[i__ + k2 * pnm_dim1] - pbr[i__ + k2 *
            pbr_dim1]) * .5f + pnm[i__ + (k2 + 1) * pnm_dim1]) *
            rdpnm;
        tmp1 = 1.f / (piln[i__ + (k2 + 1) * piln_dim1] - piln[i__ + k2 *
            piln_dim1]);
        tmp2 = piln[i__ + (k2 + 1) * piln_dim1] - pmln[i__ + k2 *
            pmln_dim1];
        tmp3 = piln[i__ + k2 * piln_dim1] - pmln[i__ + k2 * pmln_dim1];
        zinpl[i__ - 1] = tmp2 * .5f * tmp1;
        zinpl[i__] = tmp3 * -.5f * tmp1;
        zinpl[i__ + 1] = (tmp2 * .5f - tmp3) * tmp1;
        zinpl[i__ + 2] = (tmp2 - tmp3 * .5f) * tmp1;
        pinpl[i__ - 1] = (p1 + pnm[i__ + (k2 + 1) * pnm_dim1]) * .5f;
        pinpl[i__] = (p2 + pnm[i__ + k2 * pnm_dim1]) * .5f;
        pinpl[i__ + 1] = (p1 + pnm[i__ + k2 * pnm_dim1]) * .5f;
        pinpl[i__ + 2] = (p2 + pnm[i__ + (k2 + 1) * pnm_dim1]) * .5f;
        if (state->cg_strat_volcanic__) {
        aer_pth_ngh__[i__ - 1] = (d__1 = aer_mpp__[i__ + k2 *
            aer_mpp_dim1] - aer_mpp__[i__ + (k2 + 1) *
            aer_mpp_dim1], abs(d__1));
        }

    }

    for (kn = 1; kn <= 4; ++kn) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        u[i__ - 1] = uinpl[i__ + kn - 2] * (d__1 = plh2o[i__ + k2 *
            plh2o_dim1] - plh2o[i__ + (k2 + 1) * plh2o_dim1], abs(
            d__1));
        sqrtu[i__ - 1] = sqrt(u[i__ - 1]);
        dw[i__ - 1] = (d__1 = w[i__ + k2 * w_dim1] - w[i__ + (k2 + 1)
            * w_dim1], abs(d__1));
        pnew[i__ - 1] = u[i__ - 1] / (winpl[i__ + kn - 2] * dw[i__ -
            1]);
        pnew_mks__ = pnew[i__ - 1] * sslp_mks__;


        d__2 = tbar[i__ + kn - 2];
        d__1 = max(d__2,state->cg_min_tp_h2o__);
        t_p__ = min(d__1,state->cg_max_tp_h2o__);
        iest = (integer) (cam_floor__(&t_p__) -
            state->cg_min_tp_h2o__);
        esx = state->cg_estblh2o__[iest] + (
            state->cg_estblh2o__[iest + 1] -
            state->cg_estblh2o__[iest]) * (t_p__ -
            state->cg_min_tp_h2o__ - iest);
        qsx = state->cg_epsilo__ * esx / (pnew_mks__ - omeps *
            esx);
        q_path__ = dw[i__ - 1] / (d__1 = dpnm[i__ - 1], abs(d__1)) /
            state->cg_rga__;
        ds2c = (d__1 = s2c[i__ + k2 * s2c_dim1] - s2c[i__ + (k2 + 1) *
             s2c_dim1], abs(d__1));
        uc1[i__ - 1] = uinpl[i__ + kn - 2] * ds2c;
        pch2o = uc1[i__ - 1];
        uc1[i__ - 1] = (uc1[i__ - 1] + u[i__ - 1] *
            .00170000002253800631f) * (uc1[i__ - 1] * 2.f + 1.f) /
             (uc1[i__ - 1] * 15.f + 1.f);
        dtx[i__ - 1] = temh2o[i__ + kn - 2] - 250.f;
        dty[i__ - 1] = tbar[i__ + kn - 2] - 250.f;
        fwk[i__ - 1] = state->cg_fwcoef__ +
            state->cg_fwc1__ / (state->cg_fwc2__ * u[
            i__ - 1] + 1.f);
        fwku[i__ - 1] = fwk[i__ - 1] * u[i__ - 1];
        if (state->cg_strat_volcanic__) {
            aer_pth_dlt__ = uinpl[i__ + kn - 2] * aer_pth_ngh__[i__ -
                1];
            for (bnd_idx__ = 1; bnd_idx__ <= 7; ++bnd_idx__) {
            odap_aer_ttl__ = state->cg_abs_cff_mss_aer__[
                bnd_idx__ - 1] * aer_pth_dlt__;
            aer_trn_ngh__[i__ + bnd_idx__ - 2] = cam_exp64(-fdif *
                odap_aer_ttl__);

            }

        } else {
            for (cam_s4_1__ = 0; cam_s4_1__ <= 6; ++cam_s4_1__) {
            aer_trn_ngh__[i__ + (cam_s4_1__ + 1) - 2] = 1.f;

            }

        }
        te1 = temh2o[i__ + kn - 2];
        te2 = te1 * te1;
        te3 = te2 * te1;
        te4 = te3 * te1;
        te5 = te4 * te1;
        uvar = u[i__ - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        iu = min(i__3,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnew[i__ - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ip = min(i__3,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        dvar = (t_p__ - state->cg_min_tp_h2o__) /
            state->cg_dtp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        itp = min(i__3,9);
        itp1 = itp + 1;
        wtp = dvar - cam_floor__(&dvar);
        wtp1 = 1.f - wtp;


        d__2 = temh2o[i__ + kn - 2] - t_p__;
        d__1 = max(d__2,state->cg_min_te_h2o__);
        t_e__ = min(d__1,state->cg_max_te_h2o__);
        dvar = (t_e__ - state->cg_min_te_h2o__) /
            state->cg_dte_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ite = min(i__3,20);
        ite1 = ite + 1;
        wte = dvar - cam_floor__(&dvar);
        wte1 = 1.f - wte;


        d__2 = q_path__ / qsx;
        d__1 = max(d__2,state->cg_min_rh_h2o__);
        rh_path__ = min(d__1,state->cg_max_rh_h2o__);
        dvar = (rh_path__ - state->cg_min_rh_h2o__) /
            state->cg_drh_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        irh = min(i__3,6);
        irh1 = irh + 1;
        wrh = dvar - cam_floor__(&dvar);
        wrh1 = 1.f - wrh;
        w_0_0___ = wtp * wte;
        w_0_1___ = wtp * wte1;
        w_1_0___ = wtp1 * wte;
        w_1_1___ = wtp1 * wte1;
        w_0_00__ = w_0_0___ * wrh;
        w_0_01__ = w_0_0___ * wrh1;
        w_0_10__ = w_0_1___ * wrh;
        w_0_11__ = w_0_1___ * wrh1;
        w_1_00__ = w_1_0___ * wrh;
        w_1_01__ = w_1_0___ * wrh1;
        w_1_10__ = w_1_1___ * wrh;
        w_1_11__ = w_1_1___ * wrh1;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;
        ib = 1;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        a_star__ = state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ *
            wu + state->cg_ah2onw__[ip1 + (itp + (iu + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ *
            wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu1 + state->cg_ah2onw__[ip1 + (itp
            + (iu1 + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_11__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_10__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_01__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_11__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_10__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_01__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
             w00_00__ * wu1 + state->cg_ah2onw__[ip1 + (
            itp1 + (iu1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 -
            55111] * w00_10__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 -
            55111] * w00_01__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10
            - 55111] * w00_00__ * wu;


        d__2 = fa * (1.f - (1.f - a_star__) * aer_trn_ngh__[i__ + ib
            - 2]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }
        ib = 2;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        a_star__ = state->cg_ah2ow__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ah2ow__[ip + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu
            + state->cg_ah2ow__[ip + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ah2ow__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu
            + state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu +
            state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu +
            state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wu
            + state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wu +
            state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wu +
            state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu
            + state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu
            + state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ *
            wu;


        d__2 = fa * (1.f - (1.f - a_star__) * aer_trn_ngh__[i__ + ib
            - 2]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        term7[i__ - 1] = state->cg_coefj__[0] +
            state->cg_coefj__[1] * dty[i__ - 1] * (
            state->cg_c16__ * dty[i__ - 1] + 1.f);
        term8[i__ - 1] = state->cg_coefk__[0] +
            state->cg_coefk__[1] * dty[i__ - 1] * (
            state->cg_c17__ * dty[i__ - 1] + 1.f);
        term7[i__] = state->cg_coefj__[3] +
            state->cg_coefj__[4] * dty[i__ - 1] * (
            state->cg_c26__ * dty[i__ - 1] + 1.f);
        term8[i__] = state->cg_coefk__[3] +
            state->cg_coefk__[4] * dty[i__ - 1] * (
            state->cg_c27__ * dty[i__ - 1] + 1.f);

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        dtym10 = dty[i__ - 1] - 10.f;
        denom = (state->cg_c30__ + state->cg_c31__ *
            dtym10 * dtym10) * sqrtu[i__ - 1] + 1.f;
        k21 = term7[i__ - 1] + term8[i__ - 1] / denom;
        denom = (state->cg_c28__ + state->cg_c29__ *
            dtym10) * sqrtu[i__ - 1] + 1.f;
        k22 = term7[i__] + term8[i__] / denom;
        tr1 = cam_exp64(-(k21 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr2 = cam_exp64(-(k22 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr1 *= aer_trn_ngh__[i__ + 2];
        tr2 *= aer_trn_ngh__[i__ + 1];
        tr5 = cam_exp64(-((state->cg_coefh__[4] +
            state->cg_coefh__[5] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr6 = cam_exp64(-((state->cg_coefh__[6] +
            state->cg_coefh__[7] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr9[i__ - 1] = tr1 * tr5;
        tr10[i__ - 1] = tr2 * tr6;
        trab2[i__ - 1] = tr9[i__ - 1] * .64999997615814209f + tr10[
            i__ - 1] * .349999994039535522f;
        th2o[i__ - 1] = tr10[i__ - 1];

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        d__1 = tbar[i__ + kn - 2] * r293;
        te = pow_dd(&d__1, &radabs_c_b43);
        dplos = (d__1 = plos[i__ + (k2 + 1) * plos_dim1] - plos[i__ +
            k2 * plos_dim1], abs(d__1));
        u1 = zinpl[i__ + kn - 2] * 18.2900009155273438f * dplos / te;
        u2 = zinpl[i__ + kn - 2] * .564899981021881104f * dplos / te;
        tlocal = tbar[i__ + kn - 2];
        tcrfac = sqrt(tlocal * r250) * te;
        beta = r3205 * (pinpl[i__ + kn - 2] * rsslp +
            state->cg_dpfo3__ * tcrfac);
        realnu = te / beta;
        tmp1 = u1 / sqrt(u1 * (realnu + 1.f) + 4.f);
        tmp2 = u2 / sqrt(u2 * (realnu + 1.f) + 4.f);
        o3bndi = te * 74.f * cam_log64(tmp1 + 1.f + tmp2);
        abso[i__ + 1] = o3bndi * o3emm[i__ + kn - 2] * (h2otr[i__ + (
            k2 + 1) * h2otr_dim1] / h2otr[i__ + k2 * h2otr_dim1]);
        to3[i__ - 1] = 1.f / (tmp1 * .100000001490116119f + 1.f +
            tmp2 * .100000001490116119f);

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        dplco2 = plco2[i__ + (k2 + 1) * plco2_dim1] - plco2[i__ + k2 *
             plco2_dim1];
        sqwp = sqrt(uinpl[i__ + kn - 2] * dplco2);
        et = cam_exp64(-480.f / tbar[i__ + kn - 2]);
        sqti[i__ - 1] = sqrt(tbar[i__ + kn - 2]);
        rsqti = 1.f / sqti[i__ - 1];
        et2 = et * et;
        et4 = et2 * et2;
        omet = 1.f - et2 * 1.5f;
        f1co2 = omet * 899.70001220703125f * (et *
            1.94773995876312256f + 1.f + et2 *
            4.73485994338989258f) * rsqti;
        f1sqwp[i__ - 1] = f1co2 * sqwp;
        t1co2[i__ - 1] = 1.f / (omet * 245.17999267578125f * sqwp *
            rsqti + 1.f);
        oneme = 1.f - et2;

        d__1 = oneme;
        alphat = d__1 * (d__1 * d__1) * rsqti;
        pi = (d__1 = dpnm[i__ - 1], abs(d__1)) * winpl[i__ + kn - 2];
        wco2 = state->cg_co2vmr__ * 2.52209997177124023f * pi *
            state->cg_rga__;
        u7[i__ - 1] = alphat * 49411.f * et2 * wco2;
        u8 = alphat * 39744.f * et4 * wco2;
        u9 = alphat * 104470.f * et4 * et2 * wco2;
        u13 = alphat * 2838.800048828125f * et4 * wco2;
        tpath = tbar[i__ + kn - 2];
        tlocal = tbar[i__ + kn - 2];
        tcrfac = sqrt(tlocal * r250 * (tpath * r300));
        posqt = (pinpl[i__ + kn - 2] * rsslp +
            state->cg_dpfco2__ * tcrfac) * rsqti;
        rbeta7[i__ - 1] = 1.f / (posqt * 5.3228001594543457f);
        rbeta8 = 1.f / (posqt * 10.6576004028320312f);
        rbeta9 = rbeta7[i__ - 1];
        rbeta13 = rbeta9;
        f2co2[i__ - 1] = u7[i__ - 1] / sqrt(u7[i__ - 1] * (rbeta7[i__
            - 1] + 1.f) + 4.f) + u8 / sqrt(u8 * (rbeta8 + 1.f) +
            4.f) + u9 / sqrt(u9 * (rbeta9 + 1.f) + 4.f);
        f3co2[i__ - 1] = u13 / sqrt(u13 * (rbeta13 + 1.f) + 4.f);
        tmp1 = cam_log64(f1sqwp[i__ - 1] + 1.f);
        tmp2 = cam_log64(f2co2[i__ - 1] + 1.f);
        tmp3 = cam_log64(f3co2[i__ - 1] + 1.f);
        absbnd = (tmp1 + t1co2[i__ - 1] * 2.f * tmp2 + tmp3 * 2.f) *
            sqti[i__ - 1];
        abso[i__ + 2] = trab2[i__ - 1] * emm[i__ + kn - 2] * absbnd;
        tco2[i__ - 1] = 1.f / (u7[i__ - 1] * 10.f / sqrt(u7[i__ - 1] *
             (rbeta7[i__ - 1] + 1.f) + 4.f) + 1.f);

        }

        trcabn_(state, lchnk, ncol, pcols, pverp, &k2, &kn, &ucfc11[
            ucfc11_offset], &ucfc12[ucfc12_offset], &un2o0[
            un2o0_offset], &un2o1[un2o1_offset], &uch4[uch4_offset], &
            uco211[uco211_offset], &uco212[uco212_offset], &uco213[
            uco213_offset], &uco221[uco221_offset], &uco222[
            uco222_offset], &uco223[uco223_offset], tbar, bplnk,
            winpl, pinpl, tco2, th2o, to3, &uptype[uptype_offset], dw,
             &s2c[s2c_offset], u, pnew, abstrc, uinpl, aer_trn_ngh__);
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        absnxt[i__ + (k2 + kn * absnxt_dim2) * absnxt_dim1] = abso[
            i__ - 1] + abso[i__] + abso[i__ + 1] + abso[i__ + 2]
            + abstrc[i__ - 1];

        }


    }


    }

    return 0;
}__device__ int radabs_parallel_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *pbr, doublereal *pnm,
    doublereal *co2em, doublereal *co2eml, doublereal *tplnka, doublereal
    *s2c, doublereal *tcg, doublereal *w, doublereal *h2otr, doublereal *
    plco2, doublereal *plh2o, doublereal *co2t, doublereal *tint,
    doublereal *tlayr, doublereal *plol, doublereal *plos, doublereal *
    pmln, doublereal *piln, doublereal *ucfc11, doublereal *ucfc12,
    doublereal *un2o0, doublereal *un2o1, doublereal *uch4, doublereal *
    uco211, doublereal *uco212, doublereal *uco213, doublereal *uco221,
    doublereal *uco222, doublereal *uco223, doublereal *uptype,
    doublereal *bn2o0, doublereal *bn2o1, doublereal *bch4, doublereal *
    abplnk1, doublereal *abplnk2, doublereal *abstot, doublereal *absnxt,
    doublereal *plh2ob, doublereal *wb, doublereal *aer_mpp__, doublereal
    *aer_trn_ttl__)
{
 CamScratchFrame scratch_frame(state);

    integer pbr_dim1 = 0,pbr_offset = 0,pnm_dim1 = 0,pnm_offset = 0,co2em_dim1 = 0,co2em_offset = 0,co2eml_dim1 = 0,co2eml_offset = 0,tplnka_dim1 = 0,tplnka_offset = 0,s2c_dim1 = 0,s2c_offset = 0,tcg_dim1 = 0,tcg_offset = 0,w_dim1 = 0,w_offset = 0,h2otr_dim1 = 0,h2otr_offset = 0,plco2_dim1 = 0,plco2_offset = 0,plh2o_dim1 = 0,plh2o_offset = 0,co2t_dim1 = 0,co2t_offset = 0,tint_dim1 = 0,tint_offset = 0,tlayr_dim1 = 0,tlayr_offset = 0,plol_dim1 = 0,plol_offset = 0,plos_dim1 = 0,plos_offset = 0,pmln_dim1 = 0,pmln_offset = 0,piln_dim1 = 0,piln_offset = 0,plh2ob_dim2 = 0,plh2ob_offset = 0,wb_dim2 = 0,wb_offset = 0,aer_mpp_dim1 = 0,aer_mpp_offset = 0,aer_trn_ttl_dim1 = 0,aer_trn_ttl_dim2 = 0,aer_trn_ttl_dim3 = 0,aer_trn_ttl_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,uptype_dim1 = 0,uptype_offset = 0,bn2o0_dim1 = 0,bn2o0_offset = 0,bn2o1_dim1 = 0,bn2o1_offset = 0,bch4_dim1 = 0,bch4_offset = 0,abplnk1_dim2 = 0,abplnk1_offset = 0,abplnk2_dim2 = 0,abplnk2_offset = 0,abstot_dim1 = 0,abstot_dim2 = 0,abstot_offset = 0,absnxt_dim1 = 0,absnxt_dim2 = 0,absnxt_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0,i__5 = 0;

    doublereal d__1 = 0,d__2 = 0,d__3 = 0;

    doublereal pnew_mks__ = 0,sslp_mks__ = 0;

    integer i__ = 0,k = 0;

    doublereal u[1] = {};

    integer k1 = 0,k2 = 0;

    doublereal u1 = 0,u2 = 0,p1 = 0,p2 = 0,u7[1] = {},u8 = 0,u9 = 0,fa = 0;

    integer ib = 0;

    doublereal k21 = 0,k22 = 0,ub[2] = {};

    integer kn = 0;

    doublereal pi = 0,et = 0,u13 = 0,dw[1] = {},te = 0;

    integer iu = 0,ip = 0;

    doublereal wp = 0,wu = 0,r300 = 0,uc1[1] = {},et2 = 0,et4 = 0,r293 = 0,r250 = 0,te1 = 0,te2 = 0,te3 = 0,to3[1] = {},tr1 = 0,tr2 = 0,te4 = 0,tr5 = 0,tr6 = 0,te5 = 0,tr9[1] = {};

    integer iu1 = 0,ip1 = 0;

    doublereal wu1 = 0,wp1 = 0,t_e__ = 0,r3205 = 0,emm[4] = {};

    integer iuc = 0;

    doublereal t_p__ = 0;

    integer ite = 0,irh = 0;

    doublereal tr10[1] = {},fwk[1] = {};

    integer itp = 0;

    doublereal dtx[1] = {},dty[1] = {},wuc = 0,wte = 0,wrh = 0,esx = 0;

    integer wvl = 0;

    doublereal wtp = 0,qsx = 0,aer_pth_ngh__[1] = {};

    doublereal aer_pth_dlt__ = 0,aer_trn_ngh__[7] = {},ds2c = 0;

    integer iuc1 = 0,ite1 = 0,irh1 = 0;

    doublereal tco2[1] = {},wco2 = 0,th2o[1] = {};

    integer itp1 = 0;

    doublereal wuc1 = 0,wte1 = 0,tmp1 = 0,tmp2 = 0,tmp3 = 0,wrh1 = 0,wtp1 = 0,w00_00__ = 0,w00_01__ = 0,w00_10__ = 0,w00_11__ = 0,f1co2 = 0,f2co2[1] = {},f3co2[1] = {},beta = 0,w01_00__ = 0,w01_01__ = 0,w01_10__ = 0,w01_11__ = 0,abso[4] = {},tbar[4] = {},t1co2[1] = {},dpnm[1] = {},dvar = 0,pnew[1] = {},omet = 0,fwku[1] = {},sqti[1] = {},uvar = 0,uscl = 0,w10_00__ = 0,w_0_00__ = 0,w_0_01__ = 0,w_0_10__ = 0,w_0_11__ = 0,w_1_00__ = 0,sqwp = 0,w_1_01__ = 0,w_1_10__ = 0,w_1_11__ = 0,w10_01__ = 0,w10_10__ = 0,w10_11__ = 0,w11_00__ = 0,w11_01__ = 0,w11_10__ = 0,w11_11__ = 0,fch2o = 0,fdif = 0;

    integer iest = 0;

    doublereal o3emm[4] = {},trab2[1] = {},pch2o = 0,uch2o = 0,odap_aer_ttl__ = 0,term7[2] = {},term8[2] = {},w_0_0___ = 0,w_0_1___ = 0,w_1_0___ = 0,w_1_1___ = 0;

    doublereal log_p__, denom, oneme, log_u__, bplnk[56] = {}    , to3co2[1] = {}, pnewb[2] = {}, dplol, to3h2o[1] = {}, rphat, dplos,
        pinpl[4] = {}    , tpath, rdpnm, uinpl[4] = {}    , dtym10, winpl[4] = {}  , omeps,
        pnmsq[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {} , zinpl[4] = {}  ,
        rsqti, rsslp, posqt, o3bndi, sqrtu[1] = {}, dplco2, rbeta7[1] = {}, rbeta8,
        rbeta9, dplh2o[1] = {}, temh2o[4] = {}    , f1sqwp[1] = {},
        r2sslp, absbnd, rbeta13, tcrfac, log_uc__, alphat, a_star__,
        c_star__, q_path__;

    doublereal abstrc[1] = {}, tlocal, tpatha, l_star__, realnu, dbvtit[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}
        , dbvtly[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  ;

    integer cam_s1_0__ = 0,cam_s2_0__ = 0,cam_s3_0__ = 0,cam_s4_1__ = 0,bnd_idx__ = 0;

    doublereal rh_path__ = 0,rdpnmsq = 0;


    absnxt_dim1 = 1;
    absnxt_dim2 = CAM_LEVELS;
    absnxt_offset = 1 + absnxt_dim1 * (1 + absnxt_dim2);
    absnxt -= absnxt_offset;
    pmln_dim1 = 1;
    pmln_offset = 1 + pmln_dim1;
    pmln -= pmln_offset;
    co2eml_dim1 = 1;
    co2eml_offset = 1 + co2eml_dim1;
    co2eml -= co2eml_offset;
    pbr_dim1 = 1;
    pbr_offset = 1 + pbr_dim1;
    pbr -= pbr_offset;
    aer_trn_ttl_dim1 = 1;
    aer_trn_ttl_dim2 = (CAM_LEVELS+1);
    aer_trn_ttl_dim3 = (CAM_LEVELS+1);
    aer_trn_ttl_offset = 1 + aer_trn_ttl_dim1 * (1 + aer_trn_ttl_dim2 * (1 +
        aer_trn_ttl_dim3));
    aer_trn_ttl__ -= aer_trn_ttl_offset;
    aer_mpp_dim1 = 1;
    aer_mpp_offset = 1 + aer_mpp_dim1;
    aer_mpp__ -= aer_mpp_offset;
    wb_dim2 = 1;
    wb_offset = 1 + 2 * (1 + wb_dim2);
    wb -= wb_offset;
    plh2ob_dim2 = 1;
    plh2ob_offset = 1 + 2 * (1 + plh2ob_dim2);
    plh2ob -= plh2ob_offset;
    abstot_dim1 = 1;
    abstot_dim2 = (CAM_LEVELS+1);
    abstot_offset = 1 + abstot_dim1 * (1 + abstot_dim2);
    abstot -= abstot_offset;
    abplnk2_dim2 = 1;
    abplnk2_offset = 1 + 14 * (1 + abplnk2_dim2);
    abplnk2 -= abplnk2_offset;
    abplnk1_dim2 = 1;
    abplnk1_offset = 1 + 14 * (1 + abplnk1_dim2);
    abplnk1 -= abplnk1_offset;
    bch4_dim1 = 1;
    bch4_offset = 1 + bch4_dim1;
    bch4 -= bch4_offset;
    bn2o1_dim1 = 1;
    bn2o1_offset = 1 + bn2o1_dim1;
    bn2o1 -= bn2o1_offset;
    bn2o0_dim1 = 1;
    bn2o0_offset = 1 + bn2o0_dim1;
    bn2o0 -= bn2o0_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;
    piln_dim1 = 1;
    piln_offset = 1 + piln_dim1;
    piln -= piln_offset;
    plos_dim1 = 1;
    plos_offset = 1 + plos_dim1;
    plos -= plos_offset;
    plol_dim1 = 1;
    plol_offset = 1 + plol_dim1;
    plol -= plol_offset;
    tlayr_dim1 = 1;
    tlayr_offset = 1 + tlayr_dim1;
    tlayr -= tlayr_offset;
    tint_dim1 = 1;
    tint_offset = 1 + tint_dim1;
    tint -= tint_offset;
    co2t_dim1 = 1;
    co2t_offset = 1 + co2t_dim1;
    co2t -= co2t_offset;
    plh2o_dim1 = 1;
    plh2o_offset = 1 + plh2o_dim1;
    plh2o -= plh2o_offset;
    plco2_dim1 = 1;
    plco2_offset = 1 + plco2_dim1;
    plco2 -= plco2_offset;
    h2otr_dim1 = 1;
    h2otr_offset = 1 + h2otr_dim1;
    h2otr -= h2otr_offset;
    w_dim1 = 1;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    tcg_dim1 = 1;
    tcg_offset = 1 + tcg_dim1;
    tcg -= tcg_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;
    tplnka_dim1 = 1;
    tplnka_offset = 1 + tplnka_dim1;
    tplnka -= tplnka_offset;
    co2em_dim1 = 1;
    co2em_offset = 1 + co2em_dim1;
    co2em -= co2em_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;


    if (state->abs_lane == 0) {
    i__1 = state->cg_ntoplw__ - 1;
    for (k2 = 1; k2 <= i__1; ++k2) {
    i__2 = state->cg_ntoplw__ - 1;
    for (k1 = 1; k1 <= i__2; ++k1) {
        i__3 = 1 - 1;
        for (cam_s1_0__ = 0; cam_s1_0__ <= i__3; ++cam_s1_0__) {
        abstot[cam_s1_0__ + 1 + (k1 + k2 * abstot_dim2) * abstot_dim1]
             = state->cg_inf__;

        }


    }


    }

    for (k2 = 1; k2 <= 4; ++k2) {
    i__1 = state->cg_ntoplw__ - 1;
    for (k1 = 1; k1 <= i__1; ++k1) {
        i__2 = 1 - 1;
        for (cam_s2_0__ = 0; cam_s2_0__ <= i__2; ++cam_s2_0__) {
        absnxt[cam_s2_0__ + 1 + (k1 + k2 * absnxt_dim2) * absnxt_dim1]
             = state->cg_inf__;

        }


    }


    }

    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1 - 1;
    for (cam_s3_0__ = 0; cam_s3_0__ <= i__2; ++cam_s3_0__) {
        abstot[cam_s3_0__ + 1 + (k + k * abstot_dim2) * abstot_dim1] =
            state->cg_inf__;

    }


    }

    }
    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        dbvtly[i__ + k - 2] = (-2.89113668259233236e-4f + (
            2.37712515627208631e-6f + 1.13051887296045095e-10f *
            tlayr[i__ + (k + 1) * tlayr_dim1]) * tlayr[i__ + (k + 1) *
             tlayr_dim1]) / (1.f + (-.0061364821158349514f +
            1.5550320313195698e-5f * tlayr[i__ + (k + 1) * tlayr_dim1]
            ) * tlayr[i__ + (k + 1) * tlayr_dim1]);
        dbvtit[i__ + k - 2] = (-2.89113668259233236e-4f + (
            2.37712515627208631e-6f + 1.13051887296045095e-10f * tint[
            i__ + k * tint_dim1]) * tint[i__ + k * tint_dim1]) / (1.f
            + (-.0061364821158349514f + 1.5550320313195698e-5f * tint[
            i__ + k * tint_dim1]) * tint[i__ + k * tint_dim1]);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    dbvtit[i__ + (CAM_LEVELS+1) - 2] = (-2.89113668259233236e-4f + (
        2.37712515627208631e-6f + 1.13051887296045095e-10f * tint[i__
        + (CAM_LEVELS+1) * tint_dim1]) * tint[i__ + (CAM_LEVELS+1) * tint_dim1]) / (
        1.f + (-.0061364821158349514f + 1.5550320313195698e-5f * tint[
        i__ + (CAM_LEVELS+1) * tint_dim1]) * tint[i__ + (CAM_LEVELS+1) * tint_dim1]);

    }

    r293 = .0034129692832764505f;
    r250 = .0040000000000000001f;
    r3205 = 3.1201249372919126f;
    r300 = .0033333333333333335f;
    rsslp = 1.f / state->cg_sslp__;
    r2sslp = 1.f / (state->cg_sslp__ * 2.f);
    fdif = 1.65999996662139893f;
    sslp_mks__ = state->cg_sslp__ / 10.f;
    omeps = 1.f - state->cg_epsilo__;
    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {

        d__1 = pnm[i__ + k * pnm_dim1];
        pnmsq[i__ + k - 2] = d__1 * d__1;
        dtx[i__ - 1] = tplnka[i__ + k * tplnka_dim1] - 250.f;

    }


    }

    i__1 = state->cg_ntoplw__;
    for (k1 = (CAM_LEVELS+1)-state->abs_lane; k1 >= i__1; k1 -= state->abs_stride) {
    i__2 = state->cg_ntoplw__;
    for (k2 = (CAM_LEVELS+1); k2 >= i__2; --k2) {
        if (k1 == k2) {
        goto L9030;
        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        dplh2o[i__ - 1] = plh2o[i__ + k1 * plh2o_dim1] - plh2o[i__ +
            k2 * plh2o_dim1];
        u[i__ - 1] = (d__1 = dplh2o[i__ - 1], abs(d__1));
        sqrtu[i__ - 1] = sqrt(u[i__ - 1]);
        ds2c = (d__1 = s2c[i__ + k1 * s2c_dim1] - s2c[i__ + k2 *
            s2c_dim1], abs(d__1));
        dw[i__ - 1] = (d__1 = w[i__ + k1 * w_dim1] - w[i__ + k2 *
            w_dim1], abs(d__1));
        uc1[i__ - 1] = (ds2c + u[i__ - 1] * .00170000002253800631f) *
            (ds2c * 2.f + 1.f) / (ds2c * 15.f + 1.f);
        pch2o = ds2c;
        pnew[i__ - 1] = u[i__ - 1] / dw[i__ - 1];
        pnew_mks__ = pnew[i__ - 1] * sslp_mks__;
        tpatha = (d__1 = tcg[i__ + k1 * tcg_dim1] - tcg[i__ + k2 *
            tcg_dim1], abs(d__1)) / dw[i__ - 1];

        d__1 = max(tpatha,state->cg_min_tp_h2o__);
        t_p__ = min(d__1,state->cg_max_tp_h2o__);
        iest = (integer) (cam_floor__(&t_p__) -
            state->cg_min_tp_h2o__);
        esx = state->cg_estblh2o__[iest] + (
            state->cg_estblh2o__[iest + 1] -
            state->cg_estblh2o__[iest]) * (t_p__ -
            state->cg_min_tp_h2o__ - iest);
        qsx = state->cg_epsilo__ * esx / (pnew_mks__ - omeps *
            esx);
        q_path__ = dw[i__ - 1] / (d__1 = pnm[i__ + k1 * pnm_dim1] -
            pnm[i__ + k2 * pnm_dim1], abs(d__1)) /
            state->cg_rga__;
        ub[0] = (d__1 = plh2ob[(i__ + k1 * plh2ob_dim2 << 1) + 1] -
            plh2ob[(i__ + k2 * plh2ob_dim2 << 1) + 1], abs(d__1))
            / psi_(state, &t_p__, &radabs_c__1);
        ub[1] = (d__1 = plh2ob[(i__ + k1 * plh2ob_dim2 << 1) + 2] -
            plh2ob[(i__ + k2 * plh2ob_dim2 << 1) + 2], abs(d__1))
            / psi_(state, &t_p__, &radabs_c__2);
        pnewb[0] = ub[0] / (d__1 = wb[(i__ + k1 * wb_dim2 << 1) + 1]
            - wb[(i__ + k2 * wb_dim2 << 1) + 1], abs(d__1)) *
            phi_(state, &t_p__, &radabs_c__1);
        pnewb[1] = ub[1] / (d__1 = wb[(i__ + k1 * wb_dim2 << 1) + 2]
            - wb[(i__ + k2 * wb_dim2 << 1) + 2], abs(d__1)) *
            phi_(state, &t_p__, &radabs_c__2);
        dtx[i__ - 1] = tplnka[i__ + k2 * tplnka_dim1] - 250.f;
        dty[i__ - 1] = tpatha - 250.f;
        fwk[i__ - 1] = state->cg_fwcoef__ +
            state->cg_fwc1__ / (state->cg_fwc2__ * u[
            i__ - 1] + 1.f);
        fwku[i__ - 1] = fwk[i__ - 1] * u[i__ - 1];
        te1 = tplnka[i__ + k2 * tplnka_dim1];
        te2 = te1 * te1;
        te3 = te2 * te1;
        te4 = te3 * te1;
        te5 = te4 * te1;
        dvar = (t_p__ - state->cg_min_tp_h2o__) /
            state->cg_dtp_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        itp = min(i__4,9);
        itp1 = itp + 1;
        wtp = dvar - cam_floor__(&dvar);
        wtp1 = 1.f - wtp;


        d__2 = tplnka[i__ + k2 * tplnka_dim1] - t_p__;
        d__1 = max(d__2,state->cg_min_te_h2o__);
        t_e__ = min(d__1,state->cg_max_te_h2o__);
        dvar = (t_e__ - state->cg_min_te_h2o__) /
            state->cg_dte_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        ite = min(i__4,20);
        ite1 = ite + 1;
        wte = dvar - cam_floor__(&dvar);
        wte1 = 1.f - wte;


        d__2 = q_path__ / qsx;
        d__1 = max(d__2,state->cg_min_rh_h2o__);
        rh_path__ = min(d__1,state->cg_max_rh_h2o__);
        dvar = (rh_path__ - state->cg_min_rh_h2o__) /
            state->cg_drh_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        irh = min(i__4,6);
        irh1 = irh + 1;
        wrh = dvar - cam_floor__(&dvar);
        wrh1 = 1.f - wrh;
        w_0_0___ = wtp * wte;
        w_0_1___ = wtp * wte1;
        w_1_0___ = wtp1 * wte;
        w_1_1___ = wtp1 * wte1;
        w_0_00__ = w_0_0___ * wrh;
        w_0_01__ = w_0_0___ * wrh1;
        w_0_10__ = w_0_1___ * wrh;
        w_0_11__ = w_0_1___ * wrh1;
        w_1_00__ = w_1_0___ * wrh;
        w_1_01__ = w_1_0___ * wrh1;
        w_1_10__ = w_1_1___ * wrh;
        w_1_11__ = w_1_1___ * wrh1;
        fch2o = fh2oself_(state, &t_p__);
        uch2o = pch2o * state->cg_epsilo__ / (q_path__ * fch2o);
        ib = 1;
        uvar = ub[ib - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        iu = min(i__4,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnewb[ib - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        ip = min(i__4,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        a_star__ = state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ *
            wu + state->cg_ah2onw__[ip1 + (itp + (iu + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ *
            wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu1 + state->cg_ah2onw__[ip1 + (itp
            + (iu1 + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_11__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_10__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_01__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_11__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_10__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_01__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
             w00_00__ * wu1 + state->cg_ah2onw__[ip1 + (
            itp1 + (iu1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 -
            55111] * w00_10__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 -
            55111] * w00_01__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10
            - 55111] * w00_00__ * wu;


        d__2 = fa * (1.f - (1.f - a_star__) * aer_trn_ttl__[i__ + (k1
            + (k2 + ib * aer_trn_ttl_dim3) * aer_trn_ttl_dim2) *
            aer_trn_ttl_dim1]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }
        ib = 2;
        uvar = ub[ib - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        iu = min(i__4,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnewb[ib - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        ip = min(i__4,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;


        d__3 = uch2o * fdif;
        d__2 = max(d__3,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_uc__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_uc__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__5 = (integer) d_int(&dvar) + 1;
        i__4 = max(i__5,1);
        iuc = min(i__4,24);
        iuc1 = iuc + 1;
        wuc = dvar - cam_floor__(&dvar);
        wuc1 = 1.f - wuc;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        l_star__ = state->cg_ln_ah2ow__[ip + (itp + (iu + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ *
            wu1 + state->cg_ln_ah2ow__[ip + (itp + (iu + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__
            * wu1 + state->cg_ln_ah2ow__[ip + (itp + (iu + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w11_01__
            * wu1 + state->cg_ln_ah2ow__[ip + (itp + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w11_00__ * wu1 + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w11_11__ * wu + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
             w11_10__ * wu + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
             w11_01__ * wu + state->cg_ln_ah2ow__[ip + (itp
            + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111]
            * w11_00__ * wu + state->cg_ln_ah2ow__[ip + (
            itp1 + (iu + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w10_11__ * wu1 + state->cg_ln_ah2ow__[
            ip + (itp1 + (iu + (ite + irh1 * 21) * 25) * 10) * 10
            - 55111] * w10_10__ * wu1 +
            state->cg_ln_ah2ow__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ln_ah2ow__[ip + (itp1 + (iu + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ *
            wu1 + state->cg_ln_ah2ow__[ip + (itp1 + (iu1 + (
            ite + irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ *
             wu + state->cg_ln_ah2ow__[ip + (itp1 + (iu1 + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__
            * wu + state->cg_ln_ah2ow__[ip + (itp1 + (iu1 +
            (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w10_01__ * wu + state->cg_ln_ah2ow__[ip + (itp1
            + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111]
            * w10_00__ * wu + state->cg_ln_ah2ow__[ip1 + (
            itp + (iu + (ite + irh * 21) * 25) * 10) * 10 - 55111]
             * w01_11__ * wu1 + state->cg_ln_ah2ow__[ip1 + (
            itp + (iu + (ite + irh1 * 21) * 25) * 10) * 10 -
            55111] * w01_10__ * wu1 + state->cg_ln_ah2ow__[
            ip1 + (itp + (iu + (ite1 + irh * 21) * 25) * 10) * 10
            - 55111] * w01_01__ * wu1 +
            state->cg_ln_ah2ow__[ip1 + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu1
            + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu
            + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 + (ite
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ *
            wu + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__
            * wu + state->cg_ln_ah2ow__[ip1 + (itp + (iu1 +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu + state->cg_ln_ah2ow__[ip1 + (
            itp1 + (iu + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu1 + state->cg_ln_ah2ow__[
            ip1 + (itp1 + (iu + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wu1 +
            state->cg_ln_ah2ow__[ip1 + (itp1 + (iu + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ *
            wu1 + state->cg_ln_ah2ow__[ip1 + (itp1 + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wu1 + state->cg_ln_ah2ow__[ip1 + (
            itp1 + (iu1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu + state->cg_ln_ah2ow__[
            ip1 + (itp1 + (iu1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wu +
            state->cg_ln_ah2ow__[ip1 + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu
            + state->cg_ln_ah2ow__[ip1 + (itp1 + (iu1 + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wu;
        c_star__ = state->cg_cn_ah2ow__[ip + (itp + (iuc + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ *
            wuc1 + state->cg_cn_ah2ow__[ip + (itp + (iuc + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__
            * wuc1 + state->cg_cn_ah2ow__[ip + (itp + (iuc
            + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w11_01__ * wuc1 + state->cg_cn_ah2ow__[ip + (
            itp + (iuc + (ite1 + irh1 * 21) * 25) * 10) * 10 -
            55111] * w11_00__ * wuc1 + state->cg_cn_ah2ow__[
            ip + (itp + (iuc1 + (ite + irh * 21) * 25) * 10) * 10
            - 55111] * w11_11__ * wuc +
            state->cg_cn_ah2ow__[ip + (itp + (iuc1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wuc
            + state->cg_cn_ah2ow__[ip + (itp + (iuc1 + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w11_01__
            * wuc + state->cg_cn_ah2ow__[ip + (itp + (iuc1
            + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w11_00__ * wuc + state->cg_cn_ah2ow__[ip + (
            itp1 + (iuc + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w10_11__ * wuc1 + state->cg_cn_ah2ow__[
            ip + (itp1 + (iuc + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w10_10__ * wuc1 +
            state->cg_cn_ah2ow__[ip + (itp1 + (iuc + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ *
            wuc1 + state->cg_cn_ah2ow__[ip + (itp1 + (iuc +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w10_00__ * wuc1 + state->cg_cn_ah2ow__[ip + (
            itp1 + (iuc1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w10_11__ * wuc + state->cg_cn_ah2ow__[
            ip + (itp1 + (iuc1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w10_10__ * wuc +
            state->cg_cn_ah2ow__[ip + (itp1 + (iuc1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ *
            wuc + state->cg_cn_ah2ow__[ip + (itp1 + (iuc1 +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w10_00__ * wuc + state->cg_cn_ah2ow__[ip1 + (
            itp + (iuc + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w01_11__ * wuc1 + state->cg_cn_ah2ow__[
            ip1 + (itp + (iuc + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w01_10__ * wuc1 +
            state->cg_cn_ah2ow__[ip1 + (itp + (iuc + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ *
            wuc1 + state->cg_cn_ah2ow__[ip1 + (itp + (iuc +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wuc1 + state->cg_cn_ah2ow__[ip1 + (
            itp + (iuc1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w01_11__ * wuc + state->cg_cn_ah2ow__[
            ip1 + (itp + (iuc1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w01_10__ * wuc +
            state->cg_cn_ah2ow__[ip1 + (itp + (iuc1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ *
            wuc + state->cg_cn_ah2ow__[ip1 + (itp + (iuc1 +
            (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wuc + state->cg_cn_ah2ow__[ip1 + (
            itp1 + (iuc + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wuc1 + state->cg_cn_ah2ow__[
            ip1 + (itp1 + (iuc + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wuc1 +
            state->cg_cn_ah2ow__[ip1 + (itp1 + (iuc + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ *
            wuc1 + state->cg_cn_ah2ow__[ip1 + (itp1 + (iuc
            + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wuc1 + state->cg_cn_ah2ow__[ip1 + (
            itp1 + (iuc1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wuc + state->cg_cn_ah2ow__[
            ip1 + (itp1 + (iuc1 + (ite + irh1 * 21) * 25) * 10) *
            10 - 55111] * w00_10__ * wuc +
            state->cg_cn_ah2ow__[ip1 + (itp1 + (iuc1 + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__
            * wuc + state->cg_cn_ah2ow__[ip1 + (itp1 + (
            iuc1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_00__ * wuc;


        d__2 = fa * (1.f - l_star__ * c_star__ * aer_trn_ttl__[i__ + (
            k1 + (k2 + ib * aer_trn_ttl_dim3) * aer_trn_ttl_dim2)
            * aer_trn_ttl_dim1]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }

        }

        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        term7[i__ - 1] = state->cg_coefj__[0] +
            state->cg_coefj__[1] * dty[i__ - 1] * (
            state->cg_c16__ * dty[i__ - 1] + 1.f);
        term8[i__ - 1] = state->cg_coefk__[0] +
            state->cg_coefk__[1] * dty[i__ - 1] * (
            state->cg_c17__ * dty[i__ - 1] + 1.f);
        term7[i__] = state->cg_coefj__[3] +
            state->cg_coefj__[4] * dty[i__ - 1] * (
            state->cg_c26__ * dty[i__ - 1] + 1.f);
        term8[i__] = state->cg_coefk__[3] +
            state->cg_coefk__[4] * dty[i__ - 1] * (
            state->cg_c27__ * dty[i__ - 1] + 1.f);

        }

        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        k21 = term7[i__ - 1] + term8[i__ - 1] / ((
            state->cg_c30__ + state->cg_c31__ * (dty[
            i__ - 1] - 10.f) * (dty[i__ - 1] - 10.f)) * sqrtu[i__
            - 1] + 1.f);
        k22 = term7[i__] + term8[i__] / ((state->cg_c28__ +
            state->cg_c29__ * (dty[i__ - 1] - 10.f)) *
            sqrtu[i__ - 1] + 1.f);
        tr1 = cam_exp64(-(k21 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr2 = cam_exp64(-(k22 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr1 *= aer_trn_ttl__[i__ + (k1 + (k2 + (aer_trn_ttl_dim3 << 2)
            ) * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
        tr2 *= aer_trn_ttl__[i__ + (k1 + (k2 + aer_trn_ttl_dim3 * 3) *
             aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
        tr5 = cam_exp64(-((state->cg_coefh__[4] +
            state->cg_coefh__[5] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr6 = cam_exp64(-((state->cg_coefh__[6] +
            state->cg_coefh__[7] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr9[i__ - 1] = tr1 * tr5;
        tr10[i__ - 1] = tr2 * tr6;
        th2o[i__ - 1] = tr10[i__ - 1];
        trab2[i__ - 1] = tr9[i__ - 1] * .64999997615814209f + tr10[
            i__ - 1] * .349999994039535522f;

        }

        if (k2 < k1) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            to3h2o[i__ - 1] = h2otr[i__ + k1 * h2otr_dim1] / h2otr[
                i__ + k2 * h2otr_dim1];

        }

        } else {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            to3h2o[i__ - 1] = h2otr[i__ + k2 * h2otr_dim1] / h2otr[
                i__ + k1 * h2otr_dim1];

        }

        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        dpnm[i__ - 1] = pnm[i__ + k1 * pnm_dim1] - pnm[i__ + k2 *
            pnm_dim1];
        to3co2[i__ - 1] = (pnm[i__ + k1 * pnm_dim1] * co2t[i__ + k1 *
            co2t_dim1] - pnm[i__ + k2 * pnm_dim1] * co2t[i__ + k2
            * co2t_dim1]) / dpnm[i__ - 1];
        d__1 = to3co2[i__ - 1] * r293;
        te = pow_dd(&d__1, &radabs_c_b43);
        dplos = plos[i__ + k1 * plos_dim1] - plos[i__ + k2 *
            plos_dim1];
        dplol = plol[i__ + k1 * plol_dim1] - plol[i__ + k2 *
            plol_dim1];
        u1 = abs(dplos) * 18.2900009155273438f / te;
        u2 = abs(dplos) * .564899981021881104f / te;
        rphat = dplol / dplos;
        tlocal = tint[i__ + k2 * tint_dim1];
        tcrfac = sqrt(tlocal * r250) * te;
        beta = r3205 * (rphat + state->cg_dpfo3__ * tcrfac);
        realnu = te / beta;
        tmp1 = u1 / sqrt(u1 * (realnu + 1.f) + 4.f);
        tmp2 = u2 / sqrt(u2 * (realnu + 1.f) + 4.f);
        o3bndi = te * 74.f * cam_log64(tmp1 + 1.f + tmp2);
        abso[i__ + 1] = o3bndi * to3h2o[i__ - 1] * dbvtit[i__ + k2 -
            2];
        to3[i__ - 1] = 1.f / (tmp1 * .100000001490116119f + 1.f +
            tmp2 * .100000001490116119f);

        }

        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        sqwp = sqrt((d__1 = plco2[i__ + k1 * plco2_dim1] - plco2[i__
            + k2 * plco2_dim1], abs(d__1)));
        et = cam_exp64(-480.f / to3co2[i__ - 1]);
        sqti[i__ - 1] = sqrt(to3co2[i__ - 1]);
        rsqti = 1.f / sqti[i__ - 1];
        et2 = et * et;
        et4 = et2 * et2;
        omet = 1.f - et2 * 1.5f;
        f1co2 = omet * 899.70001220703125f * (et *
            1.94773995876312256f + 1.f + et2 *
            4.73485994338989258f) * rsqti;
        f1sqwp[i__ - 1] = f1co2 * sqwp;
        t1co2[i__ - 1] = 1.f / (omet * 245.17999267578125f * sqwp *
            rsqti + 1.f);
        oneme = 1.f - et2;

        d__1 = oneme;
        alphat = d__1 * (d__1 * d__1) * rsqti;
        pi = (d__1 = dpnm[i__ - 1], abs(d__1));
        wco2 = state->cg_co2vmr__ * 2.52209997177124023f * pi *
            state->cg_rga__;
        u7[i__ - 1] = alphat * 49411.f * et2 * wco2;
        u8 = alphat * 39744.f * et4 * wco2;
        u9 = alphat * 104470.f * et4 * et2 * wco2;
        u13 = alphat * 2838.800048828125f * et4 * wco2;
        tpath = to3co2[i__ - 1];
        tlocal = tint[i__ + k2 * tint_dim1];
        tcrfac = sqrt(tlocal * r250 * tpath * r300);
        posqt = ((pnm[i__ + k2 * pnm_dim1] + pnm[i__ + k1 * pnm_dim1])
             * r2sslp + state->cg_dpfco2__ * tcrfac) *
            rsqti;
        rbeta7[i__ - 1] = 1.f / (posqt * 5.3228001594543457f);
        rbeta8 = 1.f / (posqt * 10.6576004028320312f);
        rbeta9 = rbeta7[i__ - 1];
        rbeta13 = rbeta9;
        f2co2[i__ - 1] = u7[i__ - 1] / sqrt(u7[i__ - 1] * (rbeta7[i__
            - 1] + 1.f) + 4.f) + u8 / sqrt(u8 * (rbeta8 + 1.f) +
            4.f) + u9 / sqrt(u9 * (rbeta9 + 1.f) + 4.f);
        f3co2[i__ - 1] = u13 / sqrt(u13 * (rbeta13 + 1.f) + 4.f);

        }

        if (k2 >= k1) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            sqti[i__ - 1] = sqrt(tlayr[i__ + k2 * tlayr_dim1]);

        }

        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        tmp1 = cam_log64(f1sqwp[i__ - 1] + 1.f);
        tmp2 = cam_log64(f2co2[i__ - 1] + 1.f);
        tmp3 = cam_log64(f3co2[i__ - 1] + 1.f);
        absbnd = (tmp1 + t1co2[i__ - 1] * 2.f * tmp2 + tmp3 * 2.f) *
            sqti[i__ - 1];
        abso[i__ + 2] = trab2[i__ - 1] * co2em[i__ + k2 * co2em_dim1]
            * absbnd;
        tco2[i__ - 1] = 1.f / (u7[i__ - 1] / sqrt(u7[i__ - 1] * (
            rbeta7[i__ - 1] + 1.f) + 4.f) * 10.f + 1.f);

        }

        trcab_(state, lchnk, ncol, pcols, pverp, &k1, &k2, &ucfc11[ucfc11_offset]
            , &ucfc12[ucfc12_offset], &un2o0[un2o0_offset], &un2o1[
            un2o1_offset], &uch4[uch4_offset], &uco211[uco211_offset],
             &uco212[uco212_offset], &uco213[uco213_offset], &uco221[
            uco221_offset], &uco222[uco222_offset], &uco223[
            uco223_offset], &bn2o0[bn2o0_offset], &bn2o1[bn2o1_offset]
            , &bch4[bch4_offset], to3co2, &pnm[pnm_offset], dw, pnew,
            &s2c[s2c_offset], &uptype[uptype_offset], u, &abplnk1[
            abplnk1_offset], tco2, th2o, to3, abstrc, &aer_trn_ttl__[
            aer_trn_ttl_offset]);
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        abstot[i__ + (k1 + k2 * abstot_dim2) * abstot_dim1] = abso[
            i__ - 1] + abso[i__] + abso[i__ + 1] + abso[i__ + 2]
            + abstrc[i__ - 1];

        }

L9030:
        ;
    }


    }

    i__1 = state->cg_ntoplw__;
    for (k2 = CAM_LEVELS-state->abs_lane; k2 >= i__1; k2 -= state->abs_stride) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        tbar[i__ - 1] = (tint[i__ + (k2 + 1) * tint_dim1] + tlayr[i__ + (
            k2 + 1) * tlayr_dim1]) * .5f;
        emm[i__ - 1] = (co2em[i__ + (k2 + 1) * co2em_dim1] + co2eml[i__ +
            k2 * co2eml_dim1]) * .5f;
        tbar[i__] = (tlayr[i__ + (k2 + 1) * tlayr_dim1] + tint[i__ + k2 *
            tint_dim1]) * .5f;
        emm[i__] = (co2em[i__ + k2 * co2em_dim1] + co2eml[i__ + k2 *
            co2eml_dim1]) * .5f;
        tbar[i__ + 1] = (tbar[i__] + tbar[i__ - 1]) * .5f;
        emm[i__ + 1] = emm[i__ - 1];
        tbar[i__ + 2] = tbar[i__ + 1];
        emm[i__ + 2] = emm[i__];
        o3emm[i__ - 1] = (dbvtit[i__ + (k2 + 1) - 2] + dbvtly[i__ + k2 -
            2]) * .5f;
        o3emm[i__] = (dbvtit[i__ + k2 - 2] + dbvtly[i__ + k2 - 2]) * .5f;
        o3emm[i__ + 1] = o3emm[i__ - 1];
        o3emm[i__ + 2] = o3emm[i__];
        temh2o[i__ - 1] = tbar[i__ - 1];
        temh2o[i__] = tbar[i__];
        temh2o[i__ + 1] = tbar[i__ - 1];
        temh2o[i__ + 2] = tbar[i__];
        dpnm[i__ - 1] = pnm[i__ + (k2 + 1) * pnm_dim1] - pnm[i__ + k2 *
            pnm_dim1];

    }

    for (wvl = 1; wvl <= 14; ++wvl) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        bplnk[wvl + (i__ + 1) * 14 - 29] = (abplnk1[wvl + (i__ + (k2
            + 1) * abplnk1_dim2) * 14] + abplnk2[wvl + (i__ + k2 *
             abplnk2_dim2) * 14]) * .5f;
        bplnk[wvl + (i__ + 2) * 14 - 29] = (abplnk1[wvl + (i__ + k2 *
            abplnk1_dim2) * 14] + abplnk2[wvl + (i__ + k2 *
            abplnk2_dim2) * 14]) * .5f;
        bplnk[wvl + (i__ + 3) * 14 - 29] = bplnk[wvl + (i__ + 1) * 14
            - 29];
        bplnk[wvl + (i__ + 4) * 14 - 29] = bplnk[wvl + (i__ + 2) * 14
            - 29];

        }


    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        rdpnmsq = 1.f / (pnmsq[i__ + (k2 + 1) - 2] - pnmsq[i__ + k2 - 2]);
        rdpnm = 1.f / dpnm[i__ - 1];
        p1 = (pbr[i__ + k2 * pbr_dim1] + pnm[i__ + (k2 + 1) * pnm_dim1]) *
             .5f;
        p2 = (pbr[i__ + k2 * pbr_dim1] + pnm[i__ + k2 * pnm_dim1]) * .5f;

        d__1 = p1;
        uinpl[i__ - 1] = (pnmsq[i__ + (k2 + 1) - 2] - d__1 * d__1) *
            rdpnmsq;

        d__1 = p2;
        uinpl[i__] = -(pnmsq[i__ + k2 - 2] - d__1 * d__1) * rdpnmsq;

        d__1 = p1;
        uinpl[i__ + 1] = -(pnmsq[i__ + k2 - 2] - d__1 * d__1) * rdpnmsq;

        d__1 = p2;
        uinpl[i__ + 2] = (pnmsq[i__ + (k2 + 1) - 2] - d__1 * d__1) *
            rdpnmsq;
        winpl[i__ - 1] = (pnm[i__ + (k2 + 1) * pnm_dim1] - pbr[i__ + k2 *
            pbr_dim1]) * .5f * rdpnm;
        winpl[i__] = (-pnm[i__ + k2 * pnm_dim1] + pbr[i__ + k2 * pbr_dim1]
            ) * .5f * rdpnm;
        winpl[i__ + 1] = ((pnm[i__ + (k2 + 1) * pnm_dim1] + pbr[i__ + k2 *
             pbr_dim1]) * .5f - pnm[i__ + k2 * pnm_dim1]) * rdpnm;
        winpl[i__ + 2] = ((-pnm[i__ + k2 * pnm_dim1] - pbr[i__ + k2 *
            pbr_dim1]) * .5f + pnm[i__ + (k2 + 1) * pnm_dim1]) *
            rdpnm;
        tmp1 = 1.f / (piln[i__ + (k2 + 1) * piln_dim1] - piln[i__ + k2 *
            piln_dim1]);
        tmp2 = piln[i__ + (k2 + 1) * piln_dim1] - pmln[i__ + k2 *
            pmln_dim1];
        tmp3 = piln[i__ + k2 * piln_dim1] - pmln[i__ + k2 * pmln_dim1];
        zinpl[i__ - 1] = tmp2 * .5f * tmp1;
        zinpl[i__] = tmp3 * -.5f * tmp1;
        zinpl[i__ + 1] = (tmp2 * .5f - tmp3) * tmp1;
        zinpl[i__ + 2] = (tmp2 - tmp3 * .5f) * tmp1;
        pinpl[i__ - 1] = (p1 + pnm[i__ + (k2 + 1) * pnm_dim1]) * .5f;
        pinpl[i__] = (p2 + pnm[i__ + k2 * pnm_dim1]) * .5f;
        pinpl[i__ + 1] = (p1 + pnm[i__ + k2 * pnm_dim1]) * .5f;
        pinpl[i__ + 2] = (p2 + pnm[i__ + (k2 + 1) * pnm_dim1]) * .5f;
        if (state->cg_strat_volcanic__) {
        aer_pth_ngh__[i__ - 1] = (d__1 = aer_mpp__[i__ + k2 *
            aer_mpp_dim1] - aer_mpp__[i__ + (k2 + 1) *
            aer_mpp_dim1], abs(d__1));
        }

    }

    for (kn = 1; kn <= 4; ++kn) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        u[i__ - 1] = uinpl[i__ + kn - 2] * (d__1 = plh2o[i__ + k2 *
            plh2o_dim1] - plh2o[i__ + (k2 + 1) * plh2o_dim1], abs(
            d__1));
        sqrtu[i__ - 1] = sqrt(u[i__ - 1]);
        dw[i__ - 1] = (d__1 = w[i__ + k2 * w_dim1] - w[i__ + (k2 + 1)
            * w_dim1], abs(d__1));
        pnew[i__ - 1] = u[i__ - 1] / (winpl[i__ + kn - 2] * dw[i__ -
            1]);
        pnew_mks__ = pnew[i__ - 1] * sslp_mks__;


        d__2 = tbar[i__ + kn - 2];
        d__1 = max(d__2,state->cg_min_tp_h2o__);
        t_p__ = min(d__1,state->cg_max_tp_h2o__);
        iest = (integer) (cam_floor__(&t_p__) -
            state->cg_min_tp_h2o__);
        esx = state->cg_estblh2o__[iest] + (
            state->cg_estblh2o__[iest + 1] -
            state->cg_estblh2o__[iest]) * (t_p__ -
            state->cg_min_tp_h2o__ - iest);
        qsx = state->cg_epsilo__ * esx / (pnew_mks__ - omeps *
            esx);
        q_path__ = dw[i__ - 1] / (d__1 = dpnm[i__ - 1], abs(d__1)) /
            state->cg_rga__;
        ds2c = (d__1 = s2c[i__ + k2 * s2c_dim1] - s2c[i__ + (k2 + 1) *
             s2c_dim1], abs(d__1));
        uc1[i__ - 1] = uinpl[i__ + kn - 2] * ds2c;
        pch2o = uc1[i__ - 1];
        uc1[i__ - 1] = (uc1[i__ - 1] + u[i__ - 1] *
            .00170000002253800631f) * (uc1[i__ - 1] * 2.f + 1.f) /
             (uc1[i__ - 1] * 15.f + 1.f);
        dtx[i__ - 1] = temh2o[i__ + kn - 2] - 250.f;
        dty[i__ - 1] = tbar[i__ + kn - 2] - 250.f;
        fwk[i__ - 1] = state->cg_fwcoef__ +
            state->cg_fwc1__ / (state->cg_fwc2__ * u[
            i__ - 1] + 1.f);
        fwku[i__ - 1] = fwk[i__ - 1] * u[i__ - 1];
        if (state->cg_strat_volcanic__) {
            aer_pth_dlt__ = uinpl[i__ + kn - 2] * aer_pth_ngh__[i__ -
                1];
            for (bnd_idx__ = 1; bnd_idx__ <= 7; ++bnd_idx__) {
            odap_aer_ttl__ = state->cg_abs_cff_mss_aer__[
                bnd_idx__ - 1] * aer_pth_dlt__;
            aer_trn_ngh__[i__ + bnd_idx__ - 2] = cam_exp64(-fdif *
                odap_aer_ttl__);

            }

        } else {
            for (cam_s4_1__ = 0; cam_s4_1__ <= 6; ++cam_s4_1__) {
            aer_trn_ngh__[i__ + (cam_s4_1__ + 1) - 2] = 1.f;

            }

        }
        te1 = temh2o[i__ + kn - 2];
        te2 = te1 * te1;
        te3 = te2 * te1;
        te4 = te3 * te1;
        te5 = te4 * te1;
        uvar = u[i__ - 1] * fdif;

        d__2 = max(uvar,state->cg_min_u_h2o__);
        d__1 = d_lg10(&d__2);
        log_u__ = min(d__1,state->cg_max_lu_h2o__);
        dvar = (log_u__ - state->cg_min_lu_h2o__) /
            state->cg_dlu_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        iu = min(i__3,24);
        iu1 = iu + 1;
        wu = dvar - cam_floor__(&dvar);
        wu1 = 1.f - wu;


        d__3 = pnew[i__ - 1];
        d__2 = max(d__3,state->cg_min_p_h2o__);
        d__1 = d_lg10(&d__2);
        log_p__ = min(d__1,state->cg_max_lp_h2o__);
        dvar = (log_p__ - state->cg_min_lp_h2o__) /
            state->cg_dlp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ip = min(i__3,9);
        ip1 = ip + 1;
        wp = dvar - cam_floor__(&dvar);
        wp1 = 1.f - wp;
        dvar = (t_p__ - state->cg_min_tp_h2o__) /
            state->cg_dtp_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        itp = min(i__3,9);
        itp1 = itp + 1;
        wtp = dvar - cam_floor__(&dvar);
        wtp1 = 1.f - wtp;


        d__2 = temh2o[i__ + kn - 2] - t_p__;
        d__1 = max(d__2,state->cg_min_te_h2o__);
        t_e__ = min(d__1,state->cg_max_te_h2o__);
        dvar = (t_e__ - state->cg_min_te_h2o__) /
            state->cg_dte_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        ite = min(i__3,20);
        ite1 = ite + 1;
        wte = dvar - cam_floor__(&dvar);
        wte1 = 1.f - wte;


        d__2 = q_path__ / qsx;
        d__1 = max(d__2,state->cg_min_rh_h2o__);
        rh_path__ = min(d__1,state->cg_max_rh_h2o__);
        dvar = (rh_path__ - state->cg_min_rh_h2o__) /
            state->cg_drh_h2o__;


        i__4 = (integer) d_int(&dvar) + 1;
        i__3 = max(i__4,1);
        irh = min(i__3,6);
        irh1 = irh + 1;
        wrh = dvar - cam_floor__(&dvar);
        wrh1 = 1.f - wrh;
        w_0_0___ = wtp * wte;
        w_0_1___ = wtp * wte1;
        w_1_0___ = wtp1 * wte;
        w_1_1___ = wtp1 * wte1;
        w_0_00__ = w_0_0___ * wrh;
        w_0_01__ = w_0_0___ * wrh1;
        w_0_10__ = w_0_1___ * wrh;
        w_0_11__ = w_0_1___ * wrh1;
        w_1_00__ = w_1_0___ * wrh;
        w_1_01__ = w_1_0___ * wrh1;
        w_1_10__ = w_1_1___ * wrh;
        w_1_11__ = w_1_1___ * wrh1;
        w00_00__ = wp * w_0_00__;
        w00_01__ = wp * w_0_01__;
        w00_10__ = wp * w_0_10__;
        w00_11__ = wp * w_0_11__;
        w01_00__ = wp * w_1_00__;
        w01_01__ = wp * w_1_01__;
        w01_10__ = wp * w_1_10__;
        w01_11__ = wp * w_1_11__;
        w10_00__ = wp1 * w_0_00__;
        w10_01__ = wp1 * w_0_01__;
        w10_10__ = wp1 * w_0_10__;
        w10_11__ = wp1 * w_0_11__;
        w11_00__ = wp1 * w_1_00__;
        w11_01__ = wp1 * w_1_01__;
        w11_10__ = wp1 * w_1_10__;
        w11_11__ = wp1 * w_1_11__;
        ib = 1;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        a_star__ = state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu
            + state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ah2onw__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu
            + state->cg_ah2onw__[ip + (itp1 + (iu1 + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ *
            wu + state->cg_ah2onw__[ip1 + (itp + (iu + (ite
            + irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ *
            wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite + irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh * 21) * 25) * 10) * 10 - 55111] * w01_01__
            * wu1 + state->cg_ah2onw__[ip1 + (itp + (iu + (
            ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu1 + state->cg_ah2onw__[ip1 + (itp
            + (iu1 + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_11__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_10__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w01_01__ * wu + state->cg_ah2onw__[ip1 + (itp +
            (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w01_00__ * wu + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_11__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite + irh1 * 21) * 25) * 10) * 10 - 55111] *
            w00_10__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh * 21) * 25) * 10) * 10 - 55111] *
            w00_01__ * wu1 + state->cg_ah2onw__[ip1 + (itp1
            + (iu + (ite1 + irh1 * 21) * 25) * 10) * 10 - 55111] *
             w00_00__ * wu1 + state->cg_ah2onw__[ip1 + (
            itp1 + (iu1 + (ite + irh * 21) * 25) * 10) * 10 -
            55111] * w00_11__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite + irh1 * 21) * 25) * 10) * 10 -
            55111] * w00_10__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh * 21) * 25) * 10) * 10 -
            55111] * w00_01__ * wu + state->cg_ah2onw__[ip1
            + (itp1 + (iu1 + (ite1 + irh1 * 21) * 25) * 10) * 10
            - 55111] * w00_00__ * wu;


        d__2 = fa * (1.f - (1.f - a_star__) * aer_trn_ngh__[i__ + ib
            - 2]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }
        ib = 2;
        fa = state->cg_fat__[ib * 6 - 6] +
            state->cg_fat__[ib * 6 - 5] * te1 +
            state->cg_fat__[ib * 6 - 4] * te2 +
            state->cg_fat__[ib * 6 - 3] * te3 +
            state->cg_fat__[ib * 6 - 2] * te4 +
            state->cg_fat__[ib * 6 - 1] * te5;
        a_star__ = state->cg_ah2ow__[ip + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu1
            + state->cg_ah2ow__[ip + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_11__ * wu +
            state->cg_ah2ow__[ip + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_10__ * wu
            + state->cg_ah2ow__[ip + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w11_01__ * wu +
            state->cg_ah2ow__[ip + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w11_00__ * wu
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu1
            + state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_11__ * wu +
            state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_10__ * wu
            + state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w10_01__ * wu +
            state->cg_ah2ow__[ip + (itp1 + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w10_00__ * wu
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu1
            + state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_11__ * wu +
            state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_10__ * wu
            + state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w01_01__ * wu +
            state->cg_ah2ow__[ip1 + (itp + (iu1 + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w01_00__ * wu
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite1 +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu + (ite1 +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ * wu1
            + state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite +
            irh * 21) * 25) * 10) * 10 - 55111] * w00_11__ * wu +
            state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite +
            irh1 * 21) * 25) * 10) * 10 - 55111] * w00_10__ * wu
            + state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite1
            + irh * 21) * 25) * 10) * 10 - 55111] * w00_01__ * wu
            + state->cg_ah2ow__[ip1 + (itp1 + (iu1 + (ite1
            + irh1 * 21) * 25) * 10) * 10 - 55111] * w00_00__ *
            wu;


        d__2 = fa * (1.f - (1.f - a_star__) * aer_trn_ngh__[i__ + ib
            - 2]);
        d__1 = max(d__2,0.);
        abso[i__ + ib - 2] = min(d__1,1.);
        if (uvar < state->cg_min_u_h2o__) {
            uscl = uvar / state->cg_min_u_h2o__;
            abso[i__ + ib - 2] *= uscl;
        }

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        term7[i__ - 1] = state->cg_coefj__[0] +
            state->cg_coefj__[1] * dty[i__ - 1] * (
            state->cg_c16__ * dty[i__ - 1] + 1.f);
        term8[i__ - 1] = state->cg_coefk__[0] +
            state->cg_coefk__[1] * dty[i__ - 1] * (
            state->cg_c17__ * dty[i__ - 1] + 1.f);
        term7[i__] = state->cg_coefj__[3] +
            state->cg_coefj__[4] * dty[i__ - 1] * (
            state->cg_c26__ * dty[i__ - 1] + 1.f);
        term8[i__] = state->cg_coefk__[3] +
            state->cg_coefk__[4] * dty[i__ - 1] * (
            state->cg_c27__ * dty[i__ - 1] + 1.f);

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        dtym10 = dty[i__ - 1] - 10.f;
        denom = (state->cg_c30__ + state->cg_c31__ *
            dtym10 * dtym10) * sqrtu[i__ - 1] + 1.f;
        k21 = term7[i__ - 1] + term8[i__ - 1] / denom;
        denom = (state->cg_c28__ + state->cg_c29__ *
            dtym10) * sqrtu[i__ - 1] + 1.f;
        k22 = term7[i__] + term8[i__] / denom;
        tr1 = cam_exp64(-(k21 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr2 = cam_exp64(-(k22 * (sqrtu[i__ - 1] + state->cg_fc1__ *
            fwku[i__ - 1])));
        tr1 *= aer_trn_ngh__[i__ + 2];
        tr2 *= aer_trn_ngh__[i__ + 1];
        tr5 = cam_exp64(-((state->cg_coefh__[4] +
            state->cg_coefh__[5] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr6 = cam_exp64(-((state->cg_coefh__[6] +
            state->cg_coefh__[7] * dtx[i__ - 1]) * uc1[i__
            - 1]));
        tr9[i__ - 1] = tr1 * tr5;
        tr10[i__ - 1] = tr2 * tr6;
        trab2[i__ - 1] = tr9[i__ - 1] * .64999997615814209f + tr10[
            i__ - 1] * .349999994039535522f;
        th2o[i__ - 1] = tr10[i__ - 1];

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        d__1 = tbar[i__ + kn - 2] * r293;
        te = pow_dd(&d__1, &radabs_c_b43);
        dplos = (d__1 = plos[i__ + (k2 + 1) * plos_dim1] - plos[i__ +
            k2 * plos_dim1], abs(d__1));
        u1 = zinpl[i__ + kn - 2] * 18.2900009155273438f * dplos / te;
        u2 = zinpl[i__ + kn - 2] * .564899981021881104f * dplos / te;
        tlocal = tbar[i__ + kn - 2];
        tcrfac = sqrt(tlocal * r250) * te;
        beta = r3205 * (pinpl[i__ + kn - 2] * rsslp +
            state->cg_dpfo3__ * tcrfac);
        realnu = te / beta;
        tmp1 = u1 / sqrt(u1 * (realnu + 1.f) + 4.f);
        tmp2 = u2 / sqrt(u2 * (realnu + 1.f) + 4.f);
        o3bndi = te * 74.f * cam_log64(tmp1 + 1.f + tmp2);
        abso[i__ + 1] = o3bndi * o3emm[i__ + kn - 2] * (h2otr[i__ + (
            k2 + 1) * h2otr_dim1] / h2otr[i__ + k2 * h2otr_dim1]);
        to3[i__ - 1] = 1.f / (tmp1 * .100000001490116119f + 1.f +
            tmp2 * .100000001490116119f);

        }

        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        dplco2 = plco2[i__ + (k2 + 1) * plco2_dim1] - plco2[i__ + k2 *
             plco2_dim1];
        sqwp = sqrt(uinpl[i__ + kn - 2] * dplco2);
        et = cam_exp64(-480.f / tbar[i__ + kn - 2]);
        sqti[i__ - 1] = sqrt(tbar[i__ + kn - 2]);
        rsqti = 1.f / sqti[i__ - 1];
        et2 = et * et;
        et4 = et2 * et2;
        omet = 1.f - et2 * 1.5f;
        f1co2 = omet * 899.70001220703125f * (et *
            1.94773995876312256f + 1.f + et2 *
            4.73485994338989258f) * rsqti;
        f1sqwp[i__ - 1] = f1co2 * sqwp;
        t1co2[i__ - 1] = 1.f / (omet * 245.17999267578125f * sqwp *
            rsqti + 1.f);
        oneme = 1.f - et2;

        d__1 = oneme;
        alphat = d__1 * (d__1 * d__1) * rsqti;
        pi = (d__1 = dpnm[i__ - 1], abs(d__1)) * winpl[i__ + kn - 2];
        wco2 = state->cg_co2vmr__ * 2.52209997177124023f * pi *
            state->cg_rga__;
        u7[i__ - 1] = alphat * 49411.f * et2 * wco2;
        u8 = alphat * 39744.f * et4 * wco2;
        u9 = alphat * 104470.f * et4 * et2 * wco2;
        u13 = alphat * 2838.800048828125f * et4 * wco2;
        tpath = tbar[i__ + kn - 2];
        tlocal = tbar[i__ + kn - 2];
        tcrfac = sqrt(tlocal * r250 * (tpath * r300));
        posqt = (pinpl[i__ + kn - 2] * rsslp +
            state->cg_dpfco2__ * tcrfac) * rsqti;
        rbeta7[i__ - 1] = 1.f / (posqt * 5.3228001594543457f);
        rbeta8 = 1.f / (posqt * 10.6576004028320312f);
        rbeta9 = rbeta7[i__ - 1];
        rbeta13 = rbeta9;
        f2co2[i__ - 1] = u7[i__ - 1] / sqrt(u7[i__ - 1] * (rbeta7[i__
            - 1] + 1.f) + 4.f) + u8 / sqrt(u8 * (rbeta8 + 1.f) +
            4.f) + u9 / sqrt(u9 * (rbeta9 + 1.f) + 4.f);
        f3co2[i__ - 1] = u13 / sqrt(u13 * (rbeta13 + 1.f) + 4.f);
        tmp1 = cam_log64(f1sqwp[i__ - 1] + 1.f);
        tmp2 = cam_log64(f2co2[i__ - 1] + 1.f);
        tmp3 = cam_log64(f3co2[i__ - 1] + 1.f);
        absbnd = (tmp1 + t1co2[i__ - 1] * 2.f * tmp2 + tmp3 * 2.f) *
            sqti[i__ - 1];
        abso[i__ + 2] = trab2[i__ - 1] * emm[i__ + kn - 2] * absbnd;
        tco2[i__ - 1] = 1.f / (u7[i__ - 1] * 10.f / sqrt(u7[i__ - 1] *
             (rbeta7[i__ - 1] + 1.f) + 4.f) + 1.f);

        }

        trcabn_(state, lchnk, ncol, pcols, pverp, &k2, &kn, &ucfc11[
            ucfc11_offset], &ucfc12[ucfc12_offset], &un2o0[
            un2o0_offset], &un2o1[un2o1_offset], &uch4[uch4_offset], &
            uco211[uco211_offset], &uco212[uco212_offset], &uco213[
            uco213_offset], &uco221[uco221_offset], &uco222[
            uco222_offset], &uco223[uco223_offset], tbar, bplnk,
            winpl, pinpl, tco2, th2o, to3, &uptype[uptype_offset], dw,
             &s2c[s2c_offset], u, pnew, abstrc, uinpl, aer_trn_ngh__);
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        absnxt[i__ + (k2 + kn * absnxt_dim2) * absnxt_dim1] = abso[
            i__ - 1] + abso[i__] + abso[i__ + 1] + abso[i__ + 2]
            + abstrc[i__ - 1];

        }


    }


    }

    return 0;
}






__device__ int trcab_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pverp, integer *k1, integer *k2, doublereal *ucfc11,
    doublereal *ucfc12, doublereal *un2o0, doublereal *un2o1, doublereal *
    uch4, doublereal *uco211, doublereal *uco212, doublereal *uco213,
    doublereal *uco221, doublereal *uco222, doublereal *uco223,
    doublereal *bn2o0, doublereal *bn2o1, doublereal *bch4, doublereal *
    to3co2, doublereal *pnm, doublereal *dw, doublereal *pnew, doublereal
    *s2c, doublereal *uptype, doublereal *dplh2o, doublereal *abplnk1,
    doublereal *tco2, doublereal *th2o, doublereal *to3, doublereal *
    abstrc, doublereal *aer_trn_ttl__)
{
 CamScratchFrame scratch_frame(state);

    doublereal g1[6] = { .0468555986881256104,.0397454015910625458,
        .0407663993537425995,.0304380003362894058,.0540397986769676208,
        .0321962013840675354 };

    doublereal g2[6] = { 14.4832000732421875,4.30242013931274414,
        5.2352299690246582,3.25342011451721191,.698934972286224365,
        16.5599002838134766 };

    doublereal g3[6] = { 26.1898002624511719,18.4475994110107422,
        15.3633003234863281,12.1927003860473633,9.14992046356201172,
        8.07091999053955078 };

    doublereal g4[6] = { .0261781997978687286,.036951601505279541,
        .0307266004383563995,.024385400116443634,.0182932000607252121,
        .0161418002098798752 };

    doublereal ab[6] = { .0308570004999637604,.0235239993780851364,
        .017309999093413353,.0266609992831945419,.0280740000307559967,
        .0229150000959634781 };

    doublereal bb[6] = { -1.35120004415512085e-4,
        -6.83199978084303439e-5,-3.26090012094937265e-5,
        -1.02280000646715052e-5,-9.57430020207539201e-5,
        -1.03040001704357564e-4 };

    doublereal abp[6] = { .0291290003806352615,.0241010002791881561,
        .0198209993541240692,.0269039999693632126,.0294579993933439255,
        .0198919996619224548 };

    doublereal bbp[6] = { -1.3138999929651618e-4,
        -5.56880004296544939e-5,-4.63799988210666925e-5,
        -8.03619986982084811e-5,-1.01149998954497278e-4,
        -8.80610023159533739e-5 };

    integer pnm_dim1 = 0,pnm_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,bn2o0_dim1 = 0,bn2o0_offset = 0,bn2o1_dim1 = 0,bn2o1_offset = 0,bch4_dim1 = 0,bch4_offset = 0,s2c_dim1 = 0,s2c_offset = 0,uptype_dim1 = 0,uptype_offset = 0,abplnk1_dim2 = 0,abplnk1_offset = 0,aer_trn_ttl_dim1 = 0,aer_trn_ttl_dim2 = 0,aer_trn_ttl_dim3 = 0,aer_trn_ttl_offset = 0,i__1 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,l = 0;

    doublereal p1 = 0,w1 = 0,tt[1] = {},tw[6] = {},du1 = 0,du2 = 0,du01 = 0,du02 = 0,du03 = 0,du11 = 0,du12 = 0,du13 = 0,du21 = 0,du22 = 0,du23 = 0,tlw = 0,ach4 = 0,ds2c[1] = {},phi1 = 0,tch4 = 0,psi1 = 0,aco21 = 0,aco22 = 0,an2o1 = 0,an2o2 = 0,an2o3 = 0,acfc1 = 0,acfc2 = 0,acfc3 = 0,sqti[1] = {},acfc4 = 0,acfc5 = 0,acfc6 = 0,acfc7 = 0,acfc8 = 0,tcfc3 = 0,tcfc4 = 0,tcfc6 = 0,tcfc7 = 0,duch4 = 0,tcfc8 = 0,dbetc1 = 0,dbetc2 = 0,dbeta01 = 0,dbeta11 = 0,dbeta02 = 0,dbeta03 = 0,dbetac = 0,duptyp[1] = {};


    --abstrc;
    --to3;
    --th2o;
    --tco2;
    --dplh2o;
    --pnew;
    --dw;
    --to3co2;
    aer_trn_ttl_dim1 = 1;
    aer_trn_ttl_dim2 = (CAM_LEVELS+1);
    aer_trn_ttl_dim3 = (CAM_LEVELS+1);
    aer_trn_ttl_offset = 1 + aer_trn_ttl_dim1 * (1 + aer_trn_ttl_dim2 * (1 +
        aer_trn_ttl_dim3));
    aer_trn_ttl__ -= aer_trn_ttl_offset;
    abplnk1_dim2 = 1;
    abplnk1_offset = 1 + 14 * (1 + abplnk1_dim2);
    abplnk1 -= abplnk1_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;
    bch4_dim1 = 1;
    bch4_offset = 1 + bch4_dim1;
    bch4 -= bch4_offset;
    bn2o1_dim1 = 1;
    bn2o1_offset = 1 + bn2o1_dim1;
    bn2o1 -= bn2o1_offset;
    bn2o0_dim1 = 1;
    bn2o0_offset = 1 + bn2o0_dim1;
    bn2o0 -= bn2o0_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    sqti[i__ - 1] = sqrt(to3co2[i__]);
    tt[i__ - 1] = (d__1 = to3co2[i__] - 250.f, abs(d__1));
    ds2c[i__ - 1] = (d__1 = s2c[i__ + *k1 * s2c_dim1] - s2c[i__ + *k2 *
        s2c_dim1], abs(d__1));
    duptyp[i__ - 1] = (d__1 = uptype[i__ + *k1 * uptype_dim1] - uptype[
        i__ + *k2 * uptype_dim1], abs(d__1));

    }

    for (l = 1; l <= 6; ++l) {
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        psi1 = cam_exp64(abp[l - 1] * tt[i__ - 1] + bbp[l - 1] * tt[i__ - 1] *
            tt[i__ - 1]);
        phi1 = cam_exp64(ab[l - 1] * tt[i__ - 1] + bb[l - 1] * tt[i__ - 1] * tt[
            i__ - 1]);
        p1 = pnew[i__] * (psi1 / phi1) / state->cg_sslp__;
        w1 = dw[i__] * phi1;
        tw[i__ + l - 2] = cam_exp64(-g1[l - 1] * p1 * (sqrt(g2[l - 1] * (w1 /
            p1) + 1.f) - 1.f) - g3[l - 1] * ds2c[i__ - 1] - g4[l - 1]
            * duptyp[i__ - 1]);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tw[i__ - 1] *= aer_trn_ttl__[i__ + (*k1 + (*k2 + (aer_trn_ttl_dim3 <<
        2)) * aer_trn_ttl_dim2) * aer_trn_ttl_dim1] *
        .699999988079071045f + aer_trn_ttl__[i__ + (*k1 + (*k2 +
        aer_trn_ttl_dim3 * 5) * aer_trn_ttl_dim2) * aer_trn_ttl_dim1]
        * .300000011920928955f;
    tw[i__] *= aer_trn_ttl__[i__ + (*k1 + (*k2 + aer_trn_ttl_dim3 * 5) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 1] *= aer_trn_ttl__[i__ + (*k1 + (*k2 + aer_trn_ttl_dim3 * 5)
         * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 2] *= aer_trn_ttl__[i__ + (*k1 + (*k2 + aer_trn_ttl_dim3 * 5)
         * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 3] *= aer_trn_ttl__[i__ + (*k1 + (*k2 + aer_trn_ttl_dim3 * 6)
         * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    tw[i__ + 4] *= aer_trn_ttl__[i__ + (*k1 + (*k2 + aer_trn_ttl_dim3 * 6)
         * aer_trn_ttl_dim2) * aer_trn_ttl_dim1];

    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    du1 = (d__1 = ucfc11[i__ + *k1 * ucfc11_dim1] - ucfc11[i__ + *k2 *
        ucfc11_dim1], abs(d__1));
    du2 = (d__1 = ucfc12[i__ + *k1 * ucfc12_dim1] - ucfc12[i__ + *k2 *
        ucfc12_dim1], abs(d__1));
    tcfc3 = cam_exp64(du1 * -175.0050048828125f);
    tcfc4 = cam_exp64(du1 * -1202.1800537109375f);
    tcfc6 = cam_exp64(du2 * -5786.72998046875f);
    tcfc7 = cam_exp64(du2 * -2873.510009765625f);
    tcfc8 = cam_exp64(du2 * -2085.590087890625f);
    acfc1 = (1.f - cam_exp64(du1 * -54.0900001525878906f)) * 50.f * tw[i__ - 1]
        * abplnk1[(i__ + *k2 * abplnk1_dim2) * 14 + 7];
    acfc2 = (1.f - cam_exp64(du1 * -5130.02978515625f)) * 60.f * tw[i__] *
        abplnk1[(i__ + *k2 * abplnk1_dim2) * 14 + 8];
    acfc3 = (1.f - tcfc3) * 60.f * tw[i__ + 2] * tcfc6 * abplnk1[(i__ + *
        k2 * abplnk1_dim2) * 14 + 9];
    acfc4 = (1.f - tcfc4) * 100.f * tw[i__ + 3] * abplnk1[(i__ + *k2 *
        abplnk1_dim2) * 14 + 10];
    acfc5 = (1.f - cam_exp64(du2 * -1272.3499755859375f)) * 45.f * tw[i__ + 1] *
         abplnk1[(i__ + *k2 * abplnk1_dim2) * 14 + 11];
    acfc6 = (1.f - tcfc6) * 50.f * tw[i__ + 2] * abplnk1[(i__ + *k2 *
        abplnk1_dim2) * 14 + 12];
    acfc7 = (1.f - tcfc7) * 80.f * tw[i__ + 3] * tcfc4 * abplnk1[(i__ + *
        k2 * abplnk1_dim2) * 14 + 13];
    acfc8 = (1.f - tcfc8) * 70.f * tw[i__ + 4] * abplnk1[(i__ + *k2 *
        abplnk1_dim2) * 14 + 14];
    tlw = cam_exp64(sqrt(dplh2o[i__]) * -1.f);
    tlw *= aer_trn_ttl__[i__ + (*k1 + (*k2 + aer_trn_ttl_dim3 * 7) *
        aer_trn_ttl_dim2) * aer_trn_ttl_dim1];
    duch4 = (d__1 = uch4[i__ + *k1 * uch4_dim1] - uch4[i__ + *k2 *
        uch4_dim1], abs(d__1));
    dbetac = (d__1 = bch4[i__ + *k1 * bch4_dim1] - bch4[i__ + *k2 *
        bch4_dim1], abs(d__1)) / duch4;
    ach4 = sqti[i__ - 1] * 6.0044398307800293f * cam_log64(duch4 / sqrt(4.f +
        duch4 * (1.f + 1.f / dbetac)) + 1.f) * tlw * abplnk1[(i__ + *
        k2 * abplnk1_dim2) * 14 + 3];
    tch4 = 1.f / (duch4 / sqrt(4.f + duch4 * (1.f + 1.f / dbetac)) *
        .0199999995529651642f + 1.f);
    du01 = (d__1 = un2o0[i__ + *k1 * un2o0_dim1] - un2o0[i__ + *k2 *
        un2o0_dim1], abs(d__1));
    du11 = (d__1 = un2o1[i__ + *k1 * un2o1_dim1] - un2o1[i__ + *k2 *
        un2o1_dim1], abs(d__1));
    dbeta01 = (d__1 = bn2o0[i__ + *k1 * bn2o0_dim1] - bn2o0[i__ + *k2 *
        bn2o0_dim1], abs(d__1)) / du01;
    dbeta11 = (d__1 = bn2o1[i__ + *k1 * bn2o1_dim1] - bn2o1[i__ + *k2 *
        bn2o1_dim1], abs(d__1)) / du11;
    an2o1 = sqti[i__ - 1] * 2.35558009147644043f * cam_log64(du01 / sqrt(4.f +
        du01 * (1.f + 1.f / dbeta01)) + 1.f + du11 / sqrt(4.f + du11 *
         (1.f + 1.f / dbeta11))) * tlw * tch4 * abplnk1[(i__ + *k2 *
        abplnk1_dim2) * 14 + 4];
    du02 = du01 * .100089997053146362f;
    du12 = du11 * .0992745980620384216f;
    dbeta02 = dbeta01 * .964281976222991943f;
    an2o2 = sqti[i__ - 1] * 2.65581011772155762f * cam_log64(du02 / sqrt(4.f +
        du02 * (1.f + 1.f / dbeta02)) + 1.f + du12 / sqrt(4.f + du12 *
         (1.f + 1.f / dbeta02))) * th2o[i__] * tco2[i__] * abplnk1[(
        i__ + *k2 * abplnk1_dim2) * 14 + 5];
    du03 = du01 * .0333767011761665344f;
    dbeta03 = dbeta01 * .982142984867095947f;
    an2o3 = sqti[i__ - 1] * 2.54033994674682617f * cam_log64(du03 / sqrt(4.f +
        du03 * (1.f + 1.f / dbeta03)) + 1.f) * tw[i__ + 4] * tcfc8 *
        abplnk1[(i__ + *k2 * abplnk1_dim2) * 14 + 6];
    du11 = (d__1 = uco211[i__ + *k1 * uco211_dim1] - uco211[i__ + *k2 *
        uco211_dim1], abs(d__1));
    du12 = (d__1 = uco212[i__ + *k1 * uco212_dim1] - uco212[i__ + *k2 *
        uco212_dim1], abs(d__1));
    du13 = (d__1 = uco213[i__ + *k1 * uco213_dim1] - uco213[i__ + *k2 *
        uco213_dim1], abs(d__1));
    dbetc1 = (d__1 = pnm[i__ + *k1 * pnm_dim1] + pnm[i__ + *k2 * pnm_dim1]
        , abs(d__1)) * 2.97557997703552246f / (state->cg_sslp__
        * 2.f * sqti[i__ - 1]);
    dbetc2 = dbetc1 * 2.f;
    aco21 = sqti[i__ - 1] * 3.75710010528564453f * cam_log64(du11 / sqrt(4.f +
        du11 * (1.f + 1.f / dbetc1)) + 1.f + du12 / sqrt(4.f + du12 *
        (1.f + 1.f / dbetc2)) + du13 / sqrt(4.f + du13 * (1.f + 1.f /
        dbetc2))) * to3[i__] * tw[i__ + 3] * tcfc4 * tcfc7 * abplnk1[(
        i__ + *k2 * abplnk1_dim2) * 14 + 2];
    du21 = (d__1 = uco221[i__ + *k1 * uco221_dim1] - uco221[i__ + *k2 *
        uco221_dim1], abs(d__1));
    du22 = (d__1 = uco222[i__ + *k1 * uco222_dim1] - uco222[i__ + *k2 *
        uco222_dim1], abs(d__1));
    du23 = (d__1 = uco223[i__ + *k1 * uco223_dim1] - uco223[i__ + *k2 *
        uco223_dim1], abs(d__1));
    aco22 = sqti[i__ - 1] * 3.8443000316619873f * cam_log64(du21 / sqrt(4.f +
        du21 * (1.f + 1.f / dbetc1)) + 1.f + du22 / sqrt(4.f + du22 *
        (1.f + 1.f / dbetc1)) + du23 / sqrt(4.f + du23 * (1.f + 1.f /
        dbetc2))) * tw[i__ + 2] * tcfc3 * tcfc6 * abplnk1[(i__ + *k2 *
         abplnk1_dim2) * 14 + 1];
    abstrc[i__] = acfc1 + acfc2 + acfc3 + acfc4 + acfc5 + acfc6 + acfc7 +
        acfc8 + an2o1 + an2o2 + an2o3 + ach4 + aco21 + aco22;

    }

    return 0;
}





__device__ int trcabn_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pverp, integer *k2, integer *kn, doublereal *ucfc11,
    doublereal *ucfc12, doublereal *un2o0, doublereal *un2o1, doublereal *
    uch4, doublereal *uco211, doublereal *uco212, doublereal *uco213,
    doublereal *uco221, doublereal *uco222, doublereal *uco223,
    doublereal *tbar, doublereal *bplnk, doublereal *winpl, doublereal *
    pinpl, doublereal *tco2, doublereal *th2o, doublereal *to3,
    doublereal *uptype, doublereal *dw, doublereal *s2c, doublereal *up2,
    doublereal *pnew, doublereal *abstrc, doublereal *uinpl, doublereal *
    aer_trn_ngh__)
{
 CamScratchFrame scratch_frame(state);

    doublereal g1[6] = { .0468555986881256104,.0397454015910625458,
        .0407663993537425995,.0304380003362894058,.0540397986769676208,
        .0321962013840675354 };

    doublereal g2[6] = { 14.4832000732421875,4.30242013931274414,
        5.2352299690246582,3.25342011451721191,.698934972286224365,
        16.5599002838134766 };

    doublereal g3[6] = { 26.1898002624511719,18.4475994110107422,
        15.3633003234863281,12.1927003860473633,9.14992046356201172,
        8.07091999053955078 };

    doublereal g4[6] = { .0261781997978687286,.036951601505279541,
        .0307266004383563995,.024385400116443634,.0182932000607252121,
        .0161418002098798752 };

    doublereal ab[6] = { .0308570004999637604,.0235239993780851364,
        .017309999093413353,.0266609992831945419,.0280740000307559967,
        .0229150000959634781 };

    doublereal bb[6] = { -1.35120004415512085e-4,
        -6.83199978084303439e-5,-3.26090012094937265e-5,
        -1.02280000646715052e-5,-9.57430020207539201e-5,
        -1.03040001704357564e-4 };

    doublereal abp[6] = { .0291290003806352615,.0241010002791881561,
        .0198209993541240692,.0269039999693632126,.0294579993933439255,
        .0198919996619224548 };

    doublereal bbp[6] = { -1.3138999929651618e-4,
        -5.56880004296544939e-5,-4.63799988210666925e-5,
        -8.03619986982084811e-5,-1.01149998954497278e-4,
        -8.80610023159533739e-5 };

    integer tbar_dim1 = 0,tbar_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,bplnk_dim2 = 0,bplnk_offset = 0,winpl_dim1 = 0,winpl_offset = 0,pinpl_dim1 = 0,pinpl_offset = 0,s2c_dim1 = 0,s2c_offset = 0,uptype_dim1 = 0,uptype_offset = 0,uinpl_dim1 = 0,uinpl_offset = 0,aer_trn_ngh_dim1 = 0,aer_trn_ngh_offset = 0,i__1 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,l = 0;

    doublereal p1 = 0,w1 = 0,tt[1] = {},tw[6] = {},du1 = 0,du2 = 0,du01 = 0,du02 = 0,du03 = 0,du11 = 0,du12 = 0,du13 = 0,du21 = 0,du22 = 0,du23 = 0,tlw = 0,ach4 = 0,ds2c[1] = {},phi1 = 0,tch4 = 0,psi1 = 0,aco21 = 0,aco22 = 0,an2o1 = 0,an2o2 = 0,an2o3 = 0,acfc1 = 0,acfc2 = 0,acfc3 = 0,sqti[1] = {},acfc4 = 0,acfc5 = 0,acfc6 = 0,acfc7 = 0,acfc8 = 0,tcfc3 = 0,tcfc4 = 0,tcfc6 = 0,tcfc7 = 0,duch4 = 0,tcfc8 = 0,rsqti[1] = {},dbetc1 = 0,dbetc2 = 0,dbeta01 = 0,dbeta11 = 0,dbeta02 = 0,dbeta03 = 0,dbetac = 0,duptyp[1] = {};


    aer_trn_ngh_dim1 = 1;
    aer_trn_ngh_offset = 1 + aer_trn_ngh_dim1;
    aer_trn_ngh__ -= aer_trn_ngh_offset;
    uinpl_dim1 = 1;
    uinpl_offset = 1 + uinpl_dim1;
    uinpl -= uinpl_offset;
    --abstrc;
    --pnew;
    --up2;
    --dw;
    --to3;
    --th2o;
    --tco2;
    pinpl_dim1 = 1;
    pinpl_offset = 1 + pinpl_dim1;
    pinpl -= pinpl_offset;
    winpl_dim1 = 1;
    winpl_offset = 1 + winpl_dim1;
    winpl -= winpl_offset;
    bplnk_dim2 = 1;
    bplnk_offset = 1 + 14 * (1 + bplnk_dim2);
    bplnk -= bplnk_offset;
    tbar_dim1 = 1;
    tbar_offset = 1 + tbar_dim1;
    tbar -= tbar_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    sqti[i__ - 1] = sqrt(tbar[i__ + *kn * tbar_dim1]);
    rsqti[i__ - 1] = 1.f / sqti[i__ - 1];
    tt[i__ - 1] = (d__1 = tbar[i__ + *kn * tbar_dim1] - 250.f, abs(d__1));
    ds2c[i__ - 1] = (d__1 = s2c[i__ + (*k2 + 1) * s2c_dim1] - s2c[i__ + *
        k2 * s2c_dim1], abs(d__1)) * uinpl[i__ + *kn * uinpl_dim1];
    duptyp[i__ - 1] = (d__1 = uptype[i__ + (*k2 + 1) * uptype_dim1] -
        uptype[i__ + *k2 * uptype_dim1], abs(d__1)) * uinpl[i__ + *kn
        * uinpl_dim1];

    }

    for (l = 1; l <= 6; ++l) {
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        psi1 = cam_exp64(abp[l - 1] * tt[i__ - 1] + bbp[l - 1] * tt[i__ - 1] *
            tt[i__ - 1]);
        phi1 = cam_exp64(ab[l - 1] * tt[i__ - 1] + bb[l - 1] * tt[i__ - 1] * tt[
            i__ - 1]);
        p1 = pnew[i__] * (psi1 / phi1) / state->cg_sslp__;
        w1 = dw[i__] * winpl[i__ + *kn * winpl_dim1] * phi1;
        tw[i__ + l - 2] = cam_exp64(-g1[l - 1] * p1 * (sqrt(g2[l - 1] * (w1 /
            p1) + 1.f) - 1.f) - g3[l - 1] * ds2c[i__ - 1] - g4[l - 1]
            * duptyp[i__ - 1]);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tw[i__ - 1] *= aer_trn_ngh__[i__ + (aer_trn_ngh_dim1 << 2)] *
        .699999988079071045f + aer_trn_ngh__[i__ + aer_trn_ngh_dim1 *
        5] * .300000011920928955f;
    tw[i__] *= aer_trn_ngh__[i__ + aer_trn_ngh_dim1 * 5];
    tw[i__ + 1] *= aer_trn_ngh__[i__ + aer_trn_ngh_dim1 * 5];
    tw[i__ + 2] *= aer_trn_ngh__[i__ + aer_trn_ngh_dim1 * 5];
    tw[i__ + 3] *= aer_trn_ngh__[i__ + aer_trn_ngh_dim1 * 6];
    tw[i__ + 4] *= aer_trn_ngh__[i__ + aer_trn_ngh_dim1 * 6];

    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    du1 = (d__1 = ucfc11[i__ + (*k2 + 1) * ucfc11_dim1] - ucfc11[i__ + *
        k2 * ucfc11_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    du2 = (d__1 = ucfc12[i__ + (*k2 + 1) * ucfc12_dim1] - ucfc12[i__ + *
        k2 * ucfc12_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    tcfc3 = cam_exp64(du1 * -175.0050048828125f);
    tcfc4 = cam_exp64(du1 * -1202.1800537109375f);
    tcfc6 = cam_exp64(du2 * -5786.72998046875f);
    tcfc7 = cam_exp64(du2 * -2873.510009765625f);
    tcfc8 = cam_exp64(du2 * -2085.590087890625f);
    acfc1 = (1.f - cam_exp64(du1 * -54.0900001525878906f)) * 50.f * tw[i__ - 1]
        * bplnk[(i__ + *kn * bplnk_dim2) * 14 + 7];
    acfc2 = (1.f - cam_exp64(du1 * -5130.02978515625f)) * 60.f * tw[i__] *
        bplnk[(i__ + *kn * bplnk_dim2) * 14 + 8];
    acfc3 = (1.f - tcfc3) * 60.f * tw[i__ + 2] * tcfc6 * bplnk[(i__ + *kn
        * bplnk_dim2) * 14 + 9];
    acfc4 = (1.f - tcfc4) * 100.f * tw[i__ + 3] * bplnk[(i__ + *kn *
        bplnk_dim2) * 14 + 10];
    acfc5 = (1.f - cam_exp64(du2 * -1272.3499755859375f)) * 45.f * tw[i__ + 1] *
         bplnk[(i__ + *kn * bplnk_dim2) * 14 + 11];
    acfc6 = (1.f - tcfc6) * 50.f * tw[i__ + 2] * bplnk[(i__ + *kn *
        bplnk_dim2) * 14 + 12];
    acfc7 = (1.f - tcfc7) * 80.f * tw[i__ + 3] * tcfc4 * bplnk[(i__ + *kn
        * bplnk_dim2) * 14 + 13];
    acfc8 = (1.f - tcfc8) * 70.f * tw[i__ + 4] * bplnk[(i__ + *kn *
        bplnk_dim2) * 14 + 14];
    tlw = cam_exp64(sqrt(up2[i__]) * -1.f);
    tlw *= aer_trn_ngh__[i__ + aer_trn_ngh_dim1 * 7];
    duch4 = (d__1 = uch4[i__ + (*k2 + 1) * uch4_dim1] - uch4[i__ + *k2 *
        uch4_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    dbetac = pinpl[i__ + *kn * pinpl_dim1] * 2.94448995590209961f * rsqti[
        i__ - 1] / state->cg_sslp__;
    ach4 = sqti[i__ - 1] * 6.0044398307800293f * cam_log64(duch4 / sqrt(4.f +
        duch4 * (1.f + 1.f / dbetac)) + 1.f) * tlw * bplnk[(i__ + *kn
        * bplnk_dim2) * 14 + 3];
    tch4 = 1.f / (duch4 / sqrt(4.f + duch4 * (1.f + 1.f / dbetac)) *
        .0199999995529651642f + 1.f);
    du01 = (d__1 = un2o0[i__ + (*k2 + 1) * un2o0_dim1] - un2o0[i__ + *k2 *
         un2o0_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    du11 = (d__1 = un2o1[i__ + (*k2 + 1) * un2o1_dim1] - un2o1[i__ + *k2 *
         un2o1_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    dbeta01 = pinpl[i__ + *kn * pinpl_dim1] * 19.3990001678466797f *
        rsqti[i__ - 1] / state->cg_sslp__;
    dbeta11 = dbeta01;
    an2o1 = sqti[i__ - 1] * 2.35558009147644043f * cam_log64(du01 / sqrt(4.f +
        du01 * (1.f + 1.f / dbeta01)) + 1.f + du11 / sqrt(4.f + du11 *
         (1.f + 1.f / dbeta11))) * tlw * tch4 * bplnk[(i__ + *kn *
        bplnk_dim2) * 14 + 4];
    du02 = du01 * .100089997053146362f;
    du12 = du11 * .0992745980620384216f;
    dbeta02 = dbeta01 * .964281976222991943f;
    an2o2 = sqti[i__ - 1] * 2.65581011772155762f * cam_log64(du02 / sqrt(4.f +
        du02 * (1.f + 1.f / dbeta02)) + 1.f + du12 / sqrt(4.f + du12 *
         (1.f + 1.f / dbeta02))) * tco2[i__] * th2o[i__] * bplnk[(i__
        + *kn * bplnk_dim2) * 14 + 5];
    du03 = du01 * .0333767011761665344f;
    dbeta03 = dbeta01 * .982142984867095947f;
    an2o3 = sqti[i__ - 1] * 2.54033994674682617f * cam_log64(du03 / sqrt(4.f +
        du03 * (1.f + 1.f / dbeta03)) + 1.f) * tw[i__ + 4] * tcfc8 *
        bplnk[(i__ + *kn * bplnk_dim2) * 14 + 6];
    du11 = (d__1 = uco211[i__ + (*k2 + 1) * uco211_dim1] - uco211[i__ + *
        k2 * uco211_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    du12 = (d__1 = uco212[i__ + (*k2 + 1) * uco212_dim1] - uco212[i__ + *
        k2 * uco212_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    du13 = (d__1 = uco213[i__ + (*k2 + 1) * uco213_dim1] - uco213[i__ + *
        k2 * uco213_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    dbetc1 = pinpl[i__ + *kn * pinpl_dim1] * 2.97557997703552246f * rsqti[
        i__ - 1] / state->cg_sslp__;
    dbetc2 = dbetc1 * 2.f;
    aco21 = sqti[i__ - 1] * 3.75710010528564453f * cam_log64(du11 / sqrt(4.f +
        du11 * (1.f + 1.f / dbetc1)) + 1.f + du12 / sqrt(4.f + du12 *
        (1.f + 1.f / dbetc2)) + du13 / sqrt(4.f + du13 * (1.f + 1.f /
        dbetc2))) * to3[i__] * tw[i__ + 3] * tcfc4 * tcfc7 * bplnk[(
        i__ + *kn * bplnk_dim2) * 14 + 2];
    du21 = (d__1 = uco221[i__ + (*k2 + 1) * uco221_dim1] - uco221[i__ + *
        k2 * uco221_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    du22 = (d__1 = uco222[i__ + (*k2 + 1) * uco222_dim1] - uco222[i__ + *
        k2 * uco222_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    du23 = (d__1 = uco223[i__ + (*k2 + 1) * uco223_dim1] - uco223[i__ + *
        k2 * uco223_dim1], abs(d__1)) * winpl[i__ + *kn * winpl_dim1];
    aco22 = sqti[i__ - 1] * 3.8443000316619873f * cam_log64(du21 / sqrt(4.f +
        du21 * (1.f + 1.f / dbetc1)) + 1.f + du22 / sqrt(4.f + du22 *
        (1.f + 1.f / dbetc1)) + du23 / sqrt(4.f + du23 * (1.f + 1.f /
        dbetc2))) * tw[i__ + 2] * tcfc3 * tcfc6 * bplnk[(i__ + *kn *
        bplnk_dim2) * 14 + 1];
    abstrc[i__] = acfc1 + acfc2 + acfc3 + acfc4 + acfc5 + acfc6 + acfc7 +
        acfc8 + an2o1 + an2o2 + an2o3 + ach4 + aco21 + aco22;

    }

    return 0;
}





__device__ int radclwmx_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *lwupcgs, doublereal *tnm,
    doublereal *qnm, doublereal *o3vmr, doublereal *pmid, doublereal *
    pint, doublereal *pmln, doublereal *piln, doublereal *n2o, doublereal
    *ch4, doublereal *cfc11, doublereal *cfc12, doublereal *cld,
    doublereal *emis, doublereal *pmxrgn, integer *nmxrgn, doublereal *
    qrl, doublereal *qrlcs, logical *doabsems, doublereal *abstot,
    doublereal *absnxt, doublereal *emstot, doublereal *flns, doublereal *
    flnt, doublereal *flnsc, doublereal *flntc, doublereal *flwds,
    doublereal *flut, doublereal *flutc, doublereal *flup, doublereal *
    flupc, doublereal *fldn, doublereal *fldnc, doublereal *aer_mass__)
{
 CamScratchFrame scratch_frame(state);

    integer pmxrgn_dim1 = 0,pmxrgn_offset = 0,tnm_dim1 = 0,tnm_offset = 0,qnm_dim1 = 0,qnm_offset = 0,o3vmr_dim1 = 0,o3vmr_offset = 0,pmid_dim1 = 0,pmid_offset = 0,pint_dim1 = 0,pint_offset = 0,pmln_dim1 = 0,pmln_offset = 0,piln_dim1 = 0,piln_offset = 0,n2o_dim1 = 0,n2o_offset = 0,ch4_dim1 = 0,ch4_offset = 0,cfc11_dim1 = 0,cfc11_offset = 0,cfc12_dim1 = 0,cfc12_offset = 0,cld_dim1 = 0,cld_offset = 0,emis_dim1 = 0,emis_offset = 0,aer_mass_dim1 = 0,aer_mass_offset = 0,qrl_dim1 = 0,qrl_offset = 0,qrlcs_dim1 = 0,qrlcs_offset = 0,flup_dim1 = 0,flup_offset = 0,flupc_dim1 = 0,flupc_offset = 0,fldn_dim1 = 0,fldn_offset = 0,fldnc_dim1 = 0,fldnc_offset = 0,abstot_dim1 = 0,abstot_dim2 = 0,abstot_offset = 0,absnxt_dim1 = 0,absnxt_dim2 = 0,absnxt_offset = 0,emstot_dim1 = 0,emstot_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0,i__5 = 0,i__6 = 0,i__7 = 0,i__8 = 0;

    doublereal d__1 = 0,d__2 = 0;

    integer i__ = 0,k = 0,l = 0,n = 0,cam_red_i__ = 0,cam_red_k__ = 0;

    doublereal *s = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *w = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   ;

    integer k1 = 0,k2 = 0,k3 = 0,l1 = 0;

    integer ii = 0,km = 0;

    doublereal *wb = cam_alloc<doublereal>(state, (2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *bk1 = cam_alloc<doublereal>(state, 1), *bk2 = cam_alloc<doublereal>(state, 1);

    integer km1 = 0,km3 = 0,km4 = 0;

    doublereal *s2c = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    ;

    integer *kx1 = cam_alloc<integer>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *kx2 = cam_alloc<integer>(state, (1-(1)+1)*((CAM_LEVELS+1)-(0)+1))    ;

    doublereal *fdl = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , *tcg = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)) , *ful = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)) , *emx = cam_alloc<doublereal>(state, ((CAM_LEVELS+1)-(0)+1)), *tmp = cam_alloc<doublereal>(state, 1);

    integer *kxs = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *nxs = cam_alloc<integer>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    ;

    doublereal *bch4 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , cld0, cld1, *uch4 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , *co2t = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , emx0,
        *aer_trn_ttl__ = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))    , *tmp2 = cam_alloc<doublereal>(state, 1), *bn2o0 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))
        , *bn2o1 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *cldp = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))
        , *delt = cam_alloc<doublereal>(state, 1);

    integer ilon = 0,iimx = 0,irgn = 0;

    doublereal *plol = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *plos = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , *fsul = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , *fsdl = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , *tint = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , atmp;

    integer *indx = cam_alloc<integer>(state, 1),*nrgn = cam_alloc<integer>(state, 1),npts = 0,nxsk = 0;

    doublereal *fclb4 = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(-1)+1)) ;

    integer ktmp = 0;

    doublereal *un2o0 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *un2o1 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *co2em = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *delt1 = cam_alloc<doublereal>(state, 1), *uco211 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    , *uco212 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *fclt4 = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(0)+1))   , *uco213 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *uco221 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *plco2 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *uco222 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *uco223 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *plh2o = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *h2otr = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *tint4 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *ucfc11 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *ucfc12 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *absbt = cam_alloc<doublereal>(state, 1), trans, *asort = cam_alloc<doublereal>(state, (CAM_LEVELS-(1)+1)), *tlayr = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   ;

    integer *ksort = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1));

    doublereal *co2eml = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)) , *plh2ob = cam_alloc<doublereal>(state, (2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))    ;

    integer pverp2 = 0,pverp3 = 0,pverp4 = 0;

    doublereal *tlayr4 = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)) ;

    doublereal *maxcld = cam_alloc<doublereal>(state, 1);

    doublereal *tplnka = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)) , *tplnke = cam_alloc<doublereal>(state, 1);

    integer *ncolmx = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(1)+1));

    integer *indxmx = cam_alloc<integer>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   ;

    doublereal *uptype = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)) , *abplnk1 = cam_alloc<doublereal>(state, (14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  , *abplnk2 = cam_alloc<doublereal>(state, (14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  ;

    integer cam_s1_0__ = 0,cam_s1_1__ = 0,cam_s2_0__ = 0,cam_s3_0__ = 0,cam_s3_1__ = 0,cam_s4_0__ = 0,cam_s4_1__ = 0,cam_s5_0__ = 0,cam_s5_1__ = 0,cam_s6_0__ = 0,cam_s6_1__ = 0,cam_s7_0__ = 0,cam_s7_1__ = 0,cam_s8_0__ = 0,cam_s8_1__ = 0;

    doublereal *aer_mpp__ = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))  ;


    --flutc;
    --flut;
    --flwds;
    --flntc;
    --flnsc;
    --flnt;
    --flns;
    --nmxrgn;
    --lwupcgs;
    aer_mass_dim1 = 1;
    aer_mass_offset = 1 + aer_mass_dim1;
    aer_mass__ -= aer_mass_offset;
    absnxt_dim1 = 1;
    absnxt_dim2 = CAM_LEVELS;
    absnxt_offset = 1 + absnxt_dim1 * (1 + absnxt_dim2);
    absnxt -= absnxt_offset;
    qrlcs_dim1 = 1;
    qrlcs_offset = 1 + qrlcs_dim1;
    qrlcs -= qrlcs_offset;
    qrl_dim1 = 1;
    qrl_offset = 1 + qrl_dim1;
    qrl -= qrl_offset;
    emis_dim1 = 1;
    emis_offset = 1 + emis_dim1;
    emis -= emis_offset;
    cld_dim1 = 1;
    cld_offset = 1 + cld_dim1;
    cld -= cld_offset;
    cfc12_dim1 = 1;
    cfc12_offset = 1 + cfc12_dim1;
    cfc12 -= cfc12_offset;
    cfc11_dim1 = 1;
    cfc11_offset = 1 + cfc11_dim1;
    cfc11 -= cfc11_offset;
    ch4_dim1 = 1;
    ch4_offset = 1 + ch4_dim1;
    ch4 -= ch4_offset;
    n2o_dim1 = 1;
    n2o_offset = 1 + n2o_dim1;
    n2o -= n2o_offset;
    pmln_dim1 = 1;
    pmln_offset = 1 + pmln_dim1;
    pmln -= pmln_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    o3vmr_dim1 = 1;
    o3vmr_offset = 1 + o3vmr_dim1;
    o3vmr -= o3vmr_offset;
    qnm_dim1 = 1;
    qnm_offset = 1 + qnm_dim1;
    qnm -= qnm_offset;
    tnm_dim1 = 1;
    tnm_offset = 1 + tnm_dim1;
    tnm -= tnm_offset;
    fldnc_dim1 = 1;
    fldnc_offset = 1 + fldnc_dim1;
    fldnc -= fldnc_offset;
    fldn_dim1 = 1;
    fldn_offset = 1 + fldn_dim1;
    fldn -= fldn_offset;
    flupc_dim1 = 1;
    flupc_offset = 1 + flupc_dim1;
    flupc -= flupc_offset;
    flup_dim1 = 1;
    flup_offset = 1 + flup_dim1;
    flup -= flup_offset;
    emstot_dim1 = 1;
    emstot_offset = 1 + emstot_dim1;
    emstot -= emstot_offset;
    abstot_dim1 = 1;
    abstot_dim2 = (CAM_LEVELS+1);
    abstot_offset = 1 + abstot_dim1 * (1 + abstot_dim2);
    abstot -= abstot_offset;
    pmxrgn_dim1 = 1;
    pmxrgn_offset = 1 + pmxrgn_dim1;
    pmxrgn -= pmxrgn_offset;
    piln_dim1 = 1;
    piln_offset = 1 + piln_dim1;
    piln -= piln_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    pverp2 = CAM_LEVELS + 2;
    pverp3 = CAM_LEVELS + 3;
    pverp4 = CAM_LEVELS + 4;
    aer_pth_(state, &aer_mass__[aer_mass_offset], aer_mpp__, ncol, pcols, pver,
        pverp);
    radtpl_(state, lchnk, ncol, pcols, pver, pverp, &tnm[tnm_offset], &lwupcgs[1], &
        qnm[qnm_offset], &pint[pint_offset], plco2, plh2o, tplnka, s2c,
        tcg, w, tplnke, tint, tint4, tlayr, tlayr4, &pmln[pmln_offset], &
        piln[piln_offset], plh2ob, wb);
    if (*doabsems) {
    if (state->abs_phase != 2) {
    radoz2_(state, lchnk, ncol, pcols, pver, pverp, &o3vmr[o3vmr_offset], &pint[
        pint_offset], plol, plos, &state->cg_ntoplw__);
    trcpth_(state, lchnk, ncol, pcols, pver, pverp, &tnm[tnm_offset], &pint[
        pint_offset], &cfc11[cfc11_offset], &cfc12[cfc12_offset], &
        n2o[n2o_offset], &ch4[ch4_offset], &qnm[qnm_offset], ucfc11,
        ucfc12, un2o0, un2o1, uch4, uco211, uco212, uco213, uco221,
        uco222, uco223, bn2o0, bn2o1, bch4, uptype);
    aer_trn_(state, aer_mpp__, aer_trn_ttl__, pcols, pver, pverp);
    radems_(state, lchnk, ncol, pcols, pver, pverp, s2c, tcg, w, tplnke, plh2o, &
        pint[pint_offset], plco2, tint, tint4, tlayr, tlayr4, plol,
        plos, ucfc11, ucfc12, un2o0, un2o1, uch4, uco211, uco212,
        uco213, uco221, uco222, uco223, uptype, bn2o0, bn2o1, bch4,
        co2em, co2eml, co2t, h2otr, abplnk1, abplnk2, &emstot[
        emstot_offset], plh2ob, wb, aer_trn_ttl__);
    }
    if (state->abs_phase == 1)
        for (int i=0;i<CAM_LEVELS+1;++i) state->abs_cache[(CAM_LEVELS+1)*(CAM_LEVELS+1)+CAM_LEVELS*4+i]=emstot[emstot_offset+i];
    if (state->abs_phase == 2)
        for (int i=0;i<CAM_LEVELS+1;++i) emstot[emstot_offset+i]=state->abs_cache[(CAM_LEVELS+1)*(CAM_LEVELS+1)+CAM_LEVELS*4+i];
    radabs_(state, lchnk, ncol, pcols, pver, pverp, &pmid[pmid_offset], &pint[
        pint_offset], co2em, co2eml, tplnka, s2c, tcg, w, h2otr,
        plco2, plh2o, co2t, tint, tlayr, plol, plos, &pmln[
        pmln_offset], &piln[piln_offset], ucfc11, ucfc12, un2o0,
        un2o1, uch4, uco211, uco212, uco213, uco221, uco222, uco223,
        uptype, bn2o0, bn2o1, bch4, abplnk1, abplnk2, &abstot[
        abstot_offset], &absnxt[absnxt_offset], plh2ob, wb, aer_mpp__,
         aer_trn_ttl__);
    if (state->abs_phase == 1) return 0;
    }
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    delt[i__ - 1] = tint4[i__ + CAM_LEVELS - 2] - tlayr4[i__ + (CAM_LEVELS+1) - 2];
    delt1[i__ - 1] = tlayr4[i__ + (CAM_LEVELS+1) - 2] - tint4[i__ + (CAM_LEVELS+1) - 2];
    s[i__ + ((CAM_LEVELS+1) + (CAM_LEVELS+1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = state->cg_stebol__ * (
        delt1[i__ - 1] * absnxt[i__ + (CAM_LEVELS + absnxt_dim2) *
        absnxt_dim1] + delt[i__ - 1] * absnxt[i__ + (CAM_LEVELS + (
        absnxt_dim2 << 2)) * absnxt_dim1]);
    s[i__ + (CAM_LEVELS + (CAM_LEVELS+1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = state->cg_stebol__ * (
        delt[i__ - 1] * absnxt[i__ + (CAM_LEVELS + (absnxt_dim2 << 1)) *
        absnxt_dim1] + delt1[i__ - 1] * absnxt[i__ + (CAM_LEVELS +
        absnxt_dim2 * 3) * absnxt_dim1]);

    }

    i__1 = CAM_LEVELS - 1;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        bk2[i__ - 1] = (abstot[i__ + (k + CAM_LEVELS * abstot_dim2) *
            abstot_dim1] + abstot[i__ + (k + (CAM_LEVELS+1) * abstot_dim2) *
            abstot_dim1]) * .5f;
        bk1[i__ - 1] = bk2[i__ - 1];
        s[i__ + (k + (CAM_LEVELS+1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = state->cg_stebol__ * (bk2[
            i__ - 1] * delt[i__ - 1] + bk1[i__ - 1] * delt1[i__ - 1]);

    }


    }

    i__1 = state->cg_ntoplw__ + 1;
    for (km = CAM_LEVELS; km >= i__1; --km) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        delt[i__ - 1] = tint4[i__ + (km - 1) - 2] - tlayr4[i__ + km - 2];
        delt1[i__ - 1] = tlayr4[i__ + km - 2] - tint4[i__ + km - 2];

    }

    i__2 = state->cg_ntoplw__;
    for (k = (CAM_LEVELS+1); k >= i__2; --k) {
        if (k == km) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            bk2[i__ - 1] = absnxt[i__ + (km - 1 + (absnxt_dim2 << 2))
                * absnxt_dim1];
            bk1[i__ - 1] = absnxt[i__ + (km - 1 + absnxt_dim2) *
                absnxt_dim1];

        }

        } else if (k == km - 1) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            bk2[i__ - 1] = absnxt[i__ + (km - 1 + (absnxt_dim2 << 1))
                * absnxt_dim1];
            bk1[i__ - 1] = absnxt[i__ + (km - 1 + absnxt_dim2 * 3) *
                absnxt_dim1];

        }

        } else {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            bk2[i__ - 1] = (abstot[i__ + (k + (km - 1) * abstot_dim2)
                * abstot_dim1] + abstot[i__ + (k + km *
                abstot_dim2) * abstot_dim1]) * .5f;
            bk1[i__ - 1] = bk2[i__ - 1];

        }

        }
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        s[i__ + (k + km * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = s[i__ + (k + (km + 1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) -
            ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] + state->cg_stebol__ * (bk2[i__ - 1] * delt[
            i__ - 1] + bk1[i__ - 1] * delt1[i__ - 1]);

        }


    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    fsul[i__ + (CAM_LEVELS+1) - 2] = lwupcgs[i__];

    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tmp[i__ - 1] = fsul[i__ + (CAM_LEVELS+1) - 2] - state->cg_stebol__ *
        tint4[i__ + (CAM_LEVELS+1) - 2];
    fsul[i__ + state->cg_ntoplw__ - 2] = fsul[i__ + (CAM_LEVELS+1) - 2] -
        abstot[i__ + (state->cg_ntoplw__ + (CAM_LEVELS+1) * abstot_dim2)
         * abstot_dim1] * tmp[i__ - 1] + s[i__ + (
        state->cg_ntoplw__ + (state->cg_ntoplw__ + 1) *
        (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))];

    d__1 = tplnke[i__ - 1], d__1 *= d__1;
    fsdl[i__ + state->cg_ntoplw__ - 2] = state->cg_stebol__ *
        (d__1 * d__1) * emstot[i__ + state->cg_ntoplw__ *
        emstot_dim1];

    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__ + 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        fsul[i__ + k - 2] = fsul[i__ + (CAM_LEVELS+1) - 2] - abstot[i__ + (k + *
            pverp * abstot_dim2) * abstot_dim1] * tmp[i__ - 1] + s[
            i__ + (k + (k + 1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))];

        d__1 = tplnke[i__ - 1], d__1 *= d__1;
        fsdl[i__ + k - 2] = state->cg_stebol__ * (d__1 * d__1) *
            emstot[i__ + k * emstot_dim1] - (s[i__ + (k + (
            state->cg_ntoplw__ + 1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] - s[i__ + (k + (
            k + 1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))]);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {

    d__1 = tplnke[i__ - 1], d__1 *= d__1;
    absbt[i__ - 1] = state->cg_stebol__ * (d__1 * d__1) * emstot[
        i__ + (CAM_LEVELS+1) * emstot_dim1];
    fsdl[i__ + (CAM_LEVELS+1) - 2] = absbt[i__ - 1] - s[i__ + ((CAM_LEVELS+1) + (
        state->cg_ntoplw__ + 1) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))];

    }

    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
    i__2 = CAM_LEVELS - state->cg_ntoplw__;
    for (cam_s1_1__ = 0; cam_s1_1__ <= i__2; ++cam_s1_1__) {
        cldp[cam_s1_0__ + 1 + (state->cg_ntoplw__ + cam_s1_1__) - 2]
             = cld[cam_s1_0__ + 1 + (state->cg_ntoplw__ +
            cam_s1_1__) * cld_dim1];

    }


    }

    i__1 = 1 - 1;
    for (cam_s2_0__ = 0; cam_s2_0__ <= i__1; ++cam_s2_0__) {
    cldp[cam_s2_0__ + 1 + (CAM_LEVELS+1) - 2] = 0.f;

    }

    i__1 = 1;
    for (cam_red_i__ = 1; cam_red_i__ <= i__1; ++cam_red_i__) {
    maxcld[cam_red_i__ - 1] = cldp[cam_red_i__ + state->cg_ntoplw__
        - 2];
    i__2 = CAM_LEVELS;
    for (cam_red_k__ = state->cg_ntoplw__ + 1; cam_red_k__ <= i__2;
        ++cam_red_k__) {

        d__1 = maxcld[cam_red_i__ - 1], d__2 = cldp[cam_red_i__ +
            cam_red_k__ - 2];
        maxcld[cam_red_i__ - 1] = max(d__1,d__2);

    }


    }

    npts = 0;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    if (maxcld[i__ - 1] < 1e-80) {
        ++npts;
        indx[npts - 1] = i__;
    }

    }

    i__1 = npts;
    for (ii = 1; ii <= i__1; ++ii) {
    i__ = indx[ii - 1];
    i__2 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__2; ++k) {
        fdl[i__ + k - 2] = fsdl[i__ + k - 2];
        ful[i__ + k - 2] = fsul[i__ + k - 2];

    }


    }

    npts = 0;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    if (maxcld[i__ - 1] >= 1e-80) {
        ++npts;
        indx[npts - 1] = i__;
    }

    }

    i__1 = npts;
    for (ii = 1; ii <= i__1; ++ii) {
    i__ = indx[ii - 1];
    fdl[i__ + state->cg_ntoplw__ - 2] = fsdl[i__ +
        state->cg_ntoplw__ - 2];
    fdl[i__ + (CAM_LEVELS+1) - 2] = 0.f;
    ful[i__ + state->cg_ntoplw__ - 2] = 0.f;
    ful[i__ + (CAM_LEVELS+1) - 2] = fsul[i__ + (CAM_LEVELS+1) - 2];
    i__2 = CAM_LEVELS;
    for (k = state->cg_ntoplw__ + 1; k <= i__2; ++k) {
        fdl[i__ + k - 2] = 0.f;
        ful[i__ + k - 2] = 0.f;

    }

    i__2 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__2; ++k) {
        fclt4[i__ + (k - 1) - 1] = state->cg_stebol__ * tint4[i__ +
            k - 2];
        fclb4[i__ + (k - 1)] = state->cg_stebol__ * tint4[i__ + (k
            + 1) - 2];

    }

    fclb4[i__ + (state->cg_ntoplw__ - 2)] =
        state->cg_stebol__ * tint4[i__ +
        state->cg_ntoplw__ - 2];
    fclt4[i__ + CAM_LEVELS - 1] = state->cg_stebol__ * tint4[i__ + *
        pverp - 2];
    i__2 = nmxrgn[i__];
    for (irgn = 0; irgn <= i__2; ++irgn) {
        kx2[i__ + irgn - 1] = state->cg_ntoplw__ - 1;

    }

    nrgn[i__ - 1] = 0;

    }

    i__1 = npts;
    for (ii = 1; ii <= i__1; ++ii) {
    ilon = indx[ii - 1];
    i__2 = nmxrgn[ilon];
    for (irgn = 1; irgn <= i__2; ++irgn) {
        n = 0;
        if (kx2[ilon + (irgn - 1) - 1] < CAM_LEVELS) {
        nrgn[ilon - 1] = irgn;
        k1 = kx2[ilon + (irgn - 1) - 1] + 1;
        kx1[ilon + irgn - 2] = k1;
        kx2[ilon + irgn - 1] = 0;
        i__3 = k1;
        for (k2 = CAM_LEVELS; k2 >= i__3; --k2) {
            if (pmid[ilon + k2 * pmid_dim1] <= pmxrgn[ilon + irgn *
                pmxrgn_dim1]) {
            kx2[ilon + irgn - 1] = k2;
            goto L9063;
            }

        }
L9063:
        i__3 = k2;
        for (k = k1; k <= i__3; ++k) {
            if (cldp[ilon + k - 2] >= 1e-80) {
            ++n;
            indxmx[n + irgn - 2] = ilon;
            goto L9065;
            }

        }
L9065:
        ;
        }
        ncolmx[irgn - 1] = n;
        i__3 = ncolmx[irgn - 1];
        for (iimx = 1; iimx <= i__3; ++iimx) {
        i__ = indxmx[iimx + irgn - 2];
        n = 0;
        i__4 = kx2[i__ + irgn - 1];
        for (k = kx1[i__ + irgn - 2]; k <= i__4; ++k) {
            if (cldp[i__ + k - 2] >= 1e-80) {
            ++n;
            ksort[n] = k;
            asort[n - 1] = 1.f - cldp[i__ + k - 2];
            }

        }

        nxs[i__ + irgn - 2] = n;
        if (nxs[i__ + irgn - 2] == 2) {
            if (asort[1] < asort[0]) {
            ktmp = ksort[1];
            ksort[1] = ksort[2];
            ksort[2] = ktmp;
            atmp = asort[0];
            asort[0] = asort[1];
            asort[1] = atmp;
            }
        } else if (nxs[i__ + irgn - 2] >= 3) {
            sortarray_(state, &nxs[i__ + irgn - 2], asort, &ksort[1]);
        }
        i__4 = nxs[i__ + irgn - 2];
        for (l = 1; l <= i__4; ++l) {
            kxs[l + (i__ + irgn) * (((1)*(((CAM_LEVELS+1)-(0)+1)))*((1-(1)+1))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))+(1)*(((1)*(((CAM_LEVELS+1)-(0)+1)))*((1-(1)+1))))] = ksort[l];

        }


        }


    }

    i__2 = nmxrgn[ilon];
    for (irgn = 1; irgn <= i__2; ++irgn) {
        iimx = 1;
        if (ilon < indxmx[iimx + irgn - 2] && irgn <= nrgn[ilon - 1]) {
        k1 = kx1[ilon + irgn - 2];
        i__3 = k1 - 2;
        for (km1 = state->cg_ntoplw__ - 2; km1 <= i__3; ++km1) {
            km4 = km1 + 3;
            k2 = k1;
            k3 = k2 + 1;

            i__4 = 1, i__5 = pverp2 - k3;
            tmp[ilon - 1] = s[ilon + (k2 + min(k3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))]
                * min(i__4,i__5);
            emx0 = (fdl[ilon + k1 - 2] - fsdl[ilon + k1 - 2]) / (
                fclb4[ilon + km1] - s[ilon + (k2 + km4 * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))]
                 + tmp[ilon - 1] - fsdl[ilon + k1 - 2]);
            if (emx0 >= 0.f && emx0 <= 1.f) {
            goto L9075;
            }

        }
L9075:

        i__3 = km1, i__4 = k1 - 2;
        km1 = min(i__3,i__4);
        i__3 = kx2[ilon + irgn - 1] + 1;
        for (k2 = kx1[ilon + irgn - 2] + 1; k2 <= i__3; ++k2) {
            k3 = k2 + 1;

            i__4 = 1, i__5 = pverp2 - k3;
            tmp[ilon - 1] = s[ilon + (k2 + min(k3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))]
                * min(i__4,i__5);
            fdl[ilon + k2 - 2] = (1.f - emx0) * fsdl[ilon + k2 - 2] +
                emx0 * (fclb4[ilon + km1] - s[ilon + (k2 + km4 *
                (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] + tmp[ilon - 1]);

        }

        } else if (ilon == indxmx[iimx + irgn - 2] && iimx <= ncolmx[irgn
            - 1]) {
        ++iimx;
        }
        i__3 = ncolmx[irgn - 1];
        for (iimx = 1; iimx <= i__3; ++iimx) {
        i__ = indxmx[iimx + irgn - 2];
        k1 = kx1[i__ + irgn - 2];
        i__4 = k1 - 2;
        for (km1 = state->cg_ntoplw__ - 2; km1 <= i__4; ++km1) {
            km4 = km1 + 3;
            k2 = k1;
            k3 = k2 + 1;

            i__5 = 1, i__6 = pverp2 - k3;
            tmp[i__ - 1] = s[i__ + (k2 + min(k3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] *
                min(i__5,i__6);

            i__5 = 1, i__6 = pverp2 - km4;
            tmp2[i__ - 1] = s[i__ + (k2 + min(km4,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))]
                * min(i__5,i__6);
            emx0 = (fdl[i__ + k1 - 2] - fsdl[i__ + k1 - 2]) / (fclb4[
                i__ + km1] - tmp2[i__ - 1] + tmp[i__ - 1] - fsdl[
                i__ + k1 - 2]);
            if (emx0 >= 0.f && emx0 <= 1.f) {
            goto L9081;
            }

        }
L9081:

        i__4 = km1, i__5 = k1 - 2;
        km1 = min(i__4,i__5);
        ksort[0] = km1 + 1;
        nxsk = 0;
        i__4 = kx2[i__ + irgn - 1];
        for (k = kx1[i__ + irgn - 2]; k <= i__4; ++k) {
            if (nxsk < nxs[i__ + irgn - 2]) {
            nxsk = 0;
            i__5 = nxs[i__ + irgn - 2];
            for (l = 1; l <= i__5; ++l) {
                k1 = kxs[l + (i__ + irgn) * (((1)*(((CAM_LEVELS+1)-(0)+1)))*((1-(1)+1))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))+(1)*(((1)*(((CAM_LEVELS+1)-(0)+1)))*((1-(1)+1))))];
                if (k >= k1) {
                ++nxsk;
                ksort[nxsk] = k1;
                }

            }

            }
            ksort[nxsk + 1] = (CAM_LEVELS+1);
            i__5 = nxsk;
            for (l = 1; l <= i__5; ++l) {
            emx[l] = emis[i__ + ksort[l] * emis_dim1];

            }

            emx[0] = emx0;
            cld0 = 1.f;
            k2 = k + 1;
            k3 = k2 + 1;

            i__5 = 1, i__6 = pverp2 - k3;
            tmp[i__ - 1] = s[i__ + (k2 + min(k3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] *
                min(i__5,i__6);
            i__5 = nxsk + 1;
            for (l = 1; l <= i__5; ++l) {

            i__6 = 1, i__7 = nxsk + 1 - l;
            cld1 = cldp[i__ + ksort[l] - 2] * min(i__6,i__7);
            if (cld0 != cld1) {
                fdl[i__ + k2 - 2] += (cld0 - cld1) * fsdl[i__ +
                    k2 - 2];
                i__6 = l - 1;
                for (l1 = 0; l1 <= i__6; ++l1) {
                km1 = ksort[l1] - 1;
                km4 = km1 + 3;

                i__7 = 1, i__8 = pverp2 - km4;
                tmp2[i__ - 1] = s[i__ + (k2 + min(km4,(CAM_LEVELS+1))
                    * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] * min(i__7,i__8);
                fdl[i__ + k2 - 2] += (cld0 - cld1) * emx[l1] *
                     (fclb4[i__ + km1] - tmp2[i__ - 1] +
                    tmp[i__ - 1] - fsdl[i__ + k2 - 2]);

                }

            }
            cld0 = cld1;
            if (l <= nxsk) {
                k1 = ksort[l];
                trans = 1.f - emis[i__ + k1 * emis_dim1];
                i__6 = nxsk;
                for (l1 = 0; l1 <= i__6; ++l1) {
                if (ksort[l1] < k1) {
                    emx[l1] *= trans;
                }

                }

            }

            }


        }


        }


    }

    for (irgn = nmxrgn[ilon]; irgn >= 1; --irgn) {
        iimx = 1;
        if (ilon < indxmx[iimx + irgn - 2] && irgn <= nrgn[ilon - 1]) {
        k1 = kx2[ilon + irgn - 1] + 1;
        if (k1 < (CAM_LEVELS+1)) {
            i__2 = kx2[ilon + irgn - 1];
            for (km1 = CAM_LEVELS - 1; km1 >= i__2; --km1) {
            km3 = km1 + 2;
            k2 = k1;
            k3 = k2 + 1;

            i__3 = 1, i__4 = pverp2 - km3;
            tmp[ilon - 1] = s[ilon + (k2 + min(km3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))
                - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] * min(i__3,i__4);
            emx0 = (ful[ilon + k1 - 2] - fsul[ilon + k1 - 2]) / (
                fclt4[ilon + km1 - 1] + s[ilon + (k2 + k3 *
                (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] - tmp[ilon - 1] - fsul[ilon + k1 -
                2]);
            if (emx0 >= 0.f && emx0 <= 1.f) {
                goto L9097;
            }

            }
L9097:

            i__2 = km1, i__3 = kx2[ilon + irgn - 1];
            km1 = max(i__2,i__3);
        } else {
            km1 = k1 - 1;
            km3 = km1 + 2;
            emx0 = 1.f;
        }
        i__2 = kx2[ilon + irgn - 1];
        for (k2 = kx1[ilon + irgn - 2]; k2 <= i__2; ++k2) {
            k3 = k2 + 1;

            i__3 = 1, i__4 = pverp2 - km3;
            tmp[ilon - 1] = s[ilon + (k2 + min(km3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))]
                 * min(i__3,i__4);
            ful[ilon + k2 - 2] = (1.f - emx0) * fsul[ilon + k2 - 2] +
                emx0 * (fclt4[ilon + km1 - 1] + s[ilon + (k2 + k3
                * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] - tmp[ilon - 1]);

        }

        } else if (ilon == indxmx[iimx + irgn - 2] && iimx <= ncolmx[irgn
            - 1]) {
        ++iimx;
        }
        i__2 = ncolmx[irgn - 1];
        for (iimx = 1; iimx <= i__2; ++iimx) {
        i__ = indxmx[iimx + irgn - 2];
        k1 = kx2[i__ + irgn - 1] + 1;
        if (k1 < (CAM_LEVELS+1)) {
            i__3 = kx2[i__ + irgn - 1];
            for (km1 = CAM_LEVELS - 1; km1 >= i__3; --km1) {
            km3 = km1 + 2;
            k2 = k1;
            k3 = k2 + 1;

            i__4 = 1, i__5 = pverp2 - km3;
            tmp[i__ - 1] = s[i__ + (k2 + min(km3,(CAM_LEVELS+1)) * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) -
                ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] * min(i__4,i__5);
            emx0 = (ful[i__ + k1 - 2] - fsul[i__ + k1 - 2]) / (
                fclt4[i__ + km1 - 1] + s[i__ + (k2 + k3 * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))
                - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] - tmp[i__ - 1] - fsul[i__ + k1 - 2]);
            if (emx0 >= 0.f && emx0 <= 1.f) {
                goto L9103;
            }

            }
L9103:

            i__3 = km1, i__4 = kx2[i__ + irgn - 1];
            km1 = max(i__3,i__4);
        } else {
            emx0 = 1.f;
            km1 = k1 - 1;
        }
        ksort[0] = km1 + 1;
        nxsk = 0;
        i__3 = kx1[i__ + irgn - 2];
        for (k = kx2[i__ + irgn - 1]; k >= i__3; --k) {
            if (nxsk < nxs[i__ + irgn - 2]) {
            nxsk = 0;
            i__4 = nxs[i__ + irgn - 2];
            for (l = 1; l <= i__4; ++l) {
                k1 = kxs[l + (i__ + irgn) * (((1)*(((CAM_LEVELS+1)-(0)+1)))*((1-(1)+1))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))+(1)*(((1)*(((CAM_LEVELS+1)-(0)+1)))*((1-(1)+1))))];
                if (k <= k1) {
                ++nxsk;
                ksort[nxsk] = k1;
                }

            }

            }
            ksort[nxsk + 1] = (CAM_LEVELS+1);
            i__4 = nxsk;
            for (l = 1; l <= i__4; ++l) {
            emx[l] = emis[i__ + ksort[l] * emis_dim1];

            }

            emx[0] = emx0;
            cld0 = 1.f;
            k2 = k;
            k3 = k2 + 1;
            i__4 = nxsk + 1;
            for (l = 1; l <= i__4; ++l) {

            i__5 = 1, i__6 = nxsk + 1 - l;
            cld1 = cldp[i__ + ksort[l] - 2] * min(i__5,i__6);
            if (cld0 != cld1) {
                ful[i__ + k2 - 2] += (cld0 - cld1) * fsul[i__ +
                    k2 - 2];
                i__5 = l - 1;
                for (l1 = 0; l1 <= i__5; ++l1) {
                km1 = ksort[l1] - 1;
                km3 = km1 + 2;

                i__6 = 1, i__7 = pverp2 - km3;
                tmp[i__ - 1] = s[i__ + (k2 + min(km3,(CAM_LEVELS+1)) *
                     (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] * min(i__6,i__7);
                ful[i__ + k2 - 2] += (cld0 - cld1) * emx[l1] *
                     (fclt4[i__ + km1 - 1] + s[i__ + (k2
                    + k3 * (((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] - tmp[i__ - 1] -
                    fsul[i__ + k2 - 2]);

                }

            }
            cld0 = cld1;
            if (l <= nxsk) {
                k1 = ksort[l];
                trans = 1.f - emis[i__ + k1 * emis_dim1];
                i__5 = nxsk;
                for (l1 = 0; l1 <= i__5; ++l1) {
                if (ksort[l1] > k1) {
                    emx[l1] *= trans;
                }

                }

            }

            }


        }


        }


    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    flwds[i__] = fdl[i__ + (CAM_LEVELS+1) - 2];
    flns[i__] = ful[i__ + (CAM_LEVELS+1) - 2] - fdl[i__ + (CAM_LEVELS+1) - 2];
    flnsc[i__] = fsul[i__ + (CAM_LEVELS+1) - 2] - fsdl[i__ + (CAM_LEVELS+1) - 2];
    flnt[i__] = ful[i__ + state->cg_ntoplw__ - 2] - fdl[i__ +
        state->cg_ntoplw__ - 2];
    flntc[i__] = fsul[i__ + state->cg_ntoplw__ - 2] - fsdl[i__ +
        state->cg_ntoplw__ - 2];
    flut[i__] = ful[i__ + state->cg_ntoplw__ - 2];
    flutc[i__] = fsul[i__ + state->cg_ntoplw__ - 2];

    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        qrl[i__ + k * qrl_dim1] = (ful[i__ + k - 2] - fdl[i__ + k - 2] -
            ful[i__ + (k + 1) - 2] + fdl[i__ + (k + 1) - 2]) *
            9.99999974737875164e-5f * state->cg_gravit__ / (
            pint[i__ + k * pint_dim1] - pint[i__ + (k + 1) *
            pint_dim1]);

    }


    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        qrlcs[i__ + k * qrlcs_dim1] = (fsul[i__ + k - 2] - fsdl[i__ + k -
            2] - fsul[i__ + (k + 1) - 2] + fsdl[i__ + (k + 1) - 2]) *
            9.99999974737875164e-5f * state->cg_gravit__ / (
            pint[i__ + k * pint_dim1] - pint[i__ + (k + 1) *
            pint_dim1]);

    }


    }

    if (state->cg_ntoplw__ > 1) {
    i__1 = 1 - 1;
    for (cam_s3_0__ = 0; cam_s3_0__ <= i__1; ++cam_s3_0__) {
        i__2 = state->cg_ntoplw__ - 2;
        for (cam_s3_1__ = 0; cam_s3_1__ <= i__2; ++cam_s3_1__) {
        qrl[cam_s3_0__ + 1 + (cam_s3_1__ + 1) * qrl_dim1] = 0.f;

        }


    }

    i__1 = 1 - 1;
    for (cam_s4_0__ = 0; cam_s4_0__ <= i__1; ++cam_s4_0__) {
        i__2 = state->cg_ntoplw__ - 2;
        for (cam_s4_1__ = 0; cam_s4_1__ <= i__2; ++cam_s4_1__) {
        qrlcs[cam_s4_0__ + 1 + (cam_s4_1__ + 1) * qrlcs_dim1] = 0.f;

        }


    }

    }
    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        flup[i__ + k * flup_dim1] = ful[i__ + k - 2];
        flupc[i__ + k * flupc_dim1] = fsul[i__ + k - 2];
        fldn[i__ + k * fldn_dim1] = fdl[i__ + k - 2];
        fldnc[i__ + k * fldnc_dim1] = fsdl[i__ + k - 2];

    }


    }

    if (state->cg_ntoplw__ > 1) {
    i__1 = 1 - 1;
    for (cam_s5_0__ = 0; cam_s5_0__ <= i__1; ++cam_s5_0__) {
        i__2 = state->cg_ntoplw__ - 2;
        for (cam_s5_1__ = 0; cam_s5_1__ <= i__2; ++cam_s5_1__) {
        flup[cam_s5_0__ + 1 + (cam_s5_1__ + 1) * flup_dim1] = 0.f;

        }


    }

    i__1 = 1 - 1;
    for (cam_s6_0__ = 0; cam_s6_0__ <= i__1; ++cam_s6_0__) {
        i__2 = state->cg_ntoplw__ - 2;
        for (cam_s6_1__ = 0; cam_s6_1__ <= i__2; ++cam_s6_1__) {
        flupc[cam_s6_0__ + 1 + (cam_s6_1__ + 1) * flupc_dim1] = 0.f;

        }


    }

    i__1 = 1 - 1;
    for (cam_s7_0__ = 0; cam_s7_0__ <= i__1; ++cam_s7_0__) {
        i__2 = state->cg_ntoplw__ - 2;
        for (cam_s7_1__ = 0; cam_s7_1__ <= i__2; ++cam_s7_1__) {
        fldn[cam_s7_0__ + 1 + (cam_s7_1__ + 1) * fldn_dim1] = 0.f;

        }


    }

    i__1 = 1 - 1;
    for (cam_s8_0__ = 0; cam_s8_0__ <= i__1; ++cam_s8_0__) {
        i__2 = state->cg_ntoplw__ - 2;
        for (cam_s8_1__ = 0; cam_s8_1__ <= i__2; ++cam_s8_1__) {
        fldnc[cam_s8_0__ + 1 + (cam_s8_1__ + 1) * fldnc_dim1] = 0.f;

        }


    }

    }
    return 0;
}






__device__ integer radtpl_c__1 = 1;
__device__ integer radtpl_c__2 = 2;


__device__ int radtpl_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *tnm, doublereal *lwupcgs,
    doublereal *qnm, doublereal *pnm, doublereal *plco2, doublereal *
    plh2o, doublereal *tplnka, doublereal *s2c, doublereal *tcg,
    doublereal *w, doublereal *tplnke, doublereal *tint, doublereal *
    tint4, doublereal *tlayr, doublereal *tlayr4, doublereal *pmln,
    doublereal *piln, doublereal *plh2ob, doublereal *wb)
{
    integer tnm_dim1 = 0,tnm_offset = 0,qnm_dim1 = 0,qnm_offset = 0,pnm_dim1 = 0,pnm_offset = 0,pmln_dim1 = 0,pmln_offset = 0,piln_dim1 = 0,piln_offset = 0,plco2_dim1 = 0,plco2_offset = 0,plh2o_dim1 = 0,plh2o_offset = 0,tplnka_dim1 = 0,tplnka_offset = 0,s2c_dim1 = 0,s2c_offset = 0,tcg_dim1 = 0,tcg_offset = 0,w_dim1 = 0,w_offset = 0,tint_dim1 = 0,tint_offset = 0,tint4_dim1 = 0,tint4_offset = 0,tlayr_dim1 = 0,tlayr_offset = 0,tlayr4_dim1 = 0,tlayr4_offset = 0,plh2ob_dim2 = 0,plh2ob_offset = 0,wb_dim2 = 0,wb_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0,d__2 = 0;

    integer i__ = 0,k = 0;

    doublereal dw = 0,dy = 0;

    integer kp1 = 0;

    doublereal dpnm = 0,cpwpl = 0,dplh2o = 0,repsil = 0,dpnmsq = 0;


    --tplnke;
    --lwupcgs;
    pmln_dim1 = 1;
    pmln_offset = 1 + pmln_dim1;
    pmln -= pmln_offset;
    qnm_dim1 = 1;
    qnm_offset = 1 + qnm_dim1;
    qnm -= qnm_offset;
    tnm_dim1 = 1;
    tnm_offset = 1 + tnm_dim1;
    tnm -= tnm_offset;
    wb_dim2 = 1;
    wb_offset = 1 + 2 * (1 + wb_dim2);
    wb -= wb_offset;
    plh2ob_dim2 = 1;
    plh2ob_offset = 1 + 2 * (1 + plh2ob_dim2);
    plh2ob -= plh2ob_offset;
    piln_dim1 = 1;
    piln_offset = 1 + piln_dim1;
    piln -= piln_offset;
    tlayr4_dim1 = 1;
    tlayr4_offset = 1 + tlayr4_dim1;
    tlayr4 -= tlayr4_offset;
    tlayr_dim1 = 1;
    tlayr_offset = 1 + tlayr_dim1;
    tlayr -= tlayr_offset;
    tint4_dim1 = 1;
    tint4_offset = 1 + tint4_dim1;
    tint4 -= tint4_offset;
    tint_dim1 = 1;
    tint_offset = 1 + tint_dim1;
    tint -= tint_offset;
    w_dim1 = 1;
    w_offset = 1 + w_dim1;
    w -= w_offset;
    tcg_dim1 = 1;
    tcg_offset = 1 + tcg_dim1;
    tcg -= tcg_offset;
    s2c_dim1 = 1;
    s2c_offset = 1 + s2c_dim1;
    s2c -= s2c_offset;
    tplnka_dim1 = 1;
    tplnka_offset = 1 + tplnka_dim1;
    tplnka -= tplnka_offset;
    plh2o_dim1 = 1;
    plh2o_offset = 1 + plh2o_dim1;
    plh2o -= plh2o_offset;
    plco2_dim1 = 1;
    plco2_offset = 1 + plco2_dim1;
    plco2 -= plco2_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;


    repsil = 1.f / state->cg_epsilo__;
    cpwpl = state->cg_amco2__ / state->cg_amd__ * .5f / (
        state->cg_gravit__ * state->cg_p0__);
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    plh2o[i__ + state->cg_ntoplw__ * plh2o_dim1] =
        state->cg_rgsslp__ * qnm[i__ + state->cg_ntoplw__
        * qnm_dim1] * pnm[i__ + state->cg_ntoplw__ * pnm_dim1] *
         pnm[i__ + state->cg_ntoplw__ * pnm_dim1];
    plco2[i__ + state->cg_ntoplw__ * plco2_dim1] =
        state->cg_co2vmr__ * cpwpl * pnm[i__ +
        state->cg_ntoplw__ * pnm_dim1] * pnm[i__ +
        state->cg_ntoplw__ * pnm_dim1];

    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {

        d__1 = pnm[i__ + (k + 1) * pnm_dim1];

        d__2 = pnm[i__ + k * pnm_dim1];
        plh2o[i__ + (k + 1) * plh2o_dim1] = plh2o[i__ + k * plh2o_dim1] +
            state->cg_rgsslp__ * (d__1 * d__1 - d__2 * d__2) *
            qnm[i__ + k * qnm_dim1];

        d__1 = pnm[i__ + (k + 1) * pnm_dim1];
        plco2[i__ + (k + 1) * plco2_dim1] = state->cg_co2vmr__ *
            cpwpl * (d__1 * d__1);

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tint4[i__ + (CAM_LEVELS+1) * tint4_dim1] = lwupcgs[i__] /
        state->cg_stebol__;
    tint[i__ + (CAM_LEVELS+1) * tint_dim1] = sqrt(sqrt(tint4[i__ + (CAM_LEVELS+1) *
        tint4_dim1]));
    tplnka[i__ + state->cg_ntoplw__ * tplnka_dim1] = tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1];
    tint[i__ + state->cg_ntoplw__ * tint_dim1] = tplnka[i__ +
        state->cg_ntoplw__ * tplnka_dim1];

    d__1 = tplnka[i__ + state->cg_ntoplw__ * tplnka_dim1], d__1 *=
        d__1;
    tlayr4[i__ + state->cg_ntoplw__ * tlayr4_dim1] = d__1 * d__1;
    tint4[i__ + state->cg_ntoplw__ * tint4_dim1] = tlayr4[i__ +
        state->cg_ntoplw__ * tlayr4_dim1];

    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__ + 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        dy = (piln[i__ + k * piln_dim1] - pmln[i__ + k * pmln_dim1]) / (
            pmln[i__ + (k - 1) * pmln_dim1] - pmln[i__ + k *
            pmln_dim1]);
        tint[i__ + k * tint_dim1] = tnm[i__ + k * tnm_dim1] - dy * (tnm[
            i__ + k * tnm_dim1] - tnm[i__ + (k - 1) * tnm_dim1]);

        d__1 = tint[i__ + k * tint_dim1], d__1 *= d__1;
        tint4[i__ + k * tint4_dim1] = d__1 * d__1;

    }


    }

    i__1 = (CAM_LEVELS+1);
    for (k = state->cg_ntoplw__ + 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        tlayr[i__ + k * tlayr_dim1] = tnm[i__ + (k - 1) * tnm_dim1];

        d__1 = tlayr[i__ + k * tlayr_dim1], d__1 *= d__1;
        tlayr4[i__ + k * tlayr4_dim1] = d__1 * d__1;
        tplnka[i__ + k * tplnka_dim1] = (tint[i__ + k * tint_dim1] + tint[
            i__ + (k - 1) * tint_dim1]) * .5f;

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tplnke[i__] = tplnka[i__ + state->cg_ntoplw__ * tplnka_dim1];
    tlayr[i__ + state->cg_ntoplw__ * tlayr_dim1] = tint[i__ +
        state->cg_ntoplw__ * tint_dim1];

    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    tcg[i__ + state->cg_ntoplw__ * tcg_dim1] =
        state->cg_rga__ * qnm[i__ + state->cg_ntoplw__ *
        qnm_dim1] * pnm[i__ + state->cg_ntoplw__ * pnm_dim1] *
        tnm[i__ + state->cg_ntoplw__ * tnm_dim1];
    w[i__ + state->cg_ntoplw__ * w_dim1] = state->cg_sslp__ *
        (plh2o[i__ + state->cg_ntoplw__ * plh2o_dim1] * 2.f) /
        pnm[i__ + state->cg_ntoplw__ * pnm_dim1];
    wb[(i__ + state->cg_ntoplw__ * wb_dim2 << 1) + 1] = w[i__ +
        state->cg_ntoplw__ * w_dim1] * phi_(state, &tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1], &radtpl_c__1);
    wb[(i__ + state->cg_ntoplw__ * wb_dim2 << 1) + 2] = w[i__ +
        state->cg_ntoplw__ * w_dim1] * phi_(state, &tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1], &radtpl_c__2);
    plh2ob[(i__ + state->cg_ntoplw__ * plh2ob_dim2 << 1) + 1] =
        plh2o[i__ + state->cg_ntoplw__ * plh2o_dim1] * psi_(state, &
        tnm[i__ + state->cg_ntoplw__ * tnm_dim1], &radtpl_c__1);
    plh2ob[(i__ + state->cg_ntoplw__ * plh2ob_dim2 << 1) + 2] =
        plh2o[i__ + state->cg_ntoplw__ * plh2o_dim1] * psi_(state, &
        tnm[i__ + state->cg_ntoplw__ * tnm_dim1], &radtpl_c__2);
    s2c[i__ + state->cg_ntoplw__ * s2c_dim1] = plh2o[i__ +
        state->cg_ntoplw__ * plh2o_dim1] * fh2oself_(state, &tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1]) * qnm[i__ +
        state->cg_ntoplw__ * qnm_dim1] * repsil;

    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        dpnm = pnm[i__ + (k + 1) * pnm_dim1] - pnm[i__ + k * pnm_dim1];

        d__1 = pnm[i__ + (k + 1) * pnm_dim1];

        d__2 = pnm[i__ + k * pnm_dim1];
        dpnmsq = d__1 * d__1 - d__2 * d__2;
        dw = state->cg_rga__ * qnm[i__ + k * qnm_dim1] * dpnm;
        kp1 = k + 1;
        w[i__ + kp1 * w_dim1] = w[i__ + k * w_dim1] + dw;
        wb[(i__ + kp1 * wb_dim2 << 1) + 1] = wb[(i__ + k * wb_dim2 << 1)
            + 1] + dw * phi_(state, &tnm[i__ + k * tnm_dim1], &radtpl_c__1);
        wb[(i__ + kp1 * wb_dim2 << 1) + 2] = wb[(i__ + k * wb_dim2 << 1)
            + 2] + dw * phi_(state, &tnm[i__ + k * tnm_dim1], &radtpl_c__2);
        dplh2o = plh2o[i__ + kp1 * plh2o_dim1] - plh2o[i__ + k *
            plh2o_dim1];
        plh2ob[(i__ + kp1 * plh2ob_dim2 << 1) + 1] = plh2ob[(i__ + k *
            plh2ob_dim2 << 1) + 1] + dplh2o * psi_(state, &tnm[i__ + k *
            tnm_dim1], &radtpl_c__1);
        plh2ob[(i__ + kp1 * plh2ob_dim2 << 1) + 2] = plh2ob[(i__ + k *
            plh2ob_dim2 << 1) + 2] + dplh2o * psi_(state, &tnm[i__ + k *
            tnm_dim1], &radtpl_c__2);
        tcg[i__ + kp1 * tcg_dim1] = tcg[i__ + k * tcg_dim1] + dw * tnm[
            i__ + k * tnm_dim1];
        s2c[i__ + kp1 * s2c_dim1] = s2c[i__ + k * s2c_dim1] +
            state->cg_rgsslp__ * dpnmsq * qnm[i__ + k *
            qnm_dim1] * fh2oself_(state, &tnm[i__ + k * tnm_dim1]) * qnm[i__
            + k * qnm_dim1] * repsil;

    }


    }

    return 0;
}





__device__ int radoz2_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *o3vmr, doublereal *pint,
    doublereal *plol, doublereal *plos, integer *ntoplw)
{
    integer o3vmr_dim1 = 0,o3vmr_offset = 0,pint_dim1 = 0,pint_offset = 0,plol_dim1 = 0,plol_offset = 0,plos_dim1 = 0,plos_offset = 0,i__1 = 0,i__2 = 0;

    integer i__ = 0,k = 0;


    o3vmr_dim1 = 1;
    o3vmr_offset = 1 + o3vmr_dim1;
    o3vmr -= o3vmr_offset;
    plos_dim1 = 1;
    plos_offset = 1 + plos_dim1;
    plos -= plos_offset;
    plol_dim1 = 1;
    plol_offset = 1 + plol_dim1;
    plol -= plol_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    plos[i__ + *ntoplw * plos_dim1] = state->cg_cplos__ *
        .100000001490116119f * o3vmr[i__ + *ntoplw * o3vmr_dim1] *
        pint[i__ + *ntoplw * pint_dim1];
    plol[i__ + *ntoplw * plol_dim1] = state->cg_cplol__ *
        .00999999977648258209f * o3vmr[i__ + *ntoplw * o3vmr_dim1] *
        pint[i__ + *ntoplw * pint_dim1] * pint[i__ + *ntoplw *
        pint_dim1];

    }

    i__1 = (CAM_LEVELS+1);
    for (k = *ntoplw + 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        plos[i__ + k * plos_dim1] = plos[i__ + (k - 1) * plos_dim1] +
            state->cg_cplos__ * .100000001490116119f * o3vmr[
            i__ + (k - 1) * o3vmr_dim1] * (pint[i__ + k * pint_dim1]
            - pint[i__ + (k - 1) * pint_dim1]);
        plol[i__ + k * plol_dim1] = plol[i__ + (k - 1) * plol_dim1] +
            state->cg_cplol__ * .00999999977648258209f * o3vmr[
            i__ + (k - 1) * o3vmr_dim1] * (pint[i__ + k * pint_dim1] *
             pint[i__ + k * pint_dim1] - pint[i__ + (k - 1) *
            pint_dim1] * pint[i__ + (k - 1) * pint_dim1]);

    }


    }

    return 0;
}





__device__ int aer_pth_(CamState* state, doublereal *aer_mass__, doublereal *aer_mpp__,
    integer *ncol, integer *pcols, integer *plev, integer *plevp)
{
    integer aer_mass_dim1 = 0,aer_mass_offset = 0,aer_mpp_dim1 = 0,aer_mpp_offset = 0,i__1 = 0,i__2 = 0;

    integer k = 0,cam_s1_0__ = 0,cam_s2_0__ = 0;


    aer_mass_dim1 = 1;
    aer_mass_offset = 1 + aer_mass_dim1;
    aer_mass__ -= aer_mass_offset;
    aer_mpp_dim1 = 1;
    aer_mpp_offset = 1 + aer_mpp_dim1;
    aer_mpp__ -= aer_mpp_offset;


    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
    aer_mpp__[cam_s1_0__ + 1 + aer_mpp_dim1] = 0.;

    }

    i__1 = *plevp;
    for (k = 2; k <= i__1; ++k) {
    i__2 = 1 - 1;
    for (cam_s2_0__ = 0; cam_s2_0__ <= i__2; ++cam_s2_0__) {
        aer_mpp__[cam_s2_0__ + 1 + k * aer_mpp_dim1] = aer_mpp__[
            cam_s2_0__ + 1 + (k - 1) * aer_mpp_dim1] + aer_mass__[
            cam_s2_0__ + 1 + (k - 1) * aer_mass_dim1];

    }


    }

    return 0;
}





__device__ int aer_trn_(CamState* state, doublereal *aer_mpp__, doublereal *
    aer_trn_ttl__, integer *pcols, integer *plev, integer *plevp)
{
    integer aer_mpp_dim1 = 0,aer_mpp_offset = 0,aer_trn_ttl_dim1 = 0,aer_trn_ttl_dim2 = 0,aer_trn_ttl_dim3 = 0,aer_trn_ttl_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,k1 = 0,k2 = 0;

    doublereal aer_pth_dlt__ = 0,odap_aer_ttl__ = 0;

    integer cam_s1_0__ = 0,cam_s1_1__ = 0,cam_s1_2__ = 0,cam_s1_3__ = 0,bnd_idx__ = 0;


    aer_trn_ttl_dim1 = 1;
    aer_trn_ttl_dim2 = *plevp;
    aer_trn_ttl_dim3 = *plevp;
    aer_trn_ttl_offset = 1 + aer_trn_ttl_dim1 * (1 + aer_trn_ttl_dim2 * (1 +
        aer_trn_ttl_dim3));
    aer_trn_ttl__ -= aer_trn_ttl_offset;
    aer_mpp_dim1 = 1;
    aer_mpp_offset = 1 + aer_mpp_dim1;
    aer_mpp__ -= aer_mpp_offset;


    if (state->cg_strat_volcanic__) {
    for (bnd_idx__ = 1; bnd_idx__ <= 7; ++bnd_idx__) {
        i__1 = 1;
        for (i__ = 1; i__ <= i__1; ++i__) {
        aer_trn_ttl__[i__ + ((bnd_idx__ * aer_trn_ttl_dim3 + 1) *
            aer_trn_ttl_dim2 + 1) * aer_trn_ttl_dim1] = 1.f;

        }

        i__1 = *plevp;
        for (k1 = 2; k1 <= i__1; ++k1) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
            aer_trn_ttl__[i__ + (k1 + (k1 + bnd_idx__ *
                aer_trn_ttl_dim3) * aer_trn_ttl_dim2) *
                aer_trn_ttl_dim1] = 1.f;
            aer_pth_dlt__ = (d__1 = aer_mpp__[i__ + k1 * aer_mpp_dim1]
                 - aer_mpp__[i__ + aer_mpp_dim1], abs(d__1));
            odap_aer_ttl__ = state->cg_abs_cff_mss_aer__[
                bnd_idx__ - 1] * aer_pth_dlt__;
            aer_trn_ttl__[i__ + ((k1 + bnd_idx__ * aer_trn_ttl_dim3) *
                 aer_trn_ttl_dim2 + 1) * aer_trn_ttl_dim1] = cam_exp64(
                odap_aer_ttl__ * -1.65999996662139893f);

        }


        }

        i__1 = *plev;
        for (k1 = 2; k1 <= i__1; ++k1) {
        i__2 = *plevp;
        for (k2 = k1 + 1; k2 <= i__2; ++k2) {
            i__3 = 1;
            for (i__ = 1; i__ <= i__3; ++i__) {
            aer_trn_ttl__[i__ + (k1 + (k2 + bnd_idx__ *
                aer_trn_ttl_dim3) * aer_trn_ttl_dim2) *
                aer_trn_ttl_dim1] = aer_trn_ttl__[i__ + ((k2
                + bnd_idx__ * aer_trn_ttl_dim3) *
                aer_trn_ttl_dim2 + 1) * aer_trn_ttl_dim1] /
                aer_trn_ttl__[i__ + ((k1 + bnd_idx__ *
                aer_trn_ttl_dim3) * aer_trn_ttl_dim2 + 1) *
                aer_trn_ttl_dim1];

            }


        }


        }

        i__1 = *plevp;
        for (k1 = 2; k1 <= i__1; ++k1) {
        i__2 = k1 - 1;
        for (k2 = 1; k2 <= i__2; ++k2) {
            i__3 = 1;
            for (i__ = 1; i__ <= i__3; ++i__) {
            aer_trn_ttl__[i__ + (k1 + (k2 + bnd_idx__ *
                aer_trn_ttl_dim3) * aer_trn_ttl_dim2) *
                aer_trn_ttl_dim1] = aer_trn_ttl__[i__ + (k2 +
                (k1 + bnd_idx__ * aer_trn_ttl_dim3) *
                aer_trn_ttl_dim2) * aer_trn_ttl_dim1];

            }


        }


        }


    }

    } else {
    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
        i__2 = *plevp - 1;
        for (cam_s1_1__ = 0; cam_s1_1__ <= i__2; ++cam_s1_1__) {
        i__3 = *plevp - 1;
        for (cam_s1_2__ = 0; cam_s1_2__ <= i__3; ++cam_s1_2__) {
            for (cam_s1_3__ = 0; cam_s1_3__ <= 6; ++cam_s1_3__) {
            aer_trn_ttl__[cam_s1_0__ + 1 + (cam_s1_1__ + 1 + (
                cam_s1_2__ + 1 + (cam_s1_3__ + 1) *
                aer_trn_ttl_dim3) * aer_trn_ttl_dim2) *
                aer_trn_ttl_dim1] = 1.f;

            }


        }


        }


    }

    }
    return 0;
}





__device__ int sortarray_(CamState* state, integer *n, doublereal *ain, integer *indxa)
{
    integer i__1 = 0;

    integer i__ = 0,j = 0,ni = 0;

    doublereal atmp = 0;

    integer itmp = 0;


    --indxa;
    --ain;


    ni = 1;
    while(TRUE_) {
    ni = ni * 3 + 1;
    if (ni <= *n) {
        goto L9002;
    }
    goto L9003;
L9002:
    ;
    }
L9003:
    while(TRUE_) {
    ni /= 3;
    i__1 = *n;
    for (i__ = ni + 1; i__ <= i__1; ++i__) {
        atmp = ain[i__];
        itmp = indxa[i__];
        j = i__;
        while(TRUE_) {
        if (ain[j - ni] <= atmp) {
            goto L9009;
        }
        ain[j] = ain[j - ni];
        indxa[j] = indxa[j - ni];
        j -= ni;
        if (j > ni) {
            goto L9008;
        }
        goto L9009;
L9008:
        ;
        }
L9009:
        ain[j] = atmp;
        indxa[j] = itmp;

    }

    if (ni > 1) {
        goto L9004;
    }
    goto L9005;
L9004:
    ;
    }
L9005:
    return 0;
}






__device__ doublereal trcpth_c_b2 = 3.;
__device__ doublereal trcpth_c_b4 = 2.;


__device__ int trcpth_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *tnm, doublereal *pnm,
    doublereal *cfc11, doublereal *cfc12, doublereal *n2o, doublereal *
    ch4, doublereal *qnm, doublereal *ucfc11, doublereal *ucfc12,
    doublereal *un2o0, doublereal *un2o1, doublereal *uch4, doublereal *
    uco211, doublereal *uco212, doublereal *uco213, doublereal *uco221,
    doublereal *uco222, doublereal *uco223, doublereal *bn2o0, doublereal
    *bn2o1, doublereal *bch4, doublereal *uptype)
{
 CamScratchFrame scratch_frame(state);

    doublereal diff = 1.65999996662139893;

    integer tnm_dim1 = 0,tnm_offset = 0,pnm_dim1 = 0,pnm_offset = 0,qnm_dim1 = 0,qnm_offset = 0,cfc11_dim1 = 0,cfc11_offset = 0,cfc12_dim1 = 0,cfc12_offset = 0,n2o_dim1 = 0,n2o_offset = 0,ch4_dim1 = 0,ch4_offset = 0,ucfc11_dim1 = 0,ucfc11_offset = 0,ucfc12_dim1 = 0,ucfc12_offset = 0,un2o0_dim1 = 0,un2o0_offset = 0,un2o1_dim1 = 0,un2o1_offset = 0,uch4_dim1 = 0,uch4_offset = 0,uco211_dim1 = 0,uco211_offset = 0,uco212_dim1 = 0,uco212_offset = 0,uco213_dim1 = 0,uco213_offset = 0,uco221_dim1 = 0,uco221_offset = 0,uco222_dim1 = 0,uco222_offset = 0,uco223_dim1 = 0,uco223_offset = 0,bn2o0_dim1 = 0,bn2o0_offset = 0,bn2o1_dim1 = 0,bn2o1_offset = 0,bch4_dim1 = 0,bch4_offset = 0,uptype_dim1 = 0,uptype_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,k = 0;

    doublereal rt[1] = {},pbar[1] = {},dpnm[1] = {},co2fac[1] = {},alpha1[1] = {},alpha2[1] = {},rsqrt[1] = {};


    qnm_dim1 = 1;
    qnm_offset = 1 + qnm_dim1;
    qnm -= qnm_offset;
    ch4_dim1 = 1;
    ch4_offset = 1 + ch4_dim1;
    ch4 -= ch4_offset;
    n2o_dim1 = 1;
    n2o_offset = 1 + n2o_dim1;
    n2o -= n2o_offset;
    cfc12_dim1 = 1;
    cfc12_offset = 1 + cfc12_dim1;
    cfc12 -= cfc12_offset;
    cfc11_dim1 = 1;
    cfc11_offset = 1 + cfc11_dim1;
    cfc11 -= cfc11_offset;
    tnm_dim1 = 1;
    tnm_offset = 1 + tnm_dim1;
    tnm -= tnm_offset;
    uptype_dim1 = 1;
    uptype_offset = 1 + uptype_dim1;
    uptype -= uptype_offset;
    bch4_dim1 = 1;
    bch4_offset = 1 + bch4_dim1;
    bch4 -= bch4_offset;
    bn2o1_dim1 = 1;
    bn2o1_offset = 1 + bn2o1_dim1;
    bn2o1 -= bn2o1_offset;
    bn2o0_dim1 = 1;
    bn2o0_offset = 1 + bn2o0_dim1;
    bn2o0 -= bn2o0_offset;
    uco223_dim1 = 1;
    uco223_offset = 1 + uco223_dim1;
    uco223 -= uco223_offset;
    uco222_dim1 = 1;
    uco222_offset = 1 + uco222_dim1;
    uco222 -= uco222_offset;
    uco221_dim1 = 1;
    uco221_offset = 1 + uco221_dim1;
    uco221 -= uco221_offset;
    uco213_dim1 = 1;
    uco213_offset = 1 + uco213_dim1;
    uco213 -= uco213_offset;
    uco212_dim1 = 1;
    uco212_offset = 1 + uco212_dim1;
    uco212 -= uco212_offset;
    uco211_dim1 = 1;
    uco211_offset = 1 + uco211_dim1;
    uco211 -= uco211_offset;
    uch4_dim1 = 1;
    uch4_offset = 1 + uch4_dim1;
    uch4 -= uch4_offset;
    un2o1_dim1 = 1;
    un2o1_offset = 1 + un2o1_dim1;
    un2o1 -= un2o1_offset;
    un2o0_dim1 = 1;
    un2o0_offset = 1 + un2o0_dim1;
    un2o0 -= un2o0_offset;
    ucfc12_dim1 = 1;
    ucfc12_offset = 1 + ucfc12_dim1;
    ucfc12 -= ucfc12_offset;
    ucfc11_dim1 = 1;
    ucfc11_offset = 1 + ucfc11_dim1;
    ucfc11 -= ucfc11_offset;
    pnm_dim1 = 1;
    pnm_offset = 1 + pnm_dim1;
    pnm -= pnm_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    ucfc11[i__ + state->cg_ntoplw__ * ucfc11_dim1] = cfc11[i__ +
        state->cg_ntoplw__ * cfc11_dim1] * 1.79999995231628418f
        * pnm[i__ + state->cg_ntoplw__ * pnm_dim1] *
        state->cg_rga__;
    ucfc12[i__ + state->cg_ntoplw__ * ucfc12_dim1] = cfc12[i__ +
        state->cg_ntoplw__ * cfc12_dim1] * 1.79999995231628418f
        * pnm[i__ + state->cg_ntoplw__ * pnm_dim1] *
        state->cg_rga__;
    un2o0[i__ + state->cg_ntoplw__ * un2o0_dim1] = diff * 102346.f *
         n2o[i__ + state->cg_ntoplw__ * n2o_dim1] * pnm[i__ +
        state->cg_ntoplw__ * pnm_dim1] * state->cg_rga__ /
         sqrt(tnm[i__ + state->cg_ntoplw__ * tnm_dim1]);
    un2o1[i__ + state->cg_ntoplw__ * un2o1_dim1] = diff *
        2.01908993721008301f * un2o0[i__ + state->cg_ntoplw__ *
        un2o0_dim1] * cam_exp64(-847.3599853515625f / tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1]);
    uch4[i__ + state->cg_ntoplw__ * uch4_dim1] = diff *
        86095.703125f * ch4[i__ + state->cg_ntoplw__ * ch4_dim1]
         * pnm[i__ + state->cg_ntoplw__ * pnm_dim1] *
        state->cg_rga__ / sqrt(tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1]);
    co2fac[i__ - 1] = diff * state->cg_co2mmr__ * pnm[i__ +
        state->cg_ntoplw__ * pnm_dim1] * state->cg_rga__;
    d__1 = 1.f - cam_exp64(-1540.f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    alpha1[i__ - 1] = pow_dd(&d__1, &trcpth_c_b2) / sqrt(tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1]);
    d__1 = 1.f - cam_exp64(-1360.f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    alpha2[i__ - 1] = pow_dd(&d__1, &trcpth_c_b2) / sqrt(tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1]);
    uco211[i__ + state->cg_ntoplw__ * uco211_dim1] = co2fac[i__ - 1]
         * 3422.169921875f * alpha1[i__ - 1] * cam_exp64(
        -1849.699951171875f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    uco212[i__ + state->cg_ntoplw__ * uco212_dim1] = co2fac[i__ - 1]
         * 6024.5400390625f * alpha1[i__ - 1] * cam_exp64(
        -2782.10009765625f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    uco213[i__ + state->cg_ntoplw__ * uco213_dim1] = co2fac[i__ - 1]
         * 5531.43017578125f * alpha1[i__ - 1] * cam_exp64(
        -3723.199951171875f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    uco221[i__ + state->cg_ntoplw__ * uco221_dim1] = co2fac[i__ - 1]
         * 3889.840087890625f * alpha2[i__ - 1] * cam_exp64(
        -1997.5999755859375f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    uco222[i__ + state->cg_ntoplw__ * uco222_dim1] = co2fac[i__ - 1]
         * 3671.080078125f * alpha2[i__ - 1] * cam_exp64(
        -3843.800048828125f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    uco223[i__ + state->cg_ntoplw__ * uco223_dim1] = co2fac[i__ - 1]
         * 6506.419921875f * alpha2[i__ - 1] * cam_exp64(
        -2989.699951171875f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    bn2o0[i__ + state->cg_ntoplw__ * bn2o0_dim1] = diff *
        19.3990001678466797f * pow_dd(&pnm[i__ +
        state->cg_ntoplw__ * pnm_dim1], &trcpth_c_b4) * n2o[i__ +
        state->cg_ntoplw__ * n2o_dim1] * 102346.f *
        state->cg_rga__ / (state->cg_sslp__ * tnm[i__ +
        state->cg_ntoplw__ * tnm_dim1]);
    bn2o1[i__ + state->cg_ntoplw__ * bn2o1_dim1] = bn2o0[i__ +
        state->cg_ntoplw__ * bn2o0_dim1] * cam_exp64(
        -847.3599853515625f / tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]) * 206646.f;
    bch4[i__ + state->cg_ntoplw__ * bch4_dim1] = diff *
        2.94448995590209961f * ch4[i__ + state->cg_ntoplw__ *
        ch4_dim1] * pow_dd(&pnm[i__ + state->cg_ntoplw__ *
        pnm_dim1], &trcpth_c_b4) * state->cg_rga__ * 86095.703125f / (
        state->cg_sslp__ * tnm[i__ + state->cg_ntoplw__ *
        tnm_dim1]);
    uptype[i__ + state->cg_ntoplw__ * uptype_dim1] = diff * qnm[i__
        + state->cg_ntoplw__ * qnm_dim1] * pow_dd(&pnm[i__ +
        state->cg_ntoplw__ * pnm_dim1], &trcpth_c_b4) * cam_exp64((1.f / tnm[
        i__ + state->cg_ntoplw__ * tnm_dim1] -
        .0033783783783783786f) * 1800.f) * state->cg_rga__ /
        state->cg_sslp__;

    }

    i__1 = CAM_LEVELS;
    for (k = state->cg_ntoplw__; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        rt[i__ - 1] = 1.f / tnm[i__ + k * tnm_dim1];
        rsqrt[i__ - 1] = sqrt(rt[i__ - 1]);
        pbar[i__ - 1] = (pnm[i__ + (k + 1) * pnm_dim1] + pnm[i__ + k *
            pnm_dim1]) * .5f / state->cg_sslp__;
        dpnm[i__ - 1] = (pnm[i__ + (k + 1) * pnm_dim1] - pnm[i__ + k *
            pnm_dim1]) * state->cg_rga__;
        d__1 = 1.f - cam_exp64(-1540.f / tnm[i__ + k * tnm_dim1]);
        alpha1[i__ - 1] = diff * rsqrt[i__ - 1] * pow_dd(&d__1, &trcpth_c_b2);
        d__1 = 1.f - cam_exp64(-1360.f / tnm[i__ + k * tnm_dim1]);
        alpha2[i__ - 1] = diff * rsqrt[i__ - 1] * pow_dd(&d__1, &trcpth_c_b2);
        ucfc11[i__ + (k + 1) * ucfc11_dim1] = ucfc11[i__ + k *
            ucfc11_dim1] + cfc11[i__ + k * cfc11_dim1] *
            1.79999995231628418f * dpnm[i__ - 1];
        ucfc12[i__ + (k + 1) * ucfc12_dim1] = ucfc12[i__ + k *
            ucfc12_dim1] + cfc12[i__ + k * cfc12_dim1] *
            1.79999995231628418f * dpnm[i__ - 1];
        un2o0[i__ + (k + 1) * un2o0_dim1] = un2o0[i__ + k * un2o0_dim1] +
            diff * 102346.f * n2o[i__ + k * n2o_dim1] * rsqrt[i__ - 1]
             * dpnm[i__ - 1];
        un2o1[i__ + (k + 1) * un2o1_dim1] = un2o1[i__ + k * un2o1_dim1] +
            diff * 206646.f * n2o[i__ + k * n2o_dim1] * rsqrt[i__ - 1]
             * cam_exp64(-847.3599853515625f / tnm[i__ + k * tnm_dim1]) *
            dpnm[i__ - 1];
        uch4[i__ + (k + 1) * uch4_dim1] = uch4[i__ + k * uch4_dim1] +
            diff * 86095.703125f * ch4[i__ + k * ch4_dim1] * rsqrt[
            i__ - 1] * dpnm[i__ - 1];
        uco211[i__ + (k + 1) * uco211_dim1] = uco211[i__ + k *
            uco211_dim1] + alpha1[i__ - 1] * 3935.495328565361f *
            state->cg_co2mmr__ * cam_exp64(-1849.699951171875f / tnm[
            i__ + k * tnm_dim1]) * dpnm[i__ - 1];
        uco212[i__ + (k + 1) * uco212_dim1] = uco212[i__ + k *
            uco212_dim1] + alpha1[i__ - 1] * 6928.2209012856474f *
            state->cg_co2mmr__ * cam_exp64(-2782.10009765625f / tnm[
            i__ + k * tnm_dim1]) * dpnm[i__ - 1];
        uco213[i__ + (k + 1) * uco213_dim1] = uco213[i__ + k *
            uco213_dim1] + alpha1[i__ - 1] * 6361.1445702688652f *
            state->cg_co2mmr__ * cam_exp64(-3723.199951171875f / tnm[
            i__ + k * tnm_dim1]) * dpnm[i__ - 1];
        uco221[i__ + (k + 1) * uco221_dim1] = uco221[i__ + k *
            uco221_dim1] + alpha2[i__ - 1] * 4473.3160083332041f *
            state->cg_co2mmr__ * cam_exp64(-1997.5999755859375f / tnm[
            i__ + k * tnm_dim1]) * dpnm[i__ - 1];
        uco222[i__ + (k + 1) * uco222_dim1] = uco222[i__ + k *
            uco222_dim1] + alpha2[i__ - 1] * 4221.7420023183804f *
            state->cg_co2mmr__ * cam_exp64(-3843.800048828125f / tnm[
            i__ + k * tnm_dim1]) * dpnm[i__ - 1];
        uco223[i__ + (k + 1) * uco223_dim1] = uco223[i__ + k *
            uco223_dim1] + alpha2[i__ - 1] * 7482.3827550311107f *
            state->cg_co2mmr__ * cam_exp64(-2989.699951171875f / tnm[
            i__ + k * tnm_dim1]) * dpnm[i__ - 1];
        bn2o0[i__ + (k + 1) * bn2o0_dim1] = bn2o0[i__ + k * bn2o0_dim1] +
            diff * 19.3990001678466797f * pbar[i__ - 1] * rt[i__ - 1]
            * 102346.f * n2o[i__ + k * n2o_dim1] * dpnm[i__ - 1];
        bn2o1[i__ + (k + 1) * bn2o1_dim1] = bn2o1[i__ + k * bn2o1_dim1] +
            diff * 19.3990001678466797f * pbar[i__ - 1] * rt[i__ - 1]
            * 206646.f * cam_exp64(-847.3599853515625f / tnm[i__ + k *
            tnm_dim1]) * n2o[i__ + k * n2o_dim1] * dpnm[i__ - 1];
        bch4[i__ + (k + 1) * bch4_dim1] = bch4[i__ + k * bch4_dim1] +
            diff * 2.94448995590209961f * rt[i__ - 1] * pbar[i__ - 1]
            * 86095.703125f * ch4[i__ + k * ch4_dim1] * dpnm[i__ - 1];
        uptype[i__ + (k + 1) * uptype_dim1] = uptype[i__ + k *
            uptype_dim1] + diff * qnm[i__ + k * qnm_dim1] * cam_exp64((1.f /
             tnm[i__ + k * tnm_dim1] - .0033783783783783786f) *
            1800.f) * pbar[i__ - 1] * dpnm[i__ - 1];

    }


    }

    return 0;
}






__device__ integer radcswmx_c__1 = 1;


__device__ int radcswmx_(CamState* state, integer *jj, integer *lchnk, integer *ncol,
    integer *pcols, integer *pver, integer *pverp, doublereal *pint,
    doublereal *pmid, doublereal *h2ommr, doublereal *rh, doublereal *
    o3mmr, doublereal *aermmr, doublereal *cld, doublereal *cicewp,
    doublereal *cliqwp, doublereal *rel, doublereal *rei, doublereal *
    tauxcl, doublereal *tauxci, doublereal *eccf, doublereal *coszrs,
    doublereal *scon, doublereal *solin, real *solcon, doublereal *asdir,
    doublereal *asdif, doublereal *aldir, doublereal *aldif, integer *
    nmxrgn, doublereal *pmxrgn, doublereal *qrs, doublereal *qrscs,
    doublereal *fsnt, doublereal *fsntc, doublereal *fsntoa, doublereal *
    fsntoac, doublereal *fsnirtoa, doublereal *fsnrtoac, doublereal *
    fsnrtoaq, doublereal *fsns, doublereal *fsnsc, doublereal *fsdsc,
    doublereal *fsds, doublereal *sols, doublereal *soll, doublereal *
    solsd, doublereal *solld, doublereal *frc_day__, doublereal *fsup,
    doublereal *fsupc, doublereal *fsdn, doublereal *fsdnc, doublereal *
    fsdndir, doublereal *fsdncdir, doublereal *fsdndif, doublereal *
    fsdncdif, doublereal *fsdsdir, doublereal *fsdsdif, doublereal *
    fsdscdir, doublereal *fsdscdif, doublereal *aertau, doublereal *
    aerssa, doublereal *aerasm, doublereal *aerfwd)
{
 CamScratchFrame scratch_frame(state);

    integer pmid_dim1 = 0,pmid_offset = 0,pint_dim1 = 0,pint_offset = 0,h2ommr_dim1 = 0,h2ommr_offset = 0,o3mmr_dim1 = 0,o3mmr_offset = 0,aermmr_dim1 = 0,aermmr_dim2 = 0,aermmr_offset = 0,rh_dim1 = 0,rh_offset = 0,cld_dim1 = 0,cld_offset = 0,cicewp_dim1 = 0,cicewp_offset = 0,cliqwp_dim1 = 0,cliqwp_offset = 0,rel_dim1 = 0,rel_offset = 0,rei_dim1 = 0,rei_offset = 0,pmxrgn_dim1 = 0,pmxrgn_offset = 0,qrs_dim1 = 0,qrs_offset = 0,qrscs_dim1 = 0,qrscs_offset = 0,tauxcl_dim1 = 0,tauxcl_offset = 0,tauxci_dim1 = 0,tauxci_offset = 0,fsup_dim1 = 0,fsup_offset = 0,fsupc_dim1 = 0,fsupc_offset = 0,fsdn_dim1 = 0,fsdn_offset = 0,fsdnc_dim1 = 0,fsdnc_offset = 0,fsdndir_dim1 = 0,fsdndir_offset = 0,fsdncdir_dim1 = 0,fsdncdir_offset = 0,fsdndif_dim1 = 0,fsdndif_offset = 0,fsdncdif_dim1 = 0,fsdncdif_offset = 0,aertau_dim1 = 0,aertau_offset = 0,aerssa_dim1 = 0,aerssa_offset = 0,aerasm_dim1 = 0,aerasm_offset = 0,aerfwd_dim1 = 0,aerfwd_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0;

    doublereal d__1 = 0,d__2 = 0;

    doublereal tau_w_f_dst_tot__ = 0,tau_w_g_dst_tot__ = 0,tau_w_bg__ = 0,tau_w_cb__ = 0;

    integer nconfigm = 0;

    doublereal tau_volc__ = 0;

    logical new_term__ = 0;

    doublereal fswdndir[((CAM_LEVELS+1)-(0)+1)] = {}, tau_sslt__, trayoslp;

    integer i__ = 0,j = 0,l = 0,k = 0,n = 0,cam_red_k__ = 0;

    doublereal tau_cphob__ = 0,tau_cphil__ = 0;

    integer k1 = 0,k2 = 0,l0 = 0,n0 = 0,n1 = 0;

    doublereal fswdncdir[((CAM_LEVELS+1)-(0)+1)] = {}, tau_w_dst__[4] = {}, tau_w_sul__, f_dst_tot__,
         g_dst_tot__, tau_w_tot__, w_dst_tot__, fluxdndir[((CAM_LEVELS+1)-(0)+1)] = {};

    doublereal ga[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {} , fa[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}  , wa[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}  ;

    integer ns = 0;

    doublereal tau_w_f_cb__ = 0,tau_w_g_cb__ = 0,tau_w_f_bg__ = 0,tau_w_g_bg__ = 0;

    integer km1 = 0,is0 = 0,is1 = 0,kp1 = 0;

    doublereal tau_w_volc__ = 0;

    integer kx1[((CAM_LEVELS+1)-(0)+1)] = {}, kx2[((CAM_LEVELS+1)-(0)+1)] = {};

    doublereal uo2[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}    , uo3[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {} , tau_w_sslt__, fci[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}   , gci[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {} , fcl[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {} , gcl[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {} ;

    doublereal ucb[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , ubg[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {} , wci[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {} ;

    integer krh = 0;

    doublereal wcl[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}    ;

    integer isn = 0;

    doublereal psf[19] = {},wrh = 0;

    integer nxs = 0,ksz = 0;

    doublereal cld0, tau_w_cphob__, tau_w_cphil__, tau_w_g_dst__[4] = {},
        tau_w_f_dst__[4] = {}, tau_w_f_sul__, tau_w_g_sul__, tau_w_f_tot__,
        tau_w_g_tot__, uco2[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}   , uh2o[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}    , tmp1, tmp2;

    integer ptr0[15] = {},ptr1[15] = {};

    doublereal tau_dst_tot__ = 0,f_cb__ = 0,uto3[1] = {},uto2[1] = {};

    doublereal f_bg__ = 0;

    integer *ccon = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))    , kaer;

    doublereal pdel = 0,path = 0,atmp = 0;

    integer irgn, mrgn, *cstr = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*((CAM_LEVELS+1)-(1)+1))    , istr[((CAM_LEVELS+1)+1-(1)+1)] = {},
        ktmp, nrgn, ptrc[15] = {}, mstr, nstr[((CAM_LEVELS+1)-(1)+1)] = {};

    doublereal wgtv[15] = {}, xwgt, usul[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}   , udst[(4-(1)+1)*(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}
        , pflx[(1-(1)+1)*((CAM_LEVELS+1)-(0)+1)] = {}    , ptop, uaer[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}  , *wstr = cam_alloc<doublereal>(state, ((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , rdir[CAM_LEVELS+1] = {}
        , rdif[CAM_LEVELS+1] = {} , tdir[CAM_LEVELS+1] = {} , tdif[CAM_LEVELS+1] = {} , tau_w_f_volc__, tau_w_g_volc__;

    logical region_found__ = 0;

    doublereal tau_w_f_sslt__ = 0,tau_w_g_sslt__ = 0,ptho2 = 0,ptho3 = 0,tmp1l = 0,tmp2l = 0,tmp3l = 0,tmp1i = 0,tmp2i = 0,tmp3i = 0,uth2o[1] = {},utco2[1] = {};

    integer idayc[1] = {};

    doublereal rdifc[CAM_LEVELS+1] = {}   ;

    integer *icond = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))   , ndayc;

    doublereal f_dst__[4] = {}, tdifc[CAM_LEVELS+1] = {}   , rdirc[CAM_LEVELS+1] = {}    , tdirc[CAM_LEVELS+1] = {}    , f_sul__, f_tot__, xrdnd, yrdnd, ytdnd, asort[((CAM_LEVELS+1)-(1)+1)] = {}, zrdnd;

    integer *iconu = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))   , nuniq;

    doublereal fswdn[((CAM_LEVELS+1)-(0)+1)] = {}, uvolc[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}   , xtdnt,
        xrupd, yrupd, ytupd, zrupd, xexpt, yexpt, zexpt, ztdnt;

    integer ksort[((CAM_LEVELS+1)-(1)+1)] = {};

    doublereal xrups, fswup[((CAM_LEVELS+1)-(0)+1)] = {}, zrups, usslt[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {} , tau_w_g_cphil__, tau_w_g_cphob__, tau_w_f_cphil__,
        tau_w_f_cphob__, w_tot__, g_tot__, ksuli, wsuli, gsuli, pthco2,
        pthh2o, sqrco2, tau_w_dst_tot__, albdif[19] = {} ,
        h2ostr, abarli, albdir[19] = {}  , tau_cb__, bbarli,
        cbarli, tau_bg__, dbarli, ebarli, fbarli, zenfac[1] = {}, f_volc__,
        *rdndif = cam_flux_alloc<doublereal>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))   , abarii, ucphob[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  , bbarii, fswdnc[((CAM_LEVELS+1)-(0)+1)] = {}, ucphil[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  , rdenom,
        wavmid[19] = {}, cbarii, *rupdif = cam_flux_alloc<doublereal>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))   , f_sslt__;

    integer nuniqd[((CAM_LEVELS+1)-(0)+1)] = {}, indxsl;

    doublereal totfld[(CAM_LEVELS-(0)+1)] = {}, gsslti, solflx, fswupc[((CAM_LEVELS+1)-(0)+1)] = {};

    integer *istrtd = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))    , nuniqu[((CAM_LEVELS+1)-(0)+1)] = {};

    doublereal wgtint, sfltot, ksslti, wsslti, totwgt, tauxar[(1-(1)+1)*(CAM_LEVELS-(0)+1)] = {}
        ;

    integer *istrtu = cam_alloc<integer>(state, ((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))    ;

    doublereal explay[CAM_LEVELS+1] = {}  , flxdiv, *exptdn = cam_flux_alloc<doublereal>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1)) , *rupdir = cam_flux_alloc<doublereal>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1)) ,
        *tdntot = cam_flux_alloc<doublereal>(state, ((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))   , fluxup[((CAM_LEVELS+1)-(0)+1)] = {}, fluxdn[((CAM_LEVELS+1)-(0)+1)] = {}, dbarii,
        ebarii, fbarii;

    integer cam_s1_0__ = 0,cam_s2_0__ = 0,cam_s3_0__ = 0,cam_s4_0__ = 0,cam_s5_0__ = 0,cam_s6_0__ = 0,cam_s6_1__ = 0,cam_s7_1__ = 0,cam_s8_1__ = 0,cam_s9_0__ = 0;

    doublereal f_cphob__, f_cphil__, rdndifc[((CAM_LEVELS+1)-(0)+1)] = {};

    integer iconfig = 0;

    doublereal gcphili = 0;

    integer nconfig = 0;

    doublereal kcphili, rupdifc[((CAM_LEVELS+1)-(0)+1)] = {}, totfldc[(CAM_LEVELS-(0)+1)] = {}, wcphili;

    integer cam_s10_0__ = 0,cam_s11_1__ = 0,cam_s12_1__ = 0,cam_s13_1__ = 0,cam_s14_1__ = 0,cam_s15_1__ = 0,cam_s16_1__ = 0,cam_s17_1__ = 0,cam_s18_1__ = 0,cam_s19_1__ = 0,cam_s20_1__ = 0,cam_s21_1__ = 0,npasses = 0;

    doublereal rhtrunc, tau_sul__, tau_dst__[4] = {}, tau_tot__, rdirexp,
        tdnmexp, explayc[CAM_LEVELS+1] = {}   , exptdnc[((CAM_LEVELS+1)-(0)+1)] = {},
        rupdirc[((CAM_LEVELS+1)-(0)+1)] = {}, tdntotc[((CAM_LEVELS+1)-(0)+1)] = {}, wexptdn;


    aerfwd_dim1 = 1;
    aerfwd_offset = 1 + aerfwd_dim1 * 20;
    aerfwd -= aerfwd_offset;
    aerasm_dim1 = 1;
    aerasm_offset = 1 + aerasm_dim1 * 20;
    aerasm -= aerasm_offset;
    aerssa_dim1 = 1;
    aerssa_offset = 1 + aerssa_dim1 * 20;
    aerssa -= aerssa_offset;
    aertau_dim1 = 1;
    aertau_offset = 1 + aertau_dim1 * 20;
    aertau -= aertau_offset;
    --fsdscdif;
    --fsdscdir;
    --fsdsdif;
    --fsdsdir;
    --frc_day__;
    --solld;
    --solsd;
    --soll;
    --sols;
    --fsds;
    --fsdsc;
    --fsnsc;
    --fsns;
    --fsnrtoaq;
    --fsnrtoac;
    --fsnirtoa;
    --fsntoac;
    --fsntoa;
    --fsntc;
    --fsnt;
    --nmxrgn;
    --aldif;
    --aldir;
    --asdif;
    --asdir;
    --solin;
    --coszrs;
    qrscs_dim1 = 1;
    qrscs_offset = 1 + qrscs_dim1;
    qrscs -= qrscs_offset;
    qrs_dim1 = 1;
    qrs_offset = 1 + qrs_dim1;
    qrs -= qrs_offset;
    tauxci_dim1 = 1;
    tauxci_offset = 1 + tauxci_dim1 * 0;
    tauxci -= tauxci_offset;
    tauxcl_dim1 = 1;
    tauxcl_offset = 1 + tauxcl_dim1 * 0;
    tauxcl -= tauxcl_offset;
    rei_dim1 = 1;
    rei_offset = 1 + rei_dim1;
    rei -= rei_offset;
    rel_dim1 = 1;
    rel_offset = 1 + rel_dim1;
    rel -= rel_offset;
    cliqwp_dim1 = 1;
    cliqwp_offset = 1 + cliqwp_dim1;
    cliqwp -= cliqwp_offset;
    cicewp_dim1 = 1;
    cicewp_offset = 1 + cicewp_dim1;
    cicewp -= cicewp_offset;
    cld_dim1 = 1;
    cld_offset = 1 + cld_dim1;
    cld -= cld_offset;
    aermmr_dim1 = 1;
    aermmr_dim2 = CAM_LEVELS;
    aermmr_offset = 1 + aermmr_dim1 * (1 + aermmr_dim2);
    aermmr -= aermmr_offset;
    o3mmr_dim1 = 1;
    o3mmr_offset = 1 + o3mmr_dim1;
    o3mmr -= o3mmr_offset;
    rh_dim1 = 1;
    rh_offset = 1 + rh_dim1;
    rh -= rh_offset;
    h2ommr_dim1 = 1;
    h2ommr_offset = 1 + h2ommr_dim1;
    h2ommr -= h2ommr_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    fsdncdif_dim1 = 1;
    fsdncdif_offset = 1 + fsdncdif_dim1;
    fsdncdif -= fsdncdif_offset;
    fsdndif_dim1 = 1;
    fsdndif_offset = 1 + fsdndif_dim1;
    fsdndif -= fsdndif_offset;
    fsdncdir_dim1 = 1;
    fsdncdir_offset = 1 + fsdncdir_dim1;
    fsdncdir -= fsdncdir_offset;
    fsdndir_dim1 = 1;
    fsdndir_offset = 1 + fsdndir_dim1;
    fsdndir -= fsdndir_offset;
    fsdnc_dim1 = 1;
    fsdnc_offset = 1 + fsdnc_dim1;
    fsdnc -= fsdnc_offset;
    fsdn_dim1 = 1;
    fsdn_offset = 1 + fsdn_dim1;
    fsdn -= fsdn_offset;
    fsupc_dim1 = 1;
    fsupc_offset = 1 + fsupc_dim1;
    fsupc -= fsupc_offset;
    fsup_dim1 = 1;
    fsup_offset = 1 + fsup_dim1;
    fsup -= fsup_offset;
    pmxrgn_dim1 = 1;
    pmxrgn_offset = 1 + pmxrgn_dim1;
    pmxrgn -= pmxrgn_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    fsds[i__] = 0.;
    fsnirtoa[i__] = 0.;
    fsnrtoac[i__] = 0.;
    fsnrtoaq[i__] = 0.;
    fsns[i__] = 0.;
    fsnsc[i__] = 0.;
    fsdsc[i__] = 0.;
    fsdscdir[i__] = 0.;
    fsdscdif[i__] = 0.;
    fsdsdir[i__] = 0.;
    fsdsdif[i__] = 0.;
    fsnt[i__] = 0.;
    fsntc[i__] = 0.;
    fsntoa[i__] = 0.;
    fsntoac[i__] = 0.;
    solin[i__] = 0.;
    sols[i__] = 0.;
    soll[i__] = 0.;
    solsd[i__] = 0.;
    solld[i__] = 0.;
    i__2 = (CAM_LEVELS+1);
    for (k = 1; k <= i__2; ++k) {
        fsup[i__ + k * fsup_dim1] = 0.;
        fsupc[i__ + k * fsupc_dim1] = 0.;
        fsdn[i__ + k * fsdn_dim1] = 0.;
        fsdnc[i__ + k * fsdnc_dim1] = 0.;
        fsdndir[i__ + k * fsdndir_dim1] = 0.;
        fsdncdir[i__ + k * fsdncdir_dim1] = 0.;
        fsdndif[i__ + k * fsdndif_dim1] = 0.;
        fsdncdif[i__ + k * fsdncdif_dim1] = 0.;
        tauxcl[i__ + (k - 1) * tauxcl_dim1] = 0.;
        tauxci[i__ + (k - 1) * tauxci_dim1] = 0.;

    }

    i__2 = CAM_LEVELS;
    for (k = 1; k <= i__2; ++k) {
        qrs[i__ + k * qrs_dim1] = 0.;
        qrscs[i__ + k * qrscs_dim1] = 0.;

    }

    for (kaer = 1; kaer <= 7; ++kaer) {
        for (ns = 1; ns <= 19; ++ns) {
        frc_day__[i__] = 0.;
        aertau[i__ + (ns + kaer * 19) * aertau_dim1] = 0.;
        aerssa[i__ + (ns + kaer * 19) * aerssa_dim1] = 0.;
        aerasm[i__ + (ns + kaer * 19) * aerasm_dim1] = 0.;
        aerfwd[i__ + (ns + kaer * 19) * aerfwd_dim1] = 0.;

        }


    }


    }

    ndayc = 0;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    if (coszrs[i__] > 0.) {
        ++ndayc;
        idayc[ndayc - 1] = i__;
    }

    }

    if (ndayc == 0) {
    return 0;
    }
    tmp1 = .5 / (state->cg_gravit__ * state->cg_sslp__);
    tmp2 = state->cg_delta__ / state->cg_gravit__;
    sqrco2 = sqrt(state->cg_co2mmr__);
    i__1 = ndayc;
    for (n = 1; n <= i__1; ++n) {
    i__ = idayc[n - 1];
    solin[i__] = *solcon * coszrs[i__] * 1e3f;
    pflx[i__ - 1] = 0.;
    i__2 = (CAM_LEVELS+1);
    for (k = 1; k <= i__2; ++k) {
        pflx[i__ + k - 1] = pint[i__ + k * pint_dim1];

    }

    ptop = pflx[i__];
    ptho2 = state->cg_o2mmr__ * ptop / state->cg_gravit__;
    ptho3 = o3mmr[i__ + o3mmr_dim1] * ptop / state->cg_gravit__;
    pthco2 = sqrco2 * (ptop / state->cg_gravit__);
    h2ostr = sqrt(1. / h2ommr[i__ + h2ommr_dim1]);
    zenfac[i__ - 1] = sqrt(coszrs[i__]);

    d__1 = ptop;
    pthh2o = d__1 * d__1 * tmp1 + ptop * state->cg_rga__ * (h2ostr *
         zenfac[i__ - 1] * state->cg_delta__);
    uh2o[i__ - 1] = h2ommr[i__ + h2ommr_dim1] * pthh2o;
    uco2[i__ - 1] = zenfac[i__ - 1] * pthco2;
    uo2[i__ - 1] = zenfac[i__ - 1] * ptho2;
    uo3[i__ - 1] = ptho3;
    uaer[i__ - 1] = 0.;
    i__2 = CAM_LEVELS;
    for (k = 1; k <= i__2; ++k) {
        pdel = pflx[i__ + (k + 1) - 1] - pflx[i__ + k - 1];
        path = pdel / state->cg_gravit__;
        ptho2 = state->cg_o2mmr__ * path;
        ptho3 = o3mmr[i__ + k * o3mmr_dim1] * path;
        pthco2 = sqrco2 * path;
        h2ostr = sqrt(1. / h2ommr[i__ + k * h2ommr_dim1]);

        d__1 = pflx[i__ + (k + 1) - 1];

        d__2 = pflx[i__ + k - 1];
        pthh2o = (d__1 * d__1 - d__2 * d__2) * tmp1 + pdel * h2ostr *
            zenfac[i__ - 1] * tmp2;
        uh2o[i__ + k - 1] = h2ommr[i__ + k * h2ommr_dim1] * pthh2o;
        uco2[i__ + k - 1] = zenfac[i__ - 1] * pthco2;
        uo2[i__ + k - 1] = zenfac[i__ - 1] * ptho2;
        uo3[i__ + k - 1] = ptho3;
        usul[i__ + k - 2] = aermmr[i__ + (k + state->cg_idxsul__ *
            aermmr_dim2) * aermmr_dim1] * path;
        ubg[i__ + k - 2] = aermmr[i__ + (k + state->cg_idxbg__ *
            aermmr_dim2) * aermmr_dim1] * path;
        usslt[i__ + k - 2] = aermmr[i__ + (k + state->cg_idxsslt__ *
             aermmr_dim2) * aermmr_dim1] * path;
        if (usslt[i__ + k - 2] < 0.f) {
        usslt[i__ + k - 2] = 0.f;
        }
        ucphil[i__ + k - 2] = aermmr[i__ + (k +
            state->cg_idxocphi__ * aermmr_dim2) * aermmr_dim1] *
             path;
        ucphob[i__ + k - 2] = aermmr[i__ + (k +
            state->cg_idxocpho__ * aermmr_dim2) * aermmr_dim1] *
             path;
        ucb[i__ + k - 2] = (aermmr[i__ + (k + state->cg_idxbcpho__ *
             aermmr_dim2) * aermmr_dim1] + aermmr[i__ + (k +
            state->cg_idxbcphi__ * aermmr_dim2) * aermmr_dim1])
            * path;
        uvolc[i__ + k - 2] = aermmr[i__ + (k + state->cg_idxvolc__ *
             aermmr_dim2) * aermmr_dim1];
        for (ksz = 1; ksz <= 4; ++ksz) {
        udst[ksz + (i__ + k << 2) - 9] = aermmr[i__ + (k + (
            state->cg_idxdustfirst__ - 1 + ksz) *
            aermmr_dim2) * aermmr_dim1] * path;

        }


    }

    uth2o[i__ - 1] = 0.;
    uto3[i__ - 1] = 0.;
    utco2[i__ - 1] = 0.;
    uto2[i__ - 1] = 0.;
    i__2 = CAM_LEVELS;
    for (k = 1; k <= i__2; ++k) {
        uth2o[i__ - 1] += uh2o[i__ + k - 1];
        uto3[i__ - 1] += uo3[i__ + k - 1];
        utco2[i__ - 1] += uco2[i__ + k - 1];
        uto2[i__ - 1] += uo2[i__ + k - 1];

    }

    tauxcl[i__] = 0.;
    wcl[i__ - 1] = .999999;
    gcl[i__ - 1] = .85;
    fcl[i__ - 1] = .725;
    tauxci[i__] = 0.;
    wci[i__ - 1] = .999999;
    gci[i__ - 1] = .85;
    fci[i__ - 1] = .725;
    tauxar[i__ - 1] = 0.;
    wa[i__ - 1] = .925;
    ga[i__ - 1] = .85;
    fa[i__ - 1] = .7225;

    }



    i__1 = ndayc;
    for (n = 1; n <= i__1; ++n) {
    i__ = idayc[n - 1];
    npasses = 0;
    while(TRUE_) {
        i__2 = nmxrgn[i__];
        for (irgn = 0; irgn <= i__2; ++irgn) {
        kx2[irgn] = 0;

        }

        mrgn = 0;
        i__2 = nmxrgn[i__];
        for (irgn = 1; irgn <= i__2; ++irgn) {
        region_found__ = FALSE_;
        if (kx2[irgn - 1] < CAM_LEVELS) {
            k1 = kx2[irgn - 1] + 1;
            kx1[irgn] = k1;
            kx2[irgn] = k1 - 1;
            i__3 = k1;
            for (k2 = CAM_LEVELS; k2 >= i__3; --k2) {
            if (pmid[i__ + k2 * pmid_dim1] <= pmxrgn[i__ + irgn *
                pmxrgn_dim1]) {
                kx2[irgn] = k2;
                ++mrgn;
                region_found__ = TRUE_;
                goto L9065;
            }

            }
L9065:
            ;
        } else {
            goto L9063;
        }
        if (region_found__) {
            nxs = 0;
            if (FALSE_) {
            i__3 = k2;
            for (k = k1; k <= i__3; ++k) {
                if (cld[i__ + k * cld_dim1] >= 1e-80 && cld[i__ +
                    k * cld_dim1] >= 0.) {
                ++nxs;
                ksort[nxs - 1] = k;
                d__1 = cld[i__ + k * cld_dim1] / 0.;
                asort[nxs - 1] = 1. - cam_floor__(&d__1) * 0.;
                }

            }

            } else {
            i__3 = k2;
            for (k = k1; k <= i__3; ++k) {
                if (cld[i__ + k * cld_dim1] >= 1e-80) {
                ++nxs;
                ksort[nxs - 1] = k;
                asort[nxs - 1] = 1. - cld[i__ + k * cld_dim1];
                }

            }

            }
            if (nxs == 2) {
            if (asort[1] < asort[0]) {
                ktmp = ksort[0];
                ksort[0] = ksort[1];
                ksort[1] = ktmp;
                atmp = asort[0];
                asort[0] = asort[1];
                asort[1] = atmp;
            }
            } else if (nxs >= 3) {
            sortarray_(state, &nxs, asort, ksort);
            }
            i__3 = k2 - k1;
            for (cam_s6_0__ = 0; cam_s6_0__ <= i__3; ++cam_s6_0__) {
            i__4 = nxs;
            for (cam_s6_1__ = 0; cam_s6_1__ <= i__4; ++cam_s6_1__)
                 {
                cstr[k1 + cam_s6_0__ + (cam_s6_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))]
                     = 0;

            }


            }

            mstr = 1;
            cld0 = 0.;
            i__3 = nxs;
            for (l = 1; l <= i__3; ++l) {
            if (asort[l - 1] != cld0) {
                wstr[mstr + mrgn * ((1)*(((CAM_LEVELS+1)-(1)+1))) - ((1)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(1)+1))))] = asort[l - 1] - cld0;
                cld0 = asort[l - 1];
                ++mstr;
            }
            i__4 = nxs + 1 - mstr;
            for (cam_s7_1__ = 0; cam_s7_1__ <= i__4; ++cam_s7_1__)
                 {
                cstr[ksort[l - 1] + (mstr + cam_s7_1__) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))]
                     = 1;

            }


            }

            nstr[mrgn - 1] = mstr;
            wstr[mstr + mrgn * ((1)*(((CAM_LEVELS+1)-(1)+1))) - ((1)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(1)+1))))] = 1. - cld0;
        }

        }
L9063:
        nrgn = mrgn;
        i__2 = nstr[0] - 1;
        for (cam_s8_1__ = 0; cam_s8_1__ <= i__2; ++cam_s8_1__) {
        cstr[(cam_s8_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 0;

        }

        nconfigm = 1;
        i__2 = nrgn;
        for (cam_red_k__ = 1; cam_red_k__ <= i__2; ++cam_red_k__) {
        nconfigm *= nstr[cam_red_k__ - 1];

        }

        i__2 = nrgn - 1;
        for (cam_s9_0__ = 0; cam_s9_0__ <= i__2; ++cam_s9_0__) {
        istr[cam_s9_0__] = 1;

        }

        nconfig = 0;
        totwgt = 0.;
        new_term__ = TRUE_;
        i__2 = nconfigm;
        for (iconfig = 1; iconfig <= i__2; ++iconfig) {
        xwgt = 1.;
        i__3 = nrgn;
        for (mrgn = 1; mrgn <= i__3; ++mrgn) {
            xwgt *= wstr[istr[mrgn - 1] + mrgn * ((1)*(((CAM_LEVELS+1)-(1)+1))) - ((1)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(1)+1))))];

        }

        if (xwgt >= .01) {
            ++nconfig;
            if (nconfig <= 15) {
            j = nconfig;
            ptrc[nconfig - 1] = nconfig;
            } else {
            nconfig = 15;
            if (new_term__) {
                j = findvalue_(state, &radcswmx_c__1, &nconfig, wgtv, ptrc);
            }
            if (wgtv[j - 1] < xwgt) {
                totwgt -= wgtv[j - 1];
                new_term__ = TRUE_;
            } else {
                new_term__ = FALSE_;
            }
            }
            if (new_term__) {
            wgtv[j - 1] = xwgt;
            totwgt += xwgt;
            i__3 = nrgn;
            for (mrgn = 1; mrgn <= i__3; ++mrgn) {
                i__4 = kx2[mrgn] - kx1[mrgn];
                for (cam_s10_0__ = 0; cam_s10_0__ <= i__4;
                    ++cam_s10_0__) {
                ccon[kx1[mrgn] + cam_s10_0__ + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] =
                    cstr[kx1[mrgn] + cam_s10_0__ + istr[
                    mrgn - 1] * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];

                }


            }

            }
        }
        mrgn = nrgn;
        ++istr[mrgn - 1];
        while(istr[mrgn - 1] > nstr[mrgn - 1] && mrgn > 1) {
            istr[mrgn - 1] = 1;
            --mrgn;
            ++istr[mrgn - 1];

        }


        }

        if (totwgt > 0.f) {
        goto L9059;
        } else {
        ++npasses;
        if (npasses >= 2) {
            cam_overlap_failure__();
            endrun_();
        }
        nmxrgn[i__] = 1;
        pmxrgn[i__ + pmxrgn_dim1] = 1.00000001504746622e30f;
        }

    }
L9059:
    for (cam_s11_1__ = 0; cam_s11_1__ <= 14; ++cam_s11_1__) {
        ccon[(cam_s11_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 0;

    }

    for (cam_s12_1__ = 0; cam_s12_1__ <= 14; ++cam_s12_1__) {
        ccon[(CAM_LEVELS+1) + (cam_s12_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 0;

    }

    nuniqd[0] = 1;
    nuniqu[(CAM_LEVELS+1)] = 1;
    istrtd[((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 1;
    istrtu[(CAM_LEVELS+1) + ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 1;
    i__2 = nconfig;
    for (j = 1; j <= i__2; ++j) {
        icond[j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = j;
        iconu[(CAM_LEVELS+1) + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = j;

    }

    istrtd[((2)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = nconfig + 1;
    istrtu[(CAM_LEVELS+1) + ((2)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = nconfig + 1;
    i__2 = (CAM_LEVELS+1);
    for (k = 1; k <= i__2; ++k) {
        km1 = k - 1;
        nuniq = 0;
        istrtd[k + ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 1;
        i__3 = nuniqd[km1];
        for (l0 = 1; l0 <= i__3; ++l0) {
        is0 = istrtd[km1 + l0 * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
        is1 = istrtd[km1 + (l0 + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1;
        n0 = 0;
        n1 = 0;
        i__4 = is1;
        for (isn = is0; isn <= i__4; ++isn) {
            j = icond[km1 + isn * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            if (ccon[k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] == 0) {
            ++n0;
            ptr0[n0 - 1] = j;
            }
            if (ccon[k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] == 1) {
            ++n1;
            ptr1[n1 - 1] = j;
            }

        }

        if (n0 > 0) {
            ++nuniq;
            istrtd[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = istrtd[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] +
                n0;
            i__4 = istrtd[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1 - istrtd[k +
                nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            for (cam_s13_1__ = 0; cam_s13_1__ <= i__4; ++cam_s13_1__)
                {
            icond[k + (istrtd[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] + cam_s13_1__) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))
                - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = ptr0[cam_s13_1__];

            }

        }
        if (n1 > 0) {
            ++nuniq;
            istrtd[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = istrtd[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] +
                n1;
            i__4 = istrtd[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1 - istrtd[k +
                nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            for (cam_s14_1__ = 0; cam_s14_1__ <= i__4; ++cam_s14_1__)
                {
            icond[k + (istrtd[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] + cam_s14_1__) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))
                - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = ptr1[cam_s14_1__];

            }

        }

        }

        nuniqd[k] = nuniq;

    }

    for (k = CAM_LEVELS; k >= 0; --k) {
        kp1 = k + 1;
        nuniq = 0;
        istrtu[k + ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = 1;
        i__2 = nuniqu[kp1];
        for (l0 = 1; l0 <= i__2; ++l0) {
        is0 = istrtu[kp1 + l0 * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
        is1 = istrtu[kp1 + (l0 + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1;
        n0 = 0;
        n1 = 0;
        i__3 = is1;
        for (isn = is0; isn <= i__3; ++isn) {
            j = iconu[kp1 + isn * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            if (ccon[k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] == 0) {
            ++n0;
            ptr0[n0 - 1] = j;
            }
            if (ccon[k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] == 1) {
            ++n1;
            ptr1[n1 - 1] = j;
            }

        }

        if (n0 > 0) {
            ++nuniq;
            istrtu[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = istrtu[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] +
                n0;
            i__3 = istrtu[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1 - istrtu[k +
                nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            for (cam_s15_1__ = 0; cam_s15_1__ <= i__3; ++cam_s15_1__)
                {
            iconu[k + (istrtu[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] + cam_s15_1__) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))
                - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = ptr0[cam_s15_1__];

            }

        }
        if (n1 > 0) {
            ++nuniq;
            istrtu[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = istrtu[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] +
                n1;
            i__3 = istrtu[k + (nuniq + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1 - istrtu[k +
                nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            for (cam_s16_1__ = 0; cam_s16_1__ <= i__3; ++cam_s16_1__)
                {
            iconu[k + (istrtu[k + nuniq * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] + cam_s16_1__) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))
                - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] = ptr1[cam_s16_1__];

            }

        }

        }

        nuniqu[k] = nuniq;

    }

    i__2 = CAM_LEVELS;
    for (k = 0; k <= i__2; ++k) {
        totfld[k] = 0.;
        totfldc[k] = 0.;
        fswup[k] = 0.;
        fswdn[k] = 0.;
        fswupc[k] = 0.;
        fswdnc[k] = 0.;
        fswdndir[k] = 0.;
        fswdncdir[k] = 0.;

    }

    sfltot = 0.;
    fswup[(CAM_LEVELS+1)] = 0.;
    fswdn[(CAM_LEVELS+1)] = 0.;
    fswupc[(CAM_LEVELS+1)] = 0.;
    fswdnc[(CAM_LEVELS+1)] = 0.;
    fswdndir[(CAM_LEVELS+1)] = 0.;
    fswdncdir[(CAM_LEVELS+1)] = 0.;
    for (ns = 1; ns <= 19; ++ns) {
    if (state->cg_wavmax__[ns - 1] <= .7) {
        indxsl = 1;
    } else if (state->cg_wavmin__[ns - 1] == .7) {
        indxsl = 2;
    } else if (state->cg_wavmin__[ns - 1] == .701) {
        indxsl = 3;
    } else if (state->cg_wavmin__[ns - 1] == .702 ||
        state->cg_wavmin__[ns - 1] > 2.38) {
        indxsl = 4;
    }
    abarli = state->cg_abarl__[indxsl - 1];
    bbarli = state->cg_bbarl__[indxsl - 1];
    cbarli = state->cg_cbarl__[indxsl - 1];
    dbarli = state->cg_dbarl__[indxsl - 1];
    ebarli = state->cg_ebarl__[indxsl - 1];
    fbarli = state->cg_fbarl__[indxsl - 1];
    abarii = state->cg_abari__[indxsl - 1];
    bbarii = state->cg_bbari__[indxsl - 1];
    cbarii = state->cg_cbari__[indxsl - 1];
    dbarii = state->cg_dbari__[indxsl - 1];
    ebarii = state->cg_ebari__[indxsl - 1];
    fbarii = state->cg_fbari__[indxsl - 1];
    psf[ns - 1] = 1.;
    if (state->cg_ph2o__[ns - 1] != 0.) {
        psf[ns - 1] *= state->cg_ph2o__[ns - 1];
    }
    if (state->cg_pco2__[ns - 1] != 0.) {
        psf[ns - 1] *= state->cg_pco2__[ns - 1];
    }
    if (state->cg_po2__[ns - 1] != 0.) {
        psf[ns - 1] *= state->cg_po2__[ns - 1];
    }
    i__1 = ndayc;
    for (n = 1; n <= i__1; ++n) {
        i__ = idayc[n - 1];
        frc_day__[i__] = 1.;
        for (kaer = 1; kaer <= 7; ++kaer) {
        aertau[i__ + (ns + kaer * 19) * aertau_dim1] = 0.f;
        aerssa[i__ + (ns + kaer * 19) * aerssa_dim1] = 0.f;
        aerasm[i__ + (ns + kaer * 19) * aerasm_dim1] = 0.f;
        aerfwd[i__ + (ns + kaer * 19) * aerfwd_dim1] = 0.f;

        }

        i__2 = CAM_LEVELS;
        for (k = 1; k <= i__2; ++k) {
        tmp1l = abarli + bbarli / rel[i__ + k * rel_dim1];
        tmp2l = 1. - cbarli - dbarli * rel[i__ + k * rel_dim1];
        tmp3l = fbarli * rel[i__ + k * rel_dim1];
        tmp1i = abarii + bbarii / rei[i__ + k * rei_dim1];
        tmp2i = 1. - cbarii - dbarii * rei[i__ + k * rei_dim1];
        tmp3i = fbarii * rei[i__ + k * rei_dim1];
        if (cld[i__ + k * cld_dim1] >= 1e-80 && cld[i__ + k *
            cld_dim1] >= 0.) {
            tauxcl[i__ + k * tauxcl_dim1] = cliqwp[i__ + k *
                cliqwp_dim1] * tmp1l;
            tauxci[i__ + k * tauxci_dim1] = cicewp[i__ + k *
                cicewp_dim1] * tmp1i;
        } else {
            tauxcl[i__ + k * tauxcl_dim1] = 0.f;
            tauxci[i__ + k * tauxci_dim1] = 0.f;
        }
        wcl[i__ + k - 1] = min(tmp2l,.999999);
        gcl[i__ + k - 1] = ebarli + tmp3l;
        fcl[i__ + k - 1] = gcl[i__ + k - 1] * gcl[i__ + k - 1];
        wci[i__ + k - 1] = min(tmp2i,.999999);
        gci[i__ + k - 1] = ebarii + tmp3i;
        fci[i__ + k - 1] = gci[i__ + k - 1] * gci[i__ + k - 1];
        rhtrunc = rh[i__ + k * rh_dim1];

        d__1 = rh[i__ + k * rh_dim1];
        rhtrunc = min(d__1,1.);

        d__1 = rhtrunc * 1000;
        i__3 = cam_floor__(&d__1) + 1;
        krh = min(i__3,999);
        wrh = rhtrunc * 1000 - krh;
        ksuli = state->cg_ksul__[krh + 1 + ns * 1000 - 1001] * (
            wrh + 1) - state->cg_ksul__[krh + ns * 1000 -
            1001] * wrh;
        ksslti = state->cg_ksslt__[krh + 1 + ns * 1000 - 1001] *
             (wrh + 1) - state->cg_ksslt__[krh + ns * 1000
            - 1001] * wrh;
        kcphili = state->cg_kcphil__[krh + 1 + ns * 1000 - 1001]
             * (wrh + 1) - state->cg_kcphil__[krh + ns *
            1000 - 1001] * wrh;
        wsuli = state->cg_wsul__[krh + 1 + ns * 1000 - 1001] * (
            wrh + 1) - state->cg_wsul__[krh + ns * 1000 -
            1001] * wrh;
        wsslti = state->cg_wsslt__[krh + 1 + ns * 1000 - 1001] *
             (wrh + 1) - state->cg_wsslt__[krh + ns * 1000
            - 1001] * wrh;
        wcphili = state->cg_wcphil__[krh + 1 + ns * 1000 - 1001]
             * (wrh + 1) - state->cg_wcphil__[krh + ns *
            1000 - 1001] * wrh;
        gsuli = state->cg_gsul__[krh + 1 + ns * 1000 - 1001] * (
            wrh + 1) - state->cg_gsul__[krh + ns * 1000 -
            1001] * wrh;
        gsslti = state->cg_gsslt__[krh + 1 + ns * 1000 - 1001] *
             (wrh + 1) - state->cg_gsslt__[krh + ns * 1000
            - 1001] * wrh;
        gcphili = state->cg_gcphil__[krh + 1 + ns * 1000 - 1001]
             * (wrh + 1) - state->cg_gcphil__[krh + ns *
            1000 - 1001] * wrh;
        tau_sul__ = ksuli * 1e4f * usul[i__ + k - 2];
        tau_sslt__ = ksslti * 1e4f * usslt[i__ + k - 2];
        tau_cphil__ = kcphili * 1e4f * ucphil[i__ + k - 2];
        tau_cphob__ = state->cg_kcphob__[ns - 1] * 1e4f *
            ucphob[i__ + k - 2];
        tau_cb__ = state->cg_kcb__[ns - 1] * 1e4f * ucb[i__ + k
            - 2];
        tau_volc__ = state->cg_kvolc__[ns - 1] * 1e3f * uvolc[
            i__ + k - 2];
        for (cam_s1_0__ = 0; cam_s1_0__ <= 3; ++cam_s1_0__) {
            tau_dst__[cam_s1_0__] = state->cg_kdst__[cam_s1_0__
                + 1 + (ns << 2) - 5] * 1e4f * udst[cam_s1_0__ + 1
                + (i__ + k << 2) - 9];

        }

        tau_bg__ = state->cg_kbg__[ns - 1] * 1e4f * ubg[i__ + k
            - 2];
        tau_w_sul__ = tau_sul__ * wsuli;
        tau_w_sslt__ = tau_sslt__ * wsslti;
        tau_w_cphil__ = tau_cphil__ * wcphili;
        tau_w_cphob__ = tau_cphob__ * state->cg_wcphob__[ns - 1]
            ;
        tau_w_cb__ = tau_cb__ * state->cg_wcb__[ns - 1];
        tau_w_volc__ = tau_volc__ * state->cg_wvolc__[ns - 1];
        for (cam_s2_0__ = 0; cam_s2_0__ <= 3; ++cam_s2_0__) {
            tau_w_dst__[cam_s2_0__] = tau_dst__[cam_s2_0__] *
                state->cg_wdst__[cam_s2_0__ + 1 + (ns << 2)
                - 5];

        }

        tau_w_bg__ = tau_bg__ * state->cg_wbg__[ns - 1];
        tau_w_g_sul__ = tau_w_sul__ * gsuli;
        tau_w_g_sslt__ = tau_w_sslt__ * gsslti;
        tau_w_g_cphil__ = tau_w_cphil__ * gcphili;
        tau_w_g_cphob__ = tau_w_cphob__ * state->cg_gcphob__[ns
            - 1];
        tau_w_g_cb__ = tau_w_cb__ * state->cg_gcb__[ns - 1];
        tau_w_g_volc__ = tau_w_volc__ * state->cg_gvolc__[ns -
            1];
        for (cam_s3_0__ = 0; cam_s3_0__ <= 3; ++cam_s3_0__) {
            tau_w_g_dst__[cam_s3_0__] = tau_w_dst__[cam_s3_0__] *
                state->cg_gdst__[cam_s3_0__ + 1 + (ns << 2)
                - 5];

        }

        tau_w_g_bg__ = tau_w_bg__ * state->cg_gbg__[ns - 1];
        f_sul__ = gsuli * gsuli;
        f_sslt__ = gsslti * gsslti;
        f_cphil__ = gcphili * gcphili;
        f_cphob__ = state->cg_gcphob__[ns - 1] *
            state->cg_gcphob__[ns - 1];
        f_cb__ = state->cg_gcb__[ns - 1] *
            state->cg_gcb__[ns - 1];
        f_volc__ = state->cg_gvolc__[ns - 1] *
            state->cg_gvolc__[ns - 1];
        for (cam_s4_0__ = 0; cam_s4_0__ <= 3; ++cam_s4_0__) {
            f_dst__[cam_s4_0__] = state->cg_gdst__[cam_s4_0__ +
                1 + (ns << 2) - 5] * state->cg_gdst__[
                cam_s4_0__ + 1 + (ns << 2) - 5];

        }

        f_bg__ = state->cg_gbg__[ns - 1] *
            state->cg_gbg__[ns - 1];
        tau_w_f_sul__ = tau_w_sul__ * f_sul__;
        tau_w_f_bg__ = tau_w_bg__ * f_bg__;
        tau_w_f_sslt__ = tau_w_sslt__ * f_sslt__;
        tau_w_f_cphil__ = tau_w_cphil__ * f_cphil__;
        tau_w_f_cphob__ = tau_w_cphob__ * f_cphob__;
        tau_w_f_cb__ = tau_w_cb__ * f_cb__;
        tau_w_f_volc__ = tau_w_volc__ * f_volc__;
        for (cam_s5_0__ = 0; cam_s5_0__ <= 3; ++cam_s5_0__) {
            tau_w_f_dst__[cam_s5_0__] = tau_w_dst__[cam_s5_0__] *
                f_dst__[cam_s5_0__];

        }

        tau_dst_tot__ = 0.;
        for (cam_red_k__ = 1; cam_red_k__ <= 4; ++cam_red_k__) {
            tau_dst_tot__ += tau_dst__[cam_red_k__ - 1];

        }

        tau_w_dst_tot__ = 0.;
        for (cam_red_k__ = 1; cam_red_k__ <= 4; ++cam_red_k__) {
            tau_w_dst_tot__ += tau_w_dst__[cam_red_k__ - 1];

        }

        tau_w_g_dst_tot__ = 0.;
        for (cam_red_k__ = 1; cam_red_k__ <= 4; ++cam_red_k__) {
            tau_w_g_dst_tot__ += tau_w_g_dst__[cam_red_k__ - 1];

        }

        tau_w_f_dst_tot__ = 0.;
        for (cam_red_k__ = 1; cam_red_k__ <= 4; ++cam_red_k__) {
            tau_w_f_dst_tot__ += tau_w_f_dst__[cam_red_k__ - 1];

        }

        if (tau_dst_tot__ > 0.f) {
            w_dst_tot__ = tau_w_dst_tot__ / tau_dst_tot__;
        } else {
            w_dst_tot__ = 0.f;
        }
        if (tau_w_dst_tot__ > 0.f) {
            g_dst_tot__ = tau_w_g_dst_tot__ / tau_w_dst_tot__;
            f_dst_tot__ = tau_w_f_dst_tot__ / tau_w_dst_tot__;
        } else {
            g_dst_tot__ = 0.f;
            f_dst_tot__ = 0.f;
        }
        tau_tot__ = tau_sul__ + tau_sslt__ + tau_cphil__ +
            tau_cphob__ + tau_cb__ + tau_dst_tot__;
        tau_tot__ = tau_tot__ + tau_bg__ + tau_volc__;
        tau_w_tot__ = tau_w_sul__ + tau_w_sslt__ + tau_w_cphil__ +
            tau_w_cphob__ + tau_w_cb__ + tau_w_dst_tot__;
        tau_w_tot__ = tau_w_tot__ + tau_w_bg__ + tau_w_volc__;
        tau_w_g_tot__ = tau_w_g_sul__ + tau_w_g_sslt__ +
            tau_w_g_cphil__ + tau_w_g_cphob__ + tau_w_g_cb__ +
            tau_w_g_dst_tot__;
        tau_w_g_tot__ = tau_w_g_tot__ + tau_w_g_bg__ + tau_w_g_volc__;
        tau_w_f_tot__ = tau_w_f_sul__ + tau_w_f_sslt__ +
            tau_w_f_cphil__ + tau_w_f_cphob__ + tau_w_f_cb__ +
            tau_w_f_dst_tot__;
        tau_w_f_tot__ = tau_w_f_tot__ + tau_w_f_bg__ + tau_w_f_volc__;
        if (tau_tot__ > 0.f) {
            w_tot__ = tau_w_tot__ / tau_tot__;
        } else {
            w_tot__ = 0.f;
        }
        if (tau_w_tot__ > 0.f) {
            g_tot__ = tau_w_g_tot__ / tau_w_tot__;
            f_tot__ = tau_w_f_tot__ / tau_w_tot__;
        } else {
            g_tot__ = 0.f;
            f_tot__ = 0.f;
        }
        tauxar[i__ + k - 1] = tau_tot__;
        wa[i__ + k - 1] = min(w_tot__,.999999);
        if (g_tot__ > 1.) {
            cam_optics_warning__();
        }
        if (g_tot__ < -1.) {
            cam_optics_warning__();
        }
        ga[i__ + k - 1] = g_tot__;
        if (f_tot__ > 1.) {
            cam_optics_warning__();
        }
        if (f_tot__ < 0.) {
            cam_optics_warning__();
        }
        fa[i__ + k - 1] = f_tot__;
        aertau[i__ + (ns + 19) * aertau_dim1] += tau_sul__;
        aertau[i__ + (ns + 38) * aertau_dim1] += tau_sslt__;
        aertau[i__ + (ns + 57) * aertau_dim1] = aertau[i__ + (ns + 57)
             * aertau_dim1] + tau_cphil__ + tau_cphob__ +
            tau_cb__;
        aertau[i__ + (ns + 76) * aertau_dim1] += tau_dst_tot__;
        aertau[i__ + (ns + 95) * aertau_dim1] += tau_bg__;
        aertau[i__ + (ns + 114) * aertau_dim1] += tau_volc__;
        aertau[i__ + (ns + 133) * aertau_dim1] += tau_tot__;
        aerssa[i__ + (ns + 19) * aerssa_dim1] += tau_w_sul__;
        aerssa[i__ + (ns + 38) * aerssa_dim1] += tau_w_sslt__;
        aerssa[i__ + (ns + 57) * aerssa_dim1] = aerssa[i__ + (ns + 57)
             * aerssa_dim1] + tau_w_cphil__ + tau_w_cphob__ +
            tau_w_cb__;
        aerssa[i__ + (ns + 76) * aerssa_dim1] += tau_w_dst_tot__;
        aerssa[i__ + (ns + 95) * aerssa_dim1] += tau_w_bg__;
        aerssa[i__ + (ns + 114) * aerssa_dim1] += tau_w_volc__;
        aerssa[i__ + (ns + 133) * aerssa_dim1] += tau_w_tot__;
        aerasm[i__ + (ns + 19) * aerasm_dim1] += tau_w_g_sul__;
        aerasm[i__ + (ns + 38) * aerasm_dim1] += tau_w_g_sslt__;
        aerasm[i__ + (ns + 57) * aerasm_dim1] = aerasm[i__ + (ns + 57)
             * aerasm_dim1] + tau_w_g_cphil__ + tau_w_g_cphob__ +
            tau_w_g_cb__;
        aerasm[i__ + (ns + 76) * aerasm_dim1] += tau_w_g_dst_tot__;
        aerasm[i__ + (ns + 95) * aerasm_dim1] += tau_w_g_bg__;
        aerasm[i__ + (ns + 114) * aerasm_dim1] += tau_w_g_volc__;
        aerasm[i__ + (ns + 133) * aerasm_dim1] += tau_w_g_tot__;
        aerfwd[i__ + (ns + 19) * aerfwd_dim1] += tau_w_f_sul__;
        aerfwd[i__ + (ns + 38) * aerfwd_dim1] += tau_w_f_sslt__;
        aerfwd[i__ + (ns + 57) * aerfwd_dim1] = aerfwd[i__ + (ns + 57)
             * aerfwd_dim1] + tau_w_f_cphil__ + tau_w_f_cphob__ +
            tau_w_f_cb__;
        aerfwd[i__ + (ns + 76) * aerfwd_dim1] += tau_w_f_dst_tot__;
        aerfwd[i__ + (ns + 95) * aerfwd_dim1] += tau_w_f_bg__;
        aerfwd[i__ + (ns + 114) * aerfwd_dim1] += tau_w_f_volc__;
        aerfwd[i__ + (ns + 133) * aerfwd_dim1] += tau_w_f_tot__;

        }

        for (kaer = 1; kaer <= 7; ++kaer) {
        if (aerssa[i__ + (ns + kaer * 19) * aerssa_dim1] > 0.f) {
            aerasm[i__ + (ns + kaer * 19) * aerasm_dim1] /= aerssa[
                i__ + (ns + kaer * 19) * aerssa_dim1];
            aerfwd[i__ + (ns + kaer * 19) * aerfwd_dim1] /= aerssa[
                i__ + (ns + kaer * 19) * aerssa_dim1];
        } else {
            aerasm[i__ + (ns + kaer * 19) * aerasm_dim1] = 0.;
            aerfwd[i__ + (ns + kaer * 19) * aerfwd_dim1] = 0.;
        }
        if (aertau[i__ + (ns + kaer * 19) * aertau_dim1] > 0.f) {
            aerssa[i__ + (ns + kaer * 19) * aerssa_dim1] /= aertau[
                i__ + (ns + kaer * 19) * aertau_dim1];
        } else {
            aerssa[i__ + (ns + kaer * 19) * aerssa_dim1] = 0.;
        }

        }


    }

    wavmid[ns - 1] = (state->cg_wavmin__[ns - 1] +
        state->cg_wavmax__[ns - 1]) * .5;
    if (wavmid[ns - 1] < .7) {
        i__1 = ndayc;
        for (n = 1; n <= i__1; ++n) {
        i__ = idayc[n - 1];
        albdir[i__ + ns - 2] = asdir[i__];
        albdif[i__ + ns - 2] = asdif[i__];

        }

    } else {
        i__1 = ndayc;
        for (n = 1; n <= i__1; ++n) {
        i__ = idayc[n - 1];
        albdir[i__ + ns - 2] = aldir[i__];
        albdif[i__ + ns - 2] = aldif[i__];

        }

    }
    trayoslp = state->cg_raytau__[ns - 1] / state->cg_sslp__;
    raddedmx_(state, pver, pverp, pcols, &coszrs[1], &ndayc, idayc, &
        state->cg_abh2o__[ns - 1], &state->cg_abo3__[ns -
        1], &state->cg_abco2__[ns - 1], &state->cg_abo2__[
        ns - 1], uh2o, uo3, uco2, uo2, &trayoslp, pflx, &ns, &tauxcl[
        tauxcl_offset], wcl, gcl, fcl, &tauxci[tauxci_offset], wci,
        gci, fci, tauxar, wa, ga, fa, rdir, rdif, tdir, tdif, explay,
        rdirc, rdifc, tdirc, tdifc, explayc);


        wgtint = state->cg_nirwgt__[ns - 1];
        i__2 = nconfig - 1;
        for (cam_s17_1__ = 0; cam_s17_1__ <= i__2; ++cam_s17_1__) {
        rdndif[state->sw_flux_stride*((cam_s17_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = 0.;

        }

        i__2 = nconfig - 1;
        for (cam_s18_1__ = 0; cam_s18_1__ <= i__2; ++cam_s18_1__) {
        exptdn[state->sw_flux_stride*((cam_s18_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = 1.;

        }

        i__2 = nconfig - 1;
        for (cam_s19_1__ = 0; cam_s19_1__ <= i__2; ++cam_s19_1__) {
        tdntot[state->sw_flux_stride*((cam_s19_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = 1.;

        }

        i__2 = (CAM_LEVELS+1);
        for (k = 1; k <= i__2; ++k) {
        km1 = k - 1;
        i__3 = nuniqd[km1];
        for (l0 = 1; l0 <= i__3; ++l0) {
            is0 = istrtd[km1 + l0 * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            is1 = istrtd[km1 + (l0 + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1;
            j = icond[km1 + is0 * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            xexpt = exptdn[state->sw_flux_stride*(km1 + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            xrdnd = rdndif[state->sw_flux_stride*(km1 + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            tdnmexp = tdntot[state->sw_flux_stride*(km1 + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] - xexpt;
            if (ccon[km1 + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] == 1) {
            ytdnd = tdif[1 + (i__ + km1) * 1 - 2];
            yrdnd = rdif[1 + (i__ + km1) * 1 - 2];
            rdenom = 1. / (1. - yrdnd * xrdnd);
            rdirexp = rdir[1 + (i__ + km1) * 1 - 2] * xexpt;
            zexpt = xexpt * explay[1 + (i__ + km1) * 1 - 2];

            d__1 = ytdnd;
            zrdnd = yrdnd + xrdnd * (d__1 * d__1) * rdenom;
            ztdnt = xexpt * tdir[1 + (i__ + km1) * 1 - 2] +
                ytdnd * (tdnmexp + xrdnd * rdirexp) * rdenom;
            } else {
            ytdnd = tdifc[1 + (i__ + km1) * 1 - 2];
            yrdnd = rdifc[1 + (i__ + km1) * 1 - 2];
            rdenom = 1. / (1. - yrdnd * xrdnd);
            rdirexp = rdirc[1 + (i__ + km1) * 1 - 2] * xexpt;
            zexpt = xexpt * explayc[1 + (i__ + km1) * 1 - 2];

            d__1 = ytdnd;
            zrdnd = yrdnd + xrdnd * (d__1 * d__1) * rdenom;
            ztdnt = xexpt * tdirc[1 + (i__ + km1) * 1 - 2] +
                ytdnd * (tdnmexp + xrdnd * rdirexp) * rdenom;
            }
            i__4 = is1;
            for (isn = is0; isn <= i__4; ++isn) {
            j = icond[km1 + isn * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            exptdn[state->sw_flux_stride*(k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = zexpt;
            rdndif[state->sw_flux_stride*(k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = zrdnd;
            tdntot[state->sw_flux_stride*(k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = ztdnt;

            }


        }


        }

        i__2 = nconfig - 1;
        for (cam_s20_1__ = 0; cam_s20_1__ <= i__2; ++cam_s20_1__) {
        rupdir[state->sw_flux_stride*((CAM_LEVELS+1) + (cam_s20_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = albdir[i__ +
            ns - 2];

        }

        i__2 = nconfig - 1;
        for (cam_s21_1__ = 0; cam_s21_1__ <= i__2; ++cam_s21_1__) {
        rupdif[state->sw_flux_stride*((CAM_LEVELS+1) + (cam_s21_1__ + 1) * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = albdif[i__ +
            ns - 2];

        }

        for (k = CAM_LEVELS; k >= 0; --k) {
        i__2 = nuniqu[k];
        for (l0 = 1; l0 <= i__2; ++l0) {
            is0 = istrtu[k + l0 * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            is1 = istrtu[k + (l0 + 1) * ((1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] - 1;
            j = iconu[k + is0 * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            xrupd = rupdif[state->sw_flux_stride*(k + 1 + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            xrups = rupdir[state->sw_flux_stride*(k + 1 + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            if (ccon[k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))] == 1) {
            yexpt = explay[1 + (i__ + k) * 1 - 2];
            yrupd = rdif[1 + (i__ + k) * 1 - 2];
            ytupd = tdif[1 + (i__ + k) * 1 - 2];
            rdenom = 1. / (1. - yrupd * xrupd);
            tdnmexp = tdir[1 + (i__ + k) * 1 - 2] - yexpt;
            rdirexp = xrups * yexpt;

            d__1 = ytupd;
            zrupd = yrupd + xrupd * (d__1 * d__1) * rdenom;
            zrups = rdir[1 + (i__ + k) * 1 - 2] + ytupd * (
                rdirexp + xrupd * tdnmexp) * rdenom;
            } else {
            yexpt = explayc[1 + (i__ + k) * 1 - 2];
            yrupd = rdifc[1 + (i__ + k) * 1 - 2];
            ytupd = tdifc[1 + (i__ + k) * 1 - 2];
            rdenom = 1. / (1. - yrupd * xrupd);
            tdnmexp = tdirc[1 + (i__ + k) * 1 - 2] - yexpt;
            rdirexp = xrups * yexpt;

            d__1 = ytupd;
            zrupd = yrupd + xrupd * (d__1 * d__1) * rdenom;
            zrups = rdirc[1 + (i__ + k) * 1 - 2] + ytupd * (
                rdirexp + xrupd * tdnmexp) * rdenom;
            }
            i__3 = is1;
            for (isn = is0; isn <= i__3; ++isn) {
            j = iconu[k + isn * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1))))];
            rupdif[state->sw_flux_stride*(k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = zrupd;
            rupdir[state->sw_flux_stride*(k + j * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))] = zrups;

            }


        }


        }

        i__2 = (CAM_LEVELS+1);
        for (k = 0; k <= i__2; ++k) {
        fluxup[k] = 0.;
        fluxdn[k] = 0.;
        fluxdndir[k] = 0.;
        i__3 = nconfig;
        for (iconfig = 1; iconfig <= i__3; ++iconfig) {
            xwgt = wgtv[iconfig - 1];
            xexpt = exptdn[state->sw_flux_stride*(k + iconfig * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            xtdnt = tdntot[state->sw_flux_stride*(k + iconfig * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            xrdnd = rdndif[state->sw_flux_stride*(k + iconfig * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            xrupd = rupdif[state->sw_flux_stride*(k + iconfig * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            xrups = rupdir[state->sw_flux_stride*(k + iconfig * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) - ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];
            rdenom = 1. / (1. - xrdnd * xrupd);
            fluxup[k] += xwgt * ((xexpt * xrups + (xtdnt - xexpt) *
                xrupd) * rdenom);
            fluxdn[k] += xwgt * (xexpt + (xtdnt - xexpt + xexpt *
                xrups * xrdnd) * rdenom);
            fluxdndir[k] += xwgt * xexpt;

        }

        fluxup[k] /= totwgt;
        fluxdn[k] /= totwgt;
        fluxdndir[k] /= totwgt;

        }

        wexptdn = 0.;
        i__2 = nconfig;
        for (iconfig = 1; iconfig <= i__2; ++iconfig) {
        wexptdn += wgtv[iconfig - 1] * exptdn[state->sw_flux_stride*((CAM_LEVELS+1) + iconfig * ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))) -
            ((0)*(1)+(1)*((1)*(((CAM_LEVELS+1)-(0)+1)))))];

        }

        wexptdn /= totwgt;
        solflx = solin[i__] * state->cg_frcsol__[ns - 1] * psf[ns -
            1];
        fsnt[i__] += solflx * (fluxdn[1] - fluxup[1]);
        fsntoa[i__] += solflx * (fluxdn[0] - fluxup[0]);
        fsns[i__] += solflx * (fluxdn[(CAM_LEVELS+1)] - fluxup[(CAM_LEVELS+1)]);
        sfltot += solflx;
        fswup[0] += solflx * fluxup[0];
        fswdn[0] += solflx * fluxdn[0];
        fswdndir[0] += solflx * fluxdndir[0];
        if (wavmid[ns - 1] < .7) {
        sols[i__] += wexptdn * solflx * .001;
        solsd[i__] += (fluxdn[(CAM_LEVELS+1)] - wexptdn) * solflx * .001;
        } else {
        soll[i__] += wexptdn * solflx * .001;
        solld[i__] += (fluxdn[(CAM_LEVELS+1)] - wexptdn) * solflx * .001;
        fsnrtoaq[i__] += solflx * (fluxdn[0] - fluxup[0]);
        }
        fsnirtoa[i__] += wgtint * solflx * (fluxdn[0] - fluxup[0]);
        i__2 = CAM_LEVELS;
        for (k = 0; k <= i__2; ++k) {
        kp1 = k + 1;
        flxdiv = fluxdn[k] - fluxdn[kp1] + (fluxup[kp1] - fluxup[k]);
        totfld[k] += solflx * flxdiv;
        fswdn[kp1] += solflx * fluxdn[kp1];
        fswup[kp1] += solflx * fluxup[kp1];
        fswdndir[kp1] += solflx * fluxdndir[kp1];

        }

        exptdnc[0] = 1.;
        rdndifc[0] = 0.;
        tdntotc[0] = 1.;
        rupdirc[(CAM_LEVELS+1)] = albdir[i__ + ns - 2];
        rupdifc[(CAM_LEVELS+1)] = albdif[i__ + ns - 2];
        i__2 = (CAM_LEVELS+1);
        for (k = 1; k <= i__2; ++k) {
        km1 = k - 1;
        xexpt = exptdnc[km1];
        xrdnd = rdndifc[km1];
        yrdnd = rdifc[1 + (i__ + km1) * 1 - 2];
        ytdnd = tdifc[1 + (i__ + km1) * 1 - 2];
        exptdnc[k] = xexpt * explayc[1 + (i__ + km1) * 1 - 2];
        rdenom = 1. / (1. - yrdnd * xrdnd);
        rdirexp = rdirc[1 + (i__ + km1) * 1 - 2] * xexpt;
        tdnmexp = tdntotc[km1] - xexpt;
        tdntotc[k] = xexpt * tdirc[1 + (i__ + km1) * 1 - 2] +
            ytdnd * (tdnmexp + xrdnd * rdirexp) * rdenom;

        d__1 = ytdnd;
        rdndifc[k] = yrdnd + xrdnd * (d__1 * d__1) * rdenom;

        }

        for (k = CAM_LEVELS; k >= 0; --k) {
        xrupd = rupdifc[k + 1];
        yexpt = explayc[1 + (i__ + k) * 1 - 2];
        yrupd = rdifc[1 + (i__ + k) * 1 - 2];
        ytupd = tdifc[1 + (i__ + k) * 1 - 2];
        rdenom = 1. / (1. - yrupd * xrupd);
        rupdirc[k] = rdirc[1 + (i__ + k) * 1 - 2] + ytupd * (
            rupdirc[k + 1] * yexpt + xrupd * (tdirc[1 + (i__ + k)
             * 1 - 2] - yexpt)) * rdenom;

        d__1 = ytupd;
        rupdifc[k] = yrupd + xrupd * (d__1 * d__1) * rdenom;

        }

        for (k = 0; k <= 1; ++k) {
        rdenom = 1. / (1. - rdndifc[k] * rupdifc[k]);
        fluxup[k] = (exptdnc[k] * rupdirc[k] + (tdntotc[k] - exptdnc[
            k]) * rupdifc[k]) * rdenom;
        fluxdn[k] = exptdnc[k] + (tdntotc[k] - exptdnc[k] + exptdnc[k]
             * rupdirc[k] * rdndifc[k]) * rdenom;
        fswupc[k] += solflx * fluxup[k];
        fswdnc[k] += solflx * fluxdn[k];
        fswdncdir[k] += solflx * exptdnc[k];

        }

        i__2 = (CAM_LEVELS+1);
        for (k = 2; k <= i__2; ++k) {
        rdenom = 1. / (1. - rdndifc[k] * rupdifc[k]);
        fluxup[k] = (exptdnc[k] * rupdirc[k] + (tdntotc[k] - exptdnc[
            k]) * rupdifc[k]) * rdenom;
        fluxdn[k] = exptdnc[k] + (tdntotc[k] - exptdnc[k] + exptdnc[k]
             * rupdirc[k] * rdndifc[k]) * rdenom;
        fswupc[k] += solflx * fluxup[k];
        fswdnc[k] += solflx * fluxdn[k];
        fswdncdir[k] += solflx * exptdnc[k];

        }

        i__2 = CAM_LEVELS;
        for (k = 0; k <= i__2; ++k) {
        kp1 = k + 1;
        flxdiv = fluxdn[k] - fluxdn[kp1] + (fluxup[kp1] - fluxup[k]);
        totfldc[k] += solflx * flxdiv;

        }

        fsntc[i__] += solflx * (fluxdn[1] - fluxup[1]);
        fsntoac[i__] += solflx * (fluxdn[0] - fluxup[0]);
        fsnsc[i__] += solflx * (fluxdn[(CAM_LEVELS+1)] - fluxup[(CAM_LEVELS+1)]);
        fsdsc[i__] += solflx * fluxdn[(CAM_LEVELS+1)];
        fsnrtoac[i__] += wgtint * solflx * (fluxdn[0] - fluxup[0]);

    }

    i__2 = CAM_LEVELS;
    for (k = 1; k <= i__2; ++k) {
        qrs[i__ + k * qrs_dim1] = state->cg_gravit__ *
            -9.99999974737875164e-5f * totfld[k] / (pint[i__ + k *
            pint_dim1] - pint[i__ + (k + 1) * pint_dim1]);
        qrscs[i__ + k * qrscs_dim1] = state->cg_gravit__ *
            -9.99999974737875164e-5f * totfldc[k] / (pint[i__ + k *
            pint_dim1] - pint[i__ + (k + 1) * pint_dim1]);

    }

    i__2 = (CAM_LEVELS+1);
    for (k = 1; k <= i__2; ++k) {
        fsup[i__ + k * fsup_dim1] = fswup[k];
        fsupc[i__ + k * fsupc_dim1] = fswupc[k];
        fsdn[i__ + k * fsdn_dim1] = fswdn[k];
        fsdnc[i__ + k * fsdnc_dim1] = fswdnc[k];
        fsdndir[i__ + k * fsdndir_dim1] = fswdndir[k];
        fsdncdir[i__ + k * fsdncdir_dim1] = fswdncdir[k];
        fsdndif[i__ + k * fsdndif_dim1] = fswdn[k] - fswdndir[k];
        fsdncdif[i__ + k * fsdncdif_dim1] = fswdnc[k] - fswdncdir[k];

    }

    fsds[i__] = fswdn[(CAM_LEVELS+1)];
    fsdscdir[i__] = fsdncdir[i__ + (CAM_LEVELS+1) * fsdncdir_dim1];
    fsdscdif[i__] = fsdncdif[i__ + (CAM_LEVELS+1) * fsdncdif_dim1];
    fsdsdir[i__] = fsdndir[i__ + (CAM_LEVELS+1) * fsdndir_dim1];
    fsdsdif[i__] = fsdndif[i__ + (CAM_LEVELS+1) * fsdndif_dim1];

    }

    return 0;
}





__device__ int raddedmx_(CamState* state, integer *pver, integer *pverp, integer *pcols,
    doublereal *coszrs, integer *ndayc, integer *idayc, doublereal *abh2o,
     doublereal *abo3, doublereal *abco2, doublereal *abo2, doublereal *
    uh2o, doublereal *uo3, doublereal *uco2, doublereal *uo2, doublereal *
    trayoslp, doublereal *pflx, integer *ns, doublereal *tauxcl,
    doublereal *wcl, doublereal *gcl, doublereal *fcl, doublereal *tauxci,
     doublereal *wci, doublereal *gci, doublereal *fci, doublereal *
    tauxar, doublereal *wa, doublereal *ga, doublereal *fa, doublereal *
    rdir, doublereal *rdif, doublereal *tdir, doublereal *tdif,
    doublereal *explay, doublereal *rdirc, doublereal *rdifc, doublereal *
    tdirc, doublereal *tdifc, doublereal *explayc)
{
 CamScratchFrame scratch_frame(state);

    integer pflx_dim1 = 0,pflx_offset = 0,uh2o_dim1 = 0,uh2o_offset = 0,uo3_dim1 = 0,uo3_offset = 0,uco2_dim1 = 0,uco2_offset = 0,uo2_dim1 = 0,uo2_offset = 0,tauxcl_dim1 = 0,tauxcl_offset = 0,wcl_dim1 = 0,wcl_offset = 0,gcl_dim1 = 0,gcl_offset = 0,fcl_dim1 = 0,fcl_offset = 0,tauxci_dim1 = 0,tauxci_offset = 0,wci_dim1 = 0,wci_offset = 0,gci_dim1 = 0,gci_offset = 0,fci_dim1 = 0,fci_offset = 0,tauxar_dim1 = 0,tauxar_offset = 0,wa_dim1 = 0,wa_offset = 0,ga_dim1 = 0,ga_offset = 0,fa_dim1 = 0,fa_offset = 0,rdir_dim2 = 0,rdir_offset = 0,rdif_dim2 = 0,rdif_offset = 0,tdir_dim2 = 0,tdir_offset = 0,tdif_dim2 = 0,tdif_offset = 0,explay_dim2 = 0,explay_offset = 0,rdirc_dim2 = 0,rdirc_offset = 0,rdifc_dim2 = 0,rdifc_offset = 0,tdirc_dim2 = 0,tdirc_offset = 0,tdifc_dim2 = 0,tdifc_offset = 0,explayc_dim2 = 0,explayc_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,k = 0;

    doublereal ne = 0,lm = 0,gs = 0;

    integer nn = 0;

    doublereal ue = 0,ts = 0,ws = 0,wt = 0,gam = 0,amg = 0,apg = 0,arg = 0,alp = 0,gtot = 0,ftot = 0,wtau = 0,wtot = 0,taugab[1] = {},taucsc = 0,tauray[1] = {},extins = 0,tautot = 0;


    explayc_dim2 = 1;
    explayc_offset = 1 + 1 * (1 + explayc_dim2 * 0);
    explayc -= explayc_offset;
    tdifc_dim2 = 1;
    tdifc_offset = 1 + 1 * (1 + tdifc_dim2 * 0);
    tdifc -= tdifc_offset;
    tdirc_dim2 = 1;
    tdirc_offset = 1 + 1 * (1 + tdirc_dim2 * 0);
    tdirc -= tdirc_offset;
    rdifc_dim2 = 1;
    rdifc_offset = 1 + 1 * (1 + rdifc_dim2 * 0);
    rdifc -= rdifc_offset;
    rdirc_dim2 = 1;
    rdirc_offset = 1 + 1 * (1 + rdirc_dim2 * 0);
    rdirc -= rdirc_offset;
    explay_dim2 = 1;
    explay_offset = 1 + 1 * (1 + explay_dim2 * 0);
    explay -= explay_offset;
    tdif_dim2 = 1;
    tdif_offset = 1 + 1 * (1 + tdif_dim2 * 0);
    tdif -= tdif_offset;
    tdir_dim2 = 1;
    tdir_offset = 1 + 1 * (1 + tdir_dim2 * 0);
    tdir -= tdir_offset;
    rdif_dim2 = 1;
    rdif_offset = 1 + 1 * (1 + rdif_dim2 * 0);
    rdif -= rdif_offset;
    rdir_dim2 = 1;
    rdir_offset = 1 + 1 * (1 + rdir_dim2 * 0);
    rdir -= rdir_offset;
    fa_dim1 = 1;
    fa_offset = 1 + fa_dim1 * 0;
    fa -= fa_offset;
    ga_dim1 = 1;
    ga_offset = 1 + ga_dim1 * 0;
    ga -= ga_offset;
    wa_dim1 = 1;
    wa_offset = 1 + wa_dim1 * 0;
    wa -= wa_offset;
    tauxar_dim1 = 1;
    tauxar_offset = 1 + tauxar_dim1 * 0;
    tauxar -= tauxar_offset;
    fci_dim1 = 1;
    fci_offset = 1 + fci_dim1 * 0;
    fci -= fci_offset;
    gci_dim1 = 1;
    gci_offset = 1 + gci_dim1 * 0;
    gci -= gci_offset;
    wci_dim1 = 1;
    wci_offset = 1 + wci_dim1 * 0;
    wci -= wci_offset;
    tauxci_dim1 = 1;
    tauxci_offset = 1 + tauxci_dim1 * 0;
    tauxci -= tauxci_offset;
    fcl_dim1 = 1;
    fcl_offset = 1 + fcl_dim1 * 0;
    fcl -= fcl_offset;
    gcl_dim1 = 1;
    gcl_offset = 1 + gcl_dim1 * 0;
    gcl -= gcl_offset;
    wcl_dim1 = 1;
    wcl_offset = 1 + wcl_dim1 * 0;
    wcl -= wcl_offset;
    tauxcl_dim1 = 1;
    tauxcl_offset = 1 + tauxcl_dim1 * 0;
    tauxcl -= tauxcl_offset;
    pflx_dim1 = 1;
    pflx_offset = 1 + pflx_dim1 * 0;
    pflx -= pflx_offset;
    uo2_dim1 = 1;
    uo2_offset = 1 + uo2_dim1 * 0;
    uo2 -= uo2_offset;
    uco2_dim1 = 1;
    uco2_offset = 1 + uco2_dim1 * 0;
    uco2 -= uco2_offset;
    uo3_dim1 = 1;
    uo3_offset = 1 + uo3_dim1 * 0;
    uo3 -= uo3_offset;
    uh2o_dim1 = 1;
    uh2o_offset = 1 + uh2o_dim1 * 0;
    uh2o -= uh2o_offset;
    --idayc;
    --coszrs;


    i__1 = CAM_LEVELS;
    for (k = 0; k <= i__1; ++k) {
    i__2 = *ndayc;
    for (nn = 1; nn <= i__2; ++nn) {
        i__ = idayc[nn];
        tauray[i__ - 1] = *trayoslp * (pflx[i__ + (k + 1) * pflx_dim1] -
            pflx[i__ + k * pflx_dim1]);
        taugab[i__ - 1] = *abh2o * uh2o[i__ + k * uh2o_dim1] + *abo3 *
            uo3[i__ + k * uo3_dim1] + *abco2 * uco2[i__ + k *
            uco2_dim1] + *abo2 * uo2[i__ + k * uo2_dim1];
        tautot = tauxcl[i__ + k * tauxcl_dim1] + tauxci[i__ + k *
            tauxci_dim1] + tauray[i__ - 1] + taugab[i__ - 1] + tauxar[
            i__ + k * tauxar_dim1];
        taucsc = tauxcl[i__ + k * tauxcl_dim1] * wcl[i__ + k * wcl_dim1]
            + tauxci[i__ + k * tauxci_dim1] * wci[i__ + k * wci_dim1]
            + tauxar[i__ + k * tauxar_dim1] * wa[i__ + k * wa_dim1];
        wtau = tauray[i__ - 1] * .999998986721038818;
        wt = wtau + taucsc;
        wtot = wt / tautot;
        gtot = (wtau * 0. + gcl[i__ + k * gcl_dim1] * wcl[i__ + k *
            wcl_dim1] * tauxcl[i__ + k * tauxcl_dim1] + gci[i__ + k *
            gci_dim1] * wci[i__ + k * wci_dim1] * tauxci[i__ + k *
            tauxci_dim1] + ga[i__ + k * ga_dim1] * wa[i__ + k *
            wa_dim1] * tauxar[i__ + k * tauxar_dim1]) / wt;
        ftot = (wtau * .100000001490116119 + fcl[i__ + k * fcl_dim1] *
            wcl[i__ + k * wcl_dim1] * tauxcl[i__ + k * tauxcl_dim1] +
            fci[i__ + k * fci_dim1] * wci[i__ + k * wci_dim1] *
            tauxci[i__ + k * tauxci_dim1] + fa[i__ + k * fa_dim1] *
            wa[i__ + k * wa_dim1] * tauxar[i__ + k * tauxar_dim1]) /
            wt;
        ts = (1. - wtot * ftot) * tautot;
        ws = (1. - ftot) * wtot / (1. - wtot * ftot);
        gs = (gtot - ftot) / (1. - ftot);
        lm = sqrt(3. * (1. - ws) * (1. - ws * gs));
        alp = .75 * ws * coszrs[i__] * ((1. + gs * (1. - ws)) / (1. - lm *
             lm * coszrs[i__] * coszrs[i__]));
        gam = .5 * ws * ((3. * gs * (1. - ws) * coszrs[i__] * coszrs[i__]
            + 1.) / (1. - lm * lm * coszrs[i__] * coszrs[i__]));
        ue = 1.5 * (1. - ws * gs) / lm;

        d__1 = lm * ts;
        arg = min(d__1,25.);
        extins = cam_exp64(-arg);
        ne = (ue + 1.) * (ue + 1.) / extins - (ue - 1.) * (ue - 1.) *
            extins;
        rdif[1 + (i__ + k * rdif_dim2) * 1] = (ue + 1.) * (ue - 1.) * (
            1. / extins - extins) / ne;
        tdif[1 + (i__ + k * tdif_dim2) * 1] = ue * 4. / ne;

        d__1 = ts / coszrs[i__];
        arg = min(d__1,25.);
        explay[1 + (i__ + k * explay_dim2) * 1] = cam_exp64(-arg);
        apg = alp + gam;
        amg = alp - gam;
        rdir[1 + (i__ + k * rdir_dim2) * 1] = amg * (tdif[1 + (i__ +
            k * tdif_dim2) * 1] * explay[1 + (i__ + k *
            explay_dim2) * 1] - 1.) + apg * rdif[1 + (i__ + k *
            rdif_dim2) * 1];
        tdir[1 + (i__ + k * tdir_dim2) * 1] = apg * tdif[1 + (i__ +
            k * tdif_dim2) * 1] + (amg * rdif[1 + (i__ + k *
            rdif_dim2) * 1] - (apg - 1.)) * explay[1 + (i__ + k *
            explay_dim2) * 1];

        d__1 = rdir[1 + (i__ + k * rdir_dim2) * 1];
        rdir[1 + (i__ + k * rdir_dim2) * 1] = max(d__1,0.);

        d__1 = tdir[1 + (i__ + k * tdir_dim2) * 1];
        tdir[1 + (i__ + k * tdir_dim2) * 1] = max(d__1,0.);

        d__1 = rdif[1 + (i__ + k * rdif_dim2) * 1];
        rdif[1 + (i__ + k * rdif_dim2) * 1] = max(d__1,0.);

        d__1 = tdif[1 + (i__ + k * tdif_dim2) * 1];
        tdif[1 + (i__ + k * tdif_dim2) * 1] = max(d__1,0.);
        if (tauxcl[i__ + k * tauxcl_dim1] == 0. && tauxci[i__ + k *
            tauxci_dim1] == 0.) {
        rdirc[1 + (i__ + k * rdirc_dim2) * 1] = rdir[1 + (i__ +
            k * rdir_dim2) * 1];
        tdirc[1 + (i__ + k * tdirc_dim2) * 1] = tdir[1 + (i__ +
            k * tdir_dim2) * 1];
        rdifc[1 + (i__ + k * rdifc_dim2) * 1] = rdif[1 + (i__ +
            k * rdif_dim2) * 1];
        tdifc[1 + (i__ + k * tdifc_dim2) * 1] = tdif[1 + (i__ +
            k * tdif_dim2) * 1];
        explayc[1 + (i__ + k * explayc_dim2) * 1] = explay[1 + (
            i__ + k * explay_dim2) * 1];
        } else {
        tautot = tauray[i__ - 1] + taugab[i__ - 1] + tauxar[i__ + k *
            tauxar_dim1];
        taucsc = tauxar[i__ + k * tauxar_dim1] * wa[i__ + k * wa_dim1]
            ;
        wt = wtau + taucsc;
        wtot = wt / tautot;
        gtot = (wtau * 0. + ga[i__ + k * ga_dim1] * wa[i__ + k *
            wa_dim1] * tauxar[i__ + k * tauxar_dim1]) / wt;
        ftot = (wtau * .100000001490116119 + fa[i__ + k * fa_dim1] *
            wa[i__ + k * wa_dim1] * tauxar[i__ + k * tauxar_dim1])
             / wt;
        ts = (1. - wtot * ftot) * tautot;
        ws = (1. - ftot) * wtot / (1. - wtot * ftot);
        gs = (gtot - ftot) / (1. - ftot);
        lm = sqrt(3. * (1. - ws) * (1. - ws * gs));
        alp = .75 * ws * coszrs[i__] * ((1. + gs * (1. - ws)) / (1. -
            lm * lm * coszrs[i__] * coszrs[i__]));
        gam = .5 * ws * ((3. * gs * (1. - ws) * coszrs[i__] * coszrs[
            i__] + 1.) / (1. - lm * lm * coszrs[i__] * coszrs[i__]
            ));
        ue = 1.5 * (1. - ws * gs) / lm;

        d__1 = lm * ts;
        arg = min(d__1,25.);
        extins = cam_exp64(-arg);
        ne = (ue + 1.) * (ue + 1.) / extins - (ue - 1.) * (ue - 1.) *
            extins;
        rdifc[1 + (i__ + k * rdifc_dim2) * 1] = (ue + 1.) * (ue -
            1.) * (1. / extins - extins) / ne;
        tdifc[1 + (i__ + k * tdifc_dim2) * 1] = ue * 4. / ne;

        d__1 = ts / coszrs[i__];
        arg = min(d__1,25.);
        explayc[1 + (i__ + k * explayc_dim2) * 1] = cam_exp64(-arg);
        apg = alp + gam;
        amg = alp - gam;
        rdirc[1 + (i__ + k * rdirc_dim2) * 1] = amg * (tdifc[1 +
            (i__ + k * tdifc_dim2) * 1] * explayc[1 + (i__ + k
            * explayc_dim2) * 1] - 1.) + apg * rdifc[1 + (i__
            + k * rdifc_dim2) * 1];
        tdirc[1 + (i__ + k * tdirc_dim2) * 1] = apg * tdifc[1 + (
            i__ + k * tdifc_dim2) * 1] + (amg * rdifc[1 + (i__
            + k * rdifc_dim2) * 1] - (apg - 1.)) * explayc[1 +
            (i__ + k * explayc_dim2) * 1];

        d__1 = rdirc[1 + (i__ + k * rdirc_dim2) * 1];
        rdirc[1 + (i__ + k * rdirc_dim2) * 1] = max(d__1,0.);

        d__1 = tdirc[1 + (i__ + k * tdirc_dim2) * 1];
        tdirc[1 + (i__ + k * tdirc_dim2) * 1] = max(d__1,0.);

        d__1 = rdifc[1 + (i__ + k * rdifc_dim2) * 1];
        rdifc[1 + (i__ + k * rdifc_dim2) * 1] = max(d__1,0.);

        d__1 = tdifc[1 + (i__ + k * tdifc_dim2) * 1];
        tdifc[1 + (i__ + k * tdifc_dim2) * 1] = max(d__1,0.);
        }

    }


    }

    return 0;
}





__device__ integer findvalue_(CamState* state, integer *ix, integer *n, doublereal *ain, integer *indxa)
{
    integer ret_val = 0;

    integer i__ = 0,j = 0,ia = 0,il = 0,im = 0,ir = 0,itmp = 0;


    --indxa;
    --ain;


    il = 1;
    ir = *n;
    while(TRUE_) {
    if (ir - il <= 1) {
        if (ir - il == 1) {
        if (ain[indxa[ir]] < ain[indxa[il]]) {
            itmp = indxa[il];
            indxa[il] = indxa[ir];
            indxa[ir] = itmp;
        }
        }
        ret_val = indxa[*ix];
        return ret_val;
    } else {
        im = (il + ir) / 2;
        itmp = indxa[im];
        indxa[im] = indxa[il + 1];
        indxa[il + 1] = itmp;
        if (ain[indxa[il + 1]] > ain[indxa[ir]]) {
        itmp = indxa[il + 1];
        indxa[il + 1] = indxa[ir];
        indxa[ir] = itmp;
        }
        if (ain[indxa[il]] > ain[indxa[ir]]) {
        itmp = indxa[il];
        indxa[il] = indxa[ir];
        indxa[ir] = itmp;
        }
        if (ain[indxa[il + 1]] > ain[indxa[il]]) {
        itmp = indxa[il + 1];
        indxa[il + 1] = indxa[il];
        indxa[il] = itmp;
        }
        i__ = il + 1;
        j = ir;
        ia = indxa[il];
        while(TRUE_) {
        while(TRUE_) {
            ++i__;
            if (ain[indxa[i__]] >= ain[ia]) {
            goto L9007;
            }

        }
L9007:
        while(TRUE_) {
            --j;
            if (ain[indxa[j]] <= ain[ia]) {
            goto L9009;
            }

        }
L9009:
        if (j < i__) {
            goto L9005;
        }
        itmp = indxa[i__];
        indxa[i__] = indxa[j];
        indxa[j] = itmp;

        }
L9005:
        indxa[il] = indxa[j];
        indxa[j] = ia;
        if (j >= *ix) {
        ir = j - 1;
        }
        if (j <= *ix) {
        il = i__;
        }
    }

    }

    return ret_val;
}





__device__ int param_cldoptics_calc_(CamState* state, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, integer *pverr, integer *pverrp,
    integer *ppcnst, doublereal *q, doublereal *cldn, doublereal *
    landfrac, doublereal *landm, doublereal *icefrac, doublereal *pdel,
    doublereal *t, doublereal *ps, doublereal *pmid, doublereal *pint,
    doublereal *cicewp, doublereal *cliqwp, doublereal *emis, doublereal *
    rel, doublereal *rei, doublereal *pmxrgn, integer *nmxrgn, doublereal
    *snowh)
{
 CamScratchFrame scratch_frame(state);

    integer q_dim1 = 0,q_dim2 = 0,q_offset = 0,cldn_dim1 = 0,cldn_offset = 0,pdel_dim1 = 0,pdel_offset = 0,t_dim1 = 0,t_offset = 0,pmid_dim1 = 0,pmid_offset = 0,pint_dim1 = 0,pint_offset = 0,cicewp_dim1 = 0,cicewp_offset = 0,cliqwp_dim1 = 0,cliqwp_offset = 0,emis_dim1 = 0,emis_offset = 0,rel_dim1 = 0,rel_offset = 0,rei_dim1 = 0,rei_offset = 0,pmxrgn_dim1 = 0,pmxrgn_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0,d__2 = 0;

    integer i__ = 0,k = 0;

    doublereal cwp[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , gwp[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {} , tpw[1] = {}, tgwp[1] = {};

    integer lchnk = 0;

    doublereal rgrav, effcld[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  ;

    doublereal ficemr[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {} ;

    doublereal gicewp[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {} , gliqwp[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  ;

    integer cam_s1_0__ = 0,cam_s2_0__ = 0,cam_s3_0__ = 0,cam_s4_0__ = 0,cam_s4_1__ = 0,cam_s5_0__ = 0,cam_s5_1__ = 0,cam_s6_0__ = 0;

    doublereal tgicewp[1] = {},tgliqwp[1] = {};


    --snowh;
    --nmxrgn;
    --ps;
    --icefrac;
    --landm;
    --landfrac;
    pmxrgn_dim1 = 1;
    pmxrgn_offset = 1 + pmxrgn_dim1;
    pmxrgn -= pmxrgn_offset;
    rei_dim1 = 1;
    rei_offset = 1 + rei_dim1;
    rei -= rei_offset;
    rel_dim1 = 1;
    rel_offset = 1 + rel_dim1;
    rel -= rel_offset;
    emis_dim1 = 1;
    emis_offset = 1 + emis_dim1;
    emis -= emis_offset;
    cliqwp_dim1 = 1;
    cliqwp_offset = 1 + cliqwp_dim1;
    cliqwp -= cliqwp_offset;
    cicewp_dim1 = 1;
    cicewp_offset = 1 + cicewp_dim1;
    cicewp -= cicewp_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    t_dim1 = 1;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    pdel_dim1 = 1;
    pdel_offset = 1 + pdel_dim1;
    pdel -= pdel_offset;
    cldn_dim1 = 1;
    cldn_offset = 1 + cldn_dim1;
    cldn -= cldn_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;
    q_dim1 = 1;
    q_dim2 = CAM_LEVELS;
    q_offset = 1 + q_dim1 * (1 + q_dim2);
    q -= q_offset;


    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
    tgicewp[cam_s1_0__] = 0.f;

    }

    i__1 = 1 - 1;
    for (cam_s2_0__ = 0; cam_s2_0__ <= i__1; ++cam_s2_0__) {
    tgliqwp[cam_s2_0__] = 0.f;

    }

    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        gicewp[i__ + k - 2] = q[i__ + (k + state->cg_ixcldice__ *
            q_dim2) * q_dim1] * pdel[i__ + k * pdel_dim1] /
            state->cg_gravmks__ * 1e3f;
        gliqwp[i__ + k - 2] = q[i__ + (k + state->cg_ixcldliq__ *
            q_dim2) * q_dim1] * pdel[i__ + k * pdel_dim1] /
            state->cg_gravmks__ * 1e3f;

        d__1 = .01, d__2 = cldn[i__ + k * cldn_dim1];
        cicewp[i__ + k * cicewp_dim1] = gicewp[i__ + k - 2] / max(d__1,
            d__2);

        d__1 = .01, d__2 = cldn[i__ + k * cldn_dim1];
        cliqwp[i__ + k * cliqwp_dim1] = gliqwp[i__ + k - 2] / max(d__1,
            d__2);

        d__1 = 1e-10, d__2 = q[i__ + (k + state->cg_ixcldice__ *
            q_dim2) * q_dim1] + q[i__ + (k +
            state->cg_ixcldliq__ * q_dim2) * q_dim1];
        ficemr[i__ + k - 2] = q[i__ + (k + state->cg_ixcldice__ *
            q_dim2) * q_dim1] / max(d__1,d__2);
        tgicewp[i__ - 1] += gicewp[i__ + k - 2];
        tgliqwp[i__ - 1] += gliqwp[i__ + k - 2];

    }


    }

    i__1 = 1 - 1;
    for (cam_s3_0__ = 0; cam_s3_0__ <= i__1; ++cam_s3_0__) {
    tgwp[cam_s3_0__] = tgicewp[cam_s3_0__] + tgliqwp[cam_s3_0__];

    }

    i__1 = 1 - 1;
    for (cam_s4_0__ = 0; cam_s4_0__ <= i__1; ++cam_s4_0__) {
    i__2 = CAM_LEVELS - 1;
    for (cam_s4_1__ = 0; cam_s4_1__ <= i__2; ++cam_s4_1__) {
        gwp[cam_s4_0__ + 1 + (cam_s4_1__ + 1) - 2] = gicewp[cam_s4_0__ +
            1 + (cam_s4_1__ + 1) - 2] + gliqwp[cam_s4_0__ + 1 + (
            cam_s4_1__ + 1) - 2];

    }


    }

    i__1 = 1 - 1;
    for (cam_s5_0__ = 0; cam_s5_0__ <= i__1; ++cam_s5_0__) {
    i__2 = CAM_LEVELS - 1;
    for (cam_s5_1__ = 0; cam_s5_1__ <= i__2; ++cam_s5_1__) {
        cwp[cam_s5_0__ + 1 + (cam_s5_1__ + 1) - 2] = cicewp[cam_s5_0__ +
            1 + (cam_s5_1__ + 1) * cicewp_dim1] + cliqwp[cam_s5_0__ +
            1 + (cam_s5_1__ + 1) * cliqwp_dim1];

    }


    }

    i__1 = 1 - 1;
    for (cam_s6_0__ = 0; cam_s6_0__ <= i__1; ++cam_s6_0__) {
    tpw[cam_s6_0__] = 0.f;

    }

    rgrav = 1.f / state->cg_gravmks__;
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        tpw[i__ - 1] += pdel[i__ + k * pdel_dim1] * q[i__ + (k + q_dim2) *
             q_dim1] * rgrav;

    }


    }

    cldefr_(state, &lchnk, ncol, pcols, pver, pverp, &landfrac[1], &t[t_offset], &
        rel[rel_offset], &rei[rei_offset], &ps[1], &pmid[pmid_offset], &
        landm[1], &icefrac[1], &snowh[1]);
    cldems_(state, &lchnk, ncol, pcols, pver, pverp, cwp, ficemr, &rei[rei_offset], &
        emis[emis_offset]);
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        effcld[i__ + k - 2] = cldn[i__ + k * cldn_dim1] * emis[i__ + k *
            emis_dim1];

    }


    }

    cldovrlap_(state, &lchnk, ncol, pcols, pver, pverp, &pint[pint_offset], &cldn[
        cldn_offset], &nmxrgn[1], &pmxrgn[pmxrgn_offset]);
    return 0;
}






__device__ integer radctl_c__1 = 1;
__device__ integer radctl_c__11 = 11;


__device__ int radctl_(CamState* state, integer *j, integer *lchnk, integer *ncol,
    integer *pcols, integer *pver, integer *pverp, integer *pverr,
    integer *pverrp, integer *ppcnst, integer *pcnst, doublereal *lwups,
    doublereal *emis, doublereal *pmid, doublereal *pint, doublereal *
    pmln, doublereal *piln, doublereal *pdel, doublereal *t, doublereal *
    qm1, doublereal *cld, doublereal *cicewp, doublereal *cliqwp,
    doublereal *tauxcl, doublereal *tauxci, doublereal *coszrs,
    doublereal *clat, doublereal *asdir, doublereal *asdif, doublereal *
    aldir, doublereal *aldif, real *solcon, real *gmt, integer *julday,
    real *julian, real *dt, real *xtime, doublereal *pin, doublereal *
    ozmixmj, doublereal *ozmix, integer *levsiz, integer *num_months__,
    integer *ghg_input__, doublereal *m_psp__, doublereal *m_psn__,
    doublereal *aerosoljp, doublereal *aerosoljn, doublereal *m_hybi__,
    integer *paerlev, integer *naer_c__, doublereal *pmxrgn, integer *
    nmxrgn, logical *dolw, logical *dosw, logical *doabsems, doublereal *
    abstot, doublereal *absnxt, doublereal *emstot, doublereal *fsup,
    doublereal *fsupc, doublereal *fsdn, doublereal *fsdnc, doublereal *
    fsdndir, doublereal *fsdncdir, doublereal *fsdndif, doublereal *
    fsdncdif, doublereal *flup, doublereal *flupc, doublereal *fldn,
    doublereal *fldnc, doublereal *swcf, doublereal *lwcf, doublereal *
    flut, doublereal *fsns, doublereal *fsnt, doublereal *flns,
    doublereal *flnt, doublereal *qrs, doublereal *qrscs, doublereal *qrl,
     doublereal *qrlcs, doublereal *flwds, doublereal *rel, doublereal *
    rei, doublereal *sols, doublereal *soll, doublereal *solsd,
    doublereal *solld, doublereal *n2ovmr, doublereal *ch4vmr, doublereal
    *f11vmr, doublereal *f12vmr, doublereal *landfrac, doublereal *zm,
    doublereal *fsds, doublereal *fsdsdir, doublereal *fsdsdif)
{
 CamScratchFrame scratch_frame(state);

    integer emis_dim1 = 0,emis_offset = 0,pmid_dim1 = 0,pmid_offset = 0,pint_dim1 = 0,pint_offset = 0,pmln_dim1 = 0,pmln_offset = 0,rel_dim1 = 0,rel_offset = 0,rei_dim1 = 0,rei_offset = 0,piln_dim1 = 0,piln_offset = 0,pdel_dim1 = 0,pdel_offset = 0,t_dim1 = 0,t_offset = 0,qm1_dim1 = 0,qm1_dim2 = 0,qm1_offset = 0,cld_dim1 = 0,cld_offset = 0,cicewp_dim1 = 0,cicewp_offset = 0,cliqwp_dim1 = 0,cliqwp_offset = 0,tauxcl_dim1 = 0,tauxcl_offset = 0,tauxci_dim1 = 0,tauxci_offset = 0,zm_dim1 = 0,zm_offset = 0,ozmixmj_dim1 = 0,ozmixmj_dim2 = 0,ozmixmj_offset = 0,ozmix_dim1 = 0,ozmix_offset = 0,aerosoljp_dim1 = 0,aerosoljp_dim2 = 0,aerosoljp_offset = 0,aerosoljn_dim1 = 0,aerosoljn_dim2 = 0,aerosoljn_offset = 0,pmxrgn_dim1 = 0,pmxrgn_offset = 0,qrs_dim1 = 0,qrs_offset = 0,qrscs_dim1 = 0,qrscs_offset = 0,fsup_dim1 = 0,fsup_offset = 0,fsupc_dim1 = 0,fsupc_offset = 0,fsdn_dim1 = 0,fsdn_offset = 0,fsdnc_dim1 = 0,fsdnc_offset = 0,fsdndir_dim1 = 0,fsdndir_offset = 0,fsdncdir_dim1 = 0,fsdncdir_offset = 0,fsdndif_dim1 = 0,fsdndif_offset = 0,fsdncdif_dim1 = 0,fsdncdif_offset = 0,flup_dim1 = 0,flup_offset = 0,flupc_dim1 = 0,flupc_offset = 0,fldn_dim1 = 0,fldn_offset = 0,fldnc_dim1 = 0,fldnc_offset = 0,qrl_dim1 = 0,qrl_offset = 0,qrlcs_dim1 = 0,qrlcs_offset = 0,abstot_dim1 = 0,abstot_dim2 = 0,abstot_offset = 0,absnxt_dim1 = 0,absnxt_dim2 = 0,absnxt_offset = 0,emstot_dim1 = 0,emstot_offset = 0,i__1 = 0,i__2 = 0;

    doublereal *fsdscdif = cam_alloc<doublereal>(state, 1),*fsdscdir = cam_alloc<doublereal>(state, 1);

    integer *nmxrgnrf = cam_alloc<integer>(state, 1);

    doublereal *pmxrgnrf = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1))   , *fsnirtsq = cam_alloc<doublereal>(state, 1);

    integer i__ = 0,k = 0;

    doublereal *rh = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)) , *ch4 = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)) , *n2o = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)) ;

    integer if11 = 0,if12 = 0;

    doublereal *pbr = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))    , *pnm = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)) ;

    integer ich4 = 0;

    integer in2o = 0;

    doublereal *cfc11 = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))  , *cfc12 = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))   , eccf, *ftem = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))  , *esat = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))    , *qsat = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))    , *o3vmr = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))   , *o3mmr = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1))   ;

    doublereal *fsdsc = cam_alloc<doublereal>(state, 1),*flnsc = cam_alloc<doublereal>(state, 1),*flntc = cam_alloc<doublereal>(state, 1);

    doublereal *solin = cam_alloc<doublereal>(state, 1),*fsntc = cam_alloc<doublereal>(state, 1),*fsnsc = cam_alloc<doublereal>(state, 1),*flutc = cam_alloc<doublereal>(state, 1);

    doublereal *aerasm = cam_alloc<doublereal>(state, 133),*aerfwd = cam_alloc<doublereal>(state, 133),*scales = cam_alloc<doublereal>(state, 13);

    doublereal *aerssa = cam_alloc<doublereal>(state, 133),*aertau = cam_alloc<doublereal>(state, 133),*fsntoa = cam_alloc<doublereal>(state, 1),*fsnrtc = cam_alloc<doublereal>(state, 1),*fsnirt = cam_alloc<doublereal>(state, 1);

    integer cam_s1_0__ = 0,cam_s1_1__ = 0,cam_s2_0__ = 0,cam_s2_1__ = 0,cam_s3_0__ = 0,cam_s4_0__ = 0,cam_s4_1__ = 0,cam_s5_0__ = 0,cam_s5_1__ = 0;

    doublereal *frc_day__ = cam_alloc<doublereal>(state, 1), *fsntoac = cam_alloc<doublereal>(state, 1), *aerosol = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))   , *lwupcgs = cam_alloc<doublereal>(state, 1);


    --fsdsdif;
    --fsdsdir;
    --fsds;
    --landfrac;
    --solld;
    --solsd;
    --soll;
    --sols;
    --flwds;
    --flnt;
    --flns;
    --fsnt;
    --fsns;
    --flut;
    --lwcf;
    --swcf;
    --nmxrgn;
    --m_psn__;
    --m_psp__;
    --aldif;
    --aldir;
    --asdif;
    --asdir;
    --clat;
    --coszrs;
    --lwups;
    zm_dim1 = 1;
    zm_offset = 1 + zm_dim1;
    zm -= zm_offset;
    rei_dim1 = 1;
    rei_offset = 1 + rei_dim1;
    rei -= rei_offset;
    rel_dim1 = 1;
    rel_offset = 1 + rel_dim1;
    rel -= rel_offset;
    qrlcs_dim1 = 1;
    qrlcs_offset = 1 + qrlcs_dim1;
    qrlcs -= qrlcs_offset;
    qrl_dim1 = 1;
    qrl_offset = 1 + qrl_dim1;
    qrl -= qrl_offset;
    qrscs_dim1 = 1;
    qrscs_offset = 1 + qrscs_dim1;
    qrscs -= qrscs_offset;
    qrs_dim1 = 1;
    qrs_offset = 1 + qrs_dim1;
    qrs -= qrs_offset;
    absnxt_dim1 = 1;
    absnxt_dim2 = CAM_LEVELS;
    absnxt_offset = 1 + absnxt_dim1 * (1 + absnxt_dim2);
    absnxt -= absnxt_offset;
    tauxci_dim1 = 1;
    tauxci_offset = 1 + tauxci_dim1 * 0;
    tauxci -= tauxci_offset;
    tauxcl_dim1 = 1;
    tauxcl_offset = 1 + tauxcl_dim1 * 0;
    tauxcl -= tauxcl_offset;
    cliqwp_dim1 = 1;
    cliqwp_offset = 1 + cliqwp_dim1;
    cliqwp -= cliqwp_offset;
    cicewp_dim1 = 1;
    cicewp_offset = 1 + cicewp_dim1;
    cicewp -= cicewp_offset;
    cld_dim1 = 1;
    cld_offset = 1 + cld_dim1;
    cld -= cld_offset;
    t_dim1 = 1;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    pmln_dim1 = 1;
    pmln_offset = 1 + pmln_dim1;
    pmln -= pmln_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    emis_dim1 = 1;
    emis_offset = 1 + emis_dim1;
    emis -= emis_offset;
    fldnc_dim1 = 1;
    fldnc_offset = 1 + fldnc_dim1;
    fldnc -= fldnc_offset;
    fldn_dim1 = 1;
    fldn_offset = 1 + fldn_dim1;
    fldn -= fldn_offset;
    flupc_dim1 = 1;
    flupc_offset = 1 + flupc_dim1;
    flupc -= flupc_offset;
    flup_dim1 = 1;
    flup_offset = 1 + flup_dim1;
    flup -= flup_offset;
    fsdncdif_dim1 = 1;
    fsdncdif_offset = 1 + fsdncdif_dim1;
    fsdncdif -= fsdncdif_offset;
    fsdndif_dim1 = 1;
    fsdndif_offset = 1 + fsdndif_dim1;
    fsdndif -= fsdndif_offset;
    fsdncdir_dim1 = 1;
    fsdncdir_offset = 1 + fsdncdir_dim1;
    fsdncdir -= fsdncdir_offset;
    fsdndir_dim1 = 1;
    fsdndir_offset = 1 + fsdndir_dim1;
    fsdndir -= fsdndir_offset;
    fsdnc_dim1 = 1;
    fsdnc_offset = 1 + fsdnc_dim1;
    fsdnc -= fsdnc_offset;
    fsdn_dim1 = 1;
    fsdn_offset = 1 + fsdn_dim1;
    fsdn -= fsdn_offset;
    fsupc_dim1 = 1;
    fsupc_offset = 1 + fsupc_dim1;
    fsupc -= fsupc_offset;
    fsup_dim1 = 1;
    fsup_offset = 1 + fsup_dim1;
    fsup -= fsup_offset;
    emstot_dim1 = 1;
    emstot_offset = 1 + emstot_dim1;
    emstot -= emstot_offset;
    abstot_dim1 = 1;
    abstot_dim2 = (CAM_LEVELS+1);
    abstot_offset = 1 + abstot_dim1 * (1 + abstot_dim2);
    abstot -= abstot_offset;
    pmxrgn_dim1 = 1;
    pmxrgn_offset = 1 + pmxrgn_dim1;
    pmxrgn -= pmxrgn_offset;
    pdel_dim1 = 1;
    pdel_offset = 1 + pdel_dim1;
    pdel -= pdel_offset;
    piln_dim1 = 1;
    piln_offset = 1 + piln_dim1;
    piln -= piln_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;
    qm1_dim1 = 1;
    qm1_dim2 = CAM_LEVELS;
    qm1_offset = 1 + qm1_dim1 * (1 + qm1_dim2);
    qm1 -= qm1_offset;
    ozmix_dim1 = 1;
    ozmix_offset = 1 + ozmix_dim1;
    ozmix -= ozmix_offset;
    --pin;
    ozmixmj_dim1 = 1;
    ozmixmj_dim2 = *levsiz;
    ozmixmj_offset = 1 + ozmixmj_dim1 * (1 + ozmixmj_dim2);
    ozmixmj -= ozmixmj_offset;
    --m_hybi__;
    aerosoljn_dim1 = 1;
    aerosoljn_dim2 = *paerlev;
    aerosoljn_offset = 1 + aerosoljn_dim1 * (1 + aerosoljn_dim2);
    aerosoljn -= aerosoljn_offset;
    aerosoljp_dim1 = 1;
    aerosoljp_dim2 = *paerlev;
    aerosoljp_offset = 1 + aerosoljp_dim1 * (1 + aerosoljp_dim2);
    aerosoljp -= aerosoljp_offset;


    oznint_(state, julday, julian, dt, gmt, xtime, &ozmixmj[ozmixmj_offset], &ozmix[
        ozmix_offset], levsiz, num_months__, pcols);
    radozn_(state, lchnk, ncol, pcols, pver, &pmid[pmid_offset], &pin[1], levsiz, &
        ozmix[ozmix_offset], o3vmr);
    radinp_(state, lchnk, ncol, pcols, pver, pverp, &pmid[pmid_offset], &pint[
        pint_offset], o3vmr, pbr, pnm, &eccf, o3mmr);
    if (CAM_SHORTWAVE && state->abs_phase != 1) {
    aqsat_(state, &t[t_offset], &pmid[pmid_offset], esat, qsat, pcols, ncol,
        pver, &radctl_c__1, pver);
    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
        i__2 = CAM_LEVELS - 1;
        for (cam_s1_1__ = 0; cam_s1_1__ <= i__2; ++cam_s1_1__) {
        rh[cam_s1_0__ + 1 + (cam_s1_1__ + 1) - 2] = qm1[cam_s1_0__ +
            1 + (cam_s1_1__ + 1 + qm1_dim2) * qm1_dim1] / qsat[
            cam_s1_0__ + 1 + (cam_s1_1__ + 1) - 2] * ((1.f -
            state->cg_epsilo__) * qsat[cam_s1_0__ + 1 + (
            cam_s1_1__ + 1) - 2] + state->cg_epsilo__) / ((
            1.f - state->cg_epsilo__) * qm1[cam_s1_0__ + 1
            + (cam_s1_1__ + 1 + qm1_dim2) * qm1_dim1] +
            state->cg_epsilo__);

        }


    }

    if (state->cg_radforce__) {
        i__1 = 1 - 1;
        for (cam_s2_0__ = 0; cam_s2_0__ <= i__1; ++cam_s2_0__) {
        i__2 = (CAM_LEVELS+1) - 1;
        for (cam_s2_1__ = 0; cam_s2_1__ <= i__2; ++cam_s2_1__) {
            pmxrgnrf[cam_s2_0__ + 1 + (cam_s2_1__ + 1) - 2] = pmxrgn[
                cam_s2_0__ + 1 + (cam_s2_1__ + 1) * pmxrgn_dim1];

        }


        }

        i__1 = 1 - 1;
        for (cam_s3_0__ = 0; cam_s3_0__ <= i__1; ++cam_s3_0__) {
        nmxrgnrf[cam_s3_0__] = nmxrgn[cam_s3_0__ + 1];

        }

        get_rf_scales_(state, scales);
        get_aerosol_(state, lchnk, julday, julian, dt, gmt, xtime, &m_psp__[1],
            &m_psn__[1], &aerosoljp[aerosoljp_offset], &aerosoljn[
            aerosoljn_offset], &m_hybi__[1], paerlev, &radctl_c__11, &pint[
            pint_offset], pcols, pver, pverp, pverr, pverrp, aerosol,
            scales);
        aerosol_indirect_(state, ncol, lchnk, pcols, pver, ppcnst, &landfrac[1],
             &pmid[pmid_offset], &t[t_offset], &qm1[qm1_offset], &cld[
            cld_offset], &zm[zm_offset], &rel[rel_offset]);
        radcswmx_(state, j, lchnk, ncol, pcols, pver, pverp, pnm, pbr, &qm1[
            qm1_offset], rh, o3mmr, aerosol, &cld[cld_offset], &
            cicewp[cicewp_offset], &cliqwp[cliqwp_offset], &rel[
            rel_offset], &rei[rei_offset], &tauxcl[tauxcl_offset], &
            tauxci[tauxci_offset], &eccf, &coszrs[1], &
            state->cg_scon__, solin, solcon, &asdir[1], &asdif[
            1], &aldir[1], &aldif[1], nmxrgnrf, pmxrgnrf, &qrs[
            qrs_offset], &qrscs[qrscs_offset], &fsnt[1], fsntc,
            fsntoa, fsntoac, fsnirt, fsnrtc, fsnirtsq, &fsns[1],
            fsnsc, fsdsc, &fsds[1], &sols[1], &soll[1], &solsd[1], &
            solld[1], frc_day__, &fsup[fsup_offset], &fsupc[
            fsupc_offset], &fsdn[fsdn_offset], &fsdnc[fsdnc_offset], &
            fsdndir[fsdndir_offset], &fsdncdir[fsdncdir_offset], &
            fsdndif[fsdndif_offset], &fsdncdif[fsdncdif_offset], &
            fsdsdir[1], &fsdsdif[1], fsdscdir, fsdscdif, aertau,
            aerssa, aerasm, aerfwd);
        i__1 = 1;
        for (i__ = 1; i__ <= i__1; ++i__) {
        solin[i__ - 1] *= .00100000004749745131f;
        fsnt[i__] *= .00100000004749745131f;
        fsns[i__] *= .00100000004749745131f;
        fsntc[i__ - 1] *= .00100000004749745131f;
        fsnsc[i__ - 1] *= .00100000004749745131f;

        }

        i__1 = 1 - 1;
        for (cam_s4_0__ = 0; cam_s4_0__ <= i__1; ++cam_s4_0__) {
        i__2 = CAM_LEVELS - 1;
        for (cam_s4_1__ = 0; cam_s4_1__ <= i__2; ++cam_s4_1__) {
            ftem[cam_s4_0__ + 1 + (cam_s4_1__ + 1) - 2] = qrs[
                cam_s4_0__ + 1 + (cam_s4_1__ + 1) * qrs_dim1] /
                state->cg_cpair__;

        }


        }

    }
    get_int_scales_(state, scales);
    get_aerosol_(state, lchnk, julday, julian, dt, gmt, xtime, &m_psp__[1], &
        m_psn__[1], &aerosoljp[aerosoljp_offset], &aerosoljn[
        aerosoljn_offset], &m_hybi__[1], paerlev, &radctl_c__11, &pint[
        pint_offset], pcols, pver, pverp, pverr, pverrp, aerosol,
        scales);
    aerosol_indirect_(state, ncol, lchnk, pcols, pver, ppcnst, &landfrac[1], &
        pmid[pmid_offset], &t[t_offset], &qm1[qm1_offset], &cld[
        cld_offset], &zm[zm_offset], &rel[rel_offset]);
    radcswmx_(state, j, lchnk, ncol, pcols, pver, pverp, pnm, pbr, &qm1[
        qm1_offset], rh, o3mmr, aerosol, &cld[cld_offset], &cicewp[
        cicewp_offset], &cliqwp[cliqwp_offset], &rel[rel_offset], &
        rei[rei_offset], &tauxcl[tauxcl_offset], &tauxci[
        tauxci_offset], &eccf, &coszrs[1], &state->cg_scon__,
        solin, solcon, &asdir[1], &asdif[1], &aldir[1], &aldif[1], &
        nmxrgn[1], &pmxrgn[pmxrgn_offset], &qrs[qrs_offset], &qrscs[
        qrscs_offset], &fsnt[1], fsntc, fsntoa, fsntoac, fsnirt,
        fsnrtc, fsnirtsq, &fsns[1], fsnsc, fsdsc, &fsds[1], &sols[1],
        &soll[1], &solsd[1], &solld[1], frc_day__, &fsup[fsup_offset],
         &fsupc[fsupc_offset], &fsdn[fsdn_offset], &fsdnc[
        fsdnc_offset], &fsdndir[fsdndir_offset], &fsdncdir[
        fsdncdir_offset], &fsdndif[fsdndif_offset], &fsdncdif[
        fsdncdif_offset], &fsdsdir[1], &fsdsdif[1], fsdscdir,
        fsdscdif, aertau, aerssa, aerasm, aerfwd);
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        solin[i__ - 1] *= .00100000004749745131f;
        fsds[i__] *= .00100000004749745131f;
        fsnirt[i__ - 1] *= .00100000004749745131f;
        fsnrtc[i__ - 1] *= .00100000004749745131f;
        fsnirtsq[i__ - 1] *= .00100000004749745131f;
        fsnt[i__] *= .00100000004749745131f;
        fsns[i__] *= .00100000004749745131f;
        fsntc[i__ - 1] *= .00100000004749745131f;
        fsnsc[i__ - 1] *= .00100000004749745131f;
        fsdsc[i__ - 1] *= .00100000004749745131f;
        fsntoa[i__ - 1] *= .00100000004749745131f;
        fsntoac[i__ - 1] *= .00100000004749745131f;
        swcf[i__] = fsntoa[i__ - 1] - fsntoac[i__ - 1];
        fsdsdir[i__] *= .00100000004749745131f;
        fsdsdif[i__] *= .00100000004749745131f;
        fsdscdir[i__ - 1] *= .00100000004749745131f;
        fsdscdif[i__ - 1] *= .00100000004749745131f;

    }

    i__1 = 1 - 1;
    for (cam_s5_0__ = 0; cam_s5_0__ <= i__1; ++cam_s5_0__) {
        i__2 = CAM_LEVELS - 1;
        for (cam_s5_1__ = 0; cam_s5_1__ <= i__2; ++cam_s5_1__) {
        ftem[cam_s5_0__ + 1 + (cam_s5_1__ + 1) - 2] = qrs[cam_s5_0__
            + 1 + (cam_s5_1__ + 1) * qrs_dim1] /
            state->cg_cpair__;

        }


    }

    i__1 = (CAM_LEVELS+1);
    for (k = 1; k <= i__1; ++k) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        fsup[i__ + k * fsup_dim1] *= .00100000004749745131f;
        fsupc[i__ + k * fsupc_dim1] *= .00100000004749745131f;
        fsdn[i__ + k * fsdn_dim1] *= .00100000004749745131f;
        fsdnc[i__ + k * fsdnc_dim1] *= .00100000004749745131f;

        }


    }

    }
    if (CAM_LONGWAVE) {
    get_int_scales_(state, scales);
    get_aerosol_(state, lchnk, julday, julian, dt, gmt, xtime, &m_psp__[1], &
        m_psn__[1], &aerosoljp[aerosoljp_offset], &aerosoljn[
        aerosoljn_offset], &m_hybi__[1], paerlev, &radctl_c__11, &pint[
        pint_offset], pcols, pver, pverp, pverr, pverrp, aerosol,
        scales);
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        lwupcgs[i__ - 1] = lwups[i__];

    }

    if (state->cg_trace_gas__) {
        radclwmx_(state, lchnk, ncol, pcols, pver, pverp, lwupcgs, &t[t_offset],
            &qm1[(qm1_dim2 + 1) * qm1_dim1 + 1], o3vmr, pbr, pnm, &
            pmln[pmln_offset], &piln[piln_offset], &qm1[(in2o *
            qm1_dim2 + 1) * qm1_dim1 + 1], &qm1[(ich4 * qm1_dim2 + 1)
            * qm1_dim1 + 1], &qm1[(if11 * qm1_dim2 + 1) * qm1_dim1 +
            1], &qm1[(if12 * qm1_dim2 + 1) * qm1_dim1 + 1], &cld[
            cld_offset], &emis[emis_offset], &pmxrgn[pmxrgn_offset], &
            nmxrgn[1], &qrl[qrl_offset], &qrlcs[qrlcs_offset],
            doabsems, &abstot[abstot_offset], &absnxt[absnxt_offset],
            &emstot[emstot_offset], &flns[1], &flnt[1], flnsc, flntc,
            &flwds[1], &flut[1], flutc, &flup[flup_offset], &flupc[
            flupc_offset], &fldn[fldn_offset], &fldnc[fldnc_offset], &
            aerosol[(state->cg_idxvolc__ * ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))) - ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))]);
    if (state->abs_phase == 1) return 0;
    } else {
        if (*ghg_input__ == 1) {
        if (FALSE_) {
        }
        trcmix_clwrf_(state, lchnk, ncol, pcols, pver, &pmid[pmid_offset], &
            clat[1], n2ovmr, ch4vmr, f11vmr, f12vmr, n2o, ch4,
            cfc11, cfc12);
        } else {
        trcmix_(state, lchnk, ncol, pcols, pver, &pmid[pmid_offset], &clat[1]
            , n2o, ch4, cfc11, cfc12);
        }
        if (FALSE_) {
        }
        radclwmx_(state, lchnk, ncol, pcols, pver, pverp, lwupcgs, &t[t_offset],
            &qm1[(qm1_dim2 + 1) * qm1_dim1 + 1], o3vmr, pbr, pnm, &
            pmln[pmln_offset], &piln[piln_offset], n2o, ch4, cfc11,
            cfc12, &cld[cld_offset], &emis[emis_offset], &pmxrgn[
            pmxrgn_offset], &nmxrgn[1], &qrl[qrl_offset], &qrlcs[
            qrlcs_offset], doabsems, &abstot[abstot_offset], &absnxt[
            absnxt_offset], &emstot[emstot_offset], &flns[1], &flnt[1]
            , flnsc, flntc, &flwds[1], &flut[1], flutc, &flup[
            flup_offset], &flupc[flupc_offset], &fldn[fldn_offset], &
            fldnc[fldnc_offset], &aerosol[(state->cg_idxvolc__
            * ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))) - ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))]);
    if (state->abs_phase == 1) return 0;
    }
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
        flnt[i__] *= .00100000004749745131f;
        flut[i__] *= .00100000004749745131f;
        flutc[i__ - 1] *= .00100000004749745131f;
        flns[i__] *= .00100000004749745131f;
        flntc[i__ - 1] *= .00100000004749745131f;
        flnsc[i__ - 1] *= .00100000004749745131f;
        flwds[i__] *= .00100000004749745131f;
        lwcf[i__] = flutc[i__ - 1] - flut[i__];

    }

    i__1 = (CAM_LEVELS+1);
    for (k = 1; k <= i__1; ++k) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        flup[i__ + k * flup_dim1] *= .00100000004749745131f;
        flupc[i__ + k * flupc_dim1] *= .00100000004749745131f;
        fldn[i__ + k * fldn_dim1] *= .00100000004749745131f;
        fldnc[i__ + k * fldnc_dim1] *= .00100000004749745131f;

        }


    }

    }
    return 0;
}






__device__ logical c_false = FALSE_;
__device__ real c_b10 = 1440.f;


__device__ int camrad_(CamState* state, real *rthratenlw, real *rthratensw, real *
    rthratenlwc, real *rthratenswc, logical *dolw, logical *dosw, real *
    swupt, real *swuptc, real *swdnt, real *swdntc, real *lwupt, real *
    lwuptc, real *lwdnt, real *lwdntc, real *swupb, real *swupbc, real *
    swdnb, real *swdnbc, real *lwupb, real *lwupbc, real *lwdnb, real *
    lwdnbc, real *swcf, real *lwcf, real *olr, real *cemiss, real *
    taucldc, real *taucldi, real *coszr, real *gsw, real *glw, real *xlat,
     real *xlong, real *albedo, real *t_phy__, real *tsk, real *emiss,
    real *qv3d, real *qc3d, real *qr3d, real *qi3d, real *qs3d, real *
    qg3d, real *alswvisdir, real *alswvisdif, real *alswnirdir, real *
    alswnirdif, real *swvisdir, real *swvisdif, real *swnirdir, real *
    swnirdif, integer *sf_surface_physics__, real *swddir, real *swddif,
    real *swddni, logical *f_qv__, logical *f_qc__, logical *f_qr__,
    logical *f_qi__, logical *f_qs__, logical *f_qg__, real *f_ice_phy__,
    real *f_rain_phy__, real *p_phy__, real *p8w, real *z__, real *
    pi_phy__, real *rho_phy__, real *dz8w, real *cldfra, real *xland,
    real *xice, real *snow, real *ozmixm, real *pin0, integer *levsiz,
    integer *num_months__, real *m_psp__, real *m_psn__, real *aerosolcp,
    real *aerosolcn, real *m_hybi0__, integer *cam_abs_dim1__, integer *
    cam_abs_dim2__, integer *paerlev, integer *naer_c__, real *gmt,
    integer *julday, real *julian, integer *yr, real *dt, real *xtime,
    real *declin, real *solcon, real *radt, real *degrad, integer *
    n_cldadv__, real *abstot_3d__, real *absnxt_3d__, real *emstot_3d__,
    logical *doabsems, integer *ghg_input__, integer *ids, integer *ide,
    integer *jds, integer *jde, integer *kds, integer *kde, integer *ims,
    integer *ime, integer *jms, integer *jme, integer *kms, integer *kme,
    integer *its, integer *ite, integer *jts, integer *jte, integer *kts,
    integer *kte, real *coszen)
{
 CamScratchFrame scratch_frame(state);

    integer p_phy_dim1 = 0,p_phy_dim2 = 0,p_phy_offset = 0,p8w_dim1 = 0,p8w_dim2 = 0,p8w_offset = 0,z_dim1 = 0,z_dim2 = 0,z_offset = 0,pi_phy_dim1 = 0,pi_phy_dim2 = 0,pi_phy_offset = 0,rho_phy_dim1 = 0,rho_phy_dim2 = 0,rho_phy_offset = 0,dz8w_dim1 = 0,dz8w_dim2 = 0,dz8w_offset = 0,t_phy_dim1 = 0,t_phy_dim2 = 0,t_phy_offset = 0,qv3d_dim1 = 0,qv3d_dim2 = 0,qv3d_offset = 0,qc3d_dim1 = 0,qc3d_dim2 = 0,qc3d_offset = 0,qr3d_dim1 = 0,qr3d_dim2 = 0,qr3d_offset = 0,qi3d_dim1 = 0,qi3d_dim2 = 0,qi3d_offset = 0,qs3d_dim1 = 0,qs3d_dim2 = 0,qs3d_offset = 0,qg3d_dim1 = 0,qg3d_dim2 = 0,qg3d_offset = 0,cldfra_dim1 = 0,cldfra_dim2 = 0,cldfra_offset = 0,rthratenlw_dim1 = 0,rthratenlw_dim2 = 0,rthratenlw_offset = 0,rthratenlwc_dim1 = 0,rthratenlwc_dim2 = 0,rthratenlwc_offset = 0,rthratensw_dim1 = 0,rthratensw_dim2 = 0,rthratensw_offset = 0,rthratenswc_dim1 = 0,rthratenswc_dim2 = 0,rthratenswc_offset = 0,xlat_dim1 = 0,xlat_offset = 0,xlong_dim1 = 0,xlong_offset = 0,xland_dim1 = 0,xland_offset = 0,xice_dim1 = 0,xice_offset = 0,snow_dim1 = 0,snow_offset = 0,emiss_dim1 = 0,emiss_offset = 0,tsk_dim1 = 0,tsk_offset = 0,albedo_dim1 = 0,albedo_offset = 0,ozmixm_dim1 = 0,ozmixm_dim2 = 0,ozmixm_dim3 = 0,ozmixm_offset = 0,m_psp_dim1 = 0,m_psp_offset = 0,m_psn_dim1 = 0,m_psn_offset = 0,aerosolcp_dim1 = 0,aerosolcp_dim2 = 0,aerosolcp_dim3 = 0,aerosolcp_offset = 0,aerosolcn_dim1 = 0,aerosolcn_dim2 = 0,aerosolcn_dim3 = 0,aerosolcn_offset = 0,gsw_dim1 = 0,gsw_offset = 0,glw_dim1 = 0,glw_offset = 0,alswvisdir_dim1 = 0,alswvisdir_offset = 0,alswvisdif_dim1 = 0,alswvisdif_offset = 0,alswnirdir_dim1 = 0,alswnirdir_offset = 0,alswnirdif_dim1 = 0,alswnirdif_offset = 0,swvisdir_dim1 = 0,swvisdir_offset = 0,swvisdif_dim1 = 0,swvisdif_offset = 0,swnirdir_dim1 = 0,swnirdir_offset = 0,swnirdif_dim1 = 0,swnirdif_offset = 0,swddir_dim1 = 0,swddir_offset = 0,swddni_dim1 = 0,swddni_offset = 0,swddif_dim1 = 0,swddif_offset = 0,abstot_3d_dim1 = 0,abstot_3d_dim2 = 0,abstot_3d_dim3 = 0,abstot_3d_offset = 0,absnxt_3d_dim1 = 0,absnxt_3d_dim2 = 0,absnxt_3d_dim3 = 0,absnxt_3d_offset = 0,emstot_3d_dim1 = 0,emstot_3d_dim2 = 0,emstot_3d_offset = 0,swupt_dim1 = 0,swupt_offset = 0,swuptc_dim1 = 0,swuptc_offset = 0,swdnt_dim1 = 0,swdnt_offset = 0,swdntc_dim1 = 0,swdntc_offset = 0,lwupt_dim1 = 0,lwupt_offset = 0,lwuptc_dim1 = 0,lwuptc_offset = 0,lwdnt_dim1 = 0,lwdnt_offset = 0,lwdntc_dim1 = 0,lwdntc_offset = 0,swupb_dim1 = 0,swupb_offset = 0,swupbc_dim1 = 0,swupbc_offset = 0,swdnb_dim1 = 0,swdnb_offset = 0,swdnbc_dim1 = 0,swdnbc_offset = 0,lwupb_dim1 = 0,lwupb_offset = 0,lwupbc_dim1 = 0,lwupbc_offset = 0,lwdnb_dim1 = 0,lwdnb_offset = 0,lwdnbc_dim1 = 0,lwdnbc_offset = 0,swcf_dim1 = 0,swcf_offset = 0,lwcf_dim1 = 0,lwcf_offset = 0,olr_dim1 = 0,olr_offset = 0,coszr_dim1 = 0,coszr_offset = 0,cemiss_dim1 = 0,cemiss_dim2 = 0,cemiss_offset = 0,taucldc_dim1 = 0,taucldc_dim2 = 0,taucldc_offset = 0,taucldi_dim1 = 0,taucldi_dim2 = 0,taucldi_offset = 0,f_ice_phy_dim1 = 0,f_ice_phy_dim2 = 0,f_ice_phy_offset = 0,f_rain_phy_dim1 = 0,f_rain_phy_dim2 = 0,f_rain_phy_offset = 0,coszen_dim1 = 0,coszen_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0;

    real r__1 = 0,r__2 = 0;

    doublereal *landfrac = cam_alloc<doublereal>(state, 1), *fsdncdif = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))  ;

    integer begchunk = 0;

    doublereal *fsdncdir = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))   ;

    integer endchunk = 0;

    doublereal doydatam = 0,doydatap = 0,doymodel = 0;

    integer i__ = 0,j = 0,k = 0,m = 0,n = 0;

    doublereal *q = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))    , *t = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))   , *aerosoljp = cam_alloc<doublereal>(state, 377)    , *aerosoljn = cam_alloc<doublereal>(state, 377)    ;

    integer ii = 0,kk = 0;

    doublereal *ps = cam_alloc<doublereal>(state, 1), *zm = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))  ;

    integer kk1 = 0;

    doublereal *cld = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))    , *rel = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)) ;

    real xt24 = 0;

    doublereal *rei = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))    , *qrs = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)) , *qrl = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)) , *pin = cam_alloc<doublereal>(state, 59), *fldn = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))   , *pdel = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))    ;

    integer ncol = 0;

    doublereal *pmid = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))   , *pint = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))    ;

    integer pver = 0;

    doublereal *emis = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))   , *fsns = cam_alloc<doublereal>(state, 1), *fsnt = cam_alloc<doublereal>(state, 1), *flns = cam_alloc<doublereal>(state, 1),
        *flnt = cam_alloc<doublereal>(state, 1), *fsup = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1)) , *fsdn = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))    ;

    integer nyrm = 0;

    doublereal *flup = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))   ;

    integer nyrp = 0;

    doublereal *sols = cam_alloc<doublereal>(state, 1),*soll = cam_alloc<doublereal>(state, 1),*fsds = cam_alloc<doublereal>(state, 1),*clat = cam_alloc<doublereal>(state, 1),fact1 = 0,fact2 = 0;

    doublereal *aldif = cam_alloc<doublereal>(state, 1), *asdif = cam_alloc<doublereal>(state, 1), *fldnc = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))  ,
        *landm = cam_alloc<doublereal>(state, 1), *aldir = cam_alloc<doublereal>(state, 1), *fsdnc = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1)) ;

    integer lchnk = 0;

    real hrang = 0;

    doublereal *asdir = cam_alloc<doublereal>(state, 1), *flupc = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))    ;

    integer pcols = 0,pnats = 0,pcnst = 0;

    doublereal *fsupc = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))  ;

    integer pverp = 0,pverr = 0;

    real xxlat = 0;

    doublereal *snowh = cam_alloc<doublereal>(state, 1), *solsd = cam_alloc<doublereal>(state, 1), *solld = cam_alloc<doublereal>(state, 1), *lwups = cam_alloc<doublereal>(state, 1), *qrscs = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))  , *qrlcs = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))   , *flwds = cam_alloc<doublereal>(state, 1), *ozmix = cam_alloc<doublereal>(state, 59)
            , f11vmr, f12vmr;

    doublereal ch4vmr, co2vmr, n2ovmr, *m_hybi__ = cam_alloc<doublereal>(state, 29), *cicewp = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))   ;

    doublereal deltat, *lnpmid = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)) , *m_psjn__ = cam_alloc<doublereal>(state, 1)
        ;

    real tloctm = 0;

    integer ppcnst = 0;

    doublereal *lnpint = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1)) ;

    integer pverrp = 0;

    doublereal *cliqwp = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)) , *tauxcl = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))  , *coszrs = cam_alloc<doublereal>(state, 1), *tauxci = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))   ,
        *pmxrgn = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))    ;

    integer *nmxrgn = cam_alloc<integer>(state, 1);

    doublereal *olrtoa = cam_alloc<doublereal>(state, 1), *m_psjp__ = cam_alloc<doublereal>(state, 1), *abstot = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))  , *absnxt = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))    , *emstot = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS+1-(1)+1))
        , *icefrac = cam_alloc<doublereal>(state, 1), *fsdndif = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1)) ,
        *fsdsdif = cam_alloc<doublereal>(state, 1), *fsdndir = cam_alloc<doublereal>(state, (1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))   , *lwcftoa = cam_alloc<doublereal>(state, 1),
        *fsdsdir = cam_alloc<doublereal>(state, 1), *swcftoa = cam_alloc<doublereal>(state, 1), *ozmixmj = cam_alloc<doublereal>(state, 767)    ;


    --pin0;
    --m_hybi0__;
    coszen_dim1 = *ime - *ims + 1;
    coszen_offset = *ims + coszen_dim1 * *jms;
    coszen -= coszen_offset;
    aerosolcn_dim1 = *ime - *ims + 1;
    aerosolcn_dim2 = *paerlev;
    aerosolcn_dim3 = *jme - *jms + 1;
    aerosolcn_offset = *ims + aerosolcn_dim1 * (1 + aerosolcn_dim2 * (*jms +
        aerosolcn_dim3));
    aerosolcn -= aerosolcn_offset;
    aerosolcp_dim1 = *ime - *ims + 1;
    aerosolcp_dim2 = *paerlev;
    aerosolcp_dim3 = *jme - *jms + 1;
    aerosolcp_offset = *ims + aerosolcp_dim1 * (1 + aerosolcp_dim2 * (*jms +
        aerosolcp_dim3));
    aerosolcp -= aerosolcp_offset;
    m_psn_dim1 = *ime - *ims + 1;
    m_psn_offset = *ims + m_psn_dim1 * *jms;
    m_psn__ -= m_psn_offset;
    m_psp_dim1 = *ime - *ims + 1;
    m_psp_offset = *ims + m_psp_dim1 * *jms;
    m_psp__ -= m_psp_offset;
    ozmixm_dim1 = *ime - *ims + 1;
    ozmixm_dim2 = *levsiz;
    ozmixm_dim3 = *jme - *jms + 1;
    ozmixm_offset = *ims + ozmixm_dim1 * (1 + ozmixm_dim2 * (*jms +
        ozmixm_dim3));
    ozmixm -= ozmixm_offset;
    snow_dim1 = *ime - *ims + 1;
    snow_offset = *ims + snow_dim1 * *jms;
    snow -= snow_offset;
    xice_dim1 = *ime - *ims + 1;
    xice_offset = *ims + xice_dim1 * *jms;
    xice -= xice_offset;
    xland_dim1 = *ime - *ims + 1;
    xland_offset = *ims + xland_dim1 * *jms;
    xland -= xland_offset;
    swddni_dim1 = *ime - *ims + 1;
    swddni_offset = *ims + swddni_dim1 * *jms;
    swddni -= swddni_offset;
    swddif_dim1 = *ime - *ims + 1;
    swddif_offset = *ims + swddif_dim1 * *jms;
    swddif -= swddif_offset;
    swddir_dim1 = *ime - *ims + 1;
    swddir_offset = *ims + swddir_dim1 * *jms;
    swddir -= swddir_offset;
    swnirdif_dim1 = *ime - *ims + 1;
    swnirdif_offset = *ims + swnirdif_dim1 * *jms;
    swnirdif -= swnirdif_offset;
    swnirdir_dim1 = *ime - *ims + 1;
    swnirdir_offset = *ims + swnirdir_dim1 * *jms;
    swnirdir -= swnirdir_offset;
    swvisdif_dim1 = *ime - *ims + 1;
    swvisdif_offset = *ims + swvisdif_dim1 * *jms;
    swvisdif -= swvisdif_offset;
    swvisdir_dim1 = *ime - *ims + 1;
    swvisdir_offset = *ims + swvisdir_dim1 * *jms;
    swvisdir -= swvisdir_offset;
    alswnirdif_dim1 = *ime - *ims + 1;
    alswnirdif_offset = *ims + alswnirdif_dim1 * *jms;
    alswnirdif -= alswnirdif_offset;
    alswnirdir_dim1 = *ime - *ims + 1;
    alswnirdir_offset = *ims + alswnirdir_dim1 * *jms;
    alswnirdir -= alswnirdir_offset;
    alswvisdif_dim1 = *ime - *ims + 1;
    alswvisdif_offset = *ims + alswvisdif_dim1 * *jms;
    alswvisdif -= alswvisdif_offset;
    alswvisdir_dim1 = *ime - *ims + 1;
    alswvisdir_offset = *ims + alswvisdir_dim1 * *jms;
    alswvisdir -= alswvisdir_offset;
    emiss_dim1 = *ime - *ims + 1;
    emiss_offset = *ims + emiss_dim1 * *jms;
    emiss -= emiss_offset;
    tsk_dim1 = *ime - *ims + 1;
    tsk_offset = *ims + tsk_dim1 * *jms;
    tsk -= tsk_offset;
    albedo_dim1 = *ime - *ims + 1;
    albedo_offset = *ims + albedo_dim1 * *jms;
    albedo -= albedo_offset;
    xlong_dim1 = *ime - *ims + 1;
    xlong_offset = *ims + xlong_dim1 * *jms;
    xlong -= xlong_offset;
    xlat_dim1 = *ime - *ims + 1;
    xlat_offset = *ims + xlat_dim1 * *jms;
    xlat -= xlat_offset;
    glw_dim1 = *ime - *ims + 1;
    glw_offset = *ims + glw_dim1 * *jms;
    glw -= glw_offset;
    gsw_dim1 = *ime - *ims + 1;
    gsw_offset = *ims + gsw_dim1 * *jms;
    gsw -= gsw_offset;
    coszr_dim1 = *ime - *ims + 1;
    coszr_offset = *ims + coszr_dim1 * *jms;
    coszr -= coszr_offset;
    olr_dim1 = *ime - *ims + 1;
    olr_offset = *ims + olr_dim1 * *jms;
    olr -= olr_offset;
    lwcf_dim1 = *ime - *ims + 1;
    lwcf_offset = *ims + lwcf_dim1 * *jms;
    lwcf -= lwcf_offset;
    swcf_dim1 = *ime - *ims + 1;
    swcf_offset = *ims + swcf_dim1 * *jms;
    swcf -= swcf_offset;
    lwdnbc_dim1 = *ime - *ims + 1;
    lwdnbc_offset = *ims + lwdnbc_dim1 * *jms;
    lwdnbc -= lwdnbc_offset;
    lwdnb_dim1 = *ime - *ims + 1;
    lwdnb_offset = *ims + lwdnb_dim1 * *jms;
    lwdnb -= lwdnb_offset;
    lwupbc_dim1 = *ime - *ims + 1;
    lwupbc_offset = *ims + lwupbc_dim1 * *jms;
    lwupbc -= lwupbc_offset;
    lwupb_dim1 = *ime - *ims + 1;
    lwupb_offset = *ims + lwupb_dim1 * *jms;
    lwupb -= lwupb_offset;
    swdnbc_dim1 = *ime - *ims + 1;
    swdnbc_offset = *ims + swdnbc_dim1 * *jms;
    swdnbc -= swdnbc_offset;
    swdnb_dim1 = *ime - *ims + 1;
    swdnb_offset = *ims + swdnb_dim1 * *jms;
    swdnb -= swdnb_offset;
    swupbc_dim1 = *ime - *ims + 1;
    swupbc_offset = *ims + swupbc_dim1 * *jms;
    swupbc -= swupbc_offset;
    swupb_dim1 = *ime - *ims + 1;
    swupb_offset = *ims + swupb_dim1 * *jms;
    swupb -= swupb_offset;
    lwdntc_dim1 = *ime - *ims + 1;
    lwdntc_offset = *ims + lwdntc_dim1 * *jms;
    lwdntc -= lwdntc_offset;
    lwdnt_dim1 = *ime - *ims + 1;
    lwdnt_offset = *ims + lwdnt_dim1 * *jms;
    lwdnt -= lwdnt_offset;
    lwuptc_dim1 = *ime - *ims + 1;
    lwuptc_offset = *ims + lwuptc_dim1 * *jms;
    lwuptc -= lwuptc_offset;
    lwupt_dim1 = *ime - *ims + 1;
    lwupt_offset = *ims + lwupt_dim1 * *jms;
    lwupt -= lwupt_offset;
    swdntc_dim1 = *ime - *ims + 1;
    swdntc_offset = *ims + swdntc_dim1 * *jms;
    swdntc -= swdntc_offset;
    swdnt_dim1 = *ime - *ims + 1;
    swdnt_offset = *ims + swdnt_dim1 * *jms;
    swdnt -= swdnt_offset;
    swuptc_dim1 = *ime - *ims + 1;
    swuptc_offset = *ims + swuptc_dim1 * *jms;
    swuptc -= swuptc_offset;
    swupt_dim1 = *ime - *ims + 1;
    swupt_offset = *ims + swupt_dim1 * *jms;
    swupt -= swupt_offset;
    emstot_3d_dim1 = *ime - *ims + 1;
    emstot_3d_dim2 = *kme - *kms + 1;
    emstot_3d_offset = *ims + emstot_3d_dim1 * (*kms + emstot_3d_dim2 * *jms);
    emstot_3d__ -= emstot_3d_offset;
    absnxt_3d_dim1 = *ime - *ims + 1;
    absnxt_3d_dim2 = *kme - *kms + 1;
    absnxt_3d_dim3 = *cam_abs_dim1__;
    absnxt_3d_offset = *ims + absnxt_3d_dim1 * (*kms + absnxt_3d_dim2 * (1 +
        absnxt_3d_dim3 * *jms));
    absnxt_3d__ -= absnxt_3d_offset;
    abstot_3d_dim1 = *ime - *ims + 1;
    abstot_3d_dim2 = *kme - *kms + 1;
    abstot_3d_dim3 = *cam_abs_dim2__;
    abstot_3d_offset = *ims + abstot_3d_dim1 * (*kms + abstot_3d_dim2 * (1 +
        abstot_3d_dim3 * *jms));
    abstot_3d__ -= abstot_3d_offset;
    cldfra_dim1 = *ime - *ims + 1;
    cldfra_dim2 = *kme - *kms + 1;
    cldfra_offset = *ims + cldfra_dim1 * (*kms + cldfra_dim2 * *jms);
    cldfra -= cldfra_offset;
    dz8w_dim1 = *ime - *ims + 1;
    dz8w_dim2 = *kme - *kms + 1;
    dz8w_offset = *ims + dz8w_dim1 * (*kms + dz8w_dim2 * *jms);
    dz8w -= dz8w_offset;
    rho_phy_dim1 = *ime - *ims + 1;
    rho_phy_dim2 = *kme - *kms + 1;
    rho_phy_offset = *ims + rho_phy_dim1 * (*kms + rho_phy_dim2 * *jms);
    rho_phy__ -= rho_phy_offset;
    pi_phy_dim1 = *ime - *ims + 1;
    pi_phy_dim2 = *kme - *kms + 1;
    pi_phy_offset = *ims + pi_phy_dim1 * (*kms + pi_phy_dim2 * *jms);
    pi_phy__ -= pi_phy_offset;
    z_dim1 = *ime - *ims + 1;
    z_dim2 = *kme - *kms + 1;
    z_offset = *ims + z_dim1 * (*kms + z_dim2 * *jms);
    z__ -= z_offset;
    p8w_dim1 = *ime - *ims + 1;
    p8w_dim2 = *kme - *kms + 1;
    p8w_offset = *ims + p8w_dim1 * (*kms + p8w_dim2 * *jms);
    p8w -= p8w_offset;
    p_phy_dim1 = *ime - *ims + 1;
    p_phy_dim2 = *kme - *kms + 1;
    p_phy_offset = *ims + p_phy_dim1 * (*kms + p_phy_dim2 * *jms);
    p_phy__ -= p_phy_offset;
    f_rain_phy_dim1 = *ime - *ims + 1;
    f_rain_phy_dim2 = *kme - *kms + 1;
    f_rain_phy_offset = *ims + f_rain_phy_dim1 * (*kms + f_rain_phy_dim2 * *
        jms);
    f_rain_phy__ -= f_rain_phy_offset;
    f_ice_phy_dim1 = *ime - *ims + 1;
    f_ice_phy_dim2 = *kme - *kms + 1;
    f_ice_phy_offset = *ims + f_ice_phy_dim1 * (*kms + f_ice_phy_dim2 * *jms);
    f_ice_phy__ -= f_ice_phy_offset;
    qg3d_dim1 = *ime - *ims + 1;
    qg3d_dim2 = *kme - *kms + 1;
    qg3d_offset = *ims + qg3d_dim1 * (*kms + qg3d_dim2 * *jms);
    qg3d -= qg3d_offset;
    qs3d_dim1 = *ime - *ims + 1;
    qs3d_dim2 = *kme - *kms + 1;
    qs3d_offset = *ims + qs3d_dim1 * (*kms + qs3d_dim2 * *jms);
    qs3d -= qs3d_offset;
    qi3d_dim1 = *ime - *ims + 1;
    qi3d_dim2 = *kme - *kms + 1;
    qi3d_offset = *ims + qi3d_dim1 * (*kms + qi3d_dim2 * *jms);
    qi3d -= qi3d_offset;
    qr3d_dim1 = *ime - *ims + 1;
    qr3d_dim2 = *kme - *kms + 1;
    qr3d_offset = *ims + qr3d_dim1 * (*kms + qr3d_dim2 * *jms);
    qr3d -= qr3d_offset;
    qc3d_dim1 = *ime - *ims + 1;
    qc3d_dim2 = *kme - *kms + 1;
    qc3d_offset = *ims + qc3d_dim1 * (*kms + qc3d_dim2 * *jms);
    qc3d -= qc3d_offset;
    qv3d_dim1 = *ime - *ims + 1;
    qv3d_dim2 = *kme - *kms + 1;
    qv3d_offset = *ims + qv3d_dim1 * (*kms + qv3d_dim2 * *jms);
    qv3d -= qv3d_offset;
    t_phy_dim1 = *ime - *ims + 1;
    t_phy_dim2 = *kme - *kms + 1;
    t_phy_offset = *ims + t_phy_dim1 * (*kms + t_phy_dim2 * *jms);
    t_phy__ -= t_phy_offset;
    taucldi_dim1 = *ime - *ims + 1;
    taucldi_dim2 = *kme - *kms + 1;
    taucldi_offset = *ims + taucldi_dim1 * (*kms + taucldi_dim2 * *jms);
    taucldi -= taucldi_offset;
    taucldc_dim1 = *ime - *ims + 1;
    taucldc_dim2 = *kme - *kms + 1;
    taucldc_offset = *ims + taucldc_dim1 * (*kms + taucldc_dim2 * *jms);
    taucldc -= taucldc_offset;
    cemiss_dim1 = *ime - *ims + 1;
    cemiss_dim2 = *kme - *kms + 1;
    cemiss_offset = *ims + cemiss_dim1 * (*kms + cemiss_dim2 * *jms);
    cemiss -= cemiss_offset;
    rthratenswc_dim1 = *ime - *ims + 1;
    rthratenswc_dim2 = *kme - *kms + 1;
    rthratenswc_offset = *ims + rthratenswc_dim1 * (*kms + rthratenswc_dim2 *
        *jms);
    rthratenswc -= rthratenswc_offset;
    rthratenlwc_dim1 = *ime - *ims + 1;
    rthratenlwc_dim2 = *kme - *kms + 1;
    rthratenlwc_offset = *ims + rthratenlwc_dim1 * (*kms + rthratenlwc_dim2 *
        *jms);
    rthratenlwc -= rthratenlwc_offset;
    rthratensw_dim1 = *ime - *ims + 1;
    rthratensw_dim2 = *kme - *kms + 1;
    rthratensw_offset = *ims + rthratensw_dim1 * (*kms + rthratensw_dim2 * *
        jms);
    rthratensw -= rthratensw_offset;
    rthratenlw_dim1 = *ime - *ims + 1;
    rthratenlw_dim2 = *kme - *kms + 1;
    rthratenlw_offset = *ims + rthratenlw_dim1 * (*kms + rthratenlw_dim2 * *
        jms);
    rthratenlw -= rthratenlw_offset;


    lchnk = 1;
    begchunk = *ims;
    endchunk = *ime;
    ncol = *ite - *its + 1;
    pcols = *ite - *its + 1;
    pver = *kte - *kts + 1;
    pverp = pver + 1;
    pverr = *kte - *kts + 1;
    pverrp = pverr + 1;
    ppcnst = *n_cldadv__;
    pnats = 0;
    pcnst = ppcnst - pnats;
    if (*naer_c__ != 13) {
    cam_driver_failure__();
    }
    if (*ghg_input__ == 1) {
    read_camgases__(state, yr, julian, &c_false, "cam", &co2vmr, &n2ovmr, &
        ch4vmr, &f11vmr, &f12vmr, (ftnlen)3);
    if (FALSE_) {
    }
    } else {
    nyrm = *yr - state->cg_yrdata__[0] + 1;
    nyrp = nyrm + 1;
    doymodel = *yr * 365.f + *julian;
    doydatam = state->cg_yrdata__[nyrm - 1] * 365.f + 1.f;
    doydatap = state->cg_yrdata__[nyrp - 1] * 365.f + 1.f;
    deltat = doydatap - doydatam;
    fact1 = (doydatap - doymodel) / deltat;
    fact2 = (doymodel - doydatam) / deltat;
    co2vmr = (state->cg_co2__[nyrm - 1] * fact1 +
        state->cg_co2__[nyrp - 1] * fact2) *
        9.99999997475242708e-7f;
    }
    state->cg_co2mmr__ = co2vmr * state->cg_mwco2__ /
        state->cg_mwdry__;
    i__1 = *levsiz;
    for (k = 1; k <= i__1; ++k) {
    pin[k - 1] = pin0[k];

    }

    i__1 = *paerlev;
    for (k = 1; k <= i__1; ++k) {
    m_hybi__[k - 1] = m_hybi0__[k];

    }

    if (abstot_3d__[*its + (*kts + (*kts + *jts * abstot_3d_dim3) *
        abstot_3d_dim2) * abstot_3d_dim1] == 0.f && ! (*doabsems) && CAM_LONGWAVE) {
    *doabsems = TRUE_;
    }
    i__1 = *jte;
    for (j = *jts; j <= i__1; ++j) {
    if (TRUE_) {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;
        clat[ii - 1] = xlat[i__ + j * xlat_dim1] * *degrad;
        coszrs[ii - 1] = coszen[i__ + j * coszen_dim1];

        }

    } else {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;
        r__1 = *xtime + *radt * .5f;
        xt24 = r_mod(&r__1, &c_b10);
        tloctm = *gmt + xt24 / 60.f + xlong[i__ + j * xlong_dim1] /
            15.f;
        hrang = (tloctm - 12.f) * 15.f * *degrad;
        xxlat = xlat[i__ + j * xlat_dim1] * *degrad;
        clat[ii - 1] = xxlat;
        coszrs[ii - 1] = sin(xxlat) * sin(*declin) + cos(xxlat) * cos(
            *declin) * cos(hrang);

        }

    }
    i__2 = *kte;
    for (k = *kts; k <= i__2; ++k) {
        kk = *kte - k + *kts;
        i__3 = *ite;
        for (i__ = *its; i__ <= i__3; ++i__) {
        ii = i__ - *its + 1;

        r__1 = 1.00000001335143196e-10f, r__2 = qv3d[i__ + (k + j *
            qv3d_dim2) * qv3d_dim1] / (qv3d[i__ + (k + j *
            qv3d_dim2) * qv3d_dim1] + 1.f);
        q[ii + (kk + ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] = max(r__1,r__2);
        if (*f_qi__ && *f_qc__ && *f_qs__) {

            r__1 = 0.f, r__2 = qc3d[i__ + (k + j * qc3d_dim2) *
                qc3d_dim1] / (qv3d[i__ + (k + j * qv3d_dim2) *
                qv3d_dim1] + 1.f);
            q[ii + (kk + (state->cg_ixcldliq__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                max(r__1,r__2);

            r__1 = 0.f, r__2 = (qi3d[i__ + (k + j * qi3d_dim2) *
                qi3d_dim1] + qs3d[i__ + (k + j * qs3d_dim2) *
                qs3d_dim1]) / (qv3d[i__ + (k + j * qv3d_dim2) *
                qv3d_dim1] + 1.f);
            q[ii + (kk + (state->cg_ixcldice__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                max(r__1,r__2);
        } else if (*f_qc__ && *f_qi__) {

            r__1 = 0.f, r__2 = qi3d[i__ + (k + j * qi3d_dim2) *
                qi3d_dim1] / (qv3d[i__ + (k + j * qv3d_dim2) *
                qv3d_dim1] + 1.f);
            q[ii + (kk + (state->cg_ixcldice__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                max(r__1,r__2);

            r__1 = 0.f, r__2 = qc3d[i__ + (k + j * qc3d_dim2) *
                qc3d_dim1] / (qv3d[i__ + (k + j * qv3d_dim2) *
                qv3d_dim1] + 1.f);
            q[ii + (kk + (state->cg_ixcldliq__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                max(r__1,r__2);
        } else if (*f_qc__ && *f_qr__) {
            q[ii + (kk + (state->cg_ixcldliq__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                0.f;
            q[ii + (kk + (state->cg_ixcldice__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                0.f;
            if (t_phy__[i__ + (k + j * t_phy_dim2) * t_phy_dim1] >
                273.149993896484375f) {

            r__1 = 0.f, r__2 = qc3d[i__ + (k + j * qc3d_dim2) *
                qc3d_dim1] / (qv3d[i__ + (k + j * qv3d_dim2) *
                 qv3d_dim1] + 1.f);
            q[ii + (kk + (state->cg_ixcldliq__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))]
                = max(r__1,r__2);
            }
            if (t_phy__[i__ + (k + j * t_phy_dim2) * t_phy_dim1] <=
                273.149993896484375f) {

            r__1 = 0.f, r__2 = qc3d[i__ + (k + j * qc3d_dim2) *
                qc3d_dim1] / (qv3d[i__ + (k + j * qv3d_dim2) *
                 qv3d_dim1] + 1.f);
            q[ii + (kk + (state->cg_ixcldice__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))]
                = max(r__1,r__2);
            }
        } else {
            q[ii + (kk + (state->cg_ixcldliq__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                0.f;
            q[ii + (kk + (state->cg_ixcldice__ * ((1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-1+1-(1)+1)))+(1)*(((1)*((1-1+1-(1)+1)))*((CAM_LEVELS-1+1-(1)+1))))] =
                0.f;
        }
        cld[ii + kk - 2] = cldfra[i__ + (k + j * cldfra_dim2) *
            cldfra_dim1];

        }


    }

    i__2 = *ite;
    for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;
        landfrac[ii - 1] = 2.f - xland[i__ + j * xland_dim1];
        landm[ii - 1] = landfrac[ii - 1];
        snowh[ii - 1] = snow[i__ + j * snow_dim1] *
            .00100000004749745131f;
        icefrac[ii - 1] = xice[i__ + j * xice_dim1];

    }

    i__2 = *num_months__ - 1;
    for (m = 1; m <= i__2; ++m) {
        i__3 = *levsiz;
        for (k = 1; k <= i__3; ++k) {
        i__4 = *ite;
        for (i__ = *its; i__ <= i__4; ++i__) {
            ii = i__ - *its + 1;
            ozmixmj[ii + (k + m * 59) - 61] = ozmixm[i__ + (k + (j + (
                m + 1) * ozmixm_dim3) * ozmixm_dim2) *
                ozmixm_dim1];

        }


        }


    }

    i__2 = *ite;
    for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;
        m_psjp__[ii - 1] = m_psp__[i__ + j * m_psp_dim1];
        m_psjn__[ii - 1] = m_psn__[i__ + j * m_psn_dim1];

    }

    i__2 = *naer_c__;
    for (n = 1; n <= i__2; ++n) {
        i__3 = *paerlev;
        for (k = 1; k <= i__3; ++k) {
        i__4 = *ite;
        for (i__ = *its; i__ <= i__4; ++i__) {
            ii = i__ - *its + 1;
            aerosoljp[ii + (k + n * 29) - 31] = aerosolcp[i__ + (k + (
                j + n * aerosolcp_dim3) * aerosolcp_dim2) *
                aerosolcp_dim1];
            aerosoljn[ii + (k + n * 29) - 31] = aerosolcn[i__ + (k + (
                j + n * aerosolcn_dim3) * aerosolcn_dim2) *
                aerosolcn_dim1];

        }


        }


    }

    i__2 = *ite;
    for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;

        r__1 = tsk[i__ + j * tsk_dim1], r__1 *= r__1;
        lwups[ii - 1] = state->cg_stebol__ * emiss[i__ + j *
            emiss_dim1] * (r__1 * r__1);

    }

    i__2 = *kte + 1;
    for (k = *kts; k <= i__2; ++k) {
        kk = *kte - k + *kts + 1;
        i__3 = *ite;
        for (i__ = *its; i__ <= i__3; ++i__) {
        ii = i__ - *its + 1;
        pint[ii + kk - 2] = p8w[i__ + (k + j * p8w_dim2) * p8w_dim1];
        if (k == *kts) {
            ps[ii - 1] = pint[ii + kk - 2];
        }
        lnpint[ii + kk - 2] = cam_log64(pint[ii + kk - 2]);

        }


    }

    if (! (*doabsems) && CAM_LONGWAVE) {
        i__2 = *cam_abs_dim2__;
        for (kk = 1; kk <= i__2; ++kk) {
        i__3 = *kte + 1;
        for (kk1 = *kts; kk1 <= i__3; ++kk1) {
            i__4 = *ite;
            for (i__ = *its; i__ <= i__4; ++i__) {
            abstot[i__ + (kk1 + kk * (((1)*((1-(1)+1)))*((CAM_LEVELS+1-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*((CAM_LEVELS+1-(1)+1))))] = abstot_3d__[i__
                + (kk1 + (kk + j * abstot_3d_dim3) *
                abstot_3d_dim2) * abstot_3d_dim1];

            }


        }


        }

        i__2 = *cam_abs_dim1__;
        for (kk = 1; kk <= i__2; ++kk) {
        i__3 = *kte;
        for (kk1 = *kts; kk1 <= i__3; ++kk1) {
            i__4 = *ite;
            for (i__ = *its; i__ <= i__4; ++i__) {
            absnxt[i__ + (kk1 + (kk * ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))] = absnxt_3d__[
                i__ + (kk1 + (kk + j * absnxt_3d_dim3) *
                absnxt_3d_dim2) * absnxt_3d_dim1];

            }


        }


        }

        i__2 = *kte + 1;
        for (kk = *kts; kk <= i__2; ++kk) {
        i__3 = *ite;
        for (i__ = *its; i__ <= i__3; ++i__) {
            emstot[i__ + kk - 2] = emstot_3d__[i__ + (kk + j *
                emstot_3d_dim2) * emstot_3d_dim1];

        }


        }

    }
    i__2 = *kte;
    for (k = *kts; k <= i__2; ++k) {
        kk = *kte - k + *kts;
        i__3 = *ite;
        for (i__ = *its; i__ <= i__3; ++i__) {
        ii = i__ - *its + 1;
        pmid[ii + kk - 2] = p_phy__[i__ + (k + j * p_phy_dim2) *
            p_phy_dim1];
        lnpmid[ii + kk - 2] = cam_log64(pmid[ii + kk - 2]);
        lnpint[ii + kk - 2] = cam_log64(pint[ii + kk - 2]);
        pdel[ii + kk - 2] = pint[ii + (kk + 1) - 2] - pint[ii + kk -
            2];
        t[ii + kk - 2] = t_phy__[i__ + (k + j * t_phy_dim2) *
            t_phy_dim1];
        zm[ii + kk - 2] = z__[i__ + (k + j * z_dim2) * z_dim1];

        }


    }

    param_cldoptics_calc_(state, &ncol, &pcols, &pver, &pverp, &pverr, &pverrp,
        &ppcnst, q, cld, landfrac, landm, icefrac, pdel, t, ps, pmid,
        pint, cicewp, cliqwp, emis, rel, rei, pmxrgn, nmxrgn, snowh);
    if (*sf_surface_physics__ == 8) {
        if (*xtime > 1.f) {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
            ii = i__ - *its + 1;
            if (xland[i__ + j * xland_dim1] < 1.5f) {
            asdir[ii - 1] = alswvisdir[i__ + j * alswvisdir_dim1];
            asdif[ii - 1] = alswvisdif[i__ + j * alswvisdif_dim1];
            aldir[ii - 1] = alswnirdir[i__ + j * alswnirdir_dim1];
            aldif[ii - 1] = alswnirdif[i__ + j * alswnirdif_dim1];
            } else {
            asdir[ii - 1] = albedo[i__ + j * albedo_dim1];
            asdif[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldir[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldif[ii - 1] = albedo[i__ + j * albedo_dim1];
            }

        }

        } else {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
            ii = i__ - *its + 1;
            asdir[ii - 1] = albedo[i__ + j * albedo_dim1];
            asdif[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldir[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldif[ii - 1] = albedo[i__ + j * albedo_dim1];

        }

        }
    } else if (*sf_surface_physics__ == 5) {
        if (*xtime > 1.f) {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
            ii = i__ - *its + 1;
            if (xland[i__ + j * xland_dim1] < 1.5f) {
            asdir[ii - 1] = alswvisdir[i__ + j * alswvisdir_dim1];
            asdif[ii - 1] = alswvisdif[i__ + j * alswvisdif_dim1];
            aldir[ii - 1] = alswnirdir[i__ + j * alswnirdir_dim1];
            aldif[ii - 1] = alswnirdif[i__ + j * alswnirdif_dim1];
            } else {
            asdir[ii - 1] = albedo[i__ + j * albedo_dim1];
            asdif[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldir[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldif[ii - 1] = albedo[i__ + j * albedo_dim1];
            }

        }

        } else {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
            ii = i__ - *its + 1;
            asdir[ii - 1] = albedo[i__ + j * albedo_dim1];
            asdif[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldir[ii - 1] = albedo[i__ + j * albedo_dim1];
            aldif[ii - 1] = albedo[i__ + j * albedo_dim1];

        }

        }
    } else {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;
        asdir[ii - 1] = albedo[i__ + j * albedo_dim1];
        asdif[ii - 1] = albedo[i__ + j * albedo_dim1];
        aldir[ii - 1] = albedo[i__ + j * albedo_dim1];
        aldif[ii - 1] = albedo[i__ + j * albedo_dim1];

        }

    }
    radctl_(state, &j, &lchnk, &ncol, &pcols, &pver, &pverp, &pverr, &pverrp, &
        ppcnst, &pcnst, lwups, emis, pmid, pint, lnpmid, lnpint, pdel,
         t, q, cld, cicewp, cliqwp, tauxcl, tauxci, coszrs, clat,
        asdir, asdif, aldir, aldif, solcon, gmt, julday, julian, dt,
        xtime, pin, ozmixmj, ozmix, levsiz, num_months__, ghg_input__,
         m_psjp__, m_psjn__, aerosoljp, aerosoljn, m_hybi__, paerlev,
        naer_c__, pmxrgn, nmxrgn, dolw, dosw, doabsems, abstot,
        absnxt, emstot, fsup, fsupc, fsdn, fsdnc, fsdndir, fsdncdir,
        fsdndif, fsdncdif, flup, flupc, fldn, fldnc, swcftoa, lwcftoa,
         olrtoa, fsns, fsnt, flns, flnt, qrs, qrscs, qrl, qrlcs,
        flwds, rel, rei, sols, soll, solsd, solld, &n2ovmr, &ch4vmr, &
        f11vmr, &f12vmr, landfrac, zm, fsds, fsdsdir, fsdsdif);
    if (state->abs_phase == 1) return 0;
    i__2 = *kte;
    for (k = *kts; k <= i__2; ++k) {
        kk = *kte - k + *kts;
        i__3 = *ite;
        for (i__ = *its; i__ <= i__3; ++i__) {
        ii = i__ - *its + 1;
        if (CAM_LONGWAVE) {
            rthratenlw[i__ + (k + j * rthratenlw_dim2) *
                rthratenlw_dim1] = qrl[ii + kk - 2] * 1e4f / (
                state->cg_cpair__ * pi_phy__[i__ + (k + j *
                pi_phy_dim2) * pi_phy_dim1]);
        }
        if (CAM_SHORTWAVE) {
            rthratensw[i__ + (k + j * rthratensw_dim2) *
                rthratensw_dim1] = qrs[ii + kk - 2] * 1e4f / (
                state->cg_cpair__ * pi_phy__[i__ + (k + j *
                pi_phy_dim2) * pi_phy_dim1]);
        }
        if (CAM_LONGWAVE) {
            rthratenlwc[i__ + (k + j * rthratenlwc_dim2) *
                rthratenlwc_dim1] = qrlcs[ii + kk - 2] * 1e4f / (
                state->cg_cpair__ * pi_phy__[i__ + (k + j *
                pi_phy_dim2) * pi_phy_dim1]);
        }
        if (CAM_SHORTWAVE) {
            rthratenswc[i__ + (k + j * rthratenswc_dim2) *
                rthratenswc_dim1] = qrscs[ii + kk - 2] * 1e4f / (
                state->cg_cpair__ * pi_phy__[i__ + (k + j *
                pi_phy_dim2) * pi_phy_dim1]);
        }
        cemiss[i__ + (k + j * cemiss_dim2) * cemiss_dim1] = emis[ii +
            kk - 2];
        taucldc[i__ + (k + j * taucldc_dim2) * taucldc_dim1] = tauxcl[
            ii + kk - 1];
        taucldi[i__ + (k + j * taucldi_dim2) * taucldi_dim1] = tauxci[
            ii + kk - 1];

        }


    }

    if (*doabsems && CAM_LONGWAVE) {
        i__2 = *cam_abs_dim2__;
        for (kk = 1; kk <= i__2; ++kk) {
        i__3 = *kte + 1;
        for (kk1 = *kts; kk1 <= i__3; ++kk1) {
            i__4 = *ite;
            for (i__ = *its; i__ <= i__4; ++i__) {
            abstot_3d__[i__ + (kk1 + (kk + j * abstot_3d_dim3) *
                abstot_3d_dim2) * abstot_3d_dim1] = abstot[
                i__ + (kk1 + kk * (((1)*((1-(1)+1)))*((CAM_LEVELS+1-(1)+1)))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*((CAM_LEVELS+1-(1)+1))))];

            }


        }


        }

        i__2 = *cam_abs_dim1__;
        for (kk = 1; kk <= i__2; ++kk) {
        i__3 = *kte;
        for (kk1 = *kts; kk1 <= i__3; ++kk1) {
            i__4 = *ite;
            for (i__ = *its; i__ <= i__4; ++i__) {
            absnxt_3d__[i__ + (kk1 + (kk + j * absnxt_3d_dim3) *
                absnxt_3d_dim2) * absnxt_3d_dim1] = absnxt[
                i__ + (kk1 + (kk * ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))];

            }


        }


        }

        i__2 = *kte + 1;
        for (kk = *kts; kk <= i__2; ++kk) {
        i__3 = *ite;
        for (i__ = *its; i__ <= i__3; ++i__) {
            emstot_3d__[i__ + (kk + j * emstot_3d_dim2) *
                emstot_3d_dim1] = emstot[i__ + kk - 2];

        }


        }

    }
    if (TRUE_) {
        if (CAM_SHORTWAVE) {
        i__2 = *kte + 1;
        for (k = *kts; k <= i__2; ++k) {
            kk = *kte + 1 - k + *kts;
            i__3 = *ite;
            for (i__ = *its; i__ <= i__3; ++i__) {
            ii = i__ - *its + 1;
            if (k == *kte + 1) {
                swupt[i__ + j * swupt_dim1] = fsup[ii + kk - 2];
                swuptc[i__ + j * swuptc_dim1] = fsupc[ii + kk - 2]
                    ;
                swdnt[i__ + j * swdnt_dim1] = fsdn[ii + kk - 2];
                swdntc[i__ + j * swdntc_dim1] = fsdnc[ii + kk - 2]
                    ;
            }
            if (k == *kts) {
                swupb[i__ + j * swupb_dim1] = fsup[ii + kk - 2];
                swupbc[i__ + j * swupbc_dim1] = fsupc[ii + kk - 2]
                    ;
                swdnb[i__ + j * swdnb_dim1] = fsdn[ii + kk - 2];
                swdnbc[i__ + j * swdnbc_dim1] = fsdnc[ii + kk - 2]
                    ;
            }

            }


        }

        }
        if (CAM_LONGWAVE) {
        i__2 = *kte + 1;
        for (k = *kts; k <= i__2; ++k) {
            kk = *kte + 1 - k + *kts;
            i__3 = *ite;
            for (i__ = *its; i__ <= i__3; ++i__) {
            ii = i__ - *its + 1;
            if (k == *kte + 1) {
                lwupt[i__ + j * lwupt_dim1] = flup[ii + kk - 2];
                lwuptc[i__ + j * lwuptc_dim1] = flupc[ii + kk - 2]
                    ;
                lwdnt[i__ + j * lwdnt_dim1] = fldn[ii + kk - 2];
                lwdntc[i__ + j * lwdntc_dim1] = fldnc[ii + kk - 2]
                    ;
            }
            if (k == *kts) {
                lwupb[i__ + j * lwupb_dim1] = flup[ii + kk - 2];
                lwupbc[i__ + j * lwupbc_dim1] = flupc[ii + kk - 2]
                    ;
                lwdnb[i__ + j * lwdnb_dim1] = fldn[ii + kk - 2];
                lwdnbc[i__ + j * lwdnbc_dim1] = fldnc[ii + kk - 2]
                    ;
            }

            }


        }

        }
    }
    i__2 = *ite;
    for (i__ = *its; i__ <= i__2; ++i__) {
        ii = i__ - *its + 1;
        if (CAM_LONGWAVE) {
        glw[i__ + j * glw_dim1] = flwds[ii - 1];
        lwcf[i__ + j * lwcf_dim1] = lwcftoa[ii - 1];
        olr[i__ + j * olr_dim1] = olrtoa[ii - 1];
        }
        if (CAM_SHORTWAVE) {
        gsw[i__ + j * gsw_dim1] = fsns[ii - 1];
        swcf[i__ + j * swcf_dim1] = swcftoa[ii - 1];
        coszr[i__ + j * coszr_dim1] = coszrs[ii - 1];
        swddir[i__ + j * swddir_dim1] = fsdsdir[ii - 1];
        swddni[i__ + j * swddni_dim1] = fsdsdir[ii - 1] / coszrs[ii -
            1];
        swddif[i__ + j * swddif_dim1] = fsdsdif[ii - 1];
        }

    }

    if (*sf_surface_physics__ == 8) {
        if (CAM_SHORTWAVE) {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
            ii = i__ - *its + 1;
            swvisdir[i__ + j * swvisdir_dim1] = sols[ii - 1];
            swvisdif[i__ + j * swvisdif_dim1] = solsd[ii - 1];
            swnirdir[i__ + j * swnirdir_dim1] = soll[ii - 1];
            swnirdif[i__ + j * swnirdif_dim1] = solld[ii - 1];

        }

        }
    } else if (*sf_surface_physics__ == 5) {
        if (CAM_SHORTWAVE) {
        i__2 = *ite;
        for (i__ = *its; i__ <= i__2; ++i__) {
            ii = i__ - *its + 1;
            swvisdir[i__ + j * swvisdir_dim1] = sols[ii - 1];
            swvisdif[i__ + j * swvisdif_dim1] = solsd[ii - 1];
            swnirdir[i__ + j * swnirdir_dim1] = soll[ii - 1];
            swnirdif[i__ + j * swnirdif_dim1] = solld[ii - 1];

        }

        }
    }

    }

    return 0;
}





__device__ int radinp_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *pmid, doublereal *pint,
    doublereal *o3vmr, doublereal *pmidrd, doublereal *pintrd, doublereal
    *eccf, doublereal *o3mmr)
{
    integer pmid_dim1 = 0,pmid_offset = 0,pint_dim1 = 0,pint_offset = 0,o3vmr_dim1 = 0,o3vmr_offset = 0,pmidrd_dim1 = 0,pmidrd_offset = 0,pintrd_dim1 = 0,pintrd_offset = 0,o3mmr_dim1 = 0,o3mmr_offset = 0,i__1 = 0,i__2 = 0;

    integer i__ = 0,k = 0;

    doublereal vmmr = 0;


    o3mmr_dim1 = 1;
    o3mmr_offset = 1 + o3mmr_dim1;
    o3mmr -= o3mmr_offset;
    pmidrd_dim1 = 1;
    pmidrd_offset = 1 + pmidrd_dim1;
    pmidrd -= pmidrd_offset;
    o3vmr_dim1 = 1;
    o3vmr_offset = 1 + o3vmr_dim1;
    o3vmr -= o3vmr_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    pintrd_dim1 = 1;
    pintrd_offset = 1 + pintrd_dim1;
    pintrd -= pintrd_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    *eccf = 1.f;
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        pmidrd[i__ + k * pmidrd_dim1] = pmid[i__ + k * pmid_dim1] * 10.f;
        pintrd[i__ + k * pintrd_dim1] = pint[i__ + k * pint_dim1] * 10.f;

    }


    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    pintrd[i__ + (CAM_LEVELS+1) * pintrd_dim1] = pint[i__ + (CAM_LEVELS+1) * pint_dim1] *
        10.f;

    }

    vmmr = state->cg_amo__ / state->cg_amd__;
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        o3mmr[i__ + k * o3mmr_dim1] = vmmr * o3vmr[i__ + k * o3vmr_dim1];

    }


    }

    return 0;
}





__device__ int radozn_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, doublereal *pmid, doublereal *pin, integer *levsiz,
    doublereal *ozmix, doublereal *o3vmr)
{
 CamScratchFrame scratch_frame(state);

    integer pmid_dim1 = 0,pmid_offset = 0,ozmix_dim1 = 0,ozmix_offset = 0,o3vmr_dim1 = 0,o3vmr_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0;

    integer i__ = 0,k = 0,kk = 0;

    doublereal dpl = 0,dpu = 0;

    integer kount = 0,kupper[1] = {},kkstart = 0;


    o3vmr_dim1 = 1;
    o3vmr_offset = 1 + o3vmr_dim1;
    o3vmr -= o3vmr_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    ozmix_dim1 = 1;
    ozmix_offset = 1 + ozmix_dim1;
    ozmix -= ozmix_offset;
    --pin;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    kupper[i__ - 1] = 1;

    }

    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    kkstart = *levsiz;
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {

        i__3 = kkstart, i__4 = kupper[i__ - 1];
        kkstart = min(i__3,i__4);

    }

    kount = 0;
    i__2 = *levsiz - 1;
    for (kk = kkstart; kk <= i__2; ++kk) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
        if (pin[kk] < pmid[i__ + k * pmid_dim1] && pmid[i__ + k *
            pmid_dim1] <= pin[kk + 1]) {
            kupper[i__ - 1] = kk;
            ++kount;
        }

        }

        if (kount == 1) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            dpu = pmid[i__ + k * pmid_dim1] - pin[kupper[i__ - 1]];
            dpl = pin[kupper[i__ - 1] + 1] - pmid[i__ + k * pmid_dim1]
                ;
            o3vmr[i__ + k * o3vmr_dim1] = (ozmix[i__ + kupper[i__ - 1]
                 * ozmix_dim1] * dpl + ozmix[i__ + (kupper[i__ -
                1] + 1) * ozmix_dim1] * dpu) / (dpl + dpu);

        }

        goto L35;
        }

    }

    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        if (pmid[i__ + k * pmid_dim1] < pin[1]) {
        o3vmr[i__ + k * o3vmr_dim1] = ozmix[i__ + ozmix_dim1] * pmid[
            i__ + k * pmid_dim1] / pin[1];
        } else if (pmid[i__ + k * pmid_dim1] > pin[*levsiz]) {
        o3vmr[i__ + k * o3vmr_dim1] = ozmix[i__ + *levsiz *
            ozmix_dim1];
        } else {
        dpu = pmid[i__ + k * pmid_dim1] - pin[kupper[i__ - 1]];
        dpl = pin[kupper[i__ - 1] + 1] - pmid[i__ + k * pmid_dim1];
        o3vmr[i__ + k * o3vmr_dim1] = (ozmix[i__ + kupper[i__ - 1] *
            ozmix_dim1] * dpl + ozmix[i__ + (kupper[i__ - 1] + 1)
            * ozmix_dim1] * dpu) / (dpl + dpu);
        }

    }

    if (kount > 1) {
        cam_driver_failure__();
    }
L35:

    ;
    }

    return 0;
}





__device__ int oznint_(CamState* state, integer *julday, real *julian, real *dt, real *
    gmt, real *xtime, doublereal *ozmixmj, doublereal *ozmix, integer *
    levsiz, integer *num_months__, integer *pcols)
{


    integer date_oz__[12] = { 16,45,75,105,136,166,197,228,258,289,319,
        350 };

    integer ozmixmj_dim1 = 0,ozmixmj_dim2 = 0,ozmixmj_offset = 0,ozmix_dim1 = 0,ozmix_offset = 0,i__1 = 0,i__2 = 0;

    logical finddate = 0;

    integer i__ = 0,k = 0,m = 0;

    doublereal intjulian = 0;

    integer nm = 0,np = 0,np1 = 0;

    integer ijul = 0;

    doublereal fact1 = 0,fact2 = 0,cdayozm = 0,cdayozp = 0;


    ozmix_dim1 = 1;
    ozmix_offset = 1 + ozmix_dim1;
    ozmix -= ozmix_offset;
    ozmixmj_dim1 = 1;
    ozmixmj_dim2 = *levsiz;
    ozmixmj_offset = 1 + ozmixmj_dim1 * (1 + ozmixmj_dim2);
    ozmixmj -= ozmixmj_offset;


    intjulian = *julian + 1.;
    ijul = (integer) intjulian;
    intjulian -= (real) ijul;
    ijul %= 365;
    if (ijul == 0) {
    ijul = 365;
    }
    intjulian += ijul;
    np1 = 1;
    finddate = FALSE_;
    for (m = 1; m <= 12; ++m) {
    if ((doublereal) date_oz__[m - 1] > intjulian && ! finddate) {
        np1 = m;
        finddate = TRUE_;
    }

    }

    cdayozp = (doublereal) date_oz__[np1 - 1];
    if (np1 > 1) {
    cdayozm = (doublereal) date_oz__[np1 - 2];
    np = np1;
    nm = np - 1;
    } else {
    cdayozm = (doublereal) date_oz__[11];
    np = np1;
    nm = 12;
    }
    getfactors_(state, &state->cg_ozncyc__, &np1, &cdayozm, &cdayozp, &
        intjulian, &fact1, &fact2);
    i__1 = *levsiz;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        ozmix[i__ + k * ozmix_dim1] = ozmixmj[i__ + (k + nm *
            ozmixmj_dim2) * ozmixmj_dim1] * fact1 + ozmixmj[i__ + (k
            + np * ozmixmj_dim2) * ozmixmj_dim1] * fact2;

    }


    }

    return 0;
}






__device__ logical c_true = TRUE_;


__device__ int get_aerosol_(CamState* state, integer *c__, integer *julday, real *
    julian, real *dt, real *gmt, real *xtime, doublereal *m_psp__,
    doublereal *m_psn__, doublereal *aerosoljp, doublereal *aerosoljn,
    doublereal *m_hybi__, integer *paerlev, integer *naer_c__, doublereal
    *pint, integer *pcols, integer *pver, integer *pverp, integer *pverr,
    integer *pverrp, doublereal *aerosolt, doublereal *scale)
{
 CamScratchFrame scratch_frame(state);

    doublereal mid[12] = { 16.5,46.,75.5,106.,136.5,167.,197.5,228.5,
        259.,289.5,320.,350.5 };

    integer pint_dim1 = 0,pint_offset = 0,aerosoljp_dim1 = 0,aerosoljp_dim2 = 0,aerosoljp_offset = 0,aerosoljn_dim1 = 0,aerosoljn_dim2 = 0,aerosoljn_offset = 0,aerosolt_dim1 = 0,aerosolt_dim2 = 0,aerosolt_offset = 0,i__1 = 0,i__2 = 0;

    doublereal *aerosolm = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))    , *aerosolp = cam_alloc<doublereal>(state, (1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))
        ;

    integer i__ = 0,k = 0,m = 0;

    doublereal caldayloc = 0,intjulian = 0;

    integer nm = 0,np = 0;

    doublereal speciesmin[11] = {};

    integer ncol = 0,ijul = 0;

    doublereal fact1 = 0,fact2 = 0,cdaym = 0,cdayp = 0;

    integer mo_prv__ = 0,mo_nxt__ = 0;

    integer cam_s1_0__ = 0,cam_s1_1__ = 0,cam_s2_0__ = 0;


    --m_hybi__;
    aerosoljn_dim1 = 1;
    aerosoljn_dim2 = *paerlev;
    aerosoljn_offset = 1 + aerosoljn_dim1 * (1 + aerosoljn_dim2);
    aerosoljn -= aerosoljn_offset;
    aerosoljp_dim1 = 1;
    aerosoljp_dim2 = *paerlev;
    aerosoljp_offset = 1 + aerosoljp_dim1 * (1 + aerosoljp_dim2);
    aerosoljp -= aerosoljp_offset;
    --m_psn__;
    --m_psp__;
    aerosolt_dim1 = 1;
    aerosolt_dim2 = CAM_LEVELS;
    aerosolt_offset = 1 + aerosolt_dim1 * (1 + aerosolt_dim2);
    aerosolt -= aerosolt_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;
    --scale;


    intjulian = *julian + 1.;
    ijul = (integer) intjulian;
    intjulian -= (real) ijul;
    ijul %= 365;
    if (ijul == 0) {
    ijul = 365;
    }
    caldayloc = intjulian + ijul;
    if (caldayloc < mid[0]) {
    mo_prv__ = 12;
    mo_nxt__ = 1;
    } else if (caldayloc >= mid[11]) {
    mo_prv__ = 12;
    mo_nxt__ = 1;
    } else {
    for (i__ = 2; i__ <= 12; ++i__) {
        if (caldayloc < mid[i__ - 1]) {
        mo_prv__ = i__ - 1;
        mo_nxt__ = i__;
        goto L9003;
        }

    }
L9003:
    ;
    }
    cdaym = mid[mo_prv__ - 1];
    cdayp = mid[mo_nxt__ - 1];
    getfactors_(state, &c_true, &mo_nxt__, &cdaym, &cdayp, &caldayloc, &fact1, &
        fact2);
    ncol = 1;
    vert_interpolate_(state, &m_psp__[1], &aerosoljp[aerosoljp_offset], &m_hybi__[1]
        , paerlev, naer_c__, &pint[pint_offset], &nm, aerosolm, pcols,
        pver, pverp, &ncol, c__);
    vert_interpolate_(state, &m_psn__[1], &aerosoljn[aerosoljn_offset], &m_hybi__[1]
        , paerlev, naer_c__, &pint[pint_offset], &np, aerosolp, pcols,
        pver, pverp, &ncol, c__);
    for (m = 1; m <= 11; ++m) {
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
        i__2 = ncol;
        for (i__ = 1; i__ <= i__2; ++i__) {
        aerosolt[i__ + (k + m * aerosolt_dim2) * aerosolt_dim1] =
            aerosolm[i__ + (k + (m * ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))] * fact1 +
            aerosolp[i__ + (k + (m * ((1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1)))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*((CAM_LEVELS-(1)+1))))] * fact2;

        }


    }


    }

    background_(state, c__, &ncol, &pint[pint_offset], pcols, pverr, pverrp, &
        aerosolt[(state->cg_idxbg__ * aerosolt_dim2 + 1) *
        aerosolt_dim1 + 1]);
    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
    i__2 = CAM_LEVELS - 1;
    for (cam_s1_1__ = 0; cam_s1_1__ <= i__2; ++cam_s1_1__) {
        aerosolt[cam_s1_0__ + 1 + (cam_s1_1__ + 1 +
            state->cg_idxvolc__ * aerosolt_dim2) *
            aerosolt_dim1] = 0.;

    }


    }

    for (cam_s2_0__ = 0; cam_s2_0__ <= 10; ++cam_s2_0__) {
    speciesmin[cam_s2_0__] = 0.f;

    }

    for (m = 1; m <= 11; ++m) {
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
        i__2 = ncol;
        for (i__ = 1; i__ <= i__2; ++i__) {
        if (aerosolt[i__ + (k + m * aerosolt_dim2) * aerosolt_dim1] <
            speciesmin[m - 1]) {
            cam_driver_failure__();
            cam_driver_failure__();
        }

        }


    }


    }

    scale_aerosols_(state, &aerosolt[aerosolt_offset], pcols, pver, &ncol, c__, &
        scale[1]);
    return 0;
}





__device__ int aerosol_indirect_(CamState* state, integer *ncol, integer *lchnk,
    integer *pcols, integer *pver, integer *ppcnst, doublereal *landfrac,
    doublereal *pmid, doublereal *t, doublereal *qm1, doublereal *cld,
    doublereal *zm, doublereal *rel)
{
 CamScratchFrame scratch_frame(state);

    integer pmid_dim1 = 0,pmid_offset = 0,t_dim1 = 0,t_offset = 0,qm1_dim1 = 0,qm1_dim2 = 0,qm1_offset = 0,cld_dim1 = 0,cld_offset = 0,zm_dim1 = 0,zm_offset = 0,rel_dim1 = 0,rel_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0,d__2 = 0;

    integer i__ = 0,k = 0;
    doublereal locrhoair[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  , aso4[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , wrel[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , wlwc[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , ntot[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , acoef, bgaer,
        locpi, reexp, ntotb, recoef, cldfrq[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}   ,
        relmod[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , lwcwat[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}  ,
        rhowat;

    integer cam_s1_0__ = 0,cam_s1_1__ = 0;
    doublereal so4mass[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}    , rekappa, rdryair,
        sulfmix[(1-(1)+1)*(CAM_LEVELS-(1)+1)] = {}   ;


    --landfrac;
    rel_dim1 = 1;
    rel_offset = 1 + rel_dim1;
    rel -= rel_offset;
    zm_dim1 = 1;
    zm_offset = 1 + zm_dim1;
    zm -= zm_offset;
    cld_dim1 = 1;
    cld_offset = 1 + cld_dim1;
    cld -= cld_offset;
    t_dim1 = 1;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    qm1_dim1 = 1;
    qm1_dim2 = CAM_LEVELS;
    qm1_offset = 1 + qm1_dim1 * (1 + qm1_dim2);
    qm1 -= qm1_offset;


    if (state->cg_indirect__) {
    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
        i__2 = CAM_LEVELS - 1;
        for (cam_s1_1__ = 0; cam_s1_1__ <= i__2; ++cam_s1_1__) {
        sulfmix[cam_s1_0__ + 1 + (cam_s1_1__ + 1) - 2] = 0.;

        }


    }

    locpi = 3.14159274101257324f;
    rdryair = 287.040008544921875f;
    rhowat = 1e3f;
    acoef = 129299998834688.f;
    recoef = 3.f / (locpi * 4.f * rhowat);
    reexp = .33333333333333331f;
    for (k = CAM_LEVELS; k >= 1; --k) {
        i__1 = 1;
        for (i__ = 1; i__ <= i__1; ++i__) {
        locrhoair[i__ + k - 2] = pmid[i__ + k * pmid_dim1] / (rdryair
            * t[i__ + k * t_dim1]);

        d__1 = .01, d__2 = cld[i__ + k * cld_dim1];
        lwcwat[i__ + k - 2] = qm1[i__ + (k +
            state->cg_ixcldliq__ * qm1_dim2) * qm1_dim1] /
            max(d__1,d__2) * locrhoair[i__ + k - 2];
        so4mass[i__ + k - 2] = sulfmix[i__ + k - 2] * locrhoair[i__ +
            k - 2] * .00100000004749745131f;
        aso4[i__ + k - 2] = so4mass[i__ + k - 2] * acoef;
        if (aso4[i__ + k - 2] <= 280.f) {

            d__1 = 36., d__2 = aso4[i__ + k - 2];
            aso4[i__ + k - 2] = max(d__1,d__2);

            d__1 = aso4[i__ + k - 2];
            ntot[i__ + k - 2] = d__1 * d__1 * -.00115000002551823854f
                + aso4[i__ + k - 2] * .962999999523162842f +
                5.30000019073486328f;
            rekappa = .800000011920928955f;
        } else {

            d__1 = 1500., d__2 = aso4[i__ + k - 2];
            aso4[i__ + k - 2] = min(d__1,d__2);

            d__1 = aso4[i__ + k - 2];
            ntot[i__ + k - 2] = d__1 * d__1 *
                -2.09999998332932591e-4f + aso4[i__ + k - 2] *
                .56800001859664917f - 27.8999996185302734f;
            rekappa = .670000016689300537f;
        }
        if ((doublereal) i_dnnt(&landfrac[i__]) > .5) {
            bgaer = cam_exp64(-(zm[i__ + k * zm_dim1] / 2e3)) * 100.;

            d__1 = bgaer, d__2 = ntot[i__ + k - 2];
            ntot[i__ + k - 2] = max(d__1,d__2);
        } else {
            bgaer = cam_exp64(-(zm[i__ + k * zm_dim1] / 1e3)) * 50.;

            d__1 = bgaer, d__2 = ntot[i__ + k - 2];
            ntot[i__ + k - 2] = max(d__1,d__2);
        }
        if (k == CAM_LEVELS) {
            ntotb = ntot[i__ + k - 2];
        } else {
            ntotb = ntot[i__ + (k + 1) - 2];
        }
        d__1 = recoef * lwcwat[i__ + k - 2] / (rekappa * ntotb);
        relmod[i__ + k - 2] = pow_dd(&d__1, &reexp) * 1e4f;

        d__1 = 4., d__2 = relmod[i__ + k - 2];
        relmod[i__ + k - 2] = max(d__1,d__2);

        d__1 = 20., d__2 = relmod[i__ + k - 2];
        relmod[i__ + k - 2] = min(d__1,d__2);
        if (cld[i__ + k * cld_dim1] >= .00999999977648258209f) {
            cldfrq[i__ + k - 2] = 1.f;
        } else {
            cldfrq[i__ + k - 2] = 0.f;
        }
        wrel[i__ + k - 2] = relmod[i__ + k - 2] * cldfrq[i__ + k - 2];
        wlwc[i__ + k - 2] = lwcwat[i__ + k - 2] * cldfrq[i__ + k - 2];

        }


    }

    } else {
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        relmod[i__ + k - 2] = rel[i__ + k * rel_dim1];

        }


    }

    }
    return 0;
}






__device__ doublereal trcmix_clwrf_c_b6 = 2.;


__device__ int trcmix_clwrf_(CamState* state, integer *lchnk, integer *ncol, integer *
    pcols, integer *pver, doublereal *pmid, doublereal *clat, doublereal *
    n2ovmr, doublereal *ch4vmr, doublereal *f11vmr, doublereal *f12vmr,
    doublereal *n2o, doublereal *ch4, doublereal *cfc11, doublereal *
    cfc12)
{
 CamScratchFrame scratch_frame(state);

    integer pmid_dim1 = 0,pmid_offset = 0,n2o_dim1 = 0,n2o_offset = 0,ch4_dim1 = 0,ch4_offset = 0,cfc11_dim1 = 0,cfc11_offset = 0,cfc12_dim1 = 0,cfc12_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,k = 0;

    doublereal ch40 = 0,n2o0 = 0,xch4 = 0,xn2o = 0,dlat = 0,cfc110 = 0,cfc120 = 0,xcfc11 = 0,xcfc12 = 0,rmwf11 = 0,rmwf12 = 0,ptrop = 0,rmwch4 = 0,rmwn2o = 0,coslat[1] = {},pratio = 0;


    --clat;
    cfc12_dim1 = 1;
    cfc12_offset = 1 + cfc12_dim1;
    cfc12 -= cfc12_offset;
    cfc11_dim1 = 1;
    cfc11_offset = 1 + cfc11_dim1;
    cfc11 -= cfc11_offset;
    ch4_dim1 = 1;
    ch4_offset = 1 + ch4_dim1;
    ch4 -= ch4_offset;
    n2o_dim1 = 1;
    n2o_offset = 1 + n2o_dim1;
    n2o -= n2o_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;


    rmwn2o = state->cg_mwn2o__ / state->cg_mwdry__;
    rmwch4 = state->cg_mwch4__ / state->cg_mwdry__;
    rmwf11 = state->cg_mwf11__ / state->cg_mwdry__;
    rmwf12 = state->cg_mwf12__ / state->cg_mwdry__;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    coslat[i__ - 1] = cos(clat[i__]);

    }

    ch40 = rmwch4 * *ch4vmr;
    n2o0 = rmwn2o * *n2ovmr;
    cfc110 = rmwf11 * *f11vmr;
    cfc120 = rmwf12 * *f12vmr;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    coslat[i__ - 1] = cos(clat[i__]);

    }

    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        dlat = (d__1 = clat[i__] * 57.2957992553710938f, abs(d__1));
        if (dlat <= 45.f) {
        xn2o = dlat * .00115999998524785042f + .347799986600875854f;
        xch4 = .235300004482269287f;
        xcfc11 = dlat * .00606000004336237907f + .727299988269805908f;
        xcfc12 = dlat * .00221999990753829479f + .400000005960464478f;
        } else {
        xn2o = (dlat - 45) * .0133330002427101135f +
            .400000005960464478f;
        xch4 = (dlat - 45) * .0225489009171724319f +
            .235300004482269287f;
        xcfc11 = (dlat - 45) * .0133330002427101135f + 1.f;
        xcfc12 = (dlat - 45) * .0244440007954835892f + .5f;
        }
        ptrop = 2.5e4f - pow_dd(&coslat[i__ - 1], &trcmix_clwrf_c_b6) * 1.5e4f;
        if (pmid[i__ + k * pmid_dim1] >= ptrop) {
        ch4[i__ + k * ch4_dim1] = ch40;
        n2o[i__ + k * n2o_dim1] = n2o0;
        cfc11[i__ + k * cfc11_dim1] = cfc110;
        cfc12[i__ + k * cfc12_dim1] = cfc120;
        } else {
        pratio = pmid[i__ + k * pmid_dim1] / ptrop;
        ch4[i__ + k * ch4_dim1] = ch40 * pow_dd(&pratio, &xch4);
        n2o[i__ + k * n2o_dim1] = n2o0 * pow_dd(&pratio, &xn2o);
        cfc11[i__ + k * cfc11_dim1] = cfc110 * pow_dd(&pratio, &
            xcfc11);
        cfc12[i__ + k * cfc12_dim1] = cfc120 * pow_dd(&pratio, &
            xcfc12);
        }

    }


    }

    return 0;
}





__device__ int aqsat_(CamState* state, doublereal *t, doublereal *p, doublereal *es,
    doublereal *qs, integer *ii, integer *ilen, integer *kk, integer *
    kstart, integer *kend)
{
    integer t_dim1 = 0,t_offset = 0,p_dim1 = 0,p_offset = 0,es_dim1 = 0,es_offset = 0,qs_dim1 = 0,qs_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0,d__2 = 0;

    integer i__ = 0,k = 0;

    doublereal omeps = 0;


    qs_dim1 = *ii;
    qs_offset = 1 + qs_dim1;
    qs -= qs_offset;
    es_dim1 = *ii;
    es_offset = 1 + es_dim1;
    es -= es_offset;
    p_dim1 = *ii;
    p_offset = 1 + p_dim1;
    p -= p_offset;
    t_dim1 = *ii;
    t_offset = 1 + t_dim1;
    t -= t_offset;


    omeps = 1.f - state->cg_epsqs__;
    i__1 = *kend;
    for (k = *kstart; k <= i__1; ++k) {
    i__2 = *ilen;
    for (i__ = 1; i__ <= i__2; ++i__) {
        es[i__ + k * es_dim1] = estblf_(state, &t[i__ + k * t_dim1]);
        qs[i__ + k * qs_dim1] = state->cg_epsqs__ * es[i__ + k *
            es_dim1] / (p[i__ + k * p_dim1] - omeps * es[i__ + k *
            es_dim1]);

        d__1 = 1., d__2 = qs[i__ + k * qs_dim1];
        qs[i__ + k * qs_dim1] = min(d__1,d__2);
        if (qs[i__ + k * qs_dim1] < 0.f) {
        qs[i__ + k * qs_dim1] = 1.f;
        es[i__ + k * es_dim1] = p[i__ + k * p_dim1];
        }

    }


    }

    return 0;
}





__device__ int cldefr_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *landfrac, doublereal *t,
    doublereal *rel, doublereal *rei, doublereal *ps, doublereal *pmid,
    doublereal *landm, doublereal *icefrac, doublereal *snowh)
{
    integer t_dim1 = 0,t_offset = 0,pmid_dim1 = 0,pmid_offset = 0,rel_dim1 = 0,rel_offset = 0,rei_dim1 = 0,rei_offset = 0;


    --snowh;
    --icefrac;
    --landm;
    --ps;
    --landfrac;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;
    rei_dim1 = 1;
    rei_offset = 1 + rei_dim1;
    rei -= rei_offset;
    rel_dim1 = 1;
    rel_offset = 1 + rel_dim1;
    rel -= rel_offset;
    t_dim1 = 1;
    t_offset = 1 + t_dim1;
    t -= t_offset;


    reltab_(state, ncol, pcols, pver, &t[t_offset], &landfrac[1], &landm[1], &
        icefrac[1], &rel[rel_offset], &snowh[1]);
    reitab_(state, ncol, pcols, pver, &t[t_offset], &rei[rei_offset]);
    return 0;
}





__device__ int cldems_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, integer *pverp, doublereal *clwp, doublereal *fice,
    doublereal *rei, doublereal *emis)
{
    integer clwp_dim1 = 0,clwp_offset = 0,rei_dim1 = 0,rei_offset = 0,fice_dim1 = 0,fice_offset = 0,emis_dim1 = 0,emis_offset = 0,i__1 = 0,i__2 = 0;

    integer i__ = 0,k = 0;

    doublereal kabs = 0,kabsi = 0;


    emis_dim1 = 1;
    emis_offset = 1 + emis_dim1;
    emis -= emis_offset;
    rei_dim1 = 1;
    rei_offset = 1 + rei_dim1;
    rei -= rei_offset;
    fice_dim1 = 1;
    fice_offset = 1 + fice_dim1;
    fice -= fice_offset;
    clwp_dim1 = 1;
    clwp_offset = 1 + clwp_dim1;
    clwp -= clwp_offset;


    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        kabsi = 1.f / rei[i__ + k * rei_dim1] + .00499999988824129105f;
        kabs = (1.f - fice[i__ + k * fice_dim1]) * .0903609991073608398 +
            kabsi * fice[i__ + k * fice_dim1];
        emis[i__ + k * emis_dim1] = 1.f - cam_exp64(kabs *
            -1.65999996662139893f * clwp[i__ + k * clwp_dim1]);

    }


    }

    return 0;
}





__device__ int cldovrlap_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
     integer *pver, integer *pverp, doublereal *pint, doublereal *cld,
    integer *nmxrgn, doublereal *pmxrgn)
{
 CamScratchFrame scratch_frame(state);

    integer pint_dim1 = 0,pint_offset = 0,cld_dim1 = 0,cld_offset = 0,pmxrgn_dim1 = 0,pmxrgn_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0;

    integer i__ = 0,k = 0,n = 0,cam_red_k__ = 0;
    logical cld_found__, cld_layer__[(CAM_LEVELS-(1)+1)] = {};

    integer cam_count_4__ = 0;
    doublereal pnm[(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)] = {}    ;

    integer cam_s1_0__ = 0,cam_s2_1__ = 0,cam_s3_1__ = 0;


    --nmxrgn;
    cld_dim1 = 1;
    cld_offset = 1 + cld_dim1;
    cld -= cld_offset;
    pmxrgn_dim1 = 1;
    pmxrgn_offset = 1 + pmxrgn_dim1;
    pmxrgn -= pmxrgn_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    cld_found__ = FALSE_;
    i__2 = CAM_LEVELS - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__2; ++cam_s1_0__) {
        cld_layer__[cam_s1_0__] = cld[i__ + (cam_s1_0__ + 1) * cld_dim1]
            > 0.;

    }

    i__2 = (CAM_LEVELS+1) - 1;
    for (cam_s2_1__ = 0; cam_s2_1__ <= i__2; ++cam_s2_1__) {
        pmxrgn[i__ + (cam_s2_1__ + 1) * pmxrgn_dim1] = 0.f;

    }

    i__2 = (CAM_LEVELS+1) - 1;
    for (cam_s3_1__ = 0; cam_s3_1__ <= i__2; ++cam_s3_1__) {
        pnm[i__ + (cam_s3_1__ + 1) - 2] = pint[i__ + (cam_s3_1__ + 1) *
            pint_dim1] * 10.f;

    }

    n = 1;
    i__2 = CAM_LEVELS;
    for (k = 1; k <= i__2; ++k) {
        if (cld_layer__[k - 1] && ! cld_found__) {
        cld_found__ = TRUE_;
        } else if (! cld_layer__[k - 1] && cld_found__) {
        cld_found__ = FALSE_;
        cam_count_4__ = 0;
        i__3 = CAM_LEVELS;
        for (cam_red_k__ = k; cam_red_k__ <= i__3; ++cam_red_k__) {
            if (cld_layer__[cam_red_k__ - 1]) {
            ++cam_count_4__;
            }

        }

        if (cam_count_4__ == 0) {
            goto L9011;
        }
        pmxrgn[i__ + n * pmxrgn_dim1] = pnm[i__ + k - 2];
        ++n;
        }

    }
L9011:
    pmxrgn[i__ + n * pmxrgn_dim1] = pnm[i__ + (CAM_LEVELS+1) - 2];
    nmxrgn[i__] = n;

    }

    return 0;
}





__device__ int getfactors_(CamState* state, logical *cycflag, integer *np1, doublereal *
    cdayminus, doublereal *cdayplus, doublereal *cday, doublereal *fact1,
    doublereal *fact2)
{
    doublereal deltat = 0;

    if (*cycflag) {
    if (*cday < 1.f || *cday > 366.) {
        cam_driver_failure__();
    }
    } else {
    if (*cday < 1.f) {
        cam_driver_failure__();
    }
    }
    if (*cycflag && *np1 == 1) {
    deltat = *cdayplus + 365. - *cdayminus;
    if (*cday > *cdayplus) {
        *fact1 = (*cdayplus + 365. - *cday) / deltat;
        *fact2 = (*cday - *cdayminus) / deltat;
    } else {
        *fact1 = (*cdayplus - *cday) / deltat;
        *fact2 = (*cday + 365. - *cdayminus) / deltat;
    }
    } else {
    deltat = *cdayplus - *cdayminus;
    *fact1 = (*cdayplus - *cday) / deltat;
    *fact2 = (*cday - *cdayminus) / deltat;
    }
    if (! validfactors_(state, fact1, fact2)) {
    cam_driver_failure__();
    }
    return 0;
}





__device__ int get_rf_scales_(CamState* state, doublereal *scales)
{
    integer i__1 = 0;

    integer i__ = 0;


    --scales;


    scales[state->cg_idxbg__] = state->cg_bgscl_rf__;
    scales[state->cg_idxsul__] = state->cg_sulscl_rf__;
    scales[state->cg_idxsslt__] = state->cg_ssltscl_rf__;
    i__1 = state->cg_idxcarbonfirst__ + 3;
    for (i__ = state->cg_idxcarbonfirst__; i__ <= i__1; ++i__) {
    scales[i__] = state->cg_carscl_rf__;

    }

    i__1 = state->cg_idxdustfirst__ + 3;
    for (i__ = state->cg_idxdustfirst__; i__ <= i__1; ++i__) {
    scales[i__] = state->cg_dustscl_rf__;

    }

    scales[state->cg_idxvolc__] = state->cg_volcscl_rf__;
    return 0;
}





__device__ int get_int_scales_(CamState* state, doublereal *scales)
{
    integer i__1 = 0;

    integer i__ = 0,cam_s1_0__ = 0;


    --scales;


    for (cam_s1_0__ = 0; cam_s1_0__ <= 12; ++cam_s1_0__) {
    scales[cam_s1_0__ + 1] = 1.f;

    }

    scales[state->cg_idxbg__] = 1.;
    scales[state->cg_idxsul__] = state->cg_sulscl__;
    scales[state->cg_idxsslt__] = state->cg_ssltscl__;
    i__1 = state->cg_idxcarbonfirst__ + 3;
    for (i__ = state->cg_idxcarbonfirst__; i__ <= i__1; ++i__) {
    scales[i__] = state->cg_carscl__;

    }

    i__1 = state->cg_idxdustfirst__ + 3;
    for (i__ = state->cg_idxdustfirst__; i__ <= i__1; ++i__) {
    scales[i__] = state->cg_dustscl__;

    }

    scales[state->cg_idxvolc__] = state->cg_volcscl__;
    return 0;
}





__device__ int vert_interpolate_(CamState* state, doublereal *match_ps__, doublereal *
    aerosolc, doublereal *m_hybi__, integer *paerlev, integer *naer_c__,
    doublereal *pint, integer *n, doublereal *aerosol_mmr__, integer *
    pcols, integer *pver, integer *pverp, integer *ncol, integer *c__)
{
 CamScratchFrame scratch_frame(state);

    integer aerosol_mmr_dim1 = 0,aerosol_mmr_dim2 = 0,aerosol_mmr_offset = 0,pint_dim1 = 0,pint_offset = 0,aerosolc_dim1 = 0,aerosolc_dim2 = 0,aerosolc_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0,i__4 = 0;

    logical lev_interp_comp__ = 0;

    doublereal aer_diff__ = 0,m_to_mmr__ = 0;

    integer i__ = 0,k = 0,m = 0,kk = 0;

    doublereal dpl = 0,dpu = 0;

    integer kount = 0,kupper[1] = {},cam_s1_0__ = 0,cam_s1_2__ = 0;

    doublereal v_coord__, *aerosol = cam_alloc<doublereal>(state, (1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(11-(1)+1))  ;

    integer kkstart = 0;


    --m_hybi__;
    aerosolc_dim1 = 1;
    aerosolc_dim2 = *paerlev;
    aerosolc_offset = 1 + aerosolc_dim1 * (1 + aerosolc_dim2);
    aerosolc -= aerosolc_offset;
    --match_ps__;
    aerosol_mmr_dim1 = 1;
    aerosol_mmr_dim2 = CAM_LEVELS;
    aerosol_mmr_offset = 1 + aerosol_mmr_dim1 * (1 + aerosol_mmr_dim2);
    aerosol_mmr__ -= aerosol_mmr_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    kupper[i__ - 1] = 1;

    }

    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    for (m = 1; m <= 11; ++m) {
        aerosol[i__ + (m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) + 1) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = aerosolc[i__ + (m *
            aerosolc_dim2 + 1) * aerosolc_dim1];

    }


    }

    i__1 = CAM_LEVELS;
    for (k = 2; k <= i__1; ++k) {
    kkstart = *paerlev;
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {

        i__3 = kkstart, i__4 = kupper[i__ - 1];
        kkstart = min(i__3,i__4);

    }

    kount = 0;
    lev_interp_comp__ = FALSE_;
    i__2 = *paerlev - 1;
    for (kk = kkstart; kk <= i__2; ++kk) {
        if (! lev_interp_comp__) {
        i__3 = 1;
        for (i__ = 1; i__ <= i__3; ++i__) {
            v_coord__ = pint[i__ + k * pint_dim1];
            if (m_hybi__[kk] * match_ps__[i__] < v_coord__ &&
                v_coord__ <= m_hybi__[kk + 1] * match_ps__[i__]) {
            kupper[i__ - 1] = kk;
            ++kount;
            }

        }

        if (kount == 1) {
            i__3 = 1;
            for (i__ = 1; i__ <= i__3; ++i__) {
            for (m = 1; m <= 11; ++m) {
                dpu = pint[i__ + k * pint_dim1] - m_hybi__[kupper[
                    i__ - 1]] * match_ps__[i__];
                dpl = m_hybi__[kupper[i__ - 1] + 1] * match_ps__[
                    i__] - pint[i__ + k * pint_dim1];
                aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = (aerosolc[i__
                    + (kupper[i__ - 1] + m * aerosolc_dim2) *
                    aerosolc_dim1] * dpl + aerosolc[i__ + (
                    kupper[i__ - 1] + 1 + m * aerosolc_dim2) *
                     aerosolc_dim1] * dpu) / (dpl + dpu);

            }


            }

            lev_interp_comp__ = TRUE_;
        }
        }

    }

    if (! lev_interp_comp__) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        for (m = 1; m <= 11; ++m) {
            if (pint[i__ + k * pint_dim1] < m_hybi__[1] * match_ps__[
                i__]) {
            aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = aerosolc[i__ + (m *
                 aerosolc_dim2 + 1) * aerosolc_dim1];
            } else if (pint[i__ + k * pint_dim1] > m_hybi__[*paerlev]
                * match_ps__[i__]) {
            aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = 0.f;
            } else {
            dpu = pint[i__ + k * pint_dim1] - m_hybi__[kupper[i__
                - 1]] * match_ps__[i__];
            dpl = m_hybi__[kupper[i__ - 1] + 1] * match_ps__[i__]
                - pint[i__ + k * pint_dim1];
            aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = (aerosolc[i__ + (
                kupper[i__ - 1] + m * aerosolc_dim2) *
                aerosolc_dim1] * dpl + aerosolc[i__ + (kupper[
                i__ - 1] + 1 + m * aerosolc_dim2) *
                aerosolc_dim1] * dpu) / (dpl + dpu);
            }

        }


        }

        if (kount > 1) {
        cam_driver_failure__();
        }
    }

    }

    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
    for (cam_s1_2__ = 0; cam_s1_2__ <= 10; ++cam_s1_2__) {
        aerosol[cam_s1_0__ + 1 + ((CAM_LEVELS+1) + (cam_s1_2__ + 1) * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] =
            0.f;

    }


    }

    for (m = 1; m <= 11; ++m) {
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        if (aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] < 1e-40) {
            aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] = 0.f;
        }

        }


    }


    }

    for (m = 1; m <= 11; ++m) {
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
        i__2 = 1;
        for (i__ = 1; i__ <= i__2; ++i__) {
        aer_diff__ = aerosol[i__ + (k + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] - aerosol[i__ +
            (k + 1 + m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))];
        if (abs(aer_diff__) < aerosol[i__ + (m * ((1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1)))) + 1) - ((1)*(1)+(1)*((1)*((1-(1)+1)))+(1)*(((1)*((1-(1)+1)))*(((CAM_LEVELS+1)-(1)+1))))] *
            1.00000000362749373e-15f) {
            aer_diff__ = 0.f;
        }
        m_to_mmr__ = state->cg_gravmks__ / (pint[i__ + (k + 1) *
             pint_dim1] - pint[i__ + k * pint_dim1]);
        aerosol_mmr__[i__ + (k + m * aerosol_mmr_dim2) *
            aerosol_mmr_dim1] = aer_diff__ * m_to_mmr__;
        if (aerosol_mmr__[i__ + (k + m * aerosol_mmr_dim2) *
            aerosol_mmr_dim1] < 0.) {
            cam_driver_failure__();
            cam_driver_failure__();
        }

        }


    }


    }

    return 0;
}





__device__ int background_(CamState* state, integer *lchnk, integer *ncol, doublereal *
    pint, integer *pcols, integer *pverr, integer *pverrp, doublereal *
    mmr)
{
    integer pint_dim1 = 0,pint_offset = 0,mmr_dim1 = 0,mmr_offset = 0,i__1 = 0,i__2 = 0;

    doublereal mass2mmr = 0;

    integer i__ = 0,k = 0;

    doublereal mass = 0;


    mmr_dim1 = 1;
    mmr_offset = 1 + mmr_dim1;
    mmr -= mmr_offset;
    pint_dim1 = 1;
    pint_offset = 1 + pint_dim1;
    pint -= pint_offset;


    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    mass2mmr = state->cg_gravmks__ / (pint[i__ + *pverrp *
        pint_dim1] - pint[i__ + (*pverrp - state->cg_mxaerl__) *
         pint_dim1]);
    i__2 = *pverr;
    for (k = 1; k <= i__2; ++k) {
        if (k >= *pverrp - state->cg_mxaerl__) {
        mass = state->cg_tauback__ / (state->cg_kbg__[7] *
             1e3f);
        mmr[i__ + k * mmr_dim1] = mass2mmr * mass;
        } else {
        mmr[i__ + k * mmr_dim1] = 0.;
        }

    }


    }

    return 0;
}





__device__ int scale_aerosols_(CamState* state, doublereal *aerosolt, integer *pcols,
    integer *pver, integer *ncol, integer *lchnk, doublereal *scale)
{
    integer aerosolt_dim1 = 0,aerosolt_dim2 = 0,aerosolt_offset = 0,i__1 = 0,i__2 = 0;

    integer m = 0,cam_s1_0__ = 0,cam_s1_1__ = 0;


    aerosolt_dim1 = 1;
    aerosolt_dim2 = CAM_LEVELS;
    aerosolt_offset = 1 + aerosolt_dim1 * (1 + aerosolt_dim2);
    aerosolt -= aerosolt_offset;
    --scale;


    for (m = 1; m <= 13; ++m) {
    i__1 = 1 - 1;
    for (cam_s1_0__ = 0; cam_s1_0__ <= i__1; ++cam_s1_0__) {
        i__2 = CAM_LEVELS - 1;
        for (cam_s1_1__ = 0; cam_s1_1__ <= i__2; ++cam_s1_1__) {
        aerosolt[cam_s1_0__ + 1 + (cam_s1_1__ + 1 + m * aerosolt_dim2)
             * aerosolt_dim1] = scale[m] * aerosolt[cam_s1_0__ +
            1 + (cam_s1_1__ + 1 + m * aerosolt_dim2) *
            aerosolt_dim1];

        }


    }


    }

    return 0;
}





__device__ doublereal estblf_(CamState* state, doublereal *td)
{
    doublereal ret_val = 0,d__1 = 0;

    doublereal e = 0;

    integer i__ = 0;

    doublereal ai = 0;


    d__1 = min(*td,state->cg_tmax__);
    e = max(d__1,state->cg_tmin__);
    i__ = (integer) (e - state->cg_tmin__) + 1;
    d__1 = e - state->cg_tmin__;
    ai = d_int(&d__1);
    ret_val = (state->cg_tmin__ + ai - e + 1.f) *
        state->cg_estbl__[i__ - 1] - (state->cg_tmin__ + ai -
        e) * state->cg_estbl__[i__];
    return ret_val;
}





__device__ logical validfactors_(CamState* state, doublereal *fact1, doublereal *fact2)
{
    doublereal d__1 = 0;

    logical ret_val = 0;

    ret_val = TRUE_;
    if ((d__1 = *fact1 + *fact2 - 1.f, abs(d__1)) > 9.99999997475242708e-7f ||
         *fact1 > 1.00000095367431641f || *fact1 <
        -9.99999997475242708e-7f || *fact2 > 1.00000095367431641f || *
        fact2 < -9.99999997475242708e-7f) {
    ret_val = FALSE_;
    }
    return ret_val;
}





__device__ int reltab_(CamState* state, integer *ncol, integer *pcols, integer *pver,
    doublereal *t, doublereal *landfrac, doublereal *landm, doublereal *
    icefrac, doublereal *rel, doublereal *snowh)
{
    integer t_dim1 = 0,t_offset = 0,rel_dim1 = 0,rel_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0,d__2 = 0,d__3 = 0,d__4 = 0;

    doublereal rliqland = 0;

    integer i__ = 0,k = 0;

    doublereal rliqocean = 0,rliqice = 0;


    --snowh;
    --icefrac;
    --landm;
    --landfrac;
    rel_dim1 = 1;
    rel_offset = 1 + rel_dim1;
    rel -= rel_offset;
    t_dim1 = 1;
    t_offset = 1 + t_dim1;
    t -= t_offset;


    rliqocean = 14.;
    rliqice = 14.;
    rliqland = 8.;
    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {


        d__3 = 0., d__4 = (state->cg_tmelt__ - t[i__ + k * t_dim1])
            * .0500000007450580597f;
        d__1 = 1., d__2 = max(d__3,d__4);
        rel[i__ + k * rel_dim1] = rliqland + (rliqocean - rliqland) * min(
            d__1,d__2);


        d__3 = 0., d__4 = snowh[i__] * 10.f;
        d__1 = 1., d__2 = max(d__3,d__4);
        rel[i__ + k * rel_dim1] += (rliqocean - rel[i__ + k * rel_dim1]) *
             min(d__1,d__2);


        d__3 = 0., d__4 = 1.f - landm[i__];
        d__1 = 1., d__2 = max(d__3,d__4);
        rel[i__ + k * rel_dim1] += (rliqocean - rel[i__ + k * rel_dim1]) *
             min(d__1,d__2);


        d__3 = 0., d__4 = icefrac[i__];
        d__1 = 1., d__2 = max(d__3,d__4);
        rel[i__ + k * rel_dim1] += (rliqice - rel[i__ + k * rel_dim1]) *
            min(d__1,d__2);

    }


    }

    return 0;
}





__device__ int reitab_(CamState* state, integer *ncol, integer *pcols, integer *pver,
    doublereal *t, doublereal *re)
{
    integer re_dim1 = 0,re_offset = 0,t_dim1 = 0,t_offset = 0,i__1 = 0,i__2 = 0,i__3 = 0;

    integer i__ = 0,k = 0;

    doublereal corr = 0;

    integer index = 0;


    re_dim1 = 1;
    re_offset = 1 + re_dim1;
    re -= re_offset;
    t_dim1 = 1;
    t_offset = 1 + t_dim1;
    t -= t_offset;


    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        index = (integer) (t[i__ + k * t_dim1] - 179.f);

        i__3 = max(index,1);
        index = min(i__3,94);
        corr = t[i__ + k * t_dim1] - (integer) t[i__ + k * t_dim1];
        re[i__ + k * re_dim1] = state->cg_retab__[index - 1] * (1.f
            - corr) + state->cg_retab__[index] * corr;

    }


    }

    return 0;
}






__device__ doublereal trcmix_c_b6 = 2.;


__device__ int trcmix_(CamState* state, integer *lchnk, integer *ncol, integer *pcols,
    integer *pver, doublereal *pmid, doublereal *clat, doublereal *n2o,
    doublereal *ch4, doublereal *cfc11, doublereal *cfc12)
{
 CamScratchFrame scratch_frame(state);

    integer pmid_dim1 = 0,pmid_offset = 0,n2o_dim1 = 0,n2o_offset = 0,ch4_dim1 = 0,ch4_offset = 0,cfc11_dim1 = 0,cfc11_offset = 0,cfc12_dim1 = 0,cfc12_offset = 0,i__1 = 0,i__2 = 0;

    doublereal d__1 = 0;

    integer i__ = 0,k = 0;

    doublereal ch40 = 0,n2o0 = 0,xch4 = 0,xn2o = 0,dlat = 0,cfc110 = 0,cfc120 = 0,xcfc11 = 0,xcfc12 = 0,rmwf11 = 0,rmwf12 = 0,ptrop = 0,rmwch4 = 0,rmwn2o = 0,coslat[1] = {},pratio = 0;


    --clat;
    cfc12_dim1 = 1;
    cfc12_offset = 1 + cfc12_dim1;
    cfc12 -= cfc12_offset;
    cfc11_dim1 = 1;
    cfc11_offset = 1 + cfc11_dim1;
    cfc11 -= cfc11_offset;
    ch4_dim1 = 1;
    ch4_offset = 1 + ch4_dim1;
    ch4 -= ch4_offset;
    n2o_dim1 = 1;
    n2o_offset = 1 + n2o_dim1;
    n2o -= n2o_offset;
    pmid_dim1 = 1;
    pmid_offset = 1 + pmid_dim1;
    pmid -= pmid_offset;


    rmwn2o = state->cg_mwn2o__ / state->cg_mwdry__;
    rmwch4 = state->cg_mwch4__ / state->cg_mwdry__;
    rmwf11 = state->cg_mwf11__ / state->cg_mwdry__;
    rmwf12 = state->cg_mwf12__ / state->cg_mwdry__;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    coslat[i__ - 1] = cos(clat[i__]);

    }

    ch40 = rmwch4 * state->cg_ch4vmr__;
    n2o0 = rmwn2o * state->cg_n2ovmr__;
    cfc110 = rmwf11 * state->cg_f11vmr__;
    cfc120 = rmwf12 * state->cg_f12vmr__;
    i__1 = 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
    coslat[i__ - 1] = cos(clat[i__]);

    }

    i__1 = CAM_LEVELS;
    for (k = 1; k <= i__1; ++k) {
    i__2 = 1;
    for (i__ = 1; i__ <= i__2; ++i__) {
        dlat = (d__1 = clat[i__] * 57.2957992553710938f, abs(d__1));
        if (dlat <= 45.f) {
        xn2o = dlat * .00115999998524785042f + .347799986600875854f;
        xch4 = .235300004482269287f;
        xcfc11 = dlat * .00606000004336237907f + .727299988269805908f;
        xcfc12 = dlat * .00221999990753829479f + .400000005960464478f;
        } else {
        xn2o = (dlat - 45) * .0133330002427101135f +
            .400000005960464478f;
        xch4 = (dlat - 45) * .0225489009171724319f +
            .235300004482269287f;
        xcfc11 = (dlat - 45) * .0133330002427101135f + 1.f;
        xcfc12 = (dlat - 45) * .0244440007954835892f + .5f;
        }
        ptrop = 2.5e4f - pow_dd(&coslat[i__ - 1], &trcmix_c_b6) * 1.5e4f;
        if (pmid[i__ + k * pmid_dim1] >= ptrop) {
        ch4[i__ + k * ch4_dim1] = ch40;
        n2o[i__ + k * n2o_dim1] = n2o0;
        cfc11[i__ + k * cfc11_dim1] = cfc110;
        cfc12[i__ + k * cfc12_dim1] = cfc120;
        } else {
        pratio = pmid[i__ + k * pmid_dim1] / ptrop;
        ch4[i__ + k * ch4_dim1] = ch40 * pow_dd(&pratio, &xch4);
        n2o[i__ + k * n2o_dim1] = n2o0 * pow_dd(&pratio, &xn2o);
        cfc11[i__ + k * cfc11_dim1] = cfc110 * pow_dd(&pratio, &
            xcfc11);
        cfc12[i__ + k * cfc12_dim1] = cfc120 * pow_dd(&pratio, &
            xcfc12);
        }

    }


    }

    return 0;
}


extern "C" __global__ __launch_bounds__(32) void cam_emissivity(int columns,const unsigned char* bytes,unsigned char* workspace,double* a0,double* a1,double* a2,double* a3,double* a4,double* a5,double* a6,double* a7,double* a8,double* a9,double* a10,double* a11,double* a12,double* a13,double* a14,double* a15,double* a16,double* a17,double* a18,double* a19,double* a20,double* a21,double* a22,double* a23,double* a24,double* a25,double* a26,double* a27,double* a28,double* a29,double* a30,double* a31,double* a32,double* a33,double* a34,double* a35,double* a36,double* a37){
int column=blockIdx.x*blockDim.x+threadIdx.x;
if(column>=columns)return;
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
CamState* state=(CamState*)(workspace+CAM_WORK_BYTES*column);
state->cg_abari__=(double*)(bytes+0);
state->cg_abarl__=(double*)(bytes+32);
state->cg_abco2__=(double*)(bytes+64);
state->cg_abh2o__=(double*)(bytes+216);
state->cg_abo2__=(double*)(bytes+368);
state->cg_abo3__=(double*)(bytes+520);
state->cg_abs_cff_mss_aer__=(double*)(bytes+672);
state->cg_ah2onw__=(double*)(bytes+728);
state->cg_ah2ow__=(double*)(bytes+2940728);
state->cg_amco2__=*(double*)(bytes+5880728);
state->cg_amd__=*(double*)(bytes+5880736);
state->cg_amo__=*(double*)(bytes+5880744);
state->cg_bbari__=(double*)(bytes+5880752);
state->cg_bbarl__=(double*)(bytes+5880784);
state->cg_bgscl_rf__=*(double*)(bytes+5880816);
state->cg_c16__=*(double*)(bytes+5880824);
state->cg_c17__=*(double*)(bytes+5880832);
state->cg_c26__=*(double*)(bytes+5880840);
state->cg_c27__=*(double*)(bytes+5880848);
state->cg_c28__=*(double*)(bytes+5880856);
state->cg_c29__=*(double*)(bytes+5880864);
state->cg_c30__=*(double*)(bytes+5880872);
state->cg_c31__=*(double*)(bytes+5880880);
state->cg_cappa__=*(double*)(bytes+5880888);
state->cg_carscl__=*(double*)(bytes+5880896);
state->cg_carscl_rf__=*(double*)(bytes+5880904);
state->cg_cbari__=(double*)(bytes+5880912);
state->cg_cbarl__=(double*)(bytes+5880944);
state->cg_ch4vmr__=*(double*)(bytes+5880976);
state->cg_cn_ah2ow__=(double*)(bytes+5880984);
state->cg_cn_eh2ow__=(double*)(bytes+8820984);
state->cg_co2__=(double*)(bytes+11760984);
state->cg_co2mmr__=*(double*)(bytes+11762848);
state->cg_co2vmr__=*(double*)(bytes+11762856);
state->cg_coefh__=(double*)(bytes+11762864);
state->cg_coefj__=(double*)(bytes+11762928);
state->cg_coefk__=(double*)(bytes+11762976);
state->cg_cpair__=*(double*)(bytes+11763024);
state->cg_cplol__=*(double*)(bytes+11763032);
state->cg_cplos__=*(double*)(bytes+11763040);
state->cg_dayspy__=*(double*)(bytes+11763048);
state->cg_dbari__=(double*)(bytes+11763056);
state->cg_dbarl__=(double*)(bytes+11763088);
state->cg_delta__=*(double*)(bytes+11763120);
state->cg_dlp_h2o__=*(double*)(bytes+11763128);
state->cg_dlu_h2o__=*(double*)(bytes+11763136);
state->cg_dpfco2__=*(double*)(bytes+11763144);
state->cg_dpfo3__=*(double*)(bytes+11763152);
state->cg_drh_h2o__=*(double*)(bytes+11763160);
state->cg_dte_h2o__=*(double*)(bytes+11763168);
state->cg_dtp_h2o__=*(double*)(bytes+11763176);
state->cg_dustscl__=*(double*)(bytes+11763184);
state->cg_dustscl_rf__=*(double*)(bytes+11763192);
state->cg_ebari__=(double*)(bytes+11763200);
state->cg_ebarl__=(double*)(bytes+11763232);
state->cg_eh2onw__=(double*)(bytes+11763264);
state->cg_epsilo__=*(double*)(bytes+14703264);
state->cg_epsqs__=*(double*)(bytes+14703272);
state->cg_estbl__=(double*)(bytes+14703280);
state->cg_estblh2o__=(double*)(bytes+14705280);
state->cg_f11vmr__=*(double*)(bytes+14706824);
state->cg_f12vmr__=*(double*)(bytes+14706832);
state->cg_fat__=(double*)(bytes+14706840);
state->cg_fbari__=(double*)(bytes+14706936);
state->cg_fbarl__=(double*)(bytes+14706968);
state->cg_fc1__=*(double*)(bytes+14707000);
state->cg_fet__=(double*)(bytes+14707008);
state->cg_frcsol__=(double*)(bytes+14707104);
state->cg_fwc1__=*(double*)(bytes+14707256);
state->cg_fwc2__=*(double*)(bytes+14707264);
state->cg_fwcoef__=*(double*)(bytes+14707272);
state->cg_gbg__=(double*)(bytes+14707280);
state->cg_gcb__=(double*)(bytes+14707432);
state->cg_gcphil__=(double*)(bytes+14707584);
state->cg_gcphob__=(double*)(bytes+14859584);
state->cg_gdst__=(double*)(bytes+14859736);
state->cg_gravit__=*(double*)(bytes+14860344);
state->cg_gravmks__=*(double*)(bytes+14860352);
state->cg_gsslt__=(double*)(bytes+14860360);
state->cg_gsul__=(double*)(bytes+15012360);
state->cg_gvolc__=(double*)(bytes+15164360);
state->cg_inf__=*(double*)(bytes+15164512);
state->cg_kbg__=(double*)(bytes+15164520);
state->cg_kcb__=(double*)(bytes+15164672);
state->cg_kcphil__=(double*)(bytes+15164824);
state->cg_kcphob__=(double*)(bytes+15316824);
state->cg_kdst__=(double*)(bytes+15316976);
state->cg_ksslt__=(double*)(bytes+15317584);
state->cg_ksul__=(double*)(bytes+15469584);
state->cg_kvolc__=(double*)(bytes+15621584);
state->cg_latice__=*(double*)(bytes+15621736);
state->cg_latvap__=*(double*)(bytes+15621744);
state->cg_ln_ah2ow__=(double*)(bytes+15621752);
state->cg_ln_eh2ow__=(double*)(bytes+18561752);
state->cg_max_lp_h2o__=*(double*)(bytes+21501752);
state->cg_max_lu_h2o__=*(double*)(bytes+21501760);
state->cg_max_rh_h2o__=*(double*)(bytes+21501768);
state->cg_max_te_h2o__=*(double*)(bytes+21501776);
state->cg_max_tp_h2o__=*(double*)(bytes+21501784);
state->cg_min_lp_h2o__=*(double*)(bytes+21501792);
state->cg_min_lu_h2o__=*(double*)(bytes+21501800);
state->cg_min_p_h2o__=*(double*)(bytes+21501808);
state->cg_min_rh_h2o__=*(double*)(bytes+21501816);
state->cg_min_te_h2o__=*(double*)(bytes+21501824);
state->cg_min_tp_h2o__=*(double*)(bytes+21501832);
state->cg_min_u_h2o__=*(double*)(bytes+21501840);
state->cg_mwch4__=*(double*)(bytes+21501848);
state->cg_mwco2__=*(double*)(bytes+21501856);
state->cg_mwdry__=*(double*)(bytes+21501864);
state->cg_mwf11__=*(double*)(bytes+21501872);
state->cg_mwf12__=*(double*)(bytes+21501880);
state->cg_mwh2o__=*(double*)(bytes+21501888);
state->cg_mwn2o__=*(double*)(bytes+21501896);
state->cg_n2ovmr__=*(double*)(bytes+21501904);
state->cg_nirwgt__=(double*)(bytes+21501912);
state->cg_o2mmr__=*(double*)(bytes+21502064);
state->cg_p0__=*(double*)(bytes+21502072);
state->cg_pcf__=(double*)(bytes+21502080);
state->cg_pco2__=(double*)(bytes+21502128);
state->cg_ph2o__=(double*)(bytes+21502280);
state->cg_pie__=*(double*)(bytes+21502432);
state->cg_po2__=(double*)(bytes+21502440);
state->cg_r_universal__=*(double*)(bytes+21502592);
state->cg_rair__=*(double*)(bytes+21502600);
state->cg_raytau__=(double*)(bytes+21502608);
state->cg_retab__=(double*)(bytes+21502760);
state->cg_rga__=*(double*)(bytes+21503520);
state->cg_rgsslp__=*(double*)(bytes+21503528);
state->cg_scon__=*(double*)(bytes+21503536);
state->cg_sslp__=*(double*)(bytes+21503544);
state->cg_ssltscl__=*(double*)(bytes+21503552);
state->cg_ssltscl_rf__=*(double*)(bytes+21503560);
state->cg_stebol__=*(double*)(bytes+21503568);
state->cg_sulscl__=*(double*)(bytes+21503576);
state->cg_sulscl_rf__=*(double*)(bytes+21503584);
state->cg_tauback__=*(double*)(bytes+21503592);
state->cg_tmax__=*(double*)(bytes+21503600);
state->cg_tmelt__=*(double*)(bytes+21503608);
state->cg_tmin__=*(double*)(bytes+21503616);
state->cg_v_abo3_35__=*(double*)(bytes+21503624);
state->cg_v_abo3_64__=*(double*)(bytes+21503632);
state->cg_v_raytau_35__=*(double*)(bytes+21503640);
state->cg_v_raytau_64__=*(double*)(bytes+21503648);
state->cg_volcscl__=*(double*)(bytes+21503656);
state->cg_volcscl_rf__=*(double*)(bytes+21503664);
state->cg_wavmax__=(double*)(bytes+21503672);
state->cg_wavmin__=(double*)(bytes+21503824);
state->cg_wbg__=(double*)(bytes+21503976);
state->cg_wcb__=(double*)(bytes+21504128);
state->cg_wcphil__=(double*)(bytes+21504280);
state->cg_wcphob__=(double*)(bytes+21656280);
state->cg_wdst__=(double*)(bytes+21656432);
state->cg_wsslt__=(double*)(bytes+21657040);
state->cg_wsul__=(double*)(bytes+21809040);
state->cg_wvolc__=(double*)(bytes+21961040);
state->cg_zvir__=*(double*)(bytes+21961192);
state->cg_have_ozone__=*(int*)(bytes+21961200);
state->cg_idxbcphi__=*(int*)(bytes+21961204);
state->cg_idxbcpho__=*(int*)(bytes+21961208);
state->cg_idxbg__=*(int*)(bytes+21961212);
state->cg_idxcarbonfirst__=*(int*)(bytes+21961216);
state->cg_idxdustfirst__=*(int*)(bytes+21961220);
state->cg_idxocphi__=*(int*)(bytes+21961224);
state->cg_idxocpho__=*(int*)(bytes+21961228);
state->cg_idxsslt__=*(int*)(bytes+21961232);
state->cg_idxsul__=*(int*)(bytes+21961236);
state->cg_idxvolc__=*(int*)(bytes+21961240);
state->cg_indirect__=*(int*)(bytes+21961244);
state->cg_ixcldice__=*(int*)(bytes+21961248);
state->cg_ixcldliq__=*(int*)(bytes+21961252);
state->cg_levsiz_ozone_save__=*(int*)(bytes+21961256);
state->cg_masterproc__=*(int*)(bytes+21961260);
state->cg_mxaerl__=*(int*)(bytes+21961264);
state->cg_ntoplw__=*(int*)(bytes+21961268);
state->cg_ozncyc__=*(int*)(bytes+21961272);
state->cg_radforce__=*(int*)(bytes+21961276);
state->cg_strat_volcanic__=*(int*)(bytes+21961280);
state->cg_trace_gas__=*(int*)(bytes+21961284);
state->cg_yrdata__=(int*)(bytes+21961288);
state->scratch=workspace+column*CAM_WORK_BYTES+2048; state->sw_flux_stride=min(32,columns-(column/32)*32); state->sw_flux_offset=0; state->sw_flux=(double*)(workspace+columns*CAM_WORK_BYTES)+(column/32)*32*CAM_SW_FLUX_VALUES+(column%32); state->scratch_offset=0; state->abs_phase=0; state->abs_lane=0; state->abs_stride=1; state->abs_padding=0;
radems_(state,&lchnk,&ncol,&pcols,&pver,&pverp,a0+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a1+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a2+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a3+column*((1-(1)+1)),a4+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a5+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a6+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a7+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a8+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a9+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a10+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a11+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a12+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a13+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a14+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a15+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a16+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a17+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a18+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a19+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a20+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a21+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a22+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a23+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a24+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a25+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a26+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a27+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a28+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a29+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a30+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a31+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a32+column*((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a33+column*((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a34+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a35+column*((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a36+column*((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a37+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1)));
}

extern "C" __global__ __launch_bounds__(32) void cam_lw(int columns,const unsigned char* bytes,unsigned char* workspace,double* a0,double* a1,double* a2,double* a3,double* a4,double* a5,double* a6,double* a7,double* a8,double* a9,double* a10,double* a11,double* a12,double* a13,double* a14,int* a15,double* a16,double* a17,int* a18,double* a19,double* a20,double* a21,double* a22,double* a23,double* a24,double* a25,double* a26,double* a27,double* a28,double* a29,double* a30,double* a31,double* a32,double* a33){
int column=blockIdx.x*blockDim.x+threadIdx.x;
if(column>=columns)return;
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
CamState* state=(CamState*)(workspace+CAM_WORK_BYTES*column);
state->cg_abari__=(double*)(bytes+0);
state->cg_abarl__=(double*)(bytes+32);
state->cg_abco2__=(double*)(bytes+64);
state->cg_abh2o__=(double*)(bytes+216);
state->cg_abo2__=(double*)(bytes+368);
state->cg_abo3__=(double*)(bytes+520);
state->cg_abs_cff_mss_aer__=(double*)(bytes+672);
state->cg_ah2onw__=(double*)(bytes+728);
state->cg_ah2ow__=(double*)(bytes+2940728);
state->cg_amco2__=*(double*)(bytes+5880728);
state->cg_amd__=*(double*)(bytes+5880736);
state->cg_amo__=*(double*)(bytes+5880744);
state->cg_bbari__=(double*)(bytes+5880752);
state->cg_bbarl__=(double*)(bytes+5880784);
state->cg_bgscl_rf__=*(double*)(bytes+5880816);
state->cg_c16__=*(double*)(bytes+5880824);
state->cg_c17__=*(double*)(bytes+5880832);
state->cg_c26__=*(double*)(bytes+5880840);
state->cg_c27__=*(double*)(bytes+5880848);
state->cg_c28__=*(double*)(bytes+5880856);
state->cg_c29__=*(double*)(bytes+5880864);
state->cg_c30__=*(double*)(bytes+5880872);
state->cg_c31__=*(double*)(bytes+5880880);
state->cg_cappa__=*(double*)(bytes+5880888);
state->cg_carscl__=*(double*)(bytes+5880896);
state->cg_carscl_rf__=*(double*)(bytes+5880904);
state->cg_cbari__=(double*)(bytes+5880912);
state->cg_cbarl__=(double*)(bytes+5880944);
state->cg_ch4vmr__=*(double*)(bytes+5880976);
state->cg_cn_ah2ow__=(double*)(bytes+5880984);
state->cg_cn_eh2ow__=(double*)(bytes+8820984);
state->cg_co2__=(double*)(bytes+11760984);
state->cg_co2mmr__=*(double*)(bytes+11762848);
state->cg_co2vmr__=*(double*)(bytes+11762856);
state->cg_coefh__=(double*)(bytes+11762864);
state->cg_coefj__=(double*)(bytes+11762928);
state->cg_coefk__=(double*)(bytes+11762976);
state->cg_cpair__=*(double*)(bytes+11763024);
state->cg_cplol__=*(double*)(bytes+11763032);
state->cg_cplos__=*(double*)(bytes+11763040);
state->cg_dayspy__=*(double*)(bytes+11763048);
state->cg_dbari__=(double*)(bytes+11763056);
state->cg_dbarl__=(double*)(bytes+11763088);
state->cg_delta__=*(double*)(bytes+11763120);
state->cg_dlp_h2o__=*(double*)(bytes+11763128);
state->cg_dlu_h2o__=*(double*)(bytes+11763136);
state->cg_dpfco2__=*(double*)(bytes+11763144);
state->cg_dpfo3__=*(double*)(bytes+11763152);
state->cg_drh_h2o__=*(double*)(bytes+11763160);
state->cg_dte_h2o__=*(double*)(bytes+11763168);
state->cg_dtp_h2o__=*(double*)(bytes+11763176);
state->cg_dustscl__=*(double*)(bytes+11763184);
state->cg_dustscl_rf__=*(double*)(bytes+11763192);
state->cg_ebari__=(double*)(bytes+11763200);
state->cg_ebarl__=(double*)(bytes+11763232);
state->cg_eh2onw__=(double*)(bytes+11763264);
state->cg_epsilo__=*(double*)(bytes+14703264);
state->cg_epsqs__=*(double*)(bytes+14703272);
state->cg_estbl__=(double*)(bytes+14703280);
state->cg_estblh2o__=(double*)(bytes+14705280);
state->cg_f11vmr__=*(double*)(bytes+14706824);
state->cg_f12vmr__=*(double*)(bytes+14706832);
state->cg_fat__=(double*)(bytes+14706840);
state->cg_fbari__=(double*)(bytes+14706936);
state->cg_fbarl__=(double*)(bytes+14706968);
state->cg_fc1__=*(double*)(bytes+14707000);
state->cg_fet__=(double*)(bytes+14707008);
state->cg_frcsol__=(double*)(bytes+14707104);
state->cg_fwc1__=*(double*)(bytes+14707256);
state->cg_fwc2__=*(double*)(bytes+14707264);
state->cg_fwcoef__=*(double*)(bytes+14707272);
state->cg_gbg__=(double*)(bytes+14707280);
state->cg_gcb__=(double*)(bytes+14707432);
state->cg_gcphil__=(double*)(bytes+14707584);
state->cg_gcphob__=(double*)(bytes+14859584);
state->cg_gdst__=(double*)(bytes+14859736);
state->cg_gravit__=*(double*)(bytes+14860344);
state->cg_gravmks__=*(double*)(bytes+14860352);
state->cg_gsslt__=(double*)(bytes+14860360);
state->cg_gsul__=(double*)(bytes+15012360);
state->cg_gvolc__=(double*)(bytes+15164360);
state->cg_inf__=*(double*)(bytes+15164512);
state->cg_kbg__=(double*)(bytes+15164520);
state->cg_kcb__=(double*)(bytes+15164672);
state->cg_kcphil__=(double*)(bytes+15164824);
state->cg_kcphob__=(double*)(bytes+15316824);
state->cg_kdst__=(double*)(bytes+15316976);
state->cg_ksslt__=(double*)(bytes+15317584);
state->cg_ksul__=(double*)(bytes+15469584);
state->cg_kvolc__=(double*)(bytes+15621584);
state->cg_latice__=*(double*)(bytes+15621736);
state->cg_latvap__=*(double*)(bytes+15621744);
state->cg_ln_ah2ow__=(double*)(bytes+15621752);
state->cg_ln_eh2ow__=(double*)(bytes+18561752);
state->cg_max_lp_h2o__=*(double*)(bytes+21501752);
state->cg_max_lu_h2o__=*(double*)(bytes+21501760);
state->cg_max_rh_h2o__=*(double*)(bytes+21501768);
state->cg_max_te_h2o__=*(double*)(bytes+21501776);
state->cg_max_tp_h2o__=*(double*)(bytes+21501784);
state->cg_min_lp_h2o__=*(double*)(bytes+21501792);
state->cg_min_lu_h2o__=*(double*)(bytes+21501800);
state->cg_min_p_h2o__=*(double*)(bytes+21501808);
state->cg_min_rh_h2o__=*(double*)(bytes+21501816);
state->cg_min_te_h2o__=*(double*)(bytes+21501824);
state->cg_min_tp_h2o__=*(double*)(bytes+21501832);
state->cg_min_u_h2o__=*(double*)(bytes+21501840);
state->cg_mwch4__=*(double*)(bytes+21501848);
state->cg_mwco2__=*(double*)(bytes+21501856);
state->cg_mwdry__=*(double*)(bytes+21501864);
state->cg_mwf11__=*(double*)(bytes+21501872);
state->cg_mwf12__=*(double*)(bytes+21501880);
state->cg_mwh2o__=*(double*)(bytes+21501888);
state->cg_mwn2o__=*(double*)(bytes+21501896);
state->cg_n2ovmr__=*(double*)(bytes+21501904);
state->cg_nirwgt__=(double*)(bytes+21501912);
state->cg_o2mmr__=*(double*)(bytes+21502064);
state->cg_p0__=*(double*)(bytes+21502072);
state->cg_pcf__=(double*)(bytes+21502080);
state->cg_pco2__=(double*)(bytes+21502128);
state->cg_ph2o__=(double*)(bytes+21502280);
state->cg_pie__=*(double*)(bytes+21502432);
state->cg_po2__=(double*)(bytes+21502440);
state->cg_r_universal__=*(double*)(bytes+21502592);
state->cg_rair__=*(double*)(bytes+21502600);
state->cg_raytau__=(double*)(bytes+21502608);
state->cg_retab__=(double*)(bytes+21502760);
state->cg_rga__=*(double*)(bytes+21503520);
state->cg_rgsslp__=*(double*)(bytes+21503528);
state->cg_scon__=*(double*)(bytes+21503536);
state->cg_sslp__=*(double*)(bytes+21503544);
state->cg_ssltscl__=*(double*)(bytes+21503552);
state->cg_ssltscl_rf__=*(double*)(bytes+21503560);
state->cg_stebol__=*(double*)(bytes+21503568);
state->cg_sulscl__=*(double*)(bytes+21503576);
state->cg_sulscl_rf__=*(double*)(bytes+21503584);
state->cg_tauback__=*(double*)(bytes+21503592);
state->cg_tmax__=*(double*)(bytes+21503600);
state->cg_tmelt__=*(double*)(bytes+21503608);
state->cg_tmin__=*(double*)(bytes+21503616);
state->cg_v_abo3_35__=*(double*)(bytes+21503624);
state->cg_v_abo3_64__=*(double*)(bytes+21503632);
state->cg_v_raytau_35__=*(double*)(bytes+21503640);
state->cg_v_raytau_64__=*(double*)(bytes+21503648);
state->cg_volcscl__=*(double*)(bytes+21503656);
state->cg_volcscl_rf__=*(double*)(bytes+21503664);
state->cg_wavmax__=(double*)(bytes+21503672);
state->cg_wavmin__=(double*)(bytes+21503824);
state->cg_wbg__=(double*)(bytes+21503976);
state->cg_wcb__=(double*)(bytes+21504128);
state->cg_wcphil__=(double*)(bytes+21504280);
state->cg_wcphob__=(double*)(bytes+21656280);
state->cg_wdst__=(double*)(bytes+21656432);
state->cg_wsslt__=(double*)(bytes+21657040);
state->cg_wsul__=(double*)(bytes+21809040);
state->cg_wvolc__=(double*)(bytes+21961040);
state->cg_zvir__=*(double*)(bytes+21961192);
state->cg_have_ozone__=*(int*)(bytes+21961200);
state->cg_idxbcphi__=*(int*)(bytes+21961204);
state->cg_idxbcpho__=*(int*)(bytes+21961208);
state->cg_idxbg__=*(int*)(bytes+21961212);
state->cg_idxcarbonfirst__=*(int*)(bytes+21961216);
state->cg_idxdustfirst__=*(int*)(bytes+21961220);
state->cg_idxocphi__=*(int*)(bytes+21961224);
state->cg_idxocpho__=*(int*)(bytes+21961228);
state->cg_idxsslt__=*(int*)(bytes+21961232);
state->cg_idxsul__=*(int*)(bytes+21961236);
state->cg_idxvolc__=*(int*)(bytes+21961240);
state->cg_indirect__=*(int*)(bytes+21961244);
state->cg_ixcldice__=*(int*)(bytes+21961248);
state->cg_ixcldliq__=*(int*)(bytes+21961252);
state->cg_levsiz_ozone_save__=*(int*)(bytes+21961256);
state->cg_masterproc__=*(int*)(bytes+21961260);
state->cg_mxaerl__=*(int*)(bytes+21961264);
state->cg_ntoplw__=*(int*)(bytes+21961268);
state->cg_ozncyc__=*(int*)(bytes+21961272);
state->cg_radforce__=*(int*)(bytes+21961276);
state->cg_strat_volcanic__=*(int*)(bytes+21961280);
state->cg_trace_gas__=*(int*)(bytes+21961284);
state->cg_yrdata__=(int*)(bytes+21961288);
state->scratch=workspace+column*CAM_WORK_BYTES+2048; state->sw_flux_stride=min(32,columns-(column/32)*32); state->sw_flux_offset=0; state->sw_flux=(double*)(workspace+columns*CAM_WORK_BYTES)+(column/32)*32*CAM_SW_FLUX_VALUES+(column%32); state->scratch_offset=0; state->abs_phase=0; state->abs_lane=0; state->abs_stride=1; state->abs_padding=0;
radclwmx_(state,&lchnk,&ncol,&pcols,&pver,&pverp,a0+column*((1-(1)+1)),a1+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a2+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a3+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a4+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a5+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a6+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a7+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a8+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a9+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a10+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a11+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a12+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a13+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a14+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a15+column*((1-(1)+1)),a16+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a17+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a18+column*(1),a19+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a20+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1)),a21+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a22+column*((1-(1)+1)),a23+column*((1-(1)+1)),a24+column*((1-(1)+1)),a25+column*((1-(1)+1)),a26+column*((1-(1)+1)),a27+column*((1-(1)+1)),a28+column*((1-(1)+1)),a29+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a30+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a31+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a32+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a33+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)));
}

extern "C" __global__ __launch_bounds__(32) void cam_sw(int columns,const unsigned char* bytes,unsigned char* workspace,int* a0,double* a1,double* a2,double* a3,double* a4,double* a5,double* a6,double* a7,double* a8,double* a9,double* a10,double* a11,double* a12,double* a13,double* a14,double* a15,double* a16,double* a17,float* a18,double* a19,double* a20,double* a21,double* a22,int* a23,double* a24,double* a25,double* a26,double* a27,double* a28,double* a29,double* a30,double* a31,double* a32,double* a33,double* a34,double* a35,double* a36,double* a37,double* a38,double* a39,double* a40,double* a41,double* a42,double* a43,double* a44,double* a45,double* a46,double* a47,double* a48,double* a49,double* a50,double* a51,double* a52,double* a53,double* a54,double* a55,double* a56,double* a57,double* a58){
int column=blockIdx.x*blockDim.x+threadIdx.x;
if(column>=columns)return;
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
CamState* state=(CamState*)(workspace+CAM_WORK_BYTES*column);
state->cg_abari__=(double*)(bytes+0);
state->cg_abarl__=(double*)(bytes+32);
state->cg_abco2__=(double*)(bytes+64);
state->cg_abh2o__=(double*)(bytes+216);
state->cg_abo2__=(double*)(bytes+368);
state->cg_abo3__=(double*)(bytes+520);
state->cg_abs_cff_mss_aer__=(double*)(bytes+672);
state->cg_ah2onw__=(double*)(bytes+728);
state->cg_ah2ow__=(double*)(bytes+2940728);
state->cg_amco2__=*(double*)(bytes+5880728);
state->cg_amd__=*(double*)(bytes+5880736);
state->cg_amo__=*(double*)(bytes+5880744);
state->cg_bbari__=(double*)(bytes+5880752);
state->cg_bbarl__=(double*)(bytes+5880784);
state->cg_bgscl_rf__=*(double*)(bytes+5880816);
state->cg_c16__=*(double*)(bytes+5880824);
state->cg_c17__=*(double*)(bytes+5880832);
state->cg_c26__=*(double*)(bytes+5880840);
state->cg_c27__=*(double*)(bytes+5880848);
state->cg_c28__=*(double*)(bytes+5880856);
state->cg_c29__=*(double*)(bytes+5880864);
state->cg_c30__=*(double*)(bytes+5880872);
state->cg_c31__=*(double*)(bytes+5880880);
state->cg_cappa__=*(double*)(bytes+5880888);
state->cg_carscl__=*(double*)(bytes+5880896);
state->cg_carscl_rf__=*(double*)(bytes+5880904);
state->cg_cbari__=(double*)(bytes+5880912);
state->cg_cbarl__=(double*)(bytes+5880944);
state->cg_ch4vmr__=*(double*)(bytes+5880976);
state->cg_cn_ah2ow__=(double*)(bytes+5880984);
state->cg_cn_eh2ow__=(double*)(bytes+8820984);
state->cg_co2__=(double*)(bytes+11760984);
state->cg_co2mmr__=*(double*)(bytes+11762848);
state->cg_co2vmr__=*(double*)(bytes+11762856);
state->cg_coefh__=(double*)(bytes+11762864);
state->cg_coefj__=(double*)(bytes+11762928);
state->cg_coefk__=(double*)(bytes+11762976);
state->cg_cpair__=*(double*)(bytes+11763024);
state->cg_cplol__=*(double*)(bytes+11763032);
state->cg_cplos__=*(double*)(bytes+11763040);
state->cg_dayspy__=*(double*)(bytes+11763048);
state->cg_dbari__=(double*)(bytes+11763056);
state->cg_dbarl__=(double*)(bytes+11763088);
state->cg_delta__=*(double*)(bytes+11763120);
state->cg_dlp_h2o__=*(double*)(bytes+11763128);
state->cg_dlu_h2o__=*(double*)(bytes+11763136);
state->cg_dpfco2__=*(double*)(bytes+11763144);
state->cg_dpfo3__=*(double*)(bytes+11763152);
state->cg_drh_h2o__=*(double*)(bytes+11763160);
state->cg_dte_h2o__=*(double*)(bytes+11763168);
state->cg_dtp_h2o__=*(double*)(bytes+11763176);
state->cg_dustscl__=*(double*)(bytes+11763184);
state->cg_dustscl_rf__=*(double*)(bytes+11763192);
state->cg_ebari__=(double*)(bytes+11763200);
state->cg_ebarl__=(double*)(bytes+11763232);
state->cg_eh2onw__=(double*)(bytes+11763264);
state->cg_epsilo__=*(double*)(bytes+14703264);
state->cg_epsqs__=*(double*)(bytes+14703272);
state->cg_estbl__=(double*)(bytes+14703280);
state->cg_estblh2o__=(double*)(bytes+14705280);
state->cg_f11vmr__=*(double*)(bytes+14706824);
state->cg_f12vmr__=*(double*)(bytes+14706832);
state->cg_fat__=(double*)(bytes+14706840);
state->cg_fbari__=(double*)(bytes+14706936);
state->cg_fbarl__=(double*)(bytes+14706968);
state->cg_fc1__=*(double*)(bytes+14707000);
state->cg_fet__=(double*)(bytes+14707008);
state->cg_frcsol__=(double*)(bytes+14707104);
state->cg_fwc1__=*(double*)(bytes+14707256);
state->cg_fwc2__=*(double*)(bytes+14707264);
state->cg_fwcoef__=*(double*)(bytes+14707272);
state->cg_gbg__=(double*)(bytes+14707280);
state->cg_gcb__=(double*)(bytes+14707432);
state->cg_gcphil__=(double*)(bytes+14707584);
state->cg_gcphob__=(double*)(bytes+14859584);
state->cg_gdst__=(double*)(bytes+14859736);
state->cg_gravit__=*(double*)(bytes+14860344);
state->cg_gravmks__=*(double*)(bytes+14860352);
state->cg_gsslt__=(double*)(bytes+14860360);
state->cg_gsul__=(double*)(bytes+15012360);
state->cg_gvolc__=(double*)(bytes+15164360);
state->cg_inf__=*(double*)(bytes+15164512);
state->cg_kbg__=(double*)(bytes+15164520);
state->cg_kcb__=(double*)(bytes+15164672);
state->cg_kcphil__=(double*)(bytes+15164824);
state->cg_kcphob__=(double*)(bytes+15316824);
state->cg_kdst__=(double*)(bytes+15316976);
state->cg_ksslt__=(double*)(bytes+15317584);
state->cg_ksul__=(double*)(bytes+15469584);
state->cg_kvolc__=(double*)(bytes+15621584);
state->cg_latice__=*(double*)(bytes+15621736);
state->cg_latvap__=*(double*)(bytes+15621744);
state->cg_ln_ah2ow__=(double*)(bytes+15621752);
state->cg_ln_eh2ow__=(double*)(bytes+18561752);
state->cg_max_lp_h2o__=*(double*)(bytes+21501752);
state->cg_max_lu_h2o__=*(double*)(bytes+21501760);
state->cg_max_rh_h2o__=*(double*)(bytes+21501768);
state->cg_max_te_h2o__=*(double*)(bytes+21501776);
state->cg_max_tp_h2o__=*(double*)(bytes+21501784);
state->cg_min_lp_h2o__=*(double*)(bytes+21501792);
state->cg_min_lu_h2o__=*(double*)(bytes+21501800);
state->cg_min_p_h2o__=*(double*)(bytes+21501808);
state->cg_min_rh_h2o__=*(double*)(bytes+21501816);
state->cg_min_te_h2o__=*(double*)(bytes+21501824);
state->cg_min_tp_h2o__=*(double*)(bytes+21501832);
state->cg_min_u_h2o__=*(double*)(bytes+21501840);
state->cg_mwch4__=*(double*)(bytes+21501848);
state->cg_mwco2__=*(double*)(bytes+21501856);
state->cg_mwdry__=*(double*)(bytes+21501864);
state->cg_mwf11__=*(double*)(bytes+21501872);
state->cg_mwf12__=*(double*)(bytes+21501880);
state->cg_mwh2o__=*(double*)(bytes+21501888);
state->cg_mwn2o__=*(double*)(bytes+21501896);
state->cg_n2ovmr__=*(double*)(bytes+21501904);
state->cg_nirwgt__=(double*)(bytes+21501912);
state->cg_o2mmr__=*(double*)(bytes+21502064);
state->cg_p0__=*(double*)(bytes+21502072);
state->cg_pcf__=(double*)(bytes+21502080);
state->cg_pco2__=(double*)(bytes+21502128);
state->cg_ph2o__=(double*)(bytes+21502280);
state->cg_pie__=*(double*)(bytes+21502432);
state->cg_po2__=(double*)(bytes+21502440);
state->cg_r_universal__=*(double*)(bytes+21502592);
state->cg_rair__=*(double*)(bytes+21502600);
state->cg_raytau__=(double*)(bytes+21502608);
state->cg_retab__=(double*)(bytes+21502760);
state->cg_rga__=*(double*)(bytes+21503520);
state->cg_rgsslp__=*(double*)(bytes+21503528);
state->cg_scon__=*(double*)(bytes+21503536);
state->cg_sslp__=*(double*)(bytes+21503544);
state->cg_ssltscl__=*(double*)(bytes+21503552);
state->cg_ssltscl_rf__=*(double*)(bytes+21503560);
state->cg_stebol__=*(double*)(bytes+21503568);
state->cg_sulscl__=*(double*)(bytes+21503576);
state->cg_sulscl_rf__=*(double*)(bytes+21503584);
state->cg_tauback__=*(double*)(bytes+21503592);
state->cg_tmax__=*(double*)(bytes+21503600);
state->cg_tmelt__=*(double*)(bytes+21503608);
state->cg_tmin__=*(double*)(bytes+21503616);
state->cg_v_abo3_35__=*(double*)(bytes+21503624);
state->cg_v_abo3_64__=*(double*)(bytes+21503632);
state->cg_v_raytau_35__=*(double*)(bytes+21503640);
state->cg_v_raytau_64__=*(double*)(bytes+21503648);
state->cg_volcscl__=*(double*)(bytes+21503656);
state->cg_volcscl_rf__=*(double*)(bytes+21503664);
state->cg_wavmax__=(double*)(bytes+21503672);
state->cg_wavmin__=(double*)(bytes+21503824);
state->cg_wbg__=(double*)(bytes+21503976);
state->cg_wcb__=(double*)(bytes+21504128);
state->cg_wcphil__=(double*)(bytes+21504280);
state->cg_wcphob__=(double*)(bytes+21656280);
state->cg_wdst__=(double*)(bytes+21656432);
state->cg_wsslt__=(double*)(bytes+21657040);
state->cg_wsul__=(double*)(bytes+21809040);
state->cg_wvolc__=(double*)(bytes+21961040);
state->cg_zvir__=*(double*)(bytes+21961192);
state->cg_have_ozone__=*(int*)(bytes+21961200);
state->cg_idxbcphi__=*(int*)(bytes+21961204);
state->cg_idxbcpho__=*(int*)(bytes+21961208);
state->cg_idxbg__=*(int*)(bytes+21961212);
state->cg_idxcarbonfirst__=*(int*)(bytes+21961216);
state->cg_idxdustfirst__=*(int*)(bytes+21961220);
state->cg_idxocphi__=*(int*)(bytes+21961224);
state->cg_idxocpho__=*(int*)(bytes+21961228);
state->cg_idxsslt__=*(int*)(bytes+21961232);
state->cg_idxsul__=*(int*)(bytes+21961236);
state->cg_idxvolc__=*(int*)(bytes+21961240);
state->cg_indirect__=*(int*)(bytes+21961244);
state->cg_ixcldice__=*(int*)(bytes+21961248);
state->cg_ixcldliq__=*(int*)(bytes+21961252);
state->cg_levsiz_ozone_save__=*(int*)(bytes+21961256);
state->cg_masterproc__=*(int*)(bytes+21961260);
state->cg_mxaerl__=*(int*)(bytes+21961264);
state->cg_ntoplw__=*(int*)(bytes+21961268);
state->cg_ozncyc__=*(int*)(bytes+21961272);
state->cg_radforce__=*(int*)(bytes+21961276);
state->cg_strat_volcanic__=*(int*)(bytes+21961280);
state->cg_trace_gas__=*(int*)(bytes+21961284);
state->cg_yrdata__=(int*)(bytes+21961288);
state->scratch=workspace+column*CAM_WORK_BYTES+2048; state->sw_flux_stride=min(32,columns-(column/32)*32); state->sw_flux_offset=0; state->sw_flux=(double*)(workspace+columns*CAM_WORK_BYTES)+(column/32)*32*CAM_SW_FLUX_VALUES+(column%32); state->scratch_offset=0; state->abs_phase=0; state->abs_lane=0; state->abs_stride=1; state->abs_padding=0;
radcswmx_(state,a0+column*(1),&lchnk,&ncol,&pcols,&pver,&pverp,a1+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a2+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a3+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a4+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a5+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a6+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1)),a7+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a8+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a9+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a10+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a11+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a12+column*((1-(1)+1)*(CAM_LEVELS-(0)+1)),a13+column*((1-(1)+1)*(CAM_LEVELS-(0)+1)),a14+column*(1),a15+column*((1-(1)+1)),a16+column*(1),a17+column*((1-(1)+1)),a18+column*(1),a19+column*((1-(1)+1)),a20+column*((1-(1)+1)),a21+column*((1-(1)+1)),a22+column*((1-(1)+1)),a23+column*((1-(1)+1)),a24+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a25+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a26+column*((1-(1)+1)*(CAM_LEVELS-(1)+1)),a27+column*((1-(1)+1)),a28+column*((1-(1)+1)),a29+column*((1-(1)+1)),a30+column*((1-(1)+1)),a31+column*((1-(1)+1)),a32+column*((1-(1)+1)),a33+column*((1-(1)+1)),a34+column*((1-(1)+1)),a35+column*((1-(1)+1)),a36+column*((1-(1)+1)),a37+column*((1-(1)+1)),a38+column*((1-(1)+1)),a39+column*((1-(1)+1)),a40+column*((1-(1)+1)),a41+column*((1-(1)+1)),a42+column*((1-(1)+1)),a43+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a44+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a45+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a46+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a47+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a48+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a49+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a50+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)),a51+column*((1-(1)+1)),a52+column*((1-(1)+1)),a53+column*((1-(1)+1)),a54+column*((1-(1)+1)),a55+column*((1-(1)+1)*(19-(1)+1)*(7-(1)+1)),a56+column*((1-(1)+1)*(19-(1)+1)*(7-(1)+1)),a57+column*((1-(1)+1)*(19-(1)+1)*(7-(1)+1)),a58+column*((1-(1)+1)*(19-(1)+1)*(7-(1)+1)));
}


#define CAM_ABS_VALUES ((CAM_LEVELS+1)*(CAM_LEVELS+1)+CAM_LEVELS*4+CAM_LEVELS+1)
#define CAM_ABS_WORK_BYTES (0ULL+((2048+((64+7)&~7))))
extern "C" __global__ __launch_bounds__(32) void cam_driver(int columns,const unsigned char* bytes,unsigned char* workspace,float* a0,float* a1,float* a2,float* a3,int* a4,int* a5,float* a6,float* a7,float* a8,float* a9,float* a10,float* a11,float* a12,float* a13,float* a14,float* a15,float* a16,float* a17,float* a18,float* a19,float* a20,float* a21,float* a22,float* a23,float* a24,float* a25,float* a26,float* a27,float* a28,float* a29,float* a30,float* a31,float* a32,float* a33,float* a34,float* a35,float* a36,float* a37,float* a38,float* a39,float* a40,float* a41,float* a42,float* a43,float* a44,float* a45,float* a46,float* a47,float* a48,float* a49,float* a50,int* a51,float* a52,float* a53,float* a54,int* a55,int* a56,int* a57,int* a58,int* a59,int* a60,float* a61,float* a62,float* a63,float* a64,float* a65,float* a66,float* a67,float* a68,float* a69,float* a70,float* a71,float* a72,float* a73,float* a74,int* a75,int* a76,float* a77,float* a78,float* a79,float* a80,float* a81,int* a82,int* a83,int* a84,int* a85,float* a86,int* a87,float* a88,int* a89,float* a90,float* a91,float* a92,float* a93,float* a94,float* a95,int* a96,float* a97,float* a98,float* a99,int* a100,int* a101,int* a102,int* a103,int* a104,int* a105,int* a106,int* a107,int* a108,int* a109,int* a110,int* a111,int* a112,int* a113,int* a114,int* a115,int* a116,int* a117,int* a118,int* a119,float* a120,double* gases){
int worker=blockIdx.x*blockDim.x+threadIdx.x;
int workers=gridDim.x*blockDim.x;
for(int column=worker;column<columns;column+=workers){
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
CamState* state=(CamState*)(workspace+CAM_WORK_BYTES*worker);
state->cg_abari__=(double*)(bytes+0);
state->cg_abarl__=(double*)(bytes+32);
state->cg_abco2__=(double*)(bytes+64);
state->cg_abh2o__=(double*)(bytes+216);
state->cg_abo2__=(double*)(bytes+368);
state->cg_abo3__=(double*)(bytes+520);
state->cg_abs_cff_mss_aer__=(double*)(bytes+672);
state->cg_ah2onw__=(double*)(bytes+728);
state->cg_ah2ow__=(double*)(bytes+2940728);
state->cg_amco2__=*(double*)(bytes+5880728);
state->cg_amd__=*(double*)(bytes+5880736);
state->cg_amo__=*(double*)(bytes+5880744);
state->cg_bbari__=(double*)(bytes+5880752);
state->cg_bbarl__=(double*)(bytes+5880784);
state->cg_bgscl_rf__=*(double*)(bytes+5880816);
state->cg_c16__=*(double*)(bytes+5880824);
state->cg_c17__=*(double*)(bytes+5880832);
state->cg_c26__=*(double*)(bytes+5880840);
state->cg_c27__=*(double*)(bytes+5880848);
state->cg_c28__=*(double*)(bytes+5880856);
state->cg_c29__=*(double*)(bytes+5880864);
state->cg_c30__=*(double*)(bytes+5880872);
state->cg_c31__=*(double*)(bytes+5880880);
state->cg_cappa__=*(double*)(bytes+5880888);
state->cg_carscl__=*(double*)(bytes+5880896);
state->cg_carscl_rf__=*(double*)(bytes+5880904);
state->cg_cbari__=(double*)(bytes+5880912);
state->cg_cbarl__=(double*)(bytes+5880944);
state->cg_ch4vmr__=*(double*)(bytes+5880976);
state->cg_cn_ah2ow__=(double*)(bytes+5880984);
state->cg_cn_eh2ow__=(double*)(bytes+8820984);
state->cg_co2__=(double*)(bytes+11760984);
state->cg_co2mmr__=*(double*)(bytes+11762848);
state->cg_co2vmr__=*(double*)(bytes+11762856);
state->cg_coefh__=(double*)(bytes+11762864);
state->cg_coefj__=(double*)(bytes+11762928);
state->cg_coefk__=(double*)(bytes+11762976);
state->cg_cpair__=*(double*)(bytes+11763024);
state->cg_cplol__=*(double*)(bytes+11763032);
state->cg_cplos__=*(double*)(bytes+11763040);
state->cg_dayspy__=*(double*)(bytes+11763048);
state->cg_dbari__=(double*)(bytes+11763056);
state->cg_dbarl__=(double*)(bytes+11763088);
state->cg_delta__=*(double*)(bytes+11763120);
state->cg_dlp_h2o__=*(double*)(bytes+11763128);
state->cg_dlu_h2o__=*(double*)(bytes+11763136);
state->cg_dpfco2__=*(double*)(bytes+11763144);
state->cg_dpfo3__=*(double*)(bytes+11763152);
state->cg_drh_h2o__=*(double*)(bytes+11763160);
state->cg_dte_h2o__=*(double*)(bytes+11763168);
state->cg_dtp_h2o__=*(double*)(bytes+11763176);
state->cg_dustscl__=*(double*)(bytes+11763184);
state->cg_dustscl_rf__=*(double*)(bytes+11763192);
state->cg_ebari__=(double*)(bytes+11763200);
state->cg_ebarl__=(double*)(bytes+11763232);
state->cg_eh2onw__=(double*)(bytes+11763264);
state->cg_epsilo__=*(double*)(bytes+14703264);
state->cg_epsqs__=*(double*)(bytes+14703272);
state->cg_estbl__=(double*)(bytes+14703280);
state->cg_estblh2o__=(double*)(bytes+14705280);
state->cg_f11vmr__=*(double*)(bytes+14706824);
state->cg_f12vmr__=*(double*)(bytes+14706832);
state->cg_fat__=(double*)(bytes+14706840);
state->cg_fbari__=(double*)(bytes+14706936);
state->cg_fbarl__=(double*)(bytes+14706968);
state->cg_fc1__=*(double*)(bytes+14707000);
state->cg_fet__=(double*)(bytes+14707008);
state->cg_frcsol__=(double*)(bytes+14707104);
state->cg_fwc1__=*(double*)(bytes+14707256);
state->cg_fwc2__=*(double*)(bytes+14707264);
state->cg_fwcoef__=*(double*)(bytes+14707272);
state->cg_gbg__=(double*)(bytes+14707280);
state->cg_gcb__=(double*)(bytes+14707432);
state->cg_gcphil__=(double*)(bytes+14707584);
state->cg_gcphob__=(double*)(bytes+14859584);
state->cg_gdst__=(double*)(bytes+14859736);
state->cg_gravit__=*(double*)(bytes+14860344);
state->cg_gravmks__=*(double*)(bytes+14860352);
state->cg_gsslt__=(double*)(bytes+14860360);
state->cg_gsul__=(double*)(bytes+15012360);
state->cg_gvolc__=(double*)(bytes+15164360);
state->cg_inf__=*(double*)(bytes+15164512);
state->cg_kbg__=(double*)(bytes+15164520);
state->cg_kcb__=(double*)(bytes+15164672);
state->cg_kcphil__=(double*)(bytes+15164824);
state->cg_kcphob__=(double*)(bytes+15316824);
state->cg_kdst__=(double*)(bytes+15316976);
state->cg_ksslt__=(double*)(bytes+15317584);
state->cg_ksul__=(double*)(bytes+15469584);
state->cg_kvolc__=(double*)(bytes+15621584);
state->cg_latice__=*(double*)(bytes+15621736);
state->cg_latvap__=*(double*)(bytes+15621744);
state->cg_ln_ah2ow__=(double*)(bytes+15621752);
state->cg_ln_eh2ow__=(double*)(bytes+18561752);
state->cg_max_lp_h2o__=*(double*)(bytes+21501752);
state->cg_max_lu_h2o__=*(double*)(bytes+21501760);
state->cg_max_rh_h2o__=*(double*)(bytes+21501768);
state->cg_max_te_h2o__=*(double*)(bytes+21501776);
state->cg_max_tp_h2o__=*(double*)(bytes+21501784);
state->cg_min_lp_h2o__=*(double*)(bytes+21501792);
state->cg_min_lu_h2o__=*(double*)(bytes+21501800);
state->cg_min_p_h2o__=*(double*)(bytes+21501808);
state->cg_min_rh_h2o__=*(double*)(bytes+21501816);
state->cg_min_te_h2o__=*(double*)(bytes+21501824);
state->cg_min_tp_h2o__=*(double*)(bytes+21501832);
state->cg_min_u_h2o__=*(double*)(bytes+21501840);
state->cg_mwch4__=*(double*)(bytes+21501848);
state->cg_mwco2__=*(double*)(bytes+21501856);
state->cg_mwdry__=*(double*)(bytes+21501864);
state->cg_mwf11__=*(double*)(bytes+21501872);
state->cg_mwf12__=*(double*)(bytes+21501880);
state->cg_mwh2o__=*(double*)(bytes+21501888);
state->cg_mwn2o__=*(double*)(bytes+21501896);
state->cg_n2ovmr__=*(double*)(bytes+21501904);
state->cg_nirwgt__=(double*)(bytes+21501912);
state->cg_o2mmr__=*(double*)(bytes+21502064);
state->cg_p0__=*(double*)(bytes+21502072);
state->cg_pcf__=(double*)(bytes+21502080);
state->cg_pco2__=(double*)(bytes+21502128);
state->cg_ph2o__=(double*)(bytes+21502280);
state->cg_pie__=*(double*)(bytes+21502432);
state->cg_po2__=(double*)(bytes+21502440);
state->cg_r_universal__=*(double*)(bytes+21502592);
state->cg_rair__=*(double*)(bytes+21502600);
state->cg_raytau__=(double*)(bytes+21502608);
state->cg_retab__=(double*)(bytes+21502760);
state->cg_rga__=*(double*)(bytes+21503520);
state->cg_rgsslp__=*(double*)(bytes+21503528);
state->cg_scon__=*(double*)(bytes+21503536);
state->cg_sslp__=*(double*)(bytes+21503544);
state->cg_ssltscl__=*(double*)(bytes+21503552);
state->cg_ssltscl_rf__=*(double*)(bytes+21503560);
state->cg_stebol__=*(double*)(bytes+21503568);
state->cg_sulscl__=*(double*)(bytes+21503576);
state->cg_sulscl_rf__=*(double*)(bytes+21503584);
state->cg_tauback__=*(double*)(bytes+21503592);
state->cg_tmax__=*(double*)(bytes+21503600);
state->cg_tmelt__=*(double*)(bytes+21503608);
state->cg_tmin__=*(double*)(bytes+21503616);
state->cg_v_abo3_35__=*(double*)(bytes+21503624);
state->cg_v_abo3_64__=*(double*)(bytes+21503632);
state->cg_v_raytau_35__=*(double*)(bytes+21503640);
state->cg_v_raytau_64__=*(double*)(bytes+21503648);
state->cg_volcscl__=*(double*)(bytes+21503656);
state->cg_volcscl_rf__=*(double*)(bytes+21503664);
state->cg_wavmax__=(double*)(bytes+21503672);
state->cg_wavmin__=(double*)(bytes+21503824);
state->cg_wbg__=(double*)(bytes+21503976);
state->cg_wcb__=(double*)(bytes+21504128);
state->cg_wcphil__=(double*)(bytes+21504280);
state->cg_wcphob__=(double*)(bytes+21656280);
state->cg_wdst__=(double*)(bytes+21656432);
state->cg_wsslt__=(double*)(bytes+21657040);
state->cg_wsul__=(double*)(bytes+21809040);
state->cg_wvolc__=(double*)(bytes+21961040);
state->cg_zvir__=*(double*)(bytes+21961192);
state->cg_have_ozone__=*(int*)(bytes+21961200);
state->cg_idxbcphi__=*(int*)(bytes+21961204);
state->cg_idxbcpho__=*(int*)(bytes+21961208);
state->cg_idxbg__=*(int*)(bytes+21961212);
state->cg_idxcarbonfirst__=*(int*)(bytes+21961216);
state->cg_idxdustfirst__=*(int*)(bytes+21961220);
state->cg_idxocphi__=*(int*)(bytes+21961224);
state->cg_idxocpho__=*(int*)(bytes+21961228);
state->cg_idxsslt__=*(int*)(bytes+21961232);
state->cg_idxsul__=*(int*)(bytes+21961236);
state->cg_idxvolc__=*(int*)(bytes+21961240);
state->cg_indirect__=*(int*)(bytes+21961244);
state->cg_ixcldice__=*(int*)(bytes+21961248);
state->cg_ixcldliq__=*(int*)(bytes+21961252);
state->cg_levsiz_ozone_save__=*(int*)(bytes+21961256);
state->cg_masterproc__=*(int*)(bytes+21961260);
state->cg_mxaerl__=*(int*)(bytes+21961264);
state->cg_ntoplw__=*(int*)(bytes+21961268);
state->cg_ozncyc__=*(int*)(bytes+21961272);
state->cg_radforce__=*(int*)(bytes+21961276);
state->cg_strat_volcanic__=*(int*)(bytes+21961280);
state->cg_trace_gas__=*(int*)(bytes+21961284);
state->cg_yrdata__=(int*)(bytes+21961288);
state->driver_gases[0]=gases[column*5+0];
state->driver_gases[1]=gases[column*5+1];
state->driver_gases[2]=gases[column*5+2];
state->driver_gases[3]=gases[column*5+3];
state->driver_gases[4]=gases[column*5+4];

state->scratch=workspace+worker*CAM_WORK_BYTES+2048; state->sw_flux_stride=min(32,workers-(worker/32)*32); state->sw_flux_offset=0; state->sw_flux=(double*)(workspace+workers*CAM_WORK_BYTES)+(worker/32)*32*CAM_SW_FLUX_VALUES+(worker%32); state->scratch_offset=0; state->abs_phase=0; state->abs_lane=0; state->abs_stride=1; state->abs_padding=0;
camrad_(state,a0+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a1+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a2+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a3+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a4+column*(1),a5+column*(1),a6+column*((1-(1)+1)*(1-(1)+1)),a7+column*((1-(1)+1)*(1-(1)+1)),a8+column*((1-(1)+1)*(1-(1)+1)),a9+column*((1-(1)+1)*(1-(1)+1)),a10+column*((1-(1)+1)*(1-(1)+1)),a11+column*((1-(1)+1)*(1-(1)+1)),a12+column*((1-(1)+1)*(1-(1)+1)),a13+column*((1-(1)+1)*(1-(1)+1)),a14+column*((1-(1)+1)*(1-(1)+1)),a15+column*((1-(1)+1)*(1-(1)+1)),a16+column*((1-(1)+1)*(1-(1)+1)),a17+column*((1-(1)+1)*(1-(1)+1)),a18+column*((1-(1)+1)*(1-(1)+1)),a19+column*((1-(1)+1)*(1-(1)+1)),a20+column*((1-(1)+1)*(1-(1)+1)),a21+column*((1-(1)+1)*(1-(1)+1)),a22+column*((1-(1)+1)*(1-(1)+1)),a23+column*((1-(1)+1)*(1-(1)+1)),a24+column*((1-(1)+1)*(1-(1)+1)),a25+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a26+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a27+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a28+column*((1-(1)+1)*(1-(1)+1)),a29+column*((1-(1)+1)*(1-(1)+1)),a30+column*((1-(1)+1)*(1-(1)+1)),a31+column*((1-(1)+1)*(1-(1)+1)),a32+column*((1-(1)+1)*(1-(1)+1)),a33+column*((1-(1)+1)*(1-(1)+1)),a34+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a35+column*((1-(1)+1)*(1-(1)+1)),a36+column*((1-(1)+1)*(1-(1)+1)),a37+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a38+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a39+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a40+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a41+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a42+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a43+column*((1-(1)+1)*(1-(1)+1)),a44+column*((1-(1)+1)*(1-(1)+1)),a45+column*((1-(1)+1)*(1-(1)+1)),a46+column*((1-(1)+1)*(1-(1)+1)),a47+column*((1-(1)+1)*(1-(1)+1)),a48+column*((1-(1)+1)*(1-(1)+1)),a49+column*((1-(1)+1)*(1-(1)+1)),a50+column*((1-(1)+1)*(1-(1)+1)),a51+column*(1),a52+column*((1-(1)+1)*(1-(1)+1)),a53+column*((1-(1)+1)*(1-(1)+1)),a54+column*((1-(1)+1)*(1-(1)+1)),a55+column*(1),a56+column*(1),a57+column*(1),a58+column*(1),a59+column*(1),a60+column*(1),a61+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a62+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a63+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a64+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a65+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a66+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a67+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a68+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a69+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a70+column*((1-(1)+1)*(1-(1)+1)),a71+column*((1-(1)+1)*(1-(1)+1)),a72+column*((1-(1)+1)*(1-(1)+1)),a73+column*((1-(1)+1)*(59-(1)+1)*(1-(1)+1)*(13-(1)+1)),a74+column*((59-(1)+1)),a75+column*(1),a76+column*(1),a77+column*((1-(1)+1)*(1-(1)+1)),a78+column*((1-(1)+1)*(1-(1)+1)),a79+column*((1-(1)+1)*(29-(1)+1)*(1-(1)+1)*(13-(1)+1)),a80+column*((1-(1)+1)*(29-(1)+1)*(1-(1)+1)*(13-(1)+1)),a81+column*((29-(1)+1)),a82+column*(1),a83+column*(1),a84+column*(1),a85+column*(1),a86+column*(1),a87+column*(1),a88+column*(1),a89+column*(1),a90+column*(1),a91+column*(1),a92+column*(1),a93+column*(1),a94+column*(1),a95+column*(1),a96+column*(1),a97+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a98+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(4-(1)+1)*(1-(1)+1)),a99+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a100+column*(1),a101+column*(1),a102+column*(1),a103+column*(1),a104+column*(1),a105+column*(1),a106+column*(1),a107+column*(1),a108+column*(1),a109+column*(1),a110+column*(1),a111+column*(1),a112+column*(1),a113+column*(1),a114+column*(1),a115+column*(1),a116+column*(1),a117+column*(1),a118+column*(1),a119+column*(1),a120+column*((1-(1)+1)*(1-(1)+1)));
}
}

extern "C" __global__ void cam_log10_probe(int count,const double* input,double* output){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){double x=input[i];output[i]=d_lg10(&x);}}
extern "C" __global__ void cam_log10_variants(int count,const double* input,double* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 double x=input[i],old=x;
 output[i*5]=d_lg10(&old);
 int exponent;double m=frexp(x,&exponent);
 const double inv=0.43429448190325182765;
 const double hi=0x1.3441350900000p-2,lo=0x1.ef3fde623e256p-35;
 output[i*5+1]=double(exponent)*hi+(double(exponent)*lo+inv*cam_log64(m));
 if(exponent>0){m*=2;--exponent;}
 output[i*5+2]=double(exponent)*hi+(double(exponent)*lo+inv*cam_log64(m));
 m=frexp(x,&exponent);if(m<0.70710678118654752440){m*=2;--exponent;}
 output[i*5+3]=double(exponent)*hi+(double(exponent)*lo+inv*cam_log64(m));
 double a=double(exponent)*0.30102999566398119521,b=inv*cam_log64(m);
 double residual=__fma_rn(double(exponent),0.30102999566398119521,-a)+__fma_rn(inv,cam_log64(m),-b);
 output[i*5+4]=a+(b+residual);
}
extern "C" __global__ __launch_bounds__(32) void cam_driver_stage(int columns,const unsigned char* bytes,unsigned char* workspace,int phase,void** descriptors,double* absorption,float* a0,float* a1,float* a2,float* a3,int* a4,int* a5,float* a6,float* a7,float* a8,float* a9,float* a10,float* a11,float* a12,float* a13,float* a14,float* a15,float* a16,float* a17,float* a18,float* a19,float* a20,float* a21,float* a22,float* a23,float* a24,float* a25,float* a26,float* a27,float* a28,float* a29,float* a30,float* a31,float* a32,float* a33,float* a34,float* a35,float* a36,float* a37,float* a38,float* a39,float* a40,float* a41,float* a42,float* a43,float* a44,float* a45,float* a46,float* a47,float* a48,float* a49,float* a50,int* a51,float* a52,float* a53,float* a54,int* a55,int* a56,int* a57,int* a58,int* a59,int* a60,float* a61,float* a62,float* a63,float* a64,float* a65,float* a66,float* a67,float* a68,float* a69,float* a70,float* a71,float* a72,float* a73,float* a74,int* a75,int* a76,float* a77,float* a78,float* a79,float* a80,float* a81,int* a82,int* a83,int* a84,int* a85,float* a86,int* a87,float* a88,int* a89,float* a90,float* a91,float* a92,float* a93,float* a94,float* a95,int* a96,float* a97,float* a98,float* a99,int* a100,int* a101,int* a102,int* a103,int* a104,int* a105,int* a106,int* a107,int* a108,int* a109,int* a110,int* a111,int* a112,int* a113,int* a114,int* a115,int* a116,int* a117,int* a118,int* a119,float* a120,double* gases){
int worker=blockIdx.x*blockDim.x+threadIdx.x;
int workers=gridDim.x*blockDim.x;
for(int column=worker;column<columns;column+=workers){
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
CamState* state=(CamState*)(workspace+CAM_WORK_BYTES*worker);
state->cg_abari__=(double*)(bytes+0);
state->cg_abarl__=(double*)(bytes+32);
state->cg_abco2__=(double*)(bytes+64);
state->cg_abh2o__=(double*)(bytes+216);
state->cg_abo2__=(double*)(bytes+368);
state->cg_abo3__=(double*)(bytes+520);
state->cg_abs_cff_mss_aer__=(double*)(bytes+672);
state->cg_ah2onw__=(double*)(bytes+728);
state->cg_ah2ow__=(double*)(bytes+2940728);
state->cg_amco2__=*(double*)(bytes+5880728);
state->cg_amd__=*(double*)(bytes+5880736);
state->cg_amo__=*(double*)(bytes+5880744);
state->cg_bbari__=(double*)(bytes+5880752);
state->cg_bbarl__=(double*)(bytes+5880784);
state->cg_bgscl_rf__=*(double*)(bytes+5880816);
state->cg_c16__=*(double*)(bytes+5880824);
state->cg_c17__=*(double*)(bytes+5880832);
state->cg_c26__=*(double*)(bytes+5880840);
state->cg_c27__=*(double*)(bytes+5880848);
state->cg_c28__=*(double*)(bytes+5880856);
state->cg_c29__=*(double*)(bytes+5880864);
state->cg_c30__=*(double*)(bytes+5880872);
state->cg_c31__=*(double*)(bytes+5880880);
state->cg_cappa__=*(double*)(bytes+5880888);
state->cg_carscl__=*(double*)(bytes+5880896);
state->cg_carscl_rf__=*(double*)(bytes+5880904);
state->cg_cbari__=(double*)(bytes+5880912);
state->cg_cbarl__=(double*)(bytes+5880944);
state->cg_ch4vmr__=*(double*)(bytes+5880976);
state->cg_cn_ah2ow__=(double*)(bytes+5880984);
state->cg_cn_eh2ow__=(double*)(bytes+8820984);
state->cg_co2__=(double*)(bytes+11760984);
state->cg_co2mmr__=*(double*)(bytes+11762848);
state->cg_co2vmr__=*(double*)(bytes+11762856);
state->cg_coefh__=(double*)(bytes+11762864);
state->cg_coefj__=(double*)(bytes+11762928);
state->cg_coefk__=(double*)(bytes+11762976);
state->cg_cpair__=*(double*)(bytes+11763024);
state->cg_cplol__=*(double*)(bytes+11763032);
state->cg_cplos__=*(double*)(bytes+11763040);
state->cg_dayspy__=*(double*)(bytes+11763048);
state->cg_dbari__=(double*)(bytes+11763056);
state->cg_dbarl__=(double*)(bytes+11763088);
state->cg_delta__=*(double*)(bytes+11763120);
state->cg_dlp_h2o__=*(double*)(bytes+11763128);
state->cg_dlu_h2o__=*(double*)(bytes+11763136);
state->cg_dpfco2__=*(double*)(bytes+11763144);
state->cg_dpfo3__=*(double*)(bytes+11763152);
state->cg_drh_h2o__=*(double*)(bytes+11763160);
state->cg_dte_h2o__=*(double*)(bytes+11763168);
state->cg_dtp_h2o__=*(double*)(bytes+11763176);
state->cg_dustscl__=*(double*)(bytes+11763184);
state->cg_dustscl_rf__=*(double*)(bytes+11763192);
state->cg_ebari__=(double*)(bytes+11763200);
state->cg_ebarl__=(double*)(bytes+11763232);
state->cg_eh2onw__=(double*)(bytes+11763264);
state->cg_epsilo__=*(double*)(bytes+14703264);
state->cg_epsqs__=*(double*)(bytes+14703272);
state->cg_estbl__=(double*)(bytes+14703280);
state->cg_estblh2o__=(double*)(bytes+14705280);
state->cg_f11vmr__=*(double*)(bytes+14706824);
state->cg_f12vmr__=*(double*)(bytes+14706832);
state->cg_fat__=(double*)(bytes+14706840);
state->cg_fbari__=(double*)(bytes+14706936);
state->cg_fbarl__=(double*)(bytes+14706968);
state->cg_fc1__=*(double*)(bytes+14707000);
state->cg_fet__=(double*)(bytes+14707008);
state->cg_frcsol__=(double*)(bytes+14707104);
state->cg_fwc1__=*(double*)(bytes+14707256);
state->cg_fwc2__=*(double*)(bytes+14707264);
state->cg_fwcoef__=*(double*)(bytes+14707272);
state->cg_gbg__=(double*)(bytes+14707280);
state->cg_gcb__=(double*)(bytes+14707432);
state->cg_gcphil__=(double*)(bytes+14707584);
state->cg_gcphob__=(double*)(bytes+14859584);
state->cg_gdst__=(double*)(bytes+14859736);
state->cg_gravit__=*(double*)(bytes+14860344);
state->cg_gravmks__=*(double*)(bytes+14860352);
state->cg_gsslt__=(double*)(bytes+14860360);
state->cg_gsul__=(double*)(bytes+15012360);
state->cg_gvolc__=(double*)(bytes+15164360);
state->cg_inf__=*(double*)(bytes+15164512);
state->cg_kbg__=(double*)(bytes+15164520);
state->cg_kcb__=(double*)(bytes+15164672);
state->cg_kcphil__=(double*)(bytes+15164824);
state->cg_kcphob__=(double*)(bytes+15316824);
state->cg_kdst__=(double*)(bytes+15316976);
state->cg_ksslt__=(double*)(bytes+15317584);
state->cg_ksul__=(double*)(bytes+15469584);
state->cg_kvolc__=(double*)(bytes+15621584);
state->cg_latice__=*(double*)(bytes+15621736);
state->cg_latvap__=*(double*)(bytes+15621744);
state->cg_ln_ah2ow__=(double*)(bytes+15621752);
state->cg_ln_eh2ow__=(double*)(bytes+18561752);
state->cg_max_lp_h2o__=*(double*)(bytes+21501752);
state->cg_max_lu_h2o__=*(double*)(bytes+21501760);
state->cg_max_rh_h2o__=*(double*)(bytes+21501768);
state->cg_max_te_h2o__=*(double*)(bytes+21501776);
state->cg_max_tp_h2o__=*(double*)(bytes+21501784);
state->cg_min_lp_h2o__=*(double*)(bytes+21501792);
state->cg_min_lu_h2o__=*(double*)(bytes+21501800);
state->cg_min_p_h2o__=*(double*)(bytes+21501808);
state->cg_min_rh_h2o__=*(double*)(bytes+21501816);
state->cg_min_te_h2o__=*(double*)(bytes+21501824);
state->cg_min_tp_h2o__=*(double*)(bytes+21501832);
state->cg_min_u_h2o__=*(double*)(bytes+21501840);
state->cg_mwch4__=*(double*)(bytes+21501848);
state->cg_mwco2__=*(double*)(bytes+21501856);
state->cg_mwdry__=*(double*)(bytes+21501864);
state->cg_mwf11__=*(double*)(bytes+21501872);
state->cg_mwf12__=*(double*)(bytes+21501880);
state->cg_mwh2o__=*(double*)(bytes+21501888);
state->cg_mwn2o__=*(double*)(bytes+21501896);
state->cg_n2ovmr__=*(double*)(bytes+21501904);
state->cg_nirwgt__=(double*)(bytes+21501912);
state->cg_o2mmr__=*(double*)(bytes+21502064);
state->cg_p0__=*(double*)(bytes+21502072);
state->cg_pcf__=(double*)(bytes+21502080);
state->cg_pco2__=(double*)(bytes+21502128);
state->cg_ph2o__=(double*)(bytes+21502280);
state->cg_pie__=*(double*)(bytes+21502432);
state->cg_po2__=(double*)(bytes+21502440);
state->cg_r_universal__=*(double*)(bytes+21502592);
state->cg_rair__=*(double*)(bytes+21502600);
state->cg_raytau__=(double*)(bytes+21502608);
state->cg_retab__=(double*)(bytes+21502760);
state->cg_rga__=*(double*)(bytes+21503520);
state->cg_rgsslp__=*(double*)(bytes+21503528);
state->cg_scon__=*(double*)(bytes+21503536);
state->cg_sslp__=*(double*)(bytes+21503544);
state->cg_ssltscl__=*(double*)(bytes+21503552);
state->cg_ssltscl_rf__=*(double*)(bytes+21503560);
state->cg_stebol__=*(double*)(bytes+21503568);
state->cg_sulscl__=*(double*)(bytes+21503576);
state->cg_sulscl_rf__=*(double*)(bytes+21503584);
state->cg_tauback__=*(double*)(bytes+21503592);
state->cg_tmax__=*(double*)(bytes+21503600);
state->cg_tmelt__=*(double*)(bytes+21503608);
state->cg_tmin__=*(double*)(bytes+21503616);
state->cg_v_abo3_35__=*(double*)(bytes+21503624);
state->cg_v_abo3_64__=*(double*)(bytes+21503632);
state->cg_v_raytau_35__=*(double*)(bytes+21503640);
state->cg_v_raytau_64__=*(double*)(bytes+21503648);
state->cg_volcscl__=*(double*)(bytes+21503656);
state->cg_volcscl_rf__=*(double*)(bytes+21503664);
state->cg_wavmax__=(double*)(bytes+21503672);
state->cg_wavmin__=(double*)(bytes+21503824);
state->cg_wbg__=(double*)(bytes+21503976);
state->cg_wcb__=(double*)(bytes+21504128);
state->cg_wcphil__=(double*)(bytes+21504280);
state->cg_wcphob__=(double*)(bytes+21656280);
state->cg_wdst__=(double*)(bytes+21656432);
state->cg_wsslt__=(double*)(bytes+21657040);
state->cg_wsul__=(double*)(bytes+21809040);
state->cg_wvolc__=(double*)(bytes+21961040);
state->cg_zvir__=*(double*)(bytes+21961192);
state->cg_have_ozone__=*(int*)(bytes+21961200);
state->cg_idxbcphi__=*(int*)(bytes+21961204);
state->cg_idxbcpho__=*(int*)(bytes+21961208);
state->cg_idxbg__=*(int*)(bytes+21961212);
state->cg_idxcarbonfirst__=*(int*)(bytes+21961216);
state->cg_idxdustfirst__=*(int*)(bytes+21961220);
state->cg_idxocphi__=*(int*)(bytes+21961224);
state->cg_idxocpho__=*(int*)(bytes+21961228);
state->cg_idxsslt__=*(int*)(bytes+21961232);
state->cg_idxsul__=*(int*)(bytes+21961236);
state->cg_idxvolc__=*(int*)(bytes+21961240);
state->cg_indirect__=*(int*)(bytes+21961244);
state->cg_ixcldice__=*(int*)(bytes+21961248);
state->cg_ixcldliq__=*(int*)(bytes+21961252);
state->cg_levsiz_ozone_save__=*(int*)(bytes+21961256);
state->cg_masterproc__=*(int*)(bytes+21961260);
state->cg_mxaerl__=*(int*)(bytes+21961264);
state->cg_ntoplw__=*(int*)(bytes+21961268);
state->cg_ozncyc__=*(int*)(bytes+21961272);
state->cg_radforce__=*(int*)(bytes+21961276);
state->cg_strat_volcanic__=*(int*)(bytes+21961280);
state->cg_trace_gas__=*(int*)(bytes+21961284);
state->cg_yrdata__=(int*)(bytes+21961288);
state->driver_gases[0]=gases[column*5+0];
state->driver_gases[1]=gases[column*5+1];
state->driver_gases[2]=gases[column*5+2];
state->driver_gases[3]=gases[column*5+3];
state->driver_gases[4]=gases[column*5+4];

state->scratch=workspace+worker*CAM_WORK_BYTES+2048; state->sw_flux_stride=min(32,workers-(worker/32)*32); state->sw_flux_offset=0; state->sw_flux=(double*)(workspace+workers*CAM_WORK_BYTES)+(worker/32)*32*CAM_SW_FLUX_VALUES+(worker%32); state->scratch_offset=0; state->abs_phase=0; state->abs_lane=0; state->abs_stride=1; state->abs_phase=phase; state->abs_args=descriptors+worker*41; state->abs_cache=absorption+worker*CAM_ABS_VALUES; state->abs_padding=0;
camrad_(state,a0+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a1+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a2+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a3+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a4+column*(1),a5+column*(1),a6+column*((1-(1)+1)*(1-(1)+1)),a7+column*((1-(1)+1)*(1-(1)+1)),a8+column*((1-(1)+1)*(1-(1)+1)),a9+column*((1-(1)+1)*(1-(1)+1)),a10+column*((1-(1)+1)*(1-(1)+1)),a11+column*((1-(1)+1)*(1-(1)+1)),a12+column*((1-(1)+1)*(1-(1)+1)),a13+column*((1-(1)+1)*(1-(1)+1)),a14+column*((1-(1)+1)*(1-(1)+1)),a15+column*((1-(1)+1)*(1-(1)+1)),a16+column*((1-(1)+1)*(1-(1)+1)),a17+column*((1-(1)+1)*(1-(1)+1)),a18+column*((1-(1)+1)*(1-(1)+1)),a19+column*((1-(1)+1)*(1-(1)+1)),a20+column*((1-(1)+1)*(1-(1)+1)),a21+column*((1-(1)+1)*(1-(1)+1)),a22+column*((1-(1)+1)*(1-(1)+1)),a23+column*((1-(1)+1)*(1-(1)+1)),a24+column*((1-(1)+1)*(1-(1)+1)),a25+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a26+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a27+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a28+column*((1-(1)+1)*(1-(1)+1)),a29+column*((1-(1)+1)*(1-(1)+1)),a30+column*((1-(1)+1)*(1-(1)+1)),a31+column*((1-(1)+1)*(1-(1)+1)),a32+column*((1-(1)+1)*(1-(1)+1)),a33+column*((1-(1)+1)*(1-(1)+1)),a34+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a35+column*((1-(1)+1)*(1-(1)+1)),a36+column*((1-(1)+1)*(1-(1)+1)),a37+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a38+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a39+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a40+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a41+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a42+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a43+column*((1-(1)+1)*(1-(1)+1)),a44+column*((1-(1)+1)*(1-(1)+1)),a45+column*((1-(1)+1)*(1-(1)+1)),a46+column*((1-(1)+1)*(1-(1)+1)),a47+column*((1-(1)+1)*(1-(1)+1)),a48+column*((1-(1)+1)*(1-(1)+1)),a49+column*((1-(1)+1)*(1-(1)+1)),a50+column*((1-(1)+1)*(1-(1)+1)),a51+column*(1),a52+column*((1-(1)+1)*(1-(1)+1)),a53+column*((1-(1)+1)*(1-(1)+1)),a54+column*((1-(1)+1)*(1-(1)+1)),a55+column*(1),a56+column*(1),a57+column*(1),a58+column*(1),a59+column*(1),a60+column*(1),a61+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a62+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a63+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a64+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a65+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a66+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a67+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a68+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a69+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a70+column*((1-(1)+1)*(1-(1)+1)),a71+column*((1-(1)+1)*(1-(1)+1)),a72+column*((1-(1)+1)*(1-(1)+1)),a73+column*((1-(1)+1)*(59-(1)+1)*(1-(1)+1)*(13-(1)+1)),a74+column*((59-(1)+1)),a75+column*(1),a76+column*(1),a77+column*((1-(1)+1)*(1-(1)+1)),a78+column*((1-(1)+1)*(1-(1)+1)),a79+column*((1-(1)+1)*(29-(1)+1)*(1-(1)+1)*(13-(1)+1)),a80+column*((1-(1)+1)*(29-(1)+1)*(1-(1)+1)*(13-(1)+1)),a81+column*((29-(1)+1)),a82+column*(1),a83+column*(1),a84+column*(1),a85+column*(1),a86+column*(1),a87+column*(1),a88+column*(1),a89+column*(1),a90+column*(1),a91+column*(1),a92+column*(1),a93+column*(1),a94+column*(1),a95+column*(1),a96+column*(1),a97+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a98+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(4-(1)+1)*(1-(1)+1)),a99+column*((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(1-(1)+1)),a100+column*(1),a101+column*(1),a102+column*(1),a103+column*(1),a104+column*(1),a105+column*(1),a106+column*(1),a107+column*(1),a108+column*(1),a109+column*(1),a110+column*(1),a111+column*(1),a112+column*(1),a113+column*(1),a114+column*(1),a115+column*(1),a116+column*(1),a117+column*(1),a118+column*(1),a119+column*(1),a120+column*((1-(1)+1)*(1-(1)+1)));
}
}


extern "C" __global__ __launch_bounds__(64,8) void cam_absorption(int columns,const unsigned char* parents,void** descriptors,double* absorption,unsigned char* workspace){
int thread=blockIdx.x*blockDim.x+threadIdx.x;
int column=thread/(CAM_LEVELS+1),lane=thread%(CAM_LEVELS+1);
if(column>=columns)return;
CamState* state=(CamState*)(workspace+thread*CAM_ABS_WORK_BYTES);
*state=*(const CamState*)(parents+column*CAM_WORK_BYTES);
state->scratch=workspace+thread*CAM_ABS_WORK_BYTES+2048;
state->scratch_offset=0; state->abs_phase=0; state->abs_lane=lane; state->abs_stride=CAM_LEVELS+1;
int lchnk=1,ncol=1,pcols=1,pver=CAM_LEVELS,pverp=CAM_LEVELS+1;
double* cache=absorption+column*CAM_ABS_VALUES;
void** operands=descriptors+column*41;
radabs_parallel_(state,&lchnk,&ncol,&pcols,&pver,&pverp,(double*)operands[0],(double*)operands[1],(double*)operands[2],(double*)operands[3],(double*)operands[4],(double*)operands[5],(double*)operands[6],(double*)operands[7],(double*)operands[8],(double*)operands[9],(double*)operands[10],(double*)operands[11],(double*)operands[12],(double*)operands[13],(double*)operands[14],(double*)operands[15],(double*)operands[16],(double*)operands[17],(double*)operands[18],(double*)operands[19],(double*)operands[20],(double*)operands[21],(double*)operands[22],(double*)operands[23],(double*)operands[24],(double*)operands[25],(double*)operands[26],(double*)operands[27],(double*)operands[28],(double*)operands[29],(double*)operands[30],(double*)operands[31],(double*)operands[32],(double*)operands[33],(double*)operands[34],cache,cache+(CAM_LEVELS+1)*(CAM_LEVELS+1),(double*)operands[37],(double*)operands[38],(double*)operands[39],(double*)operands[40]);
}

static_assert(CAM_WORK_BYTES>=2048+(0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+(((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+(((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1)*(3-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(377)*8+7+(377)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(59)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+(59)*8+7+(29)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(1)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+1-(0)+1))*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*4+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(4-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS+1-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+((1-1+1-(1)+1)*(CAM_LEVELS-1+2-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(767)*8+7+0+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+(((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15-(1)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7+(((CAM_LEVELS+1)-(0)+1)*(15+1-(0)+1))*4+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(0)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(7-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*4+7+(1)*4+7+((1-(1)+1)*(CAM_LEVELS-(-1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(0)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(((CAM_LEVELS+1)-(0)+1))*4+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((2-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+(((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((14-(1)+1)*(1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+0+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1)*(11-(1)+1))*8+7+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(11-(1)+1))*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+(1)*8+7+(1)*8+7+(1)*4+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*((CAM_LEVELS+1)-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1))*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(133)*8+7+(133)*8+7+(13)*8+7+(133)*8+7+(133)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+(1)*8+7+((1-(1)+1)*(CAM_LEVELS-(1)+1)*(13-(1)+1))*8+7+(1)*8+7+0+0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+0),"CAM main frame exceeds reserved arena");
static_assert(CAM_WORK_BYTES>=2048+(0+0+0),"CAM main frame exceeds reserved arena");
