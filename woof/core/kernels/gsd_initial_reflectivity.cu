// module_diag_gsd.F:17-153, NOAA-EMC/HRRR 40ee6058. The included fork
// transcription owns calc_refl10cm; this entry only prepares its t=0 inputs.
// 13 float, 4 double and 3 bool arena slots, bounded to threads in flight.
extern "C" __global__ void gsd_refl_first_time_step(
 const float *th, const float *pii, const float *p, const float *rho_curr,
 const float *qv, const float *qc, const float *qnr, const float *qr,
 const float *qs, const float *qg, float *refl,
 float *ws_f, double *ws_d, bool *ws_b,
 int nz, int ncol, int state_inputs, int thb_full)
{
 const size_t thread=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 const size_t stride=(size_t)gridDim.x*blockDim.x;
 for(size_t col=thread;col<(size_t)ncol;col+=stride){
  TfArena ws={ws_f+thread,ws_d+thread,ws_b+thread,stride,(size_t)nz+1,0,0,0};
  TfCol<float> t1d=ws.F(),nr1d=ws.F();
  for(int k=0;k<nz;++k){
   const size_t i=(size_t)k*ncol+col;
   // State route fuses the fork's initial phy_prep thermodynamics. The
   // prepared-input route below is also the native-routine oracle door.
   const float thermal=state_inputs ? th[thb_full ? i : k]+pii[i] : th[i];
#ifdef GPUWM_USE_THETA_M
   const float theta=state_inputs ? __fdiv_rn(thermal,
       __fadd_rn(1.f,__fmul_rn(RVOVRD,qv[i]))) : thermal;
#else
   const float theta=thermal;
#endif
   const float exner=state_inputs ? gfk_pow(FDIV(p[i],P0),RCP) : pii[i];
   const float rho=state_inputs ? FDIV(1.f,rho_curr[i])*(1.f+qv[i]) : rho_curr[i];
   if(qnr[i]<=0.f && qr[i]>=1.e-12f){
    nr1d[k]=FDIV(8.e6f,rho)*tf_powf(FDIV(rho*qr[i],3.14159f*1000.f*8.e6f),0.25f);
   }else nr1d[k]=qnr[i];
   t1d[k]=theta*exner;
  }
  // TfCol views read the unchanged model columns directly. Only t1d and
  // nr1d are private; calc_refl10cm does not modify any input species.
  TfCol<float> qv1d={const_cast<float*>(qv)+col,(size_t)ncol};
  TfCol<float> qc1d={const_cast<float*>(qc)+col,(size_t)ncol};
  TfCol<float> qr1d={const_cast<float*>(qr)+col,(size_t)ncol};
  TfCol<float> qs1d={const_cast<float*>(qs)+col,(size_t)ncol};
  TfCol<float> qg1d={const_cast<float*>(qg)+col,(size_t)ncol};
  TfCol<float> p1d={const_cast<float*>(p)+col,(size_t)ncol};
  TfCol<float> dbz={refl+col,(size_t)ncol};
  tf_refl10cm(qv1d,qc1d,qr1d,nr1d,qs1d,qg1d,t1d,p1d,dbz,nz,1,ws);
  for(int k=0;k<nz;++k)dbz[k]=fmax_(-35.f,dbz[k]);
 }
}
