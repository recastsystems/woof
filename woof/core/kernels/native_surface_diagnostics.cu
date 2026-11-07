// WRF v4.7.1 public-domain module_diag_rasm.F, module_diag_solar.F,
// and diag_functions. One CUDA thread per column.
#ifdef GPUWM_WRF_EXACT
#define nd_pow gfk_pow
#define nd_exp gfk_exp
#define nd_acos glibc_acosf
#else
#define nd_pow powf
#define nd_exp expf
#define nd_acos acosf
#endif
extern "C" __global__ void solar_prepare(const float* alt,const float* qv,
 const float* ph,const float* phb,float* rho,float* dz,int nz,int n,int full){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 for(int k=0;k<nz;++k){int i=k*n+c;
  rho[i]=(1.f/alt[i])*(1.f+qv[i]);
  float a=ph[i]+phb[full?i:k],b=ph[i+n]+phb[full?i+n:k+1];
  dz[i]=b/9.81f-a/9.81f;
 }
}
__device__ float rasm_pmsl(float t,float p,float pb,float moist,float ht,float psfc){
 float rd=(float)(6.02214e26*1.38065e-23/28.966),cp=1004.64f,g=9.80616f,xlapse=.0065f;
 float tt=(t+300.f)*nd_pow((p+pb)/100000.f,rd/cp);
 float tv=tt*(1.f+moist*.61f),z=ht*g,alpha=rd*xlapse/g;
 if(fabsf(z/g)<1.e-4f)return psfc;
 float tstar=tv*(1.f+alpha*(psfc/(p+pb)-1.f));
 float tt0=tstar+xlapse*z/g,alph;
 if(tstar<=290.5f&&tt0>290.5f)alph=rd/z*(290.5f-tstar);
 else if(tstar>290.5f&&tt0>290.5f){alph=0.f;tstar=.5f*(290.5f+tstar);}
 else{alph=alpha;if(tstar<255.f)tstar=.5f*(255.f+tstar);}
 float beta=z/(rd*tstar),ab=alph*beta;
 return psfc*nd_exp(beta*(1.f-alph*beta/2.f+ab*ab/3.f));
}
extern "C" __global__ void rasm_mean(const float* psfc,const float* tsk,const float* t2,const float* th2,const float* q2,const float* u10,const float* v10,const float* hfx,const float* lh,const float* swdnb,const float* glw,const float* lwupb,const float* swupb,const float* swupt,const float* swdnt,const float* lwupt,const float* lwdnt,const float* t,const float* p,const float* pb,const float* moist,const float* ht,float* o0,float* o1,float* o2,float* o3,float* o4,float* o5,float* o6,float* o7,float* o8,float* o9,float* o10,float* o11,float* o12,float* o13,float* o14,float* o15,float* o16,float* o17,int* steps,int n,int reset,int finish){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
int count=reset?1:steps[c]+1;steps[c]=count;
o0[c]=(reset?0.f:o0[c])+psfc[c];if(finish)o0[c]=o0[c]/count;
o1[c]=(reset?0.f:o1[c])+tsk[c];if(finish)o1[c]=o1[c]/count;
o2[c]=(reset?0.f:o2[c])+rasm_pmsl(t[c],p[c],pb[c],moist[c],ht[c],psfc[c]);if(finish)o2[c]=o2[c]/count;
o3[c]=(reset?0.f:o3[c])+t2[c];if(finish)o3[c]=o3[c]/count;
o4[c]=(reset?0.f:o4[c])+th2[c];if(finish)o4[c]=o4[c]/count;
o5[c]=(reset?0.f:o5[c])+q2[c];if(finish)o5[c]=o5[c]/count;
o6[c]=(reset?0.f:o6[c])+u10[c];if(finish)o6[c]=o6[c]/count;
o7[c]=(reset?0.f:o7[c])+v10[c];if(finish)o7[c]=o7[c]/count;
o8[c]=(reset?0.f:o8[c])+hfx[c];if(finish)o8[c]=o8[c]/count;
o9[c]=(reset?0.f:o9[c])+lh[c];if(finish)o9[c]=o9[c]/count;
o10[c]=(reset?0.f:o10[c])+swdnb[c];if(finish)o10[c]=o10[c]/count;
o11[c]=(reset?0.f:o11[c])+glw[c];if(finish)o11[c]=o11[c]/count;
o12[c]=(reset?0.f:o12[c])+lwupb[c];if(finish)o12[c]=o12[c]/count;
o13[c]=(reset?0.f:o13[c])+swupb[c];if(finish)o13[c]=o13[c]/count;
o14[c]=(reset?0.f:o14[c])+swupt[c];if(finish)o14[c]=o14[c]/count;
o15[c]=(reset?0.f:o15[c])+swdnt[c];if(finish)o15[c]=o15[c]/count;
o16[c]=(reset?0.f:o16[c])+lwupt[c];if(finish)o16[c]=o16[c]/count;
o17[c]=(reset?0.f:o17[c])+lwdnt[c];if(finish)o17[c]=o17[c]/count;
}
extern "C" __global__ void rasm_diurnal(const float* psfc,const float* tsk,const float* t2,const float* th2,const float* q2,const float* u10,const float* v10,const float* hfx,const float* lh,const float* swdnb,const float* glw,const float* lwupb,const float* swupb,const float* swupt,const float* swdnt,const float* lwupt,const float* lwdnt,float* a0,float* a1,float* a2,float* a3,float* a4,float* a5,float* a6,float* a7,float* a8,float* a9,float* a10,float* a11,float* a12,float* a13,float* a14,float* a15,float* a16,float* b0,float* b1,float* b2,float* b3,float* b4,float* b5,float* b6,float* b7,float* b8,float* b9,float* b10,float* b11,float* b12,float* b13,float* b14,float* b15,float* b16,int* steps,int* days,int n,int reset,int resetmonth,int finish,int finishmonth,int cycle){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
int count=reset?1:steps[c]+1;steps[c]=count;int d=resetmonth?0:days[c];if(finish&&cycle==8)++d;days[c]=d;
a0[c]=(reset?0.f:a0[c])+psfc[c];if(finish)a0[c]=a0[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b0[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b0[x]=b0[x]+a0[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b0[x]=b0[x]/d;}}
a1[c]=(reset?0.f:a1[c])+tsk[c];if(finish)a1[c]=a1[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b1[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b1[x]=b1[x]+a1[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b1[x]=b1[x]/d;}}
a2[c]=(reset?0.f:a2[c])+t2[c];if(finish)a2[c]=a2[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b2[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b2[x]=b2[x]+a2[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b2[x]=b2[x]/d;}}
a3[c]=(reset?0.f:a3[c])+th2[c];if(finish)a3[c]=a3[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b3[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b3[x]=b3[x]+a3[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b3[x]=b3[x]/d;}}
a4[c]=(reset?0.f:a4[c])+q2[c];if(finish)a4[c]=a4[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b4[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b4[x]=b4[x]+a4[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b4[x]=b4[x]/d;}}
a5[c]=(reset?0.f:a5[c])+u10[c];if(finish)a5[c]=a5[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b5[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b5[x]=b5[x]+a5[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b5[x]=b5[x]/d;}}
a6[c]=(reset?0.f:a6[c])+v10[c];if(finish)a6[c]=a6[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b6[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b6[x]=b6[x]+a6[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b6[x]=b6[x]/d;}}
a7[c]=(reset?0.f:a7[c])+hfx[c];if(finish)a7[c]=a7[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b7[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b7[x]=b7[x]+a7[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b7[x]=b7[x]/d;}}
a8[c]=(reset?0.f:a8[c])+lh[c];if(finish)a8[c]=a8[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b8[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b8[x]=b8[x]+a8[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b8[x]=b8[x]/d;}}
a9[c]=(reset?0.f:a9[c])+swdnb[c];if(finish)a9[c]=a9[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b9[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b9[x]=b9[x]+a9[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b9[x]=b9[x]/d;}}
a10[c]=(reset?0.f:a10[c])+glw[c];if(finish)a10[c]=a10[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b10[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b10[x]=b10[x]+a10[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b10[x]=b10[x]/d;}}
a11[c]=(reset?0.f:a11[c])+lwupb[c];if(finish)a11[c]=a11[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b11[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b11[x]=b11[x]+a11[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b11[x]=b11[x]/d;}}
a12[c]=(reset?0.f:a12[c])+swupb[c];if(finish)a12[c]=a12[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b12[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b12[x]=b12[x]+a12[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b12[x]=b12[x]/d;}}
a13[c]=(reset?0.f:a13[c])+swupt[c];if(finish)a13[c]=a13[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b13[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b13[x]=b13[x]+a13[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b13[x]=b13[x]/d;}}
a14[c]=(reset?0.f:a14[c])+swdnt[c];if(finish)a14[c]=a14[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b14[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b14[x]=b14[x]+a14[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b14[x]=b14[x]/d;}}
a15[c]=(reset?0.f:a15[c])+lwupt[c];if(finish)a15[c]=a15[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b15[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b15[x]=b15[x]+a15[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b15[x]=b15[x]/d;}}
a16[c]=(reset?0.f:a16[c])+lwdnt[c];if(finish)a16[c]=a16[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b16[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b16[x]=b16[x]+a16[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b16[x]=b16[x]/d;}}
}

__device__ float solar_integral(const float* var,const float* radius,const float* rho,const float* dz,const float* qv,int c,int n,int first,int last){
 float v=0.f;
 for(int k=first;k<last;++k){size_t x=(size_t)k*n+c;float rhodz=rho[x]*dz[x]/(1.f+qv[x]);
  if(radius)v=v+var[x]*radius[x]*rhodz;else v=v+var[x]*rhodz;}
 return v;
}
__device__ float solar_wc(int k,int c,int n,const float* qc,const float* qi,const float* qs,int fc,int fi,int fs){
 size_t x=(size_t)k*n+c;float wc=0.f;if(fc)wc=wc+qc[x];if(fi)wc=wc+qi[x];if(fs)wc=wc+qs[x];return wc;
}
__device__ float solar_height(int k,int c,int n,const float* ph,const float* phb){
 size_t x=(size_t)k*n+c;return ((ph[x]+phb[x])+(ph[x+n]+phb[x+n]))/(2.f*9.81f);
}
extern "C" __global__ void solar_diag(const float* rho,const float* dz8w,const float* ph,const float* phb,const float* cldfrac3d,const float* qv,const float* qc,const float* qi,const float* qs,const float* qc_tot,const float* qi_tot,const float* re_cloud,const float* re_ice,const float* re_snow,const float* coszen,const float* swdnb,const float* swdnt,float* clrnidx,float* sza,float* ghi_accum,float* cldfrac2d,float* wvp2d,float* lwp2d,float* iwp2d,float* swp2d,float* wp2d_sum,float* lwp2d_tot,float* iwp2d_tot,float* wp2d_tot_sum,float* re_cloud_path,float* re_ice_path,float* re_snow_path,float* re_cloud_path_tot,float* re_ice_path_tot,float* tau_qc,float* tau_qi,float* tau_qs,float* tau_qc_tot,float* tau_qi_tot,float* cbase,float* ctop,float* cbase_tot,float* ctop_tot,int nz,int n,int first,int last,float dt,int fv,int fc,int fi,int fs,int hc,int hi,int hs){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 sza[c]=(180.f/(4.f*0.7853981633974483f))*fmaxf(nd_acos(coszen[c]),0.f);
 clrnidx[c]=(sza[c]<90.f&&swdnt[c]>0.f)?swdnb[c]/swdnt[c]:-999.f;
 ghi_accum[c]=ghi_accum[c]+swdnb[c]*dt;
 float frac=cldfrac3d[(size_t)first*n+c];for(int k=first+1;k<last;++k)frac=fmaxf(frac,cldfrac3d[(size_t)k*n+c]);cldfrac2d[c]=frac;
 if(fv)wvp2d[c]=solar_integral(qv,0,rho,dz8w,qv,c,n,first,last);
if(fc){float q=solar_integral(qc,0,rho,dz8w,qv,c,n,first,last);lwp2d[c]=fabsf(fmaxf(q,0.f));
if(hc==1){float r=q>0.f?solar_integral(qc,re_cloud,rho,dz8w,qv,c,n,first,last)/q:0.f;re_cloud_path[c]=r;
tau_qc[c]=r>0.f?1.5f*q/r/1000.f:0.f;}
}else{lwp2d[c]=-999.f;re_cloud_path[c]=-999.f;tau_qc[c]=-999.f;}
if(fc){float q=solar_integral(qc_tot,0,rho,dz8w,qv,c,n,first,last);lwp2d_tot[c]=fabsf(fmaxf(q,0.f));
if(hc==1){float r=q>0.f?solar_integral(qc_tot,re_cloud,rho,dz8w,qv,c,n,first,last)/q:0.f;re_cloud_path_tot[c]=r;
tau_qc_tot[c]=r>0.f?1.5f*q/r/1000.f:0.f;}
}else{lwp2d_tot[c]=-999.f;re_cloud_path_tot[c]=-999.f;tau_qc_tot[c]=-999.f;}
if(fi){float q=solar_integral(qi,0,rho,dz8w,qv,c,n,first,last);iwp2d[c]=fabsf(fmaxf(q,0.f));
if(hi==1){float r=q>0.f?solar_integral(qi,re_ice,rho,dz8w,qv,c,n,first,last)/q:0.f;re_ice_path[c]=r;
tau_qi[c]=r>0.f?q*1000.f*(.02f+4.2f/(2.f*r*1.e6f)):0.f;}
}else{iwp2d[c]=-999.f;re_ice_path[c]=-999.f;tau_qi[c]=-999.f;}
if(fi){float q=solar_integral(qi_tot,0,rho,dz8w,qv,c,n,first,last);iwp2d_tot[c]=fabsf(fmaxf(q,0.f));
if(hi==1){float r=q>0.f?solar_integral(qi_tot,re_ice,rho,dz8w,qv,c,n,first,last)/q:0.f;re_ice_path_tot[c]=r;
tau_qi_tot[c]=r>0.f?q*1000.f*(.02f+4.2f/(2.f*r*1.e6f)):0.f;}
}else{iwp2d_tot[c]=-999.f;re_ice_path_tot[c]=-999.f;tau_qi_tot[c]=-999.f;}
if(fs){float q=solar_integral(qs,0,rho,dz8w,qv,c,n,first,last);swp2d[c]=fabsf(fmaxf(q,0.f));
if(hs==1){float r=q>0.f?solar_integral(qs,re_snow,rho,dz8w,qv,c,n,first,last)/q:0.f;re_snow_path[c]=r;
tau_qs[c]=r>0.f?q*1000.f*(.02f+4.2f/(2.f*r*1.e6f)):0.f;}
}else{swp2d[c]=-999.f;re_snow_path[c]=-999.f;tau_qs[c]=-999.f;}
if(fc||fi||fs){
 wp2d_sum[c]=fmaxf(lwp2d[c],0.f)+fmaxf(iwp2d[c],0.f)+fmaxf(swp2d[c],0.f);
 wp2d_tot_sum[c]=fmaxf(lwp2d_tot[c],0.f)+fmaxf(iwp2d_tot[c],0.f)+fmaxf(swp2d[c],0.f);
if(wp2d_sum[c]>0.f){
 int k=first;while(solar_wc(k,c,n,qc,qi,qs,fc,fi,fs)<1.e-5f&&k<last)++k;
 cbase[c]=k==last?-999.f:solar_height(k,c,n,ph,phb);
 k=last;while(solar_wc(k,c,n,qc,qi,qs,fc,fi,fs)<1.e-5f&&k>first)--k;
 ctop[c]=k==first?-999.f:solar_height(k,c,n,ph,phb);
 }else{cbase[c]=-999.f;ctop[c]=-999.f;}
if(wp2d_tot_sum[c]>0.f){
 int k=first;while(solar_wc(k,c,n,qc_tot,qi_tot,qs,fc,fi,fs)<1.e-5f&&k<last)++k;
 cbase_tot[c]=k==last?-999.f:solar_height(k,c,n,ph,phb);
 k=last;while(solar_wc(k,c,n,qc_tot,qi_tot,qs,fc,fi,fs)<1.e-5f&&k>first)--k;
 ctop_tot[c]=k==first?-999.f:solar_height(k,c,n,ph,phb);
 }else{cbase_tot[c]=-999.f;ctop_tot[c]=-999.f;}
}else{wp2d_sum[c]=-999.f;wp2d_tot_sum[c]=-999.f;cbase[c]=-999.f;ctop[c]=-999.f;cbase_tot[c]=-999.f;ctop_tot[c]=-999.f;}
}

// WRF module_radiation_driver.F cal_cldfra1/cal_cldfra2. The original
// expression boundaries are retained in strict mode (fmad=false). A single
// device pass replaces the shared array-expression path's temporary fields.
extern "C" __global__ void solar_cloud_fraction(const float* qv,const float* qc,
 const float* qi,const float* qs,const float* t,const float* p,float* out,
 int n,int fc,int fi,int fs,int simple,int ferrier){
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=n)return;
 if(!fc){out[x]=0.f;return;}
 if(simple){out[x]=(qc[x]+(fi?qi[x]:0.f)>1.e-6f)?1.f:0.f;return;}
 float tc=t[x]-273.15f;
 float esw=1000.f*.61078f*nd_exp(17.2693882f*tc/(t[x]-35.86f));
 float esi=1000.f*.61078f*nd_exp(21.8745584f*tc/(t[x]-7.66f));
 float ep2=287.f/461.6f;
 float qvsw=ep2*esw/(p[x]-esw),qvsi=ep2*esi/(p[x]-esi);
 float qcld,weight;
 if(fi&&fs){qcld=qi[x]+qc[x]+qs[x];weight=qcld<1.e-12f?0.f:(qi[x]+qs[x])/fmaxf(qcld,1.e-12f);}
 else if(fi){qcld=qi[x]+qc[x];weight=qcld<1.e-12f?0.f:qi[x]/fmaxf(qcld,1.e-12f);
  if(ferrier&&qcld>=1.e-12f&&tc< -40.f)weight=1.f;
 }else{qcld=qc[x];weight=qcld<1.e-12f?0.f:(t[x]>273.15f?0.f:1.f);}
 float qvs=(1.f-weight)*qvsw+weight*qvsi,rh=qv[x]/qvs;
 float subsat=fmaxf(1.e-10f,qvs-qv[x]);
 float arg=fmaxf(-6.9f,-100.f*qcld/nd_pow(subsat,.49f));
 float frac=nd_pow(fmaxf(1.e-10f,rh),.25f)*(1.f-nd_exp(arg));
 if(frac<.01f)frac=0.f;
 out[x]=qcld<1.e-12f?0.f:(rh>=1.f?1.f:frac);
}

// Fast forecast bindings. The strict replay entries above retain their ABI and arithmetic.

// WRF getResetState/getAvgState in REAL arithmetic, on the resident columns.
__device__ void rasm_alarm(float seconds,float dt,float interval,int& reset,int& finish){
 float x=seconds/60.f;
 int divisor=(int)floorf(__fadd_rn(interval/dt,.5f));
 reset=((int)floorf(__fadd_rn(__fmul_rn(x,60.f)/dt,.5f)))%divisor==0;
 finish=((int)floorf(__fadd_rn(__fmul_rn(__fadd_rn(x,dt/60.f),60.f)/dt,.5f)))%divisor==0;
}
extern "C" __global__ void rasm_mean_bound(const float* psfc,const float* tsk,const float* t2,const float* th2,const float* q2,const float* u10,const float* v10,const float* hfx,const float* lh,const float* swdnb,const float* glw,const float* lwupb,const float* swupb,const float* swupt,const float* swdnt,const float* lwupt,const float* lwdnt,const float* t,const float* p,const float* pb,const float* moist,const float* ht,float* o0,float* o1,float* o2,float* o3,float* o4,float* o5,float* o6,float* o7,float* o8,float* o9,float* o10,float* o11,float* o12,float* o13,float* o14,float* o15,float* o16,float* o17,int* steps,int n,float seconds,float dt,float interval,int monthly,int rm,int fm,int full){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
int reset,finish;if(monthly){reset=rm;finish=fm;}else rasm_alarm(seconds,dt,interval,reset,finish);
int count=reset?1:steps[c]+1;steps[c]=count;
o0[c]=(reset?0.f:o0[c])+psfc[c];if(finish)o0[c]=o0[c]/count;
o1[c]=(reset?0.f:o1[c])+tsk[c];if(finish)o1[c]=o1[c]/count;
o2[c]=(reset?0.f:o2[c])+rasm_pmsl(t[c],p[c]-pb[full?c:0],pb[full?c:0],moist[c],ht[c],psfc[c]);if(finish)o2[c]=o2[c]/count;
o3[c]=(reset?0.f:o3[c])+t2[c];if(finish)o3[c]=o3[c]/count;
o4[c]=(reset?0.f:o4[c])+th2[c];if(finish)o4[c]=o4[c]/count;
o5[c]=(reset?0.f:o5[c])+q2[c];if(finish)o5[c]=o5[c]/count;
o6[c]=(reset?0.f:o6[c])+u10[c];if(finish)o6[c]=o6[c]/count;
o7[c]=(reset?0.f:o7[c])+v10[c];if(finish)o7[c]=o7[c]/count;
o8[c]=(reset?0.f:o8[c])+hfx[c];if(finish)o8[c]=o8[c]/count;
o9[c]=(reset?0.f:o9[c])+lh[c];if(finish)o9[c]=o9[c]/count;
o10[c]=(reset?0.f:o10[c])+swdnb[c];if(finish)o10[c]=o10[c]/count;
o11[c]=(reset?0.f:o11[c])+glw[c];if(finish)o11[c]=o11[c]/count;
o12[c]=(reset?0.f:o12[c])+lwupb[c];if(finish)o12[c]=o12[c]/count;
o13[c]=(reset?0.f:o13[c])+swupb[c];if(finish)o13[c]=o13[c]/count;
o14[c]=(reset?0.f:o14[c])+swupt[c];if(finish)o14[c]=o14[c]/count;
o15[c]=(reset?0.f:o15[c])+swdnt[c];if(finish)o15[c]=o15[c]/count;
o16[c]=(reset?0.f:o16[c])+lwupt[c];if(finish)o16[c]=o16[c]/count;
o17[c]=(reset?0.f:o17[c])+lwdnt[c];if(finish)o17[c]=o17[c]/count;
}
extern "C" __global__ void rasm_diurnal_bound(const float* psfc,const float* tsk,const float* t2,const float* th2,const float* q2,const float* u10,const float* v10,const float* hfx,const float* lh,const float* swdnb,const float* glw,const float* lwupb,const float* swupb,const float* swupt,const float* swdnt,const float* lwupt,const float* lwdnt,float* a0,float* a1,float* a2,float* a3,float* a4,float* a5,float* a6,float* a7,float* a8,float* a9,float* a10,float* a11,float* a12,float* a13,float* a14,float* a15,float* a16,float* b0,float* b1,float* b2,float* b3,float* b4,float* b5,float* b6,float* b7,float* b8,float* b9,float* b10,float* b11,float* b12,float* b13,float* b14,float* b15,float* b16,int* steps,int* days,int n,float seconds,float dt,int resetmonth,int finishmonth,int cycle){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
int reset,finish;rasm_alarm(seconds,dt,10800.f,reset,finish);
int count=reset?1:steps[c]+1;steps[c]=count;int d=resetmonth?0:days[c];if(finish&&cycle==8)++d;days[c]=d;
a0[c]=(reset?0.f:a0[c])+psfc[c];if(finish)a0[c]=a0[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b0[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b0[x]=b0[x]+a0[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b0[x]=b0[x]/d;}}
a1[c]=(reset?0.f:a1[c])+tsk[c];if(finish)a1[c]=a1[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b1[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b1[x]=b1[x]+a1[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b1[x]=b1[x]/d;}}
a2[c]=(reset?0.f:a2[c])+t2[c];if(finish)a2[c]=a2[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b2[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b2[x]=b2[x]+a2[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b2[x]=b2[x]/d;}}
a3[c]=(reset?0.f:a3[c])+th2[c];if(finish)a3[c]=a3[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b3[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b3[x]=b3[x]+a3[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b3[x]=b3[x]/d;}}
a4[c]=(reset?0.f:a4[c])+q2[c];if(finish)a4[c]=a4[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b4[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b4[x]=b4[x]+a4[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b4[x]=b4[x]/d;}}
a5[c]=(reset?0.f:a5[c])+u10[c];if(finish)a5[c]=a5[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b5[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b5[x]=b5[x]+a5[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b5[x]=b5[x]/d;}}
a6[c]=(reset?0.f:a6[c])+v10[c];if(finish)a6[c]=a6[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b6[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b6[x]=b6[x]+a6[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b6[x]=b6[x]/d;}}
a7[c]=(reset?0.f:a7[c])+hfx[c];if(finish)a7[c]=a7[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b7[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b7[x]=b7[x]+a7[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b7[x]=b7[x]/d;}}
a8[c]=(reset?0.f:a8[c])+lh[c];if(finish)a8[c]=a8[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b8[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b8[x]=b8[x]+a8[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b8[x]=b8[x]/d;}}
a9[c]=(reset?0.f:a9[c])+swdnb[c];if(finish)a9[c]=a9[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b9[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b9[x]=b9[x]+a9[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b9[x]=b9[x]/d;}}
a10[c]=(reset?0.f:a10[c])+glw[c];if(finish)a10[c]=a10[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b10[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b10[x]=b10[x]+a10[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b10[x]=b10[x]/d;}}
a11[c]=(reset?0.f:a11[c])+lwupb[c];if(finish)a11[c]=a11[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b11[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b11[x]=b11[x]+a11[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b11[x]=b11[x]/d;}}
a12[c]=(reset?0.f:a12[c])+swupb[c];if(finish)a12[c]=a12[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b12[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b12[x]=b12[x]+a12[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b12[x]=b12[x]/d;}}
a13[c]=(reset?0.f:a13[c])+swupt[c];if(finish)a13[c]=a13[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b13[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b13[x]=b13[x]+a13[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b13[x]=b13[x]/d;}}
a14[c]=(reset?0.f:a14[c])+swdnt[c];if(finish)a14[c]=a14[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b14[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b14[x]=b14[x]+a14[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b14[x]=b14[x]/d;}}
a15[c]=(reset?0.f:a15[c])+lwupt[c];if(finish)a15[c]=a15[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b15[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b15[x]=b15[x]+a15[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b15[x]=b15[x]/d;}}
a16[c]=(reset?0.f:a16[c])+lwdnt[c];if(finish)a16[c]=a16[c]/count;
if(resetmonth){for(int k=0;k<8;++k)b16[k*n+c]=0.f;}
if(finish){int x=(cycle-1)*n+c;b16[x]=b16[x]+a16[c];}
if(finishmonth){for(int k=0;k<8;++k){int x=k*n+c;b16[x]=b16[x]/d;}}
}

extern "C" __global__ void solar_prepare_bound(const float* qv,const float* qc,
 const float* qi,const float* qs,const float* t,const float* p,float* out,
 const float* alt,const float* ph,const float* phb,float* rho,float* dz,float* qc_tot,float* qi_tot,int n,int columns,int full,int fc,int fi,int fs,int simple,int ferrier){
 int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=n)return;

 int k=x/columns;
 rho[x]=(1.f/alt[x])*(1.f+qv[x]);
 float a=ph[x]+phb[full?x:k],b=ph[x+columns]+phb[full?x+columns:k+1];
 dz[x]=b/9.81f-a/9.81f;qc_tot[x]=qc[x];qi_tot[x]=qi[x];
 if(!fc){out[x]=0.f;return;}
 if(simple){out[x]=(qc[x]+(fi?qi[x]:0.f)>1.e-6f)?1.f:0.f;return;}
 float tc=t[x]-273.15f;
 float esw=1000.f*.61078f*nd_exp(17.2693882f*tc/(t[x]-35.86f));
 float esi=1000.f*.61078f*nd_exp(21.8745584f*tc/(t[x]-7.66f));
 float ep2=287.f/461.6f;
 float qvsw=ep2*esw/(p[x]-esw),qvsi=ep2*esi/(p[x]-esi);
 float qcld,weight;
 if(fi&&fs){qcld=qi[x]+qc[x]+qs[x];weight=qcld<1.e-12f?0.f:(qi[x]+qs[x])/fmaxf(qcld,1.e-12f);}
 else if(fi){qcld=qi[x]+qc[x];weight=qcld<1.e-12f?0.f:qi[x]/fmaxf(qcld,1.e-12f);
  if(ferrier&&qcld>=1.e-12f&&tc< -40.f)weight=1.f;
 }else{qcld=qc[x];weight=qcld<1.e-12f?0.f:(t[x]>273.15f?0.f:1.f);}
 float qvs=(1.f-weight)*qvsw+weight*qvsi,rh=qv[x]/qvs;
 float subsat=fmaxf(1.e-10f,qvs-qv[x]);
 float arg=fmaxf(-6.9f,-100.f*qcld/nd_pow(subsat,.49f));
 float frac=nd_pow(fmaxf(1.e-10f,rh),.25f)*(1.f-nd_exp(arg));
 if(frac<.01f)frac=0.f;
 out[x]=qcld<1.e-12f?0.f:(rh>=1.f?1.f:frac);
}

__device__ float solar_height_bound(int k,int c,int n,const float* ph,const float* phb,int full){
 size_t x=(size_t)k*n+c;
 return ((ph[x]+phb[full?x:k])+(ph[x+n]+phb[full?x+n:k+1]))/(2.f*9.81f);
}
extern "C" __global__ void solar_diag_bound(const float* rho,const float* dz8w,const float* ph,const float* phb,const float* cldfrac3d,const float* qv,const float* qc,const float* qi,const float* qs,const float* qc_tot,const float* qi_tot,const float* re_cloud,const float* re_ice,const float* re_snow,const float* coszen,const float* swdnb,const float* swdnt,float* clrnidx,float* sza,float* ghi_accum,float* cldfrac2d,float* wvp2d,float* lwp2d,float* iwp2d,float* swp2d,float* wp2d_sum,float* lwp2d_tot,float* iwp2d_tot,float* wp2d_tot_sum,float* re_cloud_path,float* re_ice_path,float* re_snow_path,float* re_cloud_path_tot,float* re_ice_path_tot,float* tau_qc,float* tau_qi,float* tau_qs,float* tau_qc_tot,float* tau_qi_tot,float* cbase,float* ctop,float* cbase_tot,float* ctop_tot,int nz,int n,int first,int last,float dt,int fv,int fc,int fi,int fs,int hc,int hi,int hs,int full){
int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 sza[c]=(180.f/(4.f*0.7853981633974483f))*fmaxf(nd_acos(coszen[c]),0.f);
 clrnidx[c]=(sza[c]<90.f&&swdnt[c]>0.f)?swdnb[c]/swdnt[c]:-999.f;
 ghi_accum[c]=ghi_accum[c]+swdnb[c]*dt;
 float frac=cldfrac3d[(size_t)first*n+c];for(int k=first+1;k<last;++k)frac=fmaxf(frac,cldfrac3d[(size_t)k*n+c]);cldfrac2d[c]=frac;

 float v=0.f,l=0.f,i=0.f,s=0.f,lt=0.f,it=0.f;
 float rl=0.f,ri=0.f,rs=0.f,rlt=0.f,rit=0.f;
 for(int k=first;k<last;++k){size_t x=(size_t)k*n+c;
  float rhodz=rho[x]*dz8w[x]/(1.f+qv[x]);
  if(fv)v=v+qv[x]*rhodz;
  if(fc){l=l+qc[x]*rhodz;lt=lt+qc_tot[x]*rhodz;
   if(hc==1){float r=re_cloud[x]*1.e-6f;rl=rl+qc[x]*r*rhodz;rlt=rlt+qc_tot[x]*r*rhodz;}}
  if(fi){i=i+qi[x]*rhodz;it=it+qi_tot[x]*rhodz;
   if(hi==1){float r=re_ice[x]*1.e-6f;ri=ri+qi[x]*r*rhodz;rit=rit+qi_tot[x]*r*rhodz;}}
  if(fs){s=s+qs[x]*rhodz;if(hs==1)rs=rs+qs[x]*(re_snow[x]*1.e-6f)*rhodz;}
 }
 if(fv)wvp2d[c]=v;
if(fc){float q=l;lwp2d[c]=fabsf(fmaxf(q,0.f));
if(hc==1){float r=q>0.f?rl/q:0.f;re_cloud_path[c]=r;
tau_qc[c]=r>0.f?1.5f*q/r/1000.f:0.f;}
}else{lwp2d[c]=-999.f;re_cloud_path[c]=-999.f;tau_qc[c]=-999.f;}
if(fc){float q=lt;lwp2d_tot[c]=fabsf(fmaxf(q,0.f));
if(hc==1){float r=q>0.f?rlt/q:0.f;re_cloud_path_tot[c]=r;
tau_qc_tot[c]=r>0.f?1.5f*q/r/1000.f:0.f;}
}else{lwp2d_tot[c]=-999.f;re_cloud_path_tot[c]=-999.f;tau_qc_tot[c]=-999.f;}
if(fi){float q=i;iwp2d[c]=fabsf(fmaxf(q,0.f));
if(hi==1){float r=q>0.f?ri/q:0.f;re_ice_path[c]=r;
tau_qi[c]=r>0.f?q*1000.f*(.02f+4.2f/(2.f*r*1.e6f)):0.f;}
}else{iwp2d[c]=-999.f;re_ice_path[c]=-999.f;tau_qi[c]=-999.f;}
if(fi){float q=it;iwp2d_tot[c]=fabsf(fmaxf(q,0.f));
if(hi==1){float r=q>0.f?rit/q:0.f;re_ice_path_tot[c]=r;
tau_qi_tot[c]=r>0.f?q*1000.f*(.02f+4.2f/(2.f*r*1.e6f)):0.f;}
}else{iwp2d_tot[c]=-999.f;re_ice_path_tot[c]=-999.f;tau_qi_tot[c]=-999.f;}
if(fs){float q=s;swp2d[c]=fabsf(fmaxf(q,0.f));
if(hs==1){float r=q>0.f?rs/q:0.f;re_snow_path[c]=r;
tau_qs[c]=r>0.f?q*1000.f*(.02f+4.2f/(2.f*r*1.e6f)):0.f;}
}else{swp2d[c]=-999.f;re_snow_path[c]=-999.f;tau_qs[c]=-999.f;}
if(fc||fi||fs){
 wp2d_sum[c]=fmaxf(lwp2d[c],0.f)+fmaxf(iwp2d[c],0.f)+fmaxf(swp2d[c],0.f);
 wp2d_tot_sum[c]=fmaxf(lwp2d_tot[c],0.f)+fmaxf(iwp2d_tot[c],0.f)+fmaxf(swp2d[c],0.f);
if(wp2d_sum[c]>0.f){
 int k=first;while(solar_wc(k,c,n,qc,qi,qs,fc,fi,fs)<1.e-5f&&k<last)++k;
 cbase[c]=k==last?-999.f:solar_height_bound(k,c,n,ph,phb,full);
 k=last;while(solar_wc(k,c,n,qc,qi,qs,fc,fi,fs)<1.e-5f&&k>first)--k;
 ctop[c]=k==first?-999.f:solar_height_bound(k,c,n,ph,phb,full);
 }else{cbase[c]=-999.f;ctop[c]=-999.f;}
if(wp2d_tot_sum[c]>0.f){
 int k=first;while(solar_wc(k,c,n,qc_tot,qi_tot,qs,fc,fi,fs)<1.e-5f&&k<last)++k;
 cbase_tot[c]=k==last?-999.f:solar_height_bound(k,c,n,ph,phb,full);
 k=last;while(solar_wc(k,c,n,qc_tot,qi_tot,qs,fc,fi,fs)<1.e-5f&&k>first)--k;
 ctop_tot[c]=k==first?-999.f:solar_height_bound(k,c,n,ph,phb,full);
 }else{cbase_tot[c]=-999.f;ctop_tot[c]=-999.f;}
}else{wp2d_sum[c]=-999.f;wp2d_tot_sum[c]=-999.f;cbase[c]=-999.f;ctop[c]=-999.f;cbase_tot[c]=-999.f;ctop_tot[c]=-999.f;}
}
