// WRF module_sf_sfcdiags.F:50-72, UA on the shared no-FTZ loader.
extern "C" __global__ void noah_ua_sfcdiags(const float* ps,const float* ts,
 const float* ch,const float* ch2,const float* cq2,const float* qs,
 const float* qv,const float* t1,float* q2,float* t2,float* th2,float rcp,long long n) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float q=qs[i],t=ts[i];
 if(cq2[i]>=1.e-5f)q=__fsub_rn(q,__fmul_rn(__fdiv_rn(ch[i],cq2[i]),__fsub_rn(q,qv[i])));
 if(ch2[i]>=1.e-5f)t=__fsub_rn(t,__fmul_rn(__fdiv_rn(ch[i],ch2[i]),__fsub_rn(t,t1[i])));
 q2[i]=q;t2[i]=t;th2[i]=__fmul_rn(t,gfk_pow(__fdiv_rn(1.e5f,ps[i]),rcp));
}
