// Public-domain WRF v4.6.1 / v4.7.1 phys/module_diag_cl.F.
// Preserve its Celsius accumulation, NINT/MOD clock, reset timestamps,
// untouched reset wind components, and INTEGER incrementing counter.
__device__ inline int clwrf_nint(float x) {
 int whole=__float2int_rz(x);
 float fraction=x-(float)whole;
 return whole+(fraction>=.5f?1:(fraction<=-.5f?-1:0));
}
__device__ inline void clwrf_finish(float& mean,float& sd,int n) {
 mean=mean/n;sd=sqrtf(fmaxf(sd/n-mean*mean,0.f));
}
__device__ inline void clwrf_six(float x,float tt,bool reset,bool finish,int n,
 float& lo,float& hi,float& mean,float& sd,float& tlo,float& thi,bool temperature) {
 if(reset) {lo=x;hi=x;mean=x;sd=x*x;tlo=tt;thi=tt;}
 else {
  if(x<lo){lo=x;tlo=tt;}if(x>hi){hi=x;thi=tt;}
  mean=mean+x;sd=sd+x*x;
  if(finish){clwrf_finish(mean,sd,n);if(temperature){mean=mean+273.f;lo=lo+273.f;hi=hi+273.f;}}
 }
}
__device__ inline void clwrf_four(float x,float tt,bool reset,bool finish,int n,
 float& hi,float& mean,float& sd,float& thi) {
 if(reset){hi=x;mean=x;sd=x*x;thi=tt;}
 else{if(x>hi){hi=x;thi=tt;}mean=mean+x;sd=sd+x*x;
  if(finish)clwrf_finish(mean,sd,n);}
}
extern "C" __global__ void clwrf_output_calc(
 const float* t2,const float* q2,const float* tsk,const float* u10,const float* v10,
 const float* raincv,const float* rainncv,
 float* t2min,float* t2max,float* t2mean,float* t2std,float* tt2min,float* tt2max,
 float* q2min,float* q2max,float* q2mean,float* q2std,float* tq2min,float* tq2max,
 float* skintempmin,float* skintempmax,float* skintempmean,float* skintempstd,
 float* tskintempmin,float* tskintempmax,
 float* u10max,float* v10max,float* spduv10max,float* tspduv10max,
 float* u10mean,float* v10mean,float* spduv10mean,float* u10std,float* v10std,float* spduv10std,
 float* raincmax,float* raincmean,float* raincstd,float* traincmax,
 float* rainncmax,float* rainncmean,float* rainncstd,float* trainncmax,
 int* nsteps,int count,float dt,float xtime,float seconds,int interval) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int period=clwrf_nint(interval*60.f/dt);
 bool reset=clwrf_nint(seconds/dt)%period==0;
 bool finish=clwrf_nint((seconds+dt)/dt)%period==0;
 int n=reset?1:nsteps[i]+1;nsteps[i]=n;
 float tt=xtime+dt/60.f;
 clwrf_six(t2[i]-273.f,tt,reset,finish,n,t2min[i],t2max[i],t2mean[i],t2std[i],tt2min[i],tt2max[i],true);
 clwrf_six(q2[i],tt,reset,finish,n,q2min[i],q2max[i],q2mean[i],q2std[i],tq2min[i],tq2max[i],false);
 clwrf_six(tsk[i]-273.f,tt,reset,finish,n,skintempmin[i],skintempmax[i],skintempmean[i],skintempstd[i],tskintempmin[i],tskintempmax[i],true);
 clwrf_four(raincv[i]/dt,tt,reset,finish,n,raincmax[i],raincmean[i],raincstd[i],traincmax[i]);
 clwrf_four(rainncv[i]/dt,tt,reset,finish,n,rainncmax[i],rainncmean[i],rainncstd[i],trainncmax[i]);
 float u=u10[i],v=v10[i],speed2=u*u+v*v,speed=sqrtf(speed2);
 if(reset){
  spduv10max[i]=speed;tspduv10max[i]=tt;
  u10mean[i]=u;v10mean[i]=v;spduv10mean[i]=speed;
  u10std[i]=u*u;v10std[i]=v*v;spduv10std[i]=speed2;
 }else{
  if(speed>spduv10max[i]){u10max[i]=u;v10max[i]=v;spduv10max[i]=speed;tspduv10max[i]=tt;}
  spduv10mean[i]=spduv10mean[i]+speed;spduv10std[i]=spduv10std[i]+speed*speed;
  u10mean[i]=u10mean[i]+u;v10mean[i]=v10mean[i]+v;
  u10std[i]=u10std[i]+u*u;v10std[i]=v10std[i]+v*v;
  if(finish){clwrf_finish(u10mean[i],u10std[i],n);clwrf_finish(v10mean[i],v10std[i],n);clwrf_finish(spduv10mean[i],spduv10std[i],n);}
 }
}
