// NOAA-EMC/HRRR 40ee6058, phys/module_diag_gsd.F:724-813.
// Retained fork extension (prec_acc_dt1); the stock selector uses the
// independently pinned WRF 4.7.1 module_diag_misc.F:415-448 entry below.
extern "C" __global__ void precipitation_buckets(
 const real* rainc,const real* rain,const real* snow,const real* graup,
 const real* t2,real* pc,real* pn,real* sn,real* gn,int count,int reset) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
 real c=rainc[i];
 pc[i]=fmaxf(0.,(reset?0.:pc[i])+c);
 pn[i]=fmaxf(0.,(reset?0.:pn[i])+rain[i]);
 sn[i]=fmaxf(0.,(reset?0.:sn[i])+snow[i]+(t2[i]<273.15f?c:0.f));
 gn[i]=fmaxf(0.,(reset?0.:gn[i])+graup[i]);
}

extern "C" __global__ void native_precipitation_buckets(
 const real* rainc,const real* rain,const real* snow,const real* graup,
 const real* t2,real* pc,real* pn,real* sn,real* gn,
 const real* epoch,int count,double now,real minutes) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
 // Native real_time subtracts domain_get_start_time before converting
 // to default REAL. The epoch is split into two exact float32 parts so
 // native restart origins retain their fraction without a device read.
 real seconds=(real)(now-((double)epoch[0]+(double)epoch[1]));
 bool reset=fmodf(seconds,60.f*minutes)==0.f;
 if(reset){pc[i]=0.f;pn[i]=0.f;sn[i]=0.f;gn[i]=0.f;}
 pc[i]=fmaxf(pc[i]+rainc[i],0.f);
 pn[i]=fmaxf(pn[i]+rain[i],0.f);
 sn[i]=sn[i]+snow[i];
 if(t2[i]<273.15f){sn[i]=sn[i]+rainc[i];sn[i]=fmaxf(sn[i],0.f);}
 // Additional graupel output retained from the operational fork.
 gn[i]=fmaxf(gn[i]+graup[i],0.f);
}

// WRF f52c197e phys/module_diag_misc.F:229-291, separate FP32 product and sum.
__device__ __forceinline__ void native_tendency_add(
 real* acc,const real* rate,int i,real dt,bool reset) {
 real a=reset?0.f:acc[i];
 real r=rate?rate[i]:0.f;
 acc[i]=a+r*dt;
}

// WRF f52c197e phys/module_diag_misc.F:229-291. Native inactive CU/SH
// arrays keep their previous values even at reset; they are not cleared.
extern "C" __global__ void native_physics_tendency_accumulation(
 const real* r0,const real* r1,const real* r2,const real* r3,
 const real* r4,const real* r5,const real* r6,const real* r7,
 const real* r8,const real* r9,const real* r10,const real* r11,
 const real* r12,const real* r13,const real* r14,const real* r15,
 real* a0,real* a1,real* a2,real* a3,real* a4,real* a5,
 real* a6,real* a7,real* a8,real* a9,real* a10,real* a11,
 real* a12,real* a13,real* a14,real* a15,
 const real* qv,const real* before,const real* epoch,
 int count,int nx,int ny,int sz,int mp,int cu,int sh,
 double now,real interval,real dt) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 real seconds=(real)(now-((double)epoch[0]+(double)epoch[1]));
 bool reset=fmodf(seconds,interval)==0.f;
 native_tendency_add(a0,r0,i,dt,reset);
 // Replay passes real native qv_diabatic directly; production reconstructs
 // the same difference/division from actual pre/post-MP vapor, before any
 // prognostic transport, with the native excluded-ring and no-heating rules.
 real qr=0.f;
 if(r1)qr=r1[i];
 else if(mp){
  int x=i%nx,y=(i/nx)%ny;
  if(x>=sz && x<nx-sz && y>=sz && y<ny-sz)qr=(qv[i]-before[i])/dt;
 }
 a1[i]=(reset?0.f:a1[i])+qr*dt;
 if(cu==1){
  native_tendency_add(a2,r2,i,dt,reset);native_tendency_add(a3,r3,i,dt,reset);
  native_tendency_add(a4,r4,i,dt,reset);native_tendency_add(a5,r5,i,dt,reset);
 }
 if(sh==1){
  native_tendency_add(a6,r6,i,dt,reset);native_tendency_add(a7,r7,i,dt,reset);
  native_tendency_add(a8,r8,i,dt,reset);native_tendency_add(a9,r9,i,dt,reset);
 }
 native_tendency_add(a10,r10,i,dt,reset);native_tendency_add(a11,r11,i,dt,reset);
 native_tendency_add(a12,r12,i,dt,reset);native_tendency_add(a13,r13,i,dt,reset);
 native_tendency_add(a14,r14,i,dt,reset);native_tendency_add(a15,r15,i,dt,reset);
}

// WRF f52c197e phys/module_diag_misc.F:296-431, unchanged arithmetic.
extern "C" __global__ void native_accumulation_buckets(
 real* a0, real* a1, real* a2, real* a3, real* a4, real* a5,
 real* a6, real* a7, real* a8, real* a9, real* a10, real* a11,
 real* a12, real* a13, real* a14, real* a15, real* a16, real* a17,
 int* c0, int* c1, int* c2, int* c3, int* c4, int* c5,
 int* c6, int* c7, int* c8, int* c9, int* c10, int* c11,
 int* c12, int* c13, int* c14, int* c15, int* c16, int* c17,
 const real* epoch,int count,double now,real xtime,real mm,real joules) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 real seconds=(real)(now-((double)epoch[0]+(double)epoch[1]));
 if(fmodf(roundf(seconds),3600.f)!=0.f)return;
 // Test the shared clock once per column. Inactive calls visit one grid,
 // rather than launching eighteen grids that all return on the same test.
 #pragma unroll
 for(int field=0;field<18;++field){
 real* amount=0;int* counter=0;
 switch(field){
 case 0: amount=a0;counter=c0;break;
 case 1: amount=a1;counter=c1;break;
 case 2: amount=a2;counter=c2;break;
 case 3: amount=a3;counter=c3;break;
 case 4: amount=a4;counter=c4;break;
 case 5: amount=a5;counter=c5;break;
 case 6: amount=a6;counter=c6;break;
 case 7: amount=a7;counter=c7;break;
 case 8: amount=a8;counter=c8;break;
 case 9: amount=a9;counter=c9;break;
 case 10: amount=a10;counter=c10;break;
 case 11: amount=a11;counter=c11;break;
 case 12: amount=a12;counter=c12;break;
 case 13: amount=a13;counter=c13;break;
 case 14: amount=a14;counter=c14;break;
 case 15: amount=a15;counter=c15;break;
 case 16: amount=a16;counter=c16;break;
 case 17: amount=a17;counter=c17;break;
 }
 if(!amount || !counter)continue;
 real threshold=field<2?mm:joules;
 if(xtime==0.f)counter[i]=0;
 if(amount[i]>threshold){amount[i]=amount[i]-threshold;counter[i]=counter[i]+1;}
 }
}
