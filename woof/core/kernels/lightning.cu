// WRF v4.6.1 module_lightning_driver and module_ltng_*.
// Strict replay retains the scalar source's float32 arithmetic. The fast
// runtime permits contraction and uses CUDA's float32 power function.
__device__ inline float lightning_pow(float x,float y) {
#ifdef GPUWM_WRF_EXACT
 return gfk_pow(x,y);
#else
 return powf(x,y);
#endif
}
// One cooperative block per level, then one cooperative patch reduction.
// Counts are integers until conversion back to WRF's REAL counter. Native
// repeated +1 stops advancing at 2**24, so retain that saturation explicitly.
extern "C" __global__ void lightning_levels(
 const float* refl,const float* w,int nz,int ny,int nx,
 int i0,int i1,int j0,int j1,float* levels) {
 int k=blockIdx.x,tid=threadIdx.x,n=ny*nx,width=i1-i0;
 int area=width*(j1-j0),count=0;
 float maximum=__int_as_float(0xff800000u);
 __shared__ int counts[256];
 __shared__ float maxima[256];
 for(int a=tid;a<area;a+=blockDim.x) {
   int c=(j0+a/width)*nx+i0+a%width,p=k*n+c;
   count+=(refl[p]>20.f);
   if(w[p]>maximum) maximum=w[p];
 }
 counts[tid]=count;maxima[tid]=maximum;__syncthreads();
 for(int stride=blockDim.x/2;stride;stride/=2) {
   if(tid<stride) {
     counts[tid]+=counts[tid+stride];
     if(maxima[tid+stride]>maxima[tid]) maxima[tid]=maxima[tid+stride];
   }
   __syncthreads();
 }
 if(tid==0) {
   levels[k]=(float)min(counts[0],16777216);
   levels[nz+k]=maxima[0];
 }
}
extern "C" __global__ void lightning_patch(
 const float* refl,const float* t,const float* z,
 const float* ht,const float* land,const float* levels,
 int nz,int ny,int nx,int option,int i0,int i1,int j0,int j1,float* summary) {
 int tid=threadIdx.x,n=ny*nx,kmax=0,ktop=0;
 float maxcount=0.f,wmax=__int_as_float(0xff800000u);
 __shared__ float counts[256],maxima[256],heights[256],lands[256];
 __shared__ int tops[256],indices[256],coldcounts[256];
 for(int k=tid;k<nz;k+=blockDim.x) {
   float count=levels[k];
   if(count>0.f) ktop=k;
   if(count>maxcount) {maxcount=count;kmax=k;}
   if(levels[nz+k]>wmax) wmax=levels[nz+k];
 }
 counts[tid]=maxcount;maxima[tid]=wmax;tops[tid]=ktop;indices[tid]=kmax;
 __syncthreads();
 for(int stride=blockDim.x/2;stride;stride/=2) {
   if(tid<stride) {
     int other=tid+stride;
     if(counts[other]>counts[tid] ||
        (counts[other]==counts[tid] && indices[other]<indices[tid])) {
       counts[tid]=counts[other];indices[tid]=indices[other];
     }
     tops[tid]=max(tops[tid],tops[other]);
     if(maxima[other]>maxima[tid]) maxima[tid]=maxima[other];
   }
   __syncthreads();
 }
 ktop=tops[0];
 float zmax=0.f,mostlyland=0.f;int coldcount=0;
 if(option==2 && counts[0]!=0.f) {
   int width=i1-i0,area=width*(j1-j0);
   for(int a=tid;a<area;a+=blockDim.x) {
     int c=(j0+a/width)*nx+i0+a%width,p=ktop*n+c;
     if(refl[p]>20.f && t[p]<273.15f) {
       if(z[p]-ht[c]>zmax) zmax=z[p]-ht[c];
       coldcount++;mostlyland+=land[c];
     }
   }
 }
 heights[tid]=zmax;lands[tid]=mostlyland;coldcounts[tid]=coldcount;
 __syncthreads();
 for(int stride=blockDim.x/2;stride;stride/=2) {
   if(tid<stride) {
     if(heights[tid+stride]>heights[tid]) heights[tid]=heights[tid+stride];
     lands[tid]+=lands[tid+stride];coldcounts[tid]+=coldcounts[tid+stride];
   }
   __syncthreads();
 }
 if(tid==0) {
   float total=0.f;
   if(counts[0]!=0.f) {
     if(option==1) total=5.7e-6f*lightning_pow(maxima[0],4.5f);
     else {
       // XLAND is WRF's categorical 1/2 field. The sum is exact while
       // its total is <= 2**24 (at most 2**23 all-water points).
       // Native's 0/0 on a warm top is deliberately retained.
       mostlyland=lands[0]/(float)coldcounts[0];
       zmax=heights[0]*1.e-3f;
       total=(mostlyland<1.5f ? 3.44e-5f:6.57e-6f)*lightning_pow(zmax,4.9f);
     }
     total=total/counts[0]/60.f;
   }
   summary[0]=total;summary[1]=(float)indices[0];
 }
}
extern "C" __global__ void lightning_columns(
 const float* refl,const float* w,const float* t,const float* z,
 const float* dz,const float* theta,const float* pi,
 const float* qc,const float* qr,const float* qi,const float* qs,const float* qg,
 const float* ht,const float* land,const float* lat,const float* lon,
 const int* top,const float* inputnum,const float* inputden,const float* summary,
 int nz,int ny,int nx,int option,int method,int month,
 float dx,float dy,float adjust,float num,float den,float factor,float interval,
 int i0,int i1,int j0,int j1,
 float* ic,float* cg,float* icc,float* cgc,float* lpi,int* invalid) {
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=ny*nx;
 if(c>=n) return;
 int j=c/nx,i=c%nx;
 if(i<i0||i>=i1||j<j0||j>=j1) return;
 if(option==3) {
   // calclpi builds cumulative heights, then calc_lpi subtracts them.
   float height=0.f,ave=0.f,totaldz=0.f;
   for(int k=0;k<nz-1;k++) {
     int p=k*n+c;
     float oldheight=height;height=height+dz[p];
     float delz=height-oldheight;
     // A contraction here changes which side of the 0/-20 C gates a
     // native-rounded temperature occupies. Keep the branch authority
     // rounded as WRF even in the fast runtime.
     float temperature=__fsub_rn(__fmul_rn(theta[p],pi[p]),273.16f);
     if(temperature<=0.f && temperature>-20.f) {
       float deni=qi[p]+qg[p],dens=qs[p]+qg[p],ni=0.f,ns=0.f;
       if(qs[p]<=1.e-10f||qg[p]<=1.e-10f) dens=10000.f;
       else ns=sqrtf(qs[p]*qg[p]);
       if(qi[p]<=1.e-10f||qg[p]<=1.e-10f) deni=10000.f;
       else ni=sqrtf(qi[p]*qg[p]);
       float ice=qg[p]*(ni/deni+ns/dens);
       float liquid=qc[p]+qr[p],a=0.f,b=10000.f;
       if(liquid>1.e-10f && ice!=0.f) {a=sqrtf(liquid*ice);b=liquid+ice;}
       float lower=w[p],upper=(k<nz-2 ? w[p+n]:0.f);
       float wave=0.5f*(upper+lower);
       totaldz=totaldz+delz;
       if(a>0.f) ave=ave+((delz*(2.f*a/b))*(wave*wave));
     }
   }
   if(totaldz==0.f) totaldz=100000.f;
   lpi[c]=ave/totaldz;
   return; // WRF LPI does not update any flash field.
 }
 float total=0.f; int kt=0;
 if(option==11) {
   kt=top[c]-1; // WRF cloud top uses one-based levels.
   if(kt < -1 || kt>=nz) {atomicOr(invalid,1);return;}
   if(kt>=0 && t[kt*n+c]<273.15f) {
     float zkm=(z[kt*n+c]-ht[c])/1.e3f+adjust;
     if(zkm>0.f) total=(land[c]<1.5f ? 3.44e-5f:6.57e-6f)*lightning_pow(zkm,4.9f)/60.f;
   }
   float area=dx*dy/1.e6f;
   total=total*area/1296.f;
 } else {
   kt=(int)summary[1];
   if(refl[kt*n+c]>20.f) total=summary[0];
 }
 float ir=0.f,cr=0.f;
 if(method==1) {
   if(den==0.f) ir=total;
   else if(num==0.f) cr=total;
   else if(total!=0.f) {float ratio=num/den;cr=total*(1.f/(ratio+1.f));ir=total-cr;}
 } else if(method==0 || method==2) {
   if(total>0.f) {
     if(lat[c]<25.f||lat[c]>55.f||lon[c]<-120.f||lon[c]>-70.f) {
       if(den!=0.f) {float ratio=num/den;cr=total*(1.f/(ratio+1.f));}
     } else cr=total*((lon[c]>-90.f||lon[c]<-105.f) ? .4f:.22f);
     ir=total-cr;
   }
 } else if(method==3) {
   if(total>0.f) {
     if(option!=11) {
       kt=nz-1;
       while(refl[kt*n+c]<20.f && kt>0) kt--;
     }
     int kfreeze=kt;
     while(t[kfreeze*n+c]<273.15f && kfreeze>0) kfreeze--;
     if(option==11 && t[kfreeze*n+c]<273.15f) {atomicOr(invalid,2);return;}
     float depth=(z[kt*n+c]-z[kfreeze*n+c])*1.e-3f;
     if(option==11) depth=depth+adjust;
     depth=fmaxf(5.5f,fminf(14.f,depth));
     float ratio=(((.021f*depth-.648f)*depth+7.493f)*depth-36.54f)*depth+63.09f;
     cr=total*(1.f/(ratio+1.f));ir=total-cr;
   }
 } else {
   int p=(month-1)*n+c;
   // WRF method 4 overwrites its nominal 0/0 fallback with all CG.
   if(inputden[p]==0.f) cr=total;
   else {
     float ratio=inputnum[p]/inputden[p];
     if(ratio==-1.f) {atomicOr(invalid,4);return;}
     cr=total*(1.f/(ratio+1.f));
   }
   ir=total-cr;
 }
 ir=ir*factor;cr=cr*factor;ic[c]=ir;cg[c]=cr;
 icc[c]=icc[c]+ir*interval;cgc[c]=cgc[c]+cr*interval;
}
