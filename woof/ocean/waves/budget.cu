// Optional point budgets. Read-only with respect to the prognostic arrays.
// A thread owns one probe/frequency/direction bin; no reduction atomics.
extern "C" __global__ void wave_budget_capture(const float* n,const float* f,
 const int* cells,double* previous,double* totals,int probes,int slot){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=probes*NF*ND)return;
 int p=z/(NF*ND),b=z%(NF*ND);double e=(double)n[(long long)cells[p]*NF*ND+b]*(2*PI*f[b/ND]);
 if(slot>=0)totals[(long long)slot*probes*NF*ND+z]+=e-previous[z];
 previous[z]=e;
}
extern "C" __global__ void wave_budget_sources(const float* n,const float* freq,const float* m,
 const float* beta,const float* loss,const float* seed,const float* dia,const float* diagonal,
 const float* components,const int* cells,double* totals,int probes,int first,int nc,int nfx,float dt){
 int z=blockIdx.x*blockDim.x+threadIdx.x;if(z>=probes*NF*ND)return;
 int p=z/(NF*ND),b=z%(NF*ND),c=cells[p]-first,f=b/ND,d=b%ND;if(c<0||c>=nc)return;
 long long local=(long long)c*NF*ND+b,ext=((long long)c*nfx+f)*ND+d;
 const float* mm=m+c*16;float old=n[local]*(2*PI*freq[f]);
 float raw=seed[local]+(beta[local]-loss[local])*old+dia[ext];
 float den=1+dt*fmaxf(0.f,loss[local]-beta[local]-diagonal[ext]);
 float delta=dt*raw/den;
 float fm=mm[5]>1e-20f?mm[5]/fmaxf(mm[6],1e-30f):freq[NF-1]/2.5f;
 float cap=3e-7f*G*fmaxf(mm[7],G*.0056f/freq[f])*powf(freq[f],-4.f)*fm*dt;
 float limited=fmaxf(-old,fminf(cap,fmaxf(-cap,delta)));
 double weight=(double)dt/den*(delta!=0?(double)limited/delta:1.);
 float terms[8]={beta[local]*old,seed[local],dia[ext],0,0,0,0,0};
 for(int j=0;j<5;j++)terms[j+3]=-components[((long long)j*probes+p)*NF*ND+b]*old;
 for(int j=0;j<8;j++){
  totals[(long long)(7+j)*probes*NF*ND+z]+=(double)dt*terms[j];
  totals[(long long)(15+j)*probes*NF*ND+z]+=weight*terms[j];
 }
}
