// Setup/coupling glue for public-domain WRF observation FDDA.
// Included after kernels.cu. All clocks and time-window decisions stay on device.
extern "C" __global__ void fddaobs_stamp(unsigned long long*start,
 unsigned long long*sum,unsigned long long*count,int stage,int finish){
 unsigned long long now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
 if(finish){sum[stage]+=now-start[stage];++count[stage];}else start[stage]=now;
}
extern "C" __global__ void fddaobs_publish(float*time,float seconds,float dt){time[0]=seconds;time[1]=dt;}
extern "C" __global__ void fddaobs_clock(long long*step,Frame*f,const float*time,
 const float*nlev,const float*lev,int count,int capacity,float dt,float begin,
 float end,float window,int dyn,float ramp,int interval,const float*published){
 if(threadIdx.x||blockIdx.x)return;
 int k=int(*step);bool first=f->ktau<0;f->ktau=k;f->xtime=published[0]/60.f;f->dtmin=published[1]/60.f;
 f->update=first||k%interval==0;
 f->enabled=f->xtime>=begin&&f->xtime<=end;
 f->tfaci=1.f;
 if(dyn){float r=fabsf(ramp);f->tfaci=f->xtime<end-r?1.f:f->xtime<=end?(end-f->xtime)*(1.f/r):0.f;}
 if(f->update){
  float back=f->xtime/60.f-window,forward=f->xtime/60.f+window;
  int lo=0,hi=count;while(lo<hi){int mid=(lo+hi)/2;if(time[mid]<back)lo=mid+1;else hi=mid;}f->first=lo;
  lo=0;hi=count;while(lo<hi){int mid=(lo+hi)/2;if(time[mid]<=forward)lo=mid+1;else hi=mid;}
  f->last=lo<count?min(count,lo+int(nlev[lo]-lev[lo])+1):count;
  if(f->last-f->first>capacity){f->enabled=-1;f->tfaci=__int_as_float(0x7fc00000);}
 }
 ++*step;
}
extern "C" __global__ void fddaobs_pack(
 const float*u,const float*v,const float*thp,const float*thb,const float*q,
 const float*p,const float*basep,const float*ppnative,const float*ph,const float*phb,
 const float*terrain,const float*mub,const float*mup,const float*msft,
 const float*msfu,const float*msfv,const float*u10,const float*v10,const float*th2,
 const float*regime,const float*pblh,const int*kpbl,
 float*ub,float*vb,float*tb,float*qv,float*pb,float*pp,float*z,
 float*ur,float*vr,float*tr,float*mu,float*muu,float*muv,float*mx,float*my,
 float*mux,float*muy,float*vx,float*vy,float*ht,float*rg,float*pbl,int*kp,
 float*au,float*av,float*at,float*aq,int nx,int ny,int nk,int base3,int pb3,
 int phb3,int exactp,int native_regime,int sfclay,int stepbl,const Frame*frame){
 int a=int(blockIdx.x*blockDim.x+threadIdx.x),nxy=(nx+1)*(ny+1);if(a>=nxy*nk)return;
 int i=a%(nx+1),k=(a/(nx+1))%nk,j=a/((nx+1)*nk),ii=min(i,nx-1),jj=min(j,ny-1);
 int h=jj*nx+ii,b=(k*ny+jj)*nx+ii,fh=i+(nx+1)*j;
 ub[a]=u[(k*ny+jj)*(nx+1)+i];vb[a]=v[(k*(ny+1)+j)*nx+ii];
 float theta=thp[b]+thb[base3?b:k];tb[a]=theta-300.f;qv[a]=q[b];
 pb[a]=basep[pb3?b:k];pp[a]=exactp?ppnative[b]:p[b]-pb[a];
 int w0=(k*ny+jj)*nx+ii,w1=w0+ny*nx;
 z[a]=.5f*(ph[w0]+phb[phb3?w0:k]+ph[w1]+phb[phb3?w1:k+1])/9.81f;
 au[a]=0.f;av[a]=0.f;at[a]=0.f;aq[a]=0.f;
 if(k==0){
  float ua=.5f*(u[jj*(nx+1)+ii]+u[jj*(nx+1)+ii+1]);
  float va=.5f*(v[jj*nx+ii]+v[(jj+1)*nx+ii]);
  if(frame->ktau==0||(frame->ktau+1)%stepbl==0){
  ur[fh]=!sfclay?0.f:fabsf(u10[h])>1.e-10f?ua/u10[h]:1.2f;
  vr[fh]=!sfclay?0.f:fabsf(v10[h])>1.e-10f?va/v10[h]:1.2f;
  float temp=theta*powf(p[b]/100000.f,287.f/1004.5f);
  tr[fh]=!sfclay?0.f:temp*powf(100000.f*.001f/(p[b]/1000.f),287.f/1004.5f)/th2[h];
  }
  mu[fh]=mub[h]+mup[h];ht[fh]=terrain[h];pbl[fh]=pblh[h];kp[fh]=kpbl[h];
  int left=jj*nx+max(0,i-1),south=max(0,j-1)*nx+ii;
  muu[fh]=i==0||i==nx?mu[fh]:.5f*(mub[h]+mup[h]+mub[left]+mup[left]);
  muv[fh]=j==0||j==ny?mu[fh]:.5f*(mub[h]+mup[h]+mub[south]+mup[south]);
  mx[fh]=msft[h];my[fh]=msft[h];
  mux[fh]=msfu[jj*(nx+1)+i];muy[fh]=mux[fh];
  vx[fh]=msfv[j*nx+ii];vy[fh]=vx[fh];
  rg[fh]=native_regime?regime[h]:(pblh[h]>(ph[ny*nx+h]+phb[phb3?ny*nx+h:1])/9.81f-terrain[h]?4.f:1.f);
 }
}
extern "C" __global__ void fddaobs_fold(
 const float*au,const float*av,const float*at,const float*aq,
 float*ru,float*rv,float*rt,float*rq,int nx,int ny,int nk){
 int a=int(blockIdx.x*blockDim.x+threadIdx.x);
 int nm=nx*ny*nk,nu=(nx+1)*ny*nk,nv=nx*(ny+1)*nk;
 if(a<nu){int i=a%(nx+1),j=a/(nx+1)%ny,k=a/((nx+1)*ny);ru[a]+=au[i+(nx+1)*(k+nk*j)];}
 if(a<nv){int i=a%nx,j=a/nx%(ny+1),k=a/(nx*(ny+1));rv[a]+=av[i+(nx+1)*(k+nk*j)];}
 if(a<nm){int i=a%nx,j=a/nx%ny,k=a/(nx*ny),b=i+(nx+1)*(k+nk*j);rt[a]+=at[b];rq[a]+=aq[b];}
}
