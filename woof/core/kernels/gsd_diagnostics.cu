// NOAA-EMC/HRRR 40ee6058, phys/module_diag_gsd.F:817-1647.
// One thread per column; reductions over the domain use parallel device kernels.
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))
__device__ real gd_t(const real* thp,const real* thb,int thb3,const real* p,const real* qv,int k,int c,int n){
 real theta=A(thp[k*n+c],thb[thb3?k*n+c:k]);
#ifdef GPUWM_USE_THETA_M
 theta=D(theta,A(1.f,M(D(461.6f,287.f),qv[k*n+c])));
#endif
 return M(theta,gfk_pow(M(p[k*n+c],1.e-5f),.28589641f));
}
__device__ real gd_rho(const real* alt,const real* qv,int i){return M(D(1.f,alt[i]),A(1.f,qv[i]));}
__device__ real gd_z(const real* ph,const real* phb,int phb3,int k,int c,int n){
 return D(A(ph[k*n+c],phb[phb3?k*n+c:k]),9.81f);
}
// phy_prep (module_big_step_utilities_em.F:5119-5201), held at time t.
extern "C" __global__ void gsd_prepare(
 const real* u,const real* v,const real* ph,const real* phb,int phb3,
 const real* thp,const real* thb,int thb3,const real* p,const real* alt,
 const real* qv,const real* fnm,const real* fnp,
 real* pw,real* temperature,real* rho,real* wind10,real* wind80,
 int nx,int ny,int nz,int diagflag){
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny;if(c>=n)return;
 int y=c/nx,x=c%nx;
 real previous=0.f,previous_z=0.f,ter=gd_z(ph,phb,phb3,0,c,n);
 if(diagflag){wind10[c]=-99.f;wind80[c]=-99.f;}
 for(int k=0;k<nz;++k){
  int t=k*n+c;
  real theta=A(thp[t],thb[thb3?t:k]);
#ifdef GPUWM_USE_THETA_M
  theta=D(theta,A(1.f,M(D(461.6f,287.f),qv[t])));
#endif
  temperature[t]=M(theta,gfk_pow(D(p[t],100000.f),.2857142857142857f));
  rho[t]=gd_rho(alt,qv,t);
  if(k>0)pw[t]=A(M(fnm[k],p[t]),M(fnp[k],p[t-n]));
  if(diagflag){
   real zm=M(.5f,A(gd_z(ph,phb,phb3,k,c,n),gd_z(ph,phb,phb3,k+1,c,n)));
   real uu=M(.5f,A(u[(k*ny+y)*(nx+1)+x],u[(k*ny+y)*(nx+1)+x+1]));
   real vv=M(.5f,A(v[(k*(ny+1)+y)*nx+x],v[(k*(ny+1)+y+1)*nx+x]));
   real sp=sqrtf(A(M(uu,uu),M(vv,vv)));
   real targets[2]={10.f,80.f};real* winds[2]={wind10,wind80};
   for(int h=0;h<2;++h)if(S(zm,ter)>targets[h]&&winds[h][c]==-99.f){
    real wt=k==0?0.f:fminf(1.f,fmaxf(0.f,D(S(S(zm,ter),targets[h]),S(zm,previous_z))));
    winds[h][c]=A(M(S(1.f,wt),sp),M(wt,previous));
   }
   previous=sp;previous_z=zm;
  }
 }
 real z0=gd_z(ph,phb,phb3,0,c,n);
 real z1=M(.5f,A(z0,gd_z(ph,phb,phb3,1,c,n)));
 real z2=M(.5f,A(gd_z(ph,phb,phb3,1,c,n),gd_z(ph,phb,phb3,2,c,n)));
 real a=D(S(z0,z2),S(z1,z2));pw[c]=A(M(a,p[c]),M(S(1.f,a),p[n+c]));
 z0=gd_z(ph,phb,phb3,nz,c,n);
 z1=M(.5f,A(z0,gd_z(ph,phb,phb3,nz-1,c,n)));
 z2=M(.5f,A(gd_z(ph,phb,phb3,nz-1,c,n),gd_z(ph,phb,phb3,nz-2,c,n)));
 a=D(S(z0,z2),S(z1,z2));
 pw[nz*n+c]=gfk_exp(A(M(a,gfk_log(p[(nz-1)*n+c])),M(S(1.f,a),gfk_log(p[(nz-2)*n+c]))));
}
__device__ real gd_refl(real qr,real qs,real qg,real rho){
 real r=qr>0.f?M(M(gfk_pow(M(qr,rho),1.75f),3.630803e-9f),1.e18f):0.f;
 real s=qs>0.f?M(M(gfk_pow(M(qs,rho),1.75f),2.185e-10f),1.e18f):0.f;
 real g=qg>0.f?M(M(gfk_pow(M(qg,rho),1.75f),1.033267e-9f),1.e18f):0.f;
 real z=A(A(r,s),g);if(z>0.f)z=M(10.f,gfk_log10(z));return fmaxf(-10.f,z);
}
// Fork Thompson hail: 50 logarithmic bins, alpha=1, 0.0005 m^-3 tail.
// The modern stock PSD uses another intercept relation and an analytic tail.
__device__ real gd_hail(real mass){
 if(mass<1.e-6f)return 0.f;
 real xam=M(D(3.1415926536f,6.f),500.f);
 real zans=A(2.5f,M(D(2.f,7.f),A(gfk_log10(mass),7.f)));
 zans=fmaxf(2.f,fminf(zans,7.f));
 double n0=(double)gfk_pow(10.f,zans);
 double lam_exp=glibc_pow(n0*(double)xam*6./(double)mass,(double)D(1.f,4.f));
 double lamg=lam_exp*(double)gfk_pow(D(D(24.f,1.f),6.f),D(1.f,3.f));
 double n0g=n0/(1.*lam_exp)*lamg*lamg;
 real number=(real)(n0g*1.*glibc_pow(lamg,-2.));
 // The source rounds this parenthesized REAL expression before **.
 real ratio=D(M(M(D(M(xam,24.f),1.f),number),1.f),mass);
 lamg=(double)gfk_pow(ratio,D(1.f,3.f));
 n0g=(double)D(number,1.f)*lamg*lamg;
 double sum=0.,previous=0.,diam=0.,nextdiam=0.;int bin;
 for(bin=50;bin>=1;--bin){
  diam=gsd_hail_d[bin-1];
  sum+=n0g*diam*glibc_exp(-lamg*diam)*gsd_hail_width[bin-1];
  if(sum>.0005)break;
  previous=sum;nextdiam=diam;
 }
 if(bin>=50)return (real)diam;
 if(nextdiam>.001)return (real)(diam-(sum-.0005)/(sum-previous)*(diam-nextdiam));
 return 1.e-4f;
}
extern "C" __global__ void gsd_columns(
 const real* u,const real* v,const real* w,const real* ph,const real* phb,int phb3,
 const real* thp,const real* thb,int thb3,const real* p,const real* alt,
 const real* qv,const real* qc,const real* qr,const real* qi,const real* qs,const real* qg,const real* qh,
 const real* znw,const real* density,const real* temperature,const real* wind10,const real* wind80,
 const real* msft,const real* qcbl,const real* cfbl,const real* u10,const real* v10,
 real* pw,
 real* wspdmax,real* umax,real* vmax,real* wup,real* wdn,
 real* wmean,real* wsum,real* count,real* grplmax,real* refmax,real* refm10,
 real* ltg1max,real* ltg2max,real* ltg1,real* ltg2,real* wspd10,real* wspd80,
 real* lwp,real* iwp,real* maxcf,real* hailk1,real* hail2d,
 int nx,int ny,int nz,real ptop,int diagflag,int icloud_bl,int thompson){
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny;if(c>=n)return;
 int y=c/nx,x=c%nx;
 real wc=0.f,nc=0.f,grpl=0.f,ice=0.f,meanprev=0.f,prevspd=0.f;
 real ter=gd_z(ph,phb,phb3,0,c,n),lw=0.f,iw=0.f,cf=0.f;
 bool threat=false,minus10=false;
 for(int k=0;k<=nz;++k){
  int t=k*n+c;
  if(pw[t]>10000.f){if(w[t]>wup[c])wup[c]=w[t];if(w[t]<wdn[c])wdn[c]=w[t];}
  if(znw[k]>=.5f&&znw[k]<=.8f){wc=A(wc,w[t]);nc=A(nc,1.f);}
  if(k==nz)break;
  real zlo=gd_z(ph,phb,phb3,k,c,n),zhi=gd_z(ph,phb,phb3,k+1,c,n);
  real depth=S(zhi,zlo),rho=density[t];
  grpl=A(grpl,M(M(qg[t],depth),rho));
  ice=A(ice,M(M(A(A(qi[t],qs[t]),qg[t]),depth),rho));
  real temp=gd_t(thp,thb,thb3,p,qv,k,c,n);
  if(!threat&&k+1<nz){
   real tb=gd_t(thp,thb,thb3,p,qv,k+1,c,n);
   if(M(A(temp,tb),.5f)<258.15f){
    threat=true;real val=D(M(M(M(M(.042f,1000.f),1.22f),w[t]),M(A(qg[t],qg[t+n]),.5f)),msft[c]);
    if(val<1.5f)val=0.f;ltg1[c]=val;if(val>ltg1max[c])ltg1max[c]=val;
   }
  }
  // Find the highest -10 C crossing by allowing later qualifying levels
  // to replace earlier ones (the authority walks the levels downward).
  if(k>0){
   real ta=gd_t(thp,thb,thb3,p,qv,k-1,c,n);
   if(temp<=263.15f&&ta>=263.15f){
    real a=D(S(263.15f,ta),S(temp,ta));
    real rr=A(qr[t-n],M(a,S(qr[t],qr[t-n])));
    real ss=A(qs[t-n],M(a,S(qs[t],qs[t-n])));
    real gg=A(qg[t-n],M(a,S(qg[t],qg[t-n])));
    real ra=density[t-n],r=A(ra,M(a,S(rho,ra)));
    meanprev=gd_refl(rr,ss,gg,r);minus10=true;
   }
  }
  if(thompson){
   real size=gd_hail(fmaxf(1.e-12f,M(qg[t],rho)));
   if(k==0&&size>hailk1[c])hailk1[c]=size;if(size>hail2d[c])hail2d[c]=size;
  }
  if(diagflag){
   real qct=qc[t],qit=qi[t];
   if(icloud_bl>0&&qc[t]<1.e-6f&&qi[t]<1.e-8f&&cfbl[t]>.01f){
    real cb=qcbl[t]<1.e-9f?0.f:qcbl[t],cl=cfbl[t]<.01f?0.f:cfbl[t];
    real fraction=fminf(1.f,fmaxf(0.f,D(S(temperature[t],254.f),15.f)));
    qct=M(M(cb,fraction),cl);qit=M(M(cb,S(1.f,fraction)),cl);
   }
   real dp=S(pw[t],pw[t+n]);
   lw=A(lw,fmaxf(M(D(dp,9.81f),qct),0.f));iw=A(iw,fmaxf(M(D(dp,9.81f),qit),0.f));
   cf=fmaxf(cf,cfbl[t]);
  }
 }
 real speed=sqrtf(A(M(u10[c],u10[c]),M(v10[c],v10[c])));
 if(speed>wspdmax[c]){wspdmax[c]=speed;umax[c]=u10[c];vmax[c]=v10[c];}
 count[c]=A(count[c],1.f);wsum[c]=A(wsum[c],D(wc,nc));wmean[c]=D(wsum[c],count[c]);
 if(grpl>grplmax[c])grplmax[c]=grpl;
 real threat2=D(M(M(.20f,1.22f),ice),msft[c]);if(threat2<M(.40f,1.22f))threat2=0.f;
 ltg2[c]=threat2;if(threat2>ltg2max[c])ltg2max[c]=threat2;
 real ref=gd_refl(qr[c],qs[c],qg[c],density[c]);if(ref>refmax[c])refmax[c]=ref;
 if(minus10&&meanprev>refm10[c])refm10[c]=meanprev;
 if(diagflag){
  lwp[c]=M(lw,1000.f);iwp[c]=M(iw,1000.f);maxcf[c]=cf;
  if(wspd10[c]==-99.f)wspd10[c]=wind10[c];if(wspd80[c]==-99.f)wspd80[c]=wind80[c];
 }
}
extern "C" __global__ void gsd_lightning3(const real* a,const real* b,real* out,
 const real* ma,const real* mb,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 real value=M(.95f,a[i]);if(mb[0]>0.f)value=A(value,M(M(.05f,D(ma[0],mb[0])),b[i]));
 out[i]=value<M(.02f,1.22f)?0.f:value;
}

extern "C" __global__ void gsd_reflectivity(real* refl,const real* phb,int phb3,
 const real* ht,const real* rainc,real* composite,real* at1,real* at4,
 int nx,int ny,int nz,real dt){
 int c=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny;if(c>=n)return;
 real rate=D(M(rainc[c],3600.f),dt),conv=M(300.f,gfk_pow(rate,1.4f));
 composite[c]=-10.f;at1[c]=-10.f;at4[c]=-10.f;bool found1=false,found4=false;
 for(int k=0;k<nz;++k){
  int t=k*n+c;real h=S(D(phb[phb3?t:k],9.81f),ht[c]);
  real ze=gfk_pow(10.f,M(.1f,refl[t]));real dbz=fmaxf(-10.f,M(10.f,gfk_log10(A(ze,conv))));
  refl[t]=dbz;if(!found1&&h>1000.f){at1[c]=dbz;found1=true;}
  if(!found4&&h>4000.f){at4[c]=dbz;found4=true;}composite[c]=fmaxf(composite[c],dbz);
 }
}

extern "C" __global__ void gsd_hail_probe(const real* mass,real* out,int n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)out[c]=gd_hail(mass[c]);
}
