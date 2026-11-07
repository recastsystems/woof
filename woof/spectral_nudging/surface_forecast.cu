// WRF v4.6.1 FDDAGD caller and calculate_phy_tend / phy_fdda_ten.
// WRF public-domain notice and source pins: surface_sources.json.
// This adapter keeps native C-grid momentum; no A-to-C interpolation.
__device__ float sf_phi(const float *a,const float *b,int k,int j,int i,
                       int ny,int nx,int terrain) {
    int q=(k*ny+j)*nx+i; return a[q]+b[terrain?q:k];
}
extern "C" __global__ void sf_clock(double *clock,float *factors,int *control,
    const double *offsets,int records,double dt,float interval,float end_hours,
    float upper_end_hours,int ramping,float ramp_minutes,int cadence,int zero) {
    if(threadIdx.x || blockIdx.x)return;
    double now=clock[0]; int step=(int)llround(now/dt)+1;
    float minutes=(float)(now/60.0);
    int slot=0;
    for(int s=1;s<records;s++) if(offsets[s]<=now)slot=s;
    control[0]=step==1 || step%cadence==0;
    control[1]=zero && (step-1)%cadence==0;
    control[2]=slot;
    float old=floorf(minutes/interval)*interval;
    factors[0]=(minutes-old)/interval;
    float last=end_hours*60.f;
    float actual=last+(ramping && ramp_minutes>0.f?fabsf(ramp_minutes):0.f);
    factors[1]=1.f;
    if(ramping && fabsf(ramp_minutes)>0.f) {
        if(minutes>=actual-fabsf(ramp_minutes) && minutes<=actual) {
            factors[1]=(actual-minutes)/fabsf(ramp_minutes);
            if(ramp_minutes>0.f)factors[0]=0.f;
        } else if(minutes>actual)factors[1]=0.f;
    }
    // The native outer early return tests both upper and surface windows.
    float upper=upper_end_hours*60.f;
    if(ramping && ramp_minutes>0.f)upper+=fabsf(ramp_minutes);
    float surface_outer=last+fabsf(ramp_minutes);
    if(minutes>upper && minutes>surface_outer)factors[1]=0.f;
    clock[0]=now+dt;
}

extern "C" __global__ void sf_select(const int *control,
    const float *analyses,const float *densities,float *old,float *next,
    float *density,int plane) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=plane)return;
    int slot=control[2];
    for(int f=0;f<4;f++) {
        old[f*plane+c]=analyses[((slot*2)*4+f)*plane+c];
        next[f*plane+c]=analyses[((slot*2+1)*4+f)*plane+c];
    }
    density[c]=densities[slot*2*plane+c];
    density[plane+c]=densities[(slot*2+1)*plane+c];
}

// module_fdda_psufddagd.F:254-261,531-569,621-666. The native upper
// contribution is added to SFDDAGD's raw tendencies before mass coupling.
extern "C" __global__ void sf_upper(const double *clock,const double *offsets,
    int records,const float *analyses,const float *u,const float *v,
    const float *theta,const float *qv,const float *wpbl,const float *wzfac,
    float *raw,int nz,int ny,int nx,double dt,float interval,float end_hours,
    float surface_end,int ramping,float ramp_minutes,float guv,float gt,float gq) {
    int c=blockIdx.x*blockDim.x+threadIdx.x,N=(ny+1)*(nx+1);
    if(c>=N)return;
    int j=c/(nx+1),i=c%(nx+1);
    if(j>=ny || i>=nx)return;
    double now=clock[0]-dt;
    float minutes=(float)(now/60.0),old=floorf(minutes/interval)*interval;
    float coef=(minutes-old)/interval,tfac=1.f;
    float actual=end_hours*60.f+(ramping && ramp_minutes>0.f?ramp_minutes:0.f);
    if(ramping && fabsf(ramp_minutes)>0.f) {
        if(minutes>=actual-fabsf(ramp_minutes) && minutes<=actual) {
            tfac=(actual-minutes)/fabsf(ramp_minutes);
            if(ramp_minutes>0.f)coef=0.f;
        } else if(minutes>actual)tfac=0.f;
    }
    if(minutes>actual && minutes>surface_end*60.f+fabsf(ramp_minutes))tfac=0.f;
    int slot=0,volume=nz*N;
    for(int s=1;s<records;s++)if(offsets[s]<=now)slot=s;
    for(int k=0;k<nz;k++) {
        int r=k*N+c;
        for(int f=0;f<4;f++) {
            if((f==0 && i==0) || (f==1 && j==0))continue;
            int a=((slot*2)*4+f)*volume+r,b=a+4*volume,q=f*volume+r;
            float target=analyses[a]*(1.f-coef)+analyses[b]*coef;
            float delta=f==0?target-u[r]:(f==1?target-v[r]:(f==2?target-theta[r]+300.f:target-qv[r]));
            float gain=f<2?guv:(f==2?gt:gq);
            raw[q]=raw[q]+gain*wpbl[q]*wzfac[f*nz+k]*tfac*delta;
        }
    }
}

extern "C" __global__ void sf_pack(const float *u,const float *v,
    const float *thp,const float *thb,const float *eos_p,const float *p_hyd,
    const float *qv,const float *php,const float *phb,const float *ht,
    const float *pbl,const float *regime,const float *znt,
    float *out_u,float *out_v,float *theta,float *temperature,float *out_qv,
    float *pressure,float *z,float *out_ht,float *out_regime,float *out_znt,
    float *wpbl,int nz,int ny,int nx,int terrain,int native_regime) {
    int c=blockIdx.x*blockDim.x+threadIdx.x,H=ny+1,W=nx+1,N=H*W;
    if(c>=N)return;
    int j=c/W,i=c%W,jm=min(j,ny-1),im=min(i,nx-1),m=jm*nx+im;
    out_ht[c]=ht[m];out_znt[c]=znt[m];
    float second=sf_phi(php,phb,1,jm,im,ny,nx,terrain)/9.81f-ht[m];
    out_regime[c]=native_regime?regime[m]:(pbl[m]>second?4.f:1.f);
    int kpbl[4]={1,1,1,1};
    for(int f=0;f<4;f++) {
        int ia=f==0?max(0,i-1):im,ib=im;
        int ja=f==1?max(0,j-1):jm,jb=jm;
        float depth=.5f*(pbl[ja*nx+ia]+pbl[jb*nx+ib]);
        for(int k=0;k<nz;k++) {
            float lo=.5f*((sf_phi(php,phb,k,ja,ia,ny,nx,terrain)/9.81f-ht[ja*nx+ia])
                        +(sf_phi(php,phb,k,jb,ib,ny,nx,terrain)/9.81f-ht[jb*nx+ib]));
            float hi=.5f*((sf_phi(php,phb,k+1,ja,ia,ny,nx,terrain)/9.81f-ht[ja*nx+ia])
                        +(sf_phi(php,phb,k+1,jb,ib,ny,nx,terrain)/9.81f-ht[jb*nx+ib]));
            if(depth>=lo && depth<hi){kpbl[f]=k+1;break;}
        }
    }
    for(int k=0;k<nz;k++) {
        int s=k*ny*nx+m,o=k*N+c;
        out_u[o]=u[(k*ny+jm)*(nx+1)+i];
        out_v[o]=v[(k*(ny+1)+j)*nx+im];
        float th=thp[s]+thb[terrain?s:k];
        theta[o]=th;
        temperature[o]=th*powf(eos_p[s]/100000.f,287.f/1004.5f);
        out_qv[o]=qv[s];pressure[o]=p_hyd[s];
        float lo=sf_phi(php,phb,k,jm,im,ny,nx,terrain)/9.81f;
        float hi=sf_phi(php,phb,k+1,jm,im,ny,nx,terrain)/9.81f;
        z[o]=.5f*(lo+hi);
        for(int f=0;f<4;f++)
            wpbl[f*nz*N+o]=(k+1<=kpbl[f]?0.f:(k+1==kpbl[f]+1?.1f:1.f));
    }
}

// The native arrays are mass-coupled IN PLACE after fddagd_driver, even
// between FDDA calls. Carry that representation, including FGDTZERO's
// native (ITIMESTEP-1) clearing predicate, through restart/checkpoint.
extern "C" __global__ void sf_couple(const int *control,const float *raw,
    const float *mup,const float *mub,const float *c1,const float *c2,
    const float *msfu,const float *msfv,const float *msft,
    float *held_u,float *held_v,float *held_t,float *held_q,
    float *ru,float *rv,float *rt,float *rq,int nz,int ny,int nx,int has_msf,
    int strict) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,nu=nz*ny*(nx+1),nv=nz*(ny+1)*nx;
    int n=nz*ny*nx,N=(ny+1)*(nx+1),volume=nz*N;
    bool recalc=control[0],clear=control[1];
    if(q<nu) {
        int i=q%(nx+1),j=q/(nx+1)%ny,k=q/((nx+1)*ny);
        float value=recalc?raw[k*N+j*(nx+1)+i]:(clear?0.f:held_u[q]);
        if(i>0 && i<nx) {
            int a=j*nx+i-1,b=a+1;
            float mu=.5f*((mup[a]+mub[a])+(mup[b]+mub[b]));
            value=(c1[k]*mu+c2[k])*value;
        }
        held_u[q]=value;
        if(i==0 || i==nx || j==0 || j==ny-1)value=0.f;
        ru[q]=has_msf && !strict?value/msfu[j*(nx+1)+i]:value;
    }
    if(q<nv) {
        int i=q%nx,j=q/nx%(ny+1),k=q/(nx*(ny+1));
        float value=recalc?raw[volume+k*N+j*(nx+1)+i]:(clear?0.f:held_v[q]);
        if(j>0 && j<ny) {
            int a=(j-1)*nx+i,b=a+nx;
            float mu=.5f*((mup[a]+mub[a])+(mup[b]+mub[b]));
            value=(c1[k]*mu+c2[k])*value;
        }
        held_v[q]=value;
        if(j==0 || j==ny || i==0 || i==nx-1)value=0.f;
        rv[q]=has_msf && !strict?value/msfv[j*nx+i]:value;
    }
    if(q<n) {
        int i=q%nx,j=q/nx%ny,k=q/(nx*ny),a=j*nx+i,r=k*N+j*(nx+1)+i;
        float mass=c1[k]*(mup[a]+mub[a])+c2[k];
        float t=recalc?raw[2*volume+r]:(clear?0.f:held_t[q]);
        float v=recalc?raw[3*volume+r]:(clear?0.f:held_q[q]);
        held_t[q]=t=mass*t;held_q[q]=v=mass*v;
        if(i==0 || i==nx-1 || j==0 || j==ny-1){t=0.f;v=0.f;}
        rt[q]=has_msf && !strict?t/msft[a]:t;rq[q]=v;
    }
}

extern "C" __global__ void sf_add(const float *au,const float *av,
    const float *at,const float *aq,const float *bu,const float *bv,
    const float *bt,const float *bq,float *u,float *v,float *t,float *q,
    int nu,int nv,int nt) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<nu)u[i]=au[i]+bu[i];
    if(i<nv)v[i]=av[i]+bv[i];
    if(i<nt){t[i]=at[i]+bt[i];q[i]=aq[i]+bq[i];}
}

// module_fdda_psufddagd.F:655: before mass coupling, updated at FDDA cadence.
extern "C" __global__ void sf_diagnostic(const int *control,const float *raw,
    const float *p,const float *theta,const float *php,const float *phb,
    float *hfx,int nz,int ny,int nx,int terrain) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,n=ny*nx;
    if(q>=n*nz || !control[0])return;
    int i=q%nx,j=q/nx%ny,k=q/n,N=(ny+1)*(nx+1),r=k*N+j*(nx+1)+i;
    float rho=p[r]/(287.f*theta[r]);
    float lo=sf_phi(php,phb,k,j,i,ny,nx,terrain)/9.81f;
    float hi=sf_phi(php,phb,k+1,j,i,ny,nx,terrain)/9.81f;
    hfx[q]=rho*1004.f*raw[2*nz*N+r]*(hi-lo);
}

// Surface driver (Noah arm) followed by module_sf_noahdrv.F's physical
// conversion. SDA survives FGDTZERO; only the nudging call replaces it.
extern "C" __global__ void sf_sda_commit(const int *control,const float *h,
 const float *q,float *held_h,float *held_q,int ny,int nx) {
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c>=ny*nx || !control[0])return;
 int src=(c/nx)*(nx+1)+c%nx; held_h[c]=h[src]; held_q[c]=q[src];
}
extern "C" __global__ void sf_noah_inputs(float *sda_h,float *sda_q,
 const float *norm,const float *rho,const float *cpm,const float *dz,
 const float *land,const float *hfx,const float *qfx,float *heat,float *moisture,
 float *both_h,float *both_q,int count) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=count)return;
 if(land[c]>1.5f){sda_h[c]=0.f;sda_q[c]=0.f;}
 float q=sda_q[c]*rho[c]*dz[c];
 both_q[c]=qfx[c]+q*norm[c];
 float h=sda_h[c]*rho[c]*cpm[c]*dz[c];
 both_h[c]=hfx[c]+h; heat[c]=h;moisture[c]=q;
}

// module_pbl_driver.F:1257-1267: the YSU arm alone updates QNORM.
extern "C" __global__ void sf_qnorm(const float *dqv,const float *qv,
 float *norm,float dt,int count) {
 int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=count)return;
 float x=__fdiv_rn(dqv[c]*(2.0f*dt),qv[c]);
 x=x*100.0f;x=fmaxf(0.0f,x);x=fminf(5.0f,x);
 norm[c]=__fdiv_rn(x,100.0f);
}
