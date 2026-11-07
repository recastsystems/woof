// Public-domain WRF v4.6.1 module_advect_em.F advect_scalar/u/v.
// Native fifth-order horizontal and third/fifth-order vertical ladders.
// Longitude inputs include three ghost columns plus the east duplicate;
// meridional fluxes are one-sided at poles, with no cross-pole wrap.
static __device__ __forceinline__ size_t ga3(int k,int j,int i,int ny,int nx) {
    return ((size_t)k*ny+j)*(nx+7)+(i+3);
}
static __device__ __forceinline__ float ga5(
    float a,float b,float c,float d,float e,float f,float velocity,bool vertical=false) {
    float center=(37.0f*(d+c)-8.0f*(e+b)+(f+a))/60.0f;
    float sign=copysignf(1.0f,vertical?-velocity:velocity);
    return velocity*(center-sign*((f-a)-5.0f*(e-b)+10.0f*(d-c))/60.0f);
}
static __device__ __forceinline__ float ga3flux(
    float a,float b,float c,float d,float velocity,bool vertical=false) {
    float center=(7.0f*(c+b)-(d+a))/12.0f;
    float sign=copysignf(1.0f,vertical?-velocity:velocity);
    return velocity*(center+sign*((d-a)-3.0f*(c-b))/12.0f);
}
static __device__ __forceinline__ float ga_x(
    const float* q,float vel,int k,int j,int face,int ny,int nx) {
    return ga5(q[ga3(k,j,face-3,ny,nx)],q[ga3(k,j,face-2,ny,nx)],
               q[ga3(k,j,face-1,ny,nx)],q[ga3(k,j,face,ny,nx)],
               q[ga3(k,j,face+1,ny,nx)],q[ga3(k,j,face+2,ny,nx)],vel);
}
static __device__ __forceinline__ float ga_y(
    const float* q,float vel,int k,int face,int i,int nys,int nx) {
    // The outside faces of scalar/u volumes never enter the polar sum.
    if(face<=0 || face>=nys) return 0.0f;
    int d=min(face,nys-face);
    if(d==1) return 0.5f*vel*(q[ga3(k,face,i,nys,nx)]+q[ga3(k,face-1,i,nys,nx)]);
    if(d==2) return ga3flux(q[ga3(k,face-2,i,nys,nx)],q[ga3(k,face-1,i,nys,nx)],
                            q[ga3(k,face,i,nys,nx)],q[ga3(k,face+1,i,nys,nx)],vel);
    return ga5(q[ga3(k,face-3,i,nys,nx)],q[ga3(k,face-2,i,nys,nx)],
               q[ga3(k,face-1,i,nys,nx)],q[ga3(k,face,i,nys,nx)],
               q[ga3(k,face+1,i,nys,nx)],q[ga3(k,face+2,i,nys,nx)],vel);
}
static __device__ __forceinline__ float ga_z(
    const float* q,float vel,int k,int j,int i,int nz,int nys,int nx,
    const float* fnm,const float* fnp,int order) {
    if(k==0 || k==nz) return 0.0f;
    if(k==1 || k==nz-1)
        return vel*(fnm[k]*q[ga3(k,j,i,nys,nx)]+fnp[k]*q[ga3(k-1,j,i,nys,nx)]);
    if(order==5 && k>=3 && k<=nz-3)
        return ga5(q[ga3(k-3,j,i,nys,nx)],q[ga3(k-2,j,i,nys,nx)],
                   q[ga3(k-1,j,i,nys,nx)],q[ga3(k,j,i,nys,nx)],
                   q[ga3(k+1,j,i,nys,nx)],q[ga3(k+2,j,i,nys,nx)],vel,true);
    return ga3flux(q[ga3(k-2,j,i,nys,nx)],q[ga3(k-1,j,i,nys,nx)],
                   q[ga3(k,j,i,nys,nx)],q[ga3(k+1,j,i,nys,nx)],vel,true);
}

// kind: 0 scalar, 1 u, 2 v. Each field is native (not total-theta shifted).
extern "C" __global__ void global_advect(
    int nx,int ny,int nz,int kind,int vorder,float rdx,float rdy,
    const float* q,const float* ru,const float* rv,const float* ww,
    const float* msf,const float* vx,const float* rdzw,
    const float* fnm,const float* fnp,float* tend) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int nxs=nx+(kind==1),nys=ny+(kind==2);
    if(tid>=nz*nys*nxs) return;
    int i=tid%nxs,j=(tid/nxs)%nys,k=tid/(nxs*nys);
    // Native advect_v prescribes zero pole tendencies and excludes those
    // faces from every horizontal and vertical flux divergence loop.
    if(kind==2 && (j==0 || j==ny)) { tend[tid]=0.0f; return; }
    float fy[2],fx[2],fz[2];
    for(int s=0;s<2;++s) {
        int g=j+s,f=i+s,kf=k+s;
        float velx,vely,velz;
        if(kind==1) {
            velx=0.5f*(ru[ga3(k,j,f,ny,nx)]+ru[ga3(k,j,f-1,ny,nx)]);
            vely=0.5f*(rv[ga3(k,g,i,ny+1,nx)]+rv[ga3(k,g,i-1,ny+1,nx)]);
            velz=0.5f*(ww[ga3(kf,j,i,ny,nx)]+ww[ga3(kf,j,i-1,ny,nx)]);
        } else if(kind==2) {
            velx=0.5f*(ru[ga3(k,j,f,ny,nx)]+ru[ga3(k,j-1,f,ny,nx)]);
            vely=0.5f*(rv[ga3(k,g,i,ny+1,nx)]+rv[ga3(k,g-1,i,ny+1,nx)]);
            velz=0.5f*(ww[ga3(kf,j,i,ny,nx)]+ww[ga3(kf,j-1,i,ny,nx)]);
        } else {
            velx=ru[ga3(k,j,f,ny,nx)];
            vely=rv[ga3(k,g,i,ny+1,nx)];
            velz=ww[ga3(kf,j,i,ny,nx)];
        }
        fx[s]=ga_x(q,velx,k,j,f,nys,nx);
        fy[s]=ga_y(q,vely,k,g,i,nys,nx);
        fz[s]=ga_z(q,velz,kf,j,i,nz,nys,nx,fnm,fnp,vorder);
    }
    float value=tend[tid],m=msf[(size_t)j*nxs+i];
    if(kind!=2 && j==0) value=value-(m*rdy)*fy[1];
    else if(kind!=2 && j==ny-1) value=value+(m*rdy)*fy[0];
    else value=value-(m*rdy)*(fy[1]-fy[0]);
    value=value-(m*rdx)*(fx[1]-fx[0]);
    if(kind==2) value=value-(m/vx[(size_t)j*nx+i])*rdzw[k]*(fz[1]-fz[0]);
    else value=value-rdzw[k]*(fz[1]-fz[0]);
    tend[tid]=value;
}

// Fill native-sized longitude read halos in parallel. The packed east u
// duplicate is retained at its own index; all outside columns are periodic.
extern "C" __global__ void global_zonal_pack(
    int nx,int ny,int nz,int stagger_x,const float* packed,float* halo) {
    int q=blockIdx.x*blockDim.x+threadIdx.x,width=nx+7;
    if(q>=nz*ny*width) return;
    int i=q%width-3,row=q/width;
    int src=(stagger_x && i==nx)?nx:((i%nx)+nx)%nx;
    halo[q]=packed[(size_t)row*(nx+stagger_x)+src];
}
