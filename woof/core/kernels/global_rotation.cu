// WRF v4.6.1 module_big_step_utilities_em.F: coriolis and polar curvature.
// Public-domain WRF source; pin and provenance: licenses/NOTICE-WRF-HS31.txt.
// Distinct directional metrics and native one-sided polar loop bounds.
// Inputs are native coupled momenta. Each routine retains its own store.
static __device__ __forceinline__ size_t rot3(int k,int j,int i,int ny,int nx) {
    return ((size_t)k*ny+j)*nx+i;
}
static __device__ __forceinline__ float rot_rv(
    const float* rv,const float* halo,int k,int j,int i,int nx,int ny) {
    return i==nx ? halo[4*ny+k*(ny+1)+j] : rv[rot3(k,j,i,ny+1,nx)];
}
static __device__ __forceinline__ float rot_rw(
    const float* rw,const float* halo,int k,int j,int i,int nx,int ny,int nz) {
    return i==nx ? halo[4*ny+nz*(ny+1)+k*ny+j] : rw[rot3(k,j,i,ny,nx)];
}

extern "C" __global__ void global_rotation_halo(
    int nx,int ny,int nz,const float* f,const float* e,const float* cosa,
    const float* latitude,const float* rv,const float* rw,float* halo) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q<ny) {
        halo[q]=f[q*nx]; halo[ny+q]=e[q*nx];
        halo[2*ny+q]=cosa[q*nx]; halo[3*ny+q]=latitude[q*nx];
    }
    if(q<nz*(ny+1)) halo[4*ny+q]=rv[(size_t)q*nx];
    if(q<(nz+1)*ny) halo[4*ny+nz*(ny+1)+q]=rw[(size_t)q*nx];
}

extern "C" __global__ void global_coriolis(
    int nx,int ny,int nz,const float* ru,const float* rv,const float* rw,
    const float* tx,const float* ty,const float* ux,const float* uy,
    const float* vx,const float* vy,const float* f,const float* e,
    const float* sina,const float* cosa,const float* fnm,const float* fnp,const float* halo,
    float* rut,float* rvt,float* rwt)
{
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int nxf=nx+1,nyf=ny+1;
    if(q>=nz*nyf*nxf) return;
    int i=q%nxf,j=(q/nxf)%nyf,k=q/(nxf*nyf);
    if(j<ny) {
        int ip=i,im=i==0?nx-1:i-1;
        size_t a=(size_t)j*nx+(i==nx?0:i),b=(size_t)j*nx+im,h=(size_t)j*nxf+i;
        float fa=i==nx?halo[j]:f[a],ea=i==nx?halo[ny+j]:e[a];
        float ca=i==nx?halo[2*ny+j]:cosa[a];
        size_t o=rot3(k,j,i,ny,nxf);
        rut[o]=rut[o]+(ux[h]/uy[h])*0.5f*(fa+f[b])*0.25f*
            (rv[rot3(k,j+1,im,nyf,nx)]+rot_rv(rv,halo,k,j+1,ip,nx,ny)+
             rv[rot3(k,j,im,nyf,nx)]+rot_rv(rv,halo,k,j,ip,nx,ny))
            -0.5f*(ea+e[b])*0.5f*(ca+cosa[b])*0.25f*
            (rw[rot3(k+1,j,im,ny,nx)]+rw[rot3(k,j,im,ny,nx)]+
             rot_rw(rw,halo,k+1,j,ip,nx,ny,nz)+rot_rw(rw,halo,k,j,ip,nx,ny,nz));
    }
    if(i<nx && j>0 && j<ny) {
        size_t a=(size_t)j*nx+i,b=(size_t)(j-1)*nx+i,h=(size_t)j*nx+i;
        size_t o=rot3(k,j,i,nyf,nx);
        rvt[o]=rvt[o]-(vy[h]/vx[h])*0.5f*(f[a]+f[b])*0.25f*
            (ru[rot3(k,j,i,ny,nxf)]+ru[rot3(k,j,i+1,ny,nxf)]+
             ru[rot3(k,j-1,i,ny,nxf)]+ru[rot3(k,j-1,i+1,ny,nxf)])
            +(vy[h]/vx[h])*0.5f*(e[a]+e[b])*0.5f*(sina[a]+sina[b])*0.25f*
            (rw[rot3(k+1,j-1,i,ny,nx)]+rw[rot3(k,j-1,i,ny,nx)]+
             rw[rot3(k+1,j,i,ny,nx)]+rw[rot3(k,j,i,ny,nx)]);
    }
    if(i<nx && j<ny && k>0) {
        size_t h=(size_t)j*nx+i,o=rot3(k,j,i,ny,nx);
        rwt[o]=rwt[o]+e[h]*(cosa[h]*0.5f*
            (fnm[k]*(ru[rot3(k,j,i,ny,nxf)]+ru[rot3(k,j,i+1,ny,nxf)])+
             fnp[k]*(ru[rot3(k-1,j,i,ny,nxf)]+ru[rot3(k-1,j,i+1,ny,nxf)]))
            -(tx[h]/ty[h])*sina[h]*0.5f*
            (fnm[k]*(rv[rot3(k,j,i,nyf,nx)]+rv[rot3(k,j+1,i,nyf,nx)])+
             fnp[k]*(rv[rot3(k-1,j,i,nyf,nx)]+rv[rot3(k-1,j+1,i,nyf,nx)])));
    }
}

extern "C" __global__ void global_curvature(
    int nx,int ny,int nz,float reradius,float degrad,
    const float* ru,const float* rv,const float* rw,const float* u,const float* v,
    const float* tx,const float* ty,const float* ux,const float* uy,
    const float* vx,const float* vy,const float* latitude,const float* fnm,const float* fnp,const float* halo,
    float* rut,float* rvt,float* rwt)
{
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    int nxf=nx+1,nyf=ny+1;
    if(q>=nz*nyf*nxf) return;
    int i=q%nxf,j=(q/nxf)%nyf,k=q/(nxf*nyf);
    if(j<ny) {
        int ip=i,im=i==0?nx-1:i-1;
        size_t h=(size_t)j*nxf+i,o=rot3(k,j,i,ny,nxf);
        rut[o]=rut[o]+u[o]*reradius*((ux[h]/uy[h])*0.25f*
            (rv[rot3(k,j+1,im,nyf,nx)]+rot_rv(rv,halo,k,j+1,ip,nx,ny)+
             rv[rot3(k,j,im,nyf,nx)]+rot_rv(rv,halo,k,j,ip,nx,ny))*
            glibc_tanf((i==nx?halo[3*ny+j]:latitude[(size_t)j*nx+ip])*degrad)-0.25f*
            (rw[rot3(k+1,j,im,ny,nx)]+rw[rot3(k,j,im,ny,nx)]+
             rot_rw(rw,halo,k+1,j,ip,nx,ny,nz)+rot_rw(rw,halo,k,j,ip,nx,ny,nz)));
    }
    if(i<nx && j>0 && j<ny) {
        size_t a=(size_t)j*nx+i,b=(size_t)(j-1)*nx+i,o=rot3(k,j,i,nyf,nx);
        rvt[o]=rvt[o]-(vy[a]/vx[a])*reradius*(0.25f*
            (u[rot3(k,j,i,ny,nxf)]+u[rot3(k,j,i+1,ny,nxf)]+
             u[rot3(k,j-1,i,ny,nxf)]+u[rot3(k,j-1,i+1,ny,nxf)])*
            glibc_tanf((latitude[a]+latitude[b])*0.5f*degrad)*0.25f*
            (ru[rot3(k,j,i,ny,nxf)]+ru[rot3(k,j,i+1,ny,nxf)]+
             ru[rot3(k,j-1,i,ny,nxf)]+ru[rot3(k,j-1,i+1,ny,nxf)])+
            v[o]*0.25f*(rw[rot3(k+1,j-1,i,ny,nx)]+rw[rot3(k,j-1,i,ny,nx)]+
                         rw[rot3(k+1,j,i,ny,nx)]+rw[rot3(k,j,i,ny,nx)]));
    }
    if(i<nx && j<ny && k>0) {
        size_t h=(size_t)j*nx+i,o=rot3(k,j,i,ny,nx);
        rwt[o]=rwt[o]+reradius*(0.5f*
            (fnm[k]*(ru[rot3(k,j,i,ny,nxf)]+ru[rot3(k,j,i+1,ny,nxf)])+
             fnp[k]*(ru[rot3(k-1,j,i,ny,nxf)]+ru[rot3(k-1,j,i+1,ny,nxf)]))*0.5f*
            (fnm[k]*(u[rot3(k,j,i,ny,nxf)]+u[rot3(k,j,i+1,ny,nxf)])+
             fnp[k]*(u[rot3(k-1,j,i,ny,nxf)]+u[rot3(k-1,j,i+1,ny,nxf)]))+
            (tx[h]/ty[h])*0.5f*
            (fnm[k]*(rv[rot3(k,j,i,nyf,nx)]+rv[rot3(k,j+1,i,nyf,nx)])+
             fnp[k]*(rv[rot3(k-1,j,i,nyf,nx)]+rv[rot3(k-1,j+1,i,nyf,nx)]))*0.5f*
            (fnm[k]*(v[rot3(k,j,i,nyf,nx)]+v[rot3(k,j+1,i,nyf,nx)])+
             fnp[k]*(v[rot3(k-1,j,i,nyf,nx)]+v[rot3(k-1,j+1,i,nyf,nx)])));
    }
}
