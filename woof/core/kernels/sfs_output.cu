// WRF 4.7.1 module_diffusion_em.F cal_titau_* stress output arithmetic.
// REAL32 evaluation order includes the density multiply and divide in
// off-diagonal stresses; algebraic cancellation changes native words.
__device__ __forceinline__ real sfs_normal(real km, real deform)
{
    return -km * deform;
}

__device__ __forceinline__ real sfs_shear(real weighted_km, real rho, real deform)
{
    return __fdiv_rn(-weighted_km * deform, rho);
}

extern "C" __global__
void wrf_stress_output_raw(const real* km, const real* rho, const real* deform,
                          const real* fnm, const real* fnp, real* out,
                          int nz, int ny, int nx, int mode,
                          int i0, int i1, int j0, int j1, int k0, int k1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    if (i < i0 || i > i1 || j < j0 || j > j1 || k < k0 || k > k1) return;
    size_t at = I3S(k,j,i,ny,nx);
    if (mode == 0) {
        out[at] = sfs_normal(km[at], deform[at]);
        return;
    }
    size_t a, b, c, d;
    real density, coefficient;
    if (mode == 1) {
        a=I3S(k,j,i-1,ny,nx); b=at;
        c=I3S(k,j-1,i-1,ny,nx); d=I3S(k,j-1,i,ny,nx);
        density=0.25f*(rho[a]+rho[b]+rho[c]+rho[d]);
        coefficient=density*0.25f*(km[a]+km[b]+km[c]+km[d]);
    } else {
        a=at; b=mode==2 ? I3S(k,j,i-1,ny,nx) : I3S(k,j-1,i,ny,nx);
        c=I3S(k-1,j,i,ny,nx);
        d=mode==2 ? I3S(k-1,j,i-1,ny,nx) : I3S(k-1,j-1,i,ny,nx);
        // Native xz density sums the western point first; yz does not.
        density=mode==2
            ? 0.5f*(fnm[k]*(rho[b]+rho[a])+fnp[k]*(rho[d]+rho[c]))
            : 0.5f*(fnm[k]*(rho[a]+rho[b])+fnp[k]*(rho[c]+rho[d]));
        coefficient=density*0.5f*(fnm[k]*(km[a]+km[b])+fnp[k]*(km[c]+km[d]));
    }
    out[at] = sfs_shear(coefficient, density, deform[at]);
}

// Registry/registry.les stores all six history fields on the mass array,
// including the interior vertical-interface values of m13/m23.
extern "C" __global__
void wrf_stress_output(const real* u,const real* v,const real* w,
                      const real* php,const real* phb,const real* alt,const real* qv,
                      const real* msft,const real* msfu,const real* msfv,
                      const real* fnm,const real* fnp,const real* dn,const real* dnw,
                      real rdx,real rdy,real dx,real dy,real cf1,real cf2,real cf3,
                      int moist, const real* kmh, const real* kmv,
                      const real* ustm, int surface_active, int isfflx, real cd0,
                      real* m11, real* m22, real* m33,
                      real* m12, real* m13, real* m23,
                      int vertical, int nz, int ny, int nx, int phb3d,
                      int boundary_x, int boundary_y)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y, k=blockIdx.z;
    if (i>=nx || j>=ny || k>=nz) return;
    WrfSmagGrid q=wrf_grid(u,v,w,php,phb,alt,qv,msft,msfu,msfv,
        fnm,fnp,dn,dnw,rdx,rdy,dx,dy,cf1,cf2,cf3,nz,ny,nx,phb3d,
        boundary_x,boundary_y,moist);
    size_t at=I3S(k,j,i,ny,nx);
    bool inner_x=!boundary_x || i>0;
    bool inner_y=!boundary_y || j>0;
    if(vertical && k==0) {
        int ic=wrf_ix(q,i), im=wrf_ix(q,i-1);
        int jc=wrf_iy(q,j), jm=wrf_iy(q,j-1);
        real uu=u[I3S(0,j,wrf_iu(q,i),ny,nx+1)];
        real vv=(v[I3S(0,wrf_jv(q,j),ic,ny+1,nx)]
                +v[I3S(0,wrf_jv(q,j+1),ic,ny+1,nx)]
                +v[I3S(0,wrf_jv(q,j),im,ny+1,nx)]
                +v[I3S(0,wrf_jv(q,j+1),im,ny+1,nx)])/4.0f;
        real speed=sqrtf(uu*uu+vv*vv)+1.e-15f;
        real us=0.5f*(ustm[jc*nx+ic]+ustm[jc*nx+im]);
        m13[at]=isfflx==0 ? -(cd0*speed*uu)
            : surface_active ? -__fdiv_rn(us*us*uu,speed) : -0.0f;
        vv=v[I3S(0,wrf_jv(q,j),i,ny+1,nx)];
        uu=(u[I3S(0,jc,wrf_iu(q,i),ny,nx+1)]
           +u[I3S(0,jm,wrf_iu(q,i),ny,nx+1)]
           +u[I3S(0,jc,wrf_iu(q,i+1),ny,nx+1)]
           +u[I3S(0,jm,wrf_iu(q,i+1),ny,nx+1)])/4.0f;
        speed=sqrtf(vv*vv+uu*uu)+1.e-15f;
        us=0.5f*(ustm[jc*nx+ic]+ustm[jm*nx+ic]);
        m23[at]=isfflx==0 ? -(cd0*speed*vv)
            : surface_active ? -__fdiv_rn(us*us*vv,speed) : -0.0f;
    }
    if(inner_y) m11[at]=sfs_normal(wrf_k(q,kmh,k,j,i),wrf_defor11(q,k,j,i));
    if(inner_x) m22[at]=sfs_normal(wrf_k(q,kmh,k,j,i),wrf_defor22(q,k,j,i));
    if(inner_x && inner_y) {
        if(vertical) m33[at]=sfs_normal(wrf_k(q,kmh,k,j,i),
            2.0f*(w[I3S(k+1,j,i,ny,nx)]-w[at])*wrf_rdzw(q,k,j,i));
        real density=0.25f*(wrf_rho(q,k,j,i-1)+wrf_rho(q,k,j,i)
            +wrf_rho(q,k,j-1,i-1)+wrf_rho(q,k,j-1,i));
        real coefficient=density*0.25f*(wrf_k(q,kmh,k,j,i-1)+wrf_k(q,kmh,k,j,i)
            +wrf_k(q,kmh,k,j-1,i-1)+wrf_k(q,kmh,k,j-1,i));
        m12[at]=sfs_shear(coefficient,density,wrf_defor12(q,k,j,i));
        if(k>0) {
            density=0.5f*(fnm[k]*(wrf_rho(q,k,j,i-1)+wrf_rho(q,k,j,i))
                +fnp[k]*(wrf_rho(q,k-1,j,i-1)+wrf_rho(q,k-1,j,i)));
            coefficient=density*0.5f*(fnm[k]*(wrf_k(q,kmv,k,j,i)+wrf_k(q,kmv,k,j,i-1))
                +fnp[k]*(wrf_k(q,kmv,k-1,j,i)+wrf_k(q,kmv,k-1,j,i-1)));
            m13[at]=sfs_shear(coefficient,density,wrf_defor13(q,k,j,i));
            density=0.5f*(fnm[k]*(wrf_rho(q,k,j,i)+wrf_rho(q,k,j-1,i))
                +fnp[k]*(wrf_rho(q,k-1,j,i)+wrf_rho(q,k-1,j-1,i)));
            coefficient=density*0.5f*(fnm[k]*(wrf_k(q,kmv,k,j,i)+wrf_k(q,kmv,k,j-1,i))
                +fnp[k]*(wrf_k(q,kmv,k-1,j,i)+wrf_k(q,kmv,k-1,j-1,i)));
            m23[at]=sfs_shear(coefficient,density,wrf_defor23(q,k,j,i));
        }
    }
}
