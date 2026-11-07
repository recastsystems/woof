// WRF v4.6.1 calc_p_rho_phi hydrostatic branch, public domain.
// Source pin and full notice: licenses/NOTICE-WRF-HS31.txt.
// One thread per column, with the native downward pressure and upward
// geopotential recurrences. All inputs and outputs remain on the device.
extern "C" __global__ void hydrostatic_vertical_velocity(
    int nx, int ny, int nz, const float *constants, const float *profiles,
    const float *mut, const float *msftx, const float *msfty,
    const float *height_halo, const float *u, const float *v,
    const float *ph_tend, const float *ph_new, const float *ph_old, float *w) {
    int xy=blockIdx.x*blockDim.x+threadIdx.x, plane=nx*ny;
    if (xy>=plane) return;
    int j=xy/nx, i=xy%nx, h=(j+1)*(nx+2)+i+1;
    int ui=j*(nx+1)+i, vi=j*nx+i, up=ny*(nx+1), vp=(ny+1)*nx;
    float g=constants[0], dt=constants[1], cf1=constants[2];
    float cf2=constants[3], cf3=constants[4], rdx=constants[5], rdy=constants[6];
    w[xy]=msfty[xy]*0.5f*rdy*(
        (height_halo[h+nx+2]-height_halo[h])
        *(cf1*v[vi+nx]+cf2*v[vi+nx+vp]+cf3*v[vi+nx+2*vp])
        +(height_halo[h]-height_halo[h-nx-2])
        *(cf1*v[vi]+cf2*v[vi+vp]+cf3*v[vi+2*vp]))
        +msftx[xy]*0.5f*rdx*(
        (height_halo[h+1]-height_halo[h])
        *(cf1*u[ui+1]+cf2*u[ui+1+up]+cf3*u[ui+1+2*up])
        +(height_halo[h]-height_halo[h-1])
        *(cf1*u[ui]+cf2*u[ui+up]+cf3*u[ui+2*up]));
    for (int k=1;k<=nz;++k) {
        int at=k*plane+xy;
        w[at]=msfty[xy]*((ph_new[at]-ph_old[at])/dt
            -ph_tend[at]/(profiles[k]*mut[xy]+profiles[nz+1+k]))/g;
    }
}

extern "C" __global__ void hydrostatic_acoustic(
    int nx, int ny, int nz, int step, float t0, float smdiv,
    const float *profiles, const float *mu, const float *mut,
    const float *alt, const float *theta_coupled, const float *theta_ref,
    const float *c2a, float *p, float *al, float *ph, float *pm1) {
    int xy=blockIdx.x*blockDim.x+threadIdx.x, plane=nx*ny;
    if (xy>=plane) return;
    const float *c1h=profiles, *c2h=profiles+nz;
    const float *c3h=profiles+2*nz, *dnw=profiles+3*nz;
    for (int k=0;k<nz;++k) {
        int at=k*plane+xy;
        float pressure=mu[xy]*c3h[k];
        float alpha=alt[at]*(theta_coupled[at]-(c1h[k]*mu[xy])*theta_ref[at])
            /((c1h[k]*mut[xy]+c2h[k])*(t0+theta_ref[at]))-pressure/c2a[at];
        ph[at+plane]=ph[at]-dnw[k]*((c1h[k]*mut[xy]+c2h[k])*alpha
                                  +(c1h[k]*mu[xy])*alt[at]);
        al[at]=alpha;
        p[at]=(step==0 ? pressure : pressure+smdiv*(pressure-pm1[at]));
        pm1[at]=pressure;
    }
}

extern "C" __global__ void hydrostatic_diagnostics(
    int nx, int ny, int nz, int nmoist, int first_scalar, int qv_index,
    int use_theta_m, int hypso, int j0, int i0, int nyw, int nxw,
    const float *constants, const float *profiles,
    const float *mu, const float *muts, const float *alb,
    const float *pb, const float *t, const float *phb,
    const float *moist, float *p, float *al, float *ph,
    float *full_pressure, float *alt) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col>=nyw*nxw) return;
    int j=j0+col/nxw, i=i0+col%nxw, xy=j*nx+i, plane=nx*ny;
    int stride=nz+1;
    const float *c1=profiles, *c2=profiles+stride;
    const float *c3h=profiles+2*stride, *c4h=profiles+3*stride;
    const float *c3f=profiles+4*stride, *c4f=profiles+5*stride;
    const float *dnw=profiles+6*stride, *rdnw=profiles+7*stride;
    const float *rdn=profiles+8*stride;
    float p0=constants[0], t0=constants[1], ptop=constants[2];
    float rd=constants[3], rvovrd=constants[4], cvpm=constants[5];
    for (int k=nz-1;k>=0;--k) {
        int at=k*plane+xy;
        float qt=0.0f;
        for (int s=first_scalar;s<nmoist;++s) {
            int m=s*nz*plane+at;
            qt=qt+(k==nz-1 ? moist[m] : 0.5f*(moist[m]+moist[m+plane]));
        }
        float qf2=1.0f, qf1=qt*qf2;
        if (k==nz-1)
            p[at]=(-0.5f*((c1[k]*mu[xy])+qf1*(c1[k]*muts[xy]+c2[k])))/rdnw[k]/qf2;
        else
            p[at]=p[at+plane]-((c1[k]*mu[xy])+qf1*(c1[k]*muts[xy]+c2[k]))/qf2/rdn[k+1];
        float a=(rd/100000.0f)*(t[at]+t0);
        if (!use_theta_m && nmoist>first_scalar)
            a=a*(1.0f+rvovrd*moist[qv_index*nz*plane+at]);
        al[at]=a*gfk_pow((p[at]+pb[at])/100000.0f,cvpm)-alb[at];
        full_pressure[at]=p[at]+pb[at];
        alt[at]=al[at]+alb[at];
    }
    if (hypso==1) {
        for (int k=0;k<nz;++k) {
            int at=k*plane+xy;
            ph[at+plane]=ph[at]-dnw[k]*((c1[k]*muts[xy]+c2[k])*al[at]+(c1[k]*mu[xy])*alb[at]);
        }
    } else {
        ph[xy]=phb[xy];
        for (int k=1;k<=nz;++k) {
            int at=k*plane+xy;
            float pfu=c3f[k]*muts[xy]+c4f[k]+ptop;
            float pfd=c3f[k-1]*muts[xy]+c4f[k-1]+ptop;
            float phm=c3h[k-1]*muts[xy]+c4h[k-1]+ptop;
            ph[at]=ph[at-plane]+(al[at-plane]+alb[at-plane])*phm*gfk_log(pfd/pfu);
        }
        for (int k=0;k<=nz;++k) ph[k*plane+xy]=ph[k*plane+xy]-phb[k*plane+xy];
    }
}
