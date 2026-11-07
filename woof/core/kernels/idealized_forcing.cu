// WRF 4.6.1 module_ra_hs.F and module_damping_em.F, public domain.
// Scalar operation order; transcendental calls use WOOF's own float32 math.
extern "C" __global__ void relax_temperature(
    int nx, int ny, int nz, const float *lat, const float *p,
    const float *ps, const float *pi, const float *temperature,
    float degrad, float rd, float cp, float *rth) {
    int n = blockDim.x*blockIdx.x+threadIdx.x;
    if (n >= nx*ny*nz) return;
    int col = n % (nx*ny);
    float angle = __fmul_rn(lat[col], degrad);
    float s = glibc_sinf(angle), co = glibc_cosf(angle);
    float ratio = __fdiv_rn(p[n],100000.0f);
    float first = __fsub_rn(315.0f,__fmul_rn(60.0f,__fmul_rn(s,s)));
    float vertical = __fmul_rn(__fmul_rn(10.0f,gfk_log(ratio)),__fmul_rn(co,co));
    float ttmp = __fsub_rn(first,vertical);
    float teq = fmaxf(200.0f,__fmul_rn(ttmp,gfk_pow(ratio,__fdiv_rn(rd,cp))));
    float sig = __fdiv_rn(p[n],ps[col]);
    float sigterm = fmaxf(0.0f,__fdiv_rn(__fsub_rn(sig,0.7f),__fsub_rn(1.0f,0.7f)));
    float kka = __fdiv_rn(1.0f,40.0f);
    float seasonal = __fmul_rn(__fmul_rn(__fsub_rn(0.25f,kka),sigterm),gfk_pow(co,4.0f));
    float kkt = __fadd_rn(kka, seasonal);
    float tend = __fdiv_rn(__fmul_rn(-kkt,__fsub_rn(temperature[n],teq)),86400.0f);
    rth[n] = __fadd_rn(rth[n],__fdiv_rn(tend,pi[n]));
}

// Pack the pressure halo into persistent stage scratch in one GPU launch.
// The regional state stores either total or original perturbation pressure;
// the native friction expression must keep perturbation and base separate.
extern "C" __global__ void relax_pressure_halo(
    int nx, int ny, int nz, int base_1d, int perturbation,
    int edge_x, int edge_y, const float *pressure, const float *base,
    float *p, float *pb) {
    int n=blockDim.x*blockIdx.x+threadIdx.x;
    int hx=nx+2, hy=ny+2;
    if(n>=hx*hy*nz) return;
    int i=n%hx-1, j=(n/hx)%hy-1, k=n/(hx*hy);
    i=edge_x ? max(0,min(nx-1,i)) : (i+nx)%nx;
    j=edge_y ? max(0,min(ny-1,j)) : (j+ny)%ny;
    int source=(k*ny+j)*nx+i;
    float b=base[base_1d ? k : source];
    pb[n]=b;
    p[n]=perturbation ? pressure[source] : __fsub_rn(pressure[source],b);
}

// p and pb include one horizontal halo, k=0 is the lowest mass level.
// axis=0: u(nx+1,ny,nz); axis=1: v(nx,ny+1,nz).
// stop is the exclusive end face from the WRF tile bounds.
extern "C" __global__ void relax_momentum(
    int nx, int ny, int nz, int axis, int stop,
    const float *p, const float *pb, const float *wind, float *tend) {
    int n=blockDim.x*blockIdx.x+threadIdx.x;
    int sx=nx+(axis==0), sy=ny+(axis==1);
    if(n>=sx*sy*nz) return;
    int i=n%sx, j=(n/sx)%sy, k=n/(sx*sy);
    if((axis==0 && i>=stop) || (axis==1 && (j==0 || j>=stop))) return;
    int hx=nx+2, plane=hx*(ny+2);
    int a=k*plane+(j+1)*hx+i+1;
    int b=a-(axis==0?1:hx);
    int a0=a-k*plane, b0=b-k*plane;
    float top=__fadd_rn(__fadd_rn(__fadd_rn(p[b],pb[b]),p[a]),pb[a]);
    float bottom=__fadd_rn(__fadd_rn(__fadd_rn(p[b0],pb[b0]),p[a0]),pb[a0]);
    float sig=__fdiv_rn(top,bottom);
    float sigterm=fmaxf(0.0f,__fdiv_rn(__fsub_rn(sig,0.7f),__fsub_rn(1.0f,0.7f)));
    float kkv=__fmul_rn(__fdiv_rn(1.0f,86400.0f),sigterm);
    tend[n]=__fsub_rn(tend[n],__fmul_rn(kkv,wind[n]));
}

// The same native v expression with domain-boundary geography, windowed by
// the ordinary staggered gather. The mask pointer stays fixed under graphs;
// its contents follow the tile, including repeated periodic domain images.
extern "C" __global__ void relax_momentum_masked(
    int nx, int ny, int nz, const float *active,
    const float *p, const float *pb, const float *wind, float *tend) {
    int n=blockDim.x*blockIdx.x+threadIdx.x;
    int sy=ny+1;
    if(n>=nx*sy*nz) return;
    int i=n%nx, j=(n/nx)%sy, k=n/(nx*sy);
    if(active[j*nx+i]==0.0f) return;
    int hx=nx+2, plane=hx*(ny+2);
    int a=k*plane+(j+1)*hx+i+1, b=a-hx;
    int a0=a-k*plane, b0=b-k*plane;
    float top=__fadd_rn(__fadd_rn(__fadd_rn(p[b],pb[b]),p[a]),pb[a]);
    float bottom=__fadd_rn(__fadd_rn(__fadd_rn(p[b0],pb[b0]),p[a0]),pb[a0]);
    float sig=__fdiv_rn(top,bottom);
    float sigterm=fmaxf(0.0f,__fdiv_rn(__fsub_rn(sig,0.7f),__fsub_rn(1.0f,0.7f)));
    float kkv=__fmul_rn(__fdiv_rn(1.0f,86400.0f),sigterm);
    tend[n]=__fsub_rn(tend[n],__fmul_rn(kkv,wind[n]));
}
