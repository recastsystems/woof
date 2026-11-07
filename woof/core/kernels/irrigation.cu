// WRF 4.6.1 phys/module_irrigation.F, public domain.
// Rounded operations retain scalar Fortran operation order in both modes.
__device__ int irr_mod(int x, int n) { int r=x%n; return r<0 ? r+n : r; }
__device__ int irr_on(float jul, float xtime, float gmt,
                     int sj, int ej, int sh, int nh) {
    int eh=sh+nh; if(eh>23) eh-=24;
    int xt24=(int)fmodf(xtime,1440.0f);
    int tloc=(int)floorf(__fadd_rn(gmt,__fdiv_rn((float)xt24,60.0f)));
    if(tloc<0) tloc+=24;
    bool day= sj<ej ? (jul>=sj && jul<ej) : !(jul>=sj && jul<ej);
    bool hour= sh<eh ? (tloc>=sh && tloc<eh) : !(tloc>=sh && tloc<eh);
    return day && hour;
}
extern "C" __global__ void irrigation_column(
    const float* fraction, const int* random_phase, float* rainbl,
    float* channel, float* qr, const float* rho, const float* dz,
    int option, float amount, int sh, int nh, int sj, int ej,
    int freq, int ph, float jul, float xtime, float gmt, float dt,
    int nx, int ny, int x0, int y0, int ring) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=nx*ny) return;
    int x=c%nx,y=c/nx;
    if(x<ring || y<ring || x>=nx-ring || y>=ny-ring) return;
    float f=fraction[c];
    if(option!=3) channel[c]=0.0f;
    if(option!=3 && !(rainbl[c]<=0.01f && f>=0.001f)) return;
    if(option==3 && !(f>=0.1f)) return;
    int timing=irr_mod((int)jul-(option==3?0:sj),freq);
    int phase=0;
    if(ph==1) phase=irr_mod((int)__fmul_rn((float)((x+x0)*(y+y0)),f),freq);
    if(ph==2) phase=random_phase[c];
    if(!irr_on(jul,xtime,gmt,sj,ej,sh,nh)) return;
    if(option==3) {
        if(ph!=0 && timing!=phase) return;
        float den=__fmul_rn(__fmul_rn((float)(nh*3600),rho[c]),dz[c]);
        den=__fmul_rn(den,100.0f);
        float rate=__fdiv_rn(__fmul_rn((float)freq,amount),den);
        qr[c]=__fadd_rn(qr[c],__fmul_rn(__fmul_rn(f,rate),dt));
    } else {
        if(timing!=0 || (ph!=0 && timing!=phase)) return;
        float rate=__fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn((float)freq,amount),0.000277778f),0.01f),(float)nh);
        float water=__fmul_rn(__fmul_rn(dt,f),rate);
        if(option==1) channel[c]=water;
        if(option==2) rainbl[c]=__fadd_rn(rainbl[c],water);
    }
}
