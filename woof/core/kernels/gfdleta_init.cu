// WRF 4.6.1 GFDL radiation initialization and cloud overlap primitives.
// These entry points do not implement a forecast radiation scheme.
// Float32 transcendentals use WOOF's shared math routines. Every basic
// operation is rounded separately, in the Fortran expression order.
#define GD_A(a,b) __fadd_rn((a),(b))
#define GD_S(a,b) __fsub_rn((a),(b))
#define GD_M(a,b) __fmul_rn((a),(b))
#define GD_D(a,b) __fdiv_rn((a),(b))

__device__ float gd_gausin(float xsd) {
    float pi = glibc_acosf(-1.0f);
    float rsqr = GD_D(1.0f, __fsqrt_rn(GD_M(2.0f,pi)));
    float a1 = GD_M(xsd,rsqr);
    float a2 = gfk_exp(GD_M(-0.5f,GD_M(xsd,xsd)));
    float rn=1.0f,b1=1.0f,b2=1.0f,b3=1.0f,sum=1.0f;
    while (b2>1.e-3f) {
        rn=GD_A(rn,1.0f);
        b2=GD_D(GD_M(xsd,xsd),GD_S(GD_M(2.0f,rn),1.0f));
        b3=GD_M(b1,b2);
        sum=GD_A(sum,b3);
        b1=b3;
    }
    return GD_M(GD_M(a1,a2),sum);
}

extern "C" __global__ void gd_gaussian(const float *x, float *y, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)y[i]=gd_gausin(x[i]);
}

__device__ float gd_solard(int year,int month,int day,int hour) {
    const float pi=3.1415926f,pi2=2.0f*pi;
    int jd=day-32075+1461*(year+4800+(month-14)/12)/4
          +367*(month-2-(month-14)/12*12)/12
          -3*((year+4900+(month-14)/12)/100)/4;
    float fjd;
    if(hour<12) {
        --jd;
        fjd=GD_A(GD_A(0.5f,GD_M(0.041666667f,(float)hour)),GD_M(0.00069444444f,0.0f));
    } else {
        fjd=GD_A(GD_M(0.041666667f,(float)(hour-12)),GD_M(0.00069444444f,0.0f));
    }
    float dayinc=GD_D((float)hour,24.0f);
    float fjd1=GD_A(GD_A((float)jd,fjd),dayinc);
    jd=(int)fjd1;
    fjd=GD_S(fjd1,(float)jd);
    float dat=GD_A(GD_S((float)(jd-2415020),1.55f),fjd);
    float t=GD_D((float)(jd-2415020),36525.0f);
    float orbital_year=GD_A(0.25964134f,GD_M(0.304e-5f,t));
    float ec=GD_S(0.01675104f,GD_M(GD_A(0.418e-4f,GD_M(0.126e-6f,t)),t));
    orbital_year=GD_A(orbital_year,365.0f);
    float date=fmodf(dat,orbital_year);
    float em=GD_D(GD_M(pi2,date),orbital_year);
    float e=1.0f;
    int iter=0;
    while(true) {
        float ep=GD_S(e,GD_D(GD_S(GD_S(e,GD_M(ec,glibc_sinf(e))),em),GD_S(1.0f,GD_M(ec,glibc_cosf(e)))));
        float cr=fabsf(GD_S(e,ep));
        e=ep;
        ++iter;
        if(iter>10 || cr<=1.3e-6f)break;
    }
    return GD_S(1.0f,GD_M(ec,glibc_cosf(e)));
}

extern "C" __global__ void gd_solar(const int *date,float *out,int count) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count)out[c]=gd_solard(date[4*c],date[4*c+1],date[4*c+2],date[4*c+3]);
}

// Per-column storage: CAMT/KTOP/KBTM have nlay+1 slots, with slot zero
// reserved exactly as in WRF. CLDFAC uses Fortran order within a column.
extern "C" __global__ void gd_clo89(const float *camt,const int *nclds,
        const int *kbtm,const int *ktop,float *cldfac,int ncol,int nlay) {
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=ncol)return;
    int nv=nlay+1;
    float *f=cldfac+c*nv*nv;
    for(int j=0;j<nv;j++)for(int i=0;i<nv;i++)f[i+nv*j]=1.0f;
    for(int nc=1;nc<=nclds[c];nc++) {
        float xcld=GD_S(1.0f,camt[c*nv+nc]);
        int k1=ktop[c*nv+nc]+1,k2=kbtm[c*nv+nc];
        int kb=max(k1,k2+1),kt=min(k1-1,k2);
        for(int k=kb;k<=nv;k++)for(int kp=1;kp<=nv;kp++) {
            float row=kp<=k2?xcld:1.0f;
            int ix=(kp-1)+nv*(k-1);
            f[ix]=nc==1?row:GD_M(f[ix],row);
        }
        for(int k=1;k<=kt;k++)for(int kp=1;kp<=nv;kp++) {
            float row=kp>=k1?xcld:1.0f;
            int ix=(kp-1)+nv*(k-1);
            f[ix]=nc==1?row:GD_M(f[ix],row);
        }
        if(k1<=k2)for(int j=k1;j<=k2;j++)for(int i=1;i<=nv;i++) {
            int ix=(i-1)+nv*(j-1);
            f[ix]=nc==1?xcld:GD_M(f[ix],xcld);
        }
    }
}

extern "C" __global__ void gd_fp_probe(const float *x,float *out) {
    out[0]=GD_M(x[0],x[1]);
    out[1]=GD_D(x[2],x[3]);
    out[2]=GD_A(GD_M(x[4],x[5]),x[6]);
}
