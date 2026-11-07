// Strict replay uses the host-compatible functions. Production uses CUDA math.
#ifdef GPUWM_WRF_EXACT
#define skin_exp gfk_exp
#define skin_pow gfk_pow
#else
#define skin_exp expf
#define skin_pow powf
#endif
// WRF v4.6.1 phys/module_sf_sstskin.F:69-129, public domain.
// The surface driver reset/write-back is module_surface_driver.F:1801-1818.
extern "C" __global__ void sstskin_column(
    const float* xland, const float* glw, const float* gsw,
    const float* hfx, const float* qfx, float* tsk,
    const float* ust, const float* emiss, const float* sst,
    float* dtw1, float* sstsk, float* tslb,
    float dt, float stbolt, int sst_update, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = xland[i];
    if (x > 1.5f && sst_update != 1) {
        tsk[i] = sst[i];
        tslb[i] = sst[i];
    }
    if (x >= 1.5f) {
        const float z1=3.f, an=.3f, zk=.4f, rho=1.2f, rhow=1025.f, cw=4190.f;
        const float g=9.8f, znuw=1.e-6f, zkw=1.4e-7f;
        float s2 = sstsk[i] * sstsk[i];
        float qo = glw[i] - emiss[i]*stbolt*(s2*s2) - 2.5e6f*qfx[i] - hfx[i];
        float swo = gsw[i];
        float us = fmaxf(ust[i], .01f);
        float tb = tsk[i] - 273.15f;
        float dtwo = dtw1[i], delt = dt;
        float q = qo/(rhow*cw), sw = swo/(rhow*cw);
        float f1 = 1.f - .27f*skin_exp(-2.8f*z1) - .45f*skin_exp(-.07f*z1);
        // f1 absorption
        float alw = 1.e-5f * fmaxf(tb,1.f);
        float con4 = 16.f*g*alw*((znuw*znuw)*znuw)/(zkw*zkw);
        float usw = FSQRT(rho/rhow)*us;
        float u2 = usw*usw;
        float con5 = con4/(u2*u2);
        float q2 = fmaxf(1.f/(rhow*cw),-q);
        float zlan = 6.f/skin_pow(1.f+skin_pow(con5*q2,.75f),.333f);
        float dep = zlan*znuw/usw;
        float fs = .065f+11.f*dep-(6.6e-5f/dep)*(1.f-skin_exp(-dep/8.e-4f));
        fs = fmaxf(fs,.01f);
        float dtc = dep*(q+sw*fs)/zkw;
        dtc = fminf(dtc,0.f);
        alw = 1.e-5f*fmaxf(tb,1.f);
        float con1 = FSQRT(5.f*z1*g*alw/an);
        float con2 = zk*g*alw;
        float qn = q+sw*f1;
        usw = FSQRT(rho/rhow)*us;
        if (dtwo>0.f && qn<0.f) {
            float qn1 = FSQRT(dtwo)*(usw*usw)/con1;
            qn = fmaxf(qn,qn1);
        }
        float zeta = z1*con2*qn/((usw*usw)*usw);
        float phi = zeta>0.f ? 1.f+5.f*zeta : 1.f/FSQRT(1.f-16.f*zeta);
        float con3 = zk*usw/(z1*phi);
        float dtw = (dtwo+(an+1.f)/an*(q+sw*f1)*delt/z1)/(1.f+(an+1.f)*con3*delt);
        dtw = fmaxf(0.f,dtw);
        float ts = (tb+dtw)+dtc;
        sstsk[i] = ts+273.15f;
        dtw1[i] = dtw;
    }
    if (x>1.5f) tsk[i] = sstsk[i];
}
