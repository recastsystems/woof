// WRF v4.6.1, NCAR/noahmp 848f54ad3d28c4303151fe5ad83724e232694422,
// src/module_sf_noahmplsm.F:2148-2171. WRF public-domain distribution.
// Compiled after the existing noahmp_leaves.cu math, only for OPT_BTR=2/3.
// One thread owns one column, including the ordered root-layer reduction.
// The default OPT_BTR=1 keeps its existing translation unit and launch.

#ifdef GPUWM_WRF_EXACT
#define BT_ADD(a,b) __fadd_rn((a),(b))
#define BT_SUB(a,b) __fsub_rn((a),(b))
#define BT_MUL(a,b) __fmul_rn((a),(b))
#define BT_DIV(a,b) __fdiv_rn((a),(b))
#define BT_POW(a,b) r_pow((a),(b))
#define BT_EXP(a) r_exp(a)
#define BT_LOG(a) r_log(a)
#else
#define BT_ADD(a,b) ((a)+(b))
#define BT_SUB(a,b) ((a)-(b))
#define BT_MUL(a,b) ((a)*(b))
#define BT_DIV(a,b) ((a)/(b))
#define BT_POW(a,b) powf((a),(b))
#define BT_EXP(a) expf(a)
#define BT_LOG(a) logf(a)
#endif

extern "C" __global__ void noahmp_root_fractions_selected(
    const float* water, const float* psisat, const float* smcmax,
    const float* bexp, const float* zsoil, const float* dz,
    const int* roots, int n, float mpe, float* total, float* fractions)
{
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=n) return;
    const int root=roots[c];
    float sum=0.0f;
    for(int layer=0;layer<NMP_BT_NSOIL;layer++) {
        int slot=c*NMP_BT_NSOIL+layer;
        float value=0.0f;
        if(layer<root) {
            float w=water[slot];
            w=0.01f>w?0.01f:w;
            float psi=BT_MUL(-psisat[slot],
                BT_POW(BT_DIV(w,smcmax[slot]),-bexp[slot]));
            psi=-150.0f>psi?-150.0f:psi;
#if NMP_BT_OPTION == 2
            float gx=BT_DIV(BT_SUB(1.0f,BT_DIV(psi,-150.0f)),
                            BT_ADD(1.0f,BT_DIV(psisat[slot],-150.0f)));
#elif NMP_BT_OPTION == 3
            float gx=BT_SUB(1.0f,
                BT_EXP(BT_MUL(-5.8f,BT_LOG(BT_DIV(-150.0f,psi)))));
#else
#error This kernel implements only WRF OPT_BTR=2 and OPT_BTR=3
#endif
            gx=0.0f>gx?0.0f:gx;
            gx=1.0f<gx?1.0f:gx;
            float depth=-zsoil[c*NMP_BT_NSOIL+root-1];
            value=BT_MUL(BT_DIV(dz[c*NMP_BT_NLAY+NMP_BT_NSNOW+layer],depth),gx);
            value=mpe>value?mpe:value;
            sum=BT_ADD(sum,value);
        }
        // ENERGY does not define BTRANI outside NROOT. The existing slab
        // convention writes zero there, and consumes only the active roots.
        fractions[slot]=value;
    }
    sum=mpe>sum?mpe:sum;
    total[c]=sum;
    for(int layer=0;layer<root;layer++) {
        int slot=c*NMP_BT_NSOIL+layer;
        fractions[slot]=BT_DIV(fractions[slot],sum);
    }
}
