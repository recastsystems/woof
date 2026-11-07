"""Public WRF 4.6.1 setvtz :7135-7514; emitted only into variant units.

Source licence/pin are in SOURCE_PIN.json. Preserve separate native mass,
number, and reflectivity expressions and the infdo=1 alpha-zero shortcut.
"""
PREFIX='''
#ifndef NSSL2_DRAG_G_WORD
#define NSSL2_DRAG_G_WORD 7
#endif
#ifndef NSSL2_DRAG_H_WORD
#define NSSL2_DRAG_H_WORD 7
#endif
#define NSSL2_DRAG_G ((int)((unsigned int)(NSSL2_DRAG_G_WORD-1ULL)))
#define NSSL2_DRAG_H ((int)((unsigned int)(NSSL2_DRAG_H_WORD-1ULL)))
#define NSSL2_BX_G (NSSL2_DRAG_G==6 ? 0.6f : (NSSL2_DRAG_G==0 ? 0.37f : 0.5f))
#define NSSL2_AX_G (NSSL2_DRAG_G==6 ? 157.71f : (NSSL2_DRAG_G==0 ? 19.3f : 75.7149f))
#define NSSL2_BX_H (NSSL2_DRAG_H==6 ? 0.593f : (NSSL2_DRAG_H>0 ? 0.5f : 0.6384f))
#define NSSL2_AX_H (NSSL2_DRAG_H==6 ? 179.36f : (NSSL2_DRAG_H>0 ? 75.7149f : 206.984f))
'''
HELPERS='''
__device__ __forceinline__ float nssl2_drag_cd(float density, bool hail, bool number) {
    const int selector=hail ? NSSL2_DRAG_H : NSSL2_DRAG_G;
    if (selector==1) return hail ? 0.45f : 0.8f;
    if (!hail && selector==2)
        return fmaxf(0.45f,fminf(1.0f,0.45f+__fdiv_rn(0.35f*(800.0f-fmaxf(500.0f,fminf(800.0f,density))),800.0f-500.0f)));
    if (selector==4 || (hail && selector==5 && number)) {
        const float upper=hail ? 0.6f : 0.8f;
        // Native number-weighted hail selector 5 has an explicit 0.15.
        const float delta=(hail && selector==5 && number) ? 0.15f : upper-0.45f;
        return fmaxf(0.45f,fminf(upper,0.45f+__fdiv_rn(delta*(800.0f-fmaxf(500.0f,fminf(800.0f,density))),800.0f-500.0f)));
    }
    if (selector==5) return 0.8f*gfk_pow(__fdiv_rn(density,500.0f),2.0f/3.0f);
    const float lower=hail ? 500.0f : 170.0f;
    return fmaxf(0.45f,fminf(1.2f,0.45f+__fdiv_rn(0.55f*(800.0f-fmaxf(lower,fminf(800.0f,density))),800.0f-170.0f)));
}
template <class Gamma>
__device__ __forceinline__ float nssl2_drag_mass(
    float density,float diameter,float alpha,float rhovt,bool hail,
    float* axx,float* bxx,float* cd, Gamma gamma) {
    const int selector=hail ? NSSL2_DRAG_H : NSSL2_DRAG_G;
    *cd=nssl2_drag_cd(density,hail,false);
    float a=hail ? NSSL2_AX_H : NSSL2_AX_G;
    float b=hail ? NSSL2_BX_H : NSSL2_BX_G;
    if (selector==6) NSSL2_MM_COEFFICIENTS(density,&a,&b);
    if (alpha==0.0f && selector>0 && selector!=6) {
        *axx=sqrtf(__fdiv_rn(4.0f*density*9.8f,3.0f*(*cd)*1.225f));
        *bxx=0.5f;
        return ((__fdiv_rn(nssl2_gamma_sp(4.0f+0.5f),6.0f)*rhovt)*(*axx))*sqrtf(diameter);
    }
    const float x=gamma((4.0f+alpha)+b),y=gamma(4.0f+alpha);
    if (selector>0 && selector!=6) {
        a=sqrtf(__fdiv_rn(4.0f*density*9.8f,3.0f*(*cd)*1.225f));
        *axx=a;*bxx=b;
        return __fdiv_rn(((rhovt*a)*sqrtf(diameter))*x,y);
    }
    *axx=a;*bxx=b;
    if (selector==6) return __fdiv_rn(((rhovt*a)*gfk_pow(diameter,b))*x,y);
    if (hail) return __fdiv_rn(rhovt*((a*gfk_pow(diameter,b))*x),y);
    return __fdiv_rn((rhovt*a)*(gfk_pow(diameter,b)*x),y);
}
template <class Gamma>
__device__ __forceinline__ float nssl2_drag_number(
    float density,float diameter,float alpha,float rho,float rhovt,bool hail,
    int infdo,float mass_a,float mass_b, float* number_a,float* number_b,Gamma gamma,float inherited_cd=0.f) {
    const int selector=hail ? NSSL2_DRAG_H : NSSL2_DRAG_G;
    const bool carry=hail?(selector==2||selector>6):(selector>6);
    const float cd=carry?inherited_cd:nssl2_drag_cd(density,hail,true);
    float a=selector==6 ? mass_a : (hail ? NSSL2_AX_H : NSSL2_AX_G);
    const float b=selector==6 ? mass_b : (hail ? NSSL2_BX_H : NSSL2_BX_G);
    if (alpha==0.0f && infdo<2 && selector>0 && selector!=6) {
        *number_a=mass_a;*number_b=b;
        return sqrtf(__fdiv_rn(((density*diameter)*3.14159f)*9.8f,(3.0f*cd)*fmaxf(0.05f,rho)));
    }
    const float x=gamma((1.0f+alpha)+b),y=gamma(1.0f+alpha);
    if (selector>0 && selector!=6) a=sqrtf(__fdiv_rn(4.0f*density*9.8f,3.0f*cd*1.225f));
    *number_a=a;*number_b=b;
    if (selector>0) return __fdiv_rn(((rhovt*a)*gfk_pow(diameter,b))*x,y);
    return __fdiv_rn((rhovt*a)*(gfk_pow(diameter,b)*x),y);
}
'''

def extend_gs(s):
 s=s.replace('                     0.6f, 157.71f,','                     NSSL2_BX_G, NSSL2_AX_G,').replace('                     __int_as_float(0x3F17CED9), 179.36f,','                     NSSL2_BX_H, NSSL2_AX_H,')
 a=s.index('        // graupel / hail mass-weighted velocities');b=s.index('\n    }\n\n    // ---- :15193',a)
 s=s[:a]+'''        // Public WRF setvtz infdo=1, including alpha-zero number shortcut.
        for (int il=LH;il<=lhab;++il) {
            vtxbar[il][0]=0.0f;vtxbar[il][1]=0.0f;
            if (!(qx[il]>species(il).qxmin)) continue;
            const bool hail=il==LHL;
            const auto gamma=[] (float x) { return gmoi_blend(x); };
            vtxbar[il][0]=nssl2_drag_mass(xdn[il],xdia[il][0],alpha[il],rhovt,hail,&axx[il],&bxx[il],&cdxgs[il],gamma);
            float an,bn;
            vtxbar[il][1]=nssl2_drag_number(xdn[il],xdia[il][0],alpha[il],rho0,rhovt,hail,1,axx[il],bxx[il],&an,&bn,gamma);
        }
        vtxbar[LC][1]=vtxbar[LC][0];
''' +s[b:]
 anchor='struct GsColumn {'
 assert anchor in s
 s=s.replace(anchor,'#define NSSL2_MM_COEFFICIENTS nssl2_graupel_mm_coefficients_wrf\n'+HELPERS+'\n'+anchor)
 from carry_transcription import extend_gs as carry_gs
 return PREFIX+carry_gs(s)

def extend_driver(s):
 # Static power-law coefficients need one gamma interval; MM needs its 20
 # slots over the tabulated exponent range. Retain native interpolation.
 s=s.replace('nssl2_gmoi_index((1.0f+a)+0.5f)','nssl2_gmoi_index((1.0f+a)+(shape ? (NSSL2_DRAG_H==6 ? 0.5f : NSSL2_BX_H-0.01f) : (NSSL2_DRAG_G==6 ? 0.5f : NSSL2_BX_G-0.01f)))')
 a=s.index('    float aax;\n    float bbx;\n',s.index('void nssl2_dense_frozen_velocities'))
 b=s.index('    if (v2 > v1',a)
 s=s[:a]+'''    const float rhovt=sqrtf(1.225f*fminf(20.0f,1.0f/rho));
    const int shape=HAIL ? 1 : 0;
    const auto gamma=[&] (float x) {
        const float base=1.0f+alpha;
        const int group=(int)(x-base)/3;
        // Denominator arguments are not in the exponent cache.
        if (x==base || x==4.0f+alpha || x==7.0f+alpha) return nssl2_gamma_lookup(x);
        return nssl2_velocity_gamma_lookup<CACHED>(x,cached_gamma,shape);
    };
    float a,b,cd,an,bn;
    float v1=nssl2_drag_mass(xdn,xdia,alpha,rhovt,HAIL,&a,&b,&cd,gamma);
    float v2=nssl2_drag_number(xdn,xdia,alpha,rho,rhovt,HAIL,2,a,b,&an,&bn,gamma);
    float v3=__fdiv_rn(rhovt*((an*gfk_pow(xdia,bn))*gamma((7.0f+alpha)+bn)),gamma(7.0f+alpha));
''' +s[b:]
 anchor='// ziegfall1d (module_mp_nssl_2mom.F:7595)'
 assert anchor in s
 s=s.replace(anchor,'#define NSSL2_MM_COEFFICIENTS nssl2_graupel_mm_coefficients\n'+HELPERS+'\n'+anchor)
 from carry_transcription import extend_driver as carry_driver
 return PREFIX+carry_driver(s)
