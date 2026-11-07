// WRF v4 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422.
// Selected-option extension compiled after the unchanged default unit.
// Public-domain source notice: licenses/LICENSE-WRF-public-domain.txt.

__device__ float nmp_sfc2_expf(float x)
{
#ifdef USE_DEVICE_LIBM
    return expf(x);   // negative control
#else
    unsigned int ux = __float_as_uint(x);
    unsigned int abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return FADD(x, x);
        if (x > 88.72283935546875f) return __int_as_float(0x7f800000);
        if (x < -103.97207641601562f) return 0.0f;
    }

    double xd = (double)x;
    double kd = DFMA(__longlong_as_double(C_EXP2F_MISC[2]), xd, __longlong_as_double(C_EXP2F_MISC[1]));
    unsigned long long ki = __double_as_longlong(kd);
    kd = DSUB(kd, __longlong_as_double(C_EXP2F_MISC[1]));
    double r = DFMA(__longlong_as_double(C_EXP2F_MISC[2]), xd, -kd);

    unsigned long long t = C_EXP2F_TAB[ki & 31ull];
    t += ki << 47;
    double s = __longlong_as_double((long long)t);

    double z = DFMA(__longlong_as_double(C_EXP2F_POLY_SCALED[0]), r, __longlong_as_double(C_EXP2F_POLY_SCALED[1]));
    double r2 = DMUL(r, r);
    double y = DFMA(r, __longlong_as_double(C_EXP2F_POLY_SCALED[2]), 1.0);
    y = DFMA(z, r2, y);
    y = DMUL(y, s);
    return (float)y;
#endif
}


#define NMP_SFC2_EXP nmp_sfc2_expf
// SFCDIF2, WRF v4 module_sf_noahmplsm.F:4747-4958.
// Native ILECH is assigned 0 unconditionally; Paulson functions are active.
// Arguments retain native units, including AKMS/AKHS wind-speed rescaling.
__device__ float nmp_sfc2_pspmu(float x){
 return __fsub_rn(__fadd_rn(__fsub_rn(__fmul_rn(-2.0f,glibc_logf(__fmul_rn(__fadd_rn(x,1.0f),0.5f))),glibc_logf(__fmul_rn(__fadd_rn(__fmul_rn(x,x),1.0f),0.5f))),__fmul_rn(2.0f,glibc_atanf(x))),__fdiv_rn(3.14159265f,2.0f));
}
__device__ float nmp_sfc2_psphu(float x){
 return __fmul_rn(-2.0f,glibc_logf(__fmul_rn(__fadd_rn(__fmul_rn(x,x),1.0f),0.5f)));
}
__device__ void nmp_sfcdif2(int iter,float z0,float thz0,float thlm,
 float speed,float zlm,float czil,float grav,float* akms,float* akhs,
 float* rlmo,float* wstar2,float* ustar){
 float btg=__fmul_rn(__fdiv_rn(1.0f,270.0f),grav);
 float elfc=__fmul_rn(0.4f,btg);
 float zilfc=__fmul_rn(__fmul_rn(-czil,0.4f),258.2f);
 float zu=z0,rdz=__fdiv_rn(1.0f,zlm),cxch=__fmul_rn(0.001f,rdz);
 float dthv=__fsub_rn(thlm,thz0),du2=fmaxf(__fmul_rn(speed,speed),1.0e-4f);
 float btgh=__fmul_rn(btg,1000.0f);
 if(iter==1){
  float buoy=__fmul_rn(__fmul_rn(btgh,*akhs),dthv);
  *wstar2=buoy!=0.0f?__fmul_rn(__fmul_rn(1.2f,1.2f),glibc_powf(fabsf(buoy),__fdiv_rn(2.0f,3.0f))):0.0f;
  *ustar=fmaxf(__fsqrt_rn(__fmul_rn(*akms,__fsqrt_rn(__fadd_rn(du2,*wstar2)))),0.07f);
  *rlmo=__fdiv_rn(__fmul_rn(__fmul_rn(elfc,*akhs),dthv),__fmul_rn(__fmul_rn(*ustar,*ustar),*ustar));
 }
 float zt=fmaxf(1.0e-6f,__fmul_rn(NMP_SFC2_EXP(__fmul_rn(zilfc,__fsqrt_rn(__fmul_rn(*ustar,z0)))),z0));
 float zslu=__fadd_rn(zlm,zu),zslt=__fadd_rn(zlm,zt);
 float rlogu=glibc_logf(__fdiv_rn(zslu,zu)),rlogt=glibc_logf(__fdiv_rn(zslt,zt));
 float zetalt=fmaxf(__fmul_rn(zslt,*rlmo),-5.0f);*rlmo=__fdiv_rn(zetalt,zslt);
 float zetalu=__fmul_rn(zslu,*rlmo),zetau=__fmul_rn(zu,*rlmo),zetat=__fmul_rn(zt,*rlmo);
 float simm,simh;
 if(*rlmo<0.0f){
  float xlu=__fsqrt_rn(__fsqrt_rn(__fsub_rn(1.0f,__fmul_rn(16.0f,zetalu))));
  float xlt=__fsqrt_rn(__fsqrt_rn(__fsub_rn(1.0f,__fmul_rn(16.0f,zetalt))));
  float xu=__fsqrt_rn(__fsqrt_rn(__fsub_rn(1.0f,__fmul_rn(16.0f,zetau))));
  float xt=__fsqrt_rn(__fsqrt_rn(__fsub_rn(1.0f,__fmul_rn(16.0f,zetat))));
  simm=__fadd_rn(__fsub_rn(nmp_sfc2_pspmu(xlu),nmp_sfc2_pspmu(xu)),rlogu);
  simh=__fadd_rn(__fsub_rn(nmp_sfc2_psphu(xlt),nmp_sfc2_psphu(xt)),rlogt);
 }else{
  zetalu=fminf(zetalu,1.0f);zetalt=fminf(zetalt,1.0f);
  zetau=fminf(zetau,__fdiv_rn(1.0f,__fdiv_rn(zslu,zu)));
  zetat=fminf(zetat,__fdiv_rn(1.0f,__fdiv_rn(zslt,zt)));
  simm=__fadd_rn(__fsub_rn(__fmul_rn(5.0f,zetalu),__fmul_rn(5.0f,zetau)),rlogu);
  simh=__fadd_rn(__fsub_rn(__fmul_rn(5.0f,zetalt),__fmul_rn(5.0f,zetat)),rlogt);
 }
 *ustar=fmaxf(__fsqrt_rn(__fmul_rn(*akms,__fsqrt_rn(__fadd_rn(du2,*wstar2)))),0.07f);
 zt=fmaxf(1.0e-6f,__fmul_rn(NMP_SFC2_EXP(__fmul_rn(zilfc,__fsqrt_rn(__fmul_rn(*ustar,z0)))),z0));
 zslt=__fadd_rn(zlm,zt);rlogt=glibc_logf(__fdiv_rn(zslt,zt));
 float ustark=__fmul_rn(*ustar,0.4f);
 if(simm<1.0e-6f)simm=1.0e-6f;*akms=fmaxf(__fdiv_rn(ustark,simm),cxch);
 if(simh<1.0e-6f)simh=1.0e-6f;*akhs=fmaxf(__fdiv_rn(ustark,simh),cxch);
 float buoy=__fmul_rn(__fmul_rn(btgh,*akhs),dthv);
 *wstar2=buoy!=0.0f?__fmul_rn(__fmul_rn(1.2f,1.2f),glibc_powf(fabsf(buoy),__fdiv_rn(2.0f,3.0f))):0.0f;
 float rlmn=__fdiv_rn(__fmul_rn(__fmul_rn(elfc,*akhs),dthv),__fmul_rn(__fmul_rn(*ustar,*ustar),*ustar));
 *rlmo=__fadd_rn(__fmul_rn(*rlmo,0.15f),__fmul_rn(rlmn,__fsub_rn(1.0f,0.15f)));
}
#undef NMP_SFC2_EXP

__device__ void d_bare_flux_selected(const float *p, const int *q, float *o,
                           int opt_stc=1, int opt_sfc=1, float czil=0.0f)
{
    const float sag = p[1], lwdn = p[2], ur = p[3], uu = p[4], vv = p[5];
    const float sfctmp = p[6], qair = p[8], eair = p[9], rhoair = p[10];
    const float snowh = p[11], zlvl = p[12], zpd = p[13], z0m = p[14];
    const float emg = p[16], rsurf = p[17], lathea = p[18], gamma = p[19];
    const float rhsur = p[20], pahb = p[22], psfc = p[26];
    const float *dzsnso = p + 32, *stc = p + 39, *df = p + 46;
    const int isnow = q[0];
    const int urban_flag = q[4];

    // Fortran declares these arrays (-NSNOW+1:NSOIL), so element k lives at
    // index k + NSNOW - 1 of the packed row.
    const int kk = isnow + 1 + NSNOW - 1;

    float tgb = p[28];
    float h = K_ZERO;

    Sfcdif1State s;
    s.moz = K_ZERO;
    s.mozsgn = 0;
    s.fm = K_ZERO;
    s.fh = K_ZERO;
    s.fm2 = K_ZERO;
    s.fh2 = K_ZERO;
    s.fv = K_P1;
    s.cm = K_ZERO;
    s.ch = K_ZERO;
    s.ch2 = K_ZERO;

    float wstar=0.0f;
    if(opt_sfc==2){s.cm=p[29];s.ch=p[30];}
    const float cir = FMUL(emg, K_SB);
    const float cgh = FDIV(FMUL(K_TWO, df[kk]), dzsnso[kk]);

    float csh = K_ZERO, cev = K_ZERO, estg = K_ZERO, ehb = K_ZERO;
    float irb = K_ZERO, shb = K_ZERO, evb = K_ZERO, ghb = K_ZERO;
    float qsfc = p[31];
    float z0h = z0m;

    for (int it = 1; it <= NITERB; ++it) {
        z0h = z0m;                       // both arms of the ITER==1 test

        if(opt_sfc==2){
            nmp_sfcdif2(it,z0m,tgb,p[7],ur,zlvl,czil,K_GRAV,&s.cm,&s.ch,&s.moz,&wstar,&s.fv);
            s.ch=FDIV(s.ch,ur);s.cm=FDIV(s.cm,ur);
            if(snowh>0.0f){s.cm=f_min(0.01f,s.cm);s.ch=f_min(0.01f,s.ch);}
        }else{
        d_sfcdif1(&s, it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h,
                  ur, K_MPE);
        }

        float rahb = f_max(K_ONE, FDIV(K_ONE, FMUL(s.ch, ur)));
        float rawb = rahb;
        ehb = FDIV(K_ONE, rahb);

        float t = f_min(K_P50, f_max(K_N50, FSUB(tgb, K_TFRZ)));
        float destg;
        if (t > K_ZERO) {
            estg = esat_poly(0, t);
            destg = esat_poly(14, t);
        } else {
            estg = esat_poly(7, t);
            destg = esat_poly(21, t);
        }

        csh = FDIV(FMUL(rhoair, K_CPAIR), rahb);
        cev = FDIV(FDIV(FMUL(rhoair, K_CPAIR), gamma), FADD(rsurf, rawb));

        irb = FSUB(FMUL(cir, powi4(tgb)), FMUL(emg, lwdn));
        shb = FMUL(csh, FSUB(tgb, sfctmp));
        evb = FMUL(cev, FSUB(FMUL(estg, rhsur), eair));
        ghb = FMUL(cgh, FSUB(tgb, stc[kk]));

        float b = FADD(FSUB(FSUB(FSUB(FSUB(sag, irb), shb), evb), ghb), pahb);
        float cir4t3 = FMUL(FMUL(K_FOUR, cir), powi3(tgb));
        float a = FADD(FADD(FADD(cir4t3, csh), FMUL(cev, destg)), cgh);
        float dtg = FDIV(b, a);

        irb = FADD(irb, FMUL(cir4t3, dtg));
        shb = FADD(shb, FMUL(csh, dtg));
        evb = FADD(evb, FMUL(FMUL(cev, destg), dtg));
        ghb = FADD(ghb, FMUL(cgh, dtg));

        tgb = FADD(tgb, dtg);
        h = FMUL(csh, FSUB(tgb, sfctmp));

        t = f_min(K_P50, f_max(K_N50, FSUB(tgb, K_TFRZ)));
        estg = (t > K_ZERO) ? esat_poly(0, t) : esat_poly(7, t);
        float er = FMUL(estg, rhsur);
        qsfc = FDIV(FMUL(K_P622, er), FSUB(psfc, FMUL(K_P378, er)));
        // QFX is computed here in WRF and never used again; omitted.
    }

    // Native STC=1 reset / STC=3 snow-fraction blend (:4445-4457).
    if (opt_stc != 2 && snowh > K_P05 && tgb > K_TFRZ) {
        tgb = opt_stc == 3
            ? FADD(FMUL(FSUB(K_ONE, p[15]), tgb), FMUL(p[15], K_TFRZ))
            : K_TFRZ;
        irb = FSUB(FMUL(cir, powi4(tgb)), FMUL(emg, lwdn));
        shb = FMUL(csh, FSUB(tgb, sfctmp));
        evb = FMUL(cev, FSUB(FMUL(estg, rhsur), eair));
        ghb = FSUB(FADD(sag, pahb), FADD(FADD(irb, shb), evb));
    }

    float tauxb = FSUB(K_ZERO, FMUL(FMUL(FMUL(rhoair, s.cm), ur), uu));
    float tauyb = FSUB(K_ZERO, FMUL(FMUL(FMUL(rhoair, s.cm), ur), vv));

    // opt_sfc == 1 2 m diagnostics.  WRF assigns EHB2 twice; the first
    // assignment is dead and is not reproduced.
    float ehb2 = FDIV(FMUL(s.fv, K_VKC),
                      FSUB(glibc_logf(FDIV(FADD(K_TWO, z0h), z0h)), s.fh2));
    float cq2b = ehb2;
    float t2mb, q2b;
    if (ehb2 < K_E5) {
        t2mb = tgb;
        q2b = qsfc;
    } else {
        t2mb = FSUB(tgb, FDIV(FMUL(FDIV(shb, FMUL(rhoair, K_CPAIR)), K_ONE),
                              ehb2));
        q2b = FSUB(qsfc, FMUL(FDIV(evb, FMUL(lathea, rhoair)),
                              FADD(FDIV(K_ONE, cq2b), rsurf)));
    }
    if (urban_flag) q2b = qsfc;

    o[0] = tgb;
    o[1] = s.cm;
    o[2] = ehb;          // "update CH": BARE_FLUX overwrites CH with EHB
    o[3] = qsfc;
    o[4] = tauxb;
    o[5] = tauyb;
    o[6] = irb;
    o[7] = shb;
    o[8] = evb;
    o[9] = ghb;
    o[10] = t2mb;
    o[11] = q2b;
    o[12] = ehb2;
}

// Same packed ABI; FSNO is already input slot 15.
extern "C" __global__ void noahmp_bare_flux_stc3(const float *in, const int *ii,
                                               float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_bare_flux_selected(in + 53 * i, ii + 5 * i, out + 13 * i, 3);
}

// Fixed-capacity surface slabs: inactive water/ice rows never run this leaf.
extern "C" __global__ void noahmp_bare_flux_masked(
 const float* in,const int* ii,float* out,int n,const unsigned char* active){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 if(!active[i]){for(int k=0;k<13;k++)out[13*i+k]=0.0f;return;}
 d_bare_flux(in+53*i,ii+5*i,out+13*i);
}

extern "C" __global__ void noahmp_bare_flux_stc2(const float* in,const int* ii,float* out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)d_bare_flux_selected(in+53*i,ii+5*i,out+13*i,2);
}
extern "C" __global__ void noahmp_bare_flux_masked_stc2(const float* in,const int* ii,float* out,int n,const unsigned char* active){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!active[i]){for(int k=0;k<13;k++)out[13*i+k]=0.0f;return;}
 d_bare_flux_selected(in+53*i,ii+5*i,out+13*i,2);
}

extern "C" __global__ void noahmp_bare_flux_sfc2(const float* in,const int* ii,float* out,int n,int stc){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d_bare_flux_selected(in+54*i,ii+5*i,out+13*i,stc,2,in[54*i+53]);
}
extern "C" __global__ void noahmp_bare_flux_masked_sfc2(const float* in,const int* ii,float* out,int n,const unsigned char* active,int stc){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!active[i]){for(int j=0;j<13;++j)out[13*i+j]=0.0f;return;}
 d_bare_flux_selected(in+54*i,ii+5*i,out+13*i,stc,2,in[54*i+53]);
}
