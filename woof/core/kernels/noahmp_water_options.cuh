// Native WRF v4 Noah-MP 848f54ad; original WRF public-domain notice in
// licenses/LICENSE-WRF-public-domain.txt. Source pin and tables are unchanged.
// Complete SRT/SOILWATER/WATER variants reuse unchanged native snow, canopy,
// infiltration and tridiagonal procedures from noahmp_water.cu. No v5 code.
// Runtime compilation appends this file, preserving all default unit bytes.
namespace noahmp_water_options {
// Arm optimized-routines MIT logf, reused from noahmp_bareflux.cu.
// Notice: licenses/LICENSE-Arm-optimized-routines-MIT.txt.
__constant__ unsigned long long C_LOGF_TAB[32] = {
    0x3FF661EC79F8F3BEULL, 0xBFD57BF7808CAADEULL,
    0x3FF571ED4AAF883DULL, 0xBFD2BEF0A7C06DDBULL,
    0x3FF49539F0F010B0ULL, 0xBFD01EAE7F513A67ULL,
    0x3FF3C995B0B80385ULL, 0xBFCB31D8A68224E9ULL,
    0x3FF30D190C8864A5ULL, 0xBFC6574F0AC07758ULL,
    0x3FF25E227B0B8EA0ULL, 0xBFC1AA2BC79C8100ULL,
    0x3FF1BB4A4A1A343FULL, 0xBFBA4E76CE8C0E5EULL,
    0x3FF12358F08AE5BAULL, 0xBFB1973C5A611CCCULL,
    0x3FF0953F419900A7ULL, 0xBFA252F438E10C1EULL,
    0x3FF0000000000000ULL, 0x0000000000000000ULL,
    0x3FEE608CFD9A47ACULL, 0x3FAAA5AA5DF25984ULL,
    0x3FECA4B31F026AA0ULL, 0x3FBC5E53AA362EB4ULL,
    0x3FEB2036576AFCE6ULL, 0x3FC526E57720DB08ULL,
    0x3FE9C2D163A1AA2DULL, 0x3FCBC2860D224770ULL,
    0x3FE886E6037841EDULL, 0x3FD1058BC8A07EE1ULL,
    0x3FE767DCF5534862ULL, 0x3FD4043057B6EE09ULL,
};
__constant__ unsigned long long C_LOGF_MISC[4] = {
    0x3FE62E42FEFA39EFULL, 0xBFD00EA348B88334ULL, 0x3FD5575B0BE00B6AULL, 0xBFDFFFFEF20A4123ULL,
};
__device__ float glibc_logf(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3F800000u) return 0.0f;           // log(1) == +0

    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return __uint_as_float(0xFF800000u);   // log(0)
        if (ix == 0x7F800000u) return __uint_as_float(0x7F800000u); // log(inf)
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xFF000000u)
            return __uint_as_float(0x7FC00000u);  // negative or NaN -> NaN
        ix = __float_as_uint(FMUL(x, 8388608.0f));   // subnormal: * 0x1p23
        ix -= (23u << 23);
    }

    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    int k = ((int) tmp) >> 23;                    // arithmetic shift
    unsigned int iz = ix - (tmp & 0xFF800000u);
    double invc = __longlong_as_double(C_LOGF_TAB[2 * i]);
    double logc = __longlong_as_double(C_LOGF_TAB[2 * i + 1]);
    double z = (double) __uint_as_float(iz);

    double ln2 = __longlong_as_double(C_LOGF_MISC[0]);
    double a0 = __longlong_as_double(C_LOGF_MISC[1]);
    double a1 = __longlong_as_double(C_LOGF_MISC[2]);
    double a2 = __longlong_as_double(C_LOGF_MISC[3]);

    double r = DFMA(z, invc, -1.0);
    double y0 = DFMA((double) k, ln2, logc);
    double r2 = DMUL(r, r);
    double y = DFMA(a1, r, a2);
    y = DFMA(r2, a0, y);
    y = DFMA(r2, y, DADD(y0, r));
    return __double2float_rn(y);
}

// The inherited WATER powf intentionally excludes exponent zero, which
// XAJ reaches for the table value BXAJ=1. Native powf(x,+/-0) returns 1.
__device__ float runoff_powf(float x,float y) {
    return y==K_ZERO ? K_ONE : glibc_powf(x,y);
}
// Complete native CANWATER, with native signed-zero MIN tie at L:6353.
__device__ void d_canwater(float ch2op, float dt, float fcev, float fctr,
                           float elai, float esai, float fveg, float bdfall,
                           int frozen, float canliq, float canice, float tv,
                           float *o)
{
    float lsai = FADD(elai, esai);
    float maxliq = FMUL(FMUL(fveg, ch2op), lsai);                    // :6323
    float etran, qevac, qdewc, qsubc, qfroc;

    if (!frozen) {                                                   // :6327
        etran = f_max(FDIV(fctr, K_HVAP), K_ZERO);
        qevac = f_max(FDIV(fcev, K_HVAP), K_ZERO);
        qdewc = f_abs(f_min(FDIV(fcev, K_HVAP), K_ZERO));
        qsubc = K_ZERO;
        qfroc = K_ZERO;
    } else {                                                         // :6333
        etran = f_max(FDIV(fctr, K_HSUB), K_ZERO);
        qevac = K_ZERO;
        qdewc = K_ZERO;
        qsubc = f_max(FDIV(fcev, K_HSUB), K_ZERO);
        qfroc = f_abs(f_min(FDIV(fcev, K_HSUB), K_ZERO));
    }

    // The strict native MIN at :6344 selects the second operand on a
    // signed-zero tie. Real WRF captures retain -0 here when FCEV is -0
    // and CANLIQ is +0; the generic first-on-tie helper loses that word.
    float canopy_rate = FDIV(canliq, dt);
    qevac = (canopy_rate < qevac) ? canopy_rate : qevac;            // :6344
    canliq = f_max(K_ZERO, FADD(canliq, FMUL(FSUB(qdewc, qevac), dt)));
    if (canliq <= K_1EM6_LIQ) canliq = K_ZERO;                       // :6346

    // FVEG * 6.6*(0.27+46.0/BDFALL) * (ELAI+ESAI) is a left-associative chain.
    float maxsno = FMUL(FMUL(FMUL(fveg, K_6P6),
                             FADD(K_0P27, FDIV(K_46, bdfall))), lsai);

    // Native MIN selects QSUBC on equality, including +0/-0 ties.
    float ice_rate = FDIV(canice, dt);
    qsubc = (ice_rate < qsubc) ? ice_rate : qsubc;                   // :6353
    canice = f_max(K_ZERO, FADD(canice, FMUL(FSUB(qfroc, qsubc), dt)));
    if (canice <= K_1EM6_ICE) canice = K_ZERO;                       // :6355

    float fwet;
    if (canice > K_ZERO && canice >= canliq) {                       // :6359
        fwet = FDIV(f_max(K_ZERO, canice), f_max(maxsno, K_1EM6_SNO));
    } else {
        fwet = FDIV(f_max(K_ZERO, canliq), f_max(maxliq, K_1EM6_LIQF));
    }
    fwet = glibc_powf(f_min(fwet, K_ONE), K_0P667);                  // :6364

    float qmeltc = K_ZERO, qfrzc = K_ZERO;
    float cmc = FADD(canliq, canice);                                // :6370

    if (canice > K_1EM6_MELT && tv > K_TFRZ) {                       // :6372
        qmeltc = f_min(FDIV(canice, dt),
                       FDIV(FDIV(FMUL(FMUL(FSUB(tv, K_TFRZ), K_CICE), canice),
                                 K_DENICE),
                            FMUL(dt, K_HFUS)));
        canice = f_max(K_ZERO, FSUB(canice, FMUL(qmeltc, dt)));
        canliq = f_max(K_ZERO, FSUB(cmc, canice));
        tv = FADD(FMUL(fwet, K_TFRZ), FMUL(FSUB(K_ONE, fwet), tv));
    }

    if (canliq > K_1EM6_FRZ && tv < K_TFRZ) {                        // :6379
        qfrzc = f_min(FDIV(canliq, dt),
                      FDIV(FDIV(FMUL(FMUL(FSUB(K_TFRZ, tv), K_CWAT), canliq),
                                K_DENH2O),
                           FMUL(dt, K_HFUS)));
        canliq = f_max(K_ZERO, FSUB(canliq, FMUL(qfrzc, dt)));
        canice = f_max(K_ZERO, FSUB(cmc, canliq));
        tv = FADD(FMUL(fwet, K_TFRZ), FMUL(FSUB(K_ONE, fwet), tv));
    }

    cmc = FADD(canliq, canice);                                      // :6388
    float ecan = FSUB(FSUB(FADD(qevac, qsubc), qdewc), qfroc);       // :6392

    o[0] = canliq; o[1] = canice; o[2] = tv;  o[3] = cmc;
    o[4] = ecan;   o[5] = etran;  o[6] = fwet;
    o[7] = qsubc;  o[8] = qfroc;  o[9] = qfrzc;
    o[10] = qmeltc; o[11] = qevac; o[12] = qdewc;
}

// Native WRF v4 Noah-MP 848f54ad, GROUNDWATER:9582-9773; public domain.
// One resident CUDA thread per column. Native double S_NODE is retained.
__device__ void groundwater(const float *p,const float *z,float dt,
    const float *ice,const float *wcnd,float fcrmax,float *liquid,
    float *zwt,float *wa,float *wt,float *qin,float *qdis) {
    float dz[NSOIL],node[NSOIL],ml[NSOIL],ep[NSOIL],hk[NSOIL],smc[NSOIL];
    dz[0]=FMUL(-z[0],1000.0f);node[0]=FDIV(-z[0],2.0f);
    for(int k=1;k<NSOIL;++k) {
        dz[k]=FMUL(1000.0f,FSUB(z[k-1],z[k]));
        node[k]=FADD(-z[k-1],FMUL(0.5f,FSUB(z[k-1],z[k])));
    }
    for(int k=0;k<NSOIL;++k) {
        smc[k]=FADD(liquid[k],ice[k]);ml[k]=FMUL(liquid[k],dz[k]);
        ep[k]=f_max(0.01f,FSUB(p[P_SMCMAX+k],ice[k]));hk[k]=FMUL(1000.0f,wcnd[k]);
    }
    int iwt=NSOIL-1;
    for(int k=1;k<NSOIL;++k)if(*zwt<=-z[k]) {iwt=k-1;break;}
    float fff=FDIV(p[P_BEXP+iwt],3.0f);
    // Native EXP(3.0) is folded once to binary32 by the strict compiler.
    float rsbmx=FMUL(FMUL(hk[iwt],1000.0f),20.085536956787109375f);
    *qdis=FMUL(FMUL(FMUL(FSUB(1.0f,fcrmax),rsbmx),glibc_expf(-p[W_P_STRIDE+1])),glibc_expf(FMUL(-fff,*zwt)));
    double saturation=(double)f_min(1.0f,FDIV(smc[iwt],p[P_SMCMAX+iwt]));
    saturation=saturation>(double)0.01f?saturation:(double)0.01f;
    float suction=(float)__dmul_rn((double)FMUL(-p[W_P_STRIDE+2+iwt],1000.0f),pow(saturation,(double)-p[P_BEXP+iwt]));
    suction=f_max(-120000.0f,FMUL(0.8f,suction));
    float ka=FDIV(FMUL(2.0f,FMUL(FMUL(hk[iwt],p[P_DKSAT+iwt]),1000.0f)),FADD(hk[iwt],FMUL(p[P_DKSAT+iwt],1000.0f)));
    float whz=FMUL(-*zwt,1000.0f),wh=FSUB(suction,FMUL(node[iwt],1000.0f));
    *qin=FDIV(FMUL(-ka,FSUB(whz,wh)),FMUL(FSUB(*zwt,node[iwt]),1000.0f));
    *qin=f_max(FDIV(-10.0f,dt),f_min(FDIV(10.0f,dt),*qin));
    *wt=FADD(*wt,FMUL(FSUB(*qin,*qdis),dt));
    if(iwt==NSOIL-1) {
        *wa=FADD(*wa,FMUL(FSUB(*qin,*qdis),dt));*wt=*wa;
        *zwt=FSUB(FADD(-z[NSOIL-1],25.0f),FDIV(FDIV(*wa,1000.0f),0.2f));
        ml[NSOIL-1]=FSUB(ml[NSOIL-1],FMUL(*qin,dt));
        ml[NSOIL-1]=FADD(ml[NSOIL-1],f_max(0.0f,FSUB(*wa,5000.0f)));
        *wa=f_min(*wa,5000.0f);
    } else {
        if(iwt==NSOIL-2)
            *zwt=FSUB(-z[NSOIL-1],FDIV(FDIV(FSUB(*wt,5000.0f),ep[NSOIL-1]),1000.0f));
        else {
            float ws=0.0f;for(int k=iwt+2;k<NSOIL;++k)ws=FADD(ws,FMUL(ep[k],dz[k]));
            *zwt=FSUB(-z[iwt+1],FDIV(FDIV(FSUB(FSUB(*wt,5000.0f),ws),ep[iwt+1]),1000.0f));
        }
        float total=0.0f;for(int k=0;k<NSOIL;++k)total=FADD(total,FMUL(hk[k],dz[k]));
        for(int k=0;k<NSOIL;++k)ml[k]=FSUB(ml[k],FDIV(FMUL(FMUL(FMUL(*qdis,dt),hk[k]),dz[k]),total));
    }
    *zwt=f_max(1.5f,*zwt);
    for(int k=0;k<NSOIL-1;++k) {
        float xs=ml[k]<0.0f?FSUB(0.01f,ml[k]):0.0f;
        ml[k]=FADD(ml[k],xs);ml[k+1]=FSUB(ml[k+1],xs);
    }
    float xs=ml[NSOIL-1]<0.01f?FSUB(0.01f,ml[NSOIL-1]):0.0f;
    ml[NSOIL-1]=FADD(ml[NSOIL-1],xs);*wa=FSUB(*wa,xs);*wt=FSUB(*wt,xs);
    for(int k=0;k<NSOIL;++k)liquid[k]=FDIV(ml[k],dz[k]);
}

// ZWTEQ:7587-7625, the entire native 100-layer equilibrium search.
__device__ void equilibrium_water_table(const float *p,const float *z,
    const float *dz,const float *sh,float *zwt) {
    float deficit=0.0f;
    for(int k=0;k<NSOIL;++k)deficit=FADD(deficit,FMUL(FSUB(p[P_SMCMAX],sh[k]),dz[k]));
    float fine=FDIV(FMUL(3.0f,-z[NSOIL-1]),100.0f);
    *zwt=FSUB(FMUL(-3.0f,z[NSOIL-1]),0.001f);
    float accumulated=0.0f;
    for(int k=1;k<=100;++k) {
        float depth=FMUL((float)k,fine);
        float temp=FADD(1.0f,FDIV(FSUB(*zwt,depth),p[W_P_STRIDE+2]));
        accumulated=FADD(accumulated,FMUL(FMUL(p[P_SMCMAX],FSUB(1.0f,runoff_powf(temp,FDIV(-1.0f,p[P_BEXP])))),fine));
        if(fabsf(FSUB(accumulated,deficit))<=0.01f) {*zwt=depth;break;}
    }
}

// Native COMPUTE_VIC_SURFRUNOFF, LSM:7977-8047. Cumulative ZSOIL
// weighting is intentional; replacing it with layer thickness changes WRF.
__device__ void vic(const float *p,float dt,const float *smc,const float *z,
                    float rain,float *run,float *infil) {
    float moist=K_ZERO, capacity=K_ZERO;
    for(int k=0;k<NSOIL-2;++k) {
        moist=FADD(moist,FMUL(FMUL(smc[k],-1.0f),z[k]));
        capacity=FADD(capacity,FMUL(FMUL(p[P_SMCMAX+k],-1.0f),z[k]));
    }
    float b=p[W_P_STRIDE+4];
    float exponent=FDIV(b,FADD(K_ONE,b));
    float area=FSUB(K_ONE,runoff_powf(f_max(K_ZERO,FSUB(K_ONE,FDIV(moist,capacity))),exponent));
    area=f_min(K_ONE,f_max(K_ZERO,area));
    float imax=FMUL(FADD(K_ONE,b),capacity);
    float initial=FMUL(imax,FSUB(K_ONE,runoff_powf(FSUB(K_ONE,area),FDIV(K_ONE,b))));
    float amount=FMUL(rain,dt);
    if(rain==K_ZERO)*run=K_ZERO;
    else if(imax==K_ZERO)*run=amount;
    else if(FADD(initial,amount)>imax)*run=FADD(FSUB(amount,capacity),moist);
    else {
        float basis=FSUB(K_ONE,FDIV(FADD(initial,amount),imax));
        *run=FADD(FADD(FSUB(amount,capacity),moist),FMUL(capacity,runoff_powf(basis,FADD(K_ONE,b))));
    }
    *run=FDIV(*run,dt);
    if(*run<K_ZERO)*run=K_ZERO;
    if(*run>rain)*run=rain;
    *infil=FSUB(rain,*run);
}
// Native COMPUTE_XAJ_SURFRUNOFF, LSM:8053-8130. All arithmetic keeps
// Fortran's expression ordering, including the frozen impervious fraction.
__device__ void xaj(const float *p,float dt,const float *fcr,const float *smc,
                    const float *z,float rain,float *run,float *infil) {
    float wm=K_ZERO,wmmax=K_ZERO,sm=K_ZERO,smmax=K_ZERO;
    for(int k=0;k<NSOIL-2;++k) {
        float ref=p[W_P_STRIDE+k];
        if(FSUB(smc[k],ref)>K_ZERO) {
            sm=FADD(sm,FMUL(FMUL(FSUB(smc[k],ref),-1.0f),z[k]));
            wm=FADD(wm,FMUL(FMUL(ref,-1.0f),z[k]));
        } else wm=FADD(wm,FMUL(FMUL(smc[k],-1.0f),z[k]));
        wmmax=FADD(wmmax,FMUL(FMUL(ref,-1.0f),z[k]));
        smmax=FADD(smmax,FMUL(FMUL(FSUB(p[P_SMCMAX+k],ref),-1.0f),z[k]));
    }
    wm=f_min(wm,wmmax);sm=f_min(sm,smmax);
    float a=p[W_P_STRIDE+5],b=p[W_P_STRIDE+6],xx=p[W_P_STRIDE+7];
    float imp=FMUL(FMUL(fcr[0],rain),dt);
    float per=FMUL(FMUL(FSUB(K_ONE,fcr[0]),rain),dt);
    float fraction=FDIV(wm,wmmax);
    if(fraction<=FSUB(0.5f,a))
        per=FMUL(FMUL(per,runoff_powf(FSUB(0.5f,a),FSUB(K_ONE,b))),runoff_powf(fraction,b));
    else per=FMUL(per,FSUB(K_ONE,FMUL(runoff_powf(FADD(0.5f,a),FSUB(K_ONE,b)),runoff_powf(FSUB(K_ONE,fraction),b))));
    *run=rain==K_ZERO?K_ZERO:FADD(FMUL(per,FSUB(K_ONE,runoff_powf(FSUB(K_ONE,FDIV(sm,smmax)),xx))),imp);
    *run=FDIV(*run,dt);
    *run=f_max(K_ZERO,*run);*run=f_min(rain,*run);
    *infil=FSUB(rain,*run);
}

// BEGIN GENERATED NATIVE DYNAMIC VIC
// Public domain WRF v4; licenses/LICENSE-WRF-public-domain.txt.
// Generated by tools/noahmp_options_wrf461_oracle/generate_dynamic_vic.py.
// Native module_sf_noahmplsm.F:8408, complete RR1.
__device__ void rr1(const float*p,float i_0,float i_max,float yd,float&r1) {
    float tdepth;
    tdepth=FADD(i_0,yd);
    if((tdepth>i_max)) tdepth=i_max;
    r1=FSUB(yd,FMUL(FDIV(i_max,FADD(p[W_P_STRIDE+1],1.0f)),FSUB(runoff_powf(FSUB(1.0f,FDIV(i_0,i_max)),FADD(p[W_P_STRIDE+1],1.0f)),runoff_powf(FSUB(1.0f,FDIV(tdepth,i_max)),FADD(p[W_P_STRIDE+1],1.0f)))));
    if((r1<0.0f)) r1=0.0f;
}

// Native module_sf_noahmplsm.F:8433, complete RR2.
__device__ void rr2(float yd,float y0,float r1,float fmax,float fsur,float dt,float dp,float bb,float&r2) {
    if((yd>=y0)) {
    r2=FSUB(FSUB(dp,r1),FMUL(FMUL(fmax,dt),FSUB(1.0f,runoff_powf(FSUB(1.0f,FDIV(FSUB(dp,r1),FMUL(fmax,dt))),FADD(bb,1.0f)))));
    } else {
    r2=FSUB(FSUB(dp,r1),FMUL(fmax,dt));
    }
    if((r2<0.0f)) r2=0.0f;
}

// Native module_sf_noahmplsm.F:8607, complete PHILIP_INFIL.
__device__ void philip_infil(const float*p,const float*smc,const float*sice,float qinsur,float dt,float&facc,float&fsur,int inflmax) {
    float wdf;
    float wcnd;
    float sp;
    float ap;
    float fmax;
    int isoil;
    isoil=1;
    if((inflmax==1.0f)) {
    d_wdfcnd2(p[P_SMCWLT+(int)(isoil)-1],0.0f,p[P_SMCMAX+(int)(isoil)-1],p[P_BEXP+(int)(isoil)-1],p[P_DWSAT+(int)(isoil)-1],p[P_DKSAT+(int)(isoil)-1],&wdf,&wcnd);
    sp=sqrtf(FMUL(FMUL(2.0f,FSUB(p[P_SMCMAX+(int)(isoil)-1],p[P_SMCWLT+(int)(isoil)-1])),FSUB(p[P_DWSAT+(int)(isoil)-1],wdf)));
    ap=f_min(wcnd,FMUL(FDIV(2.0f,3.0f),p[P_DKSAT+(int)(isoil)-1]));
    ap=f_max(ap,FMUL(FDIV(1.0f,3.0f),p[P_DKSAT+(int)(isoil)-1]));
    fsur=FADD(FMUL(FMUL(FDIV(1.0f,2.0f),sp),runoff_powf(dt,FDIV(f_neg(1.0f),2.0f))),ap);
    if((fsur<0.0f)) fsur=wcnd;
    } else {
    d_wdfcnd2(smc[(int)(isoil)-1],sice[(int)(isoil)-1],p[P_SMCMAX+(int)(isoil)-1],p[P_BEXP+(int)(isoil)-1],p[P_DWSAT+(int)(isoil)-1],p[P_DKSAT+(int)(isoil)-1],&wdf,&wcnd);
    sp=sqrtf(FMUL(FMUL(2.0f,f_max(0.0f,FSUB(p[P_SMCMAX+(int)(isoil)-1],smc[(int)(isoil)-1]))),FSUB(p[P_DWSAT+(int)(isoil)-1],wdf)));
    ap=f_min(wcnd,FMUL(FDIV(2.0f,3.0f),p[P_DKSAT+(int)(isoil)-1]));
    ap=f_max(ap,FMUL(FDIV(1.0f,3.0f),p[P_DKSAT+(int)(isoil)-1]));
    fsur=FADD(FMUL(FMUL(FDIV(1.0f,2.0f),sp),runoff_powf(dt,FDIV(f_neg(1.0f),2.0f))),ap);
    if((p[P_DKSAT+(int)(isoil)-1]<qinsur)) {
    fsur=f_min(qinsur,fsur);
    } else {
    fsur=qinsur;
    }
    facc=FADD(facc,fsur);
    }
}

// Native module_sf_noahmplsm.F:8540, complete GREEN_AMPT_INFIL.
__device__ void green_ampt_infil(const float*p,const float*zsoil,const float*smc,const float*sice,float qinsur,float&facc,float&fsur,int inflmax) {
    float wdf;
    float wcnd;
    float jj;
    int isoil;
    isoil=1;
    if((inflmax==1.0f)) {
    d_wdfcnd2(p[P_SMCWLT+(int)(isoil)-1],0.0f,p[P_SMCMAX+(int)(isoil)-1],p[P_BEXP+(int)(isoil)-1],p[P_DWSAT+(int)(isoil)-1],p[P_DKSAT+(int)(isoil)-1],&wdf,&wcnd);
    jj=FMUL(FMUL(FMUL(p[W_P_STRIDE+2],FSUB(p[P_SMCMAX+(int)(isoil)-1],p[P_SMCWLT+(int)(isoil)-1])),f_neg(1.0f)),zsoil[(int)(isoil)-1]);
    fsur=FADD(p[P_DKSAT+(int)(isoil)-1],FMUL(FDIV(jj,1e-05f),FSUB(p[P_DKSAT+(int)(isoil)-1],wcnd)));
    if((fsur<0.0f)) fsur=wcnd;
    } else {
    d_wdfcnd2(smc[(int)(isoil)-1],sice[(int)(isoil)-1],p[P_SMCMAX+(int)(isoil)-1],p[P_BEXP+(int)(isoil)-1],p[P_DWSAT+(int)(isoil)-1],p[P_DKSAT+(int)(isoil)-1],&wdf,&wcnd);
    jj=FMUL(FMUL(FMUL(p[W_P_STRIDE+2],f_max(0.0f,FSUB(p[P_SMCMAX+(int)(isoil)-1],smc[(int)(isoil)-1]))),f_neg(1.0f)),zsoil[(int)(isoil)-1]);
    fsur=FADD(p[P_DKSAT+(int)(isoil)-1],FMUL(FDIV(jj,facc),FSUB(p[P_DKSAT+(int)(isoil)-1],wcnd)));
    if((p[P_DKSAT+(int)(isoil)-1]<qinsur)) {
    fsur=f_min(qinsur,fsur);
    } else {
    fsur=qinsur;
    }
    facc=FADD(facc,fsur);
    }
}

// Native module_sf_noahmplsm.F:8458, complete SMITH_PARLANGE_INFIL.
__device__ void smith_parlange_infil(const float*p,const float*zsoil,const float*smc,const float*sice,float qinsur,float&facc,float&fsur,int inflmax) {
    float wdf;
    float wcnd;
    float gam;
    float jj;
    int isoil;
    gam=0.82f;
    isoil=1;
    if((inflmax==1.0f)) {
    d_wdfcnd2(p[P_SMCWLT+(int)(isoil)-1],0.0f,p[P_SMCMAX+(int)(isoil)-1],p[P_BEXP+(int)(isoil)-1],p[P_DWSAT+(int)(isoil)-1],p[P_DKSAT+(int)(isoil)-1],&wdf,&wcnd);
    jj=FMUL(FMUL(FMUL(p[W_P_STRIDE+2],FSUB(p[P_SMCMAX+(int)(isoil)-1],p[P_SMCWLT+(int)(isoil)-1])),f_neg(1.0f)),zsoil[(int)(isoil)-1]);
    fsur=FADD(p[P_DKSAT+(int)(isoil)-1],FDIV(FMUL(gam,FSUB(p[P_DKSAT+(int)(isoil)-1],wcnd)),FSUB(glibc_expf(FDIV(FMUL(gam,facc),jj)),1.0f)));
    if((p[P_DKSAT+(int)(isoil)-1]<qinsur)) {
    fsur=f_min(qinsur,fsur);
    } else {
    fsur=qinsur;
    }
    if((fsur<0.0f)) fsur=wcnd;
    } else {
    d_wdfcnd2(smc[(int)(isoil)-1],sice[(int)(isoil)-1],p[P_SMCMAX+(int)(isoil)-1],p[P_BEXP+(int)(isoil)-1],p[P_DWSAT+(int)(isoil)-1],p[P_DKSAT+(int)(isoil)-1],&wdf,&wcnd);
    jj=FMUL(FMUL(FMUL(p[W_P_STRIDE+2],f_max(0.0f,FSUB(p[P_SMCMAX+(int)(isoil)-1],smc[(int)(isoil)-1]))),f_neg(1.0f)),zsoil[(int)(isoil)-1]);
    if((jj==0.0f)) {
    fsur=wcnd;
    } else {
    fsur=FADD(p[P_DKSAT+(int)(isoil)-1],FDIV(FMUL(gam,FSUB(p[P_DKSAT+(int)(isoil)-1],wcnd)),FSUB(glibc_expf(FDIV(FMUL(gam,facc),jj)),1.0f)));
    }
    if((p[P_DKSAT+(int)(isoil)-1]<qinsur)) {
    fsur=f_min(qinsur,fsur);
    } else {
    fsur=qinsur;
    }
    facc=FADD(facc,fsur);
    }
}

// Native module_sf_noahmplsm.F:8133, complete DYNAMIC_VIC.
__device__ void dynamic_vic(const float*p,float dt,const float*smc,const float*sh2o,const float*sice,float sicemax,const float*zsoil,float qinsur,float&facc,float&pddum,float&runsrf,int opt_infdv) {
    float bb;
    float top_moist;
    float top_max_moist;
    float dp;
    float i_0;
    float i_max;
    float fsur;
    float fmax;
    float runoffsat;
    float runoffinf;
    float infiltrtn;
    float tempr1;
    float tempr2;
    float r1;
    float r2;
    float yd;
    float yd_old;
    float yd0;
    float temp1, error;
    int iz, izmax, inflmax;
    yd=__int_as_float(0x7fc00000);
    top_moist=0.0f;
    top_max_moist=0.0f;
    bb=1.0f;
    dp=0.0f;
    i_max=0.0f;
    i_0=0.0f;
    runoffsat=0.0f;
    runoffinf=0.0f;
    infiltrtn=0.0f;
    runsrf=0.0f;
    izmax=20;
    error=FMUL(1.388889e-07f,dt);
    bb=p[W_P_STRIDE+0];
    for(iz=1;iz<=NSOIL-2;++iz) {
    top_moist=FADD(top_moist,FMUL(FMUL(smc[(int)(iz)-1],f_neg(1.0f)),zsoil[(int)(iz)-1]));
    top_max_moist=FADD(top_max_moist,FMUL(FMUL(p[P_SMCMAX+(int)(iz)-1],f_neg(1.0f)),zsoil[(int)(iz)-1]));
    }
    if((top_moist>top_max_moist)) top_moist=top_max_moist;
    dp=FMUL(qinsur,dt);
    i_max=FMUL(top_max_moist,FADD(p[W_P_STRIDE+1],1.0f));
    i_0=FMUL(i_max,FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(top_moist,top_max_moist),FDIV(1.0f,FADD(1.0f,p[W_P_STRIDE+1]))))));
    inflmax=0;
    if((opt_infdv==1.0f)) {
    philip_infil(p,smc,sice,qinsur,dt,facc,fsur,inflmax);
    } else if((opt_infdv==2.0f)) {
    green_ampt_infil(p,zsoil,smc,sice,qinsur,facc,fsur,inflmax);
    } else if((opt_infdv==3.0f)) {
    smith_parlange_infil(p,zsoil,smc,sice,qinsur,facc,fsur,inflmax);
    }
    fmax=FMUL(FADD(bb,1.0f),fsur);
    if((dp<=0.0f)) {
    runoffsat=0.0f;
    runoffinf=0.0f;
    infiltrtn=0.0f;
    goto label_2001;
    } else {
    if(((top_moist>=top_max_moist)&&(i_0>=i_max))) {
    top_moist=top_max_moist;
    i_0=i_max;
    runoffsat=dp;
    runoffinf=0.0f;
    infiltrtn=0.0f;
    goto label_2001;
    } else {
    i_0=FMUL(i_max,FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(top_moist,top_max_moist),FDIV(1.0f,FADD(1.0f,p[W_P_STRIDE+1]))))));
    if((FADD(dp,i_0)>i_max)) {
    if((FMUL(fmax,dt)>=dp)) {
    yd=FSUB(i_max,i_0);
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    temp1=FSUB(FSUB(FSUB(i_max,i_0),tempr1),FMUL(FMUL(fsur,dt),FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(FSUB(dp,tempr1),FMUL(fmax,dt)),FADD(bb,1.0f))))));
    if((temp1<=0.0f)) {
    yd=FSUB(i_max,i_0);
    infiltrtn=FSUB(top_max_moist,top_moist);
    runoffsat=FSUB(dp,infiltrtn);
    runoffinf=0.0f;
    top_moist=top_max_moist;
    i_0=i_max;
    goto label_2001;
    } else {
    yd=0.0f;
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    yd=FADD(tempr1,FMUL(FMUL(fsur,dt),FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(FSUB(dp,tempr1),FMUL(fmax,dt)),FADD(bb,1.0f))))));
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    goto label_1003;
    }
    }
    }
    } else {
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    if((FADD(tempr1,FMUL(fmax,dt))<=dp)) {
    if((FSUB(FSUB(FSUB(i_max,i_0),tempr1),FMUL(fmax,dt))<=0.0f)) {
    yd=FSUB(i_max,i_0);
    infiltrtn=FSUB(top_max_moist,top_moist);
    runoffsat=FSUB(dp,infiltrtn);
    runoffinf=0.0f;
    top_moist=top_max_moist;
    i_0=i_max;
    goto label_2001;
    } else {
    yd=0.0f;
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    yd=FADD(tempr1,FMUL(fsur,dt));
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    goto label_1003;
    }
    }
    }
    } else {
    yd=FDIV(dp,2.0f);
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    yd=FADD(FSUB(FSUB(yd,tempr1),FMUL(fsur,dt)),dp);
    if((yd<=0.0f)) yd=0.0f;
    if((yd>=dp)) yd=dp;
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    yd0=yd;
    break;
    }
    }
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    tempr2=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    rr2(yd,yd0,tempr1,fmax,fsur,dt,dp,bb,tempr2);
    yd=FSUB(dp,tempr2);
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    goto label_1003;
    }
    }
    label_1003: ;
    if((yd<=0.0f)) yd=0.0f;
    if((yd>=dp)) yd=dp;
    rr1(p,i_0,i_max,yd,r1);
    runoffsat=r1;
    runoffinf=FSUB(dp,yd);
    infiltrtn=FSUB(yd,runoffsat);
    top_moist=FADD(top_moist,infiltrtn);
    yd=FADD(i_0,yd);
    if((top_moist<=0.0f)) top_moist=0.0f;
    if((top_moist>=top_max_moist)) top_moist=top_max_moist;
    i_0=FMUL(i_max,FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(top_moist,top_max_moist),FDIV(1.0f,FADD(1.0f,p[W_P_STRIDE+1]))))));
    goto label_2001;
    }
    }
    } else {
    if((FMUL(fmax,dt)>=dp)) {
    yd=FDIV(dp,2.0f);
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    yd=FADD(tempr1,FMUL(FMUL(fsur,dt),FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(FSUB(dp,tempr1),FMUL(fmax,dt)),FADD(bb,1.0f))))));
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    goto label_1004;
    }
    }
    } else {
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    if((FADD(tempr1,FMUL(fmax,dt))<=dp)) {
    yd=FDIV(dp,2.0f);
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    yd=FADD(tempr1,FMUL(fsur,dt));
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    goto label_1004;
    }
    }
    } else {
    yd=0.0f;
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    yd=FSUB(FADD(FSUB(dp,FMUL(fmax,dt)),yd),tempr1);
    if((yd<=0.0f)) yd=0.0f;
    if((yd>=dp)) yd=dp;
    tempr1=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    if(((fabsf(FSUB(FADD(tempr1,FMUL(fmax,dt)),dp))<=error)||(iz==izmax))) {
    yd0=yd;
    break;
    }
    }
    for(iz=1;iz<=izmax;++iz) {
    yd_old=yd;
    tempr1=0.0f;
    tempr2=0.0f;
    rr1(p,i_0,i_max,yd,tempr1);
    rr2(yd,yd0,tempr1,fmax,fsur,dt,dp,bb,tempr2);
    yd=FSUB(dp,tempr2);
    if(((fabsf(FSUB(yd,yd_old))<=error)||(iz==izmax))) {
    goto label_1004;
    }
    }
    }
    }
    label_1004: ;
    if((yd<=0.0f)) yd=0.0f;
    if((yd>=dp)) yd=dp;
    r1=0.0f;
    rr1(p,i_0,i_max,yd,r1);
    runoffsat=r1;
    runoffinf=FSUB(dp,yd);
    infiltrtn=FSUB(yd,runoffsat);
    top_moist=FADD(top_moist,infiltrtn);
    if((top_moist<=0.0f)) top_moist=0.0f;
    if((top_moist>=top_max_moist)) top_moist=top_max_moist;
    i_0=FMUL(i_max,FSUB(1.0f,FSUB(1.0f,runoff_powf(FDIV(top_moist,top_max_moist),FDIV(1.0f,FADD(1.0f,p[W_P_STRIDE+1]))))));
    }
    }
    }
    label_2001: ;
    runsrf=FDIV(FADD(runoffsat,runoffinf),dt);
    runsrf=f_min(runsrf,qinsur);
    runsrf=f_max(runsrf,0.0f);
    pddum=FSUB(qinsur,runsrf);
}
// END GENERATED NATIVE DYNAMIC VIC

// BEGIN GENERATED NATIVE IRRIGATION
// WRF public domain; licenses/LICENSE-WRF-public-domain.txt.
// Complete native IRR_PHILIP_INFIL; module_sf_noahmplsm.F:9525
__device__ void irr_philip_infil(const float*irr,const float*smc,const float*sh2o,const float*sice,float dt,float&fsur) {
    float wdf;
    float wcnd;
    float sp;
    float ap;
    float sicemax;
    int isoil;
    int k;
    sicemax=0.0f;
    for(k =(int)(1.0f);k <=(int)(NSOIL);++k ) {
    if((sice[(int)(k)-1]>sicemax)) sicemax=sice[(int)(k)-1];
    }
    isoil=1.0f;
    d_wdfcnd2(sh2o[(int)(isoil)-1],sicemax,irr[0+(int)(isoil)-1],irr[12+(int)(isoil)-1],irr[20+(int)(isoil)-1],irr[16+(int)(isoil)-1],&wdf,&wcnd);
    sp=sqrtf(FMUL(FMUL(2.0f,f_max(0.0f,FSUB(irr[0+(int)(isoil)-1],smc[(int)(isoil)-1]))),FSUB(irr[20+(int)(isoil)-1],wdf)));
    ap=f_min(wcnd,FMUL(FDIV(2.0f,3.0f),irr[16+(int)(isoil)-1]));
    ap=f_max(ap,FMUL(FDIV(1.0f,3.0f),irr[16+(int)(isoil)-1]));
    fsur=FADD(FMUL(FMUL(0.5f,sp),runoff_powf(dt,f_neg(0.5f))),ap);
    fsur=f_max(0.0f,fsur);
}

// Complete native TRIGGER_IRRIGATION; module_sf_noahmplsm.F:9235
__device__ void trigger_irrigation(const float*irr,int nroot,int opt_irr,int opt_irrm,const float*zsoil,const float*sh2o,float fveg,float julian,float irrfra,float lai,float sifac,float mifac,float fifac,int&ircntsi,int&ircntmi,int&ircntfi,float&iramtsi,float&iramtmi,float&iramtfi) {
    float smcavl;
    float smclim;
    float smcsat;
    float irrwatamt;
    bool irr_active;
    int k;
    irr_active=true;
    if((opt_irr==2.0f)) {
    if(((julian<irr[33])||(julian>FSUB(irr[34],irr[25])))) irr_active=false;
    } else if((opt_irr==3.0f)) {
    if((lai<irr[26])) irr_active=false;
    } else if(((opt_irr>3.0f)||(opt_irr<1.0f))) {
    irr_active=false;
    }
    if(irr_active) {
    smcavl=0.0f;
    smclim=0.0f;
    smcavl=FMUL(FMUL(FSUB(sh2o[(int)(1.0f)-1],irr[4+(int)(1.0f)-1]),f_neg(1.0f)),zsoil[(int)(1.0f)-1]);
    smclim=FMUL(FMUL(FSUB(irr[8+(int)(1.0f)-1],irr[4+(int)(1.0f)-1]),f_neg(1.0f)),zsoil[(int)(1.0f)-1]);
    for(k =(int)(2.0f);k <=(int)(nroot);++k ) {
    smcavl=FADD(smcavl,FMUL(FSUB(sh2o[(int)(k)-1],irr[4+(int)(k)-1]),FSUB(zsoil[(int)(FSUB(k,1.0f))-1],zsoil[(int)(k)-1])));
    smclim=FADD(smclim,FMUL(FSUB(irr[8+(int)(k)-1],irr[4+(int)(k)-1]),FSUB(zsoil[(int)(FSUB(k,1.0f))-1],zsoil[(int)(k)-1])));
    }
    if((FDIV(smcavl,smclim)<=irr[27])) {
    irrwatamt=FMUL(FMUL(FSUB(smclim,smcavl),irrfra),fveg);
    if(((iramtsi==0.0f)&&(sifac>0.0f)&&(opt_irrm==0.0f))) {
    iramtsi=FMUL(sifac,irrwatamt);
    ++ircntsi;
    } else if(((iramtsi==0.0f)&&(opt_irrm==1.0f))) {
    iramtsi=irrwatamt;
    ++ircntsi;
    }
    if(((iramtmi==0.0f)&&(mifac>0.0f)&&(opt_irrm==0.0f))) {
    iramtmi=FMUL(mifac,irrwatamt);
    ++ircntmi;
    } else if(((iramtmi==0.0f)&&(opt_irrm==2.0f))) {
    iramtmi=irrwatamt;
    ++ircntmi;
    }
    if(((iramtfi==0.0f)&&(fifac>0.0f)&&(opt_irrm==0.0f))) {
    iramtfi=FMUL(FMUL(fifac,irrwatamt),FDIV(1.0f,FSUB(1.0f,irr[28])));
    ++ircntfi;
    } else if(((iramtfi==0.0f)&&(opt_irrm==3.0f))) {
    iramtfi=FMUL(irrwatamt,FDIV(1.0f,FSUB(1.0f,irr[28])));
    ++ircntfi;
    }
    } else {
    irrwatamt=0.0f;
    iramtsi=0.0f;
    iramtmi=0.0f;
    iramtfi=0.0f;
    }
    }
}

// Complete native SPRINKLER_IRRIGATION; module_sf_noahmplsm.F:9352
__device__ void sprinkler_irrigation(const float*irr,float dt,const float*sh2o,const float*smc,float t2,float windu,float windv,float eair,float sifac,float&iramtsi,float&irevplos,float&irsirate) {
    int lane_k;
    float sice[NSOIL];
    float fsur;
    float temp_rate;
    float windspeed;
    float irrloss;
    float esat1;
    for(lane_k=1;lane_k<=NSOIL;++lane_k) sice[(int)(lane_k)-1]=f_max(0.0f,FSUB(smc[(int)(lane_k)-1],sh2o[(int)(lane_k)-1]));
    irr_philip_infil(irr,smc,sh2o,sice,dt,fsur);
    temp_rate=FDIV(FMUL(FMUL(irr[29],FDIV(1.0f,1000.0f)),dt),3600.0f);
    irsirate=f_min(f_min(FMUL(fsur,dt),iramtsi),temp_rate);
    windspeed=sqrtf(FADD(FMUL(windu,windu),FMUL(windv,windv)));
    esat1=FMUL(610.8f,glibc_expf(FDIV(FMUL(17.27f,FSUB(t2,273.15f)),FADD(237.3f,FSUB(t2,273.15f)))));
    if((t2>273.15f)) {
    irrloss=FMUL(FMUL(FMUL(4.375f,glibc_expf(FMUL(0.106f,windspeed))),runoff_powf(FMUL(FSUB(esat1,eair),0.01f),f_neg(0.092f))),runoff_powf(FSUB(t2,273.15f),f_neg(0.102f)));
    } else {
    irrloss=FMUL(FMUL(4.337f,glibc_expf(FMUL(0.077f,windspeed))),runoff_powf(FMUL(FSUB(esat1,eair),0.01f),f_neg(0.098f)));
    }
    if(isnan(irrloss)) irrloss=4.0f;
    if(((irrloss>100.0f)||(irrloss<0.0f))) irrloss=4.0f;
    irsirate=FMUL(irsirate,sifac);
    if((irsirate>=iramtsi)) {
    irsirate=iramtsi;
    iramtsi=0.0f;
    } else {
    iramtsi=FSUB(iramtsi,irsirate);
    }
    irevplos=FMUL(FMUL(irsirate,irrloss),FDIV(1.0f,100.0f));
    irsirate=FSUB(irsirate,irevplos);
}

// Complete native MICRO_IRRIGATION; module_sf_noahmplsm.F:9434
__device__ void micro_irrigation(const float*irr,float dt,const float*sh2o,const float*smc,const float*sice,float mifac,float&iramtmi,float&irmirate) {
    float fsur;
    float temp_rate;
    irr_philip_infil(irr,smc,sh2o,sice,dt,fsur);
    temp_rate=FDIV(FMUL(FMUL(irr[30],FDIV(1.0f,1000.0f)),dt),3600.0f);
    irmirate=f_min(f_min(FMUL(FMUL(0.5f,fsur),dt),iramtmi),temp_rate);
    irmirate=FMUL(irmirate,mifac);
    if((irmirate>=iramtmi)) {
    irmirate=iramtmi;
    iramtmi=0.0f;
    } else {
    iramtmi=FSUB(iramtmi,irmirate);
    }
}

// Complete native FLOOD_IRRIGATION; module_sf_noahmplsm.F:9478
__device__ void flood_irrigation(const float*irr,float dt,const float*sh2o,const float*smc,const float*sice,float fifac,float&iramtfi,float&irfirate) {
    float fsur;
    float temp_rate;
    irr_philip_infil(irr,smc,sh2o,sice,dt,fsur);
    irfirate=FMUL(FMUL(fsur,dt),irr[31]);
    irfirate=FMUL(irfirate,fifac);
    if((irfirate>=iramtfi)) {
    irfirate=iramtfi;
    iramtfi=0.0f;
    } else {
    iramtfi=FSUB(iramtfi,irfirate);
    }
}
// END GENERATED NATIVE IRRIGATION

// BEGIN GENERATED NATIVE TILE DRAINAGE
// WRF public domain; licenses/LICENSE-WRF-public-domain.txt.
// Complete native TD_EQUIVALENT_DEPTH; module_sf_noahmplsm.F:9055
__device__ void td_equivalent_depth(float td_d,float td_l,float td_rd,float&td_de) {
    float pii =3.142857074737548828125f;
    float td_x;
    float td_fx;
    float ex;
    float term;
    int i;
    td_fx=0.0f;
    ex=0.0f;
    term=0.0f;
    td_x=FDIV(FMUL(FMUL(2.0f,pii),td_d),td_l);
    if((td_x>0.5f)) {
    for(i=1;i<=45;i+=2) {
    ex=glibc_expf(FMUL(FMUL(f_neg(2.0f),i),td_x));
    term=FDIV(FMUL(4.0f,ex),FMUL(i,FSUB(1.0f,ex)));
    td_fx=FADD(td_fx,term);
    if((term<1e-06f)) {
    td_de=FDIV(FDIV(FMUL(pii,td_l),8.0f),FADD(glibc_logf(FDIV(td_l,FMUL(pii,td_rd))),td_fx));
    break;
    }
    }
    } else if((td_x<1e-08f)) {
    td_de=td_d;
    } else {
    td_fx=FADD(FDIV(FMUL(pii,pii),FMUL(4.0f,td_x)),glibc_logf(FDIV(td_x,FMUL(2.0f,pii))));
    td_de=FDIV(FDIV(FMUL(pii,td_l),8.0f),FADD(glibc_logf(FDIV(td_l,FMUL(pii,td_rd))),td_fx));
    }
    if(((td_de<0.0f)&&(i<=2.0f))) td_de=td_d;
}

// Complete native TD_FINDZWAT; module_sf_noahmplsm.F:9101
__device__ void td_findzwat(const float*p,const float*wlt,const float*smc,const float*sh2o,const float*sice,const float*zsoil,const float*sldpth,float&watbled) {
    float cwatavail;
    int satlyrchk;
    int k;
    int i;
    int j;
    satlyrchk=0.0f;
    cwatavail=0.0f;
    for(k=NSOIL;k>=1;k+=-1) {
    if(((smc[(int)(k)-1]>=p[(int)(k)-1])&&(p[(int)(k)-1]>wlt[(int)(k)-1]))) {
    if(((satlyrchk==FADD(k,1.0f))||(k==NSOIL))) satlyrchk=k;
    }
    }
    if((satlyrchk!=0.0f)) {
    if((satlyrchk!=1.0f)) {
    watbled=f_neg(zsoil[(int)(FSUB(satlyrchk,1.0f))-1]);
    } else {
    watbled=0.0f;
    }
    for(k=satlyrchk;k<=NSOIL;k+=1) {
    cwatavail=FADD(cwatavail,FMUL(FSUB(smc[(int)(k)-1],p[(int)(k)-1]),sldpth[(int)(k)-1]));
    }
    } else {
    watbled=f_neg(zsoil[(int)(NSOIL)-1]);
    satlyrchk=FADD(NSOIL,1.0f);
    }
}

// Complete native TILE_DRAIN; module_sf_noahmplsm.F:8687
__device__ void tile_drain(const float*p,float*sh2o,float*smc,const float*sice,const float*zsoil,float&qtldrn,float dt) {
    int k;
    float tdrvol;
    float ovrfc[NSOIL];
    float avfc[NSOIL];
    float zlayer[NSOIL];
    float tdfrac[NSOIL];
    float tddc;
    float tdsum;
    tdrvol=0.0f;
    for(int q=0;q<NSOIL;++q) ovrfc[q]=0.0f;
    qtldrn=0.0f;
    for(int q=0;q<NSOIL;++q) zlayer[q]=0.0f;
    for(int q=0;q<NSOIL;++q) avfc[q]=0.0f;
    tdsum=0.0f;
    for(int q=0;q<NSOIL;++q) tdfrac[q]=0.0f;
    tddc=FDIV(FMUL(p[6],dt),FMUL(24.0f,3600.0f));
    for(k=1;k<=NSOIL;k+=1) {
    if((k==1.0f)) {
    zlayer[(int)(k)-1]=FMUL(f_neg(1.0f),zsoil[(int)(k)-1]);
    } else {
    zlayer[(int)(k)-1]=FSUB(zsoil[(int)(FSUB(k,1.0f))-1],zsoil[(int)(k)-1]);
    }
    }
    if((p[14]==0.0f)) {
    k=p[13];
    avfc[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FMUL(p[5],avfc[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]>0.0f)) {
    if((ovrfc[(int)(k)-1]>tddc)) ovrfc[(int)(k)-1]=tddc;
    tdrvol=FADD(tdrvol,ovrfc[(int)(k)-1]);
    sh2o[(int)(k)-1]=FSUB(sh2o[(int)(k)-1],FDIV(ovrfc[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    }
    } else if((p[14]==1.0f)) {
    for(k=1;k<=2;k+=1) {
    avfc[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FMUL(p[5],avfc[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]<0.0f)) ovrfc[(int)(k)-1]=0.0f;
    tdsum=FADD(tdsum,ovrfc[(int)(k)-1]);
    }
    for(k=1;k<=2;k+=1) {
    if((ovrfc[(int)(k)-1]!=0.0f)) {
    tdfrac[(int)(k)-1]=FDIV(ovrfc[(int)(k)-1],tdsum);
    }
    }
    if((tdsum>0.0f)) {
    if((tdsum>tddc)) tdsum=tddc;
    tdrvol=FADD(tdrvol,tdsum);
    for(k=1;k<=2;k+=1) {
    ovrfc[(int)(k)-1]=FMUL(tdfrac[(int)(k)-1],tdsum);
    sh2o[(int)(k)-1]=FSUB(sh2o[(int)(k)-1],FDIV(ovrfc[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    }
    }
    } else if((p[14]==2.0f)) {
    for(k=1;k<=3;k+=1) {
    avfc[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FMUL(p[5],avfc[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]<0.0f)) ovrfc[(int)(k)-1]=0.0f;
    tdsum=FADD(tdsum,ovrfc[(int)(k)-1]);
    }
    for(k=1;k<=3;k+=1) {
    if((ovrfc[(int)(k)-1]!=0.0f)) {
    tdfrac[(int)(k)-1]=FDIV(ovrfc[(int)(k)-1],tdsum);
    }
    }
    if((tdsum>0.0f)) {
    if((tdsum>tddc)) tdsum=tddc;
    tdrvol=FADD(tdrvol,tdsum);
    for(k=1;k<=3;k+=1) {
    ovrfc[(int)(k)-1]=FMUL(tdfrac[(int)(k)-1],tdsum);
    sh2o[(int)(k)-1]=FSUB(sh2o[(int)(k)-1],FDIV(ovrfc[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    }
    }
    } else if((p[14]==3.0f)) {
    for(k=2;k<=3;k+=1) {
    avfc[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FMUL(p[5],avfc[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]<0.0f)) ovrfc[(int)(k)-1]=0.0f;
    tdsum=FADD(tdsum,ovrfc[(int)(k)-1]);
    }
    for(k=2;k<=3;k+=1) {
    if((ovrfc[(int)(k)-1]!=0.0f)) {
    tdfrac[(int)(k)-1]=FDIV(ovrfc[(int)(k)-1],tdsum);
    }
    }
    if((tdsum>0.0f)) {
    if((tdsum>tddc)) tdsum=tddc;
    tdrvol=FADD(tdrvol,tdsum);
    for(k=2;k<=3;k+=1) {
    ovrfc[(int)(k)-1]=FMUL(tdfrac[(int)(k)-1],tdsum);
    sh2o[(int)(k)-1]=FSUB(sh2o[(int)(k)-1],FDIV(ovrfc[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    }
    }
    } else if((p[14]==4.0f)) {
    for(k=3;k<=4;k+=1) {
    avfc[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FMUL(p[5],avfc[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]<0.0f)) ovrfc[(int)(k)-1]=0.0f;
    tdsum=FADD(tdsum,ovrfc[(int)(k)-1]);
    }
    for(k=3;k<=4;k+=1) {
    if((ovrfc[(int)(k)-1]!=0.0f)) {
    tdfrac[(int)(k)-1]=FDIV(ovrfc[(int)(k)-1],tdsum);
    }
    }
    if((tdsum>0.0f)) {
    if((tdsum>tddc)) tdsum=tddc;
    tdrvol=FADD(tdrvol,tdsum);
    for(k=3;k<=4;k+=1) {
    ovrfc[(int)(k)-1]=FMUL(tdfrac[(int)(k)-1],tdsum);
    sh2o[(int)(k)-1]=FSUB(sh2o[(int)(k)-1],FDIV(ovrfc[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    }
    }
    } else if((p[14]==5.0f)) {
    for(k=1;k<=4;k+=1) {
    avfc[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FMUL(p[5],avfc[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]<0.0f)) ovrfc[(int)(k)-1]=0.0f;
    tdsum=FADD(tdsum,ovrfc[(int)(k)-1]);
    }
    for(k=1;k<=4;k+=1) {
    if((ovrfc[(int)(k)-1]!=0.0f)) {
    tdfrac[(int)(k)-1]=FDIV(ovrfc[(int)(k)-1],tdsum);
    }
    }
    if((tdsum>0.0f)) {
    if((tdsum>tddc)) tdsum=tddc;
    tdrvol=FADD(tdrvol,tdsum);
    for(k=1;k<=4;k+=1) {
    ovrfc[(int)(k)-1]=FMUL(tdfrac[(int)(k)-1],tdsum);
    sh2o[(int)(k)-1]=FSUB(sh2o[(int)(k)-1],FDIV(ovrfc[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    }
    }
    }
    qtldrn=FDIV(tdrvol,dt);
}

// Complete native TILE_HOOGHOUDT; module_sf_noahmplsm.F:8868
__device__ void tile_hooghoudt(const float*p,const float*wlt,const float*wcnd,float*sh2o,float*smc,const float*sice,const float*zsoil,const float*dzsnso,float dt,float dx,float&qtldrn,float&zwt) {
    int k;
    float td_satz[NSOIL];
    float klatk[NSOIL];
    float td_ttsz;
    float td_lq;
    float dtopl;
    float xx;
    float yy;
    float klat;
    float td_hail;
    float td_depth;
    float td_hemd;
    float zlayer[NSOIL];
    float ovrfc[NSOIL];
    int ndrains;
    float tddc;
    float rmsh2o[NSOIL];
    float qtldrn1;
    float td_dd;
    float ovrfcs;
    for(int q=0;q<NSOIL;++q) td_satz[q]=0.0f;
    dtopl=0.0f;
    td_lq=0.0f;
    td_ttsz=0.0f;
    tddc=FDIV(FMUL(FMUL(p[7],1000.0f),dt),FMUL(24.0f,3600.0f));
    for(k=1;k<=NSOIL;k+=1) {
    if((k==1.0f)) {
    zlayer[(int)(k)-1]=FMUL(f_neg(1.0f),zsoil[(int)(k)-1]);
    } else {
    zlayer[(int)(k)-1]=FSUB(zsoil[(int)(FSUB(k,1.0f))-1],zsoil[(int)(k)-1]);
    }
    }
    td_findzwat(p,wlt,smc,sh2o,sice,zsoil,zlayer,zwt);
    yy=zwt;
    if((yy>p[11])) yy=p[11];
    for(k=1;k<=NSOIL;k+=1) {
    if((yy>FMUL(f_neg(1.0f),zsoil[(int)(k)-1]))) {
    td_satz[(int)(k)-1]=0.0f;
    } else {
    td_satz[(int)(k)-1]=FSUB(FMUL(f_neg(1.0f),zsoil[(int)(k)-1]),yy);
    xx=FSUB(FMUL(f_neg(1.0f),zsoil[(int)(k)-1]),dtopl);
    if((td_satz[(int)(k)-1]>xx)) td_satz[(int)(k)-1]=xx;
    }
    dtopl=FMUL(f_neg(1.0f),zsoil[(int)(k)-1]);
    }
    ovrfcs=0.0f;
    for(k=1;k<=NSOIL;k+=1) {
    ovrfc[(int)(k)-1]=FMUL(FMUL(FSUB(sh2o[(int)(k)-1],FSUB(p[(int)(k)-1],sice[(int)(k)-1])),zlayer[(int)(k)-1]),1000.0f);
    if((ovrfc[(int)(k)-1]<0.0f)) ovrfc[(int)(k)-1]=0.0f;
    ovrfcs=FADD(ovrfcs,ovrfc[(int)(k)-1]);
    }
    for(k=1;k<=NSOIL;k+=1) {
    klatk[(int)(k)-1]=FMUL(FMUL(wcnd[(int)(k)-1],p[4]),dt);
    td_lq=FADD(td_lq,FMUL(td_satz[(int)(k)-1],klatk[(int)(k)-1]));
    td_ttsz=FADD(td_ttsz,td_satz[(int)(k)-1]);
    }
    if((td_ttsz<0.001f)) td_ttsz=0.001f;
    if((td_lq<0.001f)) td_lq=0.0f;
    klat=FDIV(td_lq,td_ttsz);
    td_dd=FSUB(p[11],p[10]);
    td_equivalent_depth (td_dd,p[9],p[8],td_hail);
    td_depth=FADD(td_hail,p[10]);
    td_hemd=FSUB(p[10],yy);
    if((td_hemd<=0.0f)) {
    qtldrn=0.0f;
    } else {
    qtldrn=FDIV(FADD(FMUL(FMUL(FMUL(8.0f,klat),td_hail),td_hemd),FMUL(FMUL(FMUL(4.0f,klat),td_hemd),td_hemd)),FMUL(p[9],p[9]));
    }
    qtldrn=FMUL(qtldrn,1000.0f);
    if((qtldrn<=0.0f)) qtldrn=0.0f;
    if((qtldrn>tddc)) qtldrn=tddc;
    ndrains=int(FDIV(dx,p[9]));
    qtldrn=FMUL(qtldrn,ndrains);
    if((qtldrn>ovrfcs)) qtldrn=ovrfcs;
    qtldrn1=qtldrn;
    for(k=1;k<=NSOIL;k+=1) {
    if((qtldrn1>0.0f)) {
    if(((td_satz[(int)(k)-1]>0.0f)&&(ovrfc[(int)(k)-1]>0.0f))) {
    rmsh2o[(int)(k)-1]=FSUB(ovrfc[(int)(k)-1],qtldrn1);
    if((rmsh2o[(int)(k)-1]>0.0f)) {
    sh2o[(int)(k)-1]=FADD(FSUB(p[(int)(k)-1],sice[(int)(k)-1]),FDIV(rmsh2o[(int)(k)-1],FMUL(zlayer[(int)(k)-1],1000.0f)));
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    break;
    } else {
    sh2o[(int)(k)-1]=FSUB(p[(int)(k)-1],sice[(int)(k)-1]);
    smc[(int)(k)-1]=FADD(sh2o[(int)(k)-1],sice[(int)(k)-1]);
    qtldrn1=FSUB(qtldrn1,ovrfc[(int)(k)-1]);
    }
    }
    }
    }
    qtldrn=FDIV(qtldrn,dt);
}
// END GENERATED NATIVE TILE DRAINAGE

__device__ void groundwater_shallowwatertable(const float*,const float*,const float*,const float*,const float*,const float*,float,float*,float&,float&,float&,float&);

__device__ float d_sstep5(const float *par, float dt, const float *dz, const float*zsoil,float zwt,float*smcwtd,float*qdrain,float*deeprech,
                         const float *sice, float *sh2o, float *ai, float *bi,
                         float *ci, float *rhstt, float *smc)
{
    float wplus = K_ZERO;                                            // :7894

    for (int k = 0; k < NSOIL; ++k) {                                // :7896
        rhstt[k] = FMUL(rhstt[k], dt);
        ai[k] = FMUL(ai[k], dt);
        bi[k] = FADD(K_ONE, FMUL(bi[k], dt));
        ci[k] = FMUL(ci[k], dt);
    }

    float rhsttin[NSOIL], ciin[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {                                // :7906
        rhsttin[k] = rhstt[k];
        ciin[k] = ci[k];
    }

    d_rosr12(ai, bi, ciin, rhsttin, ci, rhstt);                      // :7913

    for (int k = 0; k < NSOIL; ++k) sh2o[k] = FADD(sh2o[k], ci[k]);  // :7915

    if(zwt<FSUB(zsoil[3],dz[3]))*deeprech=FADD(*deeprech,FMUL(dt,*qdrain));
    else {
        *smcwtd=FADD(*smcwtd,FDIV(FMUL(dt,*qdrain),dz[3]));
        wplus=FMUL(f_max(FSUB(*smcwtd,par[P_SMCMAX+3]),0.0f),dz[3]);
        float wminus=FMUL(f_max(FSUB(1.0e-4f,*smcwtd),0.0f),dz[3]);
        *smcwtd=f_max(f_min(*smcwtd,par[P_SMCMAX+3]),1.0e-4f);
        sh2o[3]=FADD(sh2o[3],FDIV(wplus,dz[3]));
        *qdrain=FSUB(*qdrain,FDIV(wplus,dt));*deeprech=FSUB(*deeprech,wminus);
    }

    for (int k = NSOIL - 1; k >= 1; --k) {                           // :7951
        float epore = f_max(K_1EM4_UP, FSUB(par[P_SMCMAX + k], sice[k]));
        wplus = FMUL(f_max(FSUB(sh2o[k], epore), K_ZERO), dz[k]);
        sh2o[k] = f_min(epore, sh2o[k]);
        sh2o[k - 1] = FADD(sh2o[k - 1], FDIV(wplus, dz[k - 1]));
    }

    float epore = f_max(K_1EM4_L1, FSUB(par[P_SMCMAX], sice[0]));    // :7958
    wplus = FMUL(f_max(FSUB(sh2o[0], epore), K_ZERO), dz[0]);        // :7959
    sh2o[0] = f_min(epore, sh2o[0]);                                 // :7960

    if (wplus > K_ZERO) {                                            // :7962
        sh2o[1] = FADD(sh2o[1], FDIV(wplus, dz[1]));                 // :7963
        for (int k = 1; k < NSOIL - 1; ++k) {                        // :7964
            float e = f_max(K_1EM4_DOWN, FSUB(par[P_SMCMAX + k], sice[k]));
            wplus = FMUL(f_max(FSUB(sh2o[k], e), K_ZERO), dz[k]);
            sh2o[k] = f_min(e, sh2o[k]);
            sh2o[k + 1] = FADD(sh2o[k + 1], FDIV(wplus, dz[k + 1]));
        }
        float e = f_max(K_1EM4_DOWN,
                        FSUB(par[P_SMCMAX + NSOIL - 1], sice[NSOIL - 1]));
        wplus = FMUL(f_max(FSUB(sh2o[NSOIL - 1], e), K_ZERO), dz[NSOIL - 1]);
        sh2o[NSOIL - 1] = f_min(e, sh2o[NSOIL - 1]);                 // :7971
    }

    for (int k = 0; k < NSOIL; ++k) smc[k] = FADD(sh2o[k], sice[k]); // :7974
    return wplus;
}


template<int INF, int RUN>
__device__ void d_srt(const float *par, const float *zsoil, float pddum,
                      const float *etrani, float qseva, const float *smc,
                      const float *fcr, const float *sh2o, float sicemax, float fcrmax, float *rhstt, float *ai, float *bi,
                      float *ci, float *qdrain, float *wcnd, float zwt, float smcwtd)
{
    float wdf[NSOIL], smx[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {                                // :7775
        // WRF LSM:7772-7785: native nonlinear permeability reads liquid
        // water and the column maximum ice, not total moisture or FCR.
        if (INF == 1) {
            d_wdfcnd1(smc[k], fcr[k], par[P_SMCMAX+k], par[P_BEXP+k],
                       par[P_DWSAT+k], par[P_DKSAT+k], &wdf[k], &wcnd[k]);
            smx[k] = smc[k];
        } else {
            d_wdfcnd2(sh2o[k], sicemax, par[P_SMCMAX+k], par[P_BEXP+k],
                       par[P_DWSAT+k], par[P_DKSAT+k], &wdf[k], &wcnd[k]);
            smx[k] = sh2o[k];
        }
    }

    float denom[NSOIL], ddz[NSOIL], dsmdz[NSOIL], wflux[NSOIL];
    for (int k = 0; k < NSOIL; ++k) ddz[k] = dsmdz[k] = K_ZERO;
    *qdrain = K_ZERO;

    for (int k = 0; k < NSOIL; ++k) {                                // :7792
        if (k == 0) {
            denom[k] = f_neg(zsoil[k]);
            float temp1 = f_neg(zsoil[k + 1]);
            ddz[k] = FDIV(K_TWO, temp1);
            dsmdz[k] = FDIV(FMUL(K_TWO, FSUB(smx[k], smx[k + 1])), temp1);
            wflux[k] = FADD(FADD(FSUB(FADD(FMUL(wdf[k], dsmdz[k]), wcnd[k]),
                                      pddum), etrani[k]), qseva);
        } else if (k < NSOIL - 1) {
            denom[k] = FSUB(zsoil[k - 1], zsoil[k]);
            float temp1 = FSUB(zsoil[k - 1], zsoil[k + 1]);
            ddz[k] = FDIV(K_TWO, temp1);
            dsmdz[k] = FDIV(FMUL(K_TWO, FSUB(smx[k], smx[k + 1])), temp1);
            wflux[k] = FADD(FSUB(FSUB(FADD(FMUL(wdf[k], dsmdz[k]), wcnd[k]),
                                      FMUL(wdf[k - 1], dsmdz[k - 1])),
                                 wcnd[k - 1]), etrani[k]);
        } else {
            denom[k] = FSUB(zsoil[k - 1], zsoil[k]);
            *qdrain = (RUN == 1 || RUN == 2) ? K_ZERO : RUN == 4 ? FMUL(FSUB(K_ONE, fcrmax), wcnd[k])
                                : FMUL(par[P_SLOPE], wcnd[k]);                   // :7807
            if(RUN==5) {
                float temp1=FMUL(2.0f,denom[k]);
                float smxbot=zwt<FSUB(zsoil[3],denom[3])?
                  FSUB(smx[k],FDIV(FMUL(FMUL(FSUB(smx[k],smcwtd),denom[k]),2.0f),FSUB(FADD(denom[k],zsoil[k]),zwt))):smcwtd;
                dsmdz[k]=FDIV(FMUL(2.0f,FSUB(smx[k],smxbot)),temp1);
                *qdrain=FADD(FMUL(wdf[k],dsmdz[k]),wcnd[k]);
            }
            wflux[k] = FADD(FADD(FSUB(f_neg(FMUL(wdf[k - 1], dsmdz[k - 1])),
                                      wcnd[k - 1]), etrani[k]), *qdrain);
        }
    }

    for (int k = 0; k < NSOIL; ++k) {                                // :7828
        if (k == 0) {
            ai[k] = K_ZERO;
            bi[k] = FDIV(FMUL(wdf[k], ddz[k]), denom[k]);
            ci[k] = f_neg(bi[k]);
        } else if (k < NSOIL - 1) {
            ai[k] = f_neg(FDIV(FMUL(wdf[k - 1], ddz[k - 1]), denom[k]));
            ci[k] = f_neg(FDIV(FMUL(wdf[k], ddz[k]), denom[k]));
            bi[k] = f_neg(FADD(ai[k], ci[k]));
        } else {
            ai[k] = f_neg(FDIV(FMUL(wdf[k - 1], ddz[k - 1]), denom[k]));
            ci[k] = K_ZERO;
            bi[k] = f_neg(FADD(ai[k], ci[k]));
        }
        rhstt[k] = FDIV(wflux[k], f_neg(denom[k]));                  // :7843
    }
}

template<int INF, int RUN>
__device__ void d_soilwater(const float *par, int urban, float dt,
                            const float *zsoil, const float *dz, float qinsur,
                            float qseva, const float *etrani, const float *sice,
                            float *sh2o, float *smc, float *runsub,
                            float *runsrf_o, float *qdrain_o, float *wcnd_o,
                            float *fcrmax_o, float *zwt, int infdv, int tdrn, const float *tile, float tdfrac, float dx, float *qtldrn, float*smcwtd,float*deeprech)
{
    float runsrf = K_ZERO;                                           // :7318
    float pddum = K_ZERO;                                            // :7319
    float rsat = K_ZERO;                                             // :7320

    for (int k = 0; k < NSOIL; ++k) {                                // :7324
        float epore = f_max(K_1EM4_RSAT, FSUB(par[P_SMCMAX + k], sice[k]));
        rsat = FADD(rsat, FMUL(f_max(K_ZERO, FSUB(sh2o[k], epore)), dz[k]));
        sh2o[k] = f_min(epore, sh2o[k]);
    }

    // Scalar, one layer at a time.  gfortran vectorises this loop into
    // libmvec at -O2 and gets a different answer; see the header.
    float fcr[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {                                // :7333
        float fice = f_min(K_ONE, FDIV(sice[k], par[P_SMCMAX + k]));
        fcr[k] = FDIV(f_max(K_ZERO,
                            FSUB(glibc_expf(f_neg(FMUL(K_A, FSUB(K_ONE, fice)))),
                                 glibc_expf(f_neg(K_A)))),
                      FSUB(K_ONE, glibc_expf(f_neg(K_A))));
    }

    float sicemax = K_ZERO;                                          // :7341
    float fcrmax = K_ZERO;                                           // :7342
    for (int k = 0; k < NSOIL; ++k) {                                // :7344
        if (sice[k] > sicemax) sicemax = sice[k];
        if (fcr[k] > fcrmax) fcrmax = fcr[k];
    }

    if (RUN == 2) {
        equilibrium_water_table(par,zsoil,dz,sh2o,zwt);
        *runsub=FMUL(FMUL(FMUL(FSUB(K_ONE,fcrmax),4.0f),glibc_expf(-par[W_P_STRIDE+1])),glibc_expf(FMUL(-2.0f,*zwt)));
    }
    if (urban) fcr[0] = K_0P95;                                      // :7361

    if (RUN == 1 || RUN == 2 || RUN==5) {
        float fff=RUN==1?FDIV(par[P_BEXP],3.0f):2.0f;
        float fsat=RUN==5?FMUL(par[W_P_STRIDE],glibc_expf(FMUL(-3.0f,f_max(f_neg(*zwt),0.0f)))):FMUL(par[W_P_STRIDE],glibc_expf(FMUL(FMUL(-0.5f,fff),*zwt)));
        if(qinsur>K_ZERO) {
            runsrf=FMUL(qinsur,FADD(FMUL(FSUB(K_ONE,fcr[0]),fsat),fcr[0]));
            pddum=FSUB(qinsur,runsrf);
        }
    }
    if (RUN == 3) {
        d_infil(par, dt, zsoil, sh2o, sice, sicemax, qinsur, &pddum, &runsrf);
    } else if (RUN == 4) {
        // Native BATS saturation and complete drainage coupling, L:7399-7413.
        float smctot=K_ZERO, dztot=K_ZERO;
        for(int k=0;k<NSOIL;++k) {
            dztot=FADD(dztot,dz[k]);
            smctot=FADD(smctot,FMUL(FDIV(smc[k],par[P_SMCMAX+k]),dz[k]));
            if(dztot>=2.0f)break;
        }
        smctot=FDIV(smctot,dztot);
        float fsat=glibc_powf(f_max(0.01f,smctot),4.0f);
        if(qinsur>K_ZERO) {
            runsrf=FMUL(qinsur,FADD(FMUL(FSUB(K_ONE,fcr[0]),fsat),fcr[0]));
            pddum=FSUB(qinsur,runsrf);
        }
    }

    if (RUN == 6) vic(par,dt,smc,zsoil,qinsur,&runsrf,&pddum);
    if (RUN == 7) xaj(par,dt,fcr,smc,zsoil,qinsur,&runsrf,&pddum);

    float facc=1.0e-6f; // Native SOILWATER:7425, reset before fine solves.
    if(RUN==8)dynamic_vic(par,dt,smc,sh2o,sice,sicemax,zsoil,qinsur,facc,pddum,runsrf,infdv);
    int niter = 3;                                                   // :7440
    if (FMUL(pddum, dt) > FMUL(dz[0], par[P_SMCMAX])) niter *= 2;    // :7443
    float dtfine = FDIV(dt, (float) niter);                          // :7449

    float qdrain_save = K_ZERO, runsrf_save = K_ZERO;                // :7453
    float qdrain = K_ZERO;
    float rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];

    facc=1.0e-6f; // Native SOILWATER:7452. FACC is not persistent state.
    for (int it = 0; it < niter; ++it) {                             // :7456
        if (qinsur > K_ZERO && RUN == 3) {                                       // :7457
            d_infil(par, dtfine, zsoil, sh2o, sice, sicemax, qinsur,
                    &pddum, &runsrf);
        }
        if(qinsur>K_ZERO && RUN == 6)
            vic(par,dtfine,smc,zsoil,qinsur,&runsrf,&pddum);
        if(qinsur>K_ZERO && RUN == 7)
            xaj(par,dtfine,fcr,smc,zsoil,qinsur,&runsrf,&pddum);
        if(qinsur>K_ZERO && RUN==8)
            dynamic_vic(par,dtfine,smc,sh2o,sice,sicemax,zsoil,qinsur,facc,pddum,runsrf,infdv);
        // SRT reads SMC, not SH2O, under OPT_INF == 1.
        d_srt<INF,RUN>(par, zsoil, pddum, etrani, qseva, smc, fcr, sh2o, sicemax, fcrmax,
              rhstt, ai, bi, ci, &qdrain, wcnd_o, *zwt,*smcwtd);
        float wplus = RUN==5?d_sstep5(par,dtfine,dz,zsoil,*zwt,smcwtd,&qdrain,deeprech,sice,sh2o,ai,bi,ci,rhstt,smc):d_sstep(par, dtfine, dz, sice, sh2o, ai, bi, ci,
                              rhstt, smc);
        rsat = FADD(rsat, wplus);                                    // :7489
        qdrain_save = FADD(qdrain_save, qdrain);                     // :7490
        runsrf_save = FADD(runsrf_save, runsrf);                     // :7491
    }

    qdrain = FDIV(qdrain_save, (float) niter);                       // :7494
    runsrf = FDIV(runsrf_save, (float) niter);                       // :7495
    runsrf = FADD(FMUL(runsrf, K_1000_RSR), FDIV(FMUL(rsat, K_1000_RST), dt));
    qdrain = FMUL(qdrain, K_1000_QDR);                               // :7498

    if(RUN==3 && tdrn==1 && tdfrac>0.3f) tile_drain(tile,sh2o,smc,sice,zsoil,*qtldrn,dt);
    else if(RUN==3 && tdrn==2 && tdfrac>0.1f) tile_hooghoudt(tile,par+P_SMCWLT,wcnd_o,sh2o,smc,sice,zsoil,dz,dt,dx,*qtldrn,*zwt);
    if (RUN == 2) {
        float total=K_ZERO;
        for(int k=0;k<NSOIL;++k)total=FADD(total,FMUL(wcnd_o[k],dz[k]));
        for(int k=0;k<NSOIL;++k) {
            float removed=FDIV(FMUL(FMUL(*runsub,dt),FMUL(wcnd_o[k],dz[k])),total);
            sh2o[k]=FSUB(sh2o[k],FDIV(removed,FMUL(dz[k],1000.0f)));
        }
    }
    if (RUN != 1) {
    float mliq[NSOIL];
    for (int k = 0; k < NSOIL; ++k)                                  // :7548
        mliq[k] = FMUL(FMUL(sh2o[k], dz[k]), K_1000_MLQ);

    for (int iz = 0; iz < NSOIL - 1; ++iz) {                         // :7552
        float xs = (mliq[iz] < K_ZERO) ? FSUB(K_WATMIN, mliq[iz]) : K_ZERO;
        mliq[iz] = FADD(mliq[iz], xs);
        mliq[iz + 1] = FSUB(mliq[iz + 1], xs);
    }
    {
        int iz = NSOIL - 1;                                          // :7562
        float xs = (mliq[iz] < K_WATMIN) ? FSUB(K_WATMIN, mliq[iz]) : K_ZERO;
        mliq[iz] = FADD(mliq[iz], xs);
        if(RUN==5)*deeprech=FSUB(*deeprech,FMUL(xs,0.001f));else *runsub = FSUB(*runsub, FDIV(xs, dt));                       // :7569
    }
    for (int iz = 0; iz < NSOIL; ++iz)                               // :7572
        sh2o[iz] = FDIV(mliq[iz], FMUL(dz[iz], K_1000_BAC));

    } // Native SOILWATER: OPT_RUN /= 1 minimum-water correction.

    // SMC is deliberately NOT recomputed: the WATMIN fixup rewrites SH2O and
    // never touches SMC, so on exit the two are inconsistent by exactly that
    // correction.  The oracle pins the difference.
    *runsrf_o = runsrf;
    *qdrain_o = qdrain;
    *fcrmax_o = fcrmax;
}

template<int INF, int RUN>
__device__ void d_water(const float *par, int urban, int nroot, int ist,
                        Col *c, float *smc,
                        const int *imelt, const float *ficeold,
                        const float *zsoil, const float *btrani,
                        float dt, float fcev, float fctr, float elai,
                        float esai, float fveg, float bdfall,
                        int frozen_canopy, int frozen_ground, float sfctmp,
                        float qvap, float qdew, float qsnow, float qrain,
                        float snowhin, float ponding,
                        float canliq, float canice, float tv, float wslake,
                        float acc_qinsur, float acc_qseva, float *acc_etrani,
                        float *o, float *zwt, float *wa, float *wt, int infdv, int soil_steps, int calculate_soil, int tdrn, const float *tile, float tdfrac, float dx, const float*irrpar, const float*irrin, float*irrout,float*smcwtd,float*deeprech,float*rech,const float*smceq)
{
    float etrani[NSOIL];
    for (int k = 0; k < NSOIL; ++k) etrani[k] = K_ZERO;               // :6107
    float snoflow = K_ZERO;                                          // :6108
    float runsub  = K_ZERO;                                          // :6109
    float runsrf  = K_ZERO;                                          // :6110
    float qinsur  = K_ZERO;                                          // :6111
    float qtldrn  = K_ZERO;                                          // :6112
    // Genuine bottom drainage, separate from exfiltration and snow runoff.
    // Diagnostic only; existing WATER calculations and outputs are unchanged.
    float soldrain = K_ZERO;

    // canopy-intercepted snowfall/rainfall, drips, and throughfall    :6116
    float cw[13];
    d_canwater(par[P_CH2OP], dt, fcev, fctr, elai, esai, fveg, bdfall,
               frozen_canopy, canliq, canice, tv, cw);
    canliq = cw[0]; canice = cw[1]; tv = cw[2];
    float cmc = cw[3], ecan = cw[4], etran = cw[5], fwet = cw[6];
    float qsubc = cw[7], qfroc = cw[8], qfrzc = cw[9];
    float qmeltc = cw[10], qevac = cw[11], qdewc = cw[12];

    // sublimation, frost, evaporation, and dew
    float qsnsub = K_ZERO;                                           // :6126
    if (c->sneqv > K_ZERO)                                           // :6127
        qsnsub = f_min(qvap, FDIV(c->sneqv, dt));                    // :6128
    float qseva = FSUB(qvap, qsnsub);                                // :6130

    float qsnfro = K_ZERO;                                           // :6132
    if (c->sneqv > K_ZERO)                                           // :6133
        qsnfro = qdew;                                               // :6134
    float qsdew = FSUB(qdew, qsnfro);                                // :6136

    float qsnbot, ponding1, ponding2;
    snowwater(c, dt, zsoil, imelt, ficeold, sfctmp, snowhin, qsnow,
              qsnfro, qsnsub, qrain, par[P_SSI], par[P_SRF],
              &qsnbot, &snoflow, &ponding1, &ponding2);              // :6138

    // SNOWWATER restores DZSNSO(1:NSOIL) to a positive thickness before it
    // returns, which is what 6146 indexes.
    float *dz = c->dzsnso + NSNOW;

    if (frozen_ground) {                                             // :6145
        c->sice[0] = FADD(c->sice[0],
                          FDIV(FMUL(FSUB(qsdew, qseva), dt),
                               FMUL(dz[0], W_K_1000)));              // :6146
        qsdew = K_ZERO;                                              // :6147
        qseva = K_ZERO;                                              // :6148
        if (c->sice[0] < K_ZERO) {                                   // :6149
            c->sh2o[0] = FADD(c->sh2o[0], c->sice[0]);               // :6150
            c->sice[0] = K_ZERO;                                     // :6151
        }
        smc[0] = FADD(c->sh2o[0], c->sice[0]);                       // :6153
    }

    // convert units (mm/s -> m/s)
    qinsur = FMUL(FDIV(FADD(FADD(ponding, ponding1), ponding2), dt),
                  W_K_MILLI);                                        // :6159

    if (c->isnow == 0)                                               // :6161
        qinsur = FADD(qinsur,
                      FMUL(FADD(FADD(qsnbot, qsdew), qrain),
                           W_K_MILLI));                              // :6162
    else                                                             // :6163
        qinsur = FADD(qinsur,
                      FMUL(FADD(qsnbot, qsdew), W_K_MILLI));         // :6164

    qseva = FMUL(qseva, W_K_MILLI);                                  // :6167

    for (int iz = 0; iz < nroot; ++iz)                               // :6169
        etrani[iz] = FMUL(FMUL(etran, btrani[iz]), W_K_MILLI);       // :6170

    // added soil timestep capability
    acc_qinsur = FADD(acc_qinsur, qinsur);                           // :6178
    acc_qseva  = FADD(acc_qseva, qseva);                             // :6179
    float acc_e[NSOIL];
    for (int k = 0; k < NSOIL; ++k)
        acc_e[k] = FADD(acc_etrani[k], etrani[k]);                   // :6180

    if(irrpar) {irrout[0]=irrin[3];irrout[1]=irrin[4];irrout[2]=0.0f;irrout[3]=0.0f;}
    if(calculate_soil) {
    float dt_soil = FMUL(dt,float(soil_steps));
    // Native WATER:6187-6202, after accumulation and before soil averaging.
    if(irrpar && irrin[0]!=0.0f) {
      if(irrout[1]>0.0f) {
        flood_irrigation(irrpar,dt_soil,c->sh2o,smc,c->sice,irrin[2],irrout[1],irrout[3]);
        acc_qinsur=FADD(acc_qinsur,FMUL(FDIV(irrout[3],dt_soil),float(soil_steps)));
      }
      if(irrout[0]>0.0f) {
        micro_irrigation(irrpar,dt_soil,c->sh2o,smc,c->sice,irrin[1],irrout[0],irrout[2]);
        c->sh2o[0]=FADD(c->sh2o[0],FDIV(irrout[2],FMUL(f_neg(1.0f),zsoil[0])));
      }
    }
    float qseva_avg = FDIV(acc_qseva,float(soil_steps));
    float qinsur_avg = FDIV(acc_qinsur,float(soil_steps));
    float etrani_avg[NSOIL];
    for(int k=0;k<NSOIL;++k)etrani_avg[k]=FDIV(acc_e[k],float(soil_steps));

    if (ist == 2) {                                                  // :6209
        runsrf = K_ZERO;                                             // :6210
        if (wslake >= W_K_WSLMAX)                                    // :6211
            runsrf = FMUL(FMUL(qinsur_avg, W_K_1000), dt_soil);
        wslake = FSUB(FADD(wslake,
                           FMUL(FMUL(FSUB(qinsur_avg, qseva_avg), W_K_1000),
                                dt_soil)),
                      runsrf);                                       // :6212
        // QDRAIN, WCND and FCRMAX stay undefined on this branch and nothing
        // reads them: 6235 and 6252-6254 are inside the ELSE.
    } else {                                                         // :6213
        float qdrain, wcnd[NSOIL], fcrmax;
        d_soilwater<INF,RUN>(par, urban, dt_soil, zsoil, dz, qinsur_avg, qseva_avg,
                    etrani_avg, c->sice, c->sh2o, smc, &runsub,
                    &runsrf, &qdrain, wcnd, &fcrmax, zwt, infdv, tdrn, tile, tdfrac, dx, &qtldrn,smcwtd,deeprech);                // :6214
        soldrain = FMUL(qdrain, dt_soil);  // qdrain mm/s -> mm per LSM step

        if (RUN == 1) {
            float qin,qdis;
            groundwater(par,zsoil,dt_soil,c->sice,wcnd,fcrmax,c->sh2o,zwt,wa,wt,&qin,&qdis);
            runsub=qdis;
        }
        if(RUN==3 || RUN==4 || RUN==6 || RUN==7 || RUN==8)
            runsub = FADD(runsub, qdrain);                               // :6235

        for (int iz = 0; iz < NSOIL; ++iz)                           // :6238
            smc[iz] = FADD(c->sh2o[iz], c->sice[iz]);                // :6239

        if(RUN==5) {
            groundwater_shallowwatertable(par+P_SMCMAX,par+W_P_STRIDE+1,par+P_BEXP,zsoil,dz,smceq,dt_soil,smc,*zwt,*smcwtd,*rech,qdrain);
            c->sh2o[3]=FSUB(smc[3],c->sice[3]);runsub=FADD(runsub,qdrain);*wa=0.0f;
        }

        runsrf = FMUL(runsrf, dt_soil);                              // :6252
        runsub = FMUL(runsub, dt_soil);                              // :6253
        qtldrn = FMUL(qtldrn, dt_soil);                              // :6254
    }

    } // native calculate_soil, module_sf_noahmplsm.F:6256
    runsub = FADD(runsub, FMUL(snoflow, dt));                        // :6259

    for (int k = 0; k < NSOIL; ++k) {
        o[O_SMC + k] = smc[k];
        o[O_ACC_ETRANI + k] = acc_e[k];
    }
    o[O_CANLIQ] = canliq; o[O_CANICE] = canice; o[O_TV] = tv;
    o[O_WSLAKE] = wslake;
    o[O_ACC_QINSUR] = acc_qinsur; o[O_ACC_QSEVA] = acc_qseva;
    o[O_CMC] = cmc; o[O_ECAN] = ecan; o[O_ETRAN] = etran; o[O_FWET] = fwet;
    o[O_RUNSRF] = runsrf; o[O_RUNSUB] = runsub; o[O_QTLDRN] = qtldrn;
    o[O_PONDING1] = ponding1; o[O_PONDING2] = ponding2;
    o[O_QSNBOT] = qsnbot; o[O_QSNSUB] = qsnsub; o[O_QSNFRO] = qsnfro;
    o[O_QSUBC] = qsubc; o[O_QFROC] = qfroc; o[O_QFRZC] = qfrzc;
    o[O_QMELTC] = qmeltc; o[O_QEVAC] = qevac; o[O_QDEWC] = qdewc;
    o[O_QSEVA] = qseva; o[O_QSDEW] = qsdew; o[O_QINSUR] = qinsur;
    o[O_SNOFLOW] = snoflow;
    o[O_SOILDRAIN] = soldrain;
}

template<int INF, int RUN>
__device__ void column(const float *par, const float *fin,
                                   const int *iin, float *fout, int *iout,
                                   int n, int infdv, int soil_steps, int calculate_soil, int tdrn, int irrigation)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int base_tail=RUN==5?5:RUN==8?3:RUN>=6?8:RUN<=2?6:0;
    int in_gw=RUN==5?9:RUN<=2?3:0,out_gw=RUN==5?5:RUN<=2?3:0;
    const float *p = par + (size_t) tid * (W_P_STRIDE+base_tail+(tdrn?15:0)+(irrigation?35:0));
    const float *x = fin + (size_t) tid * (W_IN_STRIDE+in_gw+(tdrn?3:0)+(irrigation?5:0));
    const int   *ip = iin + (size_t) tid * W_INT_STRIDE;
    float *q = fout + (size_t) tid * (W_OUT_STRIDE+out_gw+(tdrn?1:0)+(irrigation?4:0));

    Col col;
    col_load(&col, x, ip[0]);

    float smc[NSOIL], acc_etrani[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = x[W_SMC + k];
        acc_etrani[k] = x[W_ACC_ETRANI + k];
    }

    float zwt=K_ZERO,wa=K_ZERO,wt=K_ZERO,smcwtd=K_ZERO,deeprech=K_ZERO,rech=K_ZERO;
    if(RUN==5){zwt=x[W_IN_STRIDE];wa=x[W_IN_STRIDE+1];smcwtd=x[W_IN_STRIDE+2];deeprech=x[W_IN_STRIDE+3];rech=x[W_IN_STRIDE+4];}
    if(RUN<=2) {zwt=x[W_IN_STRIDE];wa=x[W_IN_STRIDE+1];wt=x[W_IN_STRIDE+2];}
    const float *tx=x+W_IN_STRIDE+in_gw;
    if(tdrn && RUN>2)zwt=tx[2];
    d_water<INF,RUN>(p, ip[1], ip[2], ip[3], &col, smc, ip + 6, x + W_FICEOLD,
            x + W_ZSOIL, x + W_BTRANI,
            x[W_DT], x[W_FCEV], x[W_FCTR], x[W_ELAI], x[W_ESAI], x[W_FVEG],
            x[W_BDFALL], ip[4], ip[5], x[W_SFCTMP], x[W_QVAP], x[W_QDEW],
            x[W_QSNOW], x[W_QRAIN], x[W_SNOWHIN], x[W_PONDING],
            x[W_CANLIQ], x[W_CANICE], x[W_TV], x[W_WSLAKE],
            x[W_ACC_QINSUR], x[W_ACC_QSEVA], acc_etrani, q, &zwt, &wa, &wt, infdv, soil_steps, calculate_soil, tdrn, tdrn?p+W_P_STRIDE+base_tail:nullptr, tdrn?tx[0]:0.0f, tdrn?tx[1]:0.0f,
            irrigation?p+W_P_STRIDE+base_tail+(tdrn?15:0):nullptr,
            irrigation?x+W_IN_STRIDE+in_gw+(tdrn?3:0):nullptr,
            irrigation?q+W_OUT_STRIDE+out_gw+(tdrn?1:0):nullptr,&smcwtd,&deeprech,&rech,RUN==5?x+W_IN_STRIDE+5:nullptr);
    if(RUN==5){q[W_OUT_STRIDE]=zwt;q[W_OUT_STRIDE+1]=wa;q[W_OUT_STRIDE+2]=smcwtd;q[W_OUT_STRIDE+3]=deeprech;q[W_OUT_STRIDE+4]=rech;}
    if(RUN<=2) {q[W_OUT_STRIDE]=zwt;q[W_OUT_STRIDE+1]=wa;q[W_OUT_STRIDE+2]=wt;}

    if(tdrn)q[W_OUT_STRIDE+out_gw]=zwt;
    col_store(&col, q, iout + tid);
}
}

extern "C" __global__ void k_water_selected(const float *par,const float *fin,
    const int *iin,float *fout,int *iout,int n,int inf,int run,int infdv,int soil_steps,int calculate_soil,int tdrn,int irrigation) {
    if(inf==1) {
        if(run==1)noahmp_water_options::column<1,1>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==2)noahmp_water_options::column<1,2>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==3)noahmp_water_options::column<1,3>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==5)noahmp_water_options::column<1,5>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==6)noahmp_water_options::column<1,6>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==7)noahmp_water_options::column<1,7>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==8)noahmp_water_options::column<1,8>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else noahmp_water_options::column<1,4>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
    } else {
        if(run==1)noahmp_water_options::column<2,1>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==2)noahmp_water_options::column<2,2>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==3)noahmp_water_options::column<2,3>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==5)noahmp_water_options::column<2,5>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==6)noahmp_water_options::column<2,6>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==7)noahmp_water_options::column<2,7>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else if(run==8)noahmp_water_options::column<2,8>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
        else noahmp_water_options::column<2,4>(par,fin,iin,fout,iout,n,infdv,soil_steps,calculate_soil,tdrn,irrigation);
    }
}

// WRF-v4 NOAHMP_SFLX:878-936; complete irrigation dispatch before PRECIP_HEAT.
extern "C" __global__ void k_irrigation_prefix(const float*par,const float*fin,
 const int*iin,float*fout,int*iout,int n,int opt_irr,int opt_irrm,int calculate_soil) {
 using namespace noahmp_water_options;
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 const float*p=par+(size_t)c*35;const float*x=fin+(size_t)c*28;const int*i=iin+(size_t)c*5;
 float*y=fout+(size_t)c*11;int*iy=iout+(size_t)c*3;
 bool croplu=i[0]!=0;int a=i[2],b=i[3],d=i[4];
 float rain=x[17],irrfra=x[21],si=x[22],mi=x[23],fi=x[24];
 float asi=x[25],ami=x[26],afi=x[27],irsirate=0.0f,irevplos=0.0f,firr=0.0f,eirr=0.0f;
 if(opt_irrm==0 && si==0.0f && mi==0.0f && fi==0.0f && irrfra>=p[24])si=1.0f;
 if(opt_irrm==1){si=1.0f;mi=0.0f;fi=0.0f;}
 else if(opt_irrm==2){si=0.0f;mi=1.0f;fi=0.0f;}
 else if(opt_irrm==3){si=0.0f;mi=0.0f;fi=1.0f;}
 if(calculate_soil) {
  if(croplu && irrfra>=p[24] && rain<FDIV(p[32],3600.0f) && FADD(FADD(asi,ami),afi)==0.0f)
   trigger_irrigation(p,i[1],opt_irr,opt_irrm,x,x+4,x[18],x[20],irrfra,x[19],si,mi,fi,a,b,d,asi,ami,afi);
  if(rain>=FDIV(p[32],3600.0f) || irrfra<p[24]){asi=0.0f;ami=0.0f;afi=0.0f;}
 }
 if(croplu && asi>0.0f) {
  sprinkler_irrigation(p,x[12],x+4,x+8,x[13],x[14],x[15],x[16],si,asi,irevplos,irsirate);
  rain=FADD(rain,FDIV(FMUL(irsirate,1000.0f),x[12]));
  firr=FDIV(FMUL(FMUL(irevplos,1000.0f),2510400.0f),x[12]);
  eirr=FDIV(FMUL(irevplos,1000.0f),x[12]);
 }
 y[0]=rain;y[1]=firr;y[2]=eirr;y[3]=irsirate;y[4]=irevplos;
 y[5]=asi;y[6]=ami;y[7]=afi;y[8]=si;y[9]=mi;y[10]=fi;
 iy[0]=a;iy[1]=b;iy[2]=d;
}
