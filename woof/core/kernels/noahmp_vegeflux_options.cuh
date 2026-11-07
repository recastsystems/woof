// WRF v4 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422.
// Selected-option extension compiled after the unchanged default unit.
// Public-domain source notice: licenses/LICENSE-WRF-public-domain.txt.

// STC3 keeps the fixed-width device mask used by the current runtime.
extern "C" __global__
void k_vegeflux_masked_stc3(const float *in, const float *pin, const int *isnow,
                            const float *dzsnso, const float *stc, const float *df,
                            const float *fsno, float *out, int n,
                            const unsigned char *active)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!active[i]) {
        for (int j = 0; j < VO_NOUT; ++j) out[VO_NOUT * i + j] = 0.0f;
        return;
    }
    dev_vege_flux(in + VI_NFLOAT * i, pin + 14 * i, isnow[i],
                  dzsnso + VF_NLAYER * i, stc + VF_NLAYER * i,
                  df + VF_NLAYER * i, out + VO_NOUT * i, 3, fsno[i]);
}

#define NMP_SFC2_EXP glibc_expf
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

// CANRES and CALHUM, pinned module_sf_noahmplsm.F:5141-5255.
// CRS2 extra parameters are actual TRANSFER_MP_PARAMETERS values.
__device__ void noahmp_canres_selected(const float* p,float par,float tv,
 float btran,float eah,float sfcprs,float fveg,float* rc,float* psn){
 float par_scale=FD(par,fmxf(fveg,1.0e-6f));
 float q2=FD(FM(0.622f,eah),FS(sfcprs,FM(0.378f,eah)));
 q2=FD(q2,FS(1.0f,q2));
 float es=FM(0.611f,glibc_expf(FM(FD(2.501e6f,461.0f),FS(FD(1.0f,273.15f),FD(1.0f,tv)))));
 float sp=FM(sfcprs,1.0e-3f);
 float qsat=FD(FM(0.622f,es),FS(sp,es));
 qsat=FM(qsat,1.0e3f);qsat=FD(qsat,1.0e3f);
 float ff=FD(FM(2.0f,par_scale),p[14]);
 float rcs=FD(FA(ff,FD(p[15],p[16])),FA(1.0f,ff));rcs=fmxf(rcs,0.0001f);
 float delta=fabsf(FS(p[17],tv));
 float delta2=delta==0.0f?0.0f:glibc_powf(delta,2.0f);
 float rct=FS(1.0f,FM(0.0016f,delta2));rct=fmxf(rct,0.0001f);
 float rcq=FD(1.0f,FA(1.0f,FM(p[18],fmxf(0.0f,FS(qsat,q2)))));rcq=fmxf(rcq,0.01f);
 *rc=FD(p[15],FM(FM(FM(rcs,rct),rcq),btran));*psn=-999.99f;
}

__device__ void dev_vege_flux_selected(const float *in, const float *p, int isnow,
                              const float *dzsnso, const float *stc,
                              const float *df, float *out,
                              int opt_stc=1, float fsno=0.0f, int opt_crs=1, int opt_sfc=1, float thair=0.0f, float czil=0.0f)
{
    const float dt = in[VI_DT], sav = in[VI_SAV], sag = in[VI_SAG];
    const float lwdn = in[VI_LWDN], ur = in[VI_UR], uu = in[VI_UU], vv = in[VI_VV];
    const float sfctmp = in[VI_SFCTMP], qair = in[VI_QAIR], eair = in[VI_EAIR];
    const float rhoair = in[VI_RHOAIR], snowh = in[VI_SNOWH], vai = in[VI_VAI];
    const float gammav = in[VI_GAMMAV], gammag = in[VI_GAMMAG], fwet = in[VI_FWET];
    const float laisun = in[VI_LAISUN], laisha = in[VI_LAISHA], cwp = in[VI_CWP];
    const float zlvl = in[VI_ZLVL], zpd = in[VI_ZPD], z0m = in[VI_Z0M];
    const float fveg = in[VI_FVEG], z0mg = in[VI_Z0MG], emv = in[VI_EMV];
    const float emg = in[VI_EMG], canliq = in[VI_CANLIQ], canice = in[VI_CANICE];
    const float rsurf = in[VI_RSURF], latheav = in[VI_LATHEAV];
    const float parsun = in[VI_PARSUN], parsha = in[VI_PARSHA], igs = in[VI_IGS];
    const float foln = in[VI_FOLN], co2air = in[VI_CO2AIR], o2air = in[VI_O2AIR];
    const float btran = in[VI_BTRAN], sfcprs = in[VI_SFCPRS], rhsur = in[VI_RHSUR];
    const float pahv = in[VI_PAHV], pahg = in[VI_PAHG];
    const float psfc = in[VI_PSFC];

    float eah = in[VI_EAH], tah = in[VI_TAH], tv = in[VI_TV], tg = in[VI_TG];
    float cm = in[VI_CM], ch = in[VI_CH], qsfc = in[VI_QSFC];

    const float mpe = 1.0e-6f;
    int liter = 0;
    float fv = 0.1f;
    float dtv = 0.0f, dtg = 0.0f;
    float wstar=0.0f;
    float moz = 0.0f, fm = 0.0f, fh = 0.0f, fm2 = 0.0f, fh2 = 0.0f;
    int mozsgn = 0;
    float hg = 0.0f, h = 0.0f;
    float mozg = 0.0f, fhg = 0.0f;
    float rssun = 0.0f, rssha = 0.0f, psnsun = 0.0f, psnsha = 0.0f;

    float vaie = fmnf(6.0f, vai);
    float laisune = fmnf(6.0f, laisun);
    float laishae = fmnf(6.0f, laisha);

    float esatw, esati, dsatw, dsati;
    float t = tdc(tg);
    dev_esat(t, &esatw, &esati, &dsatw, &dsati);
    float estg = (t > 0.0f) ? esatw : esati;

    qsfc = FD(FM(0.622f, eair), FS(psfc, FM(0.378f, eair)));

    float hcan = p[PP_HVT];
    float uc = FD(FM(ur, glibc_logf(FD(FA(FS(hcan, zpd), z0m), z0m))),
                  glibc_logf(FD(zlvl, z0m)));

    float air = FS(-FM(FM(emv, FA(1.0f, FM(FS(1.0f, emv), FS(1.0f, emg)))), lwdn),
                   FM(FM(FM(emv, emg), PH_SB), p4f(tg)));
    float cir = FM(FM(FS(2.0f, FM(emv, FS(1.0f, emg))), emv), PH_SB);

    float cah = 0.0f, cvh = 0.0f, rahg = 1.0f, rawg = 1.0f, z0h = z0m;
    float shc = 0.0f, evc = 0.0f, tr = 0.0f, irc = 0.0f, canhs = 0.0f;

    for (int it = 1; it <= 20; ++it) {
        z0h = z0m;
        float z0hg = z0mg;
        float ch2;
        if(opt_sfc==2){
            nmp_sfcdif2(it,z0m,tah,thair,ur,zlvl,czil,PH_GRAV,&cm,&ch,&moz,&wstar,&fv);
            ch=FD(ch,ur);cm=FD(cm,ur);
        }else{
        dev_sfcdif1(it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur, mpe,
                    &moz, &mozsgn, &fm, &fh, &fm2, &fh2, &fv, &cm, &ch, &ch2);
        }

        float rahc = fmxf(1.0f, FD(1.0f, FM(ch, ur)));
        float rawc = rahc;

        float ramg, rb;
        dev_ragrb(it, vaie, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc,
                  z0h, fv, cwp, mpe, p[PP_DLEAF], &mozg, &fhg,
                  &ramg, &rahg, &rawg, &rb);

        t = tdc(tv);
        dev_esat(t, &esatw, &esati, &dsatw, &dsati);
        float estv = (t > 0.0f) ? esatw : esati;
        float destv = (t > 0.0f) ? dsatw : dsati;

        if (it == 1) {
            if(opt_crs==2){
                noahmp_canres_selected(p,parsun,tv,btran,eah,sfcprs,fveg,&rssun,&psnsun);
                noahmp_canres_selected(p,parsha,tv,btran,eah,sfcprs,fveg,&rssha,&psnsha);
            }else{
            dev_stomata(p, mpe, parsun, foln, tv, estv, eah, sfctmp, sfcprs,
                        fveg, o2air, co2air, igs, btran, rb, &rssun, &psnsun);
            dev_stomata(p, mpe, parsha, foln, tv, estv, eah, sfctmp, sfcprs,
                        fveg, o2air, co2air, igs, btran, rb, &rssha, &psnsha);
            }
        }

        cah = FD(1.0f, rahc);
        cvh = FD(FM(2.0f, vaie), rb);
        float cgh = FD(1.0f, rahg);
        float cond = FA(FA(cah, cvh), cgh);
        float ata = FD(FA(FM(sfctmp, cah), FM(tg, cgh)), cond);
        float bta = FD(cvh, cond);
        float csh = FM(FM(FM(FS(1.0f, bta), rhoair), PH_CPAIR), cvh);

        float caw = FD(1.0f, rawc);
        float cew = FD(FM(fwet, vaie), rb);
        float ctw = FM(FS(1.0f, fwet),
                       FA(FD(laisune, FA(rb, rssun)), FD(laishae, FA(rb, rssha))));
        float cgw = FD(1.0f, FA(rawg, rsurf));
        cond = FA(FA(FA(caw, cew), ctw), cgw);
        float aea = FD(FA(FM(eair, caw), FM(estg, cgw)), cond);
        float bea = FD(FA(cew, ctw), cond);
        float cev = FD(FM(FM(FM(FS(1.0f, bea), cew), rhoair), PH_CPAIR), gammav);
        float ctr = FD(FM(FM(FM(FS(1.0f, bea), ctw), rhoair), PH_CPAIR), gammav);

        tah = FA(ata, FM(bta, tv));
        eah = FA(aea, FM(bea, estv));

        irc = FM(fveg, FA(air, FM(cir, p4f(tv))));
        shc = FM(FM(FM(FM(fveg, rhoair), PH_CPAIR), cvh), FS(tv, tah));
        evc = FD(FM(FM(FM(FM(fveg, rhoair), PH_CPAIR), cew), FS(estv, eah)), gammav);
        tr = FD(FM(FM(FM(FM(fveg, rhoair), PH_CPAIR), ctw), FS(estv, eah)), gammav);
        if (tv > PH_TFRZ) evc = fmnf(FD(FM(canliq, latheav), dt), evc);
        else              evc = fmnf(FD(FM(canice, latheav), dt), evc);

        float hcv = FA(FA(FM(FM(p[PP_CBIOM], vaie), PH_CWAT),
                          FD(FM(canliq, PH_CWAT), PH_DENH2O)),
                       FD(FM(canice, PH_CICE), PH_DENICE));

        float b = FA(FS(FS(FS(FS(sav, irc), shc), evc), tr), pahv);
        float a = FM(fveg, FA(FA(FA(FM(FM(4.0f, cir), p3f(tv)), csh),
                                 FM(FA(cev, ctr), destv)), FD(hcv, dt)));
        dtv = FD(b, a);

        irc = FA(irc, FM(FM(FM(FM(fveg, 4.0f), cir), p3f(tv)), dtv));
        shc = FA(shc, FM(FM(fveg, csh), dtv));
        evc = FA(evc, FM(FM(FM(fveg, cev), destv), dtv));
        tr = FA(tr, FM(FM(FM(fveg, ctr), destv), dtv));
        canhs = FD(FM(FM(dtv, fveg), hcv), dt);

        tv = FA(tv, dtv);

        h = FD(FM(FM(rhoair, PH_CPAIR), FS(tah, sfctmp)), rahc);
        hg = FD(FM(FM(rhoair, PH_CPAIR), FS(tg, tah)), rahg);

        qsfc = FD(FM(0.622f, eah), FS(sfcprs, FM(0.378f, eah)));

        if (liter == 1) break;
        if (it >= 5 && fabsf(dtv) <= 0.01f && liter == 0) liter = 1;
    }

    air = FS(-FM(FM(emg, FS(1.0f, emv)), lwdn), FM(FM(FM(emg, emv), PH_SB), p4f(tv)));
    cir = FM(emg, PH_SB);
    float csh = FD(FM(rhoair, PH_CPAIR), rahg);
    float cev = FD(FM(rhoair, PH_CPAIR), FM(gammag, FA(rawg, rsurf)));
    float cgh = FD(FM(2.0f, df[isnow + 1 + VF_LOFF]), dzsnso[isnow + 1 + VF_LOFF]);

    float irg = 0.0f, shg = 0.0f, evg = 0.0f, gh = 0.0f, destg = 0.0f;
    estg = 0.0f;
    for (int it = 0; it < 5; ++it) {
        t = tdc(tg);
        dev_esat(t, &esatw, &esati, &dsatw, &dsati);
        estg = (t > 0.0f) ? esatw : esati;
        destg = (t > 0.0f) ? dsatw : dsati;

        irg = FA(FM(cir, p4f(tg)), air);
        shg = FM(csh, FS(tg, tah));
        evg = FM(cev, FS(FM(estg, rhsur), eah));
        gh = FM(cgh, FS(tg, stc[isnow + 1 + VF_LOFF]));

        float b = FA(FS(FS(FS(FS(sag, irg), shg), evg), gh), pahg);
        float a = FA(FA(FA(FM(FM(4.0f, cir), p3f(tg)), csh), FM(cev, destg)), cgh);
        dtg = FD(b, a);

        irg = FA(irg, FM(FM(FM(4.0f, cir), p3f(tg)), dtg));
        shg = FA(shg, FM(csh, dtg));
        evg = FA(evg, FM(FM(cev, destg), dtg));
        gh = FA(gh, FM(cgh, dtg));
        tg = FA(tg, dtg);
    }

    // WRF-lineage Noah-MP 848f54ad :4125-4136, STC=1 reset / STC=3 blend.
    // Original notice: licenses/LICENSE-WRF-public-domain.txt. No Noah-MP 5.
    if (opt_stc != 2 && snowh > 0.05f && tg > PH_TFRZ) {
        tg = opt_stc == 3
            ? FA(FM(FS(1.0f, fsno), tg), FM(fsno, PH_TFRZ)) : PH_TFRZ;
        irg = FS(FS(FM(cir, p4f(tg)), FM(FM(emg, FS(1.0f, emv)), lwdn)),
                 FM(FM(FM(emg, emv), PH_SB), p4f(tv)));
        shg = FM(csh, FS(tg, tah));
        evg = FM(cev, FS(FM(estg, rhsur), eah));
        gh = FS(FA(sag, pahg), FA(FA(irg, shg), evg));
    }

    float tauxv = -FM(FM(FM(rhoair, cm), ur), uu);
    float tauyv = -FM(FM(FM(rhoair, cm), ur), vv);

    float cah2 = FD(FM(fv, PH_VKC), FS(glibc_logf(FD(FA(2.0f, z0h), z0h)), fh2));
    float cq2v = cah2;
    float t2mv, q2v;
    if (cah2 < 1.0e-5f) {
        t2mv = tah;
        q2v = qsfc;
    } else {
        t2mv = FS(tah, FD(FM(FD(FA(shg, FD(shc, fveg)), FM(rhoair, PH_CPAIR)), 1.0f),
                          cah2));
        q2v = FS(qsfc, FD(FM(FD(FA(FD(FA(evc, tr), fveg), evg),
                                FM(latheav, rhoair)), 1.0f), cq2v));
    }

    out[VO_EAH] = eah;   out[VO_TAH] = tah;   out[VO_TV] = tv;   out[VO_TG] = tg;
    out[VO_CM] = cm;     out[VO_CH] = cah;
    out[VO_TAUXV] = tauxv; out[VO_TAUYV] = tauyv;
    out[VO_IRG] = irg;   out[VO_IRC] = irc;   out[VO_SHG] = shg; out[VO_SHC] = shc;
    out[VO_EVG] = evg;   out[VO_EVC] = evc;   out[VO_TR] = tr;   out[VO_GH] = gh;
    out[VO_T2MV] = t2mv; out[VO_PSNSUN] = psnsun; out[VO_PSNSHA] = psnsha;
    out[VO_CANHS] = canhs; out[VO_QSFC] = qsfc; out[VO_Q2V] = q2v;
    out[VO_CAH2] = cah2; out[VO_CHLEAF] = cvh; out[VO_CHUC] = FD(1.0f, rahg);
    out[VO_RSSUN] = rssun; out[VO_RSSHA] = rssha;
    out[VO_SAV] = sav;   out[VO_SAG] = sag;   out[VO_FSR] = in[VI_FSR];
    (void)dtg;
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

extern "C" __global__ void k_vegeflux_stc2(const float* in,const float* pin,const int* isnow,
 const float* dzsnso,const float* stc,const float* df,float* out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 dev_vege_flux_selected(in+VI_NFLOAT*i,pin+14*i,isnow[i],dzsnso+VF_NLAYER*i,
 stc+VF_NLAYER*i,df+VF_NLAYER*i,out+VO_NOUT*i,2);
}

extern "C" __global__ void k_vegeflux_masked_stc2(const float* in,const float* pin,const int* isnow,
 const float* dzsnso,const float* stc,const float* df,float* out,int n,const unsigned char* active){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!active[i]){for(int j=0;j<VO_NOUT;j++)out[VO_NOUT*i+j]=0.0f;return;}
 dev_vege_flux_selected(in+VI_NFLOAT*i,pin+14*i,isnow[i],dzsnso+VF_NLAYER*i,
 stc+VF_NLAYER*i,df+VF_NLAYER*i,out+VO_NOUT*i,2);
}

extern "C" __global__ void k_vegeflux_crs2(
 const float* in,const float* p,const int* isnow,const float* dz,const float* stc,
 const float* df,const float* fsno,float* out,int n,int stc_option){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 dev_vege_flux_selected(in+VI_NFLOAT*i,p+19*i,isnow[i],dz+VF_NLAYER*i,
 stc+VF_NLAYER*i,df+VF_NLAYER*i,out+VO_NOUT*i,stc_option,fsno[i],2);
}
extern "C" __global__ void k_vegeflux_masked_crs2(
 const float* in,const float* p,const int* isnow,const float* dz,const float* stc,
 const float* df,const float* fsno,float* out,int n,const unsigned char* active,int stc_option){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!active[i]){for(int j=0;j<VO_NOUT;++j)out[VO_NOUT*i+j]=0.0f;return;}
 dev_vege_flux_selected(in+VI_NFLOAT*i,p+19*i,isnow[i],dz+VF_NLAYER*i,
 stc+VF_NLAYER*i,df+VF_NLAYER*i,out+VO_NOUT*i,stc_option,fsno[i],2);
}

extern "C" __global__ void k_vegeflux_masked_sfc2(
 const float* in,const float* p,const int* isnow,const float* dz,const float* stc,
 const float* df,const float* fsno,float* out,int n,const unsigned char* active,int stc_option,int crs){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!active[i]){for(int j=0;j<VO_NOUT;++j)out[VO_NOUT*i+j]=0.0f;return;}
 int pw=crs==2?20:15;const float* pi=p+pw*i;
 dev_vege_flux_selected(in+51*i,pi,isnow[i],dz+VF_NLAYER*i,
 stc+VF_NLAYER*i,df+VF_NLAYER*i,out+VO_NOUT*i,stc_option,fsno[i],crs,2,in[51*i+50],pi[pw-1]);
}
extern "C" __global__ void k_vegeflux_sfc2(
 const float* in,const float* p,const int* isnow,const float* dz,const float* stc,
 const float* df,const float* fsno,float* out,int n,int stc_option,int crs){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int pw=crs==2?20:15;const float* pi=p+pw*i;
 dev_vege_flux_selected(in+51*i,pi,isnow[i],dz+VF_NLAYER*i,
 stc+VF_NLAYER*i,df+VF_NLAYER*i,out+VO_NOUT*i,stc_option,fsno[i],crs,2,in[51*i+50],pi[pw-1]);
}
