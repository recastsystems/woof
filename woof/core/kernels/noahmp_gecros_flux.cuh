// WRF 4.7.1 VEGE_FLUX GECROS extension; WRF public-domain notice applies.
__device__ void dev_vege_flux_gecros(const float *in, const float *p, int isnow,
                              const float *dzsnso, const float *stc,
                              const float *df, float *out,
                              int opt_stc, float fsno, int opt_crs, int opt_sfc, float thair, float czil, float *gs, const float *gx)
{
    const float dt = in[VI_DT]; float sav = in[VI_SAV], sag = in[VI_SAG], fsr = in[VI_FSR];
    const bool crop_active = gx[4]>0.0f && gs[40]>0.0f && gs[41]<0.0f;
    float frsu=0.0f, glaie=0.0f;
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
            if(crop_active){
                float thickness=0.0f,wul=0.0f,wll=0.0f,rd=FD(gs[32],100.0f);
                int nroot=0;
                for(int j=0;j<4;++j){thickness=FA(thickness,dzsnso[3+j]);if(thickness<rd)++nroot;}
                nroot=max(1,nroot+1); thickness=0.0f;
                // Native RDMX=130 cm fits the four-layer 200 cm soil column.
                for(int j=0;j<nroot && j<4;++j){
                    thickness=FA(thickness,dzsnso[3+j]);
                    float depth=thickness>rd?FA(FS(rd,thickness),dzsnso[3+j]):dzsnso[3+j];
                    wul=FA(wul,FM(FM(depth,1000.0f),FS(gx[5+j],gx[9+j])));
                }
                for(int j=0;j<4;++j)wll=FA(wll,FM(FM(dzsnso[3+j],1000.0f),FS(gx[5+j],gx[9+j])));
                wll=FS(wll,wul);
                noahmp_gecros::Context ctx; noahmp_gecros::initial_context(ctx);
                noahmp_gecros::gecros(ctx,gx[0],dt,1,rb,rahc,FA(rahg,rsurf),gx[3],snowh,ur,sfctmp,eair,gx[1],lwdn,gx[2],wul,wll,gx[9],p[PP_DLEAF],gs,sav,sag,fsr,frsu,rssun,rssha);
                glaie=fmnf(6.0f,FD(gs[48],fveg));
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
        if(crop_active)ctw=FM(FS(1.0f,fwet),FA(FD(1.0f,FA(FD(rb,FM(frsu,glaie)),rssun)),FD(1.0f,FA(FD(rb,FM(FS(1.0f,frsu),glaie)),rssha))));
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
    out[VO_SAV] = sav;   out[VO_SAG] = sag;   out[VO_FSR] = fsr;
    (void)dtg;
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

extern "C" __global__ void k_gecros_flux(const float *in,const float *p,const int *isnow,const float *dz,const float *stc,const float *df,const float *fsno,float *out,int n,const unsigned char *active,int stcopt,int crsopt,int sfcopt,float *gs,const float *gx){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!active[i]){for(int j=0;j<VO_NOUT;++j)out[i*VO_NOUT+j]=0.0f;return;}
 dev_vege_flux_gecros(in+(VI_NFLOAT+(sfcopt==2))*i,p+(14+5*(crsopt==2)+(sfcopt==2))*i,isnow[i],dz+7*i,stc+7*i,df+7*i,out+VO_NOUT*i,stcopt,fsno[i],crsopt,sfcopt,gx[15*i+13],gx[15*i+14],gs+60*i,gx+15*i);
}
