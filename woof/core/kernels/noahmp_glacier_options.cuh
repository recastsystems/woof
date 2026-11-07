// WRF v4 Noah-MP 848f54ad3d28c4303151fe5ad83724e232694422.
// Selected-option extension compiled after the unchanged default unit.
// Public-domain source notice: licenses/LICENSE-WRF-public-domain.txt.

template<int GLA, int ALB, bool LOCAL = false>
__device__ void ng_glacier_column(
    const float *__restrict__ in, const int *__restrict__ ii,
    const float *__restrict__ zsoil_row, float dt,
    float *__restrict__ out, int *__restrict__ io, int n, int opt_stc=1, int opt_tbot=2, const float *snow=nullptr, const float *solar=nullptr, float *grains=nullptr)
{
    int tid = LOCAL ? 0 : blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float *x = in + (size_t) tid * NG_NIN;
    float *o = out + (size_t) tid * NG_NOUT;
    int isnow = ii[tid];
    int err = 0;

    const float cosz = x[0], sfctmp = x[1], sfcprs = x[2];
    const float uu = x[3], vv = x[4], q2 = x[5], soldn = x[6];
    const float prcp = x[7], lwdn = x[8], tbot = x[9], zref = x[10];
    float qsnow = x[11], sneqvo = x[12], albold = x[13];
    float cm = x[14], ch = x[15];
    float sneqv = x[16], snowh = x[17], tg = x[18], tauss = x[19];
    float qsfc = x[20];

    float ficeold[3], smc[4], sh2o[4], zsnso[7], stc[7], snice[3], snliq[3];
    float zsoil[4];
    for (int k = 0; k < 3; ++k) ficeold[k] = x[21 + k];
    for (int k = 0; k < 4; ++k) smc[k] = x[24 + k];
    for (int k = 0; k < 4; ++k) sh2o[k] = x[28 + k];
    for (int k = 0; k < 7; ++k) zsnso[k] = x[32 + k];
    for (int k = 0; k < 7; ++k) stc[k] = x[39 + k];
    for (int k = 0; k < 3; ++k) snice[k] = x[46 + k];
    for (int k = 0; k < 3; ++k) snliq[k] = x[49 + k];
    for (int k = 0; k < 4; ++k) zsoil[k] = zsoil_row[k];

    // ---- ATM_GLACIER :330-347 (THAIR feeds nothing and is skipped) ------
    const float qair = q2;
    const float eair = DV(MU(qair, sfcprs), AD(0.622f, MU(0.378f, qair)));
    const float rhoair = DV(SU(sfcprs, MU(0.378f, eair)),
                            MU(NMP_RAIR, sfctmp));
    float swdown = (cosz <= 0.0f) ? 0.0f : soldn;
    float solad[2], solai[2];
    solad[0] = MU(MU(swdown, 0.7f), 0.5f);
    solad[1] = MU(MU(swdown, 0.7f), 0.5f);
    solai[0] = MU(MU(swdown, 0.3f), 0.5f);
    solai[1] = MU(MU(swdown, 0.3f), 0.5f);

    const float beg_wb = sneqv;                                    // :225

    // ---- layer thickness :229-235 ---------------------------------------
    float dzsnso[7];
    for (int k = 0; k < 7; ++k) dzsnso[k] = 0.0f;
    for (int iz = isnow + 1; iz <= NMP_NSOIL; ++iz) {
        if (iz == isnow + 1) dzsnso[GS(iz)] = SU(0.0f, zsnso[GS(iz)]);
        else dzsnso[GS(iz)] = SU(zsnso[GS(iz - 1)], zsnso[GS(iz)]);
    }

    // =====================================================================
    // ENERGY_GLACIER :352-532
    // =====================================================================
    // :449  UR = MAX(SQRT(UU**2.+VV**2.), 1.) -- REAL exponents -> powf.
    float ur = ng_max(__fsqrt_rn(AD(r_pow(uu, 2.0f), r_pow(vv, 2.0f))),
                      1.0f);
    const float zpd = snowh;                                       // :454
    const float zlvl = AD(zpd, zref);                              // :456

    // ---- THERMOPROP_GLACIER :534-604 (CSNOW_GLACIER :607-661) -----------
    float df[7], hcpct[7], fact[7];
    for (int k = 0; k < 7; ++k) { df[k] = 0.0f; hcpct[k] = 0.0f; fact[k] = 0.0f; }
    {
        float snicev[3], epore[3], snliqv[3], bdsnoi[3];
        for (int k = 0; k < 3; ++k) { snicev[k] = 0.0f; epore[k] = 0.0f; snliqv[k] = 0.0f; bdsnoi[k] = 0.0f; }
        for (int iz = isnow + 1; iz <= 0; ++iz) {                  // :639-643
            snicev[SN(iz)] = ng_min(1.0f, DV(snice[SN(iz)],
                                MU(dzsnso[GS(iz)], NMP_DENICE)));
            epore[SN(iz)] = SU(1.0f, snicev[SN(iz)]);
            snliqv[SN(iz)] = ng_min(epore[SN(iz)], DV(snliq[SN(iz)],
                                MU(dzsnso[GS(iz)], NMP_DENH2O)));
        }
        for (int iz = isnow + 1; iz <= 0; ++iz) {                  // :645-649
            bdsnoi[SN(iz)] = DV(AD(snice[SN(iz)], snliq[SN(iz)]),
                                dzsnso[GS(iz)]);
            hcpct[GS(iz)] = AD(MU(NMP_CICE, snicev[SN(iz)]),
                               MU(NMP_CWAT, snliqv[SN(iz)]));
        }
        for (int iz = isnow + 1; iz <= 0; ++iz)                    // :653-659
            df[GS(iz)] = MU(3.2217e-6f, r_pow(bdsnoi[SN(iz)], 2.0f));

        for (int iz = 1; iz <= NMP_NSOIL; ++iz) {                  // :580-587
            float zmid = MU(0.5f, dzsnso[GS(iz)]);
            for (int iz2 = 1; iz2 <= iz - 1; ++iz2)
                zmid = AD(zmid, dzsnso[GS(iz2)]);
            hcpct[GS(iz)] = MU(1.0e6f, AD(0.8194f, MU(0.1309f, zmid)));
            df[GS(iz)] = AD(0.32333f, MU(0.10073f, zmid));
        }
        for (int iz = isnow + 1; iz <= NMP_NSOIL; ++iz)            // :591-593
            fact[GS(iz)] = DV(dt, MU(hcpct[GS(iz)], dzsnso[GS(iz)]));
        if (isnow == 0) {                                          // :597-601
            df[GS(1)] = DV(AD(MU(df[GS(1)], dzsnso[GS(1)]),
                              MU(0.35f, snowh)),
                           AD(snowh, dzsnso[GS(1)]));
        } else {
            df[GS(1)] = DV(AD(MU(df[GS(1)], dzsnso[GS(1)]),
                              MU(df[GS(0)], dzsnso[GS(0)])),
                           AD(dzsnso[GS(0)], dzsnso[GS(1)]));
        }
    }

    // ---- RADIATION_GLACIER :663-756 (opt_alb = 2) -----------------------
    float sag, fsa, fsr;
    {
        float albsnd[2] = {0.0f, 0.0f};
        float albsni[2] = {0.0f, 0.0f};
        #ifndef GPUWM_NOAHMP_NIGHT_AGE
        #define GPUWM_NOAHMP_NIGHT_AGE 1
        #endif
        if (ALB == 3) { tauss=snow[4]; albold=snow[5]; }
        if (ALB != 3 && (GPUWM_NOAHMP_NIGHT_AGE || cosz > 0.0f)) {
            // SNOW_AGE_GLACIER :758-809
            if (sneqv <= 0.0f) {
                tauss = 0.0f;
            } else if (sneqv > 800.0f) {
                tauss = 0.0f;
            } else {
                float dela0 = MU(1.0e-6f, dt);
                float arg = MU(5.0e3f, SU(DV(1.0f, NMP_TFRZ),
                                          DV(1.0f, tg)));
                float age1 = r_exp(arg);
                float age2 = r_exp(ng_min(0.0f, MU(10.0f, arg)));
                float age3 = 0.3f;
                float tage = AD(AD(age1, age2), age3);
                float dela = MU(dela0, tage);
                float dels = DV(ng_max(0.0f, SU(sneqv, sneqvo)), NG_SWEMX);
                float sge = MU(AD(tauss, dela), SU(1.0f, dels));
                tauss = ng_max(0.0f, sge);
            }
        }
        if (cosz > 0.0f) {
            if (ALB == 3) {
                albsnd[0]=snow[0]; albsnd[1]=snow[1];
                albsni[0]=snow[2]; albsni[1]=snow[3];
            } else if (ALB == 1) {
                // SNOWALB_BATS_GLACIER :812-859 has native fixed constants;
                // land's seven transferred BATS parameters do not apply here.
                float fage = DV(tauss, AD(tauss, 1.0f));
                float sl1 = DV(1.0f, 2.0f), sl2 = MU(2.0f, 2.0f);
                float cf1 = SU(DV(AD(1.0f, sl1),
                                  AD(1.0f, MU(sl2, cosz))), sl1);
                float fzen = ng_max(cf1, 0.0f);
                albsni[0] = MU(0.95f, SU(1.0f, MU(0.2f, fage)));
                albsni[1] = MU(0.65f, SU(1.0f, MU(0.5f, fage)));
                albsnd[0] = AD(albsni[0],
                    MU(MU(0.4f, fzen), SU(1.0f, albsni[0])));
                albsnd[1] = AD(albsni[1],
                    MU(MU(0.4f, fzen), SU(1.0f, albsni[1])));
            } else {
            // SNOWALB_CLASS_GLACIER :861-904
            float alb = AD(0.55f, MU(SU(albold, 0.55f),
                           r_exp(DV(MU(-0.01f, dt), 3600.0f))));
            if (qsnow > 0.0f) {
                alb = AD(alb, DV(MU(ng_min(MU(qsnow, dt), NG_SWEMX),
                                    SU(0.84f, alb)), NG_SWEMX));
            }
            albold = alb;
            albsni[0] = alb; albsni[1] = alb;
            albsnd[0] = alb; albsnd[1] = alb;
            }
        }
        sag = 0.0f; fsa = 0.0f; fsr = 0.0f;                        // :731-733
        if (ALB == 3) {
            albsnd[0]=snow[0]; albsnd[1]=snow[1];
            albsni[0]=snow[2]; albsni[1]=snow[3];
        }
        float fsno = (ALB == 3 || sneqv > 0.0f) ? 1.0f : 0.0f;                 // :735-736
        const float albice[2] = {0.80f, 0.55f};                    // :708-709
        for (int ib = 0; ib < 2; ++ib) {                           // :740-754
            albsnd[ib] = AD(MU(albice[ib], SU(1.0f, fsno)),
                            MU(albsnd[ib], fsno));
            albsni[ib] = AD(MU(albice[ib], SU(1.0f, fsno)),
                            MU(albsni[ib], fsno));
            float absorbed = AD(MU(solad[ib], SU(1.0f, albsnd[ib])),
                                MU(solai[ib], SU(1.0f, albsni[ib])));
            sag = AD(sag, absorbed);
            fsa = AD(fsa, absorbed);
            float ref = AD(MU(solad[ib], albsnd[ib]),
                           MU(solai[ib], albsni[ib]));
            fsr = AD(fsr, ref);
        }
    }

    const float gamma = DV(MU(NMP_CPAIR, sfcprs),
                           MU(0.622f, NG_HSUB));                   // :484
    const float lathea = NG_HSUB;                                  // :483

    // ---- GLACIER_FLUX :906-1121 -----------------------------------------
    float irb, shb, evb, ghb, t2mb, q2b, ehb2;
    {
        float mpe = NG_MPE;
        float h = 0.0f;
        NgSfc st;
        st.moz = 0.0f; st.fm = 0.0f; st.fh = 0.0f; st.fm2 = 0.0f;
        st.fh2 = 0.0f; st.fv = 0.1f; st.cm = 0.0f; st.ch = 0.0f;
        st.ch2 = 0.0f; st.mozsgn = 0;

        const float cir = MU(NG_EMG, NG_SB);                       // :1018
        const float cgh = DV(MU(2.0f, df[GS(isnow + 1)]),
                             dzsnso[GS(isnow + 1)]);               // :1019
        float csh = 0.0f, cev = 0.0f, estg = 0.0f, rahb = 1.0f;
        irb = 0.0f; shb = 0.0f; evb = 0.0f; ghb = 0.0f;
        const float rsurf = 1.0f, rhsur = 1.0f;                    // :478-479

        for (int it = 1; it <= NG_NITERB; ++it) {                  // :1022-1087
            float z0h = NG_Z0SNO;                                  // :1024
            ng_sfcdif1(&st, it, sfctmp, rhoair, h, qair, zlvl, zpd,
                       NG_Z0SNO, z0h, ur, mpe);                    // :1028-1031

            rahb = ng_max(1.0f, DV(1.0f, MU(st.ch, ur)));          // :1034
            float rawb = rahb;                                     // :1035

            float t = ng_tdc(tg);                                  // :1039
            float esatw = ng_esat_poly(0, t);
            float esati = ng_esat_poly(7, t);
            float dsatw = ng_esat_poly(14, t);
            float dsati = ng_esat_poly(21, t);
            float destg;
            if (t > 0.0f) { estg = esatw; destg = dsatw; }
            else { estg = esati; destg = dsati; }

            csh = DV(MU(rhoair, NMP_CPAIR), rahb);                 // :1049
            cev = DV(DV(MU(rhoair, NMP_CPAIR), gamma),
                     AD(rsurf, rawb));                             // :1051
            if (GLA == 2 && !(snowh > 0.0f)) cev = 0.0f;

            irb = SU(MU(cir, ng_powi4(tg)), MU(NG_EMG, lwdn));     // :1058
            shb = MU(csh, SU(tg, sfctmp));                         // :1059
            evb = MU(cev, SU(MU(estg, rhsur), eair));              // :1060
            ghb = MU(cgh, SU(tg, stc[GS(isnow + 1)]));             // :1061

            float b = SU(SU(SU(SU(sag, irb), shb), evb), ghb);     // :1063
            float cir4t3 = MU(MU(4.0f, cir), ng_powi3(tg));
            float a = AD(AD(AD(cir4t3, csh), MU(cev, destg)), cgh);
            float dtg = DV(b, a);                                  // :1065

            irb = AD(irb, MU(cir4t3, dtg));                        // :1067
            shb = AD(shb, MU(csh, dtg));
            evb = AD(evb, MU(MU(cev, destg), dtg));
            ghb = AD(ghb, MU(cgh, dtg));

            tg = AD(tg, dtg);                                      // :1073
            h = MU(csh, SU(tg, sfctmp));                           // :1076

            t = ng_tdc(tg);                                        // :1078-1084
            esatw = ng_esat_poly(0, t);
            esati = ng_esat_poly(7, t);
            estg = (t > 0.0f) ? esatw : esati;
            float er = MU(estg, rhsur);
            qsfc = DV(MU(0.622f, er), SU(sfcprs, MU(0.378f, er))); // :1085
        }

        // :1092-1105 -- OPT_STC = 1, OPT_GLA = 1 reset over ice/snow.
        float max_sice = SU(smc[0], sh2o[0]);
        for (int j = 2; j <= NMP_NSOIL; ++j) {
            float s = SU(smc[j - 1], sh2o[j - 1]);
            if (s > max_sice) max_sice = s;
        }
        if (opt_stc != 2 && GLA == 1 && (max_sice > 0.0f || snowh > 0.0f) && tg > NMP_TFRZ) {
            tg = NMP_TFRZ;
            float t = ng_tdc(tg);
            float esati = ng_esat_poly(7, t);
            estg = esati;                                          // :1098
            float er = MU(estg, rhsur);
            qsfc = DV(MU(0.622f, er), SU(sfcprs, MU(0.378f, er)));
            irb = SU(MU(cir, ng_powi4(tg)), MU(NG_EMG, lwdn));     // :1100
            shb = MU(csh, SU(tg, sfctmp));
            evb = MU(cev, SU(MU(estg, rhsur), eair));
            ghb = SU(sag, AD(AD(irb, shb), evb));                  // :1103
        }

        // 2 m diagnostics :1108-1116
        float z0h = NG_Z0SNO;
        ehb2 = DV(MU(st.fv, NG_VKC),
                  SU(r_log(DV(AD(2.0f, z0h), z0h)), st.fh2));
        float cq2b = ehb2;
        if (ehb2 < 1.0e-5f) {
            t2mb = tg;
            q2b = qsfc;
        } else {
            t2mb = SU(tg, DV(MU(DV(shb, MU(rhoair, NMP_CPAIR)), 1.0f),
                             ehb2));
            q2b = SU(qsfc, MU(DV(evb, MU(lathea, rhoair)),
                              AD(DV(1.0f, cq2b), rsurf)));
        }
        cm = st.cm;
        ch = DV(1.0f, rahb);                                       // :1119
    }

    const float fira = irb, ssoil = ghb;
    float fsh = shb, fgev = evb; // GLA=2 WATER adjusts snow-only latent flux.

    const float fire = AD(lwdn, fira);                             // :498
    if (fire <= 0.0f) err = 1;                                     // :500
    const float emissi = NG_EMG;                                   // :503
    const float trad = r_pow(DV(SU(fire, MU(SU(1.0f, emissi), lwdn)),
                                MU(emissi, NG_SB)), 0.25f);        // :509

    // ---- TSNOSOI_GLACIER :1333-1393 -------------------------------------
    float eflxb = 0.0f;
    {
        const float zbotsno = SU(NG_ZBOT, snowh);                  // :1379
        float ai[7], bi[7], ci[7], rhsts[7], ddz[7], denom[7], dtsdz[7];
        float eflux[7];
        for (int k = 0; k < 7; ++k) {
            ai[k] = 0.0f; bi[k] = 0.0f; ci[k] = 0.0f; rhsts[k] = 0.0f;
            ddz[k] = 0.0f; denom[k] = 0.0f; dtsdz[k] = 0.0f; eflux[k] = 0.0f;
        }
        float thermal_ssoil=ssoil;
        if(ALB==3)for(int k=isnow+2;k<=NMP_NSOIL;++k)thermal_ssoil-=solar[GS(k)];
        for (int k = isnow + 1; k <= NMP_NSOIL; ++k) {             // :1442-1467
            const float phi=(ALB==3&&k>isnow+1)?solar[GS(k)]:0.f;
            if (k == isnow + 1) {
                denom[GS(k)] = MU(SU(0.0f, zsnso[GS(k)]), hcpct[GS(k)]);
                float temp1 = SU(0.0f, zsnso[GS(k + 1)]);
                ddz[GS(k)] = DV(2.0f, temp1);
                dtsdz[GS(k)] = DV(MU(2.0f, SU(stc[GS(k)], stc[GS(k + 1)])),
                                  temp1);
                eflux[GS(k)] = SU(SU(MU(df[GS(k)], dtsdz[GS(k)]), thermal_ssoil),
                                  phi);
            } else if (k < NMP_NSOIL) {
                denom[GS(k)] = MU(SU(zsnso[GS(k - 1)], zsnso[GS(k)]),
                                  hcpct[GS(k)]);
                float temp1 = SU(zsnso[GS(k - 1)], zsnso[GS(k + 1)]);
                ddz[GS(k)] = DV(2.0f, temp1);
                dtsdz[GS(k)] = DV(MU(2.0f, SU(stc[GS(k)], stc[GS(k + 1)])),
                                  temp1);
                eflux[GS(k)] = SU(SU(MU(df[GS(k)], dtsdz[GS(k)]),
                                     MU(df[GS(k - 1)], dtsdz[GS(k - 1)])),
                                  phi);
            } else {
                denom[GS(k)] = MU(SU(zsnso[GS(k - 1)], zsnso[GS(k)]),
                                  hcpct[GS(k)]);
                if (opt_tbot == 1) eflxb = 0.0f;
                else {
                // OPT_TBOT == 2 :1461-1464
                dtsdz[GS(k)] = DV(SU(stc[GS(k)], tbot),
                                  SU(MU(0.5f, AD(zsnso[GS(k - 1)],
                                                 zsnso[GS(k)])), zbotsno));
                eflxb = MU(SU(0.0f, df[GS(k)]), dtsdz[GS(k)]);
                }
                eflux[GS(k)] = SU(SU(SU(0.0f, eflxb),
                                     MU(df[GS(k - 1)], dtsdz[GS(k - 1)])),
                                  phi);
            }
        }
        for (int k = isnow + 1; k <= NMP_NSOIL; ++k) {             // :1469-1489
            if (k == isnow + 1) {
                ai[GS(k)] = 0.0f;
                ci[GS(k)] = DV(MU(SU(0.0f, df[GS(k)]), ddz[GS(k)]),
                               denom[GS(k)]);
                bi[GS(k)] = SU(0.0f, ci[GS(k)]);
                if (opt_stc == 2)
                    bi[GS(k)] = AD(SU(0.0f,ci[GS(k)]), DV(df[GS(k)],
                        MU(MU(MU(0.5f,zsnso[GS(k)]),zsnso[GS(k)]),hcpct[GS(k)])));
            } else if (k < NMP_NSOIL) {
                ai[GS(k)] = DV(MU(SU(0.0f, df[GS(k - 1)]), ddz[GS(k - 1)]),
                               denom[GS(k)]);
                ci[GS(k)] = DV(MU(SU(0.0f, df[GS(k)]), ddz[GS(k)]),
                               denom[GS(k)]);
                bi[GS(k)] = SU(0.0f, AD(ai[GS(k)], ci[GS(k)]));
            } else {
                ai[GS(k)] = DV(MU(SU(0.0f, df[GS(k - 1)]), ddz[GS(k - 1)]),
                               denom[GS(k)]);
                ci[GS(k)] = 0.0f;
                bi[GS(k)] = SU(0.0f, AD(ai[GS(k)], ci[GS(k)]));
            }
            rhsts[GS(k)] = DV(eflux[GS(k)], SU(0.0f, denom[GS(k)]));
        }
        // HSTEP_GLACIER :1522-1544
        for (int k = isnow + 1; k <= NMP_NSOIL; ++k) {
            rhsts[GS(k)] = MU(rhsts[GS(k)], dt);
            ai[GS(k)] = MU(ai[GS(k)], dt);
            bi[GS(k)] = AD(1.0f, MU(bi[GS(k)], dt));
            ci[GS(k)] = MU(ci[GS(k)], dt);
        }
        float rhstsin[7], ciin[7];
        for (int k = 0; k < 7; ++k) { rhstsin[k] = rhsts[k]; ciin[k] = ci[k]; }
        // ROSR12_GLACIER :1548-1605 (P := ci, DELTA := rhsts)
        {
            int ntop = isnow + 1;
            ciin[GS(NMP_NSOIL)] = 0.0f;                            // :1579
            ci[GS(ntop)] = DV(SU(0.0f, ciin[GS(ntop)]), bi[GS(ntop)]);
            rhsts[GS(ntop)] = DV(rhstsin[GS(ntop)], bi[GS(ntop)]); // :1584
            for (int k = ntop + 1; k <= NMP_NSOIL; ++k) {          // :1588-1592
                ci[GS(k)] = MU(SU(0.0f, ciin[GS(k)]),
                               DV(1.0f, AD(bi[GS(k)],
                                           MU(ai[GS(k)], ci[GS(k - 1)]))));
                rhsts[GS(k)] = MU(SU(rhstsin[GS(k)],
                                     MU(ai[GS(k)], rhsts[GS(k - 1)])),
                                  DV(1.0f, AD(bi[GS(k)],
                                              MU(ai[GS(k)], ci[GS(k - 1)]))));
            }
            ci[GS(NMP_NSOIL)] = rhsts[GS(NMP_NSOIL)];              // :1596
            for (int k = ntop + 1; k <= NMP_NSOIL; ++k) {          // :1600-1603
                int kk = NMP_NSOIL - k + (ntop - 1) + 1;
                ci[GS(kk)] = AD(MU(ci[GS(kk)], ci[GS(kk + 1)]),
                                rhsts[GS(kk)]);
            }
        }
        for (int k = isnow + 1; k <= NMP_NSOIL; ++k)               // :1542-1544
            stc[GS(k)] = AD(stc[GS(k)], ci[GS(k)]);
    }

    if (opt_stc == 2 && snowh > 0.05f && tg > NMP_TFRZ) tg = NMP_TFRZ;

    // ---- PHASECHANGE_GLACIER :1608-1995 (OPT_GLA == 1) ------------------
    float qmelt = 0.0f, ponding = 0.0f;
    int imelt[7];
    {
        float hm[7], xm[7], wmass0[7], wice0[7], mice[7], mliq[7], heatr[7];
        for (int k = 0; k < 7; ++k) {
            imelt[k] = 0; hm[k] = 0.0f; xm[k] = 0.0f; wmass0[k] = 0.0f;
            wice0[k] = 0.0f; mice[k] = 0.0f; mliq[k] = 0.0f; heatr[k] = 0.0f;
        }
        for (int j = isnow + 1; j <= 0; ++j) {                     // :1664-1676
            mice[GS(j)] = snice[SN(j)];
            mliq[GS(j)] = snliq[SN(j)];
            wice0[GS(j)] = mice[GS(j)];
            wmass0[GS(j)] = AD(mice[GS(j)], mliq[GS(j)]);
        }
        for (int j = isnow + 1; j <= 0; ++j) {                     // :1678-1686
            if (mice[GS(j)] > 0.0f && stc[GS(j)] >= NMP_TFRZ) imelt[GS(j)] = 1;
            if (mliq[GS(j)] > 0.0f && stc[GS(j)] < NMP_TFRZ) imelt[GS(j)] = 2;
        }
        for (int j = isnow + 1; j <= 0; ++j) {                     // :1690-1705
            if (imelt[GS(j)] > 0) {
                hm[GS(j)] = DV(SU(stc[GS(j)], NMP_TFRZ), fact[GS(j)]);
                stc[GS(j)] = NMP_TFRZ;
            }
            if (imelt[GS(j)] == 1 && hm[GS(j)] < 0.0f) {
                hm[GS(j)] = 0.0f; imelt[GS(j)] = 0;
            }
            if (imelt[GS(j)] == 2 && hm[GS(j)] > 0.0f) {
                hm[GS(j)] = 0.0f; imelt[GS(j)] = 0;
            }
            xm[GS(j)] = DV(MU(hm[GS(j)], dt), NG_HFUS);
        }
        // Simple-glacier shallow snow melt :1710-1734; no ice phase change.
        if (GLA == 2 && isnow == 0 && sneqv > 0.0f && stc[GS(1)] >= NMP_TFRZ) {
            hm[GS(1)] = DV(SU(stc[GS(1)], NMP_TFRZ), fact[GS(1)]);
            stc[GS(1)] = NMP_TFRZ;
            xm[GS(1)] = DV(MU(hm[GS(1)], dt), NG_HFUS);
            float temp1 = sneqv;
            sneqv = ng_max(0.0f, SU(temp1, xm[GS(1)]));
            float propor = DV(sneqv, temp1);
            snowh = ng_max(0.0f, MU(propor, snowh));
            heatr[GS(1)] = SU(hm[GS(1)],
                DV(MU(NG_HFUS, SU(temp1, sneqv)), dt));
            if (heatr[GS(1)] > 0.0f) {
                xm[GS(1)] = DV(MU(heatr[GS(1)], dt), NG_HFUS);
                stc[GS(1)] = AD(stc[GS(1)], MU(fact[GS(1)], heatr[GS(1)]));
            } else {
                xm[GS(1)] = 0.0f; hm[GS(1)] = 0.0f;
            }
            qmelt = DV(ng_max(0.0f, SU(temp1, sneqv)), dt);
            ponding = SU(temp1, sneqv);
        }
        for (int j = isnow + 1; j <= 0; ++j) {                     // :1737-1759
            if (imelt[GS(j)] > 0 && fabsf(hm[GS(j)]) > 0.0f) {
                heatr[GS(j)] = 0.0f;
                if (xm[GS(j)] > 0.0f) {
                    mice[GS(j)] = ng_max(0.0f, SU(wice0[GS(j)], xm[GS(j)]));
                    heatr[GS(j)] = SU(hm[GS(j)],
                        DV(MU(NG_HFUS, SU(wice0[GS(j)], mice[GS(j)])), dt));
                } else if (xm[GS(j)] < 0.0f) {
                    mice[GS(j)] = ng_min(wmass0[GS(j)],
                                         SU(wice0[GS(j)], xm[GS(j)]));
                    heatr[GS(j)] = SU(hm[GS(j)],
                        DV(MU(NG_HFUS, SU(wice0[GS(j)], mice[GS(j)])), dt));
                }
                mliq[GS(j)] = ng_max(0.0f, SU(wmass0[GS(j)], mice[GS(j)]));
                if (fabsf(heatr[GS(j)]) > 0.0f) {
                    stc[GS(j)] = AD(stc[GS(j)],
                                    MU(fact[GS(j)], heatr[GS(j)]));
                    if (MU(mliq[GS(j)], mice[GS(j)]) > 0.0f)
                        stc[GS(j)] = NMP_TFRZ;
                }
                qmelt = AD(qmelt, DV(ng_max(0.0f,
                            SU(wice0[GS(j)], mice[GS(j)])), dt));
            }
        }
        if (GLA == 1) { // native ice phase change :1761-1977
        // ice (soil) layers :1763-1810
        for (int j = 1; j <= NMP_NSOIL; ++j) {
            mliq[GS(j)] = MU(MU(sh2o[j - 1], dzsnso[GS(j)]), 1000.0f);
            mice[GS(j)] = MU(MU(SU(smc[j - 1], sh2o[j - 1]),
                                dzsnso[GS(j)]), 1000.0f);
        }
        for (int j = 1; j <= NMP_NSOIL; ++j) {
            imelt[GS(j)] = 0; hm[GS(j)] = 0.0f; xm[GS(j)] = 0.0f;
            wice0[GS(j)] = mice[GS(j)];
            wmass0[GS(j)] = AD(mice[GS(j)], mliq[GS(j)]);
        }
        for (int j = 1; j <= NMP_NSOIL; ++j) {
            if (mice[GS(j)] > 0.0f && stc[GS(j)] >= NMP_TFRZ) imelt[GS(j)] = 1;
            if (mliq[GS(j)] > 0.0f && stc[GS(j)] < NMP_TFRZ) imelt[GS(j)] = 2;
            if (isnow == 0 && sneqv > 0.0f && j == 1) {            // :1786-1790
                if (stc[GS(j)] >= NMP_TFRZ) imelt[GS(j)] = 1;
            }
        }
        for (int j = 1; j <= NMP_NSOIL; ++j) {                     // :1795-1810
            if (imelt[GS(j)] > 0) {
                hm[GS(j)] = DV(SU(stc[GS(j)], NMP_TFRZ), fact[GS(j)]);
                stc[GS(j)] = NMP_TFRZ;
            }
            if (imelt[GS(j)] == 1 && hm[GS(j)] < 0.0f) {
                hm[GS(j)] = 0.0f; imelt[GS(j)] = 0;
            }
            if (imelt[GS(j)] == 2 && hm[GS(j)] > 0.0f) {
                hm[GS(j)] = 0.0f; imelt[GS(j)] = 0;
            }
            xm[GS(j)] = DV(MU(hm[GS(j)], dt), NG_HFUS);
        }
        // snow without a layer :1814-1832
        if (isnow == 0 && sneqv > 0.0f && xm[GS(1)] > 0.0f) {
            float temp1 = sneqv;
            sneqv = ng_max(0.0f, SU(temp1, xm[GS(1)]));
            float propor = DV(sneqv, temp1);
            snowh = ng_max(0.0f, MU(propor, snowh));
            heatr[GS(1)] = SU(hm[GS(1)],
                DV(MU(NG_HFUS, SU(temp1, sneqv)), dt));
            if (heatr[GS(1)] > 0.0f) {
                xm[GS(1)] = DV(MU(heatr[GS(1)], dt), NG_HFUS);
                hm[GS(1)] = heatr[GS(1)];
                imelt[GS(1)] = 1;
            } else {
                xm[GS(1)] = 0.0f;
                hm[GS(1)] = 0.0f;
                imelt[GS(1)] = 0;
            }
            qmelt = DV(ng_max(0.0f, SU(temp1, sneqv)), dt);
            ponding = SU(temp1, sneqv);
        }
        // melting/freezing for soil :1836-1863
        for (int j = 1; j <= NMP_NSOIL; ++j) {
            if (imelt[GS(j)] > 0 && fabsf(hm[GS(j)]) > 0.0f) {
                heatr[GS(j)] = 0.0f;
                if (xm[GS(j)] > 0.0f) {
                    mice[GS(j)] = ng_max(0.0f, SU(wice0[GS(j)], xm[GS(j)]));
                    heatr[GS(j)] = SU(hm[GS(j)],
                        DV(MU(NG_HFUS, SU(wice0[GS(j)], mice[GS(j)])), dt));
                } else if (xm[GS(j)] < 0.0f) {
                    mice[GS(j)] = ng_min(wmass0[GS(j)],
                                         SU(wice0[GS(j)], xm[GS(j)]));
                    heatr[GS(j)] = SU(hm[GS(j)],
                        DV(MU(NG_HFUS, SU(wice0[GS(j)], mice[GS(j)])), dt));
                }
                mliq[GS(j)] = ng_max(0.0f, SU(wmass0[GS(j)], mice[GS(j)]));
                if (fabsf(heatr[GS(j)]) > 0.0f)
                    stc[GS(j)] = AD(stc[GS(j)],
                                    MU(fact[GS(j)], heatr[GS(j)]));
                // (the J <= 0 and J < 1 arms are dead in this 1..NSOIL loop)
            }
        }
        for (int k = 0; k < 7; ++k) { heatr[k] = 0.0f; xm[k] = 0.0f; }

        // residual redistribution :1867-1975
        bool any_warm = false, any_cold = false;
        for (int j = 1; j <= 4; ++j) {
            if (stc[GS(j)] > NMP_TFRZ) any_warm = true;
            if (stc[GS(j)] < NMP_TFRZ) any_cold = true;
        }
        if (any_warm && any_cold) {                                // :1871-1892
            for (int j = 1; j <= NMP_NSOIL; ++j) {
                if (stc[GS(j)] > NMP_TFRZ) {
                    heatr[GS(j)] = DV(SU(stc[GS(j)], NMP_TFRZ), fact[GS(j)]);
                    for (int k = 1; k <= NMP_NSOIL; ++k) {
                        if (j != k && stc[GS(k)] < NMP_TFRZ
                                && heatr[GS(j)] > 0.1f) {
                            heatr[GS(k)] = DV(SU(stc[GS(k)], NMP_TFRZ),
                                              fact[GS(k)]);
                            if (fabsf(heatr[GS(k)]) > heatr[GS(j)]) {
                                heatr[GS(k)] = AD(heatr[GS(k)], heatr[GS(j)]);
                                stc[GS(k)] = AD(NMP_TFRZ,
                                    MU(heatr[GS(k)], fact[GS(k)]));
                                heatr[GS(j)] = 0.0f;
                            } else {
                                heatr[GS(j)] = AD(heatr[GS(j)], heatr[GS(k)]);
                                heatr[GS(k)] = 0.0f;
                                stc[GS(k)] = NMP_TFRZ;
                            }
                        }
                    }
                    stc[GS(j)] = AD(NMP_TFRZ, MU(heatr[GS(j)], fact[GS(j)]));
                }
            }
        }
        any_warm = false; any_cold = false;
        for (int j = 1; j <= 4; ++j) {
            if (stc[GS(j)] > NMP_TFRZ) any_warm = true;
            if (stc[GS(j)] < NMP_TFRZ) any_cold = true;
        }
        if (any_warm && any_cold) {                                // :1896-1917
            for (int j = 1; j <= NMP_NSOIL; ++j) {
                if (stc[GS(j)] < NMP_TFRZ) {
                    heatr[GS(j)] = DV(SU(stc[GS(j)], NMP_TFRZ), fact[GS(j)]);
                    for (int k = 1; k <= NMP_NSOIL; ++k) {
                        if (j != k && stc[GS(k)] > NMP_TFRZ
                                && heatr[GS(j)] < -0.1f) {
                            heatr[GS(k)] = DV(SU(stc[GS(k)], NMP_TFRZ),
                                              fact[GS(k)]);
                            if (heatr[GS(k)] > fabsf(heatr[GS(j)])) {
                                heatr[GS(k)] = AD(heatr[GS(k)], heatr[GS(j)]);
                                stc[GS(k)] = AD(NMP_TFRZ,
                                    MU(heatr[GS(k)], fact[GS(k)]));
                                heatr[GS(j)] = 0.0f;
                            } else {
                                heatr[GS(j)] = AD(heatr[GS(j)], heatr[GS(k)]);
                                heatr[GS(k)] = 0.0f;
                                stc[GS(k)] = NMP_TFRZ;
                            }
                        }
                    }
                    stc[GS(j)] = AD(NMP_TFRZ, MU(heatr[GS(j)], fact[GS(j)]));
                }
            }
        }
        any_warm = false;
        bool any_ice = false;
        for (int j = 1; j <= 4; ++j) {
            if (stc[GS(j)] > NMP_TFRZ) any_warm = true;
            if (mice[GS(j)] > 0.0f) any_ice = true;
        }
        if (any_warm && any_ice) {                                 // :1921-1946
            for (int j = 1; j <= NMP_NSOIL; ++j) {
                if (stc[GS(j)] > NMP_TFRZ) {
                    heatr[GS(j)] = DV(SU(stc[GS(j)], NMP_TFRZ), fact[GS(j)]);
                    xm[GS(j)] = DV(MU(heatr[GS(j)], dt), NG_HFUS);
                    for (int k = 1; k <= NMP_NSOIL; ++k) {
                        if (j != k && mice[GS(k)] > 0.0f
                                && xm[GS(j)] > 0.1f) {
                            if (mice[GS(k)] > xm[GS(j)]) {
                                mice[GS(k)] = SU(mice[GS(k)], xm[GS(j)]);
                                stc[GS(k)] = NMP_TFRZ;
                                xm[GS(j)] = 0.0f;
                            } else {
                                xm[GS(j)] = SU(xm[GS(j)], mice[GS(k)]);
                                mice[GS(k)] = 0.0f;
                                stc[GS(k)] = NMP_TFRZ;
                            }
                            mliq[GS(k)] = ng_max(0.0f,
                                SU(wmass0[GS(k)], mice[GS(k)]));
                        }
                    }
                    heatr[GS(j)] = DV(MU(xm[GS(j)], NG_HFUS), dt);
                    stc[GS(j)] = AD(NMP_TFRZ, MU(heatr[GS(j)], fact[GS(j)]));
                }
            }
        }
        any_cold = false;
        bool any_liq = false;
        for (int j = 1; j <= 4; ++j) {
            if (stc[GS(j)] < NMP_TFRZ) any_cold = true;
            if (mliq[GS(j)] > 0.0f) any_liq = true;
        }
        if (any_cold && any_liq) {                                 // :1950-1975
            for (int j = 1; j <= NMP_NSOIL; ++j) {
                if (stc[GS(j)] < NMP_TFRZ) {
                    heatr[GS(j)] = DV(SU(stc[GS(j)], NMP_TFRZ), fact[GS(j)]);
                    xm[GS(j)] = DV(MU(heatr[GS(j)], dt), NG_HFUS);
                    for (int k = 1; k <= NMP_NSOIL; ++k) {
                        if (j != k && mliq[GS(k)] > 0.0f
                                && xm[GS(j)] < -0.1f) {
                            if (mliq[GS(k)] > fabsf(xm[GS(j)])) {
                                mice[GS(k)] = SU(mice[GS(k)], xm[GS(j)]);
                                stc[GS(k)] = NMP_TFRZ;
                                xm[GS(j)] = 0.0f;
                            } else {
                                xm[GS(j)] = AD(xm[GS(j)], mliq[GS(k)]);
                                mice[GS(k)] = wmass0[GS(k)];
                                stc[GS(k)] = NMP_TFRZ;
                            }
                            mliq[GS(k)] = ng_max(0.0f,
                                SU(wmass0[GS(k)], mice[GS(k)]));
                        }
                    }
                    heatr[GS(j)] = DV(MU(xm[GS(j)], NG_HFUS), dt);
                    stc[GS(j)] = AD(NMP_TFRZ, MU(heatr[GS(j)], fact[GS(j)]));
                }
            }
        }
        } // GLA == 1
        for (int j = isnow + 1; j <= 0; ++j) {                     // :1979-1982
            if(ALB==3){int k=SN(j);float frozen=fmaxf(0.f,mice[GS(j)]-wice0[GS(j)]);
                float total=wice0[GS(j)]+frozen;
                if(total>0)grains[7*k+1]=(wice0[GS(j)]*grains[7*k+1]+1000.f*frozen)/total;
                grains[7*k+2]=mliq[GS(j)];}
            snliq[SN(j)] = mliq[GS(j)];
            snice[SN(j)] = mice[GS(j)];
        }
        for (int j = 1; j <= NMP_NSOIL; ++j) {                     // :1984-1993
            if (GLA == 1) {
                sh2o[j - 1] = DV(mliq[GS(j)], MU(1000.0f, dzsnso[GS(j)]));
                sh2o[j - 1] = ng_max(0.0f, ng_min(1.0f, sh2o[j - 1]));
            } else sh2o[j - 1] = 0.0f;
            smc[j - 1] = 1.0f;
        }
    }

    // ---- post-ENERGY :250-255 -------------------------------------------
    float sice[4];
    for (int j = 0; j < 4; ++j)
        sice[j] = ng_max(0.0f, SU(smc[j], sh2o[j]));
    const float sneqvo_out = sneqv;                                // :251
    float qvap = ng_max(DV(fgev, lathea), 0.0f);             // :253
    float qdew = fabsf(ng_min(DV(fgev, lathea), 0.0f));      // :254
    float edir = SU(qvap, qdew);                                   // :255

    // =====================================================================
    // WATER_GLACIER :1997-2171
    // =====================================================================
    float runsrf, runsub, fpice, qsnbot, ponding1 = 0.0f, ponding2 = 0.0f;
    {
        float sice_save[4], sh2o_save[4];
        for (int j = 0; j < 4; ++j) {
            sice_save[j] = sice[j];
            sh2o_save[j] = sh2o[j];
        }
        // OPT_SNF == 1 (Jordan 91) :2079-2091
        if (sfctmp > AD(NMP_TFRZ, 2.5f)) {
            fpice = 0.0f;
        } else {
            if (sfctmp <= AD(NMP_TFRZ, 0.5f)) fpice = 1.0f;
            else if (sfctmp <= AD(NMP_TFRZ, 2.0f))
                fpice = SU(1.0f, AD(-54.632f, MU(0.2f, sfctmp)));
            else fpice = 0.6f;
        }
        const float bdfall = ng_min(120.0f,
            AD(67.92f, MU(51.25f,
                          r_exp(DV(SU(sfctmp, NMP_TFRZ), 2.59f)))));
        const float qrain = MU(prcp, SU(1.0f, fpice));             // :2115
        qsnow = MU(prcp, fpice);                                   // :2116
        const float snowhin = DV(qsnow, bdfall);                   // :2117
        float qsnsub = qvap;                                       // :2122
        float qsnfro = qdew;                                       // :2123

        // ---- SNOWWATER_GLACIER :2174-2300 -------------------------------
        float snoflow = 0.0f;
        {
            // SNOWFALL_GLACIER :2302-2364
            int newnode = 0;
            if (isnow == 0 && qsnow > 0.0f) {
                snowh = AD(snowh, MU(snowhin, dt));
                sneqv = AD(sneqv, MU(qsnow, dt));
            }
            if (isnow == 0 && qsnow > 0.0f && snowh >= 0.05f) {
                isnow = -1;
                newnode = 1;
                dzsnso[GS(0)] = snowh;
                snowh = 0.0f;
                stc[GS(0)] = ng_min(273.16f, sfctmp);
                snice[SN(0)] = sneqv;
                snliq[SN(0)] = 0.0f;
            }
            if (isnow < 0 && newnode == 0 && qsnow > 0.0f) {
                snice[SN(isnow + 1)] = AD(snice[SN(isnow + 1)],
                                          MU(qsnow, dt));
                dzsnso[GS(isnow + 1)] = AD(dzsnso[GS(isnow + 1)],
                                           MU(snowhin, dt));
            }

            if (isnow < 0) {                                       // :2230-2242
                // COMPACT_GLACIER :2367-2464
                float burden = 0.0f;
                for (int j = isnow + 1; j <= 0; ++j) {
                    float wx = AD(snice[SN(j)], snliq[SN(j)]);
                    float fice_j = DV(snice[SN(j)], wx);
                    float voidf = SU(1.0f,
                        DV(AD(DV(snice[SN(j)], NMP_DENICE),
                              DV(snliq[SN(j)], NMP_DENH2O)),
                           dzsnso[GS(j)]));
                    if (voidf > 0.001f && snice[SN(j)] > 0.1f) {
                        float bi_ = DV(snice[SN(j)], dzsnso[GS(j)]);
                        float td = ng_max(0.0f, SU(NMP_TFRZ, stc[GS(j)]));
                        float dexpf = r_exp(MU(SU(0.0f, NG_C4), td));
                        float ddz1 = MU(SU(0.0f, NG_C3), dexpf);
                        if (bi_ > NG_DM)
                            ddz1 = MU(ddz1,
                                r_exp(MU(-46.0e-3f, SU(bi_, NG_DM))));
                        if (snliq[SN(j)] > MU(0.01f, dzsnso[GS(j)]))
                            ddz1 = MU(ddz1, NG_C5);
                        float ddz2 = DV(MU(SU(0.0f,
                                              AD(burden, MU(0.5f, wx))),
                                           r_exp(SU(MU(-0.08f, td),
                                                    MU(NG_C2, bi_)))),
                                        NG_ETA0);
                        float ddz3;
                        if (imelt[GS(j)] == 1) {
                            ddz3 = ng_max(0.0f,
                                DV(SU(ficeold[SN(j)], fice_j),
                                   ng_max(1.0e-6f, ficeold[SN(j)])));
                            ddz3 = DV(SU(0.0f, ddz3), dt);
                        } else {
                            ddz3 = 0.0f;
                        }
                        float pdzdtc = MU(AD(AD(ddz1, ddz2), ddz3), dt);
                        pdzdtc = ng_max(-0.5f, pdzdtc);
                        dzsnso[GS(j)] = MU(dzsnso[GS(j)], AD(1.0f, pdzdtc));
                    }
                    burden = AD(burden, wx);
                }

                // COMBINE_GLACIER :2466-2634
                {
                    int isnow_old = isnow;
                    for (int j = isnow_old + 1; j <= 0; ++j) {
                        if (snice[SN(j)] <= 0.1f) {
                            if (j != 0) {
                                snliq[SN(j + 1)] = AD(snliq[SN(j + 1)],
                                                      snliq[SN(j)]);
                                snice[SN(j + 1)] = AD(snice[SN(j + 1)],
                                                      snice[SN(j)]);
                            } else {
                                if (isnow_old < -1) {
                                    snliq[SN(j - 1)] = AD(snliq[SN(j - 1)],
                                                          snliq[SN(j)]);
                                    snice[SN(j - 1)] = AD(snice[SN(j - 1)],
                                                          snice[SN(j)]);
                                } else {
                                    ponding1 = AD(ponding1, snliq[SN(j)]);
                                    sneqv = snice[SN(j)];
                                    snowh = dzsnso[GS(j)];
                                    snliq[SN(j)] = 0.0f;
                                    snice[SN(j)] = 0.0f;
                                    dzsnso[GS(j)] = 0.0f;
                                }
                            }
                            if (j > isnow + 1 && isnow < -1) {
                                for (int i2 = j; i2 >= isnow + 2; --i2) {
                                    stc[GS(i2)] = stc[GS(i2 - 1)];
                                    snliq[SN(i2)] = snliq[SN(i2 - 1)];
                                    snice[SN(i2)] = snice[SN(i2 - 1)];
                                    dzsnso[GS(i2)] = dzsnso[GS(i2 - 1)];
                                }
                            }
                            isnow = isnow + 1;
                        }
                    }
                    if (sice[0] < 0.0f) {                          // :2543-2546
                        sh2o[0] = AD(sh2o[0], sice[0]);
                        sice[0] = 0.0f;
                    }
                    if (isnow != 0) {
                        sneqv = 0.0f;
                        snowh = 0.0f;
                        float zwice = 0.0f, zwliq = 0.0f;
                        for (int j = isnow + 1; j <= 0; ++j) {
                            sneqv = AD(AD(sneqv, snice[SN(j)]),
                                       snliq[SN(j)]);
                            snowh = AD(snowh, dzsnso[GS(j)]);
                            zwice = AD(zwice, snice[SN(j)]);
                            zwliq = AD(zwliq, snliq[SN(j)]);
                        }
                        if (snowh < 0.05f && isnow < 0) {          // :2566-2571
                            isnow = 0;
                            sneqv = zwice;
                            ponding2 = AD(ponding2, zwliq);
                            if (sneqv <= 0.0f) snowh = 0.0f;
                        }
                        if (isnow < -1) {                          // :2582-2632
                            int isnow_old2 = isnow;
                            int mssi = 1;
                            const float dzmin[3] = {0.045f, 0.05f, 0.2f};
                            for (int i2 = isnow_old2 + 1; i2 <= 0; ++i2) {
                                if (dzsnso[GS(i2)] < dzmin[mssi - 1]) {
                                    int neibor;
                                    if (i2 == isnow + 1) neibor = i2 + 1;
                                    else if (i2 == 0) neibor = i2 - 1;
                                    else {
                                        neibor = i2 + 1;
                                        if (AD(dzsnso[GS(i2 - 1)],
                                               dzsnso[GS(i2)])
                                                < AD(dzsnso[GS(i2 + 1)],
                                                     dzsnso[GS(i2)]))
                                            neibor = i2 - 1;
                                    }
                                    int jj, ll;
                                    if (neibor > i2) { jj = neibor; ll = i2; }
                                    else { jj = i2; ll = neibor; }
                                    ng_combo(&dzsnso[GS(jj)], &snliq[SN(jj)],
                                             &snice[SN(jj)], &stc[GS(jj)],
                                             dzsnso[GS(ll)], snliq[SN(ll)],
                                             snice[SN(ll)], stc[GS(ll)]);
                                    if (jj - 1 > isnow + 1) {
                                        for (int k2 = jj - 1;
                                             k2 >= isnow + 2; --k2) {
                                            stc[GS(k2)] = stc[GS(k2 - 1)];
                                            snice[SN(k2)] = snice[SN(k2 - 1)];
                                            snliq[SN(k2)] = snliq[SN(k2 - 1)];
                                            dzsnso[GS(k2)] =
                                                dzsnso[GS(k2 - 1)];
                                        }
                                    }
                                    isnow = isnow + 1;
                                    if (isnow >= -1) break;
                                } else {
                                    mssi = mssi + 1;
                                }
                            }
                        }
                    }
                }

                // DIVIDE_GLACIER :2689-2812
                {
                    float dzl[3], swice[3], swliq[3], tsno[3];
                    for (int k = 0; k < 3; ++k) {
                        dzl[k] = 0.0f; swice[k] = 0.0f;
                        swliq[k] = 0.0f; tsno[k] = 0.0f;
                    }
                    for (int j = 1; j <= 3; ++j) {
                        if (j <= -isnow) {
                            dzl[j - 1] = dzsnso[GS(j + isnow)];
                            swice[j - 1] = snice[SN(j + isnow)];
                            swliq[j - 1] = snliq[SN(j + isnow)];
                            tsno[j - 1] = stc[GS(j + isnow)];
                        }
                    }
                    int msno = -isnow;
                    if (msno == 1) {
                        if (dzl[0] > 0.05f) {
                            msno = 2;
                            dzl[0] = DV(dzl[0], 2.0f);
                            swice[0] = DV(swice[0], 2.0f);
                            swliq[0] = DV(swliq[0], 2.0f);
                            dzl[1] = dzl[0];
                            swice[1] = swice[0];
                            swliq[1] = swliq[0];
                            tsno[1] = tsno[0];
                        }
                    }
                    if (msno > 1) {
                        if (dzl[0] > 0.05f) {
                            float drr = SU(dzl[0], 0.05f);
                            float propor = DV(drr, dzl[0]);
                            float zwice = MU(propor, swice[0]);
                            float zwliq = MU(propor, swliq[0]);
                            propor = DV(0.05f, dzl[0]);
                            swice[0] = MU(propor, swice[0]);
                            swliq[0] = MU(propor, swliq[0]);
                            dzl[0] = 0.05f;
                            ng_combo(&dzl[1], &swliq[1], &swice[1], &tsno[1],
                                     drr, zwliq, zwice, tsno[0]);
                            if (msno <= 2 && dzl[1] > 0.10f) {     // :2763
                                msno = 3;
                                float dtdz = DV(SU(tsno[0], tsno[1]),
                                                DV(AD(dzl[0], dzl[1]), 2.0f));
                                dzl[1] = DV(dzl[1], 2.0f);
                                swice[1] = DV(swice[1], 2.0f);
                                swliq[1] = DV(swliq[1], 2.0f);
                                dzl[2] = dzl[1];
                                swice[2] = swice[1];
                                swliq[2] = swliq[1];
                                tsno[2] = SU(tsno[1],
                                    DV(MU(dtdz, dzl[1]), 2.0f));
                                if (tsno[2] >= NMP_TFRZ) {
                                    tsno[2] = tsno[1];
                                } else {
                                    tsno[1] = AD(tsno[1],
                                        DV(MU(dtdz, dzl[1]), 2.0f));
                                }
                            }
                        }
                    }
                    if (msno > 2) {
                        if (dzl[1] > 0.2f) {
                            float drr = SU(dzl[1], 0.2f);
                            float propor = DV(drr, dzl[1]);
                            float zwice = MU(propor, swice[1]);
                            float zwliq = MU(propor, swliq[1]);
                            propor = DV(0.2f, dzl[1]);
                            swice[1] = MU(propor, swice[1]);
                            swliq[1] = MU(propor, swliq[1]);
                            dzl[1] = 0.2f;
                            ng_combo(&dzl[2], &swliq[2], &swice[2], &tsno[2],
                                     drr, zwliq, zwice, tsno[1]);
                        }
                    }
                    isnow = -msno;
                    for (int j = isnow + 1; j <= 0; ++j) {
                        dzsnso[GS(j)] = dzl[j - isnow - 1];
                        snice[SN(j)] = swice[j - isnow - 1];
                        snliq[SN(j)] = swliq[j - isnow - 1];
                        stc[GS(j)] = tsno[j - isnow - 1];
                    }
                }
            }

            for (int iz = -NMP_NSNOW + 1; iz <= isnow; ++iz) {     // :2246-2252
                snice[SN(iz)] = 0.0f;
                snliq[SN(iz)] = 0.0f;
                stc[GS(iz)] = 0.0f;
                dzsnso[GS(iz)] = 0.0f;
                zsnso[GS(iz)] = 0.0f;
            }

            // SNOWH2O_GLACIER :2814-2971
            {
                if (sneqv == 0.0f) {                               // :2868-2876
                    if (GLA == 1) {
                    sice[0] = AD(sice[0],
                        DV(MU(SU(qsnfro, qsnsub), dt),
                           MU(dzsnso[GS(1)], 1000.0f)));
                    } else {
                        fsh = SU(fsh, MU(SU(qsnfro, qsnsub), NG_HSUB));
                        qsnfro = 0.0f; qsnsub = 0.0f;
                    }
                }
                if (isnow == 0 && sneqv > 0.0f) {                  // :2883-2904
                    if (GLA == 1) {
                    float temp = sneqv;
                    sneqv = AD(SU(sneqv, MU(qsnsub, dt)), MU(qsnfro, dt));
                    float propor = DV(sneqv, temp);
                    snowh = ng_max(0.0f, MU(propor, snowh));
                    } else {
                        fsh = SU(fsh, MU(SU(qsnfro, qsnsub), NG_HSUB));
                        qsnfro = 0.0f; qsnsub = 0.0f;
                    }
                    if (sneqv < 0.0f) {
                        sice[0] = AD(sice[0],
                            DV(sneqv, MU(dzsnso[GS(1)], 1000.0f)));
                        sneqv = 0.0f;
                        snowh = 0.0f;
                    }
                    if (sice[0] < 0.0f) {
                        sh2o[0] = AD(sh2o[0], sice[0]);
                        sice[0] = 0.0f;
                    }
                }
                if (snowh <= 1.0e-8f || sneqv <= 1.0e-6f) {        // :2906-2909
                    snowh = 0.0f;
                    sneqv = 0.0f;
                }
                if (isnow < 0) {                                   // :2913-2929
                    float wgdif = AD(SU(snice[SN(isnow + 1)],
                                        MU(qsnsub, dt)), MU(qsnfro, dt));
                    snice[SN(isnow + 1)] = wgdif;
                    if (wgdif < 1.0e-6f && isnow < 0) {
                        // COMBINE_GLACIER again (:2918-2921)
                        int isnow_old = isnow;
                        for (int j = isnow_old + 1; j <= 0; ++j) {
                            if (snice[SN(j)] <= 0.1f) {
                                if (j != 0) {
                                    snliq[SN(j + 1)] = AD(snliq[SN(j + 1)],
                                                          snliq[SN(j)]);
                                    snice[SN(j + 1)] = AD(snice[SN(j + 1)],
                                                          snice[SN(j)]);
                                } else {
                                    if (isnow_old < -1) {
                                        snliq[SN(j - 1)] =
                                            AD(snliq[SN(j - 1)],
                                               snliq[SN(j)]);
                                        snice[SN(j - 1)] =
                                            AD(snice[SN(j - 1)],
                                               snice[SN(j)]);
                                    } else {
                                        ponding1 = AD(ponding1,
                                                      snliq[SN(j)]);
                                        sneqv = snice[SN(j)];
                                        snowh = dzsnso[GS(j)];
                                        snliq[SN(j)] = 0.0f;
                                        snice[SN(j)] = 0.0f;
                                        dzsnso[GS(j)] = 0.0f;
                                    }
                                }
                                if (j > isnow + 1 && isnow < -1) {
                                    for (int i2 = j; i2 >= isnow + 2; --i2) {
                                        stc[GS(i2)] = stc[GS(i2 - 1)];
                                        snliq[SN(i2)] = snliq[SN(i2 - 1)];
                                        snice[SN(i2)] = snice[SN(i2 - 1)];
                                        dzsnso[GS(i2)] = dzsnso[GS(i2 - 1)];
                                    }
                                }
                                isnow = isnow + 1;
                            }
                        }
                        if (sice[0] < 0.0f) {
                            sh2o[0] = AD(sh2o[0], sice[0]);
                            sice[0] = 0.0f;
                        }
                        if (isnow != 0) {
                            sneqv = 0.0f;
                            snowh = 0.0f;
                            float zwice = 0.0f, zwliq = 0.0f;
                            for (int j = isnow + 1; j <= 0; ++j) {
                                sneqv = AD(AD(sneqv, snice[SN(j)]),
                                           snliq[SN(j)]);
                                snowh = AD(snowh, dzsnso[GS(j)]);
                                zwice = AD(zwice, snice[SN(j)]);
                                zwliq = AD(zwliq, snliq[SN(j)]);
                            }
                            if (snowh < 0.05f && isnow < 0) {
                                isnow = 0;
                                sneqv = zwice;
                                ponding2 = AD(ponding2, zwliq);
                                if (sneqv <= 0.0f) snowh = 0.0f;
                            }
                            if (isnow < -1) {
                                int isnow_old2 = isnow;
                                int mssi = 1;
                                const float dzmin[3] = {0.045f, 0.05f, 0.2f};
                                for (int i2 = isnow_old2 + 1; i2 <= 0;
                                     ++i2) {
                                    if (dzsnso[GS(i2)] < dzmin[mssi - 1]) {
                                        int neibor;
                                        if (i2 == isnow + 1) neibor = i2 + 1;
                                        else if (i2 == 0) neibor = i2 - 1;
                                        else {
                                            neibor = i2 + 1;
                                            if (AD(dzsnso[GS(i2 - 1)],
                                                   dzsnso[GS(i2)])
                                                    < AD(dzsnso[GS(i2 + 1)],
                                                         dzsnso[GS(i2)]))
                                                neibor = i2 - 1;
                                        }
                                        int jj, ll;
                                        if (neibor > i2) {
                                            jj = neibor; ll = i2;
                                        } else {
                                            jj = i2; ll = neibor;
                                        }
                                        ng_combo(&dzsnso[GS(jj)],
                                                 &snliq[SN(jj)],
                                                 &snice[SN(jj)],
                                                 &stc[GS(jj)],
                                                 dzsnso[GS(ll)],
                                                 snliq[SN(ll)],
                                                 snice[SN(ll)],
                                                 stc[GS(ll)]);
                                        if (jj - 1 > isnow + 1) {
                                            for (int k2 = jj - 1;
                                                 k2 >= isnow + 2; --k2) {
                                                stc[GS(k2)] =
                                                    stc[GS(k2 - 1)];
                                                snice[SN(k2)] =
                                                    snice[SN(k2 - 1)];
                                                snliq[SN(k2)] =
                                                    snliq[SN(k2 - 1)];
                                                dzsnso[GS(k2)] =
                                                    dzsnso[GS(k2 - 1)];
                                            }
                                        }
                                        isnow = isnow + 1;
                                        if (isnow >= -1) break;
                                    } else {
                                        mssi = mssi + 1;
                                    }
                                }
                            }
                        }
                    }
                    if (isnow < 0) {                               // :2924-2927
                        snliq[SN(isnow + 1)] = AD(snliq[SN(isnow + 1)],
                                                  MU(qrain, dt));
                        snliq[SN(isnow + 1)] =
                            ng_max(0.0f, snliq[SN(isnow + 1)]);
                    }
                }
                // porosity / percolation :2935-2965
                float vol_ice[3], epore2[3], vol_liq[3];
                for (int k = 0; k < 3; ++k) {
                    vol_ice[k] = 0.0f; epore2[k] = 0.0f; vol_liq[k] = 0.0f;
                }
                for (int j = -NMP_NSNOW + 1; j <= 0; ++j) {
                    if (j >= isnow + 1) {
                        vol_ice[SN(j)] = ng_min(1.0f,
                            DV(snice[SN(j)], MU(dzsnso[GS(j)], NMP_DENICE)));
                        epore2[SN(j)] = SU(1.0f, vol_ice[SN(j)]);
                        vol_liq[SN(j)] = ng_min(epore2[SN(j)],
                            DV(snliq[SN(j)], MU(dzsnso[GS(j)], NMP_DENH2O)));
                    }
                }
                float qin = 0.0f, qout = 0.0f;
                for (int j = -NMP_NSNOW + 1; j <= 0; ++j) {
                    if (j >= isnow + 1) {
                        snliq[SN(j)] = AD(snliq[SN(j)], qin);
                        if (j <= -1) {
                            if (epore2[SN(j)] < 0.05f
                                    || epore2[SN(j + 1)] < 0.05f) {
                                qout = 0.0f;
                            } else {
                                qout = ng_max(0.0f,
                                    MU(SU(vol_liq[SN(j)],
                                          MU(NG_SSI, epore2[SN(j)])),
                                       dzsnso[GS(j)]));
                                qout = ng_min(qout,
                                    MU(SU(SU(1.0f, vol_ice[SN(j + 1)]),
                                          vol_liq[SN(j + 1)]),
                                       dzsnso[GS(j + 1)]));
                            }
                        } else {
                            qout = ng_max(0.0f,
                                MU(SU(vol_liq[SN(j)],
                                      MU(NG_SSI, epore2[SN(j)])),
                                   dzsnso[GS(j)]));
                        }
                        qout = MU(qout, 1000.0f);
                        snliq[SN(j)] = SU(snliq[SN(j)], qout);
                        qin = qout;
                    }
                }
                qsnbot = DV(qout, dt);                             // :2969
            }

            if (sneqv > 5000.0f) {                                 // :2263-2269
                float bdsnow = DV(snice[SN(0)], dzsnso[GS(0)]);
                snoflow = SU(sneqv, 5000.0f);
                snice[SN(0)] = SU(snice[SN(0)], snoflow);
                dzsnso[GS(0)] = SU(dzsnso[GS(0)], DV(snoflow, bdsnow));
                snoflow = DV(snoflow, dt);
            }
            if (isnow != 0) {                                      // :2273-2278
                sneqv = 0.0f;
                for (int iz = isnow + 1; iz <= 0; ++iz)
                    sneqv = AD(AD(sneqv, snice[SN(iz)]), snliq[SN(iz)]);
            }
            for (int iz = isnow + 1; iz <= 0; ++iz)                // :2282-2298
                dzsnso[GS(iz)] = SU(0.0f, dzsnso[GS(iz)]);
            dzsnso[GS(1)] = zsoil[0];
            for (int iz = 2; iz <= NMP_NSOIL; ++iz)
                dzsnso[GS(iz)] = SU(zsoil[iz - 1], zsoil[iz - 2]);
            zsnso[GS(isnow + 1)] = dzsnso[GS(isnow + 1)];
            for (int iz = isnow + 2; iz <= NMP_NSOIL; ++iz)
                zsnso[GS(iz)] = AD(zsnso[GS(iz - 1)], dzsnso[GS(iz)]);
            for (int iz = isnow + 1; iz <= NMP_NSOIL; ++iz)
                dzsnso[GS(iz)] = SU(0.0f, dzsnso[GS(iz)]);
        }

        runsrf = DV(AD(AD(ponding, ponding1), ponding2), dt);      // :2135
        if (isnow == 0) runsrf = AD(runsrf, AD(qsnbot, qrain));    // :2137-2141
        else runsrf = AD(runsrf, qsnbot);

        if (GLA == 1) {
        // OPT_GLA == 1 :2147-2154
        float replace = 0.0f;
        for (int ilev = 1; ilev <= NMP_NSOIL; ++ilev) {
            replace = AD(replace, MU(dzsnso[GS(ilev)],
                SU(AD(SU(sice[ilev - 1], sice_save[ilev - 1]),
                      sh2o[ilev - 1]), sh2o_save[ilev - 1])));
        }
        replace = DV(MU(replace, 1000.0f), dt);
        for (int ilev = 0; ilev < 4; ++ilev)
            sice[ilev] = ng_min(1.0f, sice_save[ilev]);
        for (int ilev = 0; ilev < 4; ++ilev)                       // :2158
            sh2o[ilev] = SU(1.0f, sice[ilev]);
        runsub = AD(snoflow, replace);                             // :2163-2164
        } else {
            for (int ilev = 0; ilev < 4; ++ilev) {
                sice[ilev] = 1.0f; sh2o[ilev] = SU(1.0f, sice[ilev]);
            }
            runsub = snoflow;
            qvap = qsnsub; qdew = qsnfro;
        }
    }

    if (GLA == 2) { // :269-272, modified snow-only sublimation returned by WATER
        edir = SU(qvap, qdew);
        fgev = MU(edir, lathea);
    }

    // ---- ERROR_GLACIER :2974-3048 ---------------------------------------
    {
        float errsw = SU(swdown, AD(fsa, fsr));                    // :3008
        if (errsw > 0.01f && err == 0) err = 2;
        float erreng = SU(sag, AD(AD(AD(fira, fsh), fgev), ssoil));
        if (erreng > 0.01f && err == 0) err = 3;                   // :3018-3025
        float end_wb = sneqv;                                      // :3027-3028
        float errwat = SU(SU(end_wb, beg_wb),
                          MU(SU(SU(SU(prcp, edir), runsrf), runsub), dt));
        if (fabsf(errwat) > 0.1f && err == 0) err = 4;             // :3031-3045
    }

    if (snowh <= 1.0e-6f || sneqv <= 1.0e-3f) {                    // :285-288
        snowh = 0.0f;
        sneqv = 0.0f;
    }

    float albedo;
    if (swdown != 0.0f) albedo = DV(fsr, swdown);                  // :290-294
    else albedo = -999.9f;

    // ---- outputs --------------------------------------------------------
    o[0] = tg; o[1] = tauss; o[2] = qsfc; o[3] = qsnow; o[4] = sneqvo_out;
    o[5] = albold; o[6] = cm; o[7] = ch; o[8] = sneqv; o[9] = snowh;
    o[10] = fsa; o[11] = fsr; o[12] = fira; o[13] = fsh; o[14] = fgev;
    o[15] = ssoil; o[16] = trad; o[17] = edir; o[18] = runsrf;
    o[19] = runsub; o[20] = sag; o[21] = albedo; o[22] = qsnbot;
    o[23] = ponding; o[24] = ponding1; o[25] = ponding2; o[26] = t2mb;
    o[27] = q2b; o[28] = emissi; o[29] = fpice; o[30] = ehb2;
    o[31] = qmelt; o[32] = eflxb;
    for (int k = 0; k < 4; ++k) o[33 + k] = smc[k];
    for (int k = 0; k < 4; ++k) o[37 + k] = sh2o[k];
    for (int k = 0; k < 7; ++k) o[41 + k] = stc[k];
    for (int k = 0; k < 7; ++k) o[48 + k] = zsnso[k];
    for (int k = 0; k < 7; ++k) o[55 + k] = hcpct[k];
    for (int k = 0; k < 3; ++k) o[62 + k] = snice[k];
    for (int k = 0; k < 3; ++k) o[65 + k] = snliq[k];
    io[2 * tid] = isnow;
    io[2 * tid + 1] = err;
}

// Native simple-glacier/STC=1-or-3 variants. Packed row and output ABI unchanged.
extern "C" __global__ void noahmp_glacier_column_gla2(
    const float *in, const int *ii, const float *zsoil_row, float dt,
    float *out, int *io, int n)
{
    ng_glacier_column<2, 2>(in, ii, zsoil_row, dt, out, io, n);
}

extern "C" __global__ void noahmp_glacier_column_gla2_bats(
    const float *in, const int *ii, const float *zsoil_row, float dt,
    float *out, int *io, int n)
{
    ng_glacier_column<2, 1>(in, ii, zsoil_row, dt, out, io, n);
}

// Full driver glacier arm, module_sf_noahmpdrv.F:729-796,1026-1150,1223-1400.
// Grid capacity is fixed. The live device mask gates all reads and writes.
extern "C" __global__ void noahmp_glacier_column_masked(
 const float* in,const int* ii,const float* zsoil_row,float dt,float* out,
 int* io,int n,const bool* active,int alb,int gla){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 if(!active[c]){io[2*c]=0;io[2*c+1]=0;return;}
 if(gla==2){
  if(alb==1)ng_glacier_column<2,1,true>(in+(size_t)c*NG_NIN,ii+c,zsoil_row,dt,out+(size_t)c*NG_NOUT,io+2*c,1);
  else ng_glacier_column<2,2,true>(in+(size_t)c*NG_NIN,ii+c,zsoil_row,dt,out+(size_t)c*NG_NOUT,io+2*c,1);
 }else{
  if(alb==1)ng_glacier_column<1,1,true>(in+(size_t)c*NG_NIN,ii+c,zsoil_row,dt,out+(size_t)c*NG_NOUT,io+2*c,1);
  else ng_glacier_column<1,2,true>(in+(size_t)c*NG_NIN,ii+c,zsoil_row,dt,out+(size_t)c*NG_NOUT,io+2*c,1);
 }
}
__device__ void ng_glacier_driver_pack(
const float* temperature,
const float* p_interface,
const float* u,
const float* v,
const float* qv,
const float* dz,
const float* cosz,
const float* swdown,
const float* rainbl,
const float* glw,
const float* tmn,
const float* qsnowxy,
const float* sneqvoxy,
const float* alboldxy,
const float* cmxy,
const float* chxy,
const float* snow,
const float* snowh,
const float* tgxy,
const float* taussxy,
const float* qsfc,
const float* snicexy,
const float* snliqxy,
const float* smois,
const float* sh2o,
const float* zsnsoxy,
const float* tsnoxy,
const float* tslb,
const int* isnow,const bool* active,float dt,int n,float* packed,int c){
 if(!active[c])return;
 float* x=packed;
 x[0]=cosz[c];
 x[1]=temperature[c];
 x[2]=MU(AD(p_interface[c+n],p_interface[c]),0.5f);
 x[3]=u[c];
 x[4]=v[c];
 x[5]=DV(qv[c],AD(1.0f,qv[c]));
 x[6]=swdown[c];
 x[7]=DV(rainbl[c],dt);
 x[8]=glw[c];
 x[9]=ng_min(tmn[c],263.15f);
 x[10]=MU(0.5f,dz[c]);
 x[11]=qsnowxy[c];
 x[12]=sneqvoxy[c];
 x[13]=alboldxy[c];
 x[14]=cmxy[c];
 x[15]=chxy[c];
 x[16]=snow[c];
 x[17]=snowh[c];
 x[18]=tgxy[c];
 x[19]=taussxy[c];
 x[20]=qsfc[c];
 for(int j=0;j<3;j++){
  float ice=snicexy[j*n+c],liq=snliqxy[j*n+c];
  x[21+j]=j>=isnow[c]+3?DV(ice,AD(ice,liq)):0.0f;
  x[46+j]=ice;x[49+j]=liq;x[39+j]=tsnoxy[j*n+c];
 }
 for(int j=0;j<4;j++){x[24+j]=smois[j*n+c];x[28+j]=sh2o[j*n+c];x[42+j]=tslb[j*n+c];}
 for(int j=0;j<7;j++)x[32+j]=zsnsoxy[j*n+c];
}
__device__ void ng_glacier_driver_write(
const float* out,const int* io,const bool* active,const float* rainbl,
float dt,int n,int has_drain,int* isnowxy,
float* tsk,
float* hfx,
float* grdflx,
float* snow,
float* snowh,
float* emiss,
float* qsfc,
float* tgxy,
float* cmxy,
float* chxy,
float* sneqvoxy,
float* alboldxy,
float* qsnowxy,
float* taussxy,
float* t2mbxy,
float* tradxy,
float* edirxy,
float* fsaxy,
float* firaxy,
float* chbxy,
float* fpicexy,
float* qsnbotxy,
float* qmeltxy,
float* eflxbxy,
float* qfx,
float* lh,
float* smstav,
float* smstot,
float* canwat,
float* canliqxy,
float* canicexy,
float* fvegxy,
float* ecanxy,
float* etranxy,
float* rs,
float* canhsxy,
float* tvxy,
float* eahxy,
float* tahxy,
float* fwetxy,
float* wslakexy,
float* waxy,
float* lai,
float* xsaixy,
float* t2mvxy,
float* rssunxy,
float* rsshaxy,
float* chvxy,
float* sfcrunoff,
float* udrunoff,
float* albedo,
float* snowc,
float* smois,
float* sh2o,
float* tslb,
float* tsnoxy,
float* zsnsoxy,
float* snicexy,
float* snliqxy,
float* acsnow,
float* acsnom,
float* pondingxy,
float* z0,
float* znt,
float* q2mvxy,
float* q2mbxy,
float* runsfxy,
float* runsbxy,
float* soldrain,
float* soilenergy,
float* snowenergy,int c){
 if(!active[c])return;
 const float* o=out;const float undef=-1.0e36f;
 tsk[c]=o[16];
 hfx[c]=o[13];
 grdflx[c]=o[15];
 snow[c]=o[8];
 snowh[c]=o[9];
 emiss[c]=o[28];
 qsfc[c]=o[2];
 tgxy[c]=o[0];
 cmxy[c]=o[6];
 chxy[c]=o[7];
 sneqvoxy[c]=o[4];
 alboldxy[c]=o[5];
 qsnowxy[c]=o[3];
 taussxy[c]=o[1];
 t2mbxy[c]=o[26];
 tradxy[c]=o[16];
 edirxy[c]=o[17];
 fsaxy[c]=o[10];
 firaxy[c]=o[12];
 chbxy[c]=o[7];
 fpicexy[c]=o[29];
 qsnbotxy[c]=o[22];
 qmeltxy[c]=o[31];
 eflxbxy[c]=o[32];
 qfx[c]=o[17];
 lh[c]=o[14];
 smstav[c]=0.0f;
 smstot[c]=0.0f;
 canwat[c]=0.0f;
 canliqxy[c]=0.0f;
 canicexy[c]=0.0f;
 fvegxy[c]=0.0f;
 ecanxy[c]=0.0f;
 etranxy[c]=0.0f;
 rs[c]=0.0f;
 canhsxy[c]=0.0f;
 tvxy[c]=undef;
 eahxy[c]=undef;
 tahxy[c]=undef;
 fwetxy[c]=undef;
 wslakexy[c]=undef;
 waxy[c]=undef;
 lai[c]=undef;
 xsaixy[c]=undef;
 t2mvxy[c]=undef;
 rssunxy[c]=undef;
 rsshaxy[c]=undef;
 chvxy[c]=undef;

 float runsf=MU(o[18],dt),runsb=MU(o[19],dt);
 sfcrunoff[c]=AD(sfcrunoff[c],runsf);udrunoff[c]=AD(udrunoff[c],runsb);
 if(o[21]>-999.0f)albedo[c]=o[21];snowc[c]=1.0f;
 for(int j=0;j<4;j++){smois[j*n+c]=o[33+j];sh2o[j*n+c]=o[37+j];tslb[j*n+c]=o[44+j];}
 for(int j=0;j<3;j++){tsnoxy[j*n+c]=o[41+j];snicexy[j*n+c]=o[62+j];snliqxy[j*n+c]=o[65+j];}
 for(int j=0;j<7;j++)zsnsoxy[j*n+c]=o[48+j];
 int snow_index=io[0];isnowxy[c]=snow_index;
 acsnow[c]=AD(acsnow[c],MU(rainbl[c],o[29]));
 acsnom[c]=AD(AD(AD(AD(acsnom[c],MU(o[31],dt)),o[23]),o[24]),o[25]);
 pondingxy[c]=AD(AD(o[23],o[24]),o[25]);z0[c]=0.002f;znt[c]=0.002f;
 q2mvxy[c]=DV(undef,SU(1.0f,undef));q2mbxy[c]=DV(o[27],SU(1.0f,o[27]));
 runsfxy[c]=runsf;runsbxy[c]=runsb;if(has_drain)soldrain[c]=0.0f;
 float soil=0.0f,snowe=0.0f;
 for(int j=-2;j<=4;j++){
  int k=j+2;float above=k>0?o[48+k-1]:0.0f;
  float thick=j==snow_index+1?-o[48+k]:SU(above,o[48+k]);
  float term=MU(MU(MU(thick,o[55+k]),SU(o[41+k],273.16f)),0.001f);
  float contrib=j>=snow_index+1?term:0.0f;
  if(j>=1)soil=AD(soil,contrib);else snowe=AD(snowe,contrib);
 }
 soilenergy[c]=soil;snowenergy[c]=snowe;
}

// One launch per grid. Stack rows belong only to an active glacier thread.
extern "C" __global__ void noahmp_glacier_driver_resident(
float* temperature,
float* p_interface,
float* u,
float* v,
float* qv,
float* dz,
float* cosz,
float* swdown,
float* rainbl,
float* glw,
float* tmn,
float* qsnowxy,
float* sneqvoxy,
float* alboldxy,
float* cmxy,
float* chxy,
float* snow,
float* snowh,
float* tgxy,
float* taussxy,
float* qsfc,
float* snicexy,
float* snliqxy,
float* smois,
float* sh2o,
float* zsnsoxy,
float* tsnoxy,
float* tslb,
float* tsk,
float* hfx,
float* grdflx,
float* emiss,
float* t2mbxy,
float* tradxy,
float* edirxy,
float* fsaxy,
float* firaxy,
float* chbxy,
float* fpicexy,
float* qsnbotxy,
float* qmeltxy,
float* eflxbxy,
float* qfx,
float* lh,
float* smstav,
float* smstot,
float* canwat,
float* canliqxy,
float* canicexy,
float* fvegxy,
float* ecanxy,
float* etranxy,
float* rs,
float* canhsxy,
float* tvxy,
float* eahxy,
float* tahxy,
float* fwetxy,
float* wslakexy,
float* waxy,
float* lai,
float* xsaixy,
float* t2mvxy,
float* rssunxy,
float* rsshaxy,
float* chvxy,
float* sfcrunoff,
float* udrunoff,
float* albedo,
float* snowc,
float* acsnow,
float* acsnom,
float* pondingxy,
float* z0,
float* znt,
float* q2mvxy,
float* q2mbxy,
float* runsfxy,
float* runsbxy,
float* soldrain,
float* soilenergy,
float* snowenergy,
int* isnowxy,const bool* active,const float* zsoil,float dt,int n,
int has_drain,int alb,int gla,int stc_option,int tbot_option,
float* sagxy,
float* irgxy,
float* shgxy,
float* evgxy,
float* ghvxy,
float* pahgxy,
float* irbxy,
float* shbxy,
float* evbxy,
float* ghbxy,
float* pahbxy,
float* savxy,
float* ircxy,
float* shcxy,
float* evcxy,
float* trxy,
float* pahvxy,
float* qintsxy,
float* qintrxy,
float* qdripsxy,
float* qdriprxy,
float* qthrosxy,
float* qthrorxy,
float* qsnsubxy,
float* qsnfroxy,
float* qsubcxy,
float* qfrocxy,
float* qevacxy,
float* qdewcxy,
float* qfrzcxy,
float* qmeltcxy,
float* rainlsm,
float* snowlsm,
int has_output,const float* snow_optics,const float* solar,float* grains,int* errors){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 errors[c]=0;if(!active[c])return;
 float packed[NG_NIN],out[NG_NOUT];int io[2];
 ng_glacier_driver_pack(temperature,p_interface,u,v,qv,dz,cosz,swdown,rainbl,glw,tmn,qsnowxy,sneqvoxy,alboldxy,cmxy,chxy,snow,snowh,tgxy,taussxy,qsfc,snicexy,snliqxy,smois,sh2o,zsnsoxy,tsnoxy,tslb,isnowxy,active,dt,n,packed,c);
 if(gla==2){
  if(alb==3)ng_glacier_column<2,3,true>(packed,isnowxy+c,zsoil,dt,out,io,1,stc_option,tbot_option,snow_optics+(size_t)c*6,solar+(size_t)c*7,grains+(size_t)c*21);
  else if(alb==1)ng_glacier_column<2,1,true>(packed,isnowxy+c,zsoil,dt,out,io,1,stc_option,tbot_option);
  else ng_glacier_column<2,2,true>(packed,isnowxy+c,zsoil,dt,out,io,1,stc_option,tbot_option);
 }else{
  if(alb==3)ng_glacier_column<1,3,true>(packed,isnowxy+c,zsoil,dt,out,io,1,stc_option,tbot_option,snow_optics+(size_t)c*6,solar+(size_t)c*7,grains+(size_t)c*21);
  else if(alb==1)ng_glacier_column<1,1,true>(packed,isnowxy+c,zsoil,dt,out,io,1,stc_option,tbot_option);
  else ng_glacier_column<1,2,true>(packed,isnowxy+c,zsoil,dt,out,io,1,stc_option,tbot_option);
 }
 ng_glacier_driver_write(out,io,active,rainbl,dt,n,has_drain,isnowxy,
tsk,hfx,grdflx,snow,snowh,emiss,qsfc,tgxy,cmxy,chxy,sneqvoxy,alboldxy,qsnowxy,taussxy,t2mbxy,tradxy,edirxy,fsaxy,firaxy,chbxy,fpicexy,qsnbotxy,qmeltxy,eflxbxy,qfx,lh,smstav,smstot,canwat,canliqxy,canicexy,fvegxy,ecanxy,etranxy,rs,canhsxy,tvxy,eahxy,tahxy,fwetxy,wslakexy,waxy,lai,xsaixy,t2mvxy,rssunxy,rsshaxy,chvxy,sfcrunoff,udrunoff,albedo,snowc,smois,sh2o,tslb,tsnoxy,zsnsoxy,snicexy,snliqxy,acsnow,acsnom,pondingxy,z0,znt,q2mvxy,q2mbxy,runsfxy,runsbxy,soldrain,soilenergy,snowenergy,c);
 if(has_output){
  // Complete glacier component outputs: native driver :1100-1150.
  evcxy[c]=0.0f;
  evgxy[c]=0.0f;
  ghvxy[c]=0.0f;
  ircxy[c]=0.0f;
  irgxy[c]=0.0f;
  pahbxy[c]=0.0f;
  pahgxy[c]=0.0f;
  pahvxy[c]=0.0f;
  qdewcxy[c]=0.0f;
  qdriprxy[c]=0.0f;
  qdripsxy[c]=0.0f;
  qevacxy[c]=0.0f;
  qfrocxy[c]=0.0f;
  qfrzcxy[c]=0.0f;
  qintrxy[c]=0.0f;
  qintsxy[c]=0.0f;
  qmeltcxy[c]=0.0f;
  qsubcxy[c]=0.0f;
  savxy[c]=0.0f;
  shcxy[c]=0.0f;
  shgxy[c]=0.0f;
  trxy[c]=0.0f;
  sagxy[c]=out[20];irbxy[c]=out[12];shbxy[c]=out[13];
  evbxy[c]=out[14];ghbxy[c]=out[15];
  qthrosxy[c]=MU(packed[7],out[29]);
  qthrorxy[c]=MU(packed[7],SU(1.0f,out[29]));
  qsnsubxy[c]=ng_max(out[17],0.0f);
  qsnfroxy[c]=fabsf(ng_min(out[17],0.0f));
  rainlsm[c]=qthrorxy[c];snowlsm[c]=qthrosxy[c];
 }
 errors[c]=io[1];
}

// Complete selected glacier leaf, same pack ABI, explicit thermal boundaries.
extern "C" __global__ void noahmp_glacier_column_options(
 const float* in,const int* ii,const float* zsoil,float dt,float* out,int* io,
 int n,int alb,int gla,int stc,int tbot){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
 if(gla==2){
  if(alb==1)ng_glacier_column<2,1,true>(in+(size_t)c*NG_NIN,ii+c,zsoil,dt,out+(size_t)c*NG_NOUT,io+2*c,1,stc,tbot);
  else ng_glacier_column<2,2,true>(in+(size_t)c*NG_NIN,ii+c,zsoil,dt,out+(size_t)c*NG_NOUT,io+2*c,1,stc,tbot);
 }else{
  if(alb==1)ng_glacier_column<1,1,true>(in+(size_t)c*NG_NIN,ii+c,zsoil,dt,out+(size_t)c*NG_NOUT,io+2*c,1,stc,tbot);
  else ng_glacier_column<1,2,true>(in+(size_t)c*NG_NIN,ii+c,zsoil,dt,out+(size_t)c*NG_NOUT,io+2*c,1,stc,tbot);
 }
}
