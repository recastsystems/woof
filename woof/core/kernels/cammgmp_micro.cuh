// cammgmp_micro.cuh -- MG1 two-moment stratiform microphysics of the CAMMGMP
// port: mmicro_pcond (WRF v4.6.1 phys/module_cam_mp_cldwat2m_micro.F
// :383-3552) for one column, sub_column = .false., liu_in = .true.
// ArWen's own work, Apache-2.0.
//
// Transcribed in the Fortran's operation order (cammgmp_support.cuh states
// the -O2 lowering rules).  Everything that reaches a CAMMGMP output is
// computed; the history-only diagnostics (reflectivity, rercld, rflx/sflx,
// the *o process-rate outputs, nfice, ncai/ncal, effc_fn, deffi, rate1ord,
// reff_rain/snow, dsout) are not, since none of them is read again.
#ifndef CAMMGMP_MICRO_CUH
#define CAMMGMP_MICRO_CUH

struct MgMicroOut {
    V tlat, qvlat, qctend, qitend, nctend, nitend;   // tendencies
    V effc, effi, qrout, qsout, nrout, nsout, vtrmc;
    R8 prect, preci;
};

__device__ void mg_mmicro_pcond(int pver, R8 deltatin, V tn, V qn, V qc_in,
                                V qi_in, V nc_in, V ni_in, V p, V pdel,
                                V cldn, V liqcldf, V icecldf, V naai,
                                V npccnin, V rndst3, V nacon3, MgMicroOut& o,
                                Ws& ws, int* bad MG_SAT_DECL) {
    WsMark mark(ws);
    // inout copies (mmicro_pcond updates qc/qi/nc/ni during the substeps
    // and restores them before returning, so the caller's are untouched)
    V qc = ws.r8(pver), qi = ws.r8(pver), nc = ws.r8(pver), ni = ws.r8(pver);
    V t1 = ws.r8(pver), q1 = ws.r8(pver), qc1 = ws.r8(pver), qi1 = ws.r8(pver);
    V nc1 = ws.r8(pver), ni1 = ws.r8(pver);
    V tlat1 = ws.r8(pver), qvlat1 = ws.r8(pver), qctend1 = ws.r8(pver);
    V qitend1 = ws.r8(pver), nctend1 = ws.r8(pver), nitend1 = ws.r8(pver);
    V q = ws.r8(pver), t = ws.r8(pver), rho = ws.r8(pver), dv = ws.r8(pver);
    V mu = ws.r8(pver), sc = ws.r8(pver), rhof = ws.r8(pver);
    V arn = ws.r8(pver), asn = ws.r8(pver), acn = ws.r8(pver), ain = ws.r8(pver);
    V dz = ws.r8(pver), cldmax = ws.r8(pver), cldm = ws.r8(pver);
    V icldm = ws.r8(pver), lcldm = ws.r8(pver), cmei = ws.r8(pver);
    V berg = ws.r8(pver), qcic = ws.r8(pver), qiic = ws.r8(pver);
    V qniic = ws.r8(pver), qric = ws.r8(pver), ncic = ws.r8(pver);
    V niic = ws.r8(pver), nsic = ws.r8(pver), nric = ws.r8(pver);
    V esl = ws.r8(pver), esi = ws.r8(pver), relhum = ws.r8(pver);
    V dum2i = ws.r8(pver);
    V qnitend = ws.r8(pver), nstend = ws.r8(pver), qrtend = ws.r8(pver);
    V nrtend = ws.r8(pver);
    V dumc = ws.r8(pver), dumnc = ws.r8(pver), dumi = ws.r8(pver), dumni = ws.r8(pver);
    // per-level process arrays (dimension pver in the Fortran)
    V lami = ws.r8(pver), n0i = ws.r8(pver), lamc = ws.r8(pver);
    V lams = ws.r8(pver), n0s = ws.r8(pver), lamr = ws.r8(pver), n0r = ws.r8(pver);
    V cdist1 = ws.r8(pver), pgam = ws.r8(pver), nnuccd = ws.r8(pver);
    V mnuccd = ws.r8(pver), npccn = ws.r8(pver), mnuccc = ws.r8(pver);
    V nnuccc = ws.r8(pver), mnucct = ws.r8(pver), nnucct = ws.r8(pver);
    V msacwi = ws.r8(pver), nsacwi = ws.r8(pver), prc = ws.r8(pver);
    V nprc = ws.r8(pver), nprc1 = ws.r8(pver), nsagg = ws.r8(pver);
    V psacws = ws.r8(pver), npsacws = ws.r8(pver), uns = ws.r8(pver);
    V ums = ws.r8(pver), unr = ws.r8(pver), umr = ws.r8(pver);
    V pracs = ws.r8(pver), npracs = ws.r8(pver), mnuccr = ws.r8(pver);
    V nnuccr = ws.r8(pver), pra = ws.r8(pver), npra = ws.r8(pver);
    V nragg = ws.r8(pver), prci = ws.r8(pver), nprci = ws.r8(pver);
    V prai = ws.r8(pver), nprai = ws.r8(pver), pre = ws.r8(pver);
    V prds = ws.r8(pver), bergs = ws.r8(pver), nsubi = ws.r8(pver);
    V nsubc = ws.r8(pver), nsubs = ws.r8(pver), nsubr = ws.r8(pver);
    V fi = ws.r8(pver), fni = ws.r8(pver), fc = ws.r8(pver), fnc = ws.r8(pver);
    V falouti = ws.r8(pver), faloutni = ws.r8(pver), faloutc = ws.r8(pver);
    V faloutnc = ws.r8(pver);

    V tlat = o.tlat, qvlat = o.qvlat, qctend = o.qctend, qitend = o.qitend;
    V nctend = o.nctend, nitend = o.nitend, effc = o.effc, effi = o.effi;
    V qrout = o.qrout, qsout = o.qsout, nrout = o.nrout, nsout = o.nsout;
    V vtrmc = o.vtrmc;

    const R8 g(MG_G), r(MG_R), rv(MG_RV), cpp(MG_CPP), rhow(MG_RHOW);
    const R8 xxlv(MG_XXLV), xlf(MG_XLF), xxls(MG_XXLS), rhosn(MG_RHOSN);
    const R8 rhoi(MG_RHOI), as(MG_AS), bs(MG_BS), ai(MG_AI), bi(MG_BI);
    const R8 ar(MG_AR), br(MG_BR), ci(MG_CI), di(MG_DI), cs(MG_CS), ds(MG_DS);
    const R8 ac(MG_AC), bc(MG_BC);
    const R8 f1s(MG_F1S), f2s(MG_F2S), Eii(MG_EII), ecr(MG_ECR), f1r(MG_F1R);
    const R8 f2r(MG_F2R), dcs(MG_DCS), qsmall(MG_QSMALL), bimm(MG_BIMM);
    const R8 aimm(MG_AIMM), rhosu(MG_RHOSU), mi0(MG_MI0), qcvar(MG_QCVAR);
    const R8 pi(MG_PI);
    const R8 cons1(MG_CONS1), cons2(MG_CONS2), cons3(MG_CONS3), cons4(MG_CONS4);
    const R8 cons5(MG_CONS5), cons6(MG_CONS6), cons7(MG_CONS7), cons8(MG_CONS8);
    const R8 cons9(MG_CONS9), cons10(MG_CONS10), cons11(MG_CONS11);
    const R8 cons12(MG_CONS12), cons13(MG_CONS13), cons14(MG_CONS14);
    const R8 cons16(MG_CONS16), cons17(MG_CONS17);
    const R8 cons18(MG_CONS18), cons19(MG_CONS19), cons20(MG_CONS20);
    const R8 cons22(MG_CONS22), cons23(MG_CONS23), cons24(MG_CONS24);
    const R8 cons25(MG_CONS25), cons27(MG_CONS27), cons28(MG_CONS28);
    const R8 lammini(MG_LAMMINI), lammaxi(MG_LAMMAXI), lamminr(MG_LAMMINR);
    const R8 lammaxr(MG_LAMMAXR), lammins(MG_LAMMINS), lammaxs(MG_LAMMAXS);
    const R8 tmelt(PC_TMELT), epsqs(WV_EPSQS);
    const R8 cdnl = R8(0.e6);
    const R8 rhmini = R8(0.80);
    const R8 ONE(1.0), ZERO(0.0);
    // Exponents written as constant quotients in the Fortran, folded.
    const R8 K13 = R8(0x1.5555555555555p-2);   // 1._r8/3._r8
    const R8 K23 = R8(0x1.5555555555555p-1);   // 2._r8/3._r8
    const R8 K43 = R8(0x1.5555555555555p+0);   // 4._r8/3._r8

    for (int k = 1; k <= pver; ++k) {
        qc(k) = qc_in(k); qi(k) = qi_in(k); nc(k) = nc_in(k); ni(k) = ni_in(k);
        effc(k) = ZERO; effi(k) = ZERO; vtrmc(k) = ZERO;
    }

    R8 deltat = deltatin;
    const R8 omsm = R8(0.99999);
    const R8 mincld = R8(0.0001);
    for (int k = 1; k <= pver; ++k) {
        q(k) = qn(k);
        t(k) = tn(k);
    }
    for (int k = 1; k <= pver; ++k) {
        rho(k) = p(k) / (r * t(k));
        dv(k) = R8(8.794E-5) * mg_pow(t(k), R8(1.81)) / p(k);
        mu(k) = R8(1.496E-6) * mg_pow(t(k), R8(1.5)) / (t(k) + R8(120.));
        sc(k) = mu(k) / (rho(k) * dv(k));
        rhof(k) = mg_pow(rhosu / rho(k), SP(0.54));
        arn(k) = ar * rhof(k);
        asn(k) = as * rhof(k);
        acn(k) = ac * rhof(k);
        ain(k) = ai * rhof(k);
        dz(k) = pdel(k) / (rho(k) * g);
    }
    for (int k = 1; k <= pver; ++k) {
        t1(k) = t(k); q1(k) = q(k); qc1(k) = qc(k); qi1(k) = qi(k);
        nc1(k) = nc(k); ni1(k) = ni(k);
        tlat1(k) = ZERO; qvlat1(k) = ZERO; qctend1(k) = ZERO;
        qitend1(k) = ZERO; nctend1(k) = ZERO; nitend1(k) = ZERO;
        qrout(k) = ZERO; qsout(k) = ZERO; nrout(k) = ZERO; nsout(k) = ZERO;
        cldmax(k) = mincld;
        dum2i(k) = ZERO;
    }
    R8 prect1 = ZERO, preci1 = ZERO;

    // ---- humidity, saturation, Bergeron and deposition (F:967-1278) ----
    R8 qvi, qvl, dqsidt, abi, epsi, prd, bergtsf = ZERO, dum, dum1, dum2;
    R8 qvs, dqsdt, ab, qclr, epss, epsr;
    for (int k = 1; k <= pver; ++k) {
        R8 es, qs, gammas;
        mg_vqsatd_water(t(k), p(k), es, qs, gammas MG_SAT_POINT_ARG(k));
        esl(k) = MG_SAT(k, t(k), 0);
        esi(k) = MG_SAT(k, t(k), 1);
        if (t(k) > tmelt) esi(k) = esl(k);
        relhum(k) = q(k) / qs;
        cldm(k) = mg_max(cldn(k), mincld);
        icldm(k) = mg_max(icecldf(k), mincld);
        lcldm(k) = mg_max(liqcldf(k), mincld);
        R8 ninew, qinew, dumnnuc;
        if (t(k) < tmelt - R8(5.)) {
            dum2 = naai(k);
            dumnnuc = (dum2 - ni(k) / icldm(k)) / deltat * icldm(k);
            dumnnuc = mg_max(dumnnuc, ZERO);
            ninew = ni(k) + dumnnuc * deltat;
            qinew = qi(k) + dumnnuc * deltat * mi0;
        } else {
            ninew = ni(k);
            qinew = qi(k);
        }
        cmei(k) = ZERO;
        berg(k) = ZERO;
        prd = ZERO;
        if (icldm(k) > ZERO) {
            qiic(k) = qinew / icldm(k);
            niic(k) = ninew / icldm(k);
        } else {
            qiic(k) = ZERO;
            niic(k) = ZERO;
        }
        if (t(k) < SP(273.15)) {
            if (qi(k) > qsmall) {
                bergtsf = ZERO;
                qvi = epsqs * esi(k) / (p(k) - (ONE - epsqs) * esi(k));
                qvl = epsqs * esl(k) / (p(k) - (ONE - epsqs) * esl(k));
                dqsidt = xxls * qvi / (rv * ip2(t(k)));
                abi = ONE + dqsidt * xxls / cpp;
                if (qiic(k) >= qsmall) {
                    lami(k) = mg_pow(cons1 * ci * niic(k) / qiic(k), ONE / di);
                    n0i(k) = niic(k) * lami(k);
                    if (lami(k) < lammini) {
                        lami(k) = lammini;
                        n0i(k) = mg_pow(lami(k), di + ONE) * qiic(k) / (ci * cons1);
                    } else if (lami(k) > lammaxi) {
                        lami(k) = lammaxi;
                        n0i(k) = mg_pow(lami(k), di + ONE) * qiic(k) / (ci * cons1);
                    }
                    epsi = R8(2.) * pi * n0i(k) * rho(k) * dv(k) / (lami(k) * lami(k));
                    if (qc(k) > qsmall) {
                        prd = epsi * (qvl - qvi) / abi;
                    } else {
                        prd = ZERO;
                    }
                    prd = prd * mg_min(icldm(k), lcldm(k));
                    berg(k) = mg_max(ZERO, prd);
                }
                if (berg(k) > ZERO) {
                    bergtsf = mg_max(ZERO, (qc(k) / berg(k)) / deltat);
                    if (bergtsf < ONE) berg(k) = mg_max(ZERO, qc(k) / deltat);
                }
                if (bergtsf < ONE || icldm(k) > lcldm(k)) {
                    if (qiic(k) >= qsmall) {
                        if (qc(k) >= qsmall) {
                            R8 rhin = (R8(1.0) + relhum(k)) / R8(2.);
                            if ((rhin * esl(k) / esi(k)) > ONE) {
                                prd = epsi * (rhin * qvl - qvi) / abi;
                                prd = prd * mg_min(icldm(k), lcldm(k));
                                cmei(k) = cmei(k) + (prd * (ONE - bergtsf));
                            }
                        }
                        if (qc(k) < qsmall || icldm(k) > lcldm(k)) {
                            if (qc(k) < qsmall) dum = ZERO;
                            else dum = lcldm(k);
                            R8 rhin = relhum(k);
                            if ((rhin * esl(k) / esi(k)) > ONE) {
                                prd = epsi * (rhin * qvl - qvi) / abi;
                                prd = prd * mg_max((icldm(k) - dum), ZERO);
                                cmei(k) = cmei(k) + prd;
                            }
                        }
                    }
                }
                if (cmei(k) > R8(0.0) && (relhum(k) * esl(k) / esi(k)) > ONE)
                    cmei(k) = mg_min(cmei(k), (q(k) - qs * esi(k) / esl(k)) / abi / deltat);
            }
        }
        if ((-berg(k)) < -qc(k) / deltat)
            berg(k) = mg_max(qc(k) / deltat, ZERO);
        if ((relhum(k) * esl(k) / esi(k)) < ONE && qiic(k) >= qsmall) {
            qvi = epsqs * esi(k) / (p(k) - (ONE - epsqs) * esi(k));
            qvl = epsqs * esl(k) / (p(k) - (ONE - epsqs) * esl(k));
            dqsidt = xxls * qvi / (rv * ip2(t(k)));
            abi = ONE + dqsidt * xxls / cpp;
            lami(k) = mg_pow(cons1 * ci * niic(k) / qiic(k), ONE / di);
            n0i(k) = niic(k) * lami(k);
            if (lami(k) < lammini) {
                lami(k) = lammini;
                n0i(k) = mg_pow(lami(k), di + ONE) * qiic(k) / (ci * cons1);
            } else if (lami(k) > lammaxi) {
                lami(k) = lammaxi;
                n0i(k) = mg_pow(lami(k), di + ONE) * qiic(k) / (ci * cons1);
            }
            epsi = R8(2.) * pi * n0i(k) * rho(k) * dv(k) / (lami(k) * lami(k));
            prd = epsi * (relhum(k) * qvl - qvi) / abi * icldm(k);
            cmei(k) = mg_min(prd, ZERO);
        }
        if (cmei(k) < -qi(k) / deltat) cmei(k) = -qi(k) / deltat;
        if (cmei(k) < R8(0.0) && (relhum(k) * esl(k) / esi(k)) < ONE)
            cmei(k) = mg_min(ZERO, mg_max(cmei(k), (q(k) - qs * esi(k) / esl(k)) / abi / deltat));
        cmei(k) = cmei(k) * omsm;
        if (t(k) < (tmelt - R8(5.))) {
            dum2i(k) = naai(k);
        } else {
            dum2i(k) = ZERO;
        }
    }

    int ltrue = 0;
    for (int k = 1; k <= pver; ++k)
        if (qc(k) >= qsmall || qi(k) >= qsmall || cmei(k) >= qsmall) ltrue = 1;

    const int iter = 2;
    deltat = deltat / R8(2.0);            // real(iter)
    const R8 mtime = ONE;
    R8 prect = ZERO, preci = ZERO;

    if (ltrue == 0) {
        for (int k = 1; k <= pver; ++k) {
            tlat(k) = ZERO; qvlat(k) = ZERO; qctend(k) = ZERO; qitend(k) = ZERO;
            qnitend(k) = ZERO; qrtend(k) = ZERO; nctend(k) = ZERO; nitend(k) = ZERO;
            nrtend(k) = ZERO; nstend(k) = ZERO;
            qniic(k) = ZERO; qric(k) = ZERO; nsic(k) = ZERO; nric(k) = ZERO;
        }
        prect = ZERO;
        preci = ZERO;
    } else {
        for (int it = 1; it <= iter; ++it) {
            for (int k = 1; k <= pver; ++k) {
                tlat(k) = ZERO; qvlat(k) = ZERO; qctend(k) = ZERO; qitend(k) = ZERO;
                qnitend(k) = ZERO; qrtend(k) = ZERO; nctend(k) = ZERO; nitend(k) = ZERO;
                nrtend(k) = ZERO; nstend(k) = ZERO;
                qniic(k) = ZERO; qric(k) = ZERO; nsic(k) = ZERO; nric(k) = ZERO;
            }
            R8 qrtot = ZERO, nrtot = ZERO, qstot = ZERO, nstot = ZERO;
            prect = ZERO;
            preci = ZERO;
            R8 nimax, ncmax;
            for (int k = 1; k <= pver; ++k) {
                R8 cwml = qc(k), cwmi = qi(k);
                ums(k) = ZERO; uns(k) = ZERO; umr(k) = ZERO; unr(k) = ZERO;
                if (k == 1) {
                    cldmax(k) = cldm(k);
                } else {
                    if (qric(k - 1) >= qsmall || qniic(k - 1) >= qsmall) {
                        cldmax(k) = mg_max(cldmax(k - 1), cldm(k));
                    } else {
                        cldmax(k) = cldm(k);
                    }
                }
                if (cmei(k) < ZERO && qi(k) > qsmall && cldm(k) > mincld) {
                    nsubi(k) = cmei(k) / qi(k) * ni(k) / cldm(k);
                } else {
                    nsubi(k) = ZERO;
                }
                nsubc(k) = ZERO;
                // ice nucleation
                if (dum2i(k) > ZERO && t(k) < (tmelt - R8(5.))
                    && relhum(k) * esl(k) / esi(k) > rhmini + R8(0.05)) {
                    nnuccd(k) = (dum2i(k) - ni(k) / icldm(k)) / deltat * icldm(k);
                    nnuccd(k) = mg_max(nnuccd(k), ZERO);
                    nimax = dum2i(k) * icldm(k);
                    mnuccd(k) = nnuccd(k) * mi0;
                    cmei(k) = cmei(k) + mnuccd(k) * mtime;
                    qvi = epsqs * esi(k) / (p(k) - (ONE - epsqs) * esi(k));
                    dqsidt = xxls * qvi / (rv * ip2(t(k)));
                    abi = ONE + dqsidt * xxls / cpp;
                    cmei(k) = mg_min(cmei(k), (q(k) - qvi) / abi / deltat);
                    cmei(k) = cmei(k) * omsm;
                } else {
                    nnuccd(k) = ZERO;
                    nimax = ZERO;
                    mnuccd(k) = ZERO;
                }
                // in-cloud values
                qcic(k) = mg_min(cwml / lcldm(k), R8(5.e-3));
                qiic(k) = mg_min(cwmi / icldm(k), R8(5.e-3));
                ncic(k) = mg_max(nc(k) / lcldm(k), ZERO);
                niic(k) = mg_max(ni(k) / icldm(k), ZERO);
                if (qc(k) - berg(k) * deltat < qsmall) {
                    qcic(k) = ZERO;
                    ncic(k) = ZERO;
                    if (qc(k) - berg(k) * deltat < ZERO) berg(k) = qc(k) / deltat * omsm;
                }
                if (qi(k) + (cmei(k) + berg(k)) * deltat < qsmall) {
                    qiic(k) = ZERO;
                    niic(k) = ZERO;
                    if (qi(k) + (cmei(k) + berg(k)) * deltat < ZERO)
                        cmei(k) = (-qi(k) / deltat - berg(k)) * omsm;
                }
                // droplet activation
                if (qcic(k) >= qsmall) {
                    npccn(k) = mg_max(ZERO, npccnin(k));
                    R8 d2l = (nc(k) + npccn(k) * deltat) / lcldm(k);
                    d2l = mg_max(d2l, cdnl / rho(k));
                    ncmax = d2l * lcldm(k);
                } else {
                    npccn(k) = ZERO;
                    ncmax = ZERO;
                }
                // cloud ice size distribution
                if (qiic(k) >= qsmall) {
                    niic(k) = mg_min(niic(k), qiic(k) * R8(1.e20));
                    lami(k) = mg_pow(cons1 * ci * niic(k) / qiic(k), ONE / di);
                    n0i(k) = niic(k) * lami(k);
                    if (lami(k) < lammini) {
                        lami(k) = lammini;
                        n0i(k) = mg_pow(lami(k), di + ONE) * qiic(k) / (ci * cons1);
                        niic(k) = n0i(k) / lami(k);
                    } else if (lami(k) > lammaxi) {
                        lami(k) = lammaxi;
                        n0i(k) = mg_pow(lami(k), di + ONE) * qiic(k) / (ci * cons1);
                        niic(k) = n0i(k) / lami(k);
                    }
                } else {
                    lami(k) = ZERO;
                    n0i(k) = ZERO;
                }
                // cloud droplet size distribution
                if (qcic(k) >= qsmall) {
                    ncic(k) = mg_min(ncic(k), qcic(k) * R8(1.e20));
                    ncic(k) = mg_max(ncic(k), cdnl / rho(k));
                    pgam(k) = R8(0.0005714) * (ncic(k) / R8(1.e6) * rho(k)) + R8(0.2714);
                    pgam(k) = ONE / ip2(pgam(k)) - ONE;
                    pgam(k) = mg_max(pgam(k), R8(2.));
                    pgam(k) = mg_min(pgam(k), R8(15.));
                    lamc(k) = mg_pow(pi / R8(6.) * rhow * ncic(k) * mg_gamma(pgam(k) + R8(4.), bad)
                                     / (qcic(k) * mg_gamma(pgam(k) + ONE, bad)), K13);
                    R8 lammin = (pgam(k) + ONE) / R8(50.e-6);
                    R8 lammax = (pgam(k) + ONE) / R8(2.e-6);
                    if (lamc(k) < lammin) {
                        lamc(k) = lammin;
                        ncic(k) = R8(6.) * ip3(lamc(k)) * qcic(k) * mg_gamma(pgam(k) + ONE, bad)
                                  / (pi * rhow * mg_gamma(pgam(k) + R8(4.), bad));
                    } else if (lamc(k) > lammax) {
                        lamc(k) = lammax;
                        ncic(k) = R8(6.) * ip3(lamc(k)) * qcic(k) * mg_gamma(pgam(k) + ONE, bad)
                                  / (pi * rhow * mg_gamma(pgam(k) + R8(4.), bad));
                    }
                    cdist1(k) = ncic(k) / mg_gamma(pgam(k) + ONE, bad);
                } else {
                    lamc(k) = ZERO;
                    cdist1(k) = ZERO;
                }
                // autoconversion of cloud liquid to rain
                if (qcic(k) >= R8(1.e-8)) {
                    prc(k) = cons2 / (cons3 * cons18) * R8(1350.) * mg_pow(qcic(k), R8(2.47))
                             * mg_pow(ncic(k) / R8(1.e6) * rho(k), R8(-1.79));
                    nprc(k) = prc(k) / cons22;
                    nprc1(k) = prc(k) / (qcic(k) / ncic(k));
                } else {
                    prc(k) = ZERO;
                    nprc(k) = ZERO;
                    nprc1(k) = ZERO;
                }
                dum = R8(0.45);
                dum1 = R8(0.45);
                if (k == 1) {
                    qric(k) = prc(k) * lcldm(k) * dz(k) / cldmax(k) / dum;
                    nric(k) = nprc(k) * lcldm(k) * dz(k) / cldmax(k) / dum;
                } else {
                    if (qric(k - 1) >= qsmall) {
                        dum = umr(k - 1);
                        dum1 = unr(k - 1);
                    }
                    if (qric(k - 1) >= R8(1.e-9) || qniic(k - 1) >= R8(1.e-9)) nprc(k) = ZERO;
                    qric(k) = (rho(k - 1) * umr(k - 1) * qric(k - 1) * cldmax(k - 1)
                               + (rho(k) * dz(k) * ((pra(k - 1) + prc(k)) * lcldm(k)
                                  + (pre(k - 1) - pracs(k - 1) - mnuccr(k - 1)) * cldmax(k))))
                              / (dum * rho(k) * cldmax(k));
                    nric(k) = (rho(k - 1) * unr(k - 1) * nric(k - 1) * cldmax(k - 1)
                               + (rho(k) * dz(k) * (nprc(k) * lcldm(k)
                                  + (nsubr(k - 1) - npracs(k - 1) - nnuccr(k - 1) + nragg(k - 1)) * cldmax(k))))
                              / (dum1 * rho(k) * cldmax(k));
                }
                // autoconversion of cloud ice to snow
                if (t(k) <= R8(273.15) && qiic(k) >= qsmall) {
                    nprci(k) = n0i(k) / (lami(k) * R8(180.)) * mg_exp(-lami(k) * dcs);
                    prci(k) = pi * rhoi * n0i(k) / (R8(6.) * R8(180.))
                              * (cons23 / lami(k) + R8(3.) * cons24 / ip2(lami(k))
                                 + R8(6.) * dcs / ip3(lami(k)) + R8(6.) / ip4(lami(k)))
                              * mg_exp(-lami(k) * dcs);
                } else {
                    prci(k) = ZERO;
                    nprci(k) = ZERO;
                }
                dum = (asn(k) * cons25);
                dum1 = (asn(k) * cons25);
                if (k == 1) {
                    qniic(k) = prci(k) * icldm(k) * dz(k) / cldmax(k) / dum;
                    nsic(k) = nprci(k) * icldm(k) * dz(k) / cldmax(k) / dum;
                } else {
                    if (qniic(k - 1) >= qsmall) {
                        dum = ums(k - 1);
                        dum1 = uns(k - 1);
                    }
                    qniic(k) = (rho(k - 1) * ums(k - 1) * qniic(k - 1) * cldmax(k - 1)
                                + (rho(k) * dz(k) * ((prci(k) + prai(k - 1) + psacws(k - 1) + bergs(k - 1)) * icldm(k)
                                   + (prds(k - 1) + pracs(k - 1) + mnuccr(k - 1)) * cldmax(k))))
                               / (dum * rho(k) * cldmax(k));
                    nsic(k) = (rho(k - 1) * uns(k - 1) * nsic(k - 1) * cldmax(k - 1)
                               + (rho(k) * dz(k) * (nprci(k) * icldm(k)
                                  + (nsubs(k - 1) + nsagg(k - 1) + nnuccr(k - 1)) * cldmax(k))))
                              / (dum1 * rho(k) * cldmax(k));
                }
                if (qniic(k) < qsmall) { qniic(k) = ZERO; nsic(k) = ZERO; }
                if (qric(k) < qsmall) { qric(k) = ZERO; nric(k) = ZERO; }
                nric(k) = mg_max(nric(k), ZERO);
                nsic(k) = mg_max(nsic(k), ZERO);
                // rain size distribution
                if (qric(k) >= qsmall) {
                    lamr(k) = mg_pow(pi * rhow * nric(k) / qric(k), K13);
                    n0r(k) = nric(k) * lamr(k);
                    if (lamr(k) < lamminr) {
                        lamr(k) = lamminr;
                        n0r(k) = ip4(lamr(k)) * qric(k) / (pi * rhow);
                        nric(k) = n0r(k) / lamr(k);
                    } else if (lamr(k) > lammaxr) {
                        lamr(k) = lammaxr;
                        n0r(k) = ip4(lamr(k)) * qric(k) / (pi * rhow);
                        nric(k) = n0r(k) / lamr(k);
                    }
                    unr(k) = mg_min(arn(k) * cons4 / mg_pow(lamr(k), br), R8(9.1) * rhof(k));
                    umr(k) = mg_min(arn(k) * cons5 / (R8(6.) * mg_pow(lamr(k), br)), R8(9.1) * rhof(k));
                } else {
                    lamr(k) = ZERO; n0r(k) = ZERO; umr(k) = ZERO; unr(k) = ZERO;
                }
                // snow size distribution
                if (qniic(k) >= qsmall) {
                    lams(k) = mg_pow(cons6 * cs * nsic(k) / qniic(k), ONE / ds);
                    n0s(k) = nsic(k) * lams(k);
                    if (lams(k) < lammins) {
                        lams(k) = lammins;
                        n0s(k) = mg_pow(lams(k), ds + ONE) * qniic(k) / (cs * cons6);
                        nsic(k) = n0s(k) / lams(k);
                    } else if (lams(k) > lammaxs) {
                        lams(k) = lammaxs;
                        n0s(k) = mg_pow(lams(k), ds + ONE) * qniic(k) / (cs * cons6);
                        nsic(k) = n0s(k) / lams(k);
                    }
                    ums(k) = mg_min(asn(k) * cons8 / (R8(6.) * mg_pow(lams(k), bs)), R8(1.2) * rhof(k));
                    uns(k) = mg_min(asn(k) * cons7 / mg_pow(lams(k), bs), R8(1.2) * rhof(k));
                } else {
                    lams(k) = ZERO; n0s(k) = ZERO; ums(k) = ZERO; uns(k) = ZERO;
                }
                // heterogeneous freezing of cloud water
                if (qcic(k) >= qsmall && t(k) < R8(269.15)) {
                    mnuccc(k) = cons9 / (cons3 * cons19) * pi * pi / R8(36.) * rhow
                                * cdist1(k) * mg_gamma(R8(7.) + pgam(k), bad)
                                * bimm * (mg_exp(aimm * (R8(273.15) - t(k))) - ONE)
                                / ip3(lamc(k)) / ip3(lamc(k));
                    nnuccc(k) = cons10 / (cons3 * qcvar) * pi / R8(6.) * cdist1(k)
                                * mg_gamma(pgam(k) + R8(4.), bad) * bimm
                                * (mg_exp(aimm * (R8(273.15) - t(k))) - ONE) / ip3(lamc(k));
                    R8 tcnt = mg_pow(R8(270.16) - t(k), R8(1.3));
                    R8 viscosity = R8(1.8e-5) * mg_pow(t(k) / R8(298.0), R8(0.85));
                    R8 mfp = R8(2.0) * viscosity
                             / (p(k) * mg_sqrt(R8(8.0) * R8(28.96e-3) / (pi * R8(8.314409) * t(k))));
                    R8 rn1 = R8(MA_RN_DST1), rn2 = R8(MA_RN_DST2);
                    R8 rn3 = rndst3(k), rn4 = R8(MA_RN_DST4);
                    R8 nslip1 = ONE + (mfp / rn1) * (R8(1.257) + (R8(0.4) * mg_exp(-(R8(1.1) * rn1 / mfp))));
                    R8 nslip2 = ONE + (mfp / rn2) * (R8(1.257) + (R8(0.4) * mg_exp(-(R8(1.1) * rn2 / mfp))));
                    R8 nslip3 = ONE + (mfp / rn3) * (R8(1.257) + (R8(0.4) * mg_exp(-(R8(1.1) * rn3 / mfp))));
                    R8 nslip4 = ONE + (mfp / rn4) * (R8(1.257) + (R8(0.4) * mg_exp(-(R8(1.1) * rn4 / mfp))));
                    R8 ndfaer1 = R8(1.381e-23) * t(k) * nslip1 / (R8(6.) * pi * viscosity * rn1);
                    R8 ndfaer2 = R8(1.381e-23) * t(k) * nslip2 / (R8(6.) * pi * viscosity * rn2);
                    R8 ndfaer3 = R8(1.381e-23) * t(k) * nslip3 / (R8(6.) * pi * viscosity * rn3);
                    R8 ndfaer4 = R8(1.381e-23) * t(k) * nslip4 / (R8(6.) * pi * viscosity * rn4);
                    // nacon(1,2,4) are zero (MODAL_AERO_3MODE sets only bin 3)
                    R8 sum = ndfaer1 * (ZERO * tcnt) + ndfaer2 * (ZERO * tcnt)
                             + ndfaer3 * (nacon3(k) * tcnt) + ndfaer4 * (ZERO * tcnt);
                    mnucct(k) = mg_gamma(qcvar + K43, bad) / (cons3 * mg_pow(qcvar, K43))
                                * sum * pi * pi / R8(3.) * rhow
                                * cdist1(k) * mg_gamma(pgam(k) + R8(5.), bad) / ip4(lamc(k));
                    nnucct(k) = mg_gamma(qcvar + K13, bad) / (cons3 * mg_pow(qcvar, K13))
                                * sum * R8(2.) * pi
                                * cdist1(k) * mg_gamma(pgam(k) + R8(2.), bad) / lamc(k);
                    if (nnuccc(k) * lcldm(k) > nnuccd(k)) {
                        dum = (nnuccd(k) / (nnuccc(k) * lcldm(k)));
                        mnuccc(k) = mnuccc(k) * dum;
                        nnuccc(k) = nnuccd(k) / lcldm(k);
                    }
                } else {
                    mnuccc(k) = ZERO; nnuccc(k) = ZERO; mnucct(k) = ZERO; nnucct(k) = ZERO;
                }
                // snow self-aggregation
                if (qniic(k) >= qsmall && t(k) <= R8(273.15)) {
                    nsagg(k) = -(R8(1108.) * asn(k) * Eii
                                 * mg_pow(pi, (ONE - bs) / R8(3.)) * mg_pow(rhosn, (R8(-2.) - bs) / R8(3.))
                                 * mg_pow(rho(k), (R8(2.) + bs) / R8(3.))
                                 * mg_pow(qniic(k), (R8(2.) + bs) / R8(3.))
                                 * mg_pow(nsic(k) * rho(k), (R8(4.) - bs) / R8(3.))
                                 / (R8(4.) * R8(720.) * rho(k)));
                } else {
                    nsagg(k) = ZERO;
                }
                // accretion of cloud droplets onto snow
                if (qniic(k) >= qsmall && t(k) <= tmelt && qcic(k) >= qsmall) {
                    R8 dc0 = (pgam(k) + ONE) / lamc(k);
                    R8 ds0 = ONE / lams(k);
                    dum = dc0 * dc0 * uns(k) * rhow / (R8(9.) * mu(k) * ds0);
                    R8 eci = dum * dum / ((dum + R8(0.4)) * (dum + R8(0.4)));
                    eci = mg_max(eci, ZERO);
                    eci = mg_min(eci, ONE);
                    psacws(k) = pi / R8(4.) * asn(k) * qcic(k) * rho(k) * n0s(k) * eci * cons11
                                / mg_pow(lams(k), bs + R8(3.));
                    npsacws(k) = pi / R8(4.) * asn(k) * ncic(k) * rho(k) * n0s(k) * eci * cons11
                                 / mg_pow(lams(k), bs + R8(3.));
                } else {
                    psacws(k) = ZERO;
                    npsacws(k) = ZERO;
                }
                // Hallet-Mossop
                R8 ni_secp;
                if ((t(k) < R8(270.16)) && (t(k) >= R8(268.16))) {
                    ni_secp = R8(3.5e8) * (R8(270.16) - t(k)) / R8(2.0) * psacws(k);
                    nsacwi(k) = ni_secp;
                    msacwi(k) = mg_min(ni_secp * mi0, psacws(k));
                } else if ((t(k) < R8(268.16)) && (t(k) >= R8(265.16))) {
                    ni_secp = R8(3.5e8) * (t(k) - R8(265.16)) / R8(3.0) * psacws(k);
                    nsacwi(k) = ni_secp;
                    msacwi(k) = mg_min(ni_secp * mi0, psacws(k));
                } else {
                    ni_secp = R8(0.0);
                    nsacwi(k) = R8(0.0);
                    msacwi(k) = R8(0.0);
                }
                psacws(k) = mg_max(R8(0.0), psacws(k) - ni_secp * mi0);
                // accretion of rain by snow
                if (qric(k) >= R8(1.e-8) && qniic(k) >= R8(1.e-8) && t(k) <= R8(273.15)) {
                    pracs(k) = pi * pi * ecr
                               * (mg_pow(ip2(R8(1.2) * umr(k) - R8(0.95) * ums(k))
                                         + R8(0.08) * ums(k) * umr(k), R8(0.5))
                                  * rhow * rho(k) * n0r(k) * n0s(k)
                                  * (R8(5.) / (ip6(lamr(k)) * lams(k))
                                     + R8(2.) / (ip5(lamr(k)) * ip2(lams(k)))
                                     + R8(0.5) / (ip4(lamr(k)) * ip3(lams(k)))));
                    npracs(k) = pi / R8(2.) * rho(k) * ecr
                                * mg_pow(R8(1.7) * ip2(unr(k) - uns(k)) + R8(0.3) * unr(k) * uns(k), R8(0.5))
                                * n0r(k) * n0s(k)
                                * (ONE / (ip3(lamr(k)) * lams(k))
                                   + ONE / (ip2(lamr(k)) * ip2(lams(k)))
                                   + ONE / (lamr(k) * ip3(lams(k))));
                } else {
                    pracs(k) = ZERO;
                    npracs(k) = ZERO;
                }
                // heterogeneous freezing of rain
                if (t(k) < R8(269.15) && qric(k) >= qsmall) {
                    mnuccr(k) = R8(20.) * pi * pi * rhow * nric(k) * bimm
                                * (mg_exp(aimm * (R8(273.15) - t(k))) - ONE) / ip3(lamr(k)) / ip3(lamr(k));
                    nnuccr(k) = pi * nric(k) * bimm
                                * (mg_exp(aimm * (R8(273.15) - t(k))) - ONE) / ip3(lamr(k));
                } else {
                    mnuccr(k) = ZERO;
                    nnuccr(k) = ZERO;
                }
                // accretion of cloud liquid by rain
                if (qric(k) >= qsmall && qcic(k) >= qsmall) {
                    pra(k) = cons12 / (cons3 * cons20) * R8(67.) * mg_pow(qcic(k) * qric(k), R8(1.15));
                    npra(k) = pra(k) / (qcic(k) / ncic(k));
                } else {
                    pra(k) = ZERO;
                    npra(k) = ZERO;
                }
                // self-collection of rain
                if (qric(k) >= qsmall) nragg(k) = R8(-8.) * nric(k) * qric(k) * rho(k);
                else nragg(k) = ZERO;
                // accretion of cloud ice by snow
                if (qniic(k) >= qsmall && qiic(k) >= qsmall && t(k) <= R8(273.15)) {
                    prai(k) = pi / R8(4.) * asn(k) * qiic(k) * rho(k) * n0s(k) * Eii * cons11
                              / mg_pow(lams(k), bs + R8(3.));
                    nprai(k) = pi / R8(4.) * asn(k) * niic(k) * rho(k) * n0s(k) * Eii * cons11
                               / mg_pow(lams(k), bs + R8(3.));
                } else {
                    prai(k) = ZERO;
                    nprai(k) = ZERO;
                }
                // evaporation / sublimation of precipitation
                pre(k) = ZERO;
                prds(k) = ZERO;
                if (qcic(k) + qiic(k) < R8(1.e-6) || cldmax(k) > lcldm(k)) {
                    if (qcic(k) + qiic(k) < R8(1.e-6)) dum = ZERO;
                    else dum = lcldm(k);
                    R8 esn = MG_SAT(k, t(k), 0);
                    R8 qsn = mg_min(epsqs * esn / (p(k) - (ONE - epsqs) * esn), ONE);
                    esl(k) = esn;
                    esi(k) = MG_SAT(k, t(k), 1);
                    if (t(k) > tmelt) esi(k) = esl(k);
                    qclr = (q(k) - dum * qsn) / (ONE - dum);
                    if (qric(k) >= qsmall) {
                        qvs = epsqs * esl(k) / (p(k) - (ONE - epsqs) * esl(k));
                        dqsdt = xxlv * qvs / (rv * ip2(t(k)));
                        ab = ONE + dqsdt * xxlv / cpp;
                        epsr = R8(2.) * pi * n0r(k) * rho(k) * dv(k)
                               * (f1r / (lamr(k) * lamr(k))
                                  + f2r * mg_pow(arn(k) * rho(k) / mu(k), R8(0.5))
                                    * mg_pow(sc(k), K13) * cons13
                                    / (mg_pow(lamr(k), R8(5.) / R8(2.) + br / R8(2.))));
                        pre(k) = epsr * (qclr - qvs) / ab;
                        pre(k) = mg_min(pre(k) * (cldmax(k) - dum), ZERO);
                        pre(k) = pre(k) / cldmax(k);
                    }
                    if (qniic(k) >= qsmall) {
                        qvi = epsqs * esi(k) / (p(k) - (ONE - epsqs) * esi(k));
                        dqsidt = xxls * qvi / (rv * ip2(t(k)));
                        abi = ONE + dqsidt * xxls / cpp;
                        epss = R8(2.) * pi * n0s(k) * rho(k) * dv(k)
                               * (f1s / (lams(k) * lams(k))
                                  + f2s * mg_pow(asn(k) * rho(k) / mu(k), R8(0.5))
                                    * mg_pow(sc(k), K13) * cons14
                                    / (mg_pow(lams(k), R8(5.) / R8(2.) + bs / R8(2.))));
                        prds(k) = epss * (qclr - qvi) / abi;
                        prds(k) = mg_min(prds(k) * (cldmax(k) - dum), ZERO);
                        prds(k) = prds(k) / cldmax(k);
                    }
                    R8 qtmp = q(k) - (cmei(k) + (pre(k) + prds(k)) * cldmax(k)) * deltat;
                    R8 ttmp = t(k) + ((pre(k) * cldmax(k)) * xxlv
                                      + (cmei(k) + prds(k) * cldmax(k)) * xxls) * deltat / cpp;
                    ttmp = mg_max(R8(180.), mg_min(ttmp, R8(323.)));
                    esn = MG_SAT(k, ttmp, 0);
                    qsn = mg_min(epsqs * esn / (p(k) - (ONE - epsqs) * esn), ONE);
                    if (qtmp > qsn) {
                        if (pre(k) + prds(k) < SP(-1.e-20)) {
                            dum1 = pre(k) / (pre(k) + prds(k));
                            qtmp = q(k) - (cmei(k)) * deltat;
                            ttmp = t(k) + (cmei(k) * xxls) * deltat / cpp;
                            esn = MG_SAT(k, ttmp, 0);
                            qsn = mg_min(epsqs * esn / (p(k) - (ONE - epsqs) * esn), ONE);
                            dum = (qtmp - qsn) / (ONE + cons27 * qsn / (cpp * rv * ip2(ttmp)));
                            dum = mg_min(dum, ZERO);
                            pre(k) = dum * dum1 / deltat / cldmax(k);
                            esn = MG_SAT(k, ttmp, 1);
                            qsn = mg_min(epsqs * esn / (p(k) - (ONE - epsqs) * esn), ONE);
                            dum = (qtmp - qsn) / (ONE + cons28 * qsn / (cpp * rv * ip2(ttmp)));
                            dum = mg_min(dum, ZERO);
                            prds(k) = dum * (ONE - dum1) / deltat / cldmax(k);
                        }
                    }
                }
                // Bergeron: droplet evaporation and deposition onto snow
                if (qniic(k) >= qsmall && qcic(k) >= qsmall && t(k) < tmelt) {
                    qvi = epsqs * esi(k) / (p(k) - (ONE - epsqs) * esi(k));
                    qvs = epsqs * esl(k) / (p(k) - (ONE - epsqs) * esl(k));
                    dqsidt = xxls * qvi / (rv * ip2(t(k)));
                    abi = ONE + dqsidt * xxls / cpp;
                    epss = R8(2.) * pi * n0s(k) * rho(k) * dv(k)
                           * (f1s / (lams(k) * lams(k))
                              + f2s * mg_pow(asn(k) * rho(k) / mu(k), R8(0.5))
                                * mg_pow(sc(k), K13) * cons14
                                / (mg_pow(lams(k), R8(5.) / R8(2.) + bs / R8(2.))));
                    bergs(k) = epss * (qvs - qvi) / abi;
                } else {
                    bergs(k) = ZERO;
                }
                // conservation checks
                R8 qce = (qc(k) - berg(k) * deltat);
                R8 nce = (nc(k) + npccn(k) * deltat * mtime);
                R8 qie = (qi(k) + (cmei(k) + berg(k)) * deltat);
                R8 nie = (ni(k) + nnuccd(k) * deltat * mtime);
                R8 ratio;
                dum = (prc(k) + pra(k) + mnuccc(k) + mnucct(k) + msacwi(k) + psacws(k) + bergs(k))
                      * lcldm(k) * deltat;
                if (dum > qce) {
                    if (uw_abs(prc(k) + pra(k) + mnuccc(k) + mnucct(k) + msacwi(k) + psacws(k) + bergs(k)) > R8(1.e-20)) {
                        ratio = qce / deltat / lcldm(k)
                                / (prc(k) + pra(k) + mnuccc(k) + mnucct(k) + msacwi(k) + psacws(k) + bergs(k)) * omsm;
                    } else {
                        ratio = R8(0.5);
                    }
                    prc(k) = prc(k) * ratio;
                    pra(k) = pra(k) * ratio;
                    mnuccc(k) = mnuccc(k) * ratio;
                    mnucct(k) = mnucct(k) * ratio;
                    msacwi(k) = msacwi(k) * ratio;
                    psacws(k) = psacws(k) * ratio;
                    bergs(k) = bergs(k) * ratio;
                }
                dum = (nprc1(k) + npra(k) + nnuccc(k) + nnucct(k) + npsacws(k) - nsubc(k))
                      * lcldm(k) * deltat;
                if (dum > nce) {
                    if (uw_abs(nprc1(k) + npra(k) + nnuccc(k) + nnucct(k) + npsacws(k) - nsubc(k)) > R8(1.e-20)) {
                        ratio = nce / deltat
                                / ((nprc1(k) + npra(k) + nnuccc(k) + nnucct(k) + npsacws(k) - nsubc(k)) * lcldm(k))
                                * omsm;
                    } else {
                        ratio = R8(0.5);
                    }
                    nprc1(k) = nprc1(k) * ratio;
                    npra(k) = npra(k) * ratio;
                    nnuccc(k) = nnuccc(k) * ratio;
                    nnucct(k) = nnucct(k) * ratio;
                    npsacws(k) = npsacws(k) * ratio;
                    nsubc(k) = nsubc(k) * ratio;
                }
                dum = ((-mnuccc(k) - mnucct(k) - msacwi(k)) * lcldm(k) + (prci(k) + prai(k)) * icldm(k)) * deltat;
                if (dum > qie) {
                    if (uw_abs(prci(k) + prai(k)) > SP(1.0e-20)) {
                        ratio = (qie / deltat + (mnuccc(k) + mnucct(k) + msacwi(k)) * lcldm(k))
                                / ((prci(k) + prai(k)) * icldm(k)) * omsm;
                    } else {
                        ratio = SP(0.5);
                    }
                    prci(k) = prci(k) * ratio;
                    prai(k) = prai(k) * ratio;
                }
                dum = ((-nnucct(k) - nsacwi(k)) * lcldm(k) + (nprci(k) + nprai(k) - nsubi(k)) * icldm(k)) * deltat;
                if (dum > nie) {
                    if (uw_abs(((nprci(k) + nprai(k) - nsubi(k)) * icldm(k)) * omsm) > R8(1.0e-20)) {
                        ratio = (nie / deltat + (nnucct(k) + nsacwi(k)) * lcldm(k))
                                / ((nprci(k) + nprai(k) - nsubi(k)) * icldm(k)) * omsm;
                    } else {
                        ratio = SP(0.5);
                    }
                    nprci(k) = nprci(k) * ratio;
                    nprai(k) = nprai(k) * ratio;
                    nsubi(k) = nsubi(k) * ratio;
                }
                if (((prc(k) + pra(k)) * lcldm(k) + (-mnuccr(k) + pre(k) - pracs(k)) * cldmax(k))
                        * dz(k) * rho(k) + qrtot < ZERO) {
                    if (-pre(k) + pracs(k) + mnuccr(k) >= qsmall) {
                        ratio = (qrtot / (dz(k) * rho(k)) + (prc(k) + pra(k)) * lcldm(k))
                                / ((-pre(k) + pracs(k) + mnuccr(k)) * cldmax(k)) * omsm;
                        pre(k) = pre(k) * ratio;
                        pracs(k) = pracs(k) * ratio;
                        mnuccr(k) = mnuccr(k) * ratio;
                    }
                }
                nsubr(k) = ZERO;
                if ((nprc(k) * lcldm(k) + (-nnuccr(k) + nsubr(k) - npracs(k) + nragg(k)) * cldmax(k))
                        * dz(k) * rho(k) + nrtot < ZERO) {
                    if (-nsubr(k) - nragg(k) + npracs(k) + nnuccr(k) >= qsmall) {
                        ratio = (nrtot / (dz(k) * rho(k)) + nprc(k) * lcldm(k))
                                / ((-nsubr(k) - nragg(k) + npracs(k) + nnuccr(k)) * cldmax(k)) * omsm;
                        nsubr(k) = nsubr(k) * ratio;
                        npracs(k) = npracs(k) * ratio;
                        nnuccr(k) = nnuccr(k) * ratio;
                        nragg(k) = nragg(k) * ratio;
                    }
                }
                if (((bergs(k) + psacws(k)) * lcldm(k) + (prai(k) + prci(k)) * icldm(k)
                     + (pracs(k) + mnuccr(k) + prds(k)) * cldmax(k)) * dz(k) * rho(k) + qstot < ZERO) {
                    if (-prds(k) >= qsmall) {
                        ratio = (qstot / (dz(k) * rho(k)) + (bergs(k) + psacws(k)) * lcldm(k)
                                 + (prai(k) + prci(k)) * icldm(k) + (pracs(k) + mnuccr(k)) * cldmax(k))
                                / (-prds(k) * cldmax(k)) * omsm;
                        prds(k) = prds(k) * ratio;
                    }
                }
                nsubs(k) = ZERO;
                if ((nprci(k) * icldm(k) + (nnuccr(k) + nsubs(k) + nsagg(k)) * cldmax(k))
                        * dz(k) * rho(k) + nstot < ZERO) {
                    if (-nsubs(k) - nsagg(k) >= qsmall) {
                        ratio = (nstot / (dz(k) * rho(k)) + nprci(k) * icldm(k) + nnuccr(k) * cldmax(k))
                                / ((-nsubs(k) - nsagg(k)) * cldmax(k)) * omsm;
                        nsubs(k) = nsubs(k) * ratio;
                        nsagg(k) = nsagg(k) * ratio;
                    }
                }
                // tendencies
                qvlat(k) = qvlat(k) - (pre(k) + prds(k)) * cldmax(k) - cmei(k);
                tlat(k) = tlat(k) + ((pre(k) * cldmax(k)) * xxlv + (prds(k) * cldmax(k) + cmei(k)) * xxls
                                     + ((bergs(k) + psacws(k) + mnuccc(k) + mnucct(k) + msacwi(k)) * lcldm(k)
                                        + (mnuccr(k) + pracs(k)) * cldmax(k) + berg(k)) * xlf);
                qctend(k) = qctend(k) + (-pra(k) - prc(k) - mnuccc(k) - mnucct(k) - msacwi(k)
                                         - psacws(k) - bergs(k)) * lcldm(k) - berg(k);
                qitend(k) = qitend(k) + (mnuccc(k) + mnucct(k) + msacwi(k)) * lcldm(k)
                            + (-prci(k) - prai(k)) * icldm(k) + cmei(k) + berg(k);
                qrtend(k) = qrtend(k) + (pra(k) + prc(k)) * lcldm(k)
                            + (pre(k) - pracs(k) - mnuccr(k)) * cldmax(k);
                qnitend(k) = qnitend(k) + (prai(k) + prci(k)) * icldm(k) + (psacws(k) + bergs(k)) * lcldm(k)
                             + (prds(k) + pracs(k) + mnuccr(k)) * cldmax(k);
                nctend(k) = nctend(k) + npccn(k) * mtime
                            + (-nnuccc(k) - nnucct(k) - npsacws(k) + nsubc(k) - npra(k) - nprc1(k)) * lcldm(k);
                nitend(k) = nitend(k) + nnuccd(k) * mtime + (nnucct(k) + nsacwi(k)) * lcldm(k)
                            + (nsubi(k) - nprci(k) - nprai(k)) * icldm(k);
                nstend(k) = nstend(k) + (nsubs(k) + nsagg(k) + nnuccr(k)) * cldmax(k) + nprci(k) * icldm(k);
                nrtend(k) = nrtend(k) + nprc(k) * lcldm(k)
                            + (nsubr(k) - npracs(k) - nnuccr(k) + nragg(k)) * cldmax(k);
                if (nctend(k) > ZERO && nc(k) + nctend(k) * deltat > ncmax)
                    nctend(k) = mg_max(ZERO, (ncmax - nc(k)) / deltat);
                if (nitend(k) > ZERO && ni(k) + nitend(k) * deltat > nimax)
                    nitend(k) = mg_max(ZERO, (nimax - ni(k)) / deltat);
                // final precipitation q and N (MG2008 eq. 15-16)
                if (qric(k) >= qsmall) {
                    if (k == 1) {
                        qric(k) = qrtend(k) * dz(k) / cldmax(k) / umr(k);
                        nric(k) = nrtend(k) * dz(k) / cldmax(k) / unr(k);
                    } else {
                        qric(k) = (rho(k - 1) * umr(k - 1) * qric(k - 1) * cldmax(k - 1)
                                   + (rho(k) * dz(k) * qrtend(k))) / (umr(k) * rho(k) * cldmax(k));
                        nric(k) = (rho(k - 1) * unr(k - 1) * nric(k - 1) * cldmax(k - 1)
                                   + (rho(k) * dz(k) * nrtend(k))) / (unr(k) * rho(k) * cldmax(k));
                    }
                } else {
                    qric(k) = ZERO;
                    nric(k) = ZERO;
                }
                if (qniic(k) >= qsmall) {
                    if (k == 1) {
                        qniic(k) = qnitend(k) * dz(k) / cldmax(k) / ums(k);
                        nsic(k) = nstend(k) * dz(k) / cldmax(k) / uns(k);
                    } else {
                        qniic(k) = (rho(k - 1) * ums(k - 1) * qniic(k - 1) * cldmax(k - 1)
                                    + (rho(k) * dz(k) * qnitend(k))) / (ums(k) * rho(k) * cldmax(k));
                        nsic(k) = (rho(k - 1) * uns(k - 1) * nsic(k - 1) * cldmax(k - 1)
                                   + (rho(k) * dz(k) * nstend(k))) / (uns(k) * rho(k) * cldmax(k));
                    }
                } else {
                    qniic(k) = ZERO;
                    nsic(k) = ZERO;
                }
                prect = prect + (qrtend(k) * dz(k) * rho(k) + qnitend(k) * dz(k) * rho(k)) / rhow;
                preci = preci + qnitend(k) * dz(k) * rho(k) / rhow;
                qrtot = mg_max(qrtot + qrtend(k) * dz(k) * rho(k), ZERO);
                qstot = mg_max(qstot + qnitend(k) * dz(k) * rho(k), ZERO);
                nrtot = mg_max(nrtot + nrtend(k) * dz(k) * rho(k), ZERO);
                nstot = mg_max(nstot + nstend(k) * dz(k) * rho(k), ZERO);
                // melt snow at +2 C
                if (t(k) + tlat(k) / cpp * deltat > R8(275.15)) {
                    if (qstot > ZERO) {
                        dum = -xlf / cpp * qstot / (dz(k) * rho(k));
                        if (t(k) + tlat(k) / cpp * deltat + dum < R8(275.15)) {
                            dum = (t(k) + tlat(k) / cpp * deltat - R8(275.15)) * cpp / xlf;
                            dum = dum / (xlf / cpp * qstot / (dz(k) * rho(k)));
                            dum = mg_max(ZERO, dum);
                            dum = mg_min(ONE, dum);
                        } else {
                            dum = ONE;
                        }
                        qric(k) = qric(k) + dum * qniic(k);
                        nric(k) = nric(k) + dum * nsic(k);
                        qniic(k) = (ONE - dum) * qniic(k);
                        nsic(k) = (ONE - dum) * nsic(k);
                        R8 tmp = -xlf * dum * qstot / (dz(k) * rho(k));
                        tlat(k) = tlat(k) + tmp;
                        qrtot = qrtot + dum * qstot;
                        nrtot = nrtot + dum * nstot;
                        qstot = (ONE - dum) * qstot;
                        nstot = (ONE - dum) * nstot;
                        preci = (ONE - dum) * preci;
                    }
                }
                // freeze all rain at -5 C
                if (t(k) + tlat(k) / cpp * deltat < (tmelt - R8(5.))) {
                    if (qrtot > ZERO) {
                        dum = xlf / cpp * qrtot / (dz(k) * rho(k));
                        if (t(k) + tlat(k) / cpp * deltat + dum > (tmelt - R8(5.))) {
                            dum = -(t(k) + tlat(k) / cpp * deltat - (tmelt - R8(5.))) * cpp / xlf;
                            dum = dum / (xlf / cpp * qrtot / (dz(k) * rho(k)));
                            dum = mg_max(ZERO, dum);
                            dum = mg_min(ONE, dum);
                        } else {
                            dum = ONE;
                        }
                        qniic(k) = qniic(k) + dum * qric(k);
                        nsic(k) = nsic(k) + dum * nric(k);
                        qric(k) = (ONE - dum) * qric(k);
                        nric(k) = (ONE - dum) * nric(k);
                        R8 tmp = xlf * dum * qrtot / (dz(k) * rho(k));
                        tlat(k) = tlat(k) + tmp;
                        qstot = qstot + dum * qrtot;
                        qrtot = (ONE - dum) * qrtot;
                        nstot = nstot + dum * nrtot;
                        nrtot = (ONE - dum) * nrtot;
                        preci = preci + dum * (prect - preci);
                    }
                }
                if (qniic(k) < qsmall) { qniic(k) = ZERO; nsic(k) = ZERO; }
                if (qric(k) < qsmall) { qric(k) = ZERO; nric(k) = ZERO; }
                nric(k) = mg_max(nric(k), ZERO);
                nsic(k) = mg_max(nsic(k), ZERO);
                // rain fallspeed
                if (qric(k) >= qsmall) {
                    lamr(k) = mg_pow(pi * rhow * nric(k) / qric(k), K13);
                    n0r(k) = nric(k) * lamr(k);
                    if (lamr(k) < lamminr) {
                        lamr(k) = lamminr;
                        n0r(k) = ip4(lamr(k)) * qric(k) / (pi * rhow);
                        nric(k) = n0r(k) / lamr(k);
                    } else if (lamr(k) > lammaxr) {
                        lamr(k) = lammaxr;
                        n0r(k) = ip4(lamr(k)) * qric(k) / (pi * rhow);
                        nric(k) = n0r(k) / lamr(k);
                    }
                    unr(k) = mg_min(arn(k) * cons4 / mg_pow(lamr(k), br), R8(9.1) * rhof(k));
                    umr(k) = mg_min(arn(k) * cons5 / (R8(6.) * mg_pow(lamr(k), br)), R8(9.1) * rhof(k));
                } else {
                    lamr(k) = ZERO; n0r(k) = ZERO; umr(k) = ZERO; unr(k) = ZERO;
                }
                // snow fallspeed
                if (qniic(k) >= qsmall) {
                    lams(k) = mg_pow(cons6 * cs * nsic(k) / qniic(k), ONE / ds);
                    n0s(k) = nsic(k) * lams(k);
                    if (lams(k) < lammins) {
                        lams(k) = lammins;
                        n0s(k) = mg_pow(lams(k), ds + ONE) * qniic(k) / (cs * cons6);
                        nsic(k) = n0s(k) / lams(k);
                    } else if (lams(k) > lammaxs) {
                        lams(k) = lammaxs;
                        n0s(k) = mg_pow(lams(k), ds + ONE) * qniic(k) / (cs * cons6);
                        nsic(k) = n0s(k) / lams(k);
                    }
                    ums(k) = mg_min(asn(k) * cons8 / (R8(6.) * mg_pow(lams(k), bs)), R8(1.2) * rhof(k));
                    uns(k) = mg_min(asn(k) * cons7 / mg_pow(lams(k), bs), R8(1.2) * rhof(k));
                } else {
                    lams(k) = ZERO; n0s(k) = ZERO; ums(k) = ZERO; uns(k) = ZERO;
                }
                // sums over the substep
                qrout(k) = qrout(k) + qric(k) * cldmax(k);
                qsout(k) = qsout(k) + qniic(k) * cldmax(k);
                nrout(k) = nrout(k) + nric(k) * rho(k) * cldmax(k);
                nsout(k) = nsout(k) + nsic(k) * rho(k) * cldmax(k);
                tlat1(k) = tlat1(k) + tlat(k);
                qvlat1(k) = qvlat1(k) + qvlat(k);
                qctend1(k) = qctend1(k) + qctend(k);
                qitend1(k) = qitend1(k) + qitend(k);
                nctend1(k) = nctend1(k) + nctend(k);
                nitend1(k) = nitend1(k) + nitend(k);
                t(k) = t(k) + tlat(k) * deltat / cpp;
                q(k) = q(k) + qvlat(k) * deltat;
                qc(k) = qc(k) + qctend(k) * deltat;
                qi(k) = qi(k) + qitend(k) * deltat;
                nc(k) = nc(k) + nctend(k) * deltat;
                ni(k) = ni(k) + nitend(k) * deltat;
            }
            prect1 = prect1 + prect;
            preci1 = preci1 + preci;
        }
    }
    deltat = deltat * R8(2.0);             // real(iter)

    if (ltrue == 0) {
        for (int k = 1; k <= pver; ++k) {
            effc(k) = R8(10.);
            effi(k) = R8(25.);
        }
    } else {
        int nstep = 1;
        prect = prect1 / R8(2.0);
        preci = preci1 / R8(2.0);
        R8 uni, umi, unc, umc;
        for (int k = 1; k <= pver; ++k) {
            t(k) = t1(k); q(k) = q1(k); qc(k) = qc1(k); qi(k) = qi1(k);
            nc(k) = nc1(k); ni(k) = ni1(k);
            tlat(k) = tlat1(k) / R8(2.0);
            qvlat(k) = qvlat1(k) / R8(2.0);
            qctend(k) = qctend1(k) / R8(2.0);
            qitend(k) = qitend1(k) / R8(2.0);
            nctend(k) = nctend1(k) / R8(2.0);
            nitend(k) = nitend1(k) / R8(2.0);
            qrout(k) = qrout(k) / R8(2.0);
            qsout(k) = qsout(k) / R8(2.0);
            nrout(k) = nrout(k) / R8(2.0);
            nsout(k) = nsout(k) / R8(2.0);
            // sedimentation of cloud water and ice: in-cloud slopes
            dumc(k) = (qc(k) + qctend(k) * deltat) / lcldm(k);
            dumi(k) = (qi(k) + qitend(k) * deltat) / icldm(k);
            dumnc(k) = mg_max((nc(k) + nctend(k) * deltat) / lcldm(k), ZERO);
            dumni(k) = mg_max((ni(k) + nitend(k) * deltat) / icldm(k), ZERO);
            if (dumi(k) >= qsmall) {
                dumni(k) = mg_min(dumni(k), dumi(k) * R8(1.e20));
                lami(k) = mg_pow(cons1 * ci * dumni(k) / dumi(k), ONE / di);
                lami(k) = mg_max(lami(k), lammini);
                lami(k) = mg_min(lami(k), lammaxi);
            } else {
                lami(k) = ZERO;
            }
            if (dumc(k) >= qsmall) {
                dumnc(k) = mg_min(dumnc(k), dumc(k) * R8(1.e20));
                dumnc(k) = mg_max(dumnc(k), cdnl / rho(k));
                pgam(k) = R8(0.0005714) * (ncic(k) / R8(1.e6) * rho(k)) + R8(0.2714);
                pgam(k) = ONE / ip2(pgam(k)) - ONE;
                pgam(k) = mg_max(pgam(k), R8(2.));
                pgam(k) = mg_min(pgam(k), R8(15.));
                lamc(k) = mg_pow(pi / R8(6.) * rhow * dumnc(k) * mg_gamma(pgam(k) + R8(4.), bad)
                                 / (dumc(k) * mg_gamma(pgam(k) + ONE, bad)), K13);
                R8 lammin = (pgam(k) + ONE) / R8(50.e-6);
                R8 lammax = (pgam(k) + ONE) / R8(2.e-6);
                lamc(k) = mg_max(lamc(k), lammin);
                lamc(k) = mg_min(lamc(k), lammax);
            } else {
                lamc(k) = ZERO;
            }
            if (dumc(k) >= qsmall) {
                unc = acn(k) * mg_gamma(ONE + bc + pgam(k), bad)
                      / (mg_pow(lamc(k), bc) * mg_gamma(pgam(k) + ONE, bad));
                umc = acn(k) * mg_gamma(R8(4.) + bc + pgam(k), bad)
                      / (mg_pow(lamc(k), bc) * mg_gamma(pgam(k) + R8(4.), bad));
                vtrmc(k) = umc;
            } else {
                umc = ZERO;
                unc = ZERO;
            }
            if (dumi(k) >= qsmall) {
                uni = ain(k) * cons16 / mg_pow(lami(k), bi);
                umi = ain(k) * cons17 / (R8(6.) * mg_pow(lami(k), bi));
                uni = mg_min(uni, R8(1.2) * rhof(k));
                umi = mg_min(umi, R8(1.2) * rhof(k));
            } else {
                umi = ZERO;
                uni = ZERO;
            }
            fi(k) = g * rho(k) * umi;
            fni(k) = g * rho(k) * uni;
            fc(k) = g * rho(k) * umc;
            fnc(k) = g * rho(k) * unc;
            R8 rgvm = mg_max(mg_max(mg_max(fi(k), fc(k)), fni(k)), fnc(k));
            int ns = uw_int(rgvm * deltat / pdel(k) + ONE);
            nstep = ns > nstep ? ns : nstep;
            dumc(k) = (qc(k) + qctend(k) * deltat);
            dumi(k) = (qi(k) + qitend(k) * deltat);
            dumnc(k) = mg_max((nc(k) + nctend(k) * deltat), ZERO);
            dumni(k) = mg_max((ni(k) + nitend(k) * deltat), ZERO);
            if (dumc(k) < qsmall) dumnc(k) = ZERO;
            if (dumi(k) < qsmall) dumni(k) = ZERO;
        }
        const R8 rnstep = uw_real(nstep);
        for (int n = 1; n <= nstep; ++n) {
            for (int k = 1; k <= pver; ++k) {
                falouti(k) = fi(k) * dumi(k);
                faloutni(k) = fni(k) * dumni(k);
                faloutc(k) = fc(k) * dumc(k);
                faloutnc(k) = fnc(k) * dumnc(k);
            }
            int k = 1;
            R8 faltndi = falouti(k) / pdel(k);
            R8 faltndni = faloutni(k) / pdel(k);
            R8 faltndc = faloutc(k) / pdel(k);
            R8 faltndnc = faloutnc(k) / pdel(k);
            qitend(k) = qitend(k) - faltndi / rnstep;
            nitend(k) = nitend(k) - faltndni / rnstep;
            qctend(k) = qctend(k) - faltndc / rnstep;
            nctend(k) = nctend(k) - faltndnc / rnstep;
            dumi(k) = dumi(k) - faltndi * deltat / rnstep;
            dumni(k) = dumni(k) - faltndni * deltat / rnstep;
            dumc(k) = dumc(k) - faltndc * deltat / rnstep;
            dumnc(k) = dumnc(k) - faltndnc * deltat / rnstep;
            for (k = 2; k <= pver; ++k) {
                dum = lcldm(k) / lcldm(k - 1);
                dum = mg_min(dum, ONE);
                dum1 = icldm(k) / icldm(k - 1);
                dum1 = mg_min(dum1, ONE);
                R8 faltndqie = (falouti(k) - falouti(k - 1)) / pdel(k);
                faltndi = (falouti(k) - dum1 * falouti(k - 1)) / pdel(k);
                faltndni = (faloutni(k) - dum1 * faloutni(k - 1)) / pdel(k);
                R8 faltndqce = (faloutc(k) - faloutc(k - 1)) / pdel(k);
                faltndc = (faloutc(k) - dum * faloutc(k - 1)) / pdel(k);
                faltndnc = (faloutnc(k) - dum * faloutnc(k - 1)) / pdel(k);
                qitend(k) = qitend(k) - faltndi / rnstep;
                nitend(k) = nitend(k) - faltndni / rnstep;
                qctend(k) = qctend(k) - faltndc / rnstep;
                nctend(k) = nctend(k) - faltndnc / rnstep;
                qvlat(k) = qvlat(k) - (faltndqie - faltndi) / rnstep;
                qvlat(k) = qvlat(k) - (faltndqce - faltndc) / rnstep;
                tlat(k) = tlat(k) + (faltndqie - faltndi) * xxls / rnstep;
                tlat(k) = tlat(k) + (faltndqce - faltndc) * xxlv / rnstep;
                dumi(k) = dumi(k) - faltndi * deltat / rnstep;
                dumni(k) = dumni(k) - faltndni * deltat / rnstep;
                dumc(k) = dumc(k) - faltndc * deltat / rnstep;
                dumnc(k) = dumnc(k) - faltndnc * deltat / rnstep;
                fni(k) = mg_max(fni(k) / pdel(k), fni(k - 1) / pdel(k - 1)) * pdel(k);
                fi(k) = mg_max(fi(k) / pdel(k), fi(k - 1) / pdel(k - 1)) * pdel(k);
                fnc(k) = mg_max(fnc(k) / pdel(k), fnc(k - 1) / pdel(k - 1)) * pdel(k);
                fc(k) = mg_max(fc(k) / pdel(k), fc(k - 1) / pdel(k - 1)) * pdel(k);
            }
            prect = prect + (faloutc(pver) + falouti(pver)) / g / rnstep / R8(1000.);
            preci = preci + (falouti(pver)) / g / rnstep / R8(1000.);
        }
        for (int k = 1; k <= pver; ++k) {
            dumc(k) = mg_max(qc(k) + qctend(k) * deltat, ZERO);
            dumi(k) = mg_max(qi(k) + qitend(k) * deltat, ZERO);
            dumnc(k) = mg_max(nc(k) + nctend(k) * deltat, ZERO);
            dumni(k) = mg_max(ni(k) + nitend(k) * deltat, ZERO);
            if (dumc(k) < qsmall) dumnc(k) = ZERO;
            if (dumi(k) < qsmall) dumni(k) = ZERO;
            // melting of cloud ice
            if (t(k) + tlat(k) / cpp * deltat > tmelt) {
                if (dumi(k) > ZERO) {
                    dum = -dumi(k) * xlf / cpp;
                    if (t(k) + tlat(k) / cpp * deltat + dum < tmelt) {
                        dum = (t(k) + tlat(k) / cpp * deltat - tmelt) * cpp / xlf;
                        dum = dum / dumi(k) * xlf / cpp;
                        dum = mg_max(ZERO, dum);
                        dum = mg_min(ONE, dum);
                    } else {
                        dum = ONE;
                    }
                    qctend(k) = qctend(k) + dum * dumi(k) / deltat;
                    nctend(k) = nctend(k) + R8(3.) * dum * dumi(k) / deltat
                                / (R8(4.) * pi * R8(5.12e-16) * rhow);
                    qitend(k) = ((ONE - dum) * dumi(k) - qi(k)) / deltat;
                    nitend(k) = ((ONE - dum) * dumni(k) - ni(k)) / deltat;
                    tlat(k) = tlat(k) - xlf * dum * dumi(k) / deltat;
                }
            }
            // homogeneous freezing at -40 C
            if (t(k) + tlat(k) / cpp * deltat < R8(233.15)) {
                if (dumc(k) > ZERO) {
                    dum = dumc(k) * xlf / cpp;
                    if (t(k) + tlat(k) / cpp * deltat + dum > R8(233.15)) {
                        dum = -(t(k) + tlat(k) / cpp * deltat - R8(233.15)) * cpp / xlf;
                        dum = dum / dumc(k) * xlf / cpp;
                        dum = mg_max(ZERO, dum);
                        dum = mg_min(ONE, dum);
                    } else {
                        dum = ONE;
                    }
                    qitend(k) = qitend(k) + dum * dumc(k) / deltat;
                    nitend(k) = nitend(k) + dum * R8(3.) * dumc(k)
                                / (R8(4.) * R8(3.14) * R8(1.563e-14) * R8(500.)) / deltat;
                    qctend(k) = ((ONE - dum) * dumc(k) - qc(k)) / deltat;
                    nctend(k) = ((ONE - dum) * dumnc(k) - nc(k)) / deltat;
                    tlat(k) = tlat(k) + xlf * dum * dumc(k) / deltat;
                }
            }
            // remove excess supersaturation
            R8 qtmp = q(k) + qvlat(k) * deltat;
            R8 ttmp = t(k) + tlat(k) / cpp * deltat;
            R8 esn = MG_SAT(k, ttmp, 0);
            R8 qsn = mg_min(epsqs * esn / (p(k) - (ONE - epsqs) * esn), ONE);
            if (qtmp > qsn && qsn > R8(0.0)) {
                dum = (qtmp - qsn) / (ONE + cons27 * qsn / (cpp * rv * ip2(ttmp))) / deltat;
                if (ttmp > R8(268.15)) {
                    dum1 = R8(0.0);
                } else if (ttmp < R8(238.15)) {
                    dum1 = R8(1.0);
                } else {
                    dum1 = (R8(268.15) - ttmp) / R8(30.);
                }
                dum = (qtmp - qsn) / (ONE + ip2(xxls * dum1 + xxlv * (ONE - dum1)) * qsn
                                      / (cpp * rv * ip2(ttmp))) / deltat;
                qctend(k) = qctend(k) + dum * (ONE - dum1);
                qitend(k) = qitend(k) + dum * dum1;
                qvlat(k) = qvlat(k) - dum;
                tlat(k) = tlat(k) + dum * (ONE - dum1) * xxlv + dum * dum1 * xxls;
            }
            // effective radii
            dumc(k) = mg_max(qc(k) + qctend(k) * deltat, ZERO) / lcldm(k);
            dumi(k) = mg_max(qi(k) + qitend(k) * deltat, ZERO) / icldm(k);
            dumnc(k) = mg_max(nc(k) + nctend(k) * deltat, ZERO) / lcldm(k);
            dumni(k) = mg_max(ni(k) + nitend(k) * deltat, ZERO) / icldm(k);
            dumc(k) = mg_min(dumc(k), R8(5.e-3));
            dumi(k) = mg_min(dumi(k), R8(5.e-3));
            if (dumi(k) >= qsmall) {
                dumni(k) = mg_min(dumni(k), dumi(k) * R8(1.e20));
                lami(k) = mg_pow(cons1 * ci * dumni(k) / dumi(k), ONE / di);
                if (lami(k) < lammini) {
                    lami(k) = lammini;
                    n0i(k) = mg_pow(lami(k), di + ONE) * dumi(k) / (ci * cons1);
                    niic(k) = n0i(k) / lami(k);
                    nitend(k) = (niic(k) * icldm(k) - ni(k)) / deltat;
                } else if (lami(k) > lammaxi) {
                    lami(k) = lammaxi;
                    n0i(k) = mg_pow(lami(k), di + ONE) * dumi(k) / (ci * cons1);
                    niic(k) = n0i(k) / lami(k);
                    nitend(k) = (niic(k) * icldm(k) - ni(k)) / deltat;
                }
                effi(k) = R8(1.5) / lami(k) * R8(1.e6);
            } else {
                effi(k) = R8(25.);
            }
            if (dumc(k) >= qsmall) {
                dumnc(k) = mg_min(dumnc(k), dumc(k) * R8(1.e20));
                if (dumnc(k) < cdnl / rho(k))
                    nctend(k) = (cdnl / rho(k) * lcldm(k) - nc(k)) / deltat;
                dumnc(k) = mg_max(dumnc(k), cdnl / rho(k));
                pgam(k) = R8(0.0005714) * (ncic(k) / R8(1.e6) * rho(k)) + R8(0.2714);
                pgam(k) = ONE / ip2(pgam(k)) - ONE;
                pgam(k) = mg_max(pgam(k), R8(2.));
                pgam(k) = mg_min(pgam(k), R8(15.));
                lamc(k) = mg_pow(pi / R8(6.) * rhow * dumnc(k) * mg_gamma(pgam(k) + R8(4.), bad)
                                 / (dumc(k) * mg_gamma(pgam(k) + ONE, bad)), K13);
                R8 lammin = (pgam(k) + ONE) / R8(50.e-6);
                R8 lammax = (pgam(k) + ONE) / R8(2.e-6);
                if (lamc(k) < lammin) {
                    lamc(k) = lammin;
                    ncic(k) = R8(6.) * ip3(lamc(k)) * dumc(k) * mg_gamma(pgam(k) + ONE, bad)
                              / (pi * rhow * mg_gamma(pgam(k) + R8(4.), bad));
                    nctend(k) = (ncic(k) * lcldm(k) - nc(k)) / deltat;
                } else if (lamc(k) > lammax) {
                    lamc(k) = lammax;
                    ncic(k) = R8(6.) * ip3(lamc(k)) * dumc(k) * mg_gamma(pgam(k) + ONE, bad)
                              / (pi * rhow * mg_gamma(pgam(k) + R8(4.), bad));
                    nctend(k) = (ncic(k) * lcldm(k) - nc(k)) / deltat;
                }
                effc(k) = mg_gamma(pgam(k) + R8(4.), bad) / mg_gamma(pgam(k) + R8(3.), bad)
                          / lamc(k) / R8(2.) * R8(1.e6);
            } else {
                effc(k) = R8(10.);
            }
        }
    }
    // if updated q is below qsmall, zero the number (F:3349-3354)
    for (int k = 1; k <= pver; ++k) {
        if (qc(k) + qctend(k) * deltat < qsmall) nctend(k) = -nc(k) / deltat;
        if (qi(k) + qitend(k) * deltat < qsmall) nitend(k) = -ni(k) / deltat;
    }
    o.prect = prect;
    o.preci = preci;
}

#endif  // CAMMGMP_MICRO_CUH
