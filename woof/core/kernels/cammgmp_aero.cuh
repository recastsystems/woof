// cammgmp_aero.cuh -- aerosol activation of the CAMMGMP port: microp_aero_ts
// with nucleati/hetero/hf (module_cam_mp_microp_aero.F) and dropmixnuc with
// explmix/activate_modal/maxsat/loadaer (module_cam_mp_ndrop.F), WRF v4.6.1,
// MODAL_AERO_3MODE with the prescribed aerosol of the non-chemistry build.
// ArWen's own work, Apache-2.0.
//
// CAM column order: k = 1 is the model top, pver the lowest layer.
//
// What is carried and what is not.  Outside WRF-Chem the driver prescribes
// the aerosol afresh on every call (module_mp_cammgmp_driver.F:941-982):
// each of the 12 constituents beyond the five water species is one value
// for the whole column, and the cloud-borne array qqcw starts at 1e-38.
// dropmixnuc's aerosol tendencies are written into ptend_loc_q(:,:,6:12),
// which the driver never sums or applies (ptend_loc_lq(6:12) stay .false.),
// so of its work only the droplet-number tendency leaves the routine.  The
// mixing of the aerosol NUMBER moments still matters -- it feeds the
// activation source of later substeps -- and is carried here; the aerosol
// MASS mixing (explmix over lmass/lmasscw) reaches no output and is not
// computed.  Likewise nsource/ndropmix/ccncalc/coltend are history fields.
//
// Stock WRF indexing is the default in both arithmetic modes. At the
// lowest layer, ndrop.F:759-760 and :782-783 read raercol, including for
// the cloud-borne source. The alternate cloud-borne read is available
// only in the CAMMGMP_CORRECTED_NDROP_DIAGNOSTIC attribution instrument.
// An altered referee is never a native WRF parity gate.
#ifndef CAMMGMP_AERO_CUH
#define CAMMGMP_AERO_CUH

#define MG_NMODE 3
// modal_aero_initialize_data_phys: modes accum 1, aitken 2, coarse 3;
// species 6 accum mass, 7 accum num, 8 aitken mass, 9 aitken num, 10 coarse
// dust mass, 11 coarse sea-salt mass, 12 coarse num.  qaer[l] is
// state1_q(:,:,l), one value per column (prescribed).
struct MgAerosol {
    R8 q[13];        // 1-based constituent index, 6..12 used
};

__device__ __forceinline__ R8 md_voltonumbhi(int m) {
    return m == 1 ? R8(MD_VOLTONUMBHI1) : (m == 2 ? R8(MD_VOLTONUMBHI2) : R8(MD_VOLTONUMBHI3));
}
__device__ __forceinline__ R8 md_voltonumblo(int m) {
    return m == 1 ? R8(MD_VOLTONUMBLO1) : (m == 2 ? R8(MD_VOLTONUMBLO2) : R8(MD_VOLTONUMBLO3));
}
__device__ __forceinline__ R8 nd_npv(int m) {
    return m == 1 ? R8(ND_NPV1) : (m == 2 ? R8(ND_NPV2) : R8(ND_NPV3));
}
__device__ __forceinline__ R8 nd_alogsig(int m) {
    return m == 1 ? R8(ND_ALOGSIG1) : (m == 2 ? R8(ND_ALOGSIG2) : R8(ND_ALOGSIG3));
}
__device__ __forceinline__ R8 nd_exp45logsig(int m) {
    return m == 1 ? R8(ND_EXP45LOGSIG1) : (m == 2 ? R8(ND_EXP45LOGSIG2) : R8(ND_EXP45LOGSIG3));
}
__device__ __forceinline__ R8 nd_f1(int m) {
    return m == 1 ? R8(ND_F11) : (m == 2 ? R8(ND_F12) : R8(ND_F13));
}
__device__ __forceinline__ R8 nd_f2(int m) {
    return m == 1 ? R8(ND_F21) : (m == 2 ? R8(ND_F22) : R8(ND_F23));
}
__device__ __forceinline__ int md_numptr(int m) { return m == 1 ? 7 : (m == 2 ? 9 : 12); }

// ndrop loadaer, phase 1 (interstitial), one column, one level
// (module_cam_mp_ndrop.F:1705-1822).
__device__ void nd_loadaer(const MgAerosol& raer, R8 cs, int m,
                           R8& naerosol, R8& vaerosol, R8& hygro) {
    vaerosol = R8(0.0);
    hygro = R8(0.0);
    // lmassptr_amode / lspectype_amode (modal_aero_initialize_data_phys.F):
    // specdens_amode = 1.0e3 and spechygro = 0.5 for every type.
    const int lmass1 = (m == 1) ? 6 : (m == 2 ? 8 : 10);
    const int nspec = (m == 3) ? 2 : 1;
    for (int l = 1; l <= nspec; ++l) {
        const int lmass = (l == 1) ? lmass1 : 11;
        R8 vol = mg_max(raer.q[lmass], R8(0.0)) / R8(MD_SPECDENS);
        vaerosol = vaerosol + vol;
        hygro = hygro + vol * R8(MD_SPECHYGRO);
    }
    if (vaerosol > R8(1.0e-30)) {
        hygro = hygro / vaerosol;
        vaerosol = vaerosol * cs;
    } else {
        hygro = R8(0.0);
        vaerosol = R8(0.0);
    }
    naerosol = raer.q[md_numptr(m)] * cs;
    naerosol = mg_max(naerosol, vaerosol * md_voltonumbhi(m));
    naerosol = mg_min(naerosol, vaerosol * md_voltonumblo(m));
}

// ndrop maxsat (module_cam_mp_ndrop.F:1547-1597).
__device__ R8 nd_maxsat(const R8* zeta, const R8* eta, const R8* smc) {
    R8 smax = R8(0.0);
    bool significant = false;
    for (int m = 1; m <= MG_NMODE; ++m) {
        if (zeta[m] > R8(1.e5) * eta[m] || smc[m] * smc[m] > R8(1.e5) * eta[m]) {
            smax = R8(1.e-20);
        } else {
            significant = true;
            break;
        }
    }
    if (!significant) return smax;
    R8 sum = R8(0.0);
    for (int m = 1; m <= MG_NMODE; ++m) {
        if (eta[m] > R8(1.e-20)) {
            R8 g1 = zeta[m] / eta[m];
            R8 g1sqrt = mg_sqrt(g1);
            g1 = g1sqrt * g1;
            R8 g2 = smc[m] / mg_sqrt(eta[m] + R8(3.0) * zeta[m]);
            R8 g2sqrt = mg_sqrt(g2);
            g2 = g2sqrt * g2;
            sum = sum + (nd_f1(m) * g1 + nd_f2(m) * g2) / (smc[m] * smc[m]);
        } else {
            sum = R8(1.e20);
        }
    }
    return R8(1.0) / mg_sqrt(sum);
}

// ndrop activate_modal (module_cam_mp_ndrop.F:1099-1545) on the branch
// dropmixnuc can reach: both call sites pass sigw = wmix = 0 and wdiab = 0,
// so the single-updraft arm (:1505-1542) runs; the updraft-spectrum
// integration (:1268-1504) needs sigw > 1e-5 and is unreachable.  Only
// fluxn and flux_fullact leave (fn/fm feed the growing-cloud source).
__device__ void nd_activate_modal(R8 wbar, R8 tair, R8 rhoair, const R8* na,
                                  const R8* volume, const R8* hygro,
                                  R8* fn, R8* fluxn, R8& flux_fullact) {
    for (int m = 1; m <= MG_NMODE; ++m) { fn[m] = R8(0.0); fluxn[m] = R8(0.0); }
    flux_fullact = R8(0.0);
    const R8 sigw = R8(0.0), wdiab = R8(0.0);
    if (sigw <= R8(1.e-5) && wbar <= SP(0.)) return;
    const R8 p0 = R8(1013.25e2);
    R8 pres = R8(PC_RAIR) * rhoair * tair;
    R8 diff0 = R8(0.211e-4) * (p0 / pres) * mg_pow(tair / R8(ND_T0), SP(1.94));
    R8 conduct0 = (R8(5.69) + R8(0.017) * (tair - R8(ND_T0))) * R8(4.186e2) * R8(1.e-5);
    R8 es = uw_estblf(tair);
    R8 qs = R8(PC_EPSILO) * es / (pres - (R8(1.0) - R8(WV_EPSQS)) * es);
    R8 dqsdt = R8(MG_XXLV) / (R8(PC_RH2O) * tair * tair) * qs;
    R8 alpha = R8(PC_GRAVIT) * (R8(MG_XXLV) / (R8(MG_CPP) * R8(PC_RH2O) * tair * tair)
                                - SP(1.) / (R8(PC_RAIR) * tair));
    R8 gammav = (R8(1.0) + R8(MG_XXLV) / R8(MG_CPP) * dqsdt) / (rhoair * qs);
    const R8 wmaxf = R8(10.0);
    R8 etafactor2max = SP(1.e10) / mg_pow(alpha * wmaxf, SP(1.5));
    R8 sqrtg[MG_NMODE + 1], smc[MG_NMODE + 1], lnsm[MG_NMODE + 1];
    R8 etafactor2[MG_NMODE + 1], eta[MG_NMODE + 1], zeta[MG_NMODE + 1];
    for (int m = 1; m <= MG_NMODE; ++m) {
        R8 g;
        if (volume[m] > R8(1.e-39) && na[m] > R8(1.e-39)) {
            R8 amcube = SP(3.) * volume[m] / (SP(4.) * R8(ND_PI) * nd_exp45logsig(m) * na[m]);
            g = R8(1.0) / (R8(MG_RHOW) / (diff0 * rhoair * qs)
                           + R8(MG_XXLV) * R8(MG_RHOW) / (conduct0 * tair)
                             * (R8(MG_XXLV) / (R8(PC_RH2O) * tair) - R8(1.0)));
            sqrtg[m] = mg_sqrt(g);
            R8 beta = R8(2.0) * R8(ND_PI) * R8(MG_RHOW) * g * gammav;
            etafactor2[m] = R8(1.0) / (na[m] * beta * sqrtg[m]);
            if (hygro[m] > SP(1.e-10)) {
                smc[m] = SP(2.) * R8(ND_ATEN) * mg_sqrt(R8(ND_ATEN) / (SP(27.) * hygro[m] * amcube));
            } else {
                smc[m] = SP(100.);
            }
        } else {
            g = R8(1.0) / (R8(MG_RHOW) / (diff0 * rhoair * qs)
                           + R8(MG_XXLV) * R8(MG_RHOW) / (conduct0 * tair)
                             * (R8(MG_XXLV) / (R8(PC_RH2O) * tair) - R8(1.0)));
            sqrtg[m] = mg_sqrt(g);
            smc[m] = R8(1.0);
            etafactor2[m] = etafactor2max;
        }
        lnsm[m] = mg_log(smc[m]);
    }
    // single updraft (sigw <= 1e-5)
    R8 wnuc = wbar + wdiab;
    if (wnuc > R8(0.0)) {
        R8 w = wbar;
        R8 alw = alpha * wnuc;
        R8 sqrtalw = mg_sqrt(alw);
        R8 etafactor1 = alw * sqrtalw;
        for (int m = 1; m <= MG_NMODE; ++m) {
            eta[m] = etafactor1 * etafactor2[m];
            zeta[m] = R8(ND_TWOTHIRD) * sqrtalw * R8(ND_ATEN) / sqrtg[m];
        }
        R8 smax = nd_maxsat(zeta, eta, smc);
        R8 lnsmax = mg_log(smax);
        for (int m = 1; m <= MG_NMODE; ++m) {
            R8 x = R8(ND_TWOTHIRD) * (lnsm[m] - lnsmax) / (R8(ND_SQ2) * nd_alogsig(m));
            fn[m] = R8(0.5) * (R8(1.0) - cam_erf(x));
            if (wbar > R8(0.0)) fluxn[m] = fn[m] * w;
        }
        flux_fullact = w;
    }
}

// ndrop explmix (module_cam_mp_ndrop.F:948-1021), surfrate = flxconv = 0 at
// every CAMMGMP call site (depvel = 0, flxconv = 0.): the k = pver loss line
// still runs, as q - 0*qold*dt + 0*dt, and is kept for the sign of zero.
__device__ void nd_explmix(V q, V src, V ekkp, V ekkm, V overlapp, V overlapm,
                           V qold, int pver, R8 dt, bool is_unact, V qactold) {
    const R8 surfrate = R8(0.0), flxconv = R8(0.0);
    if (is_unact) {
        for (int k = 1; k <= pver; ++k) {
            int kp1 = k + 1 < pver ? k + 1 : pver;
            int km1 = k - 1 > 1 ? k - 1 : 1;
            q(k) = qold(k) + dt * (-src(k)
                     + ekkp(k) * (qold(kp1) - qold(k) + qactold(kp1) * (R8(1.0) - overlapp(k)))
                     + ekkm(k) * (qold(km1) - qold(k) + qactold(km1) * (R8(1.0) - overlapm(k))));
            q(k) = mg_max(q(k), R8(0.0));
        }
    } else {
        for (int k = 1; k <= pver; ++k) {
            int kp1 = k + 1 < pver ? k + 1 : pver;
            int km1 = k - 1 > 1 ? k - 1 : 1;
            q(k) = qold(k) + dt * (src(k) + ekkp(k) * (overlapp(k) * qold(kp1) - qold(k))
                                   + ekkm(k) * (overlapm(k) * qold(km1) - qold(k)));
            q(k) = mg_max(q(k), R8(0.0));
        }
    }
    q(pver) = q(pver) - surfrate * qold(pver) * dt + flxconv * dt;
    q(pver) = mg_max(q(pver), R8(0.0));
}

// ndrop dropmixnuc (module_cam_mp_ndrop.F:42-946), one column.  Returns the
// droplet-number tendency tendnd (#/kg/s); everything else it computes is a
// history field or an aerosol tendency the driver discards (see header).
__device__ void nd_dropmixnuc(int pver, V ncldwtr, V tendnd, V temp, V pmid,
                              V pint, V rpdel, V zm, V kvh, V wsub, V cldn,
                              V cldo, const MgAerosol& raer, R8 dtmicro,
                              Ws& ws) {
    WsMark mark(ws);
    V qcld = ws.r8(pver), qncld = ws.r8(pver), srcn = ws.r8(pver);
    V zs = ws.r8(pver), cs = ws.r8(pver), dz = ws.r8(pver), zn = ws.r8(pver);
    V ekd = ws.r8(pver), csbot = ws.r8(pver), csbot_cscen = ws.r8(pver);
    V wtke = ws.r8(pver), wtke_cen = ws.r8(pver);
    V ekk = ws.r8(pver + 1);             // ekk(0:pver) -> ekk(k+1)
    V ekkp = ws.r8(pver), ekkm = ws.r8(pver);
    V overlapp = ws.r8(pver), overlapm = ws.r8(pver);
    V source = ws.r8(pver), flux_fullact = ws.r8(pver);
    V nact[MG_NMODE + 1];
    V raercol[MG_NMODE + 1][3], raercol_cw[MG_NMODE + 1][3];  // [m][nsav]
    for (int m = 1; m <= MG_NMODE; ++m) {
        nact[m] = ws.r8(pver);
        raercol[m][1] = ws.r8(pver); raercol[m][2] = ws.r8(pver);
        raercol_cw[m][1] = ws.r8(pver); raercol_cw[m][2] = ws.r8(pver);
    }
    const R8 zkmin = R8(0.01), zkmax = R8(100.);
    const R8 wmixmin = SP(0.1);
    const R8 sq2pi = R8(2.5066283);
    const R8 dtinv = SP(1.) / dtmicro;
    const R8 depvel = R8(0.0);

    for (int k = 1; k <= pver - 1; ++k) zs(k) = R8(1.0) / (zm(k) - zm(k + 1));
    zs(pver) = zs(pver - 1);
    for (int k = 1; k <= pver; ++k) {
        qcld(k) = ncldwtr(k);
        qncld(k) = R8(0.0);
        srcn(k) = R8(0.0);
        cs(k) = pmid(k) / (R8(PC_RAIR) * temp(k));
        dz(k) = R8(1.0) / (cs(k) * R8(PC_GRAVIT) * rpdel(k));
        for (int m = 1; m <= MG_NMODE; ++m) nact[m](k) = R8(0.0);
        zn(k) = R8(PC_GRAVIT) * rpdel(k);
        if (k < pver) {
            ekd(k) = kvh(k + 1);
            ekd(k) = mg_max(ekd(k), zkmin);
            ekd(k) = mg_min(ekd(k), zkmax);
            csbot(k) = R8(2.0) * pint(k + 1) / (R8(PC_RAIR) * (temp(k) + temp(k + 1)));
            csbot_cscen(k) = csbot(k) / cs(k);
        } else {
            ekd(k) = R8(0.0);
            csbot(k) = cs(k);
            csbot_cscen(k) = R8(1.0);
        }
        if (k == pver) {
            wtke(k) = sq2pi * depvel;
            wtke(k) = mg_max(wtke(k), wmixmin);
        } else {
            wtke(k) = sq2pi * ekd(k) / dz(k);
        }
        wtke_cen(k) = wsub(k);
        wtke(k) = wsub(k);
        wtke_cen(k) = mg_max(wtke_cen(k), wmixmin);
        wtke(k) = mg_max(wtke(k), wmixmin);
    }
    int nsav = 1, nnew = 2;
    const R8 qqcw0 = R8(1.e-38);
    for (int m = 1; m <= MG_NMODE; ++m) {
        for (int k = 1; k <= pver; ++k) {
            raercol_cw[m][nsav](k) = qqcw0;
            raercol[m][nsav](k) = raer.q[md_numptr(m)];
        }
    }
    const R8 tau_cld_regenerate = R8(3600.0) * R8(3.0);
    R8 na[MG_NMODE + 1], va[MG_NMODE + 1], hy[MG_NMODE + 1];
    R8 fn[MG_NMODE + 1], fluxn[MG_NMODE + 1];

    // growing / shrinking cloud
    for (int k = 1; k <= pver; ++k) {
        R8 cldo_tmp = cldo(k);
        R8 cldn_tmp = cldn(k) * mg_exp(-dtmicro / tau_cld_regenerate);
        if (cldn_tmp < cldo_tmp) {
            qcld(k) = qcld(k) * (SP(1.) + (cldn_tmp - cldo_tmp) / cldo_tmp);
            R8 dumc = (cldn_tmp - cldo_tmp) / cldo_tmp;
            for (int m = 1; m <= MG_NMODE; ++m) {
                R8 dact = raercol_cw[m][nsav](k) * dumc;
                raercol_cw[m][nsav](k) = raercol_cw[m][nsav](k) + dact;
                raercol[m][nsav](k) = raercol[m][nsav](k) - dact;
            }
        }
        cldo_tmp = cldn_tmp;
        cldn_tmp = cldn(k);
        if (cldn_tmp - cldo_tmp > SP(0.01)) {
            R8 wbar = wtke_cen(k);
            for (int m = 1; m <= MG_NMODE; ++m) nd_loadaer(raer, cs(k), m, na[m], va[m], hy[m]);
            R8 ffa;
            nd_activate_modal(wbar, temp(k), cs(k), na, va, hy, fn, fluxn, ffa);
            flux_fullact(k) = ffa;
            R8 dumc = cldn_tmp - cldo_tmp;
            for (int m = 1; m <= MG_NMODE; ++m) {
                R8 dact = dumc * fn[m] * raer.q[md_numptr(m)];
                qcld(k) = qcld(k) + dact;
                raercol_cw[m][nsav](k) = raercol_cw[m][nsav](k) + dact;
                raercol[m][nsav](k) = raercol[m][nsav](k) - dact;
            }
        }
    }

    // old cloud activation
    R8 taumix_internal_pver_inv = R8(0.0);
    for (int k = 1; k <= pver; ++k) {
        int kp1 = k + 1 < pver ? k + 1 : pver;
        taumix_internal_pver_inv = SP(0.0);
        if (cldn(k) > SP(0.01)) {
            R8 wbar = wtke(k);
            if (k == pver) wbar = wtke_cen(k);
            if (cldn(k) - cldn(kp1) > SP(0.01) || k == pver) {
                ekd(k) = wtke(k) * dz(k) / sq2pi;
                ekd(k) = wbar / zs(k);
                for (int m = 1; m <= MG_NMODE; ++m) nd_loadaer(raer, cs(kp1), m, na[m], va[m], hy[m]);
                R8 ffa;
                nd_activate_modal(wbar, temp(k), cs(k), na, va, hy, fn, fluxn, ffa);
                flux_fullact(k) = ffa;
                R8 dumc = (k < pver) ? cldn(k) - cldn(kp1) : cldn(k);
                R8 dum = csbot_cscen(k) / dz(k);
                if (k == pver) taumix_internal_pver_inv = flux_fullact(k) / dz(k);
                for (int m = 1; m <= MG_NMODE; ++m) {
                    fluxn[m] = fluxn[m] * dumc;
                    nact[m](k) = nact[m](k) + fluxn[m] * dum;
                }
            }
        } else {
            qcld(k) = R8(0.0);
            for (int m = 1; m <= MG_NMODE; ++m) {
                raercol[m][nsav](k) = raercol[m][nsav](k) + raercol_cw[m][nsav](k);
                raercol_cw[m][nsav](k) = SP(0.);
            }
        }
    }
    { int t = nsav; nsav = nnew; nnew = t; }

    R8 dtmin = dtmicro;
    ekk(1) = SP(0.0);                  // ekk(0)
    ekk(pver + 1) = SP(0.0);           // ekk(pver)
    for (int k = 1; k <= pver - 1; ++k) ekk(k + 1) = ekd(k) * csbot(k);
    for (int k = 1; k <= pver; ++k) {
        int km1 = k - 1 > 1 ? k - 1 : 1;
        ekkp(k) = zn(k) * ekk(k + 1) * zs(k);
        ekkm(k) = zn(k) * ekk(k) * zs(km1);
        R8 tinv = ekkp(k) + ekkm(k);
        if (k == pver) tinv = tinv + R8(0.0);            // surfratemax = 0
        if (k == pver) tinv = tinv + taumix_internal_pver_inv;
        if (tinv > SP(1.e-6)) {
            R8 dtt = SP(1.) / tinv;
            dtmin = mg_min(dtmin, dtt);
        }
    }
    R8 dtmix = SP(0.9) * dtmin;
    // nsubmix = dtmicro/dtmix + 1, the REAL sum truncated on assignment.
    int nsubmix = uw_int(dtmicro / dtmix + R8(1.0));
    dtmix = dtmicro / uw_real(nsubmix);
    for (int k = 1; k <= pver; ++k) {
        int kp1 = k + 1 < pver ? k + 1 : pver;
        int km1 = k - 1 > 1 ? k - 1 : 1;
        if (cldn(kp1) > R8(1.e-10)) overlapp(k) = mg_min(cldn(k) / cldn(kp1), R8(1.0));
        else overlapp(k) = SP(1.);
        if (cldn(km1) > R8(1.e-10)) overlapm(k) = mg_min(cldn(k) / cldn(km1), R8(1.0));
        else overlapm(k) = SP(1.);
    }
    for (int k = 1; k <= pver - 1; ++k)
        for (int m = 1; m <= MG_NMODE; ++m) nact[m](k) = mg_min(nact[m](k), ekkp(k));

    for (int n = 1; n <= nsubmix; ++n) {
        for (int k = 1; k <= pver; ++k) qncld(k) = qcld(k);
        { int t = nsav; nsav = nnew; nnew = t; }
        for (int k = 1; k <= pver; ++k) srcn(k) = SP(0.0);
        for (int m = 1; m <= MG_NMODE; ++m) {
            for (int k = 1; k <= pver - 1; ++k)
                srcn(k) = srcn(k) + nact[m](k) * raercol[m][nsav](k + 1);
#ifndef CAMMGMP_CORRECTED_NDROP_DIAGNOSTIC
            R8 actold = raercol[m][nsav](pver);
#else
            R8 actold = raercol_cw[m][nsav](pver);
#endif
            R8 tmpa = raercol[m][nsav](pver) * nact[m](pver)
                      + actold * (nact[m](pver) - taumix_internal_pver_inv);
            srcn(pver) = srcn(pver) + mg_max(R8(0.0), tmpa);
        }
        nd_explmix(qcld, srcn, ekkp, ekkm, overlapp, overlapm, qncld, pver,
                   dtmix, false, qncld);
        for (int m = 1; m <= MG_NMODE; ++m) {
            for (int k = 1; k <= pver - 1; ++k)
                source(k) = nact[m](k) * raercol[m][nsav](k + 1);
#ifndef CAMMGMP_CORRECTED_NDROP_DIAGNOSTIC
            R8 actold = raercol[m][nsav](pver);
#else
            R8 actold = raercol_cw[m][nsav](pver);
#endif
            R8 tmpa = raercol[m][nsav](pver) * nact[m](pver)
                      + actold * (nact[m](pver) - taumix_internal_pver_inv);
            source(pver) = mg_max(R8(0.0), tmpa);
            nd_explmix(raercol_cw[m][nnew], source, ekkp, ekkm, overlapp, overlapm,
                       raercol_cw[m][nsav], pver, dtmix, false, raercol_cw[m][nsav]);
            nd_explmix(raercol[m][nnew], source, ekkp, ekkm, overlapp, overlapm,
                       raercol[m][nsav], pver, dtmix, true, raercol_cw[m][nsav]);
        }
    }
    for (int k = 1; k <= pver; ++k)
        if (cldn(k) == SP(0.)) qcld(k) = SP(0.);
    for (int k = 1; k <= pver; ++k)
        tendnd(k) = (mg_max(qcld(k), R8(1.e-6)) - ncldwtr(k)) * dtinv;
}

// microp_aero hetero (module_cam_mp_microp_aero.F:1396-1427).
__device__ void ma_hetero(R8 T, R8 ww, R8 Ns, R8& Nis, R8& Nid) {
    const R8 A11(0.0263), A12(-0.0185), A21(2.758), A22(1.3221);
    const R8 B11(-0.008), B12(-0.0468), B21(-0.2667), B22(-1.4588);
    R8 B = (A11 + B11 * mg_log(Ns)) * mg_log(ww) + (A12 + B12 * mg_log(Ns));
    R8 C = A21 + B21 * mg_log(Ns);
    Nis = mg_exp(A22) * mg_pow(Ns, B22) * mg_exp(B * T) * mg_pow(ww, C);
    Nis = mg_min(Nis, Ns);
    Nid = R8(0.0);
}

// microp_aero hf (module_cam_mp_microp_aero.F:1430-1506).
__device__ R8 ma_hf(R8 T, R8 ww, R8 RH, R8 subgrid, R8 Na) {
    const R8 A1_fast(0.0231), A21_fast(-1.6387), A22_fast(-6.045);
    const R8 B1_fast(-0.008), B21_fast(-0.042), B22_fast(-0.112);
    const R8 C1_fast(0.0739), C2_fast(1.2372);
    const R8 A1_slow(-0.3949), A2_slow(1.282), B1_slow(-0.0156);
    const R8 B2_slow(0.0111), B3_slow(0.0217), C1_slow(0.120), C2_slow(2.312);
    R8 Ni = R8(0.0);
    R8 A = R8(6.0e-4) * mg_log(ww) + R8(6.6e-3);
    R8 B = R8(6.0e-2) * mg_log(ww) + R8(1.052);
    R8 C = R8(1.68) * mg_log(ww) + R8(129.35);
    R8 RHw = (A * T * T + B * T + C) * R8(0.01);
    if ((T <= R8(-37.0)) && ((RH * subgrid) >= RHw)) {
        R8 regm = R8(6.07) * mg_log(ww) - R8(55.0);
        if (T >= regm) {
            R8 A2_fast, B2_fast;
            if (T > R8(-64.0)) { A2_fast = A21_fast; B2_fast = B21_fast; }
            else { A2_fast = A22_fast; B2_fast = B22_fast; }
            R8 k1_fast = mg_exp(A2_fast + B2_fast * T + C2_fast * mg_log(ww));
            R8 k2_fast = A1_fast + B1_fast * T + C1_fast * mg_log(ww);
            Ni = k1_fast * mg_pow(Na, k2_fast);
            Ni = mg_min(Ni, Na);
        } else {
            R8 k1_slow = mg_exp(A2_slow + (B2_slow + B3_slow * mg_log(ww)) * T + C2_slow * mg_log(ww));
            R8 k2_slow = A1_slow + B1_slow * T + C1_slow * mg_log(ww);
            Ni = k1_slow * mg_pow(Na, k2_slow);
            Ni = mg_min(Ni, Na);
        }
    }
    return Ni;
}

// microp_aero nucleati (module_cam_mp_microp_aero.F:1147-1394), MODAL_AERO
// with MODAL_AERO_3MODE.  qaerpt is qaer(i,k,:)*rhoair; dgnum(aitken) is the
// driver's default-REAL 0.02e-6.  Returns nuci (#/kg) and onihf.
__device__ void ma_nucleati(R8 wbar, R8 tair, R8 relhum, R8 cldn, R8 qc,
                            R8 rhoair, const R8* qaerpt, R8 dgnum_aitken,
                            R8& nuci, R8& onihf MG_SAT_POINT_DECL) {
    R8 so4_num = R8(0.0), soot_num = R8(0.0), dst_num = R8(0.0);
    soot_num = qaerpt[7] * R8(1.0e-6);
    R8 dmc = qaerpt[10];
    R8 ssmc = qaerpt[11];
    if (dmc > R8(0.0)) {
        dst_num = dmc / (ssmc + dmc) * qaerpt[12] * R8(1.0e-6);
    } else {
        dst_num = R8(0.0);
    }
    if (dgnum_aitken > R8(0.0)) {
        so4_num = qaerpt[9] * R8(1.0e-6)
                  * (R8(0.5) - R8(0.5) * cam_erf(mg_log(R8(0.1e-6) / dgnum_aitken)
                                                 / (R8(0x1.6a09e667f3bcdp+0) * mg_log(R8(MD_SIGMAG2)))));
    } else {
        so4_num = R8(0.0);
    }
    so4_num = mg_max(R8(0.0), so4_num);
    soot_num = R8(0.0);
    R8 ni = R8(0.0);
    R8 tc = tair - R8(273.15);
    R8 niimm = R8(0.0), nidep = R8(0.0), nihf = R8(0.0), n1;
    if (so4_num >= R8(1.0e-10) && (soot_num + dst_num) >= R8(1.0e-10) && cldn > R8(0.0)) {
        R8 subgrid = R8(1.2);
        if ((tc <= R8(-35.0))
            && ((relhum * MG_SAT(sat_k, tair, 0) / MG_SAT(sat_k, tair, 1) * subgrid) >= R8(1.2))) {
            R8 A = -(R8(1.4938) * mg_log(soot_num + dst_num)) + R8(12.884);
            R8 B = -(R8(10.41) * mg_log(soot_num + dst_num)) - R8(67.69);
            R8 regm = A * mg_log(wbar) + B;
            if (tc > regm) {
                if (tc < R8(-40.) && wbar > R8(1.)) {
                    nihf = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                    niimm = R8(0.); nidep = R8(0.);
                    n1 = nihf;
                } else {
                    ma_hetero(tc, wbar, soot_num + dst_num, niimm, nidep);
                    nihf = R8(0.);
                    n1 = niimm + nidep;
                }
            } else if (tc < regm - R8(5.)) {
                nihf = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                niimm = R8(0.); nidep = R8(0.);
                n1 = nihf;
            } else {
                if (tc < R8(-40.) && wbar > R8(1.)) {
                    nihf = ma_hf(tc, wbar, relhum, subgrid, so4_num);
                    niimm = R8(0.); nidep = R8(0.);
                    n1 = nihf;
                } else {
                    nihf = ma_hf(regm - R8(5.), wbar, relhum, subgrid, so4_num);
                    ma_hetero(regm, wbar, soot_num + dst_num, niimm, nidep);
                    if (nihf <= (niimm + nidep)) {
                        n1 = nihf;
                    } else {
                        n1 = (niimm + nidep) * mg_pow((niimm + nidep) / nihf, (tc - regm) / R8(5.));
                    }
                }
            }
            ni = n1;
        }
    }
    R8 nimey;
    if (tc < R8(0.) && tc > R8(-37.) && qc > R8(1.e-12)) {
        R8 esl = MG_SAT(sat_k, tair, 0);
        R8 esi = MG_SAT(sat_k, tair, 1);
        R8 deles = (esl - esi);
        nimey = R8(1.e-3) * mg_exp(R8(12.96) * deles / esi - R8(0.639));
    } else {
        nimey = R8(0.);
    }
    nuci = ni + nimey;
    if (nuci > R8(9999.) || nuci < R8(0.)) nuci = R8(0.);
    nuci = nuci * R8(1.e+6) / rhoair;
    onihf = nihf * R8(1.e+6) / rhoair;
}

// microp_aero_ts (module_cam_mp_microp_aero.F:381-893), MODAL_AERO,
// wsubTKE = .true. (the WRF_PORT setting).  Fills naai, npccn, rndst(3)
// and nacon(3); rndst(1,2,4) stay rn_dst1/2/4 and nacon(1,2,4) zero, which
// mmicro_pcond reads as the constants they are.  tkem/smm (computed from
// turbtype/smaw) reach no output and are not formed.
__device__ void ma_microp_aero_ts(int pver, R8 deltatin, V t, V qn, V qc, V qi,
                                  V nc, V p, V rpdel, V cldn, V liqcldf,
                                  V icecldf, V cldo, V pint, V zm,
                                  const MgAerosol& qaer, V kkvh, V tke,
                                  V naai, V npccn, V rndst3, V nacon3,
                                  R8 dgnum_aitken, R8 dgnumwet_coarse, Ws& ws MG_SAT_DECL) {
    WsMark mark(ws);
    V wsub = ws.r8(pver), wsubi = ws.r8(pver), rho = ws.r8(pver);
    V lcldn = ws.r8(pver), lcldo = ws.r8(pver), ncloc = ws.r8(pver);
    V nctend_mixnuc = ws.r8(pver);
    const R8 mincld = R8(0.0001);
    for (int k = 1; k <= pver; ++k) {
        naai(k) = R8(0.0);
        npccn(k) = R8(0.0);
        nacon3(k) = R8(0.0);
        rndst3(k) = R8(MA_RN_DST3);
        ncloc(k) = nc(k);
        rho(k) = p(k) / (R8(MA_R) * t(k));
    }
    for (int k = 1; k <= pver; ++k) {
        R8 dum = (kkvh(k) + kkvh(k + 1)) / R8(2.0) / R8(30.0);
        dum = mg_min(dum, R8(10.0));
        wsub(k) = dum;
        wsub(k) = mg_sqrt(R8(0.5) * (tke(k) + tke(k + 1)) * R8(0x1.5555555555555p-1));
        wsub(k) = mg_min(wsub(k), R8(10.0));
        wsubi(k) = mg_max(R8(0.001), wsub(k));
        wsubi(k) = mg_min(wsubi(k), R8(0.2));
        wsub(k) = mg_max(R8(0.20), wsub(k));
    }
    for (int k = 1; k <= pver; ++k) {
        R8 es, qs, gam;
        mg_vqsatd_water(t(k), p(k), es, qs, gam MG_SAT_POINT_ARG(k));
        R8 relhum = qn(k) / qs;
        R8 icldm = mg_max(icecldf(k), mincld);
        R8 nfice = R8(0.0);
        R8 dumfice = qc(k) + qi(k);
        if (dumfice > R8(MA_QSMALL) && qi(k) > R8(MA_QSMALL)) nfice = qi(k) / dumfice;
        (void)nfice;
        if (t(k) < R8(PC_TMELT) - R8(5.0)) {
            R8 qaerpt[13];
            for (int l = 1; l <= 12; ++l) qaerpt[l] = qaer.q[l] * rho(k);
            R8 dum2, nihf2;
            ma_nucleati(wsubi(k), t(k), relhum, icldm, qc(k), rho(k), qaerpt,
                        dgnum_aitken, dum2, nihf2 MG_SAT_POINT_ARG(k));
            naai(k) = dum2;
        }
    }
    for (int k = 1; k <= pver; ++k) {
        R8 qcld = qc(k) + qi(k);
        if (qcld > R8(MA_QSMALL)) {
            lcldn(k) = cldn(k) * qc(k) / qcld;
            lcldo(k) = cldo(k) * qc(k) / qcld;
        } else {
            lcldn(k) = R8(0.0);
            lcldo(k) = R8(0.0);
        }
    }
    nd_dropmixnuc(pver, ncloc, nctend_mixnuc, t, p, pint, rpdel, zm, kkvh, wsub,
                  lcldn, lcldo, qaer, deltatin, ws);
    for (int k = 1; k <= pver; ++k) npccn(k) = nctend_mixnuc(k);
    for (int k = 1; k <= pver; ++k) {
        if (t(k) < R8(269.15)) {
            R8 dmc = qaer.q[10];
            R8 ssmc = qaer.q[11];
            if (dmc > R8(0.0)) {
                nacon3(k) = dmc / (ssmc + dmc) * qaer.q[12] * rho(k);
            } else {
                nacon3(k) = R8(0.0);
            }
            rndst3(k) = R8(0.5) * dgnumwet_coarse;
            if (rndst3(k) <= R8(0.0)) rndst3(k) = R8(MA_RN_DST3);
        }
    }
}

#endif  // CAMMGMP_AERO_CUH
