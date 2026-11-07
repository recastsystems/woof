// cammgmp.cu -- WRF v4.6.1 mp_physics = 11, CAM5.1 Morrison-Gettelman
// two-moment stratiform microphysics (CAMMGMP), one thread per column.
// ArWen's own work, Apache-2.0.
//
// The loader (gpuwm/core/kernels/__init__.py _EXTRA_HEADERS["cammgmp"])
// prepends, in order: glibc_flt64.cuh, uwpbl_common.cuh, uwpbl_wvsat.cuh,
// cammgmp_words.cuh, cammgmp_support.cuh, cammgmp_aero.cuh,
// cammgmp_micro.cuh.  This file is phys/module_mp_cammgmp_driver.F: the
// CAMMGMP column body (:97-1397) with Macrophysics_simple (:1747-2099),
// cal_cldfra_mp_1d (:1651-1743), findsp_water (:2169-2490) and the
// physics_update / ptend helpers (:1403-1647).
//
// Layout.  Engine order (k = 1 at the bottom), level-major with the column
// index fastest: element k of column c is base[(k-1)*ncols + c]; surface
// fields are base[c].  Inside, CAM order (kcam = 1 at the top) on the
// per-column workspace pool (uwpbl_common.cuh Ws).
//
// Status words written to *err (the launcher refuses the step by name):
//   1 workspace pool overflow
//   2 MG GAMMA reached its reflection branch (argument <= 0, never in MG)
//   3 findsp_water did not converge     (WRF: wrf_error_fatal('FINDSP'))
//   4 findsp_water lost enthalpy > 1e-4 (WRF: wrf_error_fatal('FINDSP'))

struct F32V {                           // engine float32 column, 1-based WRF k
    float* p;
    long long s;
    __device__ __forceinline__ float& operator()(int k) const { return p[(k - 1) * s]; }
};

__device__ __forceinline__ R8 w8(float x) { return R8((double)x); }
__device__ __forceinline__ float n4(R8 x) { return __double2float_rn(x.v); }

// Macrophysics/microphysics tendency set: s plus the five water species.
struct MgPtend {
    V s, q1, q2, q3, q4, q5;
};

__device__ void mg_ptend_zero(const MgPtend& p, int pver) {
    for (int k = 1; k <= pver; ++k) {
        p.s(k) = R8(0.0); p.q1(k) = R8(0.0); p.q2(k) = R8(0.0);
        p.q3(k) = R8(0.0); p.q4(k) = R8(0.0); p.q5(k) = R8(0.0);
    }
}

// physics_ptend_sum (:1592-1647) for the flags the CAMMGMP calls set.
__device__ void mg_ptend_sum(const MgPtend& loc, const MgPtend& all, int pver,
                             bool lq3, bool lq5) {
    for (int k = 1; k <= pver; ++k) all.s(k) = all.s(k) + loc.s(k);
    for (int k = 1; k <= pver; ++k) all.q1(k) = all.q1(k) + loc.q1(k);
    for (int k = 1; k <= pver; ++k) all.q2(k) = all.q2(k) + loc.q2(k);
    if (lq3) for (int k = 1; k <= pver; ++k) all.q3(k) = all.q3(k) + loc.q3(k);
    for (int k = 1; k <= pver; ++k) all.q4(k) = all.q4(k) + loc.q4(k);
    if (lq5) for (int k = 1; k <= pver; ++k) all.q5(k) = all.q5(k) + loc.q5(k);
}

// physics_update (:1403-1527): s += ptend_s*dt; each flagged constituent
// q += ptend*dt, then qneg3 (q < qmin -> qmin) for the masses and the
// [1e-12, 1e10] clip for the numbers (NUMLIQ = 4, NUMICE = 5).
__device__ void mg_physics_update(V s, V q1, V q2, V q3, V q4, V q5,
                                  const MgPtend& pt, int pver, R8 dt,
                                  bool lq3, bool lq5) {
    for (int k = 1; k <= pver; ++k) s(k) = s(k) + pt.s(k) * dt;
    for (int k = 1; k <= pver; ++k) q1(k) = q1(k) + pt.q1(k) * dt;
    for (int k = 1; k <= pver; ++k) if (q1(k) < R8(UW_QMIN_Q)) q1(k) = R8(UW_QMIN_Q);
    for (int k = 1; k <= pver; ++k) q2(k) = q2(k) + pt.q2(k) * dt;
    for (int k = 1; k <= pver; ++k) if (q2(k) < R8(0.0)) q2(k) = R8(0.0);
    if (lq3) {
        for (int k = 1; k <= pver; ++k) q3(k) = q3(k) + pt.q3(k) * dt;
        for (int k = 1; k <= pver; ++k) if (q3(k) < R8(0.0)) q3(k) = R8(0.0);
    }
    for (int k = 1; k <= pver; ++k) {
        q4(k) = q4(k) + pt.q4(k) * dt;
        q4(k) = mg_max(R8(1.e-12), q4(k));
        q4(k) = mg_min(R8(1.e10), q4(k));
    }
    if (lq5) {
        for (int k = 1; k <= pver; ++k) {
            q5(k) = q5(k) + pt.q5(k) * dt;
            q5(k) = mg_max(R8(1.e-12), q5(k));
            q5(k) = mg_min(R8(1.e10), q5(k));
        }
    }
}

// The stratus critical relative humidity of Macrophysics_simple and
// cal_cldfra_mp_1d (:1680-1692, :1858-1870).
__device__ __forceinline__ R8 mg_cldcr(R8 pp, R8 xland_pt, R8 snowh_pt) {
    R8 cldcr;
    if (pp >= R8(70000.)) {
        if (xland_pt < R8(1.1) && snowh_pt < R8(0.001)) cldcr = R8(0.7875);
        else cldcr = R8(0.8875);
    } else if (pp <= R8(40000.)) {
        cldcr = R8(0.8);
    } else {
        // (0.8875_r8 - 0.8_r8)/30000._r8 is a folded constant subexpression
        cldcr = R8(0x1.8777e75094fbcp-19) * (pp - R8(40000.)) + R8(0.8);
    }
    return mg_min(mg_max(R8(0.0), cldcr), R8(1.0));
}

// cal_cldfra_mp_1d (:1651-1743).
__device__ void mg_cal_cldfra(R8& cldl, R8& cldi, R8 qqv, R8 qqc, R8 qqi,
                              R8 tt, R8 pp, R8 xland_pt, R8 snowh_pt MG_SAT_POINT_DECL) {
#ifdef GPUWM_CAM_FAST
    if (MgSaturation::same(sat.at(4, sat_k), tt)
        && MgSaturation::same(sat.at(5, sat_k), qqv)
        && MgSaturation::same(sat.at(6, sat_k), qqc)) {
        cldl = sat.at(7, sat_k); cldi = sat.at(8, sat_k);
        return;
    }
#endif
    R8 qv = mg_max(qqv, R8(0.0));
    R8 qc = mg_max(qqc, R8(0.0));
    R8 qi = mg_max(qqi, R8(0.0));
    cldi = R8(0.0);
    cldl = R8(0.0);
    const R8 cldrh = R8(1.0);
    R8 cldcr = mg_cldcr(pp, xland_pt, snowh_pt);
    R8 dV = cldrh - cldcr;
    R8 esl = MG_SAT(sat_k, tt, 0);
    R8 qvl = mg_max(R8(1.e-12), R8(WV_EPSQS) * esl / (pp - (R8(1.0) - R8(WV_EPSQS)) * esl));
    R8 rhl = (qv + qc) / qvl;
    rhl = mg_min(R8(1.1), mg_max(R8(0.0), rhl));
    if (rhl >= R8(1.0)) {
        cldl = R8(1.0);
    } else if (rhl > (cldrh - dV / R8(6.)) && rhl < R8(1.0)) {
        // -3._r8/sqrt(2._r8)*(rhl-cldrh)/dV, 3/sqrt(2) folded
        cldl = R8(1.0) - mg_pow(-(R8(0x1.0f876ccdf6cd9p+1) * (rhl - cldrh) / dV),
                                R8(0x1.5555555555555p-1));
    } else if (rhl > (cldrh - dV) && rhl <= (cldrh - dV / R8(6.))) {
        // 4*(cos((1/3)*(acos((3/2/sqrt 2)*(1+(rhl-cldrh)/dV)) - 2*3.14159)))**2._r8
        R8 c = uw_cos(R8(0x1.5555555555555p-2)
                      * (uw_acos(R8(0x1.0f876ccdf6cd9p+0) * (R8(1.0) + (rhl - cldrh) / dV))
                         - R8(0x1.921f9f01b866ep+2)));
        cldl = R8(4.) * (c * c);
    } else if (rhl <= (cldrh - dV)) {
        cldl = R8(0.0);
    }
    R8 esi = MG_SAT(sat_k, tt, 1);
    if (tt > R8(PC_TMELT)) esi = esl;
    R8 qvi = mg_max(R8(1.e-12), R8(WV_EPSQS) * esi / (pp - (R8(1.0) - R8(WV_EPSQS)) * esi));
    R8 rhi = (qv + qi) / qvi;
    rhi = mg_min(R8(1.1), mg_max(R8(0.0), rhi));
    if (tt < R8(273.15)) {
        R8 rhdif = (rhi - R8(0.8)) / R8(0.3);
        R8 m = mg_max(rhdif, R8(0.0));
        cldi = mg_min(R8(1.0), m * m);
        if (qi < R8(1.e-12)) {
            cldi = R8(0.0);
        } else {
            cldi = mg_max(R8(1.e-4), cldi);
        }
        if (qi >= R8(1.e-12)) {
            R8 icimr = qi / (mg_max(cldi, R8(1.e-20)));
            if (icimr < R8(1.e-7)) cldi = mg_max(R8(0.0), mg_min(R8(1.0), qi / R8(1.e-7)));
            if (icimr > R8(5.e-3)) cldi = mg_max(R8(0.0), mg_min(R8(1.0), qi / R8(5.e-3)));
        }
    }
    cldl = mg_min(mg_max(R8(0.0), cldl), R8(1.0));
    cldi = mg_min(mg_max(R8(0.0), cldi), R8(1.0));
#ifdef GPUWM_CAM_FAST
    sat.at(4, sat_k) = tt; sat.at(5, sat_k) = qqv; sat.at(6, sat_k) = qqc;
    sat.at(7, sat_k) = cldl; sat.at(8, sat_k) = cldi;
#endif
}

// findsp_water (:2169-2490), one point.  Uses wv_saturation's hlatv, cp,
// rgasv, tmin and epsqs; its own rair = 287.04 and a1 = 7.5*log(10).
__device__ void mg_findsp_water(R8 q, R8 t, R8 p, R8& tsp, R8& qsp, int* err MG_SAT_POINT_DECL) {
    const R8 epsqs(WV_EPSQS), hlatv(WV_HLATV), cp(WV_CP), rgasv(WV_RGASV);
    const R8 omeps = R8(1.0) - epsqs;
    const R8 a1 = R8(0x1.144f69ff9ffc5p+4);      // 7.5_r8*log(10._r8), folded
    const R8 rair = R8(287.04);
    const R8 c3 = rair * a1 / cp;
    R8 dtm = R8(0.0), dqm = R8(0.0);
    const R8 dttol = R8(1.e-4), dqtol = R8(1.e-4);
    const R8 tt0 = R8(273.15);
    const int iter = 8;
    R8 tlim = mg_min(mg_max(t, R8(128.)), R8(373.));
    R8 es = MG_SAT(sat_k, tlim, 0);
    R8 denom = p - omeps * es;
    R8 qs = epsqs * es / denom;
    int doit = 0;
    R8 enout = R8(1.0), enin, hltalt;
    if (p > R8(5.) * es && qs > R8(0.) && qs < R8(0.5)) {
        qs = mg_min(epsqs * es / denom, R8(1.0));
        R8 hlatvp = hlatv - SP(2369.0) * (tlim - tt0);
        R8 hlatsb = hlatv;
        hltalt = (tlim < tt0) ? hlatsb : hlatvp;
        enin = cp * tlim + hltalt * q;
        R8 tmp = q - qs;
        R8 c1 = hltalt * c3;
        R8 c2 = ip2(tlim + R8(36.));
        R8 r1b = c2 / (c2 + c1 * qs);
        R8 qvd = r1b * tmp;
        tsp = tlim + ((hltalt / cp) * qvd);
        es = MG_SAT(sat_k, tsp, 0);
        qsp = mg_min(epsqs * es / (p - omeps * es), R8(1.0));
    } else {
        doit = 1;
        tsp = tlim;
        qsp = q;
        enin = R8(1.0);
    }
    for (int l = 1; l <= iter; ++l) {
        dtm = R8(0.0);
        dqm = R8(0.0);
        if (doit == 0) {
            es = MG_SAT(sat_k, tsp, 0);
            qs = mg_min(epsqs * es / (p - omeps * es), R8(1.0));
            R8 hlatvp = hlatv - SP(2369.0) * (tsp - tt0);
            R8 hlatsb = hlatv;
            hltalt = (tsp < tt0) ? hlatsb : hlatvp;
            R8 desdt = hltalt * es / (rgasv * tsp * tsp);
            R8 dqsdt = (epsqs + omeps * qs) / (p - omeps * es) * desdt;
            R8 g = enin - (cp * tsp + hltalt * qsp);
            R8 dgdt = -(cp + hltalt * dqsdt);
            R8 t1 = tsp - g / dgdt;
            R8 dt = uw_abs(t1 - tsp) / t1;
            tsp = mg_max(t1, R8(WV_TMIN));
            es = MG_SAT(sat_k, tsp, 0);
            R8 q1 = mg_min(epsqs * es / (p - omeps * es), R8(1.0));
            R8 dq = uw_abs(q1 - qsp) / mg_max(q1, R8(1.e-12));
            qsp = q1;
            dtm = mg_max(dtm, dt);
            dqm = mg_max(dqm, dq);
            if (dt < dttol && dq < dqtol) doit = 2;
            enout = cp * tsp + hltalt * qsp;
            if (tsp < R8(174.16)) doit = 4;
        }
        if (dtm < dttol && dqm < dqtol) break;
    }
    if (dtm > dttol || dqm > dqtol) {
        if (doit == 0) *err = 3;
    }
    if (doit == 2 && uw_abs((enin - enout) / (enin + enout)) > R8(1.e-4)) *err = 4;
}

struct MgColumnIn {
    F32V p8w, p_hyd, t_phy, pi_phy, z, tke, kvh, alt;
    F32V icwmrsh, icwmrdp, shfrc, cmfmc, cmfmc2;
    float ht, xland, snowh;
    int nz, itimestep, cu_coupled, shcu_coupled;
    float dt, accum, aitken, coarse;
};
struct MgColumnOut {   // inout fields carry the inputs on entry
    F32V th, qv, qc, qi, qs, qr, nc, ni, ns, nr, qndrop;
    F32V cldfra_old_mp, cldfra_mp, cldfra_mp_all, lradius, iradius;
    F32V cldfrai, cldfral, cldfra_conv, wsedl3d, rh_old_mp, lcd_old_mp;
    float* rainnc; float* rainncv; float* snownc; float* snowncv; float* sr;
};

__device__ void cammgmp_column(const MgColumnIn& in, MgColumnOut& out, Ws& ws,
                               int* err) {
    const int kte = in.nz;
    const int pver = kte;
    const R8 gravit(PC_GRAVIT), cpair(MG_CPP), latvap(MG_XXLV);
    const R8 epsqs(WV_EPSQS);
    const R8 qthresh_mass = SP(1.0e-11);
    const R8 qthresh_numl = SP(5.0e2);
    const R8 qthresh_numi = SP(1.0e-1);
    const R8 dp1 = R8(0.10), dp2 = R8(500.0);
    const R8 dtime = w8(in.dt);
    const bool is_first_step = (in.itimestep == 1);
    const R8 xland_pt = w8(in.xland), snowh_pt = w8(in.snowh);
    int gbad = 0;

    // driver locals (CAM order, kcam = 1 at the top)
    V pmid = ws.r8(pver), pdel = ws.r8(pver), rpdel = ws.r8(pver), zm = ws.r8(pver);
    V st_t = ws.r8(pver), st_s = ws.r8(pver);
    V sq1 = ws.r8(pver), sq2 = ws.r8(pver), sq3 = ws.r8(pver), sq4 = ws.r8(pver), sq5 = ws.r8(pver);
    V s1_t = ws.r8(pver), s1_s = ws.r8(pver);
    V s1q1 = ws.r8(pver), s1q2 = ws.r8(pver), s1q3 = ws.r8(pver), s1q4 = ws.r8(pver), s1q5 = ws.r8(pver);
    V qi_mac = ws.r8(pver), cldo = ws.r8(pver);
    V sh_icwmr = ws.r8(pver), sh_frac = ws.r8(pver), dp_frac = ws.r8(pver), dp_icwmr = ws.r8(pver);
    V pint = ws.r8(pver + 1), tke = ws.r8(pver + 1), kkvh = ws.r8(pver + 1);
    V cmfmc = ws.r8(pver + 1), cmfmc2 = ws.r8(pver + 1);
    V ast = ws.r8(pver), aist = ws.r8(pver), alst = ws.r8(pver), cv_frac = ws.r8(pver);
    MgPtend loc{ws.r8(pver), ws.r8(pver), ws.r8(pver), ws.r8(pver), ws.r8(pver), ws.r8(pver)};
    MgPtend all{ws.r8(pver), ws.r8(pver), ws.r8(pver), ws.r8(pver), ws.r8(pver), ws.r8(pver)};
    V naai = ws.r8(pver), npccn = ws.r8(pver), rndst3 = ws.r8(pver), nacon3 = ws.r8(pver);
    MgMicroOut mo;
    mo.tlat = ws.r8(pver); mo.qvlat = ws.r8(pver); mo.qctend = ws.r8(pver);
    mo.qitend = ws.r8(pver); mo.nctend = ws.r8(pver); mo.nitend = ws.r8(pver);
    mo.effc = ws.r8(pver); mo.effi = ws.r8(pver); mo.qrout = ws.r8(pver);
    mo.qsout = ws.r8(pver); mo.nrout = ws.r8(pver); mo.nsout = ws.r8(pver);
    mo.vtrmc = ws.r8(pver);

#ifdef GPUWM_CAM_FAST
    MgSaturation sat{ws.r8(9 * pver), pver};
    sat.init();
#endif

    // ---- first-step state (:607-621): zeroed before the column loop ----
    if (is_first_step) {
        for (int kw = 1; kw <= kte; ++kw) {
            out.cldfra_old_mp(kw) = 0.0f;
            out.cldfrai(kw) = 0.0f;
            out.cldfral(kw) = 0.0f;
            out.cldfra_mp(kw) = 0.0f;
            out.cldfra_mp_all(kw) = 0.0f;
            out.cldfra_conv(kw) = 0.0f;
        }
    }

    // ---- map WRF to CAM (:648-799) ----
    const R8 phis = w8(in.ht) * gravit;
    for (int kw = 1; kw <= kte; ++kw) {
        const int kf = kte + 1 - kw;
        pmid(kf) = w8(in.p_hyd(kw));
        const R8 dp = w8(__fsub_rn(in.p8w(kw), in.p8w(kw + 1)));
        pdel(kf) = dp;
        rpdel(kf) = R8(1.0) / dp;
        zm(kf) = w8(__fsub_rn(in.z(kw), in.ht));
        st_t(kf) = w8(in.t_phy(kw));
        st_s(kf) = cpair * w8(out.th(kw)) * w8(in.pi_phy(kw)) + gravit * zm(kf) + phis;
        const R8 multFrc = R8(1.0) / (R8(1.0) + w8(out.qv(kw)));
        sq1(kf) = w8(out.qv(kw)) * multFrc;
        sq2(kf) = w8(out.qc(kw)) * multFrc;
        sq3(kf) = w8(out.qi(kw)) * multFrc;
        sq4(kf) = w8(out.nc(kw)) * multFrc;
        sq5(kf) = w8(out.ni(kw)) * multFrc;
        qi_mac(kf) = w8(out.qi(kw)) * multFrc;
        if (uw_abs(sq1(kf)) < qthresh_mass && sq1(kf) < R8(UW_QMIN_Q)) sq1(kf) = R8(UW_QMIN_Q);
        if (uw_abs(sq2(kf)) < qthresh_mass && sq2(kf) < R8(0.0)) sq2(kf) = R8(0.0);
        if (uw_abs(sq3(kf)) < qthresh_mass && sq3(kf) < R8(0.0)) sq3(kf) = R8(0.0);
        if (uw_abs(sq4(kf)) < qthresh_numl && sq4(kf) < R8(0.0)) sq4(kf) = R8(0.0);
        if (uw_abs(sq5(kf)) < qthresh_numi && sq5(kf) < R8(0.0)) sq5(kf) = R8(0.0);
        if (sq1(kf) < R8(0.0)) sq1(kf) = R8(UW_QMIN_Q);
        if (sq2(kf) < R8(0.0)) sq2(kf) = R8(0.0);
        if (sq3(kf) < R8(0.0)) sq3(kf) = R8(0.0);
        if (sq4(kf) < R8(0.0)) sq4(kf) = R8(0.0);
        if (sq5(kf) < R8(0.0)) sq5(kf) = R8(0.0);
        cldo(kf) = w8(out.cldfra_mp_all(kw));
        sh_icwmr(kf) = R8(0.0);
        sh_frac(kf) = R8(0.0);
        dp_frac(kf) = R8(0.0);
        if (in.shcu_coupled) {
            sh_icwmr(kf) = w8(in.icwmrsh(kw));
            sh_frac(kf) = w8(in.shfrc(kw));
        }
        dp_icwmr(kf) = R8(0.0);
        if (in.cu_coupled) dp_icwmr(kf) = w8(in.icwmrdp(kw));
    }
    for (int kw = 1; kw <= kte + 1; ++kw) {
        const int kf = kte - kw + 2;
        pint(kf) = w8(in.p8w(kw));
        tke(kf) = w8(in.tke(kw));
        kkvh(kf) = w8(in.kvh(kw));
        cmfmc(kf) = R8(0.0);
        cmfmc2(kf) = R8(0.0);
        if (in.shcu_coupled || in.cu_coupled) cmfmc(kf) = w8(in.cmfmc(kw));
        if (in.shcu_coupled) cmfmc2(kf) = w8(in.cmfmc2(kw));
    }
    for (int kc = 1; kc <= kte; ++kc)
        dp_frac(kc) = mg_max(R8(0.0), mg_min(dp1 * mg_log(R8(1.0) + dp2 * (cmfmc(kc + 1) - cmfmc2(kc + 1))),
                                             R8(0.60)));
    for (int k = 1; k <= pver; ++k) {
        s1_t(k) = st_t(k); s1_s(k) = st_s(k);
        s1q1(k) = sq1(k); s1q2(k) = sq2(k); s1q3(k) = sq3(k); s1q4(k) = sq4(k); s1q5(k) = sq5(k);
    }
    mg_ptend_zero(loc, pver);
    mg_ptend_zero(all, pver);
    if (is_first_step) {
        for (int k = 1; k <= pver + 1; ++k) { kkvh(k) = R8(0.0); tke(k) = R8(0.0); }
    }

    // ---- Macrophysics_simple (:1747-2099) ----
    {
        const R8 factcc = R8(0.1), fact_adj = R8(0.35);
        for (int k = 1; k <= pver; ++k) {
            R8 cldcr = mg_cldcr(pmid(k), xland_pt, snowh_pt);
            R8 cl, ci;
            mg_cal_cldfra(cl, ci, s1q1(k), s1q2(k), qi_mac(k), s1_t(k), pmid(k), xland_pt, snowh_pt MG_SAT_POINT_ARG(k));
            alst(k) = cl; aist(k) = ci;
            R8 del_qq = R8(0.0);
            cv_frac(k) = mg_max(R8(0.0), mg_min(R8(0.8), dp_frac(k) + sh_frac(k)));
            alst(k) = (R8(1.0) - cv_frac(k)) * alst(k);
            aist(k) = (R8(1.0) - cv_frac(k)) * aist(k);
            ast(k) = mg_max(alst(k), aist(k));
            R8 st_frac = mg_min(R8(1.0), ast(k) + cv_frac(k));
            R8 esl = MG_SAT(k, s1_t(k), 0);
            R8 qvs = mg_max(R8(1.e-30), epsqs * esl / (pmid(k) - (R8(1.0) - epsqs) * esl));
            R8 rh_curr = mg_max(R8(0.0), s1q1(k) / qvs);
            R8 qc_nc = s1q2(k);
            if (is_first_step) cldo(k) = mg_min(R8(1.0), ast(k) + cv_frac(k));
            if (rh_curr > R8(1.0)) {
                R8 tsp, qsll;
                mg_findsp_water(s1q1(k), s1_t(k), pmid(k), tsp, qsll, err MG_SAT_POINT_ARG(k));
                del_qq = mg_min(R8(3.e-3), mg_max(R8(0.0), (s1q1(k) - qsll) * R8(0.999)));
                loc.q2(k) = del_qq / dtime;
                loc.s(k) = del_qq * latvap / dtime;
                loc.q1(k) = -del_qq / dtime;
            } else if (rh_curr > cldcr && rh_curr <= R8(1.0)) {
                R8 del_cf = st_frac - cldo(k);
                if (del_cf < R8(0.0)) {
                    del_qq = qc_nc * del_cf / cldo(k) * factcc;
                    loc.q2(k) = del_qq / dtime;
                    loc.s(k) = del_qq * latvap / dtime;
                    loc.q1(k) = -del_qq / dtime;
                    loc.q4(k) = loc.q2(k) * s1q4(k) / mg_max(R8(1.e-30), s1q2(k));
                    loc.q4(k) = mg_min(R8(0.0), mg_max(loc.q4(k), -s1q4(k) / dtime));
                } else if (del_cf > R8(0.0) && cldo(k) > R8(1.e-8)) {
                    del_qq = mg_min(s1q1(k) * R8(0.1),
                                    mg_min(R8(3.e-3) * st_frac, qc_nc * del_cf / cldo(k))) * factcc;
                    loc.q2(k) = del_qq / dtime;
                    loc.s(k) = del_qq * latvap / dtime;
                    loc.q1(k) = -del_qq / dtime;
                    loc.q4(k) = R8(0.0);
                } else if (del_cf > R8(0.0) && cldo(k) < R8(1.e-8)) {
                    del_qq = mg_min(s1q1(k) * R8(0.1), R8(2.e-5) * st_frac) * factcc;
                    loc.q2(k) = del_qq / dtime;
                    loc.s(k) = del_qq * latvap / dtime;
                    loc.q1(k) = -del_qq / dtime;
                    loc.q4(k) = R8(0.0);
                } else {
                    del_qq = R8(0.0);
                    loc.q2(k) = R8(0.0);
                    loc.s(k) = R8(0.0);
                    loc.q1(k) = R8(0.0);
                    loc.q4(k) = R8(0.0);
                }
            } else if (rh_curr < cldcr && cldo(k) < R8(1.e-2) && qc_nc > R8(1.e-12)) {
                R8 tsp, qsll;
                mg_findsp_water(s1q1(k), s1_t(k), pmid(k), tsp, qsll, err MG_SAT_POINT_ARG(k));
                del_qq = mg_max(mg_min(qc_nc, (qsll - s1q1(k))), R8(0.0)) * R8(0.99);
                loc.q2(k) = -del_qq / dtime;
                loc.s(k) = -del_qq * latvap / dtime;
                loc.q1(k) = del_qq / dtime;
                loc.q4(k) = loc.q2(k) * s1q4(k) / mg_max(R8(1.e-30), s1q2(k));
                loc.q4(k) = mg_min(R8(0.0), mg_max(loc.q4(k), -s1q4(k) / dtime));
            } else if (rh_curr < cldcr && cldo(k) >= R8(1.e-2) && qc_nc > R8(1.e-12)) {
                R8 tsp, qsll;
                mg_findsp_water(s1q1(k), s1_t(k), pmid(k), tsp, qsll, err MG_SAT_POINT_ARG(k));
                del_qq = mg_max(mg_min(qc_nc, qsll - s1q1(k)), R8(0.0)) * factcc;
                loc.q2(k) = -del_qq / dtime;
                loc.s(k) = -del_qq * latvap / dtime;
                loc.q1(k) = del_qq / dtime;
                loc.q4(k) = loc.q2(k) * s1q4(k) / mg_max(R8(1.e-30), s1q2(k));
                loc.q4(k) = mg_min(R8(0.0), mg_max(loc.q4(k), -s1q4(k) / dtime));
            } else {
                loc.q2(k) = R8(0.0);
                loc.s(k) = R8(0.0);
                loc.q1(k) = R8(0.0);
                loc.q4(k) = R8(0.0);
            }
            s1_t(k) = s1_t(k) + loc.s(k) * dtime / cpair;
        }
        mg_ptend_sum(loc, all, pver, false, false);
        mg_physics_update(s1_s, s1q1, s1q2, s1q3, s1q4, s1q5, loc, pver, dtime, false, false);
        mg_ptend_zero(loc, pver);
        // Macro2: limiters
        for (int k = 1; k <= pver; ++k) {
            R8 del_qq = R8(0.0);
            R8 cl, ci;
            mg_cal_cldfra(cl, ci, s1q1(k), s1q2(k), qi_mac(k), s1_t(k), pmid(k), xland_pt, snowh_pt MG_SAT_POINT_ARG(k));
            alst(k) = (R8(1.0) - cv_frac(k)) * cl;
            aist(k) = (R8(1.0) - cv_frac(k)) * ci;
            ast(k) = mg_max(alst(k), aist(k));
            R8 st_frac = alst(k);
            R8 qc_nc = mg_max(R8(0.0), s1q2(k)
                              - mg_max(R8(0.0), (dp_frac(k) * dp_icwmr(k) + sh_frac(k) * sh_icwmr(k))));
            if (st_frac > R8(1.e-5)) {
                if (qc_nc > R8(3.e-3) * st_frac) {
                    del_qq = qc_nc - R8(3.e-3) * st_frac * fact_adj;
                    loc.q2(k) = -del_qq / dtime;
                    loc.q1(k) = del_qq / dtime;
                    loc.s(k) = -del_qq * latvap / dtime;
                    loc.q4(k) = loc.q2(k) * s1q4(k) / mg_max(R8(1.e-20), s1q2(k));
                    loc.q4(k) = mg_min(R8(0.0), mg_max(loc.q4(k), -s1q4(k) / dtime));
                } else if (qc_nc < R8(2.e-5) * st_frac) {
                    del_qq = R8(2.e-5) * st_frac - qc_nc;
                    del_qq = mg_max(R8(0.0), mg_min(del_qq, s1q1(k) * R8(0.1))) * fact_adj;
                    loc.q2(k) = del_qq / dtime;
                    loc.q1(k) = -del_qq / dtime;
                    loc.s(k) = del_qq * latvap / dtime;
                }
            }
            s1_t(k) = s1_t(k) + loc.s(k) * dtime / cpair;
        }
        mg_ptend_sum(loc, all, pver, false, false);
        mg_physics_update(s1_s, s1q1, s1q2, s1q3, s1q4, s1q5, loc, pver, dtime, false, false);
        mg_ptend_zero(loc, pver);
        // label 2000: positive state and final fractions
        for (int k = 1; k <= pver; ++k) {
            s1q1(k) = mg_max(R8(UW_QMIN_Q), s1q1(k));
            s1q2(k) = mg_max(R8(0.0), s1q2(k));
            s1q3(k) = mg_max(R8(0.0), s1q3(k));
            s1q4(k) = mg_max(R8(0.0), s1q4(k));
            s1q5(k) = mg_max(R8(0.0), s1q5(k));
            R8 cl, ci;
            mg_cal_cldfra(cl, ci, s1q1(k), s1q2(k), qi_mac(k), s1_t(k), pmid(k), xland_pt, snowh_pt MG_SAT_POINT_ARG(k));
            aist(k) = (R8(1.0) - cv_frac(k)) * ci;
            alst(k) = (R8(1.0) - cv_frac(k)) * cl;
            ast(k) = mg_max(alst(k), aist(k));
        }
    }

    // ---- cloud fractions to WRF, micro cldo from CLDFRA_OLD_MP (:867-878) ----
    for (int kw = 1; kw <= kte; ++kw) {
        const int kf = kte - kw + 1;
        out.cldfral(kw) = n4(alst(kf));
        out.cldfrai(kw) = n4(aist(kf));
        out.cldfra_mp(kw) = n4(ast(kf));
        out.cldfra_mp_all(kw) = n4(mg_min(ast(kf) + mg_min(R8(0.8), dp_frac(kf) + sh_frac(kf)), R8(1.0)));
        out.cldfra_conv(kw) = n4(mg_max(mg_min(R8(0.8), dp_frac(kf) + sh_frac(kf)), R8(0.0)));
        cldo(kf) = w8(out.cldfra_old_mp(kw));
    }

    // ---- prescribed aerosol (:941-982) ----
    MgAerosol qaer;
    for (int l = 0; l <= 12; ++l) qaer.q[l] = R8(0.0);
    {
        R8 tmpnaer = w8(in.accum);
        qaer.q[7] = tmpnaer;
        qaer.q[6] = R8(MD_SPECDENS) * tmpnaer / R8(MD_VOLTONUMB1);
        tmpnaer = w8(in.aitken);
        qaer.q[9] = tmpnaer;
        qaer.q[8] = R8(MD_SPECDENS) * tmpnaer / R8(MD_VOLTONUMB2);
        tmpnaer = w8(in.coarse);
        qaer.q[12] = tmpnaer;
        R8 tmpmaer = R8(MD_SPECDENS) * tmpnaer / R8(MD_VOLTONUMB3);
        qaer.q[10] = tmpmaer * SP(0.5);
        qaer.q[11] = tmpmaer * SP(0.5);
    }
    // dgnumwet: 0.1e-6_r8 everywhere, then default-REAL 0.02e-6 (aitken) and
    // 1.0e-6 (coarse) (:584-597); dgnum = dgnumwet.
    const R8 dgnum_aitken = SP(0.02e-6);
    const R8 dgnumwet_coarse = SP(1.0e-6);

    // ---- microp_aero_ts and mmicro_pcond (:1035-1079) ----
    ma_microp_aero_ts(pver, dtime, s1_t, s1q1, s1q2, s1q3, s1q4, pmid, rpdel,
                      ast, ast, ast, cldo, pint, zm, qaer, kkvh, tke, naai, npccn,
                      rndst3, nacon3, dgnum_aitken, dgnumwet_coarse, ws MG_SAT_ARG);
    mg_mmicro_pcond(pver, dtime, s1_t, s1q1, s1q2, s1q3, s1q4, s1q5, pmid, pdel,
                    ast, ast, ast, naai, npccn, rndst3, nacon3, mo, ws, &gbad MG_SAT_ARG);
    if (gbad) *err = 2;

    for (int k = 1; k <= pver; ++k) {
        loc.s(k) = mo.tlat(k);
        loc.q1(k) = mo.qvlat(k);
        loc.q2(k) = mo.qctend(k);
        loc.q3(k) = mo.qitend(k);
        loc.q4(k) = mo.nctend(k);
        loc.q5(k) = mo.nitend(k);
    }
    const R8 prec_pcw = mo.prect, snow_pcw = mo.preci;
    const R8 prec_sed = R8(0.0), snow_sed = R8(0.0);
    mg_ptend_sum(loc, all, pver, true, true);
    mg_physics_update(s1_s, s1q1, s1q2, s1q3, s1q4, s1q5, loc, pver, dtime, true, true);

    // radiation-facing radii (:1203-1293): in-stratus mixing ratios on the
    // updated state1, 'inter' treatment (icecldf = liqcldf = ast).
    for (int kw = 1; kw <= kte; ++kw) {
        const int kf = kte - kw + 1;
        R8 icimrst = mg_min(s1q3(kf) / mg_max(R8(0.0001), ast(kf)), R8(0.005));
        R8 icwmrst = mg_min(s1q2(kf) / mg_max(R8(0.0001), ast(kf)), R8(0.005));
        R8 efcout = R8(10.), efiout = R8(25.);
        if (ast(kf) > R8(0.01) && icwmrst > R8(5.e-5)) efcout = mo.effc(kf);
        if (ast(kf) > R8(0.01) && icimrst > R8(1.e-6)) efiout = mo.effi(kf);
        out.lradius(kw) = n4(efcout);
        out.iradius(kw) = n4(efiout);
    }

    // final update of the step-start state with the summed tendencies
    mg_physics_update(st_s, sq1, sq2, sq3, sq4, sq5, all, pver, dtime, true, true);

    // ---- back to WRF (:1324-1392) ----
    for (int kw = 1; kw <= kte; ++kw) {
        const int kf = kte - kw + 1;
        const float qv_new = n4(sq1(kf) / (R8(1.0) - sq1(kf)));
        out.qv(kw) = qv_new;
        const R8 multFrc = R8(1.0) + w8(qv_new);
        out.qc(kw) = n4(sq2(kf) * multFrc);
        out.qi(kw) = n4(sq3(kf) * multFrc);
        out.qs(kw) = n4(mo.qsout(kf) * multFrc);
        out.qr(kw) = n4(mo.qrout(kf) * multFrc);
        const float nc_new = n4(sq4(kf) * multFrc);
        out.nc(kw) = nc_new;
        out.qndrop(kw) = nc_new;
        out.ni(kw) = n4(sq5(kf) * multFrc);
        out.ns(kw) = n4(mo.nsout(kf) * w8(in.alt(kw)) * multFrc);
        out.nr(kw) = n4(mo.nrout(kf) * w8(in.alt(kw)) * multFrc);
        out.th(kw) = n4((st_s(kf) - gravit * w8(in.z(kw))) / (cpair * w8(in.pi_phy(kw))));
        out.wsedl3d(kw) = n4(mo.vtrmc(kf));
        out.cldfra_old_mp(kw) = n4(ast(kf));
        const R8 ttt = (st_s(kf) - gravit * w8(in.z(kw))) / cpair;
        const R8 esl = MG_SAT(kf, ttt, 0);
        const R8 qvs = mg_max(R8(1.e-30), epsqs * esl / (pmid(kf) - (R8(1.0) - epsqs) * esl));
        out.rh_old_mp(kw) = n4(mg_max(R8(0.0), sq1(kf) / qvs));
        out.lcd_old_mp(kw) = n4(alst(kf));
    }
    const float rainncv = n4((prec_sed + prec_pcw) * dtime * R8(1.0e3));
    *out.rainncv = rainncv;
    *out.rainnc = __fadd_rn(*out.rainnc, rainncv);
    const float snowncv = n4((snow_sed + snow_pcw) * dtime * R8(1.0e3));
    *out.snowncv = snowncv;
    *out.snownc = __fadd_rn(*out.snownc, snowncv);
    *out.sr = n4(w8(snowncv) / (w8(rainncv) + R8(1.E-12)));
}

extern "C" __global__ void cammgmp_columns(
    const float* __restrict__ p8w, const float* __restrict__ p_hyd,
    const float* __restrict__ t_phy, const float* __restrict__ pi_phy,
    const float* __restrict__ z, const float* __restrict__ tke,
    const float* __restrict__ kvh, const float* __restrict__ alt,
    const float* __restrict__ icwmrsh, const float* __restrict__ icwmrdp,
    const float* __restrict__ shfrc, const float* __restrict__ cmfmc,
    const float* __restrict__ cmfmc2, const float* __restrict__ ht,
    const float* __restrict__ xland, const float* __restrict__ snowh,
    float* th, float* qv, float* qc, float* qi, float* qs, float* qr,
    float* nc, float* ni, float* ns, float* nr, float* qndrop,
    float* cldfra_old_mp, float* cldfra_mp, float* cldfra_mp_all,
    float* lradius, float* iradius, float* cldfrai, float* cldfral,
    float* cldfra_conv, float* wsedl3d, float* rh_old_mp, float* lcd_old_mp,
    float* rainnc, float* rainncv, float* snownc, float* snowncv, float* sr,
    long long ncols, long long col0, int nchunk, int pool_stride, int nz,
    float dt, int itimestep, int cu_coupled, int shcu_coupled,
    float accum, float aitken, float coarse,
    double* r8pool, int r8cap, int* err)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nchunk) return;
    const long long c = col0 + tid;
    const long long s = ncols;
    int i4dummy = 0;
    Ws ws{r8pool + tid, &i4dummy, pool_stride, 0, r8cap, 0, 0, err};
    #define MG_IN(a) F32V{const_cast<float*>(a) + c, s}
    #define MG_IO(a) F32V{(a) + c, s}
    MgColumnIn in{MG_IN(p8w), MG_IN(p_hyd), MG_IN(t_phy), MG_IN(pi_phy),
                  MG_IN(z), MG_IN(tke), MG_IN(kvh), MG_IN(alt),
                  MG_IN(icwmrsh), MG_IN(icwmrdp), MG_IN(shfrc), MG_IN(cmfmc),
                  MG_IN(cmfmc2), ht[c], xland[c], snowh[c], nz, itimestep,
                  cu_coupled, shcu_coupled, dt, accum, aitken, coarse};
    MgColumnOut out{MG_IO(th), MG_IO(qv), MG_IO(qc), MG_IO(qi), MG_IO(qs),
                    MG_IO(qr), MG_IO(nc), MG_IO(ni), MG_IO(ns), MG_IO(nr),
                    MG_IO(qndrop), MG_IO(cldfra_old_mp), MG_IO(cldfra_mp),
                    MG_IO(cldfra_mp_all), MG_IO(lradius), MG_IO(iradius),
                    MG_IO(cldfrai), MG_IO(cldfral), MG_IO(cldfra_conv),
                    MG_IO(wsedl3d), MG_IO(rh_old_mp), MG_IO(lcd_old_mp),
                    rainnc + c, rainncv + c, snownc + c, snowncv + c, sr + c};
    #undef MG_IN
    #undef MG_IO
    int status = 0;
    cammgmp_column(in, out, ws, &status);
    if (status) atomicMax(err, status);
}

// Unit probe of the support functions against the oracle's helper words
// (tests/test_cammgmp_wrf461_parity.py): which = 0 estblf, 1 polysvp(t,0),
// 2 polysvp(t,1), 3 the CAM erf, 4 MG's GAMMA.
extern "C" __global__ void cammgmp_helpers(const double* __restrict__ x,
                                           double* __restrict__ y, int n,
                                           int which, int* err)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const R8 a(x[i]);
    R8 r(0.0);
    int bad = 0;
    if (which == 0) r = uw_estblf(a);
    else if (which == 1) r = mg_polysvp(a, 0);
    else if (which == 2) r = mg_polysvp(a, 1);
    else if (which == 3) r = cam_erf(a);
    else r = mg_gamma(a, &bad);
    if (bad) atomicMax(err, 2);
    y[i] = r.v;
}

// Arithmetic receipt from the actual CAM translation unit and loader.
extern "C" __global__ void cammgmp_ieee_probe(const float* x, float* y) {
    y[0] = __fmul_rn(x[0], x[1]);
    y[1] = __fdiv_rn(x[2], 10.0f);
}
